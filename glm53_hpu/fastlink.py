# SPDX-License-Identifier: Apache-2.0
# The last-stage forward in this file follows HPUModelRunner._execute_model_generic of vllm-gaudi as patched by
# the gaudi-cookbook's pipeline-parallel fix (both Apache-2.0), so this file stays under that licence (see NOTICE).
"""Fast stage-to-stage link for pipeline parallelism on HPU: decode-sized hidden states go card to card.

The gaudi-cookbook pipeline fix moves a stage's output to the next stage as CPU tensors over gloo: the sender copies
to the host (a device sync), the receiver copies back and synchronises before it launches its own graph. That costs
milliseconds per hop per token (measured here between two servers: 2.4 ms with the hosts in the performance CPU
governor, 9 ms in schedutil), which is fine for prefill chunks but not for decode steps, and rules out a pipeline
with one card per stage.

With GLM_PP_FASTLINK=1, forwards of at most GLM_PP_FASTLINK_MAX_TOKENS tokens (decode steps) use an HCCL collective
on a two-rank group instead:

    stage i   ... last layer -> all_reduce([hidden | residual], group {i, i+1})         (peer adds zeros)
    stage i+1 all_reduce(zeros, group {i, i+1}) -> [hidden | residual] -> first layer ...

Both calls are lazy device operations inside each stage's captured HPU graph, so no stage copies to the host and
every stage's host can launch its forward as soon as the step arrives; the device work then cascades down the chain.
Between cards of one server the collective uses the internal links, between the servers the cards' scale-out
ports and the switch. Larger forwards (prefill chunks) and warm-up forwards keep the cookbook's gloo link.

Hooks (patches/fastlink_hooks.py appends them):
  hpu_worker.py        init_worker_distributed_environment = fastlink.wrap_init(...)   pair groups after start-up
  hpu_model_runner.py  fastlink.install_runner(module)                                 stage forwards + adapter
"""
from __future__ import annotations

import contextvars
import os
import time

import torch

ENABLED = os.environ.get("GLM_PP_FASTLINK", "0") == "1"
MAX_TOKENS = int(os.environ.get("GLM_PP_FASTLINK_MAX_TOKENS", "64"))
# GLM_DECODE_NOCUT=1: forwards of at most MAX_TOKENS tokens are not cut into one compiled graph per decoder layer.
# vllm-gaudi calls mark_step() after every decoder layer; on a single-card stage with no collectives inside, each
# cut is one more recipe the host has to launch (measured: decode is bound by these launches, about 0.2 ms per
# layer). Prefill keeps the cuts: without them a multi-layer prompt graph takes minutes to compile.
NOCUT = os.environ.get("GLM_DECODE_NOCUT", "0") == "1"
TAG = "[glm53-fastlink]"

TRACE = os.environ.get("GLM_PP_FASTLINK_TRACE", "0") == "1"
_trace: list[tuple[float, float, float]] = []


def _trace_add(t0: float, t1: float) -> None:
    """Diagnostic only (it adds a device sync to every stage). Host timestamps of a fast forward: entry, launched,
    and device done (after torch.hpu.synchronize()). Printed once for steps 300..303, in ms relative to the entry
    of step 300 on this rank; every stage sees the same step sequence and enters within 0.2 ms of the others."""
    torch.hpu.synchronize()
    _trace.append((t0, t1, time.time()))
    if len(_trace) % 200 == 0:  # every 200 fast forwards: the four that ended 100 forwards ago
        z = _trace[-100][0]
        rows = " | ".join(f"in {1e3 * (a - z):7.2f} launched +{1e3 * (b - a):.2f} done +{1e3 * (c - a):.2f}"
                          for a, b, c in _trace[-100:-96])
        _log(f"trace rank {_state['rank']} step {len(_trace) - 100}: {rows}")
        del _trace[:-200]


_fast = contextvars.ContextVar("glm53_fastlink", default=False)
_small = contextvars.ContextVar("glm53_small_forward", default=False)
_groups: dict[tuple[int, int], object] = {}  # (lower global rank, upper global rank) -> process group
_state = {"rank": None, "tp": 1, "pp": 1, "logged": False}


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ groups
def wrap_init(orig):
    """Wrap init_worker_distributed_environment: after the stock groups exist, create one HCCL group per pair of
    neighbouring stages (per tensor-parallel rank) and check each with a tiny all-reduce."""

    def init_worker_distributed_environment(vllm_config, rank, distributed_init_method=None, local_rank=-1):
        orig(vllm_config, rank, distributed_init_method, local_rank)
        pc = vllm_config.parallel_config
        pp, tp = pc.pipeline_parallel_size, pc.tensor_parallel_size
        _state.update(rank=rank, tp=tp, pp=pp)
        if not ENABLED or pp <= 1:
            return
        # every process creates every group, in the same order (torch.distributed requires it)
        for s in range(pp - 1):
            for t in range(tp):
                lo, hi = s * tp + t, (s + 1) * tp + t
                g = torch.distributed.new_group(ranks=[lo, hi], backend="hccl")
                if rank in (lo, hi):
                    _groups[(lo, hi)] = g
        # check: the lower rank of each pair sends ones, the upper zeros; both must read ones
        for (lo, hi), g in sorted(_groups.items()):
            x = (torch.ones(8) if rank == lo else torch.zeros(8)).to(torch.bfloat16).to("hpu")
            torch.distributed.all_reduce(x, group=g)
            got = float(x.float().sum().item())
            if got != 8.0:
                raise RuntimeError(f"{TAG} rank {rank}: pair {lo}-{hi} returned {got}, expected 8")
        _log(f"rank {rank}: PP {pp} x TP {tp}, pair links {sorted(_groups)} ok, fast path for <= {MAX_TOKENS} tokens")

    return init_worker_distributed_environment


def _pair(lower: bool):
    """This rank's group with the previous (lower=True) or the next stage."""
    r, tp = _state["rank"], _state["tp"]
    key = (r - tp, r) if lower else (r, r + tp)
    return _groups[key]


def is_fast(token_ids, warmup_mode) -> bool:
    return bool(ENABLED and _groups and not warmup_mode and token_ids.numel() <= MAX_TOKENS)


# ------------------------------------------------------------------------------------------------ model
def _wrap_model_forward():
    import vllm_gaudi.models.deepseek_v2  # noqa: F401  (installs the HPU forward we wrap)
    from vllm.distributed import get_pp_group
    from vllm.model_executor.models import deepseek_v2
    from vllm.sequence import IntermediateTensors

    orig = deepseek_v2.DeepseekV2Model.forward
    if getattr(orig, "_glm53_fastlink", False):
        return

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
        if not _fast.get():
            return orig(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        pp = get_pp_group()
        h = self.config.hidden_size
        if not pp.is_first_rank:
            n = positions.numel()
            buf = torch.zeros(n, 2 * h, dtype=self.embed_dtype_for_link, device=positions.device)
            torch.distributed.all_reduce(buf, group=_pair(lower=True))
            shape = tuple(positions.shape) + (h,)
            intermediate_tensors = IntermediateTensors({"hidden_states": buf[:, :h].reshape(shape),
                                                        "residual": buf[:, h:].reshape(shape)})
        out = orig(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        if not pp.is_last_rank:
            hs, rs = out["hidden_states"], out["residual"]
            if not _state["logged"]:
                _state["logged"] = True
                _log(f"rank {_state['rank']}: sending hidden {tuple(hs.shape)} residual {tuple(rs.shape)} {hs.dtype}")
            buf = torch.cat([hs.reshape(-1, h), rs.reshape(-1, h)], dim=-1)
            torch.distributed.all_reduce(buf, group=_pair(lower=False))
            out = IntermediateTensors({"hidden_states": buf[:, :h], "residual": buf[:, h:]})
        return out

    forward._glm53_fastlink = True
    deepseek_v2.DeepseekV2Model.forward = forward


# ------------------------------------------------------------------------------------------------ runner
def install_runner(mod) -> None:
    """Called at the end of vllm_gaudi/v1/worker/hpu_model_runner.py (after the cookbook's pipeline-parallel code)."""
    runner, adapter = mod.HPUModelRunner, mod.HpuModelAdapter

    # Speculative decoding under pipeline parallelism: the draft model (the MTP layer) needs the final hidden
    # states, so it only ever runs on the last stage. vllm-gaudi would still build and load it on every stage
    # (about 1.5 GB per card of zero experts, embedding and attention on the stages that never call it).
    orig_runner_init = runner.__init__

    def runner_init(self, *args, **kwargs):
        orig_runner_init(self, *args, **kwargs)
        r, tp, pp = _state["rank"], _state["tp"], _state["pp"]
        if pp > 1 and r is not None and r < (pp - 1) * tp and hasattr(self, "drafter"):
            del self.drafter
            _log(f"rank {r}: not the last pipeline stage, the draft model is not loaded here")

    runner.__init__ = runner_init

    from glm53_hpu import debug_spec
    debug_spec.install(mod)
    from glm53_hpu import draft_chain
    draft_chain.install(mod)
    from glm53_hpu import fastmeta
    fastmeta.install(mod)
    from glm53_hpu import kvfp8
    kvfp8.install(mod)

    if not (ENABLED or NOCUT):
        return
    htorch, get_pp_group, trim = mod.htorch, mod.get_pp_group, mod.trim_attn_metadata

    if NOCUT:
        # same walk as vllm-gaudi's modify_model_layers, but the hook leaves small (decode) forwards in one graph
        def modify_model_layers(module, suffix_list, n=1, counter=None):
            def forward_hook(module, args, output):
                if not _small.get():
                    htorch.core.mark_step()
                return output

            if counter is None:
                counter = [0]
            for _name, child in module.named_children():
                if any(child.__class__.__name__.endswith(sfx) for sfx in suffix_list):
                    counter[0] += 1
                    if counter[0] % n == 0:
                        child.register_forward_hook(forward_hook)
                else:
                    modify_model_layers(child, suffix_list, n, counter)

        mod.modify_model_layers = modify_model_layers
        _log(f"decode forwards (<= {MAX_TOKENS} tokens) are not cut per layer")

    if ENABLED:
        if not hasattr(runner, "_gfpp_stage_forward"):
            raise RuntimeError(f"{TAG} the gaudi-cookbook pipeline-parallel fix is not applied")
        _wrap_model_forward()

    # The adapter is what the HPU graph wraps: the flag is one of its keyword arguments, so fast and gloo forwards
    # of the same shape are captured as different graphs. It is turned into a context variable for the model.
    orig_adapter_forward = adapter.forward

    def adapter_forward(self, *args, **kwargs):
        fast = bool(kwargs.pop("glm53_fast", False))
        if fast and not hasattr(self.model.model, "embed_dtype_for_link"):
            self.model.model.embed_dtype_for_link = self.dtype
        ids = kwargs.get("input_ids")
        small = NOCUT and ids is not None and ids.numel() <= MAX_TOKENS
        tok, tok2 = _fast.set(fast), _small.set(small)
        try:
            return orig_adapter_forward(self, *args, **kwargs)
        finally:
            _fast.reset(tok)
            _small.reset(tok2)

    adapter.forward = adapter_forward
    if not ENABLED:
        return

    orig_stage_forward = runner._gfpp_stage_forward
    orig_execute = runner._execute_model_generic

    def _common(self, token_ids, attn_metadata, warmup_mode):
        batch_size = token_ids.size(0)
        seq_len = self._seq_len(attn_metadata)
        num_blocks = self._num_blocks(attn_metadata)
        self._check_config(batch_size, seq_len, num_blocks, attn_metadata, warmup_mode)
        kw = {"glm53_fast": True}
        use_graphs = False
        if htorch.utils.internal.is_lazy():
            use_graphs = self._use_graphs(attn_metadata, batch_size)
            kw["bypass_hpu_graphs"] = not use_graphs
        if self.model_has_chunked_attention:
            kw["model_has_chunked_attention"] = True
        return batch_size, seq_len, num_blocks, use_graphs, kw

    def stage_forward(self, token_ids, position_ids, attn_metadata, lora_mask, warmup_mode, inputs_embeds=None,
                      model_mm_kwargs=None):
        """Non-last stage. Fast forwards end with the pair collective inside the graph: nothing to send here."""
        if not is_fast(token_ids, warmup_mode):
            return orig_stage_forward(self, token_ids, position_ids, attn_metadata, lora_mask, warmup_mode,
                                      inputs_embeds=inputs_embeds, model_mm_kwargs=model_mm_kwargs)
        t0 = time.time() if TRACE else 0.0
        _, _, _, _, kw = _common(self, token_ids, attn_metadata, warmup_mode)
        self.model.forward(input_ids=token_ids, positions=position_ids, attn_metadata=trim(attn_metadata),
                           kv_caches=self.kv_caches, inputs_embeds=inputs_embeds, model_mm_kwargs=model_mm_kwargs,
                           lora_mask=lora_mask, **kw)
        if TRACE:
            _trace_add(t0, time.time())

    def execute_model_generic(self, token_ids, position_ids, attn_metadata, logits_indices, kv_caches,
                              lora_logits_mask, lora_mask, warmup_mode=False, inputs_embeds=None,
                              model_mm_kwargs=None):
        """Last stage. Same as the stock method, with the hidden states received inside the graph."""
        pp = get_pp_group()
        if pp.world_size == 1 or not pp.is_last_rank or not is_fast(token_ids, warmup_mode):
            return orig_execute(self, token_ids, position_ids, attn_metadata, logits_indices, kv_caches,
                                lora_logits_mask, lora_mask, warmup_mode=warmup_mode, inputs_embeds=inputs_embeds,
                                model_mm_kwargs=model_mm_kwargs)
        t0 = time.time() if TRACE else 0.0
        batch_size, seq_len, num_blocks, use_graphs, kw = _common(self, token_ids, attn_metadata, warmup_mode)
        if self.is_driver_worker:
            event = f"model_forward_bs{batch_size}_seq{seq_len}_ctx{num_blocks}_graphs{'T' if use_graphs else 'F'}"
        else:
            event = "model_executable"
        with self.profiler.record_event("internal", event):
            hidden_states = self.model.forward(input_ids=token_ids, positions=position_ids,
                                               attn_metadata=trim(attn_metadata), kv_caches=kv_caches,
                                               inputs_embeds=inputs_embeds, model_mm_kwargs=model_mm_kwargs,
                                               lora_mask=lora_mask, **kw)
        if TRACE:
            _trace_add(t0, time.time())
        if self.use_aux_hidden_state_outputs:
            non_flattened_hidden_states, aux_hidden_states = hidden_states
            hidden_states = non_flattened_hidden_states
        else:
            non_flattened_hidden_states = hidden_states
            aux_hidden_states = None
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = hidden_states[logits_indices]
        mod.LoraMask.setLoraMask(lora_logits_mask)
        with self.profiler.record_event("internal", f"compute_logits{batch_size}_seq{seq_len}_ctx{num_blocks}"):
            logits = self.model.compute_logits(hidden_states)
        return non_flattened_hidden_states, aux_hidden_states, hidden_states, logits

    runner._gfpp_stage_forward = stage_forward
    runner._execute_model_generic = execute_model_generic
    _log("runner hooks installed")
