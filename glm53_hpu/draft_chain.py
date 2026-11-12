# SPDX-License-Identifier: Apache-2.0
# The early_draft function in this file extends decode_lean.early_draft of the gaudi-cookbook's GLM-5 port
# (thediymaker/gaudi-cookbook, Apache-2.0), so this file stays under that licence (see NOTICE).
"""More than one MTP draft token per step without leaving the device (GLM_DRAFT_CHAIN=1, with GLM_EARLY_DRAFT=1 and
GLM_SPEC_DEVICE_REJECT=1 and --speculative-config num_speculative_tokens >= 2).

Why: at long context every decode step pays for the indexer scans (about 7 ms at 700K tokens) whatever the number
of rows in the step, so tokens per step is the lever. The gaudi-cookbook's early-draft lever is one-draft only; with
two drafts vllm-gaudi falls back to its host-side loop (copy the sampled tokens to the host, build the next
position's attention metadata there, launch the draft model again): measured about 12 ms per step for the second
draft, which cancels its gain.

Here the whole proposal is one HPU graph, enqueued right after the rejection sampler:
  pass 1   the draft model over the step's verify rows, exactly as the one-draft lever does: draft 1 is the argmax at
           each request's last accepted row;
  pass j   the draft model again over the same rows and the same attention metadata, with draft j-1 as the input
           token, the previous pass's hidden state at the last accepted row as the hidden input, positions shifted
           by j-1, and every cache write sent to the pad block. Only the last accepted row of each request is read.
A later pass therefore sees the keys of the last accepted row (not a key of its own position) and, above the sparse
threshold, the first pass's index selection (index_share_for_mtp_iteration in the model config). Both only affect
how good the drafts are: every draft is verified by the target model, the output is unchanged.

The hook replaces glm5_hpu.decode_lean.early_draft when the switch is on; one-draft steps keep the stock code.
"""
from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("GLM_DRAFT_CHAIN", "0") == "1"
TAG = "[glm53-draft-chain]"
_state = {"chain": None, "logged": False}


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


def _make_chain(mod, runner, k: int):
    """The proposal as one function of tensors (wrapped in an HPU graph by the caller)."""
    from glm5_hpu import dsa

    adapter = runner.drafter.model           # HpuModelAdapter whose instance forward is the graph wrapper
    raw_forward = type(adapter).forward      # the adapter's own forward: no nested graph capture
    pad_block = int(runner._PAD_BLOCK_ID)
    max_len = int(runner.max_model_len)

    def unpack(ret, fallback):
        # DeepSeekMTP returns (pre-norm hidden for the logits, hidden for the next pass)
        if isinstance(ret, tuple):
            return ret[0], ret[1]
        return ret, fallback

    def chain(input_ids, positions, hidden_states, last, attn_metadata):
        rows, b = input_ids.shape[0], last.shape[0]
        per = rows // b
        ret = raw_forward(adapter, input_ids=input_ids, positions=positions, hidden_states=hidden_states,
                          inputs_embeds=None, attn_metadata=attn_metadata)
        lh, hs = unpack(ret, hidden_states)
        h = lh.shape[-1]
        d = adapter.compute_logits(lh.reshape(-1, h).index_select(0, last)).argmax(dim=-1)
        drafts = [d]
        if k > 1:
            req = torch.arange(b, device=last.device, dtype=torch.long).repeat_interleave(per)
            sm = attn_metadata.slot_mapping
            bs = int(attn_metadata.block_size)
            pad = pad_block * bs + torch.arange(rows, device=sm.device, dtype=torch.long) % bs
            md2 = attn_metadata._replace(slot_mapping=pad.to(sm.dtype).view(sm.shape))
            h_in = hs.reshape(-1, h).index_select(0, last)
            try:
                for it in range(1, k):
                    dsa._STATE["draft_iter"] = it
                    tok = d.index_select(0, req).view(input_ids.shape).to(input_ids.dtype)
                    hid = h_in.index_select(0, req).view(hidden_states.shape)
                    pos = positions + it
                    pos = torch.where(pos >= max_len, torch.zeros_like(pos), pos)
                    ret = raw_forward(adapter, input_ids=tok, positions=pos, hidden_states=hid, inputs_embeds=None,
                                      attn_metadata=md2)
                    lh, hs = unpack(ret, hid)
                    d = adapter.compute_logits(lh.reshape(-1, h).index_select(0, last)).argmax(dim=-1)
                    h_in = hs.reshape(-1, h).index_select(0, last)
                    drafts.append(d)
            finally:
                dsa._STATE["draft_iter"] = 0
        return torch.stack(drafts, dim=1)

    return mod.htorch.hpu.wrap_in_hpu_graph_func(chain, disable_tensor_cache=True)


def install(mod) -> None:
    """Called from fastlink.install_runner with the hpu_model_runner module."""
    if not ENABLED:
        return
    from glm5_hpu import decode_lean as dl

    if not (dl.EARLY_DRAFT and dl.DEVICE_REJECT):
        _log("needs GLM_EARLY_DRAFT=1 and GLM_SPEC_DEVICE_REJECT=1: not installed")
        return
    stock = dl.early_draft

    def early_draft(runner, out, md, hidden_states, num_decodes, num_prefills, decode_data, warmup_mode):
        sc = runner.speculative_config
        k = int(sc.num_speculative_tokens) if sc is not None else 1
        if k < 2:
            return stock(runner, out, md, hidden_states, num_decodes, num_prefills, decode_data, warmup_mode)
        runner._glm_early_draft = None
        if (
            warmup_mode
            or num_prefills > 0
            or decode_data is None
            or md is None
            or num_decodes <= 0
            or not sc.use_eagle()
            or runner.use_aux_hidden_state_outputs
            or decode_data.position_ids is None
        ):
            return
        nd = list(md.num_draft_tokens)
        b, rows = len(nd), k + 1
        if out.dim() != 2 or out.shape[0] < b or out.shape[1] != rows or num_decodes > b:
            return
        dev = out.device
        o = out[:b]
        length = ((o != -1) & (o < runner.input_batch.vocab_size)).sum(dim=1).to(torch.int64)
        has = dl._h2d(torch.tensor([1 if n > 0 else 0 for n in nd], dtype=torch.int64), dev)
        eff = torch.where(has > 0, length, torch.ones_like(length))
        base = torch.arange(b, dtype=torch.int64, device=dev) * rows
        last = base + eff - 1
        j = torch.arange(rows, dtype=torch.int64, device=dev)
        picked = torch.where(j.unsqueeze(0) < eff.unsqueeze(1), base.unsqueeze(1) + j.unsqueeze(0),
                             torch.full((b, rows), -1, dtype=torch.int64, device=dev)).reshape(-1)
        ths = hidden_states[picked]
        if ths.dim() == 2:
            ths = ths.unsqueeze(1)
        if _state["chain"] is None:
            _state["chain"] = _make_chain(mod, runner, k)
        draft = _state["chain"](input_ids=out.reshape(-1, 1)[picked], positions=decode_data.position_ids[picked],
                                hidden_states=ths, last=last,
                                attn_metadata=mod.trim_attn_metadata(decode_data.attn_metadata))
        if not _state["logged"]:
            _state["logged"] = True
            _log(f"{k} drafts per step in one graph ({b} request(s), {rows} rows each)")
        runner._glm_early_draft = draft[:num_decodes]

    dl.early_draft = early_draft
    _log("installed: chained drafts replace the one-draft early-draft lever for num_speculative_tokens >= 2")
