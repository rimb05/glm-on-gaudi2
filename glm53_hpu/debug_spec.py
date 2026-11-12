# SPDX-License-Identifier: MIT
"""Diagnostics for speculative decoding on HPU (GLM_DEBUG_SPEC=1): shapes that reach the draft model.
Installed from fastlink.install_runner; prints the first few calls on tensor-parallel rank 0 only."""
import os

import torch

ON = os.environ.get("GLM_DEBUG_SPEC", "0") == "1"
_n = {"propose": 0, "cache": 0}


def _shape(x):
    if torch.is_tensor(x):
        return f"{tuple(x.shape)}:{str(x.dtype).replace('torch.', '')}"
    return type(x).__name__


def install(mod):
    if not ON:
        return
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm_gaudi.extension import utils as ext_utils
    from vllm_gaudi.v1.spec_decode import hpu_eagle

    orig_propose = hpu_eagle.HpuEagleProposer.propose

    def propose(self, target_token_ids, target_positions, target_hidden_states, last_token_indices,
                common_attn_metadata, block_table_cpu_tensor, model_runner):
        if get_tensor_model_parallel_rank() == 0 and _n["propose"] < 4:
            _n["propose"] += 1
            md = common_attn_metadata
            fields = {k: _shape(getattr(md, k, None)) for k in ("slot_mapping", "block_list", "block_groups",
                                                               "block_mapping", "attn_bias", "seq_lens_tensor")}
            print(f"[glm53-debug] propose #{_n['propose']}: ids {_shape(target_token_ids)} pos "
                  f"{_shape(target_positions)} hidden {_shape(target_hidden_states)} last_idx "
                  f"{_shape(last_token_indices)} = {last_token_indices.tolist() if torch.is_tensor(last_token_indices) else last_token_indices} "
                  f"is_prompt {getattr(md, 'is_prompt', None)} md {fields} block_table {_shape(block_table_cpu_tensor)}",
                  flush=True)
            _n["cache"] = 0
        return orig_propose(self, target_token_ids, target_positions, target_hidden_states, last_token_indices,
                            common_attn_metadata, block_table_cpu_tensor, model_runner)

    hpu_eagle.HpuEagleProposer.propose = propose

    orig_cache_fwd = ext_utils.VLLMKVCache.forward

    def cache_forward(self, input, cache, slot_mapping, *args, **kwargs):
        if _n["propose"] and _n["cache"] < 2 and get_tensor_model_parallel_rank() == 0:
            _n["cache"] += 1
            print(f"[glm53-debug]   cache write after propose #{_n['propose']}: input {_shape(input)} cache "
                  f"{_shape(cache)} slots {_shape(slot_mapping)}", flush=True)
        return orig_cache_fwd(self, input, cache, slot_mapping, *args, **kwargs)

    ext_utils.VLLMKVCache.forward = cache_forward

    orig_decode = mod.HPUModelRunner.propose_eagle_decode

    def propose_eagle_decode(self, sampled_token_ids, decode_sampled_token_ids_tensor, hidden_states,
                             aux_hidden_states, num_decodes, decode_data):
        if get_tensor_model_parallel_rank() == 0 and _n["propose"] < 4:
            sm = getattr(decode_data, "spec_decode_metadata", None)
            print(f"[glm53-debug] propose_eagle_decode: sampled {sampled_token_ids} sampled_t "
                  f"{_shape(decode_sampled_token_ids_tensor)} hidden {_shape(hidden_states)} num_decodes "
                  f"{num_decodes} positions {_shape(decode_data.position_ids)} token_ids "
                  f"{_shape(getattr(decode_data, 'token_ids', None))} num_draft "
                  f"{getattr(sm, 'num_draft_tokens', None)}", flush=True)
        return orig_decode(self, sampled_token_ids, decode_sampled_token_ids_tensor, hidden_states,
                           aux_hidden_states, num_decodes, decode_data)

    mod.HPUModelRunner.propose_eagle_decode = propose_eagle_decode
    print("[glm53-debug] speculative-decoding shape tracing on", flush=True)
