# SPDX-License-Identifier: Apache-2.0
"""GLM_KV_FP8=int8 (or fp8): make vllm-gaudi account for and allocate the MLA latent caches as 8-bit tensors.

The latent cache is the device memory that limits the context: 576 values per token and layer, replicated on every
tensor-parallel rank. In bf16 a 720K-token pool is what fits beside the FP8 weights on the first pipeline stage
(48.1 GiB of weights + 34.6 GiB of cache on a 94.6 GiB card). With 8-bit codes the same cards hold more than a
million tokens.

This module only changes the cache spec (an 8-bit dtype for the latent caches, not for the indexer-key caches):
vLLM's block accounting and vllm-gaudi's allocation follow the spec. The values are written and read by
glm5_hpu.dsa.LatentInt8Cache (GLM_KV_FP8=int8, the deployed codec: linear codes, one scale per token for every 64
latent dimensions) or LatentFP8Cache (GLM_KV_FP8=1 or fp8: E4M3 codes, one scale per token; four times the error,
kept for comparison), which glm5_hpu.dsa.alloc_buffers installs on every MLA layer whose cache tensor has that dtype
(target and draft model).
"""
from __future__ import annotations

import dataclasses
import os

import torch

MODE = os.environ.get("GLM_KV_FP8", "0")  # "int8": linear codes, a scale per token and 64 dims; "1" / "fp8": E4M3
ENABLED = MODE in ("1", "fp8", "int8")
DTYPE = torch.int8 if MODE == "int8" else torch.float8_e4m3fn
TAG = "[glm53-kvfp8]"


def install(mod) -> None:
    """Called from fastlink.install_runner with the hpu_model_runner module."""
    if not ENABLED:
        return
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole

    runner = mod.HPUModelRunner
    stock = runner.get_kv_cache_spec
    logged = {"done": False}

    def get_kv_cache_spec(self):
        specs = stock(self)
        out, n = {}, 0
        for name, spec in specs.items():
            if (isinstance(spec, MLAAttentionSpec) and spec.cache_role != SparseCacheRole.INDEXER
                    and spec.dtype == torch.bfloat16):
                spec = dataclasses.replace(spec, dtype=DTYPE)
                n += 1
            out[name] = spec
        if n and not logged["done"]:
            logged["done"] = True
            print(f"{TAG} latent caches of {n} layers are 8-bit ({DTYPE})", flush=True)
        return out

    runner.get_kv_cache_spec = get_kv_cache_spec
