# SPDX-License-Identifier: MIT
"""Decode-step metadata without per-block Python work (GLM_FAST_META=1).

vllm-gaudi rebuilds the paged-attention block lists of every decode step with Python lists: one list per row of
the sequence's block ids, list comprehensions for the groups and usages, itertools flatten and pad, torch.tensor
on the result, and before that _resolve_block called once per block. Nothing in it depends on the context length
except the amount of work: with a 700K-token context and three rows per step (two MTP drafts) it is about 17,000
block entries per step, measured 4.0 ms for the list building alone plus about as much for the per-block calls, on
every worker and before the step's forward can be launched. At 2K tokens the same code takes microseconds.

Two replacements, both returning exactly what the stock code returns:
  _resolve_all_blocks            identity when there is no defragmenter (the stock per-block call returns its
                                 argument in that case);
  get_habana_paged_attn_buffers  the non-contiguous case built with numpy slices; the contiguous-PA and
                                 sliding-window-head cases go to the stock function.
"""
from __future__ import annotations

import os

import numpy as np
import torch

ENABLED = os.environ.get("GLM_FAST_META", "0") == "1"
TAG = "[glm53-fastmeta]"


def paged_buffers(block_tables, slot_mapping, bucket_size, block_size, pad_block_id, dtype):
    """block_tables: one sequence of block ids per row (list, numpy array or tensor); slot_mapping: one [slot]
    per row. Returns (block_list long [bucket], block_groups long [bucket], block_usage dtype [bucket]) laid
    out as vllm-gaudi's get_habana_paged_attn_buffers does: rows concatenated, padded with (pad_block_id, -1, 1);
    every block of a row is full except its last, which holds slot % block_size + 1 tokens."""
    bl = np.full(bucket_size, pad_block_id, dtype=np.int64)
    bg = np.full(bucket_size, -1, dtype=np.int64)
    bu = np.ones(bucket_size, dtype=np.float32)
    off = 0
    for i, bt in enumerate(block_tables):
        n = len(bt)
        if n == 0:
            continue
        bl[off:off + n] = bt
        bg[off:off + n] = i
        bu[off:off + n - 1] = block_size
        bu[off + n - 1] = slot_mapping[i][0] % block_size + 1
        off += n
    return torch.from_numpy(bl), torch.from_numpy(bg), torch.from_numpy(bu).to(dtype)


def install(mod) -> None:
    """Called from fastlink.install_runner with the hpu_model_runner module."""
    if not ENABLED:
        return
    runner = mod.HPUModelRunner
    stock_buffers = runner.get_habana_paged_attn_buffers
    stock_resolve = runner._resolve_all_blocks

    def _resolve_all_blocks(self, block_table_list):
        if not getattr(self, "defragmenter", None):
            return block_table_list
        return stock_resolve(self, block_table_list)

    def get_habana_paged_attn_buffers(self, block_tables, slot_mapping, batch_size, block_size=None,
                                      force_non_contiguous=False, head_mask_full_len=None):
        if head_mask_full_len or (self.use_contiguous_pa and not force_non_contiguous):
            return stock_buffers(self, block_tables, slot_mapping, batch_size, block_size=block_size,
                                 force_non_contiguous=force_non_contiguous, head_mask_full_len=head_mask_full_len)
        bs = self.attn_block_size if block_size is None else block_size
        total = sum(len(bt) for bt in block_tables)
        size = self.bucketing_manager.find_decode_bucket(batch_size, total)[2]
        size += self.get_dp_padding(size)
        if size < total:  # the stock pad_list would return the longer list unpadded
            size = total
        return paged_buffers(block_tables, slot_mapping, size, bs, self._PAD_BLOCK_ID, self.model_config.dtype)

    runner._resolve_all_blocks = _resolve_all_blocks
    runner.get_habana_paged_attn_buffers = get_habana_paged_attn_buffers
    if os.environ.get("GLM_FAST_META_TRACE", "0") == "1":
        # host time of the decode-step input preparation, mean over every 300 steps
        import time

        stock_prep = runner._create_decode_input_data
        acc = {"n": 0, "t": 0.0, "blocks": 0}

        def _create_decode_input_data(self, *args, **kwargs):
            t0 = time.perf_counter()
            out = stock_prep(self, *args, **kwargs)
            acc["t"] += time.perf_counter() - t0
            acc["n"] += 1
            md = getattr(out, "attn_metadata", None)
            bl = getattr(md, "block_list", None)
            acc["blocks"] = int(bl.shape[0]) if bl is not None else 0
            if acc["n"] % 300 == 0:
                print(f"{TAG} decode input preparation: {acc['t'] / 300 * 1e3:.2f} ms per step (last bucket "
                      f"{acc['blocks']} blocks)", flush=True)
                acc["t"] = 0.0
            return out

        runner._create_decode_input_data = _create_decode_input_data
    print(f"{TAG} installed: decode block lists are built with numpy", flush=True)
