# SPDX-License-Identifier: Apache-2.0
"""CPU check: glm53_hpu.fastmeta.paged_buffers returns what vllm-gaudi's list code returns.  python3 test_fastmeta.py"""
import itertools
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from glm53_hpu import fastmeta  # noqa: E402


def stock(block_tables, slot_mapping, size, block_size, pad_id, dtype):
    """The non-contiguous branch of vllm-gaudi v0.30.0 get_habana_paged_attn_buffers, verbatim logic."""
    last_block_usage = [slot[0] % block_size + 1 for slot in slot_mapping]
    block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
    block_usage = [[block_size] * (len(bt) - 1) + [lbu] for bt, lbu in zip(block_tables, last_block_usage) if bt]
    block_list = list(itertools.chain(*block_tables))
    block_groups = list(itertools.chain(*block_groups))
    block_usage = list(itertools.chain(*block_usage))

    def pad(x, v):
        return x + list(itertools.islice(itertools.repeat(v), size - len(x)))

    return (torch.tensor(pad(block_list, pad_id), dtype=torch.long),
            torch.tensor(pad(block_groups, -1), dtype=torch.long),
            torch.tensor(pad(block_usage, 1), dtype=dtype))


def main():
    rng = random.Random(3)
    for case in range(200):
        rows = rng.choice([1, 2, 3, 4, 6, 8])
        bs = 128
        tables, slots = [], []
        for _ in range(rows):
            n = rng.choice([0, 1, 2, 7, 40, 313])
            tables.append([rng.randrange(0, 5632) for _ in range(n)])
            slots.append([rng.randrange(0, 720000)])
        total = sum(len(t) for t in tables)
        size = total + rng.choice([0, 1, 5, 64])
        a = stock(tables, slots, size, bs, 5632, torch.bfloat16)
        b = fastmeta.paged_buffers(tables, slots, size, bs, 5632, torch.bfloat16)
        for x, y in zip(a, b):
            assert x.dtype == y.dtype and torch.equal(x, y), case
    # the 700K, three-row step
    bt = list(range(1, 5482))
    tables = [bt[:5480], bt[:5480], bt]
    slots = [[701440 + j] for j in range(3)]
    t0 = time.perf_counter()
    for _ in range(20):
        stock(tables, slots, 16896, 128, 5632, torch.bfloat16)
    t1 = time.perf_counter()
    for _ in range(20):
        fastmeta.paged_buffers(tables, slots, 16896, 128, 5632, torch.bfloat16)
    t2 = time.perf_counter()
    print(f"fastmeta: ALL OK; 700K three-row step: stock {(t1 - t0) / 20 * 1e3:.2f} ms, numpy {(t2 - t1) / 20 * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
