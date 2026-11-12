# SPDX-License-Identifier: Apache-2.0
"""Exact top-k selection shaped for Gaudi 2's TPC kernels (plain torch; runs on the CPU for tests).

Measured on one card (tools/scan_bench.py): torch.topk / torch.sort run one row per TPC core, so a [2, 90K] top-2048
keeps 2 of the 24 cores busy for about 170 us, while 48 rows of 4096 finish in two rounds of 18 us, and a small
elementwise op costs about 1 us. Hence:

  sorted_topk   level 0: top-k of fixed-size chunks (many short rows, all cores), then pairwise merges
  merge_lists   pairwise merge of sorted lists by the bitonic split: for two descending lists A and B of length k,
                max(A[t], B[k-1-t]) over t is exactly the multiset of the k largest of A and B (Batcher); a sort of
                k elements per list makes it a sorted list again for the next level.

Both return exactly the k largest values with their indices; among equal values at the cut the choice is arbitrary
but deterministic (the same input gives the same output on every rank).
"""
import torch

NEG_INF = -float("inf")


def merge_lists(v, i, sort_last=True):
    """v, i [J, L, k]: L lists per row, each sorted descending along the last dim (pad lists with -inf).
    Returns (v [J, k], i [J, k]): the k largest of each row's lists, sorted descending if sort_last."""
    j, _, k = v.shape
    while v.shape[1] > 1:
        n = v.shape[1]
        if n % 2:
            v = torch.cat((v, torch.full((j, 1, k), NEG_INF, dtype=v.dtype, device=v.device)), 1)
            i = torch.cat((i, torch.zeros((j, 1, k), dtype=i.dtype, device=i.device)), 1)
            n += 1
        v = v.view(j, n // 2, 2, k)
        i = i.view(j, n // 2, 2, k)
        a, b = v[:, :, 0], v[:, :, 1].flip(-1)
        take = a >= b
        v = torch.maximum(a, b)
        i = torch.where(take, i[:, :, 0], i[:, :, 1].flip(-1))
        if n > 2 or sort_last:
            v, p = torch.sort(v, dim=-1, descending=True)
            i = torch.gather(i, -1, p)
    return v[:, 0], i[:, 0]


def sorted_topk(s, k, chunk=4096, sort_last=True):
    """s [J, n] -> (values [J, k], indices [J, k]): the exact top-k of every row. n >= k; chunk >= k."""
    j, n = s.shape
    if n <= chunk:
        return torch.topk(s, k, dim=-1)
    g = -(-n // chunk)
    if g * chunk != n:
        s = torch.nn.functional.pad(s, (0, g * chunk - n), value=NEG_INF)
    s = s.view(j, g, chunk)
    if chunk == k:
        v, i = torch.sort(s, dim=-1, descending=True)
    else:
        v, i = torch.topk(s, k, dim=-1)
    i = i + (torch.arange(g, device=s.device, dtype=i.dtype) * chunk).view(1, g, 1)
    return merge_lists(v, i, sort_last)
