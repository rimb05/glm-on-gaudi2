# SPDX-License-Identifier: MIT
"""CPU checks for glm53_hpu/topsel.py:  python3 test_topsel.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from glm53_hpu import topsel  # noqa: E402


def check(s, k, chunk):
    v, i = topsel.sorted_topk(s, k, chunk)
    wv, _ = torch.topk(s, k, dim=-1)
    assert torch.equal(v, wv), (s.shape, k, chunk)
    assert torch.equal(torch.gather(s, 1, i), v)                    # indices point at the values
    for r in range(s.shape[0]):
        assert len(set(i[r].tolist())) == k                         # no index twice


def main():
    g = torch.Generator().manual_seed(7)
    for n, k, chunk in ((90240, 2048, 4096), (90240, 2048, 2048), (90240, 2048, 8192), (16384, 2048, 2048),
                        (5000, 64, 128), (4097, 64, 64), (300, 16, 32), (2048 * 3 + 5, 2048, 2048)):
        s = torch.randn(2, n, generator=g)
        check(s, k, chunk)
        check((s * 4).round() / 4, k, chunk)                        # many ties
        s2 = s.clone()
        s2[:, n // 3:] = -float("inf")                              # rows with masked tails
        check(s2, k, chunk)
    # all the large values in one chunk
    s = torch.randn(3, 20000, generator=g)
    s[:, 4096:8192] += 100
    check(s, 2048, 4096)
    # merge of lists from several "ranks", some with fewer valid entries
    lists = torch.randn(2, 8, 2048, generator=g)
    lists[0, 3, 100:] = -float("inf")
    lists[1, 5] = -float("inf")
    v, _ = torch.sort(lists, dim=-1, descending=True)
    idx = torch.arange(8 * 2048).view(1, 8, 2048).expand(2, -1, -1)
    idx = torch.gather(idx, -1, torch.sort(lists, dim=-1, descending=True)[1])
    for sort_last in (True, False):
        mv, mi = topsel.merge_lists(v, idx, sort_last)
        want = torch.topk(lists.reshape(2, -1), 2048, dim=-1)[0]
        assert torch.equal(torch.sort(mv, dim=-1, descending=True)[0], want)
        assert torch.equal(torch.gather(lists.reshape(2, -1), 1, mi), mv)
    print("topsel: ALL OK")


if __name__ == "__main__":
    main()
