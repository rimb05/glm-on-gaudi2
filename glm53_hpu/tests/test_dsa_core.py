# SPDX-License-Identifier: Apache-2.0
# Tests of glm53_hpu/dsa.py, which derives from the gaudi-cookbook (Apache-2.0); same licence (see NOTICE).
"""CPU checks for the long-context revision of the sparse-attention module (glm53_hpu/dsa.py).

  python3 test_dsa_core.py        (needs torch; no Gaudi device)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("GLM_DSA_SPARSE", "1")
from glm53_hpu import dsa  # noqa: E402


def make_decode_case(rows, blocks_per_row, t, d, hi, pad, seed):
    g = torch.Generator().manual_seed(seed)
    nb_valid = sum(blocks_per_row)
    nb = nb_valid + pad
    phys_blocks = max(blocks_per_row) + 3
    buf = torch.randn(phys_blocks * t, d, generator=g)
    block_list, groups = [], []
    for r, n in enumerate(blocks_per_row):  # every row lists the same physical blocks 1..n (like verify rows do)
        block_list += list(range(1, n + 1))
        groups += [r] * n
    block_list += [0] * pad
    groups += [-1] * pad
    bias = torch.zeros(nb, t)
    off = 0
    for r, n in enumerate(blocks_per_row):  # the last block of each row is partly filled
        used = 5 + 7 * r
        bias[off + n - 1, used:] = -float("inf")
        off += n
    bias[nb_valid:] = -float("inf")
    q = torch.randn(rows, hi, d, generator=g)
    w = torch.rand(rows, hi, generator=g) + 0.1
    return buf, torch.tensor(block_list), torch.tensor(groups), bias, q, w


def sharded(buf, block_list, groups, bias, q, w, topk, t, world):
    nb = block_list.shape[0]
    d = q.shape[2]
    per = -(-nb // world)
    cand = []
    for rank in range(world):
        pos = torch.arange(rank * per, (rank + 1) * per)
        inside = pos < nb
        posc = pos.clamp(max=nb - 1)
        grp = torch.where(inside, groups.index_select(0, posc), -1)
        k_loc = buf.unflatten(0, (-1, t)).index_select(0, block_list.index_select(0, posc)).reshape(per, t, d)
        v, flat = dsa.decode_topk_local(q, w, k_loc, grp, bias.index_select(0, posc), posc, topk)
        cand.append(torch.stack((v, flat), dim=1))
    return dsa.decode_topk_merge(torch.stack(cand, 0), topk)


def test_sharded_topk():
    t, d, hi, topk = 16, 12, 5, 64
    for seed, (bpr, pad, world) in enumerate([((40, 40), 0, 8), ((37, 38), 5, 8), ((90,), 6, 8), ((33, 34, 20, 21), 4, 4)]):
        rows = len(bpr)
        buf, bl, grp, bias, q, w = make_decode_case(rows, bpr, t, d, hi, pad, seed)
        kb = buf.unflatten(0, (-1, t)).index_select(0, bl)
        want = dsa.decode_topk(q, w, kb, grp, bias, topk)
        got = sharded(buf, bl, grp, bias, q, w, topk, t, world)
        for r in range(rows):
            a = set(want[r][want[r] >= 0].tolist())
            b = set(got[r][got[r] >= 0].tolist())
            assert a == b, (seed, r, len(a), len(b), len(a ^ b))
            assert len(a) == topk
    # a row with fewer valid keys than topk keeps them all and reports -1 for the rest
    buf, bl, grp, bias, q, w = make_decode_case(2, (40, 2), t, d, hi, 6, 9)
    kb = buf.unflatten(0, (-1, t)).index_select(0, bl)
    want = dsa.decode_topk(q, w, kb, grp, bias, topk)
    got = sharded(buf, bl, grp, bias, q, w, topk, t, 8)
    assert set(want[1][want[1] >= 0].tolist()) == set(got[1][got[1] >= 0].tolist())
    assert int((got[1] >= 0).sum()) == 16 + 12  # one full block + 12 used slots of the second
    print("sharded top-k: ok")


def test_pool_scan():
    """The in-place pool scan picks the same physical slots as the original scan over the gathered block lists."""
    t, d, hi, topk, world = 16, 12, 5, 64, 8
    for seed, (bpr, pad) in enumerate([((40, 40), 0), ((37, 38), 5), ((90,), 6), ((33, 34, 20, 21), 4)]):
        rows = len(bpr)
        buf, bl, grp, bias, q, w = make_decode_case(rows, bpr, t, d, hi, pad, 20 + seed)
        total = buf.shape[0] // t
        kb = buf.unflatten(0, (-1, t)).index_select(0, bl)
        want = dsa.decode_topk(q, w, kb, grp, bias, topk)                    # flat positions in the block list
        ob = dsa.owner_bias(bl, grp, bias, rows, total)
        cand = []
        for rank in range(world):
            p0, p1 = dsa._pool_split(total, world, rank)
            if p1 == p0:
                cand.append(torch.stack((torch.full((rows, topk), dsa.SENT), torch.zeros(rows, topk)), 1))
                continue
            k_loc = buf[p0 * t:p1 * t]
            b = ob[:, p0:p1].reshape(rows, -1)
            n = k_loc.shape[0]
            if n < topk:                                                    # tiny pools in this test only
                k_loc = torch.cat((k_loc, torch.zeros(topk - n, d)))
                b = torch.cat((b, torch.full((rows, topk - n), -float("inf"))), 1)
            cand.append(torch.stack(dsa.pool_topk_local(q, w, k_loc, b, p0 * t, topk), 1))
        got = dsa.decode_topk_merge(torch.stack(cand, 0), topk)             # physical slots
        for r in range(rows):
            wi = want[r][want[r] >= 0]
            a = set((bl[wi // t] * t + wi % t).tolist())
            b = set(got[r][got[r] >= 0].tolist())
            assert a == b, (seed, r, len(a), len(b), len(a ^ b))
    print("pool scan: ok")


class RefSDPA:
    @staticmethod
    def apply(q, k, v, mask, dropout, causal, scale, *rest):
        s = torch.matmul(q, k.transpose(-1, -2)) * scale + mask
        return torch.matmul(torch.softmax(s, dim=-1), v)


def test_dense_masked():
    dsa._FSDPA = RefSDPA
    g = torch.Generator().manual_seed(3)
    lora, rope, nope, vdim = 10, 4, 6, 10
    qk = nope + rope
    h, q_len, cb = 8, 512, 256
    kv = cb + q_len
    latent = torch.randn(kv, lora + rope, generator=g)
    w = torch.randn(h * (nope + vdim), lora, generator=g) / 3
    q = torch.randn(q_len, h, qk, generator=g)
    sel = torch.rand(q_len, kv, generator=g) < 0.3
    j = torch.arange(kv).view(1, kv)
    r = torch.arange(q_len).view(-1, 1)
    sel = sel & ((j < cb) | (j - cb <= r))
    sel[:, 0] = True
    mask = torch.where(sel, 0.0, dsa.NEG)
    # reference: all heads, all rows at once
    k_c, k_r = latent.split([lora, rope], dim=-1)
    kvn = torch.nn.functional.linear(k_c, w).view(kv, h, nope + vdim)
    k = torch.cat((kvn[..., :nope], k_r.view(kv, 1, rope).expand(kv, h, rope)), dim=-1)
    v = kvn[..., nope:]
    s = torch.einsum("qhd,khd->hqk", q, k) * 0.25 + mask.unsqueeze(0)
    want = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v)
    dims = (lora, rope, nope, vdim, qk)
    packed = dsa.pack_bits(sel.to(torch.bfloat16).bool()) if False else None
    for kvb, mb in ((1 << 40, 1 << 40), (kv * ((nope + vdim) + 2 * qk) * 2 * 2, 1 << 40), (1, 128 * kv * 2)):
        dsa.KV_BUDGET_BYTES, dsa.MASK_BUDGET_BYTES = kvb, mb
        plan = dsa.dense_plan(q_len, kv, h, kv * ((nope + vdim) + 2 * qk) * 2)
        got = dsa.dense_masked_attn(q, latent, w, dims, mask, cb, 0.25)[..., :vdim]
        err = float((got - want).abs().max())
        assert err < 1e-4, (plan, err)
        print(f"dense-masked plan (heads per group, rows per block) = {plan}: max abs error {err:.2e}")
    # bit-packed selection (the >131K-key path)
    wts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float32).view(8, 1)
    packed = torch.matmul(sel.reshape(q_len * kv // 8, 8).float(), wts).view(q_len, kv // 8).to(torch.uint8)
    got = dsa.dense_masked_attn(q, latent, w, dims, None, cb, 0.25, packed=packed)[..., :vdim]
    assert float((got - want).abs().max()) < 1e-4
    print("dense-masked attention: ok")


def test_bits_to_bias():
    g = torch.Generator().manual_seed(5)
    sel = torch.rand(6, 80, generator=g) < 0.4
    wts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.float32).view(8, 1)
    packed = torch.matmul(sel.reshape(6 * 10, 8).float(), wts).view(6, 10).to(torch.uint8)
    for kend in (80, 77, 64, 9):
        got = dsa.bits_to_bias(packed, kend, torch.float32)
        want = torch.where(sel[:, :kend], 0.0, dsa.NEG)
        assert torch.equal(got, want), kend
    print("bits_to_bias: ok")


def test_plan():
    dsa.KV_BUDGET_BYTES, dsa.MASK_BUDGET_BYTES = 3 << 30, int(0.75 * 2**30)
    per_head = lambda kv: kv * ((192 + 256) + 2 * 256) * 2  # noqa: E731
    assert dsa.dense_plan(8192, 40000, 8, per_head(40000)) == (8, 2048)
    assert dsa.dense_plan(8192, 131072, 8, per_head(131072)) == (8, 2048)
    g, rb = dsa.dense_plan(8192, 725000, 8, per_head(725000))
    assert g == 2 and rb == 512, (g, rb)
    assert dsa.dense_plan(128, 725000, 8, per_head(725000)) == (2, 128)
    print("plans: ok")


if __name__ == "__main__":
    test_sharded_topk()
    test_pool_scan()
    test_dense_masked()
    test_bits_to_bias()
    test_plan()
    print("ALL OK")


def test_int8_cache():
    """Linear 8-bit latent cache with group scales: round trip error with parts of very different size, the three
    access paths agree, sparse attention on the fetched rows = attention on the true rows within the code's error."""
    g = torch.Generator().manual_seed(11)
    slots, c, lora, t, grp = 64 * 16, 40, 32, 16, 8
    cache = torch.zeros(slots, c, dtype=torch.int8)
    op = dsa.LatentInt8Cache(slots, "cpu", lora, c, grp)
    op.scale = op.scale.float()                                  # bf16 scales need the device; fp32 on the CPU
    x = torch.randn(200, c, generator=g) * torch.rand(200, 1, generator=g).add(0.1)   # tokens of different size
    x[:, lora:] *= 60.0                                          # the rotary part far above the compressed KV
    where = torch.randperm(slots, generator=g)[:200]
    op(x, cache, where)
    back = op.fetch_slots(cache, where)
    for name, sl in (("kv", slice(0, lora)), ("rope", slice(lora, c))):
        err = float((back[:, sl] - x[:, sl]).norm() / x[:, sl].norm())
        assert err < 0.01, (name, err)
    assert int(cache.abs().max()) == 127 and op.scale.shape == (slots, c // grp)
    blocks = torch.unique(where // t)
    by_block = op.fetch_from_cache(cache.unflatten(0, (-1, t)), blocks).reshape(-1, c)
    flat = (blocks.view(-1, 1) * t + torch.arange(t).view(1, -1)).reshape(-1)
    assert torch.equal(by_block, op.fetch_slots(cache, flat))
    four = op.fetch_from_cache(cache.unsqueeze(1).unflatten(0, (-1, t)), blocks)
    assert four.shape == (blocks.numel(), t, 1, c) and torch.equal(four.reshape(-1, c), by_block)
    phys = where[:150].view(3, 50)
    q = torch.randn(3, 4, c, generator=g)
    q[..., lora:] /= 60.0
    bias = torch.zeros(3, 50)
    bias[:, -4:] = dsa.NEG
    got = dsa.sparse_decode_attn(q, cache, phys, bias, 0.3, lora, op)
    lat = x[:150].view(3, 50, c)
    ref = torch.bmm(torch.softmax(torch.bmm(q * 0.3, lat.transpose(1, 2)) + bias.unsqueeze(1), -1), lat[..., :lora])
    assert float((got - ref).norm() / ref.norm()) < 0.02
    print("int8 latent cache: ok")


def test_kth_threshold():
    g = torch.Generator().manual_seed(5)
    acc = torch.randn(6, 5000, generator=g)
    ok = torch.rand(6, 5000, generator=g) > 0.2
    ok[5, 40:] = False                                           # a row with fewer valid keys than k
    acc = torch.where(ok, acc, -float("inf"))
    ref = dsa.bisect_threshold(acc, ok, 64, 40)
    for mode in ("topk", "topk2"):
        dsa.PREFILL_KTH = mode
        thr = dsa.kth_threshold(acc, ok, 64)
        sel = (acc >= thr) & ok
        assert sel[:5].sum(1).tolist() == [64] * 5 and int(sel[5].sum()) == int(ok[5].sum()), mode
        assert torch.equal(sel, (acc >= ref) & ok), mode         # 40 halvings: the bisection is exact here
    dsa.PREFILL_KTH = "bisect"
    print("prefill threshold by top-k: ok")


def test_prefill_scores():
    g = torch.Generator().manual_seed(3)
    n, hi, d, kv = 48, 8, 16, 700
    q = torch.randn(n, hi, d, generator=g)
    w = torch.randn(n, hi, generator=g)
    kt = torch.randn(d, kv, generator=g)
    ref = (torch.relu(torch.einsum("nhd,dk->nhk", q, kt)) * w.unsqueeze(-1)).sum(1)
    old_budget, old_parts = dsa.SCORE_BUDGET_BYTES, dsa.PREFILL_PARTS
    for budget in (1 << 40, 16 * hi * kv * 2):                   # one sub-block, and sub-blocks of 16 rows
        for parts in (1, 2, 4):
            dsa.SCORE_BUDGET_BYTES, dsa.PREFILL_PARTS = budget, parts
            got = dsa.prefill_scores(q, w, kt)
            assert got.shape == (n, kv) and torch.allclose(got, ref, atol=1e-4, rtol=1e-4), (budget, parts)
    dsa.SCORE_BUDGET_BYTES, dsa.PREFILL_PARTS = old_budget, old_parts
    # the selection of a whole prefill is the same with both ways of scoring
    seq = torch.tensor([n], dtype=torch.int32)
    ctx = torch.tensor([kv - n], dtype=torch.int32)
    outs = []
    for mode in ("chunk", "rows"):
        dsa.PREFILL_SCORE = mode
        outs.append(dsa.prefill_select(q, w, kt, 0, n, kv - n, ctx, seq, 64, 16))
    dsa.PREFILL_SCORE = "rows"
    same = float((outs[0][0] == outs[1][0]).float().mean())
    assert same > 0.9995, same                                   # fp32 summation order may move a boundary key
    print(f"prefill scores: ok (selection agreement {same:.5f})")


test_int8_cache()
test_kth_threshold()
test_prefill_scores()
