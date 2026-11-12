#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Device-time benchmark of the decode-time indexer scan and sparse attention on Gaudi 2 (one process per card).

  RANK=r WORLD_SIZE=n MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 python3 scan_bench.py        (see run_scan_bench.sh)

Every variant is an nn.Module wrapped in an HPU graph (as vllm-gaudi wraps the model), holding `reps` repetitions of
the operation with slightly different queries. Time per call is measured for reps = 1 and reps = REPS; the slope
(t_REPS - t_1) / (REPS - 1) is the device time of one repetition without the graph launch, which is what one more
operation inside a decoder layer's graph costs. With WORLD_SIZE > 1 the variants that exchange candidates do it with
a real HCCL collective, so the figure includes the collective's latency.

Sizes follow the deployment: 5632 KV blocks of 128 tokens (+1 pad block), 8 ranks, 2 query rows (MTP), 32 indexer
heads of 128 dims, top 2048, latent 576, 8 local attention heads.
"""
import os
import sys
import time

import torch
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glm53_hpu import topsel  # noqa: E402
from glm53_hpu import dsa  # noqa: E402

rank = int(os.environ.get("RANK", "0"))
world = int(os.environ.get("WORLD_SIZE", "1"))
REPS = int(os.environ.get("REPS", "5"))
ITERS = int(os.environ.get("ITERS", "200"))
VARIANTS = [v for v in os.environ.get("VARIANTS", "").split(",") if v]
BLOCKS = int(os.environ.get("BLOCKS", "5632"))
T, D, HI, R, TOPK, C, H = 128, 128, 32, int(os.environ.get("ROWS", "2")), 2048, 576, 8
SHARDS = int(os.environ.get("SHARDS", "8"))  # the scan is always sized for 8 ranks, also when WORLD_SIZE=1
SENT = -1.0e30
dev = torch.device("hpu")

if world > 1:
    import habana_frameworks.torch.distributed.hccl  # noqa: F401

    torch.distributed.init_process_group(backend="hccl", rank=rank, world_size=world)


def log(msg):
    if rank == 0:
        print(msg, flush=True)


def all_reduce(x):
    if world > 1:
        torch.distributed.all_reduce(x)
    return x


def all_gather(x):
    """[..] -> [world, ..]"""
    if world == 1:
        return x.unsqueeze(0)
    out = [torch.empty_like(x) for _ in range(world)]
    torch.distributed.all_gather(out, x)
    return torch.stack(out, 0)


g = torch.Generator().manual_seed(1234)
SLOTS = (BLOCKS + 1) * T
PER = -(-(BLOCKS + 1) // SHARDS)            # physical blocks per rank in the pool scan
NLOC = PER * T
key_buf = (torch.randn(SLOTS, 1, D, generator=g) * 0.3).to(torch.bfloat16).to(dev)
q0 = (torch.randn(R, HI, D, generator=g) * 0.3).to(torch.bfloat16).to(dev)
w0 = (torch.randn(R, HI, generator=g) * 0.1).to(torch.bfloat16).to(dev)
# a sequence that owns blocks 1..BLOCKS-8 in order (as a fresh server allocates them); two rows list the same blocks
NB1 = BLOCKS - 8
bl_rows = torch.arange(1, NB1 + 1, dtype=torch.long)
block_list = torch.cat([bl_rows] * R + [torch.zeros(8 * R, dtype=torch.long)]).to(dev)       # [R*BLOCKS]
block_groups = torch.cat([torch.full((NB1,), r, dtype=torch.long) for r in range(R)] +
                         [torch.full((8 * R,), -1, dtype=torch.long)]).to(dev)
bb = torch.zeros(R * BLOCKS, T)
bb[R * NB1:] = -float("inf")
for r in range(R):
    bb[(r + 1) * NB1 - 1, 40 + r:] = -float("inf")                                            # last block partly used
block_bias = bb.to(torch.bfloat16).to(dev)
# pool scan: additive ownership bias per physical slot and row, this rank's part
own = torch.full((BLOCKS + 1, T, R), -float("inf"))
own[1:NB1 + 1] = 0.0
for r in range(R):
    own[NB1, 40 + r:, r] = -float("inf")
P0 = min(rank, SHARDS - 1) * PER
P1 = min(P0 + PER, BLOCKS + 1)
pool_bias = own[P0:P1].reshape(-1, R).to(dev)                                                 # [n_loc, R] fp32
N_MINE = (P1 - P0) * T
key_t = key_buf[P0 * T:P1 * T].reshape(N_MINE, D).transpose(0, 1).contiguous()               # [D, n_loc]
pool_bias_t = pool_bias.transpose(0, 1).contiguous()                                          # [R, n_loc]
lat_buf = (torch.randn(SLOTS, 1, C, generator=g) * 0.2).to(torch.bfloat16).to(dev)
qabs0 = (torch.randn(R, H, C, generator=g) * 0.2).to(torch.bfloat16).to(dev)
phys0 = torch.randint(T, SLOTS - T, (R, TOPK), generator=g).to(dev)
torch.hpu.synchronize()


def topk2(s, k, m):
    """Exact top-k of [R, N] in two passes over groups of m (m <= 0: one pass)."""
    r, n = s.shape
    if m <= 0 or n <= 2 * m:
        return torch.topk(s, k, dim=-1)
    pad = (-n) % m
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=-float("inf"))
    gcount = (n + pad) // m
    v1, i1 = torch.topk(s.view(r, gcount, m), k, dim=-1)
    i1 = i1 + (torch.arange(gcount, device=s.device, dtype=i1.dtype) * m).view(1, gcount, 1)
    v2, i2 = torch.topk(v1.reshape(r, gcount * k), k, dim=-1)
    return v2, torch.gather(i1.reshape(r, gcount * k), 1, i2)


def exchange(v, flat, how):
    """Candidates of every rank -> [W, R, 2, k]."""
    mine = torch.stack((v, flat), dim=1)
    if how == "none":
        return mine.unsqueeze(0).expand(SHARDS, -1, -1, -1)
    if how == "gather":
        return all_gather(mine.contiguous())
    parts = []
    if rank:
        parts.append(torch.zeros(rank, *mine.shape, dtype=mine.dtype, device=mine.device))
    parts.append(mine.unsqueeze(0))
    if world - rank - 1 > 0:
        parts.append(torch.zeros(world - rank - 1, *mine.shape, dtype=mine.dtype, device=mine.device))
    return all_reduce(torch.cat(parts, 0) if len(parts) > 1 else parts[0])


def merge(cand, k):
    wn, r = cand.shape[0], cand.shape[1]
    cv = cand[:, :, 0].permute(1, 0, 2).reshape(r, wn * k)
    cp = cand[:, :, 1].permute(1, 0, 2).reshape(r, wn * k)
    v, i = torch.topk(cv, k, dim=-1)
    return torch.where(v > 0.5 * SENT, torch.gather(cp, 1, i).to(torch.int32), -1)


class Base(torch.nn.Module):
    def __init__(self, reps, **kw):
        super().__init__()
        self.reps = reps
        self.kw = kw

    def forward(self, q, w):
        # every repetition depends on the previous one (a term that is numerically zero), so repetitions cannot
        # overlap on idle TPC cores: total / reps is the latency of one
        o = None
        for _ in range(self.reps):
            o = self.one(q, w).float().sum()
            q = q + (torch.nan_to_num(o, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0) * 1e-30).to(q.dtype)
        return o


class GatherScan(Base):
    """The current revision at full context: this rank's share of the block list (both rows' copies), gathered."""

    def one(self, q, w):
        nb = block_list.shape[0]
        per = -(-nb // SHARDS)
        pos = torch.arange(min(rank, SHARDS - 1) * per, min(rank, SHARDS - 1) * per + per, device=dev, dtype=torch.long)
        inside = pos < nb
        posc = pos.clamp(max=nb - 1)
        bl = block_list.index_select(0, posc)
        grp = torch.where(inside, block_groups.index_select(0, posc), -1)
        k_loc = key_buf.unflatten(0, (-1, T)).index_select(0, bl).reshape(per, T, D)
        lg = torch.relu(torch.matmul(k_loc.reshape(per * T, D), q.reshape(R * HI, D).transpose(0, 1)))
        eye = torch.eye(R, device=dev, dtype=lg.dtype)
        wm = (w.unsqueeze(-1) * eye.unsqueeze(1)).reshape(R * HI, R)
        s = torch.matmul(lg, wm).float().view(per, T, R) + block_bias.index_select(0, posc).float().unsqueeze(-1)
        member = grp.unsqueeze(-1) == torch.arange(R, device=dev, dtype=torch.long).unsqueeze(0)
        s = torch.where(member.unsqueeze(1), s, -float("inf")).permute(2, 0, 1).reshape(R, per * T)
        if self.kw.get("stop") == "score":
            return s[:, :8]
        v, i = topk2(s, TOPK, 16384)
        flat = (posc.index_select(0, (i // T).reshape(-1)).view(i.shape) * T + i % T).to(torch.float32)
        if self.kw.get("stop") == "local":
            return v + flat
        return merge(exchange(v.clamp(min=SENT), flat, self.kw.get("x", "reduce")), TOPK)


class PoolScan(Base):
    """This rank's physical 1/8 of the key pool, scored in place for every row; ownership is an additive bias."""

    def one(self, q, w):
        k_loc = key_buf[P0 * T:P1 * T].reshape(N_MINE, D)
        qf = q.reshape(R * HI, D)
        if self.kw.get("fc", True):
            lg = torch.relu(torch.nn.functional.linear(k_loc, qf))
        else:
            lg = torch.relu(torch.matmul(k_loc, qf.transpose(0, 1)))
        eye = torch.eye(R, device=dev, dtype=lg.dtype)
        wm = (w.unsqueeze(-1) * eye.unsqueeze(1)).reshape(R * HI, R)
        s = torch.matmul(lg, wm)
        if self.kw.get("stop") == "gemm":
            return s[:8]
        s = (s.float() + pool_bias).transpose(0, 1)
        if self.kw.get("contig", True):
            s = s.contiguous()
        if self.kw.get("stop") == "score":
            return s[:, :8]
        v, i = topk2(s, TOPK, self.kw.get("m", 16384))
        flat = (i + P0 * T).to(torch.float32)
        if self.kw.get("stop") == "local":
            return v + flat
        return merge(exchange(v.clamp(min=SENT), flat, self.kw.get("x", "reduce")), TOPK)


class TopkOnly(Base):
    def __init__(self, reps, n, m, sorted_=True, rows=R):
        super().__init__(reps)
        self.n, self.m, self.sorted, self.rows = n, m, sorted_, rows
        self.s = (torch.randn(rows, n, generator=g)).to(dev)

    def one(self, q, w):
        s = self.s + q[0, 0, 0].float()
        if self.m == -1:
            v, i = torch.topk(s, TOPK, dim=-1, sorted=self.sorted)
        else:
            v, i = topk2(s, TOPK, self.m)
        return v[:, :4] + i[:, :4].float()


class TopkFull(Base):
    """top-k with every output consumed (a sliced output could let the compiler shrink the work)."""

    def __init__(self, reps, rows, n, k=TOPK, m=-1, op="topk", dtype=torch.float32):
        super().__init__(reps)
        self.rows, self.n, self.k, self.m, self.op = rows, n, k, m, op
        self.s = torch.randn(rows, n, generator=g).to(dtype).to(dev)
        self.dtype = dtype

    def one(self, q, w):
        s = self.s + q[0, 0, 0].to(self.dtype)
        if self.op == "sort":
            v, i = torch.sort(s, dim=-1, descending=True)
        elif self.op == "bisect":
            lo = s.amin(1, keepdim=True)
            hi = s.amax(1, keepdim=True)
            for _ in range(self.m):
                mid = (lo + hi) * 0.5
                ge = (s >= mid).sum(1, keepdim=True) >= self.k
                lo = torch.where(ge, mid, lo)
                hi = torch.where(ge, hi, mid)
            return lo
        elif self.op == "multi":      # 64-way threshold counting, self.m rounds
            lo = s.amin(1, keepdim=True)
            hi = s.amax(1, keepdim=True)
            steps = torch.arange(1, 65, device=dev, dtype=torch.float32).view(1, 64) / 64.0
            for _ in range(self.m):
                th = lo + (hi - lo) * steps                                   # [rows, 64] ascending
                cnt = (s.unsqueeze(-1) >= th.unsqueeze(1)).sum(1)             # [rows, 64]
                ok = cnt >= self.k                                            # true for a prefix of thresholds
                nok = ok.sum(1, keepdim=True)                                 # how many thresholds still have >= k
                width = (hi - lo) / 64.0
                lo = lo + width * nok.float()
                hi = lo + width
            return lo
        elif self.op == "amax":       # the cost of the chaining op alone (s + q) plus one reduction
            return s.amax(1, keepdim=True)
        elif self.op == "kth":        # what a prefill row block needs: the k-th largest value per row
            if self.m > 0:
                return dsa.kth_largest(s, self.k, self.m)
            return torch.topk(s, self.k, dim=-1).values[:, -1:]
        elif self.op == "bisectp":    # dsa.bisect_threshold as the prefill uses it (with its graph cuts)
            return dsa.bisect_threshold(s, torch.ones_like(s, dtype=torch.bool), self.k, self.m)
        elif self.op == "cumsum":
            return torch.cumsum((s > 0.5).to(torch.int32), dim=1)[:, -1:].float()
        elif self.m > 0:
            v, i = topk2(s, self.k, self.m)
        else:
            v, i = torch.topk(s, self.k, dim=-1)
        return v.float().sum(1, keepdim=True) + i.float().sum(1, keepdim=True)


class Tiny(Base):
    """A chain of 20 small elementwise ops on [2, 4096] fp32: the per-op cost of small tensors inside a graph."""

    def one(self, q, w):
        x = q[:, 0, :1].float().expand(2, 4096) + 0.0
        for j in range(10):
            x = torch.maximum(x, x.flip(1) * 0.5) + 1.0
        return x


class Gemm(Base):
    """Stages of the pooled score: 1 = first product, 2 = + relu, 3 = + head-weighted product."""

    def one(self, q, w):
        k_loc = key_buf[P0 * T:P1 * T].reshape(N_MINE, D)
        lg = torch.matmul(k_loc, q.reshape(R * HI, D).transpose(0, 1))
        if self.kw["stage"] == 1:
            return lg
        lg = torch.relu(lg)
        if self.kw["stage"] == 2:
            return lg
        eye = torch.eye(R, device=dev, dtype=lg.dtype)
        return torch.matmul(lg, (w.unsqueeze(-1) * eye.unsqueeze(1)).reshape(R * HI, R))


class ExchangeOnly(Base):
    def one(self, q, w):
        v = q[:, 0, :1].float().expand(R, TOPK) + 0.0
        return exchange(v, v * 2, self.kw["x"])[:, :, :, :2]


class SmallReduce(Base):
    """The size of a TP all-reduce in a decoder layer: [R, 6144] bf16."""

    def one(self, q, w):
        x = q[:, 0, :1].expand(R, 6144) + 0.0
        return all_reduce(x.contiguous())[:, :4]


class SparseAttn(Base):
    def one(self, q, w):
        qa = qabs0 + q[0, 0, 0]
        lat = lat_buf.index_select(0, phys0.reshape(-1)).view(R, TOPK, C)
        if self.kw.get("stop") == "gather":
            return lat.float().sum(1)
        s = torch.bmm(qa * 0.07, lat.transpose(1, 2)).float()
        p = torch.softmax(s, dim=-1).to(lat.dtype)
        return torch.bmm(p, lat[..., :512])


class Sel(Base):
    """Exact local top-k of pooled scores with topsel.sorted_topk (scores come from a cheap op on a stored tensor)."""

    def __init__(self, reps, chunk, n=None, rows=R):
        super().__init__(reps)
        self.chunk = chunk
        self.s = torch.randn(rows, n or NLOC, generator=g).to(dev)

    def one(self, q, w):
        v, i = topsel.sorted_topk(self.s + q[0, 0, 0].float(), TOPK, self.chunk)
        return v.sum(1, keepdim=True) + i.float().sum(1, keepdim=True)


class Merge8(Base):
    def __init__(self, reps, how):
        super().__init__(reps)
        self.how = how
        v = torch.sort(torch.randn(R, SHARDS, TOPK, generator=g), dim=-1, descending=True)[0]
        self.v = v.to(dev)
        self.i = torch.randint(0, SLOTS, (R, SHARDS, TOPK), generator=g).to(dev)

    def one(self, q, w):
        v = self.v + q[0, 0, 0].float()
        if self.how == "topk":
            mv, p = torch.topk(v.reshape(R, SHARDS * TOPK), TOPK, dim=-1)
            mi = torch.gather(self.i.reshape(R, SHARDS * TOPK), 1, p)
        else:
            mv, mi = topsel.merge_lists(v, self.i, sort_last=False)
        return mv.sum(1, keepdim=True) + mi.float().sum(1, keepdim=True)


class PoolV3(Base):
    """In-place pooled scores, chunked selection, one all-reduce, pairwise merge."""

    def one(self, q, w):
        if self.kw.get("layout") in ("t", "tv"):
            kt = key_t if self.kw["layout"] == "t" else key_buf[P0 * T:P1 * T].reshape(N_MINE, D).transpose(0, 1)
            lg = torch.relu(torch.matmul(q.reshape(R * HI, D), kt))                          # [R*HI, N]
            eye = torch.eye(R, device=dev, dtype=lg.dtype)
            wm = (w.unsqueeze(-1) * eye.unsqueeze(1)).reshape(R * HI, R).transpose(0, 1)     # [R, R*HI]
            s = torch.matmul(wm, lg).float() + pool_bias_t
        else:
            k_loc = key_buf[P0 * T:P1 * T].reshape(N_MINE, D)
            lg = torch.relu(torch.matmul(k_loc, q.reshape(R * HI, D).transpose(0, 1)))
            eye = torch.eye(R, device=dev, dtype=lg.dtype)
            wm = (w.unsqueeze(-1) * eye.unsqueeze(1)).reshape(R * HI, R)
            s = (torch.matmul(lg, wm).float() + pool_bias).transpose(0, 1).contiguous()
        if self.kw.get("stop") == "score":
            return s.sum(1, keepdim=True)
        v, i = topsel.sorted_topk(s, TOPK, self.kw.get("chunk", 4096))
        flat = (i + P0 * T).to(torch.float32)
        if self.kw.get("stop") == "local":
            return v.sum(1, keepdim=True) + flat.sum(1, keepdim=True)
        cand = exchange(v.clamp(min=SENT), flat, self.kw.get("x", "reduce"))                 # [W, R, 2, k]
        mv, mi = topsel.merge_lists(cand[:, :, 0].transpose(0, 1), cand[:, :, 1].transpose(0, 1), sort_last=False)
        return torch.where(mv > 0.5 * SENT, mi, -1.0)


class PScore(Base):
    """Prefill indexer scores of one row block (rows x 32 heads x n keys), then nothing else.
    mode chunk: head chunks of hc heads with an fp32 running sum (dsa.prefill_select);
    mode rows:  sub-blocks of `sub` rows, all heads in one product, bf16 result."""

    def __init__(self, reps, rows, n, mode, hc=2, sub=32, out="bf16"):
        super().__init__(reps)
        self.rows, self.n, self.mode, self.hc, self.sub, self.out = rows, n, mode, hc, sub, out
        self.q = (torch.randn(rows, HI, D, generator=g) * 0.3).to(torch.bfloat16).to(dev)
        self.w = (torch.randn(rows, HI, generator=g) * 0.1).to(torch.bfloat16).to(dev)
        self.kt = (torch.randn(D, n, generator=g) * 0.3).to(torch.bfloat16).to(dev)

    def one(self, q, w):
        rows, n = self.rows, self.n
        qq = self.q + q[0, 0, 0]
        if self.mode == "chunk":
            acc = None
            for h0 in range(0, HI, self.hc):
                h1 = h0 + self.hc
                sc = torch.matmul(qq[:, h0:h1].reshape(rows * self.hc, D), self.kt)
                sc = torch.relu_(sc).view(rows, self.hc, n)
                sc = torch.bmm(self.w[:, h0:h1].unsqueeze(1), sc).squeeze(1).float()
                acc = sc if acc is None else acc + sc
        else:
            outs = []
            for a in range(0, rows, self.sub):
                b = a + self.sub
                lg = torch.relu(torch.matmul(qq[a:b].reshape(self.sub * HI, D), self.kt)).view(self.sub, HI, n)
                outs.append(torch.bmm(self.w[a:b].unsqueeze(1), lg).squeeze(1))
            acc = torch.cat(outs, 0)
            if self.out == "f32":
                acc = acc.float()
        return acc.amax(1, keepdim=True)


def split2(va, pa, vb, pb):
    """The k largest of two descending lists [.., k], as an unsorted set (bitonic split)."""
    b = vb.flip(-1)
    return torch.maximum(va, b), torch.where(va >= b, pa, pb.flip(-1))


def local_sel(s, k, g, tail):
    """Exact top-k (values, positions) of s [R, N] with a first top-k over g groups.
    tail 'topk': a second top-k over the g*k candidates (sorted result).
    tail 'split': pairwise bitonic splits of the group results, then top-k over what is left: g=2 nothing (unsorted
    result), g=4 one top-k over [R, 2k] (sorted), g>=8 one over [R, 2, g/4*k] and a last split (unsorted).
    tail 'splitsorted': splits, then one top-k over [R, g/2*k] (sorted)."""
    r, n = s.shape
    if g == 1:
        return torch.topk(s, k, dim=-1)
    m = -(-n // g)
    if g * m - n:
        s = torch.nn.functional.pad(s, (0, g * m - n), value=-float("inf"))
    v1, i1 = torch.topk(s.view(r, g, m), k, dim=-1)
    i1 = i1 + (torch.arange(g, device=s.device, dtype=i1.dtype) * m).view(1, g, 1)
    if tail == "topk":
        v2, i2 = torch.topk(v1.reshape(r, g * k), k, dim=-1)
        return v2, torch.gather(i1.reshape(r, g * k), 1, i2)
    v1 = v1.view(r, g // 2, 2, k)
    i1 = i1.view(r, g // 2, 2, k)
    v, p = split2(v1[:, :, 0], i1[:, :, 0], v1[:, :, 1], i1[:, :, 1])
    h = g // 2
    if h == 1:
        return v[:, 0], p[:, 0]
    if tail == "splitsorted" or h == 2:
        v2, i2 = torch.topk(v.reshape(r, h * k), k, dim=-1)
        return v2, torch.gather(p.reshape(r, h * k), 1, i2)
    v2, i2 = torch.topk(v.reshape(r, 2, h // 2 * k), k, dim=-1)
    p2 = torch.gather(p.reshape(r, 2, h // 2 * k), 2, i2)
    return split2(v2[:, 0], p2[:, 0], v2[:, 1], p2[:, 1])


def merge_pair(cand, k):
    """cand [W, R, 2, k] with every rank's list sorted: pairwise splits, one top-k over [R, 2, W/4*k], last split."""
    wn, r = cand.shape[0], cand.shape[1]
    cv = cand[:, :, 0].permute(1, 0, 2).reshape(r, wn // 2, 2, k)
    cp = cand[:, :, 1].permute(1, 0, 2).reshape(r, wn // 2, 2, k)
    v, p = split2(cv[:, :, 0], cp[:, :, 0], cv[:, :, 1], cp[:, :, 1])
    v2, i2 = torch.topk(v.reshape(r, 2, wn // 4 * k), k, dim=-1)
    p2 = torch.gather(p.reshape(r, 2, wn // 4 * k), 2, i2)
    bv, bp = split2(v2[:, 0], p2[:, 0], v2[:, 1], p2[:, 1])
    return torch.where(bv > 0.5 * SENT, bp.to(torch.int32), -1)


class Scan(Base):
    """The production pool scan (scores as dsa.pool_topk_local computes them) with a selectable selection:
    g = groups of the first top-k, tail = local_sel's tail, gm = half (dsa.decode_topk_merge) | pair (merge_pair),
    stop = score | local | (nothing: the whole scan with the exchange)."""

    def one(self, q, w):
        kw = self.kw
        kt = key_buf[P0 * T:P1 * T].reshape(N_MINE, D).transpose(0, 1)
        lg = torch.relu(torch.matmul(q.reshape(R * HI, D), kt))
        eye = torch.eye(R, device=dev, dtype=lg.dtype)
        wm = (w.unsqueeze(-1) * eye.unsqueeze(1)).reshape(R * HI, R).transpose(0, 1)
        s = torch.matmul(wm, lg).float() + pool_bias_t
        if kw.get("stop") == "score":
            return s.sum(1, keepdim=True)
        v, i = local_sel(s, TOPK, int(kw.get("g", 8)), kw.get("tail", "topk"))
        flat = (i + P0 * T).to(torch.float32)
        if kw.get("stop") == "local":
            return v.sum(1, keepdim=True) + flat.sum(1, keepdim=True)
        cand = exchange(v.clamp(min=SENT), flat, kw.get("x", "reduce"))
        if kw.get("gm", "half") == "pair":
            return merge_pair(cand, TOPK)
        return dsa.decode_topk_merge(cand, TOPK)


class AttnQ8(Base):
    """Sparse decode attention from an 8-bit latent cache. how = deq (LatentInt8Cache.fetch_slots, then the bf16
    attention) | e4m3 (LatentFP8Cache rows, then the bf16 attention) | gather / cast / mul1 / fold1 (pieces and
    single-scale variants, for timing only)."""

    def __init__(self, reps, how):
        super().__init__(reps)
        self.how = how
        if how == "e4m3":
            self.op = dsa.LatentFP8Cache(SLOTS, dev)
            self.cache = torch.zeros(SLOTS, C, dtype=torch.float8_e4m3fn, device=dev)
        else:
            self.op = dsa.LatentInt8Cache(SLOTS, dev, 512, C)
            self.cache = torch.randint(-127, 128, (SLOTS, C), generator=g, dtype=torch.int8).to(dev)
            self.op.scale = (torch.rand(SLOTS, self.op.ngroups, generator=g) * 0.02 + 0.01).to(torch.bfloat16).to(dev)
        self.bias = torch.zeros(R, TOPK, device=dev)

    def one(self, q, w):
        qa = qabs0 + q[0, 0, 0]
        flat = phys0.reshape(-1)
        if self.how in ("gather", "cast", "mul1", "fold1"):
            codes = self.cache.index_select(0, flat).view(R, TOPK, C)
            if self.how == "gather":
                return codes[:, :, :8].float().sum(1) + codes[:, :, -8:].float().sum(1)
            lat = codes.to(qa.dtype)
            sc = self.op.scale.index_select(0, flat).view(R, TOPK, -1)[..., :1]
            if self.how == "fold1":      # one scale per token, applied to scores and weights
                s = torch.bmm(qa * 0.07, lat.transpose(1, 2)).float() * sc.float().transpose(1, 2) + self.bias.unsqueeze(1)
                p = torch.softmax(s, dim=-1) * sc.float().transpose(1, 2)
                return torch.bmm(p.to(lat.dtype), lat[..., :512])
            if self.how == "mul1":       # one scale per token, applied to the gathered rows
                lat = lat * sc
            s = torch.bmm(qa * 0.07, lat.transpose(1, 2)).float() + self.bias.unsqueeze(1)
            p = torch.softmax(s, dim=-1).to(lat.dtype)
            return torch.bmm(p, lat[..., :512])
        lat = self.op.fetch_slots(self.cache, flat).view(R, TOPK, C)
        s = torch.bmm(qa * 0.07, lat.transpose(1, 2)).float() + self.bias.unsqueeze(1)
        p = torch.softmax(s, dim=-1).to(lat.dtype)
        return torch.bmm(p, lat[..., :512])


class Q8Stock(torch.nn.Module):
    """A decode step's cache access exactly as vllm-gaudi does it for a dense MLA layer: write the step's rows,
    then vllm_gaudi.extension.ops.flat_pa_mla with the cache op's fetch function. With op None the cache is a plain
    bf16 tensor written and read as vllm-gaudi's VLLMKVCache does (the reference build)."""

    def __init__(self, op, cache):
        super().__init__()
        from vllm_gaudi.extension.utils import B2BMatmul, Matmul

        self.op, self.cache = op, cache
        self.mqk, self.mav, self.b2b, self.b2b2 = Matmul(), Matmul(), B2BMatmul(), B2BMatmul()

    def fetch(self, c4, blocks, *a, **k):
        return self.op.fetch_from_cache(c4, blocks) if self.op is not None else c4.index_select(0, blocks)

    def forward(self, x3, where, query, block_list, block_mapping, block_bias, block_groups):
        from vllm_gaudi.extension import ops as vg

        if self.op is not None:
            self.op(x3, self.cache, where)
        else:
            self.cache.index_copy_(0, where, x3)
        return vg.flat_pa_mla(query=query, key_cache=self.cache.unsqueeze(1), value_cache=None,
                              block_list=block_list, block_mapping=block_mapping, block_bias=block_bias,
                              block_groups=block_groups, block_size=T, scale=0.07, matmul_qk_op=self.mqk,
                              matmul_av_op=self.mav, batch2block_matmul_op=self.b2b,
                              block2batch_matmul_op=self.b2b2, keys_fetch_func=self.fetch, values_fetch_func=None,
                              kv_lora_rank=512)


class Q8Sparse(torch.nn.Module):
    """A sparse decode layer's cache access: write the step's rows, then dsa.sparse_decode_attn on gathered slots."""

    def __init__(self, op, cache):
        super().__init__()
        self.op, self.cache = op, cache

    def forward(self, x3, where, q_abs, phys, bias):
        if self.op is not None:
            self.op(x3, self.cache, where)
        else:
            self.cache.index_copy_(0, where, x3)
        return dsa.sparse_decode_attn(q_abs, self.cache, phys, bias, 0.07, 512, self.op)


class Q8WriteOnly(torch.nn.Module):
    """A prefill without context as far as the cache is concerned: rows are written, the output does not read them."""

    def __init__(self, op, cache):
        super().__init__()
        self.op, self.cache = op, cache

    def forward(self, x, where):
        self.op(x, self.cache, where)
        return (x.float() * 2.0).sum(-1)


class VarCache(dsa.LatentInt8Cache):
    """LatentInt8Cache with pieces switched off or replaced, to see what each costs in a decode layer.
    write: std | none (no write at all) | codes (codes only, scales untouched) | lean (bf16 arithmetic, 1/126)
           | fast (fp32 codes from the unrounded scale, no eps, no clamp) | fastnr (fast without round: timing only)
    read:  std | cast (codes cast to bf16, no scale)"""

    def __init__(self, slots, device, split, width, write="std", read="std"):
        super().__init__(slots, device, split, width)
        self.wm, self.rm = write, read

    def forward(self, input, cache, slot_mapping, *args, **kwargs):
        if self.wm == "std":
            return super().forward(input, cache, slot_mapping)
        if self.wm == "none":
            return cache
        x = input.reshape(-1, self.group)
        if self.wm == "preonly":      # only the codes are written (a constant), scales untouched
            cache.index_copy_(0, slot_mapping, self.const)
            return cache
        if self.wm == "sconly":       # only the scales are written
            a = x.abs().amax(dim=-1, keepdim=True)
            self.scale.index_copy_(0, slot_mapping, (a * (1.0 / 127.0)).view(-1, self.ngroups))
            return cache
        if self.wm == "sc1":          # codes + a [slots, 1] scale tensor instead of [slots, 9]
            a = x.abs().amax(dim=-1, keepdim=True)
            cache.index_copy_(0, slot_mapping, self.const)
            self.scale1.index_copy_(0, slot_mapping, a.view(-1, self.ngroups)[:, :1] * (1.0 / 127.0))
            return cache
        if self.wm in ("pre", "iput", "scat", "i32", "viabf"):
            a = x.abs().amax(dim=-1, keepdim=True)
            y = torch.round(x.float() * (127.0 / a.float()))
            if self.wm == "pre":          # no cast in the graph: a constant int8 row block is copied
                if not hasattr(self, "const"):
                    raise RuntimeError("const not set")
                self.scale.index_copy_(0, slot_mapping, (a * (1.0 / 127.0) + (y.sum() * 0.0).to(a.dtype)).view(-1, self.ngroups))
                cache.index_copy_(0, slot_mapping, self.const)
                return cache
            if self.wm == "i32":
                c8 = y.to(torch.int32).to(cache.dtype).view(-1, *cache.shape[1:])
            else:
                c8 = y.to(cache.dtype).view(-1, *cache.shape[1:])
            if self.wm == "iput":
                cache.index_put_((slot_mapping,), c8)
            elif self.wm == "scat":
                cache.scatter_(0, slot_mapping.view(-1, 1).expand(-1, c8.shape[-1]), c8)
            else:
                cache.index_copy_(0, slot_mapping, c8)
            self.scale.index_copy_(0, slot_mapping, (a * (1.0 / 127.0)).view(-1, self.ngroups))
            return cache
        if self.wm in ("fast", "fastnr", "fastclamp"):
            # codes from the unrounded fp32 scale (the stored bf16 scale differs by at most 0.2 %), no eps, no clamp
            a = x.abs().amax(dim=-1, keepdim=True)
            y = x.float() * (127.0 / a.float())
            if self.wm == "fastclamp":
                y = y.clamp(-127.0, 127.0)
            y = y if self.wm == "fastnr" else torch.round(y)
            cache.index_copy_(0, slot_mapping, y.to(cache.dtype).view(-1, *cache.shape[1:]))
            self.scale.index_copy_(0, slot_mapping, (a * (1.0 / 127.0)).view(-1, self.ngroups))
            return cache
        if self.wm == "codes":
            cache.index_copy_(0, slot_mapping, torch.round(x.float() * 30.0).clamp(-127.0, 127.0).to(cache.dtype)
                              .view(-1, *cache.shape[1:]))
            return cache
        sc = (x.abs().amax(dim=-1, keepdim=True) * (1.0 / 126.0)).clamp(min=1e-12)       # lean: all in bf16
        cache.index_copy_(0, slot_mapping, torch.round(x / sc).to(cache.dtype).view(-1, *cache.shape[1:]))
        self.scale.index_copy_(0, slot_mapping, sc.view(-1, self.ngroups))
        return cache

    def dequant(self, codes, sc):
        if self.rm == "std":
            return super().dequant(codes, sc)
        return codes.to(sc.dtype)


def q8_time():
    """Per-call time of one decode layer's cache access (write 3 rows, read, attend) for several caches: the dense
    path through vllm-gaudi's flat_pa_mla at 2K tokens of context, and the sparse path over 3 x 2048 keys."""
    gg = torch.Generator().manual_seed(77)
    nblk, rows, per, fill = 64, 3, 17, 40
    slots = nblk * T
    xb = latents(slots, gg)
    ctx = torch.randperm(nblk - 1, generator=gg)[:per] + 1
    wr = ctx[-1] * T + fill + torch.arange(rows)
    block_list = ctx.repeat(rows)
    block_groups = torch.arange(rows).repeat_interleave(per)
    block_mapping = torch.nn.functional.one_hot(block_groups, rows).to(torch.bfloat16)
    bias = torch.zeros(rows * per, T)
    for r in range(rows):
        bias[(r + 1) * per - 1, fill + r + 1:] = -float("inf")
    query = (torch.randn(rows, H, C, generator=gg) * 0.3).to(torch.bfloat16)
    x3 = latents(rows, gg)
    k = 2048
    phys = torch.randint(0, slots, (rows, k), generator=gg)
    sbias = torch.zeros(rows, k)
    n = int(os.environ.get("Q8ITERS", "400"))

    def make(kind):
        if kind == "bf16":
            return None, xb.clone().to(dev)
        if kind == "e4m3":
            op = dsa.LatentFP8Cache(slots, dev)
            cache = torch.zeros(slots, C, dtype=torch.float8_e4m3fn, device=dev)
        else:
            w, r = (kind.split("/") + ["std", "std"])[1:3] if "/" in kind else ("std", "std")
            op = VarCache(slots, dev, 512, C, write=w, read=r)
            op.const = torch.randint(-127, 128, (rows, C), dtype=torch.int8).to(dev)
            op.scale1 = torch.ones(slots, 1, dtype=torch.bfloat16, device=dev)
            cache = torch.zeros(slots, C, dtype=torch.int8, device=dev)
        wm = getattr(op, "wm", "std")
        if wm in ("pre", "preonly", "sconly", "sc1"):
            op.wm = "fast"
        op(xb.to(dev), cache, torch.arange(slots, device=dev))
        op.wm = wm
        return op, cache

    def clock(mod, args):
        out = None
        for _ in range(40):
            out = mod(*args)
        out.float().sum().item()
        t0 = time.perf_counter()
        for _ in range(n):
            out = mod(*args)
            out.float().sum().item()
        return (time.perf_counter() - t0) / n * 1e6

    class Chain(torch.nn.Module):
        """N layers' cache accesses in one graph, each layer with its own cache: what one more layer costs."""

        def __init__(self, mods):
            super().__init__()
            self.mods = torch.nn.ModuleList(mods)

        def forward(self, *args):
            out = None
            x3 = args[0]
            for m in self.mods:
                o = m(x3, *args[1:])
                out = o if out is None else out + o
                x3 = x3 + (o.float().mean() * 1e-6).to(x3.dtype)     # the next layer's rows depend on this one
            return out

    nl = int(os.environ.get("Q8LAYERS", "0"))
    if nl:
        dargs = (x3.to(dev), wr.to(dev), query.to(dev), block_list.to(dev), block_mapping.to(dev),
                 bias.to(torch.bfloat16).to(dev), block_groups.to(dev))
        sargs = (x3.to(dev), wr.to(dev), query.to(dev), phys.to(dev), sbias.to(dev))
        for kind in os.environ.get("Q8KINDS", "bf16,int8,int8/lean,e4m3,int8/none,int8/std/cast").split(","):
            try:
                res = []
                for cls, args in ((Q8Stock, dargs), (Q8Sparse, sargs)):
                    t = {}
                    for cnt in (2, nl):
                        mods = [cls(*make(kind)) for _ in range(cnt)]
                        t[cnt] = clock(ht.hpu.wrap_in_hpu_graph(Chain(mods), disable_tensor_cache=True), args)
                    res.append((t[nl] - t[2]) / (nl - 2))
                log(f"TIME per layer in a {nl}-layer graph, {kind:16s} dense (2K ctx) {res[0]:7.1f} us   sparse {res[1]:7.1f} us")
            except Exception as e:
                log(f"TIME {kind}: FAILED {type(e).__name__}: {str(e)[:200]}")
        return
    kinds = os.environ.get("Q8KINDS", "bf16,bf16,int8,e4m3,int8/none,int8/codes,int8/lean,int8/std/cast,int8/none/cast").split(",")
    dargs = (x3.to(dev), wr.to(dev), query.to(dev), block_list.to(dev), block_mapping.to(dev),
             bias.to(torch.bfloat16).to(dev), block_groups.to(dev))
    sargs = (x3.to(dev), wr.to(dev), query.to(dev), phys.to(dev), sbias.to(dev))
    for kind in kinds:
        try:
            op, cache = make(kind)
            td = clock(ht.hpu.wrap_in_hpu_graph(Q8Stock(op, cache), disable_tensor_cache=True), dargs)
            op, cache = make(kind)
            ts = clock(ht.hpu.wrap_in_hpu_graph(Q8Sparse(op, cache), disable_tensor_cache=True), sargs)
            log(f"TIME {kind:18s} dense layer (2K ctx) {td:7.1f} us   sparse layer {ts:7.1f} us")
        except Exception as e:
            log(f"TIME {kind}: FAILED {type(e).__name__}: {str(e)[:200]}")


def latents(n, gen, ratio=50.0):
    """Latents shaped like GLM-5.3's early layers: a small compressed-KV part, a rotary part `ratio` times larger,
    tokens of different size."""
    x = torch.randn(n, C, generator=gen) * (torch.rand(n, 1, generator=gen) * 2 + 0.2)
    x[:, :512] *= 0.02
    x[:, 512:] *= 0.02 * ratio
    return x.to(torch.bfloat16)


def q8_check():
    """The two-scale int8 latent cache on the device, in plain lazy mode and inside compiled graphs, against fp32
    references on the CPU computed from the TRUE latents (so the figures are what the model would see)."""
    gg = torch.Generator().manual_seed(21)
    nblk, rows, per, fill = 64, 3, 12, 40
    slots = nblk * T
    xb = latents(slots, gg)
    xf = xb.float()

    def part_err(y, ref):
        y, ref = y.float().cpu().reshape(-1, C), ref.reshape(-1, C)
        return (float((y[:, :512] - ref[:, :512]).norm() / ref[:, :512].norm()),
                float((y[:, 512:] - ref[:, 512:]).norm() / ref[:, 512:].norm()))

    def fresh():
        cache = torch.zeros(slots, C, dtype=torch.int8, device=dev)
        op = dsa.LatentInt8Cache(slots, dev, 512, C)
        op(xb.to(dev), cache, torch.arange(slots, device=dev))          # plain lazy mode, as a long prefill writes
        return op, cache

    op, cache = fresh()
    blocks = torch.randperm(nblk, generator=gg)[:40]
    ek, er = part_err(op.fetch_slots(cache, torch.arange(slots, device=dev)), xf)
    log(f"CHECK lazy write + fetch_slots: error of the kv part {ek:.4f}, of the rope part {er:.4f}; "
        f"cpu reference: {dsa.codec_report(xf, 512)}")
    c4 = cache.unsqueeze(1).unflatten(0, (-1, T))
    ek, er = part_err(op.fetch_from_cache(c4, blocks.to(dev)), xf.view(nblk, T, C)[blocks])
    log(f"CHECK lazy fetch_from_cache (4-D): kv {ek:.4f}, rope {er:.4f}")

    # --- a write-only graph (a prefill without context): are codes and scales kept?
    where = torch.randperm(slots, generator=gg)[:128]
    mod = ht.hpu.wrap_in_hpu_graph(Q8WriteOnly(op, cache), disable_tensor_cache=True)
    for i in range(3):
        xi = latents(128, torch.Generator().manual_seed(300 + i), ratio=10.0 * (i + 1))
        mod(xi.to(dev), where.to(dev)).cpu()
        ek, er = part_err(op.fetch_slots(cache, where.to(dev)), xi.float())
        log(f"CHECK write-only graph, call {i}: kv {ek:.4f}, rope {er:.4f}")

    # --- a dense decode step through vllm-gaudi's flat_pa_mla, int8 cache against a bf16 cache and the truth
    ctx = torch.randperm(nblk - 1, generator=gg)[:per] + 1
    wr = ctx[-1] * T + fill + torch.arange(rows)
    block_list = ctx.repeat(rows)
    block_groups = torch.arange(rows).repeat_interleave(per)
    block_mapping = torch.nn.functional.one_hot(block_groups, rows).to(torch.bfloat16)
    bias = torch.zeros(rows * per, T)
    for r in range(rows):
        bias[(r + 1) * per - 1, fill + r + 1:] = -float("inf")
    query = (torch.randn(rows, H, C, generator=gg) * 0.3).to(torch.bfloat16)
    query[..., :512] *= 50.0                                              # so both parts matter in the scores
    outs = {}
    truth = None
    # the first graph of the process that goes through flat_pa_mla returns garbage whatever the cache (order
    # dependent, also with a plain bf16 cache): one throw-away run first
    for kind in ("bf16warmup", "bf16", "int8"):
        if kind.startswith("bf16"):
            op, cache = None, xb.clone().to(dev)
        else:
            op, cache = fresh()
        try:
            mod = ht.hpu.wrap_in_hpu_graph(Q8Stock(op, cache), disable_tensor_cache=True)
            out = None
            full = xf.clone()
            for i in range(3):
                xi = latents(rows, torch.Generator().manual_seed(400 + i))
                out = mod(xi.to(dev), wr.to(dev), query.to(dev), block_list.to(dev), block_mapping.to(dev),
                          bias.to(torch.bfloat16).to(dev), block_groups.to(dev))
                full[wr] = xi.float()
            outs[kind] = out.float().cpu()
        except Exception as e:
            log(f"CHECK dense decode graph, {kind}: FAILED {type(e).__name__}: {str(e)[:200]}")
            continue
        if truth is None:
            ref = []
            for r in range(rows):
                lat = full.view(nblk, T, C)[ctx].reshape(per * T, C)
                sco = (query[r].float() * 0.07) @ lat.t() + bias[r * per:(r + 1) * per].reshape(-1)
                ref.append(torch.softmax(sco, -1) @ lat[:, :512])
            truth = torch.stack(ref)
    for kind, o in outs.items():
        if kind == "bf16warmup":
            continue
        log(f"CHECK dense decode graph (flat_pa_mla), {kind}: relative error against the fp32 truth "
            f"{float((o - truth).norm() / truth.norm()):.4f}; per row "
            f"{[round(float((o[r] - truth[r]).norm() / truth[r].norm()), 4) for r in range(rows)]}; output rms "
            f"{float(o.pow(2).mean().sqrt()):.4f} (truth {float(truth.pow(2).mean().sqrt()):.4f}), nan {bool(torch.isnan(o).any())}")

    # --- a sparse decode step (dsa.sparse_decode_attn), int8 cache / bf16 cache
    k = 2048
    phys = torch.randint(0, slots, (rows, k), generator=gg)
    phys[:, :rows] = wr                                                   # the step's own rows are among the keys
    sbias = torch.zeros(rows, k)
    sbias[:, -7:] = dsa.NEG
    q_abs = query
    full = xf.clone()
    res = {}
    for kind in ("int8", "bf16"):
        if kind == "bf16":
            op, cache = None, xb.clone().to(dev)
        else:
            op, cache = fresh()
        mod = ht.hpu.wrap_in_hpu_graph(Q8Sparse(op, cache), disable_tensor_cache=True)
        out = None
        for i in range(3):
            xi = latents(rows, torch.Generator().manual_seed(500 + i))
            out = mod(xi.to(dev), wr.to(dev), q_abs.to(dev), phys.to(dev), sbias.to(dev))
            full[wr] = xi.float()
        res[kind] = out.float().cpu()
    lat = full[phys.reshape(-1)].view(rows, k, C)
    sref = torch.bmm(q_abs.float() * 0.07, lat.transpose(1, 2)) + sbias.unsqueeze(1)
    tr = torch.bmm(torch.softmax(sref, -1), lat[..., :512])
    for kind, o in res.items():
        log(f"CHECK sparse decode graph, {kind}: relative error against the fp32 truth "
            f"{float((o - tr).norm() / tr.norm()):.4f}")


class DeqBench(Base):
    """Dequantizing n gathered int8 latent rows with two scales per row ([kv part, rope part]) -> bf16 [n, C].
    how: cat (scale each part, concatenate) | where | isel (per-dim scale by index_select) | lerp (s0 + (s1-s0)*m)
    | fix (scale all by s0, then the rope columns by s1/s0 and concatenate) | one (a single scale: the floor)
    | none (cast only)."""

    def __init__(self, reps, n, how):
        super().__init__(reps)
        self.n, self.how = n, how
        self.codes = torch.randint(-127, 128, (n, C), generator=g, dtype=torch.int8).to(dev)
        self.sc = (torch.rand(n, 2, generator=g) * 0.02 + 0.01).to(torch.bfloat16).to(dev)
        self.sc9 = (torch.rand(n, 9, generator=g) * 0.02 + 0.01).to(torch.bfloat16).to(dev)
        m = torch.zeros(C)
        m[512:] = 1
        self.maskb, self.maskf, self.idx = m.bool().to(dev), m.to(torch.bfloat16).to(dev), m.long().to(dev)

    def one(self, q, w):
        cf = self.codes.to(torch.bfloat16)
        sc = self.sc + q[0, 0, :1]
        s0, s1 = sc[:, :1], sc[:, 1:]
        how = self.how
        if how == "cat":
            y = torch.cat((cf[:, :512] * s0, cf[:, 512:] * s1), -1)
        elif how == "where":
            y = cf * torch.where(self.maskb, s1, s0)
        elif how == "isel":
            y = cf * sc.index_select(-1, self.idx)
        elif how == "lerp":
            y = cf * (s0 + (s1 - s0) * self.maskf)
        elif how == "fix":
            y = cf * s0
            y = torch.cat((y[:, :512], y[:, 512:] * (s1.float() / s0.float()).to(y.dtype)), -1)
        elif how == "group":       # one scale per 64 dims, three-dimensional broadcast
            y = (cf.view(self.n, 9, 64) * (self.sc9 + q[0, 0, :1]).unsqueeze(-1)).view(self.n, C)
        elif how == "group2d":     # the same with every group as a row (what LatentInt8Cache does)
            y = (cf.view(self.n * 9, 64) * (self.sc9 + q[0, 0, :1]).view(self.n * 9, 1)).view(self.n, C)
        elif how == "one":
            y = cf * s0
        else:
            y = cf
        return y.float().sum(0, keepdim=True)[:, :8]


class Empty(Base):
    def one(self, q, w):
        return q[:, 0, :4] * 2


def spec(name):
    """'scan:g=8:tail=split:gm=pair:stop=local', 'tk:rows=24:n=16384[:k=2048]', 'q8:how=fold' -> a variant factory."""
    kind, *kv = name.split(":")
    kw = dict(x.split("=", 1) for x in kv)
    if kind == "scan":
        return lambda r: Scan(r, **kw)
    if kind == "tk":
        return lambda r: TopkFull(r, int(kw["rows"]), int(kw["n"]), k=int(kw.get("k", TOPK)), m=int(kw.get("m", -1)),
                                  op=kw.get("op", "topk"),
                                  dtype=torch.bfloat16 if kw.get("dt") == "bf16" else torch.float32)
    if kind == "pscore":
        return lambda r: PScore(r, int(kw["rows"]), int(kw["n"]), kw.get("mode", "chunk"), hc=int(kw.get("hc", 2)),
                                sub=int(kw.get("sub", 32)), out=kw.get("out", "bf16"))
    if kind == "q8":
        return lambda r: AttnQ8(r, kw.get("how", "fold"))
    if kind == "deq":
        return lambda r: DeqBench(r, int(kw["n"]), kw.get("how", "cat"))
    raise KeyError(name)


ALL = {
    "empty": lambda r: Empty(r),
    "gather_score": lambda r: GatherScan(r, stop="score"),
    "gather_local": lambda r: GatherScan(r, stop="local"),
    "gather_full": lambda r: GatherScan(r),
    "pool_gemm": lambda r: PoolScan(r, stop="gemm"),
    "pool_gemm_mm": lambda r: PoolScan(r, stop="gemm", fc=False),
    "pool_score": lambda r: PoolScan(r, stop="score"),
    "pool_local": lambda r: PoolScan(r, stop="local"),
    "pool_local_1p": lambda r: PoolScan(r, stop="local", m=0),
    "pool_local_m8k": lambda r: PoolScan(r, stop="local", m=8192),
    "pool_local_m4k": lambda r: PoolScan(r, stop="local", m=4096),
    "pool_full": lambda r: PoolScan(r),
    "pool_full_gather": lambda r: PoolScan(r, x="gather"),
    "pool_full_nox": lambda r: PoolScan(r, x="none"),
    "topk_90k_2p": lambda r: TopkOnly(r, NLOC, 16384),
    "topk_90k_1p": lambda r: TopkOnly(r, NLOC, -1),
    "topk_16k": lambda r: TopkOnly(r, 16384, -1),
    "topk_16k_unsorted": lambda r: TopkOnly(r, 16384, -1, sorted_=False),
    "topk_16k_12rows": lambda r: TopkOnly(r, 16384, -1, rows=12),
    "topk_4k_44rows": lambda r: TopkOnly(r, 4096, -1, rows=44),
    "tk_2x90k_2p": lambda r: TopkFull(r, 2, NLOC, m=16384),
    "tk_2x90k": lambda r: TopkFull(r, 2, NLOC),
    "tk_2x16k": lambda r: TopkFull(r, 2, 16384),
    "tk_2x12k": lambda r: TopkFull(r, 2, 12288),
    "tk_2x4k": lambda r: TopkFull(r, 2, 4096),
    "tk_12x16k": lambda r: TopkFull(r, 12, 16384),
    "tk_24x8k": lambda r: TopkFull(r, 24, 8192),
    "tk_48x4k": lambda r: TopkFull(r, 48, 4096),
    "tk_6x4k": lambda r: TopkFull(r, 6, 4096),
    "tk_2x16k_k512": lambda r: TopkFull(r, 2, 16384, k=512),
    "tk_2x90k_k256": lambda r: TopkFull(r, 2, NLOC, k=256),
    "sort_2x16k": lambda r: TopkFull(r, 2, 16384, op="sort"),
    "sort_2x4k": lambda r: TopkFull(r, 2, 4096, op="sort"),
    "bisect20_2x90k": lambda r: TopkFull(r, 2, NLOC, m=20, op="bisect"),
    "bisect20_2x16k": lambda r: TopkFull(r, 2, 16384, m=20, op="bisect"),
    "multi3_2x90k": lambda r: TopkFull(r, 2, NLOC, m=3, op="multi"),
    "multi3_2x16k": lambda r: TopkFull(r, 2, 16384, m=3, op="multi"),
    "cumsum_2x90k": lambda r: TopkFull(r, 2, NLOC, op="cumsum"),
    "tiny20": lambda r: Tiny(r),
    "gemm1": lambda r: Gemm(r, stage=1),
    "gemm2": lambda r: Gemm(r, stage=2),
    "gemm3": lambda r: Gemm(r, stage=3),
    "tk_4x45k": lambda r: TopkFull(r, 4, NLOC // 2),
    "tk_4x45k_2p": lambda r: TopkFull(r, 4, NLOC // 2, m=16384),
    "tk_8x22k": lambda r: TopkFull(r, 8, NLOC // 4),
    "tk_16x11k": lambda r: TopkFull(r, 16, NLOC // 8),
    "tk_4x8k": lambda r: TopkFull(r, 4, 8192),
    "tk_8x4k": lambda r: TopkFull(r, 8, 4096),
    "tk_4x4k": lambda r: TopkFull(r, 4, 4096),
    "tk_2x8k": lambda r: TopkFull(r, 2, 8192),
    "tk_24x8k_b": lambda r: TopkFull(r, 24, 8192),
    "tk_24x4k": lambda r: TopkFull(r, 24, 4096),
    "bf_3x90k_2p": lambda r: TopkFull(r, 3, NLOC, m=16384, dtype=torch.bfloat16),
    "f32_3x90k_2p": lambda r: TopkFull(r, 3, NLOC, m=16384),
    "bf_18x16k": lambda r: TopkFull(r, 18, 16384, dtype=torch.bfloat16),
    "f32_18x16k": lambda r: TopkFull(r, 18, 16384),
    "bf_3x16k": lambda r: TopkFull(r, 3, 16384, dtype=torch.bfloat16),
    "f32_3x16k": lambda r: TopkFull(r, 3, 16384),
    "bf_6x8k": lambda r: TopkFull(r, 6, 8192, dtype=torch.bfloat16),
    "f32_6x8k": lambda r: TopkFull(r, 6, 8192),
    "bf_3x4k": lambda r: TopkFull(r, 3, 4096, dtype=torch.bfloat16),
    "i32_3x16k": lambda r: TopkFull(r, 3, 16384, dtype=torch.int32),
    "i32_18x16k": lambda r: TopkFull(r, 18, 16384, dtype=torch.int32),
    "i16_3x16k": lambda r: TopkFull(r, 3, 16384, dtype=torch.int16),
    "f32_bisect20_3x90k": lambda r: TopkFull(r, 3, NLOC, m=20, op="bisect"),
    "bf_bisect16_3x90k": lambda r: TopkFull(r, 3, NLOC, m=16, op="bisect", dtype=torch.bfloat16),
    "f32_cumsum_3x90k": lambda r: TopkFull(r, 3, NLOC, op="cumsum"),
    "sel_c2048": lambda r: Sel(r, 2048),
    "sel_c4096": lambda r: Sel(r, 4096),
    "sel_c8192": lambda r: Sel(r, 8192),
    "sel_c4096_11k": lambda r: Sel(r, 4096, n=11264),
    "sel_c4096_23k": lambda r: Sel(r, 4096, n=22528),
    "sel_c4096_45k": lambda r: Sel(r, 4096, n=45056),
    "merge8_topk": lambda r: Merge8(r, "topk"),
    "merge8_pair": lambda r: Merge8(r, "pair"),
    "v3_score": lambda r: PoolV3(r, stop="score"),
    "v3_score_t": lambda r: PoolV3(r, stop="score", layout="t"),
    "v3_score_tv": lambda r: PoolV3(r, stop="score", layout="tv"),
    "tk_9x32k": lambda r: TopkFull(r, 9, 32768),
    "tk_6x45k": lambda r: TopkFull(r, 6, NLOC // 2),
    "tk_3x8k": lambda r: TopkFull(r, 3, 8192),
    "tk_3x12k": lambda r: TopkFull(r, 3, 12288),
    "v3_local": lambda r: PoolV3(r, stop="local"),
    "v3_local_t": lambda r: PoolV3(r, stop="local", layout="t"),
    "v3_full": lambda r: PoolV3(r),
    "v3_full_t": lambda r: PoolV3(r, layout="t"),
    "v3_full_t_c2048": lambda r: PoolV3(r, layout="t", chunk=2048),
    "v3_full_t_c8192": lambda r: PoolV3(r, layout="t", chunk=8192),
    "x_reduce": lambda r: ExchangeOnly(r, x="reduce"),
    "x_gather": lambda r: ExchangeOnly(r, x="gather"),
    "x_small_reduce": lambda r: SmallReduce(r),
    "attn": lambda r: SparseAttn(r),
    "attn_gather": lambda r: SparseAttn(r, stop="gather"),
}


SYNC = os.environ.get("SYNC", "0") == "1"   # read every call's result (some graphs time as if they never ran otherwise)


def timeit(mod, n):
    out = None
    for _ in range(max(10, n // 10)):
        out = mod(q0, w0)
    out.item()
    t0 = time.perf_counter()
    for _ in range(n):
        out = mod(q0, w0)
        if SYNC:
            out.item()
    out.item()
    return (time.perf_counter() - t0) / n


log(f"world {world}, shards {SHARDS}, pool {BLOCKS} blocks, {NLOC} keys per rank, rows {R}, reps {REPS}, iters {ITERS}")
if os.environ.get("Q8CHECK") == "1" and rank == 0:
    try:
        q8_check()
    except Exception as e:
        log(f"CHECK FAILED {type(e).__name__}: {str(e)[:400]}")
if os.environ.get("Q8TIME") == "1" and rank == 0:
    q8_time()
for name in VARIANTS or list(ALL):
    try:
        make = ALL[name] if name in ALL else spec(name)
        t1 = timeit(ht.hpu.wrap_in_hpu_graph(make(1), disable_tensor_cache=True), ITERS)
        tn = timeit(ht.hpu.wrap_in_hpu_graph(make(REPS), disable_tensor_cache=True), ITERS)
        log(f"RESULT {name:20s} one call {t1 * 1e6:8.1f} us   {REPS} chained reps {tn * 1e6:8.1f} us   "
            f"slope {(tn - t1) / (REPS - 1) * 1e6:8.1f} us   total/reps {tn / REPS * 1e6:8.1f} us")
    except Exception as e:  # keep going: a failed variant is a result too
        log(f"RESULT {name:20s} FAILED {type(e).__name__}: {str(e)[:300]}")
        if world > 1:
            sys.exit(1)  # the ranks would be out of step
log("done")
