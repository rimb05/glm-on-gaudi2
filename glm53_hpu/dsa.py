# SPDX-License-Identifier: Apache-2.0
"""DeepSeek sparse attention (DSA) for GLM-5.x on vllm-gaudi's HPU MLA path: long-context revision.

Derived from the gaudi-cookbook's ports/glm-5-mxfp4/glm5_hpu/glm5_hpu/dsa.py (Apache-2.0, thediymaker/gaudi-cookbook
at 1540263). patches/dsa_override.py installs this file over the image's glm5_hpu/dsa.py when GLM_DSA_V2=1.
Changes against the original (each has its own switch, see the constants below):
  * the indexer keys live in vllm-gaudi's own BF16 indexer-key caches, which are part of the KV pool and otherwise
    unused, instead of a second set of buffers allocated after memory profiling (GLM_DSA_REUSE_NATIVE);
  * the draft model's MTP layer gets the sparse path too (it owns an indexer; the original left it dense);
  * decode: the indexer scan is split across the TP ranks: every rank scores 1/world of the keys with one GEMM,
    keeps its local top-k, and the candidates are exchanged with one all-reduce per indexer layer (GLM_DSA_SHARD);
    every rank holds the full latent cache, so only the scan is split. When the rows' contexts fill most of the
    key pool the rank scores its physical 1/world of the pool in place, without gathering (GLM_DSA_POOL_FRAC);
  * prefill above 131,072 keys: the row groups whose key unions are gathered shrink from 64 to 32 or 16 rows, so
    every layer takes the gather path at any context length (the dense-masked path costs 20 s per chunk there);
    the indexer scores take all heads of a sub-block of rows through one product (GLM_DSA_PREFILL_SCORE), and the
    scoring and the threshold bisection are cut into several lazy graphs so their intermediates are not all
    alive at once;
  * prefill, dense-masked layers (still used below 131,072 keys): K/V are decompressed for a few heads at a time
    and the row blocks shrink with the key count (GLM_DSA_KV_BUDGET_GB, GLM_DSA_MASK_BUDGET_GB);
  * a layer that attends densely ties the indexer keys it wrote to its output, so a compiled graph cannot drop
    the write (the keys are needed once the context is long enough for the indexer);
  * GLM_DSA_TRACE=1 prints where a prefill's time goes (adds device syncs; for measurements only);
  * GLM_KV_FP8=int8: the latent cache holds linear 8-bit codes with one scale per token for every 64 dimensions
    instead of bf16, half the bytes per token (LatentInt8Cache; glm53_hpu/kvfp8.py makes vllm-gaudi allocate the
    8-bit tensors). This is what a 1M-token pool needs on 16 cards. It changes the model's numbers slightly:
    every latent read back is the 8-bit rounding of what was written (0.6 % error). GLM_KV_FP8=fp8 selects E4M3
    codes with one scale per token (LatentFP8Cache), which measured four times that error.

Original description:

GLM-5.x is trained with DSA: a lightning indexer scores every earlier token and each query attends only to its top
``index_topk`` (2048) keys. vllm-gaudi serves these layers as dense MLA (``patches/glm_dsa.py`` turns the native
sparse path off, it does not run under this configuration). Dense attention is exact up to 2048 keys and an
approximation beyond. With the accurate FusedSDPA softmax (``patches/fsdpa_softmax_mode.py``) dense attention
passes the long-context needles to 128K; this module adds the real top-k selection and makes long contexts faster.

Switch: ``GLM_DSA_SPARSE=1`` (off: dense MLA everywhere). Below :data:`MIN_KEYS` keys per sequence attention stays
dense with the accurate softmax; above it:

* Prefill indexer: each TP rank scores 1/world of the query rows with fp32 scores, finds every row's
  ``index_topk``-th score by bisection (reproduces the exact top-k selection on GLM-5.3 scores), and the
  bit-packed selections are all-gathered, so all ranks attend over the same keys.
* Prefill attention, decided per indexer group (one full indexer layer and the shared layers after it): when 64
  neighbouring queries select overlapping keys (early layers), each block gathers the union of its keys from the
  latent cache and runs absorbed-MLA FusedSDPA on it; when selections are scattered (the union approaches the whole
  context) the selection is a mask on dense FusedSDPA with causal row-block trimming, which is then cheaper.
* Decode: each sequence gathers exactly its ``index_topk`` latent rows (fp32 softmax), so decode cost no longer
  grows with context. Steps with several query rows per sequence and batched prefills use the selection as a mask
  on the dense paths instead.

The indexer keys live in one bf16 buffer per full indexer layer, sized like the MLA cache and allocated after vLLM's
memory profiling, so serve with ``--gpu-memory-utilization 0.86`` and keep prefills with more than 2048 tokens out
of HPU graphs (``--max-cudagraph-capture-size 2048``).
"""

import math
import os

import torch

ENABLED = os.environ.get("GLM_DSA_SPARSE", "0") == "1"
TAG = "[glm-dsa]"


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_gb(name, default):
    try:
        return int(float(os.environ.get(name, default)) * 2**30)
    except ValueError:
        return int(default * 2**30)


REUSE_NATIVE = os.environ.get("GLM_DSA_REUSE_NATIVE", "1") == "1"
KV_FP8 = os.environ.get("GLM_KV_FP8", "0") in ("1", "fp8", "int8")
Q8_DEBUG = os.environ.get("GLM_KV_Q8_DEBUG", "0") == "1"  # see LatentInt8Cache.enable_debug
SHARD = os.environ.get("GLM_DSA_SHARD", "1") == "1"
TRACE = os.environ.get("GLM_DSA_TRACE", "0") == "1"
# Decode goes sparse above this many keys per row (the prefill threshold is MIN_KEYS below).
MIN_KEYS_DECODE = _env_int("GLM_DSA_MIN_KEYS_DECODE", 32768)
# Decode scan over the whole key pool in place (no gather, every key scored once for all rows) when the rows'
# contexts together fill at least this fraction of the pool; below it the rows' own blocks are gathered.
# 0 = always the pool scan, above 1 = never.
try:
    POOL_FRAC = float(os.environ.get("GLM_DSA_POOL_FRAC", "0.4"))
except ValueError:
    POOL_FRAC = 0.4
# Dense-masked prefill: bytes of decompressed K/V per head group, and of one row block's additive mask.
KV_BUDGET_BYTES = _env_gb("GLM_DSA_KV_BUDGET_GB", 3.0)
MASK_BUDGET_BYTES = _env_gb("GLM_DSA_MASK_BUDGET_GB", 0.75)

NEG = -1.0e30
# Attention stays dense (exact up to index_topk keys, and fast) until a prefill chunk (context + chunk, bucket
# padded) or a decode batch (per sequence) has more keys than this.
MIN_KEYS = _env_int("GLM_DSA_MIN_KEYS", 32768)
# Scratch for the indexer scores of one (row block, head group): rows x heads x keys x 2 bytes. Not seen by vLLM's
# memory profiling, so keep it small.
ROW_BUDGET_BYTES = 1 << 30
# Row blocks shrink to 512 before heads are split: top-k uses the TPCs badly on fewer rows.
MAX_ROWS = 2048
# Queries per gathered key set in the sparse prefill attention.
UNION_ROWS = 64
# ... and the smaller groups tried above FULLMASK_MAX_KEYS keys when the union of UNION_ROWS rows is too large.
SMALL_UNION_ROWS = (32, 16)
# Gather a layer group only when its padded union is at most this fraction of the keys, and at most GATHER_MAX_UCAP.
GATHER_MAX_FRAC = 0.5
GATHER_MAX_UCAP = 32768
BISECT_ITERS = 20
# prefill indexer scores: rows (all heads of a sub-block of rows in one product) | chunk (head groups, fp32 sum)
PREFILL_SCORE = os.environ.get("GLM_DSA_PREFILL_SCORE", "rows")
PREFILL_PARTS = max(1, _env_int("GLM_DSA_PREFILL_PARTS", 2))  # partial head sums added in fp32 (1: bf16 scores)
SCORE_BUDGET_BYTES = 3 << 29  # the first product of a sub-block: rows x heads x keys bf16
# prefill row threshold: bisect (20 halvings over [rows, keys]) | topk (one top-k per row) | topk2 (two passes)
PREFILL_KTH = os.environ.get("GLM_DSA_PREFILL_KTH", "bisect")
# Above this many scores per row block, the bisection and the per-head-group scoring are cut into several graphs.
BISECT_CUT_ELEMS = 48 << 20
BISECT_CUT_1 = 96 << 20  # above this many scores per row block the bisection is cut after every halving
BIG_KV = 196608
HUGE_KV = 655360  # 128-row indexer blocks above this many keys (1M: both stages peaked 1 GiB below the device limit)
# FusedSDPA softmax mode of the sparse prefill attention: "fast" is badly wrong for some rows of deep GLM layers.
SOFTMAX_MODE = "None"
# Gathered latent rows per FusedSDPA call are capped (one 9.7 GB gather hung all eight cards into the driver's 600 s
# "no progress" timeout).
ATTN_BUDGET_BYTES = 512 << 20
DENSE_ROW_BLOCKS = 4
# Above this many keys a dense-masked prefill keeps the selection bit-packed and builds each row block's mask on
# the fly (a [8192, 262144] bf16 mask is 4.3 GB).
FULLMASK_MAX_KEYS = 131072

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA as _FSDPA
except ImportError:  # CPU tests
    _FSDPA = None

_STATE = {"sel": None, "active": None}
_INFO = {"topk": None, "ready": False}
_ORIG_FORWARD_IMPL = None


def _graph_break():
    """Lazy mode: cut the graph around every row block of the prefill indexer. Unrolled into one graph per step,
    the row blocks of 21 layers made graphs whose first compile ran into the 600 s timeout. Never inside an HPU
    graph capture."""
    try:
        import habana_frameworks.torch as ht
        import habana_frameworks.torch.core as htcore
    except ImportError:
        return
    if ht.utils.internal.is_lazy() and not ht.hpu.is_current_stream_capturing():
        htcore.mark_step()


_TR = {"t": None, "acc": {}, "n": {}, "calls": 0}


def _tick(label=None):
    """GLM_DSA_TRACE: close the section that started at the previous tick and charge it to `label` (None: only
    restart the clock). Forces the pending lazy graph to run, so it changes timing; never inside a graph capture."""
    if not TRACE:
        return
    import time

    try:
        import habana_frameworks.torch as ht
        import habana_frameworks.torch.core as htcore

        if ht.hpu.is_current_stream_capturing():
            return
        htcore.mark_step()
        torch.hpu.synchronize()
    except ImportError:
        pass
    now = time.perf_counter()
    if label is not None and _TR["t"] is not None:
        _TR["acc"][label] = _TR["acc"].get(label, 0.0) + (now - _TR["t"])
        _TR["n"][label] = _TR["n"].get(label, 0) + 1
    _TR["t"] = now


def _trace_report(tag):
    if not TRACE or not _TR["acc"]:
        return
    try:
        if _tp()[1] != 0:  # one line per stage: tensor-parallel rank 0 only
            _TR["acc"].clear()
            _TR["n"].clear()
            return
    except Exception:
        pass
    tot = sum(_TR["acc"].values())
    parts = ", ".join(f"{k} {v * 1e3:.0f} ms/{_TR['n'][k]}" for k, v in sorted(_TR["acc"].items(), key=lambda x: -x[1]))
    print(f"{TAG} trace {tag}: {tot * 1e3:.0f} ms in attention paths: {parts}", flush=True)
    if _TR.get("u"):
        print(f"{TAG} trace {tag}: largest union of {UNION_ROWS} rows > bucket, per full indexer layer: "
              f"{' '.join(_TR['u'])}", flush=True)
        _TR["u"] = []
    _TR["acc"].clear()
    _TR["n"].clear()


# ---------------------------------------------------------------------------------------------------------------------
# Core: plain torch, no vLLM imports (tests/test_dsa_core.py runs it on the CPU)


def kth_largest(s, k, m=16384):
    """k-th largest value per row of s [R, N]. Two top-k passes over groups of m are ~3x faster on Gaudi 2 than one
    over the full row, and exact: every global top-k element is inside its group's top-k."""
    r, n = s.shape
    if n <= 2 * m:
        return torch.topk(s, k, dim=-1).values[:, -1:]
    pad = (-n) % m
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=-float("inf"))
    c = torch.topk(s.view(r, -1, m), k, dim=-1).values.reshape(r, -1)
    return torch.topk(c, k, dim=-1).values[:, -1:]


def topk_indices(s, k, m=16384):
    """Exact top-k (values, indices) per row of s [R, N], two passes for long rows."""
    r, n = s.shape
    if n <= 2 * m:
        return torch.topk(s, k, dim=-1)
    pad = (-n) % m
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=-float("inf"))
    g = (n + pad) // m
    v1, i1 = torch.topk(s.view(r, g, m), k, dim=-1)
    i1 = i1 + (torch.arange(g, device=s.device, dtype=i1.dtype) * m).view(1, g, 1)
    v2, i2 = torch.topk(v1.reshape(r, g * k), k, dim=-1)
    return v2, torch.gather(i1.reshape(r, g * k), 1, i2)


def prefill_mask_bias(q_idx, w, k_all, attn_bias, topk):
    """Selection as a mask (batched prefill). q_idx [B,Q,Hi,D], w [B,Q,Hi], k_all [B,KV,D], attn_bias [B,1,Q,KV]
    (0 / -inf). Returns attn_bias with every visible key outside the query's top-k set to NEG."""
    b_, q_, hi, d = q_idx.shape
    kv = k_all.shape[1]
    if kv <= topk:
        return attn_bias
    rows = min(q_, MAX_ROWS)
    while rows > 512 and rows * hi * kv * 2 > ROW_BUDGET_BYTES:
        rows //= 2
    hc = max(1, min(hi, ROW_BUDGET_BYTES // (rows * kv * 2)))
    neg = torch.tensor(NEG, dtype=attn_bias.dtype, device=attn_bias.device)
    out = []
    _graph_break()
    for b in range(b_):
        kt = k_all[b].transpose(0, 1)
        parts = []
        for r0 in range(0, q_, rows):
            r1 = min(q_, r0 + rows)
            n = r1 - r0
            acc = None
            for h0 in range(0, hi, hc):
                h1 = min(hi, h0 + hc)
                s = torch.matmul(q_idx[b, r0:r1, h0:h1].reshape(n * (h1 - h0), d), kt)
                s = torch.relu_(s).view(n, h1 - h0, kv)
                s = torch.bmm(w[b, r0:r1, h0:h1].to(s.dtype).unsqueeze(1), s).squeeze(1).float()
                acc = s if acc is None else acc + s
            bias_r = attn_bias[b, 0, r0:r1]
            s = acc.to(bias_r.dtype) + bias_r
            parts.append(torch.where(s >= kth_largest(s, topk), bias_r, neg))
            _graph_break()
        out.append(torch.cat(parts, 0) if len(parts) > 1 else parts[0])
    res = torch.stack(out, 0).unsqueeze(1)
    _graph_break()
    return res


def decode_mask_bias(q_idx, w, k_blocks, block_groups, block_bias, topk):
    """Selection as a block mask (decode with several query rows per sequence). q_idx [bs,Hi,D], w [bs,Hi],
    k_blocks [nb,T,D], block_groups [nb] (row id, -1 padding), block_bias [nb,T] (0 / -inf)."""
    bs = q_idx.shape[0]
    nb, t, _ = k_blocks.shape
    if nb * t <= topk:
        return block_bias
    g = block_groups.to(torch.long)
    valid = (g >= 0) & (g < bs)
    gc = g.clamp(0, bs - 1)
    s = torch.relu(torch.bmm(q_idx.index_select(0, gc), k_blocks.transpose(1, 2)))
    s = torch.bmm(w.index_select(0, gc).to(s.dtype).unsqueeze(1), s).squeeze(1).float()
    s = s + block_bias.float()
    s = torch.where(valid.unsqueeze(-1), s, -float("inf"))
    seq = torch.arange(bs, device=s.device, dtype=torch.long)
    member = (gc.unsqueeze(0) == seq.unsqueeze(1)) & valid.unsqueeze(0)
    dense = torch.where(member.unsqueeze(-1), s.unsqueeze(0), -float("inf"))
    thr = kth_largest(dense.reshape(bs, nb * t), topk)[:, 0]
    thr_b = thr.index_select(0, gc).unsqueeze(-1)
    # Padding blocks keep their bias: flat paged attention rescales each block by its group's sum, a padding block's
    # group sum is 0, and a fully masked padding block would compute 0/0 and poison the batch.
    keep = torch.where((s >= thr_b) | (~valid).unsqueeze(-1), 0.0, NEG)
    return (block_bias.float() + keep).to(block_bias.dtype)


def causal_bias(bsz, q_len, seq_lens, dtype, device):
    """The no-context prompt bias vllm-gaudi builds when FusedSDPA is not used."""
    len_mask = (
        torch.arange(0, q_len, device=device, dtype=torch.int32)
        .view(1, q_len)
        .ge(seq_lens.unsqueeze(-1))
        .view(bsz, 1, 1, q_len)
    )
    causal = torch.triu(torch.ones((bsz, 1, q_len, q_len), device=device, dtype=torch.bool), diagonal=1)
    mask = causal.logical_or(len_mask)
    return torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf)


def prefill_visible(r0, r1, kv, cb, ctx_len, seq_len, device):
    """Visibility [R, KV] of prefill rows r0..r1 of one sequence in the HPU prefill key layout: cb context slots
    (valid below ctx_len), then the chunk (causal, valid below seq_len)."""
    j = torch.arange(kv, device=device, dtype=torch.int32).view(1, kv)
    r = torch.arange(r0, r1, device=device, dtype=torch.int32).view(-1, 1)
    jc = j - cb
    return (j < ctx_len) | ((jc >= 0) & (jc <= r) & (jc < seq_len))


def bisect_threshold(acc, ok, k, iters=BISECT_ITERS):
    """Per-row threshold t with count(acc >= t) >= k, as tight as `iters` halvings allow; -inf when a row has at
    most k valid keys (keep them all). On large inputs the lazy graph is cut every few halvings: left in one
    graph, the [rows, keys] comparison of every halving can be alive at once."""
    big = torch.tensor(float("inf"), device=acc.device, dtype=acc.dtype)
    lo = torch.where(ok, acc, big).amin(1, keepdim=True)
    hi = torch.where(ok, acc, -big).amax(1, keepdim=True)
    nvalid = ok.sum(1, keepdim=True)
    cut = acc.numel() > BISECT_CUT_ELEMS
    # the count of a [rows, keys] comparison goes through an integer copy of it (int64 by default: 2 GiB for 256
    # rows x 1M keys); on the largest inputs one halving per graph
    every = 1 if acc.numel() > BISECT_CUT_1 else 4
    if cut:
        _graph_break()
    for it in range(iters):
        mid = (lo + hi) * 0.5
        ge = (acc >= mid).sum(1, keepdim=True, dtype=torch.int32) >= k
        lo = torch.where(ge, mid, lo)
        hi = torch.where(ge, hi, mid)
        if cut and it % every == every - 1:
            _graph_break()
    return torch.where(nvalid <= k, -big, lo)


def kth_threshold(acc, ok, k):
    """Per-row threshold for the prefill selection: the k-th largest score (-inf when a row has fewer than k valid
    keys: acc is -inf on the others), or the bisection's bound (GLM_DSA_PREFILL_KTH=bisect)."""
    if PREFILL_KTH == "bisect":
        return bisect_threshold(acc, ok, k)
    if acc.shape[1] <= k:
        return torch.full((acc.shape[0], 1), -float("inf"), dtype=acc.dtype, device=acc.device)
    if PREFILL_KTH == "topk2":
        return kth_largest(acc, k)
    return torch.topk(acc, k, dim=-1).values[:, -1:]


def prefill_scores(q, w, kt):
    """Indexer scores [n, KV] fp32 of n query rows: sum over the heads of w * relu(q . k), q [n,Hi,D], w [n,Hi],
    kt [D,KV]. Every sub-block of rows takes all its heads through one product; the head sum comes out of a second
    product as PREFILL_PARTS partial sums per row (bf16), which are added in fp32. Measured on Gaudi 2 for 256 rows
    x 725K keys: 40 ms, against 75 ms for products over 2 heads at a time with an fp32 running sum over the 16
    head groups (the elementwise passes over [rows, keys] fp32 are what costs there)."""
    n, hi, d = q.shape
    kv = kt.shape[1]
    sub = n
    while sub > 8 and sub % 2 == 0 and sub * hi * kv * 2 > SCORE_BUDGET_BYTES:
        sub //= 2
    parts = PREFILL_PARTS if hi % PREFILL_PARTS == 0 else 1
    eye = torch.eye(parts, device=q.device, dtype=q.dtype).view(1, parts, parts, 1)
    outs = []
    for a in range(0, n, sub):
        b = min(n, a + sub)
        m = b - a
        lg = torch.relu(torch.matmul(q[a:b].reshape(m * hi, d), kt)).view(m, hi, kv)
        # [m, parts, Hi]: row p holds the weights of the heads of part p, zeros elsewhere
        wm = (w[a:b].to(lg.dtype).view(m, 1, parts, hi // parts) * eye).reshape(m, parts, hi)
        s = torch.bmm(wm, lg)
        outs.append(s.float().sum(1) if parts > 1 else s.squeeze(1).float())
        if sub < n and n * kv > BISECT_CUT_ELEMS:
            _graph_break()  # one sub-block's products at a time: in one graph they can all be alive at once
    return torch.cat(outs, 0) if len(outs) > 1 else outs[0]


def prefill_select(q_idx, w, kt, r0, r1, cb, ctx_len, seq_len, topk, rr):
    """Indexer selection for prefill rows r0..r1 of one sequence. q_idx [Q,Hi,D], w [Q,Hi], kt [D,KV]. Returns
    sel [n, KV] (row keeps key) and u [n/rr, KV] (union over each block of rr rows)."""
    _, hi, d = q_idx.shape
    kv = kt.shape[1]
    rows = min(r1 - r0, MAX_ROWS)
    # 256-row blocks above BIG_KV keys: the fp32 scores of a block and their copies in the bisection are several
    # times rows x keys x 4 bytes (stage 0 peaked 1 GiB below the device limit at 700K keys with 512 rows)
    min_rows = 512 if kv <= BIG_KV else (256 if kv <= HUGE_KV else 128)
    while rows > min_rows and rows * hi * kv * 2 > ROW_BUDGET_BYTES:
        rows //= 2
    rows = max(rr, rows - rows % rr)
    hc = max(1, min(hi, ROW_BUDGET_BYTES // (rows * kv * 2)))
    sels, us = [], []
    _graph_break()
    for a in range(r0, r1, rows):
        b = min(r1, a + rows)
        n = b - a
        acc = None
        if PREFILL_SCORE == "rows":
            acc = prefill_scores(q_idx[a:b], w[a:b], kt)
        for h0 in range(0, hi if acc is None else 0, hc):
            h1 = min(hi, h0 + hc)
            s = torch.matmul(q_idx[a:b, h0:h1].reshape(n * (h1 - h0), d), kt)
            s = torch.relu_(s).view(n, h1 - h0, kv)
            s = torch.bmm(w[a:b, h0:h1].to(s.dtype).unsqueeze(1), s).squeeze(1).float()
            acc = s if acc is None else acc + s
            if hc < hi and n * kv > BISECT_CUT_ELEMS:
                _graph_break()  # one head group's scores at a time: in one graph they can all be alive at once
        ok = prefill_visible(a, b, kv, cb, ctx_len, seq_len, acc.device)
        row = torch.arange(a, b, device=acc.device, dtype=torch.int32).view(-1, 1)
        ok = ok & (row < seq_len)  # bucket-padding rows select nothing
        acc = torch.where(ok, acc, -float("inf"))  # invisible keys never count toward the top-k
        sel = (acc >= kth_threshold(acc, ok, topk)) & ok
        sels.append(sel)
        us.append(sel.view(n // rr, rr, kv).any(1))
        _graph_break()
    if len(sels) == 1:
        return sels[0], us[0]
    return torch.cat(sels, 0), torch.cat(us, 0)


def union_bucket(n, topk):
    """Static bucket for a union size: multiples of topk/2 up to 4*topk, then powers of 2."""
    step = topk // 2
    if n <= 4 * topk:
        return max(topk, -(-n // step) * step)
    c = 4 * topk
    while c < n:
        c *= 2
    return c


def compact(sel, u, rr, ucap, dtype):
    """Per block of rr rows: cl [nb, ucap] = its union keys in key order (padded with key 0), bias [nb*rr, ucap] =
    0 where the row selected that key, NEG elsewhere and on padding."""
    nb, kv = u.shape
    dev = u.device
    ui = u.to(torch.int32)
    cnt = ui.sum(1, keepdim=True)
    pos = torch.cumsum(ui, 1, dtype=torch.int32) - 1
    tgt = torch.where(u, pos, ucap).to(torch.long)
    src = torch.arange(kv, device=dev, dtype=torch.int32).view(1, kv).expand(nb, kv).contiguous()
    cl = torch.zeros(nb, ucap + 1, dtype=torch.int32, device=dev).scatter_(1, tgt, src)[:, :ucap].contiguous()
    g = torch.gather(sel.view(nb, rr, kv), 2, cl.to(torch.long).unsqueeze(1).expand(nb, rr, ucap))
    live = torch.arange(ucap, device=dev, dtype=torch.int32).view(1, 1, ucap) < cnt.view(nb, 1, 1)
    bias = torch.where(g & live, 0.0, NEG).to(dtype).view(nb * rr, ucap)
    return cl, bias


def pack_bits(sel):
    """[n, KV] bool -> [n, KV/8] uint8 (KV % 8 == 0)."""
    n, kv = sel.shape
    wts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.bfloat16, device=sel.device).view(8, 1)
    return torch.matmul(sel.reshape(n * kv // 8, 8).to(torch.bfloat16), wts).view(n, kv // 8).to(torch.uint8)


def unpack_bits(p, kv):
    wts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=p.device)
    return (torch.bitwise_and(p.unsqueeze(-1), wts) != 0).view(p.shape[0], kv)


def bits_to_bias(p, kend, dtype):
    """Bit-packed selection rows [n, KV/8] -> additive bias [n, kend] in `dtype` (0 keep / NEG drop), written in
    `dtype` directly: torch.where with Python scalars would go through fp32, twice the bytes of the result."""
    wts = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=p.device)
    drop = (torch.bitwise_and(p[:, : -(-kend // 8)].unsqueeze(-1), wts) == 0).view(p.shape[0], -1)[:, :kend]
    return drop.to(dtype) * torch.tensor(NEG, dtype=dtype, device=p.device)


class LatentFP8Cache(torch.nn.Module):
    """Drop-in for vllm-gaudi's VLLMKVCache on an MLA layer whose latent cache tensor is float8_e4m3fn: writes
    quantize each token's latent with its own scale (max |x| / FP8 max, the same dynamic quantization the FP8
    expert op applies to its inputs), reads return bf16. The scales live here, one bf16 value per slot.

    forward(input, cache, slot_mapping)   write [n, C] latents at slots
    fetch_from_cache(cache, blocks)       cache unflattened to [blocks, T, (1,) C]: gather whole blocks
    fetch_slots(cache, slots)             cache [slots, C]: gather single rows (the sparse paths)
    """

    def __init__(self, slots, device):
        super().__init__()
        self.scale = torch.ones(slots, 1, dtype=torch.bfloat16, device=device)

    @staticmethod
    def _dequant(codes, scale):
        return torch.ops.hpu.cast_from_fp8(codes, 1.0, torch.bfloat16) * scale

    def forward(self, input, cache, slot_mapping, *args, **kwargs):
        if input is not None:
            from vllm_gaudi.extension import ops as _ops

            codes, scale = _ops.dynamic_quant(input)
            cache.index_copy_(0, slot_mapping, codes)
            self.scale.index_copy_(0, slot_mapping, scale.to(self.scale.dtype))
        return cache

    def fetch_from_cache(self, cache, blocks, *args, **kwargs):
        codes = cache.index_select(0, blocks)
        sc = self.scale.unflatten(0, (cache.shape[0], -1)).index_select(0, blocks)  # [nb, T, 1]
        if codes.dim() == 4:
            sc = sc.unsqueeze(2)
        return self._dequant(codes, sc)

    def fetch_slots(self, cache, slots):
        return self._dequant(cache.index_select(0, slots), self.scale.index_select(0, slots))


class LatentInt8Cache(torch.nn.Module):
    """Like LatentFP8Cache, with linear 8-bit codes: round(x / scale) in -127..127, and one scale per token for every
    `group` (64) latent dimensions: 9 scales for GLM-5.3's 576 values (8 groups of the normalised compressed KV, one
    for the rotary part), each max |x| / 127 of its group, stored as bf16 [slots, 9]. The cache tensor is int8.

    Why groups. One scale per token does not work: in the model's first ten layers the rotary part of a latent is
    16 to 100 times larger than the compressed-KV part (measured with GLM_KV_Q8_DEBUG=1; the kv_a_layernorm gains
    there are 0.003-0.03), so the KV part rounds to zero. The relative error of the whole latent still looks fine
    (3 %) and a prefill-only check passes, while decode produces degenerate text. Two scales (KV part, rotary part)
    are correct, but applying them needs a per-dimension scale tensor (torch.where over a mask, or expand + concat:
    115 us per layer); scales per group of equal size apply with one broadcast product on a reshaped view, cost
    the same as a single scale and are a little more precise (error 0.5-0.6 % of a part's norm; E4M3 codes with one
    scale: 2.5-2.9 %, they keep 3 mantissa bits at any magnitude).

    forward(input, cache, slot_mapping)   write [n, C] latents at slots
    fetch_from_cache(cache, blocks)       cache unflattened to [blocks, T, (1,) C]: gather whole blocks
    fetch_slots(cache, slots)             cache [slots, C]: gather single rows (the sparse paths)

    Everything this module keeps between graphs is created in __init__: a tensor first created inside a compiled
    graph and reused by a later one has no storage ("Empty tensor optional" on its next use).

    What it costs in a decode step (tools/scan_bench.py Q8TIME=1, 24 chained layers): reading 8-bit codes is cheaper
    than reading bf16 (31 against 40 us per sparse layer); the write is what costs, about 38 us per layer: a dozen
    small operations (max, scale, divide, round, cast, two scatters), roughly 3 us of device time each whatever
    their size. Tried and not better: fewer operations (bf16 arithmetic, no clamp: 74 us per layer against 69);
    attending before writing, with the step's rows patched into the gathered latents (109 us) or added as extra
    score columns (84 us): the write is work, not only latency, and the patching adds to it."""

    def __init__(self, slots, device, split, width, group=64):
        super().__init__()
        self.split, self.width, self.group = int(split), int(width), int(group)
        assert self.width % self.group == 0 and self.split % self.group == 0, (split, width, group)
        self.ngroups = self.width // self.group
        self.scale = torch.ones(slots, self.ngroups, dtype=torch.bfloat16, device=device)

    def forward(self, input, cache, slot_mapping, *args, **kwargs):
        if input is not None:
            x = input.reshape(-1, self.group)  # every group a row: [n * G, group] against [n * G, 1] scales
            amax = x.abs().amax(dim=-1, keepdim=True).float()
            sc = ((amax + 1e-8) * (1.0 / 127.0)).to(self.scale.dtype)  # rounded to the stored precision before use
            codes = torch.round(x.float() / sc.float()).clamp(-127.0, 127.0)
            cache.index_copy_(0, slot_mapping, codes.to(cache.dtype).view(-1, *cache.shape[1:]))
            self.scale.index_copy_(0, slot_mapping, sc.view(-1, self.ngroups))
            if getattr(self, "shadow", None) is not None:
                self._debug_write(input, cache, slot_mapping)
        return cache

    def dequant(self, codes, sc):
        """codes [..., C] int8 and their scales sc [..., G] (same leading shape) -> bf16 latents [..., C]."""
        lead = codes.shape[:-1]
        # every group a row, [N, group] x [N, 1]: the plain two-dimensional broadcast (same cost as one scale per
        # token, which is how the 2K decode speed of the bf16 cache is kept)
        y = codes.reshape(-1, self.group).to(sc.dtype) * sc.reshape(-1, 1)
        return y.view(*lead, self.width)

    def fetch_from_cache(self, cache, blocks, *args, **kwargs):
        codes = cache.index_select(0, blocks)
        sc = self.scale.unflatten(0, (cache.shape[0], -1)).index_select(0, blocks)  # [nb, T, G]
        out = self.dequant(codes, sc)
        if getattr(self, "shadow", None) is not None:
            out = self._debug_fetch(out, cache, blocks)
        return out

    def fetch_slots(self, cache, slots):
        return self.dequant(cache.index_select(0, slots), self.scale.index_select(0, slots))

    def enable_debug(self, cache):
        """GLM_KV_Q8_DEBUG=1: keep the true latents of this layer in a bf16 shadow and track, inside the graphs, how
        far the block fetches of the dense paths are from them, separately for the compressed-KV part and the
        rotary part of the latent. dbg = [fetches, tokens, sum of kv errors, sum of rope errors, worst kv error,
        worst rope error, smallest rope/kv peak ratio of a decode write, largest]; printed whenever a new prefill
        shape is traced (the only place where Python runs and may sync). The counters feed the layer's output
        (times zero): the graph compiler drops updates nothing depends on."""
        self.shadow = torch.zeros(cache.shape[0], cache.shape[-1], dtype=self.scale.dtype, device=cache.device)
        self.dbg = torch.zeros(8, dtype=torch.float32, device=cache.device)
        self.dbg[6] = 1.0e9

    def _debug_fetch(self, out, cache, blocks):
        sp = self.split
        ref = self.shadow.unflatten(0, (cache.shape[0], -1)).index_select(0, blocks).view(out.shape).float()
        o = out.float()
        nk, nr = ref[..., :sp].pow(2).sum(-1), ref[..., sp:].pow(2).sum(-1)
        live = (nk > 1e-6).float()
        ek = ((o[..., :sp] - ref[..., :sp]).pow(2).sum(-1) / (nk + 1e-6)).sqrt() * live
        er = ((o[..., sp:] - ref[..., sp:]).pow(2).sum(-1) / (nr + 1e-6)).sqrt() * live
        d = self.dbg
        self.dbg.copy_(torch.stack((d[0] + 1, d[1] + live.sum(), d[2] + ek.sum(), d[3] + er.sum(),
                                    torch.maximum(d[4], ek.amax()), torch.maximum(d[5], er.amax()), d[6], d[7])))
        return out + (self.dbg.sum() * 0.0).to(out.dtype)

    def _debug_write(self, input, cache, slot_mapping):
        x = input.reshape(-1, input.shape[-1])
        self.shadow.index_copy_(0, slot_mapping, x.to(self.shadow.dtype))
        if x.shape[0] < 64:
            a = x.float().abs()
            ratio = a[:, self.split :].amax(-1) / (a[:, : self.split].amax(-1) + 1e-12)
            d = self.dbg
            self.dbg.copy_(torch.cat((d[:6], torch.minimum(d[6], ratio.amin()).view(1),
                                      torch.maximum(d[7], ratio.amax()).view(1))))
        else:
            v = self.dbg.cpu().tolist()
            n = max(v[1], 1.0)
            print(f"{TAG} q8-debug {getattr(self, 'tag', '?')}: {int(v[0])} block fetches, {int(v[1])} tokens; error of "
                  f"the kv part mean {v[2] / n:.4f} worst {v[4]:.4f}; of the rope part mean {v[3] / n:.4f} worst "
                  f"{v[5]:.4f}; rope/kv peak ratio of the decode writes {v[6]:.2f} to {v[7]:.2f}", flush=True)


def codec_report(x, split):
    """Round-trip error of 8-bit codes on latents x [n, C] (CPU, fp32): relative L2 error over all tokens of linear
    codes with one scale per 64 dims (what LatentInt8Cache stores), with two scales (KV part, rotary part), with one
    scale, and of E4M3 codes with one scale (LatentFP8Cache); plus how peaked the two parts are (mean max|x| / rms)
    and their size. A small error of the whole latent says nothing about a part that is much smaller than the other."""
    def lin(v):
        sc = v.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
        return torch.round(v / sc).clamp(-127, 127) * sc

    def rel(y):
        return float((y - x).norm() / x.norm().clamp(min=1e-12))

    a, b = x[:, :split], x[:, split:]
    two = torch.cat((lin(a), lin(b)), -1)
    out = f"n {x.shape[0]}  int8x2 {rel(two):.4f}  int8x1 {rel(lin(x)):.4f}"
    if x.shape[1] % 64 == 0:
        out = f"n {x.shape[0]}  int8 per 64 dims {rel(lin(x.view(x.shape[0], -1, 64)).view(x.shape)):.4f}  " + out[out.index("int8x2"):]
    try:
        sc = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 448.0
        out += f"  e4m3 {rel((x / sc).to(torch.float8_e4m3fn).float() * sc):.4f}"
    except Exception as e:  # the cast is not available on every build
        out += f"  e4m3 n/a ({type(e).__name__})"
    peak = lambda v: float((v.abs().amax(-1) / v.pow(2).mean(-1).sqrt().clamp(min=1e-12)).mean())
    return out + f"  peak kv {peak(a):.1f} rope {peak(b):.1f}  rms kv {float(a.pow(2).mean().sqrt()):.3f} rope {float(b.pow(2).mean().sqrt()):.3f}"


def latent_rows(cache_op, kv_flat, slots):
    """Latent rows [n, C] at physical slots, as the model's dtype whatever the cache stores."""
    fetch = getattr(cache_op, "fetch_slots", None)
    return fetch(kv_flat, slots) if fetch is not None else kv_flat.index_select(0, slots)


def sparse_prefill_attn(q_abs, kv_flat, phys, bias, scale, rr, lora, cache_op=None):
    """Prefill MLA over gathered union blocks with FusedSDPA in absorbed form: head dim L+P, one latent 'kv head'
    shared by the H query heads, V = the latent's first L dims. q_abs [Q,H,C], kv_flat [S,C], phys [nb,ucap],
    bias [Q,ucap]. Returns [Q,H,lora]."""
    _, h, c = q_abs.shape
    nb, ucap = phys.shape
    # an 8-bit cache has the codes, their cast and the product alive next to the latents: half the rows per call
    budget = ATTN_BUDGET_BYTES // (2 if hasattr(cache_op, "dequant") else 1)
    g = max(1, min(nb, budget // (ucap * c * 2)))
    outs = []
    for b0 in range(0, nb, g):
        b1 = min(nb, b0 + g)
        n = b1 - b0
        lat = latent_rows(cache_op, kv_flat, phys[b0:b1].reshape(-1)).view(n, 1, ucap, c)
        qb = q_abs[b0 * rr : b1 * rr].view(n, rr, h, c).transpose(1, 2)
        o = _FSDPA.apply(
            qb,
            lat,
            lat,
            bias[b0 * rr : b1 * rr].view(n, 1, rr, ucap),
            0.0,
            False,
            scale,
            SOFTMAX_MODE,
            True,
            None,
            "right",
            False,
            False,
            (-1, -1),
            None,
        )
        outs.append(o[..., :lora].transpose(1, 2).reshape(n * rr, h, lora))
        if g < nb:
            _graph_break()
    return torch.cat(outs, 0) if len(outs) > 1 else outs[0]


def dense_plan(q_len, kv, heads, per_head_bytes):
    """Head-group size and row-block size of a dense-masked prefill: the decompressed K/V of one head group stay
    within KV_BUDGET_BYTES and one row block's additive mask within MASK_BUDGET_BYTES (row blocks keep multiples of
    128 rows). Small prefills get the original plan: all heads at once, up to DENSE_ROW_BLOCKS row blocks."""
    g = max(1, min(heads, KV_BUDGET_BYTES // max(1, per_head_bytes)))
    while heads % g:
        g -= 1
    nb = DENSE_ROW_BLOCKS
    while nb > 1 and (q_len % nb or (q_len // nb) % 128):
        nb //= 2
    rb = q_len // nb
    while rb % 256 == 0 and rb * kv * 2 > MASK_BUDGET_BYTES:
        rb //= 2
    return g, rb


def dense_masked_attn(q, latent, kvb_w, dims, mask, cb, scale, packed=None):
    """Dense attention under a selection mask, one sequence. q [Q,H,D]; latent [KV, L+P] (cb context slots, then the
    Q chunk rows); kvb_w [H*(N+V), L] (the kv_b_proj weight of this rank); dims = (L, P, N, V, D); mask [Q,KV]
    (0 keep / NEG drop) or the bit-packed selection. K and V are decompressed for one head group at a time (see
    dense_plan); query rows go in blocks that read keys only up to the block's last row (the chunk part is
    causal), which skips about 3/8 of the first chunk's work."""
    lora, rope, nope, vdim, qk = dims
    q_, h, _ = q.shape
    kv = latent.shape[0]
    g, rb = dense_plan(q_, kv, h, kv * ((nope + vdim) + 2 * qk) * 2)
    k_c, k_r = latent.split([lora, rope], dim=-1)
    k_r = k_r.view(kv, 1, rope)
    nrb = q_ // rb
    outs = [[] for _ in range(nrb)]
    cut = packed is not None or g < h
    for h0 in range(0, h, g):
        h1 = h0 + g
        kvn = torch.nn.functional.linear(k_c, kvb_w[h0 * (nope + vdim) : h1 * (nope + vdim)])
        k_nope, v = kvn.view(kv, g, nope + vdim).split([nope, vdim], dim=-1)
        k = torch.cat((k_nope, k_r.expand(kv, g, rope)), dim=-1)
        if qk > vdim:
            v = torch.nn.functional.pad(v, (0, qk - vdim))
        v = v.contiguous()
        if cut:
            _tick("dense-kv")
        for i in range(nrb):
            r0, r1 = i * rb, (i + 1) * rb
            kend = cb + r1
            if packed is not None:
                m = bits_to_bias(packed[r0:r1], kend, q.dtype)
                _tick("dense-mask")
            else:
                m = mask[r0:r1, :kend].contiguous()
            o = _FSDPA.apply(
                q[r0:r1, h0:h1].unsqueeze(0).transpose(1, 2),
                k[:kend].unsqueeze(0).transpose(1, 2),
                v[:kend].unsqueeze(0).transpose(1, 2),
                m.view(1, 1, rb, kend),
                0.0,
                False,
                scale,
                SOFTMAX_MODE,
                True,
                None,
                "right",
                False,
                False,
                (-1, -1),
                None,
            )
            outs[i].append(o.transpose(1, 2)[0])
            if cut:
                _graph_break()
                _tick("dense-sdpa")
    rows = [torch.cat(r, dim=1) if len(r) > 1 else r[0] for r in outs]
    return torch.cat(rows, 0) if nrb > 1 else rows[0]


def decode_topk(q_idx, w, k_blocks, block_groups, block_bias, topk):
    """Decode indexer: q_idx [bs,Hi,D], w [bs,Hi], k_blocks [nb,T,D], block_groups [nb] (sequence id, -1 padding),
    block_bias [nb,T]. Returns idx [bs, topk] int32: flat positions (block_pos*T + t) in the block list, -1 = none."""
    bs = q_idx.shape[0]
    nb, t, _ = k_blocks.shape
    g = block_groups.to(torch.long)
    valid = (g >= 0) & (g < bs)
    gc = g.clamp(0, bs - 1)
    s = torch.relu(torch.bmm(q_idx.index_select(0, gc), k_blocks.transpose(1, 2)))
    s = torch.bmm(w.index_select(0, gc).to(s.dtype).unsqueeze(1), s).squeeze(1).float() + block_bias.float()
    s = torch.where(valid.unsqueeze(-1), s, -float("inf"))
    seq = torch.arange(bs, device=s.device, dtype=torch.long)
    member = (gc.unsqueeze(0) == seq.unsqueeze(1)) & valid.unsqueeze(0)
    dense = torch.where(member.unsqueeze(-1), s.unsqueeze(0), -float("inf"))
    v, i = topk_indices(dense.reshape(bs, nb * t), topk)
    return torch.where(v > -float("inf"), i.to(torch.int32), -1)


SENT = -1.0e30  # finite stand-in for "no key" in exchanged scores (an all-reduce must not see inf - inf)


def decode_topk_local(q_idx, w, k_loc, groups, bias, pos, topk):
    """One rank's share of the decode indexer scan. q_idx [R,Hi,D], w [R,Hi]; k_loc [P,T,D] = the key blocks at
    block-list positions pos [P] (int64); groups [P] = owning row of each block (-1: padding or beyond the list);
    bias [P,T] (0 / -inf). Every key is scored for every row with one GEMM (a row's own blocks are picked by
    `groups` afterwards), which is cheaper on Gaudi than one small product per block. Returns vals [R,topk] fp32
    (SENT where a row has fewer keys here) and flat positions [R,topk] fp32 (block position * T + slot)."""
    r, hi, d = q_idx.shape
    p, t, _ = k_loc.shape
    lg = torch.relu(torch.matmul(k_loc.reshape(p * t, d), q_idx.reshape(r * hi, d).transpose(0, 1)))
    eye = torch.eye(r, device=lg.device, dtype=lg.dtype)
    wm = (w.to(lg.dtype).unsqueeze(-1) * eye.unsqueeze(1)).reshape(r * hi, r)  # block-diagonal head weights
    s = torch.matmul(lg, wm).float().view(p, t, r) + bias.float().unsqueeze(-1)
    seq = torch.arange(r, device=lg.device, dtype=torch.long)
    member = groups.to(torch.long).unsqueeze(-1) == seq.unsqueeze(0)  # [P,R]
    s = torch.where(member.unsqueeze(1), s, -float("inf")).permute(2, 0, 1).reshape(r, p * t)
    v, i = topk_indices(s, topk)
    flat = pos.index_select(0, (i // t).reshape(-1)).view(i.shape) * t + i % t
    return v.clamp(min=SENT), flat.to(torch.float32)


def decode_topk_merge(cand, topk):
    """cand [W,R,2,topk]: every rank's (vals, flat positions). Returns idx [R,topk] int32 (-1 = none), the same on
    every rank; the order of the selected keys is arbitrary.

    With an even number of ranks the candidates go in two halves: one top-k call over [R, 2, W/2 * topk] (twice
    the rows at half the length: 62 us instead of 87 on Gaudi 2, where a top-k runs one row per TPC core), then
    the two sorted results are combined by the bitonic split: for descending A and B, max(A[t], B[k-1-t]) over t
    is exactly the k largest of both."""
    wn, r, _, k = cand.shape
    if wn % 2 or wn < 4 or k != topk:
        cv = cand[:, :, 0].permute(1, 0, 2).reshape(r, wn * k)
        cp = cand[:, :, 1].permute(1, 0, 2).reshape(r, wn * k)
        v, i = torch.topk(cv, topk, dim=-1)
        return torch.where(v > 0.5 * SENT, torch.gather(cp, 1, i).to(torch.int32), -1)
    half = wn // 2 * k
    cv = cand[:, :, 0].permute(1, 0, 2).reshape(r, 2, half)
    cp = cand[:, :, 1].permute(1, 0, 2).reshape(r, 2, half)
    v, i = torch.topk(cv, topk, dim=-1)
    p = torch.gather(cp, 2, i)
    a, b = v[:, 0], v[:, 1].flip(-1)
    best = torch.maximum(a, b)
    pos = torch.where(a >= b, p[:, 0], p[:, 1].flip(-1))
    return torch.where(best > 0.5 * SENT, pos.to(torch.int32), -1)


def owner_bias(block_list, block_groups, block_bias, rows, blocks_total):
    """Who may attend to which physical slot: [rows, blocks_total, T] fp32, 0 where the row's block list holds the
    block and the slot is in use, -inf everywhere else. block_list [nb] physical block ids, block_groups [nb] row of
    each entry (-1 padding), block_bias [nb, T] (0 / -inf). Padding entries land in a spare row that is cut off."""
    nb, t = block_bias.shape
    g = block_groups.to(torch.long)
    ok = (g >= 0) & (g < rows)
    dst = torch.where(ok, g.clamp(0, rows - 1) * blocks_total + block_list.to(torch.long), rows * blocks_total)
    canvas = torch.full((rows * blocks_total + 1, t), -float("inf"), dtype=torch.float32, device=block_bias.device)
    canvas.index_copy_(0, dst, block_bias.to(torch.float32))
    return canvas[: rows * blocks_total].view(rows, blocks_total, t)


def pool_topk_local(q_idx, w, k_loc, bias, first_slot, topk):
    """One rank's share of the key pool scored in place for every query row. q_idx [R,Hi,D], w [R,Hi]; k_loc [N,D]
    (a contiguous slice of the key buffer, physical slots first_slot ..); bias [R,N] from owner_bias. Returns
    vals [R,topk] fp32 (SENT where a row owns fewer keys here) and physical slots [R,topk] fp32."""
    r, hi, d = q_idx.shape
    # queries x keys^T with the keys as a transposed view: the scores come out as [rows, keys], the layout the
    # top-k wants (measured 80 us against 100 us for keys x queries^T followed by a transpose, 3 rows x 90K keys)
    lg = torch.relu(torch.matmul(q_idx.reshape(r * hi, d), k_loc.transpose(0, 1)))
    eye = torch.eye(r, device=lg.device, dtype=lg.dtype)
    wm = (w.to(lg.dtype).unsqueeze(-1) * eye.unsqueeze(1)).reshape(r * hi, r).transpose(0, 1)
    s = torch.matmul(wm, lg).float() + bias
    v, i = topk_indices(s, topk)
    return v.clamp(min=SENT), (i + first_slot).to(torch.float32)


def sparse_decode_attn(q_abs, kv_flat, phys, bias, scale, lora, cache_op=None):
    """q_abs [bs,H,C], phys [bs,k] latent rows, bias [bs,k] (0 / NEG). fp32 softmax."""
    bs, _, c = q_abs.shape
    k = phys.shape[1]
    lat = latent_rows(cache_op, kv_flat, phys.reshape(-1)).view(bs, k, c)
    s = torch.bmm(q_abs * scale, lat.transpose(1, 2)).float() + bias.unsqueeze(1)
    p = torch.softmax(s, dim=-1).to(lat.dtype)
    return torch.bmm(p, lat[..., :lora])


# ---------------------------------------------------------------------------------------------------------------------
# Glue: vllm-gaudi's HPU MLA wrapper and attention


def _tp():
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()


def _all_gather(x):
    from vllm.distributed import tensor_model_parallel_all_gather

    return tensor_model_parallel_all_gather(x.contiguous(), dim=0)


def _all_reduce(x):
    from vllm.distributed import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(x.contiguous())


def _put_keys(buf, slots, k):
    """Write indexer keys [n,D] at slots [n]; buf is [slots,D] (own buffer) or [slots,1,D] (vllm-gaudi's cache)."""
    buf.index_put_((slots,), k.to(buf.dtype).reshape(-1, *buf.shape[1:]))


def _exchange(v, flat, world, rank):
    """Every rank's candidates [R,topk] x 2 -> [W,R,2,topk] with one all-reduce (each rank fills its own slot)."""
    mine = torch.stack((v, flat), dim=1).unsqueeze(0)
    parts = []
    if rank:
        parts.append(torch.zeros(rank, *mine.shape[1:], dtype=mine.dtype, device=mine.device))
    parts.append(mine)
    if world - rank - 1:
        parts.append(torch.zeros(world - rank - 1, *mine.shape[1:], dtype=mine.dtype, device=mine.device))
    return _all_reduce(torch.cat(parts, 0) if len(parts) > 1 else mine)


def _pool_split(total_blocks, world, rank):
    per = -(-total_blocks // world)
    p0 = min(rank * per, total_blocks)
    return p0, min(p0 + per, total_blocks)


def _decode_select_pool(buf, md, q_i, w, topk, world, rank):
    """Decode selection by scanning this rank's physical 1/world of the key pool in place. Returns physical slots
    [R,topk] int32 (-1 = none), the same on every rank."""
    t = md.block_size
    r, _, d = q_i.shape
    total = buf.shape[0] // t
    p0, p1 = _pool_split(total, world, rank)
    if _STATE.get("pool_key") is not md.block_list:  # once per forward: every full layer has the same block lists
        _STATE["pool_key"] = md.block_list
        _STATE["pool_bias"] = owner_bias(md.block_list, md.block_groups, md.attn_bias, r, total)
    bias = _STATE["pool_bias"][:, p0:p1].reshape(r, (p1 - p0) * t)
    k_loc = buf[p0 * t : p1 * t].reshape((p1 - p0) * t, d)
    v, flat = pool_topk_local(q_i, w, k_loc, bias, p0 * t, topk)
    return decode_topk_merge(_exchange(v, flat, world, rank), topk)


def _decode_select_sharded(buf, md, q_i, w, topk, world, rank):
    """Decode selection with the scan split across the TP ranks (decode_topk_local on this rank's blocks, one
    all-reduce to exchange the candidates, decode_topk_merge)."""
    t = md.block_size
    d = q_i.shape[2]
    nb = md.block_list.shape[0]
    per = -(-nb // world)
    pos = torch.arange(rank * per, (rank + 1) * per, device=md.block_list.device, dtype=torch.long)
    inside = pos < nb
    posc = pos.clamp(max=nb - 1)
    bl = md.block_list.index_select(0, posc)
    grp = torch.where(inside, md.block_groups.index_select(0, posc).to(torch.long), -1)
    k_loc = buf.unflatten(0, (-1, t)).index_select(0, bl).reshape(per, t, d)
    v, flat = decode_topk_local(q_i, w, k_loc, grp, md.attn_bias.index_select(0, posc), posc, topk)
    return decode_topk_merge(_exchange(v, flat, world, rank), topk)


def _index_qkw(ind, rope, hidden_states, q_c, positions):
    """vLLM Indexer.forward up to the scores, in bf16."""
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    q_c = q_c.reshape(-1, q_c.shape[-1])
    q, _ = ind.wq_b(q_c)
    q = q.view(-1, ind.n_head, ind.head_dim)
    kw, _ = ind.wk_weights_proj(hidden_states)
    k = ind.k_norm(kw[:, : ind.head_dim])
    w = kw[:, ind.head_dim :]
    q_pe, q_nope = torch.split(q, [ind.rope_dim, ind.head_dim - ind.rope_dim], dim=-1)
    k_pe, k_nope = torch.split(k, [ind.rope_dim, ind.head_dim - ind.rope_dim], dim=-1)
    q_pe, k_pe = rope(positions, q_pe.contiguous(), k_pe.unsqueeze(1).contiguous())
    q = torch.cat([q_pe.reshape(-1, ind.n_head, ind.rope_dim), q_nope], dim=-1)
    k = torch.cat([k_pe.reshape(-1, ind.rope_dim), k_nope], dim=-1)
    # vLLM's positive per-head constants do not change the top-k; kept so scores match vLLM's scale
    w = w * (ind.softmax_scale * ind.n_head**-0.5)
    return q, k, w


def _native_keys(m, like):
    """vllm-gaudi's own BF16 indexer-key cache of a full indexer layer ([slots, 1, D], part of the KV pool, unused
    while the native sparse path is off), if it can hold this module's keys; else None."""
    kc = getattr(getattr(m.indexer, "k_cache", None), "kv_cache", None)
    if isinstance(kc, tuple | list):
        kc = kc[0] if kc else None
    if not isinstance(kc, torch.Tensor) or kc.dtype != KEY_DTYPE or kc.device != like.device:
        return None
    if kc.shape[0] != like.shape[0] or kc.numel() != like.shape[0] * m.indexer.head_dim:
        return None
    return kc


KEY_DTYPE = torch.bfloat16  # indexer keys, whatever the latent cache stores


def _alloc_for(model, kv_caches, what):
    from vllm_gaudi.attention.oot_mla import HPUMultiHeadLatentAttentionWrapper

    n = reused = nbytes = fp8 = 0
    kind = ""
    for m in model.modules():
        if isinstance(m, HPUMultiHeadLatentAttentionWrapper):
            kv = kv_caches.get(m.mla_attn.layer_name)
            kc = kv[0] if isinstance(kv, tuple | list) else kv
            if isinstance(kc, torch.Tensor) and kc.dtype == torch.float8_e4m3fn:
                m.mla_attn.impl.latent_cache_k = LatentFP8Cache(kc.shape[0], kc.device)
                kind = "E4M3 codes, one scale"
                fp8 += 1
            elif isinstance(kc, torch.Tensor) and kc.dtype == torch.int8:
                op = LatentInt8Cache(kc.shape[0], kc.device, m.kv_lora_rank, kc.shape[-1])
                op.tag = f"{what} #{fp8}"
                if Q8_DEBUG and _tp()[1] == 0:
                    op.enable_debug(kc)
                m.mla_attn.impl.latent_cache_k = op
                kind = f"linear int8 codes, {op.ngroups} scales"
                fp8 += 1
    if fp8:
        print(f"{TAG} {what}: 8-bit latent cache ({kind} per token) on {fp8} layers", flush=True)
    for m in model.modules():
        if isinstance(m, HPUMultiHeadLatentAttentionWrapper) and getattr(m, "indexer", None) is not None:
            kv = kv_caches.get(m.mla_attn.layer_name)
            if kv is None:
                continue
            kc = kv[0] if isinstance(kv, tuple | list) else kv
            nat = _native_keys(m, kc) if REUSE_NATIVE else None
            if nat is not None:
                m._glm_dsa_buf = nat
                reused += 1
            else:
                m._glm_dsa_buf = torch.zeros(kc.shape[0], m.indexer.head_dim, dtype=KEY_DTYPE, device=kc.device)
                nbytes += m._glm_dsa_buf.numel() * m._glm_dsa_buf.element_size()
            n += 1
            _INFO["topk"] = int(m.topk_tokens)
    if n:
        print(f"{TAG} {what}: indexer keys for {n} full layers, {reused} in vllm-gaudi's own key caches, "
              f"{nbytes / 2**30:.2f} GiB newly allocated per rank", flush=True)
    return n


def alloc_buffers(model, kv_caches):
    """Runner hook after the KV cache is bound: where each full indexer layer keeps its keys. The draft model's MTP
    layer owns an indexer too; it is reached through the calling runner (the hook only passes the target model)."""
    if not ENABLED:
        return
    import sys

    from vllm_gaudi.extension.runtime import get_config

    if get_config().defrag:
        raise RuntimeError(f"{TAG} GLM_DSA_SPARSE=1 needs KV defragmentation off: it moves blocks, not these buffers")
    if not _alloc_for(model, kv_caches, "target model"):
        raise RuntimeError(f"{TAG} GLM_DSA_SPARSE=1 but the model has no MLA layer with an indexer")
    runner = sys._getframe(1).f_locals.get("self")
    draft = getattr(getattr(runner, "drafter", None), "model", None)
    if isinstance(draft, torch.nn.Module):
        _alloc_for(draft, kv_caches, "draft model")
    _INFO["ready"] = True


def skip_prompt_bias(md, batch_size, seq_len):
    """Runner hook: the sparse prefill builds its own masks, so the dense prompt bias [bs,1,Q,ctx+Q] is not needed
    for a bs-1 prefill with more than MIN_KEYS keys."""
    if not (ENABLED and _INFO["ready"]) or batch_size != 1:
        return False
    nctx = md.block_list.shape[0] if md.block_list is not None else 0
    return nctx * md.block_size + seq_len > max(_INFO["topk"], MIN_KEYS)


def _batched_prefill_bias(self, md, q_i, k_i, w, hidden_states):
    buf = self._glm_dsa_buf
    bsz = md.seq_lens_tensor.shape[0]
    q_len = q_i.shape[0] // bsz
    hi, d = q_i.shape[1], q_i.shape[2]
    q_i = q_i.view(bsz, q_len, hi, d)
    w = w.view(bsz, q_len, hi)
    cur = k_i.to(buf.dtype).view(-1, d)
    if md.block_list is not None:
        past = buf.unflatten(0, (-1, md.block_size)).index_select(0, md.block_list).reshape(-1, d)
        k_all = torch.cat((past, cur), 0).view(bsz, -1, d)
        attn_bias = md.attn_bias
    else:
        k_all = cur.view(bsz, q_len, d)
        attn_bias = causal_bias(bsz, q_len, md.seq_lens_tensor, hidden_states.dtype, hidden_states.device)
    if attn_bias is None or k_all.shape[1] <= self.topk_tokens:
        return None
    return prefill_mask_bias(q_i, w, k_all, attn_bias, self.topk_tokens)


def _full_layer(self, md, hidden_states, q_c, positions):
    """Full indexer layer. Returns the selection the following shared layers reuse: a dict (sparse prefill,
    dense-masked prefill, sparse decode), a mask tensor (batched prefill, several query rows per sequence in decode)
    or None (dense attention is exact or the cheaper choice)."""
    q_i, k_i, w = _index_qkw(self.indexer, self.indexer_rope_emb, hidden_states, q_c, positions)
    buf = self._glm_dsa_buf
    _put_keys(buf, md.slot_mapping.flatten(), k_i)
    # When this layer goes on to attend densely, nothing in the graph reads the keys just written. A compiled graph
    # may drop an update of a module's tensor that no output depends on (seen with debug counters), and these keys
    # are needed later, once the context is long enough for the indexer: _wrapper_forward adds this value, times
    # zero, to the layer's output.
    _STATE["tie"] = buf.index_select(0, md.slot_mapping.flatten()[:1].to(torch.long).clamp(min=0))
    topk = self.topk_tokens
    bs_ = md.block_size
    d = q_i.shape[2]
    if md.is_prompt:
        bsz = md.seq_lens_tensor.shape[0]
        q_len = q_i.shape[0] // bsz
        nctx = md.block_list.shape[0] // bsz if md.block_list is not None else 0
        kv = nctx * bs_ + q_len
        if kv <= max(topk, MIN_KEYS):
            return None
        if bsz != 1:
            return _batched_prefill_bias(self, md, q_i, k_i, w, hidden_states)
        cur = k_i.to(buf.dtype).view(-1, d)
        bl = None
        if nctx:
            bl = md.block_list.to(torch.long).clamp(min=0)
            past = buf.unflatten(0, (-1, bs_)).index_select(0, bl).reshape(-1, d)
            kt = torch.cat((past, cur), 0).transpose(0, 1).contiguous()
        else:
            kt = cur.transpose(0, 1).contiguous()
        ctx_len = md.context_lens_tensor[:1].to(torch.int32)
        seq_len = md.seq_lens_tensor[:1].to(torch.int32)
        world, rank = _tp()
        split = world > 1 and q_len % world == 0  # each rank scores 1/world of the rows
        n = q_len // world if split else q_len
        rr = UNION_ROWS
        while n % rr:
            rr //= 2
        r0 = rank * n if split else 0
        sel, u = prefill_select(q_i, w, kt, r0, r0 + n, nctx * bs_, ctx_len, seq_len, topk, rr)
        lmax = u.to(torch.int32).sum(1, keepdim=True).amax(0, keepdim=True).to(torch.float32)
        gmax = _all_gather(lmax) if split else lmax
        umax = max(int(gmax.amax().cpu().item()), 1)  # host sync: the static union bucket
        ucap = union_bucket(umax, topk)
        if kv > FULLMASK_MAX_KEYS and not (ucap <= GATHER_MAX_FRAC * kv and ucap <= GATHER_MAX_UCAP):
            # Long context, scattered selections: a dense-masked layer would build a [rows, keys] mask per row
            # block and head group (measured: 20 s per 4096-token chunk at 230K keys). Smaller row groups have
            # smaller unions (16 rows select at most 16 x topk keys), so the gather path always applies.
            cands = [r2 for r2 in SMALL_UNION_ROWS if r2 < rr and n % r2 == 0]
            if cands:
                us = [sel.view(n // r2, r2, kv).any(1) for r2 in cands]
                lm = torch.stack([x.to(torch.int32).sum(1).amax() for x in us]).to(torch.float32).view(1, -1)
                gm = (_all_gather(lm) if split else lm).amax(0).cpu().tolist()  # host sync
                best = None
                for r2, x, m in zip(cands, us, gm):
                    cap2 = union_bucket(max(int(m), 1), topk)
                    if cap2 <= GATHER_MAX_UCAP and (best is None or cap2 / r2 < best[2] / best[0]):
                        best = (r2, x, cap2, int(m))
                if best is not None:
                    rr, u, ucap, umax = best
                del us
        if TRACE:
            _TR.setdefault("u", []).append(f"{umax}>{ucap}/{rr}" if ucap <= GATHER_MAX_FRAC * kv
                                           and ucap <= GATHER_MAX_UCAP else f"{umax}:dense")
        if ucap <= GATHER_MAX_FRAC * kv and ucap <= GATHER_MAX_UCAP:
            cl, bias = compact(sel, u, rr, ucap, hidden_states.dtype)
            if split:
                cl, bias = _all_gather(cl), _all_gather(bias)
            cur_slots = md.slot_mapping.flatten().to(torch.long).clamp(min=0)
            if nctx:
                ctx_slots = bl.view(-1, 1) * bs_ + torch.arange(bs_, device=bl.device, dtype=torch.long).view(1, -1)
                vslot = torch.cat((ctx_slots.reshape(-1), cur_slots), 0)
            else:
                vslot = cur_slots
            phys = vslot.index_select(0, cl.reshape(-1).to(torch.long)).view(cl.shape)
            _graph_break()
            return {"kind": "prefill", "phys": phys, "bias": bias, "rr": rr}
        if kv > FULLMASK_MAX_KEYS:
            packed = pack_bits(sel)
            del sel, u
            if split:
                packed = _all_gather(packed)
            _graph_break()
            return {"kind": "prefill_dense", "mask": None, "packed": packed, "cb": nctx * bs_, "bl": bl}
        if split:
            sel = unpack_bits(_all_gather(pack_bits(sel)), kv)
        mask = torch.where(sel, 0.0, NEG).to(hidden_states.dtype)
        del sel, u
        _graph_break()
        return {"kind": "prefill_dense", "mask": mask, "packed": None, "cb": nctx * bs_, "bl": bl}
    nb = md.block_list.shape[0]
    bs = md.block_mapping.shape[1] if md.block_mapping is not None else q_i.shape[0]
    if nb * bs_ <= topk or nb * bs_ <= MIN_KEYS_DECODE * bs:
        return None
    world, rank = _tp()
    total_blocks = buf.shape[0] // bs_
    if (
        SHARD
        and world > 1
        and q_i.shape[0] == bs
        and nb >= POOL_FRAC * total_blocks
        and (total_blocks - (world - 1) * -(-total_blocks // world)) * bs_ >= topk
    ):
        slots = _decode_select_pool(buf, md, q_i, w, topk, world, rank)
        return {"kind": "decode", "phys": slots.clamp(min=0).to(torch.long),
                "bias": torch.where(slots >= 0, 0.0, NEG).to(torch.float32)}
    if SHARD and world > 1 and q_i.shape[0] == bs and -(-nb // world) * bs_ >= topk:
        idx = _decode_select_sharded(buf, md, q_i, w, topk, world, rank)
    else:
        kb = buf.unflatten(0, (-1, bs_)).index_select(0, md.block_list).reshape(nb, bs_, d)
        if q_i.shape[0] != bs:
            return decode_mask_bias(q_i, w, kb, md.block_groups, md.attn_bias, topk)
        idx = decode_topk(q_i, w, kb, md.block_groups, md.attn_bias, topk)
    ok = idx >= 0
    ic = idx.clamp(min=0).to(torch.long)
    blk = md.block_list.to(torch.long).index_select(0, (ic // bs_).reshape(-1)).view(ic.shape).clamp(min=0)
    phys = blk * bs_ + ic % bs_
    return {"kind": "decode", "phys": phys, "bias": torch.where(ok, 0.0, NEG).to(torch.float32)}


def _wrapper_forward(self, positions, hidden_states, llama_4_scaling=None):
    """MultiHeadLatentAttentionWrapper.forward (the MLA part GLM uses), plus the indexer."""
    from vllm.forward_context import get_forward_context

    q_c = None
    if self.q_lora_rank is not None:
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]
    else:
        kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        q = self.q_proj(hidden_states)[0]
    kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_c_normed = self.kv_a_layernorm(kv_c)
    q = q.view(-1, self.num_heads, self.qk_head_dim)
    k_pe = k_pe.unsqueeze(1)
    if self.rotary_emb is not None:
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(positions, q[..., self.qk_nope_head_dim :], k_pe)

    sel = None
    md = get_forward_context().attn_metadata
    if isinstance(md, dict):
        md = md.get(self.mla_attn.layer_name)
    if TRACE:
        name = self.mla_attn.layer_name
        if _TR.get("first") is None:
            _TR["first"] = name
        if name == _TR["first"]:
            _trace_report(f"forward {_TR['calls']}")
            _TR["calls"] += 1
        _tick("other")
    if md is not None and q_c is not None:
        if _STATE.get("draft_iter") and getattr(self, "indexer", None) is not None:
            # second and later MTP draft passes of one step (glm53_hpu.draft_chain): the first pass's selection,
            # nothing written to the key cache
            sel = _STATE["sel"]
        elif getattr(self, "indexer", None) is not None and getattr(self, "_glm_dsa_buf", None) is not None:
            sel = _full_layer(self, md, hidden_states, q_c, positions)
            _STATE["sel"] = sel
        elif getattr(self, "indexer", None) is None and getattr(self, "skip_topk", False):
            sel = _STATE["sel"]  # shared layer: reuse the group's selection
    if llama_4_scaling is not None:
        q *= llama_4_scaling
    if TRACE:
        _tick("indexer")
    _STATE["active"] = sel
    tie = _STATE.pop("tie", None)
    try:
        attn_out = self.mla_attn(
            q, kv_c_normed, k_pe, output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim)
        )
    finally:
        _STATE["active"] = None
    if tie is not None and not isinstance(sel, dict):
        attn_out = attn_out + (tie.float().sum() * 0.0).to(attn_out.dtype)
    if TRACE:
        kind = sel["kind"] if isinstance(sel, dict) else ("mask" if sel is not None else "dense")
        _tick(f"attn-{kind}")
    return self.o_proj(attn_out)[0]


def _prefill_dense_impl(self, q, k_c_normed, k_pe, kv_cache, md, sel):
    """Prefill whose layer group is dense-masked: write the latent cache, then dense attention under the selection
    mask over every visible slot, with K/V decompressed from the latents per head group."""
    impl = self.impl
    slot_mapping = md.slot_mapping.flatten() if md.slot_mapping is not None else None
    latent = torch.concat((k_c_normed, k_pe.view(*k_c_normed.shape[:-1], self.qk_rope_head_dim)), dim=-1)
    latent = latent.view(-1, self.qk_rope_head_dim + self.kv_lora_rank)
    impl.latent_cache_k(latent, kv_cache[0], slot_mapping)
    if sel["cb"]:
        past = impl.latent_cache_k.fetch_from_cache(kv_cache[0].unflatten(0, (-1, md.block_size)), sel["bl"])
        latent = torch.concat((past.view(-1, past.shape[-1]), latent), dim=0)
    q = q.view(-1, self.num_heads, self.qk_head_dim)
    dims = (self.kv_lora_rank, self.qk_rope_head_dim, self.qk_nope_head_dim, self.v_head_dim, self.qk_head_dim)
    _tick(None)
    o = dense_masked_attn(q, latent, impl.kv_b_proj.weight, dims, sel["mask"], sel["cb"], self.scale, sel["packed"])
    _tick("dense-tail")
    return o[..., : self.v_head_dim].reshape(-1, self.num_heads * self.v_head_dim)


def _forward_impl(
    self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None
):
    """HPUMLAAttention.forward_impl: the sparse paths for a dict selection, the original otherwise (dense, or a mask
    tensor that forward_mha / forward_mqa below substitute for the attention bias)."""
    sel = _STATE["active"]
    if isinstance(sel, dict) and sel["kind"] == "prefill_dense":
        return _prefill_dense_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, sel)
    if not isinstance(sel, dict):
        return _ORIG_FORWARD_IMPL(
            self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output, output_scale, output_block_scale
        )
    slot_mapping = attn_metadata.slot_mapping.flatten() if attn_metadata.slot_mapping is not None else None
    latent = torch.concat((k_c_normed, k_pe.view(*k_c_normed.shape[:-1], self.qk_rope_head_dim)), dim=-1)
    latent = latent.view(-1, self.qk_rope_head_dim + self.kv_lora_rank)
    self.impl.latent_cache_k(latent, kv_cache[0], slot_mapping)
    kc = kv_cache[0]
    q = q.view(-1, self.num_heads, self.qk_head_dim)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    ql = torch.bmm(q_nope.transpose(0, 1).contiguous(), self.W_UK_T).transpose(0, 1)
    q_abs = torch.cat((ql, q_pe), dim=-1).contiguous()
    if sel["kind"] == "prefill":
        o = sparse_prefill_attn(q_abs, kc, sel["phys"], sel["bias"], self.scale, sel["rr"], self.kv_lora_rank,
                                self.impl.latent_cache_k)
    else:
        o = sparse_decode_attn(q_abs, kc, sel["phys"], sel["bias"], self.scale, self.kv_lora_rank,
                               self.impl.latent_cache_k)
    return self._v_up_proj(o)


def _forward_mha(self, q, latent_vec_k, k_cache, attn_metadata):
    """HPUMLAImpl.forward_mha with a mask selection substituted for the prompt bias."""
    from vllm_gaudi.extension import ops

    if attn_metadata.block_list is not None:
        current = latent_vec_k
        if isinstance(k_cache, tuple):
            k_cache = k_cache[0]
        past = self.latent_cache_k.fetch_from_cache(
            k_cache.unflatten(0, (-1, attn_metadata.block_size)), attn_metadata.block_list
        )
        latent_vec_k = torch.concat((past.view(-1, past.shape[-1]), current), dim=0)
    k_c_normed, k_pe = latent_vec_k.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
    kv_nope = self.kv_b_proj(k_c_normed)[0].view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
    batch_size = 1 if self.use_merged_prefill else attn_metadata.seq_lens_tensor.shape[0]
    q = q.view(batch_size, -1, self.num_heads, self.qk_head_dim)
    k = k.view(batch_size, -1, self.num_heads, self.qk_head_dim)
    v = v.view(batch_size, -1, self.num_heads, self.v_head_dim)
    if self.qk_head_dim > self.v_head_dim:
        pad = torch.zeros(*v.shape[:-1], q.shape[-1] - v.shape[-1], device=v.device, dtype=v.dtype)
        v_padded = torch.cat((v, pad), dim=-1)
    else:
        v_padded = v
    sel = _STATE["active"]
    output = ops.prompt_attention(
        impl=self.prefill_impl,
        query=q,
        key=k,
        value=v_padded,
        is_causal=True,
        attn_bias=sel if sel is not None else attn_metadata.attn_bias,
        position_bias=None,
        valid_seq_lengths=attn_metadata.seq_lens_tensor,
        scale=self.scale,
        matmul_qk_op=self.matmul_qk,
        softmax_op=self.softmax,
        matmul_av_op=self.matmul_av,
        keys_fetch_func=self.latent_cache_k.fetch_from_cache,
        values_fetch_func=None,
        fsdpa_op=self.fused_scaled_dot_product_attention,
    )
    output = output.view(batch_size, -1, self.num_heads, q.shape[-1])[..., : v.shape[-1]]
    return output.reshape(-1, self.num_heads * v.shape[-1])


def _forward_mqa(self, q_nope, q_pe, k_cache, attn_metadata):
    """HPUMLAImpl.forward_mqa with a mask selection substituted for the block bias."""
    from vllm_gaudi.attention.ops.hpu_paged_attn import HPUPagedAttention

    if isinstance(k_cache, tuple):
        k_cache = k_cache[0]
    query = torch.cat([q_nope, q_pe], dim=-1)
    key_cache = k_cache.unsqueeze(1) if k_cache is not None else None
    sel = _STATE["active"]
    return HPUPagedAttention.forward_decode(
        query=query,
        key_cache=key_cache,
        value_cache=None,
        block_list=attn_metadata.block_list,
        block_mapping=attn_metadata.block_mapping,
        block_bias=sel if sel is not None else attn_metadata.attn_bias,
        block_groups=attn_metadata.block_groups,
        block_size=attn_metadata.block_size,
        scale=self.scale,
        matmul_qk_op=self.matmul_qk,
        matmul_av_op=self.matmul_av,
        batch2block_matmul_op=self.batch2block_matmul,
        block2batch_matmul_op=self.block2batch_matmul,
        keys_fetch_func=self.latent_cache_k.fetch_from_cache,
        values_fetch_func=None,
        kv_lora_rank=self.kv_lora_rank,
    )


def install(wrapper_cls, attention_cls):
    """Swap in the DSA methods (called at the end of vllm_gaudi/attention/oot_mla.py)."""
    global _ORIG_FORWARD_IMPL
    if not ENABLED or _ORIG_FORWARD_IMPL is not None:
        return
    print(f"{TAG} long-context revision: reuse_native={REUSE_NATIVE} shard={SHARD} decode above {MIN_KEYS_DECODE} "
          f"keys, pool scan from {POOL_FRAC:.2f} of the pool, kv budget {KV_BUDGET_BYTES / 2**30:.1f} GiB, mask budget {MASK_BUDGET_BYTES / 2**30:.1f} GiB, "
          f"trace={TRACE}", flush=True)
    from vllm_gaudi.attention.backends.hpu_attn import HPUMLAImpl

    HPUMLAImpl.forward_mha = _forward_mha
    HPUMLAImpl.forward_mqa = _forward_mqa
    _ORIG_FORWARD_IMPL = attention_cls.forward_impl
    wrapper_cls.forward = _wrapper_forward
    attention_cls.forward_impl = _forward_impl
    print(f"{TAG} sparse attention on above {MIN_KEYS} keys per sequence", flush=True)
