#!/usr/bin/env python3
"""Latency of a two-rank HCCL all-reduce used as a point-to-point send, as a function of how early the receiver posts.

Run one process per rank on the same host (so wall clocks agree):
  RANK=0 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 python3 hccl_lat.py &
  RANK=1 WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 python3 hccl_lat.py
Rank 0 sends [N] bf16 values, rank 1 receives (adds zeros). Each iteration both ranks wait for a common wall-clock
tick; the receiver posts at the tick, the sender DELAY ms later, then both force completion with .item().
Reported: receiver completion time minus the sender's post time (ms), per delay.
"""
import os
import statistics as st
import sys
import time

import torch
import habana_frameworks.torch as ht  # noqa: F401
import habana_frameworks.torch.distributed.hccl  # noqa: F401

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
N = int(os.environ.get("N", str(2 * 6144)))
ITERS = int(os.environ.get("ITERS", "60"))
DELAYS = [float(x) for x in os.environ.get("DELAYS", "0,0.3,1,3,8").split(",")]
MODE = os.environ.get("MODE", "pair")  # pair: new_group([0,1]); world: default group

torch.distributed.init_process_group(backend="hccl", rank=rank, world_size=world)
dev = torch.device("hpu")
g = torch.distributed.new_group(ranks=[0, 1], backend="hccl") if MODE == "pair" else None


def now():
    return time.time()


def wait_until(t):
    while True:
        d = t - now()
        if d <= 0:
            return
        if d > 0.002:
            time.sleep(d - 0.001)


def one(delay_ms, tick):
    """Returns (post_time, done_time) on this rank."""
    if rank == 0:
        x = torch.full((N,), 1.0, dtype=torch.bfloat16, device=dev) * 1.0
        ht.core.mark_step()
        _ = x[0].item()                       # the payload exists on the device before the tick
        wait_until(tick + delay_ms / 1e3)
    else:
        x = torch.zeros(N, dtype=torch.bfloat16, device=dev)
        ht.core.mark_step()
        wait_until(tick)
    t_post = now()
    torch.distributed.all_reduce(x, group=g)
    y = x[:8].float().sum()
    v = y.item()                              # forces the collective and a tiny reduction
    t_done = now()
    assert v == 8.0, v
    return t_post, t_done


# warm-up (compiles the tiny graphs)
t = (int(now() * 10) + 20) / 10.0
for i in range(5):
    one(0.0, t + i * 0.1)

period = 0.05
res = {}
t = (int(now() * 10) + 10) / 10.0
k = 0
for d in DELAYS:
    out = []
    for i in range(ITERS):
        out.append(one(d, t + k * period))
        k += 1
    res[d] = out

# local cost of the same sequence without a collective (item + tiny reduction), for calibration
cal = []
for i in range(30):
    x = torch.full((N,), 1.0, dtype=torch.bfloat16, device=dev) * 1.0
    ht.core.mark_step()
    _ = x[0].item()
    t0 = now()
    v = x[:8].float().sum().item()
    cal.append((now() - t0) * 1e3)

path = f"/tmp/hccl_lat_rank{rank}.txt"
with open(path, "w") as f:
    for d, out in res.items():
        for a, b in out:
            f.write(f"{d} {a:.6f} {b:.6f}\n")
print(f"rank {rank}: local sync+reduce without a collective: median {st.median(cal):.3f} ms", flush=True)
if rank == 1:
    time.sleep(1.0)
    snd = {}
    for line in open("/tmp/hccl_lat_rank0.txt"):
        d, a, b = line.split()
        snd.setdefault(float(d), []).append((float(a), float(b)))
    print(f"payload {N} bf16 values ({N * 2 / 1024:.1f} KiB), mode {MODE}")
    print("delay_ms  recv_done-send_post (median, p10, p90) ms   send post->done median ms   recv post->done median ms")
    for d in DELAYS:
        lat = [(rb - sa) * 1e3 for (sa, sb), (ra, rb) in zip(snd[d], res[d])]
        s_self = [(sb - sa) * 1e3 for sa, sb in snd[d]]
        r_self = [(rb - ra) * 1e3 for ra, rb in res[d]]
        q = sorted(lat)
        print(f"{d:7.1f}   {st.median(lat):7.3f} {q[len(q) // 10]:7.3f} {q[9 * len(q) // 10]:7.3f}"
              f"            {st.median(s_self):7.3f}                  {st.median(r_self):7.3f}")
sys.stdout.flush()
