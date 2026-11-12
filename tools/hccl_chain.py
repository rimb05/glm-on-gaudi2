#!/usr/bin/env python3
"""End-to-end latency of a chain of single-card stages linked by two-rank HCCL all-reduces, in lockstep.

One process per rank on one host:  for r in 0..n-1: RANK=r WORLD_SIZE=n MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 python3 hccl_chain.py
Each iteration starts on a common wall-clock tick. Ranks 1..n-1 post their forward EARLY ms before the tick (they
wait for data, like pipeline stages whose host has already launched the step); rank 0 posts at the tick. A stage's
forward is: receive from the left pair (all-reduce with zeros), WORK matmuls, send to the right pair. The last rank
forces completion and reports done - tick. MODE=graph wraps the forward in an HPU graph (replay), MODE=lazy does not.
"""
import os
import statistics as st
import sys
import time

import torch
import habana_frameworks.torch as ht
import habana_frameworks.torch.distributed.hccl  # noqa: F401

rank = int(os.environ["RANK"])
n = int(os.environ["WORLD_SIZE"])
H = 2 * 6144
ITERS = int(os.environ.get("ITERS", "80"))
EARLY = float(os.environ.get("EARLY", "5"))
WORK = int(os.environ.get("WORK", "0"))
MODE = os.environ.get("MODE", "graph")
PERIOD = float(os.environ.get("PERIOD", "0.04"))

torch.distributed.init_process_group(backend="hccl", rank=rank, world_size=n)
dev = torch.device("hpu")
groups = {}
for r in range(n - 1):
    g = torch.distributed.new_group(ranks=[r, r + 1], backend="hccl")
    if rank in (r, r + 1):
        groups[r] = g
left, right = groups.get(rank - 1), groups.get(rank)


class Stage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.eye(H, dtype=torch.bfloat16) * 1.0, requires_grad=False)

    def forward(self, x):
        if left is not None:
            buf = torch.zeros(1, H, dtype=torch.bfloat16, device=x.device)
            torch.distributed.all_reduce(buf, group=left)
            x = buf
        for _ in range(WORK):
            x = torch.matmul(x, self.w)
            ht.core.mark_step()
        if right is not None:
            y = x + 0
            torch.distributed.all_reduce(y, group=right)
            return y
        return x


m = Stage().to(dev).eval()
if MODE == "graph":
    m = ht.hpu.wrap_in_hpu_graph(m, disable_tensor_cache=True)
x0 = torch.ones(1, H, dtype=torch.bfloat16, device=dev)


def now():
    return time.time()


def wait_until(t):
    while True:
        d = t - now()
        if d <= 0:
            return
        if d > 0.002:
            time.sleep(d - 0.001)


def one(tick):
    wait_until(tick if rank == 0 else tick - EARLY / 1e3)
    t_post = now()
    with torch.inference_mode():
        y = m(x0)
        ht.core.mark_step()
    t_launched = now()
    if rank == n - 1:
        v = y[0, :4].float().sum().item()
        assert v == 4.0, v
    return t_post, t_launched, now()


base = (int(now()) + 3) * 1.0
for i in range(8):
    one(base + i * 0.25)                     # warm-up: graph capture / compile
base = (int(now()) + 2) * 1.0
rows = [(base + i * PERIOD,) + one(base + i * PERIOD) for i in range(ITERS)]
host = [(r[2] - r[1]) * 1e3 for r in rows]
print(f"rank {rank}: host time to launch one forward: median {st.median(host):.3f} ms", flush=True)
if rank == n - 1:
    lat = sorted((r[3] - r[0]) * 1e3 for r in rows)
    print(f"RESULT n={n} mode={MODE} work={WORK} early={EARLY}: chain latency median {st.median(lat):.3f} ms "
          f"(p10 {lat[len(lat) // 10]:.3f}, p90 {lat[9 * len(lat) // 10]:.3f})", flush=True)
