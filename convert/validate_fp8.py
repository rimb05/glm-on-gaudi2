#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check an FP8 checkpoint written by bf16_to_fp8.py against its BF16 source, with independent code.

For every routed-expert tensor of the requested layers:
  - the stored scale equals amax / 240 of the source row (1.0 for an all-zero row), fp32, shape [rows, 1];
  - every stored code byte equals the code an explicit table search gives for source / scale (fp8.quantize_rows_grid:
    nearest representable value, ties to the even code, then the no-subnormal / no-negative-zero rule), i.e. not the
    torch cast the converter used;
  - no byte is 0x80, a subnormal code, or uses exponent field 15.
For every other tensor (with --rest): dtype, shape and bytes equal the source's.
With --index: the index lists exactly the source's tensors plus one scale per expert weight, every listed shard
exists, there are no stray shards, and config.json carries the quantization config.

  validate_fp8.py --src SRC --dst DST --layers 3-12 [--threads 8]     (parallelise by layer range)
  validate_fp8.py --src SRC --dst DST --rest
  validate_fp8.py --src SRC --dst DST --index

Exit code 0 only if everything checked is right.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8  # noqa: E402
from bf16_to_fp8 import EXPERT_RE, EXPERT_TARGET, Source, parse_layers, shard_files  # noqa: E402


def check_experts(src: Source, dst: Source, layers: set[int], stride: int = 1) -> int:
    n_exp = src.expert_layers()
    todo = sorted(set(n_exp) & layers)
    if not todo:
        raise SystemExit("no expert layers in the requested range")
    bad = 0
    for layer in todo:
        t0 = time.time()
        n = 0
        # with a stride, the checked experts shift by layer so that every expert index is covered across layers
        for e in range(layer % stride, n_exp[layer], stride):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
                w = src.get(name)
                if dst.meta.get(name, ("", []))[0] != "F8_E4M3" or dst.meta[name][1] != list(w.shape):
                    print(f"FAIL {name}: stored as {dst.meta.get(name)}")
                    bad += 1
                    continue
                if dst.meta.get(name + "_scale") != ("F32", [w.shape[0], 1]):
                    print(f"FAIL {name}_scale: stored as {dst.meta.get(name + '_scale')}")
                    bad += 1
                    continue
                q = dst.get(name).view(torch.uint8)
                s = dst.get(name + "_scale")
                amax = w.to(torch.float32).abs().amax(dim=1, keepdim=True)
                want_s = torch.where(amax > 0, amax / torch.tensor(240.0, dtype=torch.float32), torch.ones_like(amax))
                if not torch.equal(s, want_s):
                    print(f"FAIL {name}_scale: {int((s != want_s).sum())} rows differ from amax / 240")
                    bad += 1
                    continue
                want_q, _ = fp8.quantize_rows_grid(w, s)
                if not torch.equal(q, want_q):
                    k = (q != want_q).nonzero()
                    print(f"FAIL {name}: {k.shape[0]} codes differ, first at {k[0].tolist()}")
                    bad += 1
                    continue
                try:
                    fp8.check_codes(q)
                except ValueError as ex:
                    print(f"FAIL {name}: {ex}")
                    bad += 1
                    continue
                n += 1
        want = 3 * len(range(layer % stride, n_exp[layer], stride))
        print(f"[validate] layer {layer}: {n} of {want} checked expert tensors match the independent grid"
              f"{'' if stride == 1 else f' (every {stride}th expert)'}, {time.time() - t0:.0f} s", flush=True)
    return bad


def check_rest(src: Source, dst: Source) -> int:
    bad = n = 0
    for name in sorted(src.shard_of):
        if EXPERT_RE.match(name):
            continue
        if dst.meta.get(name) != src.meta[name]:
            print(f"FAIL {name}: {dst.meta.get(name)} vs source {src.meta[name]}")
            bad += 1
            continue
        a, b = src.get(name), dst.get(name)
        if not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
            print(f"FAIL {name}: bytes differ from the source")
            bad += 1
            continue
        n += 1
    print(f"[validate] rest: {n} tensors byte-identical to the source", flush=True)
    return bad


def check_index(src: Source, dst_dir: str) -> int:
    bad = 0
    with open(os.path.join(dst_dir, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    want = set()
    for name in src.shard_of:
        want.add(name)
        if EXPERT_RE.match(name):
            want.add(name + "_scale")
    if set(wm) != want:
        print(f"FAIL index: {len(want - set(wm))} missing, {len(set(wm) - want)} unexpected")
        bad += 1
    on_disk = set(shard_files(dst_dir))
    if on_disk != set(wm.values()):
        print(f"FAIL shards: on disk but not indexed {sorted(on_disk - set(wm.values()))[:3]}, indexed but missing "
              f"{sorted(set(wm.values()) - on_disk)[:3]}")
        bad += 1
    dst = Source(dst_dir)
    wrong = [n for n, fn in wm.items() if os.path.basename(dst.shard_of.get(n, "")) != fn]
    if wrong:
        print(f"FAIL index: {len(wrong)} tensors are not in the shard the index names, e.g. {wrong[0]}")
        bad += 1
    with open(os.path.join(dst_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(src.path, "config.json")) as f:
        scfg = json.load(f)
    qc = cfg.get("quantization_config") or {}
    grp = (qc.get("config_groups") or {}).get("routed_experts") or {}
    ok = (qc.get("quant_method") == "compressed-tensors" and grp.get("targets") == [EXPERT_TARGET]
          and (grp.get("weights") or {}).get("strategy") == "channel"
          and (grp.get("input_activations") or {}).get("dynamic") is True and "kv_cache_scheme" not in qc)
    if not ok:
        print("FAIL config.json: quantization_config is not the expected one")
        bad += 1
    if {k: v for k, v in cfg.items() if k != "quantization_config"} != scfg:
        print("FAIL config.json: differs from the source beyond quantization_config")
        bad += 1
    if not bad:
        print(f"[validate] index: {len(wm)} tensors in {len(on_disk)} shards, config ok", flush=True)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    job = ap.add_mutually_exclusive_group(required=True)
    job.add_argument("--layers")
    job.add_argument("--rest", action="store_true")
    job.add_argument("--index", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="check every N-th expert of each layer (1 = all)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    src = Source(a.src)
    if a.index:
        bad = check_index(src, a.dst)
    else:
        dst = Source(a.dst)
        bad = check_rest(src, dst) if a.rest else check_experts(src, dst, parse_layers(a.layers), max(a.stride, 1))
    print("PASS" if not bad else f"FAIL: {bad} problems")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
