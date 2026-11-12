#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a "performance twin" of GLM-5.3: N layers with the real per-layer dimensions, for timing and memory work.

The twin is NOT a usable model: its text is meaningless. It exists because the real checkpoint (744B parameters)
takes a long time to obtain and to load, while per-layer decode time, pipeline overhead, sparse-attention cost at
long context and KV memory per token depend only on the layer dimensions, which the twin shares exactly.

Source: inference-optimization/GLM-5.3-3.5B-A3.3B-small (3 layers with GLM-5.3's dimensions, 10 experts):
  template 0 = its layer 0 (full DSA indexer + dense MLP)       -> twin layers 0, 1, 2
  template 1 = its layer 1 (full DSA indexer + MoE)             -> twin layers 6, 10, 14, ... (the indexer owners)
  template 2 = its layer 2 (shared indexer + MoE)               -> every other twin layer
The 10 source experts are repeated to E experts per layer; the router weights are random, so routing is spread
over all experts like a real model's (this is what makes expert-parallel load imbalance show up). Routed experts
are written directly as FP8 per output channel (fp8.quantize_rows), everything else as BF16, with the same names
and quantization config as bf16_to_fp8.py produces.

Optional --mtp DIR: a directory with the shards of the real model that hold its MTP layer (tensors
model.layers.78.*; the index file of the checkpoint tells which shards those are, and the whole checkpoint
directory works too). The layer is renamed to index N; its first E routed experts are converted to FP8. With --mtp
the config says num_nextn_predict_layers = 1.

  make_twin.py --small SMALL_DIR --out OUT --layers 26 --experts 256 [--mtp MTP_DIR] [--link-from OTHER_TWIN]

--link-from hard-links layer shards that an existing twin already has (same experts count), so a shorter twin costs
no extra disk for the layers it shares.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8  # noqa: E402
from bf16_to_fp8 import AUX_FILES, EXPERT_RE, Source, quant_config  # noqa: E402

REAL_MTP_LAYER = 78


def is_full(i: int, offset: int = 3, freq: int = 4) -> bool:
    """vLLM's rule for layers that own a DSA indexer (deepseek_v2.py: _skip_topk)."""
    return max(i - offset + 1, 0) % freq == 0


_ST_DTYPE = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float8_e4m3fn: "F8_E4M3", torch.float16: "F16"}


def save(out_dir: str, fn: str, tensors: dict, wm: dict) -> None:
    """Write one safetensors file from raw bytes. safetensors.torch.save_file refuses tensors that share memory,
    and the repeated experts of a twin layer do; here each name gets its own copy of the bytes on disk."""
    path = os.path.join(out_dir, fn)
    header, off = {}, 0
    for name, t in tensors.items():
        nb = t.numel() * t.element_size()
        header[name] = {"dtype": _ST_DTYPE[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + nb]}
        off += nb
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path + ".tmp", "wb") as f:
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        for t in tensors.values():
            f.write(memoryview(t.contiguous().view(torch.uint8).numpy()).cast("B"))
    os.replace(path + ".tmp", path)
    os.chmod(path, 0o644)
    wm.update(dict.fromkeys(tensors, fn))
    print(f"[twin] wrote {fn}: {len(tensors)} tensors, {os.path.getsize(path) / 1e9:.2f} GB", flush=True)


def layer_names(src: Source, layer: int) -> list[str]:
    p = f"model.layers.{layer}."
    return sorted(n for n in src.shard_of if n.startswith(p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--mtp")
    ap.add_argument("--link-from")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    n_layers, n_exp = a.layers, a.experts
    src = Source(a.small)
    with open(os.path.join(a.small, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["num_hidden_layers"] == 3 and cfg["first_k_dense_replace"] == 1, "unexpected source model"
    src_exp = cfg["n_routed_experts"]
    os.makedirs(a.out, exist_ok=True)
    wm: dict[str, str] = {}
    t0 = time.time()

    # top-level tensors
    save(a.out, "model-top.safetensors",
         {n: src.get(n) for n in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")}, wm)

    # quantize the source experts once per template layer
    qexp: dict[int, list[dict[str, tuple[torch.Tensor, torch.Tensor]]]] = {}
    for tl in (1, 2):
        qexp[tl] = []
        for e in range(src_exp):
            d = {}
            for proj in ("gate_proj", "up_proj", "down_proj"):
                w = src.get(f"model.layers.{tl}.mlp.experts.{e}.{proj}.weight")
                q, s = fp8.quantize_rows(w)
                fp8.check_codes(q.view(torch.uint8))
                d[proj] = (q, s)
            qexp[tl].append(d)
    print(f"[twin] quantized {2 * src_exp} source experts in {time.time() - t0:.0f} s", flush=True)

    gen = torch.Generator().manual_seed(a.seed)
    for i in range(n_layers):
        fn = f"model-L{i:02d}.safetensors"
        tl = 0 if i < 3 else (1 if is_full(i) else 2)
        if a.link_from and os.path.exists(os.path.join(a.link_from, fn)):
            with open(os.path.join(a.link_from, "model.safetensors.index.json")) as f:
                other = json.load(f)["weight_map"]
            names = [n for n, s in other.items() if s == fn]
            dst = os.path.join(a.out, fn)
            if os.path.exists(dst):
                os.remove(dst)
            os.link(os.path.join(a.link_from, fn), dst)
            wm.update(dict.fromkeys(names, fn))
            print(f"[twin] linked {fn} ({len(names)} tensors)", flush=True)
            continue
        out = {}
        for n in layer_names(src, tl):
            if EXPERT_RE.match(n) or ".mlp.gate." in n:
                continue
            out[n.replace(f"model.layers.{tl}.", f"model.layers.{i}.", 1)] = src.get(n)
        if tl:
            g = src.get(f"model.layers.{tl}.mlp.gate.weight")
            b = src.get(f"model.layers.{tl}.mlp.gate.e_score_correction_bias")
            # random router rows with the source rows' scale: routing spreads over all experts
            std = float(g.float().std())
            out[f"model.layers.{i}.mlp.gate.weight"] = (torch.randn(n_exp, g.shape[1], generator=gen) * std).to(g.dtype)
            out[f"model.layers.{i}.mlp.gate.e_score_correction_bias"] = torch.zeros(n_exp, dtype=b.dtype)
            for e in range(n_exp):
                for proj, (q, s) in qexp[tl][e % src_exp].items():
                    out[f"model.layers.{i}.mlp.experts.{e}.{proj}.weight"] = q
                    out[f"model.layers.{i}.mlp.experts.{e}.{proj}.weight_scale"] = s
        save(a.out, fn, out, wm)

    mtp_layers = 0
    if a.mtp:
        m = Source(a.mtp)
        p78 = f"model.layers.{REAL_MTP_LAYER}."
        names = sorted(n for n in m.shard_of if n.startswith(p78))
        if not names:
            raise SystemExit(f"no {p78}* tensors under {a.mtp}")
        out = {}
        for n in names:
            new = n.replace(p78, f"model.layers.{n_layers}.", 1)
            em = EXPERT_RE.match(n)
            if em:
                if int(em.group(2)) >= n_exp:
                    continue
                q, s = fp8.quantize_rows(m.get(n))
                fp8.check_codes(q.view(torch.uint8))
                out[new] = q
                out[new + "_scale"] = s
            elif n.endswith("mlp.gate.weight") or n.endswith("mlp.gate.e_score_correction_bias"):
                out[new] = m.get(n)[:n_exp].contiguous()
            else:
                out[new] = m.get(n)
        save(a.out, "model-mtp.safetensors", out, wm)
        mtp_layers = 1

    cfg["num_hidden_layers"] = n_layers
    cfg["n_routed_experts"] = n_exp
    cfg["first_k_dense_replace"] = 3
    cfg["num_nextn_predict_layers"] = mtp_layers
    cfg["indexer_types"] = ["full" if is_full(i) else "shared" for i in range(n_layers)]
    cfg["mlp_layer_types"] = ["dense" if i < 3 else "sparse" for i in range(n_layers)]
    if "layer_types" in cfg:
        cfg["layer_types"] = ["deepseek_sparse_attention"] * n_layers
    cfg["quantization_config"] = quant_config()
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    total = sum(os.path.getsize(os.path.join(a.out, s)) for s in set(wm.values()))
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))}, f, indent=1)
    for aux in AUX_FILES:
        if aux != "README.md" and os.path.exists(os.path.join(a.small, aux)):
            shutil.copy2(os.path.join(a.small, aux), os.path.join(a.out, aux))
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(f"Performance twin of GLM-5.3: {n_layers} layers, {n_exp} experts per MoE layer, FP8 experts, "
                f"{'with' if mtp_layers else 'without'} the MTP layer. Made by make_twin.py for timing and memory "
                "tests only: the text it produces is meaningless.\n")
    full = [i for i in range(n_layers) if is_full(i)]
    print(f"[twin] {a.out}: {n_layers} layers ({len(full)} with a full indexer: {full}), {n_exp} experts, "
          f"{total / 1e9:.1f} GB, {len(wm)} tensors, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
