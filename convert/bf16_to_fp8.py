#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a BF16 GLM-5.3 (glm_moe_dsa) checkpoint into the FP8 checkpoint vllm-gaudi serves on Gaudi 2.

  routed experts   model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight  BF16 -> F8_E4M3, one fp32 scale per output
                   channel (amax / 240) stored next to it as ...weight_scale [rows, 1]. Names stay per expert, so
                   vLLM's expert_params_mapping fuses them and --enable-ep-weight-filter can skip non-local experts
                   before reading them. The MTP layer's experts (layer index num_hidden_layers) are converted too.
  everything else  copied unchanged, dtype and bytes: attention, DSA indexer, shared experts, router, norms, dense
                   layers, embeddings, lm_head, the MTP layer's own projections.
  config.json      + quantization_config (compressed-tensors, float-quantized, weights channel / activations
                   dynamic per token, routed experts only).

It loads through HPUCompressedTensorsW8A8Fp8MoEMethod (the stock per-channel FP8 MoE op). Serve with
VLLM_SCALE_ADJUSTMENT=false: the codes already fit Gaudi 2's +-240 range (see fp8.py).

Workers (run the layer ranges in parallel; each writes its own shards and a manifest last):

  bf16_to_fp8.py --src SRC --dst DST --layers 3-12 [--threads 8]
  ...
  bf16_to_fp8.py --src SRC --dst DST --rest
  bf16_to_fp8.py --src SRC --dst DST --finalize        # after every worker has finished
  bf16_to_fp8.py --src SRC --list                      # print the expert layers found and exit

Then check the result with validate_fp8.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8  # noqa: E402

EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
EXPERT_TARGET = r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
AUX_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json",
             "chat_template.jinja", "LICENSE", "README.md")


class Source:
    """Lazy reader over every *.safetensors file of a directory."""

    def __init__(self, path: str):
        self.path = path
        self.shard_of: dict[str, str] = {}
        self.meta: dict[str, tuple[str, list[int]]] = {}
        self._open: dict[str, object] = {}
        for sp in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with open(sp, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            for name, m in hdr.items():
                if name in self.shard_of:
                    raise SystemExit(f"{name} is in two shards: {self.shard_of[name]} and {sp}")
                self.shard_of[name] = sp
                self.meta[name] = (m["dtype"], m["shape"])
        if not self.shard_of:
            raise SystemExit(f"no safetensors under {path}")

    def get(self, name: str) -> torch.Tensor:
        sp = self.shard_of[name]
        if sp not in self._open:
            self._open[sp] = safe_open(sp, framework="pt")
        return self._open[sp].get_tensor(name)

    def expert_layers(self) -> dict[int, int]:
        n: dict[int, int] = {}
        for name in self.shard_of:
            m = EXPERT_RE.match(name)
            if m:
                layer, e = int(m.group(1)), int(m.group(2))
                n[layer] = max(n.get(layer, 0), e + 1)
        return n

    def check_bf16(self) -> None:
        bad = [n for n in self.shard_of if n.endswith((".weight_scale", ".weight_scale_inv", ".weight_packed"))]
        if bad:
            raise SystemExit(f"{self.path} is already quantized ({bad[0]}): this converter needs the BF16 release")
        wrong = [n for n in self.shard_of if EXPERT_RE.match(n) and self.meta[n][0] != "BF16"]
        if wrong:
            raise SystemExit(f"routed expert {wrong[0]} is {self.meta[wrong[0]][0]}, expected BF16")


class Writer:
    """Buffers tensors and writes model-<tag>-NNNNN.safetensors (tmp + rename) once past the size limit."""

    def __init__(self, dst: str, tag: str, shard_gb: float):
        self.dst, self.tag = dst, tag
        self.limit = int(shard_gb * 2**30)
        self.buf: dict[str, torch.Tensor] = {}
        self.bytes = 0
        self.idx = 0
        self.written: dict[str, str] = {}
        for p in glob.glob(os.path.join(dst, f"model-{tag}-*.safetensors*")) + \
                glob.glob(os.path.join(dst, f"_manifest-{tag}.json")) + glob.glob(os.path.join(dst, f"_stats-{tag}.json")):
            os.remove(p)  # a restarted worker rewrites its own files

    def emit(self, name: str, t: torch.Tensor) -> None:
        self.buf[name] = t.contiguous()
        self.bytes += t.numel() * t.element_size()
        if self.bytes >= self.limit:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        fn = f"model-{self.tag}-{self.idx:05d}.safetensors"
        path = os.path.join(self.dst, fn)
        save_file(self.buf, path + ".tmp", metadata={"format": "pt"})
        os.replace(path + ".tmp", path)
        os.chmod(path, 0o644)
        self.written.update(dict.fromkeys(self.buf, fn))
        print(f"[fp8] wrote {fn} ({self.bytes / 1e9:.2f} GB, {len(self.buf)} tensors)", flush=True)
        self.buf, self.bytes = {}, 0
        self.idx += 1

    def close(self, stats: dict) -> None:
        self.flush()
        with open(os.path.join(self.dst, f"_stats-{self.tag}.json"), "w") as f:
            json.dump(stats, f, indent=1)
        with open(os.path.join(self.dst, f"_manifest-{self.tag}.json"), "w") as f:  # last: the completion marker
            json.dump(self.written, f)


def shard_files(path: str) -> list[str]:
    """Weight shards (and leftover temporary ones) in a directory; the index json is not a shard."""
    return sorted(os.path.basename(p) for pat in ("*.safetensors", "*.safetensors.tmp")
                  for p in glob.glob(os.path.join(path, pat)))


def parse_layers(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def convert_experts(src: Source, out: Writer, layers: set[int]) -> dict:
    n_exp = src.expert_layers()
    todo = sorted(set(n_exp) & layers)
    if not todo:
        raise SystemExit(f"no expert layers of {sorted(n_exp)} in the requested range")
    stats = {}
    for layer in todo:
        t0 = time.time()
        num = den = 0.0
        worst = 0.0
        for e in range(n_exp[layer]):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"
                w = src.get(name)
                if w.dtype != torch.bfloat16 or w.dim() != 2:
                    raise SystemExit(f"{name}: {w.dtype} {tuple(w.shape)}, expected a 2-D bf16 weight")
                q, scale = fp8.quantize_rows(w)
                fp8.check_codes(q.view(torch.uint8))
                w32 = w.to(torch.float32)
                d = fp8.dequantize(q, scale) - w32
                n2, d2 = float((d * d).sum()), float((w32 * w32).sum())
                num += n2
                den += d2
                worst = max(worst, (n2 / max(d2, 1e-30)) ** 0.5)
                out.emit(name, q)
                out.emit(name + "_scale", scale)
        rel = (num / max(den, 1e-30)) ** 0.5
        stats[str(layer)] = {"experts": n_exp[layer], "rel_err": rel, "worst_tensor_rel_err": worst}
        print(f"[fp8] layer {layer}: {n_exp[layer]} experts, weight error {rel * 100:.3f}% (worst tensor "
              f"{worst * 100:.3f}%), {time.time() - t0:.0f} s", flush=True)
    return stats


def convert_rest(src: Source, out: Writer) -> dict:
    n = nbytes = 0
    for name in sorted(src.shard_of):
        if EXPERT_RE.match(name):
            continue
        t = src.get(name)
        out.emit(name, t)
        n += 1
        nbytes += t.numel() * t.element_size()
    print(f"[fp8] rest: {n} tensors, {nbytes / 1e9:.2f} GB copied unchanged", flush=True)
    return {"tensors": n, "bytes": nbytes}


def quant_config() -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "quantization_status": "compressed",
        "fp8_dtype_flavor": "float8_e4m3fnuz",
        "config_groups": {
            "routed_experts": {
                "targets": [EXPERT_TARGET],
                "weights": {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "channel", "dynamic": False},
                "input_activations": {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "token",
                                      "dynamic": True},
                "format": "float-quantized",
            },
        },
        "ignore": [],
    }


def finalize(src: Source, dst: str) -> None:
    written: dict[str, str] = {}
    for mf in sorted(glob.glob(os.path.join(dst, "_manifest-*.json"))):
        with open(mf) as f:
            part = json.load(f)
        dup = set(part) & set(written)
        if dup:
            raise SystemExit(f"{len(dup)} tensors written by two workers, e.g. {sorted(dup)[0]}")
        written.update(part)
    if not os.path.exists(os.path.join(dst, "_manifest-rest.json")):
        raise SystemExit("the --rest worker has not finished")
    want = set()
    for name in src.shard_of:
        want.add(name)
        if EXPERT_RE.match(name):
            want.add(name + "_scale")
    missing, extra = sorted(want - set(written)), sorted(set(written) - want)
    if missing or extra:
        raise SystemExit(f"output does not match the source: {len(missing)} missing (e.g. {missing[:2]}), "
                         f"{len(extra)} unexpected (e.g. {extra[:2]})")
    files = set(written.values())
    stray = sorted(set(shard_files(dst)) - files)
    if stray:
        raise SystemExit(f"stray files in {dst}: {stray[:4]} (vLLM would not load them or would trip on them)")
    total = sum(os.path.getsize(os.path.join(dst, f)) for f in files)
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(written.items()))}, f, indent=1)
    with open(os.path.join(src.path, "config.json")) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = quant_config()
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for aux in AUX_FILES:
        if os.path.exists(os.path.join(src.path, aux)):
            shutil.copy2(os.path.join(src.path, aux), os.path.join(dst, aux))
    stats = {}
    for sf in sorted(glob.glob(os.path.join(dst, "_stats-*.json"))):
        with open(sf) as f:
            s = json.load(f)
        stats.update({k: v for k, v in s.items() if isinstance(v, dict) and "rel_err" in v})
    n_exp = src.expert_layers()
    if set(stats) != {str(k) for k in n_exp}:
        raise SystemExit(f"expert layers without statistics: {sorted({str(k) for k in n_exp} - set(stats))}")
    with open(os.path.join(dst, "conversion.json"), "w") as f:
        json.dump({"source": os.path.abspath(src.path), "format": "fp8 e4m3 per output channel, scale = amax / 240",
                   "expert_layers": sorted(n_exp), "tensors": len(written), "bytes": total,
                   "weight_rel_err_by_layer": {k: stats[k]["rel_err"] for k in sorted(stats, key=int)}}, f, indent=1)
    errs = [stats[k]["rel_err"] for k in stats]
    print(f"[fp8] finalized {dst}: {total / 1e9:.1f} GB, {len(written)} tensors in {len(files)} shards, "
          f"{len(n_exp)} expert layers, weight error mean {100 * sum(errs) / len(errs):.3f}% max {100 * max(errs):.3f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst")
    job = ap.add_mutually_exclusive_group(required=True)
    job.add_argument("--layers")
    job.add_argument("--rest", action="store_true")
    job.add_argument("--finalize", action="store_true")
    job.add_argument("--list", action="store_true")
    ap.add_argument("--shard-gb", type=float, default=5.0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    src = Source(a.src)
    src.check_bf16()
    if a.list:
        n = src.expert_layers()
        print(json.dumps({"expert_layers": sorted(n), "experts_per_layer": sorted(set(n.values())),
                          "tensors": len(src.shard_of)}))
        return 0
    if not a.dst:
        raise SystemExit("--dst is required")
    os.makedirs(a.dst, exist_ok=True)
    if a.finalize:
        finalize(src, a.dst)
        return 0
    if a.rest:
        out = Writer(a.dst, "rest", a.shard_gb)
        out.close(convert_rest(src, out))
        return 0
    out = Writer(a.dst, "L" + a.layers.replace(",", "_"), a.shard_gb)
    out.close(convert_experts(src, out, parse_layers(a.layers)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
