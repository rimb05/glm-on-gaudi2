#!/usr/bin/env python3
"""Small serving checks against an OpenAI-compatible endpoint. Standard library only.

  evalkit.py greedy OUT.json [--max-tokens N]     fixed prompts, greedy, token ids + logprobs saved
  evalkit.py nll OUT.json [--tokens N]            teacher-forced log-probabilities of fixed texts (prompt_logprobs)
  evalkit.py longnll OUT.json [--chars N] [--skip N]   the same for one long text (later prefill chunks read the cache)
  evalkit.py compare A.json B.json                token agreement and logprob differences between two runs
  evalkit.py speed [--prompt-tokens N] [--max-tokens N] [--runs N] [--streams N]
                                                  decode tok/s and time to first token, greedy, streamed

Environment: BASE (default http://127.0.0.1:8000), MODEL (default: first model listed).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import statistics as st
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")

# API_KEY in the environment: sent as a bearer token with every request (for a server started with --api-key)
if os.environ.get("API_KEY"):
    _opener = urllib.request.build_opener()
    _opener.addheaders = [("Authorization", "Bearer " + os.environ["API_KEY"])]
    urllib.request.install_opener(_opener)
_MODEL = os.environ.get("MODEL")

PROMPTS = [
    "According to all known laws of aviation, there is no way a bee should be able to fly.",
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "In 1969, the first humans landed on the Moon. The mission was called",
    "The three primary colours are red,",
    "SELECT name, email FROM users WHERE",
    "Once upon a time, in a village by the sea, there lived an old fisherman who",
    "The derivative of x^2 with respect to x is",
    "Water boils at 100 degrees Celsius at sea level because",
    "To install the package, run the following command in your terminal:\n",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    "Q: What is 17 multiplied by 23?\nA:",
]

TEXTS = [
    "According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small "
    "to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans "
    "think is impossible.",
    "A pipeline splits a model by layers across machines. The first stage runs the embedding and the lower layers, "
    "then hands the hidden state of every token to the second stage, which runs the upper layers and the output "
    "head. Only one row per token crosses the network, so an ordinary link is enough, but the stages must stay in "
    "step: the second stage has to receive exactly the batch the first stage produced, in the same order.",
    "def quantize_rows(w):\n    amax = w.abs().amax(dim=1, keepdim=True)\n    scale = amax / 240.0\n    scale = "
    "torch.where(amax > 0, scale, torch.ones_like(scale))\n    return (w / scale).to(torch.float8_e4m3fn), scale\n",
    "The harbour master measured the eastern footbridge before dawn because the previous record was illegible. A "
    "visiting botanist catalogued three crates of slate after the autumn fair to settle a long dispute. The night "
    "clerk repaired an old brass sextant during the long drought for an upcoming exhibition.",
]


def model():
    global _MODEL
    if not _MODEL:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as r:
            _MODEL = json.loads(r.read())["data"][0]["id"]
    return _MODEL


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def cmd_greedy(a):
    out = []
    for p in PROMPTS:
        d = post("/v1/completions", {"model": model(), "prompt": p, "max_tokens": a.max_tokens, "temperature": 0,
                                     "logprobs": 5, "return_token_ids": True, "seed": 0})
        c = d["choices"][0]
        lp = c.get("logprobs") or {}
        out.append({"prompt": p, "text": c["text"], "token_ids": c.get("token_ids"),
                    "token_logprobs": lp.get("token_logprobs"), "top_logprobs": lp.get("top_logprobs")})
        print(f"{len(out):2d} {c['text'][:70]!r}", flush=True)
    json.dump({"kind": "greedy", "model": model(), "items": out}, open(a.out, "w"))
    print(f"saved {a.out}")


def cmd_nll(a):
    out = []
    for t in TEXTS:
        d = post("/v1/completions", {"model": model(), "prompt": t, "max_tokens": 1, "temperature": 0,
                                     "prompt_logprobs": 1, "return_token_ids": True})
        c = d["choices"][0]
        pl = c.get("prompt_logprobs") or d.get("prompt_logprobs")
        ids = c.get("prompt_token_ids") or d.get("prompt_token_ids")
        # prompt_logprobs: one dict per prompt token (None for the first); the actual token's entry is listed first
        lps, toks = [], []
        for i, ent in enumerate(pl or []):
            if not ent:
                continue
            if ids is not None and str(ids[i]) in ent:
                e = ent[str(ids[i])]
            else:
                e = next(iter(ent.values()))
            lps.append(e["logprob"])
            toks.append(ids[i] if ids is not None else None)
        nll = -sum(lps) / max(len(lps), 1)
        out.append({"n": len(lps), "nll": nll, "logprobs": lps, "token_ids": toks})
        print(f"text {len(out)}: {len(lps)} positions, NLL {nll:.4f}", flush=True)
    tot = sum(o["n"] for o in out)
    mean = sum(o["nll"] * o["n"] for o in out) / max(tot, 1)
    print(f"all: {tot} positions, NLL {mean:.4f}")
    json.dump({"kind": "nll", "model": model(), "items": out, "nll": mean, "positions": tot}, open(a.out, "w"))


def long_text(chars):
    """About `chars` characters of natural technical English, the same on every run on this host: Python's own
    language reference (pydoc_data.topics, part of the standard library), topics in sorted order."""
    from pydoc_data import topics

    parts, n = [], 0
    for k in sorted(topics.topics):
        parts.append(topics.topics[k])
        n += len(parts[-1]) + 2
        if n >= chars:
            break
    return "\n\n".join(parts)[:chars]


def cmd_longnll(a):
    """Teacher-forced log-probabilities of one long text. A prompt longer than one prefill chunk (4096 tokens) is the
    cheapest way to test what the KV cache gives back: every later chunk attends to the earlier chunks' entries as
    they were stored. Positions before --skip (the first chunk, which never reads the cache) are left out."""
    text = long_text(a.chars)
    d = post("/v1/completions", {"model": model(), "prompt": text, "max_tokens": 1, "temperature": 0,
                                 "prompt_logprobs": 1, "return_token_ids": True}, timeout=14400)
    c = d["choices"][0]
    pl = c.get("prompt_logprobs") or d.get("prompt_logprobs")
    ids = c.get("prompt_token_ids") or d.get("prompt_token_ids")
    lps, toks = [], []
    for i, ent in enumerate(pl or []):
        if not ent or i < a.skip:
            continue
        e = ent[str(ids[i])] if ids is not None and str(ids[i]) in ent else next(iter(ent.values()))
        lps.append(e["logprob"])
        toks.append(ids[i] if ids is not None else None)
    nll = -sum(lps) / max(len(lps), 1)
    print(f"long text: {len(pl or [])} prompt tokens, {len(lps)} positions from {a.skip} on, NLL {nll:.5f}")
    json.dump({"kind": "nll", "model": model(), "items": [{"n": len(lps), "nll": nll, "logprobs": lps,
                                                              "token_ids": toks}], "nll": nll,
               "positions": len(lps)}, open(a.out, "w"))


def cmd_compare(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    assert A["kind"] == B["kind"], (A["kind"], B["kind"])
    if A["kind"] == "greedy":
        same_seq, first_div, n_tok, n_same, dl = 0, [], 0, 0, []
        for x, y in zip(A["items"], B["items"]):
            ta, tb = x["token_ids"] or [], y["token_ids"] or []
            k = 0
            while k < min(len(ta), len(tb)) and ta[k] == tb[k]:
                k += 1
            same_seq += int(ta == tb)
            first_div.append(k)
            n_tok += min(len(ta), len(tb))
            n_same += k
            la, lb = x["token_logprobs"] or [], y["token_logprobs"] or []
            dl += [abs(la[i] - lb[i]) for i in range(k)]
        print(f"greedy: {same_seq}/{len(A['items'])} prompts token-identical; common prefix {n_same}/{n_tok} tokens; "
              f"first divergence at {first_div}")
        if dl:
            print(f"logprob |diff| on the common prefix: mean {st.mean(dl):.5f} max {max(dl):.5f} (n={len(dl)})")
        return 0 if same_seq == len(A["items"]) else 1
    d = []
    for x, y in zip(A["items"], B["items"]):
        assert x["token_ids"] == y["token_ids"], "different tokenization"
        d += [abs(p - q) for p, q in zip(x["logprobs"], y["logprobs"])]
    print(f"nll: A {A['nll']:.5f} B {B['nll']:.5f} diff {B['nll'] - A['nll']:+.5f}; per-position |dlogprob| mean "
          f"{st.mean(d):.5f} p99 {sorted(d)[int(len(d) * 0.99)]:.5f} max {max(d):.5f} (n={len(d)})")
    return 0


SPEED_WORDS = ("the harbour master measured the eastern footbridge before dawn because the previous record was "
               "illegible and a visiting botanist catalogued three crates of slate after the autumn fair ").split()


def stream_once(prompt, max_tokens):
    body = {"model": model(), "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "ignore_eos": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("text"):
                now = time.perf_counter()
                first = first or now
                last = now
    n = (usage or {}).get("completion_tokens", 0)
    ttft = (first - t0) if first else float("nan")
    # rate over the whole decode: tokens after the first one divided by the time since the first one arrived
    rate = (n - 1) / (last - first) if first and last and last > first and n > 1 else float("nan")
    return {"ttft": ttft, "tok_s": rate, "completion_tokens": n, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "total_s": time.perf_counter() - t0}


def cmd_speed(a):
    import random
    res = []
    for run in range(a.runs):
        prompts = []
        for s in range(a.streams):
            rng = random.Random(1000 * run + s + a.salt)
            # roughly 1.3 tokens per word for this filler; a fresh shuffle per run keeps the prefix cache out of it
            words = [rng.choice(SPEED_WORDS) for _ in range(int(a.prompt_tokens / 1.3))]
            prompts.append(" ".join(words) + "\nSummary:")
        with cf.ThreadPoolExecutor(a.streams) as ex:
            rs = list(ex.map(lambda p: stream_once(p, a.max_tokens), prompts))
        res.append(rs)
        print(f"run {run}: " + "; ".join(f"prompt {r['prompt_tokens']} ttft {r['ttft']:.2f}s {r['tok_s']:.1f} tok/s"
                                         for r in rs), flush=True)
    flat = [r for rs in res for r in rs]
    ok = sorted(r["tok_s"] for r in flat if not math.isnan(r["tok_s"]))
    mid = ok[len(ok) // 4: len(ok) - len(ok) // 4] or ok   # middle half: drops a compile run and outliers
    print(f"decode per stream: median {st.median(ok):.1f} tok/s = {1000 / st.median(ok):.2f} ms/token "
          f"(middle half {mid[0]:.1f}..{mid[-1]:.1f}, min {min(ok):.1f}, max {max(ok):.1f}); "
          f"TTFT median {st.median(r['ttft'] for r in flat):.2f} s; streams {a.streams}")


NAT_PROMPTS = [
    "Explain in detail how TCP congestion control works, covering slow start, congestion avoidance, fast "
    "retransmit and fast recovery.",
    "Write a Python class implementing an LRU cache with get and put in O(1), followed by a few unit tests.",
    "Describe the causes and consequences of the fall of the Western Roman Empire.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Explain how a B-tree index speeds up database queries, and when it does not help.",
    "Give a step-by-step derivation of the quadratic formula.",
]


def spec_counters():
    """(accepted, drafted) speculative tokens from /metrics, or None when there are no such counters."""
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
            txt = r.read().decode()
    except Exception:  # noqa: BLE001
        return None
    acc = drf = None
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        if "spec_decode_num_accepted_tokens_total" in line:
            acc = float(line.rsplit(" ", 1)[1])
        elif "spec_decode_num_draft_tokens_total" in line:
            drf = float(line.rsplit(" ", 1)[1])
    return (acc, drf) if acc is not None and drf is not None else None


def chat_stream_once(prompt, max_tokens, effort, temperature=0.0, top_p=None):
    body = {"model": model(), "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "stream": True, "stream_options": {"include_usage": True}}
    if top_p is not None:
        body["top_p"] = top_p
    if effort != "none":
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")
                       or ch[0].get("delta", {}).get("reasoning")):
                now = time.perf_counter()
                first = first or now
                last = now
    n = (usage or {}).get("completion_tokens", 0)
    rate = (n - 1) / (last - first) if first and last and last > first and n > 1 else float("nan")
    return {"ttft": (first - t0) if first else float("nan"), "tok_s": rate, "completion_tokens": n}


def cmd_natspeed(a):
    """Decode speed on natural chat prompts (what speculative decoding is tuned for), greedy, one stream."""
    c0 = spec_counters()
    rs = []
    for i, p in enumerate(NAT_PROMPTS[: a.prompts]):
        r = chat_stream_once(p, a.max_tokens, a.effort, a.temperature, a.top_p)
        rs.append(r)
        print(f"prompt {i}: {r['completion_tokens']} tokens, ttft {r['ttft']:.2f}s, {r['tok_s']:.1f} tok/s", flush=True)
    c1 = spec_counters()
    ok = sorted(r["tok_s"] for r in rs if not math.isnan(r["tok_s"]))
    extra = ""
    if c0 and c1 and c1[1] > c0[1]:
        extra = f"; speculative acceptance {(c1[0] - c0[0]) / (c1[1] - c0[1]):.3f} over {int(c1[1] - c0[1])} drafts"
    mode = "greedy" if a.temperature == 0 else f"temperature {a.temperature}, top_p {a.top_p}"
    print(f"natural text ({mode}), 1 stream: median {st.median(ok):.1f} tok/s = {1000 / st.median(ok):.2f} ms/token "
          f"(min {ok[0]:.1f}, max {ok[-1]:.1f}){extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("greedy"); g.add_argument("out"); g.add_argument("--max-tokens", type=int, default=48)
    n = sub.add_parser("nll"); n.add_argument("out")
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    s = sub.add_parser("speed")
    s.add_argument("--prompt-tokens", type=int, default=2000); s.add_argument("--max-tokens", type=int, default=256)
    s.add_argument("--runs", type=int, default=3); s.add_argument("--streams", type=int, default=1)
    s.add_argument("--salt", type=int, default=0)
    ln = sub.add_parser("longnll"); ln.add_argument("out")
    ln.add_argument("--chars", type=int, default=50000); ln.add_argument("--skip", type=int, default=4096)
    ns = sub.add_parser("natspeed")
    ns.add_argument("--max-tokens", type=int, default=400); ns.add_argument("--prompts", type=int, default=6)
    ns.add_argument("--effort", default="low")
    ns.add_argument("--temperature", type=float, default=0.0); ns.add_argument("--top-p", type=float, default=None)
    a = ap.parse_args()
    return {"greedy": cmd_greedy, "nll": cmd_nll, "compare": cmd_compare, "speed": cmd_speed,
            "natspeed": cmd_natspeed, "longnll": cmd_longnll}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
