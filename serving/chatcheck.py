#!/usr/bin/env python3
"""A few real chat prompts against the served model: prints answers, token counts and speed. Stdlib only.
   chatcheck.py [--max-tokens N] [--effort low|high|none]      env BASE, MODEL"""
import argparse, json, os, sys, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")

# API_KEY in the environment: sent as a bearer token with every request (for a server started with --api-key)
if os.environ.get("API_KEY"):
    _opener = urllib.request.build_opener()
    _opener.addheaders = [("Authorization", "Bearer " + os.environ["API_KEY"])]
    urllib.request.install_opener(_opener)
PROMPTS = [
    "What is 17 * 23? Answer with just the number.",
    "Write a Python function that returns the n-th Fibonacci number iteratively. Code only.",
    "In two sentences: why is the sky blue?",
    "A farmer has 3 fields. Each field has 4 rows of 12 apple trees. A storm destroys 9 trees. How many trees are left? Show the arithmetic briefly.",
    "Translate to French: 'The harbour master measured the eastern footbridge before dawn.'",
    "List the first five prime numbers greater than 50, comma separated.",
]


def model():
    m = os.environ.get("MODEL")
    if m:
        return m
    with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as r:
        return json.loads(r.read())["data"][0]["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    m = model()
    for p in PROMPTS:
        body = {"model": m, "messages": [{"role": "user", "content": p}], "max_tokens": a.max_tokens,
                "temperature": a.temperature}
        if a.effort != "none":
            body["chat_template_kwargs"] = {"reasoning_effort": a.effort}
        t0 = time.time()
        req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                d = json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"Q: {p}\n  ERROR {e}\n")
            continue
        dt = time.time() - t0
        msg = d["choices"][0]["message"]
        u = d.get("usage", {})
        txt = (msg.get("content") or "").strip()
        rc = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
        print(f"Q: {p}")
        if rc:
            print(f"  [reasoning, {len(rc)} chars] {rc[:160]!r}")
        print(f"  A: {txt[:600]!r}")
        print(f"  ({u.get('prompt_tokens')} prompt + {u.get('completion_tokens')} completion tokens, {dt:.1f} s, "
              f"{(u.get('completion_tokens') or 0) / dt:.1f} tok/s incl. prefill; finish {d['choices'][0].get('finish_reason')})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
