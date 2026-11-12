#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does the model still see what it wrote while the context was short, once the context is long?

  crosscheck.py [--prompt-tokens 3000] [--essay-words 900]      env BASE, MODEL, API_KEY

Below SPARSE_DECODE_FROM cached tokens (deploy/glm53.env) a decode step attends to every key and nothing reads the
indexer keys it writes; above it, attention goes through the indexer's top 2048 keys. A token generated in the
first regime is only visible in the second if its indexer key was really stored. Turn 1 makes the model state a
random code and then write an essay long enough to cross the threshold; turn 2 asks for the code. Standard library.
"""
import argparse
import json
import os
import re
import sys
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
if os.environ.get("API_KEY"):
    _o = urllib.request.build_opener()
    _o.addheaders = [("Authorization", "Bearer " + os.environ["API_KEY"])]
    urllib.request.install_opener(_o)
WORDS = ("the harbour master measured the eastern footbridge before dawn because the previous record was "
         "illegible and a visiting botanist catalogued three crates of slate after the autumn fair ").split()


def model():
    m = os.environ.get("MODEL")
    if m:
        return m
    with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as r:
        return json.loads(r.read())["data"][0]["id"]


def chat(messages, max_tokens):
    body = {"model": model(), "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"].get("content") or "").strip(), d.get("usage", {})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-tokens", type=int, default=3000)
    ap.add_argument("--essay-words", type=int, default=900)
    a = ap.parse_args()
    filler = " ".join(WORDS[i % len(WORDS)] for i in range(int(a.prompt_tokens * 0.78)))
    ask = (f"Background notes (ignore their content): {filler}\n\nTask: on the first line of your reply write one "
           f"random code of exactly 10 characters, capital letters and digits mixed, nothing else on that line. "
           f"Then write an essay of about {a.essay_words} words on the history of lighthouse keeping.")
    msgs = [{"role": "user", "content": ask}]
    first, u1 = chat(msgs, int(a.essay_words * 2.2) + 200)
    line = first.splitlines()[0].strip() if first else ""
    m = re.search(r"[A-Z0-9]{10}", line)
    code = m.group(0) if m else line
    msgs += [{"role": "assistant", "content": first},
             {"role": "user", "content": "What exactly was the code on the first line of your previous reply? "
                                         "Answer with the code only."}]
    second, u2 = chat(msgs, 200)
    ok = bool(code) and code in second
    print(f"turn 1: {u1.get('prompt_tokens')} prompt + {u1.get('completion_tokens')} generated tokens, code {code!r}; "
          f"turn 2 at {u2.get('prompt_tokens')} tokens answered {second[:60]!r}")
    print(("PASS" if ok else "FAIL") + " crosscheck: a code written below the sparse-decode threshold "
          + ("is" if ok else "is NOT") + " recalled above it")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
