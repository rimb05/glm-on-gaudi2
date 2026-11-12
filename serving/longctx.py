#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Long-context check for the GLM-5.3 endpoint: correctness (three needles), time to first token and decode speed.

  longctx.py needle --tokens 128000 [--kind varied|survey] [--followup 300] [--out FILE.jsonl]

Request 1: a filler document of about --tokens tokens with three reference codes at 10 / 50 / 90 % depth and a
question asking for them (the cookbook's needle prompt), greedy, reasoning low, streamed. Records the prompt size the
server counted, the time to the first streamed token (= prefill time for an uncached prompt), the decode rate and
whether all three codes are in the answer.
Request 2 (--followup N): the same conversation plus a second question, N tokens of output. The long prefix comes
from the prefix cache, so its time to first token shows the cached-turn latency, and its decode rate is the decode
speed at that context length.

Standard library only. Environment: BASE (default http://127.0.0.1:8000), MODEL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "cookbook_gates"))
import glm5_gates as g  # noqa: E402

FOLLOWUP = (
    "Thank you. Now describe, in plain prose and in about {w} words, what kinds of events this document records "
    "and who carried them out. Do not use lists."
)


def run_stream(messages, max_tokens, effort="low", **kw):
    """One streamed chat request. Returns a dict with text, timings and token counts."""
    body = {"model": g.model(), "messages": messages, "max_tokens": max_tokens, "temperature": 0, **kw}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    t0 = time.time()
    first = last = None
    text, reasoning, usage, finish, n_events = [], [], None, None, 0
    for ev in g.stream("/v1/chat/completions", body, timeout=14400):
        now = time.time()
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            piece = d.get("content") or ""
            rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
            if piece or rpiece:
                first = first or now
                last = now
                n_events += 1
                text.append(piece)
                reasoning.append(rpiece)
            finish = ch.get("finish_reason") or finish
    out_tok = (usage or {}).get("completion_tokens", 0)
    gen_s = (last - first) if first and last else 0.0
    return {
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": out_tok,
        "ttft_s": round((first or time.time()) - t0, 3),
        "gen_s": round(gen_s, 3),
        "tok_s": round((out_tok - 1) / gen_s, 2) if gen_s > 0 and out_tok > 1 else None,
        "events": n_events,
        "finish": finish,
        "text": "".join(text),
        "reasoning": "".join(reasoning),
    }


def cmd_needle(a):
    if a.batch_id:
        # a fixed prompt (sizing calls and first line): a later run with the same id and size reuses the
        # server's prefix cache
        import random

        st = random.getstate()
        random.seed(a.batch_id)
        n = g.sentences_for(a.kind, a.tokens)
        random.seed(a.batch_id)
        prompt = g.needle_prompt(a.kind, n)
        random.setstate(st)
    else:
        n = g.sentences_for(a.kind, a.tokens)
        prompt = g.needle_prompt(a.kind, n)
    msgs = [{"role": "user", "content": prompt}]
    r1 = run_stream(msgs, a.max_tokens)  # the needle answer is always greedy
    hits = [c in r1["text"] for _, c in g.CODES]
    row = {
        "when": time.strftime("%FT%TZ", time.gmtime()),
        "kind": a.kind,
        "target": a.tokens,
        "prompt_tokens": r1["prompt_tokens"],
        "needles": f"{sum(hits)}/3",
        "ttft_s": r1["ttft_s"],
        "prefill_tok_s": round(r1["prompt_tokens"] / r1["ttft_s"], 1) if r1["prompt_tokens"] else None,
        "answer_tokens": r1["completion_tokens"],
        "answer_tok_s": r1["tok_s"],
        "answer_tail": r1["text"][-100:],
    }
    if a.followup:
        msgs2 = msgs + [
            {"role": "assistant", "content": r1["text"]},
            {"role": "user", "content": FOLLOWUP.format(w=max(50, int(a.followup * 0.6)))},
        ]
        kw = {}
        if a.temperature > 0:
            kw = {"temperature": a.temperature}
            if a.top_p is not None:
                kw["top_p"] = a.top_p
        r2 = run_stream(msgs2, a.followup, **kw)
        row.update({
            "turn2_prompt_tokens": r2["prompt_tokens"],
            "turn2_cached_tokens": r2["cached_tokens"],
            "turn2_ttft_s": r2["ttft_s"],
            "turn2_tokens": r2["completion_tokens"],
            "turn2_tok_s": r2["tok_s"],
            "turn2_sampling": f"temperature {a.temperature}, top_p {a.top_p}" if a.temperature > 0 else "greedy",
            "turn2_finish": r2["finish"],
            "turn2_head": (r2["text"] or r2["reasoning"])[:160],
        })
    line = json.dumps(row, ensure_ascii=False)
    print(line, flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")
    return 0 if all(hits) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("needle")
    n.add_argument("--tokens", type=int, required=True)
    n.add_argument("--kind", default="varied", choices=("varied", "survey"))
    n.add_argument("--max-tokens", type=int, default=2500)
    n.add_argument("--followup", type=int, default=0)
    n.add_argument("--out")
    n.add_argument("--batch-id", help="fixed prompt for this id (repeat runs hit the prefix cache)")
    n.add_argument("--temperature", type=float, default=0.0, help="sampling of the follow-up turn (0 = greedy)")
    n.add_argument("--top-p", type=float, default=None)
    a = ap.parse_args()
    return cmd_needle(a)


if __name__ == "__main__":
    sys.exit(main())
