# Measurement logs

Raw output of the runs behind the figures in the main README. Times are UTC. Model answers in these files are
test output (reference codes, summaries of a synthetic filler document).

| File | What it is |
| --- | --- |
| `bench.tsv` | one line per ad-hoc launch (`bin/bench2.sh`, `serving/bench1.sh`): time, name, layout, load time, weights per card, KV pool size, decode speed at about 2K tokens, the switches and the extra `vllm serve` arguments. The first lines are timing twins (`twin6`, `twin10`, `twin14`, `twin26`: the number is the layer count), the later ones the real model. Lines before 16:18 on the first day were taken under the `schedutil` CPU governor |
| `longctx.jsonl` | `serving/longctx.py needle` runs during the long-context work, in order: 40K and 128K with the cookbook's sparse attention module (first touch of each shape, then warm), the same with the revisions of this repository, then 256K and the 700K runs (598 s on first touch, then 416, 327 and 299 s as the recipes warmed) |
| `run256k-1.log`, `run700k-2.log` ... `run700k-5.log` | the output of single runs of that series |
| `final700k.log` | five consecutive runs of one 700K prompt on the 720K profile: one ingest (363 s), then four turns answered from the prefix cache |
| `run1m-1.log`, `longctx-1m.jsonl`, `run1m-1.mem` | the first 1M-token run (1,015 s, every shape new) and the memory `hl-smi` reported for card 0 during it |
| `gates-prod/gates.log` | the gaudi-cookbook's serving gates on the 720K profile |
| `gates-1m/` | full validation of the 1M profile (`serving/validation/validate1m.sh`): warm-up, `crosscheck.py`, the gates, two 700K runs and three 1M runs |
| `gates-720k/` | a shorter check of the 720K profile after the 8-bit cache work (`serving/validation/validate720.sh`) |

Notes for reading them:

- **The `mtp` gate prints FAIL** in every gate log. Its threshold (acceptance of at least 0.70) was set for one
  draft token per step. This deployment proposes two; per draft, acceptance at the default sampling is 0.65 to
  0.68 (0.83 with one draft). More tokens are accepted per step with two drafts, which is what counts for speed.
- **The `tools` gate** misses one or two of 32 cases in some runs, always `required-history` at the default
  sampling: a case that forces a tool call where none fits. Both profiles miss it, so it is not the 8-bit cache.
- **`gates-720k/long.jsonl`**: the decode rates of the follow-up turns in this file (21 to 49 tok/s) were taken
  while another client was generating on the same endpoint. They show what sharing the decode steps does; they
  are not single-user figures. Ingest time, cached-turn latency and the reference codes are unaffected.
- The per-token dumps behind the agreement and log-likelihood figures (`*.greedy.json`, `*.nll.json`,
  `long-*.json`, about 5 MB) are not part of the repository.
