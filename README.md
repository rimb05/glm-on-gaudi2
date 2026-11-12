# GLM-5.3 (744B) on two Gaudi 2 servers

GLM-5.3, a 744-billion-parameter mixture-of-experts model, served from two 8-card Intel Gaudi 2 servers as one
OpenAI-compatible endpoint, with a 720,000-token context window (1,047,000 tokens with an 8-bit KV cache) and 55
to 77 tokens per second for one user.

| | |
| --- | --- |
| Model | GLM-5.3 complete: 78 layers plus its MTP draft layer. Routed experts in FP8 (8-bit), everything else bf16; 773 GB on disk |
| Hardware | 2 servers x 8 Gaudi 2 cards (16 x 96 GiB), joined through the cards' own scale-out ports and one switch |
| Layout | 2 pipeline stages x 8 cards, tensor and expert parallelism inside each stage |
| Context | 720,000 tokens with the cache as the model computes it, or 1,047,000 tokens with the cache stored as 8-bit codes |
| Decode speed, one user | 70-77 tok/s at 2K tokens of context, 68 at 40K, 56-58 at 700K, 50-55 at 1,038K |
| Time to first token | 0.34 s at 2K. A new 700K-token prompt is read in 5-6 minutes, a 1M-token prompt in 11-13. A cached conversation of that size answers after 3 to 8 s |
| Software | vLLM + vllm-gaudi + the gaudi-cookbook's GLM-5 port, with the runtime modules and patches of this repository on top |

This repository holds the code and a full account of the method: what had to be built, why, how each piece works,
what it measured, and which mistakes cost time. It is written so that someone with this kind of hardware can do it
again, and so that someone bringing another large model to Gaudi can see which problems appear and how they were
solved. No model weights are included.

## Contents

1. [Goal and result](#1-goal-and-result)
2. [Hardware and software](#2-hardware-and-software)
3. [Background: the model and the platform](#3-background-the-model-and-the-platform)
4. [How a request flows through the two servers](#4-how-a-request-flows-through-the-two-servers)
5. [What is new here, and what comes from other projects](#5-what-is-new-here-and-what-comes-from-other-projects)
6. [How each piece works](#6-how-each-piece-works)
7. [Doing it yourself, step by step](#7-doing-it-yourself-step-by-step)
8. [The two profiles: 720K and 1M](#8-the-two-profiles-720k-and-1m)
9. [Measurements](#9-measurements)
10. [Where the time, the memory and the power go](#10-where-the-time-the-memory-and-the-power-go)
11. [Pitfalls, and how each was solved](#11-pitfalls-and-how-each-was-solved)
12. [Tried and dropped](#12-tried-and-dropped)
13. [Tests and tools](#13-tests-and-tools)
14. [Operating it](#14-operating-it)
15. [Switch reference](#15-switch-reference)
16. [Repository layout](#16-repository-layout)
17. [Credits and licences](#17-credits-and-licences)
18. [Ideas not done](#18-ideas-not-done)

---

## 1. Goal and result

The goal: the whole of GLM-5.3 on sixteen Gaudi 2 cards in two servers, with

- weights at 8 bits or better,
- a context window of at least 700,000 tokens, then 1,000,000,
- single-user decode speed as close to 60 tokens per second as the hardware allows, at every context length,
- all traffic between the servers on the cards' own network, not on the hosts' Ethernet.

The starting point was the gaudi-cookbook's GLM-5 port, which runs the GLM-5 family on Gaudi 2, has a fix for
pipeline parallelism across hosts, and validates contexts up to 128K tokens. Three things stood between that and
the goal:

- a decode step crossed between the two servers as CPU tensors, which cost 2.4 ms per token with the hosts at
  full clock and 9 ms at their default CPU governor (measured on a 26-layer twin);
- speculative decoding with the model's own draft layer did not run across pipeline stages;
- the long-context module was built for a 131K window. Memory and time per prompt chunk grew too fast beyond it.

Where it ended:

| Context | First token, new prompt | First token, cached conversation | Decode, 720K profile | Decode, 1M profile |
| --- | --- | --- | --- | --- |
| 2K | 0.34 s | - | 70-77 tok/s | 61-70 |
| 40K | 6 s | 0.6-0.8 s | 68 | 62-64 |
| 128K | 31 s | 0.8 s | - | 57 |
| 700K | 5-6 min | 2.5-3.6 s | 56-58 | 55-57 |
| 1,038K | 11-13 min | 6.6-8.4 s | does not fit | 50-55 |

60 tok/s is reached and passed at short and medium contexts. At 700K it is 56-58 and at 1M 50-55;
[section 10](#10-where-the-time-the-memory-and-the-power-go) accounts for the remaining milliseconds.

Figures that depend on the weights themselves (conversion error, log-likelihoods, gate scores, draft acceptance)
come from the GLM-5.3 checkpoint that was served during this work. Another checkpoint of the same architecture
will give slightly different values. Speed and memory depend only on the architecture.

---

## 2. Hardware and software

### 2.1 Hardware

- **Two 8-card Gaudi 2 servers**, each with two Xeon Scalable sockets, 256 GB of RAM and NVMe storage. Host
  memory and core count are not critical (the conversion ran 12 workers of 8 threads within 200 GB), but decode
  speed depends on the hosts' core frequency
  ([section 6.2](#62-performance-twins-finding-the-bottlenecks-without-the-weights)).
- **16 Gaudi 2 cards** (HL-225), 96 GiB each, of which PyTorch can use 94.62 GiB. On a card, matrix products run
  on the matrix engine (MME) and everything else (elementwise operations, gathers, top-k) on 24 programmable
  cores (TPC).
- **Card-to-card networking** is part of the cards. Inside a server the eight cards are linked directly. Each
  server also exposes 24 scale-out ports of 100G through six QSFP-DD 400G connectors. Both servers plug into one
  400G data-centre switch, with the ports broken out to 100G and set up as a lossless RoCEv2 network (priority
  flow control, MTU 8000). This is "the fabric" in this document.
- **IP on the fabric.** Scale-out ports can carry an IPv4 address, and the servers then also reach each other by
  TCP over them. One such port per server is the "fabric interface" of the settings; the stage rendezvous and
  the prefill link use it.
- **Disk.** 1.5 TB for the bf16 download and 773 GB for the converted checkpoint on the server that converts,
  773 GB on the other. The timing twins and test models of the bring-up take about 250 GB more.

### 2.2 Software base

| | |
| --- | --- |
| Gaudi software | 1.24.1-482 on both hosts; PyTorch 2.11.0 from Habana's container, pinned by digest |
| vLLM | commit `ced6857a` |
| vllm-gaudi | `releases/v0.30.0` at `058f107` (the plugin that runs vLLM on Gaudi) |
| gaudi-cookbook | `thediymaker/gaudi-cookbook` at `1540263`: fixes for vllm-gaudi and a GLM-5 port (`ports/glm-5-mxfp4`) |
| This repository | converter, runtime modules (`glm53_hpu/`), engine patches (`patches/`), deployment, tests |

Everything is pinned by commit or digest in `image/`. The patches of this repository check the exact source text
they expect and stop if it has drifted, so a newer vLLM or vllm-gaudi needs them revisited.

---

## 3. Background: the model and the platform

### 3.1 The model

Figures from `meta/glm-5.3.config.json`:

| | |
| --- | --- |
| Architecture | `GlmMoeDsaForCausalLM`: 78 decoder layers, hidden size 6144, 64 attention heads, vocabulary 154,880, positions up to 1,048,576 |
| Feed-forward | layers 0-2 dense (width 12,288); layers 3-77 mixture of experts: 256 routed experts per layer (3 matrices of 6144 x 2048 each, 37.7M parameters), 8 chosen per token, plus 1 shared expert |
| Attention | MLA (multi-head latent attention): the cache does not hold keys and values per head but one **latent** of 576 values per token and layer (512 compressed-KV values + 64 rotary values) |
| Sparse attention | DSA: an **indexer** scores every earlier token and each query attends only to its top 2,048 keys. 21 layers own an indexer (layers 0, 1, 2, 6, 10, ..., 74: "full"); the three layers after each reuse its selection ("shared"). An indexer has 32 heads of 128 dimensions and stores one 128-value key per token |
| MTP | one extra layer (index 78) that predicts the token after the next one: a built-in draft model for speculative decoding |

The name says 744B; the checkpoint holds 753 billion parameters, which includes the MTP layer.

Three properties of this architecture decide everything that follows:

1. **The weights are almost all experts.** 75 layers x 256 experts x 37.7M parameters. In bf16 the checkpoint is
   1.5 TB; with 8-bit experts it is 773 GB, 48 GiB per card across 16 cards.
2. **The cache is small per token but replicated.** A latent is 1,152 bytes in bf16, but tensor parallelism keeps
   the full latent cache on every card of a stage, so the context length is set by what fits on one card next to
   its share of the weights.
3. **Long contexts cost a scan.** With DSA every decode step has to score all cached keys in 21 layers (plus the
   MTP layer) to find the top 2,048. Done naively that grows linearly with the context.

### 3.2 Two facts about the Gaudi software stack

- **Lazy mode and graphs.** PyTorch operations on a Gaudi card are not executed one by one. They are collected
  and compiled into a *recipe* that is then launched. vllm-gaudi wraps the model's forward so that every shape
  (batch size, query length, number of context blocks) is traced once and its recipe replayed afterwards. A
  collective operation between cards (an all-reduce) ends a recipe, and a decode step of this model contains 156
  of them (two per layer). The duration of a step is set by how many small operations and launches it contains,
  not by arithmetic: one card alone is only about 20 % faster per layer than eight cards sharing the layer.
- **Buckets.** Shapes are padded to a fixed list of sizes ("buckets") so that the number of recipes stays finite.
  The first request of each bucket compiles for seconds; recipes are cached on disk
  (`/workspace/glm53/recipe-cache`), so a restart does not pay again.

Most of the performance work below is about removing launches, host round trips and small operations from the
decode step, and most of the long-context work is about memory that a lazy graph keeps alive.

---

## 4. How a request flows through the two servers

```
 client (any OpenAI-compatible program)
      │  HTTP (SSH forward to node 0's loopback, or a private address)
      ▼
 node 0: vllm serve ── API server, tokenizer, scheduler, engine loop
      │
      │  per step: one batch = up to 4 sequences; for one user 3 rows (the next token + 2 drafts to verify)
      ▼
 ┌──────────────────────── stage 0: node 0, 8 cards ─────────────────────────┐
 │ embedding, layers 0-41                                                    │
 │ every layer: attention heads and dense weights split over the 8 cards     │
 │ (tensor parallel), each card holds 32 of the 256 experts (expert parallel)│
 │ two all-reduces per layer between the 8 cards                             │
 │ every card: 48.1 GiB of weights + the full latent cache of layers 0-41    │
 └──────────────┬────────────────────────────────────────────────────────────┘
                │ hidden state + residual, card r → card r
                │   decode steps:   HCCL all-reduce on a 2-card group, over the fabric (inside the graphs)
                │   prefill chunks: TCP on the fabric interface, 1/8 of the payload per rank
                ▼
 ┌──────────────────────── stage 1: node 1, 8 cards ─────────────────────────┐
 │ layers 42-77, final norm, LM head → logits                                │
 │ rejection sampler: which drafts are accepted, next token                  │
 │ MTP layer (the draft model): proposes the next 2 tokens                   │
 │ every card: 44.0 GiB of weights + 1.6 GiB draft model + latent cache      │
 └──────────────┬────────────────────────────────────────────────────────────┘
                ▼
        sampled tokens and new drafts back to the engine on node 0
```

- **Pipeline parallelism (2 stages)** splits the layers between the servers. The split must begin on a layer
  that owns an indexer, and stage 1 also carries the LM head and the MTP layer; 42 + 36 keeps the larger stage
  as small as possible.
- **Tensor parallelism (8 ranks per stage)** splits each layer across the eight cards of a server.
- **Expert parallelism** gives each card 32 whole experts instead of a slice of all 256, and
  `--enable-ep-weight-filter` loads only those from disk.
- **Prefill** (reading a prompt) runs in chunks of 4,096 tokens. Chunk *k+1* is on stage 0 while chunk *k* is on
  stage 1. **Decode** (generating) runs one step at a time through both stages.
- **Speculative decoding**: each step verifies the two tokens the MTP layer proposed in the previous step and
  proposes two more. A step therefore yields 1 to 3 tokens, 2.3-2.45 on average.

Why this layout and not another (from runs on timing twins, [section 6.2](#62-performance-twins-finding-the-bottlenecks-without-the-weights), and the memory arithmetic of [section 10.3](#103-memory-on-a-stage-0-card-9462-gib-usable)):

| Layout | Cache | Verdict |
| --- | --- | --- |
| 2 stages x 8 cards | about 720K tokens in bf16 | chosen: fastest prefill, enough context |
| 1 stage x 16 cards | half of that | every card would hold the latent cache of all 78 layers |
| 4 stages x 4 cards | 1M tokens would fit in bf16 | half the cards per prompt chunk: prefill about 2x slower by estimate; a four-way scan split is worse at long context; not run on the real model |
| 16 stages x 1 card | far more: each card holds the cache of 5 layers only | about 20 % faster per layer, but 15 hand-overs per step and prefill would need a different scheduler; not pursued |

---

## 5. What is new here, and what comes from other projects

### 5.1 Used from other projects

| Project | What is used | How |
| --- | --- | --- |
| **vLLM** | engine, scheduler, prefix cache, OpenAI server, the `mtp` speculative decoding method, pipeline / tensor / expert parallelism, the expert weight filter | built unmodified into the image; three source patches to `EngineCore` are applied at container start |
| **vllm-gaudi** | the Gaudi backend: graph capture per shape, bucketing, paged MLA attention, the per-channel FP8 MoE operator, its (unused) indexer-key caches | built unmodified; this repository's hooks are appended at container start |
| **gaudi-cookbook** | its fix set for vllm-gaudi v0.30.0, pipeline parallelism across hosts (`pipeline-parallel`), and the GLM-5 port: 14 patches and the `glm5_hpu` package (sparse attention module, MTP fixes, three decode levers), plus its serving gates | applied in the image as published. Its `dsa.py` is replaced at start by the revision in this repository; its gates are copied into `serving/cookbook_gates/` with API-key support added |
| **Wright** | three design ideas, no code | see [section 17](#17-credits-and-licences) |

The cookbook's three decode levers are used as they are, and their greedy output is bit-identical with the
levers off:

- `GLM_ROUTER_FAST=1`: the MoE router's grouped top-k has a group stage that is a no-op for GLM-5.x (one group);
  skipping it saves about 25 us per MoE layer.
- `GLM_SPEC_DEVICE_REJECT=1`: the rejection sampler's accept/reject walk stays on the device instead of copying
  to the host, which was a device sync in the middle of every step.
- `GLM_EARLY_DRAFT=1`: the draft model is launched right after the sampler with indices computed on the device,
  so target forward, sampler and draft run back to back.

### 5.2 Built here

Each entry is something that vLLM, vllm-gaudi and the cookbook at the pinned commits did not do. The numbers are
what the piece moved when it was introduced.

| Piece | Problem | Effect | Where |
| --- | --- | --- | --- |
| **FP8 expert conversion with an independent validator** | the bf16 master is 1.5 TB; the published 8-bit release uses block-wise scales that the Gaudi FP8 MoE operator does not take | 773 GB, 48 GiB per card, every tensor checked by a second implementation | `convert/` |
| **Performance twins** | the real checkpoint takes hours to fetch and minutes to load; bottlenecks had to be found without it | found the CPU governor, the stage link and the engine loop problems before the weights existed | `convert/make_twin.py` |
| **Card-to-card stage link** | a decode step crossed between the servers as CPU tensors: a device sync on each side | 39.5 → 62.6 tok/s on a 26-layer twin (84.5 → 105.9 with the hosts at full clock) | `glm53_hpu/fastlink.py` |
| **Engine loop patches** | with pipeline stages vLLM dispatched empty steps and slept; speculative decoding failed to compile across stages; the first fix for that lost prefill overlap | 1.1 ms per token and extra stage back; MTP works across two servers (39.7 → 55.6 tok/s); prefill at 40K back from 4,130 to 7,060 tok/s | `patches/engine_*.py` |
| **Two chained MTP drafts in one device graph** | the stock path for a second draft token costs 12 ms per step and cancels its gain | 62.7 → 70-77 tok/s at 2K | `glm53_hpu/draft_chain.py` |
| **Long-context sparse attention** | the cookbook module was built for a 131K window: its memory and its time per prompt chunk grew too fast beyond it, and every card scanned all keys | contexts of 720K and 1M tokens at all; one indexer scan from 547 to 370 us (22 per decode step); with the engine patch, 128K ingest from 102 s to 31 s | `glm53_hpu/dsa.py` |
| **Decode-step metadata in numpy** | Python lists rebuilt per step: 8 ms at 700K | 54.6 → 56-58 tok/s at 700K | `glm53_hpu/fastmeta.py` |
| **8-bit latent KV cache** | a bf16 cache for 1M tokens does not fit next to the weights | 1,047,552-token pool on the same cards, 0.6 % error per latent, 8-10 % slower decode | `glm53_hpu/kvfp8.py`, `LatentInt8Cache` in `dsa.py` |
| **Bucket sets for large windows** | every new shape compiles for seconds and every padded token costs compute; a 720K or 1M window needs a shape list that balances the two | 449 shapes for the 720K window and 559 for 1M, at most 16K tokens of padding on a prefill | `tools/make_buckets.py` |
| **Deployment** | two containers on two servers are one engine and fail as one | launcher, pair watchdog, service units, warm-up | `deploy/` |
| **Cache-aware tests** | the usual checks never read the KV cache | `longnll`, `crosscheck`, long-context needles, card-level benchmarks | `serving/`, `tools/` |

How the single-user speed at 2K tokens developed on the real model:

| Step | tok/s |
| --- | --- |
| Real model on two servers with the card-to-card link, no speculation | 39.7 |
| + MTP, 1 draft per step (acceptance 0.83) | 55.6 |
| + the cookbook's three decode levers | 62.7 |
| + two chained drafts and numpy step metadata | 70-77 on natural text |

And decode at 700K tokens:

| Step | tok/s |
| --- | --- |
| First run: gather scan, one draft, tracing on | 44 (about 48 without tracing) |
| Rank-split and in-place pool scan, one draft | about 50 |
| Two chained drafts, cheaper scoring and merge | 54.6 |
| Numpy step metadata | 56-58 |

---

## 6. How each piece works

Every feature has an environment switch and is off by default; `deploy/serve_node.sh` turns on the deployed set.
[Section 15](#15-switch-reference) lists all switches.

### 6.1 FP8 conversion (`convert/`)

Start from the bf16 release, `zai-org/GLM-5.3-BF16` (291 files, 1.5 TB). The 8-bit release, `zai-org/GLM-5.3`,
cannot be served as it is: it stores almost every tensor as E4M3 with one scale per 128 x 128 block, while
vllm-gaudi's FP8 MoE operator takes one scale per output channel, and Gaudi 2's matrix engine uses an E4M3
variant whose largest finite value is 240, not 448. Any checkpoint with GLM-5.3's tensor names and shapes (a
fine-tune, for example) converts the same way.

The format, defined in `convert/fp8.py`:

- Only the routed experts are converted (`model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight`, including
  the MTP layer's). Attention, indexer, shared experts, router, norms, embeddings and LM head stay bf16, byte for
  byte.
- Each expert matrix gets one fp32 scale **per output row**: `scale = max|w| / 240`. The code is the nearest E4M3
  value to `w / scale`, ties to even. With 240 as the top of the range no code needs the top exponent, so the
  stored bytes mean the same in both E4M3 conventions and the server runs with `VLLM_SCALE_ADJUSTMENT=false`.
- Two conservative byte rules, taken over from the cookbook's FP8 codecs because the hardware's handling of
  these codes is not documented: no subnormal codes and no negative zero.
- Tensor names stay per expert, so vLLM's expert loader fuses them and the weight filter can skip experts that
  belong to other cards before reading them. `config.json` gets a `compressed-tensors` quantization block, and
  the checkpoint loads through vllm-gaudi's stock per-channel FP8 MoE operator.

Tools:

- `bf16_to_fp8.py`: parallel workers by layer range, then `--finalize` writes the index. `convert_full.sh` runs
  12 workers and the validation.
- `validate_fp8.py`: an **independent** check. It recomputes every scale and every code byte with a second
  implementation that searches the explicit table of the 120 representable magnitudes instead of using
  PyTorch's cast, and compares every non-expert tensor byte for byte with the source.
- `tests/test_fp8.py`: the two quantizers agree.

Measured: conversion in 12 minutes on one server (12 workers x 8 threads), 294 shards, 773 GB, mean weight error
2.65 % against bf16, full validation passed.

### 6.2 Performance twins: finding the bottlenecks without the weights

A 1.5 TB download takes hours. `convert/make_twin.py` builds a "twin" in minutes: N layers with GLM-5.3's exact
per-layer dimensions and 256 FP8 experts per layer, made by repeating the layers of a 3-layer public test model,
optionally with the real MTP layer appended. Its text is meaningless, but per-layer decode time, pipeline
overhead and memory per token are the real model's. Every twin run is a line in `results/bench.tsv`.

What the twins showed:

| Finding | Evidence |
| --- | --- |
| **CPU frequency governor.** The servers idled in `schedutil`. Decode is bound by host-side graph launches, so core frequency is decode speed | 26-layer twin, 2 stages x 8 cards: 62.6 → 105.9 tok/s with `performance` |
| **Stage link.** The cookbook moves a stage's output to the next stage as CPU tensors over gloo (TCP): a device sync on the sender, another on the receiver | across the two servers, 39.5 → 62.6 tok/s with the card-to-card link under `schedutil` (9 ms per token), 84.5 → 105.9 under `performance` (2.4 ms) |
| **Empty engine steps.** With more than one pipeline stage vLLM dispatches an empty step and sleeps 1 ms per token at one stream | about 1.1 ms per token per extra stage |
| **Per-layer cost is launches, not arithmetic** | one card alone is only about 20 % faster per layer than eight with tensor and expert parallelism. Removing the per-layer graph cut for decode did not help a stage, and removing it for prefill makes compiles take minutes |
| **Layout** | the table in [section 4](#4-how-a-request-flows-through-the-two-servers) |

Also verified on the 0.6B test model before the real one: bf16 and FP8 experts agree within bf16 rounding, and
two stages across the two servers give bit-identical output to eight cards on one server.

### 6.3 The stage link (`glm53_hpu/fastlink.py`, hooks in `patches/fastlink_hooks.py`)

The cookbook's pipeline fix hands a stage's output to the next stage as CPU tensors over gloo. The sender must
copy from the card to the host, which forces the card to finish first; the receiver copies back. Measured on a
26-layer twin across the two servers: 9 ms per token with the hosts in `schedutil`, 2.4 ms in `performance`.

With `GLM_PP_FASTLINK=1`, forwards of at most 64 tokens (decode steps) use a collective instead:

```
stage i     ... last layer → all_reduce([hidden | residual], group {card r of stage i, card r of stage i+1})
stage i+1   all_reduce(zeros, same group) → [hidden | residual] → first layer ...
```

- At start-up one HCCL process group is created per pair of neighbouring cards (card *r* of node 0 with card *r*
  of node 1) and checked with a tiny all-reduce.
- The receiver contributes zeros, so the all-reduce is a point-to-point send. Both calls are ordinary lazy device
  operations inside each stage's compiled graph. No stage copies to the host, and node 1's host can launch its
  whole forward as soon as the step is scheduled; the device work then cascades from node 0's cards to node 1's
  through the scale-out ports.
- The flag travels as a keyword argument of the wrapped forward, so the fast and the gloo version of the same
  shape are compiled as different graphs.
- Prefill chunks keep the cookbook's link: it already splits a large payload into eighths, one per rank, and is
  not the bottleneck there.
- `tools/hccl_lat.py` and `tools/hccl_chain.py` measured the primitive first: 30-80 us per hop.

`fastlink.install_runner` is also where the other modules of this repository are installed into vllm-gaudi's
model runner (`draft_chain`, `fastmeta`, `kvfp8`, `debug_spec`).

### 6.4 The engine loop (`patches/engine_*.py`)

Three source patches to vLLM's `EngineCore`, each idempotent, each checked against the exact source text it
expects (exit code 2 if the source has drifted), each behind a switch:

- **`engine_skip_empty.py`** (`GLM_ENGINE_SKIP_EMPTY=1`): with a batch queue, vLLM keeps scheduling until the
  queue is full. At one stream the second schedule is empty, is still sent to every worker, and the outer loop
  then sleeps 1 ms. The patch does not dispatch an empty schedule while a batch is in flight and skips the sleep
  after a step that produced output.
- **`engine_single_batch.py`** (`GLM_ENGINE_SINGLE_BATCH=1`): one batch in flight, the plain engine loop. This
  made speculative decoding work across stages but gave up prefill overlap (4,130 tok/s at 40K against about
  7,000). Kept for diagnosis, not used in the deployment.
- **`engine_wave.py`** (`GLM_ENGINE_PREFILL_WAVE=1`, deployed): keeps the queue but restricts it. A second batch
  is scheduled only while every running request is still prefilling; as soon as anything decodes, the loop waits
  for the batch in flight. Decode steps therefore run exactly like the plain loop (schedule, run, process the
  output, fetch drafts), and prompt chunks still overlap across the stages. It also turns off the cookbook's
  per-step cap on running requests.

### 6.5 Speculative decoding across two servers; two drafts in one graph (`glm53_hpu/draft_chain.py`)

Speculative decoding with the model's MTP layer needed two fixes before it ran under pipeline parallelism:

1. vllm-gaudi built the draft model on every stage, although only the last stage has the final hidden states it
   needs. `fastlink.py` removes it on the other stages (1.5 GB per card saved).
2. vLLM's multi-batch engine loop fetched the draft tokens and scheduled the next step *before* the previous
   step's sampled tokens were processed. It then scheduled a draft token on its own, and the draft model's cache
   write failed to compile (`index_copy_: GLUE_INCOMPATIBLE_INPUT_SIZE`). One batch in flight during decode fixes
   it (`engine_single_batch.py` first, `engine_wave.py` in the end).

The cookbook's early-draft lever handles one draft token. For a second one vllm-gaudi falls back to a host loop:
copy the sampled tokens to the host, build the next position's metadata there, launch the draft model again.
That costs about 12 ms per step, as much as the second draft gains. `GLM_DRAFT_CHAIN=1` builds the whole proposal
as one graph, launched right after the rejection sampler:

- pass 1 is the draft model over the step's rows, exactly as with one draft;
- pass 2 runs the MTP layer again with draft 1 as input token, the previous pass's hidden state, positions + 1,
  the first pass's index selection, and its cache writes sent to the pad block.

Pass 2 is an approximation of what the draft model would compute a step later (it does not see a key for its own
position). That only affects how good draft 2 is: every draft is verified by the full model, so the output does
not change. The second draft now costs about 5 ms per step. Acceptance is 0.73 per draft on greedy natural text.
Three drafts did not pay: acceptance fell to 0.59 per draft.

### 6.6 Sparse attention at long context (`glm53_hpu/dsa.py`, installed by `patches/dsa_override.py`)

A revision of the cookbook's `glm5_hpu/dsa.py`; the file's header lists every change. `GLM_DSA_V2=1` copies it
over the image's file at container start. The plain-torch core is tested on the CPU
(`glm53_hpu/tests/test_dsa_core.py`).

What the original did at long context, measured before any change (warm):

| Context | First token (uncached) | Decode |
| --- | --- | --- |
| 2K | 0.33 s | 63.1 tok/s |
| 40K | 9.7 s | 54.5 tok/s |
| 128K | 102 s | 48.4 tok/s |

Device memory grew from 87.9 to 91.5 GiB during the 128K prefill, and its dense-masked prefill path took 20 s
per 4,096-token chunk at 230K keys. At that rate it would not reach 700K.

**When attention is dense and when sparse.** Below a threshold of cached keys a layer attends to everything
(exact up to 2,048 keys, slightly more than the model was trained with above that); above it, through the
indexer's selection. Thresholds: prefill 32,768 keys; decode 12,288 keys (720K profile) or 4,096 (1M profile).
The decode threshold is where the two paths cross: dense decode measured 57.6 / 52.8 / 46.9 tok/s at 8K / 16K /
30K, sparse decode 55 at 40K.

**Where the keys live.** vllm-gaudi already allocates a bf16 indexer-key cache per indexer layer inside the KV
pool, and nothing used it; the cookbook module allocated the same buffers a second time after memory profiling.
Reusing the native ones (`GLM_DSA_REUSE_NATIVE`) freed 2.06 GiB per stage-0 card, which is what makes a
720,896-token pool fit. The draft model's MTP layer owns an indexer too and is included; the original left it
dense at any context length.

**Prefill** (a 4,096-token chunk attending to everything before it):

1. *Scores.* Each card handles 1/8 of the chunk's rows. For a block of rows, all 32 indexer heads go through one
   matrix product against all keys, a ReLU, and a second product with the head weights that yields two partial
   sums per row, added in fp32 (`prefill_scores`). The original code went through the heads two at a time with
   an fp32 running sum, whose elementwise passes over [rows, keys] were the cost: 75 ms against 40 ms per 256
   rows x 725K keys.
2. *Thresholds.* Each row needs its 2,048-th largest score. A bisection with 20 halvings finds it
   (`bisect_threshold`). On a card this is three times faster than `torch.topk` when there are many long rows.
3. *Unions.* Neighbouring rows select overlapping keys. For each group of 64 rows the union of selected keys is
   compacted into a list of at most 32,768. Above 131,072 keys, a group whose union is larger shrinks to 32 or
   16 rows (16 rows select at most 32,768 keys), so this path always applies there. Below 131,072 keys a layer
   with scattered selections uses dense attention under a mask instead. The lists and the per-row masks are
   all-gathered so every card attends over the same keys. At 700K the unions measured 5K-30K keys per 64 rows on
   stage 0 and 17K-31K per 16-64 rows on stage 1.
4. *Attention.* Each group gathers its union's latents from the cache and runs fused scaled-dot-product
   attention in "absorbed" MLA form: the latent serves as the one shared key/value head for the card's query
   heads.

Memory controls, all found the hard way: the scoring and the bisection are cut into several lazy graphs (left in
one graph all intermediates can be alive at once, 8 GiB at 260K keys); row blocks shrink to 256 rows above 196K
keys and 128 above 655K; the bisection counts in int32 and is cut after every halving on the largest inputs; one
attention call gathers at most 512 MB of latents (a 9.7 GB gather once hung all eight cards).

**Decode** (3 rows attending to up to a million keys, in 22 layers per step):

1. *Rank-split scan* (`GLM_DSA_SHARD`). Every card scores a different eighth of the keys, takes its local top
   2,048 per row, and the candidates are exchanged with one all-reduce (each card writes its own slot of a
   zero-padded tensor). Every card then merges the 8 x 2,048 candidates the same way, so all agree on the
   selection. Every card holds the full latent cache, so only the scan is split.
2. *Pool scan* (above 40 % pool fill, `GLM_DSA_POOL_FRAC`). Instead of gathering a sequence's key blocks, each
   card scores its *physical* eighth of the key pool in place. Which keys a row may see is an additive bias
   (0 or -inf) built once per step from the block lists and reused by all 22 scans.
3. *Scoring layout.* `queries @ keys^T` on a transposed view of the key buffer gives the scores already as
   [rows, keys], the layout top-k wants: 80 us against 100 us.
4. *Top-k.* `torch.topk` runs one row per TPC core, so a [3, 131072] top-k is slow. It is done in two passes:
   top-2,048 of groups of 16,384 keys (many rows, all cores busy), then top-2,048 of the survivors.
5. *Merge.* The 8 candidate lists are merged with one top-k over two halves and a bitonic split
   (`max(A[t], B[k-1-t])` over two descending lists is exactly the k largest of both).
6. *Attention.* Each row gathers exactly its 2,048 latents and attends to them.

One scan costs 370 us at 700K keys and 385 us with the 1M pool; 22 scans are the 8-9 ms a long-context step costs
over a short one. The first version (gathering) took 547 us.

Other details: later MTP draft passes reuse the first pass's selection; a layer that attends densely ties the
indexer keys it wrote to its output so that a compiled graph cannot drop the write as dead code;
`GLM_DSA_TRACE=1` prints where a prefill's time goes.

### 6.7 Step metadata (`glm53_hpu/fastmeta.py`)

For every decode step vllm-gaudi rebuilds the paged-attention block lists with Python lists and calls a resolver
once per block. At 700K with three rows that is about 17,000 block entries per step: 4 ms for the lists plus
about as much for the calls, on every worker, before the step can be launched. `GLM_FAST_META=1` replaces both
with numpy code that returns exactly the same tensors (`tests/test_fastmeta.py`): 0.83 ms at 2K, 1.05 ms at 700K.

### 6.8 The 8-bit latent cache (`glm53_hpu/kvfp8.py`, `LatentInt8Cache` in `glm53_hpu/dsa.py`)

`kvfp8.py` changes one thing: the dtype in the KV cache *spec* of the latent caches (not the indexer-key
caches). vLLM's block accounting and vllm-gaudi's allocation follow the spec, so the pool simply holds twice as
many tokens. `LatentInt8Cache` replaces the object that writes and reads the cache on every MLA layer.

The codec:

- A latent of 576 values is treated as 9 groups of 64 (8 groups of compressed KV, 1 rotary group).
- Each group gets its own scale, `max|x| / 127`, stored as bf16 (9 scales per token and layer).
- Codes are `round(x / scale)`, integers from -127 to 127, stored as int8.
- Reading: `codes * scale`, applied as a plain two-dimensional broadcast with every group as a row.
- Error: 0.6 % of each part's norm (checked on a card with latents that have the real size ratio between the
  parts, `tools/scan_bench.py` with `Q8CHECK=1`).

How it got that shape, because each wrong turn looked fine at first:

1. **E4M3 codes, one scale per token** (what vllm-gaudi's own dynamic quantization does): 2.5-2.9 % error per
   latent, because E4M3 keeps 3 mantissa bits at any magnitude. Dropped.
2. **Linear codes, one scale per token**: every check that only reads a prompt passed, but generated text
   degenerated ("11.11.11.11...", repeated tags) and draft acceptance fell from 0.69 to 0.59. Cause: in the
   model's first ten layers the rotary part of a latent is 16 to 100 times larger than the compressed-KV part
   (the normalisation gains of those layers are 0.003-0.03), so with one scale the KV part rounds to zero. The
   error of the *whole* latent still looks fine (3-5 % for the worst token), which is why it was invisible until
   the error was measured per part.
3. **Linear codes, two scales** (KV part and rotary part): correct, but building a per-dimension scale tensor
   from two scales is slow on the cards (expand + concat: 115 us per layer; `torch.where` over a mask: about
   50 us per layer). Groups of equal size apply with one broadcast product and are slightly more precise.

How the cause was found: `GLM_KV_Q8_DEBUG=1` keeps the true latents of every layer in a bf16 shadow (with a small
pool) and counts, inside the compiled graphs, the error of every cache read separately for the two parts.

What it costs is in [section 8](#8-the-two-profiles-720k-and-1m).

### 6.9 Shapes (`tools/make_buckets.py`)

Every distinct prompt shape and decode shape is compiled the first time it is used, and a padded shape costs
compute in proportion to its size. `make_buckets.py` writes the list for a given pool size: fine query sizes for
short prompts; five query sizes for long contexts; context sizes every 64 blocks up to 131K tokens and every 128
above (a prefill pads its context by at most 16K tokens); decode sizes every 64 / 128 / 256 blocks. The served
lists are `serving/buckets-720k-c4096.txt` (449 shapes) and `serving/buckets-1m-c4096.txt` (559).

The pool size itself is set exactly with `--num-gpu-blocks-override` (5,632 or 8,184 blocks of 128 tokens)
instead of through `--gpu-memory-utilization`, which gave 667,520 tokens at 0.86.

---

## 7. Doing it yourself, step by step

The commands assume `/workspace/glm53` on both servers, owned by an account that can run `sudo -n docker`, and a
workstation with SSH access to both. Paths and image tags are the ones the scripts use.

### 7.1 Prerequisites

- Two servers with eight Gaudi 2 cards each, Gaudi software 1.24.1 on the hosts, Docker with the `habana`
  runtime.
- The scale-out ports of both servers on one switch, configured for RoCE, and one scale-out port per server with
  an IPv4 address on a common subnet. Check that the servers can ping each other on it.
- Free disk as in [section 2.1](#21-hardware).
- The CPU frequency governor on `performance` while serving. If nothing else on the servers manages it:

  ```bash
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee $g >/dev/null; done
  ```

  The unit files carry the same step as an optional `ExecStartPre` line.

### 7.2 Site settings and code

On the workstation:

```bash
git clone <this repository> && cd gaudi-glm53
cp deploy/site.env.example deploy/site.env      # then edit: SSH destinations, fabric interface and addresses
bin/sync_code.sh                                # code and settings to /workspace/glm53/code on both servers
```

`deploy/site.env` holds everything that belongs to one installation: how to reach the servers, the fabric
interface and the two addresses on it, the host interface HCCL uses for its control connections, and optional
replacements for the model path, the served name and the API binding. It is not tracked by git.
`bin/sync_code.sh` prints a checksum per server; the two must match.

### 7.3 Images

On both servers:

```bash
cd /workspace/glm53/code/image
sudo docker build -t glm53-base:v0.30.0-058f107 -f Dockerfile.base .
sudo docker build -t glm53-serve:dev1 serve
```

- `Dockerfile.base`: Habana's PyTorch container pinned by digest, vllm-gaudi and vLLM at the pinned commits (a
  mirror of vllm-gaudi's own Dockerfile with both commits fixed).
- `serve/Dockerfile`: adds the gaudi-cookbook at its pinned commit, applies its fixes (the v0.30.0 profile plus
  `vllm-long-prefill-share`, `glm47-streaming-text-markers`, `pipeline-parallel`) and the 14 patches of its GLM-5
  port, and checks that all report "applied".
- This repository's code is **not** in the image. The containers mount `/workspace/glm53`, and the entry script
  (`serving/entry.sh`) applies the hooks and patches at start. A code change needs a sync and a restart, not a
  build.

### 7.4 Weights

On the server with the most free space:

```bash
cd /workspace/glm53 && mkdir -p meta logs src models
python3 code/fetch/make_manifest.py zai-org/GLM-5.3-BF16 > meta/glm-5.3-bf16.manifest.tsv   # prints the commit
code/fetch/download.sh meta/glm-5.3-bf16.manifest.tsv src/GLM-5.3-BF16 zai-org/GLM-5.3-BF16 <commit> 8
```

- `make_manifest.py` lists size and hash of every file of one revision from the Hugging Face API.
- `download.sh` fetches exactly those files with `curl`, 8 streams, resumable, refuses to start if fewer than
  300 GiB would stay free, and verifies every hash at the end (marker `.VERIFIED`). No token is needed and no
  code from the repository is executed. Expect hours: 1.5 TB at the line's speed. Run it under `systemd-run`,
  `tmux` or `nohup`.

Convert and validate (CPU only, in the serving image):

```bash
sudo docker run --rm --name glm53-convert --memory 200g --entrypoint bash \
  -v /workspace/glm53:/workspace/glm53 -w /workspace/glm53 glm53-serve:dev1 \
  -c "code/convert/convert_full.sh src/GLM-5.3-BF16 models/GLM-5.3-FP8 12 8 1"
```

The last argument is the validation stride: 1 checks every expert tensor, the default 8 every eighth. The run
ends with `CONVERTED_AND_VALIDATED`. The container writes as root, so hand the result to the serving account
afterwards (`sudo chown -R`). `fetch/convert_after_download.sh` chains download check, conversion, full
validation and the ownership change as one unattended job with marker files, for when nobody will be watching.

Then copy `models/GLM-5.3-FP8` to the same path on the other server. Any tool works; the fabric is the fast way
(a parallel copy over the 100G ports moved the checkpoint in 2 minutes 15 seconds, where one rsync stream over
SSH needs on the order of an hour). The bf16 source is only needed again to convert again.

### 7.5 A small model first (recommended)

Two public test models with GLM-5.3's layer structure make the whole path testable in minutes: download
(`meta/tiny-0.6b.*`, `meta/small-3.5b.*` hold their manifests and revisions), convert
(`convert/convert_small.sh`), serve on one server (`serving/bench1.sh`) and on two (`bin/bench2.sh`), compare
(`serving/evalkit.py greedy`, `nll`, `compare`). Two things are worth confirming before the real model:

- bf16 and FP8 experts agree within bf16 rounding;
- two stages across the two servers give bit-identical greedy output to eight cards on one server.

A twin ([section 6.2](#62-performance-twins-finding-the-bottlenecks-without-the-weights)) then gives real timings:

```bash
# on a server (CPU only, in the serving image): 26 layers, 256 experts each, the real MTP layer appended.
# The twin has to be on both servers: build it on both or copy it.
sudo docker run --rm --entrypoint bash -v /workspace/glm53:/workspace/glm53 -w /workspace/glm53 glm53-serve:dev1 -c \
  "python3 code/convert/make_twin.py --small models/GLM-5.3-3.5B-A3.3B-small --out twins/twin26 \
           --layers 26 --experts 256 --mtp src/GLM-5.3-BF16"
# on the workstation: two servers, 2 stages x 8 cards, 14 + 12 layers
bin/bench2.sh twin26-fast "GLM_ENGINE_SKIP_EMPTY=1 GLM_PP_FASTLINK=1" /workspace/glm53/twins/twin26 2 8 14,12 \
        --enable-expert-parallel --enable-ep-weight-filter
```

`bench2.sh` starts both nodes, waits, measures and appends one line to `results/bench.tsv`. It removes other
`glm53-*` containers first, so it is for bring-up, not for a server that is in use.

### 7.6 Start the real model

For a first start, run the launcher in a terminal on each server and watch the output:

```bash
/workspace/glm53/deploy/serve_node.sh 0      # on node 0: stage 0 + API
/workspace/glm53/deploy/serve_node.sh 1      # on node 1: stage 1; waits for node 0's rendezvous port
```

The load takes about 2 minutes with the checkpoint in the page cache and about 4 from a cold start. Then, on
node 0:

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3", "messages": [{"role": "user", "content": "Explain what a KV cache is in two sentences."}],
  "max_tokens": 400, "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

Once that works, install the units so that systemd supervises both sides (the header of each unit file has the
install command), start them, and warm up:

```bash
sudo systemctl start glm53-node0             # node 0
sudo systemctl start glm53-node1             # node 1
/workspace/glm53/deploy/warmup.sh full       # node 0: touches the common shapes, then one long prompt
```

### 7.7 Check it

```bash
cd /workspace/glm53/results
python3 ../serving/chatcheck.py                                      # a few real prompts with timing
python3 ../serving/evalkit.py natspeed --prompts 4 --max-tokens 300  # decode speed and draft acceptance
python3 ../serving/longctx.py needle --tokens 128000 --followup 300  # needles, ingest time, cached turn, decode
python3 ../serving/cookbook_gates/glm5_gates.py tools                # and: gsm8k, needles, mtp, logprobs, speed, stress
python3 ../serving/crosscheck.py                                     # keys written below the sparse threshold are found above it
```

`serving/validation/` has the two scripts that ran the full validation of each profile.

### 7.8 Reaching the API

The API is OpenAI-compatible and listens on node 0's loopback. `bin/api_tunnel.sh` forwards it to the
workstation over SSH. To serve a private network instead, set `API_HOST` (and an `API_KEY`) in
`deploy/site.env`, or install the socket forwarder `deploy/glm53-lan.socket`; the launcher refuses an address
that is not private.

A client needs the base URL, the served model name (`glm-5.3` unless `SERVED_NAME` says otherwise) and the
context limit of the served profile: 720,000 or 1,047,000 tokens. vLLM returns the model's reasoning in the
`reasoning` field of a message.

A client that sends no sampling or reasoning settings gets the server's defaults:

- **temperature 1.0, top-p 0.95**: from the checkpoint's `generation_config.json`, also passed explicitly at
  start. These are Z.ai's defaults for GLM-5.3; their guide advises tuning only one of the two.
- **reasoning effort max**: the chat template's default when nothing is passed (`low` and `high` are the other
  values). Use `"chat_template_kwargs": {"reasoning_effort": "low"}` in a request to shorten the reasoning.
- **earlier reasoning is kept**: when a client returns the assistant's reasoning with each request, vLLM hands it
  to the template, and the template keeps it (`clear_thinking` is false unless asked).
- Tool calls use the `glm47` parser.

---

## 8. The two profiles: 720K and 1M

| | 720K (default) | 1M |
| --- | --- | --- |
| Settings file | `deploy/profile-720k-bf16.env` | `deploy/profile-1m-int8.env` |
| Largest request | 720,000 tokens | 1,047,000 tokens |
| KV pool | 5,632 blocks = 720,896 tokens | 8,184 blocks = 1,047,552 tokens |
| Latent cache | bf16, exactly what the model computes | 8-bit codes, 0.6 % error per latent |
| Sparse decode from | 12,288 cached tokens | 4,096 cached tokens |
| Decode speed | reference | 8-10 % lower at the same context |

What the 8-bit cache does to quality, measured on the model:

- **Log-likelihood of a long text** (`evalkit.py longnll`, positions that read cached entries): 11,184 tokens
  0.0772 against 0.0763 with bf16; 24,810 tokens 0.0912 against 0.0909. Unchanged on average.
- **Per token**: log-probabilities move by 0.02 on average. Where the bf16 model is sure of the token (93 % of
  positions) they move by 0.002; where it is unsure, by 0.2 to 0.8, in both directions. The bf16 build is
  bit-reproducible on this test, so all of that is the cache.
- **Functional checks**: reference codes found at 16K, 64K, 120K, 700K and 1,038K tokens; GSM8K 49/50 (the same
  miss as bf16); log-probabilities 20/20; stress test passed.
- **Tool-call gate**: 32/32, 31/32 and 31/32. The misses are one case that forces a tool call where none fits, at
  the default sampling; the bf16 profile misses the same case (32/32 and 30/32), so it is not the cache.

What it does to speed: reading 8-bit entries is not slower than reading bf16. Writing is: quantizing the step's
three new rows is a dozen small operations per layer (max, scale, divide, round, cast, two scatters), about 3 us
of device time each whatever their size, 3-4 ms per decode step in total. Fewer operations, bf16 arithmetic, and
attending before writing were all tried and none was faster ([section 12](#12-tried-and-dropped)).

The decode threshold differs because dequantizing makes the dense path slower: with the 8-bit cache dense decode
fell to 49 tok/s at 11K tokens, while through the indexer it is 62-66 tok/s from 4K to 40K.

To switch profiles: copy the profile over `deploy/glm53.env`, run `bin/sync_code.sh`, restart both units, and set
the clients' context limit.

---

## 9. Measurements

One user unless noted. Greedy decoding unless noted. Raw logs are in `results/` (see `results/README.md`).

### 9.1 Latency and decode speed

| Context | First token, new prompt | First token, cached conversation | Decode, 720K profile | Decode, 1M profile |
| --- | --- | --- | --- | --- |
| 2K | 0.34 s | - | 70-77 tok/s natural text (up to 89 on repetitive text) | 61-70 |
| 8K | 1.0 s | - | 72 | 63-66 |
| 40K | 6 s | 0.6-0.8 s | 68 | 62-64 |
| 128K | 31 s | 0.8 s | not re-measured with two drafts | 57 |
| 700K | 5-6 min | 2.5-3.6 s | 56-58 | 55-57 |
| 1,038K | 11-13 min | 6.6-8.4 s | does not fit | 50-55 (43-49 at the default sampling) |

- At the default sampling (temperature 1.0, top-p 0.95) fewer drafts are accepted on free prose than with greedy
  decoding: 2K natural text 66 tok/s sampled against 77 greedy.
- A multi-turn agentic coding workload with tool calls, measured from the server's metrics (720K profile, 35-57K
  tokens of context, default sampling): 53-59 tok/s, first token 1.3-3.8 s per turn, 98 % of each prompt from
  the prefix cache.
- First touch of a new shape compiles: the very first 700K prompt took 598 s, the first 1M prompt 1,015 s.
  Ingest of the same 700K prompt then went 416 s, 327 s, 299 s as the recipes warmed.

### 9.2 More than one user

- Four streams at 2K: 47 tok/s each, 145 tok/s in total (both profiles), against 80-89 for one stream on the same
  repetitive test text. The deployment allows four concurrent sequences (`MAX_SEQS`), and the bucket files hold
  decode shapes for up to four.
- The KV pool is shared: every running and every cached conversation lives in the same 720,896 or 1,047,552
  tokens. Idle conversations are evicted when the pool is needed and are read again on their next turn.
- One user's new prompt slows the others while it is read: each 4,096-token chunk shares the engine with their
  decode steps. In the stress gate (four decoding streams while 64K and 100K prompts arrive) the largest gap
  between two tokens was 1.7 s on the 720K profile and 7.1 s on the 1M profile.

### 9.3 Power

Read with `hl-smi` while one request was decoding: about 120 W per card (1.9 kW for 16 cards); idle with the
model loaded about 87 W per card; bursts to about 280 W per card while a prompt is ingested. A card can draw
600 W; [section 10.1](#101-a-decode-step) explains why decode stays far below that.

---

## 10. Where the time, the memory and the power go

### 10.1 A decode step

At 2K tokens with three rows a step takes about 34 ms and yields about 2.4 tokens.

- Stage 0: 42 layers. The host has launched all of the stage's graphs after about 10.5 ms and the cards finish
  at about 15 ms. Stage 1: 36 layers + LM head + sampler + drafts, about 11 ms of device time. (Trace taken with
  one draft, when a step was 28.5 ms.)
- Each layer ends in two collectives, and each collective ends a compiled graph: 156 all-reduces per step.
- Each extra row costs about 5 ms per step (28.5 ms with two rows, about 34 with three). The likely reason is
  the experts: every (row, expert) pair reads a 37.7 MB expert. That is why a second draft pays and a third
  does not.
- At long context add 22 indexer scans of 0.37-0.39 ms each: 8-9 ms. This is the whole difference between 2K and
  700K.
- With the 8-bit cache add 3-4 ms for the cache writes.

An estimate for 1M that matches the measurement: 34.5 + 22 x 0.385 + 3.5 = 46.5 ms for 2.4 tokens, 52 tok/s.

What one scan consists of at the 1M pool size (3 rows, 131,072 keys per card): scoring 86-107 us, local top-k
209 us, exchange and merge about 85 us. `torch.topk` on the cards is the floor: regrouping its first pass,
replacing second passes by bitonic splits, 16-bit or integer inputs, bisection and cumulative sums were all
measured and none is faster.

This is also why the cards draw a fifth of their rated power while decoding. A step for one user is a chain of
several hundred small, dependent operations: only one server works at a time, each card waits for its seven
neighbours twice per layer, and the arithmetic of three rows is tiny. Decode uses a small fraction of the matrix
engine's capacity; a prompt being read uses far more, and so do several streams at once (145 tok/s in total for
four streams against 70-89 for one).

### 10.2 Ingesting a prompt

Per 4,096-token chunk and stage, at 700K tokens already ingested: the experts and everything that does not depend
on context about 0.35 s; gather attention 0.5-0.75 s; the indexer (scores and thresholds for 12 layers) 2-3 s.
The two stages overlap. The indexer part grows with the context, which is why ingest time grows faster than
linearly: 6 s for 40K, 31 s for 128K, 5-6 min for 700K, 11-13 min for 1M.

A conversation that grows turn by turn only pays for the new tokens: vLLM keeps the cache of earlier turns
(prefix cache) until the pool is needed for something else.

### 10.3 Memory on a stage-0 card (94.62 GiB usable)

| | 720K profile | 1M profile |
| --- | --- | --- |
| Weights (layers 0-41, 8-bit experts) | 48.1 GiB | 48.1 GiB |
| Bytes of cache per token | 51,456 = 42 x 1,152 (latents) + 12 x 256 (indexer keys) | 27,264 = 42 x 576 + 12 x 256 |
| KV pool | 34.55 GiB (720,896 tokens) | 26.68 GiB (1,047,552 tokens) |
| Used after start | 82.66 GiB | 74.79 GiB |

In bf16 a pool for 1,047,552 tokens would need about 50 GiB next to 48 GiB of weights on a 94.6 GiB card, before
any working memory: it does not fit. During a long prefill `hl-smi` shows up to 95 of 96 GiB in use on both
profiles. That figure includes the allocator's reuse pool; no run failed with an out-of-memory error (three full
ingests at 1M, at least six at 700K).

---

## 11. Pitfalls, and how each was solved

Each of these cost hours and would be easy to repeat.

### 11.1 Speed

| Symptom | Cause | Fix |
| --- | --- | --- |
| Decode 40 % slower than the cards allow, on every configuration | hosts idle in the `schedutil` governor; a decode step is hundreds of host-side graph launches | `performance` governor while serving: 62.6 → 105.9 tok/s on a twin |
| Crossing between the servers costs 2.4 to 9 ms per token | the stage hand-over copies to the host and back: a device sync on each side | card-to-card all-reduce inside the graphs (`fastlink.py`) |
| About 1 ms per token lost per extra stage at one stream | vLLM dispatches an empty batch to every worker and sleeps 1 ms | `engine_skip_empty.py` |
| A second draft token gains nothing | vllm-gaudi builds it in a host loop: 12 ms per step | both drafts in one device graph (`draft_chain.py`): the second costs 5 ms |
| Decode slows with context although attention is sparse | (a) all cached keys are scored on every card; (b) block lists are rebuilt in Python per step: 8 ms at 700K | (a) rank-split scan and in-place pool scan; (b) numpy (`fastmeta.py`) |
| The local top-k is the largest part of an indexer scan (209 of 385 us) | top-k and sort run one row per TPC core and need at least about 40 us per call | two passes: groups of 16,384 keys (many rows, all cores busy), then the survivors. Nothing else measured was faster |
| A layer gets 100 us slower after a harmless-looking change | `torch.cat`, `expand` and `index_select` along the last dimension are slow in a decode layer (115 us, 1.7 ms) | constant masks with `torch.where`, or broadcast products on reshaped views |
| Prefill at 40K halves after fixing speculative decoding | one batch in flight removes the overlap of chunks across the stages | `engine_wave.py`: two batches in flight only while everything is prefilling |

### 11.2 Correctness

| Symptom | Cause | Fix |
| --- | --- | --- |
| The draft model's cache write fails to compile across stages (`index_copy_: GLUE_INCOMPATIBLE_INPUT_SIZE`) | the multi-batch engine loop schedules a draft token on its own, before the last step's output is processed | one decode step in flight (`engine_wave.py`) |
| With an 8-bit cache, prompts are read correctly but generated text degenerates | one scale per token: in the first ten layers the rotary part of a latent is 16-100x the KV part, which rounds to zero | one scale per 64 dimensions (`LatentInt8Cache`) |
| "The 8-bit cache does not change the log-likelihood" | the test prompt fit one prefill chunk, and one chunk never reads the cache: attention works on the rows in flight | a multi-chunk text (`evalkit.py longnll`) plus a decode check |
| A whole-latent error of 3-5 % looked acceptable | one part is a hundred times smaller than the other | measure the error per part (`GLM_KV_Q8_DEBUG=1`) |
| A sample of "real" latents showed both parts the same size | it was the profiling run's dummy input | check what a sample is before designing on it |
| Text written below the sparse threshold could be invisible above it | a layer that attends densely does not read the indexer keys it writes, so a compiled graph may drop the write as dead code | tie the written key to the layer output, times zero; `serving/crosscheck.py` tests it |
| The MTP layer attends densely at any context length | the cookbook module walks only the target model | include the draft model's layer |
| Two launches of one build give different greedy text | sums in the paged attention follow the block layout, which depends on what was served before; outputs part at near-ties (4 to 6 of 12 prompts stay identical) | compare agreement statistics and log-likelihoods, not token identity, between runs |
| A tool-call gate fails once in a while | one case forces a tool call where none fits, at the default sampling; it fails on both profiles | judge a cache codec on repeated runs of both profiles |

### 11.3 Lazy mode and compiled graphs

| Symptom | Cause | Fix |
| --- | --- | --- |
| A `print` or `.cpu()` in the forward fires once and never again | the forward runs as Python only when a shape is first traced; afterwards the recipe is replayed | debug through values that are part of the graph |
| The engine dies with "Empty tensor optional" on a later request | a tensor first created inside a traced graph and kept on a module for later graphs has no storage | create such tensors at construction time |
| Debug counters stay at zero | an update of a module's own tensor that nothing depends on is dropped | make the counter feed the layer's output, times zero |
| Device memory climbs by gigabytes during a long prefill | in one lazy graph all intermediates can be alive at once (8 GiB at 260K keys) | cut scoring and bisection into several graphs; shrink row blocks with the key count |
| All eight cards hang | one gather of 9.7 GB of latents | at most 512 MB per attention call |
| The first graph of a benchmark process returns garbage | seen only outside the server, through vllm-gaudi's decode routine, whatever the cache | the check runs one throw-away graph first |
| Card micro-benchmarks contradict the model | identical repetitions are merged, independent ones overlap on idle cores | chained, dependent repetitions; still off by about 10 us, so decide with the real model |
| `hl-smi` shows 95 of 96 GiB during prefill | it counts the allocator's reuse pool | not an error: no run failed |

### 11.4 Sizing and operation

| Symptom | Cause | Fix |
| --- | --- | --- |
| The pool comes out at 667,520 tokens instead of 720K | sizing through `--gpu-memory-utilization` | `--num-gpu-blocks-override` with the exact block count |
| 2 GiB per card missing at 720K | the sparse module allocates indexer-key buffers that vllm-gaudi already has | reuse the native caches |
| The first long prompt after a start takes twice as long | every new shape compiles (598 s against about 300 s at 700K) | recipes are cached on disk; `deploy/warmup.sh full` after a start |
| One container exits and the endpoint hangs | the two containers are one engine; the survivor waits forever | pair watchdog in `serve_node.sh`: both units restart together |
| The checkpoint loads slowly after other work on the server | page cache of earlier jobs competes with a 773 GB read | the launcher drops the page cache before the load |
| An interactive session slows to 20-45 tok/s for minutes | a 700K test prompt was sent to the same endpoint; it shares every decode step and fills the pool | check `vllm:num_requests_running` in `/metrics` before testing a server that is in use |
| `pkill -f pattern` over SSH silently does nothing | it matches, and kills, the remote shell that carries the pattern, before the rest of the command runs | stop containers by name, or use a pattern that cannot match itself |
| A script behaves strangely after a sync | bash reads a running script from disk as it goes | `bin/sync_code.sh` replaces files instead of rewriting them in place |

---

## 12. Tried and dropped

| Tried | Result |
| --- | --- |
| Removing the per-layer graph cut for decode (`GLM_DECODE_NOCUT`) | helps a single card, not a stage with collectives; removing it for prefill makes compiles take minutes |
| Three draft tokens per step | acceptance 0.59 per draft, no gain over two |
| Bisection, cumulative sum + compaction, bitonic merges, 16-bit or integer scores for the decode top-k | none faster than the two-pass top-k (`glm53_hpu/topsel.py` keeps the exact chunked / bitonic variant) |
| `torch.topk` for the prefill thresholds | 90 ms against 31 ms for the bisection (256 rows x 725K keys); the two-pass form 1.4 s |
| E4M3 codes for the latent cache | four times the error of linear codes |
| One scale per token for the latent cache | destroys the compressed-KV part in the early layers |
| Cheaper cache writes: bf16 arithmetic, no clamp | 74 us per layer against 69 |
| Attending before writing, with the step's rows patched into the gathered latents, or added as extra score columns | 109 us and 84 us per layer against 69: the write is work, not only latency |
| Dense decode with the 8-bit cache up to 12K tokens | 49 tok/s at 11K; through the indexer from 4K it is 62-66 |
| A single pipeline stage of 16 cards | halves the context: the latent cache of all 78 layers on every card |

---

## 13. Tests and tools

| Tool | What it does |
| --- | --- |
| `serving/chatcheck.py` | a few real chat prompts with answers and timing |
| `serving/evalkit.py greedy / nll / compare` | fixed prompts with token ids and log-probabilities; teacher-forced log-likelihood; agreement between two runs |
| `serving/evalkit.py longnll` | log-likelihood of one long text. A prompt longer than one prefill chunk is the cheapest test of what the KV cache gives back: the short `nll` test never reads the cache |
| `serving/evalkit.py speed / natspeed` | decode speed on filler and on natural prompts, with draft acceptance |
| `serving/longctx.py needle` | a document of N tokens with three reference codes at 10 / 50 / 90 % depth, a question for them, then a follow-up turn: correctness, ingest time, cached-turn latency, decode speed at that context |
| `serving/crosscheck.py` | makes the model state a code while the context is short, write until it is past the sparse threshold, then asks for the code: tests that indexer keys written in the dense regime are really stored |
| `serving/cookbook_gates/` | the gaudi-cookbook's serving gates (tool calls, GSM8K, needles, MTP acceptance, log-probabilities, speed, stress), with API-key support added |
| `serving/validation/` | the two scripts that ran the full validation of each profile |
| `tools/scan_bench.py` | card-level benchmark and check program: scan variants, top-k shapes, the 8-bit cache through vllm-gaudi's real decode routine against fp32 references, per-layer cost in a chain of layers. Needs free cards |
| `tools/hccl_lat.py`, `tools/hccl_chain.py` | latency of a two-card all-reduce used as a send, alone and in a chain of stages |
| `tools/make_buckets.py` | writes the bucket file for a given pool size |
| `glm53_hpu/tests/`, `convert/tests/` | CPU tests of the selection code, the 8-bit cache, the step metadata and the FP8 quantizers |
| `bin/bench2.sh`, `serving/run2.sh`, `serving/bench1.sh`, `serving/run1.sh` | start an ad-hoc two-server or one-server configuration with a given set of switches and append a line to `results/bench.tsv` |

Two things to know when reading gate output (`results/README.md` has more):

- The cookbook's `mtp` gate prints FAIL on this deployment. Its threshold, an acceptance of 0.70, was set for one
  draft per step; with two drafts the acceptance per draft is 0.65-0.68 at the default sampling (0.83 with one
  draft), and more tokens are accepted per step.
- The `tools` gate misses one case in some runs on both profiles ([section 8](#8-the-two-profiles-720k-and-1m)).

The CPU tests need only PyTorch. On a server, in the serving image:

```bash
sudo docker run --rm -v /workspace/glm53/code:/code --entrypoint python3 glm53-serve:dev1 /code/glm53_hpu/tests/test_dsa_core.py
```

---

## 14. Operating it

```bash
# node 0                                   # node 1 (any order)
sudo systemctl start glm53-node0           sudo systemctl start glm53-node1
sudo systemctl stop glm53-node0            sudo systemctl stop glm53-node1      # stop both
journalctl -u glm53-node0 -f               # log
/workspace/glm53/deploy/warmup.sh [full]   # on node 0 after a start: touch the common shapes (full: one long prompt)
```

- **The units** (`deploy/glm53-node0.service`, `glm53-node1.service`) run `deploy/serve_node.sh N` in the
  foreground with `Restart=always`. They are not enabled at boot unless you enable them, and each takes all
  eight cards of its server. A stopped unit may show "failed"; a start works regardless.
- **`serve_node.sh`** reads `deploy/glm53.env` (the active profile) and `deploy/site.env`, checks that the
  node's fabric address is on this server, refuses an API address that is not private, makes node 1 wait for
  node 0's rendezvous port, drops the page cache before the checkpoint is read, and runs the container with the
  deployed switches.
- **Pair watchdog.** The launcher watches for TCP connections to the other node's fabric address and stops its
  container when they are gone, so both units restart together.
- **Code changes**: edit, `bin/sync_code.sh`, restart both units.
- **Start time**: about 4 minutes. The first request of each shape after a start loads its recipe (seconds).
- **Do not test while someone is using it.** A long test prompt shares every decode step with the other streams
  and fills the pool. Check `vllm:num_requests_running` in `http://127.0.0.1:8000/metrics` first.

On the servers:

| Path | Size | What |
| --- | --- | --- |
| `/workspace/glm53/models/GLM-5.3-FP8` | 773 GB (both) | the served checkpoint |
| `/workspace/glm53/src/GLM-5.3-BF16` | 1.5 TB (one) | the bf16 download; only needed to convert again |
| `/workspace/glm53/twins`, small test models | about 250 GB | timing twins and test models from the bring-up |
| `/workspace/glm53/code`, `serving`, `deploy`, `meta` | small | this repository as the containers use it |
| `/workspace/glm53/recipe-cache`, `results`, `logs` | about 10 GB | compiled graphs, measurements, logs |
| images `glm53-base:v0.30.0-058f107`, `glm53-serve:dev1` | | built from `image/` |

---

## 15. Switch reference

Set by `deploy/serve_node.sh` in the deployment unless marked otherwise.

| Switch | Effect | Where |
| --- | --- | --- |
| `GLM_PP_FASTLINK=1` | card-to-card stage link for decode steps | `glm53_hpu/fastlink.py` |
| `GLM_PP_FASTLINK_MAX_TOKENS` (64) | largest forward that uses it | same |
| `GLM_PP_FASTLINK_TRACE=1` (off) | per-stage timing of a step; adds a device sync | same |
| `GLM_DECODE_NOCUT=1` (off) | no per-layer graph cut for small forwards | same |
| `GLM_ENGINE_SKIP_EMPTY=1` | no empty engine steps | `patches/engine_skip_empty.py` |
| `GLM_ENGINE_PREFILL_WAVE=1` | overlapped prefill chunks, one decode step in flight | `patches/engine_wave.py` |
| `GLM_ENGINE_SINGLE_BATCH=1` (off) | plain engine loop | `patches/engine_single_batch.py` |
| `GLM_MTP=2` | speculative decoding with the MTP layer, drafts per step | `serving/entry.sh` |
| `GLM_DRAFT_CHAIN=1` | all drafts of a step in one graph | `glm53_hpu/draft_chain.py` |
| `GLM_ROUTER_FAST=1`, `GLM_EARLY_DRAFT=1`, `GLM_SPEC_DEVICE_REJECT=1` | the cookbook's decode levers | gaudi-cookbook `decode_lean.py` |
| `GLM_FAST_META=1` | numpy step metadata (`GLM_FAST_META_TRACE=1` prints its time) | `glm53_hpu/fastmeta.py` |
| `GLM_DSA_SPARSE=1` | sparse attention through the indexer | `glm53_hpu/dsa.py` |
| `GLM_DSA_V2=1` | use this repository's `dsa.py` instead of the image's | `patches/dsa_override.py` |
| `GLM_DSA_MIN_KEYS_DECODE` | decode goes sparse above this many keys (`SPARSE_DECODE_FROM` in the profile) | `glm53_hpu/dsa.py` |
| `GLM_DSA_MIN_KEYS` (32768) | the same for prefill | same |
| `GLM_DSA_SHARD` (1), `GLM_DSA_POOL_FRAC` (0.4) | rank-split scan; pool scan above this pool fill | same |
| `GLM_DSA_REUSE_NATIVE` (1) | keys in vllm-gaudi's own key caches | same |
| `GLM_DSA_PREFILL_SCORE` (rows), `GLM_DSA_PREFILL_PARTS` (2), `GLM_DSA_PREFILL_KTH` (bisect) | prefill scoring and threshold method | same |
| `GLM_DSA_KV_BUDGET_GB` (3), `GLM_DSA_MASK_BUDGET_GB` (0.75) | working memory of the dense-masked prefill path | same |
| `GLM_DSA_TRACE=1` (off) | where a prefill's time goes; slows decode | same |
| `GLM_KV_FP8=int8` (`KV_CODEC` in the profile) | 8-bit latent cache (`fp8`: E4M3 codes, for comparison) | `glm53_hpu/kvfp8.py`, `dsa.py` |
| `GLM_KV_Q8_DEBUG=1` (off) | per-part cache error against a bf16 shadow; small pools only | `glm53_hpu/dsa.py` |
| `GLM_PARSERS=1` | reasoning and tool-call parsers, sampling defaults | `serving/entry.sh` |
| `GLM_DEBUG_SPEC=1` (off) | shapes that reach the draft model | `glm53_hpu/debug_spec.py` |

Stock settings that matter: `VLLM_PP_LAYER_PARTITION=42,36`, `VLLM_BUCKETING_FROM_FILE`,
`VLLM_HPU_FSDPA_SOFTMAX_MODE=None` (the accurate softmax), `VLLM_SCALE_ADJUSTMENT=false` (the FP8 codes already
fit Gaudi 2's range), `VLLM_GRAPH_RESERVED_MEM=0.2`, `--max-cudagraph-capture-size 2048` (longer prefill chunks
stay out of compiled graphs), `--long-prefill-token-threshold 4096`, `--block-size 128`,
`--num-gpu-blocks-override`.

---

## 16. Repository layout

```
README.md                 this document
bin/                      run on the workstation
  lib.sh                    reads deploy/site.env, reaches the servers
  sync_code.sh              push code and settings to both servers
  bench2.sh twin_batch.sh   ad-hoc two-server benchmark launcher; the twin benchmark batch
  api_tunnel.sh             SSH forward of the API
convert/                  bf16 → FP8 converter, independent validator, twin builder, safetensors lister, tests
deploy/                   settings (glm53.env = the active profile, site.env.example), the two profiles,
                          launcher, service units, private-network forwarder, warm-up
fetch/                    manifest of a Hugging Face revision, verified download, unattended convert chain
glm53_hpu/                runtime modules (imported inside the serving containers)
  fastlink.py               card-to-card stage link; installs the other modules into the runner
  dsa.py                    sparse attention for long context; the 8-bit latent cache
  draft_chain.py            two MTP drafts in one graph
  fastmeta.py               decode-step metadata with numpy
  kvfp8.py                  8-bit dtype in the KV cache spec
  topsel.py                 chunked / bitonic exact top-k (correct, not faster on the cards, unused)
  debug_spec.py             diagnostics for speculative decoding
  tests/                    CPU tests
image/                    Dockerfiles of the base and the serving image
meta/                     the model's config; manifests of the two public test models
patches/                  source patches applied at container start: engine loop, hooks, module override
results/                  benchmark history and the logs of the validation runs
serving/                  container entry script, ad-hoc launchers, bucket files, test tools, gates
tools/                    card-level benchmarks, bucket file generator
```

---

## 17. Credits and licences

- **GLM-5.3** is Z.ai's model (`zai-org/GLM-5.3-BF16`), under its own licence. No weights are in this repository.
- **vLLM** and **vllm-gaudi**: Apache-2.0. Patched at container start; no copies here.
- **gaudi-cookbook** (`thediymaker/gaudi-cookbook`, Apache-2.0): the image applies its fixes and its GLM-5 port.
  Two parts of this repository derive from it: `glm53_hpu/dsa.py` (a revision of its `dsa.py`, as the file's
  header says) and `serving/cookbook_gates/` (its serving gates and question file, with API-key support added).
  The three decode levers and the pipeline-parallel fix are its work, used unchanged. The byte rules of the FP8
  format follow its codecs.
- **Wright** (`abotsis/wright`): a separate inference engine for Gaudi that drives the Synapse API directly. Its
  kernels are under a noncommercial licence and none of its code is used. Three of its design ideas were
  considered here: splitting the decode-time indexer scan across cards (done, in PyTorch), an 8-bit latent cache
  with a per-token scale (done for the 1M profile, in a different form: one scale per token turned out to be too
  coarse for this model), and slicing experts across cards instead of expert parallelism (not measured on the
  real model).
- **Test models**: `inference-optimization/GLM-5.3-0.6B-A0.4B` and `GLM-5.3-3.5B-A3.3B-small` (MIT).
- **GSM8K**: the 50 questions in `serving/cookbook_gates/gsm8k50.json` come with the cookbook's gates.
- Everything else was written for this project. Most of the Python sources carry an
  `SPDX-License-Identifier: Apache-2.0` header; the repository has no licence file of its own yet.

---

## 18. Ideas not done

- Keep the latent cache of some layers in bf16 in the 1M profile, as far as memory allows (roughly 15 layers on
  stage 0 and 25 on stage 1), after measuring which layers matter most.
- A fused quantize operation would remove most of the 8-bit cache's write cost.
- 8-bit indexer keys (1.6 GiB per stage, and a faster scan).
- Reuse layer 74's selection for the MTP layer's own scan (one scan of 22 per step).
- A card-to-card link for prefill chunks.
- 4 stages x 4 cards: cheaper collectives per layer, but a four-way scan split is worse at long context.
- More than four concurrent sequences: needs decode shapes for larger batches in the bucket files and a look at
  how ingest and decode share the engine.
