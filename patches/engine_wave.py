#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# This patch contains source text of EngineCore from vLLM (vllm-project/vllm, Apache-2.0) and modified versions
# of it, so this file stays under that licence (see NOTICE).
"""engine-wave: overlap prefill chunks across the pipeline stages, one step at a time for decode
(GLM_ENGINE_PREFILL_WAVE=1). Apply after engine_skip_empty.py and engine_single_batch.py; use it INSTEAD of
GLM_ENGINE_SINGLE_BATCH=1 (leave that off: this mode needs vLLM's batch queue).

vLLM keeps pipeline_parallel_size batches in flight. For a long prompt that is what makes the stages overlap: the
scheduler advances a request's computed-token count when it schedules a chunk, so chunk k+1 runs on stage 0 while
chunk k runs on stage 1. GLM_ENGINE_SINGLE_BATCH=1 (needed for speculative decoding, see that patch) gave this up:
measured 4,130 tok/s prefill at 40K tokens against about 7,000 with overlap.

This mode keeps the queue and restricts it:
  1. while a batch is in flight, a new one is scheduled only if every running request is still prefilling (after
     the last schedule). As soon as a request's final prompt chunk is scheduled, or anything decodes, the loop
     waits for the in-flight batch first. Decode steps therefore run exactly like the plain engine loop: schedule,
     run through every stage, process the output, then fetch the draft tokens.
  2. after dispatching a prefill-only batch the loop returns "nothing executed", so the engine does not call
     take_draft_token_ids (that call queues behind the batch on the last stage and would serialise the stages
     again). Draft tokens are fetched after a step whose output left a request decoding and the queue empty, i.e.
     in the same order as the plain loop: output processed first, then the drafts.
  3. the gaudi-cookbook's per-step cap on running requests (microbatches) is off, as with GLM_ENGINE_SINGLE_BATCH.
Off (default): stock behaviour. Idempotent; exit 0 applied, 1 (--check) not applied, 2 source drift.
"""
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[glm53:wave]"

# --- vllm/v1/engine/core.py (after engine_skip_empty.py)
OLD_SCHED = """        _glm53_out = None
        if self.scheduler.has_requests():
            _glm53_out = self.scheduler.schedule(self._should_throttle_prefills())
"""
NEW_SCHED = """        _glm53_out = None
        # [glm53:wave] with a batch in flight, schedule another only while every running request is prefilling
        _glm53_wave = os.environ.get("GLM_ENGINE_PREFILL_WAVE", "0") == "1"
        if self.scheduler.has_requests() and not (
            _glm53_wave and batch_queue and not _glm53_prefill_only(self.scheduler)
        ):
            _glm53_out = self.scheduler.schedule(self._should_throttle_prefills())
"""
OLD_EARLY = """                if len(batch_queue) < self.batch_queue_size and (
                    model_executed or self.scheduler.has_requests()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, model_executed
"""
NEW_EARLY = """                if len(batch_queue) < self.batch_queue_size and (
                    model_executed or self.scheduler.has_requests()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    if not _glm53_wave:
                        return None, model_executed
                    if _glm53_prefill_only(self.scheduler):  # [glm53:wave] go on to the next prompt chunk
                        return None, False
"""
OLD_RET = """            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed
"""
NEW_RET = """            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        if _glm53_wave:  # [glm53:wave] fetch draft tokens only after a step that left a request decoding
            model_executed = (
                scheduler_output.total_num_scheduled_tokens > 0
                and not batch_queue
                and not _glm53_prefill_only(self.scheduler)
            )
        return engine_core_outputs, model_executed
"""
CORE_TAIL = '''

def _glm53_prefill_only(scheduler) -> bool:
    """[glm53:wave] True while every running request still has prompt tokens to schedule."""
    running = getattr(scheduler, "running", None)
    return bool(running) and all(r.is_prefill_chunk for r in running)
'''

# --- vllm/v1/core/sched/scheduler.py (after engine_single_batch.py)
OLD_CAP = """            and self.vllm_config.max_concurrent_batches > 1
        ):
            _gfpp_cap = max(1, -(-len(self.running) // self.parallel_config.pipeline_parallel_size))
"""
NEW_CAP = """            and self.vllm_config.max_concurrent_batches > 1
            and __import__("os").environ.get("GLM_ENGINE_PREFILL_WAVE", "0") != "1"  # [glm53:wave]
        ):
            _gfpp_cap = max(1, -(-len(self.running) // self.parallel_config.pipeline_parallel_size))
"""


def patch(src: str, pairs) -> str | None:
    for old, new in pairs:
        if src.count(old) != 1:
            print(f"{MARKER} anchor found {src.count(old)} times, expected 1: {old.strip().splitlines()[0]}")
            return None
        src = src.replace(old, new, 1)
    return src


def main() -> int:
    check = "--check" in sys.argv
    root = Path(sys.argv[sys.argv.index("--root") + 1]) / "vllm" if "--root" in sys.argv else \
        Path(list(importlib.util.find_spec("vllm").submodule_search_locations)[0])
    core, sched = root / "v1/engine/core.py", root / "v1/core/sched/scheduler.py"
    todo = []
    src = core.read_text()
    if MARKER not in src:
        if "[glm53:skip-empty]" not in src:
            print(f"{MARKER} apply engine_skip_empty.py first")
            return 2
        new = patch(src, ((OLD_SCHED, NEW_SCHED), (OLD_EARLY, NEW_EARLY), (OLD_RET, NEW_RET)))
        if new is None:
            return 2
        todo.append((core, new.rstrip("\n") + "\n" + CORE_TAIL))
    src = sched.read_text()
    if MARKER not in src:
        if "[glm53:single-batch]" not in src:
            print(f"{MARKER} apply engine_single_batch.py first")
            return 2
        new = patch(src, ((OLD_CAP, NEW_CAP),))
        if new is None:
            return 2
        todo.append((sched, new))
    if not todo:
        print(f"{MARKER} applied")
        return 0
    if check:
        print(f"{MARKER} not applied")
        return 1
    for p, text in todo:
        compile(text, str(p), "exec")
        tmp = p.with_name(p.name + ".glm53.tmp")
        tmp.write_text(text)
        os.chmod(tmp, p.stat().st_mode & 0o7777)
        os.replace(tmp, p)
        for pyc in (p.parent / "__pycache__").glob(f"{p.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)
        print(f"{MARKER} patched {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
