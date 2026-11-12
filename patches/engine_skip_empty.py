#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# This patch contains source text of EngineCore from vLLM (vllm-project/vllm, Apache-2.0) and modified versions
# of it, so this file stays under that licence (see NOTICE).
"""engine-skip-empty: stop the batch-queue engine loop from paying for empty steps (pipeline parallelism).

vLLM's EngineCore.step_with_batch_queue (used when the executor allows more than one batch in flight, i.e. with
--pipeline-parallel-size N) keeps scheduling until its queue of N batches is full. At one stream every schedule
after the first finds the only request in flight and returns an empty batch, which is still sent to every worker;
each such iteration also returns "nothing executed", and the outer loop then sleeps 1 ms. Measured on Gaudi 2:
about 1.1 ms per token for every pipeline stage beyond the first (2 stages +1 ms, 6 stages +5.6 ms).

With GLM_ENGINE_SKIP_EMPTY=1:
  1. an empty schedule (no tokens, no new, finished or preempted requests, no KV-connector work) is not dispatched
     while a batch is in flight; the loop blocks on the oldest batch instead;
  2. the outer loop does not take its 1 ms sleep after a step that produced outputs.
Off (the default) leaves the stock behaviour. Idempotent; exit 0 applied, 1 (--check) not applied, 2 source drift.
"""
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[glm53:skip-empty]"

OLD_SCHED = """        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
"""
NEW_SCHED = """        # [glm53:skip-empty] an empty schedule is not sent to the workers while a batch is in flight
        _glm53_out = None
        if self.scheduler.has_requests():
            _glm53_out = self.scheduler.schedule(self._should_throttle_prefills())
            if (
                os.environ.get("GLM_ENGINE_SKIP_EMPTY", "0") == "1"
                and batch_queue
                and _glm53_out.total_num_scheduled_tokens == 0
                and not _glm53_out.scheduled_new_reqs
                and not _glm53_out.finished_req_ids
                and not _glm53_out.free_encoder_mm_hashes
                and not _glm53_out.preempted_req_ids
                and _glm53_out.kv_connector_metadata is None
            ):
                _glm53_out = None
        if _glm53_out is not None:
            scheduler_output = _glm53_out
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
"""
OLD_SLEEP = """        if not model_executed and self.scheduler.has_requests():
            time.sleep(0.001)
"""
NEW_SLEEP = """        # [glm53:skip-empty] no sleep after a step that produced outputs
        if (
            not model_executed
            and self.scheduler.has_requests()
            and not (outputs and os.environ.get("GLM_ENGINE_SKIP_EMPTY", "0") == "1")
        ):
            time.sleep(0.001)
"""


def main() -> int:
    check = "--check" in sys.argv
    spec = importlib.util.find_spec("vllm")
    p = Path(list(spec.submodule_search_locations)[0]) / "v1/engine/core.py"
    src = p.read_text()
    if MARKER in src:
        print(f"{MARKER} applied")
        return 0
    for old in (OLD_SCHED, OLD_SLEEP):
        if src.count(old) != 1:
            print(f"{MARKER} anchor found {src.count(old)} times, expected 1: {old.strip().splitlines()[0]}")
            return 2
    if "\nimport os\n" not in src or "\nimport time\n" not in src:
        print(f"{MARKER} core.py does not import os and time")
        return 2
    if check:
        print(f"{MARKER} not applied")
        return 1
    text = src.replace(OLD_SCHED, NEW_SCHED, 1).replace(OLD_SLEEP, NEW_SLEEP, 1)
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
