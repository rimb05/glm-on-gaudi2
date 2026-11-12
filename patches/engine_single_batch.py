#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# This patch contains source text of EngineCore from vLLM (vllm-project/vllm, Apache-2.0) and modified versions
# of it, so this file stays under that licence (see NOTICE).
"""engine-single-batch: one batch in flight at a time under pipeline parallelism (GLM_ENGINE_SINGLE_BATCH=1).

vLLM lets pipeline_parallel_size batches be in flight (EngineCore.step_with_batch_queue). That loop is wrong for
speculative decoding: right after it dispatches a step it runs post_step, which fetches the step's draft tokens,
and then schedules again BEFORE the step's sampled tokens have been processed. The scheduler then sees
num_tokens_with_spec one ahead of num_computed_tokens and schedules the draft token on its own (one scheduled token,
one draft token). On HPU the last stage builds a one-row decode while the draft model is fed two rows, and the
MTP layer's KV write fails to compile (index_copy_: GLUE_INCOMPATIBLE_INPUT_SIZE).

With GLM_ENGINE_SINGLE_BATCH=1, VllmConfig.max_concurrent_batches is 1: the engine uses its plain step loop
(schedule, run through every stage, process the output, fetch drafts), exactly as on one node, and the
gaudi-cookbook's per-step cap on running requests (meant to form overlapping microbatches) is off, so all running
requests share each step. One stream loses nothing; several streams no longer overlap in the stages.
Off (default): stock behaviour. Idempotent; exit 0 applied, 1 (--check) not applied, 2 source drift.
"""
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[glm53:single-batch]"

OLD_PROP = """    def max_concurrent_batches(self) -> int:
        # PP requires PP-size concurrent batches to fill the pipeline.
"""
NEW_PROP = """    def max_concurrent_batches(self) -> int:
        # [glm53:single-batch] one batch in flight: the plain engine step loop, also under pipeline parallelism
        if os.environ.get("GLM_ENGINE_SINGLE_BATCH", "0") == "1":
            return 1
        # PP requires PP-size concurrent batches to fill the pipeline.
"""
OLD_CAP = """        if self.use_pp and not self.scheduler_config.async_scheduling:
            _gfpp_cap = max(1, -(-len(self.running) // self.parallel_config.pipeline_parallel_size))
"""
NEW_CAP = """        # [glm53:single-batch] no microbatch cap when only one batch is in flight
        if (
            self.use_pp
            and not self.scheduler_config.async_scheduling
            and self.vllm_config.max_concurrent_batches > 1
        ):
            _gfpp_cap = max(1, -(-len(self.running) // self.parallel_config.pipeline_parallel_size))
"""
TARGETS = (("config/vllm.py", OLD_PROP, NEW_PROP), ("v1/core/sched/scheduler.py", OLD_CAP, NEW_CAP))


def main() -> int:
    check = "--check" in sys.argv
    root = Path(sys.argv[sys.argv.index("--root") + 1]) / "vllm" if "--root" in sys.argv else \
        Path(list(importlib.util.find_spec("vllm").submodule_search_locations)[0])
    todo = []
    for rel, old, new in TARGETS:
        p = root / rel
        src = p.read_text()
        if MARKER in src:
            continue
        if src.count(old) != 1:
            print(f"{MARKER} {rel}: anchor found {src.count(old)} times, expected 1")
            return 2
        if "os.environ" in new and "\nimport os\n" not in src:
            print(f"{MARKER} {rel}: does not import os")
            return 2
        todo.append((p, src.replace(old, new, 1)))
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
