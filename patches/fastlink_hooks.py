#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Append the glm53_hpu.fastlink hooks to the installed vllm_gaudi (idempotent; needs the cookbook's
pipeline-parallel fix applied first). The hooks do nothing unless GLM_PP_FASTLINK=1.

Usage: fastlink_hooks.py [--check]     exit 0 applied / already applied, 1 (--check) not applied, 2 unexpected source
"""
import importlib.util
import os
import sys
from pathlib import Path

MARKER = "[glm53:fastlink]"
RUNNER_TAIL = f'''

# --- {MARKER} fast pipeline link for decode-sized forwards (glm53_hpu.fastlink; no-op unless GLM_PP_FASTLINK=1)
import sys as _glm53_sys  # noqa: E402

from glm53_hpu import fastlink as _glm53_fastlink  # noqa: E402

_glm53_fastlink.install_runner(_glm53_sys.modules[__name__])
'''
WORKER_TAIL = f'''

# --- {MARKER} pair groups for the fast pipeline link (glm53_hpu.fastlink; no-op unless GLM_PP_FASTLINK=1)
from glm53_hpu import fastlink as _glm53_fastlink  # noqa: E402

init_worker_distributed_environment = _glm53_fastlink.wrap_init(init_worker_distributed_environment)
'''
TARGETS = (("v1/worker/hpu_model_runner.py", RUNNER_TAIL, "[gaudi-fix:pipeline-parallel]"),
           ("v1/worker/hpu_worker.py", WORKER_TAIL, "[gaudi-fix:pipeline-parallel]"))


def main() -> int:
    check = "--check" in sys.argv
    spec = importlib.util.find_spec("vllm_gaudi")
    root = Path(list(spec.submodule_search_locations)[0])
    todo = []
    for rel, tail, need in TARGETS:
        p = root / rel
        src = p.read_text()
        if MARKER in src:
            continue
        if need not in src:
            print(f"{MARKER} {rel}: the pipeline-parallel fix is not applied")
            return 2
        todo.append((p, src.rstrip("\n") + "\n" + tail))
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
