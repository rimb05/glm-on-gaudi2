#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""GLM_DSA_V2=1: install this project's glm53_hpu/dsa.py (the long-context revision of the cookbook's sparse
attention module) over the image's glm5_hpu/dsa.py. The original is kept next to it as dsa.py.orig. Idempotent.

Usage: dsa_override.py          exit 0 installed / already installed / switched off, 2 unexpected layout
"""
import importlib.util
import os
import shutil
import sys
from pathlib import Path

MARKER = "long-context revision"


def main() -> int:
    if os.environ.get("GLM_DSA_V2", "0") != "1":
        print("[glm53:dsa-v2] off")
        return 0
    spec = importlib.util.find_spec("glm5_hpu")
    if spec is None or not spec.submodule_search_locations:
        print("[glm53:dsa-v2] glm5_hpu is not installed")
        return 2
    dst = Path(list(spec.submodule_search_locations)[0]) / "dsa.py"
    src = Path(__file__).resolve().parent.parent / "glm53_hpu" / "dsa.py"
    if not dst.is_file() or not src.is_file():
        print(f"[glm53:dsa-v2] missing {dst} or {src}")
        return 2
    new = src.read_text()
    compile(new, str(dst), "exec")
    if MARKER not in new:
        print("[glm53:dsa-v2] the source file is not the revision")
        return 2
    if dst.read_text() == new:
        print("[glm53:dsa-v2] installed")
        return 0
    if not dst.with_name("dsa.py.orig").exists():
        shutil.copy2(dst, dst.with_name("dsa.py.orig"))
    tmp = dst.with_name("dsa.py.glm53.tmp")
    tmp.write_text(new)
    os.chmod(tmp, dst.stat().st_mode & 0o7777)
    os.replace(tmp, dst)
    for pyc in (dst.parent / "__pycache__").glob("dsa.*.pyc"):
        pyc.unlink(missing_ok=True)
    print(f"[glm53:dsa-v2] installed over {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
