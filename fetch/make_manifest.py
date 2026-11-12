#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write the download manifest of one revision of a Hugging Face model repository.

  make_manifest.py REPO [REVISION] > MANIFEST.tsv        (REVISION: a commit hash, a tag or a branch; default main)

One line per file: SIZE <tab> KIND <tab> HASH <tab> PATH, largest files first.
  KIND sha256    an LFS file; HASH is the SHA-256 of its content
  KIND gitblob   a small file stored in git; HASH is its git blob id (SHA-1 of "blob SIZE\\0" + content)
download.sh fetches exactly these files and checks every hash. The commit the revision resolved to is printed on
stderr: pass that hash to download.sh so that later pushes to the repository cannot change what is fetched.

Standard library only; public repositories need no token. --source FILE.json also writes a small record
(repository, commit, file count, bytes).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="for example zai-org/GLM-5.3-BF16")
    ap.add_argument("revision", nargs="?", default="main")
    ap.add_argument("--source", help="also write a JSON record of what the manifest describes")
    a = ap.parse_args()
    url = (f"https://huggingface.co/api/models/{a.repo}/revision/{urllib.parse.quote(a.revision, safe='')}"
           "?blobs=true")
    req = urllib.request.Request(url, headers={"User-Agent": "make_manifest/1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        info = json.load(r)
    rows = []
    for s in info["siblings"]:
        path = s["rfilename"]
        lfs = s.get("lfs")
        if lfs:
            rows.append((int(lfs["size"]), "sha256", lfs["sha256"], path))
        else:
            rows.append((int(s["size"]), "gitblob", s["blobId"], path))
    rows.sort(key=lambda r: (-r[0], r[3]))
    for size, kind, digest, path in rows:
        print(f"{size}\t{kind}\t{digest}\t{path}")
    total = sum(r[0] for r in rows)
    print(f"{a.repo} at {info['sha']}: {len(rows)} files, {total} bytes", file=sys.stderr)
    if a.source:
        with open(a.source, "w") as f:
            json.dump({"repo": a.repo, "revision": info["sha"], "files": len(rows), "bytes": total}, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
