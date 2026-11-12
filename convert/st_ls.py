#!/usr/bin/env python3
"""List the tensors of safetensors files, grouped by name pattern (stdlib only).  st_ls.py FILE_OR_DIR [--all]"""
import collections, glob, json, os, re, struct, sys
paths = []
for a in sys.argv[1:]:
    if a.startswith("--"): continue
    paths += sorted(glob.glob(os.path.join(a, "*.safetensors"))) if os.path.isdir(a) else [a]
h = {}
for p in paths:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        d = json.loads(f.read(n))
    d.pop("__metadata__", None)
    h.update(d)
if "--all" in sys.argv:
    for k in sorted(h): print(k, h[k]["dtype"], h[k]["shape"])
    sys.exit()
grp = collections.OrderedDict()
for k in sorted(h, key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]):
    g = re.sub(r"experts\.\d+\.", "experts.E.", k)
    grp.setdefault(g, []).append(k)
tot = 0
for g, ks in grp.items():
    m = h[ks[0]]
    nb = sum((h[k]["data_offsets"][1] - h[k]["data_offsets"][0]) for k in ks)
    tot += nb
    print(f"{len(ks):4d} {g:72s} {m['dtype']:8s} {str(m['shape']):16s} {nb/1e6:10.1f} MB")
print(f"total {len(h)} tensors, {tot/1e9:.3f} GB")
