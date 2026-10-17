"""Summarise a base-vs-new sweep: per-grid ms (stream = back-to-back device time per call; median = synced
wall per call), speedup, 5N log2 N TF/s of mode 7 (two 3-D transforms per column), and the output
comparison (bitwise / max |diff| / max|ref|)."""
import json, sys, os, numpy as np
d, base, new = sys.argv[1], sys.argv[2], sys.argv[3]
load = lambda v: {json.dumps(r["n"]) + f"/{r['ns']}/{r['mx']}": r for r in map(json.loads, open(f"{d}/{v}.jsonl"))}
B, N = load(base), load(new)
names = {}
for f in os.listdir(d):
    if f.startswith(base + ".") and f.endswith(".npz"):
        names[f[len(base) + 1:-4]] = f
rows = []
for key in B:
    if key not in N: continue
    b, n = B[key], N[key]
    nk, ns, mx, my = b["nk"], b["ns"], b["mx"], b["my"]
    cols = mx * my * ns * ns
    flops = 2 * cols * 5 * nk * np.log2(nk)          # inverse + forward 3-D transform per column
    r = dict(grid="x".join(map(str, b["n"])), ns=ns, mx=mx, my=my, n_parent=b["n_parent"],
             m7_base_ms=b["mode7"]["stream_ms"], m7_new_ms=n["mode7"]["stream_ms"],
             m7_base_med=b["mode7"]["median_ms"], m7_new_med=n["mode7"]["median_ms"],
             speedup=b["mode7"]["stream_ms"] / n["mode7"]["stream_ms"],
             tf5n_base=flops / b["mode7"]["stream_ms"] / 1e9, tf5n_new=flops / n["mode7"]["stream_ms"] / 1e9)
    if "mode9" in b and "mode9" in n:
        r.update(m9_base_ms=b["mode9"]["stream_ms"], m9_new_ms=n["mode9"]["stream_ms"])
    rows.append(r)
cmp = {}
for nm, f in sorted(names.items()):
    fn = f"{d}/{new}.{nm}.npz"
    if not os.path.exists(fn): continue
    a, b = np.load(fn), np.load(f"{d}/{f}")
    cmp[nm] = {k: dict(bitwise=bool(np.array_equal(a[k], b[k])),
                       rel=float(np.max(np.abs(a[k] - b[k])) / (np.max(np.abs(b[k])) or 1.0)),
                       nan=bool(np.isnan(a[k]).any())) for k in a.files}
print(json.dumps(dict(rows=rows, compare=cmp)))
