"""Rank (0, 0)'s pass-`w` load tables of the Sigma-tau mode-7 / mode-9 calls, from dump_tables.py's pickle.

The window cut is subtile_stream.window_load's: left sources [s*ns, (s+R)*ns) of the X-shard slice,
-1 outside the live rows [lo, hi), shifted by s*ns; the right tables are rank (0, 0)'s Y-shard slice.
Writes an npz that kconv_bench.py --tables reads (mesh (1, 1)).
"""
import sys, pickle, numpy as np
src, out = sys.argv[1], sys.argv[2]
wi = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rec = pickle.load(open(src, "rb"))
g = rec["gplan"][-1]["tables"]; win = rec["windows"][-1]; w = rec["kfft"][-1]["tables"]
k7 = rec["kconv"][-1]
px, py = (int(v) for v in g["mesh_shape"])
R = int(win["R"]); s, lo, hi = (int(v) for v in win["windows"][wi][:3])
ns = int(np.asarray(g["spin"]).shape[-1])
print("mesh", (px, py), "R", R, "windows", win["windows"][:6], "n", len(win["windows"]), "ns", ns,
      "n_parent", g["n_parent"], "w n_parent", w["n_parent"], "conj_trs g/w", g.get("conj_trs"), w.get("conj_trs"))
def cut(t, nsl):
    lsrc = np.asarray(t["lsrc"]); rsrc = np.asarray(t["rsrc"])
    ml = lsrc.shape[1] // px; nl = rsrc.shape[1] // py
    L = lsrc[:, :ml]; M = np.asarray(t["mph"])[:, :ml]
    width = R * nsl
    rows = L[:, s * nsl: s * nsl + width]
    live = (np.arange(width) // nsl >= lo) & (np.arange(width) // nsl < hi)
    rows = np.where((rows >= 0) & live[None, :], rows - s * nsl, -1).astype(np.int32)
    return dict(row=np.asarray(t["row"], np.int32), trs=np.asarray(t["trs"], np.int32), lsrc=rows,
                rsrc=rsrc[:, :nl].astype(np.int32), mph=M[:, s * nsl: s * nsl + width].astype(np.complex128),
                nph=np.asarray(t["nph"])[:, :nl].astype(np.complex128),
                spin=np.asarray(t["spin"]).astype(np.complex128), n_parent=int(t["n_parent"]))
G = cut(g, ns); Wt = cut(w, 1)
store_rows = np.asarray(k7["kw"]["store_rows"])
out_d = dict(G, store_rows=store_rows, live=np.array([lo, hi], np.int32))
out_d.update({"w_" + k: v for k, v in Wt.items()})
np.savez(out, **out_d)
sp = G["spin"]
print("lsrc", G["lsrc"].shape, "rsrc", G["rsrc"].shape, "valid lsrc", int((G["lsrc"] >= 0).sum()),
      "trs", int(G["trs"].sum()), "diag spin", bool(np.all(sp[:, 0, 1] == 0) and np.all(sp[:, 1, 0] == 0)) if ns > 1 else True,
      "store_rows", store_rows.size, "w lsrc", Wt["lsrc"].shape, "w trs", int(Wt["trs"].sum()))
# locality of the permutations: how far a source moves
d = np.abs(G["rsrc"] // ns - np.arange(G["rsrc"].shape[1])[None, :] // ns)
print("rsrc |src - dst| centroid displacement: median", float(np.median(d)), "p90", float(np.percentile(d, 90)), "max", int(d.max()))
