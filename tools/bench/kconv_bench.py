"""Standalone microbenchmark of the nvidia-mathdx Sigma k-convolution (mode 7) and the W prep (mode 9).

One GPU, a (1, 1) mesh, the production factories (ffi.fft.make_kconv_klead_unfold /
make_kfft_klead_unfold) on placed device load tables, random operands from a fixed seed.

Tables: either the PRODUCTION tables dumped from a deck (--tables kc_tables.npz, made by
tools/bench/kconv_tables_from_dump.py) or SYNTHETIC tables of the magnetic point group
4/mm'm' (bcc Fe, m || z): the 16 k maps of D4h on the grid give the stars (parent = first
member), half of the operations are antiunitary (read the transposed partner), centroids come in
orbits of 16 laid out contiguously and an operation permutes each orbit by the group's regular
action, the umklapp phases are exp(2 pi i q.L) with small random wraps, and U_k is the op's
SU(2) spin matrix (diagonal for 4/mm'm'; --spin random gives a dense random SU(2) per op).

Timing: warm calls dispatched back to back (--reps), each timed by wall clock with a device sync,
the kernel time is taken from nsys when run under it (see --mark); the script prints
median/min per call.  --save writes the output for a cross-build comparison (--compare A B).
"""
from __future__ import annotations

import argparse, json, os, sys, time
try:
    import runtime  # noqa: F401  (LORRAX: before numpy, the OpenBLAS spin gate)
except ImportError:
    pass
import numpy as np


# ---------------------------------------------------------------- symmetry model (4/mm'm')
_SUBGROUP = None   # --subgroup: indices into the 16 ops of D4h (a subgroup; identity first)


def d4h_ops():
    """16 integer 3x3 matrices (reduced coordinates of a tetragonal/cubic grid, n1 == n2),
    and the antiunitary flag of the magnetic group 4/mm'm' (m || z)."""
    c4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    c2x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    inv = -np.eye(3, dtype=int)
    rots = []
    for a in range(4):
        r = np.linalg.matrix_power(c4, a)
        rots.append((r, a, False))                       # C4z^a: unitary
    for a in range(4):
        r = np.linalg.matrix_power(c4, a) @ c2x
        rots.append((r, a, True))                        # C2' (in-plane 2-fold) x T: antiunitary
    ops = []
    for r, a, anti in rots:
        ops.append((r, a, anti, False))
        ops.append((inv @ r, a, anti, True))             # times inversion
    if _SUBGROUP is not None:
        ops = [ops[i] for i in _SUBGROUP]
    return ops


def spin_matrix(a, anti, rng=None):
    """SU(2) part of the op: C4z^a -> diag(e^{-i a pi/4}, e^{+i a pi/4}); an antiunitary C2'.T has
    spin part (C2' spin)(i sigma_y), which is diagonal too.  rng: a random SU(2) instead."""
    if rng is not None:
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        return np.array([[q[0] + 1j * q[1], q[2] + 1j * q[3]], [-q[2] + 1j * q[3], q[0] - 1j * q[1]]])
    th = a * np.pi / 2
    u = np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2)])
    if anti:
        u = u @ np.diag([1j, -1j])
    return u


def stars(n):
    """Stars of the grid n = (n1, n2, n3) under the 16 k maps: (row, sym, parent_rows)."""
    ops = d4h_ops()
    nk = n[0] * n[1] * n[2]
    idx = np.indices(n).reshape(3, -1).T            # flat k (C order, kz fastest)
    nn = np.array(n)
    flat = lambda v: (v[..., 0] * n[1] + v[..., 1]) * n[2] + v[..., 2]
    images = np.stack([flat(np.mod(idx @ r.T, nn)) for r, _, _, _ in ops])   # (16, nk)
    row = -np.ones(nk, np.int64); sym = np.zeros(nk, np.int64); parents = []
    for k in range(nk):
        if row[k] >= 0:
            continue
        p = len(parents); parents.append(k)
        for s in range(len(ops)):
            km = images[s, k]
            if row[km] < 0:
                row[km] = p; sym[km] = s
    return row, sym, np.array(parents), ops


def synth_tables(n, mx, my, ns, spin="group", seed=1):
    """(row, trs, lsrc, rsrc, mph, nph, spin, kout, n_parent) of the model, local to one rank."""
    rng = np.random.default_rng(seed)
    row, sym, parents, ops = stars(n)
    nk = row.size
    nops = len(ops)
    assert mx % nops == 0 and my % nops == 0, "orbit blocks of 16 centroids"
    # Group multiplication table on the 16 ops (as k maps): perm of an orbit by left action.
    mats = [r for r, _, _, _ in ops]
    key = {tuple(m.ravel()): i for i, m in enumerate(mats)}
    mult = np.array([[key[tuple((mats[a] @ mats[b]).ravel())] for b in range(nops)] for a in range(nops)])
    def centroid_perm(m):                                     # (nops, m): source of centroid i
        base = np.arange(m) // nops * nops
        return base[None, :] + mult[:, np.arange(m) % nops]
    pl, pr = centroid_perm(mx), centroid_perm(my)
    anti = np.array([o[2] for o in ops])
    trs = anti[sym].astype(np.int32)
    slot = np.arange(ns)
    lsrc = (pl[sym][:, :, None] * ns + slot).reshape(nk, mx * ns).astype(np.int32)
    rsrc = (pr[sym][:, :, None] * ns + slot).reshape(nk, my * ns).astype(np.int32)
    q = np.indices(n).reshape(3, -1).T[parents][row] / np.array(n)          # parent q of each k
    Ll = rng.integers(-1, 2, size=(nops, mx, 3)); Lr = rng.integers(-1, 2, size=(nops, my, 3))
    phl = np.exp(2j * np.pi * np.einsum("ki,kmi->km", q, Ll[sym]))
    phr = np.exp(2j * np.pi * np.einsum("ki,kmi->km", q, Lr[sym]))
    mph = np.repeat(np.where(trs[:, None] > 0, np.conj(phl), phl), ns, axis=1)
    nph = np.repeat(np.where(trs[:, None] > 0, phr, np.conj(phr)), ns, axis=1)
    if ns == 1:
        U = np.ones((nops, 1, 1), complex)
    else:
        srng = np.random.default_rng(seed + 7) if spin == "random" else None
        U = np.stack([spin_matrix(o[1], o[2], srng) for o in ops])
    spin_k = U[sym]
    kout = -np.ones(nk, np.int32); kout[parents] = np.arange(parents.size)
    return dict(row=row.astype(np.int32), trs=trs, lsrc=lsrc, rsrc=rsrc, mph=mph, nph=nph,
                spin=spin_k.astype(np.complex128), kout=kout, store_rows=parents, n_parent=int(parents.size))


# ---------------------------------------------------------------- the calls
def build(args):
    import jax, jax.numpy as jnp
    from jax.sharding import Mesh
    jax.config.update("jax_enable_x64", True)
    import ffi.fft as F
    from symmetry_maps import UnfoldLoadTables
    from symmetry_maps.unfold_load import DeviceLoadTables
    n = tuple(args.grid)
    nk = int(np.prod(n))
    ns, mx, my = args.ns, args.mx, args.my
    if args.tables:
        z = np.load(args.tables)
        T = {k: z[k] for k in z.files}
        T["n_parent"] = int(T["n_parent"]); mx = int(T["lsrc"].shape[1]) // ns; my = int(T["rsrc"].shape[1]) // ns
    else:
        T = synth_tables(n, mx, my, ns, args.spin, args.seed)
    npar = T["n_parent"]
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    t7 = UnfoldLoadTables(row=T["row"], trs=T["trs"], lsrc=T["lsrc"], rsrc=T["rsrc"], mph=T["mph"],
                          nph=T["nph"], spin=T["spin"], n_parent=npar, mesh_shape=(1, 1))
    dev = lambda a: jax.device_put(np.asarray(a))
    load7 = DeviceLoadTables(*(dev(a) for a in (t7.row, t7.trs, t7.lsrc, t7.rsrc, t7.mph, t7.nph,
                                                 t7.spin, t7.spin)))
    rng = np.random.default_rng(args.seed + 100)
    cz = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(np.complex128)
    G = dev(cz(npar, mx, ns, my, ns)); Gt = dev(cz(npar, mx, ns, my, ns))
    Wr = dev(cz(nk, mx, my) / np.sqrt(nk))
    live = dev(np.asarray(T["live"], np.int32) if "live" in T else np.array([0, mx], np.int32))
    mult = -1.0 / np.sqrt(float(nk))
    conv = F.make_kconv_klead_unfold(mesh, n, t7, store_rows=T["store_rows"], norm="ortho", mult=mult)
    f7 = jax.jit(lambda G, Gt, W, live, load: conv(G, Gt, W, load=load, live=(live if args.live else None)))
    run7 = lambda: f7(G, Gt, Wr, live, load7)
    out = dict(n=n, nk=nk, ns=ns, mx=mx, my=my, n_parent=npar,
               antiunitary=int(np.sum(T["trs"])), diag_spin=bool(ns == 1 or np.all(np.abs(T["spin"][:, 0, 1]) == 0)))
    run9 = None
    if args.mode9:
        # The wedge W (scalar endpoints, pair-transpose rule on antiunitary rows): the k tables at ns=1.
        W = T if args.tables else synth_tables(n, mx, my, 1, "group", args.seed)
        if args.tables and "w_lsrc" in T:
            W = dict(row=T["w_row"], trs=T["w_trs"], lsrc=T["w_lsrc"], rsrc=T["w_rsrc"], mph=T["w_mph"],
                     nph=T["w_nph"], spin=T["w_spin"], n_parent=int(T["w_n_parent"]))
        t9 = UnfoldLoadTables(row=W["row"], trs=W["trs"], lsrc=W["lsrc"], rsrc=W["rsrc"], mph=W["mph"],
                              nph=W["nph"], spin=W["spin"], n_parent=int(W["n_parent"]), mesh_shape=(1, 1),
                              conj_trs=0, spin_r=None)
        load9 = DeviceLoadTables(*(dev(a) for a in (t9.row, t9.trs, t9.lsrc, t9.rsrc, t9.mph, t9.nph,
                                                     t9.spin, t9.spin)))
        nw = int(W["n_parent"])
        Wq = dev(cz(nw, mx, my)); Wqt = dev(cz(nw, mx, my))
        k9 = F.make_kfft_klead_unfold(mesh, n, t9, norm="ortho")
        f9 = jax.jit(lambda w, wt, live, load: k9(w, wt, load=load, live=(live if args.live else None)))
        run9 = lambda: f9(Wq, Wqt, live, load9)
    return run7, run9, out


def timeit(fn, reps):
    import jax
    jax.block_until_ready(fn()); jax.block_until_ready(fn())          # compile + warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(fn()); ts.append(time.perf_counter() - t0)
    # back-to-back stream (no host sync between launches): the device-bound time per call
    nb = max(reps, 20)
    t0 = time.perf_counter()
    o = None
    for _ in range(nb):
        o = fn()                      # the previous output is released once its call has run
    jax.block_until_ready(o)
    stream = (time.perf_counter() - t0) / nb
    del o
    return dict(median_ms=1e3 * float(np.median(ts)), min_ms=1e3 * float(np.min(ts)), stream_ms=1e3 * stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=3, default=[8, 8, 8])
    ap.add_argument("--mx", type=int, default=80)
    ap.add_argument("--my", type=int, default=640)
    ap.add_argument("--ns", type=int, default=2)
    ap.add_argument("--spin", default="group", choices=["group", "random"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tables", default="")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--live", type=int, default=1)
    ap.add_argument("--mode9", type=int, default=1)
    ap.add_argument("--mode7", type=int, default=1)
    ap.add_argument("--subgroup", default="", help="comma list of D4h op indices (a subgroup, 0 first)")
    ap.add_argument("--save", default="", help="npz path for the outputs")
    ap.add_argument("--json", default="")
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        a, b = (np.load(p) for p in args.compare)
        res = {}
        for k in a.files:
            x, y = a[k], b[k]
            den = float(np.max(np.abs(y))) or 1.0
            ew = np.abs(x - y) / (np.abs(y) + 1e-3 * den)
            res[k] = dict(max_abs=float(np.max(np.abs(x - y))), max_rel_norm=float(np.max(np.abs(x - y)) / den),
                          max_rel_elem_floor=float(np.max(ew)), bitwise=bool(np.array_equal(x, y)))
        print(json.dumps(res)); return
    import jax
    global _SUBGROUP
    if args.subgroup:
        _SUBGROUP = [int(v) for v in args.subgroup.split(",")]
    run7, run9, info = build(args)
    rec = dict(info)
    if args.subgroup:
        rec["subgroup"] = args.subgroup
    if args.mode7:
        rec["mode7"] = timeit(run7, args.reps)
    if run9 is not None:
        rec["mode9"] = timeit(run9, args.reps)
    if args.save:
        o = {"mode7": np.asarray(run7())} if args.mode7 else {}
        if run9 is not None:
            o["mode9"] = np.asarray(run9())
        np.savez(args.save, **o)
    print("KCB " + json.dumps(rec), flush=True)
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
