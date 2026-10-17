#!/usr/bin/env python3
"""kconv landing microbenchmark: modes 9 and 7 of several standalone builds of
src/ffi/cpp/cufft/kconv_mathdx_cuda_ffi.cc (``--lib tag=path``, the first is the reference), one GPU,
one process, identical inputs.  Every lib is timed SEQUENTIALLY (CUDA events on the XLA stream around
the one custom call, bench_ev.cc); ``--rounds`` alternates the libs round-robin and reports the
median of the per-round medians.  Outputs are compared BITWISE with the reference (np.array_equal).

Cases (``--cases``, comma separated):  GRID:NS[:opt]  with GRID a key of GRIDS (synthetic D4h tables,
as tools/bench/kconv_bench.py of kconv/gather) or ``fe8real`` (the REAL Fe 8^3 rank-0 tables dumped
from the deck, ``--tables``); opt ``live40`` = a padded pass with live rows [0, 40); ``xb`` = an
x-block store (x0 0, bx 8, xs 16, xn mx/16).  Mode 9 always uses scalar (n_s = 1) W tables; its
reference output is mode 7's W_R input for every lib.
"""
import argparse, ctypes, importlib.util, itertools, json, os, sys, time
import numpy as np
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

GRIDS = {
    "4^3": ((4, 4, 4), 80, 640), "6^3": ((6, 6, 6), 80, 640), "8^3": ((8, 8, 8), 80, 640),
    "9^3": ((9, 9, 9), 64, 560), "12^3": ((12, 12, 12), 32, 480), "15^3": ((15, 15, 15), 32, 240),
    "16^3": ((16, 16, 16), 32, 208), "20^3": ((20, 20, 20), 16, 208),
    "12x12x1": ((12, 12, 1), 80, 640), "10x10x4": ((10, 10, 4), 80, 640),
}


def d4h_cart():
    ops = []
    for sw in (0, 1):
        for sx, sy, sz in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), int)
            if sw:
                M[0, 1], M[1, 0] = sx, sy
            else:
                M[0, 0], M[1, 1] = sx, sy
            M[2, 2] = sz
            ops.append(M)
    return ops


def group(kg):
    """(k-actions in grid coordinates, unitary flags, multiplication table) of the 16-element group."""
    cart = d4h_cart()
    unitary = [round(np.linalg.det(M[:2, :2])) == 1 for M in cart]
    if kg[0] == kg[1] == kg[2]:
        A = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]).T / 2.0
        B = np.linalg.inv(A).T
        acts = [np.rint(np.linalg.inv(B) @ R @ B).astype(int) for R in cart]
    else:
        assert kg[0] == kg[1], "the tetragonal group needs nkx == nky"
        acts = cart
    key = {M.tobytes(): i for i, M in enumerate(cart)}
    mult = np.array([[key[(cart[g] @ cart[h]).astype(int).tobytes()] for h in range(16)] for g in range(16)])
    order = [i for i in range(16) if unitary[i]] + [i for i in range(16) if not unitary[i]]
    return acts, np.array(unitary), mult, order


def tables(kg, mx, my, ns, rng, frac_neg=0.002):
    acts, unitary, mult, order = group(kg)
    n = np.array(kg)
    nk = int(np.prod(kg))
    row = -np.ones(nk, np.int32)
    sym = np.zeros(nk, np.int32)
    reps = []
    for k in range(nk):
        if row[k] >= 0:
            continue
        p = len(reps)
        reps.append(k)
        v = np.array([k // (kg[1] * kg[2]), (k // kg[2]) % kg[1], k % kg[2]])
        for g in order:
            w = (acts[g] @ v) % n
            kk = (w[0] * kg[1] + w[1]) * kg[2] + w[2]
            if row[kk] < 0:
                row[kk], sym[kk] = p, g
    n_par = len(reps)
    assert mx % 16 == 0 and my % 16 == 0

    def cent(m):                       # P_g on orbit-contiguous centroids, merged over the spin slot
        o, h = np.divmod(np.arange(m), 16)
        perm = o[None, :] * 16 + mult[:, h]                      # (16, m)
        src = (perm[sym][:, :, None] * ns + np.arange(ns)).reshape(nk, m * ns).astype(np.int32)
        src[rng.random(src.shape) < frac_neg] = -1
        ph = np.exp(2j * np.pi * rng.random((nk, m)))
        return src, np.repeat(ph, ns, axis=1)

    lsrc, mph = cent(mx)
    rsrc, nph = cent(my)
    q, _ = np.linalg.qr(rng.standard_normal((16, ns, ns)) + 1j * rng.standard_normal((16, ns, ns)))
    kout = -np.ones(nk, np.int32)
    kout[reps] = np.arange(n_par, dtype=np.int32)
    return dict(row=row, trs=(~unitary[sym]).astype(np.int32), lsrc=lsrc, rsrc=rsrc, mph=mph, nph=nph,
                spin=q[sym].astype(np.complex128), kout=kout, n_par=n_par)


def mathdx_root():
    spec = importlib.util.find_spec("nvidia.mathdx")
    for r in spec.submodule_search_locations:
        if os.path.exists(os.path.join(r, "include", "cufftdx.hpp")):
            return r
    raise RuntimeError("no nvidia-mathdx")



def real_tables(path):
    z = np.load(path)
    T = {k: z[k] for k in z.files}
    nk = T["row"].size
    kout = -np.ones(nk, np.int32); kout[T["store_rows"]] = np.arange(T["store_rows"].size, dtype=np.int32)
    t7 = dict(row=T["row"], trs=T["trs"], lsrc=T["lsrc"], rsrc=T["rsrc"], mph=T["mph"], nph=T["nph"],
              spin=T["spin"], kout=kout, n_par=int(T["n_parent"]))
    t9 = dict(row=T["w_row"], trs=T["w_trs"], lsrc=T["w_lsrc"], rsrc=T["w_rsrc"], mph=T["w_mph"], nph=T["w_nph"],
              spin=T["w_spin"], n_par=int(T["w_n_parent"]))
    ns = int(T["spin"].shape[-1])
    return t7, t9, ns, int(T["lsrc"].shape[1]) // ns, int(T["rsrc"].shape[1]) // ns, T.get("live")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True, help="tag=path; the first is the reference")
    ap.add_argument("--ev", required=True)
    ap.add_argument("--cases", default="8^3:2")
    ap.add_argument("--modes", default="9,7")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tables", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    libs = []
    for spec in args.lib:
        tag, path = spec.split("=", 1)
        lib = ctypes.CDLL(path)
        jax.ffi.register_ffi_target(f"kc7_{tag}", jax.ffi.pycapsule(lib.KConvMathdxKleadUnfoldXBlockCudaFfi), platform="CUDA")
        jax.ffi.register_ffi_target(f"kc9_{tag}", jax.ffi.pycapsule(lib.KFftMathdxKleadUnfoldCudaFfi), platform="CUDA")
        libs.append(tag)
    ev = ctypes.CDLL(args.ev)
    ev.lrx_bench_elapsed.restype = ctypes.c_float
    jax.ffi.register_ffi_target("kc_ev", jax.ffi.pycapsule(ev.LrxBenchEvRec), platform="CUDA")
    root = mathdx_root()
    dev = jax.devices()[0]
    print(f"[bench] device {dev.device_kind}  libs {libs}", flush=True)

    def stamp(x, slot):
        return jax.ffi.ffi_call("kc_ev", jax.ShapeDtypeStruct(x.shape, x.dtype),
                                input_output_aliases={0: 0})(x, slot=np.int64(slot))

    def timeit(fn, *a):
        for _ in range(3):
            jax.block_until_ready(fn(*a))
        ts = []
        for _ in range(args.reps):
            jax.block_until_ready(fn(*a))
            ts.append(float(ev.lrx_bench_elapsed(0, 1)))
        return float(np.median(ts))

    def timed(fns, a):
        """fns: tag -> jitted fn.  Round-robin rounds, sequential; median of the round medians."""
        per = {tag: [] for tag in fns}
        for r in range(args.rounds):
            for tag in fns:
                per[tag].append(timeit(fns[tag], *a))
        return {tag: (float(np.median(v)), v) for tag, v in per.items()}

    results = []
    for case in args.cases.split(","):
        parts = case.split(":")
        gname, ns = parts[0], int(parts[1])
        opt = parts[2] if len(parts) > 2 else ""
        rng = np.random.default_rng(20261017)
        rec = dict(case=case)
        try:
            if gname == "fe8real":
                t7, t9, ns_r, mx, my, live_t = real_tables(args.tables)
                assert ns_r == ns
                kg = (8, 8, 8)
            else:
                kg, mx, my = GRIDS[gname]
                t7 = tables(kg, mx, my, ns, rng)
                t9 = tables(kg, mx, my, 1, rng)
                live_t = None
            nk = int(np.prod(kg))
            n_par, n_w = t7["n_par"], t9["n_par"]
            live = None
            if opt == "live40":
                live = jnp.asarray(np.array([0, 40], np.int32))
            elif live_t is not None:
                live = jnp.asarray(np.asarray(live_t, np.int32))
            x0, bx, xs, xn = (0, 8, 16, mx // 16) if opt == "xb" else (0, 0, 0, 0)
            rows_out = xn * bx if bx else mx
            c128 = lambda *s: jnp.asarray(rng.standard_normal(s) + 1j * rng.standard_normal(s))
            dput = lambda a: jax.device_put(jnp.asarray(a))
            common = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]), mathdx_root=root, cubin_dir="")
            rec.update(kgrid=kg, nk=nk, n_parent=n_par, mx=mx, my=my, ns=ns)
            Wp, Wt = c128(n_w, mx, my), c128(n_w, mx, my)
            T9 = {k: dput(v) for k, v in t9.items() if k not in ("n_par", "kout")}
            lv = () if live is None else (live,)
            f9s = {}
            for tag in libs:
                def f9(row, Wp, Wt, trs, lsrc, rsrc, mph, nph, sl, sr, *lv, tag=tag):
                    row = stamp(row, 0)
                    y = jax.ffi.ffi_call(f"kc9_{tag}", jax.ShapeDtypeStruct((nk, mx, my), jnp.complex128))(
                        Wp, Wt, row, trs, lsrc, rsrc, mph, nph, sl, sr, *lv,
                        scale=np.float64(1.0 / np.sqrt(nk)), conj_trs=np.int64(0), **common)
                    return stamp(y, 1)
                f9s[tag] = jax.jit(f9)
            a9 = (T9["row"], Wp, Wt, T9["trs"], T9["lsrc"], T9["rsrc"], T9["mph"], T9["nph"], T9["spin"], T9["spin"]) + lv
            ref9 = np.asarray(f9s[libs[0]](*a9))
            for tag in (libs[1:] if "9" in args.modes else []):
                rec[f"m9_{tag}_bitwise"] = bool(np.array_equal(np.asarray(f9s[tag](*a9)), ref9))
            if "9" in args.modes:
                for tag, (m, v) in timed(f9s, a9).items():
                    rec[f"m9_{tag}_ms"] = m; rec[f"m9_{tag}_rounds"] = v
            V = dput(ref9); del ref9
            if "7" in args.modes:
                Gp, Gt = c128(n_par, mx * ns, my * ns), c128(n_par, mx * ns, my * ns)
                T7 = {k: dput(v) for k, v in t7.items() if k != "n_par"}
                f7s = {}
                for tag in libs:
                    def f7(kout, Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, V, *lv, tag=tag):
                        kout = stamp(kout, 0)
                        u = jax.ffi.ffi_call(f"kc7_{tag}", jax.ShapeDtypeStruct((n_par, ns, rows_out, ns, my), jnp.complex128))(
                            Gp, Gt, row, trs, lsrc, rsrc, mph, nph, spin, kout, V, *lv,
                            scale=np.float64(0.7 / nk), conj_src=np.int64(0), x0=np.int64(x0), bx=np.int64(bx),
                            xs=np.int64(xs), xn=np.int64(xn), **common)
                        return stamp(u, 1)
                    f7s[tag] = jax.jit(f7)
                a7 = (T7["kout"], Gp, Gt, T7["row"], T7["trs"], T7["lsrc"], T7["rsrc"], T7["mph"], T7["nph"], T7["spin"], V) + lv
                ref7 = np.asarray(f7s[libs[0]](*a7))
                rec["m7_ref_maxabs"] = float(np.max(np.abs(ref7)))
                for tag in libs[1:]:
                    o = np.asarray(f7s[tag](*a7))
                    rec[f"m7_{tag}_bitwise"] = bool(np.array_equal(o, ref7))
                    if not rec[f"m7_{tag}_bitwise"]:
                        rec[f"m7_{tag}_maxdiff"] = float(np.max(np.abs(o - ref7)))
                    del o
                del ref7
                for tag, (m, v) in timed(f7s, a7).items():
                    rec[f"m7_{tag}_ms"] = m; rec[f"m7_{tag}_rounds"] = v
                del Gp, Gt, T7
            del V, Wp, Wt, T9
        except Exception as e:  # a grid/ns without a path: recorded, the sweep goes on
            rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if not k.endswith("_rounds")}
        print("[bench] " + json.dumps(short), flush=True)
        results.append(rec)
        json.dump(dict(device=dev.device_kind, libs=args.lib, results=results), open(args.out, "w"), indent=1)
    print("[bench] done", flush=True)


if __name__ == "__main__":
    main()
