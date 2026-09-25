"""Benchmark: one τ node of the mixed-basis pair convolution on a real crystal's shapes, synthetic values.

Every arm runs in its own process (per-GPU peak memory is the process's):

    lx run -N 1 -G 4 -n 4 python3 -u tests/bench/bench_mixed_basis_pair_conv.py \
        --wfn WFN.h5 --ns 1 --arm router --out arm.json [--nc N --J J] [--stages]

* shapes and symmetry come from the WFN header: the ψ spheres at every full-grid k
  and at the parents, the SymMaps tables (the typed G-space transport), the FFT box,
  and the χ spheres at the q IBZ at ``ecutwfc`` (``common.coulomb_sphere``, the ζ
  sphere's default cutoff); ``--ecut-scale s`` keeps only ``|k+G|² ≤ s·ecutwfc`` on
  both and ``--box`` replaces the box (the naive arm's reduced size);
* values are synthetic: ψ random on the parent spheres, A = Gc and C = Gv built by
  ``gw.greens_function_kernel.build_G_parents`` (one face GEMM per operand) with
  real band weights (conduction for A, valence for C), so antiunitary rows read
  conj(G);
* arms: ``router`` (LocalFourierPlan CUDA leg + mathdx mode 6), ``xla`` (the gather
  + jnp.fft fallback), ``naive`` (the fallback with every column of the rank in one
  batch: A_k(r, r') and C_k(r, r') for all k materialized, N_k·N_r²/P per rank);
* reports: plan receipt, compact-build wall (cold, warm), kernel wall by stage
  (cold one-shot including compiles, then warm one-shot), per-GPU peak bytes, a
  checksum for cross-arm parity; ``--stages`` adds warm per-sub-stage timings of one
  middle batch and one expand chunk on the rank's own GPU, and the all-to-all alone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import numpy as np  # noqa: E402  (after the runtime: its BLAS thread setting must come first)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402


def say(*a):
    if jax.process_index() == 0:
        print("[pairconv-bench]", *a, flush=True)


def _filter(gvecs, ngk, frac, bvec, cut):
    """Keep the live slots with |k+G|² ≤ cut (Cartesian), in their order (``cut`` sits in a gap
    of the norm spectrum, see ``_gap_cut``, so rotated images agree on membership)."""
    out = np.full_like(gvecs, 0)
    n = np.zeros(len(ngk), np.int64)
    for i in range(len(ngk)):
        g = gvecs[i, :ngk[i]]
        kg = (frac[i][None, :] + g) @ bvec
        keep = g[np.einsum("gi,gi->g", kg, kg) <= cut * (1 + 1e-12)]
        out[i, :len(keep)] = keep
        n[i] = len(keep)
    w = int(n.max())
    return out[:, :w], n


def setup(args, mesh):
    from file_io import WfnLoader
    from common.coulomb_sphere import compute_per_q_bare_coulomb_components
    from common.gvec_fft_box import build_sphere_box_index
    from symmetry_maps import bgw_integer_q_to_fractional
    from vcoul import CoulombGeometry
    import h5py
    from gw.mixed_basis_pair_convolution import PairOperand, SphereSet, SphereTransport
    with h5py.File(args.wfn, "r") as f:
        ecut = float(f["mf_header/kpoints/ecutwfc"][()])
    with WfnLoader(args.wfn, backend="eager") as w:
        sym = w.symmetry()
        kgrid = tuple(int(v) for v in w.kgrid)
        box = tuple(int(v) for v in w.fft_grid)
        bvec = np.asarray(CoulombGeometry.from_wfn(w).bvec, dtype=np.float64)
        kf = np.asarray(w.kvecs(k="full_bz"))
        gf, nf = np.asarray(w.gvecs(k="full_bz")), np.asarray(w.ngk_valid(k="full_bz"))
        kp = np.asarray(w.kvecs(k=sym.parent_k_domain))
        gp, npar = np.asarray(w.gvecs(k=sym.parent_k_domain)), np.asarray(w.ngk_valid(k=sym.parent_k_domain))
    cut = args.ecut_scale * ecut
    if args.ecut_scale < 1.0:
        norms = np.concatenate([np.einsum("gi,gi->g", *(2 * [(kf[i][None, :] + gf[i, :nf[i]]) @ bvec]))
                                for i in range(len(nf))])
        u = np.unique(np.round(norms, 8))
        j = int(np.searchsorted(u, cut))
        cut = 0.5 * (u[max(j - 1, 0)] + u[min(j, len(u) - 1)])        # mid-gap: no shell on the edge
        gf, nf = _filter(gf, nf, kf, bvec, cut)
        gp, npar = _filter(gp, npar, kp, bvec, cut)
        wdt = max(gf.shape[1], gp.shape[1])
        gf = np.pad(gf, ((0, 0), (0, wdt - gf.shape[1]), (0, 0)))
        gp = np.pad(gp, ((0, 0), (0, wdt - gp.shape[1]), (0, 0)))
    if args.box:
        box = tuple(int(v) for v in args.box.split(","))
    q_frac = bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, kgrid)
    pkg = compute_per_q_bare_coulomb_components(fft_grid=box, bvec=bvec, q_irr_frac=q_frac,
                                                vcoul_cutoff_ry=cut, sys_dim=3)
    out = SphereSet(np.asarray(pkg["gvec_components_padded"]).transpose(0, 2, 1),
                    np.asarray(pkg["ngk_per_q"]), q_frac)
    ns = int(args.ns)
    n_sp = int(np.asarray(sym.sym_matrices).shape[0])
    sidx = np.asarray(sym.sym_idx_k, np.int32)
    spin = (np.asarray(sym.spinor_action(sidx, nspinor=2)) if ns == 2
            else np.ones((len(sidx), 1, 1), np.complex128))
    plan = SimpleNamespace(irr_idx=np.asarray(sym.irr_idx_k, np.int32), sym_idx=sidx,
                           spin_action_full=spin, k_parent_frac=kp, n_sym_spatial=n_sp,
                           spatial_ops=np.asarray(sym.sym_matrices)[:n_sp],
                           translations=np.asarray(sym.translations)[:n_sp],
                           mesh_xy=mesh, n_full=len(sidx))
    children = SphereSet(gf, nf, kf)
    sidx_par = build_sphere_box_index(gp, box, gp.shape[1], ngk_valid=npar)
    tr = SphereTransport.typed(plan, fft_grid=box, parent_sphere_index=sidx_par, children=children)
    op = PairOperand(children, tr)
    return dict(op=op, out=out, kgrid=kgrid, box=box, plan=plan, npar=npar, gp=gp, ns=ns,
                ecut=ecut, cut=cut)


def compact_build(s, mesh, M, nb, nv, seed=5):
    """A = Gc, C = Gv at the parents from random ψ on the parent spheres: one face GEMM each."""
    from distrib_la import gemm_plan
    from gw.greens_function_kernel import build_G_parents
    ns, npar = s["ns"], len(s["npar"])
    rng = np.random.default_rng(seed)
    live = np.arange(M)[None, :] < s["npar"][:, None]                    # (n_par, M)

    def cb(idx):                        # this shard of random ψ, seeded by its offsets
        starts = [sl.start or 0 for sl in idx]
        shape = [len(range(*sl.indices(n))) for sl, n in zip(idx, (npar, ns, M, nb))]
        r = np.random.default_rng([seed] + starts)
        return r.standard_normal(shape) + 1j * r.standard_normal(shape)
    mun = NamedSharding(mesh, P(None, None, "x", "y"))
    x_mun = jax.make_array_from_callback((npar, ns, M, nb), mun, cb)
    mask = jnp.asarray(live)[:, None, :, None]
    x_mun = jax.jit(lambda a: jnp.where(mask, a, 0), out_shardings=mun)(x_mun)
    x_nmu = jax.jit(lambda a: jnp.transpose(a, (0, 3, 1, 2)),
                    out_shardings=NamedSharding(mesh, P(None, "x", None, "y")))(x_mun)
    eps = np.sort(rng.standard_normal(nb)) * 0.2
    wc = jnp.asarray(np.broadcast_to(np.where(np.arange(nb) >= nv, np.exp(-np.abs(eps)), 0.0), (npar, nb)))
    wv = jnp.asarray(np.broadcast_to(np.where(np.arange(nb) < nv, np.exp(-np.abs(eps)), 0.0), (npar, nb)))
    g = gemm_plan(mesh, m=M * ns, k=nb, n=M * ns, nq=npar, dtype=jnp.complex128, layout="face")
    plan = s["plan"]

    def build():
        A = build_G_parents(x_mun, x_nmu, phases=wc, layout="face", gemm=g, k_unfold_plan=plan,
                            real_weights=True).G
        C = build_G_parents(x_mun, x_nmu, phases=wv, layout="face", gemm=g, k_unfold_plan=plan,
                            real_weights=True).G
        return A, C
    t0 = time.perf_counter()
    A, C = build()
    jax.block_until_ready((A, C))
    cold = time.perf_counter() - t0
    warm = []
    for _ in range(3):
        del A, C
        t0 = time.perf_counter()
        A, C = build()
        jax.block_until_ready((A, C))
        warm.append(time.perf_counter() - t0)
    flops = 2 * 8.0 * npar * (M * ns) ** 2 * nb
    return A, C, dict(cold_s=cold, warm_s=min(warm), flops=flops, M=M, nb=nb)


def checksum(conv, X):
    """Per-q Frobenius norms and one 4x4 corner, gathered (small)."""
    x = conv.strip(X)
    return dict(norm=np.linalg.norm(x, axis=(1, 2)).tolist(),
                corner=[[complex(v).real, complex(v).imag] for v in x[0, :4, :4].ravel()])


def stages(conv, reps=3):
    """Warm timings of one middle batch's and one expand chunk's sub-stages on this rank's GPU."""
    import gw.mixed_basis_pair_convolution as mb
    ns, nk, nr, J = conv.ns, conv.nk, conv.nr, conv.chunks.J
    dev = jax.local_devices()[0]
    kb = conv.kbox[0]
    nbox = int(np.prod(kb))
    M = conv.width_carrier[0]
    nq = conv.nq
    t = conv._tables[0]
    put = lambda a: jax.device_put(np.asarray(a), dev)
    zc = lambda shape: jax.device_put(jnp.zeros(shape, jnp.complex128), dev)

    def timeit(fn, *a):
        out = fn(*a)
        jax.block_until_ready(out)
        best = 1e30
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn(*a)
            jax.block_until_ready(out)
            best = min(best, time.perf_counter() - t0)
        return best
    res = {}
    H = zc((nk, M, ns, ns, J))
    lsrc, lph, spin = put(t["csrc"]), put(t["mph"]), put(t["spin"])
    row = jax.jit(lambda h, a, b, c: mb._row_half(h, a, b, c, n_s=ns))
    res["row_half_gather"] = timeit(row, H, lsrc, lph, spin)
    x = zc((nk, ns, 2 * J, ns) + kb)
    plan_row = conv._plan(sign=+1, norm="forward", in_support=conv.sup[0])
    res["plan_p2r"] = timeit(jax.jit(plan_row), x)
    D = zc((nk, ns, 2 * J, ns, nr))
    F = zc((nk, nr))
    if conv.backend == "router":
        from ffi.fft import make_fused_conv_kplane
        kconv = make_fused_conv_kplane(conv.mesh, conv.kgrid, ns, perm_l=list(range(ns)),
                                       phase_l=[1] * ns, perm_r=list(range(ns)), phase_r=[1] * ns)
        f = jax.jit(lambda d, f_: kconv(d.reshape(nk, 1, ns, 2 * J, ns, nr), f_.reshape(nk, 1, nr)))
    else:
        from common.fft_helpers import local_fftn3, local_ifftn3

        def f(d, f_):
            a = (d * f_[:, None, None, None, :]).reshape(conv.kgrid + (ns, 2 * J, ns, nr))
            aR = local_ifftn3(a, axes=(0, 1, 2), norm="backward").reshape(nk, ns, 2 * J, ns, nr)
            X = sum(aR[:, s1, :J, s2] * jnp.conj(aR[:, s1, J:, s2]) for s1 in range(ns) for s2 in range(ns))
            return local_fftn3(X.reshape(conv.kgrid + (J, nr)), axes=(0, 1, 2), norm="backward")
        f = jax.jit(f)
    res["kconv"] = timeit(f, D, F)
    U = zc((nk, J, nr))
    Q = zc((nq, nr))
    rows = put(conv.q_rows)
    sel = jax.jit(lambda u, q, r: (jnp.take(u, r, axis=0) * q[:, None, :]).reshape((nq, J) + conv.fft_grid))
    res["q_select_phase"] = timeit(sel, U, Q, rows)
    plan_out = conv._plan(sign=-1, norm="backward", out_support=conv.sup_out)
    Y = zc((nq, J) + conv.fft_grid)
    res["plan_r2G"] = timeit(jax.jit(plan_out), Y)
    # expand: one k chunk of the column half and p'->r' for this rank's rows
    kc, ml = conv.chunks.kc, M // conv.P
    g = zc((kc, ml, ns, M, ns))
    col = jax.jit(lambda g_, a, b, c: mb._column_half(g_, a, b, c, n_s=ns))
    res["column_half_gather"] = timeit(col, g, put(t["csrc"][:kc]), put(t["nph"][:kc]), put(t["spin"][:kc]))
    plan_col = conv._plan(sign=-1, norm="backward", in_support=conv.sup[0])
    xc = zc((kc, ml, ns, ns) + kb)
    res["plan_p2r_prime"] = timeit(jax.jit(plan_col), xc)
    res["shapes"] = dict(J=J, kc=kc, nbox=nbox, M=M, nq=nq, nk=nk, nr=nr, ns=ns)
    return res


def alltoall_time(conv, reps=3):
    """The r'-sharding all-to-all alone at one k chunk's payload."""
    from common.shard_map import shard_map
    ns, M, P_, kc = conv.ns, conv.width_carrier[0], conv.P, conv.chunks.kc
    cols = conv.cols_chunk
    shape = (kc, M, ns, ns, P_, cols)                   # global: rows over P
    spec = P(None, ("x", "y"), None, None, None, None)
    y = jax.jit(lambda: jnp.zeros(shape, jnp.complex128), out_shardings=NamedSharding(conv.mesh, spec))()
    f = jax.jit(shard_map(lambda a: jax.lax.all_to_all(a, ("x", "y"), split_axis=4, concat_axis=1, tiled=True),
                          mesh=conv.mesh, in_specs=spec, out_specs=P(None, None, None, None, ("x", "y"), None),
                          check_vma=False))
    jax.block_until_ready(f(y))
    best = 1e30
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(y))
        best = min(best, time.perf_counter() - t0)
    per_rank = kc * (M // P_) * ns * ns * P_ * cols * 16
    return dict(seconds=best, bytes_per_rank=per_rank, bytes_sent_per_rank=per_rank * (P_ - 1) // P_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn", required=True)
    ap.add_argument("--ns", type=int, default=1)
    ap.add_argument("--arm", choices=("router", "xla", "naive"), default="router")
    ap.add_argument("--nc", type=int, default=None)
    ap.add_argument("--J", type=int, default=None)
    ap.add_argument("--ecut-scale", type=float, default=1.0)
    ap.add_argument("--box", default=None)
    ap.add_argument("--nb", type=int, default=312)
    ap.add_argument("--nv", type=int, default=16)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from gw.mixed_basis_pair_convolution import MixedBasisPairConvolution
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    t0 = time.perf_counter()
    s = setup(args, mesh)
    t_tables = time.perf_counter() - t0
    backend = "router" if args.arm == "router" else "xla"
    t0 = time.perf_counter()
    kw = dict(kgrid=s["kgrid"], fft_grid=s["box"], left=s["op"], right=s["op"], out=s["out"],
              backend=backend)
    if args.arm == "naive":
        probe = MixedBasisPairConvolution(mesh, **kw, budget_bytes=int(1e15), chunks=(1, 1))
        conv = MixedBasisPairConvolution(mesh, **kw, budget_bytes=int(1e15),
                                         chunks=(1, probe.cols_rank))
    elif args.nc or args.J:
        conv = MixedBasisPairConvolution(mesh, **kw, chunks=(args.nc, args.J))
    else:
        conv = MixedBasisPairConvolution(mesh, **kw)
    t_plan = time.perf_counter() - t0
    say(conv.describe())
    rec = dict(arm=args.arm, ns=args.ns, wfn=args.wfn, ecut_scale=args.ecut_scale, box=list(s["box"]),
               kgrid=list(s["kgrid"]), n_parent=len(s["npar"]), nq=conv.nq, P=conv.P,
               widths=list(conv.width_carrier), width_out=conv.mo_axis.carrier,
               kbox=[list(k) for k in conv.kbox], kbox_out=list(conv.kbox_out),
               chunks=dict(n_c=conv.chunks.n_c, J=conv.chunks.J, kc=conv.chunks.kc, qc=conv.chunks.qc,
                           n_batch=conv.n_batch, nr_carrier=conv.nr_carrier),
               model=dict(resident=conv.chunks.bytes_resident, expand=conv.chunks.bytes_expand,
                          middle=conv.chunks.bytes_middle, final=conv.chunks.bytes_final,
                          hwm=conv.chunks.hwm, target=conv.chunks.target),
               receipt=conv.describe(), t_tables=t_tables, t_plan=t_plan)
    A, C, rec["compact"] = compact_build(s, mesh, conv.width_carrier[0], args.nb, args.nv)
    say(f"compact build: cold {rec['compact']['cold_s']:.3f} s, warm {rec['compact']['warm_s']:.4f} s, "
        f"{rec['compact']['flops'] / rec['compact']['warm_s'] / 1e12 / conv.P:.2f} TF/s per GPU")
    stats = jax.local_devices()[0].memory_stats() or {}
    rec["bytes_in_use_before"] = int(stats.get("bytes_in_use", 0))
    runs = []
    for i in range(1 + int(args.warm)):
        tm = {}
        t0 = time.perf_counter()
        X = conv(A, C, timings=tm)
        jax.block_until_ready(X)
        tm["total"] = time.perf_counter() - t0
        runs.append(dict(kind="cold one-shot" if i == 0 else "warm one-shot", **tm))
        say(f"{runs[-1]['kind']}: " + ", ".join(f"{k} {v:.3f} s" for k, v in tm.items()
                                                 if not k.endswith("_chunks")))
        if conv.chunks.n_c > 1:
            say("per r' chunk: expand " + " ".join(f"{v:.3f}" for v in tm["expand_chunks"])
                + " s; middle " + " ".join(f"{v:.3f}" for v in tm["middle_chunks"]) + " s")
        if i < args.warm:
            del X
    rec["runs"] = runs
    stats = jax.local_devices()[0].memory_stats() or {}
    rec["memory_stats"] = {k: int(v) for k, v in stats.items() if isinstance(v, (int, np.integer))}
    rec["peak_bytes_in_use"] = int(stats.get("peak_bytes_in_use", 0))
    rec["bytes_limit"] = int(stats.get("bytes_limit", 0))
    rec["checksum"] = checksum(conv, X)
    say(f"peak {rec['peak_bytes_in_use'] / 1e9:.2f} GB per GPU (in use before the kernel "
        f"{rec['bytes_in_use_before'] / 1e9:.2f} GB; model HWM {conv.chunks.hwm / 1e9:.2f} GB)")
    del X
    if args.stages:
        rec["stages"] = stages(conv)
        rec["alltoall"] = alltoall_time(conv)
        say("stages", json.dumps(rec["stages"]), "a2a", json.dumps(rec["alltoall"]))
    if jax.process_index() == 0:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:                                   # noqa: BLE001 — print before the exit
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        finalize_process(rc)
