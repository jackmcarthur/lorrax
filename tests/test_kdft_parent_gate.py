"""The ROCM ports of mathdx modes 1 and 10 against the XLA route on the same GCD.

Mode 1 (``lorrax_kdft_kconv_parent``): ``ffi.fft.make_fused_conv_kparent`` built by default (the kdft
target on a ROCm mesh; a case that silently took XLA fails) and inside ``ffi.fft.xla_reference()``,
applied to the same random parent projectors and random valid typed tables (parent rows, symmetry
operations with random owner-local maps, lattice wraps, antiunitary rows, dense open-spin coefficients)
with random vertices in {1, i, -1, -i}, both ``centroid_major`` layouts.  Mode 10
(``lorrax_kdft_plane_fft_gather``): ``ffi.fft.make_plane_fft_gather`` on random cylinders (coprime
splits, prime powers <= 40, partial rows, batch axes, the slab form with a traced start).  Both:
``max|a - b| <= 1e-12 max|b|``; plus the named refusals.  A 2x2 mesh also runs mode 1 sharded through
the caller's ``shard_map`` (the isdf.core pair spec).  ``--bench``: per call, kdft vs the XLA route, at
Fe- and hsuite-shaped operands.  ROCm only: ``python -m tests.test_kdft_parent_gate``.
"""
import argparse
import sys
import time

TOL = 1e-12


def _timed(fn, *args, reps=10):
    import jax
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(reps):
        y = fn(*args)
    jax.block_until_ready(y)
    return (time.perf_counter() - t0) / reps * 1e3


def _parent_case(rng, kg, ns, n_parent, mu, nu, ops, dev=None, dense=True):
    """Random D_l, D_r (n_parent, ns, mu, ns, nu) and the ten typed tables of isdf.core."""
    import jax
    import numpy as np
    nk = int(np.prod(kg))
    rnd = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    put = (lambda x: jax.device_put(x, dev)) if dev is not None else (lambda x: x)
    D_l, D_r = put(rnd(n_parent, ns, mu, ns, nu)), put(rnd(n_parent, ns, mu, ns, nu))
    irr = rng.integers(0, n_parent, nk).astype(np.int32)
    irr[:n_parent] = np.arange(n_parent)
    sym = rng.integers(0, ops, nk).astype(np.int32)
    left = np.stack([rng.permutation(mu) for _ in range(ops)]).astype(np.int32)
    right = np.stack([rng.permutation(nu) for _ in range(ops)]).astype(np.int32)
    left[0], right[0] = np.arange(mu), np.arange(nu)
    L = rng.integers(-1, 2, (ops, mu, 3)).astype(np.float64)
    R = rng.integers(-1, 2, (ops, nu, 3)).astype(np.float64)
    q = rng.random((n_parent, 3))
    trs = (rng.random(nk) < 0.4).astype(np.int32)
    coef = lambda: rnd(nk, ns * ns, ns * ns) if dense else np.tile(np.eye(ns * ns, dtype=np.complex128), (nk, 1, 1))
    tables = tuple(put(np.asarray(t)) for t in (irr, sym, left, right, L, R, q, trs, coef(), coef()))
    qv = np.asarray([1, 1j, -1, -1j])
    tabs = dict(perm_l=rng.permutation(ns), phase_l=qv[rng.integers(0, 4, ns)],
                perm_r=rng.permutation(ns), phase_r=qv[rng.integers(0, 4, ns)])
    return D_l, D_r, tables, tabs


def _plane_case(rng, nb, nc, occ_frac, lead, n_col_pad=0, dev=None):
    """A random cylinder F (*lead, n_col) of an (nb, nc) plane: whole dead rows and random cells."""
    import jax
    import numpy as np
    occ = rng.random(nb * nc) < occ_frac
    dead = rng.random(nb) < 0.25
    occ.reshape(nb, nc)[dead] = False
    n_col = int(occ.sum()) + int(n_col_pad)
    pfc = np.full(nb * nc, n_col, np.int64)
    pfc[occ] = rng.permutation(n_col)[:int(occ.sum())] if n_col_pad else np.arange(int(occ.sum()))
    F = rng.standard_normal(tuple(lead) + (n_col,)) + 1j * rng.standard_normal(tuple(lead) + (n_col,))
    return pfc, n_col, (jax.device_put(F, dev) if dev is not None else F)


def gate_cases(mesh):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from ffi import fft as F_
    from tests.test_kconv_xla_gate import _rel
    dev = jax.local_devices()[0]
    rng = np.random.default_rng(20261016)
    out = {}
    # ---- mode 1
    for kg, ns, n_parent, mu, nu, ops, cm, dense in (
            ((4, 4, 4), 2, 10, 7, 9, 3, True, True), ((4, 4, 4), 2, 10, 7, 9, 3, False, True),
            ((5, 5, 1), 2, 6, 11, 5, 2, True, True), ((3, 2, 2), 1, 4, 6, 7, 2, True, True),
            ((2, 2, 2), 4, 3, 3, 5, 2, True, True), ((2, 2, 2), 4, 3, 3, 5, 2, False, False),
            ((1, 1, 1), 2, 1, 4, 3, 1, True, True), ((6, 6, 6), 2, 9, 3, 5, 4, True, True),
            ((8, 8, 1), 2, 12, 5, 9, 4, True, False), ((3, 3, 3), 1, 5, 17, 13, 3, False, True)):
        tag = f"mode1 {kg[0]}x{kg[1]}x{kg[2]} ns{ns} np{n_parent} mu{mu} nu{nu} ops{ops} cm{int(cm)}{'' if dense else ' eye'}"
        if F_.rocm_parent_target(mesh, kg, ns) is None:
            out[f"{tag} ROUTE"] = float("inf")
            continue
        D_l, D_r, tables, tabs = _parent_case(rng, kg, ns, n_parent, mu, nu, ops, dev, dense)
        f_k = F_.make_fused_conv_kparent(mesh, kg, ns, (mu, nu), centroid_major=cm, **tabs)
        with F_.xla_reference():
            f_x = F_.make_fused_conv_kparent(mesh, kg, ns, (mu, nu), centroid_major=cm, **tabs)
        out[tag] = _rel(jax.jit(f_k)(D_l, D_r, tables), jax.jit(f_x)(D_l, D_r, tables))
    # mode 1 sharded on the mesh (isdf.core's pair spec), when the mesh has more than one device
    if mesh.devices.size > 1:
        from jax import shard_map
        from jax.sharding import NamedSharding
        px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
        kg, ns, n_parent, ml, nl, ops = (4, 4, 4), 2, 10, 6, 5, 3
        D_l, D_r, tables, tabs = _parent_case(rng, kg, ns, n_parent, ml * px, nl * py, ops)
        _, _, tl, _ = _parent_case(rng, kg, ns, n_parent, ml, nl, ops)   # owner-local maps
        spec = P(None, None, "x", None, "y")
        put = lambda x: jax.device_put(x, NamedSharding(mesh, spec))
        D_l, D_r = put(D_l), put(D_r)
        res = []
        for ref in (False, True):
            ctx = F_.xla_reference() if ref else __import__("contextlib").nullcontext()
            with ctx:
                f = F_.make_fused_conv_kparent(mesh, kg, ns, (ml, nl), centroid_major=True, **tabs)
            g = jax.jit(shard_map(lambda a, b: f(a, b, tuple(jnp.asarray(t) for t in tl)), mesh=mesh,
                                  in_specs=(spec, spec), out_specs=P(None, "x", "y"), check_vma=False))
            res.append(g(D_l, D_r))
        rel = jax.jit(_sharded_rel, out_shardings=NamedSharding(mesh, P()))(*res)
        out[f"mode1 sharded {px}x{py} 4x4x4 ns2 cm1"] = float(rel)
    for kg, ns, gate in (((41, 1, 1), 2, "kdft-axis"), ((10, 10, 10), 4, "kdft-kconv-parent-tile")):
        why = F_.kdft_parent_refusal(kg, ns)
        out[f"mode1 refusal {kg} ns{ns} -> GATE {gate}"] = 0.0 if f"GATE {gate}" in why else float("inf")
    # ---- mode 10
    for nb, nc, frac, lead, pad in ((24, 30, 0.6, (4, 3), 0), (25, 25, 0.7, (3, 5, 2), 0),
                                    (54, 54, 0.5, (2, 3), 3), (45, 36, 0.7, (2, 2, 2), 0),
                                    (16, 16, 0.9, (5,), 0), (32, 30, 0.7, (2, 3), 0), (35, 12, 0.4, (2, 3, 2, 2), 0), (2, 3, 1.0, (7, 2), 0)):
        tag = f"mode10 plane {nb}x{nc} lead{lead}"
        pfc, n_col, Fc = _plane_case(rng, nb, nc, frac, lead, pad, dev)
        f_k = F_.make_plane_fft_gather(mesh, pfc, n_col, (nb, nc))
        with F_.xla_reference():
            f_x = F_.make_plane_fft_gather(mesh, pfc, n_col, (nb, nc))
        if nb in F_.KDFT_PLANE_FFT_SLOW_AXES or nc in F_.KDFT_PLANE_FFT_SLOW_AXES:
            out[f"{tag} routed XLA (slower rule)"] = 0.0 if getattr(f_k, "route", None) == "xla" else float("inf")
            continue
        if getattr(f_k, "route", None) != "kdft":
            out[f"{tag} ROUTE"] = float("inf")
            continue
        out[tag] = _rel(jax.jit(f_k)(Fc), jax.jit(f_x)(Fc))
        if len(lead) >= 3:
            S = lead[1]
            for st, size in ((1, S - 1), (S, 1), (-2, S)):         # in range, clamped high, clamped low
                a = jax.jit(lambda F, s: f_k(F, s, size))(Fc, jnp.int32(st))
                b = jax.jit(lambda F, s: f_x(F, s, size))(Fc, jnp.int32(st))
                out[f"{tag} slab start {st} size {size}"] = _rel(a, b)
    for nb, nc, gate in ((64, 64, "kdft-plane-split"), (41, 10, "kdft-plane-split"), (80, 80, "kdft-plane-residency")):
        why = F_.kdft_plane_fft_refusal(nb, nc, nb)
        out[f"mode10 refusal ({nb},{nc}) -> GATE {gate}"] = 0.0 if f"GATE {gate}" in why else float("inf")
    return out


def _sharded_rel(a, b):
    import jax.numpy as jnp
    if a.shape != b.shape:
        return jnp.asarray(jnp.inf)
    error, scale = jnp.max(jnp.abs(a - b)), jnp.max(jnp.abs(b))
    finite = jnp.isfinite(a).all() & jnp.isfinite(b).all() & jnp.isfinite(error) & jnp.isfinite(scale)
    return jnp.where(finite, error / jnp.maximum(scale, 1e-300), jnp.inf)


def bench(mesh, parent_shapes, plane_shapes, reps):
    import jax
    import numpy as np
    from ffi import fft as F_
    dev = jax.local_devices()[0]
    rng = np.random.default_rng(7)
    print("bench mode1  k-grid  ns  np   mu   nu ops cm |  XLA ms   kdft ms   (x)", flush=True)
    for kg, ns, n_parent, mu, nu, ops, cm in parent_shapes:
        D_l, D_r, tables, tabs = _parent_case(rng, kg, ns, n_parent, mu, nu, ops, dev)
        scale = F_.conv_kpair_scale("forward", int(np.prod(kg)), 1.0)
        f_k = jax.jit(F_._kdft_parent_fn(F_.KDFT_KCONV_PARENT_TARGET, kg, ns, tabs["perm_l"], tabs["phase_l"],
                                         tabs["perm_r"], tabs["phase_r"], scale, cm))
        with F_.xla_reference():
            f_x = jax.jit(F_.make_fused_conv_kparent(mesh, kg, ns, (mu, nu), centroid_major=cm, **tabs))
        tx, tk = _timed(f_x, D_l, D_r, tables, reps=reps), _timed(f_k, D_l, D_r, tables, reps=reps)
        print(f"bench mode1  {kg[0]}x{kg[1]}x{kg[2]:<3d} {ns:2d} {n_parent:3d} {mu:4d} {nu:4d} {ops:3d} {int(cm):2d} | "
              f"{tx:8.3f} {tk:8.3f} {tx / tk:6.2f}", flush=True)
        del D_l, D_r, tables
    print("bench mode10 plane    lead                 occ  n_col |  XLA ms   kdft ms   (x)", flush=True)
    for nb, nc, frac, lead, slab in plane_shapes:
        pfc, n_col, Fc = _plane_case(rng, nb, nc, frac, lead, 0, dev)
        f_k = F_._kdft_plane_fft_fn(F_.KDFT_PLANE_FFT_GATHER_TARGET, pfc, n_col, nb, nc)
        with F_.xla_reference():
            f_x = F_.make_plane_fft_gather(mesh, pfc, n_col, (nb, nc))
        if slab:
            S = lead[1]
            gk = jax.jit(lambda F: f_k(F, 0, S))
            gx = jax.jit(lambda F: f_x(F, 0, S))
        else:
            gk, gx = jax.jit(f_k), jax.jit(f_x)
        tx, tk = _timed(gx, Fc, reps=reps), _timed(gk, Fc, reps=reps)
        print(f"bench mode10 {nb:3d}x{nc:<3d} {str(lead):22s} {frac:4.2f} {n_col:5d} | {tx:8.3f} {tk:8.3f} "
              f"{tx / tk:6.2f}", flush=True)
        del Fc


def _parse(spec, n_int):
    out = []
    for s in spec.split(","):
        if not s:
            continue
        g, *rest = s.split(":")
        out.append((tuple(int(v) for v in g.split("x")), *(int(v) for v in rest[:n_int])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    # k-grid:ns:n_parent:mu:nu:ops:centroid_major (Fe V2 P=4: 4^3, ns 2, 64 parents, 156 x 156 per rank)
    ap.add_argument("--parent", default="4x4x4:2:64:156:156:48:1,4x4x4:2:64:312:312:48:1,4x4x4:2:8:78:78:48:1,"
                                        "2x2x2:2:4:40:40:8:1,1x1x1:2:1:64:64:1:1")
    # nb x nc : occ% : lead dims (x-separated) : slab (Fe: 25^2, A=64, S=25, R=(2, 78, 2), one slab call)
    ap.add_argument("--plane", default="25x25:70:64x25x2x78x2:1,25x25:70:64x25x2x8x2:1,24x30:60:16x8x64:0,"
                                       "54x54:60:8x8x2x16x2:1,45x45:60:8x8x2x16x2:1,16x16:80:256x64:0")
    args = ap.parse_args()
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    from ffi.gate import mesh_ffi_platform
    mesh = stack.mesh
    if mesh_ffi_platform(mesh) != "ROCM":
        print(f"SKIP: needs a ROCm mesh, got {mesh_ffi_platform(mesh)}", flush=True)
        runtime.finalize_process(0)
        return 0
    if args.bench:
        planes = []
        for s in args.plane.split(","):
            g, occ, lead, slab = s.split(":")
            nb, nc = (int(v) for v in g.split("x"))
            planes.append((nb, nc, int(occ) / 100.0, tuple(int(v) for v in lead.split("x")), bool(int(slab))))
        bench(mesh, [(*p[:6], bool(p[6])) for p in _parse(args.parent, 6)], planes, args.reps)
        runtime.finalize_process(0)
        return 0
    out = gate_cases(mesh)
    bad = {k: v for k, v in out.items() if not v <= TOL}
    if jax.process_index() == 0:
        for k, v in out.items():
            print(f"{'ok  ' if v <= TOL else 'FAIL'} {k:64s} rel {v:.3e}", flush=True)
        print(f"KDFT_PARENT_PLANE_GATE {'PASS' if not bad else 'FAIL'} (modes 1 and 10 vs xla_reference, {len(out)} "
              f"checks, tol {TOL:g}, {jax.process_count()} process(es))", flush=True)
    runtime.finalize_process(0 if not bad else 1)
    return 0 if not bad else 1


def test_kdft_parent_gate():
    import pytest
    pytest.skip("needs a ROCm device: python -m tests.test_kdft_parent_gate")


if __name__ == "__main__":
    sys.exit(main())
