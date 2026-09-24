"""P=4 gate: the k-convolution router's CUDA leg (nvidia-mathdx) at the ζ-fit call sites.

Four processes, one GPU each, 2x2 mesh, rank-varying operands; the router
must resolve to ``mathdx`` on this mesh (asserted, TASTE 30).

1. ``isdf.core.c_q_downfold`` (pair mode, ns=2, k-grid 3x2x2) against the
   dense pair sum of tests/test_isdf_zq_parent_parity.py.
   Red twin: an R-side operand rolled by one k.
2. ``isdf.core.c_q_from_psi_sm`` (parent mode, centroid-major operands,
   glide ns=2 plan with spin mixing and an antiunitary row) against the
   router's cpu leg (the flat-k plan route, evaluated here on the GPU's plan
   handler). Red twin: the ψ operand rolled by one parent k.
3. ``test_isdf_parent_conv.gpu_main``: random parents, ns = 1/2/4, both
   layouts, against literal direct k sums, with its own rolled-map red twin.
4. The stored-kernel doors (modes 2-5) against NumPy ``np.fft`` on sharded
   operands, including an odd grid and 8x8x8: ``make_kconv_klead`` (Σ/COHSEX,
   prep + apply), ``make_kconv_kminor`` (BSE rung, both store layouts),
   ``make_kfft_kminor`` and ``make_kfft_klead`` (both directions, three
   norms), and the complex64 image of the k-minor conv (fp32 tolerance).
   Red twins: the stored kernel rolled by one k, the transform input rolled
   by one k.

Parity 1e-13 (mathdx vs plan route), 1e-12 vs dense sums; each red twin
must miss by more than 1e-3.
Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/kconv_router_p4.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

TAG = "[kconv-router-p4]"
TOL, TOL_REF, RED = 1.0e-13, 1.0e-12, 1.0e-3
XY = ("x", "y")


def _crand(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _put(x, sharding):
    x = np.asarray(x)
    return jax.make_array_from_callback(x.shape, sharding, lambda i: x[i])


def _host(x):
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def downfold_case(mesh, rng):
    """c_q_downfold through the router vs the dense pair sum."""
    from isdf.core import c_q_downfold
    from test_isdf_zq_parent_parity import _dense_pair_rhs
    kgrid, ns, n_rmu, nb = (3, 2, 2), 2, 8, 6
    nk = int(np.prod(kgrid))
    psi = _crand(rng, nk, nb, ns, n_rmu)
    bra = psi.conj().transpose(0, 3, 1, 2)                       # (k, μ, n, s)
    row = NamedSharding(mesh, P(None, "x", None, None))
    col = NamedSharding(mesh, P(None, None, None, "y"))
    (l0, l1), (r0, r1) = (0, 4), (2, 6)
    args = (_put(bra[:, :, l0:l1], row), _put(psi[:, l0:l1], col),
            _put(bra[:, :, r0:r1], row), _put(psi[:, r0:r1], col))
    C = _host(c_q_downfold(*args, kgrid=kgrid, mesh_xy=mesh))
    Cr = _host(c_q_downfold(*args[:3], _put(np.roll(psi[:, r0:r1], 1, axis=0), col),
                            kgrid=kgrid, mesh_xy=mesh))
    idx = np.arange(nb)
    ref = _dense_pair_rhs(psi, np.arange(n_rmu), kgrid,
                          ((idx >= l0) & (idx < l1)).astype(float),
                          ((idx >= r0) & (idx < r1)).astype(float), 0, 0)
    return dict(case="c_q_downfold", kgrid=list(kgrid), ns=ns,
                ref_mathdx_vs_dense=_rel(C, ref), red_rolled_k=_rel(Cr, ref))


def _glide_plan(mesh):
    """The order-two glide group with spin mixing and an antiunitary row (ns = 2)."""
    from types import SimpleNamespace
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from symmetry_maps import centroid_source_map_and_wrap, spinor_rotation_for_sym_row
    fft_grid, kgrid = (4, 4, 4), (2, 2, 1)
    swap = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    ops = np.stack([np.eye(3, dtype=np.int64), swap])
    tnp = np.asarray([[0.0, 0.0, 0.0], [np.pi, np.pi, 0.0]])
    kfrac = np.asarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]) / np.asarray(kgrid, float)
    theta = 0.7
    U1 = np.asarray([[np.cos(theta), -1j * np.sin(theta)], [-1j * np.sin(theta), np.cos(theta)]])
    U_spatial = np.stack([np.eye(2, dtype=np.complex128), U1])
    sym = SimpleNamespace(
        sym_matrices=ops, translations=tnp, irr_idx_k=np.asarray([0, 1, 1, 2], np.int32),
        sym_idx_k=np.asarray([0, 0, 1, 2], np.int32), unfolded_kpts=kfrac,
        kirr_fullids=np.asarray([0, 1, 3]),
        spinor_action=lambda rows, *, nspinor: spinor_rotation_for_sym_row(
            U_spatial, np.asarray(rows), 2, nspinor=nspinor, R_cart=ops))
    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in fft_grid), indexing="ij")
    grid = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], 1).astype(np.int32)
    perm_g, _ = centroid_source_map_and_wrap(grid, ops, tnp, fft_grid, extend_trs=True)
    cent = []
    for seed in (0, 5, 22, 27, 41, 50, 60, 63, 9, 14):
        orbit = sorted({int(perm_g[s, seed]) for s in range(4)})
        if len(cent) + len(orbit) <= 8 and not any(c in cent for c in orbit):
            cent.extend(orbit)
        if len(cent) == 8:
            break
    plan = build_centroid_k_unfold_plan(sym, grid[np.asarray(sorted(cent))], fft_grid, mesh,
                                        nspinor=2, parent_k_frac=kfrac[[0, 1, 3]])
    return plan, kgrid


def face_parent_case(mesh, rng):
    """c_q_from_psi_sm through the router vs the router's plan route."""
    from isdf import core
    from ffi import fft as F
    plan, kgrid = _glide_plan(mesh)
    n_par, ns, mu, nb = int(plan.n_parent), 2, int(plan.n_centroid_packed), 6
    mun = _crand(rng, n_par, ns, mu, nb)                 # ψ_{n k̄ s}(r_μ), packed μ
    nmu = np.ascontiguousarray(mun.transpose(0, 3, 1, 2))  # the same ψ, (p, n, s, μ)

    def gemm(a, b):
        return jnp.einsum("qmk,qkn->qmn", a, b, optimize=True)
    gemm.in_sharding_a = NamedSharding(mesh, P(None, "x", "y"))
    gemm.in_sharding_b = gemm.in_sharding_a
    s_mun = NamedSharding(mesh, P(None, None, "x", "y"))
    s_nmu = NamedSharding(mesh, P(None, "x", None, "y"))
    w_l = jnp.asarray((np.arange(nb) < 4).astype(float))
    w_r = jnp.asarray((np.arange(nb) >= 2).astype(float))

    def call(nmu_host):
        core._isdf_pipeline_cache.clear()
        return _host(core.c_q_from_psi_sm(
            _put(mun, s_mun), _put(nmu_host, s_nmu), w_l, w_r, k_unfold_plan=plan,
            kgrid=kgrid, mesh_xy=mesh, gemm=gemm))
    assert F.kconv_backend(mesh) == "mathdx"
    C_mathdx = call(nmu)
    C_red = call(np.roll(nmu, 1, axis=0))
    router = F.make_fused_conv_kparent

    def plan_route(mesh_, kgrid_, ns_, trailing, *, perm_l, phase_l, perm_r, phase_r,
                   centroid_major=False):
        del mesh_, trailing, centroid_major
        return F._plan_kparent(kgrid_, ns_, perm_l, phase_l, perm_r, phase_r,
                               F.conv_kpair_scale("forward", int(np.prod(kgrid_))))
    F.make_fused_conv_kparent = plan_route
    try:
        C_plan = call(nmu)
    finally:
        F.make_fused_conv_kparent = router
        core._isdf_pipeline_cache.clear()
    return dict(case="c_q_from_psi_sm", kgrid=list(kgrid), n_parent=n_par, mu=mu,
                mathdx_vs_plan=_rel(C_mathdx, C_plan), red_rolled_k=_rel(C_red, C_plan))


def _np3(x, kg, axis0, kind, norm):
    """np.fft over three consecutive k axes starting at axis0 of the reshaped array."""
    f = np.fft.ifftn if kind == "ifftn" else np.fft.fftn
    return f(x, axes=(axis0, axis0 + 1, axis0 + 2), norm=norm)


def stored_cases(mesh, rng):
    """The mode 2-5 doors vs NumPy on sharded, rank-varying operands."""
    from ffi import fft as F
    recs = []
    for kg in ((3, 2, 4), (7, 3, 5), (8, 8, 8)):
        nk = int(np.prod(kg))
        a = b = 2
        mx, my = 8, 4
        T = _crand(rng, nk, a, mx, b, my)
        Wq = _crand(rng, nk, mx, my)
        mult = -1.0 / np.sqrt(nk)
        t_spec = P(None, None, None, None, "x", None, "y")
        w_spec = P(None, None, None, "x", "y")
        conv = F.make_kconv_klead(mesh, kg, t_spec, w_spec, norm="ortho", mult=mult)
        sh_t = NamedSharding(mesh, P(None, None, "x", None, "y"))
        sh_w = NamedSharding(mesh, P(None, "x", "y"))
        U = _host(conv.apply(_put(T, sh_t), conv.prep(_put(Wq, sh_w))))
        Ured = _host(conv.apply(_put(T, sh_t), conv.prep(_put(np.roll(Wq, 1, 0), sh_w))))
        T3 = T.reshape(kg + T.shape[1:])
        W3 = Wq.reshape(kg + Wq.shape[1:])
        ref = mult * _np3(_np3(T3, kg, 0, "ifftn", "ortho")
                          * _np3(W3, kg, 0, "ifftn", "ortho")[:, :, :, None, :, None, :],
                          kg, 0, "fftn", "ortho")
        ref = ref.reshape(U.shape)
        recs.append(dict(case="kconv_klead", kgrid=list(kg), ref_mathdx_vs_numpy=_rel(U, ref),
                         red_rolled_W=_rel(Ured, ref)))

        # BSE rung: X (b, mu, nu, t, s, nk) k-minor, K_R (mu, nu, nk)
        nb, mu, nu, ns = 1, 8, 4, 2
        X = _crand(rng, nb, mu, nu, ns, ns, nk)
        K = _crand(rng, mu, nu, nk)
        x_spec, k_spec = P(None, "x", "y", None, None, None), P("x", "y", None)
        Kr = np.fft.ifftn(K.reshape(mu, nu, *kg), axes=(2, 3, 4), norm="ortho").reshape(mu, nu, nk)
        X6 = X.reshape(nb, mu, nu, ns, ns, *kg)
        ref0 = np.fft.fftn(np.fft.ifftn(X6, axes=(5, 6, 7), norm="ortho")
                           * Kr.reshape(mu, nu, *kg)[None, :, :, None, None],
                           axes=(5, 6, 7), norm="ortho").reshape(X.shape)
        for lay in (0, 1):
            cv = F.make_kconv_kminor(mesh, kg, x_spec, k_spec, norm="ortho", out_layout=lay)
            got = _host(cv(_put(X, NamedSharding(mesh, x_spec)), _put(Kr, NamedSharding(mesh, k_spec))))
            red = _host(cv(_put(X, NamedSharding(mesh, x_spec)),
                           _put(np.roll(Kr, 1, 2), NamedSharding(mesh, k_spec))))
            want = ref0 if lay == 0 else ref0.transpose(0, 5, 3, 1, 4, 2)
            recs.append(dict(case=f"kconv_kminor_layout{lay}", kgrid=list(kg),
                             ref_mathdx_vs_numpy=_rel(got, want), red_rolled_K=_rel(red, want)))
        # complex64 (the fp32-GMRES BSE arm): the single-precision image, fp32 tolerance
        cv = F.make_kconv_kminor(mesh, kg, x_spec, k_spec, norm="ortho", out_layout=1)
        got = _host(cv(_put(X.astype(np.complex64), NamedSharding(mesh, x_spec)),
                       _put(Kr.astype(np.complex64), NamedSharding(mesh, k_spec))))
        recs.append(dict(case="kconv_kminor_c64", kgrid=list(kg), c64_vs_numpy=_rel(got, ref0.transpose(0, 5, 3, 1, 4, 2)),
                         dtype=str(got.dtype)))

        # transforms: k-minor (mu, nu, kx, ky, kz) and k-leading (nk, mx, my)
        Y = _crand(rng, mu, nu, *kg)
        y_spec = P("x", "y", None, None, None)
        L = _crand(rng, nk, mx, my)
        l_spec3 = P(None, None, None, "x", "y")
        for kind, norm in (("ifftn", "ortho"), ("fftn", None), ("ifftn", "forward")):
            f = F.make_kfft_kminor(mesh, kg, y_spec, kind=kind, norm=norm)
            got = _host(f(_put(Y, NamedSharding(mesh, y_spec))))
            red = _host(f(_put(np.roll(Y, 1, 2), NamedSharding(mesh, y_spec))))
            want = _np3(Y, kg, 2, kind, norm)
            recs.append(dict(case=f"kfft_kminor_{kind}_{norm}", kgrid=list(kg),
                             ref_mathdx_vs_numpy=_rel(got, want), red_rolled_k=_rel(red, want)))
            g = F.make_kfft_klead(mesh, kg, l_spec3, kind=kind, norm=norm)
            got = _host(g(_put(L, NamedSharding(mesh, P(None, "x", "y")))))
            red = _host(g(_put(np.roll(L, 1, 0), NamedSharding(mesh, P(None, "x", "y")))))
            want = _np3(L.reshape(*kg, mx, my), kg, 0, kind, norm).reshape(L.shape)
            recs.append(dict(case=f"kfft_klead_{kind}_{norm}", kgrid=list(kg),
                             ref_mathdx_vs_numpy=_rel(got, want), red_rolled_k=_rel(red, want)))
    return recs


def main() -> int:
    import json
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), XY)
    rng = np.random.default_rng(20260924)
    recs = [downfold_case(mesh, rng), face_parent_case(mesh, rng)] + stored_cases(mesh, rng)
    bad = []
    for r in recs:
        for k, v in r.items():
            lim = (TOL if k == "mathdx_vs_plan" else TOL_REF if k.startswith("ref_")
                   else 1.0e-5 if k == "c64_vs_numpy" else None)
            if lim is not None and not v <= lim:
                bad.append(f"{r['case']}.{k}={v:.2e} > {lim}")
            if k.startswith("red_") and not v > RED:
                bad.append(f"{r['case']}.{k}={v:.2e} <= {RED} (red twin did not fire)")
        if jax.process_index() == 0:
            print(TAG, json.dumps(r), flush=True)
    import test_isdf_parent_conv as tpc
    try:
        tpc.gpu_main()
    except AssertionError as exc:
        bad.append(f"test_isdf_parent_conv.gpu_main: {exc}")
    if jax.process_index() == 0:
        print(TAG, "FAIL: " + "; ".join(bad) if bad else "PASS", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    rc = main()
    finalize_process()
    sys.exit(rc)
