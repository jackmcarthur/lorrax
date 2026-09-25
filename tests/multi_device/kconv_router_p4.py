"""P=4 gate: the k-convolution router's CUDA leg (nvidia-mathdx) at the ζ-fit call sites.

Four processes, one GPU each, 2x2 mesh, rank-varying operands; the router
must resolve to ``mathdx`` on this mesh (asserted, TASTE 30).

1. ``isdf.core.c_q_downfold`` (pair mode, ns=2, k-grid 3x2x2) against the
   dense pair sum of tests/test_isdf_zq_parent_parity.py.
   Red twin: an R-side operand rolled by one k.
2. ``isdf.core.c_q_from_psi_sm`` (parent mode, centroid-major operands,
   glide ns=2 plan with spin mixing and an antiunitary row) against the
   router's cpu-leg composition (typed parent load + spin contraction), its
   transforms evaluated with jnp.fft on the GPU. Red twin: the ψ operand
   rolled by one parent k.
3. ``test_isdf_parent_conv.gpu_main``: random parents, ns = 1/2/4, both
   layouts, against literal direct k sums, with its own rolled-map red twin.
3b. ``make_fused_conv_kplane`` (mode 6, the route-G plane door) at the
   production operand layout D (nk, g, ns, 2c, ns, p) with its Bloch phase:
   within 2 ulp of max|ref| of the chain it replaced (XLA phase + moveaxis + L/R
   split, then the parent door on the identity plan; bitwise today, reported),
   and 1e-12 of the dense sum.
   Red twin: the phase rolled by one k.
3c. ``make_kconv_klead_unfold`` (mode 7, the Σ door from the raw-parent
   Green): within 2 ulp of max|ref| of the chain it replaced (the typed
   unfold, the spin-rotate FFI, sigma_conv_operand, then mode 2; the load's
   products are fused, so round-off equal, reported) on the glide plans (ns 2, 4; spin mixing, an antiunitary row),
   A-cubic (48 operations, ns 1) and C3 with a general complex U and q = n/3
   (ns 2, 4; plus nk = 196 at ns 4, the per-bank load).  Red twin: the right
   source table rolled by one slot.
3d. ``make_kconv_lorentz_unfold`` (mode 8, the four-current Σ door): within
   2 ulp of max|ref| of the Lorentz chain it replaced (the typed unfold,
   mode-3 transforms of G and V, the XLA scan over γ̃ blocks, the forward
   transform; round-off equal through mode 7's fused load, reported) for the CC, CT and TT classes on the
   glide plans (ns 2, 4), a rectangular (charge x current) glide class and C3
   with a general complex U (ns 4).  Red twin: the right source table rolled.
3e. ``make_kfft_klead_unfold`` (mode 9, an interaction's R-space operand read
   from its q wedge): within 2 ulp of max|ref| of the chain it replaces
   (``unfold_isdf_operator``, then the mode-3 prep; round-off equal through
   mode 7's fused load, reported) on both antiunitary rules for the glide, A-cubic and C3 plans,
   and a 3x3 Lorentz block against unfold-then-rotate in XLA.  Red twin: the
   right source table rolled.
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
    def jnp_kfft(x_flat, kgrid_, kind):
        # The plan composition's transforms as jnp.fft (test reference; the
        # CUDA leg has no plan handler since the flat-k transform moved to mathdx).
        kg = tuple(int(v) for v in kgrid_)
        f = jnp.fft.ifftn if kind == "ifftn" else jnp.fft.fftn
        norm = "forward" if kind == "ifftn" else "backward"
        return f(x_flat.reshape(kg + tuple(x_flat.shape[1:])), axes=(0, 1, 2),
                 norm=norm).reshape(x_flat.shape)
    plan_kfft = F._plan_kfft
    F.make_fused_conv_kparent, F._plan_kfft = plan_route, jnp_kfft
    try:
        C_plan = call(nmu)
    finally:
        F.make_fused_conv_kparent, F._plan_kfft = router, plan_kfft
        core._isdf_pipeline_cache.clear()
    return dict(case="c_q_from_psi_sm", kgrid=list(kgrid), n_parent=n_par, mu=mu,
                mathdx_vs_plan=_rel(C_mathdx, C_plan), red_rolled_k=_rel(C_red, C_plan))


def _xla_cmul_form(rng):
    """Which FMA spelling XLA:GPU uses for an HLO complex multiply (diagnostic).

    Mode 6 forms F·D itself and must round like the XLA multiply it replaced;
    the candidate spellings are compared exactly (Fraction arithmetic).
    """
    from fractions import Fraction as Q
    a = _crand(rng, 4096)
    b = _crand(rng, 4096)
    got = np.asarray(jax.device_get(jax.jit(lambda x, y: x * y)(jnp.asarray(a), jnp.asarray(b))))
    fma = lambda x, y, z: float(Q(x) * Q(y) + Q(z))
    forms = {
        "fma(ac,-bd)/fma(ad,bc)": lambda x, y: (fma(x.real, y.real, -(x.imag * y.imag)),
                                                 fma(x.real, y.imag, x.imag * y.real)),
        "fma(-bd,ac)/fma(bc,ad)": lambda x, y: (fma(-x.imag, y.imag, x.real * y.real),
                                                 fma(x.imag, y.real, x.real * y.imag)),
        "no-fma": lambda x, y: (x.real * y.real - x.imag * y.imag,
                                x.real * y.imag + x.imag * y.real),
    }
    hits = {}
    for name, f in forms.items():
        ref = np.asarray([complex(*f(x, y)) for x, y in zip(a, b)])
        hits[name] = int(np.sum((ref.real == got.real) & (ref.imag == got.imag)))
    return hits


def plane_case(mesh, rng):
    """Mode 6 vs the old route-G chain (bitwise) and the dense k sum; g sharded over ranks."""
    from ffi import fft as F
    from functools import partial
    from common.shard_map import shard_map
    from test_kconv_plane import _identity_tables, _literal
    kg, ns, g, c, p = (4, 2, 1), 2, 8, 3, 50
    nk = int(np.prod(kg))
    D = _crand(rng, nk, g, ns, 2 * c, ns, p)
    kfrac = np.stack(np.unravel_index(np.arange(nk), kg), 1) / np.asarray(kg, float)
    xg = rng.uniform(size=(g, p, 3))
    Fb = np.exp(-2j * np.pi * np.einsum("kd,gpd->kgp", kfrac, xg)) / np.sqrt(80.0)
    sd = NamedSharding(mesh, P(None, XY, None, None, None, None))
    sf = NamedSharding(mesh, P(None, XY, None))
    su = P(None, None, XY)
    perm, phase = np.arange(ns), np.ones(ns)
    door = F.make_fused_conv_kplane(mesh, kg, ns, perm_l=perm, phase_l=phase,
                                    perm_r=perm, phase_r=phase)
    parent = F.make_fused_conv_kparent(mesh, kg, ns, None, perm_l=perm, phase_l=phase,
                                       perm_r=perm, phase_r=phase)
    run_new = jax.jit(shard_map(door, mesh=mesh, in_specs=(sd.spec, sf.spec),
                                out_specs=su, check_vma=False))

    @partial(shard_map, mesh=mesh, in_specs=(sd.spec, sf.spec), out_specs=su, check_vma=False)
    def run_old(d, f):
        gl = d.shape[1]
        x = d * f[:, :, None, None, None, :]
        Dk = jnp.moveaxis(x, 1, 4).reshape(nk, ns, 2 * c, ns, gl * p)
        return parent(Dk[:, :, :c], Dk[:, :, c:], _identity_tables(nk, ns, c, gl * p, kg))
    Dd, Fd = _put(D, sd), _put(Fb, sf)
    U = _host(run_new(Dd, Fd))
    U_old = _host(jax.jit(run_old)(Dd, Fd))
    U_red = _host(run_new(Dd, _put(np.roll(Fb, 1, axis=0), sf)))
    ref = _literal(D, Fb, ns, c, perm, phase, kg)
    return dict(case="kconv_plane", kgrid=list(kg), ns=ns, g=g, c=c, p=p,
                bitwise_vs_old_chain=int(np.array_equal(U, U_old)),
                max_abs_vs_old_chain=float(np.max(np.abs(U - U_old))),
                ulp_vs_old_chain=float(np.max(np.abs(U - U_old)) / (np.finfo(float).eps * np.max(np.abs(U_old)))),
                ref_mathdx_vs_dense=_rel(U, ref), red_rolled_phase=_rel(U_red, ref),
                xla_cmul_form=_xla_cmul_form(rng))


def unfold_cases(mesh, rng):
    """Mode 7 on the glide plans and A-cubic vs the old Σ chain (test_kconv_klead_unfold.unfold_case)."""
    import zeta_mubatch_fixtures as fixtures
    from test_kconv_klead_unfold import unfold_case
    recs = []
    from test_kconv_klead_unfold import c3_fixture
    fxs = ([fixtures._glide_fixture(mesh, rng, ns) for ns in (2, 4)] + [fixtures._acubic_fixture(mesh, rng)]
           + [c3_fixture(mesh, ns) for ns in (2, 4)]
           # nk = 196 at ns = 4: fewer rows than one 16-row spin group fit the
           # 48 KiB budget, so mode 7 takes its per-bank load (audit M2).
           + [c3_fixture(mesh, 4, kgrid=(14, 14, 1))])
    for fx in fxs:
        r = unfold_case(mesh, fx)
        recs.append(dict(case=f"kconv_klead_unfold_ns{r['ns']}_nk{r['nk']}", nk=r["nk"], n_parent=r["n_parent"],
                         antiunitary=r["antiunitary"], bitwise_vs_old_chain=int(r["door_bitwise"]),
                         bitwise_tables_vs_unfold=int(r["tables_bitwise"]),
                         bitwise_parent_rows=int(r["rows_bitwise"]),
                         max_abs_vs_old_chain=r["max_abs"], rel_vs_old_chain=r["rel"],
                         ulp_vs_old_chain=r["rel"] / np.finfo(float).eps,
                         red_rolled_rsrc=r["red_rel"]))
    return recs


def lorentz_cases(mesh, rng):
    """Mode 8 vs the old Lorentz chain (test_kconv_lorentz_unfold.lorentz_case)."""
    from test_kconv_lorentz_unfold import cases, lorentz_case
    recs = []
    for fx, cls, rplan, pref in cases(mesh, rng):
        r = lorentz_case(mesh, fx, cls, right_plan=rplan, prefactor=pref)
        recs.append(dict(case=f"kconv_lorentz_{cls}_ns{r['ns']}_nk{r['nk']}"
                         + ("_rect" if r["rectangular"] else ""),
                         antiunitary=r["antiunitary"], bitwise_vs_old_chain=int(r["door_bitwise"]),
                         bitwise_parent_rows=int(r["rows_bitwise"]),
                         max_abs_vs_old_chain=r["max_abs"], rel_vs_old_chain=r["rel"],
                         ulp_vs_old_chain=r["rel"] / np.finfo(float).eps,
                         red_rolled_rsrc=r["red_rel"]))
    return recs


def wedge_cases(mesh, rng):
    """Mode 9 vs unfold_isdf_operator + the mode-3 prep (test_kfft_klead_unfold)."""
    import zeta_mubatch_fixtures as fixtures
    from test_kconv_klead_unfold import c3_fixture
    from test_kfft_klead_unfold import lorentz_wedge_case, wedge_case
    recs = []
    fxs = [fixtures._glide_fixture(mesh, rng, 2), fixtures._acubic_fixture(mesh, rng),
           c3_fixture(mesh, 2)]
    for fx in fxs:
        for rule in ("conj", "pair_transpose"):
            r = wedge_case(mesh, fx, rule)
            recs.append(dict(case=f"kfft_klead_unfold_{rule}_nk{r['nk']}", nk=r["nk"],
                             antiunitary=r["antiunitary"], bitwise_vs_old_chain=int(r["door_bitwise"]),
                             bitwise_parent_rows=int(r["device_tables_bitwise"]),
                             max_abs_vs_old_chain=r["max_abs"], rel_vs_old_chain=r["rel"],
                             ulp_vs_old_chain=r["rel"] / np.finfo(float).eps,
                             red_rolled_rsrc=r["red_rel"]))
    for fx in (fxs[0], fxs[2]):
        r = lorentz_wedge_case(mesh, fx)
        recs.append(dict(case=f"kfft_klead_unfold_lorentz3_nk{r['nk']}", nk=r["nk"],
                         ulp_vs_old_chain=r["rel"] / np.finfo(float).eps, red_rolled_rsrc=r["red_rel"]))
    return recs


def block_cases(mesh, rng):
    """Mode 7's output spin blocks and conj-on-load partner vs the whole door (bitwise)."""
    import zeta_mubatch_fixtures as fixtures
    from test_kconv_klead_unfold import block_case, c3_fixture
    recs = []
    for fx in [fixtures._glide_fixture(mesh, rng, ns) for ns in (2, 4)] + [c3_fixture(mesh, 4)]:
        r = block_case(mesh, fx)
        recs.append(dict(case=f"kconv_unfold_blocks_ns{r['ns']}", antiunitary=r["antiunitary"],
                         bitwise_parent_rows=int(r["conj_bitwise"] and r["blocks_max_abs"] == 0.0),
                         blocks_max_abs=r["blocks_max_abs"]))
    return recs


def chi_cases(mesh, rng):
    """Mode 11 vs the incumbent chi0 chain (test_kconv_chi_unfold.chi_case), both k-box arms:
    the single pass on the unit-test plans, the split pass on grids whose pair does not fit."""
    from test_kconv_chi_unfold import chi_case, chi_cases as cases
    from test_kconv_klead_unfold import c3_fixture
    recs = []
    split = [(c3_fixture(mesh, 4, kgrid=(8, 8, 8)), 1, True, True),
             (c3_fixture(mesh, 2, kgrid=(12, 12, 9)), 2, False, False)]
    for fx, n_out, complete, conj_src in cases(mesh, rng) + split:
        r = chi_case(mesh, fx, n_out=n_out, complete=complete, conj_src=conj_src)
        recs.append(dict(case=f"kconv_chi_ns{r['ns']}_nk{r['nk']}_o{n_out}"
                         + ("_real" if complete else "_complex") + ("_conjsrc" if conj_src else "_tile"),
                         antiunitary=r["antiunitary"], ulp_chi_vs_old_chain=r["rel"] / np.finfo(float).eps,
                         red_rolled_rsrc=r["red_rel"]))
    return recs


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
    recs = ([downfold_case(mesh, rng), face_parent_case(mesh, rng), plane_case(mesh, rng)]
            + unfold_cases(mesh, rng) + lorentz_cases(mesh, rng) + wedge_cases(mesh, rng)
            + chi_cases(mesh, rng) + block_cases(mesh, rng)
            + stored_cases(mesh, rng))
    bad = []
    for r in recs:
        for k, v in r.items():
            lim = (TOL if k == "mathdx_vs_plan" else TOL_REF if k.startswith("ref_")
                   else 1.0e-5 if k == "c64_vs_numpy" else None)
            if lim is not None and not v <= lim:
                bad.append(f"{r['case']}.{k}={v:.2e} > {lim}")
            if k.startswith("red_") and not v > RED:
                bad.append(f"{r['case']}.{k}={v:.2e} <= {RED} (red twin did not fire)")
            # Mode 6 spells every product as the chain it replaces (bitwise today,
            # reported); modes 7/8/9 form the unfold load with fused products
            # (owner 2026-09-25: round-off equal is fine; 1.8 ulp at most on these
            # cases, U2c).  The gate is 2 ulp of the largest value (audit L2).
            if k == "ulp_vs_old_chain" and not v <= 2.0:
                bad.append(f"{r['case']}.{k}={v:.2f} > 2 ulp of max|ref|")
            # Mode 11 transforms ifftn(G) where the chain it replaces transforms fftn(conj G):
            # the same number, not the same rounding (test_kconv_chi_unfold).
            if k == "ulp_chi_vs_old_chain" and not v <= 8.0:
                bad.append(f"{r['case']}.{k}={v:.2f} > 8 ulp of max|chi|")
            # The parent-row store is the same kernel with fewer stores: bitwise.
            if k == "bitwise_parent_rows" and v != 1:
                bad.append(f"{r['case']}.{k}=0 (parent-row store differs from those rows)")
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
