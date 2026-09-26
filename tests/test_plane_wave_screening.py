"""``gw.plane_wave_screening`` against dense references, on CPU meshes (P1, P4).

* χ's factor: the pair convolution's raw sum times −s/(Ω·N_r²) equals a BerkeleyGW
  M-matrix band sum (n_s = 1 and 2); the red twin is K1's Ω/N_r² note.
* v_q(G): the ISDF path's owner (``compute_v_q_per_G`` on the ζ-sphere layout of
  ``compute_per_q_bare_coulomb_components``) and the closed forms, bulk and slab, on
  the same G set; red twins: the 1/Ω table units, the wrong sys_dim.
* the Dyson solve against explicit per-q inverses; red twin: pref 2.
* the Γ cell: the body ignores χ's G=0 row and column; the fold's S_eff equals the
  head of the explicit bordered inverse along random directions; red twins: a body
  with v(0) ≠ 0, the transposed wings.  The omitted rank-3 body term is printed.
* the MPA hookup: tiles fit on P4 equal one dense fit of every element; red twin: a
  permuted sample grid.
* the antiunitary rule on the glide group (an antiunitary row, spin mixing, TR broken):
  at fixed τ the child is conj of the parent's image; at fixed z it is conj of the
  image at z̄; red twins: no conj at τ, the parent at z instead of z̄.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import mixed_basis_pair_conv_cases as cases
import plane_wave_screening_cases as pw
import zeta_mubatch_fixtures as fixtures

TOL = 1e-11


def _mesh(n):
    if n == 1:
        return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    from lxkit.testing import require_devices
    require_devices(4)
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _geometry(lat=pw.LAT):
    from ffi import _services
    _services.ensure_on_path()
    import vcoul
    return vcoul.CoulombGeometry(bvec=pw.bvec_rows(lat), cell_volume=abs(np.linalg.det(lat)))


def _put_stack(x, mesh, spec):
    return fixtures._put(np.asarray(x), NamedSharding(mesh, spec))


def _gather(x):
    from jax.experimental import multihost_utils
    return np.asarray(x) if x.is_fully_addressable else \
        np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _pad_to(a, M, axes):
    pad = [(0, 0)] * a.ndim
    for ax in axes:
        pad[ax] = (0, M - a.shape[ax])
    return np.pad(a, pad)


# ---------------------------------------------------------------------------
# χ's factor
# ---------------------------------------------------------------------------

def _chi_case(ns, *, kgrid=(2, 2, 1), fft_grid=(12, 12, 12), ecut=4.0, nb=6, nv=3, tau=0.7):
    from gw.mixed_basis_pair_convolution import SphereSet
    from gw.plane_wave_screening import response_spheres
    rng = np.random.default_rng(11 + ns)
    kfrac = cases.kgrid_frac(kgrid)
    c, sph, ngk = pw.orthonormal_bands(rng, kfrac, pw.LAT, ecut, nb, ns)
    E = np.sort(np.concatenate([rng.uniform(-1.0, -0.4, (len(kfrac), nv)),
                                rng.uniform(0.3, 1.1, (len(kfrac), nb - nv))], axis=1), axis=1)
    wc = np.where(np.arange(nb) >= nv, 1.0, 0.0)[None, :] * np.exp(-E * tau)
    wv = np.where(np.arange(nb) < nv, 1.0, 0.0)[None, :] * np.exp(E * tau)
    A = np.einsum("knap,kn,knbq->kpaqb", c, wc, np.conj(c))
    C = np.einsum("knap,kn,knbq->kpaqb", c, wv, np.conj(c))
    rs = response_spheres(fft_grid=fft_grid, psi=SphereSet(sph, ngk, kfrac), bvec=pw.bvec_rows(),
                          kgrid=kgrid, q_irr_frac=kfrac[[0, 1, 3]], ecutwfc=ecut,
                          screened_coulomb_cutoff=0.8 * ecut)
    return dict(c=c, E=E, nv=nv, tau=tau, A=A, C=C, sph=sph, ngk=ngk, kfrac=kfrac, kgrid=kgrid,
                fft_grid=fft_grid, rs=rs, ns=ns)


def _pair_sums(mesh, c, out):
    from gw.mixed_basis_pair_convolution import (MixedBasisPairConvolution, PairOperand,
                                                SphereSet, SphereTransport)
    ps = SphereSet(c["sph"], c["ngk"], c["kfrac"])
    op = PairOperand(ps, SphereTransport.identity(ps, c["ns"]))
    conv = MixedBasisPairConvolution(mesh, kgrid=c["kgrid"], fft_grid=c["fft_grid"], left=op,
                                     right=op, out=out, budget_bytes=int(1e10))
    s5 = NamedSharding(mesh, P(None, "x", None, "y", None))
    put = lambda a: fixtures._put(cases.pad_tiles(a, conv.width_carrier[0]), s5)
    return conv, conv(put(c["A"]), put(c["C"]))


@pytest.mark.parametrize("n_mesh", [1, 4])
@pytest.mark.parametrize("ns", [1, 2])
def test_chi_factor_matches_the_band_sum(n_mesh, ns):
    from gw.plane_wave_screening import chi_pair_sum_scale
    c = _chi_case(ns)
    out = c["rs"].irr
    conv, X = _pair_sums(_mesh(n_mesh), c, out)
    X = conv.strip(X)
    omega = abs(np.linalg.det(pw.LAT))
    n_r = int(np.prod(c["fft_grid"]))
    s = 2.0 / ns
    ref = pw.band_sum_chi(c["c"], c["E"], c["nv"], c["sph"], c["ngk"], c["kfrac"], c["kgrid"],
                          out.gvecs, out.ngk, out.frac, tau=c["tau"], cell_volume=omega, s=s)
    got = -chi_pair_sum_scale(cell_volume=omega, n_r=n_r, n_spinor=ns) * X
    e = cases.rel(got, ref)
    print(f"chi factor ns={ns} P{n_mesh}: rel {e:.2e}, max|chi| {np.max(np.abs(ref)):.3e}")
    assert e <= TOL, e
    k1_note = -(omega / n_r ** 2) * (2.0 / ns) * X          # K1's Ω/N_r² note: the red twin
    assert cases.rel(k1_note, ref) > 1e-3


@pytest.mark.parametrize("n_mesh", [1, 4])
def test_minimax_rule_on_pair_sums_matches_the_band_sum(n_mesh):
    """χ(iω) from the imaginary-axis minimax rule (``minimax_screening``) applied to the pair
    sums of both orientations equals the band sum of x/(x² + ω²) to the rule's error."""
    from gw.minimax_screening import solve_laplace_minimax_imag_interval
    from gw.plane_wave_screening import accumulate_chi, chi_pair_sum_scale
    mesh = _mesh(n_mesh)
    c = _chi_case(1)
    out = c["rs"].irr
    E, nv, nb = c["E"], c["nv"], c["E"].shape[1]
    x_min = float(E[:, nv:].min() - E[:, :nv].max())
    x_max = float(E[:, nv:].max() - E[:, :nv].min())
    omega = 0.45
    rule = solve_laplace_minimax_imag_interval(x_min, x_max, omega, target_error=1e-9)
    omega_cell = abs(np.linalg.det(pw.LAT))
    scale = -chi_pair_sum_scale(cell_volume=omega_cell, n_r=int(np.prod(c["fft_grid"])))
    acc = None
    cond = np.arange(nb) >= nv
    for t, a in zip(rule.tau, rule.alpha):
        wc = np.where(cond, np.exp(-E * t), 0.0)
        wv = np.where(~cond, np.exp(E * t), 0.0)
        A = np.einsum("knap,kn,knbq->kpaqb", c["c"], wc, np.conj(c["c"]))
        C = np.einsum("knap,kn,knbq->kpaqb", c["c"], wv, np.conj(c["c"]))
        for L, R in ((A, C), (C, A)):                       # the two orientations
            conv, X = _pair_sums(mesh, dict(c, A=L, C=R), out)
            if acc is None:
                M = int(X.shape[1])
                acc = _put_stack(np.zeros((2, out.n, M, M), np.complex128), mesh,
                                 P(None, None, "x", "y"))
            acc = accumulate_chi(acc, X, [a, 0.0], scale=scale, mesh=mesh)   # sample 1 stays 0
    got = _gather(acc)[0, :, :out.width, :out.width]
    w = omega
    ref = (pw.band_sum_response(c["c"], E, np.flatnonzero(cond), np.flatnonzero(~cond), c["sph"],
                                c["ngk"], c["kfrac"], c["kgrid"], out.gvecs, out.ngk, out.frac,
                                lambda d: d / (d * d + w * w), cell_volume=omega_cell, s=2.0)
           + pw.band_sum_response(c["c"], E, np.flatnonzero(~cond), np.flatnonzero(cond), c["sph"],
                                  c["ngk"], c["kfrac"], c["kgrid"], out.gvecs, out.ngk, out.frac,
                                  lambda d: -d / (d * d + w * w), cell_volume=omega_cell, s=2.0))
    e = cases.rel(got, ref)
    ref0 = (pw.band_sum_response(c["c"], E, np.flatnonzero(cond), np.flatnonzero(~cond), c["sph"],
                                 c["ngk"], c["kfrac"], c["kgrid"], out.gvecs, out.ngk, out.frac,
                                 lambda d: 1.0 / d, cell_volume=omega_cell, s=2.0))
    print(f"minimax chi(i{omega}) P{n_mesh}: {len(rule.tau)} nodes (rule error {rule.max_error:.1e}); "
          f"rel vs band sum {e:.1e}; vs the static target (red twin) {cases.rel(got, ref0):.1e}")
    assert e <= 1e-6 and cases.rel(got, ref0) > 1e-3


# ---------------------------------------------------------------------------
# v_q(G)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sys_dim", [3, 2])
def test_coulomb_matches_the_isdf_owner_and_the_closed_form(sys_dim):
    from common.coulomb_sphere import compute_per_q_bare_coulomb_components
    from gw.compute_vcoul import compute_v_q_per_G
    from gw.plane_wave_screening import sphere_coulomb
    c = _chi_case(1)
    geo = _geometry()
    sph = c["rs"].full
    M = sph.width + 5
    v = sphere_coulomb(sph, geometry=geo, sys_dim=sys_dim, carrier=M)
    assert np.all(v[:, sph.width:] == 0.0) and np.all(v[~np.pad(sph.live(), ((0, 0), (0, 5)))] == 0.0)
    ref = pw.coulomb_closed_form(sph.frac, sph.gvecs, sph.ngk, geo.bvec, sys_dim=sys_dim)
    e_closed = cases.rel(v[:, :sph.width], ref)
    # the ISDF path's call on its own (ζ-sphere) layout, gathered onto this G set
    cut = float(np.max(np.sum(((sph.frac[:, None, :] + sph.gvecs) @ geo.bvec) ** 2, -1)[sph.live()]))
    pkg = compute_per_q_bare_coulomb_components(c["fft_grid"], geo.bvec, sph.frac, cut * (1 + 1e-9))
    gz = np.asarray(pkg["gvec_components_padded"])                 # (n_q, 3, ngkmax)
    vz = compute_v_q_per_G(sph.frac, gz, bvec=geo.bvec, cell_volume=geo.cell_volume,
                           sys_dim=sys_dim) * geo.cell_volume
    iso = np.zeros_like(ref)
    for q in range(sph.n):
        look = {tuple(g): i for i, g in enumerate(gz[q].T[:int(pkg["ngk_per_q"][q])])}
        for s_ in range(int(sph.ngk[q])):
            iso[q, s_] = vz[q, look[tuple(sph.gvecs[q, s_])]]
    e_isdf = cases.rel(v[:, :sph.width], iso)
    print(f"v_q(G) sys_dim={sys_dim}: rel vs closed form {e_closed:.2e}, vs ISDF owner {e_isdf:.2e}")
    assert e_closed <= 1e-13 and e_isdf <= 1e-13
    assert cases.rel(v[:, :sph.width] / geo.cell_volume, ref) > 1e-3        # table units
    other = pw.coulomb_closed_form(sph.frac, sph.gvecs, sph.ngk, geo.bvec, sys_dim=5 - sys_dim)
    assert cases.rel(v[:, :sph.width], other) > 1e-3                        # wrong sys_dim


# ---------------------------------------------------------------------------
# the Dyson solve
# ---------------------------------------------------------------------------

def _screening(mesh, sph, **kw):
    from gw.plane_wave_screening import SphereScreening
    return SphereScreening(mesh, sphere=sph, geometry=_geometry(), sys_dim=3, kgrid=(2, 2, 1), **kw)


def _random_chi(rng, sph, M, scale):
    """A Hermitian negative χ per q on the live slots, garbage-free pads."""
    n, w = sph.n, sph.width
    B = rng.standard_normal((n, w, w)) + 1j * rng.standard_normal((n, w, w))
    chi = -np.einsum("qab,qcb->qac", B, np.conj(B)) * scale
    live = sph.live()
    chi = np.where(live[:, :, None] & live[:, None, :], chi, 0.0)
    return _pad_to(chi, M, (1, 2))


@pytest.mark.parametrize("n_mesh", [1, 4])
def test_dyson_matches_explicit_inverses(n_mesh, linalg="local"):
    """``linalg = local``; the P4 GPU gate also runs ``distributed`` (cuSOLVERMp)."""
    mesh = _mesh(n_mesh)
    c = _chi_case(1)
    scr = _screening(mesh, c["rs"].irr, linalg=linalg)
    rng = np.random.default_rng(3)
    chi = _random_chi(rng, scr.sphere, scr.M, 2e-4)
    ref = pw.dense_dyson(scr.v, chi)
    W = _gather(scr.solve(_put_stack(chi, mesh, P(None, "x", "y"))))
    e = cases.rel(W, ref)
    print(f"Dyson P{n_mesh} {linalg}: rel {e:.2e}, ||v chi|| {np.max(np.abs(scr.v[:, :, None] * chi)):.2f}; "
          + scr.describe(n_z=16, n_p=8))
    assert e <= TOL
    assert np.all(W[:, scr.sphere.width:, :] == 0) and np.all(W[:, :, scr.sphere.width:] == 0)
    red = pw.dense_dyson(scr.v, 2.0 * chi)                                   # pref 2
    assert cases.rel(W, red) > 1e-3
    # the wedge-q chunk count comes from the budget: one when it fits, more when it does not
    stack = 16 * scr.M ** 2 / scr.P
    assert scr.plan_q_chunks(16, 8, budget_bytes=int(1e10)) == 1
    tight = int(scr.fit_q_batch(8, budget_bytes=int(1e10)) * 0 + 41 * stack
                + max(16 * 5 * scr.M ** 2, scr._fit_temp[(8, False, 1e-13, "loewner")]))
    assert scr.plan_q_chunks(16, 8, budget_bytes=tight) == scr.sphere.n
    with pytest.raises(ValueError, match="pw-screening-budget"):
        scr.plan_q_chunks(16, 8, budget_bytes=1)


# ---------------------------------------------------------------------------
# the Γ cell
# ---------------------------------------------------------------------------

def test_gamma_body_fold_and_head():
    from gw.plane_wave_screening import SphereScreening
    mesh = _mesh(4)
    c = _chi_case(1)
    rs = c["rs"]
    scr = _screening(mesh, rs.irr)
    ig, M, w = scr.gamma, scr.M, scr.sphere.width
    assert ig is not None and scr.v[ig, 0] == 0.0
    rng = np.random.default_rng(5)
    chi = _random_chi(rng, scr.sphere, M, 2e-4)
    chi_poison = chi.copy()
    chi_poison[ig, 0, :w] = 1e3 * (1 + 1j)                                  # G=0 row and column
    chi_poison[ig, :w, 0] = -1e3j
    W = _gather(scr.solve(_put_stack(chi_poison, mesh, P(None, "x", "y"))))
    body = np.zeros((M, M), np.complex128)
    vb = scr.v[ig, 1:w]
    body[1:w, 1:w] = np.linalg.inv(np.eye(w - 1) - vb[:, None] * chi[ig, 1:w, 1:w]) * vb[None, :]
    e_body = cases.rel(W[ig], body)
    assert e_body <= TOL, e_body
    # the fold against the head of the explicit bordered inverse
    S = -0.02 * (np.eye(3) + 0.3 * rng.standard_normal((3, 3)))
    Y = np.zeros((3, M), np.complex128)
    Z = np.zeros((M, 3), np.complex128)
    Y[:, 1:w] = 0.02 * (rng.standard_normal((3, w - 1)) + 1j * rng.standard_normal((3, w - 1)))
    Z[1:w, :] = 0.02 * (rng.standard_normal((w - 1, 3)) + 1j * rng.standard_normal((w - 1, 3)))
    Wz = _put_stack(W[None], mesh, P(None, None, "x", "y"))
    Yd = _put_stack(Y[None], mesh, P(None, None, "x"))
    Zd = _put_stack(Z[None], mesh, P(None, "y", None))
    vc0, w0, S_eff = scr.gamma_head(Wz, S[None], Yd, Zd)
    S_eff = S_eff[0]
    geo = _geometry()
    worst, worst_red, body_term = 0.0, np.inf, 0.0
    for d in rng.standard_normal((12, 3)):
        q = 1e-4 * d / np.linalg.norm(d)
        Wd = pw.bordered_head(q, S, Y[:, 1:w], Z[1:w, :], chi[ig, 1:w, 1:w], vb)
        v0 = 8.0 * np.pi / float(q @ q)
        qSq = (1.0 - v0 / Wd[0, 0]) / v0                                       # = qᵀ S_eff q
        worst = max(worst, abs(qSq - q @ S_eff @ q) / abs(q @ S_eff @ q))
        red = S + Z[1:w].T @ body[1:w, 1:w] @ Y[:, 1:w].T                        # transposed wings
        worst_red = min(worst_red, abs(qSq - q @ red @ q) / abs(qSq))
        body_term = max(body_term, np.max(np.abs(Wd[1:, 1:] - body[1:w, 1:w])))
    print(f"Γ: body rel {e_body:.1e}; fold vs bordered inverse {worst:.1e} (transposed-wing twin "
          f"{worst_red:.1e}); vc0 {vc0.real:.4f}, wcoul0 {w0[0]:.4f}; omitted rank-3 body term, max "
          f"over directions |ΔW_bb| / max|W_body| = {body_term / np.max(np.abs(body)):.2e} (synthetic "
          f"wings; the O(q⁰) wing is zero by construction here)")
    assert worst <= 1e-8 and worst_red > 1e-3
    # S = 0 and no wings: the screened head is the bare one
    _, w00, _ = scr.gamma_head(Wz, np.zeros((1, 3, 3)), _put_stack(0 * Y[None], mesh, P(None, None, "x")),
                               _put_stack(0 * Z[None], mesh, P(None, "y", None)))
    assert abs(w00[0] - vc0) <= 1e-12 * abs(vc0)
    # red twin: a body with v(G=0) ≠ 0 at Γ is not the head-removed body
    vbad = scr.v.copy()
    vbad[ig, 0] = 8.0 * np.pi / 1e-6
    assert cases.rel(pw.dense_dyson(vbad, chi)[ig], body) > 1e-3
    # W^c: Γ head slot and the returned v
    Wc, v = scr.correlation(Wz, wcoul0=w0, vc0=vc0)
    Wc = _gather(Wc)[0]
    assert Wc[ig, 0, 0] == w0[0] - vc0 and v[ig, 0] == vc0.real
    assert cases.rel(Wc[ig, 1:w, 1:w], body[1:w, 1:w] - np.diag(vb)) <= TOL


def test_gamma_head_slab():
    """sys_dim = 2: the slab kernel's Γ cell (exact Wigner–Seitz cubature) through the same
    fold; S = 0 gives the bare head, a screening S lowers it, and the Γ body has v(0) = 0."""
    from gw.plane_wave_screening import SphereScreening
    mesh = _mesh(4)
    c = _chi_case(1)
    scr = SphereScreening(mesh, sphere=c["rs"].irr, geometry=_geometry(), sys_dim=2, kgrid=(2, 2, 1))
    ig, M = scr.gamma, scr.M
    assert scr.v[ig, 0] == 0.0
    W = _put_stack(np.zeros((1, scr.sphere.n, M, M), np.complex128), mesh, P(None, None, "x", "y"))
    Y = _put_stack(np.zeros((1, 3, M), np.complex128), mesh, P(None, None, "x"))
    Z = _put_stack(np.zeros((1, M, 3), np.complex128), mesh, P(None, "y", None))
    vc0, w0, _ = scr.gamma_head(W, np.zeros((1, 3, 3)), Y, Z)
    _, w1, _ = scr.gamma_head(W, -0.05 * np.diag([1.0, 1.0, 0.0])[None], Y, Z)
    print(f"slab Γ: vc0 {vc0.real:.4f}, wcoul0(S=0) {w0[0].real:.4f}, wcoul0(S=-0.05 in-plane) "
          f"{w1[0].real:.4f}")
    assert abs(w0[0] - vc0) <= 1e-12 * abs(vc0) and 0.0 < w1[0].real < vc0.real


# ---------------------------------------------------------------------------
# the MPA hookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_mesh", [1, 4])
def test_pole_fit_tiles_equal_one_dense_fit(n_mesh):
    from gw.mpa import pade_fit, sampling
    mesh = _mesh(n_mesh)
    c = _chi_case(1)
    scr = _screening(mesh, c["rs"].irr)
    n_q, M, n_p = scr.sphere.n, scr.M, 3
    z = sampling.double_parallel_grid(n_p, 2.0, energy_unit="Ry")
    rng = np.random.default_rng(9)
    Om = rng.uniform(0.3, 1.8, (n_q, M, M, n_p)) - 1j * rng.uniform(0.01, 0.1, (n_q, M, M, n_p))
    B = rng.standard_normal((n_q, M, M, n_p)) + 1j * rng.standard_normal((n_q, M, M, n_p))
    Wc = np.einsum("qabp,jqabp->jqab", 2 * Om * B, 1.0 / (z[:, None, None, None, None] ** 2 - Om[None] ** 2))
    live = scr.sphere.live()
    livep = np.zeros((n_q, M), bool)
    livep[:, :scr.sphere.width] = live
    mask = livep[:, :, None] & livep[:, None, :]
    wing = np.zeros((n_q, M, M), bool)
    wing[scr.gamma, 0, 1:] = wing[scr.gamma, 1:, 0] = True                 # Γ wings: not fitted
    Wc = np.where(mask[None], Wc, 0.0)
    Wd = _put_stack(Wc, mesh, P(None, None, "x", "y"))
    got_O, got_B, _, cond = scr.fit_poles(Wd, z, n_p)
    got_O, got_B = _gather(got_O), _gather(got_B)
    assert np.all(got_O[:, wing] == 0) and np.all(got_B[:, wing] == 0)
    mask = mask & ~wing
    one_O, one_B, _, _ = scr.fit_poles(Wd, z, n_p, budget_bytes=1)         # one q row per batch
    assert scr.fit_q_batch(n_p, budget_bytes=1) == 1
    assert np.array_equal(_gather(one_O), got_O) and np.array_equal(_gather(one_B), got_B)
    dO, dB, _ = pade_fit.fit_mpa_poles_batched(jnp.asarray(np.moveaxis(Wc, 0, -1).reshape(-1, 2 * n_p)),
                                                jnp.asarray(z), n_p, eig="jax_qr")
    dO = np.moveaxis(np.asarray(dO).reshape(n_q, M, M, n_p), -1, 0)
    dB = np.moveaxis(np.asarray(dB).reshape(n_q, M, M, n_p), -1, 0)
    e = max(cases.rel(np.where(mask, got_O, 0), np.where(mask, dO, 0)),
            cases.rel(np.where(mask, got_B, 0), np.where(mask, dB, 0)))
    model = np.asarray(pade_fit.eval_mpa_model(jnp.asarray(np.moveaxis(got_O, 0, -1)),
                                               jnp.asarray(np.moveaxis(got_B, 0, -1)),
                                               jnp.asarray(z)[:, None, None, None]))
    e_model = cases.rel(np.where(mask[None], model, 0), np.where(mask[None], Wc, 0))
    print(f"MPA P{n_mesh}: tiles vs dense fit {e:.1e}; model vs samples {e_model:.1e}; max cond {float(cond):.2e}")
    assert e <= 1e-10 and e_model <= 1e-8
    assert np.all(got_O[:, ~(mask | wing)] == 0) and np.all(got_B[:, ~(mask | wing)] == 0)
    zp = z[np.r_[1, 0, 2:2 * n_p]]                                           # permuted grid
    red_O, red_B, _, _ = scr.fit_poles(_put_stack(Wc, mesh, P(None, None, "x", "y")), zp, n_p)
    red = np.asarray(pade_fit.eval_mpa_model(jnp.asarray(np.moveaxis(_gather(red_O), 0, -1)),
                                             jnp.asarray(np.moveaxis(_gather(red_B), 0, -1)),
                                             jnp.asarray(z)[:, None, None, None]))
    assert cases.rel(np.where(mask[None], red, 0), np.where(mask[None], Wc, 0)) > 1e-3


# ---------------------------------------------------------------------------
# the antiunitary rule (glide group, TR broken)
# ---------------------------------------------------------------------------

def antiunitary_check(mesh, fx_spin, fx_scalar, c, geo, *, taus=(0.35,), zs=(0.6j, 0.25 + 0.4j),
                      scale=None):
    """χ at every full-grid q from the two orientations of covariant operands (K1, dense
    columns), W at z and z̄ through the Dyson solve, then the typed transport on the χ sphere
    (the scalar plan).  Returns the relative misses of each rule and its red twins, and the
    (r, r') asymmetry of χ at Γ that makes the case TR broken."""
    from gw.mixed_basis_pair_convolution import (MixedBasisPairConvolution, PairOperand,
                                                SphereSet, SphereTransport)
    from gw.plane_wave_screening import SphereScreening
    from common.gvec_fft_box import build_sphere_box_index
    out = SphereSet(*c["out_full"])
    ps = SphereSet(c["sph"], c["ngk"], c["kfrac"])
    op = PairOperand(ps, SphereTransport.identity(ps, c["ns"]))
    conv = MixedBasisPairConvolution(mesh, kgrid=c["kgrid"], fft_grid=c["fft_grid"], left=op,
                                     right=op, out=out, budget_bytes=int(1e10))
    s5 = NamedSharding(mesh, P(None, "x", None, "y", None))
    put = lambda a: fixtures._put(cases.pad_tiles(a, conv.width_carrier[0]), s5)
    X = conv.strip(conv(put(c["A"]), put(c["C"])))
    Xt = conv.strip(conv(put(c["C"]), put(c["A"])))             # the other orientation
    nq, w = X.shape[0], X.shape[1]
    # (r, r') asymmetry at Γ: X_0(G, G') against X_0(−G', −G)
    ig = int(np.flatnonzero(np.all(np.abs(out.frac) < 1e-12, axis=1))[0])
    look = {tuple(g): i for i, g in enumerate(out.gvecs[ig, :out.ngk[ig]])}
    neg = np.asarray([look[tuple(-g)] for g in out.gvecs[ig, :out.ngk[ig]]])
    Xg = X[ig, :out.ngk[ig], :out.ngk[ig]]
    asym = cases.rel(Xg, Xg[np.ix_(neg, neg)].T)
    scr = SphereScreening(mesh, sphere=out, geometry=geo, sys_dim=3, kgrid=c["kgrid"])
    M = scr.M
    if scale is None:
        scale = 0.3 / (np.max(np.abs(scr.v)) * np.max(np.abs(X)))
    tau = taus[0]

    def W_at(z):
        chi = scale * (np.exp(z * tau) * X + np.exp(-z * tau) * Xt)
        return _gather(scr.solve(_put_stack(_pad_to(chi, M, (1, 2)), mesh, P(None, "x", "y"))))[:, :w, :w]

    Wz = {z: (W_at(z), W_at(np.conj(z))) for z in zs}
    # the scalar plan's transport of the sphere from its parent rows
    plan = fx_scalar["plan"]
    kpar = np.asarray(plan.k_parent_frac)
    # each parent's full-grid row (any representative: q_row = q_par + n, G_par = G_row + n)
    d = out.frac[None, :, :] - kpar[:, None, :]
    hit = np.all(np.abs(d - np.rint(d)) < 1e-8, axis=2)
    assert np.all(hit.sum(axis=1) == 1), "a parent q is not one full-grid row"
    prow = [int(np.flatnonzero(h)[0]) for h in hit]
    shift = np.rint(out.frac[prow] - kpar).astype(np.int64)
    gpar = np.where(out.live()[prow][..., None], out.gvecs[prow] + shift[:, None, :], out.gvecs[prow])
    sidx = build_sphere_box_index(gpar, tuple(c["fft_grid"]), w, ngk_valid=out.ngk[prow])
    tr = SphereTransport.typed(plan, fft_grid=c["fft_grid"], parent_sphere_index=sidx, children=out)
    assert np.allclose(tr.spin, 1.0)
    # a rule's miss is its worst row; a red twin's miss is its worst row too (it must be seen
    # somewhere: an antiunitary row whose W is nearly real cannot tell the twin apart)
    res = dict(asym=asym, tau=0.0, tau_red=0.0, z=0.0, z_red=0.0, n_anti=int(np.sum(tr.anti)))
    # W at τ: W(τ) = Σ_z [e^{-zτ'} W(z) + e^{-z̄τ'} W(z̄)] (any real-analytic weights)
    tp = 0.4
    Wt = sum(np.exp(-z * tp) * a + np.exp(-np.conj(z) * tp) * b for z, (a, b) in Wz.items())
    for k in range(nq):
        par, src, ph, anti = int(tr.row[k]), tr.src[k], tr.phase[k], bool(tr.anti[k])
        live = np.arange(w) < out.ngk[k]
        pr = prow[par]
        img = pw.transport_operator(Wt[pr], np.where(live, src, 0), np.where(live, ph, 0), anti)
        m = live[:, None] & live[None, :]
        res["tau"] = max(res["tau"], cases.rel(np.where(m, img, 0), np.where(m, Wt[k], 0)))
        if anti:
            twin = pw.transport_operator(Wt[pr], np.where(live, src, 0), np.where(live, ph, 0), False)
            res["tau_red"] = max(res["tau_red"], cases.rel(np.where(m, twin, 0), np.where(m, Wt[k], 0)))
        for z, (Wa, Wb) in Wz.items():
            srcW = Wb[pr] if anti else Wa[pr]                  # z̄ on an antiunitary row
            img = pw.transport_operator(srcW, np.where(live, src, 0), np.where(live, ph, 0), anti)
            res["z"] = max(res["z"], cases.rel(np.where(m, img, 0), np.where(m, Wa[k], 0)))
            if anti:
                red = pw.transport_operator(Wa[pr], np.where(live, src, 0), np.where(live, ph, 0), anti)
                res["z_red"] = max(res["z_red"], cases.rel(np.where(m, red, 0), np.where(m, Wa[k], 0)))
    return res


def test_antiunitary_rule_glide():
    """{E, Θ·glide}: an antiunitary row, spin mixing, no Θ (TR broken)."""
    import test_mixed_basis_pair_convolution as t
    mesh = _mesh(4)
    fx2 = pw.glide_tr_broken_fixture(mesh, 2)
    fx1 = pw.glide_tr_broken_fixture(mesh, 1)
    c = t.covariant_case(fx2, ecut=1.3, metric=np.eye(3), box=(6, 6, 5))
    assert c["leak"] <= 1e-12, c["leak"]
    from ffi import _services
    _services.ensure_on_path()
    import vcoul
    geo = vcoul.CoulombGeometry(bvec=np.eye(3), cell_volume=(2 * np.pi) ** 3)
    r = antiunitary_check(mesh, fx2, fx1, c, geo)
    print(f"antiunitary ({{E, Θ·glide}}, {r['n_anti']} anti rows): χ(Γ) (r,r') asymmetry {r['asym']:.2e}; "
          f"τ rule {r['tau']:.1e} (no-conj twin {r['tau_red']:.1e}); z rule {r['z']:.1e} "
          f"(parent-at-z twin {r['z_red']:.1e})")
    assert r["n_anti"] > 0 and r["asym"] > 1e-3
    assert r["tau"] <= 1e-10 and r["z"] <= 1e-10
    assert r["tau_red"] > 1e-3 and r["z_red"] > 1e-3
