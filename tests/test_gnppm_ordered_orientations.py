"""The GN-PPM on a time-reversal-broken deck: ordered particle-hole
orientations in chi0(i*omega_p), the two-residue fit, and the Sigma
assignment.  Every identity of ``docs/dev/notes/DERIVATION_gnppm_
nonhermitian.md`` is measured here against an exact oracle, and every gate
ships with the arm in which it returns FALSE.

The oracles are literal: the Adler-Wiser ordered-pair sum for chi0, the
imaginary-axis contour integral for Sigma, and pole models whose residues
are known.  Nothing here re-derives a production kernel; the production
kernels are CALLED (``w_isdf.compute_chi0_imag_ordered`` through the real
``_get_chi_minimax_kernel`` with the emulated flat-k FFT of
``tests/test_chi_contour_kernel.py``; ``minimax_screening.
fit_gn_ppm_from_wc_pair``; ``ppm_sigma._residue_for_space``;
``common.chi_from_dipole.compute_S_omega``).

Single CPU device suffices; a 2x2 emulated mesh is used when available::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_ENABLE_X64=1 \\
        python -m pytest tests/test_gnppm_ordered_orientations.py -q
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw import minimax_screening as ms  # noqa: E402
from gw import w_isdf  # noqa: E402
from gw.mpa import evaluator as mpa_evaluator  # noqa: E402
from gw.mpa import pade_fit as mpa_pade_fit  # noqa: E402
from gw.ppm_sigma import _residue_for_space  # noqa: E402
from gw.screening import assert_probe_chi_reuse_supported  # noqa: E402
from gw.wavefunction_bundle import (  # noqa: E402
    BandSlices,
    PSI_XN_SPEC,
    PSI_XR_SPEC,
    PSI_YN_SPEC,
    PSI_YR_SPEC,
    Wavefunctions,
)

OMEGA_P = 2.0


# ---------------------------------------------------------------------------
#  Which tree is this file testing?  (lane G's in-band canary, same reason)
# ---------------------------------------------------------------------------

def test_this_file_is_testing_the_tree_it_was_launched_from():
    want = os.environ.get("LORRAX_CHECKOUT")
    assert want, ("LORRAX_CHECKOUT is unset, so this file cannot say which "
                  "tree it tested; set it before trusting any verdict here")
    root = os.path.realpath(want) + os.sep
    for mod in (w_isdf, ms):
        assert os.path.realpath(mod.__file__).startswith(root), (
            f"{mod.__name__} resolved to {mod.__file__}, which is NOT under "
            f"{want} -- this run tested a different checkout")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _adj(a):
    return np.conj(np.swapaxes(a, -1, -2))


def _herm_rel(a):
    return float(np.max(np.abs(a - _adj(a))) / np.max(np.abs(a)))


def _herm_split(a):
    return 0.5 * (a + _adj(a)), 0.5 * (a - _adj(a))


def _mesh_xy():
    devices = np.asarray(jax.devices("cpu"), dtype=object)
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    elif devices.size >= 2:
        devices = devices[:2].reshape(1, 2)
    else:
        devices = devices[:1].reshape(1, 1)
    return Mesh(devices, ("x", "y"))


def _put(a, mesh, spec):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))


def _emulated_flat_k_fftn(mesh, kgrid, spec, *, norm="ortho",
                          out_spec=None):
    from common.fft_helpers import make_sharded_fftn_3d

    assert out_spec is None or out_spec == spec
    fft3 = make_sharded_fftn_3d(mesh, spec, spec, axes=(0, 1, 2), norm=norm)

    def flat(x):
        return fft3(jnp.reshape(x, tuple(kgrid) + x.shape[1:])).reshape(x.shape)

    return flat


def _random_states(rng, nk, nb, nr):
    """Orthonormal complex states at every k -- a time-reversal-BROKEN model
    (no k <-> -k conjugation relation is imposed)."""
    out = []
    for _ in range(nk):
        a = rng.normal(size=(nr, nb)) + 1j * rng.normal(size=(nr, nb))
        q, _r = np.linalg.qr(a)
        out.append(q)
    return np.stack(out)                      # (nk, nr, nb)


def _lehmann_chi0(psi, eps, nocc, z):
    """Exact chi0(r, r'; z) of a finite model, BOTH ordered orientations.

    ``psi`` is (nk, nr, nb) with orthonormal columns; the k-sum is diagonal
    (a q = 0 object).  This is ``reports/four_current_head_frequency_audit_
    2026-09-01/chi_herm_toy.py``'s definition, kept verbatim.
    """
    nk, nr, nb = psi.shape
    f = np.array([1.0] * nocc + [0.0] * (nb - nocc))
    out = np.zeros((nr, nr), complex)
    for k in range(nk):
        p, e = psi[k], eps[k]
        for i in range(nb):
            for j in range(nb):
                if f[i] == f[j]:
                    continue
                rho = np.conj(p[:, i]) * p[:, j]
                out += (f[i] - f[j]) * np.outer(rho, np.conj(rho)) / (
                    z - (e[j] - e[i]))
    return out


# ---------------------------------------------------------------------------
#  1. The exact chi0(i*omega) is Hermitian only under Theta (the toy of the
#     four-current audit, promoted to a gate).
# ---------------------------------------------------------------------------

def test_exact_chi0_on_the_imaginary_axis_is_hermitian_only_under_theta():
    rng = np.random.default_rng(0)
    nr, nb, nocc, nk = 12, 6, 3, 8
    eps = []
    for _ in range(nk):
        e = np.sort(rng.uniform(0, 1, nb))
        e[nocc:] += 1.0
        eps.append(e)
    eps = np.stack(eps)
    broken = _random_states(rng, nk, nb, nr)
    half = _random_states(rng, nk // 2, nb, nr)
    # Theta-symmetric partner set: psi_{-k} = conj(psi_k), same energies.
    sym = np.concatenate([half, np.conj(half)])
    eps_sym = np.concatenate([eps[: nk // 2], eps[: nk // 2]])

    chi_b = _lehmann_chi0(broken, eps, nocc, 2.0j)
    chi_s = _lehmann_chi0(sym, eps_sym, nocc, 2.0j)
    assert _herm_rel(chi_b) > 1.0e-1, "the TR-broken model is not broken"
    assert _herm_rel(chi_s) < 1.0e-13
    assert _herm_rel(_lehmann_chi0(broken, eps, nocc, 0.0j)) < 1.0e-13, (
        "the odd channel carries a factor omega and must vanish at omega=0")
    # chi0(i*omega) is REAL in a real-space basis for any system, and its
    # anti-Hermitian part is therefore real antisymmetric.
    assert np.max(np.abs(chi_b.imag)) < 1.0e-13 * np.max(np.abs(chi_b))
    _h, a = _herm_split(chi_b)
    assert np.max(np.abs(a + a.T)) < 1.0e-13 * np.max(np.abs(a))


# ---------------------------------------------------------------------------
#  2. The production route reproduces the exact ordered chi0 through the
#     real chi kernel, and pins the orientation of the kernel's own object.
# ---------------------------------------------------------------------------

def _toy_bundle(mesh, rng):
    nk, nv, nc, ns, nmu = 3, 2, 2, 2, 4
    nb = nv + nc
    psi = (rng.normal(size=(nk, nb, ns, nmu))
           + 1j * rng.normal(size=(nk, nb, ns, nmu)))
    # Energies constant in k so the transition set has FOUR distinct gaps
    # {0.9, 1.1, 1.3, 1.5}: four tau nodes then interpolate any resolvent
    # on that set exactly, which removes the quadrature from the gate.
    enk = np.tile(np.array([-0.6, -0.4, 0.5, 0.9]), (nk, 1))
    occ = np.tile(np.array([1.0, 1.0, 0.0, 0.0]), (nk, 1))
    slices = BandSlices.from_band_edges(0, 0, nv, nb, nb)
    wfns = Wavefunctions(
        psi_xn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_XN_SPEC),
        psi_xr=_put(psi, mesh, PSI_XR_SPEC),
        psi_yr=_put(psi, mesh, PSI_YR_SPEC),
        psi_yn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_YN_SPEC),
        enk=_put(enk, mesh, P(None, None)),
        occ=_put(occ, mesh, P(None, None)),
        slices=slices,
    )
    return psi, enk, slices, wfns


def _exact_interpolating_rule(deltas, omega_p, tau):
    """alpha, beta with sum_l alpha_l e^{-tau_l D} == D/(D^2+w^2) EXACTLY on
    the finite gap set (generalised Vandermonde solve)."""
    V = np.exp(-np.outer(deltas, tau))
    even = deltas / (deltas ** 2 + omega_p ** 2)
    odd = omega_p / (deltas ** 2 + omega_p ** 2)
    return np.linalg.solve(V, even), np.linalg.solve(V, odd)


def _exact_ordered_chi0_flat_q(psi, enk, slices, z):
    """Adler-Wiser at every flat q of a 1-D k grid, both orientations.

    Pair index convention of the production oracle
    (``tests/test_chi_contour_kernel._direct_node_sum``): at index q the
    kernel's own object is ``conj(psi_c[k-q]) psi_v[k]``, the -Delta-pole
    orientation; its partner ``conj(psi_v[k]) psi_c[k+q]`` is the +Delta
    one.  ``1/sqrt(nk)`` is the ortho-FFT normalisation of the kernel.
    """
    nk, _nb, _ns, nmu = psi.shape
    psi_v, psi_c = psi[:, slices.val], psi[:, slices.cond]
    eps_v, eps_c = enk[:, slices.val], enk[:, slices.cond]
    out = np.zeros((nk, nmu, nmu), np.complex128)
    for q in range(nk):
        for k in range(nk):
            kmq, kpq = (k - q) % nk, (k + q) % nk
            for v in range(psi_v.shape[1]):
                for c in range(psi_c.shape[1]):
                    rho_p = np.einsum("sm,sm->m", np.conj(psi_v[k, v]),
                                      psi_c[kpq, c])
                    d_p = eps_c[kpq, c] - eps_v[k, v]
                    rho_m = np.einsum("sm,sm->m", np.conj(psi_c[kmq, c]),
                                      psi_v[k, v])
                    d_m = eps_c[kmq, c] - eps_v[k, v]
                    out[q] += (np.outer(rho_p, np.conj(rho_p)) / (z - d_p)
                               - np.outer(rho_m, np.conj(rho_m)) / (z + d_m))
    return out / np.sqrt(float(nk))


def test_ordered_route_reproduces_the_exact_chi0_through_the_production_kernel(
        monkeypatch):
    import common.fft_helpers as fft_helpers
    from symmetry_maps import q_negation_index

    monkeypatch.setattr(fft_helpers, "make_flat_k_fftn", _emulated_flat_k_fftn)
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260901)
    psi, enk, slices, wfns = _toy_bundle(mesh, rng)
    nk = psi.shape[0]
    deltas = np.array([0.9, 1.1, 1.3, 1.5])
    tau = np.array([0.3, 0.8, 1.5, 2.6])
    alpha, beta = _exact_interpolating_rule(deltas, OMEGA_P, tau)
    meta = SimpleNamespace(nkx=nk, nky=1, nkz=1, nk_tot=nk)
    q_neg = q_negation_index((nk, 1, 1))

    quad = SimpleNamespace(tau=tau, alpha=alpha, alpha_odd=beta, n_odd_extra=0)
    got = np.asarray(jax.device_get(w_isdf.compute_chi0_imag_ordered(
        wfns, quad, meta, mesh, q_neg_index=q_neg)))
    want = _exact_ordered_chi0_flat_q(psi, enk, slices, 1j * OMEGA_P)
    scale = np.max(np.abs(want))
    assert np.max(np.abs(got - want)) < 1.0e-12 * scale

    # NON-VACUOUS: the exact object has an O(1) anti-Hermitian part at q=0
    # (this is a broken-TR model), which the incumbent even route misses.
    _h, a = _herm_split(want[0])
    assert np.max(np.abs(a)) > 1.0e-2 * scale, "model degenerated to TRS"
    inc_quad = SimpleNamespace(tau=tau, alpha=alpha)
    inc = np.asarray(jax.device_get(w_isdf.compute_chi0(
        wfns, inc_quad, meta, mesh)))
    assert np.max(np.abs(inc[0] - want[0])) > 1.0e-2 * scale
    assert np.max(np.abs(inc - _herm_split_flat(want)[0])) < 1.0e-12 * scale, (
        "the incumbent even route must equal the Hermitian half exactly")

    # RECIPROCITY holds by construction on the ordered route.
    assert np.max(np.abs(got - np.conj(got[q_neg]))) < 1.0e-12 * scale

    # THE SIGN OF i*beta IS PHYSICS: the kernel's own object is the -Delta
    # pole orientation, so flipping the odd weight's sign is a magnetisation
    # flip and must FAIL against the same oracle by twice the odd part.
    wrong = SimpleNamespace(tau=tau, alpha=alpha, alpha_odd=-beta,
                            n_odd_extra=0)
    got_wrong = np.asarray(jax.device_get(w_isdf.compute_chi0_imag_ordered(
        wfns, wrong, meta, mesh, q_neg_index=q_neg)))
    assert np.max(np.abs(got_wrong - want)) > 1.0e-2 * scale


def _herm_split_flat(a):
    return _herm_split(a)


def test_ordered_route_refuses_a_quadrature_without_the_odd_kernel(monkeypatch):
    import common.fft_helpers as fft_helpers
    from symmetry_maps import q_negation_index

    monkeypatch.setattr(fft_helpers, "make_flat_k_fftn", _emulated_flat_k_fftn)
    mesh = _mesh_xy()
    _psi, _enk, _slices, wfns = _toy_bundle(mesh, np.random.default_rng(3))
    quad = SimpleNamespace(tau=np.array([0.3, 0.8]), alpha=np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="chi0_imag_ordered_needs_odd_kernel"):
        w_isdf.compute_chi0_imag_ordered(
            wfns, quad, SimpleNamespace(nkx=3, nky=1, nkz=1), mesh,
            q_neg_index=q_negation_index((3, 1, 1)))


def test_ordered_contour_route_reproduces_exact_imaginary_chi0_in_one_sweep(
        monkeypatch):
    """The MPA contour route keeps the broken-TR anti-Hermitian channel.

    The negative control is the former symmetric ``+/-time`` completion on
    exactly the same wavefunctions, nodes and frequency.  The production
    kernel factory is counted because a numerically right result obtained by
    a second response sweep would violate the route's scaling contract.
    """
    import common.fft_helpers as fft_helpers
    from symmetry_maps import q_negation_index

    monkeypatch.setattr(fft_helpers, "make_flat_k_fftn", _emulated_flat_k_fftn)
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260902)
    psi, enk, slices, wfns = _toy_bundle(mesh, rng)
    nk = psi.shape[0]
    meta = SimpleNamespace(nkx=nk, nky=1, nkz=1, nk_tot=nk)
    q_neg = q_negation_index((nk, 1, 1))
    z = np.asarray([1j * OMEGA_P], dtype=np.complex128)
    rule = mpa_evaluator.damped_line_rule(
        OMEGA_P, 1.5, rel_tol=1.0e-14, max_order=256)

    kernel_calls = 0
    get_kernel = w_isdf._get_chi_minimax_kernel

    def _counted_kernel(*args, **kwargs):
        nonlocal kernel_calls
        kernel_calls += 1
        return get_kernel(*args, **kwargs)

    monkeypatch.setattr(w_isdf, "_get_chi_minimax_kernel", _counted_kernel)
    got = np.asarray(jax.device_get(w_isdf.compute_chi0_contour_ordered(
        wfns, rule["t"], rule["h"], z, meta, mesh,
        q_neg_index=q_neg)))
    assert kernel_calls == 1, "ordered contour used more than one chi kernel"

    want = _exact_ordered_chi0_flat_q(psi, enk, slices, z[0])
    scale = np.max(np.abs(want))
    assert np.max(np.abs(got - want)) < 1.0e-12 * scale
    _want_h, want_a = _herm_split(want[0])
    _got_h, got_a = _herm_split(got[0])
    assert np.max(np.abs(want_a)) > 1.0e-2 * scale, (
        "negative-control model degenerated to time-reversal symmetry")
    assert np.max(np.abs(got_a - want_a)) < 1.0e-12 * scale

    # RED TWIN: the incumbent contour assigns both resolvents to the same
    # orientation and therefore returns the Hermitian half at imaginary z.
    t, h = rule["t"], rule["h"]
    tau = np.concatenate((1j * t, -1j * t))
    signs = np.concatenate((np.ones(t.size, np.int8),
                            -np.ones(t.size, np.int8)))
    weights = np.broadcast_to(
        np.concatenate((1j * h, -1j * h)), (1, 2 * t.size))
    incumbent = np.asarray(jax.device_get(w_isdf.compute_chi0_contour(
        wfns, tau, weights, signs, z, meta, mesh)))
    assert np.max(np.abs(incumbent[0] - want[0])) > 1.0e-2 * scale
    assert np.max(np.abs(_herm_split(incumbent[0])[1])) < 1.0e-12 * scale


# ---------------------------------------------------------------------------
#  3. The two-point fit recovers two Hermitian residues from a pole model
#     whose W(i*omega_p) is NOT Hermitian; the incumbent formula fed the same
#     data breaks its own crossing-closure premise.
# ---------------------------------------------------------------------------

def _pole_model(rng, nq=2, nmu=6, *, distinct=True):
    Om = 1.2 + 0.8 * rng.uniform(size=(nq, nmu, nmu))
    Om = 0.5 * (Om + np.swapaxes(Om, -1, -2))                # real symmetric
    def herm():
        m = rng.normal(size=(nq, nmu, nmu)) + 1j * rng.normal(size=(nq, nmu, nmu))
        return 0.5 * (m + _adj(m))
    R_plus = herm()
    R_minus = herm() if distinct else R_plus.copy()
    # Keep W^c(0) - W^c(i w_p) well separated from zero on every lane, as
    # the fit's own ``safe`` predicate requires.
    Wc0 = -(R_plus + R_minus) / Om
    Wcp = -((R_plus + R_minus) * Om + 1j * OMEGA_P * (R_plus - R_minus)) / (
        OMEGA_P ** 2 + Om ** 2)
    return Om, R_plus, R_minus, Wc0, Wcp


def _fit(Wc0, Wcp, *, ordered):
    fit = ms.fit_gn_ppm_from_wc_pair(
        jnp.asarray(Wc0), jnp.asarray(Wcp), 1j * OMEGA_P,
        fallback_omega=2.0, n_mu_logical=Wc0.shape[-1],
        ordered_orientations=ordered)
    om = np.asarray(jax.device_get(fit.omega_qmunu))
    B = np.asarray(jax.device_get(fit.B_qmunu))
    D = (None if fit.B_odd_qmunu is None
         else np.asarray(jax.device_get(fit.B_odd_qmunu)))
    good = np.asarray(jax.device_get(fit.valid_qmunu))
    return om, B, D, good


def test_ordered_fit_recovers_two_hermitian_residues_from_a_nonhermitian_probe():
    rng = np.random.default_rng(11)
    Om, R_plus, R_minus, Wc0, Wcp = _pole_model(rng)
    assert _herm_rel(Wcp) > 1.0e-1, "probe degenerated to Hermitian"
    assert _herm_rel(Wc0) < 1.0e-13

    om, B, D, good = _fit(Wc0, Wcp, ordered=True)
    assert good.all()
    scale = np.max(np.abs(R_plus)) + np.max(np.abs(R_minus))
    np.testing.assert_allclose(om, Om, rtol=0, atol=1.0e-12)
    np.testing.assert_allclose(B, 0.5 * (R_plus + R_minus), rtol=0,
                               atol=1.0e-12 * scale)
    np.testing.assert_allclose(D, 0.5 * (R_plus - R_minus), rtol=0,
                               atol=1.0e-12 * scale)
    # Both residues Hermitian, D Hermitian, Omega symmetric -- the premises
    # of the crossing closure, supplied PER BRANCH.
    for r in (B, D, _residue_for_space("cond", B, D),
              _residue_for_space("val", B, D)):
        assert _herm_rel(r) < 1.0e-12
    assert np.max(np.abs(om - np.swapaxes(om, -1, -2))) < 1.0e-12
    np.testing.assert_allclose(_residue_for_space("cond", B, D), R_plus,
                               atol=1.0e-12 * scale)
    np.testing.assert_allclose(_residue_for_space("val", B, D), R_minus,
                               atol=1.0e-12 * scale)
    assert _residue_for_space("cond", B, None) is B

    # RED TWIN: the incumbent single-residue formula on the same probe gives
    # a non-Hermitian B and a non-symmetric Omega -- the closure premise
    # (DERIVATION_channel_hermiticity.md section 1.3) fails.
    om_raw, B_raw, D_raw, _g = _fit(Wc0, Wcp, ordered=False)
    assert D_raw is None
    assert _herm_rel(B_raw) > 1.0e-3
    assert np.max(np.abs(om_raw - np.swapaxes(om_raw, -1, -2))) > 1.0e-3


def test_ordered_fit_is_the_incumbent_fit_on_a_hermitian_probe():
    rng = np.random.default_rng(12)
    Om, R_plus, R_minus, Wc0, Wcp = _pole_model(rng, distinct=False)
    assert _herm_rel(Wcp) < 1.0e-13
    om_o, B_o, D_o, _ = _fit(Wc0, Wcp, ordered=True)
    om_i, B_i, D_i, _ = _fit(Wc0, Wcp, ordered=False)
    assert D_i is None
    assert np.max(np.abs(D_o)) < 1.0e-14 * np.max(np.abs(B_o))
    np.testing.assert_allclose(om_o, om_i, rtol=0, atol=1.0e-13)
    np.testing.assert_allclose(B_o, B_i, rtol=0, atol=1.0e-13 * np.max(np.abs(B_i)))


def test_hermitian_faraday_ct_family_has_zero_ordered_residue():
    """A Hermitian CT/TC probe is the exact ``D_H=0`` negative control.

    This does not assume that a real magnet's *completed* probe is Hermitian:
    the charge/body environment can supply its ordered anti-Hermitian part.
    It proves that when that part is absent the shared GN split returns
    ``D=0`` exactly.
    """
    omega = np.full((1, 4, 4), 1.7, dtype=np.float64)
    residue = np.zeros((1, 4, 4), dtype=np.complex128)
    ct = np.asarray((0.2j, -0.35j, 0.11j))
    residue[0, 0, 1:] = ct
    residue[0, 1:, 0] = np.conj(ct)
    Wc0 = -2.0 * residue / omega
    Wcp = -2.0 * residue * omega / (OMEGA_P ** 2 + omega ** 2)
    np.testing.assert_array_equal(Wcp, _adj(Wcp))

    _om, B, D, _good = _fit(Wc0, Wcp, ordered=True)
    np.testing.assert_array_equal(D, np.zeros_like(D))
    production = _residue_for_space("cond", B, D)
    d_zero_twin = _residue_for_space("cond", B, np.zeros_like(D))
    np.testing.assert_array_equal(production - d_zero_twin, 0.0)

    Wcp_hermitized = 0.5 * (Wcp + _adj(Wcp))
    _om_h, B_h, D_h, _good_h = _fit(Wc0, Wcp_hermitized, ordered=True)
    np.testing.assert_array_equal(D_h, np.zeros_like(D_h))
    np.testing.assert_array_equal(B_h, B)


def test_faraday_sigma_consumer_is_hall_on_off_and_magnetisation_odd(
        monkeypatch):
    """The rank-four analytic head reaches Sigma; its zero twin does not."""
    from gw.head_correction import FaradayHeadPPMFactorCarrier
    from gw.photon_layout import (
        PhotonBasisLayout, pack_photon_channel_vectors)
    from gw.photon_sigma import compute_ppm_faraday_head_sigma_omega

    # This cell exercises the q0 closed form, not the host FFI implementation
    # of the final small band projection.  Production GPU gates exercise the
    # mandatory FFI route; the announced XLA fallback keeps this CPU oracle
    # about the pole/branch algebra.
    monkeypatch.setenv("LORRAX_BANDS_GEMM_FFI", "0")
    mesh = _mesh_xy()
    rng = np.random.default_rng(404)
    nk, nb, ns, nmu = 1, 4, 4, 4
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)

    def bundle():
        psi = (rng.normal(size=(nk, nb, ns, nmu))
               + 1j * rng.normal(size=(nk, nb, ns, nmu)))
        return Wavefunctions(
            psi_xn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_XN_SPEC),
            psi_xr=_put(psi, mesh, PSI_XR_SPEC),
            psi_yr=_put(psi, mesh, PSI_YR_SPEC),
            psi_yn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_YN_SPEC),
            enk=_put(np.array([[-0.8, -0.3, 0.5, 1.1]]),
                     mesh, P(None, None)),
            occ=_put(np.array([[1.0, 1.0, 0.0, 0.0]]),
                     mesh, P(None, None)),
            slices=slices)

    charge = bundle()
    transverse = bundle()
    layout = PhotonBasisLayout.from_centroid_extents(nmu, nmu, mesh)

    def one_channel_rows(channel, values, axis_name):
        spec = P(None, axis_name)
        vectors = []
        for index in range(4):
            row = (np.asarray(values, dtype=np.complex128)
                   if index == channel else np.zeros(nmu, np.complex128))
            vectors.append(_put(row[None], mesh, spec))
        packed = pack_photon_channel_vectors(
            tuple(vectors), layout, mesh, axis_name=axis_name)[0]
        row = jnp.sum(packed, axis=0)
        return jnp.zeros_like(packed).at[0].set(row)

    left_charge = one_channel_rows(
        0, [0.3 + 0.1j, -0.2 + 0.4j, 0.1 - 0.3j, 0.2], "x")
    right_current = one_channel_rows(
        1, [-0.1 + 0.2j, 0.5 - 0.3j, 0.2 + 0.1j, -0.4], "y")
    left_current = one_channel_rows(
        1, [0.2 - 0.1j, 0.1 + 0.3j, -0.5j, 0.4], "x")
    right_charge = one_channel_rows(
        0, [-0.3j, 0.2 + 0.1j, -0.4 + 0.2j, 0.5], "y")
    B_pairs = ((left_charge, right_current),
               (left_current, right_charge))
    D_pairs = tuple(
        [(0.2 * left, right) for left, right in B_pairs]
        + [(jnp.zeros_like(left), right) for left, right in B_pairs])
    omega_h = 1.3
    static_pairs = tuple(
        (-2.0 * left / omega_h, right) for left, right in B_pairs)
    probe_ratio = omega_h ** 2 / (OMEGA_P ** 2 + omega_h ** 2)
    probe_pairs = tuple(
        (probe_ratio * left, right) for left, right in static_pairs)

    def carrier(sign):
        return FaradayHeadPPMFactorCarrier(
            omega_h_ry=omega_h,
            B_pairs=tuple((sign * left, right)
                          for left, right in B_pairs),
            D_pairs=tuple((sign * left, right)
                          for left, right in D_pairs),
            static_pairs=tuple((sign * left, right)
                               for left, right in static_pairs),
            probe_pairs=tuple((sign * left, right)
                              for left, right in probe_pairs),
            sigma_H_static=sign * np.array([0.0, 0.0, 1.0e-4]),
            sigma_H_probe=sign * np.array([0.0, 0.0, 3.0e-5]),
            probe_frequency_ry=1j * OMEGA_P,
            probe_fit_relative_error=0.0,
            odd_even_residue_ratio=0.2,
            raw_pair_overlaps=(0.0j, 0.0j))

    kwargs = dict(
        wfns_charge=charge, wfns_transverse=transverse,
        photon_layout=layout,
        meta=SimpleNamespace(nk_tot=nk, kgrid=(1, 1, 1)),
        mesh_xy=mesh, omega_grid_ry=np.array([-0.2, 0.3]),
        efermi_ry=0.0)
    plus = np.asarray(compute_ppm_faraday_head_sigma_omega(
        faraday_ppm=carrier(+1.0), **kwargs))
    minus = np.asarray(compute_ppm_faraday_head_sigma_omega(
        faraday_ppm=carrier(-1.0), **kwargs))
    zero = np.asarray(compute_ppm_faraday_head_sigma_omega(
        faraday_ppm=carrier(0.0), **kwargs))

    assert np.max(np.abs(plus)) > 1.0e-6
    np.testing.assert_allclose(minus, -plus, rtol=0.0, atol=1.0e-13)
    np.testing.assert_array_equal(zero, np.zeros_like(zero))


def test_debug_odd_residue_off_reproduces_even_only_residues_and_refuses_trs(
        monkeypatch):
    """The debug A/B arm keeps ordered B but makes D=0, and is magnet-only."""
    rng = np.random.default_rng(1202)
    _Om, _R_plus, _R_minus, Wc0, Wcp = _pole_model(rng)
    messages = []
    monkeypatch.setenv("LORRAX_DEBUG_GN_ODD_RESIDUE_OFF", "1")

    fit = ms.fit_gn_ppm_from_wc_pair(
        jnp.asarray(Wc0), jnp.asarray(Wcp), 1j * OMEGA_P,
        fallback_omega=2.0, n_mu_logical=Wc0.shape[-1],
        ordered_orientations=True, print_fn=messages.append)
    B = np.asarray(jax.device_get(fit.B_qmunu))
    D = np.asarray(jax.device_get(fit.B_odd_qmunu))
    assert np.array_equal(D, np.zeros_like(D))
    np.testing.assert_array_equal(_residue_for_space("cond", B, D), B)
    np.testing.assert_array_equal(_residue_for_space("val", B, D), B)
    assert any("WARNING -- DEBUG" in line and "D=0" in line
               for line in messages)

    with pytest.raises(ValueError, match="debug_gn_odd_residue_off_scope"):
        ms.fit_gn_ppm_from_wc_pair(
            jnp.asarray(Wc0), jnp.asarray(Wcp), 1j * OMEGA_P,
            fallback_omega=2.0, n_mu_logical=Wc0.shape[-1],
            ordered_orientations=False, print_fn=messages.append)


def test_ordered_fit_refuses_a_real_axis_probe():
    rng = np.random.default_rng(13)
    _Om, _Rp, _Rm, Wc0, Wcp = _pole_model(rng)
    with pytest.raises(ValueError, match="gn_ppm_ordered_probe_axis"):
        ms.fit_gn_ppm_from_wc_pair(
            jnp.asarray(Wc0), jnp.asarray(Wcp), 200.0 + 0.0j,
            fallback_omega=2.0, n_mu_logical=Wc0.shape[-1],
            ordered_orientations=True)


# ---------------------------------------------------------------------------
#  4. Crossing closure: the pair-adjoint identity holds for R_+ and R_-
#     separately, and fails for the raw non-Hermitian probe's B.
# ---------------------------------------------------------------------------

def test_crossing_closure_premise_holds_per_branch_not_for_the_raw_probe():
    rng = np.random.default_rng(14)
    Om, R_plus, R_minus, Wc0, Wcp = _pole_model(rng)
    om, B, D, _ = _fit(Wc0, Wcp, ordered=True)
    om_raw, B_raw, _d, _g = _fit(Wc0, Wcp, ordered=False)
    tau, w = 0.83, 1.21
    for R in (_residue_for_space("cond", B, D), _residue_for_space("val", B, D)):
        u = w - om
        Z = R * np.exp(-1j * tau * u)
        Z_missing = R * np.exp(+1j * tau * u)
        assert np.max(np.abs(Z_missing - _adj(Z))) < 1.0e-12 * np.max(np.abs(Z))
    u = w - om_raw
    Z = B_raw * np.exp(-1j * tau * u)
    Z_missing = B_raw * np.exp(+1j * tau * u)
    assert np.max(np.abs(Z_missing - _adj(Z))) > 1.0e-3 * np.max(np.abs(Z)), (
        "the raw probe's B satisfied the pair-adjoint identity -- the red "
        "twin is not red")


# ---------------------------------------------------------------------------
#  5. Sigma closure: which residue each branch consumes, against the
#     imaginary-axis contour integral.
# ---------------------------------------------------------------------------

def _sigma_closed_form(psi, eps, nocc, R_plus, R_minus, Om, E):
    """Sigma_c(E)(mu,nu) = sum_occ psi psi^H . R_-/(E-e+Om) + sum_emp ... R_+/(E-e-Om)."""
    nb = psi.shape[1]
    out = np.zeros_like(R_plus)
    for m in range(nb):
        pp = np.outer(psi[:, m], np.conj(psi[:, m]))
        if m < nocc:
            out += pp * R_minus / (E - eps[m] + Om)
        else:
            out += pp * R_plus / (E - eps[m] - Om)
    return out


def _sigma_contour(psi, eps, R_plus, R_minus, Om, E, n=4000):
    """-(1/2pi) int dnu G(E - i nu) W_c(i nu), nu = tan(theta), Gauss-Legendre."""
    x, wgl = np.polynomial.legendre.leggauss(n)
    theta = 0.5 * np.pi * x
    nu = np.tan(theta)
    jac = 0.5 * np.pi * wgl / np.cos(theta) ** 2
    nb = psi.shape[1]
    out = np.zeros_like(R_plus)
    for t, j in zip(nu, jac):
        z = 1j * t
        G = np.zeros_like(R_plus)
        for m in range(nb):
            G += np.outer(psi[:, m], np.conj(psi[:, m])) / (E - z - eps[m])
        Wc = R_plus / (z - Om) - R_minus / (z + Om)
        out += j * G * Wc
    return -out / (2.0 * np.pi)


def test_sigma_assigns_R_plus_to_empty_states_and_R_minus_to_occupied():
    rng = np.random.default_rng(15)
    nmu, nb, nocc = 3, 4, 2
    psi, _r = np.linalg.qr(rng.normal(size=(nmu, nb)) + 1j * rng.normal(size=(nmu, nb)))
    eps = np.array([-1.0, -0.6, 0.7, 1.2])
    Om = 1.5 + 0.3 * rng.uniform(size=(nmu, nmu))
    Om = 0.5 * (Om + Om.T)
    def herm():
        m = rng.normal(size=(nmu, nmu)) + 1j * rng.normal(size=(nmu, nmu))
        return 0.5 * (m + m.conj().T)
    R_plus, R_minus = herm(), herm()
    E = 0.05                                   # midgap: no crossed poles
    want = _sigma_contour(psi, eps, R_plus, R_minus, Om, E)
    got = _sigma_closed_form(psi, eps, nocc, R_plus, R_minus, Om, E)
    scale = np.max(np.abs(want))
    assert np.max(np.abs(got - want)) < 1.0e-9 * scale
    # RED TWIN: residues swapped between the two A-spaces.
    swapped = _sigma_closed_form(psi, eps, nocc, R_minus, R_plus, Om, E)
    assert np.max(np.abs(swapped - want)) > 1.0e-2 * scale
    # And the production selector says the same thing.
    B, D = 0.5 * (R_plus + R_minus), 0.5 * (R_plus - R_minus)
    np.testing.assert_allclose(_residue_for_space("cond", B, D), R_plus)
    np.testing.assert_allclose(_residue_for_space("val", B, D), R_minus)


def test_mpa_nonhermitian_fit_and_sigma_close_the_imaginary_contour():
    """Ordered MPA samples retain D through the fit and Sigma branches.

    The pole model is sampled at four imaginary frequencies, reconstructed
    on a dense imaginary-axis grid, and consumed by the same R+/R- branch
    algebra as production.  Hermitising those samples is the red twin.
    """
    rng = np.random.default_rng(20260903)
    nmu, nb, nocc, n_p = 3, 4, 2, 2
    psi, _ = np.linalg.qr(
        rng.normal(size=(nmu, nb)) + 1j * rng.normal(size=(nmu, nb)))
    eps = np.array([-1.1, -0.55, 0.65, 1.25])
    poles = np.array([0.9, 1.8])

    def hermitian():
        value = (rng.normal(size=(n_p, nmu, nmu))
                 + 1j * rng.normal(size=(n_p, nmu, nmu)))
        return 0.5 * (value + _adj(value))

    R_plus, R_minus = hermitian(), hermitian()
    B = 0.5 * (R_plus + R_minus)
    D = 0.5 * (R_plus - R_minus)
    z_fit = 1j * np.array([0.22, 0.61, 1.37, 3.1])

    def model(z, b=B, d=D):
        zz = np.asarray(z, np.complex128)
        den = zz[:, None, None, None] ** 2 - poles[None, :, None, None] ** 2
        num = (2.0 * poles[None, :, None, None] * b[None]
               + 2.0 * zz[:, None, None, None] * d[None])
        return np.sum(num / den, axis=1)

    W_positive = model(z_fit)
    W_negative = model(-z_fit)
    positive_tile = W_positive.transpose(1, 2, 0).reshape(nmu * nmu, -1)
    negative_tile = W_negative.transpose(1, 2, 0).reshape(nmu * nmu, -1)
    Omega_f, B_f, D_f, diagnostics = mpa_pade_fit.fit_mpa_poles_batched(
        jnp.asarray(positive_tile), jnp.asarray(z_fit), n_p,
        W_negative_tile=jnp.asarray(negative_tile), return_odd=True)
    Omega_f, B_f, D_f, diagnostics = jax.device_get(
        (Omega_f, B_f, D_f, diagnostics))
    Omega_f = np.asarray(Omega_f).reshape(nmu, nmu, n_p).transpose(2, 0, 1)
    B_f = np.asarray(B_f).reshape(nmu, nmu, n_p).transpose(2, 0, 1)
    D_f = np.asarray(D_f).reshape(nmu, nmu, n_p).transpose(2, 0, 1)
    assert np.all(np.asarray(diagnostics["valid"]))

    dense_z = 1j * np.linspace(-40.0, 40.0, 2001)
    exact_dense = model(dense_z)
    fitted_dense = np.empty_like(exact_dense)
    for mu in range(nmu):
        for nu in range(nmu):
            fitted_dense[:, mu, nu] = np.asarray(
                mpa_pade_fit.eval_mpa_model(
                    Omega_f[:, mu, nu], B_f[:, mu, nu], dense_z,
                    B_odd=D_f[:, mu, nu]))
    dense_error = np.max(np.abs(fitted_dense - exact_dense))
    assert dense_error < 1.0e-9

    E = 0.04
    direct = sum(
        _sigma_contour(
            psi, eps, R_plus[p], R_minus[p], poles[p], E)
        for p in range(n_p))
    fitted = sum(
        _sigma_closed_form(
            psi, eps, nocc,
            _residue_for_space("cond", B_f[p], D_f[p]),
            _residue_for_space("val", B_f[p], D_f[p]), Omega_f[p], E)
        for p in range(n_p))
    sigma_error = np.max(np.abs(fitted - direct))
    assert sigma_error < 1.0e-9

    # RED TWIN: Hermitising W(i nu) removes its odd half and makes R+=R-=B.
    W_hermitian = 0.5 * (W_positive + _adj(W_positive))
    herm_tile = W_hermitian.transpose(1, 2, 0).reshape(nmu * nmu, -1)
    Om_h, B_h, D_h, _ = mpa_pade_fit.fit_mpa_poles_batched(
        jnp.asarray(herm_tile), jnp.asarray(z_fit), n_p,
        W_negative_tile=jnp.asarray(herm_tile), return_odd=True)
    Om_h, B_h, D_h = map(np.asarray, jax.device_get((Om_h, B_h, D_h)))
    Om_h = Om_h.reshape(nmu, nmu, n_p).transpose(2, 0, 1)
    B_h = B_h.reshape(nmu, nmu, n_p).transpose(2, 0, 1)
    D_h = D_h.reshape(nmu, nmu, n_p).transpose(2, 0, 1)
    hermitian_d = np.max(np.abs(D_h))
    assert hermitian_d < 1.0e-12
    hermitian_sigma = sum(
        _sigma_closed_form(
            psi, eps, nocc,
            _residue_for_space("cond", B_h[p], D_h[p]),
            _residue_for_space("val", B_h[p], D_h[p]), Om_h[p], E)
        for p in range(n_p))
    negative_control = np.max(np.abs(hermitian_sigma - direct))
    assert negative_control > 1.0e-3
    print(
        "MPA_ORDERED_CLOSURE "
        f"dense_W={dense_error:.12e} sigma_contour={sigma_error:.12e} "
        f"hermitized_D={hermitian_d:.12e} "
        f"hermitized_sigma_shift={negative_control:.12e}")


def test_broken_tr_sigma_table_prints_the_measured_odd_contribution(tmp_path):
    """The public ``sigC_odd`` value is ordered Sigma minus its D=0 twin."""
    from file_io.sigma_output import write_sigma_to_file
    from tests.harness import parse_eqp_rows

    rng = np.random.default_rng(1501)
    nmu, nb, nocc = 3, 4, 2
    psi, _ = np.linalg.qr(
        rng.normal(size=(nmu, nb)) + 1j * rng.normal(size=(nmu, nb)))
    eps = np.array([-1.0, -0.6, 0.7, 1.2])
    Om = 1.5 + 0.3 * rng.uniform(size=(nmu, nmu))
    Om = 0.5 * (Om + Om.T)

    def herm():
        m = rng.normal(size=(nmu, nmu)) + 1j * rng.normal(size=(nmu, nmu))
        return 0.5 * (m + m.conj().T)

    R_plus, R_minus = herm(), herm()
    B = 0.5 * (R_plus + R_minus)
    total = _sigma_closed_form(
        psi, eps, nocc, R_plus, R_minus, Om, 0.05)
    even = _sigma_closed_form(psi, eps, nocc, B, B, Om, 0.05)
    odd_diag = np.diagonal(total - even)[None, :]
    corr_diag = np.diagonal(total)[None, :]
    zero = np.zeros_like(corr_diag)

    out = tmp_path / "broken_tr_sigma.dat"
    write_sigma_to_file(
        zero, filename=str(out), sigma_coh_kij_eV=corr_diag,
        hartree_kij_eV=zero, kpoints_crys=np.zeros((1, 3)),
        sx_label="sigX", corr_label="sigC", total_label="sigXC",
        sigma_c_odd_kn_eV=odd_diag)
    rows = parse_eqp_rows(
        out, labels=("sigX", "sigC", "sigXC", "sigC_odd"))
    np.testing.assert_allclose(rows[:, 5], np.real(odd_diag[0]), atol=5.1e-7)

    trs = tmp_path / "trs_sigma.dat"
    write_sigma_to_file(
        zero, filename=str(trs), sigma_coh_kij_eV=corr_diag,
        hartree_kij_eV=zero, kpoints_crys=np.zeros((1, 3)),
        sx_label="sigX", corr_label="sigC", total_label="sigXC")
    assert "sigC_odd=" not in trs.read_text()


# ---------------------------------------------------------------------------
#  6. Magnetisation flip: the odd residue is exactly odd, the even fit is
#     invariant, the two residues swap.
# ---------------------------------------------------------------------------

def _w_from_states(psi, eps, nocc, V, z):
    chi = _lehmann_chi0(psi, eps, nocc, z)
    n = V.shape[0]
    return np.linalg.solve(np.eye(n) - V @ chi, V)


def test_magnetisation_flip_is_exactly_odd_in_the_odd_residue():
    rng = np.random.default_rng(16)
    nr, nb, nocc, nk = 8, 6, 3, 4
    eps = []
    for _ in range(nk):
        e = np.sort(rng.uniform(0, 1, nb))
        e[nocc:] += 1.0
        eps.append(e)
    eps = np.stack(eps)
    psi = _random_states(rng, nk, nb, nr)
    # Real symmetric positive-definite bare interaction (q = 0, real space).
    a = rng.normal(size=(nr, nr))
    V = 0.05 * (a @ a.T) + 0.5 * np.eye(nr)

    def fit_for(states):
        W0 = _w_from_states(states, eps, nocc, V, 0.0j)
        Wp = _w_from_states(states, eps, nocc, V, 1j * OMEGA_P)
        om, B, D, good = _fit((W0 - V)[None], (Wp - V)[None], ordered=True)
        return Wp, om[0], B[0], D[0], good[0]

    Wp, om, B, D, good = fit_for(psi)
    Wp_f, om_f, B_f, D_f, good_f = fit_for(np.conj(psi))     # Theta: m -> -m
    # A random-state model has a few lanes with Omega^2 <= 0 (the ordinary
    # GN "invalid mode", served the fallback pole, as production does); the
    # flip maps lane (mu, nu) to (nu, mu) and must map the validity mask
    # with it.  Every identity below then holds on every lane, fallback
    # lanes included, because the fallback pole is symmetric.
    assert np.array_equal(good_f, good.T)
    assert good.mean() > 0.8, "model degenerated: most lanes invalid"
    scale = np.max(np.abs(B))
    assert np.max(np.abs(D)) > 1.0e-3 * scale, "odd residue vanished: not a magnet"
    # Theta transposes every real-space kernel: W_flip = W^T.  In the real
    # q = 0 basis W(i*omega) is REAL, so its Hermitian part is real
    # symmetric (invariant) and its anti-Hermitian part is real
    # antisymmetric (flips sign outright).
    h, a = _herm_split(Wp)
    h_f, a_f = _herm_split(Wp_f)
    assert np.max(np.abs(Wp.imag)) < 1.0e-12 * np.max(np.abs(Wp))
    assert np.max(np.abs(h_f - h)) < 1.0e-11 * np.max(np.abs(h))
    assert np.max(np.abs(a_f + a)) < 1.0e-11 * np.max(np.abs(a))
    # The even fit is invariant; the odd residue is EXACTLY odd.  Here D =
    # i a s with a real antisymmetric and s real symmetric, so D is imaginary
    # antisymmetric -- Hermitian, and its transpose is its negative.
    np.testing.assert_allclose(om_f, om, rtol=0, atol=1.0e-11)
    assert np.max(np.abs(B_f - B)) < 1.0e-11 * scale
    assert np.max(np.abs(D_f + D)) < 1.0e-11 * scale
    assert np.max(np.abs(D.real)) < 1.0e-11 * scale
    assert np.max(np.abs(D + D.T)) < 1.0e-11 * scale
    # ... so the two pole branches swap roles under time reversal: R_+ of
    # the flipped magnet is R_- of the original and vice versa.
    R_plus, R_minus = _residue_for_space("cond", B, D), _residue_for_space("val", B, D)
    assert np.max(np.abs(_residue_for_space("cond", B_f, D_f) - R_minus)) < 1.0e-11 * scale
    assert np.max(np.abs(_residue_for_space("val", B_f, D_f) - R_plus)) < 1.0e-11 * scale
    assert np.max(np.abs(R_plus - R_minus)) > 1.0e-3 * scale


# ---------------------------------------------------------------------------
#  7. The scalar charge head is exactly time-reversal-even.
# ---------------------------------------------------------------------------

def test_scalar_head_annihilates_the_odd_part_of_the_head_tensor():
    from common.chi_from_dipole import compute_S_omega

    rng = np.random.default_rng(17)
    nk, nv, nc = 3, 2, 3
    nb = nv + nc
    v = rng.normal(size=(3, nk, nc, nv)) + 1j * rng.normal(size=(3, nk, nc, nv))
    dE = 0.8 + 0.6 * rng.uniform(size=(nk, nc, nv))
    cell_volume, nspin, nspinor = 100.0, 1, 1
    pref = 4.0 / (cell_volume * nk * nspin * nspinor)
    z = 1j * OMEGA_P

    # Exact Adler-Wiser head tensor, both orientations, q^2 coefficient.
    S = np.zeros((3, 3), complex)
    for k in range(nk):
        for c in range(nc):
            for vv in range(nv):
                Pab = np.outer(np.conj(v[:, k, c, vv]), v[:, k, c, vv])
                d = dE[k, c, vv]
                S += pref / (2.0 * d * d) * (Pab / (z - d) - Pab.T / (z + d))
    S_even, S_odd = _herm_split(S)
    assert np.max(np.abs(S_odd)) > 1.0e-2 * np.max(np.abs(S)), "not a magnet"
    # The odd part is ANTISYMMETRIC in ab ...
    assert np.max(np.abs(S_odd + S_odd.T)) < 1.0e-13 * np.max(np.abs(S))
    # ... so every direction average of it is zero, exactly.
    for _ in range(64):
        qh = rng.normal(size=3)
        qh /= np.linalg.norm(qh)
        assert abs(qh @ S_odd @ qh) < 1.0e-14 * np.max(np.abs(S))
        assert abs(qh @ S @ qh - qh @ S_even @ qh) < 1.0e-13 * np.max(np.abs(S))

    # The production dipole -> S(omega) builder equals the EVEN part in its
    # symmetric block and in every scalar head; its antisymmetric block is
    # the single-orientation artefact, not the physical odd part (note in
    # the lane M report).
    deltaE = np.zeros((nk, nb, nb))
    dipole = np.zeros((3, nk, nb, nb), complex)
    for k in range(nk):
        for c in range(nc):
            for vv in range(nv):
                deltaE[k, nv + c, vv] = dE[k, c, vv]
                dipole[:, k, nv + c, vv] = v[:, k, c, vv]
    f_nk = np.zeros((nk, nb))
    f_nk[:, :nv] = 1.0
    S_prod = np.asarray(compute_S_omega(
        jnp.asarray(dipole), jnp.asarray(deltaE), jnp.asarray(f_nk),
        cell_volume, nk, nspin, nspinor, jnp.asarray([z])))[0]
    sym = lambda m: 0.5 * (m + m.T)
    assert np.max(np.abs(sym(S_prod) - sym(S_even))) < 1.0e-12 * np.max(np.abs(S))
    for _ in range(16):
        qh = rng.normal(size=3)
        qh /= np.linalg.norm(qh)
        assert abs(qh @ S_prod @ qh - qh @ S @ qh) < 1.0e-12 * np.max(np.abs(S))


# ---------------------------------------------------------------------------
#  8. Refusals and the odd-kernel rule.
# ---------------------------------------------------------------------------

def test_probe_chi_reuse_refuses_by_name_on_a_magnet():
    with pytest.raises(RuntimeError, match="gn_probe_chi_reuse_tr_broken"):
        assert_probe_chi_reuse_supported("auto", tr_odd=True)
    assert_probe_chi_reuse_supported("off", tr_odd=True)
    assert_probe_chi_reuse_supported("auto", tr_odd=False)


@pytest.mark.parametrize("x_min,x_max", [(0.12, 6.0), (0.0687, 8.0), (0.04, 10.0)])
def test_odd_kernel_rule_meets_the_even_rules_gate_and_leaves_it_untouched(
        x_min, x_max):
    even = ms.solve_laplace_minimax_imag_interval(
        x_min, x_max, OMEGA_P, target_error=1.0e-6, max_nodes=64)
    both = ms.solve_laplace_minimax_imag_interval(
        x_min, x_max, OMEGA_P, target_error=1.0e-6, max_nodes=64,
        with_odd_kernel=True)
    n_even = even.node_count
    assert both.n_odd_extra == both.node_count - n_even
    np.testing.assert_array_equal(both.tau[:n_even], even.tau)
    np.testing.assert_array_equal(both.alpha[:n_even], even.alpha)
    assert np.all(both.alpha[n_even:] == 0.0)
    gate = max(1.0e-6, float(even.max_error))
    assert both.max_error_odd <= gate
    x = np.geomspace(x_min, x_max, 20000)
    E = np.exp(-np.outer(x, both.tau))
    assert np.max(np.abs(E @ both.alpha_odd - OMEGA_P / (x * x + OMEGA_P ** 2))) <= 1.5 * gate
    assert np.max(np.abs(E @ both.alpha - x / (x * x + OMEGA_P ** 2))) <= 1.5 * gate
    # The rule is what the ordered chi0 route consumes: gamma is
    # -(alpha - i beta) e^{-tau E_gap}, real nodes, complex weights.
    gamma = -(both.alpha - 1j * both.alpha_odd)
    assert np.max(np.abs(E @ gamma + 1.0 / (x + 1j * OMEGA_P))) <= 3.0 * gate
