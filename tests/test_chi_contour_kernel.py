"""Complex-contour chi keeps the production FFT contraction and sharding."""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw import w_isdf  # noqa: E402
from gw.wavefunction_bundle import (  # noqa: E402
    BandSlices,
    PSI_XN_SPEC,
    PSI_XR_SPEC,
    PSI_YN_SPEC,
    PSI_YR_SPEC,
    Wavefunctions,
)


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
    fft3 = make_sharded_fftn_3d(
        mesh, spec, spec, axes=(0, 1, 2), norm=norm)

    def flat(x):
        return fft3(jnp.reshape(x, tuple(kgrid) + x.shape[1:])).reshape(x.shape)

    return flat


def _toy(mesh):
    rng = np.random.default_rng(20260811)
    nk, nv, nc, ns, nmu = 2, 2, 2, 2, 4
    psi = (rng.normal(size=(nk, nv + nc, ns, nmu))
           + 1j * rng.normal(size=(nk, nv + nc, ns, nmu)))
    enk = np.array([
        [-1.20, -0.50, 0.70, 1.50],
        [-1.10, -0.40, 0.90, 1.70],
    ])
    slices = BandSlices.from_band_edges(0, 0, nv, nv + nc, nv + nc)
    wfns = Wavefunctions(
        psi_xn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_XN_SPEC),
        psi_xr=_put(psi, mesh, PSI_XR_SPEC),
        psi_yr=_put(psi, mesh, PSI_YR_SPEC),
        psi_yn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_YN_SPEC),
        enk=_put(enk, mesh, P(None, None)),
        occ=_put(np.zeros_like(enk), mesh, P(None, None)),
        slices=slices,
    )
    return psi, enk, slices, wfns


def _direct_node_sum(psi, enk, slices, tau, alpha_rows, *, wrong_time=False):
    nk, _, _, nmu = psi.shape
    psi_v, psi_c = psi[:, slices.val], psi[:, slices.cond]
    eps_v, eps_c = enk[:, slices.val], enk[:, slices.cond]
    vmax, cmin = np.max(eps_v), np.min(eps_c)
    out = np.zeros((alpha_rows.shape[0], nk, nmu, nmu), np.complex128)
    tau_c = np.conj(tau) if wrong_time else tau
    for q in range(nk):
        for k in range(nk):
            kmq = (k - q) % nk
            for c in range(psi_c.shape[1]):
                dc = eps_c[kmq, c] - cmin
                for v in range(psi_v.shape[1]):
                    dv = vmax - eps_v[k, v]
                    M = np.einsum(
                        "sm,sm->m", np.conj(psi_c[kmq, c]), psi_v[k, v])
                    node_value = np.exp(-tau * dv - tau_c * dc)
                    for row in range(alpha_rows.shape[0]):
                        out[row, q] += np.dot(
                            alpha_rows[row], node_value) * np.outer(
                                M, np.conj(M))
    return out / np.sqrt(float(nk))


def test_complex_contour_matches_direct_k_minus_q_sum(monkeypatch):
    import common.fft_helpers as fft_helpers

    monkeypatch.setattr(
        fft_helpers, "make_flat_k_fftn", _emulated_flat_k_fftn)
    mesh = _mesh_xy()
    psi, enk, slices, wfns = _toy(mesh)
    tau = np.array([0.21 + 0.16j, 0.44 - 0.09j, 0.73 + 0.12j])
    signs = np.array([+1, -1, +1])
    z = np.array([0.25 + 0.08j, 0.70 + 0.12j])
    weights = np.array([
        [0.31 + 0.07j, 0.22 - 0.05j, 0.18 + 0.03j],
        [0.27 - 0.02j, 0.00 + 0.00j, 0.24 + 0.06j],
    ])
    got = w_isdf.compute_chi0_contour(
        wfns, tau, weights, signs, z,
        SimpleNamespace(nkx=2, nky=1, nkz=1), mesh)
    got = np.stack([np.asarray(jax.device_get(x)) for x in got])
    E_gap = np.min(enk[:, slices.cond]) - np.max(enk[:, slices.val])
    alpha = w_isdf._chi0_contour_alpha_rows(
        tau, weights, signs, z, E_gap)
    want = _direct_node_sum(psi, enk, slices, tau, alpha)
    np.testing.assert_allclose(got, want, rtol=2e-13, atol=2e-13)

    tau_real = np.real(tau).astype(np.complex128)
    np.testing.assert_array_equal(
        _direct_node_sum(psi, enk, slices, tau_real, alpha),
        _direct_node_sum(psi, enk, slices, tau_real, alpha, wrong_time=True))
    assert np.max(np.abs(
        want - _direct_node_sum(
            psi, enk, slices, tau, alpha, wrong_time=True))) > 1e-3


def test_contour_rows_encode_both_resolvent_signs():
    tau0 = 0.37 + 0.11j
    tau = np.array([tau0, tau0])
    weights = np.array([
        [0.42 - 0.09j, 0.31 + 0.04j],
        [0.42 - 0.09j, 0.42 - 0.09j],
    ])
    signs = np.array([+1, -1])
    z = np.array([0.63 + 0.08j, 0.0])
    got = w_isdf._chi0_contour_alpha_rows(tau, weights, signs, z, 1.20)
    np.testing.assert_allclose(
        got[0, 0], -weights[0, 0] * np.exp(-tau0 * (1.20 - z[0])))
    np.testing.assert_allclose(
        got[0, 1], -weights[0, 1] * np.exp(-tau0 * (1.20 + z[0])))
    np.testing.assert_allclose(
        np.sum(got[1]), -2.0 * weights[1, 0] * np.exp(-tau0 * 1.20))


def test_mpa_line_adapter_matches_the_two_resolvents():
    """The exact arrays assembled by the MPA line route pin both signs."""
    from gw.mpa.evaluator import damped_line_rule

    delta = 1.2
    gap_reference = 0.7
    z = np.asarray([0.4 + 0.3j])
    rule = damped_line_rule(
        0.3, 1.6, rel_tol=1.0e-8, max_order=96)
    t, h = rule["t"], rule["h"]
    tau = np.concatenate((1j * t, -1j * t))
    signs = np.concatenate((np.ones(t.size, np.int8),
                            -np.ones(t.size, np.int8)))
    weights = np.broadcast_to(
        np.concatenate((1j * h, -1j * h)), (z.size, 2 * t.size))
    alpha = w_isdf._chi0_contour_alpha_rows(
        tau, weights, signs, z, gap_reference)
    got = np.sum(
        alpha[0] * np.exp(-tau * (delta - gap_reference)))
    want = 1.0 / (z[0] - delta) - 1.0 / (z[0] + delta)
    np.testing.assert_allclose(got, want, rtol=1.0e-8, atol=1.0e-8)


def test_fractional_contour_matches_kubo_on_oriented_three_point_grid(
    monkeypatch,
):
    """Finite occupations preserve the explicit q and -q orientations."""
    import common.fft_helpers as fft_helpers

    monkeypatch.setattr(
        fft_helpers, "make_flat_k_fftn", _emulated_flat_k_fftn)
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260814)
    nk, nb, ns, nmu = 3, 4, 2, 4
    psi = (
        rng.normal(size=(nk, nb, ns, nmu))
        + 1j * rng.normal(size=(nk, nb, ns, nmu))
    )
    enk = np.array([
        [-1.3, -0.4, 0.2, 1.1],
        [-1.1, -0.2, 0.5, 1.4],
        [-1.4, -0.1, 0.7, 1.2],
    ])
    occ = np.array([
        [1.0, 0.82, 0.10, 0.0],
        [1.0, 0.61, 0.25, 0.0],
        [1.0, 0.74, -0.01, 0.0],
    ])
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)
    wfns = Wavefunctions(
        psi_xn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_XN_SPEC),
        psi_xr=_put(psi, mesh, PSI_XR_SPEC),
        psi_yr=_put(psi, mesh, PSI_YR_SPEC),
        psi_yn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_YN_SPEC),
        enk=_put(enk, mesh, P(None, None)),
        occ=_put(occ, mesh, P(None, None)),
        slices=slices,
    )
    time = np.array([0.13, 0.41, 0.79])
    z = np.array([0.32 + 0.18j, 0.77 + 0.24j])
    weights = np.array([
        [0.19, 0.31, 0.17],
        [0.23, 0.27, 0.11],
    ])
    got = w_isdf.compute_chi0_contour_fractional(
        wfns,
        time,
        weights,
        z,
        SimpleNamespace(nkx=3, nky=1, nkz=1),
        mesh,
    )
    got = np.stack([np.asarray(jax.device_get(value)) for value in got])

    want = np.zeros((z.size, nk, nmu, nmu), np.complex128)
    projection = weights * np.exp(1j * z[:, None] * time[None, :])
    for q in range(nk):
        for k in range(nk):
            kmq = (k - q) % nk
            for a in range(nb):
                for b in range(nb):
                    delta = enk[kmq, b] - enk[k, a]
                    fdiff = occ[k, a] - occ[kmq, b]
                    M = np.einsum(
                        "sm,sm->m",
                        np.conj(psi[kmq, b]),
                        psi[k, a],
                    )
                    time_sum = np.sum(
                        -1j * projection
                        * np.exp(-1j * delta * time)[None, :],
                        axis=1,
                    )
                    want[:, q] += (
                        time_sum[:, None, None]
                        * fdiff
                        * np.outer(M, np.conj(M))[None, :, :]
                    )
    want /= np.sqrt(float(nk))
    np.testing.assert_allclose(got, want, rtol=3e-13, atol=3e-13)

    f_slice, u_slice = w_isdf._occupation_support_slices(wfns.occ)
    assert f_slice == slice(0, 3)
    assert u_slice == slice(1, 4)


def test_static_fractional_gamma_matches_divided_difference():
    """The z=0 metal body includes ordered pairs and the FS diagonal."""
    mesh = _mesh_xy()
    psi, enk, slices, base = _toy(mesh)
    occ = np.array([
        [1.0, 0.83, 0.14, -0.01],
        [1.0, 0.69, 0.22, 0.00],
    ])
    surface = np.array([
        [0.01, 0.42, 0.31, -0.02],
        [0.02, 0.37, 0.28, -0.01],
    ])
    wfns = Wavefunctions(
        psi_xn=base.psi_xn,
        psi_xr=base.psi_xr,
        psi_yr=base.psi_yr,
        psi_yn=base.psi_yn,
        enk=base.enk,
        occ=_put(occ, mesh, P(None, None)),
        slices=slices,
    )
    got = w_isdf.compute_chi0_static_fractional_gamma(
        wfns,
        enk,
        occ,
        surface,
        SimpleNamespace(nk_tot=enk.shape[0], n_rmu=psi.shape[-1]),
        mesh,
        nb_logical=enk.shape[1],
    )
    got = np.asarray(jax.device_get(got))[0]

    want = np.zeros_like(got)
    for k in range(enk.shape[0]):
        for a in range(enk.shape[1]):
            for b in range(enk.shape[1]):
                divided = (
                    -surface[k, a]
                    if a == b
                    else (occ[k, a] - occ[k, b])
                    / (enk[k, a] - enk[k, b])
                )
                density = np.einsum(
                    "sm,sm->m", psi[k, a], np.conj(psi[k, b]))
                want += divided * np.outer(density, np.conj(density))
    want /= np.sqrt(float(enk.shape[0]))
    np.testing.assert_allclose(got, want, rtol=3e-13, atol=3e-13)


def _mp1_tables(enk, mu, width):
    """Consistent (f, -df/dE) tables from the production MP1 helpers."""
    from gw import efermi

    f = np.asarray(jax.device_get(efermi.mp1_occupations(enk, mu, width)))
    s = np.asarray(jax.device_get(
        efermi.mp1_negative_derivative(enk, mu, width)))
    return f, s


def _dense_static_finite_q(psi, enk, occ, surface, kminq_rows):
    """Divided-difference oracle: a at k, b at k-q, MP1 midpoint diagonal."""
    n_q = kminq_rows.shape[0]
    nk, nb = enk.shape
    nmu = psi.shape[-1]
    out = np.zeros((n_q, nmu, nmu), np.complex128)
    for j in range(n_q):
        for k in range(nk):
            kmq = int(kminq_rows[j, k])
            for a in range(nb):
                for b in range(nb):
                    de = enk[k, a] - enk[kmq, b]
                    scale = max(1.0, abs(enk[k, a]), abs(enk[kmq, b]))
                    if abs(de) > 64.0 * np.finfo(np.float64).eps * scale:
                        divided = (occ[k, a] - occ[kmq, b]) / de
                    else:
                        divided = -0.5 * (surface[k, a] + surface[kmq, b])
                    density = np.einsum(
                        "sm,sm->m", psi[k, a], np.conj(psi[kmq, b]))
                    out[j] += divided * np.outer(density, np.conj(density))
    return out / np.sqrt(float(nk))


def _dense_dynamic_finite_q(psi, enk, occ, kminq_rows, z):
    """Literal finite-z ordered-pair oracle."""
    n_q = kminq_rows.shape[0]
    nk, nb = enk.shape
    nmu = psi.shape[-1]
    out = np.zeros((n_q, nmu, nmu), np.complex128)
    for j in range(n_q):
        for k in range(nk):
            kmq = int(kminq_rows[j, k])
            for a in range(nb):
                for b in range(nb):
                    weight = ((occ[k, a] - occ[kmq, b])
                              / (enk[k, a] - enk[kmq, b] + z))
                    density = np.einsum(
                        "sm,sm->m", psi[k, a], np.conj(psi[kmq, b]))
                    out[j] += weight * np.outer(density, np.conj(density))
    return out / np.sqrt(float(nk))


def test_static_fractional_finite_q_matches_divided_difference():
    """The finite-q static kernel: b rides at k-q; Gamma row = Gamma kernel.

    Scope: CPU emulated mesh, 3-point 1-D k grid.  The production wedge-row
    to kq_map-column alignment is asserted separately by the Gamma-identity
    refusal in gw.mpa.model._metal_kminq_rows and lands numerically at the
    R2 BGW comparison.
    """
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260815)
    nk, nb, ns, nmu = 3, 4, 2, 4
    psi = (rng.normal(size=(nk, nb, ns, nmu))
           + 1j * rng.normal(size=(nk, nb, ns, nmu)))
    enk = np.array([
        [-1.3, -0.4, 0.20, 1.1],
        [-1.1, -0.2, 0.20, 1.4],
        [-1.4, -0.1, 0.70, 1.2],
    ])  # E[0,2] == E[1,2]: a degenerate (k, k-q) pair at q=2 exercises
    # the -df/dE midpoint limit at finite q.
    mu, width = 0.15, 0.08
    occ, surface = _mp1_tables(enk, mu, width)
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)
    wfns = Wavefunctions(
        psi_xn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_XN_SPEC),
        psi_xr=_put(psi, mesh, PSI_XR_SPEC),
        psi_yr=_put(psi, mesh, PSI_YR_SPEC),
        psi_yn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_YN_SPEC),
        enk=_put(enk, mesh, P(None, None)),
        occ=_put(occ, mesh, P(None, None)),
        slices=slices,
    )
    state = SimpleNamespace(
        f_kn=occ, mu_ry=mu, smearing_family="mp1", smearing_width_ry=width)
    kminq = np.stack([[(k - q) % nk for k in range(nk)] for q in range(nk)])

    got = w_isdf.compute_chi0_static_fractional(
        wfns, SimpleNamespace(nk_tot=nk, n_rmu=nmu), mesh,
        occupation_state=state, kminq_rows=kminq)
    got = np.asarray(jax.device_get(got))
    want = _dense_static_finite_q(psi, enk, occ, surface, kminq)
    np.testing.assert_allclose(got, want, rtol=3e-13, atol=3e-13)

    # The refactored Gamma kernel and the finite-q kernel with an identity
    # map are the same computation (value-level parity of the shared body).
    gamma = w_isdf.compute_chi0_static_fractional_gamma(
        wfns, enk, occ, surface,
        SimpleNamespace(nk_tot=nk, n_rmu=nmu), mesh, nb_logical=nb)
    np.testing.assert_allclose(
        np.asarray(jax.device_get(gamma))[0], got[0], rtol=1e-13, atol=1e-13)

    with pytest.raises(ValueError, match="static_fractional_needs_mp1"):
        w_isdf.compute_chi0_static_fractional(
            wfns, SimpleNamespace(nk_tot=nk, n_rmu=nmu), mesh,
            occupation_state=SimpleNamespace(
                f_kn=occ, mu_ry=mu, smearing_family="fixed",
                smearing_width_ry=0.0),
            kminq_rows=kminq)

    shifted_z = 2.0e-5j
    shifted = w_isdf.compute_chi0_direct_fractional(
        wfns, np.asarray([shifted_z]),
        SimpleNamespace(nk_tot=nk, n_rmu=nmu), mesh,
        occupation_state=state, kminq_rows=kminq)
    np.testing.assert_allclose(
        np.asarray(jax.device_get(shifted)),
        _dense_dynamic_finite_q(psi, enk, occ, kminq, shifted_z),
        rtol=3e-13, atol=3e-13)


def test_origin_row_static_value_bounds_the_shifted_sample():
    """W1.a-2: |chi(i*varpi1) - chi(0)| at finite q obeys the resolvent bound.

    Pure-numpy resolvent oracles; the deck-level 4e-8 Na number is rung R2's.
    """
    rng = np.random.default_rng(20260815)
    nk, nb, ns, nmu = 3, 4, 1, 3
    psi = (rng.normal(size=(nk, nb, ns, nmu))
           + 1j * rng.normal(size=(nk, nb, ns, nmu)))
    enk = np.array([
        [-1.3, -0.4, 0.2, 1.1],
        [-1.1, -0.2, 0.5, 1.4],
        [-1.4, -0.1, 0.7, 1.2],
    ])
    occ = np.array([
        [1.0, 0.82, 0.10, 0.0],
        [1.0, 0.61, 0.25, 0.0],
        [1.0, 0.74, -0.01, 0.0],
    ])
    varpi1 = 2.0e-5
    q = 1

    def resolvent(z):
        out = np.zeros((nmu, nmu), np.complex128)
        for k in range(nk):
            kmq = (k - q) % nk
            for a in range(nb):
                for b in range(nb):
                    delta = enk[kmq, b] - enk[k, a]
                    fdiff = occ[k, a] - occ[kmq, b]
                    if fdiff == 0.0 and delta == 0.0:
                        continue
                    density = np.einsum(
                        "sm,sm->m", psi[k, a], np.conj(psi[kmq, b]))
                    out += (fdiff / (z - delta)) * np.outer(
                        density, np.conj(density))
        return out / np.sqrt(float(nk))

    static = resolvent(0.0)
    shifted = resolvent(1j * varpi1)
    rel = np.max(np.abs(shifted - static)) / np.max(np.abs(static))
    gaps = []
    for k in range(nk):
        kmq = (k - q) % nk
        for a in range(nb):
            for b in range(nb):
                if occ[k, a] != occ[kmq, b]:
                    gaps.append(abs(enk[kmq, b] - enk[k, a]))
    # On a random (non-TRS) fixture the O(varpi) term survives, so the
    # honest bound here is linear; the quadratic (varpi/(q*v_F))**2 claim
    # holds for TRS-even physical spectra and is rung R2's deck-level
    # measurement.
    bound = varpi1 / min(gaps)
    assert rel <= 10.0 * bound, (rel, bound)
    assert rel > 0.0  # the bound is measuring something


def test_occupation_support_bandwidth_keeps_the_overshoot_edge():
    """An MP1 overshoot band at the f-support edge widens the bandwidth.

    The overshoot (f = -0.01, exactly nonzero) sits at the LOWEST band, so
    dropping it is what a tolerance-based support would do -- and would
    shrink the bandwidth by 1 Ry.  The no-slop rule keeps it.
    """
    enk = np.array([[-3.0, -2.0, 0.0, 1.0, 2.0]])
    occ = np.array([[-0.01, 1.0, 0.5, 0.0, 0.0]])
    got = w_isdf.occupation_support_bandwidth(enk, occ)
    assert got == 2.0 - (-3.0)
    trimmed = occ.copy()
    trimmed[0, 0] = 0.0  # what clipping would have produced
    assert w_isdf.occupation_support_bandwidth(enk, trimmed) == 2.0 - (-2.0)


def test_metal_plan_dispatch_census(monkeypatch):
    """Every metal point reaches its metal kernel; insulators are untouched.

    Fails if any metal line point reaches compute_chi0_contour or any metal
    existing point reaches compute_chi0 -- the exact bug W1 closes.
    """
    from gw.mpa import model, sample_plan

    calls = []

    def _fake(name, wedge=False):
        def _f(*args, **kwargs):
            calls.append(name)
            class _Chi:
                def at(self, *a):  # pragma: no cover - override path unused
                    raise AssertionError("no override in census")
                block_until_ready = staticmethod(lambda: None)
            return np.zeros((3, 2, 2), np.complex128) if wedge else \
                np.zeros((3, 2, 2), np.complex128)
        return _f

    monkeypatch.setattr(w_isdf, "compute_chi0", _fake("insulating_static"))
    monkeypatch.setattr(
        w_isdf, "compute_chi0_contour", _fake("insulating_contour"))
    monkeypatch.setattr(
        w_isdf, "compute_chi0_contour_fractional",
        _fake("fractional_contour"))
    monkeypatch.setattr(
        w_isdf, "compute_chi0_direct_fractional",
        _fake("direct_pair", True))
    monkeypatch.setattr(
        w_isdf, "occupation_support_bandwidth", lambda *a, **k: 3.0)

    written = []
    quad = SimpleNamespace(x_max=1.5)
    config = SimpleNamespace(
        mpa=SimpleNamespace(material_class="metal",
                            occupation_window_threshold=1.0),
        minimax_config=SimpleNamespace(target_error=1e-6, max_nodes=64))
    state = SimpleNamespace(
        f_kn=np.array([[1.0, 0.5, 0.0]]), mu_ry=0.1,
        smearing_family="mp1", smearing_width_ry=0.04)
    plan = sample_plan.mpa_plan(
        3, 1.5, material_class="metal", energy_unit="Ry")
    routes = sample_plan.plan_routes(plan)

    model._evaluate_samples(
        SimpleNamespace(enk=np.array([[0.0]]), occ=None), routes, quad,
        config, meta=None, mesh_xy=None, energy_reference=0.0,
        occupation_state=state,
        write_full=lambda p, chi: written.append(("full", p["role"])),
        write_wedge=lambda p, chi: written.append(("wedge", p["role"])),
        static_gamma_override=None, gamma_row=None,
        kminq_rows=np.zeros((3, 1), np.int32))

    assert "insulating_static" not in calls
    assert "insulating_contour" not in calls
    assert calls.count("direct_pair") == 1
    roles = {role: kind for kind, role in written}
    assert roles.get("near_00") == "wedge"
    assert all(kind == "full" for kind, role in written if role != "near_00")
    # 2*n_p points total, every one written exactly once
    assert len(written) == 6

    # Insulating census: only the historical kernels fire.
    calls.clear()
    written.clear()
    config_i = SimpleNamespace(
        mpa=SimpleNamespace(material_class="insulator",
                            occupation_window_threshold=1.0),
        minimax_config=SimpleNamespace(target_error=1e-6, max_nodes=64))
    monkeypatch.setattr(
        "gw.minimax_screening.build_imag_quadrature",
        lambda *a, **k: quad)
    plan_i = sample_plan.mpa_plan(
        3, 1.5, material_class="insulator", energy_unit="Ry")
    model._evaluate_samples(
        SimpleNamespace(enk=np.array([[0.0]]), occ=None),
        sample_plan.plan_routes(plan_i), quad, config_i, meta=None,
        mesh_xy=None, energy_reference=0.0, occupation_state=None,
        write_full=lambda p, chi: written.append(("full", p["role"])),
        write_wedge=lambda p, chi: written.append(("wedge", p["role"])),
        static_gamma_override=None, gamma_row=None, kminq_rows=None)
    assert "fractional_contour" not in calls
    assert "static_dd" not in calls
    assert len(written) == 6

    # A metal plan with no occupations refuses by name.
    with pytest.raises(ValueError, match="mpa_metal_needs_occupations"):
        model._evaluate_samples(
            SimpleNamespace(enk=None, occ=None), routes, quad, config,
            meta=None, mesh_xy=None, energy_reference=0.0,
            occupation_state=None,
            write_full=lambda p, chi: None, write_wedge=lambda p, chi: None,
            static_gamma_override=None, gamma_row=None, kminq_rows=None)
