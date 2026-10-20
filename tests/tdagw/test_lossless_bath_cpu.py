"""Lossless screening bath gates, on small host arrays without GPU setup."""
import numpy as np
import pytest

from tdagw import keldysh_rational as kr
from tdagw import snapshot_screening as ss


def _build(energy, rho, vertices, **kwargs):
    sites = np.linspace(-3, 3, 24) + 0.4j
    return ss.lossless_bath_snapshot(vertices, energy, rho, np.arange(len(energy)),
        sites, sites + 0.17j, max_order=32, fit_tolerance=1e-9, **kwargs)


@pytest.mark.parametrize("beta", [np.inf, 2.3])
def test_lossless_bath_collective_kms_and_matrix_psd(beta):
    energy = np.array([[-0.4, 0.6, 1.2]])
    f = (np.array([1., 0., 0.]) if np.isinf(beta)
         else 1 / (1 + np.exp(beta * energy[0])))
    vertices = np.zeros((1, 3, 3, 2), complex)
    vertices[0, 1, 0] = [0.5, 0.1j]
    vertices[0, 2, 0] = [0.2 + 0.1j, 0.4]
    vertices[0, 0, 1] = vertices[0, 1, 0].conj()
    vertices[0, 0, 2] = vertices[0, 2, 0].conj()
    model = _build(energy, np.diag(f)[None], vertices)
    poles, residues = kr.lossless_residues(model["correlation"])
    lpoles, lesser = kr.lossless_residues(model["lesser"])
    np.testing.assert_allclose(poles, lpoles)
    n = (np.where(poles > 0, 0., -1.) if np.isinf(beta)
         else 1 / np.expm1(beta * poles))
    np.testing.assert_allclose(lesser, n[:, None, None] * residues, atol=2e-12)
    for l, r in zip(lesser, residues):
        assert np.linalg.eigvalsh(l).min() > -2e-12
        assert np.linalg.eigvalsh(l + r).min() > -2e-12
    assert model["receipt"]["collective_normalization_error"] < 1e-10


def test_equal_fractional_population_retains_independent_noise():
    energy = np.array([[0., 0.6]])
    vertices = np.zeros((1, 2, 2, 1), complex)
    vertices[0, 0, 1, 0] = vertices[0, 1, 0, 0] = np.sqrt(0.2)
    model = _build(energy, 0.5 * np.eye(2)[None], vertices)
    for z in (0.2 + 0.3j, -0.7 + 0.2j):
        np.testing.assert_allclose(kr.evaluate(model["correlation"], z), 0, atol=1e-14)
        expected = 0.05 / (z - 0.6) + 0.05 / (z + 0.6)
        np.testing.assert_allclose(kr.evaluate(model["lesser"], z), [[expected]], atol=1e-13)


def test_screening_pinching_reports_coherence_without_mutating_density():
    energy = np.array([[0., 1.]])
    rho = np.array([[[0.8, 0.15j], [-0.15j, 0.2]]])
    original = rho.copy()
    vertices = np.zeros((1, 2, 2, 1), complex)
    vertices[0, 0, 1, 0] = vertices[0, 1, 0, 0] = 0.3
    model = _build(energy, rho, vertices)
    np.testing.assert_array_equal(rho, original)
    assert model["receipt"]["discarded_screening_coherence_norm"] == pytest.approx(np.sqrt(2) * 0.15)


def test_lossless_bath_workspace_and_unstable_gain_refuse():
    vertices = np.zeros((1, 2, 2, 1), complex)
    vertices[0, 0, 1, 0] = vertices[0, 1, 0, 0] = 1.0
    with pytest.raises(ValueError, match="snapshot_bath_workspace"):
        _build(np.array([[0., 1.]]), np.diag([1., 0.])[None], vertices, max_pencil_bytes=1)
    with pytest.raises(ValueError, match="snapshot_lossless_stability"):
        _build(np.array([[0., 1.]]), np.diag([0., 1.])[None], vertices)


def test_stable_inversion_retains_gain_without_residue_clipping():
    vertices = np.zeros((1, 2, 2, 1), complex)
    vertices[0, 0, 1, 0] = vertices[0, 1, 0, 0] = 0.2
    model = _build(np.array([[0., 1.]]), np.diag([0., 1.])[None], vertices)
    poles, residue = kr.lossless_residues(model["correlation"])
    assert residue[poles > 0, 0, 0].real.max() < 0
    lpoles, lesser = kr.lossless_residues(model["lesser"])
    np.testing.assert_allclose(lpoles, poles)
    assert lesser[:, 0, 0].real.min() > -1e-13
    assert (lesser + residue)[:, 0, 0].real.min() > -1e-13


def test_roundoff_screening_population_correction_is_reported_and_bounded():
    energy = np.array([[0., 1.]])
    vertices = np.zeros((1, 2, 2, 1), complex)
    vertices[0, 0, 1, 0] = vertices[0, 1, 0, 0] = 0.2
    rho = np.diag([1 + 2e-14, -2e-14])[None]
    model = _build(energy, rho, vertices)
    assert model["receipt"]["screening_population_roundoff_correction"] == pytest.approx(2e-14)
    np.testing.assert_array_equal(rho, np.diag([1 + 2e-14, -2e-14])[None])
    with pytest.raises(ValueError, match="snapshot_bath_density"):
        _build(energy, np.diag([1 + 1e-6, -1e-6])[None], vertices)


def test_rank_deficient_krylov_start_preserves_coupling_span():
    coupling = np.array([[0., 1.], [0., 1.]], complex)
    projection = ss._lossless_bath_projection(np.array([1., 2.]), coupling,
        np.array([0.3 + 0.4j, 1.5 + 0.2j]), 2, 2**20)
    q = projection["basis"]
    np.testing.assert_allclose(q @ q.conj().T @ coupling, coupling, atol=1e-13)


def test_paired_screening_banks_preserve_crossing_and_collision_number():
    from tdagw.snapshot_collision import q_sigma_operators
    rng = np.random.default_rng(63)
    energy = np.array([[-1.2, -.3, .6, 1.4], [-1., -.2, .5, 1.2], [-1.1, -.4, .7, 1.3]])
    rho = np.array([np.diag(x) for x in 1 / (1 + np.exp(2 * energy))])
    partner = np.array([2, 0, 1])
    negative = np.argsort(partner)
    v = .12 * (rng.normal(size=(3, 4, 4, 2)) + 1j * rng.normal(size=(3, 4, 4, 2)))
    vm = np.empty_like(v)
    for k in range(3):
        vm[partner[k]] = v[k].swapaxes(0, 1).conj()
    fit = np.linspace(-3, 3, 24) + .4j
    checks = np.linspace(-2.8, 2.8, 21) + .53j
    opts = dict(max_order=24, fit_tolerance=.02)
    q = ss.lossless_bath_snapshot(v, energy, rho, partner, fit, checks, **opts)
    qm = ss.lossless_bath_snapshot(vm, energy, rho, negative, fit, checks,
                                   representative=q, **opts)
    assert qm["receipt"]["derived_by_crossing"]
    for z in checks:
        np.testing.assert_allclose(kr.evaluate(qm["lesser"], z),
            kr.evaluate(ss.mirror(kr.add(q["correlation"], q["lesser"])), z), atol=1e-13)
    outgoing, incoming = 0., 0.
    for vv, pp, model in ((v, partner, q), (vm, negative, qm)):
        rows = q_sigma_operators(vv.conj(), energy[pp], rho[pp], energy,
            dict(retarded=model["correlation"], lesser=model["lesser"]), eta=.03,
            active=slice(None))
        outgoing -= np.einsum("kij,kji->", rows["gamma"], rho).real
        incoming += np.trace(rows["injection"], axis1=-2, axis2=-1).sum().real
    assert abs(outgoing + incoming) / (.5 * (abs(outgoing) + abs(incoming))) < 1e-10


def test_self_inverse_screened_model_mixture_obeys_crossing():
    energy = np.array([[0., 1.]])
    rho = np.diag([.8, .2])[None]
    v = np.zeros((1, 2, 2, 1), complex)
    v[0, 0, 1, 0] = .2 + .1j
    v[0, 1, 0, 0] = .2 - .1j
    model = _build(energy, rho, v, self_negative=True)
    for z in (.3 + .4j, -.7 + .6j):
        np.testing.assert_allclose(kr.evaluate(model["correlation"], z),
            -kr.evaluate(ss.mirror(model["correlation"]), z), atol=1e-14)
        np.testing.assert_allclose(kr.evaluate(model["lesser"], z),
            kr.evaluate(ss.mirror(kr.add(model["correlation"], model["lesser"])), z), atol=1e-14)


def test_zero_gap_reservoir_matches_independent_matrix_covariance():
    energy = np.zeros((2, 1))
    f = np.array([.8, .2])
    rho = f[:, None, None].astype(complex)
    vertices = np.array([[[[.4, .1j]]], [[[.2 + .1j, .5]]]])
    fit = np.linspace(-2, 2, 12) + .3j
    model = ss.lossless_bath_snapshot(vertices, energy, rho, [1, 0], fit, fit + .2j,
                                      max_order=16, fit_tolerance=1e-10)
    v = vertices[:, 0, 0] / np.sqrt(2)
    d = f[::-1] - f
    a = f * (1 - f[::-1])
    response = np.einsum("j,jr,js->rs", d, v, v.conj())
    noise = np.einsum("j,jr,js->rs", a, v, v.conj())
    values, vectors = np.linalg.eigh(response)
    expected_r, expected_l = [], []
    for value, vector in zip(values, vectors.T):
        projector = np.outer(vector, vector.conj())
        expected_r.append(value * projector)
        expected_l.append(projector @ noise @ projector)
    poles, r = kr.lossless_residues(model["correlation"])
    lpoles, l = kr.lossless_residues(model["lesser"])
    np.testing.assert_allclose(poles, values, atol=1e-13)
    np.testing.assert_allclose(lpoles, values, atol=1e-13)
    np.testing.assert_allclose(r, expected_r, atol=1e-13)
    np.testing.assert_allclose(l, expected_l, atol=1e-13)
    assert model["receipt"]["zero_gap_unequal_population_count"] == 2


def test_opposing_zero_gap_responses_cancel_but_static_noise_survives():
    energy = np.zeros((2, 1))
    rho = np.array([[[.8]], [[.2]]], complex)
    vertices = np.full((2, 1, 1, 1), .4, complex)
    fit = np.linspace(-2, 2, 12) + .3j
    model = ss.lossless_bath_snapshot(vertices, energy, rho, [1, 0], fit, fit + .2j,
                                      max_order=16, fit_tolerance=1e-10)
    for z in (.2 + .4j, -.3 + .2j):
        np.testing.assert_allclose(kr.evaluate(model["correlation"], z), 0, atol=1e-14)
        np.testing.assert_allclose(kr.evaluate(model["lesser"], z), [[.0544 / z]], atol=1e-14)
    assert model["receipt"]["zero_gap_response_rank"] == 0


def test_thermal_static_noise_does_not_add_spurious_collective_tail():
    energy = np.array([[-.4, .6]])
    beta = 2.3
    f = 1 / (1 + np.exp(beta * energy[0]))
    vertices = np.array([[[[.2], [.3]], [[.3], [.1]]]], complex)
    model = _build(energy, np.diag(f)[None], vertices)
    rp, r = kr.lossless_residues(model["correlation"])
    lp, l = kr.lossless_residues(model["lesser"])
    for pole, residue in zip(rp, r):
        np.testing.assert_allclose(l[np.argmin(abs(lp - pole))], residue / np.expm1(beta * pole), atol=1e-12)
    assert np.any(lp == 0)
    assert model["receipt"]["zero_gap_equal_population_static_count"] == 2


def test_dense_canonical_union_capacity_is_guarded_before_allocation():
    with pytest.raises(ValueError, match="snapshot_bath_dense_workspace"):
        ss._lossless_dense_workspace(384, 0, 8, self_negative=True,
            basis_bytes=0, feedback_order=384, budget=128 * 1024**2)
    assert ss._lossless_dense_workspace(384, 0, 1, self_negative=True,
        basis_bytes=0, feedback_order=384, budget=256 * 1024**2) < 256 * 1024**2


def test_screening_gap_resolution_is_reported_and_keeps_electronic_energies():
    energy = np.array([[0.], [4e-13]])
    original = energy.copy()
    rho = np.array([[[.8]], [[.2]]], complex)
    vertices = np.array([[[[.4]]], [[[.2]]]], complex)
    fit = np.linspace(-2, 2, 12) + .3j
    model = ss.lossless_bath_snapshot(vertices, energy, rho, [1, 0], fit, fit + .2j,
                                      max_order=16, fit_tolerance=1e-10)
    receipt = model["receipt"]
    assert receipt["screening_gap_resolution_ry"] == 1e-12
    assert receipt["screening_gap_projection_count"] == 2
    assert receipt["maximum_screening_gap_adjustment_ry"] == 4e-13
    assert receipt["screening_check_error"] < 1e-10
    np.testing.assert_array_equal(energy, original)
