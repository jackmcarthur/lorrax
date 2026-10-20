"""Small full-rho and joint-screening checks, with no GPU initialization."""

import numpy as np
import pytest
import scipy.integrate as sint

from tdagw import keldysh_rational as kr
from tdagw import snapshot_screening as ss


def _state():
    rng = np.random.default_rng(23)
    psi = 0.2 * (rng.normal(size=(2, 3, 2, 4)) + 1j * rng.normal(size=(2, 3, 2, 4)))
    energy = np.array([[-0.7, -0.1, 0.8], [-0.6, 0.05, 0.9]])
    active = np.array([[[0.7, 0.08 + 0.11j], [0.08 - 0.11j, 0.2]],
                       [[0.6, -0.04 + 0.09j], [-0.04 - 0.09j, 0.3]]])
    rho = ss.embed_active_density(np.tile([1.0, 0.0, 0.0], (2, 1)), active, 1)
    factor = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    return psi, energy, rho, factor


def test_full_rho_bubble_matches_direct_ordered_time_product_and_projection():
    psi, energy, rho, factor = _state()
    kmq = np.array([1, 0])
    z, gamma = 0.13 + 0.62j, 0.07
    actual = ss.polarization_samples(psi, energy, rho, factor, kmq, [z],
                                     adiabatic_ry=gamma, band_block=2)

    def integrand(t):
        result = np.zeros((2, 2, 2), dtype=complex)
        for k, partner in enumerate(kmq):
            # Explicit full spin/centroid Green matrices, independent of the
            # projected ordered-transition contractions under test.
            pk = psi[k].transpose(2, 1, 0).reshape(8, 3)
            pq = psi[partner].transpose(2, 1, 0).reshape(8, 3)
            for channel, (x, y) in enumerate(((rho[k], np.eye(3) - rho[partner]),
                                             (np.eye(3) - rho[k], rho[partner]))):
                positive = pk @ x @ np.diag(np.exp(-1j * energy[k] * t)) @ pk.conj().T
                negative = pq @ np.diag(np.exp(1j * energy[partner] * t)) @ y @ pq.conj().T
                bubble = np.einsum("msnt,ntms->mn", positive.reshape(4, 2, 4, 2),
                                    negative.reshape(4, 2, 4, 2))
                result[channel] += factor.conj().T @ bubble @ factor / 2
        return -1j * np.exp((1j * z - gamma) * t) * result

    reference, error = sint.quad_vec(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-10)
    assert error < 1e-8
    np.testing.assert_allclose(actual[:, 0], reference, atol=2e-10, rtol=2e-10)
    diagonal = rho * np.eye(3)[None]
    twin = ss.polarization_samples(psi, energy, diagonal, factor, kmq, [z],
                                   adiabatic_ry=gamma)
    assert np.linalg.norm(actual - twin) > 1e-4


def test_projected_vertex_sampler_matches_full_centroid_bubble_including_frozen_cross_pairs():
    psi, energy, rho, factor = _state()
    kmq, sites = np.array([1, 0]), np.array([-0.4 + 0.2j, 0.3 + 0.7j])
    vertices = np.stack([np.einsum("mr,asm,csm->acr", factor.conj(), psi[k],
                                    psi[partner].conj()) for k, partner in enumerate(kmq)])
    projected = ss.polarization_from_vertices(vertices, energy, rho, kmq, sites,
                                              adiabatic_ry=0.04, band_block=2)
    full = ss.polarization_samples(psi, energy, rho, factor, kmq, sites,
                                   adiabatic_ry=0.04, band_block=2)
    np.testing.assert_allclose(projected, full, atol=2e-15, rtol=2e-14)
    twin = vertices.copy()
    twin[:, 0, 1:] = 0
    twin[:, 1:, 0] = 0
    no_cross = ss.polarization_from_vertices(twin, energy, rho, kmq, sites,
                                             adiabatic_ry=0.04)
    assert np.linalg.norm(projected - no_cross) > 1e-3


def test_thermal_bubble_residues_obey_bose_fdt():
    energy = np.array([[-0.3, 0.4]])
    beta = 4.0
    f = 1.0 / (np.exp(beta * energy[0]) + 1.0)
    psi = np.array([[[[1.0, 0.2]], [[0.3, 0.9]]]])
    rho = np.diag(f)[None]
    factor = np.eye(2)
    less, greater, right = ss.pair_vertices(psi[0], psi[0], rho[0], rho[0], factor,
                                           slice(None), slice(None))
    for b, d in ((0, 1), (1, 0)):
        rl = np.outer(less[:, b, d], right[:, b, d].conj())
        rr = np.outer(greater[:, b, d] - less[:, b, d], right[:, b, d].conj())
        bose = 1.0 / np.expm1(beta * (energy[0, b] - energy[0, d]))
        np.testing.assert_allclose(rl, bose * rr, rtol=1e-14, atol=1e-15)


def test_joint_loewner_and_screening_keep_noise_when_retarded_bubble_vanishes():
    psi = np.array([[[[0.4, 0.1]], [[0.2, 0.3]]]])
    energy, rho, factor = np.array([[-0.2, 0.35]]), 0.5 * np.eye(2)[None], np.eye(2)

    def sampler(z):
        return ss.polarization_samples(psi, energy, rho, factor, [0], z, adiabatic_ry=0.08)

    fit_sites = np.linspace(-1.1, 1.1, 32) + 0.3j
    check_sites = np.linspace(-1, 1, 9) + 0.47j
    model = ss.screen_snapshot(sampler, fit_sites, check_sites, max_order=12, fit_tolerance=1e-8)
    assert model["receipt"]["fit_order"] <= 12
    for z in check_sites:
        np.testing.assert_allclose(kr.evaluate(model["correlation"], z), 0, atol=2e-12)
        np.testing.assert_allclose(kr.evaluate(model["lesser"], z), sampler([z])[0, 0],
                                   rtol=2e-8, atol=1e-11)
        assert np.linalg.norm(kr.evaluate(model["lesser"], z)) > 1e-4


def test_joint_fit_refuses_underresolved_samples_and_unstable_collective_poles():
    poles = np.array([-0.7 - 0.1j, 0.2 - 0.2j, 0.8 - 0.08j])
    residues = np.array([0.01, -0.03, 0.02])

    def sampler(z):
        value = np.sum(residues[None] / (np.asarray(z)[:, None] - poles[None]), axis=1)
        return np.stack([value[:, None, None], 2 * value[:, None, None]])

    with pytest.raises(ValueError, match="GATE snapshot_(fit_error|causal_poles)"):
        ss.screen_snapshot(sampler, np.linspace(-1, 1, 20) + 0.3j,
                           np.linspace(-0.9, 0.9, 9) + 0.41j, max_order=1, fit_tolerance=1e-8)
    with pytest.raises(ValueError, match="snapshot_regulator"):
        ss.polarization_samples(np.ones((1, 1, 1, 1)), np.zeros((1, 1)),
                                 np.ones((1, 1, 1)), np.ones((1, 1)), [0], [1j],
                                 adiabatic_ry=0)

    def invalid_noise(z):
        values = sampler(z)
        if np.any(np.asarray(z).imag == 0):
            values[:] = np.nan
        return values

    with pytest.raises(ValueError, match="snapshot_noise_error"):
        ss.screen_snapshot(invalid_noise, np.linspace(-1, 1, 20) + 0.3j,
                           np.linspace(-0.9, 0.9, 9) + 0.41j, max_order=8, fit_tolerance=1e-8)


def test_screened_joint_realization_compacts_canceled_poles_and_diagonalizes():
    psi = np.array([[[[0.4, 0.1]], [[0.2, 0.3]]]])
    energy = np.array([[-0.2, 0.35]])
    # Population inversion is allowed: no passive PSD constraint is imposed.
    rho = np.diag([0.2, 0.8])[None]

    def sampler(z):
        return ss.polarization_samples(psi, energy, rho, np.eye(2), [0], z,
                                       adiabatic_ry=0.08)

    model = ss.screen_snapshot(sampler, np.linspace(-1.1, 1.1, 40) + 0.3j,
                               np.linspace(-1, 1, 9) + 0.47j,
                               max_order=16, fit_tolerance=1e-7)
    assert model["receipt"]["noise_check_error"] < 1e-10
    assert model["retarded"][0].shape[0] == 2
    assert model["lesser"][0].shape[0] == 4
    for key in ("retarded", "lesser"):
        A, B, C, D = model[key]
        poles, vectors = np.linalg.eig(A)
        assert np.linalg.cond(vectors) < 1e6
        residue_left, residue_right = C @ vectors, np.linalg.solve(vectors, B)
        z = 0.17 + 0.2j
        by_residues = D + (residue_left / (z - poles)[None]) @ residue_right
        np.testing.assert_allclose(by_residues, kr.evaluate(model[key], z), rtol=1e-12, atol=1e-12)


def test_coulomb_projection_reciprocity_is_factor_gauge_independent_and_rejects_complex_sketch():
    rng = np.random.default_rng(1981)
    x = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    v = x @ x.conj().T
    real_sketch = rng.normal(size=(6, 2))

    def nystrom(bare, sketch):
        y = bare @ sketch
        values, vectors = np.linalg.eigh(sketch.conj().T @ y)
        return (y @ vectors) / np.sqrt(values)[None]

    def check(fq, fm):
        return ss.projected_coulomb_reciprocity(
            fq.conj().T @ fq, fm.conj().T @ fm, fm.conj().T @ fq.conj())

    fq, fm = nystrom(v, real_sketch), nystrom(v.conj(), real_sketch)
    u, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    assert check(fq, fm @ u)["projected_bare_reciprocity_error"] < 1e-7
    complex_sketch = real_sketch + 1j * rng.normal(size=(6, 2))
    with pytest.raises(ValueError, match="snapshot_coulomb_reciprocity"):
        check(nystrom(v, complex_sketch), nystrom(v.conj(), complex_sketch))


def test_projected_sampler_k_sharding_matches_numpy_with_k_and_band_padding():
    jax = pytest.importorskip("jax")
    cpus = jax.devices("cpu")
    if len(cpus) < 4:
        pytest.skip("four CPU devices required; set XLA_FLAGS=--xla_force_host_platform_device_count=4")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    psi, energy, rho, factor = _state()
    kmq = np.array([1, 0], np.int32)
    sites = np.array([-0.4 + 0.2j, 0.3 + 0.7j, 0.0 + 0.05j])
    vertices = np.stack([np.einsum("mr,asm,csm->acr", factor.conj(), psi[k],
                                    psi[partner].conj()) for k, partner in enumerate(kmq)])
    reference = ss.polarization_from_vertices(vertices, energy, rho, kmq, sites,
                                               adiabatic_ry=0.04, band_block=2)
    mesh = Mesh(np.asarray(cpus[:4]).reshape(2, 2), ("x", "y"))
    with jax.enable_x64(True), jax.default_device(cpus[0]):
        padded = np.pad(vertices, ((0, 2), (0, 0), (0, 0), (0, 0)))
        T = jax.device_put(padded, NamedSharding(mesh, P(("x", "y"), None, None, None)))
        sample = ss.make_projected_sampler(mesh, band_block=2)
        actual = sample(T, energy, rho, kmq, sites, 0.04, 1.0)
        np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=2e-14)


def test_mirror_and_exact_bubble_crossing_with_complex_coherence():
    rng = np.random.default_rng(53)
    psi = 0.2 * (rng.normal(size=(3, 2, 1, 3)) + 1j * rng.normal(size=(3, 2, 1, 3)))
    energy = np.array([[-0.4, 0.5], [-0.3, 0.6], [-0.2, 0.45]])
    rho = np.broadcast_to(np.array([[0.7, 0.08j], [-0.08j, 0.3]]), (3, 2, 2))
    factor = rng.normal(size=(3, 1)) + 1j * rng.normal(size=(3, 1))
    kmq = np.array([2, 0, 1])
    sites = np.array([-0.3 + 0.2j, 0.6 + 0.4j])
    q = ss.polarization_samples(psi, energy, rho, factor, kmq, sites, adiabatic_ry=0.1)
    qm = ss.polarization_samples(psi, energy, rho, factor.conj(), np.argsort(kmq),
                                 -sites.conj(), adiabatic_ry=0.1)
    np.testing.assert_allclose(qm[0], -q[1].conj(), atol=2e-16, rtol=2e-14)
    np.testing.assert_allclose(qm[1], -q[0].conj(), atol=2e-16, rtol=2e-14)
    h = kr.pole_residue_realization([-0.2 - 0.1j], np.array([[[0.1 + 0.03j]]]),
                                    constant=np.array([[0.4j]]))
    for z in sites:
        np.testing.assert_allclose(kr.evaluate(ss.mirror(h), z),
                                   -kr.evaluate(h, -z.conjugate()).conj(), atol=1e-15)


def test_paired_screening_crossing_and_partner_exact_sample_gate():
    psi = np.array([[[[0.4, 0.1]]], [[[0.1, 0.3]]], [[[0.2, 0.2]]]])
    energy = np.array([[-0.3], [0.1], [0.5]])
    rho = np.array([[[0.8]], [[0.5]], [[0.2]]])
    factor = np.array([[0.5 + 0.1j], [0.2 - 0.1j]])
    kmq = np.array([2, 0, 1])

    def sample(z, negative=False):
        return ss.polarization_samples(psi, energy, rho,
            factor.conj() if negative else factor,
            np.argsort(kmq) if negative else kmq, z, adiabatic_ry=0.1)

    fit = np.linspace(-1.1, 1.1, 40) + 0.3j
    check = np.linspace(-1, 1, 9) + 0.47j
    model = ss.screen_snapshot(sample, fit, check, max_order=16, fit_tolerance=1e-7)
    partner = ss.screen_snapshot_partner(model, lambda z: sample(z, True), check)
    greater = kr.add(model["correlation"], model["lesser"])
    for z in check:
        np.testing.assert_allclose(kr.evaluate(partner["lesser"], z),
            kr.evaluate(ss.mirror(greater), z), rtol=1e-9, atol=1e-13)
    with pytest.raises(ValueError, match="snapshot_partner_error"):
        ss.screen_snapshot_partner(model, lambda z: sample(z, True) * 1.1, check)


def test_self_inverse_screening_crossing_with_full_matrix_density():
    psi, energy, rho, _ = _state()
    factor = np.array([[0.2], [0.3], [0.1], [0.2]])

    def sample(z):
        return ss.polarization_samples(psi, energy, rho, factor, [1, 0], z,
                                       adiabatic_ry=0.1)

    check = np.linspace(-1.4, 1.4, 11) + 0.47j
    model = ss.screen_snapshot(sample, np.linspace(-1.6, 1.6, 96) + 0.3j, check,
                               max_order=32, fit_tolerance=1e-7, self_negative=True)
    greater = kr.add(model["correlation"], model["lesser"])
    for z in check:
        np.testing.assert_allclose(kr.evaluate(model["lesser"], z),
            kr.evaluate(ss.mirror(greater), z), rtol=1e-7, atol=1e-11)


def test_screen_snapshot_forwards_explicit_pencil_capacity(monkeypatch):
    captured = {}

    def fit(*args, **options):
        captured.update(options)
        raise RuntimeError("pencil forwarding checked")

    monkeypatch.setattr(ss, "joint_loewner", fit)
    with pytest.raises(RuntimeError, match="pencil forwarding checked"):
        ss.screen_snapshot(lambda sites: np.zeros((2, len(sites), 1, 1), complex),
            np.arange(4) + 0.2j, np.array([0.5 + 0.3j]), max_pencil_bytes=256 * 1024 ** 2)
    assert captured["max_pencil_bytes"] == 256 * 1024 ** 2
