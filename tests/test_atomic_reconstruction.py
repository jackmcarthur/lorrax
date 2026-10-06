"""Atomic reconstruction checks independent of the scattering generator."""
import hashlib
import numpy as np
import pytest

from psp.atomic_reconstruction import (
    optimal_radial_projectors, log_radial_weights,
    write_atomic_reconstruction, load_atomic_reconstruction,
    evaluate_radial_correction, radial_operator_difference,
    read_atomic_scattering_bank, generated_upf_text,
    scattering_branch_diagnostics,
    read_atomic_frozen_core,
    radial_gradient_norm_squared,
)


def test_paired_projectors_preserve_an_independent_heldout_combination():
    r = np.geomspace(1e-4, 2, 513)
    w = log_radial_weights(r)
    base = np.stack((r * np.exp(-r), r ** 2 * np.exp(-2 * r)))
    derivative = np.stack(((1 - r) * np.exp(-r), (2 * r - 2 * r ** 2) * np.exp(-2 * r)))
    correction = r ** 2 * (2 - r) ** 2
    correction_derivative = 4 * r * (2 - r) * (1 - r)
    coefficients = np.asarray(((1, 0), (0, 1), (1, 1), (1, -2), (2, 0.3)))
    atomic_map = np.asarray((0.2, -0.13))
    bank = {"ps_u": coefficients @ base,
            "ps_du_dr": coefficients @ derivative,
            "ae_u": coefficients @ (base + atomic_map[:, None] * correction),
            "ae_du_dr": coefficients @ (derivative + atomic_map[:, None] * correction_derivative)}
    basis = optimal_radial_projectors(r, w, bank, discarded_weight=1e-12)
    heldout = np.asarray((0.37, -0.41))
    pseudo = heldout @ base
    amplitudes = basis["ps_u"].T @ (w * pseudo)
    reconstructed = pseudo + basis["delta_u"] @ amplitudes
    exact = heldout @ (base + atomic_map[:, None] * correction)
    np.testing.assert_allclose(reconstructed, exact, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(basis["ps_gram"], np.eye(2), atol=3e-12)
    assert np.max(np.abs(basis["ae_gram"] - np.eye(2))) > 0.1


@pytest.mark.parametrize("count", (512, 513))
def test_atomic_quadrature_integrates_an_independent_power(count):
    r = np.geomspace(1e-4, 2, count)
    expected = (r[-1] ** 4 - r[0] ** 4) / 4
    actual = log_radial_weights(r) @ r ** 3
    assert abs(actual - expected) / expected < 3e-7


def test_radial_correction_preserves_physical_first_derivative():
    r = np.linspace(0.1, 2, 80)
    u = (2 - r) ** 2
    du = -2 * (2 - r)
    sidecar = {"r": r, "delta_u": u[:, None], "delta_du_dr": du[:, None]}
    test_r = np.asarray((0.331, 1.139, 2, 2.4))
    value, derivative = evaluate_radial_correction(sidecar, test_r)
    expected = (2 - test_r[:2]) ** 2 / test_r[:2]
    expected_derivative = -2 * (2 - test_r[:2]) / test_r[:2] - (2 - test_r[:2]) ** 2 / test_r[:2] ** 2
    np.testing.assert_allclose(value[:2, 0], expected, atol=2e-13)
    np.testing.assert_allclose(derivative[:2, 0], expected_derivative, atol=2e-12)
    assert np.array_equal(value[2:], np.zeros((2, 1)))
    assert np.array_equal(derivative[2:], np.zeros((2, 1)))


def test_gradient_norm_keeps_the_finite_sphere_surface_term():
    r = np.geomspace(1e-4, 2, 513)
    w = log_radial_weights(r)
    # Constant R is constant in real space for l=0, although u'=1.
    assert radial_gradient_norm_squared(r, w, r, np.ones_like(r), 0) == 0
    # R=r has unit radial derivative. The independent analytic volume
    # integral tests the radial and angular pieces together for l=1.
    expected = (1 + 2) * (r[-1] ** 3 - r[0] ** 3) / 3
    actual = radial_gradient_norm_squared(r, w, r ** 2, 2 * r, 1)
    assert abs(actual - expected) / expected < 3e-7


def test_sidecar_rejects_a_different_source_and_altered_payload(tmp_path):
    source = tmp_path / "source.upf"
    source.write_text("original source")
    destination = tmp_path / "atomic.npz"
    arrays = {"r": np.asarray((0.1, 0.2)), "delta_u": np.asarray(((1,), (2,)))}
    metadata = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "operator_comparison": {"authenticated": True}, "phase_branch_validated": True}
    write_atomic_reconstruction(destination, arrays, metadata)
    load_atomic_reconstruction(destination, source)
    other = tmp_path / "other.upf"
    other.write_text("different source")
    with pytest.raises(ValueError, match="different pseudopotential"):
        load_atomic_reconstruction(destination, other)
    with np.load(destination, allow_pickle=False) as data:
        changed = {k: data[k] for k in data.files}
    changed["delta_u"] = changed["delta_u"] + 1
    altered = tmp_path / "altered.npz"
    np.savez(altered, **changed)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_atomic_reconstruction(altered, source)


def _operator_upf(path, beta, dij):
    n = beta.shape[1]
    sections = ''.join(f'<PP_BETA.{i+1}>{" ".join(map(str, beta[:, i]))}</PP_BETA.{i+1}>' for i in range(n))
    labels = ''.join(f'<PP_RELBETA.{i+1} index="{i+1}" lll="0" jjj="0.5"/>' for i in range(n))
    path.write_text(f'<UPF><PP_HEADER number_of_proj="{n}"/><PP_MESH>'
                    '<PP_R>0 0.1 0.2 0.3 0.4</PP_R></PP_MESH>'
                    '<PP_LOCAL>0 0 0 0 0</PP_LOCAL><PP_NONLOCAL>'
                    f'{sections}<PP_DIJ>{" ".join(map(str, dij.ravel()))}</PP_DIJ></PP_NONLOCAL>'
                    f'<PP_SPIN_ORB>{labels}</PP_SPIN_ORB></UPF>')


def test_operator_comparison_is_invariant_to_kb_rotations(tmp_path):
    beta = np.asarray(((0, 0), (1, 2), (2, -1), (0.5, 0.7), (0, 0)))
    dij = np.diag((2.0, -0.8))
    angle = 0.37
    rotation = np.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    source, rotated = tmp_path / "source.upf", tmp_path / "rotated.upf"
    _operator_upf(source, beta, dij)
    _operator_upf(rotated, beta @ rotation, rotation.T @ dij @ rotation)
    drift = radial_operator_difference(source, rotated)
    assert drift["hamiltonian_bound_ha"] < 1e-15


def test_incomplete_atomic_bank_and_oncv_failure_are_rejected(tmp_path):
    bank = tmp_path / "incomplete.dat"
    bank.write_text('# lorrax.atomic_scattering_bank.v1\n1 1 4 1 47\n1 0 -1 2 -1 5\n')
    with pytest.raises(ValueError, match="incomplete energy"):
        read_atomic_scattering_bank(bank)
    log = tmp_path / "failed.out"
    log.write_text('STOP ERROR phase branch failed\nPSP_UPF\n<UPF/>\nEND_PSP\n')
    with pytest.raises(ValueError, match="generation failed"):
        generated_upf_text(log)


def test_zero_boundary_residual_cannot_authenticate_the_wrong_core_branch():
    bank = {"r": np.asarray((0.1, 0.2, 0.5, 1.0)),
            "channel_l": np.asarray((0,)), "channel_kappa": np.asarray((-1,)),
            "ps_u": np.ones((1, 2, 4)), "node_matched": np.ones((1, 2), dtype=int),
            "boundary_phase_residual": np.zeros((1, 2))}
    core = {"core_kappa": np.asarray((-1,))}
    assert scattering_branch_diagnostics(bank, core)["validated"]
    bank["node_matched"][0, 1] = 2
    assert not scattering_branch_diagnostics(bank, core)["validated"]


def test_atom_without_frozen_core_has_a_zero_width_core_table(tmp_path):
    path = tmp_path / "core.dat"
    path.write_text('# lorrax.atomic_frozen_core.v1\n0 4\n')
    core = read_atomic_frozen_core(path, np.asarray((0.1, 0.2, 0.5, 1.0)))
    assert core["core_u"].shape == (4, 0)
    assert core["core_occupation"].shape == (0,)
