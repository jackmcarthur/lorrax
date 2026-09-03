"""Window-edge evaluation rules for the QSGW Hermitianization."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from gw.qsgw_utils import QSGWUpdatePolicy, build_qsgw_sigma_xc


def test_one_sided_cross_edge_uses_core_energy():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    # At omega=0 the cross coupling is 2; at omega=1 it is 6.  The ordinary
    # QSGW half-sum is 4, whereas the one-sided rule must use the core
    # state's omega=0 value for both Hermitian partners.
    sigma = np.zeros((2, 1, 2, 2), dtype=np.complex128)
    sigma[0, 0, 0, 1] = sigma[0, 0, 1, 0] = 2.0
    sigma[1, 0, 0, 1] = sigma[1, 0, 1, 0] = 6.0
    sigma = jnp.asarray(sigma)
    sigma_x = jnp.zeros((1, 2, 2), dtype=jnp.complex128)
    energies = np.asarray([[0.0, 1.0]])

    standard, standard_diag = build_qsgw_sigma_xc(
        sigma, sigma_x, np.asarray([0.0, 1.0]), energies, mesh)
    one_sided, edge_diag = build_qsgw_sigma_xc(
        sigma, sigma_x, np.asarray([0.0, 1.0]), energies, mesh,
        one_sided_core_mask=np.asarray([True, False]))

    np.testing.assert_allclose(np.asarray(standard)[0, 0, 1], 4.0)
    np.testing.assert_allclose(np.asarray(one_sided)[0, 0, 1], 2.0)
    np.testing.assert_allclose(np.asarray(one_sided)[0, 1, 0], 2.0)
    assert standard_diag["n_one_sided_edges"] == 0.0
    assert edge_diag["n_one_sided_edges"] == 2.0


def test_one_sided_mask_requires_core_and_buffer():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    sigma = jnp.zeros((2, 1, 2, 2), dtype=jnp.complex128)
    sigma_x = jnp.zeros((1, 2, 2), dtype=jnp.complex128)
    with np.testing.assert_raises_regex(ValueError, "nonempty core and buffer"):
        build_qsgw_sigma_xc(
            sigma, sigma_x, np.asarray([0.0, 1.0]),
            np.asarray([[0.0, 1.0]]), mesh,
            one_sided_core_mask=np.asarray([True, True]))


def test_sc_update_uses_fermi_offdiagonal_and_on_shell_diagonal():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    sigma = np.zeros((3, 1, 2, 2), dtype=np.complex128)
    # Off-diagonal values distinguish E_0=-1, E_F=0, and E_1=+1.
    sigma[:, 0, 0, 1] = sigma[:, 0, 1, 0] = (1.0, 5.0, 9.0)
    # Diagonals must still follow their own on-shell energies.
    sigma[:, 0, 0, 0] = (11.0, 12.0, 13.0)
    sigma[:, 0, 1, 1] = (21.0, 22.0, 23.0)

    result, diagnostics = build_qsgw_sigma_xc(
        jnp.asarray(sigma),
        jnp.zeros((1, 2, 2), dtype=jnp.complex128),
        np.asarray([-1.0, 0.0, 1.0]),
        np.asarray([[-1.0, 1.0]]), mesh,
        update_policy=(
            QSGWUpdatePolicy.DIAGONAL_ON_SHELL_OFFDIAGONAL_FERMI))

    np.testing.assert_allclose(
        np.asarray(result)[0], np.asarray([[11.0, 5.0], [5.0, 23.0]]))
    assert diagnostics["offdiagonal_efermi_ev"] == 0.0
    assert diagnostics["update_policy"] == (
        QSGWUpdatePolicy.DIAGONAL_ON_SHELL_OFFDIAGONAL_FERMI.value)
    assert diagnostics["efermi_was_clipped"] == 0.0


def test_typed_half_sum_is_exactly_the_default_path():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    rng = np.random.default_rng(91)
    sigma = (rng.normal(size=(3, 1, 2, 2))
             + 1j * rng.normal(size=(3, 1, 2, 2)))
    sigma_x = (rng.normal(size=(1, 2, 2))
               + 1j * rng.normal(size=(1, 2, 2)))
    args = (jnp.asarray(sigma), jnp.asarray(sigma_x),
            np.asarray([-1.0, 0.0, 1.0]), np.asarray([[-0.7, 0.6]]), mesh)

    default, default_diagnostics = build_qsgw_sigma_xc(*args)
    explicit, explicit_diagnostics = build_qsgw_sigma_xc(
        *args, update_policy=QSGWUpdatePolicy.HALF_SUM)

    assert np.array_equal(np.asarray(default), np.asarray(explicit))
    assert default_diagnostics.keys() == explicit_diagnostics.keys()
    for key in default_diagnostics:
        left, right = default_diagnostics[key], explicit_diagnostics[key]
        assert left == right or (
            isinstance(left, float) and isinstance(right, float)
            and np.isnan(left) and np.isnan(right))


def test_sc_fermi_law_refuses_the_legacy_one_sided_edge_law():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    with np.testing.assert_raises_regex(ValueError, "two different"):
        build_qsgw_sigma_xc(
            jnp.zeros((2, 1, 2, 2), dtype=jnp.complex128),
            jnp.zeros((1, 2, 2), dtype=jnp.complex128),
            np.asarray([0.0, 1.0]), np.asarray([[0.0, 1.0]]), mesh,
            one_sided_core_mask=np.asarray([True, False]),
            update_policy=(
                QSGWUpdatePolicy.DIAGONAL_ON_SHELL_OFFDIAGONAL_FERMI))


def test_qsgw_update_policy_refuses_unknown_value():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    with np.testing.assert_raises_regex(ValueError, "unknown QSGW update"):
        build_qsgw_sigma_xc(
            jnp.zeros((2, 1, 1, 1), dtype=jnp.complex128),
            jnp.zeros((1, 1, 1), dtype=jnp.complex128),
            np.asarray([0.0, 1.0]), np.asarray([[0.0]]), mesh,
            update_policy="typo")
