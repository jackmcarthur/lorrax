"""Conditioned C(omega) is the shared C(E_DFT)/Z assembly owner."""
from __future__ import annotations

import numpy as np
import pytest

from gw.degen_average import average_within_degenerate_sets
from gw.eqp_bgw import assemble_eqp


def _assemble(curve):
    omega = np.array([-1.0, 0.0, 1.0])
    return assemble_eqp(
        kpoints_irr_frac=np.zeros((1, 3)),
        band_offset=0,
        e_dft_ev=np.zeros((1, 2)),
        kin_ion_diag_ev=np.zeros((1, 2)),
        hartree_diag_ev=np.full((1, 2), 11.0),
        sigma_x_diag_ev=np.full((1, 2), -3.0),
        sigma_c_omega_diag_ev=curve,
        omega_rel_ev=omega,
        e_dft_rel_ev=np.zeros((1, 2)),
        hartree_source="stored",
        hartree_already_resolved=True,
        mean_field_gate=False,
        print_fn=lambda *_: None,
    )


def test_degenerate_conditioning_averages_intercepts_and_unequal_slopes():
    omega = np.array([-1.0, 0.0, 1.0])
    intercept = np.array([[1.0, 3.0]])
    unequal_slopes = np.array([[0.2, 0.4]])
    raw_curve = intercept[None, :, :] + omega[:, None, None] * unequal_slopes

    conditioned_curve = average_within_degenerate_sets(
        raw_curve,
        energies_kn_ry=np.zeros((1, 2)),
        tol_ry=1.0e-6,
    )
    expected_curve = (
        np.full((1, 2), 2.0)[None, :, :]
        + omega[:, None, None] * np.full((1, 2), 0.3)
    )
    np.testing.assert_allclose(conditioned_curve, expected_curve)

    raw = _assemble(raw_curve)
    conditioned = _assemble(conditioned_curve)
    np.testing.assert_allclose(raw.eqp0_ev, [[9.0, 11.0]])
    np.testing.assert_allclose(conditioned.eqp0_ev, [[10.0, 10.0]])
    np.testing.assert_allclose(conditioned.sigma_c_at_dft_diag_ev, [[2.0, 2.0]])

    # Both the value and derivative came from the SAME averaged curve: the
    # unequal raw slopes become their group mean, so Z is equal too.
    expected_z = np.full((1, 2), 1.0 / (1.0 - 0.3))
    np.testing.assert_allclose(conditioned.z_factor, expected_z)
    np.testing.assert_allclose(conditioned.eqp1_ev, expected_z * 10.0)


def test_eqp_g0w0_combines_after_conditioning_with_disabled_red_twin():
    """Unequal X/C diagonals collapse only under the enabled policy.

    This pins the physical discriminator without introducing a second
    grouping routine.
    """
    energies = np.zeros((1, 2))
    raw_xc_components = np.array([
        [[-5.0, -1.0]],
        [[1.0, 3.0]],
    ])
    conditioned = average_within_degenerate_sets(
        raw_xc_components, energies_kn_ry=energies, tol_ry=1.0e-6)
    np.testing.assert_allclose(conditioned.sum(axis=0), [[-1.0, -1.0]])
    # no_degen_averaging=true: no helper call, hence the unequal raw twin.
    np.testing.assert_allclose(raw_xc_components.sum(axis=0), [[-4.0, 2.0]])

def test_dynamic_assembly_refuses_a_second_c_at_dft_producer():
    curve = np.zeros((3, 1, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="duplicate the correlation producer"):
        assemble_eqp(
            kpoints_irr_frac=np.zeros((1, 3)),
            band_offset=0,
            e_dft_ev=np.zeros((1, 2)),
            kin_ion_diag_ev=np.zeros((1, 2)),
            hartree_diag_ev=np.zeros((1, 2)),
            sigma_x_diag_ev=np.zeros((1, 2)),
            sigma_c_at_dft_diag_ev=np.ones((1, 2)),
            sigma_c_omega_diag_ev=curve,
            omega_rel_ev=np.array([-1.0, 0.0, 1.0]),
            e_dft_rel_ev=np.zeros((1, 2)),
            hartree_already_resolved=True,
            mean_field_gate=False,
            print_fn=lambda *_: None,
        )
