"""Focused contract for conditioned C(E_DFT) versus the raw omega cube."""
from __future__ import annotations

import numpy as np

from gw.eqp_bgw import assemble_eqp


def test_conditioned_c_owns_newton_numerator_while_raw_cube_owns_z():
    omega = np.array([-1.0, 0.0, 1.0])
    c_raw_at_dft = np.array([[1.0, 3.0]])
    slopes = np.array([[0.2, 0.4]])
    c_cube = c_raw_at_dft[None, :, :] + omega[:, None, None] * slopes
    common = dict(
        kpoints_irr_frac=np.zeros((1, 3)),
        band_offset=0,
        e_dft_ev=np.zeros((1, 2)),
        kin_ion_diag_ev=np.zeros((1, 2)),
        hartree_diag_ev=np.full((1, 2), 11.0),
        sigma_x_diag_ev=np.full((1, 2), -3.0),
        sigma_c_omega_diag_ev=c_cube,
        omega_rel_ev=omega,
        e_dft_rel_ev=np.zeros((1, 2)),
        hartree_source="stored",
        hartree_already_resolved=True,
        mean_field_gate=False,
        print_fn=lambda *_: None,
    )

    legacy_raw = assemble_eqp(**common)
    conditioned = assemble_eqp(
        **common,
        sigma_c_at_dft_diag_ev=np.full((1, 2), 2.0),
    )

    # C(E_DFT) in the Newton numerator is the conditioned live operand.
    assert np.allclose(legacy_raw.eqp0_ev, [[9.0, 11.0]])
    assert np.allclose(conditioned.eqp0_ev, [[10.0, 10.0]])
    assert np.allclose(conditioned.sigma_c_at_dft_diag_ev, [[2.0, 2.0]])

    # Its Z still comes only from dC_raw(omega)/domega: 1 / (1 - slope).
    expected_z = 1.0 / (1.0 - slopes)
    assert np.allclose(conditioned.z_factor, expected_z)
    assert np.allclose(conditioned.eqp1_ev, expected_z * 10.0)
