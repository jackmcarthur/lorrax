"""Four-spinor and Lorentz representations on every Si SOC operation row."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from _deck_stub import read_deck
from symmetry_maps import (SymMaps, mix_lorentz_blocks,
                           mix_channels_by_proper_rotation,
                           spinor_rotation_for_sym_row)
from common.gamma_matrices import gamma_perm_phase


def test_four_spinor_action_on_all_si_soc_rows():
    """On antiunitary rows conjugate U†γU, since the current expectation uses Kψ."""
    sym = SymMaps(read_deck("si_bse_debug"))
    n = len(sym.sym_matrices)
    rows = np.arange(2 * n)
    u2 = sym.spinor_action(rows, nspinor=2)
    u4 = sym.spinor_action(rows, nspinor=4)
    parity = np.linalg.det(sym.R_cart[:n])[rows % n]
    assert n == 48
    assert np.any(parity < 0)
    np.testing.assert_array_equal(u4[:, :2, :2], u2)
    np.testing.assert_array_equal(u4[:, 2:, 2:], parity[:, None, None] * u2)
    np.testing.assert_array_equal(u4[:, :2, 2:], 0)
    np.testing.assert_allclose(u4.conj().transpose(0, 2, 1) @ u4,
                               np.broadcast_to(np.eye(4), u4.shape), atol=2e-14)
    gamma = []
    for i in range(4):
        perm, phase = gamma_perm_phase(i)
        gamma.append(np.asarray(phase)[:, None] * np.eye(4)[np.asarray(perm)])
    gamma = np.asarray(gamma)
    expected = np.einsum('qij,jab->qiab', sym.lorentz_action(rows), gamma)
    actual = np.einsum('qba,ibc,qcd->qiad', u4.conj(), gamma, u4)
    actual[n:] = actual[n:].conj()
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    wrong = np.zeros_like(u4)
    wrong[:, :2, :2] = wrong[:, 2:, 2:] = u2
    wrong_current = np.einsum('qba,ibc,qcd->qiad', wrong.conj(), gamma, wrong)
    wrong_current[n:] = wrong_current[n:].conj()
    assert np.max(np.abs(wrong_current - expected)) > 1


def test_four_spinor_requires_spatial_parity():
    with pytest.raises(ValueError, match="requires R_cart"):
        spinor_rotation_for_sym_row(np.eye(2)[None], 0, 1, nspinor=4)


def test_lorentz_mixer_tt_identity_and_rectangular_ct():
    sym = SymMaps(read_deck("si_bse_debug"))
    rows = np.arange(2 * len(sym.sym_matrices))
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x', 'y'))
    rng = np.random.default_rng(71)
    tt = {(a, b): jnp.asarray(rng.normal(size=(len(rows), 4, 4)))
          for a in (1, 2, 3) for b in (1, 2, 3)}
    old = mix_channels_by_proper_rotation(tt, sym=sym, sym_idx=rows, mesh_xy=mesh)
    new = mix_lorentz_blocks(tt, sym=sym, sym_idx=rows, mesh_xy=mesh)
    for key in tt:
        np.testing.assert_array_equal(new[key], old[key])
    ct = jnp.asarray(rng.normal(size=(len(rows), 6, 4)))
    mixed = mix_lorentz_blocks({(0, 2): ct}, sym=sym, sym_idx=rows, mesh_xy=mesh)
    action = sym.cartesian_action(rows, axial=False, time_odd=True)
    for j in (1, 2, 3):
        np.testing.assert_allclose(mixed[0, j], action[:, j-1, 1, None, None] * ct,
                                   atol=1e-14)


def test_selected_lorentz_output_matches_complete_sector():
    """Selecting one rectangular sector output avoids allocating its sibling blocks."""
    sym = SymMaps(read_deck("si_bse_debug"))
    rows = np.arange(96)
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x', 'y'))
    values = jnp.arange(96 * 6 * 4, dtype=jnp.float64).reshape(96, 6, 4)
    blocks = {(0, 1): values, (0, 2): values * 2, (0, 3): values * -3}
    complete = mix_lorentz_blocks(blocks, sym=sym, sym_idx=rows, mesh_xy=mesh)
    selected = mix_lorentz_blocks(blocks, sym=sym, sym_idx=rows,
                                  mesh_xy=mesh, keys=((0, 2), (1, 1)))
    assert set(selected) == {(0, 2)}
    np.testing.assert_allclose(selected[0, 2], complete[0, 2], atol=1e-12)
