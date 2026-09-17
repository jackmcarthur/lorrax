"""Fixed-N occupation ownership for metallic self-consistency."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gw.sc_iteration import (
    _solve_head_occupations,
    _solve_occupation_state,
)


def _inputs(material_class: str):
    return SimpleNamespace(
        config=SimpleNamespace(
            # The direct head is off.  MPA metal body physics must still
            # receive the fixed-N occupation state.
            screening=SimpleNamespace(occ_broadening_ev=0.0),
            occ_broadening_ry=0.01,
            # A metal config always carries the parse-resolved family.
            occ_smearing_family=(
                "fd" if material_class == "metal" else None),
            occupation_clamp_tol=1.0e-12,
        ),
        # The class is DERIVED from the WFN occupations and threaded on the
        # inputs; it is no longer a deck key under config.mpa.
        material_class=material_class,
        parallel_transport=None,
        # Four logical bands on a carrier that callers may pad to the mesh.
        band_slices=SimpleNamespace(b0=0, b4_logical=4),
        wfn=SimpleNamespace(
            num_electrons=2.0,
            occupation_state_capacity=1.0,
        ),
    )


def _energies():
    return np.broadcast_to(
        np.asarray([-1.0, -0.1, 0.2, 1.0], dtype=np.float64),
        (3, 4),
    ).copy()


def test_head_off_mpa_metal_still_solves_fixed_n_occupations():
    state = _solve_occupation_state(_inputs("metal"), _energies())
    assert state is not None
    assert state.f_kn.shape == (3, 4)
    assert state.smearing_family == "fd"
    np.testing.assert_allclose(
        np.mean(np.sum(np.asarray(state.f_kn), axis=1)),
        2.0,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_a_metal_without_a_declared_family_is_never_solved_as_mp1():
    import pytest
    inputs = _inputs("metal")
    inputs.config.occ_smearing_family = None
    with pytest.raises(ValueError, match="GATE metal_occupations_fermi_dirac"):
        _solve_occupation_state(inputs, _energies())


def test_mesh_padding_rows_get_exact_zero_fermi_dirac_occupations():
    """P16 Na shared-pole SC (Run446, 58449999.22): the carrier was padded 86 -> 88
    and the solve covered the padding. MP1's clamp zeroed those rows; exact
    Fermi-Dirac left them tiny but nonzero, and the Sigma head provenance
    refused ("live occupations are nonzero outside logical_nband")."""
    from gw.efermi import legacy_square_mesh_occupation_digests
    energies = np.concatenate(
        [_energies(), np.full((3, 2), 1.0, dtype=np.float64)], axis=1)
    state = _solve_occupation_state(_inputs("metal"), energies)
    f = np.asarray(state.f_kn)
    assert f.shape == (3, 6)
    assert np.all(f[:, 4:] == 0.0)
    np.testing.assert_array_equal(
        f[:, :4], np.asarray(_solve_occupation_state(_inputs("metal"), _energies()).f_kn))
    legacy_square_mesh_occupation_digests(f, 4)          # must not raise
    # The instrument can see the defect: the whole-carrier solve refuses.
    from gw.efermi import OccupationState
    whole = OccupationState.solve_smearing(
        energies, np.full(3, 1.0 / 3.0), 2.0, 0.01, state_capacity=1.0,
        family="fd", clamp_tol=1.0e-12)
    assert np.any(np.asarray(whole.f_kn)[:, 4:] != 0.0)
    import pytest
    with pytest.raises(ValueError, match="nonzero outside logical_nband"):
        legacy_square_mesh_occupation_digests(whole.f_kn, 4)


def test_head_off_mpa_metal_has_no_surface_table():
    state, surface = _solve_head_occupations(
        _inputs("metal"), _energies())
    assert state is not None
    assert surface is None


def test_headless_insulator_keeps_the_step_occupation_path():
    state = _solve_occupation_state(_inputs("insulator"), _energies())
    assert state is None
