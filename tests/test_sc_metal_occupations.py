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
            occupation_clamp_tol=1.0e-12,
        ),
        # The class is DERIVED from the WFN occupations and threaded on the
        # inputs; it is no longer a deck key under config.mpa.
        material_class=material_class,
        parallel_transport=None,
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
    assert state.smearing_family == "mp1"
    np.testing.assert_allclose(
        np.mean(np.sum(np.asarray(state.f_kn), axis=1)),
        2.0,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_head_off_mpa_metal_has_no_surface_table():
    state, surface = _solve_head_occupations(
        _inputs("metal"), _energies())
    assert state is not None
    assert surface is None


def test_headless_insulator_keeps_the_step_occupation_path():
    state = _solve_occupation_state(_inputs("insulator"), _energies())
    assert state is None
