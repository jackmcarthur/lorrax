"""exciton_bands Gamma gate: report the whole overlap spectrum.

RED TWIN recorded in FIX_driver_blockers.md.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest


def _process_local_mesh_xy():
    """A 1x1 ``('x','y')`` mesh over THIS process's own device.

    These cells are index-map gates, not distribution gates: they need
    somewhere to put two small arrays.  ``collectives.single_device_mesh`` is
    the sanctioned process-local 1x1 and is safe at any device count, which
    ``create_mesh_xy(1, 1)`` deliberately is not since 2026-08-27 — it now
    refuses a shape that does not consume the job, so the previous spelling
    here (``exciton_bands._create_mesh_xy(1, 1)``, an import this driver no
    longer carries) would refuse inside a widened multi-GPU pytest process.
    """
    from common.collectives import single_device_mesh
    return single_device_mesh()


# ===========================================================================
# 3. the Gamma gate reports the whole overlap spectrum
# ===========================================================================
def test_gamma_gate_reports_the_overlap_spectrum():
    """``min-sval`` alone cannot separate a lost subspace direction from a
    uniform normalisation error; the spectrum can, and the gate must print it.

    RED TWIN: revert to tracking only ``smin`` in
    ``exciton_bands.gate_htransform_vs_stored`` and the ``overlap svals`` line
    disappears — which is the state in which ``min-sval = 0.4626`` was
    undiagnosable.  With the line, ``[0.9985 0.4626]`` reads immediately as one
    boundary direction lost to a cut degenerate multiplet.
    """
    import bse.exciton_bands as eb

    src = Path(eb.__file__).read_text(encoding="utf-8")
    assert "overlap svals at the worst k" in src
    assert "sv_worst" in src and "k_worst" in src


def test_gamma_gate_compares_only_matched_native_points_on_8_to_12():
    """Nonnested red twin: fine 144 and stored 64 must not be subtracted.

    Common rows carry identical energies/subspaces while every noncommon row
    is deliberately poisoned.  The preserved gate must pass on the exact
    4x4 intersection and announce 16 samples.  The pre-fix implementation
    instead subtracts ``(144, 2) - (64, 2)`` and raises before validation.
    """
    import jax.numpy as jnp
    from symmetry_maps import common_uniform_grid_indices
    import bse.exciton_bands as eb

    coarse_grid = (8, 8, 1)
    fine_grid = (12, 12, 1)
    stored_k, htransform_k = common_uniform_grid_indices(
        coarse_grid, fine_grid)
    assert stored_k.size == 16

    nc, ns, nmu = 2, 1, 2
    eps_stored = np.full((64, nc), -31.0)
    eps_stored[stored_k, 0] = np.linspace(0.10, 0.20, stored_k.size)
    eps_stored[stored_k, 1] = np.linspace(0.30, 0.40, stored_k.size)
    eps_htransform = np.full((144, nc), +47.0)
    eps_htransform[htransform_k] = eps_stored[stored_k]

    psi_stored = np.full((64, nc, ns, nmu), 19.0 + 7.0j)
    psi_htransform = np.full((144, nc, ns, nmu), -13.0 + 5.0j)
    native_basis = np.zeros((stored_k.size, nc, ns, nmu), np.complex128)
    native_basis[:, 0, 0, 0] = 1.0
    native_basis[:, 1, 0, 1] = 1.0
    psi_stored[stored_k] = native_basis
    psi_htransform[htransform_k] = native_basis

    data = {
        "n_cond": nc,
        "eps_c": jnp.asarray(eps_stored),
        "psi_c_X": jnp.asarray(psi_stored),
    }
    logs = []
    mesh = _process_local_mesh_xy()
    eb.gate_htransform_vs_stored(
        jnp.asarray(psi_htransform), jnp.asarray(eps_htransform), data, mesh,
        htransform_k_indices=htransform_k, stored_k_indices=stored_k,
        log=logs.append)
    assert any("16 matched k point(s)" in line for line in logs)
    assert any("max|Δε_c| = 0.000000 meV" in line for line in logs)


def test_gamma_gate_refuses_mismatched_native_tables_without_a_map():
    """A caller may neither regain the broadcast crash nor silently trim."""
    import jax.numpy as jnp
    import bse.exciton_bands as eb

    data = {
        "n_cond": 1,
        "eps_c": jnp.zeros((64, 1)),
        "psi_c_X": jnp.ones((64, 1, 1, 1), dtype=jnp.complex128),
    }
    with pytest.raises(ValueError, match="canonical common-grid index map"):
        eb.gate_htransform_vs_stored(
            jnp.ones((144, 1, 1, 1), dtype=jnp.complex128),
            jnp.zeros((144, 1)), data, _process_local_mesh_xy(),
            log=lambda _: None)


def test_gamma_gate_refuses_mixed_coarse_energy_and_fine_psi_axes():
    """RED TWIN: post-densification --eqp must not masquerade as a join.

    The real 8→12 failure had a 64-row energy table applied after the BSE
    loader had already rebuilt psi on 144 rows.  Even a valid coarse/fine
    point map cannot make those two arrays one physical operand.
    """
    import jax.numpy as jnp
    from symmetry_maps import common_uniform_grid_indices
    import bse.exciton_bands as eb

    stored_k, htransform_k = common_uniform_grid_indices(
        (8, 8, 1), (12, 12, 1))
    data = {
        "n_cond": 1,
        "eps_c": jnp.zeros((64, 1)),
        "psi_c_X": jnp.ones((144, 1, 1, 1), dtype=jnp.complex128),
    }
    with pytest.raises(ValueError, match="internally inconsistent BSE bundle"):
        eb.gate_htransform_vs_stored(
            jnp.ones((144, 1, 1, 1), dtype=jnp.complex128),
            jnp.zeros((144, 1)), data, _process_local_mesh_xy(),
            htransform_k_indices=htransform_k,
            stored_k_indices=stored_k, log=lambda _: None)
