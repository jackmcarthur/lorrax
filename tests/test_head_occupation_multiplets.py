"""The SC head occupations must be one value per exactly degenerate multiplet.

Planted case (SCMETAL 2026-09-17): on a metal the frozen DFT head response
carried step occupations by band index.  Where the index cut runs through an
exact multiplet, the static head gives its members different Sigma_x head
values, which broke the little group and inversion times time reversal of the
next map (Na bcc Fermi-Dirac: 1.3207 eV splits at three k, bank reciprocity
1e-13 -> 1e-4, shared_pole_gram_valid refused at q=0).
"""
import numpy as np
import pytest

from gw.efermi import fd_occupations
from gw.head_correction import (
    HEAD_MULTIPLET_OCCUPATION_TOL,
    compute_static_head_terms,
    refuse_split_multiplet_head_occupations,
)

KT_RY = 0.01


def _ladder():
    """Three k, six bands; k=1 has an exact doublet at bands 3,4 (1-based) on E_F."""
    e = np.array([[-1.0, -0.40, -0.10, 0.30, 0.60, 0.90],
                  [-1.0, -0.40, 0.05, 0.05, 0.60, 0.90],
                  [-1.0, -0.50, -0.20, 0.40, 0.70, 0.95]])
    return e, 0.05


def _index_step(e, n_occ):
    occ = np.zeros_like(e)
    occ[:, :n_occ] = 1.0
    return occ


def test_index_step_through_a_multiplet_refuses_and_names_it():
    e, _ = _ladder()
    with pytest.raises(ValueError, match="GATE head_occupations_split_multiplet") as err:
        refuse_split_multiplet_head_occupations(e, _index_step(e, 3), where="planted")
    assert "k=1, bands 3 and 4" in str(err.value)


def test_the_planted_step_splits_the_static_head_inside_the_multiplet():
    """The consequence the gate protects: Sigma_x head differs inside the doublet."""
    e, _ = _ladder()
    step = compute_static_head_terms(vc0=6000.0, wcoul0_static=10.0, occ=_index_step(e, 3),
                                     cell_volume=254.0, nk_tot=512)
    x = np.real(np.asarray(step.sigma_x_diag))
    assert abs(x[1, 2] - x[1, 3]) > 1.0e-2
    fd = compute_static_head_terms(vc0=6000.0, wcoul0_static=10.0,
                                   occ=np.asarray(fd_occupations(e, 0.05, KT_RY)),
                                   cell_volume=254.0, nk_tot=512)
    x = np.real(np.asarray(fd.sigma_x_diag))
    assert x[1, 2] == x[1, 3]


def test_fermi_dirac_state_passes():
    e, mu = _ladder()
    f = np.asarray(fd_occupations(e, mu, KT_RY))
    refuse_split_multiplet_head_occupations(e, f, where="fd")


def test_fermi_dirac_inside_the_degeneracy_tolerance_passes():
    e, mu = _ladder()
    e = e.copy()
    e[1, 3] = e[1, 2] + 0.9e-9      # inside the tolerance, on E_F
    f = np.asarray(fd_occupations(e, mu, KT_RY))
    assert abs(f[1, 2] - f[1, 3]) < HEAD_MULTIPLET_OCCUPATION_TOL
    refuse_split_multiplet_head_occupations(e, f, where="fd-near")


def test_insulator_step_at_a_gap_passes():
    e = np.array([[-1.0, -0.5, -0.5, 0.4, 0.4, 0.9],
                  [-1.1, -0.6, -0.6, 0.5, 0.5, 1.0]])
    refuse_split_multiplet_head_occupations(e, (e < 0).astype(float), where="gap")


def test_resolved_pair_is_not_a_multiplet():
    e, _ = _ladder()
    e = e.copy()
    e[1, 3] = e[1, 2] + 1.0e-6      # resolved: not one multiplet
    refuse_split_multiplet_head_occupations(e, _index_step(e, 3), where="resolved")


def test_shape_mismatch_refuses():
    e, _ = _ladder()
    with pytest.raises(ValueError, match="GATE head_occupations_split_multiplet"):
        refuse_split_multiplet_head_occupations(e, _index_step(e, 3)[:, :5], where="shape")
