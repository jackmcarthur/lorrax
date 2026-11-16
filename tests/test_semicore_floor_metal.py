"""The coarse (semicore) floor of a metal counts nval down from the occupied-band boundary.

Fe-like ladder: 3s (2 bands), 3p split 2 + 4 by a 0.7 eV spin-orbit gap, ten
valence bands whose count below mu varies with k, eight conduction bands.  With
the doc rule nval = 10 the floor must sit under the QP window's lowest valence
band (band nelec - nval + 1 = 9), so the whole 3s/3p ladder is one class; the
per-k count below mu used before put it between 3p1/2 and 3p3/2
(runs/Fe/04_*/160_sc_rootcause_20261111).
"""
from types import SimpleNamespace

import numpy as np

from common.units import RYD_TO_EV
from gw.band_partition import semicore_floor
from gw.efermi import OCCUPATION_CLAMP_TOL_DEFAULT
from gw.gw_init import coarse_class_for_deck


def _fe_like(nk=8):
    k = np.arange(nk)
    semicore = np.array([-5.20, -5.10, -2.70, -2.68, -2.63, -2.61, -2.60, -2.58])
    e = np.empty((nk, 26))
    e[:, :8] = semicore[None, :] + 0.002 * np.cos(2 * np.pi * k / nk)[:, None]
    disp = 0.25 * np.cos(2 * np.pi * k / nk)[:, None]
    e[:, 8:18] = 1.30 + 0.06 * (np.arange(10)[None, :] - 6.0) + disp
    e[:, 18:26] = 2.2 + 0.1 * np.arange(8)[None, :] + 0.5 * disp
    e.sort(axis=1)
    occ = 1.0 / (1.0 + np.exp((e - 1.32) / 0.02))
    occ[:, :8] = 1.0
    occ[:, 18:] = 0.0
    wfn = SimpleNamespace(energies=e[None], nelec=18, occs=occ[None], kweights=np.ones(nk),
                          num_electrons=16.0, occupation_state_capacity=1.0)
    config = SimpleNamespace(nband=26, nval=10, ncond=8, occ_broadening_ry=0.02,
                             occ_smearing_family="fd", occupation_clamp_tol=OCCUPATION_CLAMP_TOL_DEFAULT,
                             sigma=SimpleNamespace(omega_min_ev=None), qp_solver="self_consistent")
    return wfn, config


def test_metal_floor_sits_under_the_window_valence_bottom():
    wfn, config = _fe_like()
    e_ev = wfn.energies[0] * RYD_TO_EV
    coarse = coarse_class_for_deck(config, wfn, print)
    assert coarse.n_coarse == 8 * e_ev.shape[0]            # every 3s/3p state, at every k
    assert e_ev[:, :8].max() < coarse.coarse_floor_ev <= e_ev[:, 8].min() + 1e-9


def test_per_k_count_below_mu_would_split_the_3p_shell():
    # The discriminating precondition: the old per-k reference cuts the 3p shell here.
    wfn, config = _fe_like()
    e_ev = wfn.energies[0] * RYD_TO_EV
    below = np.count_nonzero(e_ev < 1.32 * RYD_TO_EV, axis=1)
    assert below.min() < wfn.nelec
    old = semicore_floor(e_ev, n_below_k=below, nval=10, mu_ev=1.32 * RYD_TO_EV, clip_ev=10.0)
    assert 0 < old.n_coarse < 8 * e_ev.shape[0]
