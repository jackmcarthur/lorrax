"""Bank-owner partition: remote Laplace cells keep the Taylor ratio capped.

``gw.response_bank.response_windows`` derives its stream/cell edges from
the bank's own samples and keeps the campaign edges [-35, 40] eV as a
floor. Three planted band tables (Ry, [k, band]) check the three regimes:
the floor binds (partition identical to the incumbent fixed-edge rule),
a deep-semicore metal whose nearest semicore cell must join the stream,
and an upper remote window pulled into the stream. CPU only.
"""
import numpy as np
import pytest

import minimax
from gw.response_bank import REMOTE_RHO_MAX, response_windows

RY_EV = 13.605693122994


def _incumbent_masks(energy, f, u, mu):
    """The fixed-edge partition as it stood on main c4cddf83 (the oracle)."""
    ft = np.where(np.abs(f) >= 1e-14, f, 0.0)
    ut = np.where(np.abs(u) >= 1e-14, u, 0.0)
    physical = (f != 0) | (u != 0)
    ev = (energy - mu) * RY_EV
    masks = [physical & (ev < -35.0),
             physical & (ev >= -35.0) & (ev <= 40.0),
             physical & (ev > 40.0)]
    for idx in (0, 2):
        if np.any(ft * masks[idx]) and np.any(ut * masks[idx]):
            masks[1] |= masks[idx]
            masks[idx] = np.zeros_like(masks[idx])
    return masks


def _table(levels_ev, mu_ev, width_ry):
    """Two k rows of Fermi-Dirac-occupied levels about mu (eV in, Ry out)."""
    e = np.sort(np.asarray(levels_ev, dtype=np.float64)) / RY_EV
    energy = np.stack([e, e + 0.05 / RY_EV])
    mu = mu_ev / RY_EV
    if width_ry == 0:
        f = (energy < mu).astype(np.float64)
    else:
        f = 0.5 * (1.0 - np.tanh((energy - mu) / (2.0 * width_ry)))
    return energy, f, 1.0 - f, mu


def _samples(top_ev, eta_ev=1.0):
    """The recipe's shape: a line at height eta plus an imaginary ladder."""
    line = np.r_[np.arange(0.0, min(12.0, top_ev), 0.5),
                 np.arange(12.0, top_ev, 1.0), top_ev] + 1j * eta_ev
    return np.r_[line, 1j * np.geomspace(eta_ev, max(16.0, top_ev), 7)] / RY_EV


def _rho(cell, z):
    eta = z.imag.min()
    return np.abs(z * z + eta * eta).max() / (cell["delta_min_ry"]**2 + eta * eta)


def test_floor_binds_partition_is_the_incumbent():
    # Insulator: semicore -60 eV, valence to 0, conduction 2..30, upper 55..70.
    levels = [-60.0, -59.5, *np.linspace(-12.0, 0.0, 6),
              *np.linspace(2.0, 30.0, 8), 55.0, 70.0]
    energy, f, u, mu = _table(levels, 1.0, 0)
    z = _samples(16.0)
    masks, _, _, cells, receipt = response_windows(
        energy, f, u, chemical_potential_ry=mu, z_ry=z)
    for got, want in zip(masks, _incumbent_masks(energy, f, u, mu)):
        assert np.array_equal(got, want)
    assert receipt["window_ev_relative_mu"] == [-35.0, 40.0]
    assert receipt["edge_rule"]["derived_edge_binds"] is False
    assert {(c["lower"], c["upper"]) for c in cells} == {(0, 1), (0, 2), (1, 2)}
    assert all(_rho(c, z) <= REMOTE_RHO_MAX for c in cells)


def test_semicore_metal_nearest_cell_joins_stream():
    # Fe-like: 3s at -88 eV, 3p at -51 eV, d/s band -8..+38 eV, FD 0.02 Ry.
    levels = [-88.0, -87.6, -51.2, -50.8, -50.5,
              *np.linspace(-8.0, 38.0, 24)]
    energy, f, u, mu = _table(levels, 0.0, 0.02)
    z = _samples(34.0)
    incumbent = _incumbent_masks(energy, f, u, mu)
    # Negative control: the fixed edges leave 3p remote at rho > cap.
    semicore_3p = np.abs(energy * RY_EV + 51.0) < 1.0
    assert np.all(incumbent[0][semicore_3p])
    masks, ft, ut, cells, receipt = response_windows(
        energy, f, u, chemical_potential_ry=mu, z_ry=z)
    assert receipt["edge_rule"]["derived_edge_binds"] is True
    assert np.all(masks[1][semicore_3p])
    assert np.all(masks[0][energy * RY_EV < -80.0])
    # Floor: nothing the incumbent put in the stream leaves it.
    assert np.all(masks[1][incumbent[1]])
    # Partition: every physical state has exactly one owner.
    owners = masks[0].astype(int) + masks[1] + masks[2]
    assert np.array_equal(owners, ((f != 0) | (u != 0)).astype(int))
    assert len(cells) == 1 and (cells[0]["lower"], cells[0]["upper"]) == (0, 1)
    assert _rho(cells[0], z) <= REMOTE_RHO_MAX
    rule = minimax.response_laplace_rule(
        cells[0]["delta_min_ry"], cells[0]["delta_max_ry"], z, rel_tol=1e-8)
    assert rule["certificate"]["status"] == "PASS"


def test_upper_remote_cell_joins_stream():
    levels = [*np.linspace(-10.0, 0.0, 5), *np.linspace(1.0, 30.0, 10),
              41.0, 44.0, 90.0, 95.0]
    energy, f, u, mu = _table(levels, 0.5, 0)
    z = _samples(30.0)
    masks, _, _, cells, receipt = response_windows(
        energy, f, u, chemical_potential_ry=mu, z_ry=z)
    upper_edge = receipt["window_ev_relative_mu"][1]
    assert upper_edge > 40.0 and receipt["window_ev_relative_mu"][0] == -35.0
    assert np.all(masks[1][np.abs(energy * RY_EV - 42.5) < 2.0])
    assert all(_rho(c, z) <= REMOTE_RHO_MAX for c in cells)


def test_cap_is_the_documented_value():
    assert REMOTE_RHO_MAX == pytest.approx(0.3)
