"""The production line sites follow the band structure (the support rule, report section IV.B).

Against an independent evaluation of the same rule on a random insulator: delivered energies E
(levels within 5 eV of mu at every 0.25 eV offset of +/-5 eV) paired with every level eps at every
k strictly between mu and E by explicit loops, crossings above the height kept, the density summed
without binning, quantiles of rho**alpha on [omega_lo, omega_reach] with the omega_lo fixed point.
Sites agree to 0.02 eV (0.01 eV distance bins, reach/4000 grid), strictly increase, start at or
above the height and end at the reach. RED TWIN: the evenly spaced ladder on the same interval
misses by more than 0.1 eV. With no delivered level the density is flat on [omega_lo, top].
"""
import math

import numpy as np


def _reference(energies, mu, eta, height, count, alpha, window, step):
    levels = energies.ravel() - mu
    offsets = np.arange(-window, window + step / 2, step)
    cross = []
    for level in levels[np.abs(levels) <= window]:
        for d in offsets:
            e = level + d
            for eps in levels:
                if (e > 0 and 0 < eps < e) or (e < 0 and e < eps < 0):
                    cross.append(abs(e - eps))
    cross = np.asarray(cross)
    cross = cross[cross > height]
    reach = cross.max()
    x = np.linspace(0.0, reach, 4001)
    rho = np.array([eta / math.pi * np.sum(1.0 / ((v - cross) ** 2 + eta ** 2)) for v in x]) ** alpha
    lo = height
    for _ in range(200):
        k = x >= lo - 1e-12
        c = np.concatenate([[0.0], np.cumsum(np.diff(x[k]) * 0.5 * (rho[k][1:] + rho[k][:-1]))])
        sites = np.interp(np.linspace(0.0, c[-1], count), c, x[k])
        lo, previous = max(height, sites[1] - sites[0]), lo
        if abs(lo - previous) < 1e-10:
            break
    return sites, reach


def test_line_sites_follow_the_crossing_density():
    from gw.shared_pole_recipe import shared_real_pole_v1_r3b as recipe, support_rule_line_sites
    rng = np.random.default_rng(1407)
    valence = -np.sort(rng.gamma(2.0, 1.5, size=(3, 4)), axis=1)[:, ::-1] - 0.6
    conduction = np.sort(rng.gamma(2.0, 2.0, size=(3, 5)), axis=1) + 0.6
    energies = np.concatenate((valence, conduction), axis=1) + 3.0
    eta, height, count = 0.25, 1.0, 14
    got = support_rule_line_sites(energies, 3.0, eta, height, 30.0, count)
    want, reach = _reference(energies, 3.0, eta, height, count, recipe["support_density_power"],
                             recipe["support_delivery_window_ev"], recipe["support_offset_step_ev"])
    assert np.max(np.abs(got - want)) < 0.02, np.max(np.abs(got - want))
    assert np.all(np.diff(got) > 0) and got[0] >= height and abs(got[-1] - reach) < 0.011
    even = np.linspace(want[0], reach, count)
    assert np.max(np.abs(even - want)) > 0.1
    flat = support_rule_line_sites(np.array([[-7.0, 7.0]]), 0.0, eta, height, 20.0, 15)
    np.testing.assert_allclose(flat, np.linspace(20.0 / 15, 20.0, 15), atol=20.0 / 4000)
