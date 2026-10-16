"""Validation ladder, rung 5: circular selection rules on a C3-symmetric exciton toy.

Physics.  On a k grid closed under C3 with an interaction and dipoles that
transform covariantly, every exciton carries a C3 eigenvalue (1, w, w^2) and
<0|x -+ i y|S> vanishes unless that eigenvalue matches: each bright state is
pure sigma+ or pure sigma- (|eta| = 1).  A sigma+ drive E0 g(t) (cos wt, sin wt, 0)
(campaign convention: x -> y, counter-clockwise from +z) couples at first order
through <S|x + iy|0>, i.e. I+ = |D_x - i D_y|^2/2 with D = <0|r|S>, so it fills the
sigma+ states and leaves every sigma- state empty — even one whose energy
the pulse spectrum covers.  What can leak is the pulse's own counter-rotating
component, (x - iy) g~(Omega + w): below 1e-20 for tau = 2 fs (at tau = 1 fs it
is 4e-7 and first-order theory reproduces it to rounding).

System: ``haldane`` — the Haldane honeycomb (t1 1 eV, t2 0.03 eV, phi = pi/2,
M 0.95 eV) on 6x6 k with the ISDF-form Hartree + SEX interaction on the two
sublattice orbitals (lattice sums truncated on a disk in the displacement, a
C3-invariant set).  Valley gaps 1.588 (K, sigma+) and 2.212 eV (K', sigma-);
lowest excitons S0 1.4754 eV (pure sigma+, I+ 0.447 bohr^2/cell) and S1
2.0885 eV (pure sigma-, I- 0.338).  Pulses (tau 2 fs, 1e6 V/m, +-E0 odd part)
centred on S0 and on S1, both helicities; populations from the propagated
density matrix projected on the full-BSE eigenvectors (``tools/td_response.py``).

G1 every bright BSE state is pure: min(I+, I-) <= 1e-13 max_S (I+ + I-) — the
   residue algebra's rounding, relative to the strongest state (the C3
   construction holds; |eta| - 1 reaches 3e-10 only on states 1e-6 as bright)
G2 a sigma+ pulse at S0 fills sigma+ states only: N(sigma-)/N(sigma+) <= 1e-12,
   N(S0) == first-order theory to 1e-5; a sigma- pulse at S0 the mirror image
G3 a sigma+ pulse centred on the sigma- state S1 leaves S1 empty (<= 1e-12 of
   the sigma+ population it creates), the sigma- pulse there fills it
R1 the reversed convention (I+ = |D_x + i D_y|^2/2) labels S1 sigma+: G2 fails
R2 the drive (cos wt, -sin wt) presented as sigma+: G2 fails
R3 dipoles with a 1e-3 x/y anisotropy (C3 broken in r only, as finite-difference
   dipoles on a coarse grid break it): G2's 1e-12 fails, N(sigma-)/N(sigma+) ~ 1e-7
"""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import _validation_systems as vs  # noqa: E402
from _validation_systems import record  # noqa: E402,F401  (fixture)

E0, TAU_AS, TC_AS, TF_AS = 1e6, 2000.0, 12000.0, 30000.0


def _run(tmp, tag, hdr, hm, kernel, drives):
    trials = []
    for pol, w in drives:
        trials += [vs.trial_for(E0, w, TC_AS, TAU_AS, pol),
                   vs.trial_for(-E0, w, TC_AS, TAU_AS, pol)]
    nt = int(round(TF_AS / 10.0))
    res, stats = vs.run_deck(os.path.join(tmp, tag), hdr, hm, trials=trials, tf_as=TF_AS,
                             dt_measure_as=10.0, method="dp54", rtol=1e-10, max_step_as=50.0,
                             kernel_file=kernel, measurements=("Efield", "density_matrix"),
                             meas_opts={"density_matrix": {"itrange": [0, nt, 50]}})
    drho = [0.5 * (res[2 * i]["density_matrix"][1] - res[2 * i + 1]["density_matrix"][1])
            for i in range(len(drives))]
    return drho, stats


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    tdr = vs.load_td_response()
    tmp = str(tmp_path_factory.mktemp("hel"))
    hm = vs.haldane()
    hdr = vs.write_header(os.path.join(tmp, "hdr"), hm)
    kern = vs.write_kernel(os.path.join(tmp, "kernel.h5"), hm)
    e = vs.energies_ha(hm)
    modes = tdr.window_modes(e, 1, hm["K_ha"], hm["r"], np.full(e.shape[0], 1.0 / e.shape[0]))
    pos = np.where(modes.lam.real > 0)[0]
    ip, im, eta = tdr.helicity(modes.R[pos])
    om = modes.lam.real[pos] * vs.HA_IN_EV
    drives = [("+", om[0]), ("-", om[0]), ("+", om[1]), ("-", om[1])]
    drho, stats = _run(tmp, "main", hdr, hm, kern, drives)
    pops = [np.abs(tdr.project_state(modes, d, 1)[-1, pos]) ** 2 for d in drho]
    return dict(tdr=tdr, tmp=tmp, hm=hm, kern=kern, modes=modes, pos=pos, ip=ip, im=im,
                eta=eta, om=om, drives=drives, pops=pops, stats=stats)


def _bright(c):
    tot = c["ip"] + c["im"]
    return tot > 1e-6 * tot.max()


def _selects(n, plus, minus):
    """sigma+ drive fills ``plus`` states only."""
    return n[minus].sum() <= 1e-12 * n[plus].sum() and n[plus].sum() > 0


def test_bright_states_are_pure_helicity(case, record):
    b = _bright(case)
    minor = float(np.minimum(case["ip"], case["im"])[b].max() / (case["ip"] + case["im"]).max())
    record("max_minor_helicity_over_max_strength", minor)
    assert b.sum() >= 10 and minor <= 1e-13
    assert case["eta"][0] > 0.999999999 and case["eta"][1] < -0.999999999   # S0 sigma+, S1 sigma-
    assert case["stats"]["mesh"] == (2, 2) and case["stats"]["nk_pad"] == 36


def test_sigma_plus_pulse_fills_sigma_plus_states_only(case, record):
    tdr, b = case["tdr"], _bright(case)
    plus, minus = b & (case["eta"] > 0), b & (case["eta"] < 0)
    n_p, n_m = case["pops"][0], case["pops"][1]
    record("sigma_minus_over_sigma_plus_under_sigma_plus", float(n_p[minus].sum() / n_p[plus].sum()))
    record("sigma_plus_over_sigma_minus_under_sigma_minus", float(n_m[plus].sum() / n_m[minus].sum()))
    assert _selects(n_p, plus, minus) and _selects(n_m, minus, plus)
    pt = np.abs(tdr.mode_amplitudes(case["modes"], vs.pulse_au(
        tdr, E0, case["om"][0], TC_AS, TAU_AS, "+"))[case["pos"]]) ** 2
    record("N_S0_vs_first_order_rel", float(abs(n_p[0] / pt[0] - 1)))
    assert abs(n_p[0] / pt[0] - 1) <= 1e-5


def test_sigma_plus_pulse_on_the_sigma_minus_state_leaves_it_empty(case):
    b = _bright(case)
    plus = b & (case["eta"] > 0)
    n_p, n_m = case["pops"][2], case["pops"][3]
    assert n_p[1] <= 1e-12 * n_p[plus].sum()
    assert n_m[1] > 0.5 * n_m[b].sum()           # the sigma- pulse there fills S1 first


def test_red_twin_reversed_convention(case):
    b = _bright(case)
    flipped_plus = b & ((case["im"] - case["ip"]) / (case["ip"] + case["im"]) > 0)
    flipped_minus = b & ~flipped_plus
    assert not _selects(case["pops"][0], flipped_plus, flipped_minus)


def test_red_twin_cos_minus_sin_presented_as_sigma_plus(case):
    b = _bright(case)
    plus, minus = b & (case["eta"] > 0), b & (case["eta"] < 0)
    assert not _selects(case["pops"][1], plus, minus)   # the (cos, -sin) run, read as sigma+


def test_red_twin_c3_broken_dipoles(case, record):
    hm = dict(case["hm"])
    r = hm["r"].copy()
    r[1] *= 1.0 + 1e-3
    hm["r"] = r
    hdr = vs.write_header(os.path.join(case["tmp"], "hdr_aniso"), hm)
    drho, _ = _run(case["tmp"], "aniso", hdr, hm, case["kern"], [("+", case["om"][0])])
    n = np.abs(case["tdr"].project_state(case["modes"], drho[0], 1)[-1, case["pos"]]) ** 2
    b = _bright(case)
    plus, minus = b & (case["eta"] > 0), b & (case["eta"] < 0)
    leak = float(n[minus].sum() / n[plus].sum())
    record("anisotropic_dipole_leak", leak)
    assert not _selects(n, plus, minus) and 1e-9 < leak < 1e-4, leak
