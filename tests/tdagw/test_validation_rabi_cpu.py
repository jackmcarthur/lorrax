"""Validation ladder, rung 1: Rabi flopping of a two-level system, exact, every integrator.

Physics.  A circular drive on a circular transition has no counter-rotating
term: with <c|r|v> = d (1, -i, 0)/sqrt2 and sigma+ = E0 (cos wt, sin wt, 0),
H_cv = g e^{-i w t}, g = E0 d/sqrt2, and T = diag(1, e^{-i w t}) makes the
Hamiltonian time independent, so rho(t) is one matrix exponential with or
without relaxation (TDGW's -Gamma o (rho - rho_eq) commutes with T): Rabi
flopping rho_cc = (4g^2/W^2) sin^2(W t/2), W^2 = delta^2 + 4 g^2, exact
beyond the rotating-wave approximation.  The sigma- drive of the same
transition is the counter-rotating problem (detuning gap + w), exact too.  A
resonant Gaussian pi pulse obeys the area theorem rho_cc = sin^2(int g dt);
detuned pulses and pulses with relaxation are referred to a SciPy DOP853
integration of the same Bloch equations at rtol 1e-13 (no shared code; it
reproduces the area theorem to 2e-14).

System: five copies (k points) of the transition, gaps 1.95-2.06 eV, d = 1 bohr,
E0 = 2e9 V/m (Rabi period 27.6 fs at resonance) and a 3.9e9 V/m pi pulse
(tau 4 fs), 48 fs, through the whole driver path: header on disk -> deck ->
``stages`` -> ``tdgw.h5`` (density_matrix every 100 as) on the 2x2 host mesh
(k padded 5 -> 8).  Depopulation 5 meV and dephasing 20 meV in the dissipative arm.

Gates (G) and red twins (R).  Tolerances are fixed a priori, 1e-7 in rho for
the fourth-order schemes at the finer step, 3e-5 for magnus2, 1e-8 for dp54
at rtol 1e-10 (10-100x above the measured errors, 1e3-1e6x below every twin);
the order criterion certifies that what remains is truncation error, which a
modelling error (step-independent) cannot imitate.

G1 every method (dp54, rk4, etdrk4, magnus2, cfm4), Gamma = 0 and Gamma != 0:
   max |rho - rho_exact| <= tol at the finer step, observed order
   log2(e(h)/e(h/2)) within 0.3 of nominal (4; magnus2 2; cfm4 with Gamma 2:
   Strang); dp54 error ratio e(rtol 1e-8)/e(rtol 1e-10) in [20, 500]
R1 the bra/ket-swapped dipole header (r -> r^T, a sigma- transition): the
   sigma+ run misses the exact sigma+ solution by O(1)
R2 the reference built with gap/(2 Ry) instead of the header's (e/Ry) Ry/Ha
   (0.36 ppm in the gap): G1's checker fails on both criteria; the run sits
   the exact reference shift (1e-5) away at every step (order ~ 0)
"""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import _validation_systems as vs  # noqa: E402
from _validation_systems import record  # noqa: E402,F401  (fixture)
from common.units import RYD_TO_EV  # noqa: E402

TF_AS, MDT_AS = 48000.0, 100.0
W_EV, E0_CW = 2.0, 2.0e9
TAU_AS = 4000.0
TC_AS = 6 * TAU_AS
GAMMA_EV = (0.005, 0.02)                 # depopulation, dephasing

#: (method, coarse step, fine step, nominal order at Gamma = 0, at Gamma != 0, tol)
SCHEMES = [("dp54", 1e-8, 1e-10, None, None, 1e-8),
           ("rk4", 10.0, 5.0, 4, 4, 1e-7),
           ("etdrk4", 20.0, 10.0, 4, 4, 1e-7),
           ("magnus2", 5.0, 2.5, 2, 2, 3e-5),
           ("cfm4", 20.0, 10.0, 4, 2, 1e-7)]


def _setup():
    tl = vs.two_level()
    tau = TAU_AS / vs.T_AS
    g0 = np.sqrt(np.pi) / (2 * tau)                # int g dt = pi/2: full inversion
    e0_pulse = g0 * np.sqrt(2) / tl["d"] * vs.E_VM
    trials = [vs.circular_trial(E0_CW, W_EV, sign=+1), vs.circular_trial(E0_CW, W_EV, sign=-1),
              vs.circular_trial(e0_pulse, W_EV, sign=+1,
                                envelope=vs.gaussian_env(TC_AS, TAU_AS))]
    g = E0_CW / vs.E_VM * tl["d"] / np.sqrt(2)
    w = (W_EV / vs.HBAR_EV_AS) * vs.T_AS
    gfun = lambda t: g0 * np.exp(-((t - TC_AS / vs.T_AS) / tau) ** 2)   # noqa: E731
    return tl, trials, g, w, gfun


def _reference(gap, g, w, gfun, t, gam):
    g1, g2 = (x / vs.HA_IN_EV for x in gam)
    ref = np.zeros((3, t.size, gap.size, 2, 2), complex)
    for k in range(gap.size):
        ref[0, :, k] = vs.rabi_exact(gap[k], g, w, t, sign=+1, gamma1=g1, gamma2=g2)
        ref[1, :, k] = vs.rabi_exact(gap[k], g, w, t, sign=-1, gamma1=g1, gamma2=g2)
        ref[2, :, k] = vs.rabi_pulse_reference(gap[k], gfun, w, t, gamma1=g1, gamma2=g2)
    return ref


def _run(tmp, hdr, tl, trials, method, step, gam):
    kw = dict(rtol=step) if method == "dp54" else dict(dt_as=step)
    res, stats = vs.run_deck(os.path.join(tmp, f"{method}_{step}_{gam[0]}"), hdr, tl,
                             trials=trials, tf_as=TF_AS, dt_measure_as=MDT_AS, method=method,
                             atol=1e-14, depopulation_ev=gam[0], dephasing_ev=gam[1],
                             measurements=("Efield", "density_matrix"), **kw)
    t = res[0]["density_matrix"][0]
    return t, np.stack([r["density_matrix"][1] for r in res]), stats


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    tmp = str(tmp_path_factory.mktemp("rabi"))
    tl, trials, g, w, gfun = _setup()
    hdr = vs.write_header(os.path.join(tmp, "hdr"), tl)
    gap = vs.energies_ha(tl)[:, 1]
    out = dict(tl=tl, trials=trials, g=g, w=w, gfun=gfun, tmp=tmp, hdr=hdr, gap=gap, runs={})
    for gam in ((0.0, 0.0), GAMMA_EV):
        ref = None
        for method, coarse, fine, *_ in SCHEMES:
            for step in (coarse, fine):
                t, rho, stats = _run(tmp, hdr, tl, trials, method, step, gam)
                if ref is None:
                    ref = _reference(gap, g, w, gfun, t, gam)
                out["runs"][(gam, method, step)] = dict(t=t, rho=rho, stats=stats, ref=ref)
    return out


def _error(rho, ref):
    return float(np.abs(rho - ref).max())


def _passes(e_coarse, e_fine, *, order, tol, dp54=False):
    if dp54:
        return e_fine <= tol and 20.0 <= e_coarse / e_fine <= 500.0
    return e_fine <= tol and abs(np.log2(e_coarse / e_fine) - order) < 0.3


@pytest.mark.parametrize("gam", [(0.0, 0.0), GAMMA_EV], ids=["conservative", "dissipative"])
@pytest.mark.parametrize("scheme", SCHEMES, ids=[s[0] for s in SCHEMES])
def test_rabi_exact_every_integrator(ladder, scheme, gam, record):
    method, coarse, fine, p0, p1, tol = scheme
    rc, rf = ladder["runs"][(gam, method, coarse)], ladder["runs"][(gam, method, fine)]
    assert rf["stats"]["mesh"] == (2, 2) and rf["stats"]["nk_pad"] == 8
    ec, ef = _error(rc["rho"], rc["ref"]), _error(rf["rho"], rf["ref"])
    record("max_abs_rho_minus_exact_fine", ef)
    record("observed_order" if method != "dp54" else "error_ratio_rtol",
           float(np.log2(ec / ef)) if method != "dp54" else float(ec / ef))
    order = p0 if gam == (0.0, 0.0) else p1
    assert _passes(ec, ef, order=order, tol=tol, dp54=method == "dp54"), \
        (method, gam, ec, ef, None if method == "dp54" else np.log2(ec / ef))
    # the physics the runs show: full inversion by sigma+ and by the pi pulse, none by sigma-
    if gam == (0.0, 0.0):
        pop = rf["rho"][..., 1, 1].real
        assert pop[0, :, 2].max() > 0.999 and pop[2, -1, 2] > 0.999999
        assert pop[1].max() < 2e-3


def test_red_twin_swapped_dipole_header(ladder, tmp_path):
    tl = dict(ladder["tl"])
    tl["r"] = np.swapaxes(tl["r"], -1, -2)              # <c|r|v> = d (1, +i)/sqrt2: sigma-
    hdr = vs.write_header(str(tmp_path / "hdr_swap"), tl)
    t, rho, _ = _run(str(tmp_path), hdr, tl, ladder["trials"], "dp54", 1e-10, (0.0, 0.0))
    ref = ladder["runs"][((0.0, 0.0), "dp54", 1e-10)]["ref"]
    err = _error(rho[0], ref[0])
    assert not err <= 1e-8 and err > 0.5, err


def test_red_twin_rydberg_hartree_trap(ladder, record):
    """The gap converted with Ha = 2 Ry (BerkeleyGW's Ry) instead of the header's arithmetic."""
    lad = ladder
    gap_bad = lad["tl"]["e_ev"][:, 1] / (2.0 * RYD_TO_EV)
    assert 1e-7 < np.max(np.abs(gap_bad / lad["gap"] - 1)) < 1e-6
    rc = lad["runs"][((0.0, 0.0), "cfm4", 20.0)]
    rf = lad["runs"][((0.0, 0.0), "cfm4", 10.0)]
    ref = _reference(gap_bad, lad["g"], lad["w"], lad["gfun"], rf["t"], (0.0, 0.0))
    ec, ef = _error(rc["rho"], ref), _error(rf["rho"], ref)
    assert not _passes(ec, ef, order=4, tol=1e-7), (ec, ef)
    # the run sits exactly the reference shift away: the 0.36 ppm is resolved, not noise
    shift = _error(rf["ref"], ref)
    record("twin_error", ef)
    assert ef > 50 * 1e-7 and abs(ef / shift - 1) < 0.05 and abs(np.log2(ec / ef)) < 0.3, \
        (ec, ef, shift)
