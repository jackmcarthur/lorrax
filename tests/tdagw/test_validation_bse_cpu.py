"""Validation ladder, rung 2: TD-aGW linear response == the full BSE of its kernel.

Physics.  Linearized about rho_eq, the TD-aGW equation of motion for
z = (drho_cv, drho_vc) is i dz/dt = L z + E(t).u with
L = [[D + K_XX, K_XY], [-K_YX, -D - K_YY]], the full (non-TDA) BSE whose
coupling block K_XY is the kernel's (cv, v'c') block
(``docs/theory/td_agw.md``; ``tools/td_response.py`` owns the algebra).  So
(i) P(t) is the closed-form modal response of L for any weak drive,
(ii) projected on L's left eigenvectors (z^H eta z = 1) the propagated
state carries per state S the first-order amplitude c_S and rotates as
e^{-i Omega_S t} after the pulse, and (iii) chi(w + i eta) from the damped
Fourier transform of P(t) equals the Lorentzian BSE spectrum exactly
(convolution theorem: FT[P e^{-eta t}] = chi(w + i eta) FT[E e^{-eta t}]).
None of these is limited by the frequency resolution of the run: 81 states
spaced down to 1.6 meV are checked one by one in 50 fs.

System: ``synthetic_window`` — 3v3c spinors on a 3x3 grid (k padded 9 -> 12),
ISDF-form Hartree + SEX kernel (|K| <= 0.16 eV; full - TDA up to 0.7 meV),
written as a TDGW extended kernel.h5 and propagated through the deck route.
Drives: x and y kicks (Gaussian, tau 50 as, 1e7 V/m) and sigma+/sigma- pulses
(tau 1.5 fs, carrier 1.95 eV, 1e6 V/m); every drive is run at +E0 and -E0 and
the odd part is the linear response (the window has no inversion and
intra-manifold dipoles, so the even orders are physical: P2/P1 ~ |rho_cv|).
dp54 at rtol 1e-10 (max_step 10 as) for every gate; etdrk4 and cfm4 at 10 as
for the time-domain gate.

Tolerances.  The third-order residue (|rho_cv|^2 = 5e-9 kicks, 4e-8 pulses)
and the integration (rtol 1e-10) set the floor; gates sit 20x above it:
P(t) 1e-6 (dp54), 1e-5 (etdrk4/cfm4 at h/tau = 0.2); amplitudes 1e-5;
frequencies 1 ueV; spectra 1e-5 (truncation e^{-eta T} = 2.5e-7 at eta 0.2 eV).

G1 P(t) of x, y, sigma+, sigma- drives == the full-BSE closed form (dp54,
   etdrk4, cfm4)
G2 per state (every state above 1e-4 of the largest population): the
   propagated amplitude after sigma+ and sigma- pulses == first-order theory,
   and its phase velocity == Omega_S
G3 Im chi_+, Im chi_- (sigma+ = (cos, sin)) and the in-plane tensor from TD == BSE
G4 the sigma+/sigma- population ratio of every state == I+/I- of the BSE
   residues (campaign convention I+- = |D_x -+ i D_y|^2/2, D = <0|r|S>)
R1 the TDA kernel file (nblocks 1): G1 and G2 fail (the coupling block is resolved)
R2 the full run against the TDA BSE: G1 fails
R3 the reversed helicity convention (I+ <-> I-): G4 fails
"""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import _validation_systems as vs  # noqa: E402
from _validation_systems import record  # noqa: E402,F401  (fixture)

TF_AS, MDT_AS, DM_EVERY = 50000.0, 10.0, 10
KICK = (1e7, 0.0, 300.0, 50.0)                      # E0 V/m, w eV, tc as, tau as
PULSE = (1e6, 1.95, 9000.0, 1500.0)
DRIVES = [("x",) + KICK, ("y",) + KICK, ("+",) + PULSE, ("-",) + PULSE]
ETA_EV = 0.2


def _trials():
    out = []
    for pol, e0, w, tc, tau in DRIVES:
        out += [vs.trial_for(e0, w, tc, tau, pol), vs.trial_for(-e0, w, tc, tau, pol)]
    return out


def _odd(res, name):
    return [0.5 * (res[2 * i][name][1] - res[2 * i + 1][name][1]) for i in range(len(DRIVES))]


def _propagate(tmp, tag, hdr, sw, kernel, method="dp54", **kw):
    nt = int(round(TF_AS / MDT_AS))
    meas = ("Efield", "polarization", "density_matrix") if method == "dp54" else \
        ("Efield", "polarization")
    if method == "dp54":
        kw = dict(dict(rtol=1e-10, max_step_as=10.0), **kw)
    res, stats = vs.run_deck(os.path.join(tmp, tag), hdr, sw, trials=_trials(), tf_as=TF_AS,
                             dt_measure_as=MDT_AS, method=method, kernel_file=kernel,
                             measurements=meas,
                             meas_opts={"density_matrix": {"itrange": [0, nt, DM_EVERY]}}, **kw)
    out = dict(t=res[0]["polarization"][0], P=[p.real for p in _odd(res, "polarization")],
               E=_odd(res, "Efield"), stats=stats)
    if "density_matrix" in res[0]:
        out["t_dm"] = res[0]["density_matrix"][0]
        out["drho"] = _odd(res, "density_matrix")
    return out


def _tda(K, nv):
    Kt = np.zeros_like(K)
    Kt[:, nv:, :nv, :, nv:, :nv] = K[:, nv:, :nv, :, nv:, :nv]
    Kt[:, :nv, nv:, :, :nv, nv:] = K[:, :nv, nv:, :, :nv, nv:]
    return Kt


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    tdr = vs.load_td_response()
    tmp = str(tmp_path_factory.mktemp("bse"))
    sw = vs.synthetic_window()
    hdr = vs.write_header(os.path.join(tmp, "hdr"), sw)
    kfull = vs.write_kernel(os.path.join(tmp, "kernel_full.h5"), sw)
    ktda = vs.write_kernel(os.path.join(tmp, "kernel_tda.h5"), sw, tda=True)
    e, nv = vs.energies_ha(sw), sw["n_val"]
    wk = np.full(e.shape[0], 1.0 / e.shape[0])
    modes = tdr.window_modes(e, nv, sw["K_ha"], sw["r"], wk)
    modes_tda = tdr.window_modes(e, nv, _tda(sw["K_ha"], nv), sw["r"], wk)
    pulses = [vs.pulse_au(tdr, e0, w, tc, tau, pol) for pol, e0, w, tc, tau in DRIVES]
    runs = {"dp54": _propagate(tmp, "dp54", hdr, sw, kfull),
            "etdrk4": _propagate(tmp, "etdrk4", hdr, sw, kfull, "etdrk4", dt_as=10.0),
            "cfm4": _propagate(tmp, "cfm4", hdr, sw, kfull, "cfm4", dt_as=10.0),
            "tda": _propagate(tmp, "tda", hdr, sw, ktda)}
    return dict(tdr=tdr, sw=sw, modes=modes, modes_tda=modes_tda, pulses=pulses, runs=runs,
                vol=abs(np.linalg.det(sw["cell"])), g_s=1.0)


def _p_dev(c, run, modes):
    """max over drives of max|P_TD - P_BSE| / max|P_BSE|."""
    tdr, out = c["tdr"], []
    for i, p in enumerate(c["pulses"]):
        ref = -(c["g_s"] / c["vol"]) * tdr.modal_response(modes, p, run["t"]).real
        out.append(np.abs(run["P"][i] - ref).max() / np.abs(ref).max())
    return max(out)


@pytest.mark.parametrize("method,tol", [("dp54", 1e-6), ("etdrk4", 1e-5), ("cfm4", 1e-5)])
def test_p_of_t_is_the_full_bse_response(case, method, tol, record):
    run = case["runs"][method]
    assert run["stats"]["mesh"] == (2, 2) and run["stats"]["nk_pad"] == 12
    dev = _p_dev(case, run, case["modes"])
    record("P_rel_dev", dev)
    assert dev <= tol, (method, dev)


def _state_check(c, run, modes):
    """(max amplitude deviation, max |Omega_TD - Omega_BSE| in Ha, n states) over sigma+- pulses."""
    tdr = c["tdr"]
    pos = np.where(modes.lam.real > 0)[0]
    late = run["t_dm"] > (PULSE[2] + 6 * PULSE[3]) / vs.T_AS
    amp, dom, n = 0.0, 0.0, 0
    for i in (2, 3):
        c_td = tdr.project_state(modes, run["drho"][i], c["sw"]["n_val"])[:, pos]
        c_pt = tdr.mode_amplitudes(modes, c["pulses"][i])[pos]
        big = np.abs(c_pt) ** 2 > 1e-4 * np.max(np.abs(c_pt) ** 2)
        rot = c_td[late][:, big] * np.exp(1j * np.outer(run["t_dm"][late], modes.lam[pos][big]))
        amp = max(amp, float((np.abs(rot - c_pt[big][None]).max(axis=0) / np.abs(c_pt[big])).max()))
        ph = np.unwrap(np.angle(c_td[late][:, big]), axis=0)
        slope = np.polyfit(run["t_dm"][late], ph, 1)[0]
        dom = max(dom, float(np.abs(-slope - modes.lam.real[pos][big]).max()))
        n += int(big.sum())
    return amp, dom, n


def test_every_state_amplitude_and_frequency(case, record):
    amp, dom, n = _state_check(case, case["runs"]["dp54"], case["modes"])
    record("amplitude_rel_dev", amp)
    record("frequency_dev_ev", dom * vs.HA_IN_EV)
    record("n_states", n)
    assert n >= 140 and amp <= 1e-5 and dom * vs.HA_IN_EV <= 1e-6, (n, amp, dom * vs.HA_IN_EV)


def test_spectra_both_helicities(case, record):
    tdr, run = case["tdr"], case["runs"]["dp54"]
    eta = ETA_EV / vs.HA_IN_EV
    w = np.linspace(1.5, 4.0, 401) / vs.HA_IN_EV
    chi = np.zeros((w.size, 3, 3), complex)
    chi[:, :, 0] = tdr.chi_from_td(run["t"], run["P"][0], run["E"][0], w, eta)
    chi[:, :, 1] = tdr.chi_from_td(run["t"], run["P"][1], run["E"][1], w, eta)
    ref = tdr.chi_from_modes(case["modes"], w, eta, g_s=case["g_s"], cell_volume=case["vol"])
    (pt, mt), (pb, mb) = tdr.circular(chi), tdr.circular(ref)
    record("im_chi_plus_rel_dev", float(np.abs(pt.imag - pb.imag).max() / np.abs(pb.imag).max()))
    record("im_chi_minus_rel_dev", float(np.abs(mt.imag - mb.imag).max() / np.abs(mb.imag).max()))
    for a, b in ((pt.imag, pb.imag), (mt.imag, mb.imag), (chi[:, :2, :2], ref[:, :2, :2])):
        assert np.abs(a - b).max() <= 1e-5 * np.abs(b).max()
    # the two helicities are different spectra (the window breaks time reversal)
    assert np.abs(pb.imag - mb.imag).max() > 0.1 * np.abs(pb.imag).max()


def _ratio_dev(c, *, swap=False):
    """max |(N+/N-)_TD / (I+/I-)_BSE - 1| over states populated by both pulses."""
    tdr, run, modes = c["tdr"], c["runs"]["dp54"], c["modes"]
    pos = np.where(modes.lam.real > 0)[0]
    n = [np.abs(tdr.project_state(modes, run["drho"][i], c["sw"]["n_val"])[-1, pos]) ** 2
         for i in (2, 3)]
    ip, im, _ = tdr.helicity(modes.R[pos])
    if swap:
        ip, im = im, ip
    # first-order populations carry |F(Omega_S)|^2, the same for both pulses
    sel = (n[0] > 1e-4 * n[0].max()) & (n[1] > 1e-4 * n[1].max())
    return float(np.abs((n[0][sel] / n[1][sel]) / (ip[sel] / im[sel]) - 1).max()), int(sel.sum())


def test_helicity_of_every_state(case, record):
    dev, n = _ratio_dev(case)
    record("population_ratio_vs_Iplus_over_Iminus", dev)
    assert n >= 60 and dev <= 1e-5, (n, dev)


def test_red_twin_tda_kernel_fails(case, record):
    run = case["runs"]["tda"]
    record("tda_run_vs_full_bse_P", _p_dev(case, run, case["modes"]))
    assert not _p_dev(case, run, case["modes"]) <= 1e-6
    amp, dom, _ = _state_check(case, run, case["modes"])
    assert not (amp <= 1e-5 and dom * vs.HA_IN_EV <= 1e-6), (amp, dom)
    # ...and the TDA run is the TDA BSE's response: the checker resolves the block, not noise
    assert _p_dev(case, run, case["modes_tda"]) <= 1e-6


def test_red_twin_full_run_against_tda_bse(case):
    assert not _p_dev(case, case["runs"]["dp54"], case["modes_tda"]) <= 1e-6


def test_red_twin_reversed_helicity_convention(case):
    dev, _ = _ratio_dev(case, swap=True)
    assert not dev <= 1e-5, dev
