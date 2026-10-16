"""Validation ladder, rung 6: magneto-optics from TD runs == the BSE on the same Hamiltonian.

Physics.  The x and y kick responses of a TRS-broken window give the in-plane
susceptibility chi_ab(w + i eta) exactly in linear response (damped Fourier
transform, ``tools/td_response.py``), hence eps = 1 + 4 pi chi (supercell),
the sheet conductivity sigma^2D = -i w chi L_z, the circular components
chi_+- = chi_xx +- i chi_xy for the (x +- i y) field (sigma+ = (cos, sin)) and
the polarization rotation of a free-standing monolayer at normal incidence
(lab frame, from x toward y, counter-clockwise viewed from +z):

    sheet:  T = (1 + (2 pi/c) sigma^2D)^-1,  R = -(2 pi/c) sigma^2D T
    film:   eps_film - 1 = (eps - 1) L_z/d; the eigenpolarizations of the in-plane
            eps_film (circular, n_+- = sqrt(eps_xx +- i eps_xy), when eps_yy = eps_xx
            and eps_yx = -eps_xy), Airy reflection/transmission of each through a
            slab of thickness d (d = 6.6 A, the interlayer spacing of bulk CrI3, as
            Wu et al. 2019),

theta + i*epsilon of the outgoing ellipse.  The two models of the same
sigma^2D agree to O(|n| w d/c) (2 % on this toy).  Time reversal
(phi -> -phi in the Haldane model) flips eps_xy and keeps eps_xx exactly
(Onsager).  The interband length-gauge response has sigma_xy = -i w chi_xy
-> 0 as w^2: the dc anomalous Hall term would come from the intraband
commutator, absent from a velocity-built (itype 1) header.

System: ``haldane`` (6x6 k, interacting) and its time-reversed partner; x and y
kicks (tau 50 as, +-1e7 V/m odd part), dp54 rtol 1e-10, 70 fs; eta = 0.15 eV,
so the truncation of the damped transform is e^{-eta T} = 1.2e-7.

G1 chi_xx, chi_xy, eps_xy and sigma^2D_xy from TD == BSE (1e-5 of max)
G2 chi_+ and chi_- from TD == the exciton table's sigma+- spectrum
   -(g_s/Omega) sum_S [I+-(S)/(z - Omega_S) - I-+(S)/(z + Omega_S)] (1e-5), the
   circular dichroism Im(chi_+ - chi_-) likewise; the lowest sigma+ state is S0
G3 Kerr and Faraday (sheet) from TD == BSE (1e-5 of max); R e_+- = r_+- e_+-
   with r_+- = -s_+-/(1 + s_+-), s_+- = (2 pi/c)(sigma_xx +- i sigma_xy) (1e-12);
   sheet vs film of the same sigma^2D (d = 6.6 A) within |n| w d/c (they
   converge linearly in d: 1.8e-4 here for Faraday)
G4 Onsager: eps_xy(-phi) == -eps_xy(phi), eps_xx(-phi) == eps_xx(phi), TD (1e-10)
G5 sigma_xy(w)/w^2 constant to 1 % for w <= 40 meV (no dc Hall term)
R1 the exciton table built with the reversed convention (I+ <-> I-): G2 fails
R2 Onsager without the sign (eps_xy(-phi) == +eps_xy(phi)): G4 fails
G6 the film of an anisotropic sheet (the BSE chi strained in-plane, S chi S^T) -> the
   sheet linearly in d: each decade of d divides both deviations by 10 (+-20 %),
   <= 1e-5 at d = 0.066 A
R3 the transposed conductivity in the Kerr formula (sigma_xy <-> sigma_yx): G3 fails
R4 the circular-mode film on that anisotropic tensor: no convergence, G6 fails
"""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import _validation_systems as vs  # noqa: E402
from _validation_systems import record  # noqa: E402,F401  (fixture)

ETA_EV, TF_AS = 0.15, 70000.0
D_ANG = 6.6


def _run(tmp, tag, phi):
    tdr = vs.load_td_response()
    h = vs.haldane(phi=phi)
    hdr = vs.write_header(os.path.join(tmp, "hdr_" + tag), h)
    kern = vs.write_kernel(os.path.join(tmp, f"kernel_{tag}.h5"), h)
    trials = []
    for pol in ("x", "y"):
        trials += [vs.trial_for(1e7, 0.0, 300.0, 50.0, pol),
                   vs.trial_for(-1e7, 0.0, 300.0, 50.0, pol)]
    res, _ = vs.run_deck(os.path.join(tmp, "mo_" + tag), hdr, h, trials=trials, tf_as=TF_AS,
                         dt_measure_as=10.0, method="dp54", rtol=1e-10, max_step_as=10.0,
                         kernel_file=kern, measurements=("Efield", "polarization"))
    t = res[0]["polarization"][0]
    P = [0.5 * (res[2 * i]["polarization"][1].real - res[2 * i + 1]["polarization"][1].real)
         for i in range(2)]
    E = [0.5 * (res[2 * i]["Efield"][1] - res[2 * i + 1]["Efield"][1]) for i in range(2)]
    e = vs.energies_ha(h)
    modes = tdr.window_modes(e, 1, h["K_ha"], h["r"], np.full(e.shape[0], 1.0 / e.shape[0]))
    vol = abs(np.linalg.det(h["cell"]))
    w = np.linspace(0.3, 3.5, 641) / vs.HA_IN_EV
    eta = ETA_EV / vs.HA_IN_EV
    chi = np.zeros((w.size, 3, 3), complex)
    chi[:, :, 0] = tdr.chi_from_td(t, P[0], E[0], w, eta)
    chi[:, :, 1] = tdr.chi_from_td(t, P[1], E[1], w, eta)
    ref = tdr.chi_from_modes(modes, w, eta, g_s=2.0, cell_volume=vol)
    return dict(h=h, modes=modes, w=w, eta=eta, chi=chi, ref=ref, vol=vol,
                lz=float(h["cell"][2, 2]))


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    tmp = str(tmp_path_factory.mktemp("mo"))
    return dict(tdr=vs.load_td_response(), p=_run(tmp, "p", np.pi / 2),
                m=_run(tmp, "m", -np.pi / 2))


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_susceptibility_and_conductivity(case, record):
    c, tdr = case["p"], case["tdr"]
    sig_t = tdr.sheet_conductivity(c["chi"], c["w"], c["lz"])
    sig_b = tdr.sheet_conductivity(c["ref"], c["w"], c["lz"])
    devs = dict(chi_xx=_rel(c["chi"][:, 0, 0], c["ref"][:, 0, 0]),
                chi_xy=_rel(c["chi"][:, 0, 1], c["ref"][:, 0, 1]),
                eps_xy=_rel(4 * np.pi * c["chi"][:, 0, 1], 4 * np.pi * c["ref"][:, 0, 1]),
                sigma_xy=_rel(sig_t[:, 0, 1], sig_b[:, 0, 1]))
    for k, v in devs.items():
        record(k + "_rel_dev", v)
    assert max(devs.values()) <= 1e-5, devs


def _circular_ok(c, tdr, reversed_convention=False):
    """TD chi_+- == the exciton table's sigma+- spectrum; sigma+ absorption starts at S0.

    chi_+ = -(g_s/Omega) sum_S [I+(S)/(z - Omega_S) - I-(S)/(z + Omega_S)]
    (``circular_from_strengths``), with I+- of the BSE residues in the
    campaign convention; ``reversed_convention`` builds the table with
    I+ <-> I- (the reversed labels) while the TD side stays as measured.
    """
    m = c["modes"]
    pos = m.lam.real > 0
    ip, im, _ = tdr.helicity(m.R[pos])
    if reversed_convention:
        ip, im = im, ip
    tp, tm = tdr.circular_from_strengths(c["w"], c["eta"], m.lam.real[pos], ip, im,
                                         g_s=2.0, cell_volume=c["vol"])
    pt, mt = tdr.circular(c["chi"])
    lowest_p = m.lam.real[pos][np.argmax(ip > 1e-6 * ip.max())]
    return (_rel(pt, tp) <= 1e-5 and _rel(mt, tm) <= 1e-5
            and _rel((pt - mt).imag, (tp - tm).imag) <= 1e-5
            and abs(lowest_p - m.lam.real[pos][0]) < 1e-12)


def test_helicity_resolved_absorption(case):
    assert _circular_ok(case["p"], case["tdr"])


def _kerr(c, tdr, chi, transpose=False):
    sig = tdr.sheet_conductivity(chi, c["w"], c["lz"])
    if transpose:
        sig = np.swapaxes(sig, -1, -2)
    return tdr.kerr_faraday_sheet(sig)


def test_kerr_and_faraday(case, record):
    c, tdr = case["p"], case["tdr"]
    (kt, ft), (kb, fb) = _kerr(c, tdr, c["chi"]), _kerr(c, tdr, c["ref"])
    record("kerr_rel_dev", _rel(kt, kb))
    record("faraday_rel_dev", _rel(ft, fb))
    record("max_kerr_rotation_mrad", float(1e3 * np.abs(kb.real).max()))
    record("max_faraday_rotation_mrad", float(1e3 * np.abs(fb.real).max()))
    assert _rel(kt, kb) <= 1e-5 and _rel(ft, fb) <= 1e-5
    # the Jones matrix's eigenvectors are the lab circular vectors
    sig = tdr.sheet_conductivity(c["ref"], c["w"], c["lz"])[:, :2, :2]
    s = (2 * np.pi / tdr.C_AU) * sig
    Rm = -s @ np.linalg.inv(np.eye(2) + s)
    for sgn in (+1, -1):
        e = np.array([1.0, sgn * 1j]) / np.sqrt(2)
        sc = s[:, 0, 0] + sgn * 1j * s[:, 0, 1]
        assert np.abs(Rm @ e - (-sc / (1 + sc))[:, None] * e[None]).max() <= 1e-12
    # sheet and film of the same sigma^2D
    d = D_ANG / tdr.BOHR_IN_ANG
    eps_f = tdr.film_eps(c["ref"], c["lz"], d)
    kf, ff = tdr.kerr_faraday_film(eps_f, c["w"], d)
    n_max = np.abs(np.sqrt(eps_f[:, 0, 0] + 1j * np.abs(eps_f[:, 0, 1]))).max()
    bound = n_max * c["w"].max() * d / tdr.C_AU
    record("film_vs_sheet_kerr_rel", _rel(kf, kb))
    record("film_vs_sheet_faraday_rel", _rel(ff, fb))
    record("film_vs_sheet_bound", float(bound))
    assert _rel(kf, kb) <= bound and _rel(ff, fb) <= bound


def test_onsager_time_reversal(case, record):
    p, m = case["p"]["chi"], case["m"]["chi"]
    odd = float(np.abs(p[:, 0, 1] + m[:, 0, 1]).max() / np.abs(p[:, 0, 1]).max())
    even = float(np.abs(p[:, 0, 0] - m[:, 0, 0]).max() / np.abs(p[:, 0, 0]).max())
    record("eps_xy_plus_reversed", odd)
    record("eps_xx_minus_reversed", even)
    assert odd <= 1e-10 and even <= 1e-10


def test_no_dc_hall_term_in_the_length_gauge(case, record):
    c, tdr = case["p"], case["tdr"]
    w = np.array([0.01, 0.02, 0.04]) / vs.HA_IN_EV
    chi = tdr.chi_from_modes(c["modes"], w, 0.0, g_s=2.0, cell_volume=c["vol"])
    ratio = (-1j * w * chi[:, 0, 1] * c["lz"]) / w ** 2
    record("sigma_xy_over_w2_spread", float(np.ptp(ratio.real) / abs(ratio.real).mean()))
    assert np.ptp(ratio.real) <= 0.01 * abs(ratio.real).mean()


def _rel_rot(a, b):
    """_rel for theta + i*epsilon, the rotation compared modulo pi."""
    d_th = (a.real - b.real + np.pi / 2) % np.pi - np.pi / 2
    return float(np.abs(d_th + 1j * (a.imag - b.imag)).max() / np.abs(b).max())


def _strained(chi, theta=0.3, s1=1.25, s2=0.8):
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    S = rot @ np.diag([s1, s2]) @ rot.T
    out = np.array(chi, copy=True)
    out[:, :2, :2] = S @ out[:, :2, :2] @ S.T
    return out


def _film_circular(tdr, eps, w, d):
    """The circular-mode film, exact only for an isotropic in-plane eps (R4)."""
    out = []
    for sgn in (+1, -1):
        n = np.sqrt(eps[:, 0, 0] + sgn * 1j * eps[:, 0, 1])
        n = np.where(n.imag < 0, -n, n)
        r01 = (1 - n) / (1 + n)
        e1 = np.exp(1j * n * w * d / tdr.C_AU)
        den = 1 - r01 ** 2 * e1 ** 2
        out.append((r01 * (1 - e1 ** 2) / den, (2 / (1 + n)) * (2 * n / (n + 1)) * e1 / den))
    (rp, tp), (rm, tm) = out
    vac = np.exp(1j * w * d / tdr.C_AU)

    def col(a, b):
        return np.stack([(a + b) / 2, 1j * (a - b) / 2], axis=-1)
    return tdr._ellipse(col(rp, rm)), tdr._ellipse(col(tp / vac, tm / vac))


def _film_devs(c, tdr, chi, film):
    kb, fb = tdr.kerr_faraday_sheet(tdr.sheet_conductivity(chi, c["w"], c["lz"]))
    out = []
    for d_ang in (D_ANG, D_ANG / 10, D_ANG / 100):
        d = d_ang / tdr.BOHR_IN_ANG
        kf, ff = film(tdr.film_eps(chi, c["lz"], d), c["w"], d)
        out.append((_rel_rot(kf, kb), _rel_rot(ff, fb)))
    return out


def test_film_of_an_anisotropic_sheet(case, record):
    c, tdr = case["p"], case["tdr"]
    chi = _strained(c["ref"])
    aniso = float(np.abs(chi[:, 0, 0] - chi[:, 1, 1]).max() / np.abs(chi[:, 0, 0]).max())
    devs = _film_devs(c, tdr, chi, tdr.kerr_faraday_film)
    record("anisotropy_xx_minus_yy", aniso)
    for (k, f), lab in zip(devs, ("6.6A", "0.66A", "0.066A")):
        record(f"film_vs_sheet_kerr_{lab}", k)
        record(f"film_vs_sheet_faraday_{lab}", f)
    assert aniso >= 0.1
    for (k1, f1), (k2, f2) in zip(devs, devs[1:]):
        assert 8.0 <= k1 / k2 <= 12.0 and 8.0 <= f1 / f2 <= 12.0
    assert devs[-1][0] <= 1e-5 and devs[-1][1] <= 1e-5


def test_red_twin_circular_film_on_an_anisotropic_sheet(case):
    c, tdr = case["p"], case["tdr"]
    devs = _film_devs(c, tdr, _strained(c["ref"]),
                      lambda eps, w, d: _film_circular(tdr, eps, w, d))
    assert not (devs[-1][0] <= 1e-5 and devs[-1][1] <= 1e-5)


def test_red_twin_reversed_circular_convention(case):
    assert not _circular_ok(case["p"], case["tdr"], reversed_convention=True)


def test_red_twin_onsager_without_sign(case):
    p, m = case["p"]["chi"], case["m"]["chi"]
    assert not np.abs(p[:, 0, 1] - m[:, 0, 1]).max() <= 1e-10 * np.abs(p[:, 0, 1]).max()


def test_red_twin_transposed_conductivity(case):
    c, tdr = case["p"], case["tdr"]
    (kt, _), (kb, _) = _kerr(c, tdr, c["chi"], transpose=True), _kerr(c, tdr, c["ref"])
    assert not _rel(kt, kb) <= 1e-5
