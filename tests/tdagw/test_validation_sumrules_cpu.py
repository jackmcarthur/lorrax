"""Validation ladder, rung 4: the f-sum (TRK) rule of the TD spectrum, IP and interacting.

Physics.  After an instantaneous length-gauge kick
rho(0+) = e^{-i kappa r_x} rho_eq e^{i kappa r_x} (E_x = kappa delta(t)), the
propagated polarization is kappa chi_xx(t) and its odd short-time series

    P_x(t) = (g_s/Omega) kappa (m1 t - m3 t^3/6 + m5 t^5/120 - ...)

carries the moments m_n = sum_S lam_S^n R^S_xx of the TD spectrum (m0 = m2 = 0).
m1 is the window's f-sum: in closed form m1 = o.L.u (``tools/td_response.py``),
for the independent-particle window the IP oscillator sum
S_IP = 2 sum_k w_k sum_cv (E_c - E_v) |<c|x|v>|^2, and with the full kernel
S_IP + o.K_L.u — the double commutator of the interaction in the window (zero
for a local interaction in a complete basis; here -5 %).  The full (non-TDA)
BSE satisfies the sum rule of its kernel (Thouless); the TDA does not: it
drops the coupling block's 2 Re(d^H K_XY d*).  j(0+) = dP/dt(0+) is the
instantaneous form of the same rule.  The window's own sum is exact here; in a
real material it is the window's share of the full-basis TRK sum N_e (the
CrI3 numbers are in the theory page's validation section).

System: ``synthetic_window`` (3v3c, 3x3 k), IP, full Hartree + SEX kernel and the
TDA kernel file; kicks +-kappa, kappa = 1e-5 (the odd part leaves a kappa^2 |r|^2
= 3e-10 third-order residue), ``tdagw.propagate.run_propagation`` from the
kicked state (dp54 rtol 1e-12, samples every 0.5 as to 40 as), odd polynomial
to t^9 fitted on t <= 30 as (lam_max t = 0.17: the truncated series is 1e-13).

G1 IP: m1_TD == S_IP, m3_TD == o.L^3.u, j(0+) == (g_s/Omega) kappa m1
G2 full kernel: m1_TD == o.L.u (!= S_IP: the shift is measured), m3_TD, j(0+)
G3 TDA kernel: the TDA run obeys its own operator's moments
R1 the TDA run against the full kernel's m1: fails (5e-4 relative)
R2 the interacting run against S_IP: fails (the interaction moves the f-sum)
Tolerances: m1 1e-7, j(0+) 1e-8 (third order 3e-10, dp54 at rtol 1e-12);
m3 1e-6 (fit conditioning of the t^3 coefficient).
"""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import _validation_systems as vs  # noqa: E402
from _validation_systems import record  # noqa: E402,F401  (fixture)

KAPPA = 1e-5


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
    tmp = str(tmp_path_factory.mktemp("trk"))
    sw = vs.synthetic_window()
    hdr = vs.write_header(os.path.join(tmp, "hdr"), sw)
    files = {"ip": None, "full": vs.write_kernel(os.path.join(tmp, "kernel_full.h5"), sw),
             "tda": vs.write_kernel(os.path.join(tmp, "kernel_tda.h5"), sw, tda=True)}
    kernels = {"ip": None, "full": sw["K_ha"], "tda": _tda(sw["K_ha"], sw["n_val"])}
    e, nv = vs.energies_ha(sw), sw["n_val"]
    nk, nb = e.shape
    wk = np.full(nk, 1.0 / nk)
    vol = abs(np.linalg.det(sw["cell"]))
    rho0 = vs.kicked_states(sw, [KAPPA, -KAPPA], 0)
    out = {}
    for name, kf in files.items():
        t, got, _ = vs.propagate_state(hdr, sw, rho0, tf_as=40.0, dt_measure_as=0.5,
                                       rtol=1e-12, atol=1e-18, max_step_as=0.5, kernel_file=kf)
        P = 0.5 * (got["polarization"][:, 0, 0] - got["polarization"][:, 1, 0]).real
        j = 0.5 * (got["current"][:, 0, 0] - got["current"][:, 1, 0]).real
        sel = t <= 30.0 / vs.T_AS
        c, *_ = np.linalg.lstsq(np.stack([t[sel] ** n for n in (1, 3, 5, 7, 9)], axis=1),
                                P[sel], rcond=None)
        pref = 1.0 / vol * KAPPA                     # g_s = 1 (spinors)
        L, o, u = tdr.window_operator(e, nv, kernels[name], sw["r"], wk)
        out[name] = dict(m1_td=c[0] / pref, m3_td=-6 * c[1] / pref, j0=j[0] / pref,
                         m1=float((o[0] @ L @ u[0]).real),
                         m3=float((o[0] @ L @ L @ L @ u[0]).real))
    k, cc, v = tdr.pair_index(nk, nb, nv)
    s_ip = 2 * np.sum(wk[k] * (e[k, cc] - e[k, v]) * np.abs(sw["r"][0, k, cc, v]) ** 2)
    return dict(out=out, s_ip=float(s_ip))


def _rel(a, b):
    return abs(a / b - 1)


def _obeys(r, m1):
    return _rel(r["m1_td"], m1) <= 1e-7 and _rel(r["j0"], m1) <= 1e-8


@pytest.mark.parametrize("name", ["ip", "full", "tda"])
def test_f_sum_of_the_propagated_spectrum(case, name, record):
    r = case["out"][name]
    record("m1_rel", _rel(r["m1_td"], r["m1"]))
    record("m3_rel", _rel(r["m3_td"], r["m3"]))
    record("j0_rel", _rel(r["j0"], r["m1"]))
    assert _obeys(r, r["m1"]) and _rel(r["m3_td"], r["m3"]) <= 1e-6, r
    if name == "ip":
        assert _rel(r["m1"], case["s_ip"]) <= 1e-14         # o.L.u is the IP oscillator sum


def test_interaction_moves_the_window_f_sum(case, record):
    shift = case["out"]["full"]["m1"] / case["s_ip"] - 1
    record("m1_full_over_s_ip_minus_1", shift)
    assert 1e-3 < abs(shift) < 0.2, shift
    assert not _obeys(case["out"]["full"], case["s_ip"])     # R2


def test_red_twin_tda_violates_the_full_sum_rule(case, record):
    tda, full = case["out"]["tda"], case["out"]["full"]
    record("tda_vs_full_m1_rel", _rel(tda["m1_td"], full["m1"]))
    assert not _obeys(tda, full["m1"]) and _rel(tda["m1_td"], full["m1"]) > 1e-5
