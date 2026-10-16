"""Validation ladder, rung 3: conservation laws of the TD-aGW flow, every integrator.

Physics.  i d(rho)/dt = [H[rho], rho] with H Hermitian (dSigma Hermitian for
Hermitian drho: V-bar real symmetric, every W(q) Hermitian) is a unitary flow:
it conserves Tr rho_k, Hermiticity and the spectrum of rho, so a Slater
determinant stays one (rho^2 = rho) at Gamma = 0.  With
K_{knm,k'n'm'} = K_{k'm'n',kmn} (V-bar symmetric, W(-q) = W(q)^T) it is the
gradient flow of

    E[rho] = g_s sum_k w_k [Tr(eps_k rho_k) + (1/2) Tr(dSigma_k[drho] drho_k)],

the 1/2 being the Hartree and screened-exchange double counting, and
dE/dt = Omega E(t).j(t) with j the measured current (TDGW's v = v0 + i[dSigma, r];
the missing i E_b [r_b, r_a] term drops out of E.j by antisymmetry): the work
theorem during the pulse, E constant after it.  dSigma = K (rho - rho_eq), so
rho_eq is stationary without a field, with or without relaxation.

System: ``synthetic_window`` (3v3c spinors, 3x3 k, Hartree + SEX extended kernel),
a strong sigma+ pulse (2e9 V/m, tau 1.5 fs, 1.95 eV: 0.13 excitons per cell,
far from linear response, every kernel block active), 30 fs, deck route,
samples every 10 as.  Steps: dp54 rtol 1e-10, rk4 2.5 as, etdrk4 5 as,
magnus2 1 as, cfm4 5 as.

Tolerances, a priori.  Tr rho and Hermiticity are exact identities of every
scheme up to rounding accumulated over <= 3e4 steps (1e-11, 1e-12).  Purity:
the Magnus schemes conjugate rho by an exact unitary (rounding, 1e-11); the
Runge-Kutta family is not unitary and is held to its truncation error (1e-8).
Work theorem: Simpson quadrature of E.j on the 10-as samples, error
(2 w h)^4/180 ~ 7e-8 of the absorbed energy, plus magnus2's second-order
truncation at 1 as (4e-7): 1e-6.  Energy after the pulse: 1e-9 of the
absorbed energy.  Equilibrium: the Runge-Kutta family's right-hand side
vanishes identically at rho_eq (1e-15); a Magnus step conjugates the whole
rho, whose O(1) occupations set an absolute rounding of ~u per exponential
that accumulates linearly (the Magnus rounding floor of the theory page):
N_exp u = 1.7e-12 for the <= 1.5e4 exponentials of these runs (u = 2^-53 = 1.1e-16,
the unit roundoff): 2e-12.

G1 Tr rho_k = n_val, rho = rho^H, rho^2 = rho, every method, Gamma = 0
G2 E(t) - E(0) == Omega int E.j dt; E constant after the pulse; every method
G3 rho(t) == rho_eq with no field (full kernel), every method, Gamma = 0 and 5/20 meV
   (Runge-Kutta family exactly; Magnus to its rounding floor)
R1 dSigma made non-Hermitian (K -> (1 + 0.1 i) K): G1's Hermiticity fails
R2 dephasing 20 meV: G1's purity fails (rho^2 != rho), trace and Hermiticity still hold
R3 E without the 1/2 (Tr(eps rho) + Tr(dSigma drho)): G2 fails after the pulse
R4 dSigma applied to rho instead of rho - rho_eq: G3 fails
R5 the checker of G1 on a state with one k's population moved by 1e-10: fails
"""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import _validation_systems as vs  # noqa: E402
from _validation_systems import record  # noqa: E402,F401  (fixture)

TF_AS, MDT_AS = 30000.0, 10.0
PULSE = (2e9, 1.95, 9000.0, 1500.0)
METHODS = [("dp54", dict(rtol=1e-10, max_step_as=50.0)), ("rk4", dict(dt_as=2.5)),
           ("etdrk4", dict(dt_as=5.0)), ("magnus2", dict(dt_as=1.0)),
           ("cfm4", dict(dt_as=5.0))]
UNITARY = ("magnus2", "cfm4")


def _invariants(rho, nv):
    tr = float(np.abs(np.einsum("tknn->tk", rho).real - nv).max())
    herm = float(np.abs(rho - np.conj(np.swapaxes(rho, -1, -2))).max())
    pur = float(np.abs(np.einsum("tknm,tkml->tknl", rho, rho) - rho).max())
    return tr, herm, pur


def _g1(inv, method):
    tr, herm, pur = inv
    return tr <= 1e-11 and herm <= 1e-12 and pur <= (1e-11 if method in UNITARY else 1e-8)


def _g2(t, energy, work, t_end):
    scale = np.abs(work).max()
    after = t > t_end
    return (np.abs(energy - energy[0] - work).max() <= 1e-6 * scale
            and np.ptp(energy[after]) <= 1e-9 * scale)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    tmp = str(tmp_path_factory.mktemp("cons"))
    sw = vs.synthetic_window()
    hdr = vs.write_header(os.path.join(tmp, "hdr"), sw)
    kfull = vs.write_kernel(os.path.join(tmp, "kernel_full.h5"), sw)
    trial = [vs.trial_for(*PULSE, "+")]
    meas = ("Efield", "polarization", "current", "density_matrix")
    runs = {}
    for method, kw in METHODS:
        res, stats = vs.run_deck(os.path.join(tmp, method), hdr, sw, trials=trial,
                                 tf_as=TF_AS, dt_measure_as=MDT_AS, method=method,
                                 kernel_file=kfull, measurements=meas, **kw)
        runs[method] = dict(t=res[0]["density_matrix"][0], rho=res[0]["density_matrix"][1],
                            E=res[0]["Efield"][1], j=res[0]["current"][1], stats=stats)
    return dict(sw=sw, hdr=hdr, kfull=kfull, tmp=tmp, runs=runs, trial=trial,
                vol=abs(np.linalg.det(sw["cell"])), t_end=(PULSE[2] + 6 * PULSE[3]) / vs.T_AS)


@pytest.mark.parametrize("method", [m for m, _ in METHODS])
def test_trace_hermiticity_purity(case, method, record):
    run = case["runs"][method]
    inv = _invariants(run["rho"], case["sw"]["n_val"])
    nv = case["sw"]["n_val"]
    n_exc = np.einsum("tknn->t", run["rho"][:, :, nv:, nv:]).real.max() / run["rho"].shape[1]
    record("trace_hermiticity_purity", inv)
    assert n_exc > 0.1                       # strongly nonlinear: 0.13 excitons per cell
    assert _g1(inv, method), (method, inv)


@pytest.mark.parametrize("method", [m for m, _ in METHODS])
def test_energy_work_theorem(case, method, record):
    run = case["runs"][method]
    energy = vs.energy_per_cell(case["sw"], run["rho"])
    work = vs.work_per_cell(run["t"], run["E"], run["j"], case["vol"])
    after = run["t"] > case["t_end"]
    scale = np.abs(work).max()
    record("work_residual", float(np.abs(energy - energy[0] - work).max() / scale))
    record("post_pulse_drift", float(np.ptp(energy[after]) / scale))
    assert _g2(run["t"], energy, work, case["t_end"]), method


def _equilibrium(case, method, kw, gam, tag, hook=None):
    res, _ = vs.run_deck(os.path.join(case["tmp"], f"eq_{tag}_{method}_{gam[0]}"), case["hdr"],
                         case["sw"], trials=[{"E_x": 0.0, "E_y": 0.0, "E_z": 0.0}],
                         tf_as=5000.0, dt_measure_as=100.0, method=method,
                         kernel_file=case["kfull"], depopulation_ev=gam[0],
                         dephasing_ev=gam[1], measurements=("density_matrix",),
                         model_hook=hook, **kw)
    rho = res[0]["density_matrix"][1]
    nv, nb = case["sw"]["n_val"], rho.shape[-1]
    req = np.diag((np.arange(nb) < nv).astype(float))
    return float(np.abs(rho - req).max())


@pytest.mark.parametrize("gam", [(0.0, 0.0), (0.005, 0.02)], ids=["conservative", "dissipative"])
@pytest.mark.parametrize("method,kw", METHODS, ids=[m for m, _ in METHODS])
def test_equilibrium_is_stationary(case, method, kw, gam, record):
    dev = _equilibrium(case, method, kw, gam, "ok")
    record("equilibrium_drift", dev)
    assert dev <= (2e-12 if method in UNITARY else 1e-15), (method, dev)


def test_red_twin_non_hermitian_dsigma(case):
    hook = lambda m: m._replace(params=((1 + 0.1j) * m.params[0],) + tuple(m.params[1:]))  # noqa: E731
    res, _ = vs.run_deck(os.path.join(case["tmp"], "twin_nonherm"), case["hdr"], case["sw"],
                         trials=case["trial"], tf_as=TF_AS, dt_measure_as=100.0,
                         method="dp54", rtol=1e-10, max_step_as=50.0,
                         kernel_file=case["kfull"], measurements=("density_matrix",),
                         model_hook=hook)
    inv = _invariants(res[0]["density_matrix"][1], case["sw"]["n_val"])
    assert not _g1(inv, "dp54") and inv[1] > 1e-6, inv


def test_red_twin_dephasing_breaks_purity(case):
    res, _ = vs.run_deck(os.path.join(case["tmp"], "twin_deph"), case["hdr"], case["sw"],
                         trials=case["trial"], tf_as=TF_AS, dt_measure_as=100.0,
                         method="cfm4", dt_as=5.0, kernel_file=case["kfull"],
                         dephasing_ev=0.02, measurements=("density_matrix",))
    tr, herm, pur = _invariants(res[0]["density_matrix"][1], case["sw"]["n_val"])
    assert not _g1((tr, herm, pur), "cfm4") and pur > 1e-3, (tr, herm, pur)
    assert tr <= 1e-11 and herm <= 1e-12                   # what dephasing does not touch


def test_red_twin_energy_without_double_counting(case):
    run = case["runs"]["etdrk4"]
    sw = case["sw"]
    rho = run["rho"]
    good = vs.energy_per_cell(sw, rho)
    nk, nb = rho.shape[1], rho.shape[2]
    req = np.zeros((nk, nb, nb))
    req[:, np.arange(sw["n_val"]), np.arange(sw["n_val"])] = 1.0
    d = rho - req[None]
    ds = np.einsum("knmKNM,tKNM->tknm", sw["K_ha"], d)
    bad = good + 0.5 * np.einsum("tknm,tkmn->t", ds, d).real / nk      # the 1/2 dropped
    work = vs.work_per_cell(run["t"], run["E"], run["j"], case["vol"])
    assert _g2(run["t"], good, work, case["t_end"])
    assert not _g2(run["t"], bad, work, case["t_end"])


def test_red_twin_dsigma_of_rho_not_drho(case):
    hook = lambda m: m._replace(params=(m.params[0], 0.0 * m.params[1]))  # noqa: E731
    dev = _equilibrium(case, "dp54", dict(rtol=1e-10, max_step_as=50.0), (0.0, 0.0),
                       "twin", hook)
    assert not dev <= 1e-15 and dev > 1e-6, dev


def test_red_twin_trace_checker_sees_a_moved_population(case):
    rho = case["runs"]["cfm4"]["rho"].copy()
    rho[-1, 4, 0, 0] += 1e-10
    assert not _g1(_invariants(rho, case["sw"]["n_val"]), "cfm4")
