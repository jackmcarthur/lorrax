"""Systems, exact references and the deck runner of the TD-aGW validation ladder.

Imported by ``tests/tdagw/test_validation_*.py``.  Every system is written as
a TDGW header (``tdagw.td_system.bundle_header`` -> ``tdgw_io.init_output``) and,
when interacting, a TDGW ``kernel.h5`` (``tdagw.kernel_export.write_tdgw_kernel``),
and propagated through the driver's own stages (``parse_td_deck`` ->
``stages.load_system`` -> ``build_self_energy`` -> ``propagate`` -> ``tdgw.h5``),
so a test exercises exactly what ``python -m tdagw.tdgw_jax`` runs.  The
references (exact two-level solutions, the window BSE of ``tools/td_response.py``)
share no code with the propagator; they read only their own in-memory inputs.

Three systems:

* :func:`two_level` — nk independent copies of one valence and one
  conduction band with a circular dipole <c|r|v> = d (1, -i, 0)/sqrt2 (a
  sigma+ transition) and per-k gaps; a circular drive makes the rotating
  frame exact, so Rabi flopping has closed forms (:func:`rabi_exact`).
* :func:`synthetic_window` — 3v3c spinors on a 3x3 grid (k padded 9 -> 12 on
  four devices) with an ISDF-form Hartree + screened-exchange kernel built by
  explicit sums (:func:`isdf_form_kernel`) from random centroid amplitudes,
  a positive-definite V-bar and a reciprocal Hermitian W(q).
* :func:`haldane` — the C3-symmetric Haldane honeycomb (staggered mass M,
  complex next-neighbour hopping t2 e^{i phi}) on an n x n grid containing
  K and K', with the interaction of the same ISDF form on the two
  sublattice orbitals: valley-split excitons of opposite helicity.

Run the tests with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

from common.units import RYD_TO_EV
from tdagw import td_system as ts
from tdagw import tdgw_io

HA_IN_EV = tdgw_io.HA_IN_EV
T_AS = tdgw_io.T_IN_ATTOSEC
E_VM = tdgw_io.EFIELD_IN_VPERM
HBAR_EV_AS = tdgw_io.HBAR_IN_EV_AS

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_td_response():
    """``tools/td_response.py`` as a module (tools/ has no package)."""
    name = "lorrax_tools_td_response"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(_REPO, "tools", "td_response.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


@pytest.fixture
def record(record_testsuite_property, request):
    """``record(quantity, value)``: a measured value into the junitxml testsuite properties.

    Keyed ``<test id>:<quantity>``, so ``pytest --junitxml`` carries every
    number the ladder's report table quotes.
    """
    def _rec(key, value):
        record_testsuite_property(f"{request.node.name}:{key}", value)
    return _rec


# ---------------------------------------------------------------------------
# Writing a system and running a deck
# ---------------------------------------------------------------------------

def _structure(cell_bohr):
    cell = np.asarray(cell_bohr, float)
    return dict(rprim=cell, atomic_numbers=np.array([1]),
                atomic_positions_red=np.zeros((1, 3)), fftgrid=np.array([8, 8, 8]))


def write_header(path, system):
    """``system`` as a TDGW header directory (itype 1, no intraband term).

    Energies pass through ``bundle_header`` as Ry; giving e_eV / RYD_TO_EV makes
    the header's Ha energies e_eV / HA_IN_EV, the ones :func:`energies_ha`
    returns, to rounding.
    """
    os.makedirs(path, exist_ok=True)
    hdr = ts.bundle_header(structure=_structure(system["cell"]), k_frac=system["k_frac"],
                           kgrid=system["kgrid"], bands=(0, system["e_ev"].shape[1]),
                           nspinor=system["nspinor"], e_qp_ry=system["e_ev"] / RYD_TO_EV,
                           n_val=system["n_val"], r=system["r"])
    tdgw_io.init_output(os.path.join(path, "tdgw.h5"), hdr, t0=0.0, tf=1.0, mdt=0.5,
                        mtsteps=np.array([0.0, 0.5]), measurements=[])
    return str(path)


def energies_ha(system):
    """The header's band energies in Ha (what the propagator integrates).

    The same arithmetic as ``write_header`` -> ``bundle_header``: (e/Ry) * Ry / Ha.
    """
    return (np.asarray(system["e_ev"], float) / RYD_TO_EV) * RYD_TO_EV / HA_IN_EV


def write_kernel(path, system, *, tda=False, scale=1.0):
    """``system['K_ha']`` (times ``scale``) as a TDGW kernel.h5 the deck's model reads.

    ``write_tdgw_kernel`` takes Ry and TDGW reads K = -kernel/2 Ha, so 2 K_Ha
    is passed: the propagator applies exactly K_Ha.
    """
    from tdagw.kernel_export import write_tdgw_kernel
    write_tdgw_kernel(str(path), 2.0 * scale * np.asarray(system["K_ha"]),
                      kpts=np.asarray(system["k_frac"]), ibndmin=1, ivbm=system["n_val"],
                      sys_dim=2, nspinor=system["nspinor"], tda=tda)
    return str(path)


def run_deck(outdir, header_dir, system, *, trials, tf_as, dt_measure_as, method="dp54",
             dt_as=None, n_corr=2, rtol=1e-10, atol=1e-14, max_step_as=None,
             kernel_file=None, depopulation_ev=0.0, dephasing_ev=0.0,
             measurements=("Efield", "polarization", "current", "density_matrix"),
             meas_opts=None, model_hook=None):
    """Propagate ``trials`` through the driver's stages; returns (per-trial dicts, stats).

    Each dict maps a measurement name to ``(times_au, data)`` read back from
    that trial's ``tdgw.h5``.  ``meas_opts`` {measurement name: TDGW keywords}
    (e.g. ``{"density_matrix": {"itrange": [0, n, 10]}}``).
    ``model_hook(model) -> model`` may replace the self-energy model (a red twin).
    """
    import h5py
    from common.collectives import resolve_mesh
    from tdagw import stages
    from tdagw.td_config import parse_td_deck

    nb = int(system["e_ev"].shape[1])
    lorrax = {"method": method, "n_corr": int(n_corr), "intraband": False, "trials": trials}
    if dt_as is not None:
        lorrax["dt"] = float(dt_as)
    if kernel_file:
        lorrax["kernel_file"] = str(kernel_file)
        lorrax["model"] = "tdgw_kernel"
    ts_tab = {"t0": 0, "tf": float(tf_as), "dt_measure": float(dt_measure_as),
              "ntcheck": 100000}
    if max_step_as is not None:
        ts_tab["max_step"] = float(max_step_as)
    raw = {"control": {"calc_type": "COHSEX" if kernel_file else "IP", "timesteps": ts_tab,
                       "ivp": {"rtol": float(rtol), "atol": float(atol)},
                       "files": {"outdir": str(outdir), "load_header_from": str(header_dir)}},
           "system": {"ibndmin": 1, "ibndmax": nb, "ivbm": int(system["n_val"]),
                      "geometry": "slab", "depopulation": float(depopulation_ev),
                      "dephasing": float(dephasing_ev)},
           "measurements": {"types": list(measurements)},
           "lorrax": lorrax}
    for i, name in enumerate(measurements):
        if meas_opts and name in meas_opts:
            raw["measurements"][str(i)] = dict(meas_opts[name])
    deck = parse_td_deck(raw, source=str(outdir))
    mesh = resolve_mesh()
    sysd = stages.load_system(deck, mesh)
    model = stages.build_self_energy(deck, sysd, mesh)
    if model_hook is not None:
        model = model_hook(model)
    outs = stages.open_outputs(deck, sysd)
    stats = stages.propagate(deck, sysd, model, tdgw_io.efield_function(deck.trials), outs,
                             mesh, say=lambda s: None)
    res = []
    for path in outs.paths:
        with h5py.File(path, "r") as f:
            td = f["tdgw_data/time_domain"]
            res.append({k: (td[k]["times"][()], td[k]["data"][()]) for k in td})
    stats["mesh"] = tuple(int(n) for n in mesh.devices.shape)
    stats["nk_pad"] = int(sysd.nk_pad)
    return res, stats


def propagate_state(header_dir, system, rho0, *, tf_as, dt_measure_as, rtol=1e-12,
                    atol=1e-16, max_step_as=None, kernel_file=None, model_hook=None):
    """Propagate host initial states ``rho0`` (ntr, nk, nb, nb) with no field, dp54.

    The deck route always starts from rho_eq; an instantaneous length-gauge
    kick rho(0+) = e^{-i kappa r} rho_eq e^{i kappa r} needs the propagator's
    own entry point, ``tdagw.propagate.run_propagation``, with the system and
    the self-energy model built by the driver's stages from the same deck.
    Returns (t (nt,), {"polarization": (nt, ntr, 3), "current": ...}, stats).
    """
    import jax.numpy as jnp
    from common.collectives import resolve_mesh
    from tdagw import propagate as tp
    from tdagw import stages
    from tdagw.td_config import parse_td_deck

    nb = int(system["e_ev"].shape[1])
    lorrax = {"method": "dp54", "intraband": False}
    if kernel_file:
        lorrax.update(kernel_file=str(kernel_file), model="tdgw_kernel")
    ts_tab = {"t0": 0, "tf": float(tf_as), "dt_measure": float(dt_measure_as)}
    if max_step_as is not None:
        ts_tab["max_step"] = float(max_step_as)
    raw = {"control": {"calc_type": "COHSEX" if kernel_file else "IP", "timesteps": ts_tab,
                       "ivp": {"rtol": float(rtol), "atol": float(atol)},
                       "files": {"outdir": "unused", "load_header_from": str(header_dir)}},
           "system": {"ibndmin": 1, "ibndmax": nb, "ivbm": int(system["n_val"]),
                      "geometry": "slab"},
           "measurements": {"types": ["polarization", "current"]}, "lorrax": lorrax}
    deck = parse_td_deck(raw, source="propagate_state")
    v = deck.values
    mesh = resolve_mesh()
    sysd = stages.load_system(deck, mesh)
    model = stages.build_self_energy(deck, sysd, mesh)
    if model_hook is not None:
        model = model_hook(model)
    rho0 = np.asarray(rho0, complex)
    ntr = rho0.shape[0]
    host = np.zeros((ntr, sysd.nk_pad, nb, nb), complex)
    host[:, :sysd.nk] = rho0
    mtsteps = np.arange(v["control.timesteps.t0"], v["control.timesteps.tf"]
                        - 1e-8 * v["control.timesteps.dt_measure"],
                        v["control.timesteps.dt_measure"])
    # dp54 from the deck's controller; only the fields this PropagatorConfig
    # declares (n_sub went away with the fixed-step sampling rework)
    want = dict(method="dp54", t0=v["control.timesteps.t0"], tf=v["control.timesteps.tf"],
                dt_measure=v["control.timesteps.dt_measure"],
                dt=v["control.timesteps.dt_measure"], n_sub=1, n_corr=2,
                atol=v["control.ivp.atol"], rtol=v["control.ivp.rtol"],
                max_step=v["control.timesteps.max_step"], chunk=mtsteps.size,
                remainder=False)
    cfg = tp.PropagatorConfig(**{k: x for k, x in want.items()
                                 if k in tp.PropagatorConfig._fields})
    got = {}

    def on_chunk(i0, i1, samples, rho_fn, t_state):
        rho_fn()
        for k, val in samples.items():
            got[k] = np.asarray(val)

    stats, _ = tp.run_propagation(
        cfg, stages.sample_spec(deck, sysd), tp.place_state(host, mesh), sysd.ops, model,
        lambda t: jnp.zeros((ntr, 3)), mesh, cell_volume=sysd.cell_volume,
        nspinor=sysd.nspinor, count=ntr * sysd.nk * nb * nb, mtsteps=mtsteps,
        on_chunk=on_chunk)
    return mtsteps, got, stats


def kicked_states(system, kappas, direction):
    """rho(0+) = e^{-i kappa r_a} rho_eq e^{i kappa r_a} per k, one state per kappa."""
    from scipy.linalg import expm
    r = np.asarray(system["r"])[direction]
    nk, nb = r.shape[0], r.shape[1]
    req = np.diag((np.arange(nb) < system["n_val"]).astype(float))
    out = np.zeros((len(kappas), nk, nb, nb), complex)
    for i, kap in enumerate(kappas):
        for k in range(nk):
            u = expm(-1j * kap * r[k])
            out[i, k] = u @ req @ u.conj().T
    return out


def circular_trial(e0_vm, omega_ev, *, sign=+1, envelope=None):
    """TDGW Efunc strings of E0 env(t) (cos wt, sign sin wt, 0), t in as.

    sign = +1 is sigma+ (x -> y, counter-clockwise from +z).  ``envelope`` is a
    TDGW expression in t (as) or None (continuous wave from t0).
    """
    w = float(omega_ev) / HBAR_EV_AS
    env = f"{float(e0_vm)!r}" if envelope is None else f"{float(e0_vm)!r} * ({envelope})"
    return {"E_x": f"{env} * np.cos({w!r} * t)",
            "E_y": f"{int(sign)} * {env} * np.sin({w!r} * t)", "E_z": 0}


def gaussian_env(tc_as, tau_as):
    return f"np.exp(-((t - {float(tc_as)!r}) / {float(tau_as)!r})**2)"


def pulse_au(tdr, e0_vm, omega_ev, tc_as, tau_as, pol):
    """The tools' Pulse of a Gaussian drive written by :func:`trial_for`.

    pol: 'x', 'y', '+' (sigma+), '-' (sigma-).  E = E0 g(t) Re[a e^{-i w t}].
    """
    a = {"x": (1, 0, 0), "y": (0, 1, 0), "+": (1, 1j, 0), "-": (1, -1j, 0)}[pol]
    return tdr.Pulse(a=np.asarray(a, complex) * (e0_vm / E_VM),
                     w=(omega_ev / HBAR_EV_AS) * T_AS, tc=tc_as / T_AS, tau=tau_as / T_AS)


def trial_for(e0_vm, omega_ev, tc_as, tau_as, pol):
    """TDGW Efunc of the same Gaussian drive (cos/sin carriers, t in as)."""
    env = f"{float(e0_vm)!r} * {gaussian_env(tc_as, tau_as)}"
    w = float(omega_ev) / HBAR_EV_AS
    cos, sin = f"np.cos({w!r} * t)", f"np.sin({w!r} * t)"
    if pol == "x":
        return {"E_x": f"{env} * {cos}", "E_y": 0, "E_z": 0}
    if pol == "y":
        return {"E_x": 0, "E_y": f"{env} * {cos}", "E_z": 0}
    s = 1 if pol == "+" else -1
    return {"E_x": f"{env} * {cos}", "E_y": f"{s} * {env} * {sin}", "E_z": 0}


# ---------------------------------------------------------------------------
# The ISDF-form interaction, by explicit sums (shares nothing with tdagw.model)
# ---------------------------------------------------------------------------

def isdf_form_kernel(psi, vbar, w_kk, *, w_s, hartree=True, sex=True):
    """K = K^H - K^W of docs/theory/td_agw.md by explicit sums, Ha.

    psi : (nk, nb, ns, nmu) — psi_{nk,s}(mu).  vbar : (nmu, nmu) real symmetric.
    w_kk : (nk, nk, nmu, nmu) — W_{mu nu}(k - k') per (k, k').
    K^H_{knm,k'n'm'} = (w_s/nk) sum M_{k,nm}(mu) vbar_{mu nu} conj(M_{k',n'm'}(nu)),
    K^W_{knm,k'n'm'} = (1/nk) sum conj(psi_{nk,t}(mu)) psi_{n'k',t}(mu) W_{mu nu}(k,k')
                       conj(psi_{m'k',s}(nu)) psi_{mk,s}(nu).
    Returns (nk, nb, nb, nk, nb, nb) complex.
    """
    nk = psi.shape[0]
    K = 0.0
    if hartree:
        M = np.einsum("knsu,kmsu->knmu", np.conj(psi), psi)
        K = K + (w_s / nk) * np.einsum("knmu,uv,KNMv->knmKNM", M, vbar, np.conj(M))
    if sex:
        A = np.einsum("kntu,KNtu->kKnNu", np.conj(psi), psi)          # (k, k', n, n', mu)
        B = np.einsum("KMsv,kmsv->kKMmv", np.conj(psi), psi)          # (k, k', m', m, nu)
        K = K - np.einsum("kKnNu,kKuv,kKMmv->knmKNM", A, w_kk, B) / nk
    return K


def _disk_vectors(cell2, rmax):
    """Lattice vectors R = i a1 + j a2 covering a disk of radius rmax (+ one cell)."""
    amin = min(np.linalg.norm(cell2[0]), np.linalg.norm(cell2[1]))
    n = int(np.ceil(rmax / amin)) + 2
    i, j = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    return i.ravel()[:, None] * cell2[0] + j.ravel()[:, None] * cell2[1]


def _lattice_sum(f, dvec, cell2, *, rcut):
    """sum_R f(|R + d|) over |R + d| <= rcut (a disk in the displacement: C3-invariant)."""
    R = _disk_vectors(cell2, rcut)
    x = np.linalg.norm(dvec[..., None, :] + R, axis=-1)
    return np.sum(np.where(x <= rcut, f(x), 0.0), axis=-1)


def _w_of_q(f, tau, cell2, q_cart, *, rcut):
    """W_{mu nu}(q) = sum_R f(|tau_mu - tau_nu + R|) e^{-i q.R}, |tau_mu - tau_nu + R| <= rcut.

    The Bloch representation that matches centroid amplitudes carrying their
    phase, psi_nk(tau) = e^{i k.tau} u_nk(tau): the direct integral
    sum_{R1 R2} psi* psi W psi* psi collapses to this W at q = k - k'.  f real
    and decaying: W(q) is Hermitian and W(-q) = W(q)^T (reciprocity), the two
    properties dSigma's Hermiticity and energy conservation need.  The cutoff
    is a disk in the actual displacement, a set every lattice rotation maps
    onto itself, so the truncation keeps the point group (a parallelogram of
    R breaks C3 at the tail).
    """
    d = tau[:, None, :] - tau[None, :, :]                          # (mu, nu, 2)
    R = _disk_vectors(cell2, rcut)                                 # (nR, 2)
    x = np.linalg.norm(d[:, :, None, :] + R[None, None], axis=-1)  # (mu, nu, nR)
    fx = np.where(x <= rcut, f(x), 0.0)
    ph = np.exp(-1j * (q_cart @ R.T))                              # (..., nR)
    return np.einsum("...r,uvr->...uv", ph, fx)


# ---------------------------------------------------------------------------
# System 1: two levels, circular dipole
# ---------------------------------------------------------------------------

def two_level(*, gaps_ev=(1.95, 1.98, 2.0, 2.02, 2.06), d_bohr=1.0):
    """nk copies of a sigma+ two-level transition, E_v = 0, E_c = gap_k.

    <c|r|v> = d (1, -i, 0)/sqrt2: <c|x + iy|v> = sqrt2 d, <c|x - iy|v> = 0, so
    I+ = d^2, I- = 0 (D = <v|r|c>).  A sigma+ drive E0 (cos wt, sin wt, 0)
    couples H_cv = g e^{-i w t}, g = E0 d/sqrt2; sigma- couples g e^{+i w t}.
    """
    gaps = np.asarray(gaps_ev, float)
    nk = gaps.size
    e = np.zeros((nk, 2))
    e[:, 1] = gaps
    r = np.zeros((3, nk, 2, 2), complex)
    r[:2, :, 1, 0] = (d_bohr / np.sqrt(2) * np.array([1.0, -1j]))[:, None]
    r = r + np.conj(np.swapaxes(r, -1, -2))
    k_frac = np.stack([np.arange(nk) / nk, np.zeros(nk), np.zeros(nk)], axis=1)
    return dict(name="two_level", e_ev=e, n_val=1, r=r, nspinor=2, k_frac=k_frac,
                kgrid=(nk, 1, 1), cell=np.diag([8.0, 8.0, 30.0]), K_ha=None, d=d_bohr)


def rabi_exact(gap_ha, g_ha, omega_ha, t, *, sign=+1, gamma1=0.0, gamma2=0.0):
    """Exact rho(t) (nt, 2, 2) of one circular two-level copy from rho = |v><v|.

    H(t) = [[0, g e^{s i w t}], [g e^{-s i w t}, gap]] in (v, c), s = sign (+1:
    sigma+).  T(t) = diag(1, e^{-s i w t}) makes H_rot = [[0, g], [g, gap - s w]]
    time independent; relaxation -Gamma o (rho - rho_eq) (Gamma_vv = Gamma_cc =
    gamma1, Gamma_cv = gamma2) commutes with T, so rho_rot obeys a constant
    affine linear ODE, solved by one matrix exponential of the augmented
    generator (exact to rounding).
    """
    from scipy.linalg import expm
    h = np.array([[0.0, g_ha], [g_ha, gap_ha - sign * omega_ha]], complex)
    gam = np.array([[gamma1, gamma2], [gamma2, gamma1]])
    req = np.diag([1.0, 0.0]).astype(complex)
    eye = np.eye(2)
    # vec (row-major): d vec(rho)/dt = A vec(rho) + b
    A = -1j * (np.kron(h, eye) - np.kron(eye, h.T)) - np.diag(gam.ravel())
    b = (gam * req).ravel()
    G = np.zeros((5, 5), complex)
    G[:4, :4], G[:4, 4] = A, b
    x0 = np.concatenate([req.ravel(), [1.0]])
    out = np.empty((len(t), 2, 2), complex)
    for i, ti in enumerate(np.asarray(t, float)):
        rr = (expm(G * ti) @ x0)[:4].reshape(2, 2)
        ph = np.exp(-1j * sign * omega_ha * ti)
        rr[1, 0] *= ph
        rr[0, 1] *= np.conj(ph)
        out[i] = rr
    return out


def rabi_pulse_reference(gap_ha, g_of_t, omega_ha, t, *, gamma1=0.0, gamma2=0.0):
    """rho(t) of a sigma+ Gaussian-enveloped drive g(t) e^{-i w t}, rotating frame, DOP853 rtol 1e-13.

    At resonance and Gamma = 0 it reduces to the area theorem
    rho_cc = sin^2(int g dt); with detuning or relaxation no closed form
    exists and this independent integration (SciPy, no shared code) is the
    exact Bloch-equation reference to its tolerance.
    """
    from scipy.integrate import solve_ivp
    gam = np.array([[gamma1, gamma2], [gamma2, gamma1]])
    req = np.diag([1.0, 0.0]).astype(complex)

    def f(tt, y):
        rho = y.reshape(2, 2)
        g = g_of_t(tt)
        h = np.array([[0.0, g], [g, gap_ha - omega_ha]], complex)
        return (-1j * (h @ rho - rho @ h) - gam * (rho - req)).ravel()

    t = np.asarray(t, float)
    sol = solve_ivp(f, (0.0, t[-1]), req.ravel(), t_eval=t, method="DOP853",
                    rtol=1e-13, atol=1e-15)
    rr = sol.y.T.reshape(-1, 2, 2).copy()
    ph = np.exp(-1j * omega_ha * t)
    rr[:, 1, 0] *= ph
    rr[:, 0, 1] *= np.conj(ph)
    return rr


# ---------------------------------------------------------------------------
# System 2: a 3v3c spinor window with an ISDF-form full kernel
# ---------------------------------------------------------------------------

def synthetic_window(*, seed=20261016, n=3, nv=3, nc=3, nmu=8, ns=2, v0_ev=1.0,
                     w0_ev=0.6, lam_bohr=4.0, kernel_scale=1.0):
    """3v3c spinors on an n x n hexagonal grid with Hartree + SEX of ISDF form.

    Centroids at random in-plane positions tau_mu; psi_{nk,s}(mu) =
    e^{i k.tau_mu} c_{nk,s}(mu) with random c; bare V(r) = v0 e^{-r/lam}
    (lattice sum at q = 0, uniform component projected out -> V-bar), screened
    W(r) = w0 e^{-r/lam} (W(q) of the full displacement).  Energies: valence
    [-1.4, -0.4] eV, conduction [1.3, 2.6] eV; r random Hermitian, zero diagonal.
    """
    rng = np.random.default_rng(seed)
    a = 7.0
    cell2 = np.array([[a, 0.0], [a / 2, a * np.sqrt(3) / 2]])
    nb, nk = nv + nc, n * n
    ij = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    k_frac = np.concatenate([ij / n, np.zeros((nk, 1))], axis=1)
    bvec = 2 * np.pi * np.linalg.inv(cell2).T                       # rows b1, b2
    k_cart = k_frac[:, :2] @ bvec
    tau = rng.uniform(0, 1, (nmu, 2)) @ cell2
    c = rng.standard_normal((nk, nb, ns, nmu)) + 1j * rng.standard_normal((nk, nb, ns, nmu))
    c /= np.sqrt(np.sum(np.abs(c) ** 2, axis=(2, 3), keepdims=True))
    psi = c * np.exp(1j * k_cart @ tau.T)[:, None, None, :]
    v_r = lambda d: (v0_ev / HA_IN_EV) * np.exp(-d / lam_bohr)        # noqa: E731
    w_r = lambda d: (w0_ev / HA_IN_EV) * np.exp(-d / lam_bohr)        # noqa: E731
    rcut = 12.0 * lam_bohr
    v0 = _lattice_sum(v_r, tau[:, None, :] - tau[None, :, :], cell2, rcut=rcut)
    Q = np.eye(nmu) - np.full((nmu, nmu), 1.0 / nmu)
    vbar = Q @ v0 @ Q
    q = k_cart[:, None, :] - k_cart[None, :, :]                     # (k, k', 2)
    wkk = _w_of_q(w_r, tau, cell2, q, rcut=rcut)
    K = kernel_scale * isdf_form_kernel(psi, vbar, wkk, w_s=1.0 if ns == 2 else 2.0)
    ev = np.sort(rng.uniform(-1.4, -0.4, (nk, nv)), axis=1)
    ec = np.sort(rng.uniform(1.3, 2.6, (nk, nc)), axis=1)
    e = np.concatenate([ev, ec], axis=1)
    r = rng.standard_normal((3, nk, nb, nb)) + 1j * rng.standard_normal((3, nk, nb, nb))
    r = 0.4 * (r + np.conj(np.swapaxes(r, -1, -2)))
    r[2] = 0.0
    r[:, :, np.arange(nb), np.arange(nb)] = 0.0
    cell = np.array([[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 25.0]])
    return dict(name="synthetic_window", e_ev=e, n_val=nv, r=r, nspinor=ns, k_frac=k_frac,
                kgrid=(n, n, 1), cell=cell, K_ha=K, psi=psi, vbar=vbar, wkk=wkk)


# ---------------------------------------------------------------------------
# System 3: the C3-symmetric Haldane honeycomb with the same interaction
# ---------------------------------------------------------------------------

def haldane(*, n=6, t1_ev=1.0, t2_ev=0.03, phi=np.pi / 2, mass_ev=0.95, a_bohr=4.6,
            v0_ev=2.0, w0_ev=1.2, lam_bohr=3.0, interacting=True):
    """Spinless Haldane model, H(k) in the actual-position gauge, one band pair.

    Orbitals A at (a1 + a2)/3, B at 2(a1 + a2)/3; H_AB = t1 sum_j e^{i k.delta_j},
    H_AA = M + 2 t2 sum_j cos(k.b_j + phi), H_BB = -M + 2 t2 sum_j cos(k.b_j - phi),
    b_j = a1, a2 - a1, -a2.  Because every phase is e^{i k.(actual bond)}, C3
    about an A site gives H(C3 k) = H(k), so on the n x n grid (n divisible by
    3: K and K' are grid points) every exciton carries a C3 eigenvalue and
    its dipole is pure sigma+ or sigma-.  The gap at K and K' is
    2 |M -+ 3 sqrt3 t2 sin phi|: the two valleys absorb opposite helicities at
    different energies.  r_nm = <n|dH/dk|m> / (i (E_n - E_m)); spinless (g_s = 2,
    w_s = 2).
    """
    if n % 3:
        raise ValueError("haldane: n must be divisible by 3 (K, K' on the grid)")
    a1 = np.array([1.0, 0.0]) * a_bohr
    a2 = np.array([0.5, np.sqrt(3) / 2]) * a_bohr
    cell2 = np.stack([a1, a2])
    tA, tB = (a1 + a2) / 3, 2 * (a1 + a2) / 3
    delta = np.stack([tB - tA, tB - tA - a1, tB - tA - a2])
    bnn = np.stack([a1, a2 - a1, -a2])
    t1, t2, M = (x / HA_IN_EV for x in (t1_ev, t2_ev, mass_ev))
    nk = n * n
    ij = np.stack(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 2)
    k_frac = np.concatenate([ij / n, np.zeros((nk, 1))], axis=1)
    bvec = 2 * np.pi * np.linalg.inv(cell2).T
    kc = k_frac[:, :2] @ bvec                                        # (nk, 2)

    def hk(k):
        h = np.zeros((2, 2), complex)
        h[0, 0] = M + 2 * t2 * np.sum(np.cos(bnn @ k + phi))
        h[1, 1] = -M + 2 * t2 * np.sum(np.cos(bnn @ k - phi))
        h[0, 1] = t1 * np.sum(np.exp(1j * delta @ k))
        h[1, 0] = np.conj(h[0, 1])
        return h

    def dhk(k):
        out = np.zeros((2, 2, 2), complex)                            # (cart, A/B, A/B)
        for c in range(2):
            out[c, 0, 0] = -2 * t2 * np.sum(bnn[:, c] * np.sin(bnn @ k + phi))
            out[c, 1, 1] = -2 * t2 * np.sum(bnn[:, c] * np.sin(bnn @ k - phi))
            out[c, 0, 1] = t1 * np.sum(1j * delta[:, c] * np.exp(1j * delta @ k))
            out[c, 1, 0] = np.conj(out[c, 0, 1])
        return out

    e = np.zeros((nk, 2))
    cvec = np.zeros((nk, 2, 2), complex)                              # (k, band, orbital)
    r = np.zeros((3, nk, 2, 2), complex)
    for ik in range(nk):
        w, v = np.linalg.eigh(hk(kc[ik]))
        e[ik] = w
        cvec[ik] = v.T
        vel = np.einsum("an,cab,bm->cnm", np.conj(v), dhk(kc[ik]), v)
        de = w[:, None] - w[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(np.abs(de) > 1e-12, vel / (1j * de), 0.0)
        r[:2, ik] = rr
    tau = np.stack([tA, tB])
    psi = (cvec * np.exp(1j * kc @ tau.T)[:, None, :])[:, :, None, :]  # (k, n, s=1, mu)
    K = None
    if interacting:
        v_r = lambda d: (v0_ev / HA_IN_EV) * np.exp(-d / lam_bohr)    # noqa: E731
        w_r = lambda d: (w0_ev / HA_IN_EV) * np.exp(-d / lam_bohr)    # noqa: E731
        rcut = 12.0 * lam_bohr
        v0 = _lattice_sum(v_r, tau[:, None, :] - tau[None, :, :], cell2, rcut=rcut)
        Q = np.eye(2) - 0.5
        vbar = Q @ v0 @ Q
        wkk = _w_of_q(w_r, tau, cell2, kc[:, None, :] - kc[None, :, :], rcut=rcut)
        K = isdf_form_kernel(psi, vbar, wkk, w_s=2.0)
    cell = np.array([[a1[0], a1[1], 0], [a2[0], a2[1], 0], [0, 0, 20.0]])
    return dict(name="haldane", e_ev=e * HA_IN_EV, n_val=1, r=r, nspinor=1, k_frac=k_frac,
                kgrid=(n, n, 1), cell=cell, K_ha=K, psi=psi,
                gap_K_ev=2 * abs(mass_ev - 3 * np.sqrt(3) * t2_ev * np.sin(phi)),
                gap_Kp_ev=2 * abs(mass_ev + 3 * np.sqrt(3) * t2_ev * np.sin(phi)))


# ---------------------------------------------------------------------------
# Observables on propagated density matrices
# ---------------------------------------------------------------------------

def energy_per_cell(system, rho):
    """E[rho] = g_s sum_k w_k [Tr(eps_k rho_k) + (1/2) Tr(dSigma_k[drho] drho_k)], Ha.

    The functional whose flow is the TD-aGW equation: with K = K^H - K^W and
    K_{knm,k'n'm'} = K_{k'm'n',kmn} (V-bar symmetric, W reciprocal),
    dE/dt = g_s sum_k w_k Tr((eps + dSigma) d rho/dt) = Omega E(t).j(t).  The
    1/2 is the Hartree and screened-exchange double counting.
    rho : (nt, nk, nb, nb).  Returns (nt,).
    """
    e = energies_ha(system)
    nk, nb = e.shape
    g_s = 2.0 if system["nspinor"] == 1 else 1.0
    req = np.zeros((nk, nb, nb))
    req[:, np.arange(system["n_val"]), np.arange(system["n_val"])] = 1.0
    d = np.asarray(rho) - req[None]
    band = np.einsum("kn,tknn->t", e, np.asarray(rho)).real
    out = band
    if system.get("K_ha") is not None:
        ds = np.einsum("knmKNM,tKNM->tknm", system["K_ha"], d)
        out = out + 0.5 * np.einsum("tknm,tkmn->t", ds, d).real
    return g_s * out / nk


def work_per_cell(t, E, j, cell_volume):
    """Omega int_0^t E.j dt' on the samples (composite Simpson on even sub-intervals, trapezoid else)."""
    from scipy.integrate import cumulative_simpson
    p = np.einsum("ta,ta->t", np.asarray(E), np.asarray(j).real)
    return cell_volume * np.concatenate([[0.0], cumulative_simpson(p, x=t)])
