"""tdagw/propagate.py — the jitted, k-sharded propagator of the TD-aGW EOM.

Integrates Eq. (1) of :mod:`tdagw.eom` with one of five integrators of
:mod:`solvers.time_integrators`, chosen by name:

========  ==========================================================  =========
method    scheme                                                      dSigma/step
========  ==========================================================  =========
dp54      SciPy RK45 replica, adaptive, quartic dense output          6 per try
rk4       classical RK4, fixed step                                   4
etdrk4    Krogstad ETDRK4; -i[H0, .] - Gamma exact (elementwise)       4
magnus2   exponential midpoint on Delta rho, interaction picture      n_corr
cfm4      4th-order commutator-free Magnus on Delta rho, interaction  2 n_corr
          picture; Gauss-node dSigma by cubic extrapolation + n_corr
          corrections; Gamma / intraband by the triple jump (order 4)
========  ==========================================================  =========

The Magnus steps propagate Delta rho = rho - rho^eq in the interaction
picture of H0 = diag(E^QP): the band phases are exact elementwise factors
and every exponential is exp(-i tau W) of the coupling W = E·r + dSigma
only, applied as

    Delta rho' = U Delta rho U^H + (U - 1) rho^eq + rho^eq (U - 1)^H
                 + (U - 1) rho^eq (U - 1)^H,                U = exp(-i tau W),

with U - 1 = V (e^{-i tau w} - 1) V^H from ``jnp.linalg.eigh`` of the local k
block inside a ``shard_map`` (no device holds another's matrices), so each
term has the size of Delta rho and a weak-field run rounds relative to the
induced coherence, not to the O(1) occupations.  Dissipation (Gamma) and
the intraband commutator are not of the form [H_k, .]: for ``cfm4`` they
enter by Yoshida's triple jump of the Strang step (fourth order; dSigma is
frozen between the Gauss nodes, so the composition costs exponentials and
intraband gathers, not dSigma), for ``magnus2`` by Strang (second order);
ETDRK4 carries Gamma exactly in its linear part.

Samples are taken on the measurement grid from each method's dense output,
so ``dt`` need not divide ``dt_measure``: DP54 SciPy's quartic interpolant;
RK4 and ETDRK4 a cubic Hermite in the frame of the band energies and Gamma
(derivatives carried first-same-as-last); CFM4 the step's own flow with
the final frozen dSigma over [t, t + theta h].

Layout: rho, H and the k-dependent operators are k-sharded over the whole
mesh, ``P(None, ('x','y'), None, None)`` for ``(ntr, nk, nb, nb)``; nk is
padded to a multiple of the device count with zero operators and zero
weight, which every term of Eq. (1) leaves at zero.  dSigma is the injected
:class:`tdagw.self_energy.SelfEnergyModel` and may move data however it likes.
A chunk of samples is ONE jitted call (a ``lax`` loop over steps, samples
written into device buffers); the host sees the state only between chunks,
and each chunk's checkpoint carries the integrator's whole carry, so a run
restarted from it continues bitwise.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local, gather_to_host
from runtime.padding import round_up
from solvers import time_integrators as ti
from tdagw import eom
from tdagw.self_energy import SelfEnergyModel

__all__ = ["METHODS", "dsigma_per_step", "PropagatorConfig", "SampleSpec",
           "place_operators", "place_kernel", "place_state", "make_conj_dev",
           "sample_plan", "chunk_plan", "initial_carry", "carry_to_host",
           "carry_from_host", "coupling_radius", "magnus_dt_limit", "run_propagation"]

METHODS = ("dp54", "rk4", "etdrk4", "magnus2", "cfm4")
#: Bootstrap: while the predictor history is incomplete (first steps) the
#: Magnus steps take this many extra corrections.
BOOTSTRAP_EXTRA = 3
K_AXES = ("x", "y")
#: A sample within this fraction of a step from a step end is ON the step.
_ON_STEP_TOL = 1e-9
#: Power iterations of L(L(x)) in :func:`coupling_radius` (two dSigma calls each).
COUPLING_POWER_ITERATIONS = 20
#: The step check takes h (MARGIN * sigma) <= the stability limit: the power
#: iteration approaches sigma from below.
STABILITY_MARGIN = 1.1


def dsigma_per_step(method: str, n_corr: int) -> int:
    """dSigma evaluations per accepted fixed step (dp54: per try, 6)."""
    return {"dp54": 6, "rk4": 4, "etdrk4": 4, "magnus2": n_corr,
            "cfm4": 2 * n_corr}[method]


class PropagatorConfig(NamedTuple):
    """Static integration settings, Hartree atomic units.

    method : one of METHODS.
    t0, tf, dt_measure : run window and sample spacing (TDGW t0, tf, mdt).
    dt : fixed step (non-dp54); need not divide dt_measure.
    n_corr : Magnus corrections per step (>= 1).
    atol, rtol, max_step : dp54 controller (SciPy semantics; max_step inf ok).
    chunk : samples per jitted call.
    remainder : the non-commutator flow (Gamma or intraband) is nonzero; a
        purely Hamiltonian run skips it (Magnus steps exactly unitary).
    predictor : cfm4's dSigma predictor, "cubic" (the last two steps' four
        node values) or "linear" (the last step's two).
    """
    method: str
    t0: float
    tf: float
    dt_measure: float
    dt: float
    n_corr: int
    atol: float
    rtol: float
    max_step: float
    chunk: int
    remainder: bool
    predictor: str = "cubic"


class SampleSpec(NamedTuple):
    """What one sample measures (TDGW measurement workers).

    kinds : tuple of (name, kind, opts) with kind in Efield, polarization,
        current, density_matrix, lessG, carrier_density, exciton_projection
        and opts a tuple of (key, value).
    """
    kinds: tuple


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def _spec(ndim: int, k_axis: int):
    axes = [None] * ndim
    axes[k_axis] = K_AXES
    return P(*axes)


def _place(host, mesh, k_axis: Optional[int]):
    host = np.asarray(host)
    spec = P() if k_axis is None else _spec(host.ndim, k_axis)
    return device_put_process_local(host, NamedSharding(mesh, spec))


def _pad_k(a, k_axis: int, nk_pad: int, fill=0):
    a = np.asarray(a)
    pad = nk_pad - a.shape[k_axis]
    if pad == 0:
        return a
    widths = [(0, 0)] * a.ndim
    widths[k_axis] = (0, pad)
    return np.pad(a, widths, constant_values=fill)


def place_operators(host: dict, mesh) -> tuple:
    """Host operator arrays -> :class:`tdagw.eom.TdOperators` on ``mesh``.

    ``host`` keys (logical nk): e_qp (nk, nb), f_kn (nk, nb), r_inter
    (3, nk, nb, nb), gamma (nb, nb), w_k (nk,), v0_inter, v0_intra or None,
    intra: None or dict(stilde (nbvec, nk, nb, nb), kmap (nbvec, nk),
    wb (3, nbvec), rintra (3, nk, nb) or None).

    Returns (ops, nk_pad).  nk is padded to a multiple of the device count;
    a pad k has zero operators, zero weight and maps to itself under k+b.
    """
    ndev = int(mesh.devices.size)
    nk, nb = np.asarray(host["e_qp"]).shape
    nk_pad = round_up(nk, ndev)
    rho_eq = np.zeros((nk, nb, nb), complex)
    idx = np.arange(nb)
    rho_eq[:, idx, idx] = np.asarray(host["f_kn"])
    intra = None
    if host.get("intra") is not None:
        it = host["intra"]
        kmap = np.asarray(it["kmap"], np.int32)
        kmap_pad = np.concatenate(
            [kmap, np.broadcast_to(np.arange(nk, nk_pad, dtype=np.int32),
                                   (kmap.shape[0], nk_pad - nk))], axis=1)
        intra = eom.IntraBand(
            stilde=_place(_pad_k(it["stilde"], 1, nk_pad), mesh, 1),
            kmap=_place(kmap_pad, mesh, None),
            wb=_place(np.asarray(it["wb"], float), mesh, None),
            rintra=(None if it.get("rintra") is None else
                    _place(_pad_k(np.asarray(it["rintra"], float), 1, nk_pad), mesh, 1)))
    v0_intra = host.get("v0_intra")
    ops = eom.TdOperators(
        e_qp=_place(_pad_k(host["e_qp"], 0, nk_pad), mesh, 0),
        rho_eq=_place(_pad_k(rho_eq, 0, nk_pad), mesh, 0),
        r_inter=_place(_pad_k(np.asarray(host["r_inter"], complex), 1, nk_pad), mesh, 1),
        gamma=_place(np.asarray(host["gamma"], float), mesh, None),
        w_k=_place(_pad_k(host["w_k"], 0, nk_pad), mesh, 0),
        v0_inter=_place(_pad_k(np.asarray(host["v0_inter"], complex), 1, nk_pad), mesh, 1),
        v0_intra=(None if v0_intra is None else
                  _place(_pad_k(np.asarray(v0_intra, complex), 1, nk_pad), mesh, 1)),
        intra=intra)
    return ops, nk_pad


def place_projector(A, mesh, nk_pad: int):
    """BSE eigenvectors (n_states, nk, nc, nv) on the mesh, k-sharded and padded."""
    return _place(_pad_k(np.asarray(A, complex), 1, nk_pad), mesh, 1)


def place_kernel(kmat, mesh):
    """A dense (N, N) kernel, rows over the whole mesh (N padded by caller)."""
    return _place(kmat, mesh, 0)


def place_state(rho, mesh):
    """(ntr, nk_pad, nb, nb) host density -> the k-sharded state."""
    return _place(rho, mesh, 1)


# ---------------------------------------------------------------------------
# The unitary exponential on Delta rho, per k, inside shard_map
# ---------------------------------------------------------------------------

def make_conj_dev(mesh) -> Callable:
    """conj_dev(W, x, tau) = e^{-i tau W} (rho^eq + x) e^{+i tau W} - rho^eq.

    W, x : (ntr, nk, nb, nb) k-sharded; f : (nk, nb) occupations, rho^eq =
    diag(f); tau : scalar.  Each device diagonalizes its own k block
    (``jnp.linalg.eigh``, W Hermitized first), U = V e^{-i tau w} V^H and
    G = U - 1 = V (e^{-i tau w} - 1) V^H with e^{-ia} - 1 = -2 sin^2(a/2) -
    i sin(a) (no cancellation), and returns

        U x U^H + G rho^eq + rho^eq G^H + G rho^eq G^H,

    the deviation of the conjugated state, every term of the size of x or
    of tau |W|.  A pad k has W = 0, x = 0, f = 0 and stays 0.
    """
    spec4, spec2 = _spec(4, 1), _spec(2, 0)

    def local(w, x, f, tau):
        w = 0.5 * (w + jnp.conj(jnp.swapaxes(w, -1, -2)))
        e, v = jnp.linalg.eigh(w)
        vh = jnp.conj(jnp.swapaxes(v, -1, -2))
        ph = jnp.exp(-1j * tau * e)
        em1 = -2.0 * jnp.sin(0.5 * tau * e) ** 2 - 1j * jnp.sin(tau * e)
        g = v @ (em1[..., :, None] * vh)
        ux = v @ ((vh @ x @ v) * (ph[..., :, None] * jnp.conj(ph)[..., None, :])) @ vh
        gf = g * f[None, :, None, :].astype(g.dtype)
        return ux + gf + jnp.conj(jnp.swapaxes(gf, -1, -2)) + gf @ jnp.conj(jnp.swapaxes(g, -1, -2))

    return shard_map(local, mesh=mesh, in_specs=(spec4, spec4, spec2, P()),
                     out_specs=spec4, check_vma=False)


# ---------------------------------------------------------------------------
# One chunk: the physics callables, the step, the samples
# ---------------------------------------------------------------------------

def _physics(ops, mparams, model: SelfEnergyModel, drive: Callable, mesh):
    """The callables every integrator is built from (traced inside the jit)."""
    spec = NamedSharding(mesh, _spec(4, 1))
    pin = lambda x: jax.lax.with_sharding_constraint(x, spec)
    conj = make_conj_dev(mesh)
    occ = jnp.real(jnp.diagonal(ops.rho_eq, axis1=-2, axis2=-1))      # (nk, nb)
    delta = ops.e_qp[:, :, None] - ops.e_qp[:, None, :]                 # (nk, nb, nb)

    def dsig(t, rho):
        return pin(model.apply(mparams, rho, t))

    def rhs(t, rho):
        return pin(eom.eom_rhs(ops, drive(t), dsig(t, rho), rho))

    def field(t):
        """E(t)·r^inter, Schrödinger picture (the cheap part of the coupling)."""
        return pin(eom.field_coupling(ops, drive(t)))

    def free(m, s):
        """e^{-i s H0} m e^{+i s H0}: (m)_nm e^{-i s (E_n - E_m)}, exact."""
        return jnp.exp((-1j * s) * delta).astype(m.dtype) * m

    lam = (-1j * delta - ops.gamma[None].astype(jnp.complex128))[None]  # (1, nk, nb, nb)

    def nonlin(t, rho):
        e = drive(t)
        drv = eom.commutator(eom.field_coupling(ops, e) + dsig(t, rho), rho)
        if ops.intra is not None:
            drv = drv + eom.field_intraband(ops.intra, e, rho)
        return pin(-1j * drv + ops.gamma.astype(rho.dtype) * ops.rho_eq[None])

    def remainder(t, x, tau):
        """The non-commutator flow on Delta rho: Gamma decay (exact) around intraband (RK4)."""
        decay = jnp.exp(-0.5 * tau * ops.gamma).astype(x.dtype)
        x = decay * x
        if ops.intra is not None:
            f = lambda s, y: -1j * eom.field_intraband(ops.intra, drive(s), ops.rho_eq[None] + y)
            x = ti.rk4_step(f, t, x, tau)
        return pin(decay * x)

    return dict(dsig=dsig, rhs=rhs, field=field, free=free, nonlin=nonlin,
                lam=lam, remainder=remainder, pin=pin,
                conj_dev=lambda w, x, tau: pin(conj(w, x, occ, tau)),
                dsig_dev=lambda t, x: dsig(t, ops.rho_eq[None] + x),
                rho_eq=ops.rho_eq[None])


def _sample(ph, ops, spec: SampleSpec, drive, *, cell_volume, nspinor):
    """sample_fn(t, rho) -> dict of TDGW measurements."""
    def fn(t, rho):
        out = {}
        dsig_cache = None
        for name, kind, opts in spec.kinds:
            o = dict(opts)
            if kind == "Efield":
                out[name] = drive(t)
            elif kind == "polarization":
                out[name] = eom.polarization(ops, rho, cell_volume=cell_volume,
                                             nspinor=nspinor)
            elif kind == "current":
                if dsig_cache is None:
                    dsig_cache = ph["dsig"](t, rho)
                out[name] = eom.current(ops, rho, dsig_cache,
                                        cell_volume=cell_volume, nspinor=nspinor,
                                        use_v0=bool(o.get("use_v0", False)),
                                        inter_only=bool(o.get("inter_only", False)))
            elif kind in ("density_matrix", "lessG"):
                b1, b2 = o["bslice1"], o["bslice2"]
                blk = rho[:, :, b1[0]:b1[1], b2[0]:b2[1]]
                out[name] = blk if kind == "density_matrix" else 1j * blk
            elif kind == "carrier_density":
                out[name] = eom.carrier_density(ops, rho, nspinor=nspinor)
            elif kind == "exciton_projection":
                out[name] = eom.exciton_projection(ops, rho, n_val=int(o["n_val"]))
            else:
                raise ValueError(f"GATE td_measurement_kind: got: {kind!r}; want: one of "
                                 f"the kinds _sample implements; fix: internal (the deck "
                                 f"parser admits only those)")
        return out
    return fn


def _magnus_parts(cfg: PropagatorConfig):
    """(nodes, flow, remainder fractions or None, history offsets, bootstrap steps)."""
    cfm4 = cfg.method == "cfm4"
    nodes = ti.CFM4_NODES if cfm4 else ti.MIDPOINT_NODES
    flow = ti.ip_cf4_flow if cfm4 else ti.ip_midpoint_flow
    fractions = None if not cfg.remainder else (ti.TRIPLE_JUMP if cfm4 else (1.0,))
    hist = ((ti.CFM4_HISTORY if cfg.predictor == "cubic" else ti.CFM4_HISTORY[2:])
            if cfm4 else ti.MIDPOINT_HISTORY)
    return nodes, flow, fractions, hist, -(-len(hist) // len(nodes))


def _fixed_parts(cfg: PropagatorConfig, ph):
    """(step, dense, state_of) of a fixed-step method (see ti.fixed_advance)."""
    pin, lam = ph["pin"], ph["lam"]
    if cfg.method == "rk4":
        def step(t, c, h, t_next):
            y = pin(ti.rk4_step(ph["rhs"], t, c[0], h, k1=c[1]))
            return (y, ph["rhs"](t_next, y)), ()

        def dense(t, c0, c1, aux, th):
            return ti.hermite_dense(lam, c0[0], c0[1] - lam * c0[0], c1[0],
                                    c1[1] - lam * c1[0], cfg.dt, th)
        return step, dense, (lambda c: c[0])
    if cfg.method == "etdrk4":
        def step(t, c, h, t_next):
            y = pin(ti.etdrk4_step(ph["nonlin"], c[1], t, c[0], h, n1=c[2]))
            return (y, c[1], ph["nonlin"](t_next, y)), ()

        def dense(t, c0, c1, aux, th):
            return ti.hermite_dense(lam, c0[0], c0[2], c1[0], c1[2], cfg.dt, th)
        return step, dense, (lambda c: c[0])
    if cfg.method in ("magnus2", "cfm4"):
        nodes, flow, fractions, hist_offsets, n_boot = _magnus_parts(cfg)
        w = jnp.asarray(ti.extrapolation_weights(hist_offsets, list(nodes)))

        def inner(gen, t, x, tau):
            core = lambda a, y, s: flow(gen, ph["free"], ph["conj_dev"], a, y, s)
            if fractions is None:
                return core(t, x, tau)
            return ti.compose_strang(core, ph["remainder"], t, x, tau, fractions)

        def step(t, c, h, t_next):
            x, hist, k = c
            guess = tuple(jnp.tensordot(w, hist, axes=1))
            n_c = jnp.where(k < n_boot, cfg.n_corr + BOOTSTRAP_EXTRA, cfg.n_corr)
            x1, g = ti.ip_magnus_step(ph["field"], ph["dsig_dev"], inner, ph["free"],
                                      t, x, h, guess, nodes=nodes, n_corr=n_c)
            hist = ph["free"](jnp.concatenate([hist[len(nodes):], jnp.stack(g)]), h)
            return (pin(x1), hist, k + 1), g

        def dense(t, c0, c1, g, th):
            gen = ti.frozen_generator(ph["field"], ph["free"], t, cfg.dt, nodes, g)
            return ph["rho_eq"] + inner(gen, t, c0[0], th * cfg.dt)
        return step, dense, (lambda c: ph["rho_eq"] + c[0])
    raise ValueError(f"GATE td_method: got: {cfg.method!r}; want: dp54 | rk4 | etdrk4 | "
                     f"magnus2 | cfm4; fix: internal (the deck parser admits only those)")


def initial_carry(cfg: PropagatorConfig, rho0, ops, mparams, model, drive, mesh):
    """The integrator's carry at t0 (one jitted call).

    dp54 : SciPy's state.   rk4 : (rho, rhs).   etdrk4 : (rho, coef, N).
    magnus2 / cfm4 : (Delta rho, dSigma node history in the interaction
    picture of the next step's start, step counter).
    """
    def build(rho0, ops, mparams):
        ph = _physics(ops, mparams, model, drive, mesh)
        if cfg.method == "dp54":
            return ti.dp54_initial_state(ph["rhs"], cfg.t0, rho0, cfg.dt_measure)
        if cfg.method == "rk4":
            return (rho0, ph["rhs"](cfg.t0, rho0))
        if cfg.method == "etdrk4":
            return (rho0, ti.etdrk4_coefficients(ph["lam"], cfg.dt),
                    ph["nonlin"](cfg.t0, rho0))
        x0 = ph["pin"](rho0 - ph["rho_eq"])
        b0 = ph["dsig_dev"](cfg.t0, x0)
        n_hist = len(_magnus_parts(cfg)[3])
        return (x0, jnp.stack([b0] * n_hist), jnp.int32(0))
    return jax.jit(build)(rho0, ops, mparams)


# ---------------------------------------------------------------------------
# The step limit of the corrected Magnus steps
# ---------------------------------------------------------------------------

def coupling_radius(rho0, ops, model: SelfEnergyModel, drive, mesh, *, t0: float,
                    n_iter: int = COUPLING_POWER_ITERATIONS, seed: int = 0):
    """Power-iteration estimate of the spectral radius σ (Ha) of the coupling map.

        L(x) = -i [dSigma[rho^eq + x] - dSigma[rho^eq], rho^eq]
             = -i (dSigma[x] - dSigma[0]) ∘ (f_m - f_n)

    the linearized coupling of the Magnus steps (the field is left out: it
    is weak where the limit binds, and E·r is exact in the free flow's
    frame anyway).  From a fixed random x (cv and vc blocks, one trial,
    logical k), x <- L(L(x)) / ||L(L(x))||, σ_j = (||L(L x_j)|| / ||x_j||)^(1/2):
    the eigenvalues of L are ±iω (real ω for a Hermitian coupling, TDA or a
    stable full BSE), so L^2 has the dominant real eigenvalue -σ^2 and σ_j
    rises to σ.  One jitted call, 2 n_iter dSigma evaluations on the
    model's own operands and layout.  Returns (σ, the n_iter estimates).
    """
    nk_pad, nb = int(ops.e_qp.shape[0]), int(ops.e_qp.shape[1])
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, nk_pad, nb, nb)) + 1j * rng.standard_normal((1, nk_pad, nb, nb))
    f = np.real(np.diagonal(np.asarray(gather_to_host(ops.rho_eq)), axis1=-2, axis2=-1))
    gap = (f[:, None, :] - f[:, :, None])[None]                      # f_m - f_n
    w = np.asarray(gather_to_host(ops.w_k)) > 0.0                     # logical k
    x = x * (np.abs(gap) > 0.5) * w[None, :, None, None]
    x = 0.5 * (x + np.conj(np.swapaxes(x, -1, -2)))
    x0 = place_state(x, mesh)
    gap_d = place_state(np.broadcast_to(gap, x.shape).astype(complex), mesh)
    zero = place_state(np.zeros_like(x), mesh)

    def run(x, gap, zero, ops, mparams):
        ph = _physics(ops, mparams, model, drive, mesh)
        d0 = ph["dsig_dev"](t0, zero)
        lmap = lambda y: ph["pin"](-1j * (ph["dsig_dev"](t0, y) - d0) * gap)
        nrm = lambda y: jnp.sqrt(jnp.sum(jnp.abs(y) ** 2))

        def body(y, _):
            z = lmap(lmap(y))
            ratio = nrm(z) / nrm(y)
            return z / nrm(z), jnp.sqrt(ratio)
        _, est = jax.lax.scan(body, x / nrm(x), None, length=int(n_iter), unroll=1)
        return est

    est = np.asarray(gather_to_host(jax.jit(run)(x0, gap_d, zero, ops, model.params)))
    return float(est[-1]), est


def magnus_dt_limit(cfg: PropagatorConfig, sigma: float) -> tuple:
    """(dt_max a.u., y_lim): the largest stable step of cfg's Magnus scheme for radius σ.

    y_lim = :func:`solvers.time_integrators.magnus_stability_limit` of the
    scheme's nodes, predictor and n_corr; dt_max = y_lim / (STABILITY_MARGIN σ).
    """
    nodes, _, _, hist, _ = _magnus_parts(cfg)
    y_lim = ti.magnus_stability_limit(nodes, hist, int(cfg.n_corr))
    return (np.inf if sigma <= 0.0 else y_lim / (STABILITY_MARGIN * sigma)), y_lim


# ---------------------------------------------------------------------------
# Samples on the measurement grid, chunks, checkpoints
# ---------------------------------------------------------------------------

def sample_plan(cfg: PropagatorConfig, mtsteps):
    """Where each sample lies on the fixed step grid t_j = t0 + j dt.

    Returns (step (n,) int64, theta (n,) float64 in [0, 1), n_steps): sample
    i is at t_{step_i} + theta_i dt; a sample within 1e-9 of a step end is
    ON it (theta = 0, the stepped state itself).  n_steps covers the last
    sample interval: the smallest j with t_j >= t0 + n dt_measure (the
    final time of a TDGW run on its grid), n * dt_measure / dt when dt
    divides dt_measure.
    """
    r = (np.asarray(mtsteps, float) - cfg.t0) / cfg.dt
    near = np.rint(r)
    on = np.abs(r - near) <= _ON_STEP_TOL * np.maximum(1.0, np.abs(r))
    step = np.where(on, near, np.floor(r)).astype(np.int64)
    theta = np.where(on, 0.0, r - np.floor(r))
    re = np.asarray(mtsteps).size * cfg.dt_measure / cfg.dt
    n_steps = int(np.rint(re)) if abs(re - np.rint(re)) <= _ON_STEP_TOL * max(1.0, re) \
        else int(np.ceil(re))
    return step, theta, n_steps


def chunk_plan(step, n_steps: int, chunk: int, i_start: int = 0, s_start: int = 0):
    """[(i0, i1, s0, s1)]: chunk q takes steps [s0, s1) and emits samples [i0, i1).

    A chunk ends at the step holding sample i0 + chunk (or at n_steps), so
    it emits every sample whose step is below s1 and a resumed run (from
    i_start, s_start of a checkpoint) continues the same grid.
    """
    out, n = [], len(step)
    i0, s0 = int(i_start), int(s_start)
    while s0 < n_steps:
        it = i0 + int(chunk)
        s1 = int(step[it]) if it < n else int(n_steps)
        s1 = max(s1, s0 + 1)
        i1 = int(np.searchsorted(step, s1, side="left"))
        out.append((i0, i1, s0, s1))
        i0, s0 = i1, s1
    return out


def carry_to_host(carry) -> list:
    """The carry's leaves gathered to host on every rank, k padded, tree order."""
    return [np.asarray(gather_to_host(leaf)) for leaf in jax.tree.leaves(carry)]


def carry_from_host(template, leaves, mesh, nk_pad: int):
    """A carry on ``mesh`` from host leaves (k logical or padded), shaped as ``template``.

    A leaf of rank >= 3 ends in (nk, nb, nb): it is zero-padded to nk_pad
    and k-sharded; lower ranks are replicated scalars.  Refuses a leaf count,
    shape or dtype that differs from the template (a checkpoint of another
    method or window).
    """
    flat, tree = jax.tree.flatten(template)
    if len(flat) != len(leaves):
        raise ValueError(f"GATE td_restart_carry: got: {len(leaves)} carry arrays; want: "
                         f"{len(flat)} (this method's carry); fix: restart with the "
                         f"checkpointed run's [lorrax] method")
    out = []
    for i, (t, h) in enumerate(zip(flat, leaves)):
        h = np.asarray(h)
        k_axis = t.ndim - 3 if t.ndim >= 3 else None
        if k_axis is not None:
            h = _pad_k(h, k_axis, nk_pad)
        if h.shape != tuple(t.shape) or h.dtype != t.dtype:
            raise ValueError(f"GATE td_restart_carry: got: carry array {i} {h.shape} "
                             f"{h.dtype}; want: {tuple(t.shape)} {t.dtype}; fix: restart "
                             f"with the checkpointed run's window, trials and method")
        out.append(_place(h, mesh, k_axis))
    return jax.tree.unflatten(tree, out)


def _chunk_program(cfg: PropagatorConfig, spec: SampleSpec, model_apply,
                   model_name: str, drive_key, mesh, cell_volume: float,
                   nspinor: int, count: int, n_steps: int, last: bool):
    """The jitted chunk for one static configuration.

    Not cached here: ``run_propagation`` keeps one program per (steps,
    samples, last) for the duration of a run, so nothing outlives the run (a
    process-wide cache keyed by the drive and model closures, hashed by
    identity, was never hit by an equivalent run and pinned their operands).
    """
    drive = drive_key[0]
    model = SelfEnergyModel(apply=model_apply, params=None, linear=True,
                            name=model_name)

    def run(carry, ops, mparams, sample_times, sample_step, sample_theta, step0):
        ph = _physics(ops, mparams, model, drive, mesh)
        sample_fn = _sample(ph, ops, spec, drive, cell_volume=cell_volume,
                            nspinor=nspinor)
        if cfg.method == "dp54":
            return ti.dp54_advance(ph["rhs"], carry, t_bound=cfg.tf,
                                   max_step=cfg.max_step, atol=cfg.atol,
                                   rtol=cfg.rtol, count=count,
                                   sample_times=sample_times,
                                   sample_fn=sample_fn, run_to_bound=last)
        step, dense, state_of = _fixed_parts(cfg, ph)
        return ti.fixed_advance(step, dense, state_of, carry, t_start=cfg.t0, h=cfg.dt,
                                step0=step0, n_steps=n_steps, sample_step=sample_step,
                                sample_theta=sample_theta, sample_times=sample_times,
                                sample_fn=sample_fn)

    return jax.jit(run, donate_argnums=(0,))


def run_propagation(cfg: PropagatorConfig, spec: SampleSpec, rho0, ops,
                    model: SelfEnergyModel, drive: Callable, mesh, *,
                    cell_volume: float, nspinor: int, count: int,
                    mtsteps: np.ndarray, on_chunk: Callable, resume=None):
    """Integrate over ``mtsteps`` in chunks; ``on_chunk`` writes each.

    ``on_chunk(i0, i1, samples_host, checkpoint, t_state)`` receives the
    samples of measurement indices [i0, i1) gathered to host (every rank),
    a zero-argument callable ``checkpoint() -> (rho_host, carry_leaves,
    step)`` gathering the state, the integrator's whole carry and its step
    counter (every rank), and the state's time.  ``resume`` = dict(carry=
    host leaves, step=int, i_next=int) continues a checkpointed run from its
    next sample.  Returns (counters, carry): steps taken by this run,
    rejected tries, dSigma evaluations.
    """
    nk_pad = int(ops.e_qp.shape[0])
    s_start = i_start = 0
    if resume is None:
        carry = initial_carry(cfg, rho0, ops, model.params, model, drive, mesh)
    else:
        template = jax.eval_shape(
            lambda r, o, p: initial_carry(cfg, r, o, p, model, drive, mesh),
            rho0, ops, model.params)
        carry = carry_from_host(template, resume["carry"], mesh, nk_pad)
        s_start, i_start = int(resume["step"]), int(resume["i_next"])
    n_meas = int(mtsteps.size)
    fixed = cfg.method != "dp54"
    programs = {}                       # static key -> jitted chunk, this run only

    def program(key, n_steps, last):
        if key not in programs:
            programs[key] = _chunk_program(cfg, spec, model.apply, model.name, (drive,),
                                           mesh, float(cell_volume), int(nspinor),
                                           int(count), int(n_steps), bool(last))
        return programs[key]

    if fixed:
        sstep, stheta, n_total = sample_plan(cfg, mtsteps)
        s_end = s_start
        for i0, i1, s0, s1 in chunk_plan(sstep, n_total, cfg.chunk, i_start, s_start):
            prog = program(("fixed", s1 - s0, i1 - i0), s1 - s0, False)
            carry, samples = prog(carry, ops, model.params, jnp.asarray(mtsteps[i0:i1]),
                                  jnp.asarray(sstep[i0:i1]), jnp.asarray(stheta[i0:i1]), s0)
            host = jax.tree.map(gather_to_host, samples)
            state = (lambda c=carry, s=s1: (gather_to_host(c[0] if cfg.method in ("rk4", "etdrk4")
                                                          else ops.rho_eq[None] + c[0]),
                                            carry_to_host(c), s))
            on_chunk(i0, i1, host, state, cfg.t0 + s1 * cfg.dt)
            s_end = s1
        n_steps = s_end - s_start
        stats = dict(accepted=n_steps, rejected=0,
                     dsigma=n_steps * dsigma_per_step(cfg.method, cfg.n_corr))
        if cfg.method in ("magnus2", "cfm4"):
            boot = max(0, min(s_end, _magnus_parts(cfg)[4]) - s_start)
            per = 1 if cfg.method == "magnus2" else 2
            stats["dsigma"] += boot * BOOTSTRAP_EXTRA * per
        stats["dsigma"] += 1 if resume is None else 0          # the initial carry's call
        stats["t_final"] = cfg.t0 + s_end * cfg.dt
    else:
        base = (0, 0, 0) if resume is None else tuple(
            int(gather_to_host(v)) for v in (carry.n_accepted, carry.n_rejected, carry.n_rhs))
        for i0 in range(i_start, n_meas, cfg.chunk):
            i1 = min(n_meas, i0 + cfg.chunk)
            last = i1 == n_meas
            prog = program(("dp54", i1 - i0, last), 0, last)
            times = jnp.asarray(mtsteps[i0:i1])
            carry, samples = prog(carry, ops, model.params, times,
                                  jnp.zeros(i1 - i0, jnp.int64), jnp.zeros(i1 - i0), 0)
            host = jax.tree.map(gather_to_host, samples)
            state = (lambda c=carry: (gather_to_host(c.y), carry_to_host(c), 0))
            on_chunk(i0, i1, host, state, float(gather_to_host(carry.t)))
        if bool(gather_to_host(carry.failed)):
            raise RuntimeError(
                "GATE td_dp54_step_too_small: got: the RK45 controller's step fell "
                "below SciPy's min_step; want: an integrable run; why: the run "
                "cannot reach tf; fix: loosen atol/rtol or check the drive and model")
        stats = dict(accepted=int(gather_to_host(carry.n_accepted)) - base[0],
                     rejected=int(gather_to_host(carry.n_rejected)) - base[1],
                     dsigma=int(gather_to_host(carry.n_rhs)) - base[2],
                     t_final=float(gather_to_host(carry.t)))
    needs = any(kind == "current" and not dict(opts).get("use_v0", False)
                for _, kind, opts in spec.kinds)
    stats["sample_dsigma"] = (n_meas - i_start) if needs else 0
    return stats, carry
