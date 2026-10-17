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
cfm4      the same step with the Gauss-node equations solved          2 n_eval
(nodes)   (node_solver = anderson: Anderson acceleration of the node
          map to node_tol; counted per step)
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
from distrib_la import eigh_local

__all__ = ["METHODS", "NODE_SOLVERS", "dsigma_per_step", "PropagatorConfig", "SampleSpec",
           "NodeAccount", "converged_nodes",
           "place_operators", "place_kernel", "place_state", "make_conj_dev",
           "sample_plan", "chunk_plan", "initial_carry", "carry_to_host",
           "carry_from_host", "coupling_radius", "magnus_dt_limit", "step_growth",
           "run_propagation"]

METHODS = ("dp54", "rk4", "etdrk4", "magnus2", "cfm4")
#: cfm4's Gauss-node values: n_corr corrections of a prediction ("truncated"), or the
#: node equations solved to node_tol by Anderson acceleration ("anderson").
NODE_SOLVERS = ("truncated", "anderson")
#: Relative residuals whose first crossing each converged-node solve records: the
#: evaluations a looser node_tol would have taken, from one tight run.
NODE_MILESTONES = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
#: The frame recycled Anderson columns are carried in: "own" (unchanged: the node
#: map is the same operator in every step's own interaction-picture frame when
#: the coupling is time-translation invariant) or "free" (rotated by free(., h)
#: as the node history is).
RECYCLE_FRAME = "own"
#: Bootstrap: while the predictor history is incomplete (first steps) the
#: Magnus steps take this many extra corrections.
BOOTSTRAP_EXTRA = 3
K_AXES = ("x", "y")
#: A sample within this fraction of a step from a step end is ON the step.
_ON_STEP_TOL = 1e-9
#: Power iterations of L(L(x)) in :func:`coupling_radius` (two dSigma calls each).
COUPLING_POWER_ITERATIONS = 20
COUPLING_ZERO_FLOOR = 1e-12      # Ha: a coupling radius below it is no coupling (IP)
#: The step check takes h (MARGIN * sigma) <= the stability limit: the power
#: iteration approaches sigma from below.
STABILITY_MARGIN = 1.1
#: The step probe of the stability check (:func:`step_growth`): steps, the step
#: relative to the σ limit at which it measures, and the growth per step taken
#: as unstable there.
PROBE_STEPS = 48
PROBE_STEP_FACTOR = 1.25
PROBE_TOLERANCE = 2e-3
#: dt = 'auto': from the σ limit, shrink by this factor until the probe passes,
#: at most this many probes.
PROBE_BACKTRACK = 0.85
PROBE_BACKTRACKS = 8


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
    frame : magnus2's frame, "interaction" (Delta rho in the interaction
        picture of the band energies, ti.ip_magnus_step) or "whole_rho" (the
        whole rho, ti.magnus_midpoint_step: exact quadratic invariants);
        cfm4 is always "interaction".
    node_solver : cfm4's Gauss-node values, one of NODE_SOLVERS: "truncated"
        (n_corr corrections, ti.ip_magnus_step) or "anderson" (the implicit
        node equations solved per step, ti.ip_magnus_implicit_step; n_corr
        unused).
    node_tol, node_max_iter, node_memory : the converged solve's relative
        residual ||F(g) - g|| <= node_tol ||g||, its largest number of
        node-map evaluations (more refuses, GATE td_node_solve), and its
        Anderson memory.
    node_recycle : the converged solve starts each step from the previous
        step's Anderson column pairs (carried, and checkpointed, with the state).
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
    frame: str = "interaction"
    node_solver: str = "truncated"
    node_tol: float = 1e-10
    node_max_iter: int = 40
    node_memory: int = 5
    node_recycle: bool = False


def converged_nodes(cfg: PropagatorConfig) -> bool:
    """cfg's cfm4 step solves its Gauss-node equations (node_solver != truncated)."""
    return cfg.method == "cfm4" and cfg.node_solver != "truncated"


class NodeAccount(NamedTuple):
    """The node solves of a converged-node run so far, carried with the state.

    n_eval : int64, node-map evaluations since t0 (each one dSigma per node).
    n_eval_max : int32, the most in one step.
    n_fail : int32, steps whose solve did not reach node_tol in node_max_iter.
    first_fail : int64, the step index of the first such step (-1: none).
    res_fail : float64, its relative residual.
    res_max : float64, the largest final relative residual of any step.
    reached_sum, reached_max, reached_n : (len(NODE_MILESTONES),) the sum and
        the largest of the evaluations at which each milestone was first
        reached, and the number of steps that reached it.
    """
    n_eval: jax.Array
    n_eval_max: jax.Array
    n_fail: jax.Array
    first_fail: jax.Array
    res_fail: jax.Array
    res_max: jax.Array
    reached_sum: jax.Array
    reached_max: jax.Array
    reached_n: jax.Array


def _account_zero() -> NodeAccount:
    z = len(NODE_MILESTONES)
    return NodeAccount(jnp.int64(0), jnp.int32(0), jnp.int32(0), jnp.int64(-1),
                       jnp.float64(0.0), jnp.float64(0.0), jnp.zeros(z, jnp.int64),
                       jnp.zeros(z, jnp.int32), jnp.zeros(z, jnp.int64))


def _recycle_zero(cfg: PropagatorConfig, x0, ph):
    """Empty Anderson column buffers (E, D per node, n_col 0) for node_recycle."""
    m, n = int(cfg.node_memory), len(_magnus_parts(cfg)[0])
    z = tuple(ph["pin_stack"](jnp.zeros((m,) + x0.shape, x0.dtype)) for _ in range(n))
    return (z, z, jnp.int32(0))


def _account(acct: NodeAccount, info, k) -> NodeAccount:
    """acct after one step's solve ``info`` (ti.NodeSolve) at step index k."""
    bad = ~info.converged
    first = (acct.first_fail < 0) & bad
    return NodeAccount(
        n_eval=acct.n_eval + info.n_eval.astype(jnp.int64),
        n_eval_max=jnp.maximum(acct.n_eval_max, info.n_eval.astype(jnp.int32)),
        n_fail=acct.n_fail + bad.astype(jnp.int32),
        first_fail=jnp.where(first, jnp.asarray(k).astype(jnp.int64), acct.first_fail),
        res_fail=jnp.where(first, info.residual, acct.res_fail),
        res_max=jnp.maximum(acct.res_max, info.residual),
        reached_sum=acct.reached_sum + info.reached.astype(jnp.int64),
        reached_max=jnp.maximum(acct.reached_max, info.reached.astype(jnp.int32)),
        reached_n=acct.reached_n + (info.reached > 0).astype(jnp.int64))


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
        e, v = eigh_local(w)
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

    def explicit(t):
        """diag(E^QP) + E(t)·r^inter: the field part of the whole-rho generator."""
        return pin(eom.field_hamiltonian(ops, drive(t)))

    stack_spec = NamedSharding(mesh, _spec(5, 2))

    def dsig_nodes(ts, xs):
        """dSigma at every node (deviations xs): one call on the trial-stacked states
        when the model has no explicit time dependence (``model.time_independent``),
        else one call per node at its own time."""
        if model.time_independent and len(xs) > 1:
            n = xs[0].shape[0]
            d = dsig(ts[0], ops.rho_eq[None] + jnp.concatenate(xs, axis=0))
            return tuple(pin(d[i * n:(i + 1) * n]) for i in range(len(xs)))
        return tuple(dsig(tt, ops.rho_eq[None] + xx) for tt, xx in zip(ts, xs))

    no_occ = jnp.zeros_like(occ)
    return dict(dsig=dsig, rhs=rhs, field=field, free=free, nonlin=nonlin,
                lam=lam, remainder=remainder, pin=pin, explicit=explicit,
                dsig_nodes=dsig_nodes,
                pin_stack=lambda a: jax.lax.with_sharding_constraint(a, stack_spec),
                conj_dev=lambda w, x, tau: pin(conj(w, x, occ, tau)),
                conjugate=lambda w, y, tau: pin(conj(w, y, no_occ, tau)),
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
    if cfg.method == "magnus2" and cfg.frame == "whole_rho":
        w = jnp.asarray(ti.extrapolation_weights(ti.MIDPOINT_HISTORY, [0.5])[0])
        rho_eq = ph["rho_eq"]
        rem = lambda a, y, tau: rho_eq + ph["remainder"](a, y - rho_eq, tau)

        def step(t, c, h, t_next):
            y, hist, k = c
            n_it = jnp.where(k < len(ti.MIDPOINT_HISTORY), cfg.n_corr + BOOTSTRAP_EXTRA,
                             cfg.n_corr)
            guess, box = jnp.tensordot(w, hist, axes=1), []

            def core(a, yy, tau):
                y1, b = ti.magnus_midpoint_step(ph["explicit"], ph["dsig"], ph["conjugate"],
                                                a, yy, tau, guess, n_iter=n_it)
                box.append(b)
                return y1
            y1 = ti.compose_strang(core, rem, t, y, h) if cfg.remainder else core(t, y, h)
            return (pin(y1), jnp.concatenate([hist[1:], box[-1][None]]), k + 1), box[-1]

        def dense(t, c0, c1, b, th):          # the midpoint flow (magnus2 samples on steps)
            return ph["conjugate"](ph["explicit"](t + 0.5 * cfg.dt) + b, c0[0], th * cfg.dt)
        return step, dense, (lambda c: c[0])
    if cfg.method in ("magnus2", "cfm4"):
        nodes, flow, fractions, hist_offsets, n_boot = _magnus_parts(cfg)
        w = jnp.asarray(ti.extrapolation_weights(hist_offsets, list(nodes)))

        def inner(gen, t, x, tau):
            core = lambda a, y, s: flow(gen, ph["free"], ph["conj_dev"], a, y, s)
            if fractions is None:
                return core(t, x, tau)
            return ti.compose_strang(core, ph["remainder"], t, x, tau, fractions)

        implicit = converged_nodes(cfg)
        recycle = implicit and bool(cfg.node_recycle)

        def step(t, c, h, t_next):
            x, hist, k = c[:3]
            guess = tuple(jnp.tensordot(w, hist, axes=1))
            if implicit:
                out = ti.ip_magnus_implicit_step(
                    ph["field"], ph["dsig_nodes"], inner, ph["free"], t, x, h, guess,
                    nodes=nodes, tol=cfg.node_tol,
                    # once a step has failed the run (or probe) is refused: one evaluation per step
                    max_iter=jnp.where(c[3].n_fail > 0, 1, int(cfg.node_max_iter)),
                    memory=int(cfg.node_memory), constrain=ph["pin_stack"],
                    milestones=NODE_MILESTONES, recycle=c[4] if recycle else None)
                x1, g, info = out[:3]
                if recycle:
                    e_b, d_b, n_col = out[3]
                    if RECYCLE_FRAME == "free":
                        e_b = tuple(ph["free"](b, h) for b in e_b)
                        d_b = tuple(ph["free"](b, h) for b in d_b)
                    rec = (e_b, d_b, n_col)
            else:
                n_c = jnp.where(k < n_boot, cfg.n_corr + BOOTSTRAP_EXTRA, cfg.n_corr)
                x1, g = ti.ip_magnus_step(ph["field"], ph["dsig_dev"], inner, ph["free"],
                                          t, x, h, guess, nodes=nodes, n_corr=n_c)
            hist = ph["free"](jnp.concatenate([hist[len(nodes):], jnp.stack(g)]), h)
            out = (pin(x1), hist, k + 1)
            if implicit:
                out = out + (_account(c[3], info, k),) + ((rec,) if recycle else ())
            return out, g

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
    picture of the next step's start, step counter), cfm4 with converged
    nodes + its :class:`NodeAccount`; magnus2 in the whole_rho frame: (rho,
    the last three midpoint dSigma, step counter).
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
        if cfg.method == "magnus2" and cfg.frame == "whole_rho":
            b0 = ph["dsig"](cfg.t0, rho0)
            return (rho0, jnp.stack([b0] * len(ti.MIDPOINT_HISTORY)), jnp.int32(0))
        x0 = ph["pin"](rho0 - ph["rho_eq"])
        b0 = ph["dsig_dev"](cfg.t0, x0)
        n_hist = len(_magnus_parts(cfg)[3])
        carry = (x0, jnp.stack([b0] * n_hist), jnp.int32(0))
        if converged_nodes(cfg):
            carry = carry + (_account_zero(),) + ((_recycle_zero(cfg, x0, ph),)
                                                  if cfg.node_recycle else ())
        return carry
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
    model's own operands and layout.  A map without coupling (IP, or a
    kernel of zeros) has ||L(L(x))|| <= COUPLING_ZERO_FLOOR^2 ||x||: the
    iteration keeps x and the estimate is 0, never 0/0, and a radius below
    the floor (1e-12 Ha, 2.7e-11 eV: it would bound only steps above 1e11
    a.u.) returns σ = 0.0.  A map that returns non-finite values gives
    σ = nan (the gate refuses it by name).  Returns (σ, the n_iter estimates).
    """
    nk_pad, nb = int(ops.e_qp.shape[0]), int(ops.e_qp.shape[1])
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, nk_pad, nb, nb)) + 1j * rng.standard_normal((1, nk_pad, nb, nb))
    f = np.real(np.diagonal(np.asarray(gather_to_host(ops.rho_eq)), axis1=-2, axis2=-1))
    gap = (f[:, None, :] - f[:, :, None])[None]                      # f_m - f_n
    w = np.asarray(gather_to_host(ops.w_k)) > 0.0                     # logical k
    x = x * (np.abs(gap) > 0.5) * w[None, :, None, None]
    x = 0.5 * (x + np.conj(np.swapaxes(x, -1, -2)))
    if not np.any(x):                                 # no cv pair on a logical k: nothing couples
        return 0.0, np.zeros(int(n_iter))
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
            nz, ny = nrm(z), nrm(y)
            dead = nz <= COUPLING_ZERO_FLOOR ** 2 * ny       # a zero map: keep y, estimate 0
            return (jnp.where(dead, y, z / jnp.where(dead, 1.0, nz)),
                    jnp.where(dead, 0.0, jnp.sqrt(nz / ny)))
        _, est = jax.lax.scan(body, x / nrm(x), None, length=int(n_iter), unroll=1)
        return est

    est = np.asarray(gather_to_host(jax.jit(run)(x0, gap_d, zero, ops, model.params)))
    sigma = float(est[-1])
    return (0.0 if 0.0 <= sigma < COUPLING_ZERO_FLOOR else sigma), est


def magnus_dt_limit(cfg: PropagatorConfig, sigma: float) -> tuple:
    """(dt_max a.u., y_lim): the largest stable step of cfg's Magnus scheme for radius σ.

    y_lim = :func:`solvers.time_integrators.magnus_stability_limit` of the
    scheme's nodes, predictor and n_corr; dt_max = y_lim / (STABILITY_MARGIN σ).
    Converged nodes have no such limit (the converged step's amplification has
    modulus 1 on the whole imaginary axis): (inf, inf), the probe decides.
    """
    if converged_nodes(cfg):
        return np.inf, np.inf
    nodes, _, _, hist, _ = _magnus_parts(cfg)
    y_lim = ti.magnus_stability_limit(nodes, hist, int(cfg.n_corr))
    return (np.inf if sigma <= 0.0 else y_lim / (STABILITY_MARGIN * sigma)), y_lim


def step_growth(cfg: PropagatorConfig, ops, model: SelfEnergyModel, mesh, *, h: float,
                t0: float, n_steps: int = PROBE_STEPS, eps: float = None, seed: int = 1,
                with_counts: bool = False):
    """Growth per step of a small coherence under the scheme's own step at h, field off.

    The scheme-exact counterpart of :func:`coupling_radius`: from a random
    Hermitian deviation (cv and vc blocks, logical k, largest entry ``eps``;
    the deviation form keeps it exact, so the probe stays linear even where
    the step is violently unstable; 1e-8 for the whole_rho frame, which
    rounds at 1e-16 of the occupations) and a node history seeded from
    independent ones (the predictor's parasitic roots start at full size),
    ``n_steps`` steps of cfg's Magnus step past its bootstrap (predictor,
    corrections, composition; E = 0) and
    g = (|x_n| / |x_m|)^(1/(n - m)) over the second half (Frobenius norm).
    It sees what the scalar recurrence leaves out: in the interaction picture
    the coupling is time dependent, and a kernel's resonant-antiresonant
    block rotates there at about the sum of two transition energies.  One
    jitted call, h traced.  With converged nodes the probe also solves every
    step's node equations: a step whose solve does not converge makes
    g = inf (the scheme is not usable at h).  Returns (g, the norms after
    each step), and with ``with_counts`` a third item, dict(n_eval,
    n_eval_max, n_fail, dsigma) of the probe's node solves (dsigma: every
    dSigma call of the probe, the history seeds included).
    """
    whole = cfg.method == "magnus2" and cfg.frame == "whole_rho"
    if eps is None:              # the whole rho rounds at 1e-16 of its O(1) occupations
        eps = 1e-8 if whole else 1e-14
    nk_pad, nb = int(ops.e_qp.shape[0]), int(ops.e_qp.shape[1])
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, nk_pad, nb, nb)) + 1j * rng.standard_normal((1, nk_pad, nb, nb))
    f = np.real(np.diagonal(np.asarray(gather_to_host(ops.rho_eq)), axis1=-2, axis2=-1))
    gap = (f[:, None, :] - f[:, :, None])[None]
    w = np.asarray(gather_to_host(ops.w_k)) > 0.0
    x = x * (np.abs(gap) > 0.5) * w[None, :, None, None]
    x = 0.5 * (x + np.conj(np.swapaxes(x, -1, -2)))
    scale = eps / np.max(np.abs(x))
    n_hist, n_boot = ((len(ti.MIDPOINT_HISTORY), len(ti.MIDPOINT_HISTORY)) if whole else
                      (len(_magnus_parts(cfg)[3]), _magnus_parts(cfg)[4]))
    xs = [x * scale]
    for _ in range(n_hist):                       # independent seeds of the node history
        y = rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
        y = y * (np.abs(gap) > 0.5) * w[None, :, None, None]
        y = 0.5 * (y + np.conj(np.swapaxes(y, -1, -2)))
        xs.append(y * (eps / np.max(np.abs(y))))
    x0 = place_state(xs[0], mesh)
    xh = place_state(np.concatenate(xs[1:], axis=0), mesh)
    off = lambda t: jnp.zeros((1, 3))                                  # the field is off

    implicit = converged_nodes(cfg)

    def run(x0, xh, ops, mparams, h):
        ph = _physics(ops, mparams, model, off, mesh)
        step, _, _ = _fixed_parts(cfg, ph)
        base = ph["rho_eq"] if whole else 0.0
        hist = jnp.stack([ph["dsig_dev"](t0, xh[j:j + 1]) for j in range(n_hist)])
        carry = (base + x0, hist, jnp.int32(n_boot))  # past the bootstrap: the predictor acts at once
        if implicit:
            carry = carry + (_account_zero(),) + ((_recycle_zero(cfg, x0, ph),)
                                                  if cfg.node_recycle else ())

        def body(c, i):
            t = t0 + i * h
            c, _ = step(t, c, h, t + h)
            return c, jnp.sqrt(jnp.sum(jnp.abs(c[0] - base) ** 2))
        c, norms = jax.lax.scan(body, carry, jnp.arange(int(n_steps)), unroll=1)
        return norms, (c[3] if implicit else _account_zero())

    norms, acct = jax.jit(run)(x0, xh, ops, model.params, jnp.asarray(float(h)))
    norms = np.asarray(gather_to_host(norms))
    m = int(n_steps) // 2
    g = float((norms[-1] / norms[m - 1]) ** (1.0 / (int(n_steps) - m)))
    n_eval, n_max, n_fail = (int(gather_to_host(v)) for v in (acct.n_eval, acct.n_eval_max,
                                                              acct.n_fail))
    if implicit and n_fail:
        g = float("inf")
    if not with_counts:
        return g, norms
    per = len(_magnus_parts(cfg)[0])
    dsig = n_hist + (per * n_eval if implicit else int(n_steps) * dsigma_per_step(cfg.method, cfg.n_corr))
    return g, norms, dict(n_eval=n_eval, n_eval_max=n_max, n_fail=n_fail, dsigma=dsig)


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
                   nspinor: int, count: int, n_steps: int, last: bool,
                   model_static: bool = False):
    """The jitted chunk for one static configuration.

    Not cached here: ``run_propagation`` keeps one program per (steps,
    samples, last) for the duration of a run, so nothing outlives the run (a
    process-wide cache keyed by the drive and model closures, hashed by
    identity, was never hit by an equivalent run and pinned their operands).
    """
    drive = drive_key[0]
    model = SelfEnergyModel(apply=model_apply, params=None, linear=True,
                            name=model_name, time_independent=bool(model_static))

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


def _refuse_unconverged(cfg: PropagatorConfig, acct: NodeAccount):
    """GATE td_node_solve when the carry's account holds an unconverged step (every rank)."""
    n_fail = int(gather_to_host(acct.n_fail))
    if not n_fail:
        return
    from tdagw.tdgw_io import T_IN_ATTOSEC
    j = int(gather_to_host(acct.first_fail))
    raise ValueError(
        f"GATE td_node_solve: got: {n_fail} step(s) whose Gauss-node equations did not converge, "
        f"the first step {j} (t = {(cfg.t0 + j * cfg.dt) * T_IN_ATTOSEC:.6g} as) at relative "
        f"residual {float(gather_to_host(acct.res_fail)):.3g} after node_max_iter = "
        f"{int(cfg.node_max_iter)} evaluations of the node map; want: ||F(g) - g|| <= node_tol "
        f"||g|| = {float(cfg.node_tol):.3g} ||g|| at every step; why: an unconverged step "
        f"integrates neither the implicit scheme nor the equation, and is not written; fix: a "
        f"smaller [lorrax] dt, a larger node_max_iter or node_memory, or node_solver = 'truncated'")


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
    rejected tries, dSigma evaluations (converged nodes: counted from the
    carry's :class:`NodeAccount`, every node-map evaluation, plus ``nodes``,
    the solve statistics).  A converged-node chunk with a step whose solve
    did not converge refuses (GATE td_node_solve) before ``on_chunk`` writes
    it: no unconverged step reaches an output file.
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
    implicit = converged_nodes(cfg)
    programs = {}                       # static key -> jitted chunk, this run only
    acct0 = (int(gather_to_host(carry[3].n_eval)) if implicit else 0)

    def program(key, n_steps, last):
        if key not in programs:
            programs[key] = _chunk_program(cfg, spec, model.apply, model.name, (drive,),
                                           mesh, float(cell_volume), int(nspinor),
                                           int(count), int(n_steps), bool(last),
                                           bool(model.time_independent))
        return programs[key]

    if fixed:
        sstep, stheta, n_total = sample_plan(cfg, mtsteps)
        s_end = s_start
        for i0, i1, s0, s1 in chunk_plan(sstep, n_total, cfg.chunk, i_start, s_start):
            prog = program(("fixed", s1 - s0, i1 - i0), s1 - s0, False)
            carry, samples = prog(carry, ops, model.params, jnp.asarray(mtsteps[i0:i1]),
                                  jnp.asarray(sstep[i0:i1]), jnp.asarray(stheta[i0:i1]), s0)
            if implicit:
                _refuse_unconverged(cfg, carry[3])
            host = jax.tree.map(gather_to_host, samples)
            state = (lambda c=carry, s=s1: (gather_to_host(c[0] if cfg.method in ("rk4", "etdrk4")
                                                          else ops.rho_eq[None] + c[0]),
                                            carry_to_host(c), s))
            on_chunk(i0, i1, host, state, cfg.t0 + s1 * cfg.dt)
            s_end = s1
        n_steps = s_end - s_start
        stats = dict(accepted=n_steps, rejected=0,
                     dsigma=n_steps * dsigma_per_step(cfg.method, cfg.n_corr))
        if implicit:
            a = carry[3]
            n_eval = int(gather_to_host(a.n_eval)) - acct0
            stats["dsigma"] = len(_magnus_parts(cfg)[0]) * n_eval
            rs, rm, rn = (np.asarray(gather_to_host(v)) for v in (a.reached_sum, a.reached_max,
                                                                   a.reached_n))
            stats["nodes"] = dict(solver=cfg.node_solver, tol=float(cfg.node_tol),
                                  memory=int(cfg.node_memory), n_eval=n_eval,
                                  mean=n_eval / max(n_steps, 1),
                                  max_run=int(gather_to_host(a.n_eval_max)),
                                  res_max=float(gather_to_host(a.res_max)),
                                  milestones={f"{thr:.0e}": dict(mean=float(rs[i] / max(rn[i], 1)),
                                                                 max=int(rm[i]), steps=int(rn[i]))
                                              for i, thr in enumerate(NODE_MILESTONES)})
        elif cfg.method in ("magnus2", "cfm4"):
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
