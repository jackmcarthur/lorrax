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

``cfm4`` may split dSigma exactly on a small stiff subspace (``[lorrax]
stiff_modes``): with u_j an orthonormal basis of Hermitian deviations spanning
the invariant subspace of the largest eigenvalues of the coupling map L
(:func:`stiff_basis`, a real Arnoldi on L) and b_j = dSigma[u_j], dSigma[x] =
dSigma[x] - Σ_j <u_j, x> b_j + Σ_j <u_j, x> b_j holds for every x (the model
is affine), and the step treats the second sum implicitly at the Gauss nodes
(``solvers.time_integrators.FastPart``: node coefficients <u_j, x_node> by
simplified Newton on exponentials and projections, no dSigma call), so the
predictor-corrector limit binds only the slow map L(1 - Π).

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
restarted from it continues bitwise.  Every sample also records, on the
device and without a ΔΣ call, the run log's invariants (``PROGRESS``: per
trial Σ_k w_k Tr ρ_k, Σ_k w_k Tr ρ_k², |E(t)|), and every chunk hands the
host its cumulative step and ΔΣ counters (``COUNTERS``), from the same
counting as the final ones.
"""
from __future__ import annotations

import time
from typing import Callable, NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local, gather_to_host
from runtime.padding import round_up
from solvers import time_integrators as ti
from tdagw import collision as coll
from tdagw import eom
from tdagw.self_energy import SelfEnergyModel
from distrib_la import eigh_local

__all__ = ["METHODS", "PROGRESS", "COUNTERS", "dsigma_per_step", "PropagatorConfig", "SampleSpec",
           "place_operators", "place_kernel", "place_state", "make_conj_dev",
           "make_collision_maps", "collision_program", "place_window_blocks",
           "sample_plan", "chunk_plan", "initial_carry", "carry_to_host",
           "carry_from_host", "coupling_radius", "magnus_dt_limit", "step_growth",
           "coupling_arnoldi", "stiff_basis", "run_propagation"]

METHODS = ("dp54", "rk4", "etdrk4", "magnus2", "cfm4")
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
#: The stiff split: Arnoldi steps on L when the deck leaves stiff_krylov at 0
#: (at least STIFF_KRYLOV_MIN, and STIFF_KRYLOV_PER_MODE per stiff mode), and
#: the relative gap below which two Ritz moduli are one group (a cut never
#: separates the members of a +-lambda or conjugate pair).
STIFF_KRYLOV_MIN = 80
STIFF_KRYLOV_PER_MODE = 5
STIFF_GROUP_GAP = 1e-6
#: The sample entry of the run log's invariants, (n, ntr, 3) per chunk on the host:
#: Σ_k w_k Tr ρ_k, Σ_k w_k Tr ρ_k² (real parts) and |E(t)| (a.u.) per trial.
PROGRESS = "lorrax.progress"
#: The host entry of the cumulative counters at a chunk's end: steps, rejected tries,
#: ΔΣ calls in steps and at samples (this run, not a resumed checkpoint's), and
#: ``new_program``: whether this chunk built its jitted program (a chunk of a new
#: step or sample count traces and compiles, or loads from the compile cache, in
#: its own wall).
COUNTERS = "lorrax.counters"


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
    stiff : cfm4 only: the number of stiff modes split off dSigma and solved
        implicitly at the nodes (0: no split); the basis travels in
        ``ops.stiff`` (:func:`stiff_basis`).
    stiff_tol, stiff_iter : the closing solve of the stiff coefficients:
        relative residual and node-map evaluations per step; a step that
        does not reach the tolerance refuses the run (``td_stiff_solve``).
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
    stiff: int = 0
    stiff_tol: float = 1e-9
    stiff_iter: int = 6


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
# The held collision term (docs/theory/td_agw_dynamical.md §1.5), per k
# ---------------------------------------------------------------------------

def _dissipator_spec():
    return coll.Dissipator(vecs=_spec(4, 1), vals=_spec(3, 1), source=_spec(4, 1))


def make_collision_maps(mesh) -> tuple:
    """(flow, rate) of ``tdagw.collision`` on the k-sharded state, inside ``shard_map``.

    flow(d, x, tau) is Eq. (4) of that module (the exact held flow), rate(d, x)
    its generator; d a :class:`tdagw.collision.Dissipator` and x a
    (ntr, nk, nb, nb) deviation, both at ``P(None, ('x','y'), None, None)``.
    Each device transforms its own k block (four nb x nb products and an
    elementwise update): no collective, no host transfer.
    """
    spec4, dspec = _spec(4, 1), _dissipator_spec()
    flow = shard_map(coll.flow, mesh=mesh, in_specs=(dspec, spec4, P()), out_specs=spec4,
                     check_vma=False)
    rate = shard_map(coll.rate, mesh=mesh, in_specs=(dspec, spec4), out_specs=spec4,
                     check_vma=False)
    return flow, rate


def place_window_blocks(blocks, mesh, nk_pad: int):
    """Host (ntr, nk, nb, nb) blocks (propagator k order) -> k-sharded and k-padded."""
    return _place(_pad_k(np.asarray(blocks, np.complex128), 1, nk_pad), mesh, 1)


def collision_program(mesh, *, raw: bool):
    """The device half of a collision refresh, one jitted ``shard_map`` (k-local).

    ``run(gamma, injection, rho, rho_eq, gamma_eq, injection_eq, w_k)``: the held
    total width Gamma and injection D (ntr, nk, nb, nb) Ha, the state rho,
    equilibrium density diag(f) (nk, nb, nb), equilibrium operators, k weights.
    Returns dict: ``diss`` the :class:`tdagw.collision.Dissipator` of
    Gamma and S (less the imposed equilibrium counterterm unless ``raw``);
    ``c_eq`` that counterterm; per trial ``out``/``in`` the loss and injection
    parts of d(sum_k w_k Tr rho_k)/dt (subtracted: minus the same at
    equilibrium), ``trace`` sum_k w_k Tr ρ_k; ``min_rate``, ``min_injection``
    and ``min_blocking`` the smallest eigenvalues of Gamma, effective D, and
    Gamma-D.  The latter two are the held-generator Pauli-boundary tests.
    ``rho_min``/``rho_max`` are density bounds (pads excluded);
    ``c_eq_diag``/``c_eq_offdiag`` max |C[ρ_eq]| on and off
    the diagonal.  One all-reduce of a few (ntr,) scalars.
    """
    spec4, spec3, spec2 = _spec(4, 1), _spec(3, 0), _spec(1, 0)
    dspec = _dissipator_spec()

    def local(gamma, injection, rho, rho_eq, gamma_eq, injection_eq, w):
        req = rho_eq[None].astype(rho.dtype)
        c0 = coll.collision(gamma_eq, injection_eq, req)
        src = coll.source(gamma, injection, req, None if raw else c0)
        d = coll.dissipator(gamma, src, eigh=eigh_local)
        injection_eff = injection if raw else injection - c0
        dvals = eigh_local(coll.hermitian_part(injection_eff))[0]
        bvals = eigh_local(coll.hermitian_part(gamma - injection_eff))[0]
        rvals = eigh_local(coll.hermitian_part(rho))[0]
        out, inn = (coll.trace_rates(gamma, injection, rho) if raw else
                    coll.trace_rates(gamma, injection, rho, gamma_eq, injection_eq, req))
        wk = w.astype(jnp.float64)[None, :]
        tsum = lambda a: jax.lax.psum(jnp.sum(wk * a, axis=1), K_AXES)
        live = (w > 0.0)[None, :, None]
        big = jnp.asarray(jnp.inf, jnp.float64)
        nb = rho.shape[-1]
        eye = jnp.eye(nb, dtype=bool)
        dmax = lambda m: jax.lax.pmax(jnp.max(jnp.where(live[..., None] & m, jnp.abs(c0), 0.0)),
                                      K_AXES)
        return dict(diss=d, c_eq=c0, out=tsum(out), inn=tsum(inn),
                    trace=tsum(jnp.real(jnp.trace(rho, axis1=-2, axis2=-1))),
                    min_rate=jax.lax.pmin(jnp.min(jnp.where(live, jnp.real(d.vals), big)), K_AXES),
                    min_injection=jax.lax.pmin(
                        jnp.min(jnp.where(live, jnp.real(dvals), big)), K_AXES),
                    min_blocking=jax.lax.pmin(
                        jnp.min(jnp.where(live, jnp.real(bvals), big)), K_AXES),
                    rho_min=jax.lax.pmin(
                        jnp.min(jnp.where(live, jnp.real(rvals), big)), K_AXES),
                    rho_max=jax.lax.pmax(
                        jnp.max(jnp.where(live, jnp.real(rvals), -big)), K_AXES),
                    c_eq_diag=dmax(eye), c_eq_offdiag=dmax(~eye))

    out_specs = dict(diss=dspec, c_eq=spec4, out=P(), inn=P(), trace=P(), min_rate=P(),
                     min_injection=P(), min_blocking=P(), rho_min=P(), rho_max=P(), c_eq_diag=P(),
                     c_eq_offdiag=P())
    return jax.jit(shard_map(local, mesh=mesh,
                             in_specs=(spec4, spec4, spec4, spec3, spec4, spec4, spec2),
                             out_specs=out_specs, check_vma=False))


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
    # The held collision term (ops.collision): its exact flow in the remainder
    # (Magnus steps), its generator in the right-hand side (explicit steps).
    diss = ops.collision
    if diss is not None:
        cflow, crate = make_collision_maps(mesh)

    def collide(x):
        return crate(diss, x)

    def dsig(t, rho):
        return pin(model.apply(mparams, rho, t))

    def rhs(t, rho):
        out = eom.eom_rhs(ops, drive(t), dsig(t, rho), rho)
        if diss is not None:
            out = out + collide(rho - ops.rho_eq[None])
        return pin(out)

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
        out = -1j * drv + ops.gamma.astype(rho.dtype) * ops.rho_eq[None]
        if diss is not None:
            out = out + collide(rho - ops.rho_eq[None])
        return pin(out)

    def remainder(t, x, tau):
        """The non-commutator flow on Delta rho: Gamma decay (exact) around the held
        collision flow (exact, tdagw.collision Eq. 4) and the intraband term (RK4).

        Palindromic, so the Strang step stays symmetric and its triple jump fourth
        order: decay(tau/2) [D(tau/2) intra(tau) D(tau/2) | D(tau)] decay(tau/2).
        """
        decay = jnp.exp(-0.5 * tau * ops.gamma).astype(x.dtype)
        x = decay * x
        if ops.intra is not None:
            if diss is not None:
                x = cflow(diss, x, 0.5 * tau)
            f = lambda s, y: -1j * eom.field_intraband(ops.intra, drive(s), ops.rho_eq[None] + y)
            x = ti.rk4_step(f, t, x, tau)
            if diss is not None:
                x = cflow(diss, x, 0.5 * tau)
        elif diss is not None:
            x = cflow(diss, x, tau)
        return pin(decay * x)

    def explicit(t):
        """diag(E^QP) + E(t)·r^inter: the field part of the whole-rho generator."""
        return pin(eom.field_hamiltonian(ops, drive(t)))

    no_occ = jnp.zeros_like(occ)
    fast = None
    if ops.stiff is not None:                       # the stiff split's halves (FastPart)
        u, b = ops.stiff
        gap = (occ[:, None, :] - occ[:, :, None]).astype(jnp.complex128)     # f_m - f_n
        proj, comb = _stiff_maps(mesh)
        fast = dict(project=lambda x: proj(u, x), combine=lambda a: pin(comb(b, a)),
                    expand=lambda a: pin(comb(u, a)),
                    reference=lambda m: -1j * m * gap, n=int(u.shape[0]))
    return dict(dsig=dsig, rhs=rhs, field=field, free=free, nonlin=nonlin,
                lam=lam, remainder=remainder, pin=pin, explicit=explicit,
                conj_dev=lambda w, x, tau: pin(conj(w, x, occ, tau)),
                conjugate=lambda w, y, tau: pin(conj(w, y, no_occ, tau)),
                dsig_dev=lambda t, x: dsig(t, ops.rho_eq[None] + x),
                rho_eq=ops.rho_eq[None], fast=fast)


def _stiff_maps(mesh):
    """(project, combine) of the stiff split, k-local inside ``shard_map``.

    project(u, x) -> (b, n) float64: Re <u_j, x_q> = Re Σ_k tr(u_jk^H x_qk), one
    all-reduce of (b, n) (u and x Hermitian: the inner product is real, and
    the split is exact for any real-linear dSigma).  combine(b, a) -> (b, nk,
    nb, nb): Σ_j a[q, j] b_j, local.  u, b, x : (·, nk, nb, nb) k-sharded.
    """
    spec4 = _spec(4, 1)

    def project(u, x):
        return jax.lax.psum(jnp.real(jnp.einsum("jkab,tkab->tj", jnp.conj(u), x)), K_AXES)

    def combine(b, a):
        return jnp.einsum("tj,jkab->tkab", a.astype(b.dtype), b)

    return (shard_map(project, mesh=mesh, in_specs=(spec4, spec4), out_specs=P(),
                      check_vma=False),
            shard_map(combine, mesh=mesh, in_specs=(spec4, P()), out_specs=spec4,
                      check_vma=False))


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
        out[PROGRESS] = _invariants(ops, rho, drive(t))
        return out
    return fn


def _invariants(ops, rho, efield):
    """(ntr, 3): Σ_k w_k Tr ρ_k, Σ_k w_k Re Tr ρ_k², |E| (a.u.), on the device.

    Both traces are conserved by a Hamiltonian flow (Γ = 0): the commutators
    -i[H, ρ] keep Tr ρ_k and Tr ρ_k^2 at every k, and the intraband term moves
    charge between k only.  w_k is zero on the padding, so the padded rows do
    not enter.  No ΔΣ evaluation; two reductions over k per sample.
    """
    w = ops.w_k.astype(jnp.float64)[None, :]                             # (1, nk)
    tr = jnp.sum(w * jnp.real(jnp.trace(rho, axis1=-2, axis2=-1)), axis=1)
    pur = jnp.sum(w * jnp.real(jnp.sum(rho * jnp.swapaxes(rho, -1, -2), axis=(-2, -1))),
                  axis=1)
    return jnp.stack([tr, pur, jnp.sqrt(jnp.sum(jnp.abs(efield) ** 2, axis=-1))], axis=-1)


def _magnus_parts(cfg: PropagatorConfig):
    """(nodes, flow, remainder fractions or None, history offsets, bootstrap steps)."""
    cfm4 = cfg.method == "cfm4"
    nodes = ti.CFM4_NODES if cfm4 else ti.MIDPOINT_NODES
    flow = ti.ip_cf4_flow if cfm4 else ti.ip_midpoint_flow
    fractions = None if not cfg.remainder else (ti.TRIPLE_JUMP if cfm4 else (1.0,))
    hist = ((ti.CFM4_HISTORY if cfg.predictor == "cubic" else ti.CFM4_HISTORY[2:])
            if cfm4 else ti.MIDPOINT_HISTORY)
    return nodes, flow, fractions, hist, -(-len(hist) // len(nodes))


def _fixed_parts(cfg: PropagatorConfig, ph, h=None):
    """(step, dense, state_of) of a fixed-step method (see ti.fixed_advance).

    ``h`` : the step the split's Newton matrix is built for (a traced scalar
    in :func:`step_growth`, so one program serves every probed step); None
    is cfg.dt.
    """
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

        if _split(cfg, ph):
            fast = _fast_part(cfg, ph, nodes, cfg.dt if h is None else h)
            n_s, n_hist = ph["fast"]["n"], len(hist_offsets)

            def step(t, c, h, t_next):
                x, hist, hist_a, k, st = c
                ha = jnp.reshape(hist_a, (n_hist, x.shape[0], n_s))
                guess = (tuple(jnp.tensordot(w, hist, axes=1)),
                         jnp.moveaxis(jnp.tensordot(w, ha, axes=1), 0, 1))
                n_c = jnp.where(k < n_boot, cfg.n_corr + BOOTSTRAP_EXTRA, cfg.n_corr)
                x1, g, (sl, a), solve = ti.ip_magnus_step(
                    ph["field"], ph["dsig_dev"], inner, ph["free"], t, x, h, guess, nodes=nodes,
                    n_corr=n_c, fast=fast)
                hist = ph["free"](jnp.concatenate([hist[len(nodes):], jnp.stack(sl)]), h)
                ha = jnp.concatenate([ha[len(nodes):], jnp.moveaxis(a, 1, 0)])
                return (pin(x1), hist, jnp.reshape(ha, hist_a.shape), k + 1,
                        _fast_stats(st, *solve)), g

            def dense(t, c0, c1, g, th):
                gen = ti.frozen_generator(ph["field"], ph["free"], t, cfg.dt, nodes, g)
                return ph["rho_eq"] + inner(gen, t, c0[0], th * cfg.dt)
            return step, dense, (lambda c: ph["rho_eq"] + c[0])

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


def _split(cfg: PropagatorConfig, ph) -> bool:
    """cfm4 with a stiff basis on the operators: the split step."""
    return cfg.method == "cfm4" and int(cfg.stiff) > 0 and ph["fast"] is not None


def _fast_part(cfg: PropagatorConfig, ph, nodes, h) -> ti.FastPart:
    """The FastPart of ops.stiff with its Newton matrix (1 - J)^{-1} at step h.

    J (``ti.fast_jacobian``) depends on h, the basis and the band energies
    only: built once per traced program, outside the step loop (3 n_s
    projections of n_s states).
    """
    f = ph["fast"]
    fast = ti.FastPart(project=f["project"], combine=f["combine"], reference=f["reference"],
                       newton=None, tol=float(cfg.stiff_tol), max_iter=int(cfg.stiff_iter))
    jac = ti.fast_jacobian(fast, ph["free"], h, nodes, f["n"])
    return fast._replace(newton=jnp.linalg.inv(jnp.eye(jac.shape[0], dtype=jac.dtype) - jac))


#: The closing-solve counters a split carry holds (replicated scalars):
#: node-map evaluations (sum, max), unconverged steps, worst relative residual.
def _fast_stats0():
    return (jnp.int64(0), jnp.int32(0), jnp.int32(0), jnp.float64(0.0))


def _fast_stats(st, it, rel, ok):
    tot, mx, fails, worst = st
    rel = jnp.where(jnp.isnan(rel), jnp.inf, rel)
    return (tot + it.astype(tot.dtype), jnp.maximum(mx, it), fails + jnp.where(ok, 0, 1).astype(fails.dtype),
            jnp.maximum(worst, rel))


def _split_carry(ph, x0, seeds, k0, t0):
    """(x0, slow history, coefficient history, k0, counters) of the split cfm4 step.

    ``seeds`` : the states whose dSigma fills the history (initial_carry: x0
    repeated; step_growth: independent ones).  Slow part S = dSigma[x] -
    Σ_j <u_j, x> b_j, coefficients <u_j, x> (the history of the fast
    initial guess, stored (n_hist, ntr * n_s): a replicated leaf).
    """
    f = ph["fast"]
    a = [f["project"](y) for y in seeds]
    sl = jnp.stack([ph["dsig_dev"](t0, y) - f["combine"](ai) for y, ai in zip(seeds, a)])
    ha = jnp.stack(a)
    return (x0, sl, jnp.reshape(ha, (ha.shape[0], -1)), jnp.int32(k0), _fast_stats0())


def fast_solve_counters(carry) -> dict:
    """The split carry's closing-solve counters on the host (every rank)."""
    tot, mx, fails, worst = (np.asarray(gather_to_host(v)) for v in carry[4])
    return dict(evaluations=int(tot), max_evaluations=int(mx), unconverged=int(fails),
                worst_rel=float(worst))


def initial_carry(cfg: PropagatorConfig, rho0, ops, mparams, model, drive, mesh):
    """The integrator's carry at t0 (one jitted call).

    dp54 : SciPy's state.   rk4 : (rho, rhs).   etdrk4 : (rho, coef, N).
    magnus2 / cfm4 : (Delta rho, dSigma node history in the interaction
    picture of the next step's start, step counter); magnus2 in the whole_rho
    frame: (rho, the last three midpoint dSigma, step counter); cfm4 with the
    stiff split: (Delta rho, slow node history, coefficient history, step
    counter, closing-solve counters).
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
        n_hist = len(_magnus_parts(cfg)[3])
        if _split(cfg, ph):
            return _split_carry(ph, x0, [x0] * n_hist, 0, cfg.t0)
        b0 = ph["dsig_dev"](cfg.t0, x0)
        return (x0, jnp.stack([b0] * n_hist), jnp.int32(0))
    return jax.jit(build)(rho0, ops, mparams)


# ---------------------------------------------------------------------------
# The step limit of the corrected Magnus steps
# ---------------------------------------------------------------------------

def coupling_radius(rho0, ops, model: SelfEnergyModel, drive, mesh, *, t0: float,
                    n_iter: int = COUPLING_POWER_ITERATIONS, seed: int = 0,
                    timed: bool = False):
    """Power-iteration estimate of the spectral radius σ (Ha) of the coupling map.

        L(x) = -i [dSigma[rho^eq + x] - dSigma[rho^eq], rho^eq]
             = -i (dSigma[x] - dSigma[0]) ∘ (f_m - f_n)

    the linearized coupling of the Magnus steps (the field is left out: it
    is weak where the limit binds, and E·r is exact in the free flow's
    frame anyway).  With a stiff basis on the operators (``ops.stiff``,
    :func:`stiff_basis`) it is the radius of the slow map L(1 - Π),

        L(1 - Π) x = -i (dSigma[x] - dSigma[0] - Σ_j <u_j, x> b_j) ∘ (f_m - f_n),

    the part the predictor-corrector of the split step treats explicitly
    (Π = Σ_j |u_j><u_j|; for an invariant span(u) its spectrum is L's with
    the stiff eigenvalues replaced by zeros).  In the blocks of (span u, its
    complement) L(1 - Π) = [[0, L12], [0, L22]]: its nonzero spectrum is
    L22's, and the iteration runs on the compression (1 - Π) L (1 - Π) = L22,
    which has that spectrum without the transient of the slow-to-fast block
    L12 (a non-normal L12 makes the plain iterate's norm overshoot the radius
    for many iterations).  From a fixed random x (cv and vc blocks, one trial,
    logical k), x <- L(L(x)) / ||L(L(x))||, σ_j = (||L(L x_j)|| / ||x_j||)^(1/2):
    the eigenvalues of L are ±iω (real ω for a Hermitian coupling, TDA or a
    stable full BSE), so L^2 has the dominant real eigenvalue -σ^2 and σ_j
    rises to σ.  One jitted call, 2 n_iter dSigma evaluations on the
    model's own operands and layout.  A map without coupling (IP, or a
    kernel of zeros) has ||L(L(x))|| <= COUPLING_ZERO_FLOOR^2 ||x||: the
    iteration keeps x and the estimate is 0, never 0/0, and a radius below
    the floor (1e-12 Ha, 2.7e-11 eV: it would bound only steps above 1e11
    a.u.) returns σ = 0.0.  A map that returns non-finite values gives
    σ = nan (the gate refuses it by name).  Returns (σ, the n_iter estimates);
    ``timed = True`` adds dict(compile_s, exec_s, dsigma): the program is
    lowered and compiled ahead of time and its one execution timed, so
    exec_s / dsigma (2 n_iter + 1 calls; the rest of an iteration is
    elementwise) is the measured wall of one ΔΣ call at no extra cost.
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
        return (0.0, np.zeros(int(n_iter))) + (
            (dict(compile_s=0.0, exec_s=0.0, dsigma=0),) if timed else ())
    x0 = place_state(x, mesh)
    gap_d = place_state(np.broadcast_to(gap, x.shape).astype(complex), mesh)
    zero = place_state(np.zeros_like(x), mesh)

    def run(x, gap, zero, ops, mparams):
        ph = _physics(ops, mparams, model, drive, mesh)
        d0 = ph["dsig_dev"](t0, zero)
        if ph["fast"] is None:
            lmap = lambda y: ph["pin"](-1j * (ph["dsig_dev"](t0, y) - d0) * gap)
        else:                                           # the slow map, compressed: (1 - Pi) L (1 - Pi)
            f = ph["fast"]
            deflate = lambda z: z - f["expand"](f["project"](z))
            lmap = lambda y: deflate(ph["pin"](-1j * (ph["dsig_dev"](t0, deflate(y)) - d0) * gap))
        nrm = lambda y: jnp.sqrt(jnp.sum(jnp.abs(y) ** 2))

        def body(y, _):
            z = lmap(lmap(y))
            nz, ny = nrm(z), nrm(y)
            dead = nz <= COUPLING_ZERO_FLOOR ** 2 * ny       # a zero map: keep y, estimate 0
            return (jnp.where(dead, y, z / jnp.where(dead, 1.0, nz)),
                    jnp.where(dead, 0.0, jnp.sqrt(nz / ny)))
        _, est = jax.lax.scan(body, x / nrm(x), None, length=int(n_iter), unroll=1)
        return est

    args = (x0, gap_d, zero, ops, model.params)
    t_lower = time.perf_counter()
    prog = jax.jit(run).lower(*args).compile()
    t_run = time.perf_counter()
    est = np.asarray(gather_to_host(prog(*args)))
    t_done = time.perf_counter()
    sigma = float(est[-1])
    sigma = 0.0 if 0.0 <= sigma < COUPLING_ZERO_FLOOR else sigma
    if not timed:
        return sigma, est
    return sigma, est, dict(compile_s=t_run - t_lower, exec_s=t_done - t_run,
                            dsigma=2 * int(n_iter) + 1)


def magnus_dt_limit(cfg: PropagatorConfig, sigma: float) -> tuple:
    """(dt_max a.u., y_lim): the largest stable step of cfg's Magnus scheme for radius σ.

    y_lim = :func:`solvers.time_integrators.magnus_stability_limit` of the
    scheme's nodes, predictor and n_corr; dt_max = y_lim / (STABILITY_MARGIN σ).
    """
    nodes, _, _, hist, _ = _magnus_parts(cfg)
    y_lim = ti.magnus_stability_limit(nodes, hist, int(cfg.n_corr))
    return (np.inf if sigma <= 0.0 else y_lim / (STABILITY_MARGIN * sigma)), y_lim


def step_growth(cfg: PropagatorConfig, ops, model: SelfEnergyModel, mesh, *, h: float,
                t0: float, n_steps: int = PROBE_STEPS, eps: float = None, seed: int = 1,
                counters: bool = False):
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
    jitted call, h traced.  With the stiff split (cfg.stiff, ops.stiff) the
    probe runs the split step, its history seeded the same way (slow parts
    and coefficients of the independent seeds), and ``counters = True``
    returns the closing-solve counters as a third value (a probe whose
    closing solve failed measured another scheme: the gate refuses it).
    Returns (g, the norms after each step).
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

    def run(x0, xh, ops, mparams, h):
        ph = _physics(ops, mparams, model, off, mesh)
        step, _, _ = _fixed_parts(cfg, ph, h)
        base = ph["rho_eq"] if whole else 0.0
        if _split(cfg, ph):            # past the bootstrap: the predictor acts at once
            carry = _split_carry(ph, x0, [xh[j:j + 1] for j in range(n_hist)], n_boot, t0)
        else:
            hist = jnp.stack([ph["dsig_dev"](t0, xh[j:j + 1]) for j in range(n_hist)])
            carry = (base + x0, hist, jnp.int32(n_boot))

        def body(c, i):
            t = t0 + i * h
            c, _ = step(t, c, h, t + h)
            return c, jnp.sqrt(jnp.sum(jnp.abs(c[0] - base) ** 2))
        c, norms = jax.lax.scan(body, carry, jnp.arange(int(n_steps)), unroll=1)
        return norms, (c[4] if len(c) == 5 else ())

    norms, st = jax.jit(run)(x0, xh, ops, model.params, jnp.asarray(float(h)))
    norms = np.asarray(gather_to_host(norms))
    m = int(n_steps) // 2
    g = float((norms[-1] / norms[m - 1]) ** (1.0 / (int(n_steps) - m)))
    if not counters:
        return g, norms
    return g, norms, (fast_solve_counters((None,) * 4 + (st,)) if st else None)


def _coherence_start(ops, seed: int = 0):
    """A random Hermitian deviation on the cv and vc blocks of the logical k (host), and the gap.

    The start vector of :func:`coupling_radius` (same seed, same masks):
    returns (x (1, nk_pad, nb, nb), gap (1, nk_pad, nb, nb) = f_m - f_n).
    """
    nk_pad, nb = int(ops.e_qp.shape[0]), int(ops.e_qp.shape[1])
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, nk_pad, nb, nb)) + 1j * rng.standard_normal((1, nk_pad, nb, nb))
    f = np.real(np.diagonal(np.asarray(gather_to_host(ops.rho_eq)), axis1=-2, axis2=-1))
    gap = (f[:, None, :] - f[:, :, None])[None]
    w = np.asarray(gather_to_host(ops.w_k)) > 0.0
    x = x * (np.abs(gap) > 0.5) * w[None, :, None, None]
    return 0.5 * (x + np.conj(np.swapaxes(x, -1, -2))), gap


def coupling_arnoldi(ops, model: SelfEnergyModel, mesh, *, t0: float, krylov: int,
                     seed: int = 0):
    """Arnoldi on the coupling map L of :func:`coupling_radius`, on Hermitian deviations.

    L(x) = -i (dSigma[x] - dSigma[0]) ∘ (f_m - f_n) maps Hermitian deviations
    to Hermitian deviations and is real-linear there, so the Krylov space of
    the Hermitian start vector (:func:`_coherence_start`) is spanned by
    Hermitian vectors with REAL Arnoldi coefficients h_ij = Re <v_i, L v_j>
    (classical Gram-Schmidt twice; each new vector Hermitized): the Ritz
    values come in the real pairs (λ, conj λ) and (λ, -λ) of L.  One jitted
    call, ``krylov`` dSigma evaluations.  Returns (V (krylov + 1, nk_pad, nb,
    nb) on the mesh, k-sharded, orthonormal; H (krylov + 1, krylov) float64,
    host).
    """
    x, gap = _coherence_start(ops, seed)
    if not np.any(x):
        raise ValueError("GATE td_stiff_basis: got: no cv pair on a logical k; want: a coupled "
                         "window; why: the coupling map is empty, nothing is stiff; fix: "
                         "stiff_modes = 0")
    m = int(krylov)
    v0 = place_state(x / np.linalg.norm(x), mesh)
    gap_d = place_state(gap.astype(complex), mesh)
    spec = NamedSharding(mesh, _spec(4, 1))
    off = lambda t: jnp.zeros((1, 3))

    def run(v0, gap, ops, mparams):
        ph = _physics(ops, mparams, model, off, mesh)
        d0 = ph["dsig_dev"](t0, jnp.zeros_like(v0))
        lmap = lambda y: ph["pin"](-1j * (ph["dsig_dev"](t0, y) - d0) * gap)
        pinv = lambda a: jax.lax.with_sharding_constraint(a, spec)
        V = pinv(jnp.zeros((m + 1,) + v0.shape[1:], v0.dtype).at[0].set(v0[0]))
        H = jnp.zeros((m + 1, m), jnp.float64)

        def body(j, c):
            V, H = c
            w = lmap(V[j][None])[0]
            col = jnp.zeros((m + 1,), jnp.float64)
            for _ in range(2):                                   # CGS2, real coefficients
                hj = jnp.real(jnp.einsum("ikab,kab->i", jnp.conj(V), w))
                w = w - jnp.einsum("i,ikab->kab", hj.astype(w.dtype), V)
                col = col + hj
            w = 0.5 * (w + jnp.conj(jnp.swapaxes(w, -1, -2)))
            beta = jnp.sqrt(jnp.sum(jnp.abs(w) ** 2))
            H = H.at[:, j].set(col.at[j + 1].set(beta))
            V = pinv(V.at[j + 1].set(w / jnp.where(beta > 0.0, beta, 1.0)))
            return V, H
        return jax.lax.fori_loop(0, m, body, (V, H))

    V, H = jax.jit(run)(v0, gap_d, ops, model.params)
    return V, np.asarray(gather_to_host(H))


def stiff_basis(ops, model: SelfEnergyModel, mesh, *, t0: float, n_s: int, krylov: int = 0,
                arnoldi=None, seed: int = 0) -> dict:
    """The stiff split's basis: u_1..u_n spanning L's invariant subspace of the n largest |λ|.

    From the Arnoldi relation L V_m = V_{m+1} H (:func:`coupling_arnoldi`,
    ``krylov`` steps; ``arnoldi`` = a (V, H) already computed, reused for
    several n), the real Schur form H_m = Z T Z^T sorted so that the Ritz
    values above the cut lead; u = V_m Z[:, :n] (Hermitian, orthonormal up
    to rounding, re-orthonormalized by Cholesky QR) spans the invariant
    subspace of those Ritz values, both members of every (λ, -λ) and
    (λ, conj λ) group (n is raised to the end of a group of equal moduli,
    relative gap STIFF_GROUP_GAP).  Its residual
    ||L u - u T_11|| = h_{m+1,m} ||Z[m-1, :n]|| is reported: the split is exact
    for ANY u (dSigma[x] = dSigma[(1 - Π) x] + Σ_j <u_j, x> b_j), the residual
    only says how much stiffness the slow map keeps.  b_j = dSigma[u_j] -
    dSigma[0] (n dSigma evaluations).  Returns dict(u, b (n, nk_pad, nb, nb)
    on the mesh), n_s, ritz_ev (all Ritz values, eV, |λ| descending), cut_ev,
    residual_ev, krylov, dsigma (evaluations made here).
    """
    import scipy.linalg as sla
    from tdagw.tdgw_io import HA_IN_EV
    n = int(n_s)
    if arnoldi is None:
        m = int(krylov) or max(STIFF_KRYLOV_MIN, STIFF_KRYLOV_PER_MODE * n)
        V, H = coupling_arnoldi(ops, model, mesh, t0=t0, krylov=m, seed=seed)
        calls = m + 1
    else:
        (V, H), calls = arnoldi, 0
        m = int(H.shape[1])
    if not 0 < n < m:
        raise ValueError(f"GATE td_stiff_modes: got: stiff_modes = {n} with {m} Arnoldi steps; "
                         f"want: 0 < stiff_modes < stiff_krylov; fix: raise stiff_krylov")
    hm = H[:m, :m]
    lam = np.linalg.eigvals(hm)
    order = np.argsort(-np.abs(lam))
    lam, mods = lam[order], np.abs(lam[order])
    while n < m - 1 and mods[n - 1] - mods[n] <= STIFF_GROUP_GAP * mods[0]:
        n += 1                                           # never cut through a pair
    cut = 0.5 * (mods[n - 1] + mods[n])
    _, z, sdim = sla.schur(hm, output="real", sort=lambda re, im: np.hypot(re, im) > cut)
    if int(sdim) != n:
        raise ValueError(f"GATE td_stiff_basis: got: {sdim} Schur vectors above the cut "
                         f"{cut * HA_IN_EV:.6g} eV for {n} Ritz values; want: the same count; "
                         f"why: the real Schur reordering is ill-conditioned at this cut; fix: "
                         f"another stiff_modes (a gap in the Ritz moduli) or more stiff_krylov")
    q = z[:, :n]
    resid = abs(H[m, m - 1]) * np.linalg.norm(q[m - 1, :])
    qd = jnp.asarray(q, jnp.float64)
    spec = NamedSharding(mesh, _spec(4, 1))
    off = lambda t: jnp.zeros((1, 3))

    def build(V, qd, ops, mparams):
        ph = _physics(ops, mparams, model, off, mesh)
        u = jnp.einsum("ij,ikab->jkab", qd.astype(V.dtype), V[:m])
        u = 0.5 * (u + jnp.conj(jnp.swapaxes(u, -1, -2)))
        gram = jnp.real(jnp.einsum("ikab,jkab->ij", jnp.conj(u), u))
        lo = jnp.linalg.cholesky(0.5 * (gram + gram.T))
        u = jnp.einsum("ij,jkab->ikab", jnp.linalg.inv(lo).astype(u.dtype), u)
        u = jax.lax.with_sharding_constraint(u, spec)
        d0 = ph["dsig_dev"](t0, jnp.zeros_like(u[:1]))
        b = jax.lax.map(lambda uj: (ph["dsig_dev"](t0, uj[None]) - d0)[0], u)
        return u, jax.lax.with_sharding_constraint(b, spec)

    u, b = jax.jit(build)(V, qd, ops, model.params)
    return dict(u=u, b=b, n_s=n, ritz_ev=lam * HA_IN_EV, cut_ev=cut * HA_IN_EV,
                residual_ev=float(resid) * HA_IN_EV, krylov=m, dsigma=calls + n + 1)


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


def _state_rho(cfg: PropagatorConfig, ops, carry):
    """The density ρ of a carry, on the device (the propagator's layout)."""
    if cfg.method == "dp54":
        return carry.y
    if cfg.method in ("rk4", "etdrk4") or (cfg.method == "magnus2" and cfg.frame == "whole_rho"):
        return carry[0]
    return ops.rho_eq[None] + carry[0]


def _refresh_program(cfg: PropagatorConfig, model: SelfEnergyModel, drive, mesh):
    """``run(carry, ops, mparams, delta, t) -> carry``: the carry after a refresh, one jit per run.

    A refresh replaces ``ops.collision`` and the model's held term (by ``delta``,
    Ha, constant in ρ).  What the carry remembers of the old ones is made
    consistent with the new: the explicit methods' first-same-as-last
    derivative is re-evaluated (one ΔΣ call); the Magnus predictor history,
    node values of ΔΣ in the interaction picture of the next step's start at
    offsets o h, gains free(delta, -o h) for each, the history of a ΔΣ that had
    always held the new term (no ΔΣ call; exact, since the change is constant
    in ρ).  The collision flow itself is not in the history (it is the remainder).
    """
    def refresh_carry(carry, ops, mparams, delta, t):
        ph = _physics(ops, mparams, model, drive, mesh)
        if cfg.method == "dp54":
            return carry._replace(f=ph["rhs"](carry.t, carry.y), n_rhs=carry.n_rhs + 1)
        if cfg.method == "rk4":
            return (carry[0], ph["rhs"](t, carry[0]))
        if cfg.method == "etdrk4":
            return (carry[0], carry[1], ph["nonlin"](t, carry[0]))
        if cfg.method == "magnus2" and cfg.frame == "whole_rho":
            rho, hist, k = carry
            return (rho, hist + delta[None], k)
        offsets = _magnus_parts(cfg)[3]
        shift = jnp.stack([ph["free"](delta, -float(o) * cfg.dt) for o in offsets])
        if _split(cfg, ph):
            x, sl, ha, k, st = carry               # the slow history carries the held term
            return (x, sl + shift, ha, k, st)
        x, hist, k = carry
        return (x, hist + shift, k)
    return jax.jit(refresh_carry, donate_argnums=(0,))


def _fixed_counters(cfg: PropagatorConfig, s_start: int, s_end: int, fresh: bool) -> dict:
    """Steps and ΔΣ calls of a fixed-step run over steps [s_start, s_end).

    dsigma_per_step per step, the Magnus bootstrap's extra node calls on the
    steps it covers, and the initial carry's one call of a run from t0
    (``fresh``).  The per-chunk ``COUNTERS`` and the final counters are both
    this count.
    """
    n = int(s_end) - int(s_start)
    dsig = n * dsigma_per_step(cfg.method, cfg.n_corr)
    if cfg.method in ("magnus2", "cfm4"):
        boot = max(0, min(int(s_end), _magnus_parts(cfg)[4]) - int(s_start))
        dsig += boot * BOOTSTRAP_EXTRA * (1 if cfg.method == "magnus2" else 2)
    return dict(accepted=n, rejected=0, dsigma=dsig + (1 if fresh else 0))


def _jit_cache_size(programs) -> int:
    """Traced specializations held by this run's chunk programs (one per program
    unless an operand's structure, shape, dtype or sharding changed: a retrace)."""
    return int(sum(int(getattr(p, "_cache_size", lambda: 0)()) for p in programs.values()))


def run_propagation(cfg: PropagatorConfig, spec: SampleSpec, rho0, ops,
                    model: SelfEnergyModel, drive: Callable, mesh, *,
                    cell_volume: float, nspinor: int, count: int,
                    mtsteps: np.ndarray, on_chunk: Callable, resume=None, refresh=None):
    """Integrate over ``mtsteps`` in chunks; ``on_chunk`` writes each.

    ``on_chunk(i0, i1, samples_host, checkpoint, t_state)`` receives the
    samples of measurement indices [i0, i1) gathered to host (every rank;
    with the ``PROGRESS`` invariants of each sample and the run's cumulative
    ``COUNTERS`` at the chunk's end), a zero-argument callable
    ``checkpoint() -> (rho_host, carry_leaves, step)`` gathering the state,
    the integrator's whole carry and its step counter (every rank), and the
    state's time.  ``resume`` = dict(carry=
    host leaves, step=int, i_next=int) continues a checkpointed run from its
    next sample.  Returns (counters, carry): ``accepted`` steps taken by this
    run, ``rejected`` tries, ``dsigma`` evaluations in the steps,
    ``sample_dsigma`` (the samples' own ΔΣ calls), ``t_final``, ``refreshes``
    and ``refresh_dsigma`` (the collision refreshes and their ΔΣ calls),
    ``jit_cache`` (traced specializations of the chunk programs) and
    ``programs`` (chunk programs built); with the stiff split also the
    closing-solve counters (``fast``).  The per-chunk ``COUNTERS`` carry
    ``accepted``, ``rejected``, ``dsigma``, ``sample_dsigma``, ``refreshes``,
    ``refresh_dsigma`` and ``jit_cache`` so far, and ``new_program`` (whether
    the chunk built its program).  A chunk in which any step's closing
    solve missed its tolerance refuses (``td_stiff_solve``) before its
    samples reach ``on_chunk``.

    ``refresh`` (the held collision term, docs/theory/td_agw_dynamical.md §1.5,
    §7.1): None, or ``refresh(t, rho, ops, params) -> None | dict(ops, params,
    delta)``, called after every chunk but the last with the state's time and
    the state ρ on the devices (the propagator's layout; the host decision and
    its window Sigma are the hook's).  A dict replaces the operators (their
    ``collision``) and the model's parameters (its held term) for the chunks
    that follow, with the same shapes and shardings, so no chunk program is
    rebuilt (``COUNTERS``' ``jit_cache`` counts traced specializations);
    ``delta`` is the held ΔΣ term's change (Ha, ρ's layout), which the carry
    absorbs (:func:`_refresh_program`).  A collision on the operators needs
    ``cfg.remainder`` for the Magnus steps (its flow is the remainder).
    """
    nk_pad = int(ops.e_qp.shape[0])
    split = cfg.method == "cfm4" and int(cfg.stiff) > 0
    if split and ops.stiff is None:
        raise ValueError("GATE td_stiff_basis: got: stiff = %d with no basis on the operators; "
                         "want: ops.stiff from tdagw.propagate.stiff_basis; fix: internal "
                         "(stages.propagate builds it)" % int(cfg.stiff))
    if ops.collision is not None and cfg.method in ("magnus2", "cfm4") and not cfg.remainder:
        raise ValueError("GATE td_collision_remainder: got: a collision term with remainder = "
                         "False; want: remainder = True (the Magnus steps apply the held "
                         "collision flow as their remainder); fix: internal (stages.propagate "
                         "sets it)")
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
    needs = any(kind == "current" and not dict(opts).get("use_v0", False)
                for _, kind, opts in spec.kinds)        # one ΔΣ call per sample
    params = model.params
    refreshed = dict(count=0, dsigma=0)
    rebase = None

    def program(key, n_steps, last):
        """The jitted chunk of ``key`` and whether this call built it."""
        new = key not in programs
        if new:
            programs[key] = _chunk_program(cfg, spec, model.apply, model.name, (drive,),
                                           mesh, float(cell_volume), int(nspinor),
                                           int(count), int(n_steps), bool(last))
        return programs[key], new

    def after_chunk(carry, t):
        """The refresh hook at a chunk boundary: (carry, ops, params)."""
        nonlocal rebase
        if refresh is None:
            return carry, ops, params
        upd = refresh(float(t), _state_rho(cfg, ops, carry), ops, params)
        if upd is None:
            return carry, ops, params
        if rebase is None:
            rebase = _refresh_program(cfg, model, drive, mesh)
        delta = upd.get("delta")
        if delta is None:
            delta = jnp.zeros_like(_state_rho(cfg, ops, carry))
        carry = rebase(carry, upd["ops"], upd["params"], delta, jnp.asarray(t, jnp.float64))
        refreshed["count"] += 1
        refreshed["dsigma"] += int(cfg.method in ("dp54", "rk4", "etdrk4"))
        return carry, upd["ops"], upd["params"]

    if fixed:
        sstep, stheta, n_total = sample_plan(cfg, mtsteps)
        s_end = s_start
        for i0, i1, s0, s1 in chunk_plan(sstep, n_total, cfg.chunk, i_start, s_start):
            prog, new = program(("fixed", s1 - s0, i1 - i0), s1 - s0, False)
            carry, samples = prog(carry, ops, params, jnp.asarray(mtsteps[i0:i1]),
                                  jnp.asarray(sstep[i0:i1]), jnp.asarray(stheta[i0:i1]), s0)
            if split:
                fc = fast_solve_counters(carry)
                if fc["unconverged"]:
                    from tdagw.tdgw_io import T_IN_ATTOSEC
                    raise ValueError(
                        f"GATE td_stiff_solve: got: {fc['unconverged']} step(s) up to t = "
                        f"{(cfg.t0 + s1 * cfg.dt) * T_IN_ATTOSEC:.6g} as whose closing solve of the "
                        f"{int(cfg.stiff)} stiff coefficients missed rel. {cfg.stiff_tol:.1e} in "
                        f"{cfg.stiff_iter} node-map evaluations (worst {fc['worst_rel']:.3e}); "
                        f"want: the fast part converged at the nodes of every step; why: an "
                        f"unconverged fast part leaves the stiff modes explicit, the step is not "
                        f"the split scheme and its stability is unknown; fix: a smaller dt, more "
                        f"stiff_iter, or stiff_modes = 0")
            host = jax.tree.map(gather_to_host, samples)
            host[COUNTERS] = dict(_fixed_counters(cfg, s_start, s1, resume is None),
                                  sample_dsigma=(i1 - i_start) if needs else 0,
                                  new_program=new, jit_cache=_jit_cache_size(programs),
                                  refreshes=refreshed["count"],
                                  refresh_dsigma=refreshed["dsigma"])
            state = (lambda c=carry, s=s1: (gather_to_host(c[0] if cfg.method in ("rk4", "etdrk4")
                                                          else ops.rho_eq[None] + c[0]),
                                            carry_to_host(c), s))
            on_chunk(i0, i1, host, state, cfg.t0 + s1 * cfg.dt)
            s_end = s1
            if s1 < n_total:
                carry, ops, params = after_chunk(carry, cfg.t0 + s1 * cfg.dt)
        stats = _fixed_counters(cfg, s_start, s_end, resume is None)
        stats["t_final"] = cfg.t0 + s_end * cfg.dt
        if split:
            stats["fast"] = fast_solve_counters(carry)
    else:
        base = (0, 0, 0) if resume is None else tuple(
            int(gather_to_host(v)) for v in (carry.n_accepted, carry.n_rejected, carry.n_rhs))
        for i0 in range(i_start, n_meas, cfg.chunk):
            i1 = min(n_meas, i0 + cfg.chunk)
            last = i1 == n_meas
            prog, new = program(("dp54", i1 - i0, last), 0, last)
            times = jnp.asarray(mtsteps[i0:i1])
            carry, samples = prog(carry, ops, params, times,
                                  jnp.zeros(i1 - i0, jnp.int64), jnp.zeros(i1 - i0), 0)
            host = jax.tree.map(gather_to_host, samples)
            done = [int(gather_to_host(v))
                    for v in (carry.n_accepted, carry.n_rejected, carry.n_rhs)]
            host[COUNTERS] = dict(accepted=done[0] - base[0], rejected=done[1] - base[1],
                                  dsigma=done[2] - base[2],
                                  sample_dsigma=(i1 - i_start) if needs else 0,
                                  new_program=new, jit_cache=_jit_cache_size(programs),
                                  refreshes=refreshed["count"],
                                  refresh_dsigma=refreshed["dsigma"])
            state = (lambda c=carry: (gather_to_host(c.y), carry_to_host(c), 0))
            t_now = float(gather_to_host(carry.t))
            on_chunk(i0, i1, host, state, t_now)
            if not last:
                carry, ops, params = after_chunk(carry, t_now)
        if bool(gather_to_host(carry.failed)):
            raise RuntimeError(
                "GATE td_dp54_step_too_small: got: the RK45 controller's step fell "
                "below SciPy's min_step; want: an integrable run; why: the run "
                "cannot reach tf; fix: loosen atol/rtol or check the drive and model")
        stats = dict(accepted=int(gather_to_host(carry.n_accepted)) - base[0],
                     rejected=int(gather_to_host(carry.n_rejected)) - base[1],
                     dsigma=int(gather_to_host(carry.n_rhs)) - base[2],
                     t_final=float(gather_to_host(carry.t)))
    stats["sample_dsigma"] = (n_meas - i_start) if needs else 0
    stats["refreshes"] = refreshed["count"]
    stats["refresh_dsigma"] = refreshed["dsigma"]
    stats["jit_cache"] = _jit_cache_size(programs)
    stats["programs"] = len(programs)
    return stats, carry
