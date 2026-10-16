"""tdagw/propagate.py — the jitted, k-sharded propagator of the TD-aGW EOM.

Integrates Eq. (1) of :mod:`tdagw.eom` with one of five integrators of
:mod:`solvers.time_integrators`, chosen by name:

========  ==========================================================  =========
method    scheme                                                      dSigma/step
========  ==========================================================  =========
dp54      SciPy RK45 replica, adaptive, quartic dense output          6 per try
rk4       classical RK4, fixed step                                   4
etdrk4    Krogstad ETDRK4; -i[H0, .] - Gamma exact (elementwise)       4
magnus2   exponential midpoint, self-consistent midpoint H            n_corr
cfm4      4th-order commutator-free Magnus, Gauss-node H by cubic     2 n_corr
          extrapolation + n_corr corrections
========  ==========================================================  =========

The Magnus steps are unitary: H(t) is diagonalized per k at every
exponential (``jnp.linalg.eigh`` on the local k block inside a
``shard_map``, so no device holds another's matrices), and
exp(-i tau H) rho exp(i tau H) = V (e^{-i tau w} ∘ V^H rho V ∘ e^{i tau w}) V^H.
Dissipation (Gamma) and the intraband commutator are not of the form
[H_k, .], so for the Magnus family they enter by Strang splitting (exact
exponential relaxation, RK4 for the intraband term); ETDRK4 carries Gamma
exactly in its linear part.

Layout: rho, H and the k-dependent operators are k-sharded over the whole
mesh, ``P(None, ('x','y'), None, None)`` for ``(ntr, nk, nb, nb)``; nk is
padded to a multiple of the device count with zero operators and zero
weight, which every term of Eq. (1) leaves at zero.  dSigma is the injected
:class:`tdagw.self_energy.SelfEnergyModel` and may move data however it likes.
A chunk of samples is ONE jitted call (a ``lax`` loop over steps, samples
written into device buffers); the host sees the state only between chunks.
"""
from __future__ import annotations

import functools
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
           "place_operators", "place_kernel", "make_conjugate", "run_propagation"]

METHODS = ("dp54", "rk4", "etdrk4", "magnus2", "cfm4")
#: Bootstrap: while the predictor history is incomplete (first steps) the
#: Magnus steps take this many extra corrections.
BOOTSTRAP_EXTRA = 3
K_AXES = ("x", "y")


def dsigma_per_step(method: str, n_corr: int) -> int:
    """dSigma evaluations per accepted fixed step (dp54: per try, 6)."""
    return {"dp54": 6, "rk4": 4, "etdrk4": 4, "magnus2": n_corr,
            "cfm4": 2 * n_corr}[method]


class PropagatorConfig(NamedTuple):
    """Static integration settings, Hartree atomic units.

    method : one of METHODS.
    t0, tf, dt_measure : run window and sample spacing (TDGW t0, tf, mdt).
    dt : fixed step (non-dp54); dt_measure = n_sub * dt.
    n_corr : Magnus corrections per step (>= 1).
    atol, rtol, max_step : dp54 controller (SciPy semantics; max_step inf ok).
    chunk : samples per jitted call.
    remainder : the Strang remainder (Gamma or intraband) is nonzero; a
        purely Hamiltonian run skips it (Magnus steps exactly unitary).
    """
    method: str
    t0: float
    tf: float
    dt_measure: float
    dt: float
    n_sub: int
    n_corr: int
    atol: float
    rtol: float
    max_step: float
    chunk: int
    remainder: bool


class SampleSpec(NamedTuple):
    """What one sample measures (TDGW measurement workers).

    kinds : tuple of (name, kind, opts) with kind in Efield, polarization,
        current, density_matrix, lessG and opts a tuple of (key, value).
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


def place_kernel(kmat, mesh):
    """A dense (N, N) kernel, rows over the whole mesh (N padded by caller)."""
    return _place(kmat, mesh, 0)


def place_state(rho, mesh):
    """(ntr, nk_pad, nb, nb) host density -> the k-sharded state."""
    return _place(rho, mesh, 1)


# ---------------------------------------------------------------------------
# The unitary exponential, per k, inside shard_map
# ---------------------------------------------------------------------------

def make_conjugate(mesh) -> Callable:
    """exp(-i tau H) rho exp(+i tau H) for k-sharded (ntr, nk, nb, nb) stacks.

    Each device diagonalizes its own k block (``jnp.linalg.eigh``, H
    Hermitized first) and applies the phases in the eigenbasis:
    rho' = V (e^{-i tau w_n} (V^H rho V)_nm e^{+i tau w_m}) V^H.
    A pad k has H = 0 and rho = 0 and stays 0.
    """
    spec = _spec(4, 1)

    def local(h, rho, tau):
        h = 0.5 * (h + jnp.conj(jnp.swapaxes(h, -1, -2)))
        w, v = jnp.linalg.eigh(h)
        vh = jnp.conj(jnp.swapaxes(v, -1, -2))
        ph = jnp.exp(-1j * tau * w)
        x = (vh @ rho @ v) * (ph[..., :, None] * jnp.conj(ph)[..., None, :])
        return v @ x @ vh

    return shard_map(local, mesh=mesh, in_specs=(spec, spec, P()),
                     out_specs=spec, check_vma=False)


# ---------------------------------------------------------------------------
# One chunk: the physics callables, the step, the samples
# ---------------------------------------------------------------------------

def _physics(ops, mparams, model: SelfEnergyModel, drive: Callable, mesh):
    """The callables every integrator is built from (traced inside the jit)."""
    spec = NamedSharding(mesh, _spec(4, 1))
    pin = lambda x: jax.lax.with_sharding_constraint(x, spec)

    def dsig(t, rho):
        return pin(model.apply(mparams, rho, t))

    def rhs(t, rho):
        return pin(eom.eom_rhs(ops, drive(t), dsig(t, rho), rho))

    def explicit(t):
        return pin(eom.field_hamiltonian(ops, drive(t)))

    lam = (-1j * (ops.e_qp[:, :, None] - ops.e_qp[:, None, :])
           - ops.gamma[None].astype(jnp.complex128))[None]          # (1, nk, nb, nb)

    def nonlin(t, rho):
        e = drive(t)
        drv = eom.commutator(eom.field_coupling(ops, e) + dsig(t, rho), rho)
        if ops.intra is not None:
            drv = drv + eom.field_intraband(ops.intra, e, rho)
        return pin(-1j * drv + ops.gamma.astype(rho.dtype) * ops.rho_eq[None])

    def remainder(t, rho, tau):
        """The non-commutator flow: Gamma relaxation (exact) and intraband (RK4)."""
        decay = jnp.exp(-0.5 * tau * ops.gamma).astype(rho.dtype)
        rho = ops.rho_eq[None] + decay * (rho - ops.rho_eq[None])
        if ops.intra is not None:
            f = lambda s, y: -1j * eom.field_intraband(ops.intra, drive(s), y)
            rho = ti.rk4_step(f, t, rho, tau)
        return pin(ops.rho_eq[None] + decay * (rho - ops.rho_eq[None]))

    return dict(dsig=dsig, rhs=rhs, explicit=explicit, nonlin=nonlin,
                lam=lam, remainder=remainder, pin=pin)


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
            else:
                raise ValueError(f"unknown measurement kind {kind!r}")
        return out
    return fn


def _history_weights():
    return (jnp.asarray(ti.extrapolation_weights(ti.MIDPOINT_HISTORY, [0.5])[0]),
            jnp.asarray(ti.extrapolation_weights(ti.CFM4_HISTORY,
                                                 list(ti.CFM4_NODES))))


def _fixed_step_fn(cfg: PropagatorConfig, ph, conjugate):
    """step(t, carry, h) -> carry for the fixed-step methods."""
    w_mid, w_cf = _history_weights()
    pin = ph["pin"]

    if cfg.method == "rk4":
        return lambda t, c, h: (pin(ti.rk4_step(ph["rhs"], t, c[0], h)),) + c[1:]
    if cfg.method == "etdrk4":
        def step(t, c, h):
            coef = c[1]
            return (pin(ti.etdrk4_step(ph["nonlin"], coef, t, c[0], h)), coef)
        return step
    if cfg.method == "magnus2":
        def core(t, y, h, hist, k):
            guess = jnp.tensordot(w_mid, hist, axes=1)
            n_it = jnp.where(k < 3, cfg.n_corr + BOOTSTRAP_EXTRA, cfg.n_corr)
            return ti.magnus_midpoint_step(ph["explicit"], ph["dsig"], conjugate,
                                           t, y, h, guess, n_iter=n_it)

        def step(t, c, h):
            y, hist, k = c
            c_fn = lambda tt, yy, hh: core(tt, yy, hh, hist, k)
            y, b = (ti.strang_step(c_fn, ph["remainder"], t, y, h)
                    if cfg.remainder else c_fn(t, y, h))
            return (pin(y), jnp.concatenate([hist[1:], b[None]]), k + 1)
        return step
    if cfg.method == "cfm4":
        def core(t, y, h, hist, k):
            g = jnp.tensordot(w_cf, hist, axes=1)
            n_c = jnp.where(k < 2, cfg.n_corr + BOOTSTRAP_EXTRA, cfg.n_corr)
            return ti.cfm4_step(ph["explicit"], ph["dsig"], conjugate, t, y, h,
                                (g[0], g[1]), n_corr=n_c)

        def step(t, c, h):
            y, hist, k = c
            c_fn = lambda tt, yy, hh: core(tt, yy, hh, hist, k)
            y, (b1, b2) = (ti.strang_step(c_fn, ph["remainder"], t, y, h)
                           if cfg.remainder else c_fn(t, y, h))
            return (pin(y), jnp.concatenate([hist[2:], jnp.stack([b1, b2])]), k + 1)
        return step
    raise ValueError(cfg.method)


def initial_carry(cfg: PropagatorConfig, rho0, ops, mparams, model, drive, mesh):
    """The integrator's carry at t0 (one jitted call)."""
    def build(rho0, ops, mparams):
        ph = _physics(ops, mparams, model, drive, mesh)
        if cfg.method == "dp54":
            return ti.dp54_initial_state(ph["rhs"], cfg.t0, rho0, cfg.dt_measure)
        if cfg.method == "rk4":
            return (rho0,)
        if cfg.method == "etdrk4":
            return (rho0, ti.etdrk4_coefficients(ph["lam"], cfg.dt))
        b0 = ph["dsig"](cfg.t0, rho0)
        n_hist = 3 if cfg.method == "magnus2" else 4
        return (rho0, jnp.stack([b0] * n_hist), jnp.int32(0))
    return jax.jit(build)(rho0, ops, mparams)


@functools.lru_cache(maxsize=None)
def _chunk_program(cfg: PropagatorConfig, spec: SampleSpec, model_apply,
                   model_name: str, drive_key, mesh, cell_volume: float,
                   nspinor: int, count: int, n_samples: int, last: bool):
    """The jitted chunk for one static configuration (cached by its key)."""
    drive = drive_key[0]
    model = SelfEnergyModel(apply=model_apply, params=None, linear=True,
                            name=model_name)
    conjugate = make_conjugate(mesh)

    def run(carry, ops, mparams, sample_times, step0):
        ph = _physics(ops, mparams, model, drive, mesh)
        sample_fn = _sample(ph, ops, spec, drive, cell_volume=cell_volume,
                            nspinor=nspinor)
        if cfg.method == "dp54":
            return ti.dp54_advance(ph["rhs"], carry, t_bound=cfg.tf,
                                   max_step=cfg.max_step, atol=cfg.atol,
                                   rtol=cfg.rtol, count=count,
                                   sample_times=sample_times,
                                   sample_fn=sample_fn, run_to_bound=last)
        step = _fixed_step_fn(cfg, ph, conjugate)
        return ti.fixed_advance(step, carry, t_start=cfg.t0, h=cfg.dt,
                                n_sub=cfg.n_sub, n_samples=n_samples,
                                sample_fn=lambda t, c: sample_fn(t, c[0]),
                                step0=step0)

    return jax.jit(run, donate_argnums=(0,))


def run_propagation(cfg: PropagatorConfig, spec: SampleSpec, rho0, ops,
                    model: SelfEnergyModel, drive: Callable, mesh, *,
                    cell_volume: float, nspinor: int, count: int,
                    mtsteps: np.ndarray, on_chunk: Callable):
    """Integrate over ``mtsteps`` in chunks; ``on_chunk`` writes each.

    ``on_chunk(i0, i1, samples_host, state_host_fn, t_state)`` receives the
    samples of measurement indices [i0, i1) gathered to host (every rank),
    a zero-argument callable gathering the state, and the state's time.
    Returns a dict of counters (steps, rejected, rhs/dSigma evaluations).
    """
    carry = initial_carry(cfg, rho0, ops, model.params, model, drive, mesh)
    n_meas = int(mtsteps.size)
    fixed = cfg.method != "dp54"
    step0 = 0
    for i0 in range(0, n_meas, cfg.chunk):
        i1 = min(n_meas, i0 + cfg.chunk)
        last = i1 == n_meas
        prog = _chunk_program(cfg, spec, model.apply, model.name, (drive,),
                              mesh, float(cell_volume), int(nspinor), int(count),
                              int(i1 - i0), bool(last and not fixed))
        times = jnp.asarray(mtsteps[i0:i1])
        carry, samples = prog(carry, ops, model.params, times, step0)
        step0 += (i1 - i0) * cfg.n_sub
        host = jax.tree.map(gather_to_host, samples)
        if fixed:
            t_state = cfg.t0 + step0 * cfg.dt
            rho_fn = (lambda c=carry: gather_to_host(c[0]))
        else:
            t_state = float(gather_to_host(carry.t))
            rho_fn = (lambda c=carry: gather_to_host(c.y))
        on_chunk(i0, i1, host, rho_fn, t_state)
    needs = any(kind == "current" and not dict(opts).get("use_v0", False)
                for _, kind, opts in spec.kinds)
    sample_dsigma = n_meas if needs else 0
    if fixed:
        n_steps = step0
        stats = dict(accepted=n_steps, rejected=0,
                     dsigma=n_steps * dsigma_per_step(cfg.method, cfg.n_corr))
        if cfg.method in ("magnus2", "cfm4"):
            boot = min(n_steps, 3 if cfg.method == "magnus2" else 2)
            per = 1 if cfg.method == "magnus2" else 2
            stats["dsigma"] += boot * BOOTSTRAP_EXTRA * per + 1
        stats["t_final"] = cfg.t0 + n_steps * cfg.dt
        stats["sample_dsigma"] = sample_dsigma
        return stats, carry
    if bool(gather_to_host(carry.failed)):
        raise RuntimeError(
            "GATE td_dp54_step_too_small: got: the RK45 controller's step fell "
            "below SciPy's min_step; want: an integrable run; why: the run "
            "cannot reach tf; fix: loosen atol/rtol or check the drive and model")
    return dict(accepted=int(gather_to_host(carry.n_accepted)),
                rejected=int(gather_to_host(carry.n_rejected)),
                dsigma=int(gather_to_host(carry.n_rhs)),
                t_final=float(gather_to_host(carry.t)),
                sample_dsigma=sample_dsigma), carry
