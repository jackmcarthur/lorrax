"""bse/bse_davidson_helpers.py — initial-subspace + diagonal preconditioner
helpers tailored to the (block, nc_pad, nv_pad, nk) BSE vector layout.

These plug into ``solvers.davidson.davidson`` without that solver knowing
anything about excitons. The trial-vector layout matches the matvec from
``bse_stack_matvec.build_bse_stack_matvec``:

    X.shape   = (m, nc_pad, nv_pad, nk)         (m = batch axis)
    X.sharding = P(None, "x", "y", None)        (c on x, v on y)

The leading-energy initial subspace + (E_c − E_v) diagonal preconditioner
are the BSE analogues of plane-wave selection / QE's ``g_psi`` for DFT.

Multi-process notes
-------------------
JAX 0.5+ refuses to *fetch* (``np.asarray``) or *close over* sharded arrays
that span devices belonging to other processes. The helpers below therefore:

- ``init_bse_subspace`` selects transition seeds on owned band tiles and
  gathers only their bounded energy/index candidates. Gaussian rows are
  generated at physical shape under the same distributed layout.
- ``bse_diagonal_precond`` separates its array ``data`` from its jitted
  ``apply`` callable. Planned solvers pass that data explicitly; ``__call__``
  retains the host convenience interface for existing callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Callable, Optional
import weakref

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from lxkit import device_put_process_local
from common.collectives import single_device_mesh
from solvers.lanczos import _supported_start_block

from .bse_preconditioner import exchange_spin_weight


# ═══════════════════════════════════════════════════════════════════════
#  Initial subspace: lowest (c, v, k) transitions + random tail
# ═══════════════════════════════════════════════════════════════════════

def init_bse_subspace(
    eps_c,
    eps_v,
    n_eig: int,
    *,
    n_cond: Optional[int] = None,
    n_val: Optional[int] = None,
    n_random: int = 5,
    mesh: Optional[Mesh] = None,
    sharding: Optional[NamedSharding] = None,
    seed: int = 0,
    dtype=jnp.complex128,
) -> jax.Array:
    """Build a starting subspace of ``n_eig`` BSE trial vectors.

    Returns the trial subspace ``V`` of shape ``(n_eig, nc, nv, nk)``,
    composed of two parts:

    - The first ``n_eig − n_random`` vectors are unit vectors at the
      lowest ``(c, v, k)`` transitions sorted by ``E_c − E_v`` (ties by
      global transition index). Cheap
      and diverse; tends to span the low excitons because H_BSE's
      dominant diagonal is the energy difference.
    - The last ``n_random`` are deterministic complex Gaussian vectors
      (normalised), seeded by ``seed``. They cover spectral regions
      missed by the energy-sorted picks.

    A full logical-manifold request uses every physical unit vector instead:
    its Gram matrix is exactly the identity, with no redundant random tail.

    Parameters
    ----------
    eps_c : (nk, nc) jax array (any sharding) — nc is the PADDED extent
    eps_v : (nk, nv) jax array (any sharding) — nv is the PADDED extent
    n_eig : total subspace size returned
    n_cond, n_val : LOGICAL band counts (``data['n_cond']`` /
        ``data['n_val']``).  The returned subspace has exact-zero support
        outside the logical block, so no trial vector lives in the padded,
        decoupled subspace the loader's ``PAD_EPS_GUARD_RY`` sentinel
        parks at ~1e3 Ry.  Default None means "the arrays are unpadded",
        which is only true when ``n_*_pad == n_*``; every production
        caller should pass them.
    n_random : how many of those slots are random (default 5)
    mesh, sharding : if given, the result is shard-constrained to
        ``sharding`` (canonically P(None, "x", "y", None), the stack matvec's).
    seed : JAX RNG seed. Default 0 → reproducible across process grids.

    Returns
    -------
    V : (n_eig, nc, nv, nk) — unit-norm complex jax array, sharded if
        ``sharding`` was provided.
    """
    nk, nc = eps_c.shape
    if eps_v.shape[0] != nk:
        raise ValueError("init_bse_subspace: conduction and valence k counts differ")
    nv = eps_v.shape[1]
    nc_log = nc if n_cond is None else int(n_cond)
    nv_log = nv if n_val is None else int(n_val)
    if not (0 < nc_log <= nc and 0 < nv_log <= nv):
        raise ValueError(
            f"init_bse_subspace: logical extents (n_cond={nc_log}, "
            f"n_val={nv_log}) must be in (0, padded] = (0, {nc}] x (0, {nv}]")
    physical = nc_log * nv_log * nk
    if not 0 < n_eig <= physical:
        raise ValueError(f"init_bse_subspace: n_eig={n_eig} must be in [1, {physical}] "
                         "for the logical transition space")
    n_random = 0 if n_eig == physical else max(0, min(n_random, n_eig))
    n_pick = n_eig - n_random
    if sharding is None:
        if mesh is None:
            mesh = single_device_mesh()
        sharding = NamedSharding(mesh, P(None, 'x', 'y', None))
    mesh = sharding.mesh
    sc, sv = NamedSharding(mesh, P(None, 'x')), NamedSharding(mesh, P(None, 'y'))
    ec = device_put_process_local(eps_c, sc)
    ev = device_put_process_local(eps_v, sv)
    from lxkit import memory
    from common.gpu_utils import device_budget_bytes
    from runtime.aot_memory import compiled_price
    key = jnp.int64(seed)
    signature = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype,
                      sharding=x.sharding if x.committed else None)
                      for x in (ec, ev, key))
    executable = _compiled_bse_seed(signature, sharding, nc_log, nv_log,
                                    n_eig, n_random, dtype)
    stage = 'BSE initial subspace'
    price = compiled_price(compiled=executable, ranks=int(mesh.size), face=True, source=stage)
    device_budget_bytes()
    memory.book(price, stage=stage)
    memory.admit(stage, price, memory.room(bucketed=False))
    return executable(ec, ev, key)


@lru_cache(maxsize=16)
def _compiled_bse_seed(signature, sharding, nc_log, nv_log, n_eig, n_random, dtype):
    """AOT once per abstract geometry; no energy tables or seed values retained."""
    return _bse_seed_program.lower(*signature, sharding=sharding,
        nc_log=nc_log, nv_log=nv_log, n_eig=n_eig, n_random=n_random, dtype=dtype).compile()


@partial(jax.jit, static_argnames=('sharding', 'nc_log', 'nv_log', 'n_eig', 'n_random', 'dtype'))
def _bse_seed_program(ec, ev, seed, *, sharding, nc_log, nv_log, n_eig, n_random, dtype):
    """Lowest-transition unit rows and physical Gaussians on band faces.

    Only score/index candidates, at most ``P * n_pick`` entries, are gathered.
    The whole seed and the transition energy field remain distributed.
    """
    n_pick = n_eig - n_random
    mesh = sharding.mesh
    sh = NamedSharding(mesh, P(None, 'x', 'y', None))

    @partial(jax.shard_map, mesh=mesh, in_specs=(P(None, 'x'), P(None, 'y')),
             out_specs=P(None, 'x', 'y', None), check_vma=False)
    def unit_rows(ec, ev):
        nk, lc = ec.shape
        lv = ev.shape[1]
        physical = nk*nc_log*nv_log
        c = jnp.arange(lc, dtype=jnp.int64) + jax.lax.axis_index('x')*lc
        v = jnp.arange(lv, dtype=jnp.int64) + jax.lax.axis_index('y')*lv
        k = jnp.arange(nk, dtype=jnp.int64)
        indices = ((k[:, None, None]*nc_log+c[None, :, None])*nv_log
                   + v[None, None, :])
        live = (c[None, :, None] < nc_log) & (v[None, None, :] < nv_log)
        if n_pick == physical:
            # The complete physical basis needs no ordering or random tail.
            picks = jnp.arange(physical, dtype=jnp.int64)
        elif n_pick:
            energy = ec[:, :, None]-ev[:, None, :]
            scores = jnp.where(live & jnp.isfinite(energy) & (energy > 1e-12),
                               energy, jnp.inf).reshape(-1)
            ids = jnp.where(live, indices, physical).reshape(-1)
            scores, ids = jax.lax.sort((scores, ids), num_keys=2)
            keep = min(n_pick, scores.size)
            scores = jax.lax.all_gather(scores[:keep], ('x', 'y')).reshape(-1)
            ids = jax.lax.all_gather(ids[:keep], ('x', 'y')).reshape(-1)
            _, picks = jax.lax.sort((scores, ids), num_keys=2)
            picks = picks[:n_pick]
        else:
            picks = jnp.empty((0,), dtype=jnp.int64)
        return ((indices[None] == picks[:, None, None, None]) & live[None]
                ).transpose(0, 2, 3, 1).astype(dtype)

    starts = unit_rows(ec, ev)
    if n_random:
        nk, nc = ec.shape
        nv = ev.shape[1]
        shape = (nc, nv, nk)
        random, _ = _supported_start_block(
            nc*nv*nk, n_random, seed, None, vector_shape=shape,
            support_shape=(nc_log, nv_log, nk),
            constrain=lambda x: jax.lax.with_sharding_constraint(x, sh))
        random = random.T.reshape(n_random, *shape)
        norms = jnp.sqrt(jnp.sum(jnp.abs(random)**2, axis=(1, 2, 3), keepdims=True))
        random /= jnp.maximum(norms, 1e-30)
        starts = jnp.concatenate((starts, random.astype(dtype)), axis=0)
    return jax.lax.with_sharding_constraint(starts, sharding)


# ═══════════════════════════════════════════════════════════════════════
#  The EXACT transition-space diagonal of H_BSE
# ═══════════════════════════════════════════════════════════════════════

#: The transition-space dimension at or above which ``davidson_precond auto``
#: turns the exact assembled diagonal on.
#:
#: THIS NUMBER IS A PLACEHOLDER, NOT A MEASUREMENT.  What is measured, and only
#: on the 1024-dimension record deck, is that the exact diagonal is correct, is
#: physics-neutral, costs one dispatch to apply, and saves no iterations there
#: (``DAVIDSON_COMPETITIVE.md`` §14, ``PRECOND_BUILD_FREE.md``).  The reason to
#: expect it to pay at large ``bse_dim`` is that the build is a fixed one-off
#: whose cost grows like the ISDF rank while a matvec grows with the whole
#: problem, so any iteration it saves is worth more as the deck grows.  Nobody
#: has yet run a deck large enough to see that happen.  Until someone does, the
#: value below is a guess placed an order of magnitude above the largest deck we
#: have measured, and the run that would replace it with a number is written out
#: in PRECOND_BUILD_FREE.md under "what would settle it".
EXACT_PRECOND_AUTO_MIN_DIM = 100_000


def resolve_precond_route(route: str, bse_dim: int) -> str:
    """Turn ``bare`` / ``exact`` / ``auto`` into ``bare`` or ``exact``.

    ``auto`` picks the exact diagonal at or above
    :data:`EXACT_PRECOND_AUTO_MIN_DIM` and the bare transition energy below it.
    Anything else is returned unchanged, so the two explicit routes always mean
    exactly what they say.
    """
    if route != "auto":
        return route
    return "exact" if int(bse_dim) >= EXACT_PRECOND_AUTO_MIN_DIM else "bare"


# ── the build kernel, at MODULE scope ─────────────────────────────────
# This used to be an ``@jax.jit`` defined inside ``build_bse_exact_diagonal``.
# A jit wrapper created in a function body is a NEW object every call, so its
# trace / lower / compile-cache-probe cache is new every call too: the "build"
# re-constructed its XLA program on every call.  Measured on the record deck at
# P=4, that re-construction is ~26 ms per call and the program it rebuilds runs
# in 1.26 ms — arithmetic and both cross-rank reductions included.  The
# reductions are two tupled reduce-scatters carrying 4.7 MiB, so they were never
# the cost either.  The body below was character-identical to the old one when
# the jit moved; the FEAST consolidation then restructured it to carry two static
# routes, and the REAL route's output is still bit-identical to what shipped
# before either change (np.array_equal on the record deck, measured when the
# jit moved; no test of it is in the tree).
# TWO STATIC KNOBS, AND WHY EACH EXISTS.  This kernel is now the ONE
# ``diag(H_BSE)`` in the tree: ``bse_gmres.build_preconditioner_diagonal_sharded``
# used to assemble the same object independently, and the two differed in exactly
# two ways that are behaviour, not style (PRECOND_BUILD_FREE.md §7.1).  Both are
# static, so each route traces its own program and neither can perturb the other.
#
# ``complex_out``.  Davidson divides a real residual by ``diag − lambda`` and takes
# ``real()``; the shifted GMRES solves (the former FEAST route among them) divide
# by ``z − diag`` at a COMPLEX shift and want the operator's antihermitian residue
# kept.  On the record deck that residue is
# ``max|Im| = 1.13e-14 Ry = 1.5e-13 eV`` against a 0.62 Ry signal — negligible in
# size, but it is the solve's to keep or drop, not this function's, so the flag
# carries it rather than deciding it.
#
# ``W_q0=None``.  The RPA density-response route solves with
# ``include_W=False``, where the direct term is absent rather than zero-valued.
# Passing ``None`` drops the term at TRACE time, so those routes compile a program
# with no W contraction in it at all instead of multiplying by a zero tile.
@partial(jax.jit, static_argnames=("nk", "sharding", "complex_out", "exchange_channel"))
def _exact_diagonal_kernel(eps_c, eps_v, psi_c_X, psi_v_Y, W_q0, M,
                           V_q0, *, nk, sharding, complex_out: bool = False,
                           exchange_channel: str = "singlet"):
    dE = eps_c.T[:, None, :] - eps_v.T[None, :, :]          # (c, v, k)

    if W_q0 is None:
        # include_W=False: no screened-direct term.  Not "W_d = 0" — the
        # contraction is not emitted.
        W_d_c = None
    else:
        a = jnp.sum(jnp.abs(psi_c_X) ** 2, axis=2)          # (k, c, mu)
        b = jnp.sum(jnp.abs(psi_v_Y) ** 2, axis=2)          # (k, v, nu)
        Y = jnp.einsum('kcM,MN->kcN', a.astype(W_q0.dtype), W_q0)
        W_d_c = jnp.einsum('kcN,kvN->cvk', Y, b.astype(W_q0.dtype))

    S = jnp.einsum('kcvM,MN->kcvN', M, V_q0)
    # The matvec's scalar-singlet exchange weight (bse_preconditioner owns it).
    V_x_c = jnp.einsum('kcvN,kcvN->cvk', S, jnp.conj(M)) * exchange_spin_weight(
        psi_c_X.shape[2], exchange_channel)

    if complex_out:
        num = V_x_c if W_d_c is None else (V_x_c - W_d_c)
        out = dE.astype(num.dtype) + num / nk
    else:
        # Character-identical to the pre-consolidation real form — ``real()`` is
        # applied to each contraction before the subtraction, exactly as before —
        # so the Davidson route's HLO and its output are bit-identical (measured on
        # the record deck when the jit moved; no test of it is in the tree).
        V_x = jnp.real(V_x_c)
        out = (dE + V_x / nk if W_d_c is None
               else dE + (V_x - jnp.real(W_d_c)) / nk)
    if sharding is not None:
        out = jax.lax.with_sharding_constraint(out, sharding)
    return out


class _DiagMemo:
    """Single-slot identity memo for the assembled diagonal.

    WHY IDENTITY AND NOT EQUALITY.  Comparing the operands by value would cost
    a reduction over every one of them — more than the build it is trying to
    skip.  Comparing by ``id`` alone is unsound: CPython recycles ids, so a
    freed W and a newly allocated one can collide and hand back a stale
    diagonal.  This holds a WEAK reference to each operand and requires
    ``ref() is operand`` for all of them.  A live referent cannot have had its
    id recycled, so a hit is exactly "the same arrays as last time" and a
    changed W — a different object — always misses.  Weak, so the memo pins no
    device buffer; the only strong reference it keeps is the diagonal itself,
    which is one ``(nc, nv, nk)`` real array.

    This is the same idiom, and for the same reason, as the ``g_index``
    device-buffer cache in ``common/wfn_transforms`` (``_GINDEX_DEV_BY_ID``):
    identity first because hashing the content costs more than the work being
    saved, weak references because an ``id`` on its own is a recycled-address
    hazard.
    """

    __slots__ = ("_refs", "_value", "_meta", "hits", "misses")

    def __init__(self):
        self._refs = None
        self._value = None
        self._meta = None
        self.hits = 0
        self.misses = 0

    def get(self, meta, operands):
        if (self._value is None or self._refs is None
                or self._meta != meta or len(self._refs) != len(operands)):
            self.misses += 1
            return None
        for ref, op in zip(self._refs, operands):
            if ref() is not op:          # dead ref -> None is not op -> miss
                self.misses += 1
                return None
        self.hits += 1
        return self._value

    def put(self, meta, operands, value):
        try:
            refs = tuple(weakref.ref(op) for op in operands)
        except TypeError:                # un-weakref-able backend array
            self.clear()
            return value
        self._refs = refs
        self._value = value
        self._meta = meta
        return value

    def clear(self):
        self._refs = None
        self._value = None
        self._meta = None


_DIAG_MEMO = _DiagMemo()


def clear_exact_diagonal_memo():
    """Drop the memoised diagonal.  Call between decks in one process."""
    _DIAG_MEMO.clear()


def exact_diagonal_memo_stats():
    """``{hits, misses, loaded}`` — what the memo has done in this process."""
    return {"hits": _DIAG_MEMO.hits, "misses": _DIAG_MEMO.misses,
            "loaded": _DIAG_MEMO._value is not None}


def build_bse_exact_diagonal(
    eps_c, eps_v, psi_c_X, psi_v_Y, W_q0, M, V_q0, nk: int,
    *, sharding=None, memo: bool = True, complex_out: bool = False,
    exchange_channel: str = "singlet",
):
    """``diag(H_BSE)[c, v, k]`` — assembled exactly, once per solve.

    WHAT THIS IS FOR
    ----------------
    LORRAX's Davidson preconditions with the **bare** transition energy
    ``ΔE = E_c − E_v``.  BerkeleyGW hands PRIMME the **exact assembled
    diagonal** of the BSE Hamiltonian (``Common/primme_interface.f90``), and
    quantum-chemistry TDDFT calls the same object the standard preconditioner,
    of which ``ΔE`` is the acknowledged cheap approximation.  This closes that
    gap: it is what the reference implementation uses.

    THE FORM, AND HOW ITS NORMALISATION WAS ESTABLISHED
    ---------------------------------------------------
    The diagonal element of a term is what the matvec returns for a unit
    trial vector, so each term below is its own contraction read off the
    matvec (``bse_stack_matvec``):

        V_x[c,v,k] = Σ_MN  M[k,c,v,M] · V_q0[M,N] · conj(M[k,c,v,N])
        W_d[c,v,k] = Σ_MN  a[k,c,M] · W_q0[M,N] · b[k,v,N]
            with a[k,c,M] = Σ_spinor |psi_c_X[k,c,·,M]|²
                 b[k,v,N] = Σ_spinor |psi_v_Y[k,v,·,N]|²

        diag(H) = ΔE + (w·V_x − W_d) / nk,   w = exchange_spin_weight(nspinor)

    **The ``1/nk`` and the coefficient on ``V_x`` are MEASURED, not assumed.**
    Fitting ``diag(H_dense) − ΔE = α·V_x + β·W_d`` by least squares against the
    dense materialisation of this very operator returns
    ``α = +0.015625 = +1/64``, ``β = −0.015625 = −1/64`` on a deck with
    ``nk = 64``, with a fit residual of **1.5e-15 eV** against a 0.123 eV
    signal.  Note α = +1/nk and NOT +2/nk: the spin-singlet factor of two does
    not appear on this noncolinear/spin-orbit deck (w = 1). A scalar deck has
    w = 2, as in the matvec's own encode.

    COST — AND WHAT IT IS NOT
    -------------------------
    The arithmetic is ~0.30 GFLOP, under a third of one matvec.  The wall cost
    is not the arithmetic and it is not the collectives either.  Measured on the
    record deck at P=4, the whole program — both cross-rank reduce-scatters
    included — executes in **1.26 ms**; what used to cost ~0.4 s was **building
    the XLA program**.  The jit wrapper lived INSIDE this function, so a fresh wrapper
    object — and therefore a fresh trace/lower/compile-probe — was created on
    every call.  It is now at module scope (:func:`_exact_diagonal_kernel`),
    so a process pays program construction once and every later build is a
    dispatch; and the memo below removes even that.  See
    ``PRECOND_BUILD_FREE.md`` for the measured split.

    MEMO
    ----
    ``memo=True`` (default) returns the previously assembled diagonal when
    called again with **the same operand objects**.  Identity, not equality:
    a hit needs ``is`` on every operand, checked through weak references, so a
    changed W is a different object, misses, and rebuilds.  Nothing is pinned.
    :func:`clear_exact_diagonal_memo` drops the slot;
    :func:`exact_diagonal_memo_stats` reports hits/misses.

    SHARDING
    --------
    The BUILD contracts over μ (on ``x``) and ν (on ``y``) and therefore emits
    collectives — once, here.  The APPLICATION
    (:func:`bse_diagonal_precond`) is elementwise on the ``(c, v, k)`` shards
    and emits **none**.  No ``shard_map`` is opened by either.

    Parameters
    ----------
    W_q0 : (nmu, nnu) — the ``q = 0`` slice of ``W_q``.  Take it BEFORE the
        driver's donated ifft consumes ``W_q``; reading it from ``W_R``
        instead would make the answer depend on that ifft's norm convention.
        ``None`` means ``include_W=False``: the screened-direct term is not
        emitted at all (the RPA density-response route).
    complex_out : keep the assembled diagonal COMPLEX instead of taking
        ``real()``.  The shifted GMRES solves want it (they divide by
        ``z − diag`` at a complex shift); Davidson does not.  See
        the note over :func:`_exact_diagonal_kernel`.

    Returns
    -------
    diag : (nc_pad, nv_pad, nk) — real by default, complex under
        ``complex_out``; same layout as ``ΔE`` either way.
    """
    operands = (eps_c, eps_v, psi_c_X, psi_v_Y, W_q0, M, V_q0)
    # An operand that is ``None`` (``W_q0`` under include_W=False) has no weak
    # reference, so the identity memo cannot express "the same arrays as last
    # time" for it.  Skip the memo outright rather than lean on _DiagMemo.put's
    # un-weakref-able bail-out, which would do the right thing silently.
    if memo and any(op is None for op in operands):
        memo = False
    # ``complex_out`` is part of the memo key, not just the program key: the two
    # routes return DIFFERENT objects for the same operands, and a memo that
    # ignored the flag would hand a shifted solve Davidson's real diagonal.
    meta = (int(nk), sharding, bool(complex_out), exchange_channel)
    if memo:
        hit = _DIAG_MEMO.get(meta, operands)
        if hit is not None:
            return hit
    out = _exact_diagonal_kernel(*operands, nk=int(nk), sharding=sharding,
                                 complex_out=bool(complex_out),
                                 exchange_channel=exchange_channel)
    if memo:
        _DIAG_MEMO.put(meta, operands, out)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Diagonal preconditioner: 1 / (ΔE − λ + ε)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BSEDiagonalPreconditioner:
    """A diagonal preconditioner with explicit distributed array operands.

    ``apply(data, R, Lambda, X=None)`` is suitable inside a whole-solver JIT.
    ``preconditioner(R, Lambda, X=None)`` is the host convenience form.
    """

    data: tuple
    apply: Callable

    def __call__(self, R, Lambda, X=None):
        return self.apply(self.data, R, Lambda, X)


def bse_diagonal_precond(
    eps_c,
    eps_v,
    *,
    epsilon_shift: float = 1e-3,
    sharding: Optional[NamedSharding] = None,
    diag_H=None,
    olsen: bool = False,
):
    """Build a diagonal preconditioner ``precond_fn(R, Lambda) → P``.

    The BSE Hamiltonian's leading diagonal element is
    ``H_diag[c, v, k] ≈ (E_c[k] − E_v[k])`` (the V & W diagonals contribute
    a smaller correction). Davidson convergence improves dramatically
    when the residual is filtered by the inverse of ``(H_diag − λ)``.

    Multi-process safety
    --------------------
    The returned object's ``data`` holds eps_c, eps_v, and optional diag_H.
    Its ``apply(data, R, Lambda, X=None)`` callable closes only over static
    settings. Pass both parts explicitly to a planned solver; calling the
    convenience ``__call__`` inside a new JIT would capture its data.

    Parameters
    ----------
    eps_c : (nk, nc) jax array
    eps_v : (nk, nv) jax array
    epsilon_shift : Ry, regularises near band edges. Default 1e-3 ≈ 13.6 meV.
    sharding : optional NamedSharding for the (nc, nv, nk) ΔE tensor;
        canonically the X sharding with batch axis dropped (e.g.
        ``P("x", "y", None)``).

    Returns
    -------
    precond_fn : BSEDiagonalPreconditioner
        Callable as (R, Lambda, X=None) → P. R, P shape (m, nc, nv, nk);
        Lambda shape (m,). Its data/apply split supports whole-solver JIT.
    """
    @jax.jit
    def _impl(data, R, Lambda, X=None):
        eps_c_in, eps_v_in, diag_in = data
        if diag_in is None:
            # ΔE[c, v, k] = E_c[k] − E_v[k]; same convention as the stack matvec's
            # D term.  The BARE route: correct, cheap, and an approximation.
            D = eps_c_in.T[:, None, :] - eps_v_in.T[None, :, :]
        else:
            # The EXACT assembled diagonal, built once per solve.  Same shape,
            # same sharding, same elementwise application — the route change
            # costs nothing per iteration.
            D = diag_in
        if sharding is not None:
            D = jax.lax.with_sharding_constraint(D, sharding)
        # Broadcast: R is (m, c, v, k), Lambda is (m,), D is (c, v, k).
        denom = (D[None, :, :, :]
                 - Lambda[:, None, None, None]
                 + jnp.asarray(epsilon_shift, dtype=D.dtype))
        denom_safe = jnp.where(jnp.abs(denom) < 1e-12,
                               jnp.asarray(1e-12, dtype=denom.dtype),
                               denom)
        P_out = R / denom_safe
        if olsen and X is not None:
            # OLSEN correction.  Plain Jacobi returns a direction with a
            # component along the current Ritz vector, which the subspace
            # already contains; Olsen projects it out:
            #     P = (D−λ)^-1 R  −  α (D−λ)^-1 X ,
            #     α = <X, (D−λ)^-1 R> / <X, (D−λ)^-1 X>
            # Two BATCHED inner products, contracted as 'm...,m...->m', i.e.
            # two (m,) all-reduces rather than 2m scalar ones — the lesson
            # REORTH_EXPERIMENT §1.4 paid for.  This is what makes a SMALL
            # epsilon_shift safe: without it, shrinking the shift makes the
            # denominator near the band edge small and the Jacobi direction
            # collapses onto X.
            DX = X / denom_safe
            num = jnp.einsum('m...,m...->m', jnp.conj(X), P_out)
            den = jnp.einsum('m...,m...->m', jnp.conj(X), DX)
            alpha = num / jnp.where(jnp.abs(den) < 1e-30,
                                    jnp.asarray(1e-30, dtype=den.dtype), den)
            P_out = P_out - alpha[:, None, None, None] * DX
        norms = jnp.sqrt(jnp.sum(jnp.abs(P_out) ** 2, axis=(1, 2, 3),
                                 keepdims=True))
        return P_out / jnp.maximum(norms, 1e-30)

    return BSEDiagonalPreconditioner((eps_c, eps_v, diag_H), _impl)


# ═══════════════════════════════════════════════════════════════════════
#  Occupation weights: the diagonal and the start block of a weighted window
# ═══════════════════════════════════════════════════════════════════════

def occupation_diagonal(diag_H, eps_c, eps_v, weights, *, sharding=None):
    """The diagonal of ``A = dE + w K w`` from the unweighted one: ``dE + w^2
    (diag_H - dE)`` on the pairs with ``w > 0`` and
    ``bse_preconditioner.MASKED_PAIR_RY`` on the masked ones.  ``diag_H`` None
    is the bare ``dE``.  ``(n_c, n_v, nk)``, real, on ``sharding``."""
    from .bse_preconditioner import MASKED_PAIR_RY, energy_diff_cv_k

    @jax.jit
    def _diag(diag, ec, ev, w):
        dE = energy_diff_cv_k(ec, ev)
        out = dE if diag is None else dE + w * w * (diag - dE)
        out = jnp.where(w > 0, out, MASKED_PAIR_RY)
        return out if sharding is None else jax.lax.with_sharding_constraint(out, sharding)
    return _diag(diag_H, eps_c, eps_v, weights)


def occupation_start_block(eps_c, eps_v, weights, *, n_rows, n_cond, n_val, mesh, sharding):
    """A start block for a window with occupation weights: ``n_rows`` Gaussian
    vectors on the physical transitions (:func:`init_bse_subspace` with no
    unit rows: its lowest-``dE`` unit vectors would sit on masked pairs, whose
    ``dE`` may be zero or negative), zeroed on the masked pairs.  Every later
    vector of a Davidson is a combination of images and preconditioned
    residuals, which the operator keeps at zero there."""
    X0 = init_bse_subspace(eps_c, eps_v, n_eig=n_rows, n_cond=n_cond, n_val=n_val,
                           n_random=n_rows, mesh=mesh, sharding=sharding)
    return jax.jit(lambda x, w: jnp.where(w[None] > 0, x, 0), out_shardings=sharding)(X0, weights)


# ═══════════════════════════════════════════════════════════════════════
#  Full BSE: the paired diagonal preconditioner and its q = 0 slice of W
# ═══════════════════════════════════════════════════════════════════════

def w_q0_for_precond(data):
    """The ``q = 0`` slice ``W_q[:, :, 0, 0, 0]`` of a payload's screened tile,
    stashed on the payload as ``data['_W_q0_for_precond']``.

    The exact diagonal's screened-direct term reads it
    (:func:`build_bse_exact_diagonal`).  It must be taken before the solve's
    donated transform consumes ``W_q``, and one stashed object makes the
    diagonal's identity memo hit on a second solve of the same payload (the
    two reasons ``bse_solve.solve_bse_sharded`` records at its own stash,
    which uses the same key).  ``(n_mu, n_nu)`` at ``P('x', 'y')``, Ry.
    """
    if data.get("_W_q0_for_precond") is None:
        if data.get("W_q") is None:
            raise ValueError(
                "the exact diagonal needs the q=0 slice of W_q, but this payload's "
                "W_q has already been consumed by the donated ifft and no slice was "
                "stashed.  Reload the payload.")
        data["_W_q0_for_precond"] = data["W_q"][:, :, 0, 0, 0]
    return data["_W_q0_for_precond"]


def payload_w_r(data, mesh_xy, sh, *, include_W=True):
    """The payload's screened tile in R space, transformed once and left on it.

    THE transform of an eigensolver's or a Haydock recursion's payload:
    ``data["W_R"] = ifftn_ortho(W_q)`` over the k axes, so every application of
    the screened-direct term is a pointwise product in R with no transform of W
    in it.  ``W_q`` is DONATED to the memoised transform
    (``fft_helpers.get_donated_kfft_kminor``) and the caller-side reference
    dropped: one (mu, nu, k) tile per rank for the whole solve, where a
    transform inside the solve's own jit kept both alive.  A payload that
    already carries ``W_R`` (a second solve, the Haydock run after the
    eigensolve, ``exciton_bands``' per-Q payloads) is used as it is.  Without
    ``include_W`` the W term is never built and the tile is a placeholder
    operand.  Take :func:`w_q0_for_precond` first when the exact diagonal is
    wanted.  Callers that must keep ``W_q`` (the resolvent and ladder code)
    use ``bse_gmres.ensure_W_R`` instead, which borrows it."""
    if not include_W:
        return data["W_q"] if data.get("W_q") is not None else data.get("W_R")
    if data.get("W_R") is None:
        from common.fft_helpers import get_donated_kfft_kminor
        ifft = get_donated_kfft_kminor(
            mesh_xy, tuple(int(n) for n in data["W_q"].shape[-3:]), sh.W.spec,
            kind="ifftn", norm="ortho")
        data["W_R"] = ifft(data["W_q"])
        data["W_q"] = None      # release the caller-side reference
    return data["W_R"]


def bse_pair_diagonal_precond(
    eps_c,
    eps_v,
    *,
    epsilon_shift: float = 1e-3,
    sharding: Optional[NamedSharding] = None,
    diag_H=None,
    low=None,
    olsen: bool = False,
):
    """The full-BSE counterpart of :func:`bse_diagonal_precond`: the paired
    correction ``[t_X; t_Y] = (H_D - omega)^-1 [R_X; R_Y]``.

    ``olsen`` adds the Olsen correction in the metric ``Sigma``
    (``solvers.bse_sp_davidson.sp_diagonal_precond``); ``apply`` then takes
    the Ritz pair too, ``apply(data, RX, RY, omega, X, Y)``.

    ``low = (eps_c_low, eps_v_low, diag_low)`` is the finite-momentum form:
    the lower-right block of ``H`` is ``-A'`` with its own diagonal ``D_low``
    (the antiresonant pair's energies, ``bse_nontda.operand_sets``), so
    ``H_D = diag(D, -D_low)`` and ``t_Y = -R_Y / (D_low + omega + eps)``.
    ``diag_low`` is that set's exact diagonal or None for its bare ``dE``.

    ``H_D = diag(D, -D)`` is the diagonal of the full operator
    ``H = [[A, B], [-B*, -A*]]``: ``A`` is Hermitian, so its diagonal ``D`` is
    real and the lower block's is ``-D``; the coupling block ``B`` sits off
    the diagonal of ``H`` and contributes nothing.  ``D`` is therefore the
    Tamm-Dancoff operator's diagonal, from the same two sources: ``diag_H``
    from :func:`build_bse_exact_diagonal` (the exact one,
    ``dE + (w V_x - W_d) / nk``), or the bare ``dE = E_c - E_v`` when
    ``diag_H`` is None.  The division is
    ``solvers.bse_sp_davidson.sp_diagonal_precond``:

        t_X = R_X / (D - omega + eps),     t_Y = -R_Y / (D + omega + eps).

    Unlike the Tamm-Dancoff preconditioner it does not normalise its output
    per row: ``t_X`` and ``t_Y`` are one direction and share one scale, which
    the solver applies.  Elementwise on the ``(c, v, k)`` shards: no
    collective.

    Parameters
    ----------
    eps_c : (nk, nc) Ry — conduction energies, the operator's (head-shifted).
    eps_v : (nk, nv) Ry
    epsilon_shift : Ry, added to both denominators (default 1e-3).
    sharding : NamedSharding of the ``(nc, nv, nk)`` diagonal, canonically
        ``P('x', 'y', None)``.
    diag_H : (nc, nv, nk) real, Ry, or None.

    Returns
    -------
    BSEDiagonalPreconditioner
        ``data = (eps_c, eps_v, diag_H)`` and
        ``apply(data, RX, RY, omega) -> (tX, tY)`` on ``(m, nc, nv, nk)``
        blocks with ``omega`` ``(m,)`` complex; pass both parts to the solver.
        (The bundle's host call is ``precond(RX, RY, omega)`` here.)
    """
    from solvers.bse_sp_davidson import sp_diagonal_precond
    from .bse_preconditioner import energy_diff_cv_k
    divide = sp_diagonal_precond(epsilon_shift=epsilon_shift, olsen=olsen)

    def _diagonal(eps_c_in, eps_v_in, diag_in):
        D = energy_diff_cv_k(eps_c_in, eps_v_in) if diag_in is None else diag_in
        return D if sharding is None else jax.lax.with_sharding_constraint(D, sharding)

    def _apply(data, RX, RY, omega, X=None, Y=None):
        D = _diagonal(*data[:3])
        if len(data) == 3:
            return divide(D, RX, RY, omega, X, Y)
        return divide((D, _diagonal(*data[3:])), RX, RY, omega, X, Y)

    return BSEDiagonalPreconditioner(
        (eps_c, eps_v, diag_H) + (() if low is None else tuple(low)), _apply)


__all__ = ["init_bse_subspace", "bse_diagonal_precond", "BSEDiagonalPreconditioner",
           "build_bse_exact_diagonal", "clear_exact_diagonal_memo",
           "exact_diagonal_memo_stats", "resolve_precond_route",
           "EXACT_PRECOND_AUTO_MIN_DIM", "bse_pair_diagonal_precond", "w_q0_for_precond",
           "occupation_diagonal", "occupation_start_block"]
