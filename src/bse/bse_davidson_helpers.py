"""bse/bse_davidson_helpers.py — initial-subspace + diagonal preconditioner
helpers tailored to the (block, nc_pad, nv_pad, nk) BSE vector layout.

These plug into ``solvers.davidson.davidson`` without that solver knowing
anything about excitons. The trial-vector layout matches the matvec from
``bse_simple.build_bse_simple_matvec``:

    X.shape   = (m, nc_pad, nv_pad, nk)         (m = batch axis)
    X.sharding = P(None, "x", "y", None)        (c on x, v on y)

The leading-energy initial subspace + (E_c − E_v) diagonal preconditioner
are the BSE analogues of plane-wave selection / QE's ``g_psi`` for DFT.

Multi-process notes
-------------------
JAX 0.5+ refuses to *fetch* (``np.asarray``) or *close over* sharded arrays
that span devices belonging to other processes. The helpers below therefore:

- ``init_bse_subspace`` calls ``multihost_utils.process_allgather`` on the
  eps tensors before reading them on host. Each process replicates the
  full eps; this is cheap (eps is tiny, ~kilobytes).
- ``bse_diagonal_precond`` closes only over a python-side ``delta_E_host``
  (numpy) and rebuilds the device tensor inside the jit'd function from
  ``eps_c`` and ``eps_v`` passed at *call time* — the closure does not
  capture any global jax.Array.
"""
from __future__ import annotations

from functools import partial
from typing import Optional
import weakref

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import gather_to_host


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
      lowest ``(c, v, k)`` transitions sorted by ``E_c − E_v``. Cheap
      and diverse; tends to span the low excitons because H_BSE's
      dominant diagonal is the energy difference.
    - The last ``n_random`` are deterministic complex Gaussian vectors
      (normalised), seeded by ``seed``. They cover spectral regions
      missed by the energy-sorted picks.

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
        ``sharding`` (canonically P(None, "x", "y", None) for ``bse_simple``).
    seed : RNG seed (numpy keyed). Default 0 → reproducible.

    Returns
    -------
    V : (n_eig, nc, nv, nk) — unit-norm complex jax array, sharded if
        ``sharding`` was provided.
    """
    eps_c_np = gather_to_host(eps_c)
    eps_v_np = gather_to_host(eps_v)
    nk, nc = eps_c_np.shape
    _, nv = eps_v_np.shape

    n_random = max(0, min(n_random, n_eig))
    n_pick = n_eig - n_random

    # Physical extents.  DROP THE PAD BY COUNT, never by value.
    #
    # This used to be `finite = isfinite(flat) & (flat > 1e-12)` with the
    # comment "skip padded / non-physical bands (zero or negative ΔE)".
    # That is a value filter and it does not do what it says.  With the
    # loader's OLD zero ε pad the pad transitions were
    #     (c_pad, v_pad) -> 0        caught by the filter
    #     (c_pad, v_real) -> |ε_v|   POSITIVE, sails through
    #     (c_real, v_pad) -> ε_c     POSITIVE, sails through
    # and since ΔE_physical = ε_c + |ε_v| exceeds both mixed terms, those
    # two families sort BELOW every real transition — so the "lowest
    # transitions" seed picked pad states preferentially, exactly the
    # states the filter was written to exclude.  The loader now writes a
    # signed sentinel (``bse_io.PAD_EPS_GUARD_RY``) which makes the value
    # filter incidentally work, but a filter that is only correct because
    # of a constant defined in another module is not a filter.  The count
    # is exact and local: pass ``n_cond``/``n_val`` and slice.
    nc_log = nc if n_cond is None else int(n_cond)
    nv_log = nv if n_val is None else int(n_val)
    if not (0 < nc_log <= nc and 0 < nv_log <= nv):
        raise ValueError(
            f"init_bse_subspace: logical extents (n_cond={nc_log}, "
            f"n_val={nv_log}) must be in (0, padded] = (0, {nc}] x (0, {nv}]")

    # ── Part 1: lowest PHYSICAL (c, v, k) by energy ───────────────────
    delta_E = (eps_c_np[:, :nc_log, None]
               - eps_v_np[:, None, :nv_log])              # (nk, nc_log, nv_log)
    flat = delta_E.reshape(-1)
    # Residual value guard: a genuinely non-physical ΔE (NaN, or <= 0 from a
    # metallic/misordered window) is still skipped.  It is no longer load
    # bearing for the pad.
    finite = np.isfinite(flat) & (flat > 1e-12)
    order = np.argsort(np.where(finite, flat, np.inf))
    picks = order[:n_pick]
    k_idx, c_idx, v_idx = np.unravel_index(picks, (nk, nc_log, nv_log))

    V_np = np.zeros((n_eig, nc, nv, nk), dtype=np.complex128)
    V_np[np.arange(n_pick), c_idx, v_idx, k_idx] = 1.0

    # ── Part 2: random tail, PHYSICAL block only ──────────────────────
    # Drawn at the logical shape and embedded, so the pad zone stays exact
    # zero: a random start with support on the pad block hands the solver a
    # vector inside a decoupled subspace whose eigenvalues are the sentinel.
    if n_random > 0:
        rng = np.random.default_rng(seed)
        rand = (rng.standard_normal((n_random, nc_log, nv_log, nk))
                + 1j * rng.standard_normal((n_random, nc_log, nv_log, nk)))
        norms = np.sqrt(
            np.sum(np.abs(rand) ** 2, axis=(1, 2, 3), keepdims=True))
        rand = rand / np.maximum(norms, 1e-30)
        V_np[n_pick:, :nc_log, :nv_log, :] = rand

    if sharding is not None:
        V = jax.make_array_from_callback(
            V_np.shape,
            sharding,
            lambda idx: jnp.asarray(V_np[idx], dtype=dtype),
        )
    else:
        V = jnp.asarray(V_np, dtype=dtype)
    return V


# ═══════════════════════════════════════════════════════════════════════
#  The EXACT transition-space diagonal of H_BSE
# ═══════════════════════════════════════════════════════════════════════

#: The transition-space dimension at or above which ``--davidson-precond auto``
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
# before either change (np.array_equal on the record deck, gated in
# tests/test_bse_exact_diagonal.py).
# TWO STATIC KNOBS, AND WHY EACH EXISTS.  This kernel is now the ONE
# ``diag(H_BSE)`` in the tree: ``bse_feast.build_preconditioner_diagonal_sharded``
# used to assemble the same object independently, and the two differed in exactly
# two ways that are behaviour, not style (PRECOND_BUILD_FREE.md §7.1).  Both are
# static, so each route traces its own program and neither can perturb the other.
#
# ``complex_out``.  Davidson divides a real residual by ``diag − lambda`` and takes
# ``real()``; FEAST divides by ``z − diag`` at a COMPLEX quadrature node and wants
# the operator's antihermitian residue kept.  On the record deck that residue is
# ``max|Im| = 1.13e-14 Ry = 1.5e-13 eV`` against a 0.62 Ry signal — negligible in
# size, but it is FEAST's to keep or drop, not this function's, so the flag
# carries it rather than deciding it.
#
# ``W_q0=None``.  The RPA density-response and pseudopole routes solve with
# ``include_W=False``, where the direct term is absent rather than zero-valued.
# Passing ``None`` drops the term at TRACE time, so those routes compile a program
# with no W contraction in it at all instead of multiplying by a zero tile.
@partial(jax.jit, static_argnames=("nk", "sharding", "complex_out"))
def _exact_diagonal_kernel(eps_c, eps_v, psi_c_X, psi_v_Y, W_q0, M_X, M_Y,
                           V_q0, *, nk, sharding, complex_out: bool = False):
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

    S = jnp.einsum('kcvM,MN->kcvN', M_X, V_q0)
    V_x_c = jnp.einsum('kcvN,kcvN->cvk', S, jnp.conj(M_Y))

    if complex_out:
        num = V_x_c if W_d_c is None else (V_x_c - W_d_c)
        out = dE.astype(num.dtype) + num / nk
    else:
        # Character-identical to the pre-consolidation real form — ``real()`` is
        # applied to each contraction before the subtraction, exactly as before —
        # so the Davidson route's HLO and its output are bit-identical (gated in
        # tests/test_bse_exact_diagonal.py).
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
    eps_c, eps_v, psi_c_X, psi_v_Y, W_q0, M_X, M_Y, V_q0, nk: int,
    *, sharding=None, memo: bool = True, complex_out: bool = False,
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

        V_x[c,v,k] = Σ_MN  M_X[k,c,v,M] · V_q0[M,N] · conj(M_Y[k,c,v,N])
        W_d[c,v,k] = Σ_MN  a[k,c,M] · W_q0[M,N] · b[k,v,N]
            with a[k,c,M] = Σ_spinor |psi_c_X[k,c,·,M]|²
                 b[k,v,N] = Σ_spinor |psi_v_Y[k,v,·,N]|²

        diag(H) = ΔE + (V_x − W_d) / nk

    **The ``1/nk`` and the coefficient on ``V_x`` are MEASURED, not assumed.**
    Fitting ``diag(H_dense) − ΔE = α·V_x + β·W_d`` by least squares against the
    dense materialisation of this very operator returns
    ``α = +0.015625 = +1/64``, ``β = −0.015625 = −1/64`` on a deck with
    ``nk = 64``, with a fit residual of **1.5e-15 eV** against a 0.123 eV
    signal.  Note α = +1/nk and NOT +2/nk: the spin-singlet factor of two does
    not appear on this noncolinear/spin-orbit deck, and assuming it would have
    put the exchange term in at twice its weight.

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
        emitted at all (RPA density-response and pseudopole routes).
    complex_out : keep the assembled diagonal COMPLEX instead of taking
        ``real()``.  FEAST's shifted solves want it (they divide by
        ``z − diag`` at a complex quadrature node); Davidson does not.  See
        the note over :func:`_exact_diagonal_kernel`.

    Returns
    -------
    diag : (nc_pad, nv_pad, nk) — real by default, complex under
        ``complex_out``; same layout as ``ΔE`` either way.
    """
    operands = (eps_c, eps_v, psi_c_X, psi_v_Y, W_q0, M_X, M_Y, V_q0)
    # An operand that is ``None`` (``W_q0`` under include_W=False) has no weak
    # reference, so the identity memo cannot express "the same arrays as last
    # time" for it.  Skip the memo outright rather than lean on _DiagMemo.put's
    # un-weakref-able bail-out, which would do the right thing silently.
    if memo and any(op is None for op in operands):
        memo = False
    # ``complex_out`` is part of the memo key, not just the program key: the two
    # routes return DIFFERENT objects for the same operands, and a memo that
    # ignored the flag would hand FEAST Davidson's real diagonal.
    meta = (int(nk), sharding, bool(complex_out))
    if memo:
        hit = _DIAG_MEMO.get(meta, operands)
        if hit is not None:
            return hit
    out = _exact_diagonal_kernel(*operands, nk=int(nk), sharding=sharding,
                                 complex_out=bool(complex_out))
    if memo:
        _DIAG_MEMO.put(meta, operands, out)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Diagonal preconditioner: 1 / (ΔE − λ + ε)
# ═══════════════════════════════════════════════════════════════════════

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
    The jit'd ``_impl`` does NOT close over ``eps_c`` / ``eps_v``; instead
    the outer plain-Python ``precond_fn`` captures them and forwards as
    arguments at call time. This avoids the "Closing over jax.Array that
    spans non-addressable devices" runtime error multi-process JAX raises
    when sharded arrays are baked into a jit closure.

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
    precond_fn : (R, Lambda) → P
        R, P shape (m, nc, nv, nk); Lambda shape (m,).
    """
    @jax.jit
    def _impl(R, Lambda, eps_c_in, eps_v_in, diag_in, X):
        if diag_in is None:
            # ΔE[c, v, k] = E_c[k] − E_v[k]; same convention as bse_simple's
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

    def precond_fn(R, Lambda, X=None):
        return _impl(R, Lambda, eps_c, eps_v, diag_H, X)

    return precond_fn


__all__ = ["init_bse_subspace", "bse_diagonal_precond",
           "build_bse_exact_diagonal", "clear_exact_diagonal_memo",
           "exact_diagonal_memo_stats", "resolve_precond_route",
           "EXACT_PRECOND_AUTO_MIN_DIM"]
