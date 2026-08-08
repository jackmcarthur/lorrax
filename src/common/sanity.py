"""Stage-boundary invariant gates — make silent corruption LOUD.

Motivation
----------
The dominant production failure signature of the 2026-07 CPU campaign was
not a crash: it was **rc=0 but garbage**.  A QP gap of −136 eV with every
stage reporting "successful"; a back-solve that produced NaN and was
caught only because someone counted the floats in ``eqp0.dat``; a restart
that silently misindexed bands because it reused tensors built under a
different band window.  In every case the pipeline had *hours* of
downstream compute in which a one-line check would have said "stop".

The merged ``gw.gw_output._warn_on_unphysical_h0`` guard is the template
this module generalizes: at a stage boundary, evaluate a *cheap*
invariant on the tensor that just came out, fetch **one** small host
value, and print a loud, actionable message when it fails.  The
invariants are deliberately generous — they must never fire on a healthy
run, so that when one does fire it is believed.

Cost model
----------
Every check here is a full-array reduction to a handful of scalars
followed by a single ``jax.device_get``.  At production scale
(``V_q`` ≈ 144 × 2412² complex128 ≈ 13 GB sharded) a reduction is one
memory-bandwidth sweep — milliseconds — against stage costs of minutes to
hours.  Checks are executed **once per stage**, never inside a loop or a
traced kernel.

Switch
------
``LORRAX_SANITY`` selects the level:

===============  ====================================================
value            behaviour
===============  ====================================================
``0``/``off``    all checks skipped (zero cost; the escape hatch)
``1``/unset      **default** — check and warn loudly, keep running
``strict``       check and *raise* :class:`SanityError` on failure
===============  ====================================================

The default is warn-not-raise on purpose: a false positive must never be
able to kill a 40-node job, and a real positive is loud enough to see in
any log tail.  Set ``LORRAX_SANITY=strict`` in CI / regression gates,
where an early hard failure is exactly what you want.

Usage
-----
::

    from common import sanity

    sanity.check_finite("V_q", V_qmunu, print_fn=print0)
    sanity.check_hermitian("V_q[0]", V_qmunu[0], print_fn=print0)
    sanity.check_positive("V_q[0] trace", trace, print_fn=print0)

All helpers return ``True`` when the invariant holds (or when checking is
disabled) so callers can chain or ignore the result freely.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np


__all__ = [
    "SanityError",
    "sanity_enabled",
    "sanity_strict",
    "warn",
    "check_finite",
    "check_hermitian",
    "report_hermitian_residual",
    "neg_q_index",
    "check_q_conjugate_reciprocity",
    "report_q_conjugate_residual",
    "check_positive",
    "check_in_range",
    "check_sign",
    "check_shape",
    "check_count",
]


# ---------------------------------------------------------------------------
# Level switch
# ---------------------------------------------------------------------------

_OFF_VALUES = {"0", "off", "false", "no", "none"}
_STRICT_VALUES = {"strict", "2", "raise", "fatal"}


class SanityError(RuntimeError):
    """A stage-boundary invariant failed under ``LORRAX_SANITY=strict``."""


def _level() -> str:
    raw = os.environ.get("LORRAX_SANITY")
    if raw is None:
        return "warn"
    val = raw.strip().lower()
    if val in _OFF_VALUES:
        return "off"
    if val in _STRICT_VALUES:
        return "strict"
    return "warn"


def sanity_enabled() -> bool:
    """True unless ``LORRAX_SANITY`` is set to an off value."""
    return _level() != "off"


def sanity_strict() -> bool:
    """True when failures should raise :class:`SanityError`."""
    return _level() == "strict"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def warn(message: str, *, print_fn: Callable[..., Any] = print) -> None:
    """Emit one loud, greppable failure line (and raise if strict).

    The ``*** LORRAX SANITY`` prefix is the grep token: a log tail that
    contains it means the run's outputs are suspect regardless of the
    exit code.
    """
    text = f"  *** LORRAX SANITY FAILURE: {message} ***"
    try:
        print_fn(text)
    except Exception:      # a broken print_fn must not mask the failure
        print(text, flush=True)
    if sanity_strict():
        raise SanityError(message)


# ---------------------------------------------------------------------------
# Host-side reduction helpers
# ---------------------------------------------------------------------------

def _is_jax(a: Any) -> bool:
    """True for on-device arrays (avoids importing jax for numpy inputs)."""
    return hasattr(a, "addressable_shards") or type(a).__module__.startswith(
        ("jaxlib", "jax."))


_FINITE_STATS_CACHE: dict = {}


def _finite_stats_fn(shape, dtype, sharding):
    """The jitted ``(n_bad, n_nan, max_abs)`` reduction — ONE module.

    Cached at module level for exactly the reason :func:`_herm_stats`
    (twelve lines below) is: ``jax.jit`` keys on function identity, so a
    fresh ``jit`` per gate call recompiles every time.

    MEASURED (job 7884866, MoS2 4x4 b300 htransform, cold, persistent
    cache off): run op-by-op EAGERLY this reduction is not one module but
    THIRTEEN — ``real``, ``imag``, ``isfinite`` x2, ``isnan`` x2,
    ``bitwise_and``, ``bitwise_or``, ``abs``, ``_where``, ``bitwise_not``,
    ``_reduce_sum`` x2, ``convert_element_type`` x2, ``broadcast_in_dim``,
    ``concatenate``, ``_reduce_max`` — because every eager jnp op is its
    own single-primitive XLA module.  htransform's three ``check_finite``
    gates (S, ctilde, E_nk) therefore compiled 35 of the driver's 137
    modules, 1.04 s of its 4.07 s of XLA compile.  Fused: 3 modules, one
    per distinct (shape, dtype, sharding).

    Values are bit-unchanged: the ops and their order are identical, and
    they are all either exact (isfinite/isnan/and/or/not, integer sums)
    or order-independent reductions (max).
    """
    import jax
    import jax.numpy as jnp

    key = (tuple(shape), str(dtype), sharding)
    fn = _FINITE_STATS_CACHE.get(key)
    if fn is not None:
        return fn

    @jax.jit
    def fn(arr):
        if jnp.iscomplexobj(arr):
            parts = (jnp.real(arr), jnp.imag(arr))
        else:
            parts = (arr,)
        finite = jnp.isfinite(parts[0])
        isnan = jnp.isnan(parts[0])
        for p in parts[1:]:
            finite = finite & jnp.isfinite(p)
            isnan = isnan | jnp.isnan(p)
        # ``jnp.abs`` of a non-finite entry is inf/nan; ``nanmax`` over a
        # masked copy gives the largest *finite* magnitude, which is the
        # number a human wants to see next to the failure count.
        mag = jnp.where(finite, jnp.abs(arr), 0.0)
        return jnp.stack([
            jnp.sum(~finite).astype(jnp.float64),
            jnp.sum(isnan).astype(jnp.float64),
            jnp.max(mag).astype(jnp.float64),
        ])

    _FINITE_STATS_CACHE[key] = fn
    return fn


def _finite_stats(a: Any) -> tuple[int, int, float, int]:
    """Return ``(n_nonfinite, n_nan, max_abs, size)`` in ONE host fetch.

    The reduction runs where the array lives (device for ``jax.Array``,
    host for ``numpy``); exactly four scalars cross the PCIe/host seam.
    Complex inputs are decomposed so ``inf+0j`` and ``nan+1j`` are both
    caught (``jnp.isfinite`` on complex is elementwise on both parts, but
    spelling it out keeps the semantics obvious and dtype-independent).
    """
    if _is_jax(a):
        import jax

        arr = a
        stats = _finite_stats_fn(
            arr.shape, arr.dtype, getattr(arr, "sharding", None))(arr)
        n_bad, n_nan, max_abs = (float(v) for v in np.asarray(jax.device_get(stats)))
        return int(n_bad), int(n_nan), float(max_abs), int(arr.size)

    arr = np.asarray(a)
    if arr.size == 0:
        return 0, 0, 0.0, 0
    finite = np.isfinite(arr.real) & (
        np.isfinite(arr.imag) if np.iscomplexobj(arr) else True)
    isnan = np.isnan(arr.real) | (
        np.isnan(arr.imag) if np.iscomplexobj(arr) else False)
    n_bad = int(np.count_nonzero(~finite))
    max_abs = float(np.abs(arr[finite]).max()) if n_bad < arr.size else 0.0
    return n_bad, int(np.count_nonzero(isnan)), max_abs, int(arr.size)


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def check_finite(
    name: str,
    array: Any,
    *,
    print_fn: Callable[..., Any] = print,
    expect_max_abs: float | None = None,
    verbose: bool = False,
) -> bool:
    """NaN/Inf sentinel on a stage-boundary tensor.

    This is the check that would have caught the SUMMA back-solve bug at
    its source instead of via a float count in ``eqp0.dat``.  It is the
    cheapest possible statement of "the thing this stage just produced is
    a number".

    ``expect_max_abs`` optionally adds an order-of-magnitude ceiling: a
    finite but absurd magnitude (a catastrophic-cancellation blow-up, a
    unit slip) is just as fatal as a NaN and just as cheap to detect,
    since ``max|·|`` comes out of the same reduction.
    """
    if not sanity_enabled():
        return True
    n_bad, n_nan, max_abs, size = _finite_stats(array)
    if size == 0:
        return True
    if verbose:
        print_fn(f"  sanity[{name}]: {size} elems, max|·| = {max_abs:.6e}")
    ok = True
    if n_bad:
        warn(
            f"{name} contains {n_bad} non-finite entries of {size} "
            f"({n_nan} NaN, {n_bad - n_nan} Inf).  Everything downstream of "
            f"this stage is garbage; the exit code will NOT tell you so.",
            print_fn=print_fn,
        )
        ok = False
    if expect_max_abs is not None and max_abs > float(expect_max_abs):
        warn(
            f"{name} max|·| = {max_abs:.6e} exceeds the sanity ceiling "
            f"{float(expect_max_abs):.6e} — finite but unphysical "
            f"(check units, cancellation, and basis provenance).",
            print_fn=print_fn,
        )
        ok = False
    return ok


_HERM_STATS_CACHE: dict = {}


def _herm_stats(a):
    """``[max|A−Aᴴ|, max|A|]`` of a ``(..., μ, μ)`` tile — ONE compiled module.

    The jitted callable is cached at module level so repeated gate calls
    reuse the compiled module (``jax.jit`` caches on function identity +
    avals; a fresh ``jit`` per call would recompile at every gate).  See
    the sharding note inside :func:`check_hermitian` for why fusing these
    ops is load-bearing, not a style choice.
    """
    import jax
    import jax.numpy as jnp

    # Keyed by sharding as well as shape: the sharding constraint below is
    # closed over, so a differently-sharded tile needs its own compile.
    sharding = getattr(a, "sharding", None)
    key = ("fn", a.shape, str(a.dtype), sharding)
    fn = _HERM_STATS_CACHE.get(key)
    if fn is None:
        @jax.jit
        def fn(m):
            herm = jnp.conj(jnp.swapaxes(m, -1, -2))
            if sharding is not None:
                # Load-bearing, not cosmetic: without this constraint the
                # SPMD partitioner is free to resolve the (P(x,y), P(y,x))
                # operand conflict of the subtract by ALL-GATHERING BOTH
                # (μ, μ) operands to replicated — it did exactly that even
                # under one jit (AQ 4962c/P=64 cold table 2026-07-28,
                # modules jit_fn 0536/0698: 2 × 398.72 MB per gate call).
                # Pinning herm to m's sharding forces a tile-local
                # transpose reshard (O(μ²/P) per rank) instead.
                herm = jax.lax.with_sharding_constraint(herm, sharding)
            return jnp.stack([
                jnp.max(jnp.abs(m - herm)).astype(jnp.float64),
                jnp.max(jnp.abs(m)).astype(jnp.float64),
            ])
        _HERM_STATS_CACHE[key] = fn
    return fn(a)


def report_hermitian_residual(
    name: str,
    dev: float,
    scale: float,
    *,
    print_fn: Callable[..., Any] = print,
    rtol: float = 1e-8,
    verbose: bool = False,
    detail: str = "",
    always: bool = False,
    cause: str | None = None,
) -> bool:
    """Verdict + message for a Hermiticity residual already reduced to scalars.

    ``dev = max|A − Aᴴ|``, ``scale = max|A|``.  This is the half of
    :func:`check_hermitian` that never touches the array, split out so that a
    check whose reduction MUST happen inside a traced region reaches the same
    verdict, through the same tolerance, with the same wording as the
    construction-site gates.  One implementation of "is this Hermitian and what
    do I tell the human", not two — see ``solvers/lanczos.py``, which reduces
    ``α − αᴴ`` inside the Lanczos ``fori_loop`` (it cannot ``device_get`` there)
    and hands the two scalars back out through a debug callback.

    ``always=True`` evaluates and reports **regardless of ``LORRAX_SANITY``**.
    The off-switch documented at the top of this module is a *cost* escape
    hatch; a caller whose residual was already computed by the algorithm it is
    checking has no cost to escape, and an invariant that a stray environment
    variable can silence is not an invariant.  ``strict`` still raises.

    ``cause`` replaces the default one-line diagnosis ("an index/shard mixing
    bug, not a numerical one").  That default is right for a tile checked at
    its CONSTRUCTION site, where hermiticity is exact by algebra and any
    deviation is an indexing fault.  It is wrong for a residual measured
    DOWNSTREAM of a long operator chain, where the caller may already know
    what the deviation is and what it costs — see ``solvers/lanczos.py``'s
    α gate, whose asymmetry is a measured, deterministic, size-independent
    property of the screened-Coulomb tile W.  A caller that has done that
    work should be able to say so in the message instead of pointing every
    future reader at a collective bug that was ruled out.
    """
    if not (always or sanity_enabled()):
        return True
    dev = float(dev)
    scale = float(scale)
    suffix = f"  {detail}" if detail else ""
    if not np.isfinite(dev) or not np.isfinite(scale):
        warn(f"{name} hermiticity residual is not finite "
             f"(max|A-Aᴴ|={dev}, max|A|={scale}).{suffix}", print_fn=print_fn)
        return False
    rel = dev / scale if scale > 0.0 else dev
    if verbose:
        print_fn(f"  sanity[{name}]: hermiticity residual = {rel:.3e} "
                 f"(abs {dev:.3e}, scale {scale:.3e})")
    if rel > float(rtol):
        why = cause if cause is not None else (
            "A Hermitian-by-construction tile losing hermiticity means an "
            "index/shard mixing bug, not a numerical one.")
        warn(
            f"{name} is NOT Hermitian: max|A-Aᴴ|/max|A| = {rel:.3e} > "
            f"{float(rtol):.1e} (abs {dev:.3e} on scale {scale:.3e}).  "
            f"{why}{suffix}",
            print_fn=print_fn,
        )
        return False
    return True


def check_hermitian(
    name: str,
    matrix: Any,
    *,
    print_fn: Callable[..., Any] = print,
    rtol: float = 1e-8,
    verbose: bool = False,
) -> bool:
    """Hermiticity residual on a ``(..., μ, μ)`` tile, at construction.

    ``V_q``, ``W_q``, ``χ₀_q``, ``Σ_xc`` and ``kin_ion`` are all Hermitian
    by construction.  A broken index map, a mis-transposed shard, or a
    collective that mixed unrelated column blocks (the flat-mesh SUMMA
    failure) destroys hermiticity long before it destroys finiteness — so
    this is the earliest structural detector available, and it is one
    reduction.

    Residual is measured relative to the tile's own scale:
    ``max|A − Aᴴ| / max|A|``.
    """
    if not sanity_enabled():
        return True
    if _is_jax(matrix):
        import jax

        a = matrix
        if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
            return True
        # The transpose+subtract+reductions MUST live in ONE compiled
        # module.  Executed eagerly, each op is its own XLA module: the
        # transposed operand materialises with the mesh-transposed
        # sharding, and the eager binary subtract resolves the sharding
        # conflict by ALL-GATHERING BOTH (μ, μ) operands onto every rank
        # — 2 × 399 MB c128[4992,4992] gathers per gate call, seen in the
        # AQ 4962c/P=64 collective table 2026-07-28 (modules
        # jit_subtract 0730/0973).  Fused under one jit, GSPMD reshards
        # the transpose tile-locally (O(μ²/P) per rank) and the output is
        # two scalars, keeping this the "one reduction" the docstring
        # promises.  Re-verify from the HLO dump if this function changes.
        dev, scale = (
            float(v) for v in np.asarray(jax.device_get(_herm_stats(a))))
    else:
        a = np.asarray(matrix)
        if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
            return True
        dev = float(np.max(np.abs(a - np.conj(np.swapaxes(a, -1, -2)))))
        scale = float(np.max(np.abs(a)))
    return report_hermitian_residual(
        name, dev, scale, print_fn=print_fn, rtol=rtol, verbose=verbose)


_NEGQ_STATS_CACHE: dict = {}


def neg_q_index(kgrid) -> np.ndarray:
    """Flat-q permutation ``q -> -q`` on a row-major ``(qx, qy, qz)`` grid.

    The flat leading axis of ``V_qmunu`` / ``W0_qmunu`` is row-major over
    ``kgrid`` — that is the layout ``bse_io._resolve_munu_reader`` reshapes
    with (``.reshape(nkx, nky, nkz, mu, nu)``), so this permutation and the
    BSE's own k-transform agree by construction.
    """
    nx, ny, nz = (int(v) for v in kgrid)
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    return idx[
        (-np.arange(nx)) % nx, :, :][:, (-np.arange(ny)) % ny, :][
        :, :, (-np.arange(nz)) % nz].reshape(-1)


def _negq_stats(a, neg):
    """``[max|A_q − conj(A_{−q})|, max|A|]`` — ONE compiled module.

    Cached on shape/dtype/sharding for the same reason :func:`_herm_stats`
    is: ``jax.jit`` keys on function identity, so a fresh ``jit`` per gate
    call recompiles every time.  The gather is along the flat-q axis, which
    is replicated in every layout this gate is called from (``P(None, 'x',
    'y')``), so it is a rank-local reshuffle and moves no μ-tile bytes.
    """
    import jax
    import jax.numpy as jnp

    sharding = getattr(a, "sharding", None)
    key = ("fn", a.shape, str(a.dtype), sharding, neg.tobytes())
    fn = _NEGQ_STATS_CACHE.get(key)
    if fn is None:
        negj = jnp.asarray(neg)

        @jax.jit
        def fn(m):
            mirror = jnp.conj(jnp.take(m, negj, axis=0))
            if sharding is not None:
                mirror = jax.lax.with_sharding_constraint(mirror, sharding)
            return jnp.stack([
                jnp.max(jnp.abs(m - mirror)).astype(jnp.float64),
                jnp.max(jnp.abs(m)).astype(jnp.float64),
            ])

        _NEGQ_STATS_CACHE[key] = fn
    return fn(a)


def check_q_conjugate_reciprocity(
    name: str,
    tensor: Any,
    kgrid,
    *,
    print_fn: Callable[..., Any] = print,
    rtol: float = 1e-8,
    verbose: bool = False,
) -> bool:
    """``A_q == conj(A_{−q})`` on a flat-q ``(nq, μ, μ)`` tile.

    THE PROPERTY THE BSE KERNEL ACTUALLY DEPENDS ON, and it is *not*
    per-q hermiticity.  In ``bse_stack_matvec._w_stack`` the index μ
    attaches to the conduction pair and ν to the valence pair on BOTH the
    encode and the decode side, so no μ↔ν transpose survives the
    Hermitian conjugate of the assembled kernel::

        H[(cvk),(c'v'k')] ∝ Σ_{μν} conj(ψ_c^{kμ}) ψ_{c'}^{k'μ}
                                   ψ_v^{kν} conj(ψ_{v'}^{k'ν}) W(μ,ν; k−k')

    and ``H = Hᴴ`` reduces to ``W_q = conj(W_{−q})`` alone.  Equivalently:
    ``W_R = ifft_q(W_q)`` is REAL.  Per-q hermiticity is NEITHER NECESSARY
    NOR SUFFICIENT for it — both directions are demonstrable — so
    :func:`check_hermitian` on ``W[q=0]`` is a *neighbouring* property, and
    a tile can pass it at 1e-15 while failing this one by ten orders.

    Checking this at ``q=0`` ALONE is worthless: ``−0 == 0``, so the
    condition degenerates to "``A[0]`` is real", which the analytic
    assembly satisfies at machine epsilon whatever the other q's do.  The
    gate is therefore over the WHOLE flat-q axis, which is also the only
    cost that matters (one reduction, no μ-tile movement).

    Residual is measured relative to the tensor's own scale:
    ``max|A_q − conj(A_{−q})| / max|A|``.
    """
    if not sanity_enabled():
        return True
    a = tensor
    nq = int(np.prod([int(v) for v in kgrid]))
    if getattr(a, "ndim", 0) < 3 or int(a.shape[0]) != nq:
        # Not a flat-q tile on this grid — nothing this gate can say.
        return True
    neg = neg_q_index(kgrid)
    if _is_jax(a):
        import jax

        dev, scale = (
            float(v) for v in np.asarray(jax.device_get(_negq_stats(a, neg))))
    else:
        arr = np.asarray(a)
        dev = float(np.max(np.abs(arr - np.conj(arr[neg]))))
        scale = float(np.max(np.abs(arr)))
    return report_q_conjugate_residual(
        name, dev, scale, print_fn=print_fn, rtol=rtol, verbose=verbose)


def report_q_conjugate_residual(
    name: str,
    dev: float,
    scale: float,
    *,
    print_fn: Callable[..., Any] = print,
    rtol: float = 1e-8,
    verbose: bool = False,
    detail: str = "",
) -> bool:
    """Verdict + message for an already-reduced ``A_q vs conj(A_{−q})`` pair."""
    if not sanity_enabled():
        return True
    dev = float(dev)
    scale = float(scale)
    suffix = f"  {detail}" if detail else ""
    if not np.isfinite(dev) or not np.isfinite(scale):
        warn(f"{name} q↔−q conjugate residual is not finite "
             f"(max|A_q-conj(A_-q)|={dev}, max|A|={scale}).{suffix}",
             print_fn=print_fn)
        return False
    rel = dev / scale if scale > 0.0 else dev
    if verbose:
        print_fn(f"  sanity[{name}]: q↔−q conjugate residual = {rel:.3e} "
                 f"(abs {dev:.3e}, scale {scale:.3e})")
    if rel > float(rtol):
        warn(
            f"{name} violates q↔−q conjugate reciprocity: "
            f"max|A_q - conj(A_-q)|/max|A| = {rel:.3e} > {float(rtol):.1e} "
            f"(abs {dev:.3e} on scale {scale:.3e}).  Equivalently ifft_q(A) "
            f"is NOT real.  This is the property the BSE kernel's "
            f"hermiticity rests on; per-q hermiticity does NOT imply it and "
            f"a q=0-only check cannot see it.{suffix}",
            print_fn=print_fn,
        )
        return False
    return True


def check_positive(
    name: str,
    value: Any,
    *,
    print_fn: Callable[..., Any] = print,
    magnitude_hint: tuple[float, float] | None = None,
) -> bool:
    """Assert a scalar physical invariant is positive (and O(known scale)).

    ``magnitude_hint = (lo, hi)`` additionally brackets the *magnitude* —
    ``tr V_{q=0}`` is positive-definite-by-construction and O(10⁹) for the
    production MoS₂ decks; the v8 forensics showed a 27 % shift in that
    trace between two runs whose V_q must have been identical, which is
    exactly the class of thing a bracket makes visible on the spot.
    """
    if not sanity_enabled():
        return True
    v = float(np.real(np.asarray(value)))
    if not np.isfinite(v):
        warn(f"{name} = {v} is not finite.", print_fn=print_fn)
        return False
    ok = True
    if v <= 0.0:
        warn(f"{name} = {v:.6e} is not positive — this quantity is "
             f"positive by construction.", print_fn=print_fn)
        ok = False
    if magnitude_hint is not None:
        lo, hi = (float(magnitude_hint[0]), float(magnitude_hint[1]))
        if not (lo <= abs(v) <= hi):
            warn(f"{name} = {v:.6e} is outside the expected magnitude "
                 f"window [{lo:.3e}, {hi:.3e}].", print_fn=print_fn)
            ok = False
    return ok


def check_in_range(
    name: str,
    array: Any,
    lo: float,
    hi: float,
    *,
    print_fn: Callable[..., Any] = print,
    unit: str = "",
    verbose: bool = False,
) -> bool:
    """Physicality bracket on an array of observables (min/max in one fetch)."""
    if not sanity_enabled():
        return True
    if _is_jax(array):
        # Reduce on device: a globally-sharded array cannot be pulled to
        # host on a non-addressable rank, and gathering it would defeat
        # the point of a "cheap" gate.
        import jax
        import jax.numpy as jnp
        a_dev = jnp.real(array)
        if a_dev.size == 0:
            return True
        stats = jnp.stack([
            jnp.min(a_dev).astype(jnp.float64),
            jnp.max(a_dev).astype(jnp.float64),
            jnp.sum((a_dev < lo) | (a_dev > hi)).astype(jnp.float64),
            jnp.sum(~jnp.isfinite(a_dev)).astype(jnp.float64),
        ])
        amin, amax, n_bad_f, n_nonfinite = (
            float(v) for v in np.asarray(jax.device_get(stats)))
        size = int(a_dev.size)
        if n_nonfinite:
            warn(f"{name} contains {int(n_nonfinite)} non-finite entries.",
                 print_fn=print_fn)
            return False
        n_bad = int(n_bad_f)
        suffix = f" {unit}" if unit else ""
        if verbose:
            print_fn(f"  sanity[{name}]: range [{amin:.4f}, {amax:.4f}]{suffix}")
        if n_bad:
            warn(
                f"{name}: {n_bad} of {size} entries outside the physical "
                f"window [{lo:g}, {hi:g}]{suffix} (observed range "
                f"[{amin:.4f}, {amax:.4f}]{suffix}).",
                print_fn=print_fn,
            )
            return False
        return True
    a = np.asarray(array)
    if a.size == 0:
        return True
    a = np.real(a)
    if not np.all(np.isfinite(a)):
        warn(f"{name} contains non-finite entries.", print_fn=print_fn)
        return False
    amin, amax = float(a.min()), float(a.max())
    suffix = f" {unit}" if unit else ""
    if verbose:
        print_fn(f"  sanity[{name}]: range [{amin:.4f}, {amax:.4f}]{suffix}")
    n_bad = int(np.count_nonzero((a < lo) | (a > hi)))
    if n_bad:
        warn(
            f"{name}: {n_bad} of {a.size} entries outside the physical "
            f"window [{lo:g}, {hi:g}]{suffix} (observed range "
            f"[{amin:.4f}, {amax:.4f}]{suffix}).",
            print_fn=print_fn,
        )
        return False
    return True


def check_sign(
    name: str,
    array: Any,
    *,
    expect: str = "negative",
    print_fn: Callable[..., Any] = print,
    tol: float = 0.0,
    verbose: bool = False,
) -> bool:
    """Sign invariant on a physical array (e.g. Σ_x diagonal, occupied states).

    The bare exchange self-energy ``Σ_x = −Σ_occ ⟨nk,mk'|V|mk',nk⟩`` is
    strictly negative for every state in a converged calculation — it is a
    negative-definite quadratic form.  A positive entry is not "a slightly
    wrong number", it is a sign/conjugation/index error.
    """
    if not sanity_enabled():
        return True
    if expect not in ("negative", "positive"):
        raise ValueError(f"check_sign: expect must be 'negative'/'positive', "
                         f"got {expect!r}")
    if _is_jax(array):
        import jax
        import jax.numpy as jnp
        a_dev = jnp.real(array)
        if a_dev.size == 0:
            return True
        bad_dev = (a_dev > float(tol)) if expect == "negative" \
            else (a_dev < -float(tol))
        stats = jnp.stack([
            jnp.sum(bad_dev).astype(jnp.float64),
            jnp.min(a_dev).astype(jnp.float64),
            jnp.max(a_dev).astype(jnp.float64),
        ])
        n_bad_f, amin, amax = (
            float(v) for v in np.asarray(jax.device_get(stats)))
        if verbose:
            print_fn(f"  sanity[{name}]: range [{amin:.4f}, {amax:.4f}], "
                     f"expect {expect}")
        if int(n_bad_f):
            worst = amax if expect == "negative" else amin
            warn(
                f"{name}: {int(n_bad_f)} of {int(a_dev.size)} entries are not "
                f"{expect} (worst {worst:.6f}).  This quantity is {expect} by "
                f"construction — suspect a sign, conjugation or band-index "
                f"slip.",
                print_fn=print_fn,
            )
            return False
        return True
    a = np.real(np.asarray(array))
    if a.size == 0:
        return True
    bad = (a > float(tol)) if expect == "negative" else (a < -float(tol))
    n_bad = int(np.count_nonzero(bad))
    if verbose:
        print_fn(f"  sanity[{name}]: range [{a.min():.4f}, {a.max():.4f}], "
                 f"expect {expect}")
    if n_bad:
        worst = float(a.flat[int(np.argmax(np.abs(np.where(bad, a, 0.0))))])
        warn(
            f"{name}: {n_bad} of {a.size} entries are not {expect} "
            f"(worst {worst:.6f}).  This quantity is {expect} by "
            f"construction — suspect a sign, conjugation or band-index slip.",
            print_fn=print_fn,
        )
        return False
    return True


def check_shape(
    name: str,
    array: Any,
    expected: tuple,
    *,
    print_fn: Callable[..., Any] = print,
) -> bool:
    """Shape invariant; ``None`` entries in ``expected`` are wildcards."""
    if not sanity_enabled():
        return True
    got = tuple(int(v) for v in np.shape(array))
    if len(got) != len(expected) or any(
            e is not None and int(e) != g for e, g in zip(expected, got)):
        warn(f"{name} has shape {got}, expected {tuple(expected)}.",
             print_fn=print_fn)
        return False
    return True


def check_count(
    name: str,
    got: int,
    expected: int,
    *,
    print_fn: Callable[..., Any] = print,
    detail: str = "",
) -> bool:
    """Structural count invariant (rows written, floats parsed, ranks kept).

    Counting the floats in ``eqp0.dat`` is what caught the NaN-producing
    back-solve; this makes that count a property of the writer instead of
    a thing someone happened to do by hand afterwards.
    """
    if not sanity_enabled():
        return True
    if int(got) != int(expected):
        warn(f"{name}: got {int(got)}, expected {int(expected)}."
             + (f"  {detail}" if detail else ""), print_fn=print_fn)
        return False
    return True
