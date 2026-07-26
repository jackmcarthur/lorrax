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
        import jax.numpy as jnp

        arr = a
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
        stats = jnp.stack([
            jnp.sum(~finite).astype(jnp.float64),
            jnp.sum(isnan).astype(jnp.float64),
            jnp.max(mag).astype(jnp.float64),
        ])
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
        import jax.numpy as jnp

        a = matrix
        if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
            return True
        herm = jnp.conj(jnp.swapaxes(a, -1, -2))
        stats = jnp.stack([
            jnp.max(jnp.abs(a - herm)).astype(jnp.float64),
            jnp.max(jnp.abs(a)).astype(jnp.float64),
        ])
        dev, scale = (float(v) for v in np.asarray(jax.device_get(stats)))
    else:
        a = np.asarray(matrix)
        if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
            return True
        dev = float(np.max(np.abs(a - np.conj(np.swapaxes(a, -1, -2)))))
        scale = float(np.max(np.abs(a)))
    if not np.isfinite(dev) or not np.isfinite(scale):
        warn(f"{name} hermiticity residual is not finite "
             f"(max|A-Aᴴ|={dev}, max|A|={scale}).", print_fn=print_fn)
        return False
    rel = dev / scale if scale > 0.0 else dev
    if verbose:
        print_fn(f"  sanity[{name}]: hermiticity residual = {rel:.3e} "
                 f"(abs {dev:.3e}, scale {scale:.3e})")
    if rel > float(rtol):
        warn(
            f"{name} is NOT Hermitian: max|A-Aᴴ|/max|A| = {rel:.3e} > "
            f"{float(rtol):.1e} (abs {dev:.3e} on scale {scale:.3e}).  "
            f"A Hermitian-by-construction tile losing hermiticity means an "
            f"index/shard mixing bug, not a numerical one.",
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
