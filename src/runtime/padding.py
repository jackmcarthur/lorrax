"""Shape-padding helpers for sharded arrays.

Single source of truth for the μ / band mesh-divisibility padding
contract: arrays in memory may be padded to mesh-divisibility (so a JIT
boundary with a product- or single-axis sharding spec doesn't trip
``ValueError: should be divisible by N``), but files on disk store the
logical (unpadded) extent so they can be re-read on any process count
(SlabIO ``valid_shape=`` clips on write; readers re-pad via
:func:`padded_mu_extent`).

The pad zone is always zero-filled.  Downstream operators that contract
along a padded axis (e.g. einsums in V_q tile, V·χ in W solve) see no
contribution from the pad rows by construction; solves must run at the
LOGICAL extent (see ``isdf/core.solve_zeta`` and
``reports/device_invariance_2026-07-08/ROOT_CAUSE.md``).

NOT this module's contract: the G-axis ``ngkmax`` ragged padding
(WFN.h5 ``wfns/coeffs`` style — fixed ``ngkmax`` with per-k ``ngk``
valid counts and sentinel-slot masking).  That is a file-format
convention with per-row valid lengths, not a mesh-divisibility
round-up; do not unify the two.
"""
from __future__ import annotations


def round_up(n: int, divisor: int) -> int:
    """Round ``n`` up to the next multiple of ``divisor`` (≥ 1).

    THE round-up spelling for pad-extent arithmetic — new call sites
    use this instead of re-deriving ``((n + d - 1) // d) * d`` inline.
    """
    d = max(int(divisor), 1)
    return ((int(n) + d - 1) // d) * d


def extra_mu_pad() -> int:
    """TEST-ONLY: extra μ-pad rows requested via ``LORRAX_EXTRA_MU_PAD``.

    Returns the integer value of the ``LORRAX_EXTRA_MU_PAD`` environment
    variable (0 when unset/empty).  Read at call time so tests can flip
    it per-subprocess.

    Purpose: device-count-invariance testing at FIXED device count.
    ``n_rmu_padded = round_up(n_rmu, world_size)`` changes with the
    device count, so a pad-extent bug (a computation that runs on the
    padded instead of the logical μ extent) is normally only visible by
    comparing runs at different P.  This knob forces additional zero
    pad rows on top of the mesh round-up, reproducing e.g. the P=16 pad
    extent in a P=1/P=4 run (see
    ``reports/device_invariance_2026-07-08/ROOT_CAUSE.md``).  Any
    result that changes under this knob at fixed P depends on the pad
    extent and is a defect.

    NEVER set this in production runs — it only wastes memory at best
    and, before the pad-extent fixes, changed answers at worst.
    """
    import os
    raw = os.environ.get("LORRAX_EXTRA_MU_PAD", "").strip()
    if not raw:
        return 0
    val = int(raw)
    if val < 0:
        raise ValueError(f"LORRAX_EXTRA_MU_PAD must be >= 0; got {val}")
    return val


def padded_mu_extent(n_rmu: int, divisor) -> int:
    """Single source of truth for the padded μ extent (``Meta.n_rmu_padded``).

    ``round_up(n_rmu, divisor)`` — plus, when the test-only
    ``LORRAX_EXTRA_MU_PAD`` env knob is set (see :func:`extra_mu_pad`),
    that many extra pad rows, re-rounded so mesh divisibility is
    preserved.  With ``extra % divisor == 0`` (the intended use, e.g.
    extra=12 at P=4 to force the P=16 extent) the result is exactly
    ``round_up(n_rmu, divisor) + extra``.

    Parameters
    ----------
    n_rmu : int
        Logical centroid count.
    divisor : int | Mesh
        Worst-case shard divisor — ``jax.device_count()`` (= ∏ p_a) or
        a Mesh whose axis-size product is used.
    """
    if not isinstance(divisor, int):
        try:
            prod = 1
            for axis in divisor.axis_names:
                prod *= int(divisor.shape[axis])
            divisor = prod
        except AttributeError as exc:
            raise TypeError(
                f"padded_mu_extent: divisor must be an int or a Mesh; "
                f"got {type(divisor)!r}") from exc
    base = round_up(n_rmu, divisor)
    extra = extra_mu_pad()
    if extra:
        base = round_up(base + extra, divisor)
    return base


def solve_at_logical(solve_fn, n_logical, mats, rhs=None, *, pad_axes=(-2,)):
    """Run a dense solve on the LOGICAL μ block of padded operands and
    zero-embed the solution back at the padded extent.

    THE grep-able invariant for "solves run at the logical extent"
    (ROOT_CAUSE.md 2026-07-08): identity-padded factorizations regroup
    partial sums per pad extent — catastrophically for near-singular
    LU — so every dense solve must μ-slice to the logical extent and
    zero-refill.  A solver branch routed through this helper cannot
    forget the slice or the re-fill; hand-rolling the idiom at a new
    site is a review flag.

    Parameters
    ----------
    solve_fn : callable
        ``solve_fn(*mats_log)`` or ``solve_fn(*mats_log, rhs_log)`` —
        receives the sliced logical operands, returns the LOGICAL
        solution.  Ridge/regularization terms belong inside (computed
        on the logical block).
    n_logical : int
        Logical μ extent; the padded extent is read off the operands.
    mats : sequence of arrays
        Square ``(..., n, n)`` operands, each sliced
        ``[..., :n_log, :n_log]``.
    rhs : array | None
        Optional ``(..., n, k)`` right-hand side, sliced
        ``[..., :n_log, :]``.  Omit when a square operand doubles as
        the RHS (e.g. Dyson ``W = (I - Vχ)⁻¹ V``) — then set
        ``pad_axes=(-2, -1)``.
    pad_axes : tuple of int
        Output axes zero-padded back to the padded extent — ``(-2,)``
        for an ``(..., n, k)`` solution (μ rows only), ``(-2, -1)``
        for an ``(..., n, n)`` one.

    The pad values are exact zeros (their correct value: RHS pad rows
    are zero), so at zero pad this is bit-identical to calling
    ``solve_fn`` directly.
    """
    import jax.numpy as jnp

    n_log = int(n_logical)
    n_pad = int(mats[0].shape[-1])
    if n_log > n_pad:
        raise ValueError(
            f"solve_at_logical: n_logical={n_log} exceeds operand "
            f"extent {n_pad}")
    args = [m[..., :n_log, :n_log] for m in mats]
    if rhs is not None:
        args.append(rhs[..., :n_log, :])
    out = solve_fn(*args)
    if n_pad == n_log:
        return out
    widths = [(0, 0)] * out.ndim
    for ax in pad_axes:
        widths[ax % out.ndim] = (0, n_pad - n_log)
    return jnp.pad(out, widths)


def pad_last_axis_to(A, divisor):
    """Zero-pad the last axis of ``A`` up to ``round_up(extent, divisor)``.

    The NRHS-pad idiom for distributed solves whose block-cyclic RHS
    descriptor needs last-axis divisibility: zero RHS columns produce
    zero solution columns — trim with ``[..., :n_orig]`` on return.

    Returns ``(A_padded, n_orig)``; a no-op (same array) when already
    divisible.
    """
    import jax.numpy as jnp

    n = int(A.shape[-1])
    n_pad = round_up(n, divisor)
    if n_pad == n:
        return A, n
    widths = [(0, 0)] * (A.ndim - 1) + [(0, n_pad - n)]
    return jnp.pad(A, widths), n


__all__ = [
    "round_up",
    "extra_mu_pad",
    "padded_mu_extent",
    "solve_at_logical",
    "pad_last_axis_to",
]
