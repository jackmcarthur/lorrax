"""Shape-padding helpers for sharded arrays.

Single source of truth for the μ / band mesh-divisibility padding
contract: arrays in memory may be padded to mesh-divisibility (so a JIT
boundary with a product- or single-axis sharding spec doesn't trip
``ValueError: should be divisible by N``), but files on disk store the
logical (unpadded) extent so they can be re-read on any process count
(SlabIO clips the pad rows against the dataset's own extent on write —
the caller states nothing; readers re-pad by asking for the padded shape,
via :func:`padded_mu_extent`).

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


def pad_axis_to(A, divisor, *, axis: int = -1):
    """Zero-pad ``axis`` of ``A`` up to ``round_up(extent, divisor)``.

    THE mesh-divisibility pad for an array axis, shared by every consumer
    that has to make an axis divide a device mesh.  Returns
    ``(A_padded, n_orig)``; a no-op returning the SAME array when the
    extent already divides, so divisible meshes stay byte-identical.

    Two established uses, and they are the same arithmetic:

    * ``axis=-1`` (:func:`pad_last_axis_to`) — the NRHS pad for
      distributed solves whose block-cyclic RHS descriptor needs
      last-axis divisibility.  Zero RHS columns give zero solution
      columns.
    * ``axis=1`` — the BAND pad.  ``psi`` is ``(n_k, nb, nspinor,
      ngkmax)`` and band-flat sharding needs ``nb`` to divide the mesh.
      Pad bands are ψ = 0, so every quantity linear or bilinear in ψ is
      exactly zero on them: zero centroid samples in
      ``wfn_transforms.gflat_to_rmu``, zero rows AND columns of
      ⟨m|O|n⟩ in ``common.mtxel_sweep``.  Nothing about the physics
      knows the pad is there.

    The band case is not optional at production shapes: ``nb`` must
    divide ``∏ p_a``, and e.g. ``nb = 600`` on an 8×8 mesh does not
    (600 = 64·9.375).  Constructing that sharded array raises JAX's
    ``IndivisibleError`` outright (measured, job 7888869) rather than
    degrading — mesh divisibility is JAX's constraint, not ours
    (decisions.md 2026-08-04).
    """
    import jax.numpy as jnp

    ax = int(axis) % int(A.ndim)
    n = int(A.shape[ax])
    n_pad = round_up(n, divisor)
    if n_pad == n:
        return A, n
    widths = [(0, 0)] * A.ndim
    widths[ax] = (0, n_pad - n)
    return jnp.pad(A, widths), n


def pad_last_axis_to(A, divisor):
    """``pad_axis_to(A, divisor, axis=-1)`` — kept as the named NRHS spelling."""
    return pad_axis_to(A, divisor, axis=-1)


def mesh_divisor(mesh_or_int) -> int:
    """``∏ p_a`` for a Mesh, or the int itself.

    The worst-case shard divisor, for a pad that must be safe under ANY
    spec.  When the spec is known, prefer :func:`spec_divisor` — a pad to
    ``∏ p_a`` where the axis is only sharded over ``'x'`` wastes bands.
    """
    if isinstance(mesh_or_int, int):
        return int(mesh_or_int)
    try:
        prod = 1
        for axis in mesh_or_int.axis_names:
            prod *= int(mesh_or_int.shape[axis])
        return prod
    except AttributeError as exc:
        raise TypeError(
            f"mesh_divisor: expected an int or a Mesh; "
            f"got {type(mesh_or_int)!r}") from exc


def spec_divisor(mesh, spec, axis: int) -> int:
    """The divisor ``axis`` of an array under ``spec`` must satisfy.

    ``∏`` of the mesh-axis sizes that ``spec`` places on array axis
    ``axis``; 1 when that entry is ``None`` (replicated).

    THE spelling for "how much must this axis be padded by", derived from
    the SPEC rather than assumed.  It matters that this is not
    :func:`mesh_divisor`: a band axis sharded ``P(None,'x',...)`` on an
    8×8 mesh needs a multiple of 8, not 64, and padding to 64 would
    allocate 8× the pad bands for nothing.

    Single-sourced because two subsystems derive it and must agree, or
    ψ produced by one is refused by the other:

    * ``wfn_loader.WfnLoader._default_sharding`` — allocates ψ at
      ``(n_k, round_up(nb, p_band), nspinor, ngkmax)`` so the array is
      born divisible.
    * ``common.mtxel_sweep.SweepGeometry`` — consumes that ψ.

    Both default to ``P(None, ('x','y'), None, None)``, so both get
    ``px·py`` and the loader's ψ needs no further padding.  That
    agreement is the reason the sweep's own band pad is a no-op in
    production, and it is only guaranteed while both call THIS.
    """
    entry = tuple(spec)[axis] if axis < len(tuple(spec)) else None
    if entry is None:
        return 1
    names = (entry,) if isinstance(entry, str) else tuple(entry)
    prod = 1
    for a in names:
        prod *= int(mesh.shape[a])
    return int(prod)


__all__ = [
    "round_up",
    "extra_mu_pad",
    "padded_mu_extent",
    "solve_at_logical",
    "pad_axis_to",
    "pad_last_axis_to",
    "mesh_divisor",
    "spec_divisor",
]
