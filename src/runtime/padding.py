"""Shape-padding helpers for sharded arrays.

Single source of truth for the μ / band mesh-divisibility padding
contract: arrays in memory may be padded to mesh-divisibility (so a JIT
boundary with a product- or single-axis sharding spec doesn't trip
``ValueError: should be divisible by N``), but files on disk store the
logical (unpadded) extent so they can be re-read on any process count
(SlabIO clips the pad rows against the dataset's own extent on write —
the caller states nothing; readers re-pad by asking for the padded shape,
via :func:`padded_mu_extent`).

The pad zone is zero-filled BY DEFAULT.  Downstream operators that
contract along a padded axis (e.g. einsums in V_q tile, V·χ in W solve)
see no contribution from the pad rows by construction; solves must run at
the LOGICAL extent (see ``isdf/core.solve_zeta`` and
``reports/device_invariance_2026-07-08/ROOT_CAUSE.md``).

A zero pad is inert for operators LINEAR or BILINEAR in the padded axis,
and is a WRONG NUMBER for a diagonalisation — which is why
:func:`pad_axis` takes a keyword-only ``fill``: the BSE ε axis is the
diagonal of a diagonalisation and pads with a signed sentinel
(``bse.bse_window.PAD_EPS_GUARD_RY``) so pad transitions land above the
optical onset rather than under it.  ``psp/dft_operators.py`` pads
``T_diag`` with ``1e10`` for the same reason, and ``sc_iteration``'s
eigensolve refuses instead of padding.

Since 2026-08-22 there is ONE pad helper (:func:`pad_axis`) and it
returns a NAMED result, because the two it replaced returned opposite
extents from the same tuple slot — see :class:`PadAxisResult`.

NOT this module's contract: the G-axis ``ngkmax`` ragged padding
(WFN.h5 ``wfns/coeffs`` style — fixed ``ngkmax`` with per-k ``ngk``
valid counts and sentinel-slot masking).  That is a file-format
convention with per-row valid lengths, not a mesh-divisibility
round-up; do not unify the two.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple


def round_up(n: int, divisor: int) -> int:
    """Round ``n`` up to the next multiple of ``divisor`` (≥ 1).

    THE round-up spelling for pad-extent arithmetic — new call sites
    use this instead of re-deriving ``((n + d - 1) // d) * d`` inline.
    """
    d = max(int(divisor), 1)
    return ((int(n) + d - 1) // d) * d


def round_down(n: int, divisor: int) -> int:
    """Round ``n`` down to a nonnegative multiple of ``divisor``."""
    value = max(int(n), 0)
    d = max(int(divisor), 1)
    return (value // d) * d


def combined_divisor(*divisors: int) -> int:
    """Least common multiple for simultaneous shard-divisor constraints."""
    values = tuple(max(int(value), 1) for value in divisors)
    return math.lcm(*values) if values else 1


def product_divisor(*axis_sizes: int) -> int:
    """Product divisor for one array axis flattened over several mesh axes."""
    product = 1
    for value in axis_sizes:
        product *= max(int(value), 1)
    return product


@dataclass(frozen=True)
class PaddedAxis:
    """Portable receipt for one logical axis in a mesh-legal carrier.

    This is metadata, not an array wrapper.  Physics kernels continue to take
    plain JAX arrays; the producer carries this record beside the array and a
    consumer asks this module for its mask or logical slice.  Keeping the
    record independent of a live :class:`jax.sharding.Mesh` also makes it safe
    to serialize in restart metadata.

    Parameters
    ----------
    name
        Human-readable axis role used in contract errors.
    logical
        Number of physically meaningful entries.
    carrier
        Mesh-legal stored extent, including exact inert pad entries.
    divisor
        Product of the mesh axes assigned to this array axis by its
        ``PartitionSpec``.
    """

    name: str
    logical: int
    carrier: int
    divisor: int

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        logical = int(self.logical)
        carrier = int(self.carrier)
        divisor = int(self.divisor)
        if not name:
            raise ValueError("PaddedAxis.name must be nonempty")
        if logical < 0 or carrier < 0:
            raise ValueError(
                f"{name}: logical={logical} and carrier={carrier} must be >= 0")
        if logical > carrier:
            raise ValueError(
                f"{name}: logical extent {logical} exceeds carrier extent "
                f"{carrier}")
        if divisor < 1:
            raise ValueError(f"{name}: divisor must be >= 1; got {divisor}")
        if carrier % divisor:
            raise ValueError(
                f"{name}: carrier extent {carrier} is not divisible by "
                f"divisor {divisor}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "logical", logical)
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(self, "divisor", divisor)

    @property
    def padded(self) -> int:
        """Backward-compatible spelling for :attr:`carrier`."""
        return self.carrier

    @property
    def pad(self) -> int:
        """Number of inert entries following the logical interval."""
        return self.carrier - self.logical


def padded_axis(
    logical: int,
    divisor_or_mesh,
    *,
    name: str,
    spec=None,
    axis: int | None = None,
    specs: tuple[tuple[object, int], ...] | None = None,
    extra: int = 0,
) -> PaddedAxis:
    """Create the canonical logical-to-mesh-carrier receipt for an axis.

    Give either an integer divisor, or a mesh.  With a mesh and ``spec``, the
    divisor is derived from the mesh axes assigned to ``axis``.  With a mesh
    and no spec, the whole mesh product is used.  Thus callers name geometry;
    none of them reimplements divisibility arithmetic.

    ``extra`` is reserved for controlled padding-invariance tests.  It is
    applied after the ordinary canonical carrier and the result is re-rounded
    to the same divisor.
    """
    n = int(logical)
    extra_n = int(extra)
    if n < 0:
        raise ValueError(f"{name}: logical extent must be >= 0; got {n}")
    if extra_n < 0:
        raise ValueError(f"{name}: extra pad must be >= 0; got {extra_n}")
    if spec is not None and specs is not None:
        raise TypeError(f"{name}: give spec or specs, not both")
    if specs is not None:
        divisors = tuple(
            spec_divisor(divisor_or_mesh, one_spec, int(one_axis))
            for one_spec, one_axis in specs)
        divisor = math.lcm(*divisors) if divisors else 1
    elif spec is None:
        divisor = mesh_divisor(divisor_or_mesh)
    else:
        if axis is None:
            raise TypeError(f"{name}: axis is required when spec is given")
        divisor = spec_divisor(divisor_or_mesh, spec, int(axis))
    carrier = round_up(n, divisor)
    if extra_n:
        carrier = round_up(carrier + extra_n, divisor)
    return PaddedAxis(
        name=str(name), logical=n, carrier=carrier, divisor=divisor)


def authenticate_padded_axis(
    logical: int,
    carrier: int,
    divisor_or_mesh,
    *,
    name: str,
    spec=None,
    axis: int | None = None,
    specs: tuple[tuple[object, int], ...] | None = None,
) -> PaddedAxis:
    """Authenticate the canonical carrier for one logical mesh axis.

    Readers and consumers use this counterpart to :func:`padded_axis` when
    both extents already exist.  ``divisor_or_mesh`` may instead be the
    producer's :class:`PaddedAxis` receipt; that form preserves a deliberately
    requested extra carrier pad rather than reconstructing the smallest
    carrier from its divisor.  Otherwise the expected carrier and every
    divisor are derived here.  Callers never inspect a modulus or silently
    accept an over-padded shape from another convention.
    """
    if isinstance(divisor_or_mesh, PaddedAxis):
        if spec is not None or specs is not None or axis is not None:
            raise TypeError(
                f"{name}: a PaddedAxis receipt cannot be combined with "
                "spec, specs, or axis")
        receipt = divisor_or_mesh
        if int(logical) != receipt.logical:
            raise ValueError(
                f"{name}: logical extent is {int(logical)}, but the "
                f"producer receipt records {receipt.logical}")
        expected = PaddedAxis(
            name=str(name), logical=receipt.logical,
            carrier=receipt.carrier, divisor=receipt.divisor)
    else:
        expected = padded_axis(
            logical, divisor_or_mesh, name=name, spec=spec, axis=axis,
            specs=specs)
    observed = int(carrier)
    if observed != expected.carrier:
        raise ValueError(
            f"{expected.name}: carrier extent is {observed}, expected "
            f"{expected.carrier} for logical extent {expected.logical} and "
            f"divisor {expected.divisor}")
    return expected


def axis_mask(tag: PaddedAxis, *, dtype=bool):
    """Return the valid-entry mask implied by ``tag``."""
    import jax.numpy as jnp

    return (jnp.arange(tag.carrier) < tag.logical).astype(dtype)


def jax_slice_axis(A, stop: int, *, axis: int):
    """JAX/NumPy-compatible leading slice used by owner operations."""
    index = [slice(None)] * int(A.ndim)
    index[int(axis) % int(A.ndim)] = slice(0, int(stop))
    return A[tuple(index)]


def pad_to_axis(A, tag: PaddedAxis, *, axis: int = -1, fill: float = 0.0):
    """Normalize ``A`` to ``tag.carrier`` along ``axis``.

    Inputs may be logical-width or a wider reusable carrier.  Only the leading
    logical interval is meaningful.  A wider input is normalized directly to
    the canonical carrier, then the tagged tail is overwritten with ``fill``.
    This ordering never publishes a non-mesh-divisible logical-width
    intermediate when the source and destination are both legal carriers.
    """
    import jax.numpy as jnp

    ax = int(axis) % int(A.ndim)
    source = int(A.shape[ax])
    if source < tag.logical:
        raise ValueError(
            f"{tag.name}: logical extent {tag.logical} exceeds source carrier "
            f"extent {source} on axis {ax}")
    if source > tag.carrier:
        A = jax_slice_axis(A, tag.carrier, axis=ax)
    elif source < tag.carrier:
        widths = [(0, 0)] * A.ndim
        widths[ax] = (0, tag.carrier - source)
        A = jnp.pad(A, widths, mode="constant", constant_values=fill)
    if not tag.pad:
        return A
    shape = [1] * A.ndim
    shape[ax] = tag.carrier
    mask = axis_mask(tag).reshape(shape)
    return jnp.where(mask, A, jnp.asarray(fill, dtype=A.dtype))


def pad_square(
    A,
    tag: PaddedAxis,
    *,
    axes: tuple[int, int] = (-2, -1),
    fill: float = 0.0,
    pad_diagonal: float | None = None,
):
    """Normalize two matrix axes to one square carrier.

    ``pad_diagonal`` optionally sets the diagonal of the pad-only block after
    zero/sentinel padding.  It is used for identity-embedded rotations and for
    eigensolver sentinels; off-diagonal coupling to the logical block remains
    exactly zero.
    """
    import jax.numpy as jnp

    left, right = (int(a) % int(A.ndim) for a in axes)
    if int(A.shape[left]) < tag.logical or int(A.shape[right]) < tag.logical:
        raise ValueError(
            f"{tag.name}: logical square extent {tag.logical} exceeds source "
            f"carrier shape {tuple(int(n) for n in A.shape)} on axes "
            f"{(left, right)}")
    out = pad_to_axis(A, tag, axis=left, fill=fill)
    out = pad_to_axis(out, tag, axis=right, fill=fill)
    if pad_diagonal is not None and tag.pad:
        i = jnp.arange(tag.carrier)
        diag = jnp.where(i < tag.logical, jnp.diagonal(
            out, axis1=left, axis2=right), jnp.asarray(
                pad_diagonal, dtype=out.dtype))
        # Matrix axes are trailing at every current consumer.  Refuse a future
        # non-trailing spelling rather than inventing a second scatter rule.
        if (left, right) != (out.ndim - 2, out.ndim - 1):
            raise ValueError(
                f"{tag.name}: pad_diagonal requires trailing matrix axes; "
                f"got {(left, right)} for rank {out.ndim}")
        out = out.at[..., i, i].set(diag)
    return out


def strip_axis(A, tag: PaddedAxis, *, axis: int = -1):
    """Return the leading logical interval at a consumer boundary."""
    ax = int(axis) % int(A.ndim)
    source = int(A.shape[ax])
    if source < tag.logical:
        raise ValueError(
            f"{tag.name}: logical extent {tag.logical} exceeds source carrier "
            f"extent {source} on axis {ax}")
    if source == tag.logical:
        return A
    return jax_slice_axis(A, tag.logical, axis=ax)


def authenticate_axis(A, tag: PaddedAxis, *, axis: int = -1, where: str):
    """Authenticate that ``A`` carries exactly the extent named by ``tag``."""
    ax = int(axis) % int(A.ndim)
    observed = int(A.shape[ax])
    if observed != tag.carrier:
        raise ValueError(
            f"{where}: {tag.name} carrier extent is {observed}, expected "
            f"{tag.carrier} for logical extent {tag.logical} and divisor "
            f"{tag.divisor}")
    return A


def bounded_partition_tile(extent: int, max_tile: int, alignment: int) -> int:
    """Largest equal-partition tile not exceeding ``max_tile``.

    Returns a positive tile that divides ``extent`` exactly and is itself a
    multiple of ``alignment``.  The helper is for a sharded outer extent that
    must be processed as equal static-shape internal tiles: exact division
    avoids a second remainder executable, while alignment preserves the
    sharding contract of every tile.  Returns 0 when no positive aligned tile
    can fit under the cap.

    This is selection arithmetic only; it neither pads nor changes the outer
    logical extent.  In particular, an explicit outer chunk remains exactly
    the size the caller requested.
    """
    n = int(extent)
    cap = int(max_tile)
    align = max(int(alignment), 1)
    if n <= 0 or cap < align or n % align:
        return 0
    units = n // align
    cap_units = min(units, cap // align)
    if cap_units <= 0:
        return 0
    min_tiles = (units + cap_units - 1) // cap_units
    for ntiles in range(min_tiles, units + 1):
        if units % ntiles == 0:
            return (units // ntiles) * align
    return 0


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

    Since the orbit-packed in-memory centroid order (2026-09-05,
    ``common.centroid_basis``) the knob sizes only the CANONICAL carrier
    files are read and written at; the in-memory extent is the packed
    layout's and does not move with it.
    """
    import os
    raw = os.environ.get("LORRAX_EXTRA_MU_PAD", "").strip()
    if not raw:
        return 0
    val = int(raw)
    if val < 0:
        raise ValueError(f"LORRAX_EXTRA_MU_PAD must be >= 0; got {val}")
    return val


def padded_mu_axis(n_rmu: int, divisor) -> PaddedAxis:
    """Return the charge/current-centroid logical-to-carrier receipt.

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
    return padded_axis(
        int(n_rmu), divisor, name="centroid mu", extra=extra_mu_pad())


def padded_mu_extent(n_rmu: int, divisor) -> int:
    """Compatibility scalar for callers not yet carrying the μ receipt."""
    return padded_mu_axis(n_rmu, divisor).carrier


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


class PadAxisResult(NamedTuple):
    """What :func:`pad_axis` returns.  THE reason it is a named tuple.

    Before 2026-08-22 the tree had TWO mesh-divisibility pad helpers whose
    second return value was the OPPOSITE extent from the same slot:
    ``runtime.padding.pad_axis_to`` returned the LOGICAL extent,
    ``bse.bse_window._pad_axis_to_multiple`` returned the PADDED one.  Both
    were spelled ``A, n = helper(...)``, so a call site copied from the
    wrong neighbour compiles, runs, and is wrong only when the extent was
    not already a mesh multiple — invisible on every mesh-divisible
    validated run.  That wrong answer already happened once in the BSE
    (the comment recording it is in this file's git history).

    So the arithmetic is single-sourced AND the ambiguity is removed at the
    same time, which is the only safe way to unify the two: a caller has to
    write ``.logical`` or ``.padded``.  Silently swapping either helper's
    single return would have reintroduced the bug rather than fixed it
    (register row `bse/common`, "Do not swap a single-value return").

    Fields
    ------
    array
        The padded array.  The SAME object when nothing was padded, so
        divisible extents stay byte-identical.
    logical
        The pre-pad extent — how many rows carry data.
    padded
        The post-pad extent — ``array.shape[axis]``, and what a sharding
        divisibility guard (e.g. ``bse_ring_comm``'s ``n_cond_pad % px ==
        0``) must be handed.
    """

    array: object
    tag: PaddedAxis

    @property
    def logical(self) -> int:
        return self.tag.logical

    @property
    def padded(self) -> int:
        return self.tag.carrier


def pad_axis(
    A,
    divisor_or_mesh,
    *,
    axis: int = -1,
    fill: float = 0.0,
    name: str = "array axis",
    spec=None,
):
    """Pad ``axis`` of ``A`` up to ``round_up(extent, divisor)``.

    THE mesh-divisibility pad for an array axis — one implementation for
    every consumer that has to make an axis divide a device mesh.  Returns
    a :class:`PadAxisResult`; read ``.logical`` or ``.padded`` by name.

    Three established uses, and they are the same arithmetic:

    * ``axis=-1`` (:func:`pad_last_axis_to`) — the NRHS pad for
      distributed solves whose block-cyclic RHS descriptor needs
      last-axis divisibility.  Zero RHS columns give zero solution
      columns.
    * ``axis=1``, ``fill=0.0`` — the BAND pad.  ``psi`` is ``(n_k, nb,
      nspinor, ngkmax)`` and band-flat sharding needs ``nb`` to divide the
      mesh.  Pad bands are ψ = 0, so every quantity linear or bilinear in
      ψ is exactly zero on them: zero centroid samples in
      ``wfn_transforms.gflat_to_rmu``, zero rows AND columns of
      ⟨m|O|n⟩ in ``common.mtxel_sweep``.  Nothing about the physics
      knows the pad is there.
    * ``axis=1``, ``fill=±PAD_EPS_GUARD_RY`` — the BSE ε pad.  ε is the
      diagonal of a diagonalisation, where a ZERO pad is not inert but a
      wrong number (it puts pad transitions below the optical onset), so
      the pad zone carries a signed sentinel instead.  ``fill`` is
      KEYWORD-ONLY for that reason: a positionally-supplied fill lets a
      call site sign the guard by accident, which is the one way to put
      pad modes back under the onset.  ``eps_c`` pads ``+guard`` and
      ``eps_v`` pads ``-guard``.

    The band case is not optional at production shapes: ``nb`` must
    divide ``∏ p_a``, and e.g. ``nb = 600`` on an 8×8 mesh does not
    (600 = 64·9.375).  Constructing that sharded array raises JAX's
    ``IndivisibleError`` outright (measured, job 7888869) rather than
    degrading — mesh divisibility is JAX's constraint, not ours
    (decisions.md 2026-08-04).
    """
    ax = int(axis) % int(A.ndim)
    n = int(A.shape[ax])
    tag = padded_axis(
        n, divisor_or_mesh, name=name, spec=spec,
        axis=ax if spec is not None else None)
    return PadAxisResult(pad_to_axis(A, tag, axis=ax, fill=fill), tag)


def pad_last_axis_to(A, divisor):
    """``pad_axis(A, divisor, axis=-1)`` — the named NRHS spelling.

    Returns the same :class:`PadAxisResult`; NRHS consumers want
    ``.logical`` (the real column count to slice back to).
    """
    return pad_axis(A, divisor, axis=-1)


def mesh_divisor(mesh_or_int) -> int:
    """``∏ p_a`` for a Mesh, or the int itself.

    The worst-case shard divisor, for a pad that must be safe under ANY
    spec.  When the spec is known, prefer :func:`spec_divisor` — a pad to
    ``∏ p_a`` where the axis is only sharded over ``'x'`` wastes bands.
    """
    if isinstance(mesh_or_int, int):
        return int(mesh_or_int)
    try:
        shape = mesh_or_int.shape
        axes = getattr(mesh_or_int, "axis_names", tuple(shape))
        prod = 1
        for axis in axes:
            prod *= int(shape[axis])
        return prod
    except (AttributeError, TypeError) as exc:
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
    "round_down",
    "combined_divisor",
    "product_divisor",
    "PaddedAxis",
    "padded_axis",
    "authenticate_padded_axis",
    "axis_mask",
    "pad_to_axis",
    "pad_square",
    "strip_axis",
    "authenticate_axis",
    "extra_mu_pad",
    "padded_mu_axis",
    "padded_mu_extent",
    "solve_at_logical",
    "PadAxisResult",
    "pad_axis",
    "pad_last_axis_to",
    "mesh_divisor",
    "spec_divisor",
]
