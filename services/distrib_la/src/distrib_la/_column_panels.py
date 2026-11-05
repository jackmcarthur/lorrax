"""Products on the flattened mesh: in-place column-panel projections and a replicated Gram.

Two layouts that :func:`distrib_la.local_gemm_plan` does not cover in place:

* **Column face** ``(q, R, n)`` at ``P(None, None, flat)``, ``flat`` every mesh
  axis flattened (``('x','y')`` or ``('y','x')``): each rank holds ``n/P``
  complete columns.  :func:`project_out` applies ``Y <- (1 - Q Q^H) Y`` for a
  replicated ``Q (q, R, t)`` in the rank's own block, streamed over local
  column panels of at most ``panel_bytes``, and :func:`column_norms` reads
  ``||Y[b, :, a]||^2`` the same way.  A whole-face product, or one compiled
  call per panel, cannot alias its output with ``Y`` and holds a second copy
  of it (the centroid sketch measured 2.2x the sketch on device at
  N_mu = 45k); one ``shard_map`` whose panel loop writes back into its carry
  holds one panel beside ``Y``.
* **Row slab** ``(n, m)`` at ``P(flat, None)``: :func:`replicated_gram`
  forms ``a^H b`` on every rank, the ranks' partials added in rank order
  (``lxkit.ordered_psum``), so the bits depend on the rank count only.

Neither creates a native context or calls a provider: the local products are
XLA's dot, as in the axis-layout plan.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import lax, shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from lxkit import ordered_psum

__all__ = ["column_norms", "project_out", "replicated_gram"]

_DTYPES = (jnp.dtype(jnp.float64), jnp.dtype(jnp.complex128))


def _flat_axes(spec, mesh: Mesh, caller: str) -> tuple:
    """The flattened column axes of ``spec = P(None, None, flat)``; refuses any other layout."""
    flat = tuple(spec)[2] if len(tuple(spec)) == 3 else None
    flat = (flat,) if isinstance(flat, str) else flat
    if (tuple(spec)[:2] != (None, None) or not isinstance(flat, tuple)
            or sorted(flat) != sorted(mesh.axis_names)):
        raise ValueError(f"{caller}: spec must be P(None, None, flat) with flat every mesh "
                         f"axis, e.g. P(None, None, ('y', 'x')); got {spec}")
    return flat


def _panel_width(n_loc: int, column_bytes: int, panel_bytes: int) -> tuple[int, int]:
    """``(w, n_panels)``: the fewest panels of at most ``panel_bytes`` covering ``n_loc``, evened out."""
    cap = max(1, min(int(n_loc), int(panel_bytes) // max(1, int(column_bytes))))
    n_panels = -(-int(n_loc) // cap)
    return -(-int(n_loc) // n_panels), n_panels


def _per_row(fn, *xs):
    """``fn`` on each batch row of ``xs``: a 2-D call when ``q == 1``, else ``vmap``."""
    if xs[0].shape[0] == 1:
        return fn(*(x[0] for x in xs))[None]
    return jax.vmap(fn)(*xs)


def _square_norms(panel):
    """``sum_r |panel[r, a]|^2`` of a ``(R, w)`` panel."""
    if jnp.iscomplexobj(panel):
        return jnp.sum(jnp.real(panel) ** 2 + jnp.imag(panel) ** 2, axis=0)
    return jnp.sum(panel * panel, axis=0)


@lru_cache(maxsize=64)
def _panel_programs(mesh: Mesh, spec, q: int, rows: int, n: int, t: int, dtype: str,
                    chunk: int, width: int, n_panels: int, mode: str):
    """``(shard_map, jit)`` of one panel loop; ``mode`` is ``project``, ``project_norms`` or ``norms``.

    Panel ``j`` covers local columns ``[s_j, s_j + w)``, ``s_j = min(j w, n_loc - w)``:
    every panel has one width, the last is clamped to end at ``n_loc``, and the
    columns it shares with panel ``j - 1`` (``fresh`` false) are neither updated
    again nor re-normed.  Per chunk ``c`` of ``Q`` (``chunk`` columns),
    ``W = Q_c^H Y_p`` and ``Y_p + Q_c W'`` with ``W' = -W`` on live rows and fresh
    columns, zero elsewhere: the addition folds into the GEMM (beta = 1) and the
    negation touches the small ``W``.  ``ceil(n_valid / chunk)`` chunks run, at
    most ``t / chunk``; rows of ``W`` at or past ``n_valid`` are zero, so ``Q``'s
    columns there never enter.
    """
    flat = tuple(spec)[2]
    n_loc = int(n) // int(mesh.size)
    real = jnp.finfo(jnp.dtype(dtype)).dtype
    hi = lax.Precision.HIGHEST
    update = mode != "norms"
    norms = mode != "project"

    def project(Yp, Q, n_valid, n_chunks, fresh):
        def step(c, Yp):
            Qc = lax.dynamic_slice_in_dim(Q, c * chunk, chunk, axis=1)
            W = jnp.matmul(jnp.conj(Qc.T), Yp, precision=hi)
            live = (c * chunk + jnp.arange(chunk, dtype=jnp.int32)) < n_valid
            W = jnp.where(live[:, None] & fresh[None, :], -W, jnp.zeros((), W.dtype))
            return Yp + jnp.matmul(Qc, W, precision=hi)
        return lax.fori_loop(0, n_chunks, step, Yp)

    def local(Y, *rest):
        if update:
            Q, n_valid = rest
            n_chunks = jnp.minimum((n_valid + chunk - 1) // chunk, t // chunk)

        def panel(j, carry):
            Y, d = carry
            start = jnp.minimum(j * width, n_loc - width).astype(jnp.int32)
            fresh = start + jnp.arange(width, dtype=jnp.int32) >= j * width
            Yp = lax.dynamic_slice_in_dim(Y, start, width, axis=2)
            if update:
                Yp = _per_row(lambda y, b: project(y, b, n_valid, n_chunks, fresh), Yp, Q)
                Y = lax.dynamic_update_slice_in_dim(Y, Yp, start, axis=2)
            if norms:
                old = lax.dynamic_slice_in_dim(d, start, width, axis=1)
                dp = jnp.where(fresh[None, :], _per_row(_square_norms, Yp), old)
                d = lax.dynamic_update_slice_in_dim(d, dp, start, axis=1)
            return Y, d

        d0 = jnp.zeros((q, n_loc if norms else 0), real)
        Y, d = lax.fori_loop(0, n_panels, panel, (Y, d0))
        return {"project": Y, "project_norms": (Y, d), "norms": d}[mode]

    face, col = P(None, None, flat), P(None, flat)
    in_specs = (face, P(), P()) if update else (face,)
    out_specs = {"project": face, "project_norms": (face, col), "norms": col}[mode]
    body = shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
                     check_vma=False)
    return body, jax.jit(body, donate_argnums=(0,) if update else ())


def _check_face(Y, mesh: Mesh, spec, caller: str) -> tuple:
    """Shape, dtype and layout of a column face ``(q, R, n)``; returns ``(q, R, n, dtype)``."""
    _flat_axes(spec, mesh, caller)
    if getattr(Y, "ndim", None) != 3:
        raise ValueError(f"{caller}: Y must be (q, R, n); got shape {getattr(Y, 'shape', None)}")
    dtype = jnp.dtype(Y.dtype)
    if dtype not in _DTYPES:
        raise TypeError(f"{caller}: Y must be float64 or complex128; got {dtype}")
    q, rows, n = (int(v) for v in Y.shape)
    if n % int(mesh.size):
        raise ValueError(f"{caller}: column extent {n} does not tile {mesh.size} ranks")
    if not isinstance(Y, jax.core.Tracer):
        have = getattr(Y, "sharding", None)
        if getattr(have, "spec", None) != spec or getattr(have, "mesh", None) != mesh:
            raise ValueError(f"{caller}: Y must already be sharded {spec} on the mesh; refusing "
                             f"an implicit reshard of a {Y.shape} array.  Got {have!r}.")
    return q, rows, n, dtype


def project_out(Y, Q, n_valid=None, *, mesh: Mesh, panel_bytes: int, spec,
                chunk: int = 128, norms: bool = True):
    """``Y <- (1 - Q_m Q_m^H) ... (1 - Q_1 Q_1^H) Y`` in place, over local column panels.

    Each rank updates its own complete columns, ``chunk`` columns of ``Q`` at a
    time (``Q_1``, ``Q_2``, ...), with two local GEMMs per chunk and panel and no
    collective.  For orthonormal columns of ``Q`` the chunk sequence is the
    projector ``1 - Q Q^H``; in general it is block modified Gram-Schmidt.

    Parameters
    ----------
    Y : (q, R, n) float64 or complex128 at ``spec``
        The column face, ``n`` a multiple of the mesh size.  DONATED: an eager
        call consumes ``Y``; inside a caller's ``jit`` the update is in place
        when that ``jit`` donates ``Y``.
    Q : (q, R, t), Y's dtype, replicated
        ``t`` a multiple of ``chunk``.
    n_valid : int or replicated int32 scalar, optional
        The live columns of ``Q`` (traced is fine): ``ceil(n_valid / chunk)``
        chunks are applied, and columns at or past ``n_valid`` never enter.
        ``None``: all ``t``.
    mesh : the ``('x','y')`` mesh.
    panel_bytes : int
        Bytes of one local panel ``(q, R, w)``; ``w`` is the even split of the
        rank's ``n/P`` columns into the fewest such panels (at least one
        column).  Pass a fixed tile: the program holds one panel, its update
        and ``W`` beside ``Y``.
    spec : ``P(None, None, flat)``, ``flat`` every mesh axis in either order.
    chunk : int
        Columns of ``Q`` per GEMM pair (the projection's inner extent).
    norms : bool
        Also return ``||Y'[b, :, a]||^2`` of the updated face.

    Returns
    -------
    Y' : (q, R, n) at ``spec``; with ``norms``, also
    d : (q, n) real at ``P(None, flat)``.

    Trace-safe: inside a ``jit`` the ``shard_map`` runs in the caller's
    program; eagerly, a cached ``jit`` donating ``Y``.  Every column is
    updated exactly once, whatever the panel overlap.
    """
    q, rows, n, dtype = _check_face(Y, mesh, spec, "project_out")
    if getattr(Q, "ndim", None) != 3 or tuple(int(v) for v in Q.shape[:2]) != (q, rows):
        raise ValueError(f"project_out: Q must be (q, R, t) = ({q}, {rows}, t); "
                         f"got {getattr(Q, 'shape', None)}")
    if jnp.dtype(Q.dtype) != dtype:
        raise TypeError(f"project_out: Q dtype {Q.dtype} differs from Y's {dtype}")
    t, chunk = int(Q.shape[2]), int(chunk)
    if chunk <= 0 or t <= 0 or t % chunk:
        raise ValueError(f"project_out: Q's {t} columns must be a positive multiple of chunk={chunk}")
    if isinstance(n_valid, int) and not 0 <= n_valid <= t:
        raise ValueError(f"project_out: require 0 <= n_valid <= {t}; got {n_valid}")
    width, n_panels = _panel_width(n // int(mesh.size), dtype.itemsize * q * rows, panel_bytes)
    body, compiled = _panel_programs(mesh, spec, q, rows, n, t, str(dtype), chunk, width,
                                     n_panels, "project_norms" if norms else "project")
    count = jnp.asarray(t if n_valid is None else n_valid, jnp.int32)
    traced = any(isinstance(v, jax.core.Tracer) for v in (Y, Q, count))
    return (body if traced else compiled)(Y, Q, count)


def column_norms(Y, *, mesh: Mesh, panel_bytes: int, spec):
    """``||Y[b, :, a]||^2`` of a column face, streamed over the same local panels; Y unchanged.

    ``Y (q, R, n)`` at ``spec = P(None, None, flat)``; returns ``(q, n)`` real
    at ``P(None, flat)``.  No program holds ``|Y|^2`` whole: each panel of at
    most ``panel_bytes`` is reduced in turn.  Trace-safe (see
    :func:`project_out`); ``Y`` is not donated.
    """
    q, rows, n, dtype = _check_face(Y, mesh, spec, "column_norms")
    width, n_panels = _panel_width(n // int(mesh.size), dtype.itemsize * q * rows, panel_bytes)
    body, compiled = _panel_programs(mesh, spec, q, rows, n, 0, str(dtype), 1, width,
                                     n_panels, "norms")
    return (body if isinstance(Y, jax.core.Tracer) else compiled)(Y)


@lru_cache(maxsize=64)
def _gram_program(mesh: Mesh, axes: tuple, two: bool):
    """The global-view :func:`replicated_gram`: one ``shard_map`` and its ``jit``."""
    rows = P(axes, None)
    if two:
        body = shard_map(lambda a, b: replicated_gram(a, b, axes=axes), mesh=mesh,
                         in_specs=(rows, rows), out_specs=P(), check_vma=False)
    else:
        body = shard_map(lambda a: replicated_gram(a, axes=axes), mesh=mesh,
                         in_specs=(rows,), out_specs=P(), check_vma=False)
    return body, jax.jit(body)


def replicated_gram(a, b=None, *, axes, mesh: Mesh | None = None):
    """``a^H b`` of row-distributed tall panels, on every rank, summed in rank order.

    ``a (n, m)`` and ``b (n, k)`` (``b = a`` when ``None``) have their rows split
    over the mesh axes ``axes``, rank ``r`` (the flattened index of ``axes``,
    first axis major) holding rows ``[r n/P, (r + 1) n/P)``.  Each rank forms its
    local ``a_r^H b_r`` and the partials are added ``r = 0, 1, ...`` on every
    rank (``lxkit.ordered_psum``): the result is the same on every rank and on
    every placement of the same rank count.  Rows the caller pads with zeros add
    nothing.

    Parameters
    ----------
    a, b : row slabs
        With ``mesh=None`` (inside the caller's ``shard_map``): the rank's own
        slabs ``(n/P, m)``, ``(n/P, k)``.  With ``mesh``: the global
        ``(n, m)``, ``(n, k)`` at ``P(axes, None)`` (concrete or traced).
    axes : tuple of str
        The mesh axes the rows are split over, in rank order.
    mesh : Mesh, optional
        The global-view entry; its ``jit`` is cached per mesh and ``axes``.

    Returns
    -------
    (m, k) replicated: ``P()`` on ``mesh``, or the rank's identical copy
    inside a ``shard_map``.  The local product is ``m k n/P`` multiply-adds;
    the reduction moves ``(m, k)`` in bounded slabs.
    """
    axes = tuple(axes) if isinstance(axes, (tuple, list)) else (axes,)
    if mesh is not None:
        args = (a,) if b is None else (a, b)
        want = NamedSharding(mesh, P(axes, None))
        for x in args:
            have = getattr(x, "sharding", None)
            if not isinstance(x, jax.core.Tracer) and (
                    getattr(have, "spec", None) != want.spec or getattr(have, "mesh", None) != mesh):
                raise ValueError(f"replicated_gram: operands must already be sharded {want.spec} "
                                 f"on the mesh; refusing an implicit reshard.  Got {have!r}.")
        body, compiled = _gram_program(mesh, axes, b is not None)
        traced = any(isinstance(v, jax.core.Tracer) for v in args)
        return (body if traced else compiled)(*args)
    if a.ndim != 2 or (b is not None and (b.ndim != 2 or b.shape[0] != a.shape[0])):
        raise ValueError(f"replicated_gram: a (n, m) and b (n, k) row slabs; got "
                         f"{a.shape} and {None if b is None else b.shape}")
    return ordered_psum(jnp.conj(a.T) @ (a if b is None else b), axes)
