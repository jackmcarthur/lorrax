"""Fixed-shape kernels for neighbouring-k parallel transport.

The stored forward link has the orientation

``L_i(k) X(k+b_i) L_i(k)^H``

and therefore transports a band-space operator from ``k+b_i`` back to
``k``. Wavefunctions use the WfnLoader sphere layout
``(1, nb_pad, nspinor, ngkmax)``. The link kernel gathers the neighbour
onto the central G sphere, shards central bands over ``x`` and neighbour
bands over ``y``, and forms the raw overlap at ``P("x", "y")``. G and
spin are replicated contraction axes, so no process owns a full band matrix.
The planned distributed polar factor is called inside that same compiled
graph: the raw overlap never crosses a Python dispatch boundary.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


__all__ = [
    "build_forward_neighbor_table",
    "build_g_wrap_lookup",
    "band_storage_extent",
    "fourth_order_connection",
    "g_wrap_for_forward_step",
    "inverse_neighbor_table",
    "make_distributed_band_matmul",
    "make_cross_k_link",
    "wfn_fingerprint",
]

_BAND_MATMUL_CACHE = {}


def band_storage_extent(mesh, nbands: int) -> int:
    """Return the one physical band extent shared by all PT layouts.

    Wavefunctions shard bands over the composite mesh while matrices shard
    their two band axes separately. Rounding to the full mesh product is
    accepted by both layouts, including rectangular meshes.
    """
    from runtime.padding import round_up

    names = tuple(str(a) for a in mesh.axis_names)
    if names != ("x", "y"):
        raise ValueError(
            "parallel transport requires mesh axes ('x','y'); "
            f"got {names!r}")
    divisor = int(mesh.shape["x"]) * int(mesh.shape["y"])
    return round_up(int(nbands), divisor)


def wfn_fingerprint(wfn) -> str:
    """Cheap SHA-256 identity for the fixed-gauge WFN used by PT.

    Coefficients are too large to hash at startup. For a real loader the
    resolved path and full stat identity conservatively stamp its gauge
    generation; modifying, replacing, copying or moving it requires a new
    artifact.
    """
    import hashlib

    digest = hashlib.sha256()
    for value in (
        np.ascontiguousarray(np.asarray(wfn.energies, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(wfn.kpoints, dtype=np.float64)),
    ):
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    digest.update(
        str((int(wfn.nelec), int(wfn.nspinor), int(wfn.nbands))).encode())
    path = getattr(wfn, "path", None)
    if path is not None:
        resolved = os.path.realpath(os.fspath(path))
        stat = os.stat(resolved)
        digest.update(resolved.encode("utf-8"))
        digest.update(str((
            int(stat.st_dev), int(stat.st_ino), int(stat.st_size),
            int(stat.st_mtime_ns), int(stat.st_ctime_ns),
        )).encode())
    return digest.hexdigest()


def build_forward_neighbor_table(
    kvecs_asints: np.ndarray,
    kgrid: Sequence[int],
) -> np.ndarray:
    """Return full-BZ indices of the three positive mesh neighbours."""
    coords = np.asarray(kvecs_asints, dtype=np.int64)
    grid = np.asarray(kgrid, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "kvecs_asints must have shape (nk, 3); "
            f"got {tuple(coords.shape)}")
    if grid.shape != (3,) or np.any(grid <= 0):
        raise ValueError(f"kgrid must be three positive integers; got {grid}")
    if coords.shape[0] != int(np.prod(grid)):
        raise ValueError(
            "parallel transport requires a complete uniform full-BZ grid: "
            f"got {coords.shape[0]} rows for kgrid {tuple(grid)}")

    canonical = np.mod(coords, grid[None, :])
    row_for = {tuple(int(x) for x in row): i
               for i, row in enumerate(canonical)}
    if len(row_for) != canonical.shape[0]:
        raise ValueError("kvecs_asints contains duplicate full-grid rows")

    out = np.empty((canonical.shape[0], 3), dtype=np.int32)
    for ik, row in enumerate(canonical):
        for idir in range(3):
            target = row.copy()
            target[idir] = (target[idir] + 1) % grid[idir]
            try:
                out[ik, idir] = row_for[tuple(int(x) for x in target)]
            except KeyError as exc:
                raise ValueError(
                    "full-grid neighbour is absent: "
                    f"k row {ik}, direction {idir}, target {target.tolist()}"
                ) from exc
    return out


def inverse_neighbor_table(forward: np.ndarray) -> np.ndarray:
    """Invert a ``(nk, 3)`` forward-neighbour permutation table."""
    plus = np.asarray(forward, dtype=np.int64)
    if plus.ndim != 2 or plus.shape[1] != 3:
        raise ValueError(
            f"forward neighbour table must be (nk, 3); got {plus.shape}")
    nk = plus.shape[0]
    minus = np.empty_like(plus, dtype=np.int32)
    want = np.arange(nk, dtype=np.int64)
    for idir in range(3):
        col = plus[:, idir]
        if np.any(col < 0) or np.any(col >= nk) \
                or not np.array_equal(np.sort(col), want):
            raise ValueError(
                f"direction {idir} is not a permutation of the full k grid")
        minus[col, idir] = want
    return minus


def g_wrap_for_forward_step(
    unfolded_kpts: np.ndarray,
    center_full: int,
    neighbor_full: int,
    direction: int,
    kgrid: Sequence[int],
    *,
    atol: float = 2.0e-7,
) -> np.ndarray:
    """Return the reciprocal-lattice wrap for ``k + b_i``.

    The integer obeys ``k_center + e_i/N_i = k_neighbor + G_wrap``;
    neighbour coefficients are therefore gathered at
    ``G_neighbor = G_center + G_wrap``.
    """
    kpts = np.asarray(unfolded_kpts, dtype=np.float64)
    grid = np.asarray(kgrid, dtype=np.int64)
    idir = int(direction)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(f"unfolded_kpts must be (nk, 3); got {kpts.shape}")
    if idir < 0 or idir >= 3:
        raise ValueError(f"direction must be 0, 1 or 2; got {direction}")
    step = np.zeros(3, dtype=np.float64)
    step[idir] = 1.0 / float(grid[idir])
    delta = kpts[int(center_full)] + step - kpts[int(neighbor_full)]
    wrap = np.rint(delta).astype(np.int32)
    if not np.allclose(delta, wrap, rtol=0.0, atol=float(atol)):
        raise ValueError(
            "forward neighbour does not differ by a reciprocal vector: "
            f"center={center_full}, neighbor={neighbor_full}, idir={idir}, "
            f"delta={delta.tolist()}")
    return wrap


def build_g_wrap_lookup(
    g_neighbor: np.ndarray,
    g_center: np.ndarray,
    g_wrap: np.ndarray,
    *,
    ngk_neighbor: int,
    ngk_center: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map the fixed central G sphere into a wrapped neighbour sphere.

    Missing plane waves and all ``ngkmax`` pad rows receive index zero with
    a false mask.
    """
    g_n = np.asarray(g_neighbor)
    g_c = np.asarray(g_center)
    wrap = np.asarray(g_wrap)
    if g_n.ndim != 2 or g_n.shape[1] != 3:
        raise ValueError(f"g_neighbor must be (ngkmax, 3); got {g_n.shape}")
    if g_c.shape != g_n.shape:
        raise ValueError(
            "central and neighbour fixed G tables must have equal shapes; "
            f"got {g_c.shape} and {g_n.shape}")
    if wrap.shape != (3,):
        raise ValueError(f"g_wrap must have shape (3,); got {wrap.shape}")
    ngn = int(ngk_neighbor)
    ngc = int(ngk_center)
    if not (0 <= ngn <= g_n.shape[0] and 0 <= ngc <= g_c.shape[0]):
        raise ValueError(
            f"logical ngk outside ngkmax={g_n.shape[0]}: "
            f"neighbor={ngn}, center={ngc}")

    row_for = {tuple(int(x) for x in row): i
               for i, row in enumerate(g_n[:ngn])}
    index = np.zeros(g_c.shape[0], dtype=np.int32)
    valid = np.zeros(g_c.shape[0], dtype=bool)
    for ig, target in enumerate(g_c[:ngc] + wrap[None, :]):
        hit = row_for.get(tuple(int(x) for x in target))
        if hit is not None:
            index[ig] = hit
            valid[ig] = True
    return index, valid


def make_cross_k_link(mesh, polar_plan):
    """Build the fixed-shape JITs used by the streamed link sweep.

    ``polar_plan`` is the eagerly resolved, trace-safe operation returned by
    ``distrib_la.plan_polar_factor``. Planning remains outside the streamed
    IBZ loop, while overlap formation and the polar factor share one outer
    JIT so the distributed raw overlap stays on device and has no standalone
    dispatch or lifetime.

    Returns
    -------
    center_on_x, link
        ``center_on_x`` reshards the central wavefunctions once per IBZ
        point. ``link`` accepts that resident centre, one neighbour and the
        G-wrap lookup and returns ``(L, s)``. ``L`` has ``P('x','y')`` and
        the length-band singular spectrum ``s`` is replicated according to
        the distributed-linear-algebra service contract.
    """
    sphere_x = NamedSharding(mesh, P(None, "x", None, None))
    sphere_y = NamedSharding(mesh, P(None, "y", None, None))
    block_xy = NamedSharding(mesh, P("x", "y"))

    @jax.jit
    def center_on_x(center_xy):
        return jax.lax.with_sharding_constraint(center_xy, sphere_x)

    @jax.jit
    def link(center_x, neighbor_xy, g_index, g_valid):
        neighbor = jnp.take(neighbor_xy, g_index, axis=-1)
        neighbor = jnp.where(
            g_valid[None, None, None, :], neighbor,
            jnp.zeros((), dtype=neighbor.dtype))
        neighbor_y = jax.lax.with_sharding_constraint(neighbor, sphere_y)
        raw = jnp.einsum(
            "kbsg,knsg->kbn", jnp.conj(center_x), neighbor_y,
            optimize=True)
        raw = jax.lax.with_sharding_constraint(raw[0], block_xy)
        return polar_plan(raw)

    return center_on_x, link


def make_distributed_band_matmul(mesh, *, n_batch_axes: int):
    """Return ``C = A B`` with no full band matrix on any process.

    The caller's batch axes are replicated. ``A`` is gathered only along
    the column mesh axis to ``P(...,'x',None)``; ``B`` is gathered only
    along the row mesh axis to ``P(...,None,'y')``. Their contracted band
    axis is then replicated locally and the result remains
    ``P(...,'x','y')``. This is one-axis gathering of band slabs, never an
    all-gather of an ``nb*nb`` matrix.
    """
    n_batch = int(n_batch_axes)
    key = (id(mesh), n_batch)
    hit = _BAND_MATMUL_CACHE.get(key)
    if hit is not None:
        return hit
    batch = (None,) * n_batch
    left_spec = P(*batch, "x", None)
    right_spec = P(*batch, None, "y")
    out_spec = P(*batch, "x", "y")
    left_sharding = NamedSharding(mesh, left_spec)
    right_sharding = NamedSharding(mesh, right_spec)
    out_sharding = NamedSharding(mesh, out_spec)

    @jax.jit
    def matmul(left, right):
        left_x = jax.lax.with_sharding_constraint(left, left_sharding)
        right_y = jax.lax.with_sharding_constraint(right, right_sharding)
        out = jnp.einsum("...ij,...jk->...ik", left_x, right_y,
                         optimize=True)
        return jax.lax.with_sharding_constraint(out, out_sharding)

    _BAND_MATMUL_CACHE[key] = matmul
    return matmul


def fourth_order_connection(
    forward_links: jax.Array,
    forward_neighbors: np.ndarray,
    reduced_spacing: Sequence[float],
    *,
    band_matmul,
) -> jax.Array:
    """Construct the Hermitian fourth-order reduced-coordinate connection.

    ``forward_links`` is ``(3, nk, nb, nb)`` and may be tiled
    ``P(None,None,'x','y')``. The result has the same shape and is
    value-level fourth-order finite-difference parity, not bit parity with a
    continuum derivative.
    """
    links = jnp.asarray(forward_links)
    plus = np.asarray(forward_neighbors, dtype=np.int32)
    spacing = np.asarray(reduced_spacing, dtype=np.float64)
    if links.ndim != 4 or links.shape[0] != 3 \
            or links.shape[-2] != links.shape[-1]:
        raise ValueError(
            "forward_links must be (3, nk, nb, nb); "
            f"got {tuple(links.shape)}")
    if plus.shape != (links.shape[1], 3):
        raise ValueError(
            f"forward_neighbors must be ({links.shape[1]}, 3); got {plus.shape}")
    if spacing.shape != (3,) or np.any(spacing <= 0.0):
        raise ValueError(
            f"reduced_spacing must be three positive values; got {spacing}")
    minus = inverse_neighbor_table(plus)
    components = []
    for idir in range(3):
        lp1 = links[idir]
        km1 = minus[:, idir]
        km2 = minus[km1, idir]
        lp2 = band_matmul(lp1, lp1[plus[:, idir]])
        lm1 = jnp.swapaxes(jnp.conj(lp1[km1]), -1, -2)
        lm2 = band_matmul(
            lm1, jnp.swapaxes(jnp.conj(lp1[km2]), -1, -2))
        A = (1.0j / (12.0 * float(spacing[idir]))) * (
            -lp2 + 8.0 * lp1 - 8.0 * lm1 + lm2)
        A = 0.5 * (A + jnp.swapaxes(jnp.conj(A), -1, -2))
        components.append(A)
    return jnp.stack(components, axis=0)
