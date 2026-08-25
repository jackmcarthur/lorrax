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

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.wfn_layout import band_sphere_spec
from runtime.padding import round_up, spec_divisor


__all__ = [
    "build_forward_neighbor_table",
    "build_neighbor_table",
    "build_g_wrap_lookup",
    "band_storage_extent",
    "fourth_order_connection",
    "fourth_order_covariant_derivative",
    "g_wrap_for_forward_step",
    "g_wrap_for_step",
    "inverse_neighbor_table",
    "make_cross_k_overlap",
    "make_distributed_band_matmul",
    "make_cross_k_link",
    "MIN_STENCIL_POINTS",
    "undersampled_link_axes",
    "WFN_FINGERPRINT_SCHEME",
    "fingerprint_update_value",
    "wfn_fingerprint",
]

#: Minimum distinct mesh points a Cartesian direction needs to support the
#: advertised fourth-order +/-2 link stencil (``fourth_order_connection`` /
#: ``fourth_order_covariant_derivative``).  THE ONE PLACE THIS THRESHOLD IS
#: DECIDED -- both the artifact producer
#: (``file_io.parallel_transport.write_parallel_transport_artifact``) and the
#: driver-entry preflight (``gw.sc_iteration.load_head_velocity_source``) read
#: this name rather than each carrying its own literal ``5``, which is
#: exactly the shadow-accounting failure class this tree's
#: ``docs/dev/QUALITY_PATTERNS.md`` #3 names.
MIN_STENCIL_POINTS: int = 5


def undersampled_link_axes(kgrid) -> list[str]:
    """Cartesian directions whose mesh cannot support the link stencil.

    Applies PER AXIS, and ONLY to the stages that differentiate along k --
    the nearest-neighbour link build and the fourth-order connection/
    covariant-derivative stencil it feeds.  It does NOT apply to the
    unconditional, gauge-free exact-DFT-velocity write
    (``v = p + i[r,V_NL]`` needs no neighbouring k at all), which is why
    that write is no longer gated on this check (KNOWN_LORRAX_ISSUES.md row
    at ``file_io/parallel_transport.py:407-415`` / ``get_dipole_mtxels.py:
    1309-1317``, 2026-08-19).

    A genuinely collapsed direction (``kgrid[i] == 1``, e.g. the z axis of a
    9x9x1 slab deck) is reported exactly like an undersampled periodic one
    (``1 < kgrid[i] < 5``) -- NOT stamped as an analytic zero.  The fix note
    on that same register row is explicit: "Do not fabricate a z derivative
    for the separate covariant parallel-transport mode."  A 2D deck that
    wants only the velocity reaches the links-free producer/consumer path
    instead of asking this function anything; a 2D deck that asks for the
    full ``parallel_transport`` link stage is refused, by name, on every
    caller of this function.

    Returns
    -------
    list[str]
        The undersampled axis names, a subset of ``["x", "y", "z"]``, in
        that fixed order.  Empty iff every direction supports the stencil.
    """
    grid = tuple(int(n) for n in np.asarray(kgrid).reshape(3))
    return [axis for axis, n in zip("xyz", grid) if n < MIN_STENCIL_POINTS]

_BAND_MATMUL_CACHE = {}

# This name is intentionally stable and printed by provenance checks in the
# dipole-fingerprint regression.  Changing the sampled content below requires
# a new scheme name so artifacts can state exactly what was compared.
WFN_FINGERPRINT_SCHEME = (
    "mean-field-content-v1:full-mf-header+bounded-wfns")
_WFN_GVEC_SAMPLE_COUNT = 17
_WFN_COEFF_BAND_SAMPLE_COUNT = 7
_WFN_COEFF_G_SAMPLE_COUNT = 11


def band_storage_extent(mesh, nbands: int) -> int:
    """Return the one physical band extent shared by all PT layouts.

    Wavefunctions shard bands over the composite mesh while matrices shard
    their two band axes separately. Rounding to the full mesh product is
    accepted by both layouts, including rectangular meshes.
    """
    names = tuple(str(a) for a in mesh.axis_names)
    if names != ("x", "y"):
        raise ValueError(
            "parallel transport requires mesh axes ('x','y'); "
            f"got {names!r}")
    divisor = spec_divisor(mesh, band_sphere_spec(), axis=1)
    return round_up(int(nbands), divisor)


def fingerprint_update_value(digest, label: str, value) -> None:
    """Add an ndarray-like value to ``digest`` without object-pointer bytes."""
    array = np.asarray(value)
    digest.update(label.encode("utf-8"))
    digest.update(repr(tuple(int(n) for n in array.shape)).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    if array.dtype.kind in ("O", "S", "U"):
        # HDF5 text metadata can arrive as bytes, unicode, or object arrays.
        # ``repr(array.tolist())`` is stable for those values; ``tobytes`` on
        # an object array would instead digest process-local pointers.
        digest.update(repr(array.tolist()).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())


def _fingerprint_sample_indices(size: int, count: int) -> np.ndarray:
    """Return deterministic, endpoint-inclusive indices bounded by ``count``."""
    if size <= 0:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.linspace(
        0, size - 1, num=min(size, count), dtype=np.int64))


def _fingerprint_h5_attrs(digest, label: str, obj) -> None:
    for key in sorted(obj.attrs):
        fingerprint_update_value(
            digest, f"attr:{label}:{key}", obj.attrs[key])


def _fingerprint_wfn_file(digest, path) -> None:
    """Hash semantic WFN content without hashing its pathname or inode."""
    import h5py

    with h5py.File(path, "r") as h5:
        _fingerprint_h5_attrs(digest, "/", h5)

        # The mean-field header is small relative to the coefficients, so its
        # complete dataset and attribute content is part of the identity.
        if "mf_header" in h5:
            header_objects = [("mf_header", h5["mf_header"])]
            h5["mf_header"].visititems(
                lambda name, obj: header_objects.append(
                    (f"mf_header/{name}", obj)))
            for name, obj in sorted(header_objects, key=lambda pair: pair[0]):
                _fingerprint_h5_attrs(digest, name, obj)
                if isinstance(obj, h5py.Dataset):
                    fingerprint_update_value(
                        digest, f"dataset:{name}", obj[()])

        if "wfns" not in h5:
            raise ValueError(f"WFN file {path!s} has no 'wfns' group")
        wfns = h5["wfns"]
        _fingerprint_h5_attrs(digest, "wfns", wfns)
        for dataset_name in ("gvecs", "coeffs"):
            if dataset_name not in wfns:
                raise ValueError(
                    f"WFN file {path!s} has no 'wfns/{dataset_name}' dataset")
            dataset = wfns[dataset_name]
            _fingerprint_h5_attrs(
                digest, f"wfns/{dataset_name}", dataset)
            fingerprint_update_value(
                digest, f"contract:wfns/{dataset_name}",
                np.asarray(dataset.shape, dtype=np.int64))
            digest.update(dataset.dtype.str.encode("ascii"))

        gvecs = wfns["gvecs"]
        if gvecs.ndim != 2:
            raise ValueError(
                f"WFN wfns/gvecs must be rank 2; got shape {gvecs.shape}")
        for ig in _fingerprint_sample_indices(
                gvecs.shape[0], _WFN_GVEC_SAMPLE_COUNT):
            fingerprint_update_value(
                digest, f"sample:wfns/gvecs:{int(ig)}", gvecs[int(ig), :])

        coeffs = wfns["coeffs"]
        if coeffs.ndim != 4:
            raise ValueError(
                f"WFN wfns/coeffs must be rank 4; got shape {coeffs.shape}")
        bands = _fingerprint_sample_indices(
            coeffs.shape[0], _WFN_COEFF_BAND_SAMPLE_COUNT)
        g_rows = _fingerprint_sample_indices(
            coeffs.shape[2], _WFN_COEFF_G_SAMPLE_COUNT)
        for ib in bands:
            for ig in g_rows:
                # Spinor and real/imag axes are small and are included whole.
                fingerprint_update_value(
                    digest,
                    f"sample:wfns/coeffs:{int(ib)}:{int(ig)}",
                    coeffs[int(ib), :, int(ig), :],
                )


def wfn_fingerprint(wfn) -> str:
    """Cheap, location-independent SHA-256 identity for a fixed-gauge WFN.

    The digest includes the complete ``mf_header`` tree and bounded,
    endpoint-inclusive samples of ``wfns/gvecs`` and ``wfns/coeffs``.  It
    therefore detects every header change and changes in the sampled
    wavefunction rows, while byte-identical copies have the same identity.

    This deliberately is not a full coefficient checksum: a change confined
    to unsampled G-vector or coefficient rows can collide.  Hashing all
    coefficients at each consumer startup would make the provenance check as
    expensive as reading the entire WFN.  Path, inode, timestamps, and HDF5
    storage layout are not part of the mean-field identity.
    """
    import hashlib

    digest = hashlib.sha256()
    digest.update(WFN_FINGERPRINT_SCHEME.encode("ascii"))
    for label, value in (
        ("loaded:energies", np.asarray(wfn.energies, dtype=np.float64)),
        ("loaded:kpoints", np.asarray(wfn.kpoints, dtype=np.float64)),
    ):
        fingerprint_update_value(digest, label, value)
    fingerprint_update_value(
        digest, "loaded:counts",
        np.asarray(
            [int(wfn.nelec), int(wfn.nspinor), int(wfn.nbands)],
            dtype=np.int64),
    )

    path = getattr(wfn, "path", None)
    if path is not None:
        _fingerprint_wfn_file(digest, path)
    return digest.hexdigest()


def build_neighbor_table(
    kvecs_asints: np.ndarray,
    kgrid: Sequence[int],
    steps: np.ndarray,
) -> np.ndarray:
    """Return full-BZ indices for arbitrary integer mesh displacements."""
    coords = np.asarray(kvecs_asints, dtype=np.int64)
    grid = np.asarray(kgrid, dtype=np.int64)
    step_rows = np.asarray(steps, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "kvecs_asints must have shape (nk, 3); "
            f"got {tuple(coords.shape)}")
    if grid.shape != (3,) or np.any(grid <= 0):
        raise ValueError(f"kgrid must be three positive integers; got {grid}")
    if step_rows.ndim != 2 or step_rows.shape[1] != 3:
        raise ValueError(
            f"steps must have shape (nstep, 3); got {step_rows.shape}")
    if coords.shape[0] != int(np.prod(grid)):
        raise ValueError(
            "neighbor lookup requires a complete uniform full-BZ grid: "
            f"got {coords.shape[0]} rows for kgrid {tuple(grid)}")

    canonical = np.mod(coords, grid[None, :])
    row_for = {tuple(int(x) for x in row): i
               for i, row in enumerate(canonical)}
    if len(row_for) != canonical.shape[0]:
        raise ValueError("kvecs_asints contains duplicate full-grid rows")

    out = np.empty((canonical.shape[0], step_rows.shape[0]), dtype=np.int32)
    for ik, row in enumerate(canonical):
        for istep, step in enumerate(step_rows):
            target = np.mod(row + step, grid)
            try:
                out[ik, istep] = row_for[tuple(int(x) for x in target)]
            except KeyError as exc:
                raise ValueError(
                    "full-grid neighbour is absent: "
                    f"k row {ik}, step {step.tolist()}, "
                    f"target {target.tolist()}") from exc
    return out


def build_forward_neighbor_table(
    kvecs_asints: np.ndarray,
    kgrid: Sequence[int],
) -> np.ndarray:
    """Return full-BZ indices of the three positive mesh neighbours."""
    return build_neighbor_table(
        kvecs_asints, kgrid, np.eye(3, dtype=np.int32))


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


def g_wrap_for_step(
    unfolded_kpts: np.ndarray,
    center_full: int,
    neighbor_full: int,
    step_grid: Sequence[int],
    kgrid: Sequence[int],
    *,
    atol: float = 2.0e-7,
) -> np.ndarray:
    """Return G_wrap for an integer mesh step from center to neighbour."""
    kpts = np.asarray(unfolded_kpts, dtype=np.float64)
    grid = np.asarray(kgrid, dtype=np.int64)
    step_int = np.asarray(step_grid, dtype=np.int64)
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError(f"unfolded_kpts must be (nk, 3); got {kpts.shape}")
    if grid.shape != (3,) or step_int.shape != (3,):
        raise ValueError(
            f"kgrid and step_grid must be 3-vectors; got "
            f"{grid.shape} and {step_int.shape}")
    step = step_int.astype(np.float64) / grid.astype(np.float64)
    delta = kpts[int(center_full)] + step - kpts[int(neighbor_full)]
    wrap = np.rint(delta).astype(np.int32)
    if not np.allclose(delta, wrap, rtol=0.0, atol=float(atol)):
        raise ValueError(
            "neighbour does not differ by the requested mesh step plus a "
            f"reciprocal vector: center={center_full}, "
            f"neighbor={neighbor_full}, step={step_int.tolist()}, "
            f"delta={delta.tolist()}")
    return wrap


def g_wrap_for_forward_step(
    unfolded_kpts: np.ndarray,
    center_full: int,
    neighbor_full: int,
    direction: int,
    kgrid: Sequence[int],
    *,
    atol: float = 2.0e-7,
) -> np.ndarray:
    """Return the reciprocal-lattice wrap for ``k + b_i``."""
    idir = int(direction)
    if idir < 0 or idir >= 3:
        raise ValueError(f"direction must be 0, 1 or 2; got {direction}")
    step = np.zeros(3, dtype=np.int32)
    step[idir] = 1
    return g_wrap_for_step(
        unfolded_kpts, center_full, neighbor_full, step, kgrid, atol=atol)


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


def _make_cross_k_overlap_kernel(mesh, finish):
    """One overlap contraction shared by raw-density and polar-link writers."""
    sphere_x = NamedSharding(mesh, P(None, "x", None, None))
    sphere_y = NamedSharding(mesh, P(None, "y", None, None))
    sphere_kqy = NamedSharding(mesh, P(None, None, "y", None, None))
    block_xy = NamedSharding(mesh, P("x", "y"))

    @jax.jit
    def center_on_x(center_xy):
        return jax.lax.with_sharding_constraint(center_xy, sphere_x)

    @jax.jit
    def overlap(center_x, neighbor_xy, g_index, g_valid):
        if g_index.ndim == 1:
            neighbor = jnp.take(neighbor_xy, g_index, axis=-1)
            neighbor = jnp.where(
                g_valid[None, None, None, :], neighbor,
                jnp.zeros((), dtype=neighbor.dtype))
            neighbor_y = jax.lax.with_sharding_constraint(neighbor, sphere_y)
            raw = jnp.einsum(
                "kbsg,knsg->kbn", jnp.conj(center_x), neighbor_y,
                optimize=True)[0]
            raw = jax.lax.with_sharding_constraint(raw, block_xy)
        elif g_index.ndim == 3:
            neighbor = jax.vmap(
                lambda psi_k, index_k: jax.vmap(
                    lambda psi, index: jnp.take(psi, index, axis=-1),
                    in_axes=(0, 0))(psi_k, index_k),
                in_axes=(0, 0))(neighbor_xy, g_index)
            neighbor = jnp.where(
                g_valid[:, :, None, None, :], neighbor,
                jnp.zeros((), dtype=neighbor.dtype))
            neighbor_y = jax.lax.with_sharding_constraint(
                neighbor, sphere_kqy)
            raw = jnp.einsum(
                "kbsg,kqnsg->qkbn", jnp.conj(center_x), neighbor_y,
                optimize=True)
            raw = jax.lax.with_sharding_constraint(
                raw, NamedSharding(mesh, P(None, None, "x", "y")))
        else:
            raise ValueError(
                "cross-k overlap G lookup must have rank 1 or 3; "
                f"got rank {g_index.ndim}")
        return raw if finish is None else finish(raw)

    return center_on_x, overlap


def make_cross_k_overlap(mesh):
    """Return the sharded raw overlap used by finite-q density vertices.

    The result is ``rho_nm(k,q)=<u_n,k|u_m,k+q>`` with both band axes
    distributed as ``P('x','y')``. A rank-1 G lookup returns one band
    matrix; a rank-3 lookup batches k-by-q and returns
    ``P(None,None,'x','y')``. Spinor components and G are contraction axes,
    so scalar, Pauli and four-component wavefunctions use the same path. The
    batched form shares this contraction rather than introducing a second
    density-vertex implementation.
    """
    return _make_cross_k_overlap_kernel(mesh, None)


def make_cross_k_link(mesh, polar_plan):
    """Build the fixed-shape JITs used by the streamed link sweep.

    ``polar_plan`` is resolved outside the stream. The raw overlap and
    distributed polar factor remain in one compiled graph.
    """
    return _make_cross_k_overlap_kernel(mesh, polar_plan)


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


def fourth_order_covariant_derivative(
    operator_k: jax.Array,
    forward_links: jax.Array,
    forward_neighbors: np.ndarray,
    reduced_spacing: Sequence[float],
    *,
    band_matmul,
) -> jax.Array:
    r"""Differentiate a band operator after finite-link parallel transport.

    The stored forward link has the orientation

    ``L_i(k) O(k+b_i) L_i(k)^H``.

    Each neighbour is therefore expressed in the central ``k`` basis before
    the fourth-order stencil is applied.  This is the discrete, structurally
    gauge-covariant spelling of ``partial_i O - i[A_i,O]``; it avoids splitting
    two large gauge-dependent terms whose finite-grid derivatives obey no
    exact product rule.

    Parameters
    ----------
    operator_k
        ``(nk, nb, nb)`` complex band operator at ``P(None,'x','y')``.
    forward_links
        ``(3, nk, nb, nb)`` polar links at
        ``P(None,None,'x','y')``.
    forward_neighbors
        Host ``(nk,3)`` table of the positive mesh neighbours.
    reduced_spacing
        Three positive reduced-coordinate mesh spacings.
    band_matmul
        Distributed batched matrix product for one leading batch axis.

    Returns
    -------
    jax.Array
        ``(3,nk,nb,nb)`` reduced-coordinate covariant derivative, retaining
        the input's two-dimensional band tiling.
    """
    operator = jnp.asarray(operator_k)
    links = jnp.asarray(forward_links)
    plus = np.asarray(forward_neighbors, dtype=np.int32)
    spacing = np.asarray(reduced_spacing, dtype=np.float64)
    if operator.ndim != 3 or operator.shape[-2] != operator.shape[-1]:
        raise ValueError(
            "operator_k must be (nk,nb,nb); "
            f"got {tuple(operator.shape)}")
    expected_links = (3,) + tuple(operator.shape)
    if tuple(links.shape) != expected_links:
        raise ValueError(
            f"forward_links must have shape {expected_links}; "
            f"got {tuple(links.shape)}")
    if plus.shape != (operator.shape[0], 3):
        raise ValueError(
            f"forward_neighbors must be ({operator.shape[0]},3); "
            f"got {plus.shape}")
    if spacing.shape != (3,) or np.any(spacing <= 0.0):
        raise ValueError(
            f"reduced_spacing must be three positive values; got {spacing}")

    minus = inverse_neighbor_table(plus)

    def _transport(transport, neighbour_operator):
        return band_matmul(
            band_matmul(transport, neighbour_operator),
            jnp.swapaxes(jnp.conj(transport), -1, -2),
        )

    rows = []
    for idir in range(3):
        lp1 = links[idir]
        kp1 = plus[:, idir]
        kp2 = plus[kp1, idir]
        km1 = minus[:, idir]
        km2 = minus[km1, idir]
        lp2 = band_matmul(lp1, lp1[kp1])
        lm1 = jnp.swapaxes(jnp.conj(lp1[km1]), -1, -2)
        lm2 = band_matmul(
            lm1, jnp.swapaxes(jnp.conj(lp1[km2]), -1, -2))
        tp1 = _transport(lp1, operator[kp1])
        tp2 = _transport(lp2, operator[kp2])
        tm1 = _transport(lm1, operator[km1])
        tm2 = _transport(lm2, operator[km2])
        rows.append(
            (-tp2 + 8.0 * tp1 - 8.0 * tm1 + tm2)
            / (12.0 * float(spacing[idir])))
    return jnp.stack(rows, axis=0)
