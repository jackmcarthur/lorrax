"""Orbit-closed nodal (interpolative) htransform basis on the ISDF centroids.

Every coarse state of the fitted window is sampled once at the centroids,

    Psi[(k, n, s), mu] = psi_{nk,s}(r_mu),

in the full-Bloch gauge of the canonical ``ortho`` r-chunk transform of
``common.psi_G_store``.  Column-pivoted Cholesky of the unit-diagonal centroid
Gram ``G = Psi^H Psi`` (the QRCP column selection of ``Psi``) chooses the nodes
``nu``.  It pivots whole symmetry orbits of centroids
(``symmetry_maps.centroid_source_map_and_wrap`` + ``permutation_orbit_labels``)
and stops when every centroid's relative residual is below ``qr_eps``.  The
basis is then

    ctilde[k, n, (s, nu)] = psi_{nk,s}(r_nu)                  nodal values
    Z[mu, nu]             = (G[mu, nodes] G_nn^+)             interpolation
    psi_s(r)              ~ sum_nu zeta_nu(r) psi_s(r_nu),    zeta_nu(r_mu) = Z[mu, nu].

Each basis function is the interpolation function of one node, so a
symmetry operation permutes basis functions up to the lattice wraps of the
node orbit, and a node-pair tensor takes per-pair Wigner-Seitz images
(``common.ws_images``).  A node is a centroid position carrying every spinor
component; the rank axis is ``(s, nu)`` spin-major plus an exact-null mesh
pad.  ``G_nn^+`` uses the Cholesky tolerance as its relative eigenvalue
cutoff, so internally near-dependent whole-orbit pivots cost nothing.

The fit is ONE pass over the window (each state transformed once); no
full-grid basis, random matrix or state projection exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import math
import time

import jax
from lxkit import ordered_psum
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from common.psi_G_store import build_psi_G_store
from jax import shard_map
from common.sharding_fit import fit_sharding as _fit
from common.wfn_layout import band_sphere_spec
from common.wfn_transforms import (
    FULL_BLOCH_TRANSFORM_SCHEME,
    gflat_to_rchunk_aot_memory,
)
from runtime.padding import spec_divisor


__all__ = [
    "GalerkinBasis",
    "GalerkinBasisMismatch",
    "GalerkinStreamPlan",
    "NodalInterp",
    "fit_galerkin_basis",
    "galerkin_rank_record",
    "iter_galerkin_rchunks",
    "plan_galerkin_stream",
    "read_galerkin_basis",
    "validate_rank_multiplier",
    "write_galerkin_basis",
]


NODE_SELECTION_VERSION = "orbit-pivoted-centroid-gram-v1"
_BASIS_FORMAT = 3
_BASIS_ARRAYS = ("galerkin_ctilde", "galerkin_interp", "galerkin_node_gram_pinv")
_BASIS_META = "galerkin_"


class GalerkinBasisMismatch(ValueError):
    """A complete basis artifact fitted for other inputs or another format.

    Raised only by the provenance/format comparison, so a caller may refit
    on it; a corrupt or incomplete artifact still raises plain ValueError.
    """


@jax.tree_util.register_pytree_node_class
class NodalInterp:
    """The k-independent interpolation operator from nodes to centroids.

    ``values[(s, nu), s', mu] = delta_{s s'} Z[mu, nu]`` with shape
    ``(rank, ns, n_mu)``, so ``psi_s(r_mu) = sum_rho c_rho values[rho, s, mu]``
    for nodal values ``c``.  It carries the geometry every node-pair Fourier
    sum needs: ``node_frac`` ``(n_nodes, 3)`` lattice coordinates of the
    nodes (carried row ``s*n_nodes + nu`` sits at node ``nu``; rows past
    ``ns*n_nodes`` are exact-null pad), the lattice ``(3, 3)`` rows ``a_i``
    in bohr and ``sys_dim``.  The geometry is static pytree data; ``values``
    is the only array leaf.
    """

    def __init__(self, values, node_frac, lattice_bohr, sys_dim):
        self.values = values
        self.node_frac = np.asarray(node_frac, dtype=np.float64)
        self.lattice_bohr = np.asarray(lattice_bohr, dtype=np.float64)
        self.sys_dim = int(sys_dim)

    @property
    def shape(self):
        return self.values.shape

    def tree_flatten(self):
        aux = (tuple(map(tuple, self.node_frac.tolist())),
               tuple(map(tuple, self.lattice_bohr.tolist())), self.sys_dim)
        return (self.values,), aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], np.asarray(aux[0]), np.asarray(aux[1]), aux[2])


@dataclass(frozen=True)
class GalerkinBasis:
    """One fitted nodal basis.

    ``ctilde`` ``(nk, nb, rank)`` holds the nodal values
    ``psi_{nk,s}(r_nu)`` on the ``(s, nu)`` rank axis; ``basis_at_nodes`` is
    the :class:`NodalInterp` onto the centroids.  ``node_gram_pinv``
    ``(n_nodes, n_nodes)`` is ``G_nn^+``, the factor that continues the
    interpolation functions to the full grid (:func:`iter_galerkin_rchunks`).
    ``rank_physical = ns * n_nodes`` excludes the exact-null mesh pad.
    """

    ctilde: jax.Array
    basis_at_nodes: NodalInterp
    rank_physical: int
    band_range: tuple[int, int]
    node_indices: tuple[int, ...]
    node_gram_pinv: jax.Array
    qrcp_eps: float = 1.0e-3
    n_orbits: int = 0
    gram_rank: int = 0
    residual_max: float = 0.0
    pivot_hash: str = ""

    def __post_init__(self) -> None:
        if self.ctilde.ndim != 3:
            raise ValueError(
                f"GalerkinBasis.ctilde must be (nk, nb, rank); got "
                f"{tuple(self.ctilde.shape)}")
        rank = int(self.ctilde.shape[2])
        if tuple(self.basis_at_nodes.shape[:1]) != (rank,) \
                or self.basis_at_nodes.node_frac.shape != (
                    len(self.node_indices), 3):
            raise ValueError(
                "GalerkinBasis rank mismatch: ctilde carries "
                f"{rank}, interpolation operator "
                f"{tuple(self.basis_at_nodes.shape)}, node table "
                f"{self.basis_at_nodes.node_frac.shape}")
        ns = int(self.basis_at_nodes.shape[1])
        n_nodes = len(self.node_indices)
        if int(self.rank_physical) != ns * n_nodes or \
                not 0 < int(self.rank_physical) <= rank:
            raise ValueError(
                f"GalerkinBasis.rank_physical={self.rank_physical} must be "
                f"nspinor*n_nodes={ns}*{n_nodes} inside the carrier {rank}")
        b0, b1 = (int(v) for v in self.band_range)
        if b1 - b0 != int(self.ctilde.shape[1]):
            raise ValueError(
                f"GalerkinBasis band range [{b0},{b1}) has width {b1-b0}, "
                f"but ctilde carries {int(self.ctilde.shape[1])} bands")
        if tuple(self.node_gram_pinv.shape) != (n_nodes, n_nodes):
            raise ValueError(
                "GalerkinBasis.node_gram_pinv must be (n_nodes, n_nodes); "
                f"got {tuple(self.node_gram_pinv.shape)}")

    @property
    def rank_carrier(self) -> int:
        return int(self.ctilde.shape[2])


def validate_rank_multiplier(value, *, name: str = "rank_multiplier") -> float:
    """Validate the node-space ceiling multiplier (``ns*n_nodes <= m*nb``).

    ``0`` is retained as an input-compatibility spelling of the published
    default ``20``.
    """
    try:
        multiplier = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name}={value!r} is not a finite number; use 20 for the "
            "published ceiling or another value >= 1.") from None
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError(f"{name}={value!r} must be finite and >= 0.")
    if multiplier == 0.0:
        return 20.0
    if multiplier < 1.0:
        raise ValueError(
            f"{name}={multiplier:g} would retain fewer directions than bands "
            "at one k. Use the published default 20 or another value >= 1.")
    return multiplier


def galerkin_rank_record(basis: GalerkinBasis, *, meta,
                         rank_multiplier: float) -> dict:
    """The node-selection receipt of a fitted or restored basis (host only)."""
    nk, nb, carrier = (int(n) for n in basis.ctilde.shape)
    physical = int(basis.rank_physical)
    return {
        "method": "orbit_pivoted_centroid_gram",
        "selection_version": NODE_SELECTION_VERSION,
        "stacked_states": nk * nb,
        "centroids": int(basis.basis_at_nodes.shape[2]),
        "nodes": len(basis.node_indices),
        "orbits": int(basis.n_orbits),
        "retained_rank": physical,
        "carried_rank": carrier,
        "null_padding": carrier - physical,
        "gram_rank": int(basis.gram_rank),
        "rank_multiplier": validate_rank_multiplier(
            rank_multiplier, name="htransform_rank_multiplier"),
        "qr_eps": float(basis.qrcp_eps),
        "max_relative_residual": float(basis.residual_max),
        "pivot_hash": basis.pivot_hash,
    }


def _basis_text(value: str) -> np.ndarray:
    return np.frombuffer(str(value).encode(), dtype=np.uint8).astype(np.int32)


def _basis_decode(value) -> str:
    raw = np.asarray(value, dtype=np.int32).reshape(-1)
    if np.any((raw < 0) | (raw > 255)):
        raise ValueError("Galerkin basis text metadata contains a non-byte")
    return bytes(raw.astype(np.uint8)).decode()


def _basis_digest(indices) -> tuple[tuple[int, ...], str]:
    values = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
    shape = tuple(int(n) for n in values.shape)
    digest = hashlib.sha256()
    digest.update(np.asarray(shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return shape, digest.hexdigest()


def _pivot_hash(nodes) -> str:
    return hashlib.sha256(
        np.asarray(nodes, dtype="<i8").tobytes()).hexdigest()


def _basis_provenance(*, wfn, meta, centroid_indices, band_range,
                      bispinor, rank_multiplier, qrcp_eps) -> dict:
    from common.parallel_transport import (
        WFN_FINGERPRINT_SCHEME, wfn_fingerprint)

    b0, b1 = (int(n) for n in band_range)
    shape, centroid_hash = _basis_digest(centroid_indices)
    qrcp_eps = float(qrcp_eps)
    if b1 <= b0:
        raise ValueError("Galerkin basis band window is empty")
    if not 0.0 < qrcp_eps < 1.0:
        raise ValueError("htransform_qr_eps must lie in (0,1)")
    return {
        "band_range": (b0, b1), "nk": int(meta.nk_tot), "nb": b1 - b0,
        "nspinor": int(meta.nspinor),
        "fft_grid": tuple(int(n) for n in meta.fft_grid),
        "kgrid": tuple(int(n) for n in meta.kgrid),
        "bispinor": bool(bispinor), "centroid_shape": shape,
        "centroid_hash": centroid_hash, "wfn_hash": wfn_fingerprint(wfn),
        "wfn_scheme": WFN_FINGERPRINT_SCHEME,
        "transform_scheme": FULL_BLOCH_TRANSFORM_SCHEME,
        "rank_multiplier": validate_rank_multiplier(
            rank_multiplier, name="htransform_rank_multiplier"),
        "qrcp_eps": qrcp_eps, "selection": NODE_SELECTION_VERSION,
    }


def _nodal_interp(Z, *, nspinor: int, carrier: int, node_frac, lattice_bohr,
                  sys_dim: int, mesh_xy: Mesh) -> NodalInterp:
    """``(rank, ns, n_mu)`` spin-block operator from the scalar ``Z`` (n_mu, n_nodes)."""
    n_mu, n_nodes = (int(v) for v in Z.shape)
    rows_layout = _fit(mesh_xy, P(None, None, 'y'), (carrier, nspinor, n_mu),
                       "galerkin.interp(mu-axis)")

    @partial(jax.jit, out_shardings=rows_layout)
    def _blocks(z):
        out = jnp.zeros((carrier, nspinor, n_mu), dtype=jnp.complex128)
        for s in range(nspinor):
            out = out.at[s * n_nodes:(s + 1) * n_nodes, s, :].set(z.T)
        return out

    frac = np.asarray(node_frac, dtype=np.float64).reshape(n_nodes, 3)
    return NodalInterp(_blocks(Z), frac, lattice_bohr, sys_dim)


def _basis_check(basis: GalerkinBasis, provenance: dict) -> None:
    if tuple(basis.band_range) != provenance["band_range"] \
            or tuple(basis.ctilde.shape[:2]) != (
                provenance["nk"], provenance["nb"]) \
            or tuple(basis.basis_at_nodes.shape[1:]) != (
                provenance["nspinor"], provenance["centroid_shape"][0]):
        raise ValueError("Galerkin basis payload disagrees with provenance")
    if basis.qrcp_eps != provenance["qrcp_eps"]:
        raise ValueError("Galerkin basis qr_eps disagrees with provenance")
    nodes = np.asarray(basis.node_indices, dtype="<i8")
    if (np.any(nodes < 0) or np.any(nodes >= provenance["centroid_shape"][0])
            or np.unique(nodes).size != nodes.size):
        raise ValueError("Galerkin node indices are invalid")
    if _pivot_hash(nodes) != basis.pivot_hash:
        raise ValueError("Galerkin node indices/pivot hash disagree")


def _basis_write_meta(io, basis: GalerkinBasis, provenance: dict) -> None:
    integer = {
        "format": [_BASIS_FORMAT], "rank": [basis.rank_physical],
        "band_range": provenance["band_range"],
        "nk_nb_nspinor": [provenance[k] for k in ("nk", "nb", "nspinor")],
        "fft_grid": provenance["fft_grid"], "kgrid": provenance["kgrid"],
        "bispinor": [provenance["bispinor"]],
        "centroid_shape": provenance["centroid_shape"],
        "orbits_gram_rank": [basis.n_orbits, basis.gram_rank],
        "nodes": basis.node_indices,
    }
    for name, value in integer.items():
        io.write_attr(_BASIS_META + name, np.asarray(value, dtype=np.int32))
    io.write_attr(_BASIS_META + "controls", np.asarray([
        provenance["rank_multiplier"], provenance["qrcp_eps"],
        basis.residual_max], dtype=np.float64))
    for name, value in (
            ("centroid_hash", provenance["centroid_hash"]),
            ("wfn_hash", provenance["wfn_hash"]),
            ("wfn_scheme", provenance["wfn_scheme"]),
            ("transform_scheme", provenance["transform_scheme"]),
            ("selection", provenance["selection"]),
            ("pivot_hash", basis.pivot_hash)):
        io.write_attr(_BASIS_META + name, _basis_text(value))
    # SlabIO's write_attr queues a small dataset for the rank-0 reopen after
    # collective close; read_small is its matching collective reader.
    io.write_attr(_BASIS_META + "complete", np.asarray([1], dtype=np.int32))


def _interp_scalar(basis: GalerkinBasis) -> jax.Array:
    """The scalar ``Z`` (n_mu, n_nodes) block of the spin-block operator."""
    n_nodes = len(basis.node_indices)
    return jax.jit(lambda v: v[:n_nodes, 0, :].T,
                   out_shardings=NamedSharding(basis.ctilde.sharding.mesh, P()))(
        basis.basis_at_nodes.values)


def write_galerkin_basis(path, basis: GalerkinBasis, *, wfn, meta,
                         centroid_indices, bispinor, rank_multiplier,
                         qrcp_eps, mesh_xy: Mesh) -> None:
    """Collectively publish one immutable mesh-neutral basis artifact."""
    import os
    from common.collectives import barrier, process_count, process_rank
    from file_io.slab_io import SlabIO

    provenance = _basis_provenance(
        wfn=wfn, meta=meta, centroid_indices=centroid_indices,
        band_range=basis.band_range, bispinor=bispinor,
        rank_multiplier=rank_multiplier, qrcp_eps=qrcp_eps)
    _basis_check(basis, provenance)
    destination = os.path.abspath(os.fspath(path))
    job, step = os.environ.get("SLURM_JOB_ID", "local"), os.environ.get(
        "SLURM_STEP_ID")
    if process_count() > 1 and not step:
        raise RuntimeError("multi-process basis publication requires a shared step id")
    token = f"{job}.{step}" if step else f"{job}.{os.getpid()}"
    staging = destination + ".partial." + token
    if os.path.lexists(destination) or os.path.lexists(staging):
        raise FileExistsError(
            f"immutable Galerkin destination/staging exists: "
            f"{destination} / {staging}")
    barrier("galerkin_basis.staging")
    physical = int(basis.rank_physical)
    n_nodes = len(basis.node_indices)
    values = (basis.ctilde, _interp_scalar(basis), basis.node_gram_pinv)
    shapes = ((basis.ctilde.shape[0], basis.ctilde.shape[1], physical),
              (int(basis.basis_at_nodes.shape[2]), n_nodes),
              (n_nodes, n_nodes))
    with SlabIO(staging, mode="w", mesh=mesh_xy) as io:
        for name, value, shape in zip(_BASIS_ARRAYS, values, shapes):
            io.create_dataset(name, shape=shape, dtype=value.dtype)
            io.write_slab(name, value)
        _basis_write_meta(io, basis, provenance)
    if process_rank() == 0:
        try:
            os.link(staging, destination)  # no-clobber atomic visibility
            os.unlink(staging)
        except OSError:
            pass  # peers must reach the attribution barrier before refusal
    barrier("galerkin_basis.published")
    if os.path.lexists(staging) or not os.path.isfile(destination):
        raise RuntimeError("atomic Galerkin basis publication failed")


def _basis_read_meta(io) -> dict:
    small = lambda name, dtype: np.asarray(io.read_small(
        _BASIS_META + name, dtype=dtype)).reshape(-1)
    if int(small("complete", np.int32)[0]) != 1:
        raise ValueError("Galerkin basis artifact is incomplete")
    version = int(small("format", np.int32)[0])
    if version != _BASIS_FORMAT:
        raise GalerkinBasisMismatch(
            f"Galerkin basis format {version} is unsupported (this build "
            f"reads the nodal format {_BASIS_FORMAT})")
    band = small("band_range", np.int32)
    extents = small("nk_nb_nspinor", np.int32)
    controls = small("controls", np.float64)
    counts = small("orbits_gram_rank", np.int32)
    text = lambda name: _basis_decode(small(name, np.int32))
    return {
        "rank": int(small("rank", np.int32)[0]),
        "band_range": tuple(int(n) for n in band),
        "nk": int(extents[0]), "nb": int(extents[1]),
        "nspinor": int(extents[2]),
        "fft_grid": tuple(int(n) for n in small("fft_grid", np.int32)),
        "kgrid": tuple(int(n) for n in small("kgrid", np.int32)),
        "bispinor": bool(small("bispinor", np.int32)[0]),
        "centroid_shape": tuple(int(n) for n in small(
            "centroid_shape", np.int32)),
        "centroid_hash": text("centroid_hash"), "wfn_hash": text("wfn_hash"),
        "wfn_scheme": text("wfn_scheme"),
        "transform_scheme": text("transform_scheme"),
        "rank_multiplier": float(controls[0]), "qrcp_eps": float(controls[1]),
        "residual_max": float(controls[2]),
        "selection": text("selection"),
        "n_orbits": int(counts[0]), "gram_rank": int(counts[1]),
        "nodes": tuple(int(n) for n in small("nodes", np.int32)),
        "pivot_hash": text("pivot_hash"),
    }


def read_galerkin_basis(path, *, wfn, meta, centroid_indices, band_range,
                        bispinor, rank_multiplier, qrcp_eps,
                        mesh_xy: Mesh, extra_rank_pad: int = 0) -> GalerkinBasis:
    """Collectively validate and read a basis on the caller's target mesh."""
    import os
    from file_io.slab_io import SlabIO
    from common.ws_images import lattice_bohr

    expected = _basis_provenance(
        wfn=wfn, meta=meta, centroid_indices=centroid_indices,
        band_range=band_range, bispinor=bispinor,
        rank_multiplier=rank_multiplier, qrcp_eps=qrcp_eps)
    with SlabIO(os.path.abspath(os.fspath(path)), mode="r", mesh=mesh_xy) as io:
        stored = _basis_read_meta(io)
        mismatches = [key for key in expected if stored.get(key) != expected[key]]
        if mismatches:
            raise GalerkinBasisMismatch(
                "Galerkin basis provenance mismatch: " + ", ".join(mismatches))
        nodes = stored["nodes"]
        carrier = _rank_carrier(stored["rank"], mesh_xy, extra_rank_pad)
        n_mu = stored["centroid_shape"][0]
        shapes = ((stored["nk"], stored["nb"], carrier), (n_mu, len(nodes)),
                  (len(nodes), len(nodes)))
        # SlabIO's registered shape>dataset contract zero-extends the physical
        # logical datasets directly into these mesh-legal carrier shapes.
        arrays = [io.read_slab(name, shape=shape, partition_spec=spec)
                  for name, shape, spec in zip(
                      _BASIS_ARRAYS, shapes,
                      (P(None, None, ("x", "y")), P(), P()))]
    grid = np.asarray(meta.fft_grid, dtype=np.float64)
    node_frac = np.asarray(centroid_indices, dtype=np.float64)[
        np.asarray(nodes)] / grid[None, :]
    basis = GalerkinBasis(
        ctilde=arrays[0],
        basis_at_nodes=_nodal_interp(
            arrays[1], nspinor=stored["nspinor"], carrier=carrier,
            node_frac=node_frac, lattice_bohr=lattice_bohr(wfn),
            sys_dim=int(meta.sys_dim), mesh_xy=mesh_xy),
        rank_physical=stored["rank"], band_range=stored["band_range"],
        node_indices=nodes, node_gram_pinv=arrays[2],
        qrcp_eps=stored["qrcp_eps"], n_orbits=stored["n_orbits"],
        gram_rank=stored["gram_rank"], residual_max=stored["residual_max"],
        pivot_hash=stored["pivot_hash"])
    _basis_check(basis, expected)
    return basis


def _rank_carrier(physical: int, mesh_xy: Mesh, extra_rank_pad: int) -> int:
    from runtime.padding import padded_axis
    if int(extra_rank_pad) < 0:
        raise ValueError("extra_rank_pad must be non-negative")
    return padded_axis(
        int(physical), mesh_xy, name="Galerkin nodal rank carrier",
        specs=((P("x", None), 0), (P(None, "y"), 1),
               (P(None, None, ("x", "y")), 2)),
        extra=int(extra_rank_pad)).carrier


_C16 = np.dtype(np.complex128).itemsize


def _r_stream(n_rtot: int, divisor: int, local_cols: int) -> "GalerkinStreamPlan":
    """Mesh-aligned r schedule with ``local_cols`` r columns per device."""
    return plan_galerkin_stream(
        rank=1, nspinor=1, n_rtot=int(n_rtot), r_mesh_divisor=int(divisor),
        q_tile_budget=max(1, int(local_cols)) * _C16)


def _rows_pass_bytes(geom: dict, *, rows: int, fft_rows: int,
                     local_cols: int, omega_rows: int, resident: float) -> float:
    """Per-device live set of one rows-only pass (bytes).

    ``rows`` G-flat rows are resident, transformed once to a band-sharded
    full-grid slab, and cut into r chunks of ``local_cols`` columns per
    device.  The FFT transient (``fft_rows`` rows per step, each priced by
    the measured canonical transform of one row) and the chunk exchange
    (slab + its band->r reshard, plus the Gaussian rows of the sketch) are
    alternatives; the resident terms add.
    """
    ns, n_r, p = geom["ns"], geom["n_rtot"], geom["p"]
    slab = rows * ns * local_cols * p * _C16
    return (float(resident) + geom["g_index"]
            + rows * ns * (geom["ngkmax"] + n_r) * _C16
            + max(min(int(fft_rows), rows) * geom["row_fft"],
                  2 * slab + omega_rows * ns * local_cols * 8))


def _projection_bytes(geom: dict, *, band_carrier: int, rank: int,
                      local_cols: int, k_tile: int) -> float:
    """Per-device live set of the all-states projection stream (bytes).

    Resident: the r-chunked selected rows ``X`` and the coefficient
    accumulators.  Per band chunk: its full-grid slab, then either the
    ``k_tile``-row FFT transient or one r-chunk exchange.
    """
    ns, n_r, p, nk = geom["ns"], geom["n_rtot"], geom["p"], geom["nk"]
    bpd = int(band_carrier) // p
    n_chunks = geom["n_band_chunks"](band_carrier)
    full_cols = -(-n_r // p)
    resident = (rank * ns * full_cols * _C16
                + nk * (1 + 2 * n_chunks / p) * band_carrier * rank * _C16)
    return (resident + geom["g_index"] + nk * bpd * ns * n_r * _C16
            + max(int(k_tile) * bpd * geom["row_fft"],
                  2 * nk * bpd * ns * local_cols * p * _C16))


def _largest_fit(start: int, fits) -> int:
    """Largest value reached by halving ``start`` for which ``fits`` holds."""
    value = int(start)
    while value >= 1 and not fits(value):
        value //= 2
    return value


def _fit_band_carrier(geom: dict, *, carrier: int, divisor: int, rank: int,
                      capacity: float, log_fn) -> int:
    """Largest band carrier (a ``divisor`` multiple, at most ``carrier``)
    whose projection stream fits beside ``rank`` resident selected rows."""
    carrier = int(carrier)

    def fits(carrier):
        # _plan_basis_passes at rows=1 runs at one r column per device for
        # the projection and the selected-row pass beside its resident rows.
        full_cols = -(-geom["n_rtot"] // geom["p"])
        x_resident = rank * geom["ns"] * full_cols * _C16 + rank * rank * _C16
        return (_projection_bytes(geom, band_carrier=carrier, rank=rank,
                                  local_cols=1, k_tile=1) <= capacity
                and _rows_pass_bytes(geom, rows=1, fft_rows=1, local_cols=1,
                                     omega_rows=0, resident=x_resident) <= capacity)

    while carrier > divisor and not fits(carrier):
        nxt = max(divisor, (carrier // (2 * divisor)) * divisor)
        if nxt >= carrier:
            nxt = carrier - divisor
        log_fn(
            f"  Whole-state planner reduces the canonical WFN band "
            f"carrier {carrier} -> {nxt} at rank {rank}: its one-column "
            f"projection does not fit {capacity/2**30:.2f} GiB/device")
        carrier = nxt
    return carrier


def _whole_state_geometry(*, meta, mesh_xy: Mesh, nk: int, nspinor: int,
                          ngkmax: int, band_divisor: int, band_range,
                          device_pool_limit: float | None):
    """Fixed sizes, the measured one-row transform, and the stage target.

    The one-row price is the compiled canonical G-flat -> full-grid program
    (gather, ``ifftn(norm='ortho')``, Bloch phase, output) including its
    cuFFT workspace, so every FFT batch below is priced by measurement.  The
    target is ``device_pool_limit`` itself; the run budget's headroom
    (``runtime.planner_budget_bytes``) is the only margin.
    """
    n_rtot = int(meta.n_rtot)
    memory = gflat_to_rchunk_aot_memory(
        mesh=mesh_xy, nk=1, band_carrier=int(band_divisor), nspinor=nspinor,
        ngkmax=ngkmax, fft_grid=meta.fft_grid, r_carrier=n_rtot,
        norm="ortho", dtype=jnp.complex128)
    b0, b1 = (int(v) for v in band_range)
    geom = dict(
        p=int(mesh_xy.size), ns=int(nspinor), nk=int(nk),
        ngkmax=int(ngkmax), n_rtot=n_rtot,
        g_index=float(nk * ngkmax * np.dtype(np.int32).itemsize),  # sphere index
        row_fft=float(memory.total),
        n_band_chunks=lambda carrier: -(-(b1 - b0) // int(carrier)))
    if device_pool_limit is None or device_pool_limit <= 0:
        return geom, math.inf, memory
    return geom, float(device_pool_limit), memory



def _sample_pass_bytes(geom: dict, *, band_carrier: int, n_mu: int,
                       panel: int, local_cols: int, k_tile: int) -> float:
    """Per-device live set of the one centroid-sampling stream (bytes).

    The incumbent all-states stream (:func:`_projection_bytes` at zero
    resident rank) plus the band-sharded centroid samples it fills
    (``nk*nb*ns*n_mu/P``) and one replicated ``(nk, band_carrier, ns,
    panel)`` sample panel before its reduce-scatter.
    """
    ns, p, nk = geom["ns"], geom["p"], geom["nk"]
    n_chunks = geom["n_band_chunks"](band_carrier)
    stored = nk * n_chunks * band_carrier * ns * n_mu * _C16 / p
    transient = 2.0 * nk * band_carrier * ns * panel * _C16
    return stored + transient + _projection_bytes(
        geom, band_carrier=band_carrier, rank=0, local_cols=local_cols,
        k_tile=k_tile)


def _plan_sample_pass(geom: dict, *, band_carrier: int, n_mu: int,
                      capacity: float, log_fn):
    """Widest r chunk, then widest centroid panel, then widest k tile."""
    nk, p = geom["nk"], geom["p"]
    full_cols = -(-geom["n_rtot"] // p)
    price = lambda **kw: _sample_pass_bytes(
        geom, band_carrier=band_carrier, n_mu=n_mu, **kw)
    cols = _largest_fit(full_cols, lambda c: price(
        panel=1, local_cols=c, k_tile=1) <= capacity)
    if cols < 1:
        cols = 1
        from common.gpu_utils import warn_over_budget
        warn_over_budget("Galerkin centroid sampling (one column)", price(
            panel=1, local_cols=1, k_tile=1), capacity)
    panel = max(1, _largest_fit(n_mu, lambda m: price(
        panel=m, local_cols=cols, k_tile=1) <= capacity))
    k_tile = max(d for d in range(1, nk + 1) if nk % d == 0 and (
        d == 1 or price(panel=panel, local_cols=cols, k_tile=d) <= capacity))
    live = price(panel=panel, local_cols=cols, k_tile=k_tile)
    return _r_stream(geom["n_rtot"], p, cols), panel, k_tile, live


_SAMPLE_KERNELS: dict = {}
_GRAM_KERNELS: dict = {}
_ZETA_KERNELS: dict = {}


def _sample_layout():
    """Band-sharded layout of the centroid samples (the WFN band layout)."""
    return band_sphere_spec()


def _make_sample_kernel(*, mesh: Mesh, shape, panel: int, mu_carrier: int):
    """Add one (band chunk, r chunk, centroid panel) to the samples.

    ``slab`` is the canonical ``(nk, bc, ns, r)`` chunk on
    ``P(None,None,None,('y','x'))``; ``take`` holds chunk-relative grid
    indices of the panel.  Exactly one device owns each sampled point, so the
    reduce over the r owners is exact; the band axis is scattered back onto
    the sample layout in the same collective.
    """
    key = (id(mesh), tuple(int(v) for v in shape), int(panel), int(mu_carrier))
    fn = _SAMPLE_KERNELS.get(key)
    if fn is not None:
        return fn
    spec = _sample_layout()
    band_axes = spec[1] if isinstance(spec[1], tuple) else (spec[1],)
    other = tuple(a for a in mesh.axis_names if a not in band_axes)
    nx = int(mesh.shape['x'])
    r_spec = P(None, None, None, ('y', 'x'))

    @partial(shard_map, mesh=mesh, in_specs=(r_spec, P(), P(), P(), spec),
             out_specs=spec, check_vma=False)
    def _add(local, take, valid, start, acc):
        owner = jax.lax.axis_index('y') * nx + jax.lax.axis_index('x')
        width = local.shape[-1]
        t = take - owner * width
        owns = valid & (t >= 0) & (t < width)
        vals = jnp.where(owns, local[..., jnp.clip(t, 0, width - 1)], 0.0)
        if other:
            vals = jax.lax.psum(vals, other)  # exact: one owner per point
        vals = jax.lax.psum_scatter(vals, band_axes, scatter_dimension=1,
                                    tiled=True)  # exact: one owner per point
        cur = jax.lax.dynamic_slice_in_dim(acc, start, panel, axis=3)
        return jax.lax.dynamic_update_slice_in_dim(acc, cur + vals, start,
                                                   axis=3)

    rep = NamedSharding(mesh, P())
    layout = NamedSharding(mesh, spec)
    fn = jax.jit(_add, donate_argnums=(4,),
                 in_shardings=(NamedSharding(mesh, r_spec), rep, rep, rep,
                               layout),
                 out_shardings=layout)
    _SAMPLE_KERNELS[key] = fn
    return fn


def _sample_states_at_centroids(*, source, meta, mesh_xy: Mesh,
                                centroid_indices, stream, panel: int,
                                k_tile: int, log_fn):
    """``psi_{nk,s}(r_mu)`` for every fitted state, one IFFT per state.

    Returns one ``(nk, band_carrier, ns, mu_carrier)`` array per WFN band
    chunk on the band-sharded sample layout; pad bands and pad centroids are
    exact zeros.
    """
    flat = np.ravel_multi_index(
        np.asarray(centroid_indices, dtype=np.int64).T,
        tuple(int(n) for n in meta.fft_grid))
    n_mu = int(flat.size)
    mu_carrier = -(-n_mu // panel) * panel
    nk, ns = int(meta.nk_tot), int(meta.nspinor)
    bc = int(source.band_chunk_carrier)
    layout = NamedSharding(mesh_xy, _sample_layout())
    rep = NamedSharding(mesh_xy, P())
    samples, acc, current = [], None, None
    t0 = time.time()
    for bc_range, r_idx, slab in source.iter_bandchunks_rchunks(
            stream.r_chunk_ranges, product_r_spec=P(None, None, None, ('y', 'x')),
            k_tile=k_tile):
        if bc_range != current:
            if acc is not None:
                samples.append(acc)
            acc = jnp.zeros((nk, bc, ns, mu_carrier), jnp.complex128,
                            device=layout)
            current = bc_range
        r0, r1 = stream.r_chunk_ranges[r_idx]
        add = _make_sample_kernel(mesh=mesh_xy, shape=slab.shape, panel=panel,
                                  mu_carrier=mu_carrier)
        for start in range(0, n_mu, panel):
            sel = np.arange(start, start + panel)
            take = np.where(sel < n_mu, flat[np.minimum(sel, n_mu - 1)] - r0, 0)
            valid = (sel < n_mu) & (take >= 0) & (take < r1 - r0)
            acc = add(slab, device_put_process_local(take.astype(np.int32), rep),
                      device_put_process_local(valid, rep),
                      jnp.asarray(start, dtype=jnp.int32), acc)
        del slab
    samples.append(acc)
    jax.block_until_ready(samples)
    log_fn(
        f"  Nodal fit: sampled {len(source.band_chunk_ranges)} band chunk(s) "
        f"x {len(stream.r_chunk_ranges)} r chunk(s) at {n_mu} centroids "
        f"(panel {panel}, k_tile {k_tile}): {time.time()-t0:.2f}s")
    return samples, n_mu


def _centroid_gram(samples, mesh_xy: Mesh, n_mu: int):
    """``G = Psi^H Psi`` over every (k, n, s) row, replicated ``(n_mu, n_mu)``."""
    spec = _sample_layout()
    key = (id(mesh_xy), tuple(int(v) for v in samples[0].shape), int(n_mu))
    fn = _GRAM_KERNELS.get(key)
    if fn is None:
        @partial(shard_map, mesh=mesh_xy, in_specs=(spec, P()), out_specs=P(),
                 check_vma=False)
        def _acc(local, gram):
            a = local[..., :n_mu]
            return gram + ordered_psum(
                jnp.einsum('kbsm,kbsn->mn', jnp.conj(a), a, optimize=True),
                tuple(mesh_xy.axis_names))

        rep = NamedSharding(mesh_xy, P())
        fn = _GRAM_KERNELS[key] = jax.jit(
            _acc, donate_argnums=(1,),
            in_shardings=(NamedSharding(mesh_xy, spec), rep), out_shardings=rep)
    gram = jnp.zeros((n_mu, n_mu), jnp.complex128,
                     device=NamedSharding(mesh_xy, P()))
    for chunk in samples:
        gram = fn(chunk, gram)
    return gram


def _orbit_pivoted_cholesky(gram, labels, tol: float, max_nodes: int):
    """Column-pivoted Cholesky of the unit-diagonal Gram, one orbit per step.

    The orbit holding the largest residual diagonal enters whole; its
    members are orthogonalized together (eigen-decomposition of their
    residual Gram, directions below ``tol`` dropped), so no member order is
    preferred.  Stops when every residual is ``<= tol`` (``qr_eps**2``, the
    QRCP rule on normalized columns).  Returns ``(nodes, residual,
    n_dependent)``; ``nodes`` is orbit-closed by construction.
    """
    g = np.asarray(gram, dtype=np.complex128)
    scale = np.sqrt(np.maximum(np.real(np.diag(g)), 1e-300))
    g = g / scale[:, None] / scale[None, :]
    labels = np.asarray(labels, dtype=np.int64)
    n = g.shape[0]
    d = np.ones(n)
    taken = np.zeros(int(labels.max()) + 1, dtype=bool)
    basis = np.zeros((n, 0), dtype=np.complex128)
    nodes, dependent = [], 0
    while True:
        open_ = ~taken[labels]
        if not np.any(open_ & (d > tol)):
            break
        mu = int(np.argmax(np.where(open_, d, -np.inf)))
        members = np.flatnonzero(labels == labels[mu])
        if len(nodes) + members.size > max_nodes:
            raise ValueError(
                f"fit_galerkin_basis: the node selection saturated the "
                f"ceiling of {max_nodes} nodes with residual {d[mu]:.3e} > "
                f"{tol:.3e} still open. Increase htransform_rank_multiplier "
                "or increase htransform_qr_eps.")
        resid = g[:, members] - basis @ np.conj(basis[members]).T
        w, U = np.linalg.eigh(0.5 * (resid[members] + np.conj(resid[members]).T))
        keep = w > tol
        dependent += int(members.size - np.count_nonzero(keep))
        new = resid @ (U[:, keep] / np.sqrt(w[keep])[None, :])
        basis = np.concatenate([basis, new], axis=1)
        d = np.maximum(d - np.sum(np.abs(new) ** 2, axis=1), 0.0)
        d[members] = 0.0
        taken[labels[mu]] = True
        nodes.extend(int(m) for m in members)
    return np.asarray(nodes, dtype=np.int64), d, dependent


def _interpolation_from_gram(gram, nodes, tol: float):
    """``G_nn^+`` (unit-diagonal relative cutoff ``tol``) and ``Z = (G_nn^+ G_n:)^T``."""
    g = np.asarray(gram, dtype=np.complex128)
    gnn = g[np.ix_(nodes, nodes)]
    scale = np.sqrt(np.real(np.diag(gnn)))
    w, U = np.linalg.eigh(gnn / scale[:, None] / scale[None, :])
    keep = w > tol * w.max()
    pinv = (U[:, keep] / w[keep][None, :]) @ np.conj(U[:, keep]).T
    pinv = pinv / scale[:, None] / scale[None, :]
    Z = (pinv @ g[nodes, :]).T
    return Z, pinv, int(np.count_nonzero(keep)), float(w.max() / w[keep].min())


def _nodal_values(samples, nodes, *, widths, carrier: int, mesh_xy: Mesh):
    """``ctilde[k, n, (s, nu)]`` on all-P rank shards from the band samples."""
    nodes = jnp.asarray(nodes, dtype=jnp.int32)
    out = NamedSharding(mesh_xy, P(None, None, ("x", "y")))

    @partial(jax.jit, out_shardings=out)
    def _assemble(*chunks):
        rows = jnp.concatenate([c[:, :w][..., nodes] for c, w in
                                zip(chunks, widths)], axis=1)
        nk, nb, ns, nn = rows.shape
        rows = rows.reshape(nk, nb, ns * nn)
        return jnp.pad(rows, ((0, 0), (0, 0), (0, carrier - ns * nn)))

    return _assemble(*samples)


def fit_galerkin_basis(
        wfn, sym, meta, centroid_indices, mesh_xy: Mesh,
        band_range: tuple[int, int], *,
        log_fn=None,
        band_chunk_size: int = 64,
        bispinor: bool = False,
        rank_multiplier: float = 20.0,
        qr_eps: float = 1.0e-3,
        device_pool_limit: float | None,
        extra_rank_pad: int = 0,
        progress_fn=None,
        rank_record_fn=None,
) -> GalerkinBasis:
    """Fit the orbit-closed nodal htransform basis (module docstring).

    ``qr_eps`` is the sole tolerance: every centroid's relative residual
    after node selection is below it, and ``G_nn^+`` drops eigen-directions
    below ``qr_eps**2`` relative.  ``rank_multiplier`` caps the node space at
    ``ceil(m*nb)`` directions.  Requires an orbit-closed centroid set.
    """
    del progress_fn
    from common.ws_images import lattice_bohr
    from runtime.padding import padded_axis
    from symmetry_maps import (centroid_source_map_and_wrap,
                               permutation_orbit_labels)
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    b_start, b_end = (int(v) for v in band_range)
    nb = b_end - b_start
    nk = int(meta.nk_tot)
    nspinor = int(meta.nspinor)
    idx = np.asarray(centroid_indices, dtype=np.int64)
    n_mu = int(idx.shape[0])
    if nb <= 0:
        raise ValueError(
            f"fit_galerkin_basis: empty band range [{b_start},{b_end})")
    if not (0.0 < float(qr_eps) < 1.0):
        raise ValueError(
            f"fit_galerkin_basis: qr_eps={qr_eps!r} must lie in (0,1)")
    multiplier = validate_rank_multiplier(
        rank_multiplier, name="htransform_rank_multiplier")
    max_nodes = min(int(math.ceil(multiplier * nb)) // nspinor, n_mu)

    sym_matrices = np.asarray(sym.sym_matrices)
    try:
        perm, _wrap = centroid_source_map_and_wrap(
            idx, sym_matrices,
            np.asarray(sym.translations)[:sym_matrices.shape[0]],
            tuple(int(n) for n in meta.fft_grid))
    except RuntimeError as exc:
        raise ValueError(
            "fit_galerkin_basis: the centroid set is not orbit-closed under "
            f"the WFN's {sym_matrices.shape[0]} spatial operations, and the "
            "nodal htransform basis requires closure (it pivots whole "
            "orbits).  Regenerate the centroids with centroid.kmeans_cli in "
            f"orbit-aware mode.  ({exc})") from None
    labels = permutation_orbit_labels(perm)

    p_band = padded_axis(
        1, mesh_xy, name="Galerkin band divisor",
        spec=band_sphere_spec(), axis=1).divisor
    band_axis = padded_axis(
        max(p_band, min(int(band_chunk_size), nb)), mesh_xy,
        name="Galerkin band-chunk carrier", spec=band_sphere_spec(), axis=1)
    geom, capacity, _row_memory = _whole_state_geometry(
        meta=meta, mesh_xy=mesh_xy, nk=nk, nspinor=nspinor,
        ngkmax=int(wfn.ngkmax), band_divisor=band_axis.divisor,
        band_range=(b_start, b_end), device_pool_limit=device_pool_limit)
    bc_carrier = _fit_band_carrier(
        geom, carrier=band_axis.carrier, divisor=band_axis.divisor, rank=0,
        capacity=capacity, log_fn=log_fn)
    stream, panel, k_tile, live = _plan_sample_pass(
        geom, band_carrier=bc_carrier, n_mu=n_mu, capacity=capacity,
        log_fn=log_fn)
    log_fn(
        f"  Nodal fit: states={nk * nb} ({nk} k * {nb} bands * {nspinor} "
        f"spinor), centroids={n_mu} in {int(labels.max()) + 1} orbit(s) "
        f"under {sym_matrices.shape[0]} operations, qr_eps={float(qr_eps):.3e},"
        f" node ceiling {max_nodes}, band carrier {bc_carrier}, live "
        f"{live/2**30:.2f} GiB/device")

    with build_psi_G_store(
            wfn=wfn, mesh_xy=mesh_xy, meta=meta,
            band_chunk_ranges=tuple(
                (b0, min(b0 + bc_carrier, b_end))
                for b0 in range(b_start, b_end, bc_carrier)),
            bispinor=bispinor, band_pad_to=bc_carrier) as source:
        samples, _ = _sample_states_at_centroids(
            source=source, meta=meta, mesh_xy=mesh_xy, centroid_indices=idx,
            stream=stream, panel=panel, k_tile=k_tile, log_fn=log_fn)
        widths = tuple(int(hi) - int(lo) for lo, hi in source.band_chunk_ranges)
    gram = np.asarray(jax.device_get(_centroid_gram(samples, mesh_xy, n_mu)))
    tol = float(qr_eps) ** 2
    nodes, residual, dependent = _orbit_pivoted_cholesky(
        gram, labels, tol, max_nodes)
    Z, pinv, gram_rank, cond = _interpolation_from_gram(gram, nodes, tol)
    n_orbits = int(np.unique(labels[nodes]).size)
    residual_max = float(np.sqrt(residual.max()))
    rank_phys = nspinor * int(nodes.size)
    carrier = _rank_carrier(rank_phys, mesh_xy, extra_rank_pad)
    log_fn(
        f"  [nodes] {nodes.size} node(s) in {n_orbits} orbit(s), rank "
        f"{rank_phys} (+{carrier - rank_phys} exact-null pad); G_nn rank "
        f"{gram_rank}, cond {cond:.3e}, dropped {nodes.size - gram_rank} "
        f"({dependent} orbit-internal); max centroid residual "
        f"{residual_max:.3e} (qr_eps {float(qr_eps):.1e})")

    ctilde = _nodal_values(samples, nodes, widths=widths, carrier=carrier,
                           mesh_xy=mesh_xy)
    del samples
    rep = NamedSharding(mesh_xy, P())
    basis = GalerkinBasis(
        ctilde=ctilde,
        basis_at_nodes=_nodal_interp(
            device_put_process_local(Z, rep), nspinor=nspinor,
            carrier=carrier,
            node_frac=idx[nodes].astype(np.float64)
            / np.asarray(meta.fft_grid, dtype=np.float64)[None, :],
            lattice_bohr=lattice_bohr(wfn), sys_dim=int(meta.sys_dim),
            mesh_xy=mesh_xy),
        rank_physical=rank_phys,
        band_range=(b_start, b_end),
        node_indices=tuple(int(v) for v in nodes),
        node_gram_pinv=device_put_process_local(pinv, rep),
        qrcp_eps=float(qr_eps),
        n_orbits=n_orbits,
        gram_rank=gram_rank,
        residual_max=residual_max,
        pivot_hash=_pivot_hash(nodes),
    )
    if rank_record_fn is not None:
        rank_record_fn(galerkin_rank_record(
            basis, meta=meta, rank_multiplier=multiplier))
    return basis


def _make_zeta_kernel(*, mesh: Mesh, phi_shape, n_nodes: int, rank: int,
                      nspinor: int, r_carrier: int):
    """One band chunk of ``sum_{k,n,s} conj(psi(r_nu)) psi(r)`` on r shards."""
    key = (id(mesh), tuple(phi_shape), int(n_nodes), int(rank), int(nspinor),
           int(r_carrier))
    fn = _ZETA_KERNELS.get(key)
    if fn is not None:
        return fn
    psi_spec = P(None, None, None, ('y', 'x'))
    acc_spec = P(None, ('y', 'x'))

    @partial(shard_map, mesh=mesh, in_specs=(psi_spec, P(), acc_spec),
             out_specs=acc_spec, check_vma=False)
    def _acc(psi, phi, acc):
        return acc + jnp.einsum('kbsr,kbsn->nr', psi, jnp.conj(phi),
                                optimize=True)

    fn = jax.jit(_acc, donate_argnums=(2,),
                 in_shardings=(NamedSharding(mesh, psi_spec),
                               NamedSharding(mesh, P()),
                               NamedSharding(mesh, acc_spec)),
                 out_shardings=NamedSharding(mesh, acc_spec))
    _ZETA_KERNELS[key] = fn
    return fn


def iter_galerkin_rchunks(
        source, basis: GalerkinBasis, meta, mesh_xy: Mesh, *,
        r_chunk_ranges, retained_band_range: tuple[int, int]):
    """Yield bounded basis rows and requested WFN rows together.

    The continuation of the node interpolation functions to the grid,

    ``zeta_nu(r) = sum_nu' G_nn^+[nu, nu'] sum_{k,n,s} conj(psi_{nk,s}(r_nu')) psi_{nk,s}(r)``,

    is accumulated from the caller-owned canonical :class:`PsiGStore` during
    the same WFN/FFT pass that retains ``retained_band_range`` for the
    consumer.  Yields ``(r0, r1, B_chunk, psi_parts)`` with ``B_chunk``
    ``(rank, nspinor, r_carrier)`` the spin-block rows
    ``B[(s, nu), s', r] = delta_{s s'} zeta_nu(r)``, so
    ``psi(r) = sum_rho c_rho B[rho, :, r]`` for nodal values ``c``, and
    ``psi_parts`` the ordered ``((band_lo, band_hi), psi_chunk)`` pairs of the
    retained range.
    """
    b_start, b_end = (int(v) for v in basis.band_range)
    keep_start, keep_end = (int(v) for v in retained_band_range)
    if not b_start <= keep_start < keep_end <= b_end:
        raise ValueError(
            "iter_galerkin_rchunks: retained band range "
            f"[{keep_start},{keep_end}) escapes basis range "
            f"[{b_start},{b_end})")
    if (not source.band_chunk_ranges
            or int(source.band_chunk_ranges[0][0]) != b_start
            or int(source.band_chunk_ranges[-1][1]) != b_end):
        raise ValueError(
            "iter_galerkin_rchunks: PsiGStore must span the fitted basis "
            f"range [{b_start},{b_end}); got {source.band_chunk_ranges}")
    rank = int(basis.rank_carrier)
    nspinor = int(meta.nspinor)
    n_nodes = len(basis.node_indices)
    bc = int(source.band_chunk_carrier)
    rep = NamedSharding(mesh_xy, P())
    row_layout = NamedSharding(mesh_xy, P(None, None, ('y', 'x')))
    psi_layout = NamedSharding(mesh_xy, P(None, None, None, ('y', 'x')))
    r_divisor = spec_divisor(mesh_xy, row_layout.spec, axis=2)

    @partial(jax.jit, static_argnames=("lo", "hi"), out_shardings=rep)
    def _phi(ctilde, lo, hi):
        rows = ctilde[:, lo:hi, :nspinor * n_nodes].reshape(
            ctilde.shape[0], hi - lo, nspinor, n_nodes)
        return jnp.pad(rows, ((0, 0), (0, bc - (hi - lo)), (0, 0), (0, 0)))

    phis = [_phi(basis.ctilde, int(lo) - b_start, int(hi) - b_start)
            for lo, hi in source.band_chunk_ranges]

    @partial(jax.jit, out_shardings=row_layout)
    def _rows(pinv, acc):
        zeta = pinv @ acc
        out = jnp.zeros((rank, nspinor, zeta.shape[-1]), dtype=zeta.dtype)
        for s in range(nspinor):
            out = out.at[s * n_nodes:(s + 1) * n_nodes, s, :].set(zeta)
        return out

    for r0, r1 in r_chunk_ranges:
        r0, r1 = int(r0), int(r1)
        if not 0 <= r0 < r1 <= int(meta.n_rtot):
            raise ValueError(
                f"iter_galerkin_rchunks: r slab [{r0},{r1}) escapes "
                f"[0,{int(meta.n_rtot)})")
        from runtime.padding import padded_axis
        r_carrier = padded_axis(
            r1 - r0, r_divisor,
            name="Galerkin iteration real-space carrier").carrier
        acc = jnp.zeros((n_nodes, r_carrier), jnp.complex128,
                        device=NamedSharding(mesh_xy, P(None, ('y', 'x'))))
        retained = []
        for bc_idx, (bc_range, psi_bc) in enumerate(
                source.iter_rchunk_bandwise(
                    r0, r1, product_r_spec=psi_layout.spec)):
            acc = _make_zeta_kernel(
                mesh=mesh_xy, phi_shape=phis[bc_idx].shape, n_nodes=n_nodes,
                rank=rank, nspinor=nspinor, r_carrier=r_carrier)(
                    psi_bc, phis[bc_idx], acc)
            lo = max(int(bc_range[0]), keep_start)
            hi = min(int(bc_range[1]), keep_end)
            if lo < hi:
                offset = lo - int(bc_range[0])
                retained.append(
                    ((lo, hi), psi_bc[:, offset:offset + (hi - lo)]))
            else:
                del psi_bc
        basis_chunk = _rows(basis.node_gram_pinv, acc)
        del acc
        if sum(hi - lo for (lo, hi), _ in retained) != keep_end - keep_start:
            raise RuntimeError(
                "iter_galerkin_rchunks: retained WFN pieces do not cover "
                f"[{keep_start},{keep_end})")
        yield r0, r1, basis_chunk, tuple(retained)


@dataclass(frozen=True)
class GalerkinStreamPlan:
    """The mesh-aligned outer-r schedule selected after rank is known."""

    r_chunk_ranges: tuple[tuple[int, int], ...]
    max_r_logical: int
    max_r_carrier: int
    q_tile_local_bytes: int


def plan_galerkin_stream(*, rank: int, nspinor: int, n_rtot: int,
                         r_mesh_divisor: int,
                         q_tile_budget: int) -> GalerkinStreamPlan:
    """Choose the incumbent Q-budget-bounded, mesh-aligned r schedule."""
    q_bytes_per_local_r = (
        rank * nspinor * np.dtype(np.complex128).itemsize)
    r_local_cap = max(1, q_tile_budget // q_bytes_per_local_r)
    r_chunk = min(n_rtot, r_local_cap * r_mesh_divisor)
    if r_chunk < n_rtot:
        r_chunk = max(
            r_mesh_divisor,
            (r_chunk // r_mesh_divisor) * r_mesh_divisor,
        )
    r_chunk_ranges = tuple(
        (r0, min(r0 + r_chunk, n_rtot))
        for r0 in range(0, n_rtot, r_chunk)
    )
    max_r_logical = max(r1 - r0 for r0, r1 in r_chunk_ranges)
    from runtime.padding import padded_axis
    max_r_carrier = padded_axis(
        max_r_logical, r_mesh_divisor,
        name="Galerkin stream real-space carrier").carrier
    q_tile_local_bytes = (
        rank * nspinor * (max_r_carrier // r_mesh_divisor)
        * np.dtype(np.complex128).itemsize
    )
    return GalerkinStreamPlan(
        r_chunk_ranges=r_chunk_ranges,
        max_r_logical=max_r_logical,
        max_r_carrier=max_r_carrier,
        q_tile_local_bytes=q_tile_local_bytes,
    )

