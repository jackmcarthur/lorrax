"""Orbit-closed nodal htransform basis.

The nodes are an orbit-closed, charge-density-weighted k-means set (the
centroid machinery of ``centroid.kmeans_cli``): the run's own ISDF centroids
when there are at most ``KMEANS_ABOVE x NODES_PER_FITTED_BAND`` per fitted
band, else a smaller selection at ``NODES_PER_FITTED_BAND`` per band
(``bandstructure.fh_interp.htransform_nodes``, cached beside the basis
artifact).  Every coarse state of the fitted window
is sampled there once, through the GW centroid-face path
(``common.wfn_transforms.load_centroids_band_chunked``):

    Phi_k[(s, nu), n] = psi_{nk,s}(r_nu)                       nodal values.

Nodal values carry no metric, so the basis adds the diagonal quadrature
metric ``W = diag(w_nu)`` of the nodes: one non-negative weight per node
orbit, the least-squares solution of ``Phi_k^H W Phi_k = 1`` over all coarse
k (NNLS on the orbit-folded normal equations).  ``W`` is invariant under the
node permutation of every crystal operation.  A per-k Loewdin step makes the
coefficients exactly orthonormal,

    G_k = Phi_k^H W Phi_k,   C_k = W^{1/2} Phi_k G_k^{-1/2},   C_k^H C_k = 1,

and ``ctilde[k, n, (s, nu)] = C_k[(s, nu), n]`` is used exactly like an
orthonormal coefficient array (``C f C^H``).  ``C f C^H`` is unchanged by
window unitaries that commute with ``f`` and ``G_k`` is gauge-covariant, so a
node-pair kernel with per-pair Wigner-Seitz images (``common.ws_images``) is
exactly covariant.  The fit logs ``eps = max_k ||G_k - 1||_2``.  A
zero-weight node's row of ``C`` is exactly zero, so it is dropped from the
rank axis (whole orbits, since ``W`` is orbit-tied); the ISDF centroids stay
the back-map targets.

:class:`NodalInterp` carries the nodal values ``Phi``, the node geometry and,
when the nodes are not the ISDF centroids, the same states sampled at the
centroids (the back-map TARGETS, ``Phi^T``).  The BSE recovers ``psi(r_mu)``
at the centroids at a fine q from the back-map kernel ``Phi^T (-f) C^H``
(``bandstructure.bse_setup``).  :func:`iter_galerkin_rchunks` continues the
coefficient space to the full grid (least-squares rows) for the refit
consumers only.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax import shard_map
from common.collectives import device_put_process_local, gather_to_host
from common.log import get_logger, message
from common.wfn_transforms import (
    FULL_BLOCH_TRANSFORM_SCHEME,
    load_centroids_band_chunked,
)
from runtime.padding import spec_divisor

log = get_logger(__name__)


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
    "write_galerkin_basis",
]


NODE_SELECTION_VERSION = "kmeans-nodes-orbit-nnls-loewdin-v3"

#: Metric quality above which the fit warns.  Si 4^3 (36 fitted spinor
#: bands, orbit-closed k-means, fit window 0:36): eps 0.37 / 0.12 / 1.6e-2 /
#: 1.9e-3 / 1.0e-3 at 312 / 432 / 588 / 780 / 1100 nodes; off-grid DFT max
#: error 62.7 meV at 312 nodes against 27.4 at 780 (MAE 8.03 / 6.76): the
#: max doubles between eps 0.1 and 0.4.  MoS2 9x9 at 1198 centroids reads
#: eps 1.7e-2 with the same error as the 1183-node k-means set (4.6e-2).
NODE_METRIC_EPS_WARN = 1.0e-1
#: Node count per fitted band that reached eps <= 1e-2 on Si 4^3 (780/36).
NODES_PER_FITTED_BAND = 22
#: The ISDF centroids are kept as the nodes up to this factor above
#: NODES_PER_FITTED_BAND x fitted bands (a k-means set that drops only a few
#: percent of the nodes saves nothing and worsens eps: MoS2 1198 -> 1183).
KMEANS_ABOVE = 1.25
_BASIS_FORMAT = 6
_BASIS_ARRAYS = ("galerkin_ctilde", "galerkin_nodal", "galerkin_loewdin",
                 "galerkin_target")
_BASIS_META = "galerkin_"


class GalerkinBasisMismatch(ValueError):
    """A complete basis artifact fitted for other inputs or another format.

    Raised only by the provenance/format comparison, so a caller may refit
    on it; a corrupt or incomplete artifact still raises plain ValueError.
    """


@jax.tree_util.register_pytree_node_class
class NodalInterp:
    """Nodal values of the fitted states, the node geometry and the targets.

    ``nodal`` ``(nk, nb, rank)`` holds ``Phi_k[(s, nu), n] = psi_{nk,s}(r_nu)``
    on the carried ``(s, nu)`` rank axis (rows past ``ns*n_nodes`` are
    exact-null pad).  The geometry every node-pair Fourier sum needs is
    static pytree data: ``node_frac`` ``(n_nodes, 3)`` lattice coordinates
    (carried row ``s*n_nodes + nu`` sits at node ``nu``), the lattice
    ``(3, 3)`` rows ``a_i`` in bohr, ``sys_dim`` and ``nspinor``.

    The back-map TARGETS are the ISDF centroids, where the BSE reads
    ``psi``: ``target`` ``(nk, nb, t_carrier)`` holds the same states there
    on the ``(s, mu)`` axis and ``target_frac`` ``(n_mu, 3)`` their
    positions; both ``None`` when the nodes are the centroids (then the
    targets are the nodes).  ``shape`` is ``(rank, nspinor, n_mu)``, the
    extents the BSE consumers read.
    """

    def __init__(self, nodal, node_frac, lattice_bohr, sys_dim, nspinor,
                 target=None, target_frac=None):
        self.nodal = nodal
        self.node_frac = np.asarray(node_frac, dtype=np.float64)
        self.lattice_bohr = np.asarray(lattice_bohr, dtype=np.float64)
        self.sys_dim = int(sys_dim)
        self.nspinor = int(nspinor)
        if (target is None) != (target_frac is None):
            raise ValueError("NodalInterp: target and target_frac go together")
        self.target = target
        self.target_frac = (None if target_frac is None else
                            np.asarray(target_frac, dtype=np.float64))

    @property
    def n_nodes(self) -> int:
        return int(self.node_frac.shape[0])

    @property
    def targets_are_nodes(self) -> bool:
        return self.target is None

    @property
    def target_values(self):
        """``(nk, nb, t_carrier)`` states at the targets (the nodes' own
        nodal values when the targets are the nodes)."""
        return self.nodal if self.target is None else self.target

    @property
    def target_positions(self) -> np.ndarray:
        return self.node_frac if self.target is None else self.target_frac

    @property
    def shape(self):
        return (int(self.nodal.shape[2]), self.nspinor,
                int(self.target_positions.shape[0]))

    def tree_flatten(self):
        tf = (None if self.target_frac is None else
              tuple(map(tuple, self.target_frac.tolist())))
        aux = (tuple(map(tuple, self.node_frac.tolist())),
               tuple(map(tuple, self.lattice_bohr.tolist())), self.sys_dim,
               self.nspinor, tf)
        return (self.nodal, self.target), aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], np.asarray(aux[0]), np.asarray(aux[1]),
                   aux[2], aux[3], target=children[1],
                   target_frac=None if aux[4] is None else np.asarray(aux[4]))


@dataclass(frozen=True)
class GalerkinBasis:
    """One fitted nodal basis.

    ``ctilde`` ``(nk, nb, rank)`` holds the orthonormal coefficients
    ``C_k`` (module docstring) on the ``(s, nu)`` rank axis;
    ``basis_at_nodes`` is the :class:`NodalInterp` (nodal values and node
    geometry); ``loewdin`` ``(nk, nb, nb)`` is ``G_k^{-1/2}``.  ``node_weights``
    are the metric weights ``w_nu``, ``metric_eps = max_k ||G_k - 1||_2``.
    ``rank_physical = ns * n_nodes`` excludes the exact-null mesh pad.
    ``node_indices`` are the nodes' FFT-grid indices (zero-weight nodes
    dropped); ``requested_nodes`` the node set the fit was asked for, when
    it differs (the artifact's provenance).  ``refit_rcond`` (deck
    key ``htransform_qr_eps``) is used only by :func:`iter_galerkin_rchunks`,
    the relative cutoff of its node-Gram pseudo-inverse.  ``fit_seconds`` is
    the fit wall.
    """

    ctilde: jax.Array
    basis_at_nodes: NodalInterp
    rank_physical: int
    band_range: tuple[int, int]
    node_indices: tuple
    loewdin: jax.Array = None
    node_weights: tuple = ()
    metric_eps: float = 0.0
    refit_rcond: float = 1.0e-3
    n_orbits: int = 0
    fit_seconds: float = 0.0
    requested_nodes: tuple = ()

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

    @property
    def rank_carrier(self) -> int:
        return int(self.ctilde.shape[2])


def galerkin_rank_record(basis: GalerkinBasis, *, meta) -> dict:
    """The node receipt of a fitted or restored basis (host only)."""
    nk, nb, carrier = (int(n) for n in basis.ctilde.shape)
    physical = int(basis.rank_physical)
    return {
        "method": "kmeans_nodes",
        "selection_version": NODE_SELECTION_VERSION,
        "stacked_states": nk * nb,
        "centroids": int(basis.basis_at_nodes.shape[2]),
        "nodes": len(basis.node_indices),
        "orbits": int(basis.n_orbits),
        "retained_rank": physical,
        "carried_rank": carrier,
        "null_padding": carrier - physical,
        "fit_seconds": float(basis.fit_seconds),
        "metric_eps": float(basis.metric_eps),
        "zero_weight_nodes": (len(basis.requested_nodes) - len(basis.node_indices)
                              if basis.requested_nodes else 0),
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


def _basis_provenance(*, wfn, meta, centroid_indices, node_indices,
                      band_range, bispinor, refit_rcond) -> dict:
    from common.parallel_transport import (
        WFN_FINGERPRINT_SCHEME, wfn_fingerprint)

    b0, b1 = (int(n) for n in band_range)
    shape, centroid_hash = _basis_digest(centroid_indices)
    node_shape, node_hash = _basis_digest(node_indices)
    refit_rcond = float(refit_rcond)
    if b1 <= b0:
        raise ValueError("Galerkin basis band window is empty")
    if not 0.0 < refit_rcond < 1.0:
        raise ValueError("htransform_qr_eps must lie in (0,1)")
    return {
        "band_range": (b0, b1), "nk": int(meta.nk_tot), "nb": b1 - b0,
        "nspinor": int(meta.nspinor),
        "fft_grid": tuple(int(n) for n in meta.fft_grid),
        "kgrid": tuple(int(n) for n in meta.kgrid),
        "bispinor": bool(bispinor), "centroid_shape": shape,
        "centroid_hash": centroid_hash, "node_shape": node_shape,
        "node_hash": node_hash, "wfn_hash": wfn_fingerprint(wfn),
        "wfn_scheme": WFN_FINGERPRINT_SCHEME,
        "transform_scheme": FULL_BLOCH_TRANSFORM_SCHEME,
        "refit_rcond": refit_rcond, "selection": NODE_SELECTION_VERSION,
    }


def _basis_check(basis: GalerkinBasis, provenance: dict) -> None:
    if tuple(basis.band_range) != provenance["band_range"] \
            or tuple(basis.ctilde.shape[:2]) != (
                provenance["nk"], provenance["nb"]) \
            or tuple(basis.basis_at_nodes.shape[1:]) != (
                provenance["nspinor"], provenance["centroid_shape"][0]):
        raise ValueError("Galerkin basis payload disagrees with provenance")
    if basis.refit_rcond != provenance["refit_rcond"]:
        raise ValueError("Galerkin basis qr_eps disagrees with provenance")
    if len(basis.requested_nodes or basis.node_indices) != \
            provenance["node_shape"][0]:
        raise ValueError("Galerkin basis node count disagrees with provenance")


def _basis_write_meta(io, basis: GalerkinBasis, provenance: dict) -> None:
    integer = {
        "format": [_BASIS_FORMAT], "rank": [basis.rank_physical],
        "band_range": provenance["band_range"],
        "nk_nb_nspinor": [provenance[k] for k in ("nk", "nb", "nspinor")],
        "fft_grid": provenance["fft_grid"], "kgrid": provenance["kgrid"],
        "bispinor": [provenance["bispinor"]],
        "centroid_shape": provenance["centroid_shape"],
        "node_shape": provenance["node_shape"],
        "targets": [int(not basis.basis_at_nodes.targets_are_nodes)],
        "orbits": [basis.n_orbits],
    }
    for name, value in integer.items():
        io.write_attr(_BASIS_META + name, np.asarray(value, dtype=np.int32))
    io.write_attr(_BASIS_META + "controls", np.asarray([
        provenance["refit_rcond"], basis.fit_seconds, basis.metric_eps], dtype=np.float64))
    io.write_attr(_BASIS_META + "node_weights",
                  np.asarray(basis.node_weights, dtype=np.float64))
    kept = set(basis.node_indices)
    io.write_attr(_BASIS_META + "node_keep", np.asarray(
        [int(row in kept) for row in (basis.requested_nodes or basis.node_indices)],
        dtype=np.int32))
    for name, value in (
            ("centroid_hash", provenance["centroid_hash"]),
            ("node_hash", provenance["node_hash"]),
            ("wfn_hash", provenance["wfn_hash"]),
            ("wfn_scheme", provenance["wfn_scheme"]),
            ("transform_scheme", provenance["transform_scheme"]),
            ("selection", provenance["selection"])):
        io.write_attr(_BASIS_META + name, _basis_text(value))
    # SlabIO's write_attr queues a small dataset for the rank-0 reopen after
    # collective close; read_small is its matching collective reader.
    io.write_attr(_BASIS_META + "complete", np.asarray([1], dtype=np.int32))


def write_galerkin_basis(path, basis: GalerkinBasis, *, wfn, meta,
                         centroid_indices, bispinor, refit_rcond,
                         mesh_xy: Mesh) -> None:
    """Collectively publish one immutable mesh-neutral basis artifact."""
    import os
    from common.collectives import barrier, process_count, process_rank
    from file_io.slab_io import SlabIO

    provenance = _basis_provenance(
        wfn=wfn, meta=meta, centroid_indices=centroid_indices,
        node_indices=basis.requested_nodes or basis.node_indices,
        band_range=basis.band_range, bispinor=bispinor,
        refit_rcond=refit_rcond)
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
    nk, nb = (int(n) for n in basis.ctilde.shape[:2])
    nodes = basis.basis_at_nodes
    values = (basis.ctilde, nodes.nodal, basis.loewdin, nodes.target)
    shapes = ((nk, nb, physical), (nk, nb, physical), (nk, nb, nb),
              (nk, nb, int(nodes.nspinor) * int(nodes.shape[2])))
    values, shapes = (values, shapes) if nodes.target is not None else (
        values[:3], shapes[:3])
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
            f"reads the nodal format {_BASIS_FORMAT}); it is refit")
    band = small("band_range", np.int32)
    extents = small("nk_nb_nspinor", np.int32)
    controls = small("controls", np.float64)
    counts = small("orbits", np.int32)
    text = lambda name: _basis_decode(small(name, np.int32))
    if text("selection") != NODE_SELECTION_VERSION:
        raise GalerkinBasisMismatch(
            f"Galerkin basis node version {text('selection')} is not "
            f"{NODE_SELECTION_VERSION}; it is refit")
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
        "node_shape": tuple(int(n) for n in small("node_shape", np.int32)),
        "targets": bool(small("targets", np.int32)[0]),
        "centroid_hash": text("centroid_hash"),
        "node_hash": text("node_hash"), "wfn_hash": text("wfn_hash"),
        "wfn_scheme": text("wfn_scheme"),
        "transform_scheme": text("transform_scheme"),
        "refit_rcond": float(controls[0]), "fit_seconds": float(controls[1]),
        "metric_eps": float(controls[2]),
        "node_weights": tuple(float(v) for v in small(
            "node_weights", np.float64)),
        "selection": text("selection"),
        "n_orbits": int(counts[0]),
        "node_keep": small("node_keep", np.int32).astype(bool),
    }


def read_galerkin_basis(path, *, wfn, meta, centroid_indices, node_indices,
                        band_range, bispinor, refit_rcond, mesh_xy: Mesh,
                        extra_rank_pad: int = 0) -> GalerkinBasis:
    """Collectively validate and read a basis on the caller's target mesh."""
    import os
    from file_io.slab_io import SlabIO

    expected = _basis_provenance(
        wfn=wfn, meta=meta, centroid_indices=centroid_indices,
        node_indices=node_indices, band_range=band_range, bispinor=bispinor,
        refit_rcond=refit_rcond)
    with SlabIO(os.path.abspath(os.fspath(path)), mode="r", mesh=mesh_xy) as io:
        stored = _basis_read_meta(io)
        mismatches = [key for key in expected if stored.get(key) != expected[key]]
        if mismatches:
            raise GalerkinBasisMismatch(
                "Galerkin basis provenance mismatch: " + ", ".join(mismatches))
        carrier = _rank_carrier(stored["rank"], mesh_xy, extra_rank_pad)
        rows = P(None, None, ("x", "y"))
        nk, nb = stored["nk"], stored["nb"]
        # SlabIO's registered shape>dataset contract zero-extends the physical
        # logical dataset directly into the mesh-legal carrier shape.
        ctilde, nodal, loewdin = (
            io.read_slab(name, shape=shape, partition_spec=spec)
            for name, shape, spec in zip(
                _BASIS_ARRAYS[:3], ((nk, nb, carrier),) * 2 + ((nk, nb, nb),),
                (rows,) * 2 + (P(),)))
        target = None
        if stored["targets"]:
            t_carrier = _rank_carrier(
                stored["nspinor"] * stored["centroid_shape"][0], mesh_xy, 0)
            target = io.read_slab(_BASIS_ARRAYS[3], shape=(nk, nb, t_carrier),
                                  partition_spec=rows)
    requested = np.asarray(node_indices, dtype=np.int64).reshape(-1, 3)
    kept = requested[stored["node_keep"]]
    basis = GalerkinBasis(
        ctilde=ctilde,
        basis_at_nodes=_node_interp(
            kept, nodal, meta=meta, wfn=wfn,
            nspinor=stored["nspinor"], target=target,
            centroid_indices=centroid_indices),
        rank_physical=stored["rank"], band_range=stored["band_range"],
        node_indices=_index_rows(kept), loewdin=loewdin,
        node_weights=stored["node_weights"],
        metric_eps=stored["metric_eps"],
        refit_rcond=stored["refit_rcond"], n_orbits=stored["n_orbits"],
        fit_seconds=stored["fit_seconds"],
        requested_nodes=(_index_rows(requested)
                         if len(kept) != len(requested) else ()))
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


def _index_rows(indices) -> tuple:
    """FFT-grid index rows ``(n, 3)`` as a hashable tuple of int triples."""
    return tuple(tuple(int(v) for v in row)
                 for row in np.asarray(indices, dtype=np.int64).reshape(-1, 3))


def _node_interp(node_indices, nodal, *, meta, wfn, nspinor: int,
                 target=None, centroid_indices=None) -> NodalInterp:
    """The :class:`NodalInterp` of the nodes around ``nodal``; with
    ``target`` the states at the ISDF centroids ``centroid_indices``."""
    from common.ws_images import lattice_bohr
    grid = np.asarray(meta.fft_grid, dtype=np.float64)[None, :]
    frac = lambda idx: np.asarray(idx, dtype=np.int64).reshape(-1, 3) / grid
    return NodalInterp(nodal, frac(node_indices), lattice_bohr(wfn),
                       int(meta.sys_dim), nspinor, target=target,
                       target_frac=None if target is None else
                       frac(centroid_indices))


#: Host bytes of one node block of :func:`_quadrature_metric`'s normal rows.
QUADRATURE_BLOCK_BYTES = 1 << 26


def _quadrature_metric(phi, labels, *, block_bytes: int = QUADRATURE_BLOCK_BYTES):
    """Orbit-tied non-negative weights ``w`` with ``Phi_k^H W Phi_k ~ 1``.

    ``phi`` ``(nk, nb, ns, n_mu)`` host nodal values, ``labels`` the node
    orbit of each node.  The normal equations of ``sum_nu w_nu sum_s
    conj(phi_kms nu) phi_kns nu = delta_mn`` over every (k, m, n) are folded
    onto the orbits, a block of nodes at a time (one ``(ns B, ns n_mu)``
    block of ``x^T conj(x)`` is live, at most ``block_bytes``), and solved by
    NNLS (``W^{1/2}`` needs ``w >= 0``); a solve that does not converge
    refuses.  Returns ``(w (n_mu,), G (nk, nb, nb))``.
    """
    from scipy.optimize import nnls
    nk, nb, ns, n_mu = phi.shape
    labels = np.asarray(labels, dtype=np.int64)
    n_orb = int(labels.max()) + 1
    order = np.argsort(labels, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(labels[order]) != 0])
    rows = max(1, int(block_bytes) // (16 * ns * ns * n_mu))
    a_o, b_o = np.zeros((n_orb, n_orb)), np.zeros(n_orb)
    for k in range(nk):
        x = phi[k].reshape(nb, ns * n_mu)
        b_o += np.bincount(labels, weights=np.real(np.einsum(
            'msn,msn->n', np.conj(phi[k]), phi[k])), minlength=n_orb)
        for r0 in range(0, n_mu, rows):
            r1 = min(n_mu, r0 + rows)
            part = phi[k][:, :, r0:r1].reshape(nb, ns * (r1 - r0))
            q = np.abs(part.T @ np.conj(x)) ** 2
            block = q.reshape(ns, r1 - r0, ns, n_mu).sum(axis=(0, 2))
            np.add.at(a_o, labels[r0:r1],
                      np.add.reduceat(block[:, order], starts, axis=1))
    chol = np.linalg.cholesky(
        a_o + 1e-14 * np.trace(a_o) / n_orb * np.eye(n_orb))
    try:
        w_o, rnorm = nnls(chol.T, np.linalg.solve(chol, b_o),
                          maxiter=20 * n_orb)
    except RuntimeError as exc:
        raise RuntimeError(
            f"GATE quadrature_metric_nnls: got: the orbit-tied NNLS over "
            f"{n_orb} node orbits did not converge in {20 * n_orb} "
            f"iterations ({exc}); want: a converged metric (check the node "
            "set: orbit closure and N_node >= "
            f"{NODES_PER_FITTED_BAND} x fitted bands)") from None
    if not (np.isfinite(rnorm) and np.all(np.isfinite(w_o))):
        raise RuntimeError("GATE quadrature_metric_nnls: got: a non-finite "
                           "NNLS solution; want: finite node weights")
    w = w_o[labels]
    gram = np.einsum('kmsn,n,klsn->kml', np.conj(phi), w, phi, optimize=True)
    return w, gram


def _loewdin(gram):
    """``G^{-1/2}`` per k and ``eps = max_k ||G_k - 1||_2`` (host).

    A ``G_k`` whose smallest eigenvalue is within ``nb`` ulp of its largest
    is singular to roundoff (an exactly rank-deficient sampling lands either
    side of zero) and is refused like a negative one."""
    vals, vecs = np.linalg.eigh(gram)
    floor = (vals.shape[-1] * np.finfo(np.float64).eps
             * np.abs(vals).max(axis=-1, keepdims=True))
    if not np.all(vals > floor):
        raise ValueError(
            "fit_galerkin_basis: the quadrature-metric Gram Phi^H W Phi is "
            f"not positive definite (min eigenvalue {float(vals.min()):.3e}, "
            f"roundoff floor {float(floor.max()):.3e}); "
            "the centroid set does not resolve the fitted window")
    inv_half = np.einsum('kmi,ki,kni->kmn', vecs, vals ** -0.5,
                         np.conj(vecs), optimize=True)
    return inv_half, float(np.max(np.abs(vals - 1.0)))


def _orbit_labels(indices, sym, meta, what: str):
    """Orbit label of every point of an orbit-closed FFT-index set (refused
    otherwise, naming the remedy)."""
    from symmetry_maps import (centroid_source_map_and_wrap,
                               permutation_orbit_labels)
    sym_matrices = np.asarray(sym.sym_matrices)
    try:
        perm, _wrap = centroid_source_map_and_wrap(
            np.asarray(indices, dtype=np.int64), sym_matrices,
            np.asarray(sym.translations)[:sym_matrices.shape[0]],
            tuple(int(n) for n in meta.fft_grid))
    except RuntimeError as exc:
        raise ValueError(
            f"fit_galerkin_basis: the {what} are not orbit-closed under the "
            f"WFN's {sym_matrices.shape[0]} spatial operations, and the nodal "
            "htransform basis requires closure.  Regenerate them with "
            f"centroid.kmeans_cli in orbit-aware mode (--orbit).  ({exc})"
        ) from None
    return permutation_orbit_labels(perm), int(sym_matrices.shape[0])


def _sample(wfn, sym, meta, indices, bispinor, mesh_xy, band_range,
            band_chunk_size):
    """Host ``(nk, nb, ns, n)`` values of the fitted window at ``indices``
    (the GW centroid-face path)."""
    nb = int(band_range[1]) - int(band_range[0])
    psi_y, psi_x = load_centroids_band_chunked(
        wfn, sym, meta, np.asarray(indices, dtype=np.int64), bool(bispinor),
        mesh_xy, band_range=tuple(int(v) for v in band_range),
        band_chunk_size=int(band_chunk_size))
    del psi_x
    n = int(np.asarray(indices).reshape(-1, 3).shape[0])
    if int(psi_y.shape[2]) != int(meta.nspinor) or int(psi_y.shape[1]) != nb:
        raise ValueError(
            f"fit_galerkin_basis: centroid faces {tuple(psi_y.shape)} do "
            f"not carry ({nb} bands, {int(meta.nspinor)} spinor components)")
    return np.asarray(gather_to_host(psi_y))[..., :n]


def fit_galerkin_basis(
        wfn, sym, meta, centroid_indices, mesh_xy: Mesh,
        band_range: tuple[int, int], *,
        node_indices=None,
        log_fn=None,
        band_chunk_size: int = 64,
        bispinor: bool = False,
        refit_rcond: float = 1.0e-3,
        extra_rank_pad: int = 0,
        rank_record_fn=None,
) -> GalerkinBasis:
    """Sample the fitted window at the nodes (module docstring).

    ``node_indices`` (FFT-grid indices, default: the ISDF centroids
    ``centroid_indices``) and the centroids must both be orbit-closed
    (refused otherwise, naming the remedy).  When the nodes are not the
    centroids the window is also sampled at the centroids, the back-map
    targets.  ``refit_rcond`` is recorded for :func:`iter_galerkin_rchunks`.
    """
    if log_fn is None:
        log_fn = lambda *a, **kw: None
    t0 = time.perf_counter()
    b_start, b_end = (int(v) for v in band_range)
    nb = b_end - b_start
    nspinor = int(meta.nspinor)
    centroids = np.asarray(centroid_indices, dtype=np.int64).reshape(-1, 3)
    nodes = (centroids if node_indices is None else
             np.asarray(node_indices, dtype=np.int64).reshape(-1, 3))
    same = nodes.shape == centroids.shape and np.array_equal(nodes, centroids)
    n_mu = int(nodes.shape[0])
    if nb <= 0:
        raise ValueError(
            f"fit_galerkin_basis: empty band range [{b_start},{b_end})")
    if not (0.0 < float(refit_rcond) < 1.0):
        raise ValueError(
            f"fit_galerkin_basis: htransform_qr_eps={refit_rcond!r} must lie "
            "in (0,1)")
    if getattr(meta, "mu_basis", None) is not None:
        raise ValueError(
            "fit_galerkin_basis: meta carries a packed centroid basis "
            "(meta.mu_basis); the nodal htransform samples the canonical "
            "centroid table and needs meta.mu_basis = None")

    labels, n_ops = _orbit_labels(nodes, sym, meta, "nodes")
    if not same:
        _orbit_labels(centroids, sym, meta, "ISDF centroids (targets)")
    n_orbits = int(labels.max()) + 1
    phi = _sample(wfn, sym, meta, nodes, bispinor, mesh_xy, (b_start, b_end),
                  band_chunk_size)                       # (nk, nb, ns, n_mu)
    t_sample = time.perf_counter() - t0
    w, gram = _quadrature_metric(phi, labels)
    inv_half, eps = _loewdin(gram)
    # A zero-weight node's row of C = W^{1/2} Phi G^{-1/2} is exactly zero:
    # drop it from the rank axis (exact).  The ISDF centroids stay the
    # back-map targets, so a dropped centroid node keeps its sampled values.
    requested, keep = nodes, w > 0.0
    n_zero = int(np.count_nonzero(~keep))
    target_values = None
    if n_zero:
        if same:
            target_values = phi                  # every centroid, pre-drop
        nodes, phi, w, labels = nodes[keep], phi[..., keep], w[keep], labels[keep]
        n_mu, same = int(nodes.shape[0]), False
        n_orbits = int(np.unique(labels).size)
    coeff = np.einsum('kmn,kmsv,v->knsv', inv_half, phi, np.sqrt(w),
                      optimize=True)
    rank_phys = nspinor * n_mu
    carrier = _rank_carrier(rank_phys, mesh_xy, extra_rank_pad)
    rows = NamedSharding(mesh_xy, P(None, None, ("x", "y")))

    def pad(a, to):
        flat = a.reshape(a.shape[0], nb, -1)
        return np.pad(flat, ((0, 0), (0, 0), (0, to - flat.shape[2])))

    ctilde = device_put_process_local(pad(coeff, carrier), rows)
    nodal = device_put_process_local(pad(phi, carrier), rows)
    del phi
    target = None
    if not same:
        t_carrier = _rank_carrier(nspinor * centroids.shape[0], mesh_xy, 0)
        target = device_put_process_local(pad(
            target_values if target_values is not None else _sample(
                wfn, sym, meta, centroids, bispinor, mesh_xy,
                (b_start, b_end), band_chunk_size), t_carrier), rows)
    del target_values
    loewdin = device_put_process_local(inv_half, NamedSharding(mesh_xy, P()))
    seconds = time.perf_counter() - t0
    from_centroids = (requested.shape == centroids.shape
                      and np.array_equal(requested, centroids))
    log_fn(
        f"  [nodes] {n_mu} nodes"
        + (" (the ISDF centroids)" if same else
           (f" (the ISDF centroids less {n_zero} zero-weight" if from_centroids
            else " (k-means") + f"; back-map targets: the "
           f"{centroids.shape[0]} ISDF centroids)")
        + f", {n_orbits} orbit(s) under {n_ops} ops; {int(meta.nk_tot)} k x "
        f"{nb} bands x {nspinor} spinor; metric eps=max_k||Phi^H W Phi - "
        f"1||_2={eps:.3e}, {n_zero} zero-weight node(s) dropped of "
        f"{requested.shape[0]}; rank {rank_phys} (+"
        f"{carrier - rank_phys} pad); fit {seconds:.2f} s (sampling "
        f"{t_sample:.2f} s)")
    if eps > NODE_METRIC_EPS_WARN:
        log.warning(
            "the nodal Loewdin frame mixes window-edge states",
            extra=message(
                "galerkin_nodal_metric_large",
                found=("nodal metric eps", f"{eps:.3e}"),
                needs=f"eps <= {NODE_METRIC_EPS_WARN:g}",
                lines=("the off-grid interpolation error grows",
                       f"use at least {NODES_PER_FITTED_BAND} x {nb} fitted bands = "
                       f"{NODES_PER_FITTED_BAND * nb} orbit-closed nodes; this set "
                       f"has {requested.shape[0]}",),
                fix="more orbit-closed ISDF centroids: python -m centroid.kmeans_cli "
                    "--orbit"))
    basis = GalerkinBasis(
        ctilde=ctilde,
        basis_at_nodes=_node_interp(
            nodes, nodal, meta=meta, wfn=wfn, nspinor=nspinor,
            target=target, centroid_indices=None if same else centroids),
        rank_physical=rank_phys,
        band_range=(b_start, b_end),
        node_indices=_index_rows(nodes),
        requested_nodes=_index_rows(requested) if n_zero else (),
        loewdin=loewdin,
        node_weights=tuple(float(v) for v in w),
        metric_eps=eps,
        refit_rcond=float(refit_rcond),
        n_orbits=n_orbits,
        fit_seconds=float(seconds),
    )
    if rank_record_fn is not None:
        rank_record_fn(galerkin_rank_record(basis, meta=meta))
    return basis


_ZETA_KERNELS: dict = {}


def _node_gram_factor(basis: GalerkinBasis, *, nspinor: int, mesh_xy: Mesh):
    """``B`` ``(1, n, n)`` on the face ``P(None, 'x', 'y')`` with ``B B^H =
    G^+``, the pseudo-inverse of the node Gram of the coefficient rows
    (relative cutoff ``refit_rcond**2`` on the unit-diagonal Gram): the factor
    of the least-squares rows ``zeta_nu(r)`` (:func:`iter_galerkin_rchunks`
    only).  ``n`` is the node count padded to tile the mesh; pad rows of B
    are zero.  The Gram is built on the face and its eigh is placed by
    ``distrib_la`` in the room beside what is live (whole matrices per rank
    where they fit, else the whole mesh), so no rank holds the
    ``n_nodes^2`` Gram by construction."""
    from common.gpu_utils import shared_device_room
    from distrib_la import plan as linalg_plan
    from runtime.padding import padded_axis
    n_nodes = int(basis.basis_at_nodes.n_nodes)
    n = padded_axis(n_nodes, mesh_xy, name="Galerkin node Gram",
                    specs=((P("x", None), 0), (P(None, "y"), 1))).carrier
    face = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    rep = NamedSharding(mesh_xy, P())
    cut = float(basis.refit_rcond) ** 2

    @partial(jax.jit, out_shardings=(face, rep))
    def _gram(c):
        rows = c[:, :, :nspinor * n_nodes].reshape(
            c.shape[0], c.shape[1], nspinor, n_nodes)
        rows = jnp.pad(rows, ((0, 0), (0, 0), (0, 0), (0, n - n_nodes)))
        g = jax.lax.with_sharding_constraint(jnp.einsum(
            'kbsn,kbsm->nm', jnp.conj(rows), rows, optimize=True)[None], face)
        scale = jnp.sqrt(jnp.maximum(jnp.real(jnp.diagonal(g[0])), 1e-300))
        live = jnp.arange(n) < n_nodes
        unit = g / scale[None, :, None] / scale[None, None, :]
        unit = jnp.where(live[:, None] & live[None, :], unit, 0)
        return jax.lax.with_sharding_constraint(unit, face), scale

    unit, scale = _gram(basis.ctilde)
    room = shared_device_room("Galerkin node Gram eigh")
    lam, vec = linalg_plan("eigh", mesh_xy, n=n, budget_bytes=room).batched(
        unit)

    @partial(jax.jit, out_shardings=face)
    def _factor(lam, vec, scale):
        keep = lam[0] > cut * jnp.max(lam[0])
        col = jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, lam[0], 1.0)), 0.0)
        row = jnp.where(jnp.arange(n) < n_nodes, 1.0 / scale, 0.0)
        return vec * row[None, :, None] * col[None, None, :]

    return _factor(lam, vec, scale)


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
        r_chunk_ranges, retained_band_range: tuple[int, int],
        receipt: dict | None = None):
    """Yield bounded basis rows and requested WFN rows together.

    The least-squares rows of the coefficient space on the grid,

    ``zeta_nu(r) = sum_nu' G^+[nu, nu'] sum_{k,n,s} conj(C_{k,n,(s,nu')}) psi_{nk,s}(r)``,

    with ``G`` the node Gram of ``C``: the frame vector ``C_k e_n`` (the fH_k
    eigenvector of band n) maps to the WFN state ``psi_nk``, as the back-map
    does at the nodes.  This is accumulated from the caller-owned canonical :class:`PsiGStore` during
    the same WFN/FFT pass that retains ``retained_band_range`` for the
    consumer.  Yields ``(r0, r1, B_chunk, psi_parts)`` with ``B_chunk``
    ``(rank, nspinor, r_carrier)`` the spin-block rows
    ``B[(s, nu), s', r] = delta_{s s'} zeta_nu(r)``, so
    ``psi(r) = sum_rho c_rho B[rho, :, r]`` for nodal values ``c``, and
    ``psi_parts`` the ordered ``((band_lo, band_hi), psi_chunk)`` pairs of the
    retained range.  A ``receipt`` dict receives, once the generator is
    exhausted, the full-r relative residuals ``||psi - C B|| / ||psi||`` of
    the fit over the basis window (``"fh"``, from the normal equations) and
    over the retained window (``"zeta"``, direct).
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
    n_nodes = int(basis.basis_at_nodes.n_nodes)
    bc = int(source.band_chunk_carrier)
    rep = NamedSharding(mesh_xy, P())
    row_layout = NamedSharding(mesh_xy, P(None, None, ('y', 'x')))
    psi_layout = NamedSharding(mesh_xy, P(None, None, None, ('y', 'x')))
    r_divisor = spec_divisor(mesh_xy, row_layout.spec, axis=2)

    @partial(jax.jit, static_argnames=("lo", "hi", "pad"), out_shardings=rep)
    def _phi(ctilde, lo, hi, pad):
        rows = ctilde[:, lo:hi, :nspinor * n_nodes].reshape(
            ctilde.shape[0], hi - lo, nspinor, n_nodes)
        return jnp.pad(rows, ((0, 0), (0, pad), (0, 0), (0, 0)))

    phis = [_phi(basis.ctilde, int(lo) - b_start, int(hi) - b_start,
                 bc - (int(hi) - int(lo)))
            for lo, hi in source.band_chunk_ranges]
    factor = _node_gram_factor(basis, nspinor=nspinor, mesh_xy=mesh_xy)
    from common.gpu_utils import shared_device_room
    from distrib_la import matmul as face_matmul
    room = shared_device_room("Galerkin refit rows GEMM")
    mm = partial(face_matmul, mesh=mesh_xy, batched_route="auto",
                 budget_bytes=room)
    n_face = int(factor.shape[1])
    face = NamedSharding(mesh_xy, P(None, 'x', 'y'))

    @partial(jax.jit, out_shardings=(row_layout, rep))
    def _rows(factor, acc):
        # zeta = G^+ acc = B (B^H acc) by the face GEMM (distrib_la.matmul:
        # whole matrices per rank only where they fit the room).
        a = jax.lax.with_sharding_constraint(
            jnp.pad(acc, ((0, n_face - n_nodes), (0, 0)))[None], face)
        zeta = mm(factor, mm(factor, a, transa="C"))[0, :n_nodes]
        out = jnp.zeros((rank, nspinor, zeta.shape[-1]), dtype=zeta.dtype)
        for s in range(nspinor):
            out = out.at[s * n_nodes:(s + 1) * n_nodes, s, :].set(zeta)
        return out, jnp.sum(jnp.real(jnp.conj(acc) * zeta))

    @partial(jax.jit, out_shardings=(rep, rep))
    def _part_residual(rows, psi, basis_chunk):
        zeta = basis_chunk[:n_nodes, 0, :]
        diff = psi - jnp.einsum('knsv,vr->knsr', rows, zeta, optimize=True)
        return jnp.sum(jnp.abs(diff) ** 2), jnp.sum(jnp.abs(psi) ** 2)

    scalar = lambda x: float(np.asarray(x.addressable_data(0)))
    # |psi|^2 and fitted part over the basis window; residual and |psi|^2
    # over the retained window.
    totals = np.zeros(4)

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
            if receipt is not None:
                n_valid = int(bc_range[1]) - int(bc_range[0])
                totals[0] += scalar(jnp.sum(jnp.abs(psi_bc[:, :n_valid]) ** 2))
            lo = max(int(bc_range[0]), keep_start)
            hi = min(int(bc_range[1]), keep_end)
            if lo < hi:
                offset = lo - int(bc_range[0])
                retained.append(
                    ((lo, hi), psi_bc[:, offset:offset + (hi - lo)]))
            else:
                del psi_bc
        basis_chunk, fitted = _rows(factor, acc)
        del acc
        if receipt is not None:
            totals[1] += scalar(fitted)
            for (lo, hi), psi in retained:
                rows = _phi(basis.ctilde, lo - b_start, hi - b_start, 0)
                res2, nrm2 = _part_residual(rows, psi, basis_chunk)
                totals[2] += scalar(res2)
                totals[3] += scalar(nrm2)
        if sum(hi - lo for (lo, hi), _ in retained) != keep_end - keep_start:
            raise RuntimeError(
                "iter_galerkin_rchunks: retained WFN pieces do not cover "
                f"[{keep_start},{keep_end})")
        yield r0, r1, basis_chunk, tuple(retained)
    if receipt is not None:
        receipt["fh"] = float(np.sqrt(max(totals[0] - totals[1], 0.0)
                                      / max(totals[0], 1e-300)))
        receipt["zeta"] = float(np.sqrt(totals[2] / max(totals[3], 1e-300)))


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

