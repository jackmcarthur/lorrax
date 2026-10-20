"""Orbit-closed nodal htransform basis on the ISDF centroids.

The nodes are the run's own ISDF centroids (charge-density-weighted k-means,
orbit-closed under the crystal operations).  Every coarse state of the fitted
window is sampled there once, through the GW centroid-face path
(``common.wfn_transforms.load_centroids_band_chunked``):

    Phi_k[(s, nu), n] = psi_{nk,s}(r_nu)                       nodal values.

Nodal values carry no metric, so the basis adds the diagonal quadrature
metric ``W = diag(w_nu)`` of the centroids: one non-negative weight per
centroid orbit, the least-squares solution of ``Phi_k^H W Phi_k = 1`` over
all coarse k (NNLS).  ``W`` is invariant under the node permutation of every
crystal operation.  A per-k Loewdin step makes the coefficients exactly
orthonormal,

    G_k = Phi_k^H W Phi_k,   C_k = W^{1/2} Phi_k G_k^{-1/2},   C_k^H C_k = 1,

and ``ctilde[k, n, (s, nu)] = C_k[(s, nu), n]`` is used exactly like an
orthonormal Galerkin coefficient array (``C f C^H``, direct eigh).  ``C f C^H``
is unchanged by window unitaries that commute with ``f`` and ``G_k`` is
gauge-covariant, so a node-pair kernel with per-pair Wigner-Seitz images
(``common.ws_images``) is exactly covariant.  The fit logs ``eps = max_k
||G_k - 1||_2`` and the number of zero-weight nodes (they drop out of ``C``).

:class:`NodalInterp` carries the nodal values ``Phi`` and the node geometry;
the BSE recovers ``psi(r_nu)`` at a fine q from the back-map kernel
``Phi f C^H`` (``bandstructure.bse_setup``).  There is no node selection, no
pivot and no tolerance.  :func:`iter_galerkin_rchunks` continues the
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
from common.wfn_transforms import (
    FULL_BLOCH_TRANSFORM_SCHEME,
    load_centroids_band_chunked,
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


NODE_SELECTION_VERSION = "centroid-nodes-orbit-nnls-loewdin-v1"

#: Metric quality above which the fit warns.  Si 4^3 (36 fitted spinor
#: bands, orbit-closed k-means, fit window 0:36): eps 0.37 / 0.12 / 1.6e-2 /
#: 1.9e-3 / 1.0e-3 at 312 / 432 / 588 / 780 / 1100 nodes; off-grid DFT MAE
#: 8.03 meV (max 62.7) at 312 nodes against 6.76 (max 27.4) at 780.
NODE_METRIC_EPS_WARN = 1.0e-2
#: Node count per fitted band that reached eps <= 1e-2 on Si 4^3 (780/36);
#: MoS2 9x9 at 22.2 nodes per band (1198/54) reads eps 1.7e-2.
NODES_PER_FITTED_BAND = 22
_BASIS_FORMAT = 4
_BASIS_ARRAYS = ("galerkin_ctilde", "galerkin_nodal", "galerkin_loewdin")
_BASIS_META = "galerkin_"


class GalerkinBasisMismatch(ValueError):
    """A complete basis artifact fitted for other inputs or another format.

    Raised only by the provenance/format comparison, so a caller may refit
    on it; a corrupt or incomplete artifact still raises plain ValueError.
    """


@jax.tree_util.register_pytree_node_class
class NodalInterp:
    """Nodal values of the fitted states and the node geometry.

    ``nodal`` ``(nk, nb, rank)`` holds ``Phi_k[(s, nu), n] = psi_{nk,s}(r_nu)``
    on the carried ``(s, nu)`` rank axis (rows past ``ns*n_nodes`` are
    exact-null pad).  The geometry every node-pair Fourier sum needs is
    static pytree data: ``node_frac`` ``(n_nodes, 3)`` lattice coordinates
    (carried row ``s*n_nodes + nu`` sits at node ``nu``), the lattice
    ``(3, 3)`` rows ``a_i`` in bohr, ``sys_dim`` and ``nspinor``.  ``shape``
    is ``(rank, nspinor, n_nodes)``, the extents the BSE consumers read.
    """

    def __init__(self, nodal, node_frac, lattice_bohr, sys_dim, nspinor):
        self.nodal = nodal
        self.node_frac = np.asarray(node_frac, dtype=np.float64)
        self.lattice_bohr = np.asarray(lattice_bohr, dtype=np.float64)
        self.sys_dim = int(sys_dim)
        self.nspinor = int(nspinor)

    @property
    def shape(self):
        return (int(self.nodal.shape[2]), self.nspinor,
                int(self.node_frac.shape[0]))

    def tree_flatten(self):
        aux = (tuple(map(tuple, self.node_frac.tolist())),
               tuple(map(tuple, self.lattice_bohr.tolist())), self.sys_dim,
               self.nspinor)
        return (self.nodal,), aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], np.asarray(aux[0]), np.asarray(aux[1]),
                   aux[2], aux[3])


@dataclass(frozen=True)
class GalerkinBasis:
    """One fitted nodal basis.

    ``ctilde`` ``(nk, nb, rank)`` holds the orthonormal coefficients
    ``C_k`` (module docstring) on the ``(s, nu)`` rank axis;
    ``basis_at_nodes`` is the :class:`NodalInterp` (nodal values and node
    geometry); ``loewdin`` ``(nk, nb, nb)`` is ``G_k^{-1/2}``.  ``node_weights``
    are the metric weights ``w_nu``, ``metric_eps = max_k ||G_k - 1||_2``.
    ``rank_physical = ns * n_nodes`` excludes the exact-null mesh pad.
    ``qrcp_eps`` is used only by :func:`iter_galerkin_rchunks` (the relative
    cutoff of its Gram pseudo-inverse).  ``fit_seconds`` is the fit wall.
    """

    ctilde: jax.Array
    basis_at_nodes: NodalInterp
    rank_physical: int
    band_range: tuple[int, int]
    node_indices: tuple[int, ...]
    loewdin: jax.Array = None
    node_weights: tuple = ()
    metric_eps: float = 0.0
    qrcp_eps: float = 1.0e-3
    n_orbits: int = 0
    fit_seconds: float = 0.0

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


def validate_rank_multiplier(value, *, name: str = "rank_multiplier") -> float:
    """Validate ``htransform_rank_multiplier`` (parsed for archived decks).

    The nodal basis has no rank search, so the value bounds nothing; ``0``
    is retained as an input-compatibility spelling of ``20``.
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
    """The node receipt of a fitted or restored basis (host only)."""
    nk, nb, carrier = (int(n) for n in basis.ctilde.shape)
    physical = int(basis.rank_physical)
    return {
        "method": "centroid_nodes",
        "selection_version": NODE_SELECTION_VERSION,
        "stacked_states": nk * nb,
        "centroids": int(basis.basis_at_nodes.shape[2]),
        "nodes": len(basis.node_indices),
        "orbits": int(basis.n_orbits),
        "retained_rank": physical,
        "carried_rank": carrier,
        "null_padding": carrier - physical,
        "rank_multiplier": validate_rank_multiplier(
            rank_multiplier, name="htransform_rank_multiplier"),
        "fit_seconds": float(basis.fit_seconds),
        "metric_eps": float(basis.metric_eps),
        "zero_weight_nodes": int(np.count_nonzero(
            np.asarray(basis.node_weights) == 0.0)),
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


def _basis_check(basis: GalerkinBasis, provenance: dict) -> None:
    if tuple(basis.band_range) != provenance["band_range"] \
            or tuple(basis.ctilde.shape[:2]) != (
                provenance["nk"], provenance["nb"]) \
            or tuple(basis.basis_at_nodes.shape[1:]) != (
                provenance["nspinor"], provenance["centroid_shape"][0]):
        raise ValueError("Galerkin basis payload disagrees with provenance")
    if basis.qrcp_eps != provenance["qrcp_eps"]:
        raise ValueError("Galerkin basis qr_eps disagrees with provenance")
    if len(basis.node_indices) != provenance["centroid_shape"][0]:
        raise ValueError("Galerkin nodes are not the centroid set")


def _basis_write_meta(io, basis: GalerkinBasis, provenance: dict) -> None:
    integer = {
        "format": [_BASIS_FORMAT], "rank": [basis.rank_physical],
        "band_range": provenance["band_range"],
        "nk_nb_nspinor": [provenance[k] for k in ("nk", "nb", "nspinor")],
        "fft_grid": provenance["fft_grid"], "kgrid": provenance["kgrid"],
        "bispinor": [provenance["bispinor"]],
        "centroid_shape": provenance["centroid_shape"],
        "orbits": [basis.n_orbits],
    }
    for name, value in integer.items():
        io.write_attr(_BASIS_META + name, np.asarray(value, dtype=np.int32))
    io.write_attr(_BASIS_META + "controls", np.asarray([
        provenance["rank_multiplier"], provenance["qrcp_eps"],
        basis.fit_seconds, basis.metric_eps], dtype=np.float64))
    io.write_attr(_BASIS_META + "node_weights",
                  np.asarray(basis.node_weights, dtype=np.float64))
    for name, value in (
            ("centroid_hash", provenance["centroid_hash"]),
            ("wfn_hash", provenance["wfn_hash"]),
            ("wfn_scheme", provenance["wfn_scheme"]),
            ("transform_scheme", provenance["transform_scheme"]),
            ("selection", provenance["selection"])):
        io.write_attr(_BASIS_META + name, _basis_text(value))
    # SlabIO's write_attr queues a small dataset for the rank-0 reopen after
    # collective close; read_small is its matching collective reader.
    io.write_attr(_BASIS_META + "complete", np.asarray([1], dtype=np.int32))


def write_galerkin_basis(path, basis: GalerkinBasis, *, wfn, meta,
                         centroid_indices, bispinor, rank_multiplier,
                         qrcp_eps, mesh_xy: Mesh, qrcp_seed: int = 0) -> None:
    """Collectively publish one immutable mesh-neutral basis artifact.

    ``qrcp_seed`` is accepted for archived callers and is not recorded."""
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
    nk, nb = (int(n) for n in basis.ctilde.shape[:2])
    values = (basis.ctilde, basis.basis_at_nodes.nodal, basis.loewdin)
    shapes = ((nk, nb, physical), (nk, nb, physical), (nk, nb, nb))
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
    counts = small("orbits", np.int32)
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
        "fit_seconds": float(controls[2]), "metric_eps": float(controls[3]),
        "node_weights": tuple(float(v) for v in small(
            "node_weights", np.float64)),
        "selection": text("selection"),
        "n_orbits": int(counts[0]),
    }


def read_galerkin_basis(path, *, wfn, meta, centroid_indices, band_range,
                        bispinor, rank_multiplier, qrcp_eps,
                        mesh_xy: Mesh, extra_rank_pad: int = 0,
                        qrcp_seed: int = 0) -> GalerkinBasis:
    """Collectively validate and read a basis on the caller's target mesh."""
    import os
    from file_io.slab_io import SlabIO

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
        carrier = _rank_carrier(stored["rank"], mesh_xy, extra_rank_pad)
        n_mu = stored["centroid_shape"][0]
        # SlabIO's registered shape>dataset contract zero-extends the physical
        # logical dataset directly into the mesh-legal carrier shape.
        ctilde, nodal, loewdin = (
            io.read_slab(name, shape=shape, partition_spec=spec)
            for name, shape, spec in zip(
                _BASIS_ARRAYS,
                ((stored["nk"], stored["nb"], carrier),) * 2
                + ((stored["nk"], stored["nb"], stored["nb"]),),
                (P(None, None, ("x", "y")),) * 2 + (P(),)))
    basis = GalerkinBasis(
        ctilde=ctilde,
        basis_at_nodes=_centroid_nodes(
            centroid_indices, nodal, meta=meta, wfn=wfn,
            nspinor=stored["nspinor"]),
        rank_physical=stored["rank"], band_range=stored["band_range"],
        node_indices=tuple(range(n_mu)), loewdin=loewdin,
        node_weights=stored["node_weights"],
        metric_eps=stored["metric_eps"],
        qrcp_eps=stored["qrcp_eps"], n_orbits=stored["n_orbits"],
        fit_seconds=stored["fit_seconds"])
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


def _centroid_nodes(centroid_indices, nodal, *, meta, wfn,
                    nspinor: int) -> NodalInterp:
    """The :class:`NodalInterp` of the centroid nodes around ``nodal``."""
    from common.ws_images import lattice_bohr
    idx = np.asarray(centroid_indices, dtype=np.int64)
    frac = idx.astype(np.float64) / np.asarray(
        meta.fft_grid, dtype=np.float64)[None, :]
    return NodalInterp(nodal, frac, lattice_bohr(wfn), int(meta.sys_dim),
                       nspinor)


def _quadrature_metric(phi, labels):
    """Orbit-tied non-negative weights ``w`` with ``Phi_k^H W Phi_k ~ 1``.

    ``phi`` ``(nk, nb, ns, n_mu)`` host nodal values, ``labels`` the centroid
    orbit of each node.  The normal equations of ``sum_nu w_nu sum_s
    conj(phi_kms nu) phi_kns nu = delta_mn`` over every (k, m, n) are folded
    onto the orbits and solved by NNLS (``W^{1/2}`` needs ``w >= 0``).
    Returns ``(w (n_mu,), G (nk, nb, nb))``.
    """
    from scipy.optimize import nnls
    nk, nb, ns, n_mu = phi.shape
    labels = np.asarray(labels, dtype=np.int64)
    n_orb = int(labels.max()) + 1
    ata = np.zeros((n_mu, n_mu))
    atb = np.zeros(n_mu)
    for k in range(nk):
        x = phi[k].reshape(nb, ns * n_mu)
        q = x.T @ np.conj(x)
        ata += (np.abs(q) ** 2).reshape(ns, n_mu, ns, n_mu).sum(axis=(0, 2))
        atb += np.real(np.einsum('msn,msn->n', np.conj(phi[k]), phi[k]))
    fold = np.zeros((n_mu, n_orb))
    fold[np.arange(n_mu), labels] = 1.0
    a_o, b_o = fold.T @ ata @ fold, fold.T @ atb
    chol = np.linalg.cholesky(
        a_o + 1e-14 * np.trace(a_o) / n_orb * np.eye(n_orb))
    w_o = nnls(chol.T, np.linalg.solve(chol, b_o), maxiter=20 * n_orb)[0]
    w = fold @ w_o
    gram = np.einsum('kmsn,n,klsn->kml', np.conj(phi), w, phi, optimize=True)
    return w, gram


def _loewdin(gram):
    """``G^{-1/2}`` per k and ``eps = max_k ||G_k - 1||_2`` (host)."""
    vals, vecs = np.linalg.eigh(gram)
    if not np.all(vals > 0.0):
        raise ValueError(
            "fit_galerkin_basis: the quadrature-metric Gram Phi^H W Phi is "
            f"not positive definite (min eigenvalue {float(vals.min()):.3e}); "
            "the centroid set does not resolve the fitted window")
    inv_half = np.einsum('kmi,ki,kni->kmn', vecs, vals ** -0.5,
                         np.conj(vecs), optimize=True)
    return inv_half, float(np.max(np.abs(vals - 1.0)))


def fit_galerkin_basis(
        wfn, sym, meta, centroid_indices, mesh_xy: Mesh,
        band_range: tuple[int, int], *,
        log_fn=None,
        band_chunk_size: int = 64,
        bispinor: bool = False,
        rank_multiplier: float = 20.0,
        qr_eps: float = 1.0e-3,
        device_pool_limit: float | None = None,
        extra_rank_pad: int = 0,
        progress_fn=None,
        rank_record_fn=None,
        qrcp_seed: int = 0,
        distrib_la_batched_route=None,
) -> GalerkinBasis:
    """Sample the fitted window at the centroid nodes (module docstring).

    The centroid set must be orbit-closed (refused otherwise, naming the
    remedy).  ``rank_multiplier``, ``device_pool_limit``, ``qrcp_seed`` and
    ``distrib_la_batched_route`` are accepted for the caller's signature and
    bound nothing (the centroid-face loader plans its own memory); ``qr_eps``
    is recorded for :func:`iter_galerkin_rchunks`.
    """
    del progress_fn, device_pool_limit, qrcp_seed, distrib_la_batched_route
    from symmetry_maps import (centroid_source_map_and_wrap,
                               permutation_orbit_labels)
    if log_fn is None:
        log_fn = lambda *a, **kw: None
    t0 = time.perf_counter()
    b_start, b_end = (int(v) for v in band_range)
    nb = b_end - b_start
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
    if getattr(meta, "mu_basis", None) is not None:
        raise ValueError(
            "fit_galerkin_basis: meta carries a packed centroid basis "
            "(meta.mu_basis); the nodal htransform samples the canonical "
            "centroid table and needs meta.mu_basis = None")

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
            "nodal htransform basis requires closure.  Regenerate the "
            "centroids with centroid.kmeans_cli in orbit-aware mode.  "
            f"({exc})") from None
    labels = permutation_orbit_labels(perm)
    n_orbits = int(labels.max()) + 1

    psi_y, psi_x = load_centroids_band_chunked(
        wfn, sym, meta, idx, bool(bispinor), mesh_xy,
        band_range=(b_start, b_end), band_chunk_size=int(band_chunk_size))
    del psi_x
    if int(psi_y.shape[2]) != nspinor or int(psi_y.shape[1]) != nb:
        raise ValueError(
            f"fit_galerkin_basis: centroid faces {tuple(psi_y.shape)} do "
            f"not carry ({nb} bands, {nspinor} spinor components)")
    phi = np.asarray(gather_to_host(psi_y))[..., :n_mu]       # (nk, nb, ns, n_mu)
    del psi_y
    t_sample = time.perf_counter() - t0
    w, gram = _quadrature_metric(phi, labels)
    inv_half, eps = _loewdin(gram)
    coeff = np.einsum('kmn,kmsv,v->knsv', inv_half, phi, np.sqrt(w),
                      optimize=True)
    rank_phys = nspinor * n_mu
    carrier = _rank_carrier(rank_phys, mesh_xy, extra_rank_pad)
    rows = NamedSharding(mesh_xy, P(None, None, ("x", "y")))
    pad = lambda a: np.pad(a.reshape(a.shape[0], nb, rank_phys),
                           ((0, 0), (0, 0), (0, carrier - rank_phys)))
    ctilde = device_put_process_local(pad(coeff), rows)
    nodal = device_put_process_local(pad(phi), rows)
    loewdin = device_put_process_local(inv_half, NamedSharding(mesh_xy, P()))
    seconds = time.perf_counter() - t0
    n_zero = int(np.count_nonzero(w == 0.0))
    log_fn(
        f"  [nodes] {n_mu} centroid nodes, {n_orbits} orbit(s) under "
        f"{sym_matrices.shape[0]} ops; {int(meta.nk_tot)} k x {nb} bands x "
        f"{nspinor} spinor; metric eps=max_k||Phi^H W Phi - 1||_2="
        f"{eps:.3e}, {n_zero} zero-weight node(s); rank {rank_phys} (+"
        f"{carrier - rank_phys} pad); fit {seconds:.2f} s (sampling "
        f"{t_sample:.2f} s)")
    if eps > NODE_METRIC_EPS_WARN:
        log_fn(
            f"  [warn] nodal metric eps {eps:.3e} > {NODE_METRIC_EPS_WARN:.0e}: "
            "the Loewdin frame mixes window-edge states and the off-grid "
            "error grows.  Use an orbit-closed k-means set with N_node >= "
            f"{NODES_PER_FITTED_BAND} x {nb} fitted bands = "
            f"{NODES_PER_FITTED_BAND * nb} (centroid.kmeans_cli --orbit "
            f"--fit-window 0:{nb},0:{nb}); this set has {n_mu}.")
    basis = GalerkinBasis(
        ctilde=ctilde,
        basis_at_nodes=_centroid_nodes(idx, nodal, meta=meta, wfn=wfn,
                                       nspinor=nspinor),
        rank_physical=rank_phys,
        band_range=(b_start, b_end),
        node_indices=tuple(range(n_mu)),
        loewdin=loewdin,
        node_weights=tuple(float(v) for v in w),
        metric_eps=eps,
        qrcp_eps=float(qr_eps),
        n_orbits=n_orbits,
        fit_seconds=float(seconds),
    )
    if rank_record_fn is not None:
        rank_record_fn(galerkin_rank_record(
            basis, meta=meta, rank_multiplier=multiplier))
    return basis


_ZETA_KERNELS: dict = {}


def _node_gram_pinv(basis: GalerkinBasis, *, nspinor: int, mesh_xy: Mesh):
    """``G^+`` of the coefficient rows (relative cutoff ``qrcp_eps**2`` on
    the unit-diagonal Gram), replicated: the factor of the least-squares
    rows ``zeta_nu(r)`` (:func:`iter_galerkin_rchunks` only)."""
    n_nodes = len(basis.node_indices)
    rep = NamedSharding(mesh_xy, P())

    @partial(jax.jit, out_shardings=rep)
    def _gram(c):
        rows = c[:, :, :nspinor * n_nodes].reshape(
            c.shape[0], c.shape[1], nspinor, n_nodes)
        return jnp.einsum('kbsn,kbsm->nm', jnp.conj(rows), rows,
                          optimize=True)

    g = np.asarray(_gram(basis.ctilde).addressable_data(0))
    scale = np.sqrt(np.maximum(np.real(np.diag(g)), 1e-300))
    w, v = np.linalg.eigh(g / scale[:, None] / scale[None, :])
    keep = w > float(basis.qrcp_eps) ** 2 * float(w.max())
    pinv = (v[:, keep] / w[keep]) @ np.conj(v[:, keep]).T
    pinv = pinv / scale[:, None] / scale[None, :]
    return device_put_process_local(pinv, rep)


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

    The least-squares rows of the coefficient space on the grid,

    ``zeta_nu(r) = sum_nu' G^+[nu, nu'] sum_{k,n,s} conj(C_{k,n,(s,nu')}) psit_{nk,s}(r)``,

    with ``G`` the node Gram of ``C`` and ``psit = psi G_k^{-1/2}`` the states
    ``C`` represents,

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
    def _phi(ctilde, loewdin, lo, hi):
        # Y_k[m, rho] = sum_n T_k[n, m] C_k[n, rho]: the WFN states psi_m
        # enter through the Loewdin-mixed states C represents.
        y = jnp.einsum('knm,knr->kmr', loewdin, ctilde[..., :nspinor * n_nodes])
        rows = y[:, lo:hi].reshape(ctilde.shape[0], hi - lo, nspinor, n_nodes)
        return jnp.pad(rows, ((0, 0), (0, bc - (hi - lo)), (0, 0), (0, 0)))

    phis = [_phi(basis.ctilde, basis.loewdin, int(lo) - b_start,
                 int(hi) - b_start)
            for lo, hi in source.band_chunk_ranges]
    pinv = _node_gram_pinv(basis, nspinor=nspinor, mesh_xy=mesh_xy)

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
        basis_chunk = _rows(pinv, acc)
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

