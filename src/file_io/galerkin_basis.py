"""SlabIO persistence for one gauge-coupled :class:`GalerkinBasis`.

Only logical physical-rank extents are stored.  The reader reconstructs the
exact-null carrier required by its mesh: zero tails for ``ctilde`` and
``basis_at_nodes``, and an identity tail for ``selection_factor``.  These
three arrays and ``selected_state_indices`` are one atomic artifact because
independently replacing any member changes the shared alpha gauge.

The write is two-phase.  One SlabIO transaction writes every large payload and
closes.  A second metadata-only transaction stamps provenance and
``complete=1`` last.  A crash between them leaves a payload file with no valid
completion record, which every reader refuses.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.parallel_transport import wfn_fingerprint
from common.sharding_fit import fit_sharding
from file_io.slab_io import SlabIO
from isdf.galerkin import GalerkinBasis, QRCP_RNG_VERSION
from runtime.padding import pad_axis, round_up


# Version 1 on the owner branch stored the predecessor's optional dense
# projector.  The current compact basis stores its selected-state factor;
# changing that payload without a version refusal would mislabel the gauge.
FORMAT_VERSION = 2
CTILDE_DATASET = "galerkin_ctilde"
BASIS_DATASET = "galerkin_basis_at_nodes"
SELECTION_FACTOR_DATASET = "galerkin_selection_factor"
_META = "galerkin_"

__all__ = [
    "FORMAT_VERSION",
    "GalerkinBasisStamp",
    "read_galerkin_basis",
    "write_galerkin_basis",
]


def _text_i32(value: str) -> np.ndarray:
    return np.frombuffer(str(value).encode("utf-8"), dtype=np.uint8).astype(
        np.int32)


def _decode_text_i32(value) -> str:
    raw = np.asarray(value, dtype=np.int32).reshape(-1)
    if np.any(raw < 0) or np.any(raw > 255):
        raise ValueError("Galerkin basis text metadata contains a non-byte")
    return bytes(raw.astype(np.uint8).tolist()).decode("utf-8")


def _valid_sha256(value: str) -> bool:
    value = str(value)
    return len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def _selected_hash(indices) -> str:
    values = np.asarray(indices, dtype="<i8").reshape(-1)
    return hashlib.sha256(values.tobytes()).hexdigest()


@dataclass(frozen=True)
class GalerkinBasisStamp:
    """Numerical input identity required to reuse one fitted basis."""

    band_range: tuple[int, int]
    nk: int
    nb: int
    nspinor: int
    fft_grid: tuple[int, int, int]
    kgrid: tuple[int, int, int]
    bispinor: bool
    centroid_indices: np.ndarray
    wfn_path: str
    wfn_fingerprint: str
    rank_multiplier: float
    qrcp_eps: float
    qrcp_seed: int
    qrcp_rng_version: str = QRCP_RNG_VERSION

    def __post_init__(self) -> None:
        b0, b1 = (int(v) for v in self.band_range)
        if b1 <= b0 or b1 - b0 != int(self.nb):
            raise ValueError(
                f"GalerkinBasisStamp band_range={self.band_range!r} and "
                f"nb={self.nb} disagree")
        if min(int(self.nk), int(self.nb), int(self.nspinor)) < 1:
            raise ValueError("GalerkinBasisStamp extents must be positive")
        if len(tuple(self.fft_grid)) != 3 or len(tuple(self.kgrid)) != 3:
            raise ValueError("GalerkinBasisStamp FFT and k grids must be triples")
        if any(int(v) <= 0 for v in (*self.fft_grid, *self.kgrid)):
            raise ValueError("GalerkinBasisStamp grid extents must be positive")
        centroids = np.asarray(self.centroid_indices, dtype=np.int32)
        if centroids.ndim != 2 or centroids.shape[1] != 3 or not centroids.size:
            raise ValueError(
                "GalerkinBasisStamp centroid_indices must be nonempty "
                f"(n_nodes, 3); got {centroids.shape}")
        object.__setattr__(self, "centroid_indices", centroids)
        if (not np.isfinite(float(self.rank_multiplier))
                or float(self.rank_multiplier) < 1.0):
            raise ValueError(
                "GalerkinBasisStamp rank_multiplier must be finite and >= 1")
        if (not np.isfinite(float(self.qrcp_eps))
                or not 0.0 < float(self.qrcp_eps) < 1.0):
            raise ValueError(
                "GalerkinBasisStamp qrcp_eps must be finite and in (0,1)")
        if int(self.qrcp_seed) < 0:
            raise ValueError("GalerkinBasisStamp qrcp_seed must be >= 0")
        if not str(self.qrcp_rng_version):
            raise ValueError("GalerkinBasisStamp qrcp_rng_version is empty")
        if not _valid_sha256(self.wfn_fingerprint):
            raise ValueError(
                "GalerkinBasisStamp requires the canonical lowercase "
                "SHA-256 WFN fingerprint")

    @property
    def n_nodes(self) -> int:
        return int(self.centroid_indices.shape[0])

    @classmethod
    def from_runtime(
            cls, *, wfn, meta, centroid_indices, band_range,
            bispinor: bool, rank_multiplier: float, qrcp_eps: float,
            qrcp_seed: int,
    ) -> "GalerkinBasisStamp":
        """Build the stamp using the canonical location-independent WFN ID."""
        b0, b1 = (int(v) for v in band_range)
        path = (getattr(wfn, "path", None)
                or getattr(wfn, "_filename", None) or "")
        return cls(
            band_range=(b0, b1), nk=int(meta.nk_tot), nb=b1 - b0,
            nspinor=int(meta.nspinor),
            fft_grid=tuple(int(v) for v in meta.fft_grid),
            kgrid=tuple(int(v) for v in meta.kgrid),
            bispinor=bool(bispinor),
            centroid_indices=np.asarray(centroid_indices, dtype=np.int32),
            wfn_path=os.path.realpath(str(path)) if path else "",
            wfn_fingerprint=wfn_fingerprint(wfn),
            rank_multiplier=float(rank_multiplier),
            qrcp_eps=float(qrcp_eps), qrcp_seed=int(qrcp_seed),
            qrcp_rng_version=QRCP_RNG_VERSION,
        )

    def mismatch(self, expected: "GalerkinBasisStamp") -> tuple[str, ...]:
        """Return every provenance difference, with exact centroid identity."""
        out = []
        scalar_fields = (
            "band_range", "nk", "nb", "nspinor", "fft_grid", "kgrid",
            "bispinor", "wfn_fingerprint", "rank_multiplier", "qrcp_eps",
            "qrcp_seed", "qrcp_rng_version",
        )
        for name in scalar_fields:
            got = getattr(self, name)
            want = getattr(expected, name)
            if got != want:
                out.append(f"{name}: stored {got!r} != runtime {want!r}")
        if not np.array_equal(self.centroid_indices,
                              expected.centroid_indices):
            if self.centroid_indices.shape != expected.centroid_indices.shape:
                detail = (f"shape {self.centroid_indices.shape} != "
                          f"{expected.centroid_indices.shape}")
            else:
                rows = np.flatnonzero(np.any(
                    self.centroid_indices != expected.centroid_indices,
                    axis=1))
                detail = f"{rows.size} row(s) differ; first row {int(rows[0])}"
            out.append("centroid_indices: " + detail)
        return tuple(out)


def _max_abs(value) -> float:
    if not value.size:
        return 0.0
    return float(jax.device_get(jnp.max(jnp.abs(value))))


def _assert_basis_matches_stamp(
        basis: GalerkinBasis, stamp: GalerkinBasisStamp) -> None:
    if tuple(int(v) for v in basis.band_range) != tuple(stamp.band_range):
        raise ValueError("Galerkin basis/stamp band ranges disagree")
    if tuple(int(v) for v in basis.ctilde.shape[:2]) != (stamp.nk, stamp.nb):
        raise ValueError("Galerkin basis/stamp state extents disagree")
    if int(basis.basis_at_nodes.shape[1]) != int(stamp.nspinor):
        raise ValueError("Galerkin basis/stamp spinor extents disagree")
    if int(basis.basis_at_nodes.shape[2]) != int(stamp.n_nodes):
        raise ValueError("Galerkin basis/stamp node extents disagree")
    if int(basis.qrcp_seed) != int(stamp.qrcp_seed):
        raise ValueError("Galerkin basis/stamp qrcp_seed disagree")
    if float(basis.qrcp_eps) != float(stamp.qrcp_eps):
        raise ValueError("Galerkin basis/stamp qrcp_eps disagree")
    if str(basis.qrcp_rng_version) != str(stamp.qrcp_rng_version):
        raise ValueError("Galerkin basis/stamp qrcp_rng_version disagree")
    if not _valid_sha256(basis.candidate_hash):
        raise ValueError("Galerkin basis candidate_hash is not canonical SHA-256")
    if not _valid_sha256(basis.pivot_hash):
        raise ValueError("Galerkin basis pivot_hash is not canonical SHA-256")

    physical = int(basis.rank_physical)
    carrier = int(basis.rank_carrier)
    selected = np.asarray(basis.selected_state_indices, dtype=np.int64)
    if np.any(selected < 0) or np.any(selected >= stamp.nk * stamp.nb):
        raise ValueError("Galerkin selected-state index lies outside nk*nb")
    if np.unique(selected).size != physical:
        raise ValueError("Galerkin selected-state indices are not unique")
    if _selected_hash(selected) != str(basis.pivot_hash):
        raise ValueError("Galerkin selected-state indices disagree with pivot_hash")
    if physical < carrier:
        for name, tail in (
                ("ctilde", basis.ctilde[..., physical:]),
                ("basis_at_nodes", basis.basis_at_nodes[physical:]),
                ("selection_factor lower cross",
                 basis.selection_factor[physical:, :physical]),
                ("selection_factor upper cross",
                 basis.selection_factor[:physical, physical:])):
            if _max_abs(tail) != 0.0:
                raise ValueError(
                    f"Galerkin {name} carrier tail is not exact zero")
        tail = basis.selection_factor[physical:, physical:]
        identity = jnp.eye(carrier - physical, dtype=tail.dtype)
        if _max_abs(tail - identity) != 0.0:
            raise ValueError(
                "Galerkin selection_factor padding is not exact identity")


def write_galerkin_basis(
        path, basis: GalerkinBasis, stamp: GalerkinBasisStamp,
        *, mesh_xy: Mesh) -> None:
    """Collectively publish one complete, mesh-independent basis artifact."""
    _assert_basis_matches_stamp(basis, stamp)
    path = str(path)
    rank = int(basis.rank_physical)
    payloads = (
        (CTILDE_DATASET, basis.ctilde[..., :rank]),
        (BASIS_DATASET, basis.basis_at_nodes[:rank]),
        (SELECTION_FACTOR_DATASET,
         basis.selection_factor[:rank, :rank]),
    )
    with SlabIO(path, mode="w", mesh=mesh_xy) as io:
        for name, value in payloads:
            io.create_dataset(
                name, shape=tuple(int(v) for v in value.shape),
                dtype=value.dtype)
            io.write_slab(name, value)

    with SlabIO(path, mode="a", mesh=mesh_xy) as io:
        io.write_attr(_META + "format_version",
                      np.asarray([FORMAT_VERSION], dtype=np.int32))
        io.write_attr(_META + "rank_physical",
                      np.asarray([rank], dtype=np.int32))
        io.write_attr(_META + "band_range",
                      np.asarray(stamp.band_range, dtype=np.int32))
        io.write_attr(_META + "nk_nb_nspinor",
                      np.asarray([stamp.nk, stamp.nb, stamp.nspinor],
                                 dtype=np.int32))
        io.write_attr(_META + "fft_grid",
                      np.asarray(stamp.fft_grid, dtype=np.int32))
        io.write_attr(_META + "kgrid",
                      np.asarray(stamp.kgrid, dtype=np.int32))
        io.write_attr(_META + "bispinor",
                      np.asarray([stamp.bispinor], dtype=np.int32))
        io.write_attr(_META + "centroid_indices", stamp.centroid_indices)
        io.write_attr(_META + "wfn_path_bytes_i32", _text_i32(stamp.wfn_path))
        io.write_attr(_META + "wfn_fingerprint_bytes_i32",
                      _text_i32(stamp.wfn_fingerprint))
        io.write_attr(_META + "rank_multiplier_qrcp_eps",
                      np.asarray([stamp.rank_multiplier, stamp.qrcp_eps],
                                 dtype=np.float64))
        io.write_attr(_META + "qrcp_seed_raw_search",
                      np.asarray([stamp.qrcp_seed, basis.qrcp_raw_rank,
                                  basis.qrcp_search_rank], dtype=np.int64))
        io.write_attr(_META + "qrcp_rng_version_bytes_i32",
                      _text_i32(stamp.qrcp_rng_version))
        io.write_attr(_META + "selected_state_indices",
                      np.asarray(basis.selected_state_indices, dtype=np.int64))
        io.write_attr(_META + "candidate_hash_bytes_i32",
                      _text_i32(basis.candidate_hash))
        io.write_attr(_META + "pivot_hash_bytes_i32",
                      _text_i32(basis.pivot_hash))
        io.write_attr(_META + "complete", np.asarray([1], dtype=np.int32))


def _read_vector(io: SlabIO, name: str, *, dtype) -> np.ndarray:
    return np.asarray(io.read_small(_META + name, dtype=dtype)).reshape(-1)


def _read_stamp(io: SlabIO):
    try:
        complete = int(_read_vector(io, "complete", dtype=np.int32)[0])
    except Exception as exc:
        raise ValueError(
            "Galerkin basis artifact has no completion stamp; its payload "
            "is partial or it predates this format") from exc
    if complete != 1:
        raise ValueError(
            f"Galerkin basis artifact is incomplete (complete={complete})")
    version = int(_read_vector(io, "format_version", dtype=np.int32)[0])
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Galerkin basis format {version} != supported {FORMAT_VERSION}")
    rank = int(_read_vector(io, "rank_physical", dtype=np.int32)[0])
    band = _read_vector(io, "band_range", dtype=np.int32)
    extents = _read_vector(io, "nk_nb_nspinor", dtype=np.int32)
    fft_grid = _read_vector(io, "fft_grid", dtype=np.int32)
    kgrid = _read_vector(io, "kgrid", dtype=np.int32)
    bispinor = bool(_read_vector(io, "bispinor", dtype=np.int32)[0])
    centroids = np.asarray(io.read_small(
        _META + "centroid_indices", dtype=np.int32))
    path = _decode_text_i32(io.read_small(
        _META + "wfn_path_bytes_i32", dtype=np.int32))
    fingerprint = _decode_text_i32(io.read_small(
        _META + "wfn_fingerprint_bytes_i32", dtype=np.int32))
    fit = _read_vector(io, "rank_multiplier_qrcp_eps", dtype=np.float64)
    qrcp = _read_vector(io, "qrcp_seed_raw_search", dtype=np.int64)
    rng_version = _decode_text_i32(io.read_small(
        _META + "qrcp_rng_version_bytes_i32", dtype=np.int32))
    selected = _read_vector(io, "selected_state_indices", dtype=np.int64)
    candidate_hash = _decode_text_i32(io.read_small(
        _META + "candidate_hash_bytes_i32", dtype=np.int32))
    pivot_hash = _decode_text_i32(io.read_small(
        _META + "pivot_hash_bytes_i32", dtype=np.int32))
    if (band.size != 2 or extents.size != 3 or fft_grid.size != 3
            or kgrid.size != 3 or fit.size != 2 or qrcp.size != 3):
        raise ValueError("Galerkin basis metadata has malformed small extents")
    stamp = GalerkinBasisStamp(
        band_range=(int(band[0]), int(band[1])),
        nk=int(extents[0]), nb=int(extents[1]), nspinor=int(extents[2]),
        fft_grid=tuple(int(v) for v in fft_grid),
        kgrid=tuple(int(v) for v in kgrid), bispinor=bispinor,
        centroid_indices=centroids, wfn_path=path,
        wfn_fingerprint=fingerprint,
        rank_multiplier=float(fit[0]), qrcp_eps=float(fit[1]),
        qrcp_seed=int(qrcp[0]), qrcp_rng_version=rng_version,
    )
    if rank < 1 or selected.size != rank:
        raise ValueError(
            "Galerkin basis rank/selected-state metadata is inconsistent")
    if not _valid_sha256(candidate_hash) or not _valid_sha256(pivot_hash):
        raise ValueError("Galerkin basis QRCP hashes are malformed")
    if _selected_hash(selected) != pivot_hash:
        raise ValueError("Galerkin selected-state metadata fails pivot hash")
    return (stamp, rank, tuple(int(v) for v in selected),
            int(qrcp[1]), int(qrcp[2]), candidate_hash, pivot_hash)


def _carrier_rank(mesh_xy: Mesh, rank_physical: int,
                  extra_rank_pad: int) -> int:
    extra_rank_pad = int(extra_rank_pad)
    if extra_rank_pad < 0:
        raise ValueError(f"extra_rank_pad must be >= 0; got {extra_rank_pad}")
    align = math.lcm(int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"]))
    rank = round_up(int(rank_physical), align)
    if extra_rank_pad:
        rank = round_up(rank + extra_rank_pad, align)
    return rank


def read_galerkin_basis(
        path, *, mesh_xy: Mesh, expected: GalerkinBasisStamp,
        extra_rank_pad: int = 0,
) -> GalerkinBasis:
    """Collectively validate, read, and repad one complete basis artifact."""
    path = str(path)
    rep = NamedSharding(mesh_xy, P())
    with SlabIO(path, mode="r", mesh=mesh_xy) as io:
        (stored, rank_physical, selected, raw_rank, search_rank,
         candidate_hash, pivot_hash) = _read_stamp(io)
        mismatches = stored.mismatch(expected)
        if mismatches:
            raise ValueError(
                f"Galerkin basis artifact {path} does not match this fit: "
                + "; ".join(mismatches))
        ctilde = io.read_slab(CTILDE_DATASET, partition_spec=P())
        node_sharding = fit_sharding(
            mesh_xy, P(None, None, "y"),
            (rank_physical, stored.nspinor, stored.n_nodes),
            "galerkin.restart.basis_at_nodes(node-axis)")
        basis_nodes = io.read_slab(
            BASIS_DATASET, partition_spec=node_sharding.spec)
        factor = io.read_slab(SELECTION_FACTOR_DATASET, partition_spec=P())

    expected_ctilde = (stored.nk, stored.nb, rank_physical)
    expected_basis = (rank_physical, stored.nspinor, stored.n_nodes)
    expected_factor = (rank_physical, rank_physical)
    if tuple(ctilde.shape) != expected_ctilde:
        raise ValueError(
            f"Galerkin ctilde payload shape {tuple(ctilde.shape)} != stamp "
            f"{expected_ctilde}")
    if tuple(basis_nodes.shape) != expected_basis:
        raise ValueError(
            f"Galerkin node-basis payload shape {tuple(basis_nodes.shape)} != "
            f"stamp {expected_basis}")
    if tuple(factor.shape) != expected_factor:
        raise ValueError(
            f"Galerkin selection-factor shape {tuple(factor.shape)} != stamp "
            f"{expected_factor}")

    rank_carrier = _carrier_rank(mesh_xy, rank_physical, extra_rank_pad)
    if rank_carrier != rank_physical:
        ctilde = pad_axis(ctilde, rank_carrier, axis=2).array
        ctilde = jax.jit(lambda value: value, out_shardings=rep)(ctilde)
        carried_node_sharding = fit_sharding(
            mesh_xy, P(None, None, "y"),
            (rank_carrier, stored.nspinor, stored.n_nodes),
            "galerkin.restart.basis_at_nodes(node-axis)")
        basis_nodes = pad_axis(basis_nodes, rank_carrier, axis=0).array
        basis_nodes = jax.jit(
            lambda value: value,
            out_shardings=carried_node_sharding)(basis_nodes)
        factor = pad_axis(factor, rank_carrier, axis=0).array
        factor = pad_axis(factor, rank_carrier, axis=1).array
        padding_diag = jnp.concatenate([
            jnp.zeros(rank_physical, dtype=factor.dtype),
            jnp.ones(rank_carrier - rank_physical, dtype=factor.dtype)])
        factor = jax.jit(
            lambda value, diag: value + jnp.diag(diag),
            out_shardings=rep)(factor, padding_diag)

    basis = GalerkinBasis(
        ctilde=ctilde, basis_at_nodes=basis_nodes,
        rank_physical=rank_physical, band_range=stored.band_range,
        selected_state_indices=selected, selection_factor=factor,
        qrcp_seed=stored.qrcp_seed,
        qrcp_rng_version=stored.qrcp_rng_version,
        qrcp_eps=stored.qrcp_eps, qrcp_raw_rank=raw_rank,
        qrcp_search_rank=search_rank, candidate_hash=candidate_hash,
        pivot_hash=pivot_hash,
    )
    _assert_basis_matches_stamp(basis, stored)
    return basis
