"""SlabIO persistence for one gauge-coupled :class:`GalerkinBasis`.

Only logical extents are stored.  In particular, the alpha axis stops at
``rank_physical`` and the node axis stops at the exact centroid count; mesh
padding is reconstructed by :func:`read_galerkin_basis` for the reader's
current mesh.  ``ctilde``, ``basis_at_nodes`` and the optional projector are
one atomic artifact because independently replacing any member changes the
shared alpha gauge.

The write is two-phase.  One SlabIO transaction writes every large payload and
closes.  A second metadata-only transaction stamps provenance and
``complete=1`` last.  A crash between them leaves a payload file with no valid
completion record, which every reader refuses.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.parallel_transport import wfn_fingerprint
from common.sharding_fit import fit_sharding
from file_io.slab_io import SlabIO
from isdf.galerkin import GalerkinBasis
from runtime.padding import pad_axis, round_up


FORMAT_VERSION = 1
CTILDE_DATASET = "galerkin_ctilde"
BASIS_DATASET = "galerkin_basis_at_nodes"
PROJECTOR_DATASET = "galerkin_projector"
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


@dataclass(frozen=True)
class GalerkinBasisStamp:
    """Numerical identity required to reuse one fitted basis."""

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
    rtol: float
    rank_multiplier: float

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
        if not np.isfinite(float(self.rtol)) or float(self.rtol) <= 0.0:
            raise ValueError(f"GalerkinBasisStamp rtol={self.rtol!r} is invalid")
        if (not np.isfinite(float(self.rank_multiplier))
                or float(self.rank_multiplier) < 0.0):
            raise ValueError(
                "GalerkinBasisStamp rank_multiplier must be finite and >= 0")
        fingerprint = str(self.wfn_fingerprint)
        if len(fingerprint) != 64 or any(
                char not in "0123456789abcdef" for char in fingerprint):
            raise ValueError(
                "GalerkinBasisStamp requires the canonical lowercase "
                "SHA-256 WFN fingerprint")

    @property
    def n_nodes(self) -> int:
        return int(self.centroid_indices.shape[0])

    @classmethod
    def from_runtime(
            cls, *, wfn, meta, centroid_indices, band_range,
            bispinor: bool, rtol: float, rank_multiplier: float,
    ) -> "GalerkinBasisStamp":
        """Build the stamp using the canonical location-independent WFN ID."""
        b0, b1 = (int(v) for v in band_range)
        path = (getattr(wfn, "path", None)
                or getattr(wfn, "_filename", None) or "")
        return cls(
            band_range=(b0, b1),
            nk=int(meta.nk_tot),
            nb=b1 - b0,
            nspinor=int(meta.nspinor),
            fft_grid=tuple(int(v) for v in meta.fft_grid),
            kgrid=tuple(int(v) for v in meta.kgrid),
            bispinor=bool(bispinor),
            centroid_indices=np.asarray(centroid_indices, dtype=np.int32),
            wfn_path=os.path.realpath(str(path)) if path else "",
            wfn_fingerprint=wfn_fingerprint(wfn),
            rtol=float(rtol),
            rank_multiplier=float(rank_multiplier),
        )

    def mismatch(self, expected: "GalerkinBasisStamp") -> tuple[str, ...]:
        """Return every provenance difference, with exact centroid identity."""
        out = []
        scalar_fields = (
            "band_range", "nk", "nb", "nspinor", "fft_grid", "kgrid",
            "bispinor", "wfn_fingerprint", "rtol", "rank_multiplier",
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


def _assert_basis_matches_stamp(
        basis: GalerkinBasis, stamp: GalerkinBasisStamp) -> None:
    if tuple(int(v) for v in basis.band_range) != tuple(stamp.band_range):
        raise ValueError("Galerkin basis/stamp band ranges disagree")
    if int(basis.ctilde.shape[0]) != int(stamp.nk):
        raise ValueError("Galerkin basis/stamp nk extents disagree")
    if int(basis.ctilde.shape[1]) != int(stamp.nb):
        raise ValueError("Galerkin basis/stamp band extents disagree")
    if int(basis.basis_at_nodes.shape[1]) != int(stamp.nspinor):
        raise ValueError("Galerkin basis/stamp spinor extents disagree")
    if int(basis.basis_at_nodes.shape[2]) != int(stamp.n_nodes):
        raise ValueError("Galerkin basis/stamp node extents disagree")

    physical = int(basis.rank_physical)
    if physical < basis.rank_carrier:
        tails = [basis.ctilde[..., physical:],
                 basis.basis_at_nodes[physical:]]
        if basis.projector is not None:
            tails.append(basis.projector[physical:])
        for name, tail in zip(("ctilde", "basis_at_nodes", "projector"),
                              tails):
            if tail.size and float(jax.device_get(
                    jnp.max(jnp.abs(tail)))) != 0.0:
                raise ValueError(
                    f"Galerkin {name} carrier tail above rank_physical="
                    f"{physical} is not exact zero; trimming it would change "
                    "the fitted gauge")


def write_galerkin_basis(
        path, basis: GalerkinBasis, stamp: GalerkinBasisStamp,
        *, mesh_xy: Mesh) -> None:
    """Collectively publish one complete, mesh-independent basis artifact."""
    _assert_basis_matches_stamp(basis, stamp)
    path = str(path)
    rank = int(basis.rank_physical)
    ctilde = basis.ctilde[..., :rank]
    basis_nodes = basis.basis_at_nodes[:rank]
    projector = (None if basis.projector is None
                 else basis.projector[:rank])

    # Phase 1: all gauge-coupled bulk payloads, one collective handle.
    with SlabIO(path, mode="w", mesh=mesh_xy) as io:
        io.create_dataset(
            CTILDE_DATASET, shape=tuple(int(v) for v in ctilde.shape),
            dtype=ctilde.dtype)
        io.create_dataset(
            BASIS_DATASET, shape=tuple(int(v) for v in basis_nodes.shape),
            dtype=basis_nodes.dtype)
        io.write_slab(CTILDE_DATASET, ctilde)
        io.write_slab(BASIS_DATASET, basis_nodes)
        if projector is not None:
            io.create_dataset(
                PROJECTOR_DATASET,
                shape=tuple(int(v) for v in projector.shape),
                dtype=projector.dtype)
            io.write_slab(PROJECTOR_DATASET, projector)

    # Phase 2: small provenance, completion queued LAST. SlabIO preserves
    # deferred-write order when its collective handle has closed, so a crash
    # before the last dataset lands cannot leave a reusable partial stamp.
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
        io.write_attr(_META + "wfn_path_bytes_i32",
                      _text_i32(stamp.wfn_path))
        io.write_attr(_META + "wfn_fingerprint_bytes_i32",
                      _text_i32(stamp.wfn_fingerprint))
        io.write_attr(_META + "rtol_rank_multiplier",
                      np.asarray([stamp.rtol, stamp.rank_multiplier],
                                 dtype=np.float64))
        io.write_attr(_META + "projector_present",
                      np.asarray([projector is not None], dtype=np.int32))
        io.write_attr(_META + "complete", np.asarray([1], dtype=np.int32))


def _read_vector(io: SlabIO, name: str, *, dtype) -> np.ndarray:
    return np.asarray(io.read_small(_META + name, dtype=dtype)).reshape(-1)


def _read_stamp(io: SlabIO) -> tuple[GalerkinBasisStamp, int, bool]:
    try:
        complete = int(_read_vector(io, "complete", dtype=np.int32)[0])
    except Exception as exc:
        raise ValueError(
            "Galerkin basis artifact has no completion stamp; its payload "
            "is partial or it predates this format") from exc
    if complete != 1:
        raise ValueError(
            f"Galerkin basis artifact is incomplete (complete={complete})")
    try:
        version = int(_read_vector(
            io, "format_version", dtype=np.int32)[0])
    except Exception as exc:
        raise ValueError(
            "Galerkin basis artifact is marked complete but its format "
            "stamp is absent") from exc
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
    fit = _read_vector(io, "rtol_rank_multiplier", dtype=np.float64)
    projector = bool(_read_vector(
        io, "projector_present", dtype=np.int32)[0])
    if band.size != 2 or extents.size != 3 or fft_grid.size != 3 \
            or kgrid.size != 3 or fit.size != 2:
        raise ValueError("Galerkin basis metadata has malformed small extents")
    stamp = GalerkinBasisStamp(
        band_range=(int(band[0]), int(band[1])),
        nk=int(extents[0]), nb=int(extents[1]), nspinor=int(extents[2]),
        fft_grid=tuple(int(v) for v in fft_grid),
        kgrid=tuple(int(v) for v in kgrid),
        bispinor=bispinor,
        centroid_indices=centroids,
        wfn_path=path,
        wfn_fingerprint=fingerprint,
        rtol=float(fit[0]), rank_multiplier=float(fit[1]),
    )
    if rank < 1:
        raise ValueError(f"Galerkin basis rank_physical={rank} is invalid")
    return stamp, rank, projector


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
        require_projector: bool = False, extra_rank_pad: int = 0,
) -> GalerkinBasis:
    """Collectively validate, read, and repad one complete basis artifact."""
    path = str(path)
    rep = NamedSharding(mesh_xy, P())
    with SlabIO(path, mode="r", mesh=mesh_xy) as io:
        stored, rank_physical, has_projector = _read_stamp(io)
        mismatches = stored.mismatch(expected)
        if mismatches:
            raise ValueError(
                f"Galerkin basis artifact {path} does not match this fit: "
                + "; ".join(mismatches))
        if require_projector and not has_projector:
            raise ValueError(
                f"Galerkin basis artifact {path} has no projector, but this "
                "consumer requires the same-gauge full-r projector")

        ctilde = io.read_slab(CTILDE_DATASET, partition_spec=P())
        node_sharding = fit_sharding(
            mesh_xy, P(None, None, "y"),
            (rank_physical, stored.nspinor, stored.n_nodes),
            "galerkin.restart.basis_at_nodes(node-axis)")
        basis_nodes = io.read_slab(
            BASIS_DATASET, partition_spec=node_sharding.spec)
        projector = (io.read_slab(PROJECTOR_DATASET, partition_spec=P())
                     if has_projector else None)

    expected_ctilde = (stored.nk, stored.nb, rank_physical)
    expected_basis = (rank_physical, stored.nspinor, stored.n_nodes)
    if tuple(ctilde.shape) != expected_ctilde:
        raise ValueError(
            f"Galerkin ctilde payload shape {tuple(ctilde.shape)} != stamp "
            f"{expected_ctilde}")
    if tuple(basis_nodes.shape) != expected_basis:
        raise ValueError(
            f"Galerkin node-basis payload shape {tuple(basis_nodes.shape)} != "
            f"stamp {expected_basis}")
    expected_projector = (rank_physical, stored.nk * stored.nb)
    if projector is not None and tuple(projector.shape) != expected_projector:
        raise ValueError(
            f"Galerkin projector payload shape {tuple(projector.shape)} != "
            f"stamp {expected_projector}")

    rank_carrier = _carrier_rank(mesh_xy, rank_physical, extra_rank_pad)
    if rank_carrier != rank_physical:
        ctilde = pad_axis(ctilde, rank_carrier, axis=2).array
        ctilde = jax.jit(lambda value: value, out_shardings=rep)(ctilde)
        carried_node_sharding = fit_sharding(
            mesh_xy, P(None, None, "y"),
            (rank_carrier, stored.nspinor, stored.n_nodes),
            "galerkin.restart.basis_at_nodes(node-axis)")
        basis_nodes = pad_axis(
            basis_nodes, rank_carrier, axis=0).array
        basis_nodes = jax.jit(
            lambda value: value,
            out_shardings=carried_node_sharding)(basis_nodes)
        if projector is not None:
            projector = pad_axis(
                projector, rank_carrier, axis=0).array
            projector = jax.jit(
                lambda value: value, out_shardings=rep)(projector)

    return GalerkinBasis(
        ctilde=ctilde,
        basis_at_nodes=basis_nodes,
        projector=projector,
        rank_physical=rank_physical,
        band_range=stored.band_range,
    )
