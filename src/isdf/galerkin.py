"""Whole-state Hamiltonian-transform basis on a two-dimensional mesh.

This module owns the randomized-QRCP-equivalent selection, exact global
basis factorization, and physical wavefunction projection used by
htransform.  Every real-space stage streams bounded full-Bloch slabs from the
canonical WFN source; no full-grid basis or random matrix is materialized.

Wavefunction loading and G-to-r transforms remain owned by the reusable
``common.psi_G_store`` source and its canonical transform helpers.  The caller
resolves policy such as environment overrides and the device-pool limit, then
passes explicit values here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg as jsp_linalg
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.gpu_utils import bfc_fragmentation_target_utilization
from common.collectives import device_put_process_local
from common.psi_G_store import build_psi_G_store
from common.pivoted_cholesky import make_sharded_pivoted_cholesky_select
from common.shard_map import shard_map
from common.sharding_fit import fit_sharding as _fit
from common.wfn_layout import band_sphere_spec
from common.wfn_transforms import (
    FULL_BLOCH_TRANSFORM_SCHEME,
    gflat_to_rchunk_aot_memory,
    load_centroids_band_chunked,
)
from distrib_la import plan as linalg_plan
from runtime.padding import spec_divisor


__all__ = [
    "GalerkinBasis",
    "GalerkinBasisMismatch",
    "GalerkinOperatorProjection",
    "GalerkinStateExpectation",
    "GalerkinStreamPlan",
    "QRCP_RNG_VERSION",
    "fit_galerkin_basis",
    "galerkin_rank_record",
    "iter_galerkin_rchunks",
    "plan_galerkin_stream",
    "plan_galerkin_operator_stream",
    "project_galerkin_spin_operator",
    "project_galerkin_spin_z",
    "read_galerkin_basis",
    "rotate_galerkin_operator",
    "validate_rank_multiplier",
    "write_galerkin_basis",
]


QRCP_RNG_VERSION = "jax-fold-in-global-r-rows-spin-v1"
_BASIS_FORMAT = 2
_BASIS_ARRAYS = ("galerkin_ctilde", "galerkin_basis_at_nodes",
                 "galerkin_selection_factor")
_BASIS_META = "galerkin_"


class GalerkinBasisMismatch(ValueError):
    """A complete basis artifact fitted for other inputs or another format.

    Raised only by the provenance/format comparison, so a caller may refit
    on it; a corrupt or incomplete artifact still raises plain ValueError.
    """


@dataclass(frozen=True)
class GalerkinBasis:
    """One fitted interpolation basis in a single shared alpha gauge.

    ``ctilde`` and ``basis_at_nodes`` are inseparable: independently replacing
    either array changes the gauge and invalidates reconstruction.  The compact
    selected-state factor ``(selected_state_indices, selection_factor)`` is the
    source for physical basis rows away from the registered nodes.  It replaces
    the former dense ``(rank, nk*nb)`` projector; callers obtain bounded basis
    rows with :func:`iter_galerkin_rchunks` instead.  ``rank_physical`` excludes
    exact-null mesh padding, so persistence can be mesh-independent and a
    reader can reconstruct the carrier required by its own mesh.
    """

    ctilde: jax.Array
    basis_at_nodes: jax.Array
    rank_physical: int
    band_range: tuple[int, int]
    selected_state_indices: tuple[int, ...]
    selection_factor: jax.Array
    qrcp_seed: int = 0
    qrcp_rng_version: str = QRCP_RNG_VERSION
    qrcp_eps: float = 1.0e-3
    qrcp_raw_rank: int = 0
    qrcp_search_rank: int = 0
    candidate_hash: str = ""
    pivot_hash: str = ""

    def __post_init__(self) -> None:
        if self.ctilde.ndim != 3:
            raise ValueError(
                f"GalerkinBasis.ctilde must be (nk, nb, rank); got "
                f"{tuple(self.ctilde.shape)}")
        if self.basis_at_nodes.ndim != 3:
            raise ValueError(
                "GalerkinBasis.basis_at_nodes must be (rank, ns, n_nodes); "
                f"got {tuple(self.basis_at_nodes.shape)}")
        rank = int(self.ctilde.shape[2])
        if int(self.basis_at_nodes.shape[0]) != rank:
            raise ValueError(
                "GalerkinBasis gauge mismatch: ctilde rank "
                f"{rank} != basis_at_nodes rank "
                f"{int(self.basis_at_nodes.shape[0])}")
        if not 0 < int(self.rank_physical) <= rank:
            raise ValueError(
                f"GalerkinBasis.rank_physical={self.rank_physical} must lie "
                f"in [1, carried rank {rank}]")
        b0, b1 = (int(v) for v in self.band_range)
        if b1 - b0 != int(self.ctilde.shape[1]):
            raise ValueError(
                f"GalerkinBasis band range [{b0},{b1}) has width {b1-b0}, "
                f"but ctilde carries {int(self.ctilde.shape[1])} bands")
        if len(self.selected_state_indices) != int(self.rank_physical):
            raise ValueError(
                "GalerkinBasis selected-state count must equal the physical "
                f"rank; got {len(self.selected_state_indices)} and "
                f"{self.rank_physical}")
        expected = (rank, rank)
        if tuple(self.selection_factor.shape) != expected:
            raise ValueError(
                "GalerkinBasis.selection_factor must have shape "
                f"{expected}; got {tuple(self.selection_factor.shape)}")

    @property
    def rank_carrier(self) -> int:
        return int(self.ctilde.shape[2])


@dataclass(frozen=True)
class GalerkinOperatorProjection:
    """One spatially uniform spin operator in the fitted alpha basis.

    Both matrices have shape ``(rank_carrier, rank_carrier)`` and remain on
    the all-mesh ``P('x','y')`` face.  ``metric`` includes the physical
    full-grid overlap rather than assuming identity; exact-null carrier rows
    therefore remain exact zero.
    """

    operator: jax.Array
    metric: jax.Array


@dataclass(frozen=True)
class GalerkinStateExpectation:
    """A projected operator evaluated in selected htransform states."""

    value: jax.Array
    numerator: jax.Array
    norm: jax.Array


def validate_rank_multiplier(value, *, name: str = "rank_multiplier") -> float:
    """Validate the whole-state QRCP search ceiling multiplier.

    ``0`` is retained as an input-compatibility spelling of the published
    default ``20``; there is no exact-span alternate route.
    """
    try:
        multiplier = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name}={value!r} is not a finite number; use 20 for the "
            "published search or another value >= 1.") from None
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError(f"{name}={value!r} must be finite and >= 0.")
    if multiplier == 0.0:
        return 20.0
    if multiplier < 1.0:
        raise ValueError(
            f"{name}={multiplier:g} would retain fewer directions than bands "
            "at one k. Use the published default 20 or another value >= 1.")
    return multiplier


@partial(jax.jit, static_argnames=("selected_rows", "physical"))
def _galerkin_rank_metrics(ctilde, selection_factor, *,
                           selected_rows: tuple[int, ...], physical: int):
    """Reduce one fitted basis to the scalar numerical rank receipts."""
    nk, nb, carrier = ctilde.shape
    gram = jnp.einsum(
        "kna,kma->knm", ctilde, jnp.conj(ctilde), optimize=True)
    eye = jnp.eye(nb, dtype=ctilde.dtype)[None]
    row_norm = jnp.real(jnp.diagonal(gram, axis1=1, axis2=2))
    reference = selection_factor[:physical]
    selected = jnp.asarray(selected_rows, dtype=jnp.int32)
    picked = ctilde.reshape(nk * nb, carrier)[selected]
    selected_scale = jnp.maximum(1.0, jnp.max(jnp.abs(reference)))
    return (
        jnp.min(jnp.real(jnp.diag(selection_factor))[:physical]),
        jnp.max(jnp.abs(gram - eye)),
        jnp.max(jnp.abs(row_norm - 1.0)),
        jnp.sqrt(jnp.maximum(0.0, 1.0 - jnp.mean(row_norm))),
        jnp.max(jnp.abs(picked - reference)),
        selected_scale,
    )


def galerkin_rank_record(basis: GalerkinBasis, *, meta,
                         rank_multiplier: float) -> dict:
    """Return the canonical QRCP receipt for a fitted or restored basis.

    The basis already owns every physical and provenance quantity in this
    record.  Numerical diagnostics reduce its incumbent device arrays to six
    scalars; no coefficient or factor payload is gathered or rebuilt.
    """
    nk, nb, carrier = (int(n) for n in basis.ctilde.shape)
    physical = int(basis.rank_physical)
    stacked_states = nk * nb
    state_dimension = int(meta.nspinor) * int(meta.n_rtot)
    multiplier = validate_rank_multiplier(
        rank_multiplier, name="htransform_rank_multiplier")
    metrics = _galerkin_rank_metrics(
        basis.ctilde, basis.selection_factor,
        selected_rows=basis.selected_state_indices, physical=physical)
    metrics = jax.block_until_ready(metrics)
    (min_chol_diag, c_ortho, max_missing_norm2, fro_resid,
     selected_err, selected_scale) = (float(value) for value in metrics)
    selected_tol = np.sqrt(np.finfo(np.float64).eps) * selected_scale
    if not np.isfinite(selected_err) or selected_err > selected_tol:
        raise ValueError(
            "Galerkin basis selected-state orientation identity "
            f"C[selected]=L failed: max error {selected_err:.3e} "
            f"> sqrt(eps)*scale={selected_tol:.3e}")

    search_rank = int(basis.qrcp_search_rank)
    return {
        "method": "whole_state_randomized_qrcp",
        "stacked_states": stacked_states,
        "state_dimension": state_dimension,
        "search_rank": search_rank,
        "candidate_count": min(int(1.5 * search_rank), stacked_states),
        "raw_rank": int(basis.qrcp_raw_rank),
        "retained_rank": physical,
        "carried_rank": carrier,
        "null_padding": carrier - physical,
        "rank_multiplier": multiplier,
        "qr_eps": float(basis.qrcp_eps),
        "qrcp_seed": int(basis.qrcp_seed),
        "qrcp_rng_version": basis.qrcp_rng_version,
        "candidate_hash": basis.candidate_hash,
        "pivot_hash": basis.pivot_hash,
        "min_cholesky_diagonal": min_chol_diag,
        "coefficient_orthogonality_error": c_ortho,
        "max_missing_state_norm_squared": max_missing_norm2,
        "relative_frobenius_residual": fro_resid,
        "selected_orientation_error": selected_err,
        "selected_orientation_tolerance": selected_tol,
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
                      bispinor, rank_multiplier, qrcp_eps, qrcp_seed) -> dict:
    from common.parallel_transport import (
        WFN_FINGERPRINT_SCHEME, wfn_fingerprint)

    b0, b1 = (int(n) for n in band_range)
    shape, centroid_hash = _basis_digest(centroid_indices)
    qrcp_eps, qrcp_seed = float(qrcp_eps), int(qrcp_seed)
    if b1 <= b0:
        raise ValueError("Galerkin basis band window is empty")
    if not 0.0 < qrcp_eps < 1.0:
        raise ValueError("htransform_qr_eps must lie in (0,1)")
    if not 0 <= qrcp_seed <= np.iinfo(np.uint32).max:
        raise ValueError("htransform_qrcp_seed must fit uint32")
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
        "qrcp_eps": qrcp_eps, "qrcp_seed": qrcp_seed,
        "qrcp_rng": QRCP_RNG_VERSION,
    }


def _basis_check(basis: GalerkinBasis, provenance: dict) -> None:
    physical, carrier = int(basis.rank_physical), int(basis.rank_carrier)
    if tuple(basis.band_range) != provenance["band_range"] \
            or tuple(basis.ctilde.shape[:2]) != (
                provenance["nk"], provenance["nb"]) \
            or tuple(basis.basis_at_nodes.shape[1:]) != (
                provenance["nspinor"], provenance["centroid_shape"][0]):
        raise ValueError("Galerkin basis payload disagrees with provenance")
    if (basis.qrcp_seed != provenance["qrcp_seed"]
            or basis.qrcp_eps != provenance["qrcp_eps"]
            or basis.qrcp_rng_version != provenance["qrcp_rng"]):
        raise ValueError("Galerkin basis QRCP controls disagree with provenance")
    picked = np.asarray(basis.selected_state_indices, dtype="<i8")
    if (np.any(picked < 0)
            or np.any(picked >= provenance["nk"] * provenance["nb"])
            or np.unique(picked).size != physical):
        raise ValueError("Galerkin selected-state indices are invalid")
    # This is byte-for-byte the fit owner's existing pivot receipt:
    # selected.astype('<i8', copy=False).tobytes().
    if hashlib.sha256(picked.tobytes()).hexdigest() != basis.pivot_hash:
        raise ValueError("Galerkin selected-state indices/pivot hash disagree")
    if any(len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value)
            for value in (basis.candidate_hash, basis.pivot_hash)):
        raise ValueError("Galerkin basis hashes are malformed")
    if physical == carrier:
        return
    tails = (basis.ctilde[..., physical:],
             basis.basis_at_nodes[physical:],
             basis.selection_factor[physical:, :physical],
             basis.selection_factor[:physical, physical:])
    errors = [float(jnp.max(jnp.abs(value))) for value in tails if value.size]
    factor_tail = basis.selection_factor[physical:, physical:]
    errors.append(float(jnp.max(jnp.abs(
        factor_tail - jnp.eye(carrier - physical, dtype=factor_tail.dtype)))))
    if any(error != 0.0 for error in errors):
        raise ValueError("Galerkin basis carrier is not exact-null/identity")


def _basis_write_meta(io, basis: GalerkinBasis, provenance: dict) -> None:
    integer = {
        "format": [_BASIS_FORMAT], "rank": [basis.rank_physical],
        "band_range": provenance["band_range"],
        "nk_nb_nspinor": [provenance[k] for k in ("nk", "nb", "nspinor")],
        "fft_grid": provenance["fft_grid"], "kgrid": provenance["kgrid"],
        "bispinor": [provenance["bispinor"]],
        "centroid_shape": provenance["centroid_shape"],
        "qrcp_raw_search": [basis.qrcp_raw_rank, basis.qrcp_search_rank],
        "selected": basis.selected_state_indices,
    }
    for name, value in integer.items():
        io.write_attr(_BASIS_META + name, np.asarray(value, dtype=np.int32))
    io.write_attr(_BASIS_META + "qrcp_controls", np.asarray([
        provenance["rank_multiplier"], provenance["qrcp_eps"]],
        dtype=np.float64))
    io.write_attr(_BASIS_META + "qrcp_seed", np.asarray(
        [provenance["qrcp_seed"]], dtype=np.int64))
    for name, value in (
            ("centroid_hash", provenance["centroid_hash"]),
            ("wfn_hash", provenance["wfn_hash"]),
            ("wfn_scheme", provenance["wfn_scheme"]),
            ("transform_scheme", provenance["transform_scheme"]),
            ("qrcp_rng", provenance["qrcp_rng"]),
            ("candidate_hash", basis.candidate_hash),
            ("pivot_hash", basis.pivot_hash)):
        io.write_attr(_BASIS_META + name, _basis_text(value))
    # SlabIO's write_attr queues a small dataset for the rank-0 reopen after
    # collective close; read_small is its matching collective reader.
    io.write_attr(_BASIS_META + "complete", np.asarray([1], dtype=np.int32))


def write_galerkin_basis(path, basis: GalerkinBasis, *, wfn, meta,
                         centroid_indices, bispinor, rank_multiplier,
                         qrcp_eps, qrcp_seed, mesh_xy: Mesh) -> None:
    """Collectively publish one immutable mesh-neutral basis artifact."""
    from common.collectives import barrier, process_count, process_rank
    from file_io.slab_io import SlabIO

    provenance = _basis_provenance(
        wfn=wfn, meta=meta, centroid_indices=centroid_indices,
        band_range=basis.band_range, bispinor=bispinor,
        rank_multiplier=rank_multiplier, qrcp_eps=qrcp_eps,
        qrcp_seed=qrcp_seed)
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
    # These are incumbent shardings: ctilde and selection_factor are P();
    # basis_at_nodes retains its node-axis P(None,None,'y').  SlabIO consumes
    # them directly, so no bulk payload is materialized on a host/process.
    values = (basis.ctilde[..., :physical], basis.basis_at_nodes[:physical],
              basis.selection_factor[:physical, :physical])
    with SlabIO(staging, mode="w", mesh=mesh_xy) as io:
        for name, value in zip(_BASIS_ARRAYS, values):
            io.create_dataset(name, shape=value.shape, dtype=value.dtype)
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
            f"Galerkin basis format {version} is unsupported")
    band = small("band_range", np.int32)
    extents = small("nk_nb_nspinor", np.int32)
    controls = small("qrcp_controls", np.float64)
    qrcp = small("qrcp_raw_search", np.int32)
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
        "qrcp_seed": int(small("qrcp_seed", np.int64)[0]),
        "qrcp_rng": text("qrcp_rng"),
        "qrcp_raw_rank": int(qrcp[0]), "qrcp_search_rank": int(qrcp[1]),
        "selected": tuple(int(n) for n in small("selected", np.int32)),
        "candidate_hash": text("candidate_hash"), "pivot_hash": text("pivot_hash"),
    }


def read_galerkin_basis(path, *, wfn, meta, centroid_indices, band_range,
                        bispinor, rank_multiplier, qrcp_eps, qrcp_seed,
                        mesh_xy: Mesh, extra_rank_pad: int = 0) -> GalerkinBasis:
    """Collectively validate and read a basis on the caller's target mesh."""
    from file_io.slab_io import SlabIO

    expected = _basis_provenance(
        wfn=wfn, meta=meta, centroid_indices=centroid_indices,
        band_range=band_range, bispinor=bispinor,
        rank_multiplier=rank_multiplier, qrcp_eps=qrcp_eps,
        qrcp_seed=qrcp_seed)
    extra_rank_pad = int(extra_rank_pad)
    if extra_rank_pad < 0:
        raise ValueError("extra_rank_pad must be non-negative")
    with SlabIO(os.path.abspath(os.fspath(path)), mode="r", mesh=mesh_xy) as io:
        stored = _basis_read_meta(io)
        mismatches = [key for key in expected if stored.get(key) != expected[key]]
        if mismatches:
            raise GalerkinBasisMismatch(
                "Galerkin basis provenance mismatch: " + ", ".join(mismatches))
        physical = stored["rank"]
        from runtime.padding import padded_axis
        rank_axis = padded_axis(
            physical, mesh_xy, name="Galerkin restart rank carrier",
            specs=((P("x", None), 0), (P(None, "y"), 1)),
            extra=extra_rank_pad)
        carrier = rank_axis.carrier
        shapes = ((stored["nk"], stored["nb"], carrier),
                  (carrier, stored["nspinor"], stored["centroid_shape"][0]),
                  (carrier, carrier))
        node_spec = _fit(mesh_xy, P(None, None, "y"), shapes[1],
                         "galerkin.restart.basis_at_nodes").spec
        # SlabIO's registered shape>dataset contract zero-extends the physical
        # logical datasets directly into these mesh-legal carrier shapes.
        arrays = [io.read_slab(name, shape=shape, partition_spec=spec)
                  for name, shape, spec in zip(
                      _BASIS_ARRAYS, shapes, (P(), node_spec, P()))]
    rep = NamedSharding(mesh_xy, P())
    arrays[2] = jax.jit(
        lambda factor: factor + jnp.diag(
            (jnp.arange(carrier) >= physical).astype(factor.dtype)),
        out_shardings=rep)(arrays[2])
    basis = GalerkinBasis(
        ctilde=arrays[0], basis_at_nodes=arrays[1], rank_physical=physical,
        band_range=stored["band_range"], selected_state_indices=stored["selected"],
        selection_factor=arrays[2], qrcp_seed=stored["qrcp_seed"],
        qrcp_rng_version=stored["qrcp_rng"], qrcp_eps=stored["qrcp_eps"],
        qrcp_raw_rank=stored["qrcp_raw_rank"],
        qrcp_search_rank=stored["qrcp_search_rank"],
        candidate_hash=stored["candidate_hash"], pivot_hash=stored["pivot_hash"])
    _basis_check(basis, expected)
    return basis


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
                + nk * (n_chunks + 1) * band_carrier * rank * _C16)
    return (resident + geom["g_index"] + nk * bpd * ns * n_r * _C16
            + max(int(k_tile) * bpd * geom["row_fft"],
                  2 * nk * bpd * ns * local_cols * p * _C16))


def _largest_fit(start: int, fits) -> int:
    """Largest value reached by halving ``start`` for which ``fits`` holds."""
    value = int(start)
    while value >= 1 and not fits(value):
        value //= 2
    return value


def _plan_rows_pass(geom: dict, *, rows: int, omega_rows: int,
                    resident: float, capacity: float, name: str):
    """Fewest state groups, then fewest r chunks, then widest FFT batch."""
    full_cols = -(-geom["n_rtot"] // geom["p"])
    groups = 1
    while True:
        per = -(-int(rows) // groups)
        cols = _largest_fit(full_cols, lambda c: _rows_pass_bytes(
            geom, rows=per, fft_rows=1, local_cols=c,
            omega_rows=omega_rows, resident=resident) <= capacity)
        if cols >= 1:
            break
        if per <= 1:
            raise MemoryError(
                f"fit_galerkin_basis: {name} does not fit even one state "
                f"per device: {_rows_pass_bytes(geom, rows=1, fft_rows=1, local_cols=1, omega_rows=omega_rows, resident=resident)/2**30:.2f}"
                f" GiB/device against {capacity/2**30:.2f} GiB/device")
        groups *= 2
    fft = _largest_fit(per, lambda f: f * geom["row_cufft"] <= geom[
        "reserve"] and _rows_pass_bytes(
        geom, rows=-(-per // f) * f, fft_rows=f, local_cols=cols,
        omega_rows=omega_rows, resident=resident) <= capacity)
    fft = max(1, fft)
    live = _rows_pass_bytes(geom, rows=-(-per // fft) * fft, fft_rows=fft,
                            local_cols=cols, omega_rows=omega_rows,
                            resident=resident)
    return (groups, fft,
            _r_stream(geom["n_rtot"], geom["p"], cols), live)


def _plan_basis_passes(geom: dict, *, band_carrier: int, rank: int,
                       rows: int, capacity: float):
    """One shared r schedule for the selected-row pass and the projection.

    The resident ``X`` chunks built by the first are the operand of the
    second, so both use one schedule; ``k_tile`` (a divisor of ``nk``) is
    then the widest projection FFT batch that fits.
    """
    nk, p, ns = geom["nk"], geom["p"], geom["ns"]
    full_cols = -(-geom["n_rtot"] // p)
    x_resident = rank * ns * full_cols * _C16 + rank * rank * _C16
    cols = _largest_fit(full_cols, lambda c: _projection_bytes(
        geom, band_carrier=band_carrier, rank=rank, local_cols=c,
        k_tile=1) <= capacity)
    if cols < 1:
        raise MemoryError(
            "fit_galerkin_basis: the all-states projection does not fit at "
            f"band carrier {band_carrier}, rank {rank}: "
            f"{_projection_bytes(geom, band_carrier=band_carrier, rank=rank, local_cols=1, k_tile=1)/2**30:.2f}"
            f" GiB/device against {capacity/2**30:.2f} GiB/device")
    groups = 1
    while True:
        per = -(-int(rows) // groups)
        x_cols = _largest_fit(cols, lambda c: _rows_pass_bytes(
            geom, rows=per, fft_rows=1, local_cols=c, omega_rows=0,
            resident=x_resident) <= capacity)
        if x_cols == cols or per <= 1:
            break
        groups *= 2
    if x_cols < 1:
        raise MemoryError(
            "fit_galerkin_basis: the selected-row pass does not fit beside "
            f"its resident rows ({x_resident/2**30:.2f} GiB/device) against "
            f"{capacity/2**30:.2f} GiB/device")
    cols = x_cols
    bpd = int(band_carrier) // p
    k_tile = max(d for d in range(1, nk + 1) if nk % d == 0 and (
        d == 1 or (d * bpd * geom["row_cufft"] <= geom["reserve"]
                   and _projection_bytes(
                       geom, band_carrier=band_carrier, rank=rank,
                       local_cols=cols, k_tile=d) <= capacity)))
    fft = max(1, _largest_fit(per, lambda f: f * geom["row_cufft"] <= geom[
        "reserve"] and _rows_pass_bytes(
        geom, rows=-(-per // f) * f, fft_rows=f, local_cols=cols,
        omega_rows=0, resident=x_resident) <= capacity))
    live = {
        "selected_rows": _rows_pass_bytes(
            geom, rows=-(-per // fft) * fft, fft_rows=fft, local_cols=cols,
            omega_rows=0, resident=x_resident),
        "projection": _projection_bytes(
            geom, band_carrier=band_carrier, rank=rank, local_cols=cols,
            k_tile=k_tile),
    }
    return _r_stream(geom["n_rtot"], p, cols), k_tile, groups, fft, live


def _whole_state_geometry(*, meta, mesh_xy: Mesh, nk: int, nspinor: int,
                          ngkmax: int, band_divisor: int, band_range,
                          device_pool_limit: float | None):
    """Fixed sizes, the measured one-row transform, and the stage target.

    The one-row price is the compiled canonical G-flat -> full-grid program
    (gather, ``ifftn(norm='ortho')``, Bloch phase, output) including its
    cuFFT workspace, so every FFT batch below is priced by measurement; a
    batch's cuFFT workspace (rows x the one-row workspace) must also fit the
    contiguous reserve outside the stage target.
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
        row_fft=float(memory.total), row_cufft=float(memory.cufft_scratch),
        reserve=math.inf,
        n_band_chunks=lambda carrier: -(-(b1 - b0) // int(carrier)))
    if device_pool_limit is None or device_pool_limit <= 0:
        return geom, math.inf, memory
    capacity = (float(device_pool_limit)
                * bfc_fragmentation_target_utilization(nspinor))
    reserve = geom["reserve"] = float(device_pool_limit) - capacity
    if float(memory.cufft_scratch) > reserve:
        raise MemoryError(
            "fit_galerkin_basis: the canonical one-row full-grid transform "
            f"needs a {memory.cufft_scratch/2**30:.2f} GiB/device cuFFT "
            f"workspace, above the contiguous BFC reserve "
            f"{reserve/2**30:.2f} GiB/device")
    return geom, capacity, memory


def fit_galerkin_basis(
        wfn, sym, meta, centroid_indices, mesh_xy: Mesh,
        band_range: tuple[int, int], *,
        log_fn=None,
        band_chunk_size: int = 64,
        bispinor: bool = False,
        rank_multiplier: float = 20.0,
        qr_eps: float = 1.0e-3,
        qrcp_seed: int = 0,
        device_pool_limit: float | None,
        extra_rank_pad: int = 0,
        progress_fn=None,
        rank_record_fn=None,
        distrib_la_batched_route: str = "batch_reshard",
) -> GalerkinBasis:
    """Fit the published whole-state Hamiltonian-transform basis.

    The stacked full-Bloch states ``Psi[(k,n),(s,r)]`` are the only basis
    source.  A deterministic Gaussian sketch and pivoted Cholesky of its
    candidate Gram reproduce randomized QRCP's column selection; the chosen
    *physical* states ``X`` then define one global orthonormal basis

    ``X X^H = L L^H,  B = L^-1 X,  C = Psi B^H = (Psi X^H) L^-H``.

    Each pass transforms only the rows it uses, and each row once: the
    sketch transforms the candidate states, the selected-row pass the
    pivots (``X`` stays resident, r-sharded), and the projection streams
    every state once for ``Psi X^H``.  ``B`` is never materialized over the
    full FFT grid.  Centroids are used only to evaluate ``B(r_mu)`` after
    the global basis has been selected.  No centroid weighting, state-space
    SVD, or per-k gauge repair participates in basis construction.

    ``qr_eps`` is the sole rank-revealing tolerance.  Every stream size comes
    from the live-set planner against ``device_pool_limit``.
    """
    del sym
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    b_start, b_end = (int(v) for v in band_range)
    nb = b_end - b_start
    nk = int(meta.nk_tot)
    nspinor = int(meta.nspinor)
    n_rtot = int(meta.n_rtot)
    n_mu = int(np.asarray(centroid_indices).shape[0])
    if nb <= 0:
        raise ValueError(
            f"fit_galerkin_basis: empty band range [{b_start},{b_end})")
    if not (0.0 < float(qr_eps) < 1.0):
        raise ValueError(
            f"fit_galerkin_basis: qr_eps={qr_eps!r} must lie in (0,1)")
    try:
        qrcp_seed = int(qrcp_seed)
    except (TypeError, ValueError):
        raise ValueError(
            f"fit_galerkin_basis: qrcp_seed={qrcp_seed!r} is not an integer") \
            from None
    if not 0 <= qrcp_seed <= np.iinfo(np.uint32).max:
        raise ValueError(
            f"fit_galerkin_basis: qrcp_seed={qrcp_seed} must fit uint32")
    if n_rtot > np.iinfo(np.uint32).max:
        raise ValueError(
            "fit_galerkin_basis: the stateless QRCP sketch indexes global "
            f"r in uint32, but n_rtot={n_rtot} exceeds that range")
    search_multiplier = validate_rank_multiplier(
        rank_multiplier, name="htransform_rank_multiplier")
    extra_rank_pad = int(extra_rank_pad)
    if extra_rank_pad < 0:
        raise ValueError(
            f"fit_galerkin_basis: extra_rank_pad={extra_rank_pad} must be >=0")

    m_states = nk * nb
    state_dim = nspinor * n_rtot
    max_search = min(
        int(math.ceil(search_multiplier * nb)), state_dim, m_states)
    n_candidates = min(int(1.5 * max_search), m_states)
    if max_search < nb:
        raise ValueError(
            "fit_galerkin_basis: the QRCP search ceiling carries fewer "
            f"directions ({max_search}) than bands at one k ({nb})")

    rng = np.random.default_rng(qrcp_seed)
    candidates = np.asarray(
        rng.permutation(m_states)[:n_candidates], dtype=np.int64)
    candidate_hash = hashlib.sha256(
        candidates.astype("<i8", copy=False).tobytes()).hexdigest()

    from runtime.padding import padded_axis
    candidate_carrier = padded_axis(
        n_candidates, mesh_xy, name="Galerkin candidate carrier").carrier
    p_band = padded_axis(
        1, mesh_xy, name="Galerkin band divisor",
        spec=band_sphere_spec(), axis=1).divisor
    bc_hint = max(p_band, min(int(band_chunk_size), nb))
    band_axis = padded_axis(
        bc_hint, mesh_xy, name="Galerkin band-chunk carrier",
        spec=band_sphere_spec(), axis=1)
    p_band = band_axis.divisor
    bc_carrier = band_axis.carrier
    geom, capacity, row_memory = _whole_state_geometry(
        meta=meta, mesh_xy=mesh_xy, nk=nk, nspinor=nspinor,
        ngkmax=int(wfn.ngkmax), band_divisor=p_band,
        band_range=(b_start, b_end), device_pool_limit=device_pool_limit)
    # The band carrier bounds the one live full-grid band-chunk slab of the
    # projection stream; the selected rows are priced after the pivots.
    while True:
        try:
            _plan_basis_passes(geom, band_carrier=bc_carrier, rank=0,
                               rows=1, capacity=capacity)
            break
        except MemoryError as exc:
            if bc_carrier <= p_band:
                raise
            next_carrier = max(
                p_band, (bc_carrier // (2 * p_band)) * p_band)
            if next_carrier >= bc_carrier:
                next_carrier = bc_carrier - p_band
            log_fn(
                f"  Whole-state planner reduces the canonical WFN band "
                f"carrier {bc_carrier} -> {next_carrier}: {exc}")
            bc_carrier = next_carrier
    band_chunk_ranges = tuple(
        (b0, min(b0 + bc_carrier, b_end))
        for b0 in range(b_start, b_end, bc_carrier))

    log_fn(
        f"  Whole-state randomized QRCP: states={m_states} "
        f"({nk} k * {nb} bands), full-Bloch dimension={state_dim}, "
        f"max_search=ceil({search_multiplier:g}*{nb}) -> {max_search}, "
        f"candidates={n_candidates} (+{candidate_carrier-n_candidates} "
        f"inactive mesh pad), qr_eps={float(qr_eps):.3e}, seed={qrcp_seed}, "
        f"rng={QRCP_RNG_VERSION}, WFN band carrier={bc_carrier}")
    log_fn(f"  [qrcp] candidate SHA256={candidate_hash}")
    log_fn(
        f"  Whole-state stage target {capacity/2**30:.2f} GiB/device; one "
        f"transformed full-grid row {row_memory.total/2**30:.3f} GiB/device "
        f"(cuFFT workspace {row_memory.cufft_scratch/2**30:.3f} GiB)")

    rep = NamedSharding(mesh_xy, P())
    face = NamedSharding(mesh_xy, P('x', 'y'))
    row = NamedSharding(mesh_xy, P(('x', 'y'), None))

    with build_psi_G_store(
            wfn=wfn, mesh_xy=mesh_xy, meta=meta,
            band_chunk_ranges=band_chunk_ranges, bispinor=bispinor,
            band_pad_to=bc_carrier) as source:
        sketch = _build_randomized_state_sketch(
            source=source, meta=meta, mesh_xy=mesh_xy, geom=geom,
            capacity=capacity, band_start=b_start, band_count=nb,
            candidate_states=candidates,
            candidate_carrier=candidate_carrier,
            sketch_rows=max_search, seed=qrcp_seed, log_fn=log_fn)

        @partial(jax.jit, out_shardings=(face, rep, rep))
        def _normalized_sketch_gram(y):
            norms = jnp.sqrt(jnp.sum(jnp.abs(y) ** 2, axis=0))
            active = jnp.arange(y.shape[1]) < n_candidates
            safe = jnp.where(active, jnp.maximum(norms, 1.0e-300), 1.0)
            yn = jnp.where(active[None, :], y / safe[None, :], 0.0)
            gram = jnp.einsum(
                'ra,rb->ab', jnp.conj(yn), yn, optimize=True)
            gram = jax.lax.with_sharding_constraint(gram, face)
            gram = 0.5 * (gram + gram.conj().T)
            return (gram,
                    jnp.min(jnp.where(active, norms, jnp.inf)),
                    jnp.max(jnp.where(active, norms, 0.0)))

        sketch_gram, sketch_norm_min, sketch_norm_max = \
            _normalized_sketch_gram(sketch)
        del sketch
        if (not np.isfinite(float(sketch_norm_min))
                or not np.isfinite(float(sketch_norm_max))
                or float(sketch_norm_min) <= 0.0):
            raise ValueError(
                "fit_galerkin_basis: randomized sketch produced a zero or "
                "non-finite physical candidate norm: min/max="
                f"{float(sketch_norm_min):.6e}/"
                f"{float(sketch_norm_max):.6e}")
        log_fn(
            f"  [qrcp] sketch column norm min/max before normalization="
            f"{float(sketch_norm_min):.6e}/{float(sketch_norm_max):.6e}")
        sketch_gram_row = device_put_process_local(sketch_gram, row)
        del sketch_gram

        active_np = np.zeros(candidate_carrier, dtype=bool)
        active_np[:n_candidates] = True
        active = device_put_process_local(
            active_np, NamedSharding(mesh_xy, P(('x', 'y'))))
        select = make_sharded_pivoted_cholesky_select(
            mesh_xy, candidate_carrier, max_search,
            mesh_axis=('x', 'y'), tol_rel=float(qr_eps) ** 2)
        (piv, _sketch_L, rank_qr_dev, d_final, d_taken,
         tr_residual, psd_info) = select(sketch_gram_row, None, active)
        jax.block_until_ready((piv, rank_qr_dev, psd_info))
        del _sketch_L, d_final, sketch_gram_row, active

        piv_host = np.asarray(piv, dtype=np.int64)
        rank_qr = int(np.asarray(rank_qr_dev))
        d_taken_host = np.asarray(d_taken)
        tr_residual_host = np.asarray(tr_residual)
        psd_host = (
            float(np.asarray(psd_info[0])),
            int(np.asarray(psd_info[1])),
            int(np.asarray(psd_info[2])),
        )
        if rank_qr <= 0:
            raise ValueError(
                "fit_galerkin_basis: randomized QRCP found zero rank")
        if np.any((piv_host[:rank_qr] < 0)
                  | (piv_host[:rank_qr] >= n_candidates)):
            raise RuntimeError(
                "fit_galerkin_basis: QRCP returned an active pivot outside "
                f"the logical candidate set [0,{n_candidates})")
        pc_floor = float(qr_eps) ** 2
        if psd_host[0] < -pc_floor:
            raise RuntimeError(
                "fit_galerkin_basis: sketched candidate Gram is not PSD: "
                f"minimum residual {psd_host[0]:.6e} at candidate "
                f"{psd_host[1]}, step {psd_host[2]}, below "
                f"-{pc_floor:.6e}")
        rank_phys = min(rank_qr, 2500)
        structural_search = max_search >= min(state_dim, m_states)
        if rank_phys > 0.9 * max_search and not structural_search:
            raise ValueError(
                "fit_galerkin_basis: the randomized QRCP search saturated: "
                f"delivered rank {rank_phys} exceeds 90% of "
                f"max_search={max_search}. Increase "
                "htransform_rank_multiplier or increase qr_eps; silently "
                "clipping this basis makes locality a tuning artifact.")
        selected = candidates[piv_host[:rank_phys]]
        pivot_hash = hashlib.sha256(
            selected.astype("<i8", copy=False).tobytes()).hexdigest()
        rank = padded_axis(
            rank_phys, mesh_xy, name="Galerkin selected-rank carrier",
            specs=((P("x", None), 0), (P(None, "y"), 1)),
            extra=extra_rank_pad).carrier
        n_pad = rank - rank_phys
        log_fn(
            f"  [qrcp] raw rank={rank_qr}, delivered physical rank="
            f"{rank_phys}" + (" (upstream safety cap 2500)"
                              if rank_qr > 2500 else "")
            + (f", +{n_pad} exact-null mesh pad -> {rank}" if n_pad else ""))
        log_fn(
            f"  [qrcp] pivot SHA256={pivot_hash}; first/last picked "
            f"residual={d_taken_host[0]:.6e}/"
            f"{d_taken_host[rank_qr-1]:.6e}; terminal trace residual="
            f"{tr_residual_host[rank_qr]:.6e}")

        owner, _, _ = source.state_row_owners(
            selected, band_start=b_start, band_count=nb)
        basis_stream, k_tile, x_groups, x_fft, basis_live = \
            _plan_basis_passes(
                geom, band_carrier=bc_carrier, rank=rank,
                rows=int(np.bincount(owner).max()), capacity=capacity)
        log_fn(
            f"  Whole-state basis plan: {len(basis_stream.r_chunk_ranges)} "
            f"r chunk(s), selected rows in {x_groups} group(s) x "
            f"{x_fft}-row FFT batches, projection k_tile={k_tile}; "
            + ", ".join(f"{k}={v/2**30:.2f} GiB" for k, v in
                        basis_live.items()) + "/device")
        selected_gram, x_chunks = _build_selected_state_gram(
            source=source, meta=meta, mesh_xy=mesh_xy,
            band_start=b_start, band_count=nb,
            selected_states=selected, rank_carrier=rank,
            stream=basis_stream, groups=x_groups, fft_rows=x_fft,
            log_fn=log_fn)

        batch_face = NamedSharding(mesh_xy, P(None, 'x', 'y'))

        @partial(jax.jit, out_shardings=batch_face)
        def _prepare_selected(g):
            g = 0.5 * (g + g.conj().T)
            if n_pad:
                pad_diag = jnp.concatenate([
                    jnp.zeros(rank_phys, dtype=jnp.float64),
                    jnp.ones(n_pad, dtype=jnp.float64)])
                g = g + jnp.diag(pad_diag).astype(g.dtype)
            return g[None]

        gram_stack = _prepare_selected(selected_gram)
        del selected_gram
        chol_plan = linalg_plan(
            "cholesky", mesh_xy, backend="native2d", n=rank,
            batched_route=distrib_la_batched_route)
        log_fn(f"  [route] selected-state factor: {chol_plan.describe()}")
        L_stack = chol_plan.batched(gram_stack)
        del gram_stack

        @partial(jax.jit, in_shardings=batch_face,
                 out_shardings=(rep, rep))
        def _replicate_lower(factors):
            L_ = jnp.tril(factors[0])
            return L_, jnp.min(jnp.real(jnp.diag(L_))[:rank_phys])

        L, min_chol_diag = _replicate_lower(L_stack)
        del L_stack
        if (not np.isfinite(float(min_chol_diag))
                or float(min_chol_diag) <= 0.0):
            raise ValueError(
                "fit_galerkin_basis: selected physical states are linearly "
                f"dependent; min diag(L)={float(min_chol_diag):.6e}")
        log_fn(
            f"  [qrcp] selected-state min diag(L)="
            f"{float(min_chol_diag):.6e}")

        ctilde = _build_physical_coefficients(
            source=source, meta=meta, mesh_xy=mesh_xy, band_count=nb,
            rank_carrier=rank, factor=L, x_chunks=x_chunks,
            stream=basis_stream, k_tile=k_tile, log_fn=log_fn)
        del x_chunks

    # Centroids enter only here, as evaluation points of the already-fixed
    # global basis.  This is the canonical WFN centroid loader and therefore
    # retains its FFT boxing, Bloch phase, padding and sharding conventions.
    psi_rmu, _ = load_centroids_band_chunked(
        wfn, None, meta, centroid_indices, bispinor, mesh_xy,
        band_range=(b_start, b_end), band_chunk_size=bc_carrier)
    B_at_mu = _basis_at_nodes_from_selected_states(
        psi_rmu=psi_rmu, selected_states=selected,
        factor=L, rank_carrier=rank, n_nodes=n_mu, mesh_xy=mesh_xy)
    del psi_rmu

    basis = GalerkinBasis(
        ctilde=ctilde,
        basis_at_nodes=B_at_mu,
        rank_physical=rank_phys,
        band_range=(b_start, b_end),
        selected_state_indices=tuple(int(v) for v in selected),
        selection_factor=L,
        qrcp_seed=qrcp_seed,
        qrcp_rng_version=QRCP_RNG_VERSION,
        qrcp_eps=float(qr_eps),
        qrcp_raw_rank=rank_qr,
        qrcp_search_rank=max_search,
        candidate_hash=candidate_hash,
        pivot_hash=pivot_hash,
    )
    rank_record = galerkin_rank_record(
        basis, meta=meta, rank_multiplier=search_multiplier)
    log_fn(
        f"  [gate] physical projection over all coarse states: "
        f"max|C C^H-I|="
        f"{rank_record['coefficient_orthogonality_error']:.3e}, "
        f"max missing state norm^2="
        f"{rank_record['max_missing_state_norm_squared']:.3e}, "
        f"||Psi-CB||_F/||Psi||_F="
        f"{rank_record['relative_frobenius_residual']:.3e}; "
        f"max|C[selected]-L|="
        f"{rank_record['selected_orientation_error']:.3e} "
        f"(cap {rank_record['selected_orientation_tolerance']:.3e})")
    if rank_record_fn is not None:
        rank_record_fn(rank_record)

    return basis




def _make_fold_G_kernel(rank_, mesh_, sharding_q_, grid_xy_):
    """Add one already-bounded, r-sharded ``Q_chunk Q_chunk†`` to G.

    The caller owns the zeta-style outer-r loop and therefore never hands this
    executable Q over all ``r_tot``.  Each device forms the Gram contribution
    from its unique local-r shard; the established two-stage
    ``psum_scatter`` sums those shards while distributing matrix rows and
    columns onto ``P('x','y')``.
    """
    key = (id(mesh_), int(rank_), tuple(sharding_q_.spec),
           tuple(grid_xy_.spec))
    fn = _FOLD_G_KERNELS.get(key)
    if fn is not None:
        return fn

    from runtime.padding import authenticate_padded_axis
    authenticate_padded_axis(
        rank_, rank_, mesh_, name="Galerkin rank carrier",
        specs=((P('x', None), 0), (P(None, 'y'), 1)))

    @partial(
        shard_map,
        mesh=mesh_,
        in_specs=(sharding_q_.spec, P('x', 'y')),
        out_specs=P('x', 'y'),
        check_vma=False,
    )
    def _fold_local(Q_local, G_local):
        partial = jnp.einsum(
            'asr,bsr->ab', Q_local, jnp.conj(Q_local),
            optimize=True)
        partial = jax.lax.psum_scatter(
            partial, 'x', scatter_dimension=0, tiled=True)
        partial = jax.lax.psum_scatter(
            partial, 'y', scatter_dimension=1, tiled=True)
        return G_local + partial

    fn = jax.jit(
        _fold_local,
        donate_argnums=(1,),
        in_shardings=(sharding_q_, grid_xy_),
        out_shardings=grid_xy_,
    )
    _FOLD_G_KERNELS[key] = fn
    return fn


_FOLD_G_KERNELS: dict = {}
_SELECTED_FILL_KERNELS: dict = {}
_SELECTED_ZERO_KERNELS: dict = {}
_SKETCH_RANDOM_KERNELS: dict = {}
_SKETCH_ACCUM_KERNELS: dict = {}
_SELECTED_PLACE_KERNELS: dict = {}
_PARTIAL_REDUCE_KERNELS: dict = {}
_BASIS_SOLVE_KERNELS: dict = {}
_PHYSICAL_PROJECT_KERNELS: dict = {}
_COEFFICIENT_ASSEMBLERS: dict = {}
_SPIN_OPERATOR_FOLD_KERNELS: dict = {}
_OPERATOR_ROTATION_KERNELS: dict = {}


def _state_rows_for_band_chunk(
        state_indices, *, band_start: int, band_count: int,
        band_range: tuple[int, int], band_carrier: int,
        row_carrier: int):
    """Map fixed stacked-state rows into one canonical band carrier."""
    states = np.asarray(state_indices, dtype=np.int64)
    if states.ndim != 1 or states.size > int(row_carrier):
        raise ValueError(
            "_state_rows_for_band_chunk: state list must be one-dimensional "
            f"and fit row_carrier={row_carrier}; got {states.shape}")
    rows = np.full(int(row_carrier), -1, dtype=np.int64)
    rows[:states.size] = states
    k_idx = np.where(rows >= 0, rows // int(band_count), 0)
    b_rel = np.where(rows >= 0, rows % int(band_count), 0)
    lo = int(band_range[0]) - int(band_start)
    hi = int(band_range[1]) - int(band_start)
    active = ((rows >= 0) & (b_rel >= lo) & (b_rel < hi))
    take = k_idx * int(band_carrier) + np.maximum(b_rel - lo, 0)
    take = np.where(active, take, 0).astype(np.int32)
    return take, active


def _make_selected_fill_kernel(
        *, mesh: Mesh, row_count: int, nk: int, band_carrier: int,
        nspinor: int, r_carrier: int, psi_layout, row_layout):
    key = (id(mesh), int(row_count), int(nk), int(band_carrier),
           int(nspinor), int(r_carrier), tuple(psi_layout.spec),
           tuple(row_layout.spec))
    fn = _SELECTED_FILL_KERNELS.get(key)
    if fn is not None:
        return fn
    rep = NamedSharding(mesh, P())

    @partial(
        jax.jit, donate_argnums=(3,),
        in_shardings=(psi_layout, rep, rep, row_layout),
        out_shardings=row_layout)
    def _fill(psi_bc, take, active, rows):
        psi_flat = psi_bc.reshape(
            int(nk) * int(band_carrier), int(nspinor), int(r_carrier))
        picked = psi_flat[take]
        picked = jnp.where(active[:, None, None], picked, 0.0)
        return rows + picked

    _SELECTED_FILL_KERNELS[key] = _fill
    return _fill


def _make_selected_zero_kernel(
        *, mesh: Mesh, row_count: int, nspinor: int, r_carrier: int,
        row_layout):
    """Cached allocation for one selected-state row carrier."""
    key = (id(mesh), int(row_count), int(nspinor), int(r_carrier),
           tuple(row_layout.spec))
    fn = _SELECTED_ZERO_KERNELS.get(key)
    if fn is not None:
        return fn

    @partial(jax.jit, out_shardings=row_layout)
    def _zeros():
        return jnp.zeros(
            (int(row_count), int(nspinor), int(r_carrier)),
            dtype=jnp.complex128)

    _SELECTED_ZERO_KERNELS[key] = _zeros
    return _zeros


def _make_sketch_random_kernel(
        *, mesh: Mesh, sketch_rows: int, nspinor: int, r_carrier: int,
        row_layout, seed: int):
    key = (id(mesh), int(sketch_rows), int(nspinor), int(r_carrier),
           tuple(row_layout.spec), int(seed))
    fn = _SKETCH_RANDOM_KERNELS.get(key)
    if fn is not None:
        return fn
    rep = NamedSharding(mesh, P())

    @partial(
        jax.jit, in_shardings=(rep, rep), out_shardings=row_layout)
    def _draw(r_start, logical_width):
        # A physical grid point owns its PRNG key.  Drawing one normal block
        # per global r index makes the sketch invariant to r-chunk boundaries,
        # device count and Q-memory budget; only physics inputs (seed, rank,
        # spinor count and global r) can change it.  The per-key output shape
        # is fixed, so vmap length also cannot perturb retained values.
        base = jax.random.PRNGKey(int(seed))
        global_r = (r_start.astype(jnp.uint32)
                    + jnp.arange(int(r_carrier), dtype=jnp.uint32))
        keys = jax.vmap(lambda r: jax.random.fold_in(base, r))(global_r)
        omega_r = jax.vmap(
            lambda key: jax.random.normal(
                key, (int(sketch_rows), int(nspinor)), dtype=jnp.float64)
        )(keys)
        omega = jnp.moveaxis(omega_r, 0, 2)
        active_r = jnp.arange(int(r_carrier)) < logical_width
        return jnp.where(active_r[None, None, :], omega, 0.0)

    _SKETCH_RANDOM_KERNELS[key] = _draw
    return _draw


def _state_groups(owner, groups: int):
    """Split state positions into ``groups`` sets, balanced per owner device."""
    owner = np.asarray(owner)
    parts = [[] for _ in range(int(groups))]
    for o in np.unique(owner):
        for g, piece in enumerate(np.array_split(
                np.flatnonzero(owner == o), int(groups))):
            parts[g].append(piece)
    return [np.sort(np.concatenate(p)) if p else np.zeros(0, np.int64)
            for p in parts]


def _make_rows_sketch_kernel(
        *, mesh: Mesh, sketch_rows: int, candidate_carrier: int,
        n_rows: int, nspinor: int, r_carrier: int):
    """Accumulate one r chunk of ``Omega Psi_rows^T`` into device partials.

    The partial stays unreduced per device (``P(('x','y'),None)`` over a
    leading device block); :func:`_reduce_device_partials` sums it once.
    """
    key = (id(mesh), int(sketch_rows), int(candidate_carrier), int(n_rows),
           int(nspinor), int(r_carrier))
    fn = _SKETCH_ACCUM_KERNELS.get(key)
    if fn is not None:
        return fn
    rep = NamedSharding(mesh, P())
    omega_spec = P(None, None, ('y', 'x'))
    rows_spec = P(None, None, None, ('y', 'x'))
    acc_spec = P(('x', 'y'), None)

    @partial(shard_map, mesh=mesh,
             in_specs=(omega_spec, rows_spec, P(), P(), acc_spec),
             out_specs=acc_spec, check_vma=False)
    def _accum(omega, rows, destination, active, acc):
        # Upstream applies a REAL Gaussian left sketch without conjugating
        # the wavefunction columns; each device contracts its own r shard.
        part = jnp.einsum('asr,csr->ac', omega, rows[0], optimize=True)
        part = jnp.where(active[None, :], part, 0.0)
        return acc.at[:, destination].add(part)

    fn = jax.jit(
        _accum, donate_argnums=(4,),
        in_shardings=(NamedSharding(mesh, omega_spec),
                      NamedSharding(mesh, rows_spec), rep, rep,
                      NamedSharding(mesh, acc_spec)),
        out_shardings=NamedSharding(mesh, acc_spec))
    _SKETCH_ACCUM_KERNELS[key] = fn
    return fn


def _reduce_device_partials(acc, mesh: Mesh):
    """Sum the per-device partial blocks of ``acc`` once (one all-reduce).

    ``acc`` stacks one block per device along axis 0 on
    ``P(('x','y'), None, ...)``; the replicated result has the block shape.
    """
    ndim = acc.ndim
    key = (id(mesh), ndim)
    fn = _PARTIAL_REDUCE_KERNELS.get(key)
    if fn is None:
        spec = P(('x', 'y'), *([None] * (ndim - 1)))

        @partial(shard_map, mesh=mesh, in_specs=(spec,), out_specs=P(),
                 check_vma=False)
        def _sum(local):
            return jax.lax.psum(local, ('x', 'y'))

        fn = _PARTIAL_REDUCE_KERNELS[key] = jax.jit(
            _sum, in_shardings=NamedSharding(mesh, spec),
            out_shardings=NamedSharding(mesh, P()))
    return fn(acc)


def _build_randomized_state_sketch(
        *, source, meta, mesh_xy: Mesh, geom: dict, capacity: float,
        band_start: int, band_count: int,
        candidate_states, candidate_carrier: int,
        sketch_rows: int, seed: int, log_fn):
    """``Omega Psi_candidate^T`` from the candidate rows alone.

    Only the candidate states are read, each transformed once; ``Omega`` is
    drawn per r chunk from the stateless global-r generator, so the sketch
    is the incumbent one up to summation order.
    """
    nspinor = int(meta.nspinor)
    product_r_spec = P(None, None, None, ('y', 'x'))
    random_layout = NamedSharding(mesh_xy, P(None, None, ('y', 'x')))
    rep = NamedSharding(mesh_xy, P())
    p = int(mesh_xy.size)
    candidates = np.asarray(candidate_states, dtype=np.int64)
    owner, _, _ = source.state_row_owners(
        candidates, band_start=band_start, band_count=band_count)
    groups, fft_rows, plan, live = _plan_rows_pass(
        geom, rows=int(np.bincount(owner).max()), omega_rows=sketch_rows,
        resident=2.0 * sketch_rows * candidate_carrier * _C16,
        capacity=capacity, name="the candidate sketch")
    position = {int(s): i for i, s in enumerate(candidates)}
    r_divisor = spec_divisor(mesh_xy, random_layout.spec, axis=2)

    @partial(jax.jit, out_shardings=NamedSharding(mesh_xy, P(('x', 'y'), None)))
    def _zeros():
        return jnp.zeros((p * int(sketch_rows), int(candidate_carrier)),
                         dtype=jnp.complex128)

    acc = _zeros()
    t0 = time.time()
    for group in _state_groups(owner, groups):
        rows, row_k, row_state = source.gather_state_rows(
            candidates[group], band_start=band_start, band_count=band_count,
            row_multiple=fft_rows)
        active_np = row_state >= 0
        dest = np.asarray([position.get(int(s), 0) for s in row_state],
                          dtype=np.int32)
        dest_dev = device_put_process_local(dest, rep)
        active = device_put_process_local(active_np, rep)
        for r_idx, slab in source.iter_rows_rchunks(
                rows, row_k, plan.r_chunk_ranges,
                product_r_spec=product_r_spec, fft_rows=fft_rows):
            r0, r1 = plan.r_chunk_ranges[r_idx]
            from runtime.padding import padded_axis
            r_carrier = padded_axis(
                r1 - r0, r_divisor,
                name="Galerkin sketch real-space carrier").carrier
            omega = _make_sketch_random_kernel(
                mesh=mesh_xy, sketch_rows=sketch_rows, nspinor=nspinor,
                r_carrier=r_carrier, row_layout=random_layout, seed=seed)(
                    jnp.asarray(r0, dtype=jnp.int32),
                    jnp.asarray(r1 - r0, dtype=jnp.int32))
            acc = _make_rows_sketch_kernel(
                mesh=mesh_xy, sketch_rows=sketch_rows,
                candidate_carrier=candidate_carrier,
                n_rows=int(rows.shape[0]), nspinor=nspinor,
                r_carrier=r_carrier)(omega, slab, dest_dev, active, acc)
            del omega, slab
        jax.block_until_ready(acc)
        del rows, row_k
    sketch = _reduce_device_partials(acc, mesh_xy)
    jax.block_until_ready(sketch)
    log_fn(
        f"  QRCP Gaussian sketch: {len(candidates)} candidate rows in "
        f"{groups} group(s), {len(plan.r_chunk_ranges)} r chunk(s), "
        f"{fft_rows}-row FFT batches, live {live/2**30:.2f} GiB/device, "
        f"{time.time()-t0:.2f}s")
    return sketch


def _selected_maps_on_device(
        *, source, selected_states, rank_carrier: int,
        band_start: int, band_count: int, mesh_xy: Mesh):
    from common.collectives import device_put_process_local
    rep = NamedSharding(mesh_xy, P())
    return tuple(
        tuple(device_put_process_local(arr, rep) for arr in
              _state_rows_for_band_chunk(
                  selected_states, band_start=band_start,
                  band_count=band_count, band_range=bc_range,
                  band_carrier=source.band_chunk_carrier,
                  row_carrier=rank_carrier))
        for bc_range in source.band_chunk_ranges)


def _make_rows_place_kernel(
        *, mesh: Mesh, rank: int, n_rows: int, nspinor: int, r_carrier: int):
    """Place one r chunk of gathered rows at their pivot positions (local)."""
    key = (id(mesh), int(rank), int(n_rows), int(nspinor), int(r_carrier))
    fn = _SELECTED_PLACE_KERNELS.get(key)
    if fn is not None:
        return fn
    rep = NamedSharding(mesh, P())
    rows_spec = P(None, None, None, ('y', 'x'))
    x_spec = P(None, None, ('y', 'x'))

    @partial(shard_map, mesh=mesh, in_specs=(rows_spec, P(), P(), x_spec),
             out_specs=x_spec, check_vma=False)
    def _place(rows, destination, active, x):
        return x.at[destination].add(
            jnp.where(active[:, None, None], rows[0], 0.0))

    fn = jax.jit(
        _place, donate_argnums=(3,),
        in_shardings=(NamedSharding(mesh, rows_spec), rep, rep,
                      NamedSharding(mesh, x_spec)),
        out_shardings=NamedSharding(mesh, x_spec))
    _SELECTED_PLACE_KERNELS[key] = fn
    return fn


def _build_selected_state_gram(
        *, source, meta, mesh_xy: Mesh, band_start: int, band_count: int,
        selected_states, rank_carrier: int, stream, groups: int,
        fft_rows: int, log_fn):
    """Exact physical ``X X^H`` and the resident r-chunked rows ``X``.

    Only the pivots are read, each transformed once.  ``X`` chunks (rows in
    pivot order, exact-null carrier rows) stay on the product-r layout for
    the projection pass.
    """
    nspinor = int(meta.nspinor)
    product_r_spec = P(None, None, None, ('y', 'x'))
    row_layout = NamedSharding(mesh_xy, P(None, None, ('y', 'x')))
    face = NamedSharding(mesh_xy, P('x', 'y'))
    rep = NamedSharding(mesh_xy, P())
    r_divisor = spec_divisor(mesh_xy, row_layout.spec, axis=2)
    selected = np.asarray(selected_states, dtype=np.int64)
    position = {int(s): i for i, s in enumerate(selected)}
    owner, _, _ = source.state_row_owners(
        selected, band_start=band_start, band_count=band_count)
    from runtime.padding import padded_axis
    carriers = [padded_axis(r1 - r0, r_divisor,
                            name="Galerkin selected-row carrier").carrier
                for r0, r1 in stream.r_chunk_ranges]
    x_chunks = [_make_selected_zero_kernel(
        mesh=mesh_xy, row_count=rank_carrier, nspinor=nspinor,
        r_carrier=c, row_layout=row_layout)() for c in carriers]
    t0 = time.time()
    for group in _state_groups(owner, groups):
        rows, row_k, row_state = source.gather_state_rows(
            selected[group], band_start=band_start, band_count=band_count,
            row_multiple=fft_rows)
        dest = device_put_process_local(np.asarray(
            [position.get(int(s), 0) for s in row_state], dtype=np.int32),
            rep)
        active = device_put_process_local(row_state >= 0, rep)
        for r_idx, slab in source.iter_rows_rchunks(
                rows, row_k, stream.r_chunk_ranges,
                product_r_spec=product_r_spec, fft_rows=fft_rows):
            x_chunks[r_idx] = _make_rows_place_kernel(
                mesh=mesh_xy, rank=rank_carrier, n_rows=int(rows.shape[0]),
                nspinor=nspinor, r_carrier=carriers[r_idx])(
                    slab, dest, active, x_chunks[r_idx])
            del slab
        del rows, row_k

    @partial(jax.jit, out_shardings=face)
    def _zeros_face():
        return jnp.zeros(
            (int(rank_carrier), int(rank_carrier)), dtype=jnp.complex128)

    gram = _zeros_face()
    fold = _make_fold_G_kernel(rank_carrier, mesh_xy, row_layout, face)
    for x in x_chunks:
        gram = fold(x, gram)
    jax.block_until_ready(gram)
    log_fn(
        f"  Exact selected-state Gram: {len(selected)} selected rows in "
        f"{groups} group(s), {len(x_chunks)} r chunk(s): "
        f"{time.time()-t0:.2f}s")
    return gram, x_chunks


def _solve_selected_basis_rows(factor, selected_rows):
    """Evaluate ``B = L^-1 X`` with spin kept outside the solve axis."""
    rhs = jnp.moveaxis(selected_rows, 1, 0)
    basis = jax.vmap(
        lambda x: jsp_linalg.solve_triangular(factor, x, lower=True))(rhs)
    return jnp.moveaxis(basis, 0, 1)


def _make_basis_solve_kernel(
        *, mesh: Mesh, rank: int, nspinor: int, r_carrier: int,
        row_layout):
    key = (id(mesh), int(rank), int(nspinor), int(r_carrier),
           tuple(row_layout.spec))
    fn = _BASIS_SOLVE_KERNELS.get(key)
    if fn is not None:
        return fn
    rep = NamedSharding(mesh, P())

    # ``solve_triangular`` acts on the replicated rank face and independent
    # RHS columns.  Make the product-r block boundary explicit: leaving this
    # to GSPMD can lower one library solve over the full logical RHS and
    # rematerialize both selected_rows and its output on every device.
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), row_layout.spec),
        out_specs=row_layout.spec,
        check_vma=False,
    )
    def _solve_local(L, selected_rows_local):
        return _solve_selected_basis_rows(L, selected_rows_local)

    fn = jax.jit(
        _solve_local,
        donate_argnums=(1,),
        in_shardings=(rep, row_layout),
        out_shardings=row_layout,
    )
    _BASIS_SOLVE_KERNELS[key] = fn
    return fn


def iter_galerkin_rchunks(
        source, basis: GalerkinBasis, meta, mesh_xy: Mesh, *,
        r_chunk_ranges, retained_band_range: tuple[int, int] | None):
    """Yield bounded physical basis rows and requested WFN rows together.

    This is the public continuation of a fitted :class:`GalerkinBasis` away
    from its registered centroid nodes.  The compact selected-state factor

    ``B(r_chunk) = L^-1 X_selected(r_chunk)``

    is evaluated from the caller-owned canonical :class:`PsiGStore`.  During
    the same WFN/FFT pass, only the overlap with ``retained_band_range`` is
    retained for a consumer's pair-density contraction.  Pass
    ``retained_band_range=None`` when only the physical basis rows are needed;
    the same WFN/FFT pass then builds no retained wavefunction payload.  The
    yielded shape is therefore bounded by one real-space carrier; no full-grid
    ``Psi`` or ``B`` exists, and no second WFN reader or FFT convention is
    introduced.

    Yields ``(r0, r1, B_chunk, psi_parts)``.  ``B_chunk`` has shape
    ``(rank, nspinor, r_carrier)`` and ``psi_parts`` is an ordered tuple of
    ``((band_lo, band_hi), psi_chunk)`` pairs covering exactly the retained
    band range (or is empty when none was requested).  Its arrays have shape
    ``(nk, band_hi-band_lo, nspinor, r_carrier)``; the terminal carrier tail
    is exact zero.  The caller must finish a yield before advancing the
    iterator so the bounded slabs can be released promptly.
    """
    b_start, b_end = (int(v) for v in basis.band_range)
    if retained_band_range is None:
        keep_start = keep_end = None
    else:
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
    row_spec = P(None, None, ('y', 'x'))
    row_layout = NamedSharding(mesh_xy, row_spec)
    psi_layout = NamedSharding(mesh_xy, P(None, None, None, ('y', 'x')))
    r_divisor = spec_divisor(mesh_xy, row_spec, axis=2)
    maps = _selected_maps_on_device(
        source=source,
        selected_states=basis.selected_state_indices,
        rank_carrier=rank,
        band_start=b_start,
        band_count=b_end - b_start,
        mesh_xy=mesh_xy)

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
        selected_rows = _make_selected_zero_kernel(
            mesh=mesh_xy, row_count=rank, nspinor=nspinor,
            r_carrier=r_carrier, row_layout=row_layout)()
        retained = []
        fill = _make_selected_fill_kernel(
            mesh=mesh_xy, row_count=rank, nk=int(meta.nk_tot),
            band_carrier=source.band_chunk_carrier, nspinor=nspinor,
            r_carrier=r_carrier, psi_layout=psi_layout,
            row_layout=row_layout)
        for bc_idx, (bc_range, psi_bc) in enumerate(
                source.iter_rchunk_bandwise(
                    r0, r1, product_r_spec=psi_layout.spec)):
            take, active = maps[bc_idx]
            selected_rows = fill(psi_bc, take, active, selected_rows)
            lo = (None if keep_start is None else
                  max(int(bc_range[0]), keep_start))
            hi = (None if keep_end is None else
                  min(int(bc_range[1]), keep_end))
            if lo is not None and lo < hi:
                offset = lo - int(bc_range[0])
                retained.append(
                    ((lo, hi), psi_bc[:, offset:offset + (hi - lo)]))
            else:
                del psi_bc
        solve = _make_basis_solve_kernel(
            mesh=mesh_xy, rank=rank, nspinor=nspinor,
            r_carrier=r_carrier, row_layout=row_layout)
        basis_chunk = solve(basis.selection_factor, selected_rows)
        del selected_rows
        if (keep_start is not None
                and sum(hi - lo for (lo, hi), _ in retained)
                != keep_end - keep_start):
            raise RuntimeError(
                "iter_galerkin_rchunks: retained WFN pieces do not cover "
                f"[{keep_start},{keep_end})")
        yield r0, r1, basis_chunk, tuple(retained)


def _make_spin_operator_fold_kernel(
        *, rank: int, nspinor: int, r_carrier: int,
        mesh: Mesh, basis_left_layout, basis_right_layout, face_layout):
    """Fold one local-r basis slab into distributed operator and metric faces."""
    key = (id(mesh), int(rank), int(nspinor), int(r_carrier),
           tuple(basis_left_layout.spec), tuple(basis_right_layout.spec),
           tuple(face_layout.spec))
    fn = _SPIN_OPERATOR_FOLD_KERNELS.get(key)
    if fn is not None:
        return fn
    rep = NamedSharding(mesh, P())

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(basis_left_layout.spec, basis_right_layout.spec,
                  P(), P('x', 'y'), P('x', 'y')),
        out_specs=(P('x', 'y'), P('x', 'y')),
        check_vma=False,
    )
    def _fold_local(basis_left, basis_right, spin_operator,
                    operator_local, metric_local):
        # Each operand carries only one output-rank panel and one r shard.
        # Gather r within the orthogonal mesh axis, leaving rank/a on x and
        # rank/b on y.  The contraction therefore creates exactly the local
        # output face tile; no rank-by-rank intermediate exists on any device.
        basis_left = jax.lax.all_gather(
            basis_left, 'y', axis=2, tiled=True)
        basis_right = jax.lax.all_gather(
            basis_right, 'x', axis=2, tiled=True)
        # B[a,s,r] is the ket value phi_a,s(r).  Keep this bra/ket order
        # explicit because the selected-state Gram builder uses the transposed
        # row-Gram convention, which is harmless for I but wrong for a complex
        # operator matrix.
        metric_partial = jnp.einsum(
            'asr,bsr->ab', jnp.conj(basis_left), basis_right,
            optimize=True)
        operator_partial = jnp.einsum(
            'asr,st,btr->ab', jnp.conj(basis_left), spin_operator,
            basis_right, optimize=True)
        return (operator_local + operator_partial,
                metric_local + metric_partial)

    fn = jax.jit(
        _fold_local,
        donate_argnums=(3, 4),
        in_shardings=(basis_left_layout, basis_right_layout,
                      rep, face_layout, face_layout),
        out_shardings=(face_layout, face_layout),
    )
    _SPIN_OPERATOR_FOLD_KERNELS[key] = fn
    return fn


def project_galerkin_spin_operator(
        source, basis: GalerkinBasis, meta, mesh_xy: Mesh, *,
        spin_operator, q_tile_budget: int) -> GalerkinOperatorProjection:
    """Project a uniform Hermitian spin operator through the full-grid basis.

    The operator acts only on the spinor index and is constant in real space.
    Physical basis slabs come from :func:`iter_galerkin_rchunks`; each slab is
    contracted over its local real-space shard and reduced directly onto the
    all-mesh rank face.  No full spatial basis or rank-by-rank result is
    gathered.  The returned metric supports ``c^H O c / c^H M c`` even for a
    basis with exact-null carrier padding.
    """
    ns = int(meta.nspinor)
    op_np = np.asarray(spin_operator, dtype=np.complex128)
    if op_np.shape != (ns, ns):
        raise ValueError(
            "project_galerkin_spin_operator: spin_operator must have shape "
            f"({ns},{ns}); got {op_np.shape}")
    if not np.all(np.isfinite(op_np)):
        raise ValueError(
            "project_galerkin_spin_operator: spin_operator must be finite")
    if not np.allclose(op_np, op_np.conj().T, rtol=0.0, atol=1.0e-14):
        raise ValueError(
            "project_galerkin_spin_operator: spin_operator must be Hermitian")

    rank = int(basis.rank_carrier)
    plan = plan_galerkin_operator_stream(
        rank=rank, nspinor=ns, n_rtot=int(meta.n_rtot),
        mesh_xy=mesh_xy, q_tile_budget=int(q_tile_budget))
    basis_left_layout = NamedSharding(mesh_xy, P('x', None, 'y'))
    basis_right_layout = NamedSharding(mesh_xy, P('y', None, 'x'))
    face_layout = NamedSharding(mesh_xy, P('x', 'y'))
    rep = NamedSharding(mesh_xy, P())
    op = jax.device_put(op_np, rep)

    @partial(jax.jit, out_shardings=(face_layout, face_layout))
    def _zeros():
        shape = (rank, rank)
        return (jnp.zeros(shape, dtype=jnp.complex128),
                jnp.zeros(shape, dtype=jnp.complex128))

    operator, metric = _zeros()
    for _, _, basis_chunk, retained in iter_galerkin_rchunks(
            source, basis, meta, mesh_xy,
            r_chunk_ranges=plan.r_chunk_ranges, retained_band_range=None):
        if retained:
            raise RuntimeError(
                "operator-only Galerkin stream unexpectedly retained WFN rows")
        fold = _make_spin_operator_fold_kernel(
            rank=rank, nspinor=ns, r_carrier=int(basis_chunk.shape[2]),
            mesh=mesh_xy, basis_left_layout=basis_left_layout,
            basis_right_layout=basis_right_layout, face_layout=face_layout)
        basis_left = jax.device_put(basis_chunk, basis_left_layout)
        basis_right = jax.device_put(basis_chunk, basis_right_layout)
        operator, metric = fold(
            basis_left, basis_right, op, operator, metric)
        jax.block_until_ready((operator, metric))
        del basis_chunk, basis_left, basis_right
    return GalerkinOperatorProjection(operator=operator, metric=metric)


def project_galerkin_spin_z(
        source, basis: GalerkinBasis, meta, mesh_xy: Mesh, *,
        q_tile_budget: int) -> GalerkinOperatorProjection:
    """Project ``S_z/hbar = sigma_z/2`` for a two-component Pauli basis."""
    if int(meta.nspinor) != 2:
        raise ValueError(
            "project_galerkin_spin_z requires two-component Pauli spinors; "
            f"got nspinor={int(meta.nspinor)}")
    from common.gamma_matrices import sigma_z
    return project_galerkin_spin_operator(
        source, basis, meta, mesh_xy,
        spin_operator=0.5 * sigma_z, q_tile_budget=q_tile_budget)


def rotate_galerkin_operator(
        coefficients, projection: GalerkinOperatorProjection,
        mesh_xy: Mesh, *,
        logical_q_count: int | None = None) -> GalerkinStateExpectation:
    """Evaluate ``c^H O c / c^H M c`` for selected htransform states.

    ``coefficients`` has shape ``(q_carrier, rank_carrier, nband)`` and must
    come from
    the solve whose states are being published: ``h_transform(return_coeffs=True)``
    for a standalone path, including its active/guard selection, or
    ``compute_wfns_fi(return_coeffs=True).coeffs_fi`` for that consumer's
    energy-ordered window.  ``logical_q_count`` removes inert q padding from
    the returned small arrays.  One all-mesh q batch at a time is reshaped
    into the two rank-sharded views that meet the face-sharded operator; the
    full path coefficient carrier never exists in either replicated view.
    """
    if coefficients.ndim != 3:
        raise ValueError(
            "rotate_galerkin_operator: coefficients must have shape "
            f"(nq,rank,nband); got {tuple(coefficients.shape)}")
    q_input, rank, nb = (int(v) for v in coefficients.shape)
    nq = q_input if logical_q_count is None else int(logical_q_count)
    if not (0 < nq <= q_input):
        raise ValueError(
            "rotate_galerkin_operator: logical_q_count must lie in "
            f"[1,{q_input}]; got {nq}")
    expected = (rank, rank)
    if tuple(projection.operator.shape) != expected:
        raise ValueError(
            "rotate_galerkin_operator: operator shape "
            f"{tuple(projection.operator.shape)} != {expected}")
    if tuple(projection.metric.shape) != expected:
        raise ValueError(
            "rotate_galerkin_operator: metric shape "
            f"{tuple(projection.metric.shape)} != {expected}")

    q_layout = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    coeff_x = NamedSharding(mesh_xy, P(None, 'x', None))
    coeff_y = NamedSharding(mesh_xy, P(None, 'y', None))
    face = NamedSharding(mesh_xy, P('x', 'y'))
    rep = NamedSharding(mesh_xy, P())
    from runtime.padding import padded_axis
    q_batch = padded_axis(
        1, mesh_xy, name="Galerkin operator q batch",
        spec=q_layout.spec, axis=0).carrier
    q_carrier = padded_axis(
        q_input, q_batch, name="Galerkin operator q carrier").carrier
    if q_carrier != q_input:
        coefficients = jnp.pad(
            coefficients, ((0, q_carrier - q_input), (0, 0), (0, 0)))
    coefficients = jax.device_put(coefficients, q_layout)

    key = (id(mesh_xy), q_batch, rank, nb)
    fn = _OPERATOR_ROTATION_KERNELS.get(key)
    if fn is None:
        @partial(
            shard_map,
            mesh=mesh_xy,
            in_specs=(P(None, 'x', None), P('x', 'y'),
                      P(None, 'y', None), P('x', 'y')),
            out_specs=(P(), P(), P()),
            check_vma=False,
        )
        def _rotate_local(c_left, operator_local, c_right, metric_local):
            numerator = jnp.einsum(
                'qan,ab,qbn->qn', jnp.conj(c_left), operator_local,
                c_right, optimize=True)
            norm = jnp.einsum(
                'qan,ab,qbn->qn', jnp.conj(c_left), metric_local,
                c_right, optimize=True)
            for axis in ('x', 'y'):
                numerator = jax.lax.psum(numerator, axis)
                norm = jax.lax.psum(norm, axis)
            value = jnp.real(numerator) / jnp.where(
                jnp.abs(norm) > 0.0, jnp.real(norm), jnp.nan)
            return value, numerator, norm

        fn = jax.jit(
            _rotate_local,
            in_shardings=(coeff_x, face, coeff_y, face),
            out_shardings=(rep, rep, rep),
        )
        _OPERATOR_ROTATION_KERNELS[key] = fn
    values = []
    numerators = []
    norms = []
    for q0 in range(0, q_carrier, q_batch):
        q_chunk = coefficients[q0:q0 + q_batch]
        c_left = jax.device_put(q_chunk, coeff_x)
        c_right = jax.device_put(q_chunk, coeff_y)
        value, numerator, norm = fn(
            c_left, projection.operator, c_right, projection.metric)
        jax.block_until_ready((value, numerator, norm))
        values.append(value)
        numerators.append(numerator)
        norms.append(norm)
        del q_chunk, c_left, c_right
    return GalerkinStateExpectation(
        value=jnp.concatenate(values, axis=0)[:nq],
        numerator=jnp.concatenate(numerators, axis=0)[:nq],
        norm=jnp.concatenate(norms, axis=0)[:nq])


def _assemble_coefficient_chunks(
        chunks, *, logical_widths, nk: int, rank: int, mesh_xy: Mesh):
    widths = tuple(int(v) for v in logical_widths)
    key = (id(mesh_xy), widths, int(nk), int(rank))
    fn = _COEFFICIENT_ASSEMBLERS.get(key)
    if fn is None:
        rep = NamedSharding(mesh_xy, P())

        @partial(jax.jit, in_shardings=tuple(rep for _ in widths),
                 out_shardings=rep)
        def _assemble(*values):
            return jnp.concatenate(
                tuple(v[:, :w, :] for v, w in zip(values, widths)), axis=1)

        fn = _COEFFICIENT_ASSEMBLERS[key] = _assemble
    return fn(*chunks)


def _make_projection_accum_kernel(
        *, mesh: Mesh, nk: int, band_carrier: int, rank: int,
        nspinor: int, r_carrier: int):
    """Accumulate one r chunk of ``Psi X^H`` into per-device partials."""
    key = (id(mesh), int(nk), int(band_carrier), int(rank), int(nspinor),
           int(r_carrier))
    fn = _PHYSICAL_PROJECT_KERNELS.get(key)
    if fn is not None:
        return fn
    psi_spec = P(None, None, None, ('y', 'x'))
    x_spec = P(None, None, ('y', 'x'))
    acc_spec = P(('x', 'y'), None, None)

    @partial(shard_map, mesh=mesh, in_specs=(psi_spec, x_spec, acc_spec),
             out_specs=acc_spec, check_vma=False)
    def _accum(psi, x, acc):
        return acc + jnp.einsum(
            'kbsr,asr->kba', psi, jnp.conj(x), optimize=True)

    fn = jax.jit(
        _accum, donate_argnums=(2,),
        in_shardings=(NamedSharding(mesh, psi_spec),
                      NamedSharding(mesh, x_spec),
                      NamedSharding(mesh, acc_spec)),
        out_shardings=NamedSharding(mesh, acc_spec))
    _PHYSICAL_PROJECT_KERNELS[key] = fn
    return fn


def _coefficients_from_projection(projection, factor):
    """``C = (Psi X^H) L^-H``, i.e. ``C^H = L^-1 (Psi X^H)^H``."""
    nk, nb, rank = projection.shape
    rhs = jnp.conj(projection.reshape(nk * nb, rank)).T
    c_h = jsp_linalg.solve_triangular(factor, rhs, lower=True)
    return jnp.conj(c_h).T.reshape(nk, nb, rank)


def _build_physical_coefficients(
        *, source, meta, mesh_xy: Mesh, band_count: int, rank_carrier: int,
        factor, x_chunks, stream, k_tile: int, log_fn):
    """Stream every state once for ``C = (Psi X^H) L^-H``.

    Band chunks outside, r chunks inside: each band chunk is transformed
    once over the whole grid and contracted chunk by chunk against the
    resident ``X``; its per-device partials are reduced once.
    """
    nk = int(meta.nk_tot)
    nspinor = int(meta.nspinor)
    p = int(mesh_xy.size)
    band_carrier = int(source.band_chunk_carrier)
    rep = NamedSharding(mesh_xy, P())
    acc_layout = NamedSharding(mesh_xy, P(('x', 'y'), None, None))

    @partial(jax.jit, out_shardings=acc_layout)
    def _zeros_partial():
        return jnp.zeros((p * nk, band_carrier, int(rank_carrier)),
                         dtype=jnp.complex128)

    chunks, acc, current = [], None, None
    t0 = time.time()
    for bc_range, r_idx, psi in source.iter_bandchunks_rchunks(
            stream.r_chunk_ranges,
            product_r_spec=P(None, None, None, ('y', 'x')), k_tile=k_tile):
        if bc_range != current:
            if acc is not None:
                chunks.append(_reduce_device_partials(acc, mesh_xy))
            acc, current = _zeros_partial(), bc_range
        acc = _make_projection_accum_kernel(
            mesh=mesh_xy, nk=nk, band_carrier=band_carrier,
            rank=rank_carrier, nspinor=nspinor,
            r_carrier=int(psi.shape[-1]))(psi, x_chunks[r_idx], acc)
        del psi
    chunks.append(_reduce_device_partials(acc, mesh_xy))
    del acc
    widths = tuple(
        int(hi) - int(lo) for lo, hi in source.band_chunk_ranges)
    projection = _assemble_coefficient_chunks(
        tuple(chunks), logical_widths=widths, nk=nk,
        rank=rank_carrier, mesh_xy=mesh_xy)
    del chunks
    out = jax.jit(_coefficients_from_projection,
                  in_shardings=(rep, rep), out_shardings=rep)(
                      projection, factor)
    jax.block_until_ready(out)
    log_fn(
        f"  Physical C=(Psi X^H) L^-H projection: "
        f"{len(source.band_chunk_ranges)} band chunk(s) x "
        f"{len(stream.r_chunk_ranges)} r chunk(s), k_tile={k_tile}: "
        f"{time.time()-t0:.2f}s")
    return out


def _basis_at_nodes_from_selected_states(
        *, psi_rmu, selected_states, factor, rank_carrier: int,
        n_nodes: int, mesh_xy: Mesh):
    """Evaluate ``B=L^-1 X`` at registered centroids in the same gauge."""
    selected = np.asarray(selected_states, dtype=np.int64)
    rank_phys = int(selected.size)
    nspinor = int(psi_rmu.shape[2])
    mu_carrier = int(psi_rmu.shape[3])
    out_sharding = _fit(
        mesh_xy, P(None, None, 'y'),
        (int(rank_carrier), nspinor, int(n_nodes)),
        "galerkin.basis_at_nodes(mu-axis)")
    rep = NamedSharding(mesh_xy, P())
    in_sharding = psi_rmu.sharding

    @partial(
        jax.jit, in_shardings=(in_sharding, rep),
        out_shardings=out_sharding)
    def _evaluate(psi, L):
        flat = psi.reshape(-1, nspinor, mu_carrier)
        rows = flat[jnp.asarray(selected)]
        if int(rank_carrier) > rank_phys:
            rows = jnp.pad(
                rows, ((0, int(rank_carrier) - rank_phys), (0, 0), (0, 0)))
        return _solve_selected_basis_rows(L, rows)[..., :int(n_nodes)]

    return _evaluate(psi_rmu, factor)


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
    if q_bytes_per_local_r > q_tile_budget:
        raise ValueError(
            "plan_galerkin_stream: one local r column of Q needs "
            f"{q_bytes_per_local_r / 1024**3:.6f} GiB/device, exceeding "
            f"q_tile_budget={q_tile_budget / 1024**3:.6f} GiB/device. "
            "Increase that budget or reduce the retained rank.")
    r_local_cap = q_tile_budget // q_bytes_per_local_r
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


def plan_galerkin_operator_stream(
        *, rank: int, nspinor: int, n_rtot: int, mesh_xy: Mesh,
        q_tile_budget: int) -> GalerkinStreamPlan:
    """Bound the two operator-projection rank panels by one tile budget.

    The canonical basis stream initially splits r over all P ranks.  The
    operator fold reshards that tile into an x-rank/y-r left panel and the
    transposed y-rank/x-r right panel, then gathers r only within the panel's
    orthogonal mesh axis.  Relative to the product-r source tile those two
    live panels cost ``mesh_y + mesh_x`` times as much, while the original
    product-r basis tile remains live through both reshard dependencies.  Price
    the conservative ``1 + mesh_y + mesh_x`` live set here while retaining
    product-mesh alignment for the source transform.
    """
    axis_sizes = {
        str(name): int(size)
        for name, size in zip(mesh_xy.axis_names, mesh_xy.devices.shape)
    }
    if "x" not in axis_sizes or "y" not in axis_sizes:
        raise ValueError(
            "plan_galerkin_operator_stream requires mesh axes ('x','y')")
    expansion = 1 + axis_sizes["x"] + axis_sizes["y"]
    budget = int(q_tile_budget)
    source_budget = budget // expansion
    if source_budget <= 0:
        raise ValueError(
            "plan_galerkin_operator_stream: q_tile_budget is too small for "
            f"the two rank panels on a {axis_sizes['x']}x{axis_sizes['y']} "
            "mesh")
    plan = plan_galerkin_stream(
        rank=int(rank), nspinor=int(nspinor), n_rtot=int(n_rtot),
        r_mesh_divisor=int(mesh_xy.size), q_tile_budget=source_budget)
    return GalerkinStreamPlan(
        r_chunk_ranges=plan.r_chunk_ranges,
        max_r_logical=plan.max_r_logical,
        max_r_carrier=plan.max_r_carrier,
        q_tile_local_bytes=plan.q_tile_local_bytes * expansion,
    )
