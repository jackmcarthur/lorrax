"""SlabIO artifact for the fixed-gauge parallel-transport head data.

This module owns the whole preprocessing transaction. It streams one raw IBZ
centre and one positive neighbour at a time, writes compact links, then
re-reads those links through SlabIO for symmetry unfolding and fourth-order
connection construction. No wavefunction survives the streamed link stage,
and every HDF5 payload or metadata item crosses the SlabIO service door.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.mtxel_sweep import band_sphere_spec
from common.parallel_transport import (
    band_storage_extent,
    build_forward_neighbor_table,
    build_g_wrap_lookup,
    build_neighbor_table,
    fourth_order_connection,
    g_wrap_for_forward_step,
    g_wrap_for_step,
    make_cross_k_link,
    make_cross_k_overlap,
    make_distributed_band_matmul,
)
from file_io.slab_io import SlabIO


SCHEMA_VERSION = 2
LINKS_DATASET = "links_ibz"
SINGULAR_VALUES_DATASET = "singular_values_ibz"
CONNECTION_REDUCED_DATASET = "berry_connection_reduced"
CONNECTION_CART_DATASET = "berry_connection_cart"
VELOCITY_DFT_DATASET = "velocity_dft_cart"
ENERGIES_DATASET = "dft_energies_ry_full"
W_AV_DENSITY_DATASET = "w_av_density_mtxel"
W_AV_SCHEMA_VERSION = 3
OCCUPATIONS_DATASET = "dft_occupations_full"

__all__ = [
    "CONNECTION_CART_DATASET",
    "CONNECTION_REDUCED_DATASET",
    "ENERGIES_DATASET",
    "LINKS_DATASET",
    "OCCUPATIONS_DATASET",
    "SCHEMA_VERSION",
    "SINGULAR_VALUES_DATASET",
    "VELOCITY_DFT_DATASET",
    "complete_velocity_validation",
    "W_AV_DENSITY_DATASET",
    "WAvStencilMetadata",
    "WAvStencilReader",
    "covariant_velocity",
    "initialize_parallel_transport_artifact",
    "validate_covariant_velocity",
    "validate_parallel_transport_artifact",
    "write_parallel_transport_artifact",
    "write_w_av_stencil_artifact",
    "write_velocity_validation",
]


@dataclass(frozen=True)
class WAvStencilMetadata:
    """Small, replicated contract for a streamed finite-q vertex file."""

    schema_version: int
    complete: bool
    band_start: int
    band_stop: int
    effective_nspinor: int
    bispinor: bool
    kgrid: np.ndarray
    kgrid_shift: np.ndarray
    reciprocal_lattice_cart: np.ndarray
    wfn_path: str
    wfn_fingerprint: str
    first_neighbors: bool
    second_neighbors: bool
    seed_steps: np.ndarray
    seed_shell_mask: np.ndarray
    source_steps: np.ndarray
    source_full_ids: np.ndarray
    source_irr_idx: np.ndarray
    source_shell_mask: np.ndarray
    source_kplusq_full: np.ndarray
    source_g_wrap: np.ndarray
    target_steps: np.ndarray
    target_full_ids: np.ndarray
    target_source_row: np.ndarray
    target_sym_idx: np.ndarray
    target_antiunitary: np.ndarray
    target_shell_mask: np.ndarray
    target_seed_mask: np.ndarray

    @property
    def nbands(self) -> int:
        return int(self.band_stop - self.band_start)

    @property
    def nk_full(self) -> int:
        return int(np.prod(self.kgrid))

    @property
    def nq_source(self) -> int:
        return int(self.source_steps.shape[0])


def _decode_i32_text(value) -> str:
    raw = np.asarray(value, dtype=np.int32)
    if np.any(raw < 0) or np.any(raw > 255):
        raise ValueError("W-av text metadata contains a non-byte value")
    return bytes(raw.astype(np.uint8).tolist()).decode("utf-8")


def _read_w_av_metadata(io: SlabIO) -> WAvStencilMetadata:
    """Read only the small replicated datasets through SlabIO."""

    def _read(name, ndim):
        return np.asarray(io.read_slab(
            name, partition_spec=P(*((None,) * int(ndim))), as_numpy=True))

    def _scalar(name):
        value = _read(name, 1).reshape(-1)
        if value.size != 1:
            raise ValueError(f"W-av metadata {name!r} must contain one value")
        return int(value[0])

    meta = WAvStencilMetadata(
        schema_version=_scalar("w_av_schema_version"),
        complete=bool(_scalar("w_av_complete")),
        band_start=_scalar("band_start"),
        band_stop=_scalar("band_stop"),
        effective_nspinor=_scalar("effective_nspinor"),
        bispinor=bool(_scalar("bispinor")),
        kgrid=_read("kgrid", 1).astype(np.int32),
        kgrid_shift=_read("kgrid_shift", 1).astype(np.float64),
        reciprocal_lattice_cart=(
            _read("reciprocal_lattice_cart", 2).astype(np.float64)),
        wfn_path=_decode_i32_text(_read("wfn_path_bytes_i32", 1)),
        wfn_fingerprint=_decode_i32_text(
            _read("wfn_fingerprint_bytes_i32", 1)),
        first_neighbors=bool(_scalar("w_av_first_neighbors")),
        second_neighbors=bool(_scalar("w_av_second_neighbors")),
        seed_steps=_read("w_av_seed_steps", 2).astype(np.int32),
        seed_shell_mask=_read("w_av_seed_shell_mask", 1).astype(np.int32),
        source_steps=_read("w_av_source_qsteps", 2).astype(np.int32),
        source_full_ids=_read("w_av_source_q_full_ids", 1).astype(np.int32),
        source_irr_idx=_read("w_av_source_q_irr_idx", 1).astype(np.int32),
        source_shell_mask=(
            _read("w_av_source_shell_mask", 1).astype(np.int32)),
        source_kplusq_full=(
            _read("w_av_source_kplusq_full", 2).astype(np.int32)),
        source_g_wrap=_read("w_av_source_g_wrap", 3).astype(np.int32),
        target_steps=_read("w_av_target_qsteps", 2).astype(np.int32),
        target_full_ids=_read("w_av_target_q_full_ids", 1).astype(np.int32),
        target_source_row=(
            _read("w_av_target_source_row", 1).astype(np.int32)),
        target_sym_idx=_read("w_av_target_sym_idx", 1).astype(np.int32),
        target_antiunitary=(
            _read("w_av_target_antiunitary", 1).astype(np.int32)),
        target_shell_mask=(
            _read("w_av_target_shell_mask", 1).astype(np.int32)),
        target_seed_mask=(
            _read("w_av_target_seed_mask", 2).astype(np.int32)),
    )
    if meta.schema_version != W_AV_SCHEMA_VERSION:
        raise ValueError(
            f"W-av schema {meta.schema_version} != {W_AV_SCHEMA_VERSION}")
    if not meta.complete:
        raise ValueError("W-av artifact is incomplete")
    if meta.band_start != 0 or meta.band_stop <= 0:
        raise ValueError(
            f"invalid W-av band interval [{meta.band_start},{meta.band_stop})")
    if meta.kgrid.shape != (3,) or np.any(meta.kgrid <= 0):
        raise ValueError(f"invalid W-av kgrid {meta.kgrid}")
    expected_kq = (meta.nq_source, meta.nk_full)
    if meta.source_kplusq_full.shape != expected_kq:
        raise ValueError(
            f"W-av k+q table shape {meta.source_kplusq_full.shape} != "
            f"{expected_kq}")
    if meta.source_g_wrap.shape != expected_kq + (3,):
        raise ValueError(
            f"W-av G-wrap shape {meta.source_g_wrap.shape} != "
            f"{expected_kq + (3,)}")
    return meta


class WAvStencilReader:
    """Validated, one-source-q-at-a-time SlabIO reader.

    Metadata is read collectively at construction.  The large density
    dataset is never read whole: :meth:`read_source_row` returns one q row
    sharded over both band axes.
    """

    def __init__(self, path, *, mesh) -> None:
        self.path = str(path)
        self.mesh = mesh
        self._io = SlabIO(self.path, mode="r", mesh=mesh)
        try:
            self.metadata = _read_w_av_metadata(self._io)
        except Exception:
            self._io.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self._io is not None:
            self._io.close()
            self._io = None

    def validate_runtime(
        self, *, nbands, effective_nspinor, bispinor, kgrid, kgrid_shift,
        reciprocal_lattice_cart, wfn_fingerprint, sym,
    ) -> None:
        """Refuse unless file geometry and the canonical SymMaps agree."""
        m = self.metadata
        checks = (
            (m.nbands == int(nbands),
             f"nbands {m.nbands} != runtime {int(nbands)}"),
            (m.effective_nspinor == int(effective_nspinor),
             "effective spinor extent differs"),
            (m.bispinor == bool(bispinor), "bispinor setting differs"),
            (np.array_equal(m.kgrid, np.asarray(kgrid, dtype=np.int32)),
             "kgrid differs"),
            (np.allclose(m.kgrid_shift, kgrid_shift, atol=1.0e-12),
             "kgrid shift differs"),
            (np.allclose(m.reciprocal_lattice_cart,
                         reciprocal_lattice_cart, atol=1.0e-12),
             "reciprocal lattice differs"),
            (m.wfn_fingerprint == str(wfn_fingerprint),
             "WFN fingerprint differs"),
        )
        for passed, message in checks:
            if not passed:
                raise ValueError(f"W-av artifact/runtime mismatch: {message}")

        from symmetry_maps import q_stencil_orbit_table
        table = q_stencil_orbit_table(
            kgrid=m.kgrid,
            sym_mats_k=np.asarray(sym.sym_mats_k, dtype=np.int32),
            irr_idx_q=np.asarray(sym.irr_idx_q, dtype=np.int32),
            sym_idx_q=np.asarray(sym.sym_idx_q, dtype=np.int32),
            seed_steps=m.seed_steps,
            n_sym_spatial=int(len(sym.sym_mats_k) // 2),
            allow_trs=bool(sym.trs_allowed),
        )
        expected = {
            "source_steps": m.source_steps,
            "source_full_ids": m.source_full_ids,
            "source_irr_idx": m.source_irr_idx,
            "target_steps": m.target_steps,
            "target_full_ids": m.target_full_ids,
            "target_source_row": m.target_source_row,
            "target_sym_idx": m.target_sym_idx,
            "target_antiunitary": m.target_antiunitary,
            "target_seed_mask": m.target_seed_mask,
        }
        for name, stored in expected.items():
            if not np.array_equal(np.asarray(table[name]), stored):
                raise ValueError(
                    f"W-av artifact/runtime SymMaps mismatch in {name}")
        neighbors = build_neighbor_table(
            np.asarray(sym.kvecs_asints, dtype=np.int32),
            m.kgrid, m.source_steps).T
        if not np.array_equal(neighbors, m.source_kplusq_full):
            raise ValueError(
                "W-av artifact/runtime SymMaps mismatch in k+q endpoints")

    def read_source_row(self, source_row: int):
        """Read one ``(k,n,m)`` source-q vertex tile as ``P(None,x,y)``."""
        if self._io is None:
            raise RuntimeError("WAvStencilReader is closed")
        iq = int(source_row)
        m = self.metadata
        if not 0 <= iq < m.nq_source:
            raise IndexError(
                f"source_row={iq} outside [0,{m.nq_source})")
        nb_storage = band_storage_extent(self.mesh, m.nbands)
        slab = self._io.read_slab(
            W_AV_DENSITY_DATASET,
            shape=(1, m.nk_full, nb_storage, nb_storage),
            offset=(iq, 0, 0, 0),
            valid_shape=(1, m.nk_full, m.nbands, m.nbands),
            partition_spec=P(None, None, "x", "y"))
        return slab[0]


def _require_service_apis():
    """Import sibling-lane service APIs only when the opt-in path runs."""
    try:
        from distrib_la import plan_polar_factor
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "parallel-transport preprocessing requires the public service "
            "distrib_la.plan_polar_factor(mesh, n=..., "
            "backend='distributed', rcond=...); merge the distrib_la "
            "polar-factor service lane"
        ) from exc
    try:
        from symmetry_maps import (apply_band_matrix_symmetry,
                                   directed_edge_orbit_table,
                                   q_stencil_orbit_table)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "parallel-transport preprocessing requires the public symmetry "
            "service APIs for directed edges, q stencils and band-matrix "
            "actions; merge the symmetry service lane"
        ) from exc
    return (plan_polar_factor, directed_edge_orbit_table,
            apply_band_matrix_symmetry, q_stencil_orbit_table)


def _full_band_tables(wfn, sym, nbands: int) -> tuple[np.ndarray, np.ndarray]:
    """Return spin-channel-zero DFT energies/occupations on the full grid."""
    irr = np.asarray(sym.irr_idx_k, dtype=np.int32)
    energies = np.asarray(wfn.energies)
    occupations = np.asarray(wfn.occs)
    if energies.ndim == 3:
        energies = energies[0]
    if occupations.ndim == 3:
        occupations = occupations[0]
    if energies.ndim != 2 or occupations.ndim != 2:
        raise ValueError(
            "WFN energies and occupations must reduce to (nk_ibz, nb); "
            f"got {energies.shape} and {occupations.shape}")
    nb = int(nbands)
    return (np.asarray(energies[irr, :nb], dtype=np.float64),
            np.asarray(occupations[irr, :nb], dtype=np.float64))


def initialize_parallel_transport_artifact(
    path: str | Path,
    *,
    wfn,
    sym,
    mesh,
    nbands: int,
    effective_nspinor: int,
    bispinor: bool,
    velocity_dft_kmajor,
    wfn_path: str,
    wfn_fingerprint: str,
    rcond: float = 1.0e-10,
) -> None:
    """Create the schema and write exact velocity before the WFN stream.

    This is a separate transaction so the dipole producer can release its
    resident full-BZ wavefunctions and sharded velocity before loading the
    central/neighbor pairs. The only simultaneous large device objects in the
    link stage are therefore one central sphere, one neighbor sphere and one
    distributed band tile.
    """
    nb = int(nbands)
    kgrid = tuple(int(n) for n in np.asarray(wfn.kgrid).reshape(3))
    undersampled = [axis for axis, n in zip("xyz", kgrid) if n < 5]
    if undersampled:
        raise ValueError(
            "parallel-transport preprocessing requires at least five "
            "distinct mesh points along every Cartesian mesh direction for "
            "the advertised fourth-order +/-2 stencil; got "
            f"kgrid={kgrid} (undersampled axes "
            f"{','.join(undersampled)}).")
    nrk = int(np.asarray(sym.kirr_fullids).size)
    nk = int(sym.nk_tot)
    energies, occupations = _full_band_tables(wfn, sym, nb)
    velocity = jnp.asarray(velocity_dft_kmajor)
    if velocity.ndim != 4 or velocity.shape[:2] != (nk, 3):
        raise ValueError(
            "velocity_dft_kmajor must be (nk_full, 3, nb_pad, nb_pad); "
            f"got {tuple(velocity.shape)}")
    if velocity.shape[-2] != velocity.shape[-1] or velocity.shape[-1] < nb:
        raise ValueError(
            f"velocity band shape {tuple(velocity.shape[-2:])} does not "
            f"contain logical nbands={nb}")

    with SlabIO(str(path), mode="w", mesh=mesh) as io:
        io.create_dataset(
            LINKS_DATASET, shape=(nrk, 3, nb, nb), dtype=np.complex128,
            attrs={
                "k_storage": "ibz_source_edges",
                "orientation": "L_i(k) X(k+b_i) L_i(k)^H",
                "source_steps": "positive reduced-grid unit steps",
                "band_layout": "P(None,None,x,y)",
                "gauge": "WfnLoader generated full-BZ gauge",
            })
        io.create_dataset(
            SINGULAR_VALUES_DATASET, shape=(nrk, 3, nb), dtype=np.float64,
            attrs={
                "k_storage": "ibz_source_edges",
                "source_steps": "positive reduced-grid unit steps",
                "ordering": "descending",
                "distribution": "replicated O(nband) diagnostic",
            })
        io.create_dataset(
            VELOCITY_DFT_DATASET, shape=(3, nk, nb, nb),
            dtype=np.complex128,
            attrs={
                "k_storage": "full_bz",
                "components": "Cartesian",
                "band_layout": "P(None,None,x,y)",
                "units": "WFN velocity convention",
                "manifold": "bands [band_start, band_stop)",
            })
        io.create_dataset(
            CONNECTION_REDUCED_DATASET, shape=(3, nk, nb, nb),
            dtype=np.complex128,
            attrs={
                "k_storage": "full_bz",
                "components": "reduced reciprocal coordinates",
                "band_layout": "P(None,None,x,y)",
                "hermitian": True,
                "finite_difference_order": 4,
            })
        io.create_dataset(
            CONNECTION_CART_DATASET, shape=(3, nk, nb, nb),
            dtype=np.complex128,
            attrs={
                "k_storage": "full_bz",
                "components": "Cartesian",
                "band_layout": "P(None,None,x,y)",
                "hermitian": True,
                "conversion": "A_cart = B^{-T} A_reduced",
            })
        velocity_dir_major = jnp.moveaxis(velocity, 1, 0)
        velocity_dir_major = jax.lax.with_sharding_constraint(
            velocity_dir_major,
            NamedSharding(mesh, P(None, None, "x", "y")))
        io.write_slab(VELOCITY_DFT_DATASET, velocity_dir_major)
        io.write_attr(ENERGIES_DATASET, energies)
        io.write_attr(OCCUPATIONS_DATASET, occupations)
        io.write_attr("schema_version", np.int32(SCHEMA_VERSION))
        io.write_attr("band_start", np.int32(0))
        io.write_attr("band_stop", np.int32(nb))
        io.write_attr("spin_channel", np.int32(0))
        io.write_attr("effective_nspinor", np.int32(effective_nspinor))
        io.write_attr("bispinor", np.int32(bool(bispinor)))
        io.write_attr("kgrid", np.asarray(wfn.kgrid, dtype=np.int32))
        io.write_attr("kgrid_shift", np.asarray(wfn.shift, dtype=np.float64))
        io.write_attr("reduced_spacing",
                      1.0 / np.asarray(wfn.kgrid, dtype=np.float64))
        io.write_attr(
            "reciprocal_lattice_cart",
            np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat))
        io.write_attr("irr_idx_k", np.asarray(sym.irr_idx_k, dtype=np.int32))
        io.write_attr("sym_idx_k", np.asarray(sym.sym_idx_k, dtype=np.int32))
        io.write_attr(
            "wfn_path_utf8",
            np.frombuffer(str(wfn_path).encode("utf-8"), dtype=np.uint8))
        io.write_attr(
            "wfn_fingerprint_utf8",
            np.frombuffer(
                str(wfn_fingerprint).encode("utf-8"), dtype=np.uint8))
        io.write_attr("polar_rcond", np.float64(rcond))
        # Numeric convention stamps are SlabIO-readable on every backend.
        # 1 means the sole supported convention documented by this schema.
        io.write_attr("energy_units_ry", np.int32(1))
        io.write_attr("velocity_units_ry_bohr", np.int32(1))
        io.write_attr("velocity_frame_cartesian", np.int32(1))
        io.write_attr("reciprocal_units_bohr_inverse", np.int32(1))
        io.write_attr("connection_complete", np.int32(0))
        io.write_attr("velocity_validation_complete", np.int32(0))
        io.write_attr("velocity_validation_passed", np.int32(0))


def _write_link_stage(
    path: str,
    *,
    wfn,
    sym,
    mesh,
    nbands: int,
    bispinor: bool,
    polar_plan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream IBZ wavefunctions and write links plus fixed-reference data."""
    from wfn_loader import IBZRows

    nb = int(nbands)
    source_full = np.asarray(sym.kirr_fullids, dtype=np.int32)
    nrk = int(source_full.size)
    full_plus = build_forward_neighbor_table(sym.kvecs_asints, wfn.kgrid)
    source_plus = full_plus[source_full]
    wraps = np.empty((nrk, 3, 3), dtype=np.int32)
    singular_values = []
    center_on_x, link_kernel = make_cross_k_link(mesh, polar_plan)
    source_steps = np.eye(3, dtype=np.int32)

    with SlabIO(path, mode="a", mesh=mesh) as io:
        for ik_irr, center_full in enumerate(source_full):
            center_ids = IBZRows((int(ik_irr),))
            center_xy = wfn.load(
                bands=(0, nb), k=center_ids, sharding=band_sphere_spec(),
                bispinor=bool(bispinor))
            center_x = center_on_x(center_xy)
            g_center = wfn.gvecs(k=center_ids)[0]
            ngk_center = int(wfn.ngk_valid(k=center_ids)[0])

            for idir in range(3):
                neighbor_full = int(source_plus[ik_irr, idir])
                neighbor_ids = [neighbor_full]
                wrap = g_wrap_for_forward_step(
                    sym.unfolded_kpts, int(center_full), neighbor_full,
                    idir, wfn.kgrid)
                wraps[ik_irr, idir] = wrap
                g_neighbor = wfn.gvecs(k=neighbor_ids)[0]
                ngk_neighbor = int(wfn.ngk_valid(k=neighbor_ids)[0])
                g_index, g_valid = build_g_wrap_lookup(
                    g_neighbor, g_center, wrap,
                    ngk_neighbor=ngk_neighbor, ngk_center=ngk_center)

                neighbor_xy = wfn.load(
                    bands=(0, nb), k=neighbor_ids,
                    sharding=band_sphere_spec(), bispinor=bool(bispinor))
                # WfnLoader owns the exact zero band pad.  distrib_la's
                # polar contract returns the canonical partial isometry for
                # those null directions; slicing the physical leading block
                # is therefore well-defined and does not perturb the link.
                link, values = link_kernel(
                    center_x, neighbor_xy, g_index, g_valid)
                io.write_slab(
                    LINKS_DATASET, link[None, None, :, :],
                    offset=(ik_irr, idir, 0, 0),
                    global_shape=(nrk, 3, nb, nb))
                # WfnLoader owns a second collective HDF5 handle.  Finish
                # this asynchronous append before its next streamed read.
                io.sync_writes()
                # Retain only the replicated O(nb) diagnostic. Stacking and
                # writing once after the stream avoids both a per-link host
                # synchronization and a second HDF5 transaction per link.
                singular_values.append(values[:nb])
                del neighbor_xy, link, values
            del center_xy, center_x

        singular_values_device = jnp.stack(singular_values).reshape(
            nrk, 3, nb)
        singular_values_device = jax.lax.with_sharding_constraint(
            singular_values_device, NamedSharding(mesh, P()))
        io.write_slab(
            SINGULAR_VALUES_DATASET, singular_values_device,
            global_shape=(nrk, 3, nb))
        io.write_attr("source_steps", source_steps)
        io.write_attr("source_full_ids", source_full)
        io.write_attr("source_neighbor_full_ids", source_plus)
        io.write_attr("g_wrap_ibz", wraps)
        io.write_attr("full_forward_neighbors", full_plus)
    return full_plus, source_full, source_steps


def _write_connection_stage(
    path: str,
    *,
    wfn,
    sym,
    mesh,
    nbands: int,
    full_plus: np.ndarray,
    source_full: np.ndarray,
    source_steps: np.ndarray,
    directed_edge_orbit_table,
    apply_band_matrix_symmetry,
) -> None:
    """Read compact links, unfold through SymMaps, and write A once."""
    nb = int(nbands)
    nb_storage = band_storage_extent(mesh, nb)
    table = directed_edge_orbit_table(
        kgrid=np.asarray(wfn.kgrid, dtype=np.int32),
        kgrid_shift=np.asarray(wfn.shift, dtype=np.float64),
        sym_mats_k=np.asarray(sym.sym_mats_k, dtype=np.int32),
        irr_idx_k=np.asarray(sym.irr_idx_k, dtype=np.int32),
        sym_idx_k=np.asarray(sym.sym_idx_k, dtype=np.int32),
        source_full_ids=source_full,
        source_steps=source_steps,
        n_sym_spatial=int(wfn.ntran),
        target_steps=np.eye(3, dtype=np.int32),
    )
    block_spec = P(None, None, "x", "y")
    block_sharding = NamedSharding(mesh, block_spec)
    band_matmul = make_distributed_band_matmul(mesh, n_batch_axes=1)

    with SlabIO(path, mode="a", mesh=mesh) as io:
        source_links = io.read_slab(
            LINKS_DATASET,
            shape=(int(source_full.size), 3, nb_storage, nb_storage),
            partition_spec=block_spec)
        selected = source_links[
            table["source_row"], table["source_direction"]]
        full_target_major = apply_band_matrix_symmetry(
            selected,
            antiunitary=table["antiunitary"],
            reverse=table["reverse"],
            sewing_start=None,
            sewing_end=None,
        )
        full_links = jnp.moveaxis(full_target_major, 1, 0)
        full_links = jax.lax.with_sharding_constraint(
            full_links, block_sharding)

        spacing = 1.0 / np.asarray(wfn.kgrid, dtype=np.float64)
        @jax.jit
        def _connection(links):
            return fourth_order_connection(
                links, full_plus, spacing, band_matmul=band_matmul)

        connection_reduced = _connection(full_links)
        connection_reduced = jax.lax.with_sharding_constraint(
            connection_reduced, block_sharding)

        reciprocal = (
            np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat))
        try:
            from gw.qsgw_head import reduced_covector_to_cartesian
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "parallel-transport connection conversion requires "
                "gw.qsgw_head.reduced_covector_to_cartesian; merge the QSGW "
                "head service lane") from exc

        @jax.jit
        def _to_cart(A):
            out = reduced_covector_to_cartesian(A, reciprocal)
            out = 0.5 * (out + jnp.swapaxes(jnp.conj(out), -1, -2))
            return jax.lax.with_sharding_constraint(out, block_sharding)

        connection_cart = _to_cart(connection_reduced)
        connection_shape = (3, int(sym.nk_tot), nb, nb)
        io.write_slab(
            CONNECTION_REDUCED_DATASET, connection_reduced,
            global_shape=connection_shape)
        io.write_slab(
            CONNECTION_CART_DATASET, connection_cart,
            global_shape=connection_shape)
        io.write_attr("directed_edge_source_row", table["source_row"])
        io.write_attr(
            "directed_edge_source_direction", table["source_direction"])
        io.write_attr("directed_edge_sym_idx", table["sym_idx"])
        io.write_attr("directed_edge_reverse", table["reverse"])
        io.write_attr("directed_edge_antiunitary", table["antiunitary"])
        io.write_attr("connection_complete", np.int32(1))


def _w_av_seed_steps(
    first_neighbors: bool,
    second_neighbors: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stencil seeds and per-seed shell bits (first=1, second=2)."""
    steps = []
    shells = []
    axes = np.eye(3, dtype=np.int32)
    if bool(first_neighbors):
        for step in np.concatenate([axes, -axes], axis=0):
            steps.append(step)
            shells.append(1)
    if bool(second_neighbors):
        for step in np.concatenate([2 * axes, -2 * axes], axis=0):
            steps.append(step)
            shells.append(2)
        for iaxis in range(3):
            for jaxis in range(iaxis + 1, 3):
                for sign_i in (-1, 1):
                    for sign_j in (-1, 1):
                        step = np.zeros(3, dtype=np.int32)
                        step[iaxis] = sign_i
                        step[jaxis] = sign_j
                        steps.append(step)
                        shells.append(2)
    if not steps:
        return (np.empty((0, 3), dtype=np.int32),
                np.empty((0,), dtype=np.int32))
    return (np.asarray(steps, dtype=np.int32),
            np.asarray(shells, dtype=np.int32))


def _write_w_av_stage(
    path: str,
    *,
    wfn,
    sym,
    mesh,
    nbands: int,
    bispinor: bool,
    first_neighbors: bool,
    second_neighbors: bool,
    q_stencil_orbit_table,
    mode: str = "a",
    wfn_path: str = "",
    wfn_fingerprint: str = "",
) -> None:
    """Stream symmetry-reduced W-av density vertices into a SlabIO artifact."""
    seeds, seed_shells = _w_av_seed_steps(
        first_neighbors, second_neighbors)
    if seeds.shape[0] == 0:
        return
    table = q_stencil_orbit_table(
        kgrid=np.asarray(wfn.kgrid, dtype=np.int32),
        sym_mats_k=np.asarray(sym.sym_mats_k, dtype=np.int32),
        irr_idx_q=np.asarray(sym.irr_idx_q, dtype=np.int32),
        sym_idx_q=np.asarray(sym.sym_idx_q, dtype=np.int32),
        seed_steps=seeds,
        n_sym_spatial=int(wfn.ntran),
        allow_trs=bool(sym.trs_allowed),
    )
    source_steps = np.asarray(table["source_steps"], dtype=np.int32)
    target_seed_mask = np.asarray(table["target_seed_mask"], dtype=np.int32)
    target_shell_mask = np.bitwise_or.reduce(
        target_seed_mask * seed_shells[None, :], axis=1).astype(np.int32)
    source_shell_mask = target_shell_mask[
        np.asarray(table["source_target_row"], dtype=np.int32)]

    nb = int(nbands)
    nq = int(source_steps.shape[0])
    nk = int(sym.nk_tot)
    kgrid = np.asarray(wfn.kgrid, dtype=np.int32)
    qpoints_frac = source_steps / kgrid[None, :]
    target_qpoints_frac = (
        np.asarray(table["target_steps"], dtype=np.float64)
        / kgrid[None, :])
    neighbor_full = build_neighbor_table(
        np.asarray(sym.kvecs_asints, dtype=np.int32), kgrid, source_steps)
    wraps = np.empty((nq, nk, 3), dtype=np.int32)
    center_on_x, overlap = make_cross_k_overlap(mesh)
    from common.gpu_utils import get_device_memory_info
    from runtime.xla_memory import resolve_xla_gpu_memory_env
    xla_memory = resolve_xla_gpu_memory_env()
    if (jax.default_backend() == "gpu"
            and xla_memory.allocator != "platform"):
        raise RuntimeError(
            "W_AV-QSTREAM-ALLOCATOR refusal: got "
            f"XLA_PYTHON_CLIENT_ALLOCATOR={xla_memory.allocator_raw!r}; "
            "want 'platform' for finite-q W-av preprocessing. The PHDF5 "
            "q-stream reproducibly segfaults with BFC during the first "
            "collective write. Fix: export "
            "XLA_PYTHON_CLIENT_ALLOCATOR=platform before lx run; see "
            "docs/dev/env_vars.md.")
    memory = get_device_memory_info()
    tile_budget_bytes = 0.20 * float(memory["total_gb"]) * 1.0e9
    px = int(mesh.shape["x"])
    py = int(mesh.shape["y"])
    nb_storage = band_storage_extent(mesh, nb)
    nspinor = 4 if bool(bispinor) else int(wfn.nspinor)
    sphere_bytes = nspinor * int(wfn.ngkmax) * np.dtype(np.complex128).itemsize
    fixed_bytes = sphere_bytes * (
        nb_storage // px + nb_storage // (px * py))
    streamed_q_bytes = (
        sphere_bytes * (
            nb_storage // py + nb_storage // (px * py))
        + (nb_storage // px) * (nb_storage // py)
        * np.dtype(np.complex128).itemsize)
    # q is the streaming boundary. The factor of two reserves both the
    # input and the gathered/aligned neighbor while the overlap is live.
    working_bytes_per_k = fixed_bytes + 2 * streamed_q_bytes
    if working_bytes_per_k > tile_budget_bytes:
        raise MemoryError(
            "W_AV-QSTREAM-TILE refusal: the smallest k=1, q=1 raw-density "
            f"tile needs an estimated {working_bytes_per_k / 2**30:.2f} "
            f"GiB/device, above the {tile_budget_bytes / 2**30:.2f} GiB "
            "W-av budget. Fix: use a band-pair-streamed or fused screening "
            "consumer; reducing the k or q batch cannot make this path fit.")
    k_batch = max(1, min(
        nk, int(tile_budget_bytes // max(1, working_bytes_per_k))))
    if jax.process_index() == 0:
        working_gib = k_batch * working_bytes_per_k / 2**30
        artifact_gib = (
            nq * nk * nb * nb * np.dtype(np.complex128).itemsize / 2**30)
        print(
            f"  W-av streaming: k={k_batch}/{nk}, q=1/{nq} "
            f"resident (conservative overlap working set "
            f"{working_gib:.2f} GiB/device; "
            f"20% of {float(memory['total_gb']):.1f} GB device memory, "
            f"source {memory['source']}); WFN source=center+neighbor union; "
            f"raw artifact={artifact_gib:.2f} GiB")
    g_full = wfn.gvecs(k="full_bz")
    ngk_full = wfn.ngk_valid(k="full_bz")

    with SlabIO(path, mode=mode, mesh=mesh) as io:
        io.create_dataset(
            W_AV_DENSITY_DATASET,
            shape=(nq, nk, nb, nb),
            dtype=np.complex128,
            attrs={
                "orientation": "rho_nm(k,q)=<u_n,k|u_m,k+q>",
                "q_storage": "symmetry_reduced_stencil",
                "k_storage": "full_bz",
                "band_layout": "P(None,None,x,y)",
                "spinor_contraction": "sum over every stored component",
                "unfold_stage": "after full-k response sum",
            })
        io.write_attr("w_av_complete", np.asarray([0], dtype=np.int32))
        if mode == "w":
            io.write_attr(
                "artifact_kind_utf8",
                np.frombuffer(b"w_av_finite_q_stencil", dtype=np.uint8))
            io.write_attr("band_start", np.asarray([0], dtype=np.int32))
            io.write_attr("band_stop", np.asarray([nb], dtype=np.int32))
            io.write_attr("spin_channel", np.asarray([0], dtype=np.int32))
            io.write_attr(
                "effective_nspinor",
                np.asarray(
                    [4 if bool(bispinor) else int(wfn.nspinor)],
                    dtype=np.int32))
            io.write_attr(
                "bispinor", np.asarray([bool(bispinor)], dtype=np.int32))
            io.write_attr("kgrid", kgrid)
            io.write_attr(
                "kgrid_shift", np.asarray(wfn.shift, dtype=np.float64))
            io.write_attr(
                "reduced_spacing", 1.0 / kgrid.astype(np.float64))
            io.write_attr(
                "reciprocal_lattice_cart",
                np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat))
            io.write_attr(
                "reciprocal_units_bohr_inverse",
                np.asarray([1], dtype=np.int32))
            io.write_attr(
                "wfn_path_utf8",
                np.frombuffer(str(wfn_path).encode("utf-8"), dtype=np.uint8))
            io.write_attr(
                "wfn_fingerprint_utf8",
                np.frombuffer(
                    str(wfn_fingerprint).encode("utf-8"), dtype=np.uint8))
            # SlabIO's collective reader deliberately supports the numeric
            # compute dtypes, not uint8.  Keep the historical uint8 spelling
            # for external HDF5 tools and add an int32 byte view for every
            # production consumer that must remain behind the SlabIO door.
            io.write_attr(
                "wfn_path_bytes_i32",
                np.frombuffer(str(wfn_path).encode("utf-8"),
                              dtype=np.uint8).astype(np.int32))
            io.write_attr(
                "wfn_fingerprint_bytes_i32",
                np.frombuffer(str(wfn_fingerprint).encode("utf-8"),
                              dtype=np.uint8).astype(np.int32))
        for k_start in range(0, nk, k_batch):
            k_stop = min(k_start + k_batch, nk)
            center_ids = list(range(k_start, k_stop))
            n_center = k_stop - k_start
            for iq in range(nq):
                neighbor_ids = np.asarray(
                    neighbor_full[k_start:k_stop, iq], dtype=np.int32)
                # One loader call owns the complete q tile. This avoids
                # retaining a loader result across a second collective load,
                # while keeping residency bounded to one center and one
                # neighbor bank irrespective of the stencil size.
                windows_xy = wfn.load(
                    bands=(0, nb),
                    k=[*center_ids, *neighbor_ids.tolist()],
                    sharding=band_sphere_spec(), bispinor=bool(bispinor))
                center_x = center_on_x(windows_xy[:n_center])
                neighbor_xy = windows_xy[n_center:, None, ...]
                g_indices = np.empty(
                    (n_center, 1, int(wfn.ngkmax)), dtype=np.int32)
                g_masks = np.empty_like(g_indices, dtype=bool)
                for ik_local, ik in enumerate(center_ids):
                    g_center = g_full[ik]
                    ngk_center = int(ngk_full[ik])
                    ineighbor = int(neighbor_ids[ik_local])
                    wrap = g_wrap_for_step(
                        sym.unfolded_kpts, ik, ineighbor,
                        source_steps[iq], kgrid)
                    wraps[iq, ik] = wrap
                    g_index, g_valid = build_g_wrap_lookup(
                        g_full[ineighbor], g_center, wrap,
                        ngk_neighbor=int(ngk_full[ineighbor]),
                        ngk_center=ngk_center)
                    g_indices[ik_local, 0] = g_index
                    g_masks[ik_local, 0] = g_valid
                rho = overlap(
                    center_x, neighbor_xy, g_indices, g_masks)
                io.write_slab(
                    W_AV_DENSITY_DATASET, rho,
                    offset=(iq, k_start, 0, 0),
                    global_shape=(nq, nk, nb, nb))
                io.sync_writes()
                del windows_xy, center_x, neighbor_xy, rho

        io.write_attr(
            "w_av_schema_version",
            np.asarray([W_AV_SCHEMA_VERSION], dtype=np.int32))
        io.write_attr("w_av_complete", np.asarray([1], dtype=np.int32))
        io.write_attr("w_av_first_neighbors",
                      np.asarray([bool(first_neighbors)], dtype=np.int32))
        io.write_attr("w_av_second_neighbors",
                      np.asarray([bool(second_neighbors)], dtype=np.int32))
        io.write_attr("w_av_seed_steps", seeds)
        io.write_attr("w_av_seed_shell_mask", seed_shells)
        io.write_attr("w_av_source_qsteps", source_steps)
        io.write_attr("w_av_source_qpoints_frac", qpoints_frac)
        io.write_attr("w_av_source_q_full_ids",
                      np.asarray(table["source_full_ids"], dtype=np.int32))
        io.write_attr("w_av_source_q_irr_idx",
                      np.asarray(table["source_irr_idx"], dtype=np.int32))
        io.write_attr("w_av_source_shell_mask", source_shell_mask)
        io.write_attr("w_av_source_kplusq_full",
                      np.asarray(neighbor_full.T, dtype=np.int32))
        io.write_attr("w_av_source_g_wrap", wraps)
        io.write_attr("w_av_target_qsteps",
                      np.asarray(table["target_steps"], dtype=np.int32))
        io.write_attr("w_av_target_qpoints_frac", target_qpoints_frac)
        io.write_attr("w_av_target_q_full_ids",
                      np.asarray(table["target_full_ids"], dtype=np.int32))
        io.write_attr("w_av_target_source_row",
                      np.asarray(table["target_source_row"], dtype=np.int32))
        io.write_attr("w_av_target_sym_idx",
                      np.asarray(table["target_sym_idx"], dtype=np.int32))
        io.write_attr("w_av_target_antiunitary",
                      np.asarray(table["target_antiunitary"], dtype=np.int32))
        io.write_attr("w_av_target_shell_mask", target_shell_mask)
        io.write_attr("w_av_target_seed_mask",
                      np.asarray(target_seed_mask, dtype=np.int32))
    if jax.process_index() == 0:
        print(
            f"  W-av stencil: stored {nq} symmetry-inequivalent q rows "
            f"for {int(np.shape(table['target_steps'])[0])} target neighbors")


def write_w_av_stencil_artifact(
    path: str | Path,
    *,
    wfn,
    sym,
    mesh,
    nbands: int,
    bispinor: bool,
    first_neighbors: bool,
    second_neighbors: bool,
    wfn_path: str,
    wfn_fingerprint: str,
) -> None:
    """Write the standalone finite-q W-av artifact without PT preprocessing."""
    if not bool(first_neighbors) and not bool(second_neighbors):
        raise ValueError(
            "standalone W-av preprocessing requires at least one enabled "
            "neighbor shell")
    try:
        from symmetry_maps import q_stencil_orbit_table
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "W-av preprocessing requires the public symmetry service "
            "q_stencil_orbit_table API") from exc
    _write_w_av_stage(
        str(path), wfn=wfn, sym=sym, mesh=mesh, nbands=int(nbands),
        bispinor=bool(bispinor),
        first_neighbors=bool(first_neighbors),
        second_neighbors=bool(second_neighbors),
        q_stencil_orbit_table=q_stencil_orbit_table,
        mode="w", wfn_path=str(wfn_path),
        wfn_fingerprint=str(wfn_fingerprint))


def write_parallel_transport_artifact(
    path: str | Path,
    *,
    wfn,
    sym,
    mesh,
    nbands: int,
    bispinor: bool,
    rcond: float = 1.0e-10,
    w_av_first_neighbors: bool = False,
    w_av_second_neighbors: bool = False,
) -> None:
    """Append streamed links and the full-BZ connection to an initialized file."""
    plan_polar, edge_table, apply_symmetry, q_stencil_table = (
        _require_service_apis())
    nb_padded = band_storage_extent(mesh, int(nbands))
    polar_plan = plan_polar(
        mesh, n=nb_padded, backend="distributed", rcond=float(rcond))
    full_plus, source_full, source_steps = _write_link_stage(
        str(path), wfn=wfn, sym=sym, mesh=mesh, nbands=int(nbands),
        bispinor=bool(bispinor), polar_plan=polar_plan)
    _write_connection_stage(
        str(path), wfn=wfn, sym=sym, mesh=mesh, nbands=int(nbands),
        full_plus=full_plus, source_full=source_full,
        source_steps=source_steps,
        directed_edge_orbit_table=edge_table,
        apply_band_matrix_symmetry=apply_symmetry)

    if bool(w_av_first_neighbors) or bool(w_av_second_neighbors):
        _write_w_av_stage(
            str(path), wfn=wfn, sym=sym, mesh=mesh, nbands=int(nbands),
            bispinor=bool(bispinor),
            first_neighbors=bool(w_av_first_neighbors),
            second_neighbors=bool(w_av_second_neighbors),
            q_stencil_orbit_table=q_stencil_table)


def complete_velocity_validation(
    path: str | Path,
    *,
    mesh,
    kgrid,
    bvec_cart,
    energies_full,
    connection_cart,
    velocity_exact_cart,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Run and stamp the mandatory DFT covariant-velocity reconstruction.

    The spectral derivative import is lazy because ``gw.qsgw_head`` is
    owned by the self-consistent-head lane. Both preprocessing and QSGW call
    that one service spelling: one flat-k forward FFT, three inverse FFTs,
    the signed ``i 2*pi R`` multiplier and reduced-to-Cartesian conversion.
    """
    try:
        from gw.qsgw_head import covariant_cartesian_derivative
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "parallel-transport validation requires "
            "gw.qsgw_head.covariant_cartesian_derivative; merge the QSGW "
            "head service lane") from exc

    energies = jnp.asarray(energies_full)
    A = jnp.asarray(connection_cart)
    exact = jnp.asarray(velocity_exact_cart)
    if energies.ndim != 2:
        raise ValueError(
            f"energies_full must be (nk, nb); got {energies.shape}")
    if A.shape[:2] != (3, energies.shape[0]) \
            or A.shape[-2:] != (energies.shape[1], energies.shape[1]):
        raise ValueError(
            f"connection shape {A.shape} does not match energies "
            f"{energies.shape}")
    h_sharding = NamedSharding(mesh, P(None, "x", "y"))

    def _diagonal_hamiltonian(e):
        return jax.vmap(jnp.diag)(e).astype(A.dtype)

    _diagonal_hamiltonian = jax.jit(
        _diagonal_hamiltonian, out_shardings=h_sharding)
    H = _diagonal_hamiltonian(energies.astype(A.real.dtype))
    reconstructed = covariant_cartesian_derivative(
        H, A, mesh=mesh, kgrid=tuple(int(x) for x in kgrid),
        bvec_cart=np.asarray(bvec_cart, dtype=np.float64))
    metrics = _velocity_error_metrics(
        reconstructed, exact, atol=float(atol), rtol=float(rtol))
    write_velocity_validation(path, mesh=mesh, metrics=metrics)
    if not metrics["passed"]:
        raise RuntimeError(
            "parallel-transport DFT velocity validation failed: "
            f"max_abs={metrics['max_abs']:.6e}, "
            f"max_rel={metrics['max_rel']:.6e}, "
            f"atol={metrics['atol']:.6e}, rtol={metrics['rtol']:.6e}")
    return metrics


def validate_parallel_transport_artifact(
    path: str | Path,
    *,
    mesh,
    kgrid,
    bvec_cart,
    nbands: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Read the artifact through SlabIO and execute its mandatory gate."""
    nb = int(nbands)
    nb_storage = band_storage_extent(mesh, nb)
    nk = int(np.prod(tuple(int(x) for x in kgrid)))
    with SlabIO(str(path), mode="r", mesh=mesh) as io:
        connection = io.read_slab(
            CONNECTION_CART_DATASET,
            shape=(3, nk, nb_storage, nb_storage),
            partition_spec=P(None, None, "x", "y"))
        velocity = io.read_slab(
            VELOCITY_DFT_DATASET,
            shape=(3, nk, nb_storage, nb_storage),
            partition_spec=P(None, None, "x", "y"))
        energies = io.read_slab(
            ENERGIES_DATASET,
            shape=(nk, nb_storage),
            partition_spec=P(None, ("x", "y")))
    if nb <= 0 or nb > int(np.shape(energies)[1]):
        raise ValueError(
            f"nbands={nb} outside SlabIO energy extent {np.shape(energies)[1]}")
    return complete_velocity_validation(
        path, mesh=mesh, kgrid=kgrid, bvec_cart=bvec_cart,
        energies_full=energies,
        connection_cart=connection,
        velocity_exact_cart=velocity,
        atol=float(atol), rtol=float(rtol))


def covariant_velocity(
    hamiltonian_derivative_cart,
    connection_cart,
    hamiltonian_dft,
    *,
    band_matmul,
):
    """Return dH/dk_i - i[A_i, H] in the fixed reference basis."""
    dH = jnp.asarray(hamiltonian_derivative_cart)
    A = jnp.asarray(connection_cart)
    H = jnp.asarray(hamiltonian_dft)
    if dH.shape != A.shape:
        raise ValueError(f"dH and A shapes differ: {dH.shape} vs {A.shape}")
    if H.shape != A.shape[1:]:
        raise ValueError(
            f"H must have shape {A.shape[1:]}; got {tuple(H.shape)}")
    commutator = band_matmul(A, H[None]) - band_matmul(H[None], A)
    return dH - 1.0j * commutator


def validate_covariant_velocity(
    hamiltonian_derivative_cart,
    connection_cart,
    hamiltonian_dft,
    velocity_exact_cart,
    *,
    band_matmul,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare every complex diagonal/off-diagonal reconstructed velocity.

    The spectral derivative is deliberately an input: the QSGW/head lane owns
    the one shared full-k FFT derivative service, so preprocessing does not
    introduce a second FFT implementation.
    """
    reconstructed = covariant_velocity(
        hamiltonian_derivative_cart, connection_cart, hamiltonian_dft,
        band_matmul=band_matmul)
    exact = jnp.asarray(velocity_exact_cart)
    return _velocity_error_metrics(
        reconstructed, exact, atol=float(atol), rtol=float(rtol))


def _velocity_error_metrics(
    reconstructed,
    exact,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Reduce all complex matrix entries to the mandatory gate metrics."""
    reconstructed = jnp.asarray(reconstructed)
    exact = jnp.asarray(exact)
    if reconstructed.shape != exact.shape:
        raise ValueError(
            f"reconstructed/exact shapes differ: {reconstructed.shape} "
            f"vs {exact.shape}")
    error = jnp.abs(reconstructed - exact)
    scale = jnp.maximum(jnp.abs(exact), float(atol))
    relative = error / scale
    nband = exact.shape[-1]
    diagonal = jnp.eye(nband, dtype=bool)[None, None, :, :]
    off_diagonal = ~diagonal
    max_abs = float(jax.device_get(jnp.max(error)))
    max_rel = float(jax.device_get(jnp.max(relative)))
    max_abs_diag = float(jax.device_get(jnp.max(
        jnp.where(diagonal, error, 0.0))))
    max_abs_offdiag = float(jax.device_get(jnp.max(
        jnp.where(off_diagonal, error, 0.0))))
    passed = bool(jax.device_get(jnp.all(
        error <= float(atol) + float(rtol) * jnp.abs(exact))))
    return {
        "passed": passed,
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_abs_diagonal": max_abs_diag,
        "max_abs_offdiagonal": max_abs_offdiag,
    }


def write_velocity_validation(
    path: str | Path,
    *,
    mesh,
    metrics: dict[str, object],
) -> None:
    """Stamp mandatory DFT reconstruction gate metrics through SlabIO."""
    required = {
        "passed", "atol", "rtol", "max_abs", "max_rel",
        "max_abs_diagonal", "max_abs_offdiagonal",
    }
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"velocity validation metrics missing {sorted(missing)}")
    with SlabIO(str(path), mode="a", mesh=mesh) as io:
        io.write_attr("velocity_validation_complete", np.int32(1))
        io.write_attr(
            "velocity_validation_passed", np.int32(bool(metrics["passed"])))
        for name in sorted(required - {"passed"}):
            io.write_attr(
                f"velocity_validation_{name}", np.float64(metrics[name]))
