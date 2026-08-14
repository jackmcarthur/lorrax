"""SlabIO artifact for the fixed-gauge parallel-transport head data.

This module owns the whole preprocessing transaction. It streams one raw IBZ
centre and one positive neighbour at a time, writes compact links, then
re-reads those links through SlabIO for symmetry unfolding and fourth-order
connection construction. No wavefunction survives the streamed link stage,
and every HDF5 payload or metadata item crosses the SlabIO service door.
"""
from __future__ import annotations

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
    fourth_order_connection,
    g_wrap_for_forward_step,
    make_cross_k_link,
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
    "covariant_velocity",
    "initialize_parallel_transport_artifact",
    "validate_covariant_velocity",
    "validate_parallel_transport_artifact",
    "write_parallel_transport_artifact",
    "write_velocity_validation",
]


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
                                   directed_edge_orbit_table)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "parallel-transport preprocessing requires the public symmetry "
            "service APIs directed_edge_orbit_table and "
            "apply_band_matrix_symmetry; merge the symmetry service lane"
        ) from exc
    return (plan_polar_factor, directed_edge_orbit_table,
            apply_band_matrix_symmetry)


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


def write_parallel_transport_artifact(
    path: str | Path,
    *,
    wfn,
    sym,
    mesh,
    nbands: int,
    bispinor: bool,
    rcond: float = 1.0e-10,
) -> None:
    """Append streamed links and the full-BZ connection to an initialized file."""
    plan_polar, edge_table, apply_symmetry = _require_service_apis()
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
