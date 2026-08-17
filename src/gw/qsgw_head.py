"""Parallel-transport covariant velocity for the self-consistent GW head.

The preprocessing job owns wavefunctions.  This module deliberately does
not: its inputs are the saved Berry connection ``A_cart``, the independently
exact DFT velocity, and the current fixed-DFT-basis QSGW Hamiltonian.  It
implements

    D_k H = partial_k H - i [A, H]
    v_Q   = v_DFT + D_k (H_Q - H_DFT)

and rebuilds the tiny Cartesian S tensor from the resulting band-tiled
velocity.  When the current centroid wavefunctions are supplied it also
builds the two q-linear head/body wings.  The wings stay centroid-sharded;
only the final ``(n_omega, 3, 3)`` Schur-reduced tensor is replicated.

Units and coordinates
---------------------
Hamiltonians and frequencies are Ry.  ``bvec_cart`` has reciprocal lattice
vectors as rows in 1/bohr, matching ``blat * WfnLoader.bvec``.  The FFT is
over reduced coordinates kappa; multiplication by ``2*pi*R`` differentiates
with respect to kappa and ``B^{-1}`` converts that covector to Cartesian k.
There is no extra hbar conversion in LORRAX's Ry/bohr velocity convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map


__all__ = [
    "DftVelocityHeadData",
    "IterationHeadResponse",
    "IterationHeadSamples",
    "ParallelTransportHeadData",
    "assemble_delta_head_manifold",
    "assemble_head_manifold",
    "build_iteration_head_samples",
    "build_iteration_head_response",
    "covariant_cartesian_derivative",
    "covariant_structured_delta",
    "head_s_tensor_sharded",
    "head_wings_sharded",
    "static_head_wings_sharded",
    "head_samples_from_s",
    "finalize_iteration_head_sample",
    "finalize_iteration_head_samples",
    "load_dft_velocity_head",
    "load_parallel_transport_head",
    "reduced_covector_to_cartesian",
    "rotate_velocity_active_to_qp",
    "rotate_velocity_to_qp",
    "spectral_cartesian_derivative",
    "validate_dft_velocity_identity",
]


# Factory results are keyed by the mesh identity and static shape facts.  The
# SC loop calls these functions from Python, so constructing an uncached jit
# in the iteration body would pay a compile for every iteration.
_KERNEL_CACHE: dict[tuple, Callable] = {}


# Bound the only frequency-by-band-pair temporary in the direct wing kernel.
# The full Y/Z outputs are much smaller (three Cartesian rows/columns), and a
# ring step visits every frequency block before circulating its band tile.
_HEAD_WING_FREQUENCY_BLOCK = 8


@dataclass(frozen=True)
class ParallelTransportHeadData:
    """Validated, device-resident inputs held across the SC loop."""

    connection_cart: jax.Array
    velocity_dft_cart: jax.Array
    nb_logical: int
    reciprocal_lattice_cart: np.ndarray
    validation: dict[str, float]


@dataclass(frozen=True)
class DftVelocityHeadData:
    """The same head inputs MINUS the Berry connection.

    ``sc_head_update = dft_velocity`` runs the metallic head chain on the
    exact DFT p-matrix velocity written by
    ``get_dipole_mtxels --parallel-transport`` and NOTHING else from that
    artifact: no connection, so no covariant ``DΔH`` correction to the
    velocity, so no dependence on the link/rotation stage.  The velocity is
    still rotated into the current QP basis every iteration by the same
    ``U`` the head carry threads — the approximation is confined to the
    ΔH-induced *change* of the velocity operator, which this mode drops.

    ``connection_cart`` is a field, pinned at ``None``, so that every
    consumer can ask one object the same question and branch on the answer
    instead of on the mode string.

    This is the configuration every accepted sodium head number was
    produced in (claims 0180/0181/0189, through
    ``tools/qsgw_head_spectrum.py --dft-velocity-only``).  The covariant
    upgrade is parked on claim 0183.
    """

    velocity_dft_cart: jax.Array
    nb_logical: int
    reciprocal_lattice_cart: np.ndarray
    connection_cart: None = None
    validation: None = None


def _read_small(io, name: str, shape: tuple[int, ...], dtype):
    return np.asarray(
        io.read_slab(
            name,
            shape=shape,
            dtype=dtype,
            partition_spec=P(*([None] * len(shape))),
            as_numpy=True,
        )
    )


def load_parallel_transport_head(
    path: str,
    *,
    mesh: Mesh,
    wfn,
    meta,
) -> ParallelTransportHeadData:
    """Load and validate the preprocessing artifact through SlabIO only.

    Every cheap provenance/refusal is checked before either O(nk*nb^2)
    dataset is read.  The stored connection is manifold-dependent, so a
    strict subset or superset is rejected rather than sliced.
    """
    from file_io.parallel_transport import (
        CONNECTION_CART_DATASET,
        SCHEMA_VERSION,
        VELOCITY_DFT_DATASET,
    )
    from file_io.slab_io import SlabIO
    from common.parallel_transport import band_storage_extent, wfn_fingerprint

    with SlabIO(path, mode="r", mesh=mesh) as io:

        def scalar_i32(name):
            return int(_read_small(io, name, (), np.int32).reshape(()))

        schema = scalar_i32("schema_version")
        complete = scalar_i32("connection_complete")
        val_complete = scalar_i32("velocity_validation_complete")
        val_passed = scalar_i32("velocity_validation_passed")
        band_start = scalar_i32("band_start")
        band_stop = scalar_i32("band_stop")
        effective_nspinor = scalar_i32("effective_nspinor")
        artifact_bispinor = scalar_i32("bispinor")
        kgrid = _read_small(io, "kgrid", (3,), np.int32)
        reciprocal = _read_small(io, "reciprocal_lattice_cart", (3, 3), np.float64)
        fingerprint_raw = _read_small(io, "wfn_fingerprint_utf8", (64,), np.uint8)
        try:
            fingerprint = bytes(fingerprint_raw.tolist()).decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                f"{path}: wfn_fingerprint_utf8 is not a 64-byte ASCII SHA-256 "
                "dataset; regenerate with get_dipole_mtxels"
            ) from exc

        expected_nb = int(meta.b_id_4_user)
        expected_kgrid = np.asarray(wfn.kgrid, dtype=np.int32)
        expected_reciprocal = np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat)
        refusals = []
        if schema != int(SCHEMA_VERSION):
            refusals.append(f"schema_version={schema}, expected {int(SCHEMA_VERSION)}")
        if complete != 1:
            refusals.append("connection_complete is not 1")
        if val_complete != 1 or val_passed != 1:
            refusals.append(
                "mandatory full-matrix DFT velocity validation is not complete/passing"
            )
        if band_start != 0 or band_stop != expected_nb:
            refusals.append(
                f"band manifold [{band_start},{band_stop}) != current full "
                f"head manifold [0,{expected_nb})"
            )
        if effective_nspinor != int(meta.nspinor):
            refusals.append(
                f"effective_nspinor={effective_nspinor} != current {int(meta.nspinor)}"
            )
        if bool(artifact_bispinor) != bool(int(meta.nspinor) == 4):
            refusals.append("bispinor convention differs from current run")
        if not np.array_equal(kgrid, expected_kgrid):
            refusals.append(f"kgrid={tuple(kgrid)} != current {tuple(expected_kgrid)}")
        if not np.allclose(reciprocal, expected_reciprocal, rtol=0.0, atol=1.0e-13):
            refusals.append("Cartesian reciprocal lattice differs from the current WFN")
        expected_fingerprint = wfn_fingerprint(wfn)
        if fingerprint != expected_fingerprint:
            refusals.append(
                "WFN fingerprint differs (parallel-transport data are stale or "
                "were generated from another DFT solution)"
            )
        if refusals:
            raise ValueError(
                f"{path}: refusing QSGW parallel-transport head:\n  - "
                + "\n  - ".join(refusals)
            )

        validation = {}
        for key in (
            "atol",
            "rtol",
            "max_abs",
            "max_rel",
            "max_abs_diagonal",
            "max_abs_offdiagonal",
        ):
            validation[key] = float(
                _read_small(io, f"velocity_validation_{key}", (), np.float64).reshape(
                    ()
                )
            )

        spec = P(None, None, "x", "y")
        nb_storage = band_storage_extent(mesh, expected_nb)
        large_shape = (3, int(meta.nk_tot), nb_storage, nb_storage)
        connection = io.read_slab(
            CONNECTION_CART_DATASET, shape=large_shape, partition_spec=spec
        )
        velocity = io.read_slab(
            VELOCITY_DFT_DATASET, shape=large_shape, partition_spec=spec
        )

    expected_prefix = (3, int(meta.nk_tot))
    if (
        tuple(connection.shape[:2]) != expected_prefix
        or connection.shape != velocity.shape
        or connection.shape[-2] != connection.shape[-1]
        or int(connection.shape[-1]) < expected_nb
    ):
        raise ValueError(
            f"{path}: large PT dataset shapes are inconsistent: "
            f"A={connection.shape}, v={velocity.shape}, expected prefix "
            f"{expected_prefix} and at least {expected_nb} bands."
        )
    return ParallelTransportHeadData(
        connection_cart=connection,
        velocity_dft_cart=velocity,
        nb_logical=expected_nb,
        reciprocal_lattice_cart=reciprocal,
        validation=validation,
    )


def load_dft_velocity_head(
    path: str,
    *,
    mesh: Mesh,
    wfn,
    meta,
) -> DftVelocityHeadData:
    """Load the completed exact-DFT velocity stage, and only that stage.

    This is the loader ``tools/qsgw_head_spectrum.py --dft-velocity-only``
    has always used — it lived in that tool until ``sc_head_update =
    dft_velocity`` gave the driver the same route, and it moved here rather
    than being copied so the two cannot drift.

    Two differences from :func:`load_parallel_transport_head`, both
    deliberate:

    * ``connection_complete`` / ``velocity_validation_*`` are NOT required.
      The velocity is written and checked by the dipole job on its own; the
      link, connection and velocity-identity stages exist to serve the
      covariant correction this mode does not take.
    * the handful of small provenance values are read with plain h5py.
      ``SlabIO.write_attr`` stores them as rank-0 datasets and the FFI
      ``read_slab`` refuses an empty shape (claim 0188 blocker 1,
      ``_slab_io_ffi.py:1050``), so this path must not go through
      ``_read_small``.  Repairing that read is the parallel-transport
      loader's own business (claim 0187 fixer); this mode is not blocked
      behind it and does not touch it.

    Every other provenance refusal the PT loader emits is kept verbatim:
    schema, band manifold, k grid, reciprocal lattice, WFN fingerprint.
    """
    import h5py

    from common.parallel_transport import band_storage_extent, wfn_fingerprint
    from file_io.parallel_transport import (
        SCHEMA_VERSION,
        VELOCITY_DFT_DATASET,
    )
    from file_io.slab_io import SlabIO

    nb = int(meta.b_id_4_user)
    with h5py.File(path, "r") as raw:
        schema = int(raw["schema_version"][()])
        band_start = int(raw["band_start"][()])
        band_stop = int(raw["band_stop"][()])
        kgrid = np.asarray(raw["kgrid"][()], dtype=np.int32)
        reciprocal = np.asarray(
            raw["reciprocal_lattice_cart"][()], dtype=np.float64
        )
        fingerprint_raw = np.asarray(
            raw["wfn_fingerprint_utf8"][()], dtype=np.uint8
        )
    fingerprint = bytes(fingerprint_raw.tolist()).decode("ascii")
    expected_reciprocal = (
        np.asarray(wfn.bvec, dtype=np.float64) * float(wfn.blat)
    )
    refusals = []
    if schema != int(SCHEMA_VERSION):
        refusals.append(f"schema_version={schema}, expected {SCHEMA_VERSION}")
    if (band_start, band_stop) != (0, nb):
        refusals.append(
            f"band manifold [{band_start},{band_stop}) != [0,{nb})"
        )
    if not np.array_equal(kgrid, np.asarray(wfn.kgrid, dtype=np.int32)):
        refusals.append("k grid differs from the current WFN")
    if not np.allclose(
        reciprocal, expected_reciprocal, rtol=0.0, atol=1.0e-13
    ):
        refusals.append("reciprocal lattice differs from the current WFN")
    if fingerprint != wfn_fingerprint(wfn):
        refusals.append("WFN fingerprint differs from the velocity artifact")
    if refusals:
        raise ValueError(
            f"{path}: refusing DFT velocity stage:\n  - "
            + "\n  - ".join(refusals)
        )
    nb_storage = band_storage_extent(mesh, nb)
    with SlabIO(path, mode="r", mesh=mesh) as io:
        velocity = io.read_slab(
            VELOCITY_DFT_DATASET,
            shape=(3, int(meta.nk_tot), nb_storage, nb_storage),
            partition_spec=P(None, None, "x", "y"),
        )
    return DftVelocityHeadData(
        velocity_dft_cart=velocity,
        nb_logical=nb,
        reciprocal_lattice_cart=reciprocal,
    )


def _mesh_xy(mesh: Mesh) -> tuple[str, str]:
    names = tuple(str(a) for a in mesh.axis_names)
    if names != ("x", "y"):
        raise ValueError(
            "QSGW parallel-transport head requires the production ('x','y') "
            f"band mesh, got axes {names!r}."
        )
    return names[0], names[1]


def _signed_fft_rows(kgrid: tuple[int, int, int]) -> np.ndarray:
    """Integer real-space rows in the flat-k service's C ordering."""
    axes = [np.fft.fftfreq(int(n), d=1.0 / int(n)) for n in kgrid]
    rr = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return np.asarray(rr.reshape(-1, 3), dtype=np.float64)


def _cartesian_fft_multipliers(
    kgrid: tuple[int, int, int],
    bvec_cart: np.ndarray,
) -> np.ndarray:
    """Return ``2*pi * d(kappa)/d(k_cart) * R`` as ``(3,nk)``."""
    B = np.asarray(bvec_cart, dtype=np.float64)
    if B.shape != (3, 3):
        raise ValueError(f"bvec_cart must have shape (3,3), got {B.shape}.")
    if abs(float(np.linalg.det(B))) < 1.0e-14:
        raise ValueError(
            "bvec_cart is singular; Cartesian k derivatives are undefined."
        )
    # k_cart_j = sum_i kappa_i B_ij, hence
    # d/dk_cart_j = sum_i (B^-1)_ji d/dkappa_i.
    return 2.0 * np.pi * (np.linalg.inv(B) @ _signed_fft_rows(kgrid).T)


def reduced_covector_to_cartesian(covector_reduced, bvec_cart):
    """Convert a reduced-k covector using LORRAX's row-vector B convention.

    ``k_cart = kappa @ B`` because WFN reciprocal vectors are rows.  Thus
    ``D_cart[j] = sum_i (B^-1)[j,i] D_kappa[i]``.  This is the row-basis
    spelling of the conventional ``B_column^-T`` rule.
    """
    B = np.asarray(bvec_cart, dtype=np.float64)
    if B.shape != (3, 3) or abs(float(np.linalg.det(B))) < 1.0e-14:
        raise ValueError(
            f"bvec_cart must be a nonsingular (3,3) matrix, got {B.shape}."
        )
    A = jnp.asarray(covector_reduced)
    if A.ndim < 1 or int(A.shape[0]) != 3:
        raise ValueError(
            "reduced covector must have a leading Cartesian-component "
            f"axis of extent 3, got {A.shape}."
        )
    return jnp.einsum("ij,j...->i...", np.linalg.inv(B), A, optimize=True)


def _spectral_kernel(mesh: Mesh, kgrid: tuple[int, int, int]) -> Callable:
    from ffi import ffi_dial_key

    key = ("spectral_cart", id(mesh), tuple(kgrid), ffi_dial_key())
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    _mesh_xy(mesh)
    from common.fft_helpers import make_flat_k_fftn, make_flat_k_ifftn

    spec_3d = P(None, None, None, "x", "y")
    component_spec_3d = P(None, None, None, None, "x", "y")
    fft = make_flat_k_fftn(mesh, kgrid, spec_3d, norm="ortho")
    # Batch x/y/z through one inverse-FFT service call.  Besides avoiding
    # three dispatches, this keeps the shared real-space operator resident
    # exactly once in the compiled graph.
    ifft_components = make_flat_k_ifftn(mesh, kgrid, component_spec_3d, norm="ortho")
    out_sharding = NamedSharding(mesh, P(None, None, "x", "y"))

    @jax.jit
    def _kernel(operator_k, multipliers_cart_k):
        operator_R = fft(operator_k)
        weighted_R = operator_R[:, None, :, :] * (
            1j * multipliers_cart_k.T[:, :, None, None]
        )
        deriv = jnp.moveaxis(ifft_components(weighted_R), 1, 0)
        return jax.lax.with_sharding_constraint(deriv, out_sharding)

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def spectral_cartesian_derivative(
    operator_k,
    *,
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    bvec_cart,
):
    """Spectrally differentiate a full-BZ band operator.

    ``operator_k`` is ``(nk,nb,nb)`` at ``P(None,'x','y')``.  The full
    k grid remains local to each band tile, so the FFT communicates no band
    data.  One cached compiled graph contains the forward FFT and one inverse
    FFT batched across the three Cartesian components.
    """
    kgrid = tuple(int(n) for n in kgrid)
    nk = int(np.prod(kgrid))
    if tuple(operator_k.shape[:1]) != (nk,) or operator_k.ndim != 3:
        raise ValueError(
            f"operator_k must have shape ({nk},nb,nb), got {tuple(operator_k.shape)}."
        )
    if operator_k.shape[1] != operator_k.shape[2]:
        raise ValueError("spectral derivative requires square band matrices.")
    mult = jnp.asarray(
        _cartesian_fft_multipliers(kgrid, np.asarray(bvec_cart)), dtype=jnp.float64
    )
    return _spectral_kernel(mesh, kgrid)(
        jnp.asarray(operator_k, dtype=jnp.complex128), mult
    )


def _commutator_kernel(mesh: Mesh) -> Callable:
    key = ("band_commutator", id(mesh))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    _mesh_xy(mesh)

    def _local(A_row, H_col, H_row, A_col):
        AH = jnp.einsum("akim,kmj->akij", A_row, H_col, optimize=True)
        HA = jnp.einsum("kim,akmj->akij", H_row, A_col, optimize=True)
        return AH - HA

    # Gather only one band dimension of each operand.  No rank ever owns a
    # full (nb,nb) matrix; the output returns to the native 2-D band tile.
    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, None, "x", None),
            P(None, None, "y"),
            P(None, "x", None),
            P(None, None, None, "y"),
        ),
        out_specs=P(None, None, "x", "y"),
        check_vma=False,
    )

    @jax.jit
    def _kernel(A, H):
        return sm(A, H, H, A)

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def covariant_cartesian_derivative(
    operator_k,
    connection_cart,
    *,
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    bvec_cart,
):
    """Return ``partial_i operator - i[A_i,operator]`` for i=x,y,z."""
    H = jnp.asarray(operator_k, dtype=jnp.complex128)
    A = jnp.asarray(connection_cart, dtype=jnp.complex128)
    expected = (3,) + tuple(H.shape)
    if tuple(A.shape) != expected:
        raise ValueError(
            f"connection_cart must have shape {expected}, got {tuple(A.shape)}."
        )
    partial = spectral_cartesian_derivative(
        H, mesh=mesh, kgrid=kgrid, bvec_cart=bvec_cart
    )
    return partial - 1j * _commutator_kernel(mesh)(A, H)


def _structured_delta_kernel(mesh: Mesh, nb_active: int) -> Callable:
    """Covariant derivative for active-block plus diagonal-tail Delta H."""
    key = ("structured_delta", id(mesh), int(nb_active))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    from common.parallel_transport import make_distributed_band_matmul

    multiply = make_distributed_band_matmul(mesh, n_batch_axes=2)
    na = int(nb_active)

    @jax.jit
    def _kernel(delta, A, partial):
        diag = jnp.diagonal(delta, axis1=-2, axis2=-1)
        active = delta[:, :na, :na]
        active_offdiag = active - (
            jnp.eye(na, dtype=active.dtype)[None]
            * jnp.diagonal(active, axis1=-2, axis2=-1)[:, None, :]
        )
        K = jnp.broadcast_to(active_offdiag[None], (3,) + active_offdiag.shape)
        AK = multiply(A[:, :, :, :na], K)
        KA = multiply(K, A[:, :, :na, :])
        comm = A * (diag[None, :, None, :] - diag[None, :, :, None])
        comm = comm.at[:, :, :, :na].add(AK)
        comm = comm.at[:, :, :na, :].add(-KA)
        return partial - 1j * comm

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def covariant_structured_delta(
    delta_h_dft,
    connection_cart,
    *,
    U_active,
    mesh: Mesh,
    kgrid,
    bvec_cart,
):
    """Efficient D DeltaH for a dense active block and diagonal tail.

    The only band GEMMs have an active contracted dimension.  The diagonal
    tail commutator is elementwise, avoiding O(nb_head^3) work.
    """
    delta = jnp.asarray(delta_h_dft, dtype=jnp.complex128)
    A = jnp.asarray(connection_cart, dtype=jnp.complex128)
    na = int(U_active.shape[-1])
    partial = spectral_cartesian_derivative(
        delta, mesh=mesh, kgrid=kgrid, bvec_cart=bvec_cart
    )
    return _structured_delta_kernel(mesh, na)(delta, A, partial)


def _active_rotation_kernel(mesh: Mesh, nb_active: int) -> Callable:
    key = ("active_velocity_rotation", id(mesh), int(nb_active))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    from common.parallel_transport import make_distributed_band_matmul

    multiply = make_distributed_band_matmul(mesh, n_batch_axes=2)
    na = int(nb_active)

    @jax.jit
    def _kernel(v, U):
        change = U - jnp.eye(na, dtype=U.dtype)[None]
        change = jnp.broadcast_to(change[None], (3,) + change.shape)
        right = multiply(v[:, :, :, :na], change)
        tmp = v.at[:, :, :, :na].add(right)
        change_h = jnp.swapaxes(jnp.conj(change), -1, -2)
        left = multiply(change_h, tmp[:, :, :na, :])
        return tmp.at[:, :, :na, :].add(left)

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def _rotation_kernel(mesh: Mesh) -> Callable:
    key = ("velocity_rotation", id(mesh))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    _mesh_xy(mesh)

    def _right_local(v_row, U_col):
        return jnp.einsum("akim,kmn->akin", v_row, U_col, optimize=True)

    right = shard_map(
        _right_local,
        mesh=mesh,
        in_specs=(P(None, None, "x", None), P(None, None, "y")),
        out_specs=P(None, None, "x", "y"),
        check_vma=False,
    )

    def _left_local(U_free, tmp_col):
        return jnp.einsum("kmp,akmn->akpn", jnp.conj(U_free), tmp_col, optimize=True)

    left = shard_map(
        _left_local,
        mesh=mesh,
        in_specs=(P(None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, None, "x", "y"),
        check_vma=False,
    )

    @jax.jit
    def _kernel(velocity_cart, U):
        # Each contraction gathers one band axis only.  The intermediate
        # and result remain P(component,k,x,y), so no full nb^2 matrix is
        # resident on a rank.
        return left(U, right(velocity_cart, U))

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def rotate_velocity_to_qp(velocity_cart, U_dft_to_qp, *, mesh: Mesh):
    """Return ``U^dagger v_i U`` for all Cartesian components in one jit."""
    return _rotation_kernel(mesh)(velocity_cart, U_dft_to_qp)


def rotate_velocity_active_to_qp(velocity_cart, U_active, *, mesh: Mesh):
    """Apply blockdiag(U_active,I)^H v blockdiag(U_active,I).

    Work scales as O(nb_head * nb_active^2), and no dense full-manifold
    unitary is constructed.
    """
    na = int(U_active.shape[-1])
    if U_active.shape[-2] != na:
        raise ValueError("U_active must be square on its band axes")
    return _active_rotation_kernel(mesh, na)(velocity_cart, U_active)


def _assemble_kernel(mesh: Mesh, nb_storage: int) -> Callable:
    key = ("assemble_head_manifold", id(mesh), int(nb_storage))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    _mesh_xy(mesh)
    out_sharding = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit
    def _kernel(delta_active, U_active):
        nk, nb_active, _ = delta_active.shape
        delta = jnp.zeros((nk, nb_storage, nb_storage), dtype=jnp.complex128)
        delta = delta.at[:, :nb_active, :nb_active].set(delta_active)
        U = jnp.broadcast_to(
            jnp.eye(nb_storage, dtype=jnp.complex128)[None, :, :],
            (nk, nb_storage, nb_storage),
        )
        U = U.at[:, :nb_active, :nb_active].set(U_active)
        return (
            jax.lax.with_sharding_constraint(delta, out_sharding),
            jax.lax.with_sharding_constraint(U, out_sharding),
        )

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def assemble_head_manifold(
    delta_h_active,
    U_active,
    *,
    nb_storage: int,
    mesh: Mesh,
):
    """Embed the active QSGW block in the full velocity/head manifold.

    The inactive correction is zero and its basis rotation is identity.
    Keeping the full matrix is load-bearing: A-active/inactive commutators
    and high-conduction transitions would both be lost by slicing A down to
    the active Sigma window.
    """
    if delta_h_active.shape != U_active.shape or delta_h_active.ndim != 3:
        raise ValueError(
            "active delta-H and U must be equal-shaped (nk,nb,nb) arrays; "
            f"got {delta_h_active.shape}/{U_active.shape}."
        )
    if delta_h_active.shape[1] != delta_h_active.shape[2]:
        raise ValueError("active delta-H/U matrices must be square.")
    if int(delta_h_active.shape[1]) > int(nb_storage):
        raise ValueError(
            f"active nb={delta_h_active.shape[1]} exceeds head storage nb={nb_storage}."
        )
    return _assemble_kernel(mesh, int(nb_storage))(delta_h_active, U_active)


def _assemble_delta_kernel(mesh: Mesh, nb_storage: int) -> Callable:
    key = ("assemble_delta_head", id(mesh), int(nb_storage))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    out_sharding = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit
    def _kernel(delta_active, tail_diagonal):
        nk, na, _ = delta_active.shape
        delta = jnp.zeros((nk, nb_storage, nb_storage), dtype=jnp.complex128)
        delta = delta.at[:, :na, :na].set(delta_active)
        idx = jnp.arange(na, nb_storage)
        delta = delta.at[:, idx, idx].set(tail_diagonal[:, na:nb_storage])
        return jax.lax.with_sharding_constraint(delta, out_sharding)

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def assemble_delta_head_manifold(
    delta_h_active,
    tail_diagonal,
    *,
    nb_storage: int,
    mesh: Mesh,
):
    """Embed active DeltaH and the current diagonal sum-band tail."""
    delta = jnp.asarray(delta_h_active)
    tail = jnp.asarray(tail_diagonal)
    if delta.ndim != 3 or delta.shape[-2] != delta.shape[-1]:
        raise ValueError("delta_h_active must be (nk,na,na)")
    if tail.ndim != 2 or tail.shape[0] != delta.shape[0]:
        raise ValueError("tail_diagonal must be (nk,nb_storage)")
    if int(tail.shape[1]) < int(nb_storage):
        raise ValueError(f"tail diagonal extent {tail.shape[1]} < storage {nb_storage}")
    if int(delta.shape[-1]) > int(nb_storage):
        raise ValueError("active DeltaH exceeds the head manifold")
    return _assemble_delta_kernel(mesh, int(nb_storage))(delta, tail)


def _s_tensor_kernel(mesh: Mesh, *, nb_logical: int) -> Callable:
    key = ("head_s", id(mesh), int(nb_logical))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)

    def _local(v_local, e_bra, e_ket, f_bra, f_ket, omegas, prefactor, eta):
        nx, ny = v_local.shape[-2:]
        ix = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        iy = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)
        dE = e_bra[:, :, None] - e_ket[:, None, :]
        f_diff = f_ket[:, None, :] - f_bra[:, :, None]
        logical = ((ix[:, None] < nb_logical) & (iy[None, :] < nb_logical))[None, :, :]
        # Sum every energy-ordered band pair.  f_diff is SIGNED: MP1 is
        # not globally monotone and may overshoot slightly outside [0, 1],
        # so filtering on f_v-f_c>0 would not implement the Adler-Wiser
        # occupation difference.  The historical 0/1 path is unchanged
        # because its energy-ordered nonzero differences are positive.
        transition = logical & (dE > 0.0)

        def _one(omega):
            z = omega + 1j * eta
            denom = dE * (z * z - dE * dE)
            weight = jnp.where(
                transition & (jnp.abs(denom) > 1.0e-16),
                prefactor * f_diff / denom,
                jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
            )
            local = jnp.einsum(
                "akij,kij,bkij->ab", jnp.conj(v_local), weight, v_local, optimize=True
            )
            return jax.lax.psum(local, (ax_x, ax_y))

        return jax.vmap(_one)(omegas)

    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, None, "x", "y"),
            P(None, "x"),
            P(None, "y"),
            P(None, "x"),
            P(None, "y"),
            P(None),
            P(),
            P(),
        ),
        out_specs=P(None, None, None),
        check_vma=False,
    )
    kernel = jax.jit(sm)
    _KERNEL_CACHE[key] = kernel
    return kernel


def _head_wing_kernel(
    mesh: Mesh,
    *,
    nb_logical: int,
    include_surface: bool,
) -> Callable:
    r"""Return the cached all-band q-linear wing contraction.

    The velocity is already tiled on both band axes.  A naive wing einsum
    would have to gather one complete band axis because the output centroid
    axis and one velocity-band axis share the same mesh axis.  Instead each
    rank circulates its small velocity tile around that mesh axis.  After one
    ring every local centroid slice has seen every band tile, while no rank
    ever materialises a full ``nb x nb`` velocity matrix.
    """
    key = ("head_wings", id(mesh), int(nb_logical), bool(include_surface))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)
    px = int(mesh.shape[ax_x])
    py = int(mesh.shape[ax_y])
    perm_x = tuple((i, (i + 1) % px) for i in range(px))
    perm_y = tuple((i, (i + 1) % py) for i in range(py))

    def _frequency_layout(n_omega):
        block = min(_HEAD_WING_FREQUENCY_BLOCK, int(n_omega))
        padded = ((int(n_omega) + block - 1) // block) * block
        return block, padded

    def _local(
        v_local,
        psi_xn_local,
        psi_yn_local,
        energies,
        occupations,
        surface_weight,
        omegas,
        pref_inter,
        pref_surface,
        eta,
    ):
        nk = v_local.shape[1]
        nx, ny = v_local.shape[-2:]
        ns = psi_xn_local.shape[1]
        nmu_x = psi_xn_local.shape[2]
        nmu_y = psi_yn_local.shape[2]
        x_coord = jax.lax.axis_index(ax_x)
        y_coord = jax.lax.axis_index(ax_y)
        zero = jnp.asarray(0, dtype=x_coord.dtype)
        x_start = x_coord * nx
        y_start = y_coord * ny
        z = omegas + 1j * eta
        inv_z = jnp.where(
            jnp.abs(omegas) > 1.0e-15,
            1.0 / z,
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
        )
        frequency_block, n_omega_padded = _frequency_layout(
            omegas.shape[0])
        frequency_pad = n_omega_padded - int(omegas.shape[0])
        z_blocks = jnp.pad(
            z, (0, frequency_pad),
            constant_values=jnp.asarray(1.0j, dtype=jnp.complex128),
        ).reshape(-1, frequency_block)
        inv_z_blocks = jnp.pad(
            inv_z, (0, frequency_pad)).reshape(-1, frequency_block)
        block_indices = jnp.arange(z_blocks.shape[0], dtype=jnp.int32)

        def _accumulate_frequency_blocks(
            accumulator,
            dE,
            f_diff,
            transition,
            surface_pair,
            contract,
        ):
            def _block(block_acc, node):
                block_index, z_block, inv_z_block = node
                denom = (
                    z_block[:, None, None, None] ** 2
                    - dE[None, :, :, :] ** 2)
                weight = jnp.where(
                    transition[None, :, :, :]
                    & (jnp.abs(denom) > 1.0e-16),
                    pref_inter * f_diff[None, :, :, :] / denom,
                    jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
                )
                if include_surface:
                    weight = weight + (
                        pref_surface
                        * inv_z_block[:, None, None, None]
                        * surface_pair[None, :, :, :])
                contribution = contract(weight)
                start = block_index * frequency_block
                starts = (start,) + (zero,) * (block_acc.ndim - 1)
                sizes = (frequency_block,) + block_acc.shape[1:]
                old = jax.lax.dynamic_slice(block_acc, starts, sizes)
                return jax.lax.dynamic_update_slice(
                    block_acc, old + contribution, starts), None

            return jax.lax.scan(
                _block,
                accumulator,
                (block_indices, z_blocks, inv_z_blocks),
                unroll=1,
            )[0]

        e_low_y = jax.lax.dynamic_slice(
            energies, (zero, y_start), (nk, ny))
        f_low_y = jax.lax.dynamic_slice(
            occupations, (zero, y_start), (nk, ny))
        psi_low_x = jax.lax.dynamic_slice(
            psi_xn_local, (zero, zero, zero, y_start), (nk, ns, nmu_x, ny))

        y0 = jnp.zeros(
            (n_omega_padded, 3, nmu_x), dtype=jnp.complex128)

        def _left_step(step, carry):
            v_tile, acc = carry
            source_x = jnp.mod(x_coord - step, px)
            high_start = source_x * nx
            e_high = jax.lax.dynamic_slice(
                energies, (zero, high_start), (nk, nx))
            f_high = jax.lax.dynamic_slice(
                occupations, (zero, high_start), (nk, nx))
            psi_high = jax.lax.dynamic_slice(
                psi_xn_local, (zero, zero, zero, high_start),
                (nk, ns, nmu_x, nx))
            dE = e_high[:, :, None] - e_low_y[:, None, :]
            f_diff = f_low_y[:, None, :] - f_high[:, :, None]
            global_high = high_start + jnp.arange(nx)
            global_low = y_start + jnp.arange(ny)
            logical = (
                (global_high[:, None] < nb_logical)
                & (global_low[None, :] < nb_logical)
            )[None, :, :]
            transition = logical & (dE > 0.0)
            surface_pair = jnp.zeros_like(dE)
            if include_surface:
                diagonal = logical & (
                    global_high[:, None] == global_low[None, :]
                )[None, :, :]
                surface_high = jax.lax.dynamic_slice(
                    surface_weight, (zero, high_start), (nk, nx))
                surface_pair = jnp.where(
                    diagonal, surface_high[:, :, None], 0.0)

            def _contract_left(weight):
                # b_ij(mu) remains fused; no nk*nb^2*nmu tensor is stored.
                return jnp.einsum(
                    "akij,wkij,ksmi,ksmj->wam",
                    jnp.conj(v_tile), weight,
                    jnp.conj(psi_high), psi_low_x,
                    optimize=True,
                )

            acc = _accumulate_frequency_blocks(
                acc, dE, f_diff, transition, surface_pair, _contract_left)
            v_next = (
                jax.lax.ppermute(v_tile, ax_x, perm_x)
                if px > 1 else v_tile)
            return v_next, acc

        (unused_v, Y_x), _ = jax.lax.scan(
            lambda carry, step: (_left_step(step, carry), None),
            (v_local, y0), jnp.arange(px, dtype=x_coord.dtype), unroll=1)
        del unused_v
        Y_x = Y_x[:omegas.shape[0]]
        Y_x = jax.lax.psum(Y_x, ax_y)

        e_high_x = jax.lax.dynamic_slice(
            energies, (zero, x_start), (nk, nx))
        f_high_x = jax.lax.dynamic_slice(
            occupations, (zero, x_start), (nk, nx))
        psi_high_y = jax.lax.dynamic_slice(
            psi_yn_local, (zero, zero, zero, x_start), (nk, ns, nmu_y, nx))
        z0 = jnp.zeros(
            (n_omega_padded, nmu_y, 3), dtype=jnp.complex128)

        def _right_step(step, carry):
            v_tile, acc = carry
            source_y = jnp.mod(y_coord - step, py)
            low_start = source_y * ny
            e_low = jax.lax.dynamic_slice(
                energies, (zero, low_start), (nk, ny))
            f_low = jax.lax.dynamic_slice(
                occupations, (zero, low_start), (nk, ny))
            psi_low = jax.lax.dynamic_slice(
                psi_yn_local, (zero, zero, zero, low_start),
                (nk, ns, nmu_y, ny))
            dE = e_high_x[:, :, None] - e_low[:, None, :]
            f_diff = f_low[:, None, :] - f_high_x[:, :, None]
            global_high = x_start + jnp.arange(nx)
            global_low = low_start + jnp.arange(ny)
            logical = (
                (global_high[:, None] < nb_logical)
                & (global_low[None, :] < nb_logical)
            )[None, :, :]
            transition = logical & (dE > 0.0)
            surface_pair = jnp.zeros_like(dE)
            if include_surface:
                diagonal = logical & (
                    global_high[:, None] == global_low[None, :]
                )[None, :, :]
                surface_low = jax.lax.dynamic_slice(
                    surface_weight, (zero, low_start), (nk, ny))
                surface_pair = jnp.where(
                    diagonal, surface_low[:, None, :], 0.0)

            def _contract_right(weight):
                return jnp.einsum(
                    "ksmi,ksmj,wkij,bkij->wmb",
                    psi_high_y, jnp.conj(psi_low), weight, v_tile,
                    optimize=True,
                )

            acc = _accumulate_frequency_blocks(
                acc, dE, f_diff, transition, surface_pair, _contract_right)
            v_next = (
                jax.lax.ppermute(v_tile, ax_y, perm_y)
                if py > 1 else v_tile)
            return v_next, acc

        (unused_v, Z_y), _ = jax.lax.scan(
            lambda carry, step: (_right_step(step, carry), None),
            (v_local, z0), jnp.arange(py, dtype=y_coord.dtype), unroll=1)
        del unused_v
        Z_y = Z_y[:omegas.shape[0]]
        Z_y = jax.lax.psum(Z_y, ax_x)
        return Y_x, Z_y

    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, None, "x", "y"),
            P(None, None, "x", None),
            P(None, None, "y", None),
            P(None, None),
            P(None, None),
            P(None, None),
            P(None),
            P(),
            P(),
            P(),
        ),
        out_specs=(P(None, None, "x"), P(None, "y", None)),
        check_vma=False,
    )
    kernel = jax.jit(sm)
    _KERNEL_CACHE[key] = kernel
    return kernel


def head_wings_sharded(
    velocity_cart,
    wfns,
    energies_kn_ry,
    occupations_kn,
    omegas_ry,
    *,
    mesh: Mesh,
    nb_logical: int,
    nk_tot: int,
    nspin: int,
    nspinor: int,
    eta_ry: float = 0.0,
    surface_weight_kn=None,
):
    r"""Build q-linear head/body wings in the current band basis.

    For every energy-ordered interband pair, with
    ``b_ij(mu)=sum_s psi_i(mu)^* psi_j(mu)``, this evaluates

    ``Y[a,mu] = sum conj(v[a,ij]) F_ij b_ij(mu)`` and
    ``Z[mu,b] = sum conj(b_ij(mu)) F_ij v[b,ij]``,

    where ``F_ij = 4(f_j-f_i)/(Nk*nspin*nspinor*(z^2-dE^2))``.  This is
    exactly the normalization paired with ``head_s_tensor_sharded``:
    ``S_direct`` owns ``1/cell_volume`` and the later Schur fold introduces
    the sole additional ``1/cell_volume`` multiplying ``Y W Z``.

    With tetrahedron surface weights, the finite-frequency intraband wings
    ``sum delta(E-mu) v_a b_nn / z`` are included as well.  The strictly
    static metal limit is not obtained by setting ``z=0`` in that expression;
    its head remains on the separate Thomas-Fermi path.

    Every rank owns the same ``(nb/Px) * (nb/Py)`` band-pair tile.  The
    x-sharded and y-sharded centroid wavefunction copies build Y and Z,
    respectively, while band tiles circulate around only the matching mesh
    axis.  Frequencies are blocked inside each ring step, so a tile is sent
    once rather than once per frequency block and no all-frequency pair
    tensor is formed.
    """
    v = jnp.asarray(velocity_cart, dtype=jnp.complex128)
    e = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    f = jnp.asarray(occupations_kn, dtype=jnp.float64)
    omega = jnp.atleast_1d(jnp.asarray(omegas_ry, dtype=jnp.complex128))
    if v.ndim != 4 or v.shape[0] != 3 or v.shape[2] != v.shape[3]:
        raise ValueError(
            f"velocity_cart must be (3,nk,nb,nb), got {v.shape}.")
    if e.shape != f.shape or tuple(e.shape) != tuple(v.shape[1:3]):
        raise ValueError(
            f"energy/occupation shapes {e.shape}/{f.shape} do not match "
            f"velocity (nk,nb)={v.shape[1:3]}.")
    if (
        int(wfns.psi_xn.shape[0]) != int(v.shape[1])
        or int(wfns.psi_yn.shape[0]) != int(v.shape[1])
        or int(wfns.psi_xn.shape[1]) != int(wfns.psi_yn.shape[1])
    ):
        raise ValueError(
            "centroid wavefunction k/spinor axes do not match the velocity")
    if int(wfns.psi_xn.shape[-1]) < int(v.shape[-1]):
        raise ValueError("centroid wavefunctions do not cover the head manifold")
    psi_xn = wfns.psi_xn[..., : int(v.shape[-1])]
    psi_yn = wfns.psi_yn[..., : int(v.shape[-1])]
    include_surface = surface_weight_kn is not None
    surface = (
        jnp.asarray(surface_weight_kn, dtype=jnp.float64)
        if include_surface else jnp.zeros_like(e))
    if surface.shape != e.shape:
        raise ValueError(
            f"surface_weight_kn shape {surface.shape} does not match {e.shape}.")
    spin_denominator = (
        float(max(int(nspin), 1)) * float(max(int(nspinor), 1)))
    pref_inter = 4.0 / (float(nk_tot) * spin_denominator)
    pref_surface = 2.0 / (float(nk_tot) * spin_denominator)
    return _head_wing_kernel(
        mesh, nb_logical=int(nb_logical),
        include_surface=bool(include_surface))(
            v, psi_xn, psi_yn, e, f, surface, omega,
            jnp.asarray(pref_inter, dtype=jnp.complex128),
            jnp.asarray(pref_surface, dtype=jnp.complex128),
            jnp.asarray(float(eta_ry), dtype=jnp.float64),
        )


def static_head_wings_sharded(
    wfns,
    surface_weight_kn,
    *,
    mesh: Mesh,
    nb_logical: int,
    nk_tot: int,
    nspin: int,
    nspinor: int,
):
    r"""Build the strictly-static intraband centroid wings.

    In the static order of limits the diagonal density vertex survives:

    ``C_mu = (2/(Nk*nspin*nspinor)) sum_kn f'(E_kn)|psi_kn(mu)|^2``.

    ``surface_weight_kn`` is ``-f'`` in the caller's integration scheme, so
    the explicit minus sign below is physical.  The x/y centroid copies stay
    sharded on their respective mesh axes and the spinor axis is summed
    without any component-count special case.
    """
    surface = jnp.asarray(surface_weight_kn, dtype=jnp.float64)
    if surface.ndim != 2:
        raise ValueError(
            f"static head surface weights must be (nk,nb), got {surface.shape}")
    if not (0 < int(nb_logical) <= int(surface.shape[1])):
        raise ValueError(
            f"need 0 < nb_logical <= {surface.shape[1]}, got {nb_logical}")
    if (
        int(wfns.psi_xn.shape[0]) != int(surface.shape[0])
        or int(wfns.psi_yn.shape[0]) != int(surface.shape[0])
        or int(wfns.psi_xn.shape[-1]) < int(surface.shape[1])
        or int(wfns.psi_yn.shape[-1]) < int(surface.shape[1])
    ):
        raise ValueError("centroid wavefunctions do not cover static weights")
    logical = jnp.arange(surface.shape[1])[None, :] < int(nb_logical)
    weight = jnp.where(logical, surface, 0.0)
    psi_x = wfns.psi_xn[..., : int(surface.shape[1])]
    psi_y = wfns.psi_yn[..., : int(surface.shape[1])]
    density_x = jnp.sum(jnp.square(jnp.abs(psi_x)), axis=1)
    density_y = jnp.sum(jnp.square(jnp.abs(psi_y)), axis=1)
    prefactor = -2.0 / (
        float(nk_tot)
        * float(max(int(nspin), 1))
        * float(max(int(nspinor), 1))
    )
    with mesh:
        left = jax.lax.with_sharding_constraint(
            prefactor * jnp.einsum("kn,kmn->m", weight, density_x),
            NamedSharding(mesh, P("x")),
        )
        right = jax.lax.with_sharding_constraint(
            prefactor * jnp.einsum("kn,kmn->m", weight, density_y),
            NamedSharding(mesh, P("y")),
        )
    return left, right


def _drude_tensor_kernel(mesh: Mesh, *, nb_logical: int) -> Callable:
    """Compile the Fermi-surface velocity contraction once per band shape."""
    key = ("head_drude", id(mesh), int(nb_logical))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)

    def _local(v_local, surface_weight_x, prefactor):
        nx, ny = v_local.shape[-2:]
        ix = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        iy = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)
        diagonal = (
            (ix[:, None] == iy[None, :])
            & (ix[:, None] < nb_logical)
            & (iy[None, :] < nb_logical)
        )[None, :, :]
        weight = jnp.where(diagonal, surface_weight_x[:, :, None], 0.0)
        local = prefactor * jnp.einsum(
            "akij,kij,bkij->ab",
            jnp.conj(v_local), weight, v_local, optimize=True,
        )
        return jax.lax.psum(local, (ax_x, ax_y))

    sm = shard_map(
        _local,
        mesh=mesh,
        in_specs=(P(None, None, "x", "y"), P(None, "x"), P()),
        out_specs=P(None, None),
        check_vma=False,
    )
    kernel = jax.jit(sm)
    _KERNEL_CACHE[key] = kernel
    return kernel


def head_drude_tensor_sharded(
    velocity_cart,
    surface_weight_kn,
    *,
    mesh: Mesh,
    nb_logical: int,
    cell_volume: float,
    nk_tot: int,
    nspin: int,
    nspinor: int,
):
    """Return the ab-initio Drude tensor ``D_ab`` in Rydberg units.

    ``D_ab = C/(Omega Nk) sum_kn (-df/dE) v_a,nn* v_b,nn`` with state
    capacity ``C=2/(nspin*nspinor)``.  Consequently the directional plasma
    frequency in this tree's Rydberg convention is
    ``omega_p(qhat)^2 = 8*pi*qhat.D.qhat``.  The diagonal QSGW velocities
    include the nonlocal-pseudopotential and covariant-rotation terms.  At
    the initial iteration they are exactly the saved DFT ``dH/dk``; only a
    subsequent self-consistent Hamiltonian update makes them QSGW.  No fitted
    or experimental plasma frequency enters.
    """
    v = jnp.asarray(velocity_cart, dtype=jnp.complex128)
    surface = jnp.asarray(surface_weight_kn, dtype=jnp.float64)
    if v.ndim != 4 or v.shape[0] != 3 or v.shape[2] != v.shape[3]:
        raise ValueError(
            f"velocity_cart must be (3,nk,nb,nb), got {v.shape}.")
    if tuple(surface.shape) != tuple(v.shape[1:3]):
        raise ValueError(
            f"surface_weight_kn shape {surface.shape} does not match "
            f"velocity (nk,nb)={v.shape[1:3]}.")
    if not (0 < int(nb_logical) <= int(v.shape[2])):
        raise ValueError(
            f"need 0 < nb_logical <= stored nb, got "
            f"{nb_logical}, {v.shape[2]}.")
    pref = 2.0 / (
        float(cell_volume)
        * float(nk_tot)
        * float(max(int(nspin), 1))
        * float(max(int(nspinor), 1))
    )
    tensor = _drude_tensor_kernel(mesh, nb_logical=int(nb_logical))(
        v, surface, jnp.asarray(pref, dtype=jnp.complex128))
    return 0.5 * (tensor + jnp.conj(tensor.T))


def head_s_tensor_sharded(
    velocity_cart,
    energies_kn_ry,
    occupations_kn,
    omegas_ry,
    *,
    mesh: Mesh,
    nb_logical: int,
    cell_volume: float,
    nk_tot: int,
    nspin: int,
    nspinor: int,
    eta_ry: float = 0.0,
    surface_weight_kn=None,
):
    """Build interband plus optional Drude ``S(omega)`` from current velocity.

    The initial call uses the saved DFT operator.  Later self-consistent calls
    use its covariantly updated and rotated counterpart.

    The contraction runs over every pair in ``[0, nb_logical)`` and uses
    the signed factor ``f_nk - f_mk``.  There is deliberately no integer
    occupied-band boundary in this API.  If ``surface_weight_kn`` is supplied,
    the diagonal-velocity Fermi-surface tensor is added as
    ``D/(omega+i*eta)^2``.  This is the dynamic q->0 intraband limit; the
    strictly static metallic limit has a different order of limits.

    Energies and occupations are passed twice with complementary one-axis
    shardings.  Each rank forms only its local conduction-by-valence tile;
    a two-axis psum reduces the final 3x3 tensor.
    """
    v = jnp.asarray(velocity_cart, dtype=jnp.complex128)
    e = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    f = jnp.asarray(occupations_kn, dtype=jnp.float64)
    omega = jnp.atleast_1d(jnp.asarray(omegas_ry, dtype=jnp.complex128))
    if v.ndim != 4 or v.shape[0] != 3:
        raise ValueError(f"velocity_cart must be (3,nk,nb,nb), got {v.shape}.")
    if e.shape != f.shape or tuple(e.shape) != tuple(v.shape[1:3]):
        raise ValueError(
            f"energy/occupation shapes {e.shape}/{f.shape} do not match "
            f"velocity (nk,nb)={v.shape[1:3]}."
        )
    if v.shape[2] != v.shape[3]:
        raise ValueError("velocity band matrices must be square.")
    if not (0 < int(nb_logical) <= int(v.shape[2])):
        raise ValueError(
            f"need 0 < nb_logical <= stored nb, got "
            f"{nb_logical}, {v.shape[2]}."
        )
    pref = 4.0 / (
        float(cell_volume)
        * float(nk_tot)
        * float(max(int(nspin), 1))
        * float(max(int(nspinor), 1))
    )
    interband = _s_tensor_kernel(mesh, nb_logical=int(nb_logical))(
        v,
        e,
        e,
        f,
        f,
        omega,
        jnp.asarray(pref, dtype=jnp.complex128),
        jnp.asarray(float(eta_ry), dtype=jnp.float64),
    )
    if surface_weight_kn is None:
        return interband
    drude = head_drude_tensor_sharded(
        v,
        surface_weight_kn,
        mesh=mesh,
        nb_logical=int(nb_logical),
        cell_volume=float(cell_volume),
        nk_tot=int(nk_tot),
        nspin=int(nspin),
        nspinor=int(nspinor),
    )
    z = omega + 1j * jnp.asarray(float(eta_ry), dtype=jnp.float64)
    # The exact static metallic limit is Thomas-Fermi, not the omega->0
    # value of the dynamic Drude expression.  Leave an exact zero-frequency
    # slot untouched here; ``head_samples_from_s`` replaces that slot with
    # the separately averaged TF model when surface weights are present.
    inv_z2 = jnp.where(
        jnp.abs(z) > 1.0e-15,
        1.0 / jnp.square(z),
        jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
    )
    return interband + drude[None, :, :] * inv_z2[:, None, None]


@dataclass(frozen=True)
class IterationHeadResponse:
    """Direct head and centroid-sharded wings before the body Schur fold."""

    omegas: tuple[complex, ...]
    S_direct: jax.Array
    Y_x: jax.Array | None
    Z_y: jax.Array | None
    static_kappa2_bohr2: float | None
    static_Y_x: jax.Array | None
    static_Z_y: jax.Array | None
    static_chi_body_gamma: jax.Array | None
    sigma_energies_ry: np.ndarray
    sigma_occupations: np.ndarray
    efermi_ry: float


@dataclass(frozen=True)
class IterationHeadSamples:
    """Per-iteration q=0 samples plus the matching active QP spectrum."""

    omegas: tuple[complex, ...]
    samples: tuple[object, ...]
    sigma_energies_ry: np.ndarray
    sigma_occupations: np.ndarray
    efermi_ry: float

    def at(self, omega):
        z = complex(omega)
        for known, sample in zip(self.omegas, self.samples):
            if abs(z - known) <= 1.0e-12:
                return sample
        raise KeyError(
            f"QSGW iteration head has no sample at omega={z} Ry; "
            f"available={self.omegas}."
        )


def _fold_static_kappa2(response, W_body_gamma, cell_volume, mesh):
    """Return kappa_eff^2 after the scalar static wing/body/wing fold."""
    if response.static_kappa2_bohr2 is None:
        return None
    if W_body_gamma is None:
        return response.static_kappa2_bohr2
    if response.static_Y_x is None or response.static_Z_y is None:
        raise ValueError(
            "static body-screened head requested without static density wings")
    from gw.head_correction import fold_cartesian_head_wings_sharded
    direct = jnp.asarray(
        [[-float(response.static_kappa2_bohr2) / (8.0 * np.pi)]],
        dtype=jnp.complex128,
    )
    effective = fold_cartesian_head_wings_sharded(
        direct,
        response.static_Y_x[None, :],
        W_body_gamma,
        response.static_Z_y[:, None],
        float(cell_volume),
        mesh_xy=mesh,
    )[0, 0]
    value = complex(np.asarray(effective))
    scale = max(abs(value.real), 1.0)
    if abs(value.imag) > 1.0e-8 * scale:
        raise ValueError(
            "static Schur effective head is not real: "
            f"f00_eff={value!r}")
    kappa2 = -8.0 * np.pi * value.real
    if not np.isfinite(kappa2) or kappa2 <= 0.0:
        raise ValueError(
            "static Schur fold produced nonphysical screening: "
            f"kappa_eff^2={kappa2!r}")
    return float(kappa2)


def finalize_iteration_head_sample(
    response: IterationHeadResponse,
    omega_index: int,
    W_body_gamma=None,
    *,
    wfn,
    meta,
    config,
    mesh: Mesh,
):
    r"""Finalize one response frequency while its total body W is resident.

    This is the disk-bounded MPA seam: the caller passes total screened
    W_body_gamma, never Wc, and only the replicated 3x3 Schur result
    survives the call. Left and right wings remain independent at complex
    frequency.
    """
    index = int(omega_index)
    if not 0 <= index < len(response.omegas):
        raise IndexError(
            f"head frequency index {index} outside [0,{len(response.omegas)})")
    S_effective = response.S_direct[index]
    if W_body_gamma is not None:
        if response.Y_x is None or response.Z_y is None:
            raise ValueError(
                "body-screened QSGW head requested without head/body wings")
        W = jnp.asarray(W_body_gamma)
        if (
            int(W.shape[-2]) != int(response.Y_x.shape[-1])
            or int(W.shape[-1]) != int(response.Z_y.shape[-2])
        ):
            raise ValueError(
                "QSGW head-wing centroid extents do not match W(Gamma): "
                f"Y={response.Y_x.shape}, W={W.shape}, Z={response.Z_y.shape}")
        from gw.head_correction import fold_cartesian_head_wings_sharded
        S_effective = fold_cartesian_head_wings_sharded(
            response.S_direct[index],
            response.Y_x[index],
            W,
            response.Z_y[index],
            float(meta.cell_volume),
            mesh_xy=mesh,
        )
    static_kappa2 = response.static_kappa2_bohr2
    if abs(response.omegas[index]) <= 1.0e-14:
        static_kappa2 = _fold_static_kappa2(
            response, W_body_gamma, float(meta.cell_volume), mesh)
    return head_samples_from_s(
        S_effective[None, :, :],
        (response.omegas[index],),
        wfn=wfn,
        meta=meta,
        config=config,
        static_kappa2_bohr2=static_kappa2,
    )[0]


def finalize_iteration_head_samples(
    response: IterationHeadResponse,
    *,
    wfn,
    meta,
    config,
    mesh: Mesh,
    requests=None,
    W_by_role=None,
) -> IterationHeadSamples:
    """Apply the optional body Schur fold and mini-BZ-average the head.

    ``W_by_role`` is the already-screened finite-G/centroid body returned by
    :func:`gw.screening.compute_screening`.  Flat q index zero is Gamma in
    the production C-order convention, and its singular head channel is
    absent, so ``W_by_role[role][0]`` is precisely the body operand required
    by the bordered-Dyson reduction.

    Passing no ``W_by_role`` intentionally produces the direct-head result.
    This keeps the one-shot diagnostic API and X-only path unchanged.
    """
    S_effective = response.S_direct
    if W_by_role:
        if response.Y_x is None or response.Z_y is None:
            raise ValueError(
                "body-screened QSGW head requested without head/body wings")
        if requests is None:
            raise ValueError("screening requests are required to match W roles")
        reqs = tuple(requests)
        if len(reqs) != len(response.omegas):
            raise ValueError(
                f"head has {len(response.omegas)} frequencies but screening "
                f"has {len(reqs)} requests")
        W_gamma = []
        for omega, req in zip(response.omegas, reqs):
            if abs(complex(req.omega_ry) - omega) > 1.0e-12:
                raise ValueError(
                    f"head/screening frequency mismatch: {omega} vs "
                    f"{req.omega_ry} ({req.role})")
            try:
                W_role = W_by_role[req.role]
            except KeyError as exc:
                raise KeyError(
                    f"screening did not return required head role {req.role!r}") \
                    from exc
            W_gamma.append(W_role[0])
        W_gamma = jnp.stack(W_gamma, axis=0)
        if (
            int(W_gamma.shape[-2]) != int(response.Y_x.shape[-1])
            or int(W_gamma.shape[-1]) != int(response.Z_y.shape[-2])
        ):
            raise ValueError(
                "QSGW head-wing centroid extents do not match W(Gamma): "
                f"Y={response.Y_x.shape}, W={W_gamma.shape}, "
                f"Z={response.Z_y.shape}")
        from gw.head_correction import fold_cartesian_head_wings_sharded
        S_effective = fold_cartesian_head_wings_sharded(
            response.S_direct,
            response.Y_x,
            W_gamma,
            response.Z_y,
            float(meta.cell_volume),
            mesh_xy=mesh,
        )
    static_kappa2 = response.static_kappa2_bohr2
    if W_by_role and static_kappa2 is not None:
        static_indices = [
            i for i, z in enumerate(response.omegas) if abs(z) <= 1.0e-14]
        if len(static_indices) != 1:
            raise ValueError(
                "static metallic head requires exactly one z=0 response")
        static_kappa2 = _fold_static_kappa2(
            response, W_gamma[static_indices[0]], float(meta.cell_volume), mesh)
    samples = head_samples_from_s(
        S_effective,
        response.omegas,
        wfn=wfn,
        meta=meta,
        config=config,
        static_kappa2_bohr2=static_kappa2,
    )
    return IterationHeadSamples(
        omegas=response.omegas,
        samples=samples,
        sigma_energies_ry=response.sigma_energies_ry,
        sigma_occupations=response.sigma_occupations,
        efermi_ry=response.efermi_ry,
    )


def head_samples_from_s(
    S_cart_omega,
    omegas_ry,
    *,
    wfn,
    meta,
    config,
    static_kappa2_bohr2: float | None = None,
) -> tuple[object, ...]:
    """Convert replicated 3x3 S tensors to mini-BZ averaged head samples."""
    from gw.head_correction import HeadSample, resolve_head_override
    from gw.vcoul import compute_q0_averages

    S_host = np.asarray(S_cart_omega, dtype=np.complex128)
    omegas = tuple(complex(z) for z in np.asarray(omegas_ry).reshape(-1))
    if S_host.shape != (len(omegas), 3, 3):
        raise ValueError(
            f"S_cart_omega must be ({len(omegas)},3,3), got {S_host.shape}."
        )
    params = {
        "vhead": config.head.vhead,
        "whead_0freq": config.head.whead_0freq,
        "whead_imfreq": config.head.whead_imfreq,
    }
    out = []
    for z, S in zip(omegas, S_host):
        override = resolve_head_override(params, z)
        if override is not None:
            out.append(override)
            continue
        is_static_metal = (
            static_kappa2_bohr2 is not None and abs(z) <= 1.0e-14)
        vc0, wc0 = compute_q0_averages(
            wfn,
            jnp.asarray(0.0, dtype=jnp.float64),
            meta,
            S_cart=None if is_static_metal else S,
            static_kappa2=(
                jnp.asarray(static_kappa2_bohr2, dtype=jnp.float64)
                if is_static_metal else None),
            analytic_sphere=bool(getattr(
                config.head, "analytic_q0_sphere",
                config.head.head_minibz_average)),
        )
        out.append(
            HeadSample(
                vc0=complex(vc0),
                wcoul0=complex(wc0),
                source=(
                    ("qsgw_parallel_transport_tf"
                     if is_static_metal else "qsgw_parallel_transport")
                    if abs(z) <= 1.0e-14
                    else f"qsgw_parallel_transport(omega={z} Ry)"
                ),
                omega=z,
                S_cart=None if is_static_metal else S,
            )
        )
    return tuple(out)


def build_iteration_head_response(
    delta_h_dft,
    connection_cart,
    velocity_dft_cart,
    U_dft_to_qp,
    energies_qp_kn_ry,
    occupations_qp_kn,
    omegas_ry,
    *,
    surface_weight_qp_kn=None,
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    bvec_cart,
    nb_logical: int,
    sigma_energies_ry,
    efermi_ry: float,
    wfn,
    meta,
    config,
    wfns_qp=None,
    eta_ry: float | None = None,
) -> IterationHeadResponse:
    """Build current-basis direct head and, when requested, its wings.

    ``connection_cart=None`` is ``sc_head_update = dft_velocity``: no Berry
    connection is resident, so the covariant ``DΔH`` correction is dropped
    and the bare DFT p-matrix velocity enters.  ``delta_h_dft`` is then
    unused and may be None.  Everything downstream of the velocity —
    the per-iteration rotation into the QP basis, S(z), the Drude term, the
    ISDF wings, the static κ² — is the SAME code on both routes.
    """
    v_dft_basis = jnp.asarray(velocity_dft_cart, dtype=jnp.complex128)
    if connection_cart is not None:
        v_dft_basis = v_dft_basis + covariant_structured_delta(
            delta_h_dft,
            connection_cart,
            U_active=U_dft_to_qp,
            mesh=mesh,
            kgrid=kgrid,
            bvec_cart=bvec_cart,
        )
    v_qp = rotate_velocity_active_to_qp(v_dft_basis, U_dft_to_qp, mesh=mesh)
    resolved_eta_ry = (
        float(config.head.wcoul0_eta)
        if eta_ry is None else float(eta_ry)
    )
    S = head_s_tensor_sharded(
        v_qp,
        energies_qp_kn_ry,
        occupations_qp_kn,
        omegas_ry,
        mesh=mesh,
        nb_logical=nb_logical,
        cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot),
        nspin=int(wfn.nspin),
        nspinor=int(meta.nspinor),
        eta_ry=resolved_eta_ry,
        surface_weight_kn=surface_weight_qp_kn,
    )
    Y_x = Z_y = None
    static_Y_x = static_Z_y = static_chi_body_gamma = None
    if wfns_qp is not None:
        Y_x, Z_y = head_wings_sharded(
            v_qp,
            wfns_qp,
            energies_qp_kn_ry,
            occupations_qp_kn,
            omegas_ry,
            mesh=mesh,
            nb_logical=nb_logical,
            nk_tot=int(meta.nk_tot),
            nspin=int(wfn.nspin),
            nspinor=int(meta.nspinor),
            eta_ry=resolved_eta_ry,
            surface_weight_kn=surface_weight_qp_kn,
        )
    omegas = tuple(complex(z) for z in np.asarray(omegas_ry).reshape(-1))
    if (
        wfns_qp is not None
        and surface_weight_qp_kn is not None
        and any(abs(z) <= 1.0e-14 for z in omegas)
    ):
        static_Y_x, static_Z_y = static_head_wings_sharded(
            wfns_qp,
            surface_weight_qp_kn,
            mesh=mesh,
            nb_logical=int(nb_logical),
            nk_tot=int(meta.nk_tot),
            nspin=int(wfn.nspin),
            nspinor=int(meta.nspinor),
        )
        from gw.w_isdf import compute_chi0_static_fractional_gamma
        static_chi_body_gamma = compute_chi0_static_fractional_gamma(
            wfns_qp,
            energies_qp_kn_ry,
            occupations_qp_kn,
            surface_weight_qp_kn,
            meta,
            mesh,
            nb_logical=int(nb_logical),
        )
    static_kappa2 = None
    if surface_weight_qp_kn is not None:
        capacity = 2.0 / (
            float(max(int(wfn.nspin), 1))
            * float(max(int(meta.nspinor), 1)))
        # Tetrahedron weights arrive multiplied by Nk to share the distributed
        # Drude contraction's interface.  Undo that factor for the normalized
        # BZ density of states, then use kappa_TF^2=8*pi*DOS_Ry/Omega.
        dos_ry_per_cell = capacity * float(
            np.sum(np.asarray(surface_weight_qp_kn, dtype=np.float64))) / float(
                meta.nk_tot)
        static_kappa2 = (
            8.0 * np.pi * dos_ry_per_cell / float(meta.cell_volume))
    return IterationHeadResponse(
        omegas=omegas,
        S_direct=S,
        Y_x=Y_x,
        Z_y=Z_y,
        static_kappa2_bohr2=static_kappa2,
        static_Y_x=static_Y_x,
        static_Z_y=static_Z_y,
        static_chi_body_gamma=static_chi_body_gamma,
        sigma_energies_ry=np.asarray(sigma_energies_ry, dtype=np.float64),
        sigma_occupations=np.asarray(occupations_qp_kn, dtype=np.float64)[
            :, : np.shape(sigma_energies_ry)[1]
        ],
        efermi_ry=float(efermi_ry),
    )


def build_iteration_head_samples(
    delta_h_dft,
    connection_cart,
    velocity_dft_cart,
    U_dft_to_qp,
    energies_qp_kn_ry,
    occupations_qp_kn,
    omegas_ry,
    *,
    surface_weight_qp_kn=None,
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    bvec_cart,
    nb_logical: int,
    sigma_energies_ry,
    efermi_ry: float,
    wfn,
    meta,
    config,
) -> IterationHeadSamples:
    """Backward-compatible direct-head builder used by small diagnostics."""
    response = build_iteration_head_response(
        delta_h_dft,
        connection_cart,
        velocity_dft_cart,
        U_dft_to_qp,
        energies_qp_kn_ry,
        occupations_qp_kn,
        omegas_ry,
        surface_weight_qp_kn=surface_weight_qp_kn,
        mesh=mesh,
        kgrid=kgrid,
        bvec_cart=bvec_cart,
        nb_logical=nb_logical,
        sigma_energies_ry=sigma_energies_ry,
        efermi_ry=efermi_ry,
        wfn=wfn,
        meta=meta,
        config=config,
    )
    return finalize_iteration_head_samples(
        response, wfn=wfn, meta=meta, config=config, mesh=mesh)


def validate_dft_velocity_identity(
    h_dft_k,
    connection_cart,
    velocity_dft_cart,
    *,
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    bvec_cart,
) -> dict[str, float]:
    """Mandatory full-matrix gate for ``v = partial H - i[A,H]``.

    Reductions happen on device; only five scalars cross to the host.
    Diagonal and off-diagonal maxima are reported separately so a passing
    diagonal cannot hide a link-orientation error in the transition sector.
    """
    target = jnp.asarray(velocity_dft_cart, dtype=jnp.complex128)
    got = covariant_cartesian_derivative(
        h_dft_k, connection_cart, mesh=mesh, kgrid=kgrid, bvec_cart=bvec_cart
    )
    if got.shape != target.shape:
        raise ValueError(
            f"reconstructed/target velocity shapes differ: {got.shape}/{target.shape}."
        )
    diff = jnp.abs(got - target)
    nb = int(diff.shape[-1])
    eye = jnp.eye(nb, dtype=bool)[None, None, :, :]
    scale = jnp.max(jnp.abs(target))
    herm = jnp.max(jnp.abs(got - jnp.conj(jnp.swapaxes(got, -1, -2))))
    vals = (
        jnp.max(diff),
        jnp.max(jnp.where(eye, diff, 0.0)),
        jnp.max(jnp.where(~eye, diff, 0.0)),
        jnp.max(diff) / jnp.maximum(scale, 1.0e-30),
        herm,
    )
    names = ("max_abs", "max_abs_diag", "max_abs_offdiag", "max_rel", "hermiticity")
    return {name: float(value) for name, value in zip(names, vals)}
