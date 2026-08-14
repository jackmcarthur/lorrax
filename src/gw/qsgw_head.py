"""Parallel-transport covariant velocity for the self-consistent GW head.

The preprocessing job owns wavefunctions.  This module deliberately does
not: its inputs are the saved Berry connection ``A_cart``, the independently
exact DFT velocity, and the current fixed-DFT-basis QSGW Hamiltonian.  It
implements

    D_k H = partial_k H - i [A, H]
    v_Q   = v_DFT + D_k (H_Q - H_DFT)

and rebuilds the tiny Cartesian S tensor from the resulting band-tiled
velocity.  The only replicated result is ``(n_omega, 3, 3)``.

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
    "IterationHeadSamples",
    "ParallelTransportHeadData",
    "assemble_delta_head_manifold",
    "assemble_head_manifold",
    "build_iteration_head_samples",
    "covariant_cartesian_derivative",
    "covariant_structured_delta",
    "head_s_tensor_sharded",
    "head_samples_from_s",
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


@dataclass(frozen=True)
class ParallelTransportHeadData:
    """Validated, device-resident inputs held across the SC loop."""

    connection_cart: jax.Array
    velocity_dft_cart: jax.Array
    nb_logical: int
    reciprocal_lattice_cart: np.ndarray
    validation: dict[str, float]


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
            analytic_sphere=bool(config.head.head_minibz_average),
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
    """Build the complete no-wavefunction head state for one SC iteration."""
    correction = covariant_structured_delta(
        delta_h_dft,
        connection_cart,
        U_active=U_dft_to_qp,
        mesh=mesh,
        kgrid=kgrid,
        bvec_cart=bvec_cart,
    )
    v_dft_basis = jnp.asarray(velocity_dft_cart, dtype=jnp.complex128) + correction
    v_qp = rotate_velocity_active_to_qp(v_dft_basis, U_dft_to_qp, mesh=mesh)
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
        eta_ry=float(config.head.wcoul0_eta),
        surface_weight_kn=surface_weight_qp_kn,
    )
    omegas = tuple(complex(z) for z in np.asarray(omegas_ry).reshape(-1))
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
    samples = head_samples_from_s(
        S, omegas, wfn=wfn, meta=meta, config=config,
        static_kappa2_bohr2=static_kappa2)
    return IterationHeadSamples(
        omegas=omegas,
        samples=samples,
        sigma_energies_ry=np.asarray(sigma_energies_ry, dtype=np.float64),
        sigma_occupations=np.asarray(occupations_qp_kn, dtype=np.float64)[
            :, : np.shape(sigma_energies_ry)[1]
        ],
        efermi_ry=float(efermi_ry),
    )


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
