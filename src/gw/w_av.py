"""Finite-q W-av stencil response primitives.

The preprocessing artifact stores fixed-DFT-gauge density overlaps one source
q at a time.  This module supplies the first consumer seam: rotate one row to
the current QP basis without constructing a full-manifold unitary, then form
the occupation-aware scalar Adler-Wiser head on the exact source-q endpoint.
It deliberately does not interpolate or mini-BZ average yet.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from common.parallel_transport import make_distributed_band_matmul
from common.shard_map import shard_map


__all__ = [
    "HeadStencilFit",
    "finite_q_head_response",
    "fit_3d_head_stencil",
    "rotate_w_av_density_active_to_qp",
    "stream_source_q_head_responses",
    "unfold_source_q_scalars",
]


_KERNEL_CACHE: dict[tuple, Callable] = {}


# The 3-D metal dynamic head is constrained to start at q^2.  The W-av
# first/second shell contains axial and pair-mixed directions, which resolve
# every quadratic/cubic monomial except xyz.  That absent triple-mixed term
# is deliberately not guessed from a rank-deficient fit.
_DYNAMIC_HEAD_POWERS = np.asarray([
    (2, 0, 0), (1, 1, 0), (1, 0, 1),
    (0, 2, 0), (0, 1, 1), (0, 0, 2),
    (3, 0, 0), (2, 1, 0), (2, 0, 1),
    (1, 2, 0), (1, 0, 2),
    (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3),
], dtype=np.int32)
_STATIC_HEAD_POWERS = np.asarray([
    (0, 0, 0),
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (2, 0, 0), (1, 1, 0), (1, 0, 1),
    (0, 2, 0), (0, 1, 1), (0, 0, 2),
], dtype=np.int32)


def _monomial_design(points, powers):
    q = np.asarray(points, dtype=np.float64)
    p = np.asarray(powers, dtype=np.int32)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"W-av stencil points must be (n,3); got {q.shape}")
    return np.prod(q[:, None, :] ** p[None, :, :], axis=-1)


@dataclass(frozen=True)
class HeadStencilFit:
    """Small replicated polynomial for the Gamma-cell head auxiliary.

    Coordinates are reciprocal-grid steps, ``x_i=dq_frac_i*kgrid_i``.
    Dynamic fits contain only quadratic and cubic terms, so the required
    3-D-metal ``q^2`` limit is exact by construction.  Static fits use one
    shared scalar intercept and an ordinary quadratic correction.
    """

    powers: np.ndarray
    coefficients: np.ndarray
    kgrid: np.ndarray
    static: bool

    def evaluate_steps(self, steps):
        design = _monomial_design(
            np.atleast_2d(np.asarray(steps, dtype=np.float64)), self.powers)
        value = np.tensordot(design, self.coefficients, axes=(1, 0))
        return value[0] if np.asarray(steps).ndim == 1 else value

    def evaluate_fractional(self, delta_q_frac):
        q = np.asarray(delta_q_frac, dtype=np.float64)
        return self.evaluate_steps(q * self.kgrid)


def unfold_source_q_scalars(source_values, metadata):
    """Broadcast scalar source-q values over the SymMaps target stencil."""
    values = np.asarray(source_values)
    if values.ndim < 1 or values.shape[0] != metadata.nq_source:
        raise ValueError(
            "source W-av values must start with the artifact source-q axis")
    rows = np.asarray(metadata.target_source_row, dtype=np.int32)
    if np.any(rows < 0) or np.any(rows >= metadata.nq_source):
        raise ValueError("W-av target_source_row is outside the source table")
    # A scalar density response is invariant under the spatial/antiunitary
    # q-star action.  Vector wings and centroid bodies use their existing
    # symmetry-service actions instead of passing through this scalar door.
    return values[rows]


def fit_3d_head_stencil(source_values, metadata, *, static: bool):
    """Fit the symmetry-completed first/second-neighbor Gamma stencil.

    ``source_values`` may carry arbitrary trailing axes (normally MPA
    frequencies).  It must be the already Schur-folded head auxiliary for a
    production W-av fit; the direct Adler-Wiser head is useful only as a
    diagnostic.  The solve is a fixed, tiny replicated normal equation, not
    a band/centroid linear-algebra path.
    """
    if not metadata.first_neighbors or not metadata.second_neighbors:
        raise ValueError(
            "3-D metal W-av interpolation requires both "
            "W_av_first_neighbors and W_av_second_neighbors")
    if np.any(np.abs(np.asarray(metadata.kgrid_shift)) > 1.0e-12):
        raise ValueError("3-D W-av Gamma stencils require an unshifted q grid")
    values = unfold_source_q_scalars(source_values, metadata)
    powers = _STATIC_HEAD_POWERS if bool(static) else _DYNAMIC_HEAD_POWERS
    design = _monomial_design(metadata.target_steps, powers)
    gram = design.T @ design
    eig = np.linalg.eigvalsh(gram)
    if eig[0] <= 1.0e-12 * eig[-1]:
        raise ValueError(
            "W-av first/second-neighbor stencil does not resolve the "
            f"requested head polynomial (Gram spectrum {eig[0]:.3e}, "
            f"{eig[-1]:.3e})")
    trailing = values.shape[1:]
    rhs = design.T @ values.reshape(values.shape[0], -1)
    coefficients = np.linalg.solve(gram, rhs).reshape(
        (powers.shape[0],) + trailing)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("W-av head polynomial contains a non-finite value")
    return HeadStencilFit(
        powers=powers.copy(), coefficients=coefficients,
        kgrid=np.asarray(metadata.kgrid, dtype=np.float64),
        static=bool(static))


def stream_source_q_head_responses(
    reader,
    energies_ry,
    occupations,
    z_samples_ry,
    *,
    mesh: Mesh,
    nspin: int,
    nspinor: int,
    surface_weight=None,
    U_active=None,
):
    """Read, evaluate, and release exactly one stored q row at a time.

    SlabIO remains outside JIT ownership.  Blocking on each replicated,
    frequency-sized result before the next collective read ensures the large
    ``(k,band,band)`` row cannot queue behind another row on device.
    """
    out = []
    meta = reader.metadata
    for iq in range(meta.nq_source):
        rho = reader.read_source_row(iq)
        sample = finite_q_head_response(
            rho, energies_ry, occupations, meta.source_kplusq_full[iq],
            z_samples_ry,
            mesh=mesh, nb_logical=meta.nbands,
            nspin=int(nspin), nspinor=int(nspinor),
            surface_weight=surface_weight, U_active=U_active)
        sample = jax.block_until_ready(sample)
        out.append(np.asarray(sample))
        del sample, rho
    return np.stack(out, axis=0)


def _mesh_xy(mesh: Mesh) -> tuple[str, str]:
    names = tuple(str(axis) for axis in mesh.axis_names)
    if names != ("x", "y"):
        raise ValueError(
            "W-av response requires mesh axes ('x','y'); "
            f"got {names!r}")
    return names


def _active_cross_k_rotation_kernel(mesh: Mesh, nb_active: int) -> Callable:
    key = ("w_av_cross_k_rotation", id(mesh), int(nb_active))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    multiply = make_distributed_band_matmul(mesh, n_batch_axes=1)
    na = int(nb_active)

    @jax.jit
    def _kernel(rho, U_center, U_neighbor):
        eye = jnp.eye(na, dtype=U_center.dtype)[None]
        right_change = U_neighbor - eye
        right = multiply(rho[:, :, :na], right_change)
        tmp = rho.at[:, :, :na].add(right)
        left_change_h = jnp.swapaxes(
            jnp.conj(U_center - eye), -1, -2)
        left = multiply(left_change_h, tmp[:, :na, :])
        return tmp.at[:, :na, :].add(left)

    _KERNEL_CACHE[key] = _kernel
    return _kernel


def rotate_w_av_density_active_to_qp(
    rho_dft, U_active, kplusq_full, *, mesh: Mesh,
):
    """Return ``U_k^H rho_DFT(k,q) U_{k+q}`` on both band shards.

    ``U_active[k,m,i]=<DFT_mk|QP_ik>`` rotates only the active block;
    bands outside it retain the identity.  No dense block-diagonal unitary is
    constructed, and the endpoint unitary is gathered with the artifact's
    exact ``kplusq_full`` table.
    """
    rho = jnp.asarray(rho_dft)
    U = jnp.asarray(U_active)
    endpoint = np.asarray(kplusq_full, dtype=np.int32)
    if rho.ndim != 3 or rho.shape[-2] != rho.shape[-1]:
        raise ValueError(f"rho_dft must be (nk,nb,nb); got {rho.shape}")
    if U.ndim != 3 or U.shape[-2] != U.shape[-1]:
        raise ValueError(f"U_active must be (nk,na,na); got {U.shape}")
    if U.shape[0] != rho.shape[0] or endpoint.shape != (rho.shape[0],):
        raise ValueError("rho, U_active and kplusq_full k extents differ")
    if U.shape[-1] > rho.shape[-1]:
        raise ValueError("active QP block exceeds the W-av band manifold")
    if np.any(endpoint < 0) or np.any(endpoint >= rho.shape[0]):
        raise ValueError("kplusq_full contains an out-of-range endpoint")
    return _active_cross_k_rotation_kernel(mesh, int(U.shape[-1]))(
        rho, U, U[endpoint])


def _finite_q_head_kernel(mesh: Mesh, nb_logical: int) -> Callable:
    key = ("w_av_finite_q_head", id(mesh), int(nb_logical))
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit
    ax_x, ax_y = _mesh_xy(mesh)
    nb = int(nb_logical)

    def _local(rho, ea, eb, fa, fb, sa, sb, z_samples, prefactor):
        nx, ny = rho.shape[-2:]
        ix = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        iy = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)
        logical = (ix[:, None] < nb) & (iy[None, :] < nb)
        delta = ea[:, :, None] - eb[:, None, :]
        f_diff = fa[:, :, None] - fb[:, None, :]
        surface = -0.5 * (sa[:, :, None] + sb[:, None, :])
        strength = jnp.real(jnp.conj(rho) * rho)

        def _one(z):
            static = (jnp.abs(jnp.real(z)) < 1.0e-15) & (
                jnp.abs(jnp.imag(z)) < 1.0e-15)
            scale = jnp.maximum(
                1.0,
                jnp.maximum(jnp.abs(ea[:, :, None]),
                            jnp.abs(eb[:, None, :])),
            )
            separated = (
                jnp.abs(delta)
                > 64.0 * jnp.finfo(jnp.float64).eps * scale)
            static_weight = jnp.where(
                separated,
                f_diff / jnp.where(separated, delta, 1.0),
                surface)
            dynamic_weight = f_diff / (z + delta)
            weight = jnp.where(static, static_weight, dynamic_weight)
            local = prefactor * jnp.sum(
                jnp.where(logical[None, :, :], strength * weight, 0.0))
            return jax.lax.psum(local, (ax_x, ax_y))

        return jax.vmap(_one)(z_samples)

    mapped = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, "x", "y"),
            P(None, "x"), P(None, "y"),
            P(None, "x"), P(None, "y"),
            P(None, "x"), P(None, "y"),
            P(None), P(),
        ),
        out_specs=P(None),
        check_vma=False,
    )
    kernel = jax.jit(mapped)
    _KERNEL_CACHE[key] = kernel
    return kernel


def _pad_bands(table, nb_storage: int):
    value = jnp.asarray(table)
    if value.ndim != 2 or value.shape[1] > int(nb_storage):
        raise ValueError(
            f"band table must be (nk,nb<= {nb_storage}); got {value.shape}")
    return jnp.pad(value, ((0, 0), (0, int(nb_storage) - value.shape[1])))


def finite_q_head_response(
    rho_dft,
    energies_ry,
    occupations,
    kplusq_full,
    z_samples_ry,
    *,
    mesh: Mesh,
    nb_logical: int,
    nspin: int,
    nspinor: int,
    surface_weight=None,
    U_active=None,
):
    """Evaluate occupation-aware ``chi00(q,z)`` for one stored source q.

    The returned convention is

    ``2/(Nk*nspin*nspinor) sum_knm (f_nk-f_mkq)|rho_nm|^2 /
    (z+E_nk-E_mkq)``.

    Exact ``z=0`` samples use the divided-difference limit supplied by the
    positive ``surface_weight=-df/dE`` table.  Dynamic samples require no
    surface table.  MP1 occupations are signed and are never clipped.
    """
    rho = jnp.asarray(rho_dft)
    endpoint = np.asarray(kplusq_full, dtype=np.int32)
    z_host = np.asarray(z_samples_ry, dtype=np.complex128)
    if rho.ndim != 3 or rho.shape[-2] != rho.shape[-1]:
        raise ValueError(f"rho_dft must be (nk,nb,nb); got {rho.shape}")
    nk, nb_storage = int(rho.shape[0]), int(rho.shape[-1])
    nb = int(nb_logical)
    if not 0 < nb <= nb_storage:
        raise ValueError(
            f"nb_logical={nb} outside stored band extent {nb_storage}")
    if endpoint.shape != (nk,) or np.any(endpoint < 0) \
            or np.any(endpoint >= nk):
        raise ValueError("kplusq_full must contain one valid endpoint per k")
    if z_host.ndim != 1 or z_host.size == 0:
        raise ValueError("z_samples_ry must be a nonempty one-dimensional grid")
    exact_static = np.any(np.abs(z_host) < 1.0e-15)
    if exact_static and surface_weight is None:
        raise ValueError(
            "an exact z=0 W-av response requires surface_weight=-df/dE")

    energy = _pad_bands(energies_ry, nb_storage)
    occ = _pad_bands(occupations, nb_storage)
    if energy.shape[0] != nk or occ.shape[0] != nk:
        raise ValueError("energy/occupation and density k extents differ")
    surface = (_pad_bands(surface_weight, nb_storage)
               if surface_weight is not None else jnp.zeros_like(energy))
    if U_active is not None:
        rho = rotate_w_av_density_active_to_qp(
            rho, U_active, endpoint, mesh=mesh)

    prefactor = 2.0 / (
        float(nk) * float(max(1, int(nspin)))
        * float(max(1, int(nspinor))))
    return _finite_q_head_kernel(mesh, nb)(
        rho,
        energy, energy[endpoint],
        occ, occ[endpoint],
        surface, surface[endpoint],
        jnp.asarray(z_host),
        jnp.asarray(prefactor, dtype=jnp.complex128),
    )
