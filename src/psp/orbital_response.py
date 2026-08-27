"""Distributed controlled-band orbital-response contractions.

This module is the named physics-kernel boundary between the orbital-
magnetization driver and JAX sharding.  It consumes the already band-tiled
QSGW velocity; it does not load, reconstruct, or place a Hamiltonian.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map


__all__ = [
    "band_orbital_moment_mu_b",
    "controlled_band_orbital_contraction",
    "controlled_band_orbital_moment_mu_b",
]


_CONTRACTION_KERNEL_CACHE = {}
_PHYSICAL_SLICE_CACHE = {}


def _controlled_band_orbital_contraction_kernel(
    mesh: Mesh,
    nb_logical: int,
):
    """Return the explicitly band-tiled contraction kernel."""
    key = (id(mesh), int(nb_logical))
    hit = _CONTRACTION_KERNEL_CACHE.get(key)
    if hit is not None:
        return hit

    axis_names = tuple(str(axis) for axis in mesh.axis_names)
    if axis_names != ("x", "y"):
        raise ValueError(
            "controlled-band orbital contraction requires the production "
            f"('x','y') mesh, got {axis_names!r}."
        )
    ax_x, ax_y = axis_names
    nphys = int(nb_logical)

    def _local(velocity, energies_m, energies_n, tolerance):
        nx, ny = velocity.shape[-2:]
        index_m = jax.lax.axis_index(ax_x) * nx + jnp.arange(nx)
        index_n = jax.lax.axis_index(ax_y) * ny + jnp.arange(ny)

        # delta[k,m,n] = E_n - E_m, matching
        # <u_m|D_i u_n> = v_i[m,n] / (E_n-E_m).
        delta = energies_n[:, None, :] - energies_m[:, :, None]
        physical = (
            (index_m[:, None] < nphys)
            & (index_n[None, :] < nphys)
        )[None, :, :]
        off_diagonal = physical & (
            index_m[:, None] != index_n[None, :]
        )[None, :, :]
        abs_delta = jnp.abs(delta)
        allowed = off_diagonal & (abs_delta > tolerance)
        inverse = jnp.where(
            allowed, 1.0 / jnp.where(allowed, delta, 1.0), 0.0)

        # T[k,n,i,j] = <D_i u_n | v_j | u_n>.  Only the local m slab is
        # formed; psum contracts it over x while n remains tiled over y.
        local = jnp.einsum(
            "ikmn,jkmn,kmn->knij",
            jnp.conj(velocity),
            velocity,
            inverse,
            optimize=True,
        )
        contraction = jax.lax.psum(local, ax_x)
        local_min_gap = jnp.min(
            jnp.where(off_diagonal, abs_delta, jnp.inf))
        min_gap = jax.lax.pmin(local_min_gap, (ax_x, ax_y))
        return contraction, min_gap

    mapped = shard_map(
        _local,
        mesh=mesh,
        in_specs=(
            P(None, None, "x", "y"),
            P(None, "x"),
            P(None, "y"),
            P(),
        ),
        out_specs=(P(None, "y", None, None), P()),
        check_vma=False,
    )
    kernel = jax.jit(mapped)
    _CONTRACTION_KERNEL_CACHE[key] = kernel
    return kernel


def _physical_band_slice_kernel(mesh: Mesh, nb_logical: int):
    """Remove storage padding while retaining the surviving y-band shard."""
    key = (id(mesh), int(nb_logical))
    hit = _PHYSICAL_SLICE_CACHE.get(key)
    if hit is not None:
        return hit
    out_sharding = NamedSharding(mesh, P(None, "y", None, None))
    kernel = jax.jit(
        lambda contraction: contraction[:, :int(nb_logical)],
        out_shardings=out_sharding,
    )
    _PHYSICAL_SLICE_CACHE[key] = kernel
    return kernel


def controlled_band_orbital_contraction(
    velocity_qp_cart,
    energies_qp_kn_ry,
    *,
    mesh: Mesh,
    degeneracy_tolerance_ry: float,
):
    """Contract the controlled-band QSGW ``D_i psi`` and ``v_j psi``.

    This evaluates

    ``T[k,n,i,j] = <D_i u_nk | v_j^QP | u_nk>``

    with

    ``<u_mk|D_i u_nk> = v_i^QP[m,n] / (E_n^QP-E_m^QP)``.

    Both factors must use the finite-link QSGW velocity produced by
    :func:`gw.qsgw_head.build_covariant_qsgw_velocity`.  At P>1 that input
    must already have ``P(None,None,'x','y')`` sharding; this kernel never
    places or materializes a full band-pair carrier on one process.

    Parameters
    ----------
    velocity_qp_cart : (3, nk, nb_storage, nb_storage) complex
        Cartesian ``dH_QP/dk`` in the QP basis, in Ry*bohr.  Its two band
        axes are distributed over the square processor mesh.
    energies_qp_kn_ry : (nk, nb_logical) float
        Matching physical QP eigenvalues in Ry.  A larger velocity extent is
        permitted only as canonical nonphysical storage padding.
    mesh : Mesh
        Square production processor mesh with axes ``('x','y')``.
    degeneracy_tolerance_ry : float
        Required isolated-band tolerance in Ry.  If any two retained bands
        lie within it, the routine refuses rather than assigning a
        gauge-arbitrary individual-band derivative.

    Returns
    -------
    (nk, nb_logical, 3, 3) complex
        ``<D_i u_n|v_j|u_n>`` with ``P(None,'y',None,None)`` sharding.

    Notes
    -----
    The controlled band ceiling must already have passed LORRAX's canonical
    band-window degeneracy guard against a larger spectrum.  A window cannot
    certify its own outer edge.
    """
    v_shape = tuple(np.shape(velocity_qp_cart))
    e_shape = tuple(np.shape(energies_qp_kn_ry))
    if len(v_shape) != 4 or v_shape[0] != 3:
        raise ValueError(
            "velocity_qp_cart must have shape (3,nk,nb,nb), got "
            f"{v_shape}."
        )
    if len(e_shape) != 2:
        raise ValueError(
            "energies_qp_kn_ry must have shape (nk,nb), got "
            f"{e_shape}."
        )
    if v_shape[1] != e_shape[0] or v_shape[2] != v_shape[3] \
            or v_shape[2] < e_shape[1]:
        raise ValueError(
            "QSGW velocity/energy bundle mismatch: velocity has "
            f"{v_shape}, energies have {e_shape}. Both velocity axes and "
            "the energy axis must describe the same controlled band window "
            "(velocity axes may contain only canonical storage padding)."
        )
    nb_storage = int(v_shape[2])
    nb_logical = int(e_shape[1])
    px, py = (int(n) for n in mesh.devices.shape)
    if nb_storage % px or nb_storage % py:
        raise ValueError(
            f"velocity storage extent {nb_storage} must be divisible by the "
            f"processor mesh ({px},{py}); use the canonical band-storage "
            "extent before this contraction."
        )
    expected_spec = P(None, None, "x", "y")
    actual_spec = getattr(getattr(velocity_qp_cart, "sharding", None),
                          "spec", None)
    if px * py > 1 and actual_spec != expected_spec:
        raise ValueError(
            "controlled_band_orbital_contraction requires an already-sharded "
            f"QP velocity with {expected_spec}, got {actual_spec!r}."
        )
    tol = float(degeneracy_tolerance_ry)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(
            "degeneracy_tolerance_ry must be finite and non-negative, got "
            f"{degeneracy_tolerance_ry!r}."
        )

    velocity = jnp.asarray(velocity_qp_cart, dtype=jnp.complex128)
    energies = jnp.asarray(energies_qp_kn_ry, dtype=jnp.float64)
    if nb_storage > nb_logical:
        energies = jnp.pad(energies, ((0, 0), (0, nb_storage - nb_logical)))
    contraction, min_gap = _controlled_band_orbital_contraction_kernel(
        mesh, nb_logical)(
        velocity,
        energies,
        energies,
        jnp.asarray(tol, dtype=jnp.float64),
    )
    min_gap_value = float(jax.device_get(min_gap))
    if min_gap_value <= tol:
        raise ValueError(
            "controlled_band_orbital_contraction: the retained QP spectrum "
            f"contains an interband gap {min_gap_value:.6e} Ry <= the "
            f"isolated-band tolerance {tol:.6e} Ry. Individual band moments "
            "are gauge-arbitrary there; use a whole-degenerate-block moment "
            "matrix instead of masking the denominator."
        )
    return _physical_band_slice_kernel(mesh, nb_logical)(contraction)


@jax.jit
def band_orbital_moment_mu_b(contraction_knij):
    """Return the intrinsic band orbital moment in Bohr magnetons.

    For electron charge ``-e`` in this module's Rydberg convention,

    ``m_a/mu_B = +(1/2) Im epsilon_aij <D_i u|v_j|u>``.

    This is the local/self-rotation band moment, not the additional
    itinerant/Berry term in total modern-theory orbital magnetization.
    """
    contraction = jnp.asarray(contraction_knij, dtype=jnp.complex128)
    axial = jnp.stack([
        contraction[..., 1, 2] - contraction[..., 2, 1],
        contraction[..., 2, 0] - contraction[..., 0, 2],
        contraction[..., 0, 1] - contraction[..., 1, 0],
    ], axis=-1)
    return 0.5 * jnp.imag(axial)


def controlled_band_orbital_moment_mu_b(
    velocity_qp_cart,
    energies_qp_kn_ry,
    *,
    mesh: Mesh,
    degeneracy_tolerance_ry: float,
):
    """Return ``m_n(k)/mu_B`` through the QSGW ``dpsi``--``vpsi`` route."""
    contraction = controlled_band_orbital_contraction(
        velocity_qp_cart,
        energies_qp_kn_ry,
        mesh=mesh,
        degeneracy_tolerance_ry=degeneracy_tolerance_ry,
    )
    return band_orbital_moment_mu_b(contraction)
