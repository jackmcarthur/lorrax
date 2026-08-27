"""Real-CUDA P4 gate for the controlled-band QSGW orbital moment.

The gate checks the complete operator path against independent host algebra:

1. ``v_QP = U^H [v_DFT + D_link DeltaH] U`` for identity links and a
   fourth-order host stencil;
2. ``<D_i u_n|v_j|u_n>`` against an explicit controlled-band sum; and
3. the axial intrinsic moment together with the required x/y shardings.

Run with four ranks and one CUDA device per rank on a 2x2 mesh.
"""

from __future__ import annotations

from runtime import initialize_communicator_stack

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from common.parallel_transport import build_forward_neighbor_table  # noqa: E402
from gw.qsgw_head import build_covariant_qsgw_velocity  # noqa: E402
from psp.orbital_response import (  # noqa: E402
    band_orbital_moment_mu_b,
    controlled_band_orbital_contraction,
)


def _put(array, mesh, spec):
    return jax.device_put(
        np.asarray(array), NamedSharding(mesh, P(*spec)))


def _gather(array):
    if jax.process_count() == 1:
        return np.asarray(array)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(array, tiled=True))


def _haar(rng, size):
    matrix = (
        rng.standard_normal((size, size))
        + 1.0j * rng.standard_normal((size, size))
    )
    q, r = np.linalg.qr(matrix)
    phase = np.diagonal(r)
    return q * (phase / np.abs(phase))[None, :]


def _host_fourth_order(operator, plus, grid):
    minus = np.empty_like(plus)
    for idir in range(3):
        minus[plus[:, idir], idir] = np.arange(plus.shape[0])
    rows = []
    for idir in range(3):
        kp1 = plus[:, idir]
        kp2 = plus[kp1, idir]
        km1 = minus[:, idir]
        km2 = minus[km1, idir]
        spacing = 1.0 / grid[idir]
        rows.append(
            (-operator[kp2] + 8.0 * operator[kp1]
             - 8.0 * operator[km1] + operator[km2])
            / (12.0 * spacing)
        )
    return np.stack(rows)


def _host_contraction(velocity, energies):
    nk, nb_logical = energies.shape
    result = np.zeros((nk, nb_logical, 3, 3), dtype=np.complex128)
    for ik in range(nk):
        for n in range(nb_logical):
            for m in range(nb_logical):
                if m == n:
                    continue
                result[ik, n] += np.einsum(
                    "i,j->ij",
                    np.conj(velocity[:, ik, m, n]),
                    velocity[:, ik, m, n],
                ) / (energies[ik, n] - energies[ik, m])
    return result


def _host_moment(contraction):
    axial = np.stack([
        contraction[..., 1, 2] - contraction[..., 2, 1],
        contraction[..., 2, 0] - contraction[..., 0, 2],
        contraction[..., 0, 1] - contraction[..., 1, 0],
    ], axis=-1)
    return 0.5 * np.imag(axial)


def main():
    mesh = RUNTIME.mesh
    rank0 = print if jax.process_index() == 0 else (lambda *args, **kwargs: None)
    if tuple(int(n) for n in mesh.devices.shape) != (2, 2):
        raise ValueError(f"gate requires a 2x2 mesh, got {mesh.devices.shape}")

    rng = np.random.default_rng(202608264)
    grid = (5, 5, 5)
    nk = int(np.prod(grid))
    nb_logical, nb_storage, nb_active = 6, 8, 4
    coordinates = np.stack(
        np.meshgrid(*(np.arange(n) for n in grid), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    plus = build_forward_neighbor_table(coordinates, grid)

    raw_delta = (
        rng.standard_normal((nk, nb_storage, nb_storage))
        + 1.0j * rng.standard_normal((nk, nb_storage, nb_storage))
    )
    delta_h = 0.5 * (raw_delta + raw_delta.swapaxes(-1, -2).conj())
    links = np.broadcast_to(
        np.eye(nb_storage, dtype=np.complex128),
        (3, nk, nb_storage, nb_storage),
    ).copy()
    raw_velocity = (
        rng.standard_normal((3, nk, nb_storage, nb_storage))
        + 1.0j * rng.standard_normal((3, nk, nb_storage, nb_storage))
    )
    velocity_dft = 0.5 * (
        raw_velocity + raw_velocity.swapaxes(-1, -2).conj())
    rotation = np.stack([_haar(rng, nb_active) for _ in range(nk)])
    bvec = np.asarray([
        [1.7, 0.1, 0.0],
        [0.0, 1.2, 0.2],
        [0.1, 0.0, 0.9],
    ])

    velocity_qp = build_covariant_qsgw_velocity(
        _put(delta_h, mesh, (None, "x", "y")),
        _put(links, mesh, (None, None, "x", "y")),
        plus,
        _put(velocity_dft, mesh, (None, None, "x", "y")),
        _put(rotation, mesh, (None, "x", "y")),
        mesh=mesh,
        kgrid=grid,
        bvec_cart=bvec,
    )
    if velocity_qp.sharding.spec != P(None, None, "x", "y"):
        raise AssertionError(
            f"QP velocity lost x/y band tiling: {velocity_qp.sharding.spec}")

    reduced = _host_fourth_order(delta_h, plus, grid)
    derivative = np.einsum(
        "ij,jkmn->ikmn", np.linalg.inv(bvec), reduced, optimize=True)
    corrected = velocity_dft + derivative
    full_rotation = np.broadcast_to(
        np.eye(nb_storage, dtype=np.complex128),
        (nk, nb_storage, nb_storage),
    ).copy()
    full_rotation[:, :nb_active, :nb_active] = rotation
    velocity_ref = np.einsum(
        "kmp,akmn,knq->akpq",
        full_rotation.conj(), corrected, full_rotation, optimize=True)
    velocity_global = _gather(velocity_qp)
    velocity_error = float(np.max(np.abs(velocity_global - velocity_ref)))
    velocity_scale = max(float(np.max(np.abs(velocity_ref))), 1.0e-300)
    velocity_relative = velocity_error / velocity_scale

    energies = np.stack([
        0.41 * np.arange(nb_logical) + 0.002 * ik
        for ik in range(nk)
    ])
    contraction = controlled_band_orbital_contraction(
        velocity_qp,
        energies,
        mesh=mesh,
        degeneracy_tolerance_ry=1.0e-10,
    )
    if contraction.sharding.spec != P(None, "y", None, None):
        raise AssertionError(
            "controlled-band output has the wrong bounded layout: "
            f"{contraction.sharding.spec}")
    contraction_ref = _host_contraction(velocity_ref, energies)
    contraction_global = _gather(contraction)
    contraction_error = float(np.max(np.abs(
        contraction_global - contraction_ref)))
    contraction_scale = max(
        float(np.max(np.abs(contraction_ref))), 1.0e-300)
    contraction_relative = contraction_error / contraction_scale

    moment = _gather(band_orbital_moment_mu_b(contraction))
    moment_ref = _host_moment(contraction_ref)
    moment_error = float(np.max(np.abs(moment - moment_ref)))

    tolerance = 2.0e-11
    if velocity_relative > tolerance:
        raise AssertionError(
            f"covariant QSGW velocity relative error {velocity_relative:.3e}")
    if contraction_relative > tolerance:
        raise AssertionError(
            "controlled-band contraction relative error "
            f"{contraction_relative:.3e}")
    if moment_error > tolerance:
        raise AssertionError(
            f"orbital-moment absolute error {moment_error:.3e}")

    rank0(
        "QSGW_ORBITAL_MOMENT_P4_PASS "
        f"mesh=2x2 velocity_spec={velocity_qp.sharding.spec} "
        f"contraction_spec={contraction.sharding.spec} "
        f"velocity_rel={velocity_relative:.3e} "
        f"contraction_rel={contraction_relative:.3e} "
        f"moment_abs={moment_error:.3e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
