"""P=4 correctness/profile gate for parallel transport and QSGW head.

Launch through ``src/ffi/cpp/run_shifter.sh`` with four ranks.  This is a
real distributed-service gate: it requests the production polar backend,
round-trips hostile nondivisible SlabIO layouts, then times the complete
no-wavefunction per-iteration head kernel and inspects its lowered HLO for
explicit rematerialization markers.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from runtime import finalize_process, initialize_communicator_stack

RUNTIME = initialize_communicator_stack()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from common.collectives import (  # noqa: E402
    barrier,
    device_put_process_local,
    process_count,
    process_rank,
    resolve_mesh,
)
from distrib_la import plan_polar_factor  # noqa: E402
from common.parallel_transport import make_cross_k_link  # noqa: E402
from gw.qsgw_head import (  # noqa: E402
    _active_rotation_kernel,
    _assemble_delta_kernel,
    _cartesian_fft_multipliers,
    _spectral_kernel,
    _structured_delta_kernel,
    covariant_structured_delta,
    head_s_tensor_sharded,
    rotate_velocity_active_to_qp,
)


def _ready(value):
    jax.block_until_ready(value)
    return value


def _host(value):
    """Gather a tiled global array; read one local copy if replicated."""
    spec = tuple(getattr(value.sharding, "spec", ()) or ())
    if any(axis is not None for axis in spec):
        from jax.experimental import multihost_utils

        return np.asarray(multihost_utils.process_allgather(value, tiled=True))
    return np.asarray(value.addressable_shards[0].data)


def main() -> int:
    if process_count() not in (1, 4):
        raise RuntimeError(
            f"parallel_transport_profile requires 1 or 4 processes, got {process_count()}"
        )
    rank = process_rank()
    mesh = resolve_mesh()
    if tuple(mesh.shape.values()) != (2, 2):
        raise RuntimeError(f"expected the production 2x2 mesh, got {mesh.shape}")
    rng = np.random.default_rng(9001)

    n = 64
    tile = NamedSharding(mesh, P("x", "y"))
    matrix_host = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    matrix = device_put_process_local(matrix_host, tile)
    start = time.perf_counter()
    polar_backend = "distributed" if process_count() == 4 else "off"
    polar = plan_polar_factor(mesh, n=n, backend=polar_backend, rcond=1.0e-12)
    plan_s = time.perf_counter() - start
    start = time.perf_counter()
    link, singular = _ready(polar(matrix))
    cold_s = time.perf_counter() - start
    warm_s = []
    for _ in range(5):
        start = time.perf_counter()
        link, singular = _ready(polar(matrix))
        warm_s.append(time.perf_counter() - start)
    ref_u, ref_s, ref_vh = np.linalg.svd(matrix_host, full_matrices=False)
    link_error = np.max(np.abs(_host(link) - ref_u @ ref_vh)) / np.max(
        np.abs(ref_u @ ref_vh)
    )
    singular_error = np.max(np.abs(_host(singular) - ref_s)) / ref_s[0]
    if link_error > 3.0e-11 or singular_error > 3.0e-11:
        raise AssertionError(
            f"distributed polar mismatch: L={link_error}, s={singular_error}"
        )

    # Make the cross-k contraction equal matrix_host exactly, then time the
    # production composition in which the raw overlap feeds the planned polar
    # factor without returning to Python.
    center_host = np.zeros((1, n, 1, n), dtype=np.complex128)
    center_host[0, :, 0, :] = np.eye(n)
    neighbor_host = np.zeros_like(center_host)
    neighbor_host[0, :, 0, :] = matrix_host.T
    sphere = NamedSharding(mesh, P(None, ("x", "y"), None, None))
    center = device_put_process_local(center_host, sphere)
    neighbor = device_put_process_local(neighbor_host, sphere)
    center_on_x, make_link = make_cross_k_link(mesh, polar)
    center_x = _ready(center_on_x(center))
    g_index = np.arange(n, dtype=np.int32)
    g_valid = np.ones(n, dtype=bool)
    start = time.perf_counter()
    fused_link, fused_singular = _ready(
        make_link(center_x, neighbor, g_index, g_valid))
    fused_cold_s = time.perf_counter() - start
    fused_warm_s = []
    for _ in range(5):
        start = time.perf_counter()
        fused_link, fused_singular = _ready(
            make_link(center_x, neighbor, g_index, g_valid))
        fused_warm_s.append(time.perf_counter() - start)
    fused_link_error = np.max(
        np.abs(_host(fused_link) - ref_u @ ref_vh)) / np.max(
            np.abs(ref_u @ ref_vh))
    fused_singular_error = np.max(
        np.abs(_host(fused_singular) - ref_s)) / ref_s[0]
    if fused_link_error > 3.0e-11 or fused_singular_error > 3.0e-11:
        raise AssertionError(
            "fused overlap/polar mismatch: "
            f"L={fused_link_error}, s={fused_singular_error}"
        )

    slab_shapes = None
    if os.environ.get("PT_PROFILE_SKIP_SLAB", "0") != "1":
        # Lazy so a factorization-only service image need not install h5py;
        # the SlabIO leg is run in the full lorrax_J070 environment.
        from file_io.slab_io import SlabIO

        path = Path(os.environ["PT_PROFILE_PATH"])
        if rank == 0:
            path.unlink(missing_ok=True)
        barrier("pt-profile-before-slab")
        matrix_pad = np.zeros((3, 8, 8), dtype=np.complex128)
        matrix_pad[:, :5, :5] = rng.normal(size=(3, 5, 5)) + 1j * rng.normal(
            size=(3, 5, 5)
        )
        energy_pad = np.zeros((2, 8), dtype=np.float64)
        energy_pad[:, :5] = rng.normal(size=(2, 5))
        singular_pad = np.zeros((2, 3, 8), dtype=np.float64)
        singular_pad[:, :, :5] = np.sort(
            rng.random(size=(2, 3, 5)), axis=-1)[:, :, ::-1]
        matrix_dev = device_put_process_local(
            matrix_pad, NamedSharding(mesh, P(None, "x", "y"))
        )
        energy_dev = device_put_process_local(
            energy_pad, NamedSharding(mesh, P(None, ("x", "y")))
        )
        singular_dev = device_put_process_local(
            singular_pad, NamedSharding(mesh, P())
        )
        with SlabIO(path, mode="w", mesh=mesh) as io:
            io.create_dataset("matrix", shape=(3, 5, 5), dtype=np.complex128)
            io.create_dataset("energy", shape=(2, 5), dtype=np.float64)
            io.create_dataset(
                "singular", shape=(2, 3, 5), dtype=np.float64)
            io.write_slab("matrix", matrix_dev)
            io.write_slab("energy", energy_dev)
            io.write_slab("singular", singular_dev)
        with SlabIO(path, mode="r", mesh=mesh) as io:
            matrix_read = io.read_slab(
                "matrix", shape=(3, 8, 8), partition_spec=P(None, "x", "y")
            )
            energy_read = io.read_slab(
                "energy", shape=(2, 8), partition_spec=P(None, ("x", "y"))
            )
            singular_read = io.read_slab(
                "singular",
                shape=(2, 3, 8),
                partition_spec=P(None, None, ("x", "y")),
            )
        got_matrix = _host(matrix_read)
        got_energy = _host(energy_read)
        got_singular = _host(singular_read)
        if not (
            np.array_equal(got_matrix[:, :5, :5], matrix_pad[:, :5, :5])
            and not np.any(got_matrix[:, 5:, :])
            and not np.any(got_matrix[:, :, 5:])
            and np.array_equal(got_energy[:, :5], energy_pad[:, :5])
            and not np.any(got_energy[:, 5:])
            and np.array_equal(
                got_singular[:, :, :5], singular_pad[:, :, :5])
            and not np.any(got_singular[:, :, 5:])
        ):
            raise AssertionError("SlabIO common physical-extent round trip failed")
        slab_shapes = (
            got_matrix.shape, got_energy.shape, got_singular.shape)
        barrier("pt-profile-after-slab")
        if rank == 0:
            path.unlink(missing_ok=True)

    kgrid = (5, 5, 5)
    nk, nb, na = int(np.prod(kgrid)), 64, 16
    matrix_k = NamedSharding(mesh, P(None, "x", "y"))
    component_k = NamedSharding(mesh, P(None, None, "x", "y"))
    delta_host = np.zeros((nk, nb, nb), dtype=np.complex128)
    active = rng.normal(size=(nk, na, na)) + 1j * rng.normal(size=(nk, na, na))
    active += np.swapaxes(active.conj(), -1, -2)
    delta_host[:, :na, :na] = active
    tail_diagonal_host = rng.normal(size=(nk, nb))
    delta_host[:, np.arange(na, nb), np.arange(na, nb)] = (
        tail_diagonal_host[:, na:nb])
    dft_active_host = rng.normal(size=(nk, na))
    hamiltonian_active_host = active.copy()
    hamiltonian_active_host[
        :, np.arange(na), np.arange(na)] += dft_active_host
    connection_host = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(
        size=(3, nk, nb, nb)
    )
    connection_host += np.swapaxes(connection_host.conj(), -1, -2)
    velocity_host = rng.normal(size=connection_host.shape) + 1j * rng.normal(
        size=connection_host.shape
    )
    unitary_host = np.stack(
        [
            np.linalg.qr(rng.normal(size=(na, na)) + 1j * rng.normal(size=(na, na)))[0]
            for _ in range(nk)
        ]
    )
    energy_host = np.sort(rng.normal(size=(nk, nb)), axis=1)
    occupation_host = np.zeros((nk, nb))
    occupation_host[:, :8] = 1.0
    hamiltonian_active = device_put_process_local(
        hamiltonian_active_host, matrix_k)
    dft_active = device_put_process_local(
        dft_active_host, NamedSharding(mesh, P(None, None)))
    vector_k = NamedSharding(mesh, P(None, ("x", "y")))
    tail_diagonal = device_put_process_local(tail_diagonal_host, vector_k)
    assemble_delta = _assemble_delta_kernel(mesh, nb)
    delta = _ready(assemble_delta(
        hamiltonian_active, dft_active, tail_diagonal))
    assembly_error = float(np.max(np.abs(_host(delta) - delta_host)))
    if assembly_error != 0.0:
        raise AssertionError(
            f"fused active-head manifold assembly mismatch: {assembly_error}")
    connection = device_put_process_local(connection_host, component_k)
    velocity = device_put_process_local(velocity_host, component_k)
    unitary = device_put_process_local(unitary_host, matrix_k)
    energy = device_put_process_local(energy_host, vector_k)
    occupation = device_put_process_local(occupation_host, vector_k)

    def pipeline():
        correction = covariant_structured_delta(
            delta,
            connection,
            U_active=unitary,
            mesh=mesh,
            kgrid=kgrid,
            bvec_cart=np.eye(3),
        )
        velocity_qp = rotate_velocity_active_to_qp(
            velocity + correction, unitary, mesh=mesh
        )
        return head_s_tensor_sharded(
            velocity_qp,
            energy,
            occupation,
            [0.0j, 0.5j],
            mesh=mesh,
            nb_logical=nb,
            cell_volume=100.0,
            nk_tot=nk,
            nspin=1,
            nspinor=1,
        )

    start = time.perf_counter()
    result = _ready(pipeline())
    pipeline_cold_s = time.perf_counter() - start
    pipeline_warm_s = []
    for _ in range(5):
        start = time.perf_counter()
        result = _ready(pipeline())
        pipeline_warm_s.append(time.perf_counter() - start)

    partial = jnp.zeros_like(connection)
    lowers = (
        (
            "assembly",
            assemble_delta,
            (hamiltonian_active, dft_active, tail_diagonal),
        ),
        (
            "structured",
            _structured_delta_kernel(mesh, na),
            (delta, connection, partial),
        ),
        ("rotation", _active_rotation_kernel(mesh, na), (velocity, unitary)),
        (
            "spectral",
            _spectral_kernel(mesh, kgrid),
            (delta, jnp.asarray(_cartesian_fft_multipliers(kgrid, np.eye(3)))),
        ),
    )
    remat = {}
    assembly_collectives = {}
    for name, kernel, args in lowers:
        hlo = kernel.lower(*args).compiler_ir("hlo").as_hlo_text().lower()
        remat[name] = hlo.count("remat")
        if name == "assembly":
            assembly_collectives = {
                op: hlo.count(op)
                for op in (
                    "all-gather", "all-to-all", "collective-permute",
                    "reduce-scatter", "all-reduce")
            }
    if any(remat.values()):
        raise AssertionError(f"explicit rematerialization in HLO: {remat}")
    if rank == 0:
        print(
            "PT_PROFILE polar",
            {
                "plan_s": plan_s,
                "cold_s": cold_s,
                "warm_s": warm_s,
                "backend": polar.backend,
                "link_rel": link_error,
                "singular_rel": singular_error,
                "fused_cold_s": fused_cold_s,
                "fused_warm_s": fused_warm_s,
                "fused_link_rel": fused_link_error,
                "fused_singular_rel": fused_singular_error,
            },
        )
        print("PT_PROFILE slabio", slab_shapes if slab_shapes else "SKIPPED")
        print(
            "PT_PROFILE qsgw_head",
            {
                "cold_s": pipeline_cold_s,
                "warm_s": pipeline_warm_s,
                "remat": remat,
                "assembly_max_abs": assembly_error,
                "assembly_collectives": assembly_collectives,
                "max_result": float(np.max(np.abs(_host(result)))),
            },
        )
        print("PT_PROFILE PASS", flush=True)
    return 0


if __name__ == "__main__":
    finalize_process(main())
