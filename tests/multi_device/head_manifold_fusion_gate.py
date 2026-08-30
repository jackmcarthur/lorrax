"""P4 parity, sharding, and HLO gate for fused QSGW-head assembly."""

from __future__ import annotations

from functools import partial

from runtime import finalize_process, initialize_communicator_stack

RUNTIME = initialize_communicator_stack()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from common.collectives import (  # noqa: E402
    device_put_process_local,
    process_count,
    process_rank,
    resolve_mesh,
)
from gw.qsgw_head import _assemble_delta_kernel  # noqa: E402


def _host(value):
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(value, tiled=True))


def _collective_counts(hlo: str) -> dict[str, int]:
    return {
        op: hlo.count(op)
        for op in (
            "all-gather",
            "all-to-all",
            "collective-permute",
            "reduce-scatter",
            "all-reduce",
        )
    }


def main() -> int:
    if process_count() != 4:
        raise RuntimeError(
            f"head_manifold_fusion_gate requires four ranks; got {process_count()}")
    rank = process_rank()
    mesh = resolve_mesh()
    if tuple(mesh.shape.values()) != (2, 2):
        raise RuntimeError(f"expected production 2x2 mesh, got {mesh.shape}")

    rng = np.random.default_rng(260830)
    nk, na, nb_storage = 8, 16, 32
    raw = (rng.normal(size=(nk, na, na))
           + 1j * rng.normal(size=(nk, na, na)))
    delta_active_host = 0.5 * (raw + np.swapaxes(raw.conj(), -1, -2))
    dft_energies_host = rng.normal(size=(nk, na))
    hamiltonian_host = delta_active_host.copy()
    hamiltonian_host[:, np.arange(na), np.arange(na)] += dft_energies_host
    tail_host = rng.normal(size=(nk, nb_storage))

    matrix_sharding = NamedSharding(mesh, P(None, "x", "y"))
    vector_sharding = NamedSharding(mesh, P(None, ("x", "y")))
    replicated_vector = NamedSharding(mesh, P(None, None))
    hamiltonian = device_put_process_local(hamiltonian_host, matrix_sharding)
    delta_active = device_put_process_local(delta_active_host, matrix_sharding)
    dft_energies = device_put_process_local(
        dft_energies_host, replicated_vector)
    tail = device_put_process_local(tail_host, vector_sharding)

    candidate = _assemble_delta_kernel(mesh, nb_storage)
    out = candidate(hamiltonian, dft_energies, tail)
    jax.block_until_ready(out)
    if tuple(out.sharding.spec) != (None, "x", "y"):
        raise AssertionError(
            f"fused head output lost P(None,x,y): {out.sharding.spec}")

    expected = np.zeros((nk, nb_storage, nb_storage), dtype=np.complex128)
    expected[:, :na, :na] = delta_active_host
    expected[:, np.arange(na, nb_storage), np.arange(na, nb_storage)] = (
        tail_host[:, na:nb_storage])
    max_abs = float(np.max(np.abs(_host(out) - expected)))
    if max_abs != 0.0:
        raise AssertionError(f"fused head assembly mismatch: {max_abs}")

    @partial(jax.jit, out_shardings=matrix_sharding)
    def incumbent(delta_active_arg, tail_arg):
        delta = jnp.zeros(
            (nk, nb_storage, nb_storage), dtype=jnp.complex128)
        delta = delta.at[:, :na, :na].set(delta_active_arg)
        idx = jnp.arange(na, nb_storage)
        return delta.at[:, idx, idx].set(tail_arg[:, na:nb_storage])

    reference = incumbent(delta_active, tail)
    jax.block_until_ready(reference)
    reference_max_abs = float(np.max(np.abs(_host(reference) - expected)))
    if reference_max_abs != 0.0:
        raise AssertionError(f"incumbent assembly oracle mismatch: {reference_max_abs}")

    candidate_hlo = (
        candidate.lower(hamiltonian, dft_energies, tail)
        .compiler_ir("hlo").as_hlo_text().lower())
    incumbent_hlo = (
        incumbent.lower(delta_active, tail)
        .compiler_ir("hlo").as_hlo_text().lower())
    candidate_collectives = _collective_counts(candidate_hlo)
    incumbent_collectives = _collective_counts(incumbent_hlo)
    if candidate_collectives != incumbent_collectives:
        raise AssertionError(
            "fused assembly changed the incumbent collective family: "
            f"{candidate_collectives}/{incumbent_collectives}")
    if candidate_hlo.count("remat") != 0:
        raise AssertionError("fused head assembly HLO contains remat")

    if rank == 0:
        print(
            "HEAD_MANIFOLD_FUSION_RECEIPT",
            {
                "max_abs": max_abs,
                "reference_max_abs": reference_max_abs,
                "output_spec": str(out.sharding.spec),
                "candidate_collectives": candidate_collectives,
                "incumbent_collectives": incumbent_collectives,
                "candidate_remat": candidate_hlo.count("remat"),
            },
            flush=True,
        )
        print("HEAD_MANIFOLD_FUSION_PASS", flush=True)
    return 0


if __name__ == "__main__":
    finalize_process(main())
