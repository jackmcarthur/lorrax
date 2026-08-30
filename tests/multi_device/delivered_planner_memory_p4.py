"""P=4 memory evidence for delivered-window pre-collective reduction.

The pole field is genuinely tiled over the production 2-D mesh.  Inputs are
allocated before ``tracemalloc`` starts, so ``python_peak_bytes`` measures the
host-side planning work rather than construction of the synthetic field.  The
reported collective payload comes from the planner's proved fixed carrier:
three float64 moment planes over ``lattice_bins**2`` cells per leading pole.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tracemalloc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "services" / "minimax" / "src"))

import jax

# This is a planner/collective measurement, not a driver or FFI gate.  Start
# exactly the JAX process mesh it needs without bootstrapping unrelated native
# services; that also makes absence of an in-tree FFI build irrelevant to the
# stated memory claim.
# ``select_gpu.sh`` exposes one device per process.  Slurm auto-detection
# otherwise reuses the local rank as a device ordinal (1/2/3 on those ranks),
# even though each process-local CUDA namespace contains only device zero.
jax.distributed.initialize(local_device_ids=[0])

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from common.collectives import (  # noqa: E402
    all_gather_processes,
    barrier,
    device_put_process_local,
    process_count,
    process_rank,
    resolve_mesh,
)
from gw.mpa import delivered_windows as delivered_module  # noqa: E402
from gw.ppm_windows import _SigmaBranch  # noqa: E402
import minimax  # noqa: E402

_measure_pole_batch = delivered_module.measure_delivered_sigma_pole_batch


LATTICE_BINS = 25
N_POLES = 2
ETA_RY = 0.02


def _branch(n_states):
    energies = jnp.linspace(0.04, 0.82, n_states, dtype=jnp.float64)[None, :]
    omega = np.linspace(0.10, 0.70, 5, dtype=np.float64)
    return _SigmaBranch(
        "synthetic near-pole conduction",
        energies,
        jnp.ones_like(energies, dtype=bool),
        "cond",
        False,
        omega,
        np.arange(omega.size, dtype=np.int64),
    )


def _field(side, mesh):
    row, col = np.meshgrid(
        np.arange(side, dtype=np.float64),
        np.arange(side, dtype=np.float64),
        indexing="ij",
    )
    # A narrow comb around external frequencies makes this a near-pole case;
    # positive widths preserve the causal lower-half-plane convention.
    energy = 0.10 + 0.60 * ((row + 3.0 * col) % 97.0) / 96.0
    width = 1.0e-4 + 4.0e-3 * ((5.0 * row + col) % 31.0) / 30.0
    omega = np.stack(
        [energy - 1.0j * width, 1.07 * energy - 1.0j * (1.4 * width)], axis=0
    ).astype(np.complex128)
    phase = 0.017 * row - 0.023 * col
    residue = np.stack(
        [1.0 + 0.2 * np.exp(1.0j * phase),
         0.7 + 0.1 * np.exp(-1.0j * phase)],
        axis=0,
    ).astype(np.complex128)
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    return (
        device_put_process_local(np.ascontiguousarray(omega), sharding),
        device_put_process_local(np.ascontiguousarray(residue), sharding),
    )


def _measure(label, n_states, side, mesh):
    branch = _branch(n_states)
    amplitude = np.linspace(0.4, 1.6, n_states, dtype=np.float64)[None, :]
    omega, residue = _field(side, mesh)
    jax.block_until_ready((omega, residue))
    barrier(f"delivered_planner_memory_{label}_ready")

    tracemalloc.start()
    tracemalloc.reset_peak()
    result = _measure_pole_batch(
        branch, omega, residue, regularization_width_ry=ETA_RY,
        pole_split_ry=0.8, state_amplitude=amplitude,
        lattice_bins=LATTICE_BINS, mesh_xy=mesh)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del current_bytes

    raw_count = int(result[-2])
    evidence = result[-1]
    peaks = np.asarray(
        all_gather_processes(np.asarray(peak_bytes, dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)
    row = {
        "label": label,
        "states": int(n_states),
        "spatial_shape": [int(side), int(side)],
        "leading_poles": N_POLES,
        "global_live_spatial_poles": int(raw_count // n_states),
        "state_x_spatial_poles": int(raw_count),
        "python_peak_bytes_by_rank": peaks.tolist(),
        "python_peak_bytes_max": int(peaks.max()),
        "collective_cells_per_pole_per_rank": int(
            evidence["collective_spatial_cell_ceiling_per_pole_interval"]
        ),
        "collective_payload_bytes_per_pole_per_rank": int(
            evidence["collective_payload_bytes_per_pole_per_rank"]
        ),
        "collective_payload_bytes_all_poles_per_rank": int(
            N_POLES * evidence["collective_payload_bytes_per_pole_per_rank"]
        ),
        "collective_ceiling_independent_of_state_count": bool(
            evidence["collective_ceiling_independent_of_state_count"]
        ),
        "collective_ceiling_independent_of_spatial_extent": bool(
            evidence["collective_ceiling_independent_of_spatial_extent"]
        ),
    }
    barrier(f"delivered_planner_memory_{label}_done")
    return row


def main():
    if process_count() != 4:
        raise RuntimeError(
            f"delivered planner memory gate requires P=4, got P={process_count()}"
        )
    mesh = resolve_mesh()
    if tuple(mesh.shape.values()) != (2, 2):
        raise RuntimeError(f"expected the production 2x2 mesh, got {mesh.shape}")

    rows = [
        _measure("small", n_states=8, side=48, mesh=mesh),
        _measure("large", n_states=32, side=384, mesh=mesh),
    ]
    if rows[0]["collective_payload_bytes_all_poles_per_rank"] != rows[1][
            "collective_payload_bytes_all_poles_per_rank"]:
        raise AssertionError("bounded planning collective grew with problem size")
    if rows[1]["state_x_spatial_poles"] <= 100 * rows[0][
            "state_x_spatial_poles"]:
        raise AssertionError("the two evidence sizes do not separate sufficiently")

    if process_rank() == 0:
        receipt = {
            "world_size": process_count(),
            "mesh": dict(mesh.shape),
            "jax_version": jax.__version__,
            "jaxlib_version": jax.lib.__version__,
            "delivered_module": delivered_module.__file__,
            "minimax_module": minimax.__file__,
            "lattice_bins": LATTICE_BINS,
            "rows": rows,
            "verdict": "PASS_FIXED_PRECOLLECTIVE_CEILING",
        }
        output = os.environ.get("DELIVERED_PLANNER_MEMORY_RECEIPT")
        if output:
            with open(output, "w", encoding="utf-8") as handle:
                json.dump(receipt, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print("DELIVERED_PLANNER_MEMORY", json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        jax.distributed.shutdown()
