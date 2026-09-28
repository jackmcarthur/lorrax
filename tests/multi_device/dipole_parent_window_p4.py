"""P4: the collective dipole parent-window read equals direct h5py slices.

``file_io.restart_bundle.read_dipole_parent_window`` on the PHDF5 FFI tier, four
processes on a 2x2 mesh: unsorted parents, band window from 1, nb = 3 (rank 3
reads no rows) and nb = 5 (rank 2 one row, rank 3 none).  Everything outside
the parent rows and the window is NaN, so a wrong row or a pad row fails.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import h5py
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from file_io.restart_bundle import read_dipole_parent_window
    from file_io.slab_io import SlabIO
    from ffi.common.ffi_loader import loaded_lib_path

    assert jax.process_count() == 4
    mesh = RUNTIME.mesh
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = args.output.parent / "dipole_parent_window.h5"
    nk, nb_file, b0, rows = 6, 8, 1, [5, 0, 3]
    rng = np.random.default_rng(20260928)
    velocity = np.full((3, nk, nb_file, nb_file), np.nan + 1j * np.nan)
    for r in rows:
        velocity[:, r] = (rng.standard_normal((3, nb_file, nb_file))
                          + 1j * rng.standard_normal((3, nb_file, nb_file)))
    sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    values = jax.make_array_from_callback(
        velocity.shape, sharding, lambda index: velocity[index])
    with SlabIO(str(path), mode="w", mesh=mesh) as io:
        io.write_slab("dipole_cart", values)

    report = {}
    for nb in (3, 5):
        # the band window is NaN-free only inside [b0, b0+nb) on the parents
        got = read_dipole_parent_window(path, rows, b0, b0 + nb,
                                        nk_full=nk, mesh=mesh)
        with h5py.File(path, "r") as h5:
            want = np.stack([h5["dipole_cart"][:, r, b0:b0 + nb, b0:b0 + nb]
                             for r in rows])
        assert got.dtype == np.complex128 and got.shape == want.shape
        assert np.isfinite(want).all()
        np.testing.assert_array_equal(got, want)
        report[f"nb{nb}"] = "exact"

    if jax.process_index() == 0:
        args.output.write_text(json.dumps({
            "status": "PASS", "job": os.environ.get("SLURM_JOB_ID"),
            "step": os.environ.get("SLURM_STEP_ID"),
            "source_root": str(Path(__file__).resolve().parents[2]),
            "provider": loaded_lib_path("CUDA"), "cases": report,
            "scope": "P4 PHDF5 read_dipole_parent_window vs h5py slices; "
                     "unsorted parents, padded band rows, ranks with no rows",
        }, indent=2) + "\n")
    print(f"rank {jax.process_index()}: dipole parent window PASS", flush=True)


if __name__ == "__main__":
    from runtime import initialize_communicator_stack, run_main_and_finalize
    RUNTIME = initialize_communicator_stack(platform="gpu")
    run_main_and_finalize(main)
