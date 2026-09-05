"""P4 deadline pins and Si byte-identity arms; launched by a step supervisor.

Run before/after in fresh processes on the same four GPUs. The baseline files
are exact ``git show BASE:path`` exports, not a second maintained code path.
The after arm delays rank 3's selector lowering and rank 0's NCCL ID by 65 s.
``supervisor`` parks all ranks in selector lowering until Slurm kills the step.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("before", "after", "supervisor"))
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()

    from runtime import initialize_communicator_stack
    rt = initialize_communicator_stack()
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from centroid import pivoted_cholesky as pc
    from distrib_la import _cusolvermp as mp
    import distrib_la as D

    rank = jax.process_index()
    assert jax.process_count() == jax.device_count() == 4
    root = Path(pc.__file__).resolve().parents[2]
    print(f"[deadline-pin] rank={rank} arm={args.arm} source={root} "
          f"sha={subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()} "
          f"step={os.environ.get('SLURM_JOB_ID')}.{os.environ.get('SLURM_STEP_ID')}",
          flush=True)
    output = evidence / args.arm
    output.mkdir(exist_ok=True)

    if args.arm == "supervisor":
        def terminated(signum, frame):
            (output / f"signal-rank{rank}.txt").write_text(str(signum))
            if rank == 0:
                print("[deadline-supervisor] rank=0 received whole-step "
                      "SIGTERM from Slurm walltime supervisor", file=sys.stderr,
                      flush=True)

        signal.signal(signal.SIGTERM, terminated)

        class Parked:
            def lower(self, *operands):
                (output / f"entered-rank{rank}.txt").write_text("lowering")
                while True:
                    time.sleep(1)

        if rank == 0:
            print("[deadline-supervisor] rank=0 policy=Slurm step walltime; "
                  "all four ranks must terminate together", flush=True)
        pc._run_select_with_progress(
            Parked(), (), n_candidates=1028, n_groups=25, point_budget=800,
            print_fn=lambda text: print(text, flush=True) if rank == 0 else None,
            start_progress=lambda: None)
        raise AssertionError("supervisor pin unexpectedly returned")

    real_select = pc._run_select_with_progress
    if args.arm == "before":
        baseline_pc = _load("centroid._deadline_baseline",
                            evidence / "baseline_pc.py")
        baseline_kv = _load("distrib_la._deadline_baseline",
                            evidence / "baseline_collectives.py")
        mp.broadcast_bytes = baseline_kv.broadcast_bytes

        def select(step, operands, **kwargs):
            return baseline_pc._run_bounded_select(
                step, operands, time_budget_s=900, **kwargs)
    else:
        def select(step, operands, **kwargs):
            class Delayed:
                def lower(self, *values):
                    if rank == 3:
                        print("[deadline-pin] selector rank=3 skew=65s "
                              "former_short_budget=30s", flush=True)
                        time.sleep(65)
                    return step.lower(*values)
            return real_select(Delayed(), operands, **kwargs)

    def recorded_select(step, operands, **kwargs):
        result = select(step, operands, **kwargs)
        for index, value in enumerate(result):
            if isinstance(value, jax.Array):
                for shard in value.addressable_shards:
                    np.save(output / f"selector-{index}-rank{rank}.npy",
                            np.asarray(shard.data))
        return result

    pc._run_select_with_progress = recorded_select
    # Exercise the production driver against private copies of the same Si WFN.
    from centroid import kmeans_cli
    # Use one private input workspace so the source-path provenance bytes
    # agree too; move each completed output into its immutable arm directory.
    os.chdir(evidence / "input")
    sys.argv = ["kmeans_cli", "336", "--seed", "42", "--orbit"]
    assert kmeans_cli.main() == 0
    if rank == 0:
        for filename in ("centroids_frac_336.txt", "kmeans.out"):
            shutil.move(filename, output / filename)
    from jax.experimental import multihost_utils
    multihost_utils.sync_global_devices("deadline-pin-centroid-saved")
    pc._run_select_with_progress = real_select

    # Force a new bootstrap namespace/mesh layout: kmeans may already have
    # created its ordinary cuSOLVERMp context during runtime preparation.
    original_broadcast = mp.broadcast_bytes
    def broadcast(buf, *, key):
        return original_broadcast(buf, key=key + "/deadline-pin/" + args.arm)
    mp.broadcast_bytes = broadcast
    if args.arm == "after":
        original_fill = mp.loader.fill_nccl_unique_id
        def delayed_fill(address):
            print("[deadline-pin] cuSOLVERMp rank=0 skew=65s "
                  "former_budget=60s", flush=True)
            time.sleep(65)
            return original_fill(address)
        mp.loader.fill_nccl_unique_id = delayed_fill
    started = time.monotonic()
    assert mp._ctx_key(rt.mesh, False) not in mp._CTX_CACHE
    mp.get_or_init_context(rt.mesh, col_major=False)
    if args.arm == "after":
        mp.loader.fill_nccl_unique_id = original_fill
    bootstrap_elapsed = time.monotonic() - started

    rng = np.random.default_rng(20260904)
    a = rng.standard_normal((2, 64, 64)) + 1j * rng.standard_normal((2, 64, 64))
    a = (a + a.conj().transpose(0, 2, 1)) / 2 + 128 * np.eye(64)
    b = rng.standard_normal((2, 64, 32)) + 1j * rng.standard_normal((2, 64, 32))
    sharding = NamedSharding(rt.mesh, P(None, "x", "y"))
    residuals = {}
    for op in ("cholesky", "solve_lu"):
        A = device_put_process_local(a, sharding)
        B = device_put_process_local(b, sharding)
        token = D.factor(op, A, rt.mesh, backend="cusolvermp")
        assert token.backend == "cusolvermp"
        x = D.solve(token, B)
        jax.block_until_ready(x)
        # Each rank checks its own tile against the dense reference and saves
        # that tile for cross-arm byte comparison; no global result gather.
        reference = np.linalg.solve(a, b)
        residuals[op] = 0.0
        for shard in x.addressable_shards:
            local = np.asarray(shard.data)
            rel = float(np.max(np.abs(local - reference[shard.index])) /
                        np.max(np.abs(reference[shard.index])))
            assert rel < 1e-12, (op, rank, rel)
            residuals[op] = max(residuals[op], rel)
            np.save(output / f"{op}-rank{rank}.npy", local)
    (output / f"complete-rank{rank}.json").write_text(json.dumps({
        "arm": args.arm, "rank": rank, "bootstrap_elapsed_s": bootstrap_elapsed,
        "relative_errors": residuals,
        "step": f"{os.environ.get('SLURM_JOB_ID')}.{os.environ.get('SLURM_STEP_ID')}",
    }, indent=2))
    print(f"[deadline-pin] COMPLETE arm={args.arm} rank={rank} "
          f"bootstrap_elapsed_s={bootstrap_elapsed:.3f} errors={residuals}",
          flush=True)


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
