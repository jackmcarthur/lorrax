"""The finite-q dipole preparation reads its full-zone k in tiles, not one by one.

Compute-node script for P4 on one node (``srun -n 4``; ``SCRATCH`` set, as for
``tests/hsuite``; files go to the node's ``$TMPDIR``); the pytest entry skips
without the launcher.

``psp.get_dipole_mtxels --with-finite-q`` on the hsuite H2- fixture (25
full-zone k over 9 stored) runs twice: with ``finite_q_k_tile`` forced to 1
(one process-local WFN read per k, the route before 2026-11-22) and as
written (one read for the 25 k, which fit the fixed tile).  Every dataset of
``dipole.h5`` (the finite-q rho, v and alpha blocks among them) and of
``parallel_transport.h5`` must be equal bit for bit, and the batched run must
make 24 process-local reads fewer.
"""


def main():
    import os
    import shutil
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    root = Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"lorrax_finite_q_tiles_{os.environ.get('SLURM_JOB_ID', '0')}_"
        f"{os.environ.get('SLURM_STEP_ID', '0')}")
    from tests.hsuite import chain
    walls, problems = chain.run_chain(root / "chain", cache_dir=None, only=["kmeans", "kin_ion"])
    assert not problems, problems
    run = root / "chain" / "run"
    import runtime
    stack = runtime.initialize_communicator_stack()
    assert stack.mesh.size == 4, stack.mesh
    import h5py
    import numpy as np
    from common.collectives import barrier
    from psp import get_dipole_mtxels as gd
    from wfn_loader import WfnLoader

    reads = []
    orig_load = WfnLoader.load_process_local

    def counted(self, *a, **kw):
        reads.append(len(list(kw.get("k") or [])))
        return orig_load(self, *a, **kw)
    WfnLoader.load_process_local = counted

    def arm(tag, per_k):
        where = root / tag
        if rank == 0:
            shutil.copytree(run, where, ignore=shutil.ignore_patterns("tmp", "*.log", "dipole.h5",
                                                                     "parallel_transport.h5"))
        barrier(f"finite_q_tiles.{tag}")
        saved = gd.finite_q_k_tile if hasattr(gd, "finite_q_k_tile") else None
        if per_k and saved is not None:
            gd.finite_q_k_tile = lambda *a, **k: 1
        n0, cwd = len(reads), os.getcwd()
        try:
            os.chdir(where)
            # The stored q (the default 0 .. nk_full - 1 indexes past the 9 stored rows
            # of this symmetric WFN: IndexError in kq_mapping, on main as well).
            rc = gd.main(["-i", "gnppm.in", "--with-finite-q", "--iq-list", *map(str, range(9))])
            assert not rc, rc
        finally:
            os.chdir(cwd)
            if saved is not None:
                gd.finite_q_k_tile = saved
        barrier(f"finite_q_tiles.{tag}.done")
        return where, reads[n0:]

    per_k, r_per_k = arm("per_k", True)
    tiled, r_tiled = arm("tiled", False)
    WfnLoader.load_process_local = orig_load

    def members(path):
        got = {}
        with h5py.File(path, "r") as f:
            f.visititems(lambda k, o: got.__setitem__(k, o[()]) if isinstance(o, h5py.Dataset) else None)
        return got
    n_cmp = 0
    for name in ("dipole.h5", "parallel_transport.h5"):
        if not (per_k / name).exists():
            continue
        A, B = members(per_k / name), members(tiled / name)
        assert sorted(A) == sorted(B), (name, set(A) ^ set(B))
        bad = [k for k in A if not np.array_equal(np.asarray(A[k]), np.asarray(B[k]))]
        assert not bad, f"{name}: {bad} differ between one k per read and the tile"
        n_cmp += len(A)
        if name == "dipole.h5" and rank == 0:
            print(f"FINITE_Q_TILES dipole.h5 datasets: {sorted(A)}", flush=True)
    print(f"FINITE_Q_TILES rank={rank} process-local reads: one k per read {len(r_per_k)} "
          f"(sizes {sorted(set(r_per_k))}), tiled {len(r_tiled)} (sizes {sorted(set(r_tiled))})",
          flush=True)
    assert len(r_per_k) - len(r_tiled) == 24, (r_per_k, r_tiled)
    print(f"FINITE_Q_TILES PASS rank={rank} {n_cmp} datasets bitwise equal", flush=True)
    barrier("finite_q_tiles.end")
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip("run directly under the configured MPI launcher")


if __name__ == "__main__":
    main()
