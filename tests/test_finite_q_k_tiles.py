"""The finite-q dipole preparation reads its full-zone k in tiles, not one by one.

Compute-node script for P4 on one node (``srun -n 4``; ``SCRATCH`` set, as for
``tests/hsuite``; files go to the node's ``$TMPDIR``); the pytest entry skips
without the launcher.

``psp.get_dipole_mtxels --with-finite-q`` (default ``--iq-list``: every stored
q) on the hsuite H2- fixture (25 full-zone k over 9 stored) for the
two-component deck (``gnppm.in``) and the four-component one
(``bisp_sc.in``), each with ``finite_q_k_tile`` forced to 1 (one
process-local WFN read per k, the route before 2026-11-22), forced to 7 (a
tile that does not divide 25: 4 reads) and as written (one read for the 25
k).  Every dataset of ``dipole.h5`` (the finite-q rho, v and alpha blocks
among them) must equal the one-k run's bit for bit, with 24 and 21 reads
fewer.  Also: the default ``--iq-list`` is the stored q (it ran to the
full-zone count and raised IndexError before), and an index past them
refuses by name.
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
    walls, problems = chain.run_chain(root / "chain", cache_dir=None,
                                      only=["kmeans", "kin_ion", "kin_ion_bisp"])
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

    def arm(tag, deck, tile, extra=()):
        where = root / tag
        if rank == 0:
            shutil.copytree(run, where, ignore=shutil.ignore_patterns("tmp", "*.log", "dipole.h5",
                                                                     "parallel_transport.h5"))
        barrier(f"finite_q_tiles.{tag}")
        saved = getattr(gd, "finite_q_k_tile", None)
        if tile is not None and saved is not None:
            gd.finite_q_k_tile = lambda *a, **k: tile
        n0, cwd = len(reads), os.getcwd()
        try:
            os.chdir(where)
            rc = gd.main(["-i", deck, "--with-finite-q", *extra])
            assert not rc, rc
        finally:
            os.chdir(cwd)
            if saved is not None:
                gd.finite_q_k_tile = saved
        barrier(f"finite_q_tiles.{tag}.done")
        return where, reads[n0:]

    def members(path):
        got = {}
        with h5py.File(path, "r") as f:
            f.visititems(lambda k, o: got.__setitem__(k, o[()]) if isinstance(o, h5py.Dataset) else None)
        return got

    summary = []
    for deck in ("gnppm.in", "bisp_sc.in"):
        name = deck.split(".")[0]
        ref, r_ref = arm(f"{name}_t1", deck, 1)
        A = members(ref / "dipole.h5")
        fq = sorted(k for k in A if k.startswith("finite_q/"))
        assert "finite_q/rho_cvkq" in fq and "finite_q/v_cvkq" in fq, sorted(A)
        assert A["finite_q/iq_list"].tolist() == list(range(9)), A["finite_q/iq_list"]
        for tile, fewer in ((7, 21), (None, 24)):
            where, r = arm(f"{name}_t{tile or 'default'}", deck, tile)
            B = members(where / "dipole.h5")
            assert sorted(A) == sorted(B), (deck, set(A) ^ set(B))
            bad = [k for k in A if not np.array_equal(np.asarray(A[k]), np.asarray(B[k]))]
            assert not bad, f"{deck} tile {tile}: {bad} differ from one k per read"
            assert len(r_ref) - len(r) == fewer, (deck, tile, r_ref, r)
            summary.append(f"{name} tile {tile or 'default'}: {len(A)} datasets bitwise, "
                           f"reads {len(r_ref)} -> {len(r)}")
    # A small device budget: admission hands the tile exactly one k window of
    # room, so the preparation reads one k per call (the result is the same).
    from lxkit import memory as lxmem
    orig_admit = lxmem.admit

    def tight_admit(stage, routes, room_bytes, **kw):
        if stage == "dipole.finite_q_k_tile":
            room_bytes = min(p.bytes() for _, p, *_ in routes)
        return orig_admit(stage, routes, room_bytes, **kw)
    lxmem.admit = tight_admit
    try:
        where, r = arm("gnppm_tight", "gnppm.in", None)
    finally:
        lxmem.admit = orig_admit
    ref = root / "gnppm_t1"
    A, B = members(ref / "dipole.h5"), members(where / "dipole.h5")
    assert all(np.array_equal(np.asarray(A[k]), np.asarray(B[k])) for k in A), "tight budget"
    assert r.count(1) >= 25, r
    summary.append(f"gnppm tight budget: one k per read ({len(r)} reads), bitwise")
    WfnLoader.load_process_local = orig_load
    try:
        gd.finite_q_iq_list([9], 9)
        raise AssertionError("an iq past the stored q did not refuse")
    except ValueError as exc:
        assert "dipole_finite_q_iq_list" in str(exc), exc
    print(f"FINITE_Q_TILES PASS rank={rank} " + "; ".join(summary), flush=True)
    barrier("finite_q_tiles.end")
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip("run directly under the configured MPI launcher")


if __name__ == "__main__":
    main()
