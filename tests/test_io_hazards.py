"""The epsmat export opens its bundle, and bse_w_exact writes its output once.

Compute-node script for P4 on one node (``srun -n 4``; ``SCRATCH`` set, as for
``tests/hsuite``; files go to the node's ``$TMPDIR``); the pytest entry skips
without the launcher.

The H2- chain of ``tests/hsuite`` up to its GN-PPM one-shot (kmeans, kin_ion,
dipole, gnppm: a bundle with a ready ``W0_qmunu``) is built once, then:

(a) ``gw.epsmat_cli mode = export`` writes ``eps0mat.h5`` and ``epsmat.h5``.
    Before 2026-11-22 ``gw.epsmat_export.open_export_source`` read V, the
    stored q and the heads from an h5py handle its ``with`` block had already
    closed (b6c73a78f put the ``if w0_ready:`` gate after the block), so every
    export and projection raised ``KeyError``.
(b) The same open on a copy of the bundle whose ``W0_qmunu`` is not ready
    (the projection's ``require_w0 = False`` path) returns ``w0_ready = False``
    and the stored q set; before, ``q_stored`` was unbound.
(c) ``mode = project`` on the files of (a) gives back LORRAX's own W(0) at a
    stored q reached without a symmetry image (``rel_Wc_bgw_vs_lorrax`` ≤ 1e-10).
(d) ``bse.bse_w_exact`` (column mode) opens its output for writing on one
    rank only; before, every rank truncated and wrote the same path.
(e) the same with the output in an unwritable directory: rank 0's failure
    reaches every rank (``rank0_transaction``), and all raise within 2 min.
"""


def main():
    import json
    import os
    import shutil
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    root = Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"lorrax_io_hazards_{os.environ.get('SLURM_JOB_ID', '0')}_"
        f"{os.environ.get('SLURM_STEP_ID', '0')}")
    from tests.hsuite import chain, rank_session
    walls, problems = chain.run_chain(root / "chain", cache_dir=None,
                                      only=["kmeans", "kin_ion", "dipole", "gnppm"])
    assert not problems, problems
    run = root / "chain" / "run"
    import runtime
    stack = runtime.initialize_communicator_stack()
    mesh = stack.mesh
    assert mesh.size == 4, mesh
    import h5py
    from common.collectives import barrier
    from gw import epsmat_cli
    from gw.epsmat_config import EpsmatConfig
    from gw.epsmat_export import open_export_source

    def deck(name, **kw):
        path = root / f"{name}.in"
        if rank == 0:
            path.write_text("[epsmat]\n" + "".join(f"{k} = {v}\n" for k, v in kw.items()))
        barrier(f"io_hazards.deck.{name}")
        return path

    # (a) export
    eps = root / "eps"
    if rank == 0:
        eps.mkdir(parents=True, exist_ok=True)
    d = deck("export", mode="export", source_restart=run, parent_input_file=run / "gnppm.in",
             output_dir=eps, files="eps0mat epsmat")
    assert epsmat_cli.main(["-i", str(d)]) == 0
    with open(eps / "epsmat_export.json") as f:
        rx = json.load(f)["result"]
    for name in ("eps0mat", "epsmat"):
        worst = max(q["projection_err"] for q in rx[name]["q"])
        assert worst <= 1e-10, (name, worst)
        with h5py.File(eps / f"{name}.h5", "r") as f:
            assert int(f["eps_header/qpoints/qpt_done"][()].min()) == 1, name

    # (b) the open without a ready W0
    cp = root / "now0"
    if rank == 0:
        (cp / "tmp").mkdir(parents=True, exist_ok=True)
        for f in (run / "tmp").iterdir():
            if f.name.startswith("isdf_tensors_") or f.name == "zeta_q.h5":
                shutil.copy2(f, cp / "tmp" / f.name)
        with h5py.File(next((cp / "tmp").glob("isdf_tensors_*.h5")), "a") as f:
            f["W0_qmunu"].attrs["W0_ready"] = False
    barrier("io_hazards.now0")
    d = deck("now0", mode="project", source_restart=cp, parent_input_file=run / "gnppm.in",
             output_dir=root / "now0_out", bgw_reference=eps)
    src = open_export_source(EpsmatConfig.from_input_file(str(d)), mesh, require_w0=False)
    assert src.w0_ready is False and src.q_stored.shape == (9, 3) and src.n_mu > 0, src.q_stored.shape

    # (c) projection round trip
    proj = root / "proj"
    if rank == 0:
        proj.mkdir(parents=True, exist_ok=True)
    d = deck("project", mode="project", source_restart=run, parent_input_file=run / "gnppm.in",
             output_dir=proj, bgw_reference=eps)
    assert epsmat_cli.main(["-i", str(d)]) == 0
    with open(proj / "epsmat_project.json") as f:
        rows = json.load(f)["result"]["q"]
    direct = [r["static"]["rel_Wc_bgw_vs_lorrax"] for r in rows
              if r["file"] == "epsmat" and r["sym_row"] == 0]
    assert direct and max(direct) <= 1e-10, direct

    # (d) bse_w_exact writes its output on one rank
    out = root / "w_exact.h5"
    opens, orig = [], h5py.File

    class Counted(orig):
        def __init__(self, name, mode="r", *a, **kw):
            if str(name) == str(out) and mode != "r":
                opens.append(mode)
            super().__init__(name, mode, *a, **kw)
    h5py.File = Counted
    cwd = os.getcwd()
    try:
        os.chdir(run)
        from bse import bse_w_exact
        bse_w_exact.main(["-i", "gnppm.in", "--n-val", "3", "--n-cond", "2",
                          "--n-cols", "2", "--out", str(out)])
    finally:
        os.chdir(cwd)
        h5py.File = orig
    writers = rank_session.exchange(opens)
    assert sum(len(w) for w in writers) == 1, f"w_exact output opened for writing per rank: {writers}"
    with h5py.File(out, "r") as f:
        assert f["Wc"].shape[0] == 2

    # (e) rank 0's write fails (an unwritable directory): every rank raises
    # promptly instead of waiting at a barrier (BARRIER_TIMEOUT_S = 1800 s).
    import time
    ro = root / "readonly"
    if rank == 0:
        ro.mkdir(exist_ok=True)
        os.chmod(ro, 0o555)
    barrier("io_hazards.ro")
    t0, failed = time.monotonic(), None
    try:
        os.chdir(run)
        bse_w_exact.main(["-i", "gnppm.in", "--n-val", "3", "--n-cond", "2",
                          "--n-cols", "2", "--out", str(ro / "w_exact.h5")])
    except Exception as exc:                                     # noqa: BLE001
        failed = f"{type(exc).__name__}: {str(exc)[:120]}"
    finally:
        os.chdir(cwd)
    wall = time.monotonic() - t0
    outcomes = rank_session.exchange([failed, wall])
    assert all(o[0] is not None for o in outcomes), outcomes
    assert max(o[1] for o in outcomes) < 120.0, outcomes
    if rank == 0:
        os.chmod(ro, 0o755)
    print(f"IO_HAZARDS PASS rank={rank} export worst identity "
          f"{max(q['projection_err'] for n in rx for q in rx[n]['q']):.1e}; "
          f"projection direct rows {max(direct):.1e}; w_exact writers {writers}; "
          f"unwritable output: every rank raised within {max(o[1] for o in outcomes):.1f} s "
          f"({outcomes[0][0]})", flush=True)
    barrier("io_hazards.done")
    if rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip("run directly under the configured MPI launcher")


if __name__ == "__main__":
    main()
