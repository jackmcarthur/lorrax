"""The epsmat export and projection read and write per run, not per q.

Compute-node script for P4 on one node (``srun -n 4``; ``SCRATCH`` set, as for
``tests/hsuite``; files go to the node's ``$TMPDIR``); the pytest entry skips
without the launcher.

On the H2- chain of ``tests/hsuite`` up to its GN-PPM one-shot, every run is
made twice: once on the per-q route (every helper below replaced by its
one-q-per-call form, as before 2026-11-22) and once as written.  The outputs
must be equal bit for bit, and the batched run must make fewer calls:

(13) ``read_q_rows``: one bundle handle for the run (``held_bundle``), V and W
     of a q in one ``read_multi``; before, an open, two reads and a close per q.
(14) the projection: one SlabIO read handle per BerkeleyGW file for the run,
     and the frequencies of a q in one read (``frequency_tile``; a two-frequency
     Godby-Needs file is made from the exported static one); before, an open,
     a read and a close per (q, frequency).
(15) ``bgw_epsmat`` G lists, Coulomb rows and diagonals: one serial h5py open
     per file for all q (``read_q_gspaces``, ``write_q_gspaces``,
     ``write_q_diagonals``); before, one open per q.

Datasets and attributes are compared, on the wedge (IBZ) bundle and on a
full-zone one (``restart_q_storage = full``), for the export with and
without a reference, the static and the two-frequency projection, a
projection from a bundle whose W0 is not ready, and an export through a
library without the multi-read handler (``read_q_rows`` falls back to one
read per dataset).  ``--keep`` keeps the run root for a later step.
"""


def main():
    import os
    import shutil
    import sys
    from collections import Counter
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    root = Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"lorrax_epsmat_batched_{os.environ.get('SLURM_JOB_ID', '0')}_"
        f"{os.environ.get('SLURM_STEP_ID', '0')}")
    from tests.hsuite import chain
    walls, problems = chain.run_chain(root / "chain", cache_dir=None,
                                      only=["kmeans", "kin_ion", "dipole", "gnppm"])
    assert not problems, problems
    run = root / "chain" / "run"
    import runtime
    stack = runtime.initialize_communicator_stack()
    assert stack.mesh.size == 4, stack.mesh
    import h5py
    import numpy as np
    from common.collectives import barrier
    from file_io import bgw_epsmat as be
    from file_io import slab_io
    from gw import epsmat_cli
    from gw import epsmat_export as ee

    calls = Counter()
    orig = {"h5py.File": h5py.File, "SlabIO.__init__": slab_io.SlabIO.__init__,
            "read_slab": slab_io.SlabIO.read_slab, "read_multi": slab_io.SlabIO.read_multi}

    class CountedFile(orig["h5py.File"]):
        def __init__(self, name, mode="r", *a, **kw):
            calls[("h5py", Path(str(name)).name, mode)] += 1
            super().__init__(name, mode, *a, **kw)

    def counted_init(self, path, *a, **kw):
        calls[("slabio_open", Path(str(path)).name)] += 1
        return orig["SlabIO.__init__"](self, path, *a, **kw)

    def counted(kind):
        def call(self, *a, **kw):
            calls[(kind, Path(str(self.path)).name)] += 1
            return orig[kind](self, *a, **kw)
        return call

    h5py.File = CountedFile
    slab_io.SlabIO.__init__ = counted_init
    slab_io.SlabIO.read_slab = counted("read_slab")
    slab_io.SlabIO.read_multi = counted("read_multi")

    # The per-q route: each helper made one-q-per-call.
    @__import__("contextlib").contextmanager
    def no_hold(src, mesh):
        yield src
    per_q = {}
    if hasattr(ee, "held_bundle"):
        per_q[(ee, "held_bundle")] = no_hold
    if hasattr(ee, "frequency_tile"):
        per_q[(ee, "frequency_tile")] = lambda *a, **k: 1

    class OpenPerRead:
        """A reader that opens, reads and closes its file at every read (the old loop)."""
        def __init__(self, path, mesh):
            self.path, self.mesh = path, mesh

        def read_slab(self, *a, **kw):
            with slab_io.SlabIO(self.path, mode="r", mesh=self.mesh) as rd:
                return rd.read_slab(*a, **kw)

    @__import__("contextlib").contextmanager
    def readers_per_read(files, heads, mesh):
        yield {name: OpenPerRead(files[name], mesh) for name in heads}
    if hasattr(ee, "_bgw_readers"):
        per_q[(ee, "_bgw_readers")] = readers_per_read
    for name in ("read_q_gspaces", "write_q_gspaces", "write_q_diagonals"):
        if hasattr(be, name):
            batch = getattr(be, name)
            per_q[(be, name)] = (
                (lambda b: lambda path, items: [x for it in items for x in b(path, [it])])(batch)
                if name.startswith("read") else
                (lambda b: lambda path, items: [b(path, [it]) for it in items])(batch))

    def n(c, *key):
        return sum(v for k, v in c.items() if k[:len(key)] == key)
    nq_store_ = 9

    def run_cli(tag, route, source=None, parent=None, **kw):
        out = root / tag
        source = run if source is None else source
        parent = run / "gnppm.in" if parent is None else parent
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            (root / f"{tag}.in").write_text(
                "[epsmat]\n" + f"output_dir = {out}\nsource_restart = {source}\n"
                f"parent_input_file = {parent}\n"
                + "".join(f"{k} = {v}\n" for k, v in kw.items()))
        barrier(f"epsmat_batched.{tag}")
        saved = {key: getattr(*key) for key in per_q} if route == "per_q" else {}
        for (mod, name), fn in (per_q.items() if route == "per_q" else ()):
            setattr(mod, name, fn)
        before = Counter(calls)
        try:
            assert epsmat_cli.main(["-i", str(root / f"{tag}.in")]) == 0
        finally:
            for (mod, name), fn in saved.items():
                setattr(mod, name, fn)
        return out, calls - before

    def members(path):
        """Every dataset's value and every attribute of the file (``name@attr``)."""
        got = {}

        def visit(k, o):
            if isinstance(o, h5py.Dataset):
                got[k] = o[()]
            for a, v in o.attrs.items():
                got[f"{k}@{a}"] = v
        with orig["h5py.File"](path, "r") as f:
            for a, v in f.attrs.items():
                got[f"/@{a}"] = v
            f.visititems(visit)
        return got

    def same(a, b, keys=None):
        A, B = members(a), members(b)
        if keys is None:
            assert sorted(A) == sorted(B), (a, set(A) ^ set(B))
            keys = sorted(A)
        else:
            keys = [k for k in sorted(A) if k.split("@")[0] in keys]
        bad = [k for k in keys if not np.array_equal(np.asarray(A[k]), np.asarray(B[k]))]
        assert not bad, f"{a.name}: {bad} differ between the per-q and the batched run"
        return len(keys)

    # The reference files: one export as written.
    ref, _ = run_cli("ref", "batched", mode="export", files="eps0mat epsmat")
    # (13, 15) export with the reference's q list and G order, both routes.
    exp_s, c_es = run_cli("exp_s", "per_q", mode="export", files="eps0mat epsmat", bgw_reference=ref)
    exp_b, c_eb = run_cli("exp_b", "batched", mode="export", files="eps0mat epsmat", bgw_reference=ref)
    n_cmp = sum(same(exp_s / f"{n}.h5", exp_b / f"{n}.h5") for n in ("eps0mat", "epsmat"))

    # (14) a two-frequency (Godby-Needs) copy: eps⁻¹(iω_p) = 1 + (eps⁻¹(0) − 1)/2.
    gn = root / "gn"
    if rank == 0:
        gn.mkdir(parents=True, exist_ok=True)
        for fname in ("eps0mat", "epsmat"):
            with orig["h5py.File"](ref / f"{fname}.h5", "r") as f, \
                    orig["h5py.File"](gn / f"{fname}.h5", "w") as g:
                for k in f:
                    if k != "mats":
                        f.copy(k, g)
                g["mats/matrix-diagonal"] = f["mats/matrix-diagonal"][()]
                m0 = f["mats/matrix"][()]                        # (nq, 1, 1, n, n, 2)
                eye = np.zeros_like(m0[:, :, 0])
                for iq, nm in enumerate(np.asarray(f["eps_header/gspace/nmtx"][()]).reshape(-1)):
                    eye[iq, 0, np.arange(nm), np.arange(nm), 0] = 1.0
                g["mats/matrix"] = np.concatenate([m0, (0.5 * (m0[:, :, 0] + eye))[:, :, None]], axis=2)
                fr = g["eps_header/freqs"]
                for k in ("freq_dep", "nfreq", "freqs"):
                    del fr[k]
                fr["freq_dep"], fr["nfreq"] = np.int32(3), np.int32(2)
                fr["freqs"] = np.array([[0.0, 0.0], [0.0, 13.6]])
    barrier("epsmat_batched.gn")
    pro_s, c_ps = run_cli("pro_s", "per_q", mode="project", bgw_reference=gn)
    pro_b, c_pb = run_cli("pro_b", "batched", mode="project", bgw_reference=gn)
    bundle = next((run / "tmp").glob("isdf_tensors_*.h5")).name
    n_cmp += same(pro_s / "tmp" / bundle, pro_b / "tmp" / bundle,
                  keys=["W0_qmunu", "Wp_qmunu", "bgw_epsinv_head"])
    pkeys = ["W0_qmunu", "bgw_epsinv_head"]

    # Static only: an export without a reference, and a projection of the
    # static files (one frequency: main's reads exactly).
    s0, _ = run_cli("exn_s", "per_q", mode="export", files="eps0mat epsmat")
    b0, _ = run_cli("exn_b", "batched", mode="export", files="eps0mat epsmat")
    n_cmp += sum(same(s0 / f"{n}.h5", b0 / f"{n}.h5") for n in ("eps0mat", "epsmat"))
    s0, _ = run_cli("pr0_s", "per_q", mode="project", bgw_reference=ref)
    b0, _ = run_cli("pr0_b", "batched", mode="project", bgw_reference=ref)
    n_cmp += same(s0 / "tmp" / bundle, b0 / "tmp" / bundle, keys=pkeys)

    # A library without the multi-read handler: the held handle reads each dataset.
    def no_multi(self, *a, **kw):
        raise RuntimeError("GATE slab_io_read_multi: got: (test) no handler")
    slab_io.SlabIO.read_multi = no_multi
    try:
        nm, _ = run_cli("exp_nomulti", "batched", mode="export", files="eps0mat epsmat",
                        bgw_reference=ref)
    finally:
        slab_io.SlabIO.read_multi = counted("read_multi")
    n_cmp += sum(same(exp_s / f"{n}.h5", nm / f"{n}.h5") for n in ("eps0mat", "epsmat"))
    # The same route by its test-only switch (gw.epsmat_export.MULTI_READ).
    ee.MULTI_READ = False
    try:
        nm, c_nm = run_cli("exp_switch", "batched", mode="export", files="eps0mat epsmat",
                           bgw_reference=ref)
    finally:
        ee.MULTI_READ = True
    assert n(c_nm, "read_multi", bundle) == 0 and n(c_nm, "read_slab", bundle) == 18, c_nm
    n_cmp += sum(same(exp_s / f"{n_}.h5", nm / f"{n_}.h5") for n_ in ("eps0mat", "epsmat"))

    # A small device budget (Frontier's MI250X gives 48.99 GB per GCD, not 80): the
    # projection's admission is handed exactly the one-frequency price as its room,
    # so the two-frequency read does not fit and the projection reads one frequency
    # at a time instead of refusing; the result is the same.
    from lxkit import memory as lxmem
    orig_admit = lxmem.admit

    def tight_admit(stage, routes, room_bytes, **kw):
        if stage == "epsmat.project":
            room_bytes = min(p.bytes() for _, p, *_ in routes)
        return orig_admit(stage, routes, room_bytes, **kw)
    lxmem.admit = tight_admit
    try:
        tight, c_t = run_cli("pro_tight", "batched", mode="project", bgw_reference=gn)
    finally:
        lxmem.admit = orig_admit
    assert n(c_t, "read_slab", "epsmat.h5") == 2 * (nq_store_ - 1), c_t
    n_cmp += same(pro_s / "tmp" / bundle, tight / "tmp" / bundle,
                  keys=["W0_qmunu", "Wp_qmunu", "bgw_epsinv_head"])

    # A bundle whose W0 is not ready: the projection writes W0 from BerkeleyGW alone.
    cp = root / "now0"
    if rank == 0:
        (cp / "tmp").mkdir(parents=True, exist_ok=True)
        for f in (run / "tmp").iterdir():
            if f.name.startswith("isdf_tensors_") or f.name == "zeta_q.h5":
                shutil.copy2(f, cp / "tmp" / f.name)
        with orig["h5py.File"](cp / "tmp" / bundle, "a") as f:
            f["W0_qmunu"].attrs["W0_ready"] = False
    barrier("epsmat_batched.now0")
    s0, _ = run_cli("now0_s", "per_q", source=cp, mode="project", bgw_reference=gn)
    b0, _ = run_cli("now0_b", "batched", source=cp, mode="project", bgw_reference=gn)
    n_cmp += same(s0 / "tmp" / bundle, b0 / "tmp" / bundle,
                  keys=["W0_qmunu", "Wp_qmunu", "bgw_epsinv_head"])

    # Full-zone q storage: a fresh GN-PPM bundle on a centroid set that is not
    # orbit-closed (one point moved), which restart_q_storage = auto stores on
    # the full zone; then the export and the projection on both routes.
    full = root / "full"
    if rank == 0:
        shutil.copytree(run, full, ignore=shutil.ignore_patterns("tmp", "*.log", "*_sigma*",
                                                                "*eqp*", "*.out"))
        cfile = next(full.glob("centroids_frac_*.txt"))
        pts = np.loadtxt(cfile)
        pts[-1] = (0.37, 0.11, 0.53)                     # off every orbit of the set
        np.savetxt(cfile, pts)
    barrier("epsmat_batched.full")
    recs, _ = chain.run_stage(full, "gnppm_full", "gw.gw_jax", ["-i", "gnppm.in"],
                              chain._env(None), 900)
    assert all(r["rc"] == 0 and not r["hits"] for r in recs), recs
    fbundle = next((full / "tmp").glob("isdf_tensors_*.h5"))
    with orig["h5py.File"](fbundle, "r") as f:
        storage = str(f["V_qmunu"].attrs.get("q_storage", "full"))
        nq_full = int(f["V_qmunu"].shape[0])
    assert storage == "full" and nq_full == 25, (storage, nq_full)
    s0, _ = run_cli("fexp_s", "per_q", source=full, parent=full / "gnppm.in", mode="export",
                    files="eps0mat epsmat")
    b0, _ = run_cli("fexp_b", "batched", source=full, parent=full / "gnppm.in", mode="export",
                    files="eps0mat epsmat")
    n_cmp += sum(same(s0 / f"{n}.h5", b0 / f"{n}.h5") for n in ("eps0mat", "epsmat"))
    s0, _ = run_cli("fpro_s", "per_q", source=full, parent=full / "gnppm.in", mode="project",
                    bgw_reference=gn)
    b0, _ = run_cli("fpro_b", "batched", source=full, parent=full / "gnppm.in", mode="project",
                    bgw_reference=gn)
    n_cmp += same(s0 / "tmp" / fbundle.name, b0 / "tmp" / fbundle.name,
                  keys=["W0_qmunu", "Wp_qmunu", "bgw_epsinv_head"])

    h5py.File = orig["h5py.File"]
    slab_io.SlabIO.__init__ = orig["SlabIO.__init__"]
    slab_io.SlabIO.read_slab = orig["read_slab"]
    slab_io.SlabIO.read_multi = orig["read_multi"]

    def n(c, *key):
        return sum(v for k, v in c.items() if k[:len(key)] == key)
    nq_store = 9
    with orig["h5py.File"](gn / "epsmat.h5", "r") as f:
        nq_eps = int(f["eps_header/qpoints/nq"][()])
    report = {
        "export bundle opens": (n(c_es, "slabio_open", bundle), n(c_eb, "slabio_open", bundle)),
        "export bundle reads": (n(c_es, "read_slab", bundle) + n(c_es, "read_multi", bundle),
                                n(c_eb, "read_slab", bundle) + n(c_eb, "read_multi", bundle)),
        "export epsmat.h5 h5py opens": (n(c_es, "h5py", "epsmat.h5"), n(c_eb, "h5py", "epsmat.h5")),  # [15]
        "project bundle opens": (n(c_ps, "slabio_open", bundle), n(c_pb, "slabio_open", bundle)),
        "project epsmat.h5 SlabIO opens": (n(c_ps, "slabio_open", "epsmat.h5"),  # [14]
                                           n(c_pb, "slabio_open", "epsmat.h5")),  # [14]
        "project epsmat.h5 matrix reads": (n(c_ps, "read_slab", "epsmat.h5"),  # [14]
                                           n(c_pb, "read_slab", "epsmat.h5")),  # [14]
        "project epsmat.h5 h5py opens": (n(c_ps, "h5py", "epsmat.h5"), n(c_pb, "h5py", "epsmat.h5")),  # [15]
    }
    print(f"EPSMAT_BATCHED rank={rank} calls (per-q, batched): {report}", flush=True)
    rows_eps = nq_store - 1
    assert report["export bundle opens"][1] == 1, report
    # The projection also opens its output, a copy of the bundle with the same name.
    assert report["project bundle opens"][1] == 2, report
    assert report["project epsmat.h5 SlabIO opens"][1] == 1, report  # [14]
    assert report["project epsmat.h5 matrix reads"] == (2 * rows_eps, rows_eps), report  # [14]
    for key, (s, b) in report.items():
        assert b < s, (key, report)
    print(f"EPSMAT_BATCHED PASS rank={rank} {n_cmp} datasets bitwise equal; "
          f"{nq_eps} epsmat q, {nq_store} stored q", flush=True)
    barrier("epsmat_batched.done")
    if "--keep" in sys.argv:
        print(f"EPSMAT_BATCHED kept {root}", flush=True)
    elif rank == 0:
        shutil.rmtree(root, ignore_errors=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip("run directly under the configured MPI launcher")


if __name__ == "__main__":
    main()
