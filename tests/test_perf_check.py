"""Pure judgement rules of scripts/perf_check.py (no GPU, no Slurm)."""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "perf_check", Path(__file__).resolve().parents[1] / "scripts" / "perf_check.py")
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)


def _fatbin(entries):
    body = b""
    for kind, arch in entries:
        payload = b"\0" * 16
        head = struct.pack("<HHIQ", kind, 0x101, 64, len(payload)) + b"\0" * 12
        head += struct.pack("<I", arch) + b"\0" * (64 - 32)
        body += head + payload
    return struct.pack("<IHHQ", pc.FATBIN_MAGIC, 1, 16, len(body)) + body


def test_fatbin_kinds_and_arches():
    assert pc.parse_fatbin(_fatbin([(2, 80), (2, 90), (1, 90)])) == [
        ("sass", 80), ("sass", 90), ("ptx", 90)]


def _rank(rank, mask, omp, phys_of=lambda c: c % 64, host="nid1"):
    cores = {phys_of(c) for c in mask}
    return {"rank": rank, "binding": {
        "host": host, "mask": list(mask), "logical": len(mask), "physical": len(cores),
        "numa": [0], "OMP_NUM_THREADS": omp, "OPENBLAS_NUM_THREADS": None,
        "MKL_NUM_THREADS": None}, "bench_cold": {}}


def test_binding_clone_shape_passes():
    # -c 16 --threads-per-core=1, OMP_NUM_THREADS=14: 16 distinct cores per rank
    ranks = [_rank(r, range(16 * r, 16 * r + 16), "14") for r in range(4)]
    j = pc.judge_binding(ranks)
    assert not j["fails"] and not j["warns"], j


def test_binding_docs_line_without_omp_is_flagged():
    # the docs srun line: 32 CPUs = 16 cores + their SMT siblings, OMP unset
    ranks = [_rank(r, [*range(16 * r, 16 * r + 16), *range(64 + 16 * r, 64 + 16 * r + 16)], None)
             for r in range(4)]
    j = pc.judge_binding(ranks)
    flagged = j["fails"] if pc.SMT_TEAM_SEVERITY == "FAIL" else j["warns"]
    assert any(f.startswith("ranks 0,1,2,3: oversubscription: a 32-thread OpenMP team on 16 physical")
               for f in flagged), j


def test_binding_overlap_and_omp_256_fail():
    ranks = [_rank(r, range(128), "256") for r in range(2)]
    j = pc.judge_binding(ranks)
    assert any("share CPUs" in f for f in j["fails"])
    assert any("256-thread OpenMP team on 128 CPUs" in f for f in j["fails"])


def test_reference_threshold():
    ref = {"compile_cold_s": 20.0, "gemm_tflops": 18.0}
    assert pc.vs_ref("compile_cold_s", 22.9, ref)[0] == "PASS"
    assert pc.vs_ref("compile_cold_s", 23.5, ref)[0] == "FAIL"
    assert pc.vs_ref("gemm_tflops", 15.0, ref, higher_better=True)[0] == "FAIL"
    assert pc.vs_ref("fft_ms", 3.0, ref)[0] == "INFO"


def test_slow_cache_root_and_launch_fixes():
    # a cache tree on a DVS home: FAIL naming the knob; Lustre scratch: no line
    slow = {"cubin": {"path": "/home/u/.cache/lorrax/kconv_mathdx", "fstype": "dvs", "mount": "/home",
                      "open": {"n": 5, "median_ms": 9.0}, "miss": {"n": 20, "total_s": 0.06}}}
    fast = {"cubin": {**slow["cubin"], "fstype": "lustre", "open": {"n": 5, "median_ms": 0.2},
                      "miss": {"n": 20, "total_s": 0.002}}}
    st, lines = pc.judge_cache_latency([fast, slow])
    assert st == "FAIL" and "LORRAX_CACHE_ROOT" in lines[0] and "/home/u/.cache/lorrax" in lines[0], lines
    assert pc.judge_cache_latency([fast])[0] == "PASS"
    # a slow tree launched without lorrax-run on an image install: the fix is lorrax-run
    assert "launch with lorrax-run" in pc.stage_fix({"lorrax_run": "/p/bin/lorrax-run"})
    assert "shared copy" in pc.stage_fix({"image_root": "/tmp/lorrax-u/rt-x", "image_fs": {"fstype": "lustre"}})


def test_second_process_miss_fails():
    ranks = [{"rank": 0, "bench_warm": {"cache_stats": {"enabled": True, "probes": 12,
                                                         "hits": 9, "compiles": 3}}}]
    assert pc.judge_caches({}, ranks)["status"] == "FAIL"


def test_nvrtc_bands_and_private_probe_cache(tmp_path, monkeypatch):
    # node-to-node NVRTC spread on fast installs (7.26 s vs 5.83 s, 1.25x) passes; the module
    # on DVS (16.5 s, 2.83x; 32.6 s, 5.6x) fails; in between warns
    ref = {"nvrtc_s": 5.83}
    assert pc.vs_ref("nvrtc_s", 7.26, ref)[0] == "PASS"
    assert pc.vs_ref("nvrtc_s", 10.0, ref)[0] == "WARN"
    assert pc.vs_ref("nvrtc_s", 16.5, ref)[0] == "FAIL"
    # the probe's image cache is private even when the user exports LORRAX_CACHE_ROOT
    monkeypatch.setenv("LORRAX_CACHE_ROOT", "/users/store")
    env = pc.probe_env(tmp_path, 0)
    assert env["LORRAX_CACHE_ROOT"].startswith(str(tmp_path))
    assert env["SCRATCH"].startswith(str(tmp_path))

def test_context_schedulers_and_workstation():
    none = lambda c: None                                   # no scheduler command on PATH
    ws = pc.context({}, none, "GPUtop")
    assert ws["in_step"] and ws["workstation"] and ws["nranks"] == 1
    mpi = pc.context({"OMPI_COMM_WORLD_RANK": "1", "OMPI_COMM_WORLD_SIZE": "2", "PMIX_NAMESPACE": "7"},
                     none, "GPUtop")
    assert mpi["in_step"] and (mpi["rank"], mpi["nranks"], mpi["launcher"]) == (1, 2, "ompi")
    assert pc.context({}, none, "login21")["in_step"] is False          # login host
    assert pc.context({}, lambda c: "/usr/bin/" + c, "nid001")["in_step"] is False   # scheduler, no job
    assert pc.context({"PBS_JOBID": "1.pbs"}, none, "x3001c0s1b0n0")["in_step"]
    assert not pc.context({"LSB_JOBID": "5"}, none, "batch1")["in_step"]           # launch node
    assert pc.context({"LSB_JOBID": "5", "JSM_NAMESPACE_RANK": "0"}, none, "h01")["in_step"]
    assert pc.context({"FLUX_JOB_ID": "f1", "FLUX_TASK_RANK": "0"}, none, "n1")["in_step"]
    sl = {"SLURM_JOB_ID": "9"}
    assert not pc.context(sl, none, "nid1")["in_step"]
    assert pc.context({**sl, "SLURM_STEP_ID": "0", "SLURM_PROCID": "0"}, none, "nid1")["launch_id"] == "9.0"


def _open(groups):
    stats = {g: {"n": n, "median_ms": ms} for g, (ms, n) in groups.items()}
    allv = {"n": sum(n for _, n in groups.values()),
            "median_ms": sorted(ms for ms, n in groups.values() for _ in range(n))[
                sum(n for _, n in groups.values()) // 2]}
    return {"passes": [{"all": allv, **stats}, {"all": allv, **stats}],
            "groups": {g: {"example": f"/x/{g}/f", "fstype": "9p" if g == "source" else "ext4",
                           "mount": "/mnt/c" if g == "source" else "/"} for g in groups}}


def test_filesystem_per_group_and_ld_warn():
    # W: the source tree on 9p (1.47 ms) with the venv on ext4: the overall median stays fast
    o = _open({"site-packages": (0.002, 120), "cuda headers": (0.002, 120), "source": (1.47, 120)})
    c = pc.judge_filesystem({}, [o], [], {}, [])
    assert c["status"] == "FAIL" and "the source files take 1.47 ms" in c["why"][0] and "/mnt/c" in c["why"][0]
    # an LD_LIBRARY_PATH entry at 0.64 ms per failed lookup (9p) warns
    lk = {"LD_LIBRARY_PATH": [{"dir": "/mnt/c/lib", "n": 20, "total_s": 0.0128, "fstype": "9p"}]}
    st, lines = pc.judge_lookups([lk])
    assert st == "WARN" and "/mnt/c/lib" in lines[0]


def test_verdict_quotes_the_failing_cache_line():
    # W: 2 caches FAILed on the store probe, but the VERDICT quoted the passing latency line
    fast = {"cubin": {"path": "/c/kconv_mathdx", "fstype": "ext4", "mount": "/",
                      "open": {"n": 5, "median_ms": 0.001}, "miss": {"n": 20, "total_s": 0.00002}}}
    ranks = [{"rank": 0, "cache_latency": fast, "store_probe": {"error": "FfiLibraryNotBuilt: x"}}]
    c = pc.judge_caches({}, ranks)
    assert c["status"] == "FAIL" and not c["lines"][0].startswith("cubin store probe failed")
    text = pc.report("FAIL", [c], {"gpu": True}, {"host": "h", "nranks": 1}, 1.0)
    assert "2 caches: cubin store probe failed" in text.split("VERDICT")[1], text
