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


def test_second_process_miss_fails():
    ranks = [{"rank": 0, "bench_warm": {"cache_stats": {"enabled": True, "probes": 12,
                                                         "hits": 9, "compiles": 3}}}]
    assert pc.judge_caches({}, ranks)["status"] == "FAIL"
