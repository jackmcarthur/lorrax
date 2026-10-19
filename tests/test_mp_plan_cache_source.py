"""cuSOLVERMp descriptors and bufferSize queries are made once per shape, never per call.

cuSOLVERMp 0.9.1's bufferSize builds a cuBLASMp grid (cublasMpGridCreate -> ncclCommSplit ->
ncclCommInitRank) for the descriptors it is given: a query per call was a communicator per call,
and a rank deadlocked in libfabric's memory hooks inside such a split (run30k, 2026-10-19, gdb on
all 16 ranks). Every descriptor creation and bufferSize call in the cuSOLVERMp FFIs must sit in a
plan builder: lu_plan / getrs_chunking (batched_solve_lu_ffi.cc) or an mp_plan(...) callback
(mp_plan_cache.h), each of which runs once per (context, shape)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "ffi" / "cpp" / "cusolvermp"
CALL = re.compile(r"cusolverMpCreateMatrixDesc\(|BufferSize<[^>]*>\(|cusolverMp\w+_bufferSize\(")
FUNC = re.compile(r"^(?:static |inline |extern \"C\" )[^\n;]*?\b(\w+)\(", re.M)
PLAN_FUNCS = {"lu_plan", "getrs_chunking", "mp_plan"}
INTERFACE = {"cusolvermp_interface.h", "mp_plan_cache.h"}   # the vendor shims and the cache itself


def offenders(text):
    funcs = [(m.start(), m.group(1)) for m in FUNC.finditer(text)]
    bad = []
    for m in CALL.finditer(text):
        owner = max((f for f in funcs if f[0] < m.start()), default=(0, "?"))
        body = text[owner[0]:m.start()]
        if owner[1] in PLAN_FUNCS or "mp_plan(MpPlanKey" in body:
            continue
        bad.append(f"{owner[1]}: line {text.count(chr(10), 0, m.start()) + 1}")
    return bad


@pytest.mark.parametrize("path", sorted(p for p in ROOT.glob("*.[ch]*") if p.name not in INTERFACE),
                         ids=lambda p: p.name)
def test_descriptors_and_buffer_sizes_only_in_plan_builders(path):
    assert offenders(path.read_text()) == []
