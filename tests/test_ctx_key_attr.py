"""The vendor handlers' context attribute is a pure function of the configuration.

An FFI attribute is baked into the HLO.  ``ctx_handle`` (a heap address)
made every process emit a new module, so the persistent compile cache could
never hit a cuBLASMp/cuSolverMp/SLATE module; ``ctx_key`` is a hash of the
context's configuration, resolved to the live context by the native
registry (``src/ffi/cpp/common/ctx_registry.h``).  These cells need no
library: the key's determinism across processes (hash salting) and the
absence of any address-valued attribute in the handlers and their callers.
The registry's refusals and the teardown red twin run on GPU in
``tests/multi_device/ctx_key_p4.py``.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CONFIGS = ("lorrax-ctx/v1|cusolvermp|2x2|row", "lorrax-ctx/v1|cusolvermp|2x2|col",
            "lorrax-ctx/v1|slate|world|2x2|cpu", "lorrax-ctx/v1|slate|subrow|2x2|cpu")


def _keys_in_subprocess(seed: str) -> list[int]:
    code = ("from distrib_la.loader import context_key; "
            f"print(' '.join(str(context_key(c)) for c in {_CONFIGS!r}))")
    env = dict(os.environ, PYTHONHASHSEED=seed,
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, check=True)
    return [int(v) for v in out.stdout.split()]


def test_context_key_is_the_same_in_every_process_and_distinct_per_configuration():
    """Two processes with different hash seeds agree; four configurations, four keys."""
    from distrib_la.loader import context_key
    here = [context_key(c) for c in _CONFIGS]
    assert _keys_in_subprocess("1") == here == _keys_in_subprocess("12345")
    assert len(set(here)) == len(here)
    assert all(0 < k < 2**63 for k in here)


def test_no_handler_or_caller_carries_an_address_valued_context_attribute():
    """No C++ handler binds ``ctx_handle`` as an Attr; no Python FFI call passes it."""
    cpp = _REPO / "src" / "ffi" / "cpp"
    for sub in ("cusolvermp", "cublasmp", "active_subspace", "slate", "scalapack"):
        for f in sorted((cpp / sub).glob("*.cc")):
            assert 'Attr<int64_t>("ctx_handle")' not in f.read_text(), f
    # A keyword ``ctx_handle=`` survives only where the live address is the
    # point: the GemmPlan record (read by the ctypes workspace query) and the
    # query itself.  Anywhere else it would be an FFI attribute again.
    allowed = {"GemmPlan", "_gemm_workspace_details"}
    callers = [_REPO / "src" / "ffi" / "cublasmp" / "batched.py",
               *sorted((_REPO / "services" / "distrib_la" / "src" / "distrib_la").glob("*.py"))]
    for f in callers:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Call) and any(k.arg == "ctx_handle" for k in node.keywords):
                name = getattr(node.func, "id", getattr(node.func, "attr", "?"))
                assert name in allowed, f"{f.name}:{node.lineno} passes ctx_handle= to {name}"
