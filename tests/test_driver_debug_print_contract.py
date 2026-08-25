"""One print-verbosity contract for every production driver.

The source used to expose a collection of subsystem-local log toggles.  This
gate keeps those spellings retired while allowing forensic writers and probes
that change execution to retain their separately named controls.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Pure output controls retired into LORRAX_DEBUG_PRINT.  Do not add
# extra-compute diagnostics (W residual, staged-tau timing, LU sync checks) or
# artifact writers (H5 journal, Davidson/key dumps): those are not verbosity.
RETIRED_PRINT_ENVS = (
    "LORRAX_COLLECTIVE_CHUNK_LOG",
    "LORRAX_CONV_KLEAD_LOG",
    "LORRAX_CONV_KMINOR_LOG",
    "LORRAX_CONV_KPAIR_LOG",
    "LORRAX_CUFFT_LOG",
    "LORRAX_FFI_DEBUG_SHARDS",
    "LORRAX_FFT_FFI_LOG",
    "LORRAX_JAX_CACHE_EXPLAIN",
    "LORRAX_MEM_DEBUG",
    "LORRAX_MKLBLAS_LOG",
    "LORRAX_MKLFFT_LOG",
    "LORRAX_PHDF5_CLOSE_VERBOSE",
    "LORRAX_PHDF5_DUMP_HINTS",
    "LORRAX_PHDF5_LOG",
    "LORRAX_PHDF5_TIME",
    "LORRAX_PHDF5_WRITE_DEBUG",
    "LORRAX_RCHUNK_DEBUG",
    "LORRAX_RESTART_WRITE_LOG",
    "LORRAX_SCALAPACK_EIGH_LOG",
    "LORRAX_SCALAPACK_PROVIDER_LOG",
    "LORRAX_SELECT_GPU_QUIET",
    "LORRAX_TIMING_TRACE",
    "LORRAX_TIMING_TRACE_DEPTH",
    "LORRAX_ZETA_RANK_LOG",
)


def _active_source_text() -> str:
    roots = (ROOT / "src", ROOT / "services")
    suffixes = {".py", ".cc", ".h", ".sh"}
    return "\n".join(
        path.read_text(errors="replace")
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in suffixes
    )


def test_retired_print_controls_cannot_regrow_as_runtime_strings():
    source = _active_source_text()
    offenders = [
        name for name in RETIRED_PRINT_ENVS
        if f'"{name}"' in source or f"'{name}'" in source
    ]
    assert offenders == [], (
        "print verbosity has more than one runtime control: "
        + ", ".join(offenders)
    )


def test_env_registry_has_one_print_verbosity_row():
    registry = (ROOT / "docs" / "dev" / "env_vars.md").read_text()
    assert registry.count("| `LORRAX_DEBUG_PRINT` |") == 1
    stale_rows = [
        name for name in RETIRED_PRINT_ENVS
        if f"| `{name}` |" in registry
    ]
    assert stale_rows == []


def test_native_and_python_layers_share_the_exact_spelling():
    runtime = (ROOT / "src" / "runtime" / "__init__.py").read_text()
    native = (ROOT / "src" / "ffi" / "cpp" / "common"
              / "mkl_thread_pin.h").read_text()
    phdf5 = "\n".join((ROOT / "src" / "ffi" / "cpp" / "phdf5" / p).read_text()
                       for p in ("context.cc", "read_ffi.cc", "write_ffi.cc"))
    selector = (ROOT / "src" / "ffi" / "cpp" / "select_gpu.sh").read_text()
    assert 'DEBUG_PRINT_ENV = "LORRAX_DEBUG_PRINT"' in runtime
    assert 'std::getenv("LORRAX_DEBUG_PRINT")' in native
    assert 'env_flag("LORRAX_DEBUG_PRINT", false)' in phdf5
    assert 'LORRAX_DEBUG_PRINT' in selector
