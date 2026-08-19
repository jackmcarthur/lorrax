"""Gate for the ONE startup entry point and the block it prints.

``runtime.initialize_communicator_stack`` is the single call every core
LORRAX driver makes; ``runtime.format_startup_report`` turns what it
resolved into the rank-0 block the owner reads to answer performance
questions without re-running anything.

WHY THIS FILE IS MOSTLY ABOUT THE *WORDING*.  The block is the product.  A
report that omits a demotion is worse than no report, because its silence
reads as "nothing was demoted"; and a report whose numbers came from
``os.environ`` rather than from the live client is actively wrong (measured,
job 7882443: identical ``os.environ``, ``bytes_limit`` 11.805 GB vs 0.000
GB).  ``format_startup_report`` is a pure function of a facts dict precisely
so those branches can be exercised here, on a login node, including the ones
that only occur on hardware nobody can allocate on demand.

NOT-VOID DISCIPLINE (standing lesson #1: make every instrument prove it can
fail).  Four checks here are built to fail rather than to pass:

* :func:`test_the_sentence_check_can_fail` feeds the sentence checker a line
  that is not a sentence — if that passed, every "every line is a sentence"
  assertion below would be vacuous.
* :func:`test_no_demotion_means_no_demotion_line` is the negative control for
  the demotion assertions: without it, a formatter that printed "DEMOTION:"
  unconditionally would satisfy them.
* :func:`test_the_dial_roster_scanners_can_fail` points both halves of the
  gate-roster linkage at things that have no gate, so a scanner that always
  answered could not make the linkage test vacuous.
* :func:`test_oversubscription_warning_is_not_unconditional` is the negative
  control for the thread warning.

AND THE FILE ITSELF WAS SHOWN FAILING.  Six source mutations were applied to
a scratch copy of ``src/`` (2026-07-31) and each was caught by exactly the
intended test, with the other 46 still green:

===================================================== =========================
mutation                                              test that caught it
===================================================== =========================
formatter drops the ``DEMOTION:`` lines               ``..._reaches_the_report``
gloo branch loses its corruption wording              ``..._silent_corruption``
one ``_record_demotion`` deleted from the plugin skip  ``..._feeds_the_ledger``
failfast hook moved below ``bootstrap()``             ``..._correct_order``
formatter re-reads ``os.environ`` for a dial          ``..._cannot_change_...``
a new ``Gate`` added to ``src/ffi``, uncollected       ``..._every_gate_...``
===================================================== =========================

The first version of the ledger check asserted a TOTAL count and did NOT
catch mutation 3 (deleting one of six sites still left five); it is now a
per-function minimum, which does.
"""
import ast
import ctypes
import os
import sys

import pytest

import runtime


_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src")


@pytest.mark.parametrize("relative", [
    "centroid/kmeans_cli.py",
    "psp/get_dipole_mtxels.py",
    "gw/kin_ion_io.py",
    "gw/gw_jax.py",
    "bse/bse_jax.py",
    "bandstructure/htransform.py",
    "bse/exciton_bands.py",
])
def test_every_core_driver_uses_the_shared_finalize_boundary(relative):
    tree = ast.parse(open(os.path.join(_SRC, relative), encoding="utf-8").read())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_main_and_finalize"
    ]
    assert len(calls) == 1, relative


@pytest.mark.parametrize("outcome, expected", [
    (None, 0),
    (7, 7),
    (SystemExit(), 0),
    (SystemExit(3), 3),
    (SystemExit("bad input"), 1),
    (RuntimeError("boom"), 1),
])
def test_shared_finalize_boundary_preserves_failure_status(
        monkeypatch, capsys, outcome, expected):
    seen = []
    monkeypatch.setattr(runtime, "finalize_process", seen.append)
    monkeypatch.setattr(runtime, "_resolve_proc_count", lambda: 1)

    def main():
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    runtime.run_main_and_finalize(main)
    assert seen == [expected]
    captured = capsys.readouterr()
    if isinstance(outcome, RuntimeError):
        assert "RuntimeError: boom" in captured.err
    if isinstance(outcome, SystemExit) and isinstance(outcome.code, str):
        assert outcome.code in captured.err


def test_shared_boundary_delegates_rank_local_failure_to_failfast(
        monkeypatch):
    """A P>1 RuntimeError must never enter collective ordered teardown."""
    seen = []

    class FailFastExit(Exception):
        pass

    def hook(exc_type, exc_value, exc_tb):
        seen.append((exc_type, str(exc_value), exc_tb is not None))
        raise FailFastExit

    monkeypatch.setattr(runtime, "finalize_process",
                        lambda rc: pytest.fail("ordered teardown was entered"))
    monkeypatch.setattr(runtime, "_resolve_proc_count", lambda: 4)
    monkeypatch.setattr(sys, "excepthook", hook)
    monkeypatch.setattr(sys, "_lorrax_failfast_installed", True, raising=False)

    with pytest.raises(FailFastExit):
        runtime.run_main_and_finalize(
            lambda: (_ for _ in ()).throw(RuntimeError("rank-local boom")))
    assert seen == [(RuntimeError, "rank-local boom", True)]


@pytest.mark.parametrize("exc", [SystemExit(3), SystemExit("bad input")])
def test_shared_boundary_delegates_nonzero_system_exit_to_failfast(
        monkeypatch, exc):
    """A P>1 validation exit must not enter collective ordered teardown."""
    seen = []

    class FailFastExit(Exception):
        pass

    def hook(exc_type, exc_value, exc_tb):
        seen.append((exc_type, str(exc_value), exc_tb is not None))
        raise FailFastExit

    monkeypatch.setattr(runtime, "finalize_process",
                        lambda rc: pytest.fail("ordered teardown was entered"))
    monkeypatch.setattr(runtime, "_resolve_proc_count", lambda: 4)
    monkeypatch.setattr(sys, "excepthook", hook)
    monkeypatch.setattr(sys, "_lorrax_failfast_installed", True, raising=False)

    with pytest.raises(FailFastExit):
        runtime.run_main_and_finalize(
            lambda: (_ for _ in ()).throw(exc))
    assert seen == [(SystemExit, str(exc), True)]


@pytest.mark.parametrize("provided, should_pass", [(1, False), (2, False),
                                                    (3, True)])
def test_cpu_mpi_thread_gate_uses_the_live_grant(
        monkeypatch, tmp_path, provided, should_pass):
    wrapper = tmp_path / "libmpiwrapper.so"
    wrapper.touch()
    monkeypatch.setenv("JAX_CPU_COLLECTIVES_IMPLEMENTATION", "mpi")
    monkeypatch.setenv("MPITRAMPOLINE_LIB", str(wrapper))

    class FakeJax:
        @staticmethod
        def default_backend():
            return "cpu"

        @staticmethod
        def process_count():
            return 4

    class Query:
        argtypes = None
        restype = None

        def __call__(self, pointer):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = provided
            return 0

    class Adapter:
        MPIABI_Query_thread = Query()

    monkeypatch.setitem(sys.modules, "jax", FakeJax)
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: Adapter())
    lines = []
    if should_pass:
        runtime._enforce_cpu_mpi_thread_multiple(lines.append)
        assert lines and "MPI_THREAD_MULTIPLE (3)" in lines[0]
    else:
        with pytest.raises(RuntimeError, match="live MPI grant"):
            runtime._enforce_cpu_mpi_thread_multiple(lines.append)


# ---------------------------------------------------------------------------
# A complete, healthy facts dict — the baseline every test mutates
# ---------------------------------------------------------------------------

def _facts(**over):
    f = {
        "process_count": 16,
        "process_index": 0,
        "n_local_devices": 1,
        "n_devices": 16,
        "distributed_form": (
            "jax.distributed.initialize() took its explicit form with "
            "coordinator_address='c123-456:30301', num_processes=16 and "
            "local_device_ids=None"),
        "backend": "cpu",
        "device_kind": "cpu",
        "jax_platforms_env": "cpu",
        "x64": True,
        "jax_version": "0.9.1",
        "mesh_shape": (4, 4),
        "mesh_axes": ("x", "y"),
        "demotions": [],
        "collectives": {"applicable": True, "impl": "mpi",
                        "impl_was_set": True,
                        "wrapper": "/work2/libmpiwrapper.so",
                        "finalize_fix": "skip_atexit",
                        "force_thread_main": None},
        "pool": {"stats": None, "error": None, "env": None,
                 "corroboration": "none", "disagreement": ""},
        "ffi_library": {"platform": "cpu",
                        "path": "/work2/liblorrax_ffi_host.so",
                        "loaded": True, "reason": "",
                        "env_var": "LORRAX_FFI_HOST_SO",
                        "env_value": "/work2/liblorrax_ffi_host.so"},
        "ffi_dials": [
            {"env": "LORRAX_BANDS_GEMM_FFI", "mode": "on", "enabled": True,
             "default": "on", "target": "lorrax_mklblas_gemm_batch",
             "platforms": ("cpu",), "raw": None,
             "off_policy": "fallback",
             "off_label": "native XLA dot lowering",
             "detail": "the contract_bands right-GEMM contraction"},
            {"env": "LORRAX_FFT_FFI", "mode": "on", "enabled": True,
             "default": "on", "target": "lorrax_mklfft_flat_k",
             "platforms": ("cpu", "CUDA"), "raw": None,
             "off_policy": "refuse",
             "off_label": "(deleted) XLA flat-k FFT path",
             "detail": "the flat-k 3-D FFT helper path"},
            {"env": "LORRAX_FFT_FFI_FUSED", "mode": "off", "enabled": False,
             "default": "on", "target": "lorrax_mklfft_gw_conv",
             "platforms": ("cpu", "CUDA"), "raw": "0",
             "off_policy": "fallback",
             "off_label": "decomposed three-FFT chain (still FFI-served)",
             "detail": "the fused IFFT-multiply-FFT tau kernel"},
            # The one OPT-IN dial: default OFF, where `off` is the CERTIFIED
            # state.  It is in this baseline so the roster cell below has
            # something to find, and so the formatter's default=="off" branch
            # is exercised by every rendering test rather than only by its own.
            {"env": "LORRAX_CONV_KMINOR_FFI", "mode": "auto",
             "enabled": True, "default": "auto",
             "target": "lorrax_cufft_conv_kminor",
             "platforms": ("CUDA",), "raw": None,
             "off_policy": "fallback",
             "off_label": "the caller's XLA ifft/multiply/fft chain",
             "auto_capability": "CUDA mesh, handler present, row resident",
             "detail": "the fused k-MINOR ifft-multiply-fft conv"},
            {"env": "LORRAX_CONV_KLEAD_FFI", "mode": "off",
             "enabled": False, "default": "off",
             "target": "lorrax_cufft_conv_klead",
             "platforms": ("CUDA",), "raw": None,
             "off_policy": "fallback",
             "off_label": "the certified plan-based k-leading gw_conv handler",
             "auto_capability": "CUDA mesh, handler present, row resident",
             "detail": "the direct fused k-LEADING Sigma conv"},
        ],
        "linalg": {"eigh": ["native", "scalapack"], "cholesky": ["native"],
                   "solve_lu": ["native", "scalapack"]},
        "threads": {"affinity": 28, "OMP_NUM_THREADS": "28",
                    "MKL_NUM_THREADS": "28", "OPENBLAS_NUM_THREADS": "28",
                    "LORRAX_MKLBLAS_THREADS": None,
                    "LORRAX_SCALAPACK_MKL_THREADS": None},
        "compile_cache": {"enabled": True, "n_proc": 16,
                          "dir": "/scratch2/lorrax_jax_cache/np16"},
        "compile_cache_error": None,
        "malloc_tune": {"applied": True, "mmap_mb": 1, "trim_mb": 128,
                        "reason": None},
        "failfast": True,
        "failfast_env": None,
        "elapsed": {"env_and_distributed": 43.8, "mesh_and_warmup": 0.15,
                    "compile_cache": 29.0, "measurement": 0.4,
                    "total": 73.35},
    }
    f.update(over)
    return f


def _body(lines):
    """The report minus its two rules and its title."""
    return [ln for ln in lines
            if not set(ln.strip()) <= {"="} and "resolved startup" not in ln]


def _is_sentence(line):
    """A report body line: two-space indent, a capital or a WARNING/DEMOTION
    tag, and a terminating period."""
    return (line.startswith("  ")
            and line.strip().endswith(".")
            and len(line.strip()) > 10)


def _text(f):
    return "\n".join(runtime.format_startup_report(f))


# ---------------------------------------------------------------------------
# 1.  Sentences, with periods — the owner's explicit request
# ---------------------------------------------------------------------------

def test_every_line_is_a_complete_sentence():
    bad = [ln for ln in _body(runtime.format_startup_report(_facts()))
           if not _is_sentence(ln)]
    assert not bad, f"report lines that are not sentences: {bad}"


def test_the_sentence_check_can_fail():
    """NOT-VOID control for the check above.

    If ``_is_sentence`` accepted anything, every sentence assertion in this
    file would be green for the wrong reason.
    """
    assert not _is_sentence("  a line with no terminator")
    assert not _is_sentence("no indent, has a period.")
    assert not _is_sentence("  short.")
    assert _is_sentence("  This one is a real sentence.")


@pytest.mark.parametrize("mut", [
    {"backend": "gpu", "device_kind": "Quadro RTX 5000"},
    {"process_count": 1, "process_index": 0, "n_devices": 1,
     "collectives": {"applicable": False, "impl": "gloo",
                     "impl_was_set": False, "wrapper": None,
                     "finalize_fix": None, "force_thread_main": None}},
    {"collectives": {"applicable": True, "impl": "gloo",
                     "impl_was_set": False, "wrapper": None,
                     "finalize_fix": None, "force_thread_main": None}},
    {"ffi_library": {"platform": "cpu", "path": None, "loaded": False,
                     "reason": "FileNotFoundError: no such .so",
                     "env_var": "LORRAX_FFI_HOST_SO", "env_value": None}},
    {"malloc_tune": {"applied": False, "mmap_mb": None, "trim_mb": None,
                     "reason": "mallopt() returned 0"}},
    {"failfast": False},
    {"compile_cache": {"enabled": False, "n_proc": 16, "dir": None}},
    {"compile_cache_error": "OSError: read-only filesystem"},
    {"linalg": {"error": "ImportError: no distrib_la"}},
    {"demotions": ["JAX_PLATFORMS was pinned to 'cpu'."]},
])
def test_every_branch_still_produces_sentences(mut):
    """The unhappy branches are exactly the ones nobody reads before a
    release; they must be sentences too."""
    bad = [ln for ln in _body(runtime.format_startup_report(_facts(**mut)))
           if not _is_sentence(ln)]
    assert not bad, f"{mut} -> non-sentences {bad}"


# ---------------------------------------------------------------------------
# 2.  Demotions — the honesty requirement
# ---------------------------------------------------------------------------

def test_a_recorded_demotion_reaches_the_report():
    d = ("JAX_PLATFORMS was requested as 'cuda,cpu' and was pinned to 'cpu' "
         "before backend init because this node exposes no NVIDIA device "
         "node; this run has no GPU.")
    out = _text(_facts(demotions=[d]))
    assert "DEMOTION:" in out
    assert d in out


def test_no_demotion_means_no_demotion_line():
    """NEGATIVE CONTROL.  Without this, a formatter that printed a
    'DEMOTION:' header unconditionally would satisfy the test above."""
    assert "DEMOTION:" not in _text(_facts(demotions=[]))


def test_the_rank_scope_of_the_demotion_list_is_stated():
    """A rank-0-only report that silently drops rank 7's demotion would be
    the exact failure mode the announcement doctrine exists to prevent, so
    the block has to say that is what it is doing."""
    out = _text(_facts(demotions=["something happened."]))
    assert "only on another rank" in out


#: Every function in ``runtime`` that can resolve something differently from
#: what was asked, and how many such outcomes it has.  Counted per FUNCTION,
#: not in total: a total-only check passes while one branch of a
#: three-outcome function goes unrecorded, which is the exact shape of the
#: bug (a demotion nobody is told about).
_DEMOTION_SITES = {
    # LORRAX_MALLOC_TUNE=0, and mallopt refusing the threshold.
    "tune_glibc_malloc": 2,
    # the opt-out, the skip being armed, and arm 2's CPU pin.
    "skip_gpu_plugin_discovery": 3,
    # the auto-detected initialize() form failing over to the explicit one.
    "init_jax_distributed": 1,
    # the post-backend-init CPU downgrade.
    "fallback_to_cpu_if_no_gpu_backend": 1,
}


def test_every_demotion_site_in_runtime_feeds_the_ledger():
    """Source-level: every function that can demote records it, in EVERY
    branch that can demote, and every record carries a real message."""
    src = open(os.path.join(_SRC, "runtime", "__init__.py")).read()
    tree = ast.parse(src)
    for fname, want in _DEMOTION_SITES.items():
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == fname), None)
        assert fn is not None, f"{fname} no longer exists"
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "_record_demotion"]
        assert len(calls) >= want, (
            f"{fname} has {len(calls)} demotion records but {want} outcomes "
            f"that resolve differently from what was asked; a branch that "
            f"does not record is a demotion the startup block will not show")
        for c in calls:
            assert c.args, f"a _record_demotion() call in {fname} has no message"


# ---------------------------------------------------------------------------
# 3.  The transport — gloo is a correctness hazard, not a slow option
# ---------------------------------------------------------------------------

def test_mpi_transport_names_the_wrapper():
    out = _text(_facts())
    assert "JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi" in out
    assert "/work2/libmpiwrapper.so" in out


def test_live_cpu_transport_refuses_gloo_on_every_rank(monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_CPU_COLLECTIVES_IMPLEMENTATION", "gloo")
    with pytest.raises(RuntimeError, match="plausible wrong data silently"):
        runtime.announce_cpu_collectives()


def test_live_mpi_transport_refuses_missing_wrapper(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_CPU_COLLECTIVES_IMPLEMENTATION", "mpi")
    monkeypatch.setenv("MPITRAMPOLINE_LIB", str(tmp_path / "absent.so"))
    with pytest.raises(RuntimeError, match="existing adapter"):
        runtime.announce_cpu_collectives()


def test_mpi_transport_reports_cray_launch_controls_when_present():
    out = _text(_facts(collectives={
        "applicable": True, "impl": "mpi", "impl_was_set": True,
        "wrapper": "/global/libmpiwrapper.so", "finalize_fix": None,
        "force_thread_main": None, "pmi_preloaded": True,
        "mpich_async_progress": "1"}))
    assert "LD_PRELOAD containing libpmi.so.0=True" in out
    assert "MPICH_ASYNC_PROGRESS='1'" in out


def test_mpi_without_a_wrapper_warns():
    out = _text(_facts(collectives={
        "applicable": True, "impl": "mpi", "impl_was_set": True,
        "wrapper": None, "finalize_fix": None, "force_thread_main": None}))
    assert "MPITRAMPOLINE_LIB is unset" in out
    assert "WARNING" in out


def test_gloo_is_reported_as_silent_corruption():
    out = _text(_facts(collectives={
        "applicable": True, "impl": "gloo", "impl_was_set": False,
        "wrapper": None, "finalize_fix": None, "force_thread_main": None}))
    assert "WARNING" in out
    assert "reduce_scatter" in out
    assert "cannot be trusted" in out


def test_gpu_run_does_not_warn_about_cpu_collectives():
    """Crying wolf on every GPU run is how a warning stops being read."""
    out = _text(_facts(backend="gpu", collectives={
        "applicable": False, "impl": "gloo", "impl_was_set": False,
        "wrapper": None, "finalize_fix": None, "force_thread_main": None}))
    assert "NCCL" in out
    assert "reduce_scatter" not in out


# ---------------------------------------------------------------------------
# 4.  The allocator — the number must be the CLIENT's
# ---------------------------------------------------------------------------

_GPU_ENV = {"allocator": "default", "allocator_raw": None,
            "allocator_is_valid": True, "preallocate": False,
            "preallocate_raw": "false",
            "preallocate_looks_like_a_typo": False,
            "mem_fraction": None, "mem_fraction_var": None,
            "tf_gpu_allocator_raw": None}


def test_the_reported_pool_figures_are_the_clients():
    f = _facts(backend="gpu", device_kind="Quadro RTX 5000",
               pool={"stats": {"bytes_limit": 11_805_000_000,
                               "bytes_in_use": 2_000_000_000,
                               "peak_bytes_in_use": 8_130_000_000},
                     "error": None, "env": _GPU_ENV,
                     "corroboration": "arena", "disagreement": "",
                     "accounting_present": True})
    out = _text(f)
    assert "memory_stats()" in out and "not from os.environ" in out
    # The numbers are the client's, to the digit.
    assert f"{11_805_000_000/1e9:.2f} GB" in out
    assert f"{8_130_000_000/1e9:.2f} GB" in out
    assert f"{2_000_000_000/1e9:.2f} GB" in out


def test_environment_cannot_change_the_reported_number(monkeypatch):
    """The formatter is PURE: exporting an allocator variable after the fact
    must not move a single digit of the report."""
    f = _facts(backend="gpu",
               pool={"stats": {"bytes_limit": 11_805_000_000,
                               "bytes_in_use": 0, "peak_bytes_in_use": 0},
                     "error": None, "env": _GPU_ENV,
                     "corroboration": "arena", "disagreement": "",
                     "accounting_present": True})
    before = _text(f)
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    monkeypatch.setenv("LORRAX_FFT_FFI", "1")
    assert _text(f) == before


def test_a_client_environment_disagreement_is_a_warning():
    f = _facts(backend="gpu",
               pool={"stats": {}, "error": None, "env": _GPU_ENV,
                     "corroboration": "none",
                     "disagreement": ("the allocator resolves to 'default' "
                                      "but the client reports bytes_limit=0"),
                     "accounting_present": False})
    out = _text(f)
    assert "WARNING" in out
    assert "bytes_limit=0" in out


def test_a_non_canonical_allocator_pair_says_so():
    env = dict(_GPU_ENV, allocator="cuda_async", allocator_raw="cuda_async")
    out = _text(_facts(backend="gpu",
                       pool={"stats": {"bytes_limit": 0, "bytes_in_use": 1,
                                       "peak_bytes_in_use": 1},
                             "error": None, "env": env,
                             "corroboration": "arena", "disagreement": "",
                             "accounting_present": True}))
    assert "NOT LORRAX's canonical pair" in out


def test_cpu_run_says_the_gpu_knobs_do_not_apply():
    out = _text(_facts(backend="cpu"))
    assert "do not apply on the cpu backend" in out


# ---------------------------------------------------------------------------
# 5.  The FFI dials — every dial, every run, on or off
# ---------------------------------------------------------------------------

def _const_str(node):
    """The string literal in ``node``, across the 3.7 ``Str`` / 3.8+
    ``Constant`` split (``ast.Str`` is removed in 3.14)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if node.__class__.__name__ == "Str":
        return node.s
    return None


def _gate_env_names_in_tree():
    """Every ``Gate(env=...)`` literal under ``src/ffi``.

    Scanned rather than listed so a NEW dial added without a report entry
    fails here instead of quietly going unreported.
    """
    found = set()
    for root, _dirs, files in os.walk(os.path.join(_SRC, "ffi")):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(root, fn)).read(), fn)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and getattr(node.func, "id", None) == "Gate"):
                    for kw in node.keywords:
                        if kw.arg == "env":
                            s = _const_str(kw.value)
                            if s:
                                found.add(s)
    return found


def _gate_env_of(module_dotted, symbol):
    """The ``env=`` literal of the ``Gate`` assigned to ``symbol`` in
    ``module_dotted`` — resolved from source, so no jax import is needed."""
    path = os.path.join(_SRC, *module_dotted.split(".")) + ".py"
    if not os.path.exists(path):
        return None
    tree = ast.parse(open(path).read(), path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if symbol not in names:
            continue
        v = node.value
        if isinstance(v, ast.Call) and getattr(v.func, "id", None) == "Gate":
            for kw in v.keywords:
                if kw.arg == "env":
                    return _const_str(kw.value)
    return None


def _dials_reported_by_runtime():
    """Which dials ``runtime._ffi_dial_facts`` actually collects.

    Follows the function's own ``from ffi.… import GATE as …`` statements
    back to the ``Gate(env=…)`` literal they name.  This is the LINKAGE
    check: comparing the tree's gates against a list hand-copied into this
    test would only ever catch the test drifting, which is the failure mode
    that does not matter.
    """
    src = open(os.path.join(_SRC, "runtime", "__init__.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_ffi_dial_facts")
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                env = _gate_env_of(node.module, alias.name)
                if env:
                    out.add(env)
    return out


def test_report_covers_every_gate_the_tree_defines():
    names = _gate_env_names_in_tree()
    assert names, "the Gate scan found nothing — the scanner is broken"
    collected = _dials_reported_by_runtime()
    assert names <= collected, (
        f"FFI dials defined in src/ffi that runtime._ffi_dial_facts does NOT "
        f"collect, so they would never appear in the startup block: "
        f"{sorted(names - collected)}.  Add the gate to _ffi_dial_facts.")
    out = _text(_facts())
    for n in names:
        assert n in out, f"{n} is not stated in the report"


def test_the_dial_roster_scanners_can_fail():
    """NOT-VOID control for BOTH halves of the roster cross-check.

    Each scanner must return nothing when pointed at something that has no
    gate; a scanner that always returns the right answer, or always returns
    the empty set, would make the linkage test above vacuous in one
    direction or the other.
    """
    assert _gate_env_of("runtime", "GATE") is None
    assert _gate_env_of("no.such.module", "GATE") is None
    assert _gate_env_of("ffi.gemm", "NOT_A_GATE") is None
    # ...and it really does resolve the ones that exist.
    assert _gate_env_of("ffi.gemm", "GATE") == "LORRAX_BANDS_GEMM_FFI"
    assert _gate_env_of("ffi.fft", "FUSED_GATE") == \
        "LORRAX_FFT_FFI_FUSED"


def test_a_dial_missing_from_the_facts_is_missing_from_the_block():
    """The formatter states what it is given and invents nothing — which is
    why the linkage test above has to police the COLLECTOR."""
    thinned = [d for d in _facts()["ffi_dials"]
               if d["env"] != "LORRAX_FFT_FFI"]
    out = _text(_facts(ffi_dials=thinned))
    assert "LORRAX_FFT_FFI_FUSED" in out
    assert "  The LORRAX_FFT_FFI dial" not in out


def test_a_dial_that_is_off_is_still_stated():
    """Silence about an off dial is indistinguishable from silence about an
    on one; the block must state both — including that an =0 opt-out is
    uncertified (the FFI layer is required, decisions.md 2026-08-01)."""
    out = _text(_facts())
    assert ("The LORRAX_FFT_FFI_FUSED dial is set to '0' and resolved to "
            "off") in out
    assert "uncertified for production" in out
    assert "decomposed three-FFT chain" in out


def test_an_on_dial_states_the_required_route():
    out = _text(_facts())
    assert "The LORRAX_FFT_FFI dial is unset and resolved to on" in out
    assert "routed through the FFI handler (the required layer)" in out


def test_an_on_dial_out_of_platform_scope_states_the_native_route():
    """On a GPU run the LORRAX_BANDS_GEMM_FFI dial does not exist (its
    handler is in the host symbol table only; XLA:GPU's dot lowering
    already dispatches cuBLAS) and ``Gate.enforce`` skips it silently by
    declared policy.  The P1.2 GPU certification (job 7885151) caught the
    block claiming the contraction was "routed through the FFI handler"
    on a CUDA mesh — a platform-blind sentence.  The block must state the
    platform truth instead, while the in-scope dials keep the required
    wording."""
    out = _text(_facts(backend="gpu", device_kind="Quadro RTX 5000",
                       jax_platforms_env="cuda,cpu"))
    assert ("The LORRAX_BANDS_GEMM_FFI dial is unset and resolved to on, "
            "so the contract_bands right-GEMM contraction rides the "
            "platform's native lowering") in out
    assert "exists on cpu only" in out
    assert "native lowering IS the required path there" in out
    # the platform-scoped truth must not leak onto in-scope dials: the FFT
    # dial exists on CUDA and still states the required route.
    assert ("so the flat-k 3-D FFT helper path is routed through the FFI "
            "handler (the required layer)") in out


def test_an_off_dial_whose_twin_was_deleted_states_the_refusal():
    """LORRAX_FFT_FFI=0 has nothing to run (the XLA duplicate was deleted);
    the report must say so rather than describe a path that is gone."""
    dials = [dict(d) for d in _facts()["ffi_dials"]]
    for d in dials:
        if d["env"] == "LORRAX_FFT_FFI":
            d.update(mode="off", enabled=False, raw="0")
    out = _text(_facts(ffi_dials=dials))
    assert "native duplicate was deleted" in out
    assert "startup enforcement refuses this setting" in out


def test_a_missing_ffi_library_is_stated_once_and_clearly():
    out = _text(_facts(ffi_library={
        "platform": "cpu", "path": None, "loaded": False,
        "reason": "FileNotFoundError: could not locate liblorrax_ffi_host.so",
        "env_var": "LORRAX_FFI_HOST_SO", "env_value": None}))
    assert "No cpu FFI library could be loaded" in out
    assert "The FFI layer is REQUIRED" in out
    assert "docs/environment/overview.md" in out


def test_linalg_capability_is_reported_per_op():
    out = _text(_facts())
    for op in ("eigh", "cholesky", "solve_lu"):
        assert f"available for {op} on this mesh" in out
    assert "input-file key, not an environment variable" in out


def test_linalg_facts_answers_for_real_on_a_healthy_tree():
    """``_linalg_facts`` must return CAPABILITY, not a printable excuse.

    ARM B, SEAM 1 — the quietest seam in the replumb.  ``_linalg_facts``
    wraps its import in ``except Exception: return {"error": …}``, and the
    fixture row ``{"linalg": {"error": "ImportError: no distrib_la"}}``
    above asserts that the report still forms a SENTENCE when it does.
    Between them, a broken import degrades to a grammatical line, that row
    passes on the literal failure it anticipates, and the startup banner
    stops reporting backend capability forever with nothing failing.

    Nothing asserted the OTHER half.  This does: on a tree where
    ``distrib_la`` imports (this one — the assert says so), the dict has an
    entry per op and no ``error`` key at all.

    RED ARM: the FALSE case is the fixture row itself, and the twin below
    constructs it.
    """
    pytest.importorskip("jax")
    from ffi import _services
    _services.ensure_on_path()
    import distrib_la                                        # noqa: F401
    import jax
    from jax.sharding import Mesh
    import numpy as np
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    facts = runtime._linalg_facts(mesh)
    assert "error" not in facts, (
        f"the startup report's linalg block degraded to a printed error on "
        f"a tree where distrib_la imports: {facts['error']}")
    assert set(facts) >= {"eigh", "cholesky", "solve_lu"}, facts
    assert all(isinstance(v, (list, tuple)) and v for v in facts.values()), (
        f"an op reported no backend at all — 'native' is the floor "
        f"everywhere, so an empty list is a defect, not a capability: "
        f"{facts}")


def test_the_linalg_facts_error_path_is_labelled_when_it_is_taken(monkeypatch):
    """The FALSE case, constructed — and it must stay LABELLED.

    Swallowing is acceptable here (a startup banner must not abort a run)
    only while the swallow is legible: the dict says ``error`` and the
    string names the exception TYPE, so the sentence the report prints is
    diagnosable rather than merely grammatical.
    """
    pytest.importorskip("jax")
    import sys
    import jax
    from jax.sharding import Mesh
    import numpy as np
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    monkeypatch.setitem(sys.modules, "distrib_la", None)     # -> ImportError
    facts = runtime._linalg_facts(mesh)
    assert set(facts) == {"error"}, facts
    import builtins
    kind = facts["error"].split(":", 1)[0]
    assert issubclass(getattr(builtins, kind, object), ImportError), (
        f"the swallow is only acceptable while it stays legible; the label "
        f"{kind!r} is not an exception type a reader can act on: "
        f"{facts['error']}")


# ---------------------------------------------------------------------------
# 6.  Threads
# ---------------------------------------------------------------------------

def test_startup_phase_timings_are_reported():
    """The two largest unexplained startup costs of the campaign — 43.8 s of
    jax.distributed init at P=16 and 29 s of compile-cache prefetch — happen
    before any driver's timing table starts.  The block has to name them."""
    out = _text(_facts())
    assert "43.8 s for the environment" in out
    assert "0.1 s to build the mesh" in out
    assert "29.0 s to arm the compile cache" in out


def test_the_block_survives_without_phase_timings():
    """``collect_startup_facts`` can be called by a probe that did not time
    anything; the block must still be a block."""
    f = _facts()
    del f["elapsed"]
    bad = [ln for ln in _body(runtime.format_startup_report(f))
           if not _is_sentence(ln)]
    assert not bad, bad
    assert "Bringing this stack up took" not in _text(f)


def test_threads_are_reported_from_affinity_not_cpu_count():
    out = _text(_facts())
    assert "pinned to 28 schedulable CPUs" in out
    assert "OMP_NUM_THREADS='28'" in out


def test_oversubscription_warns():
    t = dict(_facts()["threads"], affinity=28, OMP_NUM_THREADS="56")
    out = _text(_facts(threads=t))
    assert "exceeds this process's 28-CPU affinity" in out


def test_oversubscription_warning_is_not_unconditional():
    """NEGATIVE CONTROL for the warning above."""
    assert "exceeds this process" not in _text(_facts())


# ---------------------------------------------------------------------------
# 7.  Compile cache
# ---------------------------------------------------------------------------

def test_compile_cache_on_names_the_directory():
    out = _text(_facts())
    assert "/scratch2/lorrax_jax_cache/np16" in out
    assert "agreement layer" in out


def test_compile_cache_at_P1_does_not_say_it_is_shared_by_1_rank():
    out = _text(_facts(process_count=1, n_devices=1,
                       compile_cache={"enabled": True, "n_proc": 1,
                                      "dir": "/scratch/np1"},
                       collectives={"applicable": False, "impl": "gloo",
                                    "impl_was_set": False, "wrapper": None,
                                    "finalize_fix": None,
                                    "force_thread_main": None}))
    assert "used by this single rank" in out
    assert "shared by all 1 ranks" not in out


def test_compile_cache_off_is_stated_not_omitted():
    out = _text(_facts(compile_cache={"enabled": False, "n_proc": 16,
                                      "dir": None}))
    assert "compile cache is OFF" in out
    assert "ISDF_JAX_CACHE_DIR" in out


def test_compile_cache_always_carries_the_shape_key_caveat():
    """Standing lesson: the key includes shapes, so it does nothing for a
    system size this machine has not run.  A warm-looking cache that cannot
    help is exactly the thing a startup block must not imply."""
    for cc in ({"enabled": True, "n_proc": 16, "dir": "/x"},
               {"enabled": False, "n_proc": 16, "dir": None}):
        assert "cache key includes every array shape" in _text(
            _facts(compile_cache=cc))


# ---------------------------------------------------------------------------
# 8.  Guards
# ---------------------------------------------------------------------------

def test_failfast_missing_at_P_gt_1_warns():
    out = _text(_facts(failfast=False, failfast_env="0"))
    assert "WARNING" in out and "fail-fast excepthook is NOT installed" in out


def test_failfast_absent_at_P_1_is_not_a_warning():
    out = _text(_facts(process_count=1, n_devices=1, failfast=False,
                       collectives={"applicable": False, "impl": "gloo",
                                    "impl_was_set": False, "wrapper": None,
                                    "finalize_fix": None,
                                    "force_thread_main": None}))
    assert "fail-fast excepthook is NOT installed" not in out


def test_malloc_tuning_failure_is_stated():
    out = _text(_facts(malloc_tune={"applied": False, "mmap_mb": None,
                                    "trim_mb": None,
                                    "reason": "mallopt() returned 0"}))
    assert "glibc malloc tuning is NOT in force" in out
    assert "7874803" in out


def test_malloc_tuning_never_attempted_is_stated():
    """``applied is None`` means ``set_default_env`` never ran, i.e. this
    process was not configured by LORRAX at all.  Reporting that as "off"
    would hide a much larger problem."""
    out = _text(_facts(malloc_tune={"applied": None, "mmap_mb": None,
                                    "trim_mb": None, "reason": None}))
    assert "never attempted" in out


# ---------------------------------------------------------------------------
# 9.  The entry point's ORDER — the part that is load-bearing
# ---------------------------------------------------------------------------

def _entry_point_call_sequence():
    src = open(os.path.join(_SRC, "runtime", "__init__.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "initialize_communicator_stack")
    seq = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(
                node.func, "attr", None)
            if name:
                seq.append((node.lineno, name))
    return [n for _, n in sorted(seq)]


def test_the_entry_point_calls_the_phases_in_the_one_correct_order():
    seq = _entry_point_call_sequence()
    wanted = ["install_failfast_excepthook", "bootstrap", "prepare_mesh",
              "ensure_jax_compile_cache", "_install_pjrt_log_filter",
              "collect_startup_facts", "format_startup_report"]
    idx = []
    for w in wanted:
        assert w in seq, f"{w} is not called by initialize_communicator_stack"
        idx.append(seq.index(w))
    assert idx == sorted(idx), (
        f"phase order violated: {list(zip(wanted, idx))}.  The failfast hook "
        f"must precede the collectives it protects, bootstrap must precede "
        f"the mesh (backend init), and the report must be LAST because it "
        f"reads the live client.")


def test_the_order_is_implemented_once():
    """``initialize_communicator_stack`` must delegate the env/distributed/
    CPU-fallback half to ``bootstrap`` rather than re-listing its steps —
    two copies of a load-bearing order is how the drift this consolidation
    removes came back last time."""
    seq = _entry_point_call_sequence()
    for step in ("set_default_env", "init_jax_distributed",
                 "fallback_to_cpu_if_no_gpu_backend"):
        assert step not in seq, (
            f"{step} is called directly by initialize_communicator_stack; it "
            f"belongs to bootstrap(), which is the single implementation of "
            f"the order")


def test_a_second_mesh_shape_is_refused_not_silently_ignored():
    """A second ``Mesh`` is a second set of communicators.  The guard has to
    refuse, not hand back the first mesh under a different name."""
    src = open(os.path.join(_SRC, "runtime", "__init__.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "initialize_communicator_stack")
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    assert raises, "the re-entry guard cannot refuse anything"


def test_reshape_exists_for_the_one_case_the_entry_point_cannot_serve():
    """``bse.exciton_bands`` / ``bse.bse_w_exact`` take --px/--py, which the
    module-level startup call cannot know.  ``RuntimeStack.reshape`` is how
    they stay on ONE startup call; it must warm the new mesh and announce
    the swap, or the block above it describes a mesh the run is not using.
    """
    src = open(os.path.join(_SRC, "runtime", "__init__.py")).read()
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "reshape"), None)
    assert fn is not None, "RuntimeStack.reshape is missing"
    names = {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
             for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert "prepare_mesh" in names, (
        "reshape() builds a mesh without warming its cliques — the parallel "
        "thunk executor would then create communicators from a pool worker")
    assert "say" in names, "reshape() swaps the mesh without announcing it"
    assert any(isinstance(n, ast.Raise) for n in ast.walk(fn)), (
        "reshape() cannot refuse a shape that does not match the device count")
    assert hasattr(runtime.RuntimeStack, "reshape")


def test_the_public_surface_is_the_entry_point():
    for name in ("initialize_communicator_stack", "RuntimeStack",
                 "collect_startup_facts", "format_startup_report"):
        assert name in runtime.__all__
        assert hasattr(runtime, name)
