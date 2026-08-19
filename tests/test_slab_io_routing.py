"""Parse-level tests for what is LEFT of the SlabIO configuration surface,
plus the two-plan ``w_dyson_solver`` vocabulary (gw.gw_config).

THIS FILE USED TO TEST A ROUTER.  Until 2026-08-06 SlabIO had three tiers
(``PHDF5_FFI`` / ``PHDF5_HOST`` / ``H5PY_ALLGATHER``), a ``slab_io`` deck
key, a deprecated ``use_ffi_io`` boolean, an ``auto`` platform router, and
seven separate refusals keeping the allgather tier out of any run with
more than one process.  All of it is gone: there is one transport, the
deck does not select it, and a stack that cannot serve it refuses at the
file open naming the probe that declined.

So the ten precedence/refusal tests that lived here are deleted rather
than adapted — they pinned the behaviour of code that no longer exists,
and a test that survives the deletion of its subject is testing something
else.  What remains, and is still worth pinning:

1. Retired transport/lifecycle deck keys are refused with a targeted removal
   message, so an old deck cannot pretend it still selects a deleted route.
2. ``LorraxConfig`` no longer carries a transport selector at all.
3. The MPI launcher/singleton probes, which moved from ``gw.gw_config``
   (L1) down to ``file_io._slab_io_ffi`` (L3) with the availability
   check.  That move also deleted an uphill L3 -> L1 import.
4. ``normalize_w_dyson_solver`` and ``hartree_source`` — unrelated
   parse-time vocabulary that has always lived in this file.

Style follows tests/test_qp_solver_config.py: throwaway input files, no
WFN, no probes, no GPU.  Warning assertions are substring matches.
"""
from __future__ import annotations

import pathlib
import warnings

import pytest

import gw.gw_config as gw_config
import file_io._slab_io_ffi as slab_ffi
from gw.gw_config import LorraxConfig, normalize_w_dyson_solver


BASE_INPUT = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""


def _config(tmp_path, extra: str = "", name: str = "slab_io.in",
            print_fn=None):
    path = tmp_path / name
    path.write_text(BASE_INPUT + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=print_fn or (lambda *a, **k: None))


# ---------------------------------------------------------------------------
# The retired deck keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "slab_io = phdf5_ffi",
    "slab_io = h5py_allgather",
    "slab_io = auto",
    "use_ffi_io = false",
    "use_ffi_io = true",
    "gspace_mode = host_cache",
    "gspace_mode = file_reread",
])
def test_retired_transport_keys_are_refused(tmp_path, line):
    """An old deck must remove selectors for routes that no longer exist."""
    with pytest.raises(ValueError, match="must be removed"):
        _config(tmp_path, line + "\n")


def test_config_carries_no_transport_selector(tmp_path):
    """The selector is gone from the resolved config, not just the deck.

    ``BackendConfig.slab_io`` and the ``LorraxConfig.use_ffi_io`` property
    were the two places a caller could still branch on transport after
    parsing.  Both answered a question with one answer.
    """
    cfg = _config(tmp_path)
    assert not hasattr(cfg.backend, "slab_io")
    assert not hasattr(cfg, "use_ffi_io")
    assert not hasattr(cfg.backend, "gspace_io")
    assert not hasattr(cfg, "gspace_mode")


def test_unknown_key_check_still_sees_real_typos(tmp_path):
    """Retiring a key must not blunt the typo check that guards the rest.

    ``slab_io`` has a dedicated refusal via ``_LEGACY_DECK_KEYS``; a
    neighbouring misspelling must still be reported by the typo check.
    """
    with pytest.raises(ValueError, match="slab_iox"):
        _config(tmp_path, "strict_keys = true\nslab_iox = phdf5_ffi\n")
    # ...and the retired spelling keeps its targeted refusal under strictness.
    with pytest.raises(ValueError, match="must be removed"):
        _config(tmp_path, "strict_keys = true\nslab_io = phdf5_ffi\n")


def test_gw_config_no_longer_defines_the_tier_vocabulary(tmp_path):
    """The enum, the router and the seven refusals are gone from L1."""
    for name in ("SlabIOBackend", "resolve_slab_io_backend",
                 "GspaceIO",
                 "_route_cpu_slab_io", "_route_gpu_slab_io",
                 "_validate_slab_io_key", "_refuse_explicit_h5py_allgather",
                 "_refuse_slab_io_no_parallel_writer",
                 "_slab_io_no_parallel_writer", "_probe_phdf5_host_tier"):
        assert not hasattr(gw_config, name), name


def test_slab_io_module_exposes_one_transport():
    """The public surface has neither a route nor an ignored layout knob."""
    import inspect
    from file_io import slab_io
    init_params = inspect.signature(slab_io.SlabIO.__init__).parameters
    create_params = inspect.signature(slab_io.SlabIO.create_dataset).parameters
    write_params = inspect.signature(slab_io.SlabIO.write_slab).parameters
    assert set(init_params) == {"self", "path", "mode", "mesh"}, init_params
    assert "chunks" not in create_params
    assert "chunks" not in write_params
    assert not hasattr(slab_io, "SlabIOBackend")


def test_deleted_hdf5_selectors_do_not_survive_in_neighbouring_apis():
    """Pin the less visible selector threads removed with the deck keys."""
    import ast
    import inspect
    from common.wfn_transforms import load_centroids_band_chunked
    from gw.gflat_memory_model import plan_gflat_chunks

    # kmeans_cli initializes the communicator stack at import time, so parser
    # vocabulary is intentionally checked without importing the driver on a
    # login/zero-GPU test process.
    cli_path = (pathlib.Path(__file__).resolve().parents[1]
                / "src" / "centroid" / "kmeans_cli.py")
    cli_tree = ast.parse(cli_path.read_text())
    options = {
        arg.value
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    assert "--use-phdf5" not in options
    assert "use_phdf5" not in inspect.signature(
        load_centroids_band_chunked).parameters
    assert "slab_io_replicates" not in inspect.signature(
        plan_gflat_chunks).parameters


@pytest.mark.parametrize("value, expected", [
    (None, 1),
    ("", 1),
    ("0", 0),
    ("false", 0),
    ("1", 2),
    ("true", 2),
])
def test_close_logging_defaults_to_compact(monkeypatch, value, expected):
    """Empty closes stay quiet by default while explicit 0/1 remain compatible."""
    if value is None:
        monkeypatch.delenv("LORRAX_PHDF5_CLOSE_VERBOSE", raising=False)
    else:
        monkeypatch.setenv("LORRAX_PHDF5_CLOSE_VERBOSE", value)
    assert slab_ffi._close_log_level() == expected


def test_deleted_backend_modules_are_gone():
    """The two retired backends are deleted, not merely unreachable.

    The point of the 2026-08-06 change: a code path that cannot legally be
    reached is dead code wearing a safety label.  Import must fail.
    """
    import importlib
    for mod in ("file_io._slab_io_allgather", "file_io._slab_io_mpi_host"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)


def test_wfn_loader_has_no_phdf5_host_tier_either():
    """The recurrence: the same shape of tier, in the WFN read path.

    ``WfnLoader`` kept a ``phdf5_host`` backend after SlabIO's tiers went.
    It was not a third transport — it read with the SAME independent
    POSIX h5py hyperslabs the eager backend already uses at P>1 (no MPI,
    no MPI-IO, no FFI), differing only in where the symmetry unfold ran.
    A duplicate compute path auto-selected by a missing ``.so`` is what
    decisions.md 2026-08-01 forbids demoting to.  Deleted 2026-08-06.
    """
    import inspect
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader
    assert not hasattr(WfnLoader, "_phdf5_host_build")
    ann = inspect.signature(WfnLoader.__init__).parameters["backend"]
    assert "phdf5_host" not in str(ann.annotation), ann.annotation


# ---------------------------------------------------------------------------
# The two writers of one file must request ONE layout
# ---------------------------------------------------------------------------
#
# SCOPE, up front: everything below is a SOURCE-LEVEL pin plus, where a
# compiler is present, an arithmetic cross-check of an EXTRACTED pair of
# functions.  Neither rebuilds `liblorrax_ffi*.so`, so neither says the
# deployed library requests the policy layout — only that this tree's C++
# would, if built.  A rebuild on the cluster is the missing evidence.

_CONTEXT_CC = (pathlib.Path(__file__).resolve().parents[1]
               / "src" / "ffi" / "cpp" / "phdf5" / "context.cc")


def test_cpp_no_longer_carries_a_second_stripe_default():
    """`context.cc` must derive the unset stripe layout, not hardcode it.

    Python resolves `clamp(nranks, 4, 128)` + a 1->4 MiB unit ramp; C++
    wrote the literals `"16"` and 1 MiB.  In-tree that was masked because
    `_FfiBackend.__init__` calls `_export_striping_env()` before
    `_open_file()`, so the getenv always found a value — but a caller
    reaching `ffi.io.open_file` DIRECTLY got the old constants with no
    warning, and two writers requesting different layouts is worse than
    both being wrong the same way.
    """
    src = _CONTEXT_CC.read_text()
    assert 'info_set("striping_factor", "16")' not in src
    assert "stripe_policy_count(ctx->world_size)" in src
    assert "stripe_policy_unit(ctx->world_size)" in src


def test_cpp_stripe_policy_transcribes_the_python_one():
    """Compile the two extracted C++ policy functions and diff them
    against `_stripe_policy` over 0..4100 ranks.

    This is the only thing that can hold a transcription in step, and it
    is still NOT a build of the FFI library: it compiles two `static long`
    functions in isolation with the host g++.  Skipped where no C++
    compiler exists; a FAILURE to extract the functions is a failure, not
    a skip, because that means they were renamed or removed.
    """
    import re
    import shutil
    import subprocess
    import tempfile
    from file_io._slab_io_ffi import _stripe_policy

    src = _CONTEXT_CC.read_text()
    bodies = re.findall(
        r"^static long stripe_policy_(?:count|unit)\(int world_size\) \{.*?^\}",
        src, re.S | re.M)
    assert len(bodies) == 2, (
        f"expected stripe_policy_count + stripe_policy_unit in "
        f"{_CONTEXT_CC}, found {len(bodies)}")

    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler; source pin above still applies")

    prog = ('#include <cstdio>\n' + "\n".join(bodies) +
            '\nint main(){for(int n=0;n<=4100;++n)'
            'printf("%d %ld %ld\\n",n,stripe_policy_count(n),'
            'stripe_policy_unit(n));}\n')
    with tempfile.TemporaryDirectory() as td:
        cc = pathlib.Path(td) / "p.cc"
        exe = pathlib.Path(td) / "p"
        cc.write_text(prog)
        subprocess.run([cxx, "-O1", "-std=c++17", "-o", str(exe), str(cc)],
                       check=True, capture_output=True)
        out = subprocess.run([str(exe)], check=True, capture_output=True,
                             text=True).stdout

    seen = set()
    for line in out.splitlines():
        n, c, u = (int(x) for x in line.split())
        assert _stripe_policy(n) == (c, u), (
            f"C++ and Python stripe policies disagree at nranks={n}: "
            f"C++ {(c, u)} vs Python {_stripe_policy(n)}")
        seen.add((c, u))
    # Non-vacuity: the sweep must actually cross both clamps and both
    # unit steps, or "they agree" would only mean "both are constant".
    assert (4, 1 << 20) in seen and (128, 4 << 20) in seen
    assert len({u for _c, u in seen}) == 3


@pytest.mark.parametrize("spelling,expected", [
    (None, "local"),
    ("auto", "local"),
    ("local", "local"),
    ("LOCAL", "local"),
    ("distributed", "distributed"),
])
def test_w_dyson_solver_normalization(spelling, expected):
    assert normalize_w_dyson_solver(spelling) == expected


def test_w_dyson_solver_lu_warns_and_maps_to_local():
    with pytest.warns(DeprecationWarning, match="lu"):
        assert normalize_w_dyson_solver("lu") == "local"


def test_w_dyson_solver_lstsq_removed_raises():
    with pytest.raises(ValueError, match="lstsq"):
        normalize_w_dyson_solver("lstsq")


def test_w_dyson_solver_unknown_raises():
    with pytest.raises(ValueError, match="w_dyson_solver"):
        normalize_w_dyson_solver("bogus")


def test_deck_level_lu_warns_and_parses(tmp_path):
    # Normalisation happens at PARSE time (fails/warns here, not
    # 20 minutes into a run).
    with pytest.warns(DeprecationWarning, match="lu"):
        cfg = _config(tmp_path, "w_dyson_solver = lu\n")
    assert cfg.backend.w_dyson_solver == "local"


def test_deck_level_lstsq_raises_at_parse(tmp_path):
    with pytest.raises(ValueError, match="lstsq"):
        _config(tmp_path, "w_dyson_solver = lstsq\n")


# ---------------------------------------------------------------------------
# hartree_source — validated at parse time
# ---------------------------------------------------------------------------

def test_hartree_source_invalid_rejected(tmp_path):
    with pytest.raises(ValueError, match="hartree_source"):
        _config(tmp_path, "hartree_source = bogus\n")


def test_hartree_source_valid_accepted(tmp_path):
    cfg = _config(tmp_path, "hartree_source = stored\n")
    assert cfg.hartree_source == "stored"


# ---------------------------------------------------------------------------
# MPI bootstrapability probe (the job-7884926 fastloop catch)
# ---------------------------------------------------------------------------
#
# slab_io=auto used to route to PHDF5_FFI on handler presence alone; on a
# bare launch (no PMI environment) the tier's MPI_Init_thread then ABORTS
# the process inside MPIR_pmi_init — Intel MPI does not return an error.
# The router now requires MPI bootstrapability: a launcher PMI variable,
# else a throwaway-subprocess singleton-init probe.  These tests cover the
# pure pieces; the end-to-end demotion is exercised by the sandbox fastloop
# (bare P=1 gw stage with slab_io=auto).

def test_launcher_env_detects_pmi_and_pmix(monkeypatch):
    for var in ("PMI_RANK", "PMI_FD", "PMIX_RANK", "PMIX_SERVER_URI21",
                "HYDI_CONTROL_FD"):
        monkeypatch.setenv(var, "0")
        assert slab_ffi._mpi_launcher_env() is not None, var
        monkeypatch.delenv(var)


def test_launcher_env_ignores_plain_slurm_batch_vars(monkeypatch):
    # A bare python inside an sbatch allocation has SLURM_* but no PMI
    # server — the exact launch shape that aborted (job 7884926).
    for var in list(__import__("os").environ):
        if var.startswith(("PMI_", "PMIX_")) or var == "HYDI_CONTROL_FD":
            monkeypatch.delenv(var)
    monkeypatch.setenv("SLURM_JOB_ID", "1234567")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    assert slab_ffi._mpi_launcher_env() is None


def test_singleton_probe_reports_a_dead_child_without_raising():
    ok, how = slab_ffi._mpi_singleton_probe(
        "import sys; sys.exit(13)\n", "MPI_Init_thread (synthetic)")
    assert ok is False
    assert "rc=13" in how


def test_singleton_probe_reports_a_live_child():
    ok, how = slab_ffi._mpi_singleton_probe(
        "pass\n", "MPI_Init_thread (synthetic)")
    assert ok is True
    assert "succeeded" in how


def test_singleton_probe_kills_a_hung_child():
    ok, how = slab_ffi._mpi_singleton_probe(
        "import time; time.sleep(600)\n", "MPI_Init_thread (synthetic)",
        timeout_s=2.0)
    assert ok is False
    assert "hung" in how
