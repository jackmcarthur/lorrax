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

1. Both retired deck keys are accepted and IGNORED with a warning, not
   refused: they never changed a number, only which library moved the
   bytes, so an old deck keeps running and says what it lost.
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
])
def test_retired_transport_keys_are_ignored_not_refused(tmp_path, line):
    """An old deck still parses, whatever it asked for.

    Including ``slab_io = h5py_allgather``, which used to REFUSE above one
    process.  That refusal was about a tier that could not hold the run's
    own arrays; with the tier gone there is nothing to refuse -- the key
    names a choice that no longer exists, which is a stale deck, not a
    dangerous one.  The run gets the one transport either way.
    """
    cfg = _config(tmp_path, line + "\n")
    assert cfg is not None


def test_config_carries_no_transport_selector(tmp_path):
    """The selector is gone from the resolved config, not just the deck.

    ``BackendConfig.slab_io`` and the ``LorraxConfig.use_ffi_io`` property
    were the two places a caller could still branch on transport after
    parsing.  Both answered a question with one answer.
    """
    cfg = _config(tmp_path)
    assert not hasattr(cfg.backend, "slab_io")
    assert not hasattr(cfg, "use_ffi_io")


def test_unknown_key_check_still_sees_real_typos(tmp_path):
    """Retiring a key must not blunt the typo check that guards the rest.

    ``slab_io`` is exempted by name via ``_LEGACY_DECK_KEYS``; a
    neighbouring misspelling must still be reported.
    """
    with pytest.raises(ValueError, match="slab_iox"):
        _config(tmp_path, "strict_keys = true\nslab_iox = phdf5_ffi\n")
    # ...and the exempted spelling still parses under the same strictness.
    assert _config(tmp_path, "strict_keys = true\nslab_io = phdf5_ffi\n")


def test_gw_config_no_longer_defines_the_tier_vocabulary(tmp_path):
    """The enum, the router and the seven refusals are gone from L1."""
    for name in ("SlabIOBackend", "resolve_slab_io_backend",
                 "_route_cpu_slab_io", "_route_gpu_slab_io",
                 "_validate_slab_io_key", "_refuse_explicit_h5py_allgather",
                 "_refuse_slab_io_no_parallel_writer",
                 "_slab_io_no_parallel_writer", "_probe_phdf5_host_tier"):
        assert not hasattr(gw_config, name), name


def test_slab_io_module_exposes_one_transport():
    """file_io.slab_io's public surface: no backend selector anywhere."""
    import inspect
    from file_io import slab_io
    params = inspect.signature(slab_io.SlabIO.__init__).parameters
    assert set(params) == {"self", "path", "mode", "mesh"}, params
    assert not hasattr(slab_io, "SlabIOBackend")


def test_deleted_backend_modules_are_gone():
    """The two retired backends are deleted, not merely unreachable.

    The point of the 2026-08-06 change: a code path that cannot legally be
    reached is dead code wearing a safety label.  Import must fail.
    """
    import importlib
    for mod in ("file_io._slab_io_allgather", "file_io._slab_io_mpi_host"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)


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
