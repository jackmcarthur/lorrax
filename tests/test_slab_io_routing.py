"""Parse-level tests for the SlabIO routing precedence chain and the
two-plan ``w_dyson_solver`` vocabulary (gw.gw_config).

Pins the AM deliverable that had zero test coverage (audit fix/zq
2026-07-28):

1. Precedence: explicit ``slab_io=<backend>`` (honored verbatim, probes
   skipped) > deprecated ``use_ffi_io=false`` (forces H5PY_ALLGATHER) >
   the ``slab_io=auto`` platform router (always runs otherwise).
2. ``use_ffi_io`` tri-state (unset / true / false) deprecation: each
   deck warns exactly ONCE, from the resolution site.
3. ``normalize_w_dyson_solver``: ``lu`` -> ``local`` with a
   DeprecationWarning; ``lstsq`` -> ValueError (removed, fails
   informatively); ``auto``/None alias to ``local``.
4. ``hartree_source`` validates at parse time.

Style follows tests/test_qp_solver_config.py: throwaway input files, the
platform routers monkeypatched to sentinels (no WFN, no probes, no GPU).
Warning-message assertions are substring matches, not exact text.
"""
from __future__ import annotations

import warnings

import pytest

import gw.gw_config as gw_config
from gw.gw_config import LorraxConfig, SlabIOBackend, normalize_w_dyson_solver


BASE_INPUT = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""

#: Sentinel the monkeypatched routers return — distinct from the
#: H5PY_ALLGATHER the use_ffi_io=false override forces, so router-vs-
#: override outcomes cannot be confused.
_ROUTED = SlabIOBackend.PHDF5_FFI


@pytest.fixture
def router_calls(monkeypatch):
    """Replace both platform routers with recording sentinels.

    ``from_input_file`` reads the routers as module globals, so
    monkeypatching the module attributes intercepts the real call.
    Returns the call log (one entry per router invocation).
    """
    calls: list[str] = []

    def _fake_cpu(print_fn):
        calls.append("cpu")
        return _ROUTED

    def _fake_gpu(print_fn):
        calls.append("gpu")
        return _ROUTED

    monkeypatch.setattr(gw_config, "_route_cpu_slab_io", _fake_cpu)
    monkeypatch.setattr(gw_config, "_route_gpu_slab_io", _fake_gpu)
    return calls


def _config(tmp_path, extra: str = "", name: str = "slab_io.in",
            print_fn=None):
    path = tmp_path / name
    path.write_text(BASE_INPUT + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=print_fn or (lambda *a, **k: None))


def _use_ffi_io_warnings(rec):
    return [w for w in rec
            if issubclass(w.category, DeprecationWarning)
            and "use_ffi_io" in str(w.message)]


# ---------------------------------------------------------------------------
# Precedence chain
# ---------------------------------------------------------------------------

def test_auto_default_calls_router(tmp_path, router_calls):
    cfg = _config(tmp_path)
    assert cfg.backend.slab_io is _ROUTED
    assert len(router_calls) == 1     # the router is THE default path


def test_explicit_slab_io_honored_verbatim_probes_skipped(
        tmp_path, router_calls):
    cfg = _config(tmp_path, "slab_io = phdf5_host\n")
    assert cfg.backend.slab_io is SlabIOBackend.PHDF5_HOST
    assert router_calls == []         # a named writer pays no probe


def test_explicit_slab_io_beats_use_ffi_io(tmp_path, router_calls):
    lines: list[str] = []
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cfg = _config(
            tmp_path, "slab_io = h5py_allgather\nuse_ffi_io = true\n",
            print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert cfg.backend.slab_io is SlabIOBackend.H5PY_ALLGATHER
    assert router_calls == []
    ws = _use_ffi_io_warnings(rec)
    assert len(ws) == 1
    assert "precedence" in str(ws[0].message).lower() \
        or "ignored" in str(ws[0].message).lower()
    assert any("ignored" in l for l in lines)


def test_use_ffi_io_false_forces_allgather(tmp_path, router_calls):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cfg = _config(tmp_path, "use_ffi_io = false\n")
    assert cfg.backend.slab_io is SlabIOBackend.H5PY_ALLGATHER
    assert router_calls == []         # the override beats the router
    ws = _use_ffi_io_warnings(rec)
    # Exactly ONE warning per deck (the double-warn from
    # read_lorrax_input + from_input_file was collapsed), and it names
    # what the key resolved to.
    assert len(ws) == 1
    assert "h5py_allgather" in str(ws[0].message)


def test_use_ffi_io_true_is_noop_still_routes(tmp_path, router_calls):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cfg = _config(tmp_path, "use_ffi_io = true\n")
    assert cfg.backend.slab_io is _ROUTED
    assert len(router_calls) == 1     # true does NOT bypass the router
    ws = _use_ffi_io_warnings(rec)
    assert len(ws) == 1
    assert "no-op" in str(ws[0].message)


def test_use_ffi_io_unset_never_warns(tmp_path, router_calls):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _config(tmp_path)
    assert _use_ffi_io_warnings(rec) == []


def test_invalid_slab_io_rejected(tmp_path, router_calls):
    with pytest.raises(ValueError, match="slab_io"):
        _config(tmp_path, "slab_io = bogus\n")


# ---------------------------------------------------------------------------
# H5PY_ALLGATHER is a refusal, not a fallback (owner ruling 2026-08-05,
# enforced 2026-08-06).  Three properties, and all three are load-bearing:
# the escape must survive (or somebody deletes the refusal), the refusal
# must fire above one process, and the routers must not have a third tier.
# ---------------------------------------------------------------------------

@pytest.fixture
def at_processes(monkeypatch):
    """Make ``_slab_io_process_count`` report N.

    Uses the LAUNCHER spelling rather than patching ``jax.process_count``
    on purpose: the max-of-all-evidence rule is the part that must not
    regress — a deck parsed before ``jax.distributed.initialize`` sees
    ``process_count()==1`` on every one of 16 ranks, and the launcher
    variable is the only thing standing between that and a green light.
    """
    def _set(n: int):
        monkeypatch.setenv("SLURM_NTASKS", str(n))
    return _set


def test_explicit_allgather_refuses_above_one_process(
        tmp_path, router_calls, at_processes):
    at_processes(4)
    with pytest.raises(RuntimeError) as ei:
        _config(tmp_path, "slab_io = h5py_allgather\n")
    msg = str(ei.value)
    assert "REFUSED at 4 processes" in msg
    assert "SLURM_NTASKS=4" in msg              # names the evidence
    assert "slab_io.md" in msg                  # names the doc


def test_deprecated_use_ffi_io_false_also_refuses_above_one_process(
        tmp_path, router_calls, at_processes):
    at_processes(16)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="use_ffi_io = false"):
            _config(tmp_path, "use_ffi_io = false\n")


def test_explicit_allgather_honored_at_one_process(
        tmp_path, router_calls, at_processes):
    at_processes(1)
    lines: list[str] = []
    cfg = _config(tmp_path, "slab_io = h5py_allgather\n",
                  print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert cfg.backend.slab_io is SlabIOBackend.H5PY_ALLGATHER
    # ...and it says so.  A silently-honoured escape is how the tier got
    # into multi-rank production logs in the first place.
    assert any("EXPLICIT" in l and "single-process" in l for l in lines)


@pytest.mark.parametrize("probe_reason,fix_marker", [
    ("unknown target: 'lorrax_phdf5_write' is not a target",
     "three-way"),                              # probe stage
    (None, "LAUNCHER fact"),                    # mpi stage
])
def test_router_refuses_instead_of_demoting(monkeypatch, at_processes,
                                            probe_reason, fix_marker):
    """No third tier: out of parallel tiers ⇒ RuntimeError, and the fix
    line is keyed to WHICH probe declined."""
    at_processes(8)
    import ffi.common.ffi_loader as loader
    if probe_reason is None:                    # handler present, MPI down
        monkeypatch.setattr(loader, "probe_target",
                            lambda t, p: (True, "available"))
        monkeypatch.setattr(gw_config, "_probe_mpi_bootstrap_ffi",
                            lambda platform="cpu": (False, "no PMI"))
    else:
        monkeypatch.setattr(loader, "probe_target",
                            lambda t, p: (False, probe_reason))
    monkeypatch.setattr(gw_config, "_probe_phdf5_host_tier",
                        lambda: (False, "no mpi4py"))
    for router in (gw_config._route_cpu_slab_io, gw_config._route_gpu_slab_io):
        with pytest.raises(RuntimeError) as ei:
            router(lambda *a, **k: None)
        msg = str(ei.value)
        assert "REFUSED" in msg
        assert fix_marker in msg
        assert "srun -n 1" in msg                 # the escape is stated


def test_router_at_one_process_selects_allgather_loudly(
        monkeypatch, at_processes):
    """The softening that keeps the refusal alive: at ONE process the same
    dead-end SELECTS the tier instead of refusing, because there the
    gather IS the per-rank write — and it says so, including that the
    identical configuration refuses at larger P."""
    at_processes(1)
    import ffi.common.ffi_loader as loader
    monkeypatch.setattr(loader, "probe_target",
                        lambda t, p: (False, "unknown target: nope"))
    monkeypatch.setattr(gw_config, "_probe_phdf5_host_tier",
                        lambda: (False, "no mpi4py"))
    lines: list[str] = []
    got = gw_config._route_gpu_slab_io(
        lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert got is SlabIOBackend.H5PY_ALLGATHER
    blob = " ".join(lines)
    assert "SINGLE-PROCESS" in blob
    assert "REFUSES" in blob                     # states the boundary
    assert "unknown target: nope" in blob        # states the tier-1 reason


# ---------------------------------------------------------------------------
# normalize_w_dyson_solver — the two-plan vocabulary
# ---------------------------------------------------------------------------

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


def test_deck_level_lu_warns_and_parses(tmp_path, router_calls):
    # Normalisation happens at PARSE time (fails/warns here, not
    # 20 minutes into a run).
    with pytest.warns(DeprecationWarning, match="lu"):
        cfg = _config(tmp_path, "w_dyson_solver = lu\n")
    assert cfg.backend.w_dyson_solver == "local"


def test_deck_level_lstsq_raises_at_parse(tmp_path, router_calls):
    with pytest.raises(ValueError, match="lstsq"):
        _config(tmp_path, "w_dyson_solver = lstsq\n")


# ---------------------------------------------------------------------------
# hartree_source — validated at parse time
# ---------------------------------------------------------------------------

def test_hartree_source_invalid_rejected(tmp_path, router_calls):
    with pytest.raises(ValueError, match="hartree_source"):
        _config(tmp_path, "hartree_source = bogus\n")


def test_hartree_source_valid_accepted(tmp_path, router_calls):
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
        assert gw_config._mpi_launcher_env() is not None, var
        monkeypatch.delenv(var)


def test_launcher_env_ignores_plain_slurm_batch_vars(monkeypatch):
    # A bare python inside an sbatch allocation has SLURM_* but no PMI
    # server — the exact launch shape that aborted (job 7884926).
    for var in list(__import__("os").environ):
        if var.startswith(("PMI_", "PMIX_")) or var == "HYDI_CONTROL_FD":
            monkeypatch.delenv(var)
    monkeypatch.setenv("SLURM_JOB_ID", "1234567")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    assert gw_config._mpi_launcher_env() is None


def test_singleton_probe_reports_a_dead_child_without_raising():
    ok, how = gw_config._mpi_singleton_probe(
        "import sys; sys.exit(13)\n", "MPI_Init_thread (synthetic)")
    assert ok is False
    assert "rc=13" in how


def test_singleton_probe_reports_a_live_child():
    ok, how = gw_config._mpi_singleton_probe(
        "pass\n", "MPI_Init_thread (synthetic)")
    assert ok is True
    assert "succeeded" in how


def test_singleton_probe_kills_a_hung_child():
    ok, how = gw_config._mpi_singleton_probe(
        "import time; time.sleep(600)\n", "MPI_Init_thread (synthetic)",
        timeout_s=2.0)
    assert ok is False
    assert "hung" in how
