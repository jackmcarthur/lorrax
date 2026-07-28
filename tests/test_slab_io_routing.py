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
