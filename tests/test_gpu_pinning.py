"""One GPU per process — and the case where that used to be skipped.

``tests/conftest.py`` pins ``CUDA_VISIBLE_DEVICES`` at module scope, before
the first CUDA init.  That is the one moment a test cannot observe from
inside the same process, so the DECISION is a pure function in
``tests/harness.pin_one_gpu`` and this file constructs every case.

THE REGRESSION THIS PINS.  The pin used to be inside
``if PYTEST_XDIST_WORKER.startswith("gw"):`` — i.e. it only happened when
pytest-xdist was fanning out.  A NON-xdist run of the same suite on the
same node therefore saw all N GPUs, and three things break there:

  * the e2e gates' subprocesses build an N-device mesh and compare against
    1-GPU-frozen reference numbers;
  * SLATE refuses outright — MEASURED on Perlmutter 2026-08-07,
    ``slate.potrf: blas::get_device_count()=4 but JAX one-process-per-GPU
    model requires exactly 1`` — which killed 8 contract cells in the
    SERVICE-ONLY leg (``pytest services/distrib_la/tests``, which never
    loads this conftest);
  * ``services/distrib_la/tests/conftest.py`` has its own copy of the pin,
    but guarded on ``"jax" not in sys.modules``, and in a full-suite run
    ``testpaths = ["tests", "services"]`` collects ``tests/`` first, some
    module there imports jax during collection, and that copy is inert by
    the time it loads.  It cannot cover the full-suite leg by
    construction; only ``tests/conftest.py`` loads early enough.

CORRECTION, 2026-08-07 (step 4) — READ THIS BEFORE REUSING THE STORY
ABOVE.  An earlier revision of this docstring said the ``=4`` refusal was
what killed 8 cells in the FULL-SUITE ``-m distrib_la`` leg.  That was a
misdiagnosis and it is worth leaving the correction here, because "we
already fixed that" is how a second cause hides behind a first.  The
full-suite leg failed with ``blas::get_device_count()=0`` at EXACTLY ONE
visible device, before AND after this pin became unconditional.  Different
number, different cause: the two platform ``.so``s share ``libslate.so.2``
and ``libblaspp.so.2`` by SONAME, the host build's blaspp has no CUDA and
answers 0, and whichever library is dlopened first wins for both.  The fix
is a load-order rule in both loaders (``_open_cuda_before_host``); the
evidence is ``dladdr`` in both legs.

This pin is still correct and still free, and all three reasons above
stand — it just never was the cause of the eight.  ``test_a_bare_process_
still_gets_pinned`` is the cell that fails on the old code; the rest keep
the fan-out and the no-op cases from regressing in the other direction.
"""
from __future__ import annotations

import pytest

import harness


def _probe(n):
    return lambda: n


# ---------------------------------------------------------------------------
#  THE REGRESSION
# ---------------------------------------------------------------------------

def test_a_bare_process_still_gets_pinned():
    """No xdist worker id, four visible GPUs → pin to the first.

    RED ARM: this is exactly what the pre-fix guard returned nothing for.
    """
    assert harness.pin_one_gpu("0,1,2,3", "") == "0"
    assert harness.pin_one_gpu(None, "", probe=_probe(4)) == "0"


def test_the_pin_respects_slurms_selection_not_the_global_index():
    """SLURM hands a SUBSET; the pick indexes into that list, not into the
    node's physical devices.  Picking ``str(i)`` directly would hand a
    process a GPU its cgroup does not contain."""
    assert harness.pin_one_gpu("2,3", "") == "2"
    assert harness.pin_one_gpu("2,3", "gw1") == "3"


# ---------------------------------------------------------------------------
#  …without losing the fan-out it was written for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wid,want", [
    ("gw0", "0"), ("gw1", "1"), ("gw2", "2"), ("gw3", "3"),
    ("gw4", "0"),                    # more workers than GPUs: wrap
])
def test_xdist_workers_still_fan_out(wid, want):
    assert harness.pin_one_gpu("0,1,2,3", wid) == want


def test_a_non_worker_id_is_not_read_as_a_worker():
    """``master`` (the xdist controller) and any other spelling take the
    first device rather than crashing on ``int(...)``."""
    assert harness.pin_one_gpu("0,1,2,3", "master") == "0"
    assert harness.pin_one_gpu("0,1,2,3", "gw") == "0"


# ---------------------------------------------------------------------------
#  …and leaves alone what it must
# ---------------------------------------------------------------------------

def test_no_gpus_means_no_pin():
    """Every CPU leg.  Returning ``"0"`` here would hand JAX a device that
    does not exist; returning ``None`` leaves the environment untouched."""
    assert harness.pin_one_gpu(None, "", probe=_probe(0)) is None
    assert harness.pin_one_gpu(None, "gw2", probe=_probe(0)) is None


def test_an_explicit_mask_is_honoured_not_overridden():
    """``CUDA_VISIBLE_DEVICES=""`` means "no GPU, deliberately" — runtime/
    __init__.py reads exactly that spelling to decide a node has no GPU.
    Overriding it would un-mask a device the caller masked on purpose."""
    assert harness.pin_one_gpu("", "") is None
    assert harness.pin_one_gpu("", "gw1") is None


def test_the_probe_is_not_consulted_when_the_env_already_says():
    """A preset list is authoritative; probing nvidia-smi past it would let
    a device outside this process's cgroup back in."""
    def _explode():
        raise AssertionError("probe called despite a preset list")
    assert harness.pin_one_gpu("1", "gw3", probe=_explode) == "1"


# ---------------------------------------------------------------------------
#  …and the OTHER thing that decides whether SLATE can see the device
# ---------------------------------------------------------------------------
# One visible GPU is necessary and not sufficient.  ``liblorrax_ffi.so`` and
# ``liblorrax_ffi_host.so`` both carry NEEDED libslate.so.2 / libblaspp.so.2
# out of DIFFERENT builds, ld.so keys a loaded object by SONAME, and the host
# build's blas::get_device_count() is a compiled-in 0 -- so opening the host
# library first gives every CUDA SLATE handler a device count of ZERO at one
# visible device.  ``src/ffi/common/ffi_loader.py`` is the loader that lost
# that race (a module-scope probe_target(FLAT_K_TARGET, "cpu") in
# tests/test_fft_flat_k_numerics.py, at collection), so it is the copy of the
# rule that has to hold.  distrib_la's own copy is covered by
# services/distrib_la/tests/test_distrib_la_contract.py; this is the twin for
# lorrax's, because a rule enforced in two places with a test in one is a rule
# with a hole in it.

def _record_ffi_loader_open_order(monkeypatch, *, disable_rule=False,
                                  present=("CUDA", "cpu"), cuda_capable=True):
    """``ffi_loader.get_lib('cpu')`` with every native step stubbed; returns
    the platform library paths it dlopened, in order.

    ``cuda_capable`` stands in for the process's platform — the cells that
    own the PREDICATE construct its inputs directly (below); these own the
    ORDER."""
    import ctypes
    import pathlib
    from ffi.common import ffi_loader as F

    opened = []

    class _FakeLib:
        def __getattr__(self, name):
            return _FakeLib()

        def __setattr__(self, name, value):
            pass

    def _fake_locate(platform):
        if platform not in present:
            raise FileNotFoundError(f"no {platform} library in this fixture")
        return pathlib.Path("/fixture") / F._PLATFORMS[platform]["so_name"]

    def _fake_cdll(path, mode=0):
        opened.append(str(path))
        return _FakeLib()

    monkeypatch.setattr(F, "_LIBS", {})
    monkeypatch.setattr(F, "_LIB_PATHS", {})
    monkeypatch.setattr(F, "_CUDA_FIRST_TRIED", False)
    monkeypatch.setattr(F, "_locate_so", _fake_locate)
    monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)
    monkeypatch.setattr(F, "_set_argtypes", lambda lib, platform: None)
    monkeypatch.setattr(F, "_register_ffi_targets", lambda lib, plat: None)
    monkeypatch.setattr(F, "_process_can_use_cuda", lambda: bool(cuda_capable))
    if disable_rule:
        monkeypatch.setattr(F, "_open_cuda_before_host", lambda: None)

    F.get_lib("cpu")
    return opened


def test_lorraxs_loader_opens_the_cuda_library_before_the_host_one(monkeypatch):
    """CUDA-CAPABLE ARM: the load-order rule, in the loader that lost the race.

    RED ARM: disable ``_open_cuda_before_host`` and only the host library is
    opened — which is the process state that produced
    ``blas::get_device_count()=0``.
    """
    opened = _record_ffi_loader_open_order(monkeypatch, cuda_capable=True)
    assert opened == ["/fixture/liblorrax_ffi.so",
                      "/fixture/liblorrax_ffi_host.so"], opened


def test_the_lorrax_loader_open_order_cell_can_fail(monkeypatch):
    """The FALSE case, constructed."""
    opened = _record_ffi_loader_open_order(monkeypatch, disable_rule=True)
    assert opened == ["/fixture/liblorrax_ffi_host.so"], opened


def test_a_cpu_platform_process_opens_only_the_host_library(monkeypatch):
    """CPU-PLATFORM ARM: B1, in the loader that carries it for lorrax.

    A process whose jax platform is cpu has no CUDA SLATE handler for the
    order to protect, and dlopening the CUDA library there brought a second
    libslate/libblaspp AND a second phdf5 into it: ``tests/test_file_io.py``
    on a CPU-platform Perlmutter leg went 42 passed / 1 skipped at the two
    commits before the rule to three failures and ``Fatal Python error:
    Aborted`` at the commit that added it.
    """
    opened = _record_ffi_loader_open_order(monkeypatch, cuda_capable=False)
    assert opened == ["/fixture/liblorrax_ffi_host.so"], (
        f"a CPU-platform process dlopened {opened} — it must open the host "
        f"library and NOTHING else")


def test_the_lorrax_loader_cpu_platform_cell_can_fail(monkeypatch):
    """The FALSE case: same fixture, capability gate forced TRUE, and the
    CUDA library IS opened first.  Without this twin the cell above would
    stay green on any machine with no CUDA library to find — which is every
    WSL leg, i.e. green for a reason unrelated to the rule."""
    opened = _record_ffi_loader_open_order(monkeypatch, cuda_capable=True)
    assert opened == ["/fixture/liblorrax_ffi.so",
                      "/fixture/liblorrax_ffi_host.so"], opened


def test_a_cpu_only_tree_pays_nothing_for_the_rule(monkeypatch):
    """No CUDA library to locate: the host library still loads and the
    refusal is swallowed, so every CPU-only tree is untouched."""
    opened = _record_ffi_loader_open_order(monkeypatch, present=("cpu",))
    assert opened == ["/fixture/liblorrax_ffi_host.so"], opened


@pytest.mark.parametrize("env,devices,want", [
    ({"JAX_PLATFORMS": "cpu"},      True,  False),   # the leg B1 killed
    ({"JAX_PLATFORMS": "cpu,cuda"}, True,  False),
    ({"JAX_PLATFORMS": "cuda,cpu"}, True,  True),
    ({"JAX_PLATFORMS": "gpu"},      True,  True),
    ({},                            True,  True),    # every `lx test` leg
    ({},                            False, False),   # login node, WSL
    ({"CUDA_VISIBLE_DEVICES": ""},  True,  False),
    ({"CUDA_VISIBLE_DEVICES": "0"}, True,  True),
])
def test_the_lorrax_loader_cuda_capability_gate(monkeypatch, env, devices, want):
    """The predicate, every input constructed.

    Both loaders carry their own copy — the service may not import lorrax —
    so both get their own table, and the two tables are the same table.
    ``jax.default_backend()`` is deliberately NOT the signal: it
    INITIALIZES the XLA backend, so asking it inside a loader call would
    let the loader decide the process's platform instead of reading it.
    """
    from ffi.common import ffi_loader as F

    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(F, "_nvidia_device_visible", lambda: bool(devices))
    assert F._process_can_use_cuda() is want
