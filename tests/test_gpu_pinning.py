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
    full-suite ``-m distrib_la`` leg while the SAME cells were green in
    the service-only leg;
  * ``services/distrib_la/tests/conftest.py`` has its own copy of the pin,
    but guarded on ``"jax" not in sys.modules``, and in a full-suite run
    ``testpaths = ["tests", "services"]`` collects ``tests/`` first, some
    module there imports jax during collection, and that copy is inert by
    the time it loads.  It cannot cover the full-suite leg by
    construction; only ``tests/conftest.py`` loads early enough.

``test_a_bare_process_still_gets_pinned`` is the cell that fails on the
old code.  The rest keep the fan-out and the no-op cases from regressing
in the other direction while fixing it.
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
