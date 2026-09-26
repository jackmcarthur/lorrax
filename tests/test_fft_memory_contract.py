"""The FFT-memory microservice: ``runtime.aot_memory`` must MEASURE, not assume.

``compiled.memory_analysis()`` structurally cannot contain the cuFFT plan
workspace (jaxlib's ``FftThunk`` takes that from a runtime scratch allocator,
outside XLA's buffer assignment), so :func:`runtime.aot_memory.aot_kernel_peak_bytes`
adds a ``cufftMakePlanMany`` query on top.  These cells pin its arithmetic
and its failure policy.

Every test here runs on CPU: each asserts about the *path taken*, not about a
cuFFT number.  The GPU-only assertions (that a real libcufft query returns
non-zero, and that XLA:GPU still emits a parseable ``fft`` op) live in
``tests/test_aot_memory.py``.

* ``test_cufft_query_failure_is_announced_and_flagged`` fails if the
  unavailable-cuFFT case silently returns 0 again.
* ``test_announce_once_speaks_then_dedupes`` fails if the announcement path
  prints unconditionally or never.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# The suite is normally run with PYTHONPATH=<repo>/src; make the file
# self-sufficient so it also works from a bare checkout.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import runtime.aot_memory as aot                                # noqa: E402


# A representative optimized-HLO line carrying an XLA fft op.
_HLO_WITH_FFT = (
    "ENTRY %main {\n"
    "  %arg.0 = c128[8,4,4,4]{3,2,1,0} parameter(0)\n"
    "  ROOT %fft.1 = c128[8,4,4,4]{3,2,1,0} fft(%arg.0), fft_type=FFT, "
    "fft_length={4,4,4}\n"
    "}\n"
)


@pytest.fixture(autouse=True)
def _clean_module_state():
    """Per-test isolation: ``_announced`` would otherwise swallow the second
    test's announcement (making a loud path look silent).
    """
    aot._announced.clear()
    yield
    aot._announced.clear()


def _fake_compiled(*, temp=0, arg=0, out=0, alias=0, hlo=_HLO_WITH_FFT):
    """Duck-typed ``jax.stages.Compiled`` for the parts aot_memory reads."""
    return SimpleNamespace(
        memory_analysis=lambda: SimpleNamespace(
            temp_size_in_bytes=temp, argument_size_in_bytes=arg,
            output_size_in_bytes=out, alias_size_in_bytes=alias),
        as_text=lambda: hlo,
    )


# ---------------------------------------------------------------------------
# The microservice's own arithmetic and failure policy
# ---------------------------------------------------------------------------


def test_breakdown_adds_cufft_scratch_to_compiled_peak(monkeypatch):
    """``total = (temp + arg + out - alias) + cuFFT workspace``."""
    monkeypatch.setattr(aot, "_query_one_plan_workspace_bytes",
                        lambda spec: 4_096)
    got = aot.aot_kernel_peak_bytes(
        _fake_compiled(temp=100, arg=200, out=300, alias=50), platform="gpu")
    assert got.compiled_peak == 550
    assert got.cufft_scratch == 4_096
    assert got.total == 4_646
    assert got.resident_increment == 4_446  # temp + out - alias + cuFFT
    assert got.cufft_measured is True
    assert len(got.fft_specs) == 1


def test_cufft_query_failure_is_announced_and_flagged(monkeypatch, capsys):
    """An unavailable cuFFT query ON A CUDA BACKEND must flag itself AND
    speak.

    The pre-fix behaviour returned a bare 0 that was indistinguishable from
    "this kernel has no FFTs", which is how a >13.7 GB term went missing at
    the CrI3 V_q box.
    """
    def boom(spec):
        raise aot.CufftQueryError("no libcufft in this process")

    monkeypatch.setattr(aot, "_query_one_plan_workspace_bytes", boom)
    got = aot.aot_kernel_peak_bytes(_fake_compiled(temp=10, out=10),
                                    platform="gpu")

    assert got.cufft_scratch == 0
    assert got.cufft_measured is False, (
        "cuFFT scratch of 0 from a FAILED query must not be reported as a "
        "measurement.")
    out = capsys.readouterr().out
    assert "memory-model" in out and "UNAVAILABLE" in out, (
        f"the demotion was silent; stdout was {out!r}")


def test_cpu_platform_zero_scratch_is_exact_and_silent(monkeypatch, capsys):
    """On a non-CUDA platform, 0 cuFFT scratch is a FACT, not a demotion.

    This is a false-alarm regression test.  XLA:CPU keeps the ``fft`` op in
    its optimized HLO exactly as XLA:GPU does (measured on jax 0.9.1, job
    7882062) — my first cut inferred "there is an fft op, so cuFFT must be
    involved", which made every CPU run print an alarming (and wrong) low-
    bound warning on every FFT-box query.  The platform decides, not the HLO.
    """
    def boom(spec):                       # must never be reached on CPU
        raise AssertionError("cuFFT query attempted on a non-CUDA platform")

    monkeypatch.setattr(aot, "_query_one_plan_workspace_bytes", boom)
    got = aot.aot_kernel_peak_bytes(_fake_compiled(temp=64), platform="cpu")

    assert len(got.fft_specs) == 1, "the CPU HLO does carry a parseable fft op"
    assert got.cufft_scratch == 0
    assert got.cufft_measured is True, (
        "0 cuFFT scratch on a platform with no cuFFT is exact, not a "
        "demotion.")
    assert capsys.readouterr().out == "", "no warning belongs on a CPU run"


def test_no_fft_ops_is_an_exact_zero():
    """A kernel with no FFT at all costs no FFT scratch anywhere."""
    got = aot.aot_kernel_peak_bytes(
        _fake_compiled(temp=1, hlo="ROOT %add.1 = f64[10] add(%a, %b)\n"),
        platform="gpu")
    assert got.fft_specs == ()
    assert got.cufft_scratch == 0
    assert got.cufft_measured is True


def test_hlo_format_drift_raises_rather_than_reporting_zero():
    """A ``fft(`` the regex cannot parse must not read as "no FFTs"."""
    with pytest.raises(aot.HloFftParseError):
        aot.aot_kernel_peak_bytes(
            _fake_compiled(hlo="  %x = c128[10] fft(c128[10] %y)\n"),
            platform="gpu")


# ---------------------------------------------------------------------------
# The announcement machinery itself (an instrument that must be able to fail)
# ---------------------------------------------------------------------------


def test_announce_once_speaks_then_dedupes(capsys):
    """Positive AND negative control for the announcement path: a fresh key
    prints, a repeated key does not, and a different key prints again.

    Without the negative half, an ``announce_once`` that printed
    unconditionally (or never) would still look fine to the tests above.
    """
    aot.announce_once("k1", "first message")
    first = capsys.readouterr().out
    assert "first message" in first and "memory-model" in first

    aot.announce_once("k1", "first message")
    assert capsys.readouterr().out == "", "repeated key must not re-print"

    aot.announce_once("k2", "second message")
    assert "second message" in capsys.readouterr().out
