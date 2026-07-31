"""The FFT-memory contract: the planner's FFT term must MEASURE, not assume.

These tests pin the wiring that was broken until 2026-07-30, when
``common.fft_helpers.query_fft_peak_bytes`` promised in its docstring that its
result "includes cuFFT scratch" while its body computed only
``compiled.memory_analysis()`` — a number that structurally cannot contain the
cuFFT plan workspace (jaxlib's ``FftThunk`` takes that from a runtime scratch
allocator, outside XLA's buffer assignment).  ``runtime.aot_memory``, the one
module that actually queries ``cufftMakePlanMany``, had zero callers in
``src/`` while ``docs/architecture/memory-model.md`` said it was wired in.

Every test here runs on CPU: none of them needs a GPU, because each one
asserts about the *path taken*, not about a cuFFT number.  The GPU-only
assertions (that a real libcufft query returns non-zero, and that XLA:GPU
still emits a parseable ``fft`` op) live in ``tests/test_aot_memory.py``.

If any of these goes green while the wiring is broken, it is a void
instrument.  Each is written so that reverting the fix turns it RED:

* ``test_planner_fft_term_flows_through_aot_kernel_peak_bytes`` fails the
  moment ``query_fft_peak_bytes`` computes its own peak again.
* ``test_probe_compiles_the_helper_production_uses`` fails the moment the
  probe goes back to a form no production path executes.
* ``test_cufft_query_failure_is_announced_and_flagged`` fails if the
  unavailable-cuFFT case silently returns 0 again.
* ``test_no_mesh_fallback_announces`` fails if the analytic fallback goes
  quiet again.
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

import numpy as np                                              # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

import common.fft_helpers as fft_helpers                        # noqa: E402
import gw.gflat_memory_model as gmm                             # noqa: E402
import runtime.aot_memory as aot                                # noqa: E402


# A tiny FFT box: the probe compile has to be real (that is the point), but
# it does not have to be big.  128 complex elements.
_NK, _BC, _NS, _GRID = 2, 1, 1, (4, 4, 4)

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
    """Per-test isolation for the two process-global caches.

    ``_fft_workspace_cache`` would otherwise serve a previous test's number
    (making a monkeypatch look ineffective), and ``_announced`` would swallow
    the second test's announcement (making a loud path look silent).
    """
    fft_helpers._fft_workspace_cache.clear()
    aot._announced.clear()
    yield
    fft_helpers._fft_workspace_cache.clear()
    aot._announced.clear()


def _unit_mesh() -> Mesh:
    """A real 1x1 ('x','y') Mesh over the first available device."""
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))


def _fake_compiled(*, temp=0, arg=0, out=0, alias=0, hlo=_HLO_WITH_FFT):
    """Duck-typed ``jax.stages.Compiled`` for the parts aot_memory reads."""
    return SimpleNamespace(
        memory_analysis=lambda: SimpleNamespace(
            temp_size_in_bytes=temp, argument_size_in_bytes=arg,
            output_size_in_bytes=out, alias_size_in_bytes=alias),
        as_text=lambda: hlo,
    )


# ---------------------------------------------------------------------------
# The wiring: planner -> fft_helpers -> aot_memory (the cuFFT query path)
# ---------------------------------------------------------------------------


def test_planner_fft_term_flows_through_aot_kernel_peak_bytes(monkeypatch):
    """``_fft_box_bytes`` must obtain its number from the cuFFT-aware
    microservice, not from a private ``memory_analysis()`` of its own.

    Proof by sentinel: we replace ``aot_kernel_peak_bytes`` with a wrapper
    that adds a value no shape algebra could produce.  If the planner's FFT
    term contains that value, it went through the wrapper.
    """
    sentinel = 777_000_111  # not a multiple of 16; no byte formula yields it
    calls = []
    real = aot.aot_kernel_peak_bytes

    def spy(compiled, **kw):
        got = real(compiled, **kw)
        calls.append((got, kw))
        return aot.AotPeakBreakdown(
            compiled_peak=got.compiled_peak,
            cufft_scratch=got.cufft_scratch + sentinel,
            total=got.total + sentinel,
            cufft_measured=got.cufft_measured,
            fft_specs=got.fft_specs,
        )

    monkeypatch.setattr(aot, "aot_kernel_peak_bytes", spy)

    box = gmm._fft_box_bytes(nk=_NK, bc=_BC, ns=_NS, fft_grid=_GRID,
                             mesh_xy=_unit_mesh(), p_xy=1)

    assert calls, (
        "gflat_memory_model._fft_box_bytes did not call "
        "runtime.aot_memory.aot_kernel_peak_bytes.  The FFT-box term is back "
        "to a peak that cannot include cuFFT plan workspace.")
    assert box >= sentinel, (
        f"FFT-box term {box} does not carry the sentinel {sentinel}: the "
        f"planner is not using the value the cuFFT-aware path returned.")
    assert calls[0][1].get("platform") in ("cpu", "gpu", "cuda"), (
        "the caller must DECLARE the mesh platform: XLA:CPU also emits a "
        "parseable fft op, so the microservice cannot infer from the HLO "
        f"whether 0 cuFFT scratch is a fact or a gap.  Got kwargs "
        f"{calls[0][1]!r}.")


def test_probe_compiles_the_helper_production_uses(monkeypatch):
    """The probe must compile ``make_sharded_fftn_3d`` — the same factory
    every production FFT box goes through (``wfn_transforms._local_box_fft``,
    ``zeta_loader._do_disk_to_G``, the flat-k helpers).

    Modelling a different FFT form sizes cuFFT plans nothing ever builds;
    that is precisely what the per-axis ``custom_partitioning`` probe did.
    """
    seen = []
    real = fft_helpers.make_sharded_fftn_3d

    def spy(*a, **kw):
        seen.append((a, kw))
        return real(*a, **kw)

    monkeypatch.setattr(fft_helpers, "make_sharded_fftn_3d", spy)

    fft_helpers.query_fft_peak_bytes(
        input_shape=(_NK, _BC, _NS, *_GRID), fft_axes=(-3, -2, -1),
        sharding=NamedSharding(
            _unit_mesh(), P(None, ('x', 'y'), None, None, None, None)),
        dtype=jnp.complex128)

    assert seen, (
        "query_fft_peak_bytes did not compile make_sharded_fftn_3d — the "
        "memory model is probing an FFT form production does not run.")


def test_query_result_is_the_breakdown_total(monkeypatch):
    """``query_fft_peak_bytes`` returns ``AotPeakBreakdown.total`` verbatim
    — compiled peak PLUS cuFFT scratch, not just one of them."""
    monkeypatch.setattr(aot, "aot_kernel_peak_bytes", lambda compiled, **kw:
                        aot.AotPeakBreakdown(compiled_peak=1_000,
                                             cufft_scratch=2_000,
                                             total=3_000,
                                             cufft_measured=True,
                                             fft_specs=()))
    got = fft_helpers.query_fft_peak_bytes(
        input_shape=(_NK, _BC, _NS, *_GRID), fft_axes=(-3, -2, -1),
        sharding=NamedSharding(
            _unit_mesh(), P(None, ('x', 'y'), None, None, None, None)),
        dtype=jnp.complex128)
    assert got == 3_000


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


def test_no_mesh_fallback_announces(capsys):
    """Without a real ``Mesh`` the FFT box falls back to the analytic 4.0x
    box-copy bound — which does not model cuFFT workspace — and says so."""
    fake_mesh = SimpleNamespace(shape={'x': 4, 'y': 4})
    box = gmm._fft_box_bytes(nk=8, bc=16, ns=2, fft_grid=(10, 10, 10),
                             mesh_xy=fake_mesh, p_xy=16)
    # 16 bytes * bc * ns * n_rtot / p_xy * 4.0
    assert box == pytest.approx(16 * 16 * 2 * 1000 / 16 * 4.0)
    out = capsys.readouterr().out
    assert "memory-model" in out and "UNDER-predict" in out, (
        f"the analytic fallback was silent; stdout was {out!r}")


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
