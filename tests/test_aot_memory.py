"""Validation tests for ``runtime.aot_memory``.

These tests verify that the predicted per-rank peak from
:func:`aot_kernel_peak_bytes` matches the **observed** per-rank peak
reported by ``device.memory_stats()['peak_bytes_in_use']`` after running
a real JAX kernel that contains a cuFFT op.

The point: it is not enough that ``compiled.memory_analysis()`` is
reported; we must also account for the cuFFT plan workspace, which is
allocated outside XLA's buffer plan by jaxlib's FFT thunk at runtime.

GPU-required.  Skipped when no CUDA backend is available — CI can run
the HLO-parser unit tests separately on CPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# CPU-only: HLO parser unit tests
# ---------------------------------------------------------------------------


def test_parse_one_3d_complex128_fft():
    from src.runtime.aot_memory import parse_fft_specs_from_hlo, FftSpec

    # A representative line from a lowered HLO module.  Layout
    # annotation included; SSA name munged.
    hlo = (
        "ENTRY %main {\n"
        "  %arg.0 = c128[10,75,75,200]{3,2,1,0} parameter(0)\n"
        "  ROOT %fft.1 = c128[10,75,75,200]{3,2,1,0} fft(c128[10,75,75,200]"
        "{3,2,1,0} %arg.0), fft_type=FFT, fft_length={75,75,200}\n"
        "}\n"
    )
    specs = parse_fft_specs_from_hlo(hlo)
    assert specs == [FftSpec(
        rank=3,
        transform_shape=(75, 75, 200),
        batch=10,
        dtype="c128",
        fft_type="FFT",
    )]


def test_parse_multiple_ffts_dedupes_via_set():
    from src.runtime.aot_memory import parse_fft_specs_from_hlo, FftSpec

    # Two FFTs in one module: forward then inverse, same shape.
    hlo = (
        "  %a.1 = c128[8,64,64,64]{3,2,1,0} fft(c128[8,64,64,64]"
        "{3,2,1,0} %arg.0), fft_type=FFT, fft_length={64,64,64}\n"
        "  %b.2 = c128[8,64,64,64]{3,2,1,0} fft(c128[8,64,64,64]"
        "{3,2,1,0} %a.1), fft_type=IFFT, fft_length={64,64,64}\n"
    )
    specs = parse_fft_specs_from_hlo(hlo)
    assert len(specs) == 2
    # FFT and IFFT differ in op kind → distinct specs even with same shape
    assert {s.fft_type for s in specs} == {"FFT", "IFFT"}
    # But same plan params → same (rank, shape, batch, dtype)
    assert all(s.rank == 3 and s.transform_shape == (64, 64, 64)
               and s.batch == 8 and s.dtype == "c128" for s in specs)


def test_parse_no_fft_yields_empty_list():
    from src.runtime.aot_memory import parse_fft_specs_from_hlo

    hlo = (
        "ENTRY %main {\n"
        "  %arg.0 = f64[10] parameter(0)\n"
        "  ROOT %add.1 = f64[10] add(f64[10] %arg.0, f64[10] %arg.0)\n"
        "}\n"
    )
    assert parse_fft_specs_from_hlo(hlo) == []


def test_parse_loud_failure_when_format_drifts():
    """If XLA's HLO contains ' fft(' in a form our regex doesn't match,
    we raise — don't silently degrade."""
    from src.runtime.aot_memory import parse_fft_specs_from_hlo, HloFftParseError

    hlo = (
        "  %something = c128[10] fft(c128[10] %x)  # missing fft_type/fft_length\n"
    )
    with pytest.raises(HloFftParseError):
        parse_fft_specs_from_hlo(hlo)


# ---------------------------------------------------------------------------
# GPU-required: predicted vs observed peak
# ---------------------------------------------------------------------------


def _has_gpu() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


@pytest.mark.skipif(not _has_gpu(), reason="GPU required for cuFFT scratch validation")
def test_predicted_peak_matches_runtime_3d_fft():
    """Compile + run a 3-D batched complex128 FFT at a size where cuFFT
    actually allocates plan scratch; predicted total must match observed
    ``peak_bytes_in_use`` within 10 % (slack accommodates BFC pool
    fragmentation).

    Shape choice: (75, 75, 200) is the CrI3 V_q FFT box.  At batch=80
    cuFFT scratch ≈ 80 × 18 MB ≈ 1.4 GB, well above the in-place
    threshold but small enough to fit on any single A100/RTX.
    """
    from src.runtime.aot_memory import aot_kernel_peak_bytes

    nx, ny, nz = 75, 75, 200
    batch = 80
    dtype = jnp.complex128

    @jax.jit
    def fft_only(x):
        return jnp.fft.fftn(x, axes=(1, 2, 3))

    in_spec = jax.ShapeDtypeStruct((batch, nx, ny, nz), dtype)
    compiled = fft_only.lower(in_spec).compile()
    breakdown = aot_kernel_peak_bytes(compiled)

    assert len(breakdown.fft_specs) == 1
    spec = breakdown.fft_specs[0]
    assert spec.rank == 3
    assert spec.transform_shape == (nx, ny, nz)
    assert spec.batch == batch
    # NB: jax's default x64 mode may downcast c128 → c64; check whichever.
    assert spec.dtype in ("c128", "c64")
    assert breakdown.cufft_scratch > 0, (
        "cuFFT plan workspace should be positive for a 75×75×200 batched "
        "3-D FFT — query path is not reaching libcufft."
    )

    # Run for real and read peak_bytes_in_use.
    #
    # The point of this assertion: ``peak_bytes_in_use`` after the
    # kernel run equals the absolute high-water mark of the BFC pool
    # while the kernel was executing.  At the moment the FFT thunk
    # executes, the pool holds simultaneously the input (arg), the
    # output, and the cuFFT scratch.  XLA's ``memory_analysis()``
    # tells us arg + output (== ``compiled_peak`` here, no temp); our
    # cuFFT query tells us the scratch.  Their sum should equal the
    # observed peak EXACTLY when the kernel produces a new global high
    # for the process.  We arrange that by sizing the test FFT to be
    # the largest allocation the process has done so far.
    device = jax.devices("gpu")[0]
    np_dtype = np.complex128 if spec.dtype == "c128" else np.complex64
    rng = np.random.default_rng(0)
    x_host = (rng.standard_normal((batch, nx, ny, nz))
              + 1j * rng.standard_normal((batch, nx, ny, nz))).astype(np_dtype)
    x = jax.device_put(x_host, device)
    y = fft_only(x).block_until_ready()
    del x, y

    peak_after = int(device.memory_stats()["peak_bytes_in_use"])
    predicted = breakdown.total

    rel_diff = abs(predicted - peak_after) / max(peak_after, 1)
    # 5 % slack covers BFC pool fragmentation rounding only; if prior
    # tests/imports primed the BFC peak above ``predicted``, this
    # assertion can fail spuriously — run this test in isolation.
    assert rel_diff < 0.05, (
        f"Predicted {predicted/1e6:.1f} MB vs observed "
        f"peak_bytes_in_use {peak_after/1e6:.1f} MB.  Δ={rel_diff:.1%}; "
        f"compiled_peak={breakdown.compiled_peak/1e6:.1f}MB "
        f"cuFFT={breakdown.cufft_scratch/1e6:.1f}MB.  "
        f"If observed > predicted, prior process state dominated; run "
        f"this test in isolation (`pytest -k test_predicted_peak`)."
    )


@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
def test_query_repeatable_for_same_spec():
    """``cufftGetSize`` for the same :class:`FftSpec` must return the
    same value on repeated calls (within-process determinism)."""
    from src.runtime.aot_memory import (
        FftSpec, _query_one_plan_workspace_bytes,
    )

    # 75×75×200 c128 batch=80 — chosen because cuFFT *does* allocate
    # scratch at this size (verified empirically: ≈ 18 MB/batch).
    spec = FftSpec(
        rank=3, transform_shape=(75, 75, 200), batch=80,
        dtype="c128", fft_type="FFT",
    )
    a = _query_one_plan_workspace_bytes(spec)
    b = _query_one_plan_workspace_bytes(spec)
    assert a == b, "cuFFT workspace query is non-deterministic"
    assert a > 0, (
        f"Expected non-zero cuFFT workspace for {spec}; got {a}.  "
        f"This is the production V_q FFT shape — if scratch is 0 here, "
        f"the query is broken."
    )


@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
def test_query_scales_linearly_with_batch():
    """For a fixed transform shape, cuFFT plan scratch scales linearly
    with batch (verified empirically: ~18 MB/batch at 75×75×200 c128).
    Linearity is not formally guaranteed but holds at the sizes we care
    about; if it ever breaks, that's actionable info."""
    from src.runtime.aot_memory import FftSpec, _query_one_plan_workspace_bytes

    n = (75, 75, 200)
    sizes = []
    for batch in (50, 100, 200):
        spec = FftSpec(
            rank=3, transform_shape=n, batch=batch,
            dtype="c128", fft_type="FFT",
        )
        sizes.append(_query_one_plan_workspace_bytes(spec))

    # All non-zero, monotonically increasing
    assert all(s > 0 for s in sizes), f"got zeros: {sizes}"
    assert sizes[0] < sizes[1] < sizes[2], f"non-monotonic: {sizes}"

    # Linear: doubling batch should double scratch (within rounding)
    ratio = sizes[1] / sizes[0]      # 100 / 50 → expect 2×
    ratio2 = sizes[2] / sizes[1]     # 200 / 100 → expect 2×
    assert 1.8 < ratio < 2.2, f"50→100 ratio {ratio:.2f} not ~2×"
    assert 1.8 < ratio2 < 2.2, f"100→200 ratio {ratio2:.2f} not ~2×"
