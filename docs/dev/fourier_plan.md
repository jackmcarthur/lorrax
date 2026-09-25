# `LocalFourierPlan`: the local transform service

```python
from common.fourier_plan import LocalFourierPlan
y = LocalFourierPlan(N, axes, sign=-1, norm=n)(x)      # ≡ jnp.fft.fftn(x, axes=axes, norm=n)
y = LocalFourierPlan(N, axes, sign=+1, norm=n)(x)      # ≡ jnp.fft.ifftn(x, axes=axes, norm=n)
```

`N` holds the full extents aligned with `axes` (at most 3). `x` must be complex128. The plan computes

```text
y = R_out · F_{sign,norm} · E_in · x
```

- **Supports.** `in_support={ax: idx}` means `x` holds only the indices `idx mod N` on that axis, and every other index is zero. `out_support={ax: idx'}` returns only the indices `idx'`. A sphere enters through its tight bounding box, one index set per axis. An index repeated in an input support refuses. A support that lists the whole axis in order is no support: that axis is full.
- **`out_perm`** returns `jnp.transpose(y, out_perm)`.
- **`in_gather=(plane_from_col, n_col)`** (with `mesh`, over axes `(-2, -1)`, forward, `'backward'`) takes a cylinder `x (…, n_col)` and returns the `(n_b, n_c)` plane's transform without writing the zero plane. The slab form `plan(F, start, size)` reads `F[:, start:start+size]` in place. On CUDA this is mathdx mode 10. The plan chooses mode 10 or the XLA route once, when it is built, from the plane shape and the device's attributes: both axes must split into thread FFTs, and plane + row tables must fit the opt-in shared memory (`ffi.fft.plane_resident_bytes`). It announces the choice. The kernel's staging and threads come from the same attributes. Its rules are in [`ffi_layout.md`](../architecture/ffi_layout.md#plane-fft-with-gather-on-load-mode-10).

**Per-axis backend.** An axis of extent `N` with `n_in` input and `n_out` output indices is a GEMM with the stored matrix `A[j', j] = s·exp(sign·2πi·(idx'[j']·idx[j] mod N)/N)` (built once in float64, the phase index reduced exactly) iff

```text
full axis:       N ∈ full[kind]
supported axis:  N ∈ sup[kind]  (sup12[kind] for a plan over one or two axes)  and  n_in·n_out ≤ κ[kind]·N²
```

and otherwise joins the one FFT group. `GEMM_CROSSOVER` in `common/fourier_plan.py` holds `(full, sup, κ, sup12)` per `device_kind` prefix; A100 is `(∅, 16–128, 0.54, 32–128)`. An unknown device and CPU take the FFT on every axis (decisions.md 2026-09-25), so no plan loses `O(N log N)` scaling. Stages run shrinking GEMMs, the FFT group (embed, transform, restrict), then expanding GEMMs. `plan.stages` lists the choice.

**Legs.** The leg follows the platform the call is lowered for. On CUDA it is one `lorrax_fourier_plan_mathdx` custom call: cuBLAS GEMMs, the fused cuBLASDx pair when the two trailing axes are back-to-back GEMM axes, and cuFFT for the FFT group. [`ffi_layout.md`](../architecture/ffi_layout.md#local-fourier-plan-localfourierplan) owns its kernels, caches, kernel refusals and determinism. Elsewhere the leg is XLA ops: each GEMM axis is contracted in place (`tensordot`) and the FFT group is one `jnp.fft` call. Every CUDA build carries the plan's custom call, so on NVIDIA this leg runs a GEMM axis only when a caller forces it; cpu has no GEMM row.

**Cost.** The fused pair keeps the intermediate in shared memory: at Fe 25³ (K = 13) the whole sphere→box plan is 0.80× the cuBLAS chain and box→sphere 0.64× (0.40× and 0.48× the cuFFT arm). A GEMM axis wins only on a supported axis of bounded length whose support is small. There, one GEMM pass does what the FFT arm does in three: the zero-filling embed, the transform and the restriction (A100: 0.45–0.87 of the FFT arm at N 16–128). The GEMM costs `K = n_in·n_out/N` multiply-adds per grid element, while the FFT arm costs about one pass. So the row also bounds `K/N`: on A100 a supported axis takes the GEMM only when `n_in·n_out/N² ≤ 0.54`. In the sweep (2-D and 3-D, N 24–128, in and out supports) the GEMM wins 1.08–2.8× at every point at or below 0.54. The first loss above 0.54 is N = 128 at 0.55 (0.74–0.83×). The exception is 2-D out-support at small N (0.60–0.97× at N 16–30), so a plan over one or two axes takes a supported-axis GEMM only from N = 32. 3-D supports keep the floor at 16. A full axis always takes the FFT on A100, because cuFFT is about one HBM pass and a skinny ZGEMM is more. The plane form wins because the zero plane is never written: one read of the cylinder plus one write of the plane. It is 1.5–2.3× faster than the concatenate plus cuFFT 2-D at plane sides 24–100.

**Platforms.** Only A100 has a row; every other device takes the FFT until a row is measured there. The rows and the fused pair are kept for real-space GW, whose sphere↔box transforms (p→r, p′→r′, r→G) are 55–90% of the FFT time at Fe 8³. The GEMM rows cut that FFT time by 27–45% and the pair by a further 8–17% (an estimate; K1's benchmark measures it). Per device, as the ratio of FP64 tensor throughput to HBM bandwidth:
- **H100/H200:** 20 and 14 flop/B, against 12.5 on A100, so the supported-axis window should be at least as wide as A100's. The pair fits boxes to about 64³ in 227 KiB.
- **B200/GB200:** about 5 flop/B, so the window can shrink.
- **B300, sm_86/89/120:** with about 1 TF of FP64, a complex128 FFT is itself compute-bound. The empty full-axis column is the slot for an emulated ZGEMM row, not yet measured.

**Refusals.**
- `GATE fourier-plan-contract`: not complex128, or not 1–3 axes, at construction, on every platform.
- A missing `lorrax_fourier_plan_mathdx` at construction, when CUDA can lower the plan.
- For `in_gather`: `GATE plane-fft-dtype`, and `plane_from_col` entries outside `[0, n_col]`.
- The CUDA leg's own refusals (`GATE fourier-plan-int32`, `GATE mathdx-pair-wheel`) are in [`ffi_layout.md`](../architecture/ffi_layout.md#local-fourier-plan-localfourierplan).

**Reference patterns (NVIDIA).**
- Mode 10's structure is cuFFTDx's `05_fft_Xd/fft_3d_box_single_block`: the whole volume sits in shared memory, thread FFTs run per line, one sync per pass, and the store is coalesced. Mode 10 adds row sparsity and several planes per block.
- nvmath-python's FFT "truncation" (`examples/fft/truncation.py`) is a prefix copy followed by a full FFT. It is not a substitute for supports or the gather.

**Gates.** `tests/test_fourier_plan.py` runs on every leg the platform has. `tests/test_plane_fft_gather.py` and `tests/multi_device/plane_fft_gather_p4.py` cover the plane form.
