# The GN-PPM kernel's FFTs are already on the FFI, and they cost 32 ms

> **CORRECTED THE SAME DAY — read this first.** Everything measured on this page
> is reproducible and stands *on the deck it was measured on*. The
> generalisation does not. `tests/regression/gnppm_debug` is a fixture whose
> driver wall is 56 % bring-up and whose sigma stage is 5 % of the run, and the
> FFT share of the tau dispatch **rises steeply with k-point count**: 16 % at
> this deck's 9 k-points, **60 % at Si 4x4x4 (64 k)** and **85 % at Si 6x6x6
> (216 k)**, all at four processes on A100s under BFC@0.85 at HEAD `dc766220`.
> At Si 6x6x6 the flat-k FFT convolution is about **28 % of the driver wall**,
> not 0.07 %. In particular the section below headed "The share is not a
> small-deck artifact" is wrong: it is a small-deck artifact. The two decks it
> cites as agreeing agree by coincidence — one is this fixture, the other a
> CPU run at nb=128 — and neither varied the k-point axis that turns out to
> govern the share.
> Measured and adjudicated in
> `2026-08-11-gnppm-sigma-performance-claims-adjudicated.md`; evidence
> `/pscratch/sd/j/jackm/gnppmdecomp_0811/`.
>
> What this page settles and what it does not: the wiring question is still
> answered — the FFTs *are* already on the FFI, `LORRAX_FFT_FFI=0` still
> refuses, and both of the owner's safety conditions still hold as measured
> here. What it got wrong is the cost, and therefore the conclusion that there
> is no lever.

2026-08-11. A claim reached this lane from the MPA session: that the GN-PPM
kernel the MPA path uses wastes substantial time in FFT calls, and that the
in-tree FFT FFI would likely accelerate it. The owner attached two conditions
before any wiring — that the FFT FFI be safe beside cuSOLVERMp, and that it
work on at least some CPU backend.

Nothing was wired, because there is nothing left to wire. This file records
why, and the numbers that say so, so the next lane that hears the same claim
can stop at this page instead of re-measuring it.

## The kernel has been FFI-served since 2026-08-01

`gw/ppm_tau_kernel.py` builds its per-tau body from `make_flat_k_gw_conv` (the
fused single-call IFFT-multiply-FFT entry) when `LORRAX_FFT_FFI_FUSED` is on,
which is its default, and from `make_flat_k_ifftn`/`make_flat_k_fftn` when it
is not. All four of those factories live in `common/fft_helpers.py`, and every
one of them now ends in `ffi.fft`. The gated XLA twin that used to sit beside
them was deleted under the FFI-required ruling
(`docs/architecture/decisions.md`, 2026-08-01), which is also why
`LORRAX_FFT_FFI=0` refuses rather than falling back: there is no longer a
`jnp.fft` flat-k path to fall back to.

So the proposal's default — "FFI when available, refusal-named fallback to
`jnp.fft` otherwise" — describes a tree that was replaced ten days ago. The
refusal is the whole convention now, and it is the second half of the fallback
that no longer exists rather than the first.

## What the FFT actually costs, measured at four processes on a GPU

The in-tree Si production deck is COHSEX (`use_ppm_sigma = false`) and never
enters the tau kernel at all, so the profile was taken on
`tests/regression/gnppm_debug`, which is the only in-tree deck that runs it.
Four processes, one A100 each, mesh 2x2, allocator BFC@0.85, HEAD `0670fd34`,
`LORRAX_SIGMA_TAU_TIMING=1`:

| row | fused (default) | decomposed (`LORRAX_FFT_FFI_FUSED=0`) |
|---|---|---|
| `sigma.tau.dispatch`, 155 dispatches | 0.194 s | 0.216 s |
| the FFT step inside it | `GW_conv_ffi` **0.032 s** | `G_ifft` 0.018 + `V_ifft` 0.014 + `GW_mult_fft` 0.025 = **0.057 s** |
| FFT share of the tau dispatch | **16.5 %** | **26.4 %** |
| `sigma.exec` | 2.295 s | 0.689 s (warm compile cache; not comparable across legs) |
| driver wall | 44.908 s | 22.972 s |

The number that answers the claim **on this deck** is the first column's third
row against its last: the FFTs of the whole self-energy integration are **32 ms
of a 44.9 s run**, which is 0.07 %. Removing their cost entirely — not
accelerating it, removing it — would return seven parts in ten thousand **here**.

That last sentence is where this page went wrong, and it is worth naming the
mechanism rather than just the number. `gnppm_debug` is a 9-k-point fixture; its
sigma stage is 5 % of the driver wall and its bring-up is 56 %. Dividing a sigma
sub-row by *that* wall answers "how much of a fixture run is this", which is not
what anybody asking about sigma means. On the Si 6x6x6 deck the same division
gives ~28 %. See the correction banner at the top.

~~The share is not a small-deck artifact.~~ **IT IS.** The Frontera CPU
campaign measured 15.1 % of the staged tau dispatch decomposed and 7.6 % fused
at nb=128/P=64 (`wk_REL/FFI_EVIDENCE_AUDIT.md`, F25), and this lane read 16.5 %
here, and the agreement between those two numbers was taken as evidence that
the ratio is scale-free. It is not. Both decks are small in the axis that
matters — **k-points** — and neither varied it. Measured across a three-rung
ladder at four processes on A100s, BFC@0.85, HEAD `dc766220`:

| deck | k (full BZ) | centroids | FFT share of the staged tau dispatch |
|---|---|---|---|
| `gnppm_debug` (this page) | 9 | 399 | **16.1 %** |
| Si 4x4x4 | 64 | 1128 | **60.5 %** |
| Si 6x6x6 | 216 | 1104 | **84.9 %** (85.7 % decomposed) |

and the tau dispatch is a tenth of `sigma.exec` **here only** — at Si 6x6x6 the
tau kernel's device wall is 83 % of `sigma.exec`. The cost goes as
`n_tau · nk · mu_local · N_grid log N_grid`, so the k-point count is the axis
that governs it and the one every prior measurement held small.

What the claim was probably remembering is the number from *before* the FFI:
65 % of the staged tau dispatch went to FFT-adjacent layout churn on XLA:CPU
(`wk_REL/sigma_perf_results.md`). That was real, and the FFI is what closed it
— `sigma.exec` 272.0 s to 71.9 s, a 3.8x, with the XLA `fft` op count going
3 to 0 and the transposes 6 to 0.

## The two safety conditions, both measured

**cuSOLVERMp coexistence: safe, and it is what production already does.** Two
legs ran the same four processes with `w_dyson_solver = distributed`, so each
process drove cuSOLVERMp and the cuFFT flat-k handler in turn. Both printed
both, and both exited 0:

    [W solve] w_dyson_solver=distributed -> solve_lu: 'distributed' -> cusolvermp
              (ONE tile over the 2x2 mesh at P('x','y'), n=400)     # gnppm deck
              (ONE tile over the 2x2 mesh at P('x','y'), n=960)     # Si production deck
    [fft_ffi] flat-k 3-D FFTs -> cuFFT strided CUDA FFI handler
              (lorrax_mklfft_flat_k / lorrax_mklfft_gw_conv)

The Si 4x4x4 production deck run this way reproduces its frozen reference
exactly — max |delta| 0.0000 meV across all 3840 rows of `eqp_si_ref.dat` — so
the cuSOLVERMp-backed W solve is not merely tolerated beside the FFT FFI, it
leaves the BerkeleyGW anchor bit-unmoved. This is unsurprising in hindsight:
both handlers are compiled into the same `liblorrax_ffi.so`, the cuFFT side
binds to the XLA compute stream while the cuSOLVERMp side runs on `ctx->stream`
and joins with recorded events, and the only interaction the fleet has ever
measured between them is VRAM contention, which is what `MEM_FRACTION=0.90`
exists for (`ALLOCATOR_DECISION.md` section 5).

There is one shape where the FFT FFI genuinely cannot serve, and it is not
about cuSOLVERMp at all: an **in-process** multi-device mesh dies
`CUFFT_EXEC_FAILED` at every size. Production never has that shape — it is one
process per GPU — and `tests/harness.mesh_subprocess_env` already refuses the
path by name in the mesh child rather than letting it abort.

**CPU backend: the handler exists and is the FFI's original platform, but a
missing `.so` is a refusal, not a fallback.** The host handler is the FFTW3
ABI (`src/ffi/cpp/mklfft/fft_flat_k_ffi.cc`), it serves `cpu` meshes, and every
speedup quoted above was measured on it. What it does *not* do is degrade
gracefully. On a tree with no `liblorrax_ffi_host.so` — a plain WSL checkout at
this HEAD, `JAX_PLATFORMS=cpu` — both factories refuse by name:

    RuntimeError: The required FFTW3-ABI host backend is unavailable: FFI target
    'lorrax_mklfft_flat_k' is unusable on platform 'cpu': ...
    FfiLibraryNotBuilt: Could not locate liblorrax_ffi_host.so (platform=cpu).
    Build with: bash src/ffi/cpp/build_host.sh

So the condition as it was phrased — that the kernel must still run with no GPU
and no FFI `.so`, falling back to `jnp.fft` — cannot be met without reverting
the 2026-08-01 ruling. The condition as it was *meant* — that some CPU backend
is served — is met, and has been for a week.

## One thing worth an owner's eye

Every leg printed this, and it is not this lane's to fix:

    [distrib_la] NOTE: .../merge_ckpt_2026-08-08/build_dev/liblorrax_ffi.so
    carries no handler-ABI stamp, so it cannot be checked against this package
    (abi=3).  Pre-2026-08-08 build.

That is the `.so` pin several 2026-08-11 lanes are using on `main` after the
kchunk signature change moved the ABI to 3. Nothing on these legs touches the
kchunk read path, so nothing failed, but the pair in general use is unstamped
and unverifiable against the package it is loaded into. The front-door build
chain that would produce a stamped replacement is the open blocker row 2 in
`PIPELINE_HEALTH.md` (`build_host.sh` refuses without a SLATE
`gpu_backend=none` install; `build.sh` refuses until `LORRAX_NVHPC_ROOT` names
a stage), and this lane did not hand-build around it.

## Evidence

`/pscratch/sd/j/jackm/gnppmfft_0811/` — `EVIDENCE.md`, the four leg logs under
`_logs/`, and the run directories `gA` (fused + stage timing), `gB` (decomposed
+ stage timing), `gD` (GN-PPM deck, distributed W solve), `sC` (Si production
deck, distributed W solve). One `lx batch`, `-N 1 -G 4 -n 4 -P 1`, allocation
56657915, four legs 4/4 exit 0, each asserting `device_count=4` and
`device mesh is 2x2` on its own startup block. No source file changed, no
reference re-frozen, no `.so` built, no tolerance moved.
