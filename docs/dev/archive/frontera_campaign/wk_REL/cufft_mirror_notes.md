# cuFFT flat-k FFI mirror notes (2026-07-29) — LORRAX_FFT_FFI platform-uniform

Tree: /work2/08271/jackmc/frontera/lorrax @ 5894dcd, WORKING TREE ONLY (not
committed), owner-approved workstream.  Mirrors the MKL FFT (DFTI API) host
backend (wk_REL/docs/ffi_fft_proto_notes.md) onto the CUDA platform so
`LORRAX_FFT_FFI` / `LORRAX_FFT_FFI_FUSED` behave identically on cpu and CUDA
meshes.

## Design

* **Same target names, per-platform symbols** (the phdf5 platform_seam.h
  registration split): the CUDA lib exports `CufftFlatKCudaFfi` /
  `CufftGwConvCudaFfi` registered under platform="CUDA" against the SAME
  target strings `lorrax_mklfft_flat_k` / `lorrax_mklfft_gw_conv` as the
  host lib, so every `jax.ffi.ffi_call` site in `common/fft_helpers.py`
  stays platform-agnostic and resolves by lowering platform.
* **cufftPlanMany64 advanced data layout = the DFTI stride-descriptor
  analog**: `inembed = onembed = {d0,d1,d2}`, `istride = ostride = T`
  (trail stride), `idist = odist = 1`, `batch = T` — element (x,y,z) of
  batch b sits at `((x·d1+y)·d2+z)·T + b`, exactly the (nk, *trail)
  dot-layout tile.  One-to-one (batch ≤ T), no overlap.  NO compact-chunk
  staging on GPU: batch-at-distance-1 means consecutive threads read
  consecutive addresses (coalesced); the host engine's per-thread L2 chunk
  was a CLX cache artifact, not part of the contract.
* **Fused G·W multiply = a plain CUDA kernel, not a cuFFT callback**
  (owner instruction).  Because the Frontera pip toolchain has ptxas but NO
  nvcc driver and the CUDA .so build deliberately has no CUDA-language step
  (house fact reused: SPEEDUP_SCORECARD.md AE.4b "compiles these TUs with
  g++ against the CUDA headers, not nvcc"), the two kernels (norm scale;
  broadcast multiply) live in an NVRTC source string compiled ONCE per
  process at first use for the compute capability queried from the runtime
  device (also neutralizes the CMAKE_CUDA_ARCHITECTURES=80 default vs
  rtx-dev sm_75 mismatch — nothing is baked at build time), cubin
  preferred over PTX (no driver-JIT ISA dependence), loaded via the driver
  API resolved with dlsym (no link-time libcuda; the dlsym pattern from
  blacs_grid.h's MKL pin).
* **Norm scaling matches jnp.fft exactly by construction**: the Python
  helper still computes the one true scale (`_ffi_fft_scale`, unchanged);
  cuFFT scales neither direction, so the plain handler applies `scale`
  with an elementwise kernel (skipped at 1.0) and the fused handler folds
  `scale_i²·scale_f` into the multiply kernel (scales commute with the
  linear FFT — same value-level reassociation class as the host handler's
  scale-fold; gated, never claimed bit-exact).
* **In-place**: same `input_output_aliases={0:0}`; granted alias ⇒
  `idata == odata` ⇒ cuFFT in-place exec (identical advanced layouts).
* **Memory policy**: plans created with auto-allocation OFF; ONE shared
  grow-only workspace arena sized to the largest plan request + the fused
  handler's V_R arena (nk·mx·my·16 B — the sharded W tile, never a μ²
  object).  Both cudaMalloc'd outside XLA — sanctioned by the production
  env (ffi_env.sh: `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`,
  `PREALLOCATE=false`) — and logged under `LORRAX_CUFFT_LOG`.  Arena
  growth `cudaDeviceSynchronize()`s before freeing the old block.
* **Concurrency**: one mutex serializes plan-cache/arena/enqueue; device
  ordering relies on XLA's single compute stream per module — stated
  honestly (same single-consumer class as the host V_R arena lock).
* **fft_helpers routing**: `_require_fft_ffi` now platform-dispatches
  (cpu → "cpu", gpu/cuda → "CUDA", anything else refuses); announce is
  per (target, platform); refusal quotes the `probe_target` reason.  The
  helpers stay THE single FFT entry point (owner rule) — no call-site
  changes anywhere.

## Files touched (working tree)

- src/ffi/cufft/cpp/fft_flat_k_cuda_ffi.cc     NEW: the two CUDA handlers
- src/ffi/common/cpp/CMakeLists.txt            LORRAX_FFI_HAVE_CUFFT block
                                               (find cufft/nvrtc, TU, link)
- src/ffi/common/ffi_loader.py                 _CUDA_TARGET_SYMBOLS entries
- src/common/fft_helpers.py                    platform dispatch + announce
- config/frontera/stage_ffi_deps.sh            merge nvidia-cufft +
                                               nvidia-cuda-nvrtc wheels into
                                               cuda_root (they were absent)
Harness (wk_REL): cufft_unit_gate.py, cufft_gpu_gate.sbatch.

## Historical record reused (owner directive: grep first)

- SPEEDUP_SCORECARD.md:4952 (AE.4b) — the CUDA .so is compiled by g++
  against CUDA headers; **no nvcc exists** in the pip toolchain (checked:
  `site-packages/nvidia/cuda_nvcc/bin/` contains only `ptxas`).  This is
  what forces NVRTC for the device kernels.
- AE.4b also records the per-platform same-name build gate pattern
  (nm -D for the CUDA names, no *HostFfi leak) — copied in the build cell.
- ffi_env.sh already puts every `site-packages/nvidia/*/lib` on
  LD_LIBRARY_PATH (libcufft.so.11 / libnvrtc.so.12 resolve at dlopen) and
  sets cuda_async + preallocate=false (device-side cudaMalloc coexistence
  is the documented pattern, src/ffi/AGENTS.md env table).
- wk_REL/harness/rtx_smoke_merged.sbatch — the rtx-dev harness skeleton (stage
  cell on a compute node: containers cannot run on login nodes; bare-input
  g108/g402 cells; eqp0 gates vs run_am_p1 / run_400c at 2e-5 eV).
- No prior NVRTC/driver-API handler exists in the record (grepped
  SPEEDUP_SCORECARD, SESSION_REPORT*, wk_*/ for nvrtc|cuModule|driver api|
  libcuda — only venv file listings hit).

## Gates (rtx-dev, 2 nodes / 2 h / PHY25006) — cufft_gpu_gate.sbatch

1. stage + build cell: fresh stage at $WORK/lorrax_ffi_cufft_cuda; build
   from the working tree with LORRAX_FFI_PHDF5=1; nm -D symbol gate
   (CufftFlatKCudaFfi, CufftGwConvCudaFfi + the phdf5/eigh names intact);
   readelf DT_NEEDED shows libcufft/libnvrtc.
2. unit gate (1 GPU): handlers vs jnp.fft (GPU) at production LOCAL shapes
   (16,2,624,2,624)/(16,624,624), fused vs decomposed, round trip,
   donation smoke (granted-alias in-place exec), odd shapes (3,2,1)×(5,7,3)
   all four norms.  Gate 1e-13, expect ~1e-15/bit-class.
3. g108+g402 bare-input GPU smokes, legs OFF and ON
   (ON = LORRAX_FFT_FFI=1 + LORRAX_FFT_FFI_FUSED=1: τ kernel rides
   gw_conv, χ0/W/ζ ride flat_k) run concurrently one leg per node.
   Gates: eqp0 vs pinned CPU baselines (run_am_p1 / run_400c, 2e-5 eV)
   for BOTH legs; eqp0 ON-vs-OFF at 1e-6 (engine-swap value parity);
   chi/W/sigma exec timing rows OFF vs ON.

## RESULTS (job 7879275, rtx-dev c196-[011-012], 2026-07-29)

* **stage**: PASS — cufft.h / nvrtc.h / libcufft.so now staged into
  cuda_root (the stage_ffi_deps.sh additions verified by its sanity block).
* **build**: PASS — clean configure ("cufft: strided flat-k FFT handlers
  ON"), only the known jaxlib-header warning noise (AE.4b class).  Symbol
  gate: CufftFlatKCudaFfi OK, CufftGwConvCudaFfi OK; PhdfWriteFfi /
  PhdfReadKchunkUnionFfi / EighMpFfi intact (no regression to the existing
  CUDA targets).  DT_NEEDED gained exactly libcufft.so.11 + libnvrtc.so.12.
* **unit gate (1 RTX 5000): PASS, all 15 cells.**
  - G tile (16,2,624,2,624) ifftn/fftn ortho: **0.0 (bit-identical to
    jnp/cuFFT-minor-most at THIS kgrid** — 4·4·1 pure radix + exact 0.25
    scale, same as the CPU round; general claim stays ~1e-15).  Round trip
    2.2e-16; fused gw_conv vs decomposed 2.9e-16 (scale-fold rounding, as
    designed); donation smokes 0.0/2.7e-16 (alias granted: handler logged
    `inplace=1`); odd (3,2,1)×(5,7,3) all four norms 0…3.7e-16.
  - NVRTC compiled the two kernels for **sm_75** at first use (19816 B
    cubin) — the g++-only build + runtime-arch design works on Turing.
  - cuFFT workspace for these geometries: **0 bytes** (plan-internal);
    V_R arena 99.7 MB (= the (nk, mx, my) W tile, as specified).
  - Timing (best of 5, production local shapes, eager-jit dispatch):
    | path | G ifft | gw_conv |
    |---|---|---|
    | XLA (transpose + minor-most cuFFT) | 12.63 ms (63.1 GB/s) | 23.89 ms |
    | FFI strided | **7.08 ms (112.6 GB/s)** | **13.53 ms (117.8 GB/s)** |
    ~1.8× both — the GPU pays its layout transposes in fast HBM, so the
    win is real but far milder than the CPU round's 16-38×; the point of
    this workstream is platform uniformity, not a GPU speed claim.
* **g108+g402 bare-input GPU smokes, OFF (c196-011) / ON (c196-012)
  concurrent: ALL FOUR runs rc=0** (walls: off 64/45 s, on 154/45 s — the
  ON g108 excess is first-compile skew: 14 fresh XLA modules for the FFI
  variants + the one-time NVRTC compile; the persistent compile cache
  already held every OFF variant).  Handler telemetry (LORRAX_CUFFT_LOG,
  rank0): both targets fired in BOTH ON runs — `flat_k` (χ0/ζ; T=11664 /
  163216) and `gw_conv` (τ kernel via LORRAX_FFT_FFI_FUSED; Tg=11664
  a=2,mx=54 / Tg=163216 a=2,mx=202), `aliased=1` everywhere (XLA granted
  the in-place alias in production), V_R arena 0.7 / 10.4 MB, cuFFT
  workspace 0 B at these geometries, NVRTC sm_75 cubin once per process.

  **eqp0 gates (multidev_compare, 5160 values each): ALL SIX PASS.**
  | gate | max abs diff | verdict |
  |---|---|---|
  | g108 OFF vs CPU run_am_p1 (2e-5 eV) | 0.00e+00 | PASS |
  | g108 ON  vs CPU run_am_p1 (2e-5 eV) | 0.00e+00 | PASS |
  | g402 OFF vs CPU run_400c (2e-5 eV)  | 1.00e-09 | PASS |
  | g402 ON  vs CPU run_400c (2e-5 eV)  | 1.00e-09 | PASS |
  | g108 ON vs OFF, engine swap (1e-6)  | **0.00e+00** | PASS |
  | g402 ON vs OFF, engine swap (1e-6)  | **0.00e+00** | PASS |
  The ON-vs-OFF zeros are TEXT-PRECISION identity (9-decimal files,
  ~5e-10 eV resolution) — same caveat as the CPU round; the value-level
  evidence is the unit gate's 0.0…3.7e-16 raw-tile errors.  No GPU h5
  tensor gate was run (no pinned GPU h5 baseline exists for this deck).

  **Production timing (steady-state rows)**: parity-class at this deck
  scale — g402 sigma.exec 6.169 (ON) vs 6.009 (OFF); g108 5.368 vs
  5.439; second-pass chi.exec 0.077 both legs.  First-pass ON rows carry
  the compile/NVRTC skew (g108 chi.exec 4.669 vs 0.025).  The GPU perf
  evidence at production LOCAL shapes is the unit gate's ~1.8×; this
  deck's τ tiles are far smaller.  Total job wall 6:37 of the 2 h window.

## Verdict

**The mirror works: LORRAX_FFT_FFI (+ _FUSED) is now platform-uniform.**
Same target names resolved per lowering platform; announce-or-refuse
doctrine preserved (probe on the right library, loud refusal otherwise);
norm scales still computed in ONE place (Python) and matched to jnp.fft;
value parity gated on GPU at the unit level (0…4e-16) and end-to-end
(eqp0 exact-to-text vs both the CPU baselines and OFF-leg twins).

## Honest risks / not-done

1. **Engine-swap parity is text-precision at this deck** — a GPU
   sigma_mnk.h5 tensor gate (the CPU round's job 7878845 analog) was not
   run; expect the same ~1e-14 eV ULP class if ever measured.
2. **Single-compute-stream assumption**: the shared cuFFT workspace + V_R
   arenas are mutex-serialized at ENQUEUE only; two concurrent XLA
   executables on different streams would race them on-device.  Matches
   XLA:GPU's one-compute-stream-per-module behavior today; stated in the
   TU header.
3. **Arenas are cudaMalloc'd outside XLA** — safe under the production
   env (cuda_async allocator, preallocate=false, both set by ffi_env.sh);
   a deployment with default BFC preallocation could starve them at large
   shapes.  Workspace was 0 B at every gated geometry, but cuFFT may
   request real workspace at other kgrids.
4. **No production-scale GPU A/B** — there is no AQ-class GPU deck; the
   1.8× at production local shapes is standalone-jit evidence only.  On
   GPU the XLA transposes ride HBM, so nothing like the CPU 3.78× should
   ever be claimed.
5. **Perlmutter container build unverified this round** — the CMake block
   auto-detects cufft/nvrtc under /usr/local/cuda and skips loudly if
   absent; an AE.4b-style build gate there is owed before GPU deployment
   on that site.
6. Multi-node GPU untested (single-node × 4 GPU legs only), consistent
   with the pre-existing scope of the rtx harness.
7. In-place strided cufftExecZ2Z (granted-alias path) is exercised and
   green on cuFFT 11.4.1 (toolkit 12.9); other cuFFT generations should
   re-run the unit gate's donation smoke.
8. Harness artifact, NOT an FFT finding: with the 2-node allocation the
   slab_io=auto router declined the CUDA FFI writer ("multi-node GPU run
   detected (2 nodes)") in every leg — it keys on the JOB's node count,
   not the step's -N1 — so these smokes wrote through the fallback tier
   (unlike the reference rtx_smoke's -N1 PHDF5_FFI banner).  Identical in
   OFF and ON legs; irrelevant to the FFT gates.
