# wk_REL — GEMM vendor portability + AUTO default + BSE adoption (2026-07-29)

Owner orders, three parts, tree /work2/08271/jackmc/frontera/lorrax @
4f77842 (was 5894dcd when the work started; the cuFFT mirror committed
underneath mid-workstream), WORKING TREE ONLY (not committed;
orchestrator merges).

⚠ TREE CONFOUND (recorded before any gate): a concurrent agent's cuFFT
flat-k mirror landed on disk mid-workstream and is now COMMITTED as
4f77842 (job lrxCUFFT/7879275): src/common/fft_helpers.py (platform
dispatch: MKL-DFTI on cpu / cuFFT strided on CUDA, same target names),
src/ffi/common/ffi_loader.py (CUDA table entries for lorrax_mklfft_*),
src/ffi/common/cpp/CMakeLists.txt (cufft block), config/frontera/
stage_ffi_deps.sh.  CPU-side gates here run through the same
fft_helpers entry points — behavior-identical on cpu meshes by that
change's own contract.  GATE 1 job 7879288 ran on the tree with that
change COMMITTED (clean base), so the confound is retired for the
unit gates.  (Also inherited: Laplace merge is DEFAULT since 7879005.)

⚠ ALSO IN THE WORKING TREE, NOT OURS: manual/05_isdf/
5.1_pair_density_factorization.md carries owner prose edits (mid-
sentence at the time of writing).  Untouched by this workstream; listed
here only so the orchestrator does not attribute it to the diff below.

## PART 1 — GEMM vendor portability (Cray OR Intel)

`cblas_?gemm_batch` is an MKL extension (OpenBLAS ships it; LibSci does
not).  Changes:

- src/ffi/mklblas/cpp/gemm_batch_ffi.cc — two build-time macros:
  * `LORRAX_MKLBLAS_MKL_HEADER` -> `<mkl_cblas.h>` + `MKL_INT`; else
    `<cblas.h>` + `int` (standard LP64 CBLAS: LibSci/OpenBLAS/BLIS).
  * `LORRAX_MKLBLAS_HAVE_BATCH` -> ONE `cblas_{d,z}gemm_batch` call
    (preferred when present); else a portable loop of plain
    `cblas_{d,z}gemm` (one internally-threaded GEMM per batch slot;
    loop sequential BY DESIGN — an outer OMP loop would fight the BLAS
    team).  Same B-cycling broadcast rule both paths.  MklLocalPin stays
    (dlsym'd — no-op on non-MKL).  LORRAX_MKLBLAS_LOG line now names the
    entry used.
- src/ffi/common/cpp/host/CMakeLists.txt — the mklblas block moved OUT of
  the MKL-probe-success branch to after the link-line resolution, so it
  also builds on the explicit `-DLORRAX_SCALAPACK_LIBRARIES` (LibSci)
  route.  Header resolution: mkl_cblas.h under the MKL prefix, else
  find_path(cblas.h) (hints: CRAY_LIBSCI_PREFIX_DIR, LORRAX_CBLAS_DIR).
  Feature check: `check_cxx_symbol_exists(cblas_zgemm_batch ...)` against
  the actual headers + link line (CheckCXXSymbolExists — the project is
  `project(lorrax_ffi_host LANGUAGES CXX)`, no C compiler enabled, so
  this IS check_symbol_exists for this project).  Configure log prints
  which path was compiled; the gate ASSERTS the Frontera/MKL build keeps
  the batched entry.

### PART 1 BUG FOUND AND FIXED AT THE GATE (probe false-negative)

The predecessor's two gate submits BOTH failed the build assertion with
`batched_msg=0` — i.e. the Frontera/MKL build silently compiled the SLOW
plain-GEMM loop against an MKL that has the batched entry.  Root cause
read off `build_host_GBP/CMakeFiles/CMakeConfigureLog.yaml`, not guessed:

`check_symbol_exists` links a try_compile EXECUTABLE, and ld defaults to
`--no-allow-shlib-undefined` there — it demands the WHOLE shared-library
closure resolve, which the real target (a .so) never has to.  Two
successive false negatives:

  (a) bare `_scalapack_link` ->
      `libmkl_blacs_intelmpi_lp64.so: undefined reference to MPI_Waitall`
      (BLACS is dragged in by `-Wl,--no-as-needed`; the process supplies
      MPI at load time, so the .so legally leaves those open)  [job
      7879278];
  (b) the predecessor's fix — appending `${LORRAX_MPI_LIBRARY}` to close
      (a) -> `libmpi.so: undefined reference to fi_getinfo@FABRIC_1.1`
      (libfabric is on LD_LIBRARY_PATH at run time, not on the link
      search path)  [job 7879281].

FIX (this session): drop the libmpi append; set
`CMAKE_REQUIRED_LINK_OPTIONS "-Wl,--allow-shlib-undefined"` around the
probe (and unset it after).  That tolerates undefined symbols *inside
the dependency .so's* while still hard-failing an undefined reference
from the probe's OWN object — exactly the cblas_zgemm_batch question.
Validated on the login node BEFORE resubmitting, with a negative
control: link A (bare) rc=2 MPI_Waitall; link B (+flag) rc=0; NEGATIVE
control (`cblas_zgemm_batch_NOSUCH_` + flag) rc=1 "undefined reference"
— so the flag does not turn the probe into a rubber stamp.  Confirmed
in-container at job 7879288: `-- Looking for cblas_zgemm_batch - found`
-> `mklblas: GEMM host handler ON — batched entry cblas_?gemm_batch`,
and at RUN time `[mklblas] gemm_batch first call: ... via
cblas_?gemm_batch (batched entry)`.
A linker that rejects the flag simply fails the probe and gets the
portable plain-GEMM loop: slower, never wrong (documented in §3.4).

Claim scope: the LibSci route is compile-logic only — no Cray machine in
this campaign; UNTESTED beyond the CMake/TU structure.  Note also that
on the explicit-LibSci route the mklfft (DFTI) block is still inside the
MKL branch, so a LibSci host build gets the GEMM handler but NOT the FFT
handlers — LORRAX_FFT_FFI refuses there, which linalg_ffi.md already
states ("Intel-specific by construction").

## PART 2 — LORRAX_BANDS_GEMM_FFI registry row + AUTO default

- docs/dev/env_vars.md §2b: LORRAX_BANDS_GEMM_FFI registered next to
  LORRAX_FFI_SO/LORRAX_FFI_HOST_SO with the performance comment (XLA:CPU
  Eigen GEMM 1.6–1.9× below vendor BLAS at full threads; jobs
  7879008/7879010; measured project_rs 29.4->19.6 s, sigma.exec
  58.3->49.2 at nb=128/P=64).  Also registered: LORRAX_MKLBLAS_THREADS
  (§2b), LORRAX_MKLBLAS_LOG (§3b), and the previously unregistered FFT
  siblings LORRAX_FFT_FFI / LORRAX_FFT_FFI_FUSED.
- src/common/contract_bands.py — grammar now on|off|auto (auto = unset
  default).  `bands_gemm_ffi_mode()` exposed; `bands_gemm_ffi_enabled()`
  resolves auto: **ON iff platform is CPU (JAX_PLATFORMS read via
  ffi_loader.platform_from_env — never initializes the backend) AND
  probe_target finds the handler in the host .so**; announced ONCE on
  rank 0 either way (doctrine #8: capability detection, announced).  On
  CUDA auto is OFF silently-by-design; explicit =1 on CUDA still REFUSES
  loudly (unchanged); explicit =0 disables.  Auto additionally
  quiet-falls-back per call site where the dial cannot apply:
  extra="minor" (not GEMM-reachable) and non-f64/c128 dtypes (BSE
  fp32-GMRES complex64) — under explicit =1 both REFUSE with the fix
  named (the fft_helpers TypeError pattern).  Kernel-cache-key contract
  unchanged (ppm_tau_kernel keys on bands_gemm_ffi_enabled()).
- tests/test_contract_bands.py — NEW test_auto_default (auto==probe
  verdict; =0 disables; auto minor + c64 fall back to the XLA plan
  without refusal, custom-call counts pinned).  tests/
  test_projection_lgemm.py now pins LORRAX_BANDS_GEMM_FFI=0 itself (it
  pins the XLA lowering; under the new default the FFI plan would
  replace its pinned dots).
- staged_reshard_primitive.md §1 refusal list + §3.4 mirror the wording.

## PART 3 — BSE adoption of the shared FFI machinery

(a) FFT call-site sweep (grep jnp.fft/np.fft over src/bse + src/
bandstructure): bandstructure was already clean (htransform uses
make_flat_k_ifftn).  BSE stragglers routed through the fft_helpers local
kernels local_ifftn3/local_fftn3:
  bse_feast.ensure_W_R (:84), bse_kpm (:158), bse_pseudopoles (:180,
  :251), bse_io coarse->fine W interp (:713/:715), bse_serial reference
  (:62/:63/:65), bse_ring_comm smoke/correctness drivers (:993, :1055,
  :1078-1081, :1089), vq_interp refit (:1489).

> ⚠ CLAIM-DECAY (corrected 2026-07-29 by the CPU/GPU sweep, §(c)):
> the predecessor's §(a) wrote that this makes BSE "inherit
> LORRAX_FFT_FFI / the cuFFT-mirror work automatically".  **That is
> false as implemented and has been corrected in the docs.**
> `local_ifftn3`/`local_fftn3` are literal one-line aliases of
> `jnp.fft.ifftn`/`fftn` (fft_helpers.py:306/318) — they carry NO gate.
> The LORRAX_FFT_FFI switch lives ONLY in the `make_flat_k_*` factories
> (`_make_flat_k_fft_ffi`, `make_flat_k_gw_conv`), and `grep -rn
> "make_flat_k_\|fft_ffi_enabled" src/bse/` returns **nothing**: BSE has
> zero flat-k call sites, so the flag (and the cuFFT mirror) does not
> reach BSE on either platform.  What the sweep actually bought is
> call-site centralization — one place to flip later — plus the honest
> property that the aliases are dtype-agnostic, which is what keeps
> BSE's complex64 fp32-GMRES FFTs working (the flat-k FFI handlers are
> c128-only on cpu AND CUDA and would have to refuse them).  The
> predecessor's own §(c) said "the FFT routings are alias-level and
> carry no gate at all", i.e. the two paragraphs contradicted each
> other; §(c) was the correct one.

All routings are alias-level (bit-identical values).  HONEST SCOPE:
vq_interp:240/:249 remain **host numpy np.fft** in gates-only
recon/to_sphere setup — not JAX call sites; converting them would change
the execution engine, not the routing.  Left as-is, recorded here.  NOT
done (named): upgrading the sharded W_q->W_R sites to
make_sharded_ifftn_3d factories (a gather-avoidance perf change needing
its own parity gate, not a routing fix), and any real BSE flat-k
adoption (the thing that would actually make LORRAX_FFT_FFI reach BSE).

(b) bse_ring_comm._apply_W_from_T (TDA builder AND non-TDA builder —
identical bodies, one replace) now decodes through
`contract_bands_block_reshard(mesh_xy, extra="leading")` per adoption
map §6.2 (the CLEAN drop-in): O = conv output U transposed rank-locally
(b,M,N,t,s,k)->(b,k,t,M,s,N) (sharded axes preserved: M/'x', N/'y'),
psi_right = psi_v_Y transposed (k,v,s,N)->(k,s,N,v), out (b,nk,c_X,v_Y)
-> transpose -> sh.X.  Converts the partitioner-chosen collectives of
the einsum pair (c-replicated intermediate, LARGE payload on strided
'x' — memo §6.1 finding-#1 inversion class) into the structural chain:
large partial over node-local 'y' groups, ONE stacked collective per
mesh axis for all b trials (AK.9), impl=mpi world-first warm-up
inherited from the factory.  Value-level identical (reassociation, 1e-12
class).  Under the AUTO default on CPU the decode right-GEMM rides the
FFI handler (c128 zgemm); the fp32-GMRES path (complex64) rides the
primitive's XLA lowering via the dtype boundary.  Per orders: two-phase
trial-stacking variant NOT implemented (open owner API decision);
bse_stack_matvec._w_stack NOT touched (layout inversion stays pending
that decision); bse_simple/bse_serial reference einsums left independent
(they are the things the gates compare AGAINST); vq_interp.make_eval_vq
untouched (map §6.3: honestly not a contract_bands instance).

(c) CPU/GPU ROBUSTNESS SWEEP (owner priority; code + platform-table
reading, 2026-07-29 — see the verdict table at the end of this file).

## Gates (order: unit gates first; A/B only on PASS)

| # | job | cell | verdict |
|---|-----|------|---------|
| — | 7879278 | GATE 1, predecessor submit #1 | **FAIL** — build assertion (probe false-negative (a), MPI_Waitall) + parity fixture crashed on BOTH trees (pre-existing V-ring vma) |
| — | 7879281 | GATE 1, predecessor submit #2 | **FAIL** — fixture fixed (monkeypatched V-ring, parity 0/0/0/0 PASS at ≤6.7e-16) but build assertion still failed (probe false-negative (b), fi_getinfo) |
| 1 | **7879288** | GATE 1, after the probe fix | **PASS** — every cell 0; see below |
| 2 | **7879296** | GATE 2 A/B (32N/P=64, AUTO) | **PASS** — .dat exact-0, h5 **bit-identical** to the GEMM-ON pass, AUTO-ON banner, 4+1 mklblas custom-calls |
| 3 | **7879363** | GBP_dlsym: probe removal (owner directive 1) + FFT-layout HLO probe round 1 | **PASS** — `build=0 dlsym=0 announce=0 cbands=0 lgemm=0 fftprobe=0` |
| — | 7879366 | FFT probe round 2, attempt 1 | **FAIL** (my bad `jax.shard_map` import; the harness also mis-reported it as rc=0 — grep's status masked srun's.  Both fixed) |
| 4 | **7879370** | FFT probe round 2 (HLO dumps, owner directive 2) | **PASS** — 8/8 modules dumped to `wk_REL/results/hlo/fftlayout_hlo/` |
| — | 7879427 | GBP_c64 attempt 1 (`small`) | **FAIL** `cbands=1` — my test bug (real-operand run vs complex-ψ reference), not the handler |
| 5 | **7879437** | GBP_c64: all four BLAS precisions + BSE fp32 end-to-end (`small`) | **PASS** — `build=0 cbands=0 lgemm=0 bse_fp32=0`; 9 passed 0 failed |

### GATE 1 — job 7879288 (development, c207-035, 01:11:10–01:16:33, 5:23)

sacct: COMPLETED.  Verdict line: `build=0 pycompile=0 cbands=0 lgemm=0
parity(old/xla/auto/cmp)=0/0/0/0` -> `GBP GATE 1: PASS`.

| cell | result |
|------|--------|
| build-host | **OK** — `-- Looking for cblas_zgemm_batch - found`; `mklblas: GEMM host handler ON — batched entry cblas_?gemm_batch (mkl_cblas.h at /opt/intel/…2020.1.217/linux/mkl/include)`; 0 compile errors; exports MklBlasGemmBatchHostFfi + MklFftFlatKHostFfi + ScalapackEighHostFfi present |
| pycompile | rc=0 |
| cbands (`tests/test_contract_bands.py`) | **8 passed, 0 failed** — incl. NEW `test_auto_default`; runtime banner `[bands_gemm] AUTO-ON: CPU platform and FFI target 'lorrax_mklblas_gemm_batch' resolves…`; `[mklblas] gemm_batch first call: dtype=c128 BA=12 BB=4 M=16 N=8 K=16 threads=28 **via cblas_?gemm_batch (batched entry)**`; `[auto] AUTO-ON engages; minor + c64 fall back to XLA` |
| lgemm (`tests/test_projection_lgemm.py`) | **2 passed, 0 failed** (its own `LORRAX_BANDS_GEMM_FFI=0` pin holds the XLA lowering it asserts on) |
| BSE pytest sweep, base(5894dcd) vs new(worktree) | **IDENTICAL, 8/8 files** — compared not just rc but the sorted per-test `ERROR …::testname` set + the count line: dense_reference `12 errors;1 deselected` (same 12 names), stack_matvec `3 errors` (same 3), w0_resolvent `1 passed;2 errors`, w_omega_chain `2 errors`, nontda `5 passed`, vq_interp/kgrid `3 skipped`, exciton_bands `1 skipped`.  Root cause of every rc=1 is PRE-EXISTING and environmental: `ValueError: mesh 2×2=4 != jax.process_count()=1` (those files need a 4-rank launch; the cell is `-n1`) — nothing to do with this branch |
| parity fixture (gbp_bse_parity.py) | old / new-xla / new-auto all rc=0.  Each tree's TDA matvec vs the single-device serial reference: 5.609e-16 / 6.694e-16 / 6.564e-16.  Cross-compare `HX_tda` new-xla-vs-old **5.292e-16**, new-auto-vs-old **4.403e-16**, auto-vs-xla **2.936e-16**; `HXY_full` (non-TDA) 4.676e-16 / 6.235e-16 / 2.707e-16; `HX_serialref` **0.000e+00** across all three (the untouched reference).  **PASS (tol 1e-12)** on all three tensors.  The par-auto cell shows both required banners (AUTO-ON + `[mklblas] … via cblas_?gemm_batch (batched entry)`, BA=8 BB=4 M=32 N=8 K=32) |

Fixture scope (honest, unchanged from the predecessor's design): the
fixture monkeypatches `apply_V_ring` to a vma-safe zero on BOTH trees
and zeroes V_q0 in the serial reference, because
`bse_ring_comm.apply_V_ring`'s `lax.fori_loop` carries an unvarying zero
initializer inside shard_map and fails current-jax vma typing at TRACE
time — identically at 5894dcd and on the adopted tree, untouched by this
branch (job 7879278 shows both tracebacks side by side).  So what GATE 1
gates is the D-term + encode + conv + **the decode**, i.e. exactly the
changed path; the V-term is out of scope and is a NAMED PRE-EXISTING
DEFECT, not a regression.

### GATE 2 — job 7879296 (32N/P=64, 01:44:25–01:48:45, ExitCode 0:0) — **PASS**

Queued 01:16:58 behind the concurrent workstream's 32-node jobs
(`QOSMaxNodePerUserLimit`); started when they cleared.  No foreign job
touched.  ONE restart-gated nb=128 production pass, `coll=mpi`,
cache-cold, `LORRAX_BANDS_GEMM_FFI` **unset** (verified in the launcher:
`inner.sh:8 unset LORRAX_BANDS_GEMM_FFI`), pass rc=0 wall 243 s.

| gate | result |
|------|--------|
| AUTO-ON announcement | **fires, exactly once, rank 0** — verified by grepping `run_GBP_a128auto/gw.log` directly, NOT from the harness summary (see the harness flaw below): `[bands_gemm] AUTO-ON: CPU platform and FFI target 'lorrax_mklblas_gemm_batch' resolves in the host .so …` |
| handler actually used | `[mklblas] gemm_batch first call: dtype=f64 BA=16 BB=16 M=1248 N=128 K=1248 threads=28 **via cblas_?gemm_batch (batched entry)**` — on **all 64 ranks** |
| refusals | **zero** (`grep -cE "refus\|LORRAX_BANDS_GEMM_FFI:" gw.log` = 0) |
| `.dat` parity vs `run_AQ_c4962_p64_mpi` | `sigma_diag.dat` n=16400 **max\|diff\|=0.000e+00**; `eqp0.dat` n=5160 **0.000e+00**; `eqp1.dat` **0.000e+00** → PASS |
| h5 vs B128 (truth baseline) | PASS (tol 1e-12), value-level: worst `sigma_c_kij_ev` max\|diff\|=**2.495e-14**, maxrel **3.380e-15**; `sigma_total_kij_ev` 1.138e-13 / 2.010e-16; hartree/omega/sigma_sx exactly 0 |
| h5 vs `run_CBANDS_g128f` (job 7879010's explicit GEMM-ON pass) | **BIT-IDENTICAL payload — 0.000e+00 on every dataset.**  This is the strongest form of the owner's gate: AUTO-resolved-ON produces byte-for-byte the same output as explicit `=1` |
| tau-kernel HLO (rank-0, cache-cold) | `module_0551.jit__tau_kernel`: **4** `lorrax_mklblas_gemm_batch` custom-calls; `module_0553`: **1**.  f64 dots at μ shapes: **0** (the de-promotion pin holds).  reduce-scatter 2 / all-gather 0 in both |
| timing | `sigma.exec` **50.188 s** vs g128f ref 49.224 (+2.0 %).  Single cache-cold pass, no repeat — treat as "no regression", NOT as a measured delta; the h5 bit-identity is the real evidence that the same plan ran |

Harness flaw found (recorded, not fixed — it affects reporting only):
`gbp_ab.sbatch:118` greps the banners with `… | head -14`, and the seven
ranks' CUDA-plugin `cuInit` tracebacks (benign, every CPU run emits them)
filled all 14 lines, so **the AUTO-ON banner did not appear in the job's
own summary**.  The banner is present in `gw.log`; the gate was confirmed
by going to the source. Anyone re-using this harness should filter the
CUDA-plugin noise before `head`.

## PART 4 — ENVIRONMENT ROBUSTNESS: the build-time probe is DELETED (owner directive, 2026-07-29)

Owner: *"i really don't want things to be set before and after specific
probes… i really want a not scary more robust way to set up the
environment."*  The false-negative found at GATE 1 is the argument: a
link-based `check_symbol_exists` answers "no" for **environmental**
reasons, and its "no" silently selects the ~1.6–1.9× slower loop.  It did
so twice, for two different link-closure reasons.  Patching it with
`--allow-shlib-undefined` fixed *that* manifestation; it did not remove
the class.

**What changed**

* `src/ffi/mklblas/cpp/gemm_batch_ffi.cc` — `cblas_{d,z}gemm_batch` are
  now resolved by `dlsym` at first use (`RTLD_DEFAULT`, then `RTLD_NEXT`
  as a second chance), exactly the house idiom already used for
  `MKL_Set_Num_Threads_Local` (`scalapack/cpp/blacs_grid.h`) and the
  cuFFT driver entries.  The prototypes are declared here as
  **function-pointer typedefs**, so no header need declare them; the
  CBLAS enum parameters are typed `int` because vendors spell the enum
  types differently (`CBLAS_LAYOUT` / `CBLAS_ORDER`) while the values are
  standard and int-passed.  `#if LORRAX_MKLBLAS_HAVE_BATCH` is gone; the
  batched and plain bodies are now two arms of a runtime `if`, numerically
  unchanged.
* `src/ffi/common/cpp/host/CMakeLists.txt` — the `check_cxx_symbol_exists`
  block and **every** `CMAKE_REQUIRED_*` set/unset are deleted.  What
  remains is a plain `EXISTS` test choosing which CBLAS header to include,
  whose failure mode is "handler not built at all" (announced, and refused
  loudly at runtime) — never "built, but silently slow".
* The announce is now **UNCONDITIONAL on first use** (rank-0 via
  `SLURM_PROCID`/`PMI_RANK`, always in single-process), not behind
  `LORRAX_MKLBLAS_LOG`: `[mklblas] GEMM entry: …`.  A silent downgrade is
  impossible by construction, which is the point.

**An unlooked-for structural win** — `nm -D --undefined-only`:

| .so | undefined `cblas_?gemm_batch` refs |
|---|---|
| `build_host_GBP` (macro build, 01:12) | **2** (`U cblas_dgemm_batch`, `U cblas_zgemm_batch`) |
| `build_host_GBPD` (dlsym build, 01:55) | **0** |

The old binary *link-referenced* the batched entries, so on a BLAS that
lacks them it would have failed to load or failed at first call — the
"vendor-portable" claim was only realised if you rebuilt with the right
macro.  The new binary references neither: **one binary genuinely runs on
either vendor**, which is a stronger property than the design it replaces.

**Compile-level portability proof (login node, no job):** the TU compiles
clean both against `mkl_cblas.h` **and** against a synthetic plain
`cblas.h` that deliberately omits the batched declarations (classic
`enum CBLAS_ORDER` spelling, plain `int` indices — the LibSci/netlib
shape).  The LibSci claim is now supported by compilation, not assertion.
Still no Cray machine, so still untested end-to-end there.

**GATE — job 7879363 (development, 01:54:08–01:56:34, 2:26): PASS**
`build=0 dlsym=0 announce=0 cbands=0 lgemm=0 fftprobe=0`

| cell | result |
|---|---|
| build | configure log shows `mklblas: GEMM host handler ON (mkl_cblas.h at …); batched-vs-plain entry is chosen at RUNTIME by dlsym…`; **zero** `Looking for cblas_zgemm_batch` lines (probe really gone); exports present |
| dlsym control (`dlsym_control.py`) | **PASS, both directions**, through the production loader call (`ctypes.CDLL(so, RTLD_GLOBAL)` = `ffi_loader.py:514`): bogus symbol not found before **or** after load (negative control); `cblas_{d,z}gemm_batch` **not** found before load, **found** after; `MKL_Set_Num_Threads_Local` found (the pre-existing pin idiom resolves by the same mechanism — previously never directly verified) |
| announce (with `LORRAX_MKLBLAS_LOG` **unset**) | `[mklblas] GEMM entry: cblas_?gemm_batch (batched) — resolved by dlsym at first use.` present (1 line); the LOG-gated detail line correctly absent (0 lines).  This is the "silent downgrade impossible" gate |
| cbands / lgemm | 8 passed 0 failed / 2 passed 0 failed against the rebuilt .so — numerical paths unchanged |

**Sweep for the same anti-pattern (owner asked):** `grep` for
`check_symbol_exists` / `check_cxx_symbol_exists` / `check_function_exists`
/ `check_library_exists` / `check_include*` / `try_compile` / `try_run` /
`check_cxx_source_compiles` / `CMAKE_REQUIRED_*` across **all** of `src/`
and `config/` returns **nothing** but my own comments — the deleted block
was the only instance.  Two weaker order-dependencies remain and are now
documented in-file rather than left implicit:

1. `_mkl_incdir` is set **only** by the MKL branch of the ScaLAPACK
   resolution, yet is read afterwards by both the mklfft block and the
   GEMM header choice.  On the explicit `-DLORRAX_SCALAPACK_LIBRARIES`
   (LibSci) route it is unset, so (a) the mklfft/DFTI handlers are not
   built at all and `LORRAX_FFT_FFI` refuses there, (b) the GEMM handler
   falls through to `find_path(cblas.h)`.  Both outcomes are correct and
   loud; neither can yield a silently-degraded binary.  Not the same class
   of footgun, but it does couple two unrelated decisions.
2. `find_path`/`find_library` results are CMake **cache** entries, so an
   incremental re-configure in an existing build dir reuses them.  The
   gates all use `build_ffi_host.sh --fresh` (which `rm -rf`s the build
   dir), so this never bit us; a non-fresh rebuild after an environment
   change could.  Named, not fixed.

## PART 5 — c64/f32 in the GEMM handler (owner decision, 2026-07-29)

Owner, on the fp32 risk flagged in the robustness sweep: *"can we just add
a c64 ffi function and call that?"* — yes, and that ruling is the
precedent: **extend the handler, do not add a fallback grammar.**

**What changed**

* `gemm_batch_ffi.cc` now serves **all four BLAS precisions** —
  f64/f32/c128/c64 onto `cblas_{d,s,z,c}gemm[_batch]` — through the same
  `dlsym(RTLD_DEFAULT → RTLD_NEXT)` path built in PART 4, with
  function-pointer typedefs so no vendor header need declare the batched
  entries.  Two design points:
  - **batched availability is tracked PER PRECISION**, not all-or-nothing:
    a BLAS shipping the double batched entries but not the single ones
    keeps the batched path where it has it instead of losing it globally;
  - the announce is therefore **per precision** (bounded at four lines per
    process) — a single global line could no longer tell the truth.
  The B-cycling broadcast rule and `MklLocalPin` are shared by all four
  through two lambdas (`run_real`/`run_cplx`), so they cannot drift apart.
* `contract_bands.py` — the non-f64/c128 exclusion is **gone**.
  `extra="minor"` stays excluded (structural: the contracted axis is not
  GEMM-reachable).  Explicit `=1` on a c64 site now SUCCEEDS.  The
  refusal message was rewritten to distinguish its two remaining causes: a
  genuinely unserved precision (f16/bf16/c256) vs a **mismatched pair**,
  which means the de-promotion policy failed to split upstream — a bug to
  report, not a dial to unset.
* Docs: `env_vars.md` + `staged_reshard_primitive.md` state all four
  precisions are served on CPU.

**Tolerance choice, and why not 1e-6.**  Unit roundoff for binary32 is
u = 2⁻²⁴ = 5.96e-8.  The contracted length in the unit fixture is
K = NS·MU = 32, so one evaluation carries up to K·u = 1.9e-6, and we
compare two different summation orders (vendor GEMM vs Eigen dot), giving
~2·K·u = 3.8e-6.  **TOL32 = 1e-5** is the smallest round number strictly
above that bound.  The owner's suggested ~1e-6 sits *below* the
theoretical worst case and would have been flaky by construction.  The
tests print the measured error so the real margin is visible:

| comparison | measured | tol |
|---|---|---|
| c64 handler-vs-XLA (both f32) | **1.51e-07** | 1e-5 |
| c64 handler-vs-exact (f64 ref) | **1.34e-07** | 1e-5 |
| f32 handler-vs-exact | **1.11e-07** | 1e-5 |
| BSE W-term c64 handler-vs-XLA | **1.736e-07** | 1e-5 |
| BSE W-term c64-vs-c128 (representation) | **2.427e-07** | 5e-5 |

Measured errors land ~1.5–2.5× u — i.e. **~25× inside** the bound, which
is the honest reason the tolerance is not tight: it is set by the proof,
not by the observation.

**GATE — job 7879437 (`small` queue, 02:38:40–02:41:10, 2:30): PASS**
`build=0 cbands=0 lgemm=0 bse_fp32=0`; unit suite **9 passed, 0 failed**
(was 8 — `test_single_precision_ffi` is new), lgemm 2/2.

* All four entries resolve and announce independently:
  `GEMM entry (c128): cblas_zgemm_batch`, `(f64): cblas_dgemm_batch`,
  `(c64): cblas_cgemm_batch`, `(f32): cblas_sgemm_batch`.
* `nm -D --undefined-only | grep -c gemm_batch` = **0** — all four batched
  entries are dlsym'd, none link-referenced, so the one binary still runs
  on a BLAS that has none of them.
* `test_auto_default`'s c64 assertion **flipped** (custom-call count
  0 → 1) and passes: c64 rides the handler instead of falling back.
* **`gbp_bse_fp32.py`: the previously-CRASHING command line PASSES.**
  `LORRAX_BANDS_GEMM_FFI=1` with a complex64 BSE W-term now runs
  (`dtype=complex64`, **1** mklblas custom-call in the compiled matvec —
  proving c64 really rode the handler, not that the error merely went
  away), dial=0 emits 0, and the two agree at 1.7e-07.

**FIRST ATTEMPT FAILED — job 7879427, `cbands=1`, recorded honestly.**
The failure was **mine, in the test, not in the handler**: the f32
sub-case compared a real-operand run against a complex-ψ reference
(`_operands(real_o=True)` realifies only `O`, but I projected all three
inputs to real), giving a structural 6.778e-01 discrepancy that looked
like a numerical catastrophe.  Fixed by taking the reference from the
same real operands; the trap is recorded in a comment so it is not
reintroduced.  Everything else in that run was already green, including
all four announces and the whole c64 half — so the deliverable was never
in doubt, only my comparison.

## CPU/GPU robustness sweep — verdict per newly-gated path

Method: read the code + the `ffi_loader` platform tables + the CMake
resolution, on disk, at 4f77842+worktree.  **No GPU execution was run
for any row** (this campaign's meshes are all CPU; the rtx-dev cells
that exist belong to the concurrent cuFFT workstream, job 7879275).
Every row is therefore a *resolves-sanely* verdict, never a perf claim.

| path | CPU (Frontera/MKL) | CUDA | verdict |
|------|--------------------|------|---------|
| `LORRAX_BANDS_GEMM_FFI` **auto** (default) | ON: `platform_from_env`=cpu **and** `probe_target` finds `lorrax_mklblas_gemm_batch` in the host .so -> announced once on rank 0.  Gate-verified live (7879288) | OFF, **silently by design** — `platform_from_env` returns `CUDA` and short-circuits *before* any `dlopen`, so a GPU node never tries to load the host .so; the factory then re-checks `mesh.devices.flat[0].platform` as belt-and-braces | **SANE both.**  Non-obvious bit now documented: the platform read is LEXICAL (`JAX_PLATFORMS`).  Production CPU runs are covered (harness exports `cpu`; `runtime.bootstrap()`'s GPU-less downgrade *forces* `cpu`), but a bare driver with `JAX_PLATFORMS` unset, or one that reaches a CPU mesh while it still reads `cuda,cpu` (`set_default_env`'s gpu default), gets auto-OFF — safe direction, no error, just no speedup.  Sentence added to env_vars.md + the `_bands_gemm_auto_enabled` docstring (which previously over-claimed "LORRAX's CPU entry points always export JAX_PLATFORMS=cpu") |
| `LORRAX_BANDS_GEMM_FFI=1` **explicit** | announce-or-refuse: refuses on missing/unloadable handler (quotes `probe_target` reason), refuses `extra="minor"`, refuses non-f64/c128 | **REFUSES LOUDLY** on the mesh-platform check in `_require_bands_gemm_ffi`, *before* touching `ffi_loader` — correct, and the target is not in the CUDA table anyway | **SANE both.**  A GPU user never has to unset this dial for the auto default; they only hit an error if they explicitly set `=1`, which is the intended doctrine |
| `LORRAX_BANDS_GEMM_FFI` **dtype boundary** (new) | c128/f64 -> handler; c64/f32 -> auto quietly keeps XLA, explicit `=1` raises `TypeError` naming the fix | same logic, but unreachable (dial is off on CUDA) | **SANE.**  ⚠ RISK worth the owner's eye: a BSE **fp32-GMRES** production run (`bse_pseudopoles._feast_filter`, `bse_kpm`) with an explicit `LORRAX_BANDS_GEMM_FFI=1` will now **crash** at trace time on complex64 operands.  That is the never-silently-downgrade doctrine working as designed, and the message names the fix — but it is a new way to break a previously-working command line.  Documented in §3.4 + the env_vars row |
| the primitive's `shard_map` / `psum_scatter` body | runs (all gates) | **lowers unchanged** — the bodies are only `jnp.einsum`, `jax.lax.psum_scatter(tiled=True)`, `jax.lax.complex`; `psum_scatter` lowers to NCCL reduce-scatter on GPU | **SANE, UNEXERCISED.**  Stated as such in §3.4 (new "Platform reach of this module" block) |
| `ensure_grouped_collectives_ready()` (called by every factory) | issues the world-barrier under `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`, P>1 | **no-op** — returns early because that variable is CPU-backend-only | **SANE both** |
| the mesh-minor-axis refusal (§3.2) | enforced | enforced, and the *rationale* survives: on a row-major `jax.devices()` GPU mesh the last axis is likewise the intra-node one.  A hand-permuted GPU device mesh will hit the refusal and must pass `axes=` matching its layout | **SANE**, now said explicitly in §3.4 rather than left as a CPU-flavoured argument |
| BSE `_apply_W_from_T` adoption (TDA + non-TDA) | gate-verified at 1e-16 (7879288) | lowers unchanged (it is the primitive + rank-local transposes); the GEMM dial just stays off | **SANE, UNEXERCISED on GPU** |
| BSE FFT routings under `LORRAX_FFT_FFI` | **NOT AFFECTED** — `local_ifftn3`/`local_fftn3` are aliases of `jnp.fft`; BSE has **zero** `make_flat_k_*` call sites (`grep` over `src/bse/` is empty), so the flag never reaches BSE | **NOT AFFECTED**, same reason — the cuFFT mirror does not reach BSE either | **CORRECTED CLAIM** (see the CLAIM-DECAY block in §3(a)).  The docs said BSE "inherits the backend switch automatically"; it does not.  env_vars.md's `LORRAX_FFT_FFI` row now carries an explicit SCOPE paragraph and staged_reshard_primitive.md §5 a matching bullet |
| BSE fp32/complex64 FFT paths | keep working *because* the routing is alias-level: `jnp.fft` is dtype-agnostic | same | **SANE.**  Load-bearing detail: the flat-k FFI handlers are **complex128-only on BOTH platforms** (`_make_flat_k_fft_ffi` raises `TypeError` for any other dtype; `make_flat_k_gw_conv` likewise) — so a future real BSE flat-k adoption would have to refuse or fall back for the c64 GMRES path.  Recorded in the env_vars row |
| `mklblas` TU on a non-MKL BLAS | `MklLocalPin` is `dlsym`'d -> no-op; `OMP_NUM_THREADS` governs | n/a (host-only TU) | **SANE** |
| LibSci host build (no Cray machine here) | GEMM handler builds via `cblas.h` + plain-GEMM loop; **mklfft/DFTI handlers are still skipped** (that block stayed inside the MKL branch), so `LORRAX_FFT_FFI` refuses on a LibSci host build | n/a | **SANE but UNTESTED** — compile-logic only, no Cray machine in this campaign |

## SIZING (analysis only, no jobs run): can the `local_*` FFT family get the FFI backend?

Coordinator question, answered from code + the on-disk measured record.
**Nothing was implemented and no job was submitted for this section.**
Headline: *the literal change asked about is feasible but is the wrong
lever; the measured record says a plain FFI transform is not a win at a
layout XLA already likes, and the coherent route for BSE is the one
already named in the adoption map (§6.1 route (a)) — move BSE's conv to
k-leading and call the EXISTING fused conv handler.*

### (1) How the two families differ, and can the existing handlers serve?

| | `make_flat_k_*` family | `local_*` / `make_sharded_*` family |
|---|---|---|
| operand form | `(nk, *trail)`, k **flattened and LEADING** | arbitrary rank, FFT over 3 axes given by `axes=` |
| FFT axes in memory | k-MAJOR: element strides `{nky·nkz·T, nkz·T, T}`, transforms batched over the trail at **distance 1** | k-MINOR-MOST at **every** BSE/vq call site on disk — `axes=(2,3,4)` of a rank-5 `W_q`, `(5,6,7)` of the rank-8 `T_k`, `(4,5,6)` of the rank-7 stack `T_k`, `(1,2,3)` of `zeta_box`: i.e. contiguous k-block, batch = product of the LEADING dims, distance `nk` |
| why the FFI exists | XLA:CPU's `fft` (DUCC) demands the transformed axes minor-most, so a k-major tile pays a full-tile transpose in and out — *measured 60–65 % of the Σ τ kernel* | **the pathology is absent by construction**: the axes are already minor-most, so XLA's own path pays no transpose |

### (1a) MEASURED: HLO at the actual call sites (jobs 7879363 / 7879370)

Owner directive: *"the local_* rejection must rest on HLO at the ACTUAL
call sites, not on axis-index reasoning… expects BATCHED ffts not to need
[minor-most]."*  Done — `wk_REL/fftlayout_probe{,2}.py` rebuild each
distinct call site's real shape+sharding, compile, and count
transpose/copy instructions with bytes.  All eight
`after_optimizations` modules are on disk in `wk_REL/results/hlo/fftlayout_hlo/`.
(Counts below are the ROOT-aware re-analysis on the login node; the
in-job parser under-counted because it skipped `ROOT %x = …` lines —
which is also why round 1 reported "0 fft ops" for case D and only
74.76 MB for the controls.  The corrected numbers are larger, not
smaller.)

| case | site | layout | fft | transp | copy | **bytes moved** | input |
|---|---|---|---|---|---|---|---|
| A | `W_q` ifft (feast/kpm/pseudopoles/io/lanczos) | minor-most | 1 | 0 | 0 | **0.00 MB** | 24.92 MB |
| B | ring conv `_apply_W_from_T` — THE hot site | minor-most | 2 | 0 | 0 | **0.00 MB** | 49.84 MB |
| C | stack conv `_w_stack` | minor-most | 2 | 0 | 0 | **0.00 MB** | 24.92 MB |
| D | `vq_interp` ζ fft | minor-most | 1 | 0 | 0 | **0.00 MB** | 35.39 MB |
| D2 | ζ fft, k-grid-sized extents | minor-most | 1 | 0 | 0 | **0.00 MB** | 0.02 MB |
| E | ring conv **inside shard_map** (production form) | minor-most | 2 | 0 | 0 | **0.00 MB** | 49.84 MB |
| **F** | **flat-k control (GW dot layout)** | **k-major** | 1 | **2** | **3** | **124.60 MB** | 24.92 MB |
| **G** | **flat-k control inside shard_map** | **k-major** | 1 | **2** | **3** | **124.60 MB** | 24.92 MB |

Case D is the cleanest single data point: its entire optimized module is
`parameter → fft → ROOT`, nothing else.

**(a) Does XLA:CPU insert layout transposes at the `local_*` sites?
NO — zero, at every one of the six reconstructions**, single-device and
sharded, plain transform and fused ifft·W·fft.

**(b) Does the compiler behave differently for batched-minor-most vs
flat-k? YES, decisively — the owner's intuition is CONFIRMED.**  Same
compiler, same 24.92 MB tile, same `fft_length={4,4,1}`: minor-most moves
**0 bytes**; k-major moves **124.60 MB = 5.0× the tile**.  The smoking
gun is in the HLO text — the fft's operand in the flat-k module is a
fusion literally named `%transpose_copy_fusion`:

```
%transpose.0 = c128[4,4,1,1,312,1,312]{2,1,0,6,5,4,3} transpose(%mul.0), dimensions={4,5,6,0,1,2,3}
%copy.4      = c128[4,4,1,1,312,1,312]{6,5,4,3,2,1,0} copy(%transpose.0)
%fft.0       = c128[1,312,1,312,4,4,1]{6,5,4,3,2,1,0} fft(%transpose_copy_fusion), fft_type=IFFT, fft_length={4,4,1}
%transpose.1 = c128[1,312,1,312,4,4,1]{3,2,1,0,6,5,4} transpose(%bitcast.3), dimensions={3,4,5,6,0,1,2}
ROOT %copy.5 = c128[1,312,1,312,4,4,1]{6,5,4,3,2,1,0} copy(%transpose.1)
```

So the minor-most requirement is real, but a **batched** fft whose
transform axes are already minor-most satisfies it for free — exactly as
the owner expected.  The transposes are not a property of `fft`; they are
a property of *k-major operands*.

**(c) Since no transposes are present at the BSE sites, Option A is NOT
re-opened.**  The previous verdict stands — but note the status change:
it was an INFERENCE from axis indices plus a borrowed 151-vs-128 ms
number; it is now a MEASUREMENT at the real shapes and shardings.  No
claim-decay banner is warranted because the conclusion did not move; the
*evidence class* was upgraded from "reasoning" to "HLO", which is what
the directive asked for.

**NEW finding that DOES change a recommendation** — this cuts against
§6.1 **route (b)** and tightens route (a).  The 124.60 MB is what a
k-major operand costs *on the XLA path*.  BSE's conv is currently
minor-most and pays 0.  Therefore:

* **route (b) ("same relabel + move the FFT axes handling to leading-k
  3-D form", XLA path) is REFUTED by this trace**: it would relabel BSE
  into precisely the layout that costs 5.0× the tile per transform, with
  no FFI backend to redeem it.  `contract_bands_notes.md` §6.1 flagged it
  "transpose cost priced by HLO before claiming — do not wire without the
  trace".  This is that trace, and the price is 5.0× the tile.
* **route (a) must land TOGETHER with `LORRAX_FFT_FFI`, never before
  it.**  The k-leading relabel and the FFI conv backend are not two
  independent improvements that can be sequenced — the relabel alone is a
  large regression, and only the FFI handler (which reads k-major via
  stride descriptors) makes it free.  Any staged rollout that lands the
  layout change first would look like a serious performance bug.

Verdict on reuse: **`lorrax_mklfft_flat_k` (and its CUDA twin) cannot
serve the local case as-is** — its DFTI descriptor derives strides from
the k-major arrangement (`{nky·nkz·T, nkz·T, T}`, distance 1); the local
case needs `{nky·nkz, nkz, 1}`, distance `nk`, `NUMBER_OF_TRANSFORMS =
∏(leading dims)`.  It is **not structurally unsuitable** — that stride
set is the *simpler*, default batched-contiguous DFTI/cuFFT
configuration — so it is a new entry point (`lorrax_mklfft_local_3d`,
~50–80 lines each side) or a generalization of the existing one to take
strides/distance as attributes.  Cheap to build.

**And the payoff argument fails on two independent bodies of evidence —
the HLO measured above (0 bytes moved at the BSE sites: there is no
layout churn for an FFI backend to remove) and the timing already on
disk:**
`wk_REL/docs/ffi_fft_proto_notes.md` (re-gate job 7878719, compact-chunk
engine, 28 threads, the 400 MB production G tile):

* PLAIN transform: **FFI-DFTI 151 ms vs XLA 128 ms** — the FFI plain
  transform is ~18 % *slower*, and that is the case where the FFI at
  least *avoids* a transpose.
* FUSED conv entry: **163 ms vs 1141 ms** for the XLA-equivalent staged
  composition (**7.0×**) — "the fused entry point is the decisive lever"
  (their words).

So the whole flat-k win is (a) the fused entry and (b) deleting layout
churn.  A plain `local_*` FFI backend gets neither: it would hand
MKL/cuFFT a transform XLA is already doing in the good layout, and the
one measured plain-vs-plain comparison has the FFI losing.  **Expected
outcome: neutral-to-negative on CPU, ≈0 on GPU (XLA:GPU already
dispatches cuFFT for `jnp.fft`).**

**The lever that does exist**: BSE's hot FFTs are not three independent
transforms — they are exactly the fused conv the existing handler
implements.  `bse_ring_comm._apply_W_from_T` (both builders) and
`bse_stack_matvec._w_stack` compute
`U = fftn( ifftn(T) · ifftn(W_q) )`, which is `make_flat_k_gw_conv`'s
`sigma = fftn( ifftn(G) · ifftn(W)[:, None, :, None, :] · mult )` with
`mult = 1/√N_k` (already folded by the wrapper's `mult` argument).
The index mapping works out with **no new entry point**:

* handler wants `G (nk, a, mx, b, my)`, `W (nk, mx, my)`, W broadcast
  over axes `a` and `b`;
* BSE has `T (b_trial, M, N, t, s, k…)` with `W_R (M, N, k)` broadcast
  over `b_trial`, `t`, `s` — so set `a = b_trial·t` (fold; W is
  broadcast over both anyway) and `b = s`, giving trail
  `(b_trial·t, M, s, N)`.
* That is a **relabel of the encode einsum**: `"kctM,cksN->MNtsk"` →
  `"kctM,cksN->ktMsN"` — verbatim §6.1 **route (a)** of
  `wk_REL/docs/contract_bands_notes.md`, which already says "the layouts the
  primitive wants and the FFT-FFI wants are the SAME layout, so the two
  adoptions compose".
* Bonus, stated precisely (not "the transposes disappear"): the
  adoption in PART 3(b) currently pays **one LARGE rank-local transpose**
  of the whole conv output, `U (b,M,N,t,s,k) → O (b,k,t,M,s,N)`, plus a
  ψ_v transpose and a small output transpose.  Route (a) makes the conv
  emit `(nk, b·t, M, s, N)` — the primitive's `extra="leading"` O order
  *up to the leading two axes* — so that large transpose collapses to a
  reshape + a **leading-axis** permute `(k, b, …) → (b, k, …)` (cheap:
  it does not touch the μ/ν/spinor trail).  The ψ_v transpose is
  per-solve hoistable already; the small `(b,nk,c,v) → (b,c,v,nk)`
  output transpose stays.  Net: the expensive one shrinks, the other two
  are unchanged — worth pricing in HLO before claiming, per QP#4.

One real contract mismatch to price: `make_flat_k_gw_conv` takes **W in
q-space** and does `ifftn(W)` *inside every call*, whereas BSE hoists
`W_R = ifftn(W_q)` once per solve and passes it into the matvec.  W is
`(μ_x_loc, μ_y_loc, nk)` — a factor `b_trial·t·s` smaller than T, so the
per-matvec re-transform is small but not free.  Either accept it, or add
a `w_in_r_space` attribute that skips `scale_i` on W (a few lines in
both handlers).

### (2) The dtype question (c64) — and the recommendation

Both handlers are **complex128-only on BOTH platforms**, enforced in the
Python wrappers: `_make_flat_k_fft_ffi` raises
`TypeError("FFI flat-k backend supports complex128 only, got … — unset
LORRAX_FFT_FFI for this call path")`, and `make_flat_k_gw_conv` raises
`TypeError("gw_conv supports complex128 only.")`.

What a c64 consumer hits today: **nothing** — BSE has no flat-k call
site, so the refusal is unreachable from BSE.  What it *would* hit after
any adoption: a hard `TypeError` at trace time.  The BSE c64 population
is real and named: `bse_pseudopoles._feast_filter` (`gmres_fp32` →
`_build_gmres_data_fp32`, `runner_dtype = jnp.complex64`, and its
`data_gmres["W_R"]`), and `bse_kpm`'s `data_fp32`.

Why this is sharper than the GEMM case: **`LORRAX_FFT_FFI=1` is a
production-global setting**, not a niche experiment — this very
workstream's GATE 2 harness exports it, as did every Σ A/B. A bare
refusal would make "the flag every GW production run sets" mutually
exclusive with "BSE's fp32 filter", decided per-process by an env var
that the BSE code cannot see coming.  That is exactly the broken-promise
shape QUALITY_PATTERNS warns about, arriving from the *other* direction.

**Recommendation (two parts, in order):**

1. **Fix the boundary, don't paper it.**  c128-only is a *choice*, not a
   necessity: DFTI has `DFTI_SINGLE` and cuFFT has `CUFFT_C2C` — both
   handlers already carry the descriptor/plan-cache machinery, so adding
   single precision is a precision-templated dispatch on the buffer
   dtype (the same shape the GEMM handler already uses for d/z), not new
   architecture.  Estimate: ~1 day per handler including the unit gate.
   This makes the question moot and is the honest fix.
2. **Until then, adopt the GEMM dial's three-state grammar** for
   `LORRAX_FFT_FFI` (`on|off|auto`, auto = default), with the dtype
   boundary as a **quiet per-call fallback under auto** and a **refusal
   under explicit `=1`** — i.e. copy `contract_bands._ffi_dtypes_ok`
   verbatim in spirit.  Rationale: dtype is a property of the CALL SITE,
   not of the user's request, so falling back is capability detection
   (doctrine #8), while a user who typed `=1` still gets told.  Do
   **not** keep the bare refusal if `local_*`/BSE ever comes under the
   flag.

### (3) Which BSE sites would actually benefit, and the runtime share

| site | frequency | benefits? |
|---|---|---|
| `bse_ring_comm._apply_W_from_T` (TDA `:343/:345` + non-TDA `:605/:607`, via `make_sharded_i/fftn_3d`) | **per matvec, every Krylov/Chebyshev/GMRES iteration** | **YES — the target.** This is the fused-conv shape |
| `bse_stack_matvec._w_stack` (`:118/:120`) | **per matvec per trial** (inside `lax.scan`) | **YES — same shape**; also the §6.1 finding-#1 decode target |
| `bse_lanczos:186`, `absorption_haydock:212`, `davidson_absorption:118`, `exciton_bands:702`, `bse_nontda:79`, `test_davidson_bse:107` | **once per solve** (W_q → W_R hoist) | marginal — a single small transform |
| `bse_feast.ensure_W_R:87`, `bse_kpm:159`, `bse_pseudopoles:181/:252`, `bse_io:714/716` | once per solve / per restart load | no |
| `vq_interp:1490`, `bse_serial:62-65`, `bse_ring_comm:1052-1148` | setup / reference implementations | no — and the references must stay independent, they are what the gates compare against |

**Share of BSE runtime: UNMEASURED, and not obtainable without new
jobs.**  BSE's instrumentation stops at term granularity
(`timing.section("bse_jax.W_term")` at `bse_ring_comm:436/:755`) — the
ifft / multiply / fft inside `_apply_W_from_T` are not separately timed,
and a grep of `lorrax_setup/{SPEEDUP_SCORECARD.md, SESSION_REPORT*,
wk_*/}` and the run dirs found no BSE log carrying those rows.  Stating
it plainly rather than borrowing the Σ number: the Σ τ kernel's 60–65 %
FFT share is **not** transferable, because that share is dominated by
layout churn BSE does not currently pay.

### (4) Effort / payoff, and the gates such a change would need

| option | effort | payoff | verdict |
|---|---|---|---|
| **A. Blanket FFI backend for `local_*`** (new `lorrax_mklfft_local_3d` + CUDA twin, gate in `local_ifftn3`/`local_fftn3`) | MED (2 handlers + wrapper + gates ≈ 2–3 days) | **Neutral to NEGATIVE.**  Now measured, not inferred: the BSE sites move **0 bytes** in transposes/copies (jobs 7879363/7879370), so there is no layout churn for an FFI backend to remove — the only lever left is engine-vs-engine, where the one plain-vs-plain measurement on disk has FFI *losing* (151 vs 128 ms).  ≈0 on GPU.  Plus it drops a c128-only refusal in front of fp32 BSE under a production-global flag | **DO NOT** |
| **B. §6.1 route (a): k-leading BSE conv → existing `make_flat_k_gw_conv`, WITH `LORRAX_FFT_FFI` in the same change** | MED-HIGH (encode relabel in 2 files + 3 builders, W-hoist contract decision, c64 story, gates ≈ 3–5 days) | **The only measurement-supported lever** (fused entry 7.0× standalone on the Σ shape); *additionally* shrinks the PART 3(b) decode's large O transpose and unlocks the §6.1 finding-#1 movement fix.  **MUST ship atomically with the FFI backend** — see B′ | **THE ONE TO COST** (owner call) |
| **B′. §6.1 route (b): the same k-leading relabel on the XLA path** (no FFI backend) | LOW-MED | **REFUTED by the trace.**  k-major operands cost **124.60 MB of transpose+copy on a 24.92 MB tile (5.0×)** per transform on the XLA path (cases F/G), which is exactly what BSE would be relabelled into.  It is a large regression, and it is also why B cannot be staged as "layout first, backend later" | **DO NOT** |
| **C. Add c64 to both handlers** | LOW-MED (~1 day/handler + unit gate) | removes the whole dtype cliff; prerequisite for B reaching the fp32 filter | **DO, independently of B** |

Gates any of these would need (nothing may be claimed without them):

1. **Unit**, both platforms: fused-conv and/or local-3D value parity vs
   the `jnp.fft` chain at BSE shapes — including odd/ragged k-grids,
   all four norms, both `input_output_aliases` branches (granted and
   denied), and c64 if C lands.  CPU in `development`, CUDA in
   `rtx-dev` (the cuFFT twin has *no* production-scale GPU evidence yet
   — `wk_REL/docs/cufft_mirror_notes.md` says so itself).
2. **HLO pins**: zero full-tile transposes around the conv; conv
   custom-call count ≥1; and — for B — that the decode's LARGE O
   transpose is *gone*, not merely moved (count transposes touching the
   μ/ν trail, not transposes in general).
3. **BSE value parity**: extend `wk_REL/probes/gbp_bse_parity.py` (already
   covers TDA + non-TDA + serial reference on a 2×2 emulated mesh) to
   an ON/OFF × old/new cross-compare at 1e-12, and add the stack matvec.
4. **Refusal/fallback gates**: c64 under `auto` (quiet XLA fallback,
   custom-call count 0) and under explicit `=1` (raises, message names
   the fix) — the `test_auto_default` pattern.
5. **Production A/B**: a restart-gated BSE solve vs a pinned baseline
   (eigenvalues / ε₂ peak), **with the fp32-GMRES path explicitly
   exercised under the flag** — that is the path that would break first.
6. Route (a) only: re-run the PART 3(b) decode parity, since the
   operand layout feeding `contract_bands_block_reshard` changes.

## Docs changed by this sweep (beyond the predecessor's edits)

- `docs/dev/env_vars.md` — `LORRAX_FFT_FFI` row: replaced the false "BSE
  … inherit the backend switch automatically" sentence with an explicit
  SCOPE paragraph (which subsystems the flag reaches; BSE not among
  them; why the alias-level routing is what keeps c64 working) and the
  c128-only refusal on both platforms.  `LORRAX_BANDS_GEMM_FFI` row: the
  lexical-`JAX_PLATFORMS` caveat + "nothing to unset on a GPU run".
- `docs/dev/staged_reshard_primitive.md` §3.4 — new **configure-time
  probe caveat** (the false-negative modes and the
  `--allow-shlib-undefined` fix, with the "read the configure log line"
  instruction) and a new **"Platform reach of this module"** block
  enumerating the three platform-conditional pieces and what each does
  on GPU, closing with the honest "no GPU execution of this primitive
  has been run".  §5 — the adoption bullet now says both builders lower
  unchanged on CUDA, plus a new bullet separating the FFT *hygiene*
  routing from a *backend switch*.
- `src/common/contract_bands.py` — `_bands_gemm_auto_enabled` docstring
  corrected (it claimed CPU entry points "always export
  JAX_PLATFORMS=cpu"; `set_default_env`'s default is `cuda,cpu`).

## Files touched (worktree only, NOT committed)

- src/ffi/mklblas/cpp/gemm_batch_ffi.cc (portability)
- src/ffi/common/cpp/host/CMakeLists.txt (mklblas block: vendor-neutral
  + check_cxx_symbol_exists + the --allow-shlib-undefined probe fix)
- src/common/contract_bands.py (auto default, mode fn, dtype boundary,
  corrected JAX_PLATFORMS docstring)
- src/bse/bse_ring_comm.py (adoption + FFT routing + import)
- src/bse/{bse_feast,bse_kpm,bse_pseudopoles,bse_serial,bse_io,
  vq_interp}.py (FFT routing)
- tests/test_contract_bands.py (test_auto_default), tests/
  test_projection_lgemm.py (pin =0)
- docs/dev/{env_vars,linalg_ffi,staged_reshard_primitive}.md
- wk_REL: gbp_gate.sbatch, gbp_ab.sbatch, gbp_bse_parity.py,
  gbp_par_compare.py, gbp_base_5894dcd/ (snapshot),
  gbp_build.7879281_falseneg.log (probe-failure evidence),
  gbp_dlsym.sbatch + dlsym_control.py (probe-removal gate),
  gbp_c64.sbatch + gbp_bse_fp32.py (four-precision + BSE fp32 gate),
  gbp_fft2.sbatch + fftlayout_probe.py + fftlayout_probe2.py +
  fftlayout_hlo/ (8 after_optimizations dumps — the FFT-layout
  evidence), this file.
- Run dir: mos2_4x4_test/run_GBP_a128auto (one run per dir).
- NOT ours, in the same worktree: manual/05_isdf/
  5.1_pair_density_factorization.md (owner prose).

## Named, not done

1. **Two-phase trial-stacking primitive variant** — per owner order, NOT
   implemented (open API decision).  `bse_stack_matvec._w_stack`'s
   layout inversion stays pending it; `bse_stack_matvec` untouched.
2. **FFI backend for the `local_*` FFT family** (option A of the sizing
   section above) — **analysed and RECOMMENDED AGAINST on MEASURED
   evidence**, not merely skipped.  HLO at all six real call sites
   (jobs 7879363/7879370, dumps in `wk_REL/results/hlo/fftlayout_hlo/`) shows **zero
   transposes and zero copies** — there is no layout churn for an FFI
   backend to remove — and the existing handler cannot serve the
   minor-most case without a new entry point anyway.  Owner decision
   requested only if they disagree with that reading.
3. **§6.1 route (a): k-leading BSE conv on the existing fused
   `make_flat_k_gw_conv`** (option B) — the lever that *is* supported by
   measurement (fused entry 7.0× standalone on the Σ shape) and the one
   that composes with the PART 3(b) decode adoption (it shrinks that
   adoption's LARGE O transpose to a leading-axis permute — priced in
   HLO, not assumed) and with the §6.1 finding-#1 movement fix.  Sized
   at 3–5 days + the 6 gate classes above.  NOT
   started; needs an owner go/no-go, and a `w_in_r_space` attribute
   decision (BSE hoists `W_R` per solve; the handler re-transforms W per
   call).  **NEW CONSTRAINT from the trace: it must ship atomically with
   `LORRAX_FFT_FFI`** — the layout relabel alone costs 5.0× the tile in
   transpose+copy (item 3b), so a staged "layout first" rollout would be
   a serious regression.
3b. **§6.1 route (b) (k-leading relabel on the XLA path, no FFI
   backend) — REFUTED, do not attempt.**  Measured: a k-major operand
   costs 124.60 MB of transpose+copy on a 24.92 MB tile per transform
   (cases F/G), versus 0.00 MB for BSE's current minor-most layout.
   `contract_bands_notes.md` §6.1 asked for exactly this trace before
   anyone wired it; this is the trace and the answer is no.
4. **complex64 support in the two FFT handlers** (option C, `DFTI_SINGLE`
   / `CUFFT_C2C`) — ~1 day per handler; removes the c128-only cliff that
   would otherwise make `LORRAX_FFT_FFI=1` (a production-global setting)
   incompatible with BSE's fp32-GMRES filter.  Still OPEN — note this is
   the *FFT* handlers; the equivalent gap in the *GEMM* handler is now
   CLOSED (PART 5).  Recommended independently of (3).
5. ~~`LORRAX_FFT_FFI` three-state grammar with a quiet per-call dtype
   fallback~~ — **the equivalent for the GEMM dial is now MOOT** (PART 5:
   the handler serves all four precisions, so there is no dtype boundary
   to fall back from).  If the FFT handlers ever come under the same
   pressure, the owner's ruling here is the precedent: **add the
   precision to the handler, do not add a fallback grammar.**
6. **Upgrading BSE's sharded `W_q→W_R` sites to `make_sharded_ifftn_3d`
   factories** — a gather-avoidance perf change needing its own parity
   gate, not a routing fix.
7. **`vq_interp:240/:249` host `np.fft`** — gates-only recon/to_sphere
   setup, not JAX call sites; converting them would change the execution
   engine, not the routing.  Left as-is deliberately.
8. **Cray LibSci validation of the portable GEMM path** — compile logic
   only; no Cray machine in this campaign.  Also note a LibSci host
   build gets the GEMM handler but NOT the mklfft/DFTI handlers (that
   block is still inside the MKL branch), so `LORRAX_FFT_FFI` refuses
   there.
9. **GPU execution of `contract_bands_block_reshard`** — every GPU row
   in the robustness table is a code/platform-table reading, never a
   measurement.  No rtx-dev cell was run for this workstream.
