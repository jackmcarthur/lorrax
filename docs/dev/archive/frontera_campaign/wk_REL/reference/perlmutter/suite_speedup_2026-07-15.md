# Test-suite speedup — cache audit + bispinor fixture regen at 25 Ry

Branch: `agent/suite-speedup` on `sources/lorrax_D` (stacked on
`agent/ppm-fit-conditioning`). Goal (user): plain 1-GPU suite wall from
~4–11 min toward <60 s for the dev loop. Follows the 2026-07-09 redesign
(`reports/test_suite_redesign_2026-07-09/`, 578→~220 s).

## Finding 1 — the XLA compile cache is already working (not a lever)

`~/.cache/isdf_jax_compilation` (5.5 GB) is active for driver runs, and the
pytest harness additionally pins gate subprocesses to a repo-local
`.pytest_jax_cache`. Measured on the gnppm gate (single test, 1 GPU):

| condition | wall |
|---|---|
| warm cache (status quo) | 27.3 s |
| `ISDF_JAX_CACHE_DIR=""` (compile from scratch) | 117.0 s |
| import floor (python + jax + gw_jax, no work) | ~4.3 s |

The cache already saves ~90 s on this gate alone. Corollary: the 660 s suite
run of 2026-07-15 morning was cold-cache fallout from the Fix-3 PPM change
altering the fit jaxpr (new cache keys, every PPM gate recompiled); the warm
rerun the same day completed in ~390 s. **Suite walls quoted after any
source change are cold-cache outliers; always quote the second run.**

## Finding 2 — bispinor fixture regen at 25 Ry (landed, modest win)

Full regen at `runs/MoS2/D_25Ry_bispinor_fixture_2026-07-15/`: QE SCF/NSCF
ecutwfc 60→25 Ry (FFT grid 30×30×120 → **20×20×75**, 3.6× fewer points),
nbnd=40 → `truncate_bands.py` → 34-band WFN.h5 (**14.8 MB**, was 52.6),
`gw.kin_ion_io`, kmeans regen same seeds/flags (transverse orbit set closes
at 208, was 209). Frozen into `tests/regression/bispinor_debug/`.

Validation (all on 1 A100):
- 3 consecutive fresh runs **bit-identical** (2160 sigma values, max|Δ|=0.0).
- `LORRAX_EXTRA_MU_PAD=4` pad twin **bit-identical** to baseline.
- Charge full-BZ-direct + transverse IBZ cascade properties preserved
  (log-asserted, as before).
- **Chunk coverage requirement (user)**: all 4 ζ-fit passes run **3 r-chunks
  at `memory_per_device_gb = 30`** — no budget lowering needed — and the
  Tier-1 gate now LOG-ASSERTS ≥2 chunks in every channel so a future
  shrink cannot silently collapse the streaming seam to one chunk.

Timing, the honest part:

| measure | 60 Ry fixture | 25 Ry fixture |
|---|---|---|
| recorded driver wall (warm) | ~40 s | 29 s |
| in-suite: fresh gate + pad twin (warm) | 42.7 + 44.5 = 87.2 s | 42.0 + 37.2 = **79.3 s** |

**The 2026-07-09 floor diagnosis ("fixed cost = r-chunk streaming over the
FFT grid") does not survive this experiment**: 3.6× fewer grid points bought
9% of in-suite wall. The actual per-gate floor is subprocess launch + jax
import (~4–5 s) + per-process retracing of the full pipeline graph
(CPU-bound, grid-independent) + cache deserialization. Keep the regen
(smaller blob, faster cold compiles, chunk gate), but grid physics is
exhausted as a suite-wall lever.

## Finding 3 — one-process Tier-2 variant bundle (landed, commit `a506a71`)

All 7 gnppm Tier-2 variants (restart baseline, pad12, kij_stream, sc_iter1,
fixed_point, IBZ leg A/B) now run sequentially in ONE subprocess
(`tests/run_variant_bundle.py`, driven by `conftest.gnppm_variant_bundle`):
import + distributed init once, shared jitted kernels where shapes match,
per-variant env knobs applied/restored in-process (both LORRAX_* knobs are
read at call time — verified), per-variant stdout capture, per-variant
failure isolation. Cache-safety audited: all module-level caches on the
pipeline path hold jitted functions or pure math keyed on full inputs.

Measured (1 GPU, warm): Tier-2 chain ~114 s → ~71 s (session subprocess
~15 s + bundle ~55 s); **full suite 373.6 → 283.0 s**, 207 passed / 0 failed.

## Suite-wall ledger (plain 1-GPU serial, warm)

| point | wall |
|---|---|
| 2026-07-09 redesign benchmark | ~220 s (176 tests) |
| + FFI contract suite (07-10) + drift | ~390 s (207 tests) |
| + 25 Ry bispinor fixture (`d1847d9`) | 373.6 s |
| + variant bundle (`a506a71`) | **283.0 s** |

Remaining composition ≈ bispinor fresh+pad-twin 79 s, bundle 55 s, FFI
contract 47 s, units+collection ~32 s, Si-3D 21 s, gnppm session 15 s,
cohsex 8 s, misc.

## Where <60 s actually lives now (user decision)

The full suite cannot reach 60 s while it runs four fresh e2e pipelines and
the FFI backend matrix serially. The realistic split:

- **Dev loop**: `-m "not regression"` (units, ~30 s) — exists today; adding
  the variant bundle to it would give invariance coverage at ~85 s, or keep
  units-only for <60 s.
- **Checkpoint**: the full 283 s suite (contract unchanged).
- Optional further trims: run the bispinor pad twin only at checkpoints
  (coverage-vs-latency call, not free). NOT a trim: the "13 s" SLATE
  trsm[48] case — measured by reordering, that cost is first-SLATE-call
  initialization (lib load + CUDA/SLATE setup + first FFI compile) landing
  on whichever SLATE test runs first; the test itself is ~2 s and cutting
  it just moves the init to its neighbor.

## Evidence ledger

- Suite (full, warm, this branch): see CHANGELOG entry / suite log.
- Gate validation runs: `runs/MoS2/D_25Ry_bispinor_fixture_2026-07-15/`
  (`gw_run{1,2,3}.log`, `gw_padtwin.log`, `kmeans_*.log`, `kin_ion_gen2.log`).
- Stale-tool note: `psp.get_DFT_mtxels` main() (the known-stale debug driver,
  NEXT_TARGETS #12) crashes on a ψ-box/ρ-box broadcast before reaching its
  kin_ion writer — `gw.kin_ion_io` is the canonical generator and works.
