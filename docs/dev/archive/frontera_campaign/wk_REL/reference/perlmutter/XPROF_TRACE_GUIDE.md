# Profiling LORRAX GW Performance

Use this when a LORRAX GW calculation is slower than expected. Follow the steps in
order — each is cheaper than the next, and most investigations end before Step 3.

## Step 0: Read the timing summary and check configuration

Every LORRAX GW run prints a `--- Timing ---` table at the end of stdout.

```
--- Timing ---
Section                              Total[s]    %
gw_jax.zeta_fit_chunked                  4.7    1.1
gw_jax.V_q_compute                       1.3    0.3
gw_jax.chi0_W                            2.9    0.7
gw_jax.ppm_sigma                       417.2   97.5
```

Which section dominates?

| Section | Likely bottleneck | What to check first |
|---------|-------------------|---------------------|
| `ppm_sigma` | Minimax solver or XLA compilation | See checklist below |
| `zeta_fit_chunked` | I/O or ISDF solve | r-chunk count, H5 write sub-timings |
| `chi0_W` | Screening | minimax node count, memory pressure |
| `V_q_compute` | Coulomb kernel | mu-chunk count (usually fast) |

**If `ppm_sigma` dominates, check these before instrumenting anything:**

1. Is `regenerate_minimax_tables` set to `true` in `cohsex.in`? If so, every run
   recomputes crossing quadratures from scratch (~95s each). Set it to `false` to use
   shipped tables or the disk cache.

2. Is `LORRAX_MINIMAX_CACHE_DIR` set? If so, repeated runs with the same parameters
   will hit the cache. If not set, each run solves independently.

3. Are you running the `profile_ppm` branch or `main`? On unfixed `main`, eager-mode
   XLA compilation costs ~410s on first invocation. On `profile_ppm` with JIT kernels,
   the same work takes ~2s.

**Expected runtimes** (MoS2 1x1 Gamma-only, 1 GPU, `profile_ppm` branch):

| Configuration | `ppm_sigma` | Total |
|---------------|-------------|-------|
| Cold (no cache, no shipped tables) | ~107 s | ~137 s |
| Warm minimax disk cache | ~19 s | ~46 s |
| Shipped tables only | ~19 s | ~46 s |

If your times are much worse than this for the same system size, proceed to Step 1.

## Step 1: Targeted synchronization timing

The cheapest way to separate GPU compute from compilation/host overhead. Add
`block_until_ready()` calls gated on an environment variable.

**Where to instrument** (the hot paths in LORRAX):

| Hot path | File | What to time per iteration |
|----------|------|---------------------------|
| PPM sigma tau loop | `ppm_sigma.py:_convolve_sigma_branch_kij` | `get_G_mu_nu_fn`, `build_ppm_w_time_q`, `get_G_R_fn` + `get_sigma_mu_nu_fn`, `get_sigma_kij_channels_fn`, omega accumulation |
| Screening | `w_isdf.py:compute_chi0_minimax` | chi0 contraction, W solve |
| ISDF fitting | `gw_jax.py:fit_zeta_and_compute_V_q_chunked` | (already instrumented in the timing sub-breakdown) |

Pattern:
```python
import os, time
_PROF = bool(os.environ.get("LORRAX_PROFILE_PPM", ""))

if _PROF: _t0 = time.perf_counter()
result = jax_operation(...)
if _PROF:
    result.block_until_ready()   # forces GPU sync — without this, timing is meaningless
    print(f"  [PROF] op: {time.perf_counter()-_t0:.3f}s")
```

Run with `LORRAX_PROFILE_PPM=1`. Omit the variable for zero overhead.

**Interpretation:**

| Observation | Diagnosis | Action |
|-------------|-----------|--------|
| Loop iterations sum to N seconds, section reports 100N | Compilation or host-side setup dominates | Go to Step 2, or check if you're on unfixed `main` |
| First iteration 100x slower than the rest | Per-shape XLA compilation | `@jax.jit` the loop body |
| All iterations are uniformly slow | GPU work itself is the bottleneck | Check shapes, sharding, fusion |
| Loop time is small, but the _call_ enclosing the loop is slow | Host-side Python between the profiled region and the caller | Instrument the caller more finely; common cause is minimax solver |

## Step 2: Capture an xprof trace

Use this when Step 1 shows a gap you can't explain from instrumentation alone.

### Prerequisites

LORRAX has `common/jax_profile.py` which writes trace bundles when
`ISDF_JAX_PROFILE_DIR` is set. Stages wrapped in `jax_profile.trace_section("name")`
get individual bundles. Current hooks: `V_q_compute`, `chi0_W`, `ppm_sigma` (on
`profile_ppm`).

To trace an unhooked stage:
```python
from common import jax_profile
with jax_profile.trace_section("my_stage"):
    do_work()
```

### Capture

Use a fresh compilation cache so the trace shows cold-start behavior. Use `$PSCRATCH`
(not `/tmp`) on Perlmutter so the cache persists across srun calls on different nodes.

```bash
cd /path/to/run/directory

rm -rf $PSCRATCH/.jax_cache_trace && mkdir -p $PSCRATCH/.jax_cache_trace

SITE=$HOME/scratchperl/.isdf/isdf_venvs/isdf_site
LORRAX_SRC=/global/u2/j/jackm/software/lorrax/src   # adjust to your source tree

srun --jobid=$JOBID --gres=gpu:1 -N 1 -n 1 \
    shifter --module=gpu --image=nvcr.io/nvidia/jax:25.04-py3 \
    --env=PYTHONPATH=$LORRAX_SRC:$SITE \
    --env=JAX_ENABLE_X64=1 \
    --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env=JAX_COMPILATION_CACHE_DIR=$PSCRATCH/.jax_cache_trace \
    --env=ISDF_JAX_PROFILE_DIR=$PWD/jax_profiles \
    python3 -u -m gw.gw_jax -i $(pwd)/cohsex.in \
    > gw_traced.out 2>&1
```

Artifacts land under:
```
jax_profiles/<section>-YYYYMMDD-HHMMSS-pN/plugins/profile/<timestamp>/
    hostname.trace.json.gz
    hostname.xplane.pb
```

### Inspect the trace JSON (works on login nodes, no GPU needed)

```python
import gzip, json
from collections import Counter
from pathlib import Path

trace_gz = next(Path("jax_profiles").rglob("*.trace.json.gz"))
with gzip.open(trace_gz, "rt") as f:
    data = json.load(f)

ctr = Counter()
for ev in data.get("traceEvents", []):
    if ev.get("ph") == "X" and isinstance(ev.get("name"), str):
        ctr[ev["name"]] += ev.get("dur", 0)

for name, dur_us in ctr.most_common(50):
    print(f"{dur_us/1e6:8.1f} s  {name}")
```

Or `tensorboard --logdir jax_profiles --port 6006` if you have a browser.

## Step 3: Interpret and act

**Reconcile the trace with stdout.** Do not trust only one view. The pattern that works:
1. Stdout timing says which section is slow and how much of the time the profiled
   GPU work accounts for
2. xprof says what the host was doing during the gap
3. Source reading confirms whether the expensive path is fixable

| Trace pattern | Diagnosis | Fix category |
|---------------|-----------|--------------|
| XLA compile events >> 10s each | Eager-mode function needing JIT | **JIT/fuse** |
| Many short compiles summing >> 10s | Python-scalar loop causing retrace | **JIT** (pass as arrays) or `jax.lax.scan` |
| Large host block, no GPU underneath (e.g. `solve_crossing`, `_polish`) | CPU-bound solver | **Cache** or algorithm change |
| Dense GPU kernels, uniformly slow | GPU compute is the real cost | **Sharding/fusion/batching** |
| Warm cache eliminates the cost | Pure compilation overhead | Set `JAX_COMPILATION_CACHE_DIR` persistently |

### Identifying FFT memory peaks

The 3D FFT (`jnp.fft.ifftn`) is the dominant memory consumer during wavefunction
loading and centroid extraction. At peak, it holds **4 copies** of the per-device
psi shard (input + output + 2 staging buffers for the x→y→z decomposition).

To verify the FFT is the bottleneck (not a sharding issue or allocation leak):

```python
# Quick per-device memory check at key stages
import jax, gc
def mem():
    gc.collect()
    s = jax.local_devices()[0].memory_stats()
    return s['bytes_in_use'] / 1e9, s['peak_bytes_in_use'] / 1e9

# In a fresh process, allocate and FFT a single shard:
psi = jnp.zeros((nk, nb_shard, ns, nx, ny, nz), dtype=jnp.complex128)
shard_gb = psi.nbytes / 1e9
psi_r = jnp.fft.ifftn(psi, axes=(-3,-2,-1)); psi_r.block_until_ready()
_, peak = mem()
print(f'shard={shard_gb:.3f} GB, peak={peak:.3f} GB, ratio={peak/shard_gb:.2f}x')
# Expected: ratio ≈ 4.0x for large shards, slightly higher for small shards.
```

If the ratio is significantly above 4× (e.g., 10× or more), look for:
- **Cumulative peak contamination**: `peak_bytes_in_use` is a high-water mark
  that never resets within a process. Run each test in a separate srun/process.
- **Sharding rematerialization**: XLA log messages with "involuntary full
  rematerialization" indicate a resharding transition that forces XLA to
  recompute the full tensor. This adds the full unsharded size to the peak.
  Fix by adding intermediate `with_sharding_constraint` steps or avoiding
  the problematic sharding transition.
- **Unfused phase multiply**: After FFT, `psi_r * phase` may allocate a new
  buffer if XLA doesn't fuse it with the previous operation. This adds a 5th
  copy briefly, but the FFT staging buffers are freed by then.

### Identifying SPMD resharding OOMs

When `_sharding_constraint_impl` OOMs, XLA is trying to reshard an array from
one device layout to another. The XLA log will say:

```
Can't reduce memory use below X GiB by rematerialization; only reduced to Y GiB
```

This means XLA needs Y GB to execute the resharding, but only X GB is free.
The Y GB includes the input array (which must be fully materialized to reshard)
plus the output array. The fix is to avoid the problematic transition — e.g.,
by resharding through an intermediate layout that XLA can handle incrementally.

## Profiling the ISDF fitting pipeline

The ISDF zeta fitting (`zeta_fit_chunked` in `isdf_fitting.py`) has distinct
stages, each with a different memory profile. The stdout timing reports
sub-sections that map directly to code stages:

```
gw_jax.zeta_fit_chunked                    17.5 s   35%
  zeta_fit.load_wfns                         3.6 s    7%    ← centroid extraction (FFT-bound)
  zeta_fit.CCT                               0.4 s    1%    ← pair density + cross-correlation
  zeta_fit.cholesky                          1.6 s    3%    ← 2D blocked Cholesky of C_q
  zeta_fit.cache_gspace                      0.4 s    1%    ← G-space cache for r-chunks
  zeta_fit.chunk_loop                        3.2 s    6%    ← r-chunk processing loop
    zeta_fit.chunk.load                        0.7 s         ← FFT + phase for this r-chunk
    zeta_fit.chunk.pair_density                0.0 s         ← spin-traced P_l, P_r
    zeta_fit.chunk.ZCT                         0.5 s         ← FFT-based ZCT contraction
    zeta_fit.chunk.solve                       0.3 s         ← triangular solve ζ = L⁻¹ Z
    zeta_fit.chunk.h5_write                    1.6 s         ← async H5 write (overlapped)
```

### Memory stages during ISDF fitting

The per-device memory timeline follows this sequence (measured on Si 4×4×4,
4 A100 GPUs):

| Stage | GPU used | Peak | What's on device |
|-------|----------|------|-----------------|
| Before ISDF | 0.00 GB | 0.00 | Clean |
| `load_wfns` (centroid FFT) | 0.52 | **3.50** | FFT peak: 4× psi shard |
| After centroid extract | 0.07 | 3.50 | Only psi_rmu + psi_rmuT |
| `cache_gspace` | 0.55 | 3.52 | Cached G-space for r-chunks |
| `chunk.load` (r-chunk FFT) | 1.40 | 6.52 | G-cache + r-chunk psi_rtot |
| `chunk.ZCT` | 3.97 | 7.37 | + pair densities + Z_q |
| After chunk loop | 0.55 | 7.37 | G-cache still resident |

**Key observations:**
- The centroid FFT peak (3.50 GB) is from 4× the psi shard during the 3D FFT
- The G-space cache (0.55 GB) persists through all r-chunks
- The r-chunk ZCT peak (7.37 GB) includes pair densities, Z_q, and the solve
- After the chunk loop, only the G-cache remains (freed explicitly)

### How to profile per-stage memory

Instrument with `jax.local_devices()[0].memory_stats()` at key points.
**Critical**: use a **fresh process** for each test (cumulative peak
never resets) and `gc.collect()` before each measurement.

```python
import jax, gc

def mem(label):
    gc.collect()
    s = jax.local_devices()[0].memory_stats()
    u = s['bytes_in_use'] / 1e9
    p = s['peak_bytes_in_use'] / 1e9
    print(f'  [{label}] used={u:.2f} GB, peak={p:.2f} GB')
```

Patch the entry points in `isdf_fitting.py` (e.g., `fit_zeta_chunked_to_h5`)
to call `mem()` before/after each stage. See the handoff notes in
`reports/si_10x10x10_timing_2026-04-06/oom_handoff.md` for the monkey-patching
pattern used in the 4×4×4 profiling session.

### Which stages are memory-bound vs compute-bound

| Stage | Bound by | Scaling | Notes |
|-------|----------|---------|-------|
| `load_wfns` | **Memory** (FFT) | 4 × nk × (nb/P) × ns × n_rtot × 16 | Peak during FFT |
| `CCT` | Compute (einsum) | O(nk × n_rmu² × ns) | Small for typical n_rmu |
| `cholesky` | Compute (BLAS) | O(n_q × n_rmu³) | 2D blocked across mesh |
| `cache_gspace` | Memory (persistent) | nk × (nb/P) × ns × n_rtot × 16 | Stays resident |
| `chunk.load` | **Memory** (FFT) | Same as load_wfns but for r-chunk bands | Shares G-cache |
| `chunk.ZCT` | **Memory** (FFT + matmul) | 4 × nk × n_rmu × B_r × 16 | ZCT FFT peak |
| `chunk.solve` | Compute (trsm) | O(n_q × n_rmu² × B_r) | Dominated by BLAS |
| `chunk.h5_write` | **I/O** | B_r × n_rmu × n_q × 16 | Overlapped with next chunk |

### Known LORRAX cost centers

| Cost center | Location | Cold time | With fix | Status |
|-------------|----------|-----------|----------|--------|
| Eager tau-loop compilation | `ppm_sigma.py:_convolve_sigma_branch_kij` | ~410 s | ~2 s | **Fixed** (`@jax.jit` on `profile_ppm`) |
| Crossing minimax solver | `minimax.py:solve_crossing` via `_build_three_sigma_windows` | ~95 s | ~0 s (cache/shipped) | **Fixed** (disk cache + shipped tables) |
| Actual GPU sigma work | tau-loop einsums + FFTs | ~5 s | ~2 s | Healthy |
| ISDF fitting | `gw_init.py` | 2-5 s | — | Healthy |
| chi0 + W screening | `w_isdf.py:compute_chi0_minimax` | 1-3 s | — | Healthy |

## Step 4: Checkpoint

After a useful profiling investigation:
- Preserve the traced run in a dedicated variant directory (e.g. `03_lorrax_xprof/`)
- Save stdout alongside the trace bundle
- Note the exact source commit
- Update `report.md` and `CHANGELOG.md`

## Reference trace artifacts

The 2026-04-05 profiling session artifacts are in
`runs/MoS2/00_mos2_1x1_profiling/03_lorrax_profile_ppm_xprof/jax_profiles/`.
Source: `profile_ppm` @ `d0114a2`.