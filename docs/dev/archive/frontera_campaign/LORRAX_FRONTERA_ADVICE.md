# LORRAX on Frontera — operational advice for agents

**Purpose:** hard-won, Frontera-specific knowledge for running LORRAX (JAX GW-BSE)
fast and without dead ends. Read this before launching jobs. Maintained by the
agents working this machine — **keep it current: when you discover something that
would have saved you an hour, add it here.**

Last meaningful update: 2026-07-23.

---

## 0. TL;DR — the fast path

- **Small / fits-one-node → single-node, single-process CPU.** No
  `jax.distributed`, starts in ~12 s, gets 168 GB RAM. This is the default for
  anything that fits. Do NOT reach for multi-node until a single node genuinely
  can't hold it.
- **GPUs (RTX 5000, 16 GB) are tiny.** Anything on a 12×12 k-grid OOMs them
  regardless of centroid count (a single intermediate is ~26 GB). Use GPUs only
  for small grids / small fixtures.
- **The compile cache is SAFE at P>1 now** (2026-07-27, workstream AH). The old
  advice "always `ISDF_JAX_CACHE_DIR=\"\"`" is obsolete: the deadlock was the
  *per-rank* cache partitioning, not a shared cache, and it is fixed in-tree
  (`common/jax_compile_cache.py`). Warm runs hit on **every** rank (CPU P=8
  htransform: 152 compiles/rank → 0). **But on CPU it buys ~1 %** — measured at
  606c/P=16, the whole compile storm is only ~5 s of XLA per rank. On GPU it is
  3.6x. Turn it on; do not plan around it. See §4.
- **Use the venv python explicitly:** `$WORK/lorrax_env/.venv/bin/python`. Bare
  `python` in the container is the system 3.12 with no jax → `ModuleNotFoundError`.
- **Set `memory_per_device_gb` to ~half of physical RAM on CPU** (e.g. `90`), not
  `0`/auto. The chunk-planner's HWM model under-counts CPU peak by ~40-50 GB and
  auto-detect (168 GB) will `std::bad_alloc` (see §5).

---

## 1. Environment / where things live

- `$WORK` = `/work2/08271/jackmc/frontera`. `$SCRATCH` = `/scratch2/08271/jackmc`
  (Lustre, big, purgeable). `$HOME` = `/home1/08271/jackmc`.
- LORRAX repo: `$WORK/lorrax` (branch **`frontera-ffi`** is the Frontera work
  branch; `main` stays Perlmutter-pristine). Source is `$WORK/lorrax/src` — put it
  on `PYTHONPATH`.
- venv: `$WORK/lorrax_env/.venv` (jax[cuda12] etc.).
- Container: `$SCRATCH/lorrax_setup/py312.sif` (python:3.12-bookworm, glibc 2.36).
  **Required** — Frontera is CentOS 7 / glibc 2.17 and modern jax wheels need
  ≥2.28. `apptainer` is **blocked on login nodes**; only runs inside a job.
- Allocation: **`PHY25006`** — UPPERCASE for `sbatch -A`.
- MoS2 reference run dir: `$SCRATCH/mos2_80ry_12x12/`.

## 2. `/home1` inode quota — the thing that locks you out

`/home1` has a **200k inode** limit (not size — inodes). When it fills you can't
even ssh in. Currently ~147k used. Rules:

- **Never let caches or many-file outputs land in `$HOME`.** JAX/LORRAX compile
  cache defaults to `~/.cache/isdf_jax_compilation` — redirect it (`ISDF_JAX_CACHE_DIR`)
  to `$SCRATCH` or disable it entirely (`=""`).
- Big inode-heavy dirs (miniconda3, .vscode-server, old builds) get relocated to
  `$WORK` by `$WORK/relocate_home_dirs.sh` — run from a plain ssh shell, not inside
  Claude Code / VS Code.

## 3. Launching jobs

- Load apptainer in the job: `module load tacc-apptainer`.
- Bind the filesystems you touch: `apptainer exec --bind /home1,/work2,/scratch1,/scratch2 $SIF ...`
  Add `--nv` **only** for GPU jobs. CPU jobs must NOT pass `--nv`.
- Queues: `development` (CPU CLX nodes, 56 cores, ~192 GB, **up to 394 nodes**,
  2-hr limit, fast turnaround); `rtx-dev` / `rtx` (4× RTX 5000 per node, 16 GB each).
- **`development` allows only 2 concurrent jobs.** A third is rejected at submit
  with `Job violates accounting/QOS policy (job submit limit ...)`. Plan around
  it: pair a long run with one short one, and use `normal` for anything needing
  more than the 2-hour wall.
- **Node count has a SWEET SPOT — more is not better.** See §5b: the centroid
  ψ(G-flat) load allocates a full host array that *grows* with world_size, so
  32 nodes OOMs where 8 works. The ζ-fit also scales sub-linearly (§ scaling
  table). Practical range for this system: **4–8 nodes**. Do not reflexively
  max out the allocation; measure.
- CPU thread env: `OMP_NUM_THREADS=56 OPENBLAS_NUM_THREADS=56`. XLA:CPU uses its
  own threadpool sized to cores automatically.

## 4. The compile cache — FIXED at P>1 (workstream AH). Leave it ON.

> **2026-07-27 — THIS SECTION WAS WRONG TWICE. Read this box before anything
> below it.**
>
> **(a) "A shared cache dir deadlocks the compile barrier" is INVERTED.** What
> deadlocked was the **per-rank partitioning** (`{base}/np{P}/rank{i}/`). JAX
> writes persistent-cache entries from **process 0 only**
> (`jax/_src/compiler.py::_cache_write`, unconditional), so `rank0/` filled up
> (882 entries measured) while `rank1..P-1/` stayed **empty forever**. On the
> next run process 0 HIT and skipped compilation while its peers MISSED and
> compiled — and **XLA:GPU compilation is a collective**: `AutotunerPass`
> shards autotuning across processes and exchanges results through the JAX
> coordination service, so a process that does not compile never publishes its
> share and every peer blocks forever in `CoordinationServiceAgent::GetKeyValue`.
> Full derivation: **scorecard AG**. A *shared* directory is the fix direction,
> not the disease.
>
> **(b) A naive shared dir does fail too — for three DIFFERENT reasons, all now
> handled (scorecard AH).** (1) The cache KEY is not process-invariant: jax
> 0.9.1 strips the device assignment only when `platform == "gpu"`, and hashes
> the accelerator config as a serialized topology blob that carries
> process-local content — MEASURED, the first diverging key component on CPU is
> `accelerator_config`, so only process 0 (the only writer) ever hits.
> (2) `LRUCache.put` is a bare `write_bytes`, **not** tmp+rename, so a
> concurrent reader can get a truncated entry and silently turn a hit into a
> miss. (3) With the persistent cache on, JAX also auto-enables XLA's own
> per-fusion autotune cache in `UPDATE`(p0)/`READ`(peers) mode, which changes
> the set of fusions each process still has to autotune — i.e. the input to
> AutotunerPass' modulo-P work split — and desynchronises it one level down.
>
> **What to do now:** `common/jax_compile_cache.py` makes the key
> process-invariant, uses ONE shared dir per world size, agrees on the usable
> entry set over the coordination-service KV store before any compile, writes
> atomically, and disables the XLA sub-caches. **`ISDF_JAX_CACHE_DIR=""` is no
> longer needed and costs you the win** — leave the cache on (point it at
> `$SCRATCH` if you like). Measured: GPU 4-rank fixture 73 s → **20 s**; CPU
> P=8 htransform **152 compiles/rank → 0**; CPU P=8 gw_jax **373 → 5**.
> `LORRAX_JAX_CACHE_MULTIPROCESS=0` restores the old refusal if you ever need
> it. The stale `~/.cache/isdf_jax_compilation/np*/rank*/` dirs (14 MB, ~1700
> inodes) were deleted on 2026-07-27.
>
> **But do NOT expect the "158 s win" the old TODO promised.** Measured at 606
> centroids / P=16 (8 nodes): 174 modules/rank, **~5 s of XLA compilation per
> rank**, and every rank compiles concurrently — so a perfect cache removes
> about **1 %** of a 431 s CPU run. The compile storm is a large COUNT of cheap
> compiles; the cost behind that count is jaxpr tracing, lowering and eager
> dispatch, which the persistent cache is consulted *after* and cannot remove.
> The CAUTION box below was right all along. On GPU it is a different story —
> autotuning makes compilation cost 38.6 s/rank, and the warm run is 3.6x
> faster end to end.
>
> **Two rules for WHERE the cache lives.** (i) Put it on `$SCRATCH`:
> `export ISDF_JAX_CACHE_DIR=$SCRATCH/lorrax_jaxcache`. One entry is one small
> file and a populated world size is several hundred of them, so the `~/.cache`
> default burns the `/home1` *inode* quota that §2 says locks you out.
> (ii) It must be visible to EVERY rank — `$SCRATCH`/`$WORK` are, `/tmp` is
> not. A node-local dir gives node 0 a full cache and every other node an empty
> one, which is the original asymmetry all over again. Measured A/B on 2 nodes
> x 2 GPUs with `ISDF_JAX_CACHE_DIR=/tmp/...` (node A: 286 entries, node B: 0):
> without the in-tree agreement the warm run **hangs (rc=124 at 600 s)**; with
> it the same run completes in 48 s, printing `287 entries DROPPED`. So the
> mistake is now loud instead of fatal — but you still get no win.

### Historical note (the ORIGINAL, separate "poisoned cache" problem)

> **ALSO CORRECTED 2026-07-27.** These messages still appear on every warm run
> — 738 of them in one rank log — and they are **log noise, not rejections**.
> `prefer-no-gather` / `prefer-no-scatter` are LLVM *cost-model pseudo-features*
> that exist at compile time and never appear in a runtime CPU feature list, so
> `cpu_aot_loader.cc`'s comparison mismatches on every load on every machine.
> It is a `LOG(ERROR)`, not a reject. Measured on the very run that printed 738
> of them: **369/373 cache hits, 4 compiles**; on htransform, 304 lines with
> **152/152 hits and ZERO compiles**. So "each forcing a recompile" below is
> wrong for jax 0.9.1 — do not disable the cache because of these lines.

Symptom: thousands of
`Loading XLA:CPU AOT result. Target machine feature +prefer-no-gather is not
supported on the host machine ... machine type doesn't match` errors, each forcing
a recompile. A run that should compile once in ~7 min instead spends ~11 min in a
reload-reject-recompile storm.

Cause: XLA:CPU AOT cache entries embed a target-config (incl. cost-model pseudo-
features like `+prefer-no-gather`) that the loader rejects on a different node /
container view of the CPU. The cached entries never hit; they only cost time.

Fix: point `ISDF_JAX_CACHE_DIR` at a **fresh** directory on `$SCRATCH` (LORRAX reads
this var; note it is NOT the same as `JAX_COMPILATION_CACHE_DIR`). Do **not** just
disable caching — measured on a fresh cache:

| | first compile |
|---|---|
| cold | 7.1 s |
| warm, same node | 0.1 s |
| warm, **different** node | 1.6 s |

So the cache *does* work on Frontera CPU, including cross-node — the original
problem was a **stale, mixed-context** cache in `~/.cache`, not an incompatibility.
Clear that one; keep a fresh shared cache on scratch.

> **CAUTION — do not over-value this, and do not blame compile for slow runs.**
> Measured with NO cache at all, six *distinct* kernel shapes compiled in **0.1 s
> each**; only the first paid ~11 s, which is one-time XLA/LLVM process warmup.
> The real driver agrees: `chi.compile 1.2 s`, `W.compile 0.26 s`,
> `sigma.compile 0.5 s`. **Compilation is not a bottleneck in this code.** An
> earlier version of this document claimed a "~7 min compile"; that was a
> misattribution — the startup time is data movement (WFN read, centroid ψ load,
> C_q, L_q), not codegen. And the cache only helps when the HLO is *identical*, so
> it does nothing for development where shapes/kernels change.
>
> Corollary: XLA compile-time flags (`xla_backend_optimization_level`,
> `xla_cpu_parallel_codegen_split_count`, `xla_cpu_disable_slp_vectorizer` — all
> real and present in this jaxlib) are **not worth setting**; there is no
> meaningful compile time for them to remove. If you A/B them, run each config in
> a *separate cold job* — sequential configs in one job are confounded by page
> cache and will show a fake 10× "win" for whatever ran second.

The `cuInit(0) failed: Unknown CUDA error 303` line on CPU nodes is **harmless** —
JAX catches it and falls back to CPU. Set `CUDA_VISIBLE_DEVICES=""` and
`JAX_PLATFORMS=cpu` to keep it quiet-ish; it does not add meaningful delay.

## 5. Memory — the real wall on this machine

- **RTX 5000 = 16 GB.** The converged MoS2 80Ry/12×12/400-band system produces a
  single ~26 GB intermediate (144 q-points × 36×36×135 real grid) during the
  ISDF/ζ path. It OOMs the GPU **at any centroid count** — even 276 centroids.
  → GPU is out for 12×12. Reference Perlmutter run needed A100-80GB × 16.
- **Single CPU node** auto-detects a **168 GB** budget (192 physical). It fits the
  26 GB arrays fine. BUT the chunk-planner's `HWM estimate` **under-counts the real
  CPU peak by ~40-50 GB** (host malloc + ψ(G-flat) host cache + WFN loader resident
  + XLA scratch aren't all modeled). Auto budget (168, targeting 143 GB) →
  `std::bad_alloc` on ζ-fit r-chunk 1.
  → **Set `memory_per_device_gb = 90`** (targets ~76 GB, real peak ~125 GB, safe).
  Lower it further if you still OOM. `chunk_target_utilization` is floored at 0.85,
  so the budget number is your only real knob.
- **Centroid generation** (`kmeans_cli --orbit`) has its own wall: the pivoted-
  Cholesky candidate pruning builds a Gram over all M candidates and allocated
  **192 GB** for M≈4566 (400 reps × 12 symops) → OOM on one node. Small counts
  (≤~300) are fine on one node; large counts need multi-node or an M-chunked prune.

## 5b. The centroid ψ(G-flat) load OOMs when you use TOO MANY nodes

Counter-intuitive but load-bearing: **adding nodes makes this allocation bigger.**

`common/wfn_transforms.load_centroids_band_chunked` → `file_io/wfn_loader.load`
ends in `jax.device_put(psi_np, named_sharding)`. `psi_np` is the **full global
array as host numpy** — `device_put` shards only *after* every rank has
materialised the whole thing. On top of that the band extent is **rounded up to a
multiple of `world_size`** (`b_id_4 = _round_up(b_id_4_user, world_size)`), and
the leading axis carries a world_size factor too. Measured for MoS2 12×12,
nband=120, ngkmax=8603:

| nodes | array shape | host alloc | result |
|---|---|---|---|
| 4  | (576, 120, 2, 8603)  | ~19 GiB  | fine |
| 8  | (1152, 120, 2, 8603) | ~38 GiB  | fine |
| 16 | (2304, 128, 2, 8603) | ~81 GiB  | tight |
| 32 | (4608, 128, 2, 8603) | **151 GiB** | `numpy ArrayMemoryError`, every rank dies |

The in-code comment ("~50 GB total / mesh.size") describes the *sharded* size —
that is not what the host has to allocate to get there.

**So: 4–8 nodes for this system.** If you need genuinely large multi-node runs,
the real fix is to build the array per-process (e.g.
`jax.make_array_from_process_local_data`) instead of `device_put`-ing a full host
copy — not yet done.

## 6. Multi-node CPU — the `jax.distributed` startup is SLOW (~5 min)

The "3-minute hang" was **never a deadlock** and **is now fixed by env**. A naive
2-node CPU launch measured **~297 s** to `jax.distributed.initialize()`. With the
right env it is **~18 s total, of which initialize() is 0.4 s** — the rest is just
`import jax`. Fine-grained diagnostic (`$SCRATCH/mos2_80ry_12x12/init_diag.py`):
```
[rank 0 +  0.0s] pre-import
[rank 0 + 17.7s] post-import-jax       <- import jax dominates
[rank 0 + 18.1s] post-initialize (initialize took 0.4s)   <- handshake is instant
```

**The fix — set these in the launch env for multi-node CPU:**
```
export CUDA_VISIBLE_DEVICES=""         # no GPU probe
export JAX_COORDINATOR_ADDRESS=<first_host>:<port>   # explicit coordinator; skips
                                       #    the no-arg auto-detect path entirely
# Gloo NIC selection needs NO export: runtime.pin_gloo_interface() (in-tree,
# called from bootstrap()) auto-pins the first UP ib*/hsn* NIC (ib0 here) and
# announces the decision on rank 0; LORRAX_GLOO_IFNAME=<name|off> overrides.
# Do NOT export GLOO_SOCKET_IFNAME (inert with jax) or NCCL_SOCKET_IFNAME
# (GPU-only) on CPU runs — wk_AU stripped them from every live harness.
```
Derive the coordinator as `$(scontrol show hostnames $SLURM_NODELIST | head -1):PORT`
with a **per-launch port** (e.g. `13000 + SLURM_JOB_ID % 2000`) so two runs in one
allocation don't collide. The `cuInit 303` line still prints (lazily, after init) —
harmless.

Notes:
- **Prefer single-node whenever it fits** — it skips distributed entirely
  (`proc_count==1` → `init_jax_distributed()` returns immediately, ~12 s).
- This is a **launch-env fix, not a code change** (keeps `main`/Perlmutter clean).
  Put it in the Frontera launch scripts, not the runtime.
- Latent code bug spotted (not the cause of the stall, still worth fixing):
  `runtime._resolve_coordinator_address()` hardcodes port `12355` while its
  docstring claims a `SLURM_JOB_ID`-derived port → two concurrent runs collide if
  they fall back to it. The explicit `JAX_COORDINATOR_ADDRESS` above sidesteps it.

> **CORRECTED 2026-07-27 (scorecard AF.5 / AK.4 / AL):** the
> `GLOO_SOCKET_IFNAME=ib0` export this section used to recommend is **INERT
> for JAX** — the string appears
> nowhere in the shipped jax/jaxlib, so it never selected anything. The
> 297 s → 18 s startup win came from `CUDA_VISIBLE_DEVICES=""` + the explicit
> coordinator, not from it. Worse: without a real pin every JAX CPU collective
> in this campaign ran over **em1 (1 GbE)**, not ib0 — see §10b for the fix
> that is now IN THE TREE (`runtime.pin_gloo_interface()`, auto-ib0, override
> `LORRAX_GLOO_IFNAME`) and what it changes about every timing in this file.
> Keep `NCCL_SOCKET_IFNAME=ib0` — NCCL (GPU runs) really does read it.

## 6b. Prerequisites that only bite you AFTER the expensive stage

**`dipole.h5` is required for the q=0 Coulomb head.** A GW run will happily do the
whole (hour-long) ζ-fit and *then* die with:
```
RuntimeError: Failed to resolve q=0 Coulomb head:
neither explicit overrides nor supported sources are available.
```
The head resolver (`gw/head_correction.py`) tries, in order: explicit overrides
(needs **both** `vhead` and `whead` in the input) → `wcoul0_source` which is either
`epshead` or **`s_tensor` (the default)** — and `s_tensor` needs **`dipole.h5`**.
Generate it *before* the GW:
```
python -m psp.get_dipole_mtxels -i <your_gw_input>.in --out dipole.h5
```
It reads `wfn_file` from the input's `[cohsex]` block and **scans the input file's
directory for the UPF pseudopotentials** (needs them for the `i[r,V_NL]`
commutator; `--skip-vnl` writes p̂ only, matching BGW's `use_momentum`).

**Run `get_dipole_mtxels` MULTI-NODE for a 12×12 system.** It loads the full
G-space wavefunctions — for 144 k × 120 bands × 2 spinor × 36×36×135 that is
**~97 GB** — and then applies a `with_sharding_constraint`, which on a single
process materialises a *second* full copy → `RESOURCE_EXHAUSTED: Out of memory
allocating 96745881600 bytes` on a 168 GB node. Launch it with the same
`srun -N 4 --ntasks-per-node=1` + §6 fast-init env as the GW so the array actually
shards across the mesh (~24 GB/node on 4). Reducing `nband` in the input is the
other lever if you must stay on one node.

**`kin_ion.h5` is required for the final QP step.** Same story: the run does all
the screening + sigma work, writes `sigma_mnk.h5`, and *then* dies with
```
FileNotFoundError: kin_ion file not found: .../kin_ion.h5
```
`gw/eqp_bgw.py` reads the `kin_ion` dataset (config key `kin_ion_file`). It is
**LORRAX-native — there is no reader for pw2bgw's ASCII `kih.dat`**, so having
run pw2bgw with `kih_flag` does *not* satisfy this, and there is **no Vxc-based
alternative** (LORRAX's QP equation is KIH-form:
`E_qp = kin_ion + V_H + Σ_xc − E_DFT`). Generate it with:
```
python -m gw.kin_ion_io -i <your_gw_input>.in -o kin_ion.h5 --sys_dim 2
```

> **USE `gw.kin_ion_io` — NOT `psp.get_DFT_mtxels`.**
> `psp/get_DFT_mtxels.py` looks like it does the same job (it has a
> `write_kin_ion_h5`) but it is an old debug **scaffold** (its own docstring says
> the nonlocal term is "stubbed") and it is **broken**: `main()` hardcodes
> `grid_rho = 2 × fft_grid` and builds `V_loc` on that dense grid, while
> `_compute_local_V_k_jit` multiplies `psi_r * V_r` pointwise and so requires
> V on the *wavefunction* grid. Result:
> `TypeError: broadcast_shapes got incompatible shapes: (70,2,36,36,135), (1,1,72,72,270)`.
> (Fixing it properly means zero-padding psi up to the dense grid.)
> The reference Perlmutter run used `gw/kin_ion_io.py`, which is k-point-looped,
> memory-light, and finished in **94 s on a single device**. Single node is fine —
> no multi-node needed.

**Generate BOTH prerequisite files up front** — `dipole.h5` and `kin_ion.h5` —
before starting a long GW, or you will discover them one crash at a time.

**`restart = true` reuses `tmp/zeta_q.h5` and skips the ζ-fit.** This is the single
biggest dev-cycle win once a ζ exists: you can iterate on the *screening/sigma*
stages in minutes instead of redoing an hour of fitting. Set it whenever the
centroids/bands/grid haven't changed.

**Check your `K_POINTS` block matches your intent.** `crystal_b` is a
bandstructure *path*; the grid run wants the grid. Also remember config keys must
be **inside** `[cohsex]` — anything after `K_POINTS` gets parsed as k-point floats.

## 6a. ⚠️ DO NOT SET `distributed_cholesky = off` — it silently destroys the physics

**This is the single most expensive trap on this machine.** `off` reads like
"don't use the GPU cuSOLVERMp FFI" (a reasonable thing to want on CPU). It is not.
In `isdf/core.py` it is an *override* that short-circuits the whole route policy:
```python
if override == 'off':
    return 'sharded_cholesky'      # <-- bypasses the replicated route entirely
```
and **only the replicated route carries the rank-truncation cure** for the charge
ζ-solve. The code's own comment records the consequence: the full-BZ 12×12 fit
"silently fell back and returned ζ **4.5× too large**, rebuilding V_q to relF
**16–32** instead of 1.8e-15."

Observed here: a GW that ran to completion (rc=0, every stage "successful") and
produced **nonsense** — QP gap **−161 eV**, QP shifts of −86 eV on semicore states
and −227…+87 eV scatter near the gap, from a DFT structure that was itself
perfect (gap 1.7010 eV). Nothing warns you; the *only* trace is the route name in
the per-q `Computing L_q ... path=...` line.

**Use `distributed_cholesky = auto`** (the default). Below the cap `auto` picks the
**replicated dense** route — plain JAX, no GPU/FFI needed, so it is correct on
CPU — and `charge_zeta_solve` (default `rank_truncate`) is then honoured. Route
selection:
```python
if _replicate_charge_ok(nq, n_rmu):        # nq * n_mu**2 * 16 <= 4 GiB
    return 'replicated_rank_truncate' if charge_zeta_solve=='rank_truncate' else 'replicated_cholesky'
raise/…                                    # above the cap it now REFUSES rather than downgrade
```
For 12×12 at n_μ=276: 144·276²·16 = 0.16 GiB, far under the cap. At the reference's
n_μ=2412 it is 6.44 GiB — **over** the 4 GiB cap, which is exactly why the
production campaign had to raise it (`LORRAX_ZETA_REPLICATE_CAP_GIB`, or
`gw_probe.py --cap-gib 8`). Budget for that when you scale centroids up.

**ALWAYS verify the route in the log before trusting numbers:**
```
grep "Computing L_q" gw.out
# WANT: Computing L_q = rank-truncated pinv  [PSD, charge channel, path=replicated_rank_truncate]
# BAD:  Computing L_q = chol(C_q)            [... path=sharded_cholesky]
```
and sanity-gate the output (gap positive, |Σ| bounded, no astronomical Im Σ) —
see the reference report's `sanity_gate.py`.

**Status: fixed on `frontera-ffi` (commit `0775e35`)** — `gw_config` no longer
rewrites `auto`→`off` on CPU, and `isdf/core` refuses to auto-pick the CUDA-only
cuSOLVERMp on a host mesh, so `auto` is safe there. Confirmed on 4 CPU nodes:
`path=replicated_rank_truncate` on every rank. Regression test:
`tests/test_charge_zeta_route.py`.

**The rank-truncate route costs MORE memory than the (wrong) sharded one — budget
for it.** It replicates the CCT stack on every device *and* needs rank-revealing
eigh workspace on top, where `sharded_cholesky` distributes the factor. Observed:
276c / nband=120 / 4 nodes at `memory_per_device_gb = 90` completed the ζ-fit
fine via `sharded_cholesky`, but **OOMed at r-chunk 5/7** on the correct
`replicated_rank_truncate` route (`isdf/core.py:fit_one_rchunk`, failing on a mere
4.37 GB request — i.e. already at the ceiling). So when you switch to the correct
route, **lower `memory_per_device_gb`** (try ~55–60) rather than raising the node
count; §5b shows nodes make a different allocation worse. Remember the planner's
HWM already under-counts real CPU peak by ~40–50 GB (§5).

**Sizing the replication cap.** The replicated route needs
`nq · n_μ² · 16 B ≤ 4 GiB` (raise with `LORRAX_ZETA_REPLICATE_CAP_GIB`). For
12×12 (nq=144): n_μ=1200 → 3.32 GiB (fits); n_μ=2412 → 13.4 GiB (needs the cap
raised — this is what forced the Perlmutter campaign's `--cap-gib 8`). **This is
much cheaper on CPU than GPU**: the stack is replicated per device, and a
192 GB CPU node swallows 13.4 GiB trivially where an 80 GB A100 does not. So
CPU nodes are actually the *easier* place to run large-n_μ rank-truncated fits.

## 6d. ONE RUN PER DIRECTORY — concurrent runs collide on `tmp/zeta_q.h5`

`tmp_dir` is hardwired to **`<input_file_dir>/tmp`** (no config key), and the ζ
file is a fixed **`zeta_q.h5`** — *not* namespaced by centroid count, unlike its
sibling `isdf_tensors_{n_rmu}.h5`. So two GW jobs launched from the same directory
with different `centroids_file` **share one ζ file**. The writer opens it
`mode='a'` and the clash only surfaces deep in the slab write:
```
ValueError: could not broadcast input array from shape (144,276,8603)
                                        into shape (144,1194,8603)
```
— *after* the entire multi-hour fit is done. Cost me 58 min of 4-node compute.

**Rule: give every run its own directory** containing its own input file (symlink
`WFN.h5`, `dipole.h5`, `kin_ion.h5`, the centroid file and the `.upf`s into it).
Since `tmp_dir` follows the *input file's* directory, that isolates `tmp/`
automatically.

A precondition check now refuses this up front (commit `63f3e2d`) instead of
failing after the fit — but it only detects the clash, it does not make
concurrent same-directory runs safe. Use separate directories.

## 6c. Multi-host host-gather bugs (a whole bug CLASS — expect more)

Multi-node CPU exercises a code path the GPU runs apparently never hit, and it
was broken in several places. Symptom pair, always together:
```
ValueError:   Gathering global non-fully-addressable arrays only supports tiled=True
RuntimeError: Fetching value for `jax.Array` that spans non-addressable
              (non process local) devices is not possible
```
Cause: a helper gathers a globally-sharded array with
`multihost_utils.process_allgather(..., tiled=False)` and falls back to
`jax.device_get()`. On multi-host **both** calls raise, so the run dies.

Rule: for a **global** (non-fully-addressable) `jax.Array`, `tiled=True` is the
only legal mode, and it returns the reconstructed *global* array — which is what
these call sites want. `tiled=False` only makes sense for **process-local** data
(it stacks one value per process). So: force `tiled=True` when
`not a.is_fully_addressable`, otherwise honour the caller. That keeps
single-process/Perlmutter behaviour identical.

Fixed so far on `frontera-ffi`:
- `gw/minimax_screening._scalar_to_host_float` (commit `519f52f`) — killed every
  multi-host run at "GN-PPM + FREQUENCY-INTEGRATED SIGMA".
- `gw/ppm_windows._to_host_np` (commit `9b50768`) — killed it again in
  `compute_sigma_c_ppm_omega_grid` right after the first sigma branch.
- `psp/get_dipole_mtxels` missing `init_jax_distributed()` (commit `edbf363`).

**If you hit this pattern somewhere new, it is the same bug — fix and commit it.**
Grep for `process_allgather` / `device_get` helpers before a long run.

## 7. QE / DFT prep gotchas (for regenerating WFN.h5)

- `pw.x` needed a manual `make pw` (only `pw2bgw.x` was prebuilt). Uses
  `intel/19.1.1 impi/19.0.9`.
- **NSCF** on the full 12×12 grid hit a ScaLAPACK `pzpotrf(110)` error → add
  **`-ndiag 1`** to the `pw.x` command (LAPACK subspace diag, no ScaLAPACK).
- `pw2bgw.x` writes `WFN` into the **outdir** (`out/WFN`), not cwd. Convert with
  `wfn2hdf.x BIN out/WFN WFN.h5`.
- The converged WFN.h5 is 15.65 GB and reproduces the Perlmutter reference (gap
  1.70 eV). Full-spinor (`noncolin`+`lspinorb`), `assume_isolated='2D'`.

## 8. gw_jax input knobs that matter here

Inside the `[cohsex]` section (keys after `K_POINTS` get misparsed as k-point floats):
- `use_ffi_io = false` for CPU. ⚠️ **CORRECTED (2026-07-25): keep
  `distributed_cholesky = auto` and `distributed_lu = auto` — NEVER `off`.**
  An earlier version of this line said `off` for CPU; §6a supersedes it (`off`
  silently destroys the physics via the un-truncated charge-channel solve).
  `auto` is CPU-safe on every path post-branch (resolves to the native
  in-tree routes; the FFI backends are only picked on a CUDA mesh).
- **Multi-host CPU:** set `slab_io = h5py_allgather`. The `zeta_q.h5` writer has 4
  modes (`auto` / `phdf5_ffi` / `phdf5_host` / `h5py_allgather`). `phdf5_*` needs the
  host Intel-MPI + FFI stack; `h5py_allgather` allgathers the sharded ζ (~5.5 GB for
  276c, fits one rank) over the jax.distributed transport and writes with plain h5py
  — **no MPI needed**, the right choice for CPU multi-node. gw_jax auto-builds the
  device mesh from `process_count × local_device_count`, so 1 rank/node × N nodes →
  an N-device mesh that shards the ζ-fit. The per-q Cholesky (`L_q=chol(C_q)`) is
  data-parallel over q, so a **non-square** node count is fine here (unlike the
  cuSOLVERMp path, which needs a square mesh).
- Launch multi-host with `srun -N N --ntasks-per-node=1 -n N` + the §6 fast-init env.
- `memory_per_device_gb = 90` (see §5).
- `nband` should be ≳ n_centroids for a well-posed ISDF fit; for a fast dev run,
  276 centroids + `nband = 120` is a reasonable under-converged pipeline test.
- BSE/exciton bandstructure's replicated `fH_R` needs ~50 GB/device at 12×12 — its
  own memory wall; expect to shard it for the real run.

---

## Timings observed (single node, 276c, nband=120, 12×12)

| phase | with poisoned cache | cache opt-out |
|---|---|---|
| jax import + CPU init | 11.7 s | ~12 s |
| compile → ζ-fit start | ~11 min | ~7 min |
| ζ-fit, 1 node | — | **~5 min / r-chunk × 28 = ~140 min** |
| ζ-fit, 4 nodes, 276c, budget 90 | — | ~8–9 min/chunk × 7 = ~60 min |
| ζ-fit, 8 nodes, 1194c, budget 90 | — | **7 min 55 s/chunk × 15 = ~119 min** |

**The 2-hour `development` wall is NOT enough for a real run.** 1194 centroids on
8 nodes needs ~119 min of ζ-fit *plus* ~20 min of setup — it overruns the dev wall
and you lose everything, because `zeta_q.h5` is only written **after** the whole
fit completes. Do the sizing arithmetic before submitting (chunk cadence is
steady and visible after 2–3 chunks), and put anything near the limit on the
**`normal`** queue with a generous wall. Corollary: once a ζ exists, always
iterate with `restart = true`.

**Key timing result:** the single-node ζ-fit for 276c/nband=120/12×12 is **~140
min** and this is *intrinsic* — the memory budget only trades chunk size for chunk
count (total FLOPs constant), so lowering the budget to fit RAM does NOT speed it
up. **12×12 GW is too slow on one CPU node; it must shard the ζ-fit across nodes.**
That, plus the fact that 3200 centroids exceed one node's RAM anyway, is why the
converged run *requires* multi-node — which makes the ~5-min `jax.distributed`
startup (§6) worth fixing. (Fill in sigma / eqp / multi-node scaling as measured.)

### Multi-node scaling is SUB-LINEAR (ib0 comm-bound)
Same 276c/nband=120/12×12 problem, sharded ζ-fit:

| nodes | mesh | ζ-fit r-chunks | steady-state / chunk | ζ-fit total | speedup |
|---|---|---|---|---|---|
| 1 | 1×1 | 28 | ~5.0 min | ~140 min | 1× |
| 4 | 2×2 | 7  | ~9.3 min | **~65 min** | ~2.2× |

4 nodes gives only ~2.2×, not 4×. The sharded contraction all-reduces over the
**ConnectX-3 / ib0** fabric (PCIe-gen3, no NVLink-equivalent), which is the
bottleneck — so **adding nodes has diminishing returns**. Plus a per-run fixed floor
of ~15-17 min (init 71 s + 15.6 GB WFN load + C_q/L_q Cholesky + 7-min compile) that
does NOT shrink with node count (Amdahl). Net: a 12×12 GW is a **~1.5 hr** job on
4 nodes and won't "blast through" no matter how many nodes you throw at it on this
hardware.

**Implication for dev cycles:** do NOT iterate on 12×12. Use a coarse-grid proxy
(§ cost ladder) — the k-grid (q-point count) is the dominant cost driver, so 6×6
(36 q-pts) is ~4× cheaper *and* fits a GPU. Reserve 12×12 for final production runs.
To shrink the fixed floor, get a working persistent compile cache (open TODO, §4).

### Rough cost ladder (what to reach for)
- **Iterate on code / prove the pipeline: use the MoS₂ 4×4 / 30 Ry deck,
  `$SCRATCH/mos2_4x4_test/`** (built 2026-07-27, workstream **AJ**). Same cell
  and pseudos as the 12×12; `ecutwfc=30`, 16 full-BZ k (10 IBZ, **`ntran=2`**
  — symmetry ON, unlike the `nosym` 12×12), `ngkmax=1964`, FFT (24,24,80),
  `WFN.h5` **153 MiB** (vs 15.65 GB). Full GN-PPM G0W0 at 402 centroids:
  **183 s on ONE node (P=4)**; at 785 centroids on 8 nodes (P=16): **384 s**.
  Regenerating the whole deck from scratch is **~4.5 min** (`REGENERATE.sh`).
  H0 vs `kih.dat` = **7.8e-5 eV** over 16 k × 128 bands, implied-Vxc guard
  silent, eqp0 gap 3.58 eV (DFT 2.21). Centroid sets 108 / 402 / 785 are
  pre-built; `nband=128` is divisible by P=4 **and** P=16. Use this instead of
  hand-rolling a proxy — and do NOT iterate on the 12×12.
- Other proxies if you need something different: coarser k-grid
  (e.g. 6×6 = 36 q-pts, 4× less work *and* fits a 16 GB GPU) or few centroids +
  low nband. Minutes, not hours.
- **Converged 12×12 / 400-band / 3200-centroid science run:** multi-node CPU only.
  Budget hours and many `development` nodes. Shard the ζ-fit; expect the memory
  walls in §5.

---

## 9. Multi-device / multi-node GPU testing (interactive, rtx-dev)

**Goal:** a fast (~2 min) correctness+scaling smoke test that actually exercises
the distributed cuSOLVERMp/phdf5 path, run *interactively* on `idev`/rtx-dev.

### Confirmed queue caps (from `sacctmgr show qos`, not the docs)
| QOS | Max nodes/job | Max wall | Max jobs |
|---|---|---|---|
| `qrtxdevelopment` (rtx-dev) | **2** (= 8 GPU) | 2 h | 2 |
| `qrtx` (rtx, **batch only**) | 24 | 48 h | 20 |
| `idev` default | 1 node, 30 min, `-p development` | — | — |

### The square-mesh constraint decides legal node counts
cuSOLVERMp's `cusolverMpPotrf` needs a **square** block-cyclic mesh (`mb == nb`).
With **4 GPUs/node**, total GPU count must be a perfect square:

| nodes | GPUs | mesh | cuSOLVERMp? |
|---|---|---|---|
| 1 | 4 | 2×2 | ✅ **the blessed path** |
| 2 | 8 | 2×4 | ❌ `mesh 2x4 is not square` → job dies |
| 4 | 16 | 4×4 | ✅ but **needs batch `rtx`** (>2 nodes) |
| 9 | 36 | 6×6 | ✅ batch `rtx` |

So on **interactive rtx-dev the correct-solver ceiling is 1 node / 4 GPU.** Two
nodes can only run cuSOLVERMp by **half-populating** (2 GPU/node → 2×2, 4 GPU) —
useful *only* to exercise the **inter-node fabric**. Lighting up all 8 GPUs on
2 nodes forces the in-tree **`sharded_cholesky`** route (`distributed_cholesky =
off`), which on the charge channel bypasses the rank-truncation cure (§6a) — so
treat it as experimental and **gate its eqp against the reference**.
Genuine 16-GPU (4×4) square scaling is **batch `rtx` only**. And per
`config/frontera/README.md`, multi-node GPU is still officially *Deferred*
(NCCL-over-IB on mlx4 falls back to TCP — benchmark, don't assume).

### The harness (interactive `idev` ladder)
`$SCRATCH/lorrax_setup/multidev_run.sh` (+ `multidev_compare.py`). Runs the
60-centroid COHSEX fixture (~20 s compute) at each rung and gates `eqp_test.dat`
against the committed `tests/regression/cohsex_debug/eqp_ref.dat`
(numpy-allclose: atol 1e-4 eV, rtol 1e-6 — VH ~250 eV needs the relative term):

```
idev -p rtx-dev -A PHY25006 -N 2 -n 8 -m 60      # login node
bash $SCRATCH/lorrax_setup/multidev_run.sh       # on the compute node; all rungs
```
| rung | layout | mesh | solver | exercises |
|---|---|---|---|---|
| 0 | 1 GPU | 1×1 | dense | reference (single device) |
| 1 | 1 node / 4 GPU | 2×2 | cuSOLVERMp | intra-node distributed (blessed) |
| 2 | 2 node / 4 GPU | 2×2 | cuSOLVERMp | inter-node fabric (IB/TCP) |
| 3 | 2 node / 8 GPU | 2×4 | sharded (off-route) | **experimental**, correctness-gated |

Best-practices baked in: everything in-container; run dirs + compile cache on
`$SCRATCH` (never `$HOME` — §2 inode wall); one dir per rung (§6d); `lfs
setstripe -c 8 -S 4m` per rung; inter-node fabric env from §6 / `ffi_env.sh`; the
harmless GRPC teardown noise filtered from the console. `-N 1` auto-skips the
2-node rungs. Baseline timings for reference: 1-node/4-GPU ≈ 2m07s wall / ~21 s
compute; 2-node/4-GPU ≈ 2m00s (`$SCRATCH/lorrax_setup/gwd.*` / `gw2.*` logs).

> **How to run it non-interactively (agent-friendly).** `salloc` is **blocked on
> Frontera** (`salloc job submission is not allowed ... use idev`), and `idev`
> needs a pty. So an agent drives the ladder with **`sbatch` to rtx-dev**
> (`$SCRATCH/lorrax_setup/multidev.sbatch`, `-N 2`) — same fast dev-queue nodes
> `idev` would give, but detached, so it survives the terminal closing (no
> salloc-hold-with-`sleep` trick needed). One gotcha baked into the fix:
> **guard `module load` with `set +u ... set -u`** — Lmod's `module` function
> trips `set -u` on its own unbound vars in a non-login batch shell and kills the
> script *before any output*, which under a `2>/dev/null` looks like a silent
> 0-byte failure.

### ⚠️ RESULT: the multi-GPU cuSOLVERMp Cholesky route is NOT device-invariant
First run of the ladder (job 7872632, 2 nodes, cohsex_debug fixture) — **this is
the §6a trap, on GPU:**

| rung | route (logged) | max\|ΔsigTOT\| vs 1-GPU | states off |
|---|---|---|---|
| 0  1 GPU     | `replicated_rank_truncate` | **0.000 eV** (== `eqp_ref` to 1e-6) | 0/270 |
| 1  4 GPU 1node | `cusolvermp_cholesky` | **3.46 eV** | **270/270** |
| 2  4 GPU 2node | `cusolvermp_cholesky` | 3.46 eV (≡ rung 1 exactly) | 270/270 |
| 3  8 GPU 2×4  | `sharded_cholesky`   | 4.80 eV | 270/270 |

- The **1-GPU dense/replicated path reproduces the reference exactly**; **every
  distributed rung is wrong on every state**, by several eV of QP energy.
- rung 1 ≡ rung 2 (intra- vs inter-node identical) → **the fabric/node count is
  fine.** The `2×4` sharded route (rung 3) diverges from the `2×2` runs too (2.19 eV).

**This is TWO independent bugs — do not conflate them (I did at first):**

**(A) A compile-cache deadlock — FOUND & FIXED.** The first attempt to run the
replicated rank-truncate route on 2×2 (job 7872651) HUNG 12 min and got wall-
killed. A `faulthandler` watchdog (`$SCRATCH/lorrax_setup/hang_probe.py`, arms
`faulthandler.dump_traceback_later`) caught all 4 ranks mid-hang: ranks 0-2 wedged
in `jax/_src/compiler.py _compile_and_write_cache` (compiling `loader.load`,
`wfn_transforms.py:1837`), **rank 3 already executing the *next* pjit** (`_reshard_all`,
`:1887`). Cause: `common/jax_compile_cache.py` gives each rank its **own** cache
dir (`…/np{P}/rank{i}`), so ranks can **diverge on hit-vs-miss** for the same
program — the hitter skips JAX's cross-process compile barrier and the missers
wait on it forever. It is NOT in the ζ-solve and NOT §6c. Fix (confirmed, job
7872738): **disable the persistent cache on multi-process runs — `ISDF_JAX_CACHE_DIR=""`**
(the harness now sets this for every P>1 rung). With it, the 2×2 rank-truncate
run completes in ~29 s. Per-rank private dirs are the trap; a genuinely shared,
consistent cache would also work but is racy on Lustre. `JAX_LOG_COMPILES=1`
makes the per-rank divergence visible.
> **SUPERSEDED 2026-07-27 (workstream AH).** The diagnosis above is right; the
> *fix* is obsolete. A genuinely shared, consistent cache is now implemented in
> `common/jax_compile_cache.py` (process-invariant key + a coordination-service
> agreement on the usable entry set + atomic writes), and the Lustre raciness
> it worried about is exactly what the atomic tmp+rename and the
> snapshot-before-any-compile close. **Do not set `ISDF_JAX_CACHE_DIR=""`
> any more.** See §4 and scorecard AG/AH.

**(B) A device-count-invariance bug — FIXED (`src/isdf/core.py`, patch below).**
Once (A) was fixed, the 4-GPU `replicated_rank_truncate` result is
**byte-identical to the 4-GPU `cusolvermp_cholesky` result** (same sigTOT to all
digits) and **both differ from the 1-GPU reference by the same 3.46 eV on all
270 states.** So the ζ-solve route is irrelevant — my earlier "§6a rank-truncation
cure" attribution was **WRONG**. The multi-GPU (P=4) pipeline is simply not
device-count-invariant: something between P=1 and P=4 shifts every QP energy by
~3.5 eV, tell-tale being a **spurious imaginary V_H on band n=21** (V_H must be
real) at multiple k-points — i.e. a padding / sharding-boundary defect, the exact
class in `reports/device_invariance_2026-07-08/{ROOT_CAUSE,PADDING_AUDIT}.md`.

| route on a 2×2/2×4 GPU mesh | runs (cache off)? | matches P=1 ref? |
|---|---|---|
| `cusolvermp_cholesky` | ✅ ~20 s | ❌ 3.46 eV off |
| `replicated_rank_truncate` | ✅ ~29 s | ❌ 3.46 eV off (≡ cusolvermp) |
| `sharded_cholesky` (2×4) | ✅ ~35 s | ❌ 4.80 eV off |

**Bottom line (updated):** BOTH bugs are now fixed — multi-GPU GW RUNS (bug A:
compile-cache off at P>1) and is CORRECT (bug B: the Z_q band-gather patch below).
P=4 rank-truncate and cuSOLVERMp both match the P=1 reference to 1e-6 eV on the
default `band_chunk_size=16`. The table below is the pre-fix diagnostic record.

**Bug B bisected (h5diff of the on-disk intermediates, P=1 `rung0` vs P=4
`rung1c`, SAME `replicated_rank_truncate` route, identical inputs):**

| intermediate (`tmp/isdf_tensors_60.h5`, `zeta_q.h5`) | P=1 vs P=4 |
|---|---|
| `psi_full_y` (centroid wavefunctions, ζ-fit **input**) | **identical** |
| `enk_full` (DFT energies) | **identical** |
| `vhead` / `whead` (2-D Coulomb head) | **identical** |
| `G0_mu_nu` (G=0 term) | identical to **roundoff** (rel ~1e-6) |
| **`zeta_q_G` (ζ tensor, ζ-fit output)** | **~all differ, >10% rel** |
| `V_qmunu`, `W0_qmunu` (built from ζ) | ~all differ, >10% rel |

So the divergence is **born in the ζ-fit itself**: identical `psi` in, grossly
different ζ out (O(10–100%), not roundoff), and the solver route is irrelevant
(rank-truncate ≡ cuSOLVERMp bit-for-bit at P=4). **The bug is in the sharded
`Z_q`/`C_q` construction** (the ISDF density-fit contractions over the mesh),
NOT the load (device-invariant), NOT the ζ-solve, NOT the head/G0 terms. The
imaginary-V_H-on-n=21 in the eqp file is a *downstream symptom* of the wrong ζ,
not the seed.

**ROOT CAUSE (found & FIXED) — the band `all_gather` stride vs a contiguous
mask, in `z_q_from_psi_sm` (`core.py`, the `shard_map(check_rep=False)` body).**
It is **Z_q**, not C_q (C_q is built from *band-replicated* inputs → P-invariant
by construction; confirmed: `C_q_fro` computed clean at P=1). The band store slots
each bc into a static `bpd_max`-wide block per rank (`_slice_local_tile_bc`,
`psi_G_store.py`), real bands in the first `bpd_per_bc = (b_hi-b_lo)//P` slots and
zero pad after. After `all_gather(axis=1, tiled=True)` the gathered band axis is
**`bpd_max`-strided** (pad slots interleaved between ranks), but the `g_axis` mask
and the contiguous `psi_*_X` slice both index it as a **contiguous** global-band
axis. When `bpd_per_bc < bpd_max` — i.e. the **short remainder band chunk**
(`nband=40`, `band_chunk_size=16` → widths 16,16,**8**) — the X and Y band axes
misalign: real bands get dropped, pad zeros kept, and the einsum pairs mismatched
bands → wrong pair density → wrong ζ. At P=1 there is one block, strided ≡
contiguous, so P=1 is always correct — the exact P=1-good/P>1-bad signature.

**Two independent confirmations (both dispositive):**
1. **No-code test:** `band_chunk_size=40` (or 20) → uniform chunks, no remainder,
   `bpd_per_bc==bpd_max` → P=4 matches ref to **1e-6 eV**. Isolates the remainder
   chunk as the trigger.
2. **The fix:** a static per-bc Y-compaction gather table (`_y_compact_idx_np`,
   `jnp.take` right after the `all_gather`) reorders the strided real bands to the
   contiguous front positions the mask/X-slice assume. Identity permutation when
   `bpd_per_bc==bpd_max` → **byte-identical at P=1 and for every full chunk**;
   only the remainder chunk at P>1 changes. After the patch, with the ORIGINAL
   `band_chunk_size=16`: P=1, P=4 `replicated_rank_truncate`, AND P=4
   `cusolvermp_cholesky` all match `eqp_ref.dat` to **1e-6 eV** (were 3.46 eV off),
   and the eqp file is back to 1888 floats (the spurious imaginary V_H on n=21 is
   gone). The fix is upstream of the solver, so it repairs cuSOLVERMp/sharded too.

Sibling sites audited clean (`c_q_from_psi_sm` band-replicated; `gflat_to_rmu`
writes back each rank's own band block — no cross-rank band gather).

### Changes landed (branch `frontera-ffi`, uncommitted)
1. **The fix** — `src/isdf/core.py` `z_q_from_psi_sm`: static per-bc Y-compaction
   table + `jnp.take` after the band `all_gather` (~L490, L682). No-op at P=1.
2. **Structural asserts** (`core.py`): build-time band-chunk-width `world_size`-
   divisibility guard (in the compaction loop) + a trace-time gathered-shape
   check after the `all_gather`. Make the machine-invisible preconditions loud.
3. **`round_up` consolidation** — `src/common/wfn_transforms.py:845`: the one
   hand-rolled band ceil-div (`-(-n//p)*p`) → the canonical `runtime.padding.round_up`.
4. **`nb_sigma` loud-guard** — `src/gw/ppm_tau_kernel.py`
   `_make_project_ri_reduce_scatter`: asserts the sigma band window is mesh-
   divisible with an actionable message (the `psum_scatter` would otherwise
   crash cryptically). Loud-but-clear; no auto-padding.
5. **Regression gate** — `tests/test_zeta_mesh_invariance.py::test_zq_band_gather_is_mesh_invariant`:
   runs the REAL `z_q_from_psi_sm` across CPU host-device meshes **P=1,2,3,4,6
   (incl. non-square 2×3)** with a remainder chunk; asserts Z_q mesh-invariant
   to 1e-9. **Proven to bite**: FAILS pre-fix (0.833 frob-rel), PASSES post-fix.
   GPU-free, default suite. (The existing `test_zeta_mesh_invariance` only tested
   `factor_c_q`/`solve` on a directly-built C_q — never the band-gather; that was
   the coverage hole.)

6. **V_q reshard perf fix** (not correctness) — `src/gw/v_q_g_flat.py`
   `_make_per_q_kernel`: the per-q ζ (`[1,μ,G]`) was resharded by staging through
   `P(('x','y'),None)` — sharding the **size-1 q axis** over all 4 devices, a
   degenerate `[4,1,1]` layout XLA can't reshard to μ-sharded, so it did a full
   replicate-then-repartition (`[SPMD] Involuntary full rematerialization` in the
   profile). Fixed by dropping the q axis FIRST (`zeta_L_q[0]`) then constraining
   the real `(μ,G)` tensor to μ-on-`x`/`y` — reshard stays on a real axis (clean
   all-to-all/all-gather). Warning gone, eqp unchanged to 1e-6. Found via a
   roofline probe: the build extracts **102% of RTX 5000 FP64 peak / 83% HBM BW**
   (`$SCRATCH/lorrax_setup/roofline.py`), so the build is optimal — Frontera's GW
   slowness is the RTX 5000 (c128 ~0.36 TFLOP/s, ~27× below A100; 16 GB OOMs 12×12),
   NOT a mis-built dependency. This remat was the one code-side lever.

**Why this class is invisible to JAX (and the residual gap):** pad-to-uniform
(`bpd_max` slots) keeps every shape rectangular and mesh-divisible, so the
misalignment lives *inside* shape-valid arrays — no ragged-shape/`check_rep`
error can fire. Only numerical P-invariance testing (item 5) catches it. Sibling
seams (`gflat_to_rmu`, the μ pad-zero preconditions, sigma band windows) are
correct **by reasoning** but are NOT yet numerically gated — the honest open
work is extending item-5-style gates to them.

Tools used: `$SCRATCH/lorrax_setup/hang_probe.py` (faulthandler stacks);
`h5diff -p <rel>` (login node: `module load phdf5`); `multidev_compare.py` (eqp gate).

---

## 10. CPU multi-node performance — MEASURED (2026-07-25), supersedes guesses

**The headline: on Frontera, CPU nodes BEAT the RTX 5000s for LORRAX.** Measured
roofline (`$SCRATCH/lorrax_setup/roofline.py`): the JAX build extracts **102% of
the RTX 5000 FP64 peak and 83% of HBM BW** — the build is optimal, nothing is
mis-linked. But the RTX 5000 is Turing: **complex128 ≈ 0.36 TFLOP/s** (FP64 is
1:32), ~27× below an A100, on top of 16 GB (OOMs 12×12) and no NVLink. So the GW
FP64 workload is hardware-starved on these GPUs. On the cohsex fixture a **56-core
CPU node (6.8 s) beat 4× RTX 5000 (20.5 s) by ~3×** — and a CPU node has 192 GB
(12× the GPU). **Target CPU nodes, not the rtx GPUs, for real LORRAX runs here.**
(Perlmutter A100s are the opposite — fast FP64, IO-bound; that experience does
NOT transfer to Frontera's GPUs.)

**Rank/thread layout — 1 rank/node × 56 threads is NOT best (measured).** 56
threads span both NUMA sockets; the 2nd socket added only ~35% (memory-bound).
Single-node cohsex sweep (ranks×threads=56): 1×56 = 7.85 s, **2×28 = 7.30 s,
4×14 = 7.15 s**, 8×7 = 8.41 s (over-shard). **Default: 2 ranks/node × 28 threads,
one pinned per NUMA socket** (`--ntasks-per-node=2`, `taskset -c 0-27 / 28-55`,
`OMP_NUM_THREADS=28`). NUMA-clean, near-optimal, and keeps `world_size` moderate
for multi-node comm. The signal is modest on the tiny fixture (mostly fixed
overhead) — expect a larger rank win on a real ζ-fit. Do NOT go past ~4 ranks/node.

**Node count: match the problem's MEMORY, don't max out.** §5b's OOM (the eager
WFN load built the full host array per rank) is FIXED (process-local load, commit
in the `frontera-ffi` PR) — 40+ nodes now clear the centroid load. BUT more nodes
≠ faster for a FIXED small problem: at 40 nodes the 276c ζ-fit is badly
**over-sharded** (40 devices for 276 μ / 120 bands → each rank ~idle, comm-bound;
one r-chunk ran >15 min, node load 0.3). The payoff of high node count is FITTING
BIGGER problems (1194c+, converged systems) in the aggregate RAM — not speeding up
small ones. Practical: pick nodes so per-device work stays substantial.

**Thread cap: use `taskset`/`numactl`, NOT XLA_FLAGS (CORRECTED 2026-07-25).**
XLA:CPU sizes its Eigen threadpool to the VISIBLE core count; it does not obey
`OMP_NUM_THREADS` for the fusion pool (OMP only reaches BLAS/LAPACK). An earlier
version of this note said to set
`XLA_FLAGS="--intra_op_parallelism_threads=N"` — **that flag does not exist in
our jaxlib and F-ABORTS the process at startup**
(`parse_flags_from_env.cc: Unknown flag in XLA_FLAGS`), which killed the first
MoS2 12×12 production attempt (job 7874158) in 2 min. The correct and measured
mechanism is CPU pinning per rank: `taskset -c 0-27` / `28-55` (or `numactl
--cpunodebind --membind`) — XLA then sees 28 cores and sizes its pool to 28.
All §10 rank/thread numbers were taken this way. CPU has NO preallocation knob
(`XLA_PYTHON_CLIENT_*` are GPU-only; XLA:CPU uses glibc malloc on demand) — our
lack of CPU memory config is correct; `memory_per_device_gb` is the working-set
bound.

**GPU allocator (low-stakes here, but wrong in one script).** `lorrax-frontera.sh`
sets `XLA_PYTHON_CLIENT_ALLOCATOR=platform` — the jax docs call it *"very slow"* —
plus `TF_GPU_ALLOCATOR=cuda_malloc_async`, which JAX **ignores** (TF-only var). So
that script effectively runs the slowest allocator. Use `cuda_async` (or unset for
default 75% BFC preallocation if the GPU is owned); `ffi_env.sh` already uses
`cuda_async` correctly. Sources: docs.jax.dev/en/latest/gpu_memory_allocation.html,
jax#8670.

**Fast dev loop for an agent (TACC-specific).** `idev` (needs a pty) and `salloc`
(blocked) are dead ends for a headless agent, and login-node `srun --jobid` trips
the TACC `-p` wrapper. What WORKS: TACC blesses **`ssh <compute-node-you-own> '<cmd>'`**
(non-interactive, no pty) — `sbatch --wrap='sleep N' -N1`, then
`ssh $(scontrol show hostnames $(squeue -u $USER -h -n warm -o %N)|head -1) '<cmd>'`
per test → no re-queue. For multi-node/warm-python, an in-allocation FIFO
step-server (batch job runs each dropped command via in-allocation `srun`, which
IS wrapper-legal). Never let a big job hog the 40-node/2-job dev QOS while small
tests wait behind it (`QOSMaxNodePerUserLimit`).

## 10b. Your collectives were on the WRONG NIC — em1 (1 GbE) vs ib0 (InfiniBand), FIXED 2026-07-27 (AK/AL)

**The finding (scorecard AK.4/AK.10/AL).** JAX's CPU collectives are Gloo TCP,
and jax 0.9.1 constructs them with no `interface=` argument — Gloo then binds
whatever NIC routes to the coordinator's hostname, which on Frontera compute
nodes is the **1 GbE management NIC `em1`** (129.114.x.x, MTU 1500), not
InfiniBand `ib0` (192.168.x.x). `GLOO_SOCKET_IFNAME` appears nowhere in the
shipped jax/jaxlib — every sbatch that exported it was exporting a no-op
(§6 above was corrected accordingly). Proof: every Gloo peer address in every
failure message of the campaign is 129.114.x.x, and Σ at 606c/P=80 achieved
55 MB/s/node — half a 1 Gb link, i.e. *saturation*.

**The fix is in the tree** (branch `al-gloo-ib-pin`, workstream AL, building
on AK's PoC): `runtime.pin_gloo_interface()`, called from `bootstrap()`,
re-registers the CPU backend factory so `make_gloo_tcp_collectives` receives
`interface=ib0` (auto-detected: first UP `ib*`/`hsn*` NIC with an IPv4; the
`hsn*` pattern covers Perlmutter). Override or disable with
`LORRAX_GLOO_IFNAME=<name|off>`. The decision is announced on rank 0:
`[runtime] Gloo collectives pinned to ib0 (...)`. Every failure path
(no fabric NIC, jax internals moved, bad name) degrades LOUDLY to stock —
it never crashes and never hangs. No job-script change is needed; nothing
to set.

**Measured, MoS2 12×12 / 606c / P=80 (40 nodes, mesh 8×10) — job 7876541
(ib0, HEAD+pin, full pipeline) vs job 7874609 (July-25, em1):**

| stage | em1 (Jul-25) | ib0 (AL) | ratio |
|---|---|---|---|
| ζ-fit (`zeta_fit_chunked`) | 2023.4 s | **211.4 s** | **9.6×** |
| — chunk loop (9 r-chunks) | 1847.2 s (205 s/chunk) | 147.5 s (16.6 s/chunk) | **12.5×** |
| — back-solve | 630.0 s | 36.5 s | **17.3×** |
| `V_q_compute` | 66.3 s | 19.2 s | 3.5× |
| Σ (`gw_jax.sigma`) | 341.6 s | 73.8 s | **4.6×** |
| — `sigma.exec` | 311.8 s | 62.0 s | **5.0×** |
| Total recorded / wall | 2744.8 / 3193 s | **419.8 / 544 s** | **6.5× / 5.9×** |

with **every physics column bit-identical** (sigX/sigC/sigXC/Eo max|Δ| =
0.00e+00 over all 10 080 rows vs the July-25 artifact; VH differs by the
documented `hartree_source` seam only). The Σ τ loop went from 68 to
~365 MFLOP/s/core — from 15% to ~80% of the measured single-node throughput
(457), so the em1 flatline (AK.3's "8 nodes and 40 nodes land on the same
1.9 GFLOP/s") is gone. The July-25→AL comparison bundles transport with the
merged HEAD improvements; the transport-only A/B at P=80 is the restart
razor (scorecard AL), same conclusion.

**It also closed a hard blocker:** the end-to-end 606c/P=80 run could not
complete on em1 at all — two independent jobs (7876516/7876527) died in ζ
r-chunk 1 with eight 8-device Gloo subgroup contexts timing out at exactly
30 s simultaneously. On ib0 the same deck + env sails through that chunk in
~17 s. The "Gloo rendezvous storm at scale" failure class was an em1
saturation artifact.

**⚠ CLAIM-DECAY (pattern #9): every Gloo/collective wall time in this file
measured before 2026-07-27 is an em1 number.** That includes §10's
"over-sharded at 40 nodes / one r-chunk >15 min" observation, the ζ/Σ
scaling walls, the P=144 Gloo collapse, and any "the collective is too big"
conclusion. Byte counts and HLO inventories stay valid; wall-time
consequences re-price ~5-17× where a stage was collective-bound. Re-verify
before planning against any of them.

**What ib0 does NOT fix (scorecard AL, priced honestly):** ζ speeds up
*more* than Σ (back-solve 17× vs Σ 5×), so the owner's GW ≤ 0.5×ζ flop
invariant is NOT restored by transport — at 606c/P=80 the true ib0-native
ratio is GW/ζ ≈ 0.53 (it was 0.225 on em1 only because ζ was even more
transport-crippled than Σ). The Σ communication floor that remains is
structural (AK.3/AK.9: 4 psum_scatters per τ node, `_to_host_np` gather)
and is the next target, now measurable on a real fabric.

## 10c. Intel-MPI fabric: DELETE the `FI_PROVIDER=tcp` pin — and never bind /dev into the container (AP root cause, AS in-container certification, 2026-07-27)

**The finding (scorecard AP, certified in-container by AS).** Every MPI
consumer in the production env (ScaLAPACK pzheevd/LU FFI, PHDF5 MPI-IO,
mpi4py) was running on `FI_PROVIDER=tcp` over IPoIB — an rtx/mlx4 bring-up
workaround carried to CLX by mistake (`config/frontera/ffi_env.sh`). With
`FI_PROVIDER` **UNSET**, Intel MPI 2020.4 auto-selects the native `mlx`
(UCX) provider: **1.07 µs / 11.4 GB/s** pingpong vs tcp's 10.9 µs /
2.15 GB/s, 8 B Allreduce@32 ranks 3.45 µs vs 146 µs (42×), and pzheevd
n=2448 at the production 12×12/P=144 shape **12 s/q → 0.5–0.9 s/q**.
`fi_info` claims −61 for `mlx` even where it works — trust ONLY the
`I_MPI_DEBUG=4` rank-0 banner `libfabric provider: mlx`.

**In-container status: WORKS — the "blocker" was self-inflicted.** TACC
apptainer (1.4.1, `mount dev = yes`) exposes the FULL host `/dev`
(uverbs0/1 included, world-writable) by default. AP's probes that showed
missing uverbs were (a) an `ls | head -8` truncation and (b) explicit
`--bind /dev/...` flags: **any user bind under /dev is mounted
`nosuid,nodev` and cannot open device nodes** (even `/dev/null` breaks
through it). Rules:
* NEVER add `--bind /dev` or `--bind /dev/infiniband` — the default mount
  is already correct and device-openable.
* The container (Debian 12) lacks rdma-core/UCX userspace: stage host libs
  via `--bind /usr/lib64:/hostlibs:ro,/usr/lib64/libibverbs,/etc/libibverbs.d`
  and symlink-stage `libibverbs/librdmacm/libnl*/libucp/libucs/libuct/
  libucm/libnuma` + the `ucx/` module dir into a private dir APPENDED to
  `LD_LIBRARY_PATH` (wk_AS/as_inner.sh; a bare `/hostlibs` on the path
  shadows container glibc and kills every binary).
* Measured in-container with that recipe (AS.2): `mlx` selected, 1.07 µs /
  11.42 GB/s, pzheevd P=144 n=2448 best-rep 0.52 s/q — identical to host.

**The harness dial** (runAC.sbatch family + gw800_merged.sbatch, landed):
`LORRAX_MPI_PROVIDER=auto` (default) unsets `FI_PROVIDER` ⇒ mlx; `tcp`
restores the old pin (rtx escape hatch); any other value force-requests
that provider. `I_MPI_DEBUG` defaults to 4 so the provider is ANNOUNCED.
⚠ never request `verbs` at P≥144 with the one-block eigh layout (68 s/q
pathology, AP.4); mlx has a non-production anomaly at 4 ranks/node ×
n=5024 only (AP.4.3).

**JAX CPU collectives can ride Intel MPI too (AP.5b bring-up, AS gates).**
The jaxlib wheel is MPItrampoline-built; `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`
+ `MPITRAMPOLINE_LIB=/scratch2/08271/jackmc/lorrax_setup/wk_AS/mpiw_thr_install/lib64/libmpiwrapper.so`
runs the full 4x4-deck pipeline with eqp0/eqp1/eqp_g0w0/sigma_diag EQUAL
to the AJ baselines at P=4 and P=16, 1.18x faster e2e than gloo/ib0 at
P=16 (ζ-solve 1.4x, V_q 3.0x, ζ write_g_flat 8.2x, Σ 1.2x).  THREE
mandatory pieces, each measured (scorecard AS.3/AS.4b):
1. `mpiw_thr_install` — MPIwrapper built `--as-needed` (no libgfortran,
   no staging) AND patched to upgrade every MPI init to
   **MPI_THREAD_MULTIPLE**.  XLA requests only FUNNELED; with mpi4py/
   h5py collective I/O on the main thread that is UB — measured as an
   intermittent (~30%, provider-INDEPENDENT) segfault/hang in the MPI
   progress engine at P=16 x 8 nodes.  The plain `mpiw_install` shim
   MUST NOT be used for production impl=mpi runs.  (An LD_PRELOAD
   interposer does NOT work — the trampoline's dlopen scope bypasses
   it; and importing mpi4py first hangs the trampoline path.  Patch the
   wrapper, nothing else.)
2. `LORRAX_MPI_FINALIZE_FIX=skip_atexit` with `wk_AS/sitedir` (or the
   overlay copy) on PYTHONPATH — the SUCCESS-then-rc=1 teardown wart is
   a double MPI_Finalize (jax's atexit + the MpiCollectives destructor);
   skip_atexit leaves exactly one.  rc=0 measured; gloo runs unaffected.
3. Provider auto (the §10c block) — impl=mpi rides mlx at 1.07 us.
Gloo/ib0 remains the certified DEFAULT; impl=mpi is the measured
upgrade path once AQ-scale reps confirm (wk_AS scorecard, rep counts in
as12/as13.out).

## 10d. Transport-env audit: what each variable actually does — MEASURED (wk_AU, 2026-07-27)

Per-variable A/B, in-container, provider auto, 2-node IMB pingpong +
32-rank/16-node Allreduce + pzheevd P=144 production shape (cells in
`$SCRATCH/lorrax_setup/wk_AU/logs`). Baseline reproduced AS.2 exactly
(mlx banner, 1.07 µs / 11.38 GB/s).

| variable | measured | verdict |
|---|---|---|
| `I_MPI_FABRICS=shm:ofi` | unset ⇒ identical (1.08 µs / 11.37 GB/s; it IS Intel MPI 2019+'s default) | keep — documentation of intent + guard against stray inherited values, zero cost |
| `FI_PROVIDER_PATH=$IMPI/libfabric/lib/prov` | unset ⇒ **PMPI_Init FATAL** `addrinfo() ... No data available` (libfabric finds no providers; mpivars.sh is not sourced in-container) | **REQUIRED** — and note that error string is the rtx-era archaeology's string: some of that history was likely this, not the fabric |
| `I_MPI_DEBUG=4` | =0 ⇒ identical latency/bandwidth (banner is init-time only) | keep 4 — the provider banner is the ONLY trustworthy provider observable (mandatory telemetry) |
| inherited `UCX_TLS=knem,dc_x,rc` + `UCX_*MLX5*` timeout/retry (TACC impi module) | strip ⇒ 8 B rows unchanged (1.07 µs pp / 3.38 µs allreduce) but **1 MiB Allreduce@32 419 → 799 µs (1.9×)** | SETDEFAULT the six module values in the MPI harness blocks (inherited wins; stripped launch envs no longer silently lose 2×). Never hard-pin (rtx/mlx4 has no dc_x); add no other UCX knobs |
| `FI_PROVIDER` (login env exports `mlx`!) | the default impi module puts `FI_PROVIDER=mlx` in EVERY shell; sbatch inherits | the `auto` branch's `unset` is load-bearing — a forced mlx turns the missing-UCX-userspace fallback into a fatal init instead of an announced tcp degrade |
| `I_MPI_PMI_LIBRARY` (login env: `/usr/lib64/libpmi.so` = PMI-**1**) | wrong protocol for `--mpi=pmi2` and absent in-container | override unconditionally to the staged `$WORK/.../host_pmi/libpmi2.so.0` wherever `srun --mpi=pmi2` launches MPI code; unused (and harmless) elsewhere |
| `GLOO_SOCKET_IFNAME` / `NCCL_SOCKET_IFNAME` | inert with jax / GPU-only | REMOVED from every live harness (runAC, gw800*, gw400, gate40, cpumn_a, alloc_run); the real dial is `LORRAX_GLOO_IFNAME` (§10b) |

Also landed by wk_AU: runAC.sbatch + gw800_merged.sbatch now carry the AS.1
RDMA/UCX staging binds (`/usr/lib64:/hostlibs:ro,/usr/lib64/libibverbs,
/etc/libibverbs.d`) + the symlink-staging block — WITHOUT which the landed
`LORRAX_MPI_PROVIDER=auto` case-block was necessary-but-not-sufficient and
production auto still degraded to tcp (announced but slow). And
`config/frontera/ffi_env.sh`'s `FI_PROVIDER=tcp` seed (the original AP root
cause) is replaced by the same case-block on branch `env-audit-transport`.

## 11. Parallel-agent workflow + compile-storm anatomy — MEASURED (2026-07-25)

**Branch handoff:** the full write-up of what changed on
`fix/zq-band-gather-device-invariance` (CPU adaptation, phdf5 host FFI, exciton
sharding, linalg/bootstrap unification) lives IN the repo at
`docs/dev/HANDOFF_cpu_frontera_2026-07.md`. This section is the operational
counterpart — the harness and the perf facts that outlive the branch.

**The shared-allocation test harness that WORKS (supersedes §10's sketch).**
Dev QOS = **2 jobs / 40 nodes total** per user, so a 40-node holder eats the
whole budget — that's fine, it becomes a shared test bed; `qnormal` = **75
jobs / 1280 nodes** is the overflow for longer runs. Pattern that worked for N
concurrent agents:
- `sbatch -N40 --ntasks-per-node=2 -t 2:00:00` a holder that writes its
  nodelist to a file and `sleep`s. Keep the job-id in a file so a runner can
  be renewed without editing callers.
- Runner `alloc_run.sh <N> <tasks/node> <PYTHONPATH_src> <workdir> python -m mod|-u script`:
  `ssh <holder-head-node>` then `srun --overlap --jobid=<J> -N.. -n.. apptainer exec …`.
  Two non-obvious musts: **`--overlap`** (else concurrent steps mutually cancel),
  and **run `srun --jobid` from INSIDE a holder node via ssh** (login-node
  `srun --jobid` trips the `-p` wrapper; a compute node's `srun` is real).
  Pass a **script file or `-m module`, never inline `python -c`** (arg-splitting
  mangles it). Put the venv on PATH inside the container (`.venv/bin`) and set
  the standard CPU env (`JAX_PLATFORMS=cpu`, x64, `GLOO/NCCL_SOCKET_IFNAME=ib0`,
  `HDF5_USE_FILE_LOCKING=FALSE`, coordinator). *(The `ISDF_JAX_CACHE_DIR=""`
  that used to be in this list is obsolete since workstream AH — see §4. The
  existing `alloc_run.sh` still sets it, which is harmless, just no cache.)*
- Contention is real: with several agents sharing 40 nodes, `jax.distributed`
  steps intermittently die `DEADLINE_EXCEEDED` (a lagging rank at the SVD trips
  the coordinator) — retry, it's infra not code. Space steps out / distinct
  workdirs.
- **Parallel EDITS:** give each agent its own **git worktree** under `/work2`
  (`git worktree add -b wt-X /work2/.../wt-X <base-commit>`); the Agent-tool
  `isolation:"worktree"` fails if the session CWD isn't a git repo. Each agent
  tests with `PYTHONPATH=<its worktree>/src`. Merge the branches at the end
  (disjoint files merge clean; note D+E both touched `htransform.py` in
  non-adjacent regions → auto-merged).

**The htransform "2208-compile storm" is RANK REPLICATION, not shape variation
(measured).** Per-k `ngk` is already uniform (loader pads to `ngkmax`, sentinel
`box_index`, zero G-slot); every heavy kernel compiles **once per rank**.
Per-rank compile count is **problem-size-invariant** (~138/rank tiny fixture ≈
138/rank on the big system), so `2208 ≈ 16 ranks × 138`. Consequences:
- The only real shape residual was the **band-chunk remainder** (a non-uniform
  last chunk recompiled `to_rchunk` + `_accum`). FIXED by band-axis uniform
  zero-pad (`band_pad_to`, commit `bc58cc1`): **−23%** Galerkin wall on
  non-uniform chunks, bit-exact. Also fixed a latent `UH_bc`/ψ band-dim bug.
- **The real lever for the 158 s is a SHARED persistent compile cache** so
  ranks 1–15 hit rank-0's compiled modules instead of each recompiling ~138.
  **DONE 2026-07-27 (workstream AH)** — the shared cache is implemented and
  safe at P>1; warm runs compile 0 modules per rank at the fixture scale
  (152 → 0 for htransform at P=8). Just leave `ISDF_JAX_CACHE_DIR` unset.
  Note what the cache does NOT buy: it removes XLA *compilation*, not jaxpr
  tracing/lowering, so the remaining fixed floor is tracing + dispatch.

**CPU WFN read now has two layers (both bit-exact vs eager).** `phdf5_host`
(zero-build host `h5py` union read → the same on-device unfold kernel as the
GPU FFI) is the default CPU path today; the phdf5 **FFI** was refactored to a
shared core that also builds a **CUDA-free host lib** (`config/frontera/build_ffi_host.sh`)
— rebuild + point `LORRAX_FFI_SO` at it to get the collective MPI-IO read on
CPU, else it falls back to the twin (zero regression). The htransform loader is
now mesh-aware, so ψ is band-sharded per rank instead of replicated. Details in
the handoff doc.
