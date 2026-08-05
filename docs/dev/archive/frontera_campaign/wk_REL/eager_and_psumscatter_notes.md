# Eager-execution sweep + psum_scatter archaeology + BSE/MPI thread affinity

2026-07-29.  Owner-directed, three tasks.  Repo `/work2/08271/jackmc/frontera/lorrax`
at `b4c7bca` (branch `fix/zq-band-gather-device-invariance`); the two live
branches `wsREL-isdf-window` (`ec96ba9`) and `wt-REL10k-scale` (`d58bad5`) were
swept as separate refs, differences called out where they matter.

Nothing is committed.  All source edits are in the WORKING TREE only.

---

# TASK 1 — EAGER-EXECUTION SWEEP

## 1.0 Method

Three passes, each over `git grep <ref>` for all three refs so a fix that exists
on only one branch is visible as such:

* **static AST sweep** (`eager_scan.py`) — every `jax.devices()`/`local_devices()`,
  every `device_put`, every `device_get`/`process_allgather`, every heavy op
  (`fft*`, `einsum`, `matmul`, `@`, `tensordot`, `linalg.*`).
* **call-graph-aware re-scan** (`eager_scan2.py`) — a function is TRACED if it is
  jit/shard_map-decorated, named as an argument to a wrapper
  (`jax.jit(f)`, `shard_map(f, …)`, `lax.scan(f, …)`), or transitively called
  from a TRACED function.  Heavy ops in TRACED code are fine; the 737 raw heavy
  sites reduce to 373 genuinely-eager ones, and those were triaged by hand.
* **`device_put` branch analysis against the ACTUAL jaxlib**, not from memory.
  `/work2/08271/jackmc/frontera/lorrax_env/.venv/lib/python3.12/site-packages/jax/_src/dispatch.py:483-497`
  is the authority and it says the `multihost_utils.assert_equal`
  (= `process_allgather`, i.e. a REAL P-linear collective) branch is taken iff

  ```
  not sharding.is_fully_addressable
    AND ( (x is a jax.Array and NOT x._committed) or x is numpy / a python scalar )
  ```

  So the antipattern is precisely **numpy OR an uncommitted `jax.Array`** onto a
  multi-process sharding.  A COMMITTED global array takes the reshard branch at
  `:447` and never asserts.  This is what separates the real hits from the 30-odd
  `device_put(<jit output>, rep)` calls that look like hits and are not.
  (jax/jaxlib 0.9.1.)

Scanners: `/tmp/.../scratchpad/eager_scan{,2}.py`; raw hit JSON alongside.

## 1.1 CLASS (a) — `jax.devices()` where a process-local device/mesh is required

This is the exact 2cbd824 defect.  `jax.devices()` is the GLOBAL list, so
`jax.devices()[0]` is **process 0's device on every rank**.

### (a-1) THE 2cbd824 SITES ARE STILL UNFIXED ON TWO OF THE THREE BRANCHES — highest-priority finding of the sweep

`2cbd824` landed on **`wt-REL10k-scale` ONLY**.  On `b4c7bca`
(`fix/zq-band-gather-device-invariance`) and on **`wsREL-isdf-window`** the
original bodies are still present:

| file:line (HEAD / wsREL-isdf-window) | state on wt-REL10k-scale |
|---|---|
| `src/centroid/charge_density.py:108` | fixed |
| `src/centroid/charge_density.py:245` | fixed |
| `src/centroid/pivoted_cholesky.py:119` | fixed |
| `src/centroid/kmeans_cli.py:196` (single-device fallbacks) | fixed |

`wsREL-isdf-window` is an ISDF/centroid branch — `d79141e` ("centroid selection
must use the deck's ACTUAL sigma window (RELEASE BLOCKER)") edits the very tree
that still SIGSEGVs at every P>1.  **Action: merge/cherry-pick `2cbd824` into
`wsREL-isdf-window` before anything on it is run distributed.**  Not fixed here
— duplicating the commit by hand would create a conflict against the real one.

### (a-2) `.memory_stats()` on the global list — FIXED (3 sites)

| site | class | path | fix applied |
|---|---|---|---|
| `src/common/gpu_utils.py:42` | (a) | **budgeting** — `_get_jax_gpu_memory_bytes` feeds memory budgets/chunk sizing | `jax.devices()` → `jax.local_devices()` |
| `src/gw/gw_output.py:143` | (a) | banner | same |
| `src/gw/isdf_fitting.py:62` | (a) | `LORRAX_MEM_DEBUG` sampler; `only_rank0` is a DEFAULT, callers pass `False` for per-rank samples | same |

`gpu_utils.py` is the one that matters: every rank but 0 was sizing its chunks
against **another process's allocator**, and the whole function is wrapped in a
bare `except Exception: return None, None, None`, so a raise on a non-addressable
device degraded silently to "no GPU info" rather than announcing.

### (a-3) prefix-sliced meshes — NOT fixed, ranked

`bse_feast.py:775`, `bse_pseudopoles.py:206`, `bse_w_exact.py:81`,
`bse_ring_comm.py:1020,1078` all do

```python
mesh_devices = np.array(jax.devices()[: px * py]).reshape(px, py)
```

with a guard for `px*py > n_devices` and **none for `px*py < n_devices`**.  A
user passing `--px 2 --py 2` on a 16-process launch gets a mesh over the FIRST 4
global devices; ranks 4..15 are not in the mesh, enter the same jit, and hang or
fault.  Same root idiom as 2cbd824 (indexing the global list), one level up.
Fix is a refusal, not an idiom swap, so it is listed rather than applied.
Blast radius: BSE/exciton CLI entry points only (`--px/--py` are user args).

### (a-4) dead + wrong

`src/common/meta.py:78`
```python
rank_topo = np.where(np.asarray(jax.devices()) == rank)   # rank is an int
```
Comparing `Device` objects to an int is elementwise-False, and `rank_topo` is
never read (single occurrence in the tree).  Zero blast radius; left alone
deliberately — deleting it is not "the same idiom as an existing correct call
site".

### (a-5) audited and CORRECT (do not touch)

`bandstructure/htransform.py:39`, `gw/gw_jax.py:88`, `gw/kin_ion_io.py:222`,
`psp/get_DFT_mtxels.py:948`, `psp/get_dipole_mtxels.py:546`,
`centroid/kmeans_cli.py` sharded branch — all build the WORLD mesh from the
**full** global list, which is what `jax.devices()` is for.

## 1.2 CLASS (c) — bare `jax.device_put` where `device_put_process_local` is required

The AO sweep (`SPEEDUP_SCORECARD.md` §AO.1, branch `centroid-load-collectives`)
converted ~26 sites and explicitly documented what it "audited and deliberately
left".  The stragglers below are **not on that list** and are not covered by its
exemptions.

### (c-1) `src/solvers/davidson.py:217` — FIXED.  The one with real blast radius.

```python
V = jnp.zeros(shape, dtype=dtype)          # UNCOMMITTED
if sharding is not None:
    V = jax.device_put(V, sharding)        # -> assert_equal / process_allgather
```

Reached from **`src/bse/bse_lanczos.py:225`** (`--solver davidson`) with
`sharding = NamedSharding(mesh_xy, P(None,'x','y',None))` — multi-process — and
called once per subspace size in `{n_eig, 2·n_eig, 3·n_eig, 4·n_eig}`, i.e. the
antipattern fires **four times on the full trial-vector block before a single
matvec runs**.  Cost is `P × V.nbytes` materialised on every rank, on device and
(via `assert_equal`'s closing `np.asarray`) on the host.

Fix: `device_put_process_local` on a **zero-copy `np.broadcast_to` view** — the
same trick AO.1 used for `bse_w_exact`'s probe seed — so no rank materialises the
global buffer on the host either; the helper's `np.ascontiguousarray(arr[idx])`
realises only this rank's shard.  Replication precondition is trivial (all-zeros).

Other callers (`psp/run_nscf.py:201`, `bse/davidson_absorption.py`) pass no
sharding and are unaffected.

### (c-2) `src/gw/ppm_tau_kernel.py:524` — FIXED.

`precompile_sigma`'s AOT dummy `E_A`: `jnp.zeros((nk_tot, nb_full), f64)`
(uncommitted) onto `rep_2d = NamedSharding(mesh_xy, P(None,None))`
(multi-process) — a real P-linear collective **inside the AOT precompile**.  On
the GN-PPM path (`compute_mode = gn_ppm`, `use_ppm_sigma = true`), i.e. the
production Σ path.  Payload is small (`nk·nb·8 B`) but the collective is not
free and the site is unambiguous.  Fixed with `device_put_process_local` on host
numpy; produces the same committed `NamedSharding(P(None,None))` array, which is
the only property the dummy has to have.

### (c-3) audited, NOT the antipattern (with the reason)

* `gw/qsgw_utils.py:617` — `jnp.asarray(sig_x)` where
  `sig_x = sigma_result.sigma_x_kij_ry` is annotated and produced as a
  **committed** `jax.Array`; `jnp.asarray` on a `jax.Array` is identity and
  preserves `_committed`, so the `:447` reshard branch is taken.  Would become a
  live hit the moment any producer hands back numpy — worth a defensive
  conversion, not fixed here because it is not currently firing.
* `isdf/core.py:1709` (`jnp.concatenate` of jit outputs), `bse/bse_io.py:708,717`,
  `bse/exciton_bands.py:636,649,688,722`, `bandstructure/htransform.py:172,385,386,395,874`,
  `bandstructure/bse_setup.py:180` — all COMMITTED global operands ⇒ reshard branch.
  (This matches AO.1's own "audited and deliberately left" list.)
* `bse/bse_io.py:390,494,544` — `jax.device_put(local_np)` with **no** sharding,
  then `make_array_from_process_local_data`.  That IS the correct process-local idiom.
* `file_io/wfn_loader.py:1218` — guarded by `process_count() > 1` above it; only
  reachable at P=1.
* `runtime/__init__.py:690` — psum warm-up; AO.1 exempted it ("warming the wire
  is its job") and that still holds.
* `common/collectives.py:185,193,206,214,216` — inside the helper itself.
* `*_test.py` / `*_bench.py` / `*_sweep.py` — out of scope.

## 1.3 CLASS (b) — eager (outside-jit) full-tile ops on sharded/large arrays

No drop-in idiom exists for these (fixing them means wrapping in `jit` or
restructuring), so none were changed.  Ranked by blast radius:

| rank | site | what | path | note |
|---|---|---|---|---|
| 1 | `psp/get_DFT_mtxels.py:196` (`valence_density_from_kpoint`) and `:322` (`compute_valence_density`) | eager `jnp.fft.ifftn` | reached at `:594`/`:814` with **`wfn_k_sharded`** (`with_sharding_constraint(global_psi_G, P(('x','y'),…))`), and `wfn_k_sharded[i]` is sliced per-k at `:637`/`:835` — each slice on a mesh-sharded array is an implicit gather | **This is the function 2cbd824's SIGSEGV faulted in.** The centroid caller is now process-local, so the *correctness* half is closed; the *eagerness* half is not. Reachable only from `get_DFT_mtxels.main()`/`get_kin_ion()` (the standalone psp CLI) — GW production goes through `gw/kin_ion_io.build_valence_density_distributed`, which is the distributed twin and is documented as "SAME arithmetic". Note AO.7 already records that multi-process CLI harnesses of this shape are broken. |
| 2 | `bse/bse_io.py:713-716`, `bse/exciton_bands.py:729-731` | `local_ifftn3` / `local_fftn3` called at TOP LEVEL on `data["W_q"]` (globally sharded) | BSE W-direct setup | **Documented-contract violation**: `common/fft_helpers.py:307` says "Call this DIRECTLY from code that is already inside a `shard_map`". Values are safe because the FFT axes (the k axes) are the REPLICATED ones in `P('x','y',None,None,None)`, so XLA partitions it — but the intermediates are materialised eagerly. `e63bc8a` already hoisted the Lanczos-loop instance into a donated top-level jit; these two are the remaining ones. |
| 3 | `bandstructure/htransform.py:417-440` (`solve_q0_galerkin`) | eager `svd`/`lstsq`/`cholesky`/`@` chain | htransform Galerkin | operands are explicitly gathered to `rep` first (`:172`), so this is a deliberate replicated dense solve, not an accident — listed for completeness. |
| 4 | `isdf/core.py:1709` | eager `jnp.concatenate` over q-batch parts | ISDF C_q factor | materialises a full copy of the concatenated (sharded) stack outside jit. |
| — | `common/zeta_projection.py:744,785` | eager `cholesky`/`eigh` on `gram_S` | ζ projection | **correct as written** — `gram_S` is documented replicated and small ("replicated" is in the comment at `:785`). |

The remaining ~360 AST hits are host-numpy setup (`np.fft.fftfreq`, lattice
`@`, minimax fits, Coulomb kernels on G-vector tables) or benches/tests.

## 1.4 CLASS (d) — silent gathers

`jax.device_get` on a non-fully-addressable array **raises** (loud, not silent);
`np.asarray(<replicated global array>)` gathers to host silently.

Two modules carry the correct guard:
* `bse/exciton_bands.py:108` `_gather_host` — branches on `is_fully_addressable`,
  `device_get` when local / `process_allgather` when remote, with the reason for
  the branch written out.
* `bse/bse_davidson_helpers.py:41` `_gather_to_host` — same.

Three do **not**:
* `bse/bse_w_exact.py` — `:445,446,493,640,641,656,657,689,690,710,711,727,728,743,744,776,777`
  all `np.asarray(jax.device_get(...))` on slices of `data["W_q"]` /
  `data["V_q_full"]`, which are `P('x','y',None,None,None)` — slicing the k axes
  leaves μ,ν sharded, so these **raise at P>1**.  Oracle/`--compare-w0`/`--compare-wq`
  paths, not the default solve.
* `bse/bse_feast.py:453-455,917` and `bse/bse_pseudopoles.py:59,96,125,157,202` —
  same shape, alternative solvers.

Fix for all three is the one-line delegation to a shared `_gather_host`.  Not
applied: it is a behaviour change on paths with no gate, and `bse_w_exact`'s
tolerance for a P>1 raise is unknown.  Ranked below (c-1)/(c-2) because the
failure is loud.

## 1.5 Gate

`py_compile` PASS on all five edited files.

Value-parity job **7879693** (`wk_REL/harness/eagerfix_gate.sbatch`), 2 nodes × 2 ranks
= P=4, partition `small`, account PHY25006.  Two frozen v2 snapshots differing
in **exactly the 5 edited files** (verified by `diff -rq` at build time and
re-printed in the job log with per-file sha256), each with its manifest verified
at job START **and** END, `PYTHONDONTWRITEBYTECODE=1`:

* FIX  `wk_REL/srcsnap_eagerfix_20260729_092117_b4c7bca`
* BASE `wk_REL/srcsnap_eagerfix_20260729_092117_BASE_b4c7bca`

Cells:
1. **Placement equivalence** (`eagerfix_place_probe.py`) at P=4 on a 2×2 mesh —
   builds the dummy both ways for four `(shape, spec)` cases including the two
   production signatures, and asserts shape / dtype / sharding /
   `is_fully_addressable` / `_committed` / `max|old−new| == 0.0` exactly.  This
   is the decisive gate for these two edits: the buffers' VALUES are irrelevant
   (AOT/warm-up dummies) but their
   `(shape, dtype, sharding, committed-ness)` tuple is load-bearing — if it
   drifts, pjit silently re-traces and the precompile buys nothing.
2. **gnppm fixture value parity** at P=4 (`tests/regression/gnppm_debug`,
   `compute_mode = gn_ppm`, `use_ppm_sigma = true`, so
   `ppm_tau_kernel.precompile_sigma` IS on the path): BASE leg then FIX leg,
   same allocation, same nodes, back to back; `sigma_diag`/`eqp0`/`eqp1`
   byte-compared BASE-vs-FIX and FIX-vs-`sigma_diag_gnppm_ref.dat`.

## 1.6 Gate RESULT — job 7879693, `sacct` state **COMPLETED**, ExitCode 0:0, elapsed 00:03:34

Log: `wk_REL/results/logs/eagerfix.7879693.out`.  Run dirs
`wk_REL/eagerfix_run_7879693_{probe_FIX,gw_BASE,gw_FIX}/leg.log`.

**Cell 1 — placement equivalence: PASS**, all 4 cases × all 4 ranks.  For the
two production signatures, on every rank:

```
[probe] jax=0.9.1 processes=4 devices=4 backend=cpu coll=gloo   mesh 2x2
[probe] PASS davidson_warmup_V shape=(8,16,16,9) spec=P(None,'x','y',None)
[probe]      ok  shape / dtype / sharding
[probe]      ok  fully_addressable    False vs False
[probe]      ok  committed            True vs True
[probe]      ok  value_exact_zero     max|old-new|=0.0
[probe] PASS ppm_tau_E_A shape=(9,48) spec=P(None,None)   (same six, all ok)
```

`committed True vs True` and `fully_addressable False vs False` are the
load-bearing ones: the replacement is indistinguishable to pjit, so the AOT /
warm-up compile still keys the same and still covers the production dispatch.

**Cell 2 — gnppm P=4 value parity: PASS.**  `gw_BASE rc=0` (66 s),
`gw_FIX rc=0` (59 s), same allocation, same nodes (c211-032/033), back to back.
Five artifacts, **2160 data rows total, byte-identical**:

| artifact | data rows | md5 (comments stripped), BASE == FIX |
|---|---|---|
| `sigma_diag_gnppm_test.dat` | 441 | `322f0993b0f0f99f…` |
| `eqp0.dat` | 423 | `2c8a3503eb95e111…` |
| `eqp1.dat` | 423 | `b7198d18227bd30e…` |
| `eqp_g0w0.dat` | 441 | `241694ba3b5a3212…` |
| `sigma_freq_debug.dat` | 432 | `31f488edd303571b…` |

A whole-file `diff` reports exactly ONE differing line per artifact — the
`# Generated by LORRAX 0.1.0 at <UTC>` header stamp (14:26:51 vs 14:27:50).
Excluding that line the diff is 0 lines on all five.

**Two gate-instrument defects, recorded so the next reader is not misled:**
1. The in-job `diff -q` reported `*** DIFFERS ***` on `sigma_diag`.  That is the
   TIMESTAMP HEADER, not data.  A byte-diff over-discriminates on any artifact
   that stamps its own generation time — same lesson as 2cbd824's centroid-file
   note.  The discriminating compare is `grep -v '^#' | md5sum`.
2. The in-job "FIX vs frozen reference" awk printed
   `max|Delta| over data cols = 7.000000e+00`.  That number is an **artefact of
   the awk**, which `paste`s the two files and differences column i against
   column i+NF/2 — including the integer band/k index columns.  It is NOT a
   physics delta and must not be quoted.  BASE-vs-FIX is the A/B that isolates
   the change; the frozen-ref compare needs the project's own comparator.

**srcpin: both snapshots MANIFEST VERIFIED at job START and at job END** — "no
file changed AND none appeared" — so both legs prove their own source was
immutable for the whole run.  351/351 files hashed, no cached bytecode.

`RC_PROBE=0  RC_BASE=0  RC_FIX=0  RC_END=0`.

## 1.7 The one corner the P=4 gate does NOT reach — job 7879703, PASS

`device_put_process_local` has an early-out at
`process_count() <= 1 or sharding.is_fully_addressable` that hands the host
array **straight to `jax.device_put`**.  The `davidson.py` fix seeds it with a
`np.broadcast_to` view (0-strided, read-only, non-contiguous); the P=4 legs
never reach that branch, because they take the
`make_array_from_single_device_arrays` path where
`np.ascontiguousarray(arr[idx])` has already materialised the shard.

**P=1-with-a-sharding is a LIVE path** — `bse_lanczos.py` passes `bse_sharding`
to `warmup_davidson_jit` unconditionally, and `ppm_tau_kernel` builds `rep_2d`
unconditionally — so this is measured, not assumed.

Job **7879703** (`wk_REL/harness/eagerfix_p1_probe.sbatch`, `small`, 1 node) runs the
same probe single-process at forced device counts 1 and 4, with BOTH host seed
forms (`np.zeros` contiguous and `np.broadcast_to` 0-stride) on every case.

### RESULT — job 7879703, `sacct` **COMPLETED**, ExitCode 0:0, elapsed 00:01:09

**PASS on every corner.**  Both host seed forms (`np.zeros` contiguous AND the
`np.broadcast_to` 0-stride read-only view), all four cases, at forced device
counts **1** and **4**, single-process (so the fully-addressable early-out IS
the branch taken):

```
### device count = 1     [probe] processes=1 devices=1  mesh 1x1
PASS davidson_warmup_V [np.zeros(contig)]                 PASS ... [np.broadcast_to(0-stride view)]
PASS ppm_tau_E_A       [np.zeros(contig)]                 PASS ... [np.broadcast_to(0-stride view)]
PASS rep_3d / shard_x_only, both seeds
[probe] PLACEMENT-EQUIVALENCE PASS    rc=0
### device count = 4     [probe] processes=1 devices=4  mesh 2x2   -> same 8 PASS, rc=0
```

So `jax.device_put` accepts the 0-strided view on the early-out branch and the
result is indistinguishable from the old buffer there too.  **Both fixes are now
fully gated; nothing is owed.**

Log: `wk_REL/results/logs/eagerfix_p1.7879703.out`.

*(Historical: had this reported `**FAIL** … RAISED` on the broadcast_to seed,* the fix would have been
one line: in `src/solvers/davidson.py`, replace
`np.broadcast_to(np.zeros((), dtype=dtype), shape)` with
`np.zeros(shape, dtype=dtype)` — the same contiguous form already used and
gated in `ppm_tau_kernel.py`.  That costs one 1× host allocation instead of the
zero-copy view; it does **not** reintroduce the P× collective, which is the
whole point of the change.  Re-run 7879703 to confirm.)*

Check with:
```
sacct -j 7879703 --format=JobID,State,Elapsed,ExitCode
grep -a '^\[probe\]' /scratch2/08271/jackmc/lorrax_setup/wk_REL/results/logs/eagerfix_p1.7879703.out
```

---

# TASK 2 — psum_scatter ARCHAEOLOGY

Corpus: `SPEEDUP_SCORECARD.md`, `SESSION_REPORT_2026-07-*.md`, all `wk_*/`,
the `.claude` transcripts, and `git log --all -S/--grep`.

## 2.1 Headline

**No prior sighting.**  Passes A–F produce zero events before 2026-07-29 in
which a `psum_scatter`/reduce-scatter output was observed to be silently wrong.

* **Pass C (invariant failures): zero.**  Every `check_hermitian` occurrence in
  all 1913 job logs under `/scratch2/08271/jackmc/lorrax_setup/` is a PASS.  No
  "not Hermitian", no PSD/Cholesky failure, no parity FAIL, no exact-0 gate
  failure anywhere before job 7879491.
* **Pass B (nondeterminism-in-results): zero.**  All hits are transport deaths,
  compile flakiness or wall-time noise.
* **Pass D: zero.**  No instance anywhere in the corpus of a job that produced
  numbers, was re-run, and produced *different numbers*.
* **Pass E: zero** prior "plausible but wrong" magnitude reports on a collective.

## 2.2 The one near-miss, and why it matters

Transcript `ea1ecbf8-2796-4812-977b-cb41dccbd8f8.jsonl:909`,
**2026-07-25T01:03:26Z** — an audit hunting for silent P>1 corruption reached
`psum_scatter` and filed it under:

> `## SUSPECTED (P>1 gap — but a crash, not silent corruption)`

and at 02:03Z / 02:16Z: *"It's a **loud** failure, not silent corruption…
`psum_scatter(tiled=True)` will *raise* on an indivisible axis"*.

That reasoning is about **JAX's shape checking**, not about **gloo's data
plane**, and it is exactly the assumption 2026-07-29 overturns.  It is also the
reason no invariant was ever placed on the Σ or BSE reduce-scatter outputs.
The only written acknowledgement of any silent hazard in the primitive before
07-29 is a parenthesis in the guard landed that night
(`src/common/contract_bands.py:77-78`, commit `9c687a5`): *"or, with a future
non-tiled variant, misalign silently"* — scoped to a hypothetical variant of our
own code, not to gloo.

Earliest mention of the primitive at all is constructional and old: `e1e90c3`
(2026-01-26), `d7038d3`/`d41fedf` (2026-04-17).  26 commits touch the string.

## 2.3 What the corpus DOES contain: the loud sibling

`SPEEDUP_SCORECARD.md:5100-5128` (AC.2, job 7876062, P=144), `:5624-5700` (AF),
`:9538-9568` (AY.2, jobs 7877753/7877761, P=64),
`wk_REL/docs/scale10k_notes.md:600-615` (jobs 7879486/7879493, P=64 BSE) —
`Gloo ReduceScatter failed: Connection closed by peer`, `Read timeout`,
`GetKeyValue() timed out`.

**PARTIALLY CONSISTENT** on primitive/transport/fabric, **INCONSISTENT** on every
decisive discriminator: deterministic at scale rather than ~5 %, not
segment-shaped, `rc≠0` crash rather than a plausible wrong value, and no
all-reduce control was ever run.  Blast radius nil — these killed their jobs.

The 7879486/7879493 pair *was* dismissed as node luck ("transport luck, NOT the
fix"), and that dismissal was correct — it was settled by the back-to-back A/B
in job 7879500.

Also checked and **INCONSISTENT**: J.9 SUMMA NaNs-with-rc=0 (`:1188-1196`,
deterministic algebra bug in our code — but note it is why `check_hermitian`
exists at all); the `z_q` band-gather silent wrong answer (`a549471`,
deterministic); the ~21 % `impl=mpi` multi-node segfault (AS, `:8712-8737` —
crash, mpi not gloo, root-caused to two threads in MPI progress under
`MPI_THREAD_FUNNELED`); the M/K `H0`/`V_H` "12×12 corruption" (`:924-940`,
catastrophic cancellation, physics not parallelism); routine
`DEADLINE_EXCEEDED` retries (startup timeouts, `rc≠0`).

## 2.4 Q2 — which quoted numbers ran a gloo `psum_scatter`

Call sites (union over all refs; 5 live production files):

**AFFECTED**

| stage | site | shape | default? |
|---|---|---|---|
| GW Σ_c(ω) τ-projection | `gw/ppm_tau_kernel.py:85` → `common/contract_bands.py:584,597,609,614,621,626,632,637` | `psum_scatter('y',n)` then `psum_scatter('x',m)`; 4/τ node (2 after AK.9, `dc30af4`); O(100) τ nodes ⇒ **several hundred executions per run** | yes, but `compute_mode = gn_ppm` only. **COHSEX has none** (`gw/cohsex_sigma.py`; scorecard :917) |
| BSE matvec | `bse/bse_stack_matvec.py:126,129`; `bse/bse_ring_comm.py:274,836,966` | 2 per matvec, inside `lax.scan` inside `shard_map`; once per Lanczos/Davidson iteration | yes — **and BSE cannot run under `impl=mpi` at all** (TASK 3), so BSE is *forced* onto gloo |
| ISDF `distributed` ζ tier | `isdf/core.py:1947` | `all_gather('x')` + `psum_scatter('y')`, q-blocked at 128 MB since AF | no — default `replicated_rank_truncate` has none |
| ζ-basis projection | `common/zeta_projection.py:451,452,502,505` | the reported one-pass chain | new (`6bf28bc`, 07-29); no campaign number depends on it |

**NOT AFFECTED (verified, not assumed):** `V_q` (`gw/v_q_g_flat.py`, `psum` only);
W Dyson / `screening.py` / `w_isdf.py`; the default ζ tier; COHSEX Σ;
htransform/bandstructure/Hartree — scorecard X.6 (`:3565-3598`) is a full
optimized-HLO collective table over **250 modules**: 10 collectives total, *"no
all-to-all, no reduce-scatter, no collective-permute in any of the 250 modules"*.

**Transport ledger:** everything through 2026-07-27 was gloo (em1 before AL, ib0
after, `6ed2414`); **AQ 4962c/P=64 and every AY Σ-perf round (7878038…7879005)
ran `impl=mpi`** and are therefore not gloo evidence either way; all BSE runs
(7879463 / 7879470 / 7879500) are gloo/ib0.

**Quoted numbers that ran a gloo `psum_scatter`:** the 606c/P=80 flagship
`eqp0`/`eqp1`/`eqp_g0w0`/`sigma_diag` (AK 7876528, AL 7876541); the 785c/P=16
`eqp0` baseline (AV, AY verify 7877788); **all BSE eigenvalues** (7879463 P=4/785c;
7879470 leg 3 N_mu=10015/P=64); AD.3 GN-PPM gates; AF chunked-ζ gates; pre-AY
`sigma.exec` timing rows.

## 2.5 The counter-evidence that bounds the damage

§4.3 of the upstream report establishes the wrong value **differs every
occurrence**.  Therefore two independent gloo runs of the same path agreeing
*bit-for-bit* is strong evidence neither was corrupted.  Measured:

| gate | source | psum_scatter executions |
|---|---|---|
| AL.3 five-way **BYTE-IDENTICAL** at P=80 across em1-restart / ib0-restart / ib0-full-recompute / ib0-full-with-cache | `:7740-7752`, job 7876541 | several hundred per run × 5 |
| AK razor: `max|Δ| = 0.00e+00` over **10 080 values**; B vs C byte-identical ×4 | `:7432-7444`, `:7633-7639`, 7876528 | same order |
| AK.10 em1-vs-ib0 A/B: all four artifacts byte-identical | `:7697-7699`, 7876536 | 785c/P=16 |
| AD.3 GN-PPM P=4/P=8/GPU 2×2: `0.00e+00` over **2428** values | `:5418-5423` | the dedicated gate for this path |
| AF chunked ζ vs replicated: `0.00e+00` over **1888** sigma values at P=4 and P=16 | `:5635-5636` | the C⁺ `psum_scatter` |
| BSE P=64/10015 base-vs-fixed back to back in one allocation: eigenvalue parity **max rel = 0.000e+00** | `scale10k_notes.md` §6.2b, 7879500 | 40 iters × 2 × 2 legs |

**Inference, flagged as such:** the corrupting instance had a **253 MB**
complex128 per-instruction payload on 2-rank replica groups; Σ's are 5.11–40.9 MB
and the C⁺ one is chunked to ≤128 MB.  Payload size is the leading candidate
discriminator.  This is **not established** — §6.3 of the upstream report says
payload/shape/P were not swept, and `wk_REL/OWNER_DECISIONS.md:118-121` already
records the `contract_bands` 3.037e-16 cross-P agreement as *"unexplained
difference, NOT evidence of safety"*.

## 2.6 The structural gap this exposes

`check_hermitian` is called at exactly five sites (`common/sanity.py:53`,
`gw/ppm_sigma.py:279,281`, `gw/screening.py:472`, `gw/gw_init.py:878`) — on
`V_q[0]`, `W[0]`, and PPM `B_q`/`Ω_q`.  **None of them is downstream of a
`psum_scatter`.**  Neither `ppm_tau_kernel.py`, `ppm_accumulators.py`,
`bse_lanczos.py` nor `bse_stack_matvec.py` carries any sanity gate at all.  For
the two production paths that ARE exposed (Σ_c(ω) and BSE), the only corruption
detector that has ever run is cross-run byte-comparison — which is why §2.5 is
the entire evidence base, and why any single un-A/B'd run on those paths carries
no invariant.

Recommended (owner call): a Hermiticity/invariant gate downstream of the Σ
reduce-scatter chain and of the BSE matvec, per §7.2 of the upstream report — it
costs one reduction and is the only thing that would have caught this.

---

# TASK 3 — BSE / `impl=mpi` THREAD AFFINITY — **SOLVED AND DEMONSTRATED**

## 3.0 The discriminator: the error is XLA's, not MPI's

**The error string is emitted by jaxlib/XLA, NOT by any MPI library.**

```
$ strings /work2/08271/jackmc/frontera/lorrax_env/.venv/lib/python3.12/site-packages/jaxlib/libjax_common.so \
    | grep -i "requested from a thread"
MPI: Communicator requested from a thread that is not the one MPI was initialized
from. Multiple threads/devices per process are not yet supported.
```

The same binary carries `external/xla/xla/backends/cpu/collectives/mpi_collectives.cc`
as a source-path literal and the symbols `xla::cpu::MpiCollectives::Init()`,
`::Finalize()`, `xla::cpu::MpiCollectives` typeinfo/vtable, plus a static-init
`site` lambda inside `Init()`
(`_ZZZN3xla3cpu14MpiCollectives4InitEvENK3$_0clEvE4site`).  The identical grep
over Intel MPI's `libmpi.so*` and over wk_AS's
`mpiw_thr_install/lib64/libmpiwrapper.so` returns **nothing**.

### What this establishes

* **Verdict: (C), structural and upstream.**  The refusal is XLA's own
  `MpiCollectives` guard — it records the thread that initialised MPI and
  refuses to hand out a communicator from any other thread.  The message states
  the scope in its own words: *"Multiple threads/devices per process are not yet
  supported."*
* **(A) is REFUTED as a sufficient fix.**  No MPI-library thread level can
  satisfy a check that XLA performs itself on the calling thread's identity.
  This is why `wk_AS`'s THREAD_MULTIPLE-patched MPIwrapper — which
  `mos2_4x4_test/bse_inner.sh:42-44` DOES load on the `BSE_COLL=mpi` branch —
  never helped: it was fixing the wrong layer.  Anyone about to rebuild that
  wrapper should stop.
* It also explains `556ffa1`'s result (communicator warm-up measured
  INSUFFICIENT for standalone consumers): warming from the main thread cannot
  help if the *later* request also has to come from the main thread.

### What this does NOT establish

* The exact thread level XLA requests at `MPI_Init_thread`.  All four
  `MPI_THREAD_*` literals are present in the binary, so their presence is an
  enum-name table and proves nothing on its own.  Reading
  `xla/backends/cpu/collectives/mpi_collectives.cc` at the jaxlib-0.9.1 tag is
  the way to settle it; do not infer it from `strings`.
* Whether GW survives only because its collectives happen to be issued from the
  main thread while BSE's Lanczos-loop ones are not — that is the natural
  reading, and it is consistent with AS.7 being "GW-ONLY", but it is not
  measured here.
* Whether any XLA flag (single intra-op thread) moves the issuing thread back to
  the main one.  That is the one remaining cheap on-our-side experiment.

## 3.1 CONFIRMED MECHANISM — `MPI_Is_thread_main`, not a thread level

Disassembly of `xla::cpu::MpiCollectives::CreateCommunicators` (`@0xcc7a730` in
`jaxlib/libjax_common.so`) gives the guard verbatim:

```
callq  MPI_Is_thread_main       ; lea -0x54(%rbp),%rdi
cmpl   $0x0,-0x54(%rbp)
je     <error>                  ; -> absl::UnknownError(<the 147-byte string>)
...                             ; else: MPI_Comm_split(MPI_COMM_WORLD, color, key, &comm)
```

Two corrections to what the committed docs say:

* The test is **`MPI_Is_thread_main`** — not a thread-id comparison and not a
  thread-LEVEL test.  It is false on any non-initialising thread **at every
  level, including `MPI_THREAD_MULTIPLE`**.
* It fires **only on communicator CREATION**, once per clique key.
  `MpiCommunicator::AllReduce/ReduceScatter/AllGather/...` carry no such check.

Also from the binary: `MpiCollectives::Init()` (`@0xcc7a600`) calls
`MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided)` and **never reads
`provided`**.

**(A) FALSIFIED twice** — structurally (no thread level changes
`MPI_Is_thread_main`) and empirically: the failing job 7879458 *already* ran the
wk_AS THREAD_MULTIPLE wrapper (`mos2_4x4_test/bse_inner.sh:43`), and a probe in
that exact cell measured `granted_thread_level=3 (MULTIPLE)` on Intel MPI 2019 U9.

**(B) FALSIFIED** — mpi4py attached with `rc.initialize=False` reports
`Is_thread_main(python-main)=True`; XLA's `MPI_Init_thread` already runs on the
Python main thread and the whole stack shares one Intel MPI.  Ordering is
correct; the *executing* thread is wrong.

**(C) CONFIRMED — with a correction to the `b4c7bca` banner.**  The banner's rule
("collectives inside `lax.scan`/`while_loop` inside `shard_map` inside one jit")
is **not** the discriminator: a clean-room probe of exactly that shape passed
under `impl=mpi`, as did a bare subgroup `psum` with no world warm-up.  The real
discriminator is which XLA:CPU execution path the program takes —
`ThunkExecutor::ExecuteSequential` runs thunks inline on the caller (main)
thread, so small graphs pass; the parallel `ThunkExecutor::Execute<ReadyQueue>`
path dispatches thunks to intra-op pool workers, so real graphs fail.  Both are
in the binary (`Constructed ThunkExecutor with %d thunks: ... is_sequential=%v`).
This also explains the "unexplained" gap in `556ffa1`: it is
**small-graph-vs-parallel-graph, not standalone-vs-production**.

No config knob exists — the complete `set_xla_cpu_*` DebugOptions list in this
jaxlib contains nothing that forces sequential thunk execution, and
`jax_cpu_enable_async_dispatch=0` is not the lever (probe passed with and without).

## 3.2 THE FIX — on our side after all, and DEMONSTRATED

`MPI_Is_thread_main` in `libjax_common.so` is an **MPItrampoline stub**
(`jmpq *MPIABI_Is_thread_main`) resolved at dlopen from **MPIwrapper — which we
build**.  An env-gated override was added with the same macro-wrap trick as the
AS.4c THREAD_MULTIPLE patch:

* source  `wk_REL/MPIwrapper_thrmain/src/mpiwrapper.cxx`
* built   `wk_REL/mpiw_thrmain_install/lib64/libmpiwrapper.so`
* gate    `LORRAX_MPI_FORCE_THREAD_MAIN=1`, **default OFF** (byte-for-byte the
  certified wrapper's behaviour when unset)

Legality: the same wrapper already upgrades every init to
`MPI_THREAD_MULTIPLE`, so `MPI_Comm_split` from a pool thread is legal MPI, and
XLA creates cliques deterministically so all ranks split in the same order.
Blast radius is exactly XLA's CPU collectives — mpi4py, h5py and the FFI host
`.so` all link Intel `libmpi.so.12` directly and never see the override.

### Demonstration — VERIFIED from `sacct` and from the logs on disk

**Job 7879697** `AS7_BSEfix` COMPLETED — MoS2 4×4, 785c, P=4, TDA Lanczos (the
7879458 deck), `development`.  Per-step `sacct`:

| step | cell | sacct (verified) | result |
|---|---|---|---|
| `.0` | ctrl_mpi_thr (== job 7879458) | **FAILED 1:0** | refusal |
| `.1` | neg_mpi_new (new wrapper, gate OFF) | **FAILED 1:0** | refusal — rebuild alone changes nothing |
| `.2` | **fix_mpi_force (gate ON)** | **COMPLETED 0:0** | **0 refusals** |
| `.3` | ref_gloo | **COMPLETED 0:0** | reference |

Eigenvalues read from `wk_REL/results/logs/bse_as7_ab.7879697.out` — the impl=mpi fix cell and
the gloo reference cell print **character-identical** values:

```
Lowest 4 eigenvalues (eV): [1.30537661 1.3504201  1.42411254 1.50449023]
Lowest 4 eigenvalues (Ry): [0.09594338 0.09925401 0.1046703  0.11057795]
```

and that eV vector is the same one TASK 2 §2.4 records for the original gloo run
job 7879463 — so this is a three-way agreement, not a two-way one.

**Job 7879702** `AS7_BSEreps` COMPLETED — 4/4 reps, steps `.0`–`.3` all
**COMPLETED 0:0** (verified), identical eigenvalues each time.  With 7879697.2
that is **5/5 green under `impl=mpi`**.

**Job 7879684** `AS7_thrprobe` COMPLETED — steps `.0`–`.5`, `.7` COMPLETED 0:0;
`.6` FAILED 1:0 and that cell is **VOID** (a nonexistent XLA flag was passed and
it died at startup — not evidence of anything).

Snapshot `wk_REL/srcsnap_as7mpi_20260729_092424_1a52d51` (343/343 files hashed,
0 bytecode, byte-identical to `srcpin_1a52d51`).  No git commits; nothing under
`/work2/.../lorrax` touched.

## 3.3 Caveats — read before making this a default

* **Only measured at P=4 / 2 nodes.**  AS.4b's separate concurrent-MPI-progress
  segfault class (~29 %) appeared only at P=16 × 8 nodes.  This fix deliberately
  makes `MPI_Comm_split` run concurrently with main-thread MPI, so it needs an
  AS.4c-style rep ledger **at scale** before any default flips.
* **Doc-integrity finding:** commit `556ffa1` states all five job-7879485
  variants "die with the §3.5 error".  The captured log does not support that —
  `zproj_mpiprobe.sbatch`'s `grep | head -8` kept only CUDA noise, so only
  `ExitCode 4:0` is evidenced.  Separately, the probe contradicts the
  "world-collective-first" premise in `ensure_grouped_collectives_ready`'s
  docstring: a bare subgroup `psum` with **no** world warm-up passed.  The
  ordering was never the mechanism.

## 3.4 What this unlocks — run these next, in this order

1. **The decisive corruption experiment.**  Run
   `wk_REL/probes/gloo_psum_scatter_repro.py` under `impl=mpi` with
   `LORRAX_MPI_FORCE_THREAD_MAIN=1`.  Upstream report §6.4 lists "not compared
   against the mpi collectives backend" as an explicit NOT-established *because
   that backend would not create grouped communicators* — that blocker is now
   removed.  This separates "gloo's reduce-scatter is broken" from "jax's CPU
   collective lowering is broken on every transport", which decides whether the
   report is filed against gloo or against XLA, and whether the ~5 % also
   applies to the mpi-transport campaign numbers (AQ 4962c/P=64, every AY Σ round).
2. **BSE off gloo** at production scale, with the AS.4c-style rep ledger from §3.3.

# HANDOFF — what is established, what is not, exact next commands

## Snapshots created (do not delete; both are frozen and manifest-verified)

| id | what |
|---|---|
| `wk_REL/srcsnap_eagerfix_20260729_092117_b4c7bca` | **FIX** — worktree at `b4c7bca` + the 5 edits |
| `wk_REL/srcsnap_eagerfix_20260729_092117_BASE_b4c7bca` | **BASE** — identical except those 5 files reverted to `git HEAD` |
| `wk_REL/srcsnap_as7mpi_20260729_092424_1a52d51` | TASK 3 source pin (343/343 hashed, byte-identical to `srcpin_1a52d51`) |

TASK 3 build artifacts (new, keep): `wk_REL/MPIwrapper_thrmain/src/mpiwrapper.cxx`,
`wk_REL/mpiw_thrmain_install/lib64/libmpiwrapper.so` (gate
`LORRAX_MPI_FORCE_THREAD_MAIN=1`, default OFF).

Pointers: `wk_REL/snapshots/pointers/EAGERFIX_SNAP_FIX`, `wk_REL/snapshots/pointers/EAGERFIX_SNAP_BASE`.
Harnesses: `wk_REL/harness/eagerfix_gate.sbatch`, `wk_REL/harness/eagerfix_p1_probe.sbatch`,
`wk_REL/harness/eagerfix_inner.sh`, `wk_REL/probes/eagerfix_place_probe.py`.
Jobs: **7879693** COMPLETED (main gate, PASS), **7879703** (corner probe, in
flight at handoff), 7879691 FAILED in 1 s (`set -u` vs the apptainer module's
`BASH_COMPLETION_DEBUG` — fixed by `set +u` around `module load`; no other
meaning).

## Working-tree state — NOT COMMITTED, nothing half-applied

`/work2/08271/jackmc/frontera/lorrax` on `fix/zq-band-gather-device-invariance`:

```
M src/common/gpu_utils.py       jax.devices()  -> jax.local_devices()
M src/gw/gw_output.py           jax.devices()  -> jax.local_devices()
M src/gw/isdf_fitting.py        jax.devices()  -> jax.local_devices()
M src/gw/ppm_tau_kernel.py      device_put     -> device_put_process_local
M src/solvers/davidson.py       device_put     -> device_put_process_local
M manual/05_isdf/5.1_pair_density_factorization.md   (PRE-EXISTING, not mine)
```

All five are complete edits, `py_compile` clean, and gated by 7879693 except the
one corner in §1.7.  All five files are byte-identical across `HEAD`,
`wsREL-isdf-window` and `wt-REL10k-scale` (checked), so they apply cleanly to
any of the three.

## Established

* **TASK 2 is CLOSED with a clean negative.**  No prior sighting of the
  segment-0 / ~5 % / plausible-wrong-value signature anywhere in the corpus.
  The one near-miss (2026-07-25) actively cleared `psum_scatter` of silent
  corruption on reasoning that only covers JAX's shape checking, not gloo's data
  plane.  §2.5 lists six independent byte-identical gloo re-runs of the exposed
  paths, which — given the wrong value differs every occurrence — is real
  evidence those specific campaign numbers are clean.
* **TASK 1 fixes are gated**: placement equivalence exact at P=4, gnppm P=4
  BASE-vs-FIX byte-identical on 2160 data rows, srcpin verified start and end.
* **`2cbd824` is missing from `wsREL-isdf-window` and from `b4c7bca`.**
* **TASK 3 is SOLVED**: the refusal is XLA's own `MPI_Is_thread_main` guard in
  `MpiCollectives::CreateCommunicators`, fixable in OUR MPIwrapper, and
  demonstrated 5/5 green at P=4 with eigenvalues character-identical to gloo
  (jobs 7879697/7879702, sacct-verified per step).  (A) and (B) falsified.

## NOT established

* Whether the corruption is gloo-specific — but TASK 3 has now REMOVED the
  blocker (§3.4 step 1).  The experiment is runnable today.
* Whether the ~5 % rate holds at other P / payload / mesh — upstream §6.3;
  payload size is a *candidate* discriminator only.
* Whether the class-(b)/(d) items in §1.3/§1.4 are costing anything measurable —
  none were benchmarked, only identified.
* Whether the TASK 3 fix survives at scale — only P=4 / 2 nodes measured; AS.4b's
  concurrent-MPI-progress segfault class only appeared at P=16 x 8 nodes (§3.3).
* ~~The `davidson.py` P=1 corner~~ — CLOSED, job 7879703 PASS (§1.7).

## Highest-value next step

**Run the gloo-vs-mpi corruption comparison (§3.4 step 1).**  TASK 3 is CLOSED
with a working, demonstrated fix, and its only real value is that it unblocks
this: `wk_REL/probes/gloo_psum_scatter_repro.py` under `impl=mpi` with
`LORRAX_MPI_FORCE_THREAD_MAIN=1` and the wrapper at
`wk_REL/mpiw_thrmain_install/lib64/libmpiwrapper.so`.  It is a ~2-minute job on
`development` and it decides where the upstream report goes and how wide the
blast radius is.

Second: merge `2cbd824` into `wsREL-isdf-window` before anything on that branch
runs distributed.
