# PADDING AUDIT — synthesis (2026-07-08)

Question from the lead: **is padding done as cleanly — fewest lines, one convention — as
possible around the key GW operations?**

Tree: `sources/lorrax_D` @ `agent/memplanner-cleanup`. Scope reports:
`padding_audit_{zeta-core,vq-w-unfold,sigma-census,io-ffi}.md` (this dir). Background:
`ROOT_CAUSE.md` (the pad-extent device-invariance bug), SHARDING_RULES.md §2 (the contract).

**Commit-state correction (verified against the live tree):** the tree has moved since the
scope audits ran at 62b0365. HEAD is now `ed393fa` ("Tier-1 fixed-P pad-flip suite gate +
Tier-2 cross-P script; g0_mu logical on disk") — the previously-uncommitted `gw_init.py`
g0_mu clip is **committed**, and `mu_logical_mask` in `ppm_sigma._prepare_sigma_state` was
committed all along (62b0365), contrary to the tasking note. The only uncommitted state is
a modified `tests/multi_device/eqp_invariance_cross_p.py` plus untracked probe dirs
(`.padprobe/`, `.tier2_cross_p/`). **All findings below are findings about committed code.**

---

## 1. THE VERDICT

**No — not as clean as possible. The *design* is the minimal correct one; the
*implementation* around it is roughly 2× the code it needs to be, and dead/duplicated
pad machinery outnumbers the live pad logic.** Grade across scopes: **B−/B** (io-ffi
boundary A−, zeta-core B−, vq-w B, sigma-census B).

### What matches the ideal (and should be defended, not touched)

- **One birth**: μ pad extent is born in exactly one function, `runtime/padding.py:
  padded_mu_extent` (single source of truth incl. the `LORRAX_EXTRA_MU_PAD` test knob),
  carried as the integer pair `(Meta.n_rmu, Meta.n_rmu_padded)` (+ band twin
  `b_id_4_user`/`b_id_4`). Commit 083d209 forcibly unified the two V_q-side private
  round-ups (the knob exposed their drift as a shape crash). This IS the one-birth ideal.
- **Structural neutrality carries the physics for free**: every heavy GW operation —
  pair-density/CCT bilinears, the per-q V GEMM, χ₀, W(τ) build, TT Lorentz mixing, head
  rank-1 injection, cohsex Σ projections, ppm_windows, ppm_accumulators — has **ZERO
  pad-specific lines**. Zero pad rows in ψ/ζ/g0 make ~10 major consumers neutral by
  construction. This is the scheme working exactly as designed.
- **The disk boundary is genuinely one convention honored mechanically**: disk = logical,
  memory = padded, via SlabIO `valid_shape=`, ONE shared `_normalize_valid_shape` across
  all 3 backends, ONE C++ hyperslab clip per direction (write clip / read memset-zero).
  Not re-derived per reader/writer. Cleanest layer of the whole scheme.

### Where it falls short of "fewest lines, one convention" — the numbers

| Metric | Today (dedup across the 4 scopes) | Ideal / achievable |
|---|---|---|
| Distinct pad-aware sites | ~60 (79 raw scope entries, ~19 overlap) | **~40** |
| Structurally neutral (0 lines) | ~15 site-families — the majority of the physics | keep |
| Canonical-helper sites | ~25 | grows as ad-hoc collapses |
| AD-HOC per-site logic | ~20 sites, ~200–250 executable lines | **~8 sites** |
| MISSING / latent | 3 live bugs + 3 doc drifts + 3 unsafe defaults | **0** |
| Total pad-specific lines | ~1,150 live (padding.py 403; SlabIO contract ~225; consumers ~480; solver-divisibility ~43) **+ ~660 dead** (≈230 unused PadAxis API + 427 test lines for it) | **~600 live, 0 dead** |
| μ-extent divisor conventions | **3** (mesh-product `padded_mu_extent`; legacy `n_proc` in `Meta.*_jax`; `lcm(gx,gy)` in `bse_io.py:439`) | **1** |
| round_up spellings | **5–6** (`padded_mu_extent`, `meta._round_up`, `wfn_loader._pad_to`, 3× inline in `isdf/core.py`) | **1** |
| Logical-μ mask conventions | **3** (`fit_ppm n_mu_logical:int`; `fit_gn… mode_mask:(μ,ν)`; `_prepare_sigma_state mu_logical_mask:(μ,)`) — same mask built 3× | **1, then 0** (§2) |
| Array-level pad mechanisms | **2.5** (SlabIO valid_shape = live canon; PadAxis layer = dead shadow canon with zero src callers, still advertised by `slab_io.py:191`; ad-hoc `jnp.pad`) | **1** |
| slice→solve→re-embed skeleton | hand-rolled **6×** (core.py ×3, w_isdf ×1, NRHS pad ×3 counted with it) | **1 helper** |

So: one convention nominally exists and the recent fix series (083d209 → b0b0626 →
62b0365 → ed393fa) measurably converged on it — but three divisor conventions, three mask
conventions, five round-up spellings, a triplicated solve skeleton, and a 660-line dead
parallel API are all still alive. "Fewest lines" is off by ~2×; "one convention" is off
by a factor of ~3 at the seams.

---

## 2. Defect-class analysis: structural vs assumed neutrality

ROOT_CAUSE proved the defect CLASS: pad rows are zero (pad-*content* neutral) but pad
*shape* leaks — LU roundoff at the padded extent, census/window statistics over padded
mode space. Classifying every site by whether neutrality is structural or assumed:

**Structurally neutral (safe forever, 0 lines):** all bilinears/GEMMs/contractions with
zero pad rows; the fail-loud FFI divisibility asserts (`cholesky_2d.py:57`,
`cusolvermp/batched.py:130`); sigma band windows (pad bands live above `b_id_3`, outside
every window).

**Assumed-neutral, now fixed by per-consumer discipline (the ROOT_CAUSE class, patched
not closed):**
- ζ_T LU / per-q W-Dyson LU / charge Cholesky at P=1 — fixed by hand-rolled
  slice-to-logical + zero-refill at **4 separate sites** (core.py:1239-1341, 1379-1405,
  1407-1423; w_isdf.py:206-291). Correct, but a 5th solver branch must *remember* the
  idiom — the exact per-site-memory failure mode the bug came from.
- PPM census/window stats — fixed by `mu_logical_mask` threaded into
  `_prepare_sigma_state` + `mode_mask`/`n_mu_logical` into the two fits. Correct
  (post-fix 4g↔16g: census equal, bispinor eqp Δ ≤ 1.45e-7 eV), but see below.

**Still assumed / MISSING at HEAD (verified live at ed393fa):**
1. `isdf_fitting.py:982-992` — allgather branch writes the **padded** gathered
   `gflat_acc` into the **logical**-shaped `zeta_q_G` dataset. Crash whenever a μ pad
   exists on that backend (incl. the Tier-1 knob). Loud, not silent — but it is a
   consumer that had to individually remember the clip and didn't.
2. `tagged_arrays.py:63-65,110-114` + `gw_init.py:690` — restart files
   (`V_qmunu`, `G0_mu_nu`, `W0_qmunu`) are written at the **padded, P-dependent** extent
   with no `valid_shape` clip: a direct violation of the disk-logical contract, one hop
   downstream of the exact stage ROOT_CAUSE fixed. `bse_io` then recovers "n_rmu" from
   the dataset shape (:411), re-pads only if *smaller* (no trim-if-larger), using a
   *third* divisor (`lcm`). Ironic detail: `gw_init` now clips `g0_mu` to logical at :511
   and writes the same vector padded to the restart file ~170 lines later.
3. `w_isdf.py:387` `getattr(meta,'n_rmu',n_rmu)` — omission silently restores the
   padded-extent LU; `fit_ppm(n_mu_logical=None)` / `fit_gn…(mode_mask=None)` — a new
   caller omitting the kwarg silently regrows the census bug. The invariance fixes are
   opt-out-by-omission.
4. Documented accepted residuals (keep, don't hide): multi-device block-cyclic
   factorizations (distributed ζ-Cholesky, CUBLASMP W-solve) must run at the padded
   extent — measured ≤1e-7/1e-8 rel, absorbed by Tier-2 tolerances; removing them means
   never padding μ in kernels (architecture change per ROOT_CAUSE §AS-FIXED).

**Is the landed per-consumer masking the minimal design? For the solves, yes-with-one-
helper; for the census, no.**
- *Solves*: masking-at-the-fit is impossible (the pad block must be identity to keep the
  padded buffer non-singular; the extent itself is the poison). Slice-to-logical inside
  the solve is the right shape — post-fix `factor_c_q`/`solve_zeta` take
  `n_rmu_logical=` and internalize pad handling, which is correct. The defect is that the
  idiom is copy-pasted 6× instead of being one `_solve_at_logical` wrapper.
- *Census*: **fit-birth neutralization wins over the committed consumer masking.** At fit
  birth pad modes already come out `valid=False, B=0`; the ONLY live-looking value is
  `Ω = fallback_omega` (minimax_screening.py:415). ~3 lines in `fit_gn_ppm_from_wc_pair`
  (zero pad-mode Ω via the `mode_mask` it *already receives*) make `B_mask_raw = Ω>1e-14`
  pad-safe with no mask argument anywhere, deleting the whole `mu_logical_mask` consumer
  arm (−~20 lines, bit-identical, covered by the Tier-1 gate) and making every present
  and future `Omega_q`/`B_q` consumer structurally safe. The committed fix is pattern (b)
  where pattern (a) was 3 lines away — it patches instances, birth-masking closes the class.

---

## 3. CONSOLIDATION PROPOSAL — ranked by (lines removed × risk removed)

All bit-identical to current outputs except #1/#2 (which fix bugs); all covered by the
existing Tier-1 pad-flip gate (`tests/test_mu_pad_invariance.py`, 1 GPU).

| # | Change | Files | Δlines | Risk removed |
|---|--------|-------|--------|--------------|
| 1 | **Clip restart writes to logical**: `valid_shape=(nq, meta.n_rmu, meta.n_rmu)` in `write_restart_state_to_h5`/`write_w0_qmunu_to_h5`; re-pad on read via `padded_mu_extent` (pattern already at gw_init.py:465-475). Then delete `bse_io`'s lcm re-pad convention (:438-475) and fold the g0_mu hand clip (gw_init.py:511) into the same `valid_shape` idiom. | `file_io/tagged_arrays.py:63-65,110-114`, `gw_init.py:511,690`, `bse_io.py:438-475` | ~−15 | **Kills the biggest live latent bug** (P-dependent restart/BSE files = ROOT_CAUSE one hop later) + retires divisor convention #3 (lcm). Highest value-per-line in the codebase. |
| 2 | **Delete the allgather zeta special-case branch** — the backend's own `write_slab` already prefix-clips (`_slab_io_allgather.py:141-170`); the bypass is both the bug and dead weight. | `gw/isdf_fitting.py:982-992` | −10 | Kills the confirmed crash; unblocks the Tier-1 knob on the allgather/CPU-CI path. |
| 3 | **Fit-birth pad-mode neutralization**: zero pad-mode Ω in `fit_gn_ppm_from_wc_pair` (+3 lines), then delete `_prepare_sigma_state`'s `mu_logical_mask` arm (−14), the driver's inline mask (ppm_sigma.py:597-605, −5), and collapse `fit_ppm`'s duplicate outer-product mask (−4). Make the remaining fit-side mask arg **required** (kills the None-default regrow path). | `gw/minimax_screening.py:383-425`, `gw/ppm_sigma.py:132-175,597-605,207-236` | net −20 | Closes the census defect class structurally (3 mask conventions → 1 → 0 for consumers); future Σ/PPM consumers safe by construction. |
| 4 | **Delete the dead PadAxis half of `runtime/padding.py`** (`PadAxis`, `pad_array_to_mesh`, `unpad_array_from_mesh`, `pad_shape_to_mesh`, `valid_shape_from_pad_meta`, `logical_shape_from_padded`, `round_up_to_mesh_product` — zero src callers) + `tests/test_padding.py` + the stale `slab_io.py:191` "agent/padding-refactor branch" pointer. Keep `padded_mu_extent`/`extra_mu_pad`; add one plain `round_up(n, d)` and retire the 5 private spellings (`meta._round_up`, `wfn_loader._pad_to`, 3× inline core.py). | `runtime/padding.py`, `tests/test_padding.py`, `common/meta.py:8`, `file_io/wfn_loader.py:517`, `isdf/core.py:1266,1311,1346` | **~−760** | Kills the shadow second convention that future contributors will read as the contract; one round-up spelling. |
| 5 | **One `_solve_at_logical` wrapper** (slice operands to n_log → solve → embed into zeros) shared by the three `solve_zeta` branches and the w_isdf per-q LU, + one `_pad_nrhs/_trim_nrhs` pair for the triplicated NRHS pad; have the P=1 `factor_c_q` re-embed call `_identity_pad_block_diagonal` instead of re-implementing it. | `isdf/core.py:1049-1063,1239-1423,1264-1553`, `gw/w_isdf.py:206-291` | ~−40 (65→~25) | Makes "solves run on logical" a **grep-able invariant** instead of per-site discipline — the exact defect class of the eV bug; a 5th solver branch can no longer forget the slice. |
| 6 | **Delete `Meta.n_rmu_jax`/`nbnd_jax`/`n_rtot_jax`** + the `gw_init.py:200-215` refresh (zero readers, wrong divisor: `n_proc` = host count vs device count). Replace `w_isdf.py:387`'s `getattr(meta,'n_rmu',…)` soft fallback with a hard attribute read. | `common/meta.py:34-36,127-129,158-160`, `gw/gw_init.py:200-215`, `gw/w_isdf.py:387` | −12 | Retires divisor convention #2 (the wrong-divisor booby trap) and the opt-out-by-omission fallback. |
| 7 | **Bake the pad into the sym tables at construction**: emit `fwd_perm`/`L_table` already at `n_rmu_padded` (identity/zero tail) from the SymMaps accessor; delete the per-site pad-extension in `unfold_v_q` and its divergent duplicate in `_unfold_g0_ibz_to_full` (fwd-perm vs argsort). Add the too-large-direction guard. | `gw/symmetry_maps.py:282-310`, `gw/v_q_g_flat.py:605-635` | −37 | Closes the silent `promise_in_bounds` OOB exposure (TRS-bug failure shape) in ONE place; aligns with the unified-sym-action rule. |
| 8 | **Small honesty fixes**: divisibility assert at `_make_project_ri_reduce_scatter` build (nb_sigma is never rounded — currently satisfied by config luck); fix 3 doc drifts (`zeta_loader.py:124` claims disk stores padded — false; `wfn_transforms.py:1689` "no padding needed at this layer" above the pad it does; `slab_io.py:191`). | `gw/ppm_tau_kernel.py:84`, `file_io/zeta_loader.py:124`, `gw/wfn_transforms.py:1689` | +5/−3 | Each drift is the false-docstring failure mode ROOT_CAUSE already had to correct once (`_identity_pad_block_diagonal`). |
| 9 | *(Deferred, separate branch)* twin zeta reader merge (`zeta_reader`/`zeta_loader` duplicate `valid_mu` plumbing; task #8 — its stated blocker "needs padded-μ gate" is now **cleared** by the Tier-1 gate). | `file_io/zeta_{reader,loader}.py` | ~−350 | One reader, one valid_mu path. |

**Net for #1–#8: ~−900 lines (~760 dead + ~140 consolidation), ~60 → ~40 sites, 3
divisor conventions → 1, 3 mask conventions → 0 (consumer-side), 6 solve-skeleton copies
→ 1, 3 latent bugs → 0.** Roughly a 1–2 day pass, no behavior change outside the two bug
fixes, all gated.

## 3b. AS-CONSOLIDATED (2026-07-08, session D — executed on `agent/memplanner-cleanup`)

Every §3 item re-verified live at HEAD `7801d46` before acting (the only commits since
the audit tree `ed393fa` touched `.gitignore` + the Tier-2 script — all findings held).
Baseline: **249 passed / 24 skipped** (`LORRAX_NGPU=1`, full `tests/`).  The full suite
was re-run after EVERY commit; the 12 golden e2e gates + both Tier-1 pad-flip gates
stayed green (bit-identical goldens) at each step.  Net for the series:
**−553 lines** (530 insertions / 1,083 deletions across 20 files, incl. +127 for the
new restart-roundtrip gate).

| # | Verdict | Commit | Δ | Notes |
|---|---------|--------|---|-------|
| 1 | **DONE (bug fixed)** | `6c850bd` | +208/−20 (incl. +127 test) | Writers take REQUIRED `n_rmu_logical`; V/S/V0/G0/ψ(μ-axis)/W0 + `init_W0` placeholder clipped via SlabIO `valid_shape`; `load_restart_state_from_h5` re-pads via `padded_mu_extent` (padded legacy files at same P are a fixed point); both `bse_io` `lcm(gx,gy)` re-pads → `padded_mu_extent` (divisor convention #3 retired). New gate `tests/test_restart_pad_roundtrip.py`: forced-pad write → on-disk logical + bit-exact logical block → re-read at forced pad bit-identical. |
| 2 | **DONE (bug fixed)** | `d31b4b7` | −21 | Confirmed live at HEAD. Bypass deleted by UNIFYING: one SlabIO handle for both backends at STEP 4b, one `write_slab(valid_shape=…)`, one close. |
| 3 | **DONE** | `1fb160c` | −50 | Fit-birth neutralization: `fit_gn_ppm_from_wc_pair(n_mu_logical=…)` REQUIRED, pad modes born Ω=0/B=0/valid=False; `_prepare_sigma_state` mask arm + driver inline mask deleted; `fit_ppm` duplicate outer-product mask collapsed; **also deleted dead `GodbyNeedsPPM` + `extract_gn_ppm_parameters_from_Wc`** (grep-verified zero callers — the only remaining unmasked call site, enabling the required arg). 3 mask conventions → 1 (int at the fit) → 0 consumer-side. |
| 4 | **DONE** | `08f0705` | −729 | PadAxis half + `tests/test_padding.py` deleted (grep re-verified zero src callers at HEAD); suite count change exactly its 9 passed + 15 skipped. `round_up(n, d)` added as THE spelling; `meta._round_up` + `WfnLoader._pad_to` retired; stale `slab_io.py` branch pointer fixed; padding.py header now NAMES the G-axis ngkmax ragged pad as a distinct contract (§11 ask). |
| 5 | **DONE** | `c9542bf` | +58 net (−113 at sites, +80 helper w/ docs) | `solve_at_logical(solve_fn, n_log, mats, rhs, pad_axes)` in `runtime/padding.py`; all 5 sites route through it (solve_zeta ridged-LU / tri-back-solve / cusolvermp-LU, w_isdf per-q Dyson, factor_c_q P=1 Cholesky); `pad_last_axis_to` replaces the triplicated NRHS pad; factor_c_q P=1 re-embed → `_identity_pad_block_diagonal`; remaining inline round-ups → `round_up`. Line delta is positive because the helper carries the full ROOT_CAUSE contract in its docstring — the point was the grep-able invariant, not raw lines. |
| 6 | **DONE** | `bebb11d` | −18 | `Meta.n_rmu_jax/nbnd_jax/n_rtot_jax` deleted (readers: archive-only tests, updated) + gw_init refresh; divisor convention #2 (n_proc) retired. `w_isdf` `getattr(meta,'n_rmu',…)` → hard read; `w_solve_modes_test` synthetic meta carries `n_rmu`. |
| 7 | **DONE** | `620b501` | +50/−31 (guards carry the contract docs) | Pad baked into `sym_perm`/`L_table` at construction (`_resolve_ibz_q_list`, identity/zero tail); per-site pad-extensions in `unfold_v_q` + `_unfold_g0_ibz_to_full` deleted; strict BOTH-direction extent guards close the silent `promise_in_bounds` OOB exposure. Bit-identical: argsort of the identity-tailed permutation ≡ the old padded inv_perm. The fwd-perm-vs-argsort *convention* divergence in the g0 unfold is intentionally KEPT (changing it would alter non-Γ g0 file content; documented unobservable). |
| 8 | **PARTIAL** | (folded) | — | `slab_io.py:191` fixed in #4. The zeta_loader/wfn_transforms doc drifts + the nb_sigma assert were NOT tasked in the execution order and are left for a docs pass. |
| 9 | **DEFERRED** (as audited) | — | — | Twin zeta reader merge — separate branch; its "needs padded-μ gate" blocker is now cleared by Tier-1 + the roundtrip gate. |

**Post-series Tier-2 cross-P rerun (P=1 vs P=4, 4×A100, `run_tier2.sh`): both fixtures
PASS with every calibrated number unchanged** — gnppm ζ_C 2.849e-6 frob-rel / Σ_X 1e-6 eV /
windows+totals exact / off-pole eqp 69 meV; bispinor ζ_T 1.8–2.1e-7 / Σ_SX 1.85e-4 eV /
eqp0 33.7 meV.  This exercises the multi-device paths (incl. the rewritten distributed
solve branch) that the 1-GPU suite cannot; the consolidation moved nothing.

Known remaining (documented, out of scope):
- ψ/enk **band** axis in the restart file still stores the padded `b_id_4`
  (P-dependent when `nband % world_size != 0`); the μ contract is now honored, the
  band-pad convention (zero-ψ rows, sentinel energies) is a separate cleanup.
- The g0_mu zeta-file write (gw_init) keeps its rank-0 h5py hand clip (already
  logical-on-disk; converting a tiny rank-0 write to SlabIO would add lines, not
  remove them).
- CUBLASMP_FFI W-solve + multi-device block-cyclic factorizations still run at the
  padded extent (§4 do-not-touch; ≤1e-7/1e-8 rel, Tier-2 tolerances).

## 4. What NOT to touch

- **The (n_rmu, n_rmu_padded) integer pair in Meta.** Do NOT thread a pad mask or a
  PadInfo object through the pipeline — the trailing-contiguous-block invariant makes the
  logical integer strictly sufficient; a mask is more API for zero more information. The
  one consumer needing a mask builds it in one line, and after proposal #3 even that
  disappears from consumers.
- **Zero-row structural neutrality** of the bilinears — the 0-line majority. Add A's
  cheap zero-pad-row regression assert if anything, but no code at those sites.
- **SlabIO `valid_shape` + `_normalize_valid_shape` + the C++ clips** — already the ideal:
  stated once, honored mechanically, one implementation per transport by necessity.
- **Post-fix `n_rmu_logical=` threading** into `factor_c_q`/`solve_zeta`/`_get_w_solve_fn`
  — callers state the extent once, pad handling is internal. Right shape; only its
  internal duplication (proposal #5) needs work.
- **Multi-device block-cyclic padded factorizations** (distributed ζ-Cholesky, CUBLASMP
  W-solve) — mesh-divisibility is a hard layout requirement; ≤1e-7/1e-8 rel measured,
  documented, absorbed by Tier-2 tolerances. Architecture change, not cleanup.
- **Bispinor per-channel extent recomputation** via `padded_mu_extent`
  (v_q_g_flat `_pad`, `BispinorVqReader._padded_shape_LR`, gw_init transverse refresh) —
  can't read `meta.n_rmu_padded` (charge ≠ transverse extents); post-083d209 they cannot
  drift. Acceptable.
- **The G-axis `ngkmax` ragged padding** (WFN.h5 format, per-k `ngk_valid`, sentinel
  masks) — a genuinely different contract from mesh-divisibility padding; keep it, but
  *name* it as distinct in padding.py's header so nobody "unifies" it.
- **psi_G_store's band-pad populate loop** — three mechanisms in one loop looks dirty but
  is partially forced by static-shape io_callback + past-EOF + user-band-stop being
  different constraints; a consolidation helper is optional polish (~80→~40 lines), low
  priority.
