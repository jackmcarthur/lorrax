# Padding-cleanliness audit — Scope 3: Sigma / PPM consumers (census, windows, projection)

Tree: `sources/lorrax_D` @ `agent/memplanner-cleanup` (62b0365 + one uncommitted edit).
Judged against the ideal: pad born once at ingest; (logical, padded) carried only in `Meta`;
consumers either structurally neutral or using ONE canonical mask/slice helper.

## 0. Commit state of the "in-flight" work

The prompt's "new in-flight `mu_logical_mask` arg" is in fact **already committed**
(62b0365 "PPM mode census + window statistics on LOGICAL modes only"; the μ-slice solves are
b0b0626; the knob is 083d209). The only **uncommitted** edit in the tree is
`src/gw/gw_init.py:511-519`: the `g0_mu` h5 dataset is now clipped to `meta.n_rmu` before
`create_dataset` (+6 lines) — enforcing the "disk stores logical extent" contract that the
rest of `compute_V_q` already followed. Untracked: `tests/test_mu_pad_invariance.py`,
`tests/multi_device/`, `.padprobe/`, `.tier2_cross_p/`.

## 1. Site inventory

| # | Site | Class | Pad-specific lines (exec / total) |
|---|------|-------|------|
| 1 | `ppm_sigma._prepare_sigma_state` (`:132,148-157,173-175`) — `mu_logical_mask` arg, `mode_logical = mask⊗mask`, `B_mask_raw &= mode_logical`; drives `n_total_modes`, `n_invalid`, `B_mask` (hence window stats + the `2ry` contraction mask) | **AD-HOC per-consumer masking** (committed) | 4 / 14 |
| 2 | `ppm_sigma` driver `compute_sigma_c_ppm_omega_grid:597-605` — builds `jnp.arange(Ω.shape[-1]) < meta.n_rmu` inline | **AD-HOC mask construction** #1 | 2 / 5 |
| 3 | `ppm_sigma.fit_ppm:207,218-221,227-230,236` — `n_mu_logical:int` param, builds `mu_log` + outer-product `mode_mask` (same mask as #2, rebuilt) | **AD-HOC mask construction** #2 | 5 / 10 |
| 4 | `minimax_screening.fit_gn_ppm_from_wc_pair:367,383-389,417-425` — `mode_mask` param; masks only the **scalar** `unfulfilled_fraction`; fitted tensors keep fallback Ω on pad modes | **AD-HOC (fit-side, half-done)** — the missed birth-masking opportunity, see §3 | 8 / 17 |
| 5 | `gw_init.compute_V_q` g0_mu clip (UNCOMMITTED) | ad-hoc slice, but follows the one disk=logical convention | 2 / 6 |
| 6 | `ppm_windows` (`_masked_stats_device`, `_build_windows_for_branch`) | **structurally neutral** — consumes the already-masked `B_mask`; zero pad-aware lines | 0 |
| 7 | `ppm_tau_kernel._build_W_t_q` — W(τ)=B·phase·mask; pad B_q = −½·Wc0·Ω = 0 exactly | **structurally neutral** (bilinear-with-zero-rows class) | 0 |
| 8 | `ppm_tau_kernel._make_project_ri_reduce_scatter:84-85` — "Requires m % p_x == 0 and n % p_y == 0. Padding at the caller is the cleanest place (TODO when we hit that)" | **MISSING guard** — see §4 | 0 (doc only) |
| 9 | `cohsex_sigma` (Σ_SX/COH/X/V_H, static Σ_mn projection) | **structurally neutral** — μ-bilinears see zero ψ pad rows; the sigma band window (b1,b3) contains no pad bands (band pads live only in [b_id_4_user, b_id_4), above b_id_3) | 0 |
| 10 | `ppm_accumulators` (τ accumulator, sinks) | **structurally neutral** — pure band-space, pad-free by #9's argument | 0 |

**Totals for this scope:** 10 pad-relevant sites; **5 carry pad-specific code** (~52 lines
total, of which ~21 executable, ~31 comments/docstrings); **4 ad-hoc** (sites 1-4, three of
which construct the *same* logical-μ mask under **three different parameter conventions**:
`n_mu_logical:int` → `mode_mask:(μ,ν) Array` → `mu_logical_mask:(μ,) Array`); 4 structurally
neutral with zero pad lines; 1 missing guard.

The upstream anchor is clean and singular: `Meta.n_rmu` / `n_rmu_padded`
(`common/meta.py:24,40-45`, born via `runtime/padding.padded_mu_extent` — genuinely one
source of truth for the *extent*). But `runtime/padding.py` exposes shape/array pad/unpad
helpers only — **there is no canonical logical-mask helper**, which is exactly why sites 2
and 3 hand-roll `arange(padded) < logical` and site 1 rebuilds the outer product that site 3
already built.

## 2. What is genuinely clean

- One extent authority (`padded_mu_extent` → `Meta.n_rmu_padded`; band twin `b_id_4` /
  `b_id_4_user`), one disk convention (logical extent; the uncommitted g0_mu clip closes the
  last hole in this scope).
- The mask-once-then-consume flow *below* `_prepare_sigma_state` is right: `ppm_windows`,
  `ppm_tau_kernel`, `ppm_accumulators` contain **zero** pad code and are safe by
  construction. The defect class the lead worries about is confined to the fit→state seam.
- `cohsex_sigma` needs (and has) nothing: sigma-window bands can never be pad bands, and μ
  pads are zero rows through every bilinear.

## 3. The judgment call: mask at the consumer vs mask at the fit — **mask at the fit wins**

Root fact: at fit birth (`fit_gn_ppm_from_wc_pair`), pad modes (Wc0 = Wc_probe = 0) already
come out `valid=False` and `B=0`. The **only** non-neutral birth value is
`Ω = fallback_omega` (2 Ry) — `omega_vals = jnp.where(good, sqrt, fallback)` at `:415` hands
pad modes a *live-looking* pole frequency. Everything downstream keys "is this a mode" on
`Ω > 1e-14` (`B_mask_raw`), so that single fallback assignment is what forced the entire
committed consumer-side masking arm (sites 1-2) into existence.

Fix at birth (~3 lines in `fit_gn_ppm_from_wc_pair`, which **already receives**
`mode_mask`):

```python
if mode_mask is not None:
    m = jnp.broadcast_to(...)             # already built for `unfulfilled`
    omega_vals = jnp.where(m, omega_vals, 0.0)   # pad poles born DEAD, not fallback
    good = good & m                                # (tidiness; already False on pads)
```

Then `B_mask_raw = Ω > 1e-14` excludes pad modes with **no mask argument at all**, and the
following delete cleanly (bit-identical outputs — pad modes are today excluded by
`mode_logical` at exactly the same points):

- `_prepare_sigma_state`: `mu_logical_mask` arg, `mu_log`/`mode_logical`/`& mode_logical`,
  10-line docstring ¶ (−14 lines);
- driver `:597-605` mask construction (−5 lines);
- site 3's outer-product duplicate collapses into passing `mode_mask` (or just
  `n_mu_logical`) to the fit once (−4 lines).

Net ≈ **−20 lines, sites 1-2 deleted outright**, one convention left (mask lives where the
tensors are born), and — the real prize — **every future consumer of `Omega_q`/`B_q` is
structurally safe**: pad rows of all three fit outputs become exact zeros/False, matching
the disk contract (a padded re-read also yields zero pad rows), so the "each consumer must
remember to mask" defect class is closed for this chain, not patched per-site. The committed
fix is correct but is the (b)-pattern where the (a)-pattern was ~3 lines away.

Residual after that refactor: one `arange < logical` construction (fit caller) — worth
promoting to `runtime/padding.logical_mask(padded, logical)` (~5 lines) so the ζ/V_q scopes
can share it; and the `mode_mask=None` default on both fit functions, which silently
reproduces the historical pad-dependent census for any new caller — make the mask (or
`n_mu_logical`) **required**, or default it from the array shape vs a required logical count.

## 4. Latent-bug candidates (this scope)

1. **`ppm_tau_kernel` reduce-scatter divisibility unenforced** (`:84-85`): `nb_sigma =
   b_id_3 − b_id_0` is *never* rounded (only `b_id_4` is), so `m % p_x == 0` /
   `n % p_y == 0` is satisfied by config luck. Failure mode is a loud shard_map/psum_scatter
   shape error, not silent corruption — but it's an acknowledged TODO with no check and no
   caller-side band pad. Cheapest honest fix: assert with a message naming the remedy at
   `_make_project_ri_reduce_scatter` build time.
2. **`fit_ppm(n_mu_logical=None)` / `fit_gn_ppm_from_wc_pair(mode_mask=None)` unsafe
   defaults**: a future caller that omits the kwarg silently regrows the census bug (the
   defect class survives the committed fix; §3's birth-masking removes it).
3. **`Meta.n_rmu_jax`** (`meta.py:36-39`): legacy `round_up(n_rmu, n_proc)` — documented as
   the *wrong divisor*, kept for back-compat. Any surviving consumer is a pad-extent bug
   waiting for a multi-device-per-process topology. (Outside this scope's files, flagged
   because it is a second, contradicting padded-extent convention inside the ONE meta.)
4. `Ω > 1e-14` doubles as "is a mode" and numerical guard; after birth-masking it becomes
   unambiguous ("is a live mode"), before it, its meaning depends on who masked what.

## 5. Verdict for Scope 3

**Not maximally clean — B grade.** The extent authority and the neutral-by-construction
consumers (windows/τ-kernel/accumulators/cohsex: 0 pad lines) meet the ideal. But the
committed census fix put per-consumer masking at `_prepare_sigma_state` when a ~3-line
change at the fit (pad poles born `Ω=0, B=0, valid=False`) would have deleted both consumer
sites, collapsed three mask conventions into one, and made all present and future
Σ/PPM consumers structurally safe. Concretely achievable: 5 pad-code sites → 2 (fit +
g0-disk-clip), ~52 → ~25 lines, plus a shared `logical_mask` helper in `runtime/padding.py`
and a divisibility assert at the reduce-scatter factory. All changes bit-identical to
current outputs and covered by the existing Tier-1 pad-flip gate.
