# 2026-08-10 — the jax-native MPA pass path (Stage A)

Branch `refactor/mpa-jax-native-2026-08-10`, cut from `0a96c6ca`. Evidence
`/pscratch/sd/j/jackm/mpa_jaxnative_0810/`, registered in
`/pscratch/sd/j/jackm/EVIDENCE_MANIFEST.md`. Allocator BFC@0.85 throughout.
Handed to the landing marshal for `SOLVE_LANDING_PLAN.md` rung 4; this lane
merged nothing.

## What landed here that a future reader needs

**The gnppm digest gate was red on the integration tip, and this branch
re-anchored it.** `fix/mpa-pass-p4-2026-08-10` (`1fb7c75f`, landed as
`7c497a7f`) edits `ppm_windows.py` — a KERNEL in
`test_mpa_sigma_costs_gnppm_nothing.py`'s list — and landed without the
re-anchor that file's own procedure requires. Both of its cells therefore
failed at `0a96c6ca` and on every branch cut from it, on a change the
failing branch had not made. The re-anchor is checked, not rubber-stamped:
five of six kernels and both seams are byte-identical to `965d7beb`, and the
sixth's diff is exhaustively `_already_on_host`, which dispatches the
identical `_to_host_np` call for every operand that is not host numpy —
which is every operand the two-point driver has, because it holds them on
the mesh. `BASE_SHA` moves to `0a96c6ca` with the digests.

## Two things that are not defects but will look like them

**1. Three of five parametrized cells "fail" under
`scripts/direct_cells.py`.** That runner — inherited from the P>1 lane and
the only way to reach a ≥4-device cell, since pytest pins one GPU per
non-controller process — calls test functions with no arguments and cannot
expand `@pytest.mark.parametrize`. Every such cell raises `TypeError:
missing N required positional arguments` and is reported as FAIL. Affected
in `_reports/p4_direct_jax.log`:
`test_mpa_jax_native::test_the_device_selector_is_the_dense_mask_byte_for_byte`,
`test_mpa_jax_native::test_the_device_width_sort_is_the_host_permutation_exactly`,
`test_mpa_chi0_resolvent_jax::test_the_fused_block_step_is_byte_identical_to_the_unfused_walk`.
All three are green under pytest on a real GPU
(`_reports/gpu_cells.log`, 62 passed). **The runner needs a parametrize
expander before the next lane trusts its FAIL count**; until then, read its
failures against the pytest run.

**2. The width sort's device path is thresholded, so its benefit is
pole-dependent and two poles show none.** `_sorted_by_width` routes to the
device only above `_DEVICE_SORT_MIN_MODES = 2**20`. On the n_p = 8
production deck the census-leg wall falls from a median of 109 s to 50 s,
but poles 4 and 7 measure 46/46 s and 106/105 s — no change. The answer is
identical on either route by construction (a stable sort of a fixed key
vector is a uniquely determined permutation), so this is a performance
row and not a correctness one. Whoever re-measures at ANCHOR-2 should say
whether those two poles' Laplace buckets genuinely sit under the threshold
or whether the threshold is mis-sized.

## The environment, live rather than historical

Three legs died on `lx_pool.ProbeFailed: scontrol did not answer in 3
attempts of 15 s`, with `scontrol ping` reporting `Slurmctld(primary) at
slurmctld is DOWN` — §6's L6a of `MPA_16GPU_PLAN.md`, recurring on
2026-08-10 afternoon. `_reports/cen_cb_p2.log`, `_reports/cen_cbq_p1.log`,
`_reports/L_cb_p6.log`. All three relaunched and landed. The base-arm
census walls in the evidence README carry that contention (one 390 s
outlier, one 46 s outlier, six at 100–111 s); medians are what is quoted.

## Owed

The ANCHOR-2 re-cut of every byte comparison; gate (c) the 16-GPU farm
A/B; gate (d) P=1 vs P=4 on the pass; gate (e) as a cold/warm
`ISDF_JAX_CACHE_DIR` entry-count pair (the leg's own `xla_compiles=0`
summary reads zero on BOTH arms with `cache_probes=0` and is not a
discriminating instrument on this path); and gate (f) at full census on the
cluster. All are itemised in the addendum appended to
`~/lorrax_service_phase/MPA_PERF_HANDOFF.md`.
