# 2026-08-10 — the MPA plan store (`perf/mpa-plan-once-2026-08-10`)

One dated file, per the convention that replaced hand-resolving three
`KNOWN_FAILURES.md` conflicts in one night. This branch adds cells and
adds no known failure; the rows below are what a reader reconciling a
census run against this branch needs.

## What this branch adds to the collection

| file | cells | what they are |
|---|---:|---|
| `tests/test_mpa_plan_store.py` | 23 | the plan artifact's round trip, its content address, and the four refusals that make it safe |

Like the window farm's cells they are pure host arithmetic and file I/O —
no jax, no GPU, no fixture — because what they guard is a bookkeeping
failure and does not need a device to exhibit. The round-trip cells do
call the real planner on a small field, so they exercise the real
`WindowGroup` and `_SigmaWindow` structure rather than a stand-in.

Two of them are red twins for failures that did not exist before this
branch and now can: a plan that is absent (refused by name, and the
message distinguishes "no plan for this branch" from "a plan at a
different address"), and a `_SigmaWindow` or `WindowGroup` field the plan
store does not know how to write (refused at the write, because a dropped
field is a window that integrates to a smooth, finite, wrong Σ).

One of them is the landing plan's **B7** obligation at rung 1: perturbing
a single pole of a slab by one ulp changes the plan's address, so a plan
built against a refitted store cannot be found by a leg running against
the new one. It is parametrized over every planner input — the pole
field, the widths, the live mask, `E_A`, the ω half-grid, the source sha,
the store path, the pole index, the branch, and three quadrature scalars
— and each parametrization asserts the address moves when that input
does.

## Measured, on this branch

`tests/test_mpa_plan_store.py`, `tests/test_mpa_window_farm.py` and
`tests/test_mpa_pass_partials.py` together: **64 passed in 2.26 s** at
`04e4d8a4`, `JAX_PLATFORMS=cpu`. The window-farm and pass-partials
suites are included on purpose: this branch refactors
`window_farm.select_branch_groups` onto a shared `check_partition`, so
the cells that pin the farm's own refusals are the ones that would notice
if the plan-loading route had changed them.

## Rows a census reader may need

**No new known failure.** Nothing in this branch is expected to be red.

**The same pre-existing red the window farm registered.** Every
production leg of the `si_mpa` n_p = 8 deck exits 1 at the `MPA head
gate` in `mpa_pipeline._inject_mpa_head`, on the `as_shipped` head set of
`/pscratch/sd/j/jackm/mpa_wcprod_0809/stores/mpa_fit_np8_wc.h5`, with a
worst relative residual of 4.567e-04 against a 1e-6 tolerance. The full
account is in `2026-08-10-mpa-window-farm.md` and nothing about it has
changed: the gate is step 3 and the partial cube is written in step 2, so
a leg that exits 1 there has already produced the artifact it exists to
produce, and the refusal fires identically on both arms of this lane's
A/B, so it cannot be the source of a difference between them.

Both arms of this lane's comparison are farmed legs on that deck, so both
carry it, which is the point: **it is a property of the store, and a lane
measuring a representation change is not the lane that fixes a stored
head.**
