# 2026-08-10 — the MPA window-group farm (`perf/mpa-window-farm-2026-08-10`)

One dated file, per the convention that replaced hand-resolving three
`KNOWN_FAILURES.md` conflicts in one night. This branch adds cells and
adds no known failure; the rows below are what a reader reconciling a
census run against this branch needs.

## What this branch adds to the collection

| file | cells | what they are |
|---|---:|---|
| `tests/test_mpa_window_farm.py` | 21 | the window-group split's coverage, the balance, the manifest and its refusals |

Every one of them is pure host arithmetic and stdlib file I/O: no jax, no
GPU, no fixture. They run in the CPU census and in a zero-GPU container
step alike, which is deliberate — the failure they guard (a window group
integrated twice or never) is a bookkeeping failure and does not need a
device to exhibit.

## Measured, on this branch

`tests/test_mpa_window_farm.py` and `tests/test_mpa_pass_partials.py`
together: **41 passed in 6.81 s**, on a `-G=0` container step at
`a708e351`, `JAX_PLATFORMS=cpu`, module `lorrax_J070`. Log:
`/pscratch/sd/j/jackm/mpa_winfarm_0810/_reports/TEST_targeted.log`.

The pass-partials suite is included in the same run on purpose:
`combine_pass_partials` gained a second coverage granularity on this
branch, and the twenty cells that pin its pole-level behaviour are the
ones that would notice if the group-level path had changed it.

## Rows a census reader may need

**No new known failure.** Nothing in this branch is expected to be red.

**One pre-existing red that this branch's driver legs run into, and did
not cause.** Every production leg of the `si_mpa` n_p = 8 deck —
window-farmed, pole-farmed, and the 2026-08-09 production legs before
either existed — exits 1 at the `MPA head gate` in
`mpa_pipeline._inject_mpa_head`, with

```
the fit store's head poles do not reproduce the head samples stored
beside them — worst relative residual 4.567e-04 over 16 samples,
against a tolerance of 1.0e-06
```

on the `as_shipped` head set of
`/pscratch/sd/j/jackm/mpa_wcprod_0809/stores/mpa_fit_np8_wc.h5`. The
refusal's own text names the cause and it is cause (3) in its list: with
`2*n_p` samples and `n_p` poles the head fit interpolates, and the
mandatory residue refit after the time-ordering guard reflected two of the
eight poles leaves exactly this 4.6e-4. It is a property of that stored
head set and of the causality guard, not of any code on this branch.

**Why it does not invalidate a leg.** The gate is step 3 of the pipeline
and the partial Σ_c cube is written in step 2, so a leg that exits 1 here
has already produced the artifact it exists to produce — which is what
`[lx] nonzero exit — judge by artifacts, not the code` is telling a
reader to do. Every cube used in this lane's gates was written before
this refusal fired, and the refusal fires identically on both arms of the
comparison, so it cannot be the source of a difference between them.

It is registered here rather than fixed here because the head is the
store's, the tolerance is the head reader's, and both are outside this
lane's boundary (the fit-store reader is the wedge-unfold lane's
territory).
