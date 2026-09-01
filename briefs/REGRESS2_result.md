# REGRESS2 — merged one-shot accuracy bisection

## Achieved numbers

The first commit above 0.3 meV is **`6fdad2c3`**, the merge of
`feat/on-demand-only-2026-08-31` (executable change `69b888e0`, “fit every
delivered window on demand”).  The same cold Na 0--5 eV deck and
`control_panes_24b` comparison measured:

| tree | windows / pairs | plan (s) | max spend | Sigma_c max / RMS (meV) |
|---|---:|---:|---:|---:|
| consolidated `0bb0a6ba` | 6 / 115 | 51.678 | 0.688527 | **0.195910 / 0.008835** |
| + null-family `dd5a1227` | 6 / 115 | 52.547 | 0.686865 | **0.195911 / 0.008835** |
| + on-demand `6fdad2c3` | 6 / 77 | 61.568 | 0.734493 | **0.760271 / 0.023515** |
| + Hackbusch `da274386` | 6 / 77 | 63.033 | 0.734493 | **0.760277 / 0.023515** |

Hackbusch versus on-demand differs by only **0.00000833 meV max**.  The later
marginal-cost merge `85393189` changes only `briefs/MARGINAL_result.md`, not an
executable file; the already completed merged-tree arm measured
**0.760283 / 0.023515 meV**.  Thus usable-rank, pointwise-budget and
marginal-cost changes are acquitted by the deck bisection.

## What changed and the forced trade

On-demand-only replaced the six node counts **66, 9, 11, 12, 9, 8** with
**26, 9, 14, 10, 9, 9**.  The dominant crossing rule lost 40 nodes and the
total lost 38 pairs; its accepted refined residual rose from
**2.16e-5 to 8.16e-4**.  That saves 33.0% of execution pairs but raises the
consumer maximum by **0.564360 meV** and planning by **9.890 s**, so the
77-pair on-demand plan is not an acceptable accuracy/performance trade.

Reverting to shipped tables would violate the owner's on-demand-only ruling,
so no such “fix” is proposed.  The measured safe arm costs 115 pairs; under
the ruling, the live fitter/selector must spend more rank (especially on the
crossing window) until a consumer-level guard is met.  The minimum safe
all-on-demand pair count was not established in this sprint.

## Planner-contract finding

The present contract is **not predictive of delivered Sigma**.  Its artifact
explicitly records `physical_relative_sigma_error_claimed=False` and
`exchange_rate_calibration=not_calibrated`.  More decisively, this bad arm
passes at spend **0.734493**, while the known-good 74-pair pointwise artifact
passes at the larger spend **0.765477** and delivers **0.235789 meV**.  No
monotone `spend_frac` threshold can distinguish those outcomes.  It can become
predictive only through a consumer-level Sigma bound or measured exchange-rate
calibration; tightening the current inverse-gap-envelope threshold alone is
not justified by these data.

Evidence: `/pscratch/sd/j/jackm/wt_regress2_2026-08-31/evidence/regress2`.
All accepted arms are cold P=4/BFC@0.85 with source SHA printed in-leg.  The
initial concurrent batch is excluded: `lx batch` raced three P=4 legs onto
`nid001008`, producing 6.60/2.93 GiB OOMs; the reported arms were rerun one at
a time with one P=4 leg on the node.  The prescribed CPU gate passed
**134/134 in 98.70 s**.  Branch:
`fix/merged-accuracy-2026-09-01`.
