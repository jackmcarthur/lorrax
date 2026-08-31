# Split-on-refusal result — bounded fallback, real arm still refuses

## Numbers first

The exact `-26:-24,-5:+5 eV` P=4/BFC@0.85 arm did **not plan**.  Its first
fit wave reproduced **15 rows: 14 served, 1 refused**.  The refusing
`omega<E_F val:resonant` window measured `A/gamma=204.3667`, residual
**8.531982e-3**, and kappa_p99 **93625.7**.  The implemented fallback then
bisected omega, but the worst child retained exactly `A/gamma=204.3667` and
both children refused:

| split | A/gamma | residual | kappa_p99 | family |
|---|---:|---:|---:|---|
| whole | 204.3667 | 8.531982e-3 | 9.36257e4 | none |
| omega 1/2 | 190.5033 | 7.483235e-2 | 2.37804e5 | none |
| omega 2/2 | 204.3667 | 3.301138e2 | 1.41030e7 | none |

A second omega bisection produced four more refusals: residuals
**0.8384, 27.9997, 55.9667, 115.542** and kappa_p99 from **6.11e4 to
9.50e6**.  I stopped at **195 s** of Sigma/planner work because the measured
axis did not reduce the controlling radius.  Achieved output is therefore
**0 selected windows / 0 `(window,tau)` pairs**; there is no honest rank or
Sigma artifact to report.  This independently confirms that subdivision is
not cost-neutral and can make conditioning much worse.

## What landed

The planner now preserves already-served candidates and replaces only a
refused crossing rectangle with exact disjoint product children.  Recovery is
bounded at depth 10 and 128 total pieces.  State splitting is capped at one
bisection (`k=2`) because the frozen Na result found `k=3/4` cost more than 5x
the whole rule and failed the noise gate.  The live result above added a second
guard: if omega bisection does not reduce worst-child `A/gamma`, recovery
spends that single state bisection immediately instead of repeating the
measured-bad omega operation.

The final state-dominated routing is a measured correction made after the P=4
arm and was not rerun on the real deck within this sprint.  Thus the requested
claim “the two-patch deck plans” is **not achieved**.  The one-shot 6-window /
115-pair numerical control was also not rerun; no bit-identity claim is made.

## Verification and provenance

- Final CPU gate: **136 passed** in 90.79 s (the base's 134 plus two new
  recovery tests).  The two focused recovery tests separately passed in
  2.18 s.
- Real arm: source `da5e470356a42220e4d7d10c0749bc66725ffef2`, four ranks /
  four A100s, BFC@0.85, actual attached JID **57789884** (the brief's earlier
  JID was no longer the pool selected by `lx`).
- Evidence:
  `/pscratch/sd/j/jackm/wt_split_on_refusal_2026-08-31/tmp/two_sigma_windows_p4_20260831/two_windows_p4.log`.
- Branch `feat/split-on-refusal-2026-08-31` pushed through `fd34e938`.
