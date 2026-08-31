# Loewner conditioning result

## Verdict

The failure is the Loewner realization, not loss of digits in the small
eigensolver.  At 12 poles the incumbent one-sided pseudoinverse leaves a
retained-to-null coupling in the ordinary eigenproblem.  On the regenerated
sodium W its worst raw condition is `1.3624e13`, above the fit certificate
`1/rcond = 1.0e13`, although its backward error is small.  Row/column pencil
equilibration followed by a two-sided rank-revealing realization lowers that
worst condition to `2.7819e11` and makes the 12-pole store finalizable.  Ten
poles also improve by 62x at the worst element.  No precision change, search,
or deck dial was added.

Eight poles retain the incumbent graph exactly.  Applying equilibration there
cost 39.474 s versus 37.997 s for the incumbent full P=4 fit, so that attempted
path was not shipped.  A matched final-branch P=4 census over all 11,147,600
elements measured 36.611 s before and 35.801 s after (-2.2%, ordinary run
scatter), with bit-identical conditions, errors, valid-pole counts, and physics
outputs.

## Real-W measurements

`cond` below is the condition of the algebra producing the poles: raw L for
the preserved 8-pole solve and the retained equilibrated L subspace above
eight.  Each row covers every `(q,mu,nu)` element.

| poles | cond p50 before -> after | cond p99 before -> after | cond max before -> after |
|---:|---:|---:|---:|
| 8 | `1.830e7 -> 1.830e7` | `1.620e8 -> 1.620e8` | `3.9996e10 -> 3.9996e10` |
| 10 | `2.954e8 -> 4.387e6` | `3.400e9 -> 5.443e7` | `6.7051e11 -> 1.0841e10` |
| 12 | `3.171e9 -> 7.332e7` | `4.862e10 -> 1.320e9` | `1.3624e13 -> 2.7819e11` |

The actual 12-pole failure element is `(q,mu,nu)=(7,564,599)`.  Before:
`cond=1.36241e13 > 1.0e13`, denominator backward error `1.981e-11`, and
sample max residual `1.2340e-2`.  This is the bad-number form of the existing
reader refusal, `MPA fit failed its stored certification: ... exceeds
10000000000000.0`.  After: `cond=2.78188e11`, backward error `2.571e-17`, and
sample max residual `7.3294e-3`.

Reconstruction against every sampled W value is unchanged at useful precision:

| poles | global relative RMS before -> after | max residual / global max(W) before -> after |
|---:|---:|---:|
| 8 | `1.546907202e-3 -> 1.546907202e-3` | `1.780469743e-3 -> 1.780469743e-3` |
| 10 | `4.521998547e-4 -> 4.521998545e-4` | `5.850887261e-4 -> 5.850887988e-4` |
| 12 | `2.960855290e-4 -> 2.960855304e-4` | `5.088510980e-4 -> 5.088512382e-4` |

All returned arrays were finite and every nonzero-residue pole was in the
physical sector (`Re Omega > 0`, `Im Omega <= 0`).  Effective-pole histograms
were also unchanged: 8 poles on 11,145,478 elements and 7 on 2,122; 10 poles
on 11,131,545 and 9 on 16,055; 12 poles on 11,136,103 and 11 on 11,497.  At
the worst 12-pole element, the 11 supported poles span `Re Omega =
0.10698..6.30647 Ry`; the twelfth mode has exactly zero residue.  The largest
residue magnitude moves only `59.1246 -> 57.9427` in stored units while that
element's reconstruction improves by 41%.

## Change

`src/gw/mpa/pade_fit.py` now:

1. applies identical diagonal row and column factors to both `(L,sL)`, which
   preserves the generalized spectrum while removing sample-amplitude scale;
2. forms `diag(1/s_kept) U^H sL V` on both retained SVD axes, embedding only
   exact zero padding for the static JAX shape; and
3. takes the byte-identical incumbent one-sided path for `n_p <= 8`, the
   measured production/performance boundary.

Tests pin spectrum preservation, conditioning improvement, removal of both
null-coupling axes, and the no-extra-work eight-pole branch.  CPU: `96 passed,
1 skipped`.  Final P=4: all four ranks independently report `96 passed,
1 skipped` at commit `88fc6cc2`.

## Evidence and limitations

Evidence root:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829/codex_loewner_conditioning_20260831`.
Key artifacts are `n10_legacy_fit.h5`, `np10_cached/tmp/mpa/mpa_fit_oneshot.h5`,
`np12_cached/tmp/mpa/mpa_fit_oneshot.h5`, the four
`n12_legacy_census.rank*.npz` receipts, the matched `n8_*_census.rank*.npz`,
and `final_focused_p4.log`.

The current strict band-edge gate refuses the historical 48-band arm because
there is no 49th band with which to certify its upper boundary.  The fresh
10/12-pole W was therefore regenerated at the nearest strict-clean edge,
46 bands (63.658 meV boundary gap), rather than snapping the band count.  Its
very-wide noncrossing quadrature was runtime-generated and is marked
uncertified by the minimax service.  Thus these are valid matched measurements
of the Loewner fit on real sodium W, but not a new end-to-end sodium physics
certificate.  The historical eight-pole timing uses the original certified
sodium sample store.  `briefs/common_context.md` was absent from the supplied
tree; the inline binding common context in the task was used.
