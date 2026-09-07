# GRAM numerical audit, 2026-09-07

The checked row-map, Ry scaling, and held-row indexing are consistent. Existing
large held errors show cancellation and unstable interpolation. A short optimizer
control provides no evidence that changing the Jacobian alone cures them.
These conclusions concern the Hermitian-residue damped subcase, not unrestricted
complex residues or a successful Sigma model.

## Evidence and scope

Let `R` denote the exact directory
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox/306_gram_varpro_20260907`.

The measured fit artifacts are `R/05_fit/qXX_pPP.json` and sibling NPZ files,
from job.step **58051053.13**, source commit
`2e58e2c59bae1cc747f4edfe6122269927274628` on branch
`lane/sp-gram-2026-09-07`. Inputs are the small Grams in `R/03_gram/qXX.npz`;
their producer receipts record upstream compute provenance. No HDF5 was read
in this audit. Post-hoc calculations below ran on a login CPU and are not
GPU verification, end-to-end timing, passivity, or Sigma evidence.

In `R/05_fit/q15_p32.json`, weighted relative training error is
0.0027713755500342508 while the uniformly weighted low-line held relative error
is 47402.91197646433. The real channel design has rank 32 and condition number
17465468694.326836. Re-evaluating its saved small row map gives approximately
1.86e6 for the Euclidean norm of the complex evaluation row at
0.25+0.25i eV. Some whitened residue-coordinate norms inferred from the saved
Gram reach about 4.2e6. This case is full column rank, not an SVD rank drop.

In `R/05_fit/q00_p32.npz`, two positive poles occur at approximately
128.7689 and 128.7725 eV with widths about 8.733 eV. Their Gram-derived
whitened residue-coordinate norms are approximately 4993 and 4991. The saved
low-line held error in the sibling JSON is 0.4134549249661727. Close poles and
large residues permit cancellation on sampled lines while leaving large
responses between samples. Sparse low-line samples do not resolve every such
combination.

## Residue elimination and derivative qualification

Write each sample `W_i = H_i + i A_i`, with Hermitian `H_i` and `A_i`.
Expand both in an orthonormal real Hermitian matrix basis and stack
`Y = [H; A]`. The real channel Gram is `G = Y Y^T`, including H/A cross
blocks. Let `A = diag(sqrt(w,w)) [Re Phi; Im Phi]`. Residue elimination uses
its SVD pseudoinverse. With bandwidth `b` for dimensionless frequencies,
the physical-unit row map is `b A+ diag(sqrt(w,w))`, acting on unweighted `Y`.

For compressed Gram factor `B`, define `C=A+ B` and `R=(I-AA+)B`. At full
column rank the reduced-residual derivative is

```
dR = -(I-AA+) dA C - A+^T dA^T R.
```

Kaufman's Jacobian drops the second term. That term is orthogonal to the
residual in the objective gradient, although it changes the Gauss-Newton
curvature approximation. No normal equations are used for elimination.

The implementation discards design singular values below
`1e-13 * sigma_max`. **15 of the 116** existing q/p fits have
`design_rank < p`, counted from the 116 JSON files in `R/05_fit`.
For a mathematically full-rank matrix subjected to numerical truncation, the
formula above is not generally the derivative of the truncated-SVD projection.
Crossing the cutoff adds a rank-selection discontinuity. Their reported
Jacobian singular minima must therefore be marked **uncertified**.
The diagnostic-only update adds `jac_sigma_min_certified` and a scope string;
it changes neither objective, optimizer, initializer, nor fitted poles.
The full-rank qualification concerns the derivative formula only; it does not
certify conditioning, convergence, stable SC pole motion, or model acceptance.

The channel-Gram eigenvalue cutoff also remains explicit in
`gram_discarded_fraction`; it is not an exact untruncated data functional.
Orthonormal rational or divided-difference bases can improve arithmetic for
clustered poles, but representation alone does not remove amplification by a
physically cancelling exported pole sum.

## Login CPU checks and one continuation control

The deterministic `varpro.synthetic_check()` gave an exact-Jacobian
finite-difference relative discrepancy of 1.921815726212981e-10 and a
Kaufman-versus-exact objective-gradient discrepancy of 4.841088212148054e-16.
For an independent random real channel matrix with 7 training rows, 3 held
rows, 2 poles, and a nonconsecutive selected subset, direct held Frobenius
error was 10.600885470456292, versus 10.60088547045629 from the Gram contraction.
These are small CPU algebra checks, not real-data/GPU acceptance gates.

One login-only continuation started from `R/05_fit/q00_p16.npz`, retaining the
same input Gram, weights, bounds, and free log-pole parameters. It used the
exact reduced Jacobian, `x_scale='jac'`, and at most 100 function evaluations.
It took **0.408 s** on that login CPU; this is not a production performance
claim. Weighted relative training error moved from 0.007427150739156556 to
0.007425635542068127, while low-line uniform held error worsened from
0.09108484493454022 to 0.09144969754403381. It again stopped at the evaluation
limit. The resulting real design condition number was 157062.1040939479.
This single control does not justify a campaign-wide optimizer relaunch.
Output was reported in the agent tool transcript; no new model artifact,
cluster job, or completed experiment was created.

The VF initializer is only a seed generator. Its unconstrained relocation
can yield excess positive poles; selecting the first sorted `p` then favors
low frequencies and discards high ones. This is a possible initializer bias,
not an implemented correction or a demonstrated cure.

## Evaluation-map and subset diagnostics

For any held line the new diagnostic reports

```
|| [Re Phi_held; Im Phi_held] @ expanded_row_map ||_2.
```

This is the spectral norm from unweighted original real Htrain/Atrain data
rows to unweighted real held H/A rows, including zero columns for unused
training samples. It is a fixed-pole data-amplification diagnostic; nonlinear
motion of optimized poles is not included. The same map is reported with
physical and whitened errors because linear Coulomb congruence commutes with
the row map. It is independent of reporting quadrature. This stacked-real
operator norm is distinct from the single complex-row norm quoted above.

An explicit training rowplan selects at most 40 original training rows,
recomputes ORDER quadrature on those nodes, and expands the solved map back
to the original layout. Unselected rows have exactly zero coefficient
columns and are reported as validation, alongside separately held rows.
Available bank size and fitted sample count remain separate receipt fields.
Warm starts require identical original and held grids, ordered selected
indices, full zero-padded weights, p, and solver hash.

The smallest justified next fit control is a fixed, budgeted dense
multiheight DATA rowplan at p=16 with the existing optimizer. Inspect held
error, design rank, and evaluation-map amplification before expanding the
optimizer search. All frequencies and widths remain free; no poles are frozen.
