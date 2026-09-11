# ACONJ finding — correction prepared, verification pending

Heavy lane on `lane/sp-aconj-2026-09-11`, base `810c260b`.

The planted operator is inside the supported positive-residue even-s class.
It uses real B and positive diagonal T, with directions selected by SVD.
The missing realness is therefore real for this subset. Run35/19, job
58151651.5, measures same-Q Im/Fro 0.0445840021 and conjugated-Q
2.7853388122e-16. The pencil matches explicit projection: the lost structure
is in the selected space, not the divided differences. No evidence establishes
that q averaging saves production Sigma.

The stronger statement that scalar TRS requires entrywise real fixed-q W is
false. W(s*)=W(s)† permits complex Hermitian residues; real T alone does not
make B real. Later Run35/57/04, job 58166523.5, already measured the relevant
control: left singular partners restore the real case (2.5934103189e-16),
while the exact complex control has Im/Fro 0.2390087042. Conjugating Q alone
is coordinate dependent and does not select the adjoint support's right space.

For W=U sigma V†, choose WQ=U sigma at the adjoint support. This spans U;
the existing per-column Gram equilibration cancels the positive singular
scales. Selection runs once; line width stays 2r total, with unchanged
column_extent and maximum pencil side. Retained K can change because the
space changes. The packer references the existing WQ panel after movement,
so it transports only one Q block per pair, as before. No extra dense sample,
SVD, sampling support, or line-direction transport is introduced. Actual
per-rank peak and band costs still require measurements.

The new model_reciprocity gate checks transpose symmetry of evaluated held
W/dW where the corresponding reference is transpose symmetric. For a real
residue response this is equivalent to realness on negative s. Applicability
and both defects are recorded per sample; generic complex fixed-q data are
not forced real. This is a sampled model check, not an all-frequency or
space-group covariance theorem. The planted old-policy red twin must fail.
Degenerate within-support basis rotations and ill-conditioned Gram truncation
can introduce further covariance loss; the gate does not repair such loss.

Prepared source has only passed Python syntax checking. Required P4 planted,
Si P4 and Na P16 controls/repeats/candidates, all-parent gates, every-rank peak,
and Sigma <=2 meV are outstanding. SP-M2 allocation 58190627 is pending.
Run/evidence root:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox/360_aconj_20260911`.
