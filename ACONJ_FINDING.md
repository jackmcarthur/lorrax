# ACONJ — adjoint-partner conjugate ports, measured on GPU

Branch `lane/sp-aconj-2026-09-11`, numerical tip `3af53203`, base `810c260b`.
Pool 58209192. Full evidence:
`reports/shared_pole_push_2026-09-07/aconj/report.md` in the sandbox repo;
run root `runs/frequency_integration_sandbox/360_aconj_20260911`.

## What the correction is

`_direction_states` (`src/gw/shared_pole_constructor.py:534`) used to hand a
line support's conjugate state the identical block `Q_a`. Because
`W(s*) = W(s)+` for a Hermitian-residue response, the right singular space at
the adjoint support is the left space at `s_a`, which the already-formed output
panel `W_a Q_a = U sigma` spans. The conjugate state now takes that panel
(`:583`). One decomposition per line support, `2r` columns per pair at the same
`column_extent`, exactly-zero padded tail, and still one transported Q block
per pair: the packer recognises the alias by object identity
(`src/gw/shared_pole_local.py:45-53`) and addresses the partner through the
already-moved output panel with a negative column code (`:78-86`, decoded at
`:188-191`). Gram equilibration divides out the positive `sigma` scales.

`shared_pole_reciprocity` (`:435`) is the new discriminating predicate: gate
row `src/gw/shared_pole_recipe.py:59`, evaluated with the other model gates in
`local_model_checks`, consumed at `shared_pole_constructor.py:1052-1053`. At a
held sample it compares transpose defects of the model and of the reference,
declares the sample applicable only where the reference is transpose-symmetric
to 1e-12 relative, and then requires the model within 1e-10. No new
user-visible dial.

## Measured

**Planted counterexample flips, job 58209192.6**
(`02_si_p4/candidate/closure.json`; identical at 58209192.0). Real-residue
operator through the production selector/packer/reducer: `Im/Fro`
**5.562715712925523e-16** with adjoint partners against
**4.4584002105367296e-02** with the former same-Q policy, model versus explicit
latent projection 4.633277036072342e-15, pencil side 10 in both, 4 transported
Q columns of 8 uncompressed. The complex-residue control stays complex
(`Im/Fro` 0.4957309860279201): nothing is forced real.

**The gate separates the two policies at one threshold, in that same job.**
Adjoint partners: applicable true, model defect 1.1138964802517224e-15, PASS.
Former same-Q: applicable true, same reference (6.565503695894222e-17), model
defect 8.916800421073459e-02, **FAIL**. Complex control: not applicable in
either policy.

**Si P4 production, against a bit-identical control repeat** (0.000 ueV on all
272 QP rows, 0.0 meV on every Sigma dataset, byte-equal peaks): every pole
positive, passivity PASS, all rules certifying, `retained_subspace_moments`
5.511e-13 -> 4.002e-13, held-W defect better on six of eight parents (worst
parent 3.735e-03 -> 3.284e-03). The model's own fixed-q transpose defect falls
**37x to 567x** at all twelve held points of the three parents whose reference
has that symmetry, and is unchanged to nine significant figures at the other
five (`reciprocity_probe_si_p4.json`, control 58209192.13 versus candidate
58209192.6).

**Sigma is not neutral.** Max abs Sigma_c over all (k,m,n,omega) 50.519 meV;
conditioned eqp diagonals 5.037 meV; QP energies max 5.862 meV over 34 bands,
median 0.487 meV, but **0.011 meV over the four frontier bands** and
<= 0.058 meV over bands 2-8. Twenty of 272 QP rows exceed 2 meV, all at band
>= 13 (above ~20 eV). Attribution is exact because the control repeat is
bit-identical.

**Cost.** Kmax and the compact model payload do not move at all (2010,
94 807 744 B); sum K rises 10980 -> 11031 (+0.46 %) because the Gram cut
retains more of a different span; staging/peak model payload +188 672 B
(+0.11 %). Every-rank device peak is unchanged on ranks 1-3 and rises
**512 bytes** on rank 0. `spole.direction_selection` does not slow
(6.0935 -> 5.9518 s wall); the only band that grows is `spole.passivity_held`,
+0.563 s, which is where the new predicate runs. The line port width is
unchanged, so K's line contribution per support is unchanged.

`GATE_HASH` changes because the gate table gained a row, so a shared-pole bank
or restart member stored at `810c260b` is refused by
`shared_pole_constructor.py:722-731` and must be rebuilt. Loud, not silent.

## Limits

Two strict checks fail as written and are reported, not adjusted:
`every_rank_peak_nonincrease` (+512 B on rank 0) and `sigma_within_2mev` on the
raw Sigma_c matrix. **The gate is inert on Si production** — 0 of 64 samples
applicable, because the reference's own transpose defect is 1.2e-11 to 1.3e-9
at the symmetric parents and 1.27 to 1.41 at the rest; the production evidence
is the post-hoc probe, not the gate. The predicate is a sampled model check,
not an all-frequency or space-group covariance statement, and it does not
repair covariance lost to degenerate within-support rotations or
ill-conditioned Gram truncation. Only the `local` layout is exercised. Na P16
is still running at the time of writing; see the report. No claim about the Si
SC failure, about q averaging in Sigma, or that the candidate's Sigma is
better than the control's — only that it differs by the amounts above. Nothing
is on `origin/main`.
