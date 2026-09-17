# CHI3 metal sample and scan deletion — 2026-09-17

**Heavy lane. Delivery is blocked for landing: Na fails the requested 2 meV
eqp gate, including with heads off.** The other requested comparisons pass.
No tolerance, sample count, head implementation or fitting algorithm was changed
to obtain acceptance. This report delivers the measured failure for owner review.

Worktree: `/pscratch/sd/j/jackm/wt_sp_chi3`.
Branch: `cleanup/mpa-metal-sample-no-pair-scan-2026-09-17`.
Numerical candidate: `7a7146dad227977f37ca0eb982f2fb6202344621`.
Control: `fb64a811955873d753f6c8c90175f8951ab27877`.
The delivery adds only this report after the tested candidate.

Evidence directory:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox/461_chi3_20260917/metal_sample_finish/`.
Own pool `lx-alloc-jackm-CHI3-finish`, job **58476319**, was cancelled after
all steps completed. GPU runs used `cuda_async` with memory fraction 0.85.
Each GPU rank owned one GPU. CPU tests used four emulated devices and eight
pytest workers. No performance or utilization claim is made.

| Gate | Geometry | Job.step | Result |
|---|---|---|---|
| Si MPA | P4, both arms in one step | 58476319.0; comparisons .9/.10/.13 | PASS: every dataset bitwise, including Sigma 12/12; text identical apart from generated-at stamps |
| Si shared-pole | P4 | 58476319.0; .9/.10/.13 | PASS: numerical payload bitwise, including Sigma 12/12; provenance differences detailed below |
| MoS2 P3 shared-pole | P4 | 58476319.0; .9/.10/.13 | PASS: numerical payload bitwise, including Sigma 12/12; provenance differences detailed below |
| CrI3 G2 | P16, both arms in one step | 58476319.11; comparison .13 | PASS on both; 108 receipt fields, only two timings and the source-tree path differ |
| Na FD MPA, heads off | P4 | 58476319.0; Sigma .12, eqp .13 | FAIL: supplied eqp_ab returns 1 at 2000 micro-eV tolerance; 2494 compared rows |
| Full CPU failure set | CPU | control 58476319.1, candidate .5, comparison .8 | PASS: exactly the same 280 failing/error test names; no new or missing failures |
| Candidate fractional chi known-answer check | P4, nonzero complex frequencies | 58476319.0 | PASS receipt; control receipt failure described below |

The full CPU suite is not green. Control collected 6852 cases and candidate
6833, with 6244 and 6225 passes respectively. Both report 228 failures,
52 errors, 267 skips, 61 expected failures and 14 passing subtests. The net
19 removed cases accompany the scan/origin-shift test deletion. Individual
outcomes are retained in `evidence/suite_{control,candidate}.junit.xml`;
the supplied comparator produced `evidence/suite_failure_sets.json`.

The Si and MoS2 shared-pole byte comparisons differ only in the model's
`header_json`/`final_commit` and the ISDF bundle's `shared_pole_member` digest.
Every header/member leaf was reviewed: 34/202984 differ on Si, 22/72950 on
MoS2. They are Coulomb file path/hash, compilation and Coulomb-square-root
timings, and the resulting header/member digests. Coulomb arrays themselves
are 2/2 bitwise. Raw full-artifact FAIL labels on those metadata records remain
in `evidence/verdict.json`; the reviewed numerical interpretation is recorded
separately in `evidence/landing_assessment.json`. No physics difference is exempted.

Fresh Na diagonal Sigma differences, extracted by the supplied
`tools/compare_sigma_mnk.py`, are:

| Sigma quantity | RMS difference (meV) | Maximum absolute difference (meV) |
|---|---:|---:|
| Re Sigma, all stored frequencies | 46.9 | 3687.0 |
| Im Sigma, all stored frequencies | 43.0 | 2884.9 |
| Complex Sigma, all stored frequencies | 63.6 | 3693.0 |
| Re Sigma at omega = 0 | 26.2 | 73.3 |

The Na inputs are from `runs/Na/20_fd_scalar_2026-09-17`; both arms use
`head_correction=off` under the overnight allowance. The inherited head-enabled
candidate also failed the scalar-head condition/backward-error gate in
58457257.12 (run 464). Turning heads off does not establish the requested
parity. The fresh Sigma differences reproduce the inherited heads-off pair.
Na's changed pole fit also changes the Sigma quadrature: 631 versus 604
window-node pairs, with six candidate rule hits and six new fits from the
identically seeded existing cache. This is an end-to-end comparison.
The Sigma parser's PASS label means compatible finite data, not 2 meV acceptance.

The three inherited commits were rebased onto fetched current main, which had
advanced beyond the brief's `2df9814ee` through the minimax service migration.
The only conflict was the `model.py` import: preserve main's public minimax
calls and add `sampling`. The numerical diff remains **27 files, 431 additions,
1516 deletions**. No numerical edit was made during this finish pass. The two
`sc_iteration.py` changes are comments. Dead head/BSE/benchmark producers,
S-tensor implementations and bispinor four-current chi were not changed.

The deleted scan summed every ordered `(a,b)` pair in the logical chi band
window, for every full-BZ `k` and requested `q`, with weight
`(f[a,k]-f[b,k-q])/(e[a,k]-e[b,k-q]+z)` and spin-contracted pair densities.
It was not restricted to valence/conduction pairs. Same-band and partially
occupied pairs were included, as were occupied/occupied and empty/empty
indices; equal occupations make their weight zero. Same-band pairs at finite
q can contribute. At Gamma, an identical-state term is zero at nonzero z:
there was no static occupation-derivative replacement in this remaining scan.
Padding outside `nb_logical` was masked. The physical ordered orientation
used the inverse momentum map (`k+q`) and the other density conjugation.

The new first near-line sample is `i*nu_n`, where
`n=max(1,round(0.5 eV/(2*pi*kT)))`. `model._evaluate_samples` calls
`compute_chi0_matsubara` at `nu_indices=(n,)` with the live Fermi-Dirac
occupation state and `minimax_target_error`. Finite-temperature Green factors
pass through the existing sharded chi kernel, and the full-q writer stores
the wedge rows. No zero-frequency value is substituted. Other metal samples
retain their fractional contour producer. The origin-shift input key is deleted.

The fresh Na census proves one Matsubara producer call and no direct scan.
Its first sample uses **18 tau nodes**, `n=1`,
`nu=0.06283185307179587 Ry` (about **0.855 eV**), beta 100 inverse Ry and live
bandwidth 11.007503319246029 Ry. The rule certificate passes at target `1e-6`.
The target is 0.5 eV; requiring a positive Matsubara index gives this deck its
first nonzero frequency. See `legs/na_fd/candidate/chi_census.json`.

The ancillary control `fractional_chi_gate.py` printed passing numerical
checks, then referenced undefined receipt fields at baseline lines 250–252.
Ranks entered mismatched shutdown barriers. Its four verified Python child
PIDs were stopped by own-pool step .7, and the combined script continued.
No control PASS receipt is claimed. The inherited scan-deletion commit already
removes those invalid fields; the candidate writes a PASS receipt.

Not established: head-enabled Na after the rebase; Na agreement within 2 meV;
SC broken-time-reversal metal operation; CT/TT sector models; production
utilization; or a remedy for the changed MPA fit. No sweep was performed.
The sandbox issue ledgers record the Na failures and the control receipt bug.

`evidence/landing_gate.txt` contains `FAIL`. The final STATUS post records the
pushed delivery commit and a guarded fast-forward command that checks this
file before doing anything to main. **Do not land this branch with these results.**
