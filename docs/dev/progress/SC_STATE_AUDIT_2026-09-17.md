# Shared-pole SC state audit, 2026-09-17

Source inspected: main a559896f4, SCMETAL97cb2f026 and sector integration7ef88f39d. This audit accompanies874d6c044 on branch audit/shared-pole-takeover-2026-09-17; it is not a completed SC validation.

## Integration update

The original density audit below left a real defect open. Candidate
`c17bb176`, integrated with placement follow-ups through `d6a204dc`, now
uses the canonical full-band occupation table for density/current. On Fe,
the full ladder is 35 physical bands (carrier 36), while the active QP
Hamiltonian is 26 bands (sweep carrier 28). Density uses the full ladder
and the active rotation embedded with an inactive identity; Hartree matrix
elements are returned on the logical 26-band target. These are different
physical spaces, not interchangeable band counts.

Run477/36, P4 job `58489223.8`, completed all four sector Sigma sweeps
and reached the later scissor-frontier refusal. Source control flow places
the full-density Hartree rebuild and finite, logical-shape Hamiltonian
addition before that refusal. This establishes passage through those
boundaries, not independent Hartree accuracy or a completed SC map.
The ±5 eV window excluded crossing bands 9–10 and 19–20. Fresh Run477/37,
job `58496217.0`, uses ±12 eV and authenticates protected bands 9–20 at
every k. Its multi-map outcome is pending; no convergence is claimed.

Exact same-operator mirror samples have separately passed all 13 Fe CC
parents (claim2454). The full CC/TT/CT stores and four production Sigma
sweeps are established by claim2456. The next-map source review finds fresh
occupations, rotated bundles, samples and models for each input Hamiltonian;
only the prescribed initial photon contact is retained. Runtime receipts
across completed maps remain necessary. Head-off is the authorized campaign
diagnostic scope, not a production head or physical-convergence certificate.

The following table records the earlier audit; its density and remaining
implementation entries are superseded by this update.

| Object | Current owner/lifetime | Audit result |
|---|---|---|
| QP eigensystem | sc_iteration.gw_iteration_map, every map call | Initial canonical DFT identity bypass is intentional; later H is diagonalized. U/E are unfolded through the one k-star owner. |
| Metal chemical potential and occupations | _solve_head_occupations → _solve_occupation_state on the current full ladder | Found premature T=0 density-SC solve before FD;874d6c044 removes it and uses current FD mu. P4 regression and baseline refusal below. |
| Bundle occupation tables and frozen-head Sigma ladder | rotate_wavefunctions and fixed_dft_full_head block | SCMETAL97cb2f026 supplies the authoritative current FD table to both rotated bundles/parents and replaces the frozen head's Sigma energies/occupations/reference. This fix is on an integration branch, not baseline main. |
| Density/direct fields | rebuild_hartree_dft_basis | Rebuilds orbitals/current fields from current U/E; fixed DFT psi(G) is an immutable basis. Independently solves FD on the active ladder, whereas screening's entry solve sees logical sum bands. No mismatch measured here; thermal tail identity is an open audit question, not a diagnosed defect. b0!=0 is refused by run_sc_driver. |
| Centroids, zeta, bare V | invariant input representation | Fixed basis and bare interaction are intentional. QP endpoint bundles rotate each map. Ordinary per-map zeta refitting did not cure the measured Si instability; this does not uniquely identify its seed. |
| Samples, moments, directions, poles and factors | bind_shared_pole_census/resolve_shared_pole_recipe → screen_shared_poles | Rebound to current state each map. sc_ labels bypass reusable one-shot model membership. A retained quadrature certificate is not a retained physical W model. |
| Quadrature/support session | fixed_quadrature_session and current-map recipe | Caches mathematical rules/envelopes with recertification; current masks, states, samples and physical ranks are rebuilt. Frozen physical poles are not an authorized stabilization. |
| Bispinor static contact reference | sector integration screening_seed_cache and photon bank static_reference | Intentionally authenticated initial reference under current campaign prescription; do not confuse it with current orbitals or silently recompute it. Successive complete maps remain unmeasured. |
| Sigma support | efermi.band_in_occupation_window/ppm_windows | Production0.005 branch-weight cutoff differs from historical CD full-FD support. SIGSCORE attributes2.5meV common-grid RMS to that approximation. Compare identical support to isolate consumer error, retain full-FD score for physical accuracy. |
| Active and far-band scissor | _frozen_scissor_fits, tail fit, identity partition | Active map0 correction frozen intentionally; far sum-band tail refits. Per-k identities/multiplets select masks. Neither freezes W. |

## Focused regression

Perlmutter P4 job.step58484372.2: six tests in tests/test_sc_metal_occupations.py pass on each of four ranks. The actual gw_iteration_map is exercised through its occupation boundary with a supplied eigenspectrum [-1,0,0,1]Ry and two electrons. The FD owner returns equal0.5 weights on the degenerate pair; the original map function from a559896f4 refuses before reaching that owner. No screening, Sigma, Hartree numerical equality or full Fe SC convergence is claimed. Evidence: sandbox runs/DEV/153_shared_pole_takeover_2026-09-17/verification.json and claim2443.

## Remaining implementation boundaries

The sector continuation owns signed-contact stability admission, CT construction/publication and the sector manifest. A separate consumer continuation owns frequency-integrated CC/TT/CT/TC Sigma and the instantaneous W_infinity-V contribution. Neither existing bank completion nor primitive complex-time contractions establish this full route. Fe finite Gram and Na nonfinite spectra are different open investigations; the preliminary occupation refusal cannot explain their map0 failures.
