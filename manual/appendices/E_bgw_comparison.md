# Appendix E — Matched comparisons against BerkeleyGW

The checklist. Any LORRAX-vs-BGW discrepancy report that has not pinned every row
below is measuring conventions, not physics.

| Item | Setting |
|---|---|
| Wavefunctions | identical `WFN.h5` (same QE run, same pseudos, same grids) |
| Cutoffs | `bare_coulomb_cutoff` = BGW `screened_coulomb_cutoff` (defaults match, §6.3; pin anyway) |
| Truncation | `sys_dim` = the BGW truncation flag (§6.1) |
| $q\to0$ | same head treatment; `use_bgw_vcoul = true` removes the kernel entirely as a variable (§6.2) |
| Degeneracy averaging | on in both (LORRAX default; §4.5) |
| PPM | matched probe; `ppm_invalid_mode` ↔ `invalid_gpp_mode` (§7.4); BGW broadening at its minimum |
| Comparison target | total $\Sigma_c$ and eqp only; LORRAX Σ⁺/Σ⁻ branches never map onto BGW's printed SX−X/CH split |

Triage order when a residual survives the checklist: 1.) exchange-only run
(`compute_mode = x_only`) isolates the Coulomb/cutoff sector; 2.) static COHSEX
isolates screening from the frequency machinery; 3.) a residual that plateaus
while $n_\mu$ grows is never ISDF basis error, it is a remaining convention
difference. Absolute agreement expectations at matched settings: exchange and
static COHSEX at meV level; GN-PPM Σ_c at the level of §7.6.

BSE comparisons add six more conventions (dipole operator sign $\mathbf p \pm
\mathbf v_{NL}$, eqp source, head injection, SOC band counting, occupation
resolution, broadening and iteration count); each silently produces $O(1)$
plausible-looking errors when mismatched. The validated Si Haydock comparison in
`src/bse/BGW_COMPARE.md` is the worked example with the exact command sequence.
