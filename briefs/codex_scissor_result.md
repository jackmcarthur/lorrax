# Sodium SC scissor audit

Heavy lane.  Branch `feat/scissor-hardening-2026-08-31`, based on
`fix/sc-selector-refusal-2026-08-31` (`263ab34a`).

## Sample census

Measurement pending.  For each symmetric Sigma half-width below, `n` counts
the accepted `(k, band)` rows and `w` is their full-BZ weight.  A class is
called well-determined only when it contains at least two distinct DFT
energies; `n >= 2` alone is insufficient because all rows may be a degenerate
manifold.

| half-width (eV) | val n / w | val alpha / beta (eV) / RMSE (eV) | crossing n / w | cond n / w | cond alpha / beta (eV) / RMSE (eV) |
|---:|---:|---:|---:|---:|---:|
| 5  | pending | pending | pending | pending | pending |
| 10 | pending | pending | pending | pending | pending |
| 15 | pending | pending | pending | pending | pending |
| 20 | pending | pending | pending | pending | pending |

## Pre-measurement questions

- Pairing: compare the independent sort-and-pair assignment with band-index
  pairing, reporting changed pairs and their QP-energy effect near crossings.
- Classes: verify the occupation-derived valence / crossing / conduction
  boundaries against the measured energy and occupation tables.
- Affine adequacy: inspect residuals by energy and band, and compare an affine
  prediction with the directly solved QP energies.  No higher-order law is
  introduced unless it changes a consumed QP energy materially.

Evidence directory and achieved conclusions will replace this checkpoint.
