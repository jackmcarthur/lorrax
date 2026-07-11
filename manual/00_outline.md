# Manual outline, editorial threads, and pre-writing tasks

Approved plan (2026-07-10). ~52 pp target.

## Threads

- **T1 — The two-sums rule.** No expression ever sums over valence×conduction pairs;
  only separate v-sums and c-sums glued by an O(N⁰) set of time/frequency nodes.
  Introduced §4.2; every formalism chapter exhibits its central object in this form.
- **T2 — One picture at three resolutions.** Everything is a two-point kernel
  K(r,r′). Ch. 4 draws physics on (r,r′) as if on the explicit grid; Ch. 5 reveals
  (r,r′)→(r_μ,r_ν); Ch. 13 reveals (r_μ,r_ν)→2D processor grid.
- **T3 — Every approximation has one knob and one figure.** Centroid count,
  `minimax_target_error`, screened cutoff, ω_p, band window: each gets a convergence
  figure with the input key named in the caption.
- **T4 — BGW correspondence boxes.** Flag equivalences, convention differences, and
  where correspondence *fails* (Σ⁺/Σ⁻ ≠ SX−X/CH; only total Σ_cor comparable).
- **T5 — Spinor-first.** ψ is always a 2-spinor; σ⁰ trace in ISDF; four-density
  bispinor as the natural extension; honest capability con in §1.3.

Pedagogical stance: heuristic explanations for machinery users cannot inspect;
quantitative precision reserved for error/knob statements (T3 makes trust empirical).

## Chapters

Part I — Overview & Getting Started (~7 pp)
1. Introduction: 1.1 what LORRAX is · 1.2 capability matrix · 1.3 relation to other
   codes, ending with LORRAX-vs-legacy-BGW pros/cons · 1.4 units, notation, citing
2. Installation: 2.1 support matrix · 2.2 pure-Python (serial tier) · 2.3 cluster tier
   (phdf5 + distributed linalg strongly recommended) · 2.4 site recipes
3. Tutorial: 3.1 Si end-to-end · 3.2 same system GN-PPM · 3.3 2D+SOC teaser

Part II — Theory & Methods (~22 pp)
4. GW in real space and real time: 4.1 objects (r,r′ disclaimer box) · 4.2 two-sums
   rule · 4.3 k-points and N_k log N_k · 4.4 time integration heuristics · 4.5 QP
   solvers (+rCROP, Wan & Międlar)
5. ISDF: 5.1 pair-density factorization (spin-free) · 5.2 spinors improve it (σ⁰) ·
   5.3 the fit, stated not derived · 5.4 symmetry & orbit closure (brief) · 5.5 rank
6. The Coulomb interaction: 6.1 kernel & truncation · 6.2 mini-BZ average ·
   6.3 cutoffs (pair bandwidth 4×E_cut; BGW contrast) · 6.4 q→0 note → App. B
7. Frequency integration: 7.1 pole-convolution constraint · 7.2 minimax machinery ·
   7.3 χ · 7.4 W as multi-pole (GN two-point fit) · 7.5 Σᶜ(ω) (three-window) ·
   7.6 accuracy & validation
8. Bispinor GW: 8.1 formalism · 8.2 four-density ISDF + cost model · 8.3 Σ^B assembly ·
   8.4 status & validation
9. Supporting formalisms: 9.1 DFT operators (dipoles, i[r,Σ] upgrade) · 9.2 BSE
   (matvec-without-matrix, solver menu) · 9.3 bandstructure (htransform, distributed
   linalg mandatory at large n_μ)

Part III — Reference (~14 pp)
10. Workflow & files · 11. Complete input reference (regenerated from `gw_config.py`;
    + BSE CLI + bandstructure CLI; CI-diffed) · 12. Running at scale: 12.1 execution
    model · 12.2 memory planning · 12.3 distributed linear algebra · 12.4 parallel
    I/O · 12.5 troubleshooting

Part IV — Architecture (~5 pp)
13. Code architecture: 13.1 module map · 13.2 distribution model (T2 final
    resolution) · 13.3 host-memory discipline · 13.4 FFI layer · 13.5 testing &
    frozen gates

Appendices (~7 pp)
A symmetry (unfold equations; sym-reduced vs not ledger) · B q→0 head/wings ·
C minimax tables · D file formats · E BGW comparison cookbook · F symbols/units ·
G bibliography (all method citations; the home for every "Appendix G" reference)

## Pre-writing tasks (blockers for Ch. 7/8/App. A)

1. ~~GN B-normalization~~ **RESOLVED 2026-07-10** from `minimax_screening.py:406-423`:
   model is W_c = 2BΩ/(ω²−Ω²) = B[1/(ω−Ω) − 1/(ω+Ω)]; Ω² = −z²W_c(z)/(W_c(0)−W_c(z)),
   B = −½W_c(0)Ω, validity Re Ω² > 0. The GN guide was right; `docs/theory/physics.md`
   §6.9 states B/(ω²−Ω²) and NEEDS UPSTREAM CORRECTION. NB the q→0 head fit uses the
   *other* normalization (B_h/(ω²−Ω_h²), R_h = B_h/2Ω_h) — App. B must state both.
2. ~~Window scheme~~ **RESOLVED 2026-07-10** from `ppm_windows.py:250-353`: three
   windows (core ≤T/HGL; a_stripe E>T/GL; b_slab Ω>T/GL), T = ω_max + edge_factor·ξ,
   A_core = 2T/ξ; 4 branches {occ,emp}×{±ω}, one crossing branch per half. The
   six-window note (`dev/notes/NEW_WINDOW_MINIMAX_GUIDELINES.md`) is unshipped design
   — archive it.
3. Pin bispinor cost exponents (×4 memory / ×16 runtime claim) against the planner.
4. Confirm the non-sym-reduced stage list for Appendix A (P_k pair-density k-sum).
5. Fresh periodic Σ(ω)-vs-BGW gate before §7.6 prints accuracy numbers.
6. Bibliography (home: Appendix G): add Godby–Needs (absent from repo); collect Kim
   2020, Hackbusch, Golub–Welsch, Wan–Międlar, Hybertsen–Louie, Leon MPA set.

## Source-material map (freq chapter)

Reading order: `docs/theory/minimax-quadrature.md` (spine, incl. LORRAX-fitted error
laws) → `docs/dev/archive/ctsp_revised.md` (derivation prose, CD analogy) →
`docs/theory/physics.md` §6.9 (Σ code narrative) →
`docs/dev/notes/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` (window-edge algebra; ±ω prose
unreliable) → Kim-2020 appendix transcription (primary source) →
`src/common/minimax_assets/README.md` + `reports/sigma_ppm_tighten_2026-07-04`
(error conventions; the per-pole-term −ω identity correction).
