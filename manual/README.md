# LORRAX Manual — working draft

One Markdown file per section, LaTeX math inline (`$...$`, `$$...$$`), no serious LaTeX
formatting yet. Figures are placeholders: `<!-- FIGURE: description -->`. The governing
outline, editorial threads (T1–T5), and pre-writing task list live in
[`00_outline.md`](00_outline.md). **Prose register is governed by
[`STYLE.md`](STYLE.md) — read it before writing or revising any section.** §1.1
(as revised by Jack) is the reference implementation.

## Status

| Chapter | State |
|---|---|
| 00 Outline | current |
| 01 Introduction | 1.1 user-revised (register reference); 1.2/1.3 archived to `_archive/` pending rework |
| 02 Installation | 2.1 rewritten as JAX narrative 2026-07-11; 2.2-2.4 swept |
| 03 Tutorial | drafted; PRL-density sweep 2026-07-11 (numbers TODO: reference run) |
| 04 GW in real space and real time | drafted; reviewed; PRL-density sweep 2026-07-11 |
| 05 ISDF | drafted; reviewed; PRL-density sweep 2026-07-11 |
| 06 The Coulomb interaction | drafted; reviewed; PRL-density sweep 2026-07-11 |
| 07 Frequency integration | drafted; PRL-density sweep 2026-07-11 (§7.6 periodic numbers blocked on gate run) |
| 08 Bispinor GW | 8.1-8.4 drafted 2026-07-13, revived from `agent/manual` and re-synced to `origin/main@8b6e3cc7` on 2026-09-01; the status section routes to `docs/theory/four-current-head-corrections.md` rather than restating it |
| 09 Supporting formalisms | not started |
| 10–13 Reference + architecture | not started |
| Appendices A–F | B revived 2026-09-01 as a short router to the four-current head owner page; A, C–F not started |

## Conventions

- Rydberg atomic units internally ($e^2 = 2$); all output energies in eV. See §1.4.
- Notation follows `docs/theory/physics.md` (band $n$, k-point $\mathbf{k}$, spinor
  index $s$, centroid $\mathbf{r}_\mu$).
- Cross-references between sections use relative links; input keys are shown as
  `code_font` and always match `gw_config.py` spelling.
- Boxed asides: **BGW correspondence** (thread T4) and **Reader note** boxes are
  blockquotes beginning with a bolded tag.
