# LORRAX Manual — working draft

One Markdown file per section, LaTeX math inline (`$...$`, `$$...$$`), no serious LaTeX
formatting yet. Figures are placeholders: `<!-- FIGURE: description -->`. The governing
outline, editorial threads (T1–T5), and pre-writing task list live in
[`00_outline.md`](00_outline.md).

## Status

| Chapter | State |
|---|---|
| 00 Outline | current |
| 01 Introduction | drafted; 3-lens review applied 2026-07-10 |
| 02 Installation | drafted |
| 03 Tutorial | drafted (tutorial numbers TODO: needs reference run) |
| 04 GW in real space and real time | drafted; 3-lens review applied 2026-07-10 |
| 05 ISDF | drafted; 3-lens review applied 2026-07-10 |
| 06 The Coulomb interaction | drafted; 3-lens review applied 2026-07-10 |
| 07 Frequency integration | blocked on pre-writing tasks 1/2/5 (see outline) |
| 08 Bispinor GW | not started |
| 09 Supporting formalisms | not started |
| 10–13 Reference + architecture | not started |
| Appendices A–F | not started |

## Conventions

- Rydberg atomic units internally ($e^2 = 2$); all output energies in eV. See §1.4.
- Notation follows `docs/theory/physics.md` (band $n$, k-point $\mathbf{k}$, spinor
  index $s$, centroid $\mathbf{r}_\mu$).
- Cross-references between sections use relative links; input keys are shown as
  `code_font` and always match `gw_config.py` spelling.
- Boxed asides: **BGW correspondence** (thread T4) and **Reader note** boxes are
  blockquotes beginning with a bolded tag.
