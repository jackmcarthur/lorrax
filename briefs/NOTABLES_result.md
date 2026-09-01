# NOTABLES — on-demand-only planner (heavy lane)

## Achieved numbers

- Sign-definite one-window A/B on one CPU host, back to back: shipped lookup **0.01405 s, 7 nodes, residual 8.950e-7, kappa_p99 1.0054**; on-demand fit **0.05911 s, 6 nodes, residual 6.727e-17, kappa_p99 1.0000**. Planning cost increased by **0.04506 s (4.21x)** while node cost fell by one.
- Exact sprint gate on the consolidated base collected **136 tests**: first full run **135 passed / 1 failed in 87.52 s**. The failure identified lost refusal metrics; after the evidence handoff fix, its red cell passed **1/1 in 8.81 s**, and the complete post-fix gate passed **136/136 in 94.31 s**.
- Total diff: **499 net lines deleted** (89 added, 588 deleted); production/service code accounts for **488 deleted lines**. Removed code includes catalog walk/load, sign-definite table candidates, crossing table candidates/fallback fit, and catalog-reach omega patching.

## What changed

Every sign-definite and crossing product window now enters the existing measure-adapted ROQ fitter. Catalog reach no longer partitions wide crossing windows, and catalog lookup is no longer a candidate or retry stage. The delivered residual, runtime-noise, factor-growth, and node ceilings remain acceptance gates. Cache version 8 prevents reuse of table-derived plans. ROQ refusals now carry the best achieved residual and kappa into the production census.

Catalog-behavior tests in `tests/test_delivered_windows.py` were rewritten because lookup-only behavior, catalog certificates, and HGL-driven patch counts are intentionally gone; the replacements assert on-demand family selection, no catalog helpers, unpatched wide windows, deterministic execution, and compounded tightening.

## Not measured in this sprint

The 15-minute lane did not complete the full sodium one-shot/control-panes comparison or the 19-window formerly-null deck, so there are no honest sigma_c max/RMS meV or formerly-null-window rank/residual/kappa claims here. No GPU verification was run: this lane changed CPU planning/fitting code only, so the `AGENT_PREAMBLE.md` unit/CPU exemption applies; no P=4 execution claim is made.

Branch: `feat/on-demand-only-2026-08-31`. Evidence and handoff: this file plus the committed tests.
