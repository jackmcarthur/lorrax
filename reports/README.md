# `reports/` — in-tree, and small

This directory holds one frozen in-tree campaign record
(`device_invariance_2026-07-08/`). It is **not** where the fleet's reports live.

**Campaign reports live outside the tree, and you read their index rather than the
directory:**

| where | read first |
|---|---|
| `~/lorrax_bse_perf_2026-08-08/` (82 reports, 2026-08-08→10) | **`INDEX.md`**, then `EVIDENCE_MANIFEST.md` |
| `~/lorrax_service_phase/` (service phase) | `HANDOFF_2026-08-08.md`, and `BUILD_NOTES.md` before any cluster leg |
| `docs/reports/` (in-tree, published results) | the file itself; it is a short list |

Grepping a reports directory raw is the anti-pattern — the indexes carry the
**superseded by** markers that a grep cannot see. Efficiency doctrine rule 3 in
`AGENT_PREAMBLE.md`.
