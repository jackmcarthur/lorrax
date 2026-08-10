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

**And the duty is symmetric.** If your lane settles something a report left open, or
overtakes a statement one of them makes, the marker goes in **two** places: the report
body, where a reader who opened the file will see it, and the report's row in
`INDEX.md`, which is the only one of the two that lanes are told to read. A marker that
exists in the body alone is invisible to every lane that follows the rule. This applies
to light lanes too — it is one line in the index, not an index entry of your own.
