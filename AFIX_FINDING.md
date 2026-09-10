# AFIX — repairs retained, clock default restored

**The reproducibility defect is UNFIXED AND STILL LIVE.** The default is again
`sigma_quadrature_reduction_steps = None`, preserving historical clock-selected
reduction. A system-scaled deterministic work allowance remains the owner's
design decision. No new dial or scaled policy was implemented.

AREV P1 is resolved by this default decision. Optional integer steps retain the
cooperative refusal watchdog; they are not a strict wall limit. AREV P2 is fixed
at the production boundary: each completed box plan emits a
`Sigma quadrature receipt: ` JSON line that the ordinary `gwjax.out` writer
preserves with debug disabled or enabled. Six tests read the closed output file,
and both production re-gates parse the actual debug-off report.

Implementation commits on `lane/sp-afix-2026-09-10`: `32579a39` restores the
default and persists the receipt; `a863d1e2` corrects only the test's expected
existing 16-character digest and theory prose. Earlier repairs remain in
`a9180796` and `00bdeb6e`. The re-gated source is `a863d1e2`; final report-only
closeout does not change it. All commits are pushed; no main merge is claimed.

Evidence root **R** = `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox/349_afix_20260910`.
Independent review: [AREV_FINDING.md](/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/152_shared_pole_push_2026-09-07/exchange/arch/AREV_FINDING.md).

## Thirteen dispositions

| Finding | Final disposition |
|---|---|
| 1. Duplicate retry | Removed in both modes. Initial certification precedes reduction, and the builder already tries three tighter bases. An identical rebuild with more reduction work cannot repair a start that never certified. AREV confirms this. The remaining split/alternate-rule remedy requires numerical implementation; no automatic subdivision is claimed. |
| 2. Lost planning bound | Default restored to None. Seconds selects the last accepted rule at a pass boundary in clock mode. Explicit integer steps use seconds as a cooperative refusal watchdog; a strict wall limit remains external. |
| 3. Fixed-pass cost | Universal ten-pass default rejected by the measured Si counterexample below. No replacement constant, scaling formula or new policy is proposed. |
| 4. Default drift | `_DEFAULTS` remains the sole deck-default owner; DynamicSigmaConfig and the MPA/PPM/dispatch entry points consume it. The one owning value is now None. |
| 5. Cache invalidation | Retained explicit v3 request namespace and versioned filenames. Old entries are skipped before loading; clock and step identities remain distinct. No old cache entries were promoted or deleted. |
| 6. Receipt currency | AREV exposed the earlier dictionary-only repair. Now the report persists root/per-window policy, mode, passes, seconds, backend policy, cache schema/state, requested/certified boxes and selected node digest. Production evidence is below. |
| 7. Core fixtures | Retained explicit A GN-PPM zero passes and B MPA/SC ten passes, each with the existing 20-second watchdog. References and tolerances are unchanged. Both parent and candidate pass all 28 core tests on every P4 rank. |
| 8. Documentation | Input reference owns the reachable None default and both budget semantics. Theory examples agree; the clock-mode reproducibility defect is stated explicitly. |
| 9. Remedy | Timeout names both operative keys. Initial-certificate refusal explains why more reduction work cannot repair it. Retired-key guidance names the current controls. |
| 10. Tests | Default None is pinned. Six tests cover on-disk receipts for None/0/10 with debug off/on; they fail if either emission or production retention disappears. Prior parser, cache, certification and watchdog tests remain. |
| 11. Inert seconds | Seconds is live in both modes; positive finite validation remains appropriate. It selects a clock-mode result or bounds step-mode completion cooperatively. |
| 12. Backend condition | Fixed-work reproducibility remains conditional on source, inputs, backend/libraries/threading and cache inventory. The durable receipt names the requested backend policy; no cross-backend guarantee is made. |
| 13. Cache-state input | Widening and containing-rule reuse remain intentional. The durable receipt records both boxes, cache selection and node digest. Off/cold/warm states are not promised identical. |

The production boundary is `src/gw/sigma_box_plan.py:1190` →
`src/gw/production_report.py:135`; the disk test is
`tests/test_sigma_box_plan.py:875`. Ordinary report output uses the existing
writer, without a new reporting API, sidecar or user switch.

## Re-gate against the parent

`R/07_review_core/comparison.json`, job **58152291.5**:

| Scope | Parent 420a0b98 | Candidate a863d1e2 | Difference |
|---|---:|---:|---|
| Explicit core, each of four ranks | 28 passed | 28 passed | No failure IDs |
| Focused configuration/planner + minimax service tests | 116 passed, 1 failed | 119 passed, same 1 failed | No new failure IDs; all six disk tests pass |
| Default collection, each of four ranks | 10 errors | Same 10 errors | No new error IDs |

The focused failure is the inherited missing input-reference rows. The default
collection errors are the inherited distributed-bootstrap ordering defect, not
a green suite. Static AST groups remain 90/90, 34/34 and 16/16; broader inherited
rules/ledger failures remain (`07_review_core/checkpoint_comparison.json`).
The parent archive is immutable. The test-only digest correction was committed
before the candidate phase began; `candidate_phase_source.json` and
`production_source_continuity.json` record this explicitly.

Production gates, all at source **a863d1e2**, debug explicitly disabled:

| Scope / job.step | Exact model | Every-rank peak | Rules | Maximum stored Sigma change |
|---|---|---|---|---:|
| Na P16 / **58152291.6** | All 29 parents | All 16 unchanged: 8,517,691,768 B each | 12 certified, 258 nodes; max sup 9.799695194623072e-5 | **0.14020455076667174 meV** |
| Si P4 / **58152291.7** | All 8 parents | Rank 0 unchanged: 4,406,953,808 B; ranks 1–3 unchanged: 4,166,923,192 B | 8 certified, 698 nodes; max sup 9.452383566459818e-5 | **0.08962922306936033 meV** |

Every rule meets eps=1e-4 and both Sigma changes are below 2 meV. Evidence for
each row: `R/08_review_production/{na_p16,si_p4}/gate.json`, `exact_inputs.json`
and `summary.json`. `disk_receipt.json` authenticates the ordinary `gwjax.out`
content: mode=seconds, steps=null, seconds=120, backend_policy=numpy, schema=v3,
and every selected-window identity and certificate.

Na compares with the immediate AFIX measured parent **00bdeb6e / 58152291.4**;
420a0b98 differs only by report/comment. Si compares with the accepted clock
reference **119e713d / 58137440.16**; the prior AFIX Si production was skipped,
so no fictitious AFIX Si full-driver control is claimed. Exact factors, poles,
physical identity, K/basis metadata and byte-identical decks authenticate reuse
of **CD48 (58128243.23)** and **CD96 (58128243.22)**. No CD calculation was repeated.
Stored analytic-model/CD metrics and their original scopes are preserved in the
summaries; these are separate from the production-vs-parent Sigma gate. No new
frozen 447-call score, bit-identical Sigma or performance improvement is claimed.

Model metadata is unchanged: Na n=896, J=K ranges 1061–1589, compact payload
660,986,096 B; Si n=368, J=K ranges 1068–2010, compact payload 94,807,744 B.
Both damping fractions are zero. Existing port widths are unchanged; condition
numbers were not remeasured. All V-whitened passivity receipts pass at eta=0.25 eV
(Na maximum 0.9664696419689606; Si maximum 0.864070904623472). Exact extrema and
inherited constructor diagnostic WARN/NOT_MEASURED entries remain in the artifacts;
no blanket promotion of those diagnostics is made.

## Reproducibility remains live

The original identical-source control is **baa3258f**:

| Arm | job.step | Crossing nodes | Times/weights identity |
|---|---|---:|---|
| Historical | 58128243.23 | 87 | b5b8abef… |
| Repeat 1 | 58137440.10 | 87 | Same as historical |
| Repeat 2 | 58137440.11 | 86 | 84179bbf…; different |

Every rule certifies at eps=1e-4, yet the maximum stored Sigma difference is
**0.015627059540348685 meV**. Full hashes, certificates and absolute artifact
paths are in `R/07_review_core/live_reproducibility_evidence.json`, pointing to
Run341/07_parent_variance/variance.json. The roughly **0.067 meV** observation
is a separate paired-instrumentation arm **58137440.9**, with 85 nodes and
0.06659222118021881 meV; see [ASERV_MECHANISM.md](/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/152_shared_pole_push_2026-09-07/exchange/arch/ASERV_MECHANISM.md).
These scopes are not combined into three distinct hashes or a stronger control.
Two explicit steps=0 controls **58141542.4/.5** produced identical arrays for
all twelve boxes, 513 nodes total ([AQUAD_DETERMINISM.md](/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/152_shared_pole_push_2026-09-07/exchange/arch/AQUAD_DETERMINISM.md)).
None of this establishes a universal deterministic work allowance.

AREV also explains the former AFIX **58152291.4** Sigma movement: reference
**58137440.17** used clock mode, whereas AFIX used ten passes. Four crossing
rules changed 86→92, 11→14, 73→76 and 16→20 nodes; total 257→273. The
0.1402045508 meV was changed-policy integration, not mysterious variation at
identical policy. No per-window decomposition of that maximum was measured.

The Si S02 counterexample remains valid: **58152291.2**,
`R/02_budget_replay/Si_S02.json`, parent ten-pass repeats
504.0675688/502.2792082 s, 207 nodes, sup=9.89341440458e-5; the former default
candidate refused after 189.7084135 s against its 120-second watchdog.
The 69.71-second overrun does **not** identify the internal phase responsible;
earlier AFIX attribution to one in-flight removal pass was too specific.
The three missing Si replay receipts are incomplete evidence, not three
additional measured refusals. The new Si production above completes under the
restored clock default.

Backlog complete. Open design: a deterministic work allowance that scales with
the system, directed by the owner. The live reproducibility defect is not closed.
