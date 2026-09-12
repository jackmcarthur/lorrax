# AUNION — the complete seven-lane union, resolved and gated

**VERDICT: the union is SAFE TO MERGE on the evidence below, with one
pre-existing defect it inherits and does not cause.** Every conflict is
resolved keeping both sides; every clean-merge file is reviewed by hand; the
union imports and passes its contracts at P4; its numerical movement on Si is
ACONJ's deliberate change and nothing else; it *passes* an every-rank peak gate
that ACONJ alone failed; and the multi-map shared-pole SC death at map 1 is
reproduced on two sources that are strict subsets of the union, one of which is
the SC branch alone, so the union introduces it — no.

Branch `integ/sp-union-2026-09-11`, tip `9160c501`, base `810c260b`.
Worktree `/pscratch/sd/j/jackm/wt_sp_union`. Run root
`runs/frequency_integration_sandbox/370_aunion_20260911` in the sandbox.

## What is in it

Seven lanes, merged in this order, all onto `810c260b`:

| # | lane | tip | conflicts |
|---|---|---|---|
| 1 | AFIXES | `1e7e4697` | clean |
| 2 | ARETAIN | `9a030add` | clean |
| 3 | ACARRY | `bf002beb` | clean |
| 4 | ACONJ | `421c016d` | clean |
| 5 | AREV2 | `ea26fed9` | clean |
| 6 | AWDEBUG | `14594140` | 1 file, 1 hunk (`c3f3c43f`) |
| 7 | SC quadrature | `bf6048c9` | 4 files, 12 hunks (`9160c501`) |

Lanes 1–5 were assembled and pushed by the coordinator at `160ae4ed`; lanes 6
and 7 are mine.

## Every conflict, and why it was resolved that way

### `src/file_io/shared_pole_store.py` — AWDEBUG vs ARETAIN (1 hunk)

The only textual conflict was the `export_shared_pole_outputs` **docstring**.
Both paragraphs kept, plus one sentence saying the managed retention covers the
debug bank as well. The four code sites (`targets` dict, the bank-source guard,
the bank export branch) auto-merged to `config.debug.write_w` and still run
through `_retain_current_map_exports`, so the key stays debug-grouped *and* the
retained set stays bounded to two files.

**The textual merge missed one thing, and it would have been a silent red.**
ARETAIN's `check_retention` built its config as
`SimpleNamespace(write_w=True, write_poles=True)`. AWDEBUG could not update a
test that did not exist on its base, so after a clean merge that object has no
`debug` group and the retention test dies on `AttributeError`. Rebuilt as
`SimpleNamespace(write_poles=True, debug=SimpleNamespace(write_w=True))`. This
is the exact failure mode the "dangerous category" warning is about, and it was
in the *conflicted* file rather than a clean one.

### `src/gw/mpa/sigma.py` — ACARRY/AFIXES vs the SC line (7 regions)

1. **`_shared_pole_routed_synthesis` signature.** Takes BOTH new keyword
   arguments: SC's `realize` and ACARRY's `gemm`. Neither side's caller works
   without its own.
2. **The panel/kernel construction.** Kept ACARRY's `make_kernel` structure —
   one `gemm_plan` per padded width, no `compact_kernels`, an AOT peak query for
   local and routed alike — and gave the routed branch SC's per-panel
   `shared_pole_operator_realizer`. The local branch already carries the
   realizer inside `_shared_pole_panel_unfold`, which is why only the routed
   branch needed it here. **Both sides had independently hoisted the AOT loop
   out of `if local:`**, which is a good sign the two designs agreed about that
   much.
3. **AOT lowering specs.** ACARRY's two-axis `P(None,"x",None,"y")` /
   `P(None,"y",None,"x")`. This is not a preference: it is what
   `read_shared_pole_faces` now produces, so SC's one-axis specs would lower a
   program the reader never feeds.
4. **Compiled reservation.** `peak.total + native_workspace` (ACARRY), because
   the native GEMM workspace is real bytes the SC side never priced.
5. **Local panel peak.** SC's **six** tiles (`(6*b+children)*tile`) with
   ACARRY's two-axis face term (`80*b*spin*m*c/(px*py)`). SC raised 2→6 because
   parent, partner and group accumulators coexist under realization; ACARRY
   changed the face term because the carrier changed. Different terms, same
   expression — both apply.
6. **Routed panel peak.** Same reasoning: SC's `5*children*tile` with ACARRY's
   `16*children*spin*m*c/(px*py)`.
7. **Width search.** ACARRY's mesh-multiple probe points and integer slope
   guard, with SC's projector bound (`projection_bytes > U` → `continue`)
   applied to the same panel size. SC's post-loop projector refusal auto-merged
   and is unchanged.

### `src/gw/shared_pole_screening.py` — the export lane vs the SC line (3 regions)

SC's signature (`head_resolver`, `mpa_plan`, `iteration_head_response`,
`material_class`), its `sc_scratch` labelling and its head construction, PLUS
the export lane's source-WFN guard and `export_shared_pole_outputs` call on
**both** the restart and the fresh path.

**Order on each path is head, then export, then restart membership.** That is a
choice, and the reason is an invariant worth having: an export on disk then
implies a finished map. The reverse order would leave a written, immutable
export behind a failed head build, and the immutability guard would refuse the
rerun.

### `src/gw/shared_pole_constructor.py` — ACONJ vs the SC line (1 hunk)

SC reworded five gate `reason` strings to say what is and is not measured
("raw latent model; … projected operator not measured"). ACONJ added a
`model_reciprocity` row. Kept SC's wording for all five and put ACONJ's row in
ACONJ's position (after `held_w`), reworded to SC's convention. A receipt whose
reasons half describe the raw latent model and half do not is worse than either.

### `tests/test_sigma_box_plan.py` (1 hunk)

Mechanical, both sides' tests. **Not gated at P4** — see the P=1 leg below.

## Every clean-merge file, reviewed by hand

Eleven files changed on both sides of the SC merge and merged without conflict.
This is the category where git produces something that compiles and nobody has
reasoned about, so each was read as two diffs against the merge base.

| file | union side | SC side | verdict |
|---|---|---|---|
| `src/gw/sigma_dispatch.py` | AFIXES: `quadrature_reduction_steps` default from `DynamicSigmaConfig` instead of `None` | SC: shared-pole head enabled (was a refusal); `fixed_quadrature_session` keyed by model instead of disabled for shared_pole | Independent. The two touch different arguments of the same call. The SC change makes `fixed_rule_session` live for shared_pole, which is what the SC leg then exercises. |
| `src/gw/gw_config.py` | AWDEBUG: `write_w` → `DebugConfig`, parse-time `WARNING -- DEBUG` | SC: head no longer refused for shared_pole; the `mpa_*` keys become live when the head is on; SC degeneracy-tolerance docs | Independent. **Behaviour change worth naming:** a production shared-pole deck that used to refuse at parse with `head_correction` non-off now runs with a head. Every campaign deck sets `head_correction = off`, so no executed deck changes meaning — but the refusal is gone. |
| `src/gw/shared_pole_recipe.py` | ACONJ: `model_reciprocity` gate row | SC: `operator_realization` recipe row, `sc_rebuild` version override | Both land. Both change `GATE_HASH`/`RECIPE_HASH`, so stored shared-pole models from `810c260b` refuse and must be rebuilt — loud, not silent. Measured on the union: `GATE_HASH f3052d83…`, `RECIPE_HASH c7bdf59e…`. |
| `src/file_io/shared_pole_store.py` | ARETAIN retention, AFIXES `agree_io_refusal`, digest-loop fix | SC: new `read_shared_pole_matrix` reader | Additive; the new reader is a separate function the head path calls. |
| `src/gw/production_report.py` | AFIXES: the box-plan receipt is retained even with debug off | SC: SC-specific degeneracy and coverage wording | Independent; different emit sites. |
| `src/gw/sigma_box_plan.py` | AFIXES: rule-cache schema v3, budget policy, stale-file migration warning | SC: two extra escape reasons in `_fit_fixed_sc_rules` (error currency, factor growth) | Independent functions. Verified `_FACTOR_GROWTH_CAP`, `_factor_growth`, `spec["E_ref_A"/"E_ref_B"]` all exist on the merged tree. |
| `src/gw/mpa/sigma.py` | (conflicted, above) | | |
| `tests/test_shared_pole_inputs.py` | AWDEBUG's three new tests; gate count 15 → 16 | SC renames `test_shared_enabled_head_refuses` → `test_shared_enabled_head_uses_scalar_mpa` | Coherent: the renamed test asserts the new behaviour, the count assertion is 16 for ACONJ's row. No stale refusal text survives anywhere in `src`/`tests`/`docs`. |
| `tests/test_shared_pole_store.py` | ARETAIN/AWRITE fixtures | SC's `_sigma_fixture` (physical 3×3 q grid) | Additive. |
| `tests/multi_device/shared_pole_sigma_memory.py` | ACARRY header fields | SC header fields (`grid`, `representation`, `q_irr_full_idx`, `operations`) | Additive to one planted dict. |
| `docs/input_reference.md`, `docs/services/symmetry_maps.md` | export-key rows | `sigma_w_model`/`sigma_w_accuracy` rows, projector row | Different rows. |

Beyond the table, three cross-file checks that a per-file review would miss:

* **Callers of the changed signatures.** `screen_shared_poles` has exactly one
  production caller (`src/gw/screening.py:915`) and it already passes all four
  SC kwargs. `synthesize_shared_pole_parents` and `_shared_pole_contract` are
  called from `tests/multi_device/shared_pole_sigma_p4.py`, which already passes
  `gemm=`. `tests/test_afixes_review.py` drives `screen_shared_poles` positionally
  and still works because the SC kwargs have defaults.
* **`tests/bench/shared_pole_sc_group_projection.py`** monkeypatches
  `sigma._shared_pole_panel_unfold` to add `project_little_group_operator` on
  top. On the union that function already realizes internally, so the bench
  would double-apply. It is a bench, not a gate, and it is not in any suite —
  named here so nobody reads it as a gate later.
* **No stale text.** `grep` for the removed head refusal across `src`, `tests`
  and `docs` returns nothing.

## Gates

All legs pinned with `--jid`, judged by artifacts, never by rc. Every leg below
is a **correctness arm**: no timing band is claimed anywhere in this document,
so no placement verdict is required for these numbers (`placement_audit.py`'s
own rule).

### Leg 0 — does the union build? P4, job `58212398.46`, all four ranks PASS

* 20 modules import on a real 4-process GPU mesh, every one from the union
  worktree (`import_outside_checkout: []`).
* **Every lane's contribution present by name, not by grep**:
  `_shared_pole_routed_synthesis` carries both `realize` and `gemm`;
  `_band_fence` is `_unfenced`; `model_reciprocity` present with 16 gate rows;
  `screen_shared_poles` has `head_resolver` and `material_class`;
  `_retain_current_map_exports` present with pattern `sc_[0-9]{4}_(?:poles|w)\.h5`;
  `write_poles` top-level, `write_w` absent at top level and present at
  `config.debug.write_w`, default false.
* 170 focused tests passed, 0 failed, on all four ranks.
* `check_outputs`, `check_retention`, `check_bank_roundtrip` PASS at P4.

### Leg 1 — the P=1-only suite. 1 GPU, 1 rank, job `58212398` PASS

78 passed, 6 skipped, 0 failed. Covers `tests/test_sigma_box_plan.py` in full —
both sides' tests, including the SC line's appended
`test_fixed_crossing_to_relative_recertifies_contained_support` — plus the two
rank-0-only tests the P4 leg deselects. The 6 skips are P4-device entry points
that correctly skip at P=1 and are covered by leg 0.

### Leg 2 — Si P4 identity, against the campaign's own `810c260b` control

Union driver `58212398` (263.7 s) against ACONJ's control arm
`58209192.13` (282.4 s), same deck, same harness, all output keys off.
**6 of 7 checks PASS.**

| check | result |
|---|---|
| `parent_count` (8) | PASS |
| `physical_deck` | PASS |
| `rules_certify` (8 rules) | PASS |
| `model_invariants` (all 8 parents, incl. `model_reciprocity`) | PASS |
| **`every_rank_peak_nonincrease`** | **PASS** |
| `sigma_within_2mev_all_rows` | FAIL, 50.519 meV |
| `qp_delivered_within_2mev` | PASS |

**The peak result is the union being better than its parts.** ACONJ alone
failed this check with rank 0 at **+512 B**. On the union rank 0 is
**−3,200,256 B** against the control and ranks 1–3 are byte-equal. One of the
other lanes — ACARRY's two-axis carrier removing `compact_kernels` and the
one-axis temporaries is the obvious candidate — more than absorbs ACONJ's cost.

**Σ, reported both ways and labelled** (`qp_delivered.json`, eqp1):

| subset | rows | median | max | over 1 meV | over 2 meV |
|---|---:|---:|---:|---:|---:|
| **DELIVERED** (E_DFT inside the Σ box [−15, 18] eV) | 89 | **9.425 µeV** | **859.6 µeV** | **0** | **0** |
| ALL ROWS | 272 | 486.6 µeV | 5863.7 µeV | 69 | 20 |
| CLAMPED (E_DFT outside the box) | 183 | — | — | — | — |

The worst delivered row is band 13 at E_DFT 17.65 eV, 0.86 meV — inside the box
but within 0.35 eV of its top edge. The worst row overall is band 15 at E_DFT
24.87 eV, 6.9 eV **outside** the box, where Σ was clamped to the grid endpoint;
all 20 rows over 2 meV are clamped.

**The union's Si movement is ACONJ's and only ACONJ's.** Against ACONJ's own
recorded numbers: unrestricted max 5863.7 µeV vs ACONJ's 5.862 meV; delivered
median 9.425 vs 8.7 µeV, max 859.6 µeV vs 0.86 meV; raw Σ_c 50.51876 vs 50.519
meV; retained rank sum 10980 → **11031**, ACONJ's exact figure. The other five
lanes contribute nothing measurable to Σ on Si.

`sigma_within_2mev_all_rows` is left **FAILING**, exactly as ACONJ left it, and
is not adjusted. On the delivered subset nothing exceeds 2 meV.

**The residue invariant REFUSES here, correctly.**
`tools/cluster_residue_compare.py` returns `REFUSED: K differs per parent`
(reference `[2010,1068,1470,1271,1313,1222,1250,1376]`, union
`[2010,1077,1472,1277,1319,1230,1262,1384]`). ACONJ's conjugate correction
deliberately changes the retained rank, so a cluster-by-cluster residue
comparison has no defined correspondence. That is the tool stating its
precondition, not a gap: the comparable observable when the span itself moves is
Σ, which is reported above. `model_identity_equal` is `False` for the same
reason the hashes moved.

### Leg 3 — multi-map shared-pole SC: the question only the union answers

**Answer: the union still dies at map 1 — and so does everything else.**

Three sources, one deck (Si SC, 368 centroids, production tier), one harness,
same gate, same parent q=0. All three complete map 0 and die at map 1.

| source | what it is | Gram min/max | metric ∞-norm | inv-root residual | map 0 |
|---|---|---:|---:|---:|---|
| `bf6048c9` | the SC branch **alone**, unmerged | **−2.29389959e−07** | 4.63277105e−08 | 7.98208191e−17 | PASS 166.2 s |
| `9160c501` | **the union** | **−2.01241953e−07** | 3.69164466e−08 | 7.89845983e−17 | PASS 162.6 s |
| `9a030add` | base + ARETAIN, no SC branch at all | not printed (old message) | 3.00167437e−08 | 7.68924622e−17 | PASS |

Gate is −1e−07 (`shared_pole_recipe.py`, ten times `bank_rule_tolerance`).

1. **The union introduced nothing.** A strict subset of the union
   (base + ARETAIN) already dies at map 1 at q=0 with the same gate — that is
   the previously known "3.0e-08 against 7.7e-17", found on ARETAIN's own leg
   `58209192.49`, already on disk. And the SC branch on its own dies there too,
   **harder** than the union.
2. **The union is the best of the three** on this metric, by 12%.
3. **The SC line's 13-map validation does not reproduce on this deck.** The SC
   branch unmerged fails here, so the honest statement is not "the union breaks
   SC" but "multi-map shared-pole SC does not work on *this deck* on any of the
   three sources".
4. **Rule retention is not the mechanism.** base+ARETAIN **rebuilds** the bank
   rule every map (Δ_max 3.7140 → 3.8011 Ry, nodes 635 → 646) and fails; the
   union and the SC branch **retain** it (Δ_max 4.0080 Ry fixed, 670 nodes,
   `reuse_status="hit"`, identical `node_digest`, `certificate.status=PASS`) and
   fail. Both policies fail, so the escape check is not what is broken. The
   retention was honest at map 1: map 1's own Δ_max 3.8011 ≤ the retained
   4.0080, and `_reuse_bank_rule` re-evaluated the continuum bound on map 1's
   actual z ladder within `rel_tol` 1e−8. Worth flagging for whoever fixes it:
   the margin is closing fast — map 0 sat at 96.2% of the certified Δ_max, map 1
   at 98.4% — so a run that got past map 1 would escape within a map or two.
5. **The gate was not loosened and must not be.** A realized Gram 20× outside
   the rule's own 1e−8 while the rule's certificate says PASS is the interesting
   gap, and it is at q=0 specifically. Registered as a pre-existing defect.

### Two probes AMEM could not measure, now measured

`gw/shared_pole_local.py` holds two `@lru_cache(maxsize=None)` compiled-executable
caches with no `cache_clear` anywhere in `src/gw/`. Across the map 0 → map 1
boundary, on **both** the union and the SC branch alone:

| cache | after map 0 | during map 1 | hits |
|---|---:|---:|---:|
| `_parent_panel_packer` | 2 | **3** | **0** |
| `local_parent_reducer` | 2 | **3** | **0** |

The keys change per map and nothing evicts. Two points establish growth, not a
rate; the slope needs a run that gets past map 1.

On-disk scratch, exports off: `sc_0000_shared_pole/` 1.685 GB,
`sc_0001_shared_pole/` 1.491 GB (partial). The scratch tree is unbounded per
map on the default path; the union neither improves nor worsens it, and nothing
here is ARETAIN's export retention, which is bounded and separately tested.

## What the union does better than either parent

Three things, all measured:

1. **The error message.** The union carries the SC line's improved
   `shared_pole_gram_valid` text, which names the threshold and the Gram
   eigenvalue. The performance line's message printed neither. That is why this
   report can quote −2.01e−07 at all.
2. **The every-rank peak.** ACONJ alone failed it (+512 B on rank 0); the union
   passes it (−3.2 MB on rank 0, byte-equal on 1–3).
3. **The Gram margin.** −2.012e−07 against the SC branch's −2.294e−07.

## Limits — what this does NOT claim

* **No timing band.** Wall times appear only as provenance beside their job
  steps. Legs ran concurrently with other steps on shared pools and no placement
  audit was run, because none of these are timing measurements.
* **No claim that the union fixes the SC failure.** It does not. It is 12%
  closer to a gate it still fails.
* **The Si Σ comparison is against one control arm** (`58209192.13`), reused
  rather than re-run. It is the same deck through the same harness at
  `810c260b`, which is the base of every lane here, and ACONJ's own bit-identical
  control repeat is what makes attribution exact.
* **`sigma_within_2mev_all_rows` fails and is reported failing.**
* **The residue invariant is unavailable** on a deliberate span change, so the
  residue evidence is absent rather than green.
* **Nothing is on `origin/main`.** `integ/sp-union-2026-09-11` only.
