# AMEM — lane note, 2026-09-11

Branch `lane/sp-amem-2026-09-11`, base **`e769b4fe`**, worktree
`/pscratch/sd/j/jackm/wt_sp_amem`. **No source change.** Diagnosis lane.

No pool was allocated and none was requested: every number comes from
artifacts that were already on disk. Extracted by
`runs/frequency_integration_sandbox/368_amem_20260911/evidence/collect.py`
into `evidence/amem_compile_mechanism.json`, so it survives a scratch purge.

Full write-up: `reports/shared_pole_push_2026-09-07/amem/report.md`.

## The question

> "the entire SC loop should be under one jit and memory leaks shouldn't be
> possible"

**False.** `gw_iteration_map` (`src/gw/sc_iteration.py:2634`) is an
undecorated Python function; the iteration driver is a Python `for`
(`:4610`, or `src/mixing/acceleration.py:968` for the default rCROP, which
calls the map twice per iteration). One SC map crosses ~25 named jit
boundaries and eight Python loops that wrap jitted calls. On the
shared-pole path the tau kernel is *deliberately* not jitted and not cached
(`src/gw/ppm_tau_kernel.py:1024`, `:1038`, `:1059-1061`) — on the MPA path
the same work is one jit. Materialisation between stages is the design, not
an accident, which is why the owner's inference does not carry.

## The 2.33 s/parent mechanism — settled

**One executable per q-batch containing one lowered `lax.switch` branch
body per parent.** Not N executables, not tracing, not dispatch.

| fact | evidence |
|---|---|
| **4** backend compile events in `spole.gram_reduction` on an 8-parent deck **and** a 29-parent deck | `trace_receipt.json`, 58209192.16/.19, 58212398.26/.29 |
| Those 4 are two large + two 0.15 s, one pair per q-batch (Si 9.611/8.677; Na 35.004/31.700) | `attribution_rank0.json`, same legs |
| The switch key `dict.fromkeys(parent_extents)` (`src/gw/shared_pole_local.py:205`) dedups nothing: extents are distinct for 8/8 Si and 28/29 Na parents | each leg's own `constructor_receipt.json`, `roles[].carrier_width` |
| Large-event compile / **branch** count = **2.286 s** (Si) and **2.300 s** (Na), 0.6 % apart | `evidence/amem_compile_mechanism.json` |
| Dumped HLO carries **4 `branch_computations` per `jit_execute` module**, two modules, each branch with its own Newton-Schulz loop at its own pencil width | 58209192.30, `361_acarry_20260911/09_hlo_si_p4b/` |

**The scaling, which is the new part:** ~2.3 s of XLA lowering **per
irreducible q-point**, linear in q, paid before any physics runs and paid
again every SC map. Si 18.6 s and Na 67.0 s are the small cases; ~300
irreducible q is ~11.5 min in one band, per map.

**This re-opens the cost, not the closure.** The three grounds of the
original closure stand, and `src/gw/shared_pole_local.py:183-185`
(cuSolver's `eigh` on artificial zero blocks) is a real constraint. Nothing
here is licence to pad.

**Correction to a measurement reading, not to a repair.** AFIXES ruled
`lax.switch` out from `pencil_side` having two distinct values per deck.
That field is `q_receipts[i].constructor.capacity.price.pencil_side` — the
capacity ledger's admitted **padded batch width** (`shared_pole_constructor.py:59`,
called `:727`), batch-constant by construction. It is not
`reduction['pencil_side']` (`shared_pole_local.py:171`). No AFIXES repair
depends on it.

## Retention items, registered not fixed

| Site | What is retained | Growth |
|---|---|---|
| `src/gw/shared_pole_local.py:124` | the heaviest XLA module on the path (7252 instr., N lowered pencil solves), one per key | per q-batch x per SC map, **forever** — no `cache_clear` in the tree |
| `src/gw/shared_pole_local.py:87` | `_parent_panel_packer` executable (2162 instr.) | same count, same permanence |
| `services/distrib_la/.../matmul.py:48`, `_batch_reshard.py:43` | one staged-GEMM module per distinct operand shape, at **module scope** | per q x per SC map, forever |
| `src/gw/w_isdf.py:3437` | 8 device tables incl. sharded `L_table`, keyed on `id(plan)` with no reference held | per plan object; **stale-hit correctness hazard** |
| `src/gw/sc_iteration.py:1983` `_PSI_G_CACHE` | multi-GB psi(G) | bounded today; one moved band window frees nothing |
| `src/gw/mpa/sigma.py:342` `compact_kernels` | compiled compact panels — **still present here**; ACARRY's removal is on `lane/sp-acarry` `cc6c184d`, not on this line | per tau node, released at `:1462`; the ledger row at `:396` is not |
| `src/gw/shared_pole_recipe.py:217` | `CapacityLedger` has `reserve` and no release; `shared_pole_store.py:678` mints a row **and prints a JSON line** per face read | per tau x panel, rebuilt per map |

ARETAIN's two rows (the `sc_NNNN_shared_pole/` directories; the map-1 Gram
gate) are already registered and are **not** re-registered. Two per-map
on-disk namespaces neither lane had named are in the report: the
`sigma_quadrature_rules/request_*/` directories
(`src/gw/sigma_box_plan.py:87`, `:393`) and the `eqp{0,1}_iterNNNN.dat`
snapshots, whose only cleanup runs once per **run**, at entry
(`src/gw/sc_iteration.py:4528`).

## Cheapest measurement left

`cache_info()` on `local_parent_reducer`, `_parent_panel_packer` and
`local_model_checks` after N SC maps. It would settle whether the reducer
key actually changes per map — argued structurally and supported by the
bank-receipt diff, **not measured**, because the base cannot complete map 1.
