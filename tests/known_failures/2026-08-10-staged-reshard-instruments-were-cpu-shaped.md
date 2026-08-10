# AMENDMENT — the two staged-reshard instruments were never true on a GPU, and one of them could not go red anywhere (2026-08-10) — **FIXED**

This retires **P4** of the Perlmutter census (`test_staged_reshard::test_red_twin_the_unstaged_chain_emits_the_spmd_warning`, filed as class (d) environment) and adds the row that should have sat beside it, because the same file carried a second instrument with the same disease and that one was reporting green.

Both were invisible until the peer lane `fix/conftest-mesh-cells-2026-08-10` made mesh cells actually run: `tests/conftest.py` pins every test process to one GPU, so `jax.device_count()` is 1 and the file's `skipif(device_count < 4)` skipped in every suite run on every node. The cells passed only when someone ran the file by hand on an emulated CPU mesh — which is exactly the arrangement in which neither defect is visible.

## Found state — what each instrument actually did on CUDA

| instrument | what it asserted | what it did at `num_partitions=4` on a real 2×2 A100 mesh |
|---|---|---|
| `test_hlo_pin_two_all_to_all_zero_all_gather` | the staged module carries exactly 2 `all-to-all` and 0 `all-gather`, counted as `all-to-all(` in the optimized HLO | **counted 0 of 2, while both exchanges were present and correct.** XLA:GPU rewrites every collective into an async `all-to-all-start(` / `all-to-all-done(` pair, so the synchronous spelling the counter was written against does not occur on the platform production runs on. The module really contains `all-to-all-start` on `replica_groups={{0,2},{1,3}}` (the `x` exchange) then on `{{0,1},{2,3}}` (the `y` exchange, node-local pairs as the primitive's §3.2 promises), one shard of payload each, and zero all-gathers. **The pin was blind to its own subject**; the peer lane's reading that the collectives "fuse into `%fused_transpose`" is a mis-attribution — `%fused_transpose` is there, but it is the tiled `all_to_all`'s LOCAL data rearrangement, carrying the `op_name="jit(_reshard)/all_to_all"` metadata XLA propagates onto ordinary ops (12 such mentions), and it moves no bytes off the device |
| `test_red_twin_the_unstaged_chain_emits_the_spmd_warning` | the UNSTAGED chain emits the compiler's `Involuntary full rematerialization` warning, so that the staged path's zero count means something | **emitted nothing — and not only on CUDA.** Sixteen compiles of the chain that does replicate, at 30 KB / 7.2 MB / 67 MB of replicated payload, on CUDA and on 4 emulated CPU devices, under Shardy and under the legacy GSPMD partitioner (`JAX_USE_SHARDY_PARTITIONER=false`, verified per-arm by the presence or absence of XLA's own "Using Shardy for XLA SPMD propagation" line), with `TF_CPP_VMODULE=spmd_partitioner=3`: **not one hit**. It is not a rewording either — `strings` finds the literal exactly once in the shipped `libjax_common.so` and once in `xla_cuda_plugin.so`. The pattern simply no longer reaches the branch that logs it |

The hazard itself is untouched and still measurable, which is the reason the second row is a repair rather than a deletion: at the deck geometry the unstaged chain still names a `c128[64,84,84]` full-batch buffer on one device and still asks for **10,838,532 B of temp against a 1,806,336 B shard — 6.0×**, on GPU and on the emulated CPU mesh alike.

## Fixed state — both instruments re-written GPU-first

Both are now expressed in the two things that mean the same on every backend: the collective's **issue site** and its **payload**. One helper, `_assert_movement_only`, carries all three clauses, and the red twin puts the unstaged chain through that same helper rather than through a paraphrase of it:

1. **schedule** — exactly two `all-to-all` issue sites and no other collective. `-start`/`-done` are folded back onto the collective they implement, so the count is 2 on CUDA and 2 on CPU. Staged `{'all-to-all': 2}`; unstaged `{'all-gather': 2}`.
2. **residency** — no per-device buffer larger than one shard. Staged: exactly one shard on both platforms. Unstaged: a full `c128[64,84,84]`.
3. **accounting** — `memory_analysis().temp_size_in_bytes` below the size of the whole array, which is what replicate-then-partition costs. Staged 1.00× one shard on GPU and 2.02× on the emulated CPU mesh against a threshold at 4.0×; unstaged 6.0× on both.

The stderr grep is **retired**, with the dead proof written into the replacement cells' own docstrings. Its two cells become the same three clauses taken as a **cold compile in a fresh process at the deck geometry** `c128[64,84,84]` — the shape `common/staged_reshard`'s own SPMD diagnostic was written from — because an instrument is certified at the geometry it is consumed at, and a 30 KB toy array is small enough that a backend could reasonably choose to replicate it and be right. The child now also refuses to be emulated: it reports its device count and platform, and the parent asserts both against its own.

Two node ids change, and nothing else in the file does (14 cells before, 14 after):

| was | is |
|---|---|
| `test_red_twin_the_unstaged_chain_emits_the_spmd_warning` | `test_red_twin_the_unstaged_chain_replicates_at_the_deck_geometry` |
| `test_the_staged_chain_emits_no_spmd_warning` | `test_the_staged_chain_holds_one_shard_at_the_deck_geometry` |

## The FALSE arms fire on GPU

A repaired gate that has not been shown failing is the same defect wearing a fix, so three separate breaks were driven on the real mesh:

* **the primitive mutated to replicate-then-slice** — stage 2's `all_to_all` replaced by `all_gather` + `dynamic_slice`, the same values by the historical hazard. The pin goes red with `got {'all-to-all': 1, 'all-gather': 1}`, and every red twin stays green.
* **the input sharding perturbed** — the same primitive handed a `P('x', None, None)` input instead of the face layout, so the compiler inserts its own reshard. Red on the schedule clause.
* **the cold-compile arm, on a mutated TREE** — `_compile_in_subprocess` prepends the repo's own `src/` ahead of the inherited `PYTHONPATH` (deliberately, so the campaign harness's wk_REL copy cannot gate a different tree), so a `PYTHONPATH` mutation cannot reach the child; the tree itself has to be mutated to drive that arm red.

## One consequence outside this file, registered rather than fixed

Any A/B that quotes **"involuntary-remat lines N → 0"** is quoting an instrument that now reads zero on both arms. That includes the MEASURED table in `common/staged_reshard`'s module docstring (`P=64 … 64 → 0`, `P=16 … 16 → 0`), taken on Frontera's XLA in 2026-07-31 where the line was live. Those numbers are **history, not a check that can be re-run**, and the docstring now says so at the point of quotation. Nothing in `src/` greps the string at runtime — every remaining mention is a comment.

## Landing order — this branch depends on the peer's conftest

`pytestmark` is `pytest.mark.mesh(4)`, byte-identical to `fix/conftest-mesh-cells-2026-08-10`'s hunk, and the `mesh(n)` marker row in `pyproject.toml` is that branch's current text verbatim, so the two merge without a conflict. But the marker only *skips or reroutes* once that branch's `tests/conftest.py` is present: **on `main` alone the cells would run in the one-GPU pinned process and error in `_mesh()` rather than skip.** Land the peer's branch first, or land the two together.

## Evidence

`/pscratch/sd/j/jackm/reshard_instr_0810/` — `arm.py` and `leg1..5.sh` with `leg*.log`; `_truth3/` (the 18-arm GPU/CPU × Shardy/GSPMD × three-geometry truth table, each arm's optimized HLO, stderr and JSON); `_gate4/` (TRUE arms 14 passed on 4×A100 by direct invocation, the two FALSE arms, the emulated-CPU parameterisation, the `strings` sweep); `_gate5/` (the cold-arm FALSE arm on a mutated tree, and the census set-diff); `wt`, `wt_base`, `wt_after` (the branch, and the before/after census trees).
