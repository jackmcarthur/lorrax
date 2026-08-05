# PROPOSED amendment to `docs/dev/staged_reshard_primitive.md` §3.5

Owner follow-up #3, 2026-07-29. Wording proposed, not landed — the
coordinator lands it.

**Where:** append to §3.5 ("The impl=mpi world-collective-first warm-up
contract"), after the existing paragraph ending "...deletes the contract
entirely (explicit communicator lifecycle)." The existing text is
history and should stay; this is an amendment, not a rewrite.

**Why it matters here specifically:** §5 of this same document names the
BSE tree as the intended second consumer, and the adoption path there
runs through standalone drivers and microbenches. The paragraph being
amended is the one that tells such a consumer it is covered. It is
measured not to be.

---

## Proposed text

> **AMENDMENT (2026-07-29): the helper is measured INSUFFICIENT for a
> standalone consumer.**
>
> `ensure_grouped_collectives_ready()` implements the world-collective-first
> contract by calling `common.collectives.barrier`, i.e.
> `multihost_utils.sync_global_devices`. In the installed jaxlib that
> lowers to a `jit(_identity_fn)` all-gather over an internally
> constructed `('processes','local_devices')` mesh — a different device
> assignment from the caller's. **It does not make a standalone driver's
> grouped `psum_scatter` work.**
>
> Measured (job **7879485**; `wk_REL/zproj_mpiprobe.{py,sbatch}`; P=4,
> 2 nodes, one FRESH PROCESS per variant, the grouped chain being
> byte-for-byte the one `common.zeta_projection` issues, at tiny shapes
> so neither size nor memory is in play):
>
> | variant | transport | warm-up before the first grouped collective | verdict |
> |---|---|---|---|
> | `mpi_none` | mpi | none | **FAIL** |
> | `mpi_sgd` | mpi | `sync_global_devices` — what the helper does today | **FAIL** |
> | `mpi_psum` | mpi | `lax.psum` over BOTH mesh axes inside `shard_map`, on the caller's own mesh | **FAIL** |
> | `mpi_ag` | mpi | world all-gather on the caller's own mesh | **FAIL** |
> | `mpi_both` | mpi | psum + sgd | **FAIL** |
> | `gloo_none` | gloo (ib0 pin) | none | **PASS** |
> | `gloo_sgd` | gloo (ib0 pin) | `sync_global_devices` | **PASS** |
>
> All five mpi variants die with the §3.5 error
> (`MPI: Communicator requested from a thread that is not the one MPI was
> initialized from`). Note that `mpi_psum` is the collective the 7878883
> probe itself used, issued on the caller's mesh — so the discrepancy is
> not "the helper picked the wrong collective".
>
> **This does not contradict production.** `gw.gw_jax` runs grouped
> collectives under impl=mpi at P=64 successfully (job 7879010, 8/8
> passes rc=0). The contract is therefore satisfiable — by something in
> the full driver process that a standalone module does not reproduce.
> **What that something is has not been identified.** Until it is, the
> honest statement of the contract is:
>
> * **impl=mpi is verified only for the full `gw_jax` process.** A
>   standalone consumer — a test, a microbench, a BSE driver, anything
>   that builds its own mesh and calls the primitive directly — must not
>   assume the warm-up helper covers it, and must gate rather than trust.
> * **The helper is retained** (it is cheap and correct as far as it
>   goes) but its docstring should stop promising sufficiency.
>
> **Do NOT read this as "standalone consumers should use gloo."** The two
> transports have *disjoint* known-bad regions and neither is a safe
> default:
>
> * gloo/ib0 was found dying reproducibly at P=64 under the distributed
>   tiers (2 reps, warm and cold), which is part of why the campaign's
>   certified stack moved to impl=mpi;
> * gloo/ib0 nevertheless carries this staged reduce-scatter chain
>   cleanly — memo §4.3 at P=64 (job 7878883 step 1c, rc=0, full suite),
>   and `common.zeta_projection`'s own sweep at P = 4/16/64/144 (jobs
>   7879499/7879504), where the projected result is bit-identical across
>   a 36× range of P;
> * but a **silent, non-reproducible wrong answer** was observed once on
>   gloo at P=4 (job 7879491: a non-Hermitian `W_S` with `Im Σ = 6.25e+02`
>   where the algebra forces exactly 0; the identical configuration re-ran
>   clean in job 7879499). Incidence measured over 100 further executions
>   in job **7879519** — see `wk_REL/docs/zeta_projection_notes.md` §6.
>
> The operational rule this implies, and which the primitive's consumers
> should follow: **choose the transport per context, and carry a cheap
> invariant that the collectives cannot fake.** For a congruence
> `ψ† O ψ` with Hermitian `O` that invariant is free — the result is
> Hermitian for *any* ψ, so a Hermiticity check tests only the machinery.
> `common.zeta_projection` runs exactly that at every production-scale
> point, plus a parity check against a structurally different collective
> pattern, and both are one jitted reduction each with no gather.

---

## Companion one-line change

`src/common/contract_bands.py`, `ensure_grouped_collectives_ready`
docstring — the sentence

> "...so every consumer (tests, benches, future BSE drivers) inherits the
> contract instead of rediscovering the failure."

should become

> "...so every consumer inherits the *world-clique* warm-up. **This is
> necessary but measured NOT sufficient for a standalone consumer** (job
> 7879485: five warm-up variants, all fail; see
> `docs/dev/staged_reshard_primitive.md` §3.5 amendment). Production
> `gw_jax` does run grouped collectives under impl=mpi successfully, so
> the gap is specific to the standalone path and is unexplained."

I have deliberately NOT edited `contract_bands.py` or the primitive doc
myself: `contract_bands` is the shared GW/BSE entry point and the ladder
workstream is live against it.
