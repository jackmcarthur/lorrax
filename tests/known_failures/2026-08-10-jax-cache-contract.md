# THE JAX CACHE CONTRACT — found state, fixed state, and what is red-listed (2026-08-10)

Branch `feat/cache-contract-2026-08-10`, off `origin/main` @ `aca88841`.
Evidence directory: `~/lorrax_cache_contract_2026-08-10/`.

The owner's ask, near-verbatim: *"better gates to make sure jax correctly caches like
100% of the time and doesn't cause us PITAs."* This file is the ledger half of the
answer — what was true before the branch, what is true after it, and which sites were
deliberately not fixed.

## The found state (2026-08-10, at `aca88841`)

**There was no standing cache gate at all.** Both of the campaign's cache-PITA classes
had been root-caused and fixed at the single site where each was first seen, and
neither had a gate that would catch the next one:

* **Class A, VETO** — a program that cannot persist because it carries a host callback.
  Found in `jit__full_run`, fixed by the sink pattern, gated only by three cells in
  `tests/test_bse_nontda.py` that watch the Krylov jit specifically
  (`FIX_warmcache.md` §4). Nothing watched any other driver.
* **Class B, KEY DIVERGENCE** — rank-dependent static arguments or shapes producing one
  program per rank. Found in jax's own `jit__multi_slice`, canonicalized at `d6303e61`
  and gated by `tests/test_compile_cache_shard_slice.py`, which watches that one patch
  (`FIX_multislice_cachekey.md`).

**FIVE KNOWN SIBLINGS of class B were open**, listed in `FIX_multislice_cachekey.md`
§6.1 and unfixed at `aca88841`:

| # | site | mechanism |
|---|---|---|
| 1 | `src/bse/vq_interp.py` | `lo = rank*n_kept//nranks` reaches a memo key, a jit SHAPE (`jnp.arange(lo, hi)` into `_mbz_draw_u` and `wrap_points_to_voronoi`), and an `if hi > lo:` under which surplus ranks compile nothing |
| 2 | `src/gw/kin_ion_io.py` | ragged band-chunk bounds `nocc*i//n_bchunk`; the rank holding the short chunk traces `to_box` and the density IFFT at a band extent no peer compiles |
| 3 | `src/centroid/charge_density.py` | ragged last chunk `min(lo+nb_chunk, b_hi)` owned by one rank; `to_rbox` keys its kernel cache on `psi.shape` |
| 4 | `src/common/collectives.py` | `local_share`'s sanctioned EMPTY share: at `world > n_items` a rank compiles none of `per_k`'s modules |
| 5 | `src/ffi/__init__.py` | `ffi_dial_key`'s per-process env dials reach four GW kernel-factory cache keys and change the emitted HLO BODY; nothing compared them across ranks |

## The fixed state

**Gates landed.**

* `tests/test_jax_cache_contract.py` — the contract. Parametrized over every driver in
  `tests/fast_gate.py`'s roster, `procs(4)`-marked, census-class. **`procs(n)`, not the
  `mesh(n)` that landed at `a6b87fa9`**: `mesh(n)` widens ONE process to n devices, where
  `jax.process_count()` is 1, the compile-cache agreement layer is never installed and
  `ArrayImpl._multi_slice` is never reached — so every defect this contract gates is
  invisible to it, as that marker's own definition says. A separate marker rather than a
  redefinition, because the cells that pass under `mesh(n)` depend on what it means. Each decked driver's
  Si smoke deck runs TWICE at P=4 against a fresh cache directory and the second run
  must show `xla_compiles=0`, `vetoed=0` on every rank AND an identical cache-key set
  across ranks. Undecked drivers are named and skipped with `fast_gate.UNDECKED`'s own
  reason. Red twins: a deliberately vetoing program (a `jax.debug.callback`) must fail
  the veto arm and NOT the symmetry arm; a deliberately rank-static program must fail
  the symmetry arm.
* `tests/cache_key_lint.py` + `tests/test_cache_key_lint.py` — the class-B AST lint,
  five rules, failing by `file:line` with the canonicalization recipe in the message.
* `src/common/jax_compile_cache.py` — `LORRAX_JAX_CACHE_KEYDUMP`, the per-rank cache-key
  set. This is what makes the symmetry arm possible at all:
  `LORRAX_JAX_CACHE_EXPLAIN` prints only the keys that MISSED, so a run in which every
  rank hits its own private entries prints nothing on any rank.

**Four of the five siblings are fixed** (see the branch's commits for the per-site
argument and the bit-identity gate):

| # | site | canonicalization |
|---|---|---|
| 1 | `vq_interp` | uniform `chunk = ceil(n_kept/nranks)` with a host-side mask; `lo` leaves the memo key (it is `rank*chunk`, constant within a process); the `if hi > lo:` guard is gone, so surplus ranks compile what their peers compile |
| 2 | `kin_ion_io` | `rho_work_items` snaps `n_bchunk` DOWN to a divisor of `nocc`, so every band chunk is the same width. All three pinned contracts still hold (cover-exactly-once, serial-sweep parity at `world <= nk`, balance within one item) |
| 3 | `charge_density` | equal-width band windows that OVERLAP rather than shorten, plus a 0/1 band mask; every rank runs `ceil(n_windows/world)` rounds of one program and a rank with no work runs one fully-masked round |
| 5 | `ffi` | `FFI_DIAL_ENV` declared in `src/ffi/__init__.py` and folded into `jax_compile_cache.RANK_FINGERPRINT_ENV`, so the existing agreement compares the dials across ranks and turns the cache off LOUDLY on a mismatch |

## RED-LISTED — one condition, three spellings

**Sibling 4, the sanctioned empty share**, is red-listed rather than forced. Rows are in
`tests/cache_key_lint.py::ALLOW`, all dated 2026-08-10, and the contract reads the same
rows.

* `src/common/collectives.py` — THE CARRIER. `sweep_local_k` runs `per_k(ik)` once per
  item of `local_share`'s round-robin share; at `world > n_items` a rank draws nothing
  and compiles nothing while its peers compile the whole body.
* `src/gw/kin_ion_io.py` — A CONSUMER. Its ragged-extent half IS fixed above; what
  remains is only the empty share, which belongs to the carrier.
* `src/psp/run_nscf.py` — **found by the lint, not by the campaign.** It spells the same
  partition inline as `if ik % n_proc != rank: continue` around the per-k Davidson
  solve. Every k has the same shape, so there is no ragged half here at all; the only
  divergence is `nk < n_proc`, i.e. exactly the empty share.

**Why not forced.** The empty share is sanctioned by the design in
`common/collectives.py`'s own note ("P=64 on a 16-k deck leaves `world-nk` ranks with an
empty list") and pinned as CORRECT by two cells —
`test_kin_ion_padded_gvectors::test_sweep_local_k_empty_rank_keeps_the_collective_shape`
and `test_collectives_distribution`'s round-robin pin. The COLLECTIVE is already
protected (`item_shape` is required precisely so an empty rank contributes the right
shape); the COMPILE is not. Closing it needs either a padded work list with a sentinel
item every consumer must be taught to execute, or a per-consumer warm-up trace — a
second, parallel notion of "the work list". Either is a contract change to
`common.collectives` plus edits in every consumer, and either invalidates the two
pinning cells. That is a design decision with an owner, not a mechanical canonicalization.

## §6.1's OPEN QUESTION, per site

`FIX_multislice_cachekey.md` §6.1 answered "is a divergent key here a HANG or only
cache-key pollution?" for `_multi_slice` alone — `parameter -> kLoop fusion(slice)`, no
autotune candidates, so `AutotunerPass` has no work to shard — and said explicitly that
it does not generalise, "because several of them compile real physics kernels with gemms
in them".

XLA's autotune candidates are GENERATED FROM dots and convolutions. A module whose HLO
contains neither has no candidates on **any** backend, so the *benign* half of the answer
is device-independent and was measured here, on CPU, at the two extents a ragged slab
actually produced. `fft` is counted and deliberately NOT treated as a candidate: cuFFT
picks its plan inside the library and never enters XLA's multi-process autotuner. The
*live* half — how many candidates, and how long the handshake takes — is a GPU question
and is DEFERRED (below).

Evidence: `~/lorrax_cache_contract_2026-08-10/autotune_census_cpu.json`, produced by
`autotune_cpu.py` in the same directory. Two controls ran in the same pass: a
slice-only module (0 candidates, reproducing §6.1's finding) and a plain GEMM (3), so
the counter is known to be able to answer both ways.

| site | divergent module | dot | conv | fft | verdict |
|---|---|---:|---:|---:|---|
| 1 | `jit__mbz_draw_u` (threefry + vmap) | 0 | 0 | 0 | **BENIGN** — no candidates at all, exactly `jit__multi_slice`'s case |
| 1 | `jit_wrap_points_to_voronoi` | 2 | 0 | 0 | **BENIGN, and the reason is specific**: both dots are `f64[343,3]`, i.e. the `(2·nmax+1)³ = 343` candidate-shift × lattice matmul. That shape is a function of `nmax` ONLY and does not move with the rank slab, so ranks compiling different modules still present the autotuner the SAME dot. Divergent module, rank-INVARIANT candidate set. |
| 2, 3 | `_valence_density_kernel` / `to_rbox` | 0 | 0 | 6 | **BENIGN as a deadlock risk.** FFT-bearing, and the FFT batch axis IS the divergent one — but an FFT is not an autotune candidate. The cost was a real recompile and a per-rank cache key, not a hang. |
| 4 | `_dipole_block` via `sweep_local_k` | — | — | — | **LIVE, NOT MEASURED.** FFT plus band–band contractions, i.e. GEMM-bearing by construction, and this is the site still RED-LISTED. Needs the deferred GPU leg to quantify; the qualitative answer (a GEMM-bearing module with a rank-dependent compile is a live autotune-divergence hazard) already follows and is the reason the red-list row matters. |
| 5 | Σ_kij / Σ_τ / cohsex Σ / χ⁰ | — | — | — | **LIVE, NOT MEASURED — but CLOSED by the fix.** These carry the largest autotune candidate sets in the tree (`contract_bands` right-GEMMs). The divergence is now refused symmetrically by the fingerprint, so the hazard no longer has a route; measuring its size would be measuring a state the branch removed. |

Read the table with its bound: it says these modules have no autotune candidates and that
two controls behaved as expected. It does not prove XLA never blocks for a
non-autotunable module — the same bound §6.1 put on its own answer.

## DEFERRED P=4 LEGS (owner priority directive, 2026-08-10)

The MPA acceptance chain took fleet-wide GPU priority mid-lane. Every leg below is
ready-to-run and none was submitted; the branch is **contract-armed-but-unrun** on GPU,
which the contract itself reports honestly (the deck arms skip naming the four-GPU rule
rather than passing vacuously). One `lx run` had been submitted and was **killed by PID
before it placed any step** (`squeue --me` showed no `Xg4` step at any point); no
allocation was created.

Worktree already staged at `/pscratch/sd/j/jackm/cachecontract_0810/wt`; evidence dir
`/pscratch/sd/j/jackm/cachecontract_0810/_reports/`. Set `LX_BASE_MODULE=lorrax_J070`
first — without it the leg gets the wrong jax.

**LEG 1 — THE CONTRACT, all decked drivers, P=4.** The headline gate. One combined
dispatch (doctrine rule 2); ~20 min, dominated by the two BSE stages run twice.

    cd /pscratch/sd/j/jackm/cachecontract_0810/wt && git fetch origin \
      feat/cache-contract-2026-08-10 && git checkout --detach FETCH_HEAD
    LX_BASE_MODULE=lorrax_J070 lx run -N 1 -G 4 -n 1 --wait 900 -- \
      bash /pscratch/sd/j/jackm/cachecontract_0810/leg.sh

`leg.sh` is in the evidence dir and sets `LX_MESH4_DECKS=1`. GREEN means, per driver and
per stage: `xla_compiles=0 vetoed=0` on all four ranks and one shared cache-key set.
**Expect `mesh_launch` mode `local-gpu`** — four processes, one A100 each, on the node
the step already holds; a nested `srun` is not available inside a step and the mode line
is printed in every failure message. The lint half of this file already ran green on GPU
(step `lx-Xg4-154342`, 24 passed / 10 skipped, `_reports/contract.xml`); it is the deck
arms that are owed.

**LEG 2 — BIT-IDENTITY per fixed sibling, P=4 vs P=1.** Sites 1 and 3 regroup a floating
sum across ranks, so the claim is agreement at the tolerance each site already documents,
not bit-identity — and it has to be measured, not asserted:

    lx run -N 1 -G 4 -n 1 -- python -m gw.gw_jax -i cohsex_si_fast.in   # in si_cohsex_debug
    lx run -N 1 -G 1 -n 1 -- python -m gw.gw_jax -i cohsex_si_fast.in

and diff `eqp*.dat`. Site 2 (`rho_work_items`) only changes behaviour at `world > nk`, so
its arm needs a deck with `nk < 4` or an explicit `world` override. Site 5 changes no
arithmetic at all and needs no bit-identity arm.

**LEG 3 — the autotuner census on GPU.** `~/lorrax_cache_contract_2026-08-10/autotune_probe.py`
is the GPU version of the CPU census above: same modules, but counting
`__cublas$`/Triton/`__cudnn$` in the after-optimizations HLO. It answers the two rows
marked LIVE, NOT MEASURED.

    lx run -N 1 -G 1 -n 1 -- python autotune_probe.py

**LEG 4 — default gate + census zero-delta.** The branch touches `src/bse`, `src/gw`,
`src/centroid`, `src/ffi`, `src/common`, so the default gate selects every driver:

    lx test              # the default gate
    lx test --census     # against a same-day origin/main run, set-diff by node id

Owed because three of the touched files cannot even be IMPORTED on WSL — `src/ffi/gate.py`
refuses without the host `.so`, which is deliberate — so no off-cluster run can speak for
them. The pure functions inside them ARE gated off-cluster, by
`tests/test_cache_sibling_canonicalization.py` (46 cells), which lifts them out of the
module precisely to dodge that import.

## OWNER ROWS

1. **`src/psp/run_nscf.py:321` is a SIXTH class-B site** and was not on the campaign's
   list. Same fix shape as `charge_density`'s (uniform rounds, masked surplus). Raised
   here rather than fixed because it is the same red-listed condition and it belongs
   with the ruling on sibling 4, not ahead of it.
2. **The `if jax.process_index() == 0:` class.** While the lint was being tuned it was
   measured with a one-level taint, and reported **84** findings on this tree against
   the 8 the two-level taint reports. The large majority of the difference is the
   rank-0-only writer/logger idiom guarding a region that also compiles something. That
   IS a hazard of the same family, it has roughly twenty sites, and no owner ruling
   exists on it. The lint deliberately does NOT fire on it (a bare rank id is not a work
   partition; only arithmetic or striding on one is), and the class is recorded here
   rather than silently dropped.
