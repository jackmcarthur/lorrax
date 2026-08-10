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
  `tests/fast_gate.py`'s roster, `mesh(4)`-marked, census-class. Each decked driver's
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
