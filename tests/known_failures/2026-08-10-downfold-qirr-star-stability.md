# AMENDMENT — THE DOWNFOLD ON `q_irr`: ONE REFUSAL REMOVED, ONE CONDITION MEASURED (2026-08-10)

**Two rows, and they point opposite ways.**  The owner's instruction was that
downfolding "should only be done on `q_irr` anyways and it should unfold just
fine because of that".  Half of that is right and was being refused for no
reason; the other half is blocked by something nobody had measured — the
centroid selection breaks the star symmetry, so a downfolded child has no
unfold tables and cannot be stored on the wedge as it stands.

| item | mechanism, at this tree | disposition |
|---|---|---|
| **`write_downfolded_zeta` refused to transport a q-IBZ ζ, and the child lost a capability its parent had** | The refusal read "`T` is indexed by the restart's flat-q axis; a ζ written on the q-IBZ wedge is indexed by the irreducible list, and mapping between them is the deferred IBZ-unfold". The premise is true and the conclusion does not follow. **The downfold is q-DIAGONAL**: `T[q]` is built from `S_SS[q]` and `S_cross[q]` and from no other q, and the congruence is per-q, so the transfer at a wedge q is the same matrix whichever q set the surrounding run enumerated. A wedge ζ therefore needs no unfold — it needs the row of `T` belonging to its own q, and `_zeta_q_to_restart_q` was already the function that finds that row by exact wrapped-integer match on the q labels. The count test in front of it was rejecting inputs the matcher handles. Net effect of the refusal: a wedge parent produced a child with **no ζ at all**, described in the log as "this downfold has taken nothing away" — but the parent had a ζ and the child did not | **FIXED on `fix/downfold-qirr-native-2026-08-10`, PUSHED, NOT MERGED.** The equality test is replaced by two: `nq_disk > n_q` still refuses (more ζ than q is a grid disagreement, and the one direction the matcher cannot resolve — two ζ would share one transfer row), and `nq_disk < n_q` transports and says so. `_zeta_q_to_restart_q` gains `n_q` so its identity fallback, whose premise is the FULL-BZ writer's wrapped C-order grid, now REFUSES on a wedge with unlabelled q rather than attaching grid point `i`'s transfer to wedge slot `i` — that was a latent way to write plausible, wrong numbers, reachable only once the count test above stopped shadowing it. **This does NOT make `--vq-mode interp` work on a wedge lineage**: `vq_interp` still requires `nq == nk`, untouched here because it is a statement about that reader. The child is now exactly as capable as its parent, which is the drop-in promise held rather than asserted |
| **THE MEASURED BLOCKER: the CUR selection is not star-stable, so a downfolded child is not wedge-storable** | Unfolding is a congruence by the monomial `U_s[μ',μ] = exp(2πi q_irr·L_{s,μ'}) δ(μ, α_s(μ'))`; the downfold is a congruence by `T`. They commute — i.e. unfold(downfold(wedge)) == downfold(unfold(wedge)) — iff `T[q] = U^S_s T[i(q)] (U^L_s)†`. Both Grams inherit the parent's covariance and `S_cross = S_LL[keep,:]`, `S_SS = S_LL[keep,keep]`, so the ONE step that is not automatic is that restricting rows to `keep` must commute with `U^L_s` — which holds precisely when **`α_s(keep) = keep` for every op**. It does not hold. The q = 0 selection Gram commutes with the whole group (q = 0 is invariant, and the unfold phases are unity there), so every member of an orbit has the same Schur diagonal and pivoted Cholesky fills orbits GREEDILY in index order — but stops at exactly `mu_S`, which generically falls inside an orbit. Measured on a synthetic 8-orbit × 6 group: **7 of 46 admissible `mu_S` came back orbit-closed**; on the 3 × 4 group used by the gates, closure held at exactly the multiples of 4. A child written on the wedge with a non-closed `keep` reads back as a permutation of the WRONG centroids, silently, because every shape agrees | **MEASURED AND INSTRUMENTED, NOT YET WIRED.** `gw.downfold` gains `star_stability` (the verdict, with the violating ops and the completion cost — a bool cannot tell one stray centroid from no complete orbit), `orbit_complete_keep` (the repair: round `mu_S` UP to whole orbits; DROPPING the offenders instead would shrink the retained subspace below what the rank refusal certified) and `child_unfold_tables` (the child's α and wraps as restrictions of the parent's, refusing by name on a non-closed set). Completion costs the tail of the single orbit `mu_S` stopped inside, not the group order — gated. **What is NOT done: wiring the repair into `run_downfold` and writing the child through the q_irr writer.** Every child therefore still comes out on the full BZ today. That is correct, just larger on disk than it needs to be, since the reader unfolds the parent before the driver sees it. **This is the coordination point for the spectral-closure lane**: orbit completion changes `mu_S`, so the selection's rank certificate and the `auto` ceiling have to be re-taken on the completed set, and that is their question, not this lane's |

## Gates

Deck-free, CPU, six new cells in `tests/test_downfold.py` §6 (`-k qirr`), each
run through the **shipping** `build_transfer` / `congruence` /
`symmetry_maps.unfold_isdf_operator` rather than a reimplementation, on a
synthetic parent whose full-BZ operands are DEFINED as the unfold of a wedge
block (the honest reference: the property under test is that two maps commute,
not the physics that makes a pair-density Gram covariant, which is the parent's
own wedge-storage precondition and is measured at write time).

* **q-diagonality, BIT-FOR-BIT.** At the full-BZ q whose `sym_idx` is 0 the
  unfold is the identity, so the wedge transfer and the full-BZ transfer at that
  q have identical inputs — asserted as `array_equal`, not a tolerance, because
  anything weaker would also pass if the transfer had acquired a dependence on
  the surrounding q axis. This is the licence for the ζ change above.
* **COVARIANCE, the definitive gate.** Wedge child unfolded with the child's own
  tables against the full-BZ child: **max rel 1.690e-15** at 1×1 and
  **1.825e-15** on an emulated 2×2 mesh (`devices = 4`, `mesh = {x:2, y:2}`,
  instrument-checked, real `shard_map` collectives). NOT bit-for-bit, and the
  reason is stated rather than absorbed: the two routes apply the same unitary at
  different points of one chain — before the eigh on one side, after the
  congruence on the other — and at every non-identity op multiply by a phase and
  later by its conjugate. ~8 ulp is the reassociation floor, not an agreement
  tolerance.
* **RED TWIN, and it is load-bearing.** The same-SIZE selection that cuts through
  two orbits is refused by name by `child_unfold_tables`; forced through with an
  unguarded table it misses the full-BZ answer by **max rel 8.570e-01** — order
  one. A refusal that had only ever guarded agreeing numbers would be worth
  deleting; this one is not.
* **The measurement pinned as a cell**, so "the selection is symmetry-stable"
  cannot be assumed once and inherited forever, plus the completion cost bound.

**Suite A/B, both sides run.** `tests/test_downfold.py` +
`tests/test_exciton_bands_downfold_dropin.py`, WSL CPU, worktree pin proven by
`__file__` before measuring: base `53fd80ea` **38 passed, 10 skipped**; branch
**44 passed, 10 skipped**. Delta is exactly the six new cells; zero regressions.
The 10 skips are one reason on both sides — `liblorrax_ffi_host.so` is not built
on this box, which is the documented WSL condition for the dropin suite's
driver-import cells.

## OWED, and unrunnable this lane

The NERSC certificate expired at **2026-08-10T19:22:36 UTC** and the
`ControlPersist` master was answering past it — `ssh -O exit perlmutter` then
refuses with `Permission denied (publickey)`. No cluster leg ran. **A P=4 GPU
leg is owed and this lane claims no GPU evidence**; the emulated 2×2 CPU mesh
above covers device-count logic only and does not substitute for it, per the
four-GPU rule.

Ready to run when access returns, one combined P=4 leg:

```
lx run -N 1 -G 4 -n 4 -- python -m pytest tests/test_downfold.py \
    tests/test_exciton_bands_downfold_dropin.py -q
```

and the real-deck arm the gates above stand in for — it needs a parent whose
centroid set is orbit-closed AND whose q path reduced, which is
`si_bse_debug` at `restart_q_storage = auto` (closed since `fb046e0c`):

1. GW on `si_bse_debug` with `restart_q_storage = auto` and
   `write_restart_tensors = true` → a genuinely wedge-stored parent bundle plus
   a wedge `zeta_q.h5`.
2. `lorrax-downfold` on it. **The ζ transport is the row this lane changed**:
   before, the child got no `zeta_q.h5`; after, it gets one on the parent's own
   q set. Check the writer's own `g0` cross-check line reads AGREE.
3. The covariance arm the CPU gate models: downfold the same parent twice, once
   as stored and once after re-running it under `LORRAX_FORCE_FULL_BZ=1`, and
   compare the two children's `V_qmunu` / `W0_qmunu` and the `bse_jax --tda
   --bse` eigenvalues off them. Expect agreement at the reassociation floor
   **only if `keep_idx` happens to be orbit-closed** — check
   `downfold.star_stability(keep_idx, sym_perm)` on the real deck FIRST, and if
   it is not closed that is the measurement, not a failure.

The one number this lane could not take and should not be quoted without:
**the real-deck orbit-closure rate of the CUR selection, and the μ_S inflation
orbit completion costs on a production centroid set.** The synthetic 7-of-46 is
a structural statement about the pivot order, not a number for `si_bse_debug`.
