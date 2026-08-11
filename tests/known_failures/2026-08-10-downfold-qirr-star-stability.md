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

## THE OWED LEG WAS PAID — 2026-08-10, evidence `/pscratch/sd/j/jackm/owedlegs_0810/`

Everything below this heading down to the horizontal rule is the settlement of
the "OWED" section that follows it; that section is kept verbatim because it is
the prediction this leg tested, and a prediction is worth more standing than
edited.  Worktree `/pscratch/sd/j/jackm/owedlegs_0810/tree`, detached at
`47657990`, `git dirty-count: 0` printed inside every leg.

**The gates, on four real GPUs.**  `tests/test_downfold.py -k qirr` on one
process holding four `CudaDevice`s, so `resolve_mesh()` builds a genuine 2×2
over real cards rather than the emulated CPU mesh the amendment measured:
**6 passed, 32 deselected in 27.59 s**, `jax device_count: 4`, HEAD
`47657990` (`_logs/qirr_p4gpu.log`).  The covariance gate, the bit-for-bit
q-diagonality gate and both red twins are therefore GPU evidence now, not CPU
evidence.  The full two files at four GPUs come back **3 failed, 51 passed**
against **54 passed** on one GPU, and all three failures are the test
fixtures' own shapes rather than the product: `m = n = mu_S = 3` and a
`(3, 4, 9)` output are not divisible by a 2×2 mesh, so
`contract_bands_block_reshard` and `pjit` refuse them by name.  The cells are
1×1-shaped and cannot express themselves on a square mesh; that is a
gate-shape defect worth its own repair and it touches nothing this amendment
claims.

**The real deck, and it behaved.**  `si_bse_debug` on the shipped orbit-closed
480-centroid set at `restart_q_storage = auto` resolved, in the writer's own
words, to `auto -> ibz (centroid set is orbit-closed (worst residual 4.596e-16
at tol 1.0e-05) and the q path reduced)` — a genuinely wedge-stored parent,
`V_qmunu (8, 480, 480)` over 8 IBZ q of 64, plus a wedge `zeta_q.h5`.  The
downfold on it selected `mu_S = 185` at `mu_small = auto` and transported the
ζ, printing exactly the row this amendment fixed:

```
[downfold/zeta] parent zeta_q.h5 stores 8 q against the bundle's 64 — the
    q-IBZ WEDGE, and it transports.
[downfold/zeta] g0 cross-check: zeta_S(q=0, G=0) vs the transported g0_S
    -> AGREE (max rel 1.065e-15).
```

**The covariance arm, on the deck, is exact.**  The same deck was run a second
time at `restart_q_storage = full` and downfolded identically, and the two
children were compared dataset by dataset: `V_qmunu (64, 185, 185)`,
`W0_qmunu`, `G0_mu_nu`, both `eps_w` vectors and `zeta_q_G (8, 185, 588)` are
**BIT-IDENTICAL, max |Δ| = 0.000000e+00** (`_logs/compare.log`).  Not the ~8
ulp reassociation floor the synthetic gate reports — exactly zero, and the
reason is the one the amendment already gives: the reader unfolds the parent
before the driver sees it, so both routes hand the downfold the same operand.
`LORRAX_FORCE_FULL_BZ=1` was not needed; flipping the deck key is the same
experiment with less machinery.

**THE NUMBER THIS AMENDMENT REFUSED TO INFER, TAKEN AT LAST — and it is worse
than the synthetic.**  On `si_bse_debug` (μ_L = 480, 96 ops = 48 spatial + TRS,
`downfold_rcond = 1.1e-6`, ceiling 185), every admissible μ_S from 1 to 185 was
closure-tested against the parent's own stored `sym_perm`, using the shipping
pivoted-Cholesky kernel's actual pivot order:

> **real-deck orbit-closure rate = 0 of 185 = 0.0000.**

Not one.  The synthetic 7-of-46 (≈15 %) was optimistic by exactly the amount
that matters.  At the production selection `mu_small = auto → μ_S = 185`, **94
of the 96 ops** send a kept centroid outside the kept set, and orbit completion
would add **295 centroids, taking μ_S from 185 to 480 — the entire parent
basis**, which is the same thing as not downfolding at all.  Over the open μ_S
the inflation runs min 35, median 323, max 387 centroids.

The orbit census says why, and it retires this amendment's own consolation.
The 480-point set is **11 orbits: two of size 24 and nine of size 48**.  An
orbit-closed subset must be a union of those, so the only closed sizes below
the ceiling are the seven multiples of 24 — and the measured rate of zero means
the real pivot order does not land on a union at any of them.  **The claim that
"the CUR pivot order fills orbits greedily, so completion costs the tail of one
orbit rather than the group order" is REFUTED on a production centroid set.**
It held on the synthetic because q = 0 makes every member of an orbit share a
Schur diagonal and the tie-break was index order; on the real Gram the
tie-break interleaves orbits instead, and by μ_S = 185 the kept set already
touches all eleven.  `star_stability`'s `describe()` still prints the greedy
sentence, and that sentence should be corrected where it stands.

**Two defects the four-GPU rule found on the way, both new.**

1. **`lorrax-downfold` cannot run at P > 1 at all.**  At `-G 4 -n 4` it dies in
   `downfold_run._gate_g0_against_zeta` (`src/gw/downfold_run.py:951`), which
   does a bare `np.asarray(jax.device_get(g0_S))` on a globally sharded array:
   *"Fetching value for `jax.Array` that spans non-addressable (non process
   local) devices is not possible."*  The gate that the real-deck recipe below
   tells you to read is precisely the line that cannot run in production
   geometry.  It needs `process_allgather`, or the gather the selection path
   already uses.  Both children above were therefore written at one GPU; the
   tensors reached disk before the gate on the P=4 attempt, so the wedge child's
   `zeta_q.h5` exists from that run too and is the same size as the full-BZ
   child's.
2. **A four-device single process is refused by the driver stack**, so "one
   process, four cards" is not a way around (1): `resolve_mesh` raises
   `mesh 2×2=4 != jax.process_count()=1`, and `SlabIO` refuses the same
   mismatch from the MPI side.  The gates leg above works only because pytest
   never opens a SlabIO file; anything touching the drivers must be
   multi-process.  Note also that two P=4 JAX steps placed on ONE node collide
   in the coordination service (`different incarnation`, rc=134) — batch such
   legs at `-P 1` or place them on different nodes.

**What is still owed after this leg.**  The BSE-eigenvalue leg of the
covariance arm (step 3 below asks for `bse_jax --tda --bse` off both children)
was not run: with the tensors bit-identical it can only agree, so it was
dropped as a measurement of nothing, and that is a judgement rather than a
result.  And the `g0` cross-check has never been read at P > 1, because
defect (1) above is in front of it.

---

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
