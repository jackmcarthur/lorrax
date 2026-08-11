# The two verdicts the q-sign fix voided, re-taken on the fixed tree (2026-08-11)

**STATUS: MEASUREMENT COMPLETE. NO CODE LANDED, NO TOLERANCE MOVED.**
Base `origin/main` `0578bc89`, which carries the downfold Gram q-sign fix
(`tests/known_failures/2026-08-11-downfold-gram-q-sign.md`). Workspace
`/pscratch/sd/j/jackm/qsign_recut_0811/`, sole writer. Predictions were
written to `qsign_recut_0811/PREREG.md` and timestamped before the first
submit; every one of them is scored below, including the one that was wrong.

That fix row closed by naming two things it had poisoned but not re-run. This
row re-runs both. It also found one new defect on the way, which is registered
in §4 rather than fixed here.

## 1. The headline numbers

| what | before (void) | after (this lane) |
|---|---|---|
| BSE-window refit certification, child μ191 | **6.819 meV** (P=4) / 7.719 (P=1) | **2.593 meV** (P=4) |
| the certification gate, 0.01 meV | REFUSED | **STILL REFUSED** |
| orbit-floored 168 `eps_W(W0)` median | 4.319e-02 | **1.273e-01** |
| point-picked 185 `eps_W(W0)` median | 1.056e-02 | **1.196e-02** |
| point-picked 185 exciton error | **−348.6 meV** | **−42.6 meV** |
| orbit-floored 168 exciton error | *did not converge* | **−82.8 meV** |

## 2. JOB 1 — the refit certification with a fixed child

The child was regenerated from the same parent
(`xbdense_0810/parentfull/tmp/isdf_tensors_960.h5`) by the same procedure —
`mu_small = 191`, `downfold_rcond = 1.1e-6`, window `0:20` both sides, the deck
byte-identical to `xbdense_0810/df_191.in` but for the output path — on the
fixed tree, at four real GPUs in four processes. It reports `eps_W(W0)`
min/median/max = 4.582e-03 / 4.817e-03 / 6.205e-03 and keeps 173 independent
directions of 191. The certification was then re-run on it with the same deck
and the same flags as the arm that produced 6.819 meV.

**The comparison has exactly one variable, and that is checked rather than
assumed.** The regenerated child's kept-centroid table is **bit-identical** to
the one the poisoned child shipped —
`centroids_frac_191_downfold.txt` md5 `922208c126563f9a75e375c08a0e4cac` on
both, and the parent basis table `centroids_frac_960_parent.txt` md5
`4d8da1404b43b651a27bcc60af21e4cf` on both. The CUR selection is deterministic
given the parent bundle and `rcond`, and both are the same bytes, so the keep
set did not move. **The only thing that differs between the 6.819 meV arm and
the 2.593 meV arm is the q-sign relabel**, which is what makes the per-Q table
below readable as a statement about the defect rather than about two different
bases.

**The new number is 2.593 meV, and the 0.01 meV gate still reads REFUSED.**
The gate was not touched, no `.dat` and no `.png` were written, and per the
dispatch the dense band-structure arms were NOT run. The driver refuses to
plot behind a failed certification, and that refusal is correct.

**This was predicted before the run, and the reasoning is the useful part.**
The parent control carries no downfold at all, so `pair_density_gram` is never
called on its route and the q-sign defect cannot reach it; its 0.858 meV is
therefore a floor on the *route*, not on the child. A child at μ=191 goes
through that same route plus a basis truncation and cannot land below its own
parent. Predicted "at or above ~0.86 meV, point estimate 0.9–3 meV"; measured
2.593. The prediction that the gate would still fail was the whole reason this
lane did not promise plots.

### The per-Q table, which is where the physics is

Taken at **both** shapes, because the original headline was a P=1 leg (7.719)
and its P=4 twin was 6.819, so a shape-matched comparison is the only honest
one. Both shapes tell the same story.

| Q | fractional | −q ≡ q | P=4 before | P=4 after | factor | P=1 before | P=1 after | factor |
|---|---|---|---|---|---|---|---|---|
| Q#1 | X (0, ½, ½) | **yes** | 1.07403 | 0.40475 | 2.7x | 0.99700 | 0.29551 | 3.4x |
| Q#2 | (¼, ¾, ½) | no | 1.27721 | 0.22834 | 5.6x | 1.34498 | 0.22049 | 6.1x |
| Q#3 | L (½, ½, ½) | **yes** | 3.05016 | **2.59335** | 1.2x | 3.20314 | **2.66313** | 1.2x |
| Q#5 | Σ (¼, ½, ¼) | no | **6.81943** | 0.01654 | **412x** | **7.71905** | 0.01690 | **457x** |

At P=4 the two q the defect could not touch improved by a mean factor of
**1.9** and the two it did touch by a mean factor of **209** — a **109-fold
separation on the predictor**. At P=1 the same numbers are **2.3** and
**231**, a **101-fold separation**. The worst point moved from Σ — the
non-self-inverse point that carried the whole headline — to L, which is
self-inverse and so was never corrupted, and Σ itself is now 0.0165 meV, within
a factor of two of the gate. **The agreement between the two shapes is the
point: a mesh artefact would not reproduce a 100-fold predictor separation
twice.**

**So the attribution loop closes, and it closes on the ANOMALY rather than on
the whole error.** What the q-sign defect explains is exactly what was
anomalous about the old pattern: Σ's dominance, and the rearrangement of the
per-Q ordering between child and parent that no smooth basis-truncation story
could account for. Both are gone. What it does **not** explain — and this row
does not claim it does — is the 2.593 meV that remains, which splits into the
route's own windowed-ζ' representability error (visible on the un-downfolded
parent at 0.858 meV, a separate and already-registered question,
`2026-08-10-exciton-bands-offgrid-Q-is-slab-only.md`) plus a genuine downfold
truncation residual concentrated at L, quantified in the next section. **The
useful change is that every remaining term now sits somewhere it can be
honestly attributed**, which was not true of the 7.719 meV.

### The decomposition, restated — and the statistic the old one used was wrong

The old split compared maxima: 7.719 meV for the child against 0.858 meV for
the parent, "~89 % attributed to the child". **Those two maxima are at
different Q** — the child's was at Σ and the parent's is at W — so the ratio was
never a decomposition of one quantity into two. Now that the defect is gone the
per-Q comparison is available and it is unambiguous. Both columns below are
P=1, the shape the parent control has always been taken at.

| Q | −q ≡ q | parent μ960 | fixed child μ191 | child − parent |
|---|---|---|---|---|
| X (0, ½, ½) | **yes** | 0.03899 | 0.29551 | +0.257 |
| (¼, ¾, ½) | no | **0.85783** | 0.22049 | **−0.637** |
| L (½, ½, ½) | **yes** | 0.37907 | **2.66313** | **+2.284** |
| Σ (¼, ½, ¼) | no | 0.03410 | 0.01690 | −0.017 |

**The parent column was re-measured on the fixed tree and came back
identical to five decimals on all four Q** — 0.03899 / 0.85783 / 0.37907 /
0.03410, against `xbwin_0811/_logs/ctl_parent.log`'s 0.03899 / 0.85783 /
0.37907 / 0.03410, same worst point at Q#2. **That is the sharpest control this
lane has, and it runs in the opposite direction to every other measurement
here: the q-sign fix changed the child by up to 457x and changed the
un-downfolded parent by nothing at all, on the same tree, the same deck and the
same route.** The fix is surgical, exactly as its own row claimed
(`pair_density_gram` has no caller outside `gw/downfold*.py`), and this is that
claim measured on a production deck rather than argued.

**The entire remaining child-vs-parent gap is one point, and it is L.** At the
other three Q the fixed child is within 0.26 meV of the un-downfolded parent
and is actually *better* than it at two of them, including at W where the
parent has its own worst point. L contributes +2.284 meV of the +1.89 meV total
excess — more than all of it, because the other three are net negative.

**And L is self-inverse, so the q-sign defect could never have touched it.**
The downfold's remaining contribution on this deck is therefore concentrated
entirely at a q where the bug was provably absent, which is as clean a
statement as this measurement can make: whatever is left is basis truncation at
μ 191 against 960, not the defect and not a q-labelling error.

Pre-registered prediction 1B said the child-attributed share would fall below
50 %; on the maxima ratio it fell to ~68 %, so **1B is scored WRONG and left
standing as wrong.** It deserved to be: it committed to a ratio of maxima taken
at different Q, which is the same bad statistic the original 89 % used. The
table above is what should have been predicted.

## 3. JOB 2 — the orbit-floor deciding leg

Re-run exactly as instrumented in `/pscratch/sd/j/jackm/orbitfloor_0810/`: the
same two decks, the same two parents read read-only, the same `mu_small = auto`
and `downfold_rcond = 1.1e-6`, four real GPUs in four processes.

**The harness reproduces `wedgechild_0811` to four digits.** That lane had
already re-run `df_floor` on the fixed tree; it recorded `eps_W(W0)`
9.983e-02 / 1.273e-01 / 2.391e-01 and this lane measures
9.983e-02 / 1.273e-01 / 2.391e-01. Pre-registered as the control on my own
harness, and it holds, so the rest of Job 2 is readable.

| instrument | orbit-floored 168 | point-picked 185 | which wins |
|---|---|---|---|
| `eps_W(W0)` min/med/max | 9.983e-02 / **1.273e-01** / 2.391e-01 | 1.020e-02 / **1.196e-02** / 1.418e-02 | point, by **10.6x** on the median |
| `eps_W(V)` median | 1.432e-01 | 1.167e-02 | point, by 12.3x |
| independent directions | **122** of 168 | **171** of 185 | point |
| retained rank per q (min/med/max) | 120/123/124 | 171/173/175 | point |
| lowest exciton, 4v8c | **3.492286 eV** | **3.532525 eV** | — |
| vs parent 3.575096 eV | **−82.8 meV** | **−42.6 meV** | point, by 1.9x |
| `E_max` (Lanczos raw), parent 12.980 eV | 17.516 eV (**+35 %**) | 12.979 eV (**+0.0 %**) | point |

The `E_max` row is worth reading beside its own noise floor. It comes from a
Lanczos bound with a random start vector, and the parent — which is not
downfolded and should not have moved at all — came back at 12.980 eV against
13.001 eV before, so **0.16 % is about what this instrument's repeatability
is**. On that scale the point-picked arm's +0.0 % is no distortion at all
(it was +11 % before the fix) while the orbit-floored arm's +35 % is far
outside it (it was +47 %). All three arms converged with no `LinAlgError` and
comparable residuals, so this is three arms measured on one instrument rather
than two arms and an excuse.

**THE VERDICT IS UNCHANGED IN DIRECTION AND STRONGER IN MARGIN: the
symmetric-but-smaller orbit-floored basis is beaten, not matched, on every
instrument that returns a number.** It was beaten before the fix too, so the
recorded conclusion survives — but almost none of the numbers behind it do, and
two of the three headline claims were artefacts:

* **The −348.6 meV is gone.** The point-picked arm's exciton error is −42.6 meV,
  an eighth of what was recorded. The old figure was measuring the defect.
* **The FEAST non-convergence is gone.** The orbit-floored arm now converges and
  returns 3.492286 eV. The old row was careful to report that as "a failure to
  produce an observable, not a measured observable" and to say it had not
  separated the causes; it was right to be careful, and the cause was the
  q-sign defect.
* **The direction counts did not move**, exactly as pre-registered: 122 of 168
  and 171 of 185, μ_S 185 → 168 in 4 orbits (48+48+48+24), orbit-closed under
  all 96 ops. These are properties of the selection Gram at q = 0, where
  −q ≡ q, so the defect never reached them. The economics finding stands
  untouched, as its own row already said it would.
* **The parent is unmoved at 3.575096 eV**, reproducing the orbit-floor lane to
  all six digits. The parent is not downfolded, so this was the sharpest
  available control on the whole lane and it holds.

The two arms responded very differently to the fix, and that is itself
informative: the point-picked arm's `eps_W` barely moved (1.056e-02 →
1.196e-02, +13 %) while the orbit-floored arm's tripled (4.319e-02 →
1.273e-01). The point-picked arm keeps 171 directions of 185 and its per-q
`eps_W` spread is tiny (1.02e-02 to 1.42e-02), so its projection residual is
set by basis size and is nearly insensitive to which q the transfer was built
at. The orbit-floored arm keeps 122 of 168 and truncates hard, so the choice of
subspace matters and a wrong-q transfer really costs it. **The arm with less
headroom was the arm the defect hurt most**, which is what the fix row
suspected and did not claim.

**This decides the recorded accuracy comparison only.** The owner's
floor-to-orbits interface is untouched by this lane and stays regardless, as
the dispatch directed; nothing here re-litigates it, and the ruling about what
a user-facing count means is independent of what any deck's error bar does.

## 4. NEW — `refit_vq` cannot run at P>1 unless the basis size defeats sharding

Found by the four-GPU rule, on the parent control arm, and **not fixed here**
because this lane lands no code.

`bse/vq_interp.py:2778` in `refit_vq` does a bare
`jax.device_get(ztG_box[:, jnp.asarray(fi)])` on an array that is globally
sharded at P>1, and dies with `RuntimeError: Fetching value for jax.Array that
spans non-addressable (non process local) devices`. This is the same class of
defect the BSE-window refit lane fixed in `refit_prepare` (`_to_host` rather
than `device_get`) and is a sibling it missed, in a function that lane's own
P=4 leg never exercised on a shardable basis.

**Why it hid.** The failure depends on the ISDF basis size. The child has
n_μ = 191, which is odd, so `sharding_fit` refuses to shard it (the run says so
in as many words: `191 % 2 != 0` → `PartitionSpec(None,...)`), the array
becomes fully addressable, and `device_get` accidentally works. The parent has
n_μ = 960, which is even, so it shards for real and the call fails. The
certification arm that has run at P=4 twice now is the child arm, and it has
been passing through this line on a replicated array the whole time.

Consequence for this lane: the parent control was taken at **P=1**, which is
also the shape the original control used (`xbwin_0811` ran `ctl_parent` at
`--px 1 --py 1`), so the comparison is like-for-like. The FOUR-GPU RULE is
satisfied for every number this lane banks except that control, and the reason
is a defect that must be fixed before a P=4 parent control is possible at all.

Suggested fix, for whoever takes it: `_to_host`/`process_allgather` at
`vq_interp.py:2778`, plus a cell that runs `refit_vq` at P>1 on an EVEN n_μ —
the odd-μ case cannot catch this and the existing coverage is all odd-μ.

## 5. Gates and provenance

No source file was modified, so no test gate is owed: this lane is measurement
plus ledger. Every leg printed `git rev-parse HEAD` = `0578bc89` and
`dirty-count: 0` from inside the container, and asserted
`gw.downfold.negate_q_index` present.

| leg | shape asserted in-leg | rc | wall |
|---|---|---|---|
| `df191` (960→191) | 4 devices, 2x2, 4 processes | 0 | 34 s |
| `df_floor` (480→168) | 4 devices, 2x2, 4 processes | 0 | 25 s |
| `df_point` (480→185) | 4 devices, 2x2, 4 processes | 0 | 33 s |
| `xb_cert2` (child certification) | 4 devices, 2x2, 4 processes | 1 = the gate refusing | 95 s |
| `bse_floor168b` | 4 devices, 2x2, 4 processes | 0 | 37 s |
| `bse_point185b` | 4 devices, 2x2, 4 processes | 0 | 37 s |
| `bse_parentb` | 4 devices, 2x2, 4 processes | 0 | 44 s |
| `xb_ctl_parent3` (control) | 1 device — see §4 | 1 = the gate refusing | 442 s |

**What was collected, not just what was green.** Every one of the seven P=4
legs above shows **four** per-rank `compile-cache ... summary` lines, so all
four ranks reached the end of their work rather than one rank reporting for a
quartet that had already lost members. `xb_cert2` — the leg the headline number
comes from — carries the shape in its own words,
`jax.device_count()=4 process_count()=4 local_device_count()=1;
mesh_xy.shape={'x': 2, 'y': 2}`, together with
`evs sharding=... fully_addressable=False`, which is the part that cannot be
faked by four independent single-device sessions wearing a P=4 label. Its
wrapper prints no `MESH_SHAPE check` line **by design**: rc is non-zero because
the certification gate refused, and on a non-zero rc the wrapper stands down
and lets the payload's own result be the result.

**A harness defect of this lane's own, recorded because it nearly cost a
verdict.** The first in-leg wrapper asserted the mesh by grepping for
`exciton_bands`' banner dialect only. `bse.bse_jax` announces its shape
differently ("This is rank 0 of 4, and it addresses 1 of the 4 devices"), so
two observable legs **completed their physics and were then killed by my own
assert** with a false `REFUSING TO BANK`. The wrapper now reads all three
driver dialects and, more importantly, **skips the shape assert when the
payload's own rc is non-zero** — reporting 95 over a real traceback hides the
defect underneath, which is precisely how §4 nearly went unnoticed. The
measurements above are all from the repaired wrapper.

Separately, two legs died on the JAX coordination service
(`ALREADY_EXISTS: request from a newer incarnation`) because two of this lane's
own launchers put P=4 steps on one node in the same second. The coordinator
port derives from `SLURM_JOB_ID` (`lx_pool.py:256`), so submissions must be
serialised within an allocation; they were, and the legs were re-run. Nothing
was banked from a collided leg.

**Allocations, stated correctly after an earlier draft of this row got it
wrong.** This lane's own `salloc` was refused `QOSMaxSubmitJobPerUserLimit`
with two peer allocations already live — the trap `owedlegs_0810` hit and
recorded — so it attached pools **56624724** and **56612363** by explicit ID as
designed co-tenancy, and for most of its life it had created nothing. One leg
was lost when 56612363 expired under it, and the **final relaunch of the parent
control at 01:20 found 56624724 full and `lx` created allocation
**56626252** (1 node, `lx-alloc-jackm`, 01:21:06) to place it.** That is this
lane's own allocation, it is **released by explicit ID** at the end of the lane,
and the peer pools are left alone. An earlier draft of this row and of
`EVIDENCE.md` claimed "created none and cancelled none"; that was true when
written and stopped being true, and it is corrected here rather than quietly.

**Everything this lane wrote, named.** It is sole writer of
`/pscratch/sd/j/jackm/qsign_recut_0811/`. Outside that directory it made four
deliberate ledger writes and nothing else: this file and its siblings on branch
`ledger/qsign-recut-2026-08-11`; the VOID-BY-Q-SIGN-FIX amendment appended to
`orbitfloor_0810/manifest_row.md`; a new `xbwin_0811/SUPERSEDED.md`; and rows
in `/pscratch/sd/j/jackm/EVIDENCE_MANIFEST.md` and the canonical
`RUNS_INFLIGHT.md`. `xbdense_0810`, `owedlegs_0810`, `downfold_s1`,
`triangle_0810` and the measurement files of `xbwin_0811` and
`orbitfloor_0810` were read-only inputs throughout — no log, no bundle and no
recorded number in either of those workspaces was altered or deleted, only
annotated.
