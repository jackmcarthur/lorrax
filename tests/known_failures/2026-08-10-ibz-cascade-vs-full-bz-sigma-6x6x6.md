# THE IBZ CASCADE AND THE FORCED FULL-BZ PATH DISAGREE ON Σ AT Si 6×6×6 (2026-08-10) — **ADJUDICATED: NEITHER ARM IS CORRECT AT 6×6×6, AND THE CAUSE IS UPSTREAM OF BOTH**

**Status: the disagreement is real and reproduces, but it is a symptom rather
than the disease. The decisive 4×4×4 leg has been run and the two arms agree
there to 7.6 µeV, so neither code path is broken in general. At 6×6×6 both arms
independently violate a symmetry identity that neither violates at 4×4×4, and
the violation is already present in Σ_x, which contains no screening at all.
The 4×4×4 anchor deck and everything built on it are NOT implicated. The Si
6×6×6 GW bundle is NOT VALID for reference use in either arm.**

---

## VERDICT — written 2026-08-10 by the physics-discrimination lane

### 1. The decisive leg, and what it says

The finding lane named the cheapest decisive next step and it has now been
taken: the same two arms, on the Si **4×4×4 anchor deck** — the deck the
campaign's external BerkeleyGW anchoring was done on
(`/pscratch/sd/j/jackm/si_gnppm_0809/lorrax/deck.in`, `wcoul0_source = epshead`
against the BGW GN twin's own `eps0mat.h5`, `ppm_head_omega_h_ry` pinned to
BGW's own 1.32366562 Ry).

Both arms were re-run from scratch. Neither reuses a stored result, because
the stored one turned out to be a trap — see §7.

| | arm A (shipped IBZ cascade) | arm B (`LORRAX_FORCE_FULL_BZ=1`) |
|---|---|---|
| what the run reports about its own q axis | `q-IBZ reduction: 8 IBZ q-points / 64 full-BZ (disk shrink 8.0×)`, `n_q_disk=8 of 4·4·4=64` | `q axis on disk: full BZ (64 q-points)`, `n_q_disk=64 of 4·4·4=64` |
| exit | 0 | 0 |
| wall | 70 s, P=4, BFC@0.85 | 65 s, P=4, BFC@0.85 |

Comparing the two `eqp0.dat`, row by row — **800 data rows and 8 k-headers on
each side, which is the expected 8 irreducible k × 100 bands, asserted rather
than assumed**:

| column | quantity | row-wise max\|Δ\| |
|---|---|---|
| 0 | spin / index | **0.000000e+00** |
| 1 | band index | **0.000000e+00** |
| 2 | E_DFT (eV) | **0.000000e+00** |
| 3 | **E_QP (eV)** | **7.626000e-06** |

**MAE on E_QP: 6.21e-07 eV. Max: 7.63e-06 eV — 0.0076 meV.**

That is agreement at the level of floating-point reassociation. The same
comparison at 6×6×6 gives 110 meV, a factor of **14 000** larger.

### 2. The adjudication, stated as the lane's brief required

The brief set out two cases in advance. Case (a) is the one that obtains: **the
two arms AGREE at 4×4×4, so the defect is specific to the 6×6×6 deck and is not
a standing defect of either code path.** Three consequences follow immediately,
and they matter more than the original finding did.

* **The `LORRAX_FORCE_FULL_BZ` override is not broken.** The standing prior —
  established-code-is-not-the-suspect, therefore the rarely-run debugging
  override is the likely defect — was a reasonable prior and it is **wrong
  here**. On the anchor deck the override reproduces the shipped cascade to
  7.6 µeV. It does not get a defect row and it does not get quarantined.
* **The shipped IBZ cascade is not broken either**, on the same evidence, in
  the same measurement. Nothing in this file implicates production.
* **The 4×4×4 corpus is untouched.** Every number this campaign has anchored,
  frozen or published sits on the 4×4×4 deck, and the symmetry machinery is
  exact there to sub-µeV. No re-cut of anything is owed on account of this file.

### 3. Which Σ term carries it

The `sigma_diag_file` output carries the per-band decomposition, so the term
that moves can be read off directly rather than inferred. Both grids, arm A
against arm B, all bands and all k in the file:

| term | 4×4×4 max\|Δ\| | 4×4×4 MAE | **6×6×6 max\|Δ\|** | **6×6×6 MAE** |
|---|---|---|---|---|
| Σ_x | 0.0000 meV | 0.0000 meV | **0.115 meV** | 0.011 meV |
| **Re Σ_c** | **0.0050 meV** | 0.0004 meV | **128.079 meV** | **23.149 meV** |
| Im Σ_c | 0.0000 meV | 0.0000 meV | 0.001 meV | 0.000 meV |
| V_H | 0.0000 meV | 0.0000 meV | **0.000 meV** | 0.000 meV |
| E_o | 0.0000 meV | 0.0000 meV | **0.000 meV** | 0.000 meV |

**Re Σ_c carries essentially all of it**, which confirms the finding lane's
observation that V_H and E_o are clean and sharpens it: the imaginary part is
clean too, so this is not a pole-placement or damping question. But the row
that decides the investigation is Σ_x. It is **exactly zero** at 4×4×4 and
**nonzero at 6×6×6**, and Σ_x contains no screening whatsoever — no W, no head,
no plasmon pole. Whatever is wrong at 6×6×6 is therefore already wrong *before*
W is built, and no hypothesis about the q→0 head or the cascade's unfold of W
can be the root cause, because none of them can move Σ_x.

### 4. The instrument that settles which arm is right: the k-star identity

Comparing two arms to each other can only ever show that they differ. A
one-sided test was needed, and one is available at zero further cost, because
`sigma_diag_file` is written on the **full BZ** in both arms (64 k at 4×4×4,
216 k at 6×6×6) even when the q axis is reduced.

Σ_c and Σ_x must be **constant across a k-star** — symmetry-equivalent k must
carry the same self-energy. This is the same identity `FIX_sigma_kstar.md` used
when it took the Σ_c star spread from 43.85 eV to exactly zero. The stars can
be assigned from the data with no symmetry code involved at all: k in the same
star carry identical DFT eigenvalue vectors. The assignment's own validity
check is that it recovers **exactly 8 stars from the 64 k at 4×4×4 and exactly
16 from the 216 k at 6×6×6**, matching the two wedges' irreducible counts.

Measured on each arm **independently** — this is not an A/B, each number is one
arm judged against a symmetry requirement:

| grid | arm | max Σ_c star spread | max Σ_x star spread |
|---|---|---|---|
| 4×4×4 | A — IBZ cascade | **0.006 meV** | **0.0000 meV** |
| 4×4×4 | B — forced full BZ | **0.009 meV** | **0.0000 meV** |
| **6×6×6** | **A — IBZ cascade** | **38.785 meV** | **0.064 meV** |
| **6×6×6** | **B — forced full BZ** | **139.407 meV** | **0.149 meV** |

**Both arms fail at 6×6×6. Neither fails at 4×4×4.** The question the original
file asked — which arm is correct — has no answer at 6×6×6 because the answer
is *neither*. Arm B is the worse of the two by a factor of about 3.6, which is
why it cannot simply be adopted as the reference, and arm A's 38.8 meV is by
itself already large against every tolerance this project uses. Their 110 meV
difference is bounded below by both of these numbers and is a symptom of the
same breakage, not an independent fact about the cascade.

### 5. The class hunt, and why the obvious hypothesis is refuted

The brief's case (a) prescribed hunting for a symmetry class the 6×6×6 wedge
has and the 4×4×4 lacks. That hypothesis is **refuted by the per-star table**:

| star size | k₀ | A: Σ_c spread | A: Σ_x spread | B: Σ_c spread | B: Σ_x spread |
|---|---|---|---|---|---|
| 1 | 0 | 0.000 | 0.0000 | 0.000 | 0.0000 |
| 3 | 21 | 25.025 | 0.0640 | 80.471 | 0.1490 |
| 4 | 3 | **1.110** | **0.0000** | 25.388 | 0.0010 |
| 6 | 7 | 38.785 | 0.0310 | 85.514 | 0.1110 |
| 6 | 14 | 31.802 | 0.0280 | 139.407 | 0.1390 |
| 8 | 1 | **2.042** | **0.0010** | 30.569 | 0.0050 |
| 8 | 2 | **2.359** | **0.0000** | 23.354 | 0.0010 |
| 12 | 11 | 26.880 | 0.0210 | 81.765 | 0.0660 |
| 12 | 16 | 24.015 | 0.0500 | 81.658 | 0.1040 |
| 12 | 58 | 22.860 | 0.0460 | 68.061 | 0.0980 |
| 24 | 8 | 24.261 | 0.0270 | 80.287 | 0.1270 |
| 24 | 9 | 17.786 | 0.0220 | 78.601 | 0.0630 |
| 24 | 10 | 35.482 | 0.0230 | 77.909 | 0.0560 |
| 24 | 15 | 22.070 | 0.0480 | 78.350 | 0.0930 |
| 24 | 51 | 36.387 | 0.0250 | 77.136 | 0.0710 |
| 24 | 52 | 31.079 | 0.0180 | 73.911 | 0.0780 |

(all in meV; the 4×4×4 equivalent is ≤ 0.009 in the Σ_c column and identically
0.0000 in both Σ_x columns, star by star.)

The 4×4×4 wedge has stars of size 1, 3, 4, 6, 6, 8, 12 and 24 — the same sizes
that are dirty at 6×6×6 are present and **clean** at 4×4×4. So this is not an
orbit class the smaller grid fails to exercise. What the table does show is a
strict correlation between the two columns within each arm: the three stars
whose Σ_x spread is ≤ 0.001 meV are exactly the three whose Σ_c spread stays
under 2.4 meV, and every star with Σ_x spread ≥ 0.018 meV has Σ_c spread
≥ 17.8 meV. A ten-fold gap in the unscreened term maps onto a ten-fold gap in
the screened one, with an amplification of roughly 500–1000×.

**The breakage is carried by the deck's ISDF representation, not by its k-point
inventory**, and W amplifies it rather than causing it.

### 6. The prime suspect, labelled as the hypothesis it is

Not measured by this lane, and stated as a hypothesis with its next leg named,
per the provenance rule.

The 6×6×6 deck was built with a deliberately **over-complete** centroid basis,
because the reference was also meant to serve as a downfold parent. The
generator's own log says what that costs:

    [pivoted_cholesky] orbit-aware: 27 orbits picked → 1104 unfolded centroids (orbit-closed)
      [point rank] 27 orbits, 1104 points, 933 independent directions (84.5% of the points)
      [point rank] NOTE: 171 of the 1104 delivered points add no independent direction
      at tol*max(diag G).  The zeta back-solve will truncate about that many modes per q.

A rank truncation of a rank-deficient Gram matrix picks *some* basis of the
retained subspace, and that choice is point-group covariant only if the cut
falls between whole degenerate blocks. Where it slices through one, the
retained ζ subspace is no longer invariant under the crystal symmetry, and
every object built on ζ inherits the breakage — visibly in Σ_x at the
0.03–0.15 meV level, and amplified through W into Σ_c at 18–139 meV. The
4×4×4 deck's 1128-centroid basis shows exactly zero Σ_x star spread, which is
what a basis whose truncation does not fire (or fires on whole blocks) looks
like.

**The cheap next leg, and it is cheap because Σ_x needs no W:** re-run the
6×6×6 deck with the rank cut moved — a smaller, rank-safe centroid set, or
`zeta_rcond` chosen so the cut lands between blocks — and re-measure the **Σ_x
k-star spread** alone. If it returns to zero, this is settled and the fix is a
basis-generation constraint (orbit closure is necessary but not sufficient;
rank closure is the missing condition). If it does not, the suspect is wrong
and the hunt moves to the wavefunction rotations. Either way the instrument is
one number off a run that needs no screening and no BSE.

### 7. A trap the next lane must not fall into

While establishing the anchor side, both of today's 4×4×4 arms were compared
against the `eqp0.dat` sitting in `si_gnppm_0809/lorrax/` from 2026-08-09. That
comparison reports **max 96.45 eV, MAE 12.67 eV**, concentrated on exactly five
of the eight k-points while three agree to under 1.7 meV — and E_DFT is
bitwise identical throughout, so it is not an ordering artifact.

It is not a finding either. The three agreeing k are the TRIM points and the
five disagreeing ones are the non-TRIM k, which is the precise signature of the
defect `FIX_sigma_kstar.md` fixed: the crossing window's one-sided τ grid
completed elementwise rather than with the band adjoint, which is correct only
where k ≡ −k. That fix landed at `dd727216` on 2026-08-09 at 19:17 PDT. The
stored file was generated at 02:10 PDT the same day — **before the fix**. It is
a pre-fix scratch artifact, superseded when the owner authorised the reference
freeze at `1e64d83a`, and the corroborating detail is that its E_QP goes to
−63.8 eV for a band whose E_DFT is +33.6 eV, which is not a physical number.

Today's arms produce sensible values at those k. This is measurement rule 4
(arm ancestry) doing its job: had the stored file been reused as "arm A"
instead of re-run, this lane would have published a 96 eV disagreement that
belongs entirely to a fix that landed seventeen hours earlier.

### 8. What this means for the 6×6×6 reference

* The Si 6×6×6 GW restart bundle at `/pscratch/sd/j/jackm/si666_ref_0810` is
  **not valid as a reference in either arm.** Its Σ_c violates a symmetry
  identity by up to 38.8 meV (cascade) or 139.4 meV (forced full BZ), measured
  against the requirement itself and not against the other arm.
* The mean field underneath it is untouched by this and remains good: E_DFT is
  bitwise identical across every star, which is how the stars were assigned in
  the first place.
* **Do not hand this bundle to the C1 densifier comparison yet.** Gate (c) of
  `2026-08-10-w-densifier-head-interpolation.md` is waiting on a native fine W
  precisely because it needs to resolve differences at the tens-of-meV level,
  and this bundle's own symmetry noise sits in that range.
* The BSE that ran on this bundle (`bse_666.log`) inherits the same problem and
  its eigenvalues should not be recorded as a reference table.
* The correction is expected to be cheap — a basis-generation constraint, not a
  physics change — and the reference is worth rebuilding once §6's leg reports.

### 9. Footnote on the original MAE

This lane's recomputation of the 6×6×6 arm-A-vs-arm-B MAE gives **25.18 meV**
where the body below reports 24.77 meV. The difference is arithmetic, not
physics: the body averaged over 976 lines, which includes the 16 k-header lines
that carry no E_QP and contribute exact zeros. Over the 960 real data rows the
MAE is 25.18 meV. The max, 110.022 meV, is unchanged and reproduces exactly.

### 10. Provenance and gates

* **Both arms, both grids, under identical `.so` and environment**, stated in
  full: `LX_BASE_MODULE=lorrax_J070`;
  `LORRAX_FFI_SO=/pscratch/sd/j/jackm/merge_ckpt_2026-08-08/build_dev/liblorrax_ffi.so`;
  `LORRAX_FFI_HOST_SO=/pscratch/sd/j/jackm/merge_ckpt_2026-08-08/build_host/liblorrax_ffi_host.so`;
  `LORRAX_FFTW3_SO=$HOME/software/lorrax_fftw_cray/stage/lib/libfftw3.so.mpi31.3.6.10`;
  `JAX_ENABLE_X64=1`; `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` (BFC@0.85). The two
  arms differ in `LORRAX_FORCE_FULL_BZ` and in nothing else, and each leg
  printed its own value of that variable into its log.
* **Instrument check.** Each leg printed `git rev-parse HEAD` from inside the
  allocation: `836905748e325fb3f59b52845a89b2257aa76d55`, zero modified files,
  on a worktree created for this lane alone. Both logs were grepped for
  ignored/unknown-key lines; the only hit is the benign `output_file (line 58):
  IGNORED` deck-key rename, present identically in both arms. `origin/main` has
  since advanced to `83222bbe`, and the three commits in between touch only
  `src/bse` and `tests` — nothing under `src/gw` or `src/centroid` — so the
  measurement stands against current main for the path under test.
* **Both sides ran.** Both arms exit 0 with artifacts; the collected counts are
  asserted, not assumed: 8 k-headers and 800 data rows per arm at 4×4×4
  (8 irreducible k × 100 bands), 6400 rows in each term file (64 full-BZ k ×
  100 bands), and at 6×6×6 960 rows (16 × 60) and 12960 (216 × 60).
* **P=4 on both arms**, per the four-GPU rule, on this lane's own allocation
  **56600390** (1 node, `salloc --qos=interactive --gpus=4`), cancelled by this
  lane's own driver at the end of the run. No other lane's allocation was
  released. An earlier attempt to fan the two arms out concurrently placed both
  P=4 legs on one node, where they collided on the JAX coordination service and
  both died on SIGTERM; that is a launcher collision and produced no
  measurement. The arms were then run serially, which at 70 s and 65 s costs
  nothing worth optimising.
* **Evidence:** `/pscratch/sd/j/jackm/symgate444_0810/` — `armA/` and `armB/`
  hold the two runs with their logs (`gw_ibz.log`, `gw_fullbz.log`), `eqp0.dat`
  and `eqp_si_gnppm.dat`; `COMPARE_eqp0.txt`, `COMPARE_anchorA.txt` and
  `COMPARE_anchorB.txt` hold the row-wise comparisons; `driver.sh`, `leg.sh`
  and `inleg.sh` are the exact scripts, including the in-leg provenance block.
  The 6×6×6 side is re-analysis of the fine-grid lane's own files at
  `/pscratch/sd/j/jackm/si666_ref_0810/{,symgate/}` and consumed no new GPU
  time.

### 11. Reproducing the k-star instrument

It needs nothing but a `sigma_diag_file` from one arm.

```python
import re, collections
PAT = re.compile(r"n=(\d+)\s+sigX=\s*([-\d.eE+]+)\s+sigC=\s*([-\d.eE+]+)\+"
                 r"\s*([-\d.eE+]+)i.*?Eo=\s*([-\d.eE+]+)")
d, k = collections.OrderedDict(), None
for ln in open("eqp_si_gnppm.dat"):
    m = re.match(r"k-point (\d+):", ln.strip())
    if m:
        k = int(m.group(1)); d[k] = []; continue
    m = PAT.search(ln)
    if m:
        g = m.groups()
        d[k].append((float(g[1]), float(g[2]), float(g[4])))  # sigX, ReSigC, Eo

# stars from the mean field: symmetry-equivalent k share their DFT eigenvalues
stars = collections.defaultdict(list)
for k, rows in d.items():
    stars[tuple(round(r[2], 6) for r in rows)].append(k)

for term, idx in (("sigX", 0), ("Re sigC", 1)):
    worst = max((max(d[k][b][idx] for k in s) - min(d[k][b][idx] for k in s))
                for s in stars.values() if len(s) > 1
                for b in range(len(d[s[0]])))
    print(f"max {term} star spread: {worst*1000:.4f} meV")
```

The count of distinct stars must equal the wedge's irreducible k count; if it
does not, the grouping tolerance is wrong and nothing below it means anything.

---

## The original finding, as recorded by the fine-grid-reference lane

*Everything below is the discovery record, unchanged except that its "What is
NOT established" section is now answered by the verdict above. It remains
accurate as a description of what was seen; the ruling-out of the ordering
artifact in particular is worth keeping.*

## What was measured

The Si 6×6×6 native-fine reference (`/pscratch/sd/j/jackm/si666_ref_0810`, built
2026-08-10, mean field and GW both new) was run twice through `gw.gw_jax` with
**one environment variable between the arms and nothing else**:

* **arm A — shipped:** the IBZ cascade. The run's own startup block reports
  `q-IBZ reduction: 16 IBZ q-points / 216 full-BZ (disk shrink 13.5×)`.
* **arm B — control:** `LORRAX_FORCE_FULL_BZ=1`, which
  `docs/architecture/memory-model.md` documents as bypassing the IBZ cascade
  entirely. All 216 q computed directly.

Same deck, same `WFN.h5`, same `centroids_frac_1104.txt`, same `dipole.h5`, same
`kin_ion.h5` — arm B consumed them as symlinks to arm A's own files, so the
inputs are not merely equal, they are the same bytes. Both arms ran at P=4 and
both exited 0.

Comparing the two `eqp0.dat` (976 rows each, 16 k × 60 bands plus 16 k-header
lines):

| column | quantity | sorted max\|Δ\| | row-wise max\|Δ\| |
|---|---|---|---|
| 0 | band / k index | **0.000000e+00** | **0.000000e+00** |
| 1 | band / k index | **0.000000e+00** | **0.000000e+00** |
| 2 | E_DFT (eV) | **0.000000e+00** | **0.000000e+00** |
| 3 | **E_QP (eV)** | **1.047861e-01** | **1.100219e-01** |

**MAE on E_QP: 2.476768e-02 eV. Max: 1.100219e-01 eV — 110 meV.**
(See §9 above: the MAE over the 960 real data rows is 2.518e-02 eV.)

## Why this is not an ordering artifact, which was the first thing checked

The obvious way to fake this result is to compare two files whose rows are in
different k-order. That is ruled out by the table above rather than by argument:
columns 0, 1 and 2 agree **bitwise, row by row, with zero difference**. The rows
are therefore aligned and describing the same (k, band) in both files, and the
mean field underneath is identical — E_DFT is bit-for-bit the same number. The
only column that moves is the one the self-energy writes. A permutation cannot
produce that pattern; it would scramble the k-coordinates and the DFT energies
too, and it does not.

## Why this matters

110 meV is not numerical noise, and it is not small against anything this
project cares about. The whole BerkeleyGW parity thread closed at **0.41 meV**
(`HEAD_SCHUR_PARITY.md`). The head-placement term the campaign spent days
attributing was **3.150 meV**. This disagreement is thirty-five times the former
and two orders of magnitude above the gate tolerances used elsewhere in the tree.

It also lands on the symmetry machinery specifically, which is load-bearing for
every claim this project makes about cost: the cascade is what turns 216 q into
16 and is quoted as a 13.5× disk shrink on this very grid. If the reduced path
and the full path do not agree, then either the reduction or the unfold is
wrong, and every symmetry-reduced number is suspect until it is known which.
*(Answered above: neither. The reduction and the unfold are both exact at
4×4×4, and at 6×6×6 they are both fed a symmetry-broken ζ.)*

## What is NOT established

I want to be exact about the limits of this, because the temptation is to
over-read it.

* **Which arm is right is unknown.** The full-BZ arm is documented as a
  *debugging* override, not a certified reference. It is entirely possible that
  arm B is the broken one — for instance if forcing the full BZ changes how the
  q→0 head is resolved, since this deck resolves the head through
  `wcoul0_source = s_tensor` and the q = 0 tile is exactly where the two paths
  most obviously differ. Nothing here adjudicates that.
  *(Answered above. Both arms are broken at 6×6×6, and the head cannot be the
  cause because the breakage is visible in Σ_x, which has no head in it. Note
  the residual limitation: the 4×4×4 anchor deck pins its head through
  `epshead`, so the agreement at 4×4×4 was measured under a different head
  route and does not by itself certify the `s_tensor` route.)*
* **Whether this reproduces at 4×4×4 is untested.** That is the single most
  valuable next leg and it is cheap: if the anchor deck shows the same split,
  this is a standing defect the whole corpus has been blind to; if it does not,
  something is specific to 6×6×6 or to this deck's head route. **Run that
  first.** *(Run. It does not reproduce: 0.0076 meV against 110 meV.)*
* **No commit is implicated.** The 6×6×6 grid did not exist before today, so
  there is no "it used to agree" to bisect against. *(Still true, and now
  expected: the suspect is a property of this deck's basis, not of a commit.)*
* This lane did not have the certificate time to isolate it further, and says so
  rather than guessing.

## Reproduction

    W=/pscratch/sd/j/jackm/si666_ref_0810
    # arm A (shipped, IBZ cascade)
    lx run -G 4 -n 4 python3 -u -m gw.gw_jax -i gw_si666_gnppm.in
    # arm B (control)
    LORRAX_FORCE_FULL_BZ=1 lx run -G 4 -n 4 python3 -u -m gw.gw_jax -i gw_si666_gnppm.in

Arm A's log: `$W/gw_666.log` (98 s, P=4, BFC@0.85). Arm B's:
`$W/symgate/gw_fullbz.log` (153 s, P=4, BFC@0.85). Tree: `origin/main` at
`83690574`, worktree `$W/wt`, HEAD verified in-leg.

The 4×4×4 twin of the same pair, which is the leg that adjudicated it:

    W=/pscratch/sd/j/jackm/symgate444_0810
    bash $W/leg.sh A     # shipped IBZ cascade
    bash $W/leg.sh B     # LORRAX_FORCE_FULL_BZ=1

**Ops note worth keeping:** arm B at `-G 1` dies with
`RESOURCE_EXHAUSTED: Failed to allocate request for 47.08GiB` — forcing the full
BZ is a 13.5× memory multiplier on the ISDF pair pipeline, so the control arm
needs P=4 minimum. That is why it is not a cheap check and probably why nobody
had run it here. At 4×4×4 the multiplier is 8× and both arms fit comfortably at
P=4, which is what makes the anchor deck the right place to adjudicate.
