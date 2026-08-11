> ## DIRECTION CORRECTED — THE CUT DROPS THE BLOCK, IT DOES NOT KEEP IT (2026-08-11)
>
> **The owner's ruling, verbatim:**
>
> > "we just obtain singular values and truncate, and if we're truncating in
> > the middle of a block of degenerate singular values we should truncate the
> > whole block."
>
> Everything below is correct about WHETHER a cut may split a degenerate block
> (it may not) and WHERE the boundary is (the relative-to-neighbour criterion,
> `DEFAULT_RTOL = 1e-6`, bracketed from both sides). It is wrong about WHICH
> WAY the cut then moves. It landed snapping OUTWARD — keeping the straddled
> block — and the ruling is the opposite: **the cut moves UP and the whole
> straddled block is DROPPED.** `DEFAULT_DIRECTION = "drop_block"`.
>
> **Why the ruling is right, in the arithmetic rather than by authority.** The
> section below headed "Default is `snap`, where its sibling defaults to
> `strict`" argues from a BOUND: the admitted directions are within `rtol` of
> ones already retained, so `κ_eff` moves by at most `(1+rtol)^m` — under one
> part in 10⁴. That bound is correct and it was never the issue. The issue is
> the SIGN. Dropping the block removes the smallest retained values, so
> `λ_min(kept)` rises and **`κ_eff` falls**; keeping it admits values below the
> old `λ_min(kept)`, so **`κ_eff` rises** — through the very cap
> `common/rank_criterion` sized the cut by. The landed version needed a
> `(1+rtol)^m` slack term in each call site's cap assertion in order not to
> trip its own guard; **the correction deletes that slack rather than adding
> one, and the call sites now assert the cap bare.** A block sitting at the
> rcond boundary is noise-adjacent by construction, so keeping it is adding
> ill-conditioned directions to the pseudo-inverse the cut exists to
> condition. It is also the same floor semantics as the owner's points-budget
> rule for `mu_small`.
>
> **The two-rule family, because these guards round opposite ways on purpose.**
> KEPT-SET quantities floor to a symmetric boundary — this guard (drop the
> straddled block) and `gw/downfold`'s orbit floor (largest union of whole
> orbits not exceeding the requested μ_S) both round DOWN. BAND WINDOWS
> include whole multiplets or refuse: `common/band_degeneracy` is **UNTOUCHED**
> by this correction and keeps its `strict` default and its standing rule,
> *never set `snap` to make a gate pass* — a widened window is a different
> calculation (4v4c → 4v8c, the 0.0906 eV phantom regression). The
> discriminator: **a band window says WHICH STATES exist and rounds outward; a
> rank cut says HOW MANY DIRECTIONS are trustworthy and rounds inward.** That
> paragraph now lives in `common/spectral_closure`'s docstring, where both
> guards are visible from.
>
> **What changed, mechanically.** `DIRECTIONS`/`DEFAULT_DIRECTION` +
> `resolve_direction()` (deliberately NOT environment-readable: the mode is a
> dial, the direction is a ruling); `cluster_at_cut` reports BOTH legal cuts
> (`n_keep_dropped`, `n_keep_kept`) and a direction-resolved `n_keep_closed`;
> `snap_keep_outward` → `close_keep_mask`, which gains a REVERSE cumulative-AND
> for the drop walk (still no data-dependent trip count) and keeps the forward
> one for `keep_block`. The stale key `n_keep_snapped` was REMOVED rather than
> aliased, so a stale reader gets a `KeyError` instead of a wrong number.
>
> **One new failure mode, which only the drop direction has:** a block that
> reaches `σ_max` leaves rank zero when dropped. `SpectralBlockEmptiesCut`
> refuses on it in `snap` as well as `strict` — a repair returning an empty
> basis is not a repair — and the device face carries it out as a zero count
> that `zeta_projection`'s existing zero-rank refusal catches, with that
> refusal's message now naming closure as a possible cause.
>
> **One interaction found, and it was load-bearing:**
> `rank_criterion.rank_report` computed `n_dropped_alignment = rank_criterion −
> rank_used`, and `violations()` refuses on any non-zero value because a
> round-down that depends on the DEVICE GRID makes the physics mesh-dependent.
> A closure drop is also a round-down, so after the flip `htransform` would
> have refused every run whose cut hit a block. `rank_report` now takes
> `n_dropped_closure=` and attributes it to its own column, subtracted FIRST
> and clamped to the deficit actually present — so check 2 keeps meaning "the
> mesh changed the physics", and anything left over is still a violation.
>
> **`keep_block` survives** as a source-level per-call-site opt-out for a site
> with a MEASURED reason to differ. **No site in the tree passes it**, and two
> ratchets in `tests/test_spectral_closure.py` assert that plus "the default is
> spelled exactly once". A site that turns out to NEED keep-more is a finding
> about what consumes its retained span, to be reported rather than flagged
> away.
>
> ### Evidence for the correction (2026-08-11)
>
> **The twins are re-asserted in the flipped direction, and they discriminate.**
> With `DEFAULT_DIRECTION` set back to `keep_block`, **17 cells go red** across
> both faces, all three modes and the κ argument — so these gates fail a
> deliberately keep-more result rather than merely passing a drop one. The
> criterion cell asserts `n_keep_closed == 20` AND `!= 24` by name.
>
> **CPU, WSL, worktree pin proven by `__file__` before measuring:**
> `test_spectral_closure.py` **55 passed, 0 skipped**;
> `test_rank_criterion.py` 17 (4 new). Eleven neighbouring suites:
> **235 passed, 18 skipped**, against base `d8c7a24e` **244 passed, 18
> skipped** on the same box — the −9 is the reverted `distrib_la` consistency
> cells, and the 3 red in `test_htransform_kpath_gates.py` are **identical on
> both sides** (the FFI `.so` is absent on this box, pre-existing).
> `distrib_la`'s own suite at the reverted state: **108 passed, 62 skipped**
> with the **same 6** pre-existing CUDA-library failures base has.
>
> **THE armF CONTROL, P=4 on a real 2×2 device mesh, and it is the point of
> the whole leg.** armF's cut falls in a gap (relative gap 0.315 against rtol
> 1e-6), so the flip must be **INERT** there:
>
>     ζ rank-cut closure: ARMED (mode=strict, rtol=1.0e-06) and SILENT
>     — no cut fell inside a degenerate block of C_q on any q.
>
> **Exit 0 in 129 s under `strict`**, and the retained ranks over the 16 wedge
> q come back **1 × 1095 and 15 × 1098 of 1104 — the set {1095, 1098},
> bit-identical to what this file records for the same deck before the guard
> existed and before the direction moved.** A live truncation firing on every
> q, guard armed, silent, and unmoved: the flip changes nothing where no block
> straddles. Run's own lines confirm the shape (`device mesh is 2x2 over axes
> ('x','y')`, compile-cache ARMED at 4 processes), the deck (`nval = 8
> ncond = 60 nband = 68`, `zeta_rcond = 1e-10`, deck md5 `d9f367a6…`) and the
> tree (`HEAD 4ecbc7d2`, **dirty-count 0**, printed from inside the leg).
>
> **κ_eff, restated for the drop direction on the identical construction the
> old note used** (planted block, `rel = rtol/4`, cut at 41 of 128):
>
> | block m | keep-snap κ ratio | drop-snap κ ratio | rank keep | rank drop |
> |---|---|---|---|---|
> | 2 | 1.000000250 | 0.804473828 | 41→42 | 41→40 |
> | 4 | 1.000000750 | 0.804474230 | 41→44 | 41→40 |
> | 8 | 1.000001750 | 0.804475034 | 41→48 | 41→40 |
> | 48 | 1.000011750 | 0.804483079 | 41→88 | 41→40 |
>
> The old note's "<1e-4" is reproduced exactly in column 2. Column 3 is a
> **different kind of number**: ~0.804 at every block size, i.e. a **19.6 %
> IMPROVEMENT** in κ_eff, essentially independent of m — because the drop
> moves `λ_min(kept)` across a real gap (one smooth spectral step) instead of
> sliding it within an rtol-scale cluster. On the `rank_report` probe the same
> effect reads κ 6.449e3 → 4.160e3 against a cap of 6.449e3.
>
> **A P=4 SHAPE TRAP, caught mid-leg and worth the row.** The first suites arm
> ran `lx run -G 4 -n 4`, which places **four independent single-GPU pytest
> sessions** — it printed four identical `229 passed, 8 skipped` lines and
> would have been reported as a P=4 pass. The correct shape is `-G 4 -n 1`
> (one process, four devices), and at it the same suites report **236 passed,
> 1 skipped** — **seven cells that SKIPPED under the fake shape actually ran**,
> which is exactly the mesh-dependent half. The leg now refuses outright
> unless an in-leg probe prints `MESH_SHAPE devices=4 local=4` first.
>
> **Evidence:** `/pscratch/sd/j/jackm/spectral_drop_0811/` — `wt/` (worktree at
> `4ecbc7d2`), `arm1.sh` / `arm2.sh`, `_logs/mesh_probe.log`,
> `_logs/suites_p4_n1.log`, `_logs/armF_run2.log`, and `armF_run2/`.
> Environment: `LX_BASE_MODULE=lorrax_J070`, the `merge_ckpt_2026-08-08` `.so`
> pair, `JAX_ENABLE_X64=1`, BFC@0.85 — the same environment armF was
> originally measured under. `bandwin666_0810/armF` was read READ-ONLY through
> symlinks.
>
> Read the rest for the mechanism, the tolerance derivation, the site sweep and
> the original armF evidence — all of which stand. Only the direction moved.

# AMENDMENT — SPECTRAL CUTS NOW HAVE THE CLOSURE GUARD BAND WINDOWS HAVE HAD SINCE 53fd80ea (2026-08-10)

**The owner's question was "did we finish enforcing symmetries (no degeneracy
breaking) in the singular values kept for any of that stuff so this doesn't
happen?"  The answer was NO — at every rank/rcond truncation in the tree —
and this amendment is the yes.**

`2026-08-10-ibz-cascade-vs-full-bz-sigma-6x6x6.md` §6 conjectured that a rank
cut through a degenerate block of the ISDF Gram is not point-group covariant,
and its own §RESOLUTION then REFUTED that as the cause of the 6×6×6 break —
the cause was the band window one index over. What the refutation did not do,
and said so, is make the hazard go away. It showed the 6×6×6 ζ truncation was
whole-star covariant **by measurement after the fact** (0 of 16 q-stars carried
a non-constant `n_keep`), which is luck rather than enforcement. Nothing in
the tree stopped the next deck's cut from landing mid-block.

## What the sweep found

Every spectral-truncation site in `src/` and `services/`, classified.

### Needs the guard — a cut through a possibly-degenerate spectrum whose retained span feeds a symmetry-covariant object

| site | what is cut | why it needs it |
|---|---|---|
| `src/isdf/core.py::_charge_factor_math` (`rank_truncate`) | eigenvalues of the charge `C_q` at `zeta_rcond·λ_max` | **the seam §6 was about.** `C_q` commutes with the point group when the centroid set is orbit-closed and the window degeneracy-closed, so a symmetry maps each eigenspace onto itself and mixes a degenerate block freely. Cut between blocks and ζ's span is invariant; cut through one and `C_{Sq} = P C_q P†` does not survive the truncation |
| the same, `transverse_rank_truncate` | `\|λ\|` of the Hermitian INDEFINITE transverse CCT | same argument, and the cut is on magnitude with both signs physical |
| `src/isdf/core.py` distributed tier `_masks`, both channels | the same two cuts, different execution | same argument; **plus** the identity pad's exact-1.0 eigenvalues are mutually degenerate to the bit, so an unguarded block walk would swallow all `n_pad − n_log` of them and make the retained rank a function of the DEVICE COUNT |
| `src/common/zeta_projection.py::least_squares_transfer` | eigenvalues of the small-basis Gram `G_S` per q | `W_S` is the projection onto the retained span; a block-cutting rank makes that span differ between q and Sq |
| `src/gw/downfold.py::select_cur_centroids` | eigenvalue rank of the pool Gram — **the μ_S ceiling** | the knob-trap's own number. μ_S = ceiling with the ceiling mid-block selects a symmetry-arbitrary slice, and the CUR pivot order is then choosing between directions the spectrum cannot distinguish |
| `src/gw/downfold.py::build_transfer` | per-q `S_SS` spectra | the solve's cut, restated at the downfold's authoritative report — `build_transfer` suppresses `least_squares_transfer`'s own log line, so without this the guard would fire inside the solve and the downfold would say nothing |
| `src/bandstructure/htransform.py` | singular values of ψ-at-centroids | that function's own NUMERICS note (a) already records the rotation freedom "inside degenerate σ groups". It is harmless while a group is retained WHOLE and is exactly what breaks covariance when the group is split |

Report-only, because it selects nothing: `src/centroid/pivoted_cholesky.py::point_granularity_rank` hands an operator a number to size `n_keep` by. It now carries `point_rank_closure_note`, kept OUT of its `reason` field on purpose — that field is contracted to mean "the measurement was skipped", and its caller only prints it in that case.

### Structurally safe, with the reason stated

| site | why it is safe |
|---|---|
| `src/bse/vq_interp.py::_clean_split_body` | **not a cut at all.** `g(λ) = λ²/(λ² + (ε λ_max)²)` is a smooth ANALYTIC filter applied as `S = R g(Λ) R^H`, i.e. a function OF the operator. Degenerate eigenvalues get identical weight, so `S_q` commutes with everything `C_q` commutes with and no basis choice within a block matters. This is the model repair, and worth knowing about: a filter needs no closure guard because it never chooses a subspace |
| `src/bse/w_omega_chain.py::_gram_factor` | a DEFLATION FLOOR at `p·ε·λ_max`, not a rank selection — it deletes exactly-parallel or zero probe columns. The chain carries both `R` and `Tr` with `W = Q R` exactly, so its output is invariant to the basis within a retained block; only the count could move, and a cluster straddling machine epsilon means the seed block is numerically singular, which is what deflation is for |
| `src/gw/mpa/pade_fit.py::_solve_normalised` | **out of this argument's scope**, and the honest statement is that it is a different question. The cut is on the singular values of a per-(q, μν) Padé design matrix at `rcond = 1e-13`, a numerical-safety floor. There is no point-group action on that matrix's singular vectors — MPA's covariance question is "does MPA-of-a-rotated-W equal rotation-of-MPA", which is a property of the element-wise fit and is not answered by cluster-snapping a design-matrix spectrum. **Not wired**, deliberately, and flagged as an owner row if MPA lands |
| `src/bse/exciton_bands.py` (`svd(G[k])`), `src/bse/bse_feast.py` (`lstsq(rcond=None)`), `src/gw/minimax_screening.py` (`lstsq(rcond=None)`) | diagnostics and Krylov-solver internals. No physics subspace is selected |
| the Rayleigh–Ritz eigh in `solvers/{lanczos,davidson,thick_restart,feast}` | the retained set is by eigenvalue COUNT at a window boundary, which is `common/band_degeneracy`'s question and is already owned there |

## The guard

`src/common/spectral_closure.py`, sibling of `common/band_degeneracy`, L2 in
`tests/test_layering.py` beside `common/rank_criterion`.

**The tolerance is relative-to-NEIGHBOUR, and this is the one place the
analogy with the band-window guard breaks.** Band energies live on one scale,
so an absolute tolerance in Ry works there. Spectral cuts sit at
`σ ≈ σ_max·rtol`, eight to ten decades below the top of the spectrum, and a
tolerance measured against `σ_max` would declare the whole retained tail one
block. So two values are in the same block when `|σ_i − σ_j| ≤ rtol·max(|σ_i|,
|σ_j|)` — scale-free, equally sharp at every depth.

`DEFAULT_RTOL = 1e-6`, bracketed by measurement from both sides:

* **from above**, the gap it must not swallow: the fixed 6×6×6 `armF` arm's
  tightest cut has `λ_min_kept/λ_drop_hi = 1.46`, a relative gap of
  **0.3151**, and that arm's Σ_x k-star spread is exactly 0.0000 meV. 1e-6 is
  five decades clear of it.
* **from below**, the noise floor: symmetry-degenerate eigenvalues are
  computed with backward error `O(ε·σ_max)`, so their achievable RELATIVE
  agreement near the cut is only `ε/rcond` — 2.2e-8 at the production
  `zeta_rcond = 1e-8`, and **2.2e-6 at the 1e-10 the 6×6×6 deck used, which is
  ABOVE the default**. `degeneracy_noise_rtol` computes it and every report
  prints it beside the tolerance, saying so when the tolerance is below it: a
  guard looking for agreement finer than the arithmetic delivers finds nothing
  and reports a clean bill.

**Default is `snap`, where its sibling defaults to `strict`, and the reason is
arithmetic rather than taste.** A band-window snap is a different calculation —
4v4c → 4v8c, 1024 dimensions → 2048, a gate reading 0.0906 eV of regression no
branch caused. A closure snap admits directions within `rtol` of ones already
retained, so `κ_eff` moves by at most `(1+rtol)^m` over an m-member block:
**under one part in 10⁴ for any block a crystal can produce**, against
`rank_criterion`'s R19 anchor where +41 % of rank cost 5000 eV. Refusing that
by default would be refusing the repair. `strict` is one word away, the
post-snap `κ_eff` is reported every time, and `rank_criterion.violations()`
still guards the cap independently. **Owner row: whether `strict` should
become the default here too is the owner's call; it is a single constant.**

Two execution surfaces over one criterion, because the ζ cut lives inside a
jitted kernel that never brings its eigenvalues to host: `resolve_spectral_cut`
is host numpy with full messages, and `snap_keep_outward` is pure `jnp`,
batched, jit- and vmap-safe — a prefix-AND over adjacency links, so its trip
count does not depend on the data. A jitted kernel cannot raise, so under
`strict` a device site records through a host callback and
`raise_if_pending` refuses at the next host seam (`gw.gw_init`, immediately
after the fit and before ζ is consumed). That is `pivoted_cholesky`'s own
division of labour, and it is what stops one seam refusing while another
whispers — the failure `band_degeneracy` names in its own docstring.

## The coordination row with the q_irr lane, which is the same principle one index over

`2026-08-10-downfold-qirr-star-stability.md` measured that the CUR selection's
`keep_idx` is not orbit-closed: pivoted Cholesky fills orbits greedily but
stops at exactly μ_S, generically mid-orbit (7 of 46 admissible μ_S closed on
the synthetic). **That is this guard's disease in the selection rather than in
the spectrum, and the identification is exact rather than an analogy:** at
q = 0 the selection Gram commutes with the whole group, so every member of a
centroid orbit carries the identical Schur diagonal. An orbit IS a degenerate
block, and the pivot order's index-order tie-break inside one is precisely the
round-off-chosen slice this module refuses to let a rank cut be.

So the repair is the same repair. `select_cur_centroids` now takes `sym_perm`,
completes the kept set OUTWARD to whole orbits, and **re-takes the rank
certificate and the `auto` ceiling on the completed set** — against the
EIGENVALUE rank and never the selection certificate, which is the knob-trap
discipline that function is built around. Completion crossing the ceiling
REFUSES by name, and says which number is over it: the requested μ_S is not,
the symmetry-legal μ_S nearest it is, and the fix is a smaller μ_S at an orbit
boundary or a wider window — never a looser `downfold_rcond`.

Fence held: the q_irr lane owns the plumbing (child unfold tables, the wedge
writer), this lane owns the completion and the re-certification it forces.
`run_downfold` passes `sym_perm=None` today and **says so loudly** — building
the parent's centroid source map needs the symmetry ops and the centroid table
together, which that driver does not load, and it belongs with the wedge-writer
plumbing. Until then closure is UNMEASURED there, which is an absence and not
a pass, and the log says exactly that.

## Gates

`tests/test_spectral_closure.py`, **46 cells, 0 skipped**, every guard a
TRUE/FALSE pair because a guard that fires on both arms is not a guard.

* **the criterion**, TRUE (a cut inside a planted 4-fold block snaps outward
  to the block boundary, parametrized over three offsets) against FALSE (a
  featureless power-law spectrum fires at no cut anywhere, six positions) —
  plus the cut the guard MOVES to being itself clean, which is the difference
  between fixing the problem and relocating it.
* **the three modes**: `strict` names the block and the rank that works,
  `snap` says `SNAPPED OUTWARD` and names the move, `off` is silent and
  changes nothing, and the FALSE arm is silent in ALL THREE.
* **the armF gate** — the deck this whole saga produced must stay silent. Built
  on the recorded `λ_min_kept/λ_drop_hi ≥ 1.46`: the relative gap is 0.3151 and
  the guard must be five decades from firing. Its TRUE twin (an exact
  degeneracy at the same position) fires, so the armF gate is a
  discrimination and not a vacuous pass.
* **the bound that makes snap the default**, asserted: over blocks of 2, 4, 8
  and 48 members, `κ_snapped/κ` ≤ `(1+rtol)^m` and < 1.0001.
* **the two surfaces agree**, swept over descending / ascending / shuffled
  input order and three cut positions, because the charge route hands the
  device face ascending `eigh` output and the transverse route hands it an
  indefinite spectrum. Plus: an indefinite spectrum cuts on `|λ|`; the batch
  axis does not leak (one dirty q must not move a clean one); an exactly-null
  pad tail is never swept in; and the distributed tier's identity pad is
  withdrawn from the walk so the retained rank cannot become a function of
  `n_pad`.
* **the selection site**: TRUE (a mid-orbit selection is completed outward,
  never dropped, and the report's μ_S is the delivered length) against FALSE
  (an already-closed selection passes through `array_equal`), plus the
  ceiling refusal firing on the COMPLETED count with the right message.
* **two ratchets**: no site passes a mode literal as a default, and every
  wired site still calls the guard by name, so dropping a site from the
  wiring fails by name rather than as a silently unguarded cut.

**Suite A/B, BOTH SIDES RUN, WSL CPU, worktree pin proven by `__file__`
before measuring.** Eleven neighbouring suites: branch **238 passed, 19
skipped**, of which 46 are the new file; base `47657990`, same eleven suites,
same box, **192 passed, 19 skipped**. Delta is exactly the 46 new cells;
zero regressions.

## The cluster evidence

**P=4 on every GPU leg** (the four-GPU rule; the ζ kernels are a GPU path and
this branch changes them). One combined leg, doctrine rule 2.

* **Arm 1, the suites at P=4** — nine suites on a real 4-GPU node:
  **220 passed, 8 skipped, exit 0 in 227 s**, and the count is reported
  identically by all four ranks (206.8 / 209.9 / 209.9 / 216.3 s), so this is
  four ranks agreeing rather than one rank speaking.
* **Arm 2, THE armF SILENCE PROOF, and it is positive rather than an absence.**
  The `armF` (`nband = 68`) deck re-run under `LORRAX_SPECTRAL_CLOSURE=strict`
  — the mode that REFUSES if any of the 216 q cuts inside a degenerate block
  of `C_q`. **Exit 0 in 133 s at P=4**, and the run's own line reads

      ζ rank-cut closure: ARMED (mode=strict, rtol=1.0e-06) and SILENT
      — no cut fell inside a degenerate block of C_q on any q.

  Exit 0 under `strict` IS the proof: had the cut landed in a block anywhere,
  the run would have refused before ζ was consumed. The `ARMED` line is the
  instrument check — it proves the guard was REACHED, not merely that nothing
  was said.
* **And the cut did not move.** The retained ranks over all 16 wedge q come
  back as exactly **{1095, 1098}** of 1104 — bit-matching the values this
  saga's RESOLUTION recorded for the same deck before the guard existed. A
  live truncation, firing on every q, with the guard armed and silent: which
  is precisely the configuration §6 said should be possible and could not
  demonstrate.
* Deck window verified in-leg: `nval = 8 ncond = 60 nband = 68`,
  `zeta_rcond = 1e-10`, `q-IBZ reduction: 16 IBZ q-points / 216 full-BZ`, and
  the band-window guard's own line `edge 68 min gap 106 meV`.
* **Provenance.** Tree `3423d51b`, **dirty-count 0**, both printed from inside
  the allocation; `[lx] source tree:` confirmed the worktree.
  `LX_BASE_MODULE=lorrax_J070`, the `merge_ckpt_2026-08-08` `.so` pair,
  `JAX_ENABLE_X64=1`, BFC@0.85 — the same environment the armF arm was
  originally measured under. Ran in the shared pool `56593605`;
  **no allocation was created or cancelled by this lane** (the interactive QOS
  submit limit was already reached, and `56606148` belongs to the
  convergence-triangle lane and was not touched). `bandwin666_0810/armF` was
  read READ-ONLY through symlinks; nothing there was written or deleted while
  another lane was reading it.
* **Evidence:** `/pscratch/sd/j/jackm/spectral_closure_0810/` — `wt/` (the
  worktree), `_logs/suites_p4.log`, `_logs/armF_strict.log`,
  `_logs/driver.out`, `_logs/arm2_driver.out`, `leg.sh` and `arm2.sh` (the
  exact scripts, including the in-leg provenance block), and
  `armF_strict/` (the run directory).

**The limit, stated rather than buried.** No positive control was run ON THE
GPU PATH: there is no env knob for `rtol`, so this lane did not force the ζ
kernels to fire and watch them refuse. That the device face discriminates is
established on CPU instead — the TRUE/FALSE pairs and the
jit-face-matches-host-face sweep over three input orderings — and on the real
path by the rank values above being unchanged. A cheap way to close it if
anyone wants it: expose `rtol` as an env dial and re-run armF at `rtol = 0.5`,
which must then refuse by name.
