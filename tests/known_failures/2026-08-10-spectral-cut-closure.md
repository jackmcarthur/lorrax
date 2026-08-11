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
