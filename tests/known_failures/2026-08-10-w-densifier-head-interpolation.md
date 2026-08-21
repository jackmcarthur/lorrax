# AMENDMENT — THE COARSE→FINE W DENSIFIER TRIGONOMETRICALLY INTERPOLATED A DIVERGENT HEAD (2026-08-10) — **FIXED ON THIS BRANCH**

**This was never a red either, and it could not have been one.** Both paths
that densify W — the `bse_k_grid` deck key and `exciton_bands --w-coarse-grid`
— are opt-in, neither is exercised by any deck in the tree or on Perlmutter
(`bse_k_grid` appears only inside comment headers; no deck contains
`w_coarse`), and the suite's densification cells check shapes, solvability and
the on-grid no-op rather than the value of W between the coarse samples. The
defect was found by reading BerkeleyGW's BSE side against ours during the
Schur design pass (`~/lorrax_bse_perf_2026-08-08/SCHUR_BSE_DESIGN.md` §1.4),
not by a failing test.

The mechanism is short. `kernel.x` writes a divergence-**stripped** head, wing
and body to `bsemat.h5` and `intkernel` restores the singular factors at
assembly, per fine q, and the source says why twice in identical words:
"what we actually interpolate is only the head matrix elements (i.e. excluding
the 1/q² factor)…". LORRAX's densifier inverse-FFTs `W_q` to the coarse
R-lattice, zero-pads and FFTs back — exact band-limited trigonometric
interpolation — and until this branch its operand was the **post-injection**
tile `W_body0(q) + Δ·δ_{q,0}`, with `Δ = (whead/Ω)·conj(g₀)⊗g₀`. A Kronecker
delta is the one function shape a band-limited interpolant cannot represent.
Its interpolant is a Dirichlet kernel, so a fraction of a prefactor worth
~2600 meV on the Si anchor deck (3564 meV on the MoS₂ slab) was deposited, with
alternating sign, at fine q that should carry none of it; and the 1/q² rise
that fine q *inside* the coarse Γ cell genuinely should carry was missing
altogether, because no smooth interpolant between Γ's cell-averaged value and
its neighbours produces one. The second half is the one that matters
physically: the densified kernel under-binds excitons near the zone centre, and
the error grows with the densification factor instead of shrinking.

| item | mechanism, at this tree | disposition |
|---|---|---|
| **The coarse→fine W densifier interpolated the Γ head as a Kronecker delta, so the head channel rang in sign across the fine zone and supplied none of the 1/q² rise inside the coarse Γ cell** | `bse_io.make_w_densifier` is exact trigonometric interpolation, and its operand carried `Δ·δ_{q,0}` because the loader injects the rank-one head at `bse_io:1843` and `_interpolate_bse_data_to_grid` runs afterwards; `decimate_W_q_to_subgrid`'s docstring states the same contract from the other side ("the q=0 tile, incl. its rank-1 head, is preserved"). Measured on the shipped densifier at m = 2: the interpolated head **changes sign** — impossible for a screened Coulomb head, since `S(q) = v/(1 − 8π q̂ᵀSq̂)` is strictly positive — puts **more than half** its weight outside the coarse Γ cell, and at fine Γ stays within 5% of the coarse value where the cell average should grow like the inverse square of the cell. Not attributable to a commit: the densifier has interpolated the injected tile since the ordering was established, and both consumers are opt-in | **FIXED on `feat/schur-c1-densify-2026-08-10`, PUSHED, NOT MERGED.** The head is split off BEFORE the densifier and re-attached per fine q as `S_fine(q)·conj(g₀)⊗g₀/Ω` — the fine mini-BZ cell average at Γ, the pointwise integrand at the other fine q inside the coarse Γ cell, zero outside it (where the coarse tiles already carry their own heads through the full solve, so re-attaching would double count). Both expressions come from the ONE ratified q=0 integrand, which is **called and never modified**. New module `src/gw/head_densify.py`; one composer `bse_io.build_w_head_channel` serves both densification paths; the loader defers the injection through a new `defer_whead` on `_inject_q0_head`, kept explicitly distinct from the `w0_ready` gate so the two skips cannot be confused in the log. Default is the repaired path on both consumers; `w_head_densify = legacy` (deck key, or `--w-head-densify` on `exciton_bands`) restores the old behaviour and exists only as the A/B control. **GATES.** (a) ON-GRID IDENTITY, bitwise: with fine == coarse the re-attached array is `[whead at Γ, 0 elsewhere]` under `np.array_equal` on fcc/simple-cubic/hexagonal/triclinic cells at three grids — bitwise because the anchor is applied as `whead·(S/gamma_ref)` and the Γ entry IS `gamma_ref`, so the ratio is a float over itself; through the real loader, `bse_k_grid == coarse` returns a byte-identical bundle, and an AST cell pins that the fine grid is resolved before the head is injected. (b) THE HEAD SUM RULE: the zone average is exact at m = 1 (bitwise), converges monotonically at **16.1 / 5.1 / 4.4 / 2.8 %** for m = 2/3/4/6, and the design's RED TWIN (`gamma_cell='coarse'` — re-attach at the coarse mini-BZ scale) is invisible at m = 1 exactly as predicted and **3.3× to 6.8× worse** at every m > 1. The synthetic 16.1% at m = 2 is reproduced to three figures by the real Si deck's own log line. (c) THE A/B on the `--w-coarse-grid` harness, Si 4×4×4 anchor deck decimated to 2×2×2 and densified back, against the natively fine reference — **split verdict, and the split is the finding.** On the OBJECT, relative `‖W_dense − W_native‖`: legacy 0.74793, **C1 0.68989**, twin 0.69773 — C1 is 7.8% closer over the whole zone, **16.3% closer inside the coarse Γ cell** (0.82764 → 0.69237), and 3.8% closer OUTSIDE it, which is pure removal of the ringing since C1 adds nothing there; both arms are exact at the Γ tile and the twin is 57% wrong there. On the EXCITON SPECTRUM, C1 **loses**: MAE against native is legacy 17.6 meV, C1 28.4 meV, twin 117.2 meV. The same run says why, and it is not the head. At m = 2 — the only ratio this deck supports, since the finest bulk W grid anywhere on the system is 4×4×4 and the one 6×6×1 restart is a slab — the residual is overwhelmingly the BODY: `‖ΔW‖/‖W‖ ≈ 0.7` for *both* arms. Every arm under-binds, and they order monotonically by total deposited head weight (zone averages 2.18 twin < 3.94 C1 < 18.80 legacy, against the native 2.35; Q=0 energies 2.4952 > 2.4065 > 2.3906 > 2.3624 eV). Legacy's head is **eight times** the native zone weight, because decimation leaves the FINE cell's head average sitting on a coarse grid, and that excess attraction partially cancels a much larger body deficit — a cancellation, not a correctness signal, and demonstrably so, since legacy is farther from the native W in *every* region including the one where its eigenvalues look better. The harness on this system cannot adjudicate the head treatment on the spectrum; settling it there needs a 6×6×6 or 8×8×8 bulk restart, which does not exist today. (d) HERMITICITY: machine zero (≤1e-13 relative) by construction — `S_fine(q)` is float64 and the update is a real multiple of `conj(g₀)⊗g₀` — with a deliberately complex scalar as the red twin, breaking it by exactly `2|Im S|·|g₀|²` to 1e-10 relative; the shipped path cannot reach that twin, because `head_scalar_pointwise` refuses a complex head rather than casting it. (e) THE 0.41 meV PARITY CONFIG is untouched, structurally and by measurement: it is an on-grid configuration, C1's code runs only when a densification is pending, and no deck in the tree or on Perlmutter sets either consumer. Measured on Perlmutter, the same default-path `exciton_bands` run on this branch and at base `c3e8bda6` is **byte-identical** once the solver is converged (`--max-iter 200`). At the default 40 iterations the two differ by 3e-5 eV at X and agree exactly at Q = 0 — that X point is a six-fold near-degenerate cluster the block-Lanczos has not resolved, and it moves by 9e-4 eV between a 1-GPU and a 4-GPU run of the SAME code, i.e. thirty times the branch-vs-base difference. Convergence noise, not a delta, and the converged run settles it. (f) DEFAULT FAST GATE: **ZERO DELTA** — 11 unique failed node ids on the branch and the IDENTICAL 11 at base `c3e8bda6` (8 failed + 3 errors, 787 passed on both), every one of them the pre-existing FFI-dependent set this file already accounts for on a box with no built `.so`: the six `distrib_la` contract cells, the `vcoul` import-isolation cell, the Si BSE anchor and the three `gw_jax` regression errors. Nothing this branch touches appears in that set. The new cells are census-class and correctly deselected from the default gate (2608 deselected on the branch against 2547 at base, the difference being exactly the 61 new ones). `tests/test_w_head_densify.py`, 61 cells, fixture-free and CPU-only — **61 passed** on WSL. **TWO IMPLEMENTATION FINDINGS the design did not anticipate, both caught by their own gates.** The re-attachment domain cannot be a geometric `\|q\| ≤ \|q − K\|` predicate: it needs a tie rule, ties are generic (an even densification factor puts fine q exactly on the coarse cell boundary at every face centre), and the first cut kept **9** fine q where 8 were due on fcc and **18** where 8 were due on a hexagonal cell — an over-count no norm-based check would have seen, since the extra points carry small heads. Membership is now coset arithmetic on the indices, where the count is exactly `[Λ_f : Λ_c]` with no tolerance, and the builder asserts it. And boundary q must SHARE their weight `1/k` rather than one representative winning: `q → −q` maps coset `c` to coset `−c` preserving `\|q\|`, so a lowest-index tie-break keeps `q` and drops `−q` and the head channel stops being even in q — which is what carries reciprocity `W_q = conj(W_{−q})` through the re-attachment. A 3×3×2 → 9×9×4 hexagonal case failed outright before the weights went in. **ONE CORRECTION TO THE DESIGN'S OWN CLAIM:** §1.4 says trigonometric interpolation does not preserve the head channel's zone sum. It does — the densifier is linear with a fixed R = 0 component, so `(1/N_q)Σ_q W(q)` is conserved identically. What it fails to do is REFINE the quadrature: it re-deposits the frozen coarse answer smeared as a Dirichlet kernel, so a finer grid buys nothing. The gate is written against the refinement statement, not the conservation one. Evidence: `SCHUR_BSE_DESIGN.md` §8; `/pscratch/sd/j/jackm/c1_ab` |

## 2026-08-19 follow-up: native nonnested grids and the slab channel

The first repair still refused the requested native 8×8×1 W to 12×12×1 BSE
grid in two independent places: the body required integer nesting, and the
head weights assumed quotient-lattice cosets.  It also refused `sys_dim=2`
after stamping the dimension correctly, so a slab could choose only the
documented-defective `legacy` arm.

Branch `fix/bse-8to12-slab-densify-2026-08-19` removes only those limitations.
The body now evaluates the same coarse Fourier polynomial on any target extent
at least as large as the source, with an independent dense phase-sum oracle and
exact checks on the `gcd(8,12)=4` shared points.  Nonnested head-cell weights
come from the exact LCM common refinement: for 8→12 in one dimension they are
1 at Γ and 1/4 at each adjacent 12-grid point; the two-dimensional total is
144/64 = 2.25.  The slab Γ scalar dispatches to
`vcoul.get_kernel(2).q0_average`, and finite q obtains its Ismail-Beigi bare
factor from the public `vcoul.v_qG_table` door before applying the common
screened denominator.  No second slab Coulomb formula was added.

Verification scope: on Perlmutter JID 57269074, step 57269074.60, JAX/JAXLIB
0.9.1, the two focused files passed 82/82 and a read-only real MoS₂ 3×3×1
restart passed a nonnested 3→4 slab-C1 loader shakeout.  The latter rebuilt
the tensor from the matching sibling `dipole.h5`; its WFN is byte-identical to
the historical restart's WFN.  Evidence log:
`runs/MoS2/82_mos2_8x8_600b_scgnppm_20260819/03_exciton_dev_small/logs/focused_and_fixture.log`
(SHA-256 `e0008831792df13a94482d127cebeb8b78388420163a3217342737b2a010089c`).
This is a source/loader shakeout, not a production 8→12 exciton spectrum.
