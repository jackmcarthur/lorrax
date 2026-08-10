# AMENDMENT — `--vq-mode interp` IS NOT AVAILABLE ON A 3-D BULK DECK, AND PUNCH ROW 23 WAS NEVER AN OFF-GRID DEFECT (2026-08-10) — **MEASURED, ATTRIBUTED AND REFUSED ON THIS BRANCH**

`PIPELINE_HEALTH.md` punch row 23 opened as a **High** blocker: on
`si_bse_debug`, `bse.exciton_bands --vq-mode interp` failed two of
`vq_interp`'s gates — `makeVq_vs_disk_Vqmunu_allq_max` at 3.218e-01 against a
5e-6 tolerance and `slab_axes_offdiag` at 1.000e+00 against 1e-12 — and did so
identically on the parent bundle and on a downfolded child. The row read that
as "a defect of the off-grid exchange on this deck". It is not a defect, and
it is not off-grid. Both numbers are `vq_interp`'s slab assumption meeting a
3-D fcc cell, and both are produced by `run_gates`, which runs on the coarse
data before any interpolation has happened at all.

**The first suspect was the wrong one, and the model was already innocent.**
The obvious hypothesis was that the long-range model interpolates the product
of the form factor with the Coulomb kernel, so that the kernel's structure
pollutes a fit that should only ever see a smooth amplitude. It does not.
Stage 1 samples `Fch = e^{+2πi(q+G)·s_μ} (S_q ζ̃)_μ(q+G)` — the phase-factored
cleaned form factor, with no `v` in it; stage 2 solves a least-squares problem
whose right-hand side is exactly those samples and whose *weight* is
`v_LR(q+G)`, so the objective is `‖ΔA‖²_F` for the tile factor without the
kernel ever entering the fitted object; and stage 3 rebuilds
`v = 8π/K² · f2d/Ω · e^{−K²/4α²}` in closed form at the target Q and applies it
as `A = zt·√v`. The stripped-amplitude design was already in force, at one
site, and nothing in this landing changes the fit.

**What the residual actually is, measured three ways on the real deck.**
Rebuilding the stored `V_qmunu` from the stored ζ at all 64 q of the
`si_bse_debug` parent (n_μ = 936), varying only which `v(q+G)` is used:

| `v(q+G)` used to rebuild the tiles | `makeVq_vs_disk` max |
|---|---|
| `vq_interp`'s hardwired 2-D Ismail-Beigi slab kernel | **3.218e-01** — reproduces the punch row exactly |
| bulk 3-D `8π/K²/Ω` with the deck's 25 Ry bare-Coulomb mask | 4.593e-02 |
| the same, plus the deck's mini-BZ head-slot injection | **1.504e-15** |

So the gate is measuring kernel bookkeeping and nothing else: given the `v`
the deck actually built its tiles with, the stored ζ reproduces the stored
`V_qmunu` to machine precision. The dominant term is the truncation family —
`vq_interp` has no `sys_dim` anywhere in it and applies the slab `f2d` to a
bulk cell — and the remainder is the deck's own `mc_average_vcoul_body = true`
mini-BZ average, which the row's own "next step" column had already guessed at
and which turns out to be the smaller half.

**The second gate cannot be tightened away, and is the one that closes the
question.** `slab_axes_offdiag` = 1.000e+00 is not a small number near a
strict tolerance. On `si_bse_debug`'s reciprocal lattice
`b3 = 0.6123·(1, −1, 1)`, so `b3`'s in-plane components equal its own
z-component by construction, and `b1`, `b2` carry z-components of the same
size. The long-range model fits one in-plane polynomial `M_μ(K_x, K_y)` per
`|G_z|` channel, which is exact only where `K_z` is constant inside a channel —
i.e. `b3 ∥ z`, `b1, b2 ⊥ z`, and `q_z = 0` on the coarse grid, which together
give `K_z = G_z·|b3|` identically. On this cell none of the three holds
(`max|q_z| = 0.5`), so the fit would be regressing against a variable it cannot
see. **No change to the Coulomb kernel reaches that**, which is why fixing the
kernel alone would have moved 3.218e-01 to 4.593e-02 and still failed.

**Disposition: refused by name, at the loader, before anything expensive.**
`vq_interp.slab_scope_violations` states the three conditions once — the
restart's own Coulomb-policy stamp, the axis ratio, and the planarity of the
coarse q-grid — and `load_zeta_coarse` applies them, so the deck now gets a
sentence naming each violated condition and the two modes that do serve it
(`--vq-mode ongrid`, exact at every Q on the BSE grid, and `--vq-mode refit`)
instead of a gate battery whose numbers are consequences. Worth saying plainly,
because it is the part that stings: the stamp that answers this was **already
printed in the failing run's own log**, one screen above the gate lines
(`sys_dim=3`, `mc_average_vcoul_body=true`). Nothing was missing. Nothing read
it. The fix is therefore a refusal that quotes the stamp, not a second copy of
it.

| item | mechanism, at this tree | disposition |
|---|---|---|
| **`bse.exciton_bands --vq-mode interp` refuses on `si_bse_debug` and on every 3-D bulk deck** | `vq_interp` is a slab model in two independent ways: its only Coulomb kernel is the 2-D Ismail-Beigi truncation, and its long-range model's per-`\|G_z\|` channels are exact only on slab-separable axes with a planar coarse q-grid. A bulk deck violates both. | **BY DESIGN, and now said out loud.** The refusal names all three violated conditions and the two working modes. `--vq-mode ongrid` is exact at every Q on the BSE grid and is what the walk's step 6 already used successfully on this deck. Making `interp` serve a bulk cell is not a tuning change — it needs a long-range model whose channels are not keyed on `G_z`, which is owner-ruled work and is not registered as in progress. |

**Gates.** All on Perlmutter, shared pool, one GPU each. (a) The row-23 pair
re-measured on the parent's own artifacts, with the three-arm ladder above;
the arms differ only in `v(q+G)`, everything else is the deck's stored data.
(b) **Zero delta on a served deck, byte-level**: the MoS2 3×3 slab control ran
`bse.exciton_bands --vq-mode interp` end to end at base `c3e8bda6` and on this
branch, and `xs_slab_base.dat` / `.png` are md5-identical to
`xs_slab_interp.dat` / `.png`; its gate battery reads `makeVq_vs_disk`
1.304e-09 and `slab_axes_offdiag` 0.000e+00 on both, and all three
`run_nulls` are green. That deck's restart is **unstamped**, which also
exercises the `policy=None` arm on real data — an unstamped file is judged on
geometry alone and is not refused for lacking a stamp. (c) The fixture-gated
`tests/test_bse_vq_interp.py` LOO ladder returns **bit-identical** numbers at
base and on the branch (LOO median `0.07956779918419911` on both), which is
the strongest available statement that the arithmetic did not move; that cell
is the pre-existing red this file already carries and it is left standing.
(d) `tests/test_bse_vq_interp_scope.py`, ten fixture-free cells on cell
geometry alone, including the tetragonal twins that separate the kernel
condition from the geometry one. (e) The BerkeleyGW parity deck structurally
cannot execute this module: `bse.bse_jax --bse` reads the stored tensors
on-grid, and `bse_io`'s only `build_vq_evaluator` call sits inside
`_interpolate_bse_data_to_grid` behind `head_minibz_average`, which that deck
sets neither of.

Evidence: `/pscratch/sd/j/jackm/xd_parent` (the punch row's own run and its
`xb_parent_interp.log`), `/pscratch/sd/j/jackm/xs_scope` (the refusal, on the
branch), `/pscratch/sd/j/jackm/xs_slab` (the MoS2 control, base and branch),
and `/pscratch/sd/j/jackm/row23_measure.py` + `row23_arm3.py` (the three-arm
ladder, numpy and `vcoul` respectively).
