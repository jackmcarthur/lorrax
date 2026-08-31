# OPT-S result — heavy investigation, 2026-08-31

## Numbers first

**No valid two-window plan was produced.**  On the frozen real sodium measure,
the requested semicore delivery `-55:-49 eV` at `eta=1 eV` had
`A/gamma=90.0994`.  A fixed rank-182 ROQ fit took 57.72 s and achieved
residual `1.43576e-2` against `1e-4`, `kappa_p99=46245.3`, and
`kappa_max=1.39082e6`; both the residual and runtime-noise gates failed.
The `0:5 eV`, `eta=0.25 eV` control at rank 92 took 5.33 s and achieved
residual `3.42066e-5`, but `kappa_p99=71996.9` still failed the noise gate.

The archived 29-k x 24-band deck spans `-53.804613 .. +19.272703 eV`
relative to `mu=1.64676 eV`.  Bands 1-2 lie at
`-53.804613 .. -53.801500 eV`; bands 3-8 instead lie at
`-25.290430 .. -25.012469 eV`.  Therefore `-55:-49, -5:5` covers two
semicore bands (58 IBZ samples), not eight; it leaves bands 3-8 in a
24-eV-wide omega-grid hole.  The requested claim that bands 1-8 all sit near
-52 eV is false for this named deck.

## What the existing patch path actually supports

`sigma_omega_patches_ev` builds a disjoint union and the planner decomposes
its omega clusters, but the run still has one scalar
`sigma_regularization_ev`, one scalar `sigma_omega_step_ev`, and one
`regularization_width_ry` passed through planning and execution.  Thus it
cannot represent "main eta=0.25 eV plus semicore eta=1 eV".  Adding only the
requested patch would also make the current hull-based band partition treat
the -25 eV bands as protected; the QP hole guard must then refuse rather than
interpolate them across the gap.  No new deck dial or partial executor was
added.

The frozen p3 bulk measure gives the single-wide `-55:5 eV`, `eta=0.25 eV`
control `A/gamma=572.848`.  The measured cost law prices it at 1157 nodes;
the derived rank ceiling is 1736 before the hard cap of 512.  The split
semicore measure prices at 182 nodes, but its achieved fit above is not
accepted, so neither a total `(window,tau)` count nor finite semicore QP
energies exists.  Consequently scissor `alpha/beta/RMSE` was not fabricated
from the old endpoint-clamped QP file: the only defensible achieved scissor
number is 58 physically covered valence samples versus 0 on the mu-only
window.

## Evidence and disposition

Inputs were the frozen
`runs/DEV/80_minimax_delivered_error_toy_20260828/.../na_reconstructed_problems_v1.npz`
and the archived sodium `test_delivered_24b/eqp0.dat`.  The fit used the same
p3 fit/validation masses, shifted only by `eta_new-eta_old`, on 25 uniform
omega points.  The required CPU gate passed `134 passed` in 91.69 s.  No GPU
leg was submitted because the requested per-patch
broadening is not representable and the real deck geometry already proves
the proposed two-patch run must refuse.  Branch:
`feat/semicore-window-2026-08-31`.
