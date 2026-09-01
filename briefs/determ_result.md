# DETERM result — heavy lane

## Numbers first

| check | achieved result |
|---|---|
| amended tips | **5/5 passed** the strengthened production-planner P matrix: `95c7160b`, `f0135b9b`, `cbd3dc7c`, `fe3fb6cf`, `8b3ed208` |
| repeated P signatures | **30/30 identical** across the five tips: two plans at each of P=1,4,16, including ordered windows, node-time bytes, weight bytes, masks, omega/pole indices, node counts, and plan-cache fingerprint |
| warm-start/order probe at `f0135b9b` | **4/4 identical** (ascending, descending, scrambled rank inputs, then repeat): rank 10, residual `2.544646335001577e-05`, kappa-p99 `1.221612999034461`, byte SHA-256 `e1c5ca6e416b7435a46b960e730410f1372740bd5c62781fd43c7cc50f2b385d` |
| prescribed CPU gate | **134/134 passed**, 8 warnings, 78.73 s |
| focused audit gate | **3/3 passed**, 1 warning, 9.15 s |
| current-planner QP comparison | **0 QP rows available**: the P=4 one-shot at `f46bfd51` refused before planning and wrote no `eqp0.dat`/`eqp1.dat` |
| registered comparator | **0/2 BGW arms parsed** and **0 current-format LORRAX rows parsed** |

## Determinism verdict

No determinism failure was found, so no blocker was added to
`KNOWN_LORRAX_ISSUES.md`. The five tips are siblings, not one integrated tree;
the result is therefore a per-tip clean bill, not evidence for their future
merge. The P-count test runs every deterministic owner partition in-process;
it is exact for planner assembly but is not a real MPI P=16 execution.

The warm-start premise was slightly stale at the measured tip: final IRLS is
seeded from the **same rank's** quick fit, not a neighbouring rank. The API
sorts the candidate ranks before probing. Reordering those inputs and rerunning
the fit preserved all scalar fields and every node/weight bit. This audit adds
a permanent rank-order test and extends the P-independence test to two repeats
per P plus the cache fingerprint.

## BerkeleyGW comparison: blocked, no invented energies

The current one-shot is
`50_delivered_plan_20260829/pointwise_dp_p4_20260831`, source `f46bfd51`.
All four ranks refused at `number_bands_chi=48`: the WFN has exactly 48 bands,
so closure of the terminal multiplet cannot be checked. This occurs before the
planner and leaves no QP file. I did not use the forbidden `snap` escape.

The independent reference convention is BerkeleyGW full-frequency contour
deformation (`frequency_dependence 2`), absolute Fermi level, `dont_use_vxcdat`
and `use_kihdat`: QP energy is absolute **KIH + Sigma**, in eV. The LORRAX
deck requests MPA `one_shot_dft`, head off, `mc_average_vcoul_body=false`, also
absolute **KIH + Sigma**, in eV. Thus the accounting is explicit, but there is
no current LORRAX value to subtract.

The registered `tools/compare_bgw_gwjax.py` independently blocks both BGW
arms at `ik=1`: their rows are the full-frequency layout
`n Emf Eo X ReRes ReInt ReSig ReCor KIH ReEqp0 ReEqp1 Soln`, not either parser
layout. Its LORRAX reader also expects `sigma_diag.dat` prose fields and parses
zero rows from current BGW-format `eqp0.dat`. The registered parser lives in
the sandbox, outside this lane's source worktree, so I reported the mismatch
instead of hand-rolling extraction or editing another worktree.

## Evidence and branch

Focused JUnit artifact: `evidence/determ_20260831/focused.xml`. CPU cells only;
no GPU production path changed, so the four-GPU rule's CPU exemption applies.
Branch `audit/determinism-bgw-2026-08-31`, base `5f750a77`.
