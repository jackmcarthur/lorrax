# Contour-deformation frequency integration of Σ_c (`gw.cd`)

`sigma_frequency_integration = contour_deformation` evaluates the correlation self-energy on the
ISDF screened interaction W(z) itself (χ⁰ from the bank quadrature) instead of a pole model.  Only the body kernel changes: dispatch
(`gw.sigma_dispatch._compute_mpa_sigma`), η, the Fermi reference, occupations and the one finalizer
(`finalize_dynamic_sigma`) are the dynamic-Σ path every ansatz shares.

This is LORRAX's only contour-deformation implementation.  χ⁰ is never a Lehmann sum on this path;
the ordered Lehmann χ⁰ (`w_isdf.compute_chi0_direct_fractional(ordered=True)`) appears only as the
reference of `tests/test_cd_bank_chi0.py`.

## What is computed

For the pair (k, k−q) and intermediate band l, with x = ω_E − ε_l(k−q) and the contractions

    a[k,m,n,l] = −(1/N_k) ⟨k m, l| W_c,q(z) |k n, l⟩,   d = the same of dW_c,q/ds,  s = z²,

the deformed contour gives (Ry throughout, one η = `sigma_regularization_ev`)

    Σ_c(E) = A(E) + Σ_i I_i(E) − S_A(E) + Σ_j R_j(E).

* I_i: the imaginary axis z = iu_i, a two-panel Gauss–Legendre rule under u = η tan θ
  (`gw.cd.rule`, `cd_imaginary_nodes`), carrying a and its dagger by branch.
* A, S_A: a two-pole anchor fitted to W_c and dW_c/ds at z = iη (β = 0.5, 4 eV), integrated
  analytically, and its share of the node sum, subtracted.
* R_j: the real-axis residues of G, read from the knots z_j = jh + iη (`cd_knot_step_ev`,
  h ≤ η/4) by cubic Hermite interpolation in |x| using value and d/dx = 2z d/ds.  Unoccupied l
  (x ≥ 0) read W_c,q; occupied l (x < 0) read W_c,−q^T.  Reach covers every live |x| of the ω grid
  and of the window's own energies; a live residue past the last knot refuses.

The laws are `gw.cd.laws`, pinned on CPU against the ordered (time-reversal-exact) harness formulas
(`tests/test_cd_laws.py`, `tests/fixtures_cd/ordered_cd.py`).

## Where each piece comes from

| quantity | owner |
|---|---|
| χ⁰(z) at every node and knot | the tangential bank's response quadrature on the CD memory windows (`gw.cd.sigma.bank_chi0`, rules from `gw.cd.chi0_rules`): one rule request per chunk the streams evaluate; an insulator's imaginary windows take the ordered `minimax.response_laplace_rule` (positive real times, certified; a refused point takes the bank tree when its pencil is small, else the even-only rule on a time-reversal-authenticated deck, else `GATE cd_chi0_rule_unfit`), the knot windows `response_bank.response_quadrature` (`minimax.ResponseGroupTree` at `cd_chi0_rule_capacity` nodes and the CD mass gate max(5000, 0.01 tol/eps)); evaluated by `response_bank.integrate_response_group` (`ordered=True`, value rows). No Lehmann sum (`compute_chi0_direct_fractional` is a test reference only). Rules are precomputed with `python -m gw.cd.chi0_rules` and loaded with an identity check (`cd_chi0_rules_file`) |
| dχ⁰/ds at a knot | 9-point central d/dx on the knot grid, (dχ/dx)/(2z); ghost knots roll from chunk to chunk |
| W_c, dW_c/ds | the response bank's pair equations, X = pref·HχH, W_c = HX(1−X)⁻¹H, dW_c/ds = W(pref·dχ)W, as a local batched LU over the (q, z) flat axis (`gw.cd.sigma.build_dyson_kernels`) |
| W at star q | `symmetry_maps.QirrOperator.unfold` with the general-operator `pair_transpose` rule |
| Σ_c(k) at star k | `symmetry_maps.unfold_file_wedge_band_operator` (rule `transpose`) from the file wedge |
| the contraction | `gw.cd.sigma.build_contraction_kernels`: one `shard_map` over q; pair densities ρ_q[k,m,l,μ] recomputed per q and per band tile from the rank's pair slab; on the real axis only the banded rows of each knot chunk (below) |

## Banded residue rows (real axis)

A Hermite residue at (E, l) reads only the two knots bracketing |x| = |E − ε_l(k−q)|
(`laws.HERMITE_STENCIL`, lower knot `laws.hermite_lower_knot`), on one branch: x ≥ 0 (weight −(1−f),
W_q) or x < 0 (weight f, W_−q^T).  A (k, l) row none of whose residues reads a chunk's knots adds an
exact zero there, so `residue_band_tables` (host, once per run, one bit per branch × q × k × l × chunk)
lists for each chunk and branch the live bands of every (q, k) row (E over the ω grid and the row's own
energies; stencil widened by one knot each side, so a last-ulp difference between numpy's and XLA's
|x|/h cannot drop a row), and the kernel gathers ψ, ε, f on those bands only.  Widths are the chunk's
largest live count rounded up to a multiple of `BAND_RUNG` = 8 (capped at L): padding slots hold dead
bands, whose weights are exact zeros, so no mask is needed, and the kernel compiles once per distinct
(L₊, L₋) pair.  A branch no residue reads is not traced; a chunk neither branch reads is skipped.  The
Hermite coefficients (`laws.hermite_branch_coefficients`) are built once per (chunk, q, branch); a knot
selects them through `index == node` (`laws.real_knot_select`).  The integration rule is unchanged; the
result differs from the dense contraction only in summation order (CPU pin
`tests/test_cd_banded_residues.py`: ≤ 1e-13 relative, with a red twin).  Fe 64 k × 36 bands, 276 chunks
of 6 knots: 6 width pairs, 35 % of the dense rows contracted (live 11 % / 40 % on the x ≥ 0 / x < 0 branch).
The receipt's `residue_bands` records the widths and fractions of the run.

## Streaming and memory

W(z) is never stored.  Nodes and knots stream in chunks of `runtime.tiles.tile_units` (TASTE 96),
sized from per-unit bytes alone:

    imaginary node: 16 n² (2 N_q,irr + N_q) / P        real knot: 16 n² (5 N_q,irr + 2 N_q) / P

(n = packed centroids, N_q padded to a multiple of P with zero W rows).  The last chunk is padded with
zero-weight nodes or unreferenced knots, so every chunk has one shape.  A chunk of at least
P / gcd(N_q,irr, P) units is rounded up to a multiple of it (its (q, z) flat axis divides P); a smaller
chunk is kept, and the flat axis takes zero rows up to a multiple of P (`qz_flat_programs`; a zero row
has χ = 0 and H = 0, so its W_c is exactly zero and is dropped before the unfold).  `knot_chunk` holds the
rule; at the ZnO anchor deck (n ≈ 9600, P = 64) one knot is 3.9 GB, so the chunk is 1, where the old
rounding forced 8.  The real axis needs no W_−q gather: the rank that holds W_q′ adds pair q′'s x ≥ 0
residues and pair (−q′)'s x < 0 residues, recomputing ρ_−q′ from its pair slab.  Each kernel returns
acc + psum(increment).

**Band tiles.**  The l sum of every kernel runs `cd_band_tile` bands at a time (`tile_sum`: a scan over
the full tiles and one tail; on the real axis, columns of each branch's band table), so ρ[k,m,l,μ] and the
contraction intermediates exist for one tile only.  The planner tile is `tile_units` of
`band_bytes_per_unit` = 16 k_w [n (3 m + s) + 8 m² + 12 N_ω] (ZnO: 181 MB per band, tile 5 of 2503).  A
tile at least the band count (at least the widest table on the real axis) is the untiled program bit for
bit; smaller tiles change only the summation order (CPU pins `tests/test_cd_band_tiles.py`,
`tests/test_cd_consumer_tiles.py`: ≤ 1e-14 relative).

**ψ and the Coulomb roots.**  ψ enters as per-rank pair slabs, q-sharded like W (`pair_psi_slabs`):
ψ_{k−q,l} at the file-wedge rows of the rank's own pairs over every band of the l sum, and of the pairs
−q over the bands the x < 0 tables read (`residue_band_tables` reports that extent; the occupied bands
for an insulator).  They are built once, one band block at a time (`carrier_band_blocks`: the carrier
placed on the `axis` layout, a block gathered and unfolded, each rank writing its own rows), so no ψ
array wider than one block is replicated.  The Coulomb roots H stay on the face; `roots_qz` lays them on
the (q, z) flat rows as χ is laid.

**Per-rank plan.**  `cd_memory_plan` gives the resident objects (pair slabs, ψ_M, the three
accumulators N_ω·N_k,red·N_Σ²·16 B, the roots on the face, one band tile) and the peak of each phase
(slab build, Σ_x, imaginary chunk, real chunk); the run log and the receipt (`memory_plan_gb`) carry it.
The accumulators remain replicated (3 × 44 MB at the ZnO deck).

## Own-energy rows, checkpoints and the stop-after instrument

`cd_own_energy_band_min` restricts the own-energy diagonal (and with it the `auto` reach) to window
bands at or above it; the kernels slice the operators to the kept block before `laws.own_apply`, so the
kept rows are the full-window values at the same knots (`tests/test_cd_resume_window.py`).  The two chunk
loops are `stream_cd_chunks`.  Under `cd_checkpoint = write | resume`, `CdCheckpoint` writes the six
replicated accumulators after every chunk through `file_io.sigma_checkpoint` (state blob, schema
`CD_SCHEMA`); a resume skips the completed chunks and re-runs the chi0 calls from `real_replay_start`
to rebuild the FD-9 ghost tail, which reproduces the uninterrupted accumulators bitwise on CPU.
`LORRAX_CD_STOP_AFTER` (docs/reference/env_vars.md) ends a timing probe after a stated number of
chunks with `cd_stop_receipt.json`.

## Refusals (`GATE cd_*`, parse time unless noted)

`cd_applicability` (CD needs `compute_mode = mpa` and `screening_diagrams = w_rpa`; pole-model keys
refuse), `cd_head_unimplemented` (`head_correction = off` only), `cd_linalg_local` (`linalg = local`
only), `cd_band_extrapolation_unimplemented`, `cd_single_eta` (no per-window η), `cd_knot_step`,
`cd_imaginary_nodes`, `cd_chi0_rule_tolerance`, `cd_chi0_rule_capacity`, `cd_chi0_band_windows`,
`cd_chi0_remote_margin_ev`, `cd_own_energy_band_min`, `cd_checkpoint`, `cd_stop_after` (run time), `cd_residue_reach` / `cd_residue_coverage` (run time),
`cd_chi0_rule` / `cd_chi0_rule_unfit` / `cd_chi0_rules_identity` / `cd_chi0_point_unplanned` /
`cd_chi0_band_windows` (run time,
`gw.cd.chi0_rules`), `cd_sigma_x_identity`
(run time: the CD pair densities must reproduce the static Σ_x to 1e-3 meV,
`SIGMA_X_IDENTITY_TOL_MEV`; measured 3.3e-10 meV Fe, 2.1e-11 meV Si), `cd_finite`.

## Outputs beside the deck (rank 0)

`cd_sigma_own_energy.npz` (the diagonal at the DFT energies and its anchor, imaginary and real parts),
`cd_sigma_receipt.json` (recipe, J, reach, chunks and per-unit bytes, per-stage seconds, Σ_x identity,
symmetry certificate, `residue_bands` widths, live/contracted fractions and extents, `band_tile`,
`psi_slab_bands`, `qz_flat_rows`, `memory_plan_gb`), `cd_leak_ledger.csv` (per chunk: seconds, bytes in use, pool high-water read without reset --
cumulative since the enclosing stage boundary, which `common.timing` owns -- and live arrays; a leak
shows as a rise after the first chunk).  `LORRAX_CD_STAR_CERTIFICATE=N` (instrument, off by default) computes
one non-identity member of N k stars directly and compares it with the wedge-unfolded cube.
Under `cd_checkpoint = write | resume` the run keeps `tmp/cd_checkpoint.h5`; a run that builds its χ⁰
rules in the job writes `cd_chi0_rules.pkl`; `LORRAX_CD_STOP_AFTER` writes `cd_stop_receipt.json`.

## χ⁰ rules (`gw.cd.chi0_rules`)

One rule request per CD memory window: the chunk the imaginary or real stream evaluates, FD-9 ghosts
included (`cd_windows`), so a window's Green-pair sweep is that of its own members.  An insulator's
imaginary windows (every |Re z| < lo) take the ordered `minimax.response_laplace_rule` (reference lo;
forward rows even + odd, reverse rows even − odd on the direct stream).  A point its certificate
refuses takes the bank tree when its pencil has ≤ 20,000 samples (large u), else the even-only rule
when `SymMaps.trs_allowed` (the two orientations are equal, the odd channel is exactly zero), else
`GATE cd_chi0_rule_unfit`.  Knot windows take `response_bank.response_quadrature` with the window as its
request and the CD options of `minimax.ResponseGroupTree`: `cd_chi0_rule_capacity` nodes (default 1088;
the bank's 192 unchanged) and the mass gate max(5000, 0.01 tol/ε) (κ·ε ≤ 1 % of the tolerance); the
sampled error is accepted against the tolerance as in the bank.  A non-default capacity builds on the
host pencils.

Precompute on a CPU node (the knot windows' pencils grow like (hi − lo)/η: 16,619 samples, ~44 GB of
host memory each, on the ZnO 2503-band deck):

    python -m gw.cd.chi0_rules --wfn WFN.h5 --nocc N --nb-chi NB --b4-logical NB \
        --knots J --ndev P --nq-irr NQ --nk NK --n-packed N_MU --knot-chunk C \
        --imaginary-nodes 96 --eta-ev 0.25 --knot-step-ev 0.0625 [--time-reversal] \
        --tolerance 1e-8 --out cd_chi0_rules.pkl

The run prints these arguments on its `chi0 rule geometry` line.  `--windows imag:0,real:3,real:-1`
builds a partial (census) file and `--merge part*.pkl --out FILE` joins partial builds, one CPU process
each.  Every file carries its identity (energies and occupations as k-star-invariant hashes, tolerance,
options, the snapped interval and envelope, the CD points and windows) and a census
(`FILE.census.csv`: window, provider, points, z ranges, nodes, sampled error, κ, wall).  The run loads
`cd_chi0_rules_file` and refuses any identity mismatch; without it the rules are built in the job
(rank-parallel, `_gather_group_rules`) and written to `cd_chi0_rules.pkl` beside the deck.

### Band windows (`cd_chi0_band_windows = auto`)

The whole-interval knot rules refuse on wide decks: a pole inside [lo, hi] with hi/η in the
thousands (ZnO: (hi − lo)/η = 3705) fails the sampled-error test on every pencil geometry.  With
`auto`, each knot window's pair space is split by band ranges (`chi0_rules.band_windows`; R = the
window's max |Re z|, m = `cd_chi0_remote_margin_ev`):

| window | occupied × empty bands | rule |
|---|---|---|
| crossing | [s, nocc) × [nocc, c) | `minimax.response_imaginary_rule` on its own [lo, hi]: t = i s, Gauss–Legendre, n = ⌈(hi − lo)/η · horizon/4⌉ nodes (≤ 2 × `cd_chi0_rule_capacity`), coefficient mass O(1) |
| remote valence | [s, nocc) × [c, nb) | `minimax.response_laplace_rule` on its own interval (even-only where refused on a time-reversal deck) |
| remote semicore | [0, s) × [nocc, nb) | as remote valence |

c is the first empty band whose lowest energy exceeds E_VBM + R + m; s is the count of leading
occupied bands whose top lies below E_CBM − (R + m); so every remote transition exceeds R + m.  The
windows' band weights (f kept on the occupied range, u on the empty range, exact zeros elsewhere)
partition f u, and each window's chi0 is a sum over its own pairs: their sum is the whole window's
chi0.  Every group runs on the direct stream with its window's weights, references [max occupied, min
empty] (reference lo, no gauge shift) and its kind's union band ranges (`chi0_rules.stream_rules`);
`bank_chi0` adds the windows' rows.  A masked band of a union range keeps a finite exponential
(`GATE cd_chi0_band_windows` past max Re t · excess = 300).  The band-window definitions are part of
the rule identity (route `cd-windowed`); precompute with `--band-windows auto --remote-margin-ev M`.
Windows whose points all lie below lo (the imaginary nodes) keep the whole-interval Laplace rule.
