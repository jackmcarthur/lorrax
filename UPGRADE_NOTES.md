# Upgrade notes

User-visible changes, newest first. Binding rulings behind the breaking
changes live in `docs/architecture/decisions.md`.

## 2026-10-01 — the shared-pole Σ τ window overlaps one node's W exchange with the next node's compute

On GPU, the scalar shared-pole Σ τ window now evaluates two τ nodes per loop
trip. Only that program is compiled with XLA's latency-hiding scheduler
(`gw.ppm_accumulators.WINDOW_OVERLAP`); the global flag stays off. One node's
W(τ) synthesis and all_to_all now run beside the other node's k-convolutions.
At the Ni 20³ tile on P64, 0.549 → 0.492 s per τ node. The window's compiled
temporaries grow by about two W(τ) tiles per rank (+2.5 GB at that tile); the
Σ τ stage prices this. Results move at round-off: Fe 4³ and Na 8³ SC maps 1–2
stay inside a 4e-16 control, and map 0 is bitwise. On a cold cache the window
program compiles about twice as long.

## 2026-10-01 — QP seeds are projected on each k's little group; the four-current χ bank carries no −q rows

An external SC seed (`sc_initial_qp_rotations_file`) is now averaged over each
kept k's little group at import (H ← |G_k|⁻¹ Σ_L A_L(H), with the band
representations of the little-group operations from the WFN), so a seed written
by another run or code version cannot break this run's symmetry; gwjax.out's
"SC initial Hamiltonian" line prints the largest change (the Fe 4³ bispinor
2026-09-19 seed: 7.2e-5 eV). A seed whose Σ window cuts a multiplet refuses
(`GATE little_group_band_representation`). Seeded SC runs move once (Fe 4³
bispinor ≤ 0.16 meV); runs without a seed are unchanged. The four-current bank
on inversion-symmetric magnets now also forms χ_{−q} from the parent rows by
the inversion (Lorentz blocks mixed by the inversion's action), halving it
(Fe/Ni 20³: 21.4 → ~10.7 GB per sample per rank); bispinor SC eqp moves by
≤ 32 µeV.

## 2026-10-01 — the χ bank carries no −q rows on inversion-symmetric magnets

On the ordered (time-reversal-broken) scalar route, every response sample's
bank carried the parent q rows and their −q partners (2120 rows at Fe/Ni 20³),
although only the line samples' partner solve reads the −q rows. When the
magnetic group holds a unitary inversion (Fe, Co, Ni, CrI3), the partner
χ_{−q} = U_I χ_q U_Iᴴ is now formed from the parent row by the inversion unfold
(`symmetry_maps.unfold_isdf_operator`) at that solve, so the stream and the bank
carry only the parent rows: half the bank per sample (Fe/Ni 20³ P64: 3.4 → 1.7
GB/rank). Scalar SC results on inversion-symmetric magnets move once, by up
to 0.75 meV (Fe 4³, maps 0–2). The unfolded partner differs from the streamed
one by ≤ 4e-9 relative, and the line selection amplifies it (a 1+4e-9 scaling
of the partner moves Fe 4³ by 1.77 meV); this is inside the 2 meV gate for the
ill-conditioned fit and selection. Time-reversal-symmetric decks (Na), groups
without a unitary inversion, and the four-current route are unchanged (bitwise).

## 2026-10-01 — a price over the memory budget warns, never refuses

No planner stops a run because a priced or compiled memory figure exceeds
`memory_per_device_gb` (or its tile). The shared-pole capacity ledger, the
compiled chunk check, the shared-pole sector batch and local pencil rounds,
the ζ μ-batch, V_q, pair-convolution, GN-PPM fit, Galerkin, kmeans Gram,
centroid-load, W-av and dense-H planners each raise one `RuntimeWarning:
memory over budget at <stage>: needs X GB/rank, budget Y GB/rank, over by Z GB;
continuing (an OOM is possible)` (gwjax.out lists it under WARNINGS), take
their smallest size and run; a device that truly lacks the room OOMs. The gates `shared_pole_capacity`
(budget), `compiled_chunk_capacity`, `shared_pole_round_capacity`,
`zeta-mubatch-capacity`, `zeta-mubatch-orbit-capacity`, `vq_tile_budget`,
`pairconv-capacity`, `gn_ppm_fit_capacity`, `bispinor-v-host-park`,
`pw-screening-budget` and the Γ-projection logical bound are gone. Decks that
fit are bitwise. Kernel shape limits (`GATE response_vertex_grid`) and
correctness gates still refuse.

## 2026-09-30 — shared-pole V staging: q tiles, one sync, a chunked digest

At map 0 the shared-pole W staged the bare V wedge one parent at a time, each
write synced, and then rank 0 alone SHA256-ed the whole file: 141.6 s (Fe 20³)
and 147.2 s (Ni 20³) on 64 GPUs, for a 54.6 GB `coulomb.h5`. The wedge now
streams in `runtime.tiles` q tiles into one write transaction that is synced
once, and `response_bank.resource_digest` (the one owner; it also authenticates
`v_q_bispinor.h5`) is SHA256 over the SHA256s of 256 MiB chunks that the ranks
read round robin, so each rank reads size/P. On one node (P4) with that 54.6 GB
file the rank-0 hash took 54.5 s and the chunked one 24.5 s (2.2 GB/s per node;
about 1.6 s over 16 nodes); the per-parent write and sync took 39.4 s and the
tiled write 36.7 s (one node is bandwidth-bound; at P64 the 1062 collective
syncs set the remaining ~87 s). The digest does not depend on the process
count. It differs from the old flat SHA256, so resuming a shared-pole
constructor from a directory staged before this release refuses with
`GATE response_coulomb_identity: content hash differs`; delete that map
directory and rerun. V and every result are unchanged.

## 2026-09-30 — every response sample group runs one program

When the response sample group is smaller than the sample count (memory-bound
decks: Fe/Ni 20³ on 64 40 GB GPUs take groups of 4 for 22 samples), the last
group was short (2 samples), a new carry shape, so `bank.compile.direct`
compiled a third stream program in the middle of map 0 (47.8 s on the Ni 20³
P64 run, which then died of host memory in that dispatch). A short group now
fills its empty slots with zero weights and runs the planned group's program;
it reserves the slots it allocates. Results are unchanged: with the group
forced to 5 on Fe 4³ (groups 5,5,5,5,2), eqp0 at maps 0–2 is bitwise to main
at group 5 and at the default single group, and map 0 compiles two direct
programs instead of three. Decks whose samples fit one group are unaffected.

## 2026-09-30 — the scalar χ₀ mode-11 door reads its tables as operands

The charge response stream on a raw-parent plan (the shared-pole direct
stream, the moment correlations and the retarded stream) baked its plan's
global unfold tables (row, trs, lsrc, rsrc, mph, nph, spin) into every
program as HLO constants. The tables now enter as device operands, placed
once per run and plan (`ffi.fft.make_kconv_chi_unfold(load=)`, the R189
four-current route), and the stream binds them as a trailing argument.
At the Fe 20³ table size (8000 k, 1792 centroids, one node): compile
7.08 → 0.57 s per program, generated code 115 → 0.1 MB, and each held
executable no longer adds host memory (+1.54 GB → +0.00 GB RSS per rank for
the second program). The Fe/Ni 20³ P64 runs compiled this program three times
at map 0 (34–48 s each), and the Ni run died of host memory during the
third. At the P64-local shape the stream's temporaries are 13.74 → 13.60 GB
and its dispatch is unchanged (0.439 s per Green pair). Results are bitwise
(Fe 4³ scalar and Na 8³ SC, three maps). The Σ τ mode-7 and mode-8 doors
still bake their tables.

## 2026-09-30 — one MPI per process; CPU runs no longer hang in MPI_Init

Before it loads a sealed bundle's private SLATE closure, the native loader
(`lxkit.native_provider`) now loads the machine libraries that closure needs
from the directories the FFI leg's own DT_RPATH names: on Perlmutter, LibSci
25.09 and cray-mpich 9.0.1. Before, the private `libblaspp.so.2`, opened by
path, found the site-default LibSci 26.03 through `/opt/cray/pe/lib64`, and that
LibSci links cray-mpich 9.1.0. Every process then mapped two MPIs. A CPU run,
whose JAX MPI collectives had already started 9.0.1, hung in phdf5's
`MPI_Init`. GPU runs did all FFI and HDF5 MPI on 9.1.0; they now use 9.0.1, the
MPI the legs were built against, and results are bitwise (hsuite P4, Fe 4³
scalar SC maps 0–2). The one-MPI check now also sees `libmpi_gnu.so`
(cray-mpich ≥ 9.1) and refuses two MPIs by name. Each process prints one line,
`[lorrax native] rank=<r> mpi=<path>`. No deck or environment change is needed.
Remove any `LD_LIBRARY_PATH` LibSci 25.09 workaround.

## 2026-09-30 — SUMMA panel loops accumulate in place

When its loop has three or more band panels, `distrib_la.panel_matmul` now adds
each panel after the first into the output tile in place, through the local
beta = 1 GEMM. XLA folds one `c + a @ b` into its GEMM. Of two adjacent ones it
left one as an add, which holds two more output tiles. That happened in a
3-panel loop (XLA inlines its one-trip scan) and when a narrower tail panel
follows the full panels. Those Green builds now hold two fewer output tiles. On
the Fe 20³ P36-local CC stream (P4 proxy), a 3-panel loop compiles at 61.3 GB
instead of 73.1 GB, and the 360-band loop at 90.3 GB instead of 99.5 GB.
Loops of four or more full panels with no tail were already folded and compile
the same; P36 runs six. Two-panel loops (P4) keep XLA's GEMM. Results are
bitwise: with the panel loop forced to six panels, Fe 4³ scalar and bispinor
SP-full, Na 8³ and MoS2 SC + BSE match main, beside a 4e-16 control that moves.

`distrib_la.panel_matmul_extra_tiles` is deleted, and the photon row-pass count
no longer adds two Green tiles at p_x ≥ 3. Fe 20³ P36 at M_T 900 now prices
1,1,1,1 row passes at 70 or 75 GB (was 2,1,1,1), counted at 64.33 GB. M_T 1800
prices 3,7,7,14 at 75 GB (was 4,7,7,15). The 2,1,1,1 that the counted-passes
release (R190) gave P36 was an over-count. P36's 6-panel loop never held the two
extra tiles: its CC stream compiles at 60.68 GB before and after this change.

## 2026-09-30 — four-current row passes are counted, not fitted

`photon_response_passes` used a fitted price (2·parents + 1.5·planes). It now
counts each pass's live buffers from their shapes (`greens_function_kernel.price_photon_pass`).
The terms are the ones the XLA buffer assignments show:
- the channel planes, counted twice at 2 or more passes;
- the quadrant Greens and their partners;
- the SUMMA panels;
- both Dirac halves of the operand faces;
- the operands that other family pairs keep live (`photon_held_faces`);
- the placed door tables and mode 11's run-time scratch.
The ledger door (`memory_per_device_gb`) and `GATE response_photon_passes` are
unchanged. The stage-memory table gets one row, "photon direct stream, row passes
(…)". Across 20 AOT programs at the Fe 20³ P36-local shape, the count is within
−3.1 % to +4.9 % of the compiled peak.

Fe 20³ P36 at M_T 900 and 70 or 75 GB moves from 2,2,2,2 passes (CC, CT, TC, TT)
to 1,1,1,1 with the in-place SUMMA accumulate above. The 2×2 proxy of the P36
tile compiles one pass on every pair at 64.49 + 1.07 GB scratch = 65.56 GB.
M_T 1800 has no AOT (production is M_T 900). Fe 4³ prices 1 pass before and
after, and its results are bitwise.

## 2026-09-30 — the four-current response compiles without baked symmetry tables

The bispinor four-current χ₀ (mathdx mode 11) now reads its unfold load
tables as device operands. Each door used to bake its plan's global tables
into the program as constants. Each distinct table array is placed once per
run (stage `response.door_tables`), and every q batch and SC map reads it.
No kernel or bundle change.

Fe 20³ P36-local AOT (M_T 900, 2 row passes per family pair, cold cache, 4 ranks per node):

| | before | after |
|---|---|---|
| compile per rank | 136.7 s | 5.7 s |
| host max RSS per rank | 48.1 GB | 3.1 GB |
| node host peak | 175 of 251 GiB | 42 GiB |
| device args / temp / code | 24.63 / 40.03 GB / 644 MB | 25.44 / 39.80 GB / 0.45 MB |

What moves: nothing. Fe 4³ bispinor SP-full SC is bitwise, and its warm
per-map W response wall is unchanged. Two concurrent compiles at M_T 1800 no
longer run the host out of memory on table copies.

## 2026-09-30 — streamed loops take a fixed 1 GiB tile; `device_room_bytes` is gone

Every planner that streams over k, q, bands, centroids, samples or rows now
takes the most units whose per-rank bytes fit one fixed tile,
`runtime.tiles.TILE_BYTES` (1 GiB). The tile comes from the loop's shapes
alone. It never reads free device memory or `memory_per_device_gb`, so every
rank computes it without a collective, and no result depends on the budget.
`common.gpu_utils.device_room_bytes` (the allocator read gathered over
processes) is deleted.

Two sizes still follow `memory_per_device_gb`, through a ledger: the
shared-pole response sample group (more samples per group is more than 10 %
faster per map, and the group moves no number since the all-sample response
rule) and the Galerkin whole-state fit in htransform.

What moves: nothing on decks whose loops already fit one tile (Fe 4³ scalar
and bispinor SC, Na 8³ SC, MoS2 SC + BSE and the hsuite are bitwise). On
larger decks a loop that used to take more than 1 GiB per rank now runs in
more passes of at most 1 GiB. Most of these loops are independent per unit,
so their numbers are unchanged. Three split a sum: the exciton_bands C_q q
chunk, the V_q G panel and the Σ output spin block. A deck whose tile shrinks
there moves at round-off. Per-map wall can move either way.

## 2026-09-30 — a held SC map refits the response rule warm when line sites move

When the shared-pole line sites move at a held SC map, the χ response rule is
refitted. The refit now tries the held rule's complex times first
(`minimax.response_group_rules` looks up the previous rule by its member
set; before, it looked them up by member order, which changes when sites
move, so every refit started cold). If the held times pass the same sampled
tolerance at the new samples, they are kept. Fe 4³ charge SC: rule build
3.4 → 0.7 s at maps 1 and 3, the W stage 7.2 → 4.6 s, 79 nodes as before.

What moves: SC runs whose line sites move. From the first warm refit on, they
move once, within the rule tolerance (Fe 4³ ≤ 0.002 meV within E_F ± 10 eV).
Runs whose sample order did not change, which includes Na 8³, are bitwise.

## 2026-09-30 — the shared-pole response rule no longer depends on the memory budget

The response bank fits its complex-time rule on all samples at once
(`response_bank.response_quadrature`). The response group that
`memory_per_device_gb` picks now only batches the evaluation: each group streams
the whole rule for its own samples. Before, each group got its own rule, so eqp
depended on the budget. Smaller groups were also less accurate: against a rule
tightened 100×, the error reached 14.6 meV on Na 8³ at group 1 and 7.6 meV on
Fe 4³ at group 8.

What moves: runs whose ledger picked a response group smaller than the sample
count (gwjax.out: "Response quadrature: N samples in M shared-node groups" with
M > 1). They move once, to the default's accuracy. Runs with every sample in one
group (all default-budget symmetric decks) are bitwise. A smaller group now
costs about 1.5–2× more Green pairs than before at the same group size; its
memory is unchanged.

## 2026-09-30 — the restart W0 is formed only on the q parents

GW no longer forms a full-q static W0 when it writes `W0_qmunu` for BSE. The
shared-pole evaluators (`shared_pole_static_wc`, and `sector_static_wc` for
`bispinor_gw = full_shared_pole`) compute V + W_c(0) at V's q parents only,
and the writer stores the producer's parents with their unfold tables. The file layout is
unchanged: a deck whose q axis reduces already stored W0 on the q parents, and
one whose q axis does not reduce stores every q. Both layouts still read; BSE
unfolds the parents on load in bounded q tiles, into the full-q layout its
kernels use. Nothing to regenerate; results are bitwise.

## 2026-09-30 — the Hall current takes `vnl_velocity_sign`; regenerate −1 Hall artifacts

`get_dipole_mtxels --static-gauge-hall-only` now builds the Hall current with
the same V_NL sign it stamps (`prov_vnl_velocity_sign`), as `dipole.h5` does.
Before, the current always used the +1 arm, so a `--vnl-velocity-sign -1`
artifact carried the +1 σ_H under a −1 label. Artifacts built at +1 (the
default) are bitwise. Regenerate any Hall artifact built at −1; its σ_H and
operator fingerprint change.

## 2026-09-30 — `parallel_transport`: the Σ term is served whenever the links are usable

The `parallel_transport` head no longer sets its Σ term D_kΔH to zero on a map
whose link bound exceeds 1 %. Complete links serve it on every map. The link
error is a k-convergence measure: the dipole step warns above
`--parallel-transport-validation-rtol` and still writes the artifact, each SC
map logs the error and its bound on the Σ term, and neither gates anything.
Only links that are not usable (incomplete, or a stencil or
window-hybridization gate fails) zero the term, on every map.

SC decks whose links were above the bound move once: MoS2 3×3 SOC (link error
9.2 %) converges to a 4.48 eV gap with the term served (5.17 eV with it
zeroed). Decks that stayed below 1 % are bitwise (Fe 4³ scalar)
([self-consistency §7](docs/self_consistency.md#metals-direct-drude-head)).

## 2026-09-30 — parallel-transport links: schema 4 on the link shell; rerun the dipole step

`parallel_transport` now differentiates on the point-group-closed
Marzari–Vanderbilt link shell (`common.parallel_transport.link_stencil`).
A link artifact written before this (schema 3, three reduced axes) refuses
under `parallel_transport`; rerun the dipole step (`psp.get_dipole_mtxels`).
`dft_velocity` still reads the old file. The bcc link stage takes about 2×
longer. Orthogonal lattices keep the three axes. Fe 4³ bispinor eqp moves
≤ 1.6 meV, Si SOC ≤ 16 µeV.

`bispinor_gw = full_shared_pole` SC decks that do not name `sc_head_update`
now run `parallel_transport` when a link artifact exists, as scalar decks
do, and move once (Fe 4³ eqp0 within E_F ± 10 eV ≤ 9.7 meV). On a metal,
`bare_transverse` refuses `parallel_transport`
([self-consistency §7](docs/self_consistency.md#metals-direct-drude-head)).

## 2026-09-30 — regenerate `WFN_qp.h5` from WFNs that store both k and −k

`WFN_qp.h5` now keeps time reversal on a WFN that stores two k of one orbit
(e.g. MoS2 3×3). A file written before this from such a WFN has broken rows,
and the BSE and GW runs that read it used them. Regenerate it with
`python -m postprocess.rotate_wfn_to_qp WFN.h5 qp_wfn_rotations.h5`; WFNs
without such rows (Si) give the same file
([self-consistency §8](docs/self_consistency.md#8-seeding-restart-and-outputs)).
An SC run that writes `WFN_qp.h5` now binds `dipole_qsgw.h5` to it, so a GW
run on `WFN_qp.h5` can take that file as its `dipole.h5`.

## 2026-09-30 — SC W line sites held within max(3 meV, 0.1 × max|dE|)

Held shared-pole W line sites are re-placed only when they would move by more
than max(3 meV, a tenth of the previous map's max|dE|). SC runs that re-plan
their sites move once; Fe 4³ scalar `parallel_transport` SC now converges
(28 maps; it stalled at map 16)
([self-consistency §6](docs/self_consistency.md#shared-pole-w-with-retained-quadrature)).

## 2026-09-29 — `qp_solver = fixed_point` and `eqp_root.dat` are retired

`qp_solver = fixed_point` refuses by name; set `one_shot_dft` (Σ at E_DFT)
or `self_consistent` (Σ at each map's own energies). No QP equation
E = h₀ + ReΣ(E) is solved on any route. The dynamic one-shot run no longer
writes `eqp_root.dat`, and `sigma_diag.dat` drops its `QP_status` column
(`Z` stays). `eqp0.dat` and `eqp1.dat` are unchanged.

## 2026-09-29 — the semicore class on the bispinor (sector) route

Every dynamic self-consistent bispinor deck with a coarse class
(`bispinor_gw = full_shared_pole` or any bispinor `compute_mode = mpa`)
moves once. The coarse (semicore) class was already built once from the DFT
ladder, above the Σ route; the sector Σ now reads it as the scalar Σ does:
the coarse states are read on held windows at η_semi = 5 eV, certified at
max(`sigma_quadrature_eps`, 3e-3), instead of at Σ(ω = 0), and
`sc_semicore = dft` (the default) pins their DFT block. `sc_semicore` and
`sigma_omega_patches_ev` `lo:hi:eta` triples behave the same on both routes.
`sc_semicore = dft` named on a run without a coarse class no longer refuses
(`GATE sc_semicore` is gone); it logs that there is nothing to pin.
Scalar decks are bitwise. A `full_shared_pole` SC run's head is its sector
model, so a converged run no longer refuses at the end
(`GATE sc_final_map_requires_iteration_head`), and a one-map
`dft_velocity` run is admitted (`GATE full_shared_pole_dft_velocity_one_map`
is gone).

## 2026-09-29 — bulk bispinor V carries the mini-BZ head average; bispinor refusals move to setup

Every bulk (`sys_dim = 3`) bispinor deck with `mc_average_vcoul_body = true`
(the default) moves once: the CC and TT tiles now take the scalar V's mini-BZ
average at the q ≠ 0 head slot (`v_q_g_flat.v_head_fn_in_V`, one owner); a TT
slot takes ⟨v⟩ P^T(K̂). Slab decks are unchanged. `full_shared_pole` with
`head_correction` unset resolves to `no_local_fields` (logged); an explicit
`full` still refuses. `w_bse` and `hl_ppm` on a WFN without
measured time reversal refuse before the basis, not after ζ and V (HL-PPM used
to keep one residue silently). Headless shared-pole SC warns on bispinor FD
metals too.

## 2026-09-29 — `sc_semicore = dft`: semicore pinned at its DFT block, mixing kept

New SC key, default `dft` (owner 2026-09-29: "sure we can keep DFT the default"); `qp` is the previous behaviour. Every dynamic SC deck with a coarse class moves once.
Under `dft` the coarse (semicore) class keeps its DFT block of H in the DFT
basis and its end of every protected–semicore element reads Σ at E_DFT on the
held coarse windows ([self-consistency §2](docs/self_consistency.md#2-band-treatment)).
Fe 4³ and MoS2 3×3 prot at η_semi 5 and 8 eV: same maps to converge (14, 8),
equal or fewer τ pairs (MoS2 333 → 321), semicore QP within 27 meV of DFT
(qp: 0.1–6 eV deeper), and the protected states' η_semi 8 − 5 spread falls
3–6× (E_F ± 10 eV std Fe 5.1 → 1.4, MoS2 5.0 → 0.8 meV). The default is a
no-op on a run without a coarse class (static modes, an `nval`
that covers every occupied band). Sandbox claim 2964.

## 2026-09-29 — coarse (semicore) windows certified at max(ε, 3e-3)

Self-consistent decks with a coarse class move once. The coarse windows are
certified at max(`sigma_quadrature_eps`, 3e-3) (`qp_support.SEMICORE_EPS`,
owner 2026-09-29), so at the default ε 1e-4 they take 3e-3; every other Σ
window keeps `sigma_quadrature_eps`. At
η_semi 1 eV against ε 1e-4: map-2 τ pairs MoS2 3×3 530 → 465, Fe 4³ charge
1087 → 982; states within E_F ± 10 eV move ≤ 0.06 meV at maps 0–1 (≤ 1.9 meV
at map 2 of the unconverged Fe run); semicore QP ≤ 4.3 meV at map 0. The
planner's node law for grouping coarse windows is now evaluated on the box
each run is built on, so it equals the certified count (claim 2960).

## 2026-09-29 — the production QSGW partition: counted b3, semicore Σ read class

Every dynamic self-consistent deck with a coarse class (`qp_solver =
self_consistent`, scalar MPA/shared-pole route) moves once. See
[self-consistency §2](docs/self_consistency.md#2-band-treatment).

- **b3 counts bands, as before.** b3 = nelec + `ncond` (owner 2026-09-29: "b3
  will count bands as on main yes, and only bands between b0 and b3 will be
  rotated amongst each other"). The QP matrix [b0, b3) rotates among itself;
  [b3, number_bands) is the scissored tail (DFT ψ, rigid shift, no Σ, no
  mixing). The ζ fit is unchanged. The classes below change only where
  Σ_c(ω) is read.
- **One request key, `number_bands_protected`** (the documented form): every
  occupied band plus conduction bands up to that total. Its semicore (coarse)
  class is every occupied band below a ≥ 4 eV band gap. The `nval` / `ncond`
  form stays: there the coarse class is every occupied state below the
  lowest requested valence band (a smaller `nval` moves more valence states
  onto the coarse windows). Giving both forms refuses
  (`GATE band_request_forms`). A dipole artifact is stamped with the request
  window, so a deck switched to `number_bands_protected` needs a dipole
  written with `nval` = the occupied count.
- **Semicore moves to coarse windows.** On the scalar MPA/shared-pole route the
  coarse states are read at their own energy on held windows at η_semi = 5 eV
  (one per coarse manifold; the Σ plan groups them to the least closed-form
  node count) instead of at Σ(ω = 0) (below E_F − 15 eV) or on the near grid
  at the deck η, certified at max(`sigma_quadrature_eps`, 3e-3).
  `sigma_omega_patches_ev` accepts `lo:hi:eta` triples as user coarse windows
  (`GATE sigma_coarse_window`).
- **Continuous tail weights.** The scissored tail's rigid shift Δ_c weights
  each conduction state by min(Z, 1/Z) (0 for Z ≤ 0) instead of Z inside a
  hard Z ∈ (0, 1] cut, so the tail law has no jump where a state's Z crosses 1.
- **A new refusal.** `zeta_nband` below b3 now refuses on every run,
  one-shot included (`GATE qp_matrix_zeta_left`; it was a warning).

## 2026-09-29 — the planners check their compiled executables

- Each chosen chunk's executable is checked against its planner's price
  before it runs (`runtime.aot_memory.check_chunk`; the response direct
  stream's sample group, the ζ μ batch, and the Σ τ window's price). When a
  chunk is over its room, it is recompiled once at a corrected size. If it is
  still over, the run warns and continues (since 2026-10-01; it refused before).
- The shared-pole response group must also fit the device room, which counts
  resident bytes the capacity ledger does not own. A deck whose group was
  sized into that gap gets a smaller group and moves once. Fe 8³ charge P4 at
  36 GB keeps its group of 16 and is bitwise. With the latency-hiding flag it
  runs 15 and peaks at 34.55 GB instead of 36.04 GB.
- Leave at least 5 GB between `memory_per_device_gb` and the card for NCCL,
  the CUDA context and cuSOLVERMp. See
  `docs/architecture/memory-model.md` for what compiled statistics miss.

## 2026-09-29 — shared-pole χ₀ direct stream through mathdx mode 11

- The shared-pole bank's direct stream (charge, metal or insulator, on a
  raw-parent plan) forms each node's correlation with mathdx mode 11 from the
  parent Greens; no full-k Green is built. Decks whose response groups keep
  their size move at round-off (Fe 4³ charge SC ≤ 0.2 µeV).
- The freed memory lets the response group grow, and the group size still
  follows `memory_per_device_gb`. A deck whose group grows gets a different
  shared-node rule, with the same certified accuracy, and moves once: Fe 8³
  charge SC groups go from 2 to 16 samples, Green pairs per map from 344 to
  100, the held map from 335 to 276 s, and eqp0 within E_F ± 10 eV by at most
  1.28 meV (median ≤ 13 µeV).

## 2026-09-28 — band extrapolation on the shared-pole Σ

- Scalar `compute_mode = mpa` (shared pole or MPA fit) now extrapolates the Σ_c
  band sum with the same brackets and pooled (β, Ω) fit as GN/HL-PPM.
  `use_band_extrapolation` defaults on, so every scalar shared-pole run moves
  once, and a deck with `number_bands_sigma` < 2·n_occ now refuses at startup,
  as GN/HL-PPM already does: for example Fe 4³ scalar at 35 bands (needs 36)
  and MoS2 at 44 bands (needs 52). Raise the band count or set
  `use_band_extrapolation = false` there. Bispinor `mpa` is unchanged.

## 2026-09-28 — band extrapolation: pooled denominator shell, cuts at 70/85/100 %

- `spectral_shell` now fits one (β, Ω) over the QP window's states: band A adds
  a_i·Σ_k w_k (E_Ak − E_i + Ω)^−β to state i, with a per-state amplitude from
  the widest shell. The per-state exponent is gone. Every GN/HL-PPM run with
  `use_band_extrapolation` on moves once. On Si 4³ at 78 bands against the
  complete basis the std over the ±10 eV states drops from 109 meV (per-state,
  cuts 64/72/78) to 9.2 meV (pooled, cuts 50/64/78); see
  [Band extrapolation](docs/theory/band-extrapolation.md).
- `total_fractions` cuts are 70 % and 85 % of `number_bands_sigma` (were 80 % and
  90 %), so the three bracket counts change and the Σ τ-loop recompiles once.
- `sigma_mnk.h5`: `sigma_c_extrap_beta_kn` holds the pooled β (NaN on states
  without a tail); new attributes `pooled_beta`, `pooled_omega_ev`,
  `pooled_residual_rms_ev`, `pooled_state_count`.

## 2026-09-24 — k-axis convolutions on nvidia-mathdx (branch, not yet main)

- **NVIDIA GPUs now require the `nvidia-mathdx` wheel** (pinned in the
  `cuda12`/`cuda13` extras; header-only).  Without it a CUDA run refuses at
  startup with `GATE mathdx-headers` and the fix `pip install nvidia-mathdx`.
- `LORRAX_FFT_FFI_FUSED`, `LORRAX_CONV_KMINOR_FFI` and `LORRAX_CONV_KLEAD_FFI`
  are gone: Σ, COHSEX and the BSE ladder/stack convolutions have one route
  per platform (nvidia-mathdx on CUDA, the host FFTW plans on cpu).  A leftover
  setting is ignored.
- The FFI handler ABI is 4; a `liblorrax_ffi.so` built before this change is
  refused by name.
- The kernels compile on first use (about 6 s per k-grid) and are kept in
  `$SCRATCH/.cache/lorrax/kconv_mathdx` (else `~/.cache/lorrax/kconv_mathdx`);
  the second run of a deck loads them in ~10 ms.
- The flat-k transform (χ0, head, htransform) also runs on nvidia-mathdx on
  CUDA; `LORRAX_FFT_FFI` now governs the cpu leg only.

## 2026-08-28 — startup ownership, BSE mesh flags, emulated CPU meshes

- The runtime owns `JAX_ENABLE_X64`: it applies the resolved value even when
  jax was imported before the driver, and a resolved `False` refuses at
  startup. `LORRAX_ALLOW_X64_OFF=1` continues as an announced uncertified
  run. The per-driver `jax.config.update` lines are gone.
- Drivers no longer arm the persistent compile cache; step 7 of
  `runtime.initialize_communicator_stack` owns it. With `ISDF_JAX_CACHE_DIR`
  unset it is on, in one namespace per source release under
  `$SCRATCH/.cache/lorrax/jax_compile`, with JAX's write threshold at 0;
  `ISDF_JAX_CACHE_DIR=""` turns it off.
- BSE-family drivers: omitted `--px/--py` now means the run's canonical
  square mesh (it used to mean 1×1). An explicit shape must consume the
  job's device count exactly — under- and over-requests both refuse.
- `gw_jax`, `kin_ion_io`, `downfold_cli` and `kmeans_cli` answer `--help`
  and bad argv before any runtime exists (`runtime.cli_seam`); the other
  four drivers still pay full bring-up first.
- Single-process multi-device CPU meshes
  (`XLA_FLAGS=--xla_force_host_platform_device_count=N`) now run end to
  end: `SlabIO` serves them through an announced serial tier
  (`file_io._slab_io_serial`, CPU only). The `p*q == process_count`
  refusals stand everywhere else.
- `qp_solver = self_consistent` beside a dynamic `compute_mode`
  (`gn_ppm`/`hl_ppm`/`mpa`) refuses at driver entry; pair it with `cohsex`.

## 2026-08-18 — retired HDF5 controls now refuse or are absent

- GW decks containing `slab_io` or `use_ffi_io` now refuse with a targeted
  removal message. Remove either line; SlabIO has one collective transport.
- kmeans no longer accepts `--use-phdf5`; `WfnLoader` selects its one valid
  scalable read path from runtime capability.
- SlabIO no longer accepts `chunks=`. The argument was ignored and every
  collective dataset was already contiguous. The sigma and zeta writers no
  longer request a layout the native create cannot produce.
- `LORRAX_PHDF5_CLOSE_VERBOSE` defaults to compact logging: empty/fast closes
  are quiet, while queued or slow I/O still emits one summary. Set it to `1`
  for the former per-phase diagnostics or `0` for silence.

## Changes through 2026-08-01

The remaining entries describe the earlier origin/main-to-HEAD upgrade.

## What breaks (refusals and hard errors)

**The FFI layer is REQUIRED** (`decisions.md` 2026-08-01). Where origin/main
ran everything through native XLA, GW and htransform now route their flat-k
FFTs and the large band contraction through vendor FFI handlers, and a
missing or unloadable FFI library is a **startup refusal**
(`ffi.gate.Gate.enforce`, wired into `runtime.initialize_communicator_stack`
step 6b), naming the `.so`, the env var, and `docs/environment/overview.md`
— never a silent demotion. Practical consequence: you must build the FFI
library before running — `src/ffi/cpp/build_host.sh` (generic host) or
`config/frontera/build_ffi_host.sh` (Frontera MKL/ScaLAPACK), pointed at by
`LORRAX_FFI_HOST_SO`; the CUDA library from the complete CUDA 13 stack
(`docs/building_ffi.md`), pointed at by `LORRAX_FFI_SO`. Per-knob semantics:

- `LORRAX_FFT_FFI=0` **refuses**: the XLA flat-k twin inside
  `make_flat_k_fft` was deleted, there is nothing to opt out to (recover the
  arm from git history for a debugging build). Handlers are c128-only.
- `LORRAX_FFT_FFI_FUSED=0` was a real, announced opt-out onto the decomposed
  three-transform chain (deleted 2026-09-24, see above).
- `LORRAX_BANDS_GEMM_FFI=0` is an announced **UNCERTIFIED** opt-out onto the
  retained XLA einsum arm (retained because `extra="minor"` structurally
  cannot ride a batched GEMM and quietly keeps the XLA plan under every mode).
- CUDA differences: the FFT dial resolves to the cuFFT strided handlers
  (same target names, both flat-k and gw_conv); the GEMM dial does not exist
  on CUDA (host symbol table only — XLA:GPU's cuBLAS dot lowering IS the
  required path there, and the startup report says so); an absent CUDA
  library refuses identically (verified rtx job 7885151). BSE is out of
  scope on both platforms: its FFTs are `local_*fftn3` = `jnp.fft` aliases
  with no FFI route, kept by the ruling.

**Square process meshes only.** `resolve_mesh` refuses a device count that
is not a perfect square, naming s² and (s+1)² to request;
`create_mesh_2d` / `create_mesh_xy` / `RuntimeStack.reshape` refuse
px ≠ py; the rectangular-mesh accommodation was deleted. Note the ruling's
letter in `decisions.md` prescribes idle-rank truncation; the implementation
deliberately refuses instead, because idle ranks deadlock under the
`impl=mpi` transport (communicator creation is collective over
MPI_COMM_WORLD — full argument in the `resolve_mesh` docstring; sandbox
CLAIMS row 33 records the deviation, owner may re-open). Launch square
counts: 4, 16, 64, ...

**`sigma_omega_accumulation = kij_stream`** raises ValueError: the
single-process streamed-h5 accumulator was removed 2026-07-31. Use `kij` or
`auto`; for cubes that do not fit, `sigma_omega_layout = sharded` (below).

**`w_dyson_solver = lstsq`** raises (two-plan W cleanup): the SVD min-norm
inner solve masked a rank-deficient A = 1 − V·χ0 — reduce n_mu or raise
`zeta_rcond` instead. `lu` deprecation-warns and resolves to `local`.

**`use_low_mem_eigh = true` with `eigh_backend = off`** refuses at parse
time (a contradiction; see the new-keys section).

**`strict_keys = true`** (new, default false) upgrades the unknown-deck-key
warning to a ValueError naming every unknown key — set it in CI decks.

## What warns (deprecations and behavior you should notice)

**Unknown deck keys now warn.** Any key not in `gw_config._DEFAULTS` and not
covered by a legacy branch is reported in ONE aggregated rank-0 warning
(key + line number) and ignored. On origin/main such keys were dropped
silently. Consequence for removed keys:

- `cusolvermp_charge` / `cusolvermp_lu` (deprecated aliases on origin/main)
  were **removed** — they now warn-and-ignore, i.e. they stop steering
  anything. Use `distributed_cholesky` / `distributed_lu`.
- `isdf_memory_mode` (auto | high_mem | low_mem) was removed with the W
  cleanup — warn-and-ignore. The W Dyson solve is selected by
  `w_dyson_solver = local | distributed`.

**Env-twin deprecations**: `LORRAX_ZETA_RCOND` and the
`LORRAX_SC_*` family still win over the deck keys when non-empty, printing a
rank-0 deprecation notice; ζ-fit provenance records the EFFECTIVE
(post-override) values so dropping the env cannot silently reuse a ζ at a
different conditioning cutoff.

**FFT-FFI knob renames** (P1 wave, 2026-07-31): the C++ knobs are spelled
`LORRAX_FFT_FFI_THREADS`, `LORRAX_FFT_FFI_CHUNK`, `LORRAX_FFT_FFI_LOG` (one
spelling on both platforms; `_LOG` is rank-0-scoped, `=all` for every rank).
The old spellings `LORRAX_MKLFFT_THREADS` / `LORRAX_MKLFFT_CHUNK` /
`LORRAX_MKLFFT_LOG` / `LORRAX_CUFFT_LOG` are honored as deprecated aliases
with a one-time announcement; the new spelling wins when both are set.

**Env grammar hardening**: unrecognized values of the C++/py knobs announce
loudly and resolve to the default (grammar errors must not kill a run, and a
typo must not silently pick a known-bad policy — e.g.
`LORRAX_SCALAPACK_MKL_THREADS` garbage used to fall through `atoi()` to the
24×-slower configuration). Off-dials may refuse; typos never do.

## What changed silently-but-safely

**New deck keys** (full list: `docs/input_reference.md`, generated from
`_DEFAULTS`; load-bearing discussion in `docs/drivers.md`):

- `hartree_source = auto | stored | isdf | gspace` — the G-space vs ISDF
  V_H switch; auto resolves stored → folded → isdf.
- `distributed_zeta_solve = auto | replicated | per_q | distributed` — ζ
  back-solve tier; auto = replicated under the 4 GiB gather cap, else
  per_q; `distributed` (ScaLAPACK pzheevd factor + 2-D-sharded back-solve,
  nothing O(μ²) replicated) is a different, equally valid GAUGE (~κ·ε).
  ζ-fit provenance now records the gauge tier ('replicated' |
  'distributed'; per_q collapses to replicated — same factor bits). The
  schema was NOT bumped: a legacy `tmp/zeta_q.h5` whose stamp lacks the
  tier key is treated as a replicated-gauge fit — replicated-tier reruns
  reuse it with a one-line notice; a distributed-tier rerun refits, with
  the mismatch named. No forced refit of existing ζ files.
- `w_dyson_solver = local | distributed` — the exactly-two W Dyson plans
  (`auto` is a permanent alias of `local`); `distributed` refuses loudly
  when unavailable, never downgrades.
- `sigma_omega_layout = replicated | sharded` — Σ_c(ω,k,m,n) cube stays
  mesh-tiled end-to-end under `sharded`, for every `qp_solver`; refuses an
  indivisible window or `h5py_allgather` at P>1.  The `self_consistent`
  refusal shipped with this key was REMOVED 2026-08-05: the SC loop never
  rotates the cube, so the "rotation seam" it named does not exist, and the
  two layouts measure bit-identical under SC (jobs 7889782/7889789).
- `eigh_backend = auto | off | distributed | cusolvermp | slate |
  scalapack` — BSE/htransform distributed-eigh sites; `use_low_mem_eigh =
  true` + `auto` resolves to `distributed`.
- `strict_keys` — see above.

**Startup entry point.** All seven chain drivers (kmeans_cli,
get_dipole_mtxels, kin_ion_io, gw_jax, htransform, bse_jax, exciton_bands)
now start through ONE module-top call, `runtime.initialize_communicator_stack()`
— failfast hook, env defaults, jax.distributed, backend init, canonical
square mesh + communicator-clique warm-up (`warm_mesh_cliques`, required
under `impl=mpi`), FFI gate enforcement, and one rank-0 startup report
stating every resolved dial and demotion. Drivers no longer call
`prepare_mesh`/`bootstrap` themselves.

**Process teardown.** `gw.gw_jax`'s `__main__` ends via
`runtime.finalize_process(rc)`: ordered explicit teardown (effects barrier,
unregister jax's `clean_up` atexit, `jax.distributed.shutdown()`, run the
remaining atexit hooks, announced `os._exit`). This cures a deterministic
interpreter-teardown deadlock after fully-cold in-process compile storms
(XLA:CPU client destructor pool shutdown; jobs 7884928/7884989). If you
wrapped gw_jax in your own post-main `os._exit` harness, drop it. Note the
process does not run interpreter finalization after `main()` — atexit
duties are executed explicitly, nothing is silently skipped.

**`slab_io = auto` demotes instead of aborting on bare launches**: it now
probes MPI bootstrapability (launcher PMI env, else a throwaway-subprocess
singleton-init probe) before selecting either MPI tier, and demotes to
`h5py_allgather` with a full announcement when MPI cannot bootstrap — a
bare `python -m gw.gw_jax` no longer dies in MPI_Init_thread.

**Transport.** Production CPU collectives are
`JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` (MPItrampoline → patched
MPIwrapper → Intel MPI/mlx; recipe `docs/dev/mpi_collectives.md`,
env block `config/frontera/mpi_transport_env.sh`). gloo is banned at
distributed tiers: reproducible ReduceScatter timeouts at P=64 and ~5%
silent reduce-scatter corruption (sandbox CLAIMS rows 3-4). The startup
report warns when a multi-process CPU run lands on gloo.
`LORRAX_MPI_FINALIZE_FIX=skip_atexit` (overlay sitecustomize) is mandatory
for impl=mpi runs.

**Two behaviour changes from the distrib_la replumb, ACCEPTED as correct
rather than fixed** (2026-08-07; adjudication item 7 — recorded here because
neither is a bug and both would otherwise read as one to the next person who
finds them).

1. **`solve_zeta`'s `mu_pad` divisibility net is now UNREACHABLE for
   ScaLAPACK factors, and that is the fix, not a regression.** The net
   (`isdf/core.solve_zeta`) demoted `scalapack_lu`/`cusolvermp_lu` to the
   per-q `jnp.linalg.solve` when `n_rmu_logical` did not divide both mesh
   axes. A ScaLAPACK factor now arrives as a `distrib_la.FactorToken` and
   the token branch returns before the net, because `distrib_la.factor`
   REFUSES a non-dividing extent at FACTOR time — earlier, with the failed
   guard named, and before any collective. The supersession is strictly an
   improvement: the old solve-time demote kept the ScaLAPACK factor's own
   `ipiv` and handed it to `lax.linalg.lu_solve`, whose pivot convention is
   not ScaLAPACK's, so the "safe fallback" computed a wrong answer
   successfully. The net stays in place for the array-factor routes it is
   still correct for; its `print` (not `warnings.warn`) is deliberate —
   warning dedupe is what made the original demotion invisible in
   production logs.

2. **`use_low_mem_eigh` now threads into `compute_wfns_fi` on the two
   raw-params drivers** (`bandstructure.htransform`, `bse.exciton_bands`).
   Both used to spell the CLI-over-deck precedence inline and never call
   `gw_config.resolve_eigh_backend`, so the key parsed, defaulted, validated
   and was read by nobody on those two paths. It is live now: with
   `use_low_mem_eigh = true` and `eigh_backend = auto`, htransform's Gram-eigh
   line changes from the native description to the distributed one. Intended
   — and the consequence is that a machine which cannot serve the
   distributed eigh now REFUSES those runs where it used to run native
   silently. That refusal is armed on purpose; it is the whole point of the
   key.

**Where the docs live now**: `docs/drivers.md` (the seven drivers: flags,
outputs, failure modes), `docs/input_reference.md` (every deck key —
regenerate with `tools/gen_input_reference.py`), `docs/environment/`
(overview, transports, per-machine pages incl. `machines/frontera.md`),
`docs/dev/large_nmu_operation.md` (two-plans-per-family map, keys,
thresholds), `docs/dev/env_vars.md` (the env registry — gated by
`tests/test_env_registry.py`), `docs/architecture/decisions.md` (binding
rulings). Which page owns which fact is stated once, in the register at the
top of `docs/index.md`; certification scope lives in the sandbox `CLAIMS.md`
ledger rather than in a doc page, because a page recording it goes stale
silently.
