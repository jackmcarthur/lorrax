# Upgrade notes

User-visible changes, newest first. Each entry says what changed, which
results move, and what a user must change in decks, environment or files.
The binding rulings behind breaking changes are in
`docs/architecture/decisions.md`; older history is in git.

## 2026-10-04 — bispinor sectors build whole q-local pole models per rank when parents ≥ ranks

With at least as many irreducible q-parents as ranks, the bispinor CC/TT/CT
sector constructor now runs local rounds: each rank builds its own parent's
three models, ceil(nq/P) rounds. Before, the CT joint pencil's conservative
price (both retained spans at twice their pole budgets) sent all three sectors
to the face route. There the batch sizing admitted one parent per round, and
every parent ran on the whole mesh in turn.
- Ni 20³ bispinor SC, P64 on A100-40GB: the map-0 sector constructor takes
  643 s in 11 rounds of 64 parents, against about 10,300 s (641 face rounds
  of 16.1 s). The CT joint side is 3968–4352, against the conservative 10760.
- Each round admits the CT pencil at its actual spans; a price over the budget
  warns and runs. With fewer parents than ranks the old rule applies.
- The models match main's local route bit for bit. Decks that move from face
  to local change by the face/local difference: on Fe 4³, CT W(z) within
  9.6e-7 relative at map 0 and eqp within 0.1 µeV.
- A local round writes each model once (no 4-parent store span cap).
- Decks do not change.

## 2026-10-04 — the two-component TRS check pairs k with −k only through QE-unitary rows

The DFT-reference time-reversal check now takes the spatial operation that maps
a stored k to −k only from the rows QE types unitary (t_rev = 0). It used to
take the first matching row, including the ones QE composes with time reversal.
On a magnet where such a row maps k to −k before a unitary one does, the test
compared Θg with g and passed by construction. Monolayer CrSBr (M ∥ b; QE's
ΘC2z precedes inversion) is the case found: it reported TRS consistent at 6.6e-8
on a 6 μB/cell ferromagnet. It then ran on the even shared-pole route, with −k
and (kx, −ky) unfolded through false unitary C2z and C2x. Its q = 0 Gram refused
(GATE shared_pole_gram_valid, −1.86e-6 at 16×12).
- Now the check reads 0.29 and reports TRS broken. CrSBr takes the ordered
  route: all 63 parents construct, and the q = 0 Gram minimum is −6.1e-12.
- Any deck whose verdict flips moves from the even to the ordered route. No
  other production deck in the sandbox flips: non-magnetic and schema-less
  WFNs, CrI3 and the H2⁻ fixture keep their verdicts. Without a QE schema every
  row is still presumed unitary, and SymMaps warns as before.
- The cached verdict stamps carry the QE typing, so an old stamp is not reused.
- Decks do not change.

## 2026-10-04 — half first Anderson step; the SC criterion pairs by sorted index

The SC Anderson loop's first step is now half a plain step, x1 = x0 + f0/2.
Later steps are unchanged (β = 1). The step index is the global map index,
so a run resumed from a map-0 checkpoint takes the same half step. Along a
mode with plain-step eigenvalue λ, the map-1 residual is |1 + λ|/2 of map 0's
instead of |λ|.
- CrI3 24×24 bispinor (λ ≈ −2.5): 0.75 instead of 2.5.
- Si 4³ (λ ≈ +0.15): map 1 is worse (1117 against 343 meV), and the run
  still converges in 8 maps.
- On Si 4³, eqp0 at |E − VBM| ≤ 10 eV moves by at most 1.0 meV. The deepest
  valence pair, 12 eV below the VBM, moves 6.2 meV. That is within that
  state's dependence on the held Σ plan, which is 12.8 meV on the previous
  code.

The convergence criterion now pairs each input column with the output column
of the same sorted index. Before, input and output were matched to the map-0
labels by two separate assignments. On CrI3 24×24 charge QSGW, hybridized
states then read 1.3 eV where the sorted pair moved 6 meV, so the 1 meV
criterion fired at map 10 instead of map 9. The label now only names a pair.
The `SC map gain` line pairs adjacent inputs by sorted index.

Log lines:
- `SC identity: max |dE| = ... eV, input and output paired by sorted index`
  replaces `SC identity: max |dE| by overlap = ... (sorted-index value ...)`.
- `SC matrix residual` ends with `FLAG` when max|dE| exceeds max_k‖f_k‖₂.

Decks do not change. `sc_mixing` is still parsed and range-checked, but
nothing reads it: the in-process linear-mixing path is deleted. A checkpoint
written before this change still resumes. Its first `SC map gain` line after
the resume compares a label-paired table with a sorted one.

## 2026-10-04 — an SC run continues across processes from sc_seed/sc_checkpoint.h5

Every unconverged SC map except the budget's last now writes
`sc_seed/sc_checkpoint.h5` beside the warm seed. It holds:
- the Anderson history stacks, newest pair and residual history;
- the previous map's Z and the map-0 identity labels;
- the held plan without fitted rules: ω grid, windows, supports, K/CT/carrier
  extents, and each Σ window's boxes, including the box its rule was built on.

When a deck's `sc_initial_qp_rotations_file` names a warm seed with an
authenticated checkpoint beside it, the run continues at the next map. It
authenticates when the deck matches, apart from the `restart`, seed,
`sc_max_iter` and `sc_tol_ev` lines, and when the WFN, carry and
`sc_history_depth` match. The continuation:
- numbers maps globally (`eqp0_iterNNNN.dat`, `tmp/mpa/sc_NNNN_shared_pole`);
- rebuilds each Σ rule cold on its held box;
- plans the χ rule cold.

Before, every process took one unmixed step: on CrI3 24×24 the gap went
4.154 → 4.283 → 4.007 eV over two such steps.

Results:
- A one-map-per-process Fe 4³ bispinor SC follows the one-process run within
  3 µeV per map. A seed alone is off by 485 meV at map 2.
- The write costs 0.1 s per Fe 4³ map. On CrI3 24×24 the file is 1.8 GB.
- A converged run deletes its checkpoint. One that does not authenticate is
  left in place with a WARNING. To start fresh from the same seed, remove the
  file.

Log lines: `SC checkpoint: map N written` and `SC resume: continuing the
Anderson trajectory at map n`. `file_io.sigma_checkpoint` now publishes through
`collective_atomic_file_transaction`, with a commit over the cube bytes. A
one-shot Σ checkpoint written before this change is recomputed once.

Decks do not change. A leg chain copies the whole `sc_seed/` directory.

## 2026-10-03 — route-G stage 0 streams over parent chunks; step_up steps by the compiled figure

Stage 0 of a route-G ζ batch (X_B, the pair GEMM and the all-to-all) runs
one chunk of raw parents at a time and writes into the owner's D̃, which is
allocated once with its zero column. The all-to-all no longer holds a second
copy of D̃ (28.7 GB on CrI3 24×24 bispinor, at every P), and X_B is one
chunk's instead of growing with P. A chunk is the fewest parents whose X_B
psum and all-to-all reach the comm model's efficient payload, at most one
1 GiB tile of X_B, balanced over the parents (CrI3 24×24: 61 chunks of 1;
small decks: one chunk). The X_B band chunks (`x_chunks`) are removed.
`runtime.aot_memory.step_up` now steps to the block count the compiled figure
implies (a secant through the last two figures), rounded up to the next
count with fewer plane groups per block, instead of doubling; a batch 0.3 %
over the room at 50 blocks now runs at 63, not 100. Results: Fe 4³ bispinor
V_q bitwise on the 0.25 GB deck and within rel 1.4e-10 on the roomy deck
(claim 3177). CrI3 24×24 bispinor at P16 now fits the 56.16 GB target:
the charge batch compiles to 38.18 GB beside 16.3 GB resident, where main
modelled 75.6 GB. The plane stage then binds at 16 blocks (charge) and
50–63 blocks (currents), with t_u = 0.44 s per (block, row). Projected P16 ζ
fits: charge 18 × 110 s ≈ 33 min, currents ≈ 62 min (91 min measured with the
old doubling to 100 blocks). Decks do not change.

## 2026-10-03 — sector face rounds solve on the Ritz carrier; CT span widths are held from map 0

The CC/TT face reduction solves its kept span on `face_ritz_carrier` (the
pole budget, per-rank tile on the extent ladder), as the scalar face route
already did, and the sector batch is sized with it: the Schur and final eighs
run at the carrier and twice it instead of at H'_vv's side and the pencil
side (CrI3 24×24 bispinor P64: TT 9216/18432 instead of 12416/24832, whose
24832 stack ran on the whole mesh; CC 6144/12288 instead of 8736/17472). The
CT round's retained CC/TT span widths are held from SC map 0, so later
rounds and map 1 no longer recompile the CT reduction as the widths regrow.
Results: Fe 4^3 forced-face bispinor SC maps 0-2, eqp0 bitwise at map 0 and
within 0.077 µeV at maps 1-2; W response 197.0 / 62.9 / 13.5 s against
196.1 / 66.9 / 14.6 s (carrier). The CT binding on top is bitwise at maps
0-1 and within 0.032 µeV at map 2; map 1's CT stage drops from 20.0 to
14.2 s (W response 55.7 s against 62.9 s). Decks do not change.

## 2026-10-03 — the CC/TT/CT face batch is admitted by its compiled size

The bispinor sector constructor on the face route admits its parent batch by
the largest compiled size of the round's three whole-chain programs at the
conservative shapes (CC's and TT's reduction, on the Ritz carrier since the
entry above, and CT's cross reduction), beside
the resident sector models, as the scalar face batch already was. Every CC,
TT and CT eigh stack is decided against the room beside that batch row, which
bounds every round's retry. Each face model check of the scalar constructor
takes its eigh room beside its own whole-chain program. The analytic face
prices of the sector reduction ($14R^2+12nR$) and of the CT cross reduction
($10CT+14R^2+12n(C+T)$) are gone. Both face batches now start at every parent
and step down by room, so a deck that fits runs one round. The
one-parent-per-rank start of the previous entry is withdrawn; it split Fe 4^3
at P4 into four rounds whose different sides recompiled every program. The
sector constructor line names each sector's compiled program, its sizing
wall and the eigh rooms used. Results: Fe 4^3 forced face, SC maps 0-2.
Bispinor eqp is bitwise to main, and the W response is 197.3 / 66.7 / 14.6 s
against 150.2 / 67.2 / 14.6 s. Map 0 adds the one-time batch sizing (46 s).
The scalar returns to its pre-2026-10-03 batch of 13 and is bitwise to it; it
differs from current main by 0.071 ueV or less. Decks do not change.

## 2026-10-03 — face programs size on their first attempts; the scalar face batch is admitted by its compiled size

A whole-mesh (face) shared-pole program now holds only the first attempt and
check of every checked eigh inside it and returns their mesh-reduced failure
flag (`distrib_la.checked_program`); a failed first attempt returns zeros so
the program stays finite. Only when the flag is set does the round run again,
with every eigh's whole chain in-graph and on the whole mesh; a solve that
fails every attempt refuses by name. Route-(c) stack
decisions inside these programs are sized on the first attempt, not the
in-graph retry chain. The scalar face constructor admits its parent batch only
by the compiled size of its round program at the conservative side
(`face_batch_width`, starting at one parent per rank), and that figure
replaces the `14R^2+12nR` price in its reduction rows. Each round is admitted
by the compiled size of its whole-chain, whole-mesh program on the arrays it
runs, which is also its retry; its eigh stacks take route (c) in the room
that leaves. The constructor
line names the compiled size and its sizing wall. The CC/TT/CT sector batch
keeps its analytic price. Results move by round-off: Fe 4^3 forced face SC
maps 0-2 within 0.071 ueV of main (scalar; bispinor bitwise), CrI3 24x24
charge P16 eqp0 within 0.063 ueV. On CrI3 P16 80 GB the face batch is 8
parents (was 7), every reduction eigh stack runs one whole matrix per rank,
and the map-0 constructor drops from 902 s to 644 s. A face batch now holds at
most one parent per rank: Fe 4^3 forced face at P4 runs 4 rounds instead of 1
(+0.5 s per later map). Decks do not change.

## 2026-10-02 — distrib_la decides where a face eigh stack runs

The whole-mesh (face) shared-pole constructors, scalar and CC/TT/CT, hand
every eigh stack to `distrib_la` with the room per rank beside their admitted
batch. The service runs the stack as whole matrices per rank (route (c), the
local solver) when the compiled program that runs it fits that room, in as
many slices as it needs, and on the whole mesh otherwise. The n x n eighs
(directions, infinity, passivity) take route (c) almost always; a side-sized
Gram eigh takes it only where 7 n^2 x 16 bytes per matrix (3 n^2 compiled,
4 n^2 cuSOLVER workspace) fits beside the batch (n = 9152: 9.4 GB; n = 18304:
37.6 GB). Each decision is printed once
after the constructor line, `distrib_la eigh stack B x n^2 complex128:
...`, and the constructor line now names the face batch and its eigh rooms.
Every route-(c) eigh is checked against the service's probes with the same
retry chain as a distributed eigh, and the route is decided on the compiled
size of what the program reserves. The choice is agreed over ranks. Results
move by eigenvector gauge only: Fe 4^3 forced-face
SC maps 0-2, scalar and bispinor, agree with local main to 0.15 ueV. On the
same deck the face constructor's maps 1-2 drop from about 15 s to 2 s on 4
A100s.

The ordered (time-reversal-broken) face reduction now solves its kept span
on the pole budget's carrier, as a local round does: its Schur and final
eighs run at the carrier and twice it instead of at H'_vv's side and the
pencil side (CrI3 24×24 charge on 4×4: 6144 and 12288 instead of 9152 and
18304). The dropped columns are exact zeros: CrI3 P16 eqp0 and eqp1 are
bitwise to the run without it, and its map-0 constructor drops from 1354 s to
about 1030 s (W response 1470 to 1017 s). Decks do not change.

## 2026-10-02 — distributed eigh and LU solves are checked before they are returned

cuSOLVERMp 0.9.1's distributed eigh returns wrong eigenvectors with status 0
and info 0 on valid Hermitian inputs: exact-zero blocks, large near-zero
clusters of rank-deficient PSD matrices, sentinel-padded pencils. This was
reproduced outside LORRAX. On main the face (whole-mesh) route used some of
these silently: 10 eighs per constructor on CrI3 24×24 charge at P16, 2 per
map on Fe 4³ bispinor with the sector route on the face.

- **Every distributed eigh is checked** with fixed-seed probes (backward
  error within `distrib_la.roundoff_tol(n, dtype)`). A failure is solved
  again: shifted, with the vectors re-orthonormalized; then in the other
  cuSOLVERMp layout; then gathered when it fits 64 MiB.
  Each retry prints one `distrib_la:` line on rank 0's stderr.
- **Every distributed LU solve and Cholesky/LU `factor`/`solve`** is checked
  on its solution.
- **A result that no retry repairs refuses** with `GATE
  distrib_la_result_check` (op, n, call site, errors). Inside a program the
  result is NaN, so the caller's finite-result gate refuses too.
- **Every eigh route deflates exact-zero rows** with sentinels relative to
  the matrix. gw's own copies (`zero_row_safe_eigh`) are gone.
- **Which results move:**
  - Runs whose face eighs failed silently on main move to the correct
    decomposition.
  - Local-route runs move at round-off only: Fe 4³ scalar and bispinor maps
    0–2 ≤ 0.17 µeV, from the relative sentinels and the deflation of the
    normal and polar kernels.
  - Forced-face Fe 4³ bispinor agrees with local to ≤ 1.3 µeV.
- **Memory:** a checked distributed eigh reserves 7 n²/P elements per rank
  inside a program, priced by `distrib_la.workspace_bytes_per_rank`. An eager
  call reserves only its first attempt, about 3 n²/P, and runs any retries
  as a separate program.
- **Nothing to change in decks.** The check has no dial. The native
  cuSOLVERMp handlers do not yet read `info` (next bundle).

## 2026-10-02 — planners admit by compiled size; kin_ion writes the window GW reads

No result moves: Fe 4³ scalar and bispinor and Na 8³ SC maps 0–2 are bitwise.
What changes is which runs warn, the files kin_ion writes, and a few receipts.

- **`gw.kin_ion_io` writes b3 bands by default.** b3 = nelec + ncond, the
  window GW reads, with `number_bands_protected` resolved as GW resolves it.
  The old default, max(`number_bands`, nelec + ncond), loaded ψ(G) for bands
  no consumer reads; CrI3 24×24 went from 750 bands to 208. A `kin_ion.h5`
  written before still works, and `-n NB` still writes more.
- **Compiled chunks are admitted only when they fit.** A compiled chunk is
  admitted only when it fits the room; the analytic price no longer admits it.
  A chunk at its minimum that does not fit now prints the over-budget warning
  instead of passing silently. Every rank runs the smallest chunk any rank
  chose (`runtime.aot_memory.agreed_chunk`).
- **The ζ μ-batch plan states its real batch.** Route G keeps whole centroid
  orbits per owner, so the plan floors the batch at P × the widest orbit.
  Its HWM is that of the batch that runs, and it warns before the compile
  when that HWM is over the target. CrI3 24×24 bispinor at P36 now plans
  432 centroids at 70 GB/rank, where it used to print 36. It does not fit
  40 GB cards at any P ([zeta fit](docs/architecture/zeta_fit_mubatch.md)).
- **Hermiticity checks scale with n.** Every Hermiticity check compares
  against `distrib_la.roundoff_tol(n) = 64·n·eps`
  ([api](docs/services/distrib_la/api.md#roundoff)), not a fixed 1e-12. A
  large GEMM-built matrix no longer refuses on round-off. A matrix of side
  n < 71 meets a bar below 1e-12.
- **Receipt changes.**
  - `constructor_receipt.json` no longer carries `spectral_moment_cauchy`;
    the report never passed its support test. A non-finite sector M1 still
    refuses (`GATE shared_pole_sector_nonfinite`).
  - On a deck with broken time reversal, the parallel-transport head no
    longer computes the velocity-parity diagnostic; it prints one line
    instead.

## 2026-10-02 — XLA rematerialization is off; a module larger than the device stops by name

LORRAX turns off XLA's HLO rematerialization pass for every program: the
runtime adds `--xla_disable_hlo_passes=rematerialization` to `XLA_FLAGS`,
merged into any pass list you already set. The pass ran only on a module
whose peak was above XLA's memory limit. It could add hours of compile, and on
the decks that printed `LORRAX GATE xla_rematerialization: a module does not
fit` it freed nothing. Compile time drops where the pass ran long (on the
CrI3 8×8 kmeans Gram it gave up in under a second, so nothing changes there).
Results do not move: Fe 4³ scalar and bispinor SC maps 0–3 and kmeans
centroids are bitwise.

Every compiled module is now checked against the card before it first runs.
When its buffers (temp + arguments + outputs − aliased) exceed the device's
total memory, the run stops with
`GATE xla_rematerialization: REFUSED module '<name>': its compiled buffers
need X GB per device (…), and the device has Y GB`, instead of an
out-of-memory error deep in the allocator. Raise the rank count or lower the
module's size ([memory model](docs/architecture/memory-model.md#module-does-not-fit)). The
`memory_per_device_gb` budget still only warns. The old stderr banner is gone.

Expect one cold compile: `XLA_FLAGS` is part of the compile-cache key, so the
first run of every deck after this change recompiles every program (the P4
hsuite took 445 s cold), and later runs hit the cache again. If a future jax
changes the private executor this check hooks, the run prints one
`jax-compat: … GATE xla_rematerialization is NOT armed` line and continues
unchecked.

## 2026-10-02 — band extrapolation on the four-current Σ (`full_shared_pole`)

`bispinor_gw = full_shared_pole` now uses `use_band_extrapolation` (on by
default), as the scalar shared pole and `bare_transverse` already do. The CC
class's Green band sum is split into the three brackets and extrapolated by
the pooled `spectral_shell` fit. TT, CT and TC, which are c⁻² of CC, are
summed to N as before ([four-current Σ](docs/theory/band-extrapolation.md#four-current)).
Every `full_shared_pole` run that does not set `use_band_extrapolation = false` moves
once. A deck with `number_bands_sigma` < 2·n_occ now refuses at startup:
raise the band count or set the key to false. Decks that set it to false are
bitwise. On Fe 4³ bispinor `full_shared_pole` SC at 36 bands, map 0 eqp0 within E_F ± 10 eV
moves by a median of 342 meV (at most 891 meV, mean −385 meV). Σ τ costs 13–20 %
more per map, and the map wall 3–6 % more; device peak is unchanged.

## 2026-10-02 — "door" is gone from the code and the logs

The term "door" is replaced everywhere. A mathdx k-convolution call site and its
symmetry tables are now a "kconv call" and "kconv tables". A service's top-level
package is its "public API". A single function that serves one operation is an
"entry point". The `ffi.fft` router's functions are "factories". Three output
names change: the stage `response.door_tables` is now `response.kconv_tables`,
the four-current line reads "four-current mode-11 kconv calls: N kconv calls",
and the capacity-ledger row `door_tables` is now `kconv_tables`. A parser that
matched the old names must match the new ones. `minimax.door` is now
`minimax.serving`; `import minimax` is unchanged. No number moves.

## 2026-10-01 — the shared-pole χ and W banks stream through scratch; keep terabytes free

When a shared-pole map's χ bank (the value and slope of every response sample)
does not fit the devices, the response stream now runs once with every sample
and writes its carry to SlabIO's per-rank streamed tier, instead of re-running
every Green pair once per sample group. The W bank goes to the same tier when
it does not fit the devices. Each bank is held in host memory when it fits half
the host budget, else in one file per rank under
`<run dir>/<label>_shared_pole/streamed_bank/`. gwjax.out names the choice, for
example `Response quadrature: chi bank file, 34.94 GiB/rank; streamed; exceeds
half the host budget`. Results are bitwise (Fe 4³, Na 8³ and Fe 4³ bispinor
SC maps 0–2). The Fe 8³ P4 χ build takes 35 → 19 s per map.

- **Scratch.** A large deck needs the χ bank's bytes free on scratch, and
  under the quota, during every map: 2.4 TB for Ni 20³ charge and 15.3 TB for
  Fe 20³ bispinor at 1062 parents
  ([memory model](docs/architecture/memory-model.md#streamed-chi-bank)). When
  the disk or quota cannot hold it, the samples run in disk groups (half the
  samples per try), then in device groups. Each group re-runs every Green
  pair: the Fe 20³ bispinor χ build is about 420 s per map in one group and
  1,300–1,400 s in four.
- **Lifetime.** A file is unlinked as soon as it is opened, so `ls` shows an
  empty directory while the run holds the bytes, and every exit, SIGKILL
  included, frees them. Nothing is kept across SC maps, and an interrupted
  map cannot resume from the tier.
- **Cleanup.** Code before 2ee02941a left named files after an abnormal exit.
  The first shared-pole map of a process now removes every `streamed_bank/`
  directory in its run directory and prints `WARNING shared-pole output:
  removed streamed stores left by an earlier process`. Delete those
  directories of finished runs by hand
  ([SlabIO](docs/architecture/slab_io.md#streamed-tier)).
- **Map-1 failure, fixed in 6bc09936c.** With the W bank on per-rank files,
  a store created after others had been written counted their bytes twice
  against the quota, was refused, and was then written: the map failed with
  `GATE io_global_commit … streamed_bank.commit` and a `KeyError` (Ni 20³ P64,
  map 1). A source tree before 6bc09936c, and a module built from one, still
  fails this way.

## 2026-10-01 — row passes run as one scan; compile no longer grows with the pass count

The Σ τ node, the static τ = 0 node (Σ_x, SX, COH), the bispinor sector node
and the charge and four-current χ streams now run their row passes as equal
orbit-aligned windows in one `lax.scan`. The unfold tables are cut on the
device and the k-convolutions skip each window's padded rows
([memory model](docs/architecture/memory-model.md#the-green-side-stages)), so
one program serves every pass. Compile: the static Σ program at 223 passes
25.7 → 0.8 s; the streamed χ bank's segment programs 33 (21.6 s) → 1 (0.7 s)
on the charge stream and 203 (234.6 s) → 4 (4.9 s) on the four-current
stream, whose peak falls by 3.6 GB per rank. Walls are within ±1 %. A rank whose rows fit one
tile runs one pass and is bitwise (Fe 4³ scalar, Na 8³ and Fe 4³ bispinor
`full_shared_pole`, maps 0–2). With several Σ passes per rank the sums group
differently and move at round-off, the size of a 4e-16 control (≤ 1e-7 eV on
Fe 4³ forced to many passes); the χ streams stay bitwise. The
k-convolutions' `live` operand needs a native bundle built from 935bcede8 or
later.

## 2026-10-01 — the photon bank solves a sample's Dyson value and slope in one program

The four-current bank now forms each sample's W − W∞ and its slope in one program, as
the charge bank does; the contact and W∞ − V enter in the old order. Fe 4³ bispinor
Dyson dispatch per SC map is 18 % faster. Results move at round-off: Wc is bitwise,
the slope moves at 1.3e-15 relative on the local backend (bitwise on the distributed
one), and Fe 4³ bispinor SC maps move ≤ 0.08 µeV, inside a 4e-16 control.

## 2026-10-01 — the bispinor sector constructor solves on the Ritz carrier

A bispinor SC map's sector constructor now solves each round's kept span on the
ladder rung of the largest kept count that sector has had, as the scalar model
does, and reruns wider when a round keeps more. Map 0 is bitwise; later maps move
at round-off (Fe 4³ bispinor ≤ 0.09 µeV).

## 2026-10-01 — the bispinor sector W(τ) runs on the scalar model's synthesis

Each photon sector's W(τ) = B_A d(τ) B_B† is now formed by the scalar shared-pole
synthesis (`gw.mpa.sigma.synthesize_shared_pole_parents`) and placed by the scalar
rule: replicated pole columns when they fit, else whole parents per rank on a
`linalg = local` deck, else the face SUMMA. Only each parent's live pole columns
are contracted. The SUMMA panels and the W∞ − V constant's q panels are sized from
the shapes (one tile), no longer from `memory_per_device_gb`. Fe 4³ bispinor `full_shared_pole`
Σ τ is 19 % faster per SC map. Bispinor results move at round-off: map 0 is bitwise,
maps 1–2 move ≤ 0.19 µeV (a ±4e-16 control moves them ≤ 0.11 µeV). Scalar decks
are bitwise.

## 2026-10-01 — the shared-pole Σ τ window overlaps one node's W exchange with the next node's compute

On GPU, the scalar shared-pole Σ τ window now evaluates two τ nodes per loop
trip when the compiled paired window fits the device budget beside the live
stages. Otherwise it runs one node per trip, as before. When it pairs, the Σ τ
window line of gwjax.out's memory table says "two nodes per trip". Only
that program is compiled with XLA's latency-hiding scheduler
(`gw.ppm_accumulators.WINDOW_OVERLAP`); the global flag stays off. One node's
W(τ) synthesis and all_to_all now run beside the other node's k-convolutions.
On the Ni 20³ SC map 0 at P64 (band extrapolation on), the window executables
take 351.7 → 322.3 s (−8.4 %). The window holds a second node's live set:
compiled 9.76 → 16.95 GB per rank, and a run peak of 28.5 GB instead of 23.9 GB,
now set by Σ τ. The Σ τ stage prices this and admission reads it from the
compiled executable. On a cold cache the first window compiles 13 → 34 s.
Results move at round-off: Fe 4³ and Na 8³ SC maps 1–2 stay inside a 4e-16
control, map 0 is bitwise, and the Ni map is bitwise at printed precision.

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

## 2026-10-01 — distributed eigh runs block-cyclic

A `linalg = distributed` eigenproblem (cuSOLVERMp, `distrib_la`) now describes
each rank's tile with a square block, the largest divisor of n/p at most 256,
so no rank idles as the trailing matrix shrinks; one all_to_all over `x` puts
the eigenvalue index back in order. At n = 16000: 17.50 → 10.59 s on 16 GPUs
(1.65×), 20.05 → 15.65 s on 64 (1.28×); neutral at n/p ≤ 1000. LU and
Cholesky keep one tile per rank (block-cyclic solves measured 2–3× slower).
Eigenpairs move at round-off (eigenvalues ≤ 1e-15 relative). Fe 4³
`linalg = distributed` eqp0 at maps 0–2 is bitwise at its natural block and
moves ≤ 1.8e-4 meV at a forced block of 72, the size of a 4e-16 control
(1.4e-4 meV).

## 2026-10-01 — the static Σ runs the Σ τ kernel at τ = 0; the map-0 exchange compile is gone

Σ_x (every SC map) and static COHSEX on the parent route now run the Σ τ
sub-tile kernel at τ = 0 with the static interaction in place of W(τ), its
unfold tables read as device operands. Before, the Σ_x program baked the
Green's global unfold tables into its HLO: at Ni 20³ on 64 GPUs (1782
centroids) that compile took 93 s of the 105–120 s map-0 "Sigma exchange",
which now takes about 10 s. At the P64-local tile a cold call takes 28.7 →
6.7 s (compile 25.0 → 2.4 s), a warm call 0.720 → 0.672 s, and the peak
14.0 → 8.9 GB per rank. Results move at round-off: Σ_x by 4.1e-16 relative
at most; map 0 is bitwise, and Fe 4³ and Na 8³ SC maps 1–2 move by the size
of a 4e-16 Σ_x control.

## 2026-10-01 — no `coulomb.h5`; an older bank's constructor resume refuses

Every scalar shared-pole map wrote the bare V to `coulomb.h5` and read it
back to build the Coulomb roots (54.6 GB and 41 s at Ni 20³ map 0, plus a
content hash). The roots are now formed from V on the devices, bit for bit
the file route's, and V is named by a device digest that is the same on
every rank and at every P. Results are bitwise, and a map directory holds
no `coulomb.h5`. Resuming a constructor from a bank built before this change
refuses with `GATE shared_pole_output: the bank at … was built with another
bare V; use a fresh run directory`: delete that map directory or use a fresh
run directory. The photon V file is unchanged.

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
forced to 5 on Fe 4³ (groups 5,5,5,5,2), eqp0 at maps 0–2 is bitwise to the
previous code at group 5 and at the default single group, and map 0 compiles two direct
programs instead of three. Decks whose samples fit one group are unaffected.

## 2026-09-30 — the scalar χ₀ mode-11 kconv call reads its tables as operands

The charge response stream on a raw-parent plan (the shared-pole direct
stream, the moment correlations and the retarded stream) baked its plan's
global unfold tables (row, trs, lsrc, rsrc, mph, nph, spin) into every
program as HLO constants. The tables now enter as device operands, placed
once per run and plan (`ffi.fft.make_kconv_chi_unfold(load=)`, as the
four-current route does), and the stream binds them as a trailing argument.
At the Fe 20³ table size (8000 k, 1792 centroids, one node): compile
7.08 → 0.57 s per program, generated code 115 → 0.1 MB, and each held
executable no longer adds host memory (+1.54 GB → +0.00 GB RSS per rank for
the second program). The Fe/Ni 20³ P64 runs compiled this program three times
at map 0 (34–48 s each), and the Ni run died of host memory during the
third. At the P64-local shape the stream's temporaries are 13.74 → 13.60 GB
and its dispatch is unchanged (0.439 s per Green pair). Results are bitwise
(Fe 4³ scalar and Na 8³ SC, three maps). The Σ τ mode-7 and mode-8 kconv calls
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
`full_shared_pole`, Na 8³ and MoS2 SC + BSE match the previous code, beside a 4e-16
control that moves.

`distrib_la.panel_matmul_extra_tiles` is deleted, and the photon row-pass count
no longer adds two Green tiles at p_x ≥ 3. Fe 20³ P36 at M_T 900 now prices
1,1,1,1 row passes at 70 or 75 GB (was 2,1,1,1), counted at 64.33 GB. M_T 1800
prices 3,7,7,14 at 75 GB (was 4,7,7,15). The 2,1,1,1 that the counted-passes
entry below gave P36 was an over-count. P36's 6-panel loop never held the two
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
- the placed kconv tables and mode 11's run-time scratch.
The ledger budget (`memory_per_device_gb`) and `GATE response_photon_passes` are
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
tables as device operands. Each kconv call used to bake its plan's global tables
into the program as constants. Each distinct table array is placed once per
run (stage `response.kconv_tables`), and every q batch and SC map reads it.
No kernel or bundle change.

Fe 20³ P36-local AOT (M_T 900, 2 row passes per family pair, cold cache, 4 ranks per node):

| | before | after |
|---|---|---|
| compile per rank | 136.7 s | 5.7 s |
| host max RSS per rank | 48.1 GB | 3.1 GB |
| node host peak | 175 of 251 GiB | 42 GiB |
| device args / temp / code | 24.63 / 40.03 GB / 644 MB | 25.44 / 39.80 GB / 0.45 MB |

What moves: nothing. Fe 4³ bispinor `full_shared_pole` SC is bitwise, and its warm
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
([self-consistency](docs/self_consistency.md#metals-direct-drude-head)).

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
([self-consistency](docs/self_consistency.md#metals-direct-drude-head)).

## 2026-09-30 — regenerate `WFN_qp.h5` from WFNs that store both k and −k

`WFN_qp.h5` now keeps time reversal on a WFN that stores two k of one orbit
(e.g. MoS2 3×3). A file written before this from such a WFN has broken rows,
and the BSE and GW runs that read it used them. Regenerate it with
`python -m postprocess.rotate_wfn_to_qp WFN.h5 qp_wfn_rotations.h5`; WFNs
without such rows (Si) give the same file
([self-consistency](docs/self_consistency.md#8-seeding-restart-and-outputs)).
An SC run that writes `WFN_qp.h5` now binds `dipole_qsgw.h5` to it, so a GW
run on `WFN_qp.h5` can take that file as its `dipole.h5`.

## 2026-09-30 — SC W line sites held within max(3 meV, 0.1 × max|dE|)

Held shared-pole W line sites are re-placed only when they would move by more
than max(3 meV, a tenth of the previous map's max|dE|). SC runs that re-plan
their sites move once; Fe 4³ scalar `parallel_transport` SC now converges
(28 maps; it stalled at map 16)
([self-consistency](docs/self_consistency.md#shared-pole-w-with-retained-quadrature)).

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

New SC key, default `dft`; `qp` is the previous behaviour. Every dynamic SC
deck with a coarse class moves once.
Under `dft` the coarse (semicore) class keeps its DFT block of H in the DFT
basis and its end of every protected–semicore element reads Σ at E_DFT on the
held coarse windows ([self-consistency](docs/self_consistency.md#2-band-treatment)).
Fe 4³ and MoS2 3×3 decks requested with `number_bands_protected`, with the
semicore read at η_semi = 5 eV and, in comparison builds with
`qp_support.SEMICORE_ETA_EV` set to 8 eV, at 8 eV: same maps to converge (14, 8),
equal or fewer τ pairs (MoS2 333 → 321), semicore QP within 27 meV of DFT
(qp: 0.1–6 eV deeper), and the protected states' η_semi 8 − 5 spread falls
3–6× (E_F ± 10 eV std Fe 5.1 → 1.4, MoS2 5.0 → 0.8 meV). The default is a
no-op on a run without a coarse class (static modes, an `nval`
that covers every occupied band).

## 2026-09-29 — coarse (semicore) windows certified at max(ε, 3e-3)

Self-consistent decks with a coarse class move once. The coarse windows are
certified at max(`sigma_quadrature_eps`, 3e-3) (`qp_support.SEMICORE_EPS`),
so at the default ε 1e-4 they take 3e-3; every other Σ
window keeps `sigma_quadrature_eps`. At
η_semi 1 eV against ε 1e-4: map-2 τ pairs MoS2 3×3 530 → 465, Fe 4³ charge
1087 → 982; states within E_F ± 10 eV move ≤ 0.06 meV at maps 0–1 (≤ 1.9 meV
at map 2 of the unconverged Fe run); semicore QP ≤ 4.3 meV at map 0. The
planner's node law for grouping coarse windows is now evaluated on the box
each run is built on, so it equals the certified count.

## 2026-09-29 — the production QSGW partition: counted b3, semicore Σ read class

Every dynamic self-consistent deck with a coarse class (`qp_solver =
self_consistent`, scalar MPA/shared-pole route) moves once. See
[self-consistency](docs/self_consistency.md#2-band-treatment).

- **b3 counts bands, as before.** b3 = nelec + `ncond`. The QP matrix
  [b0, b3) rotates among itself;
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

## 2026-09-28 — no quadrature rule is stored across runs; `sigma_quadrature_cache_dir` is retired

Every run places its own Σ(ω) and χ quadrature rules and reuses them only
inside the run. The deck key `sigma_quadrature_cache_dir` refuses by name:
remove it. `LORRAX_MINIMAX_CACHE_DIR` and `LORRAX_DISABLE_MINIMAX_DISK_CACHE`
are no longer read: unset them. The directories
`$SCRATCH/.cache/lorrax/sigma_box_rules` and
`~/.cache/lorrax/minimax_quadratures` are no longer used and can be deleted.
Results equal those of a run that started with an empty cache.

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

## 2026-09-26 — `occupation_window_threshold` is retired; one occupation support for χ and Σ

The deck key `occupation_window_threshold` refuses by name: remove it. A band
belongs to a Green's-function branch of χ or Σ if and only if its weight
(f on the occupied branch, 1 − f on the empty one) is at least 1e-5 in
magnitude, which for Fermi–Dirac occupations is 11.5 k_BT from μ
(`gw.efermi.band_in_occupation_window`). The one-shot and every SC map use
this one support; the previous floor was 0.005 (5.3 k_BT), which dropped
states that still carry weight. Metal decks move once; insulators, whose
weights are 0 or 1, are unchanged.

## 2026-09-26 — `band_extrapolation_estimator = band_index_only` is retired

The value `band_index_only` refuses by name. Remove the key, or set
`band_extrapolation_estimator = spectral_shell`, the default and only accepted
value ([band extrapolation](docs/theory/band-extrapolation.md)). The
band-index fit S(N) = S_∞ + A/N ignores where the omitted bands lie in
energy. Decks that did not name the value are unchanged.

## 2026-09-24 — FFI handler ABI 6; a native library of another ABI refuses

The Python tree and its native FFI handlers agree on handler ABI 6
(`src/ffi/cpp/common/lorrax_ffi_abi.h`, mirrored by
`ffi.common.ffi_loader.LORRAX_FFI_ABI_VERSION`). A sealed native bundle whose
manifest records another `ffi_abi`, or a loose `liblorrax_ffi*.so` that stamps
another ABI, refuses at startup before any handler is called ("native bundle
ABI mismatch" or "HANDLER ABI MISMATCH"), because a mismatched handler
otherwise fails later with an argument-count error that names neither
library. A bundle's CUDA and host legs are used together:
`LORRAX_FFI_SO` and `LORRAX_FFI_HOST_SO` must both be unset or both select
one bundle, and a partial or mixed override refuses. What to change: on
Perlmutter, members of m4598 load the current `lorrax_A` module
([installation](docs/installation/perlmutter.md#module)); elsewhere rebuild and reseal
both legs from this source tree ([installation](docs/installation/index.md)).
Do not point either variable at a library built from older source.

## 2026-09-24 — k-axis convolutions on nvidia-mathdx

- **NVIDIA GPUs now require the `nvidia-mathdx` wheel** (pinned in the
  `cuda12`/`cuda13` extras; header-only).  Without it a CUDA run refuses at
  startup with `GATE mathdx-headers` and the fix `pip install nvidia-mathdx`.
- `LORRAX_FFT_FFI_FUSED`, `LORRAX_CONV_KMINOR_FFI` and `LORRAX_CONV_KLEAD_FFI`
  are gone: Σ, COHSEX and the BSE ladder/stack convolutions have one route
  per platform (nvidia-mathdx on CUDA, the host FFTW plans on cpu).  A leftover
  setting is ignored.
- The kernels compile on first use (about 6 s per k-grid) and are kept in
  `$SCRATCH/.cache/lorrax/kconv_mathdx` (else `~/.cache/lorrax/kconv_mathdx`);
  the second run of a deck loads them in ~10 ms.
- The flat-k transform (χ₀, head, htransform) also runs on nvidia-mathdx on
  CUDA.

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

## 2026-08-18 — retired HDF5 controls now refuse or are absent

- GW decks containing `slab_io` or `use_ffi_io` now refuse with a targeted
  removal message. Remove either line; SlabIO has one collective transport.
- kmeans no longer accepts `--use-phdf5`; `WfnLoader` selects its one valid
  scalable read path from runtime capability.
- SlabIO no longer accepts `chunks=`. The argument was ignored and every
  collective dataset was already contiguous. The sigma and zeta writers no
  longer request a layout the native create cannot produce.
