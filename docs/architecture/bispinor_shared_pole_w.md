# Shared-pole W for bispinor sectors

`bispinor_gw = full_shared_pole` screens the four-current interaction with
one shared-pole model per Lorentz sector. This page builds that object from
the photon propagator to the code: the sector model, the carrier its rows
live on, the response bank that samples it, the route a map takes, the
construction on that route, the store, the Σ consumer, and a worked byte
budget. The scalar shared-pole model, its derivation and the ordered
reduction are [the shared-pole theory page](../theory/shared-pole-w-model.md);
the scalar bank, directions, pencils and store schema, which the sectors
reuse, are [the shared-pole implementation page](shared_pole_model.md); the
four-current physics and the $1/c$ counting are
[bispinor GW](../theory/bispinor-gw.md). Code is cited as `file:line` at
`d9665201b`; read the file rather than the number.

## Symbols

| symbol | meaning |
|---|---|
| $P = P_x P_y$ | ranks on the square X/Y mesh, $P_x = P_y$ |
| $n_q$ | raw parents: irreducible $q$ of the magnetic group (`n_q_irr`) |
| $A, B$ | Lorentz index: 0 charge, 1–3 Cartesian current |
| C, T | the charge family ($A = 0$) and the current family ($A = 1,2,3$) |
| $n_C$, $n_T$ | packed centroid counts of the charge and current bases (`MuBasis.n_packed`) |
| $n_S$ | rows of sector $S$: $n_{CC} = n_C$, $n_{TT} = 3n_T$ |
| $K_S$ | pole budget of sector $S$, $\lceil 1.8\,n_S\rceil$ on the `production` tier |
| $R_S$ | ordered pencil side of sector $S$ for one round (the `side` of the report line) |
| $c_S$ | kept-span carrier of a face reduction, `face_ritz_carrier` of $K_S$ |
| $N$ | parents per construction round |
| $\mathcal B$ | planner budget per rank, the [memory rule](memory-model.md#budget) or a deck's `memory_per_device_gb` |

## 1 The model

The bare photon propagator $V$ couples the charge and current densities
through the Coulomb-gauge kernel; it is Hermitian but signed. With the
paramagnetic response $\chi_p$ and the diamagnetic contact $D$, the screened
interaction is $W = (I - V\chi)^{-1}V$ with $\chi = \chi_p - D$, and its
high-frequency limit is $U = W_\infty = (I + VD)^{-1}V$. Theory §9 (W 34)
shows that $W(z) - U$ is a stable ordered pencil whenever
$H_0 + C^\dagger U C$ is positive, even though $U$ is indefinite.

The code splits $W - U$ into the endpoint blocks CC, TT, CT and TC and
represents each by the ordered model of theory (W 16). In imaginary time,
for parent $q$ and the particle branch,

$$
W_{AB,+}(q,\tau) = B_A(q)\,\operatorname{diag}\!\big(d_j(\tau)\big)\,B_B(q)^\dagger,
\qquad
d_j(\tau) = \frac{e^{-i(\Omega_j - E_{\rm ref})\tau}}{2\Omega_j},
\tag{S 1}
$$

with $B_A(q)$ the endpoint factor $[n_A\text{-family rows}, K]$ and $\Omega_j > 0$.
The antiunitary partner is $\bar B_A\,d\,B_B^{\mathsf T}$, and the hole branch is
the partner at $-q$:

$$
W_{AB,-}(q,\tau) = \bar B_A(-q)\,\operatorname{diag}\!\big(d_j(\tau)\big)\,B_B(-q)^{\mathsf T}.
\tag{S 2}
$$

CC and TT each carry their own pole set. CT and TC share one pole set,
reduced on the joint span of the CC and TT directions; TC is not stored, since
$W_{TC} = (\text{partner}_{CT})^{\mathsf T}$. The instantaneous constant $U - V$ is
stored apart from the poles and enters Σ once (§7).

Each sector reduction gates positive retained $\mathcal H$ on its own span
(`_sector_gates`, `src/gw/shared_pole_sectors.py:691`). That certifies each
projected sector. It does not certify positive residues of the assembled photon
matrix, which is judged by its integrated Σ. The scalar passivity bound (W 9)
does not apply to a signed $V$.

**Treatment ceiling.** A sector model deactivates poles above

$$
\Omega_{\rm treat} = 2\Big[\max E_{{\rm cond},\chi} - \min E_{\rm val}\Big],
\tag{S 3}
$$

a numerical treatment, not a bound on collective modes
(`shared_pole_gates.shared_pole_treatment_mask`, applied in
`construct_sector_poles`, `src/gw/shared_pole_sectors.py:158`). An SC run holds
the first map's value while the current span stays inside it, and re-plans it at
twice the current span when the span exceeds it. Inactive modes have zero
factors and the inert pole $\Lambda = 1$ Ry². CC and TT have independent masks;
CT_C and CT_T share one. The refusals are
`GATE shared_pole_sector_treatment_{census,order,empty}`. The held rows score the
untreated fit.

## 2 The carrier

Every vertex of the four-current route acts on the normalized RKB lift of the
two-component wavefunction ([bispinor GW](../theory/bispinor-gw.md)). The charge
and current densities are fitted on two separate centroid bases, $n_C$ and
$n_T$ packed centroids. They are never merged.

The photon operator lives on the direct sum $C \oplus T_1 \oplus T_2 \oplus T_3$
(`PhotonBasisLayout`, `src/gw/photon_layout.py:63`). Each family is padded to a
mesh-divisible carrier, and the three current components share one carrier. On
the face `P(None,'x','y')` each rank's tile holds its C rectangle, then its three
T rectangles: the stored operator is mesh-interleaved.

A sector read converts the stored rectangle to the sector's own row order
(`_sector_indices`, `src/gw/shared_pole_sectors.py:56`). CC rows are the charge
centroids. TT and the T side of CT are μ-major and Cartesian-component-minor,
$3n_T$ rows. Inactive (padding) centroids are masked to zero. The constructor
therefore sees each sector as an $n_S \times n_S$ (or $n_C \times 3n_T$) scalar
problem, and the scalar pencil, reduction and gates apply to it unchanged.

Retained direction ranks sit on the extent ladder (`port_extent`,
`src/gw/shared_pole_directions.py:139`): a width is rounded up to an eighth of its
leading power of two, then to a mesh divisor, so a carrier, and every program
keyed on it, repeats across rounds and SC maps.

## 3 The bank

`compute_photon_bank` (`src/gw/response_bank.py:2825`) samples the photon $W$ once
per map through the scalar producer: the same shared-node response rules,
sample/derivative stream, Dyson solve, transaction masks and reader
([shared-pole model §2](shared_pole_model.md#2-the-response-bank)). It writes, per
parent,

- $W - W_\infty$ and $\partial_s W$ at the dense fitted samples (the imaginary
  axis) and the held samples;
- the ordered moments $M_0 \ldots M_3$;
- the constant $W_\infty - V$;
- for each fitted line support off the imaginary axis, each family's selected
  directions and state panels (theory (SP 7)) and its cross panels, the CT/TC
  actions on that family's directions.

The line panels are selected by the producer while the line sample is in hand
(`sector_line_selection`, `src/gw/shared_pole_sectors.py:127`). Each family's
endpoint block is cut exactly as a sector read cuts it, so the producer selects
on the bits the constructor reads.

Every map builds its own static contact $\Pi_{\rm FD}(0,0)$. Under
`head_correction = no_local_fields` the bank adds the direct Γ head field
$W_h - V_h$ to every sample through the packed $\zeta(G=0)$ vectors, so the fitted
CC, CT, TC and TT poles carry the head
([four-current heads §5](../theory/four-current-head-corrections.md#direct-bulk-head)).

The bank's residence (device, host, or one file per rank) follows the scalar
rule ([shared-pole model §1](shared_pole_model.md#1-one-map), step 3). Its
schema is the scalar bank's with `C`/`T` line families
([shared-pole model §7](shared_pole_model.md#7-store-schema)).

## 4 The route {#sector-route}

The route is decided once per map, before any bank read, from the recipe shapes
and the deck budget (`sector_route`, `src/gw/shared_pole_sectors.py:1614`). Every
rank prices the same shapes, so every rank decides alike. The report prints one
line, `Shared-pole sector constructor: route …`.

**q-local.** `sector_execution` (`src/gw/shared_pole_sectors.py:1661`) prices one
whole parent per rank for each sector at its conservative pencil side
(`constructor_side_upper_bound`). CC is priced first, TT beside CC's held
outputs, CT beside both (`held_sector_bytes`,
`src/gw/shared_pole_capacity.py:33`). The joint CT pencil is priced with both
spans at twice their pole budgets. If every sector fits, the map runs q-local:
rounds of $N = P$ parents, one per rank. One exception: with $n_q \ge P$ and a
passing CT selection, the joint pencil is admitted q-local at its actual spans,
over budget with a warning.

**Staged rounds.** Otherwise the map runs on the face in rounds of

$$
N_{\rm rounds} = \Big\lceil \frac{n_q}{\min(n_q, P)} \Big\rceil,
\qquad
N = \Big\lceil \frac{n_q}{N_{\rm rounds}} \Big\rceil
\tag{S 4}
$$

parents, balanced over the rounds (`staged_round`,
`src/gw/shared_pole_sectors.py:1566`). $N$ is set by $P$, never by the budget.
At $n_q \le P$ the whole map is one round.

## 5 Construction

### 5.1 A q-local round

`construct_diagonal_sector_round` (`src/gw/shared_pole_sectors.py:587`) runs CC,
then TT, in the batch layout `P(('x','y'))`, one whole parent per rank. Each is the
scalar ordered round of `gw.shared_pole_local` (`reduce_round`) on the sector's
rows, with the sector recipe (`sector_recipe`, `:575`: widths, line cap and
$K_S$ resized to $n_S$) and the sector keep cut $10^{-5}$. The round program
takes the round's packed columns ($\S$5.2, *Packed columns*), placed in the
batch layout before it runs (`reduce_round`, `src/gw/shared_pole_local.py:408`).
`construct_cross_sector_round` (`src/gw/shared_pole_sectors.py:998`) then reduces CT on the two diagonal spans
and keeps both endpoint outputs. Every eigh is a local dense solve.

### 5.2 A staged round

A staged round runs TT, then CC beside TT's held outputs, then CT from both
(`staged_round`). Each diagonal sector (`staged_sector`,
`src/gw/shared_pole_sectors.py:738`) has three phases.

1. **Selection.** The round's $N$ parents are read at once on the face: dense
   samples, moments and line panels. The infinity directions (eigh of
   $\operatorname{Herm} M_1$) and the state directions are selected as in the
   scalar route ([shared-pole model §3](shared_pole_model.md#3-directions)). Their
   eighs run one whole $n_S \times n_S$ matrix per rank, so the selection's unit is
   the round.
2. **Tables.** Each state panel is padded to its recipe carrier.
   `round_tables` (`src/gw/shared_pole_local.py:203`) lays every slot's selected
   columns into the round's finite extent $F$ (the pencil side
   $R_S = F + 2\,i_S$ with $i_S$ the infinity carrier); the first stage joins the
   panels and takes each slot's columns inside its program (S 4a, below). $F$ is
   grow-only over the sector's rounds and SC maps. The derivative panels $W'Q$
   are released after the first stage.
3. **Reduction.** The ordered reduction of theory §6.3 is cut at its three eighs
   into four GEMM stages (`face_reduce_decoupled`,
   `src/gw/shared_pole_execution.py:789`; stages in
   `src/gw/shared_pole_reduction.py:416,449,515,553`):

   | stage | computes | then eigh of |
   |---|---|---|
   | `paired_members` | the paired-basis members of $(G, H, O)$, equilibrated | $H'_{vv}$, side $R_S/2$ |
   | `keep_stage` | the $H'_{vv}$ keep cut and $K_S$, its metric correction, the restricted pencil | the Schur complement, side $c_S$ |
   | `paired_stage` | the Schur cut, $Y = L^{-H}$ on the kept span and its metric correction | $Y^\dagger G_r Y$, side $2c_S$ |
   | `output_stage` | the Ritz outputs, signed and positive models, zero-Ritz policy, pole sort | — |

   The metric corrections are coupled Newton–Schulz inverse roots inside their
   stages (`_metric_inverse_root`, `src/gw/shared_pole_reduction.py:20`), with the
   iteration count fixed from the initial infinity-norm bound. The receipt reports
   the largest $\|ZAZ - I\|_F/\sqrt R$.

CT (`staged_cross`, `src/gw/shared_pole_sectors.py:882`) assembles its joint
pencil (metric, value, $O_C$, $O_T$) per sub-batch from its own CT and TC samples
and both families' cross panels, at one compacted span for the round, written in
place into one stack (`_assemble`, `src/gw/shared_pole_execution.py:630`). The
diagonal sectors' selection panels are then released
(`release_selection_panels`, `src/gw/shared_pole_sectors.py:844`), and the joint reduction runs its keep and
output stages around two eighs, the metric and the Ritz step, both at the joint
side $K_{CT}$ (`face_cross_decoupled`, `src/gw/shared_pole_execution.py:882`).

**Packed columns.** A round's $S$ state panels $X_s \in \mathbb C^{n_S \times r_s}$
(one of $Q$, $WQ$, $W'Q$ per field) sit at offsets $o_s = \sum_{t<s} r_t$ of
their joined columns. `round_tables` gives each slot $b$ the map $\pi_b$ from
pencil column $f < F$ to a joined column, $o_S$ standing for a zero column.
The packed field is

$$
\hat X_b[:, f] =
\begin{cases}
X_{s,b}[:, j] & \pi_b(f) = o_s + j,\ 0 \le j < r_s,\\
0 & \pi_b(f) = o_S .
\end{cases}
\tag{S 4a}
$$

On the face the program that consumes the columns forms them itself: it joins
the $S$ panels and takes the slot's columns through the slab exchange
([dense linear algebra §2](dense_linear_algebra.md#2-layouts-and-the-moves-between-them)),
so the face programs take the panels and compile again for each panel count.
The q-local round forms them before its program instead. `pack_panels`
(`src/gw/shared_pole_local.py:286`) places one panel at a time on the batch
layout. The host inverts $\pi_b$ for panel $s$: $d_{s,b}(j)$ is the column $f$
with $\pi_b(f) = o_s + j$, or $F$ when the round does not take column $j$. A
source column appears at most once in a slot's table, so each packed column has
one writer. One program per ($r_s$, $F$) scatters a panel into one accumulator
per field (`_pack_place`, `:338`). The accumulator holds packed column $f$ of
slot $b$ as row $bF + f$, $[\text{slots}\cdot F, n_S]$ (`_pack_start`, `:321`), so
each panel column lands as one contiguous row on its parent's rank. A dropped
column is sent past the accumulator, never onto the next slot's first column.
The finish turns each rank's rows back into columns, one transpose of its
donated accumulator (`_pack_finish`, `:361`).

The point of the pack is that the q-local round program does not take the
panels. The panel count follows the line sites, which change from map to map
(Fe $4^3$ bispinor at P4: 104, 116 and 128 panels in maps 0–2, claim 4095). A
program that joins the panels inside itself has a new input structure for every
count and recompiles. The packed columns have the shape
$[\text{slots}, n_S, F]$ only. $F$ is held by the sector alone, not by its panel
count, so it carries across maps. A held $F$ past a round's own capacity is more
inert zero columns. The extra panels of a later map are extra calls of the same
place programs. The local CT takes packed columns too (`_pack_cross_spans`,
`src/gw/shared_pole_sectors.py:1095`). The sectors' $Q$ and $WQ$, the TC-on-C
output and the CT-on-T output and derivative are each placed at their own
sector's extent, so the local CT program is fixed by the two extents and spans.
The face CT pencil joins and takes the same panels in its program
(`_face_cross_pencil_equations`, `:1338`) and then runs the same equations
(`_cross_pencil_equations`, `:1356`).

**The face's block order.** Every pencil axis is a run of logical blocks: the
originals and the mirrors of the finite states, then the k0 and k1 infinity
directions; the w and v halves of the paired basis; the C and T spans of the CT
joint pencil. On a $p \times p$ face the axis is held in the tile-interleaved
order. With blocks $B_1, \dots, B_m$ of sizes $s_1, \dots, s_m$, each a multiple
of $p$, rank row (and column) $b$ holds

$$
\big(\, B_1^{(b)} \;\big|\; B_2^{(b)} \;\big|\; \cdots \;\big|\; B_m^{(b)} \,\big),
\qquad B_i^{(b)} = B_i\big[\, b\,s_i/p ,\ (b+1)\,s_i/p \,\big) .
\tag{S 4b}
$$

Four things follow. Joining or splitting logical blocks along an axis is every
rank's own concatenation or slice of its tiles, so no byte moves (`tile_join`,
`tile_split`, `tile_block`, `src/gw/shared_pole_pencil.py:82,90,120`). A state's
original and mirror columns sit on the same rank, so the paired-basis congruence
of stage 1 (theory §6.3, $w = (X(z) + X(-z))/2$, $v = (X(z) - X(-z))/2z$) runs
on each tile with that tile's pieces of the halves (`_paired_member`,
`src/gw/shared_pole_reduction.py:280`). Rows and columns share the order, so an
adjoint is the tile at the mirrored grid position: one `ppermute` across the
grid's diagonal (`tile_adjoint`, `src/gw/shared_pole_pencil.py:102`). With whole matrices ($p = 1$) S 4b is
the plain concatenation, so the q-local round runs the same equations. The
round's host tables enter the order once at the face entry points
(`interleave_tables`, `src/gw/shared_pole_pencil.py:143`, called by `face_reduce_round`, `face_reduce_decoupled`
and the face side of `_pack_cross_spans`); the replicated per-column vectors
(nodes, masks, inverse nodes) follow it (`join_vectors`, `split_vectors`). The
face span $Y$ therefore carries its rows in the face order, as the face CT takes
its columns. Every block is a multiple of $p$: the extent sits on the carrier
grain and the infinity width on a port carrier.

On the CPU census of the staged programs at CrI3 24×24 P64 shapes (TT side
25856, stage width 4; [dense linear algebra §2](dense_linear_algebra.md#what-gspmd-emits)),
the order takes TT stage 1 at 8×8 from 19.0 to 1.3 GB per device moved by GSPMD
and from 31394 to 16739 optimized operations, and leaves stage 3 at 1964
operations on every mesh (claim FACEMAP-4).

**GEMM stages.** Every face GEMM is `distrib_la.panel_matmul` (`face_matmul`,
`src/gw/shared_pole_execution.py:320`): a batched 2-D SUMMA with one exchange per
panel for every parent of the stage, $P_x \cdot 256$ contraction columns per
panel, transposed operands by one grid-transpose exchange. Face programs compile
with XLA's latency-hiding scheduler (`FACE_COMPILER_OPTIONS`, `:264`), which
runs the panel exchanges beside the local GEMMs. A stage runs over sub-batches of width

$$
w = \max\{\,N, \lceil N/2\rceil, \lceil N/4\rceil, \ldots, 1 :\
\text{stacks} + \text{program}(w) + \text{live} \le \mathcal B \,\}
\tag{S 5}
$$

(`stage_width`, `src/gw/shared_pole_execution.py:703`). The program price is the
scalar byte model tiled over the mesh, times two for the collective's staged copy
(`FACE_PROGRAM_COPIES`, `face_reduction_bytes`, `face_cross_bytes`). A short last
sub-batch repeats its last parent, so one shape compiles. The width moves no
number.

**Eigh stacks.** Each eigh runs once over the round's stack of $N$ matrices of
side $m$. It runs on route (c), every matrix whole on one rank, when

$$
\text{boundary} + \text{held}_S + 8\,m^2 \cdot 16\ \text{B} \cdot \lceil N/P\rceil
+ \text{live} \le \mathcal B ,
\tag{S 6}
$$

with $\text{held}_S$ the diagonal sector's selected panels (zero for CT), and on the whole mesh otherwise, with one `RuntimeWarning` (`staged_eigh`,
`src/gw/shared_pole_execution.py:682`). The $8m^2$ elements per matrix are
`BATCH_EIGH_TILES` (`services/distrib_la/src/distrib_la/plan.py:395`), priced by
`distrib_la.eigh_stack_bytes` (`services/distrib_la/src/distrib_la/workspace.py:306`).
The boundary is the stack live beside that eigh (`staged_sector_bytes`,
`staged_cross_bytes`, `src/gw/shared_pole_capacity.py:177,202`). Route (c) is
where the staged route's speed comes from: the $N$ eighs of a stack run
concurrently as local solves, where the mesh runs them one after another, each
over all $P$ ranks. An eigh whose result check fails reruns only that stack. The
stage programs hold no eigh.

The round line `Shared-pole sector constructor: round of parents a..b:` prints
each sector's side, stage width and eigh routes (`c` or `mesh`) whenever they
change.

### 5.3 Checks and output

The diagonal reductions refuse a parent that fails `orientation_paired`,
`gram_diagonal_positive`, `gram_valid`, `retained_metric_positive` or the zero-Ritz
policy (`GATE shared_pole_sector_*`, no repair). The held checks read one held
sample at a time, score it and release it. The treatment mask (S 3) is applied,
and each canonical parent is written once (`GATE shared_pole_sector_rounds`).

## 6 The store {#sector-store}

A map writes four ordered models, `CC`, `TT`, `CT_C` and `CT_T`
(`write_shared_pole_model`, schema `lorrax.shared-real-pole.v1`, recipe stamped
`raw-sector-endpoint-v1`). The factor dataset is `[q, μ, components, K]`, with 3
components for `TT` and `CT_T` and 1 otherwise. `CT_C` and `CT_T` share one pole
census. A manifest `sectors.json` (`lorrax.shared-real-pole-sectors.v1`,
representation `sector-ordered-ph`; `write_shared_pole_sector_manifest`,
`src/file_io/shared_pole_store.py:956`) binds the four models and the bank's
constant. Reuse of a published manifest on an interrupted SC map is
[shared-pole model §1](shared_pole_model.md#1-one-map), step 4.

**Device residence.** The four models stay on the devices for the same map's Σ
when their bytes $M$ at the retained-pole bound, plus one copy, fit half the
device budget beside the upstream stages, and the constructor's route is unchanged
with $M$ live (`_sector_model_residence`, `src/gw/shared_pole_sectors.py:448`;
`admit_resident_model`, `src/file_io/shared_pole_store.py:1633`). Otherwise they
go to files. Residence saves 5–8 % of a map on CrI3 6×6 at P4 (claim 3988).

**Static W for BSE.** `write_restart_tensors = true` stores
$W_0 = V + W_{c,CC}(0)$ from the CC model alone (`sector_static_wc`,
`src/gw/mpa/sector_sigma.py:886`), both branches of the parent pair at V's q
parents, unfolded on load by BSE ([BSE](bse.md)).

## 7 The Σ consumer {#sector-sigma}

`compute_sector_sigma` (`src/gw/mpa/sector_sigma.py:727`) is called once per Σ
evaluation. For one endpoint class the band-basis self-energy is

$$
\Sigma_{mn}(\mathbf k,\tau)
= \sum_{\mu\nu} \overline{\psi_{m\mathbf k}(\mu)}
\Big[\sum_{A,B} \tilde\gamma_A\, G(\tau)\, \tilde\gamma_B^\dagger \star W_{AB}(\tau)\Big](\mathbf k)_{\mu\nu}\,
\psi_{n\mathbf k}(\nu),
\tag{S 7}
$$

with $G(\tau)$ the four-spinor Green's function on the parent k grid,
$\tilde\gamma_A$ the signed spin permutation of channel $A$, and $\star$ the k-axis
convolution. The class's vertices form one product $A\text{-set}\times B\text{-set}$
(`gw.cohsex_sigma.lorentz_class_vertices`).

**Three windows.** The consumer integrates CC, TT and the mixed pair CT + TC, in
that order, each through the scalar frequency-quadrature executor
(`gw.mpa.sigma.compute_sigma_c_mpa_omega_grid`). The Σ rule set is planned once per
map on the union of the CC, TT and CT pole sets. The mixed pair is one fused window
(`fused_mixed_tau_factory`, `:456`): CT's W pair is synthesized once per τ node, TC's
is its transposed pair (one X↔Y exchange), and both contractions run in the same
loop trip (claim 3289). Each sector window runs one τ node per loop trip.

**Per τ node:**

1. **W(τ) on the irreducible q.** `sector_synthesis` (`:505`) reads the class's
   factors once per Σ call on the store's parent rows and never unfolds them. Each
   τ forms (S 1) and its partner through the scalar W(τ) owner
   (`synthesize_shared_pole_parents`, via `_w_program`, `:169`), in one panel of
   every parent and pole column. On a diagonal sector the partner is
   $W^{\mathsf T}$; on the mixed sector it is the same contraction on the
   conjugate factors. The hole branch
   (S 2) reads the same pair through q-negated load tables (`hole_tables`, `:149`),
   so no $-q$ gather and no conjugated weight is formed. No full-q W exists.
2. **Row passes.** `sector_node` (`:211`) runs each rank's $(\mu_X, \nu_Y)$ tile in
   passes of whole centroid orbits. The pass size comes from the fixed tile
   `runtime.tiles.TILE_BYTES` and the shapes (`gw.subtile_stream.plan_windows`), so
   every rank cuts the same passes. Per pass: the four-spinor parent Green of the
   pass's rows by one local active-range GEMM over the window's live bands; one
   $n_s = 4$ mode-8 Lorentz k-convolution that unfolds G and W on its load
   ([k-convolution](kconv.md#unfold-consumers)); the band projection into a
   rank-local partial. One band-block reduce-scatter ends the node.
3. **Band brackets.** With band extrapolation on, only the CC class splits its Green
   band sum into brackets, and all brackets reduce in one reduce-scatter. TT, CT, TC
   and the constant enter every bracket alike
   ([band extrapolation](../theory/band-extrapolation.md#four-current)).

**The constant.** $U - V$ is read and packed on its irreducible q, in parent-q
panels when the raw and packed copies exceed one tile, and contracted once with
the equal-time occupied projector (`instantaneous_sector_sigma`, `:648`). The bare
exchange Σ_x is its own owner's ([four-current wiring](four_current_wiring.md)).

**The Γ head.**

| `head_correction` | what Σ receives |
|---|---|
| `no_local_fields` (the default) | the direct Γ head, already inside the sector poles (§3); the consumer adds no head term. A store without it refuses (`GATE shared_pole_sector_head`, `src/gw/sigma_dispatch.py:885`). |
| `off` | no head; a debug setting |
| `full` | refused at configuration (`GATE full_shared_pole_head`): no wing/body local-field fold exists for this route |

**Factor placement.** One class is live at a time. Its factors are placed by the
scalar schedule with the class's extents $(n_A, n_B)$
(`_shared_pole_memory_schedule`, `src/gw/mpa/sigma.py:854`):

| `factor_layout` | placement | taken when |
|---|---|---|
| `axis` | pole columns replicated, each centroid endpoint on its own mesh axis; the per-τ GEMM is local | it needs no more panel passes than `face` |
| `local` | whole parents per rank in `distrib_la`'s batch layout; only W moves, one tile per node | `axis` does not fit, the deck has `linalg = local`, and whole parents fit |
| `face` | both factors on faces; every τ re-gathers the factor panels by SUMMA | otherwise |

Every reservation goes to the shared-pole capacity ledger. A compiled kernel is
reserved at its compiled peak plus its cuFFT plan scratch (`_admit_compiled`), and
it refuses only when that scratch cannot be measured (`GATE shared_pole_capacity`).

## 8 Worked byte budget: CrI3 24×24 at P64 {#sector-byte-budget}

The deck of claim 4083: CrI3 ferromagnet, 24×24 k grid, bispinor, P = 64 on
8×8 (16 nodes of A100-80GB), `memory_per_device_gb = 72`, so $\mathcal B = 72.0$ GB per rank, $n_q = 61$, $n_C = 3328$, $n_T = 1728$, so
$n_{CC} = 3328$, $n_{TT} = 5184$, $K_{CC} = 5991$, $K_{TT} = 9332$. The figures
below are the shape prices of §4–5 at this deck; the sides are those the map-0
selection produced.

**Bank.** The response stream's samples take 43.6 GiB per rank, more than half
the host budget, so they stream to one file per rank.

**Route.** The q-local prices are CC 71.1, TT 155.8 and CT 212.7 GB per rank
against 72.0. TT alone fails: a whole TT pencil at its conservative side
$R = 32000$ is $16R^2 = 16.4$ GB per matrix, held several times over in the
reduction. The map takes staged rounds. With $n_q = 61 \le 64$, (S 4) gives one
round of $N = 61$ parents.

**Sides and carriers.** $c_S = 8\,\mathrm{ladder}(\lceil K_S/8\rceil)$:
$c_{CC} = 6144$, $c_{TT} = 10240$.

| sector | side $R_S$ | stage width $w$ | stacks + held, GB/rank |
|---|---|---|---|
| TT | 25856 | 4 | 43.7 |
| CC | 19264 | 16 | 21.3 |
| CT | 17408 (joint $K$) | 4 | 25.4 |

**Eigh stacks.** $\lceil 61/64\rceil = 1$ matrix per rank, so (S 6) prices each
stack at $8m^2 \cdot 16$ B beside its boundary:

| eigh | side $m$ | route (c), GB | boundary, GB | sum, GB |
|---|---|---|---|---|
| CC $H'_{vv}$ | 9632 | 11.9 | 10.9 | 22.8 |
| CC Schur | 6144 | 4.8 | 7.3 | 12.1 |
| CC reduced | 12288 | 19.3 | 6.1 | 25.5 |
| TT $H'_{vv}$ | 12928 | 21.4 | 19.9 | 41.3 |
| TT Schur | 10240 | 13.4 | 19.6 | 33.1 |
| TT reduced | 20480 | 53.7 | 16.4 | 70.1 |
| CT metric | 17408 | 38.8 | 11.5 | 50.3 |
| CT Ritz | 17408 | 38.8 | 11.5 | 50.3 |

All eight run on route (c). The TT reduced eigh is the tightest: 70.1 GB before
the held panels and the upstream stages, against 72.0. A deck with a larger
$K_{TT}$, or a smaller card, sends that stack, and only that stack, to the whole
mesh with one warning. Every staged ledger row bounds its measured section peak
once the 1.2 GB/rank current-channel G-space stack upstream of the ledger is
counted (claim 3997).
