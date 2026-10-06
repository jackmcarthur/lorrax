# The shared-pole screened interaction: implementation

This page is how `sigma_w_model = shared_pole` is computed each map: the
response bank that samples $W_c$, the constructor that reduces the samples to
one real-pole model per parent $q$, the store, the Σ consumer, the gate rows and
the byte model. The model, its equations `(W n)` and why it is built this way
are [the shared-pole theory page](../theory/shared-pole-w-model.md); deck keys
are in the [input reference](../input_reference.md); rebuilding the model inside
a QSGW loop and retaining its quadrature are
[self-consistency](../self_consistency.md#shared-pole-w-with-retained-quadrature).

Notation: `q` is a raw parent (one irreducible $q$; `−q` is its own parent when
time reversal is broken); `n` is the packed centroid count
(`meta.mu_basis.n_packed`); `b` is the stored factor (dataset `factor`); faces
are `P(None,'x','y')` on the square X/Y mesh with `P = Px·Py`.

## 0 Where the code is

| stage | owner |
|---|---|
| deck surface (`sigma_w_model`, `sigma_w_accuracy`, `sigma_w_support_sites_ev`, `write_w`/`write_poles`) | `gw.gw_config._resolve_shared_pole_inputs` |
| recipe, support ladders, gate tables, capacity ledger, receipt | `gw.shared_pole_recipe` |
| per-map orchestration, bank residence | `gw.shared_pole_screening` |
| §2 response bank (samples, moments, Coulomb roots) | `gw.response_bank`; rules `minimax.response_group_rules` |
| §3 directions | `gw.shared_pole_directions` |
| §4 pencils | `gw.shared_pole_pencil` |
| §4 reduction, paired basis and cut | `gw.shared_pole_reduction` |
| §4 dedupe and measured model checks | `gw.shared_pole_gates` |
| constructor chain, route admission | `gw.shared_pole_constructor` |
| §6 parent rounds | `gw.shared_pole_local` |
| §6 whole-mesh (face) execution adapters | `gw.shared_pole_execution` |
| §5 photon CC/TT/CT sectors | `gw.shared_pole_sectors` |
| §10 byte model | `gw.shared_pole_capacity` |
| §7 store | `file_io.shared_pole_store` |
| §8 Σ consumer | `gw.mpa.sigma` |
| §8 Γ head coupled to the current body | `gw.shared_pole_head` |

## 1 One map

1. **Recipe.** `resolve_shared_pole_recipe` reads the current census
   (occupations, $\mu$, gap, active charge, cell volume) and places the supports
   (theory §5.1). It fixes `direction_cutoff`, `imaginary_width`
   $=\lceil n/4\rceil$, `infinity_width` $=\lceil n/8\rceil$,
   `line_direction_cap` $=\lceil n/16\rceil$, `pole_budget`
   $=\lceil1.8n\rceil$ and `bank_rule_tolerance` $10^{-8}$ for `production`;
   `relaxed` uses 8 uniform line sites, 2 imaginary sites, cutoff $10^{-2}$,
   no cap, no budget, and bank tolerance $10^{-7}$. The Σ quadrature reads
   `sigma_quadrature_eps` from the deck as every Σ route does; `relaxed` only
   defaults an omitted key to $5\cdot10^{-4}$.
2. **Bank** (§2): Coulomb roots, samples $W_c,\partial_sW_c$ at the supports
   on the imaginary axis and the held supports, the direction panels of every
   fitted line support (§3), moments $M_1,M_3$ (and $M_0,M_2$ when ordered)
   → `bank.h5`.
3. **Residence.** The bank is written frequency-major and read parent-major. It
   stays on the devices when the payload and one read copy fit half the device
   budget *and* the constructor route is unchanged with it live. Otherwise it
   goes to SlabIO's per-rank streamed tier ([SlabIO](slab_io.md#streamed-tier);
   q-major tiles, so one sample's q span is one contiguous run, and an
   unwritten tile reads as zeros): host memory if the payload fits half the
   host budget, else one file per rank. The tier is released once the
   constructor has committed. The shared scratch file `bank.h5` is used only
   for `write_w` (which re-reads the bank), a distributed `linalg`, and a
   per-rank store the disk or quota refuses at initialization
   (`gw.shared_pole_screening`). A field created later (line panels, photon
   contact fields) that the disk refuses is held in host memory, with one
   `RuntimeWarning` per bank (`ResidentBankPayload`); a scalar `bank.h5` is unlinked
   once the constructor has committed `model.h5` (kept for `write_w`; the
   photon Σ reads its constant), and the receipt records the bytes and the
   link count. Either way a run holds at most one bank. Its size is
   $N_q\,(2N_\text{dense}+N_\text{moments})\,N_\mu^2\cdot 16$ B plus the line
   panels (§10): 1.26e12 B at Fe 20³ (1062 parents, 1796 centroids, 8 dense
   samples), about 17× the model at $K_\max$ 2400.
4. **Constructor** (§3–§6) → `model.h5`, one parent round at a time. An SC
   map keeps the model on the devices instead (`ResidentSectorModel`, the
   photon sectors' carrier) when the model at its stored column bound and one
   copy fit half the device budget (`shared_pole_store.admit_resident_model`,
   the sectors' rule); the head, Σ and, on the accepted final map, the
   `W0_qmunu` persist read it there, and the next map's entry or the end of
   the loop releases it. Reads are the file's bytes; Σ and W0 match the file
   route bit for bit where their panel schedule is unchanged (the resident
   stage narrows their budget by the model's bytes). `write_w`, `write_poles`,
   a one-shot (restart member), a refused admission and an SC map whose bank
   is on the file tier write `model.h5`. A rerun of an interrupted SC map
   reuses a committed `model.h5` whose header binds the current identity
   (validated collectively, payload digest included) and rebuilds nothing
   upstream; with no committed model it resumes the constructor from a
   retained file bank, else rebuilds that map. An SC identity binds the source
   WFN's fingerprint (the dipole provenance's), since the SC state labels do
   not name it, so no reuse authenticates on energies alone. A committed
   `model.h5` of another identity, or at a one-shot label, still refuses. A photon
   map's published `sectors.json` is reused when it binds the current identity
   (recipe_hash included) and every model and constant it names is a file in
   that directory; sector models that were resident died with the run, so the
   map rebuilds, and the report names every removed file in one WARNING line;
   a manifest of another identity refuses (`GATE shared_pole_output`).
5. **Σ** (§8) synthesizes $W_c(\tau)$ from the factors.

Every map rebuilds samples, directions, poles and ranks from the current
state; only quadrature rules and support geometry are retained across SC maps.
No χ bank, W bank or streamed store survives its map.

## 2 The response bank

**Coulomb roots.** $V$ is frequency independent: its roots `H[q,μ_X,ν_Y]` are
solved once for all irreducible parents with the batched solver and held
through the frequency loop.

**Shared-node rule.** All samples (imaginary axis by $\operatorname{Im}z$, then
the line by $\operatorname{Re}z$) get one set of complex times $t$ from
`minimax.response_group_rules`, fitted on every sample at once. Every node is
**one Green-pair evaluation** $A(t)$ that serves every member's value and
$\partial_s$ derivative in both orientations:

$$
\frac1{d-z}\simeq\sum_j c^F_j e^{-(d-r)t_j},\qquad
\frac1{d+z}\simeq\sum_j c^B_j e^{-(d-r)\bar t_j},\qquad
\frac{\partial}{\partial s}\frac1{d\mp z}=\pm\frac1{2z(d\mp z)^2}.
\tag{SP 1}
$$

The forward product is $A(t)=G_u(t)\,\overline{G_f(\bar t)}$; the reverse is
$\overline{A(\bar t)}$ — the same damping with the orbital product reversed, not
$\overline{A(t)}$. On a raw-parent plan the charge stream forms $A(t)$ with
mathdx mode 11 from the two parent Greens, metals included
([fractional χ₀ response face](fractional_chi0_response_face.md)).
The nodes come from a stacked Hankel shift pencil; a sample set
whose shared fit fails is split in halves down to single samples. The rule never
depends on the memory budget: the evaluation runs in groups of at most the group
size, and each group streams its rule's whole node set for its own members. The
group size is the largest whose donated carry `[2·members, q, μ_X, ν_Y]` and
compiled stream temporaries fit the map ledger and the device room (the budget
less the bytes actually live): every sample in one group on symmetric decks.
When the samples do not fit one group, the stream runs once with every
sample and its carry streams to SlabIO's per-rank tier
([memory model](memory-model.md#streamed-chi-bank)); groups remain only
when the disk refuses the bank and on the full-k Green route. A
smaller group costs ⌈members/group⌉ passes over the same nodes and gives the same
eqp. (Rules fitted per group gave eqp up to 14.6 meV off a 100× tighter
reference on Na 8³ at group 1, against 0.21 meV for the all-sample rule.) The chosen group's executable is
then checked before it runs ([memory model](memory-model.md#the-compiled-check)). The accuracies are
sampled, not continuum certificates; a matched QP comparison is the acceptance
check.

**Domain and occupations.** The transition interval spans every nonzero
occupied/empty weight pair on the one branch support
(`gw.efermi.band_in_occupation_window`, $|w|\ge 10^{-5}$; samples only),
including negative transition energies and signed metallic weights; the exact
moments keep every weight. The occupation envelope
$|f_nu_m|\le A_fA_u\min(1,e^{\beta(E_m-E_n)})$, with $A=\max(1,\max|\cdot|)$ and
$\beta$ the minimum one-sided log slope of $|f|$ above $\mu$ and $|u|$ below,
bounds the fit: the tolerance is divided by $A_fA_u$ and nodes satisfy
$0\le\operatorname{Re}t\le\beta$. For $\beta>0$ the scalar reference is 0 and the
Green references are $\mu$; $\beta=0$ keeps the uniform fit with reference at the
interval bottom. No Fermi–Dirac form is assumed. In an SC session the interval
is padded by 4 eV (2 eV per one-particle endpoint) and a rule is reused while
the current interval, $\beta$ and amplitude stay inside the certified ones.

**Orientation.** Σ's $G_{k-q}W_q$ contraction needs

$$ \mathcal F_q[f](\mu,\nu) = \sum_R f(r_\mu,\, r_\nu + R)\, e^{i q\cdot R}, \qquad W_q = \mathcal F_q[W] . \tag{SP 2} $$

The TRS stream builds $\overline{G(w,t)}=G(\bar w,\bar t)^{\mathsf T}$ and
returns $\mathcal F_q[\chi^{\mathsf T}]$, equal to $\mathcal F_q[\chi]$ under
time reversal. An ordered bank (`ordered=True` in `response_bank.response_stream`)
builds $G(\bar w,\bar t)$ and gathers rows at $-q$, which is
$\mathcal F_q[\chi]$; the transposed orientation on a TR-broken deck would hand
each Green branch the other branch's residues,
$\Sigma[W^{\rm even}]-\Sigma^{\rm odd}$.

**Minus-q partner.** The ordered pencil's state $X(-z)$ acts with
$W_q(-\bar z)=\overline{W_{-q}(z)}$ on the directions of $X(z)$, and it must be
the same operator as $W_q(z)$. On an imaginary node $-\bar z=z$, so the sample
is its own partner. At each fitted line support off the imaginary axis the
producer solves it beside $W_q(z)$: the stream's output rows are the union of
the parent rows and their $-q$ rows in one panel, and the partner is formed
from the conjugated $-q$ rows with the **original parent's $V$** (and contact).
Where the plan holds a spatial inversion row that `SymMaps.active_symmetry_rows`
authorizes as unitary, with a complete centroid map, $\chi_{-q}$ comes from
that row and the $-q$ rows are not streamed (`response_bank._unitary_inversion`).
Where QE types inversion only with time reversal (a PT-symmetric
antiferromagnet), inversion alone is no symmetry and the $-q$ rows are
streamed. It is consumed by the line selection (§3) and never stored. Rebuilding it from
the $-q$ parent through the symmetry tables is not exact: the ISDF $V_q$ is
covariant only to about $2\times10^{-6}$, and the ordered Gram amplifies that
into a refusal (sandbox claim 2452, bcc Fe).

**Dyson and derivative.** Per sample and bounded $q$ span, $W_c$ is solved from
the roots and $\partial_sW=W(\partial_s\chi)W$ is formed from the committed $W$,
without a second solve. One collective writer transaction serves a group; the
per-field write masks let a value commit before its derivative, and a restart
skips committed fields. Charge stores $W-V$; photon stores $W-W_\infty$ with
the constant separate.

**Moments.** `exact_bare_moments` forms the band-sum coefficients and
`response_algebra.moments` applies (W 12): $M_1=C_2/2$, $M_3=C_4/2$, and for an
ordered bank $M_0,M_2$. The Coulomb prefactor and orthonormal FFT together
scale them by $1/N_k$.

**Cost.** Green-pair evaluations $\approx\sum_{\rm passes}(\text{rule nodes})$, each
one flat-k FFT convolution producing all members; Dyson is one batched solve per
sample and parent span.

## 3 Directions

`gw.shared_pole_directions` selects $Q_a$ per fitted support through
`distrib_la` (theory §5.1): right singular vectors of the line sample above
`direction_cutoff` (at most `line_direction_cap`, whole multiplets within
$10^{-6}$), leading eigenvectors of $-\operatorname{Herm}W_c(iu)$ to
`imaginary_width`, and of $M_1$ to `infinity_width`, with the rank floor at
$n\epsilon_{64}$. Each state is `(node, Q, O = W Q, D = W' Q)`. Direction ranks
sit on the padded-extent ladder (`runtime.padding.ladder_extent`) so selection
and round programs repeat across rounds and maps.

**Line supports are selected by the producer.** A fitted line support with
$\operatorname{Re}z\ne0$ reads nothing but its own sample: $Q$ is cut from the
singular spectrum of $W_q(z_a)$ alone (the multiplet closure included), and
every state the pencil takes from it acts on $Q$ or on $O=WQ$ of the same
sample,

$$
\begin{aligned}
&X(z):\ (Q,\ WQ,\ W'Q), &&X(\bar z):\ (O,\ W^\dagger O,\ W'^\dagger O),\\
&X(-z):\ (Q,\ W_m^\dagger Q,\ W_m'^\dagger Q), &&X(-\bar z):\ (O,\ W_mO,\ W_m'O),
\end{aligned}
\qquad W_m=W_q(-\bar z),
\tag{SP 7}
$$

(the mirrors on an ordered bank; derivatives in $z$ ordered, in $s$ TRS), and on
a photon bank the other family's rows of the same products, which are the
CT/TC actions. The Gram cut, the pole budget, the round extent and the CT
joint span act only on these panels. So the producer selects while
$W_q(z_a)$, $\partial_sW_q$ and $W_m$ are in hand (`LineSelection`, whole
parents per rank when their blocks and the $n\times n$ eigensystem fit, else the
face) and the bank stores the panels (§7). The constructor reads them
(`line_panel_states`) and selects the supports on the imaginary axis from their
dense samples per round. On the local route both sides run the same one-parent
eigensolve of $W^\dagger W$ on the same equations, so $Q$ and every count are the ones
the constructor would select, to round-off; the action GEMMs run at a different panel width
and agree to round-off.

## 4 Pencils and reduction

`gw.shared_pole_pencil` assembles (W 18)–(W 19) for a TRS bank and
(W 25)–(W 26) for an ordered bank, from `Q`, `O`, `D` and the moments only;
blocks come out `P(None,'x','y')` on either route.
`gw.shared_pole_reduction` applies
(W 20): diagonal equilibration, keep cut `normalized_gram_keep` and the pole
budget (the largest `pole_budget` directions, less an edge member tied to its
neighbour below the cut within `multiplet_relative_tolerance` or the eigh's
$R u \gamma_{\max}$, so a degenerate multiplet is never split), the coupled
Newton–Schulz inverse root (iteration count fixed from the initial
infinity-norm bound, never from an on-device residual), and one Hermitian
`eigh`. The ordered route applies the paired basis (W 28), the cut on the
$v$-block with keep $10^{-7}$, a second relative cut on the restricted pencil,
and the signed model (W 27); poles with $|\mu|\le$ keep$\cdot\max|\mu|$ are at
infinity and their output weight is reported (`infinite_weight_ok`). Dedupe
(W 29) is `gw.shared_pole_gates`.

## 5 Photon sectors

`bispinor_gw = full_shared_pole` builds CC, TT and CT stores from one photon
bank (`compute_photon_bank`) holding $W-W_\infty$, its ordered moments and the
constant $W_\infty-V$. CC uses $n_C$ rows, TT $3n_T$ rows (Cartesian component
minor), each with its own directions, reduction and $\lceil1.8n\rceil$ budget;
CT_C and CT_T share one retained mask and pole ordering on the joint span, each
endpoint passing its own lost-weight check. CC/TT spans retained for CT use the
ordered `sector_threshold` $10^{-5}$ in both paired Gram cuts. The stability
gate is positive retained $\mathcal H$; the scalar passivity bound does not
apply to the signed photon $V$ (theory §9). Stores stamp
`raw-sector-endpoint-v1`; a manifest (`lorrax.shared-real-pole-sectors.v1`,
`sector-ordered-ph`) binds the four stores and the constant.

**Treatment ceiling.** Bispinor sector models deactivate poles above

$$
\Omega_{\rm treat}=2\Big[\max E_{{\rm cond},\chi}-\min E_{\rm val}\Big],
\tag{SP 3}
$$

a numerical treatment, not a bound on collective modes. An SC run holds the
first map's value while the current span stays inside it and re-plans it at
twice the current span when the span exceeds it. Inactive modes have zero factors and the inert pole sentinel; CC and
TT have independent masks, CT one common mask. Refusals:
`GATE shared_pole_sector_treatment_{census,order,empty}`. Held rows score the
untreated fit; accuracy of the treatment is a projected-Σ comparison.

## 6 Execution

**Route.** `constructor_route` admits one route before the first bank read.
**Local parent rounds** — one whole parent per rank, batch layout
`P(('x','y'), ...)` — whenever the complete selection stack and the
conservative reduction pencil (`constructor_side_upper_bound`) fit the
device, whatever `linalg` names: `gw.shared_pole_local.round_program` packs each parent's
panels to the round extent (`round_tables`; ordered originals and mirrors as two
halves of one extent), assembles and reduces its pencil with local dense kernels
and sorts its poles; synthetic slots are skipped. Otherwise the **face route**
runs a parent batch on the complete mesh with `distrib_la` GEMM/`eigh`, both
matrix axes distributed and only spectra and masks replicated. An ordered
face reduction solves its kept span on `face_ritz_carrier` (the pole budget
with its per-rank tile on the extent ladder; the whole side on the `relaxed`
tier, which has no budget), as a local round does on its
Ritz carrier: the Schur and final eighs run at the carrier and twice it
(CrI3 24×24 on 4×4: 6144 and 12288 instead of 9152 and 18304). Its eigh stacks
(infinity, directions, partners, passivity, the Gram reduction, CT and the
Cauchy check) go to `distrib_la` with the room beside the admitted batch
(`face_eigh_room` of the selection admission row; a scalar reduction round
and a face model check beside its own whole-chain program compiled on the
arrays it runs, which is also its retry; CC, TT and CT beside the sector batch
row, which holds the largest of their whole-chain programs and the resident
sector models); the service runs a stack one or more whole matrices
per rank when one slice's compiled program fits that room, else on the whole
mesh ([eigh stacks](../services/distrib_la/api.md#eigh-stack)). The sectors
take one route together (`sector_execution`, against the same ledger): CC, TT
and CT all run on the face when the CT joint pencil or any sector prices face,
else all local. With at least as many parents as ranks the joint pencil's
conservative price (both spans at twice their pole budgets) does not send the
sectors to the face: each rank builds whole q-local models, and the round
admits the CT pencil at its actual retained spans. A local round whose CT does
not fit there warns and reruns its parents, CC and TT included, as face
batches (the slow fallback). The route is fixed before any read and the report
line `Shared-pole constructor:` names it and its prices; that fallback is the
only route change inside a stage. A face program holds its eighs' first attempts; a failed check
reruns that round's whole-chain program on the whole mesh, the program its
admission compiled (`distrib_la.checked_program`).

**Reindexing.** Matrix selection, factor sorting and unequal CT block assembly
use `common.staged_reshard`: exchange to slabs split over all ranks, select or
concatenate locally, exchange back to the face. Constraining the result of a
global take or concatenate is not enough: GSPMD can gather a large operand onto
one mesh axis.

**Checks and output.** The model checks (passivity, held $W$ and
$\partial_sW$, moments) run on the same layout (`round_checks`); receipts
report each parent at its own extent, dropping round padding from the Gram
spectrum (`own_extent_receipts`). The store receives one write of every parent
in canonical order (`canonical_factors`).

**Strong-scaling limit.** Local rounds hold a whole parent's $[R,R]$ pencil
per rank, so the per-rank peak does not fall with $P$; the face route is the
path past that point (CrI3 24×24, $n = 3328$: the local pencil needs 64 GB per
rank, so 40 GB cards take the face route at any $P$).

**Infinity directions** come from the Hermitian part of the exact $M_1$
(`shared_pole_directions.infinity_directions`), as the imaginary supports read
$-\mathrm{Herm}\,W$: the Dyson chain forming $M_1$ from bare moments
Hermitian to $4\times10^{-16}$ leaves an anti-Hermitian part up to
$1.0\times10^{-12}$ of $\max|M_1|$ at $n = 3328$, at the checked eigh's
$10^{-12}$. The bare moments are checked where they are produced
(`GATE response_moment_hermiticity`).

## 7 Store schema

`model.h5` (`lorrax.shared-real-pole.v1`) and `bank.h5`
(`lorrax.shared-real-pole-bank.v3`) share one authenticated header
(`file_io.shared_pole_store`); bulk payloads cross SlabIO only.

| field | meaning |
|---|---|
| `identity` | current-map hamiltonian, energies, occupations, wavefunctions, centroids |
| `recipe`, `recipe_hash` | the resolved recipe (support override and sector treatment fold into the hash); restart and SC maps refuse a stale one |
| `representation` | model: `scalar-trs-even-s` (W 14) or `scalar-ordered-ph` (W 16); ordered bank: `charge-ordered-z` |
| `n_q_irr`, `q_irr_full_idx`, `qirr`, `operations` | raw parents and the authorized symmetry rows |
| `n_mu_logical`, `nspinor`, `centroid_digest` | basis identity; the operator is the spin-traced $\mu\times\mu$ charge response for `nspinor` 1, 2 and the four-component kinetic-balance carrier |
| model: `factor [q, μ, components, Kmax]` complex128, `poles2_ry2 [q, Kmax]` float64, `K [q]` int64 | (W 14) per parent, units Ry$^{3/2}$ and Ry²; inactive columns have $b=0$, $\Lambda=1$ Ry² |
| bank: `Wc`, `dWc_ds [q, a, μ, μ]` at the dense samples (every id outside the line span $[p_0,p_1)$, rows $a$ skipping it); `M1`, `M3` (+ `M0`, `M2` when ordered); `constant` for photon | dense samples and moments of §2 |
| bank: `line_<family>_<sid> [q, 1+2S, rows, r]`, photon `…_cross [q, 2S, other rows, r]`; header `line_panels` (span, rows, widths, counts), mask `line_written [q, p₁−p₀]` | one line support's $Q$, then output and action per state of (SP 7), $S=2$ TRS, 4 ordered; `charge` rows in the canonical carrier, photon `C`/`T` rows in the packed sector order; $r$ is the carrier of the sample's widest count |

A bank of a retired schema (v1, v2: dense line samples) is refused by name, and
constructor resume then rebuilds it. `write_poles = true` exports $(b,\Lambda)$.
A run with `write_restart_tensors = true` persists `W0_qmunu` $=V+W_c(0)$
for BSE: `gw.mpa.sigma.shared_pole_static_wc` runs the §8 synthesis with the
$\omega=0$ coefficient $-1/(2\Lambda)$ and adds the hole branch
$W_+(-q,0)^{\mathsf T}$ (ordered) or $W_+$ (TRS), i.e. $-b\Lambda^{-1}b^\dagger$
on a TRS store. A one-shot stores the resolver's $\omega=0$ head; a
self-consistent run evaluates the accepted final map's model once, after the
loop, from the devices (step 4) or that map's retained scratch generation, and
stores the map's iteration head at $\omega=0$. The four-current sector bank
stores $V+W_{c,CC}(0)$ of its CC sector (`sector_sigma.sector_static_wc`,
both branches of the parent pair through the W tables). Both evaluators run at
the q parents of the run's V wedge, and the restart stores those parents with
their unfold tables: no full-q W is formed for the file, and BSE unfolds on
load ([BSE](bse.md)). Plain-MPA and metal restarts carry no `W0_qmunu`
([BSE](bse.md) says what a BSE on them does).

The same key keeps a one-shot's swept Σ(ω) (`file_io.sigma_checkpoint`,
`tmp/sigma_checkpoint_oneshot.h5`) when it pays for itself: only if the
measured sweep took at least 5× the predicted write (cube bytes at 1.3 GB/s,
measured on one node), logged as one `Sigma checkpoint: written|skipped` line
(Na 8³ [−100, +150] eV skips: 101 s sweep, 61.4 GB, 47.7 s write). The write
finishes before finalize, which donates the body cube. It holds the body
cube (and, when present, the odd cube and the raw twin's N₃ band-diagonal
slots) through SlabIO from their own shards, then the head
diagonal, band-extrapolation payload and band axis, then a commit digest. A
rerun with `restart = true` whose identity matches (the W model's digest and
identity, the energies Σ is read at, the ω grid, every sweep option, and
the band-extrapolation bracket plan, `gw.sigma_dispatch`) goes
straight to finalize; any other file is removed, named in one WARNING line,
and the sweep recomputed. It covers one-shots only: an SC rerun starts at
map 0, retention deletes later maps, and map 0's sweep plans the Σ windows
the later maps hold, so SC maps, map 0 included, recompute. A failed write
never costs the run: the partial file is removed with one WARNING line and
finalize proceeds. The PPM route has no W digest at this seam and always
recomputes.

`write_w = true` dumps the bank as stored (line supports as panels) and is a
debug output.

## 8 The Σ consumer

Σ contracts $G$ with $W(\tau)$ synthesized from the factors inside the Σ
window executable (`gw.mpa.sigma._shared_pole_w_synthesis`, bound to the τ body
by `gw.mpa.sigma.SynthesisTau`, the same pair the photon sectors use):

$$ W_{c,+}(q,\tau) = b\,\mathrm{diag}\big(d_j(\tau)\big)\,b^{\dagger}, \qquad
d_j(\tau) = \frac{e^{-i(\Omega_j - E_{\rm ref})\tau}}{2\Omega_j} . \tag{SP 4} $$

Synthesis uses the Green-function GEMM (`build_G`) at the irreducible parents,
then the little-group realization and the fixed-q projection, all on the
parent rows. The result is the parent pair $(W_p, W_p^{\mathsf T})$,
$[n_{q,\rm irr},\mu,\mu]$; no full-q $W$ is formed (TASTE 97). The Σ kconv call
unfolds the pair on its transform's load through the store's q-wedge tables
(`_shared_pole_q_wedge`, pair-transpose rule, mathdx mode 9), as the GN-PPM
wedge does, one row pass of whole centroid orbits at a time: the pass's rows of
the pair enter mode 9 with the load tables cut to the pass on the device
(`subtile_stream.window_load`), and the pass's parent Green comes from
band-complete ψ by one local GEMM (`ppm_tau_kernel._sigma_subtile_kernel`).
Mode 7 reads the Green's unfold tables placed once per run
(`ppm_tau_kernel.sigma_kconv_tables`, through `symmetry_maps.device_load_tables`)
and cut to the pass the same way, so no window program holds table
constants. The passes are equal windows run as one `lax.scan`
([memory model](memory-model.md#the-green-side-stages)). An endpoint map
that crosses a mesh shard refuses (`GATE shared_pole_w_parent_local`). The factors are read once per Σ call and stay resident:
$32\,n_{q,\rm irr}\,\mu\,\bar K/P$ bytes per rank face-sharded,
`P(None,'x',None,'y')`, or $16\,n_{q,\rm irr}\,\mu\,\bar K(1/P_x+1/P_y)$ when the
panel search admits the replicated pole columns ($\bar K$ the store's pole
carrier). Where those do not fit, on a `linalg = local` deck whose whole parents fit per rank
(`shared_pole_execution.whole_parent_execution`, the bank's rule), one copy is
held instead, whole parents per rank in `distrib_la`'s batch layout
($16\lceil n_{q,\rm irr}/P\rceil\mu\bar K$ bytes): each rank contracts its own
parents over their live pole columns and only $W(\tau)$ moves, batch to face
(`distrib_la.batch_gram`). The face SUMMA re-gathers $2(\mu/p)\bar K$ per
parent at every τ node: 18.6 GB per rank per node at Ni 20³ P64, against
the 0.85 GB tile. The budget left
beside them sizes one parent panel × pole-column chunk of synthesis workspace:
parent panels are a static loop inside the executable, chunks of one static
width a device loop, one of each when everything fits. A store whose resident
factors do not fit runs at one parent and one column multiple, with one
`memory over budget` warning line. The synthesized $W$ always uses both mesh axes.
The photon sectors (`mpa.sector_sigma.sector_synthesis`) synthesize through the
same owner (`mpa.sigma.synthesize_shared_pole_parents`) and placement rule, at
their tile extents $(m n_A, n n_B)$ and in one panel of every parent and pole
column. On a diagonal sector (CC, TT) the partner is $W^{\mathsf T}$; a mixed
sector's partner $\bar B_A d B_B^{\mathsf T}$ comes from the same operands (two
factors in `distrib_la.batch_gram`, the face route's one panel exchange).

**Two τ nodes per loop trip.** On GPU the scalar Σ τ window evaluates two τ
nodes per trip of its device loop, so one node's W(τ) synthesis and
exchanges run beside the other node's k-convolutions. Only that program is
compiled with XLA's latency-hiding scheduler
(`gw.ppm_accumulators.WINDOW_OVERLAP`, through `jax.jit(compiler_options=)`);
the process-wide flag stays off. With one node per trip the scheduler gains
nothing (+0.3 %). The window pairs only when its compiled paired executable
fits the device budget beside the live stages
(`gw.mpa.sigma.SynthesisTau.fits`: the largest executable peak any rank
read, against the capacity ledger, so every rank decides alike; the ledger
reservation takes the same figure, `gw.mpa.sigma._admit`); otherwise it runs one node per
trip with the default schedule. The paired program holds a second node's
live set (Ni 20³ P64: 9.76 → 16.95 GB compiled per rank), and its window
line in gwjax.out's memory table says "two nodes per trip". Every
counter-indexed read in its loops sits behind an optimization barrier and
rematerialization is off for every program, so the R82 hazard (a rematerialized slice
read after the loop counter's in-place increment) cannot arise there.
At the Ni 20³ P64 tile a node takes 0.549 → 0.492 s (claim 3115). The
photon sectors run one node per trip.

**Hole routing.** Conduction windows take $W_+(q)$. An ordered store routes
valence windows to the particle–hole partner,

$$ W_-(q,\tau) = W_+(-q,\tau)^{\mathsf T}, \tag{SP 5} $$

TRS stores never take this branch.
The valence branch reads the same parent pair through the q-negated load
tables (`sector_sigma.hole_tables`: the particle tables at $-q$, the partner
flag flipped, both phases conjugated), so no $-q$ gather or transpose of $W$
is formed. The little-group projector always forms its transposed output the same way,
as the average of the swapped pair, in the same loop step; it does no transpose
exchange, whatever the group order (large groups pay local work instead: Na 8³
Σ τ +7 % at P4). One tile exchange per τ node remains: the synthesis
$W_p^{\mathsf T}$. Both local forms equal the exchanged transposes bit for bit
(claim 2958). An exchange lets the
off-diagonal ranks move their tiles while the diagonal ranks copy theirs and
wait; that cost 26 % of a P16 node. The static $W(0)$ restart member
unfolds the parent pair only at V's q parents and at their $-q$ rows
(`_shared_pole_at_rows`); no full-q W is formed.

**Two-component decks.** $W$ is spin-scalar; $G$ carries the spinor axes and the
τ kernel broadcasts $W_q$ over both (`ppm_tau_kernel` `prep_w`).

**Band brackets.** With `use_band_extrapolation` (on by default) the scalar
consumer splits the Green band sum into the three brackets of
[band extrapolation](../theory/band-extrapolation.md) inside the same window
executable and applies the pooled fit. A sector (bispinor) consumer splits
the CC class's Green band sum the same way and adds TT, CT and TC to every
count ([four-current Σ](../theory/band-extrapolation.md#four-current)).

**Γ head** (`gw.shared_pole_head`). `head_correction = full` evaluates the
current TRS body at Γ one frequency at a time, folds the common head wings
through total $W$ and fits the scalar head with the MPA head owner; its
vertices trace the spinor index and its capacity is $2/(n_{\rm spin}n_{\rm spinor})$
states per band. It refuses on an ordered store
(`GATE shared_pole_head_ordered`: the Γ body evaluator is the even form) and on
the four-component charge store (`GATE shared_pole_head_nspinor`). An ordered
store carries `head_correction = no_local_fields`: the direct tensor $S(\omega)$,
finalized with no Γ body ([four-current heads](../theory/four-current-head-corrections.md)). Metal head routes are
[self-consistency](../self_consistency.md#metals-direct-drude-head).

## 9 Gates and tests

Every construction receipt row carries version, value, threshold and
PASS/FAIL/WARN/NOT_MEASURED. The TRS table is `shared_real_pole_gates_v1_r3b`;
`shared_real_pole_gates_ordered_v1` replaces the rows marked "ordered"; the
four-component charge store has its own `representation` row
(`shared_real_pole_charge4_v1`).

| gate | certifies | refuses? |
|---|---|---|
| `representation` | TRS: `nspinor` 1/2 and TRS allowed; ordered: TRS broken and an ordered bank | yes |
| `normalized_gram_validity` | equilibrated Gram (or $\mathcal H'_{vv}$) spectrum $\gamma_{\min}/\gamma_{\max}\ge-10^{-7}$ | yes |
| `normalized_gram_keep` | retained rank at the cut ($10^{-8}$; ordered $10^{-7}$, sector spans $10^{-5}$) | diagnostic |
| `retained_subspace_moments` | TRS: projected $M_1,M_3$ identity to $10^{-10}$; ordered: $m_0..m_3$ on the infinity directions | TRS yes; ordered diagnostic |
| `zero_ritz_policy` | $\lambda\le10^{-6}$ Ry² dropped within $10^{-6}$ factor weight; ordered also `infinite_weight_ok` | yes |
| `finite_factors_poles` | finite `b`, positive finite active $\Lambda$, exact inert sentinels | yes |
| `passivity` | V-whitened $-\operatorname{Herm}W_c(i\eta)$ in $[0,I]$ (W 9); ordered reports the anti-Hermitian part | yes |
| `model_reciprocity` | TRS only: transpose symmetry at transpose-symmetric held samples | yes (TRS) |
| `held_w` | held $W$, $\partial_sW$ relative errors | diagnostic |
| `full_m1_defect`, `full_m3_defect` | full-matrix moment defects | WARN only |
| `capacity` | aggregate live bytes within the device budget; WARN above the $3U$ scaling target, $U=16N_q(n_{\rm spinor}n_\mu)^2/P$ | above budget |
| `rule_validity`, `sc_rebuild` | bank and Σ certificates cover the current domains; SC rebuilds from current state | yes |

## 10 Byte model

Per rank, for parent batch $b$, sample batch $a$, pencil side $R$
(`shared_pole_capacity.shared_pole_byte_terms`; `ConstructorCapacity` turns the
terms into ledger rows):

$$
\text{bytes}\approx
\underbrace{16\,b\,s_f\,n^2/P}_{\text{sample/moment faces}}
+\underbrace{16\,b\,3nR/P}_{\text{narrow actions}}
+\underbrace{8b(12R+4n)}_{\text{replicated scalars}}
+\underbrace{16\,c_b\,D}_{\text{dense temporaries}}
+\text{native eigh workspace},
\tag{SP 6}
$$

with $c_b=\lceil b/P\rceil$ on local rounds ($b/P$ on the face route) and, by
phase: selection $D=24n^2$, $s_f=2a$ over the $a$ dense fitted samples plus the
moment faces and the line panels in face units,
$N_{\rm line}(1+2S)\,n\,r_{\rm cap}/n^2$ (`selection_face_count`);
reduction $D=14R^2+12nR$, $s_f=0$ (on the face route $16\,c_b D$ at the conservative side, an ordered round's kept span on `face_ritz_carrier`, admits a batch, priced from these terms and never compiled to be measured: `face_batch_width` starts at every parent and steps down in proportion to the room; the CC/TT/CT batch takes the largest of CC's, TT's and CT's program prices, `sector_batch_width`; a local paired, ordered round: $D=5R^2+3(2c)^2+12nR$ with $2c=2\min(R/2,K_{\rm budget})$, the kept-span carrier; compiled rounds hold 3.8-9.7 $R^2$ against 7.5-12 here); model
checks $D=8n^2+4nR$, $s_f=2a$; CT cross reduction (local) $D=CT+8R^2+4nR+\max(C,T)R$,
on the face its compiled program. A local round has $b=P$. The native cuSOLVERMp `eigh`
adds a private $n^2/P$ operand tile beside its workspace, which
`distrib_la.workspace_bytes_per_rank` includes. The ledger
(`CapacityLedger`) owns admission: when a stage's aggregate with the named
concurrent stages exceeds the device budget (`memory.per_device_gb`) less the
runtime reserve, the row is recorded FAIL, one warning line is printed, and
the stage is admitted and runs (owner 2026-10-01: no refusal on a price).

**Bank payload** (`shared_pole_bank_payload_bytes`, the residence admission):

$$
B=\frac{16\,N_q}{P}\Big[(2N_{\rm dense}+N_m)\,d^2
+N_{\rm line}\sum_f\big((1+2S)\,n_f+2S\,n_{f'}\big)\,r_f\Big],
\tag{SP 8}
$$

$n_{f'}$ the cross rows of a photon family (absent on a charge bank) and $r_f$
the carrier of the family's line cap. On Fe $8^3$ bispinor ($N_q=59$,
$d=3164$, 14 line and 8 dense samples) this is 28.5 face tiles per parent
against 77 with dense line samples. The producer's selection adds, beside one
group's carry, the endpoint blocks of $W$ and $\partial_sW$ and the
$n\times n$ normal matrix $W^\dagger W$ with its eigenvectors, $n$ the largest
family's rows (`line_selection_price`). On the local route that is
$16\cdot 2\lceil N_q/P\rceil (d^2 + n^2)$ B per rank, the rank's parents in one
batched eigh; on the face route $16\,(2N_q d^2 + 2N_q n^2)/P$, every parent
of the stack at once. The eigh service's workspace comes on top.
