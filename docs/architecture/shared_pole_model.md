# The shared-pole screened interaction: implementation

This page is how `sigma_w_model = shared_pole` is computed each map: the
response bank that samples $W_c$, the constructor that reduces the samples to
one real-pole model per parent $q$, the store, the Σ consumer, the gate rows and
the byte model. The model, its equations `(W n)` and why it is built this way
are [the shared-pole theory page](../theory/shared-pole-w-model.md); deck keys
are in the [input reference](../input_reference.md); rebuilding the model inside
a QSGW loop and retaining its quadrature are
[self-consistency](../self_consistency.md#shared-pole-w-with-retained-quadrature).
The full-density TD snapshot extension uses a general two-sided pole bank,
not the passive real-pole constructor; its additional contract is
[§11](#general-snapshot).

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
| photon CC/TT/CT sectors ([bispinor sectors](bispinor_shared_pole_w.md)) | `gw.shared_pole_sectors` |
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
whose shared fit fails takes the closed-form imaginary-axis set when it fits the
pencil's capacity, else is split in halves down to single samples, and one whose
pencil is wider than both halves' is split without a try. The rule never
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

**Where the pencils are computed.** Each proposal geometry's shift pencil (the
stacked Hankel QR, the SVD of its $R^\dagger$ and the projected shift) depends on
the geometry only, never on a decision. On a device leg it runs on the rank's
GPU (`gw.response_pencil.DEVICE_PENCILS`, a source with the contract of
`minimax.host_pencils`):
- QR is `jnp.linalg.qr`. The SVD is the Hermitian dilation
  $\left[\begin{smallmatrix}0&R^\dagger\\R&0\end{smallmatrix}\right]$ solved by
  `distrib_la.eigh_local`. The projected shift is GEMMs.
- The next geometry is dispatched before the current one is read, so the
  device builds it while the host scans. Each consumed geometry costs one
  device-to-host read of $(s, \text{shift})$, at most $192 + 192^2$ complex.
- Every decision stays in minimax on the host: the node count, the sampled error,
  the coefficient mass, stagnation and the group split.
- Shapes are exact. Zero-padded buckets came back non-finite from the ROCm leg on
  954 of the 1020 pencils of the Fe V2 tree; exact shapes on 0. Because a
  rank's attempts are its own, every rank compiles each round's shapes before
  the round (`DevicePencils.prepare` through `compile_ahead`), as the cross-rank
  compile agreement requires.

The dilation resolves a small singular value only to about $8\varepsilon\,\sigma_1$
(`DevicePencils.sigma_error`); host LAPACK keeps it to about $10^{-4}$ relative.
A source's acceptance at count $n$ stands only when
$\texttt{sigma\_error}\cdot\sigma_1 \le 10^{-3}\,\sigma_n$
(`minimax.complex_response._SOURCE_SIGMA_RTOL`). Otherwise minimax rebuilds that
geometry with `host_pencils` and scans it again from the same pole order, so the
rule is the host's bytes. A rejection there is the host's own. On Fe V2 only
group [0:22] triggers it ($\sigma_{80}/\sigma_1 = 8.9\times10^{-15}$, a bound of
$2.5\times10^{-2}$ against $\le 4.7\times10^{-5}$ for every other group).
Rejected routes: rocSOLVER `gesvd`, 5.3 s per geometry against 0.34 s; its Jacobi
driver, wrong singular values (relative error 0.96).

Fe V2 P=4 (lane rcgpu): `bank.rule_construction` went 49.23 → 33.76 s. With the
re-decision, eqp against int6/e5 went from 6.1e-7 / 1.53e-6 eV to
2.4e-8 / 7.3e-8 eV, for +0.2–0.3 s. On CUDA the same `jnp` and `eigh_local` calls
reach cuSOLVER; that leg has not been run.

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

`bispinor_gw = full_shared_pole` builds CC, TT and CT models from one photon
bank. Their model, carrier, route, staged construction, store and Σ consumer
are [shared-pole W for bispinor sectors](bispinor_shared_pole_w.md).

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
and a face model check beside its own whole-chain program, which is also its
retry); the service runs a stack one or more whole matrices per rank when its
shape price fits that room, else on the whole mesh
([eigh stacks](../services/distrib_la/api.md#eigh-stack)). The route is fixed
before any read and the report line `Shared-pole constructor:` names it and
its prices. A face program holds its eighs' first attempts; a failed check
reruns that round's whole-chain program on the whole mesh
(`distrib_la.checked_program`).

**Sectors.** The bispinor sectors take their own route, decided once per map
from the recipe shapes ([bispinor sectors §4–5](bispinor_shared_pole_w.md#sector-route)).

**Scalar face batches.** Admission uses the actual raw-parent count, not the
full response-grid size. A candidate capacity $c$ for $N$ parents becomes
$b=\lceil N/\lceil N/c\rceil\rceil$ before selection pricing and the compiled
reduction admission: the same number of fixed-shape rounds, with fewer
synthetic slots in a short last round (10 parents at capacity 9 use two
width-5 rounds). Each balanced candidate is compiled and admitted at its own
width and room; no monotonic workspace assumption bypasses that check.
The local P-wide schedule and bispinor sector routes are unchanged. This is
a scheduling/resource change, not a measured wall-time speedup.

**Face products.** Every face GEMM of the constructor is `distrib_la.panel_matmul`
on the square mesh, every parent of
a program in one exchange per panel, transposed operands by one
grid-transpose exchange; face programs compile with XLA's latency-hiding
scheduler (`FACE_COMPILER_OPTIONS`).

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
| `grid`, `electronic_grid` | bosonic response grid and electronic Green grid, respectively; an old header without `electronic_grid` means it equals `grid`. Both domains are authenticated on read. |
| `qgrid_response_window` | Optional explicit response-window identity; absent means `sharp`. `fejer` is a distinct recipe/model and a mismatched reader refuses. The approximation and unchanged ordinary coarse-q Σ contract are owned by [k-convolution](kconv.md#bosonic-response-grid). |
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
stores the map's iteration head at $\omega=0$. The four-current route stores
its CC sector's $V+W_{c,CC}(0)$ ([bispinor sectors §6](bispinor_shared_pole_w.md#sector-store)).
Both evaluators run at
the q parents of the run's V wedge, and the restart stores those parents with
their unfold tables: no full-q W is formed for the file, and BSE unfolds on
load ([BSE](bse.md)). Plain-MPA and metal restarts carry no `W0_qmunu`
([BSE](bse.md) says what a BSE on them does).

A one-shot Σ sweep keeps its checkpoint, `file_io.sigma_checkpoint`: on
shell the pair and slots (`tmp/sigma_checkpoint_oneshot_on_shell.h5`), under
`sigma_omega_grid_debug = true` the ω-grid cubes
(`tmp/sigma_checkpoint_oneshot.h5`); a rerun with `restart = true` whose
checkpoint authenticates skips the sweep ([on-shell Σ_c](../theory/sigma-on-shell.md#grid-debug)).

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
The photon sectors synthesize through the same owner and placement rule
([bispinor sectors §7](bispinor_shared_pole_w.md#sector-sigma)).

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
At the Ni 20³ P64 tile a node takes 0.549 → 0.492 s (claim 3115).

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
executable and applies the pooled fit. The sector consumer brackets the CC
class only ([four-current Σ](../theory/band-extrapolation.md#four-current)).

**Γ head** (`gw.shared_pole_head`). `head_correction = full` evaluates the
current body at Γ one frequency at a time, folds the common head wings
through total $W$ and fits the scalar head with the MPA head owner; its
vertices trace the spinor index and its capacity is $2/(n_{\rm spin}n_{\rm spinor})$
states per band. On an ordered store the body is the signed
$b D_+ b^\dagger - [b D_- b^\dagger]^T$, $D_\pm = 1/(2W(z\mp W))$, at $z$, and the
wings sum every ordered pair. It refuses on an ordered metal
(`GATE shared_pole_head_ordered`: the static Matsubara slot is not certified
without time reversal) and on the four-component charge store
(`GATE shared_pole_head_nspinor`). `head_correction = no_local_fields` carries
the direct tensor $S(\omega)$, finalized with no Γ body ([four-current heads](../theory/four-current-head-corrections.md)). Metal head routes are
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
reduction $D=14R^2+12nR$, $s_f=0$ (on the scalar face route the conservative
shape terms reserve the room, then `face_round_compiled_bytes` prices the
actual balanced candidate at that room; `face_batch_width` starts at every
raw parent and steps down in proportion to the admitted whole price,
balancing before each new quote/compile as in §6. The staged sector route
bounds its stage programs by the shape price, `stage_width`; a local paired,
ordered round: $D=5R^2+3(2c)^2+12nR$ with
$2c=2\min(R/2,K_{\rm budget})$, the kept-span carrier; compiled rounds hold
3.8-9.7 $R^2$ against 7.5-12 here); model
checks $D=8n^2+4nR$, $s_f=2a$; CT cross reduction (local) $D=CT+8R^2+4nR+\max(C,T)R$,
on the face the program price alone (no dense term beside it). A local round has $b=P$. The native cuSOLVERMp `eigh`
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

## 11 Full-density snapshot extension {#general-snapshot}

`gw.shared_pole_screening.screen_general_shared_poles` is the native TD
snapshot producer. This is a general first-order frequency realization,
not a relaxation of the passive constructor's Gram/pole gates. The
[TD theory owner](../theory/td_agw_dynamical.md#retarded-reference) specifies
the upper-endpoint density ordering and reference branch approximation.
This direct-FFT entry requires complete unshifted electronic and response
grids in canonical C order. They may coincide (the unchanged default), or
the optional scalar `qgrid` may divide the electronic `kgrid` axis by axis.
The centroid-plan owner authenticates electronic coordinates, not just
identity index tables; the separate coarse response domain is checked
against its own complete canonical q coordinates. Permuted or shifted
inputs refuse without implicitly reordering orbitals or the Coulomb operator.
The real-space restriction, normalization, Σ transport and approximation
limits are owned by [the bosonic response-grid contract](kconv.md#bosonic-response-grid).

`snapshot_runtime.prepare_native_source` retains the complete fine electronic
orbital and bare-Coulomb carriers. Screening selects the coarse bare-Coulomb
rows through the symmetry owner's bounded row evaluation; it does not replace
the fine V used by the electronic source. Response banks, q/−q pairing and
stored W factors use the bosonic domain, while Green phases and their
normalization retain the electronic grid. Store `grid` is bosonic and
`electronic_grid` records the fine domain; missing `electronic_grid` in a
legacy header means equal grids, not permission to infer a different one.

`greens_function_kernel.density_left_endpoint` forms the full-band
$\Psi\rho$ face; `w_isdf.full_density_response_stream` keeps its phase on
the right band index. It evaluates the two independent density insertions
of $P^>-P^<$ using the native spin-traced momentum FFT product. Quadratic
density terms cancel algebraically, but neither remaining insertion can be
replaced by the stationary conjugation shortcut. Frozen-active virtual
transitions remain. `response_bank.full_density_sample_rules` supplies
scalar rules over each exact endpoint transition interval;
`full_density_bare_moments` supplies the leading moments, including a
possible nonzero $1/z$ coefficient. No sampled dense-frequency integral
or fixed spatial Coulomb basis is introduced.

The primary stream expands the exact coherent products into FF/FA/AF/AA,
with separately weighted $\Psi\rho$ and $\Psi(I-\rho)$ endpoints. The numerical
response group can contain more bands than the propagated window; added
bands keep their actual original 0/1 density and current energies. Current
global endpoint gaps, never band-index ordering, determine signed ranges.
The identity $B(R,L;-\bar t)=\overline{B(L,R;t)}$ supplies both primitives
from one product; replacing $-\bar t$ by $t$ is not valid. Endpoint masks
enter the Green owner's phase selector, preventing unused deep-band
exponentials from forming $0\times\infty$ without changing any live phase.
Separate per-parent endpoint band ranges also enter the existing
`build_G_tau`/`panel_matmul` product. They skip only structurally zero
RIGHT columns: exact 0/1 spectators and the block's fixed support. Every
propagated coherent column remains live even at a zero diagonal occupation.
The full-band faces, current energies, coherences and all virtual transitions
are unchanged; no band repacking or physical band cutoff is introduced.

Supports are grouped by actual height and noncrossing geometry. Noncrossing
groups use the public continuum-checked `minimax.response_laplace_rule`;
crossing groups use centered `minimax.response_bank_rule` real-time panels.
Only named exhaustion of the remote Ritz continuum certificate falls back
to that SAME centered public real-time converter, also valid on noncrossing
intervals. It must independently pass the unchanged per-support bank norm
and coefficient-mass/carrier bounds; this is not acceptance of the failed
Ritz certificate (which has additional relative-kernel constraints).
Other domain, arithmetic and runtime errors still propagate. The rule's
`ritz_refusal` and distinct construction label preserve the original reason.
The adapter certifies physical $d/d(z^2)$ chain factors and each support's
own height normalization, not merely the minimum height. Both primitive
signs and the original $10^{-8}$ scalar criterion remain. Every panel's
polarization is summed before one Dyson solve. Small canonical rules cross
ranks through the existing response-bank gather owner. The explicit limit
is 32 carriers of at most 384 nodes per group; all physical bands/ports stay.
The full signed unpaired stream remains an exact-moment/independent-oracle
tool, with bounded shared-pencil planning, not an automatic production
fallback when a block rule refuses. Node counts and output-node updates
are reported separately; fewer carried outputs do not imply fewer Greens.

All original native fit and held supports retain their exact IDs, order and
values. `active_eta_verification_sites` adds low-height training at zero,
half and the edge of the current propagated active energy-difference span,
with independent held points at one-quarter and three-quarters. Explicit
position tables distinguish training from holds even though native IDs stay
unchanged. Exact duplicate/collapsed-span collisions are recorded, never
used as fictitious independent holds or to promote native held data into
training. These are numerical geometry choices, not a self-energy error certificate. Full
complex self-energy refinement, including real/offdiagonal components and
collision operators, remains necessary before claiming numerical convergence.

Only scalar rules whose signed interval contains the current complete-band
gap interval, with identical shifted support sites and tolerance, are held
inside one Hamiltonian solve. New block intervals use at most the response
owner's 4 eV hold pad, bounded by their own width and noncrossing margin.
Current phases, density insertions and W are recomputed.
The compiled response factory is keyed on topology, FFT dialect, shapes
and moment powers; energies and the scalar reference are dynamic operands.
Flushed phase starts and synchronized wall times distinguish planning,
streaming, Dyson, directions, pencil assembly, host reduction, checks and
persistence. No physics gate is inferred from these resource receipts.

Each fitted parent checks its Laplace-growth margin before fitting the next
batch. A refusal publishes collective-safe frequency-only E/A, canonical
maps/spaces/poles, final-bank per-column residue norm bounds and sample/
moment errors before raising. Optional frequency-only action Grams describe
the **raw** reduced transfer, not the later self-negative crossing mixture.
Independent microscopic Dyson probes near suspect poles distinguish fitting
defects from evidence for gain; neither shifts poles nor admits a failed
model. `stop_after_model_q=0` intentionally stops after this unadmitted
partial diagnostic, with no full-q model or self-energy claim.
On that explicit diagnostic only, a refused growing pole also probes the
native dielectric at the fitted pole itself and two offsets. The exact
operator is $\epsilon=I-\mathrm{pref}\,H P^R H$, using current microscopic
response, the same Coulomb root and prefactor as the Dyson owner, and only
the declared polarization lag decay (no additional collision eta).
No fitted W is evaluated at its singular denominator. A basic dielectric
Frobenius norm/padding witness is committed before the checked native LU;
inverse norms and verified inverse residuals go into separate transactions.
The centroid owner's mask excludes identity padding but retains physical
Coulomb-null directions. The verified inverse residual supplies an algebraic
interval for the logical dielectric's smallest singular value, with an
explicit FP64 allowance; unresolved residuals/padding refuse the interval.
These local witnesses do not certify a physical pole or quadrature error,
and a small relative error on a large total W is not pole validation.

The source GW recipe sets finite support sites and tangential direction
widths. Its pole budget counts positive-frequency modes; the general
realization's signed-frequency budget is twice that count, not an extra
factor-of-two truncation. Full microscopic Dyson actions enter
`shared_pole_pencil.general_tangential_pencil`. Independent left/right
actions produce $E,A,B,C$ with $W_c(z)=C(zE-A)^{-1}B$.
Infinity rows use exact bare/Dyson coefficients projected onto the selected
infinity directions. They do not guarantee exact full-port moments after
finite direction selection and numerical reduction. Full $C_1,C_2$ defects
are reported as diagnostics, not independent admission gates. The held-W
and Laplace-growth gates remain enforced; neither is a proof of self-energy
convergence or of accurate asymptotic moments in every port direction.
`shared_pole_constructor.reduce_general_pencil` uses the two-sided SVD of
the frequency-equilibrated descriptor E for this strictly proper finite-pole bank, with relative
rank tolerance inherited from the source scalar bank tolerance. A-only
directions unresolved in E are not promoted into nearly infinite modes.
This is a numerical approximation checked against current response data,
not a generic singular-descriptor equivalence or stable-pole projection.
Before fresh-space selection, finite/infinity equation units are matched
using the current full requested energy span (at least the electronic eta),
not an adjustable stability scale. Infinity rows/columns have one inverse
frequency factor; their actual tile-interleaved order supplies the masks.
Eight deterministic simultaneous infinity-norm Ruiz sweeps equilibrate the
combined E and A/reference magnitudes. Exact-zero rows/columns retain unit
scale; no physical population, pole or residue floor enters. For positive
diagonal left/right factors D_L,D_R the unreduced transfer is unchanged by
E'=D_L E D_R, A'=D_L A D_R, B'=D_L B, C'=C D_R.
The selected spaces are immediately mapped back as X=D_R X' and
Y=D_L^H Y'. All returned/held spaces therefore use ORIGINAL pencil
coordinates, and both fresh and held eigensolves project current original
E/A. A changed reference never reinterprets a held space. Equilibration
changes the numerical truncation metric, not exact untruncated physics;
native/held W and full complex column-Sigma refinement remain necessary.
Reference, unit masks and scale hashes are retained in diagnostic receipts.
For BOTH fresh and held spaces, the current projected descriptor is again
SVD-balanced into an ordinary non-Hermitian eigenproblem. Its left/right
inverse-square-root transformations recover the original descriptor maps;
the full eigenvector metric handles degenerate blocks. Every retained state
must remain resolved: singular/unresolved held descriptors refuse without
rank dropping. Original-pencil backward and modal-commutator gates remain.
Resolved growing poles stay unchanged, without Hermitian/PSD projection.
Held spaces remain canonical within the inner solve; current pencils and
all response/growth checks are recomputed. One common conservative host-price
owner covers the extra SVD/balance buffers before gather or factorization.
The resulting full-port factors obey

$$
W_c^R(z)=L\,\operatorname{diag}[(z-p)^{-1}]R^\dagger.
$$

Only reduced per-parent frequency pencils cross the bounded host LAPACK
seam. Its workspace is priced explicitly and refuses above its declared
capacity; microscopic samples and endpoint factors stay distributed. The
serial reduced constructor has one canonical process-zero owner. The exact
batched route (`general_pencil_batch_plan`, `collective_general_pencil_batch`)
admits a prefix of at most P small pencils from shapes **before** their host
gather. Every rank reduces at most its one assigned parent, concurrently;
each parent still has exactly one canonical owner. All ranks first
gather the same admitted reduced E/A, then exchange an ordered owner success/error record
before broadcasting exact uint64 bit patterns of complex128 poles, both
coefficient maps and both held frequency spaces. JAX x64 is required for
this handoff. Only these reduced arrays cross ranks; independent host
eigenvector gauges must never be mixed across distributed microscopic
factor faces. Current E/A are reduced anew every iteration; only numerical
frequency spaces may be held. Its 4 GiB default bound prices the maximum of
two disjoint phases: one LAPACK solve beside all other batch inputs, then
retained canonical models beside one owner result and one transfer buffer.
Existing held-space bytes are an explicit caller input. Capacity reduces
the numerical batch width (down to one), never the physical pole order,
spatial ports or q count. Per-parent receipts record reduced shapes/order,
SVD/eigen/reduction and canonical-handoff walls and the owner/batch capacity.
These are concurrent host solves of independent parents, not a batched
distributed general eigensolver. The general solver is therefore
capacity-bounded, not a claim of an
arbitrarily large distributed non-Hermitian eigensolver. At 1761 frequency
states a complex pencil is about 50 MB; the conservative reduction workspace
is about 1.39 GB. Two 978-by-1761 endpoint factors at each of 36 q total
about 1.98 GB globally before padding and reciprocal self-q columns. These
are shape prices, not measured GPU walls or peak memory.

The store `lorrax.shared-general-pole.v1` has independent
`left,right[q,mu,1,Kmax]` complex128 factors and `poles_ry[q,Kmax]`
complex128. Every q is explicit; current Hamiltonian, orbitals, full density,
centroids and recipe are authenticated. Inactive factor columns are zero.
The final digest reads bounded canonical file rows, not a full microscopic
device gather. Opposite q banks are paired algebraically, and self-inverse
q banks include the half-weight crossing partner. The final bank must pass
the actual microscopic response checks after this pairing.

`gw.mpa.sigma.ordered_shared_pole_sigma` synthesizes
$L\exp[-i(p-E_{\rm ref})\tau]R^\dagger$ on the existing faces and feeds the
same native node loop, momentum FFT and matrix projection as equilibrium
GW. Both reference electronic branches use the literal retarded denominator
$E_b-e_n-p+i\eta$. Signed/complex pole extrema enter certified product
windows; $\eta$ must exceed their growth bound. No pole is shifted and no
anti-Hermitian completion or extra on-shell average is performed.

`ppm_sigma.window_sigma_c` selects only the active destination block without
truncating virtual bands. Cumulative band brackets retain the active-only
and all-virtual intermediate results in one sweep: the first supplies
$\Gamma,D$, the last supplies $M_H$. The identity-minus-density particle
line keeps its cancellation at fully occupied states. Only small active
matrix outputs reach host diagnostics. Native joint propagation never
calls the archived projected-vertex evaluator.

The existing frozen reference head is evaluated on current electronic
energies and full density, once in each independent active/full result.
Its source pole widths are retained; bulk collision $\eta$ is not added
to those head denominators. Native receipts distinguish
`bulk_evaluation_eta_ry`, signed `head_pole_imaginary_parts_ry`, and
`head_width_convention`. The authentic head-included CPU oracle uses
different bulk/head widths and checks both retarded slots against literal
pole sums. This is coverage of the declared frozen-head approximation,
not a refreshed-head or width-convergence result.

## 12 Internal transition-Galerkin proof (not a production route)

`gw.transition_galerkin` is an internal one-parent numerical proof. It is
not selected by a deck, does not replace §11, and does not admit or write a
full-q model. Every microscopic port and band remains in the response. Its
ordered equation is

$$
P(z)=T K_\rho(z-D)^{-1}T^\dagger,\qquad
(K_\rho X)_k=X_k\rho_{k-q}-\rho_kX_k,
$$

where $D_{kbd}=E_{kb}-E_{k-q,d}-i\gamma$ and
$T_{\mu,kbd}=\sqrt{2/(N_k n_{\rm spin}n_{\rm spinor})}
\sum_s\psi_{kbs\mu}\psi^*_{k-q,ds\mu}$. In particular, $K_\rho$
does not commute through the resolvent for a coherent density. The catalogue
omits only exactly null K columns; it uses no occupation threshold. A held
catalogue can reserve every physical active column, and must refuse if a
new density activates any omitted row. Current energies, density, partner
indices and native wavefunctions are action operands, not compiled constants.

Transition rows are whole-k sharded over both mesh axes. The vertex is
formed in pair tiles by `lax.scan` inside `shard_map`; port actions and
inner products use the ordered collective owner. Only narrow full-port
panels are replicated. Source ψ face-to-k redistribution can require a
transient source-ψ replica; it must be included in any future material
residency admission. Neither microscopic W nor tall transition vectors are
gathered to host.

Exact current Dyson right and dual states seed one common Q. The dual
applies K **before** the adjoint free resolvent. Existing distributed CGS2
removes the accepted basis, then the subspace owner's TSQR orthonormalizes
the residual block. Only its small R reaches a canonical root SVD; exact
complex128 bit patterns distribute its rotation. Rank admission uses
`common.rank_criterion.probe_is_independent` relative to the ORIGINAL seed
block norm. The spectral-closure owner strictly refuses a cut through a
near-degenerate residual singular block. This is probe independence, not
an overlap inverse or a pole-stability selection. Capacity refuses an
admitted block that does not fit rather than trimming it. Binary stationary
controls use separate exact K=-1,0,+1 sectors, including coherent-reserved
K=0 coordinates. Normalized rows are restored to their known structural
sector support, not selected by an occupation/amplitude floor. Opposite-q bases are related by
conjugation and the exact $(k,b,d)\mapsto(k-q,d,b)$ permutation; self-inverse
q seeds are completed and the resulting subspace closure is checked.

With $A_Q=TK_\rho Q$, $B_Q=TQ$ and $D_Q=Q^\dagger DQ$, the full-port
proof is

$$
W_{c,Q}(z)=VA_Q[z-D_Q-B_Q^\dagger VA_Q]^{-1}B_Q^\dagger V.
$$

For a stationary binary density only, a structurally separated zero sector
may remain in the HELD Q but be omitted from MODAL MATERIALIZATION. The
implementation requires exact pure live/zero coordinate support, both
current projected D crossblocks exactly zero, and exactly $A_0=0$. In the
ordering $Q=[Q_L,Q_0]$, the generator is then
$M=\left[\begin{smallmatrix}M_L&0\\G_{0L}&D_0\end{smallmatrix}\right]$
and the output is $[VA_L,0]$. The transfer is exactly the live-block
transfer even for coincident dark eigenvalues/Jordan blocks; no pole is
inspected to decide this omission. The full Q and full projected generator
are retained. Changed coherent rho uses the full current generator; absent
the exact structural certificate no dark reduction is made. This is not an
approximate null-residue threshold or a growing-mode removal.

The existing canonical complex-pole executor diagonalizes the materialized
projected generator; it does not fit a W Loewner pencil or remove growing
modes. Only projected frequency matrices reach
the host. Returned native left/right factors distinguish logical count
from zero padding and are tested against literal complex column/row Σ.

At 36 k and 160 bands, a four-coherent-band catalogue has 303552 exact
live rows (280800 for the binary equilibrium population). At order 1000,
Q alone is about 4.52 GiB globally, and explicit full 978-port T is another
4.42 GiB, before padding. Twenty retained parent bases alone would be
about 90 GiB globally. These are shape prices, not measured GPU walls.
The one-parent receipt counts overlapping immutable basis buffers,
streaming K scratch, and the providers' workspace. Source ψ/W/V,
retained seed arrays, source redistribution and other parents are explicitly
additional. No all-parent retained-space, fixed-point, material-capacity,
full-q model, or trajectory admission follows from this proof. Current
microscopic W/column-Σ refinement and electronic boundary tests are still
required before a production integration.

The same internal module supplies a direct one-parent verification sample,
`transition_polarization`: slice the full-port columns of $T^\dagger I$
exactly, apply the current free resolvent, then K and T, and accumulate
bounded port panels on the native XY face. It keeps all ports and full
coherent density. Bridge programs are reused between sites; the column
start is dynamic, so only the regular/final panel widths create variants.
Each panel completes before the next is submitted. The returned Pi is
already PHYSICALLY normalized by T; `transition_dyson_value` supplies
`Pi / response_algebra_receipt['prefactor']` to the existing checked-LU
Dyson owner with its unchanged raw-response convention and authenticated
Coulomb root. Neither helper adds an electronic eta or changes gamma.
`transition_panel_capacity` can be called before T allocation and prices
T, retained Q, K/state scratch, narrow panels and overlapping Pi buffers.
Source residency must be supplied separately; the checked-LU owner's
workspace/ledger obligation is explicitly additional. This capability
does not itself admit a material bank, seed a production route, or certify
interpolation/column-Sigma convergence from a few direct sites.

The internal `transport_transition_basis` helper moves a held numerical Q
between instantaneous active-band gauges without holding physical energies
or screening fixed. For $S_k=U_{\rm ref,k}^\dagger U_{\rm current,k}$, frozen
identity outside the active window, each transition matrix transforms as
$X_{\rm new,k}=S_k^\dagger X_{\rm ref,k}S_{k-q}$. The catalogue must explicitly
reserve every active row/column coordinate; missing or inexact indices and
nonunitary rotations refuse. Whole-k sharded kernels process at most 32 Q
rows at a time, with no transition- or port-sized host gather. Input/output
Q, unpack/rotation workspaces and caller-supplied source residency are
priced before allocation, against an explicit device budget. Host metadata
placement uses the process-local collective owner for a remote-rank mesh.
Current T, density, energy and K/D actions must still be rebuilt/supplied;
this helper is not wired into production, and its covariance tests admit
neither a full-q model nor a physical trajectory.

A possible future capacity alternative, **not implemented**, is to retain
the ordered reference seed descriptors and their full canonical coefficient
map $Q_{\rm ref}=S_{\rm ref}C$, then regenerate one parent per staged round.
The current stored left singular rotation alone is insufficient: C must
also compose every CGS2 overlap update and the small QR/SVD factors, without
normal-equation inversion. Reference E/rho/gamma, full-W right/adjoint actions,
sector projections and q/crossing maps define the original space and may
not be silently replaced by current values during regeneration. Reconstructed
projector/orthogonality checks and gauge transport would precede projection
of the **current** T/K/D/V. This caches a numerical space, not a physical
fluctuation history. Its extra GEMMs, reconstruction conditioning and staged
memory must be measured against retaining or streaming full Q through the
existing owners; no memory reduction or speedup is admitted by this design.
