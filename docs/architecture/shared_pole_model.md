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
   $=\lceil1.8n\rceil$, `bank_rule_tolerance` $10^{-8}$ and
   `sigma_tolerance` $10^{-4}$ for `production`; `relaxed` uses 8 uniform line
   sites, 2 imaginary sites, cutoff $10^{-2}$, no cap, no budget, and
   tolerances $10^{-7}$/$10^{-3}$. `sigma_quadrature_eps` is taken from the tier
   and refuses a conflicting explicit value.
2. **Bank** (§2): Coulomb roots, samples $W_c,\partial_sW_c$ at the supports
   on the imaginary axis and the held supports, the direction panels of every
   fitted line support (§3), moments $M_1,M_3$ (and $M_0,M_2$ when ordered)
   → `bank.h5`.
3. **Residence.** The bank is written frequency-major and read parent-major. It
   stays on the devices when the payload and one read copy fit half the device
   budget *and* the constructor route is unchanged with it live; otherwise it
   goes to pinned host memory if it fits half the host budget; otherwise it
   stays a scratch file. `write_w` and a distributed `linalg` always use the
   file.
4. **Constructor** (§3–§6) → `model.h5`, one parent round at a time.
5. **Σ** (§8) synthesizes $W_c(\tau)$ from the factors.

Every map rebuilds samples, directions, poles and ranks from the current
state; only quadrature rules and support geometry are retained across SC maps.

## 2 The response bank

**Coulomb roots.** $V$ is frequency independent: its roots `H[q,μ_X,ν_Y]` are
solved once for all irreducible parents with the batched solver and held
through the frequency loop.

**Shared-node groups.** Samples are grouped (imaginary axis by $\operatorname{Im}z$,
then the line by $\operatorname{Re}z$), and each group gets one set of complex
times $t$ from `minimax.response_group_rules`. Every node is **one Green-pair
evaluation** $A(t)$ that serves every member's value and $\partial_s$ derivative
in both orientations:

$$
\frac1{d-z}\simeq\sum_j c^F_j e^{-(d-r)t_j},\qquad
\frac1{d+z}\simeq\sum_j c^B_j e^{-(d-r)\bar t_j},\qquad
\frac{\partial}{\partial s}\frac1{d\mp z}=\pm\frac1{2z(d\mp z)^2}.
\tag{SP 1}
$$

The forward product is $A(t)=G_u(t)\,\overline{G_f(\bar t)}$; the reverse is
$\overline{A(\bar t)}$ — the same damping with the orbital product reversed, not
$\overline{A(t)}$. The nodes come from a stacked Hankel shift pencil; a group
whose shared fit fails is split in halves down to single samples, so no sample
uses more nodes than its own rule needs. The group size is the largest whose
donated carry `[2·members, q, μ_X, ν_Y]` and compiled stream temporaries fit the
map ledger: every sample in one group on symmetric decks. The accuracies are
sampled, not continuum certificates; a matched QP comparison is the acceptance
check.

**Domain and occupations.** The transition interval spans every nonzero
occupied/empty weight pair after a sample-only $10^{-14}$ activity floor,
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
$\Sigma[W^{\rm even}]-\Sigma^{\rm odd}$
(`tests/test_shared_pole_stream_orientation.py`).

**Minus-q partner.** The ordered pencil's state $X(-z)$ acts with
$W_q(-\bar z)=\overline{W_{-q}(z)}$ on the directions of $X(z)$, and it must be
the same operator as $W_q(z)$. On an imaginary node $-\bar z=z$, so the sample
is its own partner. At each fitted line support off the imaginary axis the
producer solves it beside $W_q(z)$: the stream's output rows are the union of
the parent rows and their $-q$ rows in one panel, and the partner is formed
from the conjugated $-q$ rows with the **original parent's $V$** (and contact).
It is consumed by the line selection (§3) and never stored. Rebuilding it from
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
scale them by $1/N_k$ (`tests/test_shared_pole_bank_moment_roundtrip.py`).

**Cost.** Green-pair evaluations $\approx\sum_{\rm groups}(\text{nodes})$, each
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
parents per rank when their blocks and the $2n$ eigensystem fit, else the
face) and the bank stores the panels (§7). The constructor reads them
(`line_panel_states`) and selects the supports on the imaginary axis from their
dense samples per round. On the local route both sides run the same one-parent
dilation eigensolve on the same bits, so $Q$ and every count are the ones the
constructor would select; the action GEMMs run at a different panel width
and agree to round-off.

## 4 Pencils and reduction

`gw.shared_pole_pencil` assembles (W 18)–(W 19) for a TRS bank and
(W 25)–(W 26) for an ordered bank, from `Q`, `O`, `D` and the moments only;
blocks come out `P(None,'x','y')` on either route
(`tests/test_shared_pole_pencil_faces.py`). `gw.shared_pole_reduction` applies
(W 20): diagonal equilibration, keep cut `normalized_gram_keep`, the coupled
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

a numerical treatment, not a bound on collective modes. The first SC map
freezes it; later maps report the value a fresh map would choose but never
widen it. Inactive modes have zero factors and the inert pole sentinel; CC and
TT have independent masks, CT one common mask. Refusals:
`GATE shared_pole_sector_treatment_{census,order,empty}`. Held rows score the
untreated fit; accuracy of the treatment is a projected-Σ comparison.

## 6 Execution

**Route.** `constructor_route` admits one route before the first bank read.
**Local parent rounds** — one whole parent per rank, batch layout
`P(('x','y'), ...)` — whenever the complete selection stack fits the device,
whatever `linalg` names: `gw.shared_pole_local.round_program` packs each parent's
panels to the round extent (`round_tables`; ordered originals and mirrors as two
halves of one extent), assembles and reduces its pencil with local dense kernels
and sorts its poles; synthetic slots are skipped. Otherwise the **face route**
runs a parent batch on the complete mesh with `distrib_la` GEMM/`eigh`, both
matrix axes distributed and only spectra and masks replicated. Sectors resolve
their own route (`sector_execution`) against the same ledger. There is no
retry or route change inside a stage.

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
path past that point.

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
constructor resume then rebuilds it. `write_poles = true` exports $(b,\Lambda)$,
from which a BSE takes $W_c(0)=-b\Lambda^{-1}b^\dagger$ exactly.
`write_w = true` dumps the bank as stored (line supports as panels) and is a
debug output.

## 8 The Σ consumer

Σ contracts $G$ with $W(\tau)$ synthesized from the factors inside the Σ
window executable (`gw.mpa.sigma._shared_pole_w_synthesis`, bound to the τ body
by `gw.mpa.sigma.SynthesisTau`, the same pair the photon sectors use):

$$ W_{c,+}(q,\tau) = b\,\mathrm{diag}\big(d_j(\tau)\big)\,b^{\dagger}, \qquad
d_j(\tau) = \frac{e^{-i(\Omega_j - E_{\rm ref})\tau}}{2\Omega_j} . \tag{SP 4} $$

Synthesis uses the Green-function GEMM (`build_G`) at the irreducible parents,
then the fixed-q projection, the little-group realization and the unfold to
the full q grid. The factors are read once per Σ call and stay resident:
$32\,n_{q,\rm irr}\,\mu\,\bar K/P$ bytes per rank face-sharded,
`P(None,'x',None,'y')`, or $16\,n_{q,\rm irr}\,\mu\,\bar K(1/P_x+1/P_y)$ when the
panel search admits the replicated pole columns ($\bar K$ the store's pole
carrier); a nonlocal store also keeps its routed child faces. The budget left
beside them sizes one parent panel × pole-column chunk of synthesis workspace:
parent panels are a static loop inside the executable, chunks of one static
width a device loop, one of each when everything fits. A store whose resident
factors do not fit refuses (`GATE shared_pole_capacity`) with the smallest
square mesh that fits them. The synthesized $W$ always uses both mesh axes.

**Hole routing.** Conduction windows take $W_+(q)$. An ordered store routes
valence windows to the particle–hole partner,

$$ W_-(q,\tau) = W_+(-q,\tau)^{\mathsf T}, \tag{SP 5} $$

gathered at $-q$ on the replicated q axis and transposed on its faces
(`shared_pole_hole_kernel`); TRS stores never take this branch
(`tests/test_shared_pole_ordered.py`, `tests/test_shared_pole_lattice_sigma.py`).

**Two-component decks.** $W$ is spin-scalar; $G$ carries the spinor axes and the
τ kernel broadcasts $W_q$ over both (`ppm_tau_kernel` `prep_w`).

**Γ head** (`gw.shared_pole_head`). `head_correction = full` evaluates the
current TRS body at Γ one frequency at a time, folds the common head wings
through total $W$ and fits the scalar head with the MPA head owner; its
vertices trace the spinor index and its capacity is $2/(n_{\rm spin}n_{\rm spinor})$
states per band. It refuses on an ordered store
(`GATE shared_pole_head_ordered`: the Γ body evaluator is the even form) and on
the four-component charge store (`GATE shared_pole_head_nspinor`). An ordered
store carries `head_correction = no_local_fields`: the direct tensor $S(\omega)$,
finalized with no Γ body ([four-current heads](../theory/four-current-head-corrections.md),
`tests/test_head_direct_ordered.py`). Metal head routes are
[self-consistency](../self_consistency.md#metals-direct-drude-head).

**Debug.** `LORRAX_DEBUG_SHARED_POLE_EVEN_PART` (`all` or `exclude_q0`) feeds
$[W_+(q)+W_+(-q)^{\mathsf T}]/2$ to both branches of an ordered store, so
$\Sigma^{\rm odd}=\Sigma[W]-\Sigma[W^{\rm even}]$ can be measured.

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

Fast CPU tests (4 host devices where a mesh is needed):

| test | pins |
|---|---|
| `test_shared_pole_ordered.py` | planted ordered oracle, projected moments, ordered = even on TRS data at equal rank, dedupe keeps no partner, generic-q assembly, Σ orientations, two-component routing |
| `test_shared_pole_stream_orientation.py` | the ordered stream stores $\mathcal F_q[\chi]$ (SP 2) |
| `test_shared_pole_lattice_sigma.py` | ordered Σ = real-space $iGW$ on a TR-broken lattice; swapped routing fails (SP 5) |
| `test_shared_pole_bank_moment_roundtrip.py` | bank → constructor moments, $M_k=m_k/2$ |
| `test_shared_pole_pencil_faces.py` | pencil blocks are `P(None,'x','y')` on both routes |
| `test_shared_pole_head_two_component.py` | a spin-doubled two-component store reproduces the scalar head; SU(2) invariance |
| `test_shared_pole_head_capacity.py` | the head admits $N_{\rm spinor}$ 1, 2 and refuses ordered stores and the four-component lift by name |
| `test_shared_pole_support_rule.py`, `test_shared_pole_sizing.py` | support placement and recipe sizing |
| `test_shared_pole_capacity.py` | byte terms and route admission |

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
reduction $D=14R^2+12nR$, $s_f=0$; model
checks $D=8n^2+4nR$, $s_f=2a$; CT cross reduction
$D=10\,CT+14R^2+12n(C+T)$. A local round has $b=P$. The native cuSOLVERMp `eigh`
adds a private $n^2/P$ operand tile beside its workspace, which
`distrib_la.workspace_bytes_per_rank` includes. The ledger
(`CapacityLedger`) owns admission: a stage refuses before it allocates when its
aggregate with the named concurrent stages exceeds the device budget
(`memory.per_device_gb`) less the inherited stream/Σ peaks.

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
group's carry, the endpoint blocks of $W$ and $\partial_sW$ ($2\lceil N_q/P\rceil d^2$
on the local route) and one $2n$ dilation eigensystem
(`line_selection_price`).
