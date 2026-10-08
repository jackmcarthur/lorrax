# Four-current sector Σ consumer

This page describes how the dynamic self-energy of the four-current
(`bispinor_gw = full_shared_pole`) route is evaluated from the ordered
sector store: which interaction classes exist, how each one's W(τ) is formed
and convolved with the Green's function, how the instantaneous constant and
the Γ head enter, and what each step holds in memory. Read
[Bispinor (four-current) GW](../theory/bispinor-gw.md) for the physics and
[the shared-pole model](shared_pole_model.md) for how the sector store is
built; this page starts where the store exists. The code is
`gw.mpa.sector_sigma`.

## Objects

The four-current interaction couples two endpoint families. The **charge**
endpoint carries Lorentz index $A = 0$ on the charge centroid basis
($N_C$ packed centroids $\mu$); the **current** endpoint carries
$A = 1, 2, 3$ (Cartesian) on the current centroid basis ($N_T$ packed
centroids $\nu$). The two bases are fitted separately and never merged.
Band indices are $m$, $n$; $\mathbf k$ runs over the k grid and $\mathbf q$
over transfers.

The screened interaction is split into four **ordered endpoint classes**:
CC, TT, CT and TC. Each class is a shared-pole model stored on the
irreducible $\mathbf q$ (the $n_{q,\rm irr}$ parent rows of the store), in
imaginary time

$$
W_{AB}(\mathbf q, \tau) = \sum_{p=1}^{K_{\max}} B_{A,p}(\mathbf q)\, d_p(\tau)\, B_{B,p}(\mathbf q)^\dagger ,
$$

with $B_{A,p}(\mathbf q)$ the endpoint factor of pole $p$ (shape
`(n_q_irr, N_C or N_T, n_A, K_max)`, $n_A$ the endpoint's Lorentz width:
1 for charge, 3 for current), $d_p(\tau)$ the causal weight of pole $p$ at
time $\tau$, and $K_{\max}$ the largest pole count over the parent q rows
(rows with fewer poles carry zero columns). CC and TT each have their own
pole set. CT and TC share one pole set, stored as its two endpoint factors
`CT_C` and `CT_T`; TC reads them with the endpoint order swapped. The store
manifest (`sector-ordered-ph`) authenticates all four classes and the
instantaneous constant $\langle W_\infty - V\rangle$
(`file_io.shared_pole_store.validate_shared_pole_sector_manifest`).

For one class the self-energy at time $\tau$ in the band basis is

$$
\Sigma_{mn}(\mathbf k,\tau)
= \sum_{\mu\nu} \overline{\psi_{m\mathbf k}(\mu)}
\Big[\sum_{A,B} \tilde\gamma_A\, G(\tau) \,\tilde\gamma_B^\dagger \star W_{AB}(\tau)\Big](\mathbf k)_{\mu\nu}\,
\psi_{n\mathbf k}(\nu),
$$

where $G(\tau)$ is the four-spinor Green's function on the parent
$\mathbf k$, $\tilde\gamma_A$ the signed spin permutation of Lorentz channel
$A$, and $\star$ the k-axis convolution. The frequency-quadrature executor
sums the τ nodes into $\Sigma_{mn}(\mathbf k,\omega)$ on the output ω grid.
The class's blocks form one product $A\text{-set}\times B\text{-set}$
(`gw.cohsex_sigma.lorentz_class_vertices`), because the convolution's
interaction operand is laid out `(N_k, μ, n_A, ν, n_B)`.

## Evaluation

`compute_sector_sigma` is called once per Σ evaluation (`gw.sigma_dispatch`).
It integrates three windows in the fixed order CC, TT and the mixed pair
CT + TC (one fused window: the TC partner rides on the CT window), one at a
time, each through the common frequency-quadrature executor
`gw.mpa.sigma.compute_sigma_c_mpa_omega_grid`. Sectors therefore add no new
integration algorithm: the frequency planner, the τ rules and the ω
accumulator are the scalar shared-pole ones. The rule set is built once per
self-consistent iteration (map) for the union of the CC, TT and CT pole sets
and serves all four classes.

For each class and each τ node:

1. **W(τ) on the irreducible q.** `gw.mpa.sector_sigma.sector_synthesis`
   reads the class's factors once per Σ call on the store's own parent rows
   and never unfolds them. Each τ forms the pair
   $W = B_A\, d(\tau)\, B_B^\dagger$ and its antiunitary partner
   $\bar B_A\, d(\tau)\, B_B^{T}$ on those rows (`ParentW`). The occupied
   branch reads $W_-(\mathbf q) = \text{partner}(-\mathbf q)$ through
   q-negated tables (`hole_tables`), so no $-\mathbf q$ gather and no
   conjugated weight is formed. No full-q W, full-q factor or unfolded pole
   table exists.
2. **Row passes.** `sector_node` splits each rank's `(μ_X, ν_Y)` tile into
   passes of whole centroid orbits, sized from the fixed tile
   (`runtime.tiles.TILE_BYTES`) and the shapes (`gw.subtile_stream.plan_windows`).
   A fixed tile is used because a pass sized from free memory differs
   across ranks and deadlocks the collectives. Per pass: the four-spinor
   parent Green of the pass's rows by one local GEMM; one mode-8 Lorentz
   k-convolution that unfolds both G and W on its load and applies every
   vertex of the class ([k-convolution](kconv.md#unfold-consumers)); and
   the band projection of the pass's rows into a rank-local partial. One
   band-block reduce-scatter ends the node.
3. **Band brackets.** With band extrapolation on, only the CC class splits
   its Green band sum into brackets; TT, CT, TC and the constant are added
   to every bracket alike, because the current classes are of order $c^{-2}$
   of CC and their band tails are smaller still
   ([band extrapolation](../theory/band-extrapolation.md#four-current)).

## The instantaneous constant

The constant $\langle W_\infty - V\rangle$ is stored apart from the poles
and contracted exactly once per evaluation, with the equal-time occupied
projector, in the exchange mode of `gw.photon_sigma.contract_lorentz_blocks`
(`instantaneous_sector_sigma`). It is read and packed on its irreducible q,
in parent-q panels when the raw and packed copies together exceed one tile.
The bare exchange Σ_x is computed by its own owner from the bare $V$: charge
$V$, plus the transverse $V$ once when the run has both the current carrier
and the bispinor $V$.

## The Γ head

The sector route admits two head policies, checked when the Σ resources are
resolved (`gw.sigma_dispatch._mpa_sigma_model_resources`):

| `head_correction` | what Σ receives |
|---|---|
| `no_local_fields` (the default for `full_shared_pole`, `gw.gw_config`) | the direct first-order Γ head of all four classes, already inside the sector poles: the response bank adds the head field $W_h - V_h$ to every sample through the packed $\zeta(G=0)$ vectors (`gw.photon_direct_head.add_direct_gamma_field`), so the fitted CC/CT/TC/TT poles carry it. The consumer adds no separate head term. A store without that head refuses (`GATE shared_pole_sector_head`). |
| `off` | no head; a debug setting |
| `full` | refused at configuration (`GATE full_shared_pole_head`): no wing/body local-field fold exists for this route |

The direct head itself (its interband, intraband and contact parts, and the
Γ-cell cubature) is owned by
[four-current heads §5](../theory/four-current-head-corrections.md#direct-bulk-head).

## Memory

Only one endpoint class is live at a time. Its factors are placed
automatically by the scalar model's schedule
(`gw.mpa.sigma._shared_pole_memory_schedule`) with the class's extents; no
deck key names the layout, but the deck's `linalg` setting decides whether
the `local` layout is a candidate. The factors do not depend on τ, so the
schedule prefers a placement that makes each τ's GEMM local:

| `factor_layout` | placement | taken when |
|---|---|---|
| `axis` | pole columns replicated, each centroid endpoint on its own mesh axis; the per-τ GEMM is local | it needs no more panel passes than `face` |
| `local` | whole parents per rank, Lorentz components merged with their centroids, in `distrib_la`'s batch layout; only W moves, one tile per node | `axis` does not fit, the deck has `linalg = local`, and whole parents fit the capacity ledger |
| `face` | both factors on faces (centroid and pole axes distributed); every τ re-gathers the factor panels in a distributed SUMMA | otherwise |

The class keeps its placed factors and the parent poles until its frequency
integration ends. No W(τ) history and no state/pole-pair sum is retained.

Every reservation goes to the shared-pole capacity ledger
([memory model](memory-model.md)), which adds each stage to the stages live
beside it and warns, but still admits, when the sum exceeds
`memory_per_device_gb`. A compiled kernel is reserved at its compiled peak
plus the cuFFT plan scratch of its FFTs
(`gw.mpa.sector_sigma._admit_compiled`, which compiles each signature once);
it refuses only when that scratch cannot be measured
(`GATE shared_pole_capacity`). The factor placement and the constant's
panels are priced from their shapes before they are allocated.
