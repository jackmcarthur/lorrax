# The shared-pole screened interaction: implementation {#shared-pole-implementation}

This page owns **the code path** of the shared-pole route for the correlation
part of the screened interaction (`sigma_w_model = shared_pole`): the modules
and their one-line contracts, the carriers and shardings of every stage, the
recipe and store schemas, the gate and test tables, the capacity accounting,
and the refusals. The physics and mathematics — the object, the pencils, the
quadrature and error theory, the time-reversal-broken construction, the
self-energy routing, and the measured $N_\mu$ pole-count heuristic — are
[the shared-pole theory page](../theory/shared-pole-w-model.md); equations
are cited there as `(W k)` and are not restated here. Deck keys are in the
[input reference](../input_reference.md); the surrounding self-consistent
workflow is in
[self-consistency](../self_consistency.md#shared-pole-w-with-retained-quadrature).

Notation: `q` is a raw parent (one irreducible q; `−q` is its own parent when
time reversal is broken), `n = meta.n_rmu_padded` is the packed centroid
extent, `n_logical` its logical extent, `z` a complex frequency in Ry,
`s = z²`, `b` the stored factor (`factor` dataset), `Ω_j > 0` the poles and
`Λ = diag(Ω_j²)`.

## 0 What runs, and where the code is

| stage | owner | contract |
|---|---|---|
| census, recipe, gates, receipts | `gw.shared_pole_recipe` | resolves the versioned recipe from current metadata; owns `shared_real_pole_v1_r3b`, the two gate tables and the capacity ledger |
| response bank (stream, Laplace cells, moments, Dyson) | `gw.response_bank`, `gw.w_isdf`, `gw.shared_pole_screening` | samples `W_c` and `∂_sW_c` and the exact moments; never forms `W` at a real frequency |
| direction selection and state panels | `gw.shared_pole_directions` | SVD/eigenvector directions, conjugates, mirrors and their dedupe |
| pencils | `gw.shared_pole_pencil` | resolvent-identity blocks of the even and ordered pencils |
| reduction | `gw.shared_pole_reduction` | equilibration, keep cut, Newton–Schulz metric correction, Ritz model |
| rounds and layout | `gw.shared_pole_local` | parent rounds, per-round programs, canonical-order export |
| model checks | `gw.shared_pole_gates` | passivity, moments, reciprocity, pole census |
| chain | `gw.shared_pole_constructor` | the driver that calls the above per round and writes the store |
| device accounting | `gw.shared_pole_capacity` | per-stage byte terms and the `CapacityLedger` |
| scalar Γ head coupled to the body | `gw.shared_pole_head` | MPA head fit bound to the current body digest; refuses unsupported representations |
| store | `file_io.shared_pole_store` | model and bank schemas, identity, commit masks, readers, export |
| physical operator realization | `gw.qgrid_symmetry` | little-group projection of a packed tile; the raw stored model is not the physical operator |
| Σ consumer | `gw.mpa.sigma`, `gw.ppm_tau_kernel` | τ synthesis from factors, hole routing, panel schedule, capacity |

## 1 Data flow and carriers

The chain is one pass per round of parents:

```
recipe → bank (per q: samples + moments) → directions → pencils → reduction
      → model checks → store → operator realization → Σ
```

Every stage binds the **current** state. The bank and the model authenticate a
small identity (wavefunctions, energies, occupations, recipe and gate hashes,
centroid digest, q tables); a stale bank or model refuses, and a
self-consistent map rebuilds samples, moments, directions, poles and ranks
from the current wavefunctions and occupations rather than reusing a frozen
model.

| carrier | global axes and ownership | convention |
|---|---|---|
| response accumulator | `[sample,q,mu_X,nu_Y]` | q/sample panels replicated; both endpoints distribute each tile over all P ranks |
| construction samples | `[parent_XY,sample,mu,nu]` | one whole parent per rank, including its sample stack |
| directions/actions | `[parent_XY,mu,r]` | columns are directions; zero tails carry no physical state |
| pencil | `[parent_XY,R,R]` | row is left state, column is right state |
| local model | `[parent_XY,mu,K]` | real poles, sorted active prefix; synthetic parent slots skipped |
| public model | `[parent,mu_X,1,K_Y]` | canonical parent order; unit charge axis; two factor orientations on read |
| Σ W | `[q,mu_X,nu_Y]` | complete full-q tile before the spatial convolution |

The dense construction is *whole parents per rank*: the batch layout
`P(('x','y'),None,None,...)` gives rank `x·Py + y` one parent of a round, and
its pencil is assembled and reduced with node-local dense kernels
(`gw.shared_pole_local.round_program`). A round refuses before it runs when
eight `[R,R]` blocks and the eigensolve workspace do not fit one device
(`distrib_la.fits_local`). Called on face stacks instead, the same blocks are
assembled through the `distrib_la` face service
(`hermitian_block`, `hermitian_part`, `join_columns`, `on_face`); eager
concatenation and `a + a†` of face-sharded operands would otherwise come out
replicated on every rank. The response stream and Σ keep all-rank endpoint
faces; whole Green's functions never become rank-local.

## 2 Recipe resolution

`gw.shared_pole_recipe.resolve_shared_pole_recipe` turns current metadata into
a flat plan and prints every resolved field with its rule. It refuses when
the census is absent or its energy hash is stale, when `η` is not finite and
positive, when the imaginary interval is empty (`u_min ≥ u_max`), or when a
deck names a shared-pole key under a different body model.

**Production tier (`shared_real_pole_v1_r3b`, `production`).**

| field | resolution |
|---|---|
| fitted supports | 18, counted as line + imaginary; held supports and the `M₁/M₃` infinity block are extra |
| line sites | `support_rule_line_sites`: quantiles of the delivery-weighted, η-broadened crossing density, power `support_density_power = 0.5`, on `[ω_lo, ω_reach]`; from band energies, occupations and η only |
| height | `h = height_eta_factor · η = 4η` |
| top | `L = ω_p + 3.5 eV`, `ω_p = 2√(4π N_active/V)` Ry from the census |
| line reach | `ω_reach`: the largest measured crossing of the delivered states, from the same crossing census as the sites (it is at most `L`) |
| imaginary ladder | log-spaced on `[u_min, u_max]`, `u_min = max(h, gap)`, `u_max = max(16 eV, L)`, count `max(2, round[ln(16κ²)ln(4/ε)/2π²])`, `ε = 10⁻³`, `κ = L/u_min` (equation (W 23) of the theory page) |
| held supports | two line midpoints at 25 % and 65 % of the line interval, two imaginary geometric midpoints; never fitted |
| direction cutoff | `1e-3` relative |
| line direction cap | `ceil(n_logical/16)` right singular directions per line support, whole multiplets |
| widths | imaginary `ceil(0.25 n)`, infinity `ceil(0.125 n)` |
| pole budget | `ceil(1.8 n)` retained equilibrated-Gram directions per parent, largest first; `K` cannot exceed it |
| zero policy | drop `λ ≤ 10⁻⁶` Ry² only within `10⁻⁶` factor-weight |
| bank tolerance | `1e-8` remote-cell certificate |
| Σ tolerance | `1e-4` |

`relaxed` is the same geometry with no pole budget, coarser cutoffs and fixed
small ladders; it is a comparison tier, not a production accelerator.

The plan is flat native arrays — `z_ry` complex128, `role` int8, `distinct_id`
int64, `held` bool, `support_pair` int64 `[N,2]`, plus `fit_ids`/`held_ids` —
with a named `role_codes` table (`line`, `imaginary`, `infinity`, `held_line`,
`held_imaginary`); `infinity` is a reserved moment role, never a bank call.
Repeated physical points share a `distinct_id`; fitted and held IDs are
disjoint by construction.

Two consumers own adjacent state:

* `bind_shared_pole_census` binds current full-band occupations, k weights and
  the active band top, and fills the `CapacityLedger` with `U` and the
  inherited stream/Sigma lifetimes;
* `bind_shared_pole_sc_identity` plus the support session retains the line
  tuple and its envelope across self-consistent maps: an enclosed interval
  keeps the same sampled geometry while the census, capacity ledger and
  physical ranks are rebuilt; an expansion starts a new epoch, and a policy
  or basis change starts a new session.

`sigma_w_support_sites_ev` is the only sampling override. It replaces both
ladders with explicit strictly increasing eV sites, folds the site text into
`recipe_version`/`recipe_hash` (so a store built on a different ladder cannot
authenticate), and leaves height, held fractions, widths, zero policy and
every gate to the resolver.

## 3 Response bank

`gw.shared_pole_screening.screen_shared_poles` drives the bank through
`gw.response_bank` and `gw.w_isdf`.

* **Stream.** `response_stream` accumulates the real-time retarded
  correlation on a certified minimax time rule
  (`minimax.response_bank_rule`) and returns `W_c` and `∂_sW_c` at every
  fitted and held support. The stream conjugates in real space before the q
  transform so the partner orientation `conj(F_{-q})` carries its own weight;
  the ordered route (`ordered=True`, `pair_mode="laplace_ordered"`) builds
  `F_q[χ]` and gathers rows at `−q`, which is the orientation equation (W 30)
  of the theory page requires. The TRS route keeps the incumbent trace
  bit-for-bit.
* **Remote cells.** `response_windows` partitions each sample's transitions
  into window and remote cells; `minimax.response_laplace_rule` supplies the
  positive rule for `1/(d²+η²)^{n+1}` on the same time nodes. The even kernel
  `d/(d²−z²)` weights `forward − reverse`; the odd kernel `z/(d²−z²)` weights
  `forward + reverse` and is computed only for an ordered bank. The remote
  cell edges are derived from the bank's own sample set: a cell stays remote
  only while its Laplace Taylor ratio at those samples is
  `≤ REMOTE_RHO_MAX = 0.3` (`gw.response_bank`), with the historical
  `−35/+40 eV` edges as a floor; the tier's `bank_rule_tolerance`
  (`1e-8` production, `1e-7` relaxed) is the separate certificate floor.
* **Dyson.** Each sample solves `W = (I − vχ⁰)⁻¹v` through the bounded panel
  GEMM (`distrib_la.panel_matmul`) and the resolved LU plan
  (`w_isdf.response_coulomb_powers`, `sample_dyson`), storing `W_c` and
  `∂_sW_c`.
* **Moments.** `exact_bare_moments` and `compute_moment_bank` evaluate the
  band-summed coefficients of `χ₀` and run the full Dyson series (equation
  (W 12)); the bank stores `M₁`, `M₃` (physical, Ry³/Ry⁵) and, for an ordered
  bank, the odd `M₀`, `M₂`. The odd-moment flag is what makes the ordered
  infinity block measurable; a finite-state bank records it `NOT_MEASURED`.

Bank layout: `Wc`, `dWc_ds` `[nq, nsample, d, d]` complex128; `M0`–`M3`
`[nq, d, d]` complex128; a JSON header with the plan digest, per-field commit
masks (`sample_written`, `moment_written`), units, and the same identity as
the model store. `validate_shared_pole_bank` refuses a bank whose plan,
identity, representation or commit masks do not match the current state; a
model is never read from a partially written bank.

## 4 Directions and state panels

`select_round_states` batches the selection per role over the round's
`[slot × sample]` stack; only spectra cross to the host.

* Line supports select right singular vectors of `W_c(s)` above the relative
  cutoff, capped at `ceil(n_logical/16)` and closed over multiplets.
* Imaginary supports select the leading eigenvectors of
  `−Herm W_c(iu)`, width `ceil(0.25 n)`.
* Infinity supports select the leading eigenvectors of `M₁`, width
  `ceil(0.125 n)`.

Each fitted sample contributes its role state and, when its node is off the
real `s` axis, its conjugate state on the partner direction set; the ordered
route then appends the mirror `X(−z)` on the same directions after all
originals, with `W_q(−z̄) = conj W_{−q}(z)` realized through the `−q` partner
row (`partner_realization`; an antiunitary-only route refuses). A mirror of a
`Re z = 0` sample uses its own sample. Per-slot role records
(`sample_id`, `role`, `conjugate`, `mirror`, widths, carrier widths) travel
into the receipt so a padding mode can never be mistaken for a physical one.

## 5 Pencils, reduction and rounds

`gw.shared_pole_pencil` assembles the even pencil in `s` and the ordered
particle–hole pencil in `z` — equations (W 18)–(W 19) and (W 25)–(W 26) of
the theory page — as Hermitian blocks on the x/y face. The confluent block is
the only consumer of `∂_sW`; the infinity rows consume `M₁/M₃` (even) or
`M₀…M₃` (ordered).

`gw.shared_pole_reduction` then

1. equilibrates `G` (or the paired `v`-block) by `1/√diag`,
2. keeps `γ > 10⁻⁸ γ_max` **and** the largest `pole_budget` entries,
3. corrects the retained metric by coupled Newton–Schulz
   (`_metric_inverse_root`, iteration count from the initial infinity norm,
   never from an on-device convergence test),
4. solves the Hermitian Ritz problem and writes `b`, `poles2`, `active` in the
   parent-batch layout.

The ordered route first pairs the states (`w = ½[X(z)+X(−z)]`,
`v = [X(z)−X(−z)]/2z`), cuts on the `v`-block, and reduces the restricted
pencil `[[A,B],[B†,I]]` with a second relative cut; `|µ|` below the cut are
infinite poles whose output weight is reported. Every model check runs in the
same round layout (`check_round`, `round_checks`), and each parent's own
extent — not the round padding — is what the receipts report
(`own_extent_receipts`). `canonical_factors` writes the round's parents in
canonical order, so the store never sees a partial or permuted round.

## 6 The model store

`model.h5` (schema `lorrax.shared-real-pole.v1`) and `bank.h5`
(`lorrax.shared-real-pole-bank.v1`) share one authenticated header written by
`file_io.shared_pole_store`.

| field | meaning |
|---|---|
| `identity` | current-state Hamiltonian, occupations, wavefunctions, recipe and gate hashes |
| `recipe`, `recipe_hash` | the resolved recipe; restart and SC maps refuse a stale one |
| `representation` | `scalar-trs-even-s` (even model), `scalar-ordered-ph` (ordered model), `charge-ordered-z` (ordered bank) |
| `normalization`, `units` | `Wc = b/(z_Ry²−Λ_Ry²)b†` (even) or the ordered positive-pole form; `factor` Ry^(3/2), `poles2_ry2` Ry² |
| `n_q_irr`, `n_q_full`, `q_irr_full_idx`, `qirr`, `operations` | raw parents and the authorized symmetry rows |
| `n_mu_logical`, `nspinor`, `centroid_digest` | basis identity; the operator is the `μ×μ` charge response on scalar and two-component decks |
| model `factor [q, μ, 1, K]`, `poles2_ry2 [q, K]`, `K [q]` | per parent; inactive columns have `b = 0`, `Λ = 1 Ry²`, and `K` is the active prefix |
| bank `Wc`, `dWc_ds [q, a, μ, μ]`, `M1`, `M3` (+ `M0`, `M2` when ordered) | samples and moments of Section 3 |

`write_shared_pole_model` checks dtypes (`complex128`, `float64`, `int64`),
shapes, sorted positive active poles, exact sentinels, and the inactive
prefix; `finalize_shared_pole_model` refuses until every staged parent is
committed (`written_q`), then builds the compact `factor`/`poles2_ry2`
datasets and publishes a `final_commit` digest in a rank-0 transaction, so an
interrupted run leaves an unfinalized file rather than a plausible model.
Readers
(`read_shared_pole_census`, `read_shared_pole_matrix`, `read_shared_pole_faces`)
authenticate before any collective open; `export_shared_pole_outputs`
publishes the map's compact members.

## 7 Operator realization

The stored factors describe the **raw latent Ritz model**. The physical
operator is the little-group average of the packed tile,
`Wc(q,s) = Π_Gq [Σ_k b(q,k)b(q,k)†/(s−Λ(q,k))]`, applied by
`gw.qgrid_symmetry.shared_pole_operator_realizer` through
`symmetry_maps.project_little_group_operator`, with the operation typing
authenticated from the store header. The realization is versioned
(`operator_realization = little-group-reynolds-v1`); a store without it
refuses rather than silently acquiring a physical meaning.

Two consequences are worth stating where the code is read:

* the constructor's held/moment/passivity receipts certify the **raw** model;
  they do not certify the projected operator's error, and a converged
  comparison must measure the physical change;
* the projection averages unitary (or conjugate-unitary) congruences of
  positive residues, so it preserves residue positivity and real poles; at
  complex frequency or time an antiunitary operation acts on the residue
  endpoints, and the partner is the same-time transpose — conjugating the
  whole value would conjugate the scalar resolvent weight.

`shared_pole_packed_action` and `qgrid_trs_policy_from_shared_pole_store`
adapt the same tables for the packed and TRS policies of Σ and the head.

## 8 The Σ consumer

`gw.mpa.sigma.synthesize_shared_pole_parents` builds, for one τ node and one
pole interval, the factor contractions `W_+(q,τ) = b d(τ) b†` and its
transpose, with `d_j(τ) = e^{−i(Ω_j−E_ref)τ}/(2Ω_j)` (equation (W 15) of the
theory page). The weights are built once per node; the physical factors keep
their endpoint-face layouts `P(None,'x',None,'y')` / `P(None,'y',None,'x')`,
and the contraction goes through the existing Green's-function face service.

* **Hole routing.** Conduction windows take `W₊(q)`; an ordered store routes
  valence windows to `W₋(q,τ) = W₊(−q,τ)ᵀ` through
  `shared_pole_hole_kernel`, which gathers the already-built full-q tile at
  `−q` on the replicated q axis and transposes its endpoint faces. No residue
  contraction is repeated, and a TRS store never takes this branch.
* **Panels.** `_shared_pole_panel_tables`, `_shared_pole_panel_unfold` and
  `_shared_pole_routed_synthesis` select the q/pole panels; `_shared_pole_w_synthesis`
  is the production executor; `_shared_pole_panel_cost` and
  `_shared_pole_memory_schedule` size the schedule, and `_integrate_sigma_batches`
  is the shared Σ executor for the store and MPA readers.
* **Two-component decks.** `W` is spin-scalar; `G` carries the spinor axes and
  the τ kernel broadcasts `W_q` over both (`gw.ppm_tau_kernel` `prep_w`). The
  stored factor spin axis is 1 on every admitted deck.
* **Diagnostics.** `LORRAX_DEBUG_SHARED_POLE_EVEN_PART` (`all` or `exclude_q0`)
  feeds `[W₊(q)+W₊(−q)ᵀ]/2` to both branches of an ordered store so that
  `Σ^odd = Σ[W] − Σ[W^even]` can be measured on the production contraction;
  it refuses unknown values and TRS stores and prints a `WARNING -- DEBUG`
  banner.

## 9 Gates and tests

Every construction receipt row carries a version, value, threshold and a
PASS/FAIL/WARN/NOT_MEASURED verdict. The TRS table is
`shared_real_pole_gates_v1_r3b`; `shared_real_pole_gates_ordered_v1` copies it
and replaces the rows marked *ordered*.

| gate | certifies | refuses? |
|---|---|---|
| `representation` | TRS: scalar and TRS allowed; ordered: TRS broken and an ordered bank | yes |
| `normalized_gram_validity` | equilibrated Gram (or `H'_vv`) min/max above threshold | yes |
| `normalized_gram_keep` | the retained rank at the recipe cut and budget | diagnostic |
| `retained_subspace_moments` | TRS: projected `M₁/M₃` identity; ordered: `m₀…m₃` on the infinity directions | TRS yes, ordered diagnostic |
| `zero_ritz_policy` | dropped factor weight within budget; ordered also `infinite_weight_ok` | yes |
| `finite_factors_poles` | finite `b`, positive finite active `Λ`, exact inert sentinels | yes |
| `passivity` | V-whitened `−Herm W_c(iη)` in `[0, I]`; ordered reports the anti-Hermitian part | yes |
| `model_reciprocity` | TRS only: transpose symmetry of symmetric held samples; `NOT_MEASURED` when nothing was evaluated | yes (TRS) |
| `held_w` | held `W`, `∂W/∂s` relative errors | diagnostic |
| `full_m1_defect`, `full_m3_defect` | full-matrix moment defects | WARN only |
| `capacity`, `stream_peak`, `sigma_peak` | device admission within budget; inherited peaks recorded separately | capacity yes |
| `rule_validity`, `sc_rebuild` | bank and Σ certificates cover the current domains; SC rebuilds from current bands | yes |

Fast CPU tests (four host devices where a mesh is needed):

| test | pins |
|---|---|
| `test_shared_pole_ordered.py` | planted ordered oracle; projected moments per order; ordered = even on TRS data at equal rank; dedupe keeps no partner; generic-q assembly; Σ orientations; two-component routing against the Lehmann sum |
| `test_shared_pole_stream_orientation.py` | the ordered stream stores `F_q[χ]` |
| `test_shared_pole_lattice_sigma.py` | ordered Σ = real-space `iGW` on a TR-broken lattice; swapped routing fails |
| `test_shared_pole_bank_moment_roundtrip.py` | bank → constructor moments, `M_k = m_k/2` |
| `test_shared_pole_pencil_faces.py` | pencil blocks come out `P(None,'x','y')` in both routes |
| `test_shared_pole_outputs.py` | store finalization, readers and export |
| `services/distrib_la/tests/test_eigh_keeps_operand.py` | the planned eigh does not overwrite its operand |
| `test_slab_io_mode_required.py` | `SlabIO`/`open_file` refuse a missing `mode=` |

## 10 Capacity and byte model

Per rank on an `x × y` mesh with `P = Px Py` and pencil side `R`, the
reduction admits

$$
\text{reduction} \approx 16\,\big(14R^2+12nR\big)\,b/P
   + 16\cdot3nr\,b/P + \text{native eigh workspace},
\tag{I 1}
$$

with a round width `b = P` (each rank holds one whole parent). The native
cuSOLVERMp eigh adds a private operand tile of `n²/P` next to its workspace,
which `distrib_la.workspace_bytes_per_rank` includes; a byte model without
that tile under-counts the measured CrI₃ `q=1` construction peak (2.140
against 2.907 GiB per rank). `gw.shared_pole_capacity` turns the terms into
the map ledger's rows, and the constructor reserves every new object through
the `CapacityLedger` before allocation; inherited bank and Σ objects and host
staging are reported separately.

## 11 Deck keys, restart, self-consistency, refusals

| key | effect |
|---|---|
| `sigma_w_model = shared_pole` | selects the route; requires `compute_mode = mpa`; a scalar head stays MPA |
| `sigma_w_accuracy = production \| relaxed` | selects the tier (pole budget, cutoffs, ladders); not a reuse switch |
| `sigma_w_support_sites_ev` | support-study override; replaces both ladders and enters the identity |
| `LORRAX_DEBUG_SHARED_POLE_EVEN_PART` | debug-only odd-channel diagnostic; refuses production misuse |

Restart restores the invariant ISDF basis; the map-local W models are
scratch and are never published as reusable ISDF members. A self-consistent
map rebuilds the bank, moments, directions, poles and ranks from the current
wavefunctions, energies and occupations; retained quadrature rules keep
nodes and weights while recertifying masks, selectors, reference energies and
`W(τ)` for the current domains. The full Γ head supports only the
time-reversal-even, `N_spinor = 1` body and refuses an ordered or
two-component deck at input resolution (`shared_pole_head.refuse_unsupported_shared_pole_head`);
such a one-shot deck uses `head_correction = off`.

## 12 Evidence

The measured accuracy of the route — the pole-count/accuracy table, the
certified Si/Na/CrI₃ self-energies, the odd-channel magnitudes, and the
open items — is the
[theory page's Sections 8 and 9](../theory/shared-pole-w-model.md#shared-pole-pole-count).
Implementation performance measurements (batched construction, round programs,
the face-block pencil, network timings) are owned by the campaign reports
under `reports/shared_pole_model_2026-09-15/` and
`runs/frequency_integration_sandbox/460_batchw_20260917/REPORT.md`; this page
does not restate them.
