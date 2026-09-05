# Four-current (bispinor) wiring

**What this page is for.** The four-current layer touches sixteen source
files across preprocessing, ISDF, screening, Σ and output. This page is the
map: for each stage, the file and function, the object it produces with its
shape and sharding, which route it belongs to, and what refuses. Read it
before touching anything under `bispinor`; it should take about ten minutes.

**What this page is not.** It states no physics. The Γ-cell physics, the
frequency treatment and what is zero by construction are owned by
[Four-current heads and frequency](../theory/four-current-head-corrections.md).
Deck-key semantics are owned by the [input reference](../input_reference.md).
Module one-liners are owned by [Codebase](codebase.md). Where those pages
disagree with this one, they win — this page owns the *wiring* only.

> **Pinned to integration tip `34228021` (2026-09-02).** The sole phase and
> coverage statement is on the
> [theory page](../theory/four-current-head-corrections.md#four-current-phase-status).
> Every source citation below was re-read at this tip: the line is a locator,
> while the named function or gate is the durable owner.

## The routes

`bispinor_gw` has two values on one four-spinor carrier. It selects which
Lorentz blocks are screened and which Sigma owner contracts them; it does not
select a carrier.

There is one packed static photon **operator**. Three deck situations reach it:

| route | selected by | current `χ` blocks | `Σ^B` contracted by | marked below |
|---|---|---|---|---|
| **packed, screened** | `bispinor_gw = full_static_cohsex` | all sixteen `χ^{IJ}` built, one packed Dyson solve | `gw.photon_sigma`, TT part of the sixteen-block `Σ_X` | **P** |
| **packed, bare static** | `bispinor_gw = bare_transverse`, `compute_mode = cohsex`, and the envelope below | twelve current-response blocks declared **zero**; packed solve skipped, CC screened by the scalar owner and spliced in, so `W_packed = diag(W_00, D_TT)` and `W_CT = 0` | the same `gw.photon_sigma` | **P** |
| **packed, bare dynamic** | `bispinor_gw = bare_transverse`, `compute_mode` in `{gn_ppm, hl_ppm, mpa}`, and the envelope below | packed CC block **absent**, not zero-filled; no packed W carrier; `W_CURRENT = V_CURRENT` | the same `gw.photon_sigma`, current blocks only | **P** |
| **incumbent, bare** | `bispinor_gw = bare_transverse`, **outside** the envelope | none — the bare `D^{ij}` tiles are contracted directly | `gw.sigma_x_bispinor` | **B** |

Orthogonally to *which* packed operator is built, `compute_mode` decides
**how much of Σ that operator owns** — one axis, two values:

| compute mode | packed Σ blocks | charge Σ | predicate |
|---|---|---|---|
| `cohsex` | all sixteen (`blocks = "all"`) | none beside it: the CC block of the packed operator **is** the charge Σ | `packed_photon_replaces_charge_sigma` |
| `gn_ppm`, `hl_ppm`, `mpa` | the twelve with a current index (`blocks = "current"`), evaluated once at `ω = 0`; the sole consumer enforces `W_CURRENT = V_CURRENT` | the ordinary scalar `Σ_x + Σ_c(ω)` owner, with its head and GN/HL role-W or MPA fit transaction unchanged | `uses_dynamic_packed_photon_route` |

The absent-CC layout is what admits `mpa`: it needs no independent static role
because no packed consumer reads CC. `gw.mpa.sigma` remains the charge owner,
including ordered residues and `sigC_odd`. Screened-current MPA does need a
static CC role and refuses as `packed_screened_mpa_static_role_unimplemented`.
The packed dynamic route's run-record line is `Photon Sigma`, beside `Photon
route` and `Photon head`.

**ONE envelope table at `34228021`.**
It used to be written twice — six conditions in
`packed_bare_transverse_route` and seventeen in
`refuse_unsupported_bispinor_gw`, five of them restated with separately
formatted `got`/`want` strings. Both now walk
`gw_config.packed_static_envelope(config, *, screened)` (`:3655`), which
yields `(accepted, got, want, klass, why, derived_key)`:

| # | row | applies to | class |
|---|---|---|---|
| 1 | `compute_mode ∈ {cohsex, gn_ppm, hl_ppm, mpa}` (`PACKED_PHOTON_COMPUTE_MODES`) | both | IMPLEMENTATION LIMIT — `cohsex` is static; all three dynamic modes keep charge with their scalar owner and use the absent-CC current layout. Screened MPA is separately refused because that class consumes static CC |
| 2 | `qp_solver = one_shot_dft`, or `self_consistent` for bare dynamic mode | both | IMPLEMENTATION LIMIT — the bare current operator is orbital-independent and reused while each SC map re-contracts rotated orbitals; charge-consuming and screened-current maps have named refusals |
| 3 | `screening_diagrams = w_rpa` | both | IMPLEMENTATION LIMIT |
| 4 | `head_correction ∈ {full, off}` | both | PHYSICS/POLICY (row 20) |
| 5 | `restart = false` | both | IMPLEMENTATION LIMIT — normally unreachable on a slab deck, which `GATE bispinor_slab_cohsex_restart_changes_the_head_mechanism` (`:3895`) refuses at parse time |
| 6 | `low_mem_bands = true` | screened | IMPLEMENTATION LIMIT, **derived when unnamed** |
| 7 | `linalg = distributed` | screened | IMPLEMENTATION LIMIT, **declared** |
| 8 | no scalar q→0 head override named (`scalar_head_overrides_named`, `:3621`) | screened | IMPLEMENTATION LIMIT — ONE conjunct for the eight `use_bgw_vcoul` / `wcoul0_*` / `*head*` / `mc_average_placement*` keys, naming only the ones the deck set |

Row 6 may be **derived when unnamed**: `LorraxConfig.from_input_file` sets it
for `full_static_cohsex` with a `[config provenance]` line, promoting only
when every other row already passes, so a deck outside the envelope for
another reason still sees *that* reason. Row 7 is declared: the global
`linalg` default remains `local`, and a packed route must explicitly request
`distributed`. Material class is also outside the
table: the driver infers it from the loaded WFN occupations, and
`validate_material_inputs` refuses every fractional-occupation non-MPA run
before screening. `sys_dim = 2` is deliberately **not** in the table: the bare route treats it as a routing condition
(`packed_bare_transverse_route`, `:3720`) while the screened mode refuses it
only under `head_correction = full` (`GATE
static_bispinor_photon_head_slab_only`, `:4055`), and one row cannot say
both.

The route predicate returns **`(taken, reason)`**, not a bool: the driver
prints the first unmet condition into the run record as the `Photon route`
line, because the two routes differ in the `q → 0` head mechanism and
switching physics silently is the hazard the reason exists to prevent.

`bispinor_tt_head_correction` **is no longer a deck key** (removed
2026-09-01, refused by name in `read_lorrax_input`). The double-count
refusal survives as ONE gate covering *both* packed modes
(`GATE packed_bare_transverse_tt_head_double_count`, `gw_config.py:4019`,
keyed on `uses_static_photon_response`), reachable only from a hand-built
config: the Γ-cell completion already inserts `⟨D_TT⟩` and honouring an
overlay too would double count it.

The bare route additionally requires `linalg = distributed` as a routing
condition. The default `local` retains the incumbent route; an explicit
`distributed` selects the packed route on a supported multi-rank mesh.
Because `distrib_la` refuses a distributed solve on a 1×1 mesh, one-GPU
operation stays on the local incumbent route. The packed completion is the
only deck-reachable producer of a transverse q→0 head.

The three predicates, all in `src/gw/gw_config.py`:

| predicate | line | answers |
|---|---|---|
| `packed_bare_transverse_route(config) -> (taken, reason)` | `:3743` | does a bare-transverse deck take the packed operator, and if not, why not |
| `packed_photon_screens_current(config) -> bool` | `:3810` | **the one selector between the two packed modes**: `True` only for `full_static_cohsex` |
| `uses_static_photon_response(config) -> bool` | `:3824` | is the packed operator used at all — `full_static_cohsex` always, plus the bare route inside its envelope |
| `packed_photon_replaces_charge_sigma(config) -> bool` | `:3838` | does the packed operator own the CHARGE Σ too (`cohsex`), or only the current blocks — the question every driver seam that skips the scalar charge machinery must ask |
| `uses_dynamic_packed_photon_route(config) -> bool` | `:3859` | the packed operator on a frequency-dependent Σ |

`uses_coupled_photon_head` (`:3873`) adds `head.correction is FULL` on top and
decides **only** whether `gw_init` keeps the four literal-Γ channel vectors.

**`BispinorGWMode` has TWO members** (`:289`), `bare_transverse` and
`full_static_cohsex`, and they resolve to the SAME carrier — the axis picks
which Lorentz blocks are screened, never which four-spinor represents them.
Three retired spellings refuse by name from ONE table,
`_RETIRED_BISPINOR_GW_MODES` (`:324`, read by `coerce_bispinor_gw_mode`
`:353`): `charge_hall_cubature` → `full_static_cohsex`, and the two
carrier-comparison modes `pauli_reference_bare_transverse` and
`isometric_kinetic_balance_bare_transverse` → `bare_transverse`. Refusals,
not aliases: the coercer runs from a dozen resolution sites, and a mode value
is the one deck word that decides which physics runs.

**A row marked P below applies to both packed modes** unless it says
otherwise; **B** means the incumbent route only. Rows marked "both" are
common to every bispinor deck.

**What the packed bare route is worth, measured** (MoS2 3×3, 270 states,
`reports/bisp_c_bare_as_packed_2026-09-01`, claim 581): with the head off it is
**byte-identical** to the incumbent route — `max|dE_qp| = 0.000 µeV`, every
`sigma_diag.dat` data row identical — despite a completely different
contraction order and operator packing. Declaring the twelve current `χ` blocks
zero costs **0.012 µeV** against the screened packed mode. The former head-on
difference of **5.4 meV MAE / 11.9 meV max** compared different Γ-cell
quadratures as well as different insertion rules. Both routes now use the
Wigner–Seitz polygon cubature; the remaining insertion distinction is owned
by [theory §3.6](../theory/four-current-head-corrections.md).

## The map

```mermaid
flowchart TD
  subgraph DECK["deck (cohsex.in)"]
    K1["bispinor · bispinor_gw<br/>centroids_file_current"]
    K2["head_correction<br/>static_gauge_hall_file"]
    K3["transverse_zeta_solve / _rcond<br/>distrib_la_batched_route"]
  end

  subgraph CFG["config — gw_config.py, common/four_current_model.py"]
    R["resolve_four_current_representation<br/>FourCurrentRepresentation:<br/>carrier + lift + stamps"]
    F{"uses_static_photon_response"}
    S{"packed_photon_screens_current"}
    E["refuse_unsupported_bispinor_gw<br/>packed_static_envelope (9 rows)<br/>packed_bare_transverse_route"]
  end

  subgraph INIT["initialization — gw_init.py"]
    C1["charge centroids (n_mu_C,3)"]
    C2["current centroids (n_mu_T,3)<br/>+ meta_transverse"]
    L["bispinor lift<br/>common/bispinor_init.lift_to_4spinor<br/>(nk,nb,4,ngk)"]
    Z0["zeta_q.h5 — charge fit"]
    ZT["zeta_q_mu1/2/3.h5<br/>three transverse fits<br/>coupled schedule"]
    VQ["v_q_bispinor.h5<br/>7 unique of 16 D tiles<br/>optional bare TT head slot"]
  end

  subgraph SCR["screening"]
    CHI["16 no-pair chi0 blocks<br/>w_isdf.compute_experimental_no_pair_photon_chi0<br/>TT Ward-subtracted"]
    PK["photon_layout: pack C+T1+T2+T3<br/>mesh-interleaved, N_packed"]
    DY["distributed Dyson<br/>solve_w via distrib_la solve_lu"]
    HD["Gamma completion<br/>head_correction.complete_static_slab_photon_q0<br/>static_gauge_response + Hall artifact"]
    SW["scalar charge head<br/>HeadResolver / qsgw_head"]
  end

  subgraph SIG["self-energy"]
    PS["photon_sigma.compute_static_photon_sigma<br/>16-block contraction"]
    SB["sigma_x_bispinor.compute_sigma_x_bispinor<br/>9 bare (i,j) tiles"]
    CS["scalar charge Sigma<br/>cohsex_sigma / ppm_pipeline"]
    HT["_compute_live_hartree<br/>four-spinor G-space V_H + transverse"]
  end

  OUT["SigmaResult.photon_head_sigma_*<br/>gwjax.out · sigma_freq_debug · sigma_diag.dat"]

  K1 --> R
  R --> F
  K2 --> E
  K3 --> ZT
  F -->|"packed (either mode)"| S
  F -->|"outside the envelope"| SB
  S -->|"screened: 16 chi blocks"| CHI
  S -->|"bare: chi_TT = chi_CT = 0"| PK
  R --> C1
  R --> C2
  R --> L
  C1 --> Z0
  C2 --> ZT
  L --> Z0
  L --> ZT
  Z0 --> VQ
  ZT --> VQ
  K2 --> VQ
  VQ --> CHI
  VQ --> SB
  VQ --> PK
  CHI --> PK
  PK --> DY
  DY --> HD
  HD --> PS
  SW -->|incumbent and packed dynamic charge| CS
  PS --> OUT
  SB --> OUT
  CS --> OUT
  HT --> OUT
```

## Stage 1 — deck to config

Parsing and defaults are the input reference's; the wiring facts are which
object each key ends up on and who reads it.

| key | default (`gw_config.py`) | lands on | route |
|---|---|---|---|
| `bispinor` | `False` (`:1690`, parse `:5947`) | `config.bispinor` (`:5050`) — the master switch | both |
| `bispinor_gw` | `bare_transverse` (parse via `coerce_bispinor_gw_mode` `:353`; enum `:289`, **two** members; three retired spellings in `_RETIRED_BISPINOR_GW_MODES` `:324`; config construction `:5948`) | `config.bispinor_gw` | both |
| `centroids_file_current` | `""` (`:1514`, parse `:5435-5443`) | `config.paths.centroids_file_current` (`:3274`) | both |
| `head_correction` | `full` (`:2011`, coerced `:548`, read `:5451`, built into `HeadConfig` `:5474`) | `config.head.correction` (`:5064`) | both |
| ~~`bispinor_tt_head_correction`~~ | **REMOVED as a deck key 2026-09-01**; the field is wired to `False` (`gw_config.py:5495`) for the incumbent V-tile builder | `config.head.bispinor_tt_head_correction` | **B only**, and now only from a hand-built config |
| `static_gauge_hall_file` | `""` — EMPTY (`:1503`) | `config.paths.static_gauge_hall_file` | P; optional. An UNNAMED file means `σ_H = 0`, announced. Every named path is authenticated by the loader; an absent one refuses there. The bare route accepts only an exact-zero artifact and refuses a nonzero Hall vector |
| `linalg` | `local` | the parser-cached backend profile (including `config.backend.transverse_zeta_solve`) | both |
| `transverse_zeta_rcond` | `1e-10` (`:1916`, validate `:5741-5745`) | `config.backend.transverse_zeta_rcond` (`:4774`) | both |

The retired stage/backend keys refuse by name; implementation-specific
overrides, where available, are debug-only CLI or `LORRAX_*` controls.

**The one resolver.** `common/four_current_model.resolve_four_current_representation(bispinor, model)`
returns a frozen `FourCurrentRepresentation` with seven fields: `charge_bispinor`,
`charge_lift`, `current_bispinor`, `current_lift`, `scalar_head_bispinor`,
`charge_representation`, `spatial_current_representation`. **It is not stored
on the config** — its consumers call it locally (`gw_config.py:378`,
`gw_init.py:799,1459,1984,3106`, `sigma_dispatch.py:309`,
`sc_iteration.py:1588,1736`, `head_correction.py:571,1613`,
`qsgw_head.py:3394`, `file_io/kin_ion.py:389`, and
`psp/get_dipole_mtxels.py:1093`), so grep for
`resolve_four_current_representation`, not for a config attribute.

**THREE branches (2026-09-04)**: non-bispinor (everything False),
bispinor/kinetic (both lifts `raw`, `scalar_head_bispinor=True`) and
bispinor/velocity (`current_lift="velocity"`, the FAMILY name;
`current_lift_for(mu_L)` gives the per-channel carrier selector
`velocity_1/2/3`; `spatial_current_representation=
"velocity_kinetic_balance_per_channel_alpha_spatial_current_v1"`; charge
side unchanged).  The two shipped `bispinor_gw` values ride the same
carrier, so `model` is accepted and ignored; the one carrier dial is the
deck's `bispinor_current_balance`, passed as `current_lift=` by every call
site that owns a current-channel object (`gw_init` ×4, `sigma_dispatch`,
`sc_iteration` ×2, `head_correction`, `file_io/kin_ion`,
`psp/get_dipole_mtxels`).  The velocity lift needs `dV_NL/dk`: `gw_jax`
builds the projector setup from `pseudo_dir` and attaches
`psp.vnl_ops.nonlocal_velocity_lift(setup)` to `WfnLoader.nonlocal_velocity_lift`;
the loader routes its own G table, k representatives, `ngk_valid` mask and the
channel to it and hands the channel's ket `Σ_b σ^b (dV_SR/dk_b ψ_L) + σ^a
(dV_SO/dk_a ψ_L)` to `lift_to_4spinor(representation="velocity_a")`, which adds
`(α/4)` of it (Ry→Hartree ½) to the σ·p small component.  ONE CARRIER PER
CHANNEL: `gw_init._transverse_wfn_data` samples the transverse centroids once
per distinct lift (three loads under velocity, one under kinetic) and the
σ^B-side bundles become a `gw.wavefunction_bundle.LorentzCarriers`, whose
`channel(mu_L)` is the endpoint every block builder uses — `w_isdf`'s
`families = (charge, T1, T2, T3)`, `photon_sigma._bundle_for_channel`,
`sigma_x_bispinor` through `endpoint_bundles(left, right, i, j)` (G-direct and
bra from carrier i, G-conjugated and ket from carrier j).  The transverse ζ of
channel a is fit on carrier a; restart tensors store channels 2 and 3 as
`psi_full_y_transverse_c2/_c3` (+`_mun`) beside the historical channel-1 names
and stamp `transverse_carriers`/`transverse_lift_family`, and a run refuses a
file written under the other family.  `WfnLoader.lift(psi_2, k=, bispinor_lift=)`
lifts an already loaded two-spinor window into any carrier; the dipole
producer's finite-q α vertex uses it for channel a's carrier at both endpoints.
The static-gauge Hall producer is a different, exact construction of the same
current (σ·p carrier plus the explicit ICL projector jet) and refuses a
velocity carrier (`uniform_gauge_operator`), because the jet would then be
counted twice.  Two places still share ONE carrier between ρ and the Dirac
current — the exact Hartree (`sigma_dispatch` → `kin_ion_io.compute_hartree_matrix`)
and the SC density rebuild (`sc_iteration`) — and keep the charge lift for
both, announced with a printed notice. The two carrier-comparison branches went with their deck spellings.
It stays a resolver rather than collapsing into `bool(bispinor)` because the
`charge_representation` / `spatial_current_representation` provenance stamps
the `dipole.h5` / `kin_ion.h5` / ζ authenticators compare against need ONE
producer. The lift selectors are `RAW_KINETIC_BALANCE_LIFT = "raw"` and
`VELOCITY_KINETIC_BALANCE_LIFT = "velocity"` (deck-reachable, the latter for
the spatial-current carrier only); `ISOMETRIC_KINETIC_BALANCE_LIFT =
"isometric"` (`src/common/bispinor_init.py`) remains library code for the
jet tests.

**`static_bispinor_photon_envelope` is a gate id, not a function.** It is
the raise at `gw_config.py:3997`, over the eight rows of
`packed_static_envelope` (the table in Stage 1). **Eight**, down from
seventeen (`lane/bisp-l-dials-envelope-2026-09-01`), and each unmet row
prints `PHYSICS` or `IMPLEMENTATION LIMIT` with its own reason. `full` is the
default and runs the Γ completion, `off` is a DEBUG skip, and
`no_local_fields` is refused on EVERY bispinor deck
(`GATE bispinor_head_correction_no_local_fields_unavailable`, `:3943`) —
the coupled solve has no scalar diagnostic head, and on the bare route the
value used to move the deck off the packed path, which is a head dial
choosing a route. A separate gate,
`GATE static_bispinor_photon_head_slab_only` (`:4055`), refuses
`sys_dim != 2` **while the completion is on**; the DEBUG headless body keeps
the bulk reach the old envelope gave it.

## Stage 2 — initialization (`gw_init.py`)

| object | producer | shape / dtype | sharding | route |
|---|---|---|---|---|
| charge centroid indices | `gw_jax.py:392-395` → `file_io/centroids.load_centroid_basis` (`:122-174`) | `(n_mu_C, 3)` i64 | host | both |
| current centroid indices | `gw_init.py:1429-1441` (refusal `:1425-1428` if the key is unset) | `(n_mu_T, 3)` i32 | host until needed | both |
| `meta_transverse` | `gw_init.py:1442-1447` | `Meta` with `n_rmu = n_mu_T`, `nspinor = 4`, `npol = 4`, `n_rmu_padded = padded_mu_extent(...)` | — | both |
| four-spinor ψ | `common/bispinor_init.lift_to_4spinor` (`:234`) | `(n_k, nb, 4, ngkmax)` c128, `[ψ_L; ψ_S]` on the spinor axis | caller wraps; no sharding inside | both |
| face ψ carriers | `common/wfn_layout.py:12-13`; transverse copies `gw_init.py:2412-2419` | `(nk, n_X, s, mu_Y)` / `(nk, s, mu_X, n_Y)` | `P(None,'x',None,'y')` / `P(None,None,'x','y')` | both |
| charge ζ | `gw_init.py:2209-2239` → `isdf_fitting.fit_zeta_to_h5` | `tmp/zeta_q.h5`, `(n_q_disk, n_mu_C, ngkmax)` c128 | accumulator `(n_q_disk, n_mu_padded, ngkmax)` at `P(None,('x','y'),None)` (`isdf_fitting.py:1404-1409`) | both |
| three transverse ζ | `gw_init.py:2448-2499` (`_fit_transverse_channel`, `vertex_mu_L = 1,2,3`) | `tmp/zeta_q_mu{1,2,3}.h5`, `(n_q_disk, n_mu_T, ngkmax)` c128 | same | both |
| `C_q` (CCT Gram) | `isdf_fitting.py:743-751` | `(nq, n_mu_padded, n_mu_padded)` | `P(None,'x','y')` — layout contract `isdf/core.py:4960-4966` | both |
| bare `D^{IJ}` tiles | `v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5` (`:261-581`) | `v_q_bispinor.h5`, 7 datasets `(n_q_total, n_mu_L, n_mu_R)` c128 | device buffer `P(None,'x','y')` (`v_q_g_flat.py:492-494`) | both |
| tile reader | `v_q_bispinor.BispinorVqReader.get_tile` (`:682-716`) | `(n_q, n_L_padded, n_R_padded)` c128 | `P(None,'x','y')`; Hermitian companions read at `P(None,'y','x')` then conj-swapped (`:704-709`) | both |
| `photon_g0_vectors` | `v_q_bispinor.py:575-581` | 4-tuple, each `(n_q_full, n_mu_padded)` c128 | `P(None,'x')` | released to `None` unless `uses_coupled_photon_head` (`gw_init.py:2862-2864`) — **P** |

**Sixteen tiles, seven on disk.** Six `(0,i)`/`(i,0)` tiles are exactly zero
by Coulomb gauge (`ZERO_TILES`, `v_q_bispinor.py:67-70`); three `(j,i)` with
`i < j` are Hermitian companions reconstructed on read (`HERMITIAN_PAIRS`,
`:72-78`); the remaining seven — CC plus six TT — are computed and written
(`UNIQUE_TILES`, `:60-65`). Format stamp `bispinor_lorentz_v2` (`:80`),
artifact path `gw_init.py:2786`.

**The coupled ζ schedule.** The Y-side face transform and the full-spin X
broadcast are channel-independent, so μ_L = 1,2,3 share one
built-once-per-r-chunk `[3, q, μ, r]` `Z_q` stack instead of repeating it
three times (`isdf/core.py:2620-2626`). Entry `gw_init.py:2520-2550`
(coordinator `:1745-1922`, route chooser `:1924-1947`, capacity block
`:2082-2189`);
kernel `isdf/core._z_q_face_coupled_mu123` (`:2795-2833`). The coupled
out-spec is `P(None,None,'x','y')` against the scalar `P(None,'x','y')`
(`isdf/core.py:2420-2421` vs `:2337-2340`). `distrib_la_batched_route` reaches
this fit through `gw_init.py:2076-2077` → `:2465`, so the same key that
governs generic batched linalg governs the transverse ζ schedule.
Full design: [Face-ψ ζ fitting](zeta_fit_face_psi_cct.md).

**The bare TT head slot** (route B). ONE gate refuses it on either packed
route now that the deck key is gone:
`GATE packed_bare_transverse_tt_head_double_count` (`gw_config.py:4019`,
keyed on `uses_static_photon_response`), reachable only from a hand-built
config. Config plumbing `gw_init.py` →
`v_q_bispinor.py:423` → the per-tile builder
(`_make_per_q_v_builder_for_tile`, `:141-253`). The tensor
`T_ab = ⟨v(q) P^T_ab(q̂)⟩_mBZ` is `(3,3)` f64 from `_tt_head_tensor`
(`:100-138`), computed **once per tile, not per q** (`:107-109`), and refused
outside `sys_dim ∈ {2,3}` (`:118-125`). The injected value is `T[i,j]/Ω_cell`
(`:224` — the `/Ω` is applied here, not in the shared `vcoul` estimator),
substituted into the unique `q = Γ, G = 0` slot (`:240`, `:250`), after which
the single spatial-metric sign is applied once on the way out (`:251`,
`COULOMB_GAUGE_TT_SIGN = -1.0` at
`services/vcoul/src/vcoul/minibz.py:119`). No vertex and no Σ contraction
compensates that sign.

## Stage 3 — screening

### 3a. The packed body (route P — both packed modes)

`w_isdf.compute_static_photon_response` (`:1863-2279`) is the screening owner
of both packed modes. Its required `screen_current` argument comes from
`gw_config.packed_photon_screens_current`; `config` is required and the head
record is built internally, so a caller cannot inject a competing head owner.

`screen_current` is resolved once, by
`gw_config.packed_photon_screens_current`, and is never defaulted here; a
value inconsistent with `bispinor_gw` is refused (`:1962-1993`), and
`screen_current = False` without a `W_charge` is refused at `:1981-1985`. It
selects:

* **`True`** (`full_static_cohsex`) — the sixteen no-pair `χ` blocks and one
  distributed Dyson solve, below.
* **`False`** (the bare-transverse family on the packed route) — the twelve
  current blocks are zero **by declaration**, so neither they nor the packed
  solve are built at all — the branch is `:2150-2193`, against the screened
  branch at `:2135-2149`. The CC block is screened by
  the incumbent scalar owner (`gw.screening.compute_screening_model` →
  `solve_w` at `n_C`, called by the driver at `gw_jax.py:695`) and
  arrives as `W_charge`; this function assembles
  `W_packed = diag(W_00, D_TT)` with `W_CT = 0` through the sole packer.

Both modes then run **one** Γ-cell completion (§3b). The per-rank resident
byte figure for the packed body is measured and printed at the site
(`:2112-2134`, the print itself `:2129-2134`): `16·n_k·N_packed²/P` bytes each for `V_packed` and
`W_packed`. That carrier is **new to the bare route** — the incumbent route
held one TT tile at a time — and is the same object the screened mode
already held.

| object | producer | shape | sharding |
|---|---|---|---|
| one no-pair block `χ^{IJ}_0` | `w_isdf.compute_no_pair_dirac_current_block` (`:1665`) | `(nq, μ_L, μ_R)` c128 | `P(None,'x','y')` |
| the sixteen blocks (`screen_current = True` only) | `compute_experimental_no_pair_photon_chi0` (`:1788`); T1/T2/T3 share one transverse bundle | — | — |
| TT Ward subtraction | `_subtract_static_tt_contact` (`:1776`), applied inside the sixteen-block builder | `Π(q) − Π(0)` | in place, buffer donated |
| packed `V`, `χ_0`, `W` | `photon_layout.pack_photon_operator` (`:219`), `w_isdf.solve_w` (`:1431`) | `(nq, N_packed, N_packed)` c128 | `P(None,'x','y')` |

`N_packed` is `PhotonBasisLayout.packed_extent` (`photon_layout.py:107-109`),
`p_C + 3·p_T` over the mesh-padded centroid extents built by
`from_centroid_extents` (`:88-101`); one transverse centroid family must serve
T1/T2/T3, and `__post_init__` (`:69`) refuses otherwise. The packing is
**mesh-interleaved**, not a contiguous direct sum: packed row shard `x` holds
that shard's own local `C,T1,T2,T3` row chunks, and the column shard applies
the same permutation (`photon_layout.py:24-31`, offsets `local_offset`
`:119-123`). A packed operator is therefore `P O Pᵀ`, Dyson algebra is
unchanged, and both `pack_photon_operator` and `photon_block_view` (`:279`)
are local `shard_map` slices with no redistribution. Ordering stamp
`PHOTON_BASIS_ORDERING = mesh_interleaved_direct_sum_v1` (`:55`).

The sixteen-block enumeration lives in `pack_photon_operator`'s double loop,
which calls the block builder once per `(A,B)` and `block_until_ready()`s each
(`:247`) so only one block is resident.

**The Dyson solve** is `solve_w` (`w_isdf.py:1431`) with
`dyson_solver="distributed"` on this path (`:2141-2144`, inside the
`screen_current = True` branch), dispatching to
`_get_w_solve_fn_distributed` (`:1164`), which plans through the `distrib_la`
service (`plan("solve_lu", mesh_xy, backend="distributed", …)` at `:1259-1260`,
executed at `:1328`). Inputs, the assembled `A`, the LU factors and `W` all
stay `P(None,'x','y')`. The source states the invariant at `:1191-1195`: **no
rank ever materialises a full `(μ, μ)` tile** — the largest per-rank transient
is the `μ·μ/min(Px,Py)` gathered GEMM operand.

`StaticPhotonResponse` (`w_isdf.py:1846`) carries `layout`, `V_packed`,
`W_packed` (both as above), `head_completion`, and three provenance stamps
whose values **differ by route** — read them to tell which route produced an
artifact:

| stamp | screened (`full_static_cohsex`) | packed bare |
|---|---|---|
| `current_contact` | `ward_subtracted_no_pair` | `none: current channels unscreened` (`STATIC_PHOTON_BARE_CURRENT_CONTACT`, `:1842`) |
| `current_model` | `positive_energy_kinetic_balance_dirac_current_v1` (`common/bispinor_init.py:32`) | `bare_breit_no_current_response_v1` (`STATIC_PHOTON_BARE_CURRENT_MODEL`, `:1841`) |
| `approximation` | `gamma_completed_no_pair_static_photon_v1` / `DEBUG_headless_no_pair_static_photon_v1` (`:2259-2269`) | `gamma_completed_bare_transverse_photon_v1` / `DEBUG_headless_bare_transverse_photon_v1` (`:2270-2278`) |

There are **four** `approximation` strings: one per route × head state.

Under the DEBUG setting the module prints a boxed
`WARNING -- DEBUG: Gamma-cell head disabled by head_correction=off`
(`:2092-2098`); the `WARNING` token is what the production sink retains into the
run record's warning block.

### 3b. The Γ completion (route P — both packed modes, on by default)

`head_correction.complete_static_slab_photon_q0` (`:1298-1307`, called at
`w_isdf.py:2241`) takes
`V_packed`, `W_packed`, a **sealed** response record, the two
`(4, N_packed)` `g_0` vectors (`g0_X` at `P(None,'x')`, `g0_Y` at
`P(None,'y')`) and a cubature receipt, and returns updated `V`, `W` plus a
`StaticSlabPhotonHeadCompletion` (`:1130`, fields `:1133-1151`, carrying
`sigma_H` `:1149` and `hall_source` `:1150` alongside the certificates). The
old `isinstance` fork over two response types is gone: the single path is
`require_static_photon_head_response(response, mesh_xy)` at `:1326`. It runs whenever
`head_correction = full`, which is the default — it is no longer opt-in.

| input | producer | shape | sharding |
|---|---|---|---|
| cubature receipt | `vcoul.slab_minibz_photon_cubature` (`services/vcoul/src/vcoul/minibz.py:821`) — exact Wigner–Seitz polygon, fixed 16/24/32 Duffy–Gauss ladder (`:124`) | host chunks: `q_cart (n,3)`, `D_raw (n,4,4)`, `sample_weight (n,)` f64 | host, write-locked |
| `StaticPhotonHeadResponse` (`:64`, sealed — only its producer can build one) | `static_gauge_response.build_static_photon_head_response` (`:169-293`, called at `w_isdf.py:2215`); `require_static_photon_head_response` (`:103-155`) | `S_direct (2,2,4,4)`, `sigma_H (3,)` real, `hall_source` str, `Y_x (2,4,N_packed)`, `Z_y (2,N_packed,4)` | `P()`, `P()`, —, `P(None,None,'x')`, `P(None,'y',None)` |
| Hall artifact — **optional** | `file_io/static_gauge_head.load_static_gauge_hall_artifact` (`:180-189`), called at `w_isdf.py:2066-2074` behind the existence check at `:2019`; sole writer `write_static_gauge_hall_artifact` (`:130-135`) via `psp/get_dipole_mtxels.py` | `static_gauge_hall.h5`, schema 1 (`STATIC_GAUGE_HALL_SCHEMA_VERSION`, `:35`), `sigma_H_cart (3,)` f64 | replicated |

If no `static_gauge_hall_file` is present the builder sets `sigma_H = 0` and
`hall_source = HALL_SOURCE_NONE` (`static_gauge_response.py:51,214`; the shape/sharding contract is re-asserted at `:114-121`) and says
so; a present but mismatched artifact still **refuses** in the loader, so a
stale file cannot degrade a run silently. For a Chern-trivial insulator that
default is the exact answer — see
[the theory page](../theory/four-current-head-corrections.md) §4.4.

The updates are **one bare rank-4 outer product into `V`** (`:1480-1482`) and
**nine screened rank-4 outer products into `W`** (`:1491-1503`), all through
`photon_layout.add_photon_q0_low_rank` (`:556`), which takes left rows at
`P(None,'x')` and right rows at `P(None,'y')`, donates the packed buffer, and
does a purely local outer product (`:576-579`). Gates on this path:
`GATE static_gauge_head_fold_ward` (`:1365`) and
`GATE static_gauge_head_fold_hermiticity` (`:1370`), against
`_STATIC_GAUGE_WARD_RESIDUAL_MAX = 1e-8` and
`_STATIC_GAUGE_HERMITICITY_RESIDUAL_MAX = 1e-10` (`:158-159`), plus the
`static_photon_dyson_*` / `static_photon_polygon_*` numerical certificates
(`:1249-1295`, helpers `:1159-1246`, budget `1e-9` at `:1154`).

**What the mode declares it does not have.** The **module docstring of
`gw.static_gauge_response`** is the complete list. It replaced the
`CHARGE_HALL_CUBATURE_AVAILABILITY` availability grammar and its
`StaticGaugeTermAvailability` / `require_full_static_gauge_availability`
machinery, which are deleted. Present: the charge CC
`q²` head `S^{00}`, the charge wings `Y^0`/`Z^0`, and the Hall CT/TC `q¹`
term. Omitted by model: the current `q²` response (TT and CT/TC), the current
wings, the uniform static current response (zero by gauge invariance for an
insulator), the diamagnetic/contact terms and the complement-space closure —
and they are never stored as accidental zeros, because `S_direct` has charge
support only. What that list means physically is
[the theory page](../theory/four-current-head-corrections.md) §4.3.

**The dead seam is absent at `34228021`.** The former implementation carried
a producerless full-static-gauge seam — `StaticGaugeHeadResponse`,
`require_canonical_operator_fingerprint`, `require_static_gauge_head_response`
and its five `GATE static_gauge_head_*`, `LoadedStaticGaugeHeadResponse`,
`write`/`load_static_gauge_head_artifact`, the v2 head schema, the
`gauge_head_response=` argument, `GATE static_gauge_head_end_to_end_uncertified`
and the `GATE full_static_bispinor_gauge_head_unavailable` that made it
unreachable. Those symbols are deleted; `file_io/static_gauge_head.py` is now
the Hall artifact's format owner and nothing else.

### 3c. The scalar charge head (incumbent and packed dynamic charge)

The scalar head remains the owner for every incumbent route and for the CC
sector of packed GN/HL-PPM. `HeadResolver` (`head_correction.py:1589`) is built
at `gw_jax.py:533`; `head_correction = full` builds the direct DFT response at
`:625-665`, finalizes its samples at `:799-813`, and constructs the static
Sigma terms at `:852-872` through `_compute_static_head` (`:189`). Only static
packed COHSEX skips this machinery: its sixteen-block operator replaces the
charge Sigma and its coupled completion owns the charge head. The dynamic
packed route deliberately keeps the scalar dynamic charge head and adds only
the current-index blocks from the packed completion.

## Stage 4 — self-energy

The fork is `sigma_dispatch.py:845`, and it has **three** packed arms plus
an exhaustiveness refusal:

| arm | line | what it does |
|---|---|---|
| `packed_photon_replaces_charge_sigma` (`compute_mode = cohsex`) | `:845-889` | the sixteen-block contraction owns `Σ_X`, `Σ_SX`, `Σ_COH` outright. Its three refusals fire before any allocation: outside `compute_mode = cohsex`, with no packed response ("Refusing instead of falling back to charge-only screened COHSEX"), and with scalar `static_head_terms` present (double count) |
| `uses_dynamic_packed_photon_route` (GN/HL/MPA) | `:890-990` | the scalar `compute_sigma_x` runs with the incumbent `Σ^B` arms **explicitly off** (`wfns_transverse=None`, `bispinor_v_q_path=None`) and keeps `static_head_terms` (the CC bare-X head); the packed consumer is called with `blocks = "current"` and `charge_block_state = absent_dynamic_scalar_owner`. It aliases `W_CURRENT` to `V_CURRENT`, so `SX = X`, `COH = 0`, and books that static contribution into `sig_x`. Control then reaches the mode's unchanged GN/HL or MPA charge-Sigma branch |
| `uses_static_photon_response` with neither of the above | `:991-1008` | exhaustiveness refusal naming `PACKED_PHOTON_COMPUTE_MODES`; no parsed supported deck reaches it |

`finalize_dynamic_sigma` carries the packed per-sector Γ diagnostics through
to `sigma_freq_debug.dat` (`:342-549`, passed at `:1320`); on the dynamic
route the CC sector of those columns is exactly zero, because the charge head
there is the dynamic model's, not the packed completion's.

| object | producer | shape | sharding | route |
|---|---|---|---|---|
| sixteen-block `Σ_X`, `Σ_SX`, `Σ_COH` (twelve blocks under `blocks = "current"`) | `photon_sigma.compute_static_photon_sigma` (`:296`), called at `sigma_dispatch.py:871` (static) and `:946` (dynamic) | each `(nk, nb_sigma, nb_sigma)` | **replicated** `P(None,None,None)` at the output boundary | P |
| per-block operands | `photon_layout.photon_block_view` (`:279`), fetched at `photon_sigma.py:401-402` | `(nk_tot, n_left, n_right)` | `P(None,'x','y')` — a `dynamic_slice`, never a gather | P |
| Green function `G` | `greens_function_kernel.build_G`, face layout | `(nk, ns, μ_L, ns, μ_R)` | `P(None,None,'x',None,'y')` (`wavefunction_bundle.py:236`) | both |
| head-attribution diagnostics | `photon_sigma.py:410-414` (accumulated at `:450-471`) via `photon_layout.photon_q0_low_rank_block` (`:676`) | one `(1, p_A, p_B)` block | `P(None,'x','y')` | P |
| `Σ^B`, bare transverse — **incumbent route only** | `sigma_x_bispinor.compute_sigma_x_bispinor` (`:77`), nine `(i,j)` tiles at `:189-190`. Retained outside the packed envelope (including the current bulk, restart, x-only/resolvent, and explicit-local/one-GPU classes). Slab GN/HL/MPA, including the supported dynamic SC class, reach `Σ^B` through the packed current consumer instead; bare COHSEX SC refuses rather than falling back | `(nk, nb_sigma, nb_sigma)` | replicated after a gather-then-window (`:221-235`) | B |
| transverse Hartree | `sigma_dispatch._compute_live_hartree` (`:302-335`) → `kin_ion_io.compute_hartree_matrix` (`:759`) | `charge`, `transverse`, each `(nk_full, nb, nb)` Ry | `P(None,'x','y')` with `return_sharded=True` | **both** |

**The sixteen-block contraction** is a plain nested Python loop (`A` at
`photon_sigma.py:385`, `B` at `:389`), not a `vmap`. The `blocks` selection
is one `continue` inside it (`:390`) and appears nowhere else, so the static
and dynamic routes share every kernel, every certificate and the sector
closure gate. Per block the γ̃ vertices
are folded into the two G-build operands only (`:355`, `:359`), and the same
jitted kernel is called three times with a dynamic `term` selector for X, SX
and COH (`:420-443`; COH uses `W − V` with prefactor `−0.5`). Each accumulator
is `block_until_ready()`d before advancing (`:429`, `:439`, `:476`) — the
source names this the lifetime boundary that prevents two W/G body tiles from
coexisting (`:473-476`).
`GATE photon_head_sigma_sector_closure` (`:515-520`) then checks that
CC + (CT+TC) + TT closes on the direct sixteen-block total. The diagnostics
record `StaticPhotonHeadSigmaDiagnostics` (`:62-75`) has three fields.

**`Σ^B` enters twice, differently** — on the incumbent route. In
`compute_cohsex_sigma` it is added to **both** `sig_x` and `sig_sx`
(`cohsex_sigma.py:660-668`); in `compute_sigma_x` — the entry every dynamic
mode and `x_only` takes — it is added to `sig_x` only (`:735-742`). On
either packed route neither fires: `Σ^B` is the TT part of the packed `Σ_X`
instead, and `sigma_x_bispinor` is not called — on the dynamic route the
`compute_sigma_x` call passes `wfns_transverse=None` explicitly so the arm
cannot fire by accident. `photon_sigma` and `sigma_dispatch` needed **no
change** for the packed *bare* route — the sixteen-block consumer streams
whatever `V`/`W` it is handed — and needed only a block selector and one
branch for the *dynamic* one.

**No transverse operand reaches a dynamic Σ_c, and that is the design.**
`compute_ppm_sigma_pipeline` (`ppm_pipeline.py:473-490`) and the MPA body
(`mpa/sigma.py:283-303`) have no `wfns_transverse`, `bispinor_v_q_path` or
`photon_response` parameter, and they gained none in the minimal dynamic
packed route splits Σ instead: the charge half is those untouched kernels on
`W_00(ω)`, the current half is the packed consumer at `ω = 0`, and the two
meet only in the `sig_x` addition at `sigma_dispatch.py:973`. So the
four-current layer still reaches a dynamic Σ through `sig_x` and the
transverse Hartree and nowhere else — what changed is which owner computes
the `sig_x` transverse part, not how many owners there are.

**The transverse Hartree is on both routes.** Its gate is
`include_transverse = bool(config.bispinor)` (`sigma_dispatch.py:311`), not
`bispinor_gw`, so every bispinor mode and every compute mode gets it unless
`omit_v_h` (density self-consistency, which rebuilds both fields itself). It
is added through `_compute_live_hartree` at `:1045-1067`. Physics owner:
[Direct Hartree field](../theory/hartree.md).

## Stage 5 — outputs

`SigmaResult` (`sigma_dispatch.py:62`) carries **two** such fields, `None` on
every non-packed route (initialized at `:843-844` and populated from the
packed contractions at `:887-890` or `:988-990`):

| field | line | shape / value |
|---|---|---|
| `photon_head_sigma_diag_tskn_ry` | `:124` | `(3, 3, nk, nb)` — axes `(term = X/SX/COH, sector = CC / CT+TC / TT, k, band)`, DFT basis |
| `photon_head_sigma_basis` | `:125` | `"dft"` when populated; also in `BASIS_FREE_FIELDS` (`:257`) |
| `sigma_lorentz_skij_ry` | `sigma_dispatch.py` | `(3, nk, nb, nb)` — physical `Sigma_xc` sectors `(CC, CT+TC, TT)`. Packed routes accumulate the computed blocks in `photon_sigma.py`; incumbent routes retain the computed `Sigma^B` as TT and define CC as the exact total-minus-current residual. It rotates with the total under static self-consistency. |
| `sigma_c_odd_at_dft_diag_ev` | `sigma_dispatch.py` | `(nk, nb)` only on a measured-broken-TR GN deck: `Sigma_c[B,D] - Sigma_c[B,D=0]` evaluated at each DFT state. `None` on measured-TRS decks. |

A former `photon_head_sigma_operator_fingerprint` field is absent at
`34228021`; it belonged to the removed full-static-gauge seam.

Mirrored onto the output record at `gw_output.py:149-150` and forwarded by the
driver at `gw_jax.py:1067-1069` (schema zeros `:1070-1072`, final record
`:1292-1294`). `sigma_freq_debug` gains `head_CC`, `head_CTTC`, `head_TT`,
`head_total` (`gw_output.py:814-819`) and the per-term `{term}_head_{sector}`
columns (`:820-830`), gated on `photon_head_sigma_basis is not None` at
`:726` and refusing a basis other than `dft` at `:784-787`;
`sigma_diag.dat`'s Hartree column is relabelled `Hdir` on a bispinor run.
Those runs add `sigCC`, `sigTT`, and `sigCT` (`sigCT = CT+TC`) without
changing `sigTOT`/`sigXC`; their sum is gated against that old total before
write. The independently computed current sectors are rounded to the legacy
six-decimal precision and displayed CC is their residual from the unchanged
displayed total, so the identity also closes within 1e-9 eV in public text. A
measured-broken-TR GN deck additionally adds `sigC_odd`; a
measured-TRS deck omits it. Scalar files take the `None` branch and retain
their historical columns byte-for-byte.

**Reading `gwjax.out`.** These lines tell you which route ran:

| line | file:line | means |
|---|---|---|
| `Photon head    : …` | `gw_jax.py:355` (incumbent) and `:789` (packed), via `gw_config.incumbent_bispinor_head_record:3889` | **which Γ-cell head ran, on every bispinor deck.** Packed: the completion's `hall_source`, `σ_H`, Ward / Hermiticity / Dyson-bound residuals and cubature orders, or the `DEBUG … NOT a production calculation` line under `head_correction = off`. Incumbent: the same DEBUG line under `off`, and under `full` a statement that the charge head is the scalar band-diagonal one and that there is **no** transverse q=Γ head on that route |
| `Photon route   : …` | `gw_jax.py:325-355` | **which route ran, on every bispinor deck.** It names packed screened static, packed bare/dynamic, or incumbent charge-screened + `Sigma^B`; an incumbent selection includes the first predicate condition that decided it. |
| `Photon Sigma   : …` | `gw_jax.py:773-786` | Whether the packed Sigma is all sixteen static blocks or a dynamic CC block plus twelve static current blocks. |
| `Sigma blocks   : …` | `gw_jax.py:1461-1468` | Max and mean `|diag|` in eV over the Sigma window for CC, CT+TC and TT. These reduce the same per-state fields written to `sigma_diag.dat`. |
| `Head Sigma     : …` | `gw_jax.py:1470-1475` | Max and mean `|diag|` of the Gamma-cell contribution, split CC, CT+TC and TT. Dynamic CC combines the scalar bare-X and dynamic-correlation head owners; current sectors come from the packed completion. |
| `GN odd Sigma   : …` | `gw_jax.py:1478-1498` | Measured-broken-TR only: max/mean `|sigC_odd|`, its max/mean shares of `|Sigma_xc|`, the `W(iomega_p)` Hermiticity residual, and `max|D|/max|B|`. Its absence on a TRS deck is part of the schema. |
| `Slab WS cert   : …` / `Photon WS cert : …` | `gw_jax.py:803-809,1500-1506` | Exact Wigner-Seitz cubature order ladder, physical node counts and final mixed-tolerance error ratio. The photon line also reports the coupled solve's maximum Dyson backward residual. |
| `Global TRS     : …` | `common/scientific_output.py:352-386` | The measured global verdict and provenance used by the screening/ordered-residue route. Route selection reads this verdict; it is not inferred again from a deck flag. |
| `Bispinor GW policy: bispinor_gw=…` | `gw_jax.py:314` | the carrier banner. Under `full_static_cohsex` the parenthetical names the **head state**, not "experimental", and reads `DEBUG: Gamma-cell head disabled by head_correction=off` when the completion is off |
| `Σ^B tile (μ_L=i, ν_L=j): tr Σ = …` ×9 | `sigma_x_bispinor.py:217` | the **incumbent** bare route ran. Their absence beside a `Photon route: packed …` line is the packed bare route |
| `WARNING -- DEBUG: Gamma-cell head disabled by head_correction=off` | `w_isdf.py:2092-2096` | the packed body ran **headless**; the `WARNING` token is what the production sink keeps in the run record's warning block |
| `static photon response: approximation=…` | `gw_jax.py:756` | one of the **four** stamps above — the `no_pair` pair means the screened mode ran, the `bare_transverse` pair means the packed bare route did — plus `N_packed` |
| `[photon response] packed body N_packed=… : … GB/rank resident for EACH of V and W` | `w_isdf.py:2130-2135` | the measured per-rank cost of the packed carrier, printed on both packed routes |
| `Photon head    : Gamma-cell completion applied …` with `hall_source=`, `sigma_H=`, `ward=`, `hermiticity=`, `dyson_forward_bound=`, `cubature_orders=` | `gw_jax.py:789-809` | **the production run record's head line** (`41e2b6b2`). It reads `DEBUG: … NOT a production calculation` when the head was skipped, and `hall_source` says whether the Hall artifact was used or `σ_H = 0` |
| `packed photon COHSEX block (A,B) complete` ×16 | `photon_sigma.py:508` | the packed Σ ran — and there will be no `Σ^B tile` lines |
| `rho + signed J/c sweep` | `kin_ion_io.py:689` | the transverse Hartree ran |
| the `V_H` matrix label carrying `+ <m\|sum_i alpha_i A_i\|n>` | `kin_ion_io.py:991-993` | same, at the matrix sweep |

## Refusals, compressed

Every entry is a hard refusal, not a demotion. The `bispinor` refusals live in
`gw_config.refuse_unsupported_bispinor_gw` (`:3930-4083`), called at parse
and again at driver entry (`gw_init.py:3101`).

| rule id | file:line | fires when |
|---|---|---|
| `packed_bare_transverse_tt_head_double_count` | `gw_config.py:3943` | `head.bispinor_tt_head_correction = true` on a deck that takes EITHER packed route. The deck key is gone (2026-09-01), so this is the hand-built-config guard. The Γ completion already inserts `⟨D_TT⟩`; honouring the overlay too would double count it |
| `static_gauge_hall_file_missing` | `file_io/static_gauge_head.py` | the deck NAMES a Hall artifact that does not exist. This loader is the one absence owner; unnamed still means `σ_H = 0`, announced |
| `packed_bare_transverse_hall_unavailable` | `w_isdf._load_static_photon_hall` | an authenticated Hall artifact has any nonzero component on the packed **bare** route. Exact zero is accepted; otherwise the current-unscreened model's finite-q `W_CT = 0` cannot have a Γ-only CT/TC limit |
| (no id) `screen_current` inconsistency | `w_isdf.py:2195-2226` | the `screen_current` argument disagrees with `packed_photon_screens_current(config)` for the resolved `bispinor_gw`, or `W_charge` is missing on the bare branch |
| `bispinor_gw_charge_hall_cubature_retired`, `bispinor_gw_pauli_reference_retired`, `bispinor_gw_isometric_kinetic_balance_retired` | `gw_config.py:324` (`_RETIRED_BISPINOR_GW_MODES`, read by `coerce_bispinor_gw_mode` `:353`) | the deck names a retired mode value. **Refusals, not aliases**: the coercer runs from a dozen resolution sites, so an alias would print from each, and a mode value is the one deck word that decides which physics runs. Each names its replacement |
| `bispinor_head_correction_no_local_fields_unavailable` | `:3943` | `head_correction = no_local_fields` with any bispinor deck; the coupled solve does not produce that scalar diagnostic, and on the bare family the value would choose a route by changing head policy |
| `bispinor_slab_cohsex_restart_changes_the_head_mechanism` | `:3971` | explicit `restart = true` with bispinor slab COHSEX. The scalar and packed paths now share the cell-average cubature, but the restart schema has no packed response, wings, or rank-4 completion, so restart would still change content and insertion ownership |
| (deck key, not a gate) `bispinor_tt_head_correction` | `read_lorrax_input` | the key is REMOVED and refused at any value, naming the Γ-cell completion that carries the head |
| `bispinor_self_consistency_requires_live_four_current` | `:3997` | bispinor QSGW with explicit `density_self_consistent = false` |
| `packed_screened_mpa_static_role_unimplemented` | `gw_config.refuse_unsupported_bispinor_gw` | `full_static_cohsex` with `compute_mode = mpa`; the MPA fit transaction does not expose its exact static W to the packed screened assembly |
| `packed_screened_self_consistency_unimplemented` | `gw_config.refuse_unsupported_bispinor_gw` | screened-current QSGW; all sixteen chi blocks, the packed solve and coupled head would need rebuilding per map |
| `packed_bare_cohsex_self_consistency_unimplemented` | `gw_config.refuse_unsupported_bispinor_gw` | bare-family COHSEX QSGW; CC and the charge S/wing completion are consumed and have no per-map packed owner |
| `bispinor_gw_requires_bispinor` | `:4044` | `bispinor_gw = full_static_cohsex` with `bispinor = false` |
| `static_bispinor_photon_head_slab_only` | `:4055` | the coupled completion is requested outside a slab; no bulk integrator exists |
| `static_bispinor_photon_envelope` | `:4073`, over `packed_static_envelope` `:3655` | any of the nine envelope rows fails |
| `bispinor_tt_head_unsupported` | `:4086-4137` | a hand-built TT head lacks bispinor or has unsupported dimensionality |
| `packed_dynamic_sc_requires_current_operator` | `sc_iteration.run_sc_driver` | a served bare dynamic SC route reaches the map without both the immutable photon response and transverse DFT bundle |
| (no id) missing current centroids | `gw_jax.py:579-584`, `gw_init.py:1425-1428` | `bispinor = true` without `centroids_file_current` |
| (no id) packed route needs distributed dense linear algebra | `gw_config.packed_static_envelope` | Every packed route requires `linalg = distributed`. The default `local` remains the incumbent fallback for `bare_transverse` and is a refusal for `full_static_cohsex`; `distrib_la` still refuses the distributed route on a 1×1 mesh. |
| (no id) photon Σ envelope | `sigma_dispatch.py:845` ff. | the static arm: outside `cohsex`; without a packed response; with scalar `static_head_terms`. The dynamic arm refuses a mode that builds static screened channels; `photon_sigma` refuses absent CC unless `blocks = current` and no W carrier is supplied |
| (no id) packed response needs a config / distributed Dyson | `w_isdf.py:1945-1953` | the packed path is called without the run config, or with a non-distributed Dyson solver |
| `GATE static_gauge_head_fold_{ward,hermiticity}` | `head_correction.py:1365,1370` | Γ-fold residuals over `1e-8` / `1e-10` (`:158-159`) |
| `GATE static_photon_{dyson,polygon}_*` | `head_correction.py:1249-1295` (helpers `:1159-1246`) | coupled-solve and polygon certificates, budget `1e-9` |
| `GATE photon_head_sigma_sector_closure` | `photon_sigma.py:515-520` | CC + CT/TC + TT does not close on the direct sixteen-block total |
| `GATE static_gauge_hall_{schema,partial,artifact_absent}` and the authentication gates | `file_io/static_gauge_head.py:69,77,83,200,203`; identity checks `:235-256` | Hall artifact partial, wrong schema, or not authenticating against this WFN, band manifold or `nk_tot`. **An absent file is not one of these** on the packed path: it means `σ_H = 0`, announced |
| `GATE static_gauge_raw_hall_degenerate` | `qsgw_head.py:2686` | degenerate differently-occupied states |
| (no id) insulating-only Hall | `qsgw_head.py:2808` | any fractional occupation |

**Three refusals are absent at `34228021`.** `full_static_bispinor_gauge_head_unavailable`,
`GATE static_gauge_head_end_to_end_uncertified` and
`GATE static_gauge_availability` guarded the deleted full-static-gauge seam and
were removed with it. A mode that cannot be requested needs no refusal.

## Memory invariants

The standing rule is that there must always exist a path that materialises no
`N_μ²`-class object on any one rank ([design decisions](decisions.md), the
two-plans-per-solve-family entry; the plan table is
[`docs/dev/large_nmu_operation.md`](../dev/large_nmu_operation.md)). Where it
is enforced on this layer, and how:

* **Packed Dyson.** Enforced by construction: `w_isdf.py:1191-1195`
  states that inputs, `A`, the LU factors and `W` all stay `P(None,'x','y')`
  and no rank materialises a full `(μ,μ)` tile.
* **Packed Σ.** Enforced by an **assertion**:
  `photon_sigma._require_packed_operator` (`:126-134`, message `:132-134`) raises unless both
  packed operators are still `P(None,'x','y')` — "A photon body may not be
  gathered or placed on fewer than all ranks." Block views are
  `dynamic_slice`s, not gathers (`photon_layout.py:259-274`).
* **`V_q` construction.** Structural, documented rather than asserted: one
  tile at a time, streamed to HDF5 and freed, so peak equals one scalar `V_q`
  tile (`v_q_bispinor.py:29-31`); Hermitian companions never stored (`:32-35`);
  TT Lorentz mixing done at write time so the reader's per-tile contract stays
  clean (`:389-391`).
* **`Σ^B`.** Structural, delegated to the reader — nine bare tiles consumed
  sequentially, each `P(None,'x','y')` (`v_q_bispinor.py:682-694`). There is
  **no assertion** in `sigma_x_bispinor.py`; what it carries instead is a
  measured per-rank ψ-inventory disclosure (`:164-178`) recording that the
  transverse bundle doubles the primary bundle's inventory.
* **ζ fits.** The layout contract is stated once, at `isdf/core.py:4960-4966`
  ("nothing here ever replicates an O(μ²) object"), with the per-tier byte
  accounting at `isdf/core.py:3569-3573` and the runtime banner at
  `isdf_fitting.py:920-923`.

Those are comments, banners and one assertion — not a gate. Treat the two
structural cases (`v_q_bispinor`, `sigma_x_bispinor`) as design intent to
uphold when editing, not as something the tree checks for you.

## Owners

| what you want | where |
|---|---|
| the physics of every Γ-cell head, and which frequency each channel carries | [Four-current heads and frequency](../theory/four-current-head-corrections.md) |
| what a deck key does | [Input reference](../input_reference.md) |
| one-line module descriptions | [Codebase](codebase.md) |
| the ζ fit's data movement and the coupled transverse schedule | [Face-ψ ζ fitting](zeta_fit_face_psi_cct.md) |
| the mini-BZ estimators and the photon cubature | [`vcoul`](../services/vcoul.md) |
| the distributed solve behind the Dyson step | [`distrib_la`](../services/distrib_la.md) |
| bispinor tile symmetry contracts, and why 4-component rotation does not exist | [Symmetry register](symmetry_register.md) |
| the transverse current Hartree | [Direct Hartree field](../theory/hartree.md) |
| Γ-cell quadrature ownership and the remaining scalar-versus-packed insertion distinction | [Four-current heads and frequency](../theory/four-current-head-corrections.md) §3.6 — both routes use the Wigner–Seitz polygon cubature; insertion is still route-specific |
| what freezing the current blocks at ω = 0 inside a dynamic run costs (1.2 × 10⁻⁸ eV on MoS2 3×3, and it bounds the neglected frequency dependence from above) | [Four-current heads and frequency](../theory/four-current-head-corrections.md) §2.2 |
| the narrative introduction, for a reader rather than an editor | `manual/08_bispinor/` (repo only, not in this site) |
