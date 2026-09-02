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

> **Pinned to `origin/main@8b6e3cc7`, 2026-09-01.** Every `file:line` below
> was read at that commit. Line numbers are for finding code, never for
> quoting it. Six lanes of the bispinor static cleanup are in flight against
> these files; the theory page's per-section status notes name which.

## The two routes

Everything on this page belongs to one of two routes, and the fork is a
single predicate.

| route | selected by | screening of the transverse channels | marked below |
|---|---|---|---|
| **bare** | `bispinor = true` with `bispinor_gw ∈ {bare_transverse (default), pauli_reference_bare_transverse, isometric_kinetic_balance_bare_transverse}` | none — `Σ^B` contracts the *bare* `D^{ij}` tiles | **B** |
| **packed** | `bispinor_gw ∈ {full_static_cohsex, charge_hall_cubature}` | one packed 4×4 Lorentz Dyson solve, `ω = 0`, `compute_mode = cohsex` only | **P** |

`gw_config.uses_static_photon_response(config)` (`src/gw/gw_config.py:3532-3539`)
is that fork and returns True for exactly the two packed modes. Its companion
`uses_coupled_photon_head` (`:3542-3545`) adds `head.correction is FULL`.
Consulted at `gw_jax.py:103,595,637,678,711,732`, `sigma_dispatch.py:53,838`
and `gw_init.py:55,2880`.

## The map

```mermaid
flowchart TD
  subgraph DECK["deck (cohsex.in)"]
    K1["bispinor · bispinor_gw<br/>centroids_file_current"]
    K2["head_correction<br/>bispinor_tt_head_correction<br/>static_gauge_hall_file"]
    K3["transverse_zeta_solve / _rcond<br/>distrib_la_batched_route"]
  end

  subgraph CFG["config — gw_config.py, common/four_current_model.py"]
    R["resolve_four_current_representation<br/>FourCurrentRepresentation:<br/>carrier + lift + stamps"]
    F{"uses_static_photon_response"}
    E["refuse_unsupported_bispinor_gw<br/>18-conjunct envelope"]
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
    HT["_compute_live_hartree<br/>four-spinor G-space V_H + transverse"]
  end

  OUT["SigmaResult.photon_head_sigma_*<br/>gwjax.out · sigma_freq_debug · sigma_diag.dat"]

  K1 --> R
  R --> F
  K2 --> E
  K3 --> ZT
  F -->|packed| CHI
  F -->|bare| SB
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
  SW -->|bare route only| SB
  PS --> OUT
  SB --> OUT
  HT --> OUT
```

## Stage 1 — deck to config

Parsing and defaults are the input reference's; the wiring facts are which
object each key ends up on and who reads it.

| key | default (`gw_config.py`) | lands on | route |
|---|---|---|---|
| `bispinor` | `False` (`:1653`, parse `:5572`) | `config.bispinor` (`:4681`) — the master switch | both |
| `bispinor_gw` | `bare_transverse` (`:1656`, parse `:5573`, enum `:293-307`) | `config.bispinor_gw` (`:4682`) | both |
| `centroids_file_current` | `""` (`:1477`, parse `:5066-5071`) | `config.paths.centroids_file_current` (`:3229`) | both |
| `head_correction` | `full` (`:1989`, parse `:5082,5106`) | `config.head.correction` (`:3849`) | both |
| `bispinor_tt_head_correction` | `False` (`:1985`, parse `:5120`) | `config.head.bispinor_tt_head_correction` (`:3861`) | B |
| `static_gauge_hall_file` | `static_gauge_hall.h5` (`:1466`, parse `:5074`) | `config.paths.static_gauge_hall_file` (`:3232`) | P (`charge_hall_cubature` only) |
| `transverse_zeta_solve` | `ridge` (`:1871`, validate `:5348-5365`) | `config.backend.transverse_zeta_solve` (`:4404`) | both |
| `transverse_zeta_rcond` | `1e-10` (`:1879`, validate `:5366-5370`) | `config.backend.transverse_zeta_rcond` (`:4405`) | both |
| `distrib_la_batched_route` | `auto` (`:1746`, resolve `:1214-1242`, parse `:5312`) | `config.backend.distrib_la_batched_route` (`:4395`) | both |

There is no `transverse_zeta_*` key beyond those two, and no `LORRAX_*`
variable reads any key in this table.

**The one resolver.** `common/four_current_model.resolve_four_current_representation(bispinor, model)`
(`src/common/four_current_model.py:54-106`) returns a frozen
`FourCurrentRepresentation` (`:35-51`) with seven fields: `charge_bispinor`,
`charge_lift`, `current_bispinor`, `current_lift`, `scalar_head_bispinor`,
`charge_representation`, `spatial_current_representation`. **It is not stored
on the config** — a dozen sites call it locally (`gw_config.py:348`,
`gw_init.py:804,1464,1991,2817,3123`, `sigma_dispatch.py:310`,
`sc_iteration.py:1588,1736`, `head_correction.py:667,1715`,
`qsgw_head.py:3938`, `kin_ion_io.py:1168`, `file_io/kin_ion.py:393`,
`psp/get_dipole_mtxels.py:1103`), so grep for
`resolve_four_current_representation`, not for a config attribute.

The four branches: non-bispinor (`:64-73`, everything False);
`pauli_reference_bare_transverse` (`:74-84`, charge stays a two-spinor Pauli
density, current lifted `raw`); `isometric_kinetic_balance_bare_transverse`
(`:85-96`, both `isometric`, `scalar_head_bispinor=False`); and the
fall-through for `bare_transverse`, `full_static_cohsex` and
`charge_hall_cubature` (`:97-106`, both `raw`, `scalar_head_bispinor=True`).
The lift selectors are `RAW_KINETIC_BALANCE_LIFT = "raw"` and
`ISOMETRIC_KINETIC_BALANCE_LIFT = "isometric"`
(`src/common/bispinor_init.py:40-41`).

**`static_bispinor_photon_envelope` is a gate id, not a function.** The
conjunct table is `gw_config.py:3635-3699`, the loop `:3700-3702`, the raise
`:3703-3715`, and it is reached only for the two packed modes. Eighteen
conjuncts; two are mode-dependent (`:3631-3634`, `:3652-3654`):
`charge_hall_cubature` **requires** `head_correction = full` and `sys_dim = 2`,
`full_static_cohsex` **requires** `head_correction = off` and waives `sys_dim`.

## Stage 2 — initialization (`gw_init.py`)

| object | producer | shape / dtype | sharding | route |
|---|---|---|---|---|
| charge centroid indices | `gw_jax.py:356-359` → `file_io/centroids.load_centroid_basis` (`:122-174`) | `(n_mu_C, 3)` i64 | host | both |
| current centroid indices | `gw_init.py:1434-1446` (refusal `:1430-1433` if the key is unset) | `(n_mu_T, 3)` i32 | host until needed | both |
| `meta_transverse` | `gw_init.py:1447-1452` | `Meta` with `n_rmu = n_mu_T`, `nspinor = 4`, `npol = 4`, `n_rmu_padded = padded_mu_extent(...)` | — | both |
| four-spinor ψ | `common/bispinor_init.lift_to_4spinor` (`:151-206`) | `(n_k, nb, 4, ngkmax)` c128, `[ψ_L; ψ_S]` on the spinor axis | caller wraps; no sharding inside | both |
| face ψ carriers | `common/wfn_layout.py:12-13`; transverse copies `gw_init.py:2419-2426` | `(nk, n_X, s, mu_Y)` / `(nk, s, mu_X, n_Y)` | `P(None,'x',None,'y')` / `P(None,None,'x','y')` | both |
| charge ζ | `gw_init.py:2216-2246` → `isdf_fitting.fit_zeta_to_h5` | `tmp/zeta_q.h5`, `(n_q_disk, n_mu_C, ngkmax)` c128 | accumulator `(n_q_disk, n_mu_padded, ngkmax)` at `P(None,('x','y'),None)` (`isdf_fitting.py:1404-1409`) | both |
| three transverse ζ | `gw_init.py:2455-2506` (`_fit_transverse_channel`, `vertex_mu_L = 1,2,3`) | `tmp/zeta_q_mu{1,2,3}.h5`, `(n_q_disk, n_mu_T, ngkmax)` c128 | same | both |
| `C_q` (CCT Gram) | `isdf_fitting.py:743-751` | `(nq, n_mu_padded, n_mu_padded)` | `P(None,'x','y')` — layout contract `isdf/core.py:4960-4966` | both |
| bare `D^{IJ}` tiles | `v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5` (`:261-581`) | `v_q_bispinor.h5`, 7 datasets `(n_q_total, n_mu_L, n_mu_R)` c128 | device buffer `P(None,'x','y')` (`v_q_g_flat.py:492-494`) | both |
| tile reader | `v_q_bispinor.BispinorVqReader.get_tile` (`:682-716`) | `(n_q, n_L_padded, n_R_padded)` c128 | `P(None,'x','y')`; Hermitian companions read at `P(None,'y','x')` then conj-swapped (`:704-709`) | both |
| `photon_g0_vectors` | `v_q_bispinor.py:575-581` | 4-tuple, each `(n_q_full, n_mu_padded)` c128 | `P(None,'x')` | released to `None` unless `uses_coupled_photon_head` (`gw_init.py:2879-2881`) — **P** |

**Sixteen tiles, seven on disk.** Six `(0,i)`/`(i,0)` tiles are exactly zero
by Coulomb gauge (`ZERO_TILES`, `v_q_bispinor.py:67-70`); three `(j,i)` with
`i < j` are Hermitian companions reconstructed on read (`HERMITIAN_PAIRS`,
`:72-78`); the remaining seven — CC plus six TT — are computed and written
(`UNIQUE_TILES`, `:60-65`). Format stamp `bispinor_lorentz_v2` (`:80`),
artifact path `gw_init.py:2793`.

**The coupled ζ schedule.** The Y-side face transform and the full-spin X
broadcast are channel-independent, so μ_L = 1,2,3 share one
built-once-per-r-chunk `[3, q, μ, r]` `Z_q` stack instead of repeating it
three times (`isdf/core.py:2620-2626`). Entry `gw_init.py:2527` (coordinator
`:1752-1928`, route chooser `:1931-1955`, capacity block `:2082-2189`);
kernel `isdf/core._z_q_face_coupled_mu123` (`:2795-2833`). The coupled
out-spec is `P(None,None,'x','y')` against the scalar `P(None,'x','y')`
(`isdf/core.py:2420-2421` vs `:2337-2340`). `distrib_la_batched_route` reaches
this fit through `gw_init.py:2083-2084` → `:2472`, so the same key that
governs generic batched linalg governs the transverse ζ schedule.
Full design: [Face-ψ ζ fitting](zeta_fit_face_psi_cct.md).

**The bare TT head slot** (route B; refused on the packed route by
`gw_config.py:3660-3662`). Config plumbing `gw_init.py:2856-2857` →
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

### 3a. The packed body (route P only)

Fork at `gw_jax.py:637-655` → `w_isdf.compute_static_photon_response`
(`:2079`, signature `:2079-2093`), the only production caller.

| object | producer | shape | sharding |
|---|---|---|---|
| one no-pair block `χ^{IJ}_0` | `w_isdf.compute_no_pair_dirac_current_block` (`:1556`) | `(nq, μ_L, μ_R)` c128 | `P(None,'x','y')` |
| the sixteen blocks | `compute_experimental_no_pair_photon_chi0` (`:2016`); families `(charge, transverse, transverse, transverse)` (`:2049-2050`) — T1/T2/T3 share one bundle, no copies | — | — |
| TT Ward subtraction | `_subtract_static_tt_contact` (`:2004`), applied to the nine blocks with both indices nonzero (`:2057-2061`) | `Π(q) − Π(0)` | in place, buffer donated |
| packed `V`, `χ_0`, `W` | `photon_layout.pack_photon_operator` (`:203`), `w_isdf.solve_w` (`:1431`) | `(nq, N_packed, N_packed)` c128 | `P(None,'x','y')` |

`N_packed = p_C + 3·p_T` (`photon_layout.py:91-93`), with `p_C`, `p_T` the
mesh-padded centroid extents. The packing is **mesh-interleaved**, not a
contiguous direct sum: packed row shard `x` holds that shard's own local
`C,T1,T2,T3` row chunks, and the column shard applies the same permutation
(`photon_layout.py:1-20`, offsets `:103-107`). A packed operator is therefore
`P O Pᵀ`, Dyson algebra is unchanged, and both `pack_photon_operator` and
`photon_block_view` (`:263`) are local `shard_map` slices with no
redistribution. Ordering stamp `mesh_interleaved_direct_sum_v1` (`:39`).

The sixteen-block enumeration lives in `pack_photon_operator`'s double loop
(`photon_layout.py:217-218`), which calls the block builder once per `(A,B)`
and `block_until_ready()`s each (`:231`) so only one block is resident.

**The Dyson solve** is `solve_w` (`w_isdf.py:1431`) with
`dyson_solver="distributed"` hard-coded on this path (`:2199`), dispatching to
`_get_w_solve_fn_distributed` (`:1164`), which plans through the `distrib_la`
service (`plan("solve_lu", mesh_xy, backend="distributed", …)` at `:1258-1260`,
executed at `:1328`). Inputs, the assembled `A`, the LU factors and `W` all
stay `P(None,'x','y')`. The source states the invariant at `:1191-1195` and
`:1450-1452`: no rank ever materialises a full `(μ, μ)` tile — the largest
per-rank transient is the `μ·μ/min(Px,Py)` gathered GEMM operand.

`StaticPhotonResponse` (`w_isdf.py:2066`) carries `layout`, `V_packed`,
`W_packed` (both as above), `current_contact` (always
`ward_subtracted_no_pair`, `:1998`), `head_completion`, `current_model`
(`positive_energy_kinetic_balance_dirac_current_v1`,
`common/bispinor_init.py:31`) and `approximation` — either
`experimental_no_pair_bubble_screened_breit_v1` (`:2271`) or
`charge_hall_cubature_on_experimental_no_pair_body_v1` (`:2269`).

### 3b. The Γ completion (route P, `charge_hall_cubature`)

`head_correction.complete_static_slab_photon_q0` (`:1390-1399`) takes
`V_packed`, `W_packed`, a response record, the two `(4, N_packed)` `g_0`
vectors (`g0_X` at `P(None,'x')`, `g0_Y` at `P(None,'y')`, packed at
`w_isdf.py:2243-2249`) and a cubature receipt, and returns updated `V`, `W`
plus a `StaticSlabPhotonHeadCompletion` (`:1225`).

| input | producer | shape | sharding |
|---|---|---|---|
| cubature receipt | `vcoul.slab_minibz_photon_cubature` (`services/vcoul/src/vcoul/minibz.py:821`) — exact Wigner–Seitz polygon, fixed 16/24/32 Duffy–Gauss ladder (`:124`) | host chunks: `q_cart (n,3)`, `D_raw (n,4,4)`, `sample_weight (n,)` f64 | host, write-locked |
| `ChargeHallCubatureResponse` | `static_gauge_response.build_charge_hall_cubature_response` (`:214-325`) | `S_direct (2,2,4,4)`, `sigma_H (3,)`, `Y_x (2,4,N_packed)`, `Z_y (2,N_packed,4)` | `P()`, `P()`, `P(None,None,'x')`, `P(None,'y',None)` |
| Hall artifact | `file_io/static_gauge_head.load_static_gauge_hall_artifact` (`:237`), called from `w_isdf.py:2166-2174`; sole writer `psp/get_dipole_mtxels.py:1326` | `static_gauge_hall.h5`, schema 1 (`:66`), `sigma_H_cart (3,)` f64 | replicated |

The updates are **one bare rank-4 outer product into `V`** (`:1583-1585`) and
**nine screened rank-4 outer products into `W`** (`:1594-1606`), all through
`photon_layout.add_photon_q0_low_rank` (`:540`), which takes left rows at
`P(None,'x')` and right rows at `P(None,'y')`, donates the packed buffer, and
does a purely local outer product (`:560-563`). Gates on this path:
`GATE static_gauge_head_fold_ward` (`:1468`, `1e-8`),
`GATE static_gauge_head_fold_hermiticity` (`:1473`, `1e-10`), and the
`static_photon_dyson_*` / `static_photon_polygon_*` numerical certificates
(`:1246-1387`, budget `1e-9`).

**What the mode declares it does not have.**
`static_gauge_response.CHARGE_HALL_CUBATURE_AVAILABILITY` (`:97`) is the
complete list, and `is_complete_for` (`:62-73`) requires an exact match — an
extra `complete` fails as loudly as an `unavailable`. Complete (5): `cc_q2`,
`ct_q1`, `tc_q1`, `y_charge`, `z_charge`. `omitted_by_model` (10): `ct_q2`,
`tc_q2`, `tt_q0`, `tt_q1`, `tt_q2`, `y_current`, `z_current`, `contact_q0`,
`contact_q2`, `complement_space`. What that list means physically is
[the theory page](../theory/four-current-head-corrections.md) §4.3.

**A dead seam you will find and should not follow.** `StaticGaugeHeadResponse`
(`head_correction.py:122`), `require_static_gauge_head_response` (`:267`),
`require_full_static_gauge_availability` (`static_gauge_response.py:328`), the
v2 schema and loader in `file_io/static_gauge_head.py` (`:41-64`, `:322`,
`:436`) and the `else` arm at `head_correction.py:1427` have **no production
caller**: the only path in always passes a `ChargeHallCubatureResponse`
(`w_isdf.py:2229,2253`), a caller-supplied `gauge_head_response=` is refused
(`w_isdf.py:2148-2152`), and the config refuses the mode first
(`gw_config.py:3616-3630`).

### 3c. The scalar charge head (route B)

Unchanged from the scalar code and owned elsewhere: `HeadResolver`
(`head_correction.py:1691`) built at `gw_jax.py:497`; `head_correction = full`
installs `qsgw_head.build_dft_head_response` (`gw_jax.py:600-617`), finalized
at `:686-691`, applied through `_compute_static_head` (`gw_jax.py:186,742`) →
`cohsex_sigma.py:631-637,720-728`. It is **skipped entirely** under the packed
modes (`gw_jax.py:731-732`), and `sigma_dispatch.py:850-856` refuses a scalar
head overlay if one arrives there anyway.

## Stage 4 — self-energy

The fork is `sigma_dispatch.py:838`, and its three refusals fire before any
allocation: outside `compute_mode = cohsex` (`:839-844`), with no packed
response (`:845-849` — "Refusing instead of falling back to charge-only
screened COHSEX"), and with scalar `static_head_terms` present (`:850-856`,
which would double count CC and omit the coupled current wings).

| object | producer | shape | sharding | route |
|---|---|---|---|---|
| sixteen-block `Σ_X`, `Σ_SX`, `Σ_COH` | `photon_sigma.compute_static_photon_sigma` (`:276`), called at `sigma_dispatch.py:863-878` | each `(nk, nb_sigma, nb_sigma)` | **replicated** `P(None,None,None)` at the output boundary (`:433-444`) | P |
| per-block operands | `photon_layout.photon_block_view` (`:263`), fetched at `photon_sigma.py:354-355` | `(nk_tot, n_left, n_right)` | `P(None,'x','y')` — a `dynamic_slice`, never a gather | P |
| Green function `G` | `greens_function_kernel.build_G`, face layout | `(nk, ns, μ_L, ns, μ_R)` | `P(None,None,'x',None,'y')` (`wavefunction_bundle.py:236`) | both |
| head-attribution diagnostics | `photon_sigma.py:363-367` via `photon_layout.photon_q0_low_rank_block` (`:660`) | one `(1, p_A, p_B)` block | `P(None,'x','y')` | P |
| `Σ^B`, bare transverse | `sigma_x_bispinor.compute_sigma_x_bispinor` (`:91`), nine `(i,j)` tiles at `:200-201` | `(nk, nb_sigma, nb_sigma)` | replicated after a gather-then-window (`:234-248`) | B |
| transverse Hartree | `sigma_dispatch._compute_live_hartree` (`:303-336`) → `kin_ion_io.compute_hartree_matrix` (`:764`) | `charge`, `transverse`, each `(nk_full, nb, nb)` Ry | `P(None,'x','y')` with `return_sharded=True` | **both** |

**The sixteen-block contraction** is a plain nested Python loop (`A` at
`photon_sigma.py:343`, `B` at `:347`), not a `vmap`. Per block the γ̃ vertices
are folded into the two G-build operands only (`:345`, `:349`), and the same
jitted kernel is called three times with a dynamic `term` selector for X, SX
and COH (`:373`, `:383`, `:393`; COH uses `W − V` with prefactor `−0.5`).
Each accumulator is `block_until_ready()`d before advancing — the source names
this "the lifetime boundary that prevents two W/G body tiles from coexisting"
(`:426-428`). `GATE photon_head_sigma_sector_closure` (`:468-473`) then checks
that CC + (CT+TC) + TT closes on the direct sixteen-block total.

**`Σ^B` enters twice, differently.** In `compute_cohsex_sigma` it is added to
**both** `sig_x` and `sig_sx` (`cohsex_sigma.py:660-668`); in
`compute_sigma_x` — the entry every dynamic mode and `x_only` takes — it is
added to `sig_x` only (`:735-742`).

**No transverse operand reaches a dynamic Σ_c.** `compute_ppm_sigma_pipeline`
(`ppm_pipeline.py:473-490`, called `sigma_dispatch.py:1185-1196`) and the MPA
body (`mpa/sigma.py:283-303`, called `:1091-1113`) have no
`wfns_transverse`, `bispinor_v_q_path` or `photon_response` parameter. On a
dynamic route the four-current layer reaches Σ through `sig_x` and through the
transverse Hartree, and nowhere else.

**The transverse Hartree is on both routes.** Its gate is
`include_transverse = bool(config.bispinor)` (`sigma_dispatch.py:312`), not
`bispinor_gw`, so every bispinor mode and every compute mode gets it unless
`omit_v_h` (density self-consistency, which rebuilds both fields itself). It
is added to `sig_h` at `:944`, under the guard at `:936`. Physics owner:
[Direct Hartree field](../theory/hartree.md).

## Stage 5 — outputs

`SigmaResult` (`sigma_dispatch.py:60-61`) carries three fields, `None` on
every non-packed route (initialized `:835-837`, set at `:958-961` for `x_only`
and `:974-977` for `cohsex`):

| field | line | shape / value |
|---|---|---|
| `photon_head_sigma_diag_tskn_ry` | `:124` | `(3, 3, nk, nb)` — axes `(term = X/SX/COH, sector = CC / CT+TC / TT, k, band)`, DFT basis |
| `photon_head_sigma_operator_fingerprint` | `:125` | `str \| None` |
| `photon_head_sigma_basis` | `:126` | `"dft"` when populated |

Mirrored onto the output record at `gw_output.py:149,152,153` and forwarded by
the driver at `gw_jax.py:942-946,1169-1173` (schema zeros at `:947-949`).
`sigma_freq_debug` gains `head_CC`, `head_CTTC`, `head_TT`, `head_total` and
the per-term `{term}_head_{sector}` columns (`gw_output.py:821-843`);
`sigma_diag.dat`'s Hartree column is relabelled `Hdir` and gains an `H_T`
column whenever the transverse Hartree ran (`gw_output.py:770-771,1667-1668`).

**Reading `gwjax.out`.** These lines tell you which route ran:

| line | file:line | means |
|---|---|---|
| `Bispinor GW policy: bispinor_gw=…` | `gw_jax.py:300-314` | the route banner; the parenthetical note marks the packed modes experimental |
| `Σ^B tile (μ_L=i, ν_L=j): tr Σ = …` ×9 | `sigma_x_bispinor.py:228` | the bare route ran |
| `[photon response] DECLARED no-pair model …` | `w_isdf.py:2187-2193` | the packed screening ran |
| `static photon response: approximation=…, packed_extent=…` | `gw_jax.py:659-664` | which packed approximation, and `N_packed` |
| `packed photon COHSEX block (A,B) complete` ×16 | `photon_sigma.py:431` | the packed Σ ran — and there will be no `Σ^B tile` lines |
| `rho + signed J/c sweep` | `kin_ion_io.py:694-697` | the transverse Hartree ran |
| the `V_H` matrix label carrying `+ <m\|sum_i alpha_i A_i\|n>` | `kin_ion_io.py:996-1001` | same, at the matrix sweep |

## Refusals, compressed

Every entry is a hard refusal, not a demotion. The `bispinor` refusals live in
`gw_config.refuse_unsupported_bispinor_gw` (`:3548-3715`), called at parse
(`:5626,5637`) and again at driver entry (`gw_init.py:3118,3121`).

| rule id | file:line | fires when |
|---|---|---|
| `bispinor_self_consistency_requires_live_four_current` | `gw_config.py:3552-3565` | bispinor QSGW with an explicit `density_self_consistent = false` |
| `bispinor_gw_requires_bispinor` | `:3568-3573` | a non-default `bispinor_gw` with `bispinor = false` |
| `{pauli_reference,isometric_kinetic_balance}_restart{,_write}_unavailable` | `:3574-3597` | either comparison carrier with `restart` / `write_restart_tensors`. **The id is composed at runtime** (`gate_prefix`, `:3578-3581`) — grepping the literal string finds nothing |
| `isometric_kinetic_balance_current_head_unavailable` | `:3598-3605` | isometric carrier with `bispinor_tt_head_correction = true` |
| `isometric_kinetic_balance_ladder_head_unavailable` | `:3606-3614` | isometric carrier, `head_correction = full`, `screening_diagrams ≠ w_rpa` |
| `full_static_bispinor_gauge_head_unavailable` | `:3616-3630` | `full_static_cohsex` with `head_correction = full`. This is why the FULL screened head is unreachable from any deck |
| `static_bispinor_photon_envelope` | `:3635-3715` | any of the eighteen conjuncts (Stage 1) |
| `bispinor_tt_head_unsupported` | `:3718-3773` | TT head without `bispinor`, or `sys_dim ∉ {2,3}` |
| (no id) SC × dynamic | `gw_jax.py:271-289` | self-consistent solver with a dynamic mode — bispinor QSGW is static-only |
| (no id) missing current centroids | `gw_jax.py:544`, `gw_init.py:1430-1433` | `bispinor = true` without `centroids_file_current` |
| (no id) photon Σ envelope, ×3 | `sigma_dispatch.py:839-856` | outside `cohsex`; without a packed response; with scalar `static_head_terms` |
| `GATE static_gauge_head_end_to_end_uncertified` | `w_isdf.py:2134-2146` | `head_correction = full` on a packed mode other than `charge_hall_cubature` — unreachable, the config refuses first |
| `GATE static_gauge_availability` | `static_gauge_response.py:79` | the availability record is not an exact match for the mode |
| `GATE static_gauge_head_fold_{ward,hermiticity}` | `head_correction.py:1468,1473` | Γ-fold residuals over `1e-8` / `1e-10` |
| `GATE static_photon_{dyson,polygon}_*` | `head_correction.py:1257-1387` | coupled-solve certificates, budget `1e-9` |
| `GATE photon_head_sigma_sector_closure` | `photon_sigma.py:468-473` | CC + CT/TC + TT does not close on the sixteen-block total |
| `GATE static_gauge_hall_*` | `file_io/static_gauge_head.py:253,256,291-307` | Hall artifact absent, partial, or not authenticating against this WFN, band count or operator |
| `GATE static_gauge_raw_hall_degenerate` | `qsgw_head.py:3233-3237` | degenerate differently-occupied states |
| (no id) insulating-only Hall | `qsgw_head.py:3352-3355` | any fractional occupation |

## Memory invariants

The standing rule is that there must always exist a path that materialises no
`N_μ²`-class object on any one rank ([design decisions](decisions.md), the
two-plans-per-solve-family entry; the plan table is
[`docs/dev/large_nmu_operation.md`](../dev/large_nmu_operation.md)). Where it
is enforced on this layer, and how:

* **Packed Dyson.** Enforced by construction: `w_isdf.py:1191-1195,1450-1452`
  states that inputs, `A`, the LU factors and `W` all stay `P(None,'x','y')`
  and no rank materialises a full `(μ,μ)` tile.
* **Packed Σ.** Enforced by an **assertion**:
  `photon_sigma._require_packed_operator` (`:116-124`) raises unless both
  packed operators are still `P(None,'x','y')` — "A photon body may not be
  gathered or placed on fewer than all ranks." Block views are
  `dynamic_slice`s, not gathers (`photon_layout.py:243-258`).
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
| the narrative introduction, for a reader rather than an editor | `manual/08_bispinor/` (repo only, not in this site) |
