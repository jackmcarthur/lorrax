# Bispinor GW — Phase-1 Design (DHF + Bare-Breit)

> **SUPERSEDED IN PART; CURRENT IMPLEMENTATION NOTES UPDATED 2026-08-29.**
> This remains the phase-1 physics record. The current source map and ζ
> schedule supersede the original implementation plan:
>
> - Σ^B assembly lives in `src/gw/sigma_x_bispinor.py` (the planned
>   `src/gw/breit_sigma.py` was never created); V_q^{μν} tiles in
>   `src/gw/v_q_bispinor.py`.
> - Transverse ζ uses the Hermitian-indefinite CCT path. The default ridge
>   family is pivoted LU; `transverse_zeta_solve=rank_truncate` is the explicit
>   indefinite pseudo-inverse alternative. Fresh μ=1,2,3 fits may share their
>   face transform while keeping separate ordered solves. See
>   [the face ζ architecture](architecture/zeta_fit_face_psi_cct.md#current-coupled-transverse-schedule-2026-08-29).
> - File-map rows that no longer exist: `src/common/load_wfns.py`,
>   `src/common/isdf_fitting.py` (now `src/gw/isdf_fitting.py` +
>   `src/isdf/core.py`), `src/centroid/centroid_io.py` (centroid provenance
>   is written by `kmeans_cli`; the GW contract is the configured file role
>   plus a hash of its FFT-index table),
>   `docs/PHYSICS_COMPREHENSIVE.md` /
>   `docs/CODEBASE_COMPREHENSIVE.md` (see `docs/theory/physics.md` /
>   `docs/architecture/codebase.md`), and the `runs/MoS2/...` validation dirs
>   (machine-local, not shipped).
> - Current usage: manual ch. 8 (bispinor GW) and `docs/drivers.md`
>   (two-centroid-file convention, `--density-mode current`).
> - The q→0 head of every channel, the packed screened photon head, and the
>   frequency treatment are owned by
>   [Four-current heads and frequency](theory/four-current-head-corrections.md);
>   §11 below keeps only the 2026-08-01 measurements.

**Status:** historical physics design with current implementation addenda

**Last update:** 2026-08-29

## 1. Scope

DHF + bare-Breit GW with bispinor wavefunctions:

- $\chi^0\equiv\chi^0_{00}$, $W\equiv W_{00}$ — Coulomb screened in RPA (existing scalar code, ns=4 spin axis).
- $\Sigma_{\alpha\beta}=\Sigma^C_{\alpha\beta}+\Sigma^B_{\alpha\beta}$.  $\Sigma^C$ uses $W_{00}$; $\Sigma^B$ uses the **bare** $D^{ij}$ — no transverse screening, no retardation.
- Four ISDF $\zeta$ bases, one per $\tilde\gamma^{\mu_L}$, on **two centroid sets**: the charge feature-row norm for $\mu_L=0$ and the three-current feature-row norm for $\mu_L\in\{1,2,3\}$.

Deferred (phase-2+): full $\chi^{\mu\nu}/W^{\mu\nu}$, transverse screening, retarded Breit, vertex corrections, higher-order kinetic balance, bispinor-aware Sternheimer source.

## 2. Conventions

| Symbol | Range | Meaning |
|---|---|---|
| $\alpha,\beta,\gamma,\delta$ | 1–4 | bispinor (Dirac) component |
| $a,b$ | 1–2 | Pauli when blocking bispinor as $L\oplus S$ |
| $\mu_L,\nu_L$ | 0–3 | Lorentz / 4-vector |
| $i,j$ | 1–3 | spatial Lorentz subset |
| $\mu_c,\nu_c,\lambda_c$ | 1–$n_{r\mu}$ | ISDF centroid |

**γ-matrix convention** (already in [`gamma_matrices.py`](../src/common/gamma_matrices.py)): the stored matrices are $\tilde\gamma^\mu\equiv\gamma^0\gamma^\mu$, so `gamma0` $=I_4$ and `gamma_i` $=\alpha^i$. We always write $\rho^{\mu_L}=\psi^\dagger\tilde\gamma^{\mu_L}\psi$ (no explicit $\bar\psi$).

**Gauge:** Coulomb. The bare 4×4 photon propagator is block-diagonal,

$$D^{\mu_L\nu_L}(K) = \begin{pmatrix} 4\pi/|K|^2 & 0 \\ 0 & -(4\pi/|K|^2)\,(\delta_{ij}-K_iK_j/|K|^2) \end{pmatrix},\quad K=q+G.$$

Off-block ($D^{0i}=0$) is exact in Coulomb gauge.

## 3. Equations

**Bispinor lift** (kinetic balance):

$$\Psi_{nk}(G) = \begin{pmatrix}\psi_L\\\psi_S\end{pmatrix},\quad
\psi_S = \tfrac{\alpha_{\rm FS}}{2}\,\big[\sigma\!\cdot\!(k+G)\big]\,\psi_L,$$

with $(k+G)$ in Bohr⁻¹ — i.e. the BGW HDF5 `wfn.bvec` (stored in reciprocal-lattice units) is multiplied by `wfn.blat = 2π/alat` once at the `WfnLoader` file-format boundary.  [`bispinor_init.py`](../src/common/bispinor_init.py) accepts only that explicitly Cartesian basis.

> **Live addition (2026-09-04): velocity balance for the current carrier.**
> The lift above is the CHARGE carrier and stays as written (its
> small-component density is the $O(\alpha^2)$ Dirac density).  The
> SPATIAL-CURRENT carrier ($\mu_L\in\{1,2,3\}$: transverse ζ fits, $\Sigma^B$,
> the finite-q $\alpha^i$ vertex) may instead be lifted with
> $\psi_S=\tfrac{\alpha_{\rm FS}}{2}\,\sigma\!\cdot\!v\,\psi_L$,
> $v = p + i[V_{\rm NL}, r] = p + \partial V_{\rm NL}/\partial k$ (Hartree units;
> the code adds $\tfrac{\alpha}{4}\sum_a\sigma^a(\partial V^{\rm Ry}_{\rm NL}/\partial k_a)\psi_L$
> to the $\sigma\!\cdot\!p$ term), deck key `bispinor_current_balance = velocity`.
> Then at $q=0$, $\tfrac{2}{\alpha}\langle m|\alpha^i|n\rangle
> = \langle m|v^{\rm Ry}_i|n\rangle + \tfrac{i}{2}\epsilon_{ijk}\langle m|[\sigma^k, \partial_j V_{\rm NL}]|n\rangle$:
> the current vertex is the pseudo-Hamiltonian's velocity at first order,
> which the $\sigma\!\cdot\!p$ lift misses by exactly $\partial V_{\rm NL}/\partial k$
> (the "gauged nonlocal-pseudopotential track" the static head gate names).
> The commutator term survives only through the j-resolved (spin-orbit)
> part of $V_{\rm NL}$.  Both channels are the existing $\tilde\gamma$-bilinears
> on different 4-spinors, so nothing in the ζ/Σ machinery changes; the
> transverse ζ stamp and the finite-q `dipole.h5` carry the lift.  Owner:
> `common/bispinor_init.lift_to_4spinor(representation="velocity")`,
> provider `psp/vnl_ops.nonlocal_velocity_lift`, resolver
> `common/four_current_model.resolve_four_current_representation(current_lift=)`.
> Not yet: the $i[\Sigma^{GW}, r]$ piece (self-consistency-dependent), and the
> exact Hartree/SC density rebuild still shares one carrier (kept on
> $\sigma\!\cdot\!p$; see `docs/architecture/four_current_wiring.md`).

**Polarizability and screening (charge channel only):**

$$\chi^0_{00,q}(\omega) = -\mathrm{Tr}_{\rm bispinor}\big[\tilde\gamma^0\,G^0(12)\,\tilde\gamma^0\,G^0(21)\big]
= -\mathrm{Tr}_{\rm bispinor}\big[G^0(12)\,G^0(21)\big]$$

(since $\tilde\gamma^0=I$).  Reuses the existing minimax kernel with the spin axis grown from 2 to 4.  $W_{00}$ from the existing scalar Dyson, unchanged.

**Self-energy:**

$$\Sigma^C_{\alpha\beta}(12) = -G^0_{\alpha\beta}(12)\,W_{00}(12)$$

$$\Sigma^B_{\alpha\beta}(12) = -\sum_{i,j\in\{1,2,3\}} \tilde\gamma^i_{\alpha\gamma}\,G^0_{\gamma\delta}(12)\,\tilde\gamma^j_{\delta\beta}\,D^{ij}_{\rm bare}(12).$$

$\Sigma^B$ vanishes as $\alpha_{\rm FS}\to 0$ (because $\psi_S\to0$ kills the $L\!\leftrightarrow\!S$ blocks that $\alpha^i$ couples) — recovers the existing GW.

## 4. Four-density ISDF and the two-centroid-file convention

Per channel $\mu_L$, fit $\zeta^{\mu_L}_q(\mu_c, r)$ such that

$$\rho^{\mu_L}_{n_l n_r,k,q}(r) = \sum_{ab}\psi^*_{l,n_l,k,a}(r)\,\tilde\gamma^{\mu_L}_{ab}\,\psi_{r,n_r,k+q,b}(r)
\;\approx\;\sum_\lambda \zeta^{\mu_L}_{q,\lambda}(r)\;\rho^{\mu_L}_{n_l n_r,k,q}(r_\lambda).$$

**Architecture (data flow when bispinor=True):**

A bispinor run reads **two** centroid files. Each carries human-readable
feature-fit, source-WFN, and intended-channel provenance:

| File (suffix) | Intended channels | Built by | Used for |
|---|---|---|---|
| `centroids_frac_<N>.txt`         | $\tilde\gamma^0$ (charge) | `kmeans_cli` (default mode) | $\mu_L=0$ (charge channel) |
| `centroids_frac_<M>_current.txt` | $\tilde\gamma^{1,2,3}$ (current) | `kmeans_cli --density-mode current` | $\mu_L\in\{1,2,3\}$ |

`kmeans_cli` writes that provenance. At runtime, the two configured path roles
select the charge and transverse tables; ζ and restart provenance bind each
role to the content hash of its FFT-index table. The loader does not infer a
table's role by parsing comment text. $N$ and $M$ may differ because orbit
closure and pivoted-Cholesky pruning act on different weights. Wavefunctions
are sampled once per centroid set. Under
`low_mem_bands=true`, each sample is immediately converted to its own
two-face carrier; charge and transverse carriers remain separate because
their centroid axes have different meanings and extents. The resulting
$\zeta^{\mu_L}_q$ files must stay paired with the centroid identity in their
fit provenance.

For a band window $B$, define
$D_{B,k}(r)=\sum_{n\in B}|\Psi_{nk}(r)\rangle\langle\Psi_{nk}(r)|$.
The current k-means mass is the feature-row norm

$$m_J(r)=\sqrt{\sum_{k,i}w_k\,\mathrm{Tr}
[D_{L,k}(r)\alpha_iD_{R,k}(r)\alpha_i]/\alpha_{\rm FS}^2}.$$

The charge mass replaces $\alpha_i$ by the identity. It is the ordinary band
density for one k point and an equal scalar window; generally it is the norm
of the k-stacked feature row. The builder contracts bands into $D_L,D_R$ and
never forms transition densities explicitly.

**Effect on MoS2** (`run_zeta_proper_gram.py`, aggregate over 3.3M band-pair × k × q × test-point samples per channel):

| $\mu_L$ | (a) channel-aware centroids | (b) scalar-only centroids | (a)/(b) |
|---|---:|---:|---:|
| 0 | 7.77e-5 | 7.77e-5 | 1.000 (same centroid set) |
| 1 | 3.04e-3 | 3.48e-3 | 0.875 |
| 2 | 3.07e-3 | 3.56e-3 | 0.863 |
| 3 | 2.89e-3 | 3.41e-3 | 0.848 |

This historical test showed a consistent 13–15% reconstruction-error
improvement on the i-channels. The current set had about 4% more centroids
(668 versus 640) because of orbit closure, so it does not isolate the effect
of the density weight. It is centroid-selection evidence, not a validation of
the current coupled solver schedule.

### 4.1 Current CCT, sharding, and solve schedule

The production fit uses the Schur CCT construction. Its charge member is
positive semidefinite and defaults to the rank-revealing charge
pseudo-inverse. Each transverse $\tilde\gamma^i$ member is Hermitian but
indefinite. It therefore uses pivoted LU with the accepted trace-scaled ridge,
or the explicit `transverse_zeta_solve=rank_truncate` indefinite
pseudo-inverse. The old claim that all four channels share one Cholesky path
is false.

The principal layouts are:

```text
psi_mun[k,s,mu_X,n_Y]
C_q[mu_L,q,mu_X,nu_Y]
Z_mu123[mu_L,q,mu_X,r_Y]
zeta_q[q,mu_XY,r]
zeta_G[q,mu_XY,G]
```

Charge has its own centroid extent. The three transverse systems share the
current-centroid extent but remain distinct matrices and files. On an eligible
fresh fit, `_z_q_face_coupled_mu123` builds one leading-three-channel RHS. It
reuses the face-Y transform/scatter, the X-owner broadcast, and the
channel-independent left pair density. It evaluates the three right densities
in μ=1→2→3 order, keeping one right carry live at a time.

The solves are deliberately separate and ordered, not one flattened
three-channel solve. With `batch_reshard`, each channel's raw CCT and RHS move
from the 2-D face layout to whole matrices distributed over the q batch; local
JAX LU factor-and-solve runs once per r chunk. On the distributed route, each
channel is factored once into an opaque `distrib_la.FactorToken`, and the
provider applies that token to the 2-D-sharded RHS in every r chunk. No
provider factor is exposed or moved into `isdf`.

Each channel's persistent G-flat accumulator is parked in process-local host
memory. The active channel alone is restored for the canonical
`accumulate_rchunk_to_gflat` call and spilled immediately afterward. A final
barrier makes all three fits reach completion before μ=1 starts writing; final
writes, closes, and provenance remain μ=1→2→3.

Coupling is automatic only when all three transverse files are fresh, the
bounded face-Y cache is selected, host spill fits its node-RAM cap, and the
complete device live set fits the planner budget. Partial reuse fits only the
missing files on the sequential schedule. Capacity failure also falls back to
the sequential schedule without changing the fit or the requested public
`distrib_la` route.

The detailed loop and sharding contract is in
[ζ-fit CCT on the two-face carrier](architecture/zeta_fit_face_psi_cct.md#current-coupled-transverse-schedule-2026-08-29).
Closed-form memory and capacity policy belong to the
[memory model](architecture/memory-model.md); this page does not duplicate
them.

### 4.2 Historical proper-Gram alternative

The original design proposed a literal positive-semidefinite band-pair Gram
for every vertex. Production retained the cheaper Schur CCT and handles the
transverse indefiniteness in the solver. A proper Gram remains a possible
scientific alternative if a future accuracy study justifies its larger
band-pair cost; it is not the implemented path.

## 5. File map

| File | Phase-1 change |
|---|---|
| [`src/common/bispinor_init.py`](../src/common/bispinor_init.py) | Single σ·p implementation; requires an explicitly Cartesian reciprocal basis in Bohr⁻¹. |
| [`services/wfn_loader/`](services/wfn_loader.md) | The WFN-format boundary folds `wfn.blat` into raw `wfn.bvec` once before calling the lift. |
| [`src/gw/isdf_fitting.py`](../src/gw/isdf_fitting.py) + [`src/isdf/core.py`](../src/isdf/core.py) | One fit driver and one pair-density/CCT/Z/solve implementation for charge and transverse vertices; the private coordinator only schedules those owners. |
| [`services/symmetry_maps/`](services/symmetry_maps.md) (`import symmetry_maps`) | Canonical typed operation, Cartesian, spinor, and translation actions used by the fit, IBZ writer, and V reconstruction. |
| [`src/centroid/sampling_metric.py`](../src/centroid/sampling_metric.py) | Shared charge/current feature-Gram diagonal from streamed subspace projectors. |
| [`src/centroid/kmeans_cli.py`](../src/centroid/kmeans_cli.py) | `--density-mode {scalar,current}` flag; auto-suffixes the output (`""` / `"_current"`); writes feature-fit, source-WFN, and intended-channel provenance. |
| [`src/gw/sigma_x_bispinor.py`](../src/gw/sigma_x_bispinor.py) | $D^{ij}_{\rm bare}$ + $\tilde\gamma^i G^0 \tilde\gamma^j$ contraction for $\Sigma^B_{\alpha\beta}$. |
| [`src/gw/v_q_bispinor.py`](../src/gw/v_q_bispinor.py) + [`src/gw/compute_vcoul.py`](../src/gw/compute_vcoul.py) | Channel-aware $V_q$ orchestration and the Coulomb/transverse projector kernel. |
| [`src/gw/cohsex_sigma.py`](../src/gw/cohsex_sigma.py), [`ppm_sigma.py`](../src/gw/ppm_sigma.py) | Parameterise spinor axis size; $\tilde\gamma^0$ vertices made explicit (identity, but expose contraction shape for phase-2). |
| [`src/gw/gw_init.py`](../src/gw/gw_init.py), [`gw_config.py`](../src/gw/gw_config.py) | Resolve independent reuse, select coupled versus sequential transverse fitting, and bind the `bispinor_gw` policy. |

## 6. Phasing

| Stage | Deliverable | Status |
|---|---|---|
| M0 | This doc | done |
| M1 | Bispinor lift end-to-end + symmetry service | done for the production bare-transverse path |
| M2 | Four-density ISDF infra: pair-density helpers, channel-aware centroid mode | done |
| M3 | $\chi^0_{00}$, $W_{00}$ on bispinor $G^0$ | implemented |
| M4 | $\Sigma^C$ with explicit $\tilde\gamma^0$ vertices | implemented |
| M5 | $\Sigma^B$ from bare $D^{ij}$ | implemented in `sigma_x_bispinor.py` |
| M6 | External DHFB-Breit reference comparison | open validation work |

## 7. Validation strategy

1. **Identity-vertex regression**: the $\tilde\gamma^0=I$ charge kernels must agree between the legacy and face carriers at fp64 noise.
2. **$c\to\infty$ limit**: as $\alpha_{\rm FS}\to0$, $\psi_S\to0$ and $\Sigma^B\to0$.
3. **Light-atom DHFB-Breit reference** (Ne/Ar/Kr): order-of-magnitude match for $\Sigma^B$ core corrections.  Quantitative match is phase-2 (transverse screening matters at ~10%).

## 8. Out of scope (phase-2)

$\chi^{0i},\chi^{ij}$ • $W^{\mu\nu}$ Dyson (4×4 matrix) • retarded Breit ($D^{ij}(\omega)$) • Sternheimer-side bispinor source • higher-order kinetic balance (DKH4 / σ·v).

## 9. Open questions

1. Does a proper positive-semidefinite band-pair Gram improve transverse ζ accuracy enough to justify replacing the cheaper indefinite Schur CCT?
2. What external DHFB/Breit reference should certify the absolute transverse self-energy, beyond internal carrier and mesh parity?
3. Which screened transverse terms belong in the first phase-2 model, and what q→0 completion must accompany them?

## 10. Reference

Historical validation provenance was recorded under the machine-local
`runs/MoS2/B_bispinor_pd_smoke_2026-05-02/` directory. It is not shipped and
does not certify the current coupled schedule.

Internal: [`docs/theory/physics.md`](theory/physics.md) (scalar ISDF GW) and [`docs/architecture/codebase.md`](architecture/codebase.md). *These two links named `PHYSICS_COMPREHENSIVE.md` / `CODEBASE_COMPREHENSIVE.md` until 2026-08-06; both were deleted in the 2026-07-31 restructure, and the banner at the top of this page had already recorded their successors while these links kept pointing at the graves.*

## 11. q=Γ measurements on the bi4 deck (2026-08-01 audit)

The argument for how each `(μ_L, ν_L)` tile behaves at `q → 0`, and the
correction the code applies, live in
[Four-current heads and frequency](theory/four-current-head-corrections.md)
§2. This section keeps only the provenance measurements that page cites.
Deck: MoS2 4×4, 402 charge + 143 transverse centroids, P=4, `sys_dim=2`,
job 7885325 (Frontera; artifacts were machine-local and are not shipped).
These are historical measurements of the former TT-slot overlay, not a
current route claim: at `34228021` no deck key reaches that overlay, while
the packed slab routes obtain `⟨D_TT⟩` from the coupled Γ completion. See the
single [implementation-status statement](theory/four-current-head-corrections.md#four-current-phase-status).

| quantity | value |
|---|---|
| `vc0 = ⟨v⟩_mBZ` (bare, 4×4) | 2443.3 a.u. |
| `⟨v t^{11}⟩ / vc0`, `⟨v t^{22}⟩ / vc0`, `⟨v t^{33}⟩ / vc0` | 0.4993, 0.5007, 1.0000 |
| `⟨v t^{12}⟩ / vc0`; `t^{i3}` | 4e-4; exactly 0 (in-plane cell) |
| `‖ζ_T^i(Γ, μ, G=0)‖` | 1.17–1.21e3 |
| Frobenius ratio, missing rank-1 head / stored q=Γ TT slab (11/22/33) | 0.97 / 1.04 / 6.0 |
| whole q=Γ TT term (Γ-zeroed leg): Σ_X diag max / mean; eqp max | 0.347 / 0.059 meV; 0.347 meV |
| whole q=Γ TT term: share of the −1.553 eV Σ^B trace | −0.122 eV (7.8 %) |
| missing G=0 head (injected leg): Σ_X diag max / mean; eqp max | 0.209 / 0.037 meV; 0.209 meV |
| missing G=0 head: change of the Σ^B trace | −0.076 eV (4.9 %) |

The tabulated `⟨v t^{ij}⟩` are positive moments of the geometric projector;
the stored Coulomb-gauge slot is their negative (§2). The restart legs shared
every bit except the TT q=Γ slabs (A/AR restart reproducibility exact 0).
