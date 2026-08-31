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

## 11. q=Γ treatment of the CC, TT and CT tiles (2026-08-01 audit)

Live addition, not part of the historical record above: the argument and
measurement for how each (μ_L, ν_L) tile behaves at q → 0, for the Coulomb
kernel the code actually builds.  Measured on the bi4 deck (MoS2 4×4, 402
charge + 143 transverse centroids, P=4), job 7885325; artifacts under
`/scratch2/08271/jackmc/bispinor_gamma_check/`.

**Sign clarification (2026-08-25).** The tabulated `⟨v t^{ij}⟩` values below
are positive moments of the geometric projector
$P_T^{ij}=\delta_{ij}-\hat K_i\hat K_j$.  The physical Coulomb-gauge spatial
propagator slot is their negative, $D^{ij}_{TT}=-\langle vP_T^{ij}\rangle$,
as in §2.  The historical Sigma-direction sentences below predate that sign
repair; their magnitudes remain provenance, not predictions for the corrected
operator.

**The kernel as built.**  Every bispinor system currently runs `sys_dim=2`,
so `compute_v_q_per_G` evaluates the slab-truncated kernel

    v(K) = (8π/V_cell) (1 − e^{−z_c K_∥} cos(K_z z_c)) / K²,  z_c = π/b_z,

and zeroes the K = 0 slot (`denom_zero` guard).  For in-plane K → 0 this
kernel diverges as `(8π/V_cell)(z_c/K − z_c²/2 + O(K))` — a 1/K cusp, not
the 3D 1/K² pole.  The cusp is integrable in the 2D zone integral, so the
correct stand-in for the zeroed slot in a discrete q-sum is the mini-BZ
cell average of the integrand, and the error of dropping it decays only as
~1/√N_k — the same slow decay that makes the CC head correction mandatory
in 2D.

**CC tile — correct as implemented.**  The charge structure factor obeys
M_{mn}(k, q→0, G=0) → δ_{mn}, so the divergent slot contributes
band-diagonally with unit weight.  The mini-BZ average vc0 = ⟨v⟩_mBZ
(`gw/coulomb/slab_2d.py::q0_average`) enters Σ through the band-diagonal
head terms (`gw/head_correction.py`) and, where a (μ,ν)-basis form is
needed, as the rank-1 `(vc0/V_cell)·conj(ζ_C(0,μ,0))ζ_C(0,ν,0)` update.
Nothing further is required.

**TT tiles — a real, sub-meV (at 4×4), missing correction at q=Γ, G=0.**
The transverse structure factor is the current matrix element
j^i_{mn}(k, q→0, G=0) = ⟨mk| α^i |nk⟩.  No orthogonality argument kills
it: diagonal elements are band velocities (in units of c), and with
spin-orbit coupling ⟨u_↑|u_↓⟩ ≠ 0 makes spin-mixing elements generically
nonzero as well.  The projector t^{ij}(K̂) has no q → 0 limit
(direction-dependent), but v·t has a finite mini-BZ cell average; for the
in-plane (K_z = 0) cell, measured on the bi4 deck: ⟨v t^{11}⟩ = 0.4993·vc0,
⟨v t^{22}⟩ = 0.5007·vc0, ⟨v t^{33}⟩ = vc0 exactly, ⟨v t^{12}⟩ = 4e-4·vc0,
t^{i3} ≡ 0 exactly (vc0 = 2443.3 a.u. bare at 4×4).  The code stores 0 in
the q=Γ, G=0 slot of every TT tile and applies no substitute; the
transverse ζ structure factors ζ_T^i(Γ, μ, G=0) are far from zero
(vector norms 1.17–1.21e3), and the missing rank-1 head
`(⟨v t^{ij}⟩/V_cell)·conj(ζ_T^i(0,μ,0))ζ_T^j(0,ν,0)` is comparable to the
whole stored q=Γ TT slab (Frobenius ratio 0.97 / 1.04 / 6.0 for the
11/22/33 tiles — largest in the out-of-plane channel, which carries the
full vc0 weight).

Measured Σ-level effect (restart legs sharing every bit except the TT q=Γ
slabs; A/AR restart-reproducibility exact-0):

* whole q=Γ TT term as currently included (all G shells; Γ-zeroed leg):
  Σ_X diag max 0.347 meV, mean 0.059 meV; eqp max 0.347 meV;
  −0.122 eV of the −1.553 eV Σ^B trace (7.8%).
* missing G=0 mini-BZ head (injected leg): Σ_X diag max 0.209 meV,
  mean 0.037 meV; eqp max 0.209 meV; deepens the Σ^B trace by −0.076 eV
  (4.9% of Σ^B).  Off-diagonal (i≠j) tiles unaffected at print precision,
  consistent with ⟨v t^{i≠j}⟩ ≈ 0.

Verdict: genuinely missing correction; absolute size sub-meV on 4×4 eqp
but ~5% of Σ^B itself, decaying only as ~1/√N_k.  Registered in the
sandbox defect register (KNOWN_LORRAX_ISSUES.md, bispinor section).
Smallest fix, if adopted: in
`gw/v_q_bispinor.py::_make_per_q_v_builder_for_tile`, replace the q=Γ,
G=0 slot of the TT builders with the mini-BZ average ⟨v t^{ij}⟩ (in-plane
Sobol average, same sampler as `q0_average`); single file, deterministic
v-table change, gauge-preserving.  It changes physics defaults and
baselines, so landing it is an owner re-pin decision.

**CT/TC tiles — exactly zero at the bare level, at every q.**  In Coulomb
gauge the bare propagator has t^{0i} ≡ 0 identically; this is a property
of the interaction, independent of the vertices, so spin-orbit-induced
nonzero charge↔current matrix elements never meet a nonzero kernel
element.  No correction is missing from phase-1 (bare-Breit) Σ^B.  The
caveat belongs to phase 2: once W^{μν} is screened, the density–current
response χ^{0j} generates wing blocks in which the divergent charge factor
multiplies a finite transverse factor; those wings will need their own
q → 0 (mini-BZ) treatment when transverse screening lands (§8).
