# Bispinor GW — Phase-1 Design (DHF + Bare-Breit)

> **SUPERSEDED IN PART (2026-07-31).** Historical design document; the physics
> (§§1–4) still describes the implementation, but the code map has moved:
>
> - Σ^B assembly lives in `src/gw/sigma_x_bispinor.py` (the planned
>   `src/gw/breit_sigma.py` was never created); V_q^{μν} tiles in
>   `src/gw/v_q_bispinor.py`.
> - Transverse ζ-solve policy (supersedes §4's "Cholesky + 1e-14 ridge for all
>   four channels"): the transverse CCT^μ is Hermitian but indefinite, solved by
>   pivoted LU with a 1e-12·|tr|/n ridge (`isdf/core.py:2536` `solve_zeta`);
>   the charge channel defaults to `charge_zeta_solve = rank_truncate`
>   (rank-revealing eigh pseudo-inverse, `zeta_rcond` = 1e-8).
> - File-map rows that no longer exist: `src/common/load_wfns.py`,
>   `src/common/isdf_fitting.py` (now `src/gw/isdf_fitting.py` +
>   `src/isdf/core.py`), `src/centroid/centroid_io.py` (the `density:` header
>   is written/checked by `kmeans_cli` and the GW loaders directly),
>   `docs/PHYSICS_COMPREHENSIVE.md` /
>   `docs/CODEBASE_COMPREHENSIVE.md` (see `docs/theory/physics.md` /
>   `docs/architecture/codebase.md`), and the `runs/MoS2/...` validation dirs
>   (machine-local, not shipped).
> - Current usage: manual ch. 8 (bispinor GW) and `docs/drivers.md`
>   (two-centroid-file convention, `--density-mode current`).

**Status:** historical design record (was: in flight on `agent-B/bispinor-design`)
**Last update:** 2026-05-02; superseded-in-part header 2026-07-31

## 1. Scope

DHF + bare-Breit GW with bispinor wavefunctions:

- $\chi^0\equiv\chi^0_{00}$, $W\equiv W_{00}$ — Coulomb screened in RPA (existing scalar code, ns=4 spin axis).
- $\Sigma_{\alpha\beta}=\Sigma^C_{\alpha\beta}+\Sigma^B_{\alpha\beta}$.  $\Sigma^C$ uses $W_{00}$; $\Sigma^B$ uses the **bare** $D^{ij}$ — no transverse screening, no retardation.
- Four ISDF $\zeta$ bases, one per $\tilde\gamma^{\mu_L}$, on **two centroid sets**: scalar charge density for $\mu_L=0$, Gordon-decomposed Pauli current for $\mu_L\in\{1,2,3\}$.

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

$$D^{\mu_L\nu_L}(K) = \begin{pmatrix} 4\pi/|K|^2 & 0 \\ 0 & (4\pi/|K|^2)\,(\delta_{ij}-K_iK_j/|K|^2) \end{pmatrix},\quad K=q+G.$$

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

A bispinor run reads **two** centroid files, each self-identifying via a `density:` line in the header comment block:

| File (suffix) | Header `density:` | Built by | Used for |
|---|---|---|---|
| `centroids_frac_<N>.txt`         | `scalar`  | `kmeans_cli` (default mode)            | $\mu_L=0$ (charge channel) |
| `centroids_frac_<M>_current.txt` | `current` | `kmeans_cli --density-mode current`    | $\mu_L\in\{1,2,3\}$        |

Both files are loaded via [`centroid.centroid_io.read_centroids`](../src/centroid/centroid_io.py), which validates the `density:` field.  $N$ and $M$ may differ (orbit-closure and pivoted-Cholesky pruning produce different counts).  ψ at the centroids is loaded **twice** — once at the scalar-set indices, once at the current-set indices — and stored as two arrays.  The Σ-side contraction routes each $\mu_L$ to the appropriate ψ array; the resulting $\zeta^{\mu_L}_q$ tensors have different second-axis sizes, which downstream consumers must keep paired with the centroid set they were fit on.

**Current-density weight (Gordon-decomposed Pauli current):**

$$\vec j^{\,\rm Gordon}_{n,k}(r) = \underbrace{\mathrm{Im}\big[\psi_L^\dagger(r)\,\nabla\,\psi_L(r)\big]}_{\rm paramagnetic}
+ \tfrac{1}{2}\,\nabla\times\big[\psi_L^\dagger(r)\,\boldsymbol\sigma\,\psi_L(r)\big],
\qquad W_{\rm curr}(r)=\sum_{n\in\rm occ,\,k,\,i}|j^{\,\rm Gordon}_{n,k,i}(r)|^2.$$

Built from $\psi_L$ (no bispinor lift dependency, no $\alpha_{\rm FS}$ suppression) by [`current_density.build_current_density`](../src/centroid/current_density.py).

**Effect on MoS2** (`run_zeta_proper_gram.py`, aggregate over 3.3M band-pair × k × q × test-point samples per channel):

| $\mu_L$ | (a) channel-aware centroids | (b) scalar-only centroids | (a)/(b) |
|---|---:|---:|---:|
| 0 | 7.77e-5 | 7.77e-5 | 1.000 (same centroid set) |
| 1 | 3.04e-3 | 3.48e-3 | 0.875 |
| 2 | 3.07e-3 | 3.56e-3 | 0.863 |
| 3 | 2.89e-3 | 3.41e-3 | 0.848 |

A small but consistent ~13–15% reconstruction-error improvement on the i-channels.  The current set has ~4% more centroids (668 vs 640) due to orbit-closure inflation — count-matched comparison would be needed to claim the full 13–15% is structural.  CrI3 will probably show a larger gap; the architecture is in place for whatever the heavier-element data shows.

**Gram and solver — same path for all four channels:**

$$K_q(\mu,\lambda) = \sum_{n_l, n_r, k_l}\rho^*_{n_l n_r k_l q}(\mu)\,\rho_{n_l n_r k_l q}(\lambda),\quad
\rho_{n_l n_r k_l q}(r)=\sum_{ab}\psi^*_{l,n_l,k_l,a}(r)\,\tilde\gamma_{ab}\,\psi_{r,n_r,k_l+q,b}(r).$$

$K_q$ is a literal Gram of band-pair vectors → PSD for every $\tilde\gamma$.  All four channels use **Cholesky + 1e-14·|trace| ridge**.

Driver: [`runs/MoS2/B_bispinor_pd_smoke_2026-05-02/run_zeta_proper_gram.py`](../../runs/MoS2/B_bispinor_pd_smoke_2026-05-02/run_zeta_proper_gram.py).  On MoS2 3×3, all four channels confirm $\lambda_{\min}>0$, Cholesky residual ~1e-19, and single-band-pair reconstruction at $q=\Gamma$ goes to the sub-1% level (8.3e-6 / 4.4e-3 / 6.4e-3 / 7.1e-3 for $\mu_L=0,1,2,3$).

### 4.1 Why not the cheaper Schur factorization?

The Schur-product form $\mathrm{CCT}_q(\mu,\lambda)=\sum_k P_l^*(\mu,\lambda;k)\odot P_r(\mu,\lambda;k)$ (with $P=\sum_n\psi^*\tilde\gamma\psi$ already spin-summed) costs only $O(N_k n_{r\mu}^2)$ — a factor $N_l N_r$ cheaper than $K_q$.  However, expanding $\langle v|\mathrm{CCT}|v\rangle$ groups spin and spatial indices the wrong way and produces a kernel $M=\tilde\gamma^*\otimes\tilde\gamma$ on the 16-dim spin-pair index, with eigenvalues equal to products of $\tilde\gamma$'s eigenvalues.  For $\tilde\gamma^0=I_4$ this is $I_{16}\succeq0$; for $\tilde\gamma^i=\alpha^i$ (eigenvalues $\pm1$) it has 8 eigenvalues $+1$ and 8 eigenvalues $-1$ → indefinite.  The proper Gram avoids this by contracting spin at a single spatial point, leaving $\langle v|K_q|v\rangle=\sum|\sum_\mu v_\mu\rho|^2\ge0$ trivially.

For the MoS2 smoke the $N_l N_r$ band-pair factor is ~128 (8 valence × 16 conduction); $K_q$ build is ~0.5 s on 1 GPU.  For larger systems we will need chunked-band-pair accumulation in the $K_q$ build to stay within memory; not currently implemented.

## 5. File map

| File | Phase-1 change |
|---|---|
| [`src/common/bispinor_init.py`](../src/common/bispinor_init.py) | Single σ·p implementation; requires an explicitly Cartesian reciprocal basis in Bohr⁻¹. |
| [`services/wfn_loader/`](services/wfn_loader.md) | The WFN-format boundary folds `wfn.blat` into raw `wfn.bvec` once before calling the lift. |
| [`src/common/isdf_fitting.py`](../src/common/isdf_fitting.py) | Adds `compute_pair_density_with_vertex` and `compute_pair_density_lorentz` (γ̃-vertex generalization of the existing scalar helper). |
| [`services/symmetry_maps/`](services/symmetry_maps.md) (`import symmetry_maps`) | 4×4 spinor rotations (extending the 2×2 Pauli ones). **Not yet implemented**; required at M1 for symmetry-aware unfolding. The 3-vector Lorentz mixing that DOES exist is `unfold_v_q_bispinor_lorentz` — read its `R_proper_table` convention note before feeding it a rotation table. |
| **[`src/centroid/current_density.py`](../src/centroid/current_density.py)** *(new)* | Builds $W_{\rm curr}(r)$ from the canonical lifted bispinor and direct $\Psi^\dagger\alpha_i\Psi/\alpha_{\rm FS}$ vertex; the Gordon form remains the independent parity identity. |
| **[`src/centroid/centroid_io.py`](../src/centroid/centroid_io.py)** *(new)* | `read_centroids(path)` parses the `density:` header line; downstream consumers dispatch on it. |
| [`src/centroid/kmeans_cli.py`](../src/centroid/kmeans_cli.py) | `--density-mode {scalar,current}` flag; auto-suffixes the output (`""` / `"_current"`); writes `density:` and `weight:` header lines. |
| **`src/gw/breit_sigma.py`** *(new, future)* | $D^{ij}_{\rm bare}$ + $\tilde\gamma^i G^0 \tilde\gamma^j$ contraction → $\Sigma^B_{\alpha\beta}$. |
| [`src/gw/compute_vcoul.py`](../src/gw/compute_vcoul.py) | Factor $v(q+G)$ + transverse projector helper for $D^{ij}$. |
| [`src/gw/cohsex_sigma.py`](../src/gw/cohsex_sigma.py), [`ppm_sigma.py`](../src/gw/ppm_sigma.py) | Parameterise spinor axis size; $\tilde\gamma^0$ vertices made explicit (identity, but expose contraction shape for phase-2). |
| [`src/gw/gw_jax.py`](../src/gw/gw_jax.py), [`gw_config.py`](../src/gw/gw_config.py) | Activate `bispinor`; add `breit_sigma` sub-flag. |

## 6. Phasing

| Stage | Deliverable | Status |
|---|---|---|
| M0 | This doc | done |
| M1 | Bispinor lift end-to-end + 4×4 SymMaps; identity-regression vs ns=2 | partial (lift ✓, SymMaps pending) |
| M2 | Four-density ISDF infra: pair-density helpers, channel-aware centroid mode | done |
| M3 | $\chi^0_{00}$, $W_{00}$ on bispinor $G^0$ | pending |
| M4 | $\Sigma^C$ with explicit $\tilde\gamma^0$ vertices | pending |
| M5 | $\Sigma^B$ from bare $D^{ij}$ | pending |
| M6 | DHFB-Breit reference comparison (atomic Ne/Ar/Hg or solid-state Bi) | pending |

## 7. Validation strategy

1. **Identity-vertex regression**: with `breit_sigma=False` and a scalar-relativistic input, ns=4 result must reproduce the existing ns=2 result at fp64 noise.
2. **$c\to\infty$ limit**: zero $\alpha_{\rm FS}$ → $\psi_S\to 0$ → $\Sigma^B\to 0$.  Single-line knob.
3. **Light-atom DHFB-Breit reference** (Ne/Ar/Kr): order-of-magnitude match for $\Sigma^B$ core corrections.  Quantitative match is phase-2 (transverse screening matters at ~10%).

## 8. Out of scope (phase-2)

$\chi^{0i},\chi^{ij}$ • $W^{\mu\nu}$ Dyson (4×4 matrix) • retarded Breit ($D^{ij}(\omega)$) • Sternheimer-side bispinor source • channel-aware centroid pruning (phase-1 reuses charge-channel pivoted-Cholesky n_val/n_cond) • higher-order kinetic balance (DKH4 / σ·v).

## 9. Open questions

1. Switch i-channels from Schur CCT + eigh-pinv to proper Gram $K_q$ + Cholesky if $\Sigma^B$ accuracy is insufficient (cost $\sim N_l N_r$× extra in Gram build).
2. Renormalize $\Psi$ post-lift so $\|\Psi\|^2=1$ exactly?  Effect is $\alpha_{\rm FS}^2$-small but systematic.
3. ζ storage: single H5 with leading $\mu_L$ axis vs. four files.  Lean toward single H5 with a `lorentz_complete` mask.

## 10. Reference

Key validation log: [`runs/MoS2/B_bispinor_pd_smoke_2026-05-02/`](../../runs/MoS2/B_bispinor_pd_smoke_2026-05-02/) — pair-density smoke (`run_pd_smoke.py`), ζ-fit smoke (`run_zeta_fit.py`), channel-aware ζ-fit (`run_zeta_channel_aware.py`), bvec-units diagnostic (`check_bvec_units.py`).

Internal: [`docs/theory/physics.md`](theory/physics.md) (scalar ISDF GW) and [`docs/architecture/codebase.md`](architecture/codebase.md). *These two links named `PHYSICS_COMPREHENSIVE.md` / `CODEBASE_COMPREHENSIVE.md` until 2026-08-06; both were deleted in the 2026-07-31 restructure, and the banner at the top of this page had already recorded their successors while these links kept pointing at the graves.*

## 11. q=Γ treatment of the CC, TT and CT tiles (2026-08-01 audit)

Live addition, not part of the historical record above: the argument and
measurement for how each (μ_L, ν_L) tile behaves at q → 0, for the Coulomb
kernel the code actually builds.  Measured on the bi4 deck (MoS2 4×4, 402
charge + 143 transverse centroids, P=4), job 7885325; artifacts under
`/scratch2/08271/jackmc/bispinor_gamma_check/`.

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
