# LORRAX Symmetry Conventions and Procedures

**Consolidates**: `reports/trs_sym_audit_2026-05-14/SYMMETRY_CONVENTIONS.md`, `reports/trs_sym_audit_2026-05-14/pr3_design.md`, the agent-scope notes under `reports/trs_sym_audit_2026-05-14/`, and the inline docstrings in `src/common/symmetry_maps.py`, `src/centroid/orbit_syms.py`, `src/file_io/mf_header.py`, `src/gw/v_q_g_flat.py`.

**Status**: describes the convention that the current driver (`gw.gw_jax`) and the V_q g-flat path (`gw.v_q_g_flat`) actually use, as of 2026-05-15. The BGW r-action convention $r' = \mathrm{mtrx}^{-1} \cdot r + \tau$ is load-bearing; everything else in this document flows from it. Verified end-to-end against BGW reference Σ_X on the three production gates: MoS2 3×3 SOC (0.090 meV), CrI3 6×6 30 Ry SOC (0.076 meV), Si 4×4×4 SOC (0.002 meV — the canonical non-symmorphic gate). Reading order: §1 (BGW convention), §2 (`SymMaps` and the TRS-augmented table), §3 (ψ unfold), §4 (centroid permutation and L-wrap), §5 (V_q unfold), §6 (symmorphic failure modes), §7 (sym-vs-nosym recipe), §8 (verified gates), §9–§11 (file index, quick reference, reference tests).

> **Note (2026-05-15).** This file supersedes the earlier "the matrix in `wfn.sym_matrices` is the forward direct-space rotation U" reading that lived in the design notes through mid-May 2026. The corrected reading is `wfn.sym_matrices = mtrx` (BGW's k-action / G-action matrix), with the real-space forward action being $r' = \mathrm{mtrx}^{-1} \cdot r + \tau$. This is the convention that closes Si Fd-3m's 96/96 atom orbits in `validate_atomic_symmetries`. See §1.4 for the source-level evidence and §1.5 for the (verified-empirically-correct) reciprocal-action matrix.

---

## Overview

LORRAX consumes mean-field wavefunctions in BerkeleyGW's HDF5 format (`WFN.h5`), which fixes the space-group representation we work in. The symmetry pipeline has four jobs:

1. Map every full-BZ k-point and q-point to an IBZ parent + sym-op pair.
2. Unfold IBZ wavefunctions $\psi_{\bar k}$ to full-BZ $\psi_k$ (`unfold_psi`), including the bispinor TRS rule.
3. Permute centroid coordinates under each space-group op and record the integer real-space lattice wrap (`compute_centroid_sym_perm` → `sym_perm`, `L_table`).
4. Unfold IBZ Coulomb-matrix tiles $V_q$ to the full BZ (`unfold_v_q`), folding in the umklapp phase from (3) and the TRS conj on the augmented half.

The non-trivial part of all of this is bookkeeping: BGW stores one matrix (`mtrx`) per spatial sym op, and four different physical actions can be described in terms of it. §1 nails down which is which. §2–§5 describe the LORRAX-side implementation. §6 catalogs the silent-on-symmorphic failure modes that motivate Si as the canonical regression. §7 is the production recipe; §8 is the gate table.

---

## 1. The BGW Convention

This section is the source of truth: everything in LORRAX must compose with the choice BGW makes.

### 1.1 Datasets in `WFN.h5`

LORRAX reads its symmetry data via `MfHeader` at `src/file_io/mf_header.py:65-69`:

```text
ntran          : int                       — /mf_header/symmetry/ntran
sym_matrices   : np.ndarray (ntran, 3, 3)  — /mf_header/symmetry/mtrx
translations   : np.ndarray (ntran, 3)     — /mf_header/symmetry/tnp = 2π·τ_frac
```

The WFN.h5 spec (`docs/docs_bgw/wfn.h5.spec:172-185`) declares the layout `mtrx[ntran, 3, 3]` and `tnp[ntran, 3]` with `tnp` described only as "fractional translations". The factor of $2\pi$ is implicit in the writer (`pw2bgw_qe7.2_with_spinor_mag.f90:826-850`):

$$\mathrm{tnp}_s = (S_s^{-1} \cdot \mathrm{ft}_s) \cdot 2\pi,$$

where `ft` is QE's fractional translation. The BGW reader recovers $\tau_\text{frac}$ in `sources/BerkeleyGW/Sigma/sympert_utils.f90:754`:

```fortran
ftrans(:) = syms%tnp(:, isym) / 2.0d0 / PI_D
```

LORRAX divides by $2\pi$ at every consumer: `SymMaps.validate_atomic_symmetries` (`symmetry_maps.py:788`), `compute_centroid_sym_perm` (`orbit_syms.py:344`), `build_real_space_syms` (`orbit_syms.py:57`). The `MfHeader.translations` field stores the raw BGW form; helpers that need $\tau_\text{frac}$ must apply the `/(2π)` themselves. `unfold_psi` is the one exception — see §3.1.

### 1.2 The Four Actions of `mtrx`

Let $S = \mathrm{mtrx}_s$ (= `sym_matrices[s]`) and $\boldsymbol{\tau} = \mathrm{tnp}_s / (2\pi)$. In **column form** (treating $\mathbf r$ as a column vector):

| Physical action | Formula | Used by |
|---|---|---|
| Direct-space forward | $\mathbf r' = S^{-1} \cdot \mathbf r + \boldsymbol{\tau}$ | `validate_atomic_symmetries`, the forward direction in the orbit-aware kmeans helpers |
| Direct-space inverse | $\mathbf r' = S \cdot (\mathbf r - \boldsymbol{\tau})$ | `compute_centroid_sym_perm` (this is the **source-direction** decomposition driving the V_q unfold; see §4) |
| Reciprocal forward (k, q, G) | $\mathbf q' = S \cdot \mathbf q$ | BGW's `matmul(syms%mtrx, k)`; LORRAX's `unfold_psi` G-rotation; reconstruction check on `irr_idx_q / sym_idx_q` (§2.3) |
| Reciprocal inverse | $\mathbf q' = S^{-1} \cdot \mathbf q$ | Not used directly |

The row-form equivalent (numpy's natural layout) is `r' = r @ U.T + τ` for U = $S^{-1}$. `orbit_images` (`src/centroid/orbit_syms.py:73-92`) uses this convention with $\mathtt{Rinv}$ pre-computed.

### 1.3 The Sign of τ in Forward vs Inverse

Group theory pedantry that bit us repeatedly: in column form, the **forward** direct-space sym op acts on a real-space point as $g_s \cdot \mathbf r = S_s^{-1} \mathbf r + \boldsymbol{\tau}_s$, while the **inverse** op acts as $g_s^{-1} \cdot \mathbf r = S_s (\mathbf r - \boldsymbol{\tau}_s) = S_s \mathbf r - S_s \boldsymbol{\tau}_s$. Both are useful: `validate_atomic_symmetries` (`symmetry_maps.py:780`) uses the forward action with `rot = inv(mtrx)`; `compute_centroid_sym_perm` (`orbit_syms.py:351`) uses the inverse action `S·(r-τ)` because the user-spec V_q unfold formula (§5) is most naturally expressed via the source map under the inverse op.

### 1.4 How We Know This Is the Right Convention

Three independent pieces of evidence pin $r' = \mathrm{mtrx}^{-1} \cdot r + \tau$:

1. **pw2bgw writer** (`pw2bgw_qe7.2_with_spinor_mag.f90:826-850`): builds the stored `mtrx` from QE's spatial rotation `s(:,:,i)` via `invmat(3, r1, r2)` so `r2 = inv(s_QE)`, then writes `translation = (r2 · ft) * 2π`. The stored matrix is therefore the inverse of QE's spatial direct-space rotation; equivalently, **`inv(wfn.sym_matrices)` is the forward real-space rotation**.

2. **BGW Σ reader** (`sympert_utils.f90:748-755`): explicitly recovers the real-space rotation by transposing and inverting the stored `mtrx`, and reads $\tau$ as `tnp/(2π)`.

3. **Si Fd-3m empirical test**: `SymMaps.validate_atomic_symmetries` (`symmetry_maps.py:780-811`) loops over all 48 spatial sym ops of Si and applies `rot @ pos + τ` with `rot = inv(mtrx)`. Result: 96/96 atom mappings close mod-1. Using `rot = mtrx` instead gives 48/96, because the diamond glide partners cross-cancel under the wrong direction.

For symmorphic systems ($\tau = 0$), $S$ and $S^{-1}$ give the same orbit on any point because the space group is closed under inversion. CrI3 P-3 and MoS2 P-6m2 both fall in this category. The bug therefore did not surface until Si Fd-3m, with its 24 glide ops at $\tau = (1/4, 1/4, 1/4)$ and friends, was exercised. The earlier-confusing CrI3 empirical match (`wfn.sym_matrices[1]` numerically equals $U_{C3}$ in hex coords) reconciles as follows: for an *abelian* group, $\{U_s\}$ and $\{U_s^{-1}\}$ generate the same orbit, so the matrix-direction choice has no observable consequence.

### 1.5 The Reciprocal-Action Matrix and `sym_mats_k`

`SymMaps.__init__` (`symmetry_maps.py:529-543`) builds:

```python
self.sym_matrices = wfn.sym_matrices[:wfn.ntran]                # mtrx, (ntran, 3, 3)
self.sym_mats_k   = self.sym_matrices.transpose(0, 2, 1).copy() # the k-action matrix
self.sym_mats_k   = np.concatenate([self.sym_mats_k,
                                    -self.sym_mats_k], axis=0)  # TRS-augmented
```

The transpose looks suspicious at first read (§1.2 says the BGW k-action is $q' = \mathrm{mtrx} \cdot q$, not $\mathrm{mtrx}^T \cdot q$), but is in fact the correct convention for LORRAX's numpy-internal representation, **verified by the Si 4×4×4 SOC gate passing at 0.002 meV** with this definition in place. The reconciliation: LORRAX stores k as integer-tuple rows and applies `sym_mats_k[s] @ k` as a left matrix-vector product on a *row vector*, which is algebraically `k @ S.T` for a column-vector $k$. So `sym_mats_k = mtrx.T` consumed as `sym_mats_k @ k_row` is the same operation as `mtrx @ k_col` consumed as a column-form left-multiply. Audit H1 (`reports/trs_sym_audit_2026-05-14/SYMMETRY_CONVENTIONS.md`) checked this empirically: for Si full-BZ k-points, the reconstruction `sym_mats_k[sym_idx_q[i]] @ q_irr_kgrid_int[irr_idx_q[i]] (mod kg) == kvecs_asints[i]` holds for every row.

The TRS augmentation doubles the length to $2 \cdot n_\text{tran}$. Rows $[0, n_\text{tran})$ are the spatial-only sym matrices; rows $[n_\text{tran}, 2 n_\text{tran})$ are their negatives, representing $K \circ \{S | \tau\}$ where $K$ is complex conjugation. The sign lives on the k-side because TRS reverses momenta but leaves real-space coordinates fixed.

### 1.6 Group Composition

Composing two BGW r-actions $g_a, g_b$ with product $S_c = S_a \cdot S_b$:

$$g_c \cdot \mathbf r = g_a(g_b \mathbf r) = S_a^{-1}(S_b^{-1} \mathbf r + \boldsymbol\tau_b) + \boldsymbol\tau_a = (S_a S_b)^{-1} \mathbf r + (S_a^{-1} \boldsymbol\tau_b + \boldsymbol\tau_a).$$

Matching to $g_c \cdot \mathbf r = S_c^{-1} \mathbf r + \boldsymbol\tau_c$:

$$\boxed{\boldsymbol\tau_c \;=\; S_a^{-1} \cdot \boldsymbol\tau_b + \boldsymbol\tau_a \pmod{\mathbb Z^3}.}$$

Note the asymmetry — only $\tau_b$ gets rotated. On Si Fd-3m's 48 ops this closure rule passes 2304/2304 pairs (0 fails). On CrI3 P-3 and MoS2 (symmorphic, $\tau = 0$) it trivially closes.

---

## 2. LORRAX's `SymMaps` Class

`src/common/symmetry_maps.py:416-1143` is the LORRAX-side handle on all of the above. It is constructed from a `WFNReader` and exposes the tables the GW driver consumes.

### 2.1 Constructed Arrays

```text
sym_matrices       : (ntran, 3, 3) int32    = wfn.sym_matrices[:ntran]    (= BGW mtrx)
sym_mats_k         : (2·ntran, 3, 3) int32  = [mtrx.T,  -mtrx.T]          (see §1.5)
translations       : (ntran, 3) float64     = wfn.translations[:ntran]    (RAW BGW form, = 2π·τ_frac)
R_grid             : (ntran, 3, 3) int32    = round(mtrx)
Rinv_grid          : (ntran, 3, 3) int32    = round(inv(mtrx))
R_cart             : (2·ntran, 3, 3) float  = avec.T @ mtrx @ inv(avec.T), with -R for TRS half
U_spinor           : (ntran, 2, 2) complex  = Markley-quaternion(R_cart[:ntran])
```

`U_spinor` is restricted to the SPATIAL half. The TRS-row spinor is constructed lazily inside `unfold_psi` as $i\sigma_y \cdot \overline{U_s}$; see §3.2. Before PR3 the TRS half was generated by feeding $-R$ into the SU(2) algorithm, which combined with the `det<0 ⇒ -R` flip on improper rotations produced a wrong-by-sign $U_\text{spinor}$ and a multi-eV Σ_X failure on TRS-folded k-points. The restriction-to-spatial-only fix lives at `src/common/symmetry_maps.py:584-593`.

### 2.2 Full-BZ ↔ IBZ Maps

Two pairs of parallel arrays:

- **k-side** (anchored to `wfn.kpoints`, the BGW-reduced IBZ):

  ```text
  irr_idx_k[ik_full]  : int32   — row of wfn.kpoints
  sym_idx_k[ik_full]  : int32   — row of sym_mats_k mapping IBZ k_bar → unfolded k_full
  unfolded_kpts       : (n_k_full, 3) float — full BZ in crystal coords
  ```

- **q-side** (q lives on the same kgrid as k since q = k − k′; IBZ derived as lex-min orbit reps):

  ```text
  irr_idx_q[iq_full]    : int32
  sym_idx_q[iq_full]    : int32
  q_irr_kgrid_int       : (n_q_ibz, 3) int — IBZ q's in integer kgrid coords
  q_irr_full_idx[i_irr] : int32 — full-BZ row index for IBZ q i_irr
  ```

Both pairs are built by `find_irreducible_bz_points` (§2.3). The k-side anchors to the WFN's reduced k-list; the q-side derives its IBZ as the lex-min orbit representatives over the full BZ.

### 2.3 `find_irreducible_bz_points`

`src/common/symmetry_maps.py:18-107`. Two branches:

**Branch A** (`irr_kgrid_int=None`, q-side): derive the IBZ as lex-min orbit representatives.

```python
images     = einsum('sij,qj->sqi', Smk, full) % kg     # (n_sym, n_full, 3)
image_keys = lex_encode(images)                         # (n_sym, n_full)
best_sym   = argmin(image_keys, axis=0)                 # (n_full,) lex-min sym per q
canon_keys = image_keys[best_sym, arange(n_full)]       # (n_full,)
irr_keys   = unique(canon_keys)
```

The IBZ list is the set of canonical (lex-min) keys; `sym_idx` is derived in a second loop matching each full-BZ q to its IBZ parent.

**Branch B** (`irr_kgrid_int` given, k-side): the IBZ is anchored to the WFN's stored k-list. For each full-BZ point, iterate over IBZ points (highest k_bar with any match wins, lowest sym for that k_bar — preserves bit-equality with the legacy `find_symmetry_ops_simple`).

Both branches operate on **integer kgrid coords** (mod `kg`), so all comparisons are integer-exact. The `sym_mats_k` used here is TRS-augmented (length $2 \cdot n_\text{tran}$), so `sym_idx >= ntran` flags TRS-augmented rows.

### 2.4 `q_irr_frac` (the IBZ q List Consumed by V_q Unfold)

Consumers of `unfold_v_q` (§5) need the IBZ q-points in fractional reciprocal coords with the BGW $(-1/2, 1/2]$ wrap. Built at `v_q_g_flat.py:214-221`:

```python
q_irr_wrapped = np.where(q_irr_kgrid_int > kg/2,
                          q_irr_kgrid_int - kg,
                          q_irr_kgrid_int).astype(np.float64)
q_irr_frac    = q_irr_wrapped / kg
```

Wrapping matches the convention used by the per-q `gvec_components` written on disk; an unwrapped q would put the L-phase on the wrong branch by an integer multiple of $2\pi$.

### 2.5 The TRS-Augmented Sym Table at a Glance

| Row range | Matrix at `sym_mats_k[s]` | r-action | ψ rule | ζ rule | V_q rule |
|---|---|---|---|---|---|
| $0 \le s < n_\text{tran}$ | $+\mathrm{mtrx}_s^T$ | $r \to \mathrm{mtrx}_s^{-1} r + \tau_s$ | $\psi_k = e^{-i(SG)\cdot\tau} U_s \psi_{\bar k}$ | $\zeta_{Sq,\mu} = e^{2\pi i q\cdot L_\mu} \zeta_{q,\alpha(\mu)}(S^{-1}(r-\tau))$ | $V_{Sq}[\mu,\nu] = e^{2\pi i q\cdot(L_\mu - L_\nu)} V_q[\alpha(\mu),\alpha(\nu)]$ |
| $n_\text{tran} \le s < 2 n_\text{tran}$ | $-\mathrm{mtrx}_{s-n_\text{tran}}^T$ | r fixed | $\psi_k = (i\sigma_y\overline{U_s})\,e^{+i(SG)\cdot\tau}\,\overline{\psi_{\bar k}}$ | $\zeta$ complex-conjugated | $V_{Sq} \to \overline{V_{Sq}}$ |

The TRS-row spinor factor is $i\sigma_y$, the standard SOC time-reversal operator $T = i\sigma_y K$:

$$i\sigma_y = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$$

hard-coded as `_I_SIGMA_Y` at `symmetry_maps.py:303`.

---

## 3. Wavefunction Unfold (`unfold_psi`)

`src/common/symmetry_maps.py:306-413`. Rotates the IBZ wavefunction $\psi_{\bar k}$ to a full-BZ k-point. Spinor + phase + TRS in one place.

### 3.1 Spatial Case ($s < n_\text{tran}$)

For an irreducible $\bar k$ with sym $\{S | \tau\}$ taking $\bar k$ to $k = S\bar k$ (mod a reciprocal lattice vector $k_{g0}$), the cell-periodic part transforms as

$$\psi_{k, b}(\mathbf G_\text{rot}) \;=\; U_\text{spinor}(S) \;\cdot\; e^{-i\, (S \cdot \mathbf G_{\bar k}) \cdot \boldsymbol{\tau}} \;\cdot\; \psi_{\bar k, b}(\mathbf G_{\bar k}),$$

where $\mathbf G_\text{rot} = S \cdot \mathbf G_{\bar k} + \mathbf{k}_{g_0}$ is the G-vector at the full-BZ k. The umklapp $\mathbf{k}_{g_0}$ is rebuilt by the `WfnLoader`'s `gvecs`; `unfold_psi` only computes the spinor + phase factors.

The τ-phase implementation (`symmetry_maps.py:387-412`):

```python
S_full   = sym_mats_k[sym_idx]               # (3,3) int — for spatial rows == sym_mats_k[s]
tau      = translations[s_spatial]           # (3,) float — RAW tnp (= 2π·τ_frac)
rotated  = (S_full @ g_bar.T).T              # (ngk, 3) float
phase    = np.exp(-1j * (rotated @ tau))     # (ngk,) — uses RAW tau, no extra 2π
cnk      = cnk * phase[None, None, :]
cnk      = einsum('jk,nkl->njl', U_spinor[s_spatial], cnk)
```

**Note `tau` is the raw BGW value** (= $2\pi \tau_\text{frac}$). The exponent is therefore $-i \cdot (S G_{\bar k}) \cdot 2\pi \tau_\text{frac}$, exactly BGW's `cmplx(0,-1)*tpi*dot_product(...)` in `Common/gmap.f90`. We do not divide by $2\pi$ in this helper so the einsum stays in BGW-native units. Outside `unfold_psi`, every consumer of `translations` divides by $2\pi$ to recover $\tau_\text{frac}$.

### 3.2 TRS-Augmented Case ($s \ge n_\text{tran}$)

For a TRS-augmented op $T \circ \{S | \tau\}$ with $T = i\sigma_y \cdot K$, the per-element rule is

$$\psi_{TS\bar k, b}(\mathbf G_\text{rot}) \;=\; \big(i\sigma_y \cdot \overline{U_\text{spinor}(S)}\big) \;\cdot\; e^{+i\,(S \cdot \mathbf G_{\bar k}) \cdot \boldsymbol{\tau}} \;\cdot\; \overline{\psi_{\bar k, b}(\mathbf G_{\bar k})}.$$

Two non-trivial pieces:
- The spinor factor is $i\sigma_y \cdot \overline{U_s}$, **NOT** $\overline{U_{\text{TRS-row}}}$. The latter would require generating SU(2) from $-S$, which combined with the improper-rotation $R \leftarrow -R$ flip inside `get_spinor_rotations` gives a wrong sign.
- The phase flips sign: $-i \to +i$.

The implementation exploits the fact that `sym_mats_k[sym_idx]` already encodes the $-S$ sign for TRS rows (since `sym_mats_k = [Smk, -Smk]`), so

```python
rotated = (S_full @ g_bar.T).T           # (-S)·G_bar for TRS row
phase   = np.exp(-1j * (rotated @ tau))  # = exp(+i (S G_bar)·τ) for TRS automatically
```

needs no separate branch. The spinor branch is:

```python
cnk = np.conj(cnk)                          # apply K first
if phase is not None:
    cnk = cnk * phase[None, None, :]        # then phase (post-conj so sign isn't flipped)
U_eff = _I_SIGMA_Y @ np.conj(U_spinor[s_spatial])
cnk   = einsum('jk,nkl->njl', U_eff, cnk)
```

**The conj order matters**: apply `conj(cnk)` first, then multiply by `phase`. Applying the phase first and then conj inverts the phase sign. The comment in the source spells this out at `symmetry_maps.py:396-400`.

> **Note (2026-05-14, PR3).** Before PR3, `U_spinor[ntran:]` was computed by Markley's quaternion algorithm on `-R_spatial`, which mis-signed half the TRS spinors. PR3 fixed it by (a) restricting `self.U_spinor` to length `ntran` (the TRS-row spinor is constructed inside `unfold_psi` via $i\sigma_y \cdot \overline{U_s}$, making the bug unreachable) and (b) carrying the $\pm S$ sign on `sym_mats_k` instead of on the spinor. The sym table's structural symmetry means there is now exactly one place where TRS algebra lives: this function.

### 3.3 Bispinor Caveat and G-Axis Bookkeeping

For non-SOC (`ns = 1`, `nspinor = 1`) the spinor rotation is a 1×1 identity and the TRS branch reduces to plain conjugation. The bispinor branch (`ns = 2`) is the load-bearing one; CrI3 6×6 SOC, MoS2 3×3 SOC, and Si 4×4×4 SOC all exercise it.

`unfold_psi` returns the rotated coefficients indexed by the **IBZ** G-axis: `cnk_full[b, σ, g]` corresponds to the rotated G-vector $S \cdot g_{\bar k}[g]$. The caller (`WfnLoader.gvecs`, `src/file_io/wfn_loader.py`) rebuilds the full-k G-list including the umklapp $\mathbf{k}_{g_0}$. This separation keeps the helper pure and avoids the recurring "re-sorted G list but forgot to re-sort the phase" bug class.

### 3.4 What Lives Where

After PR5 the public unfold helpers (`get_gvecs_kfull`, `get_cnk_fullzone[_batch]`) moved into `file_io.wfn_loader.WfnLoader`. Both the eager and phdf5 backends call `unfold_psi` for the spinor + phase, and each handles the G-rebuild + umklapp internally. `SymMaps` retains the sym table itself (`sym_matrices`, `sym_mats_k`, `translations`, `R_cart`, `U_spinor`) and the IBZ k/q maps.

---

## 4. Centroid Permutation and Lattice-Wrap Table

`src/centroid/orbit_syms.py:229-437`. Given a centroid set $\{\mathbf x_\mu\}_{\mu=1}^{n_\text{rmu}}$ on the FFT grid and the BGW sym data, returns:

```python
sym_perm[s, μ]    = α                            # (n_sym, n_rmu) int32 or (2 n_sym, ...) if extend_trs
L_table[s, μ, :]  = L ∈ ℤ³                       # (n_sym, n_rmu, 3) int8 (or doubled)
```

such that under the BGW inverse op acting on real-space points:

$$\boxed{S_s \cdot (\mathbf x_\mu - \boldsymbol{\tau}_s) \;=\; \mathbf x_{\alpha_s(\mu)} + \mathbf L_{s, \mu}, \qquad \mathbf L \in \mathbb Z^3.}$$

In words: under the BGW inverse op $g_s^{-1}$ acting as $S \cdot (\mathbf r - \boldsymbol\tau)$, centroid $\mu$ maps to **the source** centroid $\alpha_s(\mu)$, plus an integer real-space lattice wrap $\mathbf L_{s, \mu}$. This is the **source-direction** (or **inverse-source**) decomposition.

### 4.1 Why the Source Direction, Not the Forward Direction

Both directions are valid: define $\beta_s(\mu) = \alpha_s^{-1}(\mu)$ for the forward decomposition $S^{-1} \mathbf x_\mu + \boldsymbol\tau = \mathbf x_{\beta(\mu)} + \mathbf L'_{s,\mu}$. The two are related by an argsort. The source direction is chosen because the user-spec V_q unfold formula (§5) reads

$$V_\text{full}[q, \mu', \nu'] = e^{2\pi i q_\text{irr} \cdot (L_{\mu'} - L_{\nu'})} \cdot V_\text{ibz}[\text{parent}(q),\, \alpha(\mu'),\, \alpha(\nu')]$$

directly under this convention, with no `argsort` inside the JIT. For order-2 (involutive) ops the two directions coincide; for order-3 (CrI3 C3/S6) they differ. Using the forward direction silently produced a ~4 eV gap on hex systems before commit `0735c2a` swapped to the source direction — the prior code used `inv_perm = argsort(sym_perm)`, which is a no-op for involutive groups (MoS2 σ_h, Si cubic involutions) but wrong for order-3.

For symmorphic systems ($\tau = 0$) the inverse and forward directions produce the same orbit set because the group is closed under inversion. For non-symmorphic Si Fd-3m, only the BGW-convention inverse direction closes the orbit.

### 4.2 The Implementation

Inside `compute_centroid_sym_perm` at `orbit_syms.py:347-381`:

```python
tau_frac    = translations / (2π)                                       # (n_sym, 3)
r_frac      = r_mu_fft_idx / fft_grid                                   # (n_rmu, 3)
r_shifted   = r_frac[None, :, :] - tau_frac[:, None, :]                 # (n_sym, n_rmu, 3)
# images_raw[s, μ, i] = sum_j S[s,i,j] r_shifted[s,μ,j]
images_raw  = einsum('sij,srj->sri', S.astype(float64), r_shifted)
# Snap to FFT-grid integers BEFORE floor — see §4.3.
images_int  = np.rint(images_raw * fft_grid).astype(np.int64)
L_wrap      = np.floor_divide(images_int, fft_grid).astype(np.int8)
images_int_mod = images_int - L_wrap.astype(np.int64) * fft_grid        # in [0, FFT_grid)
sym_perm[s, μ] = flat_to_mu[radix_encode(images_int_mod[s, μ])]
```

### 4.3 The FFT-Grid Integer-Snap Fix (the Si Fix)

`src/centroid/orbit_syms.py:361-371` is load-bearing for Si Fd-3m. Centroids live at multiples of $1/\text{FFT}_a$; `mtrx` and $\tau$ are commensurate with the FFT grid (BGW guarantee), so `images_raw * fft_grid` is mathematically integer. The naive

```python
L = np.floor(images_raw).astype(int)
```

flips `L` from 0 to −1 whenever the true integer part is 0 but tiny negative fp noise (~$10^{-17}$) hits `np.floor`'s discontinuity at integer values. This injects a spurious $e^{\pm 2\pi i q}$ phase factor into $V_q$ at every glide op of Si.

The fix (commit `6666a41`, `orbit_syms.py:308-318` / `:361-371`) snaps `images_raw * fft_grid` to integers first via `np.rint`, then does integer floor-division. Verified at ISDF noise floor on Si Fd-3m (24³ FFT, non-symmorphic τ): the previous code gave 14/64 q's with relative error ~0.8; after the snap, every q closes. **This is the single change that takes Si from 791 meV residual to 0.002 meV** at the Σ_X gate.

### 4.4 The `extend_trs=True` Augmentation

Under time-reversal $T$, real-space coordinates $\mathbf r_\mu$ are unchanged ($T$ acts on momenta and complex-conjugates $\psi$; $\mathbf r$ is fixed). The centroid permutation for a TRS-augmented op $T \circ \{S | \tau\}$ therefore coincides with the spatial-only permutation.

`compute_centroid_sym_perm(..., extend_trs=True)` returns tables of shape $(2 n_\text{sym}, n_\text{rmu})$ for `sym_perm` and $(2 n_\text{sym}, n_\text{rmu}, 3)$ for `L_table`, with the second half duplicating the first. This is what makes the tables index-compatible with `sym.irr_idx_q / sym.sym_idx_q` whose values range over $[0, 2 n_\text{tran})$.

Without `extend_trs=True`, a downstream gather of `sym_perm[s]` for $s \ge n_\text{tran}$ silently clips to the last spatial row under JAX `mode='promise_in_bounds'`, producing wrong $V_q$ at every TRS-folded q. This was the headline bug that motivated the entire `trs_sym_audit` session.

### 4.5 Failure Modes and Validation

By default (`validate=True`) the helper asserts:

1. Every image lands on a centroid (`sym_perm[s, μ] ≥ 0`). Failures raise `RuntimeError` pointing at the offending $(s, \mu)$.
2. Every row of `sym_perm` is a permutation of $[0, n_\text{rmu})$.

Failure mode 1 is the **orbit-closure failure**: the centroid set is not orbit-closed under the BGW sym group. The caller (`v_q_g_flat.py:_resolve_ibz_q_list`, §5.5) catches this `RuntimeError` and falls back to full-BZ V_q computation. Failure mode 2 indicates $\tau \times \text{FFT\_grid}$ is non-integer or two centroids collide; no fallback.

The int-snap fix (§4.3) absorbs single-op fp drift up to half a grid cell. Larger drift indicates a genuine orbit-closure failure; the `np.rint` will land on a half-grid point and the subsequent lookup fails.

### 4.6 `compute_rgrid_sym_perm` (Full-Grid Variant)

`src/centroid/orbit_syms.py:444-551`. Same construction, but applied to every point of the FFT grid (not just the centroid subset). Returns `sym_perm[s, r_new] = r_old` — a pull-back gather. Used by the ζ-loader for the IBZ → full-BZ unfold on the r-axis (when ζ is stored only at IBZ q's and the consumer needs the full BZ). Requires $\tau \times \text{FFT\_grid}$ to be integer; the validator refuses loudly otherwise.

---

## 5. V_q Unfold (`unfold_v_q`)

`src/common/symmetry_maps.py:110-299`. Expands the IBZ V_q tensor to the full BZ. JAX-jit'd, sharded `P(None, 'x', 'y')` on the centroid axes.

### 5.1 The Formula

For full-BZ q with IBZ parent $\bar q = q_\text{irr\_frac}[i(q)]$ and sym index $s = \text{sym\_idx}_q$:

$$V_\text{full}[q, \mu', \nu'] \;=\; e^{2\pi i\, \bar q \cdot (\mathbf L_{s,\mu'} - \mathbf L_{s,\nu'})} \;\cdot\; V_\text{ibz}\big[i(q),\, \alpha_s(\mu'),\, \alpha_s(\nu')\big].$$

For TRS-augmented rows ($s \ge n_\text{tran}$):

$$V_\text{full}[q, \mu', \nu'] \;=\; \overline{V_\text{full}^{(\text{spatial})}[q, \mu', \nu']}.$$

The conj on the result captures both the $q \to -q$ sign flip in the phase AND the index swap from V_q Hermiticity in a single operation.

### 5.2 Derivation Sketch

The ISDF identity at IBZ $(k, q)$ is $\rho_{k,q}(\mathbf r) = \sum_\mu \zeta_{q,\mu}(\mathbf r) \cdot \rho_{k,q}(\mathbf r_\mu)$. Apply the BGW forward sym $g = \{S|\tau\}$:

$$\rho_{Sk, Sq}(\mathbf r) = \rho_{k,q}(g^{-1}\mathbf r) = \rho_{k,q}(S \cdot (\mathbf r - \boldsymbol{\tau})).$$

On a centroid, the argument is $S(\mathbf r_\mu - \boldsymbol\tau) = \mathbf r_{\alpha(\mu)} + \mathbf L_\mu$ (definition of $\alpha, \mathbf L$, §4). Use the Bloch identity for the pair density, $\rho_{k,q}(\mathbf r' + \mathbf L) = e^{2\pi i q \cdot \mathbf L} \rho_{k,q}(\mathbf r')$, and match coefficients:

$$\zeta_{Sq, \mu}(\mathbf r) = e^{2\pi i q \cdot \mathbf L_\mu} \cdot \zeta_{q, \alpha(\mu)}(S \cdot (\mathbf r - \boldsymbol\tau)).$$

Then $V_{Sq}[\mu, \nu] = \iint \overline{\zeta_{Sq,\mu}(\mathbf r)} v(\mathbf r - \mathbf r') \zeta_{Sq,\nu}(\mathbf r') d\mathbf r d\mathbf r'$ with $v$ rotation-invariant and $\tilde{\mathbf r} = S(\mathbf r - \boldsymbol\tau)$:

$$V_{Sq}[\mu, \nu] = e^{-2\pi i q \cdot \mathbf L_\mu} e^{+2\pi i q \cdot \mathbf L_\nu} \cdot V_q[\alpha(\mu), \alpha(\nu)] = e^{2\pi i \bar q \cdot (\mathbf L_\nu - \mathbf L_\mu)} V_q[\alpha(\mu), \alpha(\nu)].$$

> **Note (2026-05-14).** The derivation gives the sign as $(\mathbf L_\nu - \mathbf L_\mu)$. The user-spec writes $(\mathbf L_\mu - \mathbf L_\nu)$. Both are sign-equivalent residuals on the V_q dump because V is Hermitian and the indices μ, ν are bookkeeping — `phase[:, None] * V * conj(phase)[None, :]` and `conj(phase)[:, None] * V * phase[None, :]` both close at ISDF floor. The implementation follows the user-spec sign with `phase[:, None] * V * conj(phase)[None, :]` (`symmetry_maps.py:290`).

### 5.3 The TRS Half

Time-reversal in the bilinear-in-ζ V_q satisfies

$$V_\text{full}[Tq, \mu, \nu] = \overline{V_q[\mu, \nu]} = V_q[\nu, \mu]$$

(last equality by V_q Hermiticity). Implementation: apply the spatial-formula phase + double-permute first, then `conj` if `sym_idx ≥ n_sym_spatial`. The `sym_perm` rows for $s \in [n_\text{tran}, 2 n_\text{tran})$ are identical to $[:n_\text{tran}]$ (TRS keeps r fixed, §4.4), and so are `L_table` rows.

### 5.4 The JIT Body

`src/common/symmetry_maps.py:267-299`. Key shapes (sharded on the `('x', 'y')` mesh):

```text
V_q_ibz       : (n_q_ibz, n_rmu, n_rmu) c128, sharded P(None, 'x', 'y')
irr_idx       : (n_q_full,) int32          — sym.irr_idx_q
sym_idx       : (n_q_full,) int32          — sym.sym_idx_q
sym_perm      : (2·ntran, n_rmu) int32     — α, extend_trs=True
L_table       : (2·ntran, n_rmu, 3) int8   — L, extend_trs=True
q_irr_frac    : (n_q_ibz, 3) float64       — IBZ q in fractional reciprocal coords
n_sym_spatial : int                        — ntran (drives the TRS-row mask)
V_q_full      : (n_q_full, n_rmu, n_rmu) c128, sharded P(None, 'x', 'y')
```

Inside the JIT (`out_shardings=NamedSharding(mesh, P(None, 'x', 'y'))`):

```python
perm_q     = sym_perm[sym_idx]                           # (n_q_full, n_rmu)
V_at_irr   = V_q_ibz[irr_idx]                            # (n_q_full, μ, μ)
V_perm_mu  = take_along_axis(V_at_irr,  perm_q[:, :, None], axis=1, mode='promise_in_bounds')
V_full     = take_along_axis(V_perm_mu, perm_q[:, None, :], axis=2, mode='promise_in_bounds')

L_per_q    = L_table[sym_idx]                            # (n_q_full, n_rmu, 3)
q_per_q    = q_irr_frac[irr_idx]                         # (n_q_full, 3)
qL         = einsum('qi,qmi->qm', q_per_q, L_per_q)      # (n_q_full, n_rmu)
phase      = exp(2j π qL.astype(c128))
V_full     = phase[:, :, None] * V_full * conj(phase)[:, None, :]

V_full     = where(trs_mask[:, None, None], conj(V_full), V_full)
```

Three implementation details:

1. **`mode='promise_in_bounds'`** avoids an XLA HLO-verifier dtype mismatch (s32 vs s64) inside `shard_map + x64`. The OOB-fill branch silently mints s64; the promise keeps the gather in s32 throughout. Bounds are guaranteed by construction — every entry of `perm_q` is a valid centroid index, which is what `compute_centroid_sym_perm`'s validation establishes — so `promise_in_bounds` is correct, not a hack.

2. **Identity short-circuit** (`symmetry_maps.py:194-201`): when `ntran=1` (nosym), `irr_idx` is identity and `sym_idx` is all zeros. The `take_along_axis` codegens an s64-broadcast the verifier rejects on 2×2 meshes. The code detects this case host-side and returns `V_q_ibz` unchanged.

3. **Centroid padding** (`symmetry_maps.py:228-261`): when `n_rmu_logical` is not divisible by the mesh's y-axis, `sym_perm` is identity-padded `[n_rmu, n_rmu+1, …]` and `L_table` is zero-padded. Padded slots carry no umklapp wrap.

### 5.5 Caller Wiring (the IBZ Cascade)

`gw/v_q_g_flat.py:_resolve_ibz_q_list` (`v_q_g_flat.py:154-223`) is the single source of truth for `unfold_v_q` inputs:

```python
sym_perm, L_table = compute_centroid_sym_perm(
    centroid_indices,                            # (n_rmu, 3) int FFT idx
    sym_matrices=sym.sym_matrices[:n_tran],      # (n_tran, 3, 3) int = mtrx
    translations=sym.translations[:n_tran],      # (n_tran, 3) float — RAW 2π·τ_frac
    fft_grid=fft_grid,
    extend_trs=True,
)
# On RuntimeError: use_ibz=False, fall back to full-BZ q iteration.

q_irr_kgrid_int   = sym.q_irr_kgrid_int          # (n_q_ibz, 3) int
q_irr_frac        = bgw_wrap(q_irr_kgrid_int) / kgrid    # (n_q_ibz, 3) float in (-1/2, 1/2]
q_full_to_irr_idx = sym.irr_idx_q                # (n_q_full,)
q_full_to_irr_sym = sym.sym_idx_q                # (n_q_full,)
```

When `compute_centroid_sym_perm` succeeds, the cascade activates: $V_q$ is computed at only the IBZ q's (count $n_q^\text{ibz} \approx n_q^\text{full} / n_\text{tran}$) and then unfolded. When it fails (orbit-closure `RuntimeError`), `use_ibz=False` and $V_q$ is computed at every full-BZ q directly. The override `LORRAX_FORCE_FULL_BZ=1` forces the fallback for debugging.

The IBZ cascade landed on 2026-05-11 (`project_lorrax_ibz_cascade.md`): generate centroids with orbit closure on, and V_q work collapses from $n_q^\text{full}$ to $n_q^\text{ibz}$ (~$n_\text{tran}$× reduction). CrI3 6×6: 36 → 7. MoS2 3×3: 9 → 2. Si 4×4×4: 64 → 8.

---

## 6. Symmorphic Failure Modes — Why CrI3/MoS2 Alone Is Not Enough

For symmorphic groups ($\tau = 0$ for every op), most LORRAX symmetry bugs are invisible. The shared property of these bugs is that they only fail when **(a)** $\tau \neq 0$ AND **(b)** the group is non-abelian or has non-involutive elements AND **(c)** some centroid wraps non-trivially under the sym action.

| Bug | Symptom on CrI3 / MoS2 | Symptom on Si Fd-3m |
|---|---|---|
| Forward vs source perm direction (`sym_perm = α` vs $\alpha^{-1}$) | Identical orbit, zero residual | Different orbit for order-3 glides: ~eV-scale |
| τ-sign in `compute_centroid_sym_perm` | Vanishes ($\tau=0$) | Wrong $L$-wrap, wrong umklapp phase |
| `np.floor` off-by-one on tiny negative fp noise | Vanishes ($L=0$ either way) | Spurious $e^{\pm i\pi/2}$ phase on 14/64 q's |
| TRS-row spinor via `get_spinor_rotations(-S_spatial)` | Cancels in Σ_X for involutive groups | Wrong $U_\text{spinor}$ at every TRS-folded k: ~100 eV |
| `extend_trs=False` with TRS-augmented `sym_idx` | Silent OOB-clip; on MoS2 σ_h (order 2) accidentally OK | Wrong V_q at every TRS-folded q |
| Wrong r-action direction ($\mathrm{mtrx} \cdot r + \tau$ vs $\mathrm{mtrx}^{-1} r + \tau$) | Same orbit for any abelian group | 48/96 atom mappings fail |
| Wrong `argsort` of `sym_perm` (the π vs π⁻¹ direction in V_q unfold) | Involutive op ⇒ π = π⁻¹, no effect | Wrong V_q on every order-3 q |
| Missing L-phase in `unfold_v_q` | $\tau = 0$ and $L = 0$ ⇒ phase is unity | Wrong V_q at every glide-folded q |

The takeaway: **CrI3 and MoS2 are not sufficient regression coverage** for symmetry conventions. They happily pass with multiple compensating bugs. Si Fd-3m is the load-bearing regression for non-symmorphic ops; CrI3 30 Ry is the load-bearing regression for order-3 spatial sym (where $\alpha \neq \alpha^{-1}$); MoS2 3×3 is the test for the TRS-augmentation plumbing.

---

## 7. Sym-vs-nosym Validation Recipe

Pair a "sym ON" production run against a "sym OFF" reference, and require the Σ_X residual to be below 1 meV at every $(k, n)$.

### 7.1 Generate Two QE/BGW Runs in Parallel

For system `MAT` with k-grid $(n_{kx}, n_{ky}, n_{kz})$ and energy cutoff $E_c$:

- `runs/MAT/NN_MAT_kgrid_sym_vs_nosym_DATE/run_sym/` — QE with default sym (no `nosym=.true.` in `scf.in` / `nscf.in`).
- `runs/MAT/NN_MAT_kgrid_sym_vs_nosym_DATE/run_nosym/` — QE with `nosym=.true.` AND `noinv=.true.` in both `scf.in` and `nscf.in`.

The `nosym` WFN.h5 must be **regenerated** with `pw.x` symmetry off; toggling LORRAX's flag without regenerating the WFN won't work because `wfn.kpoints` will still be the reduced set.

### 7.2 Build Matched LORRAX Inputs

Both runs must share:

| Parameter | Value | Why |
|---|---|---|
| `nval`, `ncond` | same | Σ_X depends linearly on occupied subspace |
| `bare_coulomb_cutoff` | **explicit, same value** | LORRAX default = $4 \cdot \text{ecutwfc}$ ≠ BGW default = `ecutwfc`. ALWAYS specify. |
| `ecutwfc` (LORRAX) | same as QE | trivially required |
| `memory_per_device_gb` | same | reproducibility; doesn't affect physics |
| `n_rmu` and the centroid file | same | otherwise residuals get blamed on ISDF rank |

The `bare_coulomb_cutoff` default mismatch is logged in `project_bare_coulomb_cutoff_default.md`. Same applies to BGW-vs-LORRAX comparisons.

### 7.3 Generate Matched Centroids

The two runs need centroid sets that are *physically the same* set of points, not two independent kmeans runs (kmeans is stochastic and a random seed difference would mask the sym test). Two recipes:

- Generate centroids in `nosym` (no orbit-closure requirement), then use the same file in `sym`.
- Generate orbit-closed centroids in `sym` and use the same file in `nosym`.

If the centroid orbit-closure check fails on the `sym` side, regenerate with `kmeans_cli --orbit-aware`, which threads the BGW sym group into the Lloyd iterations. Without `--orbit-aware`, raw kmeans output is almost never orbit-closed past ~10 centroids.

A subtle failure mode: kmeans may produce a set whose orbits are closed under most ops but missing one or two members. This can happen when the centroid budget is too small relative to the largest representative orbit. CrI3 30 Ry with 300 centroids hit this (48/300 missed C3-images); regenerating with a larger budget or `--ensure-closure` fixed it.

### 7.4 Run and Diff

```bash
lxrun --pool=A python -m gw.gw_jax --config cohsex.in   # in run_sym/
lxrun --pool=B python -m gw.gw_jax --config cohsex.in   # in run_nosym/
```

The two LORRAX runs use the same source tree; the only difference is which WFN.h5 each reads. Diff Σ_X with `skills/compare/SKILL.md`'s parser:

$$\max_{k, n} \big|\Sigma^X_\text{sym}[k, n] - \Sigma^X_\text{nosym}[k, n]\big| < 1 \text{ meV}.$$

### 7.5 Failure-Mode Triage

| Magnitude | Likely root cause |
|---|---|
| $\gtrsim 1$ eV on a non-symmorphic system | L-table / `sym_perm` direction. Run `validate_atomic_symmetries` first to rule out τ-scaling. |
| $\gtrsim 100$ meV on a symmorphic system | Spinor / `U_spinor` issue (τ-related bugs all collapse at $\tau = 0$). |
| $\sim 10$ meV on either | Probably a centroid mismatch, not a sym bug. Diff the centroid files. |

---

## 8. Verified Gates (2026-05-15)

End-to-end Σ_X sym-vs-nosym comparisons:

| System | k-grid | Cutoff | $\max|\Delta\Sigma^X|$ | Notes |
|---|---|---|---|---|
| MoS2 | 3×3×1 SOC | 30 Ry | **0.090 meV** | Symmorphic, σ_h involutive (canonical baseline; tests TRS-augmentation plumbing) |
| CrI3 | 6×6×1 SOC | 30 Ry | **0.076 meV** | Symmorphic P-3, order-3 ops (catches `argsort`-direction bugs) |
| Si | 4×4×4 SOC | 30 Ry | **0.002 meV** | **Non-symmorphic Fd-3m** (canonical regression for τ-aware code) |

V_q-only unit test (synthetic IBZ tensor on the CrI3 6×6 30 Ry dump, 36 q's): relative error $8.67 \times 10^{-6}$, at the ISDF noise floor (`reports/trs_sym_audit_2026-05-14/verify_umklapp_user_math.py`).

Si centroid orbit closure under BGW convention: 432 centroids, 96 sym ops, all permutations close, $|\mathbf L|_\max = 3$.

The Si result is the load-bearing one. Until 2026-05-13 Si carried a 160 eV gap; the resolution required, in order: (1) `compute_centroid_sym_perm` BGW-convention fix to use $r' = S(r - \tau)$ (commit `0b0fc37`); (2) the L-phase factor in `unfold_v_q` (commit `0735c2a`); (3) the FFT-grid integer-snap in §4.3 (commit `6666a41`); (4) `extend_trs=True` rewiring; (5) PR3's `unfold_psi` spinor/TRS unification. Any one undone takes Si back to multi-eV or multi-meV failure. CrI3 30 Ry and MoS2 are immune to (1)–(4) because $\tau = 0$ collapses the algebra.

---

## 9. File / Function Index

### LORRAX (`sources/lorrax_B/`)

| Function / Class | Path | Role |
|---|---|---|
| `MfHeader` | `src/file_io/mf_header.py:30-83` | WFN.h5 header dataclass; reads `mtrx`, `tnp`, `ntran`. |
| `_read_group` | `src/file_io/mf_header.py:89-132` | h5py reader populating `MfHeader`. |
| `SymMaps` | `src/common/symmetry_maps.py:416-1143` | Top-level sym table holder. Builds k/q-IBZ maps, R_cart, U_spinor. |
| `SymMaps.sym_matrices` | `src/common/symmetry_maps.py:529` | Spatial-only `mtrx[:ntran]`. The BGW stored matrix. |
| `SymMaps.sym_mats_k` | `src/common/symmetry_maps.py:530-543` | TRS-augmented `mtrx.T` (length $2 \cdot n_\text{tran}$). The k-action matrix for numpy-row-vector consumption (§1.5). |
| `SymMaps.translations` | `src/common/symmetry_maps.py:536-537` | Raw `tnp[:ntran]` (= $2\pi \tau_\text{frac}$). |
| `SymMaps.U_spinor` | `src/common/symmetry_maps.py:593` | Spatial-only SU(2) spinor rotations (length $n_\text{tran}$). TRS rows built inside `unfold_psi`. |
| `SymMaps.R_cart` | `src/common/symmetry_maps.py:592` | Cartesian-frame rotations (length $2 \cdot n_\text{tran}$) for Markley's quaternion input. |
| `SymMaps.irr_idx_q, sym_idx_q` | `src/common/symmetry_maps.py:625-629` | Full-BZ q → IBZ parent + sym row. |
| `SymMaps.q_irr_kgrid_int` | `src/common/symmetry_maps.py:630` | IBZ q list as kgrid integers. |
| `SymMaps.validate_atomic_symmetries` | `src/common/symmetry_maps.py:780-811` | Checks `inv(mtrx) · atom + τ` closes. Smoke test for convention. |
| `SymMaps.syms_crystal_to_cartesian` | `src/common/symmetry_maps.py:844-906` | $R_\text{cart} = A_T \cdot \mathrm{mtrx} \cdot A_T^{-1}$. |
| `find_irreducible_bz_points` | `src/common/symmetry_maps.py:18-107` | Full-BZ → IBZ index + sym row builder (q-side and k-side). |
| `unfold_psi` | `src/common/symmetry_maps.py:306-413` | ψ at full-BZ k from IBZ ψ. Spinor + phase + TRS in one place. |
| `unfold_v_q` | `src/common/symmetry_maps.py:110-299` | V_q at full BZ from IBZ. jit'd, sharded `P(None,'x','y')`. |
| `_I_SIGMA_Y` | `src/common/symmetry_maps.py:303` | $i\sigma_y$, the SOC time-reversal spinor factor. |
| `get_spinor_rotations` | `src/common/symmetry_maps.py:908-992` | Markley quaternion SU(2) from `R_cart`. |
| `compute_centroid_sym_perm` | `src/centroid/orbit_syms.py:229-437` | $(\alpha, L)$ from inverse-source decomposition. `extend_trs=True` doubles for TRS. |
| `compute_rgrid_sym_perm` | `src/centroid/orbit_syms.py:444-551` | Full-FFT-grid pull-back permutation for ζ r-axis unfold. |
| `orbit_images` | `src/centroid/orbit_syms.py:73-92` | Row-form BGW action `r' = r · Rinv.T + τ` on a rep set. |
| `unfold_orbit_unique_with_id` | `src/centroid/orbit_syms.py:170-216` | Orbit-aware kmeans backend; emits orbit IDs. |
| `build_real_space_syms` | `src/centroid/orbit_syms.py:33-59` | Pre-divides `wfn.translations / 2π` and gathers $R$, $R^{-1}$. |
| `_resolve_ibz_q_list` | `src/gw/v_q_g_flat.py:154-223` | Builds `sym_perm`, `L_table`, `q_irr_frac` for V_q driver; falls back to full-BZ on `RuntimeError`. |

### External (BGW)

| Reference | Path | Relevance |
|---|---|---|
| WFN.h5 spec | `docs/docs_bgw/wfn.h5.spec:172-185` | `mtrx`, `tnp` layout. |
| pw2bgw `tnp` writer | `sources/BerkeleyGW/MeanField/ESPRESSO/version-7.2/pw2bgw_qe7.2_with_spinor_mag.f90:826-850` | `tnp = (S⁻¹ · ft) · 2π`. |
| BGW Σ reader | `sources/BerkeleyGW/Sigma/sympert_utils.f90:748-755` | Reads $\tau$ as `tnp/(2π)`; transposes+inverts `mtrx` for r-space. |
| symmetries.f90 | `sources/BerkeleyGW/Common/symmetries.f90:189` | `mtrx` stored as `invert(mtrx_inv)` — i.e., inverse of QE's real-space rotation. |

---

## 10. Quick Reference

### 10.1 BGW r-action

```text
r' = mtrx⁻¹ · r + τ                # column form, τ = wfn.translations / (2π)
```

### 10.2 BGW group composition law

```text
S_c = S_a · S_b   ⇒   τ_c = S_a⁻¹ · τ_b + τ_a   (mod 1)
```

### 10.3 BGW reciprocal forward action

```text
q' = mtrx · q                      # column form; BGW's matmul(syms%mtrx, k)
                                   # LORRAX numpy: sym_mats_k = mtrx.T applied as row-vector left-multiply
```

### 10.4 ψ unfold (spatial)

```text
ψ_full(G_rot) = exp(-i (S·G_kbar)·τ_raw) · U_spinor(S) · ψ_kbar(G_kbar)
where  G_rot = S·G_kbar + kg_0,  τ_raw = wfn.translations[s]   (= 2π·τ_frac)
```

### 10.5 ψ unfold (TRS-augmented)

```text
ψ_full(G_rot) = (i σ_y · conj(U_spinor(S))) · conj(ψ_kbar(G_kbar)) · exp(+i (S·G_kbar)·τ_raw)
```

### 10.6 Centroid permutation + L-wrap

```text
S · (x_μ − τ_frac) = x_{α(μ)} + L_μ,   L_μ ∈ ℤ³
sym_perm[s, μ] = α(μ),    L_table[s, μ] = L_μ
```

### 10.7 V_q unfold (spatial)

```text
V_full[q, μ', ν'] = exp(2π i q_irr · (L_{s,μ'} − L_{s,ν'})) · V_ibz[i(q), α_s(μ'), α_s(ν')]
```

### 10.8 V_q unfold (TRS row)

```text
V_full[T·q, μ, ν] = conj(V_full^{spatial}[q, μ, ν])
                  = V_ibz[i(q), α(ν), α(μ)]    (by V_q Hermiticity)
```

---

## 11. Reference Tests and Dumps

| Resource | Path | What it verifies |
|---|---|---|
| V_q production unfold test | `reports/trs_sym_audit_2026-05-14/test_production_unfold_v_q.py` | CrI3 6×6 30 Ry V_q dump → all 36 q's at rel $8.7 \times 10^{-6}$ (ISDF floor). Run after any `unfold_v_q` change. |
| User-math V_q verifier | `reports/trs_sym_audit_2026-05-14/verify_umklapp_user_math.py` | Same dump, applies the §5.1 formula directly. Sanity-check the helper against the math. |
| V_q dumps (HDF5) | `reports/trs_sym_audit_2026-05-14/v_q_dumps/{Vq_ibz_sym.h5, Vqmunu_nosym.h5}` | Reference IBZ and full-BZ tensors. |
| Si 4×4×4 sym/nosym | `runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/` | Non-symmorphic Σ_X gate (0.002 meV). |
| CrI3 6×6 sym/nosym | `runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/` | Symmorphic order-3 Σ_X gate (0.076 meV). |
| MoS2 3×3 sym/nosym | `runs/MoS2/00_mos2_3x3_cohsex/` (variants) | Involutive σ_h Σ_X gate (0.090 meV). |

When in doubt about whether a code change broke sym, run all three gates plus the V_q production unit test. Symmorphic gates passing while Si fails identifies a τ-related bug; Si passing while symmorphic fails identifies a spinor or `argsort`-direction bug. Both failing implies a deeper convention error and `validate_atomic_symmetries` is the right first read.

---

## 12. Deferred Work

Items intentionally out of scope as of 2026-05-15 (recorded so future engineers don't rediscover them as bugs):

- **Unified `sym_action` helper.** ψ, ζ, and V_q transform by the same template: phase × spinor × permutation, parametrized by $(S, \tau)$. The `feedback_unified_sym_action.md` pin notes that LORRAX has accumulated parallel "rotate X at q" helpers — `unfold_psi` for ψ, `unfold_v_q` for V_q, plus the implicit ζ-side rotation inside `ZetaLoader`. Phase 2 of the TRS-aware-sym-fix consolidates these through one canonical IBZ table (`SymMaps` + accessors) and one sym-action helper. The `feedback_no_new_api_layers.md` pin constrains: no `SymAction` class wrapper, just delete the parallel routines.
- **Bispinor IBZ.** The bispinor four-density variant (`accumulate_pair_density_open_spin` in `common/isdf_fitting.py`) is wired through the GW driver but not yet exposed via a `cohsex.in` flag. It needs the same IBZ unfold as the spin-traced path; `unfold_v_q` generalizes with the same phase + permutation.
- **V_q rematerialization.** For very large $n_\text{rmu}$ the jit'd `unfold_v_q` allocates $n_q^\text{full} \times n_\text{rmu}^2$ complex128 (several GB on 80 Ry CrI3). A scan-over-q variant could rematerialize one q-slice at a time.
