# Full-Frequency GW Self-Energy via GN-PPM and Minimax-Windowed CTSP in the ISDF Basis

### Purpose

Reference for computing the **correlation self-energy** (\Sigma^c_{kij}(\omega)) using:

1. ISDF compression in real space (collocation points (\mu,\nu))
2. Minimax-windowed CTSP to compute **(W(0))** and **(W(i\omega_p))**
3. GN-PPM extracted **elementwise from (W^c)**
4. Minimax-windowed CTSP to compute (\Sigma^c(\omega)) on a real frequency grid
5. Accumulation **directly in band space** ((k,i,j)) (no storing (\Sigma(\mu,\nu,\omega)))

No analytic continuation.

---

## 1. Conventions

### 1.1 Indices

| Symbol    | Meaning                                                  |
| --------- | -------------------------------------------------------- |
| (v,c)     | valence / conduction band indices                        |
| (k,q,R)   | crystal momentum / transfer / lattice vector             |
| (\mu,\nu) | ISDF collocation indices, size (N_\mu)                   |
| (a,b)     | spinor components                                        |
| (i,j)     | projected band indices (the (\Sigma_{kij}) output basis) |

### 1.2 Energy referencing used by the code

* `enk_full` is vacuum-referenced eigenvalues (E^{\rm vac}_{nk}).
* `occ_full` defines occupied vs unoccupied: `occ_mask = occ_full > 0.5`.

The sigma code defines a single scalar (E_F) (**vacuum reference**) by:

* `fermi_reference="vbm"`: (E_F = \max(E^{\rm vac}_{nk};\text{over occ}))
* `fermi_reference="midgap"`: (E_F = \tfrac12(\text{VBM}+\text{CBM})) if any unoccupied exist, else VBM.

Then it constructs positive-axis energies used in denominators:

[
E_c \equiv \max(E^{\rm vac}*{nk}-E_F,,0),\qquad
h_v \equiv \max(E_F-E^{\rm vac}*{nk},,0),
]
and (\Omega_{q,\mu\nu}\ge 0) from GN-PPM.

**Important:** the input omega grid `omega_values_ry` is interpreted as **already** (\omega_{\rm rel}) (relative to the chosen (E_F)), and the code splits at (\omega_{\rm rel}=0). It does **not** shift `omega_values_ry` by (E_F).

### 1.3 Fourier conventions

All FFTs are `norm='ortho'`.

* (k\to R): `ifftn` on ((nkx,nky,nkz))
* (R\to k): `fftn` on ((nkx,nky,nkz))

---

## 2. Correlation self-energy structure

The code implements the standard “(\Sigma^{(+)}+\Sigma^{(-)})” decomposition (equivalent to SX+COH grouping):

[
\Sigma^c(\omega)=\Sigma^{(+)}(\omega)+\Sigma^{(-)}(\omega),
]
with denominators (using the positive-axis values above):

[
\Sigma^{(-)}:\quad \frac{1}{\omega_{\rm rel}-(E_c+\Omega)},
\qquad
\Sigma^{(+)}:\quad \frac{1}{\omega_{\rm rel}+(h_v+\Omega)}.
]

Crossing behavior depends on (\omega_{\rm rel}):

| branch                                          | (\omega_{\rm rel}>0)     | (\omega_{\rm rel}<0)     |
| ----------------------------------------------- | ------------------------ | ------------------------ |
| (\Sigma^{(-)}): (\omega_{\rm rel}-(E_c+\Omega)) | **can cross**            | sign-definite (negative) |
| (\Sigma^{(+)}): (\omega_{\rm rel}+(h_v+\Omega)) | sign-definite (positive) | **can cross**            |

**Implementation consequence:** the code splits (\omega_{\rm rel}) into (\omega_{\rm rel}\ge 0) and (\omega_{\rm rel}<0), and evaluates the negative half at (|\omega_{\rm rel}|) with a canonical sign convention (see §6.4).

---

## 3. Phase 1: compute (W(0)), (W(i\omega_p)) and extract GN-PPM from (W^c)

### 3.1 What is computed

The function `compute_w0_wiwp_and_ppm_from_minimax(...)` computes:

* (W_q(0)) and (W_q(i\omega_p)) in the ISDF basis (layout ((nkx,nky,nkz,1,\mu,1,\nu))).
* optional finite-size/head correction through the callback `head_correction_fn(V_q, W_q, omega)`.
* then constructs:
  [
  W^c_q(0)=W_q(0)-V_q,\qquad W^c_q(i\omega_p)=W_q(i\omega_p)-V_q
  ]
  where (V_q) is the (possibly head-corrected) Coulomb matrix reshaped to match (W_q).

### 3.2 How (\chi^0(0)) and (\chi^0(i\omega_p)) are computed

1. Build a **single minimax Laplace window** for static screening using:
   `build_static_minimax_window_pair(enk_v, enk_c, target_error, max_nodes)`

2. Evaluate (\chi^0_q(0)) and (\chi^0_q(i\omega_p)) by calling `w_isdf.compute_chi0(...)` twice:

   * once with the base window `w0`
   * once with the modulated window `wiw = w0.with_imag_freq_modulation(omega_p_ry)`

3. Solve the Dyson equation for (W) twice via:
   `w_isdf.solve_w_from_chi_q_jax(V_qmunu, chi_q, meta, mesh_xy)`.

### 3.3 GN-PPM model used in the code: **PPM for (W^c)**

The GN extraction is done by `extract_gn_ppm_parameters_from_Wc(Wc0_q, Wci_q, omega_p, fallback_omega)` and returns elementwise (\Omega) and (B) such that:

[
W^c_{q,\mu\nu}(\omega)\approx \frac{2,B_{q,\mu\nu},\Omega_{q,\mu\nu}}{\omega^2-\Omega_{q,\mu\nu}^2}.
]

Static identity (what the code checks):
[
W^c(0)=-\frac{2B}{\Omega}.
]

Time-domain form used later:
[
W^c_{q,\mu\nu}(t) = B_{q,\mu\nu},e^{-i,\Omega_{q,\mu\nu},t}.
]

### 3.4 Valid/invalid GN modes

The GN extractor produces:

* `valid_mask_mu_nu` and `unfulfilled_fraction` (reported as “unfulfilled=…%”).

Downstream, sigma uses:

* `B_mask = Omega_abs > 1e-14`
* and if `invalid_mode="static_limit"` then it masks out invalid GN elements: `B_mask &= valid_mask`.

---

## 4. Phase 2: (\Sigma^c(\omega)) on a real-frequency grid (minimax CTSP)

### 4.1 Outputs

`compute_sigma_c_ppm_omega_grid(...)` returns:

* `sigma_c_kij(omega)` (either in memory or streamed to H5)
* optional split contributions: `sigma_c_plus_kij`, `sigma_c_minus_kij`
* optional static correction for invalid GN modes: `sigma_c_invalid_static_kij`

### 4.2 Accumulation mode (what code actually does)

The code **always** accumulates in band space ((k,i,j)) before applying the (\omega)-kernel:

* The old “(\Sigma_{\mu\nu}(\omega)) to disk” mode is disabled; if you request `sigma_munu_h5_path`, it prints a warning and ignores it.

Modes:

* `omega_accumulation="kij"`: all (\omega) in one pass in memory
* `omega_accumulation="kij_stream"`: write `sigma_c_kij_ry[omega,k,i,j]` to HDF5 in ω-chunks
* `omega_accumulation="auto"` selects between them

### 4.3 “Two-channel” accumulation at each time node

At each time node (t), the code computes (\Sigma(t)) in (\mu\nu) and projects to band space **as two real channels**:

* `sigma_tau_kij_re = K[ Re(Σ_tau) ]`
* `sigma_tau_kij_im = K[ Im(Σ_tau) ]`

Then for each (\omega) it forms a complex scalar coefficient (c(\omega,t)) and mixes the channels to produce either:

* **real-projection windows**: (\Re(c,\Sigma))
* **imag-projection windows** (HGL core): (\Im(c,\Sigma))

This is done without storing (\Sigma(\omega)) in (\mu\nu).

---

### 4.4 `_SigmaWindow` dataclass

Each window is represented as:

```python
@dataclass(frozen=True)
class _SigmaWindow:
    name: str
    t_nodes: np.ndarray     # complex (Laplace: -iτ, core: real)
    alpha: np.ndarray       # real weights
    mask_A: np.ndarray      # boolean mask over axis A entries
    mask_B: np.ndarray      # boolean mask over axis B entries
    E_ref_A: float          # min axis-A energy in this window
    E_ref_B: float          # min axis-B energy in this window
    omega_sign: int
    project: str            # "real" or "imag"
    prefactor: float
    # plus debug metadata: x_min/x_max or crossing_A, T, z_edge
```

Laplace (sign-definite) windows have `t_nodes = -iτ` (decaying) and `project="real"`.
Crossing (HGL core) windows have real `t_nodes = τ/ξ` (oscillatory) and `project="imag"`.

---

## 5. Sigma windowing and quadrature (exactly as in `ppm_sigma.py`)

### 5.1 Kernel sign convention used by the sigma code

In sigma, axis-A energies are either (E_c) or (h_v) and axis-B energies are (\Omega).

* `kernel_sign = +1` means the **minus-kernel** class (can require 3-window crossing treatment when (\omega_{\max}>0)).
* `kernel_sign = -1` means the **plus-kernel** class (sign-definite; always single Laplace window).

The window builder is called with `omega_nonneg_ry` (always (\ge 0) inside each half-batch).

### 5.2 Single Laplace window (sign-definite)

Built by `_build_single_sigma_window(...)`:

Given masked axis values (A\in E_A), (B\in E_B(=\Omega)):

[
S_{\min}=\min(A)+\min(B),\qquad S_{\max}=\max(A)+\max(B),
]
[
x_{\min}=\max(S_{\min},10^{-12}),
]
[
x_{\max}=\begin{cases}
\max(S_{\max}+\omega_{\max}, x_{\min}(1+10^{-9})) & \text{if } \text{kernel_sign}=-1\
\max(S_{\max}, x_{\min}(1+10^{-9})) & \text{if } \text{kernel_sign}=+1
\end{cases}
]

Then:

* `q = solve_laplace_minimax_interval(x_min, x_max, target_error, max_nodes)`
* `t_nodes = -i * q.tau`
* `alpha = q.alpha`

Window fields:

* `E_ref_A = min(A_vals)`, `E_ref_B = min(B_vals)`
* `omega_sign = kernel_sign`
* `project = "real"`

Prefactor (exactly as code):

* it treats the “docs prefactor” as:

  * `docs_prefactor = -1` if `kernel_sign==+1`, else `+1`
* but `get_sigma_mu_nu_fn` already includes a global (-1), so it stores:
  [
  \texttt{prefactor}=-\texttt{docs_prefactor}
  ]
  i.e.

  * `kernel_sign=+1`: prefactor (=+1)
  * `kernel_sign=-1`: prefactor (=-1)

### 5.3 Three-window scheme (used only for `kernel_sign=+1` and (\omega_{\max}>0))

Built by `_build_three_sigma_windows(...)`:

Parameters:
[
\xi=\texttt{regularization_width_ry},\quad z_{\rm edge}=\texttt{edge_factor}\cdot\xi,\quad T=\omega_{\max}+z_{\rm edge}.
]

Masks (note: **b_slab uses all A**, so the ((A>T,B>T)) corner is included in b_slab):

* core: (A\le T) and (B\le T)
* a_stripe: (A>T) and (B\le T)
* b_slab: (A) = full, and (B>T)

For each window, define:
[
S_{\min}=\min(A)+\min(B),\qquad S_{\max}=\max(A)+\max(B).
]

**core**:

* (A_{\rm core}=\max(2T/\xi,10^{-8}))
* `q_cross = solve_phase_minimax_bandwidth(A_core, target_error, crossing_max_nodes, eps_q=crossing_eps_q, target_kind="hgl")`
* `t_nodes = q_cross.tau / xi` (real)
* `alpha = q_cross.alpha / xi` (real)
* `project="imag"`
* `omega_sign=+1`
* docs_prefactor = +1 → stored prefactor = (-1)

**a_stripe and b_slab**:
[
x_{\min}=\max(S_{\min}-(T-z_{\rm edge}),\ z_{\rm edge},\ 10^{-12})
=\max(S_{\min}-\omega_{\max},\ z_{\rm edge},\ 10^{-12})
]
[
x_{\max}=\max(S_{\max}, x_{\min}(1+10^{-9})).
]

* `q = solve_laplace_minimax_interval(x_min, x_max, ...)`
* `t_nodes=-i q.tau`
* `alpha=q.alpha`
* `project="real"`
* `omega_sign=+1`
* docs_prefactor = −1 → stored prefactor = (+1)

---

## 6. Exact sigma evaluation loop (per branch)

At a given `_SigmaWindow win`, for each time node (t) and weight (\alpha):

1. Build band weights for axis A:
   [
   \text{phase}*A = e^{-i (E_A - E^{\rm ref}*A),t},
   ]
   masked by `win.mask_A`, and used to construct a diagonal band matrix (G*{knm}) via:
   [
   G*{knm} = \delta_{nm},\text{phase}_A(k,n).
   ]

2. Build (G_k(\mu,\nu)) from the reusable callback:

* `G_k = get_G_mu_nu_fn(psi_coh_rmuT_X, psi_coh_rmu_Y, Gij)`

3. FFT to R:

* `G_R = get_G_R_fn(G_k, nkx,nky,nkz)`

4. Build (W^c_q(t)) from GN PPM:
   [
   W^c_{q,\mu\nu}(t)= B_{\mu\nu},e^{-i(\Omega_{\mu\nu}-\Omega^{\rm ref})t},
   ]
   masked by `win.mask_B`, implemented by `build_ppm_w_time_q(...)`.

5. Contract to (\Sigma(\mu,\nu;t)):

* `sigma_tau = get_sigma_mu_nu_fn(G_R, W_t_q, nk_tot, bispinor)`

6. Project to band-space channels:

* `sigma_tau_kij_re, sigma_tau_kij_im = get_sigma_kij_channels_fn(psi_proj_rmu_X, psi_proj_rmuT_Y, sigma_tau)`

7. Form the scalar coefficient per ((\omega,t)):
   [
   c(\omega,t) = \alpha , e^{-i(E^{\rm ref}*A+E^{\rm ref}*B)t},e^{i,\omega*{\rm sign},\omega*{\rm flip},\omega,t}.
   ]

8. Combine channels depending on `win.project`:

* if `project=="real"` use (\Re(c,\Sigma)):
  [
  \Re(c),\Re(\Sigma) - \Im(c),\Im(\Sigma)
  ]
* if `project=="imag"` use (\Im(c,\Sigma)):
  [
  \Re(c),\Im(\Sigma) + \Im(c),\Re(\Sigma)
  ]

9. Multiply by `prefactor * scale` and accumulate.

---

## 7. (\omega)-splitting and “canonical” negative-frequency handling (exactly as code)

Let the input grid be `omega_rel_req = omega_values_ry` (already relative to (E_F)). Split:

* `omega_pos = omega_rel_req[omega>=0]`
* `omega_neg_abs = -omega_rel_req[omega<0]`

Then the code runs two calls:

### 7.1 For (\omega_{\rm rel}\ge 0)

* conduction branch uses `kernel_sign=+1` (3 windows if ω_max>0)
* valence branch uses `kernel_sign=-1` (single Laplace window)
* `omega_sign_flip = +1` for both branches

### 7.2 For (\omega_{\rm rel}<0) (canonical map)

Evaluate at (|\omega_{\rm rel}|) with:

* conduction branch `kernel_sign=-1`
* valence branch `kernel_sign=+1`
* **and** `omega_sign_flip = -1` for both branches

Additionally:

* `neg_scale = sigma_scale`
* if `sigma_flip_neg=True`, it flips the entire negative branch sign by setting `neg_scale=-sigma_scale`.

This is exactly what the code currently does.

---

## 8. Invalid GN modes: static-limit correction

If `invalid_mode="static_limit"` and `Wc0_mu_nu` is provided and there are invalid elements:

1. Construct `Wc0_invalid = Wc0_mu_nu` on invalid mask only.

2. Compute two static contractions:

* occupied Green’s function contraction:
  [
  \Sigma_{\rm occ} = \Sigma[G^{\rm occ}, W^c_{\rm invalid}(0)]
  ]
  (implemented by building `Gij_occ` diagonal with occupied ones)

* identity contraction (RI term):
  [
  \Sigma_{\rm RI} = \Sigma[I, W^c_{\rm invalid}(0)]
  ]

3. Form static COH correction:
   [
   \Sigma_{\rm invalid}^{\rm static} = \Sigma_{\rm occ} - \tfrac12 \Sigma_{\rm RI}.
   ]

4. Add it to **all** (\omega) points (in memory or by adding to each streamed chunk).

---

## 9. Practical parameters that actually exist in code

* GN PPM:

  * `omega_p_ry` default 2.0 Ry
  * `fallback_omega` passed to GN extractor
  * head correction applied for ω=0 and ω=iω_p through `apply_head_correction(...)` callback in `gw_jax.py`

* Sigma quadrature:

  * `target_error`, `max_nodes` for Laplace minimax
  * `regularization_width_ry` (ξ), `edge_factor`, `crossing_eps_q`, `crossing_max_nodes`
  * `omega_batch_size` for streaming path
  * `sigma_scale` and debug `sigma_flip_neg`

---

