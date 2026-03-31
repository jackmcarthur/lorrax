# 2) `MINIMAX_CTSP_IMPLEMENTATION_REVISED.md` (updated to match code)

## Minimax-Windowed CTSP: Implementation Guide (current code)

**This document describes what is actually implemented in `ppm_sigma.py`** for the sigma integration path. It does **not** describe a generic `convolve_frequencies` API (that abstraction is not what the current code uses).

---

## 0. Conventions

### 0.1 Time nodes

* Laplace nodes: (t=-i\tau) (decaying)
* Crossing nodes: (t\in\mathbb{R}) (oscillatory)

Both are represented as complex numbers; evaluation is uniform.

### 0.2 Energies

Inside sigma:

* axis A is (E_A = E_c) (conduction positive axis) or (E_A=h_v) (valence hole energy)
* axis B is (E_B=\Omega) (GN pole frequencies, positive)

Axis arrays are always nonnegative.

---

## 1. Quadrature solvers used

### 1.1 Laplace minimax for (1/x) on ([x_{\min}, x_{\max}])

The code calls:

```python
q = solve_laplace_minimax_interval(x_min, x_max, target_error, max_nodes)
```

and then uses:

* `t_nodes = -1j * q.tau`
* `alpha = q.alpha`

so that, numerically, (1/x\approx \sum_u \alpha_u e^{-\tau_u x}) over the interval.

### 1.2 Phase minimax for crossing “HGL” core

The code calls:

```python
q_cross = solve_phase_minimax_bandwidth(
    A_core,
    target_error=target_error,
    max_nodes=crossing_max_nodes,
    eps_q=crossing_eps_q,
    target_kind="hgl",
)
```

and then converts:

* `t_nodes = q_cross.tau / xi`  (real)
* `alpha = q_cross.alpha / xi`

This window is evaluated with `project="imag"` (see §3.3).

---

## 2. Sigma window dataclass (exact)

The code uses:

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

---

## 3. Window construction (exactly as in code)

### 3.1 Single Laplace window (`_build_single_sigma_window`)

Inputs: masked arrays (E_A) and (E_B), batch of (\omega\ge 0), and `kernel_sign`.

Define:
[
S_{\min}=\min(A)+\min(B),\quad S_{\max}=\max(A)+\max(B),\quad \omega_{\max}=\max(\omega).
]
[
x_{\min}=\max(S_{\min},10^{-12}).
]
[
x_{\max}=\begin{cases}
S_{\max}+\omega_{\max} & \text{if } kernel_sign=-1\
S_{\max} & \text{if } kernel_sign=+1
\end{cases}
\quad\text{(with a small }(1+10^{-9})\text{ guard).}
]

Then solve Laplace minimax on ([x_{\min},x_{\max}]), with:

* (t=-i\tau), `project="real"`, `omega_sign=kernel_sign`.

Prefactor stored in the window is:

* `docs_prefactor = -1` if `kernel_sign==+1` else `+1`
* but because `get_sigma_mu_nu_fn` already has a global (-1),
  [
  \texttt{prefactor} = -\texttt{docs_prefactor}.
  ]

### 3.2 Three-window scheme (`_build_three_sigma_windows`)

Used only when:

* `kernel_sign == +1` **and** `omega_max > 1e-14`.

Parameters:
[
\xi=\texttt{regularization_width_ry},\quad z_{\rm edge}=\texttt{edge_factor}\cdot\xi,\quad
T=\omega_{\max}+z_{\rm edge}.
]

Masks (**exactly**):

* core: (A\le T) and (B\le T)
* a_stripe: (A>T) and (B\le T)
* b_slab: (A) = full and (B>T)

Note: this means the ((A>T,B>T)) corner is included in **b_slab** (no dedup).

For each window, set:

* `E_ref_A = min(A_vals)`, `E_ref_B = min(B_vals)`.

#### core

[
A_{\rm core}=\max(2T/\xi,10^{-8})
]
Phase minimax (G)-approx is solved on ([0,A_{\rm core}]), then:

* `t_nodes = tau/xi` (real)
* `alpha = alpha/xi`
* `omega_sign=+1`
* `project="imag"`
* `docs_prefactor=+1` → stored `prefactor=-1`.

#### a_stripe and b_slab

[
x_{\min}=\max(S_{\min}-(T-z_{\rm edge}), z_{\rm edge}, 10^{-12})
=\max(S_{\min}-\omega_{\max}, z_{\rm edge}, 10^{-12}),
]
[
x_{\max}=\max(S_{\max}, x_{\min}(1+10^{-9})).
]

Then Laplace minimax is solved and used with:

* `omega_sign=+1`
* `project="real"`
* `docs_prefactor=-1` → stored `prefactor=+1`.

---

## 4. Evaluation of one window in band space (`_convolve_sigma_branch_kij`)

At each time node (t):

### 4.1 Build (G_k) from axis-A weights

Axis A phase:
[
\text{phase}_A = e^{-i(E_A-E^{\rm ref}_A)t}.
]

The code applies `mask_A` and inserts this on the diagonal of `Gij`:
[
G_{knm}=\delta_{nm},\text{phase}_A(k,n).
]

Then:

* `G_k = get_G_mu_nu_fn(psi_coh_rmuT_X, psi_coh_rmu_Y, Gij)`
* `G_R = get_G_R_fn(G_k, nkx,nky,nkz)`.

### 4.2 Build (W^c_q(t)) from GN PPM

[
W^c(t)= B,e^{-i(\Omega-E^{\rm ref}_B)t}
]
masked by `mask_B`, via `build_ppm_w_time_q`.

### 4.3 Contract in (\mu\nu) and project to ((k,i,j)) channels

* `sigma_tau = get_sigma_mu_nu_fn(G_R, W_t_q, nk_tot, bispinor)`

Then project to two channels:

* `sigma_tau_kij_re = K[ Re(sigma_tau) ]`
* `sigma_tau_kij_im = K[ Im(sigma_tau) ]`

This is returned by:

* `get_sigma_kij_channels_fn(psi_proj_rmu_X, psi_proj_rmuT_Y, sigma_tau)`.

### 4.4 Apply scalar coefficient (c(\omega,t)) and projection rule

Define:
[
\alpha_{\rm eff}=\alpha,e^{-i(E^{\rm ref}*A+E^{\rm ref}*B)t},
]
and:
[
\omega*{\rm sign,eff} = (\texttt{win.omega_sign})\cdot(\texttt{omega_sign_flip}).
]
Then:
[
c(\omega,t)=\alpha*{\rm eff},e^{i,\omega_{\rm sign,eff},\omega,t}.
]

Let (c = c_r + i c_i), (\Sigma=\Sigma_r + i\Sigma_i) be the projected channels.

* For `project="real"`:
  [
  \Re(c\Sigma)=c_r\Sigma_r - c_i\Sigma_i
  ]
* For `project="imag"`:
  [
  \Im(c\Sigma)=c_r\Sigma_i + c_i\Sigma_r
  ]

Then multiply by:
[
(\texttt{win.prefactor})\cdot(\texttt{scale})
]
and accumulate.

Streaming mode writes omega chunks to HDF5 without repeating τ work.

---

## 5. Omega splitting logic in `compute_sigma_c_ppm_omega_grid` (exact)

Input `omega_values_ry` is treated as (\omega_{\rm rel}). Split at 0:

* `omega_pos = omega[>=0]`
* `omega_neg_abs = -omega[<0]`

Then:

### 5.1 For `omega_pos`

Call `_convolve_kij` with:

* conduction branch: `kernel_sign = +1`, `omega_sign_flip = +1`
* valence branch: `kernel_sign = -1`, `omega_sign_flip = +1`
* `scale = sigma_scale`

### 5.2 For `omega_neg_abs`

Call `_convolve_kij` with:

* conduction branch: `kernel_sign = -1`, `omega_sign_flip = -1`
* valence branch: `kernel_sign = +1`, `omega_sign_flip = -1`
* `scale = sigma_scale` (optionally flipped by `sigma_flip_neg`)

This is the “canonical” negative-frequency mapping currently in the file.

---

## 6. Invalid GN modes: static-limit correction (exact)

If `invalid_mode="static_limit"`:

* exclude invalid GN poles from the dynamic sum: `B_mask &= valid_mask`
* then add the static COH correction for invalid elements using `Wc0_mu_nu`:
  [
  \Sigma_{\rm invalid}^{\rm static} = \Sigma_{\rm occ} - \tfrac12 \Sigma_{\rm RI}
  ]
  and add it to all (\omega) points.

---

## 7. Debug output you see in logs

If `debug_quadrature=True`, the code prints per window:

* `[quad] ... max|1/x-approx|` for Laplace windows (it evaluates the quadrature approximation of (1/x) on log-spaced samples between `x_min` and `x_max`).
* `[quad] ... max|G-approx|` for core windows (compares the phase minimax approximation to `docs_mod.G_hgl`).
* `[mask] ... A=... B=...` showing how many axis entries are included.
* also prints reconstructed vacuum energy ranges for A when `efermi_vac` is provided.
