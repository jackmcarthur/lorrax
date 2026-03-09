# Minimax-Windowed CTSP: Implementation Guide

**Companion to:** `GN_PPM_MINIMAX_SIGMA_GUIDE.md` (theory)

---

## 0. Conventions

### 0.1 Time nodes (the physical Fourier convention)

All propagators use $\phi(t) = e^{-i\tilde{E}t}$. Laplace nodes live on the negative imaginary axis $t = -i\tau$ so that $e^{-i\tilde{E}(-i\tau)} = e^{-\tilde{E}\tau}$ decays. Crossing nodes are real. Both are complex arrays; downstream code never branches on which type.

### 0.2 Positive-axis energies

| Raw | Positive-axis $\tilde{E}$ |
|-----|--------------------------|
| $E_c \geq 0$ | $\tilde{E} = E_c$ |
| $E_v \leq 0$ | $\tilde{E} = -E_v$ |
| $\Omega \geq 0$ | $\tilde{E} = \Omega$ |

### 0.3 FFTs

| Operation | Code | Effect |
|-----------|------|--------|
| $k \to +R$ | `ifftn(f, axes=k_axes, norm='ortho')` | $\tfrac{1}{\sqrt{N_k}}\sum_k e^{+ikR}f_k$ |
| $k \to -R$ | `fftn(f, axes=k_axes, norm='ortho')` | $\tfrac{1}{\sqrt{N_k}}\sum_k e^{-ikR}f_k$ |
| $R \to q$ | `fftn(f, axes=R_axes, norm='ortho')` | $\tfrac{1}{\sqrt{N_k}}\sum_R e^{-iqR}f_R$ |

### 0.4 Spin

$G_{\mathbf{k},ab}(\mu,\nu)$ and $\Sigma_{\mathbf{k},ab}(\mu,\nu)$: spin $2\times 2$. Everything else ($\chi^0$, $\Pi$, $W$, $V$, $B$, $\Omega$): scalar.

---

## 1. Primitives

### 1.1 Quadrature factories (offline, cached)

```python
laplace_minimax(R, ε) → (τ̂, ŵ)     # 1/y ≈ Σ ŵ exp(-τ̂ y) on [1, R]
phase_minimax(A, ε)   → (τ̂, ŵ)     # G(u) ≈ Σ ŵ sin(τ̂ u) on [0, A]
```

Node counts: $N_L \sim O(\log R)$, $N_P \sim 0.75\, A$.

### 1.2 Propagator builder (one function)

```python
def build_propagator(ψ_k, E_k, mask, E_ref, t):
    """
    ψ_k:  (N_k, N_n, N_s, N_μ)  wavefunctions at centroids
    E_k:  (N_k, N_n)             positive-axis eigenvalues
    mask: (N_k, N_n)             window selection
    E_ref: float                  min of E_k[mask]
    t:     complex scalar         time node (-iτ or real)
    
    Returns G_k: (N_k, N_s, N_μ, N_s, N_μ), complex
    """
    phase = where(mask, exp(-1j * (E_k - E_ref) * t), 0.0)    # (N_k, N_n)
    G_k = einsum('knam, kn, knbv -> kambv', ψ_k, phase, ψ_k.conj())
    return G_k
```

Same code for Laplace and crossing. Laplace: $t = -i\tau$ makes phase factors real ∈ [0,1]. Crossing: real $t$ makes them unit-modulus.

**Conjugation variants.** The einsum above gives $\psi_a(r_\mu)\psi^*_b(r_\nu)$ (conduction-like). For valence propagators in $\chi^0$, swap the conjugation to $\psi^*_a(r_\mu)\psi_b(r_\nu)$:

```python
G_v_k = einsum('knam, kn, knbv -> kambv', ψ_v.conj(), phase, ψ_v)
```

This swapped conjugation is what produces the $(ba, \nu\mu)$ transposition in the $R$-space spin trace.

**PPM variant** (no spin, payload is $B_q$):

```python
def build_ppm_propagator(B_q, Ω_q, mask, Ω_ref, t):
    phase = where(mask, exp(-1j * (Ω_q - Ω_ref) * t), 0.0)
    return B_q * phase    # (N_q, N_μ, N_μ)
```

### 1.3 Two contractions

**For $\chi^0$** (spin trace + transpose):

```python
def contract_chi(G_c_R, G_v_mR):
    return einsum('kambv, kbvam -> kmv', G_c_R, G_v_mR)
```

**For $\Sigma^c$** (elementwise, spin broadcasts from $G$, $\Pi$ has no spin):

```python
def contract_sigma(G_R, Π_mR):
    return G_R * Π_mR[:, None, :, None, :]   # broadcast over (a, b)
```

---

## 2. `convolve_frequencies` — the central function

This is the only entry point for frequency-dependent quadrature. It inspects the requested frequencies and decides internally whether to use a single Laplace window or the 3-window scheme.

### 2.1 Signature

```python
def convolve_frequencies(
    omega: complex[N_ω],               # target frequencies (may be complex)
    build_A: Callable[[complex, mask, E_ref], Array],  # (t, mask, E_ref) → propagator A
    build_B: Callable[[complex, mask, E_ref], Array],  # (t, mask, E_ref) → propagator B
    contract: Callable[[Array, Array], Array],          # (A_R, B_mR) → C_R
    kernel_sign: int,          # +1 (minus-kernel: ω−S) or −1 (plus-kernel: ω+S)
    E_A: float[...],           # positive-axis energies for axis A
    E_B: float[...],           # positive-axis energies for axis B
    ξ: float,                  # regularization width
    c_edge: float,             # edge padding (typical 1–2)
    ε_q: float,                # target quadrature error
    k_axes: tuple,             # FFT axes for k-grid
) → result: complex[N_ω, ...]
```

### 2.2 Internal logic

```python
def convolve_frequencies(omega, build_A, build_B, contract,
                         kernel_sign, E_A, E_B, ξ, c_edge, ε_q, k_axes):
    
    omega_max = max(omega.real)
    z_edge = c_edge * ξ
    T = omega_max + z_edge
    
    # --- Decide windows ---
    # Plus-kernel (kernel_sign == -1): always sign-definite → single Laplace
    # Minus-kernel with omega_max == 0: also sign-definite → single Laplace
    # Minus-kernel with omega_max > 0: need 3-window scheme
    
    needs_crossing = (kernel_sign == +1) and (omega_max > 0)
    
    if needs_crossing:
        windows = _build_three_windows(E_A, E_B, T, ξ, c_edge, ε_q, kernel_sign)
    else:
        windows = [_build_single_window(E_A, E_B, omega_max, ε_q, kernel_sign)]
    
    # --- Accumulate over windows ---
    result = 0.0
    for win in windows:
        result += _evaluate_window(
            omega, build_A, build_B, contract,
            win, k_axes,
        )
    
    return result
```

### 2.3 Window dataclass

Each window carries everything needed for evaluation — no external branching:

```python
@dataclass
class Window:
    t_nodes: complex[N_q]    # -iτ (Laplace) or real (crossing)
    α: float[N_q]            # physical quadrature weights
    mask_A: bool[...]         # which A-axis energies are in range
    mask_B: bool[...]         # which B-axis energies are in range
    E_ref_A: float            # min(E_A[mask_A])
    E_ref_B: float            # min(E_B[mask_B])
    omega_sign: int           # +1 or -1
    project: str              # "imag" or "none"
    prefactor: float          # +1.0 or -1.0
```

### 2.4 Window evaluation (one code path)

```python
def _evaluate_window(omega, build_A, build_B, contract, win, k_axes):
    
    # Fold reference-energy phases into weights
    α = win.α * exp(-1j * (win.E_ref_A + win.E_ref_B) * win.t_nodes)
    
    acc = 0.0
    for ℓ, t in enumerate(win.t_nodes):
        
        # Build propagators — convolve_frequencies passes E_ref
        P_A = build_A(t, win.mask_A, win.E_ref_A)
        P_B = build_B(t, win.mask_B, win.E_ref_B)
        
        # FFT to R-space
        A_R  = ifftn(P_A, axes=k_axes, norm='ortho')      # k → +R
        B_mR = fftn(P_B, axes=k_axes, norm='ortho')       # q → -R
        
        # Contract
        C_R = contract(A_R, B_mR)
        
        # FFT R → k/q
        C_kq = fftn(C_R, axes=k_axes, norm='ortho')
        
        # ω-kernel: exp(i · omega_sign · ω · t)
        K = exp(1j * win.omega_sign * omega * t)           # (N_ω,)
        acc += α[ℓ] * K[:, ...] * C_kq[None, ...]
    
    # Project — sole branch between Laplace and crossing
    out = acc.imag if win.project == "imag" else acc.real
    return win.prefactor * out
```

No other branches. No `if laplace`. No `if crossing`. The distinction lives entirely in the `Window` fields: imaginary vs. real `t_nodes`, `"imag"` vs. `"none"` project, and the prefactor.

---

## 3. Window construction

### 3.1 Single Laplace window (sign-definite case)

Used when the kernel is plus-type or when all requested frequencies have zero real part.

```python
def _build_single_window(E_A, E_B, omega_max, ε_q, kernel_sign):
    
    S_min = E_A.min() + E_B.min()
    S_max = E_A.max() + E_B.max()
    
    # Denominator range: |kernel_sign| · ω + S  or  S - ω
    # For plus-kernel (kernel_sign == -1):  x = ω + S ∈ [S_min, omega_max + S_max]
    # For minus-kernel with ω=0:            x = S     ∈ [S_min, S_max]
    x_min = S_min                                     # both cases
    x_max = S_max + omega_max * (kernel_sign == -1)   # omega_max contribution only for plus
    
    R = x_max / x_min
    τ̂, ŵ = laplace_minimax(R, ε_q)
    
    t_nodes = -1j * τ̂ / x_min
    α = ŵ / x_min
    
    # omega_sign, project, prefactor — set once, no downstream branching
    omega_sign = kernel_sign                  # -1 for plus, +1 for minus-at-zero
    project = "none"
    prefactor = -1.0 if kernel_sign == +1 else 1.0   # minus-kernel at ω=0 still needs -1
    
    mask_A = full_like(E_A, True, dtype=bool)
    mask_B = full_like(E_B, True, dtype=bool)
    
    return Window(t_nodes, α, mask_A, mask_B,
                  E_A.min(), E_B.min(), omega_sign, project, prefactor)
```

### 3.2 Three-window scheme (minus-kernel with real frequencies)

```python
def _build_three_windows(E_A, E_B, T, ξ, c_edge, ε_q, kernel_sign):
    z_edge = c_edge * ξ
    windows = []
    
    for name in ["core", "a_stripe", "b_slab"]:
        
        # --- Masks ---
        if name == "core":
            mA = E_A <= T
            mB = E_B <= T
        elif name == "a_stripe":
            mA = E_A > T
            mB = E_B <= T
        else:  # b_slab
            mA = (E_A <= E_A.max())     # full A-range...
            mB = E_B > T               # ...but only high-B
            # Exclude corner already counted in a_stripe:
            mA = mA & ~(E_A > T)  if "dedup" else mA
            # Or just: assign overlap to b_slab, skip in a_stripe
        
        if not mA.any() or not mB.any():
            continue
        
        E_ref_A = E_A[mA].min()
        E_ref_B = E_B[mB].min()
        S_min = E_A[mA].min() + E_B[mB].min()
        S_max = E_A[mA].max() + E_B[mB].max()
        
        # --- Quadrature ---
        is_core = (name == "core")
        
        if is_core:
            A_core = 2 * T / ξ
            τ̂, ŵ = phase_minimax(A_core, ε_q)
            t_nodes = τ̂ / ξ + 0j               # real, cast to complex
            α = ŵ / ξ
        else:
            x_min = max(S_min - (T - z_edge), z_edge)
            x_max = S_max
            R = x_max / x_min
            τ̂, ŵ = laplace_minimax(R, ε_q)
            t_nodes = -1j * τ̂ / x_min
            α = ŵ / x_min
        
        # --- Kernel parameters (all complexity is here) ---
        omega_sign = +1                          # minus-kernel always
        project    = "imag" if is_core else "none"
        prefactor  = +1.0   if is_core else -1.0
        
        windows.append(Window(
            t_nodes, α, mA, mB, E_ref_A, E_ref_B,
            omega_sign, project, prefactor,
        ))
    
    return windows
```

### 3.3 Sign verification

| Case | $t$ | $e^{-i\tilde{E}t}$ | $e^{is_\omega \omega t}$ | Combined | Sums to |
|------|-----|---------------------|--------------------------|----------|---------|
| Plus, Laplace | $-i\tau$ | $e^{-\tilde{E}\tau}$ | $e^{-\omega\tau}$ | $e^{-(\omega+S)\tau}$ | $1/(\omega+S)$ ✓ |
| Minus ext., Laplace | $-i\tau$ | $e^{-\tilde{E}\tau}$ | $e^{+\omega\tau}$ | $-e^{-(S-\omega)\tau}$ | $1/(\omega-S)$ ✓ |
| Minus core, real | $t$ | $e^{-i\tilde{E}t}$ | $e^{+i\omega t}$ | $\operatorname{Im}[e^{i(\omega-S)t}]$ | $F(\omega-S;\xi)$ ✓ |

---

## 4. Calling patterns

### 4.1 $\chi^0_q(\omega)$ — polarizability

```python
# Axis definitions
E_A = E_c_k                           # conduction, shape (N_k, N_c)
E_B = -E_v_k                          # hole energies, shape (N_k, N_v)

# Builders: convolve_frequencies passes (t, mask, E_ref)
def build_G_c(t, mask, E_ref):
    phase = where(mask, exp(-1j * (E_c_k - E_ref) * t), 0.0)
    return einsum('knam, kn, knbv -> kambv', ψ_c, phase, ψ_c.conj())

def build_G_v(t, mask, E_ref):
    phase = where(mask, exp(-1j * (-E_v_k - E_ref) * t), 0.0)
    return einsum('knam, kn, knbv -> kambv', ψ_v.conj(), phase, ψ_v)
    #                                        ^^^ swapped conjugation

# Resonant: minus-kernel
χ_res = convolve_frequencies(omega, build_G_c, build_G_v, contract_chi,
                             kernel_sign=+1, E_A=E_A, E_B=E_B, ξ=ξ, ...)

# Antiresonant: plus-kernel
χ_anti = convolve_frequencies(omega, build_G_c, build_G_v, contract_chi,
                              kernel_sign=-1, E_A=E_A, E_B=E_B, ξ=ξ, ...)

χ_q = prefactor * (χ_res + χ_anti)
```

### 4.2 Phase 1: $\chi^0(0)$ and $\chi^0(i\omega_p)$

```python
# Both are sign-definite → convolve_frequencies auto-selects single Laplace window
omega_static = array([0.0 + 0j, 1j * ω_p])

# At ω=0 and ω=iωp, resonant and antiresonant combine to -2ΔE/(ΔE² + |ω|²)
# which is a single plus-kernel: denominator = ΔE² + |ω|² > 0, no crossing
# But we can also just call the resonant/antiresonant separately at these ω values
# and sum — convolve_frequencies handles both correctly.

χ_q_both = prefactor * (
    convolve_frequencies(omega_static, build_G_c, build_G_v, contract_chi,
                         kernel_sign=+1, ...)   # resonant
  + convolve_frequencies(omega_static, build_G_c, build_G_v, contract_chi,
                         kernel_sign=-1, ...)   # antiresonant
)

χ_q_0   = χ_q_both[0]
χ_q_iwp = χ_q_both[1]

# Dyson → PPM
for q in range(N_q):
    Π_0  = solve(I - v[q] @ χ_q_0[q],   χ_q_0[q])
    Π_ip = solve(I - v[q] @ χ_q_iwp[q], χ_q_iwp[q])
    Ω[q] = ω_p * sqrt(Re(Π_ip / (Π_0 - Π_ip)))
    B[q] = -0.5 * Π_0 * Ω[q]
```

Since `omega_static.real.max() == 0`, `convolve_frequencies` with `kernel_sign=+1` (resonant, minus-kernel) sees `needs_crossing = False` and uses a single Laplace window. The antiresonant call (`kernel_sign=-1`) always uses single Laplace. Both share essentially the same propagator builds.

### 4.3 Phase 2: $\Sigma^c_k(\omega)$

```python
omega_grid = linspace(0, Ω_max, N_ω) + 0j

E_A_cond = E_c_k                       # (N_k, N_c)
E_A_val  = -E_v_k                      # (N_k, N_v)
E_B_ppm  = Ω_q                         # (N_q, N_μ, N_μ) — flattened for windowing

# Conduction (minus-kernel): needs 3-window for real ω > 0
def build_G_cond(t, mask, E_ref):
    phase = where(mask, exp(-1j * (E_c_k - E_ref) * t), 0.0)
    return einsum('knam, kn, knbv -> kambv', ψ_c, phase, ψ_c.conj())

def build_Π(t, mask, E_ref):
    phase = where(mask, exp(-1j * (Ω_q - E_ref) * t), 0.0)
    return B_q * phase

Σ_cond = convolve_frequencies(omega_grid, build_G_cond, build_Π, contract_sigma,
                               kernel_sign=+1, E_A=E_A_cond, E_B=E_B_ppm, ξ=ξ, ...)

# Valence (plus-kernel): single Laplace always
def build_G_val(t, mask, E_ref):
    phase = where(mask, exp(-1j * (-E_v_k - E_ref) * t), 0.0)
    return einsum('knam, kn, knbv -> kambv', ψ_v.conj(), phase, ψ_v)

Σ_val = convolve_frequencies(omega_grid, build_G_val, build_Π, contract_sigma,
                              kernel_sign=-1, E_A=E_A_val, E_B=E_B_ppm, ξ=ξ, ...)

# Assemble
Σ_c = Σ_cond + Σ_val
Σ_c = einsum('...μη, ...ηζ, ...ζν -> ...μν', v, Σ_c, v)   # Coulomb sandwich
Σ = Σ_x + Σ_c
Σ_ij = einsum('kaμ, kaμbνω, kbν -> kijω', ψ.conj(), Σ, ψ)  # band projection
```

---

## 5. Quadrature bound formulas

### 5.1 Single Laplace window

$x_\text{min} = S_\text{min}$, $x_\text{max} = S_\text{max}$ (plus $\Omega$ for plus-kernel). $R = x_\text{max}/x_\text{min}$.

### 5.2 Crossing core

$A_\text{core} = 2T/\xi$ where $T = \omega_\text{max} + c_\text{edge} \cdot \xi$.

### 5.3 Exterior Laplace (A-stripe, B-slab)

$x_\text{min} = \max(S_\text{min} - \Omega_\text{max},\; z_\text{edge})$, $x_\text{max} = S_\text{max}$. $R = x_\text{max}/x_\text{min}$.

---

## 6. Normalizations: derive from existing code, not from this document

`convolve_frequencies` returns a **raw quadrature sum** — it handles the $1/x$ approximation, the gap exponential (via $E_\text{ref}$ folding), and the window decomposition. It does **not** include physics prefactors. The caller is responsible for all overall normalization.

**Do not try to derive the prefactors from first principles using this document.** Instead, derive them by reading the existing working CTSP code in `w_isdf.py` and `chi_omega_quadrature.md`. The procedure:

1. Read `_get_chi_kernel()` and `chi_tile_gl()` in the existing code. Identify every scalar factor that multiplies the GL quadrature sum: the $-2\zeta$ in front, the $e^{-(\zeta E_\text{gap} - 1)\tau}$ gap factor, the $1/(n_\text{spin} \cdot n_\text{spinor})$ normalization, and any $1/\sqrt{N_k}$ factors from the FFT convention.

2. Determine which of those factors are already inside the minimax quadrature weights $\alpha$ (the $1/x$ approximation and the gap exponential are — they're folded into $\alpha$ via the $E_\text{ref}$ phase in §2.4) and which are not (the $-2$, spin normalization, and any overall signs).

3. The factors that are **not** inside $\alpha$ become the `prefactor` that the caller multiplies after `convolve_frequencies` returns. Confirm by computing $\chi^0(0)$ via the new code and comparing against the existing `get_static_chi_q_jax()` output element-by-element.

4. For $\Sigma^c$, do the same comparison against existing COHSEX at $\omega = 0$. The static COH term $\Sigma^\text{COH} = \frac{1}{2} G^\text{all} \cdot (W - V)$ should relate to the $\omega \to 0$ limit of your $\Sigma^c$ (up to the factor-of-2 from the static approximation).

**The existing code is the ground truth.** The GL scale factor $\zeta = 1/\sqrt{E_\text{bw} E_\text{gap}}$ from Kim-2020 is absorbed differently in the minimax weights, so a direct formula comparison will be misleading. A numerical comparison at $\omega = 0$ is the reliable path.

---

## 7. Mapping to existing code

The agent will be editing a working static COHSEX + CTSP codebase. Here is what replaces what:

| Existing code | Replaced by | Notes |
|---------------|-------------|-------|
| `chi_tile_gl()` in `chi_omega_quadrature.md` | `convolve_frequencies` with `contract_chi`, `kernel_sign=-1` | GL propagator builds → `build_propagator`; GL weights → Laplace minimax weights |
| `chi_tile_hgl()` in `chi_omega_quadrature.md` | `convolve_frequencies` with `contract_chi`, `kernel_sign=+1` | HGL Euler-identity propagators → `build_propagator` with real $t$; sin/cos accumulation → `project="imag"` |
| `precompute_exp_weights()` | Phase computation inside `build_propagator`: `exp(-1j * (E - E_ref) * t)` | The $\zeta\tau\Delta E$ exponent becomes $(E - E_\text{ref}) \cdot \tau_\text{phys}$ with $\tau_\text{phys}$ from minimax rescaling |
| Window-pair loop `for l, m in window_pairs` | Internal window logic of `convolve_frequencies` | Old: many small GL windows. New: 3 coarse minimax windows (or 1 for sign-definite) |
| `precompute_phases()` for $\omega$-dependent sin/cos | Frequency kernel `exp(1j * omega_sign * omega * t)` inside `_evaluate_window` | Old: separate GL and HGL phase arrays. New: one complex exponential for all cases |
| `get_static_chi_q_jax()` | `convolve_frequencies(omega=[0], ...)` → auto-selects single Laplace window | Should reproduce existing results to minimax tolerance |
| `get_static_w_q_jax()` | Unchanged — Dyson solve is independent of quadrature | Same LU factorization of $(I - V\chi^0)$ |

**What to preserve unchanged:**
- All ISDF fitting code (`load_wfns.py`, zeta pipeline)
- Coulomb matrix elements (`compute_vcoul.py`)
- Dyson solve for $W$ (`w_isdf.py:get_static_w_q_jax`)
- Exchange self-energy (`gw_jax.py:get_sigma_x_kij_jax`)
- Band projection (`gw_jax.py` final einsum)
- $q = 0$ head/wings correction (`chi_from_dipole.py`)
- All sharding/chunking infrastructure

---

## 8. Array shapes through one `_evaluate_window` call

Concrete shapes for a system with `nkx, nky, nkz` $k$-grid, `n_s = 2` spinor components, `n_μ` ISDF points.

```
Input to build_propagator:
  ψ_k:    (nk, n_band, n_s, n_μ)     where nk = nkx*nky*nkz
  E_k:    (nk, n_band)
  mask:   (nk, n_band)
  
Output of build_propagator:
  G_k:    (nk, n_s, n_μ, n_s, n_μ)

Reshape for FFT (flatten k → 3D grid):
  G_k:    (n_s, n_μ, n_s, n_μ, nkx, nky, nkz)
                                ^^^  ^^^  ^^^  ← FFT axes = (-3, -2, -1)

After ifftn (k → +R):
  G_R:    (n_s, n_μ, n_s, n_μ, nRx, nRy, nRz)    same shape, R-space

PPM propagator (no spin):
  Π_q:    (nq, n_μ, n_μ)
  reshape → (n_μ, n_μ, nqx, nqy, nqz)
  after fftn (q → -R): (n_μ, n_μ, nRx, nRy, nRz)

Contract (Σ case, elementwise + spin broadcast):
  Σ_R = G_R * Π_mR[None, :, None, :, :, :, :]
  →     (n_s, n_μ, n_s, n_μ, nRx, nRy, nRz)

Contract (χ case, spin trace + transpose):
  χ_R = einsum('ambvxyz, bvamxyz -> mvxyz', G_c_R, G_v_mR)
  →     (n_μ, n_μ, nRx, nRy, nRz)

After fftn (R → k/q):
  same shape, k/q-space

ω-kernel broadcast:
  K:      (n_ω,)
  result: (n_ω, n_s, n_μ, n_s, n_μ, nkx, nky, nkz)   for Σ
          (n_ω, n_μ, n_μ, nqx, nqy, nqz)               for χ
```

**Critical:** the reshape from flat `(nk, ...)` to `(..., nkx, nky, nkz)` before the FFT must put the $k$-grid dimensions last (contiguous) for `ifftn`/`fftn` to act on the correct axes. The existing code does this with an explicit transpose — see `chi_tile_gl` in `chi_omega_quadrature.md` for the pattern.

---

## 9. PPM B-axis mask mechanics

The PPM pole frequencies $\Omega_{q,\mu\nu}$ are a dense `(N_q, N_μ, N_μ)` array — not a 1D energy axis. "Filtering to the B-range" means an elementwise boolean mask:

```python
mask_B = (Ω_q >= B_lo) & (Ω_q < B_hi)    # shape (N_q, N_μ, N_μ)
```

Some matrix elements are in the window at some $q$-points and not others. The `build_ppm_propagator` zeros out masked elements, so the $R$-space $\Pi^\text{PPM}$ from one window is a partial sum. **The sum over windows exactly recovers the full $\Pi^\text{PPM}$** — this is an exact partition, not an approximation.

For quadrature bounds, $S_\text{min}$ and $S_\text{max}$ are computed as:

```python
# All possible S = E_A[k,n] + Ω_q[q,μ,ν] where both masks are True
# In practice, take the global extrema:
S_min = E_A[mask_A].min() + Ω_q[mask_B].min()
S_max = E_A[mask_A].max() + Ω_q[mask_B].max()
```

These are conservative bounds (not every $(k,n,q,\mu,\nu)$ combination actually occurs), but they're tight enough for the quadrature — a slightly wider Laplace interval just means a slightly larger $R$, costing one or two extra nodes at worst.

---

## 10. Validation ladder

Test in this order. Each step catches a specific class of bugs.

**Step 1: $\chi^0(0)$ against existing code.**

```python
χ_new = convolve_frequencies(omega=[0], build_G_c, build_G_v, contract_chi,
                              kernel_sign=+1, ...) \
      + convolve_frequencies(omega=[0], build_G_c, build_G_v, contract_chi,
                              kernel_sign=-1, ...)
χ_new *= overall_prefactor

χ_old = get_static_chi_q_jax(...)   # existing working code

assert allclose(χ_new, χ_old, atol=ε_q)
```

**What this catches:** normalization prefactors, FFT conventions, spin trace signs, $E_\text{ref}$ handling. This is the most important test — if this passes, the quadrature machinery is correct.

**Step 2: $\chi^0(\omega)$ at a few real frequencies against brute-force.**

For a small system (e.g., 2-atom Si, Gamma-point only), compute $\chi^0(\omega)$ by direct summation over $cv$ pairs (the $O(N^4)$ formula) and compare against `convolve_frequencies`. This validates the crossing quadrature and the window routing.

**Step 3: GN-PPM parameters against reference.**

Compare $\Omega_{q,\mu\nu}$, $B_{q,\mu\nu}$ against either (a) a direct frequency-domain PPM extraction (compute $\Pi$ at $\omega = 0$ and $\omega = i\omega_p$ by explicit $cv$ summation, then solve the GN equations), or (b) published BerkeleyGW/Yambo PPM values for a standard test system.

**Step 4: $\Sigma^c(\omega = 0)$ against static COHSEX.**

At $\omega = 0$, the full-frequency $\Sigma^c$ should relate to the static COH approximation. Not an exact match (the static COHSEX uses $W(\omega=0)$ while full-frequency PPM integrates over the pole), but they should agree to $\sim 0.1$–$0.5$ eV for well-behaved semiconductors. A discrepancy of several eV indicates a sign or factor-of-2 error.

**Step 5: QP corrections for Si.**

Published $G_0W_0$-PPM band gaps for Si are $\sim 1.1$–$1.3$ eV (depending on pseudopotential and convergence parameters). This is the end-to-end validation.

---

## 11. Implementation checklist

| Component | Role |
|-----------|------|
| `laplace_minimax(R, ε)` | Dimensionless non-crossing factory |
| `phase_minimax(A, ε)` | Dimensionless crossing factory |
| `build_propagator(ψ, E, mask, E_ref, t)` | Universal $e^{-i(\tilde{E}-E_\text{ref})t}$ builder |
| `build_ppm_propagator(B, Ω, mask, Ω_ref, t)` | PPM variant (no spin) |
| `contract_chi(A_R, B_mR)` | Spin trace + transpose |
| `contract_sigma(G_R, Π_mR)` | Elementwise with spin broadcast |
| `convolve_frequencies(...)` | Entry point: owns windowing, calls builders, integrates |

Everything else — normalization prefactors (§6), Coulomb sandwiching, exchange, band projection, $q=0$ head correction — lives outside `convolve_frequencies`. Derive normalizations from the existing code by numerical comparison (§6, §10), not from first principles.
