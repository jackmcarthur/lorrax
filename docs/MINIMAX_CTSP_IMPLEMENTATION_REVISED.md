# Minimax-Windowed CTSP: Implementation Guide

**Companion to:** `GN_PPM_MINIMAX_SIGMA_GUIDE.md` (theory)

---

## 0. Conventions

### 0.1 Time nodes

Propagators use $\phi(t) = e^{-i\tilde{E}t}$. Laplace nodes: $t = -i\tau$ (negative imaginary axis, decaying). Crossing nodes: real $t$ (oscillatory). Both are complex arrays; downstream code never branches on which type.

### 0.2 Energy referencing

Eigenvalues are stored in the **vacuum reference**: $E_n^\text{vac}$. The Fermi level $E_F$ is a parameter. Positive-axis values:

$$\tilde{E}_c = E_c^\text{vac} - E_F, \qquad h_v = E_F - E_v^\text{vac}, \qquad \tilde{\Omega} = \Omega_{q,\mu\nu}$$

`convolve_frequencies` computes these internally from vacuum eigenvalues and $E_F$. Builders receive vacuum eigenvalues and a vacuum-referenced $E_\text{ref}$ — they never see $E_F$.

### 0.3 FFTs

| Operation | Code | Effect |
|-----------|------|--------|
| $k \to +R$ | `ifftn(f, axes=k_axes, norm='ortho')` | $\tfrac{1}{\sqrt{N_k}}\sum_k e^{+ikR}f_k$ |
| $k \to -R$ | `fftn(f, axes=k_axes, norm='ortho')` | $\tfrac{1}{\sqrt{N_k}}\sum_k e^{-ikR}f_k$ |
| $R \to q$ | `fftn(f, axes=R_axes, norm='ortho')` | $\tfrac{1}{\sqrt{N_k}}\sum_R e^{-iqR}f_R$ |

### 0.4 Spin

$G$ and $\Sigma$: spin $2\times 2$. Everything else: scalar.

---

## 1. Primitives

### 1.1 Quadrature factories (offline, cached)

```python
laplace_minimax(R, ε) → (τ̂, ŵ)     # 1/y ≈ Σ ŵ exp(-τ̂ y) on [1, R]
phase_minimax(A, ε)   → (τ̂, ŵ)     # G(u) ≈ Σ ŵ sin(τ̂ u) on [0, A]
```

### 1.2 Propagator builder (one function)

```python
def build_propagator(ψ_k, E_vac_k, mask, E_ref_vac, sign, t):
    """
    ψ_k:       (N_k, N_n, N_s, N_μ)  wavefunctions at centroids
    E_vac_k:   (N_k, N_n)             vacuum-referenced eigenvalues
    mask:      (N_k, N_n)             window selection
    E_ref_vac: float                   vacuum-referenced E_ref
    sign:      +1 or -1               +1 for conduction, -1 for valence
    t:         complex scalar          time node
    
    Returns G_k: complex (N_k, N_s, N_μ, N_s, N_μ)
    """
    # sign * (E_vac - E_ref_vac) ≥ 0 by construction
    phase = where(mask, exp(-1j * sign * (E_vac_k - E_ref_vac) * t), 0.0)
    
    if sign == +1:  # conduction-like: ψ ψ*
        return einsum('knam, kn, knbv -> kambv', ψ_k, phase, ψ_k.conj())
    else:           # valence-like: ψ* ψ (swapped conjugation)
        return einsum('knam, kn, knbv -> kambv', ψ_k.conj(), phase, ψ_k)
```

The phase is $e^{-i \cdot \text{sign} \cdot (E^\text{vac} - E^\text{ref,vac}) \cdot t}$:
- Conduction ($\text{sign}=+1$): $e^{-i(E_c - E_c^{\min})t}$. Exponent $\geq 0$.
- Valence ($\text{sign}=-1$): $e^{+i(E_v - E_v^{\max})t} = e^{-i(E_v^{\max} - E_v)t}$. Exponent $\geq 0$.

Laplace: decaying. Crossing: unit modulus. Same code.

**PPM variant** (no spin):

```python
def build_ppm_propagator(B_q, Ω_q, mask, Ω_ref, t):
    phase = where(mask, exp(-1j * (Ω_q - Ω_ref) * t), 0.0)
    return B_q * phase    # (N_q, N_μ, N_μ)
```

### 1.3 Two contractions

```python
def contract_chi(G_c_R, G_v_mR):
    # Spin trace + transpose → scalar (μ,ν)
    return einsum('kambv, kbvam -> kmv', G_c_R, G_v_mR)

def contract_sigma(G_R, Π_mR):
    # Elementwise, Π has no spin → broadcasts
    return G_R * Π_mR[:, None, :, None, :]
```

---

## 2. `convolve_frequencies`

### 2.1 Signature

```python
def convolve_frequencies(
    omega_vac: complex[N_ω],    # vacuum-referenced target frequencies
    E_F: float,                  # Fermi level (vacuum reference)
    build_A: Callable,           # (t, mask, E_ref_vac, sign) → propagator A
    build_B: Callable,           # (t, mask, E_ref_vac) → propagator B
    contract: Callable,          # (A_R, B_mR) → C_R
    kernel_sign: int,            # +1 (minus-kernel) or -1 (plus-kernel)
    E_A_vac: float[...],         # vacuum eigenvalues for axis A
    sign_A: int,                 # +1 (conduction) or -1 (valence)
    E_B: float[...],             # positive-axis energies for axis B (Ω, already ≥ 0)
    ξ: float,
    c_edge: float,
    ε_q: float,
    k_axes: tuple,
) → result: complex[N_ω, ...]
```

### 2.2 Internal logic — handling $\omega$ above and below $E_F$

```python
def convolve_frequencies(omega_vac, E_F, build_A, build_B, contract,
                         kernel_sign, E_A_vac, sign_A, E_B, ξ, c_edge, ε_q, k_axes):
    
    ω_rel = omega_vac.real - E_F
    
    # Positive-axis energies for axis A (from vacuum eigenvalues + E_F)
    E_A_pos = sign_A * (E_A_vac - E_F)    # ≥ 0 by construction
    
    # Split ω at E_F
    mask_pos = ω_rel >= 0
    mask_neg = ω_rel < 0
    
    result = zeros(N_ω, ...)
    
    # --- ω_rel ≥ 0: standard treatment ---
    if mask_pos.any():
        ω_pos = ω_rel[mask_pos]           # ≥ 0
        windows = _decide_windows(ω_pos, kernel_sign, E_A_pos, E_B, ξ, c_edge, ε_q)
        for win in windows:
            result[mask_pos] += _evaluate_window(ω_pos, build_A, build_B, contract,
                                                  win, E_A_vac, sign_A, E_F, k_axes)
    
    # --- ω_rel < 0: flip to |-ω_rel| and swap kernel sign ---
    if mask_neg.any():
        ω_neg_abs = -ω_rel[mask_neg]      # > 0
        flipped_sign = -kernel_sign        # minus ↔ plus
        windows = _decide_windows(ω_neg_abs, flipped_sign, E_A_pos, E_B, ξ, c_edge, ε_q)
        for win in windows:
            # The frequency kernel sign also flips
            win_flipped = win._replace(omega_sign=-win.omega_sign)
            result[mask_neg] += _evaluate_window(ω_neg_abs, build_A, build_B, contract,
                                                  win_flipped, E_A_vac, sign_A, E_F, k_axes)
    
    return result
```

**The key insight:** for $\omega_\text{rel} < 0$, the denominator $\omega_\text{rel} + S = S - |\omega_\text{rel}|$ is a minus-kernel in $|\omega_\text{rel}|$, while $\omega_\text{rel} - S = -(S + |\omega_\text{rel}|)$ is a plus-kernel in $|\omega_\text{rel}|$. So the treatment of $\omega < E_F$ is identical to $\omega > E_F$ with `kernel_sign` flipped.

### 2.3 Window decision

```python
def _decide_windows(ω_pos, kernel_sign, E_A_pos, E_B, ξ, c_edge, ε_q):
    """ω_pos is ≥ 0. kernel_sign: +1 (minus-kernel) or -1 (plus-kernel)."""
    ω_max = ω_pos.max()
    needs_crossing = (kernel_sign == +1) and (ω_max > 0)
    
    if needs_crossing:
        return _build_three_windows(E_A_pos, E_B, ω_max, ξ, c_edge, ε_q, kernel_sign)
    else:
        return [_build_single_window(E_A_pos, E_B, ω_max, ε_q, kernel_sign)]
```

### 2.4 Window dataclass

```python
@dataclass
class Window:
    t_nodes: complex[N_q]    # -iτ (Laplace) or real (crossing)
    α: float[N_q]            # physical quadrature weights
    mask_A: bool[...]
    mask_B: bool[...]
    E_ref_A_vac: float       # vacuum-referenced E_ref for axis A
    E_ref_B: float           # E_ref for axis B (already positive-axis)
    omega_sign: int           # +1 or -1
    project: str              # "imag" or "none"
    prefactor: float          # +1.0 or -1.0
```

### 2.5 Window evaluation (one code path)

```python
def _evaluate_window(ω_pos, build_A, build_B, contract,
                     win, E_A_vac, sign_A, E_F, k_axes):
    
    # Gap phase: uses POSITIVE-AXIS refs, which involve E_F for the band axis
    E_ref_A_pos = sign_A * (win.E_ref_A_vac - E_F)   # ≥ 0
    E_ref_B_pos = win.E_ref_B                          # ≥ 0
    gap_ref = E_ref_A_pos + E_ref_B_pos
    
    α = win.α * exp(-1j * gap_ref * win.t_nodes)
    
    acc = 0.0
    for ℓ, t in enumerate(win.t_nodes):
        
        # Build propagators — receive VACUUM E_ref, sign
        P_A = build_A(t, win.mask_A, win.E_ref_A_vac, sign_A)
        P_B = build_B(t, win.mask_B, win.E_ref_B)
        
        # FFT + contract + FFT back
        A_R  = ifftn(P_A, axes=k_axes, norm='ortho')
        B_mR = fftn(P_B, axes=k_axes, norm='ortho')
        C_R  = contract(A_R, B_mR)
        C_kq = fftn(C_R, axes=k_axes, norm='ortho')
        
        # ω-kernel
        K = exp(1j * win.omega_sign * ω_pos * t)
        acc += α[ℓ] * K[:, ...] * C_kq[None, ...]
    
    out = acc.imag if win.project == "imag" else acc.real
    return win.prefactor * out
```

No branches between Laplace and crossing. The distinction lives in the `Window` fields.

---

## 3. Window construction

### 3.1 Single Laplace window (sign-definite)

```python
def _build_single_window(E_A_pos, E_B, ω_max, ε_q, kernel_sign):
    S_min = E_A_pos.min() + E_B.min()
    S_max = E_A_pos.max() + E_B.max()
    
    # Plus-kernel: x = ω + S. Minus-kernel at ω=0: x = S.
    x_min = S_min
    x_max = S_max + ω_max * (kernel_sign == -1)
    
    R = x_max / x_min
    τ̂, ŵ = laplace_minimax(R, ε_q)
    t_nodes = -1j * τ̂ / x_min
    α = ŵ / x_min
    
    omega_sign = kernel_sign
    project = "none"
    prefactor = -1.0 if kernel_sign == +1 else 1.0
    
    mask_A = full(True)
    mask_B = full(True)
    E_ref_A_vac = ...   # vacuum eigenvalue corresponding to E_A_pos.min()
    E_ref_B = E_B.min()
    
    return Window(t_nodes, α, mask_A, mask_B, E_ref_A_vac, E_ref_B,
                  omega_sign, project, prefactor)
```

### 3.2 Three-window scheme (minus-kernel with real $\omega > 0$)

```python
def _build_three_windows(E_A_pos, E_B, ω_max, ξ, c_edge, ε_q, kernel_sign):
    z_edge = c_edge * ξ
    T = ω_max + z_edge
    windows = []
    
    for name in ["core", "a_stripe", "b_slab"]:
        # --- Masks ---
        mA = {"core": E_A_pos <= T,
               "a_stripe": E_A_pos > T,
               "b_slab": E_A_pos <= E_A_pos.max()}[name]    # full A for b_slab
        mB = {"core": E_B <= T,
               "a_stripe": E_B <= T,
               "b_slab": E_B > T}[name]
        # Deduplicate corner: b_slab excludes A > T
        if name == "b_slab":
            mA = mA & (E_A_pos <= T)
            # Or equivalently: assign corner to a_stripe
        
        if not mA.any() or not mB.any():
            continue
        
        E_ref_A_pos = E_A_pos[mA].min()
        E_ref_B = E_B[mB].min()
        S_min = E_A_pos[mA].min() + E_B[mB].min()
        S_max = E_A_pos[mA].max() + E_B[mB].max()
        
        is_core = (name == "core")
        
        if is_core:
            A_core = 2 * T / ξ
            τ̂, ŵ = phase_minimax(A_core, ε_q)
            t_nodes = τ̂ / ξ + 0j
            α = ŵ / ξ
        else:
            x_min = max(S_min - ω_max, z_edge)
            x_max = S_max
            R = x_max / x_min
            τ̂, ŵ = laplace_minimax(R, ε_q)
            t_nodes = -1j * τ̂ / x_min
            α = ŵ / x_min
        
        omega_sign = +1
        project    = "imag" if is_core else "none"
        prefactor  = +1.0   if is_core else -1.0
        
        E_ref_A_vac = ...   # vacuum eigenvalue corresponding to E_ref_A_pos
        
        windows.append(Window(t_nodes, α, mA, mB, E_ref_A_vac, E_ref_B,
                              omega_sign, project, prefactor))
    
    return windows
```

### 3.3 Sign verification (theory guide §5.4)

All products:

| Pairing | Phase product | $E_F$ cancels? |
|---------|--------------|----------------|
| $\chi$: $(E_c, h_v)$ | $e^{i(\omega - E_c^\text{vac} + E_v^\text{vac})t}$ | $E_F$ never enters |
| $\Sigma^{(-)}$: $(\tilde{E}_c, \Omega)$ | $e^{i(\omega - E_c^\text{vac} - \Omega)t}$ | ✓ cancels |
| $\Sigma^{(+)}$: $(h_v, \Omega)$ | $e^{-i(\omega - E_v^\text{vac} + \Omega)t}$ | ✓ cancels |

---

## 4. Calling patterns

### 4.1 Phase 1: $\chi^0(0)$ and $\chi^0(i\omega_p)$

$\chi^0$ uses transfer frequencies — no $E_F$ shift. But `convolve_frequencies` still takes $E_F$ for occupancy masks.

```python
omega_static = array([0.0 + 0j, 1j * ω_p])

# E_F is used for occupancy masks only, not frequency shifting
# For χ, the "frequencies" are transfer frequencies, not QP frequencies
# Pass E_F = 0 or handle separately — see note below

χ_q = prefactor * (
    convolve_frequencies(omega_static, E_F=0,    # transfer freq, no shift
                         build_G_c, build_G_v, contract_chi,
                         kernel_sign=+1, E_A_vac=E_c_vac, sign_A=+1,
                         E_B=h_v, ...)
  + convolve_frequencies(omega_static, E_F=0,
                         build_G_c, build_G_v, contract_chi,
                         kernel_sign=-1, ...)
)

# Dyson → PPM
for q in range(N_q):
    Π_0  = solve(I - v[q] @ χ_q[0, q], χ_q[0, q])
    Π_ip = solve(I - v[q] @ χ_q[1, q], χ_q[1, q])
    Ω[q] = ω_p * sqrt(Re(Π_ip / (Π_0 - Π_ip)))
    B[q] = -0.5 * Π_0 * Ω[q]
```

**Note on $\chi$ vs $\Sigma$ referencing:** For $\chi^0$, the "frequencies" $\omega$ are transfer frequencies that don't reference $E_F$. The positive-axis values are $E_c - E_v$ differences (not $E_c - E_F$). The cleanest approach: for $\chi^0$ calls, pass pre-computed positive-axis values $E_A = E_c^\text{vac}$ and $E_B = -E_v^\text{vac}$ directly (both $E_F$-independent), and `E_F = 0` (so $\omega_\text{rel} = \omega$, which is correct for transfer frequencies).

### 4.2 Phase 2: $\Sigma^c_k(\omega)$

```python
omega_grid = linspace(ω_min_vac, ω_max_vac, N_ω) + 0j
# spans below AND above E_F

# Conduction term Σ^(-): minus-kernel
Σ_cond = convolve_frequencies(
    omega_grid, E_F,
    build_G_cond, build_Π, contract_sigma,
    kernel_sign=+1, E_A_vac=E_c_vac, sign_A=+1, E_B=Ω_q, ξ=ξ, ...)

# Valence term Σ^(+): plus-kernel  
Σ_val = convolve_frequencies(
    omega_grid, E_F,
    build_G_val, build_Π, contract_sigma,
    kernel_sign=-1, E_A_vac=E_v_vac, sign_A=-1, E_B=Ω_q, ξ=ξ, ...)

# convolve_frequencies handles ω > E_F and ω < E_F internally:
#   ω > E_F: Σ_cond gets 3 windows (crossing), Σ_val gets single Laplace
#   ω < E_F: Σ_val gets 3 windows (crossing), Σ_cond gets single Laplace

Σ_c = Σ_cond + Σ_val
Σ_c = einsum('...μη, ...ηζ, ...ζν -> ...μν', v, Σ_c, v)
Σ = Σ_x + Σ_c
Σ_ij = einsum('kaμ, kaμbνω, kbν -> kijω', ψ.conj(), Σ, ψ)
```

The caller makes two calls (conduction + valence). `convolve_frequencies` splits each call's $\omega$ at $E_F$ and applies the correct windowing to each half internally.

---

## 5. Quadrature bounds

### 5.1 Single Laplace

$x_\text{min} = S_\text{min}$, $x_\text{max} = S_\text{max}$ (plus $|\omega_\text{rel}|_\text{max}$ for plus-kernel). $R = x_\text{max}/x_\text{min}$.

### 5.2 Crossing core

$A_\text{core} = 2T/\xi$ where $T = |\omega_\text{rel}|_\text{max} + c_\text{edge} \cdot \xi$.

### 5.3 Exterior Laplace

$x_\text{min} = \max(S_\text{min} - |\omega_\text{rel}|_\text{max},\; z_\text{edge})$, $x_\text{max} = S_\text{max}$.

---

## 6. Normalizations: derive from existing code

`convolve_frequencies` returns a raw quadrature sum. It does **not** include physics prefactors. Derive them by reading the existing `_get_chi_kernel()` and `chi_tile_gl()` in `w_isdf.py` / `chi_omega_quadrature.md`, identifying which factors are inside the minimax weights $\alpha$ (the $1/x$ approximation and gap exponential) versus external (the $-2$, spin normalization). Validate numerically against existing `get_static_chi_q_jax()`.

---

## 7. Mapping to existing code

| Existing | Replaced by |
|----------|-------------|
| `chi_tile_gl()` | `convolve_frequencies` with `contract_chi`, `kernel_sign=-1` |
| `chi_tile_hgl()` | `convolve_frequencies` with `contract_chi`, `kernel_sign=+1` |
| `precompute_exp_weights()` | Phase inside `build_propagator` |
| Window-pair loop | Internal window logic of `convolve_frequencies` |
| `precompute_phases()` | Frequency kernel inside `_evaluate_window` |

**Preserve unchanged:** ISDF fitting, Coulomb matrices, Dyson solve, exchange, band projection, $q=0$ head, sharding.

---

## 8. Array shapes

```
build_propagator  → (nk, n_s, n_μ, n_s, n_μ)
reshape           → (n_s, n_μ, n_s, n_μ, nkx, nky, nkz)
ifftn(-3,-2,-1)   → same, now R-space

PPM propagator    → (nq, n_μ, n_μ) → (n_μ, n_μ, nqx, nqy, nqz)
fftn(-3,-2,-1)    → same, now -R space

contract_sigma    → (n_s, n_μ, n_s, n_μ, nRx, nRy, nRz)    Π broadcasts over spin
contract_chi      → (n_μ, n_μ, nRx, nRy, nRz)               spin traced out

fftn(-3,-2,-1)    → same, now k/q-space
K_ω broadcast     → (n_ω, ..., nkx, nky, nkz)
```

---

## 9. PPM B-axis masks

$\Omega_{q,\mu\nu}$ is $(N_q, N_\mu, N_\mu)$. Mask is elementwise: `Ω_q[q,μ,ν] ∈ [B_lo, B_hi]`. Window decomposition of $\Pi^\text{PPM}$ is exact — masked-out elements contribute to other windows. $S_\text{min}$, $S_\text{max}$ are conservative global extrema over active masks.

---

## 10. Validation ladder

1. $\chi^0(0)$ against existing `get_static_chi_q_jax()` — catches normalization.
2. $\chi^0(\omega)$ at real frequencies against brute-force $O(N^4)$ — catches crossing routing.
3. PPM $\Omega, B$ against reference values — catches Dyson solve.
4. $\Sigma^c(\omega=E_F)$ plausibility against COHSEX — catches sign errors.
5. QP corrections for Si — end-to-end.

---

## 11. Checklist

| Component | Role |
|-----------|------|
| `laplace_minimax(R, ε)` | Non-crossing quadrature factory |
| `phase_minimax(A, ε)` | Crossing quadrature factory |
| `build_propagator(ψ, E_vac, mask, E_ref_vac, sign, t)` | Universal builder, accepts complex $t$ |
| `build_ppm_propagator(B, Ω, mask, Ω_ref, t)` | PPM variant (no spin) |
| `contract_chi`, `contract_sigma` | Two contractions |
| `convolve_frequencies(omega_vac, E_F, ...)` | Entry point: owns $\omega$-splitting, windowing, integration |

Normalizations, Coulomb sandwiching, exchange, band projection, head correction: all outside.
