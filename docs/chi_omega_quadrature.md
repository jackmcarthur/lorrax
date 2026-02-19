# χ(ω) via Complex-Time Shredded Propagator Method

**Reference**: Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020)

---

## Quick Reference

| Quantity | Formula |
|----------|---------|
| GL scale | $\zeta^{-1} = \sqrt{E^{(\mathrm{bw})} E^{(\mathrm{gap})}}$ |
| GL points | $N^{(\tau,\mathrm{GL})} = \alpha(0.4 - 0.3\ln\epsilon^{(q)})$, $\alpha = \sqrt{E^{(\mathrm{bw})}/E^{(\mathrm{gap})}}$ |
| HGL weight | $h(\tau) = \exp(-\tau - \tau^2/2)$ |
| HGL points | $N^{(\tau,\mathrm{HGL})} = c_2 x^2 + c_1 x + c_0$ where $x = \gamma E^{(\mathrm{bw})}$ |
| HGL coeffs | $c_2 = -0.0036\ln\epsilon + 0.11$, $c_1 = -0.0043(\ln\epsilon)^2 - 0.13\ln\epsilon + 0.54$, $c_0 = -0.204\ln\epsilon - 0.29$ |

---

## 1. Pole Structure of χ(ω)

The RPA polarizability has two poles per transition:
$$
\chi^0(\omega)_{rr'} = \sum_{cv\sigma\sigma'} \psi_{xc}\psi^*_{xv}\psi^*_{x'c}\psi_{x'v} \left[\underbrace{\frac{1}{\omega - \Delta E_{cv}}}_{\text{resonant}} - \underbrace{\frac{1}{\omega + \Delta E_{cv}}}_{\text{antiresonant}}\right]
$$
where $\Delta E_{cv} = E_c - E_v > 0$.

**Classification by sign of denominator:**

| Term | Denominator | Sign for $\omega > 0$ | Quadrature |
|------|-------------|----------------------|------------|
| Antiresonant | $\omega + \Delta E_{cv}$ | Always positive | GL |
| Resonant | $\omega - \Delta E_{cv}$ | Mixed if $\omega \in [E^{(\mathrm{gap})}, E^{(\mathrm{bw})}]$ | GL or HGL |

**Symmetry**: $\chi(\omega) = \chi(-\omega)^*$ — compute only $\omega \geq 0$, conjugate for $\omega < 0$.

---

## 2. Gauss-Laguerre (GL) — No Energy Crossing

When denominator $x$ has fixed sign in a window:
$$
\frac{1}{x} = \zeta \int_0^\infty d\tau\, e^{-\zeta x \tau} \approx \zeta \sum_u w_u e^{-\tau_u(\zeta x - 1)}
$$

**Optimal parameters** (Appendix A):
- Scale: $\zeta^{-1} = \sqrt{E^{(\mathrm{bw})} E^{(\mathrm{gap})}}$ equalizes error at window edges
- Grid size for max fractional error $\epsilon^{(q)}$:
$$
N^{(\tau,\mathrm{GL})} = \alpha(0.4 - 0.3\ln\epsilon^{(q)}), \quad \alpha = \sqrt{\frac{E^{(\mathrm{bw})}}{E^{(\mathrm{gap})}}}
$$

Error is symmetric about $\ln(\zeta\Delta) = 0$ and exactly zero at $\zeta\Delta = 1$.

---

## 3. Hermite-Gauss-Laguerre (HGL) — Energy Crossings

When $\omega$ lies within the transition range $[E^{(\mathrm{gap})}, E^{(\mathrm{bw})}]$, use regularized transform.

**Weight and transform** (Appendix C):
$$
h(\tau;\gamma) = \gamma\exp\left(-\tau - \frac{\tau^2}{2}\right)
$$
$$
F(x;\gamma) = \gamma\,\mathrm{Im}\left\{\sqrt{\frac{\pi}{2}} e^{-(x\gamma+i)^2/2}\left[1 + i\,\mathrm{erfi}\left(\frac{x\gamma+i}{\sqrt{2}}\right)\right]\right\}
$$

**Properties:**
- $F(x) \to 1/x + O(1/x^5)$ as $|x| \to \infty$ (faster than Lorentzian's $O(1/x^3)$)
- $F(0) = \gamma\sqrt{\pi/2}$ is finite
- Ratio 1/2 in exponent is special: any other ratio gives only $O(1/x^3)$

**Discrete form:**
$$
F(x;\gamma) \approx \gamma \sum_u w_u \sin(\tau_u x \gamma)
$$

**Grid size** (Appendix D): For scaled bandwidth $x = \gamma(E_{\max} - E_{\min})$:
$$
N^{(\tau,\mathrm{HGL})} = c_2(\epsilon) x^2 + c_1(\epsilon) x + c_0(\epsilon)
$$
with $c_2 = -0.0036\ln\epsilon + 0.11$, $c_1 = -0.0043(\ln\epsilon)^2 - 0.13\ln\epsilon + 0.54$, $c_0 = -0.204\ln\epsilon - 0.29$.

---

## 4. k-Space Structure with Spin

### Green's Function in k-Space

For spinor wavefunctions $\psi_{n\mathbf{k},a}(\mathbf{r})$ with spin indices $a,b \in \{\uparrow,\downarrow\}$:
$$
G_{\mathbf{k},ab}(\mathbf{r},\mathbf{r}';\omega) = \sum_n \frac{\psi_{n\mathbf{k},a}(\mathbf{r})\,\psi^*_{n\mathbf{k},b}(\mathbf{r}')}{\omega - E_{n\mathbf{k}}}
$$

### Lattice Fourier Transform to R-Space
$$
G_{\mathbf{R},ab}(\mathbf{r},\mathbf{r}';\omega) = \frac{1}{N_k}\sum_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{R}}\,G_{\mathbf{k},ab}
$$

Implementation: `jnp.fft.ifftn(..., norm='ortho')` on last 3 dims (kx,ky,kz).

### Polarizability as Spin-Traced G·G Product
$$
\chi^0_{\mathbf{R}}(\mathbf{r},\mathbf{r}';\omega) = -\sum_{ab} G_{\mathbf{R},ab}(\mathbf{r},\mathbf{r}')\, G_{-\mathbf{R},ba}(\mathbf{r}',\mathbf{r})
$$

Think of $(a,\mathbf{r})$ and $(b,\mathbf{r}')$ as combined indices. The trace contracts inner spin indices: $(ab) \times (ba) \to$ scalar per $(r,r')$.

---

## 5. CTSP Propagators with k and Spin

### Window-Pair Propagators

For a window pair $(l,m)$ at quadrature point $\tau_u$, define **k-space propagators** which are equivalent to the windowed Green's function G_k,ab(r,r',tau).

**Valence (occupied) propagator:**
$$
\rho^{(v)}_{\mathbf{k},ab}(\tau;\mathbf{r},\mathbf{r}') = \sum_{v \in \mathcal{L}} e^{-\zeta\tau(E_l^{(v,\max)} - E_{v\mathbf{k}})} \psi_{v\mathbf{k},a}(\mathbf{r})\psi^*_{v\mathbf{k},b}(\mathbf{r}')
$$

**Conduction (unoccupied) propagator:**
$$
\rho^{(c)}_{\mathbf{k},ab}(\tau;\mathbf{r},\mathbf{r}') = \sum_{c \in \mathcal{M}} e^{-\zeta\tau(E_{c\mathbf{k}} - E_m^{(c,\min)})} \psi_{c\mathbf{k},a}(\mathbf{r})\psi^*_{c\mathbf{k},b}(\mathbf{r}')
$$

**R-space via FFT:**
$$
\rho^{(v/c)}_{\mathbf{R},ab} = \mathrm{IFFT}_\mathbf{k}\left[\rho^{(v/c)}_{\mathbf{k},ab}\right]
$$

### Static ($\omega=0$) Contribution — GL Only

For non-crossing window pair:
$$
\chi^{0,\mathrm{static}}_{lm,\mathbf{R}}(\mathbf{r},\mathbf{r}') = -2\zeta_{lm} \sum_u w_u e^{-(\zeta_{lm} E_{lm}^{(\mathrm{gap})} - 1)\tau_u} \sum_{ab} \rho^{(c)}_{\mathbf{R},ab}(\tau_u) \, \rho^{(v)}_{-\mathbf{R},ba}(\tau_u)
$$

### Dynamic ($\omega \neq 0$) Contributions

**Antiresonant term** (always positive denominator → GL):
$$
\chi^{0,(-)}_{lm}(\omega) : \quad \text{shift } E^{(\mathrm{gap})} \to E^{(\mathrm{gap})} + \omega, \quad \text{use GL}
$$

**Resonant term** — depends on whether $\omega$ causes crossing:

*No crossing* ($\omega < E^{(\mathrm{gap})}_{lm}$ or $\omega > E^{(\mathrm{bw})}_{lm}$): Use GL with shifted gap.

*Crossing* ($E^{(\mathrm{gap})}_{lm} < \omega < E^{(\mathrm{bw})}_{lm}$): Use HGL with Euler identity.

---

## 6. HGL Separable Form via Euler Identity

When $\omega - \Delta E_{cv}$ changes sign, use **Euler's identity** to reduce memory by 2×.

### Complex Propagators (2 arrays instead of 4)

Instead of storing 4 sin/cos arrays, define 2 complex propagators per τ:
$$
G^{v}_{\mathbf{k},ab}(\tau) = \sum_{v \in \mathcal{L}} e^{i\gamma\tau E_{v\mathbf{k}}} \, \psi_{v\mathbf{k},a}\psi^*_{v\mathbf{k},b}
$$
$$
G^{c}_{\mathbf{k},ab}(\tau) = \sum_{c \in \mathcal{M}} e^{i\gamma\tau E_{c\mathbf{k}}} \, \psi_{c\mathbf{k},a}\psi^*_{c\mathbf{k},b}
$$

These satisfy $G = C + iS$ where $C = \sum_n \cos(\gamma\tau E_n)\psi_n\psi_n^\dagger$ and $S = \sum_n \sin(\gamma\tau E_n)\psi_n\psi_n^\dagger$.

### Products via Hermitian Conjugate

After FFT to R-space, form the product (single complex contraction):
$$
(G^c_{-\mathbf{R}})^\dagger \cdot G^v_{\mathbf{R}} = P_+ - i P_\times
$$

where:
- $P_+ = \mathrm{Re}[(G^c)^\dagger G^v] = C^c C^v + S^c S^v$
- $P_\times = -\mathrm{Im}[(G^c)^\dagger G^v] = S^c C^v - C^c S^v$

**Why it works**: For Hermitian matrices $\psi\psi^\dagger$, the conjugate equals the Hermitian transpose.

### Batch All Frequencies

The crossing contribution for **all** ω simultaneously:
$$
\chi^{\mathrm{cross}}_{lm}(\omega_i) = -\gamma \sum_u w_u \left[\cos(\gamma\tau_u\omega_i)\,P_\times(\tau_u) - \sin(\gamma\tau_u\omega_i)\,P_+(\tau_u)\right]
$$

**Memory savings**: 2 complex arrays instead of 4 real → **2× reduction** in peak memory.

---

## 7. Pole Symmetries and Efficiency

### Avoiding Redundant Work

1. **Frequency symmetry**: $\chi(-\omega) = \chi(\omega)^*$ — only compute $\omega \geq 0$

2. **Antiresonant term**: $1/(\omega + \Delta E)$ is always positive for $\omega > 0$ — always GL, no crossings possible

3. **Reuse propagators across frequencies**: The $\omega$-independent parts of propagators (conduction $\mathcal{S}^{(c)}, \mathcal{C}^{(c)}$) can be precomputed once per window pair and reused across all $\omega_i$

4. **R-space symmetry**: $\chi_{\mathbf{R}}(\omega) = \chi_{-\mathbf{R}}(\omega)^*$ from Hermiticity — only compute half of R-points

5. **Window classification is $\omega$-dependent**: For a set of frequencies $\{\omega_i\}$, precompute which windows are crossing/non-crossing for each $\omega_i$

---

## 8. Algorithm: JAX-Optimized Multi-Frequency Evaluation

**Design principles for JAX:**
1. Window pair loop is **outermost** — determines static array shapes
2. **Batch all frequencies** at each τ via multiplicative phase factors
3. Propagator products are ω-independent; ω enters only through phases

### 8.1 Frequency Batching via Phase Factors

**GL case** (no crossing): The ω-dependence is $e^{-\zeta\omega\tau}$
$$
\chi^{\mathrm{GL}}(\omega_i) = -2\zeta \sum_u w_u \underbrace{e^{(\zeta E^{(\mathrm{gap})} - 1)\tau_u}}_{\text{ω-indep}} \cdot \underbrace{e^{-\zeta\omega_i\tau_u}}_{\text{phase}[\omega_i, u]} \cdot \underbrace{\tilde{\chi}(\tau_u)}_{\text{tile}}
$$

**HGL case** (crossing): Use Euler identity for 2× memory savings:
$$
G^{v/c}_\tau = \sum_n e^{i\gamma\tau E_n}\psi_n\psi_n^\dagger
$$

Then form products via Hermitian conjugate:
$$
(G^c_\tau)^\dagger G^v_\tau = P_+(\tau) - i P_\times(\tau)
$$

Extract real/imag and batch all crossing frequencies:
$$
\chi^{\mathrm{HGL}}(\omega_i) = -\gamma \sum_u w_u \left[\cos(\gamma\tau_u\omega_i)\,P_\times(\tau_u) - \sin(\gamma\tau_u\omega_i)\,P_+(\tau_u)\right]
$$

### 8.2 Loop Structure (JAX-friendly)

```python
# chi_accum: shape (n_omega, nq, nmu, nmu) — accumulator

FOR (l,m) in window_pairs:  # OUTERMOST: fixes array shapes
    
    # Slice bands/energies for this window (static shapes within loop)
    E_v, E_c, psi_v, psi_c = slice_window(l, m)
    ζ = 1/sqrt(E_bw * E_gap)
    τ, w = roots_laguerre(N_tau)  # or HGL nodes
    
    # ===== GL (non-crossing) =====
    # Precompute ω-independent gap factor: shape (n_tau,)
    gap_factor = w * exp(-(ζ * E_gap - 1) * τ)
    
    # Precompute ω-dependent phase matrix: shape (n_omega, n_tau)
    phase_GL = exp(-ζ * ω_array[:, None] * τ[None, :])
    
    FOR u in range(n_tau):
        # Build propagators at τ_u (ω-independent)
        ρ_v = sum_v exp(-ζ*τ[u]*(E_v_max - E_v)) * psi_v @ psi_v.H
        ρ_c = sum_c exp(-ζ*τ[u]*(E_c - E_c_min)) * psi_c @ psi_c.H
        
        # FFT k→R, contract, FFT R→q
        χ_tile = FFT_Rq(contract(IFFT_kR(ρ_c), IFFT_kR(ρ_v)))
        
        # Accumulate ALL ω at once (vectorized over omega axis)
        chi_accum += (-2*ζ * gap_factor[u] * phase_GL[:, u])[:, None, None, None] * χ_tile
    
    # ===== HGL (crossing) =====
    IF any ω causes crossing in this window:
        τ_hgl, w_hgl = hgl_nodes(N_hgl)
        
        # Precompute sin/cos phase matrix: shape (n_cross_omega, n_tau)
        sin_phase = sin(γ * ω_cross[:, None] * τ_hgl[None, :])
        cos_phase = cos(γ * ω_cross[:, None] * τ_hgl[None, :])
        
        FOR u in range(n_tau_hgl):
            # 2 complex propagators via Euler (2× memory savings)
            G_v = sum_v exp(i*γ*τ[u]*E_v) * psi_v @ psi_v.H
            G_c = sum_c exp(i*γ*τ[u]*E_c) * psi_c @ psi_c.H
            
            # FFT to R-space
            G_v_R = IFFT_kR(G_v)
            G_c_mR = FFT_kR(G_c)  # -R for convolution
            
            # Products from Hermitian conjugate: (G_c)† G_v = P_+ - i P_×
            product = contract(conj(G_c_mR), G_v_R)
            P_plus_q  = FFT_Rq(product.real)
            P_cross_q = FFT_Rq(-product.imag)
            
            # Batch ALL crossing ω (vectorized multiply-add)
            chi_cross += γ * w_hgl[u] * (
                cos_phase[:, u, None, None, None] * P_cross_q -
                sin_phase[:, u, None, None, None] * P_plus_q
            )

# Finalize
chi_accum *= -2 / (sqrt(N_k) * n_spin * n_spinor)
write_to_file(chi_accum)
```

### 8.3 Memory Layout

| Array | Shape | Notes |
|-------|-------|-------|
| `chi_accum` | `(n_ω, nq, nμ, nμ)` | Main accumulator |
| `phase_GL` | `(n_ω, n_τ)` | Precomputed, reused per window |
| `sin/cos_phase` | `(n_ω_cross, n_τ_hgl)` | Only for crossing ω |
| `χ_tile` | `(nq, nμ, nμ)` | Temporary per τ, overwritten |
| `G_v, G_c` | `(k, s, μ, s', μ')` complex | 2 arrays via Euler (not 4 real) |
| `P_plus, P_cross` | `(nq, nμ, nμ)` | Extracted from `(G^c)† G^v` |

### 8.4 JAX Implementation Notes

1. **Static shapes**: Window loop is Python `for`; inner τ loop can be `jax.lax.fori_loop` with fixed shapes

2. **Vectorized ω update**: The key line is:
   ```python
   chi_accum += phase[:, None, None, None] * tile[None, ...]
   ```
   which broadcasts `(n_ω,)` against `(nq, nμ, nμ)` without explicit loop

3. **Streaming to disk**: For very large n_ω, split frequencies into chunks and write each chunk after completing all window pairs

4. **Sharding**: For ISDF, μ dimension is sharded; phase multiply is embarrassingly parallel over ω

### 8.5 Symmetry Shortcuts

| Symmetry | Savings |
|----------|---------|
| $\chi(-\omega) = \chi(\omega)^*$ | Compute only $\omega \geq 0$ |
| $\chi_{-\mathbf{R}} = \chi_{\mathbf{R}}^*$ | Compute half of R-points |
| Antiresonant always GL | No HGL overhead for that term |
| $C^{v/c}, S^{v/c}$ reused across ω | Build once per τ |

---

## 9. Complex Frequencies: z = ω + iη

For contour integration or analytic continuation, we need χ at complex frequencies $z = \omega + i\eta$.

### 9.1 Why It Works

The Laplace identity $1/s = \int_0^\infty d\tau\, e^{-s\tau}$ converges when $\mathrm{Re}(s) > 0$. For $s = z - \Delta E$:
- Convergence depends on $\mathrm{Re}(s) = \omega - \Delta E$, **not** on $\eta$
- The imaginary part just adds an oscillatory phase $e^{-i\eta\tau}$
- Poles are at $z = \Delta E$ (on real axis) — if $\eta \neq 0$, you're never on a pole

### 9.2 GL with Complex z

**No changes to quadrature nodes/weights needed.** Just use complex phase:

$$
\text{phase}_{\mathrm{GL}}[z, u] = e^{-\zeta(z - E^{(\mathrm{gap})})\tau_u + \tau_u} = e^{-\zeta(\omega + i\eta - E^{(\mathrm{gap})})\tau_u + \tau_u}
$$

```python
# z_array: shape (n_z,) complex, z = ω + iη
# τ: shape (n_tau,) real quadrature nodes
phase_GL = jnp.exp(-ζ * (z_array[:, None] - E_gap) * τ[None, :] + τ[None, :])
# shape: (n_z, n_tau), dtype: complex128
```

The rest of the GL algorithm is unchanged — propagator products are real, phase is complex, result is complex.

### 9.3 HGL with Complex z

For crossings, $\sin(\gamma\tau x)$ with complex $x = (\omega - \Delta E) + i\eta$:

$$
\sin(\gamma\tau(x_r + ix_i)) = \sin(\gamma\tau x_r)\cosh(\gamma\tau\eta) + i\cos(\gamma\tau x_r)\sinh(\gamma\tau\eta)
$$

```python
# Complex sin/cos for HGL phases
def complex_sincos(γτ, ω, η):
    """Returns sin(γτ(ω + iη)), cos(γτ(ω + iη))"""
    sin_r, cos_r = jnp.sin(γτ * ω), jnp.cos(γτ * ω)
    sinh_i, cosh_i = jnp.sinh(γτ * η), jnp.cosh(γτ * η)
    sin_z = sin_r * cosh_i + 1j * cos_r * sinh_i
    cos_z = cos_r * cosh_i - 1j * sin_r * sinh_i
    return sin_z, cos_z

# Then use in HGL accumulation:
sin_phase, cos_phase = complex_sincos(γ * τ_hgl[None, :], ω[:, None], η[:, None])
# P_plus, P_cross are still real; result is complex
chi_cross = γ * jnp.einsum('zu,u...->z...', sin_phase, w_hgl * P_plus) + \
            γ * jnp.einsum('zu,u...->z...', cos_phase, w_hgl * P_cross)
```

### 9.4 When to Use GL vs HGL for Complex z

| Condition | Method | Reason |
|-----------|--------|--------|
| $\|\eta\| \gg E^{(\mathrm{bw})}$ | GL everywhere | Far from real axis, no near-poles |
| $\omega \notin [E^{(\mathrm{gap})}, E^{(\mathrm{bw})}]$ | GL | No crossing even on real axis |
| $\omega \in [E^{(\mathrm{gap})}, E^{(\mathrm{bw})}]$, $\|\eta\|$ small | HGL | Near real-axis crossing region |

**Rule of thumb**: If $\|\eta\| > \gamma^{-1}$ (the HGL regularization width), you can use GL even for "crossing" windows.

### 9.5 Implementation Notes

1. **Batch complex frequencies**: Works exactly like real — just make phase arrays complex
   ```python
   z_array = ω_array + 1j * η_array  # shape (n_z,)
   phase = jnp.exp(-ζ * z_array[:, None] * τ[None, :])  # complex (n_z, n_tau)
   ```

2. **Output is complex**: $\chi(z)$ has both real and imaginary parts
   - Real part: related to absorption/loss
   - Imaginary part: related to dispersion/screening

3. **Hermitian symmetry**: $\chi(z^*) = \chi(z)^*$ — compute only upper half-plane if needed

4. **Contour choice**: For avoiding poles, typical choices:
   - Parallel line: $z = \omega + i\eta_0$ with fixed $\eta_0 > 0$
   - Matsubara: $z = i\omega_n$ with $\omega_n = (2n+1)\pi/\beta$ (imaginary axis only)

5. **Numerical stability**: For large $|\gamma\tau\eta|$, $\cosh/\sinh$ can overflow
   ```python
   # Safe complex sin for large |η|
   def safe_complex_sin(γτ, ω, η):
       x = γτ * η
       if jnp.max(jnp.abs(x)) > 20:  # threshold for overflow
           # Use exp form: sin(a+ib) = (e^{i(a+ib)} - e^{-i(a+ib)}) / 2i
           eplus = jnp.exp(1j * γτ * ω - x)
           eminus = jnp.exp(-1j * γτ * ω + x)
           return (eplus - eminus) / 2j
       else:
           return jnp.sin(γτ * ω) * jnp.cosh(x) + 1j * jnp.cos(γτ * ω) * jnp.sinh(x)
   ```

6. **Memory**: Complex arrays are 2× the size of real — plan accordingly for large $n_z$

---

## 10. HGL Quadrature: Golub-Welsch Construction

From Appendix H, nodes $\{\tau_u\}$ and weights $\{w_u\}$ for weight $h(\tau) = e^{-\tau-\tau^2/2}$:

1. **Compute moments** via GL quadrature on $e^{-\tau}$:
   $$\mu_k = \int_0^\infty \tau^k e^{-\tau - \tau^2/2} d\tau \approx \sum_j w_j^{(\mathrm{GL})} e^{-x_j^2/2} x_j^k$$

2. **Build orthogonal polynomials** $p_n(\tau)$ via three-term recurrence:
   $$p_{n+1}(\tau) = (\tau - a_n)p_n(\tau) - b_n p_{n-1}(\tau)$$
   where $a_n = \langle \tau p_n^2 \rangle / \langle p_n^2 \rangle$, $b_n = \langle p_n^2 \rangle / \langle p_{n-1}^2 \rangle$

3. **Form Jacobi matrix**: $J = \mathrm{tridiag}(\sqrt{b_2}, \ldots; a_1, a_2, \ldots; \sqrt{b_2}, \ldots)$

4. **Diagonalize**: Eigenvalues of $J_{1:n,1:n}$ give nodes $\tau_u$; weights from first eigenvector components: $w_u = \mu_0 \cdot v_1^2(u)$

See `docs/Kim-2020-CTSP-appendix.md` for MATLAB implementation.

---

## Appendix A: Derivation Sketch

**Why the 1/2 ratio in $h(\tau) = e^{-\tau - \tau^2/2}$?**

The transform $F(x) = \int_0^\infty h(\tau) e^{ix\tau} d\tau$ expands as:
$$
F(x) \sim \frac{1}{ix} + \frac{a_1}{(ix)^2} + \frac{a_2}{(ix)^3} + \cdots
$$

The coefficient $a_1$ vanishes when $\int_0^\infty \tau h(\tau) d\tau / \int_0^\infty h(\tau) d\tau = 1$, which is satisfied exactly when the linear/quadratic ratio is 1/2. This gives $F(x) = 1/x + O(1/x^5)$ instead of $O(1/x^3)$.

---

## Index of Symbols

| Symbol | Definition |
|--------|------------|
| $z = \omega + i\eta$ | Complex frequency (real part ω, imaginary part η) |
| $\zeta_{lm}$ | Energy scale for window pair $(l,m)$: $\zeta^{-1} = \sqrt{E^{(\mathrm{bw})} E^{(\mathrm{gap})}}$ |
| $\gamma$ | HGL broadening parameter (= $\zeta$ for crossing windows) |
| $\epsilon^{(q)}$ | Target fractional quadrature error |
| $\alpha$ | Bandwidth ratio $\sqrt{E^{(\mathrm{bw})}/E^{(\mathrm{gap})}}$ |
| $\mathcal{L}, \mathcal{M}$ | Index sets for valence/conduction bands in windows |
| $\rho^{(v/c)}$ | Exponentially-damped density matrix (GL): $\sum_n e^{-\zeta\tau\Delta E}\psi_n\psi_n^\dagger$ |
| $G^{v/c}$ | Complex propagator (HGL, Euler): $\sum_n e^{i\gamma\tau E_n}\psi_n\psi_n^\dagger$ |
| $P_+, P_\times$ | Products from $(G^c)^\dagger G^v$: $P_+ = \mathrm{Re}[\cdot]$, $P_\times = -\mathrm{Im}[\cdot]$ |
| $E^{(\mathrm{gap})}_{lm}$ | $E_m^{(c,\min)} - E_l^{(v,\max)}$ — minimum transition energy |
| $E^{(\mathrm{bw})}_{lm}$ | $E_m^{(c,\max)} - E_l^{(v,\min)}$ — maximum transition energy |
| `phase_GL` | Complex phase $e^{-\zeta z\tau}$, shape `(n_z, n_τ)`, dtype `complex128` |
| `sin/cos_phase` | Complex $\sin/\cos(\gamma\tau z)$, shape `(n_z, n_τ)`, dtype `complex128` |

---

## 11. Implementation: Function Decomposition

### 11.1 Design Principles

1. **Outer loop = window pairs** — fixes static array shapes for JIT
2. **Inner loop = single τ** — one tile in memory at a time
3. **Two wavefunction copies** — `psi_X` and `psi_Y` transposed for aligned matmul
4. **Precompute everything possible** — exp weights and phases before any τ loop
5. **Minimal function signatures** — bundle related data into dataclasses

### 11.2 Memory Layout for Performance

**Wavefunction copies** (for aligned G = ψ ψ†):
```
psi_X: (nk, ns, nμ, nb)  — band axis fastest, for Σ_n
psi_Y: (nk, nb, ns, nμ)  — nμ axis fastest, for outer product
```

**FFT layout** — FFT axes should be contiguous (last in C-order):
```
G_k:   (nk, ns, nμ, ns, nμ)  →  reshape to (ns, nμ, ns, nμ, nkx, nky, nkz)
       FFT on axes=(-3,-2,-1) keeps k-grid contiguous
G_R:   (ns, nμ, ns, nμ, Rx, Ry, Rz)  — R contiguous for spin contraction
χ_R:   (nμ, nμ, Rx, Ry, Rz)  — R contiguous for final FFT
```

### 11.3 Data Structures

```python
@dataclass
class WavefunctionPair:
    """Wavefunctions in two layouts for efficient G = ψ ψ† matmul."""
    psi_X: Array   # (nk, ns, nμ, nb) — nb fastest, for band sum
    psi_Y: Array   # (nk, nb, ns, nμ) — nμ fastest, for outer product  
    E: Array       # (nk, nb) band energies
    mask: Array    # (nk, nb) bool, True for valid (unpadded) bands


@dataclass
class WindowQuadrature:
    """Everything needed to evaluate one window pair."""
    # Window bounds
    E_gap: float
    E_bw: float
    zeta: float               # = 1/√(E_gap × E_bw)
    
    # GL quadrature (precomputed at window creation)
    tau_gl: Array             # (n_tau_gl,)
    w_gl: Array               # (n_tau_gl,)
    gap_factor_gl: Array      # (n_tau_gl,) = w × exp(τ)
    
    # HGL quadrature (only if any ω causes crossing)
    tau_hgl: Array | None     # (n_tau_hgl,)
    w_hgl: Array | None       # (n_tau_hgl,)
    gamma: float | None


@dataclass 
class FrequencyPhases:
    """Precomputed ω-dependent phases for one window."""
    # GL phases
    phase_gl: Array           # (n_omega_gl, n_tau_gl) complex
    omega_gl_idx: Array       # indices into full omega grid
    
    # HGL phases  
    sin_phase: Array | None   # (n_omega_hgl, n_tau_hgl) complex
    cos_phase: Array | None   # (n_omega_hgl, n_tau_hgl) complex
    omega_hgl_idx: Array | None


@dataclass
class IntegrationContext:
    """Minimal context passed to all integration functions."""
    nkx: int
    nky: int  
    nkz: int
    prefactor: float          # -2/(√Nk × nspin × nspinor)
    output_file: Path
```

### 11.4 Top-Level Driver

```python
def shred_integrate(
    wfn_v: WavefunctionPair,      # valence (both layouts + E + mask)
    wfn_c: WavefunctionPair,      # conduction
    omega_grid: Array,            # (n_ω,) complex ok
    windows: list[WindowQuadrature],
    ctx: IntegrationContext,
) -> None:
    """
    Compute χ(ω) for all frequencies via CTSP.
    
    Iterates over window pairs, accumulates to disk.
    Grad-student readable: one loop, two function calls per window.
    """
    init_output_file(ctx.output_file, omega_grid, ctx)
    
    for win in windows:
        # 1. Slice wavefunctions to this window (Python, variable shapes)
        wfn_v_win = slice_to_window(wfn_v, win.val_bounds)
        wfn_c_win = slice_to_window(wfn_c, win.cond_bounds)
        
        # 2. Precompute all exp weights for this window (once)
        exp_v = precompute_exp_weights(wfn_v_win.E, win, is_valence=True)
        exp_c = precompute_exp_weights(wfn_c_win.E, win, is_valence=False)
        
        # 3. Precompute ω-dependent phases
        phases = precompute_phases(omega_grid, win)
        
        # 4. GL contributions (all non-crossing ω)
        if phases.omega_gl_idx.size > 0:
            integrate_gl(wfn_v_win, wfn_c_win, exp_v, exp_c, phases, win, ctx)
        
        # 5. HGL contributions (crossing ω only)
        if phases.omega_hgl_idx is not None and phases.omega_hgl_idx.size > 0:
            integrate_hgl(wfn_v_win, wfn_c_win, phases, win, ctx)
    
    finalize_output(ctx)
```

### 11.5 GL Integration (Simple Case)

```python
def integrate_gl(
    wfn_v: WavefunctionPair,
    wfn_c: WavefunctionPair,
    exp_v: Array,                 # (n_tau, nk, nv) precomputed
    exp_c: Array,                 # (n_tau, nk, nc) precomputed
    phases: FrequencyPhases,
    win: WindowQuadrature,
    ctx: IntegrationContext,
) -> None:
    """
    Accumulate GL contributions for one window.
    
    Key insight: exp weights and phases are precomputed.
    Inner loop just calls JIT kernel and writes to disk.
    """
    for u in range(len(win.tau_gl)):
        # JIT kernel: compute χ_q(τ_u)
        chi_q = chi_tile_gl(
            wfn_v.psi_X, wfn_v.psi_Y, exp_v[u], wfn_v.mask,
            wfn_c.psi_X, wfn_c.psi_Y, exp_c[u], wfn_c.mask,
            ctx.nkx, ctx.nky, ctx.nkz
        )
        
        # Accumulate to all GL frequencies at once
        weights = ctx.prefactor * win.zeta * win.gap_factor_gl[u] * phases.phase_gl[:, u]
        accumulate_tile(ctx.output_file, phases.omega_gl_idx, weights, chi_q)
```

### 11.6 HGL Integration (Crossing Case)

```python
def integrate_hgl(
    wfn_v: WavefunctionPair,
    wfn_c: WavefunctionPair,
    phases: FrequencyPhases,
    win: WindowQuadrature,
    ctx: IntegrationContext,
) -> None:
    """
    Accumulate HGL contributions for crossing ω.
    
    Builds 4 sin/cos propagators per τ → 2 product tiles.
    """
    for u in range(len(win.tau_hgl)):
        tau_u = win.tau_hgl[u]
        
        # JIT kernel: compute P_+(τ), P_×(τ)
        P_plus, P_cross = chi_tile_hgl(
            wfn_v.psi_X, wfn_v.psi_Y, wfn_v.E, wfn_v.mask,
            wfn_c.psi_X, wfn_c.psi_Y, wfn_c.E, wfn_c.mask,
            tau_u, win.gamma,
            ctx.nkx, ctx.nky, ctx.nkz
        )
        
        # Accumulate: cos_phase × P_× - sin_phase × P_+
        w_cross = ctx.prefactor * win.gamma * win.w_hgl[u] * phases.cos_phase[:, u]
        w_plus = -ctx.prefactor * win.gamma * win.w_hgl[u] * phases.sin_phase[:, u]
        accumulate_tiles(ctx.output_file, phases.omega_hgl_idx, w_plus, P_plus, w_cross, P_cross)
```

### 11.7 JIT Kernels (Optimized Memory Layout)

```python
@partial(jax.jit, static_argnames=('nkx', 'nky', 'nkz'))
def chi_tile_gl(
    psi_v_X, psi_v_Y, exp_v, mask_v,    # valence: X=(nk,ns,nμ,nv), Y=(nk,nv,ns,nμ)
    psi_c_X, psi_c_Y, exp_c, mask_c,    # conduction: same layout
    nkx, nky, nkz,
) -> Array:
    """
    Compute χ_q(τ) for one GL quadrature point.
    Returns (nμ, nμ, nqx, nqy, nqz) — q-grid contiguous for disk write.
    """
    # Zero padded bands
    exp_v = jnp.where(mask_v, exp_v, 0.0)
    exp_c = jnp.where(mask_c, exp_c, 0.0)
    
    # Build propagators using aligned layouts
    # G_v[k,a,μ,b,ν] = Σ_n exp[k,n] × psi_X†[k,a,μ,n] × psi_Y[k,n,b,ν]
    G_v_k = jnp.einsum('kamn,kn,knbv->kambv', psi_v_X.conj(), exp_v, psi_v_Y)
    G_c_k = jnp.einsum('kamn,kn,knbv->kambv', psi_c_X.conj(), exp_c, psi_c_Y)
    
    # Transpose for FFT: (ns, nμ, ns, nμ, nkx, nky, nkz) — k-grid last/contiguous
    G_v_k = G_v_k.reshape(nkx, nky, nkz, *G_v_k.shape[1:])
    G_v_k = G_v_k.transpose(3, 4, 5, 6, 0, 1, 2)  # (a,μ,b,ν,kx,ky,kz)
    G_c_k = G_c_k.reshape(nkx, nky, nkz, *G_c_k.shape[1:])
    G_c_k = G_c_k.transpose(3, 4, 5, 6, 0, 1, 2)
    
    # FFT k → R (axes contiguous)
    G_v_R = jnp.fft.ifftn(G_v_k, axes=(-3, -2, -1), norm='ortho')
    G_c_mR = jnp.fft.fftn(G_c_k, axes=(-3, -2, -1), norm='ortho')  # -R
    
    # Spin trace: χ[μ,ν,R] = Σ_ab G_c[a,μ,b,ν,R] × G_v[b,ν,a,μ,-R]
    # Note: G_v[-R] = conj(G_v[R]) by Hermiticity, handled by index swap
    chi_R = jnp.einsum('ambvxyz,bvamxyz->mvxyz', G_c_mR, G_v_R)
    
    # FFT R → q
    chi_q = jnp.fft.fftn(chi_R, axes=(-3, -2, -1), norm='ortho')
    
    return chi_q


@partial(jax.jit, static_argnames=('nkx', 'nky', 'nkz'))
def chi_tile_hgl(
    psi_v_X, psi_v_Y, E_v, mask_v,
    psi_c_X, psi_c_Y, E_c, mask_c,
    tau, gamma,
    nkx, nky, nkz,
) -> tuple[Array, Array]:
    """
    Compute P_+(τ), P_×(τ) for one HGL quadrature point using Euler identity.
    Returns two (nμ, nμ, nqx, nqy, nqz) real tiles.
    
    Memory: 2 complex arrays (G_v, G_c) instead of 4 real (S_v, C_v, S_c, C_c).
    """
    # Complex phase weights: exp(iγτE)
    phase_v = jnp.where(mask_v, jnp.exp(1j * gamma * tau * E_v), 0.0)
    phase_c = jnp.where(mask_c, jnp.exp(1j * gamma * tau * E_c), 0.0)
    
    # 2 complex propagators (instead of 4 real)
    G_v_k = jnp.einsum('kamn,kn,knbv->kambv', psi_v_X.conj(), phase_v, psi_v_Y)
    G_c_k = jnp.einsum('kamn,kn,knbv->kambv', psi_c_X.conj(), phase_c, psi_c_Y)
    
    # FFT to R-space (transpose for contiguous FFT axes)
    def to_R(G_k):
        G = G_k.reshape(nkx, nky, nkz, *G_k.shape[1:])
        G = G.transpose(3, 4, 5, 6, 0, 1, 2)
        return jnp.fft.ifftn(G, axes=(-3, -2, -1), norm='ortho')
    
    def to_mR(G_k):
        G = G_k.reshape(nkx, nky, nkz, *G_k.shape[1:])
        G = G.transpose(3, 4, 5, 6, 0, 1, 2)
        return jnp.fft.fftn(G, axes=(-3, -2, -1), norm='ortho')
    
    G_v_R = to_R(G_v_k)
    G_c_mR = to_mR(G_c_k)
    
    # Hermitian product: (G_c)† @ G_v = P_+ - i P_×
    # For Hermitian propagators, conj = Hermitian transpose
    def contract(A, B):
        return jnp.einsum('ambvxyz,bvamxyz->mvxyz', A, B)
    
    product_R = contract(G_c_mR.conj(), G_v_R)
    
    # Extract P_+ and P_× from complex product
    P_plus_R = product_R.real
    P_cross_R = -product_R.imag
    
    # FFT to q
    P_plus_q = jnp.fft.fftn(P_plus_R, axes=(-3, -2, -1), norm='ortho')
    P_cross_q = jnp.fft.fftn(P_cross_R, axes=(-3, -2, -1), norm='ortho')
    
    return P_plus_q, P_cross_q
```

### 11.8 Precomputation (Called Once Per Window)

```python
def precompute_exp_weights(E: Array, win: WindowQuadrature, is_valence: bool) -> Array:
    """
    Precompute exp(-ζτΔE) for all τ. Called once per window.
    Returns (n_tau_gl, nk, nb).
    """
    if is_valence:
        E_ref = jnp.max(E)  # E_v_max
        delta_E = E_ref - E  # positive
    else:
        E_ref = jnp.min(E)  # E_c_min  
        delta_E = E - E_ref  # positive
    
    return jnp.exp(-win.zeta * win.tau_gl[:, None, None] * delta_E[None, :, :])


def precompute_phases(omega: Array, win: WindowQuadrature) -> FrequencyPhases:
    """
    Classify ω and precompute all phases. Called once per window.
    """
    # Classify: GL if ω outside [E_gap, E_bw], HGL if inside
    is_gl = (omega.real < win.E_gap) | (omega.real > win.E_bw)
    is_hgl = ~is_gl
    
    gl_idx = jnp.where(is_gl)[0]
    hgl_idx = jnp.where(is_hgl)[0] if is_hgl.any() else None
    
    # GL phases: exp(-ζ(ω - E_gap)τ)
    omega_gl = omega[is_gl]
    phase_gl = jnp.exp(-win.zeta * (omega_gl[:, None] - win.E_gap) * win.tau_gl[None, :])
    
    # HGL phases: sin/cos(γτω)
    if hgl_idx is not None:
        omega_hgl = omega[is_hgl]
        sin_phase, cos_phase = complex_sincos(win.gamma * omega_hgl[:, None] * win.tau_hgl[None, :])
    else:
        sin_phase = cos_phase = None
    
    return FrequencyPhases(phase_gl, gl_idx, sin_phase, cos_phase, hgl_idx)


def complex_sincos(z: Array) -> tuple[Array, Array]:
    """sin(z), cos(z) for complex z."""
    if jnp.iscomplexobj(z):
        sin_z = jnp.sin(z.real) * jnp.cosh(z.imag) + 1j * jnp.cos(z.real) * jnp.sinh(z.imag)
        cos_z = jnp.cos(z.real) * jnp.cosh(z.imag) - 1j * jnp.sin(z.real) * jnp.sinh(z.imag)
    else:
        sin_z, cos_z = jnp.sin(z), jnp.cos(z)
    return sin_z, cos_z
```

### 11.9 I/O (Append-Only)

```python
def accumulate_tile(path: Path, omega_idx: Array, weights: Array, tile: Array) -> None:
    """
    Atomic accumulate: χ[ω_i] += weights[i] × tile
    
    tile: (nμ, nμ, nqx, nqy, nqz) → reshape to (nq, nμ, nμ) for storage
    """
    tile_flat = tile.transpose(2, 3, 4, 0, 1).reshape(-1, tile.shape[0], tile.shape[1])
    
    with h5py.File(path, 'r+') as f:
        ds = f['chi_omega']
        for i, idx in enumerate(omega_idx):
            ds[idx] += weights[i] * tile_flat


def accumulate_tiles(path, omega_idx, w1, tile1, w2, tile2) -> None:
    """Accumulate two weighted tiles (for HGL P_+ and P_×)."""
    t1 = tile1.transpose(2, 3, 4, 0, 1).reshape(-1, tile1.shape[0], tile1.shape[1])
    t2 = tile2.transpose(2, 3, 4, 0, 1).reshape(-1, tile2.shape[0], tile2.shape[1])
    
    with h5py.File(path, 'r+') as f:
        ds = f['chi_omega']
        for i, idx in enumerate(omega_idx):
            ds[idx] += w1[i] * t1 + w2[i] * t2
```

### 11.10 Memory Budget

| Array | Shape | Size | Lifetime |
|-------|-------|------|----------|
| `wfn_v.psi_X` | `(nk, ns, nμ, nv)` | ~50 MB | Entire run |
| `wfn_v.psi_Y` | `(nk, nv, ns, nμ)` | ~50 MB | Entire run |
| `exp_v` | `(n_τ, nk, nv)` | ~1 MB | Per window |
| `phases.phase_gl` | `(n_ω, n_τ)` | ~100 KB | Per window |
| `chi_tile` | `(nμ, nμ, nqx, nqy, nqz)` | ~10 MB | Per τ (overwritten) |

**Total per-τ footprint**: ~120 MB (wfns) + 10 MB (tile) = manageable on GPU

### 11.11 HGL Memory: Euler vs Naive

The Euler identity is the **default approach** (used in `chi_tile_hgl` above):
$$
G^{v/c} = \sum_n e^{i\gamma\tau E_n}\, \psi_n \psi_n^\dagger = C + iS
$$

Products via single complex contraction:
$$
(G^c)^\dagger \cdot G^v = P_+ - i P_\times
$$

| Approach | Arrays per τ | Peak memory |
|----------|--------------|-------------|
| **Euler (default)** | 2 complex ($G^v, G^c$) | 2× (μ,μ,k) |
| Naive (4 real) | 4 real ($S^v, C^v, S^c, C^c$) | 4× (μ,μ,k) |
| **Euler (default)** | 2 complex ($G^v, G^c$) | **2× (μ,μ,k)** |

**Euler is the default** — see `chi_tile_hgl` implementation above.

---

## References

- Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020) — CTSP derivation, Appendices A–H
- Golub & Welsch, Math. Comp. 23, 221 (1969) — Gaussian quadrature from moments
- `docs/Kim-2020-CTSP-appendix.md` — Full appendix with MATLAB code
