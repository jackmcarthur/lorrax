# Minimax Quadrature for GW Frequency Integration

**Reference**: Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020)

Solver implementation: `src/common/minimax.py`

Runtime reuse layer: `src/gw/minimax_screening.py`

---

## Quick Reference

| Quantity | Formula |
|----------|---------|
| GL scale | $\zeta^{-1} = \sqrt{E^{(\mathrm{bw})} E^{(\mathrm{gap})}}$ |
| GL points | $N^{(\tau,\mathrm{GL})} = \alpha(0.4 - 0.3\ln\epsilon^{(q)})$, $\alpha = \sqrt{E^{(\mathrm{bw})}/E^{(\mathrm{gap})}}$ |
| Noncrossing error | $\epsilon(N, R) \approx 0.31 \cdot \exp[-N(3.55/\ln R + 0.68)]$ |
| Crossing error | $\epsilon(N, A) \approx \exp(-0.93 - 14.25 \cdot N/A)$ |
| HGL weight | $h(\tau) = \exp(-\tau - \tau^2/2)$ |
| HGL points | $N^{(\tau,\mathrm{HGL})} = c_2 x^2 + c_1 x + c_0$ where $x = \gamma E^{(\mathrm{bw})}$ |

---

## 1. Two Regimes of Energy Denominators

GW self-energy evaluation requires approximating energy denominators $1/x$ to enable $O(N^3)$ separable time-domain propagator products.

**Non-crossing windows** ($x > 0$, definite sign): $1/x \approx \sum_\ell w_\ell e^{-t_\ell x}$ on $[1, R]$ where $R = E_\mathrm{bw}/E_\mathrm{gap}$. Minimax error scales as $\epsilon \approx C\exp(-\pi^2 N/\ln(\beta R))$ with $\beta \approx 4$, giving $O(\ln R)$ nodes.

**Crossing windows** ($x$ changes sign): $1/x$ has a singularity at $x=0$ that must be regularized. The standard HGL approach (Kim et al. 2020) requires $O(A^2)$ nodes where $A = E_\mathrm{bw}/\xi$. Our learned-regularization approach fits $1/x$ directly with sine sums on $[u_\min, A]$ and lets the optimizer discover the regularization shape, achieving $O(A)$ node scaling. The user specifies a target effective broadening $\xi_\mathrm{eff}$; the solver binary-searches $A$ until the first-moment missing area matches a Lorentzian with width $\xi_\mathrm{eff}$.

---

## 2. Pole Structure of $\chi(\omega)$

The RPA polarizability has two poles per transition:
$$
\chi^0(\omega)_{rr'} = \sum_{cv\sigma\sigma'} \psi_{xc}\psi^*_{xv}\psi^*_{x'c}\psi_{x'v} \left[\underbrace{\frac{1}{\omega - \Delta E_{cv}}}_{\text{resonant}} - \underbrace{\frac{1}{\omega + \Delta E_{cv}}}_{\text{antiresonant}}\right]
$$
where $\Delta E_{cv} = E_c - E_v > 0$.

| Term | Denominator | Sign for $\omega > 0$ | Quadrature |
|------|-------------|----------------------|------------|
| Antiresonant | $\omega + \Delta E_{cv}$ | Always positive | GL |
| Resonant | $\omega - \Delta E_{cv}$ | Mixed if $\omega \in [E^{(\mathrm{gap})}, E^{(\mathrm{bw})}]$ | GL or HGL |

**Symmetry**: $\chi(\omega) = \chi(-\omega)^*$ — compute only $\omega \geq 0$, conjugate for $\omega < 0$.

---

## 3. Gauss-Laguerre (GL) — No Energy Crossing

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

## 4. Hermite-Gauss-Laguerre (HGL) — Energy Crossings

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

### 4.1 Golub-Welsch Construction of HGL Nodes

From Appendix H, nodes $\{\tau_u\}$ and weights $\{w_u\}$ for weight $h(\tau) = e^{-\tau-\tau^2/2}$:

1. **Compute moments** via GL quadrature on $e^{-\tau}$:
   $$\mu_k = \int_0^\infty \tau^k e^{-\tau - \tau^2/2} d\tau \approx \sum_j w_j^{(\mathrm{GL})} e^{-x_j^2/2} x_j^k$$

2. **Build orthogonal polynomials** $p_n(\tau)$ via three-term recurrence:
   $$p_{n+1}(\tau) = (\tau - a_n)p_n(\tau) - b_n p_{n-1}(\tau)$$

3. **Form Jacobi matrix**: $J = \mathrm{tridiag}(\sqrt{b_2}, \ldots; a_1, a_2, \ldots; \sqrt{b_2}, \ldots)$

4. **Diagonalize**: Eigenvalues give nodes $\tau_u$; weights from first eigenvector components: $w_u = \mu_0 \cdot v_1^2(u)$

---

## 5. Solver Methods

### 5.1 Noncrossing Solver

**Remez exchange** with damped Newton equioscillation solver, VarPro warm-starts, and continuation from $R=2$ doubling each step. The basic VarPro+Lawson IRLS solver is used for the auto-N-finder (`noncrossing_grids`); the Remez solver refines further for standalone use (`solve_noncrossing`).

Empirical error scaling (calibrated on $\epsilon \in [10^{-5}, 10^{-2}]$, $R^2 = 0.995$):
$$\epsilon(N, R) \approx 0.31 \cdot \exp\!\big[-N\big(\tfrac{3.55}{\ln R} + 0.68\big)\big]$$

### 5.2 Crossing Solver

LP backward elimination on a dense candidate grid to select $N$ frequencies, then VarPro-LM + Lawson polish, then final minimax LP for optimal weights.

Empirical error scaling (calibrated on $A = 50, 100, 200$, $R^2 = 0.996$):
$$\epsilon(N, A) \approx \exp(-0.93 - 14.25 \cdot N/A)$$

### 5.3 Imaginary-Axis Solver

Fits $x/(x^2 + \omega_p^2)$ on $[1, R]$ with the same VarPro+Lawson machinery as the noncrossing solver, but with a modified target function. Used for $\chi^0(i\omega_p)$ where the combined resonant+antiresonant denominator gives $2E/(E^2 + \omega_p^2)$.

---

## 6. Usage (from `src/common/minimax.py`)

```python
from common.minimax import (
    # Non-crossing
    solve_noncrossing, predict_N_noncrossing, noncrossing_grids,
    evaluate_noncrossing,
    # Crossing
    build_crossing_quadrature, predict_N_crossing,
    evaluate_crossing,
    # Imaginary-axis
    solve_noncrossing_imag, noncrossing_imag_grids,
)

# Noncrossing: 1/x ≈ Σ w_l exp(-t_l x)  on [1, R]
R = E_bw / E_gap
N = predict_N_noncrossing(R, target_error=0.01/R)
tau, w, err = solve_noncrossing(N, R)

# Crossing: 1/x ≈ Σ w_l sin(τ_l x / ξ_0)  for |x| > x_min
N, A_est = predict_N_crossing(xi_eff_target=0.2, E_bw=10.0, target_error=0.01/70)
tau, w, info = build_crossing_quadrature(N, xi_eff_target=0.2, E_bw=10.0)
xi_0 = info['xi_0']

# Imaginary-axis: x/(x²+ω²) ≈ Σ w_l exp(-t_l x)
tau, w, err = solve_noncrossing_imag(N=11, R=52.0, omega_hat=16.3)
```

## 6.1 Shipped quadrature tables

LORRAX can optionally skip runtime minimax tuning and load bundled node/weight tables
from `src/common/minimax_assets/`.

Enable from the GW input:

```ini
regenerate_minimax_tables = false
```

Default is `false`, meaning shipped tables are reused when a safe match exists. Set it
to `true` to force exact regeneration.

Selection rule at runtime:
- choose the smallest tabulated range that is greater than or equal to the requested one
- choose a stricter-or-equal tabulated error bound
- reject tables whose node count exceeds the caller's `max_nodes`

Current conventions:
- noncrossing tables are fit on scaled `[1, R]` using absolute `L∞` error in `1/x`
- crossing tables are fit on `[0, A_dim]` using absolute `L∞` error in the target `G(u)`

This is not a relative-at-endpoint convention.

---

## 7. k-Space Structure with Spin

### 7.1 Green's Function

For spinor wavefunctions $\psi_{n\mathbf{k},a}(\mathbf{r})$ with spin indices $a,b$:
$$
G_{\mathbf{k},ab}(\mathbf{r},\mathbf{r}';\omega) = \sum_n \frac{\psi_{n\mathbf{k},a}(\mathbf{r})\,\psi^*_{n\mathbf{k},b}(\mathbf{r}')}{\omega - E_{n\mathbf{k}}}
$$

### 7.2 Lattice Fourier Transform to R-Space
$$
G_{\mathbf{R},ab}(\mathbf{r},\mathbf{r}';\omega) = \frac{1}{N_k}\sum_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{R}}\,G_{\mathbf{k},ab}
$$

Implementation: `jnp.fft.ifftn(..., norm='ortho')` on last 3 dims (kx,ky,kz).

### 7.3 Polarizability as Spin-Traced G·G Product
$$
\chi^0_{\mathbf{R}}(\mathbf{r},\mathbf{r}';\omega) = -\sum_{ab} G_{\mathbf{R},ab}(\mathbf{r},\mathbf{r}')\, G_{-\mathbf{R},ba}(\mathbf{r}',\mathbf{r})
$$

---

## 8. CTSP Propagators with k and Spin

### 8.1 Window-Pair Propagators

**Valence (occupied) propagator:**
$$
\rho^{(v)}_{\mathbf{k},ab}(\tau;\mathbf{r},\mathbf{r}') = \sum_{v \in \mathcal{L}} e^{-\zeta\tau(E_l^{(v,\max)} - E_{v\mathbf{k}})} \psi_{v\mathbf{k},a}(\mathbf{r})\psi^*_{v\mathbf{k},b}(\mathbf{r}')
$$

**Conduction (unoccupied) propagator:**
$$
\rho^{(c)}_{\mathbf{k},ab}(\tau;\mathbf{r},\mathbf{r}') = \sum_{c \in \mathcal{M}} e^{-\zeta\tau(E_{c\mathbf{k}} - E_m^{(c,\min)})} \psi_{c\mathbf{k},a}(\mathbf{r})\psi^*_{c\mathbf{k},b}(\mathbf{r}')
$$

### 8.2 Static ($\omega=0$) Contribution — GL Only

$$
\chi^{0,\mathrm{static}}_{lm,\mathbf{R}}(\mathbf{r},\mathbf{r}') = -2\zeta_{lm} \sum_u w_u e^{-(\zeta_{lm} E_{lm}^{(\mathrm{gap})} - 1)\tau_u} \sum_{ab} \rho^{(c)}_{\mathbf{R},ab}(\tau_u) \, \rho^{(v)}_{-\mathbf{R},ba}(\tau_u)
$$

### 8.3 Dynamic ($\omega \neq 0$) Contributions

**Antiresonant term** (always positive denominator → GL):
$$
\chi^{0,(-)}_{lm}(\omega) : \quad \text{shift } E^{(\mathrm{gap})} \to E^{(\mathrm{gap})} + \omega, \quad \text{use GL}
$$

**Resonant term** — depends on whether $\omega$ causes crossing:

*No crossing* ($\omega < E^{(\mathrm{gap})}_{lm}$ or $\omega > E^{(\mathrm{bw})}_{lm}$): Use GL with shifted gap.

*Crossing* ($E^{(\mathrm{gap})}_{lm} < \omega < E^{(\mathrm{bw})}_{lm}$): Use HGL with Euler identity.

---

## 9. HGL Separable Form via Euler Identity

When $\omega - \Delta E_{cv}$ changes sign, use Euler's identity to reduce memory by 2×.

### 9.1 Complex Propagators (2 arrays instead of 4)

$$
G^{v}_{\mathbf{k},ab}(\tau) = \sum_{v \in \mathcal{L}} e^{i\gamma\tau E_{v\mathbf{k}}} \, \psi_{v\mathbf{k},a}\psi^*_{v\mathbf{k},b}
$$
$$
G^{c}_{\mathbf{k},ab}(\tau) = \sum_{c \in \mathcal{M}} e^{i\gamma\tau E_{c\mathbf{k}}} \, \psi_{c\mathbf{k},a}\psi^*_{c\mathbf{k},b}
$$

These satisfy $G = C + iS$ where $C = \sum_n \cos(\gamma\tau E_n)\psi_n\psi_n^\dagger$ and $S = \sum_n \sin(\gamma\tau E_n)\psi_n\psi_n^\dagger$.

### 9.2 Products via Hermitian Conjugate

After FFT to R-space:
$$
(G^c_{-\mathbf{R}})^\dagger \cdot G^v_{\mathbf{R}} = P_+ - i P_\times
$$

where $P_+ = \mathrm{Re}[(G^c)^\dagger G^v] = C^c C^v + S^c S^v$ and $P_\times = -\mathrm{Im}[(G^c)^\dagger G^v] = S^c C^v - C^c S^v$.

### 9.3 Batch All Frequencies

$$
\chi^{\mathrm{cross}}_{lm}(\omega_i) = -\gamma \sum_u w_u \left[\cos(\gamma\tau_u\omega_i)\,P_\times(\tau_u) - \sin(\gamma\tau_u\omega_i)\,P_+(\tau_u)\right]
$$

---

## 10. Pole Symmetries and Efficiency

1. **Frequency symmetry**: $\chi(-\omega) = \chi(\omega)^*$ — only compute $\omega \geq 0$
2. **Antiresonant term**: always GL, no crossings possible
3. **Reuse propagators across frequencies**: $\omega$-independent parts precomputed once per window pair
4. **R-space symmetry**: $\chi_{\mathbf{R}}(\omega) = \chi_{-\mathbf{R}}(\omega)^*$ — only compute half of R-points

---

## 11. Complex Frequencies: $z = \omega + i\eta$

The Laplace identity $1/s = \int_0^\infty d\tau\, e^{-s\tau}$ converges when $\mathrm{Re}(s) > 0$. For $s = z - \Delta E$, convergence depends on $\mathrm{Re}(s) = \omega - \Delta E$, not on $\eta$. The imaginary part adds an oscillatory phase $e^{-i\eta\tau}$.

**GL with complex z**: No changes to quadrature nodes/weights. Use complex phase:
$$
\text{phase}_{\mathrm{GL}}[z, u] = e^{-\zeta(z - E^{(\mathrm{gap})})\tau_u + \tau_u}
$$

**HGL with complex z**: $\sin(\gamma\tau(x_r + ix_i)) = \sin(\gamma\tau x_r)\cosh(\gamma\tau\eta) + i\cos(\gamma\tau x_r)\sinh(\gamma\tau\eta)$

| Condition | Method | Reason |
|-----------|--------|--------|
| $\|\eta\| \gg E^{(\mathrm{bw})}$ | GL everywhere | Far from real axis |
| $\omega \notin [E^{(\mathrm{gap})}, E^{(\mathrm{bw})}]$ | GL | No crossing |
| $\omega \in [E^{(\mathrm{gap})}, E^{(\mathrm{bw})}]$, $\|\eta\|$ small | HGL | Near real-axis crossing |

**Rule of thumb**: If $\|\eta\| > \gamma^{-1}$, you can use GL even for "crossing" windows.

---

## Appendix: Why the 1/2 Ratio in $h(\tau)$

The transform $F(x) = \int_0^\infty h(\tau) e^{ix\tau} d\tau$ expands as $F(x) \sim 1/(ix) + a_1/(ix)^2 + \cdots$. The coefficient $a_1$ vanishes when $\int_0^\infty \tau h(\tau) d\tau / \int_0^\infty h(\tau) d\tau = 1$, which holds exactly when the linear/quadratic ratio is 1/2. This gives $F(x) = 1/x + O(1/x^5)$ instead of $O(1/x^3)$.

---

## Index of Symbols

| Symbol | Definition |
|--------|------------|
| $z = \omega + i\eta$ | Complex frequency |
| $\zeta_{lm}$ | Energy scale for window pair: $\zeta^{-1} = \sqrt{E^{(\mathrm{bw})} E^{(\mathrm{gap})}}$ |
| $\gamma$ | HGL broadening parameter |
| $\epsilon^{(q)}$ | Target fractional quadrature error |
| $\alpha$ | Bandwidth ratio $\sqrt{E^{(\mathrm{bw})}/E^{(\mathrm{gap})}}$ |
| $R$ | Dynamic range $E_\mathrm{bw}/E_\mathrm{gap}$ |
| $A$ | Dimensionless crossing bandwidth $E_\mathrm{bw}/\xi_0$ |
| $\mathcal{L}, \mathcal{M}$ | Index sets for valence/conduction bands in windows |
| $\rho^{(v/c)}$ | Exponentially-damped density matrix (GL) |
| $G^{v/c}$ | Complex propagator (HGL, Euler) |
| $P_+, P_\times$ | Products from $(G^c)^\dagger G^v$ |

---

## References

- Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020) — CTSP derivation, Appendices A–H
- Hackbusch, Comput. Vis. Sci. 21, 1 (2019) — Exponential sums for 1/x
- Helmich-Paris & Visscher, J. Comput. Phys. 321, 927 (2016)
- Golub & Pereyra, SIAM J. Numer. Anal. 10, 413 (1973) — Variable projection
- Golub & Welsch, Math. Comp. 23, 221 (1969) — Gaussian quadrature from moments
