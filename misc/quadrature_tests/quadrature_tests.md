# CTSP Quadrature Tests: Mathematical Equations

Reference: Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020)

---

## Overview

The Complex-Time Shredded Propagator (CTSP) method recasts energy denominators as Laplace-type integrals, enabling $O(N^3)$ polarizability and self-energy calculations. Two quadrature methods are used:

| Method | Weight Function | Use Case | Test File |
|--------|-----------------|----------|-----------|
| **GL** (Gauss-Laguerre) | $e^{-\tau}$ | Non-crossing windows (static $\chi$) | `gl_quadrature_test.py` |
| **HGL** (Hermite-Gauss-Laguerre) | $e^{-\tau - \tau^2/2}$ | Crossing windows (dynamic $\chi(\omega)$) | `hgl_quadrature_test.py` |

---

## GL Quadrature (Static Polarizability)

### Single Denominator Identity

$$
\frac{1}{s} = \int_0^\infty e^{-s\tau} \, d\tau = \zeta \int_0^\infty e^{-\zeta s \tau} \, d\tau
$$

With GL weight $e^{-\tau}$:
$$
\frac{1}{s} = \zeta \int_0^\infty e^{-(\zeta s - 1)\tau} \cdot e^{-\tau} \, d\tau \approx \zeta \sum_{u=1}^{N_\tau} w_u \, e^{-(\zeta s - 1)\tau_u}
$$

### Separable Form for χ

**Target:**
$$
\chi = -\sum_{v,c} \frac{A_v A_c}{E_c - E_v}
$$

**CTSP form** (separable in $v$ and $c$):
$$
\chi = -\zeta \sum_{u=1}^{N_\tau} w_u \, e^{-(\zeta E_\mathrm{gap} - 1)\tau_u} \cdot \rho_c(\tau_u) \cdot \bar{\rho}_v(\tau_u)
$$

where:
$$
\rho_c(\tau) = \sum_c e^{-\zeta\tau(E_c - E_{c,\min})} A_c, \quad
\bar{\rho}_v(\tau) = \sum_v e^{-\zeta\tau(E_{v,\max} - E_v)} A_v
$$

**Optimal scaling:** $\zeta = 1/\sqrt{E_\mathrm{gap} \times E_\mathrm{bw}}$

**Quadrature points:** $N_\tau^{(\mathrm{GL})} = \sqrt{E_\mathrm{bw}/E_\mathrm{gap}} \times (0.4 - 0.3 \ln \epsilon)$

---

## HGL Quadrature (Dynamic Polarizability with Crossings)

### Regularization Function

$$
F(x) = \int_0^\infty \sin(x\tau) \, e^{-\tau - \tau^2/2} \, d\tau
$$

**Properties:**
- $F(x) \to 1/x$ for $|x| \gg 1$
- $F(x) \approx \tau_0 x$ for $|x| \ll 1$ where $\tau_0 \approx 0.344$
- $F(x) \cdot x$ is bounded (regularizes the singularity)

### Complex Propagators (Euler Identity — Default Approach)

For **Hermitian** propagators ($\psi\psi^\dagger$), use 2 complex arrays:
$$
G^c = \sum_c e^{i\gamma\tau E_c} A_c, \quad G^v = \sum_v e^{i\gamma\tau E_v} A_v
$$

Products emerge from Hermitian conjugate:
$$
(G^c)^\dagger G^v = P_+ - i P_\times
$$

**Extraction:**
$$
P_+ = \mathrm{Re}[(G^c)^\dagger G^v], \quad P_\times = -\mathrm{Im}[(G^c)^\dagger G^v]
$$

**Why it works:** For Hermitian matrices, $A^\dagger = A^*$, so conjugate = Hermitian transpose.

**Memory:** 2 complex arrays instead of 4 real → **2× reduction** vs naive approach.

### Full χ(ω) Formula

When $\omega$ causes an energy crossing ($E_\mathrm{gap} < \omega < E_\mathrm{bw}$):
$$
\chi(\omega) = -\gamma \sum_u w_u \left[\cos(\gamma\tau_u\omega)\,P_\times(\tau_u) - \sin(\gamma\tau_u\omega)\,P_+(\tau_u)\right]
$$

### Alternative: Naive 4-Array Approach

For reference, the products can also be computed from 4 real arrays:
$$
C^{v/c} = \sum_n \cos(\gamma\tau E_n) A_n, \quad S^{v/c} = \sum_n \sin(\gamma\tau E_n) A_n
$$
$$
P_+ = C^c C^v + S^c S^v, \quad P_\times = S^c C^v - C^c S^v
$$

This is equivalent but uses **2× more memory**.

---

## HGL Node/Weight Generation

The HGL quadrature nodes and weights are computed via the **Golub-Welsch algorithm** (see `hgl_quadrature.py`):

1. Compute moments $\mu_k = \int_0^\infty \tau^k e^{-\tau - \tau^2/2} d\tau$
2. Build orthogonal polynomials via three-term recurrence
3. Construct Jacobi matrix with recurrence coefficients
4. Eigendecomposition: nodes = eigenvalues, weights = $\mu_0 \times v_1^2$

**Quadrature points:**
$$
N_\tau^{(\mathrm{HGL})} = c_2 x^2 + c_1 x + c_0
$$
where $x = \gamma E_\mathrm{bw}$ and coefficients depend on $\ln\epsilon$.

---

## File Structure

```
misc/quadrature_tests/
├── __init__.py
├── hgl_quadrature.py          # HGL node/weight generation (Golub-Welsch)
├── gl_quadrature_test.py      # GL tests (static χ)
├── hgl_quadrature_test.py     # HGL tests (dynamic χ with Euler optimization)
└── quadrature_tests.md        # This file
```

---

## Running the Tests

```bash
cd isdf_cohsex
source .venv/bin/activate

# GL tests (static polarizability)
python misc/quadrature_tests/gl_quadrature_test.py

# HGL tests (dynamic polarizability with Euler optimization)
python misc/quadrature_tests/hgl_quadrature_test.py
```

**Expected output:** All tests pass with machine precision errors (~10⁻¹⁴).
