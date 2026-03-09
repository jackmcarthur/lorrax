# Full-Frequency GW Self-Energy via GN-PPM and Minimax-Windowed CTSP in the ISDF Basis

## Purpose

Self-contained theoretical reference for computing $\Sigma_{nk}(\omega)$ in $O(N^3)$ via three ingredients: ISDF spatial compression, GN-PPM spectral compression of $W$, and minimax-windowed CTSP for real-axis frequency evaluation. No analytic continuation. Controllable quadrature error.

---

## 1. Conventions

### 1.1 Indices

| Symbol | Meaning |
|--------|---------|
| $v, c$ | Valence / conduction band |
| $\mathbf{k}, \mathbf{q}, \mathbf{R}$ | Crystal momentum / transfer / lattice vector |
| $\mu, \nu, \eta, \zeta$ | ISDF collocation points ($1, \ldots, N_\mu$) |
| $a, b \in \{\uparrow, \downarrow\}$ | Spinor components |

### 1.2 Energies

$E_F = 0$. Conduction: $E_{c\mathbf{k}} \geq 0$. Valence: $E_{v\mathbf{k}} \leq 0$. PPM poles: $\Omega_{q,\mu\nu} \geq 0$.

### 1.3 Bloch spinors

$$\psi_{n\mathbf{k},a}(\mathbf{r}) = \frac{1}{\sqrt{N_r}} \sum_{\mathbf{G}} c_{n\mathbf{k},a}(\mathbf{G})\, e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$$

### 1.4 Lattice Fourier transforms

$$f_{\mathbf{R}} = \frac{1}{\sqrt{N_k}} \sum_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{R}}\, f_{\mathbf{k}} \quad \leftrightarrow \quad f_{\mathbf{k}} = \frac{1}{\sqrt{N_k}} \sum_{\mathbf{R}} e^{-i\mathbf{k}\cdot\mathbf{R}}\, f_{\mathbf{R}}$$

$k \to R$: `ifftn(f_k, axes=k_axes, norm='ortho')`. $R \to k$: `fftn(f_R, axes=R_axes, norm='ortho')`.

### 1.5 Spin structure

$G_{\mathbf{k},ab}(\mu,\nu)$ and $\Sigma_{\mathbf{k},ab}(\mu,\nu)$ carry spin as a $2\times 2$ matrix. The polarizability $\chi^0_q(\mu,\nu)$ is spin-traced (scalar). Everything derived from $\chi^0$ — $\Pi_q$, $W_q$, $V_q$, $B_q$, $\Omega_q$ — is spin-independent.

---

## 2. Self-energy decomposition

### 2.1 Exchange + correlation

$$\Sigma_{\mathbf{k},ab}(\mu,\nu;\omega) = \Sigma^x_{\mathbf{k},ab}(\mu,\nu) + \Sigma^c_{\mathbf{k},ab}(\mu,\nu;\omega)$$

**Exchange** is static and uses bare Coulomb $v$:

$$\Sigma^x_{\mathbf{R},ab}(\mu,\nu) = -G^\text{occ}_{\mathbf{R},ab}(\mu,\nu) \cdot V_{\mathbf{R}}(\mu,\nu)$$

computed as an elementwise product in $R$-space via FFT convolution. $V$ has no spin. Not part of the CTSP pipeline.

**Correlation** uses $W^c = v\Pi v = W - v$ and carries all frequency dependence:

$$\Sigma^c(\omega) = \frac{i}{2\pi}\int d\omega'\, G(\omega - \omega')\, W^c(\omega')$$

The Coulomb sandwiching $v(\cdots)v$ is performed **outside** the time loop: CTSP builds the $\Pi$-basis result; $v\Pi v$ is applied once after accumulating all windows.

### 2.2 Spectral form with PPM poles

With PPM poles $\Omega_s > 0$ and residues $B^s_{\mu\nu}$ (spin-independent):

$$\Sigma^{(+)}_{\mathbf{k},ab} = \sum_{v,s} \frac{\psi_{v,a}(r_\mu)\,\psi^*_{v,b}(r_\nu)\; \tilde{B}^s_{\mu\nu}}{\omega - E_v + \Omega_s}, \qquad \Sigma^{(-)}_{\mathbf{k},ab} = \sum_{c,s} \frac{\psi_{c,a}(r_\mu)\,\psi^*_{c,b}(r_\nu)\; \tilde{B}^s_{\mu\nu}}{\omega - E_c - \Omega_s}$$

where $\tilde{B}^s$ absorbs the $v B v$ sandwiching. Spin comes entirely from $G$.

### 2.3 Denominator structure

| Term | Denominator | Positive-axis sum | Sign on $\omega \geq 0$ |
|------|-------------|-------------------|-------------------------|
| $\Sigma^{(-)}$ | $\omega - E_c - \Omega$ | $S = E_c + \Omega$ | Crosses zero (minus-kernel) |
| $\Sigma^{(+)}$ | $\omega + h_v + \Omega$ | $S = h_v + \Omega$ | Always positive (plus-kernel) |

where $h_v = -E_v \geq 0$.

---

## 3. ISDF basis (summary)

Spin-traced pair density: $\rho_{mn,\mathbf{k}}(\mathbf{q};\mathbf{r}) = \sum_a \psi^*_{m,\mathbf{k}-\mathbf{q},a}(\mathbf{r})\,\psi_{n,\mathbf{k},a}(\mathbf{r}) \approx \sum_\mu \zeta_{q,\mu}(\mathbf{r})\, \rho_{mn,\mathbf{k}}(\mathbf{q};\mathbf{r}_\mu)$.

Screened interaction: $W = v + v\Pi v$, $\Pi = \chi^0[I - v\chi^0]^{-1}$. All spin-independent.

---

## 4. The GN-PPM: spectral compression of $W$

### 4.1 What it does

Replaces $O(N_t)$ poles of $\Pi$ with one pole per matrix element:

$$\Pi_{q,\mu\nu}(\omega) \approx \frac{2\, B_{q,\mu\nu}\, \Omega_{q,\mu\nu}}{\omega^2 - \Omega_{q,\mu\nu}^2}$$

The CTSP "pole axis" is then parameterized by $\{\Omega_{q,\mu\nu}\}$ directly.

### 4.2 Parameter extraction (Phase 1)

Requires $\Pi_q$ at two imaginary-axis frequencies: $\omega = 0$ and $\omega = i\omega_p$ ($\omega_p \sim 0.5$–$1.0$ Ha). Both are sign-definite — the combined resonant + antiresonant denominator $-2\Delta E/(\Delta E^2 + |\omega|^2)$ never vanishes.

**This means the entire band structure fits in a single Laplace window.** With $R = E_\text{bw}/E_\text{gap} \sim 30$, the Laplace minimax quadrature needs $\sim 12$–$15$ nodes for $10^{-6}$ accuracy. No crossing treatment.

At each Laplace node $\tau_\ell$, the computation is:

1. Build conduction and valence propagators $G^c_{\mathbf{k},ab}$, $G^v_{\mathbf{k},ab}$ at all $k$ — $O(N_k N_\text{band} N_s^2 N_\mu^2)$.
2. FFT $k \to R$ via `ifftn` (conduction) and $k \to -R$ via `fftn` (valence) — $O(N_s^2 N_\mu^2 N_k \log N_k)$.
3. Spin-traced contraction in $R$-space:
$$\tilde{\chi}^0_{\mathbf{R},\mu\nu}(\tau_\ell) = \sum_{ab} G^c_{\mathbf{R},ab}(\mu,\nu) \cdot G^v_{-\mathbf{R},ba}(\nu,\mu)$$
4. FFT $R \to q$, accumulate with weights.

**Both frequencies share steps 1–4.** Only the scalar weight differs: $\alpha_\ell(i\omega_p) = \alpha_\ell(0) \cdot e^{-\omega_p \tau_\ell}$. Phase 1 costs $\approx 1\times$ a static $\chi^0$ computation.

Then per $q$-point:

$$\Pi_q(z) = \chi^0_q(z)\,[I - v_q\chi^0_q(z)]^{-1} \qquad [O(N_\mu^3)]$$

$$\Omega_{q,\mu\nu} = \omega_p \sqrt{\operatorname{Re}\!\left[\frac{\Pi(i\omega_p)}{\Pi(0) - \Pi(i\omega_p)}\right]}, \qquad B_{q,\mu\nu} = -\tfrac{1}{2}\,\Pi(0)\,\Omega$$

Guard: if radicand $< 0$ ("unfulfilled mode"), set $\Omega = 1$ Ha.

**Output:** $\Omega_{q,\mu\nu} \geq 0$ and $B_{q,\mu\nu}$ (real, spin-independent) for all $q$.

---

## 5. Phase 2: $\Sigma^c(\omega)$ via CTSP

### 5.1 Time-domain factorization

The CTSP replaces $1/(\omega - S)$ by a time integral whose integrand factorizes:

$$e^{-iSt} = e^{-iE_n t} \cdot e^{-i\Omega t}$$

At each quadrature node $t$:

1. **Build propagators.** Green's function $G_{\mathbf{k},ab}(\mu,\nu;t)$ from windowed band sums; PPM propagator $\Pi^\text{PPM}_{q,\mu\nu}(t) = B\, e^{-i\Omega t}$ from windowed pole sums. Both use $e^{-i\tilde{E}t}$, same code, same complex $t$.

2. **FFT to $R$-space.** `ifftn` for $G$ ($k \to +R$), `fftn` for $\Pi$ ($q \to -R$). The $\pm R$ signs encode the convolution $\sum_q G_{k-q} \Pi_q \leftrightarrow G_R \cdot \Pi_{-R}$.

3. **Multiply in $R$-space.** $\Sigma^c_{\mathbf{R},ab}(\mu,\nu;t) = G_{\mathbf{R},ab}(\mu,\nu;t) \cdot \Pi^\text{PPM}_{\mathbf{R}}(\mu,\nu;t)$. Elementwise in $(\mu,\nu)$ — collocation points are real-space points. Spin $(a,b)$ rides on $G$; $\Pi$ is spin-independent and broadcasts.

4. **FFT $R \to k$.**

5. **Frequency integrate.** $\Sigma^c(\omega) = \sum_u \alpha_u\, e^{i s_\omega \omega t_u}\, \Sigma^c(t_u)$, with an $\operatorname{Im}[\cdots]$ projection for crossing windows only.

### 5.2 Window decomposition

Define $T = \Omega_\text{max} + c_\text{edge}\cdot\xi$. Three rectangles in the positive-axis plane $(A, B)$:

| Window | $A$-range | $B$-range | Character |
|--------|-----------|-----------|-----------|
| Core | $[0, T]$ | $[0, T]$ | Crossing: $S$ can equal $\omega$ |
| A-stripe | $[T, A_\text{max}]$ | $[0, T]$ | Sign-definite: $S > \Omega$ |
| B-slab | $[0, A_\text{max}]$ | $[T, B_\text{max}]$ | Sign-definite: $S > \Omega$ |

The plus-kernel ($\Sigma^{(+)}$, valence) is sign-definite everywhere — all three windows use Laplace quadrature.

**Why only three windows:** The crossing quadrature costs $O(E_\text{bw}/\xi)$ nodes (linear), and the Laplace costs $O(\log R)$. Neither is expensive enough to justify finer subdivision.

### 5.3 The two time-node regimes

Laplace nodes: $t = -i\tau$ ($\tau > 0$). Propagators $e^{-i\tilde{E}(-i\tau)} = e^{-\tilde{E}\tau}$ decay.

Crossing (phase) nodes: $t \in \mathbb{R}$. Propagators $e^{-i\tilde{E}t}$ oscillate.

Same builder code. Same integrator code. The only branch is the $\operatorname{Im}[\cdots]$ projection.

### 5.4 After all windows

1. Coulomb sandwich: $\Sigma^c = v \cdot [\text{accumulated}] \cdot v$.
2. Add exchange: $\Sigma = \Sigma^x + \Sigma^c(\omega)$.
3. Band projection: $\Sigma_{ij,\mathbf{k}}(\omega) = \sum_{ab,\mu,\nu} \psi^*_{i,a}(r_\mu)\, \Sigma_{\mathbf{k},ab}(\mu,\nu;\omega)\, \psi_{j,b}(r_\nu)$.

### 5.5 Cost

Per window, per node: $O(N_k N_\text{band} N_s^2 N_\mu^2)$ (builds) + $O(N_s^2 N_\mu^2 N_k \log N_k)$ (FFTs) + $O(N_R N_s^2 N_\mu^2)$ (multiply). Total: $O(N_k N^3)$.

---

## 6. Pipeline summary

**Phase 1** (GN-PPM, single Laplace window, $\sim 12$ nodes): build $\chi^0(0)$ and $\chi^0(i\omega_p)$ sharing propagators, Dyson-solve to $\Pi$, extract $\Omega, B$ elementwise.

**Phase 2** ($\Sigma^c(\omega)$, 3-window per term): for each (cond/val) term and each window, build propagators at quadrature nodes, FFT-convolve in $R$-space, frequency-integrate with appropriate kernel. Accumulate, Coulomb-sandwich, add $\Sigma^x$, project to bands.

---

## 7. Design principles

1. **GN-PPM collapses the pole axis.** $O(N_t)$ poles → one per $(\mu,\nu)$. Filtering $\Pi(t)$ to a window is just masking.

2. **Phase 1 is free.** $\sim 12$ Laplace nodes, single window, shared propagators for both frequencies.

3. **Minimax sine replaces Golub–Welsch.** $O(A)$ crossing nodes instead of $O(A^2)$.

4. **Three windows suffice.** Minimal partition separating crossing from non-crossing.

5. **One builder, one integrator.** Laplace vs. crossing differs only in $t$ being imaginary vs. real and one $\operatorname{Im}[\cdots]$ projection.

6. **No analytic continuation.** Real-axis $\Sigma^c(\omega)$ directly.

---

## 8. Companion documents

| Document | Content |
|----------|---------|
| `MINIMAX_CTSP_IMPLEMENTATION.md` | Code-level specification: `convolve_frequencies`, builders, routing |
| `PHYSICS_COMPREHENSIVE.md` | ISDF fitting, Galerkin equations, sharding, static COHSEX |
| `crossing_minimax_overview.md` | Minimax sine-sum algorithm, error scaling |
| `chi_omega_quadrature.md` | CTSP quadrature, GL/HGL derivations, complex-frequency extension |
| `GN-PPM_GUIDELINES.md` | GN-PPM in ISDF, $\omega_p$ selection, failure conditions |
