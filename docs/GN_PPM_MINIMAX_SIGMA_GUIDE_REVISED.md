# Full-Frequency GW Self-Energy via GN-PPM and Minimax-Windowed CTSP in the ISDF Basis

## Purpose

Self-contained theoretical reference for computing $\Sigma_{nk}(\omega)$ in $O(N^3)$ via ISDF spatial compression, GN-PPM spectral compression of $W$, and minimax-windowed CTSP for real-axis frequency evaluation. No analytic continuation. Controllable quadrature error.

---

## 1. Conventions

### 1.1 Indices

| Symbol | Meaning |
|--------|---------|
| $v, c$ | Valence / conduction band |
| $\mathbf{k}, \mathbf{q}, \mathbf{R}$ | Crystal momentum / transfer / lattice vector |
| $\mu, \nu, \eta, \zeta$ | ISDF collocation points ($1, \ldots, N_\mu$) |
| $a, b \in \{\uparrow, \downarrow\}$ | Spinor components |

### 1.2 Energy referencing

All eigenvalues are stored in the **vacuum reference** as read from DFT: $E_n^\text{vac}$. The Fermi level $E_F$ is a single scalar parameter. Positive-axis values used in the CTSP windowing are:

$$\tilde{E}_c = E_c^\text{vac} - E_F \geq 0, \qquad h_v = E_F - E_v^\text{vac} \geq 0, \qquad \Omega_{q,\mu\nu} \geq 0$$

The self-energy is evaluated at frequencies $\omega^\text{vac}$ (vacuum-referenced), converted internally to $\omega_\text{rel} = \omega^\text{vac} - E_F$.

**For self-consistent iterations:** update $E_F$ (and eigenvalues if doing eigenvalue self-consistency). All internal quantities reflow from these. $E_F$ cancels algebraically in every propagator–kernel product (§5.4).

### 1.3 Bloch spinors

$$\psi_{n\mathbf{k},a}(\mathbf{r}) = \frac{1}{\sqrt{N_r}} \sum_{\mathbf{G}} c_{n\mathbf{k},a}(\mathbf{G})\, e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}$$

### 1.4 Lattice Fourier transforms

$$f_{\mathbf{R}} = \frac{1}{\sqrt{N_k}} \sum_{\mathbf{k}} e^{i\mathbf{k}\cdot\mathbf{R}}\, f_{\mathbf{k}} \quad \leftrightarrow \quad f_{\mathbf{k}} = \frac{1}{\sqrt{N_k}} \sum_{\mathbf{R}} e^{-i\mathbf{k}\cdot\mathbf{R}}\, f_{\mathbf{R}}$$

$k \to R$: `ifftn(norm='ortho')`. $R \to k$: `fftn(norm='ortho')`.

### 1.5 Spin structure

$G_{\mathbf{k},ab}(\mu,\nu)$ and $\Sigma_{\mathbf{k},ab}(\mu,\nu)$ carry spin as $2\times 2$ matrices. $\chi^0_q$, $\Pi_q$, $W_q$, $V_q$, $B_q$, $\Omega_q$ are spin-independent (scalar).

---

## 2. Self-energy decomposition

### 2.1 Exchange + correlation

$$\Sigma_{\mathbf{k},ab}(\mu,\nu;\omega) = \Sigma^x_{\mathbf{k},ab}(\mu,\nu) + \Sigma^c_{\mathbf{k},ab}(\mu,\nu;\omega)$$

**Exchange** is static, uses bare Coulomb $v$:

$$\Sigma^x_{\mathbf{R},ab}(\mu,\nu) = -G^\text{occ}_{\mathbf{R},ab}(\mu,\nu) \cdot V_{\mathbf{R}}(\mu,\nu)$$

Not part of the CTSP pipeline.

**Correlation** uses $W^c = v\Pi v = W - v$, carries all frequency dependence. Coulomb sandwiching $v(\cdots)v$ applied outside the time loop.

### 2.2 Equivalence of SX+COH and $\Sigma^{(\pm)}$ decompositions

BerkeleyGW (Kim-2020 Eq. 6) computes $\Sigma^c$ via SX + COH. Using the PPM spectral form $W^c(\omega) = \sum_p B^p[(\omega - \omega_p)^{-1} - (\omega + \omega_p)^{-1}]$:

$$\text{SX} = -\sum_v \psi_v\psi_v^* W(\omega - E_v) = -\sum_v \psi_v V\psi_v^* + \sum_{v,p} \frac{\psi_v B^p \psi_v^*}{\omega - E_v + \omega_p} - \sum_{v,p} \frac{\psi_v B^p \psi_v^*}{\omega - E_v - \omega_p}$$

$$\text{COH} = \sum_{n,p} \frac{\psi_n B^p \psi_n^*}{\omega - E_n - \omega_p} = \sum_{v,p} \frac{\psi_v B^p \psi_v^*}{\omega - E_v - \omega_p} + \sum_{c,p} \frac{\psi_c B^p \psi_c^*}{\omega - E_c - \omega_p}$$

The occupied $(\omega - E_v - \omega_p)^{-1}$ terms cancel between SX and COH:

$$\boxed{\Sigma^c = \underbrace{\sum_{v,p} \frac{B^p \psi_v \psi_v^*}{\omega - E_v + \omega_p}}_{\Sigma^{(+)}} + \underbrace{\sum_{c,p} \frac{B^p \psi_c \psi_c^*}{\omega - E_c - \omega_p}}_{\Sigma^{(-)}}}$$

Same total, different grouping: CTSP splits by occupancy; BGW splits by pole origin.

### 2.3 Denominator structure and $\omega$ sign

Using the Fermi-referenced positive-axis values $\tilde{E}_c = E_c - E_F$, $h_v = E_F - E_v$, and $\omega_\text{rel} = \omega - E_F$:

$$\Sigma^{(-)} \text{ denominator:} \quad \omega_\text{rel} - (\tilde{E}_c + \Omega)$$
$$\Sigma^{(+)} \text{ denominator:} \quad \omega_\text{rel} + (h_v + \Omega)$$

The crossing structure depends on the sign of $\omega_\text{rel}$:

| | $\omega_\text{rel} > 0$ (above $E_F$) | $\omega_\text{rel} < 0$ (below $E_F$) |
|---|---|---|
| $\Sigma^{(-)}$: $\omega_\text{rel} - (\tilde{E}_c + \Omega)$ | **Crosses zero** | Always negative → sign-definite |
| $\Sigma^{(+)}$: $\omega_\text{rel} + (h_v + \Omega)$ | Always positive → sign-definite | **Crosses zero** |

The poles of $\Sigma^{(-)}$ are at $\omega_\text{rel} = \tilde{E}_c + \Omega > 0$, always above $E_F$. The poles of $\Sigma^{(+)}$ are at $\omega_\text{rel} = -(h_v + \Omega) < 0$, always below $E_F$. The window structure is symmetric about $E_F$.

**Consequence:** `convolve_frequencies` handles this by splitting the input $\omega$ array at $E_F$. For $\omega > E_F$: $\Sigma^{(-)}$ gets 3 windows, $\Sigma^{(+)}$ single Laplace. For $\omega < E_F$: vice versa. Each half is a standard non-negative-$\omega_\text{rel}$ problem.

---

## 3. ISDF basis (summary)

Spin-traced pair density: $\rho_{mn,\mathbf{k}}(\mathbf{q};\mathbf{r}) \approx \sum_\mu \zeta_{q,\mu}(\mathbf{r})\, \rho_{mn,\mathbf{k}}(\mathbf{q};\mathbf{r}_\mu)$.

Screened interaction: $W = v + v\Pi v$, $\Pi = \chi^0[I - v\chi^0]^{-1}$. All spin-independent.

---

## 4. GN-PPM: spectral compression of $W$ (Phase 1)

### 4.1 What it does

Replaces $O(N_t)$ poles of $\Pi$ with one per matrix element:

$$\Pi_{q,\mu\nu}(\omega) \approx \frac{2\, B_{q,\mu\nu}\, \Omega_{q,\mu\nu}}{\omega^2 - \Omega_{q,\mu\nu}^2}$$

### 4.2 Single-window simplification

$\chi^0(0)$ and $\chi^0(i\omega_p)$ are both sign-definite → single Laplace window, $\sim 12$ nodes for $R = E_\text{bw}/E_\text{gap} \sim 30$.

**$\chi^0$ is a transfer-frequency quantity: no $E_F$ enters.** The denominator is $1/(\omega - (E_c - E_v))$ which depends only on eigenvalue differences. Occupancy determines which bands enter, but the quadrature window bounds depend on $\Delta E_\text{gap}$ and $\Delta E_\text{bw}$ — both $E_F$-independent.

At each Laplace node, build conduction and valence propagators, FFT, spin-trace in $R$-space:

$$\tilde{\chi}^0_{\mathbf{R},\mu\nu}(\tau) = \sum_{ab} G^c_{\mathbf{R},ab}(\mu,\nu;\tau) \cdot G^v_{-\mathbf{R},ba}(\nu,\mu;\tau)$$

Both $\chi^0(0)$ and $\chi^0(i\omega_p)$ share the propagator builds — only the scalar weight differs: $\alpha(i\omega_p) = \alpha(0) \cdot e^{-\omega_p \tau}$. Cost $\approx 1\times$ static $\chi^0$.

### 4.3 Extraction

$$\Omega_{q,\mu\nu} = \omega_p \sqrt{\operatorname{Re}\!\left[\frac{\Pi(i\omega_p)}{\Pi(0) - \Pi(i\omega_p)}\right]}, \qquad B_{q,\mu\nu} = -\tfrac{1}{2}\,\Pi(0)\,\Omega$$

Guard: if radicand $< 0$, set $\Omega = 1$ Ha. $\omega_p \sim 0.5$–$1.0$ Ha.

**Output:** $\Omega_{q,\mu\nu} \geq 0$ and $B_{q,\mu\nu}$ for all $q$. PPM time-domain form: $\Pi^\text{PPM}_{q,\mu\nu}(t) = B_{q,\mu\nu}\, e^{-i\Omega_{q,\mu\nu} t}$.

---

## 5. Phase 2: $\Sigma^c(\omega)$ via CTSP

### 5.1 Time-domain factorization

At each quadrature node $t$: build band propagator $G(t)$ and PPM propagator $\Pi^\text{PPM}(t)$, FFT to $R$-space, multiply elementwise ($\Pi$ broadcasts over spin), FFT back, accumulate with frequency kernel.

### 5.2 Window decomposition

Three rectangles in the positive-axis plane $(A, B)$ with $T = |\omega_\text{rel}|_\text{max} + c_\text{edge}\xi$:

| Window | Character |
|--------|-----------|
| Core: $A \in [0,T]$, $B \in [0,T]$ | Crossing possible |
| A-stripe: $A \in [T, A_\text{max}]$, $B \in [0,T]$ | Sign-definite |
| B-slab: $A \in [0, A_\text{max}]$, $B \in [T, B_\text{max}]$ | Sign-definite |

### 5.3 Handling $\omega$ above and below $E_F$

`convolve_frequencies` splits the input $\omega$ array at $E_F$ and processes each half independently:

**$\omega_\text{rel} > 0$ (above $E_F$):**
- $\Sigma^{(-)}$: minus-kernel with $\omega_\text{rel} > 0$ → 3 windows (core has crossing)
- $\Sigma^{(+)}$: plus-kernel → single Laplace (always sign-definite)

**$\omega_\text{rel} < 0$ (below $E_F$):**

Rewrite: $\omega_\text{rel} + (h_v + \Omega) = (h_v + \Omega) - |\omega_\text{rel}|$. This is a minus-kernel with "frequency" $= |\omega_\text{rel}|$ and "sum" $= h_v + \Omega$:
- $\Sigma^{(+)}$: minus-kernel in $|\omega_\text{rel}|$ → 3 windows (core has crossing)
- $\Sigma^{(-)}$: plus-kernel in $|\omega_\text{rel}|$ → single Laplace (always sign-definite, since $|\omega_\text{rel}| + \tilde{E}_c + \Omega > 0$)

The structure is perfectly symmetric about $E_F$. The same `convolve_frequencies` machinery handles both halves — it just swaps which term gets the crossing treatment.

### 5.4 $E_F$ cancellation proof

All propagators use $e^{-i\tilde{E}t}$ with $\tilde{E}$ referenced to $E_F$, but $E_F$ cancels in the product. For $\Sigma^{(-)}$:

$$\underbrace{e^{-i(E_c^\text{vac} - E_c^{\min})t}}_{\text{prop A}} \cdot \underbrace{e^{-i(\Omega - \Omega^{\min})t}}_{\text{prop B}} \cdot \underbrace{e^{-i\bigl((E_c^{\min} - E_F) + \Omega^{\min}\bigr)t}}_{\text{gap phase}} \cdot \underbrace{e^{+i(\omega - E_F)t}}_{\text{freq kernel}} = e^{i(\omega - E_c^\text{vac} - \Omega)t}$$

For $\Sigma^{(+)}$:

$$\underbrace{e^{-i(E_v^{\max} - E_v^\text{vac})t}}_{\text{prop A}} \cdot \underbrace{e^{-i(\Omega - \Omega^{\min})t}}_{\text{prop B}} \cdot \underbrace{e^{-i\bigl((E_F - E_v^{\max}) + \Omega^{\min}\bigr)t}}_{\text{gap phase}} \cdot \underbrace{e^{-i(\omega - E_F)t}}_{\text{freq kernel}} = e^{-i(\omega - E_v^\text{vac} + \Omega)t}$$

For $\chi^0$ ($E_c$, $h_v$ pairing): $E_F$ cancels between the two axes, giving $e^{i(\omega - E_c^\text{vac} + E_v^\text{vac})t}$.

In all three cases, $E_F$ drops out of the final physical phase. Only vacuum eigenvalues and $\omega^\text{vac}$ survive. This is why `convolve_frequencies` can take $E_F$ as a single parameter and vacuum eigenvalues as-is.

### 5.5 After all windows

1. Coulomb sandwich: $\Sigma^c = v \cdot [\text{accumulated}] \cdot v$.
2. Add exchange: $\Sigma = \Sigma^x + \Sigma^c(\omega)$.
3. Band projection: $\Sigma_{ij,\mathbf{k}}(\omega) = \sum_{ab,\mu,\nu} \psi^*_{i,a}(r_\mu)\, \Sigma_{\mathbf{k},ab}(\mu,\nu;\omega)\, \psi_{j,b}(r_\nu)$.

### 5.6 Cost

Per window, per node: $O(N_k N_\text{band} N_s^2 N_\mu^2)$ (builds) + $O(N_s^2 N_\mu^2 N_k \log N_k)$ (FFTs) + $O(N_R N_s^2 N_\mu^2)$ (multiply). Total: $O(N_k N^3)$.

---

## 6. Pipeline summary

**Phase 1** (GN-PPM): Single Laplace window, $\sim 12$ nodes. Build $\chi^0(0)$ and $\chi^0(i\omega_p)$ sharing propagators. Dyson-solve to $\Pi$. Extract $\Omega, B$. No $E_F$ dependence.

**Phase 2** ($\Sigma^c(\omega)$): Split $\omega$ at $E_F$. For each half: conduction term and valence term, one gets 3 windows (crossing), the other gets single Laplace. Accumulate, Coulomb-sandwich, add $\Sigma^x$, project to bands.

**Self-consistent iteration:** Update $E_F$ and eigenvalues. Repeat Phase 1 + Phase 2. $E_F$ enters only through occupancy masks, positive-axis definitions, and $\omega_\text{rel} = \omega - E_F$ — all centralized in `convolve_frequencies`.

---

## 7. Design principles

1. **GN-PPM collapses the pole axis.** $O(N_t)$ poles → one per $(\mu,\nu)$.
2. **Phase 1 is free.** $\sim 12$ Laplace nodes, single window, shared propagators.
3. **Minimax sine replaces Golub–Welsch.** $O(A)$ crossing nodes instead of $O(A^2)$.
4. **Three windows suffice.** Minimal partition separating crossing from non-crossing.
5. **One builder, one integrator.** Laplace vs. crossing: $t$ imaginary vs. real, one $\operatorname{Im}[\cdots]$ projection.
6. **$E_F$ is a parameter, not baked in.** Vacuum eigenvalues throughout; $E_F$ cancels algebraically in all products.
7. **No analytic continuation.** Real-axis $\Sigma^c(\omega)$ directly.

---

## 8. Companion documents

| Document | Content |
|----------|---------|
| `MINIMAX_CTSP_IMPLEMENTATION.md` | Code-level: `convolve_frequencies`, builders, routing, $E_F$ handling |
| `PHYSICS_COMPREHENSIVE.md` | ISDF fitting, Galerkin equations, sharding, static COHSEX |
| `crossing_minimax_overview.md` | Minimax sine-sum algorithm, error scaling |
| `chi_omega_quadrature.md` | CTSP quadrature, GL/HGL derivations, complex-frequency extension |
| `GN-PPM_GUIDELINES.md` | GN-PPM in ISDF, $\omega_p$ selection, failure conditions |
