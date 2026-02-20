---
title: "FEAST Pseudopoles for BSE: Implementation with a Black-Box Mat-Vec"
author: Louie Group
date: 2026
---

## The problem

Within TDA, the two-particle correlation function $L$ has poles at the exciton energies $\Omega_S$ with residues built from the exciton wavefunctions $\Psi^S_{cv\boldsymbol{k}}$:

$$L(\omega) = \sum_S \frac{|\Psi_S\rangle\langle\Psi_S|}{\omega - \Omega_S}, \qquad H_{\mathrm{BSE}}|\Psi_S\rangle = \Omega_S|\Psi_S\rangle$$

Full diagonalization costs $\mathscr{O}((N_v N_c N_k)^3)$ and stores $\mathscr{O}((N_v N_c N_k)^2)$. Both are prohibitive for large $N_k$.

We have access to $H_{\mathrm{BSE}}|X\rangle$ for any trial vector $X \in \mathbb{C}^{N_v N_c N_k}$ via the ISDF mat-vec at cost $\mathscr{O}(N_\mu^2 N_k \log N_k)$. The goal is to construct a compressed rational approximation to $L(\omega)$ using only this mat-vec — never forming or storing $H_{\mathrm{BSE}}$.

Partition the spectrum into energy windows. In each window $w$ containing $N_{S,w}$ exciton states, approximate using $R_w \ll N_{S,w}$ pseudopoles:

$$L_w(\omega) \approx \frac{N_{S,w}}{R_w}\sum_{m=1}^{R_w} \frac{|\tilde{\Psi}_m\rangle\langle\tilde{\Psi}_m|}{\omega - \tilde{\Omega}_m}$$

## What is a spectral filter

A spectral filter is a function $f(E)$ that is $\approx 1$ inside a target energy window $[a, b]$ and $\approx 0$ outside. Applied to the Hamiltonian as $f(H_{\mathrm{BSE}})$, it acts on each eigencomponent of a vector $|X\rangle = \sum_S c_S |\Psi_S\rangle$ by:

$$f(H_{\mathrm{BSE}})|X\rangle = \sum_S f(\Omega_S)\, c_S\, |\Psi_S\rangle \approx \sum_{S \in [a,b]} c_S\, |\Psi_S\rangle$$

Starting from a random vector, $f(H_{\mathrm{BSE}})|X\rangle$ is a random linear combination of eigenstates in the window — exactly what we need for the Rayleigh-Ritz step that produces pseudopoles.

Two classes of filter:

- **Polynomial:** $f(E) = \sum_p c_p\, T_p(E)$ (Chebyshev expansion). Applied via the three-term recurrence — only mat-vecs needed. This is the default.
- **Rational:** $f(E) = \sum_j w_j/(z_j - E)$. Each pole requires solving a shifted linear system $(z_j I - H)|Y\rangle = |X\rangle$. Harder to implement, but far more efficient.

## Chebyshev filters and their cost

The standard polynomial approach: expand $f(E) \approx \Theta(E - a)\Theta(b - E)$ in Chebyshev polynomials with Jackson damping to suppress Gibbs oscillations. Apply via the three-term recurrence $|t_{p+1}\rangle = 2\tilde{H}|t_p\rangle - |t_{p-1}\rangle$ where $\tilde{H} = (H - \bar{E})/(\Delta E_{\mathrm{tot}}/2)$ maps the full spectrum into $[-1,1]$.

The filter order needed to resolve a window of width $\Delta E$ within total bandwidth $B$:

$$N_C \gtrsim \frac{2B}{\Delta E}$$

For $H_{\mathrm{BSE}}$ with $B \sim 40$ eV and a target window of $\Delta E = 0.5$ eV: $N_C \sim 160$. Each of $R$ random vectors requires $N_C$ mat-vecs, giving $R \times 160$ mat-vecs per window.

The polynomial filter must represent the step function using only powers of $H$. All eigencomponents — including those hundreds of Ritz vectors away from the window — participate in the recurrence at every step. The information per mat-vec is diluted by the full bandwidth.

## Why rational filters are cheaper

A rational filter $f(E) = \sum_{j=1}^{n} w_j/(z_j - E)$ can approximate the step function using far fewer terms than a polynomial, because poles in the complex plane provide exponentially better spectral concentration.

Quantitatively (Zolotarev, 1877): the optimal $n$-pole rational approximation to a step function on two intervals separated by a gap achieves error $\sim e^{-c\, n\, \pi^2/\log\kappa}$, versus $\sim e^{-c\, n/\kappa}$ for the best degree-$n$ polynomial. Here $\kappa = B/\Delta E$.

For $\kappa = 80$ ($B = 40$ eV, $\Delta E = 0.5$ eV) and tolerance $10^{-3}$: the polynomial needs degree $\sim 160$; the rational function needs $\sim 4$--$6$ poles.

The cost per pole: one linear solve $(z_j I - H_{\mathrm{BSE}})|Y_j\rangle = |X\rangle$. Unlike the polynomial recurrence, this solve concentrates work near the shift $z_j$ — eigencomponents far from $z_j$ are suppressed by $1/|E - z_j|$, and a preconditioner can make the solve essentially independent of $B$.

Total cost per filtered vector: $n_{\mathrm{poles}} \times n_{\mathrm{inner}}$ mat-vecs, where $n_{\mathrm{inner}}$ is the iteration count of the linear solver. With a good preconditioner: $\sim 4 \times 3 = 12$ mat-vecs, versus $\sim 160$ for Chebyshev.

## FEAST: systematic rational filtering

The FEAST algorithm (Polizzi, 2009) constructs a rational spectral filter from the contour integral representation of the spectral projector:

$$\hat{P}_{[a,b]} = \frac{1}{2\pi i}\oint_\Gamma (zI - H_{\mathrm{BSE}})^{-1}\, dz$$

where $\Gamma$ is a contour in the complex plane enclosing $[a, b]$. Discretize with $n_{\mathrm{quad}}$ quadrature nodes $\{z_j\}$ and weights $\{w_j\}$:

$$\hat{P}_{[a,b]} |X\rangle \approx \sum_{j=1}^{n_{\mathrm{quad}}} w_j\, (z_j I - H_{\mathrm{BSE}})^{-1}|X\rangle$$

Each term is a linear solve at complex shift $z_j$. The contour nodes are the rational filter poles; the quadrature weights are the residues. The integral representation guarantees that the filter is $\approx 1$ inside the contour and $\approx 0$ outside, with exponential accuracy in $n_{\mathrm{quad}}$.

Standard FEAST iterates to converge exact eigenpairs with $R \geq N_S$. We skip the iteration and use $R \ll N_S$ — the filtered subspace directly feeds Rayleigh-Ritz to produce pseudopoles.

## Spectral bounds

Window allocation requires $[E_{\min}, E_{\max}]$ of $H_{\mathrm{BSE}}$.

$E_{\min} = \min_{cv\boldsymbol{k}}(\epsilon_{c\boldsymbol{k}} - \epsilon_{v\boldsymbol{k}})$ is known from the SCF. For $E_{\max}$, run $p \sim 20$ Lanczos steps starting from a random $|X_0\rangle$:

$$\beta_{j+1}|X_{j+1}\rangle = H_{\mathrm{BSE}}|X_j\rangle - a_j|X_j\rangle - \beta_j|X_{j-1}\rangle, \qquad a_j = \langle X_j|\,H_{\mathrm{BSE}}|X_j\rangle$$

The extremal eigenvalues of the resulting $p \times p$ tridiagonal matrix converge geometrically to $E_{\min}$ and $E_{\max}$. Cost: $p$ applications of $H_{\mathrm{BSE}}|\cdot\rangle$. Each step requires one mat-vec plus one distributed inner product ($\mathrm{MPI\_Allreduce}$) for the Lanczos coefficient $a_j$. The tridiagonal is replicated on every processor and trivially diagonalized.

Pad: $E_{\max} \leftarrow 1.05\, E_{\max}^{(\mathrm{Lanczos})}$.

## Preconditioner

Each FEAST linear solve $(z_j I - H_{\mathrm{BSE}})|Y\rangle = |X\rangle$ is solved iteratively. Convergence speed depends on having a good preconditioner — an operator $M$ that is cheap to invert and approximates $H_{\mathrm{BSE}}$.

The diagonal of $H_{\mathrm{BSE}} = D + 2V_A - W_A$ in the pair basis:

$$M(cv\boldsymbol{k}) = \epsilon_{c\boldsymbol{k}} - \epsilon_{v\boldsymbol{k}} + 2V_{cv\boldsymbol{k},cv\boldsymbol{k}} - W_{cv\boldsymbol{k},cv\boldsymbol{k}}$$

Extract the kernel diagonals from ISDF ($\boldsymbol{q} = 0$ for $W$ since $\boldsymbol{k} = \boldsymbol{k}^\prime$):

$$V_{cv\boldsymbol{k},cv\boldsymbol{k}} = \frac{1}{N_k}\sum_{\mu\nu} \psi^*_{c\boldsymbol{k}}(\hat{\boldsymbol{r}}_\mu)\,\psi_{v\boldsymbol{k}}(\hat{\boldsymbol{r}}_\mu)\;\widetilde{V}_{A,\mu\nu}\;\psi^*_{v\boldsymbol{k}}(\hat{\boldsymbol{r}}_\nu)\,\psi_{c\boldsymbol{k}}(\hat{\boldsymbol{r}}_\nu)$$

$$W_{cv\boldsymbol{k},cv\boldsymbol{k}} = \frac{1}{N_k}\sum_{\mu\nu} |\psi_{c\boldsymbol{k}}(\hat{\boldsymbol{r}}_\mu)|^2\;\widetilde{W}_{0,\mu\nu}\;|\psi_{v\boldsymbol{k}}(\hat{\boldsymbol{r}}_\nu)|^2$$

Each element costs $\mathscr{O}(N_\mu^2)$; compute once for all pairs. Each entry is local to the owning processor. The preconditioner action $(M(cv\boldsymbol{k}) - z)^{-1}|r\rangle$ is elementwise division — zero communication.

## Why this preconditioner works

The true exciton energies $\Omega_S$ are perturbative corrections to the diagonal values $M(cv\boldsymbol{k})$: the off-diagonal kernel elements mix electron-hole pairs but shift eigenvalues by $\sim W_{\mathrm{off}} \sim 0.1$--$1$ eV (the exciton binding energy scale).

The eigenvalues of the preconditioned shifted operator $(M - z)^{-1}(H_{\mathrm{BSE}} - z)$ are:

$$\lambda_n = \frac{\Omega_n - z}{M_n - z} = 1 + \frac{\Omega_n - M_n}{M_n - z}$$

For states far from the shift ($|M_n - z| \gg W_{\mathrm{off}}$), $\lambda_n \approx 1$: the preconditioner eliminates them. The condition number depends only on nearby states:

$$\kappa_{\mathrm{prec}} \sim \frac{W_{\mathrm{off}}}{|\mathrm{Im}(z)|}$$

This is independent of the total bandwidth $B$ — the FEAST shifts live off the real axis, and $|\mathrm{Im}(z)| \sim 0.1$--$0.5$ eV for typical contour parameters. With $W_{\mathrm{off}} \sim 0.5$ eV: $\kappa_{\mathrm{prec}} \sim 1$--$5$, so the iterative solver converges in $\sim 2$--$4$ steps.

## Contour parameterization

Parameterize $\Gamma$ as an ellipse with foci at the window edges $a, b$:

$$z_j = \frac{a+b}{2} + r_x \cos\theta_j + i\, r_y \sin\theta_j, \qquad \theta_j = \frac{\pi(2j - 1)}{2n_{\mathrm{quad}}}$$

where $r_x = (b - a)/2$ and $r_y = \gamma\, r_x$. The $\theta_j$ are midpoint-rule nodes on $[0, \pi]$ (upper half-plane only). Quadrature weights:

$$w_j = \frac{1}{n_{\mathrm{quad}}}(-r_x \sin\theta_j + i\, r_y \cos\theta_j)$$

The aspect ratio $\gamma$ controls a tradeoff:

- **Smaller $\gamma$** (flatter): shifts closer to real axis, linear solves harder, but filter sharper
- **Larger $\gamma$** (rounder): solves easier, but wider transition region and more leakage

The hardest contour point is $\theta_1 \approx \pi/(2n_{\mathrm{quad}})$, where $\mathrm{Im}(z_1) \approx \gamma\,\Delta E\,\pi/(4n_{\mathrm{quad}})$.

## Conjugate symmetry

For Hermitian $H_{\mathrm{BSE}}$: $\overline{(zI - H)^{-1}|X\rangle} = (\bar{z}I - H)^{-1}|\bar{X}\rangle$. The lower half-contour contributes the conjugate of the upper half. For real starting vectors $|X_i\rangle$:

$$|X_i^{\mathrm{filt}}\rangle = \oint_\Gamma \frac{dz}{2\pi i}\,(zI - H_{\mathrm{BSE}})^{-1}|X_i\rangle = 2\,\mathrm{Re}\left[\sum_{j=1}^{n_{\mathrm{quad}}} w_j\, |Y_{ij}\rangle\right]$$

where $(z_j I - H_{\mathrm{BSE}})|Y_{ij}\rangle = |X_i\rangle$. Only $n_{\mathrm{quad}}$ solves per starting vector (upper half-plane), not $2n_{\mathrm{quad}}$. Use real random starting vectors to exploit this.

For non-TDA $H_{\mathrm{BSE}}$ (non-Hermitian), the conjugate trick does not apply — both half-planes require independent solves, doubling the cost.

## Choosing $\gamma$ and $n_{\mathrm{quad}}$

Practical values: $\gamma = 0.4$, $n_{\mathrm{quad}} = 4$.

For $\Delta E = 1$ eV: the worst-case imaginary part is $\mathrm{Im}(z_1) \approx 0.4 \times 0.5 \times \pi/8 \approx 0.08$ eV. With $W_{\mathrm{off}} \sim 0.5$ eV this gives $\kappa_{\mathrm{prec}} \sim 6$, so $n_{\mathrm{inner}} \sim 3$ GMRES iterations.

For $\Delta E = 0.5$ eV: $\mathrm{Im}(z_1) \approx 0.04$ eV, $\kappa_{\mathrm{prec}} \sim 12$, $n_{\mathrm{inner}} \sim 4$.

Filter leakage at distance $\Delta E$ outside the window: $\sim e^{-c\, n_{\mathrm{quad}}/\log(1/\gamma)}$. For $n_{\mathrm{quad}} = 4$, $\gamma = 0.4$: leakage $\sim 10^{-3}$. This is more than sufficient for pseudopoles, since the $1/\sqrt{R}$ stochastic error dominates. Increase to $n_{\mathrm{quad}} = 6$--$8$ only if stricter spectral isolation is needed.

## Preconditioned GMRES

Solve $(z_j I - H_{\mathrm{BSE}})|Y\rangle = |X_i\rangle$ for each contour point $z_j$ and starting vector $|X_i\rangle$.

**Initialize:** $|Y^{(0)}\rangle = (M - z_j)^{-1}|X_i\rangle$, $\quad|r^{(0)}\rangle = |X_i\rangle - (z_j I - H_{\mathrm{BSE}})|Y^{(0)}\rangle$

**Iteration $k = 1, 2, \ldots$:**

1. Precondition: $|\hat{X}_k\rangle = (M - z_j)^{-1}|r^{(k-1)}\rangle$ — elementwise, zero communication
2. Arnoldi step: $|W_k\rangle = (z_j I - H_{\mathrm{BSE}})|\hat{X}_k\rangle$ — one application of $H_{\mathrm{BSE}}|\cdot\rangle$
3. Orthogonalize $|W_k\rangle$ against $\{|W_1\rangle, \ldots, |W_{k-1}\rangle\}$ via modified Gram-Schmidt — $k$ distributed inner products
4. Solve $(k+1) \times k$ Hessenberg least-squares (replicated, trivial)

Terminate when $\|r^{(k)}\|/\|r^{(0)}\| < \tau \sim 10^{-2}$. Tighter tolerance is unnecessary — the $1/\sqrt{R}$ stochastic error dominates.

GMRES works for arbitrary (including non-Hermitian) operators. This is important: it handles both TDA and non-TDA $H_{\mathrm{BSE}}$ without modification. The complex shift $z_j$ makes the operator non-Hermitian even in the TDA case, so GMRES is the natural choice regardless.

## GMRES parallel structure

GMRES is not independent per matrix element — the mat-vec $H_{\mathrm{BSE}}|X\rangle$ couples all $(cv\boldsymbol{k})$ through the off-diagonal kernel. The Krylov subspace is a global object, distributed identically to $|X\rangle$: each processor owns a contiguous chunk of $(cv\boldsymbol{k})$ indices for every Arnoldi basis vector.

Every GMRES operation maps onto existing parallel primitives:

- $H_{\mathrm{BSE}}|X\rangle$: ISDF contractions (local) + FFT convolution for $W$ (neighbor exchange) — the mat-vec you already have
- $(M - z)^{-1}|r\rangle$: elementwise division on local chunk — no communication
- Inner products $\langle W_i | W_k \rangle$: local partial sum, then one $\mathrm{MPI\_Allreduce}$ per inner product
- Vector updates $|Y\rangle \leftarrow |Y\rangle + \alpha|W\rangle$: local AXPY — no communication

No new communication primitives. Memory per proc: $n_{\mathrm{inner}}$ Arnoldi vectors of local length $L_{\mathrm{loc}} \sim N_v N_c N_k / P$.

## Two axes of parallelism

**Axis 1 (within each solve):** Distribute $(cv\boldsymbol{k})$ across $P_{\mathrm{dist}}$ processors — identical to the existing BSE mat-vec communicator.

**Axis 2 (across independent solves):** Per window, there are $R_w \times n_{\mathrm{quad}}$ independent linear systems. Split processors into $P_{\mathrm{solve}}$ sub-communicators of size $P_{\mathrm{dist}}$ each, running independent GMRES instances concurrently.

Wall time per window:

$$T_{\mathrm{wall}} = \left\lceil\frac{R_w\, n_{\mathrm{quad}}}{P_{\mathrm{solve}}}\right\rceil \times n_{\mathrm{inner}} \times T_{\mathrm{mv}}(P_{\mathrm{dist}})$$

For $R_w = 5$, $n_{\mathrm{quad}} = 4$: 20 independent solves. With $P_{\mathrm{solve}} = 20$, the window reduces to $n_{\mathrm{inner}}$ sequential mat-vecs (plus Rayleigh-Ritz). Windows are also independent — a third axis.

Choose $P_{\mathrm{dist}}$ large enough that the mat-vec is not communication-bound; allocate remaining processors to $P_{\mathrm{solve}}$.

## Rayleigh-Ritz extraction

After filtering, we have $R_w$ filtered vectors $\{|X_i^{\mathrm{filt}}\rangle\}$ per window. These span an $R_w$-dimensional subspace concentrated in $[a_w, b_w]$.

Form the $R_w \times R_w$ projected problem:

$$S_{ij} = \langle X_i^{\mathrm{filt}}|X_j^{\mathrm{filt}}\rangle, \qquad \mathcal{H}_{ij} = \langle X_i^{\mathrm{filt}}|\,H_{\mathrm{BSE}}|X_j^{\mathrm{filt}}\rangle$$

$\boldsymbol{S}$: $R_w(R_w+1)/2$ distributed inner products (Hermitian). $\boldsymbol{\mathcal{H}}$: $R_w$ applications of $H_{\mathrm{BSE}}|\cdot\rangle$, then $R_w^2$ inner products. Solve the $R_w \times R_w$ generalized eigenvalue problem:

$$\boldsymbol{\mathcal{H}}\boldsymbol{c}_m = \tilde{\Omega}_m\,\boldsymbol{S}\boldsymbol{c}_m$$

Replicated, $\mathscr{O}(R_w^3)$, trivial. The pseudopoles:

$$\tilde{\Omega}_m = \mathrm{Ritz\ value}, \qquad |\tilde{\Psi}_m\rangle = \sum_{i=1}^{R_w} c_{m,i}\,|X_i^{\mathrm{filt}}\rangle$$

Each proc builds its local chunk of $|\tilde{\Psi}_m\rangle$ from local chunks of $|X_i^{\mathrm{filt}}\rangle$. No communication beyond broadcasting the $R_w \times R_w$ eigenvectors.

## Properties of the Ritz pseudopoles

The $R_w$ Ritz values $\tilde{\Omega}_m$ are the $R_w$-point Gauss quadrature nodes for the spectral measure in the window (Golub-Welsch theorem). They match the first $2R_w$ moments:

$$\sum_{m=1}^{R_w} W_m\, \tilde{\Omega}_m^k = \sum_{n \in w} \langle\psi_n|\cdot\rangle^2\, \Omega_n^k \qquad (k = 0, 1, \ldots, 2R_w - 1)$$

No other placement of $R_w$ poles can match more moments. This is the precise sense in which Ritz pseudopoles are optimal — they inherit the full optimality theory of Gaussian quadrature.

When $R_w \ll N_{S,w}$ (the normal operating regime), each $|\tilde{\Psi}_m\rangle$ is a mixture of many true exciton states — they are not approximate eigenstates but compressed spectral representatives.

When $R_w \geq N_{S,w}$, the Ritz pairs converge to true eigenpairs. The overlap matrix $\boldsymbol{S}$ detects this: eigenvalues of $\boldsymbol{S}$ split into $N_{S,w}$ physical values and $R_w - N_{S,w}$ near-zero values from filter leakage.

## Spectral weight

The total spectral weight in window $w$ is estimated from $\mathrm{Tr}(\boldsymbol{S})$ without counting eigenstates:

$$N_{S,w} \approx \frac{\mathrm{Tr}(\boldsymbol{S}_w) \cdot N_v N_c N_k}{R_w}$$

The correctly normalized two-particle correlator:

$$L_w(\omega) \approx \frac{N_{S,w}}{R_w}\sum_{m=1}^{R_w}\frac{|\tilde{\Psi}_m\rangle\langle\tilde{\Psi}_m|}{\omega - \tilde{\Omega}_m}$$

**Diagnostics from $\boldsymbol{S}$:**

- All eigenvalues $\sim N_{S,w}/(N_v N_c N_k)$: normal regime, Ritz values are pseudopoles
- Eigenvalue spread $> 10\times$: filter leakage — increase $n_{\mathrm{quad}}$ or decrease $\gamma$
- Near-zero eigenvalues: $R_w > N_{S,w}$ — truncate; surviving pairs approximate true eigenpairs

## Error bound

For window $w$ centered at $\bar{E}_w$, width $\Delta E_w$, $R_w$ Ritz poles, evaluated at $\omega + i\eta$. Define $|D_w| = |\omega + i\eta - \bar{E}_w|$.

**Quadrature error (any spectral measure, no smoothness assumptions):**

$$\epsilon_{\mathrm{quad}} \lesssim \frac{N_{S,w}}{\max(\eta,\, |D_w|)} \times \begin{cases} \left(\dfrac{\Delta E_w}{2|D_w|}\right)^{2R_w} & |D_w| > \Delta E_w/2 \\[6pt] \exp\!\left(-\dfrac{4R_w\eta}{\Delta E_w}\right) & |D_w| \leq \Delta E_w/2 \end{cases}$$

The top line follows from $2R_w$ matched moments ($\omega$ outside window). The bottom from the Bernstein ellipse for approximating the Lorentzian by a polynomial of degree $2R_w - 1$, with nearest singularity at distance $\eta$ from the real axis ($\omega$ inside window).

**FEAST filter leakage:** $\sim e^{-c\, n_{\mathrm{quad}}}$. Negligible for $n_{\mathrm{quad}} \geq 4$.

**Stochastic (random subspace):** $\sim 1/\sqrt{R_w}$ in matrix elements.

## Window allocation

Tile $[E_{\min}, E_{\max}]$ with non-uniform windows adapted to target frequencies $\{\omega_i\}$ with broadening $\eta$:

**Tier 1 (target region):** $\Delta E_w \sim R_w\,\eta$ so Ritz spacing $\leq \eta$ resolves the Lorentzian kernel. Use $R_w \sim 5$--$10$, $n_{\mathrm{quad}} = 4$.

**Tier 2 ($5$--$20$ eV away):** $\Delta E_w \sim 2$--$5$ eV, $R_w \sim 3$--$5$, $n_{\mathrm{quad}} = 3$. The $(\Delta E/(2d))^{2R}$ geometric convergence handles the error.

**Tier 3 ($> 20$ eV away):** One or two wide windows. $R_w = 2$--$3$, $n_{\mathrm{quad}} = 2$.

The criterion is not equal eigenstate count but equal error contribution at $\omega$, weighted by $1/|\omega - \bar{E}_w|$ and suppressed by $(\Delta E/(2d))^{2R}$. Distant windows tolerate far greater width.

FEAST contour parameters also vary by tier: $n_{\mathrm{quad}}$ can be smaller for distant tiers since leakage tolerance is higher and $R_w$ is small.

## Cost summary

Per window: $R_w \times n_{\mathrm{quad}}$ linear solves costing $n_{\mathrm{inner}}$ mat-vecs each, plus $R_w$ mat-vecs for Rayleigh-Ritz:

$$C_w = R_w\,(n_{\mathrm{quad}} \cdot n_{\mathrm{inner}} + 1)\;\mathrm{applications\ of\ } H_{\mathrm{BSE}}|\cdot\rangle$$

Concrete example: 40 eV BSE spectrum, targets $\omega \in [1, 5]$ eV, $\eta = 0.1$ eV:

| Tier | $\Delta E$ | Windows | $R_w$ | $n_{\mathrm{quad}}$ | $n_{\mathrm{inner}}$ | Total mat-vecs |
|------|-----------|---------|-------|-----|-----|----------|
| 1 | 0.5 eV | 8 | 5 | 4 | 3 | 520 |
| 2 | 3 eV | 4 | 3 | 3 | 2 | 84 |
| 3 | 15 eV | 2 | 2 | 2 | 2 | 20 |
| | | **14** | | | | **624** |

Chebyshev comparison: tier 1 alone would need $N_C \sim 160$ per vector, giving $\sim 6400$ mat-vecs — $10\times$ more.

## Memory and accumulation

Per-proc storage for one window:

- $R_w$ filtered vectors being accumulated: $R_w \times L_{\mathrm{loc}}$
- $n_{\mathrm{inner}}$ Arnoldi basis vectors for the active GMRES solve: $n_{\mathrm{inner}} \times L_{\mathrm{loc}}$
- Current solution + residual + preconditioned direction: $3 \times L_{\mathrm{loc}}$

Total: $(R_w + n_{\mathrm{inner}} + 3) \times L_{\mathrm{loc}}$ complex doubles. For $R_w = 5$, $n_{\mathrm{inner}} = 3$: 11 vectors per proc.

After each contour-point solve, accumulate $|X_i^{\mathrm{filt}}\rangle \mathrel{+}= 2\,\mathrm{Re}[w_j\,|Y_{ij}\rangle]$ and discard $|Y_{ij}\rangle$. This avoids storing the full $R_w \times n_{\mathrm{quad}}$ solution array.

The $R_w$ output pseudopole vectors per window must persist for downstream use in $\Sigma$ or $\chi$.

## Full algorithm

**Startup ($\sim 20$ mat-vecs + $\mathscr{O}(N_\mu^2 N_v N_c N_k)$):**

1. Lanczos $\to E_{\max}$
2. Build $M(cv\boldsymbol{k})$ from ISDF diagonal elements
3. Choose windows $\{[a_w, b_w]\}$, allocate $\{R_w, n_{\mathrm{quad},w}\}$, compute contour points $\{z_j^{(w)}, w_j^{(w)}\}$

**Per window $w$ (embarrassingly parallel across windows):**

4. Generate $R_w$ real random vectors $\{|X_i\rangle\}$; set $|X_i^{\mathrm{filt}}\rangle = 0$
5. For $j = 1, \ldots, n_{\mathrm{quad},w}$:
    - For $i = 1, \ldots, R_w$: solve $(z_j I - H_{\mathrm{BSE}})|Y_{ij}\rangle = |X_i\rangle$ via preconditioned GMRES
    - Accumulate $|X_i^{\mathrm{filt}}\rangle \mathrel{+}= 2\,\mathrm{Re}[w_j\,|Y_{ij}\rangle]$; discard $|Y_{ij}\rangle$
6. Form $\boldsymbol{S}$, $\boldsymbol{\mathcal{H}}$ ($R_w$ applications of $H_{\mathrm{BSE}}|\cdot\rangle$ + inner products)
7. Solve $R_w \times R_w$ generalized eigenvalue problem $\to \{(\tilde{\Omega}_m, |\tilde{\Psi}_m\rangle)\}$
8. Store $N_{S,w}$ from $\mathrm{Tr}(\boldsymbol{S})$

**Output:** $\{(\tilde{\Omega}_m, |\tilde{\Psi}_m\rangle, N_{S,w}/R_w)\}$ for all windows.
