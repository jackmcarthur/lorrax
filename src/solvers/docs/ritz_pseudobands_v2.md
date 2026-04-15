# Implementation guide: Galerkin–Ritz pseudobands with Gauss-quadrature energies

Per k-point, conduction sector. Valence sector identical on $-H$.

## Overview

Each window outputs $k$ pseudobands. Wavefunction content comes from Galerkin–Ritz on a rank-$k$ filtered subspace; energies and weights come from $k$-point Gauss quadrature for the windowed DOS measure $w_j(E)\tilde\rho(E)dE$. The two are computed independently per window: Ritz gives directions, Gauss gives spectral pole locations.

## 1. Inputs

User scalars:
- $\mathcal F$: window ratio (default $0.10$)
- $k$: pseudobands per window (default $6$)
- $M_{\max}$: Chebyshev order (default $1500$)
- $N_{\text{KPM}} = 500$, $N_z = 10$
- $N_{\text{prot}}$: requested protected bands

Derived:
- $n_{\text{margin}} = \min(16, \lceil 0.2\, N_{\text{prot}}\rceil)$
- $N_{\text{dav}} = N_{\text{prot}} + n_{\text{margin}}$

Davidson outputs $\Phi_{\text{dav}} \in \mathbb C^{N_G \times N_{\text{dav}}}$, $E_{\text{dav}} \in \mathbb R^{N_{\text{dav}}}$.

Operator $\mathtt H: \mathbb C^{N_G \times m} \to \mathbb C^{N_G \times m}$ matrix-free.

## 2. Spectral rescaling and KPM DOS

Lanczos (~30 iters) for $E_{\min}, E_{\max}$, pad $\pm 2\%$. Bandwidth $B$, center $E_c$, $\tilde H = (2/B)(H - E_c I)$. Compute Jackson-damped KPM moments $\mu_n$ from $N_z$ complex-Gaussian probes via block Chebyshev recurrence on $\tilde H$. Reconstruct $\tilde\rho(E)$ on $E_{\text{grid}} \in \mathbb R^{N_{\text{grid}}}$, $N_{\text{grid}} = 10^4$.

Arrays: $\mu \in \mathbb R^{N_{\text{KPM}}}$, $\tilde\rho_{\text{grid}} \in \mathbb R^{N_{\text{grid}}}$.

## 3. Window placement

Place boundaries $\epsilon_0 < \epsilon_1 < \cdots < \epsilon_{N_S}$ by walking up the cumulative DOS. Each window must contain at least $n_{\min} = k$ states; subject to that, equalize the per-window error
$$
\int_{\epsilon_{j-1}}^{\epsilon_j}\!\tilde\rho\,dE \cdot \left(\frac{\Delta_j}{\sqrt{12}}\right)^{2k} \cdot \bar\epsilon_j^{-(2k+1)} = \tau,
$$
with $\tau$ set globally so $\epsilon_{N_S} = E_{\max} - E_F$. The $\ge n_{\min}$ floor absorbs spectral gaps and prevents pathologically narrow windows.

First boundary: $\epsilon_0 = E_F + (\text{energy of}~N_{\text{prot}}~\text{th Davidson state}) - E_F$ (i.e., the protected region runs from $E_F$ to the $N_{\text{prot}}$-th Davidson eigenvalue; window 1 starts at $\epsilon_0$).

Convert to rescaled boundaries: $\tilde\epsilon_j = 2(E_F + \epsilon_j - E_c)/B$.

Arrays: $\epsilon_{\text{bnd}} \in \mathbb R^{N_S+1}$, mode flag $\in \{\text{dav}, \text{cj}\}^{N_S}$.

## 4. Davidson/CJ classification

Window $j$ is **Davidson** iff $\epsilon_j \le E_{\text{dav}}^{\max} - E_F$ (entirely covered by Davidson run); else **CJ**. Typically 1–3 Davidson windows above the protected region, the rest CJ.

## 5. Shifted CJ boundary cumulatives

Boundary shift $\delta = \pi/(2 M_{\max})$ in rescaled units enforces $\sum_j w_j^2 \approx 1$. Each shared internal boundary $\tilde\epsilon_j$ generates two cumulative steps at $\tilde\epsilon_j \pm \delta$. Outer boundaries $\tilde\epsilon_0, \tilde\epsilon_{N_S}$ unshifted.

Total: $2 N_S^{\text{CJ}}$ cumulatives. Coefficients for cumulative at rescaled position $b$:
$$
\gamma_0 = 1 - \frac{\arccos b}{\pi}, \qquad \gamma_n = -\frac{2}{\pi n}\sin(n\arccos b),\;n\ge 1,
$$
damped by Jackson $g_n^{M_{\max}}$. Window $j$ filter: $w_j(E) = C(\tilde\epsilon_j + \delta) - C(\tilde\epsilon_{j-1} - \delta)$.

Arrays: $c \in \mathbb R^{2N_S^{\text{CJ}} \times M_{\max}}$.

## 6. Filter recurrence

Draw block $\Omega \in \mathbb C^{N_G \times k}$ with random PW phases:
$$
\Omega[G, \alpha] = e^{2\pi i\theta_{G,\alpha}}, \qquad \theta_{G,\alpha} \sim \text{Uniform}[0, 2\pi).
$$
Allocate $2 N_S^{\text{CJ}}$ accumulators $A_b \in \mathbb C^{N_G \times k}$. Three-term recurrence on $\tilde H$ for $n = 0, \ldots, M_{\max}-1$, accumulating $A_b \mathrel{+}= c_n^{(b)} T_n$. Cost: $M_{\max}$ block matvecs of width $k$.

Per CJ window: $Y_j = A_{b_j^+} - A_{b_{j-1}^-} \in \mathbb C^{N_G \times k}$.

## 7. Ritz vectors per window

**Davidson windows** ($\Phi_j = \Phi_{\text{dav}}[:, S_j]$, the $n_j$ eigenstates with $E_n \in [\epsilon_{j-1}, \epsilon_j]$):
- If $n_j \le k$: use Davidson eigenstates directly as the $n_j$ Ritz vectors (pad output to $k$ slots with zero-weight bands; see §9).
- If $n_j > k$: random-phase project to rank $k$. Draw $R_j \in \mathbb C^{n_j \times k}$ with random phases, set $Z_j = \Phi_j R_j$, QR to get $Q_j \in \mathbb C^{N_G \times k}$. Galerkin matrix $\tilde H_j = (\Phi_j^\dagger Q_j)^\dagger \mathrm{diag}(E_{\text{dav}}[S_j])(\Phi_j^\dagger Q_j) \in \mathbb C^{k \times k}$ — no matvec, pure linear algebra on stored eigenvalues. Diagonalize: $\tilde H_j = S_j \mathrm{diag}(\theta_j^{\text{Ritz}}) S_j^\dagger$. Ritz vectors $|q_{j,\alpha}\rangle = (Q_j S_j)[:, \alpha]$.

**CJ windows:**
- Deflate filtered block against all Davidson states: $Y_j \leftarrow Y_j - \Phi_{\text{dav}}(\Phi_{\text{dav}}^\dagger Y_j)$.
- Economy QR: $Y_j = Q_j R_j$, $Q_j \in \mathbb C^{N_G \times k}$.
- Galerkin: $\tilde H_j = Q_j^\dagger \mathtt H(Q_j)$ (one block matvec of width $k$). Symmetrize. Diagonalize: $\tilde H_j = S_j \mathrm{diag}(\theta_j^{\text{Ritz}}) S_j^\dagger$. Ritz vectors $|q_{j,\alpha}\rangle = (Q_j S_j)[:, \alpha]$.

The Ritz energies $\theta_j^{\text{Ritz}}$ are discarded after sorting — only their *order* is used for pairing in §8. Ritz vectors $|q_{j,\alpha}\rangle$ are kept.

Arrays per window: $Q_j \in \mathbb C^{N_G \times k}$, Ritz vectors $\Xi_j \in \mathbb C^{N_G \times k}$.

## 8. Gauss-quadrature energies and weights

Per window $j$, compute the spectral measure moments
$$
m_n^{(j)} = \int_{E_{\min}}^{E_{\max}} E^n\, w_j(E)\, \tilde\rho(E)\, dE, \qquad n = 0, 1, \ldots, 2k-1,
$$
via trapezoidal quadrature on $E_{\text{grid}}$. For Davidson windows where $w_j$ is the hard window indicator, replace the integral with the discrete sum $m_n^{(j)} = \sum_{n \in S_j} E_n^n$ using the exact Davidson eigenvalues.

Construct the $k$-point Gauss quadrature for the measure via the Chebyshev-modified-moment / Wheeler algorithm (numerically stable for $k$ up to ~20; power moments are fine for $k \le 6$ but use modified moments for robustness). Output: nodes $\theta_\alpha^{(j)} \in \mathbb R^k$ and weights $w_\alpha^{(j)} \in \mathbb R^k$ with $\sum_\alpha w_\alpha^{(j)} = m_0^{(j)} = n_j^{\text{eff}}$.

Sort both $\{\theta^{(j)}_{\text{Ritz},\alpha}\}$ and $\{\theta^{(j)}_{\text{Gauss},\alpha}\}$ ascending. Pair the $\alpha$-th Ritz vector with the $\alpha$-th Gauss node:
$$
\text{output band } (j,\alpha): \quad |q_{j,\alpha}\rangle, \quad \theta_\alpha^{(j)} = \theta_{\text{Gauss},\alpha}^{(j)}, \quad \text{weight} = \sqrt{w_\alpha^{(j)}}.
$$

Cost per window: $2k$ scalar integrals on $E_{\text{grid}}$ + $\mathcal O(k^3)$ small linear algebra. Negligible.

Arrays per window: $\theta^{(j)} \in \mathbb R^k$, $w^{(j)} \in \mathbb R^k$.

## 9. Output assembly

$$
\Phi_{\text{out}} = [\Phi_{\text{prot}} \mid \tilde\xi_1 \mid \cdots \mid \tilde\xi_{N_S}], \qquad |\tilde\xi_{j,\alpha}\rangle = \sqrt{w_\alpha^{(j)}}\, |q_{j,\alpha}\rangle,
$$
$$
E_{\text{out}} = [E_{\text{prot}} \mid \theta^{(1)} \mid \cdots \mid \theta^{(N_S)}].
$$

Total band count: $N_{\text{prot}} + N_S k$, fixed across k-points (force at $k=0$).

For Davidson windows with $n_j < k$: only $n_j$ valid Gauss nodes exist; pad with $k - n_j$ zero-weight bands ($w_\alpha = 0$, $\theta_\alpha = \bar\epsilon_j$, $|q_\alpha\rangle = 0$). Downstream code clamps zero norms to 1.0 in division.

For valence sector: rerun on $-H$, concatenate with un-negated energies.

## 10. Diagnostics

- **Spectrum coverage:** $\sum_j n_j^{\text{eff}} = \int_{\epsilon_0}^{E_{\max}-E_F}\tilde\rho\,dE$ to $10^{-3}$.
- **Quadratic POU:** $\sum_{j \in \text{CJ}}w_j(E)^2 = 1 \pm 10^{-4}$ on $E_{\text{grid}}$ (verifies shifted CJ boundaries).
- **Gauss node containment:** all $\theta_\alpha^{(j)} \in [\epsilon_{j-1}, \epsilon_j]$ by construction (Gauss nodes lie in the support of the measure). Failure indicates moment-computation bug.
- **Ritz/Gauss energy alignment:** for well-converged windows, $|\theta_{\text{Ritz},\alpha}^{(j)} - \theta_{\text{Gauss},\alpha}^{(j)}| \lesssim \Delta_j/k$. Large discrepancies flag windows where the random subspace poorly represented the true spectrum; informational only, doesn't affect output.