# Implementation guide: hybrid stochastic + CJ-Ritz pseudobands

Everything below is per k-point; add a leading $N_k$ axis to all per-k arrays, and $\mathtt{vmap}$/$\mathtt{pmap}$ over k. The construction is written for the conduction sector above $E_F$. The valence sector is identical applied to $-H$ with energies reflected about $E_F$.

## Overview

The spectrum is partitioned into three regions:

```
  ┌────────────────┬──────────────────────┬───────────────────────────┐
  │   Protected    │     Stochastic       │        CJ-Ritz           │
  │   (Davidson)   │  (random-phase from  │   (Chebyshev-Jackson     │
  │   weight = 1   │   exact eigenstates) │    filtered Galerkin)     │
  │                │   weight = √(n/k)    │   weight = √(n_eff/k)    │
  ├────────────────┼──────────────────────┼───────────────────────────┤
  E_min        E_cross              E_det_max + margin           E_max
```

- **Protected**: Davidson eigenstates below the first window boundary, included as-is with weight 1. These are the states of interest for QP properties plus enough conduction bands for accurate exchange.

- **Stochastic**: Low-energy windows where the CJ filter cannot resolve the narrow spectral features. Uses random-phase linear combinations of exact eigenstates from extended Davidson (Eq. 5 of Altman et al.). Requires running Davidson deeper than the protected region to provide eigenstates for the transition zone.

- **CJ-Ritz**: High-energy windows where the CJ filter is reliable. Uses the Chebyshev-Jackson filtered Galerkin-Ritz extraction (§4–6 below).

The transition from stochastic to CJ is determined automatically: any window containing ≥1 Davidson eigenstate uses stochastic construction; the rest use CJ-Ritz.

## 1. Inputs and notation

User-supplied scalars:

- $\mathcal{F}$: window ratio $\Delta_j/\bar\epsilon_j$. Default $0.10$.
- $k$: Galerkin block size per window. Default $6$.
- $M_{\max}$: Chebyshev order cap. Default $1500$.
- $C_m$: filter sharpness constant. Default $1.0$ (corresponds to magnification $m \approx 3$).
- $N_{\text{KPM}}$: KPM moment count. Default $500$.
- $N_z$: KPM trace probes. Default $10$.

User-supplied arrays from the Davidson stage:

- $\Phi_{\text{det}} \in \mathbb{C}^{N_G \times N_{\text{det}}}$: deterministic eigenvectors. Should extend well past $\epsilon_{\text{cross}}$ to cover the stochastic transition zone (typically $1.5$–$2\times$ the number of protected bands).
- $E_{\text{det}} \in \mathbb{R}^{N_{\text{det}}}$: their eigenvalues.
- $\mathtt{H}: \mathbb{C}^{N_G \times m} \to \mathbb{C}^{N_G \times m}$: matrix-free Hamiltonian application to a block of $m$ vectors.

We use $\epsilon = E - E_F$ for energies measured from the Fermi level.

## 2. Spectral rescaling and KPM density of states

Run $\sim 30$ Lanczos iterations to estimate $E_{\min}, E_{\max}$. Pad by $\pm 2\%$ to ensure strict containment in the rescaled interval. Define the bandwidth $B = E_{\max} - E_{\min}$, center $E_c = (E_{\max} + E_{\min})/2$, and the rescaled operator
$$
\tilde H = \frac{2}{B}(H - E_c\, I), \qquad \tilde E(E) = \frac{2(E - E_c)}{B} \in [-1, 1].
$$
Compute the KPM moments
$$
\mu_n = \frac{1}{N_z N_G}\sum_{z=1}^{N_z} r_z^\dagger T_n(\tilde H) r_z, \qquad n = 0, \ldots, N_{\text{KPM}}-1,
$$
with $r_z \in \mathbb{C}^{N_G}$ drawn i.i.d. complex-Gaussian, via a single Chebyshev recurrence on the block $R_{\text{KPM}} \in \mathbb{C}^{N_G \times N_z}$. Apply Jackson dampers
$$
g_n^{M} = \frac{(1 - \tfrac{n}{M+2})\sin\alpha_M\cos(n\alpha_M) + \tfrac{1}{M+2}\cos\alpha_M\sin(n\alpha_M)}{\sin\alpha_M},\quad \alpha_M = \frac{\pi}{M+2},
$$
with $M = N_{\text{KPM}}$. The DOS as a function of physical energy is
$$
\tilde\rho(E) = \frac{2 N_G/B}{\pi\sqrt{1 - \tilde E(E)^2}}\!\left[\mu_0 + 2\sum_{n=1}^{N_{\text{KPM}}-1} g_n^{N_{\text{KPM}}}\mu_n\, T_n(\tilde E(E))\right],
$$
evaluated on a dense $1$D grid $E_{\text{grid}} \in \mathbb{R}^{N_{\text{grid}}}$ ($N_{\text{grid}} \sim 10^4$ is plenty).

New arrays: $\mu \in \mathbb{R}^{N_{\text{KPM}}}$, $\tilde\rho_{\text{grid}} \in \mathbb{R}^{N_{\text{grid}}}$.

## 3. Window partitioning

Compute the CJ filter resolution and crossover energy:
$$
\delta E_{\text{CJ}} = \frac{\pi B}{M_{\max}}, \qquad
\epsilon_{\text{cross}} = \frac{C_m\, \delta E_{\text{CJ}}}{\mathcal F}.
$$

The window start $E_{\text{start}}$ is set to $E_F + \epsilon_{\text{cross}}$. This defines the first window boundary and the edge of the protected region.

### DOS-weighted partition (default)

Place window boundaries so each window contributes roughly equal error to $\chi^0$. From Altman SI Eq. S36, the per-window error bound (generalized to Galerkin block order $k$) is
$$
\bigl|\mathbb{E}[\mathrm{Err}_j]\bigr| \;\lesssim\; \frac{n_j^{\mathrm{eff}}\,\sigma_j^{2k}}{\bar\epsilon_j^{2k+1}}, \qquad n_j^{\mathrm{eff}} = \int_{w_j} \tilde\rho(E)\,dE,
$$
where $\sigma_j \approx \Delta_j/\sqrt{12}$ is the energy spread within window $j$, and $\tilde\rho(E)$ is the KPM DOS from §2 — already computed, zero extra cost.

Demanding constant error per window gives the implicit boundary rule:
$$
n_j^{\mathrm{eff}} \cdot \left(\frac{\Delta_j}{\sqrt{12}}\right)^{2k} \cdot \bar\epsilon_j^{-(2k+1)} = \tau,
$$
with $\tau$ a single global tolerance. Boundaries are placed by walking up the cumulative DOS:
$$
N(\epsilon) = \int_{\epsilon_{\text{cross}}}^{\epsilon} \tilde\rho(E_F + \epsilon')\,d\epsilon',
$$
finding each $\epsilon_{j+1}$ by 1D root-finding per boundary ($\mathcal{O}(N_S \log N_{\text{grid}})$ total, negligible).

Set $\tau$ either directly (controls absolute error) or via bisection to hit a target $N_S$. Implementation: `solvers.dos.dos_weighted_windows`.

**Why not the geometric rule?** The Jornada $\mathcal{F} = \Delta_j/\bar\epsilon_j = \text{const}$ rule assumes $\rho(E) \propto E^{1/2}$ (3D free electrons) — fine for bulk semiconductors, but wrong for 2D materials ($\rho \approx \text{const}$), molecules (discrete levels), nanocrystals, or any system with pseudopotential-induced DOS structure. The DOS-weighted rule uses the *actual* computed spectrum and is strictly better at no extra cost.

### Geometric partition (fallback)

If the KPM DOS is unavailable, the geometric rule remains as a conservative fallback:
$$
\epsilon_j = (1 + \mathcal F)^j\, \epsilon_0, \qquad j = 0, 1, \ldots, N_S.
$$
Implementation: `solvers.dos.geometric_windows`.

### Common to both

Force $\epsilon_{N_S} = E_{\max} - E_F$. Window $j \in \{1,\ldots,N_S\}$ is $[\epsilon_{j-1}, \epsilon_j]$ with width $\Delta_j = \epsilon_j - \epsilon_{j-1}$.

Convert to rescaled boundaries: $\tilde\epsilon_j = \tilde E(E_F + \epsilon_j)$.

New arrays: $\epsilon_{\text{bnd}} \in \mathbb{R}^{N_S+1}$, $\tilde\epsilon_{\text{bnd}} \in \mathbb{R}^{N_S+1}$.

## 3b. Window classification: stochastic vs CJ

After window boundaries are determined, classify each window by whether exact eigenstates from Davidson are available:

```python
for j in range(N_S):
    n_det_in_window = count(E_det in [boundary[j], boundary[j+1]))
    if n_det_in_window >= 1:
        mode[j] = "stochastic"  # use exact eigenstates
    else:
        mode[j] = "CJ"          # use Chebyshev filter
```

This classification is the key difference from the original all-CJ scheme. The CJ filter has resolution $\delta E_{\text{CJ}} = \pi B / M_{\max}$; narrow windows near the conduction edge where the DOS is sparse will have Gibbs-ringing leakage that dominates the filtered vectors. Using exact eigenstates for those windows entirely bypasses the filter resolution limit.

The deterministic bands are split into:
- **Protected** ($N_{\text{prot}}$ bands): eigenvalues below $E_{\text{start}}$, included in the output as-is with weight 1.0.
- **Available**: eigenvalues above $E_{\text{start}}$, consumed by stochastic pseudoband construction. NOT included as-is in the output (they are compressed into $k$ pseudobands per window with weight $\sqrt{n/k}$).

This ensures no double-counting: each eigenstate appears either as a protected band OR as part of a stochastic pseudoband, never both.

$N_{\text{prot}}$ is fixed at the value from $k=0$ and reused for all k-points to ensure consistent band count in WFN.h5.

## 4. Chebyshev–Jackson boundary-filter coefficients

We construct $N_S + 1$ smoothed cumulative steps,
$$
C_j(E) \;\approx\; \mathbb{1}_{[E_{\min},\, E_F + \epsilon_j]}(E),
$$
each as a Chebyshev–Jackson series of order $M_{\max}$, sharing the same recurrence so that all are computed in one matvec sweep over the random block. Per Chelikowsky Eq. (5), with $a = -1, b = \tilde\epsilon_j$:
$$
\gamma^{(j)}_0 = 1 - \frac{\arccos\tilde\epsilon_j}{\pi}, \qquad
\gamma^{(j)}_n = -\frac{2}{\pi n}\sin\!\bigl(n\arccos\tilde\epsilon_j\bigr),\quad n \ge 1.
$$
Apply the Jackson dampers $g_n^{M_{\max}}$ from §2 to obtain the smoothed coefficients
$$
c^{(j)}_n = g_n^{M_{\max}}\, \gamma^{(j)}_n.
$$
The bandpass window is recovered by telescoping: $w_j(E) = C_j(E) - C_{j-1}(E)$, with coefficient array $c_n^{w_j} = c_n^{(j)} - c_n^{(j-1)}$.

The chosen cap $M_{\max}$ resolves any boundary as long as $\tilde\epsilon_j$ is at least $C_m\pi/M_{\max}$ away from $\pm 1$ — automatically satisfied because $\epsilon_{\text{cross}}$ was picked to make the lowest boundary's required order exactly $M_{\max}$.

New arrays: $c \in \mathbb{R}^{(N_S+1) \times M_{\max}}$ (the per-boundary damped Chebyshev coefficients).

## 5. Telescoping Chebyshev recurrence

Draw a single random block $\Omega \in \mathbb{C}^{N_G \times k}$ with i.i.d. complex-Gaussian entries (real and imaginary parts $\mathcal{N}(0, 1/2)$). The same $\Omega$ is used for every boundary filter — this is what enables one recurrence to populate all windows.

Allocate $N_S + 1$ accumulators $A_j \in \mathbb{C}^{N_G \times k}$ for $j = 0, \ldots, N_S$, and the two-vector recurrence state $(T_{n-2}, T_{n-1}) \in \mathbb{C}^{N_G \times k} \times \mathbb{C}^{N_G \times k}$.

Initialize:
$$
T_0 = \Omega, \qquad T_1 = \tilde H\,\Omega = \frac{2}{B}\bigl(\mathtt{H}(\Omega) - E_c\,\Omega\bigr),
$$
$$
A_j \leftarrow c^{(j)}_0\, T_0 + c^{(j)}_1\, T_1, \qquad j = 0, \ldots, N_S.
$$
Then iterate $n = 2, \ldots, M_{\max}-1$:
$$
T_n = \frac{4}{B}\bigl(\mathtt{H}(T_{n-1}) - E_c\,T_{n-1}\bigr) - T_{n-2},
$$
$$
A_j \mathrel{+}= c^{(j)}_n\, T_n, \qquad j = 0, \ldots, N_S,
$$
discarding $T_{n-2}$ and shifting $(T_{n-1}, T_n) \to (T_{n-2}, T_{n-1})$.

After the recurrence, $A_j \approx C_j(\tilde H)\,\Omega$. Form per-window filtered blocks
$$
Y_j = A_j - A_{j-1} \in \mathbb{C}^{N_G \times k}, \qquad j = 1, \ldots, N_S,
$$
and discard $\{A_j\}$.

Total filter cost: exactly $M_{\max}$ block-matvecs of width $k$, regardless of $N_S$.

Note: the telescoping filter computes ALL $N_S$ windows at once. For stochastic windows the filtered blocks $Y_j$ are simply discarded — a small waste of computation but the cost is dominated by the shared recurrence regardless.

New arrays during recurrence: $A \in \mathbb{C}^{(N_S+1) \times N_G \times k}$, $T_{n-1}, T_{n-2} \in \mathbb{C}^{N_G \times k}$, $\Omega \in \mathbb{C}^{N_G \times k}$. After: $Y \in \mathbb{C}^{N_S \times N_G \times k}$.

## 6a. Stochastic windows: random-phase pseudobands from exact eigenstates

For each stochastic window $j$ (those containing $\ge 1$ Davidson eigenstate):

Let $\{\phi_n\}_{n \in S_j}$ be the $n_j$ Davidson eigenstates with eigenvalues in $[\epsilon_{j-1}, \epsilon_j]$, and $E_{S_j}$ their eigenvalues. Construct $k$ pseudobands as random-phase linear combinations (Altman Eq. 5):
$$
\left|\xi_{j,\alpha}\right\rangle = \frac{1}{\sqrt{n_j}} \sum_{n \in S_j} e^{i\theta_{\alpha,n}} \left|\phi_n\right\rangle, \qquad \alpha = 1, \ldots, k,
$$
with $\theta_{\alpha,n} \sim \text{Uniform}[0, 2\pi)$ drawn independently.

Each pseudoband has unit norm (in expectation). The pseudo-energy is $\bar E_{S_j} = \frac{1}{n_j}\sum_{n \in S_j} E_n$.

The spectral weight is $\sqrt{n_j / k}$, absorbed into the wavefunction:
$$
|\tilde\xi_{j,\alpha}\rangle = \sqrt{n_j/k}\; |\xi_{j,\alpha}\rangle.
$$

**Key properties:**
- Cross-window orthogonality: $\langle \xi_{j,\alpha} | \xi_{j',\beta} \rangle = 0$ in expectation for $j \neq j'$ (eigenstates from different windows are orthogonal), and $\mathcal{O}(1/\sqrt{n_j})$ for same-window pairs $\alpha \neq \beta$.
- Works even when $n_j < k$: the $k$ pseudobands span an $n_j$-dimensional subspace, but the total weight $k \cdot (n_j/k) = n_j$ is correct regardless.
- No matvecs required — purely algebraic construction from known eigenstates.

## 6b. CJ-Ritz windows: Galerkin extraction from filtered blocks

For each CJ window $j$ (those with no Davidson eigenstates), process in ascending order:

**Deflate against ALL deterministic bands** (protected + available):
$$
Y_j \leftarrow Y_j - \Phi_{\text{det}}\bigl(\Phi_{\text{det}}^\dagger Y_j\bigr).
$$

**Cross-window orthogonalization (§7):** deflate against $Q_{j-1}$ from the previous CJ window:
$$
Y_j \leftarrow Y_j - Q_{j-1}\bigl(Q_{j-1}^\dagger Y_j\bigr).
$$

**Economy QR:** $Y_j = Q_j R_j$, with $Q_j \in \mathbb{C}^{N_G \times k}$ orthonormal and $R_j \in \mathbb{C}^{k \times k}$.

**Galerkin matrix:**
$$
\tilde H_j \;=\; Q_j^\dagger\,\mathtt{H}(Q_j) \;\in\; \mathbb{C}^{k \times k}.
$$
This costs one block matvec ($k$ Hamiltonian applications) plus one $k \times N_G \times k$ inner product. Symmetrize $\tilde H_j \leftarrow \tfrac{1}{2}(\tilde H_j + \tilde H_j^\dagger)$ to kill roundoff antihermiticity.

**Diagonalize the small pencil:**
$$
\tilde H_j = S_j\,\mathrm{diag}(\theta_j)\,S_j^\dagger, \qquad S_j \in \mathbb{C}^{k \times k},\;\theta_j \in \mathbb{R}^k.
$$

**Ritz eigenvalue sanity check:** if any $\theta_{j,\alpha}$ falls more than $\delta E_{\text{CJ}}$ outside the window $[\epsilon_{j-1}, \epsilon_j]$, the CJ filter has failed for this window (typically a spectral gap or insufficient resolution). Set the window weight to zero — these pseudobands carry no spectral weight and do not contribute to GW sums.

**Form Ritz pseudobands:**
$$
\Xi_j = Q_j\, S_j \;\in\; \mathbb{C}^{N_G \times k}.
$$

**Window weight from the KPM DOS:**
$$
n_j^{\text{eff}} \;=\; \dim \cdot \int_{w_j} \tilde\rho(E)\, dE,
$$
where the integral is over the window and $\tilde\rho$ is normalized to integrate to 1 (probability density). Each Ritz pseudoband carries weight $\sqrt{n_j^{\text{eff}}/k}$, absorbed into the wavefunction:
$$
|\tilde\xi_{j,\alpha}\rangle \;=\; \sqrt{n_j^{\text{eff}}/k}\;\, |\Xi_{j,\alpha}\rangle, \qquad \alpha = 1, \ldots, k.
$$

New arrays per window: $Q_j \in \mathbb{C}^{N_G \times k}$, $\tilde H_j \in \mathbb{C}^{k \times k}$, $S_j \in \mathbb{C}^{k\times k}$, $\theta_j \in \mathbb{R}^k$, $\Xi_j \in \mathbb{C}^{N_G\times k}$, $n_j^{\text{eff}} \in \mathbb{R}$.

## 7. Cross-window orthogonalization (CJ windows only)

The Jackson smoothing leaks each $w_j$ slightly into $[\epsilon_{j-2}, \epsilon_{j-1}]$ and $[\epsilon_j, \epsilon_{j+1}]$. The leakage at the lower boundary is the only one that matters during sequential processing (the upper-boundary leakage will be projected out when the next window is processed). One pass of block Gram–Schmidt against the immediately preceding CJ window's Ritz block kills it.

Stochastic windows do not participate in cross-window orthogonalization — their pseudobands are orthogonal to all other windows by construction (different eigenstates).

This adds $\mathcal O(k^2 N_G)$ per CJ window, negligible against the filter cost. Storing $Q_{j-1}$ between iterations keeps the working memory bounded by $\mathcal O(k N_G)$ above the $A_j$ accumulators.

The leading consistency check after all windows are processed:
$$
\sum_{j=1}^{N_S} n_j^{\text{eff}} \;\stackrel{?}{=}\; \int_{E_F + \epsilon_{\text{cross}}}^{E_{\max}} \tilde\rho(E)\,dE,
$$
to relative tolerance $10^{-3}$. A larger discrepancy indicates Jackson leakage outside $[\epsilon_{\text{cross}}, E_{\max} - E_F]$ or insufficient KPM resolution; raise $M_{\max}$ or $N_{\text{KPM}}$ accordingly.

## Output assembly

Concatenate the protected Davidson bands and all pseudobands (stochastic + CJ):
$$
\Phi_{\text{out}} = [\Phi_{\text{prot}}\;\;|\;\;\tilde\xi_{1}\;\;|\;\;\cdots\;\;|\;\;\tilde\xi_{N_S}] \in \mathbb{C}^{N_G \times (N_{\text{prot}} + N_S k)},
$$
$$
E_{\text{out}} = [E_{\text{prot}}\;\;|\;\;\bar E_1 \text{ or } \theta_1\;\;|\;\;\cdots\;\;|\;\;\bar E_{N_S} \text{ or } \theta_{N_S}] \in \mathbb{R}^{N_{\text{prot}} + N_S k},
$$
where pseudo-energies are $\bar E_{S_j}$ (mean eigenvalue) for stochastic windows and $\theta_j$ (Ritz eigenvalues) for CJ windows.

Note: $N_{\text{prot}} \le N_{\text{det}}$ since det bands above $E_{\text{start}}$ are consumed by the stochastic construction. $N_{\text{prot}}$ is determined at $k=0$ and fixed for all k-points for WFN format consistency.

This is the drop-in replacement for a Berkeley-GW-style wavefunction file: the GW code consumes it as a list of bands without modification, with the non-unit norm of the pseudoband wavefunctions encoding their spectral weight.

For the valence sector, run the entire procedure on $-H$ with $E_F$ unchanged and energies negated, then concatenate the resulting valence pseudobands at the bottom of the output band list with their original (un-negated) energies.

## Zero-weight band handling

Pseudobands with zero weight (from CJ-0 windows or empty stochastic windows) have all-zero coefficients in the WFN. Downstream ISDF fitting and centroid selection must guard against division by zero when normalizing by band norms:
- `WFNReader.band_norms`: clamp zero norms to 1.0 (not a small epsilon like 1e-30, which amplifies noise to infinity).
- `isdf_fitting`: replace zero norms with 1.0 via `jnp.where` before dividing.
- `get_charge_density`: skip zero-norm bands in the density loop.

## Diagnostic: `dos_cjwindows.py`

The script `scripts/dos_cjwindows.py` plots the KPM DOS (top panel) and all $N_S$ CJ window indicator functions (bottom panel), color-coded by stochastic vs CJ mode, with the protected region shaded. The CJ indicators are reconstructed analytically from boundary coefficients — no matvecs needed beyond the initial DOS computation. Their sum should equal 1 across the window region (partition of unity).
