# ISDF Galerkin Derivation with Spin Indices

This document derives the correct CCT and ZCT matrices for ISDF fitting of pair products with explicit spin treatment.

## 1. The ISDF Approximation

For GW calculations, we need to approximate products of wavefunctions:

$$
\phi_{mn,k,ab}(\mathbf{r}) = \psi^*_{m,\mathbf{k}-\mathbf{q},a}(\mathbf{r}) \, \psi_{n,\mathbf{k},b}(\mathbf{r})
$$

where $a, b \in \{\uparrow, \downarrow\}$ are spin indices, and $m, n$ are band indices.

The ISDF ansatz approximates these products using a set of interpolation vectors $\zeta_\mu(\mathbf{r})$:

$$
\phi_{mn,k,ab}(\mathbf{r}) \approx \sum_\mu \zeta_\mu(\mathbf{r}) \, \phi_{mn,k,ab}(\mathbf{r}_\mu)
$$

where $\{\mathbf{r}_\mu\}$ are the centroid points.

## 2. Two Fitting Targets

### Option A: Fit all four spin combinations

Fit each $(a,b)$ pair separately with the **same** $\zeta_\mu(\mathbf{r})$:

$$
\phi_{mn,k,ab}(\mathbf{r}) \approx \sum_\mu \zeta_\mu(\mathbf{r}) \, \psi^*_{m,\mathbf{k}-\mathbf{q},a}(\mathbf{r}_\mu) \, \psi_{n,\mathbf{k},b}(\mathbf{r}_\mu)
$$

### Option B: Fit the spin-traced product only

Fit only the spin-diagonal sum:

$$
\phi_{mn,k}(\mathbf{r}) = \sum_s \psi^*_{m,\mathbf{k}-\mathbf{q},s}(\mathbf{r}) \, \psi_{n,\mathbf{k},s}(\mathbf{r}) = \phi_{mn,k,\uparrow\uparrow}(\mathbf{r}) + \phi_{mn,k,\downarrow\downarrow}(\mathbf{r})
$$

These lead to **different** Galerkin conditions.

## 3. Galerkin Condition Derivation

The Galerkin method minimizes the squared error:

$$
\mathcal{E} = \sum_{mn,k,\alpha} \int \left| \phi_{mn,k,\alpha}(\mathbf{r}) - \sum_\mu \zeta_\mu(\mathbf{r}) \, \phi_{mn,k,\alpha}(\mathbf{r}_\mu) \right|^2 \, d\mathbf{r}
$$

where $\alpha$ indexes whatever spin combinations we're fitting.

Taking the functional derivative with respect to $\zeta_\nu(\mathbf{r})$ and setting to zero gives:

$$
\sum_\mu C_{\nu\mu} \, \zeta_\mu(\mathbf{r}) = Z_\nu(\mathbf{r})
$$

where the **CCT matrix** and **ZCT matrix** are:

$$
\boxed{C_{\nu\mu} = \sum_{mn,k,\alpha} \phi^*_{mn,k,\alpha}(\mathbf{r}_\nu) \, \phi_{mn,k,\alpha}(\mathbf{r}_\mu)}
$$

$$
\boxed{Z_\nu(\mathbf{r}) = \sum_{mn,k,\alpha} \phi^*_{mn,k,\alpha}(\mathbf{r}_\nu) \, \phi_{mn,k,\alpha}(\mathbf{r})}
$$

## 4. CCT for Option A: All Four Spin Combinations

With $\alpha = (a,b) \in \{(\uparrow,\uparrow), (\uparrow,\downarrow), (\downarrow,\uparrow), (\downarrow,\downarrow)\}$:

$$
C_{\nu\mu} = \sum_{mn,k,a,b} \phi^*_{mn,k,ab}(\mathbf{r}_\nu) \, \phi_{mn,k,ab}(\mathbf{r}_\mu)
$$

Substituting the definition of $\phi$:

$$
C_{\nu\mu} = \sum_{mn,k,a,b} \left[\psi_{m,\mathbf{k}-\mathbf{q},a}(\mathbf{r}_\nu) \, \psi^*_{n,\mathbf{k},b}(\mathbf{r}_\nu)\right] \left[\psi^*_{m,\mathbf{k}-\mathbf{q},a}(\mathbf{r}_\mu) \, \psi_{n,\mathbf{k},b}(\mathbf{r}_\mu)\right]
$$

**Factorizing** the sums over $m$ and $n$:

$$
C_{\nu\mu} = \sum_{k,a,b} \underbrace{\left[\sum_m \psi_{m,\mathbf{k}-\mathbf{q},a}(\mathbf{r}_\nu) \, \psi^*_{m,\mathbf{k}-\mathbf{q},a}(\mathbf{r}_\mu)\right]}_{= P^*_{\mathbf{k}-\mathbf{q},aa}(\nu,\mu)} \underbrace{\left[\sum_n \psi^*_{n,\mathbf{k},b}(\mathbf{r}_\nu) \, \psi_{n,\mathbf{k},b}(\mathbf{r}_\mu)\right]}_{= P_{\mathbf{k},bb}(\nu,\mu)}
$$

where we define the **pair density matrix**:

$$
P_{\mathbf{k},ss'}(\nu,\mu) = \sum_n \psi^*_{n\mathbf{k},s}(\mathbf{r}_\nu) \, \psi_{n\mathbf{k},s'}(\mathbf{r}_\mu)
$$

Note: $P^*_{\mathbf{k},ss}(\nu,\mu) = P_{\mathbf{k},ss}(\mu,\nu)$ (from hermiticity).

**Result for Option A:**

$$
\boxed{C_{\nu\mu}^{(\text{all }ab)} = \sum_{k,a,b} P^*_{\mathbf{k}-\mathbf{q},aa}(\nu,\mu) \, P_{\mathbf{k},bb}(\nu,\mu)}
$$

This involves the **diagonal** spin elements only: $P_{aa}$ and $P_{bb}$.

## 5. CCT for Option B: Spin-Traced Product

Now fitting only:
$$
\phi_{mn,k}(\mathbf{r}) = \phi_{mn,k,\uparrow\uparrow}(\mathbf{r}) + \phi_{mn,k,\downarrow\downarrow}(\mathbf{r}) = \sum_s \psi^*_{m,\mathbf{k}-\mathbf{q},s}(\mathbf{r}) \, \psi_{n,\mathbf{k},s}(\mathbf{r})
$$

The CCT becomes:

$$
C_{\nu\mu} = \sum_{mn,k} \phi^*_{mn,k}(\mathbf{r}_\nu) \, \phi_{mn,k}(\mathbf{r}_\mu)
$$

Substituting:

$$
C_{\nu\mu} = \sum_{mn,k} \left[\sum_s \psi_{m,\mathbf{k}-\mathbf{q},s}(\mathbf{r}_\nu) \, \psi^*_{n,\mathbf{k},s}(\mathbf{r}_\nu)\right] \left[\sum_{s'} \psi^*_{m,\mathbf{k}-\mathbf{q},s'}(\mathbf{r}_\mu) \, \psi_{n,\mathbf{k},s'}(\mathbf{r}_\mu)\right]
$$

Expanding and factorizing:

$$
C_{\nu\mu} = \sum_{k,s,s'} \underbrace{\left[\sum_m \psi_{m,\mathbf{k}-\mathbf{q},s}(\mathbf{r}_\nu) \, \psi^*_{m,\mathbf{k}-\mathbf{q},s'}(\mathbf{r}_\mu)\right]}_{= P^*_{\mathbf{k}-\mathbf{q},s's}(\nu,\mu)} \underbrace{\left[\sum_n \psi^*_{n,\mathbf{k},s}(\mathbf{r}_\nu) \, \psi_{n,\mathbf{k},s'}(\mathbf{r}_\mu)\right]}_{= P_{\mathbf{k},ss'}(\nu,\mu)}
$$

**Result for Option B:**

$$
\boxed{C_{\nu\mu}^{(\text{spin-trace})} = \sum_{k,s,s'} P^*_{\mathbf{k}-\mathbf{q},s's}(\nu,\mu) \, P_{\mathbf{k},ss'}(\nu,\mu)}
$$

This involves **all four** spin matrix elements: $P_{ss'}$ for $s,s' \in \{\uparrow,\downarrow\}$!

## 6. Key Insight: The Off-Diagonal Spin Elements

The crucial observation is that even when fitting **only** the spin-traced product, the CCT involves **all four** $P_{ss'}$ combinations due to how the Galerkin error factorizes.

Explicitly, $C^{(\text{spin-trace})}$ contains terms like:

$$
P^*_{\uparrow\downarrow}(\nu,\mu) \cdot P_{\uparrow\downarrow}(\nu,\mu) = |P_{\uparrow\downarrow}(\nu,\mu)|^2
$$

These off-diagonal spin elements arise because:

1. The product $\phi \cdot \phi^*$ couples different spin channels through the band sums
2. The cross-terms $\sum_m \psi_{m,s} \psi^*_{m,s'}$ with $s \neq s'$ are generally nonzero for SOC systems

## 7. Comparison of the Two Options

| Quantity | Option A (all ab) | Option B (spin-trace) |
|----------|-------------------|----------------------|
| Fitting target | $\phi_{ab}$ for each $(a,b)$ | $\sum_s \phi_{ss}$ |
| CCT formula | $\sum_{ab} P^*_{aa} P_{bb}$ | $\sum_{ss'} P^*_{s's} P_{ss'}$ |
| Uses $P_{\uparrow\downarrow}$? | No | Yes |
| Uses cross-term $P^*_{\uparrow\uparrow} P_{\downarrow\downarrow}$? | Yes | Yes |

## 8. The User's Original Proposal

The user proposed computing:
$$
C_R(\mu,\nu) = \sum_{ab} |P_{R,ab}(\mu,\nu)|^2 = |P_{\uparrow\uparrow}|^2 + |P_{\uparrow\downarrow}|^2 + |P_{\downarrow\uparrow}|^2 + |P_{\downarrow\downarrow}|^2
$$

This corresponds to **Option A** (fitting all four spin combinations), not Option B (spin-trace only).

For **Option B** (spin-trace fitting), the correct CCT from Section 5 is:
$$
C^{(\text{spin-trace})} = \sum_{ss'} P^*_{s's} P_{ss'} = |P_{\uparrow\uparrow}|^2 + |P_{\downarrow\downarrow}|^2 + P^*_{\downarrow\uparrow} P_{\uparrow\downarrow} + P^*_{\uparrow\downarrow} P_{\downarrow\uparrow} + \ldots
$$

which includes cross-terms like $P^*_{\uparrow\uparrow} P_{\uparrow\downarrow}$, not just absolute values squared.

## 9. Simplified q=0 Case

For $\mathbf{q} = 0$, where $\mathbf{k}-\mathbf{q} = \mathbf{k}$:

**Option A:**
$$
C_{\nu\mu} = \sum_{k,a,b} |P_{\mathbf{k},aa}(\nu,\mu)|^2 = \sum_k \left|\sum_a P_{\mathbf{k},aa}\right|^2 = \sum_k |P_{\uparrow\uparrow} + P_{\downarrow\downarrow}|^2
$$

Wait, this doesn't factor correctly. Let me redo this.

For $\mathbf{q}=0$:
$$
C_{\nu\mu}^{(\text{all }ab)} = \sum_{k,a,b} P^*_{\mathbf{k},aa}(\nu,\mu) \, P_{\mathbf{k},bb}(\nu,\mu) = \sum_k \left(\sum_a P^*_{\mathbf{k},aa}\right)\left(\sum_b P_{\mathbf{k},bb}\right) = \sum_k |P_\uparrow + P_\downarrow|^2
$$

For Option B with $\mathbf{q}=0$:
$$
C_{\nu\mu}^{(\text{spin-trace})} = \sum_{k,s,s'} P^*_{\mathbf{k},s's}(\nu,\mu) \, P_{\mathbf{k},ss'}(\nu,\mu)
$$

Using matrix notation $\mathbf{P}_k$ (2×2 spin matrix):
$$
= \sum_k \text{Tr}(\mathbf{P}^\dagger_k \mathbf{P}_k) = \sum_k ||\mathbf{P}_k||^2_F
$$

This is the **Frobenius norm squared** of the 2×2 spin matrix!

$$
= \sum_k \left(|P_{\uparrow\uparrow}|^2 + |P_{\uparrow\downarrow}|^2 + |P_{\downarrow\uparrow}|^2 + |P_{\downarrow\downarrow}|^2\right)
$$

## 10. Conclusion

**You are correct.** When fitting the spin-traced product $\sum_s \psi^*_s \psi_s$, the Galerkin condition leads to:

$$
\boxed{C_{\nu\mu}^{(\text{spin-trace})} = \sum_{k,s,s'} P^*_{\mathbf{k}-\mathbf{q},s's}(\nu,\mu) \, P_{\mathbf{k},ss'}(\nu,\mu)}
$$

which for $\mathbf{q}=0$ simplifies to:

$$
C_{\nu\mu} = \sum_k \sum_{s,s'} |P_{\mathbf{k},ss'}(\nu,\mu)|^2 = \sum_k \sum_{ab} |P_{ab}|^2
$$

This is exactly the formula you originally proposed: **sum over all four spin combinations of $|P_{ab}|^2$**.

The physics behind this is that even though we're fitting the spin-diagonal sum, the band summation in the Galerkin condition couples all spin channels through terms like $\sum_m \psi_{m,\uparrow} \psi^*_{m,\downarrow}$.

**My previous implementation using $|P_{\uparrow\uparrow} + P_{\downarrow\downarrow}|^2$ was INCORRECT for the spin-trace target.** The correct formula is:

$$
C_R(\mu,\nu) = \sum_{ab} |P_{R,ab}(\mu,\nu)|^2
$$

