# Analytic structure of W

$W^c(z)$ The screened interaction as the resolvent of a particle–hole Hamiltonian, and the two formalisms LORRAX builds on that form.
{: .page-summary }

<div class="section-cards"></div>

## <span class="n">1</span> Beyond the Dyson equation {: #dyson }

GW calculations usually obtain the screened interaction one frequency at a time, from the Dyson equation $W(z)=[1-v\chi^0(z)]^{-1}v$. Written that way, $W$ appears as a separate matrix inverse at every $z$. It is in fact a single rational function of frequency whose poles and residues come from one Hermitian eigenvalue problem. LORRAX is organized around this structure.
{: .intro }

The density response $\chi$, the independent-particle response $\chi^0$ and the screened interaction obey

$$
\chi=\chi^0+\chi^0v\chi,\qquad W=v+v\chi v=\big[1-v\chi^0\big]^{-1}v,\qquad W^c=W-v \tag{1.1}
$$

The Dyson form is how $W$ is computed at a given $z$. The rest of this page derives the closed form that the Dyson equation sums, and the two consequences LORRAX draws from it.

## <span class="n">2</span> W as a resolvent {: #resolvent }

With the pair densities $M_{mn\bk\bq}(\br)=\psi^*_{m\bk}(\br)\,\psi_{n\bk+\bq}(\br)$ and their Coulomb potentials $\phi_{mn\bk\bq}=vM_{mn\bk\bq}$, the independent-particle response is a sum over the $2N_t$ resonant and antiresonant transitions $mn\bk$ of $\bk\to\bk+\bq$. The same transitions span the RPA particle–hole Hamiltonian

$$
\begin{aligned}
\big(H_\bq\big)_{\substack{mn\bk\\m'n'\bk'}}&=\big(E_{n\bk+\bq}-E_{m\bk}\big)\,\delta_{\substack{mn\bk\\m'n'\bk'}}\\
&\quad+\sigma_{mn\bk}\int d\br\,d\br'\,M^*_{mn\bk\bq}(\br)\,v(\br,\br')\,M_{m'n'\bk'\bq}(\br')
\end{aligned} \tag{2.1}
$$

with the occupation differences $\sigma_{mn\bk}=f_{m\bk}-f_{n\bk+\bq}$. Its eigenvalue problem is Casida's equation [3]. Summing the Dyson series (1.1) in this basis gives the correlation part of $W$ as a resolvent of $H_\bq$ between pair potentials,

$$
W^c_\bq(z)=\phi\,(z-H_\bq)^{-1}\sigma\phi^\dagger=\sum_S\mathrm{sgn}(\Omega_S)\,\frac{b_S\,b_S^\dagger}{z-\Omega_S},\qquad b_S=\sum_{mn\bk}x_S^{mn\bk}\,\phi_{mn\bk\bq} \tag{2.2}
$$

Here $\phi$ collects the pair potentials as columns and only nonzero occupation differences are retained. $H_\bq$ is generally not Hermitian in the Euclidean inner product. In a stable system its eigenvectors can be normalized by $x_S^\dagger\sigma^{-1}x_{S\prime}=\mathrm{sgn}(\Omega_S)\delta_{SS\prime}$; (2.2) uses this metric normalization. For integer occupations the metric reduces to the resonant/antiresonant signature. Spin weights and cell/k-point normalizations are suppressed here; the implementation includes them in its matched vertices and prefactors. Each neutral excitation $S$ of the system contributes one pole, and its residue is the outer product of the potential $b_S$ of its transition density [1, 2]. The plasmon of a plasmon-pole model is one such excitation, or a weighted average over many of them.

## <span class="n">3</span> Positivity and moments {: #positivity }

With time reversal, the antiresonant transitions mirror the resonant ones and $W^c$ depends on $z$ only through $s=z^2$. For the integer-occupation RPA in a transition basis with the two time-reversed branches matched, let $\Delta$ contain the positive transition energies and $K$ their Coulomb integrals. The reduction to $N_t$ resonant transitions is

$$
W^c_\bq(s)=F^\dagger\,(s-C_\bq)^{-1}F,\qquad C_\bq=\Delta^{1/2}(\Delta+2K)\Delta^{1/2},\qquad F=\sqrt2\,\Delta^{1/2}\phi^\dagger \tag{3.1}
$$

$C_\bq$ is Hermitian and positive for $\Delta>0$ and positive semidefinite Coulomb $K$, with eigenvalues $\Omega_S^2$. A changed Hamiltonian, including the attractive ladder, needs its own stability check; an electronic band gap alone does not establish ladder stability. Three properties follow directly. The poles in $s$ are real and lie between the smallest and the largest $\Omega_S^2$. Every residue is positive semidefinite, so $-W^c$ is a matrix Stieltjes function of $-s$. The expansion in $1/s$ gives the exact high-frequency moments,

$$
W^c_\bq(z)=\frac{2M_1}{z^2}+\frac{2M_3}{z^4}+\dots,\qquad M_1=\phi\Delta\phi^\dagger,\qquad M_3=\phi\Delta^{1/2}C_\bq\Delta^{1/2}\phi^\dagger \tag{3.2}
$$

$2M_1$ is the high-frequency oscillator-strength weight of the selected band window. Neither RPA moment requires a Dyson solve. Real positive squared poles and positive residues preserve this stable-response structure, but they do not by themselves bound the approximation error between sampled frequencies. With a ladder kernel, the moments instead come from the matched full Hamiltonian as $M_1=\mathcal C H\mathcal B/2$ and $M_3=\mathcal C H^3\mathcal B/2$ in (4.1); (3.1) and (3.2) are the RPA reduction, not a replacement for those ladder actions.

## <span class="n">4</span> W as a transfer function {: #transfer }

In the language of linear systems [4], (2.2) is the transfer function of a system of order $2N_t$. A potential applied at the $N_\mu$ ISDF points enters the transitions through $\sigma\phi^\dagger$, evolves under $H_\bq$, and is read out through $\phi$ as the induced potential,

$$
W^c_\bq(z)=\mathcal C\,(z-\mathcal A)^{-1}\mathcal B,\qquad \mathcal C=\phi,\quad \mathcal A=H_\bq,\quad \mathcal B=\sigma\phi^\dagger \tag{4.1}
$$

$N_t$ runs to millions in a converged GW calculation, so $H_\bq$ cannot be diagonalized. A value of $W^c$ at one complex $z$ can nevertheless be computed without storing $H_\bq$. The Coulomb term of (2.1) has rank at most $N_\mu$, and the Woodbury identity reduces the resolvent to the $N_\mu\times N_\mu$ Dyson solve (1.1). One such sample, applied to a chosen vector $r$ on the points, is the action of the resolvent on the known vector $\sigma\phi^\dagger r$ read out through $\phi$. Samples of $W$ therefore probe $H_\bq$ through its resolvent, without the matrix ever being formed. This small Dyson reduction applies to the RPA kernel. The statically screened direct ladder instead uses matrix-free Krylov solves; its cost depends on the actual iterations and requested actions.

## <span class="n">5</span> Products with the particle–hole Hamiltonian {: #products }

The same Hamiltonian is also cheap to apply directly. In the ISDF basis a product $H_\bq X$ with a block of trial vectors consists of the diagonal transition energies, an exchange term that passes through a density on the $N_\mu$ points, and, when the electron–hole ladder is included, a convolution over the k grid that FFTs evaluate in $\mathcal O(N_k\log N_k)$. With $n_v$ occupied and $n_c$ empty bands, its cost is of order $N_kN_\mu^2(n_v+n_c)$ plus the FFTs, and its memory is that of a few $N_\mu\times N_\mu$ arrays per k-point. The matrix it represents, with $N_t^2$ entries, is never stored.

## <span class="n">6</span> Two formalisms {: #formalisms }

The resolvent form (2.2) and the low cost of both samples and products lead to the two formalisms LORRAX uses for the frequency dependence and the vertex structure of $W$.

<div class="grid cards" markdown>

-   **[Tangential W model](tangential-w.md)**

    ---

    A few samples of $W^c$ define a projection of $H_\bq$ onto the span of the sampled resolvent vectors. The Loewner framework of model reduction builds this projection from the samples alone, and its Ritz values are one set of real poles shared by every matrix element, with positive residues and the exact moments of (3.2). This is the preferred full-frequency treatment of GW in LORRAX.

-   **[Ladder vertex corrections](ladder-w.md)**

    ---

    Replacing $H^{\mathrm{RPA}}$ in (2.2) by the Bethe–Salpeter Hamiltonian adds the electron–hole ladder to the screening. Krylov solves with matrix-free products evaluate the resolvent at each frequency over the whole transition space of the $\chi^0$ band sum, at a cost that grows as $N_k\log N_k$ per $\bq$.

</div>

## Notation {: #notation }

| Symbol | Meaning |
|---|---|
| $M_{mn\bk\bq}$, $\phi_{mn\bk\bq}$, $\phi$ | pair density $\psi^*_{m\bk}\psi_{n\bk+\bq}$; its Coulomb potential $vM$; the matrix of the potentials |
| $H_\bq$, $\sigma$ | RPA particle–hole Hamiltonian (2.1); occupation differences $f_{m\bk}-f_{n\bk+\bq}$ |
| $\Omega_S$, $x_S$, $b_S$ | neutral excitation energy; eigenvector of $H_\bq$; residue vector |
| $N_t$, $N_\mu$, $N_k$ | resonant transitions; ISDF points; k-points |
| $n_v$, $n_c$ | occupied and empty bands of the transition space |
| $s$, $\Delta$, $K$, $C_\bq$, $F$ | $z^2$; resonant transition energies; their Coulomb integrals; Casida matrix; input matrix of the form in $s$ |
| $M_1$, $M_3$ | high-frequency moments of $W^c$ |
| $\mathcal C$, $\mathcal A$, $\mathcal B$ | output, state and input matrices of the realization (4.1) |

## References {: #refs }

1. L. Hedin, Phys. Rev. 139, A796 (1965).
2. G. Strinati, Riv. Nuovo Cimento 11, 1 (1988). doi:10.1007/BF02725962
3. M. E. Casida, in *Recent Advances in Density Functional Methods, Part I*, edited by D. P. Chong (World Scientific, 1995), p. 155. doi:10.1142/9789812830586_0005
4. A. C. Antoulas, C. A. Beattie and S. Gugercin, *Interpolatory Methods for Model Reduction* (SIAM, 2020). doi:10.1137/1.9781611976083
