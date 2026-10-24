# Ladder vertex corrections

`screening_diagrams = w_bse` The screened interaction with the electron–hole ladder, from matrix-free products with the Bethe–Salpeter Hamiltonian.
{: .page-summary }

<div class="section-cards"></div>

## <span class="n">1</span> Vertex corrections in W {: #vertex }

The RPA screens the Coulomb interaction by independent electron–hole pairs. When the electron and the hole of each virtual pair attract each other, the response of the system grows, and the screened interaction changes accordingly. In the language of diagrams this attraction is the ladder series of the two-particle vertex, the same series that produces bound excitons in the Bethe–Salpeter equation [1, 2].
{: .intro }

In a self-consistent GW calculation, RPA screening underestimates the dielectric response, and quasiparticle gaps come out too large. Including the electron–hole ladder in $W$ corrects much of this error [3, 4]. The [analytic structure of W](w-structure.md#resolvent) makes the correction a change of Hamiltonian. The correlation part of $W$ is a resolvent between pair potentials,

$$
W^c_\bq(z)=\phi_\bq\,(z-H_\bq)^{-1}\sigma\,\phi_\bq^\dagger,\qquad \phi_{mn\bk\bq}(\mu)=\sum_{\mu'}V_\bq(\mu,\mu')\,\psi^*_{m\bk}(\br_{\mu'})\,\psi_{n\bk+\bq}(\br_{\mu'}) \tag{1.1}
$$

with $\sigma$ the occupation differences. With $H_\bq=H^{\mathrm{RPA}}$, the transition energies plus the Coulomb exchange, (1.1) is the RPA screened interaction. With $H_\bq=H^{\mathrm{BSE}}$, which adds the direct term of the statically screened interaction $W(0)$, it is the screened interaction with the full ladder of electron–hole attraction. The second choice is the vertex-corrected $W$ of quasiparticle self-consistent GW with ladder diagrams [4].

### Spin-restricted scalar wavefunctions {: #scalar-spin }

For a nonmagnetic closed shell without spin–orbit coupling, the BSE separates
into a singlet and three equivalent triplets [5]. With positive bare exchange
$V$ and screened direct attraction $W_d$, their resonant blocks are

$$
A_S=D+2V-W_d,\qquad A_T=D-W_d.
$$

The coupling blocks have the same exchange weights, 2 and 0. The charge
vertex is the identity in spin space: it projects onto the normalized
singlet with amplitude $\sqrt{2}$ and has zero projection onto every triplet.
Consequently $W=v+v\chi_{nn}v$ needs only the singlet resolvent. The spin sum
supplies the factor 2 in the charge response; it does not multiply the
screened direct rung.

This is the non-spinor `w_bse` route, including its head and moments. The
optical `spin_triplet` flag selects a separate excitation problem. Adding
triplet fluctuations to a self-energy requires specifying a spin-resolved
vertex approximation beyond the present $\Sigma=iGW$ definition.

## <span class="n">2</span> Resolvent solves with matrix-free products {: #products }

`bse.bse_stack_matvec.build_bse_stack_matvec`
{: .in-code }

$H^{\mathrm{BSE}}_\bq$ is never formed. A Krylov solver needs only products $HX$ with a block of trial vectors, and each product is assembled from the ISDF quantities. The exchange term encodes $X$ into a density on the $N_\mu$ points, applies $V_\bq$ and decodes it. The ladder term builds the pair amplitude $\Phi_\bk(\br_\mu,\br_\nu)$, multiplies it by $W_\bR(\mu,\nu)$ after an FFT over the k grid, and transforms back ([BSE theory §3](../drivers/bse/theory.md#hx)). For $N_t=n_vn_cN_k$ transitions at one $\bq$, one product costs

$$
\mathcal C_{HX}\sim N_kN_\mu^2\,(n_v+n_c)+N_\mu^2\,N_k\log N_k,\qquad \mathcal M_{HX}\sim 16\,N_\mu^2N_k\ \text{bytes} \tag{2.1}
$$

for the band contractions, the k-convolution and the storage of $W_\bR$ and one $\Phi$. A two-component calculation multiplies the $\Phi$ storage by four. For each probe direction $r$ on the points, the applied resolvent $(z-H_\bq)^{-1}\sigma\phi^\dagger_\bq r$ follows from a few tens of such products by GMRES at each $z$.

## <span class="n">3</span> Cost against a dense kernel {: #scaling }

The dense alternative builds the kernel as an $N_t\times N_t$ matrix at every $\bq$ and inverts or diagonalizes it, at $16N_t^2$ bytes and $\mathcal O(N_t^3)$ operations. The matrix-free cost grows as $N_k\log N_k$ per $\bq$, where the dense one grows as $N_k^2$ in memory and $N_k^3$ in time. Because the products never restrict the transition space, the resolvent runs over every pair of the $\chi^0$ band sum. In effect the Bethe–Salpeter equation is solved over hundreds of bands, and the screened interaction is converged with respect to the choice of transition subspace.

| One $\bq$, $n_v=100$, $n_c=900$, $N_k=12^3$, $N_\mu=4000$ | Memory |
|---|---|
| Dense kernel, $N_t=1.56\times10^8$, complex128 | $16N_t^2\approx3.9\times10^{17}$ bytes, 0.39 EB |
| Matrix-free: $W_\bR$ and one $\Phi$ | $2\times16N_\mu^2N_k\approx0.88$ TB |
| Matrix-free: band values $\psi$ at the points | $16N_k(n_v+n_c)N_\mu\approx0.11$ TB |

The dense kernel would be needed at every irreducible $\bq$, and its diagonalization would take about $N_t^3\approx4\times10^{24}$ operations.

## <span class="n">4</span> In LORRAX {: #lorrax }

[`screening_diagrams`](../drivers/gw/keywords.md#screening_diagrams) selects the diagrams of $W$. `w_rpa` is the Dyson equation of [GW theory §2.4](../drivers/gw/theory.md#scr-w). `w_bse` evaluates (1.1) with $H^{\mathrm{BSE}}$, and `w_rpa_resolvent` evaluates it with $H^{\mathrm{RPA}}$, which reproduces the Dyson result and serves as a check of the resolvent route. The keyword entry lists the frequency treatments each choice supports.

## Notation {: #notation }

| Symbol | Meaning |
|---|---|
| $H^{\mathrm{RPA}}$, $H^{\mathrm{BSE}}$ | particle–hole Hamiltonian without and with the statically screened ladder |
| $\phi_{mn\bk\bq}$, $\sigma$ | pair potential; occupation differences |
| $\Phi_\bk(\br_\mu,\br_\nu)$, $W_\bR(\mu,\nu)$ | pair amplitude of a trial vector; static $W$ on the lattice vectors of the k grid |
| $N_t$, $n_v$, $n_c$, $N_k$, $N_\mu$ | transitions at one $\bq$; occupied and empty bands; k-points; ISDF points |

## References {: #refs }

1. G. Strinati, Riv. Nuovo Cimento 11, 1 (1988). doi:10.1007/BF02725962
2. G. Onida, L. Reining and A. Rubio, Rev. Mod. Phys. 74, 601 (2002). doi:10.1103/RevModPhys.74.601
3. M. Shishkin, M. Marsman and G. Kresse, Phys. Rev. Lett. 99, 246403 (2007). doi:10.1103/PhysRevLett.99.246403
4. B. Cunningham, M. Grüning, D. Pashov and M. van Schilfgaarde, Phys. Rev. B 108, 165104 (2023). doi:10.1103/PhysRevB.108.165104
5. M. Marsili et al., [Phys. Rev. B 103, 155152 (2021)](https://doi.org/10.1103/PhysRevB.103.155152), §IV.B(iii), Eqs. (52)–(53).
