# Ladder vertex corrections

`screening_diagrams = w_bse` The screened interaction with the electron–hole ladder, from matrix-free products with the Bethe–Salpeter Hamiltonian.
{: .page-summary }

<div class="section-cards"></div>

## <span class="n">1</span> Vertex corrections in W {: #vertex }

The RPA screens the Coulomb interaction by independent electron–hole pairs. When the electron and the hole of each virtual pair attract each other, the response of the system grows, and the screened interaction changes accordingly. In the language of diagrams this attraction is the ladder series of the two-particle vertex, the same series that produces bound excitons in the Bethe–Salpeter equation [1, 2].
{: .intro }

In the self-consistent GW calculations studied in [3, 4], adding the electron–hole ladder increases screening and reduces the gap overestimate. This motivates the correction; its magnitude and sign still require a calculation for the chosen material. The [analytic structure of W](w-structure.md#resolvent) makes the correction a change of Hamiltonian. The correlation part of $W$ is a resolvent between pair potentials,

$$
W^c_\bq(z)=\phi_\bq\,(z-H_\bq)^{-1}\sigma\,\phi_\bq^\dagger,\qquad \phi_{mn\bk\bq}(\mu)=\sum_{\mu'}V_\bq(\mu,\mu')\,\psi^*_{m\bk}(\br_{\mu'})\,\psi_{n\bk+\bq}(\br_{\mu'}) \tag{1.1}
$$

with $\sigma$ the occupation differences. With $H_\bq=H^{\mathrm{RPA}}$, the transition energies plus the Coulomb exchange, (1.1) is the RPA screened interaction. With $H_\bq=H^{\mathrm{BSE}}$, which adds the direct term of the statically screened interaction $W(0)$, it is the screened interaction with the full ladder of electron–hole attraction. The second choice adds the ladder vertex in the screening. LORRAX currently serves it in a one-shot calculation with a static $W^{\mathrm{RPA}}(0)$ rung and $\Sigma=iGW$; it does not implement the self-consistent ladder scheme of [4] or a vertex in $\Sigma$.

## <span class="n">2</span> Resolvent solves with matrix-free products {: #products }

`bse.w_ladder.build_ladder_resolvent`
{: .in-code }

$H^{\mathrm{BSE}}_\bq$ is never formed. A Krylov solver needs only products $HX$ with a block of trial vectors, and each product is assembled from the ISDF quantities. The exchange term encodes $X$ into a density on the $N_\mu$ points, applies $V_\bq$ and decodes it. The ladder term builds the pair amplitude $\Phi_\bk(\br_\mu,\br_\nu)$, multiplies it by $W_\bR(\mu,\nu)$ after an FFT over the k grid, and transforms back ([BSE theory §3](../drivers/bse/theory.md#hx)). For $N_t=n_vn_cN_k$ transitions at one $\bq$, one product costs

$$
\mathcal C_{HX}\sim N_kN_\mu^2\,(n_v+n_c)+N_\mu^2\,N_k\log N_k,\qquad \mathcal M_{HX}\sim 16\,N_\mu^2N_k\ \text{bytes} \tag{2.1}
$$

for the band contractions, the k-convolution and the storage of $W_\bR$ and one $\Phi$. These are unblocked global tensor scales, divided over the processor mesh; the implementation admits its actual block and workspace sizes. A two-component calculation multiplies the $\Phi$ storage by four. Krylov vectors, deflation and response-action carries add to this storage. For each probe direction $r$ on the points, the applied resolvent $(z-H_\bq)^{-1}\sigma\phi^\dagger_\bq r$ follows from GMRES at each $z$. The iteration count is measured and its true residual is checked; each frequency, probe column and derivative action adds solves.

## <span class="n">3</span> Cost against a dense kernel {: #scaling }

The dense alternative builds the kernel as an $N_t\times N_t$ matrix at every $\bq$ and inverts or diagonalizes it, at $16N_t^2$ bytes and $\mathcal O(N_t^3)$ operations. At fixed band counts, points and trial width, each matrix-free product grows as $N_k\log N_k$; the dense kernel grows as $N_k^2$ in memory and its diagonalization as $N_k^3$ in time. The resolvent includes every pair in the selected transition window; it makes no additional low-energy projection of that window. This permits large band windows, but their physical convergence still has to be tested. The complete response cost also includes the number of probe columns, frequencies and Krylov iterations, with orthogonalization and retained Krylov storage.

| One $\bq$, $n_v=100$, $n_c=900$, $N_k=12^3$, $N_\mu=4000$ | Memory |
|---|---|
| Dense kernel, $N_t=1.56\times10^8$, complex128 | $16N_t^2\approx3.9\times10^{17}$ bytes, 0.39 EB |
| Matrix-free: $W_\bR$ and one $\Phi$ | $2\times16N_\mu^2N_k\approx0.88$ TB |
| Matrix-free: band values $\psi$ at the points | $16N_k(n_v+n_c)N_\mu\approx0.11$ TB |

These matrix-free entries count only the listed global tensors, for one scalar trial; they are not a peak-memory prediction. Spinor branches, probe width and Krylov storage must also fit. The dense kernel would be needed at every irreducible $\bq$, and its diagonalization would take about $N_t^3\approx4\times10^{24}$ operations.

## <span class="n">4</span> In LORRAX {: #lorrax }

[`screening_diagrams`](../drivers/gw/keywords.md#screening_diagrams) selects the diagrams of $W$. `w_rpa` is the Dyson equation of [GW theory §2.4](../drivers/gw/theory.md#scr-w). `w_bse` evaluates (1.1) with $H^{\mathrm{BSE}}$, and `w_rpa_resolvent` evaluates it with $H^{\mathrm{RPA}}$, which reproduces the Dyson result and serves as a check of the resolvent route. The keyword entry lists the frequency treatments each choice supports.

### Physical resonant and antiresonant branches {: #branches }

At finite $\bq$, the resonant pair is $c(\bk+\bq)/v(\bk)$ and the antiresonant pair is $c(\bk)/v(\bk+\bq)$. Both branches retain their own band values, energies and density vertices. The physical Hamiltonian and these matched input/output vertices are used together; a branch sum from a different RPA coordinate convention is not a density vertex for this Hamiltonian. Time-reversal and even-frequency checks establish whether the resulting response can enter the even-$s$ tangential model.

With `compute_mode = tangential`, exact resolvent actions provide $W^c$, $\partial_s W^c$ and the high-frequency moments. The latter are $M_1=\mathcal C H\mathcal B/2$ and $M_3=\mathcal C H^3\mathcal B/2$ in the matched transfer-function normalization of [W's analytic structure](w-structure.md#transfer); the ladder does not reuse RPA moment formulas. The common tangential constructor and the common $G W$ executor then own the pole model and $\Sigma$.

### The long-wavelength head {: #head }

The same physical Hamiltonian supplies the microscopic reducible $3\times3$ tensor $\Xi(z)$ from velocity/dipole vertices. Its microscopic local fields are already included. The head owner restores only the omitted macroscopic Coulomb channel, then averages the singular cell; an additional RPA wing/body fold would count those local fields twice. The scalar correlation head is $\langle v^2 q^T\Xi(z)q/[1-vq^T\Xi(z)q]\rangle$ on that cell. This same nonlinear completion supplies its derivative and high-frequency moments: the $z^{-4}$ coefficient includes the square of the leading microscopic moment.

For tangential screening the completed scalar values, exact $s$ derivatives and moments enter the shared positive Hermite/Ritz constructor on one scalar port. Its scalar convention is $W^c_{\mathrm{head}}(z)=\sum_p2\Omega_pB_p/(z^2-\Omega_p^2)$, whose inverse-frequency moment $\sum_pB_p/\Omega_p^2$ differs from the high-frequency $M_1$ above. Its pole count follows the retained span. Independent held values and derivatives, the static limit, the inverse-frequency moment used by $\Sigma$, high-frequency moments and Coulomb-normalized passivity are checked before publication. Failed held sites may become supports only after failure, followed by fresh held solves within the recipe budget. Poles and residues are not pinned or repaired. These are measured-support certificates, not a uniform real-axis error bound.

The sampled bulk rule and production slab polygon share their quadrature across frequency batches. The analytic-sphere response-action extension is not implemented and refuses for this tangential full-head route. GN-PPM keeps its existing head treatment. Current diagram/frequency admission and self-consistency refusals remain in the [keyword contract](../drivers/gw/keywords.md#screening_diagrams).

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
