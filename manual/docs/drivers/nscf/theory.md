---
driver: nscf
page: theory
---

# Theory

`psp.run_nscf` The equations each stage evaluates.
{: .page-summary }

<div class="workflow-map" data-driver="nscf" data-page="theory" data-treatment="davidson"></div>

<div class="section-cards"></div>

## <span class="n">1</span> The Kohn–Sham Hamiltonian {: #hamiltonian }

$H_\bk$ is rebuilt from the UPF files and the QE valence density, not read from QE. Energies are in Rydberg ($\hbar^2/2m = 1$) throughout.
{: .intro }

### <span class="n">1.1</span> Plane-wave basis and kinetic energy {: #ham-basis }

`psp.gvec_utils`
{: .in-code }

At each k-point the basis is the G sphere set by the QE wavefunction cutoff, and the kinetic term is diagonal on it:

$$
\{|\bk+\bG\rangle : |\bk+\bG|^2 \le E_{\mathrm{cut}}\}, \qquad T_\bk(\bG,\bG') = |\bk+\bG|^2\,\delta_{\bG\bG'} \tag{1.1}
$$

$E_{\mathrm{cut}}$ is QE `ecutwfc`, read in Hartree from `data-file-schema.xml` and doubled. Every sphere is padded to the largest, `ngkmax`, with a mask, so all k-points share one array shape.

### <span class="n">1.2</span> Local ionic potential {: #ham-loc }

`psp.ionic_gspace.build_ionic_and_core`
{: .in-code }

The sum of the atomic local pseudopotentials $v^{\mathrm{loc}}_\kappa$ at the positions $\boldsymbol\tau_\kappa$, assembled in G space from radial tables:

$$
V^{\mathrm{loc}}(\br) = \sum_\kappa v^{\mathrm{loc}}_\kappa(|\br-\boldsymbol\tau_\kappa|) \tag{1.2}
$$

The $\bG = 0$ term is QE's non-Coulomb $\alpha Z$ correction. With QE `assume_isolated = '2D'` the long-range part uses the slab-truncated Coulomb interaction of §1.4. The same call builds the UPF core charge $\rho_{\mathrm{core}}$.

### <span class="n">1.3</span> Nonlocal potential {: #ham-nl }

`psp.vnl_ops.build_vnl_setup`
{: .in-code }

The separable form, with radial projectors $\beta_{\kappa i}$, real spherical harmonics $S_{lm}$ and the UPF couplings $D_{ij}$, which become spin blocks $D^{\sigma\sigma'}_{ij}$ with spin–orbit coupling:

$$
V^{\mathrm{NL}}_\bk(\bG,\bG') = \sum_{\kappa,ij} Z_{\kappa i}(\bk+\bG)\,D_{ij}\,Z^*_{\kappa j}(\bk+\bG'), \qquad
Z_{\kappa i}(\mathbf K) = \beta_{\kappa i}(|\mathbf K|)\,S_{l_im_i}(\hat{\mathbf K})\,e^{-i\mathbf K\cdot\boldsymbol\tau_\kappa} \tag{1.3}
$$

With spin–orbit coupling $E$ carries spin blocks. j-resolved or j-averaged projectors follow the `.save`'s `spinorbit` flag (QE `lspinorb`), not `noncolin`. The coupling is applied per pseudopotential channel, one GEMM per channel.

### <span class="n">1.4</span> Hartree and exchange–correlation {: #ham-hxc }

`psp.dft_operators.compute_V_H_and_V_xc`
{: .in-code }

Both terms are built once from the QE valence density $\rho_{\mathrm{val}}$ in `charge-density.hdf5`. The Hartree term sees the valence density only; the core charge enters $V^{xc}$ only:

$$
V^{\mathrm H}(\bG) = \frac{8\pi\,\rho_{\mathrm{val}}(\bG)}{|\bG|^2}\ (\bG \ne 0), \quad V^{\mathrm H}(\bG{=}0) = 0, \qquad
V^{xc}(\br) = \frac{\delta E^{\mathrm{PBE}}_{xc}}{\delta\rho}\Big[\rho_{\mathrm{val}}+\rho_{\mathrm{core}}\Big](\br) \tag{1.4}
$$

$V^{xc}$ is obtained by automatic differentiation of the `jax_xc` PBE kernels. With `assume_isolated = '2D'`, $8\pi/|\bG|^2$ is multiplied by the slab factor $1 - e^{-|\bG_\parallel|L_z/2}\cos(G_zL_z/2)$.

!!! note
    The functional is PBE whatever the QE run used, and no $B^{xc}$ is built for a magnetic run. Nothing checks either.

### <span class="n">1.5</span> Applying H {: #ham-apply }

`psp.dft_operators.apply_H_k_batched`
{: .in-code }

$H_\bk$ is never stored. On a block of vectors, $T_\bk$ acts as a diagonal, the local sum $V^{\mathrm{loc}} + V^{\mathrm H} + V^{xc}$ on the FFT grid (IFFT to $\br$, multiply, FFT back), and $V^{\mathrm{NL}}_\bk$ as project, couple, expand on the G sphere:

$$
H_\bk = T_\bk + V^{\mathrm{loc}} + V^{\mathrm{NL}}_\bk + V^{\mathrm H} + V^{xc} \tag{1.5}
$$

Vectors are applied in blocks of 32, which bounds the FFT-grid temporaries.

## <span class="n">2</span> Block Davidson {: #davidson }

At each k-point the lowest $N_b$ eigenpairs of $H_\bk\psi_{n\bk} = E_{n\bk}\psi_{n\bk}$ are found by Rayleigh–Ritz on a growing subspace. One solver is compiled for the padded G sphere and reused at every k-point.
{: .intro }

### <span class="n">2.1</span> Initial guess {: #dav-init }

`psp.dft_precond.make_pw_init`
{: .in-code }

$H_\bk$ is projected onto the $2N_b$ plane waves of lowest $|\bk+\bG|^2$ (per spinor component), and its lowest $N_b$ eigenvectors start the solve:

$$
x^{(0)}_n = \sum_{\bG \in \mathcal P} c_{n\bG}\,|\bk+\bG\rangle, \qquad |\mathcal P| = 2N_b \tag{2.1}
$$

### <span class="n">2.2</span> Rayleigh–Ritz step {: #dav-rr }

`solvers.plan_local_davidson`
{: .in-code }

The subspace Hamiltonian over the trial vectors $\{x_n\}$ is diagonalized, and the lowest $N_b$ Ritz pairs give the residuals:

$$
H_{mn\bk} = \langle x_m|H_\bk|x_n\rangle, \qquad r_n = (H_\bk - E_n)\,x_n \tag{2.2}
$$

When the subspace would exceed $10\,N_b$ vectors, the solve restarts from the current Ritz vectors.

### <span class="n">2.3</span> Preconditioner {: #dav-precond }

`psp.dft_precond.make_dft_preconditioner`
{: .in-code }

The QE diagonal preconditioner, with $h_\bG$ the kinetic energy plus the $\bG = 0$ local ionic potential and the diagonal of $V^{\mathrm{NL}}_\bk$:

$$
p_n(\bG) = \frac{r_n(\bG)}{h_\bG - E_n}, \qquad h_\bG = |\bk+\bG|^2 + V^{\mathrm{loc}}(\bG{=}0) + \big[V^{\mathrm{NL}}_\bk\big]_{\bG\bG} \tag{2.3}
$$

$|h_\bG - E_n|$ is floored at 0.01 Ry, and each $p_n$ is normalized. $V^{\mathrm H}$ and $V^{xc}$ are left out of $h_\bG$, as in QE. The corrections are orthogonalized against $\{x_m\}$ and appended.

### <span class="n">2.4</span> Convergence {: #dav-conv }

The solve stops when every one of the $N_b$ residuals passes the relative test:

$$
\|r_n\| < \texttt{tol}\cdot\max(1,|E_n|), \qquad n = 1,\dots,N_b \tag{2.4}
$$

A k-point that reaches 100 iterations, or makes no 1 % residual progress in 20, stops the run before anything is written. There are no buffer bands: the top band sets the cost.

## <span class="n">3</span> k-points and the written bands {: #output }

The k set, the occupations and the G order are fixed by QE conventions so that `WFN.h5` reads like a `pw2bgw.x` file.
{: .intro }

### <span class="n">3.1</span> k-grid and symmetry {: #out-kgrid }

`CrystalData.build_kgrid`
{: .in-code }

The Γ-centred grid $N_1\times N_2\times N_3$ is reduced by the `.save` operations $\{S\}$ and time reversal, following QE `kpoint_grid.f90`; the weights are the orbit sizes:

$$
\bk_{ijl} = \Big(\tfrac{i}{N_1},\tfrac{j}{N_2},\tfrac{l}{N_3}\Big), \qquad w_\bk = \frac{|\{S\bk,\,-S\bk\}|}{N_1N_2N_3} \tag{3.1}
$$

With `nosym = true` only the identity is used and there is no $\bk \to -\bk$ folding. Crystal coordinates are folded into $[-\tfrac12,\tfrac12)$. Symmetry itself is on the [Symmetry](../../theory/symmetry.md) page.

### <span class="n">3.2</span> Occupations {: #out-occ }

`file_io.wfn_writer.wfn_header_tables`
{: .in-code }

The file carries an insulator step from the QE electron count $N_{\mathrm{el}}$:

$$
f_{n\bk} = \begin{cases} 1, & n \le N_{\mathrm{occ}} \\ 0, & n > N_{\mathrm{occ}} \end{cases}, \qquad N_{\mathrm{occ}} = \tfrac12\,N_{\mathrm{el}}\,n_{\mathrm{spin}}\,n_{\mathrm{spinor}} \tag{3.2}
$$

No smearing or Fermi level is written. Metals need the QE route of [Required DFT inputs](../../using/inputs-from-dft.md); see [Metals](../../theory/metals.md).

### <span class="n">3.3</span> G order {: #out-gorder }

`psp.gvec_utils.reorder_to_qe`
{: .in-code }

The solver works in $|\bk+\bG|^2$ order. Before writing, each k-point's coefficients are permuted, by a permutation $\pi$, to the order of the QE master G list that `wfns/gvecs` records:

$$
\psi_{n\bk}(\bG_{\pi(j)}) \leftarrow x_{n}(\bG_j) \tag{3.3}
$$

## <span class="n">4</span> Pseudobands {: #pseudobands }

An optional stage after Davidson replaces the many bands above the protected ones by a few stochastic vectors per energy window. It writes a separate file that no driver reads yet.
{: .intro }

### <span class="n">4.1</span> Windows and filtering {: #pb-windows }

`solvers.pseudobands_v2.ritz_pseudobands_v2`
{: .in-code }

The spectrum of $H_\bk$, of width $B$, is cut into about `pb_n_windows` DOS-weighted windows. Random vectors are projected onto each window with a Chebyshev–Jackson filter of order $M$ = `pb_M_max`, whose energy resolution is

$$
\delta E = \frac{\pi B}{M} \tag{4.1}
$$

In v2 each window's filtered block of `pb_k` vectors is Ritz-rotated and given Gauss-quadrature energies, with the window weight folded into the vectors. The windows and DOS are fixed at the first k-point and reused at the others.

### <span class="n">4.2</span> The protected bands {: #pb-prot }

The lowest Davidson bands are kept exactly. With `pb_n_prot = 0`, the count is every band below the crossover energy:

$$
E_{n\bk_0} < E_F + \frac{\pi B}{M F}, \qquad F = \texttt{pb\_F} \tag{4.2}
$$

$E_F$ is passed as 0 Ry while $E_{n\bk}$ are absolute QE energies, so the automatic count depends on QE's energy zero.

!!! note
    Set `pb_n_prot` explicitly. The count found at the first k-point is used at every k-point.

## Notation {: #notation }

| Symbol | Meaning |
|---|---|
| $H_\bk$ | Kohn–Sham Hamiltonian at $\bk$, Ry |
| $T_\bk$, $V^{\mathrm{loc}}$, $V^{\mathrm{NL}}_\bk$ | kinetic, local ionic and nonlocal pseudopotential terms |
| $V^{\mathrm H}$, $V^{xc}$ | Hartree and exchange–correlation potentials |
| $\rho_{\mathrm{val}}$, $\rho_{\mathrm{core}}$ | QE valence density, UPF core charge |
| $v^{\mathrm{loc}}_\kappa$, $\beta_{\kappa i}$, $D_{ij}$ ($D^{\sigma\sigma'}_{ij}$) | UPF local potential, radial projectors and couplings of atom $\kappa$ (spin blocks with spin–orbit coupling) |
| $Z_{\kappa i}(\mathbf K)$ | projector form factor at $\mathbf K = \bk+\bG$ |
| $E_{\mathrm{cut}}$ | wavefunction cutoff, QE `ecutwfc`, Ry |
| $N_b$ | bands solved per k-point, `nbnd` |
| $x_n$, $E_n$, $r_n$, $p_n$ | trial vectors, Ritz values, residuals, preconditioned residuals |
| $h_\bG$ | preconditioner diagonal |
| $\psi_{n\bk}(\bG)$, $E_{n\bk}$, $f_{n\bk}$ | written coefficients, energies and occupations |
| $N_{\mathrm{occ}}$ | occupied band count |
| $B$, $M$, $F$ | spectral width, Chebyshev order, window ratio |

## References {: #references }

1. P. Giannozzi et al., J. Phys.: Condens. Matter 21, 395502 (2009).
2. E. R. Davidson, J. Comput. Phys. 17, 87 (1975).
3. L. Kleinman and D. M. Bylander, Phys. Rev. Lett. 48, 1425 (1982).
4. S. G. Louie, S. Froyen and M. L. Cohen, Phys. Rev. B 26, 1738 (1982).
5. J. P. Perdew, K. Burke and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996).
6. S. Ismail-Beigi, Phys. Rev. B 73, 233103 (2006).
7. J. Deslippe et al., Comput. Phys. Commun. 183, 1269 (2012).
8. A. R. Altman, S. Kundu and F. H. da Jornada, Phys. Rev. Lett. 132, 086401 (2024).

