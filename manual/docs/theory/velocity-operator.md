# Velocity operator

`psp.get_dipole_mtxels` The velocity matrix elements behind the q → 0 head, the optical dipoles and the QSGW head, and the Berry connection built from parallel-transport links.
{: .page-summary }

<div class="section-cards"></div>

<figure class="flowchart" markdown>
[![One-body Hamiltonian, velocity and Berry connection](../assets/figures/p1.png)](../assets/figures/p1.png)
<figcaption><strong>Fig. P1.</strong> Stage 2 differentiates the kinetic and nonlocal parts of $H_\bk$ with respect to $\bk$ and contracts them with the wavefunctions into $\langle m\bk|\hat v_a|n\bk\rangle$. Stage 3 overlaps neighbouring k points, takes the polar factor of each overlap as the link matrix, and forms the Berry connection from the links. Stage 1, the one-body Hamiltonian, belongs to the kin-ion utility.</figcaption>
</figure>

## <span class="n">1</span> The velocity operator {: #velocity }

Position is ill-defined in a periodic solid, but its interband matrix elements follow from the velocity. LORRAX computes the velocity analytically at each $\bk$, including the commutator of the nonlocal pseudopotential with $\hat{\mathbf r}$.
{: .intro }

### <span class="n">1.1</span> Commutator and k derivative {: #vel-def }

With $\hbar=1$ the velocity is the commutator of $H$ with position. In the Bloch representation $H_\bk=e^{-i\bk\cdot\br}He^{i\bk\cdot\br}$ it is the $\bk$ derivative:

$$
\hat{\mathbf v}=i[H,\hat{\mathbf r}]=\frac{\partial H_\bk}{\partial\bk} \tag{1.1}
$$

In Rydberg units ($m=1/2$) the kinetic energy is $\lvert\bk+\bG\rvert^2$ and its derivative is $2(\bk+\bG)$. $\bk$ is in bohr⁻¹, energies in Ry and velocities in Ry bohr. The local potential commutes with $\hat{\mathbf r}$ and drops out.

### <span class="n">1.2</span> The nonlocal commutator {: #vel-nl }

`vnl_ops.vnl_velocity_block_from_coefficients`
{: .in-code }

The nonlocal pseudopotential is separable, $V^{\mathrm{NL}}_\bk(\bG,\bG')=\sum_{\kappa,ij}Z_{\kappa i}(\bk+\bG)\,D_{ij}\,Z^*_{\kappa j}(\bk+\bG')$, with projector rows $Z_{\kappa i}(\mathbf K)=\beta_{\kappa i}(\lvert\mathbf K\rvert)\,S_{l_im_i}(\hat{\mathbf K})\,e^{-i\mathbf K\cdot\boldsymbol\tau_\kappa}$. Its projectors depend on $\bk+\bG$, so it does not commute with $\hat{\mathbf r}$. The derivative is analytic (forward-mode differentiation of the projector builder):

$$
\partial_{k_a}V^{\mathrm{NL}}_\bk=\sum_{\kappa,ij}\big[(\partial_{k_a}Z_{\kappa i})\,D_{ij}\,Z^*_{\kappa j}+Z_{\kappa i}\,D_{ij}\,(\partial_{k_a}Z_{\kappa j})^*\big]=-i\,[\hat r_a,V^{\mathrm{NL}}] \tag{1.2}
$$

The velocity written to `dipole.h5` is $\hat v_a=2(\bk+\bG)_a+\partial_{k_a}V^{\mathrm{NL}}_\bk=\partial H_\bk/\partial k_a$. Spin–orbit coupling enters only through the spin blocks $D^{\sigma\sigma'}_{ij}$.

`--skip-vnl` writes $\hat p_a=2(\bk+\bG)_a$ alone (BerkeleyGW's `use_momentum`). It is for comparison: the GW head refuses such a file (`GATE dft_head_dipole_provenance` on the full head, `GATE static_head_dipole_operator` on the direct and static head).

## <span class="n">2</span> Velocity and position forms of dipoles {: #forms }

The velocity is stored; every position matrix element is derived from it. The relation holds only for the Hamiltonian whose energies divide it.
{: .intro }

### <span class="n">2.1</span> Interband position matrix elements {: #forms-r }

Taking the matrix element of (1.1) between eigenstates of $H_\bk$ gives $v_{mn}=i(E_m-E_n)\,r_{mn}$, so for $m\neq n$:

$$
\langle m\bk|\hat{\mathbf r}|n\bk\rangle=\frac{\langle m\bk|\hat{\mathbf v}|n\bk\rangle}{i\,(E_{m\bk}-E_{n\bk})},\qquad m\neq n \tag{2.1}
$$

The intraband part of $\hat{\mathbf r}$ is the Berry connection of §3. If $H$ gains a nonlocal term that $\hat{\mathbf v}$ does not carry, such as the QP self-energy, (2.1) with QP energies is no longer exact (§2.3).

### <span class="n">2.2</span> Where each form is used {: #forms-use }

`absorption_common.slice_dipole_to_bse_window`
{: .in-code }

The q → 0 head uses the velocity form directly: $S_{ab}$ divides $v^{a*}_{mn}v^b_{mn}$ by $\Delta_{mn}(z^2-\Delta_{mn}^2)$, which equals the position form $\lvert r_{mn}\rvert^2$ times the response denominator ([The q → 0 head](q0-head.md#s-tensor)). The BSE converts each transition dipole to the position form by dividing by the DFT transition energy that `dipole.h5` carries:

$$
d^a_{cv\bk}=\frac{\langle c\bk|\hat v_a|v\bk\rangle}{E_{c\bk}-E_{v\bk}}=i\,\langle c\bk|\hat r_a|v\bk\rangle \tag{2.2}
$$

The factor $i$ is a global phase: it drops out of $\lvert\cdot\rvert^2$ and of the Haydock resolvent. A metal's Drude tensor uses the band diagonal $v_{nn}$ and the near-degenerate pairs ([Metals §3](metals.md#head-drude)).

### <span class="n">2.3</span> The quasiparticle velocity {: #forms-qp }

`gw.qsgw_head.qp_velocity`
{: .in-code }

A QSGW map works with $H=H^{\mathrm{DFT}}+\Delta H$, where $\Delta H=\Delta V^{\mathrm H}+\Sigma^{\mathrm{QSGW}}-V^{xc}$ is nonlocal. Its velocity has a second term, which in the band basis is the covariant derivative of §3.4:

$$
\hat{\mathbf v}^{\mathrm{QSGW}}=\hat{\mathbf v}^{\mathrm{DFT}}+i[\Delta H,\hat{\mathbf r}],\qquad
\big(i[\Delta H,\hat{\mathbf r}]\big)_{mn\bk}=(D_\bk\Delta H)_{mn} \tag{2.3}
$$

The QP velocity is $U^\dagger(v^{\mathrm{DFT}}+D_\bk\Delta H)\,U$, with $U$ the rotation from DFT to QP states. `sc_head_update` selects which velocity each map uses for the head, and the accepted final map writes the same velocity to `dipole_qsgw.h5` ([Quasiparticle self-consistency](qsgw.md)). Its schema default is `off`, but left unnamed on a self-consistent tangential deck with the head on it resolves to `parallel_transport` when `parallel_transport_file` exists, else to `dft_velocity` when `dipole.h5` exists.

## <span class="n">3</span> Berry connection from parallel transport {: #berry }

$\Delta H$ is known only on the k grid, in a band frame chosen independently at each $\bk$. A plain difference of matrices at neighbouring points compares two unrelated frames. The links carry one frame into the next before differencing.
{: .intro }

### <span class="n">3.1</span> Overlaps {: #berry-overlap }

`common.parallel_transport.make_cross_k_overlap`
{: .in-code }

For each step $\mathbf b$ to a neighbouring grid point the overlap of the cell-periodic states is evaluated on the plane-wave coefficients, with the reciprocal-lattice shift that returns $\bk+\mathbf b$ to the zone:

$$
M_{mn}(\bk,\mathbf b)=\langle u_{m\bk}|u_{n,\bk+\mathbf b}\rangle \tag{3.1}
$$

The overlaps are formed on an outer band set, by default $\min(N_{\mathrm{WFN}},\lceil1.25\,n_b\rceil)$ bands (`--parallel-transport-bands`), so that states leaving the window between $\bk$ and $\bk+\mathbf b$ collapse singular values above the head bands only.

### <span class="n">3.2</span> Links by polar decomposition {: #berry-links }

The link is the unitary closest to $M$, its polar factor. With the singular-value decomposition $M=V\Sigma W^\dagger$:

$$
L_{\mathbf b}(\bk)=VW^\dagger \tag{3.2}
$$

Singular values below `--parallel-transport-rcond` (default $10^{-10}$) times the largest are dropped, giving a partial isometry. $L_{\mathbf b}(\bk)$ maps the frame at $\bk+\mathbf b$ onto the frame at $\bk$; it is the discrete parallel transport. The backward link is $L_{-\mathbf b}(\bk)=L_{\mathbf b}(\bk-\mathbf b)^\dagger$, and double steps are composed links.

### <span class="n">3.3</span> The connection {: #berry-conn }

`common.parallel_transport.link_connection`
{: .in-code }

Per step the directional connection is a centred difference of links, second order on mesh lines with 3–4 points and fourth order with at least 5:

$$
A_{\mathbf b}=\tfrac{i}{2}\big[L_{+\mathbf b}-L_{-\mathbf b}\big],\qquad
A_{\mathbf b}=\tfrac{i}{12}\big[-L_{+2\mathbf b}+8L_{+\mathbf b}-8L_{-\mathbf b}+L_{-2\mathbf b}\big] \tag{3.3}
$$

To first order $M\approx1-i\,\mathbf b\cdot A$, so $A_{\mathbf b}\to\mathbf b\cdot A$ with $A_{mn}=i\langle u_{m\bk}|\nabla_\bk u_{n\bk}\rangle$, the non-Abelian Berry connection in the frame of the stored states. Reduced components are $A_a=\sum_{\mathbf b}c_{\mathbf b a}A_{\mathbf b}$ with the stencil weights of §3.5; each is Hermitized and converted to Cartesian.

### <span class="n">3.4</span> The covariant derivative {: #berry-cov }

`common.parallel_transport.link_covariant_derivative`
{: .in-code }

A band operator at a neighbour is transported into the frame at $\bk$, $T_{+\mathbf b}O=L_{\mathbf b}(\bk)\,O(\bk+\mathbf b)\,L_{\mathbf b}(\bk)^\dagger$, and only then differenced, $D_{\mathbf b}O=\tfrac12(T_{+\mathbf b}-T_{-\mathbf b})O$ (or the fourth-order form). This is the covariant derivative, and it gives the commutator of $\hat{\mathbf r}$ with any band operator:

$$
D_\bk O=\partial_\bk O-i[A_\bk,O],\qquad [\hat{\mathbf r},O]_{mn\bk}=i\,(D_\bk O)_{mn}=i\,\partial_\bk O_{mn}+[A_\bk,O]_{mn} \tag{3.4}
$$

Both terms of $\partial_\bk O-i[A,O]$ depend on the gauge and only their sum does not; the transported difference never forms them separately, so no two large gauge-dependent terms cancel on the finite grid.

### <span class="n">3.5</span> The stencil {: #berry-stencil }

`common.parallel_transport.link_stencil`
{: .in-code }

The steps and weights satisfy the Marzari–Vanderbilt completeness condition, so the gradient is exact for linear functions with error $O(b^2)$:

$$
\sum_{\mathbf b}w_{\mathbf b}\,b_a\,b_c=\delta_{ac} \tag{3.5}
$$

Candidates are mesh vectors $(n/N)\cdot B$ with $\lvert n_i\rvert\le3$, taken in whole shells of equal length, shortest first; a shell that is parallel to an accepted one, lacks 3 points on a mesh line or adds no rank is skipped. Without a complete set the run refuses (`GATE pt_link_shell_incomplete`). Whole shells are closed under the lattice point group, so the $O(b^2)$ error has the crystal's symmetry.

| Lattice | Stored steps (one per ± pair) |
|---|---|
| orthogonal | the three axes |
| bcc | 6: $(100)$, $(010)$, $(001)$, $(10\bar1)$, $(1\bar10)$, $(01\bar1)$ |
| fcc | 4: the three axes and $(111)$ |
| hexagonal slab | 3: $(100)$, $(010)$, $(1\bar10)$ |
| hexagonal bulk | 4: the slab's three and the $c$ axis |

A collapsed axis (one k point, such as a slab normal) uses the position operator itself: $A_a=Z_a=2\pi\langle m|\mathrm{wrap}(f_a-f^0_a)|n\rangle$, with the branch cut $f^0_a$ at the centre of the largest vacuum gap; occupied charge above $10^{-3}$ in the middle of the gap refuses (`GATE pt_collapsed_axis_cut_density`). A two-point axis has no stencil, and the dipole utility writes no link artifact on such a grid.

### <span class="n">3.6</span> Validating the links {: #berry-valid }

`complete_velocity_validation`
{: .in-code }

$H^{\mathrm{DFT}}$ is diagonal in its own basis, so the links must reproduce the computed velocity from the energies alone. The dipole utility checks this identity on the elements the head reads:

$$
v^{\mathrm{DFT}}=D_\bk\,\mathrm{diag}(E^{\mathrm{DFT}}),\qquad
\epsilon_{\mathrm{link}}=\frac{\lVert v^{\mathrm{links}}-v^{\mathrm{DFT}}\rVert}{\lVert v^{\mathrm{DFT}}\rVert} \tag{3.6}
$$

The check passes when the relative error of the reconstructed head response is at most `--parallel-transport-validation-rtol` (default $5\times10^{-3}$) and the transition overlap is at least one minus that value. Otherwise it warns and still writes `parallel_transport.h5`. `--parallel-transport-validation-atol` enters only the per-block diagnostics. $\epsilon_{\mathrm{link}}$ is a finite-difference error in $\bk$ and falls as the grid is refined. Only $D_\bk\Delta H$ goes through the links, so each QSGW map bounds the head's link error by $\epsilon_{\mathrm{link}}\lVert D_\bk\Delta H\rVert/\lVert v^{\mathrm{DFT}}\rVert$, printed as information.

!!! note
    A head window hybridized with bands outside it (an outer-link singular value below 0.5 among the head bands) cannot serve $D_\bk\Delta H$ (`GATE pt_head_window_hybridized`); the run then uses $U^\dagger v^{\mathrm{DFT}}U$ on every map and says so in one log line.

## <span class="n">4</span> Sign conventions {: #conventions }

Every object on this page has one convention, stamped where it is stored.
{: .intro }

### <span class="n">4.1</span> Summary {: #conv-table }

| Object | Convention | Stored as |
|---|---|---|
| velocity | $\hat v_a=\partial H_\bk/\partial k_a=\hat p_a-i[\hat r_a,V^{\mathrm{NL}}]$ | `dipole_cart` $(3,n_k,n_b,n_b)$ |
| matrix element | `dipole_cart[a,k,m,n]` $=\langle m\bk\vert\hat v_a\vert n\bk\rangle$, Ry bohr | `dipole.h5` |
| link | $L_{\mathbf b}(\bk)\,X(\bk+\mathbf b)\,L_{\mathbf b}(\bk)^\dagger$ carries $X$ to $\bk$ | `links_ibz` |
| connection | $A_{mn}=i\langle u_{m\bk}\vert\nabla_\bk u_{n\bk}\rangle$, Hermitian | `berry_connection_cart` |
| covariant derivative | $D_\bk O=\partial_\bk O-i[A,O]$; $[\hat{\mathbf r},O]=i\,D_\bk O$ | — |
| position | $r_{mn}=-i\,v_{mn}/(E_m-E_n)$, $m\neq n$ | derived on read |
| BSE dipole | $d_{cv\bk}=v_{cv\bk}/(E_c-E_v)=i\,r_{cv\bk}$, DFT energies | in memory |

The velocity is computed, not differenced: it is exact at each $\bk$ for the given states. Links are stored for the irreducible wedge only when the steps are the three axes and the point group permutes them; otherwise for the full grid.

## Notation {: #notation }

| Symbol | Meaning |
|---|---|
| $\hat{\mathbf v}$, $\hat v_a$ | velocity operator; Cartesian component |
| $Z_{\kappa i}(\mathbf K)$, $D_{ij}$ ($D^{\sigma\sigma'}_{ij}$), $\beta_{\kappa i}$, $\kappa$ | nonlocal projector rows; couplings (spin blocks with spin–orbit coupling); radial projectors; atom index |
| $u_{n\bk}$ | cell-periodic part of $\psi_{n\bk}$ |
| $\mathbf b$, $w_{\mathbf b}$ | stencil step; its weight |
| $M_{mn}(\bk,\mathbf b)$ | overlap of cell-periodic states at neighbouring k |
| $L_{\mathbf b}(\bk)$ | link matrix, the polar factor of $M$ |
| $A_\bk$, $A_{\mathbf b}$ | Berry connection; its directional component |
| $D_\bk$ | covariant derivative |
| $\Delta H$, $U$ | QSGW change of $H$ in the DFT basis; DFT-to-QP rotation |
| $Z_a$ | position matrix along a collapsed axis |
| $\epsilon_{\mathrm{link}}$ | relative link error on the head's elements |

## References {: #refs }

1. E. I. Blount, Solid State Phys. 13, 305 (1962).
2. A. F. Starace, Phys. Rev. A 3, 1242 (1971).
3. A. J. Read and R. J. Needs, Phys. Rev. B 44, 13071 (1991).
4. R. Del Sole and R. Girlanda, Phys. Rev. B 48, 11789 (1993).
5. N. Marzari and D. Vanderbilt, Phys. Rev. B 56, 12847 (1997).
6. S. Ismail-Beigi, E. K. Chang and S. G. Louie, Phys. Rev. Lett. 87, 087402 (2001).

