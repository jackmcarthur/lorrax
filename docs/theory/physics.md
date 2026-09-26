# Core ISDF and GW theory

This page states the equations every GW mode shares, what each costs, and
where each hands off to the page that owns its detail. Units are Rydberg.
Sizes: \(N_k\) k-points (\(N_q=N_k\)), \(N_b\) bands, \(N_\mu\) interpolation
points, \(N_r\) real-space grid points, \(N_G\) plane waves in a
\(\mathbf q+\mathbf G\) sphere, \(N_\tau\) time nodes. At fixed \(N_k\),
\(N_b\), \(N_\mu\) and \(N_r\) all grow linearly with the number of atoms.

## 1. Transition densities

For band \(n\), crystal momentum \(\mathbf k\) and spinor component \(a\),

$$
\psi_{n\mathbf k a}(\mathbf r)
=\frac{1}{\sqrt{N_r}}\sum_{\mathbf G}
c_{n\mathbf k a}(\mathbf G)\,
e^{i(\mathbf k+\mathbf G)\cdot\mathbf r},
\qquad
\rho_{mn\mathbf k}^{\mathbf q}(\mathbf r)
=\sum_a
\psi^*_{m,\mathbf k-\mathbf q,a}(\mathbf r)\,
\psi_{n\mathbf k a}(\mathbf r).
$$

Coulomb screening couples the spin-traced charge density; Green functions and
the self-energy keep their spinor indices open until the band projection. The
four-current (bispinor) formulation adds three current channels with their own
vertices ([four-current wiring](../architecture/four_current_wiring.md)); the
ISDF argument below is unchanged per channel.

## 2. ISDF factorization

Every GW pair sum runs over \(\rho^{\mathbf q}_{mn\mathbf k}\). ISDF replaces
the pair index by \(N_\mu\) interpolation points \(\mathbf r_\mu\), chosen by
k-means on the band density:

$$
\rho_{mn\mathbf k}^{\mathbf q}(\mathbf r)
\approx
\sum_\mu \zeta_{q\mu}(\mathbf r)\,
\rho_{mn\mathbf k}^{\mathbf q}(\mathbf r_\mu).
$$

The interpolation vectors \(\zeta_{q\mu}\) do not depend on the band pair. The
least-squares fit over all pairs \(m\in L\), \(n\in R\) has normal equations
\(C_q\zeta_q=Z_q\) with

$$
P^{A}_{\mathbf k}(\mu,\mathbf r)=\sum_{n\in A}
\psi_{n\mathbf k}(\mathbf r_\mu)\,\psi^*_{n\mathbf k}(\mathbf r),
\qquad
C_q(\mu,\nu)=\sum_{\mathbf k}
P^{L}_{\mathbf k-\mathbf q}(\mu,\nu)\,
\overline{P^{R}_{\mathbf k}(\mu,\nu)},
\qquad
Z_q(\mu,\mathbf r)=\sum_{\mathbf k}
P^{L}_{\mathbf k-\mathbf q}(\mu,\mathbf r)\,
\overline{P^{R}_{\mathbf k}(\mu,\mathbf r)}.
$$

By default \(L\) holds all occupied bands plus the Σ conduction window, and
\(R\) holds the Σ occupied window plus all empty bands: exactly the pairs that
χ₀ and Σ consume. The band sum is a GEMM, and the k sum is a product in
\((\mu,\nu)\) followed by a lattice correlation, which one FFT over the flat k
axis evaluates for every q at once.

| step | cost |
|---|---|
| \(P^{L,R}_{\mathbf k}(\mu,\nu)\), then \(C_q\) for all q | \(N_kN_bN_\mu^2+N_\mu^2N_k\log N_k\) |
| \(Z_q(\mu,\mathbf r)\) | \(N_kN_bN_\mu N_r\) |
| factor and solve | \(N_qN_\mu^3+N_qN_\mu^2N_r\) |

The charge solve is rank-truncated under the one
[rank criterion](../dev/rank_truncation_policy.md); current channels use a
ridge-regularized LU. The charge fit forms \(Z_q\) directly in G space by
[route G](../architecture/zeta_fit_mubatch.md) and applies \(C_q^+\) tile by
tile over G; the current channels build and solve on orbit-closed real-space
tiles and add each solved tile into the \(\mathbf q+\mathbf G\) sphere.
Either way the stored object is
\(\widetilde z[q_{\rm irr},\mu,G]\), \(\mathcal O(N_qN_\mu N_G)\), never
\(\zeta\) on the full real-space grid. Equations and the irreducible-q
cascade: [G-flat ζ and V](isdf-zeta-vq.md). Carriers, band windows and tiles:
[face-ψ ζ fitting](../architecture/zeta_fit_face_psi_cct.md).

After this step no band pair appears again. Response, screening and
self-energy are \(N_\mu\times N_\mu\) matrix algebra with band sums inside
GEMMs, which is what makes the method cubic in system size.

## 3. Coulomb matrix

With the Bloch phase removed, \(z_{q\mu}(\mathbf r)=e^{-i\mathbf q\cdot\mathbf r}
\zeta_{q\mu}(\mathbf r)\), and \(\widetilde z_{q\mu}(\mathbf G)\) its Fourier
coefficients,

$$
V_{q,\mu\nu}
=\sum_{\mathbf G\in{\rm sphere}(q)}
\widetilde z^*_{q\mu}(\mathbf G)\,
v(\mathbf q+\mathbf G)\,
\widetilde z_{q\nu}(\mathbf G),
\qquad \text{cost } N_qN_\mu^2N_G .
$$

The [`vcoul` service](../services/vcoul.md) owns \(v\): dimensional
truncation and the \(\mathbf q+\mathbf G=0\) cell average. \(V_q\) is stored
flat in q with its two centroid axes on the two mesh axes,
`P(None,'x','y')`.

## 4. Independent-particle response

With \(M^\mu_{vc\mathbf k}(q)=\rho^{\mathbf q}_{vc\mathbf k}(\mathbf r_\mu)\)
and \(\Delta_{vc\mathbf k}=\epsilon_{c\mathbf k}-\epsilon_{v,\mathbf k-\mathbf q}\),
the time-reversal-symmetric insulating response is

$$
\chi^0_{q,\mu\nu}(z)
=\sum_{vc\mathbf k}
M^\mu_{vc\mathbf k}\,M^{\nu *}_{vc\mathbf k}
\left[\frac{1}{z-\Delta_{vc\mathbf k}}-\frac{1}{z+\Delta_{vc\mathbf k}}\right].
$$

Summed directly this costs \(N_kN_qN_vN_cN_\mu^2\), quartic in system size.
LORRAX never forms a transition pair. It writes the denominator as a
quadrature over positive times,

$$
\frac{1}{\Delta}=\int_0^\infty e^{-\Delta t}\,dt\approx\sum_\ell w_\ell\,e^{-\Delta t_\ell},
$$

and \(e^{-\Delta t}=e^{-(\epsilon_c-E_F)t}\,e^{(\epsilon_v-E_F)t}\) factorizes
into one empty and one occupied Green function on the interpolation points,

$$
G^{\rm emp}_{\mathbf k}(\mu,\nu;t)=\sum_{c}\psi_{c\mathbf k}(\mathbf r_\mu)\psi^*_{c\mathbf k}(\mathbf r_\nu)e^{-(\epsilon_{c\mathbf k}-E_F)t},
\qquad
G^{\rm occ}_{\mathbf k}(\mu,\nu;t)=\sum_{v}\psi_{v\mathbf k}(\mathbf r_\mu)\psi^*_{v\mathbf k}(\mathbf r_\nu)e^{(\epsilon_{v\mathbf k}-E_F)t},
$$

$$
\chi^0_{q,\mu\nu}(0)=-2\sum_\ell w_\ell\sum_{\mathbf k}
G^{\rm emp}_{\mathbf k}(\mu,\nu;t_\ell)\,G^{\rm occ}_{\mathbf k-\mathbf q}(\nu,\mu;t_\ell).
$$

Each node costs two GEMMs (\(N_kN_bN_\mu^2\)) and one lattice FFT
(\(N_\mu^2N_k\log N_k\)); the certified node count is
\(N_\tau=\mathcal O(\log(\Delta_{\max}/\Delta_{\min})\log\epsilon^{-1})\).
An imaginary frequency multiplies the kernel by \(\cos\omega t\). Rules and
their certificates: [minimax quadrature](minimax-quadrature.md) (static and
plasmon-pole), [MPA sampling](THEORY_mpa_implementation.md) (complex lines),
[compact noncrossing response](response-laplace.md) (the shared-pole bank).

Two cases change the kernel, not the factorization:

- **Broken time reversal.** The two particle-hole orientations then carry
  different residues, and \(\chi^0_q(i\omega)\) has an anti-Hermitian,
  magnetization-odd part that vanishes only at \(\omega=0\). The ordered
  routes keep both orientations, \(\chi^0_q(z)=F_q(z)+F_{-q}(-z^*)^*\), from
  one contour sweep plus a q-negation gather
  ([MPA §2.1](THEORY_mpa_implementation.md#21-ordered-orientations-when-time-reversal-is-broken);
  [GN-PPM derivation](../dev/notes/DERIVATION_gnppm_nonhermitian.md)). Time
  reversal is measured from the wavefunctions (`SymMaps.trs_allowed`); no
  deck key can assert it.
- **Fractional occupations.** Occupation becomes a weight on each branch and
  band energies stay signed; the intraband \(\mathbf q\to0\) limit enters
  through the head. Owner: [metallic MPA screening](metallic-mpa-screening.md).

## 5. Screening

$$
W_q(z)=\left[I-V_q\chi^0_q(z)\right]^{-1}V_q,
\qquad
W^c_q(z)=W_q(z)-V_q,
$$

one LU per q and requested frequency: \(N_qN_zN_\mu^3\) for \(N_z\)
frequencies, local (per-q pivoted) or distributed over the 2-D mesh
(the `linalg` dial). The self-energy model decides which \(z\) are requested
(§6).

## 6. Self-energy

\(\Sigma(1,2)=iG(1,2)W(1^+,2)\). In the ISDF basis a time node builds
\(G_{\mathbf k}(t)\) and \(W_{\mathbf q}(t)\), and the k convolution is again a
product in \((\mu,\nu)\) plus a lattice FFT:

$$
\Sigma_{\mathbf k}(\mu,\nu;t)=i\sum_{\mathbf q}
G_{\mathbf k-\mathbf q}(\mu,\nu;t)\,W_{\mathbf q}(\mu,\nu;t),
\qquad
\Sigma_{\mathbf k,ij}
=\sum_{ab\mu\nu}
\psi^*_{i\mathbf k a}(\mathbf r_\mu)\,
\Sigma_{\mathbf k,a\mu,b\nu}\,
\psi_{j\mathbf k b}(\mathbf r_\nu).
$$

A real or imaginary part must be taken before the band map unless the map is
shown to commute with it: in general \(K[\operatorname{Re}X]\neq\operatorname{Re}K[X]\).

For a pole model \(W^c=\sum_p R_p\,[(z-\Omega_p)^{-1}-(z+\Omega_p)^{-1}]\), the
correlation part at real \(\omega\) is a sum of terms
\(\psi_n\psi_n^\dagger\circ R_p/(\omega-\epsilon_n-\sigma_b\Omega_p)\),
\(\sigma_b=+1\) on empty and \(-1\) on occupied branches. Each reciprocal is replaced
by an exponential or sine sum in \(t\), so every term factorizes into one
band-windowed \(G(t)\) and one pole-windowed \(W(t)\): one spatial contraction
per (window, node) pair, never a state-by-pole loop. A sign-definite window
needs \(\mathcal O(\log)\) nodes; a window whose denominator crosses zero needs
nodes linear in bandwidth\(/\eta\). Owner: [the Σ(ω) quadrature
problem](sigma-quadrature-problem.md).

The modes (`compute_mode`, and `sigma_w_model` under `mpa`) differ only in
which \(W\) samples they request and how they turn them into poles:

| mode | \(W\) requested | frequency model | owner |
|---|---|---|---|
| `x_only` | \(V\) | bare exchange, \(-G^{\rm occ}\circ V\) | — |
| `cohsex` | \(W(0)\) | static: \(\Sigma_{SX}=-G^{\rm occ}\circ W(0)\), \(\Sigma_{COH}=\tfrac12 G^{\rm all}\circ W^c(0)\) | — |
| `gn_ppm` | \(W(0)\), \(W(i\omega_p)\) | one pole per element; with broken time reversal two Hermitian residues \(R_\pm=B\pm D\), \(R_+\) on empty and \(R_-\) on occupied branches | [minimax](minimax-quadrature.md), [derivation](../dev/notes/DERIVATION_gnppm_nonhermitian.md) |
| `hl_ppm` | \(W(0)\) | one pole per element fixed by static screening and the f-sum rule | [HL-GPP](hl-gpp-derivation.md) |
| `mpa`, `sigma_w_model = mpa` | two complex-frequency lines | several complex poles per element, Loewner fit | [MPA](THEORY_mpa_implementation.md) |
| `mpa`, `sigma_w_model = shared_pole` | a response sample bank | \(W^c_q(z)=\sum_j b_jb_j^\dagger/(z^2-\Omega_j^2)\): one real pole set per q shared by all elements, \(16N_\mu K_q\) bytes per stored q | [shared-pole W](shared-pole-w-model.md) |

The \(\mathbf q\to0\) head and the four-current channels add to this table per
[four-current heads](four-current-head-corrections.md#four-current-phase-status).

## 7. Long-wavelength terms

At \(\mathbf q=0\) the Coulomb divergence multiplies a response that vanishes
as \(q^2\). LORRAX stores the nonsingular body and adds the analytic head
separately, from

$$
\chi_{00}(\mathbf q,z)=q_aS_{ab}(z)q_b+\mathcal O(q^3)
$$

in Cartesian reciprocal coordinates ([S-tensor convention](s-tensor-convention.md)).
The exchange head is direction dependent and produces the
longitudinal-transverse exciton splitting
([LT splitting](lt-exchange-head.md)).

## 8. Quasiparticle Hamiltonian

`kin_ion.h5` stores \(T+V_{\rm ion}\); the [direct field](hartree.md) is built
live from the current orbitals and occupations:

$$
H_{\rm QP}=(T+V_{\rm ion})+V_{\rm H}+\Sigma^{\rm QSGW}_{xc},
\qquad
\Sigma^{\rm QSGW}_{xc,ij}(\mathbf k)=\operatorname{Herm}\,\tfrac12\!\left[\Sigma_{xc,ij}(\mathbf k,E_i)+\Sigma_{xc,ij}(\mathbf k,E_j)\right].
$$

A dynamic \(\Sigma_c(\omega)\) is interpolated on its real-frequency grid; an
energy outside that grid takes \(\Sigma_c(\omega=0)\). `qp_solver` selects
where the energies come from: `one_shot_dft` (at \(E_{\rm DFT}\)),
`fixed_point` (after a diagonal on-shell solve) or `self_consistent` (QSGW,
which rebuilds χ₀, \(W\) and Σ from the rotated orbitals every map). In
QSGW, bands outside the Σ window follow a scissor law, not a self-energy. Owner:
[self-consistency](../self_consistency.md).

## 9. Layout invariants

| object | layout | reason |
|---|---|---|
| \(\psi_{\mathbf k}(\mu)\) faces | flat k; centroid and band axes on the two mesh axes | band GEMMs stay local |
| \(V_q,\chi_q,W_q\) | `P(None,'x','y')` | both matrix axes distributed |
| \(\Sigma_c(\omega)_{\mathbf k,ij}\) | `P(None,None,'x','y')` over \((\omega,\mathbf k,i,j)\) | no replicated frequency cube |
| \(\widetilde z\), pole stores | irreducible q wedge | symmetry commutes with their construction |

Every lattice FFT convolution consumes full-zone k/q data. Irreducible-zone
and time-reversal reductions apply only to non-FFT work and storage, followed
by an explicit unfold ([symmetry](symmetry.md)). Storage uses the wedge only
when the centroid set is closed under the full symmetry table; otherwise
every q is computed and stored, never partially unfolded. No rank
materializes an \(N_\mu^2\) matrix to write or fit it.

## 10. What the formulation refuses

The key-level contract is the [input reference](../input_reference.md); this
table gives the physical reason for each refusal.

| case | refusal | why |
|---|---|---|
| metal with `cohsex`, `gn_ppm` or `hl_ppm` | `GATE gn_ppm_refuses_metals`, `GATE fractional_occupations_require_mpa` | these split bands by a 0/1 step at \(E_F\); metals use `mpa` (production: `sigma_w_model = shared_pole`) or `x_only` |
| time-reversal-broken metal with `sigma_w_model = mpa` | `GATE mpa_ordered_metal` | the elementwise metal fit carries one residue and no odd channel; the shared-pole ordered store does |
