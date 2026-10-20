# S-tensor convention

The long-wavelength density response uses one canonical rank-two object:

$$
\chi_{00}(\mathbf q\to0,\omega)
=q_aS_{ab}(\omega)q_b .
$$

Here \(\mathbf q\) is Cartesian reciprocal momentum in \(1/\mathrm{bohr}\),
and \(S\) is the complex Cartesian \(q^2\)-coefficient in
\(1/(\mathrm{Ry}\,\mathrm{bohr}^2)\), so \(v(\mathbf q)\,q^{\mathsf T}Sq\) is
dimensionless. It contains no extra factor of two. Only the
coordinate-symmetric part is observable under \(q_aq_b\). The antisymmetric
part is nonzero on a magnet, and producers that feed a Γ-cell solve remove
it at construction
(`gw.head_correction.canonicalize_static_gauge_q2_tensor`).

Every producer returns this convention:
`common.chi_from_dipole.compute_S_omega` (the `dipole.h5` head,
`gw.head_correction.build_S_cart_omega`) and
`gw.qsgw_head.head_s_tensor_sharded` (the sharded current-velocity head).
The sum-over-pairs formula is
[four-current heads §3.1](four-current-head-corrections.md#charge-head-objects).
Consumers contract it against Cartesian mini-BZ samples,

$$
qSq=\operatorname{einsum}(\texttt{'qi,ij,qj->q'},q,S,q),
$$

so the consumer fixes both the coordinate frame and the normalization.

The Sternheimer builder naturally produces a crystal-coordinate Hessian
\(H\):

$$
\chi_{00}
=\tfrac12 q_i^{\mathrm{crys}}H_{ij}q_j^{\mathrm{crys}} .
$$

`psp.run_sternheimer` converts it before writing,

$$
S
=\tfrac12 B^{-1}HB^{-\mathsf T},
\qquad
B=\texttt{blat}\,\texttt{bvec},
$$

where the rows of \(B\) are Cartesian reciprocal basis vectors. The dataset
`s_tensor_q0` is stamped
`s_tensor_convention = "cartesian_q2_coefficient"`. The raw Hessian is not a
second public convention.

Any new long-wavelength tensor must state its frame and the power of q whose
coefficient it represents. For example,
\(\mathsf M_{ab}=\langle v(\mathbf q)q_aq_b\rangle_{\mathrm{cell}}\) is a
Cartesian coefficient of a dipole bilinear, not another spelling of \(S\)
([LT splitting](lt-exchange-head.md)).

## Sum set and cell set

The direct head is a Brillouin-zone sum,

$$
S_{\mathrm d,ab}(z)=\frac{4}{\Omega N_k n_s n_\sigma}\sum_{k}\sum_{mn}
\frac{(f_{nk}-f_{mk})\,v^{a*}_{mn}v^{b}_{mn}}{\Delta_{mn}(z^2-\Delta_{mn}^2)},
\qquad \Delta_{mn}=E_{mk}-E_{nk},
$$

evaluated at every response node \(z\) (`qsgw_head.head_s_tensor_sharded`).
Two k sets enter the head and are kept apart.

- The **sum set** is the k set of this sum, with its own \(N_k\). It is the
  GW mesh unless the deck names `head_dense_file`, a `dipole.h5` of a denser
  NSCF of the same crystal and Hamiltonian (`qsgw_head.head_source`). The dense
  file is read whole: its full-zone velocity and `band_energies`, as the
  producer stored them.
- The **cell set** is the GW mesh. It owns the mini-BZ cell (the Sobol draws,
  \(\langle v\rangle\), the Coulomb moment), \(W(\Gamma)\) and the wings
  \(Y, Z\). A dense file never changes it.

Because \(S\propto\Delta^{-3}\), the direct head converges slowly in \(k\):
Si 4³ gives \(S_\mathrm d(0)=-0.92414\) and 8³ gives \(-0.54561\)
bohr² Ry⁻¹ (ε without local fields 24.23 against 14.71). The band sum is not
the limit: 28 conduction spinor bands are within 0.06 % of 92
(`HEAD_DENSE_MIN_CONDUCTION`).

## Scaled fold

The local-field fold of the GW mesh, \(F=YW(\Gamma)Z/\Omega\), is a sum over
the same transitions as \(S_\mathrm d\). Adding it unchanged to a dense
\(S_\mathrm d\) is inconsistent. With a dense set the head is

$$
S_\mathrm{eff}=S_{\mathrm d,\mathrm{sum}}
+\tfrac12\left(MF+FM^{\mathsf T}\right),
\qquad
M(z)=S_{\mathrm d,\mathrm{sum}}(z)\,S_{\mathrm d,\mathrm{cell}}(z)^{-1},
$$

at every node (`head_correction.scaled_head_fold`; \(M\) is
`IterationHeadResponse.fold_scale`). It holds the local-field fraction
\(S_\mathrm d^{-1}F\) at its GW-mesh value. It is exact when
\(S_{\mathrm d,\mathrm{sum}}=rS_{\mathrm d,\mathrm{cell}}\), covariant
under the point group, and linear in \(F\). Without a dense set
\(M\) is absent and the fold is \(S_\mathrm d+F\), bit for bit.

The fraction is not k-independent. On Si it is 9.5 % of \(S_\mathrm d\) at
4³ and 10.6 % at 8³. The scaled fold with the 8³ sum set on the 4³ cell
gives ε_M = 13.41 against the native 8³ value of 13.25 (+1.2 %). The
additive fold \(S_{\mathrm d,\mathrm{sum}}+F\) gives 12.51 (−5.6 %) and is
not offered. The scaled fold is undefined for a metal, whose fold carries the
Fermi-surface wing, so a dense set refuses on a metal
(`GATE head_dense_metal`).

Refusals of a dense set: another velocity operator, representation, spin or
cutoff (`GATE head_dense_operator`, also at parse time for `head_correction =
off`, bispinor heads and SC heads rebuilt each map); another cell or atoms
(`GATE head_dense_lattice`); a band window without every occupied band, the
same filling, and `HEAD_DENSE_MIN_CONDUCTION` conduction bands
(`GATE head_dense_window`); a file not authenticated by its own WFN
(`GATE dft_head_dipole_provenance`). A file of the GW mesh's own WFN is the
GW mesh.
