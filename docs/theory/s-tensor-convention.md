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
