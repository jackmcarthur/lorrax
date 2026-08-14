# S-tensor convention

The long-wavelength density response uses one canonical rank-two object:

$$
\chi_{00}(\mathbf q\to0,\omega)
=q_aS_{ab}(\omega)q_b.
$$

Here \(\mathbf q\) is Cartesian reciprocal momentum in \(1/\mathrm{bohr}\),
and \(S\) is the symmetric complex Cartesian \(q^2\)-coefficient. It contains
no extra factor of two.

`common.chi_from_dipole.compute_S_omega` returns this convention. The Coulomb
head contracts it against Cartesian mini-BZ samples,

$$
qSq=\operatorname{einsum}(\texttt{'qi,ij,qj->q'},q,S,q),
$$

so the consumer independently fixes both the coordinate frame and
normalization.

The Sternheimer builder naturally produces a crystal-coordinate Hessian
\(H\):

$$
\chi_{00}
=\tfrac12 q_i^{\mathrm{crys}}H_{ij}q_j^{\mathrm{crys}}.
$$

It converts before writing,

$$
S
=\tfrac12 B^{-1}HB^{-\mathsf T},
\qquad
B=\texttt{blat}\,\texttt{bvec},
$$

where the rows of \(B\) are Cartesian reciprocal basis vectors. The dataset
is stamped
`s_tensor_convention = "cartesian_q2_coefficient"`; the raw Hessian is not a
second public convention.

Any new long-wavelength tensor must state its frame and the power of q whose
coefficient it represents. For example,
\(\mathsf M_{ab}=\langle v(\mathbf q)q_aq_b\rangle_{\mathrm{cell}}\) is a
Cartesian coefficient of a dipole bilinear, not another spelling of \(S\).
