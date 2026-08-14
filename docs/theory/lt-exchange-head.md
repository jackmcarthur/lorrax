# Long-range exchange head and LT splitting

The electron-hole exchange kernel has one nonanalytic contribution at the zone
center. Its direction dependence produces longitudinal-transverse (LT)
exciton splitting. Rydberg units are used below:
\(v(\mathbf q)=8\pi/(\Omega q^2)\) in three dimensions.

## 1. The finite \(0\times\infty\) limit

For transition \(t=(c,v,\mathbf k)\),

$$
K^x_{tt'}(\mathbf Q)
=\sum_{\mathbf G}
M_t^*(\mathbf Q,\mathbf G)
v(\mathbf Q+\mathbf G)
M_{t'}(\mathbf Q,\mathbf G).
$$

Every \(\mathbf G\ne0\) term is smooth. At \(\mathbf G=0\),
orthogonality gives \(M_t(0,0)=0\), while

$$
M_t(\mathbf Q,0)
=-i\mathbf Q\cdot\mathbf d_t+\mathcal O(Q^2).
$$

The Coulomb divergence cancels the two powers of \(Q\):

$$
K^{x,\mathrm{head}}_{tt'}(\hat{\mathbf q})
=\frac{8\pi}{\Omega}
(\hat{\mathbf q}\cdot\mathbf d_t)^*
(\hat{\mathbf q}\cdot\mathbf d_{t'})
\qquad (3\mathrm D).
$$

The magnitude has a finite limit, but the direction does not. For a fixed
\(\hat{\mathbf q}\), this is an outer product. Within a degenerate bright
multiplet it shifts the one dipole combination parallel to
\(\hat{\mathbf q}\) and leaves transverse combinations unchanged.

With a two-dimensional truncated Coulomb interaction, the singular factor
vanishes linearly with in-plane \(|\mathbf Q|\). The dispersion is still
nonanalytic but continuous rather than discontinuously split at the origin.

## 2. Finite q and a mini-BZ average

At a sampled nonzero \(\mathbf Q\), the ordinary transition vertex already
contains its exact direction and finite-q form factor. Injecting an additional
dipole head there would double count it.

At \(\Gamma\), one q point may instead represent an entire mini-Brillouin-zone
cell. If the transition dipole is approximately constant across that cell,

$$
\left\langle
v(\mathbf q)|\mathbf q\cdot\mathbf D|^2
\right\rangle_{\mathrm{cell}}
=D_a^*\,\mathsf M_{ab}\,D_b,
$$

with

$$
\mathsf M_{ab}
=\left\langle v(\mathbf q)q_aq_b\right\rangle_{\mathrm{cell}}.
$$

The six independent entries of the real symmetric tensor carry the complete
angular and radial average. A scalar \(\langle v\rangle\) is insufficient:
it discards the angular second moment and averages the wrong radial function.

In three dimensions,

$$
\operatorname{tr}\mathsf M
=\left\langle v(\mathbf q)q^2\right\rangle
=\frac{8\pi}{\Omega},
$$

which is an exact sampler diagnostic. Under slab truncation,
\(\mathsf M_{zz}=0\) and the trace vanishes linearly with the cell size.

## 3. LORRAX representation

For finite q, LORRAX evaluates exchange in the ISDF centroid basis. The
transition coefficients and q-dependent interpolation vectors already carry
the full pair amplitude; all nonanalytic behavior enters through the Coulomb
kernel.

The optional mini-BZ tensor correction lives in transition space because the
linear coefficient of the pair amplitude is exactly the dipole:

$$
\left.
\frac{\partial M_t(\mathbf q,0)}{\partial q_a}
\right|_{\mathbf q=0}
=-id_{a,t}.
$$

The correction is Hermitian and positive semidefinite because
\(\mathsf M\) is a positive weighted second moment. It is appropriate only
when the \(\Gamma\) sample stands for its cell. It is not a more accurate
pointwise evaluation along a finite-q band path.

The dipole includes the nonlocal-pseudopotential commutator and carries its
velocity-sign provenance in `dipole.h5`. The tensor follows the Cartesian
convention in [The S-tensor convention](s-tensor-convention.md).
