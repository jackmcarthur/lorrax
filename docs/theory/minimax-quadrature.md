# Minimax quadrature for static and plasmon-pole GW

This chapter describes the scalar quadratures used by static screening and the
GN/HL plasmon-pole self-energy. The current MPA body uses fitted complex poles,
a positive causal crossing rule, and complex-sector rules instead; see
[Multipole frequency integration](THEORY_mpa_implementation.md).

The physics owner supplies an energy interval and a target function. The
`minimax` service supplies immutable nodes, weights, achieved error,
amplification, and provenance. Production lookup does not silently solve a
new table; a missing certified asset refuses unless the explicit uncertified
escape hatch is enabled.

## 1. Why reciprocal kernels are separated

After a spectral window is referenced to one edge, its denominator is either
sign definite or crosses zero.

For \(x\in[x_{\min},x_{\max}]\) with \(x_{\min}>0\),

$$
\frac{1}{x}
=\int_0^\infty e^{-xt}\,dt
\approx\sum_{\ell=1}^N w_\ell e^{-x t_\ell}.
$$

The exponential basis converges rapidly over a positive interval. At fixed
accuracy its rank grows approximately as

$$
N=\mathcal O\!\left(
\log\frac{x_{\max}}{x_{\min}}\,
\log\frac1\epsilon
\right).
$$

If \(x\) changes sign, one decaying Laplace contour cannot cover the complete
interval. A crossing rule instead represents an odd, regularized reciprocal
by a sine sum,

$$
g_\xi(x)
\approx\sum_{\ell=1}^N a_\ell\sin(\tau_\ell x),
\qquad
g_\xi(-x)=-g_\xi(x),
\qquad
g_\xi(0)=0.
$$

The scale \(\xi\) is part of the observable regularization, not a numerical
way to hide a singular denominator. Crossing rank is linear in the scaled
bandwidth \(A=E_{\mathrm{bw}}/\xi\) at fixed error for the learned sine
family. The older HGL target remains a supported GN/PPM asset; it is not the
MPA crossing construction.

## 2. Static and imaginary-axis screening

For an insulating transition interval
\([\Delta_{\min},\Delta_{\max}]\), static screening needs

$$
\frac1\Delta
\approx\sum_\ell w_\ell e^{-\Delta t_\ell}.
$$

The physical planner rescales a dimensionless certified rule on
\([1,R]\), \(R=\Delta_{\max}/\Delta_{\min}\):

$$
t_\ell^{\mathrm{phys}}=\frac{t_\ell}{\Delta_{\min}},
\qquad
w_\ell^{\mathrm{phys}}=\frac{w_\ell}{\Delta_{\min}}.
$$

The GN probe at \(z=i\omega_p\) uses

$$
\frac{\Delta}{\Delta^2+\omega_p^2}
=\int_0^\infty e^{-\Delta t}\cos(\omega_p t)\,dt.
$$

It has its own certified target because fitting \(1/\Delta\) does not
automatically certify the cosine-weighted kernel. A shared-node augmentation
may reuse static nodes, but only after the combined rule passes the
imaginary-axis residual gate.

On a measured-broken-TR deck the probe needs the sine-weighted kernel as
well, \(\omega_p/(\Delta^2+\omega_p^2)=\int_0^\infty e^{-\Delta t}\sin(\omega_p t)\,dt\),
because the two particle-hole orientations then carry the complex weights
\(-1/(\Delta\pm i\omega_p)\) separately. The odd kernel is represented on the
served even nodes plus the fewest greedily added nodes (weights-only fits,
the even weights zero on the extras, so the even accumulation is the served
rule unchanged) and gated at the even rule's error; measured 2026-09-01 the
even nodes alone stall at \(10^{-3}\)–\(10^{-5}\) and one to five extras
reach \(10^{-6}\). Owner: `minimax_screening.solve_laplace_minimax_imag_interval(with_odd_kernel=True)`;
physics in [`DERIVATION_gnppm_nonhermitian.md`](../dev/notes/DERIVATION_gnppm_nonhermitian.md).
This quadrature supplies the ordered probe only. The fit owner forms the
Hermitian `B` and odd Hermitian `D`, and the Sigma owner selects `B+D` for
empty/conduction branches and `B-D` for occupied/valence branches; the
quadrature service does not choose a residue.

## 3. Real-frequency PPM windows

A PPM self-energy denominator is schematically

$$
d(\omega)=\omega-E_A-\Omega.
$$

For each causal branch, the planner partitions the Cartesian
\((E_A,\Omega)\) domain so that every cell is either:

- **noncrossing**, where \(d\) has a fixed sign and a rescaled exponential
  rule is valid; or
- **crossing**, where the requested \(\omega\) interval overlaps the cell and
  the regularized sine rule is required.

The rectangular windows are not cosmetic. They preserve separability:
one band-restricted Green function and one pole-restricted screened
interaction can be formed independently. A selector depending on each
\((E_A,\Omega)\) pair would recreate the pairwise cost.

For a sine atom,

$$
\sin(\tau d)
=\frac{e^{i\tau d}-e^{-i\tau d}}{2i}.
$$

The two exponentials become forward and backward real-time propagators. Their
completion must be performed before a band projection unless the relevant
matrix adjoint identity has been proved.

## 4. Error and stability

Every rule is evaluated in the norm associated with its target. Two numbers
are required:

$$
\epsilon_{\max}
=\max_{x\in\mathcal D}|f(x)-Q_N(x)|,
\qquad
\kappa
=\sup_{x\in\mathcal D}
\frac{\sum_\ell |w_\ell\phi_\ell(x)|}{|f(x)|}.
$$

A small residual with large \(\kappa\) can amplify roundoff, ISDF error, and
small differences between distributed reductions. Production therefore
selects by certified error and enforces the recorded amplification cap.

Tables are identified by family, target, dimensionless range, requested error,
node cap, content hash, generator provenance, achieved error, and
amplification. The range is rounded only in the conservative direction:
a served table must cover the complete requested interval.

Runtime nonlinear fitting is intentionally not the default. The historical
VarPro/Lawson solvers can converge to different local supports on different
numerical stacks even when their residuals are similar. Certified artifacts
make the mathematical object reproducible.

## 5. Execution form

At one node, the driver forms windowed Green functions such as

$$
G_A(\mathbf k,t_\ell)
=\sum_{n\in A}
\psi_{n\mathbf k}\psi^\dagger_{n\mathbf k}
e^{-i(\epsilon_{n\mathbf k}-E_{\mathrm{ref}})t_\ell}.
$$

The lattice FFT, elementwise product, and band projection are shared by all
frequency models. Quadrature weights and reference phases are scalar
coefficients applied around that common spatial kernel.

Node count, not the number of requested output frequencies, controls the
expensive contractions. Output-frequency count controls the comparatively
cheap coefficient fold and storage. Window count matters because each
physically distinct band/pole selector requires its own spatial sweep even
when two windows reuse the same scalar node set.

## 6. Ownership and tuning

`services/minimax` owns target definitions, catalog lookup, provenance,
certification, and offline solvers. `gw.minimax_screening` owns physical
intervals, energy references, and rescaling. `gw.ppm_sigma` owns PPM causal
branches and window selectors. The shared Green-function and convolution
kernels own no quadrature policy.

Exact tolerances and node caps are documented in the
[input reference](../input_reference.md). When a wider band interval exceeds
a shipped table, generate and certify a wider asset or refuse; do not clip the
interval. When a denominator approaches zero, route it to the crossing family
or change the declared physical regularization; do not lower-bound it
silently.

## References

- Kim, Martyna, and Ismail-Beigi, *Phys. Rev. B* **101**, 035139 (2020).
- Hackbusch, *Hierarchical Matrices: Algorithms and Analysis* (2015).
- Beylkin and Monzón, *Applied and Computational Harmonic Analysis* **28**,
  131 (2010).
