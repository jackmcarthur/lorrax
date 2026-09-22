# Compact noncrossing response quadrature

`minimax.response_laplace_rule` is the service door; `laplace_ritz.py` owns
placement, projection and the continuum residual certificate. Energies use one
unit (Ry in GW), times its inverse, and derivatives are with respect to `s=z²`.
The service accepts a positive transition interval `[L,H]` with
`L > max|Re z|`, plus a physical reference gap `r`. The
[service contract](../services/minimax.md#response-bank-rule-sessions) owns
padding and reuse; the [bank](../architecture/shared_pole_model.md) owns state
partitioning and response orientation.

## Geometry-only positive times

For degree N, set `W=max|Re z|`, `a=(L-W)/2`, `b=(H+W)/2`, `k'=a/b`, and
`k=sqrt(1-k'²)`. Prescribe auxiliary rates

\[
\alpha_j=b\,\mathrm{dn}\!\left((2j+1)K(k)/(2N),k\right).
\]

A complementary-modulus product evaluates these without subtracting nearly
identical numbers. In the orthonormal exponential basis, differentiation has
`A_ii=-alpha_i`, `A_ij=-2 sqrt(alpha_i alpha_j)` for `i<j`, and zero below the
diagonal. Solve `Aᵀ M + M A = -I`; the eigenvalues of the symmetric time-moment
matrix M are the positive times. The implementation scales energies before
this small solve. It never forms the ill-conditioned Cauchy Gram matrix.
When all rates coincide, these are Gauss–Laguerre times divided by `2 alpha`.

These are Ritz values of multiplication by time in the exponential subspace;
positivity follows from `integral t |f(t)|² / integral |f(t)|² > 0`.
The elliptic rate distribution controls the subspace's Blaschke-product
projection error, but does **not** by itself certify the final finite sum or
prove it is minimax. Degree selection uses the final residual certificate,
not a universal20-node promise. No nonlinear node optimization or node bank
is used.

## Values and Hermite data on the same times

At deterministic Chebyshev points in `log(d-W)`, form `B_lj=exp(-(d_l-L)t_j)`.
A column-scaled real SVD with two right-hand sides projects each exact target
with relative weighting. The targets are

\[
K_e=\frac{d}{d^2-z^2},\quad \partial_s K_e=\frac{d}{(d^2-z^2)^2},\qquad
K=\frac1{d^2-z^2},\quad \partial_sK=\frac1{(d^2-z^2)^2}.
\]

The last two are needed for ordered response: `K_o=z K` and
`partial_s K_o=K/(2z)+z partial_sK`. Values and derivatives are independently
projected exact rational targets; the derivative of a frequency-dependent
weighted least-squares solve is not used. All rows share exactly the same
times. Returned coefficients include `exp(-(r-L)t)`, and the existing Green
factors supply `exp(-(d-r)t)`. Actual physical transitions obey `d>=r`, so no
production operand grows exponentially. Complex coefficients are allowed;
no conjugate-time union or new Green/FFT kernel is required.

## Continuum certificate of the returned arrays

On short geometric panels in `d-W`, expand each exponential to degree20 in
`y=(d-centre)/halfwidth`. Its uniform remainder is bounded by
`exp(-(left-L)t) (halfwidth*t)^21/21!`. Multiply the finite polynomial by the
exact denominator (`d²-z²`, its square, or a primitive linear denominator or
its square), subtract the exact numerator, and sum the absolute Chebyshev
coefficients. Add the exponential remainder times the denominator bound and
a conventional floating-point arithmetic allowance. The squared denominators
are degree4 and certify the derivative targets with the same construction.

The certificate covers all real d in the padded interval at every supplied z;
it is neither a fit-grid residual nor a claim for unsampled frequencies.
It reconstructs K and its derivative from the **rounded returned odd rows**,
including their `1/(2z)` term, and reverses the reference shift in extended
precision before checking. It also checks the primitive orientation sums and
derivatives directly. Relative error in the odd derivative itself is undefined
at its zeros; the certificate instead bounds K and its derivative separately.
There is no omitted infinite-time tail: the finite sum is compared to the full
rational target. Arithmetic guards assume ordinary libm accuracy, not an
interval-arithmetic implementation of the exponential function.

## Production response bank

The bank now uses one occupation-weighted frequency rule over the full active
transition interval, eliminating separate remote-cell Green sweeps.
The [shared-pole architecture](../architecture/shared_pole_model.md) owns that
stream and its error convention. This page documents the standalone
noncrossing Laplace service and its stronger continuum certificate.
