# Finite-occupation RPA screening

This note fixes the algebra and implementation boundary for metallic screening.
It does not declare the full GW pipeline metal-ready.

## Retarded response and the low-scaling factorization

For a=(n,k), b=(m,k-q), Delta_ab=epsilon_b-epsilon_a, and the
density-vertex outer product X_ab(q), the independent-particle response is

```text
chi0_q(z) = C sum_ab[q] (f_a-f_b) X_ab(q) / (z-Delta_ab),  Im z > 0.
```

This is the finite-occupation Adler-Wiser expression.  The useful identity is

```text
f_a-f_b = f_a(1-f_b) - (1-f_a)f_b.
```

Define one positive-time product

```text
A_q(t) = sum_ab[q] f_a(1-f_b) exp(-i Delta_ab t) X_ab(q).
```

Hermiticity of the density vertex, with an explicit relabeling of the full
k grid, gives

```text
B_q(t) = A_{-q}(-t)^T
chi0_q(z) = -i integral_0^infinity dt exp(i z t)
            [A_q(t) - A_{-q}(-t)^T].
```

This is not a time-reversal argument.  The q and -q orientation remains
explicit, so the identity is valid for a fully relativistic ferromagnet.

In the flat-grid FFT convention used by Lorrax,

```text
FFT_R[conj(A_R(t))](q) = conj(A_{-q}(t)) = A_{-q}(-t)^T.
```

Therefore one time node needs two single-band Green sums, not four:
one weighted by f and one by 1-f.  They meet only after both band axes have
been summed.  No band-pair by centroid or G object is formed.

## Implemented seam

`gw.greens_function_kernel.build_G_tau` accepts an optional linear
`band_weight`.  The phase/window helper is unchanged.  The weight is applied
after the phase, and is never clipped or square-rooted; this matters because
Methfessel-Paxton occupations can be negative or greater than one.

`gw.w_isdf.compute_chi0_contour_fractional` implements the positive-time
formula above.  It:

- takes occupations explicitly (or uses the wavefunction bundle table);
- derives the smallest exact contiguous support of f and 1-f;
- includes every partially occupied band in both supports;
- drops only weights stored as exactly zero or exactly one, with no tolerance;
- shares one positive-time scan across all requested complex frequencies;
- applies the q/-q partner through the FFT/conjugation identity before the
  final full-grid response is symmetry-reduced.

The old disjoint valence/conduction path is untouched.

## Frequency integration

For an MPA line with z=omega+i*varpi and varpi>0, the existing positive
real-time composite rule can project the finite-occupation integrand directly.
Its frequency bound must cover

```text
max |omega| + max |epsilon_b-epsilon_a|
```

over the two occupation supports.  The dynamic line should call the
fractional contour kernel with the rule's positive nodes and weights; it must
not apply the old symmetric positive-gap kernel a second time.

The static point is a different problem.  At z=0,

```text
(f(E)-f(E')) / (E'-E) -> -df/dE.
```

A nearly-zero imaginary frequency is not an exact replacement and makes the
real-time node count diverge as the damping vanishes.  The q=0 static route
therefore evaluates the divided difference directly.  It streams fixed-size
ordered band-pair tiles and assigns each `(mu_x,nu_y)` output tile to one
`Px*Py` rank; spinor components are summed in the density vertex and no global
pair-density buffer is formed.  The diagonal uses periodic-tetrahedron
`-df/dE` weights and the off-diagonal uses the carried MP1 occupations.  This
is the exact finite-band fallback, not the finite-eta contour in disguise.

The remaining performance optimization is a certified separable static
target.  A promising form follows from a rational approximation

```text
f(E) ~= sum_j a_j/(E-z_j),

[f(E)-f(E')]/(E-E') ~= -sum_j a_j /
                       [(E-z_j)(E'-z_j)].
```

That is two single-band resolvent sums per pole and naturally retains the
diagonal -df/dE limit.  The service must be parameterized by the actual
smearing family, width, energy interval, and an absolute error bound before it
can replace the tiled fallback.  MP1 cannot borrow a Fermi-Dirac/Matsubara
certificate.

## Relation to W-av and the QSGW head

The finite-q body and the long-wavelength head solve different limits:

1. The fractional contour kernel supplies the full finite-q chi0 body at every
   dynamic MPA sample.
2. The q->0 head uses the occupation-aware interband S tensor plus the
   tetrahedron Fermi-surface Drude tensor.
3. The left and right wings are built directly as `Y[a,mu]` and `Z[nu,b]`
   from the velocity matrix and the two centroid-sharded wavefunction copies.
   Band-pair work is evenly tiled over `Px*Py`; frequency blocking bounds the
   temporary without materializing a `(k,band,band,mu)` tensor.
4. At every complex sample, the already-screened ISDF body forms
   `S_eff[a,b] = S_direct[a,b] + Y[a,mu] W_body[mu,nu] Z[nu,b]/cell_volume`.
   No reciprocal-G body or wing is constructed.
5. At the exact static sample, diagonal Fermi-surface density wings are folded
   through the divided-difference body to form an effective scalar `f00`;
   `kappa_eff^2=-8*pi*f00` then enters the 3D Thomas-Fermi mini-BZ average.
   This is the static order of limits, not the zero-frequency Drude value.
6. The resulting Wc head samples are fit on the identical MPA grid and then
   mini-BZ averaged.

Finite occupations must subsequently reach Sigma exchange/correlation and
the self-consistent Green functions before a metal calculation can be called
complete.  Until those consumers land, the input-level metal capability gate
must continue to refuse.

## References

- Sesti et al., *Efficient GW calculations for metals from an accurate ab
  initio polarizability*, arXiv:2508.06930 (2025).
  https://arxiv.org/abs/2508.06930
- Kim, Martyna, and Ismail-Beigi, *Complex time, shredded propagator method
  for large-scale GW calculations*, Phys. Rev. B 101, 035139 (2020).
  https://doi.org/10.1103/PhysRevB.101.035139
- Rojas, Godby, and Needs, *Space-Time Method for Ab Initio Calculations of
  Self-Energies and Dielectric Response Functions of Solids*,
  Phys. Rev. Lett. 74, 1827 (1995).
  https://doi.org/10.1103/PhysRevLett.74.1827
