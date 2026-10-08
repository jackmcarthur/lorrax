# Small contour references

`gw.contour_reference` owns projected scalar contour laws, physical masks,
ordered sheets and normalization-free residue consumption. A caller supplies
projected minus-Wc values and raw d/ds slopes with s=z_Ry²; the owner multiplies
by2z exactly once before the occupied-retarded partner dagger.

The default real residue uses cubic Hermite interpolation. The opt-in
`prepare_real_residue_hermite7_grid` diagnostic uses four nearest available
knots and a degree7 Hermite polynomial, clipping its stencil at actual edges.
It rejects missing support and fewer than four coarse knots. Existing anchor,
imaginary tail and default consumers retain their numerical laws.

`real_residue_diagonal_interval_terms` exposes scalar-diagonal contributions
attributed to each physical crossing interval, for bounded cache diagnostics.
Its interval sums must reproduce the ordinary residue difference independently.
These are interpolation contrasts, not an error bound: H7 can overshoot even
when its sampled response is causal. No production driver uses this prototype.
