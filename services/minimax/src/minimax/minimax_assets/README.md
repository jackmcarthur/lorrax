# Shipped Minimax Quadratures

This directory contains precomputed minimax quadrature tables for runtime reuse by
`gw.minimax_screening`.

## Selection rule

At runtime, the lookup code chooses:

1. the smallest tabulated range that is greater than or equal to the requested range
2. the loosest tabulated error bound that is still less than or equal to the requested target
3. a table whose node count does not exceed the caller's `max_nodes`

If no shipped table matches, runtime falls back to the exact solver.

## Error conventions

### Noncrossing

Tables are generated on the scaled interval `[1, R]` with the absolute L-infinity error
convention:

`max_y | 1/y - approx(y) | <= error_bound`, for `y in [1, R]`.

When used for a physical interval `[x_min, x_max]`, runtime rescales the nodes and weights.
The physical absolute error then scales as `error_bound / x_min`.

This is not a relative-at-endpoint criterion.

### Crossing

Crossing tables are generated for the absolute L-infinity error on the target function
itself:

`max_u | G(u) - approx(u) | <= error_bound`, for `u in [0, A_dim]`.

## Sweep values in this bundle

- Error bounds: 1.0e-06, 2.0e-07, 6.9e-08
- Crossing target kind: `hgl`
- Crossing `eps_q`: 1.000e-03
- Crossing `A_dim` values: 20, 40
- Noncrossing `R` values: 10, 21.5443, 46.4159, 92.3619, 100, 215.443, 464.159, 1000, 2154.43, 4641.59, 10000, 21544.3, 46415.9, 100000

The machine-readable descriptor is `catalog.json`.
