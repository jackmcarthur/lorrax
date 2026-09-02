# minimax — certified quadrature lookup

`services/minimax/` is an independently installable NumPy service. Consumers
use `import minimax`; importing its submodules from LORRAX is a layering
failure. Importing the package does not import JAX or SciPy. SciPy is loaded
only by the offline/runtime solvers.

## Caller contract

| Surface | Contract |
|---|---|
| `lookup(...)` | Searches shipped certified tables only and raises a named F1–F4 refusal on an invalid or uncovered request; it never solves. |
| `serve(...)` | Calls `lookup` first, then either announces and performs an uncertified runtime solve or raises F5 when that escape hatch is disabled. |
| `catalog()` / `nearest_certified(...)` | Enumerate certified coverage and suggest nearby covered requests without solving. |
| `family_for_character(...)`, `TARGETS`, `FAMILIES` | Define the accepted target and family vocabulary as data. |
| `Quadrature`, `Provenance` | Return nodes, weights, measured error, certification state, source, and artifact identity. |
| offline solver names | Lazily expose table-generation machinery; using them does not make the result a shipped certified rule. |

`LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE` controls the escape hatch and currently
defaults to enabled. Every runtime solve is labelled
`runtime-uncertified`, announces that it is not reproducible across hosts, and
carries no certified table identity. The complete spelling and boolean grammar
are owned by the [environment-variable registry](../dev/env_vars.md).

## Catalog and selection

Catalog entries bind a family, target, range selector, error bound, node limit,
payload hash, achieved error, amplification, generator provenance, and backend.
Selectors such as `beta_selector` and `damped_line_selector` are public module
objects at the package door because their clauses are part of catalog
selection. An explicit malformed, missing, or insufficient entry refuses; it
is not treated as a silent cache miss.

## Boundary with LORRAX's odd imaginary-axis kernel

The certified service tables and their bytes are unchanged by the magnetic
GN-PPM odd-kernel path. `gw.minimax_screening` is the adapter: when
`solve_laplace_minimax_imag_interval(..., with_odd_kernel=True)` cannot obtain
the odd component from a certified complex rule, it keeps the served even
nodes, greedily adds at most `ODD_KERNEL_MAX_EXTRA_NODES`, and fits
`omega_p / (x**2 + omega_p**2)` to the same requested error. The adapter marks
those extra weights in `LaplaceMinimaxQuadrature.alpha_odd`; it refuses if the
fit misses the gate. This augmentation is a runtime LORRAX fit, not a new
certified service table.

`gw.w_isdf.compute_chi0_imag_ordered` requires `alpha_odd` and consumes the
even and odd weights on one node axis. The derivation and limiting identities
are owned by the [non-Hermitian GN-PPM memo](../dev/notes/DERIVATION_gnppm_nonhermitian.md).

## Verification

The standalone package tests live in `services/minimax/tests/`; the monorepo
layering test enforces the top-level door. Lookup tests must run without SciPy,
while solver-generation tests may require it.
