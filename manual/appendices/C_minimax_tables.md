# Appendix C — Shipped minimax tables

`src/common/minimax_assets/` holds precomputed quadratures consumed by the
selection rule of §7.2: smallest tabulated range ≥ requested, loosest tabulated
error ≤ requested, node count within the cap; anything uncovered falls back to
the exact solvers. The shipped sweep: noncrossing ranges $R$ from $10$ to $10^5$
at three grids per decade, crossing bandwidths $A = 20$ and $40$, error bounds
$10^{-6}$ and $2\times10^{-7}$.

Error conventions. Noncrossing tables are fitted on the scaled interval $[1, R]$
under absolute $L_\infty$ error, $\max_y |1/y - \Sigma_l w_l e^{-t_l y}| \le
\varepsilon$, and rescaled to the physical interval at use (the bound scales as
$1/x_{\min}$); crossing tables on $[0, A]$ under the analogous absolute
convention. Neither is a relative-at-endpoint criterion.

The exact solvers (Remez/variable-projection with Lawson reweighting for
noncrossing; an LP-based construction for crossing) cost seconds to ~100 s on the
host, which is why the tables exist: a table lookup is sub-millisecond, and the
crossing solver used to dominate GN-PPM startup. `regenerate_minimax_tables =
true` bypasses the tables for solver development; the empirical error laws of
§7.2 were fitted across the shipped sweep and extrapolate reliably within it.
