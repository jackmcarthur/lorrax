# Appendix F — Symbols and index conventions

The core symbol table is §1.4. Additional conventions used throughout the code
and this manual:

| Convention | Meaning |
|---|---|
| $\mu, \nu$ (or $\mu_c, \nu_c$) | ISDF centroid indices; $\mu_L$ = Lorentz channel 0–3 (Ch. 8) |
| $X, Y$ subscripts | sharding over the 2D mesh axes `('x','y')` (§13.2) |
| flat-k | k- and q-grids flattened to a single batch axis; never a sharding axis |
| $\bar k$, $i(q)$ | irreducible-wedge representative and its index map (App. A) |
| $L$, $S$ spinor blocks | large/small bispinor components (§8.1) |
| $\tau_l, w_l$ | quadrature nodes and weights; $\xi$ = broadening; $R$, $A$ = range ratios (§7.2) |
| occupied energies | $\varepsilon_v < 0 < \varepsilon_c$, measured from `fermi_reference` |
| Ry internally, eV in outputs | §1.4 |
