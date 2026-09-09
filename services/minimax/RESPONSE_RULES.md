# Response-bank scalar rules

The public doors are `minimax.response_bank_rule(z_ry, delta_max_ry, *, rel_tol=1e-8)` and `minimax.response_laplace_rule(delta_lo_ry, delta_hi_ry, z_ry, *, rel_tol=1e-8)`. Frequencies and transition intervals are in Ry; times are in inverse Ry; derivatives are with respect to s=z_Ry². Both return plain dictionaries. These are additive bank services: existing Sigma quadrature calls are unchanged.

The Hermite door ports the Run212 panel/tail construction and Run307 early-panel refinement, with every supplied imaginary decay included in order selection. Its positive t/h rule returns h exp(i z t) and that row multiplied by i t/(2z). The bank supplies the retarded paired correlation. The error currency is peak-scaled absolute error (eta times value, eta³ times derivative, eta=min Im z), as in Run212; this does not certify relative W or Sigma accuracy. The certificate bounds the full signed transition interval and both exponential branches. The Bernstein ellipse bound includes the ellipse centre in the bound on |t|. Tail and panel budgets include the derivative. Orders and early-panel edges are functions of the current points and interval, with explicit resource refusals.

The remote door ports Run183's nonnegative NNLS fit of delta/(delta²+eta²)^(n+1). It rescales delta by its positive lower bound, chooses the Taylor order using all supplied s values, and refines the positive time dictionary until the certificate passes. NNLS fits and its continuum certificate are separate. For the relative fitted row R(delta), each interval uses

    sup |R-1| <= max(endpoint errors) + width²/8 sup |R''|,
    sup |R''| <= |R''(midpoint)| + width/2 sup |R'''|.

R is a sum of positive terms g=w exp(-t delta)(delta²+a²)^(n+1)/delta. Log derivatives of g give g'''=g(l1³+3 l1 l2+l3). Conservative absolute log-derivative bounds and the midpoint exponential envelope bound each term throughout the interval. A floating summation allowance is included. The certificate is thus not a claim based only on the NNLS training grid. If the interval bound fails, the grid refines; if fit or dictionary limits fail, the door refuses.

With q=(s+eta²)/(delta²+eta²), the value Taylor remainder relative to its exact resolvent is bounded by rho^(N+1). The derivative remainder is bounded by rho^N ((N+1)+N rho), where rho bounds |q| over the entire interval. Positive-row fit errors are amplified by at most (1+rho)/(1-rho) for values and its square for derivatives. The returned per-point combined bounds include both sources. A nonconvergent Taylor domain refuses and must be repartitioned by the bank owner.

Remote projections approximate delta/(delta²-z²) and delta/(delta²-z²)² respectively. The bank multiplies by -exp(-gap*t); the paired-stream factor two stays in the physics owner. Returned coefficient rows are nonnegative, but the final complex Taylor projections need not be. No Na constants, run-path imports, band masks, occupations or response arrays enter either rule.

`tests/test_response_rules.py` checks independent analytic kernels, the missing-1/(2z) derivative red twin, positivity and refusals. Campaign Run317 additionally records every per-point certificate for the published IINPUTS Na/Si production/relaxed coordinate sets at eta=0.25/0.10 eV. The transition intervals in that scalar verification are planted, not a census of actual band gaps. Production must supply and certify its current interval.
