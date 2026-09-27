# What node count does a Sigma box certificate require?

This is an evaluation-only necessary bound for the exponential ansatz in
[the dynamic Sigma quadrature](sigma-quadrature-problem.md). It is not a
node constructor or an additional production acceptance gate.

Fix one box `[a,b] + i[c,d]`, with `c > 0`. Let a rule have `N` terms,

    Q(z) = sum_l w_l exp(i t_l z).

The times and weights may be arbitrary complex numbers. This argument does
not assume a contour, a uniform time grid, positive weights, a construction
method, or a bound on cancellation.

Choose any `m` real points `u_j` in `[a/2,b/2]`. Define

    C_jk = 1 / (u_j + u_k + i c),
    A_jk = Q(u_j + u_k + i c).

Every sampled denominator belongs to the bottom box edge. Each exponential
term is an outer product: `A = sum_l w_l v_l v_l^T`, where
`v_l,j = exp(i t_l (u_j + i c/2))`. Consequently `rank(A) <= N`.
The transpose is intentional; a complex conjugate is not needed for rank.

For a peak-relative certificate `c |Q(z)-1/z| <= epsilon`,

    ||A-C||_F <= m epsilon / c = delta.

For a pointwise relative certificate `|z| |Q(z)-1/z| <= epsilon`,

    ||A-C||_F <= epsilon ||C||_F = delta.

If `s_1 >= ... >= s_m` are the singular values of `C`, Eckart--Young gives

    sqrt(sum_{j>N} s_j^2) <= delta.

The first index satisfying this inequality is a necessary node count.
This is a lower bound, not a rule and not a sufficient certificate. It
extends the old real-uniform-time floor to arbitrary exponential sums on
these sampled boxes. No node optimization occurs in its evaluation.

A numerical evaluation should state its sampling grids and SVD error
control. The QCOST evaluation uses two predetermined uniform grids and
enlarges `delta` by `100 m machine_epsilon ||C||_F`. That allowance and
grid agreement are diagnostics, not an interval-arithmetic certificate of
the computed singular values. The algebraic theorem is exact; those
floating-point evaluations retain this qualification. For an asymmetric
crossing box, a second bound on its contained symmetric core may be
stronger; take the maximum of the full-box and core bounds.

For fixed windows and separate exponential sums, summing their bounds gives
a necessary number of window/time pairs. Shared times do not evade it,
because each window still has its own rank requirement. The largest single
bound is a lower bound on global distinct times; the sum is not.

This does not rule out changed product-window geometry, a relaxed error
currency, a different regularization, or a different representation of the
operator. It does not apply to a discrete-only target unless every sampled
denominator is in that target. The computation uses the current continuous
box certificate, including its padded support, exactly as requested by the
production acceptance rule.

A small sampled rank bound does not construct a stable quadrature. It
removes no requirement on factor growth, cancellation, deterministic rule
construction, or the independent continuum certificate. An ansatz with
extra polynomial factors per node may have a different separated rank and
must price all of its additional contractions.
