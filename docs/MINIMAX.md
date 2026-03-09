# Laplace Minimax

Minimax quadrature nodes and weights for Laplace-transformed GW self-energy frequency integration. Uses `uv` for dependency management — run everything via `uv run python <script>`.

## What this solves

GW self-energy evaluation requires approximating energy denominators $1/x$ to enable $O(N^3)$ separable time-domain propagator products. There are two regimes depending on whether the energy window crosses the Fermi level:

**Non-crossing windows** ($x > 0$, definite sign): $1/x \approx \sum_\ell w_\ell e^{-t_\ell x}$ on $[1, R]$ where $R = E_\mathrm{bw}/E_\mathrm{gap}$. This is well-understood (Braess & Hackbusch 2005): minimax error scales as $\epsilon \approx C\exp(-\pi^2 N/\ln(\beta R))$ with $\beta \approx 4$, giving $O(\ln R)$ nodes. The best existing tabulated nodes are from Takatsuka, Ten-no & Hackbusch (JCP 2008) and Hackbusch (2019 book, table 11.1), but our solver matches or improves on these.

**Crossing windows** ($x$ changes sign): $1/x$ has a singularity at $x=0$ that must be regularized. The standard approach (HGL quadrature from Kim, Martyna & Ismail-Beigi, PRB 2020) requires $O(A^2)$ nodes where $A = E_\mathrm{bw}/\xi$. Our learned-regularization approach fits $1/x$ directly with sine sums on $[u_\min, A]$ and lets the optimizer discover the regularization shape, achieving $O(A)$ node scaling — the same as noncrossing. The user specifies a target effective broadening $\xi_\mathrm{eff}$ in eV; the solver binary-searches the dimensionless bandwidth $A$ until the first-moment missing area of the sine-sum fit matches that of a Lorentzian with width $\xi_\mathrm{eff}$. This means the achieved broadening is defined by equivalence to a Lorentzian, not by the exclusion window directly — the effective width is typically 3-5x narrower than the excluded region.

## Solver methods

**Noncrossing**: Remez exchange with damped Newton equioscillation solver, VarPro warm-starts, and continuation from $R=2$ doubling each step. Tried VarPro+Lawson IRLS, LP backward elimination, and L-inf RELAX; the Remez exchange gave the best results.

**Crossing**: Periodic initialization with VarPro-LM (variable projection Levenberg-Marquardt) for nonlinear frequency optimization, followed by least-squares weight solve. Tried LP backward elimination, L-inf RELAX greedy placement, multi-start continuation, and peak-based initialization; VarPro-LM with periodic init was cleanest and most robust.

## Error scaling

**Noncrossing** (calibrated on $\epsilon \in [10^{-5}, 10^{-2}]$, $R^2 = 0.995$):
$$\epsilon(N, R) \approx 0.31 \cdot \exp\!\big[-N\big(\tfrac{3.55}{\ln R} + 0.68\big)\big]$$

**Crossing** (calibrated on $A = 50, 100, 200$, $R^2 = 0.996$):
$$\epsilon(N, A) \approx \exp(-0.93 - 14.25 \cdot N/A)$$

## Project layout

- `minimax_nodes.py` — **The main module.** All solvers and prediction helpers.
- `minimax.py` — Legacy VarPro+Lawson solver, used internally by `minimax_nodes.py` for warm-starts.
- `test_minimax.py` — Tests. Run with `uv run python -m pytest test_minimax.py`.
- `plots/` — Plotting scripts and generated PNGs.
- `archive/` — Previous experimental solvers and approaches.

## Integration with isdf_cohsex

The GW code (`isdf_cohsex`) needs minimax nodes for each energy window. Here is how to use `minimax_nodes.py`:

### Step 1: Classify windows

For each energy window, you have `E_bw` (bandwidth in eV) and either `E_gap` (for noncrossing) or `xi_eff_target` (desired effective broadening in eV, for crossing).

### Step 2: Predict how many nodes you need

The recommended error tolerance is `0.01 / R` for noncrossing and `0.01 / A` for crossing — this keeps the absolute error contribution small relative to the denominator values being approximated.

```python
from minimax_nodes import predict_N_noncrossing, predict_N_crossing

R = E_bw / E_gap
eps_nc = 0.01 / R  # recommended: scales with dynamic range

N_nc = predict_N_noncrossing(R=R, target_error=eps_nc)

# Crossing: need xi_eff (broadening) and E_bw
# A is not known yet, so use the estimate returned by predict_N_crossing
N_cr, A_est = predict_N_crossing(xi_eff_target=0.2, E_bw=10.0, target_error=0.01 / 70)
# or equivalently, pass your best guess of A:
# eps_cr = 0.01 / A_est
```

Typical values: noncrossing needs 6-8 nodes for $R=10^3$ (eps=$10^{-5}$), crossing needs ~40 nodes for $A \sim 70$ (eps=$1.4 \times 10^{-4}$).

### Step 3: Obtain nodes and weights

```python
from minimax_nodes import solve_noncrossing, build_crossing_quadrature

# Noncrossing: returns (tau, w, err)
# Use as: 1/x ≈ sum_l w[l] * exp(-tau[l] * x)  for x in [1, R]
tau, w, err = solve_noncrossing(N=N_nc, R=E_bw / E_gap)

# Crossing: returns (tau, w, info)
# Use as: 1/x ≈ sum_l w[l] * sin(tau[l] * x / xi_0)  for |x| > x_min
# info['xi_0'] is the physical scale factor; info['xi_eff'] is the achieved broadening
tau, w, info = build_crossing_quadrature(N=N_cr, xi_eff_target=0.2, E_bw=10.0)
xi_0 = info['xi_0']  # needed to evaluate the approximation
```

### Step 4: Evaluate the approximation

```python
from minimax_nodes import evaluate_noncrossing, evaluate_crossing

# Noncrossing: g(x) = sum w_l exp(-tau_l x)
g = evaluate_noncrossing(x, tau, w)

# Crossing: F(x) = sum w_l sin(tau_l x / xi_0)
F = evaluate_crossing(x, tau, w, xi_0)
```

### Summary of the call sequence

```
For each energy window:
  if noncrossing:
    R = E_bw / E_gap
    N = predict_N_noncrossing(R, target_error=0.01/R)
    tau, w, err = solve_noncrossing(N, R)
    # propagator: G(tau_l) = exp(-tau_l * E) for each node
  if crossing:
    N, A_est = predict_N_crossing(xi_eff, E_bw, target_error=0.01/A_est_guess)
    tau, w, info = build_crossing_quadrature(N, xi_eff, E_bw)
    # propagator: G(tau_l) = sin(tau_l * E / info['xi_0']) for each node
```

The solver runs in seconds for typical parameters (N < 60, R < 10^6) and only needs to run once per unique (N, R) or (N, xi_eff, E_bw) combination.
