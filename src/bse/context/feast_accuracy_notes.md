# FEAST Accuracy Notes

## Reference: Full diagonalization (N=144, n_val=4, n_cond=4, 3×3×1 k-grid)

Exact BSE eigenvalues (eV), first 8:
```
1.851722, 1.851744, 1.915450, 1.915455, 1.962725, 1.962746, 2.067898, 2.067945
```
Three degenerate pairs. Zero non-interacting transitions below 1.91 eV — these
are purely excitonic states.

## Observation 1: GMRES tolerance doesn't matter

With window [0, 1.91], n_ritz=8, n_quad=8, gamma=0.4:

| gmres_tol | Ritz 1 | Ritz 2 | Ritz 3 | Ritz 4 | Ritz 5 | GMRES iters/solve |
|-----------|--------|--------|--------|--------|--------|-------------------|
| 1e-2      | 1.8553 | 1.8610 | 1.9275 | 1.9402 | 2.0137 | 2-3               |
| 1e-4      | 1.8553 | 1.8610 | 1.9275 | 1.9402 | 2.0137 | 3-5               |
| 1e-6      | 1.8553 | 1.8610 | 1.9275 | 1.9402 | 2.0137 | 4-7               |

Max change from tol=1e-2 to 1e-6: **0.02 meV**. The diagonal preconditioner is
so effective that even 2-3 GMRES iterations give a near-exact solve.

## Observation 2: Errors come from the FEAST filter, not GMRES

Comparison against exact eigenvalues:

| Window    | Ritz (eV) | Exact (eV) | Error (meV) | f(E_exact) |
|-----------|-----------|------------|-------------|------------|
| [0, 1.91] | 1.8553   | 1.8517     | +3.6        | 0.990      |
| [0, 1.91] | 1.8610   | 1.8517     | +9.3        | 0.990      |
| [0, 1.91] | 1.9275   | 1.9155     | +12.1       | 0.878*     |
| [0, 1.91] | 1.9402   | 1.9155     | +24.7       | 0.878*     |
| [0, 1.91] | 2.0137   | 1.9627     | +51.0       | 0.240*     |

*f(E) = filter response at exact eigenvalue for n_quad=8, gamma=0.4

Eigenvalues near the window boundary (where f(E) < 1) get progressively worse
Ritz approximations. The filter at 1.963 eV is only 0.24 for window [0, 1.91].

## Observation 3: Lanczos ghost eigenvalues

The 60-step Lanczos on N=144 produced ghost eigenvalues from loss of
orthogonality. The reported [1.838, 1.838, 1.853, 1.902, 1.902] were NOT the
exact eigenvalues. Lanczos is reliable for E_max bounds (well-separated extremal
eigenvalue) but not for counting/resolving near-degenerate low-energy states.

## Recommended parameter sweep

Run `bse_feast` with all combinations below and record output (which now
includes avg GMRES iterations per window).

### Axes to sweep

**GMRES (low priority — already shown to be cheap):**
- `--gmres-tol`: 1e-2, 1e-4
- `--gmres-max-iter`: 10 (generous ceiling; actual convergence is 2-7)

**Contour quadrature (high priority — controls filter quality):**
- `--n-quad`: 4, 8, 16, 32
- `--gamma`: 0.2, 0.4, 0.8 (ellipse aspect ratio; smaller = tighter imaginary
  extent = sharper filter but slower GMRES convergence)

**Subspace size:**
- `--feast-ritz-count`: 4, 8, 12

**Window placement:**
- `--window1 0 2.0 --window2 2.0 auto` (eigenvalues well inside)
- `--window1 0 1.91 --window2 1.91 auto` (eigenvalue at boundary)

### Template command

```bash
uv run python -m bse_isdf.bse_feast \
  -i cohsex_prod.in --n-val 4 --n-cond 4 --n-lanczos 30 \
  --feast-ritz \
  --feast-ritz-count {N_RITZ} \
  --n-quad {N_QUAD} \
  --gamma {GAMMA} \
  --gmres-max-iter 10 \
  --gmres-tol {GMRES_TOL} \
  --window1 {W1_A} {W1_B} \
  --window2 {W2_A} auto
```

### Key output to record per run

From stdout:
- **Ritz evals (eV)** — the eigenvalue approximations
- **GMRES avg iters** — average GMRES iterations across all solves in a window
- **total matvecs** — total BSE matvec applications (cost metric)
- **S eigenvalues** — overlap spectrum (physical/total count)
- **Wall time** — from the timing report

### What to look for

1. **n_quad convergence**: At what n_quad do Ritz values stop changing? (expect
   diminishing returns past ~16)
2. **gamma tradeoff**: Smaller gamma sharpens filter but increases GMRES cost
   (shifts closer to real axis). Is there a sweet spot?
3. **Cost scaling**: total_matvecs = n_ritz × n_quad × (avg_gmres_iters + 1).
   Compare accuracy vs cost across settings.
4. **n_ritz sufficiency**: With n_ritz < n_states_in_window, Ritz values are
   variational approximations (biased high). With n_ritz ≥ n_states + 2,
   expect near-exact results for well-filtered eigenvalues.
