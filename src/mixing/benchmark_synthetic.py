#!/usr/bin/env python3
"""
Benchmark script reproducing Example 1 (Synthetic Problem) from
Wan & Międlar, "On the Convergence of CROP-Anderson Acceleration Method".

This creates Figure 3a: convergence comparison of Anderson, CROP, and
CROP-Anderson on a tridiagonal linear system.

Run with:
    uv run python -m mixing.benchmark_synthetic
    # or
    uv run python src/mixing/benchmark_synthetic.py
"""

from __future__ import annotations

import sys

from runtime import set_default_env
set_default_env(platform="cpu")

# Set interactive matplotlib backend if --show is in args (before importing pyplot)
if "--show" in sys.argv:
    import matplotlib
    for backend in ["TkAgg", "Qt5Agg", "GTK3Agg"]:
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mixing.acceleration import (
    anderson_acceleration,
    crop,
    crop_anderson,
    rcrop,
    rcrop_anderson,
)


# -----------------------------------------------------------------------------
# Problem construction
# -----------------------------------------------------------------------------


def make_tridiag_A(n: int = 100) -> jnp.ndarray:
    """Tridiagonal matrix with entries (1, -4, 1).

    This is a symmetric negative-definite matrix (eigenvalues in [-6, -2]).
    """
    main = -4.0 * jnp.ones((n,))
    off = 1.0 * jnp.ones((n - 1,))
    A = jnp.diag(main) + jnp.diag(off, 1) + jnp.diag(off, -1)
    return A.astype(jnp.complex128)


def make_sevendiag_A(n: int = 100) -> jnp.ndarray:
    """Seven-diagonal matrix with entries (0, 0, 1, -4, 1, 1, 1).

    Diagonals at positions -3, -2, -1, 0, +1, +2, +3.
    """
    A = jnp.zeros((n, n), dtype=jnp.complex128)
    diag_vals = [(0, 0.0), (1, 0.0), (2, 1.0), (3, -4.0), (4, 1.0), (5, 1.0), (6, 1.0)]

    for offset, (diag_idx, val) in enumerate(diag_vals):
        k = diag_idx - 3  # Convert to diagonal offset (-3 to +3)
        if k >= 0:
            A = A.at[jnp.arange(n - k), jnp.arange(k, n)].set(val)
        else:
            A = A.at[jnp.arange(-k, n), jnp.arange(0, n + k)].set(val)
    return A


def build_problem(n: int = 100, matrix_type: str = "tridiag"):
    """Build the linear test problem Ax = b.

    Args:
        n: Problem dimension
        matrix_type: "tridiag" or "sevendiag"

    Returns:
        A: System matrix (n x n, complex128)
        b: Right-hand side (n, complex128)
        x0: Initial guess (zeros)
    """
    if matrix_type == "tridiag":
        A = make_tridiag_A(n)
    elif matrix_type == "sevendiag":
        A = make_sevendiag_A(n)
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}")

    b = jnp.zeros((n,), dtype=jnp.complex128).at[0].set(1.0)
    x0 = jnp.zeros((n,), dtype=jnp.complex128)
    return A, b, x0


def make_residual_fn(A: jnp.ndarray, b: jnp.ndarray):
    """Create residual function f(x) = b - Ax.

    For linear systems, the fixed-point iteration g(x) = x + f(x) = x + b - Ax
    converges if ||I - A|| < 1.
    """

    @jax.jit
    def residual_fn(x: jnp.ndarray) -> jnp.ndarray:
        return b - A @ x

    return residual_fn


# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------


def run_benchmark(
    n: int = 100,
    matrix_type: str = "tridiag",
    maxit: int = 100,
    tol: float = 1e-10,
    m_values: tuple[int, ...] = (1, 2, 4),
):
    """Run acceleration methods and collect results.

    Args:
        n: Problem size
        matrix_type: "tridiag" or "sevendiag"
        maxit: Maximum iterations
        tol: Convergence tolerance
        m_values: History depths to test

    Returns:
        List of (method_name, m, residual_norms, iterations) tuples
    """
    A, b, x0 = build_problem(n, matrix_type)
    residual_fn = make_residual_fn(A, b)

    results = []

    # Anderson Acceleration
    for m in m_values:
        result = anderson_acceleration(residual_fn, x0, m=m, maxit=maxit, tol=tol)
        results.append((f"Anderson(m={m})", m, result.residual_norms, result.iterations))
        print(f"Anderson(m={m}): {result.iterations} iters, converged={result.converged}, final_res={float(result.residual_norms[-1]):.2e}")

    # CROP (control residuals)
    for m in m_values:
        result = crop(residual_fn, x0, m=m, maxit=maxit, tol=tol)
        results.append((f"CROP(m={m})", m, result.residual_norms, result.iterations))
        print(f"CROP(m={m}): {result.iterations} iters, converged={result.converged}, final_res={float(result.residual_norms[-1]):.2e}")

    # rCROP (with real residuals - more robust)
    for m in m_values:
        result = rcrop(residual_fn, x0, m=m, maxit=maxit, tol=tol)
        results.append((f"rCROP(m={m})", m, result.residual_norms, result.iterations))
        print(f"rCROP(m={m}): {result.iterations} iters, converged={result.converged}, final_res={float(result.residual_norms[-1]):.2e}")

    return results


def plot_results(
    results: list,
    title: str = "Residual Convergence",
    save_path: str | None = None,
    show: bool = False,
):
    """Plot residual norms vs iteration for all methods.

    Args:
        results: List of (name, m, residual_norms, iterations) tuples
        title: Plot title
        save_path: If provided, save figure to this path
        show: If True, display interactively (blocking)
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color map for different m values
    colors = {1: "C0", 2: "C1", 4: "C2", 8: "C3"}
    linestyles = {"Anderson": "-", "CROP-Anderson": "--", "CROP": "-.", "rCROP": ":"}

    for name, m, res_norms, iters in results:
        # Determine method type and style
        if name.startswith("Anderson"):
            ls = linestyles["Anderson"]
            marker = "o"
        elif name.startswith("CROP-Anderson"):
            ls = linestyles["CROP-Anderson"]
            marker = "s"
        elif name.startswith("rCROP"):
            ls = linestyles["rCROP"]
            marker = "^"
        elif name.startswith("CROP"):
            ls = linestyles["CROP"]
            marker = "d"
        else:
            ls = "-"
            marker = "x"

        color = colors.get(m, "C4")
        iterations = jnp.arange(len(res_norms))

        ax.semilogy(
            iterations,
            res_norms,
            label=name,
            color=color,
            linestyle=ls,
            marker=marker,
            markevery=max(1, len(res_norms) // 10),
            markersize=5,
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$\|f(x)\|_2$", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()  # This blocks until window is closed
    
    plt.close(fig)


def main():
    """Run the synthetic problem benchmark."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="CROP acceleration method benchmark")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save plots (default: don't save)",
    )
    args = parser.parse_args()

    # Set up output directory if saving
    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Synthetic Problem Benchmark (Tridiagonal Matrix)")
    print("=" * 60)
    print()

    # Parameters matching the paper
    n = 100
    maxit = 100
    tol = 1e-10
    m_values = (1, 2, 4)

    print(f"Problem size: n = {n}")
    print(f"Matrix: tridiagonal (1, -4, 1)")
    print(f"Tolerance: {tol}")
    print(f"History depths: m = {m_values}")
    print()

    # Run tridiagonal problem (Figure 3a)
    results_tri = run_benchmark(
        n=n,
        matrix_type="tridiag",
        maxit=maxit,
        tol=tol,
        m_values=m_values,
    )

    print()
    plot_results(
        results_tri,
        title=f"Tridiagonal (1, -4, 1), n={n}, tol={tol}",
        save_path=outdir / "benchmark_tridiag.png" if outdir else None,
        show=args.show,
    )

    print()
    print("=" * 60)
    print("Seven-diagonal Matrix")
    print("=" * 60)
    print()

    # Run seven-diagonal problem (Figure 3b)
    results_seven = run_benchmark(
        n=n,
        matrix_type="sevendiag",
        maxit=maxit,
        tol=tol,
        m_values=m_values,
    )

    print()
    plot_results(
        results_seven,
        title=f"Seven-diagonal (0, 0, 1, -4, 1, 1, 1), n={n}, tol={tol}",
        save_path=outdir / "benchmark_sevendiag.png" if outdir else None,
        show=args.show,
    )


if __name__ == "__main__":
    main()
