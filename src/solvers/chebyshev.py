"""
solvers/chebyshev.py — Chebyshev polynomial expansion and KPM utilities.

Provides the Kernel Polynomial Method (KPM) building blocks:
  - Jackson damping coefficients
  - Chebyshev moment computation via stochastic trace estimation
  - DOS reconstruction from damped moments
  - Spectral window partitioning from a DOS

No physics knowledge — works for any Hermitian operator given a matvec
that acts on the rescaled H_tilde = (H - center) / half_width with
spectrum in [-1, 1].

Usage
-----
    from solvers.chebyshev import chebyshev_moments, jackson_coefficients, reconstruct_dos

    mu = chebyshev_moments(apply_h_tilde, dim, n_moments=200, n_random=5)
    sigma = jackson_coefficients(n_moments)
    rho = reconstruct_dos(mu * sigma, E_grid, center, half_width)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp

from common.collectives import gather_to_host


def jackson_coefficients(M: int) -> np.ndarray:
    """Jackson damping coefficients sigma_p for p = 0, ..., M.

    Parameters
    ----------
    M : int
        Number of Chebyshev moments.

    Returns
    -------
    sigma : ndarray of shape (M + 1,)
    """
    p = np.arange(M + 1)
    sigma = (
        (M - p + 1) * np.cos(np.pi * p / (M + 1))
        + np.sin(np.pi * p / (M + 1)) / np.tan(np.pi / (M + 1))
    ) / (M + 1)
    return sigma


# ═══════════════════════════════════════════════════════════════════════
#  JIT'd Chebyshev recurrence kernel
# ═══════════════════════════════════════════════════════════════════════

def make_chebyshev_recurrence(
    apply_h_tilde: Callable[..., jax.Array],
    n_moments: int,
    operands: tuple = (),
) -> Callable[[jax.Array], jax.Array]:
    """Return a JIT'd function that computes all Chebyshev moments for one vector.

    The returned callable runs the three-term recurrence entirely on-device
    via lax.fori_loop — one device_get per random vector instead of per moment.

    Parameters
    ----------
    apply_h_tilde : x -> H_tilde x
        Matvec for the rescaled operator (spectrum in [-1, 1]).
        Should already be @jax.jit or jittable.
    n_moments : int
        Number of Chebyshev moments M (compiled into the XLA loop).

    Returns
    -------
    recurrence : (x_rand,) -> mu
        JIT'd function mapping a random vector to (n_moments + 1,) raw moments.
    """
    # WHY THE OPERANDS ARE ARGUMENTS AND NOT A CLOSURE (registered R1).
    #
    # This used to be a single ``@jax.jit def recurrence(x_rand)`` closing over
    # ``apply_h_tilde``.  Closing over a *callable* is legal.  The problem is
    # what the callable closes over: on the BSE path
    # (``bse.bse_kpm.make_bse_h_tilde``) it is itself jitted and captures the
    # ten MESH-SHARDED operands ``psi_c_X … M_Y``.  Tracing inlines those, and
    # at P>1 jax refuses -- "Closing over jax.Array that spans non-addressable
    # devices" -- at TRACE time, before any of the P>1 ``device_get`` fixes
    # downstream can matter.  Same family as ``bse_feast._get_feast_runner``.
    #
    # So the jitted body takes them as ARGUMENTS, and a plain-Python wrapper
    # holds them and forwards them at call time.  That wrapper is not traced,
    # so its closure is legal at any P.  This is the pattern
    # ``bse.bse_davidson_helpers.bse_diagonal_precond`` already documents.
    @jax.jit
    def _recurrence(x_rand, *ops):
        t_prev = x_rand
        t_curr = apply_h_tilde(x_rand, *ops)

        mu_init = jnp.zeros(n_moments + 1, dtype=jnp.float64)
        mu_init = mu_init.at[0].set(jnp.vdot(x_rand, t_prev).real)
        mu_init = mu_init.at[1].set(jnp.vdot(x_rand, t_curr).real)

        def body(p, carry):
            mu, t_prev, t_curr = carry
            t_new = 2.0 * apply_h_tilde(t_curr, *ops) - t_prev
            mu = mu.at[p].set(jnp.vdot(x_rand, t_new).real)
            return mu, t_curr, t_new

        mu, _, _ = jax.lax.fori_loop(2, n_moments + 1, body, (mu_init, t_prev, t_curr))
        return mu

    def recurrence(x_rand):
        return _recurrence(x_rand, *operands)

    return recurrence


# ═══════════════════════════════════════════════════════════════════════
#  Stochastic trace estimation
# ═══════════════════════════════════════════════════════════════════════

def chebyshev_moments(
    apply_h_tilde: Callable[[jax.Array], jax.Array],
    dim: int,
    n_moments: int,
    n_random: int,
    *,
    seed: int = 0,
    dtype_real: jnp.dtype = jnp.float64,
    make_random_vector: Callable | None = None,
    operands: tuple = (),
    verbose: bool = True,
) -> np.ndarray:
    """Compute Chebyshev moments mu_0, ..., mu_M via stochastic trace estimation.

    Builds a JIT'd recurrence kernel (compiled once, reused per random vector)
    and accumulates moments on the host.

    Parameters
    ----------
    apply_h_tilde : x -> H_tilde x
        Matvec for the rescaled Hermitian operator with spectrum in [-1, 1].
        Must accept and return arrays of the same shape.
    dim : int
        Hilbert space dimension (for normalizing the trace).
    n_moments : int
        Number of Chebyshev moments M.
    n_random : int
        Number of stochastic random vectors.
    seed : int
        Random seed.
    dtype_real : jnp.dtype
        Real dtype for random vectors (float32 or float64).
    make_random_vector : callable, optional
        (key) -> x_random.  Custom random vector generator (e.g. for masking
        padded dimensions).  If None, generates flat Rademacher +/-1 vectors
        of shape (dim,).
    operands : tuple, optional
        Extra arrays forwarded to ``apply_h_tilde`` as RUNTIME ARGUMENTS of the
        jitted recurrence, rather than captured in its closure.  Pass the
        mesh-sharded operands here: closing over an array that spans
        non-addressable devices is refused at trace time at P>1 (registered
        R1).  Default ``()`` -- ``apply_h_tilde`` is then the one-argument
        ``x -> H_tilde x`` every existing caller already passes.
    verbose : bool
        Print per-vector progress.

    Returns
    -------
    mu : ndarray of shape (n_moments + 1,)
        Raw (undamped) Chebyshev moments.
    """
    dtype_cplx = jnp.complex64 if dtype_real == jnp.float32 else jnp.complex128

    def _default_random_vector(key):
        x = (
            2.0 * jax.random.bernoulli(key, shape=(dim,)).astype(dtype_real) - 1.0
        )
        return x.astype(dtype_cplx)

    if make_random_vector is None:
        make_random_vector = _default_random_vector

    recurrence = make_chebyshev_recurrence(apply_h_tilde, n_moments, operands)

    key = jax.random.PRNGKey(seed)
    mu = np.zeros(n_moments + 1)

    for r in range(n_random):
        if verbose:
            print(f"  Random vector {r + 1}/{n_random}...")
        key, subkey = jax.random.split(key)
        x_rand = make_random_vector(subkey)

        # One device round-trip per random vector (not per moment).
        # ``gather_to_host``, not a bare ``device_get``: the moment vector is
        # the output of reductions over a caller-supplied ``apply_h_tilde``, so
        # this module cannot know whether the mesh left it replicated or tiled,
        # and ``device_get`` raises on the tiled case at P>1.  The helper's
        # replicated/1-GPU arms ARE the plain ``device_get`` that was here, so
        # this costs nothing where it was already correct.  Gated module-wide by
        # ``tests/test_solver_p_gt_1_safety.py``.
        mu_r = recurrence(x_rand)
        mu += gather_to_host(mu_r)

    mu /= n_random * dim
    return mu


# ═══════════════════════════════════════════════════════════════════════
#  DOS reconstruction (pure NumPy)
# ═══════════════════════════════════════════════════════════════════════

def reconstruct_dos(
    mu: np.ndarray,
    E_grid: np.ndarray,
    center: float,
    half_width: float,
) -> np.ndarray:
    """Reconstruct DOS on a physical energy grid from Chebyshev moments.

    Uses the expansion:
        rho(E) = [mu_0 + 2 sum_{p=1}^M mu_p T_p(e)] / (pi * hw * sqrt(1 - e^2))
    where e = (E - center) / half_width.

    Parameters
    ----------
    mu : ndarray, shape (M+1,)
        (Jackson-damped) Chebyshev moments.
    E_grid : ndarray
        Energy grid (same units as center/half_width).
    center, half_width : float
        Rescaling parameters.

    Returns
    -------
    rho : ndarray
        DOS (states per unit energy).
    """
    E_tilde = (E_grid - center) / half_width
    E_tilde = np.clip(E_tilde, -1 + 1e-10, 1 - 1e-10)

    M = len(mu) - 1

    T_prev = np.ones_like(E_tilde)
    T_curr = E_tilde.copy()

    cheb_sum = mu[0] * T_prev + 2.0 * mu[1] * T_curr

    for p in range(2, M + 1):
        T_new = 2.0 * E_tilde * T_curr - T_prev
        cheb_sum += 2.0 * mu[p] * T_new
        T_prev = T_curr
        T_curr = T_new

    rho = cheb_sum / (np.pi * half_width * np.sqrt(1 - E_tilde**2))
    return rho


# ═══════════════════════════════════════════════════════════════════════
#  Spectral window partitioning (pure NumPy)
# ═══════════════════════════════════════════════════════════════════════

def partition_windows(
    energy_grid: np.ndarray,
    dos: np.ndarray,
    n_windows: int,
    *,
    energy_min: float | None = None,
    energy_max: float | None = None,
) -> np.ndarray:
    """Partition the spectrum into equal-weight windows from a DOS.

    Integrates the DOS to build a CDF, then places window edges at
    equal-mass quantiles.

    Parameters
    ----------
    energy_grid : ndarray
        Energy grid (monotone or unordered).
    dos : ndarray
        Density of states on the grid (same units as 1/energy_grid).
    n_windows : int
        Number of windows.
    energy_min, energy_max : float, optional
        Bounds on the grid to use.

    Returns
    -------
    windows : ndarray of shape (n_windows, 2)
        Window bounds [E_min, E_max] in the same units as energy_grid.
    """
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")

    energy_grid = np.asarray(energy_grid, dtype=float)
    dos = np.asarray(dos, dtype=float)
    if energy_grid.shape != dos.shape:
        raise ValueError("energy_grid and dos must have the same shape")

    mask = np.ones_like(energy_grid, dtype=bool)
    if energy_min is not None:
        mask &= energy_grid >= energy_min
    if energy_max is not None:
        mask &= energy_grid <= energy_max
    energy_grid = energy_grid[mask]
    dos = dos[mask]
    if energy_grid.size < 2:
        raise ValueError("energy grid must contain at least two points after masking")

    order = np.argsort(energy_grid)
    energy_grid = energy_grid[order]
    dos = dos[order]
    dos = np.maximum(dos, 0.0)

    d_energy = np.diff(energy_grid)
    avg = 0.5 * (dos[1:] + dos[:-1])
    cdf = np.concatenate(([0.0], np.cumsum(avg * d_energy)))
    total = cdf[-1]
    if total <= 0.0:
        raise ValueError("non-positive total integral; check inputs")

    targets = np.linspace(0.0, total, n_windows + 1)
    edges = np.interp(targets, cdf, energy_grid)

    return np.column_stack((edges[:-1], edges[1:]))
