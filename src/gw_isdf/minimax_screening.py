"""Minimax-window helpers for static chi0/W and GN-PPM extraction.

This module is intentionally scoped to the static path first:
- Build a single non-crossing minimax window pair compatible with ``w_isdf.compute_chi0``.
- Reuse existing sharded kernels (no duplicate FFT kernels here).
- Provide Godby-Needs PPM parameter extraction from precomputed chi matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import importlib.util
import jax
import jax.numpy as jnp
import numpy as np


_TINY = 1.0e-12


@dataclass(frozen=True)
class EnergyWindow:
    """Simple energy window descriptor compatible with ``w_isdf.compute_chi0``."""

    start_energy: float
    end_energy: float
    index: int = 0
    count: int = 1

    @property
    def upper_inclusive(self) -> bool:
        return self.index >= self.count - 1


@dataclass
class MinimaxWindowPair:
    """Single-window quadrature container matching the legacy window API."""

    val_window: EnergyWindow
    cond_window: EnergyWindow
    epsq: float
    tau_i: np.ndarray
    w_i: np.ndarray
    z_lm: float
    alpha_i: np.ndarray

    # Kept for compatibility with dynamic code paths that may touch these attrs.
    val_band_start: np.ndarray | None = None
    val_band_len: np.ndarray | None = None
    val_band_offset: np.ndarray | None = None
    cond_band_start: np.ndarray | None = None
    cond_band_len: np.ndarray | None = None
    cond_band_offset: np.ndarray | None = None
    max_val_len: int = 0
    max_cond_len: int = 0
    _has_band_ranges: bool = False

    def with_imag_freq_modulation(self, omega_imag: float) -> "MinimaxWindowPair":
        """Return a copy whose kernel weights include ``cos(omega_imag * tau)``.

        For chi(i*omega_imag), the combined resonant+antiresonant denominator
        factor is ``-2 * Delta / (Delta^2 + omega_imag^2)``, represented through
        the Laplace identity ``Delta/(Delta^2 + w^2) = int exp(-Delta t) cos(w t) dt``.
        """

        phase = np.cos(float(omega_imag) * self.tau_i)
        w_i = self.alpha_i * np.exp(-self.tau_i) * phase
        return MinimaxWindowPair(
            val_window=self.val_window,
            cond_window=self.cond_window,
            epsq=self.epsq,
            tau_i=np.asarray(self.tau_i, dtype=np.float64),
            w_i=np.asarray(w_i, dtype=np.float64),
            z_lm=float(self.z_lm),
            alpha_i=np.asarray(self.alpha_i, dtype=np.float64),
        )


@dataclass(frozen=True)
class LaplaceMinimaxQuadrature:
    """Quadrature summary for ``1/x`` on ``[x_min, x_max]``."""

    x_min: float
    x_max: float
    tau: np.ndarray
    alpha: np.ndarray
    max_error: float

    @property
    def node_count(self) -> int:
        return int(self.tau.shape[0])


@dataclass(frozen=True)
class CrossingMinimaxQuadrature:
    """Quadrature summary for crossing regularization target on ``[0, A_dim]``."""

    A_dim: float
    tau: np.ndarray
    alpha: np.ndarray
    max_error: float
    target_kind: str

    @property
    def node_count(self) -> int:
        return int(self.tau.shape[0])


@dataclass(frozen=True)
class GodbyNeedsPPM:
    """GN-PPM parameters in ISDF form."""

    omega_p: float
    omega_qmunu: jnp.ndarray
    b_qmunu: jnp.ndarray
    unfulfilled_fraction: float


@lru_cache(maxsize=1)
def _load_docs_minimax_module():
    """Load the canonical docs/minimax.py implementation."""
    root = Path(__file__).resolve().parents[2]
    path = root / "docs" / "minimax.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical minimax module not found at {path}. "
            "The minimax screening path requires docs/minimax.py."
        )
    spec = importlib.util.spec_from_file_location("isdf_docs_minimax", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load canonical minimax module from {path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "noncrossing_grids"):
        raise AttributeError(
            "Canonical minimax module is missing noncrossing_grids(R, eps, ...)."
        )
    return mod


@lru_cache(maxsize=64)
def _solve_noncrossing_scaled_cached(
    logR_key: float,
    target_key: float,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    R = float(np.exp(logR_key))
    target = float(target_key)
    docs_mod = _load_docs_minimax_module()
    tau, w, _n, err = docs_mod.noncrossing_grids(R, target, N_start=2, N_max=max_nodes)
    return np.asarray(tau, dtype=np.float64), np.asarray(w, dtype=np.float64), float(err)


@lru_cache(maxsize=128)
def _solve_crossing_scaled_cached(
    A_key: float,
    target_key: float,
    max_nodes: int,
    eps_q_key: float,
    target_kind: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    A_dim = float(A_key)
    target = float(target_key)
    eps_q = float(eps_q_key)
    docs_mod = _load_docs_minimax_module()
    if not hasattr(docs_mod, "crossing_grids"):
        raise AttributeError(
            "Canonical minimax module is missing crossing_grids(A, eps, ...)."
        )
    if target_kind == "hgl":
        G_func = docs_mod.G_hgl
        tau_max_func = docs_mod.tau_max_hgl
    elif target_kind == "fermi":
        G_func = docs_mod.G_fermi
        tau_max_func = docs_mod.tau_max_fermi
    else:
        raise ValueError(f"Unknown crossing target_kind={target_kind!r}.")
    tau, w, _n, err = docs_mod.crossing_grids(
        A_dim,
        target,
        G_func,
        tau_max_func,
        eps_q=eps_q,
        N_max=max_nodes,
    )
    return np.asarray(tau, dtype=np.float64), np.asarray(w, dtype=np.float64), float(err)


def solve_laplace_minimax_interval(
    x_min: float,
    x_max: float,
    *,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
) -> LaplaceMinimaxQuadrature:
    """Fit ``1/x ≈ sum alpha_l exp(-tau_l x)`` on ``[x_min, x_max]``."""

    x_min = max(float(x_min), _TINY)
    x_max = max(float(x_max), x_min * (1.0 + 1.0e-9))
    target_error = max(float(target_error), 1.0e-14)
    max_nodes = max(4, int(max_nodes))

    R = x_max / x_min
    logR_key = float(np.log(R))
    target_key = float(target_error)

    tau_hat, w_hat, err_hat = _solve_noncrossing_scaled_cached(
        round(logR_key, 12),
        round(target_key, 14),
        max_nodes,
    )

    tau = tau_hat / x_min
    alpha = w_hat / x_min
    err_abs = err_hat / x_min

    return LaplaceMinimaxQuadrature(
        x_min=x_min,
        x_max=x_max,
        tau=np.asarray(tau, dtype=np.float64),
        alpha=np.asarray(alpha, dtype=np.float64),
        max_error=float(err_abs),
    )


def solve_phase_minimax_bandwidth(
    A_dim: float,
    *,
    target_error: float = 1.0e-6,
    max_nodes: int = 500,
    eps_q: float = 1.0e-3,
    target_kind: str = "hgl",
) -> CrossingMinimaxQuadrature:
    """Fit crossing regularization target on ``[0, A_dim]`` as ``sum alpha_l sin(tau_l u)``."""

    A_dim = max(float(A_dim), 1.0e-12)
    target_error = max(float(target_error), 1.0e-14)
    eps_q = max(float(eps_q), 1.0e-12)
    max_nodes = max(8, int(max_nodes))
    kind = str(target_kind).strip().lower()

    tau_hat, w_hat, err = _solve_crossing_scaled_cached(
        round(A_dim, 12),
        round(target_error, 14),
        max_nodes,
        round(eps_q, 12),
        kind,
    )
    return CrossingMinimaxQuadrature(
        A_dim=A_dim,
        tau=np.asarray(tau_hat, dtype=np.float64),
        alpha=np.asarray(w_hat, dtype=np.float64),
        max_error=float(err),
        target_kind=kind,
    )


def build_static_minimax_window_pair(
    enk_v: jax.Array,
    enk_c: jax.Array,
    *,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
    print_fn: Callable[..., None] | None = None,
) -> tuple[list[MinimaxWindowPair], LaplaceMinimaxQuadrature]:
    """Build one minimax window pair that spans all valence/conduction states."""

    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64)
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64)
    if enk_v_host.size == 0 or enk_c_host.size == 0:
        raise ValueError("Cannot build minimax window with empty valence/conduction energies.")

    vmin = float(np.min(enk_v_host))
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    cmax = float(np.max(enk_c_host))

    x_min = max(cmin - vmax, _TINY)
    x_max = max(cmax - vmin, x_min * (1.0 + 1.0e-9))
    quad = solve_laplace_minimax_interval(
        x_min,
        x_max,
        target_error=target_error,
        max_nodes=max_nodes,
    )

    # Compatibility transform for legacy chi kernel:
    # with z_lm=1, passing w_i = alpha_i * exp(-tau_i) yields
    # total coefficient -2 * alpha_i * exp(-tau_i * DeltaE).
    w_kernel = quad.alpha * np.exp(-quad.tau)
    pair = MinimaxWindowPair(
        val_window=EnergyWindow(start_energy=vmin, end_energy=vmax, index=0, count=1),
        cond_window=EnergyWindow(start_energy=cmin, end_energy=cmax, index=0, count=1),
        epsq=float(target_error),
        tau_i=np.asarray(quad.tau, dtype=np.float64),
        w_i=np.asarray(w_kernel, dtype=np.float64),
        z_lm=1.0,
        alpha_i=np.asarray(quad.alpha, dtype=np.float64),
    )

    if print_fn is not None:
        R = quad.x_max / quad.x_min
        print_fn(
            "  Minimax static window: "
            f"x=[{quad.x_min:.6e}, {quad.x_max:.6e}] Ry, "
            f"R={R:.2f}, nodes={quad.node_count}, fit_err~{quad.max_error:.3e}"
        )

    return [pair], quad


def extract_gn_ppm_parameters(
    V_qmunu: jax.Array,
    chi0_q: jax.Array,
    chi_iwp_q: jax.Array,
    *,
    omega_p: float,
    fallback_omega: float = 1.0,
) -> GodbyNeedsPPM:
    """Extract Godby-Needs PPM parameters from chi(0) and chi(i*omega_p)."""

    omega_p = float(omega_p)
    fallback_omega = float(fallback_omega)
    if omega_p <= 0.0:
        raise ValueError("omega_p must be > 0 for GN-PPM extraction.")

    V = np.asarray(jax.device_get(V_qmunu), dtype=np.complex128)
    chi0 = np.asarray(jax.device_get(chi0_q), dtype=np.complex128)
    chii = np.asarray(jax.device_get(chi_iwp_q), dtype=np.complex128)

    nkx, nky, nkz = chi0.shape[0], chi0.shape[1], chi0.shape[2]
    n_q = nkx * nky * nkz
    n_rmu = chi0.shape[4]

    V_flat = V[0, 0, 0].reshape(n_q, n_rmu, n_rmu)
    chi0_flat = chi0[:, :, :, 0, :, 0, :].reshape(n_q, n_rmu, n_rmu)
    chii_flat = chii[:, :, :, 0, :, 0, :].reshape(n_q, n_rmu, n_rmu)

    eye = np.eye(n_rmu, dtype=np.complex128)
    pi0 = np.zeros_like(chi0_flat)
    pii = np.zeros_like(chii_flat)
    for iq in range(n_q):
        A0 = eye - V_flat[iq] @ chi0_flat[iq]
        Ai = eye - V_flat[iq] @ chii_flat[iq]
        pi0[iq] = np.linalg.solve(A0, chi0_flat[iq])
        pii[iq] = np.linalg.solve(Ai, chii_flat[iq])

    denom = pi0 - pii
    safe = np.abs(denom) > 1.0e-14
    ratio = np.zeros_like(pi0.real)
    ratio[safe] = np.real(pii[safe] / denom[safe])

    good = ratio > 0.0
    omega_vals = np.where(good, omega_p * np.sqrt(ratio), fallback_omega)
    B = -0.5 * pi0 * omega_vals
    unfulfilled_fraction = float(1.0 - np.mean(good.astype(np.float64)))

    omega_qmunu = omega_vals.reshape(nkx, nky, nkz, n_rmu, n_rmu)
    B_qmunu = B.reshape(nkx, nky, nkz, n_rmu, n_rmu)

    return GodbyNeedsPPM(
        omega_p=omega_p,
        omega_qmunu=jnp.asarray(omega_qmunu),
        b_qmunu=jnp.asarray(B_qmunu),
        unfulfilled_fraction=unfulfilled_fraction,
    )
