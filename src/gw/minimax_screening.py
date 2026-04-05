"""Minimax-window helpers for static chi0/W and GN-PPM extraction.

This module is intentionally scoped to the static path first:
- Build a single non-crossing minimax window pair compatible with ``w_isdf.compute_chi0``.
- Reuse existing sharded kernels (no duplicate FFT kernels here).
- Provide Godby-Needs PPM parameter extraction from precomputed chi matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.resources as importlib_resources
import json
import os
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from common import minimax as _minimax


_TINY = 1.0e-12


def _minimax_disk_cache_dir() -> Path | None:
    """Return the persistent minimax cache directory, creating it if needed."""

    if os.environ.get("LORRAX_DISABLE_MINIMAX_DISK_CACHE", "").strip().lower() in {"1", "true", "yes"}:
        return None
    cache_dir = os.environ.get("LORRAX_MINIMAX_CACHE_DIR")
    if not cache_dir:
        cache_dir = os.path.join(Path.home(), ".cache", "lorrax", "minimax_quadratures")
    path = Path(cache_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def _load_shipped_minimax_catalog() -> dict[str, object] | None:
    """Load the shipped minimax descriptor if the repo/package provides one."""

    try:
        catalog_path = importlib_resources.files("common").joinpath("minimax_assets", "catalog.json")
    except Exception:
        return None
    try:
        with catalog_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _load_shipped_minimax_table(entry: dict[str, object]) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Load one shipped quadrature table referenced by the descriptor."""

    rel_path = entry.get("file")
    if not isinstance(rel_path, str) or not rel_path:
        return None
    try:
        table_path = importlib_resources.files("common").joinpath("minimax_assets", rel_path)
        with table_path.open("rb") as fh:
            with np.load(fh, allow_pickle=False) as data:
                tau = np.asarray(data["tau"], dtype=np.float64)
                alpha = np.asarray(data["alpha"], dtype=np.float64)
                err = float(data["max_error"][()])
        return tau, alpha, err
    except Exception:
        return None


def _find_shipped_table_entry(
    family: str,
    *,
    range_value: float,
    target_error: float,
    max_nodes: int,
    target_kind: str | None = None,
    eps_q: float | None = None,
) -> dict[str, object] | None:
    """Return descriptor entry for the best shipped minimax table.

    The selection rule is intentionally conservative: the requested interval is rounded
    up to the next available tabulated range, and the requested error target is rounded
    down to the nearest stricter shipped error bound. That guarantees the loaded table
    is at least as accurate as the caller asked for under the same absolute-error
    convention used by the exact solver.
    """

    catalog = _load_shipped_minimax_catalog()
    if not catalog:
        return None
    entries = catalog.get("tables", [])
    if not isinstance(entries, list):
        return None

    candidates: list[tuple[tuple[float, float, int], dict[str, object]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("family") != family:
            continue
        try:
            entry_range = float(entry.get("range_max"))
            entry_err = float(entry.get("error_bound"))
            node_count = int(entry.get("node_count"))
        except Exception:
            continue
        if entry_range + 1.0e-12 < float(range_value):
            continue
        if entry_err - 1.0e-18 > float(target_error):
            continue
        if node_count > int(max_nodes):
            continue
        if target_kind is not None and str(entry.get("target_kind")) != str(target_kind):
            continue
        if eps_q is not None:
            try:
                if abs(float(entry.get("eps_q")) - float(eps_q)) > 1.0e-12:
                    continue
            except Exception:
                continue
        # Prefer the nearest larger range, then the least strict acceptable error,
        # then the fewest nodes.
        key = (entry_range, -entry_err, node_count)
        candidates.append((key, entry))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _pick_shipped_table(
    family: str,
    *,
    range_value: float,
    target_error: float,
    max_nodes: int,
    target_kind: str | None = None,
    eps_q: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Load the best shipped minimax table, if one safely matches the request.

    Selection rule:
      - choose the smallest tabulated range that is >= the requested range
      - choose the loosest available error bound that is still <= requested target_error
      - reject tables whose node count exceeds the caller's max_nodes

    This preserves the current absolute-error convention while avoiding retuning the
    quadrature at runtime. Using a table fitted on a larger interval is safe because
    the requested interval is a subset of the tabulated one.
    """
    entry = _find_shipped_table_entry(
        family,
        range_value=range_value,
        target_error=target_error,
        max_nodes=max_nodes,
        target_kind=target_kind,
        eps_q=eps_q,
    )
    if entry is None:
        return None
    return _load_shipped_minimax_table(entry)


def _minimax_disk_cache_path(namespace: str, payload: dict[str, object]) -> Path | None:
    cache_dir = _minimax_disk_cache_dir()
    if cache_dir is None:
        return None
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return cache_dir / f"{namespace}_{digest}.npz"


def _load_minimax_disk_cache(namespace: str, payload: dict[str, object]) -> tuple[np.ndarray, np.ndarray, float] | None:
    path = _minimax_disk_cache_path(namespace, payload)
    if path is None or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            tau = np.asarray(data["tau"], dtype=np.float64)
            w = np.asarray(data["w"], dtype=np.float64)
            err = float(data["err"][()])
        return tau, w, err
    except Exception:
        return None


def _store_minimax_disk_cache(
    namespace: str,
    payload: dict[str, object],
    tau: np.ndarray,
    w: np.ndarray,
    err: float,
) -> None:
    path = _minimax_disk_cache_path(namespace, payload)
    if path is None:
        return
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                tau=np.asarray(tau, dtype=np.float64),
                w=np.asarray(w, dtype=np.float64),
                err=np.asarray(float(err), dtype=np.float64),
            )
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


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
    valid_qmunu: jnp.ndarray
    unfulfilled_fraction: float


@lru_cache(maxsize=64)
def _solve_noncrossing_scaled_cached(
    logR_key: float,
    target_key: float,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    payload = {
        "solver": "noncrossing",
        "logR_key": float(logR_key),
        "target_key": float(target_key),
        "max_nodes": int(max_nodes),
    }
    cached = _load_minimax_disk_cache("noncrossing", payload)
    if cached is not None:
        return cached
    R = float(np.exp(logR_key))
    target = float(target_key)
    tau, w, _n, err = _minimax.noncrossing_grids(R, target, N_start=2, N_max=max_nodes)
    tau = np.asarray(tau, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    err = float(err)
    _store_minimax_disk_cache("noncrossing", payload, tau, w, err)
    return tau, w, err


@lru_cache(maxsize=64)
def _solve_noncrossing_imag_scaled_cached(
    logR_key: float,
    omega_hat_key: float,
    target_key: float,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    payload = {
        "solver": "noncrossing_imag",
        "logR_key": float(logR_key),
        "omega_hat_key": float(omega_hat_key),
        "target_key": float(target_key),
        "max_nodes": int(max_nodes),
    }
    cached = _load_minimax_disk_cache("noncrossing_imag", payload)
    if cached is not None:
        return cached
    R = float(np.exp(logR_key))
    omega_hat = float(omega_hat_key)
    target = float(target_key)
    tau, w, _n, err = _minimax.noncrossing_imag_grids(
        R, omega_hat, target, N_start=2, N_max=max_nodes,
    )
    tau = np.asarray(tau, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    err = float(err)
    _store_minimax_disk_cache("noncrossing_imag", payload, tau, w, err)
    return tau, w, err


@lru_cache(maxsize=128)
def _solve_crossing_scaled_cached(
    A_key: float,
    target_key: float,
    max_nodes: int,
    eps_q_key: float,
    target_kind: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    payload = {
        "solver": "crossing",
        "A_key": float(A_key),
        "target_key": float(target_key),
        "max_nodes": int(max_nodes),
        "eps_q_key": float(eps_q_key),
        "target_kind": str(target_kind),
    }
    cached = _load_minimax_disk_cache("crossing", payload)
    if cached is not None:
        return cached
    A_dim = float(A_key)
    target = float(target_key)
    eps_q = float(eps_q_key)
    if target_kind == "hgl":
        G_func = _minimax.G_hgl
        tau_max_func = _minimax.tau_max_hgl
    elif target_kind == "fermi":
        G_func = _minimax.G_fermi
        tau_max_func = _minimax.tau_max_fermi
    else:
        raise ValueError(f"Unknown crossing target_kind={target_kind!r}.")
    tau, w, _n, err = _minimax.crossing_grids(
        A_dim,
        target,
        G_func,
        tau_max_func,
        eps_q=eps_q,
        N_max=max_nodes,
    )
    tau = np.asarray(tau, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    err = float(err)
    _store_minimax_disk_cache("crossing", payload, tau, w, err)
    return tau, w, err


def solve_laplace_minimax_interval(
    x_min: float,
    x_max: float,
    *,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
    use_shipped_tables: bool = True,
) -> LaplaceMinimaxQuadrature:
    """Fit ``1/x ≈ sum alpha_l exp(-tau_l x)`` on ``[x_min, x_max]``.

    Error convention:
      1. The underlying table/solver works on the scaled interval ``[1, R]`` with
         ``R = x_max / x_min``.
      2. ``target_error`` is the L-infinity absolute error on that scaled problem:
         ``max_{y in [1,R]} |1/y - approx(y)|``.
      3. After rescaling back to ``[x_min, x_max]``, the physical absolute error is
         ``target_error / x_min``. This is not a relative-at-endpoint tolerance.
    """

    x_min = max(float(x_min), _TINY)
    x_max = max(float(x_max), x_min * (1.0 + 1.0e-9))
    target_error = max(float(target_error), 1.0e-14)
    max_nodes = max(4, int(max_nodes))

    R = x_max / x_min
    logR_key = float(np.log(R))
    target_key = float(target_error)

    shipped = None
    if use_shipped_tables:
        shipped = _pick_shipped_table(
            "noncrossing",
            range_value=R,
            target_error=target_error,
            max_nodes=max_nodes,
        )
    if shipped is not None:
        tau_hat, w_hat, err_hat = shipped
    else:
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


def solve_laplace_minimax_imag_interval(
    x_min: float,
    x_max: float,
    omega_p: float,
    *,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
) -> LaplaceMinimaxQuadrature:
    """Fit ``x/(x^2+omega_p^2) ≈ sum alpha_l exp(-tau_l x)`` on ``[x_min, x_max]``.

    Used for chi0(i*omega_p) where the resonant+antiresonant sum gives
    2*x/(x^2+omega_p^2) with x = E_c - E_v.
    """

    x_min = max(float(x_min), _TINY)
    x_max = max(float(x_max), x_min * (1.0 + 1.0e-9))
    omega_p = float(omega_p)
    target_error = max(float(target_error), 1.0e-14)
    max_nodes = max(4, int(max_nodes))

    R = x_max / x_min
    omega_hat = omega_p / x_min
    logR_key = float(np.log(R))

    tau_hat, w_hat, err_hat = _solve_noncrossing_imag_scaled_cached(
        round(logR_key, 12),
        round(omega_hat, 12),
        round(target_error, 14),
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
    use_shipped_tables: bool = True,
) -> CrossingMinimaxQuadrature:
    """Fit crossing regularization target on ``[0, A_dim]`` as ``sum alpha_l sin(tau_l u)``.

    Error convention:
      ``target_error`` is the L-infinity absolute error on the target function itself,
      e.g. ``max_{u in [0, A_dim]} |G(u) - approx(u)|`` for the chosen regularization
      target. This is the same absolute convention used by the current solver and the
      shipped tables below.
    """

    A_dim = max(float(A_dim), 1.0e-12)
    target_error = max(float(target_error), 1.0e-14)
    eps_q = max(float(eps_q), 1.0e-12)
    max_nodes = max(8, int(max_nodes))
    kind = str(target_kind).strip().lower()

    shipped = None
    if use_shipped_tables:
        shipped = _pick_shipped_table(
            "crossing",
            range_value=A_dim,
            target_error=target_error,
            max_nodes=max_nodes,
            target_kind=kind,
            eps_q=eps_q,
        )
    if shipped is not None:
        tau_hat, w_hat, err = shipped
    else:
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
    use_shipped_tables: bool = True,
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
        use_shipped_tables=use_shipped_tables,
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


def build_imag_freq_minimax_window_pair(
    enk_v: jax.Array,
    enk_c: jax.Array,
    omega_p: float,
    *,
    target_error: float = 1.0e-6,
    max_nodes: int = 64,
    print_fn: Callable[..., None] | None = None,
) -> tuple[list[MinimaxWindowPair], LaplaceMinimaxQuadrature]:
    """Build a minimax window pair for chi0(i*omega_p).

    Uses fresh minimax nodes that directly approximate x/(x^2+omega_p^2)
    on [x_min, x_max] where x = E_c - E_v.  This replaces the incorrect
    cos-reweighting of static nodes.
    """

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
    quad = solve_laplace_minimax_imag_interval(
        x_min,
        x_max,
        float(omega_p),
        target_error=target_error,
        max_nodes=max_nodes,
    )

    # Same compatibility transform as the static case:
    # w_i = alpha * exp(-tau) so that the chi kernel gives
    # -2 * alpha * exp(-tau * (E_c - E_v)).
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
        omega_hat = float(omega_p) / quad.x_min
        print_fn(
            f"  Minimax imag-freq window (ωp={omega_p:.4f} Ry): "
            f"x=[{quad.x_min:.6e}, {quad.x_max:.6e}] Ry, "
            f"R={R:.2f}, ω̂={omega_hat:.2f}, "
            f"nodes={quad.node_count}, fit_err~{quad.max_error:.3e}"
        )

    return [pair], quad


def extract_gn_ppm_parameters(
    V_qmunu: jax.Array,
    chi0_q: jax.Array,
    chi_iwp_q: jax.Array,
    *,
    omega_p: float,
    fallback_omega: float = 2.0,
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
        # Π = χ (I - Vχ)^-1; use right-side solve via transpose for stability.
        pi0[iq] = np.linalg.solve(A0.T, chi0_flat[iq].T).T
        pii[iq] = np.linalg.solve(Ai.T, chii_flat[iq].T).T

    denom = pi0 - pii
    safe = np.abs(denom) > 1.0e-14
    ratio = np.zeros_like(pi0.real)
    ratio[safe] = np.real(pii[safe] / denom[safe])

    good = np.isfinite(ratio) & (ratio > 0.0)
    omega_vals = np.full_like(ratio, fallback_omega, dtype=np.float64)
    if np.any(good):
        omega_vals[good] = omega_p * np.sqrt(ratio[good])
    B = -0.5 * pi0 * omega_vals
    unfulfilled_fraction = float(1.0 - np.mean(good.astype(np.float64)))

    omega_qmunu = omega_vals.reshape(nkx, nky, nkz, n_rmu, n_rmu)
    B_qmunu = B.reshape(nkx, nky, nkz, n_rmu, n_rmu)

    return GodbyNeedsPPM(
        omega_p=omega_p,
        omega_qmunu=jnp.asarray(omega_qmunu),
        b_qmunu=jnp.asarray(B_qmunu),
        valid_qmunu=jnp.asarray(good.reshape(nkx, nky, nkz, n_rmu, n_rmu)),
        unfulfilled_fraction=unfulfilled_fraction,
    )


def extract_gn_ppm_parameters_from_Wc(
    Wc0_q: jnp.ndarray,
    Wc_iwp_q: jnp.ndarray,
    *,
    omega_p: float,
    fallback_omega: float = 2.0,
) -> GodbyNeedsPPM:
    """Extract GN-PPM parameters from W^c(0) and W^c(i*omega_p)."""
    omega_p = float(omega_p)
    fallback_omega = float(fallback_omega)
    if omega_p <= 0.0:
        raise ValueError("omega_p must be > 0 for GN-PPM extraction.")

    Wc0 = np.asarray(jax.device_get(Wc0_q), dtype=np.complex128)
    Wci = np.asarray(jax.device_get(Wc_iwp_q), dtype=np.complex128)

    nkx, nky, nkz = Wc0.shape[0], Wc0.shape[1], Wc0.shape[2]
    n_q = nkx * nky * nkz
    n_rmu = Wc0.shape[4]

    Wc0_flat = Wc0[:, :, :, 0, :, 0, :].reshape(n_q, n_rmu, n_rmu)
    Wci_flat = Wci[:, :, :, 0, :, 0, :].reshape(n_q, n_rmu, n_rmu)

    denom = Wc0_flat - Wci_flat
    safe = np.abs(denom) > 1.0e-14
    ratio = np.zeros_like(Wc0_flat.real)
    ratio[safe] = np.real(Wci_flat[safe] / denom[safe])

    good = np.isfinite(ratio) & (ratio > 0.0)
    omega_vals = np.full_like(ratio, fallback_omega, dtype=np.float64)
    if np.any(good):
        omega_vals[good] = omega_p * np.sqrt(ratio[good])
    B = -0.5 * Wc0_flat * omega_vals
    unfulfilled_fraction = float(1.0 - np.mean(good.astype(np.float64)))

    omega_qmunu = omega_vals.reshape(nkx, nky, nkz, n_rmu, n_rmu)
    B_qmunu = B.reshape(nkx, nky, nkz, n_rmu, n_rmu)

    return GodbyNeedsPPM(
        omega_p=omega_p,
        omega_qmunu=jnp.asarray(omega_qmunu),
        b_qmunu=jnp.asarray(B_qmunu),
        valid_qmunu=jnp.asarray(good.reshape(nkx, nky, nkz, n_rmu, n_rmu)),
        unfulfilled_fraction=unfulfilled_fraction,
    )
