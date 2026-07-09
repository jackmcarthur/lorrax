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
from jax.experimental import multihost_utils as _mh
import numpy as np

from common import minimax as _minimax
from .minimax_config import MinimaxConfig


_TINY = 1.0e-12


def _to_host_np(a, dtype=np.complex128):
    """Gather a (possibly sharded) JAX array to host as a NumPy array."""
    try:
        return np.asarray(_mh.process_allgather(a, tiled=True), dtype=dtype)
    except Exception:
        return np.asarray(jax.device_get(a), dtype=dtype)


def _scalar_to_host_float(a) -> float:
    """Fetch a scalar JAX value in a multihost-safe way."""

    if jax.process_count() > 1:
        gathered = np.asarray(_mh.process_allgather(jnp.asarray(a), tiled=False), dtype=np.float64)
        return float(gathered.reshape(-1)[0])
    return float(np.asarray(jax.device_get(a), dtype=np.float64))


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
class MinimaxNodes:
    """τ nodes + weights in complex form, passable to jit as a pytree.

    Both chi0's Laplace quad (real τ → Im(t)=0) and sigma's crossing /
    non-crossing quads (``-1j·τ`` or ``τ/ξ``) live in the same complex128
    storage so one sibling function shape (``minimax_tau_integrate_*``)
    handles both pipelines.
    """

    t: jax.Array       # complex128, shape (n,)
    alpha: jax.Array   # complex128, shape (n,)


jax.tree_util.register_dataclass(
    MinimaxNodes, data_fields=['t', 'alpha'], meta_fields=[])


def _laplace_to_minimax_nodes(
    tau: np.ndarray, alpha: np.ndarray, *, time_axis: str,
) -> MinimaxNodes:
    """Convert a (real τ, real α) Laplace quadrature into complex ``MinimaxNodes``.

    ``time_axis``:
      * ``'real'``      — chi0 Laplace: ``t = τ + 0j``, α cast to complex.
                          exp(-t·ΔE) stays real-valued for real ΔE.
      * ``'imag'``      — sigma Laplace windows (single/a_stripe/b_slab):
                          ``t = -1j·τ``, α cast to complex.
    """
    tau_j = jnp.asarray(np.asarray(tau, dtype=np.float64), dtype=jnp.float64)
    alpha_j = jnp.asarray(np.asarray(alpha, dtype=np.float64), dtype=jnp.float64)
    if time_axis == 'real':
        t = tau_j.astype(jnp.complex128)
    elif time_axis == 'imag':
        t = (-1j) * tau_j.astype(jnp.complex128)
    else:
        raise ValueError(
            f"Unknown time_axis={time_axis!r}; expected 'real' or 'imag'.")
    return MinimaxNodes(t=t, alpha=alpha_j.astype(jnp.complex128))


def _crossing_to_minimax_nodes(
    tau: np.ndarray, alpha: np.ndarray, *, time_axis: str,
) -> MinimaxNodes:
    """Convert a crossing quadrature into complex ``MinimaxNodes``.

    ``time_axis='crossing_hgl'`` keeps τ real (cast to complex) — the
    crossing window integrates ``Im[...]`` on the real-τ axis directly.
    Callers that need to rescale by 1/ξ apply that externally.
    """
    if time_axis != 'crossing_hgl':
        raise ValueError(
            f"Unknown time_axis={time_axis!r} for crossing quadrature; "
            f"expected 'crossing_hgl'.")
    tau_j = jnp.asarray(np.asarray(tau, dtype=np.float64), dtype=jnp.float64)
    alpha_j = jnp.asarray(np.asarray(alpha, dtype=np.float64), dtype=jnp.float64)
    return MinimaxNodes(
        t=tau_j.astype(jnp.complex128),
        alpha=alpha_j.astype(jnp.complex128),
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

    def to_minimax_nodes(self, *, time_axis: str) -> MinimaxNodes:
        """Return ``MinimaxNodes`` in the caller's sign convention.

        See ``_laplace_to_minimax_nodes`` for the set of accepted
        ``time_axis`` values.  The returned pytree is safe to close over
        in a jit or pass as an argument.
        """
        return _laplace_to_minimax_nodes(
            self.tau, self.alpha, time_axis=time_axis)


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

    def to_minimax_nodes(self, *, time_axis: str = 'crossing_hgl') -> MinimaxNodes:
        """Return ``MinimaxNodes`` for the crossing-window τ axis."""
        return _crossing_to_minimax_nodes(
            self.tau, self.alpha, time_axis=time_axis)


def fit_gn_ppm_from_wc_pair(
    Wc0_qmunu: jax.Array,
    Wc_probe_qmunu: jax.Array,
    probe_omega: complex,
    *,
    fallback_omega: float,
    n_mu_logical: int,
) -> tuple[jax.Array, jax.Array, jax.Array, float]:
    """Fit GN-PPM pole data elementwise on an already-sharded ``(q,mu,nu)`` tensor pair.

    Parameters
    ----------
    Wc0_qmunu
        ``W^c(0)`` in shape ``(nkx,nky,nkz,n_rmu,n_rmu)``.
    Wc_probe_qmunu
        ``W^c(z_probe)`` in the same shape/sharding as ``Wc0_qmunu``.
    probe_omega
        Complex probe frequency ``z_probe`` in Ry. For the standard GN fit this is
        purely imaginary, e.g. ``2j``.
    fallback_omega
        Positive real fallback pole in Ry for entries that do not produce a valid
        positive-real ``Omega^2`` estimate.
    n_mu_logical
        Logical centroid count (``meta.n_rmu``).  REQUIRED — the trailing
        (μ, ν) axes may carry the padded extent, and pad modes must be born
        DEAD here: ``Ω = 0``, ``B = 0``, ``valid = False``.  Handing them the
        live-looking fallback Ω instead used to inflate the mode census and
        the masked-Ω window statistics by a pad-extent- (= device-count-)
        dependent amount (ROOT_CAUSE.md 2026-07-08).  Zeroing Ω at birth
        makes every present and future ``Omega_q``/``B_q`` consumer
        structurally pad-safe: the ``Ω > 1e-14`` mode mask excludes pads with
        no mask argument anywhere downstream.  Pass the padded extent
        (all-true mask) when the inputs are unpadded.

    Returns
    -------
    omega_qmunu, B_qmunu, valid_qmunu, unfulfilled_fraction
        Elementwise GN-PPM parameters in the same ``(nkx,nky,nkz,n_rmu,n_rmu)``
        layout; ``unfulfilled_fraction`` counts LOGICAL modes only. The fit is
        pure local algebra: no host gathers and no communication beyond
        whatever sharding is already attached to the inputs.
    """

    Wc0 = jnp.asarray(Wc0_qmunu, dtype=jnp.complex128)
    Wc_probe = jnp.asarray(Wc_probe_qmunu, dtype=jnp.complex128)
    z_probe = jnp.asarray(probe_omega, dtype=jnp.complex128)

    n_mu = int(Wc0.shape[-1])
    n_log = int(n_mu_logical)
    if not (0 < n_log <= n_mu):
        raise ValueError(
            f"fit_gn_ppm_from_wc_pair: n_mu_logical={n_log} outside "
            f"(0, {n_mu}] for input extent {n_mu}.")
    mu_log = jnp.arange(n_mu) < n_log
    mode_mask = mu_log[:, None] & mu_log[None, :]   # (μ, ν) logical selector

    denom = Wc0 - Wc_probe
    safe = jnp.abs(denom) > 1.0e-14
    ratio = jnp.where(safe, Wc_probe / denom, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    omega_sq = -(z_probe * z_probe) * ratio
    omega_sq_re = jnp.real(omega_sq)
    good = (
        safe
        & jnp.isfinite(omega_sq_re)
        & (omega_sq_re > 0.0)
        & mode_mask
    )

    fallback = jnp.asarray(fallback_omega, dtype=jnp.float64)
    # Pad modes born DEAD: Ω = 0 (hence B = -Wc0·Ω/2 = 0) outside the
    # logical block — see ``n_mu_logical`` above.
    omega_vals = jnp.where(
        mode_mask, jnp.where(good, jnp.sqrt(omega_sq_re), fallback), 0.0)
    B_vals = -0.5 * Wc0 * omega_vals.astype(jnp.complex128)
    m = jnp.broadcast_to(mode_mask, good.shape)
    n_modes = jnp.sum(m.astype(jnp.float64))
    n_good = jnp.sum(good.astype(jnp.float64))
    unfulfilled_fraction = 1.0 - _scalar_to_host_float(
        n_good / jnp.maximum(n_modes, 1.0))
    return omega_vals, B_vals, good, unfulfilled_fraction


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




# ---------------------------------------------------------------------------
#  Quadrature builders — the χ₀/Σ frequency axes, solved on G's spectrum
#  (moved from gw/w_isdf.py 2026-07-09: B1 frequency code belongs with the
#  minimax engine, not one of its consumers).
# ---------------------------------------------------------------------------

def resolve_minimax_energy_reference(
    enk_v: jax.Array,
    enk_c: jax.Array,
    *,
    reference: str | float | int | None = "midgap",
    reference_fn: Callable[[jax.Array, jax.Array], float] | None = None,
) -> float:
    """Resolve the minimax energy reference used to shift band energies.

    This shift is algebraically neutral for χ0/W (only E_c-E_v enters), but
    exposing it at the top-level minimax pipeline keeps reference conventions
    explicit and synchronized with sigma paths.
    """
    if reference_fn is not None:
        return float(reference_fn(enk_v, enk_c))

    if reference is None:
        return 0.0
    if isinstance(reference, (int, float)):
        return float(reference)

    ref = str(reference).strip().lower()
    if ref in ("none", "raw", "zero"):
        return 0.0

    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64)
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64)
    vbm_ref = float(np.max(enk_v_host))
    cbm_ref = float(np.min(enk_c_host))

    if ref == "midgap":
        return 0.5 * (vbm_ref + cbm_ref)
    if ref == "vbm":
        return vbm_ref
    if ref == "cbm":
        return cbm_ref
    raise ValueError(f"Unknown minimax energy reference '{reference}'. Expected midgap/vbm/cbm/none or float.")


# ---------------------------------------------------------------------------
#  Top-level screening helpers (used directly by gw_jax.main)
# ---------------------------------------------------------------------------

def build_static_quadrature(wfns, minimax_config, *, print_fn=None):
    """Build static minimax quadrature and energy reference from wavefunction bundle.

    Returns (quad, e_ref) where quad is a LaplaceMinimaxQuadrature for 1/x
    on the band-energy interval, and e_ref is the global energy zero.
    """
    s = wfns.slices
    enk_v = wfns.enk[:, s.val]
    enk_c = wfns.enk[:, s.cond]
    e_ref = resolve_minimax_energy_reference(
        enk_v, enk_c, reference=minimax_config.energy_reference)

    # Interval derivation for 1/x on the band-energy span [x_min, x_max].
    # (Inlined from the former minimax_screening.build_static_minimax_window_pair;
    #  the window-pair object it returned was discarded here — only ``quad`` is used.)
    enk_v_host = np.asarray(jax.device_get(enk_v), dtype=np.float64)
    enk_c_host = np.asarray(jax.device_get(enk_c), dtype=np.float64)
    if enk_v_host.size == 0 or enk_c_host.size == 0:
        raise ValueError(
            "Cannot build minimax window with empty valence/conduction energies.")
    vmin = float(np.min(enk_v_host))
    vmax = float(np.max(enk_v_host))
    cmin = float(np.min(enk_c_host))
    cmax = float(np.max(enk_c_host))
    x_min = max(cmin - vmax, _TINY)
    x_max = max(cmax - vmin, x_min * (1.0 + 1.0e-9))
    quad = solve_laplace_minimax_interval(
        x_min,
        x_max,
        target_error=float(minimax_config.target_error),
        max_nodes=int(minimax_config.max_nodes),
        use_shipped_tables=bool(minimax_config.use_shipped_tables),
    )
    if print_fn is not None:
        R = quad.x_max / quad.x_min
        print_fn(
            "  Minimax static window: "
            f"x=[{quad.x_min:.6e}, {quad.x_max:.6e}] Ry, "
            f"R={R:.2f}, nodes={quad.node_count}, fit_err~{quad.max_error:.3e}"
        )
    return quad, e_ref


def build_imag_quadrature(quad, omega_p, minimax_config, *, print_fn=None):
    """Build imaginary-frequency minimax quadrature for x/(x²+ωp²).

    Uses the same energy interval as the static quadrature.
    """
    quad_imag = solve_laplace_minimax_imag_interval(
        quad.x_min, quad.x_max, float(omega_p),
        target_error=float(minimax_config.target_error),
        max_nodes=int(minimax_config.max_nodes),
    )
    if print_fn is not None:
        R = quad_imag.x_max / quad_imag.x_min
        print_fn(
            f"  PPM imag-freq quadrature (ωp={float(omega_p):.4f} Ry): "
            f"R={R:.1f}, nodes={quad_imag.node_count}, err~{quad_imag.max_error:.1e}")
    return quad_imag


def build_real_quadrature(quad, Omega, minimax_config, *, print_fn=None):
    """Build real-frequency (HL-PPM) χ₀(Ω) quadrature without a new minimax kernel.

    Decomposes the real-axis target into two ``1/y`` pieces and reuses
    the existing static (noncrossing) Laplace minimax twice::

        x / (x² - Ω²) = (1/2) · [ 1/(x - Ω)  +  1/(x + Ω) ]
                      = -(1/2)/(Ω - x)  +  (1/2)/(Ω + x)

    For ``Ω > x_max`` both ``Ω-x`` and ``Ω+x`` are strictly positive on
    ``x ∈ [x_min, x_max]``, so each can be approximated by a standard
    ``1/y`` minimax on the shifted interval (no new solver needed).

    Combining via the substitutions ``y = Ω-x`` and ``y = Ω+x`` and
    folding the constant ``e^{-τ·Ω}`` shift into the weights gives the
    same ``Σ_l α_l e^{-τ_l x}`` representation that ``compute_chi0``
    already consumes — with mixed-sign ``τ_l``: positive on the
    ``(Ω+x)`` branch, negative on the ``(Ω-x)`` branch.

    The numerical-stability prefold inside ``compute_chi0`` works
    transparently because in the realistic HL regime (``Ω`` ≈ 200 Ry,
    ``x_max`` ≈ 5 Ry → ``R'`` of either shifted interval ≈ 1.03)
    each ``1/y`` minimax needs only 1-3 nodes and ``|τ_l|`` ≈ ``1/Ω``,
    so any residual exponent ``|τ_l|·x_range`` ≈ 0.025 is harmless.

    Requires ``Omega > quad.x_max``.
    """
    Omega = float(Omega)
    if Omega <= float(quad.x_max):
        raise ValueError(
            f"build_real_quadrature requires Omega > x_max "
            f"(got Omega={Omega}, x_max={quad.x_max}). "
            f"HL-PPM is only defined for probes above all transitions."
        )
    target_error = float(minimax_config.target_error)
    max_nodes = int(minimax_config.max_nodes)

    # (Ω + x) branch: y ∈ [Ω + x_min, Ω + x_max] (strictly positive).
    quad_plus = solve_laplace_minimax_interval(
        Omega + quad.x_min, Omega + quad.x_max,
        target_error=target_error, max_nodes=max_nodes,
    )
    tau_plus = np.asarray(quad_plus.tau, dtype=np.float64)
    alpha_plus = (
        +0.5 * np.asarray(quad_plus.alpha, dtype=np.float64)
        * np.exp(-tau_plus * Omega)
    )

    # (Ω - x) branch: y ∈ [Ω - x_max, Ω - x_min] (strictly positive for Ω > x_max).
    quad_minus = solve_laplace_minimax_interval(
        Omega - quad.x_max, Omega - quad.x_min,
        target_error=target_error, max_nodes=max_nodes,
    )
    tau_minus_raw = np.asarray(quad_minus.tau, dtype=np.float64)
    # 1/(Ω - x) ≈ Σ α e^{-τ(Ω-x)} = Σ [α e^{-τ·Ω}] e^{+τ·x}
    # Cast into the kernel's e^{-τ'·x} form by τ' = -τ.  Decomposition sign is -1/2.
    tau_minus = -tau_minus_raw
    alpha_minus = (
        -0.5 * np.asarray(quad_minus.alpha, dtype=np.float64)
        * np.exp(-tau_minus_raw * Omega)
    )

    tau = np.concatenate([tau_plus, tau_minus])
    alpha = np.concatenate([alpha_plus, alpha_minus])
    err_combined = float(0.5 * (quad_plus.max_error + quad_minus.max_error))

    fused = LaplaceMinimaxQuadrature(
        x_min=float(quad.x_min),
        x_max=float(quad.x_max),
        tau=tau,
        alpha=alpha,
        max_error=err_combined,
    )

    if print_fn is not None:
        print_fn(
            f"  PPM real-freq quadrature (Ω={Omega:.4f} Ry, "
            f"decomposed via 1/y minimax): "
            f"+branch nodes={quad_plus.node_count} (R'={Omega/quad.x_min + quad.x_max/quad.x_min:.3f}), "
            f"-branch nodes={quad_minus.node_count} "
            f"(R'={(Omega-quad.x_min)/(Omega-quad.x_max):.3f}), "
            f"err~{err_combined:.1e}")
    return fused


