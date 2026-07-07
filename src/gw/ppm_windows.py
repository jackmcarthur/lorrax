"""Host-side branch + window construction for the Σ_c(ω) GN-PPM integration.

This is the leaf of the Σ_PPM module family: it turns (E_A, Ω_q stats, ω-grid,
quadrature params) into host-side ``_SigmaBranch`` / ``_SigmaWindow`` lists.  It
imports only ``minimax_screening`` (the quadrature engine), jax, and numpy — no
GPU kernels, no accumulators, no config objects.  The driver (``ppm_sigma``) and
the accumulators (``ppm_accumulators``) import *from* this module; nothing here
imports back.

Branch decomposition
--------------------

Four branches span ω ∈ ℝ:

    (+ω, cond, kernel_sign=+1, scale=+1)   standard Laplace on E_A = E_c - E_F
    (+ω, val,  kernel_sign=-1, scale=+1)   sign-flipped kernel for H_val = E_F - E_v
    (-ω, cond, kernel_sign=-1, scale=-1)   evaluated at |ω|, symmetry factor -1
    (-ω, val,  kernel_sign=+1, scale=-1)   evaluated at |ω|, symmetry factor -1

Within each +ω branch the conduction kernel factors through a three-window
decomposition (Laplace core + crossing stripe + tail slab) when the ω range
is non-trivial; val and -ω branches use a single Laplace window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .minimax_screening import (
    MinimaxNodes,
    solve_laplace_minimax_interval,
    solve_phase_minimax_bandwidth,
)


@dataclass(frozen=True)
class _SigmaWindow:
    name: str
    nodes: MinimaxNodes        # (t, alpha) complex128, carries this window's τ points
    mask_A: np.ndarray         # (nk, nb)
    E_ref_A: float
    E_ref_B: float
    omega_sign: int
    project: str               # "full" (Laplace) or "imag" (crossing)
    prefactor: float
    mask_B_mode: str = "all"
    mask_B_threshold: float | None = None
    crossing_kind: str | None = None

    @property
    def n_tau(self) -> int:
        return int(self.nodes.t.shape[0])

    @property
    def project_code(self) -> int:
        """Int form of ``project`` — matches lax.switch branch in _project_tau_onto_omega."""
        if self.project == "full":
            return 0
        if self.project == "imag":
            return 1
        raise ValueError(f"Unknown window projection {self.project!r}; "
                         f"expected 'full' or 'imag'.")


# ---------------------------------------------------------------------------
#  Branch enumeration — the four (ω sign × cond/val) calls that together
#  sum to Σ_c(ω).  Split into a NamedTuple so every caller sees the same
#  physics labeling without copy-pasted kwargs.
# ---------------------------------------------------------------------------

class _SigmaBranch(NamedTuple):
    """One branch of the Σ_c(ω) sum.  Four branches cover ω ∈ ℝ."""
    tag: str                    # human label ("ω≥E_F cond" etc.) — drives progress output
    E_A: jax.Array              # (nk, nb) energy-above-Fermi for A-space (E_cond or H_val)
    base_mask_A: jax.Array      # (nk, nb) bool — which bands in A-space contribute
    kernel_sign: int            # +1 (Laplace on E_A ≥ 0)  /  -1 (sign-flipped kernel)
    scale: float                # global prefactor from ω ↔ -ω symmetry (±1)
    omega_abs: np.ndarray       # non-negative ω values to evaluate at (|ω_rel|)
    omega_idx: np.ndarray       # global ω indices these map into


def _iter_branches(
    *,
    omega_pos: np.ndarray, idx_pos: np.ndarray,
    omega_neg_abs: np.ndarray, idx_neg: np.ndarray,
    E_cond: jax.Array, H_val: jax.Array,
    cond_mask: jax.Array, val_mask: jax.Array,
) -> list[_SigmaBranch]:
    """Enumerate the 4 branches, skipping empty ω halves.

    Why the flipped signs?

        +ω  half:  Σ_c is a Laplace transform on E_A = E_c - E_F  (kernel_sign=+1).
                   For the val space, E_A = E_F - E_v ≥ 0 but the kernel picks up
                   the opposite sign so kernel_sign=-1 on the val side.
        -ω  half:  evaluate at |ω| and exploit Σ_c(-ω) = -[Σ_c(ω)]^* for the same
                   (E_A, mask) structure.  This means scale=-1 globally and
                   kernel_sign swaps between cond and val relative to the +ω half.
    """
    branches: list[_SigmaBranch] = []
    if omega_pos.size:
        branches += [
            _SigmaBranch(tag="ω≥E_F cond", E_A=E_cond, base_mask_A=cond_mask,
                         kernel_sign=+1, scale=+1.0,
                         omega_abs=omega_pos, omega_idx=idx_pos),
            _SigmaBranch(tag="ω≥E_F val",  E_A=H_val,  base_mask_A=val_mask,
                         kernel_sign=-1, scale=+1.0,
                         omega_abs=omega_pos, omega_idx=idx_pos),
        ]
    if omega_neg_abs.size:
        branches += [
            _SigmaBranch(tag="ω<E_F cond", E_A=E_cond, base_mask_A=cond_mask,
                         kernel_sign=-1, scale=-1.0,
                         omega_abs=omega_neg_abs, omega_idx=idx_neg),
            _SigmaBranch(tag="ω<E_F val",  E_A=H_val,  base_mask_A=val_mask,
                         kernel_sign=+1, scale=-1.0,
                         omega_abs=omega_neg_abs, omega_idx=idx_neg),
        ]
    return branches


def _to_host_np(a, dtype=np.complex128, *, tiled: bool = False):
    """Gather a possibly sharded array to host."""
    try:
        return np.asarray(
            jax.experimental.multihost_utils.process_allgather(a, tiled=tiled),
            dtype=dtype,
        )
    except Exception:
        return np.asarray(jax.device_get(a), dtype=dtype)


def _to_host_scalar(a, dtype=float):
    np_dtype = np.dtype(dtype)
    gathered = _to_host_np(jnp.asarray(a), dtype=np_dtype, tiled=False)
    return dtype(np.asarray(gathered).reshape(-1)[0])


def _masked_stats_device(values: jax.Array, mask: jax.Array) -> tuple[int, int, float | None, float | None]:
    """Return total size, masked count, and masked min/max."""
    total = int(np.prod(values.shape))
    count = int(_to_host_scalar(jnp.sum(mask, dtype=jnp.int64), int))
    if count == 0:
        return total, 0, None, None
    min_val = float(_to_host_scalar(jnp.min(jnp.where(mask, values, jnp.inf)), float))
    max_val = float(_to_host_scalar(jnp.max(jnp.where(mask, values, -jnp.inf)), float))
    return total, count, min_val, max_val


def _materialize_window_mask_B(
    window: _SigmaWindow,
    *,
    base_mask_B: jax.Array,
    Omega_q: jax.Array,
) -> jax.Array:
    """Build one window's B-side selector lazily on device."""
    mode = str(window.mask_B_mode)
    if mode == "all":
        return jnp.asarray(base_mask_B, dtype=bool)
    threshold = jnp.asarray(window.mask_B_threshold, dtype=Omega_q.dtype)
    if mode == "le_t":
        return jnp.asarray(base_mask_B, dtype=bool) & (Omega_q <= threshold)
    if mode == "gt_t":
        return jnp.asarray(base_mask_B, dtype=bool) & (Omega_q > threshold)
    raise ValueError(f"Unknown mask_B_mode={mode!r}")


# ---------------------------------------------------------------------------
#  Minimax window construction
# ---------------------------------------------------------------------------

def _build_single_sigma_window(
    *,
    E_A: np.ndarray,
    base_mask_A: np.ndarray,
    mask_B_count: int,
    mask_B_min: float | None,
    mask_B_max: float | None,
    omega_nonneg_ry: np.ndarray,
    kernel_sign: int,
    target_error: float,
    max_nodes: int,
    use_shipped_tables: bool,
) -> list[_SigmaWindow]:
    A_vals = E_A[base_mask_A]
    if A_vals.size == 0 or mask_B_count == 0 or mask_B_min is None or mask_B_max is None:
        return []
    S_min = float(np.min(A_vals) + mask_B_min)
    S_max = float(np.max(A_vals) + mask_B_max)
    omega_max = float(np.max(omega_nonneg_ry)) if omega_nonneg_ry.size else 0.0
    x_min = max(S_min, 1.0e-12)
    if kernel_sign == -1:
        x_max = max(S_max + omega_max, x_min * (1.0 + 1.0e-9))
    else:
        x_max = max(S_max, x_min * (1.0 + 1.0e-9))
    q = solve_laplace_minimax_interval(
        x_min, x_max,
        target_error=target_error,
        max_nodes=max_nodes,
        use_shipped_tables=use_shipped_tables,
    )
    prefactor = 1.0 if kernel_sign == +1 else -1.0
    return [
        _SigmaWindow(
            name="single",
            nodes=q.to_minimax_nodes(time_axis='imag'),
            mask_A=np.asarray(base_mask_A, dtype=bool),
            E_ref_A=float(np.min(A_vals)),
            E_ref_B=float(mask_B_min),
            omega_sign=int(kernel_sign),
            project="full",
            prefactor=float(prefactor),
            mask_B_mode="all",
        )
    ]


def _build_three_sigma_windows(
    *,
    E_A: np.ndarray,
    base_mask_A: np.ndarray,
    mask_B_all_count: int,
    mask_B_le_count: int,
    mask_B_le_min: float | None,
    mask_B_le_max: float | None,
    mask_B_gt_count: int,
    mask_B_gt_min: float | None,
    mask_B_gt_max: float | None,
    omega_nonneg_ry: np.ndarray,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
    use_shipped_tables: bool,
) -> list[_SigmaWindow]:
    omega_max = float(np.max(omega_nonneg_ry)) if omega_nonneg_ry.size else 0.0
    xi = max(float(regularization_width_ry), 1.0e-12)
    z_edge = float(edge_factor) * xi
    T = omega_max + z_edge
    windows: list[_SigmaWindow] = []

    for name in ("core", "a_stripe", "b_slab"):
        if name == "core":
            mA = base_mask_A & (E_A <= T)
            mask_B_mode = "le_t"
            count_B = mask_B_le_count
            B_min = mask_B_le_min
            B_max = mask_B_le_max
        elif name == "a_stripe":
            mA = base_mask_A & (E_A > T)
            mask_B_mode = "le_t"
            count_B = mask_B_le_count
            B_min = mask_B_le_min
            B_max = mask_B_le_max
        else:
            mA = base_mask_A
            mask_B_mode = "gt_t"
            count_B = mask_B_gt_count
            B_min = mask_B_gt_min
            B_max = mask_B_gt_max
        if not np.any(mA) or count_B == 0 or B_min is None or B_max is None:
            continue

        A_vals = E_A[mA]
        E_ref_A = float(np.min(A_vals))
        E_ref_B = float(B_min)

        if name == "core":
            A_core = max(2.0 * T / xi, 1.0e-8)
            q_cross = solve_phase_minimax_bandwidth(
                A_core,
                target_error=target_error,
                max_nodes=crossing_max_nodes,
                eps_q=crossing_eps_q,
                target_kind="hgl",
                use_shipped_tables=use_shipped_tables,
            )
            raw = q_cross.to_minimax_nodes(time_axis='crossing_hgl')
            # Crossing scaling: t = τ/ξ, α = α/ξ (both divided by ξ).
            nodes = MinimaxNodes(t=raw.t / xi, alpha=raw.alpha / xi)
            project = "imag"
            prefactor = -1.0
        else:
            S_min = float(np.min(A_vals) + B_min)
            S_max = float(np.max(A_vals) + B_max)
            x_min = max(S_min - (T - z_edge), z_edge, 1.0e-12)
            x_max = max(S_max, x_min * (1.0 + 1.0e-9))
            q = solve_laplace_minimax_interval(
                x_min, x_max,
                target_error=target_error,
                max_nodes=max_nodes,
                use_shipped_tables=use_shipped_tables,
            )
            nodes = q.to_minimax_nodes(time_axis='imag')
            project = "full"
            prefactor = +1.0

        windows.append(
            _SigmaWindow(
                name=name,
                nodes=nodes,
                mask_A=np.asarray(mA, dtype=bool),
                E_ref_A=E_ref_A,
                E_ref_B=E_ref_B,
                omega_sign=+1,
                project=project,
                prefactor=float(prefactor),
                mask_B_mode=mask_B_mode,
                mask_B_threshold=float(T),
                crossing_kind="hgl" if name == "core" else None,
            )
        )
    return windows


# ---------------------------------------------------------------------------
#  Sigma convolution — host-side window construction.  The device-side tau
#  loop lives in the driver (ppm_sigma); the two halves have no shared state
#  beyond the window list itself, so they're easy to test independently.
# ---------------------------------------------------------------------------

def _build_windows_for_branch(
    *,
    omega_nonneg_ry: np.ndarray,
    E_A: jax.Array,
    base_mask_A: jax.Array,
    Omega_q: jax.Array,
    base_mask_B: jax.Array,
    kernel_sign: int,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
    use_shipped_minimax_tables: bool,
    log_tag: str,
    print_fn,
) -> list[_SigmaWindow]:
    """Host-side window construction for a single branch.

    Gathers E_A and base_mask_A to host, computes masked B-side stats, and
    picks either a single-Laplace window (kernel_sign=-1 or small ω) or the
    three-window crossing+stripe+slab decomposition (kernel_sign=+1 with
    non-trivial ω range).  Prints a one-line summary per returned window.
    """
    if omega_nonneg_ry.size == 0:
        return []

    E_A_host = _to_host_np(E_A, dtype=np.float64, tiled=False)
    base_A_host = _to_host_np(base_mask_A, dtype=bool, tiled=False)

    _, mask_B_all_count, mask_B_all_min, mask_B_all_max = _masked_stats_device(
        Omega_q, base_mask_B)

    omega_max = float(np.max(omega_nonneg_ry))
    if kernel_sign == +1 and omega_max > 1.0e-14:
        xi = max(float(regularization_width_ry), 1.0e-12)
        T = omega_max + float(edge_factor) * xi
        _, mask_B_le_count, mask_B_le_min, mask_B_le_max = _masked_stats_device(
            Omega_q, base_mask_B & (Omega_q <= T))
        _, mask_B_gt_count, mask_B_gt_min, mask_B_gt_max = _masked_stats_device(
            Omega_q, base_mask_B & (Omega_q > T))
        windows = _build_three_sigma_windows(
            E_A=E_A_host, base_mask_A=base_A_host,
            mask_B_all_count=mask_B_all_count,
            mask_B_le_count=mask_B_le_count,
            mask_B_le_min=mask_B_le_min, mask_B_le_max=mask_B_le_max,
            mask_B_gt_count=mask_B_gt_count,
            mask_B_gt_min=mask_B_gt_min, mask_B_gt_max=mask_B_gt_max,
            omega_nonneg_ry=omega_nonneg_ry,
            regularization_width_ry=regularization_width_ry,
            edge_factor=edge_factor,
            target_error=target_error, max_nodes=max_nodes,
            crossing_eps_q=crossing_eps_q,
            crossing_max_nodes=crossing_max_nodes,
            use_shipped_tables=bool(use_shipped_minimax_tables),
        )
    else:
        windows = _build_single_sigma_window(
            E_A=E_A_host, base_mask_A=base_A_host,
            mask_B_count=mask_B_all_count,
            mask_B_min=mask_B_all_min, mask_B_max=mask_B_all_max,
            omega_nonneg_ry=omega_nonneg_ry,
            kernel_sign=kernel_sign,
            target_error=target_error, max_nodes=max_nodes,
            use_shipped_tables=bool(use_shipped_minimax_tables),
        )

    for win in windows:
        A_vals = E_A_host[win.mask_A]
        kind = "crossing" if win.crossing_kind else "Laplace"
        print_fn(
            f"    {log_tag} window \"{win.name}\" ({kind}): "
            f"{win.n_tau} nodes, err<{target_error:.0e}, "
            f"E_A=[{float(np.min(A_vals)):.4f}, {float(np.max(A_vals)):.4f}] Ry, "
            f"project={win.project}"
        )
    return windows
