"""Host-side branch + window construction for the Σ_c(ω) GN-PPM integration.

This is the leaf of the Σ_PPM module family: it turns (E_A, Ω_q stats, ω-grid,
quadrature params) into host-side ``_SigmaBranch`` / ``_SigmaWindow`` lists.  It
imports only ``minimax_screening`` (the quadrature engine), jax, and numpy — no
GPU kernels, no accumulators, no config objects.  The driver (``ppm_sigma``) and
the accumulators (``ppm_accumulators``) import *from* this module; nothing here
imports back.

Branch decomposition
--------------------

Four branches span ω ∈ ℝ: {conduction/empty, valence/occupied} A-space ×
{+ω, −ω} half.  Each branch is a definite-sign Laplace problem; a branch
carries only its physical identity ``(space, neg_omega_half)`` — the signs
that make Σ_c correct are written inline where the physics puts them, not
stored as ±1 fields.  The pole S = E_A + Ω enters the Σ_c denominator either
as (ω̃ − S), when S can coincide with a grid ω and the crossing (HGL)
quadrature is needed, or as the sign-definite (ω̃ + S), a single Laplace
window.  On the +ω half the conduction (empty) A-space is the crossing one;
on the −ω half — where Σ_c is evaluated at |ω| with an overall −1 baked into
each window's prefactor — the valence (occupied) A-space is.

The −ω branches are computed explicitly, term by term; they are NOT the
conjugate of the +ω result.  (The often-quoted *global* identity
Σ_c(−ω) = −[Σ_c(ω)]* is false — the ω→−ω reflection holds only per pole
term, which is exactly what swaps the crossing role between the conduction
and valence A-spaces above.)
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


# ---------------------------------------------------------------------------
#  Crossing-quadrature conditioning (the load-bearing fix for the conduction
#  Σ_c instability — reports/gw_ppm_sigma_regularization_2026-07-20).
#
#  The HGL "core" crossing window fits its regularization target with a minimax
#  sin-sum over the dimensionless bandwidth  A_core = 2·T/ξ = 2·ω_max/ξ + 2·edge
#  (T = ω_max + edge·ξ).  The sin-fit is well-conditioned only for MODEST A: the
#  exact solver's weights Σ|α_hat| jump from ~2–3 at A≲24 to ~3e4 (A=83, ξ=0.25)
#  and ~2e5 (A=43, ξ=0.5) — an ill-conditioning that is *not even monotone in ξ*.
#  Those O(1e4–1e5) weights are near-cancelling, so they amplify ANY perturbation
#  of the per-τ operand σ(τ) by Σ|α|; σ(τ) carries the (large, ~1e4 in the ISDF
#  centroid basis, and mesh-sensitive) screened W, so on a multi-device mesh the
#  amplified perturbations do NOT cancel → Σ_c blows up to O(1e5) eV (device/mesh
#  dependent, gap-inverting) with O(1e3) eV imaginary parts even on one device.
#
#  Fix: floor ξ so A_core ≤ _CROSSING_A_MAX (well-conditioned regime).  ξ is a
#  *broadening*; the QP self-energy is evaluated off-pole (at E_DFT, away from the
#  plasmon poles), so a coarser near-pole broadening does not move the QP energies.
#  The floor scales with ω_max, so it only engages when the Σ_c ω-grid is wide
#  enough to force an ill-conditioned crossing; a narrow grid keeps the user's ξ.
_CROSSING_A_MAX = 24.0     # dimensionless bandwidth ceiling (Σ|α_hat| ~ 2–3 below)


def crossing_regularization_floor(omega_max_ry: float, edge_factor: float) -> float:
    """Minimum ξ (Ry) so the HGL core bandwidth A_core ≤ _CROSSING_A_MAX.

    A_core = 2·ω_max/ξ + 2·edge ≤ A_max  ⇔  ξ ≥ 2·ω_max/(A_max − 2·edge).
    Returns 0.0 when no floor is meaningful (degenerate edge/ω_max).
    """
    denom = _CROSSING_A_MAX - 2.0 * float(edge_factor)
    if denom <= 1.0 or float(omega_max_ry) <= 0.0:
        return 0.0
    return 2.0 * float(omega_max_ry) / denom


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
        """Int form of ``project`` — the code branch in _project_tau_onto_omega_np."""
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
    """One branch of the Σ_c(ω) sum.  Four branches cover ω ∈ ℝ.

    A branch stores only its physical identity — which A-space it sums and
    which ω half it covers.  The signs (which way the pole enters the
    denominator, the −1 that the −ω half carries) are derived from
    ``(space, neg_omega_half)`` at each expression that needs them; see
    ``_build_windows_for_branch``.
    """
    tag: str                    # human label ("ω≥E_F cond" etc.) — drives progress output
    E_A: jax.Array              # (nk, nb) energy-above-Fermi for A-space (E_cond or H_val)
    base_mask_A: jax.Array      # (nk, nb) bool — which bands in A-space contribute
    space: str                  # "cond" (empty A-space, E_c−E_F) / "val" (occupied, E_F−E_v)
    neg_omega_half: bool        # True on the −ω half — Σ_c evaluated at |ω|
    omega_abs: np.ndarray       # non-negative ω values to evaluate at (|ω_rel|)
    omega_idx: np.ndarray       # global ω indices these map into


def _iter_branches(
    *,
    omega_pos: np.ndarray, idx_pos: np.ndarray,
    omega_neg_abs: np.ndarray, idx_neg: np.ndarray,
    E_cond: jax.Array, H_val: jax.Array,
    cond_mask: jax.Array, val_mask: jax.Array,
) -> list[_SigmaBranch]:
    """Enumerate the 4 branches (A-space × ω-half), skipping empty ω halves.

    Each branch is one definite-sign Laplace problem.  The +ω half sums Σ_c
    directly on E_A = E_c−E_F (cond) or E_F−E_v (val); the −ω half evaluates at
    |ω| and carries an overall −1.  No sign is stored here — the branch's
    identity ``(space, neg_omega_half)`` is what the window builder reads to
    place the pole in the denominator (crossing vs sign-definite) and to fold
    the −ω-half −1 into each window's prefactor.

    The two −ω branches are built and integrated explicitly, exactly like the
    +ω ones — they are not conjugated from the +ω result (the global
    Σ_c(−ω) = −[Σ_c(ω)]* identity does not hold; the reflection is per pole
    term, which is why the crossing role moves from cond to val across halves).
    """
    branches: list[_SigmaBranch] = []
    if omega_pos.size:
        branches += [
            _SigmaBranch(tag="ω≥E_F cond", E_A=E_cond, base_mask_A=cond_mask,
                         space="cond", neg_omega_half=False,
                         omega_abs=omega_pos, omega_idx=idx_pos),
            _SigmaBranch(tag="ω≥E_F val",  E_A=H_val,  base_mask_A=val_mask,
                         space="val",  neg_omega_half=False,
                         omega_abs=omega_pos, omega_idx=idx_pos),
        ]
    if omega_neg_abs.size:
        branches += [
            _SigmaBranch(tag="ω<E_F cond", E_A=E_cond, base_mask_A=cond_mask,
                         space="cond", neg_omega_half=True,
                         omega_abs=omega_neg_abs, omega_idx=idx_neg),
            _SigmaBranch(tag="ω<E_F val",  E_A=H_val,  base_mask_A=val_mask,
                         space="val",  neg_omega_half=True,
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
    denom_can_cross: bool,
    neg_omega_half: bool,
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
    if denom_can_cross:
        # Denominator ω̃ − S: the Laplace argument stays within [S_min, S_max]
        # (the crossing region proper only appears once ω is large enough to
        # split off the three-window path).
        x_max = max(S_max, x_min * (1.0 + 1.0e-9))
    else:
        # Sign-definite denominator ω̃ + S: its argument grows with ω, so the
        # Laplace interval must reach S_max + ω_max.
        x_max = max(S_max + omega_max, x_min * (1.0 + 1.0e-9))
    q = solve_laplace_minimax_interval(
        x_min, x_max,
        target_error=target_error,
        max_nodes=max_nodes,
        use_shipped_tables=use_shipped_tables,
    )
    # ω enters the ω-kernel as exp(+i·ω·τ) for the (ω̃ − S) branch and
    # exp(−i·ω·τ) for the definite (ω̃ + S) branch; the (ω̃ + S) branch also
    # carries the −1 from the Laplace-vs-kernel identity.  The −ω half applies
    # an additional overall −1, folded into the prefactor here.
    omega_sign = +1 if denom_can_cross else -1
    prefactor = (1.0 if denom_can_cross else -1.0) * (-1.0 if neg_omega_half else 1.0)
    return [
        _SigmaWindow(
            name="single",
            nodes=q.to_minimax_nodes(time_axis='imag'),
            mask_A=np.asarray(base_mask_A, dtype=bool),
            E_ref_A=float(np.min(A_vals)),
            E_ref_B=float(mask_B_min),
            omega_sign=omega_sign,
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
    neg_omega_half: bool,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
    use_shipped_tables: bool,
) -> list[_SigmaWindow]:
    # This is the crossing branch: the pole S can coincide with a grid ω, so
    # the denominator is ω̃ − S and ω enters the kernel as exp(+i·ω·τ)
    # (omega_sign=+1) on every window.  The −ω half carries an overall −1,
    # folded into each window's prefactor via ``neg`` below.
    neg = -1.0 if neg_omega_half else 1.0
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
            prefactor = -1.0 * neg
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
            prefactor = +1.0 * neg

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
    space: str,
    neg_omega_half: bool,
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
    picks either the three-window crossing+stripe+slab decomposition (when the
    pole S can coincide with a grid ω) or a single sign-definite Laplace window
    (otherwise, or when the ω range is negligible).  Prints a one-line summary
    per returned window.

    Which case applies is a physical fact about the branch, not a stored sign:
    the pole S = E_A + Ω crosses the evaluation point (denominator ω̃ − S) for
    the conduction (empty) A-space on the +ω half and for the valence
    (occupied) A-space on the −ω half; otherwise the denominator is the
    sign-definite ω̃ + S.
    """
    if omega_nonneg_ry.size == 0:
        return []

    E_A_host = _to_host_np(E_A, dtype=np.float64, tiled=False)
    base_A_host = _to_host_np(base_mask_A, dtype=bool, tiled=False)

    _, mask_B_all_count, mask_B_all_min, mask_B_all_max = _masked_stats_device(
        Omega_q, base_mask_B)

    denom_can_cross = (space == "cond") != neg_omega_half
    omega_max = float(np.max(omega_nonneg_ry))
    if denom_can_cross and omega_max > 1.0e-14:
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
            neg_omega_half=neg_omega_half,
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
            denom_can_cross=denom_can_cross,
            neg_omega_half=neg_omega_half,
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
