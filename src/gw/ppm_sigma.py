"""GN-PPM construction from W(0), W(iω_p) and Σ_c(ω) frequency integration.

What this module computes
-------------------------

    Σ^c_nm(k, ω) = Σ_{branches} Σ_{windows} Σ_τ  α(τ) · e^{i·ω_sign·ω·τ}
                                                 · project[ σ^τ_nmk(τ) ]
                                                 · pref · scale

Per branch the τ nodes are placed by a minimax quadrature chosen from the
range of E_A = E_c − E_F (cond) or E_F − E_v (val) and the PPM pole
frequencies Ω_q.  Each τ node fires one sharded GPU kernel (σ^τ) that
evaluates the single-tau integrand:

    σ^τ_nmk(τ) = project[ FFT[ G(τ) · W(τ) / √N_k ] ]
    G(τ)       = diag[ e^{-i(E_A - E_ref_A)·τ} ] · mask_A           (A = val or cond)
    W(τ)       = Σ_μν  B_q · e^{-i(Ω_q - E_ref_B)·τ}  · mask_B      (PPM pole sum)

The ω-dependence is *linear* in τ (only the exp(iω·τ) kernel involves ω),
so every τ contribution contributes to all ω in one shot.

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

File layout
-----------

    data classes               PPMBuildResult / SigmaOmegaResult / _SigmaWindow / _SigmaPhysicsState / _SigmaBranch
    leaf jits (physics)        _prepare_sigma_state · _project_tau_onto_omega · _accumulate_tau_into_window
    cached kernel factories    _get_sigma_channel_pipeline · _get_sigma_tau_channel_kernel
    PPM fit                    fit_gn_ppm
    host-side window build     _build_single_sigma_window · _build_three_sigma_windows
    accumulators               _SigmaAccumulator protocol
                               _BufferedGpuAccumulator · _StreamedH5Accumulator
    branch orchestration       _build_windows_for_branch (host) · _integrate_tau_windows_for_branch (device)
                               _run_sigma_branch (thin orchestrator)
    top-level driver           compute_sigma_c_ppm_omega_grid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple
import os

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import h5py

from common import jax_profile
from .minimax_config import MinimaxConfig, SigmaQuadratureConfig
from .minimax_screening import (
    build_static_minimax_window_pair,
    fit_gn_ppm_from_wc_pair,
    solve_laplace_minimax_interval,
    solve_laplace_minimax_imag_interval,
    solve_phase_minimax_bandwidth,
)
from . import w_isdf


@dataclass(frozen=True)
class PPMBuildResult:
    omega_p: float
    W0_q: jax.Array           # (nq, μ, μ) flat-q
    Wiwp_q: jax.Array         # (nq, μ, μ) flat-q
    B_q: jax.Array            # (nq, μ, μ) PPM amplitude
    Omega_q: jax.Array        # (nq, μ, μ) PPM pole frequency
    valid_mask_q: jax.Array   # (nq, μ, μ)
    unfulfilled_fraction: float
    n_nodes_static: int


@dataclass(frozen=True)
class SigmaOmegaResult:
    omega_ry: np.ndarray
    omega_ev: np.ndarray
    sigma_c_kij: jax.Array | None      # (n_omega, nk, nb, nb) or None if streamed
    sigma_kij_h5_path: str | None = None


@dataclass(frozen=True)
class _SigmaWindow:
    name: str
    t_nodes: np.ndarray        # (n_tau,)
    alpha: np.ndarray          # (n_tau,)
    mask_A: np.ndarray         # (nk, nb)
    mask_B: np.ndarray | None  # (nk, nb)
    E_ref_A: float
    E_ref_B: float
    omega_sign: int
    project: str               # "full" (Laplace) or "imag" (crossing)
    prefactor: float
    mask_B_mode: str = "explicit"
    mask_B_threshold: float | None = None
    crossing_kind: str | None = None


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
    omega_sign_flip: int        # always +1 in current scheme (kept for generality)
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
                         kernel_sign=+1, omega_sign_flip=1, scale=+1.0,
                         omega_abs=omega_pos, omega_idx=idx_pos),
            _SigmaBranch(tag="ω≥E_F val",  E_A=H_val,  base_mask_A=val_mask,
                         kernel_sign=-1, omega_sign_flip=1, scale=+1.0,
                         omega_abs=omega_pos, omega_idx=idx_pos),
        ]
    if omega_neg_abs.size:
        branches += [
            _SigmaBranch(tag="ω<E_F cond", E_A=E_cond, base_mask_A=cond_mask,
                         kernel_sign=-1, omega_sign_flip=1, scale=-1.0,
                         omega_abs=omega_neg_abs, omega_idx=idx_neg),
            _SigmaBranch(tag="ω<E_F val",  E_A=H_val,  base_mask_A=val_mask,
                         kernel_sign=+1, omega_sign_flip=1, scale=-1.0,
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
    if mode == "explicit":
        if window.mask_B is None:
            raise ValueError("window.mask_B must be provided when mask_B_mode='explicit'.")
        return jnp.asarray(window.mask_B, dtype=bool)
    if mode == "all":
        return jnp.asarray(base_mask_B, dtype=bool)
    threshold = jnp.asarray(window.mask_B_threshold, dtype=Omega_q.dtype)
    if mode == "le_t":
        return jnp.asarray(base_mask_B, dtype=bool) & (Omega_q <= threshold)
    if mode == "gt_t":
        return jnp.asarray(base_mask_B, dtype=bool) & (Omega_q > threshold)
    raise ValueError(f"Unknown mask_B_mode={mode!r}")


_sigma_tau_channel_kernel_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}
_sigma_channel_pipeline_cache: dict[tuple[object, ...], Callable[..., jax.Array]] = {}


# ---------------------------------------------------------------------------
#  Physics-state prep — single jit that collapses the scattered trace-time
#  jnp operations the driver used to emit (Fermi level, band masks, PPM
#  pole masks, invalid-count tallies).
# ---------------------------------------------------------------------------

class _SigmaPhysicsState(NamedTuple):
    efermi: jax.Array          # scalar
    E_cond: jax.Array          # (nk, nb_full)  max(enk - efermi, 0)
    H_val: jax.Array           # (nk, nb_full)  max(efermi - enk, 0)
    cond_mask: jax.Array       # (nk, nb_full)  bool
    val_mask: jax.Array        # (nk, nb_full)  bool
    B_corr: jax.Array          # (nq, μ, μ)     c128, ready-to-contract B_q
    Omega_abs: jax.Array       # (nq, μ, μ)     f64,  max(Re Ω_q, 0)
    B_mask: jax.Array          # (nq, μ, μ)     bool, B_mask_raw & valid
    n_total_modes: jax.Array   # scalar int64
    n_invalid: jax.Array       # scalar int64


@jax.jit
def _prepare_sigma_state(
    enk_full: jax.Array,
    occ_full: jax.Array,
    B_q: jax.Array,
    Omega_q: jax.Array,
    valid_mask_q: jax.Array,
    use_midgap: jax.Array,
) -> _SigmaPhysicsState:
    """Derive Fermi level + derived energy/PPM arrays in one fused trace.

    Replaces ~9 eager jnp ops previously emitted at trace time by the sigma
    driver.  ``use_midgap`` is a traced bool scalar; the caller passes
    ``jnp.asarray(fermi_reference == 'midgap')``.  ``valid_mask_q`` is always
    a real bool array (the caller substitutes ``jnp.ones_like(...)`` when
    no mask is available), so the helper doesn't branch on None.
    """
    occ_mask = occ_full > 0.5
    unocc_mask = ~occ_mask

    vbm = jnp.max(jnp.where(occ_mask, enk_full, -1.0e30))
    cbm = jnp.min(jnp.where(unocc_mask, enk_full, 1.0e30))
    has_unocc = jnp.any(unocc_mask)
    midgap_candidate = jnp.where(has_unocc, 0.5 * (vbm + cbm), vbm)
    efermi = jnp.where(use_midgap, midgap_candidate, vbm)

    E_cond = jnp.maximum(enk_full - efermi, 0.0)
    H_val = jnp.maximum(efermi - enk_full, 0.0)

    Omega_abs = jnp.maximum(jnp.real(Omega_q), 0.0).astype(jnp.float64)
    B_corr = jnp.asarray(B_q, dtype=jnp.complex128)
    B_mask_raw = Omega_abs > 1.0e-14
    valid = jnp.asarray(valid_mask_q, dtype=bool)
    B_mask = B_mask_raw & valid
    invalid_mask = B_mask_raw & (~valid)

    return _SigmaPhysicsState(
        efermi=efermi,
        E_cond=E_cond, H_val=H_val,
        cond_mask=unocc_mask, val_mask=occ_mask,
        B_corr=B_corr, Omega_abs=Omega_abs, B_mask=B_mask,
        n_total_modes=jnp.sum(B_mask_raw, dtype=jnp.int64),
        n_invalid=jnp.sum(invalid_mask, dtype=jnp.int64),
    )


def _combine_coeff_with_sigma_tau(
    coeff_re: jax.Array,
    coeff_im: jax.Array,
    sigma_tau_kij_re: jax.Array,
    sigma_tau_kij_im: jax.Array,
    project_code: jax.Array,
) -> jax.Array:
    """Multiply ω-kernel coefficient by σ^τ, keeping only the physical piece.

    σ^τ is carried as a real/imag pair because the crossing window's HGL
    quadrature needs only Im[ coeff·σ^τ ] — carrying complex σ^τ through
    the FFT pipeline would double memory for no benefit.  The window sets
    ``project_code`` via ``_SigmaWindow.project``:

        code=0 ("full")  Laplace window (stripe, slab, single) — keep the
                         full complex product  (coeff_re + i·coeff_im) · (σ_re + i·σ_im).
        code=1 ("imag")  Crossing window      — keep only  Im[coeff·σ]
                                              = coeff_re·σ_im + coeff_im·σ_re.

    The historical "real" code path (Re[coeff·σ]) is unused by every current
    window builder.  lax.switch is retained with a 2-way dispatch so that
    the generated HLO matches the previous "full" / "imag" lowering exactly
    and no minimax-table consumer gets a silent behavior change.
    """

    def _full(_):
        sigma_full = sigma_tau_kij_re[None, ...] + 1j * sigma_tau_kij_im[None, ...]
        return (coeff_re + 1j * coeff_im) * sigma_full

    def _imag(_):
        return coeff_re * sigma_tau_kij_im[None, ...] + coeff_im * sigma_tau_kij_re[None, ...]

    return jax.lax.switch(project_code, (_full, _imag), operand=None)


@jax.jit
def _project_tau_onto_omega(
    sigma_tau_kij_re: jax.Array,
    sigma_tau_kij_im: jax.Array,
    omega_vec: jax.Array,
    t_node: jax.Array,
    alpha_eff: jax.Array,
    omega_sign: jax.Array,
    pref: jax.Array,
    project_code: jax.Array,
) -> jax.Array:
    """Apply ω-kernel exp(i·ω_sign·ω·t_node) and project onto σ channels.

    Returns the single-tau contribution at every ω in ``omega_vec``:
        contrib[ω, k, i, j] = pref · α_eff · exp(i·sign·ω·t) · P(σ_re, σ_im)

    where P selects {full, imag} via ``project_code`` — see
    ``_combine_coeff_with_sigma_tau`` for why σ^τ is kept as a (re, im) pair.
    Callers either accumulate the result on-GPU (+=) or write it to disk —
    this kernel is agnostic to the consumer.
    """
    omega_kernel = jnp.exp(1j * omega_sign * omega_vec * t_node)
    coeff = alpha_eff * omega_kernel
    coeff_re = jnp.real(coeff)[:, None, None, None]
    coeff_im = jnp.imag(coeff)[:, None, None, None]
    contrib = _combine_coeff_with_sigma_tau(
        coeff_re, coeff_im, sigma_tau_kij_re, sigma_tau_kij_im, project_code
    )
    return pref * contrib.astype(jnp.complex128)


@jax.jit
def _accumulate_tau_into_window(
    acc_win: jax.Array,
    sigma_tau_kij_re: jax.Array,
    sigma_tau_kij_im: jax.Array,
    omega_vec: jax.Array,
    t_node: jax.Array,
    alpha_eff: jax.Array,
    omega_sign: jax.Array,
    pref: jax.Array,
    project_code: jax.Array,
) -> jax.Array:
    """Thin adder: ``acc_win + _project_tau_onto_omega(...)``, donated for in-place reuse."""
    return acc_win + _project_tau_onto_omega(
        sigma_tau_kij_re, sigma_tau_kij_im, omega_vec,
        t_node, alpha_eff, omega_sign, pref, project_code,
    )


def _make_project_ri_reduce_scatter(mesh_xy: Mesh) -> Callable[..., jax.Array]:
    """Build a shard_map'd ψ* σ ψ that reduce-scatters the output.

    Drop-in replacement for ``projection_kernel.project_ri`` at the tail of
    ``_sigma_channel_pipeline``.  Preserves the math exactly:

        Σ_mn(k) = Σ_{s, μ} Σ_{s', μ'}  ψ*_m(k, s, μ) · σ(k, s, μ, s', μ')
                                        · ψ_n(k, s', μ')

    Input sharding (global → per-rank):
        ψ_xr  P(None, None, None, 'x')       (nk, m, s, μ_X)
        σ     P(None, None, 'x', None, 'y')  (nk, s, μ_X, s', μ_Y)
        ψ_yn  P(None, None, 'y', None)       (nk, s', μ_Y, n)

    Output sharding:
        P(None, None, 'x', 'y')              (c=2, nk, m_X, n_Y)

    Comms inside:  the two implicit psums of the original einsum
        psum(x)       over the μ_X contraction axis
        psum(y)       over the μ_Y contraction axis
    become:
        psum_scatter(x, scatter_dim=m)   — reduces μ_X AND scatters m on x
        psum_scatter(y, scatter_dim=n)   — reduces μ_Y AND scatters n on y

    Same NCCL byte volume as the original pair of psums (on-ring LL128), but
    the output is sharded (m_X, n_Y) so every downstream coeff·σ multiply
    stays local — which is the whole point.

    Requires m % p_x == 0 and n % p_y == 0.  Padding at the caller is the
    cleanest place to handle non-divisibility (TODO when we hit that).
    """
    from jax.experimental.shard_map import shard_map

    in_specs = (
        P(None, None, None, 'x'),        # psi_xr  : (nk, m, s, μ_X)
        P(None, None, 'x', None, 'y'),   # sigma_k : (nk, s, μ_X, s', μ_Y)
        P(None, None, 'y', None),        # psi_yn  : (nk, s', μ_Y, n)
    )
    out_specs = P(None, None, 'x', 'y')  # sigma_ri : (c=2, nk, m_X, n_Y)

    def _local(psi_xr_local, sigma_k_local, psi_yn_local):
        # Stack re/im channels.  Each rank sees only its local (μ_X, μ_Y) tile
        # of sigma_k, so the stack is cheap local work.
        sigma_ri = jnp.stack(
            (jnp.real(sigma_k_local), jnp.imag(sigma_k_local)), axis=0)

        # First einsum: contract local s and local μ_X ("x" axis) slot.
        # Each x-rank computes a partial over its own μ_X chunk.  m and μ_Y
        # are still full on this rank.
        # Shapes: 'kmsx' × 'cksxty' -> 'ckmty'
        #   psi_xr_local: (nk, m, s, μ/p_x)
        #   sigma_ri:     (c, nk, s, μ/p_x, s', μ/p_y)
        #   →             (c, nk, m, s', μ/p_y)   — partial along μ_X
        left_partial = jnp.einsum(
            'kmsx,cksxty->ckmty', jnp.conj(psi_xr_local), sigma_ri,
            optimize=True)

        # psum_scatter(x, scatter_dim=m=2): sum over x, scatter m across x.
        # Output shape per-rank: (c, nk, m/p_x, s', μ/p_y).
        left_rs = jax.lax.psum_scatter(
            left_partial, 'x', scatter_dimension=2, tiled=True)

        # Second einsum: contract local s' and local μ_Y.  n is still full.
        # Shapes: 'ckmty' × 'ktyn' -> 'ckmn'
        #   left_rs:      (c, nk, m/p_x, s', μ/p_y)
        #   psi_yn_local: (nk, s', μ/p_y, n)
        #   →             (c, nk, m/p_x, n)  — partial along μ_Y
        result_partial = jnp.einsum(
            'ckmty,ktyn->ckmn', left_rs, psi_yn_local,
            optimize=True)

        # psum_scatter(y, scatter_dim=n=3): sum over y, scatter n across y.
        # Output per-rank: (c, nk, m/p_x, n/p_y).  This is what the global
        # out_spec P(None, None, 'x', 'y') demands.
        return jax.lax.psum_scatter(
            result_partial, 'y', scatter_dimension=3, tiled=True
        ).astype(jnp.complex128)

    return shard_map(_local, mesh=mesh_xy,
                     in_specs=in_specs, out_specs=out_specs,
                     check_rep=False)


def _get_sigma_channel_pipeline(
    *,
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    nk_tot: int,
    bispinor: bool,
) -> Callable[..., jax.Array]:
    """Return a jit-compatible sigma-channel pipeline with device-local FFTs.

    The tail project (ψ* σ ψ → Σ_mn) uses the reduce-scatter variant so the
    emitted σ^τ is sharded (m_X, n_Y) — matching what
    _ReduceScatterGpuAccumulator expects without any downstream reshuffle.
    """

    pipeline_key = (id(mesh_xy), nkx, nky, nkz, nk_tot, bispinor)
    if pipeline_key in _sigma_channel_pipeline_cache:
        return _sigma_channel_pipeline_cache[pipeline_key]

    from common.fft_helpers import (
        make_jittable_local_fftn_3d,
        make_jittable_local_ifftn_3d,
    )

    w_isdf._ensure_compilation_cache()
    _G_spec = P(None, None, None, None, 'x', None, 'y')
    _V_spec = P(None, None, None, 'x', 'y')
    _G_shard = NamedSharding(mesh_xy, _G_spec)
    _V_shard = NamedSharding(mesh_xy, _V_spec)
    _G_ifftn = make_jittable_local_ifftn_3d(mesh_xy, _G_spec, _G_spec, norm='ortho', axes=(0, 1, 2))
    _G_fftn = make_jittable_local_fftn_3d(mesh_xy, _G_spec, _G_spec, norm='ortho', axes=(0, 1, 2))
    _V_ifftn = make_jittable_local_ifftn_3d(mesh_xy, _V_spec, _V_spec, norm='ortho', axes=(0, 1, 2))
    nk = nkx * nky * nkz
    inv_sqrt_nk = -1.0 / np.sqrt(float(nk_tot))

    def _fft_flat_G(x_k, fft_fn):
        x_3d = jax.lax.with_sharding_constraint(x_k.reshape(nkx, nky, nkz, *x_k.shape[1:]), _G_shard)
        return fft_fn(x_3d).reshape(nk, *x_k.shape[1:])

    def _fft_flat_V(x_k):
        x_3d = jax.lax.with_sharding_constraint(x_k.reshape(nkx, nky, nkz, *x_k.shape[1:]), _V_shard)
        return _V_ifftn(x_3d).reshape(nk, *x_k.shape[1:])

    from .greens_function_kernel import build_G as _build_G

    _project_ri_rs = _make_project_ri_reduce_scatter(mesh_xy)

    @jax.jit
    def _sigma_channel_pipeline(
        psi_coh_rmuT_X, psi_coh_rmu_Y, psi_proj_rmu_X, psi_proj_rmuT_Y,
        Gij, W_q,
    ):
        """Σ_kij = project_rs[ FFT[ G(R) · W(R) / √Nk ] ].  All flat-k.

        W_q is (nq, μ, μ) flat-q — same layout as all other flat-k arrays.
        Output (Σ_ri) emerges (m_X, n_Y)-sharded from the final shard_map.
        """
        G_k = _build_G(psi_coh_rmuT_X, psi_coh_rmu_Y, Gij=Gij)
        G_R = _fft_flat_G(G_k, _G_ifftn)
        V_R = _fft_flat_V(W_q)[:, None, :, None, :]  # (nk,1,μ,1,μ) broadcast to G shape
        sigma_k = _fft_flat_G(G_R * V_R * inv_sqrt_nk, _G_fftn)
        return _project_ri_rs(psi_proj_rmu_X, sigma_k, psi_proj_rmuT_Y)

    _sigma_channel_pipeline_cache[pipeline_key] = _sigma_channel_pipeline
    return _sigma_channel_pipeline


def _get_sigma_tau_channel_kernel(
    *,
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    nk_tot: int,
    bispinor: bool,
) -> Callable[..., jax.Array]:
    """Return a cached tau-node sigma builder with jittable local FFTs."""

    cache_key = (id(mesh_xy), nkx, nky, nkz, nk_tot, bispinor)
    if cache_key in _sigma_tau_channel_kernel_cache:
        return _sigma_tau_channel_kernel_cache[cache_key]

    w_isdf._ensure_compilation_cache()
    q_mu_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    sigma_channel_pipeline = _get_sigma_channel_pipeline(
        mesh_xy=mesh_xy, nkx=nkx, nky=nky, nkz=nkz,
        nk_tot=nk_tot, bispinor=bispinor,
    )

    @jax.jit
    def _build_tau_operands(
        E_A, mask_A, B_q, Omega_q, mask_B,
        E_ref_A, E_ref_B, t_node, eye_nb,
    ):
        phase_A = jnp.exp(-1j * (E_A - E_ref_A) * t_node)
        weights_kn = jnp.where(mask_A, phase_A, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        Gij = eye_nb[None, :, :] * weights_kn[:, :, None]

        phase_B = jnp.exp(-1j * (Omega_q - E_ref_B) * t_node)
        W_t_q = jnp.where(mask_B, B_q * phase_B, jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
        W_t_q = jax.lax.with_sharding_constraint(W_t_q, q_mu_shard)
        return Gij, W_t_q

    @jax.jit
    def _tau_channel_step(
        psi_coh_rmuT_X, psi_coh_rmu_Y,
        psi_proj_rmu_X, psi_proj_rmuT_Y,
        E_A, mask_A, B_q, Omega_q, mask_B,
        E_ref_A, E_ref_B, t_node, eye_nb,
    ):
        Gij, W_t_q = _build_tau_operands(
            E_A, mask_A, B_q, Omega_q, mask_B,
            E_ref_A, E_ref_B, t_node, eye_nb,
        )
        return sigma_channel_pipeline(
            psi_coh_rmuT_X, psi_coh_rmu_Y,
            psi_proj_rmu_X, psi_proj_rmuT_Y,
            Gij, W_t_q,
        )

    _sigma_tau_channel_kernel_cache[cache_key] = _tau_channel_step
    return _tau_channel_step


# ---------------------------------------------------------------------------
#  PPM construction
# ---------------------------------------------------------------------------

def fit_gn_ppm(
    W0_q: jax.Array,
    Wiwp_q: jax.Array,
    V_q: jax.Array,
    omega_p: float,
    mesh_xy: Mesh,
    *,
    fallback_omega: float = 2.0,
    n_nodes_static: int = 0,
    print_fn=None,
) -> PPMBuildResult:
    """Fit GN-PPM pole parameters from precomputed W(0) and W(iωp).

    All input arrays are flat-q (nq, μ, μ).  Returns PPMBuildResult with
    B_q, Omega_q, valid_mask_q sharded as P(None, 'x', 'y').
    """
    import time as _t
    omega_p = float(omega_p)
    t0 = _t.perf_counter()

    Wc0_q = W0_q - V_q
    Wci_q = Wiwp_q - V_q
    omega_qmunu, b_qmunu, valid_qmunu, unfulfilled = fit_gn_ppm_from_wc_pair(
        Wc0_q, Wci_q, 1j * complex(omega_p), fallback_omega=float(fallback_omega))

    q_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    Omega = jax.lax.with_sharding_constraint(jnp.asarray(omega_qmunu), q_shard)
    B = jax.lax.with_sharding_constraint(jnp.asarray(b_qmunu), q_shard)
    valid_mask = jax.lax.with_sharding_constraint(jnp.asarray(valid_qmunu), q_shard)
    t1 = _t.perf_counter()

    if print_fn is not None:
        print_fn(f"  GN-PPM fit: {t1-t0:.2f}s, ωp={omega_p:.4f} Ry, "
                 f"unfulfilled={100.0 * unfulfilled:.2f}%")

    return PPMBuildResult(
        omega_p=omega_p,
        W0_q=W0_q,
        Wiwp_q=Wiwp_q,
        B_q=B,
        Omega_q=Omega,
        valid_mask_q=valid_mask,
        unfulfilled_fraction=unfulfilled,
        n_nodes_static=n_nodes_static,
    )



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
            t_nodes=np.asarray(-1j * q.tau, dtype=np.complex128),
            alpha=np.asarray(q.alpha, dtype=np.float64),
            mask_A=np.asarray(base_mask_A, dtype=bool),
            mask_B=None,
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
            t_nodes = np.asarray(q_cross.tau / xi, dtype=np.complex128)
            alpha = np.asarray(q_cross.alpha / xi, dtype=np.float64)
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
            t_nodes = np.asarray(-1j * q.tau, dtype=np.complex128)
            alpha = np.asarray(q.alpha, dtype=np.float64)
            project = "full"
            prefactor = +1.0

        windows.append(
            _SigmaWindow(
                name=name,
                t_nodes=t_nodes,
                alpha=alpha,
                mask_A=np.asarray(mA, dtype=bool),
                mask_B=None,
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
#  Sigma accumulators — one interface, two strategies.
#
#  The tau loop doesn't care whether its outputs land in a GPU buffer or on
#  disk; it just needs to add per-tau contributions and knows when a window
#  boundary falls.  The two implementations differ only in what "add" means.
# ---------------------------------------------------------------------------

class _SigmaAccumulator:
    """Minimal protocol used by _integrate_tau_windows_for_branch.

    Lifecycle per branch:
        acc.begin_window()
        for each tau:
            acc.add_tau(σ_re, σ_im, ω_vec, t_node, α_eff, ω_sign, pref, code)
        acc.end_window()
    At branch end the caller calls ``acc.finalize()`` — returns the on-GPU
    (n_ω, nk, nb, nb) tensor (buffered path) or None (stream path).
    """
    def begin_window(self) -> None: ...
    def add_tau(self, *args, **kwargs) -> None: ...
    def end_window(self) -> None: ...
    def finalize(self) -> jax.Array | None: ...


class _BufferedGpuAccumulator(_SigmaAccumulator):
    """Accumulate Σ^c(ω) on GPU into a single (n_ω, nk, nb, nb) tensor.

    Per-window accumulation is kept separate so that the tau loop donates
    its running buffer only once per window (helps XLA reuse the slot).
    """
    def __init__(self, shape: tuple[int, int, int, int]):
        self.total = jnp.zeros(shape, dtype=jnp.complex128)
        self._win_acc: jax.Array | None = None

    def begin_window(self) -> None:
        self._win_acc = jnp.zeros_like(self.total)

    def add_tau(self, sigma_re, sigma_im, omega_vec,
                t_node_j, alpha_eff_j, omega_sign_j, pref_j, project_code_j,
                *, omega_global_idx=None):
        assert self._win_acc is not None
        self._win_acc = _accumulate_tau_into_window(
            self._win_acc, sigma_re, sigma_im, omega_vec,
            t_node_j, alpha_eff_j, omega_sign_j, pref_j, project_code_j,
        )

    def end_window(self) -> None:
        assert self._win_acc is not None
        self.total = self.total + self._win_acc
        self._win_acc = None

    def finalize(self) -> jax.Array:
        return self.total


# ---------------------------------------------------------------------------
#  Reduce-scatter accumulator — scaffolding for the Σ_c(ω, k, m_X, n_Y) path.
#
#  Design intent (scaling target: n_rmu ≈ 10·n_b, n_k × n_b² ~ single-GPU HBM):
#
#      Σ_c(ω, k, m, n) lives on-GPU sharded (m on mesh.x, n on mesh.y) so
#      every rank holds only (n_ω, n_k, n_b/p, n_b/p).  This is ~100× smaller
#      than Σ_μν(k, q), which stays block-sharded (μ_X, ν_Y) upstream and
#      never materializes replicated.  There is thus HBM headroom for many
#      τ contributions to stack before a flush — we exploit that here.
#
#  Communication algorithm (the one to implement; not yet wired):
#
#      σ^τ   = project_ri(ψ, σ_k, ψ)           # currently: einsums auto-psum
#                                               # over x AND y, σ^τ replicated
#
#      σ^τ_sharded = shard_map(local_project)  # TODO — replaces project_ri
#          partial[c, k, m, n] = local einsum on rank's (μ_X, ν_Y) block
#          partial = psum_scatter(partial, 'x', m_axis, tiled=True)
#          partial = psum_scatter(partial, 'y', n_axis, tiled=True)
#                                               # → (c, k, m/p, n/p) per rank
#
#      Σ[ω, k, m_X, n_Y] += coeff[ω, None, None, None] * σ^τ_sharded[None, k, m_X, n_Y]
#
#      For τ-batching (also not yet wired): stack n_batch tau nodes into a
#      leading axis and use jax.lax.scan over the batch inside a single jit.
#
#  What this class does TODAY:
#
#      * Allocates Σ with the target sharding P(None, None, 'x', 'y').
#      * Places an explicit with_sharding_constraint on σ^τ before the add
#        — XLA re-plans the layout of the replicated σ^τ into the sharded
#        shape, but no reduce-scatter happens here yet (the upstream
#        project_ri still full-psums).
#      * Accumulates via _accumulate_tau_into_window which already broadcasts
#        the ω-coefficient along replicated ω, with sharded (m, n) downstream
#        — the arithmetic is local per-rank once σ^τ is sharded.
#
#  What still needs doing for real comm savings:
#
#      (1) Replace _sigma_channel_pipeline's final project_ri call with a
#          shard_map'd variant that emits σ^τ already sharded (m_X, n_Y).
#          That is the only change that actually drops bytes on the wire.
#      (2) Add m-chunking at this accumulator's add_tau entrance so partial
#          is (m_chunk, n_full) before RS rather than (m_full, n_full).
#          Default chunk = 1 output tile (m_chunk = m/p).
#      (3) Stage many τ on GPU before flush in the FFI-backed variant
#          (_ReduceScatterFfiAccumulator), using SlabIO.write_slab for the
#          collective parallel-HDF5 flush at window boundaries.
# ---------------------------------------------------------------------------


class _ReduceScatterGpuAccumulator(_SigmaAccumulator):
    """Σ_c(ω, k, m, n) sharded (m_X, n_Y) on GPU — scaffolding.

    Same external interface as _BufferedGpuAccumulator; differs in that:

        * its running buffers live at NamedSharding(mesh, P(None, None, 'x', 'y'))
        * σ^τ is restamped via with_sharding_constraint before the add,
          so downstream arithmetic on each rank touches only the local shard
        * the add closure is factored so a future lax.scan over a τ-batch
          can replace the per-tau Python call without changing the caller

    At n_b = 80 on a 2×2 mesh the per-rank buffer is 40×40×n_ω×n_k ≈ 1.2 MiB
    vs the replicated 4.8 MiB — negligible here, but the sharding-preserving
    arithmetic is the load-bearing piece at 1500+ bands.
    """

    def __init__(self, shape: tuple[int, int, int, int], mesh_xy: Mesh):
        self._sharding = NamedSharding(mesh_xy, P(None, None, 'x', 'y'))
        self._tau_sharding = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        # Allocate already-sharded.  jax.lax.with_sharding_constraint requires
        # a trace context; jax.device_put is the equivalent for eager setup.
        self.total = jax.device_put(
            jnp.zeros(shape, dtype=jnp.complex128), self._sharding)
        self._win_acc: jax.Array | None = None

    def begin_window(self) -> None:
        self._win_acc = jax.device_put(
            jnp.zeros_like(self.total), self._sharding)

    def add_tau(self, sigma_re, sigma_im, omega_vec,
                t_node_j, alpha_eff_j, omega_sign_j, pref_j, project_code_j,
                *, omega_global_idx=None):
        assert self._win_acc is not None
        # σ^τ arrives already (k, m_X, n_Y)-sharded from the shard_map'd
        # project inside _sigma_channel_pipeline.  The ω-kernel multiply and
        # add therefore touch only each rank's local (m/p_x, n/p_y) block.
        self._win_acc = _accumulate_tau_into_window(
            self._win_acc, sigma_re, sigma_im, omega_vec,
            t_node_j, alpha_eff_j, omega_sign_j, pref_j, project_code_j,
        )
        # Pin the running buffer's sharding so XLA doesn't replicate it
        # between adds.
        self._win_acc = jax.lax.with_sharding_constraint(
            self._win_acc, self._sharding)

    def end_window(self) -> None:
        assert self._win_acc is not None
        self.total = jax.lax.with_sharding_constraint(
            self.total + self._win_acc, self._sharding)
        self._win_acc = None

    def finalize(self) -> jax.Array:
        return self.total


class _StreamedH5Accumulator(_SigmaAccumulator):
    """Project each tau contribution in ω-batches and hand to a writer callable.

    The writer is expected to read-modify-write the backing HDF5 dataset;
    this class is agnostic to the storage (rank-0 h5py, SlabIO, …).

    Note on the FFI flush path (future work, comment-only here): a third
    accumulator — _ReduceScatterFfiAccumulator — would keep the running Σ
    sharded (m_X, n_Y) on GPU like _ReduceScatterGpuAccumulator, stack many
    τ contributions per window without flushing, and at end_window() issue
    a single collective parallel-HDF5 write via SlabIO.write_slab against
    a pre-opened zarr-style (n_ω, n_k, m, n) dataset.  This removes the
    per-τ read-modify-write roundtrip that makes _StreamedH5Accumulator
    catastrophic at multi-process scale.  Implement when the upstream
    reduce-scatter project lands (without it, there's no point — σ^τ is
    still gathered on every rank).
    """
    def __init__(self, writer: Callable[[np.ndarray, jax.Array], None],
                 *, omega_global_idx: np.ndarray, omega_batch_size: int):
        self._writer = writer
        self._omega_global_idx = np.asarray(omega_global_idx, dtype=np.int64)
        self._batch = int(max(1, omega_batch_size))

    def begin_window(self) -> None:
        pass

    def add_tau(self, sigma_re, sigma_im, omega_vec,
                t_node_j, alpha_eff_j, omega_sign_j, pref_j, project_code_j,
                *, omega_global_idx=None):
        n_omega = int(omega_vec.shape[0])
        for ibeg in range(0, n_omega, self._batch):
            iend = min(ibeg + self._batch, n_omega)
            batch_proj = _project_tau_onto_omega(
                sigma_re, sigma_im, omega_vec[ibeg:iend],
                t_node_j, alpha_eff_j, omega_sign_j,
                pref_j, project_code_j,
            )
            self._writer(self._omega_global_idx[ibeg:iend], batch_proj)

    def end_window(self) -> None:
        pass

    def finalize(self) -> None:
        return None


# ---------------------------------------------------------------------------
#  Sigma convolution — split into (a) host-side window construction and
#  (b) device-side tau loop.  The two halves have no shared state beyond
#  the window list itself, so they're easy to test independently and
#  trivial to reuse when the tau integration changes.
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
            f"{int(win.alpha.shape[0])} nodes, err<{target_error:.0e}, "
            f"E_A=[{float(np.min(A_vals)):.4f}, {float(np.max(A_vals)):.4f}] Ry, "
            f"project={win.project}"
        )
    return windows


def _integrate_tau_windows_for_branch(
    *,
    windows: list[_SigmaWindow],
    omega_vec: jax.Array,
    accumulator: _SigmaAccumulator,
    E_A: jax.Array,
    B_q: jax.Array,
    Omega_q: jax.Array,
    base_mask_B: jax.Array,
    psi_coh_rmuT_X: jax.Array,
    psi_coh_rmu_Y: jax.Array,
    psi_proj_rmu_X: jax.Array,
    psi_proj_rmuT_Y: jax.Array,
    eye_nb: jax.Array,
    tau_channel_step: Callable[..., jax.Array],
    mesh_xy: Mesh,
    omega_sign_flip: int,
    scale: float,
    log_tag: str,
    print_fn,
) -> None:
    """Walk windows × tau nodes, feed each (σ_re, σ_im) into ``accumulator``.

    Whether the result lands on GPU or disk is the accumulator's concern,
    not this loop's — see _BufferedGpuAccumulator / _StreamedH5Accumulator.

    Future τ-batching hook (NOT yet wired — left for the shard_map refactor):

        The per-τ python loop below serializes on a block_until_ready after
        each tau_channel_step call, so XLA cannot fuse across τ.  With the
        reduce-scatter upstream in place, tau_channel_step's output becomes
        small enough that we can:

            t_nodes_j = jnp.asarray(win.t_nodes, dtype=jnp.complex128)   # (n_τ,)
            alphas_j  = jnp.asarray(win.alpha,   dtype=jnp.float64)      # (n_τ,)

            @jax.jit
            def _scan_body(acc, tau_ctx):
                t_j, a_j = tau_ctx
                sigma_re, sigma_im = tau_channel_step(..., t_j, ...)
                acc = _accumulate_tau_into_window(
                    acc, sigma_re, sigma_im, ω_vec, t_j, a_j, ...)
                return acc, None

            win_acc, _ = jax.lax.scan(_scan_body, zeros, (t_nodes_j, alphas_j))

        That collapses N_τ jit dispatches into 1 compile, one NCCL fence per
        window instead of per τ, and lets XLA schedule the D2H overlap of
        subsequent windows.  Only safe once σ^τ is shard-scatter'd — today a
        scan over replicated σ^τ would blow up HBM because all τ contribs
        would coexist during the scan trace.

    m-chunking hook for the accumulator (also NOT wired): add_tau's
    ``omega_global_idx`` slot is currently unused by _ReduceScatterGpuAccumulator
    but is a natural place to pass an m-chunk selector; default (None) is
    one m-strip per x-rank (= m/p).  Wire when upstream RS lands.
    """
    from common.progress import LoopProgress

    branch_label = log_tag if log_tag else "sigma"
    total_tau_nodes = sum(int(win.alpha.shape[0]) for win in windows)
    progress = LoopProgress(
        total_tau_nodes, print_fn, title=f"sigma[{branch_label}]",
        item_name="tau node", max_updates=10)

    with jax_profile.annotation(f"sigma_branch[{branch_label}]"):
        for win_idx, win in enumerate(windows):
            with jax_profile.step_annotation(
                "sigma_window", step_num=win_idx,
                detail=f"{branch_label}:{win.name}:n{int(win.alpha.shape[0])}",
            ):
                mask_A = jnp.asarray(win.mask_A)
                mask_B = _materialize_window_mask_B(
                    win, base_mask_B=base_mask_B, Omega_q=Omega_q)
                E_ref_A_j = jnp.asarray(win.E_ref_A, dtype=jnp.float64)
                E_ref_B_j = jnp.asarray(win.E_ref_B, dtype=jnp.float64)
                if win.project not in ("full", "imag"):
                    raise ValueError(
                        f"Unknown window projection {win.project!r}; "
                        f"expected 'full' (Laplace) or 'imag' (crossing).")
                project_code_j = jnp.asarray(
                    {"full": 0, "imag": 1}[win.project], dtype=jnp.int32)

                accumulator.begin_window()
                for t_node, alpha_node in zip(win.t_nodes, win.alpha):
                    t_node_j = jnp.asarray(t_node, dtype=jnp.complex128)
                    with mesh_xy:
                        # σ^τ is returned as a real/imag pair rather than complex:
                        # the crossing window's HGL quadrature keeps only Im[coeff·σ],
                        # so carrying complex through the FFT stack would double
                        # HBM and collective traffic for no benefit.  See
                        # _combine_coeff_with_sigma_tau for the recombination.
                        #
                        # NB: sigma_tau_ri is (2, nk, m_X, n_Y) sharded under the
                        # reduce-scatter project — tuple-unpacking (a, b = arr)
                        # would assert is_fully_addressable in multi-process mode,
                        # so index explicitly.
                        sigma_tau_ri = tau_channel_step(
                            psi_coh_rmuT_X, psi_coh_rmu_Y,
                            psi_proj_rmu_X, psi_proj_rmuT_Y,
                            E_A, mask_A, B_q, Omega_q, mask_B,
                            E_ref_A_j, E_ref_B_j, t_node_j, eye_nb,
                        )
                        sigma_tau_kij_re = sigma_tau_ri[0]
                        sigma_tau_kij_im = sigma_tau_ri[1]
                    sigma_tau_kij_re.block_until_ready()
                    progress.step()

                    # α_eff = α(τ) · exp[-i·(E_ref_A + E_ref_B)·τ]  — the minimax
                    # quadrature weight for this τ node, phase-shifted so the
                    # Laplace transform kernel sees E_A and Ω_q as ≥ 0 arguments.
                    alpha_eff = complex(alpha_node) * np.exp(
                        -1j * (win.E_ref_A + win.E_ref_B) * t_node)
                    alpha_eff_j = jnp.asarray(alpha_eff, dtype=jnp.complex128)
                    omega_sign_j = jnp.asarray(
                        float(win.omega_sign) * float(omega_sign_flip), dtype=jnp.float64)
                    pref_j = jnp.asarray(win.prefactor * scale, dtype=jnp.float64)

                    accumulator.add_tau(
                        sigma_tau_kij_re, sigma_tau_kij_im, omega_vec,
                        t_node_j, alpha_eff_j, omega_sign_j, pref_j, project_code_j,
                    )

                accumulator.end_window()

    progress.finish()


def _run_sigma_branch(
    *,
    omega_nonneg_ry: np.ndarray,
    omega_global_idx: np.ndarray,
    E_A: jax.Array,
    base_mask_A: jax.Array,
    B_q: jax.Array,
    Omega_q: jax.Array,
    base_mask_B: jax.Array,
    kernel_sign: int,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_nodes: int,
    crossing_eps_q: float,
    crossing_max_nodes: int,
    wfns,
    mesh_xy: Mesh,
    meta,
    omega_sign_flip: int = 1,
    log_tag: str = "",
    print_fn=print,
    omega_batch_size: int = 4,
    stream_writer: Callable[[np.ndarray, jax.Array], None] | None = None,
    scale: float = 1.0,
    use_shipped_minimax_tables: bool = True,
) -> tuple[jax.Array, list[_SigmaWindow]]:
    """Orchestrator for one branch (cond or val × pos or neg ω half).

    Reads as a physics outline:
        windows = _build_windows_for_branch(...)          # host
        acc     = _integrate_tau_windows_for_branch(...)  # device
    """
    omega_nonneg_ry = np.asarray(omega_nonneg_ry, dtype=np.float64)
    n_omega = int(omega_nonneg_ry.shape[0])

    s = wfns.slices
    psi_coh_rmuT_X = wfns.xn(s.full)
    psi_coh_rmu_Y = wfns.yr(s.full)
    psi_proj_rmu_X = wfns.xr(s.sigma)
    psi_proj_rmuT_Y = wfns.yn(s.sigma)
    nk_proj = int(psi_proj_rmu_X.shape[0])
    nb_proj = int(psi_proj_rmu_X.shape[1])

    if n_omega == 0:
        return jnp.zeros((0, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), []

    windows = _build_windows_for_branch(
        omega_nonneg_ry=omega_nonneg_ry,
        E_A=E_A, base_mask_A=base_mask_A,
        Omega_q=Omega_q, base_mask_B=base_mask_B,
        kernel_sign=kernel_sign,
        regularization_width_ry=regularization_width_ry,
        edge_factor=edge_factor,
        target_error=target_error, max_nodes=max_nodes,
        crossing_eps_q=crossing_eps_q, crossing_max_nodes=crossing_max_nodes,
        use_shipped_minimax_tables=use_shipped_minimax_tables,
        log_tag=log_tag, print_fn=print_fn,
    )
    if not windows:
        return jnp.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), []

    omega_vec = jnp.asarray(omega_nonneg_ry, dtype=jnp.float64)
    eye_nb = jnp.eye(E_A.shape[1], dtype=jnp.complex128)
    tau_channel_step = _get_sigma_tau_channel_kernel(
        mesh_xy=mesh_xy,
        nkx=int(meta.nkx), nky=int(meta.nky), nkz=int(meta.nkz),
        nk_tot=int(meta.nk_tot), bispinor=bool(meta.bispinor),
    )

    if stream_writer is None:
        # Prefer the reduce-scatter-shaped accumulator: keeps the running Σ
        # (m_X, n_Y) sharded so arithmetic is local per-rank.  See the
        # _ReduceScatterGpuAccumulator module comment for what's wired today
        # (layout-only) vs what still needs the shard_map refactor upstream
        # (actual byte-level comm savings).  This is a drop-in replacement
        # at the accumulator boundary.
        accumulator: _SigmaAccumulator = _ReduceScatterGpuAccumulator(
            shape=(n_omega, nk_proj, nb_proj, nb_proj),
            mesh_xy=mesh_xy,
        )
    else:
        accumulator = _StreamedH5Accumulator(
            writer=stream_writer,
            omega_global_idx=omega_global_idx,
            omega_batch_size=int(max(1, omega_batch_size)),
        )

    _integrate_tau_windows_for_branch(
        windows=windows, omega_vec=omega_vec, accumulator=accumulator,
        E_A=E_A, B_q=B_q, Omega_q=Omega_q, base_mask_B=base_mask_B,
        psi_coh_rmuT_X=psi_coh_rmuT_X, psi_coh_rmu_Y=psi_coh_rmu_Y,
        psi_proj_rmu_X=psi_proj_rmu_X, psi_proj_rmuT_Y=psi_proj_rmuT_Y,
        eye_nb=eye_nb, tau_channel_step=tau_channel_step,
        mesh_xy=mesh_xy,
        omega_sign_flip=omega_sign_flip, scale=scale,
        log_tag=log_tag, print_fn=print_fn,
    )

    acc_total = accumulator.finalize()
    if acc_total is None:
        return jnp.zeros((0, nk_proj, nb_proj, nb_proj), dtype=jnp.complex128), windows
    return acc_total, windows


# ---------------------------------------------------------------------------
#  Top-level sigma driver
# ---------------------------------------------------------------------------

def compute_sigma_c_ppm_omega_grid(
    wfns,
    ppm,
    meta,
    mesh_xy: Mesh,
    ppm_options,
    *,
    sigma_window_quad: SigmaQuadratureConfig | None = None,
    print_fn=print,
) -> SigmaOmegaResult:
    """Compute Σ^c_kij(ω) via GN-PPM windowed minimax integration."""

    s = wfns.slices
    psi_proj_rmu_X = wfns.xr(s.sigma)
    enk_full = wfns.enk[:, s.full]
    occ_full = wfns.occ[:, s.full]
    B_q = ppm.B_q
    Omega_q = ppm.Omega_q
    valid_mask_q = getattr(ppm, 'valid_mask_q', None)
    omega_values_ry = ppm_options.omega_grid_ry

    nkx, nky, nkz = int(meta.nkx), int(meta.nky), int(meta.nkz)
    nk = int(nkx * nky * nkz)

    # Quadrature config
    if sigma_window_quad is not None:
        target_error = float(sigma_window_quad.target_error)
        max_nodes = int(sigma_window_quad.max_nodes)
        crossing_max_nodes = int(sigma_window_quad.crossing_max_nodes)
        crossing_eps_q = float(sigma_window_quad.crossing_eps_q)
        use_shipped_minimax_tables = bool(sigma_window_quad.use_shipped_tables)
    else:
        target_error, max_nodes = 1e-6, 64
        crossing_max_nodes, crossing_eps_q = 500, 1e-3
        use_shipped_minimax_tables = True

    regularization_width_ry = getattr(ppm_options, 'sigma_regularization_ry', 0.018374661087827496)
    edge_factor = getattr(ppm_options, 'sigma_edge_factor', 1.5)
    omega_batch_size = getattr(ppm_options, 'sigma_omega_batch_size', 4)
    omega_accumulation = getattr(ppm_options, 'sigma_omega_accumulation', 'auto')
    sigma_kij_h5_path = getattr(ppm_options, 'sigma_kij_h5_path', None)
    fermi_reference = getattr(ppm_options, 'fermi_reference', 'midgap')

    if nk != int(enk_full.shape[0]):
        raise ValueError(f"enk_full shape mismatch: expected first dim {nk}, got {enk_full.shape[0]}")

    omega_req = np.asarray(omega_values_ry, dtype=np.float64)
    if omega_req.ndim != 1 or omega_req.size == 0:
        raise ValueError("omega_values_ry must be a 1D non-empty array.")
    omega_batch_size = int(max(1, omega_batch_size))
    omega_accumulation = str(omega_accumulation).strip().lower()
    if omega_accumulation not in ("auto", "kij", "kij_stream"):
        raise ValueError("omega_accumulation must be one of: auto, kij, kij_stream.")

    # Split omega grid into positive and negative relative to Fermi level
    idx_pos = np.where(omega_req >= 0.0)[0]
    idx_neg = np.where(omega_req < 0.0)[0]
    omega_pos = np.asarray(omega_req[idx_pos], dtype=np.float64)
    omega_neg_abs = np.asarray(-omega_req[idx_neg], dtype=np.float64)

    # Fermi reference validation (string → traced bool for the jit)
    fermi_reference = str(fermi_reference).strip().lower()
    if fermi_reference not in ("vbm", "midgap"):
        raise ValueError("fermi_reference must be 'vbm' or 'midgap'.")

    # Derive Fermi level, energy/band masks, and PPM pole masks in one fused trace.
    # valid_mask_q=None → all-true mask at the caller so the jit sees a real array.
    if valid_mask_q is None:
        valid_mask_q = jnp.ones(Omega_q.shape, dtype=bool)
    state = _prepare_sigma_state(
        enk_full, occ_full, B_q, Omega_q, valid_mask_q,
        jnp.asarray(fermi_reference == "midgap", dtype=bool),
    )
    efermi = state.efermi
    E_cond = state.E_cond
    H_val = state.H_val
    cond_mask = state.cond_mask
    val_mask = state.val_mask
    B_corr = state.B_corr
    Omega_abs = state.Omega_abs
    B_mask = state.B_mask
    n_total_modes = int(jax.device_get(state.n_total_modes))
    n_invalid = int(jax.device_get(state.n_invalid))

    ryd2ev = 13.6056980659
    omega_step_ev = float(omega_req[1] - omega_req[0]) * ryd2ev if omega_req.size > 1 else 0.0
    print_fn(
        f"  Σc(ω) grid: "
        f"{float(np.min(omega_req)) * ryd2ev:.3f}..{float(np.max(omega_req)) * ryd2ev:.3f} eV, "
        f"Nω={omega_req.size}, Δω={omega_step_ev:.3f} eV, "
        f"ξ={float(regularization_width_ry) * ryd2ev:.3f} eV"
    )
    if n_invalid:
        print_fn(
            f"  GN invalid modes: {n_invalid}/{n_total_modes} "
            f"({100.0 * n_invalid / max(n_total_modes, 1):.2f}%)"
        )

    # Accumulation mode
    nk_proj = int(psi_proj_rmu_X.shape[0])
    nb_proj = int(psi_proj_rmu_X.shape[1])
    kij_bytes = float(omega_req.size * nk_proj * nb_proj * nb_proj * 16)
    use_kij_accum = omega_accumulation == "kij"
    use_kij_stream = omega_accumulation == "kij_stream"
    if omega_accumulation == "auto":
        use_kij_accum = (sigma_kij_h5_path is None) and (kij_bytes <= 0.5 * 1024**3)
        use_kij_stream = not use_kij_accum
    if use_kij_stream and not sigma_kij_h5_path:
        use_kij_stream = False
        use_kij_accum = True

    n_omega = int(omega_req.size)

    # Stream mode is a fine-grained read-modify-write accumulator that
    # fires once per (tau_node × omega_batch); at multi-process scale
    # every call is a collective MPI-IO or rank-0 h5py round-trip, and
    # there are hundreds of them — so it's a real perf problem under
    # the current structure.  Until we refactor to accumulate on GPU
    # and stream out at branch granularity, fall back to the accum
    # path in multi-process runs.
    use_ffi_io = bool(getattr(ppm_options, 'use_ffi_io', False))
    if use_kij_stream and jax.process_count() != 1:
        use_kij_stream = False
        use_kij_accum = True

    sigma_kij_host = None if use_kij_stream else np.zeros((n_omega, nk_proj, nb_proj, nb_proj), dtype=np.complex128)

    # Single-process stream-mode file setup.  The accumulator pattern
    # itself is unchanged from pre-SlabIO (rank-0 h5py); the final
    # sigma_mnk.h5 copy-over is already migrated via
    # write_sigma_omega_h5 in gw_jax.py.
    kij_stream_path = None
    h5_kij = None
    dset_sigma_kij = None
    if use_kij_stream and sigma_kij_h5_path and jax.process_index() == 0:
        kij_stream_path = str(sigma_kij_h5_path)
        kij_dir = os.path.dirname(os.path.abspath(kij_stream_path))
        if kij_dir:
            os.makedirs(kij_dir, exist_ok=True)
        k_chunks = max(1, min(4, nk_proj))
        o_chunks = max(1, min(omega_batch_size, n_omega))
        h5_kij = h5py.File(kij_stream_path, "w")
        h5_kij.create_dataset("omega_ry", data=np.asarray(omega_req, dtype=np.float64))
        h5_kij.create_dataset("omega_ev", data=np.asarray(omega_req * ryd2ev, dtype=np.float64))
        dset_sigma_kij = h5_kij.create_dataset(
            "sigma_c_kij_ry",
            shape=(n_omega, nk_proj, nb_proj, nb_proj),
            dtype=np.complex128,
            chunks=(o_chunks, k_chunks, nb_proj, nb_proj),
            fillvalue=0.0,
        )
        h5_kij.attrs["layout"] = "omega,k,i,j"

    try:
        if not (use_kij_accum or use_kij_stream):
            raise RuntimeError("Internal error: no valid Σc(ω) accumulation path selected.")

        def _accumulate_kij_stream(global_idx: np.ndarray, contrib_batch: jax.Array) -> None:
            if dset_sigma_kij is None:
                return
            idx = np.asarray(global_idx, dtype=np.int64)
            buf = dset_sigma_kij[idx]
            buf = buf + np.asarray(jax.device_get(contrib_batch), dtype=np.complex128)
            dset_sigma_kij[idx] = buf

        common_branch_kwargs = dict(
            B_q=B_corr,
            Omega_q=Omega_abs,
            base_mask_B=B_mask,
            regularization_width_ry=regularization_width_ry,
            edge_factor=edge_factor,
            target_error=target_error,
            max_nodes=max_nodes,
            crossing_eps_q=crossing_eps_q,
            crossing_max_nodes=crossing_max_nodes,
            wfns=wfns,
            mesh_xy=mesh_xy,
            meta=meta,
            print_fn=print_fn,
            omega_batch_size=omega_batch_size,
            stream_writer=_accumulate_kij_stream if use_kij_stream else None,
            use_shipped_minimax_tables=bool(use_shipped_minimax_tables),
        )

        # Enumerate the 4 branches (ω sign × cond/val), skipping empty ω halves.
        # See _iter_branches for the sign/scale convention derivation.
        branches = _iter_branches(
            omega_pos=omega_pos, idx_pos=idx_pos,
            omega_neg_abs=omega_neg_abs, idx_neg=idx_neg,
            E_cond=E_cond, H_val=H_val,
            cond_mask=cond_mask, val_mask=val_mask,
        )

        # Sum cond+val per ω-half before gathering to host.  Preserves the
        # original traversal order so reduction ordering stays bit-identical.
        per_half: dict[tuple, jax.Array] = {}
        for br in branches:
            sigma_kij, _ = _run_sigma_branch(
                omega_nonneg_ry=br.omega_abs, omega_global_idx=br.omega_idx,
                E_A=br.E_A, base_mask_A=br.base_mask_A,
                kernel_sign=br.kernel_sign, omega_sign_flip=br.omega_sign_flip,
                log_tag=br.tag, scale=br.scale,
                **common_branch_kwargs,
            )
            key = tuple(br.omega_idx.tolist())
            per_half[key] = (per_half[key] + sigma_kij) if key in per_half else sigma_kij

        if not use_kij_stream:
            # _ReduceScatterGpuAccumulator returns Σ sharded (m_X, n_Y), so the
            # host copy needs a cross-process gather rather than jax.device_get.
            # _to_host_np falls back to device_get for single-process / replicated.
            for key, total in per_half.items():
                idx = np.asarray(key, dtype=np.int64)
                sigma_kij_host[idx] = _to_host_np(total, dtype=np.complex128, tiled=False)
    finally:
        if h5_kij is not None:
            h5_kij.close()

    ryd2ev = 13.6056980659
    sigma_kij_req = None if sigma_kij_host is None else jnp.asarray(sigma_kij_host, dtype=jnp.complex128)
    return SigmaOmegaResult(
        omega_ry=np.asarray(omega_req, dtype=np.float64),
        omega_ev=np.asarray(omega_req * ryd2ev, dtype=np.float64),
        sigma_c_kij=sigma_kij_req,
        sigma_kij_h5_path=kij_stream_path,
    )
