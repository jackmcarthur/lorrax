"""Diagonal-Σ(E) fixed point, QSGW Σ_xc build, and SC-COHSEX diagnostics.

The post-self-energy plumbing in ``gw_jax`` operates on **replicated**
``(nk, nb, nb)`` arrays uniformly; the only object that must remain
sharded is the dynamic correlation ``Σ_c(ω, k, m_X, n_Y)`` produced by
``ppm_sigma`` because its ω-axis fan-out makes the full tensor too
large to replicate.  Everything in this module is structured around
that seam:

- :func:`solve_diagonal_sigma_fixed_point` runs on host NumPy with
  vectorised linear interpolation over the (nk, nb) energy grid.  Its
  input ``Σ_diag(ω, k, n)`` is small enough to live replicated.
- :func:`build_qsgw_sigma_xc` is a JIT'd JAX kernel that takes the
  on-device sharded ``Σ_c(ω)`` and the QP energies ``E_kn`` (replicated)
  and returns the Hermitised QSGW Σ_xc replicated on the mesh.  No disk
  round-trip; restart-friendly because the same kernel can be re-called
  with a refreshed ``Σ_c(ω)`` and ``E_kn`` at every QSGW iteration.
"""

from __future__ import annotations

import os

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV


# ---------------------------------------------------------------------------
# Vectorised per-(k, n) ω-axis linear interpolation — shared helper
# ---------------------------------------------------------------------------

#: What :func:`interp_along_omega` does with an evaluation point outside the
#: represented ω range.  THERE IS NO DEFAULT and every call site says which
#: it wants, per the owner's 2026-08-17 "arbitrary choice under degeneracy"
#: ruling: endpoint clamping is a legitimate policy for an ITERATE and an
#: illegitimate one for an OUTPUT, and a shared default cannot be both.
OUT_OF_RANGE_POLICIES = ("clamp", "mask", "refuse")

#: Announced override for the policy the Σ(ω)→at-DFT OUTPUT path resolves.
#: See :func:`resolve_out_of_range_policy`.
_OUT_OF_RANGE_ENV = "LORRAX_OMEGA_OUT_OF_RANGE"


def omega_coverage(omega_grid: np.ndarray,
                   eval_kn: np.ndarray) -> tuple[np.ndarray, int, float]:
    """``(covered_mask, n_uncovered, fraction_uncovered)`` for an ω request.

    ``covered_mask[k, n]`` is True where ``eval_kn[k, n]`` lies inside
    ``[ω_min, ω_max]``, i.e. where Σ(ω) was actually SAMPLED and the linear
    interpolation is an interpolation rather than an edge value.  Interior
    holes in a patched grid are a different question and are owned by
    :func:`assert_omega_grid_covers`; this answers the outer one.
    """
    omega = np.asarray(omega_grid, dtype=np.float64)
    arr = np.asarray(eval_kn, dtype=np.float64)
    covered = (arr >= float(omega[0])) & (arr <= float(omega[-1]))
    n_out = int(np.count_nonzero(~covered))
    frac = float(n_out) / float(arr.size) if arr.size else 0.0
    return covered, n_out, frac


def resolve_out_of_range_policy(default: str = "clamp") -> str:
    """The out-of-range policy for the Σ(ω) OUTPUT path, from the env.

    ``LORRAX_OMEGA_OUT_OF_RANGE`` selects ``clamp`` / ``mask`` / ``refuse``;
    an unrecognised token REFUSES naming the variable rather than resolving
    to the default (rule 13: off-dials may refuse, typos never do).

    WHY THE DEFAULT IS STILL ``clamp``, AND WHY THAT IS NOT AN ENDORSEMENT.
    The committed ``gnppm_debug`` regression fixture measures **132 of 414**
    (31.9 %) sigma-window (k, n) cells outside its ``[-10, +10] eV`` grid,
    and the exact-origin Na run reported ``QSGW: 10142 clipped (41.3%)``.
    Refusing by default would red the anchor fixtures and change frozen
    references on the same commit that added the instrument, which is the
    one thing the band-degeneracy grandfather clause exists to avoid.  So
    the default keeps the historical arithmetic and stops being SILENT
    about it — the count, the fraction and the grid range are reported and
    stamped — and ``refuse`` is one named variable away for anyone
    producing a deliverable that must be covered.
    """
    raw = os.environ.get(_OUT_OF_RANGE_ENV, "").strip().lower()
    if not raw:
        return default
    if raw not in OUT_OF_RANGE_POLICIES:
        raise ValueError(
            f"{_OUT_OF_RANGE_ENV}={raw!r} is not one of "
            f"{OUT_OF_RANGE_POLICIES}.  This decides what Sigma(omega) "
            f"returns for a state whose energy was never sampled; an "
            f"unrecognised value is refused rather than resolved to the "
            f"endpoint clamp.")
    return raw


def interp_along_omega(
    values_w_kn: np.ndarray,
    omega_grid: np.ndarray,
    eval_kn: np.ndarray,
    *,
    out_of_range: str,
    context: str = "",
    print_fn=None,
) -> np.ndarray:
    """Linearly interpolate ``values_w_kn[ω, k, n]`` along the ω-axis at
    per-(k, n) points ``eval_kn[k, n]``.

    Vectorised over (k, n) — one ``np.searchsorted`` + two fancy-index
    gathers, no Python loops.  Used by the diag-Σ(E) fixed point, the
    Σ_c(E_DFT) eqp.dat extractor, the Z-factor central difference, and
    the freq_debug head-correction column.

    Parameters
    ----------
    values_w_kn : (nω, nk, nb), real or complex
    omega_grid : (nω,), monotonically increasing
    eval_kn : (nk, nb), the per-(k, n) ω-evaluation points
    out_of_range : REQUIRED, one of :data:`OUT_OF_RANGE_POLICIES`

        ``"clamp"``
            Evaluate at the nearest grid endpoint.  The historical
            behaviour, bit-for-bit, and correct for an ITERATE whose
            out-of-grid bands the caller reroutes afterwards.
        ``"mask"``
            Return ``nan`` where the state was never sampled, so an
            uncovered cell cannot be mistaken for a measured one.
        ``"refuse"``
            Raise, naming the count, the fraction, the worst energy and
            the grid range.

    context, print_fn
        When ``print_fn`` is given and any cell is out of range, ONE
        counted line is emitted naming ``context``.  Not optional
        decoration: an unreported clamp is the defect this parameter
        exists for.

    WHY THIS ARGUMENT HAS NO DEFAULT.  ``np.clip`` was applied here
    unconditionally, so the at-DFT ``eqp0.dat`` / ``eqp1.dat`` writer wrote
    finite endpoint values for every uncovered state with nothing said.
    MEASURED on the exact-origin Na bandstructure run
    (``runs/Na/02_soc48b_qsgw_mpa/49_origin_exact_fresh_np_ladder_20260818/
    01_np10_nested_batch8/logs/...:5078-5113``): the deck requests only
    ``[-5, +26] eV`` relative to fixed-N mu while bands 5--8 lie near
    -25.07 eV, the log reports ``QSGW: 10142 clipped (41.3%)``, and at
    Gamma band 5's stored ``sigC = 2.791702837+0.013253924i`` is
    bit-for-shown-digits the ``omega = -5 eV`` ENDPOINT of ``sigma_mnk.h5``,
    not Σ at its DFT energy.  The wide-window control moves that band from
    -32.511754 to -28.524225 eV (+3.987528 eV), leaving -0.156302 eV
    against BGW's Eqp0 = -28.367923 eV.  The clamp is a ~4 eV error wearing
    a plausible number's clothes, and the SC Hamiltonian path already
    counts and reroutes these cells while the output path did not.

    Returns
    -------
    out : (nk, nb), same dtype as ``values_w_kn`` (``mask`` promotes an
    integer/real input to float so ``nan`` is representable).
    """
    if out_of_range not in OUT_OF_RANGE_POLICIES:
        raise ValueError(
            f"interp_along_omega: out_of_range must be one of "
            f"{OUT_OF_RANGE_POLICIES}, got {out_of_range!r}.  It has no "
            f"default on purpose: endpoint clamping is right for an iterate "
            f"and wrong for an output, and one default cannot be both.")
    omega = np.asarray(omega_grid, dtype=np.float64)
    eval_arr = np.asarray(eval_kn, dtype=np.float64)
    n_omega = omega.size
    covered, n_out, frac = omega_coverage(omega, eval_arr)
    where = f" [{context}]" if context else ""
    if n_out and print_fn is not None:
        worst = float(eval_arr[~covered].flat[
            int(np.argmax(np.abs(eval_arr[~covered])))])
        print_fn(
            f"  omega coverage{where}: {n_out} of {eval_arr.size} "
            f"({100.0 * frac:.1f}%) evaluation energies lie OUTSIDE the "
            f"sampled grid [{float(omega[0]):.3f}, {float(omega[-1]):.3f}] "
            f"eV (worst {worst:+.3f} eV); policy={out_of_range}.")
    if n_out and out_of_range == "refuse":
        worst = float(eval_arr[~covered].flat[
            int(np.argmax(np.abs(eval_arr[~covered])))])
        raise ValueError(
            f"interp_along_omega{where}: {n_out} of {eval_arr.size} "
            f"({100.0 * frac:.1f}%) evaluation energies lie outside the "
            f"sampled omega grid [{float(omega[0]):.3f}, "
            f"{float(omega[-1]):.3f}] eV; the worst is {worst:+.3f} eV.\n"
            f"  got  : an endpoint value for every one of them, which is a "
            f"finite number that is not Sigma at that energy (measured 4 eV "
            f"on the Na semicore deck).\n"
            f"  want : an omega grid covering the states you are asking "
            f"about.  Widen sigma_omega_min_ev / sigma_omega_max_ev, or add "
            f"a sigma_omega_patches_ev patch over the uncovered band.\n"
            f"  fix  : {_OUT_OF_RANGE_ENV}=clamp restores the historical "
            f"endpoint clamp (reported, never silent); "
            f"{_OUT_OF_RANGE_ENV}=mask writes non-finite there instead.")
    eval_clamped = np.clip(eval_arr, float(omega[0]), float(omega[-1]))
    idx_hi = np.clip(
        np.searchsorted(omega, eval_clamped, side="left"), 1, n_omega - 1)
    idx_lo = idx_hi - 1
    omega_lo = omega[idx_lo]
    omega_hi = omega[idx_hi]
    denom = np.where(omega_hi > omega_lo, omega_hi - omega_lo, 1.0)
    w_hi = (eval_clamped - omega_lo) / denom
    w_lo = 1.0 - w_hi
    nk, nb = eval_arr.shape
    k_idx = np.arange(nk)[:, None]
    n_idx = np.arange(nb)[None, :]
    out = (w_lo * values_w_kn[idx_lo, k_idx, n_idx]
           + w_hi * values_w_kn[idx_hi, k_idx, n_idx])
    if n_out and out_of_range == "mask":
        out = np.asarray(out)
        if not np.issubdtype(out.dtype, np.inexact):
            out = out.astype(np.float64)
        out = np.where(covered, out, np.asarray(np.nan, dtype=out.dtype))
    return out


# ---------------------------------------------------------------------------
# Diagonal-Σ(E) fixed point  (host NumPy, vectorised)
# ---------------------------------------------------------------------------

def assert_omega_grid_covers(E_kn_ry, in_grid_mask, omega_grid_ry, *,
                             context):
    """Refuse solved QP energies inside a hole of a patched ω grid.

    A patched grid (``sigma_omega_patches_ev``) has interior gaps by
    design — that is what makes the MPA crossing rule's cost independent
    of the dynamic range.  The Σ(ω)→E piecewise-linear interpolation is
    silent about a hole: an energy in the gap would be interpolated
    across it and come back plausible-looking and wrong.  So a hole is
    detected from the grid itself (a step above 3× the median step) and
    an in-grid-classified energy strictly inside one — more than one
    median step from both hole edges — is a refusal that names the
    energy and the fix (widen or add a patch).  Contiguous grids have
    no holes and return immediately.
    """
    omega = np.asarray(omega_grid_ry, dtype=np.float64)
    if omega.size < 2:
        return
    steps = np.diff(omega)
    ref = float(np.median(steps))
    holes = np.nonzero(steps > 3.0 * ref)[0]
    if holes.size == 0:
        return
    E = np.asarray(E_kn_ry, dtype=np.float64)
    mask = np.asarray(in_grid_mask, dtype=bool)
    for i in holes:
        lo, hi = float(omega[i] + ref), float(omega[i + 1] - ref)
        inside = mask & (E > lo) & (E < hi)
        if np.any(inside):
            worst = float(E[inside].flat[0])
            raise ValueError(
                f"GATE omega_grid_hole ({context}): "
                f"{int(np.sum(inside))} in-grid QP energies sit inside "
                f"the ω-grid hole ({lo * RYD_TO_EV:.2f}, "
                f"{hi * RYD_TO_EV:.2f}) eV — e.g. "
                f"{worst * RYD_TO_EV:.3f} eV — where Σ(ω) would be "
                "interpolated across the gap.  FALSE case: every solved "
                "QP energy lies on a grid patch.  Widen the nearest "
                "sigma_omega_patches_ev patch (QP energies drift between "
                "QSGW iterations; leave headroom).")


def solve_diagonal_sigma_fixed_point(
    h0_diag_ev: np.ndarray,
    sigma_omega_diag_ev: np.ndarray,
    omega_ev: np.ndarray,
    *,
    max_iter: int = 80,
    tol_ev: float = 1.0e-6,
    mixing: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve E = h0 + Re Σ(E) per (k, n) by linear mixing.

    Vectorised over (k, n): each iteration performs one ``np.searchsorted``
    + two fancy-index gathers over the ω-axis, no Python (k, n) loop.

    Parameters
    ----------
    h0_diag_ev : (nk, nb)
        Static one-body diagonal (typically ``diag(kin_ion + V_H)``) in eV.
    sigma_omega_diag_ev : (nω, nk, nb), complex
        Diagonal Σ_xc(ω) in eV.  Caller is responsible for adding the
        static Σ_x diagonal to the dynamic Σ_c diagonal before invocation.
    omega_ev : (nω,)
        ω-grid in eV, monotonically increasing.

    Returns
    -------
    E : (nk, nb)
        Converged QP eigenvalues in eV.  Bands whose final fixed-point
        argument lies outside ``[ω_min, ω_max]`` are clipped at the grid
        edge for the Σ evaluation; callers should patch out-of-grid bands
        via the scissor (see :mod:`gw.scissor`) if a flatter extrapolation
        is desired.
    converged : (nk, nb), bool
        Per-band convergence flag from the final iteration.
    n_iter : int
        Iterations performed (≤ ``max_iter``).
    """
    h0 = np.asarray(h0_diag_ev, dtype=np.float64)
    sigma_w = np.asarray(sigma_omega_diag_ev, dtype=np.complex128)
    omega = np.asarray(omega_ev, dtype=np.float64)
    if sigma_w.ndim != 3:
        raise ValueError("sigma_omega_diag_ev must have shape (nω, nk, nb).")
    if h0.shape != sigma_w.shape[1:]:
        raise ValueError(
            f"shape mismatch: h0={h0.shape}, sigma_w={sigma_w.shape}")

    i0 = int(np.argmin(np.abs(omega)))
    E = h0 + np.real(sigma_w[i0])
    mix = float(np.clip(mixing, 0.0, 1.0))

    for it in range(max_iter):
        # ``clamp`` is the right policy for an ITERATE: this is a fixed-point
        # search whose out-of-grid bands the caller patches through the
        # scissor afterwards (see the Returns section), and a refusal here
        # would kill the solve on a band the caller was going to replace
        # anyway.  Unreported on purpose — it runs up to ``max_iter`` times
        # and one line per iteration is noise, not evidence; the OUTPUT path
        # is where the count matters and it is reported there.
        sig_at_E = interp_along_omega(sigma_w, omega, E,
                                      out_of_range="clamp")
        E_new = h0 + np.real(sig_at_E)
        E_next = (1.0 - mix) * E + mix * E_new
        diff = np.abs(E_next - E)
        E = E_next
        if bool(np.all(diff < tol_ev)):
            return E, diff < tol_ev, it + 1
    return E, np.abs(E_new - E) < tol_ev, max_iter


# ---------------------------------------------------------------------------
# Σ diagonal extraction from sharded Σ_c(ω, k, m_X, n_Y)
# ---------------------------------------------------------------------------

def is_band_sharded_sigma_omega(a) -> bool:
    """True iff ``a`` is a 4-D jax.Array actually partitioned on its (m, n)
    band axes (the ``sigma_omega_layout=sharded`` tile cube).

    Layout is read off the array itself — the single source of truth
    (QUALITY_PATTERNS #3) — so consumers cannot disagree with the producer
    about which path a given tensor took.  A replicated / single-device /
    numpy Σ_c returns False and takes the historical code paths untouched.
    """
    sharding = getattr(a, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is None or getattr(a, "ndim", 0) != 4:
        return False
    spec = tuple(spec) + (None,) * (4 - len(tuple(spec)))
    return spec[2] is not None or spec[3] is not None


# Kernel cache: one jit'd extractor per mesh.  Module-scope (NOT a
# closure inside the caller) so SC iterations hit the pjit cache instead
# of retracing+recompiling a fresh function object every call.
_EXTRACT_DIAG_KERNEL_CACHE: dict[int, object] = {}

# Sharded-layout siblings (one per mesh): the diagonal-only extractor and
# the band-diagonal adder used by the ``sigma_omega_layout=sharded`` path.
_EXTRACT_DIAG_SHARDED_KERNEL_CACHE: dict[int, object] = {}
_ADD_BAND_DIAG_KERNEL_CACHE: dict[int, object] = {}


def _extract_diag_sharded_kernel(mesh_xy: Mesh):
    """Diagonal of a P(None,None,'x','y')-sharded Σ_c WITHOUT materializing
    the full cube on any device.

    Structural (inside shard_map, where the partitioner cannot hoist a
    gather — QUALITY_PATTERNS #4): each shard extracts the global-diagonal
    elements it owns (the (m, n) intersection with the diagonal), everything
    else contributes exact zeros, and ONE psum over both mesh axes
    replicates the (nω, nk, nb) diagonal — n_ω·nk·nb·16 B (5.4 MB at
    nb=512, nb² → nb vs the full-cube gather).  Every diagonal element is
    owned by exactly one shard, so the psum adds a value to exact zeros —
    movement-only up to IEEE ``x + 0.0`` (bit-exact for every x ≠ -0.0).

    Requires a SQUARE global band extent with both axes divisible by their
    mesh axis (m/p_x, n/p_y integral) — guaranteed by the sharded layout's
    resolve-time divisibility refusal (no mesh pad reaches this path).
    """
    key = id(mesh_xy)
    fn = _EXTRACT_DIAG_SHARDED_KERNEL_CACHE.get(key)
    if fn is None:
        from functools import partial
        from common.shard_map import shard_map

        p_x = int(mesh_xy.shape['x'])

        @jax.jit
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=P(None, None, 'x', 'y'),
                 out_specs=P(None, None, None),
                 check_vma=False)
        def _diag_sharded(tile):
            ix = jax.lax.axis_index('x')
            iy = jax.lax.axis_index('y')
            mb = tile.shape[2]
            nbl = tile.shape[3]
            nb = mb * p_x                       # square global extent
            i = jnp.arange(nb)
            a = i - ix * mb
            b = i - iy * nbl
            own = (a >= 0) & (a < mb) & (b >= 0) & (b < nbl)
            a_c = jnp.clip(a, 0, mb - 1)
            b_c = jnp.clip(b, 0, nbl - 1)
            vals = tile[:, :, a_c, b_c]         # (nω, nk, nb) local gather
            vals = jnp.where(own[None, None, :], vals,
                             jnp.zeros((), dtype=tile.dtype))
            return jax.lax.psum(vals, axis_name=('x', 'y'))

        fn = _diag_sharded
        _EXTRACT_DIAG_SHARDED_KERNEL_CACHE[key] = fn
    return fn


def add_band_diag_sharded(sigma_w_kij: jax.Array, diag_w_kn) -> jax.Array:
    """``Σ += diag(d)`` on a P(None,None,'x','y')-sharded Σ_c — rank-local.

    The analytic q→0 head is band-diagonal; on the sharded layout it is
    injected straight into each rank's tile (zero communication, and the
    dense (nω, nk, nb, nb) head tensor is never materialized anywhere).
    Element-for-element this performs the same IEEE add the replicated
    path's dense ``Σ + head`` performs on the diagonal; off-diagonal
    elements are left untouched (the dense path adds exact 0.0 there).

    ``diag_w_kn`` is host numpy (nω, nk, nb), bit-identical on every rank
    by construction (pure function of replicated inputs) — placed with
    ``device_put_process_local`` per the AA.1 rule.
    """
    from common.collectives import device_put_process_local

    mesh_xy = sigma_w_kij.sharding.mesh
    key = id(mesh_xy)
    fn = _ADD_BAND_DIAG_KERNEL_CACHE.get(key)
    if fn is None:
        from functools import partial
        from common.shard_map import shard_map

        @jax.jit
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=(P(None, None, 'x', 'y'), P(None, None, None)),
                 out_specs=P(None, None, 'x', 'y'),
                 check_vma=False)
        def _add_diag(tile, diag):
            ix = jax.lax.axis_index('x')
            iy = jax.lax.axis_index('y')
            mb = tile.shape[2]
            nbl = tile.shape[3]
            nb = diag.shape[2]
            i = jnp.arange(nb)
            a = i - ix * mb
            b = i - iy * nbl
            own = (a >= 0) & (a < mb) & (b >= 0) & (b < nbl)
            a_c = jnp.clip(a, 0, mb - 1)
            b_c = jnp.clip(b, 0, nbl - 1)
            contrib = jnp.where(own[None, None, :], diag,
                                jnp.zeros((), dtype=diag.dtype))
            # Non-owned i map to clipped duplicate (a, b) slots with exact-0
            # contributions — scatter-add of zeros, value- and bit-neutral.
            return tile.at[:, :, a_c, b_c].add(contrib)

        fn = _add_diag
        _ADD_BAND_DIAG_KERNEL_CACHE[key] = fn

    diag_rep = device_put_process_local(
        np.ascontiguousarray(np.asarray(diag_w_kn, dtype=np.complex128)),
        NamedSharding(mesh_xy, P(None, None, None)))
    return fn(sigma_w_kij, diag_rep)


def gather_sigma_omega_replicated_host(sigma_w_kij: jax.Array) -> np.ndarray:
    """Explicit escape hatch: reconstruct the FULL Σ_c(ω,k,m,n) on every
    rank's host from the sharded layout (the memo's ``.replicated()`` seam).

    This is exactly the replication the sharded layout exists to avoid —
    n_ω·nk·nb²·16 B per rank — so no in-tree consumer calls it; it is the
    promise-contract fallback for tooling / future consumers not yet ported
    (pattern #6: the fallback is explicit, never silent).
    """
    import jax.experimental.multihost_utils as mhu
    return np.asarray(
        mhu.process_allgather(sigma_w_kij, tiled=True), dtype=np.complex128)


def _extract_diag_kernel(mesh_xy: Mesh):
    key = id(mesh_xy)
    fn = _EXTRACT_DIAG_KERNEL_CACHE.get(key)
    if fn is None:
        rep_3d = NamedSharding(mesh_xy, P(None, None, None))
        rep_4d = NamedSharding(mesh_xy, P(None, None, None, None))

        @jax.jit
        def _extract(M):
            M_full = jax.lax.with_sharding_constraint(M, rep_4d)
            # FANCY-INDEX THE DIAGONAL, do not mask for it.  XLA:CPU lowers
            # ``einsum('...ii->...i')`` to a broadcast + select over the WHOLE
            # operand — 169.125 MiB of temporaries on the mos2_4x4 cube, 390 GB
            # at nk=144/nb=2000/nω=41.  Measured 169.125 -> 0.000 MiB.
            i = jnp.arange(M_full.shape[-1])
            diag = M_full[..., i, i]
            return jax.lax.with_sharding_constraint(diag, rep_3d)

        fn = _extract
        _EXTRACT_DIAG_KERNEL_CACHE[key] = fn
    return fn


def extract_sigma_diag_replicated(
    sigma_w_kij: jax.Array,
    mesh_xy: Mesh,
) -> jax.Array:
    """Pull ``Σ[..., n, n]`` from a sharded matrix-valued ω-tensor.

    Input ``sigma_w_kij`` is sharded ``P(None, None, 'x', 'y')`` for
    ``(nω, nk, nb, nb)``.  The (m, n) axes are on **different** mesh
    axes, so a naive ``einsum('...ii->...i')`` computes a per-shard
    block-diagonal which is the global diagonal only on the
    ``ix == iy`` shards — off-diagonal mesh shards silently produce
    garbage off-diagonal values.  We sidestep that by forcing an
    allgather of the full ω-tensor onto each device before the trace.

    Memory note: the materialised tensor is ``nω · nk · nb² · 16 B``
    per device (≈ 270 MB for MoS2 4×4×1, 80 bands, 41 ω-points).  Fits
    comfortably in the 28 GB device budget.

    Sharded layout (``sigma_omega_layout=sharded``): when the input is
    genuinely band-partitioned (read off ``sigma_w_kij.sharding`` — the
    array itself is the source of truth), the shard_map specialisation
    extracts ONLY the diagonal and psums it (nω·nk·nb·16 B moved instead
    of the full cube).  The replicated input keeps the historical kernel
    bit-for-bit untouched.
    """
    if is_band_sharded_sigma_omega(sigma_w_kij):
        return _extract_diag_sharded_kernel(mesh_xy)(sigma_w_kij)
    return _extract_diag_kernel(mesh_xy)(sigma_w_kij)


# ---------------------------------------------------------------------------
# QSGW Σ_xc build — sharded ω-tensor + replicated E_kn → replicated (k, m, n)
# ---------------------------------------------------------------------------

# Kernel cache: one jit'd QSGW-build kernel per mesh (the index/weight
# arrays are runtime args; only the replicated output sharding closes
# over the mesh).  Module-scope for the same reason as
# ``_extract_diag_kernel``: a closure inside ``build_qsgw_sigma_xc``
# retraced+recompiled the full (nω, nk, nb, nb) gather every SC
# iteration.
_QSGW_BUILD_KERNEL_CACHE: dict[tuple[int, bool], object] = {}


def _qsgw_build_kernel(mesh_xy: Mesh, *, replicated_output: bool):
    key = (id(mesh_xy), bool(replicated_output))
    fn = _QSGW_BUILD_KERNEL_CACHE.get(key)
    if fn is None:
        out_3d = NamedSharding(
            mesh_xy,
            P(None, None, None) if replicated_output
            else P(None, "x", "y"))

        @jax.jit
        def _kernel(sig_w, sig_x, ilo, ihi, wlo, whi):
            # ilo/ihi/wlo/whi: (nk, nb) replicated; sig_w: (nω, nk, nb_m_X, nb_n_Y).
            # A[k, m, n] = Σ_c[idx[k, m], k, m, n] (interp at E_m(k))
            # B[k, m, n] = Σ_c[idx[k, n], k, m, n] (interp at E_n(k))
            full = sig_w.shape  # (nω, nk, nb, nb)
            # ONE ω SLOT, NOT THE WHOLE CUBE.  ``take_along_axis`` returns the
            # index tensor's shape and the ``[0]`` below keeps only slot 0, so
            # broadcasting the indices to ``full`` built four (nω, nk, nb, nb)
            # int tensors to produce an (nk, nb, nb) result — 656.03 MiB of
            # temporaries on the mos2_4x4 deck for a 4.00 MiB output, 1.51 TB
            # at nk=144/nb=2000/nω=41.  Measured 656.031 -> 16.031 MiB.
            one = (1,) + tuple(full[1:])

            ilo_m = jnp.broadcast_to(ilo[None, :, :, None], one)
            ihi_m = jnp.broadcast_to(ihi[None, :, :, None], one)
            A_lo = jnp.take_along_axis(sig_w, ilo_m, axis=0)[0]
            A_hi = jnp.take_along_axis(sig_w, ihi_m, axis=0)[0]
            A = wlo[:, :, None] * A_lo + whi[:, :, None] * A_hi

            ilo_n = jnp.broadcast_to(ilo[None, :, None, :], one)
            ihi_n = jnp.broadcast_to(ihi[None, :, None, :], one)
            B_lo = jnp.take_along_axis(sig_w, ilo_n, axis=0)[0]
            B_hi = jnp.take_along_axis(sig_w, ihi_n, axis=0)[0]
            B = wlo[:, None, :] * B_lo + whi[:, None, :] * B_hi

            # Half-sum, then add static Σ_x.  Historical callers request a
            # replicated matrix before Hermitisation.  The fixed-Sigma evSC
            # loop instead keeps the matrix on both band axes: its next
            # operation is a distributed eigh, so gathering it here would
            # create exactly the O(nb^2)-per-rank object that loop avoids.
            M = 0.5 * (A + B) + sig_x
            M = jax.lax.with_sharding_constraint(M, out_3d)
            Mh = 0.5 * (M + jnp.conj(jnp.swapaxes(M, -1, -2)))
            return jax.lax.with_sharding_constraint(Mh, out_3d)

        fn = _kernel
        _QSGW_BUILD_KERNEL_CACHE[key] = fn
    return fn


def build_qsgw_sigma_xc(
    sigma_c_omega_ry: jax.Array,
    sigma_x_kij_ry: jax.Array,
    omega_ev: np.ndarray,
    e_qp_kn_ev: np.ndarray,
    mesh_xy: Mesh,
    *,
    replicated_output: bool = True,
) -> tuple[jax.Array, dict[str, float]]:
    """Build the static Hermitian QSGW Σ_xc[k, m, n].

    Implements the standard QSGW ansatz

        Σ_xc^QSGW_ij(k) = ½[ Σ_xc_ij(k, E_i(k)) + Σ_xc_ij(k, E_j(k)) ]ʰ

    where ``[·]ʰ`` denotes the Hermitian part (real-symmetrisation against
    interpolation noise).  Σ_xc = Σ_c + Σ_x; Σ_x is ω-independent so it
    is added once after the Σ_c(ω) interpolation.

    Energy domain
    -------------
    ``omega_ev`` and ``e_qp_kn_ev`` must share a common reference (typically
    Fermi-relative — ω-grid is centered on E_F by construction in
    ``ppm_pipeline``, and ``E_qp - E_F`` is what the diagonal fixed point
    produces after applying the scissor).

    Sharding
    --------
    - ``sigma_c_omega_ry`` is consumed in place at its native sharding
      ``P(None, None, 'x', 'y')``; the per-shard ``take_along_axis``
      gather is local because the ω-axis (over which we interp) is fully
      replicated and ``e_qp_kn_ev`` is broadcast to all shards.
    - The intermediate ``A``, ``B`` arrays inherit ``P(None, 'x', 'y')``
      sharding for ``(k, m, n)``.
    - The final result is replicated by default, preserving the historical
      caller contract.  ``replicated_output=False`` returns
      ``P(None,'x','y')`` and performs the Hermitian transpose collectively;
      the fixed-Sigma evSC loop uses that layout so a large QP Hamiltonian is
      never gathered merely to feed a distributed eigensolver.

    Returns
    -------
    sigma_xc_qsgw_kij_ry : jax.Array, (nk, nb, nb), complex128, replicated
        unless ``replicated_output=False``, then two-axis band sharded.
    diagnostics : dict with ``n_clipped`` (count of ``E_kn`` outside
        ``[ω_min, ω_max]`` clamped to the grid) and ``omega_min/max_ev``.
    """
    omega = np.asarray(omega_ev, dtype=np.float64)
    E = np.asarray(e_qp_kn_ev, dtype=np.float64)
    if sigma_c_omega_ry.ndim != 4:
        raise ValueError(
            f"sigma_c_omega_ry must have shape (nω, nk, nb, nb); "
            f"got {sigma_c_omega_ry.shape}.")
    if sigma_x_kij_ry.ndim != 3:
        raise ValueError(
            f"sigma_x_kij_ry must have shape (nk, nb, nb); "
            f"got {sigma_x_kij_ry.shape}.")
    n_omega, nk, nb, nb2 = sigma_c_omega_ry.shape
    if nb != nb2 or sigma_x_kij_ry.shape != (nk, nb, nb):
        raise ValueError(
            f"shape mismatch: sigma_c={sigma_c_omega_ry.shape}, "
            f"sigma_x={sigma_x_kij_ry.shape}")
    if E.shape != (nk, nb):
        raise ValueError(
            f"e_qp_kn_ev must have shape ({nk}, {nb}); got {E.shape}.")

    # Linear-interp index/weight arrays, host-side then pushed replicated.
    omega_lo = float(omega[0])
    omega_hi = float(omega[-1])
    E_clamped = np.clip(E, omega_lo, omega_hi)
    n_clipped = int(np.count_nonzero(E_clamped != E))
    idx_hi = np.clip(np.searchsorted(omega, E_clamped, side="left"),
                     1, n_omega - 1)
    idx_lo = idx_hi - 1
    ω_lo = omega[idx_lo]
    ω_hi = omega[idx_hi]
    denom = np.where(ω_hi > ω_lo, ω_hi - ω_lo, 1.0)
    w_hi = (E_clamped - ω_lo) / denom
    w_lo = 1.0 - w_hi

    rep_2d = NamedSharding(mesh_xy, P(None, None))
    # Numpy → replicated, placed PROCESS-LOCALLY: a bare ``jax.device_put``
    # of host numpy onto a multi-process replicated sharding silently runs
    # multihost ``assert_equal`` — four hidden P-linear all-gathers of
    # (nk, nb) tables on every fixed_point solve (AO-sweep straggler class,
    # scorecard AA.1/Y.5; converted 2026-07-28).  The idx/w tables are pure
    # deterministic functions (np.searchsorted/clip) of the replicated E and
    # ω inputs — bit-identical on every rank by construction, which is
    # device_put_process_local's documented precondition;
    # LORRAX_CHECK_REPLICA=1 restores the assertion.  (``jnp.asarray`` wrap
    # would be worse still — a single-device staging that turns device_put
    # into an all-reduce.)
    from common.collectives import device_put_process_local
    idx_lo_j = device_put_process_local(idx_lo.astype(np.int32), rep_2d)
    idx_hi_j = device_put_process_local(idx_hi.astype(np.int32), rep_2d)
    w_lo_j   = device_put_process_local(w_lo.astype(np.complex128), rep_2d)
    w_hi_j   = device_put_process_local(w_hi.astype(np.complex128), rep_2d)

    sigma_xc_qsgw = _qsgw_build_kernel(
        mesh_xy, replicated_output=bool(replicated_output))(
        sigma_c_omega_ry, sigma_x_kij_ry,
        idx_lo_j, idx_hi_j, w_lo_j, w_hi_j,
    )
    sigma_xc_qsgw.block_until_ready()

    diagnostics = {
        "n_clipped": float(n_clipped),
        "frac_clipped": float(n_clipped) / float(nk * nb) if nk * nb else 0.0,
        "omega_min_ev": omega_lo,
        "omega_max_ev": omega_hi,
    }
    return sigma_xc_qsgw, diagnostics


# ---------------------------------------------------------------------------
# The opt-in QSGW cube write — one implementation, two seams
# ---------------------------------------------------------------------------

def write_qsgw_sigma_cube(
    sigma_omega_h5_path,
    sigma_xc_qsgw_kij_ry,
    *,
    config,
    print_fn=print,
) -> bool:
    """Append ``sigma_xc_qsgw_kij_ev`` to a ``sigma_mnk.h5`` just written.

    No-op unless the deck sets ``write_qsgw_datasets`` (see
    ``gw_config._DEFAULTS`` for what the appendix is and why it is off by
    default).  Returns whether anything was written.

    THREE CALL SITES, AND THEY ARE NOT INTERCHANGEABLE.  Each hands over
    a QSGW build — Σ_x + Σ_c^QSGW, already Hermitised by
    :func:`build_qsgw_sigma_xc`'s closing ``½(M + M†)`` — at the one
    moment its own path has the build that goes into H, in the basis the
    file's other cubes are in:

    * ``gw.sigma_dispatch.compute_sigma_xc``, immediately after the QSGW
      build, on the one-shot path where ``finalize_dynamic_sigma``
      created the file a few statements earlier.  Gated there on the same
      ``write_sigma_omega_h5`` flag that decided whether the file was
      written at all, so it cannot fire during an SC iteration (which
      passes False and writes nothing).
    * :func:`solve_qp`'s ``fixed_point`` branch, which REBUILDS at the
      solved on-shell energies and therefore supersedes the at-DFT cube
      the dispatch just wrote.  The append replaces the dataset, so the
      file ends up holding the Σ that actually built the Hamiltonian.
    * ``gw.sc_iteration.dump_sigma_omega_h5_final``, right after the
      converged single write, and BEFORE ``run_sc_driver`` rotates the
      matrix back to the DFT basis.

    The cube arrives replicated — ``build_qsgw_sigma_xc`` pins it so with
    a ``with_sharding_constraint`` before it Hermitises — so the rank-0
    h5py append below has a whole array to write on every process, and
    the barrier lets the caller rely on the file being complete when this
    returns.
    """
    if not bool(getattr(config, "write_qsgw_datasets", False)):
        return False
    if not sigma_omega_h5_path or sigma_xc_qsgw_kij_ry is None:
        return False
    from common.collectives import barrier
    from file_io import append_qsgw_datasets_h5

    if jax.process_index() == 0:
        append_qsgw_datasets_h5(
            sigma_omega_h5_path,
            {"sigma_xc_qsgw_kij_ev":
                RYD_TO_EV * np.asarray(sigma_xc_qsgw_kij_ry)},
            print_fn=print_fn)
    barrier("qsgw_sigma_cube_append")
    return True


# ---------------------------------------------------------------------------
# update_H — the qp_solver dispatch (one-shot / fixed-point)
# ---------------------------------------------------------------------------

def solve_qp(
    qp_solver,
    sigma_result,
    kin_ion: jax.Array,
    *,
    config,
    meta,
    mesh_xy: Mesh,
    print_fn=print,
) -> jax.Array:
    """``update_H[Σ; qp_solver]`` — turn a :class:`~gw.sigma_dispatch.SigmaResult`
    into ``sigma_total = Σ_xc + V_H`` (Ry) whose eigh
    yields the QP eigenstates.

    The three QP-energy definitions (see ``LorraxConfig.qp_solver``):

    - ``one_shot_dft`` — the full-matrix QSGW-Hermitianised Σ_xc was
      already evaluated at E_DFT inside ``compute_sigma_xc`` (the same
      call the SC iteration map makes), so this is a pass-through.  Its
      subsequent ``eigh`` result is distinct from the fixed-DFT-state
      diagonal G0W0 result written to ``eqp0.dat``.
      Static modes (X_ONLY / COHSEX) and the streamed-Σ_c stand-in land
      here too: ``sigma_xc_kij_ry`` is the mode's total Σ_xc by
      construction.
    - ``fixed_point`` — diagonal on-shell solve E = h₀ + ReΣ(E) followed
      by a QSGW rebuild at the solved energies (+ optional per-band
      scissor for out-of-grid bands).  Dynamic, non-streamed only
      (validated at config load).  The dispatch's internal at-DFT build
      is superseded here — one redundant (cheap) QSGW contraction, the
      price of keeping ``compute_sigma_xc``'s signature uniform.
    - ``self_consistent`` is NOT handled here — the SC driver owns its
      own loop and rotation-back seam (``sc_iteration``).

    All quantities are in **Rydberg** until the scissor's print summary
    and the eV seam of the QSGW build kernel.  Σ_c(ω) lives natively in
    Ry on the Ry ω-grid; mixing that with eV-converted h0/Σ_x is a
    footgun.
    """
    from .gw_config import QPSolver, qp_solver_semantics

    # ONE BASIS ASSUMED across these two reads, and across the sig_x /
    # omega_dft_rel_ev reads in the FIXED_POINT branch below.  SC owns its
    # own loop and never reaches this function; the object here is the
    # one-shot one, DFT basis throughout.
    sig_h = sigma_result.v_h_kij_ry
    sigma_c_omega = sigma_result.sigma_c_omega_kij_ry

    if qp_solver is not QPSolver.FIXED_POINT or sigma_c_omega is None:
        if config.compute_mode.is_dynamic and sigma_c_omega is not None:
            print_fn(
                "  QP solver: one_shot_dft — "
                + qp_solver_semantics(QPSolver.ONE_SHOT_DFT).description)
        return sigma_result.sigma_xc_kij_ry + sig_h

    # QPSolver.FIXED_POINT: diagonal Σ(E) fixed point in Ry.
    # Diagonal Σ_c(ω, k, n) and Σ_x(k, n) replicated on host, in Ry.
    sig_x = sigma_result.sigma_x_kij_ry
    omega_grid_ry = np.asarray(sigma_result.omega_grid_ry, dtype=np.float64)
    E_dft_rel_ry = np.asarray(
        sigma_result.omega_dft_rel_ev, dtype=np.float64) / RYD_TO_EV

    sigma_c_diag_w_kn_ry = np.asarray(extract_sigma_diag_replicated(
        sigma_c_omega, mesh_xy))
    sigma_x_diag_kn_ry = np.real(
        np.diagonal(np.asarray(sig_x), axis1=1, axis2=2))
    sigma_xc_diag_w_kn_ry = sigma_c_diag_w_kn_ry + sigma_x_diag_kn_ry[None, :, :]

    h0_diag_ry = np.real(
        np.diagonal(np.asarray(kin_ion + sig_h), axis1=1, axis2=2))
    efermi_ry = float(sigma_result.efermi_dft_ev) / RYD_TO_EV
    E_sc_rel_ry, _, n_iter = solve_diagonal_sigma_fixed_point(
        h0_diag_ry - efermi_ry, sigma_xc_diag_w_kn_ry, omega_grid_ry,
        max_iter=120, tol_ev=1.0e-7 / RYD_TO_EV, mixing=0.6,
    )

    # Per-band scissor for out-of-grid bands.  A band is "in-grid" iff
    # E_DFT[k, n] lies in [ω_min, ω_max] for every k; if any single k
    # is outside, the band gets the scissor uniformly across k (the
    # diagonal solver clipped Σ_c at the ω-boundary for the offending
    # k, which would otherwise contaminate the band's k-dispersion).
    # The scissor itself is fitted on in-grid bands only.  Default
    # fallback when the scissor flag is off: E_DFT (the natural
    # zeroth-order QP correction = 0 estimate); the older fallback
    # of using ``eigvalsh(H_qp)`` was unreliable for pseudobands.
    from .scissor import (
        classify_bands_in_grid, fit_scissor, full_bz_k_weights)
    band_in_grid, in_grid_kn_band = classify_bands_in_grid(
        E_dft_rel_ry, float(omega_grid_ry[0]), float(omega_grid_ry[-1]))
    assert_omega_grid_covers(
        E_sc_rel_ry, in_grid_kn_band, omega_grid_ry,
        context="diagonal QSGW fixed point")
    n_bands_in = int(band_in_grid.sum())
    n_bands_total = int(band_in_grid.size)
    print_fn(
        f"  Diagonal SC: {n_bands_in}/{n_bands_total} bands fully in grid, "
        f"{n_iter} iterations")
    if (
        config.sigma.sigma_at_dft_extrapolate
        and 0 < n_bands_in < n_bands_total
    ):
        occ_mask_kn = np.broadcast_to(
            np.arange(E_sc_rel_ry.shape[1])[None, :] < meta.nelec,
            E_sc_rel_ry.shape).astype(bool)
        # Fit in eV so the printed slopes/intercepts are human-readable.
        # Sort-and-pair semantics (per-k argsort on each of E_DFT and
        # E_QP independently) live inside ``fit_scissor`` and are
        # robust to QSGW reorderings; one-shot G0W0 has no
        # reordering and the sort is a no-op.
        fit = fit_scissor(
            E_dft_kn_ev=E_dft_rel_ry * RYD_TO_EV,
            E_qp_kn_ev=E_sc_rel_ry * RYD_TO_EV,
            valence_mask_kn=occ_mask_kn,
            fit_mask_kn=in_grid_kn_band,
            # UNREDUCED k.  Every operand here descends from
            # ``sigma_result``, which compute_sigma_xc builds on the full
            # BZ (Σ is an FFT over the k-grid); this path never sees an
            # IBZ k-set.  The SC loop's own refit is the one that can, and
            # it weights by star multiplicity — sc_iteration.py.
            k_weights=full_bz_k_weights(E_dft_rel_ry.shape[0]),
        )
        print_fn(f"  Scissor fit: {fit.summary()}")
        extrap_rel_ry = E_dft_rel_ry + fit.predict(
            E_dft_rel_ry * RYD_TO_EV, occ_mask_kn) / RYD_TO_EV
        E_sc_rel_ry = np.where(in_grid_kn_band, E_sc_rel_ry, extrap_rel_ry)
    else:
        E_sc_rel_ry = np.where(in_grid_kn_band, E_sc_rel_ry, E_dft_rel_ry)
    E_sc_rel_ev = E_sc_rel_ry * RYD_TO_EV

    # QSGW Σ_xc^QSGW: sharded ω-tensor + replicated E_sc → replicated Σ_xc.
    # Build kernel takes ω-grid and evaluation energies in **eV**; we
    # convert at the seam (kernel internals convert; result is Ry).
    sig_x_rep = jax.device_put(jnp.asarray(sig_x),
        NamedSharding(mesh_xy, P(None, None, None)))
    sigma_xc_qsgw_kij_ry, qsgw_diag = build_qsgw_sigma_xc(
        sigma_c_omega, sig_x_rep,
        omega_grid_ry * RYD_TO_EV, E_sc_rel_ev, mesh_xy,
    )
    print_fn(f"  QSGW: {int(qsgw_diag['n_clipped'])} clipped "
        f"({100*qsgw_diag['frac_clipped']:.1f}%)")
    # THE REBUILD SUPERSEDES THE AT-DFT CUBE IN THE FILE TOO.  The Σ
    # dispatch already appended its own QSGW build — evaluated at E_DFT,
    # which is what ``one_shot_dft`` keeps — and on this branch that build
    # is not the one that goes into H: this one is, at the solved
    # energies.  A file holding the superseded cube would disagree with
    # ``eqp0.dat`` by the whole on-shell correction with nothing to say so,
    # and the append replaces the dataset, so the last writer on the path
    # is the one whose Σ built the Hamiltonian.
    write_qsgw_sigma_cube(
        sigma_result.sigma_omega_h5_path, sigma_xc_qsgw_kij_ry,
        config=config, print_fn=print_fn)
    return sigma_xc_qsgw_kij_ry + sig_h


# ---------------------------------------------------------------------------
# QP-energy comparison plot
# ---------------------------------------------------------------------------

def plot_qp_energy_comparison(
    output_png: str,
    e_ref_kn_ev: np.ndarray,
    e_static_kn_ev: np.ndarray,
    e_dyn0_kn_ev: np.ndarray,
    e_diag_sc_kn_ev: np.ndarray,
) -> str:
    """Scatter + k=0 trend of QP energies for the three approximations."""
    import matplotlib.pyplot as plt

    x = np.asarray(e_ref_kn_ev, dtype=np.float64).reshape(-1)
    y_static = np.asarray(e_static_kn_ev, dtype=np.float64).reshape(-1)
    y_dyn0 = np.asarray(e_dyn0_kn_ev, dtype=np.float64).reshape(-1)
    y_diag = np.asarray(e_diag_sc_kn_ev, dtype=np.float64).reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(x, y_static, s=10, alpha=0.6, label="Static COHSEX")
    axes[0].scatter(x, y_dyn0, s=10, alpha=0.6, label="Bare X + Σ_c(0)")
    axes[0].scatter(x, y_diag, s=10, alpha=0.6, label="Diagonal SC Σ(E)")
    mn = float(min(np.min(x), np.min(y_static), np.min(y_dyn0), np.min(y_diag)))
    mx = float(max(np.max(x), np.max(y_static), np.max(y_dyn0), np.max(y_diag)))
    axes[0].plot([mn, mx], [mn, mx], "k--", lw=1)
    axes[0].set_xlabel("Reference energy (eV)")
    axes[0].set_ylabel("QP energy (eV)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("All (k, n)")

    b = np.arange(e_ref_kn_ev.shape[1])
    axes[1].plot(b, e_static_kn_ev[0], "-o", ms=3, label="Static COHSEX")
    axes[1].plot(b, e_dyn0_kn_ev[0], "-o", ms=3, label="Bare X + Σ_c(0)")
    axes[1].plot(b, e_diag_sc_kn_ev[0], "-o", ms=3, label="Diagonal SC Σ(E)")
    axes[1].set_xlabel("Band index")
    axes[1].set_ylabel("Energy at k = 0 (eV)")
    axes[1].set_title("k = 0")
    axes[1].legend(fontsize=8)

    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return output_png


# ---------------------------------------------------------------------------
# Managed-file cleanup — shared by the SC loop's disk-bounded artifacts
# ---------------------------------------------------------------------------

def remove_managed(dir_path, pattern, *, keep=(), barrier_tag, print_fn=print):
    """Rank-0 scan-and-unlink of managed files, then a collective barrier.

    Removes every entry of ``dir_path`` whose NAME fullmatches ``pattern``
    and whose full path is not in ``keep``, on rank 0 only; all ranks then
    meet at ``barrier_tag`` so the directory looks the same everywhere on
    return.  Returns the removed names (empty off rank 0).  Shared by
    ``sc_iteration._clear_sc_eqp_snapshots`` and
    ``gw.mpa.model.retain_iteration_artifacts``; each caller owns its own
    regex, keep-set and report line.
    """
    import os
    import re
    from common.collectives import barrier, process_rank

    managed = re.compile(pattern)
    keep = set(keep)
    root = os.path.abspath(os.fspath(dir_path))
    removed = []
    if process_rank() == 0 and os.path.isdir(root):
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if managed.fullmatch(name) and path not in keep:
                os.remove(path)
                removed.append(name)
    barrier(barrier_tag, print_fn=print_fn)
    return removed


__all__ = [
    "build_qsgw_sigma_xc",
    "extract_sigma_diag_replicated",
    "interp_along_omega",
    "omega_coverage",
    "resolve_out_of_range_policy",
    "OUT_OF_RANGE_POLICIES",
    "plot_qp_energy_comparison",
    "remove_managed",
    "solve_diagonal_sigma_fixed_point",
    "write_qsgw_sigma_cube",
]
