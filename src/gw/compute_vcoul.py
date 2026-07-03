"""
Chunked computation of V_q(μ, ν) = Σ_G ζ̃*_μ(G) ζ̃_ν(G) from zeta stored in HDF5.

This module provides memory-efficient routines for computing the ISDF Coulomb
matrix elements when the full zeta_q(μ, r) doesn't fit in GPU memory.

Key features:
- μ-chunked FFT: Process B_μ centroids at a time
- ν-chunked contraction: Compute V blocks without caching FFT outputs
- Hermitian symmetry: Only compute upper triangle, fill lower by conjugation
- 2D sharding: Output V_q sharded P('x', 'y') for downstream use

Memory model:
- FFT workspace: O(B_μ × n_G) per chunk
- V_q output: O(n_μ²) - typically small (e.g., 2304² × 16B = 85 MB)
- Redundant FFT work: O((n_μ/B_μ)²) vs O(n_μ/B_μ) with caching

Note: For future optimization, if a single zeta_q(μ, r) fits on sqrt(P) processors,
      we could batch multiple q-points to amortize FFT setup costs.
"""

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

from common import timing
from common.fft_helpers import make_sharded_fftn_3d


# ============================================================================
# FFT grid helpers (mirrors gw_jax.py)
# ============================================================================

def fft_integer_axes(fft_nx: int, fft_ny: int, fft_nz: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return integer FFT frequency grids in numpy.fft.fftfreq order.

    ─ NOTE TO FUTURE EDITORS — THE numpy USAGE BELOW IS INTENTIONAL ─
    Host-side tiny-grid shape helper.  ``jnp.fft.fftfreq`` + reshape +
    astype each fire their own pjit at trace time — ~7 cache misses per
    call.  Commit bbff26f (2026-04-18) converted to numpy with the JAX
    cast deferred to return.  DO NOT "fix" back to ``jnp``.
    """
    gx = (np.fft.fftfreq(fft_nx) * fft_nx).astype(np.float64).reshape(fft_nx, 1, 1)
    gy = (np.fft.fftfreq(fft_ny) * fft_ny).astype(np.float64).reshape(1, fft_ny, 1)
    gz = (np.fft.fftfreq(fft_nz) * fft_nz).astype(np.float64).reshape(1, 1, fft_nz)
    return jnp.asarray(gx), jnp.asarray(gy), jnp.asarray(gz)


# ============================================================================
# Coulomb potential computation (2D truncated)
# ============================================================================

@partial(jax.jit, donate_argnums=(0,), static_argnums=(4,))
def _v_q_per_q_g_chunked_jit(
    V_acc: jax.Array,            # (n_rmu_L, n_rmu_R) c128 — donated
    zeta_q_L: jax.Array,         # (n_rmu_L, ngkmax)  c128
    zeta_q_R: jax.Array,         # (n_rmu_R, ngkmax)  c128
    v_q: jax.Array,              # (ngkmax,) c128
    g_chunk: int,                # static
) -> jax.Array:
    """Inner kernel: V += conj(ζ_L) · v · ζ_Rᵀ, G-chunked.

    See module-level docstring above for the full math + memory model.
    """
    ngkmax = int(zeta_q_L.shape[-1])

    def body(start, V):
        # jax.lax.dynamic_slice keeps a single compile shape regardless
        # of where the chunk lands; the trailing chunk is sized to fit
        # by the outer python loop (no dynamic chunk size).
        L_chunk = jax.lax.dynamic_slice_in_dim(
            zeta_q_L, start, g_chunk, axis=-1)        # (n_rmu_L, g_chunk)
        R_chunk = jax.lax.dynamic_slice_in_dim(
            zeta_q_R, start, g_chunk, axis=-1)        # (n_rmu_R, g_chunk)
        v_chunk = jax.lax.dynamic_slice_in_dim(
            v_q, start, g_chunk, axis=0)              # (g_chunk,)
        # Pre-multiply v into L (real ≥ 0 for bare Coulomb; signed /
        # complex for bispinor transverse — both fine).
        L_weighted = jnp.conj(L_chunk) * v_chunk[None, :]
        return V + L_weighted @ R_chunk.T            # (n_rmu_L, n_rmu_R)

    # Whole-Gchunk strides (the python caller pads ngkmax to a multiple
    # of g_chunk via zeta's writer-side pad slots ζ=0, so no remainder
    # branch is needed inside the jit).
    n_chunks = ngkmax // g_chunk
    V = V_acc
    for i in range(n_chunks):
        V = body(i * g_chunk, V)
    return V


def compute_v_q_per_q_g_chunked(
    zeta_q_L: jax.Array,
    zeta_q_R: jax.Array,
    v_q: jax.Array,
    *,
    g_chunk: int = 4096,
    V_acc: jax.Array | None = None,
) -> jax.Array:
    """V_q[μν] = Σ_G ζ̃*_μ(G) v(G) ζ̃_ν(G) for a single q, G-chunked.

    Parameters
    ----------
    zeta_q_L : ``(n_rmu_L, ngkmax)`` c128
        ζ̃ on the per-q WFN.h5-style sphere (writer's G-flat layout).
        Pad slots ``j ≥ ngk[q]`` carry ζ̃ = 0 — they contribute zero
        to the contract regardless of what ``v_q`` does there.
    zeta_q_R : ``(n_rmu_R, ngkmax)`` c128
        Right operand.  Pass the same array as ``zeta_q_L`` for the
        diagonal V^{0,0} case (single-ζ); pass a separate buffer for
        bispinor off-diagonal tiles V^{μ_L, ν_L}.
    v_q : ``(ngkmax,)`` c128
        Per-G Coulomb weight.  Bare Coulomb is real ≥ 0; bispinor
        transverse projector tiles are signed / complex.  Pad slots
        are don't-care (ζ̃ = 0 there).
    g_chunk : int
        Number of G's per inner contraction.  Caller pads ``ngkmax``
        up to a multiple of ``g_chunk`` (typical: pad with the
        WFN.h5 sentinel slot, ζ̃ = 0).  Memory-wise: this kernel's
        peak intermediate is ``n_rmu_L · g_chunk`` complex128 plus
        the (n_rmu_L, n_rmu_R) accumulator.  TODO(q-batch): an outer
        ``vmap`` over a small q-chunk would amortize the kernel's
        launch overhead; currently the driver runs one q at a time.
    V_acc : ``(n_rmu_L, n_rmu_R)`` c128 or None
        Donated accumulator.  Pass ``None`` to allocate a fresh
        zero-init buffer matching the sharding of ``zeta_q_L`` /
        ``zeta_q_R`` (P('x', 'y') if both are P(('x','y'), None)).

    Returns
    -------
    V : ``(n_rmu_L, n_rmu_R)`` c128
        ``V_acc + Σ_G  conj(ζ̃_L(G)) · v(G) · ζ̃_R(G)``.

    Sharding & dtype notes
    ----------------------
    All inputs are c128.  v_q can be passed as float64 and is cast
    by the caller before reaching this function.  No internal
    sharding constraints — the jit inherits the caller's input
    shardings and emits the matching output sharding.
    """
    n_rmu_L = int(zeta_q_L.shape[0])
    n_rmu_R = int(zeta_q_R.shape[0])
    ngkmax = int(zeta_q_L.shape[-1])
    if int(zeta_q_R.shape[-1]) != ngkmax:
        raise ValueError(
            f"compute_v_q_per_q_g_chunked: zeta_q_L.ngkmax={ngkmax} ≠ "
            f"zeta_q_R.ngkmax={int(zeta_q_R.shape[-1])}")
    if int(v_q.shape[0]) != ngkmax:
        raise ValueError(
            f"compute_v_q_per_q_g_chunked: v_q.ngkmax={int(v_q.shape[0])} "
            f"≠ ζ.ngkmax={ngkmax}")
    g_chunk = int(g_chunk)
    if ngkmax % g_chunk != 0:
        raise ValueError(
            f"compute_v_q_per_q_g_chunked: ngkmax ({ngkmax}) must be a "
            f"multiple of g_chunk ({g_chunk}); pad the trailing slots "
            f"with ζ̃ = 0 to satisfy this (the writer's per-q sentinel "
            f"pad already does this when ngkmax is chosen by the "
            f"caller as a multiple of g_chunk).")

    if V_acc is None:
        V_acc = jnp.zeros(
            (n_rmu_L, n_rmu_R), dtype=zeta_q_L.dtype)
    if v_q.dtype != zeta_q_L.dtype:
        v_q = v_q.astype(zeta_q_L.dtype)

    return _v_q_per_q_g_chunked_jit(V_acc, zeta_q_L, zeta_q_R, v_q, g_chunk)


def build_v_head_miniBZ_avg_3d(
    kgrid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    *,
    nmc: int = 2**18,
    seed: int = 42,
) -> np.ndarray:
    """Mini-BZ-averaged bare Coulomb head ``<v(q+δq, G=0)>_miniBZ`` per q.

    3D bulk only.  Returns ``(nkx, nky, nkz)`` real array of head values
    in Rydberg / cell-volume units.  q=0 returns 0 (the actual head is
    injected separately via a rank-1 correction in the Σ_X path).

    The MC integration draws ``nmc`` (default 2¹⁸) points uniformly on
    the Voronoi cell of the mini-BZ and averages ``8π/|q+δq|²``.  Both
    the legacy ``make_v_munu_chunked_kernel.get_sqrt_v_and_phase`` and
    the G-flat ``compute_all_V_q_g_flat`` consume this table — keep
    them in lock-step.
    """
    from .vcoul import wrap_points_to_voronoi
    nkx, nky, nkz = (int(s) for s in kgrid)
    bvec_j = jnp.asarray(bvec, dtype=jnp.float64)
    rng = np.random.RandomState(seed)
    randvals = rng.uniform(0, 1, (nmc, 3))
    randcart = (randvals @ bvec.T)
    wrapped = np.asarray(wrap_points_to_voronoi(
        jnp.asarray(randcart), bvec_j, nmax=1))
    kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)
    randlims = bvec.T @ (np.diag(1.0 / kgrid_arr) @ np.linalg.inv(bvec.T))
    dq_cart = (randlims @ wrapped.T).T  # (nmc, 3) mini-BZ offsets in Cartesian

    v_head_avg = np.zeros((nkx, nky, nkz), dtype=np.float64)
    for qx in range(nkx):
        for qy in range(nky):
            for qz in range(nkz):
                qw = np.array([qx, qy, qz], dtype=np.float64)
                qw = np.where(qw > kgrid_arr / 2, qw - kgrid_arr, qw)
                q_frac = qw / kgrid_arr
                q_cart = q_frac @ bvec
                if np.dot(q_cart, q_cart) < 1e-12:
                    v_head_avg[qx, qy, qz] = 0.0  # q=0 head handled separately
                else:
                    shifted = q_cart[None, :] + dq_cart  # (nmc, 3)
                    denom = np.sum(shifted**2, axis=1)
                    v_head_avg[qx, qy, qz] = np.mean(8.0 * np.pi / denom)
    return v_head_avg * (1.0 / float(cell_volume))


def compute_v_q_per_G(
    q_irr_frac: np.ndarray,
    gvec_components: np.ndarray,
    *,
    bvec: np.ndarray,
    cell_volume: float,
    sys_dim: int,
    vcoul_cutoff_ry: float | None = None,
    bdot: np.ndarray | None = None,
    v_head_miniBZ: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``v(q+G)`` at the per-q WFN.h5-style G-list.

    Mirrors the formula inside ``make_v_munu_chunked_kernel.get_sqrt_v_and_phase``
    but operates on a per-q ``gvec_components`` table (instead of the
    full FFT grid) — the writer's ``isdf_header/gvec_components`` is
    exactly the input the consumer needs.  Returns one ``(ngkmax,)``
    row of ``v(q+G)`` per q in ``q_irr_frac``.

    Pad slots in ``gvec_components`` (sentinel Miller index
    ``(-nx/2, -ny/2, -nz/2)``) get whatever ``v`` is at that
    position — caller need not zero them because the contract uses
    ζ̃ = 0 at those slots.

    Parameters
    ----------
    q_irr_frac : ``(n_q, 3)`` float64
        Fractional q-vectors in BGW-wrap convention (already divided
        by kgrid).
    gvec_components : ``(n_q, 3, ngkmax)`` int32
        Per-q Miller indices from ``isdf_header.gvec_components``.
    bvec, cell_volume, sys_dim, vcoul_cutoff_ry, bdot
        Same conventions as ``make_v_munu_chunked_kernel``.
        ``vcoul_cutoff_ry`` zeroes ``v`` at G's with |q+G|² past
        the cutoff (== V_q's bare-Coulomb cutoff; may be < the
        ζ-sphere cutoff that built ``gvec_components``).

    Returns
    -------
    v_q_per_G : ``(n_q, ngkmax)`` float64
        ``v(q+G)`` evaluated at every (q, G) in the components table.

    Notes
    -----
    This is a *host-side* helper — the per-q ``v(q+G)`` is built once
    at consumer setup and pushed to device.  Not jitted; not sharded.
    For very large ngkmax this could be vectorised across q on device,
    but it's a one-shot cost per V_q run.
    """
    if sys_dim not in (0, 2, 3):
        raise NotImplementedError(
            f"compute_v_q_per_G: sys_dim must be 0 / 2 / 3; got {sys_dim}")
    q_irr_frac = np.asarray(q_irr_frac, dtype=np.float64).reshape(-1, 3)
    gvec = np.asarray(gvec_components, dtype=np.float64)         # (n_q, 3, ngkmax)
    if gvec.ndim != 3 or gvec.shape[1] != 3:
        raise ValueError(
            f"gvec_components must be (n_q, 3, ngkmax); got {gvec.shape}")
    n_q, _, ngkmax = gvec.shape
    bvec_f = np.asarray(bvec, dtype=np.float64)
    fact = 1.0 / float(cell_volume)

    if sys_dim == 2:
        zc = float(np.pi / float(bvec_f[2, 2]))
    if v_head_miniBZ is not None:
        # Per-q grid index: round (q_frac * kgrid) and wrap modulo kgrid.
        # ``v_head_miniBZ`` is indexed by integer (qx, qy, qz) on the
        # k-grid (the table the legacy ``get_sqrt_v_and_phase`` consumes).
        head_arr = np.asarray(v_head_miniBZ, dtype=np.float64)
        if head_arr.ndim != 3:
            raise ValueError(
                f"v_head_miniBZ must be (nkx, nky, nkz); got shape "
                f"{head_arr.shape}")
        head_kgrid = np.array(head_arr.shape, dtype=np.float64)
    out = np.zeros((n_q, ngkmax), dtype=np.float64)

    for qi in range(n_q):
        qf = q_irr_frac[qi]
        # gvec[qi]: (3, ngkmax) -> per-G Miller; (q + G) in fractional.
        qG_frac = qf[:, None] + gvec[qi]                          # (3, ngkmax)
        qG_cart = bvec_f.T @ qG_frac                              # (3, ngkmax)
        denom = np.sum(qG_cart * qG_cart, axis=0)                 # (ngkmax,)
        denom_zero = denom < 1e-12
        denom_safe = np.where(denom_zero, 1.0, denom)
        if sys_dim == 3:
            v_reg = 8.0 * np.pi / denom_safe
            v = np.where(denom_zero, 0.0, v_reg * fact)
            if v_head_miniBZ is not None:
                # Replace the G=0 entry (the (0,0,0) Miller slot) with the
                # mini-BZ averaged head value for this q.  Same formula the
                # legacy ``get_sqrt_v_and_phase`` uses; q=0 keeps v=0 by
                # construction (the actual head is injected via a separate
                # rank-1 path in Σ_X).
                qx_i = int(np.round(qf[0] * head_kgrid[0])) % int(head_kgrid[0])
                qy_i = int(np.round(qf[1] * head_kgrid[1])) % int(head_kgrid[1])
                qz_i = int(np.round(qf[2] * head_kgrid[2])) % int(head_kgrid[2])
                g0_mask = np.all(gvec[qi] == 0.0, axis=0)         # (ngkmax,)
                v = np.where(g0_mask, head_arr[qx_i, qy_i, qz_i], v)
        elif sys_dim == 2:
            kxy = np.sqrt(qG_cart[0]**2 + qG_cart[1]**2)
            kz = qG_cart[2]
            f2d = 1.0 - np.exp(-zc * kxy) * np.cos(kz * zc)
            v_reg = (8.0 * np.pi / denom_safe) * f2d
            v = np.where(denom_zero, 0.0, v_reg * fact)
        else:
            # sys_dim == 0: caller passes ``bdot`` and we'd build the
            # FFT-grid sqrt_v0d here; not yet wired to per-q lookup.
            raise NotImplementedError(
                "compute_v_q_per_G: sys_dim=0 path not wired — the 0-D "
                "box truncation builds v on the full FFT grid via "
                "compute_sqrt_vcoul_0d; the per-q gather would map "
                "components → flat-FFT index → v(G).  Plumb when "
                "needed.")
        if vcoul_cutoff_ry is not None:
            v = np.where(denom > float(vcoul_cutoff_ry), 0.0, v)
        out[qi] = v
    return out


def compute_all_V_q(
    zeta_io,
    *,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    mesh_xy: Mesh,
    n_rmu: int,
    n_rtot: int,
    sys_dim: int = 2,
    bdot: np.ndarray | None = None,
    mc_average_vcoul_body: bool = True,
    bare_coulomb_cutoff: float | None = None,
    bgw_v_grid_fn=None,
    mu_chunk_size: int | None = None,   # legacy arg (allgather path); ignored
    q_batch_size: int | None = None,    # legacy arg (allgather path); ignored
    budget_bytes: float | None = None,
    verbose: bool = True,
    sym=None,
    centroid_indices: np.ndarray | None = None,
    use_g_flat_zeta: bool = False,
    g_chunk_size: int = 0,              # 0 = auto _pick_g_chunk(ngkmax)
) -> tuple[jax.Array, jax.Array]:
    """Compute V_qmunu(q,μ,ν) and g0_μ(q) at q=0 from a sharded ζ HDF5.

    Dispatcher with two paths:

    * On-disk ``ζ`` in **G-flat** layout (the only thing
      :func:`fit_zeta_to_h5` writes): routes to
      :func:`gw.v_q_g_flat.compute_all_V_q_g_flat` — per-q, G-chunked
      contract on the writer's per-q WFN.h5-style sphere.  No FFT,
      no shared-sphere conversion, no μ × ν tiling; the chooser
      collapses to "pick g_chunk".

    * On-disk ``ζ`` in **r-space** layout: routes to
      :func:`gw.v_q_tile.compute_V_q_tile` — the older driver that
      handles FFT + sphere gather + μ × ν tiling inline.  Reachable
      only for legacy r-space ζ files; not exercised by the current
      writer.

    Parameters mirror the old API; ``mu_chunk_size`` / ``q_batch_size``
    are kept for back-compat with r-space callers but ignored on the
    G-flat path (where the chooser is essentially trivial).
    """
    # G-flat dispatch — when the loader carries the new per-q sphere
    # components, hand off to the rewritten orchestrator.  Async
    # prefetch defaults to OFF here: the prefetcher's collective read
    # on the PHDF5 FFI backend has to interleave correctly with the
    # per-q kernel's NCCL collectives, and a simple sync loop is
    # easier to debug.  Re-enable via env once the kernel is profile-
    # validated.
    if getattr(zeta_io, 'zeta_layout', None) == 'G_flat':
        from .v_q_g_flat import compute_all_V_q_g_flat
        _async = bool(int(os.environ.get(
            'LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH', '0')))
        return compute_all_V_q_g_flat(
            zeta_io,
            kgrid=kgrid, fft_grid=fft_grid,
            bvec=bvec, cell_volume=cell_volume,
            mesh_xy=mesh_xy,
            sys_dim=sys_dim, bdot=bdot,
            bare_coulomb_cutoff_ry=bare_coulomb_cutoff,
            bgw_v_grid_fn=bgw_v_grid_fn,
            mc_average_vcoul_body=mc_average_vcoul_body,
            g_chunk=(int(g_chunk_size) if g_chunk_size > 0 else None),
            verbose=verbose, sym=sym,
            centroid_indices=centroid_indices,
            async_prefetch=_async,
        )

    raise NotImplementedError(
        "compute_all_V_q: only the G-flat zeta layout is supported. "
        "fit_zeta_to_h5 writes G_flat exclusively; the r-space tile path "
        "was removed 2026-07-02.")
