"""
Unified V_q tile kernel — single source of truth for V_q(μ, ν) construction.

Replaces the old ``_make_V_q_caseA_kernel`` / ``_make_V_q_caseB_kernel`` pair
plus the two outer drivers ``_compute_all_V_q_sharded`` /
``_compute_all_V_q_replicated`` with one inner kernel and one outer driver.

The unified function handles every V_q tile shape that occurs in production:

    * ``same_zeta=True``  — V^{0,0} self-contraction (the scalar charge tile and
      the bispinor diagonal tiles).  One read, one FFT, two one-axis gathers.
    * ``same_zeta=False`` — bispinor V^{μ_L, ν_L} (L ≠ R) tiles.  Two reads,
      two FFTs, no g0 (head term is only meaningful for V^{0,0}).
    * Case A (μ fits in budget)  — full μ × full μ in one shot per q-batch.
    * Case B (μ-tiled)            — single q, μ × ν loop.

Coulomb dimensionality is provided externally via ``v_per_G_fn`` and
``phase_fn`` callables.  The inner kernel applies ``v_per_G`` ONCE on the L
side of the contraction:

    V_block(μ, ν) = Σ_G  conj(ζ_L(G)) · v(K) · ζ_R(G)

This works for any real-valued (signed) weight ``v(K)``; in particular the
signed transverse-projector weight ``v(K) · (-K̂_i K̂_j)`` used by the
bispinor V^{μ_L, ν_L} (L ≠ R) tiles is handled with no special casing.

I/O backends
------------
The driver reads ζ via ``SlabIO.read_slab(..., partition_spec=P(None, None,
('x','y')))``.  Both the PHDF5/FFI and h5py-allgather backends honour this
partition_spec — PHDF5 lands ζ directly in μ-on-XY layout, allgather rank-0
reads then ``device_put``s with the same sharding.  The unified kernel
sees the same input layout either way.

Synchronisation
---------------
The Python loop does NOT call ``.block_until_ready()`` on intermediate
arrays.  Each kernel call is async-dispatched and returns lazily; the next
iteration's ``read_slab`` runs on the host thread while the GPU finishes
the previous iteration's compute, so PHDF5 collective reads naturally
overlap with V_q compute.
"""

import os
from functools import partial
from math import gcd

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import make_sharded_fftn_3d


# ----------------------------------------------------------------------------
# Memory-model chooser
# ----------------------------------------------------------------------------
#
# Chooses (q_chunk, μ_chunk) given a per-rank budget.  See the big comment in
# ``compute_vcoul.py`` (kept there for now since other modules also reference
# it) for the full derivation.  The chooser models two stages with different
# working-axis shardings:
#
#   (a) Gathered/post-sphere stage on (Q, μ, n_G)  sharded P_min     (one axis)
#   (b) FFT stage on (Q, μ, n_rtot)                sharded P_prod    (both axes)
#
# Pre-2026-05-06 only stage (a) was modelled; CrI3 4×4×6 with μ=1500 hit OOM
# at the predicted 41 GB because stage (b) actually controls when n_rtot
# >> n_G.  Take ``max(a, b)`` to size q_chunk.
#
# The FFT stage's overlap coefficient is env-tunable for A/B testing.  The
# default 2.0 assumes the post-(reshape × phase) intermediate stays alive
# alongside the FFT input; setting ``LORRAX_V_Q_FFT_COEF=1.0`` assumes XLA
# fuses the phase multiply into the FFT input copy (no live overlap), which
# roughly doubles the achievable q_chunk on systems where (b) dominates.
_Q_COMPUTE_COEF_GATHER = 4.4
_Q_COMPUTE_COEF_FFT = float(os.environ.get('LORRAX_V_Q_FFT_COEF', '2.0'))


def _choose_v_q_chunks(
    *,
    n_rmu: int,
    n_G: int,
    n_q_total: int,
    budget_bytes: float,
    p_x: int,
    p_y: int,
    n_rtot: int | None = None,
) -> dict:
    """Pick (q_chunk, μ_chunk) per the memory model documented above.

    Returns a dict with keys:
        q_chunk       : int — q-points per kernel call
        mu_chunk      : int — μ block size (== n_rmu in Case A)
        n_mu_blocks   : int — number of μ blocks in the tile loop (1 in A)
        tiled         : bool — True ⟺ Case B (μ × ν tiling)
        per_rank_peak : float — predicted per-rank peak bytes
        ref_bytes     : float — bytes reserved for V_ref + g0_ref per rank
        aligned       : bool — True ⟺ μ_chunk = N_μ/(P_x·k); ref slice-set
                              lands fully inside one rank's V_ref shard
                              (no collective on update)
    """
    N_zeta = 16.0  # c128
    p_min = float(min(int(p_x), int(p_y)))
    p_prod = float(int(p_x) * int(p_y))

    # Accumulator reservation (V_ref + g0_ref, per-rank sharded bytes).
    v_ref_bytes  = N_zeta * n_q_total * n_rmu * n_rmu / p_prod
    g0_ref_bytes = N_zeta * n_q_total * n_rmu / max(1, int(p_x))
    ref_bytes = v_ref_bytes + g0_ref_bytes
    B_compute = float(budget_bytes) - ref_bytes

    v_per_q_bytes = N_zeta * n_G  # sqrt_v(q+G) replicated per rank

    one_q_gather = _Q_COMPUTE_COEF_GATHER * N_zeta * n_rmu * n_G / p_min
    if n_rtot is not None and n_rtot > 0:
        one_q_fft = _Q_COMPUTE_COEF_FFT * N_zeta * n_rmu * float(n_rtot) / p_prod
    else:
        one_q_fft = 0.0
    one_q_bytes = max(one_q_gather, one_q_fft)

    # Case A check: can we hold *one* q's worth (whichever stage dominates)?
    slack_after_one_q = B_compute - (one_q_bytes + v_per_q_bytes)
    if slack_after_one_q < 0:
        # Case B — single q, tile μ × ν.
        mu_chunk_max_gather = int(
            (B_compute - v_per_q_bytes) * p_min / (2.0 * N_zeta * n_G))
        if n_rtot is not None and n_rtot > 0:
            mu_chunk_max_fft = int(
                (B_compute - v_per_q_bytes) * p_prod /
                (2.0 * N_zeta * float(n_rtot)))
        else:
            mu_chunk_max_fft = mu_chunk_max_gather
        mu_chunk_max = max(1, min(mu_chunk_max_gather, mu_chunk_max_fft))
        mu_chunk_max = min(int(n_rmu), int(mu_chunk_max))

        # CONSERVATIVE CHOICE of μ_chunk — must simultaneously satisfy:
        #   (i)  μ_chunk divides N_μ/P_x  AND  μ_chunk divides N_μ/P_y
        #   (ii) μ_chunk is divisible by P_x·P_y
        # Both hold simultaneously when μ_chunk divides gcd(N_μ/P_x,
        # N_μ/P_y) AND is a multiple of P_x·P_y.
        mu_per_x = int(n_rmu) // int(p_x) if int(n_rmu) % int(p_x) == 0 else 0
        mu_per_y = int(n_rmu) // int(p_y) if int(n_rmu) % int(p_y) == 0 else 0
        aligned_parent = gcd(mu_per_x, mu_per_y) if (mu_per_x and mu_per_y) else 0
        snap = int(p_x) * int(p_y) or 1

        aligned = False
        mu_chunk = None
        if aligned_parent and aligned_parent % snap == 0:
            candidates = sorted(
                (d for d in range(snap, aligned_parent + 1, snap)
                 if aligned_parent % d == 0 and d <= mu_chunk_max),
                reverse=True)
            if candidates:
                mu_chunk = candidates[0]
                aligned = True
        if mu_chunk is None:
            mu_chunk = max(snap, mu_chunk_max - (mu_chunk_max % snap))

        n_mu_blocks = (int(n_rmu) + mu_chunk - 1) // mu_chunk
        peak_gather = 2.0 * N_zeta * mu_chunk * n_G / p_min
        if n_rtot is not None and n_rtot > 0:
            peak_fft = 2.0 * N_zeta * mu_chunk * float(n_rtot) / p_prod
        else:
            peak_fft = 0.0
        peak = max(peak_gather, peak_fft) + v_per_q_bytes + ref_bytes
        return dict(
            q_chunk=1, mu_chunk=int(mu_chunk), n_mu_blocks=int(n_mu_blocks),
            tiled=True, aligned=aligned,
            per_rank_peak=peak, ref_bytes=ref_bytes,
        )

    # Case A — fit at least one q with full μ.  Maximise q_chunk in B_compute.
    q_max = int((B_compute - v_per_q_bytes * max(1, n_q_total)) /
                one_q_bytes) if one_q_bytes > 0 else n_q_total
    q_chunk = max(1, min(n_q_total, q_max))
    peak = q_chunk * one_q_bytes + q_chunk * v_per_q_bytes + ref_bytes
    return dict(
        q_chunk=q_chunk, mu_chunk=int(n_rmu), n_mu_blocks=1,
        tiled=False, aligned=True,
        per_rank_peak=peak, ref_bytes=ref_bytes,
    )


# ----------------------------------------------------------------------------
# Unified inner V_q tile kernel
# ----------------------------------------------------------------------------

_v_q_tile_kernel_cache: dict = {}


def _make_V_q_tile_kernel(
    *,
    sphere_idx,
    n_G_sph: int,
    fft_shape: tuple[int, int, int],
    mesh_xy: Mesh,
    q_chunk: int,
    mu_size: int,
    nu_size: int,
    n_rmu_L: int,
    n_rmu_R: int,
    same_zeta: bool,
    write_g0: bool,
):
    """Build the unified inner V_q kernel.

    Parameters
    ----------
    sphere_idx : jax.Array | None
        G-sphere indices into the flat FFT box (or None when no cutoff).
    n_G_sph : int
        Post-cutoff G-vector count (== prod(fft_shape) when sphere_idx is None).
    fft_shape : (nx, ny, nz)
        FFT-box dims.
    mesh_xy : Mesh
        2-D device mesh with axes ('x', 'y').
    q_chunk : int
        q-points per kernel call (Case A: q-batch size; Case B: 1).
    mu_size, nu_size : int
        Per-call block sizes (cache key).  In Case A both equal n_rmu_L /
        n_rmu_R; in Case B they're the chunked sizes (with min(mu_chunk,
        n_rmu - offset) for the last partial block).
    n_rmu_L, n_rmu_R : int
        Total μ counts on the left / right ζ — set the V_acc accumulator
        extents.  Equal for V^{0,0}; may differ for bispinor.
    same_zeta : bool (static)
        True → one ζ input, one FFT, two gathers.  False → two ζ inputs,
        two FFTs, two gathers.
    write_g0 : bool (static)
        Only valid when same_zeta and (μ_lo == ν_lo).  Diagonal blocks
        write ζ_μ(G=0) into the g0 accumulator.

    Returns
    -------
    A jit-compiled kernel with signature::

        kernel(V_acc, g0_acc,
               zeta_mu_disk, [zeta_nu_disk,]   # second arg only when not same_zeta
               v_per_G_batch, phase_batch,
               q_lo_dyn, mu_lo_dyn, nu_lo_dyn) -> (V_acc', g0_acc')

    plus the convenience attributes ``zeta_disk_sh``, ``V_sh``, ``g0_sh``.
    """
    cache_key = (
        'unified', id(mesh_xy), q_chunk, mu_size, nu_size,
        n_rmu_L, n_rmu_R, n_G_sph, tuple(fft_shape),
        id(sphere_idx), bool(same_zeta), bool(write_g0),
    )
    hit = _v_q_tile_kernel_cache.get(cache_key)
    if hit is not None:
        return hit

    nx, ny, nz = fft_shape
    n_rtot = nx * ny * nz

    # Sharding specs.  These match the chi0 / CCT / ZCT / sigma chain
    # downstream: V_block lands natively in P(None, 'x', 'y').
    blk_xy_sh = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
    blk_xy_5d_spec = P(None, ('x', 'y'), None, None, None)
    blk_xy_5d_sh = NamedSharding(mesh_xy, blk_xy_5d_spec)
    blk_x_sh = NamedSharding(mesh_xy, P(None, 'x', None))
    blk_y_sh = NamedSharding(mesh_xy, P(None, 'y', None))
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))
    phase_sh = NamedSharding(mesh_xy, P(None, None, None, None, None))  # (Q,1,nx,ny,nz)
    v_per_G_sh = NamedSharding(mesh_xy, P(None, None))                    # (Q, n_G_sph)
    zeta_disk_sh = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))
    rep = NamedSharding(mesh_xy, P())

    _local_fftn_3d = make_sharded_fftn_3d(
        mesh_xy, blk_xy_5d_spec, blk_xy_5d_spec)

    def _fft_and_sphere(zeta_rtot_mu, phase_batch):
        """Local transpose + 3-D FFT + sphere pick (NO v multiply).

        Returns (zeta_G, g0_blk).  ``g0_blk`` is the G=(0,0,0) column
        (shape (Q, μ_per_rank), μ-XY-sharded).  ``zeta_G`` is the
        sphere-gathered slab (shape (Q, μ_per_rank, n_G_sph), μ-XY-sharded).
        """
        zeta_mu_r = jax.lax.with_sharding_constraint(
            jnp.transpose(zeta_rtot_mu, (0, 2, 1)), blk_xy_sh)
        Q, mu_per_rank, _ = zeta_mu_r.shape
        zeta_5d = jax.lax.with_sharding_constraint(
            zeta_mu_r.reshape(Q, mu_per_rank, nx, ny, nz) * phase_batch,
            blk_xy_5d_sh)
        zeta_box = _local_fftn_3d(zeta_5d)
        zeta_box = jax.lax.with_sharding_constraint(
            zeta_box.reshape(Q, mu_per_rank, n_rtot), blk_xy_sh)
        g0_blk = zeta_box[:, :, 0]
        if sphere_idx is not None:
            zeta_G = jnp.take(zeta_box, sphere_idx, axis=-1)
        else:
            zeta_G = zeta_box
        zeta_G = jax.lax.with_sharding_constraint(zeta_G, blk_xy_sh)
        return zeta_G, g0_blk

    if same_zeta:
        # Single ζ input — one FFT, two gathers from the same source.
        @partial(jax.jit,
                 in_shardings=(V_sh, g0_sh, zeta_disk_sh,
                               v_per_G_sh, phase_sh, rep, rep, rep),
                 out_shardings=(V_sh, g0_sh),
                 donate_argnums=(0, 1))
        def _kernel(V_acc, g0_acc,
                    zeta_mu_disk,
                    v_per_G_batch, phase_batch,
                    q_lo_dyn, mu_lo_dyn, nu_lo_dyn):
            zeta_G, g0_mu = _fft_and_sphere(zeta_mu_disk, phase_batch)

            zeta_mu_X = jax.lax.with_sharding_constraint(zeta_G, blk_x_sh)
            zeta_nu_Y = jax.lax.with_sharding_constraint(zeta_G, blk_y_sh)

            # One-sided v(K) multiply on the L (μ) side — handles signed
            # transverse-projector weights for bispinor (off-diagonal call
            # sites; here it's just v(q+G) for V^{0,0}).
            zeta_mu_X = jax.lax.with_sharding_constraint(
                zeta_mu_X * v_per_G_batch[:, None, :], blk_x_sh)

            V_block = jnp.einsum('qmG,qnG->qmn',
                                 jnp.conj(zeta_mu_X), zeta_nu_Y,
                                 optimize=True)
            V_block = jax.lax.with_sharding_constraint(V_block, V_sh)

            V_new = jax.lax.dynamic_update_slice(
                V_acc, V_block, (q_lo_dyn, mu_lo_dyn, nu_lo_dyn))
            V_new = jax.lax.with_sharding_constraint(V_new, V_sh)

            if write_g0:
                g0_mu = jax.lax.with_sharding_constraint(g0_mu, g0_sh)
                g0_new = jax.lax.dynamic_update_slice(
                    g0_acc, g0_mu, (q_lo_dyn, mu_lo_dyn))
                g0_new = jax.lax.with_sharding_constraint(g0_new, g0_sh)
            else:
                g0_new = g0_acc
            return V_new, g0_new

    else:
        assert not write_g0, (
            "write_g0=True is only valid when same_zeta=True (V^{0,0} "
            "self-contraction); the bispinor V^{μ_L, ν_L} (L ≠ R) head "
            "term is not defined.")

        @partial(jax.jit,
                 in_shardings=(V_sh, g0_sh, zeta_disk_sh, zeta_disk_sh,
                               v_per_G_sh, phase_sh, rep, rep, rep),
                 out_shardings=(V_sh, g0_sh),
                 donate_argnums=(0, 1))
        def _kernel(V_acc, g0_acc,
                    zeta_mu_disk, zeta_nu_disk,
                    v_per_G_batch, phase_batch,
                    q_lo_dyn, mu_lo_dyn, nu_lo_dyn):
            zeta_mu_G, _ = _fft_and_sphere(zeta_mu_disk, phase_batch)
            zeta_nu_G, _ = _fft_and_sphere(zeta_nu_disk, phase_batch)

            zeta_mu_X = jax.lax.with_sharding_constraint(zeta_mu_G, blk_x_sh)
            zeta_nu_Y = jax.lax.with_sharding_constraint(zeta_nu_G, blk_y_sh)

            zeta_mu_X = jax.lax.with_sharding_constraint(
                zeta_mu_X * v_per_G_batch[:, None, :], blk_x_sh)

            V_block = jnp.einsum('qmG,qnG->qmn',
                                 jnp.conj(zeta_mu_X), zeta_nu_Y,
                                 optimize=True)
            V_block = jax.lax.with_sharding_constraint(V_block, V_sh)

            V_new = jax.lax.dynamic_update_slice(
                V_acc, V_block, (q_lo_dyn, mu_lo_dyn, nu_lo_dyn))
            V_new = jax.lax.with_sharding_constraint(V_new, V_sh)
            return V_new, g0_acc

    _kernel.zeta_disk_sh = zeta_disk_sh
    _kernel.V_sh = V_sh
    _kernel.g0_sh = g0_sh
    _v_q_tile_kernel_cache[cache_key] = _kernel
    return _kernel


def _make_g0_dummy(mesh_xy, nq_total, n_rmu_L):
    """Sharded dummy g0 accumulator for tiles without a head.

    The unified kernel always takes a g0_acc argument (it's in
    ``donate_argnums``); when the caller doesn't want the head, we carry
    one dummy zero slab through the loop so donation ownership stays
    explicit without per-block allocation.
    """
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))

    @partial(jax.jit, out_shardings=g0_sh)
    def _zeros():
        return jnp.zeros((nq_total, n_rmu_L), dtype=jnp.complex128)

    return _zeros()


# ----------------------------------------------------------------------------
# Outer driver — single function for every V_q tile shape and both backends
# ----------------------------------------------------------------------------


def compute_V_q_tile(
    *,
    zeta_L_io,
    zeta_R_io=None,
    v_per_G_fn,
    phase_fn,
    sphere_idx,
    fft_grid: tuple[int, int, int],
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
    n_rmu_L: int,
    n_rmu_R: int | None = None,
    V_acc=None,
    g0_acc=None,
    chooser_choice: dict | None = None,
    budget_bytes: float | None = None,
    bgw_v_grid_overlay_fn=None,
    verbose: bool = True,
    timing_label: str = "compute_V_q_tile",
):
    """Outer driver for the unified V_q tile.

    Single function; covers:
      * V^{0,0} (same_zeta=True) and bispinor V^{μ_L, ν_L} (same_zeta=False)
      * Case A (q-batched) and Case B (μ × ν tiled)
      * PHDF5/FFI and h5py-allgather backends (dispatched inside SlabIO)

    Parameters
    ----------
    zeta_L_io, zeta_R_io : SlabIO
        Open SlabIO handles in mode='r'.  When ``zeta_R_io is None`` (or
        ``zeta_R_io is zeta_L_io``) the kernel runs in single-ζ mode (one
        FFT, two gathers) — the V^{0,0} self-contraction.
    v_per_G_fn : callable((Q,3) qvec_np) -> jnp.ndarray (Q, n_G_sph) c128
        Returns the per-G weight applied on the L side of the contraction.
        Any real-valued (signed) weight allowed; for V^{0,0} this is
        ``v(q+G)`` (real, ≥ 0) and the math is bit-identical to the old
        symmetric-√v form.  For bispinor V^{μ_L, ν_L} it is
        ``v(K) · t_{μ_L, ν_L}(K)`` with signed transverse-projector ``t``.
    phase_fn : callable((Q,3) qvec_np) -> jnp.ndarray (Q, 1, nx, ny, nz) c128
        Per-q FFT-box phase factor ``exp(-2πi q·r)``.  Independent of the
        v side (factored out so any Coulomb dimensionality plugs in).
    sphere_idx : jax.Array | None
        G-sphere indices into the flat FFT box, or None when no cutoff.
    fft_grid : (nx, ny, nz)
    mesh_xy : Mesh
        2-D device mesh with axes ('x', 'y').
    kgrid : (nkx, nky, nkz)
    n_rmu_L, n_rmu_R : int
        Total μ counts.  ``n_rmu_R`` defaults to ``n_rmu_L`` (V^{0,0}).
    V_acc : jnp.ndarray (n_q_total, n_rmu_L, n_rmu_R), P(None,'x','y')
        Donated mutable accumulator.  Caller pre-allocates with
        ``jnp.zeros`` of the right layout, OR pass None to have this
        function allocate it.
    g0_acc : jnp.ndarray | None (n_q_total, n_rmu_L), P(None, 'x')
        Donated g0 accumulator — only used when ``same_zeta`` and the
        caller wants the head term.  Pass None for bispinor off-diagonal
        tiles or when the caller doesn't need g0.
    chooser_choice : dict | None
        Output of :func:`_choose_v_q_chunks`.  If None, the chooser is
        run with ``budget_bytes`` (mandatory in that case).
    budget_bytes : float | None
        Memory budget per device, bytes.  Required when chooser_choice is None.
    bgw_v_grid_overlay_fn : callable | None
        Optional host-side overlay applied to ``v_per_G`` before passing
        to the kernel.  Signature::
            f(qvec_np_batch, v_per_G_native_np) -> v_per_G_overlaid_np
        Used for byte-reproducible BGW comparisons (``use_bgw_vcoul=true``).
    verbose : bool
    timing_label : str

    Returns
    -------
    (V_acc, g0_acc_or_None)
        Both buffers in their final filled state.  When ``g0_acc`` was None
        on entry, returns ``(V_acc, None)``.

    Notes
    -----
    * The chooser's ``aligned`` knob and the (mu_size, nu_size) cache key
      give the standard 4-compile-shape envelope (or 8 with write_g0).  In
      the chooser-aligned case all interior iterations share one shape.
    * The ``LORRAX_V_Q_MU_CHUNK`` env override is honoured by the dispatcher
      ``compute_all_V_q`` before this function — this driver does not
      consult env vars.
    """
    from common import timing
    from common.progress import LoopProgress

    nkx, nky, nkz = kgrid
    nq_total = nkx * nky * nkz
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])

    n_rtot = int(np.prod(fft_grid))
    if n_rmu_R is None:
        n_rmu_R = n_rmu_L

    same_zeta = (zeta_R_io is None) or (zeta_R_io is zeta_L_io)

    if same_zeta and n_rmu_L != n_rmu_R:
        raise ValueError(
            f"compute_V_q_tile: same_zeta=True requires n_rmu_L == n_rmu_R, "
            f"got {n_rmu_L} vs {n_rmu_R}.")

    if sphere_idx is not None:
        n_G_sph = int(np.asarray(sphere_idx).shape[0])
    else:
        n_G_sph = fft_grid[0] * fft_grid[1] * fft_grid[2]

    if chooser_choice is None:
        if budget_bytes is None:
            raise ValueError(
                "compute_V_q_tile: must pass either chooser_choice or "
                "budget_bytes.")
        chooser_choice = _choose_v_q_chunks(
            n_rmu=max(n_rmu_L, n_rmu_R), n_G=n_G_sph, n_q_total=nq_total,
            budget_bytes=budget_bytes, p_x=p_x, p_y=p_y,
            n_rtot=n_rtot,
        )

    tiled = bool(chooser_choice['tiled'])
    q_chunk = int(chooser_choice['q_chunk'])
    mu_chunk = int(chooser_choice['mu_chunk'])
    n_mu_blocks_L = int(chooser_choice.get('n_mu_blocks_L',
                                            chooser_choice['n_mu_blocks']))
    n_mu_blocks_R = int(chooser_choice.get('n_mu_blocks_R',
                                            chooser_choice['n_mu_blocks']))

    if verbose and jax.process_index() == 0:
        kind = 'tiled (Case B)' if tiled else 'one-shot (Case A)'
        z_kind = 'same ζ' if same_zeta else 'distinct ζ_L/ζ_R'
        print(f"  V_q tile: mesh={p_x}x{p_y}, {kind}, {z_kind}, "
              f"q_chunk={q_chunk}, μ_chunk={mu_chunk} "
              f"({n_mu_blocks_L}×{n_mu_blocks_R} blocks), "
              f"aligned={chooser_choice.get('aligned', False)}, "
              f"N_μ_L={n_rmu_L}, N_μ_R={n_rmu_R}, N_G={n_G_sph}, "
              f"predicted peak/rank={chooser_choice['per_rank_peak']/1e9:.2f} GB "
              f"(V_ref+g0_ref={chooser_choice['ref_bytes']/1e9:.2f} GB)")

    # Allocate accumulators if caller didn't.
    V_sh_full = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    g0_sh_full = NamedSharding(mesh_xy, P(None, 'x'))
    if V_acc is None:
        @partial(jax.jit, out_shardings=V_sh_full)
        def _init_V():
            return jnp.zeros((nq_total, n_rmu_L, n_rmu_R), dtype=jnp.complex128)
        V_acc = _init_V()

    wants_g0 = g0_acc is not None
    if g0_acc is None and same_zeta:
        # Caller didn't pre-allocate but we'll let them know by allocating
        # and returning — the head term defaults to "wanted" for V^{0,0}.
        @partial(jax.jit, out_shardings=g0_sh_full)
        def _init_g0():
            return jnp.zeros((nq_total, n_rmu_L), dtype=jnp.complex128)
        g0_acc = _init_g0()
        wants_g0 = True
    g0_work = g0_acc if wants_g0 else _make_g0_dummy(
        mesh_xy, nq_total, n_rmu_L)

    kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)
    _read_spec = P(None, None, ('x', 'y'))

    def _qvec_wrap(qx, qy, qz):
        qvec = np.array([qx, qy, qz], dtype=np.float64)
        return np.where(qvec > kgrid_arr / 2, qvec - kgrid_arr, qvec)

    def _v_phase_batch(qvec_list, pad_to: int):
        """Build (v_per_G, phase) for a batch of q's, with optional pad
        and optional BGW vcoul overlay.  Both outputs are jax arrays.
        """
        n_actual = len(qvec_list)
        qvec_np = np.stack([np.asarray(qw, dtype=np.float64)
                            for qw in qvec_list], axis=0)
        if n_actual < pad_to:
            pad = np.tile(qvec_np[0:1], (pad_to - n_actual, 1))
            qvec_np = np.concatenate([qvec_np, pad], axis=0)
        v_per_G = v_per_G_fn(qvec_np)        # (Q, n_G_sph) c128
        phase = phase_fn(qvec_np)            # (Q, 1, nx, ny, nz) c128
        if bgw_v_grid_overlay_fn is not None:
            v_per_G = bgw_v_grid_overlay_fn(qvec_np, v_per_G)
        return v_per_G, phase

    vq_progress = LoopProgress(
        nq_total, print, title="V_q tile",
        item_name="q-point", max_updates=min(nq_total, 20))

    # ζ read uses ``zeta_io.read_slab(..., partition_spec=_read_spec)``.
    # PHDF5 backend lands the slab directly in μ-on-XY layout; allgather
    # backend rank-0 reads then ``device_put``s with the same sharding.
    # Either way the kernel input is properly sharded.
    if not tiled:
        # === Case A — one read per q-batch.
        q_coords = [(qx, qy, qz) for qx in range(nkx)
                    for qy in range(nky) for qz in range(nkz)]
        batches = [q_coords[i:i + q_chunk]
                   for i in range(0, nq_total, q_chunk)]

        with timing.section(timing_label):
            q_cursor = 0
            for batch in batches:
                actual = len(batch)
                kernel = _make_V_q_tile_kernel(
                    sphere_idx=sphere_idx, n_G_sph=n_G_sph,
                    fft_shape=fft_grid, mesh_xy=mesh_xy,
                    q_chunk=actual, mu_size=n_rmu_L, nu_size=n_rmu_R,
                    n_rmu_L=n_rmu_L, n_rmu_R=n_rmu_R,
                    same_zeta=same_zeta,
                    write_g0=(same_zeta and wants_g0))
                qvecs = [_qvec_wrap(*c) for c in batch]
                q_flat0 = batch[0][0] * (nky * nkz) + batch[0][1] * nkz + batch[0][2]

                zeta_L = zeta_L_io.read_slab(
                    'zeta_q',
                    shape=(actual, n_rtot, n_rmu_L),
                    dtype=np.complex128,
                    offset=(q_flat0, 0, 0),
                    mesh=mesh_xy, partition_spec=_read_spec)
                v_per_G_b, phase_b = _v_phase_batch(qvecs, pad_to=actual)
                if same_zeta:
                    V_acc, g0_work = kernel(
                        V_acc, g0_work,
                        zeta_L, v_per_G_b, phase_b,
                        jnp.int32(q_cursor), jnp.int32(0), jnp.int32(0))
                else:
                    zeta_R = zeta_R_io.read_slab(
                        'zeta_q',
                        shape=(actual, n_rtot, n_rmu_R),
                        dtype=np.complex128,
                        offset=(q_flat0, 0, 0),
                        mesh=mesh_xy, partition_spec=_read_spec)
                    V_acc, g0_work = kernel(
                        V_acc, g0_work,
                        zeta_L, zeta_R, v_per_G_b, phase_b,
                        jnp.int32(q_cursor), jnp.int32(0), jnp.int32(0))
                    del zeta_R
                del zeta_L
                q_cursor += actual
                for _ in range(actual):
                    vq_progress.step()
            vq_progress.finish()

    else:
        # === Case B — single q, μ × ν tile loop.  Diagonal blocks reuse
        # the ζ_μ slab as ζ_ν (single-FFT, on_diag path); off-diagonal
        # blocks read both.  ``write_g0`` is static per kernel.
        with timing.section(timing_label):
            for qx in range(nkx):
              for qy in range(nky):
                for qz in range(nkz):
                    q_flat = qx * (nky * nkz) + qy * nkz + qz
                    qvec_wrapped = _qvec_wrap(qx, qy, qz)
                    v_per_G_b, phase_b = _v_phase_batch(
                        [qvec_wrapped], pad_to=1)

                    for mu_i in range(n_mu_blocks_L):
                        mu_lo = mu_i * mu_chunk
                        mu_size = min(mu_chunk, n_rmu_L - mu_lo)
                        zeta_mu = zeta_L_io.read_slab(
                            'zeta_q',
                            shape=(1, n_rtot, mu_size),
                            dtype=np.complex128,
                            offset=(q_flat, 0, mu_lo),
                            mesh=mesh_xy, partition_spec=_read_spec)
                        for nu_j in range(n_mu_blocks_R):
                            nu_lo = nu_j * mu_chunk
                            nu_size = min(mu_chunk, n_rmu_R - nu_lo)
                            on_diag = (
                                same_zeta and mu_i == nu_j and
                                mu_lo == nu_lo and mu_size == nu_size
                            )
                            do_write_g0 = (
                                same_zeta and wants_g0 and on_diag
                            )
                            if on_diag:
                                k = _make_V_q_tile_kernel(
                                    sphere_idx=sphere_idx, n_G_sph=n_G_sph,
                                    fft_shape=fft_grid, mesh_xy=mesh_xy,
                                    q_chunk=1, mu_size=mu_size, nu_size=nu_size,
                                    n_rmu_L=n_rmu_L, n_rmu_R=n_rmu_R,
                                    same_zeta=True, write_g0=do_write_g0)
                                V_acc, g0_work = k(
                                    V_acc, g0_work,
                                    zeta_mu, v_per_G_b, phase_b,
                                    jnp.int32(q_flat),
                                    jnp.int32(mu_lo), jnp.int32(nu_lo))
                            else:
                                zeta_nu = zeta_R_io.read_slab(
                                    'zeta_q',
                                    shape=(1, n_rtot, nu_size),
                                    dtype=np.complex128,
                                    offset=(q_flat, 0, nu_lo),
                                    mesh=mesh_xy, partition_spec=_read_spec)
                                k = _make_V_q_tile_kernel(
                                    sphere_idx=sphere_idx, n_G_sph=n_G_sph,
                                    fft_shape=fft_grid, mesh_xy=mesh_xy,
                                    q_chunk=1, mu_size=mu_size, nu_size=nu_size,
                                    n_rmu_L=n_rmu_L, n_rmu_R=n_rmu_R,
                                    same_zeta=False, write_g0=False)
                                V_acc, g0_work = k(
                                    V_acc, g0_work,
                                    zeta_mu, zeta_nu, v_per_G_b, phase_b,
                                    jnp.int32(q_flat),
                                    jnp.int32(mu_lo), jnp.int32(nu_lo))
                                del zeta_nu
                        del zeta_mu
                    vq_progress.step()
            vq_progress.finish()

    return V_acc, (g0_work if wants_g0 else None)
