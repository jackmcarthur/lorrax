"""
Unified V_q tile kernel — replaces ``_make_V_q_caseA_kernel`` and
``_make_V_q_caseB_kernel`` with a single inner kernel that handles both:

    * Case A — μ fits in budget, full μ × full μ in one shot per q-batch.
    * Case B — single q, μ × ν tiled over (μ_block, ν_block).

The kernel also covers the upcoming bispinor V^{μ_L, ν_L}_q tile, where the
left and right ζ files differ:

    * V^{0,0}: ``zeta_L_io is zeta_R_io`` — one read, one FFT (the perf reason
      Case A existed in the old code).  ``g0`` is meaningful and written
      into a pre-allocated accumulator.
    * V^{μ_L, ν_L} (L ≠ R): two reads, two FFTs, no g0 (the head term is only
      meaningful for the V^{0,0} self-contraction).

Public surface:

    * ``compute_V_q_tile``  — outer per-tile driver (q-batch loop + per-tile
      μ × ν loop), called once per (μ_L, ν_L) tile by the bispinor driver,
      and once for V^{0,0} by ``v_q_driver.compute_all_V_q_sharded``.
    * ``_choose_v_q_chunks`` — memory-model chooser (Case A vs Case B + the
      μ_chunk size).  See the big comment in this file for the model.

The chooser sizes the tile so that, in Case A, the inner kernel runs once
per q-batch with ``mu_size = nu_size = N_μ``; in Case B, with
``mu_size = nu_size ≤ μ_chunk`` and the per-tile loop iterates
``n_mu_blocks²`` times per q.  The 4-compile-shape envelope (full/tail ×
full/tail, with/without write_g0) of the old Case-B kernel is preserved
automatically via the cache key on ``(mu_size, nu_size, write_g0,
same_zeta)``.

Why a single kernel?
--------------------
The old Case A and Case B kernels duplicated FFT + sphere + √v + gather +
einsum + DUS write logic.  Collapsing them lets the bispinor V^{μ_L, ν_L}
driver use the same primitive — Case A is just the ``mu_size = nu_size =
N_μ`` instantiation with ``same_zeta=True``, Case B is the
``mu_size, nu_size ≤ μ_chunk`` instantiation, and bispinor off-diagonal
tiles are ``same_zeta=False`` instantiations.  The 5-D μ-XY-sharded FFT
helper, the two one-axis gathers, the local gemm, and the DUS write are
shared verbatim — no per-case forks.

API note (v_per_G_fn — one-sided weight)
-----------------------------------------
The kernel takes ``v_per_G_batch`` of shape ``(Q, n_G_sph)`` c128 and
applies it ONCE on the left side of the contraction:

    V_block(μ, ν) = Σ_G  conj(ζ_L(G) · v(K))  ζ_R(G)
                  = Σ_G  conj(ζ_L(G))  v*(K)  ζ_R(G)

For the V^{0,0} self-contraction the caller passes a real, non-negative
``v(q+G)`` and the result is mathematically identical to the
symmetric-√v form ``conj(√v ζ_L) · √v ζ_R`` (the half-sqrt is just
factored to the L side).  For the bispinor V^{μ_L, ν_L} off-diagonal
tiles (``μ_L ≠ ν_L``) the projector ``v(K) · (-K̂_i K̂_j)`` is signed and
non-PSD per-G, but the one-sided multiply still yields the correct
real V_block (because ``v`` is real, ``conj(v) == v`` and the einsum
reduces to ``Σ_G v(K) · conj(ζ_L) · ζ_R``).

The old symmetric-√v form would have crashed (``sqrt`` of a negative)
or silently produced garbage on the off-diagonal projector blocks; the
one-sided form is correct for any real-valued (signed) weight.
"""

from functools import partial
from math import gcd

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import make_sharded_fftn_3d


# ----------------------------------------------------------------------------
# Memory-model chooser for V_q tile sizing
# ----------------------------------------------------------------------------
#
# Verbatim from the old ``compute_vcoul._choose_v_q_chunks``.  See the big
# docstring at the top of this file for context; the inline notes cover the
# Case A / Case B accounting.

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
    """Pick (q_chunk, μ_chunk) per the memory model documented in the
    long Design comment in ``v_q_driver``.

    Subtracts the V_ref/g0_ref accumulator bytes from the total budget
    FIRST, then sizes the compute-transient portion of the budget.

    Parameters
    ----------
    n_rmu, n_G, n_q_total : int
        Centroid count, sphere-G count (post-cutoff), kgrid prod.
    n_rtot : int | None
        Full real-space grid size = ``nx*ny*nz``.  When provided, the
        chooser adds an n_rtot-shaped term to the per-q footprint to
        account for the disk-read ζ slab that lives concurrently with
        the post-FFT G-sphere slab.  Setting ``n_rtot=None`` recovers
        the legacy n_G-only model (used by callers that haven't been
        plumbed yet).

    Returns a dict with keys:
        q_chunk       : int — number of q per sharded-compute call
        mu_chunk      : int — μ block size (== n_rmu in Case A, smaller
                              and (P_x·P_y)-divisible in Case B)
        n_mu_blocks   : int — number of μ blocks in the tile loop (1 in A)
        tiled         : bool — True ⟺ Case B (μ×ν tiling)
        per_rank_peak : float — predicted per-rank peak bytes at this choice
        ref_bytes     : float — bytes reserved for V_ref + g0_ref per rank
        aligned       : bool — True ⟺ μ_chunk = N_μ/(P_x·k), i.e. ref
                              slice-set lands fully inside one rank's
                              V_ref shard (no collective on update)
    """
    N_zeta = 16.0  # c128
    p_min = float(min(int(p_x), int(p_y)))
    p_prod = float(int(p_x) * int(p_y))

    # --- Accumulator reservation (V_ref + g0_ref, per-rank sharded bytes).
    v_ref_bytes  = N_zeta * n_q_total * n_rmu * n_rmu / p_prod
    g0_ref_bytes = N_zeta * n_q_total * n_rmu / max(1, int(p_x))
    ref_bytes = v_ref_bytes + g0_ref_bytes
    B_compute = float(budget_bytes) - ref_bytes

    # sqrt_v(q+G) table lives replicated on each rank.
    v_per_q_bytes = N_zeta * n_G

    # Per-q compute footprint, in two parts:
    #   (a) post-FFT G-sphere slab (μ × n_G / p_min) — the working set
    #       of the contraction kernel.  Coefficient ``Q_COMPUTE_COEF``
    #       absorbs FFT-side transients on top of the nominal
    #       2 × ζ_X + ζ_Y replicas.
    #   (b) pre-FFT n_rtot disk-read slab (μ × n_rtot / p_prod, sharded
    #       on ('x','y')).  Lives concurrently with (a) for at least
    #       one buffer's worth — XLA pipelines these — so we add the
    #       worst-case single buffer.
    # Empirical AOT measurement across (MoS2, Si 10³) × (4, 8, 16, 28,
    # 35 GB) showed Q_COMPUTE_COEF ≈ 4.4 matches the peak to within
    # ~5% when n_rtot ≈ n_G (older Si runs); on MoS2 / CrI3 where
    # n_rtot/n_G ≈ 8–16× the legacy model under-predicted because the
    # disk-read slab dominates the post-FFT slab.
    Q_COMPUTE_COEF = 4.4
    if n_rtot is None or int(n_rtot) <= 0:
        rtot_per_q_bytes = 0.0
    else:
        # μ × n_rtot / p_prod, single concurrent disk-read buffer.
        rtot_per_q_bytes = N_zeta * n_rmu * float(n_rtot) / p_prod
    # Case A check: can we hold *one* q's worth of gathered μ×G?
    one_q_bytes = Q_COMPUTE_COEF * N_zeta * n_rmu * n_G / p_min + rtot_per_q_bytes
    slack_after_one_q = B_compute - (one_q_bytes + v_per_q_bytes)
    if slack_after_one_q < 0:
        # Case B — single q, tile μ×ν.
        # Two concurrent gathered slabs of (μ_chunk × N_G) plus a small
        # contract workspace:  2·N_zeta·μ_chunk·N_G / p_min ≤ B_compute − v_per_q
        # In Case B the disk-read slab is per-tile (size μ_chunk × n_rtot
        # / p_prod) which is small for the tile sizes we expect, so the
        # n_rtot term doesn't drive μ_chunk down here.  We still budget
        # for it via a fixed per-tile reservation below.
        mu_chunk_max = max(
            1, int((B_compute - v_per_q_bytes) * p_min /
                   (2.0 * N_zeta * n_G)))
        mu_chunk_max = min(int(n_rmu), int(mu_chunk_max))

        # CONSERVATIVE CHOICE of μ_chunk — must simultaneously satisfy:
        #   (i)  μ_chunk divides N_μ/P_x  AND  μ_chunk divides N_μ/P_y
        #        → blocks land fully inside one x-col × y-col region of
        #          V_ref → the ref slice-set is local (no collective on
        #          write), and μ_lo (= μ_i · μ_chunk) is always a
        #          multiple of μ_chunk so alignment holds.
        #   (ii) μ_chunk is divisible by P_x·P_y
        #        → the read stage ζ(μ on ('x','y')) has integer per-rank
        #          μ rows; the one-axis gather stages likewise produce
        #          integer μ_chunk/P_x and μ_chunk/P_y.
        # Both hold simultaneously when μ_chunk divides gcd(N_μ/P_x,
        # N_μ/P_y) AND is a multiple of P_x·P_y.
        mu_per_x = int(n_rmu) // int(p_x) if int(n_rmu) % int(p_x) == 0 else 0
        mu_per_y = int(n_rmu) // int(p_y) if int(n_rmu) % int(p_y) == 0 else 0
        aligned_parent = gcd(mu_per_x, mu_per_y) if (mu_per_x and mu_per_y) else 0
        snap = int(p_x) * int(p_y) or 1

        aligned = False
        mu_chunk = None
        if aligned_parent and aligned_parent % snap == 0:
            # Divisors of aligned_parent that are multiples of snap, ≤ mu_chunk_max.
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
        # Case B per-tile peak: 2 ζ-G slabs + per-tile disk slab.
        rtot_tile_bytes = (N_zeta * mu_chunk * float(n_rtot) / p_prod
                           if n_rtot is not None and int(n_rtot) > 0 else 0.0)
        peak = (2.0 * N_zeta * mu_chunk * n_G / p_min + rtot_tile_bytes
                + v_per_q_bytes + ref_bytes)
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
# Unified inner V_q kernel
# ----------------------------------------------------------------------------

_v_q_tile_kernel_cache: dict = {}


def _make_V_q_tile_kernel(
    *,
    coulomb_kernels,
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
    coulomb_kernels : SimpleNamespace
        The bundle returned by
        ``coulomb_kernel.make_v_munu_chunked_kernel``.  This kernel
        consumes three of its fields: ``sphere_idx`` (G-sphere indices
        into the flat FFT box, or ``None`` if no cutoff), ``n_sph``
        (post-cutoff G-vector count, == n_G when sphere inactive), and
        ``fft_shape`` (the (nx, ny, nz) FFT-box dims).  Bundling rather
        than re-passing keeps the V_q tile call sites consistent with
        the rest of the chi0/W chain that already threads this
        namespace.
    mesh_xy : Mesh
        2-D device mesh (axes 'x', 'y').
    q_chunk : int
        Number of q-points handled per kernel call.  In Case A this
        equals the q-batch size; in Case B it is always 1.
    mu_size, nu_size : int
        Block sizes on the left (μ) and right (ν) ζ for *this* kernel
        call — i.e. the per-call shape, which sets the jit cache key.
        In Case A both equal ``n_rmu_L`` / ``n_rmu_R`` (full); in Case B
        they are the chunked sizes (with ``min(mu_chunk, n_rmu -
        offset)`` for the last partial block).
    n_rmu_L, n_rmu_R : int
        Total μ counts on the left / right ζ — i.e. the *full* extents
        of the V_acc accumulator.  Distinct from ``mu_size`` /
        ``nu_size`` (the per-call block sizes); the kernel uses these
        only to shape the donated V_acc / g0_acc inputs.  Equal for
        V^{0,0}; may differ for bispinor V^{μ_L, ν_L} tiles.
    same_zeta : bool (static)
        ``True`` when the left and right ζ slabs come from the same file
        AND offsets coincide on the diagonal — the kernel skips the
        second FFT and reuses ``zeta_mu_G`` for both gathers.  This is
        the original Case-A perf optimisation; lifted here so the
        bispinor caller can opt in for V^{0,0} self-contraction tiles
        but not for V^{μ_L, ν_L} (L ≠ R) tiles.
    write_g0 : bool (static)
        ``True`` only on the (μ_i == ν_j) iteration of a V^{0,0}
        tile-loop — the diagonal block is the one that writes ζ_μ(G=0)
        into the g0 accumulator.  When ``same_zeta=False`` the caller
        always passes ``write_g0=False``.

    Returns
    -------
    A jit-compiled function with signature::

        kernel(V_acc, g0_acc,
               zeta_mu_disk, [zeta_nu_disk,]   # second arg only when not same_zeta
               v_per_G_batch, phase_batch,
               q_lo_dyn, mu_lo_dyn, nu_lo_dyn) -> (V_acc', g0_acc')

    plus the convenience attributes ``zeta_disk_sh``, ``V_sh``, ``g0_sh``
    used by the driver to allocate input arrays in the right layout.
    """
    sphere_idx = coulomb_kernels.sphere_idx
    n_G_sph = coulomb_kernels.n_sph
    fft_shape = coulomb_kernels.fft_shape

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

    # Sharding specs.  These match the old Case A / Case B kernels
    # verbatim — the chi0 / CCT / ZCT / sigma chain consumes
    # P(None, 'x', 'y') for V_block, so any change here would force a
    # downstream reshard.
    blk_xy_sh = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
    # 5-D μ-sharded layout for the FFT — see Case-A comment in old code.
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

    # Local-FFT wrapper — single shard_map'd 3-D cuFFT plan.  Was the
    # ``make_jittable_local_fftn_3d`` (3×1D custom_partitioning) variant
    # in old Case B; the unified kernel uses the faster
    # ``make_sharded_fftn_3d`` form to match the chi0/CCT/ZCT/Case-A
    # pattern.  See commit 9a8e6e5 for the Si 4×4×4 BSE benchmark that
    # established the swap (8.07 s → 5.97 s walltime).
    _local_fftn_3d = make_sharded_fftn_3d(
        mesh_xy, blk_xy_5d_spec, blk_xy_5d_spec)

    def _fft_and_sphere(zeta_rtot_mu, phase_batch):
        """Local transpose + 3-D FFT + sphere pick (NO v multiply).

        Returns ``(zeta_G, g0_blk)`` where ``g0_blk`` is the G=(0,0,0)
        column (shape ``(Q, μ_per_rank)``, μ-XY-sharded) and ``zeta_G``
        is the sphere-gathered slab (shape ``(Q, μ_per_rank, n_G_sph)``,
        μ-XY-sharded).  The caller is responsible for applying the v(K)
        weight on one side of the contraction (see module docstring).
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
        # Single zeta input — one FFT, two gathers from the same source.
        # This is the old Case-A flow (also the diagonal of Case B when
        # we skipped the redundant second FFT).
        @partial(jax.jit,
                 in_shardings=(V_sh, g0_sh, zeta_disk_sh,
                               v_per_G_sh, phase_sh, rep, rep, rep),
                 out_shardings=(V_sh, g0_sh),
                 donate_argnums=(0, 1))
        def _kernel(V_acc, g0_acc,
                    zeta_mu_disk,
                    v_per_G_batch, phase_batch,
                    q_lo_dyn, mu_lo_dyn, nu_lo_dyn):
            # FFT + sphere only — no v multiply yet (see module docstring,
            # API note: v is applied on one side of the contraction).
            zeta_G, g0_mu = _fft_and_sphere(zeta_mu_disk, phase_batch)

            # Two one-axis gathers from the same post-FFT tensor.
            zeta_mu_X = jax.lax.with_sharding_constraint(zeta_G, blk_x_sh)
            zeta_nu_Y = jax.lax.with_sharding_constraint(zeta_G, blk_y_sh)

            # One-sided v(K) multiply on the L (μ) side.  Mathematically
            # identical to symmetric-√v for real, non-negative v — the
            # only difference is that this form also handles the signed,
            # non-PSD transverse-projector weight v(K) · (-K̂_i K̂_j) used
            # by the bispinor V^{μ_L, ν_L} (μ_L ≠ ν_L) tiles.
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
        # Distinct zeta inputs — two FFTs, two gathers (one per side).
        # This is the old Case-B flow + the bispinor V^{μ_L, ν_L} (L ≠ R)
        # extension.  ``write_g0`` is always ``False`` here (the head
        # term is only meaningful for V^{0,0}; the bispinor caller will
        # not request it for the off-diagonal tiles).
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
            # FFT + sphere only — no v multiply yet (see module docstring,
            # API note: v is applied on one side of the contraction).
            zeta_mu_G, _ = _fft_and_sphere(zeta_mu_disk, phase_batch)
            zeta_nu_G, _ = _fft_and_sphere(zeta_nu_disk, phase_batch)

            # One-axis gathers: μ side → μ only X-sharded; ν side → ν only
            # Y-sharded.
            zeta_mu_X = jax.lax.with_sharding_constraint(zeta_mu_G, blk_x_sh)
            zeta_nu_Y = jax.lax.with_sharding_constraint(zeta_nu_G, blk_y_sh)

            # One-sided v(K) multiply on the L (μ) side — handles the
            # signed transverse-projector weight v(K) · (-K̂_i K̂_j) used
            # by bispinor off-diagonal tiles.
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


# ----------------------------------------------------------------------------
# Outer per-tile driver
# ----------------------------------------------------------------------------


def compute_V_q_tile(
    *,
    coulomb_kernels,
    zeta_L_io,
    zeta_R_io,
    v_per_G_fn,
    phase_fn,
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
    n_rmu_L: int,
    n_rmu_R: int,
    V_acc,
    g0_acc=None,
    chooser_choice: dict | None = None,
    budget_bytes: float | None = None,
    bgw_v_grid_overlay_fn=None,
    verbose: bool = True,
    timing_label: str = "compute_V_q_tile",
):
    """Outer driver for the unified V_q tile kernel.

    Runs the q-batch loop and (μ_block, ν_block) loop, calling the inner
    JIT'd kernel once per (q_batch, μ_block, ν_block) iteration.  Reads
    ζ_L (and ζ_R when distinct) via ``SlabIO.read_slab(...,
    partition_spec=P(None, None, ('x','y')))`` — landing the disk slabs
    directly in μ-on-XY layout for the FFT.

    Parameters
    ----------
    coulomb_kernels : SimpleNamespace
        The bundle returned by
        ``coulomb_kernel.make_v_munu_chunked_kernel``.  Used here to
        derive ``sphere_idx`` (G-sphere indices into the flat FFT box,
        or ``None`` when no cutoff), ``fft_shape`` (FFT-box dims), and
        the implied ``n_rtot = prod(fft_shape)``; passed straight
        through to ``_make_V_q_tile_kernel``.
    zeta_L_io, zeta_R_io : SlabIO
        Open SlabIO handles for the left / right ζ files.  When both
        are the same object (``zeta_L_io is zeta_R_io``) the kernel
        runs in single-zeta mode (one FFT, two gathers) — this is the
        V^{0,0} self-contraction case.
    v_per_G_fn : callable(qvec_batch_np) -> jnp.ndarray (Q, n_G_sph) c128
        Returns the per-G weight applied on the L side of the
        contraction (see module docstring).  Any complex-valued weight
        is allowed (signed, non-PSD); the kernel multiplies it once on
        the μ side before the einsum.  For V^{0,0} this is just
        ``v(q+G)`` (real, ≥ 0) and the math is bit-identical to the
        old symmetric-√v form.  For bispinor V^{μ_L, ν_L} it is
        ``v(K) · t_{μ_L, ν_L}(K)`` where ``t`` is the (signed)
        transverse-projector entry — see ``v_q_lorentz``.
    phase_fn : callable(qvec_batch_np) -> jnp.ndarray (Q, 1, nx, ny, nz) c128
        Returns the per-q phase factor ``exp(-2πi q·r)`` on the FFT
        box.  Independent of the v-side (factored out of the kernel
        bundle).
    mesh_xy : Mesh
        2-D device mesh.
    kgrid : (nkx, nky, nkz)
    n_rmu_L, n_rmu_R : int
        Total μ counts on the left / right ζ.
    V_acc : jnp.ndarray, shape (n_q_total, n_rmu_L, n_rmu_R), P(None, 'x', 'y')
        Donated mutable accumulator.  The kernel performs DUS writes
        into it.  Caller pre-allocates with ``jnp.zeros`` of the right
        layout.
    g0_acc : jnp.ndarray | None, shape (n_q_total, n_rmu_L), P(None, 'x')
        Donated g0 accumulator — only used when ``same_zeta`` AND the
        caller wants the head term.  Pass ``None`` for bispinor off-
        diagonal tiles or when the caller doesn't need g0.
    chooser_choice : dict | None
        Output of ``_choose_v_q_chunks``.  If ``None``, the chooser is
        run with ``budget_bytes`` (mandatory in that case).
    budget_bytes : float | None
        Memory budget per device, in bytes.  Required when
        ``chooser_choice`` is None.
    bgw_v_grid_overlay_fn : callable | None
        Optional host-side overlay applied to ``v_per_G`` before passing
        to the kernel.  Signature::
            bgw_v_grid_overlay_fn(qvec_np_batch, v_per_G_native_np)
                                   -> v_per_G_overlaid_np
        Used for ``use_bgw_vcoul=true`` byte-reproducible BGW
        comparison.
    verbose : bool
    timing_label : str

    Returns
    -------
    (V_acc, g0_acc_or_None) — both donated buffers, returned in their
    final filled state.  When ``g0_acc is None`` on entry, returns
    ``(V_acc, None)``.

    Notes
    -----
    * The chooser's ``aligned`` knob and the (mu_size, nu_size) cache
      key combine to give the standard 4- (or 8-, with write_g0)
      compile-shape envelope of the old Case-B kernel.  In the
      chooser-aligned case all interior iterations share one compile
      shape.
    * The ``LORRAX_V_Q_MU_CHUNK`` env override is honoured by the
      driver in ``v_q_driver`` before calling here — this function does
      not consult env vars.
    """
    from common import timing
    from common.progress import LoopProgress

    nkx, nky, nkz = kgrid
    nq_total = nkx * nky * nkz
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])

    # Derive geometry from the Coulomb kernel bundle.
    sphere_idx = coulomb_kernels.sphere_idx
    fft_grid = coulomb_kernels.fft_shape
    n_rtot = int(np.prod(fft_grid))

    same_zeta = (zeta_L_io is zeta_R_io)

    # The caller must shape n_rmu_L == n_rmu_R for the V^{0,0} case to
    # match the old kernel's accumulator layout.  Bispinor off-diagonal
    # tiles can have n_rmu_L ≠ n_rmu_R.
    if same_zeta and n_rmu_L != n_rmu_R:
        raise ValueError(
            f"compute_V_q_tile: same_zeta=True requires n_rmu_L == n_rmu_R, "
            f"got {n_rmu_L} vs {n_rmu_R}.")

    # n_G_sph derived from sphere_idx (or the FFT box size when no sphere).
    if sphere_idx is not None:
        n_G_sph = int(np.asarray(sphere_idx).shape[0])
    else:
        n_G_sph = fft_grid[0] * fft_grid[1] * fft_grid[2]

    # Memory chooser.
    if chooser_choice is None:
        if budget_bytes is None:
            raise ValueError(
                "compute_V_q_tile: must pass either chooser_choice or "
                "budget_bytes.")
        # When n_rmu_L != n_rmu_R the chooser's V_ref accounting is
        # overestimated (it uses n_rmu² for the V_ref slab) — the
        # bispinor caller is expected to pass an explicit
        # chooser_choice that already accounts for the exact tile
        # geometry.  For the V^{0,0} case both equal n_rmu so the
        # chooser is correct as-is.
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
        v_per_G = v_per_G_fn(qvec_np)        # (Q, n_G_sph) c128 — driver-built
        phase = phase_fn(qvec_np)            # (Q, 1, nx, ny, nz) c128
        if bgw_v_grid_overlay_fn is not None:
            v_per_G = bgw_v_grid_overlay_fn(qvec_np, v_per_G)
        return v_per_G, phase

    vq_progress = LoopProgress(
        nq_total, print, title="V_q tile",
        item_name="q-point", max_updates=min(nq_total, 20))

    if not tiled:
        # === Case A — one contiguous read per q-batch.  Full μ in both
        # directions; in same_zeta mode one FFT + two gathers.
        q_coords = [(qx, qy, qz) for qx in range(nkx)
                    for qy in range(nky) for qz in range(nkz)]
        batches = [q_coords[i:i + q_chunk]
                   for i in range(0, nq_total, q_chunk)]

        with timing.section(timing_label):
            q_cursor = 0
            for batch in batches:
                actual = len(batch)
                kernel = _make_V_q_tile_kernel(
                    coulomb_kernels=coulomb_kernels,
                    mesh_xy=mesh_xy,
                    q_chunk=actual, mu_size=n_rmu_L, nu_size=n_rmu_R,
                    n_rmu_L=n_rmu_L, n_rmu_R=n_rmu_R,
                    same_zeta=same_zeta,
                    write_g0=(same_zeta and g0_acc is not None))
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
                    V_acc, g0_acc = kernel(
                        V_acc, _g0_or_dummy(g0_acc, mesh_xy, nq_total, n_rmu_L),
                        zeta_L, v_per_G_b, phase_b,
                        jnp.int32(q_cursor), jnp.int32(0), jnp.int32(0))
                else:
                    zeta_R = zeta_R_io.read_slab(
                        'zeta_q',
                        shape=(actual, n_rtot, n_rmu_R),
                        dtype=np.complex128,
                        offset=(q_flat0, 0, 0),
                        mesh=mesh_xy, partition_spec=_read_spec)
                    V_acc, _ = kernel(
                        V_acc, _g0_or_dummy(g0_acc, mesh_xy, nq_total, n_rmu_L),
                        zeta_L, zeta_R, v_per_G_b, phase_b,
                        jnp.int32(q_cursor), jnp.int32(0), jnp.int32(0))
                    del zeta_R
                del zeta_L
                q_cursor += actual
                for _ in range(actual):
                    vq_progress.step()
            vq_progress.finish()

    else:
        # === Case B — single q, μ × ν tile loop.  TWO separate reads per
        # iteration when same_zeta=False; same_zeta=True still reads
        # twice on off-diagonal tiles to keep the jit body uniform
        # (but uses the single-input kernel on the diagonal where
        # μ_lo == ν_lo).
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
                                same_zeta and (g0_acc is not None)
                                and on_diag
                            )
                            if on_diag:
                                # Reuse the μ slab as the ν slab too
                                # (identical offsets).  Single-FFT path.
                                k = _make_V_q_tile_kernel(
                                    coulomb_kernels=coulomb_kernels,
                                    mesh_xy=mesh_xy,
                                    q_chunk=1, mu_size=mu_size, nu_size=nu_size,
                                    n_rmu_L=n_rmu_L, n_rmu_R=n_rmu_R,
                                    same_zeta=True, write_g0=do_write_g0)
                                V_acc, g0_acc_filled = k(
                                    V_acc,
                                    _g0_or_dummy(g0_acc, mesh_xy,
                                                 nq_total, n_rmu_L),
                                    zeta_mu, v_per_G_b, phase_b,
                                    jnp.int32(q_flat),
                                    jnp.int32(mu_lo), jnp.int32(nu_lo))
                                if g0_acc is not None:
                                    g0_acc = g0_acc_filled
                            else:
                                zeta_nu = zeta_R_io.read_slab(
                                    'zeta_q',
                                    shape=(1, n_rtot, nu_size),
                                    dtype=np.complex128,
                                    offset=(q_flat, 0, nu_lo),
                                    mesh=mesh_xy, partition_spec=_read_spec)
                                k = _make_V_q_tile_kernel(
                                    coulomb_kernels=coulomb_kernels,
                                    mesh_xy=mesh_xy,
                                    q_chunk=1, mu_size=mu_size, nu_size=nu_size,
                                    n_rmu_L=n_rmu_L, n_rmu_R=n_rmu_R,
                                    same_zeta=False, write_g0=False)
                                V_acc, _ = k(
                                    V_acc,
                                    _g0_or_dummy(g0_acc, mesh_xy,
                                                 nq_total, n_rmu_L),
                                    zeta_mu, zeta_nu, v_per_G_b, phase_b,
                                    jnp.int32(q_flat),
                                    jnp.int32(mu_lo), jnp.int32(nu_lo))
                                del zeta_nu
                        del zeta_mu
                    vq_progress.step()
            vq_progress.finish()

    return V_acc, g0_acc


def _g0_or_dummy(g0_acc, mesh_xy, nq_total, n_rmu_L):
    """Return ``g0_acc`` if non-None, else a fresh sharded dummy.

    The unified kernel always takes a g0_acc argument (the JIT signature
    has both V_acc and g0_acc in ``donate_argnums=(0, 1)``).  When the
    caller doesn't want the head term we pass a dummy zero slab so the
    JIT signature is uniform; the kernel either ignores it
    (``write_g0=False``) or writes into it (which we discard on return).
    """
    if g0_acc is not None:
        return g0_acc
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))

    @partial(jax.jit, out_shardings=g0_sh)
    def _zeros():
        return jnp.zeros((nq_total, n_rmu_L), dtype=jnp.complex128)

    return _zeros()
