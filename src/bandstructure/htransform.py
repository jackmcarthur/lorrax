import os
import re
import math
import argparse
import numpy as np

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap()
# (env defaults + jax.distributed init + CPU fallback; all idempotent).
# MUST run before this module's own `import jax`.
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
from jax.scipy import linalg as jsp_linalg
from jax.scipy.special import erf
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial

from file_io import WfnLoader as WFNReader
from common import symmetry_maps
from common import Meta
from common.wfn_transforms import (
    get_enk_bandrange, load_centroids_band_chunked, iter_psi_rchunk_bandwise,
    load_psi_gflat_padded,
)
from isdf import factor_c_q
from common.fft_helpers import make_flat_k_ifftn


def _build_mesh_xy() -> Mesh:
    """Most-square 2D ('x','y') mesh — matches gw_jax._build_mesh idiom."""
    total = jax.process_count() * jax.local_device_count()
    gx = int(np.sqrt(total))
    while gx > 1 and total % gx != 0:
        gx -= 1
    return Mesh(np.asarray(jax.devices()).reshape(gx, total // gx), ('x', 'y'))


_accum_G_cache: dict = {}


# Ceiling on ONE streamed ψ band-chunk in the Galerkin build.
#
# ``iter_psi_rchunk_bandwise`` is called below with the whole r-axis
# (r0 = 0, r_end = meta.n_rtot), so the only lever on the chunk's size is the
# band count, and ``to_rchunk`` materialises the chunk as a single REPLICATED
# (nk, bc, nspinor, n_rtot) complex128 array — it is not sharded by the mesh
# (verified: the same allocation appears on a 1×1 and a 2×2 mesh).  One band of
# it costs nk·nspinor·n_rtot·16 B, which is 0.81 GB/band on the converged MoS2
# reference (nk=144, n_rtot=174 960).  So the historical fixed default of 64
# bands per chunk asks for 51.6 GB, and even a 48-band window asks for 38.7 GB
# — enough to OOM an A100-80GB once the BSE/GW restart tensors are resident.
# The cap makes the default size-aware.  Splitting the band axis is NOT a pure
# accumulation split — see the G-accumulation block below — so the loop sums
# band chunks into Q first and only then forms Q Qᴴ; this constant therefore
# changes memory and the number of ψ re-reads, never the result.
# Override with LORRAX_GALERKIN_CHUNK_GIB (GiB, float).
_GALERKIN_CHUNK_MAX_BYTES = int(
    float(os.environ.get("LORRAX_GALERKIN_CHUNK_GIB", "6")) * 1024 ** 3)


def streaming_galerkin_solve(wfn, sym, meta, centroid_indices, mesh_xy: Mesh,
                             band_range: tuple[int, int],
                             rtol: float = 1e-8, log_fn=None,
                             band_chunk_size: int = 64,
                             bispinor: bool = False,
                             return_full_proj: bool = False):
    """Galerkin projection using gw_jax shared loaders.

    Single ('x','y') mesh throughout. ψ at centroids comes from
    ``load_centroids_band_chunked``; ψ at full r is streamed via
    ``iter_psi_rchunk_bandwise`` (band+r chunked, sharded on 'y'). G is built
    sharded ``P('x','y')`` and Cholesky-factored via ``factor_c_q``
    so the 1×1-mesh dense path and the 2D-blocked distributed path both work.

    Args:
        wfn, sym, meta: standard gw_jax handles.
        centroid_indices: (n_μ, 3) FFT-grid coordinates.
        mesh_xy: 2D ``('x','y')`` device mesh.
        band_range: ``(b_start, b_end)`` — bands included in the SVD basis.
        rtol: SVD truncation tolerance (relative to s.max()).
        band_chunk_size: bands per FFT chunk inside the loader.  Treated as a
            CEILING, not a fixed size: it is lowered so one streamed ψ chunk
            ``(nk, bc, nspinor, n_rtot)`` complex128 stays under
            ``_GALERKIN_CHUNK_MAX_BYTES`` (see below).
        bispinor: passed through to ``load_centroids_band_chunked``.
        return_full_proj: also return the full-r α-basis projector
            ``W_proj = L⁻¹ diag(1/s) U^H`` (rank, nk·nb), replicated.  For
            any streamed ψ chunk (nk, bc, ns, r_c) restricted to the SAME
            band window, ``B_full[α, s·r_c] = W_proj_bc @ ψ_flat`` evaluates
            the α-basis on the full r-grid, so
            ``ψ_{n,q}(r) = Σ_α c_{n,q}[α] B_full[α](r)`` reconstructs ψ at
            ANY q off the grid — the per-Q ζ-refit consumer
            (``bse.vq_interp.refit_vq``).

    Returns ``(S, ctilde, B_at_mu)`` (legacy contract), plus ``W_proj``
    appended when ``return_full_proj``.
    """
    import time
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    b_start, b_end = band_range
    nb = b_end - b_start
    nk = meta.nk_tot
    nspinor = meta.nspinor
    n_mu = int(centroid_indices.shape[0])

    rep = NamedSharding(mesh_xy, P())               # fully replicated
    grid_xy = NamedSharding(mesh_xy, P('x', 'y'))   # (rank, rank) face

    # Size the band chunk to the ψ box rather than trusting a fixed default.
    bytes_per_band = nk * nspinor * int(meta.n_rtot) * 16
    bc_cap = max(1, int(_GALERKIN_CHUNK_MAX_BYTES // max(1, bytes_per_band)))
    band_chunk_size = max(1, min(int(band_chunk_size), bc_cap, nb))

    log_fn(
        f"  Streaming Galerkin: nk={nk}, nb={nb}, nr={meta.n_rtot}, "
        f"n_mu={n_mu}, mesh=({mesh_xy.shape['x']}x{mesh_xy.shape['y']}), "
        f"band_chunk={band_chunk_size} "
        f"({bytes_per_band * band_chunk_size / 1024**3:.2f} GB/chunk)"
    )
    if band_chunk_size < nb:
        log_fn(f"  (band axis split into "
               f"{(nb + band_chunk_size - 1)//band_chunk_size} chunks; G is "
               f"accumulated r-outer / band-inner so the split stays exact)")

    # ── 1. Load ψ(G-flat) ONCE for the whole band window ──
    # One capped+padded load (band-sharded over ('x','y')) serves BOTH the
    # centroid sampling below AND the r-streaming Galerkin sweep in step 3
    # — previously each streamed band chunk re-issued its own
    # ``loader.load`` (h5py read + symmetry unfold per chunk), so ψ was
    # read from file twice per run (once for centroids, once chunk-by-chunk
    # for the stream).  G-flat is small (~6-11% of the FFT box, same
    # tensor ``load_centroids_band_chunked`` already held), so keeping it
    # resident through the Q loop trades a few GB global for a single
    # file read + single unfold.  Numerics: per-chunk consumption is a
    # device SLICE of the same values — bit-identical.
    t0 = time.time()
    psi_G_win = load_psi_gflat_padded(
        wfn, band_range, mesh_xy=mesh_xy, bispinor=bispinor)
    if psi_G_win is None:
        raise ValueError(
            f"streaming_galerkin_solve: band window {band_range} lies "
            f"entirely past the file's band extent ({int(wfn.nbands)})")
    log_fn(f"  ψ(G-flat) window load: {time.time()-t0:.2f}s "
           f"(shape {tuple(psi_G_win.shape)}, shared by centroid sample + "
           f"r-stream)")

    # ── 1b. ψ at centroids (band-sharded internally on 'y') ──
    psi_rmu_Y, _ = load_centroids_band_chunked(
        wfn, sym, meta, centroid_indices, bispinor, mesh_xy,
        band_range=band_range, band_chunk_size=band_chunk_size,
        psi_G_flat=psi_G_win,
    )
    # psi_rmu_Y: (nk, nb, ns, n_μ), sharded P(None, None, None, 'y')
    log_fn(f"  load_centroids_band_chunked: {time.time()-t0:.2f}s")

    # ── 2. SVD of A = ψ@centroids reshaped to (nk·nb, ns·n_μ) ──
    # FFI seam: today gather to replicated, run dense SVD on each device
    # (small for our sizes); swap for distributed SVD when n_μ scales up.
    t1 = time.time()
    # Trim the sharding-pad: load_centroids_band_chunked pads the n_μ axis to a
    # multiple of the device count, but the TRUE centroid count is
    # ``n_mu = centroid_indices.shape[0]`` (= the restart's n_rmu).  Without this
    # trim a centroid count not divisible by the device count (e.g. 1496 on 16
    # GPU → padded 1504) reshape-crashes here.  No-op when already divisible.
    A = psi_rmu_Y[..., :n_mu].reshape(nk * nb, nspinor * n_mu)
    A = jax.device_put(A, rep)
    del psi_rmu_Y

    @partial(jax.jit, donate_argnums=(0,), out_shardings=(rep, rep, rep))
    def _svd_replicated(A):
        result = jnp.linalg.svd(A, full_matrices=False)
        return result.U, result.S, result.Vh

    U, s, Vh = _svd_replicated(A)
    del A
    s_host = np.asarray(s)
    rank_raw = int((s_host > s_host.max() * rtol).sum())
    # Mesh-align the retained rank.  G, its Cholesky factor, ctilde and fH all
    # live on a (rank, rank) face sharded P('x', 'y'), so ``rank`` has to divide
    # BOTH mesh axes.  Whenever the ψ-at-centroids matrix is genuinely
    # rank-deficient — nk·nb at or above the ISDF column count, which is exactly
    # the regime the band-window capacity rule is about — ``rank_raw`` is an
    # arbitrary integer and the first ``device_put`` onto that face raises
    #   "global size of its dimension 0 should be divisible by 4 ... equal to 4570".
    # Round DOWN to the aligned rank: the dropped directions are the ones
    # sitting at the rtol threshold (σ/σ_max ~ 1e-8), i.e. numerically the same
    # decision ``rtol`` already makes, one notch tighter.
    align = math.lcm(int(mesh_xy.shape['x']), int(mesh_xy.shape['y']))
    rank = max(align, (rank_raw // align) * align)
    rank = min(rank, int(s_host.size))
    U = U[:, :rank]
    s = s[:rank]
    Vh = Vh[:rank, :]                        # (rank, ns·n_μ), keep for centroid recovery
    log_fn(f"  SVD of ({nk*nb}, {nspinor*n_mu}): rank={rank}"
           + (f" (rtol rank {rank_raw} → mesh-aligned to {align})"
              if rank != rank_raw else "")
           + f", σ_max={float(s_host[0]):.3e}, "
           f"σ_min/{rtol:.0e}={float(s_host[rank-1]):.3e} "
           f"({time.time()-t1:.2f}s)")
    if rank_raw < nk * nb:
        log_fn(f"  [warn] ψ-at-centroids is RANK-DEFICIENT: {nk*nb} states vs "
               f"numerical rank {rank_raw} (nspinor·n_μ = {nspinor*n_mu}).  The "
               f"Galerkin basis cannot span the band window — fH energy "
               f"recovery will degrade.  Capacity rule: nk·nb < rank(ψ_μ) "
               f"≤ nspinor·n_μ, i.e. nb < {rank_raw/nk:.2f} here.")

    coeffs = U * s[None, :]                  # (nk·nb, rank), replicated
    inv_s = (1.0 / s)[:, None]               # (rank, 1), replicated
    UH = U.conj().T                          # (rank, nk·nb), replicated
    del U

    # ── 3. G = Q Q^H with Q = inv_s · U^H ψ, streamed r-OUTER / band-INNER ──
    #
    # Q[α, x] = Σ_{k,n} inv_s[α] U^H[α,(k,n)] ψ[(k,n), x]  — the contraction
    # runs over the PAIR index (k, n) and is FREE in x = (spinor, r).  So:
    #
    #   * splitting x (the r axis) and summing G over the pieces is EXACT —
    #     G = Σ_rc Q_rc Q_rc^H, because each Q_rc is a disjoint column block;
    #   * splitting the CONTRACTED (k, n) axis and summing G over the pieces
    #     is NOT — Σ_bc Q_bc Q_bc^H drops every bc ≠ bc' cross term of
    #     (Σ_bc Q_bc)(Σ_bc' Q_bc')^H.
    #
    # The band chunks must therefore be summed INTO Q before the outer
    # product, which is what this loop does.  Getting that backwards is a
    # silent wrong answer, not a crash: G stays Hermitian positive-definite,
    # Cholesky succeeds, and the only symptom is that ctilde comes out
    # non-orthonormal — ``ctilde[0] orthogonality error`` jumps to ~0.5 —
    # so fH = Σ_n f(ε_n) c_n c_n^H no longer has eigenvalues f(ε_n) and the
    # recovered energies drift by ~1.7 eV.  Measured on MoS2 12x12 30 Ry /
    # n_μ=640 / nb=40, on-grid max|Δε_c| vs the stored grid:
    #   one band chunk (nb ≤ band_chunk_size)  0.63 meV
    #   two band chunks, G summed per chunk    1742.48 meV
    # It stayed hidden while ``band_chunk_size`` defaulted to a fixed 64 ≥ nb
    # (one chunk, no cross terms to lose) and surfaced when the chunk was
    # sized to the ψ box instead.
    n_rtot = meta.fft_grid[0] * meta.fft_grid[1] * meta.fft_grid[2]
    # Align band_chunk to the mesh band-shard count (p_band = mesh.size): the
    # loader pads nb to round_up(band_chunk, p_band) regardless, so e.g. 7 real
    # bands on a 16-way mesh still COSTS 16 -- and pushes 9 zero-pad bands
    # through every local FFT and the UH@psi GEMM (~2.3x waste) while doubling
    # the iteration count (and the per-chunk XLA retrace storm).  Rounding up to
    # a p_band multiple is free (same padded memory) and both removes the waste
    # and halves the iterations/compiles.  Pure chunking change -> identical G.
    _p_band = max(1, int(mesh_xy.size))
    _bc = ((band_chunk_size + _p_band - 1) // _p_band) * _p_band
    band_chunk_ranges = [
        (b_start + i * _bc, min(b_start + (i + 1) * _bc, b_end))
        for i in range((nb + _bc - 1) // _bc)
    ]
    # r-chunk holds one Q block (rank, nspinor·r_len) plus the ψ tile
    # (nk, band_chunk, nspinor, r_len); size it so both stay inside the budget.
    # Q spans the FULL r axis.  ``iter_psi_rchunk_bandwise`` is only ever
    # exercised with (r_start, r_len) = (0, n_rtot) — handing it an interior
    # r_start aborts inside XLA — and Q at full r costs
    # rank·nspinor·n_rtot·16 sharded over 'y', which is 6.6 GB/device at the
    # largest case here (MoS2 12x12 / n_μ=2412 / rank 4700 on a 4x4 mesh).
    # The ψ tile stays band-chunked, so the streaming property that
    # ``band_chunk_size`` buys is unchanged; only Q is now carried across the
    # band loop, which is exactly what makes the split exact.

    def _make_accum_kernel(rank_, bc_size, nspinor_, mesh_, sharding_y, sharding_xy):
        key = (id(mesh_), rank_, bc_size, nspinor_)
        fn = _accum_G_cache.get(key)
        if fn is not None:
            return fn

        @partial(jax.jit, donate_argnums=(2, 3), out_shardings=sharding_y)
        def _accum(UH_bc, inv_s, psi_bc, Q_in):
            # UH_bc: (rank, nk·bc) replicated; inv_s: (rank,1) replicated
            # psi_bc: (nk, bc, ns, r_chunk) sharded P(None,None,None,'y')
            # Reshape psi so the contraction axis is (nk·bc) and the
            # remaining axis is (ns·r_chunk). Spinor folds with r-axis,
            # matching the (ns·n_μ) column axis of A used in the SVD.
            nkv, bcv, nsv, rcv = psi_bc.shape
            psi_flat = psi_bc.reshape(nkv * bcv, nsv * rcv)  # (nk·bc, ns·r_chunk)
            Q = inv_s * (UH_bc @ psi_flat)                   # (rank, ns·r_chunk), sharded on 'y'
            return Q_in + jax.lax.with_sharding_constraint(Q, sharding_y)

        _accum_G_cache[key] = _accum
        return _accum

    # Q is carried ACROSS the band loop now, so unlike the old per-chunk
    # transient it is a live buffer for the whole accumulation — split it over
    # BOTH mesh axes, not just 'y'.  At MoS2 12x12 / n_μ=2412 / rank 4700 that
    # is 26.3 GB global: 6.6 GB/device on 'y' alone (times two for the
    # in+out pair the loop needs) versus 1.6 GB/device over ('x','y').
    sharding_y = NamedSharding(mesh_xy, P(None, ('x', 'y')))
    grid_xy_G = grid_xy

    @partial(jax.jit, donate_argnums=(0, 1), out_shardings=grid_xy_G)
    def _fold_G(Q, G_in):
        return G_in + (Q @ Q.conj().T)                       # (rank, rank) P('x','y')

    G = jax.device_put(jnp.zeros((rank, rank), dtype=jnp.complex128), grid_xy)

    t2 = time.time()
    chunk_count = 0
    UH_kb = UH.reshape(rank, nk, nb)  # for band-chunk slicing
    # Allocate Q ALREADY SHARDED.  ``device_put(jnp.zeros(...), sharding)``
    # builds the whole (rank, nspinor·n_rtot) array on one device first and
    # only then distributes it — 23.4 GB in a single allocation at MoS2
    # 12x12 / n_μ=2412, which OOMs the card before the sharded copy exists.
    Q = jax.jit(lambda: jnp.zeros((rank, nspinor * n_rtot),
                                  dtype=jnp.complex128),
                out_shardings=sharding_y)()
    for bc_range, psi_bc_Y in iter_psi_rchunk_bandwise(
            wfn, sym, meta, mesh_xy, band_range, 0, n_rtot, bispinor,
            band_chunk_size=band_chunk_size,
            band_chunk_ranges=band_chunk_ranges,
            band_pad_to=_bc,
            psi_G_flat=psi_G_win):
        bc = bc_range[1] - bc_range[0]
        bc_lo = bc_range[0] - b_start
        bc_hi = bc_range[1] - b_start
        # ``iter_psi_rchunk_bandwise(band_pad_to=_bc)`` zero-pads every
        # chunk's band axis to the SINGLE width ``w`` (= _bc for full
        # chunks; the remainder chunk is padded up to it too), so
        # ``to_rchunk`` and ``_accum`` compile ONCE for the whole sweep
        # instead of recompiling per distinct remainder width.  The
        # contraction operand UH_bc has to match that width: the real
        # bands occupy columns [0, bc); the trailing (w - bc) columns
        # multiply psi's zero pad-bands, so zero-filling them keeps Q
        # bit-identical.  (This also removes a latent shape mismatch when
        # the loader's own p_band round-up already made w > bc.)
        w = int(psi_bc_Y.shape[1])
        UH_bc_kb = UH_kb[:, :, bc_lo:bc_hi]              # (rank, nk, bc)
        if w > bc:
            UH_bc_kb = jnp.pad(UH_bc_kb, ((0, 0), (0, 0), (0, w - bc)))
        UH_bc = UH_bc_kb.reshape(rank, nk * w)

        accum = _make_accum_kernel(rank, w, nspinor, mesh_xy,
                                   sharding_y, grid_xy)
        Q = accum(UH_bc, inv_s, psi_bc_Y, Q)
        chunk_count += 1
    del psi_G_win  # window served its last consumer; free before Q Qᴴ
    G = _fold_G(Q, G)
    del Q
    jax.block_until_ready(G)
    log_fn(f"  G accumulation: {chunk_count} band chunk(s) summed into Q, "
           f"then one Q Qᴴ fold, {time.time()-t2:.2f}s")

    # ── 4. Cholesky on G ──
    # FFI seam: gw_jax's factor_c_q (which has dual 1×1-dense /
    # 2D-blocked paths) is the natural drop-in for distributed scaling, but
    # its 1e-14·trace ridge is calibrated for ISDF's huge-trace C_q matrices
    # and biases htransform's small-trace G by ~1e-5. Use raw Cholesky for
    # numerical parity with the legacy pipeline; swap to factor_c_q
    # (or its FFI variant) when n_μ scales to where rank-deficiency dominates.
    t3 = time.time()
    L = jnp.linalg.cholesky(G)
    log_fn(f"  Cholesky of G: {time.time()-t3:.2f}s")

    # ── 5. ctilde = coeffs · L + ortho diagnostic + B = L⁻¹V^H ──
    # B is the rank-α basis evaluated at the centroids:
    #   ψ_nk(r_μ, s) = Σ_α ctilde[k,n,α] · B[α, s, μ]
    # Math: with A = ψ at centroids = U s V^H, the Cholesky-orthogonalised
    # interpolation vectors at r_μ are exactly L^{-1} V^H (in the (ns, n_μ)
    # column index of V^H). This is what downstream Fourier-upscaling +
    # reconstruction needs to recover ψ at any new k at the centroids.
    @jax.jit
    def _finalize(coeffs, L, Vh):
        coeffs = coeffs @ L
        ctilde = coeffs.reshape(nk, nb, rank)
        CtC = ctilde[0] @ ctilde[0].conj().T
        ortho_err = jnp.max(jnp.abs(CtC - jnp.eye(nb, dtype=ctilde.dtype)))
        # B = L⁻¹ V^H, then split (ns·n_μ) → (ns, n_μ).
        B_flat = jsp_linalg.solve_triangular(L, Vh, lower=True)  # (rank, ns·n_μ)
        B = B_flat.reshape(rank, nspinor, n_mu)
        return ctilde, ortho_err, B

    ctilde, ortho_err, B_at_mu = _finalize(coeffs, L, Vh)
    ctilde = jax.device_put(ctilde, rep)
    B_at_mu = jax.device_put(B_at_mu, rep)
    S = jnp.eye(rank, dtype=jnp.complex128)
    log_fn(f"  ctilde[0] orthogonality error: {float(ortho_err):.3e}")
    log_fn(f"  Total Galerkin: {time.time()-t0:.2f}s")

    if return_full_proj:
        # W_proj = L⁻¹ diag(1/s) U^H — the α-basis-on-full-r projector (see
        # docstring).  (rank, nk·nb) replicated; small (≲ tens of MB).
        W_proj = jsp_linalg.solve_triangular(L, inv_s * UH, lower=True)
        W_proj = jax.device_put(W_proj, rep)
        return S, ctilde, B_at_mu, W_proj
    return S, ctilde, B_at_mu


def solve_q0_galerkin(
    psi_mu: jax.Array,
    psi_r: jax.Array,
    *,
    rtol: float = 1e-8,
    log_fn=None,
):
    """Solve psi_mu @ Q = psi_r using a Galerkin projection (spin folded into μ)."""
    psi_mu = jnp.asarray(psi_mu, dtype=jnp.complex128)
    psi_r = jnp.asarray(psi_r, dtype=psi_mu.dtype)
    if psi_mu.shape[:-1] != psi_r.shape[:-1]:
        raise ValueError("psi_mu and psi_r must share all leading dimensions")

    nk, nb, nspin, n_mu = psi_mu.shape
    psi_mu_state = psi_mu.reshape(nk * nb, nspin * n_mu)
    psi_r_state = psi_r.reshape(nk * nb, -1)

    U, s, Vh = jnp.linalg.svd(psi_mu_state, full_matrices=False)
    mask = s > (s.max() * rtol)
    if not bool(mask.any()):
        raise ValueError("SVD dropped all singular values; increase rtol")
    U = U[:, mask]
    s = s[mask]
    #basis = Vh[mask].conj().T
    coeffs = U * s
    Q_reduced, *_ = jnp.linalg.lstsq(coeffs, psi_r_state, rcond=rtol)
    G = Q_reduced @ Q_reduced.conj().T
    G = G + 1e-11 * jnp.eye(G.shape[0], dtype=G.dtype)
    chol = jnp.linalg.cholesky(G)
    Q_reduced = jsp_linalg.solve_triangular(chol, Q_reduced, lower=True)
    coeffs = coeffs @ chol

    if log_fn is not None:
        residual = jnp.linalg.norm((coeffs @ Q_reduced) - psi_r_state, axis=1)
        log_fn(
            "Galerkin residual min/median/max: "
            f"{float(residual.min()):.3e} / {float(jnp.median(residual)):.3e} / {float(residual.max()):.3e}"
        )

    #Q_full = basis @ Q_reduced # do not uncomment.
    S = Q_reduced @ Q_reduced.conj().T
    ctilde = coeffs.reshape(nk, nb, coeffs.shape[1])
    return  S, ctilde


def _f_params_from_energies(enk_nb_nk: jax.Array, top_band_index: int,
                            a_band_index: int | None = None) -> tuple[float, float, float]:
    """Compute f-transform parameters from eigenvalues.

    Parameters
    ----------
    top_band_index : int
        Index of the highest band (sets epsilon0 = max eigenvalue of this band).
    a_band_index : int or None
        Index of the band whose bandwidth sets `a = 4 * bandwidth`.
        If None, defaults to top_band_index (original behavior).
        Set this to the highest band you want to keep accurately.
    """
    E_top_k = enk_nb_nk[top_band_index]
    shift = float(jnp.max(E_top_k))
    if a_band_index is None:
        a_band_index = top_band_index
    E_a_k = enk_nb_nk[a_band_index]
    gap = 4.0 * float(jnp.max(E_a_k) - jnp.min(E_a_k) + 1e-14)
    n = 3.0
    return gap, n, shift


@partial(jax.jit, static_argnames=('a', 'n', 'shift'))
def _fun_jit(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    erf_half = erf(n * 0.5)
    y = x - shift
    cond_left = y <= -a
    cond_mid = jnp.logical_and(y < 0, ~cond_left)
    f_left = y + 0.5 * a
    arg = n * (0.5 + y / a)
    term1 = a * (jnp.exp(-(n * 0.5) ** 2) - jnp.exp(-((n * (a + 2 * y)) / (2 * a)) ** 2)) / (2 * n * jnp.sqrt(jnp.pi) * erf_half)
    term2 = (a + 2 * y) * (erf_half - erf(arg)) / (4 * erf_half)
    f_mid = term1 + term2
    f = jnp.where(cond_left, f_left, 0.0)
    f = jnp.where(cond_mid, f_mid, f)
    f = jnp.where(y >= 0, 0.0, f)
    return jnp.where(f > 0, 0.0, f)


def fun(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    """Transform function f(x) — matches Fortran fun(). Works in unshifted space."""
    return _fun_jit(float(a), float(n), float(shift), x)


@partial(jax.jit, static_argnames=('a', 'n', 'shift'))
def _dfun_jit(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    erf_half = erf(n * 0.5)
    y = x - shift
    cond_left = y <= -a
    cond_mid = jnp.logical_and(y < 0, ~cond_left)
    df = jnp.zeros_like(y)
    df = jnp.where(cond_left, 1.0, df)
    arg = n * (0.5 + y / a)
    df = jnp.where(cond_mid, 0.5 - erf(arg) / (2 * erf_half), df)
    return jnp.where(y >= 0, 0.0, df)


def dfun(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    """Derivative of transform function — matches Fortran dfun()."""
    return _dfun_jit(float(a), float(n), float(shift), x)


def f_transform_eigs(enk_nb_nk: jax.Array,
                     a_band_index: int | None = None) -> tuple[jax.Array, float, float, float]:
    """Apply f-transform to eigenvalues. Returns (f_eps, a, n, shift)."""
    nb, _ = enk_nb_nk.shape
    a, n, shift = _f_params_from_energies(enk_nb_nk, top_band_index=nb - 1,
                                          a_band_index=a_band_index)
    f_eps = fun(a, n, shift, enk_nb_nk)
    return f_eps, a, n, shift


def build_R_grid_np(kgrid: tuple[int, int, int]) -> np.ndarray:
    """Lattice-R grid (nk, 3) matching the IFFT-shift convention used by
    ``make_flat_k_ifftn`` — symmetric around 0, with R_i ∈ {-n/2, ..., n/2-1}."""
    def _shift(n):
        a = np.arange(n, dtype=np.float64)
        return np.where(a >= (n + 1) // 2, a - n, a)
    return np.stack(np.meshgrid(
        _shift(kgrid[0]), _shift(kgrid[1]), _shift(kgrid[2]), indexing='ij'),
        axis=-1).reshape(-1, 3)


def build_fH_R(ctilde: jax.Array, enk_sigma: jax.Array,
               kgrid_co: tuple[int, int, int], mesh_xy: Mesh,
               *, a_band_index: int | None = None,
               log_fn=None):
    """f-transformed Hamiltonian in real-space lattice representation.

    Math (htransform paper):
        fH_k  = -Σ_n |sqrt(-f(ε_n,k))|² ctilde_n,k ctilde_n,k^H
              = Σ_n f(ε_n,k) ctilde_n,k ctilde_n,k^H
        fH_R  = (1/N_k) Σ_k e^{-2πi k·R} fH_k                          # IFFT
    where f(ε) is the smooth bandwidth-bound transform from
    ``f_transform_eigs`` (≤0 for ε<shift, =0 for ε≥shift). For any q,
    fH_q = Σ_R e^{-2πi q·R} fH_R recovers the rank-α-basis Hamiltonian
    whose eigenvalues are f(ε_n,q) and whose eigenvectors are c_n,q,
    enabling both bandstructure interpolation (eigvalsh + newton_inv on
    eigvals) and wfn recovery (eigh, then ψ_n,q(r_μ) = Σ_α c_n,q[α]·B[α,s,μ]).

    Args:
        ctilde:    (nk_co, nb, rank) Galerkin coefficients in the rank-α basis,
                   replicated. Output of ``streaming_galerkin_solve``.
        enk_sigma: (nb, nk_co) DFT band energies in Ry.
        kgrid_co:  (nkx, nky, nkz) coarse uniform k-grid.
        mesh_xy:   ('x','y') device mesh.
        a_band_index: optional band index whose bandwidth sets ``a``;
                   defaults to top of the htransform window (nb-1).
        log_fn:    optional logger.

    Returns:
        fH_k:    (nk_co, rank, rank), sharded P(None, 'x', 'y').
        fH_R:    (nk_co, rank, rank), sharded P(None, 'x', 'y') (lattice-R index).
        params:  (a_f, n_f, shift) — for ``newton_inv`` on the eigvals of fH_q.
        f_eps:   (nb, nk_co) f-transformed eigenvalues, replicated.
    """
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    f_eps, a_f, n_f, shift = f_transform_eigs(enk_sigma, a_band_index=a_band_index)
    log_fn(f"  f-transform: a={a_f:.6f} Ry, n={n_f:.2f}, shift={shift:.6f} Ry"
           + (f" (a from band {a_band_index})" if a_band_index is not None else ""))

    flat_xy = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    spec_3d = P(None, None, None, 'x', 'y')
    # 'backward' = 1/N normalisation in IFFT — matches Σ_R e^{-2πik·R} fH_R = fH_k.
    local_ifftn = make_flat_k_ifftn(mesh_xy, kgrid_co, spec_3d, norm='backward')

    @partial(jax.jit, out_shardings=(flat_xy, flat_xy))
    def _build(ctilde_in, f_eps_T):
        f_eps_ki = jnp.where(f_eps_T > 0, 0.0, f_eps_T)
        bw = jnp.sqrt(jnp.clip(-f_eps_ki, 0.0, None))
        weighted = ctilde_in * bw[..., None]
        fH_k = -jnp.einsum('kim,kin->kmn', weighted, jnp.conj(weighted), optimize=True)
        # Constrain the CONTRACTION OUTPUT, not just the hermitized copy.
        # ``ctilde`` arrives replicated, so without this the SPMD partitioner
        # has no reason to split the (nk, rank, rank) product and materialises
        # it — and then its transpose, and then the hermitized sum — at FULL
        # size on every device.  Measured on MoS2 12x12 / n_mu=2412 / nb=20
        # (rank 2880): the module wanted 57.84 GiB per device and OOMed an
        # A100-80GB, against a true sharded footprint of 2 x 1.2 GiB on a 4x4
        # mesh.  Pinning the einsum output makes the whole chain local.
        # Memory-only: same values, same shardings out.
        fH_k = jax.lax.with_sharding_constraint(fH_k, flat_xy)
        fH_k = 0.5 * (fH_k + jnp.swapaxes(fH_k, -1, -2).conj())
        fH_k = jax.lax.with_sharding_constraint(fH_k, flat_xy)
        fH_R = local_ifftn(fH_k)
        return fH_k, fH_R

    fH_k, fH_R = _build(ctilde, f_eps.T)
    return fH_k, fH_R, (a_f, n_f, shift), f_eps


def newton_inv(a: float, n: float, shift: float, y: jax.Array,
               max_iter: int = 50) -> jax.Array:
    """Newton inversion — matches Fortran newton_inv(). Works in unshifted space."""
    dxmax = a / 2.0
    x0 = y + shift  # Fortran initial guess: x = y + s

    def body_fun(_, x_curr):
        res = fun(a, n, shift, x_curr) - y
        df_val = dfun(a, n, shift, x_curr)
        dx = jnp.where(jnp.abs(df_val) > 1e-14, -res / df_val, 0.0)
        dx = jnp.clip(dx, -dxmax, dxmax)
        return x_curr + dx

    return lax.fori_loop(0, max_iter, body_fun, x0)


def load_wfns_and_enk_for_sigma(wfn, sym, nval: int, ncond: int, nband: int):
    nelec = int(wfn.nelec)
    nsigmarange = (int(nelec - nval), int(nelec + ncond))
    enk_sigma, _ = get_enk_bandrange(wfn, sym, nsigmarange, nsigmarange)
    return nsigmarange, jnp.asarray(enk_sigma).transpose(1, 0)


def setup_wfn_and_sym(wfn_file: str, mesh_xy: Mesh | None = None):
    # Pass the device mesh so the loader can pick a sharded read backend
    # (phdf5 FFI on GPU / phdf5_host union read on multi-process CPU),
    # band-sharding ψ instead of replicating the whole WFN on every rank.
    # Single-process / no-mesh transparently stays eager.
    wfn = WFNReader(wfn_file, mesh=mesh_xy)
    sym = symmetry_maps.SymMaps(wfn)
    return wfn, sym


def _clean_label(raw: str | None) -> str | None:
    if not raw:
        return None
    label = raw.strip()
    if not label:
        return None
    # Map common aliases to actual Unicode Greek letters so Matplotlib displays them
    if label.lower() in {'gg', 'gamma', '\\u0393'}:
        label = 'Γ'
    elif label.lower() in {'gl', 'lambda', '\\u039b'}:
        label = 'Λ'
    elif label.lower() in {'gs', 'sigma', '\\u03a3'}:
        label = 'Σ'
    return label


def generate_kpath_from_qe_segments(params: dict, wfn) -> tuple[jnp.ndarray, np.ndarray, list[str | None]] | None:
    seginfo = params.get("kpoints_crystal_b")
    if not seginfo:
        return None
    segments = seginfo.get("segments", [])
    if len(segments) < 2:
        return None
    nodes_crys = [np.asarray(seg["k"], dtype=float) for seg in segments]
    labels = [_clean_label(seg.get("label")) for seg in segments]
    pts_crys = [nodes_crys[0]]
    node_indices = [0]
    for i in range(len(nodes_crys) - 1):
        k0 = nodes_crys[i]
        k1 = nodes_crys[i + 1]
        n = max(1, int(segments[i + 1].get("n", 1)))
        for t in range(1, n + 1):
            alpha = t / float(n)
            pts_crys.append((1.0 - alpha) * k0 + alpha * k1)
        node_indices.append(len(pts_crys) - 1)
    kpoints = np.stack(pts_crys, axis=0)
    return jnp.asarray(kpoints, dtype=jnp.float64), np.asarray(node_indices, dtype=int), labels


def _load_centroids(centroids_path: str, fft_grid: tuple[int, int, int]) -> np.ndarray:
    centroids_frac = np.loadtxt(centroids_path, ndmin=2)
    if centroids_frac.size == 0:
        raise ValueError(f"Centroids file {centroids_path} is empty")
    fft_grid = np.asarray(fft_grid, dtype=int)
    centroid_indices = np.round(centroids_frac * fft_grid).astype(int)
    centroid_indices = np.mod(centroid_indices, fft_grid)
    return centroid_indices


def _shift_indices(n: int) -> jnp.ndarray:
    arr = jnp.arange(n, dtype=jnp.float64)
    return jnp.where(arr >= (n + 1) // 2, arr - n, arr)


def _make_logger(verbose: bool):
    return print if verbose else (lambda *_, **__: None)


def read_eqp_energies(eqp_file: str, sym, band_window: tuple[int, int]) -> jax.Array:
    """Parse EQP (or sigX) energies from a BerkeleyGW-style file.

    Supports lines like either:
      - "n=10  EDFT=  ...  EQP= -12.3456 + 0.0000i"
      - "n=10  sigX=  -12.3456 + 0.000000i  VH= ..." (x-only files)

    Returns energies shaped as (nb, nk_full) matching the requested window.
    """
    start, end = int(band_window[0]), int(band_window[1])
    nb = int(max(0, end - start))
    if nb == 0:
        raise ValueError("Empty band window requested for EQP override")

    # Precompile regexes
    k_header = re.compile(r"^\s*k-point\s+(\d+)\s*:\s*$")
    n_line = re.compile(r"n\s*=\s*(\d+)")
    eqp_val = re.compile(r"EQP\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
    sigx_val = re.compile(r"sigX\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")

    # Accumulate per-k maps from absolute band index -> energy
    energies_by_k: list[dict[int, float]] = []
    current_k: int | None = None

    with open(eqp_file, "r", encoding="utf8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m_k = k_header.match(line)
            if m_k is not None:
                # Start new k-point block
                current_k = int(m_k.group(1))
                # Ensure list is long enough
                while len(energies_by_k) <= current_k:
                    energies_by_k.append({})
                continue

            if current_k is None:
                continue

            # Parse band index and energy value
            m_n = n_line.search(line)
            if m_n is None:
                continue
            band_idx = int(m_n.group(1))

            m_eqp = eqp_val.search(line)
            val = None
            if m_eqp is not None:
                val = float(m_eqp.group(1))
            else:
                m_sigx = sigx_val.search(line)
                if m_sigx is not None:
                    val = float(m_sigx.group(1))

            if val is not None:
                energies_by_k[current_k][band_idx] = val

    if not energies_by_k:
        raise ValueError(f"No k-point blocks found in {os.path.basename(eqp_file)}")

    nk = len(energies_by_k)
    if hasattr(sym, "nk_tot"):
        expected_nk = int(sym.nk_tot)
        if nk != expected_nk:
            raise ValueError(
                f"EQP file has nk={nk}, but symmetry maps expect nk={expected_nk}"
            )
    # Build (nk, nb) by slicing the absolute band indices [start:end]
    energies_window = np.zeros((nk, nb), dtype=np.float64)
    for ik in range(nk):
        kmap = energies_by_k[ik]
        for j, b_abs in enumerate(range(start, end)):
            if b_abs not in kmap:
                raise ValueError(
                    f"Missing EQP for band {b_abs} at k-point {ik} in {os.path.basename(eqp_file)}"
                )
            energies_window[ik, j] = kmap[b_abs]

    # Return (nb, nk) to match internal conventions
    return jnp.asarray(energies_window.T, dtype=jnp.float64)


def initialize_wfns(input_path: str, params: dict, log_fn, eqp_file: str | None = None,
                    mesh_xy: Mesh | None = None, return_full_proj: bool = False):
    from file_io.centroids import load_centroids as _shared_load_centroids

    input_dir = os.path.dirname(os.path.abspath(input_path))

    def _resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(input_dir, path)

    # Build the mesh up front so the loader is mesh-aware (sharded read).
    if mesh_xy is None:
        mesh_xy = _build_mesh_xy()
    wfn_file = _resolve(params["wfn_file"])
    wfn, sym = setup_wfn_and_sym(wfn_file, mesh_xy=mesh_xy)
    centroid_path = _resolve(params.get("centroids_file", "centroids_frac.txt"))
    _, centroid_indices, n_rmu = _shared_load_centroids(centroid_path, tuple(int(x) for x in wfn.fft_grid))

    nval = int(params["nval"])
    ncond = int(params["ncond"])
    nband = int(params["nband"])
    bispinor = bool(params.get("bispinor", False))
    meta = Meta.from_system(wfn, sym, nval, ncond, nband, n_rmu, bispinor)
    nsigmarange, enk_sigma = load_wfns_and_enk_for_sigma(wfn, sym, nval, ncond, nband)

    # Optionally override energies with EQP values from a file only if explicitly requested via CLI
    if eqp_file:
        eqp_path = _resolve(eqp_file)
        if not os.path.isfile(eqp_path):
            log_fn(f"EQP file not found: {eqp_path}")
        else:
            try:
                enk_sigma = read_eqp_energies(eqp_path, sym, nsigmarange)
                log_fn(f"Using EQP energies from {os.path.basename(eqp_path)} for band window {nsigmarange}")
            except Exception as exc:
                log_fn(f"EQP override skipped for {os.path.basename(eqp_path)}: {exc}")

    band_range = (int(nsigmarange[0]), int(nsigmarange[1]))
    with mesh_xy:
        out = streaming_galerkin_solve(
            wfn, sym, meta, centroid_indices, mesh_xy, band_range,
            rtol=1e-8, log_fn=log_fn, bispinor=bispinor,
            return_full_proj=return_full_proj,
        )
    S, ctilde, B_at_mu = out[:3]
    log_fn(f"Loaded wavefunctions: nk={sym.nk_tot}, nb={band_range[1]-band_range[0]}, rank={ctilde.shape[2]}")
    if return_full_proj:
        return wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma, out[3]
    return wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma


def initialize_kpath(wfn, params):
    info = generate_kpath_from_qe_segments(params, wfn)
    if info is None:
        return None, None, None, None, []
    kpath_frac, node_indices, node_labels = info
    bvec = np.asarray(wfn.bvec, dtype=float)
    blat = float(wfn.blat)
    k_cart = np.asarray(kpath_frac) @ bvec * blat * (2.0 * np.pi)
    seg_len = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
    x_path = np.concatenate([[0.0], np.cumsum(seg_len)])
    gamma_positions = [int(idx) for idx, lbl in zip(node_indices, node_labels) if (lbl or '').strip() == '\\u0393']
    return kpath_frac, x_path, node_indices, node_labels, gamma_positions


def h_transform(meta, S, ctilde, enk_sigma, wfn, kpath_data, log_fn, mesh_xy: Mesh,
                a_band_index: int | None = None):
    nk = int(meta.nkx * meta.nky * meta.nkz)
    states = ctilde.shape[1]
    rank = ctilde.shape[2]
    kgrid = (meta.nkx, meta.nky, meta.nkz)

    rep = NamedSharding(mesh_xy, P())  # fully replicated, used for diagnostics

    coeffs = ctilde.reshape(nk, states, rank)
    fH_k, fH_R, (a_f, n_f, shift), f_eps = build_fH_R(
        coeffs, enk_sigma, kgrid, mesh_xy, a_band_index=a_band_index, log_fn=log_fn)

    # Diagnostics. Split into two small jits:
    #   _diag_stats_fast  — sharding-respecting reductions on the full fH_k
    #     (no gather needed; psum on the (x,y) face).
    #   _diag_eig_at_gamma — eigvalsh on fH_k[0] only. Pull fH_k[0:1] to
    #     replicated FIRST (7.4 MB at our scale), so the eigvalsh runs
    #     single-device locally rather than driving an all-gather of the
    #     full (rank, rank) face inside the eigvalsh module.
    @jax.jit
    def _diag_stats_fast(fH_k):
        return (jnp.min(jnp.real(fH_k)),
                jnp.max(jnp.real(fH_k)),
                jnp.max(jnp.abs(jnp.imag(fH_k))))

    @jax.jit
    def _diag_eig_at_gamma(fH_k0_rep, f_eps_col0):
        eigs0 = jnp.sort(jnp.linalg.eigvalsh(fH_k0_rep))
        f_exp0 = jnp.sort(f_eps_col0)
        eig_err = jnp.max(jnp.abs(eigs0[:f_exp0.shape[0]] - f_exp0))
        return eigs0, f_exp0, eig_err

    re_min, re_max, im_max = _diag_stats_fast(fH_k)
    log_fn("fH_k real range: [{:.3e}, {:.3e}], |imag|max={:.3e}".format(
        float(re_min), float(re_max), float(im_max)))
    fH_k0_rep = jax.device_put(fH_k[0], rep)  # (rank, rank) replicated, ~7 MB
    eigs_fHk0, f_expected_0, fH_eig_err_arr = _diag_eig_at_gamma(fH_k0_rep, f_eps[:, 0])
    fH_eig_err = float(fH_eig_err_arr)
    log_fn(f"fH(k=0) eigenvalue error vs f(eps): {fH_eig_err:.6f} Ry = {fH_eig_err * 13.6057:.3f} eV")
    log_fn(f"  f(eps) first 5: {np.array(f_expected_0[:5])}")
    log_fn(f"  fH eig first 5: {np.array(eigs_fHk0[:5])}")
    log_fn(f"  f(eps) last 5:  {np.array(f_expected_0[-5:])}")
    log_fn(f"  fH eig last 5:  {np.array(eigs_fHk0[states-5:states])}")

    # S Cholesky (rank × rank, replicated, small) — bundle the symmetrise +
    # ridge + cholesky into one jit so each line isn't a separate compile.
    @jax.jit
    def _build_S_chol(S):
        S_sym = (S + S.conj().T) * 0.5
        S_sym += 1e-10 * jnp.mean(jnp.real(jnp.diag(S_sym))) * jnp.eye(rank, dtype=S_sym.dtype)
        return jnp.linalg.cholesky(S_sym)
    S_chol = _build_S_chol(S)

    R_grid = jnp.asarray(build_R_grid_np(kgrid))

    # ── Kpath-batch processing ───────────────────────────────────────────
    # fH_R stays SHARDED P(None, 'x', 'y'): the (rank, rank) face is split
    # across the mesh, the lattice-R axis is not.  The q-Fourier sum contracts
    # ONLY over R, so every device builds its own (i, j) tile with NO
    # communication; the single collective is the reshard onto the q axis just
    # before the eigvalsh, after which each device owns whole (rank, rank)
    # matrices for its own q-rows and the eigvalsh runs ndev-parallel.
    #
    # It used to be ``fH_R_rep = jax.device_put(fH_R, rep)``.  That is
    # nk · rank² · 16 B on EVERY device (MoS2 12×12: ~51 GB/rank at rank 4716,
    # and it is the term that breaks a 90 GB rank envelope at n_μ ≈ 3.1k) and,
    # because the source is sharded, JAX routes it through ``x._value`` — a
    # host gather of the same size per process.  Identical de-replication to
    # ``bandstructure/bse_setup.py::_fourier`` (see its lines 166-178 / 247-259
    # for the measured OOM this removes); the arithmetic is unchanged.

    # Sharding specs for batched (bs, rank, rank) → (bs, rank) eigvalsh.
    batch_mat_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    batch_eig_shard = NamedSharding(mesh_xy, P(('x', 'y'), None))
    face_ij_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))

    @partial(jax.jit, out_shardings=batch_eig_shard)
    def _kpath_batch(batch_k, fH_R, S_chol):
        # batch_k: (bs, 3) replicated; fH_R: (nk, rank, rank) at P(None,'x','y');
        # S_chol: (rank, rank) replicated.  The einsum contracts over the
        # (replicated) R axis only → local on each (i, j) tile.
        phase = jnp.exp(-2j * jnp.pi * (batch_k @ R_grid.T))           # (bs, nk)
        mat = 0.5 * jnp.einsum('bk,kij->bij', phase, fH_R)             # (bs, rank, rank)
        # Pin the contraction OUTPUT to the (i, j) layout: left free, XLA
        # materialises the whole (bs, rank, rank) batch on every device before
        # the reshard below (bs·rank²·16 = 11.4 GiB/device at bs=32/rank 4716).
        mat = jax.lax.with_sharding_constraint(mat, face_ij_shard)
        # Reshard (i,j)→q FIRST, then hermitize: on the q-sharded layout each
        # device owns whole matrices, so the transpose is local.  The other
        # order costs a second all-to-all for ``swapaxes``.
        mat = jax.lax.with_sharding_constraint(mat, batch_mat_shard)
        mat = mat + jnp.swapaxes(mat, 1, 2).conj()

        def _solve_one(m):
            y = jsp_linalg.solve_triangular(S_chol, m, lower=True)
            z = jsp_linalg.solve_triangular(S_chol, y, lower=True, trans=2)
            return jnp.linalg.eigvalsh((z + z.conj().T) * 0.5)

        return jax.vmap(_solve_one)(mat)

    # Round-trip diagnostic at Γ — single q, kept (i, j)-sharded (never whole
    # on any device: rank²·16/ndev instead of rank²·16).
    q0 = jnp.zeros((1, 3), dtype=jnp.float64)
    face_one_shard = NamedSharding(mesh_xy, P('x', 'y'))

    @jax.jit
    def _gamma_rt(fH_R):
        phase0 = jnp.exp(-2j * jnp.pi * (q0 @ R_grid.T))
        m = 0.5 * jnp.einsum('bk,kij->bij', phase0, fH_R)
        m = jax.lax.with_sharding_constraint(m, face_ij_shard)
        m = (m + jnp.swapaxes(m, 1, 2).conj())[0]
        return jax.lax.with_sharding_constraint(m, face_one_shard)

    fH_gamma_rt = _gamma_rt(fH_R)
    diffs = jnp.max(jnp.abs(fH_k - fH_gamma_rt), axis=(1, 2))
    rt_err = float(jnp.min(diffs))
    log_fn(f"FFT Γ round-trip max error: {rt_err:.3e}")

    fermi_energy = float(wfn.efermi)
    nb_keep = int(f_eps.shape[0])

    kpath_frac, x_path, node_indices, node_labels, gamma_positions = kpath_data
    energies_on_path = None
    energies_sorted = None
    path_range = None
    gamma_exact = None

    if kpath_frac is not None:
        wrapped_k = (kpath_frac + 0.5) % 1.0 - 0.5
        # Pad nq to a multiple of batch_size — every batch has the same
        # shape, _kpath_batch compiles ONCE.
        batch_size = 32
        nq = wrapped_k.shape[0]
        n_pad = (-nq) % batch_size
        if n_pad:
            wrapped_k = jnp.concatenate(
                [wrapped_k, jnp.zeros((n_pad, 3), dtype=wrapped_k.dtype)], axis=0)
        nq_padded = wrapped_k.shape[0]
        lambda_q_list = []
        for i in range(0, nq_padded, batch_size):
            batch_eigs = _kpath_batch(wrapped_k[i:i+batch_size], fH_R, S_chol)
            lambda_q_list.append(batch_eigs)
            jax.block_until_ready(batch_eigs)

        # Bundle concat + slice + vmap(newton_inv) + sort into ONE jit so the
        # post-loop processing emits one compile rather than 4 (concatenate,
        # sort, gather, vmap-newton).
        @partial(jax.jit, static_argnames=('nq', 'nb_keep'))
        def _post_kpath(batches, nq, nb_keep):
            lambda_q = jnp.concatenate(batches, axis=0)[:nq]
            energies = jax.vmap(lambda row: newton_inv(a_f, n_f, shift, row.real))(lambda_q)
            energies_sorted = jnp.sort(energies, axis=1)[:, :nb_keep]
            return energies, energies_sorted

        energies_on_path, energies_sorted_jax = _post_kpath(
            tuple(lambda_q_list), int(nq), int(nb_keep))
        energies_sorted = np.asarray(energies_sorted_jax)
        # Determine Fermi energy as the maximum along path of the wfn.nelec-th band (1-based -> 0-based)
        fermi_band_idx = int(wfn.nelec) - 1
        if 0 <= fermi_band_idx < energies_sorted.shape[1]:
            fermi_energy = float(np.max(energies_sorted[:, fermi_band_idx]))
        if not gamma_positions:
            gamma_positions = [int(jnp.argmin(jnp.linalg.norm(wrapped_k, axis=1)))]
        gamma_exact = np.sort(np.asarray(enk_sigma[:, 0]))[:nb_keep]
        # Shift all reported energies so VBM is at 0
        if energies_on_path is not None:
            energies_on_path = energies_on_path - fermi_energy
        if energies_sorted is not None:
            energies_sorted = energies_sorted - fermi_energy
        if gamma_exact is not None:
            gamma_exact = gamma_exact - fermi_energy
        # Recompute path range after shift and report
        path_range = (float(energies_sorted.min()), float(energies_sorted.max()))
        log_fn(f"Path energy range: {path_range[0]:.6f} to {path_range[1]:.6f} Ry (VBM@0)")
        # Γ deltas (in mRy) remain unchanged by uniform shift
        delta = (energies_sorted[gamma_positions[0]] - gamma_exact) * 1000.0
        log_fn("Γ Δε (mRy): " + ", ".join(f"{d:+.2f}" for d in delta[:6]))
        # After shifting, the Fermi level indicator is at 0
        fermi_energy = 0.0

    return {
        "nk_total": nk,
        "nb_keep": nb_keep,
        "fermi_energy": fermi_energy,
        "energies_on_path": energies_on_path,
        "energies_sorted": energies_sorted,
        "path_range": path_range,
        "gamma_exact": gamma_exact,
        "kpath_data": (kpath_frac, x_path, node_indices, node_labels, gamma_positions),
    }


def plot_bands(result):
    kpath_frac, x_path, node_indices, node_labels, gamma_positions = result["kpath_data"]
    energies_sorted = result["energies_sorted"]
    gamma_exact = result["gamma_exact"]
    fermi_energy = result["fermi_energy"]
    nb_keep = result["nb_keep"]

    if kpath_frac is None or energies_sorted is None:
        raise RuntimeError("Plotting requires a K_POINTS {crystal_b} path in the input file")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots()
    for band in range(nb_keep):
        ax.plot(x_path, energies_sorted[:, band], lw=1.0, color='C0', alpha=0.9)

    x_ticks = x_path[np.asarray(node_indices, dtype=int)]
    labels = [(lbl or "") for lbl in node_labels]
    for xpos in x_ticks:
        ax.axvline(xpos, color='k', lw=0.6, alpha=0.3)
    ax.set_xticks(x_ticks, labels)

    for pos_idx, idx in enumerate(gamma_positions or [0]):
        xpos = x_path[idx]
        label_exact = 'Exact Γ' if pos_idx == 0 else None
        label_ht = 'HT Γ' if pos_idx == 0 else None
        if gamma_exact is not None:
            ax.scatter(np.full(nb_keep, xpos), gamma_exact, marker='o', facecolors='none', edgecolors='red', label=label_exact)
        ax.scatter(np.full(nb_keep, xpos), energies_sorted[idx], marker='x', color='black', label=label_ht)

    ax.axhline(fermi_energy, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='$E_F$')
    ax.set_xlabel('k-path arc length (2π-scaled)')
    ax.set_ylabel('Energy (Ry)')
    ax.set_title('Hamiltonian-transform bands')
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    fig.tight_layout()
    plt.show() 


def write_bands_to_file(output_path: str, energies_on_path, kpath_frac, x_path):
    if energies_on_path is None or kpath_frac is None or x_path is None:
        return
    energies = np.asarray(energies_on_path)
    kpoints = np.asarray(kpath_frac)
    with open(output_path, 'w', encoding='utf8') as fh:
        fh.write('# idx_k idx_b kx ky kz s energy\n')
        for ik in range(energies.shape[0]):
            for ib in range(energies.shape[1]):
                kx, ky, kz = kpoints[ik]
                s_coord = x_path[ik]
                fh.write(f"{ik:4d} {ib:4d} {kx: .8f} {ky: .8f} {kz: .8f} {s_coord: .8f} {energies[ik, ib]: .8f}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Hamiltonian interpolation driver")
    parser.add_argument("-i", "--input", default="cohsex_test.in", help="Input file")
    parser.add_argument("-wfn", "--wfn-file", default=None, help="Override WFN file (e.g. WFN_qp.h5)")
    parser.add_argument("--plot", action="store_true", help="Show interpolated band plot")
    parser.add_argument("--eqp-file", default=None, help="Path to EQP/sigX file to override DFT band energies")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic details")
    parser.add_argument("--a-band", type=int, default=None,
                        help="Band index (0-based) whose bandwidth sets 'a'. "
                             "E.g. nval+ncond_keep-1. Default: top band.")
    parser.add_argument("--eigh-backend", default=None,
                        choices=("auto", "off", "cusolvermp", "slate"),
                        help="Eigensolver for the fH_q eigendecomposition of "
                             "the get_centroids_fi handoff.  auto|off = the "
                             "q-batched native path; cusolvermp|slate spread "
                             "ONE (rank, rank) tile over the mesh via the "
                             "distributed-linalg FFI (wide band windows).  "
                             "OVERRIDES the input-file ``eigh_backend`` key "
                             "(default: use the key, which defaults to auto).")
    args = parser.parse_args(argv)
    log = _make_logger(args.verbose)

    # JAX persistent compile cache — the same call/pattern as gw_jax's
    # _warm_start (and run_nscf / run_sternheimer / kmeans_cli).  This CLI
    # was the one driver that never enabled it, so every htransform run paid
    # a cold XLA compile of the Galerkin/G-accum kernels.  Opt out (and the
    # ONLY supported setting at P > 1 on Frontera, where a shared cache
    # deadlocks the compile barrier) with ISDF_JAX_CACHE_DIR="" — that env
    # value is honoured inside ensure_jax_compile_cache, so nothing here
    # needs to test for it.  Failures are logged and swallowed.
    try:
        from common.jax_compile_cache import ensure_jax_compile_cache
        ensure_jax_compile_cache()
    except Exception as exc:
        print(f"  [jax compile cache] skipped: {exc}", flush=True)

    from gw.gw_init import read_cohsex_input
    params = read_cohsex_input(args.input)
    # Input file is the source of truth; the CLI flag is an override.
    eigh_backend = (args.eigh_backend if args.eigh_backend is not None
                    else str(params.get("eigh_backend", "auto")))
    
    # Override WFN file if provided via CLI
    if args.wfn_file is not None:
        params["wfn_file"] = args.wfn_file
        log(f"Using WFN file from CLI: {args.wfn_file}")

    from common import sanity

    wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma = initialize_wfns(
        args.input, params, log, args.eqp_file)
    # ── Galerkin-input gate ───────────────────────────────────────────
    # S is the ISDF overlap Gram matrix (Hermitian positive-definite by
    # construction) and ``enk_sigma`` is the band energies the whole
    # interpolation is anchored to — including, when ``--eqp-file`` is
    # given, energies read from a GW run that may itself have produced
    # garbage.  A −136 eV QP energy fed into htransform yields a
    # bandstructure.dat that is numerically finite, plots fine, and is
    # wrong.  Bracket it here, where the file name is still in scope.
    sanity.check_finite("htransform S (ISDF overlap)", S, print_fn=log)
    sanity.check_finite("htransform ctilde", ctilde, print_fn=log)
    sanity.check_finite("htransform E_nk (Ry)", enk_sigma, print_fn=log)
    # Bandwidth, not absolute energy: the zero of a pseudopotential
    # eigenvalue is convention-dependent, but the *spread* of a Σ-window
    # band set is not — 20 Ry (272 eV) is far wider than any real
    # semicore-to-conduction window and so only fires on gross
    # corruption.  A subtler check (comparing --eqp-file energies against
    # the DFT ones they replace) is proposed but not implemented here;
    # see the workstream-O report.
    _enk = np.asarray(jax.device_get(enk_sigma), dtype=np.float64)
    if _enk.size:
        _spread = float(_enk.max() - _enk.min())
        log(f"  E_nk spread: {_spread:.4f} Ry over {_enk.size} states")
        sanity.check_in_range(
            "htransform E_nk bandwidth (Ry)", np.array([_spread]),
            0.0, 20.0, unit="Ry", print_fn=log)

    kpath_data = initialize_kpath(wfn, params)
    with mesh_xy:
        result = h_transform(meta, S, ctilde, enk_sigma, wfn, kpath_data, log, mesh_xy,
                             a_band_index=args.a_band)

    # Optional BSE interpolation handoff: fine-k wfns at coarse centroids.
    # Driven by ``get_centroids_fi`` + ``kgrid_fi`` + ``wfn_fi_{min,max}``;
    # see ``bandstructure.bse_setup.compute_wfns_fi`` for the contract.
    if params.get("get_centroids_fi", False):
        from .bse_setup import compute_wfns_fi
        b_min = int(params["wfn_fi_min"])
        b_max = int(params["wfn_fi_max"]) or int(ctilde.shape[1])
        with mesh_xy:
            wfns_fi = compute_wfns_fi(
                ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
                kgrid_co=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
                kgrid_fi=params["kgrid_fi"],
                band_window_fi=(b_min, b_max),
                mesh_xy=mesh_xy, a_band_index=args.a_band,
                eigh_backend=eigh_backend, log_fn=log,
            )
        log(f"BSE setup: psi_rmu_Y={wfns_fi.psi_rmu_Y.shape} "
            f"P{wfns_fi.psi_rmu_Y.sharding.spec}, "
            f"psi_rmuT_X={wfns_fi.psi_rmuT_X.shape} "
            f"P{wfns_fi.psi_rmuT_X.sharding.spec}, "
            f"enk_full={wfns_fi.enk_full.shape}")

    if args.plot:
        plot_bands(result)

    # ── Writer gate ───────────────────────────────────────────────────
    # bandstructure.dat is the file downstream tooling and the regression
    # gate diff against ground truth; a NaN row silently changes the file
    # length rather than the exit code.
    sanity.check_finite("bandstructure.dat energies",
                        result['energies_sorted'], print_fn=log)

    output_dir = os.path.dirname(os.path.abspath(args.input))
    write_bands_to_file(
        os.path.join(output_dir, 'bandstructure.dat'),
        result['energies_sorted'],  # Use sorted & truncated to nb_keep, not raw eigenvalues
        kpath_data[0],
        kpath_data[1],
    )

    summary = f"HT complete: {result['nb_keep']} bands, nk={result['nk_total']}, fermi={result['fermi_energy']:.6f} Ry"
    if result['path_range'] is not None:
        summary += f", path range [{result['path_range'][0]:.6f}, {result['path_range'][1]:.6f}] Ry"
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
