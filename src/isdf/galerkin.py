"""Streamed real-space Galerkin projection over a two-dimensional mesh.

This module owns the named distributed pattern shared by operator-basis
fits: for each bounded real-space chunk, sum every contracted band chunk
into ``Q`` before folding ``Q Q^H`` into the projected Gram matrix.  The
ordering is load-bearing.  Folding once per band chunk would omit all
cross-band terms while still producing a plausible Hermitian matrix.

Wavefunction loading and G-to-r transforms remain owned by the reusable
``common.psi_G_store`` source and its canonical transform helpers.  The caller
resolves policy such as environment overrides and the device-pool limit, then
passes explicit values here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg as jsp_linalg
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import rank_criterion, spectral_closure
from common.psi_G_store import build_psi_G_store
from common.shard_map import shard_map
from common.sharding_fit import fit_sharding as _fit
from common.wfn_layout import band_sphere_spec
from common.wfn_transforms import (
    load_centroids_band_chunked,
    load_psi_gflat_padded,
)
from runtime.padding import round_up, spec_divisor


__all__ = [
    "GalerkinBasis",
    "GalerkinStreamPlan",
    "build_streamed_projected_gram",
    "fit_galerkin_basis",
    "galerkin_q_ledger",
    "plan_galerkin_stream",
    "validate_rank_multiplier",
]


@dataclass(frozen=True)
class GalerkinBasis:
    """One fitted interpolation basis in a single shared alpha gauge.

    ``ctilde`` and ``basis_at_nodes`` are inseparable: independently replacing
    either array changes the gauge and invalidates reconstruction.  The
    optional ``projector`` is in that same gauge.  ``rank_physical`` excludes
    exact-null mesh padding, so persistence can be mesh-independent and a
    reader can reconstruct the carrier required by its own mesh.
    """

    ctilde: jax.Array
    basis_at_nodes: jax.Array
    rank_physical: int
    band_range: tuple[int, int]
    projector: jax.Array | None = None

    def __post_init__(self) -> None:
        if self.ctilde.ndim != 3:
            raise ValueError(
                f"GalerkinBasis.ctilde must be (nk, nb, rank); got "
                f"{tuple(self.ctilde.shape)}")
        if self.basis_at_nodes.ndim != 3:
            raise ValueError(
                "GalerkinBasis.basis_at_nodes must be (rank, ns, n_nodes); "
                f"got {tuple(self.basis_at_nodes.shape)}")
        rank = int(self.ctilde.shape[2])
        if int(self.basis_at_nodes.shape[0]) != rank:
            raise ValueError(
                "GalerkinBasis gauge mismatch: ctilde rank "
                f"{rank} != basis_at_nodes rank "
                f"{int(self.basis_at_nodes.shape[0])}")
        if not 0 < int(self.rank_physical) <= rank:
            raise ValueError(
                f"GalerkinBasis.rank_physical={self.rank_physical} must lie "
                f"in [1, carried rank {rank}]")
        b0, b1 = (int(v) for v in self.band_range)
        if b1 - b0 != int(self.ctilde.shape[1]):
            raise ValueError(
                f"GalerkinBasis band range [{b0},{b1}) has width {b1-b0}, "
                f"but ctilde carries {int(self.ctilde.shape[1])} bands")
        if self.projector is not None:
            expected = (rank, int(self.ctilde.shape[0] * self.ctilde.shape[1]))
            if tuple(self.projector.shape) != expected:
                raise ValueError(
                    f"GalerkinBasis.projector must have shape {expected}; "
                    f"got {tuple(self.projector.shape)}")

    @property
    def rank_carrier(self) -> int:
        return int(self.ctilde.shape[2])

    def identity_metric(self, mesh_xy: Mesh) -> jax.Array:
        """Reconstruct the legacy exact-identity overlap on ``mesh_xy``."""
        rep = NamedSharding(mesh_xy, P())
        rank = self.rank_carrier
        return jax.jit(
            lambda: jnp.eye(rank, dtype=jnp.complex128),
            out_shardings=rep)()

    def as_legacy_tuple(self, mesh_xy: Mesh, *, include_projector: bool):
        """Compatibility surface for existing tuple-based consumers."""
        out = (self.identity_metric(mesh_xy), self.ctilde,
               self.basis_at_nodes)
        if include_projector:
            if self.projector is None:
                raise ValueError(
                    "GalerkinBasis has no projector, but the legacy caller "
                    "requested one")
            return out + (self.projector,)
        return out


def validate_rank_multiplier(value, *, name: str = "rank_multiplier") -> float:
    """Validate an optional shared cross-k Galerkin model-order multiplier."""
    try:
        multiplier = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name}={value!r} is not a finite number; use 0 for the exact "
            "numerical-rank path or a value >= 1.") from None
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError(f"{name}={value!r} must be finite and >= 0.")
    if 0.0 < multiplier < 1.0:
        raise ValueError(
            f"{name}={multiplier:g} would retain fewer directions than bands "
            "at one k. Use 0 for the exact path or a value >= 1.")
    return multiplier


def _lowdin_orthonormalize_band_rows(ctilde: jax.Array):
    """Per-k polar/Löwdin row orthonormalization for a reduced shared span."""
    gram = jnp.einsum('kna,kma->knm', ctilde, jnp.conj(ctilde),
                      optimize=True)
    gram = 0.5 * (gram + jnp.swapaxes(gram, -1, -2).conj())
    evals, evecs = jnp.linalg.eigh(gram)
    safe_evals = jnp.maximum(evals, jnp.finfo(evals.dtype).tiny)
    invsqrt = jnp.einsum(
        'kni,ki,kmi->knm', evecs, 1.0 / jnp.sqrt(safe_evals),
        jnp.conj(evecs), optimize=True)
    out = jnp.einsum('knm,kma->kna', invsqrt, ctilde, optimize=True)
    eye = jnp.eye(ctilde.shape[1], dtype=ctilde.dtype)[None]
    gram_out = jnp.einsum('kna,kma->knm', out, jnp.conj(out),
                          optimize=True)
    before = jnp.max(jnp.abs(gram - eye))
    after = jnp.max(jnp.abs(gram_out - eye))
    rel_move = jnp.max(
        jnp.linalg.norm(out - ctilde, axis=(-2, -1)) /
        jnp.maximum(jnp.linalg.norm(ctilde, axis=(-2, -1)),
                    jnp.finfo(ctilde.real.dtype).tiny))
    return out, jnp.min(evals), jnp.max(evals), before, after, rel_move


def fit_galerkin_basis(
        wfn, sym, meta, centroid_indices, mesh_xy: Mesh,
        band_range: tuple[int, int], *,
        rtol: float = 1e-8, log_fn=None,
        band_chunk_size: int = 64,
        bispinor: bool = False,
        include_projector: bool = False,
        eigh_backend: str = "auto",
        eigh_plan=None,
        rank_multiplier: float = 0.0,
        q_tile_budget: int,
        device_pool_limit: float | None,
        rank_policy_mode: str = "refuse",
        extra_rank_pad: int = 0,
        progress_fn=None,
        rank_record_fn=None,
) -> GalerkinBasis:
    """Fit a reusable Galerkin basis from one wavefunction band window.

    Single ('x','y') mesh throughout. ψ at centroids comes from
    ``load_centroids_band_chunked``; ψ at full r is served from one
    ``common.psi_G_store.PsiGStore`` through
    ``build_streamed_projected_gram``. G is built
    sharded ``P('x','y')`` and Cholesky-factored without changing its gauge.

    MEMORY AND SHARDING (2026-08-24).  The accumulator is
    ``Q_chunk[rank, ns, r_chunk_carrier]``: r chunks are outermost, and
    every band chunk is summed into Q_chunk before its Gram contribution.
    Q over ``r_tot`` is never materialized.  The free r axis is zero-padded
    to the mesh product and sharded over ('y','x'); each streamed ψ chunk goes
    directly from
    product-band to product-r sharding through the two volume-preserving
    exchanges owned by ``common.staged_reshard`` before the pinned
    contraction.  The y-replicated carrier and the merged-axis layout that
    forced the SPMD
    partitioner's full-replication fallback at P16 (JID 57271407) is
    gone.  ``isdf.galerkin.galerkin_q_ledger`` prices the largest Q chunk
    (accumulation overlap plus fold workspace), both equal-volume ψ
    transition shards, replicated Gram state, Vh and G per device, then
    refuses a non-fitting chunk before compilation.

    Args:
        wfn, sym, meta: standard gw_jax handles.
        centroid_indices: (n_μ, 3) FFT-grid coordinates.
        mesh_xy: 2D ``('x','y')`` device mesh.
        band_range: ``(b_start, b_end)`` — bands included in the α basis.
        rtol: σ truncation tolerance (relative to s.max()); σ come from the
            Gram-eigh of A Aᴴ (see step 2), the same criterion as the old
            replicated SVD up to sqrt-of-eigenvalue round-off.
        eigh_backend: distrib_la backend for the (nk·nb)² Gram eigh — the
            same plan family (and deck key) as the fH_q eigh downstream;
            ``auto`` = native replicated eigh, ``distributed`` = ScaLAPACK
            pzheevd, one tile over the mesh.
        rank_multiplier: 0 (default) carries the full numerical rank.  A
            value >= 1 targets ``ceil(rank_multiplier * nb)`` shared alpha
            directions, capped by the numerical rank, then Löwdin-
            orthonormalizes each k's band rows.  This is an explicit
            model-order approximation, not another spelling of ``rtol``.
        band_chunk_size: throughput ceiling for bands per FFT chunk inside
            the loader.  The r-outer planner jointly bounds Q and the ψ
            transition layouts after the retained rank is known.
        bispinor: passed through to ``load_centroids_band_chunked``.
        include_projector: also return the full-r α-basis projector
            ``W_proj = L⁻¹ diag(1/s) U^H`` (rank, nk·nb), replicated.  For
            any streamed ψ chunk (nk, bc, ns, r_c) restricted to the SAME
            band window, ``B_full[α, s·r_c] = W_proj_bc @ ψ_flat`` evaluates
            the α-basis on the full r-grid, so
            ``ψ_{n,q}(r) = Σ_α c_{n,q}[α] B_full[α](r)`` reconstructs ψ at
            ANY q off the grid — the per-Q ζ-refit consumer
            (``bse.vq_interp.refit_vq``).

    Returns a :class:`GalerkinBasis`.  Its coefficient and node-basis arrays
    share one alpha gauge and must be persisted/restarted together.
    """
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    b_start, b_end = band_range
    nb = b_end - b_start
    nk = meta.nk_tot
    nspinor = meta.nspinor
    n_mu = int(centroid_indices.shape[0])
    n_rtot = int(meta.n_rtot)
    q_spec = P(None, None, ('y', 'x'))
    r_mesh_divisor = spec_divisor(mesh_xy, q_spec, axis=2)
    n_r_carrier = round_up(n_rtot, r_mesh_divisor)
    rank_multiplier = validate_rank_multiplier(rank_multiplier)
    extra_rank_pad = int(extra_rank_pad)
    if extra_rank_pad < 0:
        raise ValueError(
            f"extra_rank_pad must be >= 0; got {extra_rank_pad}")
    if int(q_tile_budget) <= 0:
        raise ValueError(
            f"q_tile_budget must be positive; got {q_tile_budget}")
    if rank_multiplier > 0.0 and include_projector:
        raise ValueError(
            "fit_galerkin_basis: rank_multiplier cannot be combined with "
            "include_projector. The reduced route applies "
            "a k-dependent Löwdin map to ctilde, while W_proj is one global "
            "full-r projector; pretending they share one alpha gauge would "
            "give the consumer wrong wavefunctions. Use rank_multiplier=0 "
            "when a projector is required.")

    rep = NamedSharding(mesh_xy, P())               # fully replicated
    grid_xy = NamedSharding(mesh_xy, P('x', 'y'))   # (rank, rank) face

    # Bound the full-grid FFT source before the retained rank is known.  The
    # later r planner bounds Q and its sliced transition layouts, but
    # ``PsiGStore`` still transforms a full-grid band carrier before taking a
    # requested r slab. Keep that source below the same streaming budget; the
    # mesh-aligned carrier is the minimum useful width on a product-band mesh.
    stream_budget = int(q_tile_budget)
    bytes_per_band = nk * nspinor * n_r_carrier * 16
    bc_cap = max(1, stream_budget // max(1, bytes_per_band))
    band_chunk_size = max(1, min(int(band_chunk_size), bc_cap, nb))
    # A product-band chunk smaller than the mesh still occupies one band on
    # every device after the loader's canonical band pad.  Carry that actual
    # width through the banner, range construction, FFT cache and ledger.
    _p_band = spec_divisor(mesh_xy, band_sphere_spec(), axis=1)
    _bc = round_up(band_chunk_size, _p_band)
    _n_band_chunks = (nb + _bc - 1) // _bc
    _band_chunk_ranges = tuple(
        (b_start + i * _bc, min(b_start + (i + 1) * _bc, b_end))
        for i in range(_n_band_chunks))

    _rank_policy = ("numerical (exact-span default)" if rank_multiplier == 0.0
                    else f"target {rank_multiplier:g}*nb="
                         f"{math.ceil(rank_multiplier * nb)}")
    log_fn(
        f"  Streaming Galerkin: nk={nk}, nb={nb}, nr={n_rtot}"
        + (f" -> {n_r_carrier} zero-padded carrier" if n_r_carrier != n_rtot
           else "") + ", "
        f"n_mu={n_mu}, mesh=({mesh_xy.shape['x']}x{mesh_xy.shape['y']}), "
        f"band_chunk={_bc}"
        + (f" (planner {band_chunk_size}, mesh-aligned)"
           if _bc != band_chunk_size else "")
        + f", rank={_rank_policy}"
    )
    log_fn(
        f"  FFT source chunk <= "
        f"{bytes_per_band * _bc / mesh_xy.size / 1024**3:.2f} GiB/device "
        f"({_bc} mesh-aligned bands; streaming budget "
        f"{stream_budget / 1024**3:.2f} GiB)")
    # Q over r_tot is never materialized.  The retained rank is not known
    # until the centroid Gram-eigh, so the exact r-chunk and live-set ledger
    # are printed there.
    _rank_cap = min(nk * nb, nspinor * n_mu)
    log_fn(f"  Q rank ceiling={_rank_cap}; the zeta-style outer-r planner "
           f"will bound each Q shard after rank selection")
    if _n_band_chunks > 1:
        log_fn(f"  (band axis split into "
               f"{_n_band_chunks} chunks; G is "
               f"accumulated r-outer / band-inner so the split stays exact)")

    # ── 1. Load ψ(G-flat) ONCE for the whole band window ──
    # One capped+padded load (band-sharded over ('x','y')) serves centroid
    # sampling.  It must NOT be dynamically sliced along that sharded band
    # axis for the later r sweep: GSPMD gathers the full window to perform
    # that slice (15.36 GiB/device on this CrI3 deck).  Step 3 therefore uses
    # the iterator's bounded per-chunk loader and this one-shot window is
    # released as soon as its centroid consumer has been dispatched.
    t0 = time.time()
    psi_G_win = load_psi_gflat_padded(
        wfn, band_range, mesh_xy=mesh_xy, bispinor=bispinor)
    if psi_G_win is None:
        raise ValueError(
            f"fit_galerkin_basis: band window {band_range} lies "
            f"entirely past the file's band extent ({int(wfn.nbands)})")
    log_fn(f"  ψ(G-flat) window load: {time.time()-t0:.2f}s "
           f"(shape {tuple(psi_G_win.shape)}, centroid sample only)")

    # ── 1b. ψ at centroids (band-sharded internally on 'y') ──
    psi_rmu_Y, _ = load_centroids_band_chunked(
        wfn, sym, meta, centroid_indices, bispinor, mesh_xy,
        band_range=band_range, band_chunk_size=band_chunk_size,
        psi_G_flat=psi_G_win,
    )
    del psi_G_win
    # psi_rmu_Y: (nk, nb, ns, n_μ), sharded P(None, None, None, 'y')
    log_fn(f"  load_centroids_band_chunked: {time.time()-t0:.2f}s")
    # Captured for the Q memory ledger below — psi_rmu_Y is deleted before
    # the ledger runs.
    mu_pad = int(psi_rmu_Y.shape[3])

    # ── 2. Gram-eigh of A Aᴴ, A = ψ@centroids reshaped to (nk·nb, ns·n_μ) ──
    # Until 2026-08-01 this step gathered A REPLICATED and ran a dense SVD on
    # every rank — the last N_μ-scaling replicated core in the chain (A itself
    # nk·nb·ns·N_μ·16 B/rank plus the gesdd workspace, and Vh/B_at_mu
    # re-replicated downstream).  A Aᴴ is (nk·nb, nk·nb) — N_μ-FREE — and
    # eigh(A Aᴴ) gives the same left factor: λ = σ², U = eigenvectors, so
    # σ = sqrt(λ) and Vᴴ = diag(1/σ) Uᴴ A is formed μ-SHARDED where consumed
    # (see ``_vh_sharded`` below).  The Gram product contracts (s, μ) locally
    # on each μ-shard + one (nk·nb)² all-reduce; ψ@centroids is never
    # replicated and no O(N_μ) object is.
    #
    # NUMERICS.  (a) Gauge, not bits: eigenvectors of A Aᴴ match the SVD's U
    # only up to per-σ phases (rotations inside degenerate σ groups), so
    # ctilde/B/fH transform covariantly and every physical output (energies,
    # ψ reconstruction) is invariant analytically, equal to the SVD path to
    # ~κ·ε in floats — the same "different valid gauge" class as the
    # distributed ζ tier.  (b) Squaring halves the precision of SMALL σ:
    # sqrt(λ) resolves σ only down to ~sqrt(ε)·σ_max ≈ 1.5e-8·σ_max, i.e.
    # exactly the rtol=1e-8 cut.  That is safe HERE because the retained
    # spectrum is bounded away from the cut (measured job 7883150:
    # σ_min/σ_max = 2.25e-5, 4.5 decades above it — see the truncation block
    # below); a deck whose retained block hugged the cut would fail the
    # ``rank_report`` violations gate, not silently drift.
    #
    # The eigh goes through the distrib_la plan family: ``eigh_backend`` from
    # the deck (auto = native batched eigh, replicated — (nk·nb)² is small;
    # distributed = ScaLAPACK pzheevd, one tile over the mesh, for band
    # windows where even (nk·nb)² replicated is unaffordable).
    t1 = time.time()
    m_states = nk * nb

    @partial(jax.jit, out_shardings=rep)
    def _gram(psi):
        # psi: (nk, nb, ns, μ_pad) at P(None, None, None, 'y').  dot_general
        # over BOTH (s, μ) axes — no ns·μ reshape ever touches the sharded
        # axis.  Zero μ-pad columns add exact zeros to the sum.
        M = jnp.einsum('aism,bjsm->aibj', psi, jnp.conj(psi), optimize=True)
        M = M.reshape(m_states, m_states)
        return 0.5 * (M + M.conj().T)

    M = _gram(psi_rmu_Y)

    if eigh_plan is None:
        from ffi import _services
        _services.ensure_on_path()
        from distrib_la import plan as linalg_plan
        eigh_plan = linalg_plan(
            "eigh", mesh_xy, backend=eigh_backend, n=m_states)
    elif eigh_plan.op != "eigh" or int(eigh_plan.n) != int(m_states):
        raise ValueError(
            "fit_galerkin_basis: the pre-resolved distrib_la plan is "
            f"{eigh_plan.op}/n={eigh_plan.n}, expected eigh/n={m_states}.")
    if eigh_plan.is_native:
        @partial(jax.jit, out_shardings=(rep, rep))
        def _eigh_native(M_):
            w_, V_ = jnp.linalg.eigh(M_)
            return w_, V_
        w, V = _eigh_native(M)
    else:
        log_fn("  Gram eigh: " + eigh_plan.describe())
        w, V = eigh_plan(M)          # λ replicated, V one tile P('x','y')
        V = jax.device_put(V, rep)   # (nk·nb)² — N_μ-free, small
    del M

    # eigh is ascending; the SVD contract everywhere below is descending σ.
    @partial(jax.jit, out_shardings=(rep, rep))
    def _sv_from_eigs(w, V):
        s = jnp.sqrt(jnp.clip(w[::-1], 0.0, None))
        return s, V[:, ::-1]

    s, U = _sv_from_eigs(w, V)
    del w, V
    s_host = np.asarray(s)

    # ── The truncation criterion ──────────────────────────────────────────
    # ``rank_numerical`` is the ONLY place numerical admissibility is decided,
    # and the decision is a cap on how much the pseudo-inverse below (``inv_s``
    # = 1/σ) may amplify round-off: keep σ > σ_max/κ_cap with κ_cap = 1/rtol.
    # It is NOT a search for a gap — these are ISDF/Galerkin overlap spectra
    # and they are smooth by construction, with no knee to find.  See
    # ``common/rank_criterion`` for the criterion, for the three standard
    # alternatives (discrepancy principle / L-curve / GCV) and the measured
    # reason each is refuted here, and for the §R19 table in which retaining
    # 41 % MORE rank moved a QP gap from 3.13 eV to −5049 eV.
    #
    # THE STRUCTURAL CEILING, and why this route needs it more than any other.
    # ``A`` is (nk·nb, nspinor·n_μ) but the spectrum above comes from an eigh
    # of ``A Aᴴ`` at the LARGER of the two dimensions (see the NUMERICS note),
    # so the null space of a tall rank-deficient A arrives as round-off-sized
    # POSITIVE eigenvalues and a relative threshold COUNTS THEM.  Measured on
    # Na bands 1–24: A is (12288, 2032) and rtol=1e-8 selected rank 2034 —
    # two directions more than the matrix algebraically has — after which the
    # capacity line printed the self-contradiction ``rank=2034`` beside
    # ``nspinor·n_μ = 2032``.  ``min(rows, cols)`` is the ceiling; the report
    # below refuses an UNCLAMPED overshoot so a future caller that drops this
    # argument is not silently believed.
    _rank_ceiling = min(nk * nb, nspinor * n_mu)
    rank_numerical = rank_criterion.select_rank(
        s_host, rtol, ceiling=_rank_ceiling)

    # ── …and the criterion is not allowed to stop mid-multiplet ───────────
    # ``common/spectral_closure``, the sibling of the band-window guard.  A
    # cut through a degenerate σ block keeps a symmetry-ARBITRARY slice of an
    # eigenspace, and this function's own NUMERICS note (a) says so from the
    # other side: "eigenvectors of A Aᴴ match the SVD's U only up to per-σ
    # phases (rotations inside degenerate σ groups)".  That rotation freedom
    # is harmless while a degenerate group is retained WHOLE — every physical
    # output is invariant under it — and is exactly what breaks covariance
    # when the group is split.  The repair drops the straddled group whole
    # (owner ruling 2026-08-10), so ``rank_phys`` comes DOWN and the κ_cap
    # this criterion just enforced holds with room to spare; the padding
    # below then aligns whatever survives.  See that module's TWO-RULE FAMILY
    # on why the BAND-window guard rounds the other way and refuses instead.
    rank_numerical, _sc_numerical = spectral_closure.resolve_spectral_cut(
        s_host, rank_numerical,
        where="Galerkin psi@centroids Gram-eigh sigma numerical rank",
        rcond=rtol, log=log_fn)
    if not _sc_numerical["fired"]:
        log_fn(spectral_closure.describe_clean(
            _sc_numerical,
            where="Galerkin psi@centroids sigma numerical rank"))

    # ── Optional physical model-order cut (NOT the numerical rtol) ────────
    # Periodic wavefunctions sampled at different k can be strongly redundant.
    # The exact-span default above nevertheless retains up to nk*nb left-
    # singular directions, which makes every later fH(q) solve cubic in nk*nb.
    # An explicit multiplier requests a shared basis sized by bands PER k.
    # This is deliberately opt-in and separately logged/gated: changing rtol
    # to reach the same integer would falsely call a physical approximation
    # "round-off", obscure its convergence axis, and trip the condition-number
    # service's meaning.
    rank_phys = rank_numerical
    _sc_model = None
    if rank_multiplier > 0.0:
        requested = int(math.ceil(rank_multiplier * nb))
        proposed = min(rank_numerical, requested)
        rank_phys, _sc_model = spectral_closure.resolve_spectral_cut(
            s_host, proposed,
            where="reduced cross-k Galerkin model rank",
            rcond=rtol, log=log_fn)
        if not _sc_model["fired"]:
            log_fn(spectral_closure.describe_clean(
                _sc_model,
                where="reduced cross-k Galerkin model rank"))
        if rank_phys < nb:
            raise ValueError(
                "fit_galerkin_basis: the reduced cross-k Galerkin "
                f"cut retained rank {rank_phys} for {nb} bands per k after "
                "spectral closure.  Per-k row orthonormality is impossible. "
                "Increase rank_multiplier or use 0 for the "
                "exact numerical-rank path.")
        log_fn(
            f"  [reduced-galerkin] EXPLICIT MODEL ORDER: requested "
            f"ceil({rank_multiplier:g}*{nb})={requested}; numerical rank "
            f"{rank_numerical} of {nk*nb}; retaining {rank_phys} shared "
            f"cross-k directions ({rank_phys/nb:.2f} per band at one k, "
            f"{rank_phys/(nk*nb):.1%} of the exact stacked-state span).  "
            "This is an observable-convergence approximation, not rtol.")

    # ── Mesh alignment: PAD, never round the rank down ────────────────────
    # G, its Cholesky factor, ctilde, B and fH all live on a (rank, rank)
    # face sharded P('x','y'), so the carried extent has to divide BOTH mesh
    # axes; an unaligned extent makes the first ``device_put`` onto that face
    # raise "global size of its dimension 0 should be divisible by 4 …".
    #
    # This used to round the retained rank DOWN to a multiple of
    # lcm(px, py) — at P=64 on an 8×8 mesh, up to 7 physically-selected
    # directions discarded FOR A DEVICE-GRID REASON, which makes the answer a
    # function of the machine it ran on.  The rationale ("the dropped ones
    # sit at the rtol threshold, the same decision rtol already makes") does
    # not survive contact with the measured spectrum.
    #
    # MEASURED, job 7883150 (MoS2 4×4, n_μ=785, nval=26/ncond=16, rtol 1e-8):
    # the SVD is (672, 1570), rank 672 = FULL row rank, and the retained block
    # runs σ_max = 9.857614e-01 down to σ_min = 2.22115e-05, i.e.
    #     σ_min/σ_max = 2.2532e-05,
    # which is FOUR AND A HALF DECADES ABOVE the cut at σ_max·rtol.  κ_eff =
    # 4.44e4 against a cap of 1e8.  The directions a round-down removes are
    # therefore not threshold noise at all — they are ordinary members of a
    # smooth spectrum, comfortably inside the retained block.
    #
    # DO NOT re-derive that ratio from an older log line.  Until 2026-07-31
    # this function printed
    #     f"σ_min/{rtol:.0e}={float(s_host[rank-1]):.3e}"
    # whose LABEL says the value is divided by rtol and whose ARGUMENT is the
    # raw σ.  Reading its "σ_min/1e-08=2.221e-05" as σ_min/σ_max ≈ 2e-13
    # understates the true ratio by eight decades, and that misreading was
    # carried into a campaign brief as evidence that this deck sits "exactly
    # at the interpolation capacity limit".  It does not: the capacity limit
    # on this deck is at nb ≈ 98 (nk·nb = 1568 vs nspinor·n_μ = 1570), where
    # the same sweep measures rank 1562 < 1568, ctilde orthogonality 6.31e-04
    # and an on-grid energy error of 0.242 meV.  The rank report below prints
    # both ends of the retained block with correct labels; use it.
    #
    # The fix is to pad the face instead.  The α-basis is EXTENDED by
    # ``n_pad`` exactly-null directions:
    #   coeffs[:, rank_phys:] = 0,  inv_s[rank_phys:] = 0,  UH[rank_phys:] = 0
    # ⇒ Q's pad rows are exactly zero ⇒ G's pad rows/cols are exactly zero.
    # An identity block is placed on the pad diagonal of G so the Cholesky is
    # non-singular; for a block-diagonal SPD matrix ``potrf`` returns exactly
    # blockdiag(chol(G_phys), I) — every off-diagonal update is an exact 0/1
    # division — so ctilde, B and W_proj all acquire exactly-zero pad
    # columns/rows and every downstream contraction over α is bit-unchanged.
    # fH = Σ_n f(ε_n) c_n c_nᴴ likewise gains an exactly-zero block, i.e.
    # ``n_pad`` extra exact-zero eigenvalues on top of the (rank − nb) exact
    # zeros it already carries; band selection is ascending-index and the
    # f-transform makes f(ε) ≤ 0, so the extra zeros sort ABOVE every selected
    # band and no selection moves.
    #
    # ``extra_rank_pad`` adds further null directions on top of the mesh
    # round-up. It is an injected pad-extent-invariance probe: any result that
    # moves under it depends on the carrier rather than the physical span.
    align = math.lcm(int(mesh_xy.shape['x']), int(mesh_xy.shape['y']))
    rank = round_up(rank_phys, align)
    if extra_rank_pad:
        rank = round_up(rank + extra_rank_pad, align)
    n_pad = rank - rank_phys

    # Slice to the criterion's rank and null-extend to the carried extent in
    # ONE jit with an explicit replicated out_sharding: done op-by-op,
    # ``jnp.pad`` on a multi-process mesh would leave these operands with an
    # inferred sharding instead of the ``rep`` the eigh produced.
    @partial(jax.jit, static_argnames=('r_phys', 'n_pad'),
             out_shardings=(rep, rep, rep))
    def _trim_and_pad(U, s, r_phys, n_pad):
        U = U[:, :r_phys]
        s = s[:r_phys]
        coeffs = U * s[None, :]              # (nk·nb, r_phys)
        inv_s = (1.0 / s)[:, None]           # (r_phys, 1)
        UH = U.conj().T                      # (r_phys, nk·nb)
        if n_pad:
            coeffs = jnp.pad(coeffs, ((0, 0), (0, n_pad)))
            inv_s = jnp.pad(inv_s, ((0, n_pad), (0, 0)))
            UH = jnp.pad(UH, ((0, n_pad), (0, 0)))
        return coeffs, inv_s, UH

    # ── Truncation diagnostic — printed on EVERY run ──────────────────────
    # retained rank, σ range of the retained block, σ range discarded, and
    # how much of the discard was the physics criterion (all of it, now) vs
    # the mesh alignment (must be zero).  ``violations()`` is the assertion:
    # it refuses a run whose achieved amplification exceeds the cap, whose
    # retained set depends on the device grid, or whose σ_max is zero/NaN
    # (the documented `nband`-is-an-absolute-index trap, in which the SVD of
    # an all-zero ψ window returns rank 0 and everything downstream is
    # meaningless).
    # ``n_dropped_closure`` is what keeps check 2 meaningful after the
    # closure guard started lowering the rank: the block members it dropped
    # are a deficit against the criterion, and without this attribution
    # ``violations()`` would read them as a device-grid round-down and refuse
    # the run.  They are not — the mesh had no part in choosing them — and
    # anything left over after this subtraction is still a real violation.
    _numerical_report = rank_criterion.rank_report(
        s_host, rtol,
        label=f"Galerkin ψ@centroids numerical rank ({nk*nb}, "
              f"{nspinor*n_mu})",
        quantity="singular values", rank_used=rank_numerical,
        n_rows=nk * nb, n_cols=nspinor * n_mu,
        n_dropped_closure=max(
            0, _sc_numerical["n_keep"] -
            _sc_numerical["n_keep_closed"]),
        # ``rank_used`` here is a SELECTION, so the ceiling applies (the
        # padded report below is a carried EXTENT and declares none).
        rank_ceiling=_rank_ceiling,
        kappa_certified=rank_criterion.KAPPA_CERTIFIED_GRAM)
    _bad = _numerical_report.violations()
    if _bad:
        raise ValueError(
            "fit_galerkin_basis: the ψ-at-centroids numerical rank is "
            "not self-consistent — " + "  ".join(_bad))
    # THE GATE (docs/dev/rank_truncation_policy.md §2).  ``violations()``
    # above asks whether the code did what it was told; this asks whether
    # what it was told is a regime anyone certified.  Both registered
    # catastrophes satisfied the first and failed the second.
    rank_criterion.certify(
        _numerical_report,
        site="Galerkin ψ@centroids Gram-eigh σ",
        mode=str(rank_policy_mode),
        cause=(f"the Galerkin basis is over-complete for this window: "
               f"{nk * nb} stacked states against nspinor·n_μ = "
               f"{nspinor * n_mu} interpolation columns, and the retained "
               f"block runs all the way down to the rtol cut."),
        fix=("reduce the band window, raise n_μ, or declare an explicit "
             "reduced model order with rank_multiplier (>= 1; "
             "ceil(m·nb) shared cross-k directions, N_k-independent by "
             "construction)."),
        log=log_fn)

    if rank_multiplier == 0.0:
        # The numerical report above owns the full left-Gram spectrum, its
        # structural ceiling and any symmetry-closure drop.  This second
        # report describes only the physical block actually selected plus
        # the exact-null carrier pad.  Feeding it the left-Gram null tail
        # would mislabel above-rtol round-off as a device-grid round-down;
        # feeding it the pre-closure ceiling would hide closure-sized pads.
        _trunc = rank_criterion.rank_report(
            s_host[:rank_phys], rtol,
            label=f"Galerkin ψ@centroids Gram-eigh σ ({nk*nb}, "
                  f"{nspinor*n_mu})",
            quantity="singular values", rank_used=rank,
            n_rows=nk * nb, n_cols=nspinor * n_mu)
        _closure_for_report = _sc_numerical
    else:
        # The model-order tail is not a GRID drop and must never be fed to
        # rank_report as one.  Report the retained operator's conditioning;
        # the explicit reduced-galerkin line above owns the physical tail.
        _trunc = rank_criterion.rank_report(
            s_host[:rank_phys], rtol,
            label=f"retained reduced Galerkin block ({rank_phys} "
                  f"of numerical {rank_numerical})",
            quantity="singular values", rank_used=rank,
            n_rows=rank_phys, n_cols=rank_phys)
        _closure_for_report = _sc_model
    log_fn(f"  Gram-eigh σ of ({nk*nb}, {nspinor*n_mu}): rank={rank_phys}"
           + (f" (+{n_pad} null pad → carried extent {rank}, "
              f"mesh-aligned to {align})" if n_pad else "")
           + f" ({time.time()-t1:.2f}s)")
    log_fn(_trunc.describe())
    if _closure_for_report is not None and _closure_for_report["fired"]:
        # The two guards stay orthogonal and the reader is told which took
        # what.  ``rank_report`` now carries the closure deficit in its own
        # column (see ``n_dropped_closure``), so this line names the block
        # rather than re-deriving the arithmetic.
        _closure_drop = max(
            0, _closure_for_report["n_keep"] -
            _closure_for_report["n_keep_closed"])
        log_fn(f"  [rank]   of the directions not carried, "
               f"{_closure_drop} were "
               f"DROPPED BY DEGENERACY CLOSURE — the members of a block of "
               f"{len(_closure_for_report['members'])} that the cut at "
               f"{_closure_for_report['n_keep']} straddled (relative span "
               f"{_closure_for_report['span_rel']:.3e}).  They are "
               f"real directions with sigma > 0, discarded so the retained "
               f"span is a representation of the point group; the other "
               f"legal cut was {_closure_for_report['n_keep_kept']}, and the ruling of "
               f"2026-08-10 takes the lower.  The {n_pad} null pad above is "
               f"mesh alignment and unrelated.")
    _bad = _trunc.violations()
    if _bad:
        raise ValueError(
            "fit_galerkin_basis: the ψ-at-centroids truncation is not "
            "self-consistent — " + "  ".join(_bad))
    if rank_record_fn is not None:
        rank_record_fn({
            "stacked_states": int(nk * nb),
            "site_spin_columns": int(nspinor * n_mu),
            "structural_ceiling": int(_rank_ceiling),
            "numerical_rank": int(rank_numerical),
            "retained_rank": int(rank_phys),
            "carried_rank": int(rank),
            "null_padding": int(n_pad),
            "rank_multiplier": float(rank_multiplier),
            "numerical_report": _numerical_report,
            "compression": rank_criterion.singular_value_compression(
                s_host, rank_phys),
            "numerical_closure": _sc_numerical,
            "model_closure": _sc_model,
        })
    if rank_numerical < nk * nb:
        log_fn(f"  [warn] ψ-at-centroids is NUMERICALLY RANK-DEFICIENT: "
               f"{nk*nb} states vs numerical rank {rank_numerical} "
               f"(structural ceiling min(nk·nb, nspinor·n_μ) = "
               f"{_rank_ceiling}).  The "
               f"Galerkin basis cannot span the band window — fH energy "
               f"recovery will degrade.  Capacity rule: nk·nb < rank(ψ_μ) "
               f"≤ nspinor·n_μ, i.e. nb < {rank_numerical/nk:.2f} here.  (The pad "
               f"directions are exactly null and add NO capacity — the "
               f"capacity bound is on numerical rank, never on the carried "
               f"extent.  The rank quoted here is CLAMPED to the ceiling: a "
               f"Gram route counts null-space round-off as positive "
               f"eigenvalues, which is how this line once printed "
               f"rank=2034 beside nspinor·n_μ=2032.)")

    # Null-extend to the carried extent.  The zeros here are what make every
    # pad row of Q — hence of G, ctilde, B and W_proj — exactly zero.
    coeffs, inv_s, UH = _trim_and_pad(U, s, rank_phys, n_pad)
    del U, s

    # Vᴴ = diag(1/σ) Uᴴ A, formed μ-SHARDED on 'y' against the sharded ψ tile
    # — the replicated SVD Vh this replaces was (rank, ns·N_μ) on every rank.
    # The contraction runs over (k, n) only, so each device builds its own μ
    # columns with no communication; the σ-pad rows (inv_s = 0) and the
    # loader μ-pad columns come out exactly zero.  Kept at the padded μ
    # extent until ``_b_at_mu`` trims to the true centroid count.
    vh_shard = NamedSharding(mesh_xy, P(None, None, 'y'))

    @partial(jax.jit, out_shardings=vh_shard)
    def _vh_sharded(inv_s, UH, psi):
        UH_kb_ = UH.reshape(rank, nk, nb)
        Vh = jnp.einsum('akn,knsm->asm', UH_kb_, psi, optimize=True)
        return inv_s[:, :, None] * Vh        # (rank, ns, μ_pad), μ on 'y'

    Vh_sh = _vh_sharded(inv_s, UH, psi_rmu_Y)
    del psi_rmu_Y

    # ── 3. Projected real-space Gram ──────────────────────────────────────
    # ``isdf.galerkin`` owns the Q kernels, sharding, memory plan, and the
    # load-bearing r-outer / band-inner schedule.  This caller owns only the
    # exact-null identity block that makes its subsequent Cholesky nonsingular.
    # Allocate G sharded directly.  ``jax.device_put(jnp.zeros(...), sharding)``
    # on a multi-process mesh hands JAX an UNCOMMITTED fully-addressable array,
    # which takes ``_device_put_sharding_impl``'s ``multihost_utils.assert_equal``
    # branch — a real ``process_allgather`` of ``P · rank² · 16`` B (178 MB/rank
    # per process at rank 4716, so 11 GB/rank at P=64) to assert that a block of
    # zeros is the same everywhere.  Same fix, same reason, as ``Q`` below.
    #
    # PAD BLOCK.  ``inv_s`` is zero on the pad rows, so Q's pad rows — and
    # therefore Q Qᴴ's pad rows and columns — are EXACTLY zero and G would be
    # singular there.  Seed the pad diagonal with 1 so G is block-diagonal
    # ``[[G_phys, 0], [0, I]]``.  ``potrf`` on that returns exactly
    # ``blockdiag(chol(G_phys), I)``: for j ≥ rank_phys every A[j,k<j] is an
    # exact 0 and L[j,k] = 0/L[k,k] = 0, so L[j,j] = sqrt(1 − 0) = 1.  The
    # physical block is therefore factored bit-identically to the unpadded
    # run, and ctilde/B/W_proj acquire exactly-zero pad columns/rows.
    # ``n_pad == 0`` keeps the historical zeros allocation untouched.
    def _alloc_G():
        Z = jnp.zeros((rank, rank), dtype=jnp.complex128)
        if n_pad:
            d = jnp.concatenate([jnp.zeros(rank - n_pad, dtype=jnp.float64),
                                 jnp.ones(n_pad, dtype=jnp.float64)])
            Z = Z + jnp.diag(d).astype(jnp.complex128)
        return Z

    G = jax.jit(_alloc_G, out_shardings=grid_xy)()

    # Populate one host-resident G-flat source after the centroid-only window
    # has been released.  It reads each band carrier once, then serves every
    # outer-r chunk through the canonical FFT/Bloch/reshard helpers.
    with build_psi_G_store(
            wfn=wfn, mesh_xy=mesh_xy, meta=meta,
            band_chunk_ranges=_band_chunk_ranges, bispinor=bispinor,
            band_pad_to=_bc) as source:
        G = build_streamed_projected_gram(
            source=source,
            meta=meta,
            mesh_xy=mesh_xy,
            UH=UH,
            inv_s=inv_s,
            gram_init=G,
            mu_pad=mu_pad,
            q_tile_budget=stream_budget,
            device_pool_limit=device_pool_limit,
            log_fn=log_fn,
            progress_fn=progress_fn,
        )

    # ── 4. Cholesky on G ──
    # FFI seam: gw_jax's factor_c_q (which has dual 1×1-dense /
    # 2D-blocked paths) is the natural drop-in for distributed scaling, but
    # its 1e-14·trace ridge is calibrated for ISDF's huge-trace C_q matrices
    # and biases this small-trace G by ~1e-5. Use raw Cholesky for
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
    def _finalize(coeffs, L):
        coeffs = coeffs @ L
        ctilde = coeffs.reshape(nk, nb, rank)
        CtC = ctilde[0] @ ctilde[0].conj().T
        ortho_err = jnp.max(jnp.abs(CtC - jnp.eye(nb, dtype=ctilde.dtype)))
        return ctilde, ortho_err

    ctilde, ortho_err = _finalize(coeffs, L)
    ctilde = jax.device_put(ctilde, rep)
    if rank_multiplier > 0.0:
        # A truncated GLOBAL SVD is a least-squares shared span, but its band
        # rows at one k are not exactly orthonormal.  fH's coarse-grid energy
        # identity requires that local invariant.  The polar factor is the
        # closest row-isometry and changes only this explicit approximate
        # route; the historical numerical-rank coefficients never pass here.
        _lowdin = jax.jit(
            _lowdin_orthonormalize_band_rows,
            out_shardings=(rep, rep, rep, rep, rep, rep))
        (ctilde, _lowdin_lmin, _lowdin_lmax, _ortho_before,
         _ortho_after, _lowdin_move) = _lowdin(ctilde)
        _stats = [float(x) for x in jax.device_get(jnp.stack([
            _lowdin_lmin, _lowdin_lmax, _ortho_before, _ortho_after,
            _lowdin_move]))]
        lmin, lmax, ortho_before, ortho_after, lowdin_move = _stats
        if (not np.isfinite(lmin) or not np.isfinite(lmax) or lmax <= 0.0
                or lmin <= 1.0e-12 * lmax):
            raise ValueError(
                "fit_galerkin_basis: the requested reduced cross-k "
                "basis does not span all bands at every k: the smallest/"
                f"largest eigenvalue of C_k C_k^H is {lmin:.6e}/"
                f"{lmax:.6e}.  The per-k Löwdin map would amplify by more "
                "than 1e6. Increase rank_multiplier (or use 0 "
                "for the exact numerical-rank path).")
        log_fn(
            f"  [reduced-galerkin] per-k Löwdin row isometry: "
            f"eig(C C^H) min/max={lmin:.6e}/{lmax:.6e}, "
            f"max|C C^H-I| {ortho_before:.3e} -> {ortho_after:.3e}, "
            f"max relative coefficient move={lowdin_move:.3e}.  The move is "
            "the declared cross-k compression error; final acceptance comes "
            "from the on-grid wavefunction and exciton-spectrum A/B gates.")
        ortho_err = jnp.asarray(ortho_after)

    # B = L⁻¹ Vᴴ, μ-SHARDED end-to-end (replaces the replicated B_at_mu).
    # The triangular solve runs along the rank axis and is independent per μ
    # column, so it stays local on each μ shard; the spinor axis is moved in
    # front as the (broadcast) batch axis so no reshape ever merges the
    # replicated ns axis with the sharded μ axis.  The trailing slice trims
    # the loader μ-pad to the TRUE centroid count; n_μ divides the mesh axis
    # only by luck, so the output sharding is FITTED (announced replication
    # fallback, never a refusal).
    b_shard = _fit(mesh_xy, P(None, None, 'y'), (rank, nspinor, n_mu),
                   "galerkin.basis_at_nodes(mu-axis)")

    @partial(jax.jit, out_shardings=b_shard)
    def _b_at_mu(L, Vh):
        Vt = jnp.moveaxis(Vh, 1, 0)          # (ns, rank, μ_pad)
        B = jax.vmap(lambda rhs: jsp_linalg.solve_triangular(
            L, rhs, lower=True))(Vt)
        return jnp.moveaxis(B, 0, 1)[..., :n_mu]

    B_at_mu = _b_at_mu(L, Vh_sh)
    del Vh_sh
    log_fn(f"  ctilde[0] orthogonality error: {float(ortho_err):.3e}")
    log_fn(f"  Total Galerkin: {time.time()-t0:.2f}s")

    projector = None
    if include_projector:
        # W_proj = L⁻¹ diag(1/s) U^H — the α-basis-on-full-r projector (see
        # docstring).  (rank, nk·nb) replicated; small (≲ tens of MB).
        projector = jsp_linalg.solve_triangular(L, inv_s * UH, lower=True)
        projector = jax.device_put(projector, rep)
    return GalerkinBasis(
        ctilde=ctilde,
        basis_at_nodes=B_at_mu,
        projector=projector,
        rank_physical=rank_phys,
        band_range=(int(b_start), int(b_end)),
    )




_accum_G_cache: dict = {}
_fold_G_cache: dict = {}


def _make_accum_kernel(rank_, bc_size, nspinor_, mesh_, rep_,
                       psi_layout_, sharding_q_):
    """One compiled Galerkin Q-accumulation kernel per static config.

    THE psi INPUT SHARDING IS PINNED to Q's r layout (in_shardings); the
    iterator's canonical staged reshard puts it there before this kernel.
    The historical kernel reshaped (nk, bc, ns, r_'y') to
    (nk·bc, ns·r) — merging the replicated spinor axis with the sharded r
    axis, which no NamedSharding can express — and then asked for
    ('x','y')-sharded output columns.  The legacy SPMD partitioner cannot
    synthesize that band-to-r exchange: at P16 it fell back to fully
    replicating the 54.3-GiB c128[81,1,2,1406256] slab per 40-GB GPU and
    OOMed (Perlmutter JID 57271407, "Involuntary full rematerialization",
    125.60 GiB live).  With psi already r-sharded in Q's own layout and
    the spinor axis kept SEPARATE, the contraction is purely
    device-local: UH_bc is replicated, psi's contracted (nk·bc) rows are
    all present locally, and each device writes only its own r block of
    Q.  Module-level so the P>1 transition twin
    (``tests/test_htransform_q_accum.py``) drives the production kernel,
    not a re-implementation.
    """
    key = (id(mesh_), rank_, bc_size, nspinor_)
    fn = _accum_G_cache.get(key)
    if fn is not None:
        return fn

    # Only the accumulator can alias the result.  ``psi_bc`` is a streamed
    # read-only source and cannot back an output with Q's different shape;
    # advertising it as donated merely asks XLA for an impossible alias and
    # emits one warning per band chunk.
    @partial(jax.jit, donate_argnums=(3,),
             in_shardings=(rep_, rep_, psi_layout_, sharding_q_),
             out_shardings=sharding_q_)
    def _accum(UH_bc, inv_s, psi_bc, Q_in):
        # UH_bc: (rank, nk·bc) replicated; inv_s: (rank,1) replicated
        # psi_bc: (nk, bc, ns, r) sharded P(None,None,None,r_entry)
        nkv, bcv, nsv, rcv = psi_bc.shape
        # Merge only the two REPLICATED leading axes; the sharded r
        # axis and the spinor axis are never combined.
        psi_flat = psi_bc.reshape(nkv * bcv, nsv, rcv)
        Q = inv_s[:, :, None] * jnp.einsum(
            'ak,ksr->asr', UH_bc, psi_flat, optimize=True)
        return Q_in + Q

    _accum_G_cache[key] = _accum
    return _accum


def _make_fold_G_kernel(rank_, mesh_, sharding_q_, grid_xy_):
    """Add one already-bounded, r-sharded ``Q_chunk Q_chunk†`` to G.

    The caller owns the zeta-style outer-r loop and therefore never hands this
    executable Q over all ``r_tot``.  Each device forms the Gram contribution
    from its unique local-r shard; the established two-stage
    ``psum_scatter`` sums those shards while distributing matrix rows and
    columns onto ``P('x','y')``.
    """
    key = (id(mesh_), int(rank_), tuple(sharding_q_.spec),
           tuple(grid_xy_.spec))
    fn = _fold_G_cache.get(key)
    if fn is not None:
        return fn

    p_x = int(mesh_.shape['x'])
    p_y = int(mesh_.shape['y'])
    if rank_ % p_x or rank_ % p_y:
        raise ValueError(
            "_make_fold_G_kernel: the carried Galerkin rank must divide "
            f"both mesh axes; rank={rank_}, mesh={p_x}x{p_y}")

    @partial(
        shard_map,
        mesh=mesh_,
        in_specs=(sharding_q_.spec, P('x', 'y')),
        out_specs=P('x', 'y'),
        check_vma=False,
    )
    def _fold_local(Q_local, G_local):
        partial = jnp.einsum(
            'asr,bsr->ab', Q_local, jnp.conj(Q_local),
            optimize=True)
        partial = jax.lax.psum_scatter(
            partial, 'x', scatter_dimension=0, tiled=True)
        partial = jax.lax.psum_scatter(
            partial, 'y', scatter_dimension=1, tiled=True)
        return G_local + partial

    fn = jax.jit(
        _fold_local,
        donate_argnums=(1,),
        in_shardings=(sharding_q_, grid_xy_),
        out_shardings=grid_xy_,
    )
    _fold_G_cache[key] = fn
    return fn


@dataclass(frozen=True)
class GalerkinStreamPlan:
    """The mesh-aligned outer-r schedule selected after rank is known."""

    r_chunk_ranges: tuple[tuple[int, int], ...]
    max_r_logical: int
    max_r_carrier: int
    q_tile_local_bytes: int


def plan_galerkin_stream(*, rank: int, nspinor: int, n_rtot: int,
                         r_mesh_divisor: int,
                         q_tile_budget: int) -> GalerkinStreamPlan:
    """Choose the incumbent Q-budget-bounded, mesh-aligned r schedule."""
    q_bytes_per_local_r = (
        rank * nspinor * np.dtype(np.complex128).itemsize)
    if q_bytes_per_local_r > q_tile_budget:
        raise ValueError(
            "plan_galerkin_stream: one local r column of Q needs "
            f"{q_bytes_per_local_r / 1024**3:.6f} GiB/device, exceeding "
            f"q_tile_budget={q_tile_budget / 1024**3:.6f} GiB/device. "
            "Increase that budget or reduce the retained rank.")
    r_local_cap = q_tile_budget // q_bytes_per_local_r
    r_chunk = min(n_rtot, r_local_cap * r_mesh_divisor)
    if r_chunk < n_rtot:
        r_chunk = max(
            r_mesh_divisor,
            (r_chunk // r_mesh_divisor) * r_mesh_divisor,
        )
    r_chunk_ranges = tuple(
        (r0, min(r0 + r_chunk, n_rtot))
        for r0 in range(0, n_rtot, r_chunk)
    )
    max_r_logical = max(r1 - r0 for r0, r1 in r_chunk_ranges)
    max_r_carrier = round_up(max_r_logical, r_mesh_divisor)
    q_tile_local_bytes = (
        rank * nspinor * (max_r_carrier // r_mesh_divisor)
        * np.dtype(np.complex128).itemsize
    )
    return GalerkinStreamPlan(
        r_chunk_ranges=r_chunk_ranges,
        max_r_logical=max_r_logical,
        max_r_carrier=max_r_carrier,
        q_tile_local_bytes=q_tile_local_bytes,
    )


def galerkin_q_ledger(*, rank: int, nk: int, nspinor: int, n_rtot: int,
                      band_chunk: int, m_states: int, mu_pad: int,
                      psi_win_elems: int, p_total: int, q_shards: int,
                      y_shards: int) -> dict:
    """Per-device byte ledger for one streamed-Galerkin r chunk.

    ``n_rtot`` is the chunk carrier, never the full FFT grid.  The Q charge
    includes both the donated accumulation overlap and one conservative
    Q-sized conjugate/layout workspace for the subsequent Gram fold.  Thus
    the compiler may choose that workspace without violating the ledger, but
    no full-``r_tot`` Q exists to copy.

    Keys are printable labels; ``TOTAL`` is their sum.
    """
    C16 = 16.0
    led = {
        # The bounded Q accumulator, donated input/output overlap, and one
        # Q-sized fold workspace:
        # ``_accum`` donates Q_in and writes Q_in + delta into it, but the
        # GEMM's delta-Q result buffer coexists with the accumulator, so
        # the loop's floor is 2x Q per device.
        "Q r-chunk (x2 accumulation + x1 fold workspace)":
            3.0 * rank * nspinor * n_rtot * C16 / q_shards,
        # One streamed psi band chunk in Q's r layout (the _accum input).
        "psi chunk (r-layout)":
            1.0 * nk * band_chunk * nspinor * n_rtot * C16 / q_shards,
        # The source shard remains live during the two staged all-to-alls.
        # It is band-sharded over the same full mesh product, so source and
        # destination are equal-volume shards; no y-replicated carrier exists.
        "psi chunk (band-layout, transition overlap)":
            1.0 * nk * band_chunk * nspinor * n_rtot * C16 / q_shards,
        # Persistent Gram state, replicated on every device: coeffs
        # (m_states, rank), UH (rank, m_states), UH_kb (its eager-reshape
        # copy), the per-chunk UH_bc block and inv_s.
        "Gram state (replicated)":
            (3.0 * m_states * rank + 1.0 * rank * nk * band_chunk
             + 1.0 * rank) * C16,
        # Vh, mu-sharded on 'y', live from step 2 until B_at_mu.
        "Vh (mu on 'y')":
            1.0 * rank * nspinor * mu_pad * C16 / y_shards,
        # ``_fold_local`` forms the full rank×rank partial on every device
        # before the two psum_scatter operations distribute its rows/columns.
        "fold partial (replicated)": 1.0 * rank * rank * C16,
        # The G face, P('x','y').
        "G face": 1.0 * rank * rank * C16 / p_total,
        # The resident psi(G-flat) window, band-sharded over all P.
        "psi(G-flat) window": 1.0 * psi_win_elems * C16 / p_total,
    }
    led["TOTAL"] = sum(led.values())
    return led


def _refuse_unfit_galerkin_mesh(
        ledger: dict, *, rank: int, nk: int, nspinor: int, n_rtot: int,
        band_chunk: int, m_states: int, mu_pad: int, psi_win_elems: int,
        mesh_xy: Mesh, q_spec, device_pool_limit: float | None,
        log_fn) -> None:
    """Refuse a non-fitting live set before any Q compilation.

    ``device_pool_limit`` is resolved by the caller through the owned GPU
    memory reader.  Passing it explicitly keeps machine policy out of this
    numerical owner while preserving the incumbent gate and diagnostics.
    """
    limit = device_pool_limit
    if limit is None or limit <= 0:
        log_fn("  [galerkin-mem] device pool size unreadable (CPU backend "
               "or platform allocator) — ledger printed above, the "
               "non-fitting-mesh refusal DID NOT RUN")
        return
    total = float(ledger["TOTAL"])
    if total <= float(limit):
        return
    fitting = None
    # Search ALL supported square meshes: a larger current mesh can use more
    # memory when another live object replicates, so the truthful remedy is
    # the minimum-P fitting geometry, not merely the first larger one.
    for s in range(1, 65):
        n_r_carrier_s = round_up(n_rtot, s * s)
        led_s = galerkin_q_ledger(
            rank=rank, nk=nk, nspinor=nspinor, n_rtot=n_r_carrier_s,
            band_chunk=band_chunk, m_states=m_states, mu_pad=mu_pad,
            psi_win_elems=psi_win_elems, p_total=s * s,
            q_shards=s * s, y_shards=s)
        if led_s["TOTAL"] <= float(limit):
            fitting = (s, float(led_s["TOTAL"]))
            break
    detail = "; ".join(
        f"{name} {b / 1024**3:.2f} GiB" for name, b in ledger.items()
        if name != "TOTAL")
    if fitting is not None:
        s, tot_s = fitting
        remedy = (f"the smallest square mesh that fits at this pool size is "
                  f"{s}x{s} (P={s * s}: projected "
                  f"{tot_s / 1024**3:.2f} GiB/device)")
    else:
        remedy = ("no square mesh up to 64x64 fits this r chunk; lower the "
                  "caller's q_tile_budget or narrow the model rank")
    raise ValueError(
        f"build_streamed_projected_gram: the Galerkin live set does not fit "
        f"this mesh.  Projected {total / 1024**3:.2f} GiB/device against a "
        f"{limit / 1024**3:.2f} GiB pool on the "
        f"{mesh_xy.shape['x']}x{mesh_xy.shape['y']} mesh "
        f"(Q sharded {q_spec}, {spec_divisor(mesh_xy, q_spec, axis=2)}-way on "
        f"r).  Ledger: {detail}.  {remedy}.  This is refused BEFORE "
        f"compilation. Lower q_tile_budget to use more, smaller r chunks "
        f"without changing the fit.")


def build_streamed_projected_gram(
        *, source, meta, mesh_xy: Mesh,
        UH, inv_s, gram_init, mu_pad: int,
        q_tile_budget: int, device_pool_limit: float | None,
        log_fn=None, progress_fn=None):
    """Build ``G = Q Q^H`` with r outermost and contracted bands innermost.

    ``source`` is the caller-owned :class:`common.psi_G_store.PsiGStore`.
    It owns the one coefficient load, the logical band schedule and the
    canonical G-to-r/reshard route.  ``q_tile_budget`` and
    ``device_pool_limit`` are caller-resolved policy.  ``gram_init`` carries
    the caller's exact-null padding block and must be sharded on
    ``P('x', 'y')``.  ``progress_fn`` is an optional presentation callback;
    it does not participate in the numerical schedule.
    """
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    band_chunk_ranges = tuple(
        (int(lo), int(hi)) for lo, hi in source.band_chunk_ranges)
    if not band_chunk_ranges:
        raise ValueError(
            "build_streamed_projected_gram: source has no band chunks")
    b_start = band_chunk_ranges[0][0]
    b_end = band_chunk_ranges[-1][1]
    if any(band_chunk_ranges[i][1] != band_chunk_ranges[i + 1][0]
           for i in range(len(band_chunk_ranges) - 1)):
        raise ValueError(
            "build_streamed_projected_gram: source band chunks must form "
            f"one contiguous interval; got {band_chunk_ranges!r}")
    nb = b_end - b_start
    nk = int(meta.nk_tot)
    nspinor = int(meta.nspinor)
    n_rtot = int(meta.n_rtot)
    rank = int(UH.shape[0])
    m_states = nk * nb
    _bc = int(source.band_chunk_carrier)

    rep = NamedSharding(mesh_xy, P())
    grid_xy = NamedSharding(mesh_xy, P('x', 'y'))
    q_spec = P(None, None, ('y', 'x'))
    r_mesh_divisor = spec_divisor(mesh_xy, q_spec, axis=2)

    # Q[α, x] = Σ_{k,n} inv_s[α] U^H[α,(k,n)] ψ[(k,n), x].  The contraction
    # runs over the pair index (k, n) and is free in x = (spinor, r).  Thus
    # splitting r and summing G over the pieces is exact, while forming one
    # Gram per band chunk drops every cross-band term of
    # ``(Σ_bc Q_bc)(Σ_bc' Q_bc')^H``.  That wrong route stays Hermitian and
    # lets Cholesky succeed; its measured symptom on MoS2 12x12 / n_mu=640 /
    # nb=40 was a 1742.48-meV on-grid energy drift, against 0.63 meV for the
    # correct single-Q fold.  Bands are therefore innermost and every band
    # chunk is summed into Q before the fold below.
    #
    # The source's already-aligned carrier is the single band schedule.  A
    # narrower terminal logical range still arrives in the same exact-zero
    # carrier, and matching zero columns in UH below make that padding inert.

    # r is the free Q column.  Runtime padding adds exact-zero terminal
    # columns and product-shards them over the full mesh.
    sharding_q = NamedSharding(mesh_xy, q_spec)
    _q_r_entry = sharding_q.spec[2]
    q_shards = r_mesh_divisor
    psi_r_layout = NamedSharding(
        mesh_xy, P(None, None, None, _q_r_entry))

    plan = plan_galerkin_stream(
        rank=rank, nspinor=nspinor, n_rtot=n_rtot,
        r_mesh_divisor=r_mesh_divisor, q_tile_budget=q_tile_budget)
    log_fn(
        f"  Galerkin r plan: {len(plan.r_chunk_ranges)} chunk(s), "
        f"logical width <= {plan.max_r_logical}, "
        f"carrier <= {plan.max_r_carrier}; "
        f"Q <= {plan.q_tile_local_bytes / 1024**3:.2f} GiB/device "
        f"(budget {q_tile_budget / 1024**3:.2f} GiB/device)")

    _p_y_shards = int(mesh_xy.shape['y'])
    ledger = galerkin_q_ledger(
        rank=rank, nk=nk, nspinor=nspinor,
        n_rtot=plan.max_r_carrier, band_chunk=_bc,
        m_states=m_states, mu_pad=mu_pad, psi_win_elems=0,
        p_total=int(mesh_xy.size), q_shards=q_shards,
        y_shards=_p_y_shards)
    log_fn(f"  Galerkin Q ledger (per device): Q sharded "
           f"{sharding_q.spec} ({q_shards}-way on r), "
           + ", ".join(f"{name} {b / 1024**3:.2f} GiB"
                       for name, b in ledger.items()))
    _refuse_unfit_galerkin_mesh(
        ledger, rank=rank, nk=nk, nspinor=nspinor,
        n_rtot=plan.max_r_logical, band_chunk=_bc,
        m_states=m_states, mu_pad=mu_pad, psi_win_elems=0,
        mesh_xy=mesh_xy, q_spec=sharding_q.spec,
        device_pool_limit=device_pool_limit, log_fn=log_fn)

    fold_G = _make_fold_G_kernel(rank, mesh_xy, sharding_q, grid_xy)
    G = gram_init
    t0 = time.time()
    band_eval_count = 0
    UH_kb = UH.reshape(rank, nk, nb)
    from common.progress import LoopProgress
    progress_steps = len(plan.r_chunk_ranges) * len(band_chunk_ranges)
    progress = LoopProgress(
        progress_steps, progress_fn or (lambda *_: None),
        title="Galerkin real-space accumulation",
        item_name="(r, band) chunk",
        max_updates=min(progress_steps, 12),
        enabled=progress_fn is not None).start()

    @partial(jax.jit, static_argnums=(0,), out_shardings=sharding_q)
    def _alloc_Q(r_extent):
        return jnp.zeros(
            (rank, nspinor, r_extent), dtype=jnp.complex128)

    for r_idx, (r0, r1) in enumerate(plan.r_chunk_ranges):
        r_carrier = round_up(r1 - r0, r_mesh_divisor)
        Q = _alloc_Q(r_carrier)
        for bc_range, psi_bc_r in source.iter_rchunk_bandwise(
                r0, r1, product_r_spec=psi_r_layout.spec):
            if int(psi_bc_r.shape[-1]) != r_carrier:
                raise ValueError(
                    "Galerkin iterator r carrier disagrees with Q chunk: "
                    f"psi={int(psi_bc_r.shape[-1])}, Q={r_carrier}, "
                    f"r=[{r0},{r1})")
            bc = bc_range[1] - bc_range[0]
            bc_lo = bc_range[0] - b_start
            bc_hi = bc_range[1] - b_start
            w = int(psi_bc_r.shape[1])
            UH_bc_kb = UH_kb[:, :, bc_lo:bc_hi]
            if w > bc:
                UH_bc_kb = jnp.pad(
                    UH_bc_kb, ((0, 0), (0, 0), (0, w - bc)))
            UH_bc = UH_bc_kb.reshape(rank, nk * w)

            accum = _make_accum_kernel(
                rank, w, nspinor, mesh_xy,
                rep, psi_r_layout, sharding_q)
            Q = accum(UH_bc, inv_s, psi_bc_r, Q)
            del psi_bc_r
            band_eval_count += 1
            progress.step()

        G = fold_G(Q, G)
        jax.block_until_ready(G)
        del Q
        log_fn(
            f"  G r-chunk {r_idx + 1}/{len(plan.r_chunk_ranges)}: "
            f"[{r0},{r1}) -> carrier {r_carrier}")

    progress.finish()
    log_fn(
        f"  G accumulation: {len(plan.r_chunk_ranges)} r chunk(s) x "
        f"{len(band_chunk_ranges)} band chunk(s) "
        f"({band_eval_count} streamed evaluations), {time.time()-t0:.2f}s")
    return G
