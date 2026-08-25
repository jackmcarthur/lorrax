"""Whole-state Hamiltonian-transform basis on a two-dimensional mesh.

This module owns the randomized-QRCP-equivalent selection, exact global
basis factorization, and physical wavefunction projection used by
htransform.  Every real-space stage streams bounded full-Bloch slabs from the
canonical WFN source; no full-grid basis or random matrix is materialized.

Wavefunction loading and G-to-r transforms remain owned by the reusable
``common.psi_G_store`` source and its canonical transform helpers.  The caller
resolves policy such as environment overrides and the device-pool limit, then
passes explicit values here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg as jsp_linalg
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.psi_G_store import build_psi_G_store
from common.pivoted_cholesky import make_sharded_pivoted_cholesky_select
from common.shard_map import shard_map
from common.sharding_fit import fit_sharding as _fit
from common.wfn_layout import band_sphere_spec
from common.wfn_transforms import load_centroids_band_chunked
from distrib_la import plan as linalg_plan
from runtime.padding import round_up, spec_divisor


__all__ = [
    "GalerkinBasis",
    "GalerkinStreamPlan",
    "fit_galerkin_basis",
    "plan_galerkin_stream",
    "validate_rank_multiplier",
]


_QRCP_RNG_VERSION = "jax-fold-in-global-r-rows-spin-v1"


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
    selected_state_indices: tuple[int, ...] = ()
    selection_factor: jax.Array | None = None
    qrcp_seed: int = 0
    qrcp_rng_version: str = _QRCP_RNG_VERSION
    qrcp_eps: float = 1.0e-3
    qrcp_raw_rank: int = 0
    qrcp_search_rank: int = 0
    candidate_hash: str = ""
    pivot_hash: str = ""

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
        if self.selected_state_indices:
            if len(self.selected_state_indices) != int(self.rank_physical):
                raise ValueError(
                    "GalerkinBasis selected-state count must equal the "
                    f"physical rank; got {len(self.selected_state_indices)} "
                    f"and {self.rank_physical}")
            if self.selection_factor is None:
                raise ValueError(
                    "GalerkinBasis selected states require their shared "
                    "triangular factor")
        if self.selection_factor is not None:
            expected = (rank, rank)
            if tuple(self.selection_factor.shape) != expected:
                raise ValueError(
                    "GalerkinBasis.selection_factor must have shape "
                    f"{expected}; got {tuple(self.selection_factor.shape)}")

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
    """Validate the whole-state QRCP search ceiling multiplier.

    ``0`` is retained as an input-compatibility spelling of the published
    default ``20``; there is no exact-span alternate route.
    """
    try:
        multiplier = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name}={value!r} is not a finite number; use 20 for the "
            "published search or another value >= 1.") from None
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError(f"{name}={value!r} must be finite and >= 0.")
    if multiplier == 0.0:
        return 20.0
    if multiplier < 1.0:
        raise ValueError(
            f"{name}={multiplier:g} would retain fewer directions than bands "
            "at one k. Use the published default 20 or another value >= 1.")
    return multiplier


def _whole_state_memory_ledger(
        *, meta, mesh_xy: Mesh, nk: int, nspinor: int,
        band_carrier: int, state_count: int, search_rank: int,
        candidate_carrier: int, q_tile_budget: int) -> dict[str, float]:
    """Zeta-style stage-maximum ledger for the whole-state fit.

    The spatial source term is the compiled production inverse FFT including
    cuFFT plan workspace.  Other terms are explicit live arrays.  Stages are
    alternatives; the returned ``HWM`` is their maximum, never their sum.
    """
    from common.fft_helpers import query_fft_peak_bytes

    p = int(mesh_xy.size)
    rank_carrier = round_up(
        int(search_rank),
        math.lcm(int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])))
    r_spec = P(None, None, ('y', 'x'))
    r_divisor = spec_divisor(mesh_xy, r_spec, axis=2)
    stream = plan_galerkin_stream(
        rank=rank_carrier, nspinor=nspinor,
        n_rtot=int(meta.n_rtot), r_mesh_divisor=r_divisor,
        q_tile_budget=int(q_tile_budget))
    r_carrier = int(stream.max_r_carrier)
    fft_peak = float(query_fft_peak_bytes(
        input_shape=(int(nk), int(band_carrier), int(nspinor),
                     *(int(v) for v in meta.fft_grid)),
        fft_axes=(-3, -2, -1),
        sharding=NamedSharding(
            mesh_xy,
            P(None, ('x', 'y'), None, None, None, None)),
        kind="ifftn", norm="ortho", dtype=jnp.complex128))

    c16 = float(np.dtype(np.complex128).itemsize)
    f8 = float(np.dtype(np.float64).itemsize)
    g_index = float(nk * int(meta.n_rtot) * np.dtype(np.int32).itemsize)
    psi_r = float(nk * band_carrier * nspinor
                  * (r_carrier // r_divisor) * c16)
    selected_rows = float(rank_carrier * nspinor
                          * (r_carrier // r_divisor) * c16)
    random_rows = float(search_rank * nspinor
                        * (r_carrier // r_divisor) * f8)
    sketch = float(search_rank * candidate_carrier * c16)
    candidate_face = float(candidate_carrier * candidate_carrier * c16 / p)
    candidate_pc = float(candidate_carrier * search_rank * c16 / p)
    selected_face = float(rank_carrier * rank_carrier * c16 / p)
    selected_fold = float(rank_carrier * rank_carrier * c16)
    factor = float(rank_carrier * rank_carrier * c16)
    coefficients = float(state_count * rank_carrier * c16)

    stages = {
        "sketch_stream": (
            g_index + sketch + random_rows
            + max(fft_peak, psi_r + search_rank * band_carrier * c16)),
        "sketch_select": (
            g_index + sketch + candidate_face + candidate_pc),
        "selected_gram_stream": (
            g_index + selected_face + selected_rows
            + max(fft_peak, psi_r)),
        "selected_gram_fold": (
            g_index + selected_face + selected_rows + selected_fold),
        "physical_projection": (
            g_index + factor + coefficients + 2.0 * selected_rows
            + max(fft_peak, psi_r)),
    }
    stages["FFT_STANDALONE"] = fft_peak
    stages["r_chunk_carrier"] = float(r_carrier)
    stages["Q_TILE_LOCAL"] = selected_rows
    stages["HWM"] = max(
        value for name, value in stages.items()
        if name not in ("r_chunk_carrier", "Q_TILE_LOCAL"))
    return stages


def _resolve_whole_state_stream_budget(
        *, meta, mesh_xy: Mesh, nk: int, nspinor: int,
        band_carrier: int, state_count: int, search_rank: int,
        candidate_carrier: int, requested_q_tile_budget: int,
        device_pool_limit: float | None, log_fn):
    """Choose the largest measured-workspace-safe real-space carrier."""
    if requested_q_tile_budget <= 0:
        raise ValueError(
            "whole-state Galerkin stream budget must be positive")
    bytes_per_local_r = (
        round_up(search_rank, math.lcm(
            int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])))
        * int(nspinor) * np.dtype(np.complex128).itemsize)
    max_local_r = max(1, requested_q_tile_budget // bytes_per_local_r)
    tried = set()
    local_r = max_local_r
    chosen = None
    while local_r >= 1:
        budget = int(local_r * bytes_per_local_r)
        if budget not in tried:
            tried.add(budget)
            ledger = _whole_state_memory_ledger(
                meta=meta, mesh_xy=mesh_xy, nk=nk,
                nspinor=nspinor, band_carrier=band_carrier,
                state_count=state_count, search_rank=search_rank,
                candidate_carrier=candidate_carrier,
                q_tile_budget=budget)
            if (device_pool_limit is None or device_pool_limit <= 0
                    or ledger["HWM"] <= float(device_pool_limit)):
                chosen = (budget, ledger)
                break
        local_r //= 2
    if chosen is None:
        minimum = _whole_state_memory_ledger(
            meta=meta, mesh_xy=mesh_xy, nk=nk,
            nspinor=nspinor, band_carrier=band_carrier,
            state_count=state_count, search_rank=search_rank,
            candidate_carrier=candidate_carrier,
            q_tile_budget=bytes_per_local_r)
        raise MemoryError(
            "fit_galerkin_basis: the measured whole-state live set does "
            "not fit even at one local real-space column: projected HWM "
            f"{minimum['HWM']/2**30:.2f} GiB/device against pool "
            f"{float(device_pool_limit or 0)/2**30:.2f} GiB/device. The "
            "compiled production ifftn(norm='ortho') is priced at "
            f"{minimum['FFT_STANDALONE']/2**30:.2f} GiB/device.")
    budget, ledger = chosen
    log_fn(
        "  Whole-state memory plan (stage maxima, per device): "
        + ", ".join(
            f"{name}={value/2**30:.2f} GiB"
            for name, value in ledger.items()
            if name not in ("r_chunk_carrier",))
        + f", r_chunk_carrier={int(ledger['r_chunk_carrier'])}"
        + (f", pool={float(device_pool_limit)/2**30:.2f} GiB"
           if device_pool_limit is not None and device_pool_limit > 0
           else ", pool=unavailable (HWM refusal not run)"))
    if budget < int(requested_q_tile_budget):
        log_fn(
            f"  Whole-state planner reduced the Q budget from "
            f"{requested_q_tile_budget/2**30:.2f} to {budget/2**30:.2f} "
            "GiB/device so the compiled IFFT workspace and persistent fit "
            "state do not overlap past the device pool")
    return budget, ledger


def fit_galerkin_basis(
        wfn, sym, meta, centroid_indices, mesh_xy: Mesh,
        band_range: tuple[int, int], *,
        log_fn=None,
        band_chunk_size: int = 64,
        bispinor: bool = False,
        include_projector: bool = False,
        rank_multiplier: float = 20.0,
        qr_eps: float = 1.0e-3,
        qrcp_seed: int = 0,
        q_tile_budget: int,
        device_pool_limit: float | None,
        extra_rank_pad: int = 0,
        progress_fn=None,
        rank_record_fn=None,
) -> GalerkinBasis:
    """Fit the published whole-state Hamiltonian-transform basis.

    The stacked full-Bloch states ``Psi[(k,n),(s,r)]`` are the only basis
    source.  A deterministic Gaussian sketch and pivoted Cholesky of its
    candidate Gram reproduce randomized QRCP's column selection; the chosen
    *physical* states ``X`` then define one global orthonormal basis

    ``X X^H = L L^H,  B = L^-1 X,  C = Psi B^H``.

    ``B`` is never materialized over the full FFT grid.  The canonical
    :class:`common.psi_G_store.PsiGStore` supplies bounded real-space slabs;
    those slabs build the sketch, the selected-state Gram, and the physical
    projections.  Centroids are used only to evaluate ``B(r_mu)`` after the
    global basis has been selected.  No centroid weighting, state-space SVD,
    or per-k gauge repair participates in basis construction.

    ``qr_eps`` is the sole rank-revealing tolerance.
    """
    del sym
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    b_start, b_end = (int(v) for v in band_range)
    nb = b_end - b_start
    nk = int(meta.nk_tot)
    nspinor = int(meta.nspinor)
    n_rtot = int(meta.n_rtot)
    n_mu = int(np.asarray(centroid_indices).shape[0])
    if nb <= 0:
        raise ValueError(
            f"fit_galerkin_basis: empty band range [{b_start},{b_end})")
    if not (0.0 < float(qr_eps) < 1.0):
        raise ValueError(
            f"fit_galerkin_basis: qr_eps={qr_eps!r} must lie in (0,1)")
    try:
        qrcp_seed = int(qrcp_seed)
    except (TypeError, ValueError):
        raise ValueError(
            f"fit_galerkin_basis: qrcp_seed={qrcp_seed!r} is not an integer") \
            from None
    if not 0 <= qrcp_seed <= np.iinfo(np.uint32).max:
        raise ValueError(
            f"fit_galerkin_basis: qrcp_seed={qrcp_seed} must fit uint32")
    if n_rtot > np.iinfo(np.uint32).max:
        raise ValueError(
            "fit_galerkin_basis: the stateless QRCP sketch indexes global "
            f"r in uint32, but n_rtot={n_rtot} exceeds that range")
    search_multiplier = validate_rank_multiplier(
        rank_multiplier, name="htransform_rank_multiplier")
    extra_rank_pad = int(extra_rank_pad)
    if extra_rank_pad < 0:
        raise ValueError(
            f"fit_galerkin_basis: extra_rank_pad={extra_rank_pad} must be >=0")
    if int(q_tile_budget) <= 0:
        raise ValueError(
            f"fit_galerkin_basis: q_tile_budget={q_tile_budget} must be >0")

    m_states = nk * nb
    state_dim = nspinor * n_rtot
    max_search = min(
        int(math.ceil(search_multiplier * nb)), state_dim, m_states)
    n_candidates = min(int(1.5 * max_search), m_states)
    if max_search < nb:
        raise ValueError(
            "fit_galerkin_basis: the QRCP search ceiling carries fewer "
            f"directions ({max_search}) than bands at one k ({nb})")

    rng = np.random.default_rng(qrcp_seed)
    candidates = np.asarray(
        rng.permutation(m_states)[:n_candidates], dtype=np.int64)
    candidate_hash = hashlib.sha256(
        candidates.astype("<i8", copy=False).tobytes()).hexdigest()

    p_total = int(mesh_xy.size)
    candidate_carrier = round_up(n_candidates, p_total)
    align = math.lcm(
        int(mesh_xy.shape['x']), int(mesh_xy.shape['y']))
    p_band = spec_divisor(mesh_xy, band_sphere_spec(), axis=1)
    bc_hint = max(p_band, min(int(band_chunk_size), nb))
    bc_carrier = round_up(bc_hint, p_band)
    while True:
        try:
            q_tile_budget, _memory_ledger = \
                _resolve_whole_state_stream_budget(
                    meta=meta, mesh_xy=mesh_xy, nk=nk,
                    nspinor=nspinor, band_carrier=bc_carrier,
                    state_count=m_states, search_rank=max_search,
                    candidate_carrier=candidate_carrier,
                    requested_q_tile_budget=int(q_tile_budget),
                    device_pool_limit=device_pool_limit, log_fn=log_fn)
            break
        except MemoryError as exc:
            if bc_carrier <= p_band:
                raise
            next_carrier = max(
                p_band, (bc_carrier // (2 * p_band)) * p_band)
            if next_carrier >= bc_carrier:
                next_carrier = bc_carrier - p_band
            log_fn(
                f"  Whole-state planner reduces the canonical WFN band "
                f"carrier {bc_carrier} -> {next_carrier}: {exc}")
            bc_carrier = next_carrier
    band_chunk_ranges = tuple(
        (b0, min(b0 + bc_carrier, b_end))
        for b0 in range(b_start, b_end, bc_carrier))

    log_fn(
        f"  Whole-state randomized QRCP: states={m_states} "
        f"({nk} k * {nb} bands), full-Bloch dimension={state_dim}, "
        f"max_search=ceil({search_multiplier:g}*{nb}) -> {max_search}, "
        f"candidates={n_candidates} (+{candidate_carrier-n_candidates} "
        f"inactive mesh pad), qr_eps={float(qr_eps):.3e}, seed={qrcp_seed}, "
        f"rng={_QRCP_RNG_VERSION}, WFN band carrier={bc_carrier}")
    log_fn(f"  [qrcp] candidate SHA256={candidate_hash}")

    rep = NamedSharding(mesh_xy, P())
    face = NamedSharding(mesh_xy, P('x', 'y'))
    row = NamedSharding(mesh_xy, P(('x', 'y'), None))

    with build_psi_G_store(
            wfn=wfn, mesh_xy=mesh_xy, meta=meta,
            band_chunk_ranges=band_chunk_ranges, bispinor=bispinor,
            band_pad_to=bc_carrier) as source:
        sketch = _build_randomized_state_sketch(
            source=source, meta=meta, mesh_xy=mesh_xy,
            band_start=b_start, band_count=nb,
            candidate_states=candidates,
            candidate_carrier=candidate_carrier,
            sketch_rows=max_search, seed=qrcp_seed,
            q_tile_budget=int(q_tile_budget),
            device_pool_limit=device_pool_limit, log_fn=log_fn)

        @partial(jax.jit, out_shardings=(face, rep, rep))
        def _normalized_sketch_gram(y):
            norms = jnp.sqrt(jnp.sum(jnp.abs(y) ** 2, axis=0))
            active = jnp.arange(y.shape[1]) < n_candidates
            safe = jnp.where(active, jnp.maximum(norms, 1.0e-300), 1.0)
            yn = jnp.where(active[None, :], y / safe[None, :], 0.0)
            gram = jnp.einsum(
                'ra,rb->ab', jnp.conj(yn), yn, optimize=True)
            gram = jax.lax.with_sharding_constraint(gram, face)
            gram = 0.5 * (gram + gram.conj().T)
            return (gram,
                    jnp.min(jnp.where(active, norms, jnp.inf)),
                    jnp.max(jnp.where(active, norms, 0.0)))

        sketch_gram, sketch_norm_min, sketch_norm_max = \
            _normalized_sketch_gram(sketch)
        del sketch
        if (not np.isfinite(float(sketch_norm_min))
                or not np.isfinite(float(sketch_norm_max))
                or float(sketch_norm_min) <= 0.0):
            raise ValueError(
                "fit_galerkin_basis: randomized sketch produced a zero or "
                "non-finite physical candidate norm: min/max="
                f"{float(sketch_norm_min):.6e}/"
                f"{float(sketch_norm_max):.6e}")
        log_fn(
            f"  [qrcp] sketch column norm min/max before normalization="
            f"{float(sketch_norm_min):.6e}/{float(sketch_norm_max):.6e}")
        sketch_gram_row = jax.device_put(sketch_gram, row)
        del sketch_gram

        from common.collectives import device_put_process_local
        active_np = np.zeros(candidate_carrier, dtype=bool)
        active_np[:n_candidates] = True
        active = device_put_process_local(
            active_np, NamedSharding(mesh_xy, P(('x', 'y'))))
        select = make_sharded_pivoted_cholesky_select(
            mesh_xy, candidate_carrier, max_search,
            mesh_axis=('x', 'y'), tol_rel=float(qr_eps) ** 2)
        (piv, _sketch_L, rank_qr_dev, d_final, d_taken,
         tr_residual, psd_info) = select(sketch_gram_row, None, active)
        jax.block_until_ready((piv, rank_qr_dev, psd_info))
        del _sketch_L, d_final, sketch_gram_row, active

        piv_host = np.asarray(piv, dtype=np.int64)
        rank_qr = int(np.asarray(rank_qr_dev))
        d_taken_host = np.asarray(d_taken)
        tr_residual_host = np.asarray(tr_residual)
        psd_host = (
            float(np.asarray(psd_info[0])),
            int(np.asarray(psd_info[1])),
            int(np.asarray(psd_info[2])),
        )
        if rank_qr <= 0:
            raise ValueError(
                "fit_galerkin_basis: randomized QRCP found zero rank")
        if np.any((piv_host[:rank_qr] < 0)
                  | (piv_host[:rank_qr] >= n_candidates)):
            raise RuntimeError(
                "fit_galerkin_basis: QRCP returned an active pivot outside "
                f"the logical candidate set [0,{n_candidates})")
        pc_floor = float(qr_eps) ** 2
        if psd_host[0] < -pc_floor:
            raise RuntimeError(
                "fit_galerkin_basis: sketched candidate Gram is not PSD: "
                f"minimum residual {psd_host[0]:.6e} at candidate "
                f"{psd_host[1]}, step {psd_host[2]}, below "
                f"-{pc_floor:.6e}")
        rank_phys = min(rank_qr, 2500)
        structural_search = max_search >= min(state_dim, m_states)
        if rank_phys > 0.9 * max_search and not structural_search:
            raise ValueError(
                "fit_galerkin_basis: the randomized QRCP search saturated: "
                f"delivered rank {rank_phys} exceeds 90% of "
                f"max_search={max_search}. Increase "
                "htransform_rank_multiplier or increase qr_eps; silently "
                "clipping this basis makes locality a tuning artifact.")
        selected = candidates[piv_host[:rank_phys]]
        pivot_hash = hashlib.sha256(
            selected.astype("<i8", copy=False).tobytes()).hexdigest()
        rank = round_up(rank_phys, align)
        if extra_rank_pad:
            rank = round_up(rank + extra_rank_pad, align)
        n_pad = rank - rank_phys
        log_fn(
            f"  [qrcp] raw rank={rank_qr}, delivered physical rank="
            f"{rank_phys}" + (" (upstream safety cap 2500)"
                              if rank_qr > 2500 else "")
            + (f", +{n_pad} exact-null mesh pad -> {rank}" if n_pad else ""))
        log_fn(
            f"  [qrcp] pivot SHA256={pivot_hash}; first/last picked "
            f"residual={d_taken_host[0]:.6e}/"
            f"{d_taken_host[rank_qr-1]:.6e}; terminal trace residual="
            f"{tr_residual_host[rank_qr]:.6e}")

        selected_gram = _build_selected_state_gram(
            source=source, meta=meta, mesh_xy=mesh_xy,
            band_start=b_start, band_count=nb,
            selected_states=selected, rank_carrier=rank,
            q_tile_budget=int(q_tile_budget),
            device_pool_limit=device_pool_limit, log_fn=log_fn)

        batch_face = NamedSharding(mesh_xy, P(None, 'x', 'y'))

        @partial(jax.jit, out_shardings=batch_face)
        def _prepare_selected(g):
            g = 0.5 * (g + g.conj().T)
            if n_pad:
                pad_diag = jnp.concatenate([
                    jnp.zeros(rank_phys, dtype=jnp.float64),
                    jnp.ones(n_pad, dtype=jnp.float64)])
                g = g + jnp.diag(pad_diag).astype(g.dtype)
            return g[None]

        gram_stack = _prepare_selected(selected_gram)
        del selected_gram
        chol_plan = linalg_plan(
            "cholesky", mesh_xy, backend="native2d", n=rank)
        log_fn(f"  [route] selected-state factor: {chol_plan.describe()}")
        L_stack = chol_plan.batched(gram_stack)
        del gram_stack

        @partial(jax.jit, in_shardings=batch_face,
                 out_shardings=(rep, rep))
        def _replicate_lower(factors):
            L_ = jnp.tril(factors[0])
            return L_, jnp.min(jnp.real(jnp.diag(L_))[:rank_phys])

        L, min_chol_diag = _replicate_lower(L_stack)
        del L_stack
        if (not np.isfinite(float(min_chol_diag))
                or float(min_chol_diag) <= 0.0):
            raise ValueError(
                "fit_galerkin_basis: selected physical states are linearly "
                f"dependent; min diag(L)={float(min_chol_diag):.6e}")
        log_fn(
            f"  [qrcp] selected-state min diag(L)="
            f"{float(min_chol_diag):.6e}")

        ctilde = _build_physical_coefficients(
            source=source, meta=meta, mesh_xy=mesh_xy,
            band_start=b_start, band_count=nb,
            selected_states=selected, rank_carrier=rank, factor=L,
            q_tile_budget=int(q_tile_budget),
            device_pool_limit=device_pool_limit, log_fn=log_fn)

    # Centroids enter only here, as evaluation points of the already-fixed
    # global basis.  This is the canonical WFN centroid loader and therefore
    # retains its FFT boxing, Bloch phase, padding and sharding conventions.
    psi_rmu, _ = load_centroids_band_chunked(
        wfn, None, meta, centroid_indices, bispinor, mesh_xy,
        band_range=(b_start, b_end), band_chunk_size=bc_carrier)
    B_at_mu = _basis_at_nodes_from_selected_states(
        psi_rmu=psi_rmu, selected_states=selected,
        factor=L, rank_carrier=rank, n_nodes=n_mu, mesh_xy=mesh_xy)
    del psi_rmu

    @partial(jax.jit, out_shardings=(rep, rep, rep))
    def _coefficient_receipt(c):
        gram = jnp.einsum('kna,kma->knm', c, jnp.conj(c), optimize=True)
        eye = jnp.eye(c.shape[1], dtype=c.dtype)[None]
        row_norm = jnp.real(jnp.diagonal(gram, axis1=1, axis2=2))
        return (jnp.max(jnp.abs(gram - eye)),
                jnp.max(jnp.abs(row_norm - 1.0)),
                jnp.sqrt(jnp.maximum(0.0, 1.0 - jnp.mean(row_norm))))

    c_ortho, max_missing_norm2, fro_resid = _coefficient_receipt(ctilde)
    selected_dev = np.asarray(selected, dtype=np.int32)

    @partial(jax.jit, out_shardings=(rep, rep))
    def _selected_state_receipt(c, factor):
        picked = c.reshape(m_states, rank)[jnp.asarray(selected_dev)]
        reference = factor[:rank_phys, :]
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(reference)))
        return jnp.max(jnp.abs(picked - reference)), scale

    selected_err, selected_scale = _selected_state_receipt(ctilde, L)
    selected_tol = np.sqrt(np.finfo(np.float64).eps) * float(selected_scale)
    if (not np.isfinite(float(selected_err))
            or float(selected_err) > selected_tol):
        raise ValueError(
            "fit_galerkin_basis: selected-state orientation identity "
            f"C[selected]=L failed: max error {float(selected_err):.3e} "
            f"> sqrt(eps)*scale={selected_tol:.3e}")
    log_fn(
        f"  [gate] physical projection over all coarse states: "
        f"max|C C^H-I|={float(c_ortho):.3e}, "
        f"max missing state norm^2={float(max_missing_norm2):.3e}, "
        f"||Psi-CB||_F/||Psi||_F={float(fro_resid):.3e}; "
        f"max|C[selected]-L|={float(selected_err):.3e} "
        f"(cap {selected_tol:.3e})")
    if rank_record_fn is not None:
        rank_record_fn({
            "method": "whole_state_randomized_qrcp",
            "stacked_states": int(m_states),
            "state_dimension": int(state_dim),
            "search_rank": int(max_search),
            "candidate_count": int(n_candidates),
            "raw_rank": int(rank_qr),
            "retained_rank": int(rank_phys),
            "carried_rank": int(rank),
            "null_padding": int(n_pad),
            "rank_multiplier": float(search_multiplier),
            "qr_eps": float(qr_eps),
            "qrcp_seed": int(qrcp_seed),
            "qrcp_rng_version": _QRCP_RNG_VERSION,
            "candidate_hash": candidate_hash,
            "pivot_hash": pivot_hash,
            "min_cholesky_diagonal": float(min_chol_diag),
            "coefficient_orthogonality_error": float(c_ortho),
            "max_missing_state_norm_squared": float(max_missing_norm2),
            "relative_frobenius_residual": float(fro_resid),
            "selected_orientation_error": float(selected_err),
            "selected_orientation_tolerance": float(selected_tol),
        })

    projector = None
    if include_projector:
        # Compatibility carrier for the existing refit consumer.  It is the
        # sparse selector followed by the same physical triangular solve;
        # no per-k gauge exists.  The follow-on low-memory refit migration
        # consumes ``selected_state_indices`` + ``selection_factor`` directly
        # and removes this dense compatibility object.
        @partial(jax.jit, out_shardings=rep)
        def _selector_projector(L):
            E = jnp.zeros((rank, m_states), dtype=jnp.complex128)
            E = E.at[jnp.arange(rank_phys), jnp.asarray(selected)].set(1.0)
            return jsp_linalg.solve_triangular(L, E, lower=True)
        projector = _selector_projector(L)

    return GalerkinBasis(
        ctilde=ctilde,
        basis_at_nodes=B_at_mu,
        projector=projector,
        rank_physical=rank_phys,
        band_range=(b_start, b_end),
        selected_state_indices=tuple(int(v) for v in selected),
        selection_factor=L,
        qrcp_seed=qrcp_seed,
        qrcp_rng_version=_QRCP_RNG_VERSION,
        qrcp_eps=float(qr_eps),
        qrcp_raw_rank=rank_qr,
        qrcp_search_rank=max_search,
        candidate_hash=candidate_hash,
        pivot_hash=pivot_hash,
    )




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


_SELECTED_FILL_KERNELS: dict = {}
_SKETCH_RANDOM_KERNELS: dict = {}
_SKETCH_ACCUM_KERNELS: dict = {}
_BASIS_SOLVE_KERNELS: dict = {}
_PHYSICAL_PROJECT_KERNELS: dict = {}
_COEFFICIENT_ASSEMBLERS: dict = {}


def _state_rows_for_band_chunk(
        state_indices, *, band_start: int, band_count: int,
        band_range: tuple[int, int], band_carrier: int,
        row_carrier: int):
    """Map fixed stacked-state rows into one canonical band carrier."""
    states = np.asarray(state_indices, dtype=np.int64)
    if states.ndim != 1 or states.size > int(row_carrier):
        raise ValueError(
            "_state_rows_for_band_chunk: state list must be one-dimensional "
            f"and fit row_carrier={row_carrier}; got {states.shape}")
    rows = np.full(int(row_carrier), -1, dtype=np.int64)
    rows[:states.size] = states
    k_idx = np.where(rows >= 0, rows // int(band_count), 0)
    b_rel = np.where(rows >= 0, rows % int(band_count), 0)
    lo = int(band_range[0]) - int(band_start)
    hi = int(band_range[1]) - int(band_start)
    active = ((rows >= 0) & (b_rel >= lo) & (b_rel < hi))
    take = k_idx * int(band_carrier) + np.maximum(b_rel - lo, 0)
    take = np.where(active, take, 0).astype(np.int32)
    return take, active


def _make_selected_fill_kernel(
        *, mesh: Mesh, row_count: int, nk: int, band_carrier: int,
        nspinor: int, r_carrier: int, psi_layout, row_layout):
    key = (id(mesh), int(row_count), int(nk), int(band_carrier),
           int(nspinor), int(r_carrier), tuple(psi_layout.spec),
           tuple(row_layout.spec))
    fn = _SELECTED_FILL_KERNELS.get(key)
    if fn is not None:
        return fn
    rep = NamedSharding(mesh, P())

    @partial(
        jax.jit, donate_argnums=(3,),
        in_shardings=(psi_layout, rep, rep, row_layout),
        out_shardings=row_layout)
    def _fill(psi_bc, take, active, rows):
        psi_flat = psi_bc.reshape(
            int(nk) * int(band_carrier), int(nspinor), int(r_carrier))
        picked = psi_flat[take]
        picked = jnp.where(active[:, None, None], picked, 0.0)
        return rows + picked

    _SELECTED_FILL_KERNELS[key] = _fill
    return _fill


def _make_sketch_random_kernel(
        *, mesh: Mesh, sketch_rows: int, nspinor: int, r_carrier: int,
        row_layout, seed: int):
    key = (id(mesh), int(sketch_rows), int(nspinor), int(r_carrier),
           tuple(row_layout.spec), int(seed))
    fn = _SKETCH_RANDOM_KERNELS.get(key)
    if fn is not None:
        return fn
    rep = NamedSharding(mesh, P())

    @partial(
        jax.jit, in_shardings=(rep, rep), out_shardings=row_layout)
    def _draw(r_start, logical_width):
        # A physical grid point owns its PRNG key.  Drawing one normal block
        # per global r index makes the sketch invariant to r-chunk boundaries,
        # device count and Q-memory budget; only physics inputs (seed, rank,
        # spinor count and global r) can change it.  The per-key output shape
        # is fixed, so vmap length also cannot perturb retained values.
        base = jax.random.PRNGKey(int(seed))
        global_r = (r_start.astype(jnp.uint32)
                    + jnp.arange(int(r_carrier), dtype=jnp.uint32))
        keys = jax.vmap(lambda r: jax.random.fold_in(base, r))(global_r)
        omega_r = jax.vmap(
            lambda key: jax.random.normal(
                key, (int(sketch_rows), int(nspinor)), dtype=jnp.float64)
        )(keys)
        omega = jnp.moveaxis(omega_r, 0, 2)
        active_r = jnp.arange(int(r_carrier)) < logical_width
        return jnp.where(active_r[None, None, :], omega, 0.0)

    _SKETCH_RANDOM_KERNELS[key] = _draw
    return _draw


def _make_sketch_accum_kernel(
        *, mesh: Mesh, sketch_rows: int, candidate_carrier: int,
        take_count: int, nk: int, band_carrier: int, nspinor: int,
        r_carrier: int, psi_layout, random_layout):
    key = (id(mesh), int(sketch_rows), int(candidate_carrier),
           int(take_count), int(nk), int(band_carrier), int(nspinor),
           int(r_carrier), tuple(psi_layout.spec), tuple(random_layout.spec))
    fn = _SKETCH_ACCUM_KERNELS.get(key)
    if fn is not None:
        return fn
    rep = NamedSharding(mesh, P())

    @partial(
        jax.jit, donate_argnums=(5,),
        in_shardings=(random_layout, psi_layout, rep, rep, rep, rep),
        out_shardings=rep)
    def _accum(omega, psi_bc, take, destination, active, sketch):
        psi_flat = psi_bc.reshape(
            int(nk) * int(band_carrier), int(nspinor), int(r_carrier))
        picked = psi_flat[take]
        picked = jnp.where(active[:, None, None], picked, 0.0)
        # Upstream applies a REAL Gaussian left sketch without conjugating
        # the wavefunction columns.  The r contraction is globally reduced
        # because both inputs carry the canonical product-r sharding while
        # the result is replicated.
        partial = jnp.einsum(
            'asr,csr->ac', omega, picked, optimize=True)
        partial = jnp.where(active[None, :], partial, 0.0)
        return sketch.at[:, destination].add(partial)

    _SKETCH_ACCUM_KERNELS[key] = _accum
    return _accum


def _candidate_chunk_maps(
        candidate_states, *, band_start: int, band_count: int,
        band_chunk_ranges, band_carrier: int):
    """Compact candidate maps with one static width across band chunks."""
    candidates = np.asarray(candidate_states, dtype=np.int64)
    k_idx = candidates // int(band_count)
    b_rel = candidates % int(band_count)
    selections = []
    for bc_range in band_chunk_ranges:
        lo = int(bc_range[0]) - int(band_start)
        hi = int(bc_range[1]) - int(band_start)
        pos = np.flatnonzero((b_rel >= lo) & (b_rel < hi))
        take = (k_idx[pos] * int(band_carrier)
                + (b_rel[pos] - lo)).astype(np.int32)
        selections.append((take, pos.astype(np.int32)))
    width = max((int(t.size) for t, _ in selections), default=0)
    if width <= 0:
        raise ValueError("randomized QRCP candidate schedule is empty")
    maps = []
    for take, pos in selections:
        active = np.zeros(width, dtype=bool)
        active[:take.size] = True
        take_pad = np.zeros(width, dtype=np.int32)
        pos_pad = np.zeros(width, dtype=np.int32)
        take_pad[:take.size] = take
        pos_pad[:pos.size] = pos
        maps.append((take_pad, pos_pad, active))
    return width, tuple(maps)


def _build_randomized_state_sketch(
        *, source, meta, mesh_xy: Mesh,
        band_start: int, band_count: int,
        candidate_states, candidate_carrier: int,
        sketch_rows: int, seed: int, q_tile_budget: int,
        device_pool_limit: float | None, log_fn):
    """Stream ``Omega Psi_candidate^T`` without materializing ``Omega``."""
    del device_pool_limit  # exact compiled FFT HWM is joined by the planner lane
    nk = int(meta.nk_tot)
    nspinor = int(meta.nspinor)
    n_rtot = int(meta.n_rtot)
    band_carrier = int(source.band_chunk_carrier)
    product_r_spec = P(None, None, None, ('y', 'x'))
    psi_layout = NamedSharding(mesh_xy, product_r_spec)
    random_layout = NamedSharding(mesh_xy, P(None, None, ('y', 'x')))
    rep = NamedSharding(mesh_xy, P())
    r_divisor = spec_divisor(mesh_xy, random_layout.spec, axis=2)
    plan = plan_galerkin_stream(
        rank=int(sketch_rows), nspinor=nspinor, n_rtot=n_rtot,
        r_mesh_divisor=r_divisor, q_tile_budget=int(q_tile_budget))
    take_count, maps_np = _candidate_chunk_maps(
        candidate_states, band_start=band_start, band_count=band_count,
        band_chunk_ranges=source.band_chunk_ranges,
        band_carrier=band_carrier)
    from common.collectives import device_put_process_local
    maps = tuple(
        tuple(device_put_process_local(arr, rep) for arr in entry)
        for entry in maps_np)

    @partial(jax.jit, out_shardings=rep)
    def _zeros():
        return jnp.zeros(
            (int(sketch_rows), int(candidate_carrier)),
            dtype=jnp.complex128)

    sketch = _zeros()
    t0 = time.time()
    for r_idx, (r0, r1) in enumerate(plan.r_chunk_ranges):
        r_carrier = round_up(r1 - r0, r_divisor)
        draw = _make_sketch_random_kernel(
            mesh=mesh_xy, sketch_rows=sketch_rows, nspinor=nspinor,
            r_carrier=r_carrier, row_layout=random_layout, seed=seed)
        omega = draw(
            jnp.asarray(r0, dtype=jnp.int32),
            jnp.asarray(r1 - r0, dtype=jnp.int32))
        accum = _make_sketch_accum_kernel(
            mesh=mesh_xy, sketch_rows=sketch_rows,
            candidate_carrier=candidate_carrier, take_count=take_count,
            nk=nk, band_carrier=band_carrier, nspinor=nspinor,
            r_carrier=r_carrier, psi_layout=psi_layout,
            random_layout=random_layout)
        for bc_idx, (_, psi_bc) in enumerate(source.iter_rchunk_bandwise(
                r0, r1, product_r_spec=product_r_spec)):
            take, destination, active = maps[bc_idx]
            sketch = accum(
                omega, psi_bc, take, destination, active, sketch)
            del psi_bc
        jax.block_until_ready(sketch)
        del omega
        log_fn(
            f"  QRCP sketch r-chunk {r_idx+1}/{len(plan.r_chunk_ranges)}: "
            f"[{r0},{r1}) -> carrier {r_carrier}")
    log_fn(
        f"  QRCP Gaussian sketch: {len(plan.r_chunk_ranges)} r chunk(s) x "
        f"{len(source.band_chunk_ranges)} band chunk(s), "
        f"{time.time()-t0:.2f}s")
    return sketch


def _selected_maps_on_device(
        *, source, selected_states, rank_carrier: int,
        band_start: int, band_count: int, mesh_xy: Mesh):
    from common.collectives import device_put_process_local
    rep = NamedSharding(mesh_xy, P())
    return tuple(
        tuple(device_put_process_local(arr, rep) for arr in
              _state_rows_for_band_chunk(
                  selected_states, band_start=band_start,
                  band_count=band_count, band_range=bc_range,
                  band_carrier=source.band_chunk_carrier,
                  row_carrier=rank_carrier))
        for bc_range in source.band_chunk_ranges)


def _build_selected_rows_for_rchunk(
        *, source, mesh_xy: Mesh, meta, selected_maps,
        rank_carrier: int, r0: int, r1: int, row_layout, psi_layout):
    nk = int(meta.nk_tot)
    nspinor = int(meta.nspinor)
    r_carrier = round_up(
        int(r1) - int(r0),
        spec_divisor(mesh_xy, row_layout.spec, axis=2))

    @partial(jax.jit, static_argnums=(0,), out_shardings=row_layout)
    def _zeros(r_extent):
        return jnp.zeros(
            (int(rank_carrier), nspinor, r_extent), dtype=jnp.complex128)

    rows = _zeros(r_carrier)
    fill = _make_selected_fill_kernel(
        mesh=mesh_xy, row_count=rank_carrier, nk=nk,
        band_carrier=source.band_chunk_carrier, nspinor=nspinor,
        r_carrier=r_carrier, psi_layout=psi_layout, row_layout=row_layout)
    for bc_idx, (_, psi_bc) in enumerate(source.iter_rchunk_bandwise(
            r0, r1, product_r_spec=psi_layout.spec)):
        take, active = selected_maps[bc_idx]
        rows = fill(psi_bc, take, active, rows)
        del psi_bc
    return rows


def _build_selected_state_gram(
        *, source, meta, mesh_xy: Mesh, band_start: int, band_count: int,
        selected_states, rank_carrier: int, q_tile_budget: int,
        device_pool_limit: float | None, log_fn):
    """Exact physical ``X X^H`` for the sketch-selected WFN states."""
    del device_pool_limit
    nspinor = int(meta.nspinor)
    n_rtot = int(meta.n_rtot)
    row_spec = P(None, None, ('y', 'x'))
    row_layout = NamedSharding(mesh_xy, row_spec)
    psi_layout = NamedSharding(mesh_xy, P(None, None, None, ('y', 'x')))
    face = NamedSharding(mesh_xy, P('x', 'y'))
    r_divisor = spec_divisor(mesh_xy, row_spec, axis=2)
    plan = plan_galerkin_stream(
        rank=rank_carrier, nspinor=nspinor, n_rtot=n_rtot,
        r_mesh_divisor=r_divisor, q_tile_budget=q_tile_budget)
    maps = _selected_maps_on_device(
        source=source, selected_states=selected_states,
        rank_carrier=rank_carrier, band_start=band_start,
        band_count=band_count, mesh_xy=mesh_xy)

    @partial(jax.jit, out_shardings=face)
    def _zeros_face():
        return jnp.zeros(
            (int(rank_carrier), int(rank_carrier)), dtype=jnp.complex128)

    gram = _zeros_face()
    fold = _make_fold_G_kernel(
        rank_carrier, mesh_xy, row_layout, face)
    t0 = time.time()
    for r_idx, (r0, r1) in enumerate(plan.r_chunk_ranges):
        rows = _build_selected_rows_for_rchunk(
            source=source, mesh_xy=mesh_xy, meta=meta,
            selected_maps=maps, rank_carrier=rank_carrier,
            r0=r0, r1=r1, row_layout=row_layout,
            psi_layout=psi_layout)
        gram = fold(rows, gram)
        jax.block_until_ready(gram)
        del rows
        log_fn(
            f"  selected-state Gram r-chunk "
            f"{r_idx+1}/{len(plan.r_chunk_ranges)}: [{r0},{r1})")
    log_fn(f"  Exact selected-state Gram: {time.time()-t0:.2f}s")
    return gram


def _make_basis_solve_kernel(
        *, mesh: Mesh, rank: int, nspinor: int, r_carrier: int,
        row_layout):
    key = (id(mesh), int(rank), int(nspinor), int(r_carrier),
           tuple(row_layout.spec))
    fn = _BASIS_SOLVE_KERNELS.get(key)
    if fn is not None:
        return fn
    rep = NamedSharding(mesh, P())

    @partial(
        jax.jit, donate_argnums=(1,),
        in_shardings=(rep, row_layout), out_shardings=row_layout)
    def _solve(L, selected_rows):
        rhs = jnp.moveaxis(selected_rows, 1, 0)
        basis = jax.vmap(
            lambda x: jsp_linalg.solve_triangular(L, x, lower=True))(rhs)
        return jnp.moveaxis(basis, 0, 1)

    _BASIS_SOLVE_KERNELS[key] = _solve
    return _solve


def _make_physical_project_kernel(
        *, mesh: Mesh, nk: int, band_carrier: int, rank: int,
        nspinor: int, r_carrier: int, psi_layout, basis_layout):
    key = (id(mesh), int(nk), int(band_carrier), int(rank), int(nspinor),
           int(r_carrier), tuple(psi_layout.spec), tuple(basis_layout.spec))
    fn = _PHYSICAL_PROJECT_KERNELS.get(key)
    if fn is not None:
        return fn
    rep = NamedSharding(mesh, P())

    @partial(
        jax.jit, donate_argnums=(2,),
        in_shardings=(psi_layout, basis_layout, rep), out_shardings=rep)
    def _project(psi_bc, basis, coefficients):
        delta = jnp.einsum(
            'kbsr,asr->kba', psi_bc, jnp.conj(basis), optimize=True)
        return coefficients + delta

    _PHYSICAL_PROJECT_KERNELS[key] = _project
    return _project


def _assemble_coefficient_chunks(
        chunks, *, logical_widths, nk: int, rank: int, mesh_xy: Mesh):
    widths = tuple(int(v) for v in logical_widths)
    key = (id(mesh_xy), widths, int(nk), int(rank))
    fn = _COEFFICIENT_ASSEMBLERS.get(key)
    if fn is None:
        rep = NamedSharding(mesh_xy, P())

        @partial(jax.jit, in_shardings=tuple(rep for _ in widths),
                 out_shardings=rep)
        def _assemble(*values):
            return jnp.concatenate(
                tuple(v[:, :w, :] for v, w in zip(values, widths)), axis=1)

        fn = _COEFFICIENT_ASSEMBLERS[key] = _assemble
    return fn(*chunks)


def _build_physical_coefficients(
        *, source, meta, mesh_xy: Mesh, band_start: int, band_count: int,
        selected_states, rank_carrier: int, factor,
        q_tile_budget: int, device_pool_limit: float | None, log_fn):
    """Stream ``C = Psi B^H`` in the one selected-state basis gauge."""
    del device_pool_limit
    nk = int(meta.nk_tot)
    nspinor = int(meta.nspinor)
    n_rtot = int(meta.n_rtot)
    band_carrier = int(source.band_chunk_carrier)
    row_spec = P(None, None, ('y', 'x'))
    row_layout = NamedSharding(mesh_xy, row_spec)
    psi_layout = NamedSharding(mesh_xy, P(None, None, None, ('y', 'x')))
    rep = NamedSharding(mesh_xy, P())
    r_divisor = spec_divisor(mesh_xy, row_spec, axis=2)
    plan = plan_galerkin_stream(
        rank=rank_carrier, nspinor=nspinor, n_rtot=n_rtot,
        r_mesh_divisor=r_divisor, q_tile_budget=q_tile_budget)
    maps = _selected_maps_on_device(
        source=source, selected_states=selected_states,
        rank_carrier=rank_carrier, band_start=band_start,
        band_count=band_count, mesh_xy=mesh_xy)

    @partial(jax.jit, out_shardings=rep)
    def _zeros_coeff():
        return jnp.zeros(
            (nk, band_carrier, rank_carrier), dtype=jnp.complex128)

    chunks = [_zeros_coeff() for _ in source.band_chunk_ranges]
    t0 = time.time()
    for r_idx, (r0, r1) in enumerate(plan.r_chunk_ranges):
        r_carrier = round_up(r1 - r0, r_divisor)
        selected_rows = _build_selected_rows_for_rchunk(
            source=source, mesh_xy=mesh_xy, meta=meta,
            selected_maps=maps, rank_carrier=rank_carrier,
            r0=r0, r1=r1, row_layout=row_layout,
            psi_layout=psi_layout)
        solve = _make_basis_solve_kernel(
            mesh=mesh_xy, rank=rank_carrier, nspinor=nspinor,
            r_carrier=r_carrier, row_layout=row_layout)
        basis = solve(factor, selected_rows)
        project = _make_physical_project_kernel(
            mesh=mesh_xy, nk=nk, band_carrier=band_carrier,
            rank=rank_carrier, nspinor=nspinor, r_carrier=r_carrier,
            psi_layout=psi_layout, basis_layout=row_layout)
        for bc_idx, (_, psi_bc) in enumerate(source.iter_rchunk_bandwise(
                r0, r1, product_r_spec=psi_layout.spec)):
            chunks[bc_idx] = project(psi_bc, basis, chunks[bc_idx])
            del psi_bc
        jax.block_until_ready(tuple(chunks))
        del basis
        log_fn(
            f"  physical projection r-chunk "
            f"{r_idx+1}/{len(plan.r_chunk_ranges)}: [{r0},{r1})")
    widths = tuple(
        int(hi) - int(lo) for lo, hi in source.band_chunk_ranges)
    out = _assemble_coefficient_chunks(
        tuple(chunks), logical_widths=widths, nk=nk,
        rank=rank_carrier, mesh_xy=mesh_xy)
    log_fn(f"  Physical C=Psi B^H projection: {time.time()-t0:.2f}s")
    return out


def _basis_at_nodes_from_selected_states(
        *, psi_rmu, selected_states, factor, rank_carrier: int,
        n_nodes: int, mesh_xy: Mesh):
    """Evaluate ``B=L^-1 X`` at registered centroids in the same gauge."""
    selected = np.asarray(selected_states, dtype=np.int64)
    rank_phys = int(selected.size)
    nspinor = int(psi_rmu.shape[2])
    mu_carrier = int(psi_rmu.shape[3])
    out_sharding = _fit(
        mesh_xy, P(None, None, 'y'),
        (int(rank_carrier), nspinor, int(n_nodes)),
        "galerkin.basis_at_nodes(mu-axis)")
    rep = NamedSharding(mesh_xy, P())
    in_sharding = psi_rmu.sharding

    @partial(
        jax.jit, in_shardings=(in_sharding, rep),
        out_shardings=out_sharding)
    def _evaluate(psi, L):
        flat = psi.reshape(-1, nspinor, mu_carrier)
        rows = flat[jnp.asarray(selected)]
        if int(rank_carrier) > rank_phys:
            rows = jnp.pad(
                rows, ((0, int(rank_carrier) - rank_phys), (0, 0), (0, 0)))
        rhs = jnp.moveaxis(rows, 1, 0)
        basis = jax.vmap(
            lambda x: jsp_linalg.solve_triangular(L, x, lower=True))(rhs)
        return jnp.moveaxis(basis, 0, 1)[..., :int(n_nodes)]

    return _evaluate(psi_rmu, factor)


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
