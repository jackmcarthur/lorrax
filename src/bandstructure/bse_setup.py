"""BSE interpolation setup — fine-k wfn recovery via standard htransform.

Top-level entry point for the BSE-interpolation handoff. Takes the
coarse-grid htransform outputs (``ctilde``, ``B_at_mu``, ``enk_sigma``)
plus a finer uniform k-grid spec, and produces wfns at the
already-selected coarse-grid centroids on the fine grid via the
standard htransform pipeline:

    fH_R       = build_fH_R(ctilde, enk_sigma, kgrid_co)
    fH_q       = Σ_R e^{-2πi q·R} fH_R                       # one Fourier sum
    (lam, c)   = eigh(fH_q)                                  # eigvals f(ε), eigvecs in α-basis
    ψ_n,q(r_μ) = Σ_α c_n,q[α] · B_at_μ[α, s, μ]              # reconstruction

The resulting wfns are returned as the canonical X/Y-sharded bundle
used elsewhere in LORRAX (matching ``load_centroids_band_chunked``),
ready to drop in as a BSE interpolation input.

This module deliberately stays narrow: it calls the core htransform
routines (``build_fH_R``, ``build_R_grid_np``, ``newton_inv``) and
adds only the fine-grid k-list construction, the per-q eigh path, and
the X/Y reshard.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from distrib_la import plan as linalg_plan, mesh_key as _mesh_key   # noqa: E402
# Every extent this module shards is an input datum: n_μ is a k-means centroid
# count, nb_fi is the caller's band window, and the q-batch is batch_size — so
# each of them divides the mesh axis only by luck.  ``bse_io`` pads its own μ
# copy (``padded_mu_extent``); this module never did, which made
# ``exciton_bands`` unrunnable at EVERY P>1 on any deck whose centroid count is
# not a multiple of the mesh axis (785 = 5·157 on the MoS2 4×4 deck: job
# 7882476 cells exb16s/exb64s).  See common/sharding_fit.py.
from common.sharding_fit import fit_sharding as _fit, padded_extent as _pad_to
# The (i,j)-face → q-batch move below is a layout change XLA cannot lower in
# one step; ``common.staged_reshard`` is the two-all_to_all staging that makes
# it legal.  See the block at ``fH_q(qbatch)``.
from common.staged_reshard import (
    face_to_batch_reshard as _face_to_batch_reshard,
    face_to_batch_reshard_supported as _face_to_batch_supported,
    ROUTES as _RESHARD_ROUTES,
    DEFAULT_ROUTE as _RESHARD_ROUTE_DEFAULT,
)

from .htransform import build_fH_R, build_R_grid_np, newton_inv


def resolve_reshard_route(explicit=None, *, log_fn=None) -> str:
    """Which ``staged_reshard`` schedule the fH_q face→batch move uses.

    Precedence: an explicit argument, else ``LORRAX_FACE_TO_BATCH_ROUTE``,
    else :data:`common.staged_reshard.DEFAULT_ROUTE`.  The env var exists
    because the two routes have to be A/B'd through the production
    ``bse.exciton_bands`` driver, which this module cannot add an argument
    to (file ownership), and it is the SAME knob the kwarg sets — one
    resolver, not two policies.

    An unrecognised token is ANNOUNCED with the project's ``*** LORRAX
    SANITY`` marker and falls back to the default, never silently: a typo'd
    route that quietly ran the default is exactly how an A/B comes back
    "no difference".
    """
    log = log_fn if log_fn is not None else (lambda *a, **kw: None)
    if explicit is not None:
        route = str(explicit).strip().lower()
        if route not in _RESHARD_ROUTES:
            raise ValueError(
                f"reshard_route={explicit!r} unknown; expected one of "
                f"{' | '.join(_RESHARD_ROUTES)}.")
        return route
    raw = os.environ.get("LORRAX_FACE_TO_BATCH_ROUTE")
    if raw is None or not raw.strip():
        return _RESHARD_ROUTE_DEFAULT
    tok = raw.strip().lower()
    if tok in _RESHARD_ROUTES:
        return tok
    log(f"  *** LORRAX SANITY: LORRAX_FACE_TO_BATCH_ROUTE={raw!r} is not a "
        f"known route.  Accepted: {'/'.join(_RESHARD_ROUTES)}; unset = "
        f"{_RESHARD_ROUTE_DEFAULT}.  The knob is NOT in force. ***")
    return _RESHARD_ROUTE_DEFAULT


#: ``compute_wfns_fi``'s jits, keyed per SIGNATURE and held for the process.
#:
#: They used to be five ``jax.jit`` CLOSURES rebuilt on every call, and JAX
#: keys its tracing cache on callable identity — so a fresh closure is a
#: fresh cache line and every invocation re-traced all five from scratch.
#: What that costs is TRACING and lowering, not XLA compilation: the
#: executable cache underneath ``pjit`` is keyed on the jaxpr/avals/shardings
#: rather than on the function object, so the retraced program hashed to the
#: same executable and ``jax_compile_cache``'s ``compiles`` counter stayed
#: flat.  That is why this was invisible to the obvious instrument and why
#: the numbers in the commit message are WALL numbers on repeat calls.
#:
#: It bites wherever ``compute_wfns_fi`` is called more than once per
#: process, which is the ``bse.exciton_bands`` refit loop (``:889``, 5 Q via
#: ``vq_interp.refit_vq``) — every one of those five calls passes the SAME
#: shapes, so four of them were paying a full retrace for nothing.
#:
#: THE KEY MUST COVER EVERY CLOSED-OVER VALUE, not just the arguments.  The
#: arguments take care of themselves (jit keys its own cache on avals and
#: static args); what a stale entry would silently reuse is the CONSTANTS
#: baked into the trace — ``R_grid`` (a pure function of ``kgrid_co``), the
#: mesh, ``rank``, the batch width, the band window's derived extents, the
#: staged-reshard route, and ``newton_inv``'s (a, n, shift).  Every one of
#: them is in a key below.  ``distrib_la.mesh_key`` rather than ``id(mesh)``
#: because these entries do NOT retain the mesh, and the failure mode of an
#: id-keyed cache that does not retain its object is a stale HIT.
#:
#: NOT retained here: the ``_stage_q`` reshard closure.  It is still built on
#: every call, exactly as before, because its factory issues a collective
#: (``warm_mesh_cliques``) that must run on every rank in the same order —
#: caching it would move a collective, which is not a change to make for a
#: warm-up.  A cached ``_q_batch`` therefore holds the FIRST call's
#: ``_stage_q``; that is safe because ``face_to_batch_reshard`` is a pure
#: function of (mesh, axes, route) and all three are in the key.
_WFNS_FI_KERNELS: dict = {}


def _cached_kernel(key, build):
    """The jit for ``key``, built once per process.

    Deliberately a plain dict and not ``functools.lru_cache``: the key is
    assembled by the caller from things an argument-keyed decorator cannot
    see (closed-over constants), and an eviction policy would silently
    reintroduce the retrace this exists to remove.  The tree's own prior art
    is ``htransform._accum_G_cache`` / ``vq_interp._REFIT_KERNELS`` /
    ``distrib_la.plan._SCAN_CACHE``; this is the same shape.
    """
    fn = _WFNS_FI_KERNELS.get(key)
    if fn is None:
        fn = _WFNS_FI_KERNELS[key] = build()
    return fn


def _parse_kgrid_fi(spec) -> tuple[int, int, int]:
    """Accept '8 8 1', '8,8,1', or a tuple/list."""
    if isinstance(spec, (tuple, list)):
        parts = list(spec)
    else:
        parts = str(spec).replace(',', ' ').split()
    if len(parts) != 3:
        raise ValueError(f"kgrid_fi must have 3 entries, got {spec!r}")
    return tuple(int(x) for x in parts)


def _uniform_kgrid_frac(kgrid: tuple[int, int, int]) -> jax.Array:
    """Γ-centred uniform k-grid in crystal coords, wrapped to (-0.5, 0.5]."""
    nx, ny, nz = kgrid
    ix = np.arange(nx, dtype=np.float64) / nx
    iy = np.arange(ny, dtype=np.float64) / ny
    iz = np.arange(nz, dtype=np.float64) / nz
    grid = np.stack(np.meshgrid(ix, iy, iz, indexing='ij'), axis=-1).reshape(-1, 3)
    return jnp.asarray((grid + 0.5) % 1.0 - 0.5)


def compute_wfns_fi(
    *,
    ctilde: jax.Array,
    B_at_mu: jax.Array,
    enk_sigma: jax.Array,
    kgrid_co: tuple[int, int, int],
    kgrid_fi=None,
    band_window_fi: tuple[int, int],
    mesh_xy: Mesh,
    a_band_index: int | None = None,
    batch_size: int | None = None,
    q_list=None,
    return_coeffs: bool = False,
    eigh_backend: str = "auto",
    use_low_mem_eigh: bool = False,
    reshard_route: str | None = None,
    log_fn=None,
):
    """Recover ψ at the coarse-grid centroids on a finer uniform k-grid —
    or on an EXPLICIT list of arbitrary q (the arbitrary-Q generalization,
    arbitrary_q_bse.md §1/§2a: e.g. the shifted grid {k + Q}).

    Args:
        ctilde:    (nk_co, nb, rank) Galerkin coeffs in the rank-α basis,
                   replicated. From ``streaming_galerkin_solve``.
        B_at_mu:   (rank, ns, n_μ) α-basis evaluated at coarse centroids.
        enk_sigma: (nb, nk_co) DFT band energies in Ry.
        kgrid_co:  (nkx, nky, nkz) coarse uniform k-grid.
        kgrid_fi:  (nkx_fi, nky_fi, nkz_fi) fine k-grid; tuple/list/string.
                   Mutually exclusive with ``q_list``.
        band_window_fi: (b_min, b_max) — sub-window of the htransform band
                   axis to RETURN on the fine grid (0-based, exclusive end).
                   fH is ALWAYS built from every band in ``ctilde`` (the full
                   loaded window); this window only selects which eigenpairs
                   are returned.  So a full-band ctilde yields a full-band fH
                   regardless of the sub-window — the returned conduction bands
                   sit interior to it, guarded by the bands above b_max.  This
                   is the single fH builder (shared with the SP driver's
                   ``htransform.build_fH_R``); there is no windowed variant.
        mesh_xy:   ('x','y') device mesh.
        a_band_index: optional band index for the f-transform 'a' parameter
                   (defaults to the top of the htransform window).
        batch_size: q-points per fH_q chunk on the NATIVE eigh path (≥1 jit
                   compile reuse).  ``None`` / ``<=0`` (the DEFAULT) means
                   **N_q_co = prod(kgrid_co)**, the coarse k-point count —
                   input key ``wfn_fi_q_chunk``.  Rationale (owner's design):
                   the chunk's q-batch costs the same bytes, distributed the
                   same way, as ``fH_R``'s own R axis, so a deck whose
                   ``fH_q(q_co)`` fits in half of memory completes the whole
                   fine-grid pass — the chunk size is tied to a quantity the
                   run has already had to afford instead of a bare 32 that
                   means nothing on either a small or a large deck.
                   Treated as a FLOOR, not an exact width: it is rounded UP
                   to a multiple of the device count so the ``P(('x','y'),…)``
                   q-axis sharding is legal (see the ``bs`` block below).
                   The FFI backends read the SAME key but do NOT pad it —
                   their stack carries an unsharded batch axis, so a pad
                   would buy no sharding and cost real distributed solves on
                   rows that are then discarded — and their DEFAULT is 1,
                   which is the width that reproduces the historical per-q
                   path bit for bit.  Asking for more collapses the per-q
                   dispatches (measured 4.8x at C=8) and, on a library with
                   a stacked entry point, the descriptor/workspace too, at a
                   measured ~1-ULP move in the eigenvalues.  See the
                   ``bs_ffi`` block at the FFI loop for the measurement.
        q_list:    optional (nq, 3) fractional q — evaluate at exactly these
                   points instead of a uniform grid.  Wrapped to (−0.5, 0.5]
                   internally (fH_q is exactly BZ-periodic, so wrapping is a
                   no-op on values; it keeps phases well-conditioned).
        return_coeffs: also return the rank-α eigenvector coefficients
                   ``.coeffs_fi`` (nk_fi, rank, nb_fi), sharded
                   ``P(('x','y'))`` on the q axis like every ``_q_batch``
                   output — the per-q ζ-refit consumes them to rebuild ψ on
                   the full r-grid through the streamed α-basis (its chunk
                   kernels reshard as needed).
        eigh_backend: which Hermitian eigensolver decomposes fH_q —
                   ``auto|off`` (default) keeps the q-BATCHED native path,
                   ``distributed|cusolvermp|slate|scalapack`` route ONE
                   (rank, rank) tile at a time through the distributed-linalg
                   FFI.  See the eigh comment in the batch loop below for
                   which regime is which, and ``distrib_la.Plan`` for
                   the backends.
        use_low_mem_eigh: the SECOND eigenvalue path, named by intent rather
                   than by library (input key ``use_low_mem_eigh``).  False
                   (default) = reshard each chunk onto the q axis and run the
                   batched native ``eigh``, every device owning WHOLE
                   (rank, rank) matrices.  True = keep the matrix spread over
                   the μ×ν face and run a DISTRIBUTED eigh, i.e. exactly
                   ``eigh_backend='distributed'`` (ScaLAPACK ``pzheevd`` on a
                   host mesh, cuSOLVERMp on CUDA — ``distrib_la.resolve
                   ._DISTRIBUTED_DEFAULT``).  Slower per q by construction
                   (one matrix ndev-ways, q walked serially) and chosen when
                   ONE whole (rank, rank) matrix no longer fits per rank.
                   It is an OVERRIDE, not a hint: if it cannot be honoured
                   the plan REFUSES at resolve time, before any q is solved,
                   naming the guard that failed — never a silent demote to
                   the very path the caller said it has no memory for.
                   ``use_low_mem_eigh=True`` with ``eigh_backend='off'`` is a
                   contradiction and is refused here; with an explicit
                   library name the name wins (it already IS this path).
        reshard_route: which ``common.staged_reshard`` schedule performs the
                   (i,j)-face → q-batch move — ``split_b_first`` (default) or
                   ``flatten_m_first``.  ``None`` reads
                   ``LORRAX_FACE_TO_BATCH_ROUTE`` (see
                   :func:`resolve_reshard_route`).  Movement only: the two
                   routes are element-identical and differ in schedule, peer
                   counts and whether an M pad is needed.
        log_fn:    optional logger.

    Returns:
        SimpleNamespace bundle (matching ``file_io.tagged_arrays`` convention):
            .psi_rmu_Y:  (nk_fi, nb_fi, ns, n_μ)  P(None, None, None, 'y')
            .psi_rmuT_X: (nk_fi, n_μ, nb_fi, ns)  P(None, 'x', None, None)
            .enk_full:   (nk_fi, nb_fi)           recovered DFT energies (Ry)
                                                  via ``newton_inv`` of fH_q eigvals
            .lam_fi:     (nk_fi, nb_fi)           raw fH_q eigenvalues (= f(ε_n,q))
                                                  kept for diagnostics
            .lam_all_fi: (nk_fi, rank)            the FULL fH_q spectrum, all
                                                  ``rank`` bands, accumulated
                                                  chunk by chunk (see the
                                                  ``lam_all`` note at the
                                                  emit loop)
            .coeffs_fi:  (nk_fi, rank, nb_fi)     only when ``return_coeffs``

    Both wfn copies live on-device and are sharding-distinct so any
    contraction over (n_μ) along either mesh axis stays local.
    """
    log = log_fn if log_fn is not None else (lambda *a, **kw: None)
    if (kgrid_fi is None) == (q_list is None):
        raise ValueError("pass exactly one of kgrid_fi / q_list")
    nb_co = int(ctilde.shape[1])
    rank = int(ctilde.shape[2])
    nspinor = int(B_at_mu.shape[1])
    n_mu = int(B_at_mu.shape[2])

    b_min, b_max = int(band_window_fi[0]), int(band_window_fi[1])
    nb_fi = b_max - b_min
    if nb_fi <= 0 or b_min < 0 or b_max > rank:
        raise ValueError(
            f"band_window_fi {band_window_fi} not a valid sub-range of [0, {rank})")

    # ── The q-CHUNK width: default it to N_q_co, then pad ────────────────
    #
    # OWNER'S RULE.  The chunk defaults to N_q_co = prod(kgrid_co), the coarse
    # k-point count — NOT a bare constant.  ``fH_R`` is (nk_co, rank, rank) and
    # face-sharded, so ONE chunk of nk_co q-points is, byte for byte and axis
    # for axis, the same residency as fH_R itself: nk_co·rank²·16/ndev per rank
    # either way.  A deck that could build fH_R at all can therefore afford one
    # chunk of this width, and the fine-grid pass completes for ANY N_q_fi.  A
    # fixed 32 has no such property: it is 2x over-sized on this 4x4 deck
    # (nk_co=16) and 4.5x under-sized at the converged 12x12 (nk_co=144), where
    # it needlessly multiplies the number of chunks — and the number of
    # collectives — by 4.5.
    nk_co = int(kgrid_co[0]) * int(kgrid_co[1]) * int(kgrid_co[2])
    if int(ctilde.shape[0]) != nk_co:
        raise ValueError(
            f"kgrid_co {tuple(kgrid_co)} implies N_q_co={nk_co} but ctilde has "
            f"{int(ctilde.shape[0])} k-points; the q-chunk default and "
            f"build_fH_R's ifft-reshape both read this extent.")
    if batch_size is None or int(batch_size) <= 0:
        batch_size = nk_co
        _bs_explicit = False
        _bs_origin = f"default N_q_co={nk_co}"
    else:
        batch_size = int(batch_size)
        _bs_explicit = True
        _bs_origin = f"wfn_fi_q_chunk={batch_size}"

    # ── Which of the TWO eigenvalue paths ────────────────────────────────
    # ``use_low_mem_eigh`` names the INTENT; ``eigh_backend`` names the
    # LIBRARY.  One axis, resolved here, so there is exactly one place where
    # "I have no memory for a whole matrix" becomes a backend string.
    #
    # Doctrine 3 (OWNER_DECISIONS / linalg.resolve): an explicit request that
    # cannot be honoured REFUSES; only ``auto`` may demote, and then it must
    # announce.  ``distributed`` is an explicit request in
    # ``distrib_la`` — it runs the whole guard ladder and raises —
    # which is precisely the behaviour this flag needs, so the flag is a
    # rename onto it and adds no second policy.
    _eigh_requested = str(eigh_backend or "auto").strip().lower()
    if use_low_mem_eigh:
        if _eigh_requested in ("auto", "", "none"):
            _eigh_requested = "distributed"
        elif _eigh_requested in ("off", "native"):
            raise ValueError(
                "use_low_mem_eigh=True with eigh_backend='off' is a "
                "contradiction: 'off' pins the q-batched NATIVE eigh, which "
                "is the path that needs a whole (rank, rank) matrix per "
                "device — exactly what the low-memory flag says is "
                "unaffordable.  Set eigh_backend to auto (=> 'distributed', "
                "the platform's distributed library) or name one explicitly "
                "(distributed|cusolvermp|slate|scalapack), or drop the flag.")
        # else: an explicit library name already IS the low-memory path.

    # ── The q-batch width is a PLACEMENT extent, so pad it, don't fit it ──
    #
    # Every array in the native path carries the q-batch on a ``P(('x','y'),…)``
    # axis, i.e. one flattened axis over ALL px*py devices.  A batch width that
    # does not divide px*py is not a slightly-worse sharding, it is NO sharding:
    # ``_fit`` legalises it by dropping the entry, the batch goes replicated,
    # and every device then runs the WHOLE batch of ``eigh`` instead of its own
    # 1/ndev share.  Measured on the MoS2 4x4 deck, 91-Q exciton path
    # (q_list = 91 Q x 16 k = 1456 q), the fixed 32-wide batch:
    #     P=16  32 % 16 == 0  ->  sharded, 2 matrices/device, 105.7 s   (7882569)
    #     P=64  32 % 64 != 0  ->  REPLICATED, 32 matrices/device, 866.5 s (7882533)
    # — 64 devices 8x SLOWER than 16, and the gap widens with ndev because the
    # replicated cost is ndev-independent.
    #
    # Rounding the width UP to a multiple of ndev makes the axis divide at every
    # device count.  Per-device residency STRICTLY falls: whole (rank, rank)
    # matrices per device go from ``batch_size`` (replicated) to
    # ``ceil(batch_size/ndev)``, which is also exactly what the working P=16
    # case already paid.  The extra q-points are the same zero-padded rows the
    # loop already appended and the same ``[:nq]`` slice already discarded —
    # ``lam``/``psi``/``coeffs`` are per-q independent, so padding the batch
    # cannot touch a value, only how many are computed and where.
    bs = _pad_to(mesh_xy, ('x', 'y'), batch_size)
    _ndev = int(mesh_xy.shape['x']) * int(mesh_xy.shape['y'])
    _batch_note = (f"q-chunk={bs} [{_bs_origin}]" if bs == batch_size else
                   f"q-chunk={bs} ({_bs_origin}, padded to a multiple of "
                   f"ndev={_ndev} so the q axis splits)")

    if q_list is None:
        kgrid_fi = _parse_kgrid_fi(kgrid_fi)
        nq = kgrid_fi[0] * kgrid_fi[1] * kgrid_fi[2]
        log(f"  bse_setup: {kgrid_co} → {kgrid_fi} ({nq} q-pts), "
            f"bands [{b_min}, {b_max}) of {rank}, {_batch_note}")
    else:
        q_list = np.asarray(q_list, dtype=np.float64).reshape(-1, 3)
        nq = q_list.shape[0]
        log(f"  bse_setup: {kgrid_co} → explicit q-list ({nq} q-pts), "
            f"bands [{b_min}, {b_max}) of {rank}, {_batch_note}")

    # ── Build fH_R via the shared htransform core ────────────────────────
    fH_k, fH_R, (a_f, n_f, shift), _f_eps = build_fH_R(
        ctilde, enk_sigma, kgrid_co, mesh_xy,
        a_band_index=a_band_index, log_fn=log)
    del fH_k  # diagnostic-only here; not needed downstream

    # fH_R stays SHARDED P(None, 'x', 'y') — the (rank, rank) face is split
    # across the mesh, the lattice-R axis is not.  The q-Fourier sum contracts
    # ONLY over R, so every device can build its own (i, j) tile of fH_q with no
    # communication at all; the single collective is the reshard onto the q axis
    # just before the eigh (see ``_q_batch``).  B_at_mu is replicated — it is
    # (rank, ns, n_μ), three orders smaller.
    #
    # It used to be ``jax.device_put(fH_R, rep)``.  That is nk_co · rank² · 16 B
    # on EVERY device (MoS2 12×12, n_μ=2412: 11 GiB at nb=16 rising to 50 GiB at
    # nb≥36) and, because the source is sharded, JAX routes it through
    # ``x._value`` — a host gather of the same size per process.  It was the
    # single reason a converged k-grid could not run the htransform at any
    # device count (gw_converged_12x12_80ry_2026-07-21 §5, next-step #2).
    rep = NamedSharding(mesh_xy, P())
    B_rep = jax.device_put(B_at_mu, rep)
    R_grid = jnp.asarray(build_R_grid_np(kgrid_co))

    # ── q-points (uniform fine grid or explicit list), padded to batch ───
    if q_list is None:
        q_all = _uniform_kgrid_frac(kgrid_fi)
    else:
        q_all = jnp.asarray((q_list + 0.5) % 1.0 - 0.5)
    n_pad = (-nq) % bs
    q_pad = (jnp.concatenate([q_all, jnp.zeros((n_pad, 3), dtype=q_all.dtype)])
             if n_pad else q_all)

    # ── Per-q(-batch): Fourier sum → eigh → ψ-at-centroids reconstruction ─
    # fH_q is the FULL-band Hamiltonian (all bands in ctilde).  Bands
    # [b_min, b_max) are selected on the eigenvalue axis (ascending): f(eps) is
    # monotone in eps, so ascending eigenvalue index == ascending energy, and
    # this slice is exactly the lowest-energy ``nb_fi`` bands at/above b_min —
    # identical to the SP driver's sort-then-keep, but returning eigenVECTORS
    # too.  The guard bands above b_max stay in fH (they shape the
    # interpolation) but are not returned, so every returned band is interior.
    # Resolve-time backend resolution: every guard (vocabulary, platform,
    # compiled-capability, process coverage, SQUARE mesh — cusolverMpSyevd
    # DEADLOCKS on rectangular blocks — and rank divisibility) fires HERE,
    # before any q is solved.  See distrib_la.resolve_backend for the order.
    try:
        eigh_plan = linalg_plan("eigh", mesh_xy, backend=_eigh_requested,
                                n=rank)
    except (ValueError, RuntimeError) as exc:
        _why = str(exc)
        if "divisible" in _why:
            _why = (f"{_why}  ``streaming_galerkin_solve`` mesh-aligns the "
                    f"retained SVD rank to lcm(px, py); a ctilde built on a "
                    f"different mesh has to be refit on this one.")
        if use_low_mem_eigh:
            # REFUSE AT RESOLVE TIME, from the rank it happened on — never
            # fall back to the native path, which is the one the caller has
            # just declared it cannot afford.  Falling back here would turn
            # an honest "this deck needs a distributed eigh" into an OOM or,
            # worse, a run that silently replicates and finishes.
            raise type(exc)(
                f"use_low_mem_eigh=True could not be honoured on this mesh: "
                f"{_why}  (resolved request: eigh_backend="
                f"{_eigh_requested!r}; process {jax.process_index()}).  "
                f"No fallback: the native q-batched eigh needs a whole "
                f"({rank}, {rank}) matrix per device, which is what this "
                f"flag says is unaffordable.") from None
        raise type(exc)(_why) from None
    native = eigh_plan.is_native
    if use_low_mem_eigh and native:
        # Belt and braces: 'distributed' never resolves to native in
        # distrib_la's resolver (it raises instead), so this is unreachable
        # today — but it is the ONE assertion standing between a future
        # resolver change and a silent honour-in-name-only.
        raise RuntimeError(
            f"use_low_mem_eigh=True resolved to the NATIVE eigh "
            f"(requested {_eigh_requested!r}) — that is the whole-matrix "
            f"path this flag exists to avoid.  distrib_la must "
            f"refuse rather than demote for an explicit request.")
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])

    # ── The (i,j)-face → q-batch reshard, STAGED ─────────────────────────
    # Built HERE, outside every jit, because the factory issues a collective
    # (``warm_mesh_cliques``) and so must run synchronously on every rank.
    # ``_face_to_batch_supported`` is pure arithmetic on (bs, rank, rank) and
    # the mesh, so every rank takes the same branch by construction.  The
    # fallback is the historical single ``with_sharding_constraint`` — kept so
    # a shape this primitive must refuse degrades exactly as it does today
    # (slow, announced) instead of crashing; ``build_fH_R`` already pins
    # fH_R to a NON-fitted ``P(None,'x','y')``, so an indivisible rank cannot
    # actually reach here, and ``bs`` is padded to ndev a few lines above.
    _route = resolve_reshard_route(reshard_route, log_fn=log)
    _stage_q = (_face_to_batch_reshard(
                    mesh_xy, axes=('x', 'y'), route=_route, log_fn=log,
                    divisibility_hint=(
                        "bse_setup pads the q-batch with "
                        "sharding_fit.padded_extent; ``rank`` is pinned by "
                        "streaming_galerkin_solve to lcm(px, py)."))
                if native and _face_to_batch_supported(
                    mesh_xy, (bs, rank, rank), axes=('x', 'y'),
                    route=_route) else None)
    if native and _stage_q is not None:
        log(f"  fH_q reshard: STAGED route={_route} "
            f"(B={bs}, M=N={rank}) on {px}x{py}")
    if native and _stage_q is None:
        log(f"  fH_q reshard: STAGED path unavailable for "
            f"(bs={bs}, rank={rank}) on {px}x{py} — falling back to the "
            f"single sharding_constraint, which XLA lowers as "
            f"replicate-then-repartition ({bs * rank * rank * 16 / 1024**2:.0f}"
            f" MB/device/batch).  See common/staged_reshard.py.")

    # ── The per-signature key for the jit cache ──────────────────────────
    # Every value the kernels below CLOSE OVER, in one tuple: the mesh (by
    # ``distrib_la.mesh_key``, which carries device identity and not just
    # extents), ``kgrid_co`` (which is all ``R_grid`` is), the matrix extent,
    # the three output extents that ``_qbatch_out_shardings`` and the bundle
    # shardings are built from, and the reshard route plus whether the staged
    # primitive accepted this shape.  Arguments are deliberately ABSENT — jit
    # keys its own cache on avals and static args, so putting shapes here
    # would only split entries that are already correctly split.
    _mesh_id = _mesh_key(mesh_xy)
    _sig = (_mesh_id, tuple(int(v) for v in kgrid_co), int(rank),
            int(nb_fi), int(nspinor), int(n_mu), _route,
            _stage_q is not None, bool(native))

    # The two stages either side of the eigh are plain traceable functions,
    # shared verbatim by both backends; only the JIT BOUNDARIES differ.  The
    # native path keeps eigh + projection inside ONE jit — split apart, the
    # full (bs, rank, rank) eigenvector batch becomes a materialised jit output
    # instead of being fused away down to the nb_fi columns actually kept
    # (10.1 GiB/device at bs=32 / rank 4452; it killed a 16 × A100-80GB run).
    def _fourier(q, fH_R, batched):
        """fH_q = Σ_R e^{-2πi q·R} fH_R.

        ``batched``: q is (bs, 3) and the output ends q-SHARDED — the
        ``_kpath_batch`` idiom (htransform.h_transform), one all-to-all after
        which each device owns whole (rank, rank) matrices for its own q-rows
        and the native eigh runs ndev-parallel (28 ms per rank-1152 matrix at
        MoS2 12×12).  Costs bs/ndev WHOLE matrices per device — bs being the
        ndev-padded batch width, so that quotient is exact and never the
        replicated ``bs`` the raw ``batch_size`` used to degrade to.

        Otherwise q is (3,) and the single matrix is left (i, j)-sharded
        ``P('x','y')`` — never whole on any device, rank²·16/ndev instead.
        That is the FFI eigh's input layout; its hermitizing transpose IS a
        collective (rank²·16 B all-to-all, 22 MB/device at rank 4716 on 16
        devices), the price of not materialising the matrix.
        """
        if not batched:
            fH_q = jnp.einsum('k,kij->ij', jnp.exp(-2j * jnp.pi * (R_grid @ q)),
                              fH_R)
            grid = _fit(mesh_xy, P('x', 'y'), (rank, rank),
                        "bse_setup.fH_q(one)")
            fH_q = jax.lax.with_sharding_constraint(fH_q, grid)
            return jax.lax.with_sharding_constraint(
                0.5 * (fH_q + jnp.conj(fH_q).T), grid)
        bs = int(q.shape[0])
        q = jax.lax.with_sharding_constraint(
            q, _fit(mesh_xy, P(('x', 'y'), None), (bs, 3), "bse_setup.q_batch"))
        phase = jnp.exp(-2j * jnp.pi * (q @ R_grid.T))                 # (bs, nk_co)
        # fH_R is (i, j)-sharded and the sum runs over R only → local.  Pin the
        # contraction OUTPUT to that same (i, j) layout: left free, XLA
        # materialises the whole (bs, rank, rank) batch on every device before
        # the reshard below (measured: 9 batches × 11.4 GiB at nb=36 / rank
        # 4716 — the OOM that made the wide windows unmeasurable).
        fH_q = jnp.einsum('qk,kij->qij', phase, fH_R)
        fH_q = jax.lax.with_sharding_constraint(
            fH_q, _fit(mesh_xy, P(None, 'x', 'y'), (bs, rank, rank),
                       "bse_setup.fH_q(face)"))
        # Reshard (i,j)→q FIRST, then hermitize: on the q-sharded layout each
        # device owns whole matrices, so the transpose is local.  Doing it the
        # other way round costs a second all-to-all for ``swapaxes``.
        #
        # THIS RESHARD IS NOT "ONE ALL-TO-ALL" AND XLA WILL NOT DO IT FOR YOU.
        # Asked for the {1,px,py} -> {ndev,1,1} transition directly (a single
        # ``with_sharding_constraint``, which is what stood here), the SPMD
        # partitioner falls back to replicate-then-repartition and materialises
        # the WHOLE (bs, rank, rank) batch on every device — precisely what the
        # face constraint immediately above exists to prevent.  Its own
        # diagnostic, from job 7882974 (91-Q path, MoS2 4x4, rank 672), at BOTH
        # device counts and with the q-batch padding already in:
        #   [SPMD] Involuntary full rematerialization.  The compiler cannot go
        #   from sharding {devices=[1,8,8]<=[64]} to {devices=[64,1,1]<=[64]}
        #   efficiently for ... c128[64,84,84] ... jit(_q_batch)/
        #   sharding_constraint ... As the last resort, SPMD will replicate the
        #   tensor and then partition it, which is inefficient.  (b/433785288)
        # 64 such warnings at P=64 (one per rank), 16 at P=16.  Cost:
        # bs*rank^2*16 per device per batch = 462 MB at P=64 / 231 MB at P=16
        # on this deck, x23 and x46 batches respectively.  (That job logged 128
        # / 32 involuntary-remat lines in total; the OTHER half of them is a
        # different site, ``common.wfn_transforms.iter_psi_rchunk_bandwise``'s
        # band-sharded -> P(None,None,None,'y') constraint on the streamed
        # ψ r-chunk, and belongs to the htransform/FFT workstream.)
        #
        # THE FIX IS STAGING, not moving a constraint.  ``common.staged_reshard``
        # issues the two single-mesh-axis all_to_alls the move really is, inside
        # one shard_map where the partitioner cannot re-plan them.  fH_R stays
        # face-sharded throughout — the obvious alternative, contracting
        # straight into the q layout, would need fH_R gathered per device and
        # that is 51 GB at the converged 12x12 / n_mu=2412 reference, which is
        # the whole reason it is face-sharded.  Every intermediate in the staged
        # chain is one shard, bs*rank^2/ndev, so per-rank residency for this
        # array FALLS by ndev and no staging buffer is added.  Do not
        # "simplify" this by deleting either constraint without measuring: the
        # face one prevents a documented OOM, and the staged call is the only
        # thing standing between this line and a replicated batch.
        #
        # MEASURED (job 7883154 vs 7882974, same deck/args/harness, cache-cold,
        # coll=mpi; unit gate 7883151).  htransform_psi_cQ, 91-Q path:
        #     P=64  231.39 -> 145.53 s  (1.59x)      involuntary remat 64 -> 0
        #     P=16  175.96 -> 186.13 s  (0.95x)      involuntary remat 16 -> 0
        #   P=64/P=16 ratio 1.32 -> 0.78, i.e. 64 devices are finally faster
        #   than 16 on this stage.  The P=16 cell is ~6% SLOWER and that is
        #   reported as measured, not rounded away: at 16 devices the batch it
        #   stops replicating is only 231 MB, while the two all_to_alls run
        #   over 4-device groups, so the trade is close to even there.  (Same
        #   job's solve_scan_cold, which this change does not touch, moved
        #   -2.9% / -1.4%, so the machine had a small tailwind, not a headwind.)
        #   Values: cross-P pin reproduced exactly (max |dE| 0.740000 meV, row
        #   30, Q=[0,0.5,0], branch 7); vs the frozen reference .dat 0.000000
        #   meV at BOTH P and the files are byte-identical.  Max per-rank
        #   VmHWM unchanged (10.685 GiB, set by the BSE solve).
        #   Next lever if this stage ever matters again: ONE all_to_all over
        #   ('x','y') plus a rank-local (px, py) tile untranspose, trading a
        #   collective for a local (rank, rank) shuffle -- unmeasured.
        if _stage_q is not None:
            fH_q = _stage_q(fH_q)
        else:
            fH_q = jax.lax.with_sharding_constraint(
                fH_q, _fit(mesh_xy, P(('x', 'y'), None, None),
                           (bs, rank, rank), "bse_setup.fH_q(qbatch)"))
        return 0.5 * (fH_q + jnp.swapaxes(fH_q, -1, -2).conj())

    def _project(lam, U, B, b_min, b_max):
        """Band-window slice + ψ_n,q(r_μ) = Σ_α c_n,q[α] B_at_μ[α, s, μ].

        Leading-axis agnostic (``...``): the native path hands it a whole
        q-batch, the FFI path one q.  ONE reconstruction for both backends.
        """
        c = U[..., b_min:b_max]                                # (..., rank, nb_fi)
        psi = jnp.einsum('...an,asm->...nsm', c, B)            # (..., nb_fi, ns, n_μ)
        # ``lam`` (ALL ``rank`` bands) comes out LAST, appended to the tuple
        # the three existing consumers already unpack, so nothing above moves.
        # It is the owner's "store eigenvalues up to the full N_band before
        # doing other chunks": the eigensolver produced the whole spectrum
        # either way, and at (bs, rank) f64 q-sharded it is
        # bs·rank·8/ndev B/rank — 122 KB/rank on the 4x4/91-Q deck, 4 orders
        # below the (bs, rank, rank) c128 batch it rides along with — so
        # keeping it costs nothing while discarding it means the chunk has to
        # be RE-SOLVED to answer any question about a band outside
        # [b_min, b_max).  psi/coeffs stay windowed: THOSE are the arrays
        # whose full-band form does not fit.
        return lam[..., b_min:b_max], psi, c, lam

    def _qbatch_out_shardings(bs):
        """EXPLICIT out_shardings for the fused native batch.

        Left to inference, XLA propagates the q-batch sharding forward and it
        lands on whichever output axis it can, then the GSPMD→NamedSharding
        conversion raises when that axis does not divide the mesh — the P=64
        failure was ``dim_size=4 is not divisible by axis_size=8`` on the
        nb_fi=4 BAND axis, an axis nobody intended to shard.  Naming the
        outputs pins the intent (q-batched, everything else whole) and lets
        the fitter degrade the ONE axis that is actually in question.
        """
        return (_fit(mesh_xy, P(('x', 'y'), None), (bs, nb_fi),
                     "bse_setup.lam"),
                _fit(mesh_xy, P(('x', 'y'), None, None, None),
                     (bs, nb_fi, nspinor, n_mu), "bse_setup.psi"),
                _fit(mesh_xy, P(('x', 'y'), None, None), (bs, rank, nb_fi),
                     "bse_setup.coeffs"),
                _fit(mesh_xy, P(('x', 'y'), None), (bs, rank),
                     "bse_setup.lam_all"))

    def _q_batch_kernel(bs_):
        def _build():
            @partial(jax.jit, static_argnames=('b_min', 'b_max'),
                     out_shardings=_qbatch_out_shardings(bs_))
            def _q_batch(q_batch, fH_R, B, b_min, b_max):
                """Native: Fourier sum → batched eigh → projection, ONE fused
                jit."""
                lam, U = jnp.linalg.eigh(_fourier(q_batch, fH_R, True))
                return _project(lam, U, B, b_min, b_max)
            return _q_batch
        return _cached_kernel(("q_batch", _sig, int(bs_)), _build)

    lam_chunks, psi_chunks, c_chunks, lam_all_chunks = [], [], [], []

    def _emit(lam_s, psi_s, c_s, lam_all_s):
        """Stash one CHUNK's four outputs.  Both arms emit a leading chunk
        axis now — the native q-batch, and the FFI arm's ``lax.map``'d chunk
        — so the per-item ``[None]`` rank-expansions this used to carry
        (four eager ``broadcast_in_dim``s per q on the FFI arm) are gone,
        and so is the ``block_until_ready`` that stood on its last line.
        That sync fired once per q on the FFI arm (1456 host round trips on
        the reference deck).  Each loop now owns its own sync POLICY, at its
        own cadence, where the reason for it can be written down."""
        lam_chunks.append(lam_s)
        psi_chunks.append(psi_s)
        lam_all_chunks.append(lam_all_s)
        if return_coeffs:
            c_chunks.append(c_s)

    # Backend choice = parallel-over-q vs parallel-within-matrix.  The native
    # path eigh-es ``bs/ndev`` WHOLE (rank, rank) matrices per device
    # concurrently; the FFI path spreads ONE matrix over the whole mesh and
    # walks q in chunks.  At rank 4716 that is 356 MB/matrix on one device vs
    # 22 MB/device on 16 GPUs — so the FFI backends are what make a window too
    # wide for the batched path runnable at all, at the cost of nq sequential
    # distributed solves.  Native stays the default.
    if native:
        _q_batch = _q_batch_kernel(bs)
        for i in range(0, q_pad.shape[0], bs):
            _emit(*_q_batch(q_pad[i:i + bs], fH_R, B_rep, b_min, b_max))
            # RETAINED, per chunk, and it is not an oversight.  ``_q_batch``
            # materialises a (bs, rank, rank) fH_q batch INSIDE its own
            # executable, and this file records two separate OOMs caused by
            # exactly that array (the face constraint at ``_fourier``, and
            # the 10.1 GiB/device split that killed a 16 x A100-80GB run).
            # Without a sync, N chunks can be in flight at once and each one
            # holds its own copy of that temp buffer, so deleting this trades
            # 23-46 host round trips — second-order, by this campaign's own
            # ranking — for an N-fold transient the module has been OOM-killed
            # by twice.  It belongs to the native-glue item (survey R3) and
            # wants a VmHWM measurement, not an assertion.
            jax.block_until_ready(psi_chunks[-1])
    else:
        # ── The FFI arm's q-CHUNK ────────────────────────────────────────
        # A DIFFERENT extent from ``bs`` above, deliberately.  ``bs`` is a
        # PLACEMENT width, padded to a multiple of ndev because the native
        # path shards its q axis over ``P(('x','y'),…)``.  The FFI stack does
        # not shard its q axis at all — ``Plan.batch_in_sharding`` is
        # ``P(None,'x','y')``: the batch axis is whole and each MATRIX is
        # spread over the mesh — so padding it to ndev would buy no sharding
        # and cost real distributed solves on rows that are then discarded.
        # Hence no padding here, and a ragged last chunk instead: two
        # signatures, two compiles, and ``distrib_la.plan._SCAN_CACHE`` holds
        # both.
        #
        # THE DEFAULT IS 1, AND IT IS 1 FOR A VALUES REASON, NOT A MEMORY ONE.
        # MEASURED, this worktree, 2x2 A100 mesh, 4 ranks, cuSOLVERMp eigh,
        # rank 64, nk_co 4, 16 q, against the per-q loop this replaces:
        #     C = 1   every bundle field BIT-IDENTICAL (np.array_equal)
        #     C = 8   lam_fi / lam_all_fi / enk_full  max|d| 3.33e-16 Ry
        #             coeffs_fi 4.48e-15, psi 4.71e-14
        # The re-framing here is NOT the source: with the eigh taken out of
        # the picture, this file's Fourier body and projection body are
        # bit-identical as a per-q loop and as a ``lax.map`` at widths 1, 2
        # and 8 (probe: ``bsesetup_probe/fourier_frame_probe.py``, host mesh,
        # array_equal, with its own planted-ULP red twin).  What moves is
        # inside the FFI eigh: ``Plan.batched``'s ROUTE_SCAN hands
        # cuSOLVERMp a slice of a stack where ``Plan.__call__`` handed it a
        # standalone tile, and cusolverMpSyevd does not return the same last
        # bit for the two.  That is a distrib_la fact, not a bse_setup one.
        #
        # A width-derived default would therefore make the ANSWER a function
        # of how much memory the run had — and on this deck the cross-P pin
        # (0.740 meV, and a .dat that is byte-identical at both P) is a
        # headline artifact.  So the default reproduces the historical path
        # exactly, and C > 1 is something a deck ASKS for, in writing, with
        # the trade announced on the banner below.
        #
        # What C > 1 buys, same measurement: XLA dispatches per call
        # 53 -> 11 at C = 8 (4.8x), because each chunk is one Fourier program,
        # one batched-eigh call and one projection program instead of three
        # per q.  What it costs on a library WITHOUT a stacked entry point
        # (cuSOLVERMp, SLATE) is only that dispatch saving — the per-matrix
        # collective floor is untouched — plus the ULP above.  On ScaLAPACK,
        # which HAS one, ``Plan.batched_route`` becomes ROUTE_BACKEND_BATCHED
        # and the chunk additionally collapses to one descriptor and one
        # workspace in C++.  Residency, for a deck sizing the key: C
        # face-sharded matrices cost C·rank²·16/ndev per rank for the operand
        # and the same again for the eigenvector stack, so C ≥ ndev is a
        # whole matrix per device — which is the thing ``use_low_mem_eigh``
        # says is unaffordable.
        bs_ffi = (max(1, min(int(batch_size), nq)) if _bs_explicit else 1)
        if bs_ffi > 1:
            log(f"  *** LORRAX SANITY: wfn_fi_q_chunk={bs_ffi} on the FFI "
                f"eigh arm batches {bs_ffi} q per distributed solve.  The "
                f"per-q values move by ~1 ULP against the C=1 path (measured "
                f"3.3e-16 Ry on the eigenvalues, 4.7e-14 on psi, inside this "
                f"module's own chunk-width tolerance but NOT bit-identical), "
                f"so a byte-identical .dat against a C=1 run is not expected. "
                f"Unset the key for the bit-identical path. ***")
        n_chunks = -(-nq // bs_ffi)
        # Mesh/geometry guards already passed when the plan was built.
        log(f"  fH_q eigh: {eigh_plan.describe()}, {nq} q in {n_chunks} "
            f"chunk(s) of {bs_ffi} "
            f"[{'wfn_fi_q_chunk' if _bs_explicit else 'default ndev//2'}], "
            f"Plan.batched route={eigh_plan.batched_route}"
            + ("  [use_low_mem_eigh]" if use_low_mem_eigh else ""))

        # ── The FFI arm's TWO jits, CHUNK-shaped ─────────────────────────
        # Each is a ``lax.map`` — a scan with an empty carry, which is the
        # claim that the q axis is independent — over EXACTLY the single-q
        # body the Python loop used to call per item.  Two properties follow
        # from that spelling and both are why it is a map and not a vmap:
        #
        #  * THE ARITHMETIC PER q IS UNCHANGED, so this conversion is
        #    bit-identical by construction rather than by tolerance.  The
        #    obvious alternative — batching the Fourier einsum to
        #    ``'qk,kij->qij'`` the way ``_fourier(..., batched=True)`` does
        #    for the native arm — is a different XLA kernel with different
        #    reduction blocking, and this module's own chunk-width cell
        #    already measures that changing an eigh BATCH SIZE moves the
        #    answer by one ULP (``tests/test_bse_setup_qchunk.py``,
        #    ``CHUNK_TOL_RY``).  A loop that is only a loop must not move a
        #    value.  Both bodies are gated as exact at widths 1/2/8 by
        #    ``bsesetup_probe/fourier_frame_probe.py``; what does move at
        #    C > 1 is the FFI eigh, and the ``bs_ffi`` block above is where
        #    that is measured and defaulted around.
        #  * The chunk is ONE XLA program instead of C, and it hands
        #    ``Plan.batched`` a real STACK — which is the only place the
        #    batched surface's route decision can live
        #    (``distrib_la.Plan.batched_route``).  ScaLAPACK takes
        #    ROUTE_BACKEND_BATCHED: one descriptor and one workspace in C++
        #    for the whole chunk, a saving that lives in the library and no
        #    Python loop can reach.  cuSOLVERMp and SLATE take the cached
        #    ROUTE_SCAN, which is the same per-matrix FFI call this loop
        #    already made, minus C−1 dispatches.
        #
        # ``out_shardings`` is the plan's OWN declared operand contract, so
        # ``Plan.batched``'s ``ensure_sharding`` finds the stack already in
        # place and emits nothing (it compares (mesh, spec), not sharding
        # objects, for exactly this case — distrib_la.plan.ensure_sharding).
        def _build_fH_q_chunk():
            @partial(jax.jit, out_shardings=eigh_plan.batch_in_sharding)
            def _fH_q_chunk(q_chunk, fH_R):
                return jax.lax.map(lambda q: _fourier(q, fH_R, False),
                                   q_chunk)
            return _fH_q_chunk

        def _build_project_chunk():
            @partial(jax.jit, static_argnames=('b_min', 'b_max'))
            def _project_chunk(lam, U, B, b_min, b_max):
                return jax.lax.map(
                    lambda lu: _project(lu[0], lu[1], B, b_min, b_max),
                    (lam, U))
            return _project_chunk

        _fH_q_chunk = _cached_kernel(
            ("fH_q_chunk", _sig, eigh_plan.backend), _build_fH_q_chunk)
        _project_chunk = _cached_kernel(
            ("project_chunk", _sig), _build_project_chunk)

        t_eigh = time.time()
        _logged = 0
        for i in range(0, nq, bs_ffi):
            lam_c, U_c = eigh_plan.batched(
                _fH_q_chunk(q_all[i:i + bs_ffi], fH_R))
            _emit(*_project_chunk(lam_c, U_c, B_rep, b_min, b_max))
            # nq SERIAL distributed solves is a long, silent stretch — log the
            # rate so a stall is distinguishable from slow progress.  THE ONE
            # SYNC LEFT ON THIS ARM lives here, at the LOG cadence and not per
            # item, and it does two jobs.  It keeps the rate honest (without
            # it the number is a DISPATCH rate, which reads as "instant, then
            # a long pause in the concatenate" and hides exactly the stall the
            # line exists to expose), and it bounds the in-flight transient to
            # ~32 q of chunks — strictly less than one native chunk's worth,
            # and 1/32 of the round trips the per-q sync made.
            done = min(i + bs_ffi, nq)
            if done - _logged >= 32 or done == nq:
                jax.block_until_ready(psi_chunks[-1])
                _logged = done
                dt = time.time() - t_eigh
                log(f"    fH_q eigh {done}/{nq} q  {dt:.0f}s  "
                    f"{1e3 * dt / done:.0f} ms/q")
    lam_fi = jnp.concatenate(lam_chunks, axis=0)[:nq]
    psi_fi = jnp.concatenate(psi_chunks, axis=0)[:nq]
    lam_all_fi = jnp.concatenate(lam_all_chunks, axis=0)[:nq]
    coeffs_fi = (jnp.concatenate(c_chunks, axis=0)[:nq]
                 if return_coeffs else None)

    # ── Newton-invert lam_fi → DFT-equivalent energies ───────────────────
    # (a_f, n_f, shift) are TRACE CONSTANTS here, not arguments, so they are
    # in the key: two decks that differ only in their f-transform parameters
    # must not share this executable.  They are the three floats
    # ``f_transform_eigs`` returns, so the key is exact, not a fingerprint.
    def _build_inv():
        @jax.jit
        def _inv(lam):
            return jax.vmap(
                lambda row: newton_inv(a_f, n_f, shift, row.real))(lam)
        return _inv

    energies_fi = _cached_kernel(
        ("inv", _mesh_id, float(a_f), float(n_f), float(shift)),
        _build_inv)(lam_fi)

    # ── Reshard into the canonical (Y, X) wfn-bundle layout ──────────────
    # Matches ``common.wfn_transforms.load_centroids_band_chunked``:
    #   psi_rmu_Y:  (nk, nb, ns, n_μ)   P(None, None, None, 'y')   — n_μ on Y
    #   psi_rmuT_X: (nk, n_μ, nb, ns)   P(None, 'x', None, None)   — n_μ on X
    # Both of these shard the μ axis (the k-means centroid count), which is
    # exactly the extent nothing rounds here.  Fit, do not assume.
    out_Y = _fit(mesh_xy, P(None, None, None, 'y'), (nq, nb_fi, nspinor, n_mu),
                 "bse_setup.psi_rmu_Y")
    out_X = _fit(mesh_xy, P(None, 'x', None, None), (nq, n_mu, nb_fi, nspinor),
                 "bse_setup.psi_rmuT_X")

    def _build_bundle():
        @partial(jax.jit, out_shardings=(out_Y, out_X))
        def _make_bundle(psi):
            psi_Y = jax.lax.with_sharding_constraint(psi, out_Y)
            psi_X = jax.lax.with_sharding_constraint(
                jnp.transpose(psi, (0, 3, 1, 2)), out_X)
            return psi_Y, psi_X
        return _make_bundle

    # ``nq`` joins the key here and nowhere else: it is the only kernel whose
    # closed-over shardings (out_Y / out_X) carry the fine-grid q count.
    psi_rmu_Y, psi_rmuT_X = _cached_kernel(
        ("make_bundle", _mesh_id, int(nq), int(nb_fi), int(nspinor),
         int(n_mu)), _build_bundle)(psi_fi)
    log(f"  bundle: psi_rmu_Y={psi_rmu_Y.shape}, psi_rmuT_X={psi_rmuT_X.shape}")
    out = SimpleNamespace(
        psi_rmu_Y=psi_rmu_Y,
        psi_rmuT_X=psi_rmuT_X,
        enk_full=energies_fi,
        lam_fi=lam_fi,
        lam_all_fi=lam_all_fi,
    )
    if return_coeffs:
        out.coeffs_fi = coeffs_fi
    return out
