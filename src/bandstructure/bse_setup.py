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

import time
from types import SimpleNamespace
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ffi.linalg import plan as linalg_plan

from .htransform import build_fH_R, build_R_grid_np, newton_inv


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
    batch_size: int = 32,
    q_list=None,
    return_coeffs: bool = False,
    eigh_backend: str = "auto",
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
        batch_size: q-points per fH_q batch on the NATIVE eigh path (≥1 jit
                   compile reuse).  Ignored by the FFI backends, which
                   decompose one q at a time by construction.
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
                   ``cusolvermp|slate`` route ONE (rank, rank) tile at a time
                   through the distributed-linalg FFI.  See the eigh comment
                   in the batch loop below for which regime is which, and
                   ``ffi.linalg.LinalgPlan`` for the backends.
        log_fn:    optional logger.

    Returns:
        SimpleNamespace bundle (matching ``file_io.tagged_arrays`` convention):
            .psi_rmu_Y:  (nk_fi, nb_fi, ns, n_μ)  P(None, None, None, 'y')
            .psi_rmuT_X: (nk_fi, n_μ, nb_fi, ns)  P(None, 'x', None, None)
            .enk_full:   (nk_fi, nb_fi)           recovered DFT energies (Ry)
                                                  via ``newton_inv`` of fH_q eigvals
            .lam_fi:     (nk_fi, nb_fi)           raw fH_q eigenvalues (= f(ε_n,q))
                                                  kept for diagnostics
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

    if q_list is None:
        kgrid_fi = _parse_kgrid_fi(kgrid_fi)
        nq = kgrid_fi[0] * kgrid_fi[1] * kgrid_fi[2]
        log(f"  bse_setup: {kgrid_co} → {kgrid_fi} ({nq} q-pts), "
            f"bands [{b_min}, {b_max}) of {rank}, batch={batch_size}")
    else:
        q_list = np.asarray(q_list, dtype=np.float64).reshape(-1, 3)
        nq = q_list.shape[0]
        log(f"  bse_setup: {kgrid_co} → explicit q-list ({nq} q-pts), "
            f"bands [{b_min}, {b_max}) of {rank}, batch={batch_size}")

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
    n_pad = (-nq) % batch_size
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
    # before any q is solved.  See ffi.linalg.resolve for the guard order.
    try:
        eigh_plan = linalg_plan("eigh", mesh_xy, backend=eigh_backend, n=rank)
    except ValueError as exc:
        if "divisible" in str(exc):
            raise ValueError(
                f"{exc}  ``streaming_galerkin_solve`` mesh-aligns the "
                f"retained SVD rank to lcm(px, py); a ctilde built on a "
                f"different mesh has to be refit on this one.") from None
        raise
    native = eigh_plan.is_native
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])

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
        MoS2 12×12).  Costs batch_size/ndev WHOLE matrices per device.

        Otherwise q is (3,) and the single matrix is left (i, j)-sharded
        ``P('x','y')`` — never whole on any device, rank²·16/ndev instead.
        That is the FFI eigh's input layout; its hermitizing transpose IS a
        collective (rank²·16 B all-to-all, 22 MB/device at rank 4716 on 16
        devices), the price of not materialising the matrix.
        """
        if not batched:
            fH_q = jnp.einsum('k,kij->ij', jnp.exp(-2j * jnp.pi * (R_grid @ q)),
                              fH_R)
            grid = NamedSharding(mesh_xy, P('x', 'y'))
            fH_q = jax.lax.with_sharding_constraint(fH_q, grid)
            return jax.lax.with_sharding_constraint(
                0.5 * (fH_q + jnp.conj(fH_q).T), grid)
        q = jax.lax.with_sharding_constraint(
            q, NamedSharding(mesh_xy, P(('x', 'y'), None)))
        phase = jnp.exp(-2j * jnp.pi * (q @ R_grid.T))                 # (bs, nk_co)
        # fH_R is (i, j)-sharded and the sum runs over R only → local.  Pin the
        # contraction OUTPUT to that same (i, j) layout: left free, XLA
        # materialises the whole (bs, rank, rank) batch on every device before
        # the reshard below (measured: 9 batches × 11.4 GiB at nb=36 / rank
        # 4716 — the OOM that made the wide windows unmeasurable).
        fH_q = jnp.einsum('qk,kij->qij', phase, fH_R)
        fH_q = jax.lax.with_sharding_constraint(
            fH_q, NamedSharding(mesh_xy, P(None, 'x', 'y')))
        # Reshard (i,j)→q FIRST, then hermitize: on the q-sharded layout each
        # device owns whole matrices, so the transpose is local.  Doing it the
        # other way round costs a second all-to-all for ``swapaxes``.
        fH_q = jax.lax.with_sharding_constraint(
            fH_q, NamedSharding(mesh_xy, P(('x', 'y'), None, None)))
        return 0.5 * (fH_q + jnp.swapaxes(fH_q, -1, -2).conj())

    def _project(lam, U, B, b_min, b_max):
        """Band-window slice + ψ_n,q(r_μ) = Σ_α c_n,q[α] B_at_μ[α, s, μ].

        Leading-axis agnostic (``...``): the native path hands it a whole
        q-batch, the FFI path one q.  ONE reconstruction for both backends.
        """
        c = U[..., b_min:b_max]                                # (..., rank, nb_fi)
        psi = jnp.einsum('...an,asm->...nsm', c, B)            # (..., nb_fi, ns, n_μ)
        return lam[..., b_min:b_max], psi, c

    @partial(jax.jit, static_argnames=('b_min', 'b_max'))
    def _q_batch(q_batch, fH_R, B, b_min, b_max):
        """Native: Fourier sum → batched eigh → projection, ONE fused jit."""
        lam, U = jnp.linalg.eigh(_fourier(q_batch, fH_R, True))
        return _project(lam, U, B, b_min, b_max)

    @jax.jit
    def _fH_q_one(q, fH_R):
        return _fourier(q, fH_R, False)

    @partial(jax.jit, static_argnames=('b_min', 'b_max'))
    def _project_one(lam, U, B, b_min, b_max):
        return _project(lam, U, B, b_min, b_max)

    # Backend choice = parallel-over-q vs parallel-within-matrix.  The native
    # path eigh-es ``batch_size/ndev`` WHOLE (rank, rank) matrices per device
    # concurrently; the FFI path spreads ONE matrix over the whole mesh and
    # walks q serially.  At rank 4716 that is 356 MB/matrix on one device vs
    # 22 MB/device on 16 GPUs — so the FFI backends are what make a window too
    # wide for the batched path runnable at all, at the cost of nq sequential
    # distributed solves.  Native stays the default.
    if not native:
        # Mesh/geometry guards already passed when the plan was built.
        log(f"  fH_q eigh: {eigh_plan.describe()}, {nq} q serially")

    lam_chunks, psi_chunks, c_chunks = [], [], []

    def _emit(lam_s, psi_s, c_s):
        lam_chunks.append(lam_s if lam_s.ndim == 2 else lam_s[None])
        psi_chunks.append(psi_s if psi_s.ndim == 4 else psi_s[None])
        if return_coeffs:
            c_chunks.append(c_s if c_s.ndim == 3 else c_s[None])
        jax.block_until_ready(psi_chunks[-1])

    if native:
        for i in range(0, q_pad.shape[0], batch_size):
            _emit(*_q_batch(q_pad[i:i + batch_size], fH_R, B_rep, b_min, b_max))
    else:
        t_eigh = time.time()
        for i in range(nq):        # no batch padding: one q, one solve
            lam_q, U_q = eigh_plan(_fH_q_one(q_pad[i], fH_R))
            _emit(*_project_one(lam_q, U_q, B_rep, b_min, b_max))
            # nq SERIAL distributed solves is a long, silent stretch — log the
            # rate so a stall is distinguishable from slow progress.
            if (i + 1) % 32 == 0 or i + 1 == nq:
                dt = time.time() - t_eigh
                log(f"    fH_q eigh {i + 1}/{nq} q  {dt:.0f}s  "
                    f"{1e3 * dt / (i + 1):.0f} ms/q")
    lam_fi = jnp.concatenate(lam_chunks, axis=0)[:nq]
    psi_fi = jnp.concatenate(psi_chunks, axis=0)[:nq]
    coeffs_fi = (jnp.concatenate(c_chunks, axis=0)[:nq]
                 if return_coeffs else None)

    # ── Newton-invert lam_fi → DFT-equivalent energies ───────────────────
    @jax.jit
    def _inv(lam):
        return jax.vmap(lambda row: newton_inv(a_f, n_f, shift, row.real))(lam)
    energies_fi = _inv(lam_fi)

    # ── Reshard into the canonical (Y, X) wfn-bundle layout ──────────────
    # Matches ``common.wfn_transforms.load_centroids_band_chunked``:
    #   psi_rmu_Y:  (nk, nb, ns, n_μ)   P(None, None, None, 'y')   — n_μ on Y
    #   psi_rmuT_X: (nk, n_μ, nb, ns)   P(None, 'x', None, None)   — n_μ on X
    out_Y = NamedSharding(mesh_xy, P(None, None, None, 'y'))
    out_X = NamedSharding(mesh_xy, P(None, 'x', None, None))

    @partial(jax.jit, out_shardings=(out_Y, out_X))
    def _make_bundle(psi):
        psi_Y = jax.lax.with_sharding_constraint(psi, out_Y)
        psi_X = jax.lax.with_sharding_constraint(
            jnp.transpose(psi, (0, 3, 1, 2)), out_X)
        return psi_Y, psi_X

    psi_rmu_Y, psi_rmuT_X = _make_bundle(psi_fi)
    log(f"  bundle: psi_rmu_Y={psi_rmu_Y.shape}, psi_rmuT_X={psi_rmuT_X.shape}")
    out = SimpleNamespace(
        psi_rmu_Y=psi_rmu_Y,
        psi_rmuT_X=psi_rmuT_X,
        enk_full=energies_fi,
        lam_fi=lam_fi,
    )
    if return_coeffs:
        out.coeffs_fi = coeffs_fi
    return out
