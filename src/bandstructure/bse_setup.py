"""BSE interpolation setup — fine-k wfn recovery via standard htransform.

Top-level entry point for the BSE-interpolation handoff. Takes the
coarse-grid htransform outputs (``ctilde``, ``B_at_mu``, ``enk_sigma``)
plus a finer uniform k-grid spec, and produces wfns at the
already-selected coarse-grid centroids on the fine grid via the
standard htransform pipeline:

    fH_R, L_R  = build_fH_R(ctilde, enk_sigma, kgrid_co,       # one program:
                            backmap=Φ^T)                      # node-pair kernel and
    L_R        = ifft_k Φ^T_k (−f(E_k)) C̃_k†                  # back-map kernel
    X_q        = Σ_R Σ_T w_ij e^{-2πi q·(R+LT)} X_R           # per-pair WS sum
    (lam, c)   = top_block_eigh(fH_q, C̃ of nearest coarse k)  # fitted f(ε), eigvecs in C̃'s frame
    ψ_n,q(r_μ) = [L_q c_n,q]_μ / (−lam_n)                    # at the centroids (exact on-grid)

``Φ^T`` are the fitted states at the ISDF centroids (the back-map targets:
the nodes themselves when the nodes are the centroids).

The resulting wfns are returned as the canonical X/Y-sharded bundle
used elsewhere in LORRAX (matching ``load_centroids_band_chunked``),
ready to drop in as a BSE interpolation input.

This module deliberately stays narrow: it calls the core htransform
routines (``build_fH_R``, ``node_pair_table``, ``newton_inv``) and
adds only the fine-grid k-list construction, the per-q top-block path, and
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

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from distrib_la import mesh_key as _mesh_key   # noqa: E402
# Every extent this module shards is an input datum: n_μ is a k-means centroid
# count, nb_fi is the caller's band window, and the q-batch is batch_size — so
# each of them divides the mesh axis only by luck.  ``bse_io`` pads its own μ
# copy (``padded_mu_extent``); this module never did, which made
# ``exciton_bands`` unrunnable at EVERY P>1 on any deck whose centroid count is
# not a multiple of the mesh axis (785 = 5·157 on the MoS2 4×4 deck: job
# 7882476 cells exb16s/exb64s).  See common/sharding_fit.py.
from common.sharding_fit import fit_sharding as _fit
from runtime.padding import padded_axis
# The (i,j)-face → q-batch move below is a layout change XLA cannot lower in
# one step; ``common.staged_reshard`` is the two-all_to_all staging that makes
# it legal.  See the block at ``fH_q(qbatch)``.
from common.staged_reshard import (
    face_to_batch_reshard as _face_to_batch_reshard,
    face_to_batch_reshard_supported as _face_to_batch_supported,
)

from .fh_interp import (
    NODE_ROUTES,
    admit_node_route,
    build_fH_R,
    coarse_start_block,
    log_top_block,
    newton_inv,
    node_pair_table,
    node_route_map,
    require_newton_converged,
    start_columns,
    top_block_eigh,
)
from common.ws_images import pair_ws_code_fourier


#: Per-process cache of this module's small epilogue jits (``newton_inv`` and
#: the bundle reshard), keyed by EVERY closed-over value: the mesh (by
#: ``distrib_la.mesh_key``, which carries device identity, not ``id``), the
#: extents and ``newton_inv``'s (a, n, shift).  The chunk kernel is not
#: here: ``fh_interp.admit_node_route`` compiles it per call at the call's
#: operands, as the admission prices it (the persistent compile cache serves a
#: repeat).
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


#: The f-shoulder tripwire's floor on ``min_k |f(ε_b)| / max|f|``, over the
#: bands ``compute_wfns_fi`` RETURNS.  Default 0.0 — refuse only when a
#: returned band's f is EXACTLY zero somewhere in the zone, which is the
#: measured failure and not a tolerance judgement.  A NEGATIVE override
#: disables the gate (announced); a positive one tightens it.
FI_FSHOULDER_TOL_DEFAULT = 0.0


def resolve_fi_fshoulder_tol(log_fn=None) -> float:
    """``LORRAX_FI_FSHOULDER_TOL`` (default 0.0) — the f-shoulder gate floor.

    Uses the canonical ``gw_config.env_float`` announce-or-refuse grammar.
    Here ``0`` is the DEFAULT, not an off switch: a band whose ``f`` is exactly
    zero at some k is absent from ``fH``, so the floor that catches it is zero.
    Turning the gate OFF needs a value no physical threshold could be
    (negative).
    """
    from gw.gw_config import env_float
    tol = env_float("LORRAX_FI_FSHOULDER_TOL", FI_FSHOULDER_TOL_DEFAULT,
                    refuse=True)
    if tol != FI_FSHOULDER_TOL_DEFAULT and log_fn is not None:
        log_fn(f"  [gate] LORRAX_FI_FSHOULDER_TOL={tol:.3e} overrides the "
               f"default {FI_FSHOULDER_TOL_DEFAULT:.1e}"
               + ("  ** THE f-SHOULDER GATE IS DISABLED — only ever to "
                  "reproduce a known-bad run **" if tol < 0.0 else ""))
    return tol


def _f_shoulder_gate(f_eps, b_min: int, b_max: int, shift: float, log_fn,
                     rank: int | None = None,
                     where: str = "compute_wfns_fi"):
    """REFUSE when a RETURNED band is invisible to ``fH`` at some k.

    THE MECHANISM, in one sentence: ``f`` is identically zero for ε ≥
    ``shift`` and ``htransform._f_params_from_energies`` sets ``shift =
    max_k ε[nb−1]``, so the top band of the htransform's own window
    contributes EXACTLY NOTHING to ``fH = Σ_n f(ε_n) c_n c_nᴴ`` at the k that
    attains that maximum, and ``f`` vanishes to second order approaching it
    — which puts a whole SHOULDER of bands below the top at a per-cent or
    less of fH's weight.  Those eigenvector slots are degenerate with fH's
    own ``(rank − nb)``-dimensional null space and ``eigh`` fills them with
    arbitrary directions out of it.  The returned ψ for such a band is not
    slightly wrong; it is a different vector.

    WHY A GATE AND NOT A DOC LINE.  This cost five lanes.  The representation
    was fine every time — ``ctilde`` orthonormality 8.9e-15 and the Galerkin
    residual 4.3e-15 on the parent where the tile null read 140 — so every
    instrument the tree already had exonerated the htransform, correctly, and
    the defect was in the REQUEST.  Measured signature
    (``tests/known_failures/2026-08-11-fifth-wall-is-the-f-transform-shoulder.md``
    §2), from the runs' own ``enk_sigma``:

        dp2628n20  nb 20, ZERO guards:  bands 16..19 read min_k |f|/max|f| =
                   0.000000, exactly zero at 3 k each; α-space overlap
                   ‖O[m,:]‖ collapses to 0.23…0.27; tile null 1.267.
        p2628n52   nb 52, ZERO guards:  bands 50, 51 zero at 8 k each.
        The bands that SURVIVE never get closer to zero than ~4e-4 of
        max|f| and come back at ‖O[m,:]‖ = 1.000000 at every k.

    So the good and the bad are separated by four decades with nothing in
    between, and a floor at exactly zero is the whole discrimination.

    THE FIX IS GUARD BANDS, not a wider tolerance: widen the htransform
    window ABOVE what you ask back (``initialize_wfns(n_guard_bands=…)``,
    ``bse.vq_interp.refit_prepare``), so the shoulder lands on bands nobody
    reads.  ``b_min``/``b_max`` index ``ctilde``'s band axis, which is the
    same axis ``f_eps`` carries and — ``f`` being monotone in ε — the same
    ascending-eigenvalue slice the caller takes out of ``fH_q``.
    """
    tol = resolve_fi_fshoulder_tol(log_fn)
    fa = np.abs(np.asarray(f_eps, dtype=np.float64))     # (nb_co, nk_co)
    fmax = float(fa.max()) if fa.size else 0.0
    if fmax <= 0.0:
        raise ValueError(
            f"{where}: max|f(ε)| over the whole htransform window is "
            "zero — every band is at or above the f-transform's shift, so fH "
            "is identically zero and its eigenvectors are pure noise.  This "
            "is an energy-array problem (a zeroed or sentinel enk_sigma), "
            "not a window one.")
    ratio = fa[b_min:b_max, :].min(axis=1) / fmax        # per RETURNED band
    n_zero_k = int((fa[b_min:b_max, :] == 0.0).sum())
    worst_b = int(np.argmin(ratio)) + b_min
    worst = float(ratio.min())
    n_guard = int(fa.shape[0]) - int(b_max)
    if log_fn is not None:
        log_fn(f"  [gate] f-shoulder over the RETURNED bands "
               f"[{b_min}, {b_max}) of {int(fa.shape[0])} "
               f"({n_guard} guard band(s) above): "
               f"min_b min_k |f(ε_b)|/max|f| = {worst:.6e} at band {worst_b} "
               f"(shift = {shift:.8f} Ry; {n_zero_k} exactly-zero (band, k) "
               f"slot(s) in the returned window)")
    if tol < 0.0:
        return worst
    if worst <= tol:
        raise ValueError(
            f"{where}: band {worst_b} of the RETURNED window "
            f"[{b_min}, {b_max}) is invisible to fH — "
            f"min_k |f(ε_{worst_b})|/max|f| = {worst:.6e} at or below the "
            f"{tol:.1e} floor, with {n_zero_k} exactly-zero (band, k) slot(s) "
            f"in the window.  f(ε) ≡ 0 for ε ≥ shift = {shift:.8f} Ry "
            f"(= max_k ε of the TOP band of ctilde's own window), so this "
            f"band contributes nothing to fH = Σ_n f(ε_n) c_n c_nᴴ at those "
            f"k; its eigenvector slot is degenerate with fH's "
            + (f"{int(rank) - int(fa.shape[0])}-dimensional "
               if rank is not None else "")
            + f"null space and eigh fills it with an arbitrary direction out "
            f"of that space. "
            f"The returned ψ for that band is a DIFFERENT VECTOR, not an "
            f"approximate one, and every pair density built from it is wrong "
            f"— measured at 1.267 on a 5.0e-02 tile-null bracket.\n"
            f"  THE FIX IS GUARD BANDS, and it is the ONLY fix: widen the "
            f"htransform window above what you ask back, so the shoulder "
            f"lands on bands nobody reads.  There are currently {n_guard}; "
            f"how many you need is the DEPTH OF THE DEAD SET printed above "
            f"({n_zero_k} exactly-zero (band, k) slots), which was four bands "
            f"on a 20-band window and two on a 52-band one.  In a deck: raise "
            f"``nband``/``ncond`` above the window you return "
            f"(``wfn_fi_max``, the BSE window, the ζ-fit window).  In code: "
            f"``htransform.initialize_wfns(n_guard_bands=…)``, which "
            f"``bse.vq_interp.refit_prepare`` drives from "
            f"``--refit-guard-bands``.  Do NOT reach for "
            f"LORRAX_FI_FSHOULDER_TOL: a negative value reproduces the "
            f"known-bad run and nothing else.")
    return worst


def resolve_conduction_window(enk_sigma, *, nb_window: int, nval_in: int,
                              n_cond: int, tol_ry: float, mode: str,
                              input_file: str, log_fn=print):
    """``(b_min, b_max, n_guard)``: the BSE conduction bands inside the fH.

    fH is built from every band of ``ctilde`` and only the sub-window
    ``[b_min, b_max)`` is returned, so a full-band ``ctilde`` gives a full-band
    fH however few conduction bands the BSE keeps, and the kept bands sit
    strictly inside it, guarded above.  A sliver window whose top edge cuts a
    near-degenerate (Kramers) pair rings 100–1000 meV off-grid.  The window is
    two-sided: a wider one asks the nodal basis to represent more states (the
    metric eps of the fit and the node count per fitted band), which the
    all-coarse transformed-energy receipt and the consumer's on-grid energy
    gate measure.

    ``nb_window`` is the band count of the htransform fH (``ctilde``);
    ``enk_sigma`` is its ``(nb, nk)`` energies, window-relative like
    ``b_min = nval_in`` and ``b_max = nval_in + n_cond``.  The window must fit (it refuses otherwise), and its edges are
    checked for a cut multiplet (``common.band_degeneracy``, report-only:
    ``b_max`` is pinned to the loader's ``n_cond``, and widening it here would
    desynchronise the conduction caches from the BSE window).  ``n_guard`` is
    the conduction bands above the selection: fewer than 4 lets the top edge
    ring off-grid; more than 16 asks the nodal basis to represent states the
    kernel never uses.  Both warn.
    """
    from common.band_degeneracy import check_band_window
    nb_window = int(nb_window)
    b_min, b_max = int(nval_in), int(nval_in) + int(n_cond)
    n_guard = nb_window - b_max
    if b_max > nb_window:
        raise ValueError(
            f"BSE conduction window [{b_min},{b_max}) exceeds the htransform "
            f"fH window ({nb_window} bands): raise nband in {input_file} to "
            f">= {b_max}, or drop --n-cond to <= {nb_window - int(nval_in)}")
    check_band_window(
        np.asarray(enk_sigma).T, b_min, b_max, tol_ry=tol_ry, mode=mode,
        where="exciton_bands htransform conduction window", log=log_fn)
    if n_guard < 4:
        log_fn(f"  [warn] only {n_guard} conduction guard band(s) above the "
               f"BSE selection — a selection boundary near a Kramers pair can "
               f"ring off-grid; widen the input's ncond/nband (>= {b_max + 4} "
               f"bands).")
    if n_guard > 16:
        log_fn(f"  [warn] htransform fH spans {nb_window} bands with {n_guard} "
               f"conduction guards above the BSE window — a LARGE interp "
               f"window is not automatically more accurate.  Check the "
               f"mandatory all-coarse transformed-energy receipt and this "
               f"driver's on-grid energy gate; keep nband just above the BSE "
               f"window unless both remain controlled.")
    log_fn(f"  full-band htransform: fH over {nb_window} bands "
           f"({int(nval_in)}v + {nb_window - int(nval_in)}c); BSE conduction "
           f"[{b_min},{b_max}) = {int(n_cond)} band(s) + {n_guard} guard(s)")
    return b_min, b_max, n_guard


def compute_wfns_fi(
    *,
    ctilde: jax.Array,
    B_at_mu,
    enk_sigma: jax.Array,
    kgrid_co: tuple[int, int, int],
    kgrid_fi=None,
    band_window_fi: tuple[int, int],
    mesh_xy: Mesh,
    a_band_index: int | None = None,
    batch_size: int | None = None,
    q_list=None,
    return_coeffs: bool = False,
    use_low_mem_eigh: bool = False,
    htransform_quality_record_fn=None,
    log_fn=None,
):
    """Recover ψ at the ISDF centroids on a finer uniform k-grid — or on an
    EXPLICIT list of arbitrary q (the arbitrary-Q generalization,
    arbitrary_q_bse.md §1/§2a: e.g. the shifted grid {k + Q}).

    Args:
        ctilde:    (nk_co, nb, rank) orthonormal nodal coefficients C̃_k.
                   ``build_fH_R`` consumes paired rank-sharded views, so this
                   may arrive rank-sharded or replicated.
        B_at_mu:   :class:`isdf.galerkin.NodalInterp` — the nodal values,
                   the states at the centroids (the back-map targets) and
                   the geometry; ``shape`` = (rank, ns, n_μ).
        enk_sigma: (nb, nk_co) DFT band energies in Ry.
        kgrid_co:  (nkx, nky, nkz) coarse uniform k-grid.
        kgrid_fi:  (nkx_fi, nky_fi, nkz_fi) fine k-grid; tuple/list/string.
                   Mutually exclusive with ``q_list``.
        band_window_fi: (b_min, b_max) — sub-window of the htransform band
                   axis to RETURN on the fine grid (0-based, exclusive end).
                   fH is ALWAYS built from every band in ``ctilde``; this
                   window only selects which eigenpairs are returned, so the
                   returned bands sit interior to fH, guarded above.  The
                   top-block solve converges exactly these returned bands
                   (the guards above are not read; the topmost ones sit at
                   f = 0 beside the null space and need not converge).
        mesh_xy:   ('x','y') device mesh.
        a_band_index: optional band index for the f-transform 'a' parameter
                   (defaults to the top of the htransform window).
        batch_size: q-points per chunk (deck key ``wfn_fi_q_chunk``);
                   ``None`` / ``<= 0`` (the DEFAULT) means N_q_co =
                   prod(kgrid_co): one chunk then costs, on the q-batched
                   route, the bytes of fH_R itself, which the run has already
                   afforded.  Rounded UP to a multiple of the device count so
                   the q axis splits.
        q_list:    optional (nq, 3) fractional q — evaluate at exactly these
                   points instead of a uniform grid (wrapped to (−0.5, 0.5]).
        return_coeffs: also return the eigenvector coefficients
                   ``.coeffs_fi`` (nk_fi, rank, nb_fi), q-sharded.
        use_low_mem_eigh: the deck's ``linalg = distributed``: no rank may
                   hold a whole (rank, rank) matrix, so only the face route
                   is admitted.  Otherwise ``lxkit.memory`` admits the
                   q-batched route (whole matrices of a rank's own q rows,
                   one batched solve) where its compiled program fits the
                   room, else the face route (one q at a time, every matrix
                   tiled over the mesh) — the mesh backup of route
                   completeness (GATE nmu2_no_backup).
        log_fn:    optional logger.

    Returns:
        SimpleNamespace bundle (matching ``file_io.tagged_arrays`` convention):
            .psi_rmu_Y:  (nk_fi, nb_fi, ns, n_μ)  P(None, None, None, 'y')
            .psi_rmuT_X: (nk_fi, n_μ, nb_fi, ns)  P(None, 'x', None, None)
            .enk_full:   (nk_fi, nb_fi)           recovered DFT energies (Ry)
                                                  via ``newton_inv`` of fH_q eigvals
            .lam_fi:     (nk_fi, nb_fi)           raw fH_q eigenvalues (= f(ε_n,q))
            .coeffs_fi:  (nk_fi, rank, nb_fi)     only when ``return_coeffs``
            .q_frac:     (nk_fi, 3) host float64  fractional q of each row,
                                                  wrapped to (−0.5, 0.5]

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
    if nb_fi <= 0 or b_min < 0 or b_max > nb_co:
        raise ValueError(
            f"band_window_fi {band_window_fi} not a valid sub-range of the "
            f"{nb_co} fitted bands")

    nk_co = int(kgrid_co[0]) * int(kgrid_co[1]) * int(kgrid_co[2])
    if int(ctilde.shape[0]) != nk_co:
        raise ValueError(
            f"kgrid_co {tuple(kgrid_co)} implies N_q_co={nk_co} but ctilde has "
            f"{int(ctilde.shape[0])} k-points; the q-chunk default and "
            f"build_fH_R's ifft-reshape both read this extent.")
    if batch_size is None or int(batch_size) <= 0:
        batch_size, _bs_origin = nk_co, f"default N_q_co={nk_co}"
    else:
        batch_size = int(batch_size)
        _bs_origin = f"wfn_fi_q_chunk={batch_size}"
    # The q-batch width is a PLACEMENT extent: a width that does not divide
    # the mesh drops the q sharding and replicates the batch (measured 8x
    # slower at P=64), so it is rounded UP to a multiple of ndev.  Padded q
    # rows are zeros, solved and discarded by the [:nq] slice.
    bs = padded_axis(
        batch_size, mesh_xy, name="BSE q batch",
        spec=P(("x", "y"), None, None), axis=0).carrier
    from runtime.padding import mesh_divisor
    _ndev = mesh_divisor(mesh_xy)
    _batch_note = (f"q-chunk={bs} [{_bs_origin}]" if bs == batch_size else
                   f"q-chunk={bs} ({_bs_origin}, padded to a multiple of "
                   f"ndev={_ndev} so the q axis splits)")

    if q_list is None:
        kgrid_fi = _parse_kgrid_fi(kgrid_fi)
        nq = kgrid_fi[0] * kgrid_fi[1] * kgrid_fi[2]
        log(f"  bse_setup: {kgrid_co} → {kgrid_fi} ({nq} q-pts), "
            f"bands [{b_min}, {b_max}) of {nb_co} fitted, rank {rank}, "
            f"{_batch_note}")
    else:
        q_list = np.asarray(q_list, dtype=np.float64).reshape(-1, 3)
        nq = q_list.shape[0]
        log(f"  bse_setup: {kgrid_co} → explicit q-list ({nq} q-pts), "
            f"bands [{b_min}, {b_max}) of {nb_co} fitted, rank {rank}, "
            f"{_batch_note}")

    # ── fH_R and the back-map L_R: one admitted program ───────────────────
    # Both stay SHARDED P(None, 'x', 'y'): the q-Fourier sum contracts ONLY
    # over R, so every device builds its own tile of fH_q / L_q with no
    # communication.  (A replicated fH_R was nk_co·rank²·16 B on EVERY device
    # and a host gather of the same size per process: the reason a converged
    # k-grid could not run the htransform, gw_converged_12x12_80ry_2026-07-21
    # §5.)
    fH_R, (a_f, n_f, shift), _f_eps, _active_R, L_R = build_fH_R(
        ctilde, enk_sigma, kgrid_co, mesh_xy,
        a_band_index=a_band_index, log_fn=log,
        quality_record_fn=htransform_quality_record_fn,
        backmap=B_at_mu.target_values)
    _f_shoulder_gate(_f_eps, b_min, b_max, shift, log, rank=rank)
    ws_table = node_pair_table(B_at_mu, kgrid_co, mesh_xy)
    ws_code, ws_target = ws_table["code"], ws_table["target_code"]
    rep = NamedSharding(mesh_xy, P())
    columns = jax.device_put(jnp.asarray(start_columns(rank, nb_co)), rep)

    # ── q-points (uniform fine grid or explicit list), padded to batch ───
    if q_list is None:
        q_all = _uniform_kgrid_frac(kgrid_fi)
    else:
        q_all = jnp.asarray((q_list + 0.5) % 1.0 - 0.5)
    n_pad = padded_axis(nq, bs, name="BSE setup q batch").pad
    q_pad = (jnp.concatenate([q_all, jnp.zeros((n_pad, 3), dtype=q_all.dtype)])
             if n_pad else q_all)
    q_pad = jax.device_put(q_pad, rep)
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])

    # ── The (i,j)-face → q-batch reshard of the q-batched route, STAGED ──
    # Built HERE, outside every jit: the factory issues a collective
    # (``warm_mesh_cliques``) and so runs synchronously on every rank.  A
    # single ``with_sharding_constraint`` {1,px,py} -> {ndev,1,1} lowers as
    # replicate-then-repartition, the WHOLE (bs, rank, rank) batch on every
    # device ("Involuntary full rematerialization", job 7882974); the staged
    # pair of single-axis all_to_alls keeps every intermediate one shard
    # (job 7883154: P=64 231 -> 146 s).  ``bs`` is padded to ndev above.
    _stage_q = {}
    for extent in {rank, int(L_R.shape[1])}:
        _stage_q[extent] = (_face_to_batch_reshard(mesh_xy, axes=('x', 'y'))
                            if _face_to_batch_supported(
                                mesh_xy, (bs, extent, rank), axes=('x', 'y'))
                            else None)

    face_pin = lambda x: jax.lax.with_sharding_constraint(
        x, _fit(mesh_xy, P(None, 'x', 'y'), x.shape, "bse_setup.X_q(face)"))

    def _fourier(q, X_R, code, route, hermitian=True):
        """X_q = Σ_R Σ_T w_ij e^{-2πi q·(R+LT)} X_R (per-pair WS images) for
        a q batch: face tiles, local over R; on the q-batched route then the
        staged face→q exchange (each rank: whole matrices of its q rows),
        after which the Hermitian transpose is local; on the face route the
        batch (one q) stays tiled and the transpose is a face collective."""
        X_q = face_pin(pair_ws_code_fourier(X_R, ws_table, q, code=code,
                                            pin=face_pin))
        if route == "q-batched":
            stage = _stage_q[int(X_q.shape[1])]
            X_q = (stage(X_q) if stage is not None else
                   jax.lax.with_sharding_constraint(X_q, _fit(
                       mesh_xy, P(('x', 'y'), None, None), X_q.shape,
                       "bse_setup.X_q(qbatch)")))
        if not hermitian:
            return X_q
        out = 0.5 * (X_q + jnp.swapaxes(X_q, -1, -2).conj())
        return out if route == "q-batched" else face_pin(out)

    def _project(lam, U, L, b_min, b_max):
        """Band-window slice + ψ_n,q(r_μ) = [L_q c_n,q]_μ / (−lam_n).

        ``L`` is the back-map ``L_q`` (t_carrier, rank): exact on-grid, and
        the division only touches returned window bands, which the
        f-shoulder gate keeps away from ``f = 0``.
        """
        c = U[..., b_min:b_max]                                # (..., rank, nb_fi)
        nodal = (L @ c) / (-lam[..., None, b_min:b_max])       # (..., t, nb_fi)
        psi = jnp.moveaxis(nodal[..., :nspinor * n_mu, :].reshape(
            nodal.shape[:-2] + (nspinor, n_mu, nodal.shape[-1])), -1, -3)
        return lam[..., b_min:b_max], psi, c

    out_shardings = (
        _fit(mesh_xy, P(('x', 'y'), None), (bs, nb_fi), "bse_setup.lam"),
        _fit(mesh_xy, P(('x', 'y'), None, None, None),
             (bs, nb_fi, nspinor, n_mu), "bse_setup.psi"),
        _fit(mesh_xy, P(('x', 'y'), None, None), (bs, rank, nb_fi),
             "bse_setup.coeffs"),
        rep, rep)
    q_batch_x0 = _fit(mesh_xy, P(('x', 'y'), None, None),
                      (bs, rank, int(columns.shape[1]) + nb_co),
                      "bse_setup.start_block")

    def _program(route):
        """One chunk on ``route``: Fourier sum → top-block solve over every
        fitted band → back-map projection, ONE fused jit (the full
        (bs, rank, rank) eigenvector batch is never an output; it killed a
        16 x A100-80GB run when it was, 10.1 GiB/device at rank 4452)."""
        @partial(jax.jit, out_shardings=out_shardings)
        def _chunk(q_batch, fH_R, L_R, code, target_code, ctilde, columns):
            def one(q):
                x0 = coarse_start_block(ctilde, kgrid_co, q, columns)
                if route == "q-batched":
                    x0 = jax.lax.with_sharding_constraint(x0, q_batch_x0)
                lam, U, steps, res = top_block_eigh(
                    _fourier(q, fH_R, code, route), x0, (b_min, b_max))
                L = _fourier(q, L_R, target_code, route, hermitian=False)
                return _project(lam, U, L, b_min, b_max) + (steps, res)
            return node_route_map(route, one, q_batch)
        return _chunk

    route, _chunk = admit_node_route(
        f"htransform fine-q fH(q) top block ({rank}^2, q-chunk {bs})",
        _program, (q_pad[:bs], fH_R, L_R, ws_code, ws_target, ctilde,
                   columns), rank=rank, mesh_xy=mesh_xy,
        routes=("face",) if use_low_mem_eigh else NODE_ROUTES)
    log(f"  fH_q route {route}: {nq} q in {q_pad.shape[0] // bs} chunk(s) of "
        f"{bs}" + (f" ({bs // _ndev} whole ({rank}, {rank}) matrices per rank)"
                   if route == "q-batched" else
                   f", one q at a time, ({rank}, {rank}) tiled over {px}x{py}")
        + ("  [linalg = distributed]" if use_low_mem_eigh else ""))

    lam_chunks, psi_chunks, c_chunks, block_stats = [], [], [], []
    t_loop, logged = time.time(), 0
    for i in range(0, q_pad.shape[0], bs):
        lam_s, psi_s, c_s, steps, res = _chunk(
            q_pad[i:i + bs], fH_R, L_R, ws_code, ws_target, ctilde, columns)
        block_stats.append((steps, res))
        lam_chunks.append(lam_s)
        psi_chunks.append(psi_s)
        if return_coeffs:
            c_chunks.append(c_s)
        # One sync per chunk: each chunk's executable materialises a
        # (bs, rank, rank) fH_q (q-batched) inside it, and N chunks in flight
        # would hold N such transients (two OOMs in this module's history).
        jax.block_until_ready(psi_s)
        done = min(i + bs, nq)
        if done - logged >= max(32, bs) or done >= nq:
            logged = done
            dt = time.time() - t_loop
            log(f"    fH_q top-block {done}/{nq} q  {dt:.0f}s  "
                f"{1e3 * dt / max(done, 1):.0f} ms/q")
    log_top_block(log, f"fine-grid fH_q, returned bands [{b_min}, {b_max})",
                  nb_co,
                  [np.asarray(s) for s, _ in block_stats],
                  [np.asarray(r) for _, r in block_stats])
    del fH_R, L_R
    lam_fi = jnp.concatenate(lam_chunks, axis=0)[:nq]
    psi_fi = jnp.concatenate(psi_chunks, axis=0)[:nq]
    coeffs_fi = (jnp.concatenate(c_chunks, axis=0)[:nq]
                 if return_coeffs else None)
    _mesh_id = _mesh_key(mesh_xy)

    # ── Newton-invert lam_fi → DFT-equivalent energies ───────────────────
    # (a_f, n_f, shift) are TRACE CONSTANTS here, not arguments, so they are
    # in the key: two decks that differ only in their f-transform parameters
    # must not share this executable.
    def _build_inv():
        @jax.jit
        def _inv(lam):
            return newton_inv(a_f, n_f, shift, lam.real)
        return _inv

    energies_fi, inverse_residual = _cached_kernel(
        ("inv", _mesh_id, float(a_f), float(n_f), float(shift)),
        _build_inv)(lam_fi)
    require_newton_converged(
        float(inverse_residual), where="compute_wfns_fi")

    # ── Reshard into the canonical (Y, X) wfn-bundle layout ──────────────
    # Matches ``common.wfn_transforms.load_centroids_band_chunked``:
    #   psi_rmu_Y:  (nk, nb, ns, n_μ)   P(None, None, None, 'y')   — n_μ on Y
    #   psi_rmuT_X: (nk, n_μ, nb, ns)   P(None, 'x', None, None)   — n_μ on X
    # Both shard the μ axis (the k-means centroid count), which is exactly
    # the extent nothing rounds here.  Fit, do not assume.
    out_Y = _fit(mesh_xy, P(None, None, None, 'y'),
                 (nq, nb_fi, nspinor, n_mu),
                 "bse_setup.psi_rmu_Y")
    out_X = _fit(mesh_xy, P(None, 'x', None, None),
                 (nq, n_mu, nb_fi, nspinor),
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
    )
    if return_coeffs:
        out.coeffs_fi = coeffs_fi
    out.q_frac = np.asarray(q_all, dtype=np.float64)
    return out
