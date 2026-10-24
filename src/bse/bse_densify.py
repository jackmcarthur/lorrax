"""Coarse k-grid to fine: the exact interpolation, and the one densifier.

AUTHORITY RULE — coarse to fine is a PER-PAIR WIGNER–SEITZ EMBEDDING IN REAL
SPACE.  Each centroid pair's coarse R classes go to their images of minimal
|R + L T + r_μ − r_ν| on the finer R lattice (``common.ws_images``), which
evaluates a band-limited trigonometric polynomial of the coarse samples on ANY
finer uniform grid, covariant under every operation that maps the centroid set
onto itself.  When the grids nest it passes through every coarse sample; for a
non-divisor target (for example 8→12) it passes through the subset of points
the two grids share.  When the requested grid equals the coarse one the bundle
is returned untouched, byte for byte.
There is exactly one sharded implementation of that densification
(``make_w_densifier``), the (μ, ν) sharding survives it end to end, and no step
materialises a replicated N_μ²-class array.

WHAT THE INTERPOLANT IS ALLOWED TO SEE.  Only the smooth BODY.  W's Γ head is a
Kronecker delta, and the trigonometric interpolant of a delta is a Dirichlet
kernel — it smears a fraction of the head's ~10³ meV prefactor onto fine q that
should carry none of it, and it cannot produce either the bulk 1/q² rise or
the slab 1/|q| cusp inside the coarse Γ cell at all.  So the loader DEFERS the
rank-1 whead injection whenever a
densification is pending, this module densifies the head-excluded body, and
``build_w_head_channel`` re-attaches an analytic per-fine-q head from the one
ratified integrand (``gw.head_densify``).

What is grid-dependent and what is not is the module's other rule.  ψ and the
quasiparticle energies are re-interpolated through one htransform; ``W``'s
direct term is densified; the q=0 exchange BODY is built from the centroids and
the G-sphere alone and is therefore k-grid INVARIANT, so it is carried through
unchanged.  Only the rank-1 head scalar depends on the grid, and whether to
rebuild the EXCHANGE head is the deck's ``head_minibz_average`` key — read here
the same way every other consumer of that key in the tree reads it, default
off.  This path once forced that rebuild on regardless, which quietly replaced
the deck's disk body with a model reconstruction of itself.
"""
from __future__ import annotations

import math
import os
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import make_kfft_kminor, make_local_kfft_kminor
from common.ws_images import lattice_bohr, pair_ws_embed, pair_ws_images

from .bse_preconditioner import compute_pair_amplitude
from .bse_window import PAD_EPS_GUARD_RY
from common.log import get_logger, message

log = get_logger(__name__)


def pad_W_R_to_grid(W_R_coarse: jax.Array, fine_grid: tuple[int, int, int],
                    *, embedding: dict | None, sharding=None,
                    pos_b=None) -> jax.Array:
    """Embed a coarse-grid real-space ``W_R`` into a finer R lattice.

    Enables a CHEAP coarse-grid W (e.g. GW/W on 6×6) to drive the BSE direct
    term on a FINE exciton sampling (e.g. 12×12).  The tile
    ``W_{μν}(R)`` couples ``r_μ + R`` with ``r_ν`` (the full-Bloch gauge of
    the ISDF centroids), so each centroid pair keeps the images of minimal
    ``|R + L T + r_μ − r_ν|`` with ``1/N_deg`` (``common.ws_images``,
    Wannier90's ``use_ws_distance``); a fine ``fftn`` of the result is the
    covariant trigonometric interpolant of the coarse ``W(k)``, exact at
    every k common to both grids.

    Parameters
    ----------
    W_R_coarse : (n_mu, n_nu, ncx, ncy, ncz)
        ``ifftn(W_q_coarse, axes=last-3, norm='ortho')``; μ and ν share the
        centroid positions of ``embedding['pos']``.
    fine_grid : (nfx, nfy, nfz), each at least the coarse extent.
    embedding : the ``common.ws_images.pair_ws_images`` table of the coarse
        grid (:func:`w_pair_embedding`); may be None only when ``fine_grid``
        equals the coarse grid.
    sharding : optional ``NamedSharding`` of the (μ, ν, R) carrier (inside a
        jit: pins the scan carry and the per-class weights to W's layout).
    pos_b : optional (n_nu, 3) column positions
        A rectangular block's columns; rows take ``embedding['pos']``.
        Omit for the full square tensor on one centroid set.

    Returns
    -------
    W_R_fine : (n_mu, n_nu, nfx, nfy, nfz), scaled by ``√(∏nf/∏nc)`` so the
        ortho ``fftn`` reproduces the coarse samples.  ``fine == coarse``
        returns ``W_R_coarse`` unchanged (byte-identical no-op, any shape).
    """
    coarse = tuple(int(s) for s in W_R_coarse.shape[-3:])
    fine = tuple(int(s) for s in fine_grid)
    if fine == coarse:
        return W_R_coarse                         # exact no-op (fast path)
    if W_R_coarse.ndim != 5:
        raise ValueError(
            f"pad_W_R_to_grid: expected (mu, nu, kx, ky, kz), got shape "
            f"{tuple(W_R_coarse.shape)}")
    if embedding is None or tuple(embedding["kgrid"]) != coarse:
        raise ValueError(
            f"pad_W_R_to_grid: densifying {coarse} -> {fine} needs the "
            f"per-pair Wigner-Seitz table of exactly this coarse grid "
            f"(w_pair_embedding); got "
            f"{None if embedding is None else tuple(embedding['kgrid'])}.")
    pin = ((lambda x: jax.lax.with_sharding_constraint(x, sharding))
           if sharding is not None else None)
    out = pair_ws_embed(W_R_coarse, embedding, fine, pos_b=pos_b, pin=pin)
    scale = jnp.sqrt(jnp.asarray(np.prod(fine) / np.prod(coarse),
                                 dtype=W_R_coarse.real.dtype))
    out = out * scale
    return out if pin is None else pin(out)


def w_tile_centroid_frac(wfn, centroids_file, keep_idx, n_mu: int, *,
                         restart_file, log_fn=log.debug) -> np.ndarray:
    """``(n_mu, 3)`` positions ``r_μ = idx/N ∈ [0,1)`` of a W tile's μ axis.

    The centroid table's FFT points in tile order (the full-Bloch gauge of
    every μ-indexed ISDF tile: ``W_μν(R)`` couples ``r_μ + R`` with ``r_ν``):
    the ordered child rows ``keep_idx`` of a downfolded bundle, else every row
    (``bse.exciton_bands.resolve_isdf_basis`` resolves both).  Their index
    digest must match the ``centroids_charge_md5`` receipt ``restart_file``
    carries (GW and the downfold stamp it over the tile's own rows), so nothing
    positioned by them can sit on another run's centroids; a restart without a
    receipt is not checked, and ``log_fn`` says so in one line.  The table must
    have exactly the tile's logical ``n_mu`` rows.  The ONE source of these
    positions for the W densifier (:func:`w_pair_embedding`), the c1 head
    channel's gauge phase (``gw.head_densify.attach_head_channel``) and the
    srlr model rows (:func:`_srlr_terms`); a run computes them once and hands
    the same array to all three.
    """
    import h5py
    from file_io.centroids import load_centroids
    from file_io.wfn_basis import centroid_table_md5
    grid = np.asarray(tuple(int(x) for x in wfn.fft_grid), dtype=np.int64)
    idx = np.asarray(load_centroids(centroids_file, tuple(grid))[1],
                     dtype=np.int64)
    if keep_idx is not None:
        idx = idx[np.asarray(keep_idx, dtype=np.int64)]
    if idx.shape[0] != int(n_mu):
        raise ValueError(
            f"GATE w_pair_centroids: {centroids_file} gives {idx.shape[0]} "
            f"centroid rows for a W tile of {int(n_mu)} rows; the W "
            f"densifier would embed the tile on the wrong positions.")
    with h5py.File(restart_file, "r") as f:
        receipt = f.attrs.get("centroids_charge_md5")
    if receipt is not None:
        receipt = receipt.decode() if isinstance(receipt, bytes) else str(receipt)
        if receipt != centroid_table_md5(idx):
            raise ValueError(
                f"GATE w_pair_centroids: the centroid rows of "
                f"{centroids_file} are not the μ axis {restart_file} was "
                f"built on (index digest {centroid_table_md5(idx)} vs "
                f"receipt {receipt}); the W densifier would embed the tile "
                f"on the wrong positions.")
    else:
        log_fn(f"  w_tile_centroid_frac: {restart_file} carries no "
               f"centroids_charge_md5 receipt; the centroid rows of "
               f"{centroids_file} are not checked against the W tile.")
    return idx / grid[None, :]


def w_pair_embedding(coarse_grid, wfn, centroid_frac, n_mu_carrier: int,
                     sys_dim: int) -> dict:
    """The per-pair Wigner–Seitz table of a W tile's μ axis.

    ``centroid_frac`` is :func:`w_tile_centroid_frac` (FFT points ``idx/N`` in
    tile order, checked against the restart's ``centroids_charge_md5``
    receipt), computed once per run by the caller.  Carrier pad rows up to
    ``n_mu_carrier`` take the first centroid's position (their tiles are zero).
    """
    return pair_ws_images(coarse_grid, lattice_bohr(wfn),
                          np.asarray(centroid_frac, dtype=np.float64),
                          int(sys_dim), n_carrier=int(n_mu_carrier))


def make_w_densifier(
    mesh_xy: Mesh,
    w_spec: P,
    fine_grid: tuple[int, int, int],
    *,
    output: str = "k",
    embedding: dict | None = None,
):
    """THE coarse→fine W densifier — the ONE sharded implementation.

    Returns a jitted ``fn(W_q_coarse) -> W_fine`` that takes a coarse-grid
    ``W_q`` of shape ``(μ, ν, ncx, ncy, ncz)`` sharded as ``w_spec`` (the k
    axes must be replicated in ``w_spec``; only μ/ν may be sharded) and
    produces the fine-grid W via  ifftn(k→R) → per-pair Wigner–Seitz
    embedding into the fine R lattice (:func:`pad_W_R_to_grid` with
    ``embedding`` from :func:`w_pair_embedding`; covariant trig
    interpolation) → optionally fftn(R→k).  ``embedding`` may be None only
    when ``fine_grid == coarse`` (a plain inverse for ``output='R'``;
    ``output='k'`` returns its input unchanged):

      * ``output='R'`` → ``W_R_fine`` — what the exciton_bands
        ``--w-coarse-grid`` path feeds the matvec directly;
      * ``output='k'`` → ``W_q_fine`` — what the ``kgrid_fine`` bundle
        densification stores, so each solver's own ``ifftn(W_q)``
        reproduces the padded ``W_R``.

    SHARDING / SCALING ENVELOPE.  A scan inside one shard_map transforms and
    embeds independent centroid-face tiles through the k-convolution
    router's local k-minor factory (nvidia-mathdx on CUDA, the plan route on
    CPU).  Each completed tile is written directly into the final fine W.
    The coarse input and fine output remain resident; intermediate FFTs and
    Wigner-Seitz weights occupy one fixed tile, rather than additional whole
    coarse/fine W carriers.  The (μ,ν) sharding survives end to end.
    No step materializes a replicated N_μ²-class array — the eager
    ``local_ifftn3``/``local_fftn3`` + ``device_put`` form this replaces
    all-gathered the full tensor per rank (audit P0-4/P2-7) and is what the
    ``tools/lint/test_fft_shardmap_context.py`` gate now bans.
    """
    if output not in ("k", "R"):
        raise ValueError(f"make_w_densifier: output must be 'k' or 'R', got {output!r}")
    fine_grid = tuple(int(s) for s in fine_grid)
    w_sh = NamedSharding(mesh_xy, w_spec)
    @partial(jax.jit, out_shardings=w_sh)
    def _densify(W_q_coarse):
        coarse = tuple(int(n) for n in W_q_coarse.shape[-3:])
        if coarse == fine_grid:
            if output == "k":
                return W_q_coarse
            return make_kfft_kminor(mesh_xy, coarse, w_spec,
                                    kind="ifftn", norm="ortho")(W_q_coarse)
        if embedding is None or tuple(embedding["kgrid"]) != coarse:
            raise ValueError(f"make_w_densifier: {coarse} -> {fine_grid} needs "
                             "the per-pair Wigner-Seitz table of the coarse grid")
        if (W_q_coarse.ndim != 5 or W_q_coarse.shape[0] != W_q_coarse.shape[1]
                or embedding["pos"].shape[0] != W_q_coarse.shape[0]):
            raise ValueError("make_w_densifier: the square W carrier and its "
                             "centroid positions must have the same extent")
        if any(f < c for f, c in zip(fine_grid, coarse)):
            raise ValueError(f"make_w_densifier: fine grid {fine_grid} must be "
                             f"at least the coarse grid {coarse}")
        if any(axis is not None for axis in tuple(w_spec)[2:]):
            raise ValueError(f"make_w_densifier: k axes must be replicated, got {w_spec}")
        from jax import lax, shard_map
        from runtime.tiles import tile_units
        nk_c, nk_f = math.prod(coarse), math.prod(fine_grid)
        ifft = make_local_kfft_kminor(mesh_xy, coarse, kind="ifftn", norm="ortho")
        fft = make_local_kfft_kminor(mesh_xy, fine_grid, kind="fftn", norm="ortho")
        positions = jnp.asarray(embedding["pos"])
        pos_x, pos_y = P(tuple(w_spec)[0], None), P(tuple(w_spec)[1], None)

        def local(W, mu, nu):
            n_mu, n_nu = (int(v) for v in W.shape[:2])
            # One coarse inverse and the fine R/q blocks, plus this class's
            # image weights.  The fixed tile bounds transients from shapes.
            per_pair = W.dtype.itemsize * (nk_c + 2 * nk_f)
            per_pair += W.real.dtype.itemsize * int(embedding["img"].shape[1]) * 3
            cells = tile_units(per_pair, n_mu * n_nu)
            # Whole contiguous column rows avoid packing strided rectangles.
            # If one row exceeds the tile, split its columns with bm == 1.
            bn = min(n_nu, cells)
            bm = min(n_mu, max(1, cells // bn))
            nm, nn = -(-n_mu // bm), -(-n_nu // bn)
            bm, bn = -(-n_mu // nm), -(-n_nu // nn)
            # Cut from the two-dimensional view: a five-dimensional slice
            # permits XLA to hoist a whole-input FFT-layout copy out of scan.
            source = W.reshape(n_mu, n_nu * nk_c)
            def step(acc, i):
                m0 = jnp.minimum((i // nn) * bm, n_mu - bm)
                n0 = jnp.minimum((i % nn) * bn, n_nu - bn)
                block = lax.dynamic_slice(source, (m0, n0 * nk_c), (bm, bn * nk_c))
                r = ifft(block.reshape((bm, bn) + coarse))
                images = dict(embedding, pos=lax.dynamic_slice_in_dim(mu, m0, bm))
                r = pad_W_R_to_grid(r, fine_grid, embedding=images,
                                    pos_b=lax.dynamic_slice_in_dim(nu, n0, bn))
                result = r if output == "R" else fft(r)
                return lax.dynamic_update_slice(acc, result, (m0, n0, 0, 0, 0)), None

            initial = jnp.zeros((n_mu, n_nu) + fine_grid, W.dtype)
            result, _ = lax.scan(step, initial, jnp.arange(nm * nn), unroll=1)
            return result

        run = shard_map(local, mesh=mesh_xy, in_specs=(w_spec, pos_x, pos_y),
                        out_specs=w_spec, check_vma=False)
        return run(W_q_coarse, positions, positions)

    return _densify


def densify_w(W_q_coarse, *, mesh_xy, densifier):
    """Admit and execute one coarse-to-fine W interpolation.

    Price the exact output and bounded scratch beside the resident input,
    then run the executable whose price was read. The caller retains the
    coarse input until its last use; this helper keeps no operand cache.

    Parameters
    ----------
    W_q_coarse : jax.Array
        ``(mu, nu, *coarse_grid)`` complex W, with only centroid axes sharded.
    mesh_xy : Mesh
        The same mesh used to build ``densifier``.
    densifier : callable
        The jitted program from :func:`make_w_densifier`.

    Returns
    -------
    jax.Array
        ``(mu, nu, *fine_grid)`` in the program's declared k- or R-space
        convention and centroid-face sharding.
    """
    from runtime.aot_memory import agreed_chunk, compiled_new_bytes
    from common.gpu_utils import shared_device_room
    from lxkit import memory

    compiled = densifier.lower(W_q_coarse).compile()
    new_bytes = -agreed_chunk(-compiled_new_bytes(compiled))
    output_bytes = int(compiled.memory_analysis().output_size_in_bytes)
    scratch_bytes = max(0, new_bytes - output_bytes)
    stage = "BSE coarse-to-fine W interpolation"
    room = shared_device_room(stage)
    memory.admit(stage, (("centroid face", memory.Price(
        xla=memory.Term(scratch_bytes, output_bytes * int(mesh_xy.size),
                        "W interpolation (compiled output and bounded scratch)"),
        ranks=int(mesh_xy.size), label=stage)),),
        int(room) if isinstance(room, (int, np.integer)) else 0)
    return compiled(W_q_coarse)


def decimate_W_q_to_subgrid(W_q: jax.Array,
                            coarse_grid: tuple[int, int, int]) -> jax.Array:
    """Sub-sample a fine-grid ``W_q`` onto a coarse BZ sub-grid (every m-th q).

    ``W_q`` is ``(μ, ν, nfx, nfy, nfz)`` on a FINE BZ grid; returns the tiles at
    the coarse sub-grid BZ points ``q_j = j·(nf/nc)`` (which coincide with a
    coarse ``nc``-point grid since ``nf`` is a multiple of ``nc``).  This is the
    honest "what a coarse-grid W sampling looks like, in the SAME ISDF μ-basis"
    — used to drive ``pad_W_R_to_grid`` from a single fine restart (the q=0 tile,
    incl. its rank-1 head, is preserved because q=0 is on both grids).
    """
    nf = tuple(int(s) for s in W_q.shape[-3:])
    coarse_grid = tuple(int(s) for s in coarse_grid)
    for a, (f, c) in enumerate(zip(nf, coarse_grid)):
        if c <= 0 or f % c != 0:
            raise ValueError(
                f"decimate_W_q_to_subgrid: fine axis {a} length {f} must be a "
                f"positive multiple of coarse {c}.")
    rx, ry, rz = (nf[0] // coarse_grid[0], nf[1] // coarse_grid[1],
                  nf[2] // coarse_grid[2])
    return W_q[:, :, ::rx, ::ry, ::rz]


def _parse_grid_spec(spec) -> Optional[tuple[int, int, int]]:
    """Parse a ``kgrid_fine`` value → ``(nx, ny, nz)`` or ``None`` (unset).

    Accepts ``"NX NY NZ"``, ``"NX,NY,NZ"``, a 3-tuple/list, or an empty
    string / None (→ ``None``).  Single source for both the cohsex.in key and
    any driver flag.
    """
    if spec is None:
        return None
    if isinstance(spec, (tuple, list)):
        parts = list(spec)
    else:
        s = str(spec).strip()
        if not s:
            return None
        parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"kgrid_fine must have 3 integers, got {spec!r}")
    return tuple(int(x) for x in parts)


def _read_lorrax_input_quietly(input_file: Optional[str]) -> dict:
    """The deck as a dict, or ``{}`` — a config read must never crash a load.

    Same tolerance as :func:`_resolve_kgrid_fine`'s own read: the loader has
    to work on a restart with no deck beside it, and a malformed key is a
    reason to fall back to defaults loudly, not to lose the tensors.
    """
    if input_file is None or not os.path.isfile(input_file):
        return {}
    try:
        from gw.gw_config import read_lorrax_input
        return read_lorrax_input(input_file) or {}
    except Exception as exc:
        log.warning(f"BSE: deck read failed ({exc}); using defaults",
                    extra=message(
                        id="bse_deck_read_failed",
                        found=("deck read", f"failed ({exc})"),
                        needs="a readable deck for the BSE settings",
                        fix="check the deck syntax; the BSE settings fall back to defaults"))
        return {}


def _resolve_kgrid_fine(kgrid_fine, input_file: Optional[str]):
    """Resolve the fine grid: explicit ``kgrid_fine`` arg wins; else read the
    ``kgrid_fine`` key from ``input_file`` (cohsex.in).  Returns a 3-tuple or
    ``None`` (feature off)."""
    fine = _parse_grid_spec(kgrid_fine)
    if fine is not None:
        return fine
    if input_file is not None and os.path.isfile(input_file):
        try:
            from gw.gw_config import read_lorrax_input
            return _parse_grid_spec(read_lorrax_input(input_file).get("kgrid_fine"))
        except Exception as exc:               # config parse must never crash load
            log.warning(f"BSE: kgrid_fine config read failed ({exc}); feature off",
                        extra=message(
                            id="kgrid_fine_config_read_failed",
                            found=("kgrid_fine config read", f"failed ({exc})"),
                            needs="a readable deck with a valid kgrid_fine",
                            fix="check the deck syntax; kgrid_fine densification is off"))
    return None


def _resolve_kgrid_fine_shift(input_file: Optional[str]) -> tuple[float, float, float]:
    """The deck's ``kgrid_fine_shift`` (kgrid.x's dk, fractions of one fine step), or 0."""
    from bandstructure.bse_setup import parse_kgrid_shift
    params = _read_lorrax_input_quietly(input_file)
    return parse_kgrid_shift(params.get("kgrid_fine_shift", ""))


def coarse_conduction_count(params: dict, n_cond: int, *, log_fn=log.debug):
    """``number_cond_bands_coarse``: the conduction bands of the coarse grid that
    enter the H-transform (None: every conduction band of the deck's window).

    ``number_val_bands_coarse`` is accepted and has no effect: the H-transform
    fits every valence band of the window.  Fewer coarse than fine conduction
    bands refuses; fewer than four guard bands above the fine window warns.
    """
    raw_v = str(params.get("number_val_bands_coarse", "") or "").strip()
    if raw_v:
        log.warning("number_val_bands_coarse has no effect: the H-transform always fits "
                    "every valence band of the GW window (a requirement of its f(H)); "
                    "number_val_bands_fine chooses the BSE window",
                    extra=message(id="number_val_bands_coarse_ignored",
                                  found=("number_val_bands_coarse", raw_v),
                                  fix="remove number_val_bands_coarse"))
    raw = str(params.get("number_cond_bands_coarse", "") or "").strip()
    if not raw:
        return None
    try:
        nc_co = int(raw)
    except ValueError:
        raise ValueError(f"deck key `number_cond_bands_coarse` takes an integer, "
                         f"got {raw!r}") from None
    if nc_co < int(n_cond):
        raise ValueError(
            f"GATE bse_window_outside_coarse: number_cond_bands_fine = {n_cond} "
            f"exceeds number_cond_bands_coarse = {nc_co}; the fine window is "
            f"interpolated from the coarse bands, so it cannot be wider.  fix: "
            f"raise number_cond_bands_coarse (4 or more above the fine window).")
    return nc_co


#: Deck values for ``w_head_densify``: ``srlr`` only (since 2026-10-22).  It
#: moves the whole long-range part of W (head and, in a slab, its G_z
#: channels) into a fitted model (:func:`_srlr_terms`); the old ``c1`` and
#: ``legacy`` refuse with one line (``runtime.renames.DECK_VALUES``).
W_HEAD_DENSIFY_MODES = ("srlr",)


def resolve_w_head_densify(mode, params=None) -> str:
    """The coarse→fine W interpolation model: always ``'srlr'``.

    The explicit argument, else the deck's ``w_head_densify``, may only say
    ``srlr``; an old value refuses with the renames table's line.
    """
    from runtime.renames import check_deck_value
    val = mode
    if val is None and params is not None:
        val = params.get("w_head_densify")
    if val is None:
        return "srlr"
    val = check_deck_value("gw", "w_head_densify", str(val)).strip().lower()
    if val not in W_HEAD_DENSIFY_MODES:
        raise ValueError(f"w_head_densify = {val!r}: srlr is the only model")
    return val


def build_w_head_channel(wfn, sym, meta, params, *, coarse_grid, fine_grid,
                         whead, ref_grid, input_file, restart_file, geometry,
                         mesh=None,
                         gamma_cell: str = "fine", log_fn=log.debug, static_sample=None):
    """The C1 head channel ``S_fine(q)`` for a coarse→fine W, end to end.

    ONE composer for BOTH shipping densification paths — the ``kgrid_fine``
    bundle densification (:func:`_interpolate_bse_data_to_grid`) and
    ``exciton_bands --w-coarse-grid``.  They differ only in which grid the
    reference head was measured on (``ref_grid``), and that difference is an
    argument rather than two code paths.

    Resolves ``S_cart`` (restart dataset, else a ``dipole.h5`` rebuild),
    evaluates the integrand's own cell average on ``ref_grid`` so the caller's
    ``whead`` can be checked against it, and hands the anchored per-q scalars
    back.  Everything it prints is meant to be read: the provenance ratio is
    the one number that says whether the head being re-attached describes the
    same screening the run solved with.

    Parameters
    ----------
    wfn, sym, meta : loader / symmetry table / system parameters
        ``meta.sys_dim`` selects the
        bulk-3D or slab-2D ``vcoul`` kernel and is REQUIRED — a Meta without it
        is refused, not assumed bulk (see the body); ``sym`` is read only by the
        ``dipole.h5`` fallback route for ``S_cart`` (a restart that carries
        ``S_cart_head`` never touches it).  Both densification call sites
        already hold all three from their htransform leg, which is also where
        the ``sys_dim`` stamp is put on.
    params : dict
        Deck keys.  ``head_minibz_average`` must match the GW run — it selects
        the Baldereschi-Tosatti analytic-sphere branch of the estimator.
    geometry : vcoul.CoulombGeometry
        The lattice (``blat·bvec`` rows and Ω) the fine q and ``S_fine`` are
        evaluated on, ``CoulombGeometry.from_wfn(wfn)``.  The caller builds it
        once and passes the same object's ``bvec`` to
        ``gw.head_densify.attach_head_channel``, so the head's q and its gauge
        phase sit on one lattice.
    coarse_grid, fine_grid, ref_grid : tuple[int, int, int]
    whead : float
        The head scalar the loader injects, Ry·bohr³, real.
    gamma_cell : {"fine", "coarse"}
        ``"coarse"`` is the control variant; see
        :func:`gw.head_densify.build_fine_head_scalars`.

    Returns
    -------
    numpy.ndarray, shape ``fine_grid``, float64
    """
    from gw import head_densify
    from gw.head_correction import resolve_head_S_cart
    from ffi import _services
    _services.ensure_on_path()

    params = params or {}

    # ── DIMENSIONALITY FIRST, before the S tensor and before the Γ-cell
    # integral: an unsupported/unstamped deck must hear that, not "your
    # dipole.h5 is missing".
    #
    # ``Meta`` has no ``sys_dim`` field — it is STAMPED on the two Metas the
    # tree builds, by ``gw.gw_jax.main`` and by
    # ``bandstructure.htransform.initialize_wfns`` (which is the one BOTH
    # shipping densification paths hold: the ``kgrid_fine`` bundle leg below
    # and ``bse.exciton_bands``' ``--w-coarse-grid`` leg).  Until 2026-08-17
    # this read ``getattr(meta, "sys_dim", None)`` into a guard that RETURNED
    # on ``None``, and htransform did not stamp: so the bulk-3D refusal could
    # not fire on either path, and a ``sys_dim = 2`` deck re-attached
    # ``8π/|q|²`` — the untruncated 3D pole — where the true 2D head goes as
    # ``8π·z_c/|q_∥|``.  That error has no fixed size: it GROWS as the fine
    # grid densifies, which is the one thing a densification is for.
    sys_dim = getattr(meta, "sys_dim", None)
    if sys_dim is None:
        raise ValueError(
            "the W Γ head: the Meta handed to "
            "bse.bse_densify.build_w_head_channel carries no ``sys_dim``, so "
            "gw.head_densify cannot safely select a bulk or slab kernel.  "
            "``Meta`` has no sys_dim field; it is "
            "stamped, by gw.gw_jax.main on the GW driver's Meta and by "
            "bandstructure.htransform.initialize_wfns on this caller's.  A "
            "Meta arriving here without it was built by a third site, and "
            "that site is the fix — do not default it to 3 here.")
    sd = head_densify._validated_sys_dim(sys_dim)

    if static_sample is None:
        S_cart, prov = resolve_head_S_cart(
            restart_file, input_file=input_file, wfn=wfn, sym=sym, meta=meta,
            params={**params, "_mesh": mesh}, print_fn=log_fn)
    else:
        S_cart, prov = static_sample.S_cart, static_sample.source
    if S_cart is None:
        raise ValueError(
            f"the W Γ head needs the S tensor and could not get "
            f"one: {prov}.  The Γ head is re-attached per fine q from the "
            f"integrand ⟨v/(1 − v qᵀSq)⟩, and without S there is no integrand "
            f"— only the coarse cell AVERAGE, which is the number the "
            f"trigonometric interpolant already mishandles.  Fix by rerunning "
            f"GW so the restart carries S_cart_head, or by putting the run's "
            f"dipole.h5 beside the deck.")

    geom = geometry
    analytic_sphere = bool(params.get("head_minibz_average", False))
    gamma_ref = head_densify.gamma_cell_head_scalar(
        geom, ref_grid, S_cart, sys_dim=sd,
        analytic_sphere=analytic_sphere)
    ratio = float(whead) / gamma_ref

    S_fine = head_densify.build_fine_head_scalars(
        geom, coarse_grid, fine_grid, S_cart,
        head_ref=float(whead), gamma_ref=gamma_ref, ref_grid=ref_grid,
        sys_dim=sd,
        analytic_sphere=analytic_sphere, gamma_cell=gamma_cell)

    # THE SUM RULE, reported on the deck.  The head channel's zone average is
    # a property of the material and the cell, and the grid-independent number
    # it converges to is (1/N_c)·⟨S⟩ over the COARSE cell — so the target has
    # to be expressed on the coarse cell even when the reference head was
    # measured on a different one (which is exactly the ``--w-coarse-grid``
    # case, where a natively fine restart is decimated).  Getting this wrong
    # does not change a single number that ships; it changes what the log
    # claims, which is worse, because a diagnostic nobody can trust is a
    # diagnostic nobody reads.
    cg = tuple(coarse_grid)
    gamma_coarse = (gamma_ref if tuple(ref_grid) == cg
                    else head_densify.gamma_cell_head_scalar(
                        geom, cg, S_cart, sys_dim=sd,
                        analytic_sphere=analytic_sphere))
    n_c = cg[0] * cg[1] * cg[2]
    target = float(whead) * (gamma_coarse / gamma_ref) / n_c
    weight = head_densify.coarse_gamma_cell_weights(
        head_densify.fine_q_cart(geom.bvec, fine_grid), cg, fine_grid,
        bvec=geom.bvec)
    zone = head_densify.head_channel_zone_average(S_fine)
    log_fn(
        f"[w-head] S_cart: {prov}; provenance ratio whead/⟨v/(1−vqSq)⟩ on "
        f"{tuple(ref_grid)} = {ratio:.6f} (1.0 means the restart's head and "
        f"this S are the same screening)")
    log_fn(
        f"[w-head] head channel re-attached at "
        f"{int(np.count_nonzero(S_fine))} fine q carrying total weight "
        f"{float(np.sum(weight)):.6f} (= N_f/N_c = "
        f"{(fine_grid[0]*fine_grid[1]*fine_grid[2])/n_c:.6f}; more points than "
        f"weight means boundary/common-cell sharing), of "
        f"{fine_grid[0]*fine_grid[1]*fine_grid[2]} fine q total")
    log_fn(
        f"[w-head] S(Γ_fine) = {S_fine[0, 0, 0]:.4f} vs injected "
        f"{float(whead):.4f} Ry·bohr³ (×{S_fine[0, 0, 0]/float(whead):.3f} "
        f"for a {(fine_grid[0]*fine_grid[1]*fine_grid[2])/n_c:.0f}× finer "
        f"cell); SUM RULE: zone average {zone:.6f} vs the grid-independent "
        f"{target:.6f} ({100.0*abs(zone/target - 1.0):.1f}% — a midpoint "
        f"quadrature error that shrinks under refinement, 0 when fine == "
        f"coarse) [gamma_cell={gamma_cell}]")
    if abs(ratio - 1.0) > 0.05:
        import warnings
        msg = (f"W Γ head: the injected head and the S tensor "
               f"disagree by {100*abs(ratio-1.0):.1f}% on {tuple(ref_grid)} "
               f"(whead={float(whead):.4f}, integrand={gamma_ref:.4f} "
               f"Ry·bohr³).  The re-attached head keeps the INJECTED scale, "
               f"so nothing silently changes magnitude, but its q-dependence "
               f"comes from an S that does not reproduce it — most often a "
               f"deck vhead/whead_0freq override pinned to an external value, "
               f"or a head_minibz_average / wcoul0_source that differs from "
               f"the GW run's.")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        log.debug(f"[w-head] [WARN] {msg}")   # the warnings.warn above carries it
    return S_fine


def _srlr_terms(restart_file, wfn, sym, meta, params, *, coarse_grid,
                fine_grid, n_rmu, n_rmu_pad, mesh_xy, input_file, centroid_frac,
                lr_grid=None, log_fn=log.debug):
    """The long-range W terms of the srlr W interpolation, on the coarse and fine q.

    ``lr_grid``: the q grid the restart's W0_lr (and ``sym``) is on, default
    ``coarse_grid``.  ``exciton_bands --w-coarse-grid`` passes its native fine
    grid: the rows are unfolded there and decimated to the ``coarse_grid``
    sub-grid (every m-th q, as :func:`decimate_W_q_to_subgrid`), so the model
    is fitted on, and subtracted at, exactly the q the interpolant sees.

    W = J + F̃ᴴ D F̃ exactly at every coarse q (``gw.lr_channels``).  The
    densified W is

        X(q_c) = W(q_c) − w(|q_c|) LR_m(q_c)  (q_c ≠ 0),   X(0) = W(0) − LR(0),
        W(q_f) = T[X](q_f) + w(|q_f|) LR_m(q_f)  (q_f ≠ 0),  W(Γ) = T[X](0) + LR(0),

    with T the trigonometric densifier, LR_m = F̃_mᴴ [diag(1/v) − P_m]⁻¹ F̃_m the
    fitted model (``gw.lr_channels.fit_model``; v the split's own kernel at
    the coarse q, the fine grid's at the fine q) and LR(0)
    the exact coarse-Γ term (literal G = 0 has v = 0 there; the Γ head stays
    c1's).  Subtracting the MODEL, not the exact LR, keeps every coarse tile
    exact (on a mini-BZ-averaged 3D deck up to the coarse→fine change of the
    head's cell average); the model's own error only changes how smooth X is.

    The model's Bloch phase sits on ``centroid_frac``
    (:func:`w_tile_centroid_frac`, the FFT points the tile is collocated on and
    the body embedding uses), not on the centroid file's rounded spelling.

    Returns ``(A_c, D_c, A_f, D_f)``: per coarse / fine q (flat fft order) the
    rows ``A (n_q, n_L, n_rmu_pad)`` and ``D (n_q, n_L, n_L)`` (window
    included) of the rank-n_L terms Aᴴ D A.  Host numpy, n_L·N_μ per q.
    """
    from ffi import _services
    _services.ensure_on_path()
    from vcoul import CoulombGeometry, inscribed_radius, lr_channels
    from file_io.restart_bundle import read_lr_channels
    from gw import lr_channels as LC
    from gw.head_correction import resolve_head_S_cart

    lr = read_lr_channels(restart_file)
    if lr is None:
        raise ValueError(
            f"GATE srlr_restart_channels: {restart_file} has no W0_lr group; "
            "want: the dressed long-range channels the static Dyson route "
            "persists beside W0 (gw.lr_channels); fix: rerun GW with this "
            "tree (a static-Dyson W0, e.g. compute_mode = cohsex); srlr is "
            "the only W interpolation.")
    t = lr["tables"]
    if t is None:
        raise ValueError(
            "GATE srlr_wedge_tables: W0 is stored on the full zone, so the "
            "W0_lr rows carry no unfold tables; want: restart_q_storage = ibz.")
    sys_dim = int(lr["sys_dim"])
    if sys_dim != int(meta.sys_dim):
        raise ValueError(f"srlr: W0_lr was written for sys_dim {sys_dim}, "
                         f"the deck says {int(meta.sys_dim)}")
    if not np.allclose(t.q_irr_frac, lr["q_frac"], atol=1e-10):
        raise ValueError("srlr: the W0_lr rows are not W0's q wedge")
    _, n_L, n_file = lr["Ft"].shape
    if n_file != n_rmu:
        raise ValueError(f"srlr: W0_lr carries {n_file} centroids, the bundle {n_rmu}")
    c0 = int(lr["n_z"]) if sys_dim == 2 else 0
    bvec = np.asarray(CoulombGeometry.from_wfn(wfn).bvec, dtype=np.float64)

    lg = tuple(int(n) for n in (coarse_grid if lr_grid is None else lr_grid))
    cg = tuple(int(n) for n in coarse_grid)
    Ft, Pf, v_c, K_full, q_full = _unfold_lr_channels(
        lr, sym, coarse_grid=lg, n_rmu=n_rmu, n_rmu_pad=n_rmu_pad,
        mesh_xy=mesh_xy)
    if lg != cg:
        if any(f % c for f, c in zip(lg, cg)):
            raise ValueError(f"srlr: the coarse grid {cg} is not a sub-grid of "
                             f"the restart's {lg}")
        sub = np.ravel_multi_index(tuple(np.meshgrid(
            *[np.arange(c) * (f // c) for f, c in zip(lg, cg)], indexing="ij")),
            lg).ravel()
        Ft, Pf, v_c, K_full, q_full = (a[sub] for a in (Ft, Pf, v_c, K_full, q_full))
    K0 = K_full[:, c0].copy()
    if sys_dim == 2:
        K0[:, 2] = 0.0
    qc = K0 @ bvec                                         # WS-representative q, Cartesian

    # ── the model ─────────────────────────────────────────────────────────
    q_ins = inscribed_radius(bvec, sys_dim)
    r_a, r_b = LC.WINDOW_INNER * q_ins, LC.WINDOW_OUTER * q_ins
    r_mu = np.asarray(centroid_frac, dtype=np.float64) @ (2.0 * np.pi * np.linalg.inv(bvec).T)
    S_cart, _prov = resolve_head_S_cart(
        restart_file, input_file=input_file, wfn=wfn, sym=sym, meta=meta,
        params={**params, "_mesh": mesh_xy}, print_fn=log_fn)
    if S_cart is None:
        raise ValueError(f"GATE srlr_head_tensor: no S tensor ({_prov}); the P "
                         "model's head anchor needs S_cart_head or dipole.h5.")
    model, info = LC.fit_model(qc, Ft[..., :n_rmu], Pf, r_mu=r_mu, S_cart=S_cart,
                               c0=c0, sys_dim=sys_dim, r_fit=r_b)
    gamma = int(np.argmin(np.linalg.norm(qc, axis=1)))
    D0 = LC.screened_lr(Pf[gamma][None], v_c[gamma][None])[0]

    def terms(qcs, v):
        n = len(qcs)
        A = np.zeros((n, n_L, int(n_rmu_pad)), np.complex128)
        D = np.zeros((n, n_L, n_L), np.complex128)
        qn = np.linalg.norm(qcs, axis=1)
        w = LC.window(qn, r_a, r_b)
        on = np.flatnonzero((w > 0.0) & (qn > 1e-9))
        if on.size:
            A[on, :, :n_rmu] = LC.model_rows(model, qcs[on])
            D[on] = w[on, None, None] * LC.screened_lr(LC.model_P(model, qcs[on]), v[on])
        g = np.flatnonzero(qn <= 1e-9)
        A[g], D[g] = Ft[gamma], D0
        return A, D, on.size

    A_c, D_c, n_on_c = terms(qc, v_c)
    fg = np.asarray(fine_grid, dtype=np.int64)
    q_fine = np.stack(np.meshgrid(*[np.arange(n) / n for n in fg],
                                  indexing="ij"), -1).reshape(-1, 3)
    # The fine q take the kernel V would carry on the fine grid: the point
    # v, or (3D mc_average_vcoul_body, W0_lr's ``head_kernel``) the head's
    # average over the FINE mini-BZ cell, as c1's per-fine-q head does.  At a
    # shared q ≠ 0 the tile then moves by that cell-average change only.
    head_fn = None
    if lr["head_kernel"] == "minibz_average":
        from vcoul import build_v_head_miniBZ_fn_3d
        omega = float(CoulombGeometry.from_wfn(wfn).cell_volume)
        vfn = build_v_head_miniBZ_fn_3d(tuple(fine_grid), bvec, omega)
        head_fn = lambda K: omega * vfn(K)                 # noqa: E731  (bare units)
    ch_f = lr_channels(q_fine, bvec=bvec, sys_dim=sys_dim, n_z=c0, head_fn=head_fn)
    Kf = q_fine + ch_f.gvec[:, c0]
    if sys_dim == 2:
        Kf[:, 2] = 0.0
    A_f, D_f, n_on_f = terms(Kf @ bvec, ch_f.v)
    n_c, n_f = len(qc), len(Kf)
    grid = lambda g: "x".join(str(int(n)) for n in g)      # noqa: E731
    for line in (
            f"W interpolation (srlr): coarse {grid(coarse_grid)} -> fine {grid(fine_grid)}",
            f"  long-range subspace G_L: {n_L} plane wave(s) per q"
            + (f" (G_z = -{c0}..{c0})" if sys_dim == 2 else " (G = 0)"),
            f"  fit on {info['n_fit']} coarse q: F~ degree {info['deg_F']} "
            f"(rel. residual {info['relres_F']:.4f}), P degree {info['deg_P']} in q^2, "
            f"direction-resolved to degree {'/'.join(str(d) for d in info['deg_P_dir'])} "
            f"(rel. residual {info['relres_P']:.4f})",
            "  P head q.S.q, fitted S diagonal "
            + " ".join(f"{s:.4f}" for s in np.diag(info["S_fit"]))
            + ", anchor " + " ".join(f"{s:.4f}" for s in np.diag(np.real(S_cart))[:info["S_fit"].shape[0]]),
            f"  window w(q): 1 below {r_a:.4f}, 0 above {r_b:.4f} bohr^-1 "
            f"(q_ins {q_ins:.4f})",
            f"  model used at {n_on_c} of {n_c} coarse q and {n_on_f} of {n_f} fine q, "
            f"Gamma exact; skipped (w = 0, interpolant only): {n_c - n_on_c - 1} "
            f"coarse q, {n_f - n_on_f - 1} fine q",
            f"  head kernel {lr['head_kernel']}; S anchor {_prov}"):
        log.info(line)
    return A_c, D_c, A_f, D_f


def _unfold_lr_channels(lr, sym, *, coarse_grid, n_rmu, n_rmu_pad, mesh_xy):
    """W0_lr's F̃ and P from the q wedge onto the full coarse zone (flat fft order).

    Each channel row is one ζ-like leg (``symmetry_maps.unfold_isdf_one_leg``
    with the channel's G); P_ab → c_a c_b* P_ab with the same τ phases,
    conjugated on antiunitary rows; slab channels are reordered by the target
    G_z.  Returns ``Ft (n_q, n_L, n_rmu_pad)``, ``P (n_q, n_L, n_L)``, the
    split's kernel ``v (n_q, n_L)`` (a function of |K|), the target channel
    momenta ``K (n_q, n_L, 3)`` and ``q_full (n_q, 3)``, both crystal.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import (bgw_integer_q_to_fractional, tau_phase_row,
                               unfold_isdf_one_leg)
    from common.collectives import device_put_process_local

    t = lr["tables"]
    sys_dim = int(lr["sys_dim"])
    c0 = int(lr["n_z"]) if sys_dim == 2 else 0
    n_qi, n_L, _ = lr["Ft"].shape
    grid = np.asarray(coarse_grid, dtype=np.int64)
    tp = t.padded(int(n_rmu_pad))
    to_host = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh_xy, P()))
    col_sh = NamedSharding(mesh_xy, P(None, "x"))
    rows_full = []
    for a in range(n_L):
        col = np.zeros((n_qi, int(n_rmu_pad)), np.complex128)
        col[:, :n_rmu] = lr["Ft"][:, a]
        out = unfold_isdf_one_leg(
            device_put_process_local(col, col_sh),
            source_gvec_components=np.asarray(lr["gvec"][:, a], dtype=np.int32),
            sym=sym, sym_idx=tp.sym_idx_q, sym_perm=tp.sym_perm,
            L_table=tp.L_table, q_irr_frac=tp.q_irr_frac, kgrid=tuple(coarse_grid),
            mesh_xy=mesh_xy, component_action="scalar")
        rows_full.append(np.asarray(to_host(out)))
    Ft = np.stack(rows_full, axis=1)                       # (nq, n_L, μ_pad)
    nq = Ft.shape[0]

    # The same star rows the one-leg unfold used: each target channel's
    # momentum, its τ phase c_a (P transforms as c_a c_b* P, conjugated on
    # antiunitary rows), and the channel order (slab: by G_z).
    q_int = np.asarray(sym.kvecs_asints, dtype=np.int64)
    fft_int = np.stack(np.meshgrid(*[np.arange(n) for n in grid],
                                   indexing="ij"), -1).reshape(-1, 3)
    if q_int.shape != fft_int.shape or not np.array_equal(np.mod(q_int, grid), fft_int):
        raise ValueError(
            "GATE srlr_q_order: SymMaps' q rows are not the bundle's flat fft "
            "order; the unfolded rows would land on other q.")
    q_full = bgw_integer_q_to_fractional(q_int, grid)
    S_all = np.asarray(sym.sym_mats_k, dtype=np.int64)
    n_sp = int(np.asarray(sym.sym_matrices).shape[0])
    tau = np.asarray(sym.translations, dtype=np.float64)
    idx, rows = np.asarray(tp.irr_idx_q), np.asarray(tp.sym_idx_q)
    Pf = np.empty((nq, n_L, n_L), np.complex128)
    vf = np.empty((nq, n_L))
    K_full = np.empty((nq, n_L, 3))
    for iq in range(nq):
        s, p = int(rows[iq]), int(idx[iq])
        Kp = np.asarray(tp.q_irr_frac[p], dtype=np.float64) + lr["gvec"][p]
        ph = tau_phase_row(S_all[s % n_sp], tau[s % n_sp], Kp)
        c = np.ones(n_L, np.complex128) if ph is None else ph
        Pq = c[:, None] * lr["P"][p] * np.conj(c)[None, :]
        if s >= n_sp:
            Pq = np.conj(Pq)
        Kt = (S_all[s] @ Kp.T).T
        order = np.argsort(Kt[:, 2], kind="stable") if sys_dim == 2 else np.arange(1)
        if sys_dim == 2 and not np.allclose(Kt[order, 2], np.arange(-c0, c0 + 1)):
            raise ValueError(
                f"GATE srlr_channel_relabel: q {iq} maps the G_z channels to "
                f"{Kt[order, 2].tolist()}; want the set -n_z..n_z.")
        Ft[iq], Pf[iq], K_full[iq] = Ft[iq][order], Pq[np.ix_(order, order)], Kt[order]
        vf[iq] = np.asarray(lr["v"][p])[order]
    return Ft, Pf, vf, K_full, q_full


def _add_rank_lr(W_q, A, D, mesh_xy, sign):
    """``W_q[μ,ν,q] += sign · Σ_ab conj(A[q,a,μ]) D[q,a,b] A[q,b,ν]`` in W's layout, in place.

    W_q ``(μ, ν, nx, ny, nz)`` at ``P('x','y',None,None,None)`` is DONATED: the
    result is written into its buffer and the caller rebinds it.  A
    ``(n_q, n_L, μ)`` is held twice, μ on x and ν on y (the
    ``attach_head_channel`` pattern), so every rank forms only its own tile
    and no collective runs.  The term goes in q blocks, a ``lax.scan`` inside
    the ``shard_map``: the most q whose (μ_loc, ν_loc, q) term and its layout
    copy fit the fixed tile (``runtime.tiles.tile_units``), so the transient is
    one block rather than a second W (32.6 GB per rank beside W's own 32.6 GB
    at CrI3 30x30, N_μ 3000, P4).  The last block is clamped onto its
    neighbour, and the q it shares is zeroed in its term.  One block is the
    whole-grid update.
    # ponytail: A is host-built for every q (n_q·n_L·N_μ/√P per rank); build it
    # on device from the fit coefficients if that row stack ever matters.
    """
    from jax import lax, shard_map
    from common.collectives import device_put_process_local
    from runtime.tiles import tile_units
    a_x = device_put_process_local(A, NamedSharding(mesh_xy, P(None, None, "x")))
    a_y = device_put_process_local(A, NamedSharding(mesh_xy, P(None, None, "y")))
    d = device_put_process_local(D, NamedSharding(mesh_xy, P()))
    w_spec = P("x", "y", None, None, None)

    def _local(W, ax, ay, dd):                     # W (μ_loc, ν_loc, nx, ny, nz); ax/ay (n_q, n_L, μ_loc/ν_loc)
        shape = W.shape
        W = W.reshape(shape[0], shape[1], -1)
        n_q = W.shape[2]
        bq = tile_units(2 * 16 * shape[0] * shape[1], n_q)

        def term(q0):                              # (μ_loc, ν_loc, bq)
            take = lambda a: lax.dynamic_slice_in_dim(a, q0, bq, axis=0)
            return jnp.einsum("qam,qab,qbn->mnq", jnp.conj(take(ax)), take(dd), take(ay))
        if bq == n_q:
            return (W + sign * term(0)).reshape(shape)

        def step(W, i):
            q0 = jnp.minimum(i * bq, n_q - bq)
            t = jnp.where(jnp.arange(bq) >= i * bq - q0, term(q0), 0)
            blk = lax.dynamic_slice_in_dim(W, q0, bq, axis=2)
            return lax.dynamic_update_slice_in_dim(W, blk + sign * t, q0, axis=2), None
        W, _ = lax.scan(step, W, jnp.arange(-(-n_q // bq)), unroll=1)
        return W.reshape(shape)

    add = jax.jit(shard_map(_local, mesh=mesh_xy, out_specs=w_spec,
                            in_specs=(w_spec, P(None, None, "x"), P(None, None, "y"), P())),
                  donate_argnums=(0,))
    return add(W_q, a_x, a_y, d)


#: Bound of the operator identity in :func:`densified_dipoles`.  At a fine q
#: that is also a coarse k the interpolated operator is ``U^† C^T O C^* U``
#: exactly — the lattice image's FFT round trip and nothing else — so a
#: deviation above round-off is bookkeeping (a conjugation, a band order),
#: never a fit error, and the bound does not scale with the deck.
DENSIFIED_DIPOLE_ONGRID_RTOL = 1.0e-8

#: Floor of the state identity: at a coarse k the fine window states span
#: the coarse states the fH's energy order puts at the window's positions,
#: ``σ(C_k^* U_q) = 1`` on those rows, up to the fit's projection defect
#: ``max|C Cᴴ − I|`` (the measured margin is 10x that defect, plus this floor).
DENSIFIED_DIPOLE_STATE_FLOOR = 1.0e-8

#: Two coarse energies closer than this (Ry) are one degenerate level when
#: the on-grid identity reads which coarse states a window edge holds.
DENSIFIED_DIPOLE_DEGEN_RY = 1.0e-6

#: Component names of the four lattice images :func:`densified_dipoles` builds.
_DIPOLE_IMAGES = ("v_x", "v_y", "v_z", "f(H)")

#: Pair rules for same-occupation (vv', cc') pairs of the window position
#: operator, and the exact-degeneracy floor (Ry): the rule and the floor of
#: ``tdagw.td_system.position_from_velocity`` (pinned by a CPU test).
DENSIFIED_POSITION_PAIR_RULES = ("mask", "cut", "cv")
DENSIFIED_POSITION_EXACT_DEGENERACY_RY = 1.0e-12
#: q points per host block of the fine-state position rotation.
_POSITION_Q_CHUNK = 64
#: Appended to the route label when the dipoles are the exact r of the fine
#: QP states (the label is what eigenvectors.h5 / Haydock outputs record).
DENSIFIED_POSITION_FORM = "exact r in the fine QP states (DFT eigenbasis of their f(H) block, v1)"


def _position_from_velocity_rule(v, e, n_val, *, rule, gap_ry):
    r"""``r_nm = v_nm / (i (E_n − E_m))`` over every window pair, one pair rule.

    ``v`` (3, nq, nb, nb) is ``⟨n|v̂|m⟩`` between eigenstates of the
    velocity's own Hamiltonian with energies ``e`` (nq, nb), valence rows
    first.  The diagonal and exact degeneracies (|ΔE| below
    :data:`DENSIFIED_POSITION_EXACT_DEGENERACY_RY`) carry no position; c↔v
    pairs are kept as they are; same-occupation pairs follow ``rule``:
    ``mask`` multiplies by ``1 − exp(−ΔE²/(2 gap²))``, ``cut`` zeroes
    ``|ΔE| < gap``, ``cv`` zeroes them all.  Both v and r are Hermitized.
    This is ``tdagw.td_system.position_from_velocity``'s rule without its
    gates and receipt (the callers gate).
    """
    v = np.asarray(v, dtype=np.complex128)
    e = np.asarray(e, dtype=np.float64)
    nb = int(e.shape[-1])
    v = 0.5 * (v + np.conj(np.swapaxes(v, -1, -2)))
    de = e[:, :, None] - e[:, None, :]                       # E_n - E_m
    occ = np.arange(nb) < int(n_val)
    cv = occ[:, None] != occ[None, :]
    off = ~np.eye(nb, dtype=bool)
    same = off & ~cv
    exact = np.abs(de) < DENSIFIED_POSITION_EXACT_DEGENERACY_RY
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((off[None] & ~exact)[None], v / (1j * de[None]), 0.0)
    if rule == "mask":
        r = r * np.where(same[None], -np.expm1(-de ** 2 / (2.0 * gap_ry ** 2)),
                         1.0)[None]
    elif rule == "cut":
        r = np.where((same[None] & (np.abs(de) < gap_ry))[None], 0.0, r)
    else:
        r = np.where(same[None, None], 0.0, r)
    r[:, :, np.arange(nb), np.arange(nb)] = 0.0
    return 0.5 * (r + np.conj(np.swapaxes(r, -1, -2)))


def _position_in_fine_states(v_win, f_win, n_val, newton, *, rule, gap_ry,
                             chunk=_POSITION_Q_CHUNK):
    r"""⟨ĩ|r̂|j̃⟩ in fine states that are NOT eigenstates of the velocity's H.

    With ``eqp_corrections`` the fine states |ĩ⟩ diagonalize the interpolated QP
    ``f(H)``; the velocity's own (DFT) ``f(H)`` block ``f_win`` is then not
    diagonal in them.  Per q: ``f_win = W diag(φ) Wᴴ``, ``ε = f⁻¹(φ)``
    (``newton``), ``v' = Wᴴ v W``, ``r' = v'/(i Δε)`` with the pair rule
    (:func:`_position_from_velocity_rule`), ``r = W r' Wᴴ``.  Exact for
    the c↔v pairs when W does not mix the window's valence and conduction
    rows; ``offblock`` measures that mixing.

    Returns ``(r_win (3, nq, nb, nb), eps (nq, nb) Ry, offblock (nq,),
    cv_eig_sum)``, the last the Σ|v'_cv/Δε|² of the eigenbasis (the window
    sum the rotation must preserve).
    """
    nq, nb = int(f_win.shape[0]), int(f_win.shape[-1])
    r_win = np.empty((3, nq, nb, nb), np.complex128)
    eps_all = np.empty((nq, nb), np.float64)
    offblock = np.zeros(nq, np.float64)
    cv_eig_sum = 0.0
    for q0 in range(0, nq, int(chunk)):
        sl = slice(q0, min(nq, q0 + int(chunk)))
        fq = np.asarray(f_win[sl], dtype=np.complex128)
        fq = 0.5 * (fq + np.conj(np.swapaxes(fq, -1, -2)))
        phi, w = np.linalg.eigh(fq)
        eps = np.asarray(newton(phi), dtype=np.float64)
        order = np.argsort(eps, axis=1, kind="stable")
        eps = np.take_along_axis(eps, order, axis=1)
        w = np.take_along_axis(w, order[:, None, :], axis=2)
        vp = np.einsum("qim,aqij,qjn->aqmn", w.conj(), v_win[:, sl], w)
        rp = _position_from_velocity_rule(vp, eps, n_val, rule=rule,
                                          gap_ry=gap_ry)
        cv_eig_sum += float(np.sum(np.abs(rp[:, :, n_val:, :n_val]) ** 2))
        r_win[:, sl] = np.einsum("qim,aqmn,qjn->aqij", w, rp, w.conj())
        eps_all[sl] = eps
        offblock[sl] = np.maximum(
            np.linalg.norm(w[:, :n_val, n_val:], axis=(1, 2)),
            np.linalg.norm(w[:, n_val:, :n_val], axis=(1, 2)))
    return r_win, eps_all, offblock, cv_eig_sum


def _qp_window_mixing(f_win, n_val):
    """``(max v↔c mixing, max relative off-diagonal)`` of f(H) in the fine states.

    ``f_win`` (nq, nb, nb) is the velocity's DFT f(H) block in the fine
    window, valence rows first.  The two numbers of the r-fix record
    (``max_offblock_mixing``, ``fh_offdiag_rel_max``), computed the same way
    whether or not the fix is applied: they say when ``dipole_position_fix``
    is needed (the fine QP states are not DFT eigenstates).
    """
    fq = 0.5 * (f_win + np.conj(np.swapaxes(f_win, -1, -2)))
    w = np.linalg.eigh(fq)[1]
    mix = np.maximum(np.linalg.norm(w[:, :n_val, n_val:], axis=(1, 2)),
                     np.linalg.norm(w[:, n_val:, :n_val], axis=(1, 2)))
    off = np.linalg.norm(
        f_win - np.einsum("qii->qi", f_win)[..., None] * np.eye(f_win.shape[-1]),
        axis=(1, 2)) / np.maximum(np.linalg.norm(f_win, axis=(1, 2)), 1e-300)
    return float(mix.max()), float(off.max())


#: Smallest eigenvalue of the weight ``g`` over the fine window, relative to
#: its largest, that :func:`_velocity_from_weighted` divides by (the weight
#: is >= 1 - e^-4 on the window's own bands at every coarse k).
DENSIFIED_VELOCITY_WEIGHT_FLOOR = 1.0e-3


def velocity_weight(energies, e_cut, width):
    r"""``g(ε) = 1 − exp(−((e_cut − ε)/width)²)`` below ``e_cut``, 0 above.

    The weight :func:`densified_dipoles` carries the velocity with when the
    fit window's top band meets an unfitted band: ``e_cut`` is the lowest
    energy of the first unfitted band, so ``g`` vanishes (C¹) on every state
    a band crossing can move in or out of the window, and ``width`` is half
    the distance from the BSE window's top to ``e_cut``, so ``g ≥ 1 − e⁻⁴``
    on the window.  Energies in Ry, any shape.
    """
    x = (float(e_cut) - np.asarray(energies, dtype=np.float64)) / float(width)
    return np.where(x > 0.0, -np.expm1(-x * x), 0.0)


def _velocity_from_weighted(o_win, f_win, g_of_f, *, where):
    r"""``⟨i|v̂|j⟩`` in the fine window from the interpolated ``g v g``.

    ``o_win`` (3, nq, nb, nb) is the window block of the htransform-carried
    ``g(H) v g(H)`` and ``f_win`` (nq, nb, nb) the window block of the
    velocity's own f(H), both in the same fine states; ``g_of_f`` maps
    f-eigenvalues to the weight (Newton inverse, then
    :func:`velocity_weight`).  Returns ``(G^{-1} o_win G^{-1}, floor)`` with
    ``G = g(f_win)`` and ``floor`` the min/max eigenvalue of ``G``.  Exact
    where the window is an invariant subspace of f(H) (every coarse k, the
    DFT path); otherwise the error is the window's leak out of that
    subspace times ``1 − g``, small on the window.
    """
    w, u = np.linalg.eigh(0.5 * (f_win + np.conj(np.swapaxes(f_win, -1, -2))))
    g = np.asarray(g_of_f(w), dtype=np.float64)
    floor = float(np.min(g.min(axis=-1) / np.maximum(g.max(axis=-1), 1e-300)))
    if not floor > DENSIFIED_VELOCITY_WEIGHT_FLOOR:
        raise ValueError(
            f"GATE densified_velocity_weight: got min/max weight {floor:.3e} "
            f"over the fine window ({where}); want > "
            f"{DENSIFIED_VELOCITY_WEIGHT_FLOOR:.0e}; why: the velocity is "
            f"interpolated as g v g and divided back in the window, and g "
            f"vanishes where unfitted bands enter the fit window; fix: more "
            f"conduction guard bands above the BSE window (raise ncond).")
    g_inv = np.einsum("qij,qj,qkj->qik", u, 1.0 / g, np.conj(u))
    return np.einsum("qij,aqjk,qkl->aqil", g_inv, o_win, g_inv), floor


def densified_dipoles(velocity, *, sym, basis, fine, fh_energies,
                      coarse_grid, n_val, window_start, a_band_index,
                      mesh_xy, log_fn=log.debug, full_window: bool = False,
                      qp_states: bool = False, pair_rule: str = "mask",
                      pair_gap_ev: float = 0.05,
                      mixing_receipt: bool = False, probe=None) -> dict:
    r"""r-form dipoles of a ``kgrid_fine`` bundle, in the gauge of its own ψ.

    ``probe``: on a shifted fine grid, a ``compute_wfns_fi`` bundle (with
    coefficients) at the coarse k, on which the on-grid identities are read;
    no point of the shifted grid is a coarse k.

    The coarse WFN's velocity matrix ``v(k)`` (``psp.get_dipole_mtxels``'
    ``dipole.h5``, unfolded in the loader's gauge, which is the Galerkin
    fit's) is carried through the htransform that densified ψ and contracted
    with the SAME fine eigenvectors ``U_q`` that build the densified ψ
    (``compute_wfns_fi``'s ``coeffs_fi``), so no map between two sets of
    states, and no gauge fit, exists:

    .. math::
        v_B(k) = C_k^{T} g(k)\, v(k)\, g(k)\, C_k^{*},\qquad
        V_R = \frac{1}{N_k}\sum_k e^{2\pi i k\cdot R}\, v_B(k),\qquad
        v(q) = G_q^{-1}\, U_q^\dagger\,
            \mathrm{Herm}\sum_R e^{-2\pi i q\cdot R} V_R\; U_q\, G_q^{-1}

    (``bandstructure.orbital.interpolate_band_operator``, the htransform's
    band-operator route), with ``g = velocity_weight(ε)`` of the velocity's
    own energies and ``G_q = g(U_q^† f(H)(q) U_q)`` the same weight in the
    fine window (:func:`_velocity_from_weighted`).  ``g`` vanishes above the
    lowest energy of the first unfitted band: where the fit window's top band
    meets it the projected ``v`` is not smooth in k (Si 4³→8³, 36 fitted
    bands: outer-shell image 26 % unweighted, against 1.9 % for f(H)).  A
    window whose top band stays below every unfitted band takes ``g ≡ 1``,
    the unweighted route.  The transition energies are the velocity's own:
    the file's ``band_energies`` (DFT for ``dipole.h5``), never ``eqp_corrections``,
    because ``r = v/ΔE`` is scissor-invariant.  Under ``eqp_corrections`` the fine
    states are eigenvectors of the QP ``f(H)``, so each band's energy is the
    Rayleigh quotient of the file's ``f(H)`` in it, Newton-inverted (exact
    when the two Hamiltonians share eigenvectors, second order otherwise):

    .. math::
        E_n(q) = f^{-1}\big(U_{qn}^\dagger\, [f(H)]_{\mathrm{file}}(q)\,
        U_{qn}\big),\qquad
        d^\alpha_{cvq} = \frac{v^\alpha_{cv}(q)}{E_c(q) - E_v(q)},

    the ``d^α_{cvk} = ⟨ck|v̂_α|vk⟩/ΔE`` of
    ``absorption_common.slice_dipole_to_bse_window``.  No centroid inner
    product enters, so the centroid set cannot bias it.

    Two receipts.  The on-grid identity is a gate: at every fine q that is a
    coarse k the fine window states span the coarse states that hold the
    window's positions in the order of ``fh_energies`` (the fH's own,
    QP under ``eqp_corrections``, which can move a band across a window edge: MoS₂
    9×9 at Γ), degenerate partners included, ``σ(W_S) = 1`` for
    ``W = C_k^* U_q`` (catches a q list or band window that is not the
    htransform's), and the four images reproduce ``W^† O(k) W``
    to :data:`DENSIFIED_DIPOLE_ONGRID_RTOL` (catches a contraction
    convention).  Each image's outer-shell locality
    (``fh_interp.image_locality``) is a diagnostic: a velocity whose gauge
    differs from ``ctilde``'s is not smooth in k and shows up there, which no
    identity at the coarse points can see.

    Parameters
    ----------
    velocity : tuple
        ``(parents, energies, label, path)``: ``bandstructure.orbital.
        stored_velocity`` of ``path`` (authenticated against the htransform's
        WFN), ``parents`` (n_parent, 3, nb_file, nb_file) Ry·bohr at the
        file-wedge rows, ``energies`` (n_parent, nb_file) Ry.
    sym
        The WFN's symmetry maps (``irr_idx_k``, ``unfolded_kpts``, the polar
        time-odd unfold).
    basis : isdf.galerkin.GalerkinBasis
        The htransform's basis: ``ctilde`` (nk_co, nb_fit, rank) over the
        absolute bands ``basis.band_range``.
    fine : SimpleNamespace
        ``compute_wfns_fi(..., return_coeffs=True)``: ``coeffs_fi``
        (nk_f, rank, nb_fi), q-sharded ``P(('x','y'))``, valence rows first;
        ``q_frac`` (nk_f, 3).
    fh_energies : array (nk_co, nb_fit)
        The coarse energies (Ry) the htransform's ``f(H)`` was built from
        (``enk_sigma.T``), in ``ctilde``'s k order: they decide which
        coarse states a window position holds.
    coarse_grid : tuple of int
        (nkx, nky, nkz) of ``ctilde``'s flat k axis.
    n_val : int
        Valence rows of ``coeffs_fi``.
    window_start : int
        First fitted band ``coeffs_fi`` returns (``compute_wfns_fi``'s
        ``band_window_fi[0]``, relative to ``basis.band_range[0]``).
    a_band_index : int or None
        The htransform's ``a`` band (``htransform_a_band``).
    mesh_xy : jax.sharding.Mesh
        The ('x','y') mesh.  COLLECTIVE over it.
    full_window : bool
        Also return the velocity over EVERY pair of the fine window (the
        TD-aGW equation of motion needs the vv' and cc' blocks, not only
        c <- v): ``v_window`` and ``e_dft_window`` below.  Off by default;
        the default outputs do not depend on it.
    qp_states : bool
        The exact-position fix, opt-in (``dipole_position_fix``; the
        loader passes ``eqp_file is not None and velocity_position_fix``).
        Under ``eqp_corrections`` the fine states are eigenstates of a QUASIPARTICLE
        f(H); with strongly band-dependent QP shifts that interpolant mixes
        the velocity's DFT bands, the DFT f(H) block is not diagonal in them,
        and ``v/ΔE`` of its Rayleigh quotients is not their position (CrI3
        8→12: Σ|d|² 0.986, onset 0.87 against the native fine grid; the fix
        0.05 %).  Then the dipoles are the ``⟨c̃|r̂|ṽ⟩`` of
        :func:`_position_in_fine_states`, exact when the DFT f(H)
        interpolant is the more accurate one.  Off (default): ``v/ΔE_RQ``;
        Si 4³→8³, a near-scissor QP ladder, is 4x closer to the native grid
        this way (the DFT f(H) block mixes v↔c by 3.8 % there).
    mixing_receipt : bool
        Log the receipt that decides the opt-in (the loader passes
        ``eqp_file is not None``): max valence↔conduction mixing of the DFT
        f(H) eigenbasis in the fine states and the largest relative f(H)
        off-diagonal (:func:`_qp_window_mixing`).
    pair_rule, pair_gap_ev : str, float
        Same-occupation pairs of the window position (``r_window``, and the
        c̃ṽ dipoles where W mixes valence and conduction rows):
        ``mask`` (default) ``1 − exp(−ΔE²/(2 gap²))``, ``cut``, ``cv``;
        ``tdagw.td_system.position_from_velocity``'s rule and scale.

    Returns
    -------
    dict
        ``d_alpha`` (3, nk_f, nc, nv) complex128 host, bohr; ``de_cv``
        (nk_f, nc, nv) float64 host, Ry; ``q_frac`` (nk_f, 3), the fine k
        list both are indexed by; ``source`` (absolute path),
        ``velocity`` (the file's operator label), ``route``; ``on_grid_rel``
        and ``n_on_grid``; ``image_locality`` {image: (outer-shell L2
        fraction, outer-shell max / R=0)}.  With ``full_window``:
        ``v_window`` (3, nk_f, nb_fi, nb_fi) complex128 host, Ry·bohr,
        ``⟨i|v̂_α|j⟩`` (bra i, ket j, valence rows first, the same U_q
        contraction as ``d_alpha``), and ``e_dft_window`` (nk_f, nb_fi)
        float64 host, Ry, the velocity's own band energies (the Newton
        inversion above), so ``d_alpha = v_window[:, :, nv:, :nv] /
        (e_dft_window[:, nv:, None] - e_dft_window[:, None, :nv])`` on the
        DFT path; and ``r_window`` (3, nk_f, nb_fi, nb_fi) complex128 host,
        bohr, ``⟨ĩ|r̂_α|j̃⟩`` over every window pair in the same states (the
        pair rule above; no intraband position, i.e. a zero diagonal in the
        velocity's eigenbasis, which the QP path rotates into the fine
        states), so ``d_alpha = i r_window[:, :, nv:, :nv]`` on the QP path.  With ``qp_states``: ``d_alpha`` is the exact
        r-form (``route`` names :data:`DENSIFIED_POSITION_FORM`),
        ``d_alpha_rq`` the previous ``v/ΔE_RQ`` (diagnostic), ``rfix`` the
        rotation's record (off-block mixing, window-sum invariance,
        Σ|d|² after/before, the RQ shift of the lowest transition).
    """
    from bandstructure.fh_interp import (f_transform_eigs, image_locality,
                                         newton_inv, require_newton_converged)
    from bandstructure.orbital import interpolate_band_operator
    from common.collectives import device_put_process_local, gather_to_host
    from common.units import RYD_TO_EV
    from runtime.padding import padded_axis
    from symmetry_maps import unfold_file_wedge_polar_matrix

    parents, e_parent, label, path = velocity
    b0, b1 = (int(b) for b in basis.band_range)
    nb_fit = b1 - b0
    if int(parents.shape[-1]) < b1:
        raise ValueError(
            f"GATE densified_velocity_bands: {path} carries "
            f"{int(parents.shape[-1])} bands; want >= {b1}, the top of the "
            f"htransform window [{b0},{b1}); why: the velocity is carried "
            f"through every fitted band; fix: rerun psp.get_dipole_mtxels "
            f"with number_bands >= {b1}.")
    nk_co = int(np.prod(coarse_grid))
    v_full = unfold_file_wedge_polar_matrix(
        sym, np.asarray(parents)[:, :, b0:b1, b0:b1])         # (nk, 3, nb, nb)
    e_full = np.asarray(e_parent, dtype=np.float64)[
        np.asarray(sym.irr_idx_k)][:, b0:b1]                   # (nk, nb) Ry
    f_eps, a_f, n_f, shift = f_transform_eigs(
        jnp.asarray(e_full.T), a_band_index=a_band_index)
    f_full = np.asarray(f_eps, dtype=np.float64).T             # (nk, nb)
    # When the fit window's top band meets an unfitted band, v projected on
    # the window is not smooth in k (the projector jumps where they cross;
    # Si 4³, 36 fitted of 100 bands: v image outer shell 26 % against 1.9 %
    # for f(H)).  The velocity is then carried as g v g with g =
    # velocity_weight, zero above the first unfitted band's lowest energy,
    # and g is divided back in the fine window (_velocity_from_weighted).
    # An isolated window top keeps the unweighted route, bit for bit.
    nb_win = int(fine.coeffs_fi.shape[-1])
    e_all = np.asarray(e_parent, dtype=np.float64)
    e_cut = float(e_all[:, b1].min()) if e_all.shape[1] > b1 else np.inf
    weight = None
    if e_cut < float(e_all[:, b1 - 1].max()):
        e_win = float(np.sort(e_all[:, b0:b1], axis=1)[:, window_start + nb_win - 1].max())
        if not e_cut > e_win:
            raise ValueError(
                f"GATE densified_velocity_weight: the first band above the fit "
                f"window [{b0},{b1}) of {path} reaches {e_cut:.6f} Ry, at or "
                f"below the BSE window's top {e_win:.6f} Ry; why: the window's "
                f"own states then cross unfitted bands and no smooth weight "
                f"separates them; fix: raise ncond (more guard bands).")
        weight = (e_cut, 0.5 * (e_cut - e_win))
        g_full = velocity_weight(e_full, *weight)                 # (nk, nb)
        v_full = v_full * (g_full[:, None, :, None] * g_full[:, None, None, :])

    # Four band operators on one replicated carrier (48·nk·nb² B per
    # device, the htransform's own velocity layout): the three weighted
    # velocity components and f(H) of the velocity's energies.  Band axes
    # padded to the mesh so the image's (x, y) face divides.
    nbp = padded_axis(nb_fit, mesh_xy, name="densified velocity bands").carrier
    ops = np.zeros((4, nk_co, nbp, nbp), np.complex128)
    ops[:3, :, :nb_fit, :nb_fit] = np.moveaxis(v_full, 1, 0)
    ops[3][:, np.arange(nb_fit), np.arange(nb_fit)] = f_full
    ops = device_put_process_local(ops, NamedSharding(mesh_xy, P()))

    q_frac = np.asarray(fine.q_frac, dtype=np.float64)
    nq = int(q_frac.shape[0])
    coeffs = _coefficients_by_q(fine.coeffs_fi, padded_axis(
        nq, mesh_xy, name="densified dipole q carrier").carrier, mesh_xy)
    locality = {}

    def _record(component, operator_R):
        locality[_DIPOLE_IMAGES[component]] = image_locality(
            operator_R, tuple(coarse_grid))

    out = interpolate_band_operator(ops, basis.ctilde, coeffs, q_frac,
                                    tuple(coarse_grid), mesh_xy,
                                    B_at_mu=basis.basis_at_nodes,
                                    image_record_fn=_record)
    del coeffs
    if probe is not None:
        q_probe = np.asarray(probe.q_frac, dtype=np.float64)
        out_probe = interpolate_band_operator(
            ops, basis.ctilde, _coefficients_by_q(probe.coeffs_fi, padded_axis(
                int(q_probe.shape[0]), mesh_xy, name="dipole probe q carrier").carrier,
                mesh_xy),
            q_probe, tuple(coarse_grid), mesh_xy, B_at_mu=basis.basis_at_nodes)
    del ops
    win = np.asarray(gather_to_host(jax.jit(lambda o: o[:nq])(out)))  # (nq, 4, nb, nb)
    f_win = win[:, 3]
    v_win, weight_floor = np.moveaxis(win[:, :3], 1, 0), 1.0
    if weight is not None:
        def _g_of_f(phi):
            e, res = newton_inv(a_f, n_f, shift, jnp.asarray(phi))
            require_newton_converged(float(res), where="densified_dipoles weight")
            return velocity_weight(np.asarray(e), *weight)
        v_win, weight_floor = _velocity_from_weighted(v_win, f_win, _g_of_f,
                                                      where=str(path))
    del win
    v_cv = v_win[:, :, n_val:, :n_val]                          # (3, nq, nc, nv)
    lam = np.real(np.einsum("qii->qi", f_win))
    e_dft, residual = newton_inv(a_f, n_f, shift, jnp.asarray(lam))
    require_newton_converged(float(residual), where="densified_dipoles")
    e_dft = np.asarray(e_dft, dtype=np.float64)                 # (nq, nb_fi)
    de_cv = e_dft[:, n_val:, None] - e_dft[:, None, :n_val]     # (nq, nc, nv)
    if not (np.all(np.isfinite(v_cv)) and float(de_cv.min()) > 0.0):
        raise ValueError(
            f"GATE densified_dipole_gap: got min (E_c - E_v) = "
            f"{float(de_cv.min()) * RYD_TO_EV:.6g} eV over the fine "
            f"grid in {path}'s energies (finite velocity: "
            f"{bool(np.all(np.isfinite(v_cv)))}); want every transition "
            f"positive and finite; why: r = v/(E_c - E_v) is undefined at a "
            f"band inversion of the velocity's own Hamiltonian; fix: a window "
            f"whose conduction bands lie above its valence bands in that "
            f"Hamiltonian (dipole_qsgw.h5 for a QSGW state).")
    d_alpha = v_cv / de_cv[None]                                # (3, nq, nc, nv)
    if pair_rule not in DENSIFIED_POSITION_PAIR_RULES:
        raise ValueError(
            f"GATE densified_pair_rule: got {pair_rule!r}; want one of "
            f"{' | '.join(DENSIFIED_POSITION_PAIR_RULES)}; why: the rule "
            f"decides the same-occupation pairs of the window position; fix: "
            f"pass pair_rule='mask' (default).")
    gap_ry = float(pair_gap_ev) / RYD_TO_EV
    d_alpha_rq = rfix = r_window = None
    if qp_states:
        def _newton(phi):
            e, res = newton_inv(a_f, n_f, shift, jnp.asarray(phi))
            require_newton_converged(float(res), where="densified_dipoles r-fix")
            return np.asarray(e, dtype=np.float64)
        r_window, eps_eig, offblock, cv_eig_sum = _position_in_fine_states(
            v_win, f_win, n_val, _newton, rule=pair_rule, gap_ry=gap_ry)
        de_eig = eps_eig[:, n_val:, None] - eps_eig[:, None, :n_val]
        if not float(de_eig.min()) > 0.0:
            raise ValueError(
                f"GATE densified_position_gap: got min (eps_c - eps_v) = "
                f"{float(de_eig.min()) * RYD_TO_EV:.6g} eV in the DFT "
                f"eigenbasis of the fine f(H) block; want > 0 at every q; why: "
                f"r = v/(i dE) needs the window's conduction rows above its "
                f"valence rows in the velocity's Hamiltonian; fix: a window "
                f"whose QP and DFT orders agree across the gap.")
        d_alpha_rq = d_alpha
        d_alpha = 1j * r_window[:, :, n_val:, :n_val]
        f_off = float(np.max(np.linalg.norm(
            f_win - np.einsum("qii->qi", f_win)[..., None] * np.eye(f_win.shape[-1]),
            axis=(1, 2)) / np.maximum(np.linalg.norm(f_win, axis=(1, 2)), 1e-300)))
        sum_fix = float(np.sum(np.abs(d_alpha) ** 2))
        sum_rq = float(np.sum(np.abs(d_alpha_rq) ** 2))
        low_shift = (de_cv.min(axis=(1, 2)) - de_eig.min(axis=(1, 2))) * RYD_TO_EV * 1e3
        rfix = {"applied": True, "pair_rule": pair_rule, "pair_gap_ev": float(pair_gap_ev),
                "max_offblock_mixing": float(offblock.max()),
                "fh_offdiag_rel_max": f_off,
                "window_sum_eig_over_fixed": float(cv_eig_sum / sum_fix) if sum_fix > 0 else float("nan"),
                "sum_d2_fixed_over_rq": float(sum_fix / sum_rq) if sum_rq > 0 else float("nan"),
                "lowest_dE_rq_minus_eig_meV_mean": float(low_shift.mean()),
                "lowest_dE_rq_minus_eig_meV_max": float(np.abs(low_shift).max())}
        log_fn(f"[kgrid_fine dipoles] r-fix (fine states of a QP f(H)): r from the "
               f"DFT eigenbasis of their f(H) block, rotated back; max off-block "
               f"(v<->c) mixing {rfix['max_offblock_mixing']:.3e}; f(H) off-diagonal "
               f"max {f_off:.3e} (relative); window sum |d|^2 eigenbasis/fixed "
               f"{rfix['window_sum_eig_over_fixed']:.12f}; sum |d|^2 fixed/RQ "
               f"{rfix['sum_d2_fixed_over_rq']:.6f}; lowest dE RQ - eigen mean "
               f"{rfix['lowest_dE_rq_minus_eig_meV_mean']:.3f} meV, max "
               f"{rfix['lowest_dE_rq_minus_eig_meV_max']:.3f} meV")
    elif full_window:
        r_window = _position_from_velocity_rule(v_win, e_dft, n_val,
                                                rule=pair_rule, gap_ry=gap_ry)
    mixing = None
    if mixing_receipt and not qp_states:
        mixing = _qp_window_mixing(f_win, n_val)
        log_fn(f"[kgrid_fine dipoles] eqp_corrections without dipole_position_fix: d = "
               f"v/dE_RQ; QP/DFT window mixing receipt: max off-block (v<->c) "
               f"mixing {mixing[0]:.3e}, f(H) off-diagonal max {mixing[1]:.3e} "
               f"(relative); large values (strongly band-dependent QP shifts, "
               f"CrI3-type) call for dipole_position_fix")

    if probe is None:
        on_grid_rel, state_dev, defect, n_on_grid = _dipole_on_grid_identity(
            out, basis.ctilde, fine.coeffs_fi, ops_host=(v_full, f_full),
            k_coarse=sym.unfolded_kpts, q_fine=q_frac, window_start=window_start,
            fh_energies=fh_energies)
    else:
        on_grid_rel, state_dev, defect, n_on_grid = _dipole_on_grid_identity(
            out_probe, basis.ctilde, probe.coeffs_fi, ops_host=(v_full, f_full),
            k_coarse=sym.unfolded_kpts, q_fine=q_probe, window_start=window_start,
            fh_energies=fh_energies)
        del out_probe
    v_window = v_win if full_window else None                   # (3, nq, nb, nb)
    del out, f_win
    shift_mev = float(np.max(np.abs(
        e_dft - np.asarray(gather_to_host(fine.enk_full))))) * RYD_TO_EV * 1e3
    log_fn(f"[kgrid_fine dipoles] {path} ({label}) carried through the "
           f"htransform on bands [{b0},{b1}): d_alpha {d_alpha.shape} in the "
           f"densified ψ gauge (the bundle's own U_q)")
    state_tol = DENSIFIED_DIPOLE_STATE_FLOOR + 10.0 * defect
    log_fn(f"[kgrid_fine dipoles] on-grid identity at {n_on_grid} fine q "
           f"that are coarse k: states max|σ(C*U) - 1| = {state_dev:.3e} "
           f"(gate {state_tol:.1e} = 1e-8 + 10 max|C Cᴴ - I|), images max rel "
           f"deviation {on_grid_rel:.3e} (gate "
           f"{DENSIFIED_DIPOLE_ONGRID_RTOL:.0e})")
    log_fn(f"[kgrid_fine dipoles] transition energies from {path}'s "
           f"band_energies (Rayleigh quotients of its f(H) in the fine "
           f"states): max |E - eps_bundle| = {shift_mev:.3f} meV (the eqp_corrections "
           f"shift; 0 without eqp_corrections)")
    log_fn("[kgrid_fine dipoles] velocity carried unweighted (fit window top isolated)"
           if weight is None else
           f"[kgrid_fine dipoles] velocity carried as g v g (fit window top meets "
           f"band {b1} at {weight[0]:.6f} Ry; width {weight[1]:.6f} Ry); window "
           f"weight min/max {weight_floor:.6f} (gate {DENSIFIED_VELOCITY_WEIGHT_FLOOR:.0e})")
    log_fn("[kgrid_fine dipoles] image locality, outer-shell L2 fraction "
           "(diagnostic): " + ", ".join(
               f"{name} {locality[name][0]:.3e}" for name in _DIPOLE_IMAGES
               if name in locality))
    if not (state_dev <= state_tol
            and on_grid_rel <= DENSIFIED_DIPOLE_ONGRID_RTOL):
        raise ValueError(
            f"GATE densified_dipole_on_grid: got max|σ(C*U) - 1| = "
            f"{state_dev:.3e} and image deviation {on_grid_rel:.3e} at the "
            f"{n_on_grid} fine q that are coarse k; want <= {state_tol:.1e} "
            f"and <= {DENSIFIED_DIPOLE_ONGRID_RTOL:.0e} (identities of the "
            f"interpolant); why: the fine q list, the band window or the "
            f"contraction convention disagrees with the htransform that built "
            f"ψ, so the dipoles are not in its gauge; fix: report it, this is "
            f"a bookkeeping defect.")
    result = {
        "d_alpha": d_alpha,
        "de_cv": de_cv,
        "q_frac": q_frac,
        "source": os.path.abspath(str(path)),
        "velocity": str(label),
        "route": "kgrid_fine htransform velocity (bse_densify.densified_dipoles)"
                 + (f" + {DENSIFIED_POSITION_FORM}" if qp_states else ""),
        "on_grid_rel": float(on_grid_rel),
        "on_grid_state_dev": float(state_dev),
        "n_on_grid": int(n_on_grid),
        "image_locality": locality,
        "velocity_weight": None if weight is None else {
            "e_cut_ry": weight[0], "width_ry": weight[1], "floor": float(weight_floor)},
    }
    if full_window:
        result["v_window"] = v_window
        result["e_dft_window"] = e_dft
        result["r_window"] = r_window
        result["pair_rule"] = pair_rule
        result["pair_gap_ev"] = float(pair_gap_ev)
    if qp_states:
        result["d_alpha_rq"] = d_alpha_rq
        result["rfix"] = rfix
    if mixing is not None:
        result["qp_mixing"] = {"max_offblock_mixing": mixing[0],
                               "fh_offdiag_rel_max": mixing[1]}
    return result


def _coefficients_by_q(coeffs, carrier, mesh_xy):
    """``coeffs`` (nq, rank, nb) zero-padded to ``carrier`` q rows, q-sharded.

    ``compute_wfns_fi`` hands its eigenvectors over face-sharded,
    ``P(None, 'x', 'y')``; ``interpolate_band_operator`` reads them
    ``P(('x','y'), None, None)``, one whole matrix per q per device.  The pad
    is local on the face and the move is the staged two-``all_to_all``
    reshard (``common.staged_reshard``), never a replicate-then-repartition;
    a shape it cannot serve takes the plain constraint.
    """
    from common.staged_reshard import (face_to_batch_reshard,
                                       face_to_batch_reshard_supported)
    padded = jax.jit(
        lambda c: jnp.pad(c, ((0, carrier - c.shape[0]), (0, 0), (0, 0))),
        out_shardings=NamedSharding(mesh_xy, P(None, "x", "y")))(coeffs)
    if face_to_batch_reshard_supported(mesh_xy, padded.shape):
        return face_to_batch_reshard(mesh_xy)(padded)
    return jax.jit(lambda c: c, out_shardings=NamedSharding(
        mesh_xy, P(("x", "y"), None, None)))(padded)


def _dipole_on_grid_identity(out, ctilde, coeffs, *, ops_host, k_coarse,
                             q_fine, window_start, fh_energies):
    """The two on-grid identities of :func:`densified_dipoles`.

    ``out`` (carrier, 4, nb_fi, nb_fi) is ``interpolate_band_operator``'s
    output, ``ctilde`` (nk_co, nb_fit, rank), ``coeffs`` (nk_f, rank, nb_fi),
    ``ops_host = (v_full (nk_co, 3, nb_fit, nb_fit), f_full (nk_co,
    nb_fit))``.  A fine q matches the coarse row whose ``k_coarse`` equals it
    modulo a reciprocal lattice vector (to 1e-8), never by assumed order.
    With ``W = C_k^* U_q`` (nb_fit, nb_fi), the rows ``S_k`` of the coarse
    states at positions ``[window_start, window_start + nb_fi)`` of
    ``fh_energies[k]`` sorted (every state within
    :data:`DENSIFIED_DIPOLE_DEGEN_RY` of the two edge levels included)
    have ``σ(W_S) = 1``, and the four images are ``W^† O(k) W``.  Only
    the shared rows are gathered.

    Returns ``(image rel deviation, max|σ(W_window) − 1|, max|C Cᴴ − I| over
    the shared k, n_shared)``.
    """
    from common.collectives import gather_to_host

    v_full, f_full = ops_host
    kc = np.asarray(k_coarse, dtype=np.float64)
    pairs = []
    for i, q in enumerate(np.asarray(q_fine, dtype=np.float64)):
        d = kc - q[None, :]
        d -= np.round(d)
        hit = np.flatnonzero(np.all(np.abs(d) < 1e-8, axis=1))
        if hit.size:
            pairs.append((i, int(hit[0])))
    if not pairs:
        raise ValueError(
            "GATE densified_dipole_on_grid: no fine q coincides with a coarse "
            "k (Γ is on every uniform grid); want the fine and coarse grids "
            "of one zone; why: the identity cannot be read; fix: report it.")
    fi = np.asarray([p[0] for p in pairs])
    co = np.asarray([p[1] for p in pairs])
    U = np.asarray(gather_to_host(coeffs[fi]))                  # (n, rank, nb_fi)
    C = np.asarray(gather_to_host(ctilde[co]))                  # (n, nb_fit, rank)
    O = np.asarray(gather_to_host(out[fi]))                     # (n, 4, nb_fi, nb_fi)
    nb_fi = int(U.shape[2])
    W = np.einsum("nma,nai->nmi", np.conj(C), U)                # ⟨ψ_m(k)|ψ_i(q)⟩
    e_fh = np.asarray(fh_energies, dtype=np.float64)[co]          # (n, nb_fit)
    sigma = []
    for n_, e_k in enumerate(e_fh):
        e_sorted = np.sort(e_k)
        lo = e_sorted[window_start] - DENSIFIED_DIPOLE_DEGEN_RY
        hi = e_sorted[window_start + nb_fi - 1] + DENSIFIED_DIPOLE_DEGEN_RY
        rows = np.flatnonzero((e_k >= lo) & (e_k <= hi))
        sigma.append(np.linalg.svd(W[n_, rows], compute_uv=False))
    sigma = np.concatenate(sigma)
    gram = np.einsum("nma,npa->nmp", C, np.conj(C))
    defect = float(np.max(np.abs(gram - np.eye(gram.shape[1])[None])))
    v_ref = np.einsum("nmi,nxmp,npj->nxij", np.conj(W), v_full[co], W)
    f_ref = np.einsum("nmi,nm,nmj->nij", np.conj(W), f_full[co], W)
    ref = np.concatenate([v_ref, f_ref[:, None]], axis=1)
    scale = np.linalg.norm(ref.reshape(len(pairs), -1), axis=1)
    dev = np.linalg.norm((O - ref).reshape(len(pairs), -1), axis=1)
    return (float(np.max(dev / np.where(scale > 0, scale, 1.0))),
            float(np.max(np.abs(sigma - 1.0))), defect, len(pairs))


def _interpolate_bse_data_to_grid(
    data: dict,
    fine_grid: tuple[int, int, int],
    restart_file: str,
    input_file: str,
    mesh_xy: Mesh,
    *,
    head_channel: dict | None = None,
    distrib_la_batched_route: str | None = None,
    htransform_a_band: int | None = None,
    htransform_rank_record_fn=None,
    htransform_quality_record_fn=None,
    eqp_file: str | None = None,
    velocity_file: str | None = None,
    velocity_full_window: bool = False,
    velocity_pair_rule: str = "mask",
    velocity_pair_gap_ev: float = 0.05,
    velocity_position_fix: bool = False,
    fine_shift=(0.0, 0.0, 0.0),
    log_fn=log.debug,
) -> dict:
    """Interpolate the WHOLE coarse BSE ``data`` bundle onto ``fine_grid``.

    ``fine_shift`` is ``kgrid_fine_shift`` (kgrid.x's dk, fractions of one
    fine step): ψ and ε are evaluated at k_i = (i + dk)/N.  Only k − k' enters
    the direct term, and that difference does not depend on dk, so W stays on
    the unshifted fine q grid; the Q = 0 exchange tile is k-independent.

    The single coarse→fine densification the ``kgrid_fine`` knob drives, living
    in the GENERAL init so EVERY solver consumes a fine-grid bundle unchanged.
    Each piece goes through the shared builder its exciton_bands sibling uses
    (until 2026-07-31 the W leg was a SECOND, eager copy — ``local_ifftn3``/
    ``local_fftn3`` outside any shard_map, all-gathering the (μ,ν)-sharded W
    per rank; audit P0-4):

      * ψ_{v,c}(k) and QP ε_{v,c}(k) on the fine mesh ← ONE htransform fH
        (``bandstructure.htransform.initialize_wfns`` +
        ``bandstructure.bse_setup.compute_wfns_fi`` with ``kgrid_fine=fine_grid``);
        the fH is built over the full loaded band window and only the BSE
        sub-window [nval−n_val, nval+n_cond) is returned (interior, guarded).
      * V_Q exchange q=0 tile ← CARRIED THROUGH unchanged by default.  A Q=0
        exciton's exchange is the single q=0 tile (dense in k,k'), and that
        tile's BODY is built from the centroids and the G-sphere alone, so it
        is k-grid-INVARIANT: densifying it is unnecessary.  Only the rank-1
        head scalar ``<v>_mBZ`` is grid-dependent (the cell is BZ/N_k).  The
        deck's ``head_minibz_average`` key (default off) chooses: OFF keeps the
        coarse bundle's tile exactly, deck ``vhead`` and all; ON rebuilds it
        through ``bse.vq_interp.build_vq_evaluator`` (the SAME builder
        exciton_bands calls) at Q=0 with the FINE mini-BZ head
        (``minibz_head_vlr(..., kgrid=fine_grid)``), replacing the disk body
        and the deck head.  Until 2026-08-09 this path forced the ON branch
        unconditionally, ignoring the key — see the block comment below.
      * W direct ← ``make_w_densifier(output='k')`` — the ONE sharded
        coarse→fine densifier (shard_map FFTs + jitted per-pair R embedding, (μ,ν)
        sharding preserved end to end); the exciton_bands ``--w-coarse-grid``
        flag runs the SAME factory with ``output='R'``.  ``kgrid_fine`` drives
        it automatically (the banner below is the "W-pad fired" proof).
        Its operand is the head-EXCLUDED body and the Γ head is re-attached
        analytically afterwards — see ``head_channel`` below.

    Band dims (n_val/n_cond and their mesh-pads), n_rmu(_pad), and the q=0 head
    projectors g0 are grid-INVARIANT and carried through untouched; only the k
    axis (and W's k-axes) change.  Returns a new bundle with the SAME keys.

    Parameters
    ----------
    eqp_file : str or None
        The wedge ``eqp1.dat`` (``eqp_corrections``).  Its QP energies enter the fH
        through ``initialize_wfns``, the one reader on this path, so the fine
        ε are quasiparticle; the coarse bundle's ε are not re-sliced after.
    velocity_file : str or None
        The COARSE WFN's ``dipole.h5`` (``dipole_file``).  Authenticated against
        the deck WFN before the fit, then carried through the same htransform
        as ψ (:func:`densified_dipoles`); the bundle's ``dipoles`` key holds
        the fine-grid window dipoles in its own ψ gauge, ``None`` without it.
    velocity_full_window : bool
        Pass ``full_window`` to :func:`densified_dipoles`, so ``dipoles``
        also holds the velocity over every window pair (TD-aGW).
    velocity_pair_rule, velocity_pair_gap_ev : str, float
        :func:`densified_dipoles`' ``pair_rule`` / ``pair_gap_ev`` (the
        same-occupation pairs of ``r_window``).
    velocity_position_fix : bool
        With ``eqp_file``: the exact-position fix of
        :func:`densified_dipoles` (``qp_states``), opt-in
        (``dipole_position_fix``); off, ``v/ΔE_RQ`` and the mixing receipt.
    head_channel : dict or None
        C1's split-and-reattach hand-off from the loader, ``None`` on the
        ``legacy`` arm.  When present it carries ``whead`` (the head scalar the
        loader would have injected at q=0, Ry·bohr³), ``cell_volume`` (Ω), and
        optionally ``gamma_cell`` (``"fine"`` = C1, ``"coarse"`` = the red
        twin).  Its presence is also the SIGNAL that the loader deferred the
        injection, so the W tile in ``data`` is the pre-injection body; the two
        must not disagree, which is why one dict carries both the decision and
        the numbers rather than a flag beside a value.
    """
    from bandstructure import fh_interp as ht
    from bandstructure.bse_setup import compute_wfns_fi
    from gw.gw_config import (
        read_lorrax_input,
        resolve_distrib_la_batched_route,
    )

    from . import vq_interp

    coarse_grid = (int(data["nkx"]), int(data["nky"]), int(data["nkz"]))
    fine_grid = tuple(int(s) for s in fine_grid)
    for a, (f, c) in enumerate(zip(fine_grid, coarse_grid)):
        if c <= 0 or f < c:
            raise ValueError(
                f"kgrid_fine axis {a}: fine {f} must be at least the coarse "
                f"restart extent {c}; trigonometric densification cannot "
                f"coarsen an axis.")
    n_val = int(data["n_val"]); n_cond = int(data["n_cond"])
    nv_pad = int(data["n_val_pad"]); nc_pad = int(data["n_cond_pad"])
    n_rmu = int(data["n_rmu"]); n_rmu_pad = int(data["n_rmu_pad"])
    nk_f = fine_grid[0] * fine_grid[1] * fine_grid[2]
    log_fn(f"[kgrid_fine] coarse {coarse_grid[0]}x{coarse_grid[1]}x{coarse_grid[2]}"
           f" → fine {fine_grid[0]}x{fine_grid[1]}x{fine_grid[2]} "
           f"({nk_f} k-pts); interpolating ψ/ε (htransform), V_Q (vq_interp), "
           f"W (per-pair Wigner-Seitz embedding in R)")

    params = read_lorrax_input(input_file)
    _distrib_la_batched_route = resolve_distrib_la_batched_route(
        params, override=distrib_la_batched_route)
    fine_shift = tuple(float(x) for x in fine_shift)
    nc_coarse = coarse_conduction_count(params, n_cond, log_fn=log_fn)
    n_extra_guard = 0
    if nc_coarse is not None:
        # The H-transform's window is the deck's [nelec - nval, nelec + ncond);
        # number_cond_bands_coarse narrows its conduction side, or widens it
        # with guard bands loaded above ncond (initialize_wfns' one lever).
        n_extra_guard = max(0, nc_coarse - int(params["ncond"]))
        params = dict(params, ncond=min(int(params["ncond"]), nc_coarse))

    # The restart and fresh htransform WFN first meet at this seam.
    from file_io.qp_wfn import refuse_conflicting_qp_state_sources
    from .bse_window import _parse_wfn_path
    refuse_conflicting_qp_state_sources(
        wfn_path=_parse_wfn_path(input_file),
        state_artifact_path=restart_file,
        where="kgrid_fine restart/htransform state")

    # ── WHICH ISDF BASIS THE htransform FITS IN ───────────────────────────
    # On a NATIVE bundle: the deck's own table, and ``keep`` is None — this
    # block is then a no-op and everything below is the program it always was.
    #
    # On a DOWNFOLDED bundle the two questions come apart, and the densifier
    # has to answer them through ``exciton_bands.resolve_isdf_basis``, the ONE
    # owner of this contract.  The nodal fit evaluates its basis at the
    # ordered ``keep`` rows, so it is born at μ_S.
    # Without that route, the pad below is asked for
    # (μ_S_pad − μ_L) columns — NEGATIVE — and the run dies in ``jnp.pad``
    # with an index error that names neither the downfold nor the basis.
    # Same defect class as PIPELINE_HEALTH row 4, which closed this for
    # exciton_bands and never reached the ``kgrid_fine`` path.
    from .exciton_bands import resolve_isdf_basis
    centroids_path, keep_idx = resolve_isdf_basis(
        restart_file, params, input_file, n_rmu_bundle=n_rmu, log=log_fn)
    params["centroids_file"] = centroids_path
    # HOST numpy closed over the trace, not a device_put: an eager device_put
    # of a host array is the hidden-all-gather site; a numpy array becomes an
    # HLO constant and is process-local by construction.
    keep = None if keep_idx is None else np.asarray(keep_idx, dtype=np.int32)

    # ── The coarse velocity, authenticated BEFORE the fit ────────────────
    # It must be the deck WFN's own (basis and fingerprint): a fine grid's
    # dipole.h5 has the bundle's nk and the WRONG gauge, and refuses here.
    wfn_sym = velocity = None
    if velocity_file is not None:
        from bandstructure.orbital import stored_velocity
        wfn_sym = ht.setup_wfn_and_sym(_parse_wfn_path(input_file),
                                       mesh_xy=mesh_xy)
        velocity = stored_velocity(
            velocity_file, wfn=wfn_sym[0], wfn_path=_parse_wfn_path(input_file),
            sym=wfn_sym[1], mesh=mesh_xy) + (velocity_file,)

    # ── ψ_{v,c}(k_fine), ε_{v,c}(k_fine): ONE htransform fH ───────────────
    _fit_subset = keep
    (wfn, sym, meta, _mesh, basis,
     enk_sigma) = ht.initialize_wfns(
         input_file, params, log_fn, mesh_xy=mesh_xy, eqp_file=eqp_file,
         centroid_subset_idx=_fit_subset,
         rank_record_fn=htransform_rank_record_fn,
         wfn_sym=wfn_sym, n_guard_bands=n_extra_guard)
    ctilde, B_at_mu = basis.ctilde, basis.basis_at_nodes
    nb_window = int(ctilde.shape[1])
    nval_in = int(params["nval"])          # window-relative CBM index
    b_min, b_max = nval_in - n_val, nval_in + n_cond
    if b_min < 0 or b_max > nb_window:
        raise ValueError(
            f"kgrid_fine BSE window [{b_min},{b_max}) escapes the htransform fH "
            f"window [0,{nb_window}); the deck's nval and "
            f"number_cond_bands_coarse (or ncond) must load >= {n_val} valence "
            f"and >= {n_cond} conduction bands.")
    # SPLASH RADIUS OF THE f-SHOULDER (``2026-08-11-fifth-wall-is-the-f-\
    # transform-shoulder.md`` §7, audited here).  The check above refuses
    # ``b_max > nb_window`` and so PERMITS ``b_max == nb_window`` — a deck
    # whose ``nband`` equals ``nval + ncond`` has no conduction guard at all
    # and lands exactly where the refit did: f(eps) is identically zero at and
    # above ``max_k eps`` of the window's own top band, so the top of the BSE
    # conduction selection comes back from eigh as a null-space direction.
    # ``compute_wfns_fi``'s f-shoulder gate refuses; this says it first, in
    # the deck's own vocabulary.
    _n_guard = nb_window - b_max
    if _n_guard < 4:
        log.warning(f"kgrid_fine: only {_n_guard} conduction guard "
                    f"band(s) above the BSE window [{b_min},{b_max}) inside the "
                    f"{nb_window}-band htransform window.  The f-transform's shift "
                    f"is max_k eps of THAT window's top band, so f == 0 there and "
                    f"the shoulder below it carries ~1% of fH's weight; the top of "
                    f"the returned window may be arbitrary.  Raise "
                    f"number_cond_bands_coarse (or ncond).",
                    extra=message(
                        id="bse_coarse_guard_bands_few",
                        found=("number_cond_bands_coarse - number_cond_bands_fine",
                               str(_n_guard)),
                        needs="at least 4",
                        fix="raise number_cond_bands_coarse (or ncond)"))
    bundle = compute_wfns_fi(
        ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
        kgrid_co=coarse_grid, kgrid_fine=fine_grid, kgrid_shift=fine_shift,
        band_window_fi=(b_min, b_max), mesh_xy=mesh_xy, log_fn=log_fn,
        a_band_index=htransform_a_band,
        batch_size=int(params.get("wfn_fi_q_chunk", 0)),
        htransform_quality_record_fn=htransform_quality_record_fn,
        return_coeffs=velocity is not None)
    dipoles = None
    if velocity is not None:
        from common.collectives import gather_to_host
        probe = None
        if any(fine_shift):
            # No point of a shifted grid is a coarse k, so the interpolant's
            # on-grid identities are read on a probe batch AT the coarse k.
            probe = compute_wfns_fi(
                ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
                kgrid_co=coarse_grid, q_list=np.asarray(sym.unfolded_kpts),
                band_window_fi=(b_min, b_max), mesh_xy=mesh_xy, log_fn=log_fn,
                a_band_index=htransform_a_band,
                batch_size=int(params.get("wfn_fi_q_chunk", 0)),
                return_coeffs=True)
        dipoles = densified_dipoles(
            velocity, sym=sym, basis=basis, fine=bundle,
            fh_energies=np.asarray(gather_to_host(enk_sigma)).T,
            coarse_grid=coarse_grid, n_val=n_val, window_start=b_min,
            a_band_index=htransform_a_band, mesh_xy=mesh_xy, log_fn=log_fn,
            full_window=velocity_full_window,
            qp_states=eqp_file is not None and bool(velocity_position_fix),
            pair_rule=velocity_pair_rule, pair_gap_ev=velocity_pair_gap_ev,
            mixing_receipt=eqp_file is not None, probe=probe)
        del probe
        del bundle.coeffs_fi, velocity

    x4 = NamedSharding(mesh_xy, P(None, None, None, "x"))
    y4 = NamedSharding(mesh_xy, P(None, None, None, "y"))
    rep = NamedSharding(mesh_xy, P())

    @partial(jax.jit, out_shardings=(x4, y4, x4, y4, rep, rep))
    def _split_pad(psi, enk):
        # psi (nk_f, n_val+n_cond, ns, n_mu); enk (nk_f, n_val+n_cond).
        # ``initialize_wfns`` evaluated the sole nodal basis directly
        # on the resolved child centroid rows, so ``compute_wfns_fi`` and this
        # split both receive the restart's final μ width.  There is no
        # parent-width projected-wavefunction route here.
        psi_v = psi[:, :n_val]
        psi_c = psi[:, n_val:n_val + n_cond]
        psi_v = jnp.pad(psi_v, ((0, 0), (0, nv_pad - n_val), (0, 0),
                                (0, n_rmu_pad - psi.shape[3])))
        psi_c = jnp.pad(psi_c, ((0, 0), (0, nc_pad - n_cond), (0, 0),
                                (0, n_rmu_pad - psi.shape[3])))
        # ε pad = SIGNED SENTINEL, not zero (module note): ΔE = ε_c − ε_v is
        # the diagonal of the operator the BSE drivers diagonalise.
        eps_v = jnp.pad(enk[:, :n_val], ((0, 0), (0, nv_pad - n_val)),
                        constant_values=-PAD_EPS_GUARD_RY)
        eps_c = jnp.pad(enk[:, n_val:n_val + n_cond],
                        ((0, 0), (0, nc_pad - n_cond)),
                        constant_values=PAD_EPS_GUARD_RY)
        return (jax.lax.with_sharding_constraint(psi_v, x4),
                jax.lax.with_sharding_constraint(psi_v, y4),
                jax.lax.with_sharding_constraint(psi_c, x4),
                jax.lax.with_sharding_constraint(psi_c, y4),
                eps_v, eps_c)

    psi_v_X, psi_v_Y, psi_c_X, psi_c_Y, eps_v, eps_c = _split_pad(
        bundle.psi_rmu_Y, bundle.enk_full)
    M = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(psi_c_X, psi_v_X),
        NamedSharding(mesh_xy, P(None, "x", "y", None)))
    # Fine states, M and optional dipoles retain the interpolation results.
    # Drop the fitted basis and unsplit bundle
    # before the V/W interpolation allocates its outputs and scratch.
    del basis, ctilde, B_at_mu, enk_sigma, bundle

    # ── V_Q exchange q=0 tile on the fine grid ────────────────────────────
    # A Q=0 exciton's exchange kernel is DENSE in (k,k') through the ONE q=0
    # tile (the since-deleted bse_serial single-device matvec); so the fine
    # k-grid's exchange q-set is just q=0.
    #
    # WHAT IS AND IS NOT GRID-DEPENDENT.  The q=0 exchange BODY
    # V_{μν} = Σ_{G≠0} conj(ζ̃_{0,μ}(G)) v(G) ζ̃_{0,ν}(G) is built from the
    # centroids and the G-sphere alone — it never sees the k-grid, so it needs
    # no densification and the coarse bundle's tile is already the fine grid's
    # answer.  The ONE k-grid-dependent piece is the rank-1 HEAD scalar, whose
    # mini-BZ cell average <v>_mBZ is taken over BZ/N_k and therefore shrinks
    # coarse→fine.  That rescale is what the original rebuild (964c682b,
    # 2026-07-20) was after, and its comment said so.
    #
    # It bought the rescale by forcing ``head_minibz_average=True`` and
    # REPLACING the whole tile, which cost two things the deck never agreed to:
    # (i) the opt-in mini-BZ head average — documented in KNOWN_FAILURES as
    # averaging the WRONG MOMENT (a scalar <v> where the nonanalytic head needs
    # the 3×3 second moment <v q_a q_b>) — reached every coarse→fine user who
    # had not set the key; and (ii) the deck's exact disk body and its
    # loader-injected ``vhead`` rank-1 head (``bse_head._inject_q0_head``)
    # were discarded and substituted by an LR-only model reconstruction.
    #
    # Both now follow the deck's ``head_minibz_average`` key, default off, like
    # every other reader of that key in the tree (``exciton_bands.main``'s
    # ``head_mbz``, ``head_correction.HeadResolver``'s ``analytic_sphere``,
    # ``vq_interp.build_vq_evaluator``).  What the OFF arm gives
    # up is only the head RESCALE, and at Q=0 the head is annihilated by
    # ⟨u_ck|u_vk⟩ = 0 up to the ISDF orthogonality residual
    # (LT_HEAD_PROBLEM.md §2.1) — so a stale head SCALE is inert to that
    # residual, while a replaced BODY is not.
    head_mbz = bool(params.get("head_minibz_average", False))
    if head_mbz:
        # Opt-in: rebuild the tile with the FINE mini-BZ head via the SAME
        # vq_interp builder exciton_bands uses; the body is the b26p/stencil
        # model reproduced at the q=0 training point (run_nulls certifies the
        # reproduction).  This arm is bit-for-bit the pre-2026-08-09 behaviour.
        vqm = vq_interp.build_vq_evaluator(
            restart_file, mesh_xy, n_rmu_pad, input_file=input_file, head_minibz_average=True,
            distrib_la_batched_route=_distrib_la_batched_route,
            log_fn=log_fn,
            run_diagnostics=bool(params["sigma_freq_debug_output"]))
        gstar, head_val = vq_interp.minibz_head_vlr(
            vqm.zx, vqm.prep, np.zeros(3), kgrid=fine_grid)
        V_q0 = vqm.eval_vq(jnp.zeros(3), vqm.prep["V_SRc"], vqm.pinvF,
                           vqm.coeffs_packed,
                           jnp.asarray(head_val, dtype=jnp.float64),
                           jnp.asarray(gstar, dtype=jnp.int32))
        V_q0 = 0.5 * (V_q0 + jnp.conj(V_q0).T)   # Hermitize (stencil residue)
        V_q0 = jax.device_put(V_q0, NamedSharding(mesh_xy, P("x", "y")))
        log_fn(f"[kgrid_fine] V_q0 exchange tile REBUILT via vq_interp "
               f"eval_vq(Q=0), fine mini-BZ head <v_LR>={head_val:.4f} "
               f"(gstar={gstar}) — head_minibz_average=true (deck opt-in).  "
               f"The deck's disk body and vhead rank-1 head are REPLACED; that "
               f"average is the scalar <v>, not the moment tensor the head "
               f"needs (tests/KNOWN_FAILURES.md).")
    else:
        V_q0 = data["V_q0"]
        log_fn("[kgrid_fine] V_q0 exchange tile CARRIED THROUGH from the "
               "coarse bundle unchanged (head_minibz_average off = the "
               "default): the q=0 body is k-grid-independent and the deck's "
               "loader-injected vhead rank-1 head is preserved exactly.  The "
               "head SCALE carried is the coarse mini-BZ <v>; at Q=0 the head "
               "is annihilated by ISDF orthogonality, so the stale scale is "
               "inert to the orthogonality residual (LT_HEAD_PROBLEM.md §2.1). "
               "Set head_minibz_average=true to rebuild with the fine mini-BZ "
               "head instead.")

    # ── W direct: coarse W_q → ifft(R) → per-pair WS embedding → fine → fft
    # back, all inside the ONE sharded densifier (shard_map FFTs + jitted embedding with
    # out_shardings) — the (μ,ν) sharding never drops, so per-rank peak is the
    # local (μ_loc, ν_loc, nk_fine) tile, never a replicated N_μ² array.
    # srlr, the only model: the interpolant sees W minus its fitted long-range
    # model (head and, in a slab, the G_z channels), which is added back
    # exactly at every fine q (``_srlr_terms``).
    resolve_w_head_densify(None, params)                  # an old value refuses here

    def _densified_w():
        """W on the fine grid and its fine Γ head diagonal ``q0_head_ry``."""
        # Popped: once the fine W exists the coarse tile has no reference left.
        W_coarse = data.pop("W_q")
        # The tile's μ positions, read and receipt-checked ONCE: the embedding, the
        # srlr model rows and the Γ head phase all take this array.
        r_mu = w_tile_centroid_frac(wfn, centroids_path, keep, n_rmu,
                                    restart_file=restart_file, log_fn=log_fn)
        if keep is not None:
            raise ValueError(
                "GATE srlr_downfold: the srlr W interpolation on a downfolded "
                "bundle; want: the native centroid set W0_lr was written in.")
        A_c, D_c, A_f, D_f = _srlr_terms(
            restart_file, wfn, sym, meta, params, coarse_grid=coarse_grid,
            fine_grid=fine_grid, n_rmu=n_rmu, n_rmu_pad=int(W_coarse.shape[0]),
            mesh_xy=mesh_xy, input_file=input_file, centroid_frac=r_mu,
            log_fn=log_fn)
        W_coarse = _add_rank_lr(W_coarse, A_c, D_c, mesh_xy, -1.0)
        densify_W = make_w_densifier(
            mesh_xy, P("x", "y", None, None, None), fine_grid, output="k",
            embedding=w_pair_embedding(coarse_grid, wfn, r_mu,
                                       int(W_coarse.shape[0]), meta.sys_dim))
        # The fine output is not the peak: the embedding and FFT can hold
        # several local W tiles. Admit this exact executable beside the
        # resident coarse W, then run the executable whose bytes were read.
        W_q_fine = densify_w(W_coarse, mesh_xy=mesh_xy, densifier=densify_W)
        del W_coarse
        W_q_fine = _add_rank_lr(W_q_fine, A_f, D_f, mesh_xy, 1.0)
        log_fn(f"[kgrid_fine] W embedded per centroid pair (Wigner-Seitz) in R "
               f"{coarse_grid[0]}x{coarse_grid[1]}x{coarse_grid[2]}→"
               f"{fine_grid[0]}x{fine_grid[1]}x{fine_grid[2]} "
               f"(exact trig-interp; direct term now on the fine grid) with the "
               f"SR/LR long-range model split off and re-added")

        # ── re-attach W's fine Γ head on the densified BODY ───────────────────
        # The loader DEFERRED the rank-1 whead injection when it saw a pending
        # densification (``_inject_q0_head(defer_whead=True)``), so ``data["W_q"]``
        # above is the pre-injection body and what the trig interpolant just
        # handled carried no Kronecker delta.  The fine Γ head goes back on from
        # the one ratified integrand (gw.head_densify); the srlr model above owns
        # the head at every fine q ≠ 0.
        q0_head_ry = 0.0
        if head_channel is None and data.get("q0_head_ry", 0.0):
            # The loader defers W's head on a densifying run, so a nonzero coarse
            # diagonal here was normalised on the COARSE N_k and would be dropped
            # (or mis-scaled) on the fine bundle.
            raise ValueError(
                f"GATE densify_q0_head: coarse bundle carries q0_head_ry="
                f"{data['q0_head_ry']!r} Ry but no head_channel; want 0.  why: "
                f"the fine bundle would lose or mis-scale the Γ head.  fix: the "
                f"loader must defer whead on a densifying run.")
        if head_channel is not None:
            from gw.head_densify import attach_head_channel
            from vcoul import CoulombGeometry
            from .bse_head import q0_head_diagonal
            geom = CoulombGeometry.from_wfn(wfn)
            S_fine = build_w_head_channel(
                wfn, sym, meta, params,
                coarse_grid=coarse_grid, fine_grid=fine_grid,
                whead=head_channel["whead"], ref_grid=coarse_grid,
                input_file=input_file, restart_file=restart_file, geometry=geom,
                mesh=mesh_xy,
                gamma_cell=head_channel.get("gamma_cell", "fine"), log_fn=log_fn,
                static_sample=head_channel.get("static_sample"))
            # The srlr model owns the head at every fine q ≠ 0; this channel
            # keeps Γ only.
            S_fine = np.where(np.arange(S_fine.size).reshape(S_fine.shape) == 0,
                              S_fine, 0.0)
            # On a "diagonal" bundle Γ leaves the tile: at fine q = 0 the pair
            # densities are the exact overlaps δ_nn', so its head is the
            # transition-diagonal constant S(0)/(Ω N_k,fine) (bse_head module
            # docstring).  A "rank1" bundle keeps it on the tile, as before.
            S_fine = np.array(S_fine, dtype=np.float64)
            if head_channel.get("q0_head", "rank1") == "diagonal":
                q0_head_ry = q0_head_diagonal(
                    S_fine.flat[0], head_channel["cell_volume"],
                    int(np.prod(fine_grid)))
                S_fine.flat[0] = 0.0
            W_q_fine = attach_head_channel(
                W_q_fine, data["g0_X"], data["g0_Y"], S_fine,
                head_channel["cell_volume"], centroid_frac=r_mu, bvec=geom.bvec)
            log_fn(f"[kgrid_fine] fine Γ head: q0_head="
                   f"{head_channel.get('q0_head', 'rank1')}, diagonal "
                   f"{q0_head_ry:.8e} Ry (fine q ≠ 0 heads in the srlr model)")
        return W_q_fine, q0_head_ry

    # A kernel without the direct term (the loader's ``need_W=False``) has no W
    # to densify and no screened head: its placeholder rides through unchanged.
    W_q_fine, q0_head_ry = (_densified_w() if data.get("w_loaded", True)
                            else (data["W_q"], 0.0))

    out = dict(data)
    out.update({
        "psi_c_X": psi_c_X, "psi_c_Y": psi_c_Y,
        "psi_v_X": psi_v_X, "psi_v_Y": psi_v_Y,
        "M": M,
        "eps_c": eps_c, "eps_v": eps_v,
        "W_q": W_q_fine, "V_q0": V_q0, "q0_head_ry": q0_head_ry,
        "V_q_full": None,                       # finite-q resolvent not a fine use
        # The finite-q EXCHANGE tiles survive the densification, on the COARSE
        # q-grid they were stored on, and they are exact there.  V_{μν}(q) is
        # built from ζ_q and the G-sphere; it never sees the BSE k-grid, so the
        # coarse tile IS the fine solve's answer at any Q that lies on the
        # coarse grid.  What the coarse restart cannot supply is a tile at a
        # fine q that is not also a coarse q — that needs an exchange model at
        # arbitrary q (``--vq-mode interp``), not an interpolation of this
        # array, and no such tile is invented here.
        #
        # Kept under a SEPARATE key with its own grid stamp rather than left in
        # ``V_q_full``: every other consumer of ``V_q_full`` (bse_w_exact's
        # finite-q resolvent) indexes it with the bundle's own grid and would
        # silently read the wrong tile on a densified bundle.  ``None`` there is
        # the refusal those consumers already have; this key is opt-in and
        # carries the grid that indexes it.
        "V_q_coarse": data["V_q_full"],
        "V_q_coarse_grid": coarse_grid,
        "nkx": fine_grid[0], "nky": fine_grid[1], "nkz": fine_grid[2],
        # kgrid_fine_shift: the bundle's k rows are (i + dk)/N.
        "k_shift": fine_shift,
        # The fine window dipoles in this bundle's ψ gauge, or None: a
        # densified bundle never slices a dipole file (window_dipoles).
        "dipoles": dipoles,
    })
    return out
