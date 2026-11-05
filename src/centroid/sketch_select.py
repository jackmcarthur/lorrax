"""Sketched whole-orbit QRCP pruning of over-sampled ISDF candidate points.

The pivoted-Cholesky prune (:mod:`centroid.pivoted_cholesky`) forms the dense
``M x M`` q=0 pair Gram of the candidate pool and pivots on it,

    G[a,a'] = sum_k w_k sum_{l in L, b in B} conj(rho_lbk(a)) rho_lbk(a'),
    rho_lbk(a) = sum_s conj(psi_lks(a)) psi_bks(a),

which is QRCP on the pair matrix ``Z`` (rows ``(k,l,b)``, columns the
candidates) restricted to the pool.  This module replaces ``G`` by the
Khatri-Rao structured Gaussian sketch ``Y = Omega Z`` of ``r'`` rows,

    Y_j(a) = r'^(-1/2) sum_k sqrt(w_k) sum_s (sum_l g^L_jkl conj psi_lks(a))
                                             (sum_b g^B_jkb psi_bks(a)),

with ``g ~ CN(0,1)`` drawn per k row and sketch row from
``jax.random.fold_in`` (the same numbers on every rank and mesh, never stored
beyond one k), so ``E[Y^H Y] = G`` with the Gram's own conjugation, k weights
and band norms.

The selection is orbit-blocked near-greedy QRCP on ``Y``
(``docs/theory/centroid-selection.md``; Lu & Ying's randomized ISDF point
selection).  Per block:

1. orbits are scored by the mean residual of their members;
2. the top ``b`` orbits' columns ``C`` are gathered as row slabs of the
   flattened mesh and their block Gram ``H = C^H C`` is formed;
3. orbits are admitted in order while their in-block score stays above
   ``gamma`` times the block's leading score and they fit the point budget,
   each member pivoted by pivoted Cholesky on ``H`` (largest residual first);
4. the admitted directions are orthonormalized by CholQR2 on the row slabs
   and every column is deflated once, ``Y <- Y - Q (Q^H Y)``, through
   ``distrib_la``'s planned local GEMM with ``Y`` donated.

Residuals are read through the sketch's shrinkage: after ``t`` orthonormal
directions are removed, a Gaussian sketch keeps on average the fraction
``(r' - t)/r'`` of a column's true residual (``E ||(1 - Q Q^H) Omega z||^2
= (r' - t)/r' ||z_perp||^2``; measured for this sketch in the selection
audit).  Every comparison with the floor and every reported residual uses
the shrink-corrected ``d_hat = d r'/(r' - t)``, unbiased for a column the
selection did not condition on (the columns a select leaves behind keep the
selection's own downward bias, ~15-20% at ``r' - t = 64``).

One deflation pass suffices: the pivots read column norms, whose rounding
stays at ``O(eps n_blocks ||y||)`` per column whatever the drift of the
discarded basis' mutual orthogonality; the block's own basis is orthonormal
to working precision (CholQR2 below ``kappa = eps^(-1/2)``, which the floor
``sqrt(eps) max d0`` guarantees).

Memory: ``16 r' M/P`` for the sketch, the k batch's WFN face on the same
flattened-mesh candidate layout, a fixed-tile block (``16 r' m_b`` gather
buffer, ``16 r' m_b/P`` slabs) and a replicated block basis ``16 r' m_b``.

Shapes and shardings (``CAND`` is the flattened mesh, y-major):

    psi         (nk, nb, ns, M_pad)  P(None, None, None, CAND)  loader face
    Y           (1, R, M_pad)        P(None, None, CAND)        the sketch
    d, active   (M_pad,)             P(CAND)                    ||Y[:, a]||^2
    C slabs     (R, m_b)             P(CAND, None)              one block
    H, L        (m_b, m_b), (m_b, t) replicated                 block Gram, factor
    Q           (R, t)               replicated                 block basis

``R`` is ``r'`` padded to a multiple of ``P`` with zero rows; ``t`` is the
block's column count padded to the deflation chunk.
"""

from __future__ import annotations

import math
import resource
import time
from dataclasses import replace
from functools import lru_cache, partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, shard_map
from jax.experimental import multihost_utils as _mh
from jax.scipy.linalg import solve_triangular
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ffi import _services      # noqa: F401  (path bootstrap; see pivoted_cholesky)

_services.ensure_on_path()

import distrib_la                                               # noqa: E402
from lxkit import ordered_psum                                  # noqa: E402

from common import timing                                       # noqa: E402
from common.collectives import device_put_process_local         # noqa: E402
from common.log import get_logger                               # noqa: E402

from . import distribution as dist                              # noqa: E402
from .pivoted_cholesky import (                                 # noqa: E402
    _candidate_band_carrier,
    _candidate_k_schedule,
    _emit_complete_groups,
    _left_band_slice,
    _load_candidate_faces,
    _unpicked_over_floor,
    check_pair_space_window,
    refuse_unless_select_certified,
)

log = get_logger(__name__)

#: The candidate axis: every rank, y-major.  The loader's ``faces='cand'``
#: face and the sketch share it, so the sketch is built with no exchange.
CAND = ("y", "x")

#: Default sketch block width in candidate columns (b orbits of the largest
#: size fill about this many; fewer when one fixed tile holds fewer columns
#: of ``R`` rows).
BLOCK_COLUMNS = 1024

#: Default acceptance cut: an orbit enters a block while its in-block mean
#: residual is at least ``gamma`` times the block's leading score.
GAMMA_DEFAULT = 0.7

#: Deflation chunk: the block basis is applied ``_CHUNK`` columns at a time,
#: one compiled program for every block width.
_CHUNK = 128

_C128 = 16


def sketch_rows_default(n_points: int) -> int:
    """``r' = N + max(64, ceil(0.1 N))`` sketch rows for a budget of ``N`` points."""
    n = int(n_points)
    return n + max(64, int(math.ceil(0.1 * n)))


def sketch_row_carrier(n_rows: int, mesh: Mesh) -> int:
    """``R``: ``r'`` rounded up to the mesh size (the block slabs split rows over it)."""
    p = int(mesh.size)
    return -(-int(n_rows) // p) * p


def sketch_block_columns(n_rows_pad: int) -> int:
    """Block columns ``m_b``: at most :data:`BLOCK_COLUMNS` and one fixed tile of ``R`` rows."""
    from runtime.tiles import tile_units
    return int(tile_units(_C128 * int(n_rows_pad), BLOCK_COLUMNS))


def sketch_block_default(largest_orbit: int, n_rows_pad: int | None = None) -> int:
    """Orbits per block: ``max(1, m_b // |O|max)`` with ``m_b`` from :func:`sketch_block_columns`."""
    cols = BLOCK_COLUMNS if n_rows_pad is None else sketch_block_columns(n_rows_pad)
    return max(1, int(cols) // max(1, int(largest_orbit)))


def _basis_width(m_b: int) -> int:
    """The block basis carrier: ``m_b`` columns rounded up to the deflation chunk."""
    return -(-int(m_b) // _CHUNK) * _CHUNK


# ═══════════════════════════════════════════════════════════════════════
# Programs (geometry-only caches: no data, WFN or mesh payload retained)
# ═══════════════════════════════════════════════════════════════════════

_EXECUTABLES: dict = {}


def _admitted(kernel, args, stage: str, mesh: Mesh, *, face: bool):
    """Compile ``kernel`` at ``args`` once, admit its new bytes, book them.

    The price is the program that runs, compiled at its operands
    (``runtime.aot_memory.compiled_price``: temporaries plus outputs less
    aliases, the largest any rank reads), as an increment beside the bytes
    already live; ``lxkit.memory.admit`` warns and runs when it is over the
    room.  ``face`` says the figure falls with the mesh (sharded outputs).
    """
    from common.gpu_utils import device_budget_bytes
    from lxkit import memory
    from runtime.aot_memory import compiled_price
    key = (kernel, tuple((tuple(a.shape), str(a.dtype), getattr(a, "sharding", None))
                         for a in jax.tree.leaves(args)))
    if key not in _EXECUTABLES:
        _EXECUTABLES[key] = kernel.lower(*args).compile()
        compiled = _EXECUTABLES[key]
        device_budget_bytes()
        price = replace(compiled_price(compiled=compiled, ranks=int(mesh.size),
                                       face=face, source=stage), increment=True)
        memory.admit(stage, price, memory.room(bucketed=False))
        memory.book(price)
    return _EXECUTABLES[key]


@lru_cache(maxsize=None)
def _zero_kernel(mesh: Mesh, n_rows_pad: int, m_pad: int):
    """The ``(1, R, M_pad)`` zero sketch at ``P(None, None, CAND)``."""
    out = NamedSharding(mesh, P(None, None, CAND))

    @jax.jit(out_shardings=out)
    def zero():
        return jnp.zeros((1, int(n_rows_pad), int(m_pad)), dtype=jnp.complex128)

    return zero


def _factor_rows(key, rows, n: int, factors: str, *, k_pos=None, which=0,
                 n_left=1, n_right=1, n_rows=1):
    """Sketch factor rows ``g[j, :]`` (``(len(rows), n)`` complex128, replicated).

    ``factors='gaussian'``: row ``j`` is ``CN(0,1)`` from ``fold_in(key, j)``,
    so a row's numbers do not depend on the tiling or the mesh.
    ``factors='exhaustive'``: ``Omega`` is the identity on the pair index
    ``j = (k n_L + l) n_B + b`` (``r' = N_k n_L n_B``), scaled by ``r'^(1/4)``
    so ``Y^H Y = G`` exactly -- the exactness check of the construction.
    """
    if factors == "gaussian":
        return jax.vmap(lambda j: jax.random.normal(
            jax.random.fold_in(key, j), (int(n),), dtype=jnp.complex128))(rows)
    k_of = rows // (n_left * n_right)
    sel = (rows // n_right) % n_left if which == 0 else rows % n_right
    hit = (k_of == k_pos)[:, None] & (sel[:, None] == jnp.arange(int(n))[None, :])
    return hit.astype(jnp.complex128) * (float(n_rows) ** 0.25)


@lru_cache(maxsize=None)
def _sketch_kernel(mesh: Mesh, nb_r: int, ns: int, m_pad: int, n_rows_pad: int,
                   n_rows: int, width: int, left: tuple | None, nb_l: int,
                   factors: str):
    """One k batch's Khatri-Rao accumulation ``Y += Omega_k Z_k`` (Y donated).

    Operands: ``Y (1, R, M_pad)`` at ``P(None,None,CAND)``; right face
    ``(nk, nb_r, ns, M_pad)`` and, unless ``left = (lo, hi)`` names it inside
    the right one, the left face ``(nk, nb_l, ns, M_pad)``, both at
    ``P(None,None,None,CAND)`` (the loader's ``faces='cand'``: every rank
    holds its own candidate columns, every band); replicated
    ``sqrt_w (nk,)``, ``k_ids (nk,)`` (full-BZ rows, the PRNG stream),
    ``k_pos (nk,)`` (positions in the selected-row order), the key data and
    the row-tile ``starts`` / ``valid_from`` ``(n_tiles,)``.  Per k and row
    tile the two factor GEMMs ``(ns, t, nb) x (ns, nb, M_pad)`` run through
    ``distrib_la``'s planned local GEMM on the candidate columns; the spin
    sum of their Hadamard product is added into ``Y``'s rows.  Rows below
    ``valid_from`` (the clamped last tile's overlap) add nothing; rows
    ``>= r'`` of the carrier are never written.
    """
    cand = NamedSharding(mesh, P(None, None, CAND))
    plan_l = distrib_la.local_gemm_plan(
        mesh, m=width, k=nb_l, n=m_pad, nq=ns, dtype=jnp.complex128,
        out_spec=P(None, None, CAND), warmup=False)
    plan_r = distrib_la.local_gemm_plan(
        mesh, m=width, k=nb_r, n=m_pad, nq=ns, dtype=jnp.complex128,
        out_spec=P(None, None, CAND), warmup=False)
    scale = 1.0 / math.sqrt(float(n_rows))
    nested = left is not None

    def build(Y, psi_r, psi_l, sqrt_w, k_ids, k_pos, key_data, starts, valid_from):
        key = jax.random.wrap_key_data(key_data)

        def one_k(Y, xs):
            face_r, face_l, w_k, k_id, kp = xs
            B_op = jnp.transpose(face_r, (1, 0, 2))           # (ns, nb_r, M)
            A_op = (B_op[:, left[0]:left[1], :] if nested
                    else jnp.transpose(face_l, (1, 0, 2)))    # (ns, nb_l, M)
            key_k = jax.random.fold_in(key, k_id)
            key_l, key_b = jax.random.fold_in(key_k, 0), jax.random.fold_in(key_k, 1)

            def one_tile(Y, tile):
                start, first_new = tile
                rows = start + jnp.arange(width, dtype=jnp.int32)
                g_l = _factor_rows(key_l, rows, nb_l, factors, k_pos=kp, which=0,
                                   n_left=nb_l, n_right=nb_r, n_rows=n_rows)
                g_b = _factor_rows(key_b, rows, nb_r, factors, k_pos=kp, which=1,
                                   n_left=nb_l, n_right=nb_r, n_rows=n_rows)
                # sum_l g^L conj(psi_l) = conj(sum_l conj(g^L) psi_l)
                a_l = jnp.conj(plan_l(jnp.broadcast_to(
                    jnp.conj(g_l)[None], (ns, width, nb_l)), A_op))
                a_b = plan_r(jnp.broadcast_to(g_b[None], (ns, width, nb_r)), B_op)
                contrib = (w_k * scale) * jnp.sum(a_l * a_b, axis=0)
                contrib = jnp.where((rows >= first_new)[:, None], contrib,
                                    jnp.zeros((), contrib.dtype))
                zero = jnp.int32(0)
                cur = lax.dynamic_slice(Y, (zero, start, zero), (1, width, int(m_pad)))
                return lax.dynamic_update_slice(
                    Y, cur + contrib[None], (zero, start, zero)), None

            Y, _ = lax.scan(one_tile, Y, (starts, valid_from), unroll=1)
            return Y, None

        xs = (psi_r, psi_r if nested else psi_l, sqrt_w, k_ids, k_pos)
        Y, _ = lax.scan(one_k, Y, xs, unroll=1)
        return jax.lax.with_sharding_constraint(Y, cand)

    return jax.jit(build, donate_argnums=(0,), out_shardings=cand)


def sketch_from_faces(Y, psi_r, psi_l, mesh: Mesh, *, n_rows: int, sqrt_w,
                      k_ids, k_pos, key_data, left_slice,
                      factors: str = "gaussian"):
    """Add one k batch's sketch rows to ``Y`` (donated) from its WFN faces.

    Implements ``Y_j(a) += r'^(-1/2) sum_k sqrt(w_k) sum_s (g^L conj psi)(g^B psi)``
    (module docstring) for the batch's k rows.

    Parameters
    ----------
    Y : (1, R, M_pad) complex128, ``P(None, None, CAND)``
        The running sketch (``R >= r'``; rows ``>= r'`` stay zero); donated.
    psi_r : (nk, nb_r, ns, M_pad) complex128, ``P(None, None, None, CAND)``
        Right-window face (the loader's ``faces='cand'``).
    psi_l : (nk, nb_l, ns, M_pad) or None
        Left-window face; ``None`` when ``left_slice = (lo, hi)`` names it as
        the right face's band rows.
    n_rows : int
        ``r'``, the sketch rows (the scale ``r'^(-1/2)``).
    sqrt_w, k_ids, k_pos : (nk,) host arrays
        ``sqrt(w_k)`` (full-zone normalization), full-BZ row ids (the PRNG
        stream) and positions in the selected-row order.
    key_data : (2,) uint32 host array
        ``jax.random.key_data`` of the run's key.
    factors : {'gaussian', 'exhaustive'}
        ``'exhaustive'`` is the identity sketch of the exactness check.

    Returns
    -------
    Y : the updated sketch, same shape and sharding.
    """
    from runtime.tiles import tile_plan
    _, n_rows_pad, m_pad = (int(v) for v in Y.shape)
    nk, nb_r, ns, _ = (int(v) for v in psi_r.shape)
    if left_slice is not None:
        left = (int(left_slice[0]), int(left_slice[1]))
        nb_l = left[1] - left[0]
    else:
        left, nb_l = None, int(psi_l.shape[1])
    if m_pad % int(mesh.size):
        raise ValueError(f"sketch candidate carrier {m_pad} does not tile {mesh.size} ranks")
    if not 0 < int(n_rows) <= n_rows_pad:
        raise ValueError(f"sketch rows {n_rows} exceed the row carrier {n_rows_pad}")
    m_loc = m_pad // int(mesh.size)
    # Row tile from the shapes alone (fixed tile): the two factor products,
    # their Hadamard sum and the factor rows of one tile.
    per_row = _C128 * ((2 * ns + 1) * m_loc + (1 + ns) * (nb_l + nb_r))
    width, starts = tile_plan(per_row, int(n_rows))
    valid_from = np.asarray([j * width for j in range(len(starts))], dtype=np.int32)
    rep = NamedSharding(mesh, P())
    put = lambda a, dt: device_put_process_local(np.asarray(a, dtype=dt), rep)  # noqa: E731
    kernel = _sketch_kernel(mesh, nb_r, ns, m_pad, n_rows_pad, int(n_rows),
                            int(width), left, nb_l, str(factors))
    args = (Y, psi_r, psi_l, put(sqrt_w, np.float64), put(k_ids, np.int32),
            put(k_pos, np.int32), put(key_data, np.uint32),
            put(starts, np.int32), put(valid_from, np.int32))
    exe = _admitted(kernel, args, "kmeans candidate sketch build (one k batch)",
                    mesh, face=True)
    return exe(*args)


def _face_bytes_per_k(mesh: Mesh, *, ns: int, m_pad: int, left, right) -> float:
    """Per-rank bytes of one k row of the batch's candidate faces (``faces='cand'``)."""
    nb_faces = _candidate_band_carrier(right[1] - right[0], mesh, 64) + (
        0 if _left_band_slice(left, right) is not None
        else _candidate_band_carrier(left[1] - left[0], mesh, 64))
    return float(_C128 * ns * m_pad * nb_faces) / int(mesh.size)


def build_sketch_q0_via_loadwfns(
    wfn,
    sym,
    cand_idx,
    mesh_xy: Mesh,
    *,
    n_rows: int,
    seed: int,
    band_range_left: tuple[int, int],
    band_range_right: tuple[int, int],
    band_norms=None,
    k_weights=None,
    bispinor: bool = False,
    band_chunk_size: int = 64,
    memory_per_device_gb: float | None = None,
    factors: str = "gaussian",
    verbose: bool = False,
):
    """The q=0 candidate sketch over fixed-tile full-BZ k batches.

    The k batches, their weights and the band norms are the candidate Gram's
    own (:func:`centroid.pivoted_cholesky._candidate_k_schedule`,
    :func:`~centroid.pivoted_cholesky._load_candidate_faces`); the loader
    delivers the one face the sketch reads, ``faces='cand'``: every rank's
    own candidate columns with every band, so neither the Gram's ``'y'``
    gather nor its conjugate ``X`` face is formed.

    Parameters
    ----------
    wfn, sym : the open ``WfnLoader`` and its ``SymMaps``.
    cand_idx : (M, 3) int FFT-grid indices of the candidates.
    mesh_xy : the ``('x','y')`` mesh.
    n_rows : sketch rows ``r'``.
    seed : the PRNG seed (``jax.random.key``).
    band_range_left, band_range_right : 0-based half-open band windows.
    band_norms, k_weights, bispinor, band_chunk_size, memory_per_device_gb :
        as :func:`centroid.pivoted_cholesky.build_gram_q0_via_loadwfns`.
    factors : ``'gaussian'`` (production) or ``'exhaustive'``
        (``n_rows = N_k n_L n_B``; ``Y^H Y = G`` exactly).

    Returns
    -------
    Y : (1, R, M_pad) complex128 at ``P(None, None, CAND)``; pad rows and
        columns zero.
    """
    from runtime.padding import padded_mu_extent
    left = (int(band_range_left[0]), int(band_range_left[1]))
    right = (int(band_range_right[0]), int(band_range_right[1]))
    nb_l, nb_r = left[1] - left[0], right[1] - right[0]
    if min(nb_l, nb_r) <= 0:
        raise ValueError("candidate sketch band windows must be nonempty")
    M = int(np.asarray(cand_idx).shape[0])
    m_pad = padded_mu_extent(M, mesh_xy)
    ns = 4 if bispinor else int(wfn.nspinor)
    per_k = _face_bytes_per_k(mesh_xy, ns=ns, m_pad=m_pad, left=left, right=right)
    rows_sel, weights, nk, nk_full, k_batch, n_chunks = _candidate_k_schedule(
        wfn, sym, k_weights, per_k)
    if factors == "exhaustive" and int(n_rows) != nk * nb_l * nb_r:
        raise ValueError(f"the exhaustive sketch needs r' = N_k n_L n_B = "
                         f"{nk * nb_l * nb_r}; got {int(n_rows)}")
    key_data = np.asarray(jax.random.key_data(jax.random.key(int(seed))))
    Y = _zero_kernel(mesh_xy, sketch_row_carrier(n_rows, mesh_xy), int(m_pad))()
    for k0 in range(0, nk, k_batch):
        k1 = min(nk, k0 + k_batch)
        rows = (None if n_chunks == 1 and nk == nk_full else
                tuple(int(k) for k in rows_sel[k0:k1]))
        _, left_slice, psi_l, _, psi_r, _ = _load_candidate_faces(
            wfn, sym, cand_idx, mesh_xy, left_range=left, right_range=right,
            n_val=None, n_cond=None, bispinor=bispinor, band_norms=band_norms,
            band_chunk_size=band_chunk_size,
            memory_per_device_gb=memory_per_device_gb, full_k_rows=rows,
            gamma_mode="charge", verbose=verbose, faces="cand")
        with timing.section("sketch.kron"):
            Y = sketch_from_faces(
                Y, psi_r, psi_l, mesh_xy, n_rows=int(n_rows),
                sqrt_w=np.sqrt(weights[k0:k1]), k_ids=rows_sel[k0:k1],
                k_pos=np.arange(k0, k1), key_data=key_data,
                left_slice=left_slice, factors=factors)
            Y.block_until_ready()
        for face in (psi_l, psi_r):
            if face is not None and not face.is_deleted():
                face.delete()
        del psi_l, psi_r
    if verbose:
        log.debug(f"[centroid-sketch] k batches: nk={nk}, batch={k_batch}, "
                  f"chunks={n_chunks}, faces<={per_k * k_batch / 2**30:.2f} GiB/device")
    return Y


# ═══════════════════════════════════════════════════════════════════════
# Orbit-blocked near-greedy QRCP on the sketch
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=None)
def _residual_kernel(mesh: Mesh):
    """``d(a) = ||Y[:, a]||^2`` at ``P(CAND)`` (a local column sum)."""
    @partial(jax.jit, out_shardings=NamedSharding(mesh, P(CAND)))
    def residual(Y):
        return jnp.sum(jnp.abs(Y[0]) ** 2, axis=0)
    return residual


def _right_solve_upper_h(B, L):
    """``X`` with ``X L^H = B`` for lower-triangular ``L`` (a right triangular solve).

    ``(X L^H)^T = conj(L) X^T``, so ``X^T`` solves the lower system
    ``conj(L) X^T = B^T``.
    """
    return solve_triangular(jnp.conj(L), B.T, lower=True).T


@lru_cache(maxsize=None)
def _select_block_kernel(mesh: Mesh, m_pad: int, n_rows_pad: int, n_rows: int,
                         n_groups: int, s_max: int, n_slots: int, gamma: float):
    """One block of the orbit-blocked selection (module docstring, steps 1-4).

    Operands: ``Y (1, R, M_pad)`` and ``d, active, gid (M_pad,)`` on the
    candidate axis; replicated ``members (G+1, s_max)`` (global candidate
    ids, ``-1`` pad; row ``G`` is the pad sentinel), ``sizes (G+1,)``,
    ``chosen (G+1,)``, ``remaining`` (points), ``floor0 = tol max d0`` and
    ``t_done`` (directions removed so far).

    1. Orbit scores: the mean of the members' residuals above the floor
       ``floor0 (r'-t)/r'`` (a local ``segment_sum`` and one ordered
       all-reduce), over orbits not chosen that fit the remaining budget.
    2. The top ``n_slots`` orbits; ``S0`` the leading unbiased score.
    3. Their member columns as row slabs ``C (R/P, m_b)`` (an exact
       reduce-scatter: one owner per column, zeros elsewhere) and the block
       Gram ``H = C^H C`` (one ordered all-reduce of ``m_b^2``).
    4. In order of unbiased in-block score, an orbit is admitted while that
       score is at least ``gamma * S0`` (the first always) and it fits; its
       members are pivoted by pivoted Cholesky on ``H``, largest residual
       first.  A member whose residual is at or below the floor at the
       current ``t`` is delivered with a zero direction, as the Cholesky
       select delivers it.  The admitted columns ``C_sel = Q L_sel^H`` are
       orthonormalized by two CholQR passes on the slabs and the basis
       ``Q (R, t)`` is gathered.

    Returns ``(Q, piv (m_b,), d_taken (m_b,), n_acc, n_pts, n_dir, S0,
    active', chosen', remaining')``, everything but ``active'`` replicated;
    ``d_taken`` holds the unbiased residual of each delivered point (zero for
    a zero direction).
    """
    G = int(n_groups)
    s_max = int(s_max)
    n_slots = int(n_slots)
    m_b = n_slots * s_max
    t_pad = _basis_width(m_b)
    m_loc = int(m_pad) // int(mesh.size)
    r_true = float(n_rows)
    cand1 = P(CAND)
    rep = NamedSharding(mesh, P())
    minus_inf = -jnp.inf
    tiny = float(np.finfo(np.float64).tiny)

    def _scores(d, active, gid, floor):
        part = jax.ops.segment_sum(
            jnp.where(active & (d > floor), d, 0.0), gid,
            num_segments=G + 1, indices_are_sorted=False)
        return ordered_psum(part, CAND)

    scores_sm = shard_map(_scores, mesh=mesh, in_specs=(cand1, cand1, cand1, P()),
                          out_specs=P(), check_vma=False)

    def _block(Y, ids):
        local = ids - lax.axis_index(CAND) * m_loc
        own = (ids >= 0) & (local >= 0) & (local < m_loc)
        cols = jnp.take(Y[0], jnp.clip(local, 0, m_loc - 1), axis=1)
        cols = jnp.where(own[None, :], cols, jnp.zeros((), Y.dtype))
        C = lax.psum_scatter(cols, CAND, scatter_dimension=0, tiled=True)  # exact-reduce: one owner per column, zeros elsewhere
        H = ordered_psum(jnp.conj(C.T) @ C, CAND)
        return C, H

    block_sm = shard_map(_block, mesh=mesh, in_specs=(P(None, None, CAND), P()),
                         out_specs=(P(CAND, None), P()), check_vma=False)

    def _basis(C, sel, L_sel):
        pad = sel < 0
        Cs = jnp.take(C, jnp.clip(sel, 0, m_b - 1), axis=1)
        Cs = jnp.where(pad[None, :], jnp.zeros((), C.dtype), Cs)
        X = _right_solve_upper_h(Cs, L_sel)
        H2 = ordered_psum(jnp.conj(X.T) @ X, CAND)
        H2 = jnp.where(pad[:, None] | pad[None, :], jnp.zeros((), H2.dtype), H2)
        H2 = H2 + jnp.diag(pad.astype(H2.dtype))
        X = _right_solve_upper_h(X, jnp.linalg.cholesky(H2))
        return lax.all_gather(X, CAND, axis=0, tiled=True)

    basis_sm = shard_map(_basis, mesh=mesh, in_specs=(P(CAND, None), P(), P()),
                         out_specs=P(), check_vma=False)

    def step(Y, d, active, gid, members, sizes, chosen, remaining, floor0, t_done):
        t0 = t_done.astype(jnp.float64)

        def shrink(n_dir):
            return jnp.maximum(r_true - t0 - n_dir.astype(jnp.float64), 1.0) / r_true

        sh0 = shrink(jnp.int32(0))
        values = scores_sm(d, active, gid, floor0 * sh0)
        eligible = ((sizes > 0) & ~chosen & (sizes <= remaining)
                    & (jnp.arange(G + 1) < G))
        scores = jnp.where(eligible, values / jnp.maximum(
            sizes.astype(values.dtype), 1.0), minus_inf)
        top_s, top_o = lax.top_k(scores, n_slots)
        slot_ok = top_s > minus_inf
        s0 = top_s[0] / sh0
        ids = jnp.where(slot_ok[:, None], members[top_o], -1).reshape(m_b)
        col_ok = ids >= 0
        slot_size = jnp.where(slot_ok, sizes[top_o], 0)
        C, H = block_sm(Y, ids)
        dd0 = jnp.where(col_ok, jnp.real(jnp.diagonal(H)), 0.0)

        def accept_slot(j, st):
            (dd, used, L, sel, n_dir, opiv, odtk, n_pts, done, rem, n_acc, acc) = st
            base = j * s_max

            def member(_, carry):
                dd, used, L, sel, n_dir, opiv, odtk, n_pts = carry
                cols = base + jnp.arange(s_max)
                live = col_ok[cols] & ~used[cols]
                q = jnp.argmax(jnp.where(live, dd[cols], minus_inf)).astype(jnp.int32)
                p = base + q
                there = live[q]
                sh = shrink(n_dir)
                take = there & (dd[p] > floor0 * sh)
                piv_val = jnp.sqrt(jnp.maximum(dd[p], tiny))
                # Pivoted Cholesky column of p on the block Gram: exact zero on
                # delivered columns, the pivot itself on p.
                l = (H[:, p] - L @ jnp.conj(L[p, :])) / piv_val
                l = jnp.where(used | ~col_ok, jnp.zeros((), l.dtype), l)
                l = l.at[p].set(piv_val.astype(l.dtype))
                slot = jnp.minimum(n_dir, t_pad - 1)
                L = L.at[:, slot].set(jnp.where(take, l, L[:, slot]))
                sel = sel.at[slot].set(jnp.where(take, p, sel[slot]))
                dd_new = jnp.where(take, dd - jnp.abs(l) ** 2, dd)
                dd_new = dd_new.at[p].set(jnp.where(there, 0.0, dd_new[p]))
                at = jnp.minimum(n_pts, m_b - 1)
                opiv = opiv.at[at].set(jnp.where(there, ids[p], opiv[at]))
                odtk = odtk.at[at].set(jnp.where(there, jnp.where(take, dd[p] / sh, 0.0),
                                                 odtk[at]))
                used = used.at[p].set(used[p] | there)
                return (dd_new, used, L, sel, n_dir + take.astype(jnp.int32), opiv,
                        odtk, n_pts + there.astype(jnp.int32))

            carry = lax.fori_loop(0, s_max, member,
                                  (dd, used, L, sel, n_dir, opiv, odtk, n_pts))
            dd, used, L, sel, n_dir, opiv, odtk, n_pts = carry
            acc = acc.at[n_acc].set(top_o[j])
            return (dd, used, L, sel, n_dir, opiv, odtk, n_pts, done.at[j].set(True),
                    rem - slot_size[j], n_acc + 1, acc)

        def slot_step(_, carry):
            st, stopped = carry
            dd, used, *_rest = st
            done, rem, n_acc = st[8], st[9], st[10]
            sh = shrink(st[4])
            live = jnp.where(col_ok & ~used & (dd > floor0 * sh), dd, 0.0)
            in_block = live.reshape(n_slots, s_max).sum(axis=1) / (
                jnp.maximum(slot_size.astype(dd.dtype), 1.0) * sh)
            ok = slot_ok & ~done & (slot_size <= rem)
            masked = jnp.where(ok, in_block, minus_inf)
            j = jnp.argmax(masked).astype(jnp.int32)
            accept = (~stopped & (masked[j] > minus_inf)
                      & ((n_acc == 0) | (masked[j] >= gamma * s0)))
            st = lax.cond(accept, partial(accept_slot, j), lambda s: s, st)
            return st, stopped | ~accept

        st0 = (dd0, jnp.zeros((m_b,), bool), jnp.zeros((m_b, t_pad), C.dtype),
               jnp.full((t_pad,), -1, jnp.int32), jnp.int32(0),
               jnp.full((m_b,), -1, jnp.int32), jnp.zeros((m_b,), jnp.float64),
               jnp.int32(0), jnp.zeros((n_slots,), bool), remaining, jnp.int32(0),
               jnp.full((n_slots,), -1, jnp.int32))
        st, _ = lax.fori_loop(0, n_slots, slot_step, (st0, jnp.bool_(False)))
        (_, _, L, sel, n_dir, opiv, odtk, n_pts, _, rem, n_acc, acc) = st
        pad = sel < 0
        L_sel = jnp.take(L, jnp.clip(sel, 0, m_b - 1), axis=0)
        L_sel = jnp.where(pad[:, None], jnp.zeros((), L.dtype), L_sel)
        L_sel = jnp.tril(L_sel) + jnp.diag(pad.astype(L.dtype))
        Q = basis_sm(C, sel, L_sel)
        took = jnp.zeros((G + 1,), bool).at[jnp.where(acc >= 0, acc, G)].set(True)
        took = took.at[G].set(False)
        return (Q, opiv, odtk, n_acc, n_pts, n_dir, s0, active & ~took[gid],
                chosen | took, rem)

    out = (rep, rep, rep, rep, rep, rep, rep, NamedSharding(mesh, cand1), rep, rep)
    return jax.jit(step, out_shardings=out)


@lru_cache(maxsize=None)
def _deflate_kernel(mesh: Mesh, n_rows_pad: int, m_pad: int, t_pad: int):
    """``Y <- Y - Q (Q^H Y)`` once, :data:`_CHUNK` columns of ``Q`` at a time; Y donated.

    ``Q (R, t_pad)`` is the block's replicated orthonormal basis, zero beyond
    its first ``n_dir`` columns; ``ceil(n_dir / _CHUNK)`` chunks are applied
    (a sequence of chunk projections equals the block projection because the
    chunks are mutually orthogonal).  Both products are
    ``distrib_la.local_gemm_plan`` on the candidate columns, one local GEMM
    per rank and no collective.  Returns ``(Y, d, trace)``: the deflated
    sketch, ``d = ||Y[:, a]||^2`` at ``P(CAND)`` and the replicated residual
    trace over ``active`` columns (a diagnostic).
    """
    cand = NamedSharding(mesh, P(None, None, CAND))
    R = int(n_rows_pad)
    project = distrib_la.local_gemm_plan(
        mesh, m=_CHUNK, k=R, n=m_pad, nq=1, dtype=jnp.complex128,
        out_spec=P(None, None, CAND), warmup=False)
    update = distrib_la.local_gemm_plan(
        mesh, m=R, k=_CHUNK, n=m_pad, nq=1, dtype=jnp.complex128,
        alpha=-1.0, beta=1.0, out_spec=P(None, None, CAND), warmup=False)

    def deflate(Y, Q, n_dir, active):
        def chunk(c, Y):
            Qc = lax.dynamic_slice_in_dim(Q, c * _CHUNK, _CHUNK, axis=1)
            return update(Qc[None], project(jnp.conj(Qc.T)[None], Y), C=Y)

        Y = lax.fori_loop(0, (n_dir + _CHUNK - 1) // _CHUNK, chunk, Y)
        d = jnp.sum(jnp.abs(Y[0]) ** 2, axis=0)
        return Y, d, jnp.sum(jnp.where(active, d, 0.0))

    out = (cand, NamedSharding(mesh, P(CAND)), NamedSharding(mesh, P()))
    return jax.jit(deflate, donate_argnums=(0,), out_shardings=out)


def sketch_block_select(Y, mesh: Mesh, *, group_id, n_groups: int,
                        point_budget: int, n_slots: int, gamma: float,
                        n_rows: int | None = None, tol_rel: float | None = None,
                        verbose: bool = False):
    """Orbit-blocked near-greedy QRCP on the sketch under a point budget.

    The semantics of ``common.pivoted_cholesky``'s group-block select on
    ``Y^H Y``: whole groups only; a group is admitted only if it fits the
    remaining point budget; groups are ranked by mean residual over their
    members (residuals above the floor); every member of an admitted group
    is pivoted, largest residual first; members at or below the floor
    ``tol_rel * max d0`` (``tol_rel=None``: ``sqrt(eps)``) are delivered
    with zero ``d_taken``; the select stops when no remaining group fits.
    ``n_slots = 1`` with ``gamma = 1`` is exactly that greedy order.  Every
    residual is read as the unbiased ``d r'/(r' - t)`` (module docstring).

    Parameters
    ----------
    Y : (1, R, M_pad) complex128 at ``P(None, None, CAND)``
        The sketch, ``R`` a multiple of the mesh size (rows ``>= r'`` zero);
        donated (deflated in place, released on return).
    mesh : the ``('x','y')`` mesh.
    group_id : (M,) int
        Dense group labels in ``[0, n_groups)`` (pads are added here).
    n_groups : number of groups.
    point_budget : points to deliver at most.
    n_slots : groups gathered per block (``b``).
    gamma : the in-block acceptance cut, ``0 < gamma <= 1``.
    n_rows : ``r'`` (``None``: ``R``), the sketch's row count for the
        unbiased residuals.

    Returns
    -------
    ``(piv, rank, d_final, d_taken, trR_over_trG, psd_info, n_blocks, d0max)``:
    the point pivot order ``(n,)``; the directions above the floor; the
    unbiased residual ``(M,)`` on host (zero on picked points); the unbiased
    picked residuals ``(n,)``; the residual trace over the initial trace at
    each block boundary ``(n_blocks + 1,)``; ``(0.0, -1, -1)`` (PSD by
    construction); the block count; ``max d0``.
    """
    _, n_rows_pad, m_pad = (int(v) for v in Y.shape)
    if n_rows_pad % int(mesh.size):
        raise ValueError(f"sketch row carrier {n_rows_pad} does not tile {mesh.size} ranks")
    r_true = int(n_rows_pad if n_rows is None else n_rows)
    gid = np.asarray(group_id, dtype=np.int32)
    M = int(gid.shape[0])
    G = int(n_groups)
    sizes = np.bincount(gid, minlength=G).astype(np.int32)
    s_max = int(sizes.max())
    n_slots = max(1, min(int(n_slots), G))
    t_pad = _basis_width(n_slots * s_max)
    order = np.argsort(gid, kind="stable")
    starts = np.concatenate([[0], np.cumsum(sizes)])
    members = np.full((G + 1, s_max), -1, dtype=np.int32)
    for g in range(G):
        members[g, :sizes[g]] = order[starts[g]:starts[g + 1]]
    sizes_s = np.concatenate([sizes, [0]]).astype(np.int32)
    cand = NamedSharding(mesh, P(CAND))
    rep = NamedSharding(mesh, P())
    put = lambda a, dt: device_put_process_local(np.asarray(a, dtype=dt), rep)  # noqa: E731
    gid_pad = np.full((m_pad,), G, dtype=np.int32)
    gid_pad[:M] = gid
    active_host = np.zeros((m_pad,), dtype=bool)
    active_host[:M] = True
    gid_dev = device_put_process_local(gid_pad, cand)
    active = device_put_process_local(active_host, cand)
    members_dev = put(members, np.int32)
    sizes_dev = put(sizes_s, np.int32)
    chosen = put(np.zeros((G + 1,)), bool)
    remaining = put(int(point_budget), np.int32)

    d = _residual_kernel(mesh)(Y)
    d0 = np.asarray(_mh.process_allgather(d, tiled=True))[:M]
    d0max = float(d0.max())
    trG = float(d0.sum())
    tol = (float(np.sqrt(np.finfo(np.float64).eps)) if tol_rel is None
           else float(tol_rel))
    floor0 = put(tol * d0max, np.float64)

    select = _select_block_kernel(mesh, m_pad, n_rows_pad, r_true, G, s_max,
                                  n_slots, float(gamma))
    deflate = _deflate_kernel(mesh, n_rows_pad, m_pad, t_pad)
    piv_parts, dtk_parts, traces = [], [], []
    n_blocks = 0
    t_done = 0
    wall = {"block": 0.0, "deflate": 0.0}
    started = time.perf_counter()
    while True:
        args = (Y, d, active, gid_dev, members_dev, sizes_dev, chosen, remaining,
                floor0, put(t_done, np.int32))
        exe = _admitted(select, args, "kmeans sketch select block", mesh, face=False)
        t_b = time.perf_counter()
        (Q, piv_b, dtk_b, n_acc, n_pts, n_dir_dev, s0, active, chosen,
         remaining) = exe(*args)
        n_acc, n_pts, n_dir = int(n_acc), int(n_pts), int(n_dir_dev)
        wall["block"] += time.perf_counter() - t_b
        if n_acc == 0:
            del Q
            break
        if n_dir > 0:
            dargs = (Y, Q, n_dir_dev, active)
            dexe = _admitted(deflate, dargs, "kmeans sketch deflation (one pass)",
                             mesh, face=True)
            t_b = time.perf_counter()
            Y, d, trace = dexe(*dargs)
            jax.block_until_ready(d)
            wall["deflate"] += time.perf_counter() - t_b
        else:
            trace = jnp.sum(jnp.where(active, d, 0.0))
        del Q
        t_done += n_dir
        piv_parts.append(np.asarray(piv_b)[:n_pts])
        dtk_parts.append(np.asarray(dtk_b)[:n_pts])
        traces.append(trace)
        n_blocks += 1
        if verbose:
            log.debug(f"[centroid-sketch-block] block={n_blocks} orbits={n_acc} "
                      f"points={n_pts} directions={n_dir} t={t_done} "
                      f"delivered={sum(p.size for p in piv_parts)}/{int(point_budget)} "
                      f"lead_score={float(s0):.3e} "
                      f"elapsed_s={time.perf_counter() - started:.3f}")
    log.info(f"[centroid-sketch] select phases: blocks={n_blocks} directions={t_done} "
             f"block_s={wall['block']:.3f} deflate_s={wall['deflate']:.3f} "
             f"block_columns={n_slots * s_max} basis_width={t_pad}")
    unbias = float(r_true) / max(float(r_true - t_done), 1.0)
    d_final = unbias * np.asarray(_mh.process_allgather(
        jnp.where(active, d, 0.0), tiled=True))[:M]
    Y.delete()
    piv = (np.concatenate(piv_parts) if piv_parts else np.zeros((0,), np.int32))
    d_taken = (np.concatenate(dtk_parts) if dtk_parts else np.zeros((0,)))
    rank = int(np.sum(d_taken > tol * d0max))
    tr = np.asarray([1.0] + [float(t) / trG if trG > 0 else 0.0 for t in traces])
    return piv, rank, d_final, d_taken, tr, (0.0, -1, -1), n_blocks, d0max


# ═══════════════════════════════════════════════════════════════════════
# End-to-end wrapper (the return contract of the pivoted-Cholesky prune)
# ═══════════════════════════════════════════════════════════════════════

def _memory_proposal(mesh: Mesh, *, n_rows_pad: int, m_pad: int, ns: int,
                     nb_r: int, faces_per_rank: float, m_b: int):
    """The closed-form proposal of the sketch's per-rank live set.

    ``operands``: the sketch ``16 R M_pad / P`` (spread) and one k batch's
    WFN faces (``faces='cand'``, spread).  ``xla``: the larger of the build
    workspace (one transposed right face ``16 ns nb_r M_pad / P`` plus the
    fixed row tile ``TILE_BYTES``), the select block (the gather buffer
    ``16 R m_b``, the slabs and their CholQR copies ``3 16 R t / P`` and the
    replicated basis ``16 R t``, ``t`` the chunk-padded block width) and the
    deflation (the projection ``16 _CHUNK M_pad / P``, twice).  The
    executables are admitted at their own compiled prices when they run.
    """
    from lxkit import memory
    from runtime.tiles import TILE_BYTES
    ranks = int(mesh.size)
    t_pad = _basis_width(m_b)
    build = (_C128 * ns * nb_r * m_pad, float(TILE_BYTES))         # spread, fixed
    select = (3 * _C128 * n_rows_pad * t_pad,
              _C128 * n_rows_pad * (m_b + t_pad))
    deflate = (2 * _C128 * _CHUNK * m_pad, 0.0)
    xla = max((build, select, deflate), key=lambda t: t[0] / ranks + t[1])
    return memory.Price(
        operands=memory.Term(0.0, float(faces_per_rank * ranks + _C128 * n_rows_pad * m_pad),
                             "sketch 16 R M/P and one k batch's WFN face"),
        xla=memory.Term(float(xla[1]), float(xla[0]),
                        "sketch build tile / select block / deflation (closed form)"),
        ranks=ranks, label="kmeans candidate sketch (proposal)")


def _device_peak_bytes() -> int | None:
    peaks = []
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        if stats.get("peak_bytes_in_use") is not None:
            peaks.append(int(stats["peak_bytes_in_use"]))
    return max(peaks) if peaks else None


def prune_candidates_by_sketch(
    wfn,
    sym,
    cand_idx: np.ndarray,
    n_keep: int,
    mesh: Mesh,
    *,
    band_range_left: tuple[int, int],
    band_range_right: tuple[int, int],
    band_norms: np.ndarray | None = None,
    k_weights: np.ndarray | None = None,
    verbose: bool = True,
    bispinor: bool = False,
    gamma_mode: str = "charge",
    orbit_id: np.ndarray | None = None,
    tol_rel: float | None = None,
    n_point_budget: int | None = None,
    sketch_rows: int = 0,
    sketch_block: int = 0,
    sketch_gamma: float = GAMMA_DEFAULT,
    seed: int = 0,
):
    """End-to-end pruning: WFN faces -> sketch -> orbit-blocked QRCP -> keep.

    The drop-in counterpart of
    :func:`centroid.pivoted_cholesky.prune_candidates_by_pivoted_cholesky`
    (``group_block=True`` semantics): the same windows, k weights, band norms,
    budget in points, whole orbits, and refusal policy
    (:func:`~centroid.pivoted_cholesky.refuse_unless_select_certified`).

    ``sketch_rows`` (``r'``; 0: :func:`sketch_rows_default`) must be at least
    the point budget: a rank-``r'`` sketch cannot certify more directions.
    ``sketch_block`` is the orbits per block (0: :func:`sketch_block_default`);
    ``sketch_gamma`` the in-block acceptance cut; ``seed`` the PRNG seed.
    Charge mode only: ``gamma_mode='transverse'`` refuses in one line.

    Returns ``(keep_idx, rank, d_final, d_taken, trR_over_trG, psd_info,
    unpicked)``.  ``rank`` counts directions above the floor ``tol_rel *
    max d0`` and is estimated on the sketch; ``d_final`` is the unbiased
    residual ``||Y[:, a]||^2 r'/(r' - t)`` on the host (zero on picked
    points); ``trR_over_trG`` is the residual trace ratio at each block
    boundary; ``psd_info = (0.0, -1, -1)`` (PSD by construction);
    ``unpicked`` the largest never-picked residual over the floor.
    """
    if str(gamma_mode).strip().lower() != "charge":
        raise ValueError("--select sketch implements the charge Gram only; prune "
                         "current-mode (transverse) pools with --select cholesky")
    if not 0.0 < float(sketch_gamma) <= 1.0:
        raise ValueError(f"--sketch-gamma must lie in (0, 1]; got {sketch_gamma}")
    dist.require_axes(mesh, dist.MESH_AXES, "prune_candidates_by_sketch")
    M = int(np.asarray(cand_idx).shape[0])
    left = (int(band_range_left[0]), int(band_range_left[1]))
    right = (int(band_range_right[0]), int(band_range_right[1]))
    n_tot = int(wfn.nbands)
    if left[1] > n_tot or right[1] > n_tot:
        raise ValueError(f"wfn.nbands={n_tot} < max(left={left[1]}, right={right[1]})")
    check_pair_space_window(wfn, max(left[1], right[1]))
    budget = min(int(n_point_budget if orbit_id is not None and n_point_budget
                     is not None else n_keep), M)
    n_rows = int(sketch_rows) if int(sketch_rows) > 0 else sketch_rows_default(budget)
    if n_rows < budget:
        raise ValueError(f"--sketch-rows {n_rows} is below the point budget {budget}: "
                         f"a rank-{n_rows} sketch cannot certify {budget} directions")
    n_rows_pad = sketch_row_carrier(n_rows, mesh)
    if orbit_id is None:
        dense_id = np.arange(M, dtype=np.int32)
        n_groups = M
    else:
        _, dense_id = np.unique(np.asarray(orbit_id), return_inverse=True)
        dense_id = dense_id.astype(np.int32)
        n_groups = int(dense_id.max()) + 1
    sizes = np.bincount(dense_id, minlength=n_groups)
    s_max = int(sizes.max())
    n_slots = (int(sketch_block) if int(sketch_block) > 0
               else sketch_block_default(s_max, n_rows_pad))
    n_slots = max(1, min(n_slots, n_groups))

    from runtime.padding import padded_mu_extent
    from lxkit import memory
    from common.gpu_utils import device_budget_bytes
    m_pad = padded_mu_extent(M, mesh)
    ns = 4 if bispinor else int(wfn.nspinor)
    per_k = _face_bytes_per_k(mesh, ns=ns, m_pad=m_pad, left=left, right=right)
    k_batch = _candidate_k_schedule(wfn, sym, k_weights, per_k)[4]
    m_b = n_slots * s_max
    proposal = _memory_proposal(mesh, n_rows_pad=n_rows_pad, m_pad=m_pad, ns=ns,
                                nb_r=right[1] - right[0],
                                faces_per_rank=per_k * k_batch, m_b=m_b)
    device_budget_bytes()
    memory.admit("kmeans candidate sketch (proposal)", proposal,
                 memory.room(bucketed=False))
    terms = proposal.terms()
    log.info(f"[centroid-sketch] state=start candidates={M} groups={n_groups} "
             f"point_budget={budget} rows={n_rows} row_carrier={n_rows_pad} "
             f"block_orbits={n_slots} block_columns={m_b} gamma={float(sketch_gamma):g} "
             f"seed={int(seed)} proposal_GiB/rank: operands="
             f"{terms['operands'] / 2**30:.2f} "
             f"(sketch {_C128 * n_rows_pad * m_pad / mesh.size / 2**30:.2f}) "
             f"workspace={terms['xla'] / 2**30:.2f}")

    t0 = time.perf_counter()
    with timing.section("prune.sketch"):
        Y = build_sketch_q0_via_loadwfns(
            wfn, sym, cand_idx, mesh, n_rows=n_rows, seed=int(seed),
            band_range_left=left, band_range_right=right, band_norms=band_norms,
            k_weights=k_weights, bispinor=bispinor, verbose=verbose)
    t1 = time.perf_counter()
    log.info(f"[centroid-sketch] phase=build state=done rows={n_rows} "
             f"candidates={M} carrier={m_pad} wall_s={t1 - t0:.3f}")
    with timing.section("prune.select"):
        (piv, rank, d_final, d_taken, trR_over_trG, psd_info, n_blocks,
         d0max) = sketch_block_select(
            Y, mesh, group_id=dense_id, n_groups=n_groups, point_budget=budget,
            n_slots=n_slots, gamma=float(sketch_gamma), n_rows=n_rows,
            tol_rel=tol_rel, verbose=verbose)
    t2 = time.perf_counter()
    if piv.size == 0:
        raise RuntimeError(
            f"sketch select REFUSES: point budget {budget} is smaller than every "
            f"complete group (minimum size {int(sizes.min())})")
    keep_idx, _, order, piv_used = _emit_complete_groups(cand_idx, dense_id, piv)
    refuse_unless_select_certified(
        piv_used, rank, psd_info, n_keep=int(keep_idx.shape[0]), M=M,
        M_pad=m_pad, orbit_id=None, d0max=d0max, tol_rel=tol_rel)
    peak = _device_peak_bytes()
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    log.info(f"[centroid-sketch] phase=select state=done blocks={n_blocks} "
             f"orbits={len(order)} points={int(keep_idx.shape[0])}/{budget} "
             f"rank={rank} wall_s={t2 - t1:.3f}")
    log.info("[centroid-sketch] sketch: PSD by construction "
             "(d = ||Y[:,a]||^2 >= 0); the rank and residuals are read on the "
             f"sketch as d r'/(r'-t) (r'={n_rows} >= budget {budget})")
    log.info(f"[centroid-sketch] totals wall_s={t2 - t0:.3f} "
             f"device_peak_GiB={'n/a' if peak is None else f'{peak / 2**30:.2f}'} "
             f"host_peak_GiB={host_peak / 2**30:.2f}")
    return (keep_idx, int(rank), d_final, d_taken, trR_over_trG, psd_info,
            _unpicked_over_floor(d_final, d0max, tol_rel))


__all__ = [
    "BLOCK_COLUMNS",
    "CAND",
    "GAMMA_DEFAULT",
    "build_sketch_q0_via_loadwfns",
    "prune_candidates_by_sketch",
    "sketch_block_columns",
    "sketch_block_default",
    "sketch_block_select",
    "sketch_from_faces",
    "sketch_row_carrier",
    "sketch_rows_default",
]
