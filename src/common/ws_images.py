"""Per-pair Wigner–Seitz images of a coarse k-grid supercell.

A coarse uniform k grid ``kgrid`` fixes a lattice tensor ``X(R)_ij`` only on
the classes ``R + L·T`` of its supercell ``L = diag(kgrid)``.  For a tensor
indexed by POSITIONS — element ``(i, j)`` couples site ``i`` at ``r_i + R``
with site ``j`` at ``r_j``, as in the full-Bloch gauge of the ISDF centroid
tiles, ``W_ij(q) = Σ_R e^{−2πiq·R} W(r_i + R, r_j)`` — the covariant off-grid
interpolant keeps, for every pair and class, the images of minimal
``|R + L T + r_i − r_j|`` with weight ``1/N_deg`` (Wannier90's
``use_ws_distance``).  A lattice-coordinate box, or one image set shared by
every pair, is not covariant under an operation that folds the sites into the
home cell non-uniformly.

The one implementation for every position-indexed consumer:

* :func:`pair_ws_images` — host table of the candidate images of every R
  class, complete by a rigorous radius bound;
* :func:`pair_ws_weights` — the per-pair weights of one class (traceable);
* :func:`pair_ws_embed` — elementwise application: the coarse ``X(R)``
  placed on a finer R lattice, whose FFT is the interpolant on that grid;
* :func:`pair_ws_fourier` — the same interpolant at an explicit k list;
* :func:`pair_ws_code_table` / :func:`pair_ws_code` /
  :func:`pair_ws_code_fourier` — the same interpolant from a compact tie code
  (one per class and distinct pair displacement) and a per-k phase table, for
  repeated k batches against one class-major tensor (no weight stack).

Conventions.  Positions are lattice (crystal) coordinates in the tensor's
own gauge (the ISDF centroids sit at FFT index / N, in ``[0, 1)``); the
lattice is ``(3, 3)`` rows ``a_i`` in bohr (:func:`lattice_bohr`); the R
classes are in the C order of the ``ifftn`` index (``R_i = n`` for
``n < ⌈N_i/2⌉``, else ``n − N_i``).  A singleton k axis (a slab's vacuum
axis) carries ``T_i = 0``: its periodic images are not physical neighbours.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["WS_TIE_TOL_BOHR2", "lattice_bohr", "pair_ws_code", "pair_ws_local_codes",
           "pair_ws_code_fourier", "pair_ws_code_table", "pair_ws_embed",
           "pair_ws_fourier", "pair_ws_images", "pair_ws_weights"]

#: Absolute tie tolerance on squared distances (bohr²), the one tolerance of
#: every image selection.  Symmetry-related images agree to rounding
#: (~1e-12 bohr² at |R| ~ 100 bohr); distinct lattice distances of a real
#: cell differ by far more.
WS_TIE_TOL_BOHR2 = 1.0e-6

#: Pair displacements (lattice coordinates) that agree to ``2**-DISP_BITS``
#: are one displacement in :func:`pair_ws_code_table`.  Grid positions (the
#: ISDF centroids, FFT index / N) repeat a displacement exactly up to
#: rounding, and distinct ones differ by at least ``1/N``.
DISP_BITS = 19


def lattice_bohr(wfn) -> np.ndarray:
    """Real-space lattice ``(3, 3)`` in bohr, ROWS ``a_i`` (``wfn.avec·alat``)."""
    return np.asarray(wfn.avec, dtype=np.float64) * float(wfn.alat)


def _class_vectors(grid: np.ndarray) -> np.ndarray:
    """R class representatives ``(n_c, 3)`` in ``ifftn`` index C order."""
    axes = [np.where(np.arange(n) >= (n + 1) // 2, np.arange(n) - n,
                     np.arange(n)) for n in grid]
    return np.stack(np.meshgrid(*axes, indexing='ij'),
                    axis=-1).reshape(-1, 3).astype(np.float64)


def _translations(t_max: np.ndarray) -> np.ndarray:
    """All integer ``T`` with ``|T_i| <= t_max_i``, ``(n_T, 3)`` float."""
    return np.stack(np.meshgrid(*[np.arange(-t, t + 1) for t in t_max],
                                indexing='ij'), axis=-1).reshape(-1, 3) \
        .astype(np.float64)


def _t_bound(base, rho, grid, b_norm, periodic):
    """``|T_i| <= ⌊(|b_i| ρ + |R_i|)/N_i⌋`` per class and axis (0 if singleton)."""
    t = np.floor((rho[:, None] * b_norm[None, :] + np.abs(base))
                 / grid[None, :] + 1e-12).astype(np.int64)
    return np.where(periodic[None, :], t, 0)


def pair_ws_images(kgrid, lattice_bohr, pos_frac, sys_dim: int, *,
                   n_carrier: int | None = None) -> dict:
    """Candidate Wigner–Seitz images of every R class for a set of positions.

    For class ``c`` (representative ``R_c``) the images are ``v = R_c + L T``.
    A pair displacement ``d = r_i − r_j`` takes the minimal ``|v + d|``.  With
    ``v₀`` the shortest image of the class (``d = 0``) and ``ρ = |v₀| +
    2 max|d|``, every minimiser satisfies ``|v| ≤ ρ``, so its lattice
    coordinate obeys ``|v_i| ≤ |b_i| ρ`` (``b_i`` the dual vectors,
    ``a_i·b_j = δ_ij``) and ``|T_i| ≤ ⌊(|b_i| ρ + |R_i|)/N_i⌋``.  Inside
    that box an image is kept only if it can beat ``v₀`` for some ``d`` in
    the displacement box (``|v|² − |v₀|² ≤ 2 max_d d·G(v₀ − v)`` plus the
    tie tolerance), a necessary condition, so the table is complete.

    Parameters
    ----------
    kgrid : (3,) int
        Coarse k grid; axes beyond ``sys_dim`` must be singletons.
    lattice_bohr : (3, 3) rows ``a_i`` in bohr.
    pos_frac : (n, 3) site positions in lattice coordinates, in the gauge of
        the tensor (ISDF centroids: FFT index / N, in ``[0, 1)``).
    sys_dim : 2 or 3, the periodic dimension of the deck.
    n_carrier : optional carrier length ``>= n``; pad rows (zero tensor
        rows) take the first site's position, inside the displacement box.

    Returns
    -------
    dict of host arrays: ``img`` (n_c, n_t, 3) int64 image vectors (lattice
    coords), ``valid`` (n_c, n_t), ``a`` (n_c, n_t) ``|v|²`` and ``b``
    (n_c, n_t, 3) ``G v`` in bohr², ``pos`` (n_carrier, 3), ``n_logical``,
    ``tol`` (:data:`WS_TIE_TOL_BOHR2`), ``kgrid``.
    """
    grid = np.asarray([int(v) for v in kgrid], dtype=np.int64)
    lat = np.asarray(lattice_bohr, dtype=np.float64)
    r = np.asarray(pos_frac, dtype=np.float64)
    if grid.shape != (3,) or np.any(grid <= 0) or lat.shape != (3, 3):
        raise ValueError(f"pair_ws_images: invalid k grid {tuple(grid)} or "
                         f"lattice shape {lat.shape}")
    if int(sys_dim) not in (2, 3) or np.any(grid[int(sys_dim):] != 1):
        raise ValueError(f"pair_ws_images: sys_dim {sys_dim} needs singleton "
                         f"k axes beyond it; got k grid {tuple(grid)}")
    if r.ndim != 2 or r.shape[1] != 3 or r.shape[0] == 0 \
            or not np.all(np.isfinite(r)):
        raise ValueError(f"pair_ws_images: positions must be finite (n, 3) "
                         f"lattice coordinates; got shape {r.shape}")
    n = int(r.shape[0])
    n_car = n if n_carrier is None else int(n_carrier)
    if n_car < n:
        raise ValueError(f"pair_ws_images: carrier {n_car} < {n} positions")
    G = lat @ lat.T
    periodic = grid > 1
    span = r.max(axis=0) - r.min(axis=0)
    corners = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
    d_max = float(np.linalg.norm((corners * span) @ lat, axis=1).max())
    b_norm = np.linalg.norm(np.linalg.inv(lat).T, axis=1)
    base = _class_vectors(grid)
    norm = lambda v: np.sqrt(np.maximum(np.einsum('...i,ij,...j->...', v, G, v), 0.0))

    # Shortest image v0 of each class (d = 0): |v0| <= |R_c|.
    t0 = _t_bound(base, norm(base), grid, b_norm, periodic).max(axis=0)
    T = _translations(t0)
    img = base[:, None, :] + (T * grid)[None, :, :]
    a = np.einsum('kti,ij,ktj->kt', img, G, img)
    v0 = img[np.arange(img.shape[0]), np.argmin(a, axis=1)]

    # Every image that can be minimal for some displacement in the box.
    rho = norm(v0) + 2.0 * d_max + np.sqrt(WS_TIE_TOL_BOHR2)
    t_c = _t_bound(base, rho, grid, b_norm, periodic)
    T = _translations(t_c.max(axis=0))
    img = base[:, None, :] + (T * grid)[None, :, :]
    a = np.einsum('kti,ij,ktj->kt', img, G, img)
    inside = np.all(np.abs(T)[None, :, :] <= t_c[:, None, :], axis=-1)
    gvec = np.einsum('ij,ktj->kti', G, v0[:, None, :] - img)
    gain = 2.0 * (np.abs(gvec) * span[None, None, :]).sum(axis=-1)
    a0 = np.einsum('ki,ij,kj->k', v0, G, v0)
    keep = inside & (a - a0[:, None] <= gain + WS_TIE_TOL_BOHR2)

    n_t = int(keep.sum(axis=1).max())
    out = np.zeros((base.shape[0], n_t, 3), dtype=np.int64)
    valid = np.zeros((base.shape[0], n_t), dtype=bool)
    for c in range(base.shape[0]):
        sel = np.rint(img[c, keep[c]]).astype(np.int64)
        out[c, :sel.shape[0]] = sel
        valid[c, :sel.shape[0]] = True
    v = out.astype(np.float64)
    pos = np.concatenate([r, np.repeat(r[:1], n_car - n, axis=0)])
    return dict(img=out, valid=valid,
                a=np.where(valid, np.einsum('kti,ij,ktj->kt', v, G, v), 0.0),
                b=np.einsum('kti,ij->ktj', v, G), pos=pos, n_logical=n,
                tol=WS_TIE_TOL_BOHR2, kgrid=tuple(int(x) for x in grid))


def pair_ws_weights(pos, a_c, b_c, valid_c, tol, *, pos_b=None, pin=None,
                    dtype=jnp.float64):
    """Weights ``(n, m, n_t)`` of ONE class's images for every pair.

    ``|v_t + d|² = a_t + 2 d·b_t + |d|²`` with ``d = r_i − r_j`` (``r_i``
    from ``pos``, ``r_j`` from ``pos_b``, default ``pos``); the ``|d|²`` term
    is common to every ``t``, so ``a_t + 2(s_i − s_j)``, ``s = pos·b_t``,
    ranks the images and its differences are squared distances.  The minimal
    ones within ``tol`` share ``1/N_deg``; the transposed pair sees the
    negated numbers.  ``pin`` optionally constrains the ``(n, m, n_t)``
    layout inside a jit.
    """
    s = jnp.asarray(pos) @ jnp.asarray(b_c).T                  # (n, n_t)
    s_b = s if pos_b is None else jnp.asarray(pos_b) @ jnp.asarray(b_c).T
    d2 = jnp.asarray(a_c)[None, None, :] + 2.0 * (s[:, None, :] - s_b[None, :, :])
    if pin is not None:
        d2 = pin(d2)
    d2 = jnp.where(jnp.asarray(valid_c)[None, None, :], d2, jnp.inf)
    w = (d2 <= jnp.min(d2, axis=-1, keepdims=True) + tol).astype(dtype)
    return w / jnp.sum(w, axis=-1, keepdims=True)


def pair_ws_embed(X_R, images: dict, fine_grid, *, pin=None):
    """Place a coarse ``X(R)`` on a finer R lattice, pair by pair.

    ``X_R`` is ``(n, n, *kgrid)`` (``ifftn`` R order, rows and columns on
    ``images['pos']``).  Element ``(i, j)`` of class ``c`` goes to the fine
    slots ``(R_c + L T) mod fine_grid`` with its weights
    (:func:`pair_ws_weights`); ``Σ_T w = 1``, so an ``fftn`` over the fine
    lattice reproduces ``X`` at every k common to both grids.  No FFT
    normalisation is applied.  Traceable; ``pin`` constrains the
    ``(n, n, ·)`` layouts (carry and weights) inside a jit.
    """
    coarse = tuple(int(v) for v in images["kgrid"])
    fine = tuple(int(v) for v in fine_grid)
    if tuple(X_R.shape[2:]) != coarse or any(f < c for f, c in zip(fine, coarse)):
        raise ValueError(f"pair_ws_embed: tensor grid {tuple(X_R.shape[2:])}, "
                         f"table grid {coarse}, fine grid {fine} (each fine "
                         f"extent must be at least the coarse one)")
    n = int(X_R.shape[0])
    if int(X_R.shape[1]) != n or int(images["pos"].shape[0]) != n:
        raise ValueError(f"pair_ws_embed: tensor {tuple(X_R.shape[:2])} and "
                         f"{images['pos'].shape[0]} positions disagree")
    keep = pin if pin is not None else (lambda x: x)
    fi = np.mod(images["img"], np.asarray(fine))
    slot = np.where(images["valid"],
                    (fi[..., 0] * fine[1] + fi[..., 1]) * fine[2] + fi[..., 2],
                    0).astype(np.int32)
    n_c = int(np.prod(coarse))
    Xc = jnp.moveaxis(X_R.reshape(n, n, n_c), -1, 0)
    pos = jnp.asarray(images["pos"])
    tol = float(images["tol"])

    def _class(acc, xs):
        xc, a_c, b_c, v_c, s_c = xs
        w = pair_ws_weights(pos, a_c, b_c, v_c, tol, pin=pin,
                            dtype=X_R.real.dtype)
        return keep(acc.at[:, :, s_c].add(xc[:, :, None] * w)), None

    acc0 = keep(jnp.zeros((n, n, int(np.prod(fine))), dtype=X_R.dtype))
    acc, _ = jax.lax.scan(
        _class, acc0,
        (Xc, jnp.asarray(images["a"]), jnp.asarray(images["b"]),
         jnp.asarray(images["valid"]), jnp.asarray(slot)))
    return keep(acc.reshape((n, n) + fine))


def pair_ws_fourier(X_R, images: dict, k, *, pin=None):
    """The per-pair interpolant at explicit k points, ``(nk, n, n)``.

    ``X(k)_ij = Σ_c Σ_T w_ij(c, T) e^{−2πik·(R_c + L T)} X(R_c)_ij`` for a
    coarse ``X_R`` ``(n, n, *kgrid)`` in ``ifftn`` R order with
    ``X(k_c) = Σ_R e^{−2πik_c·R} X(R)`` on the coarse grid (``k`` in lattice
    coordinates of the reciprocal basis).  Equal to an ``fftn`` of
    :func:`pair_ws_embed` at every point of a uniform fine grid; batch ``k``
    to bound the ``(nk, n, n)`` carry.  Traceable; ``pin`` constrains the
    ``(·, n, n)``-trailing layouts.
    """
    coarse = tuple(int(v) for v in images["kgrid"])
    if tuple(X_R.shape[2:]) != coarse:
        raise ValueError(f"pair_ws_fourier: tensor grid {tuple(X_R.shape[2:])} "
                         f"!= table grid {coarse}")
    n = int(X_R.shape[0])
    keep = pin if pin is not None else (lambda x: x)
    k = jnp.atleast_2d(jnp.asarray(k, dtype=jnp.float64))
    Xc = jnp.moveaxis(X_R.reshape(n, n, int(np.prod(coarse))), -1, 0)
    pos = jnp.asarray(images["pos"])
    tol = float(images["tol"])
    vec = jnp.asarray(images["img"], dtype=jnp.float64)       # (n_c, n_t, 3)

    def _class(acc, xs):
        xc, a_c, b_c, v_c, img_c = xs
        w = pair_ws_weights(pos, a_c, b_c, v_c, tol, dtype=X_R.real.dtype)
        ph = jnp.where(v_c[None, :], jnp.exp(-2j * jnp.pi * (k @ img_c.T)), 0.0)
        return keep(acc + jnp.einsum('qt,ijt->qij', ph, w) * xc[None]), None

    acc0 = keep(jnp.zeros((k.shape[0], n, n), dtype=X_R.dtype))
    acc, _ = jax.lax.scan(
        _class, acc0,
        (Xc, jnp.asarray(images["a"]), jnp.asarray(images["b"]),
         jnp.asarray(images["valid"]), vec))
    return acc


def _pair_displacements(pos: np.ndarray):
    """Distinct pair displacements: ``(disp (n_d, 3), pair (n, n) int32)``.

    ``pair[i, j]`` indexes the displacement ``r_i − r_j`` in ``disp`` (the
    first pair's own value; displacements equal to ``2**-DISP_BITS`` are
    one).  Work and memory ``O(n²)`` integers, one sort.
    """
    n = int(pos.shape[0])
    d = (pos[:, None, :] - pos[None, :, :]).reshape(n * n, 3)
    q = np.rint(d * float(1 << DISP_BITS)).astype(np.int64)
    q -= q.min(axis=0)
    span = q.max(axis=0) + 1
    if float(np.prod(span.astype(np.float64))) < 2.0 ** 62:
        key = (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]
        _, first, pair = np.unique(key, return_index=True, return_inverse=True)
    else:
        _, first, pair = np.unique(q, axis=0, return_index=True,
                                   return_inverse=True)
    return d[first], pair.reshape(n, n).astype(np.int32)


@jax.jit
def _class_ties(disp, a, b, valid, tol):
    """Per class and displacement: the tie mask (int64 over the candidate
    slots), its first slot and whether it is a single image, decided by
    :func:`pair_ws_weights` against the origin; one class at a time."""
    bits = jnp.left_shift(jnp.int64(1), jnp.arange(a.shape[1], dtype=jnp.int64))
    origin = jnp.zeros((1, 3), disp.dtype)

    def one(xs):
        tie = pair_ws_weights(disp, xs[0], xs[1], xs[2], tol, pos_b=origin)[:, 0] > 0
        return (jnp.sum(jnp.where(tie, bits, 0), axis=-1),
                jnp.argmax(tie, axis=-1).astype(jnp.int32),
                jnp.sum(tie, axis=-1) == 1)

    return jax.lax.map(one, (a, b, valid))


def _tie_pattern_codes(masks, first, single, n_t):
    """Intern single-image patterns without sorting the full pair stack.

    Only entries with several minimal images need a mask sort. The usual
    single-image entry reads its code from the small image-slot lookup.
    """
    bits = np.left_shift(np.int64(1), np.arange(n_t, dtype=np.int64))
    present = np.bincount(first[single], minlength=n_t)[:n_t] > 0
    patterns = np.union1d(bits[present], np.unique(masks[~single]))
    if patterns.size >= np.iinfo(np.int16).max:
        raise ValueError("WS tie patterns exceed the int16 code")
    code = np.searchsorted(patterns, bits).astype(np.int16)[first]
    code[~single] = np.searchsorted(patterns, masks[~single])
    return patterns, code


def pair_ws_code_table(images: dict) -> dict:
    """Compact form of :func:`pair_ws_weights` for every class and pair.

    Element ``(c, i, j)``'s tie set (its minimal images, weight ``1/N_deg``
    each) depends on the pair only through its displacement
    ``d = r_i − r_j``, so it is decided once per distinct displacement, by
    :func:`pair_ws_weights` itself (``pos_b`` = the origin), so there is one
    rule; it is a bitmask over the class's candidate slots.  Distinct masks
    over all classes are interned as patterns, ascending.  Work
    ``n_c·n_d·n_t`` for ``n_d`` distinct displacements (grid positions repeat
    them: Si 1100 centroid nodes have 1.21e6 pairs), plus one ``O(n²)``
    integer sort; the full ``(n_c, n, n)`` code is never formed here.

    Returns a dict of host arrays: ``class_code`` (n_c, n_d) int16 pattern
    index of each class and displacement, ``pair`` (n, n) int32 the
    displacement of each pair (:func:`pair_ws_code` expands any block),
    ``weights`` (n_pat, n_t) the ``1/N_deg`` weight of each slot in each
    pattern, ``img``/``valid`` from ``images`` and ``kgrid``.
    """
    n_t = int(images["img"].shape[1])
    if n_t > 62:
        raise ValueError(f"pair_ws_code_table: {n_t} candidate images per "
                         "class exceed the 62-bit tie mask")
    disp, pair = _pair_displacements(np.asarray(images["pos"], np.float64))
    bits = np.left_shift(np.int64(1), np.arange(n_t, dtype=np.int64))
    masks, first, single = (np.asarray(v) for v in _class_ties(
        jnp.asarray(disp), jnp.asarray(images["a"]), jnp.asarray(images["b"]),
        jnp.asarray(images["valid"]), float(images["tol"])))
    # Patterns = the distinct masks, ascending (np.unique's order): the
    # one-image masks present, by count, and the tie masks, by a sort of the
    # (few) entries that tie.
    patterns, class_code = _tie_pattern_codes(masks, first, single, n_t)
    member = (patterns[:, None] & bits[None, :]) != 0
    weights = member / member.sum(axis=1, keepdims=True)
    return dict(class_code=class_code, pair=pair, weights=weights,
                img=images["img"], valid=images["valid"],
                kgrid=images["kgrid"])


def pair_ws_local_codes(images: dict, blocks) -> tuple[list[np.ndarray], np.ndarray]:
    """Tie codes for only the requested rectangular site-pair blocks.

    Parameters
    ----------
    images : dict
        Candidate images from :func:`pair_ws_images`.
    blocks : sequence of (rows, cols)
        Site-index arrays into ``images['pos']`` for each addressable face
        tile. Negative indices denote exact-null pad rows.

    Returns
    -------
    codes : list of ndarray
        ``(n_class, len(rows), len(cols))`` int16, indexing ``patterns``.
    patterns : ndarray
        Local tie masks, int64. The caller agrees these small dictionaries
        across processes and remaps the codes before building a shared LUT.

    Collective: every process enters. Only requested pairs are evaluated,
    in bounded displacement blocks; repeated spin-site rows are evaluated
    once. The exact tie-kernel executables are admitted before they run.
    In particular a target-by-node back-map never constructs target-by-target
    displacements or a replicated all-site integer sort. The image decision
    remains :func:`pair_ws_weights`, through ``_class_ties``.
    """
    from runtime.tiles import tile_units
    from runtime.aot_memory import agreed_chunk, compiled_new_bytes
    from common.collectives import all_gather_processes
    from common.gpu_utils import shared_device_room
    from lxkit import memory

    n_t = int(images["img"].shape[1])
    if n_t > 62:
        raise ValueError(f"pair_ws_local_codes: {n_t} candidate images per "
                         "class exceed the 62-bit tie mask")
    n_c = int(images["img"].shape[0])
    pos = np.asarray(images["pos"], dtype=np.float64)
    patterns, lookup, codes = [], {}, []
    # The returned masks, their sort/inverse buffers, and the weight kernel's
    # one-class image stack all fit the fixed tile, independent of N_mu.
    per_pair = 64 * n_c + 32 * n_t + 48
    # Spin components repeat the same scalar site, so only the distinct row
    # and column indices need a geometric image decision.
    blocks = [(np.unique(rows, return_inverse=True),
               np.unique(cols, return_inverse=True)) for rows, cols in blocks]
    widths = np.asarray(sorted({tile_units(per_pair, len(r[0]) * len(c[0]))
                                for r, c in blocks}), dtype=np.int64)
    count = int(np.max(all_gather_processes(np.array([widths.size], np.int64))))
    widths = np.unique(all_gather_processes(np.pad(widths, (0, count - widths.size))))
    widths = widths[widths > 0]
    with jax.default_device(jax.local_devices()[0]):
        a, b, valid = (jnp.asarray(images[k]) for k in ("a", "b", "valid"))
        kernels, peak = {}, 0
        for width in widths:
            kernel = _class_ties.lower(
                jax.ShapeDtypeStruct((int(width), 3), jnp.float64),
                a, b, valid, float(images["tol"])).compile()
            kernels[int(width)] = kernel
            peak = max(peak, compiled_new_bytes(kernel)
                       + int(kernel.memory_analysis().argument_size_in_bytes))
        peak = -agreed_chunk(-peak)
        stage = "htransform WS tie-mask tile"
        room = shared_device_room(stage)
        memory.admit(stage, (("local tile", memory.Price(
            xla=memory.Term(float(peak), 0.0, "tie-mask tile (compiled program)"),
            ranks=int(jax.device_count()), label=stage)),),
            int(room) if isinstance(room, (int, np.integer)) else 0)
        for (rows, row_inverse), (cols, col_inverse) in blocks:
            nr, nc = len(rows), len(cols)
            out = np.empty((n_c, nr * nc), dtype=np.int16)
            width = tile_units(per_pair, nr * nc)
            if width == 0:
                codes.append(np.empty((n_c, row_inverse.size, col_inverse.size), np.int16))
                continue
            for lo in range(0, nr * nc, width):
                flat = np.arange(lo, min(lo + width, nr * nc))
                ir, ic = rows[flat // nc], cols[flat % nc]
                disp = pos[np.maximum(ir, 0)] - pos[np.maximum(ic, 0)]
                # One compiled shape per block width, including its short
                # tail. Padded decisions are discarded before interning.
                disp = np.pad(disp, ((0, width - flat.size), (0, 0)))
                masks, first, single = (np.asarray(v)[:, :flat.size] for v in
                    kernels[width](jnp.asarray(disp), a, b, valid, float(images["tol"])))
                found, block_code = _tie_pattern_codes(masks, first, single, n_t)
                ids = []
                for mask in found:
                    mask = int(mask)
                    if mask not in lookup:
                        if len(patterns) >= np.iinfo(np.int16).max:
                            raise ValueError("pair_ws_local_codes: tie patterns exceed the int16 code")
                        lookup[mask] = len(patterns)
                        patterns.append(mask)
                    ids.append(lookup[mask])
                out[:, lo:lo + flat.size] = np.asarray(ids, np.int16)[block_code]
            codes.append(out.reshape(n_c, nr, nc)[:, row_inverse[:, None], col_inverse[None, :]])
    return codes, np.asarray(patterns, dtype=np.int64)


def pair_ws_code(table: dict, rows=None, cols=None) -> np.ndarray:
    """Pattern index ``(n_c, m_r, m_c)`` int16 of the pairs ``rows x cols``.

    ``rows``/``cols`` are site indices of :func:`pair_ws_code_table`'s
    positions (default: every site); ``-1`` marks an exact-null pad row or
    column, which takes pattern 0 (its tensor entries are zero).
    """
    n = int(table["pair"].shape[0])
    r = np.arange(n) if rows is None else np.asarray(rows, dtype=np.int64)
    c = np.arange(n) if cols is None else np.asarray(cols, dtype=np.int64)
    out = np.take(table["class_code"],
                  table["pair"][np.ix_(np.maximum(r, 0), np.maximum(c, 0))],
                  axis=1)
    pad = (r < 0)[:, None] | (c < 0)[None, :]
    if pad.any():
        out[:, pad] = 0
    return out


def pair_ws_code_fourier(X_R, table: dict, k, *, code=None, pin=None):
    """:func:`pair_ws_fourier` from the code table, ``(nk, n, n)``.

    ``X_R`` is CLASS-MAJOR ``(n_c, n, n)`` (``ifftn`` R order).  Per k the
    phases of every class's candidate images are folded into one
    ``(n_c, nk, n_pat)`` table, then each element gathers its pattern's
    phase: work ``nk * n_c * n**2``, the box sum's, and no per-pair weight
    stack.  ``code`` optionally supplies the ``(n_c, n, n)`` int code already
    placed on device (e.g. on the tensor's own sharding); ``pin`` constrains
    the ``(nk, n, n)`` carry inside a jit.
    """
    keep = pin if pin is not None else (lambda x: x)
    k = jnp.atleast_2d(jnp.asarray(k, dtype=jnp.float64))
    img = jnp.asarray(table["img"], dtype=jnp.float64)          # (n_c, n_t, 3)
    ph = jnp.where(jnp.asarray(table["valid"])[None],
                   jnp.exp(-2j * jnp.pi * jnp.einsum('qi,cti->qct', k, img)),
                   0.0)
    lut = jnp.einsum('qct,pt->cqp', ph,
                     jnp.asarray(table["weights"], dtype=jnp.float64))
    code = jnp.asarray(pair_ws_code(table)) if code is None else code
    if tuple(X_R.shape) != tuple(code.shape):
        raise ValueError(f"pair_ws_code_fourier: tensor {tuple(X_R.shape)} "
                         f"and code {tuple(code.shape)} disagree")

    def _class(acc, xs):
        x_c, code_c, lut_c = xs
        return keep(acc + jnp.take(lut_c, code_c.astype(jnp.int32), axis=1)
                    * x_c[None]), None

    acc0 = keep(jnp.zeros((k.shape[0],) + tuple(X_R.shape[1:]), X_R.dtype))
    acc, _ = jax.lax.scan(_class, acc0, (X_R, code, lut))
    return acc
