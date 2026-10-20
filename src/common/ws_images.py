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
* :func:`pair_ws_code_table` / :func:`pair_ws_code_fourier` — the same
  interpolant from a compact per-element tie code and a per-k phase table,
  for repeated k batches against one class-major tensor (no weight stack).

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

__all__ = ["WS_TIE_TOL_BOHR2", "lattice_bohr", "pair_ws_code_fourier",
           "pair_ws_code_table", "pair_ws_embed", "pair_ws_fourier",
           "pair_ws_images", "pair_ws_weights"]

#: Absolute tie tolerance on squared distances (bohr²), the one tolerance of
#: every image selection.  Symmetry-related images agree to rounding
#: (~1e-12 bohr² at |R| ~ 100 bohr); distinct lattice distances of a real
#: cell differ by far more.
WS_TIE_TOL_BOHR2 = 1.0e-6


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


def pair_ws_weights(pos, a_c, b_c, valid_c, tol, *, pin=None,
                    dtype=jnp.float64):
    """Weights ``(n, n, n_t)`` of ONE class's images for every pair.

    ``|v_t + d|² = a_t + 2 d·b_t + |d|²`` with ``d = r_i − r_j``; the
    ``|d|²`` term is common to every ``t``, so ``a_t + 2(s_i − s_j)``,
    ``s = pos·b_t``, ranks the images and its differences are squared
    distances.  The minimal ones within ``tol`` share ``1/N_deg``; the
    transposed pair sees the negated numbers.  ``pin`` optionally constrains
    the ``(n, n, n_t)`` layout inside a jit.
    """
    s = jnp.asarray(pos) @ jnp.asarray(b_c).T                  # (n, n_t)
    d2 = jnp.asarray(a_c)[None, None, :] + 2.0 * (s[:, None, :] - s[None, :, :])
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


def pair_ws_code_table(images: dict) -> dict:
    """Compact form of :func:`pair_ws_weights` for every class and pair.

    Element ``(c, i, j)``'s tie set (its minimal images, weight ``1/N_deg``
    each) is a bitmask over the class's candidate slots, decided by
    :func:`pair_ws_weights` itself, so there is one rule.  Distinct masks over
    all classes are interned as patterns.

    Returns a dict of host arrays: ``code`` (n_c, n, n) int16 pattern index,
    ``weights`` (n_pat, n_t) the ``1/N_deg`` weight of each slot in each
    pattern, ``img``/``valid`` from ``images`` and ``kgrid``.
    """
    n_t = int(images["img"].shape[1])
    if n_t > 62:
        raise ValueError(f"pair_ws_code_table: {n_t} candidate images per "
                         "class exceed the 62-bit tie mask")
    pos = images["pos"]
    bits = np.left_shift(np.int64(1), np.arange(n_t, dtype=np.int64))
    masks = np.stack([
        (np.asarray(pair_ws_weights(pos, a_c, b_c, v_c, float(images["tol"])))
         > 0).astype(np.int64) @ bits
        for a_c, b_c, v_c in zip(images["a"], images["b"], images["valid"])])
    patterns, code = np.unique(masks, return_inverse=True)
    if patterns.size >= np.iinfo(np.int16).max:
        raise ValueError(f"pair_ws_code_table: {patterns.size} tie patterns "
                         "exceed the int16 code")
    member = (patterns[:, None] & bits[None, :]) != 0
    weights = member / member.sum(axis=1, keepdims=True)
    return dict(code=code.reshape(masks.shape).astype(np.int16),
                weights=weights, img=images["img"], valid=images["valid"],
                kgrid=images["kgrid"])


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
    code = jnp.asarray(table["code"]) if code is None else code
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
