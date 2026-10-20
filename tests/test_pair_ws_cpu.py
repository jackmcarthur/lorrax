"""Per-pair Wigner–Seitz image table and weights of ``common.ws_images`` (CPU).

For a pair displacement d = r_i - r_j and class R of a coarse k grid, the
images R + L T of minimal |R + L T + d| (Wannier90 ``use_ws_distance``,
1/N_deg) must all be in the candidate table; checked by brute force over
|T| <= 6 for fcc, hexagonal-slab and a skewed basis of the fcc lattice whose
minimal images need |T| = 3 (the case a widening search heuristic missed).
Also: coincident positions give one shared Wigner–Seitz set, the weights
sum to 1 and map to the transposed pair under R -> -R, and the embedding
reproduces every coarse sample after the fine FFT; the k-list form equals
the fine FFT of the embedding.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from common.ws_images import (WS_TIE_TOL_BOHR2, pair_ws_embed,  # noqa: E402
                              pair_ws_fourier, pair_ws_images, pair_ws_weights)

FCC = 10.26 * np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
SKEW = np.array([FCC[0], FCC[0] + FCC[1], 2 * FCC[0] - FCC[1] + FCC[2]])
HEX = 5.97 * np.array([[1, 0, 0], [-.5, np.sqrt(3) / 2, 0], [0, 0, 6.0]])
CASES = [((4, 4, 4), FCC, 3), ((3, 3, 3), SKEW, 3), ((9, 9, 1), HEX, 2)]


def _pos(n, kgrid, seed):
    r = np.random.default_rng(seed).random((n, 3))
    if kgrid[2] == 1:
        r[:, 2] = 0.4 + 0.2 * r[:, 2]
    return r


def _weights(tab, c):
    return np.asarray(pair_ws_weights(tab["pos"], tab["a"][c], tab["b"][c],
                                      tab["valid"][c], tab["tol"]))


@pytest.mark.parametrize("kgrid, lat, dim", CASES)
def test_table_holds_every_minimal_image(kgrid, lat, dim):
    r = _pos(14, kgrid, 1)
    tab = pair_ws_images(kgrid, lat, r, dim)
    G = lat @ lat.T
    T = np.stack(np.meshgrid(*[np.arange(-6, 7) if n > 1 else np.zeros(1)
                               for n in kgrid], indexing="ij"), -1).reshape(-1, 3)
    d = (r[:, None, :] - r[None, :, :]).reshape(-1, 3)
    Ng = np.asarray(kgrid)
    for c in range(tab["img"].shape[0]):
        rep = np.mod(tab["img"][c, 0], Ng)
        rep = np.where(rep >= (Ng + 1) // 2, rep - Ng, rep)  # box representative
        allv = rep + T * Ng                                  # the class, |T| <= 6
        u = allv[None, :, :] + d[:, None, :]
        d2 = np.einsum("pti,ij,ptj->pt", u, G, u)
        best = d2 <= d2.min(axis=1, keepdims=True) + WS_TIE_TOL_BOHR2
        cand = {tuple(v) for v in tab["img"][c][tab["valid"][c]]}
        need = {tuple(np.rint(allv[t]).astype(int)) for t in np.unique(np.nonzero(best)[1])}
        assert need <= cand
        w = _weights(tab, c).reshape(-1, tab["img"].shape[1])
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-14)
        img = tab["img"][c]
        for p in range(len(d)):                               # weights = 1/N_deg on the minima
            sel = {tuple(img[t]) for t in np.flatnonzero(w[p] > 0)}
            mins = {tuple(np.rint(allv[t]).astype(int)) for t in np.flatnonzero(best[p])}
            assert sel == mins
    if lat is SKEW:                     # minimal images 3 supercells from the box
        N = np.asarray(kgrid)
        box = np.mod(tab["img"], N)
        box = np.where(box >= (N + 1) // 2, box - N, box)
        assert np.abs((tab["img"] - box) // N)[tab["valid"]].max() >= 3


def test_coincident_positions_give_one_shared_set():
    tab = pair_ws_images((4, 4, 4), FCC, np.full((5, 3), 0.25), 3)
    G = FCC @ FCC.T
    for c in range(tab["img"].shape[0]):
        w = _weights(tab, c)
        v = tab["img"][c].astype(float)
        a = np.einsum("ti,ij,tj->t", v, G, v)
        ref = (tab["valid"][c] & (a <= a[tab["valid"][c]].min() + WS_TIE_TOL_BOHR2)).astype(float)
        np.testing.assert_allclose(w, np.broadcast_to(ref / ref.sum(), w.shape), atol=1e-15)


def test_transposed_pair_takes_the_negated_images():
    kgrid = (4, 4, 4)
    tab = pair_ws_images(kgrid, FCC, _pos(9, kgrid, 4), 3)
    key = {tuple(np.mod(v, kgrid)): c for c, v in enumerate(tab["img"][:, 0])}
    for c in range(tab["img"].shape[0]):
        cm = key[tuple(np.mod(-tab["img"][c, 0], kgrid))]
        w, wm = _weights(tab, c), _weights(tab, cm)
        for t in np.flatnonzero(tab["valid"][c]):
            tm = [s for s in np.flatnonzero(tab["valid"][cm])
                  if np.array_equal(tab["img"][cm, s], -tab["img"][c, t])]
            assert len(tm) == 1 or not w[:, :, t].any()
            if tm:
                np.testing.assert_array_equal(w[:, :, t], wm[:, :, tm[0]].T)


@pytest.mark.parametrize("cg, fg, lat, dim", [((3, 3, 3), (6, 6, 6), FCC, 3),
                                              ((4, 4, 1), (6, 6, 1), HEX, 2)])
def test_embedding_keeps_every_coarse_sample_and_hermiticity(cg, fg, lat, dim):
    n = 6
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n) + cg) + 1j * rng.standard_normal((n, n) + cg)
    Wq = A + np.conj(np.swapaxes(A, 0, 1))
    tab = pair_ws_images(cg, lat, _pos(n, cg, 2), dim)
    XR = np.fft.ifftn(Wq, axes=(2, 3, 4))
    Wf = np.fft.fftn(np.asarray(pair_ws_embed(jnp.asarray(XR), tab, fg)), axes=(2, 3, 4))
    sub = tuple(slice(None, None, f // c) for c, f in zip(cg, fg))
    if all(f % c == 0 for c, f in zip(cg, fg)):
        np.testing.assert_allclose(Wf[(slice(None), slice(None)) + sub], Wq, atol=1e-12)
    np.testing.assert_allclose(Wf, np.conj(np.swapaxes(Wf, 0, 1)), atol=1e-12)
    q = np.random.default_rng(5).integers(0, 6, size=(5, 3)) / np.asarray(fg)
    n_q = np.rint(q * np.asarray(fg)).astype(int) % np.asarray(fg)
    got = np.asarray(pair_ws_fourier(jnp.asarray(XR), tab, q))
    want = np.moveaxis(Wf[:, :, n_q[:, 0], n_q[:, 1], n_q[:, 2]], -1, 0)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_refuses_a_periodic_vacuum_axis():
    with pytest.raises(ValueError):
        pair_ws_images((4, 4, 2), HEX, np.full((2, 3), 0.1), 2)
