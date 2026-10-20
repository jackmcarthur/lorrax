"""Per-centroid-pair Wigner–Seitz embedding of the W densifier (CPU).

``bse_densify.pad_W_R_to_grid`` embeds each coarse R class of the tile
W_{mu nu}(R) at the images R + L T of minimal |R + L T + r_mu - r_nu|
(1/N_deg, ``common.ws_images``).  On a synthetic W(q), Hermitian at every
coarse q:

* every coarse tile is reproduced and every fine tile is Hermitian;
* relabelling centroids by lattice vectors (r_mu -> r_mu - L_mu, which for
  W_{mu nu}(q) = sum_R e^{-iq.R} W(r_mu + R, r_nu) is the gauge
  W'_{mu nu}(q) = e^{-2 pi i q.(L_mu - L_nu)} W_{mu nu}(q)) commutes with
  the interpolation exactly — the property behind symmetry covariance, since
  a point operation maps the centroid set onto itself up to such L.  Two
  control variants break it: the fftfreq box with the Nyquist split (the
  previous embedding) and one Wigner–Seitz set shared by every pair.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from bse.bse_densify import pad_W_R_to_grid  # noqa: E402
from common.ws_images import pair_ws_images  # noqa: E402

FCC = 10.26 * np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
HEX = 5.97 * np.array([[1, 0, 0], [-.5, np.sqrt(3) / 2, 0], [0, 0, 6.0]])
N = 6
CASES = [((3, 3, 3), (6, 6, 6), FCC, 3), ((4, 4, 1), (8, 8, 1), HEX, 2),
         ((4, 4, 1), (6, 6, 1), HEX, 2)]


def _setup(cg, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N) + cg) + 1j * rng.standard_normal((N, N) + cg)
    W = A + np.conj(np.swapaxes(A, 0, 1))
    r = rng.random((N, 3))
    if cg[2] == 1:
        r[:, 2] *= 0.2
    return W, r


def _pair(W, fg, lat, r, dim):
    XR = np.fft.ifftn(W, axes=(2, 3, 4), norm="ortho")
    tab = pair_ws_images(W.shape[2:], lat, r, dim)
    return np.fft.fftn(np.asarray(pad_W_R_to_grid(jnp.asarray(XR), fg, embedding=tab)),
                       axes=(2, 3, 4), norm="ortho")


def _shared(W, fg, lat, r, dim):
    """Control: every pair on the d = 0 Wigner-Seitz set (coincident positions)."""
    return _pair(W, fg, lat, np.full_like(r, 0.5), dim)


def _box(W, fg, lat=None, r=None, dim=None):
    """Control: fftfreq box with the Nyquist coefficient split +-n/2."""
    XR = np.fft.ifftn(W, axes=(2, 3, 4), norm="ortho")
    for ax, nf in zip((2, 3, 4), fg):
        nc = XR.shape[ax]
        if nf == nc:
            continue
        s = (nc + 1) // 2
        lo, hi = XR.take(range(s), ax), XR.take(range(s, nc), ax)
        if nc % 2 == 0:
            half = 0.5 * XR.take([s], ax)
            lo = np.concatenate([lo, half], ax)
            hi = np.concatenate([half, XR.take(range(s + 1, nc), ax)], ax)
        z = list(XR.shape); z[ax] = nf - lo.shape[ax] - hi.shape[ax]
        XR = np.concatenate([lo, np.zeros(z, XR.dtype), hi], ax)
    XR = XR * np.sqrt(np.prod(fg) / np.prod(W.shape[2:]))
    return np.fft.fftn(XR, axes=(2, 3, 4), norm="ortho")


def _qgrid(g):
    return np.stack(np.meshgrid(*[np.arange(n) / n for n in g], indexing="ij"), -1)


@pytest.mark.parametrize("cg, fg, lat, dim", CASES)
def test_coarse_tiles_and_hermiticity(cg, fg, lat, dim):
    W, r = _setup(cg)
    Wf = _pair(W, fg, lat, r, dim)
    q = _qgrid(fg)
    on = np.all(np.abs(q * np.array(cg) - np.round(q * np.array(cg))) < 1e-9, -1)
    for idx in zip(*np.nonzero(on)):
        ic = tuple(int(round(q[idx][a] * cg[a])) % cg[a] for a in range(3))
        np.testing.assert_allclose(Wf[(slice(None), slice(None)) + idx],
                                   W[(slice(None), slice(None)) + ic], atol=1e-12)
    np.testing.assert_allclose(Wf, np.conj(np.swapaxes(Wf, 0, 1)), atol=1e-12)


@pytest.mark.parametrize("cg, fg, lat, dim", CASES)
def test_relabelling_covariance(cg, fg, lat, dim):
    W, r = _setup(cg, seed=1)
    rng = np.random.default_rng(2)
    L = rng.integers(-1, 2, size=(N, 3))
    if cg[2] == 1:
        L[:, 2] = 0
    r2 = r - L
    qc, qf = _qgrid(cg), _qgrid(fg)
    gauge = lambda q: np.exp(-2j * np.pi * np.einsum("...x,mx->...m", q, L))
    def relabel(Wq, q):
        g = np.moveaxis(gauge(q), -1, 0)                   # (mu, kx, ky, kz)
        return g[:, None] * np.conj(g)[None, :] * Wq
    W2 = relabel(W, qc)
    scale = np.abs(W).max()
    err = {}
    for name, f in (("pair", _pair), ("shared", _shared), ("box", _box)):
        want = relabel(f(W, fg, lat, r, dim), qf)
        got = f(W2, fg, lat, r2, dim)
        err[name] = np.abs(got - want).max() / scale
    assert err["pair"] < 1e-11, err
    assert err["box"] > 0.1 and err["shared"] > 0.1, err
