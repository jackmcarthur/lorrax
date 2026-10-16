"""The R zero-pad of the W densifier on even coarse axes (CPU).

``bse_densify._zeropad_R_axis`` splits the coarse Nyquist coefficient, weight
½ at both R = ±n_c/2.  On a synthetic W(k), Hermitian at every coarse k, the
padded interpolant must reproduce every coarse tile and be Hermitian at every
fine k; the single-sided pad it replaced (weight 1 at −n_c/2) reproduces the
tiles but is not Hermitian off the grid.  Odd axes keep the plain embedding.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from bse.bse_densify import _zeropad_R_axis, pad_W_R_to_grid  # noqa: E402

CG, FG, N = (4, 6, 3), (8, 12, 6), 5          # even, even, odd coarse axes


def _hermitian_wq():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((N, N) + CG) + 1j * rng.standard_normal((N, N) + CG)
    return A + np.conj(np.swapaxes(A, 0, 1))


def _single_sided(x, axis, n_fine):
    """The pre-split pad: Nyquist rep −n_c/2 only, weight 1."""
    n_c = x.shape[axis]
    if n_fine == n_c:
        return x
    s = (n_c + 1) // 2
    lo, hi = np.split(x, [s], axis=axis)
    shape = list(x.shape)
    shape[axis] = n_fine - n_c
    return np.concatenate([lo, np.zeros(shape, x.dtype), hi], axis=axis)


def _fine(W, pad):
    XR = np.fft.ifftn(W, axes=(2, 3, 4), norm="ortho")
    return np.fft.fftn(pad(XR), axes=(2, 3, 4), norm="ortho")


def _split_pad(XR):
    return np.asarray(pad_W_R_to_grid(jnp.asarray(XR), FG))


def _old_pad(XR):
    for ax, nf in zip((2, 3, 4), FG):
        XR = _single_sided(XR, ax, nf)
    return XR * np.sqrt(np.prod(FG) / np.prod(CG))


def _herm_err(Wf):
    d = Wf - np.conj(np.swapaxes(Wf, 0, 1))
    return np.linalg.norm(d, axis=(0, 1)) / np.linalg.norm(Wf, axis=(0, 1))


@pytest.mark.parametrize("pad, hermitian", [(_split_pad, True), (_old_pad, False)])
def test_even_axes_keep_coarse_tiles_and_hermiticity(pad, hermitian):
    W = _hermitian_wq()
    Wf = _fine(W, pad)
    on = Wf[:, :, ::FG[0] // CG[0], ::FG[1] // CG[1], ::FG[2] // CG[2]]
    assert np.max(np.abs(on - W)) <= 1e-12 * np.max(np.abs(W))
    err = _herm_err(Wf)
    if hermitian:
        assert err.max() <= 1e-12
    else:                                       # red twin: off-grid tiles break
        assert err.max() > 1e-2


def test_odd_axis_is_the_plain_embedding():
    x = np.random.default_rng(5).standard_normal((2, 3, 7)) + 0j
    got = np.asarray(_zeropad_R_axis(jnp.asarray(x), 2, 12))
    assert np.array_equal(got, _single_sided(x, 2, 12))
