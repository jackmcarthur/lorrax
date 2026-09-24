"""Band extrapolation on a padded Sigma band carrier (logical 3 bands in a carrier of 4)."""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gw.ppm_pipeline import _extrapolated_point  # noqa: E402
from runtime.padding import padded_axis, strip_axis  # noqa: E402


def test_per_state_weights_act_on_the_padded_cube_like_on_the_logical_one():
    rng = np.random.default_rng(5)
    nbr, nw, nk, nb, carrier = 3, 2, 4, 3, 4
    logical = rng.standard_normal((nbr, nw, nk, nb, nb)) + 1j * rng.standard_normal((nbr, nw, nk, nb, nb))
    padded = np.zeros((nbr, nw, nk, carrier, carrier), np.complex128)
    padded[..., :nb, :nb] = logical
    w = rng.standard_normal((nbr, nk, nb))

    want = np.asarray(_extrapolated_point(jnp.asarray(logical), w))
    got = np.asarray(_extrapolated_point(jnp.asarray(padded), w))
    np.testing.assert_array_equal(got[..., :nb, :nb], want)
    assert not np.any(got[..., nb:, :]) and not np.any(got[..., :, nb:])


def test_strip_axis_returns_the_logical_band_diagonal():
    tag = padded_axis(3, 4, name="Sigma band window")
    assert (tag.logical, tag.carrier) == (3, 4)
    diag = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
    np.testing.assert_array_equal(np.asarray(strip_axis(diag, tag, axis=-1)), diag[..., :3])
