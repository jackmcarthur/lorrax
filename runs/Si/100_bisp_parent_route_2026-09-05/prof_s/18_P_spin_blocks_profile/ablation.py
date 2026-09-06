"""Apply the existing dense four-spin action with elementwise contractions."""
from gw import gw_jax
import jax.numpy as jnp
import numpy as np
from symmetry_maps import maps

original = maps._rotate_open_spin_centroid_operator


def rotate(spatial, spin):
    ns = int(spatial.shape[1])
    if ns != 4:
        return original(spatial, spin)
    assert not np.any(np.asarray(spin)[:, :2, 2:]) and not np.any(np.asarray(spin)[:, 2:, :2])
    U = jnp.asarray(spin)
    if int(spatial.shape[3]) != ns or tuple(U.shape[1:]) != (ns, ns):
        raise ValueError('Spatial spin axes and action disagree.')
    left = jnp.stack([sum(U[:, a, c, None, None, None] * spatial[:, c]
                         for c in range(2*(a//2), 2*(a//2)+2)) for a in range(ns)], axis=1)
    return jnp.stack([sum(left[:, :, :, d, :] * jnp.conj(U[:, b, d, None, None, None])
                          for d in range(2*(b//2), 2*(b//2)+2)) for b in range(ns)], axis=3)


maps._rotate_open_spin_centroid_operator = rotate
import profile_driver
