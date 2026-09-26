"""Unit tests for the QSGW band-partition mask primitive.

Synthetic ``H_full`` constructions exercise:

1. **Identity case** — ``BandPartition.all_protected`` returns the input
   unchanged (no off-diagonal zeroing, no scissor override).
2. **Off-diagonal masking** — protected×non-protected and non-protected
   ×non-protected off-diagonals are zeroed; protected×protected
   off-diagonals are preserved.
3. **Scissor override on out-of-range diagonals** — non-protected
   bands flagged out-of-range take the supplied ``scissor_E_qp_kn``
   diagonal; in-range bands keep ``H_full``'s diagonal.
"""
from __future__ import annotations


import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from gw.band_partition import BandPartition, apply_band_partition


def _random_hermitian(nk: int, nb: int, seed: int = 0) -> jax.Array:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((nk, nb, nb)) + 1j * rng.standard_normal((nk, nb, nb))
    H = 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))
    return jnp.asarray(H, dtype=jnp.complex128)


def test_all_protected_is_identity():
    nk, nb = 3, 5
    H = _random_hermitian(nk, nb)
    part = BandPartition.all_protected(nb)
    out = apply_band_partition(
        H,
        protected_mask=part.protected_mask,
        in_range_mask=part.in_range_mask,
        scissor_E_qp_kn=jnp.zeros((nk, nb), dtype=H.dtype),
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(H))


def test_offdiag_masking_protected_block_only():
    nk, nb = 2, 6
    H = _random_hermitian(nk, nb, seed=1)
    # Protect bands 1, 2, 4 only.
    protected = jnp.asarray([False, True, True, False, True, False])
    in_range = jnp.ones(nb, dtype=bool)         # nothing scissored
    out = np.asarray(apply_band_partition(
        H,
        protected_mask=protected, in_range_mask=in_range,
        scissor_E_qp_kn=jnp.zeros((nk, nb), dtype=H.dtype),
    ))
    H_np = np.asarray(H)
    p = np.asarray(protected)
    for k in range(nk):
        for m in range(nb):
            for n in range(nb):
                if m == n:
                    np.testing.assert_allclose(out[k, m, n], H_np[k, m, n])
                elif p[m] and p[n]:
                    np.testing.assert_allclose(out[k, m, n], H_np[k, m, n])
                else:
                    assert abs(out[k, m, n]) < 1e-14, (
                        f"off-diag at ({k},{m},{n}) not zero: {out[k, m, n]}")


def test_only_unprotected_outofrange_diagonal_takes_scissor():
    nk, nb = 2, 4
    H = _random_hermitian(nk, nb, seed=2)
    protected = jnp.asarray([True, False, False, True])
    # Bands 0, 2 in range; bands 1, 3 out of range. Protected band 3
    # keeps its full diagonal, so only band 1 takes the scissor.
    in_range = jnp.asarray([True, False, True, False])
    scissor = jnp.asarray(
        [[10.0 + 0j, 11.0, 12.0, 13.0], [20.0 + 0j, 21.0, 22.0, 23.0]],
        dtype=H.dtype,
    )
    out = np.asarray(apply_band_partition(
        H,
        protected_mask=protected, in_range_mask=in_range,
        scissor_E_qp_kn=scissor,
    ))
    H_np = np.asarray(H)
    for k in range(nk):
        for n in range(nb):
            expected = (H_np[k, n, n] if bool(protected[n] | in_range[n])
                        else complex(scissor[k, n]))
            np.testing.assert_allclose(
                out[k, n, n], expected, atol=1e-14,
                err_msg=f"diagonal at ({k},{n}) wrong: got {out[k, n, n]}, expected {expected}")
