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


# ---------------------------------------------------------------------------
# The coarse (semicore) read class (gw.band_partition.semicore_floor)
# ---------------------------------------------------------------------------

def _coarse(e, **kw):
    from gw.band_partition import semicore_floor
    args = dict(n_below_k=3, nval=2, mu_ev=0.0, clip_ev=10.0)
    args.update(kw)
    return semicore_floor(np.asarray(e, float), **args)


def test_coarse_class_follows_nval_and_omega_min_only_lowers_the_floor():
    row = [-30.0, -3.0, -1.0, 1.0, 2.0, 4.0, 4.1, 4.2]
    e = np.array([row, [x + 0.05 for x in row]])
    # Every state below the lowest requested valence band (band 1 at nval 2).
    c = _coarse(e)
    assert c.coarse_floor_ev == -3.0 and c.n_coarse == 2
    assert _coarse(e, nval=3).n_coarse == 0              # nval covers every occupied band
    assert _coarse(e, nval=1).n_coarse == 4              # bands 0 and 1 are coarse
    assert _coarse(e, nval=1, omega_min_rel_ev=-5.0).n_coarse == 2
    assert _coarse(e, nval=1, omega_min_rel_ev=-0.5).n_coarse == 4


def test_number_bands_protected_splits_semicore_by_gap():
    row = [-30.0, -3.0, -1.0, 1.0, 2.0, 4.0, 4.1, 4.2]
    e = np.array([row, [x + 0.05 for x in row]])
    c = _coarse(e, n_protected=5, semicore_gap_ev=4.0)
    assert c.n_coarse == 2                               # band 0, below the 27 eV gap
    assert _coarse(e, n_protected=5, semicore_gap_ev=30.0).n_coarse == 0   # no such gap


def test_the_semicore_floor_sets_no_band_count():
    """Owner 2026-09-29: b3 counts bands as on main; the coarse class is a
    Sigma_c(omega) read class and returns no band index."""
    import inspect
    from gw import band_partition
    assert not hasattr(band_partition, "qp_band_cut")
    assert set(band_partition.CoarseClass._fields) == {"coarse_floor_ev", "n_coarse"}
    assert "b3" not in inspect.signature(band_partition.semicore_floor).parameters


def test_semicore_dft_pin_keeps_mixing_and_the_dft_block():
    """sc_semicore = dft: the coarse labels' block is diag(E_DFT) in the DFT
    basis; protected-semicore and protected-protected elements are kept."""
    from gw.gw_config import SCConfig
    from gw.sc_iteration import _pin_semicore_block_to_dft

    H = _random_hermitian(2, 5, seed=3)
    pin = np.zeros((2, 5), bool)
    pin[0, :2] = True
    pin[1, :1] = True
    e = np.array([[-3.0, -2.9, 0, 0, 0], [-3.1, 0, 0, 0, 0]])
    out = np.asarray(_pin_semicore_block_to_dft(H, e, pin))
    ref = np.array(H)
    ref[0, :2, :2] = np.diag(e[0, :2])
    ref[1, 0, 0] = e[1, 0]
    np.testing.assert_array_equal(out, ref)
    assert SCConfig(max_iter=1, tol_ev=1e-4, accelerator="anderson", history_depth=1,
                    mixing=1.0, dump_dir=None).semicore == "qp"
    try:
        SCConfig(max_iter=1, tol_ev=1e-4, accelerator="anderson", history_depth=1,
                 mixing=1.0, dump_dir=None, semicore="frozen")
    except ValueError as exc:
        assert "sc_semicore" in str(exc)
    else:
        raise AssertionError("sc_semicore = frozen was accepted")
