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
4. **Support reporting** — names protected states outside the requested
   window without claiming a quadrature coverage failure.
"""
from __future__ import annotations

import io
import contextlib

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from common.units import RYD_TO_EV
from gw.band_partition import (
    BandPartition, apply_band_partition, build_omega_band_partition)


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


def test_warn_on_protected_out_of_grid():
    # band 1 is protected but out of range — must warn.
    part = BandPartition(
        protected_mask=jnp.asarray([True, True, False]),
        in_range_mask=jnp.asarray([True, False, True]),
    )
    buf = io.StringIO()
    part.warn_if_protected_outside_grid(print_fn=lambda *a, **k: print(*a, file=buf, **k))
    msg = buf.getvalue()
    assert "1 protected (k,state) members outside" in msg
    assert "quadrature support must cover them" in msg


def test_no_warning_when_all_protected_in_range():
    part = BandPartition(
        protected_mask=jnp.asarray([True, True, False]),
        in_range_mask=jnp.asarray([True, True, False]),
    )
    buf = io.StringIO()
    part.warn_if_protected_outside_grid(print_fn=lambda *a, **k: print(*a, file=buf, **k))
    assert buf.getvalue() == ""


def test_reanchored_partition_hysteresis_breaks_edge_band_two_cycle():
    """The audit's 1.061/0.990 eV edge oscillator stays structurally fixed."""
    fixed_h_ev = np.asarray([[[0.50, 0.20], [0.20, 0.99]]])
    scissor_ev = np.asarray([[0.50, 0.99]])
    spectrum_ev = np.asarray([[0.50, 0.99]])
    previous = None
    masks, uppers, offdiagonals = [], [], []
    for _ in range(6):
        partition = build_omega_band_partition(
            spectrum_ev / RYD_TO_EV, spectrum_ev / RYD_TO_EV,
            band_offset=0, omega_min_abs_ev=-1.0,
            omega_max_abs_ev=1.0, previous_partition=previous,
            hysteresis_margin_ev=0.125, print_fn=lambda *_args: None)
        h_next = np.asarray(apply_band_partition(
            jnp.asarray(fixed_h_ev / RYD_TO_EV),
            protected_mask=partition.protected_mask,
            in_range_mask=partition.in_range_mask,
            scissor_E_qp_kn=jnp.asarray(scissor_ev / RYD_TO_EV)))
        spectrum_ev = np.linalg.eigvalsh(h_next) * RYD_TO_EV
        masks.append(np.asarray(partition.protected_mask, dtype=int).tolist())
        uppers.append(float(spectrum_ev[0, 1]))
        offdiagonals.append(float(h_next[0, 0, 1] * RYD_TO_EV))
        previous = partition

    assert masks == [[[1, 1]]] * 6
    np.testing.assert_allclose(uppers, [1.061267292017] * 6, atol=1e-12)
    np.testing.assert_allclose(offdiagonals, [0.2] * 6, atol=1e-14)

    # Hysteresis is a deadband, not a frozen class: 0.20 eV beyond the edge
    # exceeds the run-derived 0.125 eV margin and genuinely loses protection.
    drifted = build_omega_band_partition(
        np.asarray([[0.50, 1.20]]) / RYD_TO_EV,
        np.asarray([[0.50, 1.20]]) / RYD_TO_EV,
        band_offset=0, omega_min_abs_ev=-1.0, omega_max_abs_ev=1.0,
        previous_partition=previous, hysteresis_margin_ev=0.125,
        print_fn=lambda *_args: None)
    np.testing.assert_array_equal(drifted.protected_mask, [[True, True]])
    assert drifted.changed
    dropped = build_omega_band_partition(
        np.asarray([[0.50, 1.20]]) / RYD_TO_EV,
        np.asarray([[0.50, 1.20]]) / RYD_TO_EV,
        band_offset=0, omega_min_abs_ev=-1.0, omega_max_abs_ev=1.0,
        previous_partition=drifted, hysteresis_margin_ev=0.125,
        print_fn=lambda *_args: None)
    np.testing.assert_array_equal(dropped.protected_mask, [[True, False]])
