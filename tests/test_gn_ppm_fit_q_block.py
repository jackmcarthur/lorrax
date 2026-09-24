"""The GN-PPM fit's q-chunk sizer prices the LOCAL tile, and chunking is bit-exact.

``_gn_ppm_fit_q_block`` budgets a per-device arena, so the caller must hand
it one q-slice of the device's own shard.  Handing it the global ``(mu, nu)``
slice over-chunked by the device count (VI3 12x12, mu 3200: q_block 1 at P16
and P100, i.e. 144 eager slice/fit/reshard rounds instead of 6 and 1).  The
second cell is the parity half: the fit is elementwise in q and its census
is exact counts and extrema, so every q_block gives identical bits.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import gw.minimax_screening as ms
from symmetry_maps import q_negation_index

NQ, MU, N_LOG = 6, 16, 14


def _inputs(mesh):
    rng = np.random.default_rng(7)
    a = rng.standard_normal((NQ, MU, MU)) + 1j * rng.standard_normal((NQ, MU, MU))
    herm = 0.5 * (a + np.conj(np.swapaxes(a, -1, -2)))
    Wc0 = -2.0 - 0.1 * herm
    Wprobe = 0.3 * Wc0 + 0.02j * (a - np.conj(np.swapaxes(a, -1, -2)))
    # A few lanes whose two-point ratio cannot give a positive Omega^2.
    Wprobe[0, 1, 2] = Wprobe[0, 2, 1] = 3.0 * Wc0[0, 1, 2]
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    return (jax.device_put(Wc0, sharding), jax.device_put(Wprobe, sharding))


def _fit(Wc0, Wprobe, *, ordered):
    return ms.fit_gn_ppm_from_wc_pair(
        Wc0, Wprobe, 2.0j, fallback_omega=2.0, n_mu_logical=N_LOG,
        q_neg_index=q_negation_index((NQ, 1, 1)),
        coarsen_extreme_tails=True, ordered_orientations=ordered,
        print_fn=lambda *a, **k: None)


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("requires four real devices")
    return Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


@pytest.mark.mesh(4)
def test_fit_q_block_prices_the_local_tile(monkeypatch):
    mesh = _mesh()
    Wc0, Wprobe = _inputs(mesh)
    seen = []
    real = ms._gn_ppm_fit_q_block

    def spy(nq, tile_bytes_per_q):
        seen.append((nq, tile_bytes_per_q))
        return real(nq, tile_bytes_per_q)

    monkeypatch.setattr(ms, "_gn_ppm_fit_q_block", spy)
    _fit(Wc0, Wprobe, ordered=False)
    # One q-slice of a 2x2-sharded (16, 16) c128 tile: 8 * 8 * 16 bytes.
    assert seen == [(NQ, (MU // 2) * (MU // 2) * 16)]


@pytest.mark.mesh(4)
@pytest.mark.parametrize("ordered", [False, True])
def test_fit_is_bit_identical_at_every_q_block(monkeypatch, ordered):
    mesh = _mesh()
    Wc0, Wprobe = _inputs(mesh)
    real = ms._gn_ppm_fit_q_block
    blocks = {}
    results = {}
    for label, budget in (("single_shot", 1 << 40), ("q_block_1", 1)):
        monkeypatch.setattr(ms, "_GN_PPM_FIT_ARENA_BUDGET_BYTES", budget)
        monkeypatch.setattr(
            ms, "_gn_ppm_fit_q_block",
            lambda nq, tile, _label=label: blocks.setdefault(
                _label, real(nq, tile)))
        results[label] = _fit(Wc0, Wprobe, ordered=ordered)
    # The two arms really took the two code paths.
    assert blocks == {"single_shot": NQ, "q_block_1": 1}
    a, b = results["single_shot"], results["q_block_1"]
    for name in ("omega_qmunu", "B_qmunu", "valid_qmunu", "B_odd_qmunu"):
        x, y = getattr(a, name), getattr(b, name)
        if x is None or y is None:
            assert x is None and y is None and not ordered
            continue
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(x)), np.asarray(jax.device_get(y)),
            err_msg=name)
    for name in ("n_valid", "unfulfilled_fraction", "omega_min_raw",
                 "omega_max_raw", "pair_relative_separation_min",
                 "n_tail_low", "n_tail_high", "omega_min_after",
                 "omega_max_after"):
        assert getattr(a, name) == getattr(b, name), name
