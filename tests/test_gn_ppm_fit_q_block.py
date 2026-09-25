"""The GN-PPM fit's q block comes from the device pool, and chunking is bit-exact.

The sizer prices one q from the kernel compiled at q = 1 on the inputs' own
sharding (the LOCAL tile, whatever the kernel's layout costs; a constant
multiple of the tile read 6x LOW on the ordered kernel, CrI3 16x16 P64), and
refuses by name when one q does not fit.  The parity half: the fit is
elementwise in q and its census is exact counts and extrema, so every
q_block gives identical bits.
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
    real = ms._gn_ppm_fit_bytes_per_q
    monkeypatch.setattr(ms, "_gn_ppm_fit_bytes_per_q",
                        lambda *a: seen.append(real(*a)) or seen[-1])
    _fit(Wc0, Wprobe, ordered=False)
    block, out = seen[0]
    local = (MU // 2) * (MU // 2) * 16          # one q of a 2x2-sharded c128 tile
    # Two input slices and the outputs at least; a few local tiles at most --
    # never the global (mu, nu) slice times the device count.
    assert 2 * local <= block <= 16 * local, (block, local)
    assert 0 < out < block


@pytest.mark.mesh(4)
def test_fit_refuses_when_one_q_does_not_fit(monkeypatch):
    mesh = _mesh()
    Wc0, Wprobe = _inputs(mesh)
    monkeypatch.setattr(ms, "_gn_ppm_fit_free_bytes", lambda: 1)
    with pytest.raises(ValueError, match="GATE gn_ppm_fit_capacity"):
        _fit(Wc0, Wprobe, ordered=False)


@pytest.mark.mesh(4)
@pytest.mark.parametrize("ordered", [False, True])
def test_fit_is_bit_identical_at_every_q_block(monkeypatch, ordered):
    mesh = _mesh()
    Wc0, Wprobe = _inputs(mesh)
    real = ms._gn_ppm_fit_q_block
    blocks = {}
    results = {}
    for label, force in (("one_block", None), ("q_block_1", 1)):
        monkeypatch.setattr(
            ms, "_gn_ppm_fit_q_block",
            lambda nq, *a, _label=label, _force=force: blocks.setdefault(
                _label, real(nq, *a) if _force is None else _force))
        results[label] = _fit(Wc0, Wprobe, ordered=ordered)
    # The two arms really took one block and NQ blocks.
    assert blocks == {"one_block": NQ, "q_block_1": 1}
    a, b = results["one_block"], results["q_block_1"]
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
