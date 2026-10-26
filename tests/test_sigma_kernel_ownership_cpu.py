"""Resident tau programs inherit identity and diagnostics from their spatial owner."""
import gc
import weakref

import numpy as np
import pytest


def test_tau_identity_price_and_custom_lifetime_with_literal_pole_sum(monkeypatch):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw import ppm_tau_kernel as owner
    jax.config.update('jax_enable_x64', True)
    if len(jax.devices('cpu')) < 4:
        pytest.skip('requires four virtual CPU devices')
    mesh = Mesh(np.array(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y'))
    cache = {}
    monkeypatch.setattr(owner, '_sigma_shared_tau_kernel_cache', cache)

    def spatial():
        fn = jax.jit(lambda *args: args[8])  # expose the actual W(tau) synthesis
        fn.price = dict(scratch=0, passes=1, new=64)
        return fn

    current = [spatial()]
    # Resolve only the spatial factory in this owner test; tau synthesis,
    # program compilation, metadata and all runtime fields remain real.
    monkeypatch.setattr(owner, '_get_sigma_kij_kernel', lambda **kw: current[0])
    options = dict(mesh_xy=mesh, kgrid=(1, 1, 1), face_shape=(1, 4, 4, 1))
    tau = owner.get_shared_sigma_tau_kernel(**options)
    assert owner.get_shared_sigma_tau_kernel(**options) is tau
    assert owner.sigma_pass_price(tau) is tau.price is current[0].price
    current[0] = spatial()  # a different resolved geometry/provider program
    assert owner.get_shared_sigma_tau_kernel(**options) is not tau

    custom = spatial()
    custom_tau = owner.get_shared_sigma_tau_kernel(**options, _sigma_kij=custom)
    assert owner.get_shared_sigma_tau_kernel(**options, _sigma_kij=custom) is not custom_tau
    assert owner.sigma_pass_price(custom_tau) is custom_tau.price is custom.price
    assert len(cache) == 2  # only the two normal resolved programs are held
    residue = np.arange(16).reshape(1, 1, 4, 4)/20 + .1j
    omega = np.full_like(residue, 1.3-.07j)
    sharding = NamedSharding(mesh, P(None, None, 'x', 'y'))
    bounds = jnp.asarray([[-np.inf, np.inf, -np.inf, -np.inf, np.inf, np.inf]])
    t, reference = .23+.02j, .17
    value = custom_tau(None, None, None, None, None, None,
        jax.device_put(residue, sharding), jax.device_put(omega, sharding),
        jnp.asarray([0], jnp.int32), bounds, jnp.asarray([False]),
        0., reference, None, t)
    np.testing.assert_allclose(np.asarray(value),
        residue[0]*np.exp(-1j*(omega[0]-reference)*t), rtol=2e-14, atol=2e-14)
    ref = weakref.ref(custom_tau)
    custom_ref = weakref.ref(custom)
    del custom_tau, custom
    gc.collect()
    assert ref() is None and custom_ref() is None
