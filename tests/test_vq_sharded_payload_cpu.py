"""VQ restart payloads retain mesh-owned wavefunctions and inert carriers."""
import jax
import numpy as np
import pytest
import re
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('side,ns', ((2, 1), (2, 2), (3, 1), (3, 2)))
def test_coarse_payload_never_mirrors_wavefunctions_to_host(monkeypatch, side, ns):
    from file_io import restart_bundle as rb
    from common import collectives
    if jax.device_count() < side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))
    nk, logical_bands, disk_bands, logical_mu = 3, 3, 6, 13
    carrier = -(-logical_mu // (side*side)) * (side*side)
    shape = (nk, disk_bands, ns, carrier)
    values = np.arange(np.prod(shape)).reshape(shape) / 100 + .1j
    values[:, logical_bands:] = np.nan
    values[..., logical_mu:] = np.nan
    stored = jax.device_put(values, NamedSharding(mesh, P(None, None, None, 'x')))
    energy = np.arange(nk*disk_bands).reshape(nk, disk_bands) / 10
    monkeypatch.setattr(rb, 'read_metadata', lambda _:
        dict(logical_band_count=logical_bands, centroid_count=logical_mu,
             grid=np.array([3, 1, 1]), energies=energy,
             band_window=np.array([7, 8, 8, 9, 10]), screened_ready=True))
    requested_bands = []

    def read_wavefunctions(filename, input_file, selected_mesh, *, bands=None):
        assert selected_mesh is mesh
        requested_bands.append(None if bands is None else tuple(bands))
        return stored if bands is None else stored[:, :len(bands)]

    reads = []
    bank = jax.device_put(np.zeros((nk, carrier, carrier), complex),
                         NamedSharding(mesh, P(None, 'x', 'y')))

    def interaction(filename, kind, selected_mesh):
        assert selected_mesh is mesh
        reads.append(kind)
        return bank

    monkeypatch.setattr(rb, 'read_wavefunctions', read_wavefunctions)
    monkeypatch.setattr(rb, 'read_interaction', interaction)
    original_asarray, original_get, original_jit = np.asarray, jax.device_get, jax.jit
    programs = []

    def observe_jit(function, **kwargs):
        program = original_jit(function, **kwargs)
        if function.__name__ == 'canonical':
            programs.append(program)
        return program

    monkeypatch.setattr(jax, 'jit', observe_jit)

    def no_array_conversion(value, *a, **kw):
        if isinstance(value, jax.Array) and value.ndim >= 3:
            raise AssertionError('production payload converted a large device array to host')
        return original_asarray(value, *a, **kw)

    def no_device_get(value):
        if any(isinstance(v, jax.Array) and v.ndim >= 3 for v in jax.tree.leaves(value)):
            raise AssertionError('production payload device_get of a large array')
        return original_get(value)

    with monkeypatch.context() as guard:
        guard.setattr(np, 'asarray', no_array_conversion)
        guard.setattr(jax, 'device_get', no_device_get)
        guard.setattr(collectives, 'gather_to_host', lambda *_a, **_kw:
                      (_ for _ in ()).throw(AssertionError('large payload gather')))
        general = rb.read_coarse_interactions('restart', 'deck', mesh)
        general['psi'].block_until_ready()
        prior_reads = tuple(reads)
        pure = rb.read_coarse_interactions('restart', 'deck', mesh,
                                           include_interactions=False)
        pure['psi'].block_until_ready()
        assert tuple(reads) == prior_reads and 'Vqmunu' not in pure and 'W0' not in pure
    assert reads == ['bare', 'screened']
    assert requested_bands == [(0, 1, 2), (0, 1, 2)]
    for payload in (pure, general):
        psi = payload['psi']
        assert isinstance(psi, jax.Array)
        assert psi.shape == (nk, logical_bands, ns, carrier)
        assert psi.sharding.spec == P(None, None, None, ('x', 'y'))
        assert all(s.data.shape[-1] == carrier//(side*side) for s in psi.addressable_shards)
        assert payload['n_mu_logical'] == logical_mu
        got = np.asarray(psi)  # explicit tiny test oracle, outside the production guard
        np.testing.assert_array_equal(got[..., :logical_mu],
                                      values[:, :logical_bands, :, :logical_mu])
        assert np.isfinite(got).all() and not np.any(got[..., logical_mu:])
        np.testing.assert_array_equal(payload['enk'], energy[:, :logical_bands])
    assert general['Vqmunu'] is bank and general['W0'] is bank
    for program in programs:
        hlo = program.lower(stored[:, :logical_bands]).compile().as_text()
        assert 'all-gather(' not in hlo
        assert re.search(r'c128\[[^\]]*,13\]', hlo) is None
