"""Physical BSE seeds keep their draws and exhaustion count without host fields."""
import ast
from pathlib import Path
import re

import runtime
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from bse.bse_kpm import make_bse_random_vector
from bse.bse_window import pad_zone_mask, restrict_to_physical_transitions
from distrib_la.subspace import DistributedSubspacePlan
from solvers.lanczos import (_supported_start_block, block_lanczos_eig_jit,
                             block_lanczos_eig_jit_converged)

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).parents[1]


def _seed_program(mesh, logical, carrier, bs=2):
    sh = NamedSharding(mesh, P(None, 'x', 'y', None))
    # The real distributed layout owner; no QR/provider call is needed to
    # isolate generation, support embedding, and their compiled storage.
    plan = DistributedSubspacePlan(local=None, vector_sharding=sh)
    n = int(np.prod(carrier))

    def seed():
        q, count = _supported_start_block(
            n, bs, 42, None, vector_shape=carrier,
            support_shape=logical, subspace_plan=plan)
        return q.T.reshape(bs, *carrier), jnp.int64(count)

    return jax.jit(seed, out_shardings=(sh, NamedSharding(mesh, P())))


def _local_hlo_bound(compiled, local_entries):
    text = compiled.as_text()
    # Optimized computations (not HloModule global-layout annotations).
    shapes = re.findall(r'^\s*(?:ROOT\s+)?%[^\n=]+ = (?:c128|f64|u64|u32)\[([0-9,]+)\]',
                        text, re.MULTILINE)
    sizes = [int(np.prod(tuple(map(int, shape.split(','))))) for shape in shapes]
    assert sizes and max(sizes) <= max(local_entries, 8)
    assert not re.search(r'= .*\ball-gather(?:-start)?\(', text)
    return text


@pytest.mark.parametrize('side', (2, 3))
def test_rectangular_seeds_match_old_physical_draw_and_stay_local(side, monkeypatch):
    if jax.device_count() < side*side:
        pytest.skip('requires four or nine virtual CPU devices')
    mesh = Mesh(np.array(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))
    logical, bs = (3, 5, 7), 2
    carrier = tuple((-(-n//side)*side if axis < 2 else n)
                    for axis, n in enumerate(logical))
    program = _seed_program(mesh, logical, carrier, bs)
    old_mask = np.zeros(carrier, dtype=bool)
    old_mask[:logical[0], :logical[1], :] = True
    old, count = _supported_start_block(int(np.prod(carrier)), bs, 42, old_mask)
    expected = np.asarray(old.T.reshape(bs, *carrier))

    def no_host_indices(*args, **kwargs):
        raise AssertionError('rectangular support must not create host transition indices')

    monkeypatch.setattr(np, 'flatnonzero', no_host_indices)
    compiled = program.lower().compile()
    actual, actual_count = compiled()
    np.testing.assert_array_equal(actual, expected)
    assert int(actual_count) == count == int(np.prod(logical))
    _local_hlo_bound(compiled, bs*int(np.prod(carrier))//(side*side))


def test_large_abstract_support_has_only_local_rng_and_no_host_support(monkeypatch):
    if jax.device_count() < 9:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.array(jax.devices()[:9]).reshape(3, 3), ('x', 'y'))
    logical, carrier = (257, 263, 73), (258, 264, 73)
    def no_host_indices(*args, **kwargs):
        raise AssertionError('abstract support must not materialize a host index list')
    monkeypatch.setattr(np, 'flatnonzero', no_host_indices)
    compiled = _seed_program(mesh, logical, carrier).lower().compile()
    local = 2*int(np.prod(carrier))//9
    _local_hlo_bound(compiled, local)
    assert compiled.memory_analysis().output_size_in_bytes <= 16*local + 32


def test_lanczos_rectangular_support_preserves_physical_spectrum_and_exhaustion():
    logical, carrier, bs = (2, 3, 2), (3, 4, 2), 2
    n, nphys = int(np.prod(carrier)), int(np.prod(logical))
    mask = np.zeros(carrier, dtype=bool)
    mask[:2, :3, :] = True
    diagonal = np.full(carrier, 1000.)
    diagonal[mask] = np.arange(1, nphys+1)
    matvec = lambda x: x*jnp.asarray(diagonal.reshape(-1))
    common = dict(n=n, n_eig=3, block_size=bs, max_iter=99,
                  subspace_plan=False, vector_shape=carrier)
    fixed = jax.jit(lambda: block_lanczos_eig_jit(
        matvec, **common, support_shape=logical))
    ev, vectors = fixed()
    np.testing.assert_allclose(ev, (1., 2., 3.), atol=3e-12)
    assert np.max(np.abs(np.asarray(vectors)[:, ~mask.reshape(-1)])) == 0.
    old = jax.jit(lambda: block_lanczos_eig_jit(matvec, **common, support=mask))()
    np.testing.assert_allclose(ev, old[0], atol=3e-12)
    converged = jax.jit(lambda: block_lanczos_eig_jit_converged(
        matvec, **common, support_shape=logical, check_every=1, min_iter=99))()
    assert int(converged[2]) == nphys//bs
    np.testing.assert_allclose(converged[0], ev, atol=3e-12)
    for shape in ((0, 3, 2), (2, 5, 2), (2.0, 3, 2), (2, 3)):
        with pytest.raises(ValueError, match='support_shape'):
            _supported_start_block(n, bs, 42, None, vector_shape=carrier, support_shape=shape)
    with pytest.raises(ValueError, match='not both'):
        _supported_start_block(n, bs, 42, mask, vector_shape=carrier, support_shape=logical)


def test_kpm_and_spectral_bound_seeds_preserve_samples_without_host_masks():
    nc, nv, ncp, nvp, nk = 3, 5, 4, 6, 7
    key = jax.random.PRNGKey(19)
    mask = np.zeros((1, ncp, nvp, nk))
    mask[:, :nc, :nv, :] = 1
    for dtype in (jnp.float32, jnp.float64):
        data = dict(nkx=nk, nky=1, nkz=1, n_cond=nc, n_val=nv,
                    n_cond_pad=ncp, n_val_pad=nvp, eps_c=jnp.zeros((nk, ncp), dtype))
        for tda in (True, False):
            fn = make_bse_random_vector(data, tda)
            keys = (key,) if tda else tuple(jax.random.split(key))
            x = [2*jax.random.bernoulli(k, shape=(1, ncp, nvp, nk)).astype(dtype)-1 for k in keys]
            expected = (x[0] if tda else jnp.stack(x))*mask
            np.testing.assert_array_equal(jax.jit(fn)(key), expected)
            assert not any(isinstance(c.cell_contents, (np.ndarray, jax.Array))
                           for c in (fn.__closure__ or ()))
    # Compile the actual nested FEAST seed, rather than duplicating its math.
    tree = ast.parse((ROOT/'src/bse/bse_feast.py').read_text())
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'estimate_spectral_bounds_sharded')
    node = next(n for n in outer.body if isinstance(n, ast.FunctionDef)
                and n.name == '_make_random_vector')
    from types import SimpleNamespace
    namespace = dict(jax=jax, jnp=jnp, n_cond=nc, n_val=nv, n_cond_pad=ncp,
                     n_val_pad=nvp, nk=nk, sh=SimpleNamespace(X=None),
                     restrict_to_physical_transitions=restrict_to_physical_transitions)
    # An absent constraint is the one-device form of the same physical seed.
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
    namespace['sh'].X = NamedSharding(mesh, P(None, 'x', 'y', None))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual FEAST seed>', 'exec'), namespace)
    a, b = jax.random.split(key)
    old = (jax.random.normal(a, mask.shape, dtype=jnp.float64)
           + 1j*jax.random.normal(b, mask.shape, dtype=jnp.float64))*mask
    old = old/jnp.linalg.norm(old)
    np.testing.assert_allclose(namespace['_make_random_vector'](key), old, rtol=3e-15, atol=3e-15)
    for name in ('bse_kpm.py', 'bse_feast.py', 'bse_lanczos.py', 'exciton_bands.py'):
        assert 'pad_zone_mask_np' not in (ROOT/'src/bse'/name).read_text()


def test_physical_masks_preserve_empty_support_and_refuse_noninteger_counts():
    x = jnp.ones((2, 4, 6, 7), jnp.complex128)
    data = dict(nkx=7, nky=1, nkz=1, n_cond_pad=4, n_val_pad=6,
                eps_c=jnp.zeros((7, 4), jnp.float64))
    for nc, nv in ((0, 5), (3, 0), (0, 0)):
        assert not np.any(restrict_to_physical_transitions(x, nc, nv))
        mask = pad_zone_mask(dict(data, n_cond=nc, n_val=nv))
        assert mask.shape == (1, 4, 6, 7) and mask.dtype == jnp.float64
        assert not np.any(mask)
    for nc, nv in ((-1, 5), (3, -1), (5, 5), (3, 7), (2.5, 5), (3, 4.5), (True, 5)):
        with pytest.raises(ValueError, match='transition counts'):
            restrict_to_physical_transitions(x, nc, nv)
