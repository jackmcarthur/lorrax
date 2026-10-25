"""Table eviction releases both response callers; cached numerical programs stay reusable."""
import gc
import subprocess
import sys
import weakref
from dataclasses import replace

import runtime
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tests.test_minimax_response_rows_cpu import fixture, numpy_response

jax.config.update('jax_enable_x64', True)


@pytest.fixture
def owners(monkeypatch):
    from gw import response_bank as RB, subtile_stream as S, w_isdf as W
    # Isolate before any construction, then exercise real insertions/evictions;
    # no cache clearing is used to establish release after turnover.
    for name in ('_chi_minimax_kernel_cache', '_CHARGE_KCONV', '_PLACED_LOAD', '_PHOTON_KCONV'):
        monkeypatch.setattr(W, name, {})
    for name in ('_STREAM_KERNELS', '_STREAM_PROGRAMS', '_COMPILED'):
        shared = {}
        monkeypatch.setattr(W, name, shared, raising=False)
        monkeypatch.setattr(RB, name, shared)
        monkeypatch.setattr(S, name, shared, raising=False)
    return W, RB


def test_response_bank_import_does_not_initialize_a_backend():
    command = '''
import jax
from ffi import _services
_services.ensure_on_path()
from jax._src import xla_bridge
assert not xla_bridge.backends_are_initialized()
import gw.response_bank
assert not xla_bridge.backends_are_initialized()
'''
    run = subprocess.run([sys.executable, '-c', command], capture_output=True,
                         text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr


def _case(mesh, ns, anti, offset=0):
    grid, plan, psi, energy, valence, conduction = fixture(mesh, ns, anti)
    # Same geometry, different actual phase tables: program identity cannot
    # substitute for rebinding the authenticated plan's numerical tables.
    plan = replace(plan, k_parent_frac=plan.k_parent_frac + .037 * offset)
    put = lambda value, spec: jax.device_put(value, NamedSharding(mesh, spec))
    faces = (put(psi, P(None, None, 'x', 'y')),
             put(psi.transpose(0, 3, 1, 2), P(None, 'x', None, 'y')))
    fixed = (*faces, put(valence, P()), put(conduction, P()), put(energy, P()),
             jnp.asarray(energy[valence].max()), jnp.asarray(energy[conduction].min()))
    return grid, plan, psi, energy, valence, conduction, fixed


def _kernel(W, mesh, case):
    grid, plan, psi, *_ = case
    parents, ns, mu, nb = psi.shape
    return W._get_chi_minimax_kernel(mesh, grid, 1, complex_contour=True,
        face_shape=(parents, nb, mu, ns), k_unfold_plan=plan,
        real_times=False, sum_layout='face')


@pytest.mark.parametrize('side,ns,anti', [(2, 1, False), (3, 2, True)])
def test_minimax_reuses_owned_tables_and_releases_retired_plans(owners, side, ns, anti):
    from ffi import fft as F
    from gw.minimax_screening import MinimaxNodes
    W, _ = owners
    if jax.device_count() < side * side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:side * side]).reshape(side, side), ('x', 'y'))
    times = np.array([.1 + .07j, .4 - .09j])
    alpha = np.array([[-.3 + .2j, .12 - .03j]])
    nodes = MinimaxNodes(jnp.asarray(times), jnp.asarray(alpha[0]))
    with F.xla_reference():
        for index in range(3):
            case = _case(mesh, ns, anti, index)
            grid, plan, psi, energy, valence, conduction, fixed = case
            kernel = _kernel(W, mesh, case)
            assert _kernel(W, mesh, case) is kernel
            if index == 0:
                tables = [weakref.ref(a) for a in jax.tree.leaves(kernel.tail)]
                program = weakref.ref(kernel)
            actual = np.asarray(kernel(nodes, *fixed))
            expected = numpy_response(grid, plan, psi, energy, valence, conduction,
                                      times, alpha, False, False)[0]
            np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=3e-11)
            del kernel, case, fixed, plan
        gc.collect()
        assert program() is None, 'retired minimax program still owns its table operands'
        assert all(ref() is None for ref in tables), 'retired tables outlived both bounded owners'
        assert len(W._chi_minimax_kernel_cache) == 2


def test_placed_only_charge_tables_remain_counted_and_live(owners):
    from ffi import fft as F
    W, _ = owners
    if jax.device_count() < 4:
        pytest.skip('requires four virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    with F.xla_reference():
        grid, first, *_ = fixture(mesh, 1, False)
        _, held = W._charge_stream_kconv(mesh, grid, first)
        grid, second, *_ = fixture(mesh, 1, False)
        rows = second.n_centroid_packed // 2
        W._charge_window_kconv(mesh, grid, second, rows, complete=False)
        W._charge_window_kconv(mesh, grid, second, rows, complete=True)
        # Both kconv entries now serve second; first is still actually owned
        # by the independently bounded placement cache and must be reserved.
        assert not any(any(a is b for a in held for b in arrays)
                       for _, arrays in W._CHARGE_KCONV.values())
        assert W.placed_kconv_tables_live(held)
        unique = {id(a): a for _, arrays in W._PLACED_LOAD.values() for a in arrays}
        assert W.charge_kconv_table_bytes() == sum(
            a.addressable_shards[0].data.nbytes for a in unique.values())


def test_ordinary_turnover_retires_completed_bank_variants_and_executables(owners):
    from ffi import ffi_dial_key
    from ffi import fft as F
    W, RB = owners
    if jax.device_count() < 4:
        pytest.skip('requires four virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    with F.xla_reference():
        case = _case(mesh, 1, False)
        grid, plan, psi, _, _, _, fixed = case
        parents, ns, mu, nb = psi.shape
        kernel = RB._response_stream_kernel(mesh, grid, 1, (parents, nb, mu, ns),
            _ffi_key=ffi_dial_key(), k_unfold_plan=plan, layout='face', selected_q=(0,),
            pair_mode='direct', ordered=True)
        state = RB._STREAM_PROGRAMS[kernel]
        args = (jnp.asarray([.1 + .07j]), jnp.asarray([[[-.3 + .2j]], [[.12 - .03j]]]),
                *fixed[:2], fixed[4], fixed[2], fixed[3], jnp.stack(fixed[5:]))
        outputs, program_refs, executable_refs = [], [], []
        for layout in ('axis', 'face'):
            variant = RB._stream_variant(state, layout)
            executable = RB._program_compiled(variant, args)
            assert RB._program_compiled(variant, args) is executable
            outputs.append(np.asarray(executable(*args)))
            program_refs.append(weakref.ref(variant))
            executable_refs.append(weakref.ref(executable))
            del variant, executable
        np.testing.assert_allclose(*outputs, rtol=1e-12, atol=1e-12)
        tables = [weakref.ref(a) for a in jax.tree.leaves(kernel.tail)]
        del kernel, state, plan, case, args, fixed
        # Empty-tail programs have no placed-table lifetime to retire.
        plain = jax.jit(lambda x: x + 1)
        W._chi_minimax_kernel_cache['plain'] = plain
        for index in (1, 2):
            case = _case(mesh, 1, False, index)
            _kernel(W, mesh, case)
            del case
        gc.collect()
        assert not RB._STREAM_KERNELS and not RB._STREAM_PROGRAMS
        assert not RB._COMPILED
        assert all(ref() is None for ref in tables)
        assert all(ref() is None for ref in program_refs)
        assert all(ref() is None for ref in executable_refs)
        assert W._chi_minimax_kernel_cache['plain'] is plain
        assert RB._STREAM_KERNELS is W._STREAM_KERNELS
        assert RB._COMPILED is W._COMPILED
