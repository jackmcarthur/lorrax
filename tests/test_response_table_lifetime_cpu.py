"""Table eviction releases both response callers; cached numerical programs stay reusable."""
import gc
import subprocess
import sys
import threading
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
    for name in ('_chi_minimax_kernel_cache', '_CHARGE_KCONV', '_PHOTON_KCONV'):
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


@pytest.mark.parametrize('side,ns,anti', [(2, 1, False), (3, 2, True)])
def test_charge_variants_share_tables_until_the_last_operator_retires(owners, side, ns, anti):
    from ffi import fft as F
    W, _ = owners
    if jax.device_count() < side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))
    with F.xla_reference():
        grid, first, *_ = fixture(mesh, ns, anti)
        rows = first.n_centroid_packed // side
        stream, held = W._charge_stream_kconv(mesh, grid, first)
        window, shared = W._charge_window_kconv(mesh, grid, first, rows)
        assert stream is window and held is shared and len(W._CHARGE_KCONV) == 1
        _, completed = W._charge_window_kconv(mesh, grid, first, rows, complete=True)
        assert completed is held and len(held) == 8
        unique = {id(a): a for a in held}
        assert W.charge_kconv_table_bytes() == sum(a.addressable_shards[0].data.nbytes
                                                 for a in unique.values())
        references = [weakref.ref(a) for a in held]
        del unique
        grid, second, *_ = fixture(mesh, ns, anti)
        W._charge_window_kconv(mesh, grid, second, rows, complete=False)
        assert W.placed_kconv_tables_live(held), 'one first-plan operator still owns its tables'
        W._charge_window_kconv(mesh, grid, second, rows, complete=True)
        assert not W.placed_kconv_tables_live(held)
        unique = {id(a): a for _, arrays in W._CHARGE_KCONV.values() for a in arrays}
        assert W.charge_kconv_table_bytes() == sum(
            a.addressable_shards[0].data.nbytes for a in unique.values())
        del stream, window, held, shared, completed
        gc.collect()
        assert all(ref() is None for ref in references)


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


def test_explicit_retired_program_reuse_does_not_reinsert_an_orphan(owners):
    from ffi import fft as F
    W, RB = owners
    if jax.device_count() < 4:
        pytest.skip('requires four virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    with F.xla_reference():
        grid, plan, *_ = fixture(mesh, 1, False)
        _, arrays = W._charge_stream_kconv(mesh, grid, plan)
        table = arrays[0]
        # A real table-bound compiled program; the independent literal
        # arithmetic keeps this lifecycle control separate from response math.
        program = W._BoundTail(jax.jit(lambda x, t: x + t), (table,))
        W._chi_minimax_kernel_cache['held_control'] = program
        argument = jnp.ones(table.shape, dtype=jnp.complex128)
        expected = np.ones(table.shape, complex) + np.asarray(table)
        first = RB._program_compiled(program, (argument,), ahead=True)
        assert RB._program_compiled(program, (argument,), ahead=True) is first
        np.testing.assert_array_equal(np.asarray(first.result()(argument)), expected)
        for _ in range(2):
            grid, other, *_ = fixture(mesh, 1, False)
            W._charge_stream_kconv(mesh, grid, other)
        assert not W.placed_kconv_tables_live(program.tail)
        assert 'held_control' not in W._chi_minimax_kernel_cache
        assert not RB._COMPILED
        references = [weakref.ref(a) for a in (table, program, first)]
        for _ in range(2):
            future = RB._program_compiled(program, (argument,), ahead=True)
            # No subsequent owner turnover or explicit cleanup is needed.
            assert not RB._COMPILED, 'retired external program became a global orphan owner'
            np.testing.assert_array_equal(np.asarray(future.result()(argument)), expected)
            references.append(weakref.ref(future))
            del future
        del arrays, table, program, first
        gc.collect()
        assert all(ref() is None for ref in references)


def test_pending_compile_retains_its_tables_after_operator_retirement(owners, monkeypatch):
    from common import jax_compile_cache as JCC
    from ffi import fft as F
    W, RB = owners
    if jax.device_count() < 4:
        pytest.skip('requires four virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    started, release = threading.Event(), threading.Event()
    original = JCC._compile_in_slot

    def paused(lowered, entry):
        started.set()
        if not release.wait(10):
            raise TimeoutError('test did not release its pending compile')
        return original(lowered, entry)

    with F.xla_reference():
        grid, plan, *_ = fixture(mesh, 1, False)
        _, arrays = W._charge_stream_kconv(mesh, grid, plan)
        table = arrays[0]
        reference = weakref.ref(table)
        program = W._BoundTail(jax.jit(lambda x, t: x+t), (table,))
        argument = jnp.ones(table.shape, dtype=jnp.complex128)
        expected = np.ones(table.shape, complex)+np.asarray(table)
        monkeypatch.setattr(JCC, '_compile_in_slot', paused)
        try:
            future = RB._program_compiled(program, (argument,), ahead=True)
            assert started.wait(5), 'actual compile worker did not start'
            for _ in range(2):
                grid, other, *_ = fixture(mesh, 1, False)
                W._charge_stream_kconv(mesh, grid, other)
            assert not W.placed_kconv_tables_live(program.tail) and not RB._COMPILED
            del arrays, table, program
            gc.collect()
            assert not future.done() and not future.cancelled() and reference() is not None
            release.set()
            executable = future.result(timeout=30)
            np.testing.assert_array_equal(np.asarray(executable(argument)), expected)
            del executable, future
            gc.collect()
            assert reference() is None
        finally:
            release.set()
