"""CPU phases share physical RAM with host caches outside JAX's live set."""
import numpy as np
import pytest


@pytest.mark.parametrize('small_deck', [False, True])
def test_cpu_room_accounts_for_host_pressure_without_charging_deck_as_reserve(monkeypatch, small_deck):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common import gpu_utils
    from lxkit import memory, ordered_transient_bytes
    from lxkit.legs import HOST
    from runtime.aot_memory import compiled_new_bytes

    jax.config.update('jax_enable_x64', True)
    monkeypatch.setattr(memory, '_HOOK', dict(memory._HOOK))
    devices, processes, limit = jax.local_device_count(), 64, 256 << 20
    physical_reserve = limit * HOST.memory_reserve + ordered_transient_bytes()
    state = {'available': physical_reserve + (4 << 20)}
    reads, agreements = [], []
    def meminfo(field):
        reads.append(field)
        return ((limit if field == 'MemTotal' else state['available'])
                * processes * devices / 1e9)
    def agree(value):
        agreements.append(value)
        return value
    monkeypatch.setattr(gpu_utils, '_meminfo_gb', meminfo)
    monkeypatch.setattr(gpu_utils, '_processes_on_this_node', lambda: processes)
    monkeypatch.setattr(gpu_utils, 'minimum_process_budget_gb', agree)
    deck = (2 << 20) / 1e9 if small_deck else 0.
    gpu_utils.resolve_device_budget_gb(deck)
    mesh = Mesh(np.array(jax.devices('cpu')), ('rank',))
    prior = jax.device_put(np.ones((64, 64), complex), NamedSharding(mesh, P()))
    prior.block_until_ready()
    program = jax.jit(lambda: jnp.zeros((256, 256), jnp.complex128)).lower().compile()
    quote = compiled_new_bytes(program)
    first = memory.room(bucketed=False)
    assert abs(first - int(min(memory.run()['budget'] - memory._in_use(), 4 << 20))) <= 1
    assert first > quote
    # Only host pressure changes: the same JAX operands and executable survive.
    pressure = np.empty((devices, (15 << 18)), np.uint8)
    pressure.fill(1)
    state['available'] -= pressure.nbytes / devices
    count = len(agreements)
    second = memory.room(bucketed=False)
    assert len(agreements) == count + 1  # the existing room agreement only
    assert 0 < second < quote < first
    assert abs(second - int(state['available'] - physical_reserve)) <= 1
    with pytest.warns(RuntimeWarning, match='memory over budget'):
        memory.admit('CPU host cache beside compiled output', memory.priced(quote, 'actual compiled allocation'), second)
    np.testing.assert_array_equal(np.asarray(prior), 1)
    assert pressure.sum() == pressure.size
    assert reads.count('MemTotal') == 1 and reads.count('MemAvailable') >= 2
    # An explicit replay budget does not retain a previous run's host reader.
    memory.set_budget(1 << 30)
    assert memory._HOOK['available'] is None


def test_available_reader_is_ignored_for_accelerator_legs(monkeypatch):
    from types import SimpleNamespace
    from lxkit import memory, ordered_transient_bytes
    monkeypatch.setattr(memory, '_HOOK', dict(memory._HOOK))
    leg = SimpleNamespace(accelerator=True, memory_reserve=.18567,
                          outside_allowance=0., distributed_context=(0., ''), node_devices=4)
    monkeypatch.setattr(memory, '_leg', lambda: leg)
    def host_reader():
        raise AssertionError('a GPU room must not consume the CPU host-capacity reader')
    limit = 16_000_000_000
    budget = memory.resolve(total=limit, outside=0., available=host_reader)
    assert budget == limit * (1-leg.memory_reserve) - ordered_transient_bytes()
    assert memory._HOOK['available'] is None
    assert memory.room(bucketed=False) > 0
