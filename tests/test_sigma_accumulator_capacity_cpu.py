"""The retained Sigma output is quoted before its first allocation."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("side", [2, 3])
@pytest.mark.parametrize("grid", [False, True])
def test_outputs_are_admitted_before_dispatch_beside_prior_accumulators(monkeypatch, side, grid):
    from gw import ppm_accumulators as owner
    from gw import tangential_execution
    from lxkit import memory
    from runtime.aot_memory import compiled_new_bytes

    monkeypatch.setattr(memory, "_RUN", dict(memory.run()))
    monkeypatch.setattr(memory, "_WARNED", {})
    if len(jax.devices("cpu")) < side * side:
        pytest.skip("requires nine virtual CPU devices")
    mesh = Mesh(np.array(jax.devices("cpu")[:side * side]).reshape(side, side), ("x", "y"))
    sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    nk, nb, slots, nf = 3, 12, 3, 7
    shapes = ((nf, nk, nb, nb),) if grid else ((2, nk, nb, nb), (slots, nk, nb, side))
    payload = sum(np.prod(sharding.shard_shape(shape)) * 16 for shape in shapes)
    # A real live allocation must survive both admissions; no allocator override.
    prior = jax.device_put(np.ones((132, 132)), NamedSharding(mesh, P("x", "y")))
    prior.block_until_ready()
    original_program = owner._zeros_program
    program = original_program(sharding, shapes)
    if hasattr(program, "lower"):
        program = program.lower().compile()
    expected = compiled_new_bytes(program)
    assert expected >= payload
    assert not any(target in program.as_text() for target in ("custom-call(", "all-to-all(", "all-gather("))
    seen, dispatched = [], []
    real_admit = memory.admit
    real_book = memory.book
    budget = 1 << 30
    memory.set_budget(budget)

    def observe_admit(stage, price, room, **kwargs):
        if stage == "Sigma accumulator outputs":
            assert len(dispatched) == len(seen)  # no unadmitted zero dispatch
            assert price.bytes() == expected
            assert memory.run()["budget"] == budget
            seen.append(dict(room=room, price=price.bytes()))
        return real_admit(stage, price, room, **kwargs)

    def observe_book(price, *args, **kwargs):
        if price.label == "Sigma accumulator outputs":
            assert price.increment and len(seen) == len(dispatched) + 1
        return real_book(price, *args, **kwargs)

    class ObservedProgram:
        def __getattr__(self, name):
            return getattr(program, name)

        def __call__(self):
            assert len(seen) == len(dispatched) + 1, "output allocated before its admission"
            dispatched.append(True)
            return program()

    monkeypatch.setattr(memory, "admit", observe_admit)
    monkeypatch.setattr(memory, "book", observe_book)
    monkeypatch.setattr(owner, "_zeros_program", lambda *args: ObservedProgram())
    monkeypatch.setattr(tangential_execution, "compiled_for", lambda *_: ObservedProgram())

    def allocate():
        if grid:
            return owner.DeviceOmegaAccumulator.grid(np.linspace(-1, 1, nf),
                shape=shapes[0], sharding=sharding, omega_axis=0)
        request = SimpleNamespace(frequencies_ry=np.linspace(-1, 1, nf),
            parent_ry=np.zeros((slots - 1, nk, nb)),
            parent_index=np.zeros((slots - 1, nk, nb), np.int64), n_slot=slots)
        return owner.DeviceOmegaAccumulator(request, carrier=nb,
            shape=((shapes[0],), (shapes[1],)), sharding=sharding, reduce=None)

    first, second = allocate(), allocate()
    assert len(seen) == len(dispatched) == 2
    # The CPU reader divides this process's actual live bytes over all its
    # virtual devices, including devices outside this test's smaller mesh.
    resident_share = payload * mesh.size / jax.local_device_count()
    assert seen[1]["room"] <= seen[0]["room"] - resident_share
    for accumulator in (first, second):
        values = jax.tree.leaves(accumulator.finalize())
        assert [a.shape for a in values] == list(shapes)
        for value in values:
            assert value.sharding.is_equivalent_to(sharding, value.ndim)
            for shard in value.addressable_shards:
                np.testing.assert_array_equal(np.asarray(shard.data), 0)
    np.testing.assert_array_equal(np.asarray(prior), 1)

    # Preserve warn-and-run below the actual output price; a low budget is not
    # an injected allocation failure and the complete output still returns.
    memory.set_budget(1)
    budget = 1
    with pytest.warns(RuntimeWarning, match="priced over the limit"):
        low = allocate()
    assert len(seen) == len(dispatched) == 3 and seen[-1]["room"] == 0
    assert all(np.all(np.asarray(shard.data) == 0)
               for value in jax.tree.leaves(low.finalize()) for shard in value.addressable_shards)
