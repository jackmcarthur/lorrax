"""The round's packed pencil columns (``pack_panels``) on the CPU 2x2 mesh.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
The pack must equal the joined panels taken at ``order`` (the zero column past
the panels gives zeros), in the batch layout and on the face, for a held
extent past the panels' capacity too; and a round with more panels of the same
widths must reuse every pack program and give the round programs the same
shapes, so a new line-site count never recompiles them.
"""
import numpy as np
import pytest
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _round(rng, states, widths, slots=4, rows=12):
    """Panels of one round and its tables (ordered: originals, then mirrors)."""
    from gw.shared_pole_local import round_tables
    extent = lambda w: -(-int(w) // 2) * 2
    counts = rng.integers(1, min(widths) + 1, size=(slots, states))
    panels = [[rng.normal(size=(slots, rows, w)) + 1j * rng.normal(size=(slots, rows, w)) for w in widths]
              for _ in range(2)]
    return panels, counts, extent


def _reference(panels, order):
    joined = np.concatenate([*panels, np.zeros(panels[0].shape[:2] + (1,), panels[0].dtype)], axis=-1)
    return np.take_along_axis(joined, order[:, None, :], axis=-1)


def test_pack_equals_the_joined_take_in_both_layouts():
    from gw.shared_pole_local import BATCH, pack_panels, round_tables
    mesh = _mesh()
    rng = np.random.default_rng(7)
    widths = [4, 8, 4, 8]
    panels, counts, extent = _round(rng, len(widths), widths)
    history = {}
    # A held extent past this round's capacity (a wider earlier map) is more zero columns.
    history[(("sector", "CC", 12), "extent", 2)] = (14,)
    tables = round_tables(counts, widths, [0j] * len(widths), [0] * 4, 0, column_extent=extent,
                          ordered=True, odd_moments=False, key=("sector", "CC", 12), history=history)
    order = tables["order"]
    assert order.shape[-1] == 28 > sum(widths[:2]) + sum(widths[2:])
    want = [_reference(field, order) for field in panels]
    for layout, spec in (("batch", P(BATCH)), ("face", P(None, "x", "y"))):
        put = lambda a: jax.device_put(a, NamedSharding(mesh, spec))
        got = pack_panels(tuple([put(a) for a in field] for field in panels), order, widths,
                          mesh_xy=mesh, layout=layout)
        for g, w in zip(got, want):
            assert g.sharding.spec == spec, layout
            assert np.array_equal(np.asarray(g), w), layout


def test_more_panels_reuse_every_program():
    """Map 1 adds line panels of the same widths: no new pack program, the same packed shape."""
    from gw.shared_pole_local import _pack_place, pack_panels, round_tables
    mesh = _mesh()
    rng = np.random.default_rng(3)
    face = NamedSharding(mesh, P(None, "x", "y"))
    history, shapes = {}, []
    for states in (4, 6, 8):
        widths = [4, 8] * (states // 2)
        panels, counts, extent = _round(rng, states, widths)
        counts[:] = 0                                    # the same selection every map:
        counts[:, [0, states // 2]] = 2                  # one state per half takes two columns
        tables = round_tables(counts, widths, [0j] * states, [0] * 4, 0, column_extent=extent,
                              ordered=True, odd_moments=False, key=("sector", "TT", 12), history=history)
        before = _pack_place.cache_info().currsize
        got = pack_panels(tuple([jax.device_put(a, face) for a in field] for field in panels), tables["order"],
                          widths, mesh_xy=mesh, layout="face")
        if states > 4:
            assert _pack_place.cache_info().currsize == before, states
        shapes.append(tuple(g.shape for g in got))
        for g, field in zip(got, panels):
            assert np.array_equal(np.asarray(g), _reference(field, tables["order"]))
    # The extent is held by the sector alone, so the round programs see one shape.
    assert len(set(shapes)) == 1, shapes
