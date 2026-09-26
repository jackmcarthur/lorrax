"""The sector writer's held carrier: grow-only, logged, admitted, and inert."""

import numpy as np


def test_writer_carrier_is_held_across_rounds_and_grows_only():
    from gw.shared_pole_sectors import held_writer_width

    history, events = {}, []
    key = ("writer", "TT")
    fits = lambda width: True
    # The first round sets the carrier without a note (a map-0 or one-shot round).
    assert held_writer_width(1280, 1200, history, key, fits, None) == 1280
    assert history[key] == 1280
    # A smaller live carrier writes at the held one.
    assert held_writer_width(1152, 1100, history, key, fits, events) == 1280
    assert events == []
    # A live carrier past the hold grows it and says so in the SC log's list.
    assert held_writer_width(1408, 1300, history, key, fits, events) == 1408
    assert events == ["shared-pole writer carrier (TT): Kmax 1300 exceeds the held width 1280; "
                      "grown to 1408"]
    # No SC log bound (map 0): the hold still grows, silently.
    assert held_writer_width(1536, 1500, history, key, fits, None) == 1536
    assert len(events) == 1 and history[key] == 1536


def test_writer_carrier_falls_back_to_the_live_width_when_the_store_refuses_it():
    from gw.shared_pole_sectors import held_writer_width

    history = {("writer", "CC"): 1408}
    asked = []

    def refuse(width):
        asked.append(width)
        return False
    assert held_writer_width(1152, 1100, history, ("writer", "CC"), refuse, []) == 1152
    assert asked == [1408] and history[("writer", "CC")] == 1408
    # The live carrier equal to the hold needs no admission question.
    assert held_writer_width(1408, 1400, history, ("writer", "CC"), refuse, []) == 1408
    assert asked == [1408]


def test_face_rows_appends_zero_columns_past_its_carrier():
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw.shared_pole_local import face_rows

    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ("x", "y"))
    stack = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4) + 1
    array = jax.device_put(stack, NamedSharding(mesh, P(None, "x", "y")))
    wide = np.asarray(face_rows(mesh, (1, 0), 6)(array))
    assert wide.shape == (2, 3, 6)
    np.testing.assert_array_equal(wide[:, :, :4], stack[[1, 0]])
    assert np.all(wide[:, :, 4:] == 0)
    # At or below the carrier it only selects.
    np.testing.assert_array_equal(np.asarray(face_rows(mesh, (0,), 3)(array)), stack[[0], :, :3])
