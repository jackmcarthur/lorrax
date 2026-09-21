"""Production sizing: the per-support direction cap and the pole budget (owner ruling 2026-09-17).

* ``right_singular_vectors(max_rank=)`` keeps the largest singular values above the cutoff, at most
  ``max_rank`` of them, and never splits a degenerate multiplet at the boundary.
* ``keep_budget`` in both reducers: K <= budget, every structural gate still passes, and a budget at or
  above the natural rank returns the unbudgeted model bit for bit. The ordered route removes the exact-zero
  carrier outside a binding budget without changing its projected response. RED TWIN: a budget below the
  natural rank changes the model.
"""
import numpy as np

from test_shared_pole_ordered import ZS, _ops, _put, _signed_value, _states, _trim, rel


def test_direction_cap_keeps_the_largest_and_closes_the_boundary_multiplet():
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from lxkit.testing import require_devices
    import distrib_la as D
    require_devices(4, "cpu")
    mesh = Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(1917)
    n = 8
    u = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    v = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    spectrum = np.array([9.0, 8.0, 6.0, 6.0, 4.0, 3.0, 2.0, 1.0])
    w = np.stack([(u * spectrum) @ v.conj().T])
    face = jax.make_array_from_callback(w.shape, NamedSharding(mesh, P(None, "x", "y")), lambda i: w[i])
    plan = D.plan("eigh", mesh, backend="off", n=2 * n, batched_route="batch_reshard")
    counts = {}
    for cap in (None, 5, 3, 2):
        _, values = D.right_singular_vectors(face, 0.3, eigh_plan=plan, column_extent=lambda r: 2 * ((r + 1) // 2),
                                             max_rank=cap)
        counts[cap] = int(np.asarray(values[0]).size)
    # cutoff 0.3*9 = 2.7 keeps 6; cap 5 -> 5; cap 3 lands inside the 6,6 pair -> 4; cap 2 -> 2.
    assert counts == {None: 6, 5: 5, 3: 4, 2: 2}, counts


def _even(plant, points, budget):
    import jax.numpy as jnp
    from gw.shared_pole_pencil import assemble_shared_pole_pencil
    from gw.shared_pole_reduction import reduce_shared_pole_pencil
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates
    mm, eigh = _ops()
    qi = np.linalg.eigh(plant.moment(1))[1][:, -1:]
    infinity = tuple(_put(a) for a in (qi, plant.moment(1) / 2 @ qi, plant.moment(3) / 2 @ qi))
    pencil = assemble_shared_pole_pencil(_states(plant, points, ordered=False), infinity, matmul=mm)
    active = jnp.ones((1, pencil[0].shape[-1]), bool)
    return reduce_shared_pole_pencil(pencil, active, eigh=eigh, matmul=mm, gates=gates, keep_budget=budget)


def _ordered_budget(plant, points, budget):
    import jax.numpy as jnp
    from gw.shared_pole_pencil import assemble_ordered_shared_pole_pencil
    from gw.shared_pole_reduction import reduce_ordered_shared_pole_pencil
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    mm, eigh = _ops()
    qi = np.linalg.eigh(plant.moment(1))[1][:, -1:]
    infinity = tuple(_put(a) for a in (qi, *(plant.moment(k) / 2 @ qi for k in range(4))))
    pencil = assemble_ordered_shared_pole_pencil(_states(plant, points), infinity, matmul=mm)
    active = jnp.ones((1, pencil[0].shape[-1]), bool)
    return reduce_ordered_shared_pole_pencil(pencil, active, eigh=eigh, matmul=mm, gates=gates, keep_budget=budget)


def _ordered_budget_with_span(plant, points, budget):
    import jax.numpy as jnp
    from gw.shared_pole_pencil import assemble_ordered_shared_pole_pencil
    from gw.shared_pole_reduction import reduce_ordered_shared_pole_pencil
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    mm, eigh = _ops()
    qi = np.linalg.eigh(plant.moment(1))[1][:, -1:]
    infinity = tuple(_put(a) for a in (qi, *(plant.moment(k) / 2 @ qi for k in range(4))))
    pencil = assemble_ordered_shared_pole_pencil(_states(plant, points), infinity, matmul=mm)
    active = jnp.ones((1, pencil[0].shape[-1]), bool)
    reduced = reduce_ordered_shared_pole_pencil(
        pencil, active, eigh=eigh, matmul=mm, gates=gates,
        keep_budget=budget, retain_span=True)
    return reduced, pencil


def test_pole_budget_caps_K_on_both_routes_and_is_inert_above_the_natural_rank():
    points = (.9 + .35j, 1.7 + .35j)
    # Even route on time-reversal-symmetric data.
    trs = _trim(np.random.default_rng(31), 12, 4, eps=0.0)
    model, diag, _ = _even(trs, points, None)
    natural = int(np.asarray(diag["retained_rank"])[0])
    assert natural > 6
    same, same_diag, _ = _even(trs, points, natural)
    for a, b in zip(model, same):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # The plant has 12 exact poles; a budget below that order must change W.
    budget = min(natural - 4, 8)
    capped, capped_diag, _ = _even(trs, points, budget)
    assert int(np.asarray(capped_diag["retained_rank"])[0]) == budget
    assert int(np.asarray(capped[2]).sum()) <= budget
    assert all(bool(np.asarray(capped_diag[k]).all()) for k in ("gram_diagonal_positive", "gram_valid", "retained_metric_positive"))
    b, poles, act = (np.asarray(a[0]) for a in capped)
    value = lambda z: (b[:, act] / (z * z - poles[act])) @ b[:, act].conj().T
    full = lambda z: (np.asarray(model[0][0])[:, np.asarray(model[2][0])]
                      / (z * z - np.asarray(model[1][0])[np.asarray(model[2][0])])) @ np.asarray(model[0][0])[:, np.asarray(model[2][0])].conj().T
    assert max(rel(value(z), full(z)) for z in ZS) > 1e-6          # red twin: the budget is not inert
    # Ordered route on broken time reversal.
    tr = _trim(np.random.default_rng(32), 12, 4, eps=.4)
    model, signed, diag = _ordered_budget(tr, points, None)
    k = int(np.asarray(diag["positive_count"])[0])
    count = int(np.asarray(diag["retained_rank"])[0])
    _, same_signed, _ = _ordered_budget(tr, points, count)
    for a, b in zip(signed, same_signed):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # 12 exact positive modes and a Gram rank of 17 here: a budget of 8 binds K.
    assert k == 12 and count > 12
    _, signed_c, diag_c = _ordered_budget(tr, points, 8)
    assert int(np.asarray(diag_c["retained_rank"])[0]) == 8
    assert int(np.asarray(diag_c["positive_count"])[0]) == 8
    assert bool(np.asarray(diag_c["gram_valid"]).all()) and bool(np.asarray(diag_c["retained_metric_positive"]).all())
    assert np.isrealobj(np.asarray(signed_c[1])) or np.all(np.asarray(signed_c[1]).imag == 0)
    assert max(rel(_signed_value(signed_c, z), _signed_value(signed, z)) for z in ZS) > 1e-6


def test_ordered_budget_removes_only_the_exact_zero_carrier(monkeypatch):
    import gw.shared_pole_reduction as reduction

    plant = _trim(np.random.default_rng(33), 12, 4, eps=.4)
    points = (.9 + .35j, 1.7 + .35j)
    budget = 8
    compact_extent = reduction._budget_carrier
    monkeypatch.setattr(
        reduction, "_budget_carrier",
        lambda width, keep_budget, matrix_sharding: int(width))
    reference, _ = _ordered_budget_with_span(plant, points, budget)
    monkeypatch.setattr(reduction, "_budget_carrier", compact_extent)
    compact, pencil = _ordered_budget_with_span(plant, points, budget)

    _, reference_signed, _, reference_span = reference
    _, compact_signed, compact_diagnostics, compact_span = compact
    assert compact_signed[1].shape[-1] == 2 * budget
    assert reference_signed[1].shape[-1] > compact_signed[1].shape[-1]
    assert compact_span.shape == (1, pencil[0].shape[-1], 2 * budget)
    assert int(np.asarray(compact_diagnostics["retained_rank"])[0]) == budget
    for z in ZS:
        assert rel(_signed_value(compact_signed, z),
                   _signed_value(reference_signed, z)) < 2e-11
    np.testing.assert_allclose(
        np.asarray(pencil[2]) @ np.asarray(compact_span),
        np.asarray(compact_signed[0])
        * np.asarray(compact_signed[2])[:, None, :],
        rtol=2e-11, atol=2e-11)
