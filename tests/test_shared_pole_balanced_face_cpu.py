"""Scalar face scheduling/admission and actual tiny CPU4 reducer equivalence."""
from types import SimpleNamespace as NS

import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)


def _chooser(monkeypatch, parents, compiled_price, *, reject_selection=()):
    import gw.shared_pole_capacity as capacity
    from gw.shared_pole_execution import face_batch_width
    queries, rooms = [], []

    class Budget:
        def __init__(self, *args, **kwargs):
            self.batch_width = 0
            self.program_bytes = None

        def quote(self, side, *, phase, **kwargs):
            queries.append((phase, self.batch_width, self.program_bytes))
            price = (1001 if self.batch_width in reject_selection else 0) if phase == 'selection' else int(self.program_bytes or 0)
            return dict(resident_bytes_per_rank=price), {}

    def preview(*, resident_bytes_per_rank, workspace_bytes_per_rank, **kwargs):
        total = resident_bytes_per_rank + workspace_bytes_per_rank
        return dict(aggregate_bytes_per_rank=total, available_device_bytes_per_rank=1000,
                    workspace_bytes_per_rank=workspace_bytes_per_rank,
                    device_budget_status='PASS' if total <= 1000 else 'FAIL')

    def room(row, held, *, stage):
        rooms.append((row.copy(), held, stage))
        return 777

    monkeypatch.setattr(capacity, 'ConstructorCapacity', Budget)
    monkeypatch.setattr(capacity, 'face_eigh_room', room)
    result = face_batch_width(NS(), NS(), mesh=NS(), ledger=NS(preview=preview),
        upstream=(), side=24, nq=parents, compiled=compiled_price,
        selection=dict(sample_batch=2))
    return result, queries, rooms


@pytest.mark.parametrize('parents,cap', [(10, 6), (10, 7), (10, 9), (17, 16), (61, 10), (1, 1)])
def test_balanced_candidate_cap_rounds_and_exact_parent_coverage(monkeypatch, parents, cap):
    from gw.shared_pole_local import parent_rounds
    calls = []
    def compiled(width, room):
        calls.append((width, room))
        return -(-width * 1000 // cap)
    (width, receipt), queries, rooms = _chooser(monkeypatch, parents, compiled)
    rounds = -(-parents // cap)
    assert width == -(-parents // rounds) <= cap
    old, new = parent_rounds(parents, cap), parent_rounds(parents, width)
    assert len(old) == len(new) == rounds
    assert [q for ids, real, _ in new for q in ids[:real]] == list(range(parents))
    assert sum(len(ids) for ids, _, _ in new) <= sum(len(ids) for ids, _, _ in old)
    assert receipt['parent_batch'] == width and calls[-1] == (width, 777)
    assert receipt['program_bytes_per_rank'] == -(-width * 1000 // cap)
    assert all(w == -(-parents // -(-parents // w)) for w, _ in calls)
    assert len(rooms) == len(calls)


def test_each_nonmonotonic_compiled_candidate_is_admitted_at_its_own_width(monkeypatch):
    calls = []
    prices = {10: 1001, 5: 1002, 4: 1003, 3: 100}
    def compiled(width, room):
        calls.append(width)
        return prices[width]
    (width, receipt), queries, _ = _chooser(monkeypatch, 10, compiled)
    assert calls == [10, 5, 4, 3] and width == 3
    assert receipt['program_bytes_per_rank'] == prices[3]
    assert [(w, p) for phase, w, p in queries if phase == 'reduction' and p] == list(prices.items())


def test_selection_rejection_precedes_compile_and_width_one_warn_policy_remains(monkeypatch):
    calls = []
    (width, _), queries, _ = _chooser(monkeypatch, 10,
        lambda width, room: calls.append(width) or 1, reject_selection=(10,))
    assert calls == [5] and width == 5
    calls.clear()
    (width, receipt), _, _ = _chooser(monkeypatch, 2,
        lambda width, room: calls.append(width) or 2001)
    assert calls == [2, 1] and width == 1
    assert receipt['reduction']['device_budget_status'] == 'FAIL'


def test_constructor_admits_actual_parent_count_not_full_grid():
    import ast
    from pathlib import Path
    tree = ast.parse(Path('src/gw/shared_pole_constructor.py').read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'face_batch_width']
    assert len(calls) == 1
    count = next(k.value for k in calls[0].keywords if k.arg == 'nq')
    assert ast.unparse(count) == "int(header['n_q_irr'])"


def test_balanced_and_padded_schedules_preserve_actual_face_models_and_gates(monkeypatch):
    from jax.sharding import NamedSharding, PartitionSpec as P
    from tests.test_shared_pole_decoupled import _mesh, _pencil, _pad, _w, _dw, _moment
    from gw.shared_pole_capacity import _local_eigenplan
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_local import parent_rounds, round_tables
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1
    import gw.shared_pole_execution as ex

    mesh = _mesh()
    monkeypatch.setattr(ex, 'face_eigh', lambda mesh_, n, room=None, backend=None:
                        _local_eigenplan(mesh_, int(n)))
    states, infinity, _, exact = _pencil(mesh, seed=41, n=12, k=12, width=2, iw=2, q=10)
    # The shared fixture supplies directions and Stieltjes W(s). Materialize
    # an actual paired-frequency model W(z)=W(s=z**2), with dW/dz and the
    # owner's physical convention W(z)=sum_j 2 M_j/z**(j+1).
    states = [(node, q,
               np.stack([_w(*exact[p], node[p]**2) @ q[p] for p in range(10)]),
               np.stack([2*node[p]*_dw(*exact[p], node[p]**2) @ q[p] for p in range(10)]))
              for node, q, _, _ in states]
    qi = infinity[0]
    moments = [np.zeros((10, 12, 12), complex),
               np.stack([.5*_moment(*m, 0) for m in exact]),
               np.zeros((10, 12, 12), complex),
               np.stack([.5*_moment(*m, 1) for m in exact])]
    infinity = (qi, *(np.einsum('pij,pjk->pik', m, qi) for m in moments))
    extent = port_extent(mesh)
    panel_width, iw = extent(2), extent(2)
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    put = lambda a: jax.device_put(np.asarray(a), face)
    gram_keep = shared_real_pole_gates_ordered_v1['normalized_gram_keep']['sector_threshold']

    def evaluate(width):
        models, diagnostics = [], []
        for ids, real, _ in parent_rounds(10, width):
            selected = [(s[0][ids], put(_pad(s[1][ids], panel_width)),
                         put(_pad(s[2][ids], panel_width)), put(_pad(s[3][ids], panel_width)))
                        for s in states]
            inf = tuple(put(_pad(a[ids], iw)) for a in infinity)
            table = round_tables(np.full((width, len(states)), 2), [panel_width] * len(states),
                [s[0][0] for s in states], [2] * width, iw,
                column_extent=extent, ordered=True, odd_moments=True)
            result = ex.face_reduce_round(selected, inf, table, mesh=mesh, budget=None,
                ordered=True, odd_moments=True, keep_budget=4, eigh_backend='auto',
                room=1 << 30, admit=False, gram_keep=gram_keep,
                carrier=ex.face_ritz_carrier(mesh, 4))
            host = jax.tree.map(np.asarray, result)
            for i in range(real):
                models.append(tuple(a[i] for a in host[0]))
                diagnostics.append({k: v[i] for k, v in host[3][0].items()})
        return models, diagnostics

    padded, pd = evaluate(9)
    balanced, bd = evaluate(5)
    assert len(padded) == len(balanced) == 10
    for old, new, a, b in zip(padded, balanced, pd, bd):
        factor, pole, active = old
        factor_new, pole_new, active_new = new
        np.testing.assert_array_equal(active, active_new)
        np.testing.assert_allclose(pole[active], pole_new[active], rtol=2e-8, atol=2e-10)
        for z in (-.3 + .4j, 1.2 + .7j):
            w = (factor * (active / (z - pole))[None]) @ factor.conj().T
            wn = (factor_new * (active_new / (z - pole_new))[None]) @ factor_new.conj().T
            np.testing.assert_allclose(w, wn, rtol=3e-7, atol=3e-9)
        for gate in ('gram_valid', 'retained_metric_positive', 'orientation_paired', 'retained_rank'):
            np.testing.assert_array_equal(a[gate], b[gate])
            assert bool(a[gate]), gate
