"""Sector route edge cases that crashed main dc98076fe (review 2026-10-08, findings 1 and 2).

* The relaxed tier has no pole budget, so the face Ritz carrier is None: the CC/TT held-output
  price beside the decoupled stacks takes the whole H'_vv side (side // 2), as
  ``decoupled_stage_bytes`` does.
* CC and TT q-local with only CT on the face, nq >= P, decoupling refused: the face-rounds warning
  names CT and its q-local need instead of taking ``max()`` over an empty sequence.
"""
import warnings
from types import SimpleNamespace as NS

GB = 10**9


class _Ledger:
    """A ledger preview at a fixed budget: aggregate = resident + workspace."""
    device_budget_bytes_per_rank = 4 * GB

    def __init__(self, budget):
        self.budget = budget

    def preview(self, *, resident_bytes_per_rank, workspace_bytes_per_rank, concurrent_with):
        aggregate = int(resident_bytes_per_rank) + int(workspace_bytes_per_rank)
        return dict(aggregate_bytes_per_rank=aggregate, available_device_bytes_per_rank=self.budget,
                    device_budget_status='PASS' if aggregate <= self.budget else 'FAIL')


def _row(sector, side, packed, pole_budget):
    return dict(sector=sector, conservative_pencil_side=side, packed_extent=packed,
                infinity_width=16, pole_budget=pole_budget)


def test_decoupled_admission_prices_the_relaxed_tier():
    from gw.shared_pole_sectors import decoupled_admission
    mesh = NS(shape={'x': 2, 'y': 2}, size=4)
    rows = [_row('CC', 512, 128, None), _row('TT', 768, 384, None)]
    out = decoupled_admission(rows, 13, mesh_xy=mesh, ledger=_Ledger(40 * GB), upstream=())
    assert out['admitted'] and set(out['aggregate_bytes_per_rank']) == {'CC', 'TT'}
    # Without a budget the kept span is the whole H'_vv side: never priced below a budgeted one.
    budgeted = [_row('CC', 512, 128, 64), _row('TT', 768, 384, 96)]
    both = decoupled_admission(budgeted, 13, mesh_xy=mesh, ledger=_Ledger(40 * GB), upstream=())
    for name in ('CC', 'TT'):
        assert out['aggregate_bytes_per_rank'][name] >= both['aggregate_bytes_per_rank'][name] > 0


def test_ct_only_face_rounds_warning(monkeypatch):
    import gw.shared_pole_execution as ex
    import gw.shared_pole_sectors as sectors
    calls = []

    def constructor_execution(meta, resolution, recipe, **kwargs):
        calls.append(kwargs.get('cross_original_sides') is not None)
        local = dict(aggregate_bytes_per_rank=9 * GB, device_budget_status='FAIL')
        if calls[-1]:
            return 'face', dict(reason='CT', conservative_pencil_side=96,
                                local_selection=local, local_reduction=dict(local, aggregate_bytes_per_rank=11 * GB))
        return 'local', dict(reason='fits', conservative_pencil_side=64)
    monkeypatch.setattr(ex, 'constructor_execution', constructor_execution)
    monkeypatch.setattr(ex, 'line_panel_count', lambda recipe: 0)
    monkeypatch.setattr(ex, 'selection_face_count', lambda recipe, **kwargs: 4)
    monkeypatch.setattr(sectors, 'decoupled_admission', lambda *a, **k: dict(admitted=False))
    monkeypatch.setattr('gw.shared_pole_directions.port_extent', lambda mesh: (lambda w: int(w)))
    monkeypatch.setattr(sectors, 'sector_recipe',
                        lambda recipe, n: dict(recipe, n=n, pole_budget=16, infinity_width=4, line_direction_cap=8))
    meta = NS(shared_pole_recipe=dict(fit_ids=[0, 1]), shared_pole_capacity=_Ledger(4 * GB))
    bases = [NS(n_logical=16, n_packed=16), NS(n_logical=8, n_packed=8)]
    mesh = NS(shape={'x': 2, 'y': 2}, size=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mode, rows = sectors.sector_execution(meta, NS(backend=NS(linalg='local')), bases, 13,
                                              mesh_xy=mesh, upstream=())
    assert mode == 'face' and [r['mode'] for r in rows] == ['local', 'local']
    assert rows[0]['joint']['mode'] == 'face'
    text = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(text) == 1 and text[0].startswith('shared-pole sectors: CT local needs 11.0 GB/rank'), text
