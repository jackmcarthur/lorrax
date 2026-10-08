"""Sector route edge cases that crashed main dc98076fe (review 2026-10-08, findings 1 and 2).

* The relaxed tier has no pole budget, so the face Ritz carrier is None: the staged stacks
  then take the whole H'_vv side (side // 2) as the kept span.
* CC and TT q-local with only CT on the face: the map runs the staged face route, and the route
  line names every sector's q-local need (the old face-rounds warning took ``max()`` over an
  empty sequence there).
"""
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
                    device_budget_bytes_per_rank=self.budget,
                    device_budget_status='PASS' if aggregate <= self.budget else 'FAIL')


def test_staged_prices_take_the_whole_side_on_the_relaxed_tier():
    from gw.shared_pole_capacity import staged_sector_bytes
    relaxed = staged_sector_bytes(parents=4, ranks=4, side=768, carrier=None, packed=384)
    whole = staged_sector_bytes(parents=4, ranks=4, side=768, carrier=384, packed=384)
    budgeted = staged_sector_bytes(parents=4, ranks=4, side=768, carrier=96, packed=384)
    assert relaxed == whole
    assert [m for m, _ in relaxed[1]] == [384, 384, 768]
    assert relaxed[0] > budgeted[0] > 0


def test_ct_only_face_runs_the_staged_route(monkeypatch, capsys):
    import gw.shared_pole_execution as ex
    import gw.shared_pole_sectors as sectors
    calls = []

    def constructor_execution(meta, resolution, recipe, **kwargs):
        calls.append(kwargs.get('cross_original_sides') is not None)
        local = dict(aggregate_bytes_per_rank=9 * GB, device_budget_status='FAIL', device_budget_bytes_per_rank=4 * GB)
        if calls[-1]:
            return 'face', dict(reason='CT', conservative_pencil_side=96,
                                local_selection=local, local_reduction=dict(local, aggregate_bytes_per_rank=11 * GB))
        fits = dict(local, aggregate_bytes_per_rank=GB, device_budget_status='PASS')
        return 'local', dict(reason='fits', conservative_pencil_side=64, local_selection=fits, local_reduction=fits)
    monkeypatch.setattr(ex, 'constructor_execution', constructor_execution)
    monkeypatch.setattr(ex, 'line_panel_count', lambda recipe: 0)
    monkeypatch.setattr(ex, 'selection_face_count', lambda recipe, **kwargs: 4)
    monkeypatch.setattr('gw.shared_pole_directions.port_extent', lambda mesh: (lambda w: int(w)))
    monkeypatch.setattr(sectors, 'sector_recipe',
                        lambda recipe, n: dict(recipe, n=n, pole_budget=None, infinity_width=4, line_direction_cap=8))
    meta = NS(shared_pole_recipe=dict(fit_ids=[0, 1]), shared_pole_capacity=_Ledger(4 * GB))
    bases = [NS(n_logical=16, n_packed=16), NS(n_logical=8, n_packed=8)]
    mesh = NS(shape={'x': 2, 'y': 2}, size=4)
    mode, rows, route = sectors.sector_route(meta, NS(backend=NS(linalg='local')), bases, 13, mesh_xy=mesh, upstream=())
    assert mode == 'face' and [r['mode'] for r in rows] == ['local', 'local'] and rows[0]['joint']['mode'] == 'face'
    assert route == dict(width=4, rounds=4)
    tiles = rows[0]['staged_route']
    assert set(tiles) == {'CC', 'TT', 'CT'} and tiles['CC']['selection'] == 4 and tiles['TT']['selection'] == 4
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith('Shared-pole sector constructor: route')]
    assert line == ['Shared-pole sector constructor: route rounds of 4 of 13 parents (4 round(s)) on the face, staged; selection '
                    'tiles CC 4, TT 4 parents; q-local needs GB/rank CC 1.0, TT 1.0, CT 11.0 of 4.0'], line
