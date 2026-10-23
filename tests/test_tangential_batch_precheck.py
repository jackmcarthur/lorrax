"""Host-only route semantics: a workspace cannot rescue a resident rejection.

The hsuite's small banks do not exercise these rejected-width proposals.
These dimensionless policy prices are injected; they are not device-fit data.
"""
import ast
from pathlib import Path
import types
import sys
import unittest
from unittest.mock import patch


class BatchProposal(unittest.TestCase):
    def run_policy(self, resident, workspace, room, *, selection=True):
        source = Path(__file__).parents[1] / 'src/gw/tangential_execution.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'face_batch_width')
        module = types.ModuleType('gw.tangential_capacity')
        queried = []

        class Capacity:
            def __init__(self, *a, **kw):
                self.batch_width, self.program_bytes = 1, None
            def resident_quote(self, side, *, phase, **kw):
                value = resident[self.batch_width] if phase == 'selection' else int(self.program_bytes or 0)
                return {'resident_bytes_per_rank': value}
            def quote(self, side, *, phase, **kw):
                if phase == 'selection':
                    queried.append(self.batch_width)
                    return self.resident_quote(side, phase=phase), {'eigh': workspace[self.batch_width]}
                return self.resident_quote(side, phase=phase), {'eigh': 0}

        class Ledger:
            def preview(self, *, resident_bytes_per_rank, workspace_bytes_per_rank, **kw):
                total = resident_bytes_per_rank + workspace_bytes_per_rank
                return {'device_budget_status': 'PASS' if total <= room else 'FAIL',
                        'aggregate_bytes_per_rank': total,
                        'available_device_bytes_per_rank': room,
                        'workspace_bytes_per_rank': workspace_bytes_per_rank}

        module.ConstructorCapacity = Capacity
        module.face_eigh_room = lambda *a, **kw: max(0, room)
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
        with patch.dict(sys.modules, {'gw.tangential_capacity': module}):
            chosen, receipt = scope['face_batch_width'](
                None, None, mesh=None, ledger=Ledger(), upstream=(), side=1,
                nq=max(resident), compiled=lambda width, room: 1,
                selection={'sample_batch': 1} if selection else None)
        return chosen, receipt, queried

    def test_full_price_winner_and_last_parent_are_preserved(self):
        for room in (0, 40, 110, 220, 2000):
            # Nonmonotone workspace ensures that skipping a resident failure
            # does not assume the provider's total price is monotone.
            resident = {w: 7*w + (w % 4) for w in range(1, 24)}
            workspace = {w: (w % 5)*13 for w in resident}
            expected = max((w for w in resident if resident[w]+workspace[w] <= room), default=1)
            chosen, receipt, queried = self.run_policy(resident, workspace, room)
            self.assertEqual(chosen, expected)
            self.assertIn(chosen, queried)
            self.assertTrue(all(w == 1 or resident[w] <= room for w in queried))
            self.assertEqual(receipt['selection']['aggregate_bytes_per_rank'],
                             resident[chosen] + workspace[chosen])

    def test_absent_selection_keeps_reduction_route(self):
        chosen, receipt, queried = self.run_policy({w: w for w in range(1, 8)},
                                                   {w: 0 for w in range(1, 8)}, 10,
                                                   selection=False)
        self.assertEqual(chosen, 7)
        self.assertIsNone(receipt['selection'])
        self.assertEqual(queried, [])


if __name__ == '__main__':
    unittest.main()
