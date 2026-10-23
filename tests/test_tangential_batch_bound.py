"""A discarded quote is unnecessary only when the existing step is already forced.

These host-only prices are injected policy metadata, never device-fit data.
The hsuite's small banks do not reach the conservative rejected proposals.
"""
import ast
import copy
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch


def policy(*, legacy=False):
    source=Path(__file__).parents[1]/'src/gw/tangential_execution.py'
    node=next(n for n in ast.parse(source.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='face_batch_width')
    if legacy:
        node=copy.deepcopy(node)
        loop=next(n for n in node.body if isinstance(n,ast.While))
        # Remove only the new forced-one shortcut, preserving the actual old policy.
        candidates=[n for n in loop.body if isinstance(n,ast.If) and len(n.body)==2
            and isinstance(n.body[0],ast.Assign) and isinstance(n.body[1],ast.Continue)
            and isinstance(n.body[0].value,ast.Constant) and n.body[0].value.value==1]
        assert len(candidates)==1
        loop.body.remove(candidates[0])
    scope={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    return scope['face_batch_width']


def run(fn, *, count, room, resident, workspace, compiled, selection=None):
    module=types.ModuleType('gw.tangential_capacity')
    calls=[]
    class Capacity:
        def __init__(self,*a,**kw):
            self.batch_width=1
            self.program_bytes=None
        def resident_quote(self,side,*,phase,**kw):
            return dict(resident_bytes_per_rank=(0 if phase=='selection' else
                resident[self.batch_width]+int(self.program_bytes or 0)))
        def quote(self,side,*,phase,**kw):
            return self.resident_quote(side,phase=phase),dict(eigh=(0 if phase=='selection'
                else workspace[self.batch_width]))
    class Ledger:
        def preview(self,*,resident_bytes_per_rank,workspace_bytes_per_rank,**kw):
            total=resident_bytes_per_rank+workspace_bytes_per_rank
            return dict(device_budget_status='PASS' if total<=room else 'FAIL',
                        aggregate_bytes_per_rank=total,available_device_bytes_per_rank=room,
                        workspace_bytes_per_rank=workspace_bytes_per_rank)
    module.ConstructorCapacity=Capacity
    module.face_eigh_room=lambda row,*a,**kw:row['available_device_bytes_per_rank']-row['aggregate_bytes_per_rank']
    def quote(width,room):
        calls.append(width)
        return compiled(width)
    with patch.dict(sys.modules,{'gw.tangential_capacity':module}):
        result=fn(None,None,mesh=None,ledger=Ledger(),upstream=(),side=1,nq=count,
                  compiled=quote,selection=selection)
    return result,calls


class ForcedMinimum(unittest.TestCase):
    def test_real_native_receipt_skips_only_the_doomed_quote(self):
        # Exact scalar rows from the archived P4 Fe native AOT replay.
        room=24_719_883_000
        resident={6:9_311_366_400,1:1_551_894_400}
        workspace={6:80_224_667_313,1:33_683_351_377}
        prices={6:205_488_697_493,1:31_215_187_903}
        args=dict(count=6,room=room,resident=resident,workspace=workspace)
        old,old_calls=run(policy(legacy=True),compiled=lambda w:prices[w],**args)
        def selected_only(width):
            self.assertEqual(width,1,'doomed full-program compile must not run')
            return prices[width]
        new,new_calls=run(policy(),compiled=selected_only,**args)
        self.assertEqual(new,old)
        self.assertEqual(old_calls,[6,1])
        self.assertEqual(new_calls,[1])

    def test_boundary_and_nonmonotone_price_properties(self):
        rng=random.Random(317)
        for _ in range(1000):
            n=rng.randint(1,31)
            room=rng.randint(0,600)
            resident={w:rng.randint(0,1200) for w in range(1,n+1)}
            workspace={w:rng.randint(0,1200) for w in resident}
            prices={w:rng.randint(0,2400) for w in resident}
            args=dict(count=n,room=room,resident=resident,workspace=workspace,
                      compiled=lambda w:prices[w])
            old,old_calls=run(policy(legacy=True),**args)
            new,new_calls=run(policy(),**args)
            self.assertEqual(new,old)
            self.assertTrue(set(new_calls).issubset(old_calls))
        # Floor division equals one at its boundary; two still needs its quote.
        for lower,expected in ((101, [1]),(100,[2,1])):
            result,calls=run(policy(),count=2,room=100,resident={1:0,2:lower},
                workspace={1:0,2:0},compiled=lambda w:200 if w==2 else 1)
            self.assertEqual(calls,expected)

    def test_zero_bound_and_selected_exceptions_are_preserved(self):
        result,calls=run(policy(),count=2,room=0,resident={1:0,2:0},workspace={1:0,2:0},
                         compiled=lambda w:1)
        self.assertEqual(calls,[2,1])
        error=RuntimeError('selected compile failure')
        def fail(width):
            raise error
        for legacy in (False,True):
            with self.assertRaises(RuntimeError) as caught:
                run(policy(legacy=legacy),count=1,room=0,resident={1:1},workspace={1:1},compiled=fail)
            self.assertIs(caught.exception,error)


if __name__=='__main__':
    unittest.main()
