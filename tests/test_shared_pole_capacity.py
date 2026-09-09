"""Scalar plan-time capacity checks; no device, HDF5 or dense allocations."""
import unittest
from types import SimpleNamespace as NS

from gw.shared_pole_recipe import CapacityLedger, construction_receipt


class CapacityTests(unittest.TestCase):
    def test_quote_preserves_ledger_and_matches_admission(self):
        ledger = self.ledger()
        ledger.reserve('ambient', resident_bytes_per_rank=128,
                       workspace_bytes_per_rank=64)
        for resident, expected in ((576, 'PASS'), (577, 'FAIL')):
            args = dict(resident_bytes_per_rank=resident,
                        workspace_bytes_per_rank=0, concurrent_with=('ambient',))
            quote = ledger.quote('candidate', **args)
            self.assertEqual(quote['device_budget_status'], expected)
            self.assertEqual(len(ledger.entries), 1)
        args['resident_bytes_per_rank'] = 576
        self.assertEqual(ledger.quote('candidate', **args),
                         ledger.reserve('candidate', **args))

    def ledger(self):
        return CapacityLedger(NS(nk_tot=4,nspinor=1,n_rmu=4),
                              mesh_xy=NS(shape={'x':2,'y':2}))

    def test_receipt_segments_reconstruct_without_changing_gates(self):
        ledger = self.ledger()
        ledger.reserve('first', resident_bytes_per_rank=256, workspace_bytes_per_rank=0)
        first = construction_receipt(capacity=ledger, capacity_entry_start=0)
        ledger.reserve('second', resident_bytes_per_rank=128, workspace_bytes_per_rank=0,
                       concurrent_with=('first',))
        second = construction_receipt(capacity=ledger, capacity_entry_start=1)
        complete = construction_receipt(capacity=ledger)
        self.assertEqual(second['gates'], complete['gates'])
        self.assertEqual(first['capacity']['entry_span'], [0, 1])
        self.assertEqual(second['capacity']['entry_span'], [1, 2])
        self.assertEqual(first['capacity']['entries'] + second['capacity']['entries'],
                         complete['capacity']['entries'])
        first['capacity']['entries'][0]['status'] = 'mutated'
        self.assertEqual(ledger.entries[0]['status'], 'PASS')
        self.assertNotIn('entry_span', complete['capacity'])
        for start in (-1, 3, True):
            with self.assertRaises(ValueError):
                ledger.receipt(entry_start=start)

    def test_receipt_segment_keeps_earlier_failed_admission(self):
        ledger = self.ledger()
        with self.assertRaises(MemoryError):
            ledger.reserve('failed', resident_bytes_per_rank=769, workspace_bytes_per_rank=0)
        segment = construction_receipt(capacity=ledger, capacity_entry_start=1)
        complete = construction_receipt(capacity=ledger)
        self.assertEqual(segment['capacity']['entries'], [])
        self.assertEqual(segment['gates'], complete['gates'])
        capacity_gate = next(row for row in segment['gates'] if row['name'] == 'capacity')
        self.assertEqual(capacity_gate['status'], 'FAIL')

    def test_geometry_and_exact_boundary(self):
        ledger=self.ledger()
        self.assertEqual(ledger.U_bytes_per_rank,256)
        self.assertEqual(ledger.limit_bytes_per_rank,768)
        self.assertEqual(ledger.reserve('fit',resident_bytes_per_rank=512,
                         workspace_bytes_per_rank=256)['status'],'PASS')

    def test_workspace_and_concurrent_refusal(self):
        ledger=self.ledger()
        ledger.reserve('inputs',resident_bytes_per_rank=512,workspace_bytes_per_rank=0)
        with self.assertRaisesRegex(MemoryError,'Px=2, Py=2'):
            ledger.reserve('fit',resident_bytes_per_rank=128,workspace_bytes_per_rank=129,
                           concurrent_with=('inputs',))
        row=ledger.receipt()['entries'][-1]
        self.assertEqual(row['status'],'FAIL')
        self.assertEqual(row['aggregate_bytes_per_rank'],769)
        self.assertEqual(row['max_mesh_ranks_at_fixed_bytes'],3)

    def test_sequential_not_accumulated(self):
        ledger=self.ledger()
        for stage in ('bank','constructor','sigma'):
            self.assertEqual(ledger.reserve(stage,resident_bytes_per_rank=768,
                             workspace_bytes_per_rank=0)['status'],'PASS')

    def test_explicit_concurrency_deduplicates_without_stale_lifetimes(self):
        ledger=self.ledger()
        ledger.reserve('inputs',resident_bytes_per_rank=256,workspace_bytes_per_rank=64)
        ledger.reserve('bank',resident_bytes_per_rank=128,workspace_bytes_per_rank=0,
                       concurrent_with=('inputs',))
        row=ledger.reserve('fit',resident_bytes_per_rank=256,workspace_bytes_per_rank=0,
                           concurrent_with=('inputs','bank','inputs'))
        self.assertEqual(row['aggregate_bytes_per_rank'],704)
        later=ledger.reserve('later',resident_bytes_per_rank=65,workspace_bytes_per_rank=0,
                             concurrent_with=('fit',))
        self.assertEqual(later['aggregate_bytes_per_rank'],321)
        with self.assertRaises(MemoryError):
            ledger.reserve('extra',resident_bytes_per_rank=65,workspace_bytes_per_rank=0,
                           concurrent_with=('inputs','bank','fit'))

    def test_unreserved_and_failed_dependencies_refuse(self):
        ledger=self.ledger()
        with self.assertRaises(MemoryError):
            ledger.reserve('failed',resident_bytes_per_rank=769,workspace_bytes_per_rank=0)
        for dependency in ('unknown','failed'):
            with self.assertRaises(ValueError):
                ledger.reserve('fit',resident_bytes_per_rank=0,workspace_bytes_per_rank=0,
                               concurrent_with=(dependency,))

    def test_invalid_bytes_and_geometry(self):
        for value in (-1,float('nan'),float('inf'),1.5,True,None):
            with self.assertRaises(ValueError):
                self.ledger().reserve('bad',resident_bytes_per_rank=value,workspace_bytes_per_rank=0)
        with self.assertRaises(ValueError):
            CapacityLedger(NS(nk_tot=0,nspinor=1,n_rmu=4),mesh_xy=NS(shape={'x':2,'y':2}))

    def test_duplicate_name_and_string_dependency(self):
        ledger=self.ledger()
        ledger.reserve('inputs',resident_bytes_per_rank=1,workspace_bytes_per_rank=0)
        with self.assertRaises(ValueError):
            ledger.reserve('inputs',resident_bytes_per_rank=1,workspace_bytes_per_rank=0)
        with self.assertRaises(ValueError):
            ledger.reserve('fit',resident_bytes_per_rank=1,workspace_bytes_per_rank=0,
                           concurrent_with='inputs')

    def test_receipt_preserves_fail_and_unmeasured_peak(self):
        ledger=self.ledger()
        self.assertEqual(construction_receipt(capacity=ledger)['gates'][-1]['status'],'NOT_MEASURED')
        with self.assertRaises(MemoryError):
            ledger.reserve('fit',resident_bytes_per_rank=1000,workspace_bytes_per_rank=0)
        receipt=construction_receipt(capacity=ledger)
        self.assertEqual(receipt['gates'][-1]['status'],'FAIL')
        self.assertEqual(receipt['capacity']['measured_peak']['status'],'NOT_MEASURED')

    def test_measured_peak_independent_of_plan(self):
        ledger=self.ledger()
        ledger.reserve('fit',resident_bytes_per_rank=500,workspace_bytes_per_rank=0)
        self.assertEqual(ledger.record_measured_peak(800,reason='measured max across ranks')['status'],'FAIL')
        self.assertEqual(ledger.entries[0]['status'],'PASS')
        self.assertEqual(ledger.record_measured_peak(700,reason='later smaller reading')['status'],'FAIL')
        self.assertEqual(construction_receipt(capacity=ledger)['gates'][-1]['status'],'FAIL')

    def test_receipt_is_detached(self):
        ledger=self.ledger()
        row=ledger.reserve('fit',resident_bytes_per_rank=100,workspace_bytes_per_rank=0)
        row['resident_bytes_per_rank']=1000
        ledger.receipt()['entries'][0]['status']='FAIL'
        self.assertEqual(ledger.receipt()['entries'][0]['status'],'PASS')
        self.assertEqual(ledger.receipt()['entries'][0]['resident_bytes_per_rank'],100)

    def test_callee_lifetimes_require_explicit_binding(self):
        ledger=self.ledger()
        with self.assertRaisesRegex(ValueError,'unbound caller lifetimes'):
            _=ledger.live_stages
        ledger.live_stages=()
        self.assertEqual(ledger.live_stages,())
        ledger.reserve('inputs',resident_bytes_per_rank=512,workspace_bytes_per_rank=0)
        ledger.live_stages=('inputs','inputs')
        self.assertEqual(ledger.live_stages,('inputs',))
        with self.assertRaises(MemoryError):
            ledger.reserve('reader',resident_bytes_per_rank=257,workspace_bytes_per_rank=0,
                           concurrent_with=ledger.live_stages)
        with self.assertRaises(ValueError):
            ledger.live_stages=('unknown',)
        self.assertEqual(ledger.live_stages,('inputs',))
        ledger.live_stages=()
        self.assertEqual(ledger.reserve('sequential',resident_bytes_per_rank=768,
                         workspace_bytes_per_rank=0,concurrent_with=ledger.live_stages)['status'],'PASS')

    def test_inherited_stream_separate_from_new_objects(self):
        ledger=self.ledger()
        ledger.reserve('bank_outputs',resident_bytes_per_rank=768,workspace_bytes_per_rank=0)
        row=ledger.record_stream_peak(1050,1000,reason='same deck/P and compile method; scalar twin')
        self.assertEqual(row['status'],'PASS')
        self.assertEqual(row['value']['incumbent_bytes_per_rank'],1000)
        receipt=construction_receipt(capacity=ledger)
        gates={row['name']:row for row in receipt['gates']}
        self.assertEqual(gates['capacity']['status'],'PASS')
        self.assertEqual(gates['stream_peak']['status'],'PASS')
        self.assertEqual(len(ledger.entries),1)
        with self.assertRaises(ValueError):
            ledger.live_stages=('stream_peak',)

    def test_stream_regression_refuses_and_cannot_erase_failure(self):
        ledger=self.ledger()
        with self.assertRaisesRegex(MemoryError,'inherited stream_peak regressed'):
            ledger.record_stream_peak(1051,1000,reason='same deck/P and compile method; red twin')
        self.assertEqual(ledger.receipt()['stream_peak']['status'],'FAIL')
        with self.assertRaises(ValueError):
            ledger.record_stream_peak(1000,1000,reason='cannot overwrite earlier failure')

    def test_stream_missing_and_invalid_baseline(self):
        ledger=self.ledger()
        self.assertEqual(ledger.record_stream_peak(1000,None,reason='incumbent absent')['status'],
                         'NOT_MEASURED')
        with self.assertRaises(ValueError):
            ledger.record_stream_peak(0,0,reason='invalid incumbent')
        self.assertEqual(ledger.record_stream_peak(1000,1000,reason='paired measurements')['status'],
                         'PASS')

    def test_sigma_inherited_and_second_w_are_separate(self):
        ledger=self.ledger()
        ledger.record_sigma_peak(1050,1000,reason='same deck/P/window and compile method; scalar twin')
        ledger.reserve('faces',resident_bytes_per_rank=512,workspace_bytes_per_rank=0)
        with self.assertRaises(MemoryError):
            ledger.reserve('second_w',resident_bytes_per_rank=257,workspace_bytes_per_rank=0,
                           concurrent_with=('faces',))
        gates={r['name']:r for r in construction_receipt(capacity=ledger)['gates']}
        self.assertEqual(gates['sigma_peak']['status'],'PASS')
        self.assertEqual(gates['capacity']['status'],'FAIL')
        self.assertEqual(gates['stream_peak']['status'],'NOT_MEASURED')

    def test_sigma_regression_and_missing_baseline(self):
        ledger=self.ledger()
        self.assertEqual(ledger.record_sigma_peak(None,1000,reason='shared measurement absent')['status'],
                         'NOT_MEASURED')
        with self.assertRaisesRegex(MemoryError,'shared_pole_sigma_peak'):
            ledger.record_sigma_peak(1051,1000,reason='same deck/P/window; red twin')
        self.assertEqual(ledger.receipt()['sigma_peak']['status'],'FAIL')


if __name__=='__main__':
    unittest.main(verbosity=2)
