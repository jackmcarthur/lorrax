"""Scalar restart policy with a stubbed store; no HDF5 authentication claim."""
import copy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from gw.shared_pole_recipe import (
    CapacityLedger, shared_pole_restart_handle, bind_shared_pole_sc_identity,
)


class Missing(ValueError):
    pass


class Refused(ValueError):
    pass


class RestartRecipeTests(unittest.TestCase):
    def setUp(self):
        self.mesh = NS(shape={'x': 2, 'y': 2})
        self.recipe = dict(recipe_version='recipe-v1', recipe_hash='recipe-hash',
                           gate_version='gate-v1', gate_hash='gate-hash',
                           accuracy='production', eta_ev=.25, n=4)
        self.meta = NS(nk_tot=4, nspinor=1, n_rmu=4,
                       shared_pole_recipe=self.recipe)
        self.meta.shared_pole_capacity = CapacityLedger(self.meta, mesh_xy=self.mesh)
        self.meta.shared_pole_capacity.live_stages = ()
        self.identity = {'iteration_id': 'oneshot', 'energies': 'current'}
        self.member = dict(path='shared_pole.h5', digest='payload-digest',
                           schema='schema', iteration_id='oneshot')
        self.header = dict(recipe=copy.deepcopy(self.recipe), identity=self.identity, K=[2, 0, 3])
        self.reader = Mock(return_value=(self.member, self.header))
        module = ModuleType('file_io.tagged_arrays')
        module.read_shared_pole_restart_member = self.reader
        module.SharedPoleMemberMissing = Missing
        module.SharedPoleMemberRefused = Refused
        self.module_patch = patch.dict(sys.modules, {'file_io.tagged_arrays': module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.messages = []

    def call(self):
        return shared_pole_restart_handle(
            '/tmp/iinputs_scalar/bundle.h5', expected_identity=self.identity,
            meta=self.meta, mesh_xy=self.mesh, print_fn=self.messages.append)

    def test_present_uses_one_authentication_and_small_handle(self):
        handle = self.call()
        self.assertEqual(set(handle), {'path', 'identity', 'digest', 'K'})
        self.assertEqual(handle['path'], str(Path('/tmp/iinputs_scalar/shared_pole.h5').resolve()))
        self.assertEqual(handle['K'], [2, 0, 3])
        self.reader.assert_called_once_with(
            '/tmp/iinputs_scalar/bundle.h5', expected_identity=self.identity,
            mesh_xy=self.mesh, capacity=self.meta.shared_pole_capacity, return_header=True)
        self.assertIn('skip bank', self.messages[0])
        handle['K'][0] = 999
        self.assertEqual(self.header['K'][0], 2)

    def test_only_typed_missing_allows_rebuild(self):
        self.reader.side_effect = Missing('committed bundle has no member')
        self.assertIsNone(self.call())
        self.assertIn('build and register', self.messages[0])

    def test_refusals_are_not_reclassified_as_missing(self):
        for error in (Refused('corrupt payload'), ValueError('partial bundle'),
                      FileNotFoundError('missing linked model')):
            with self.subTest(error=error):
                self.reader.side_effect = error
                with self.assertRaises(type(error)) as caught:
                    self.call()
                self.assertIs(caught.exception, error)

    def test_current_physical_recipe_mismatches_refuse(self):
        for key in self.recipe:
            with self.subTest(key=key):
                self.header['recipe'] = copy.deepcopy(self.recipe)
                self.header['recipe'][key] = 'changed'
                with self.assertRaisesRegex(Refused, f'recipe {key} mismatch'):
                    self.call()

    def test_mesh_planning_bytes_do_not_change_physical_identity(self):
        self.recipe['U_bytes_per_rank'] = 256
        self.header['recipe']['U_bytes_per_rank'] = 1024
        self.assertEqual(self.call()['digest'], 'payload-digest')

    def test_missing_recipe_or_unbound_lifetimes_refuses_before_store(self):
        self.meta.shared_pole_recipe = None
        with self.assertRaisesRegex(ValueError, 'resolved recipe is missing'):
            self.call()
        self.meta.shared_pole_recipe = self.recipe
        self.meta.shared_pole_capacity = CapacityLedger(self.meta, mesh_xy=self.mesh)
        with self.assertRaisesRegex(ValueError, 'unbound caller lifetimes'):
            self.call()
        self.reader.assert_not_called()

    def test_sc_labels_are_non_authenticating_and_rebound(self):
        first = bind_shared_pole_sc_identity(
            self.meta, NS(iteration=2), occupation_state=NS(occ_hash='current-occ'),
            print_fn=self.messages.append)
        self.assertEqual(first['hamiltonian'], 'sc_map_2:current-occ')
        self.assertEqual(first['wavefunctions'], 'qp_rotation_unreceipted')
        self.assertEqual(first['authentication'], 'NON-AUTHENTICATING')
        self.assertIn('NON-AUTHENTICATING', self.messages[-1])
        second = bind_shared_pole_sc_identity(
            self.meta, NS(iteration=3), occupation_state=NS(occ_hash='next-occ'),
            print_fn=self.messages.append)
        self.assertNotEqual(first['hamiltonian'], second['hamiltonian'])
        self.assertEqual(self.meta.shared_pole_state_identity, second)
        self.assertEqual(second['recipe_hash'], self.recipe['recipe_hash'])

    def test_insulating_sc_reuses_existing_census_label(self):
        self.meta.shared_pole_census = {'occupation_sha256': 'step-occ'}
        identity = bind_shared_pole_sc_identity(
            self.meta, NS(iteration=0), occupation_state=None, print_fn=self.messages.append)
        self.assertEqual(identity['hamiltonian'], 'sc_map_0:step-occ')

    def test_sc_cannot_enter_restart_authentication(self):
        for identity in ({'iteration_id': 'sc_0000'},
                         {'wavefunctions': 'qp_rotation_unreceipted'},
                         {'authentication': 'NON-AUTHENTICATING'},
                         {'hamiltonian': 'sc_map_0:occ'}):
            with self.subTest(identity=identity):
                self.identity = identity
                with self.assertRaisesRegex(ValueError, 'NON-AUTHENTICATING'):
                    self.call()
        self.reader.assert_not_called()

    def test_sc_missing_current_occupation_refuses(self):
        with self.assertRaisesRegex(ValueError, 'occupation label is missing'):
            bind_shared_pole_sc_identity(
                self.meta, NS(iteration=0), occupation_state=NS(occ_hash=None),
                print_fn=self.messages.append)


if __name__ == '__main__':
    unittest.main(verbosity=2)
