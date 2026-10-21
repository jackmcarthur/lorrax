"""Tiny proposal proof; no production reducer or material admission."""
import importlib.util
from pathlib import Path


def test_actual_finite_owner_positive_H_proof_and_null_refusals():
    path = Path(__file__).with_name('diagnostics')/'verify_equilibrium_hermitian_pencil_host.py'
    spec = importlib.util.spec_from_file_location('eq_host_proof', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt = module.run_host_proof()
    assert receipt['status'] == 'HOST_ONLY_PROPOSED_EQ_PENCIL_PASS'
    assert receipt['state_ranks'] == [4, 4]
    assert not receipt['material_qualified'] and not receipt['production_route']
    assert len(receipt['required_refusals']) == 2
