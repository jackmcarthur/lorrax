"""Authenticated constructor-only resume does not accept a partial producer."""
import json

from file_io import shared_pole_store as store
from gw.shared_pole_screening import _authenticated_constructor_resume


def _plant(root, identity):
    root.mkdir()
    coulomb = dict(path='/old/run/coulomb.h5', dataset='V', basis='canonical',
                   q_irr_full_idx=[0], sha256='abc')
    for name, extra in (('bank', {}), ('moments', {'bank_complete': True})):
        (root/f'{name}_receipt.json').write_text(json.dumps(dict(
            identity=identity, completion=True, coulomb_identity=coulomb,
            **extra)))
    (root/'coulomb.h5').write_bytes(b'coulomb')
    (root/'bank.h5').write_bytes(b'validated by the store service')


def test_constructor_resume_requires_exact_identity_recipe_and_complete_receipts(tmp_path,
                                                                                  monkeypatch):
    identity={'iteration_id':'sc_0000','hamiltonian':'map:abc'}
    recipe={'eta_ev': 0.2, 'z_ry': [0.1j]}
    root=tmp_path/'sc_0000_shared_pole'
    _plant(root,identity)
    def validate(path, *, expected_identity, mesh_xy, require_complete,
                 expected_recipe):
        assert path == root/'bank.h5' and mesh_xy is None and require_complete
        if expected_recipe != recipe:
            raise ValueError('stale recipe')
        return {'identity': expected_identity}
    monkeypatch.setattr(store, 'validate_shared_pole_bank', validate)
    assert _authenticated_constructor_resume(root,identity,recipe)
    assert not _authenticated_constructor_resume(
        root,dict(identity,hamiltonian='map:other'),recipe)
    assert not _authenticated_constructor_resume(root,identity,{'eta_ev': 0.3})
    receipt=json.loads((root/'moments_receipt.json').read_text())
    receipt['bank_complete']=False
    (root/'moments_receipt.json').write_text(json.dumps(receipt))
    assert not _authenticated_constructor_resume(root,identity,recipe)


def test_photon_constructor_resume_requires_complete_bound_bank(tmp_path, monkeypatch):
    identity = {'iteration_id': 'sc_0000', 'hamiltonian': 'map:abc'}
    recipe = {'eta_ev': 0.2, 'fit_ids': [1, 2]}
    root = tmp_path/'sc_0000_shared_pole'
    root.mkdir()
    source_bank = tmp_path/'completed_bank.h5'
    source_bank.write_bytes(b'validated by the store service')
    bank_path = root/'bank.h5'
    bank_path.symlink_to(source_bank)
    receipt_path = root/'bank_receipt.json'
    receipt = dict(identity=identity, completion=True, bank_complete=True,
                   stage='photon', static_reference=dict(identity=identity,
                                                         path=str(source_bank)))
    receipt_path.write_text(json.dumps(receipt))

    def validate(path, *, expected_identity, mesh_xy, require_complete,
                 expected_recipe):
        assert path == bank_path and mesh_xy is None and require_complete
        if expected_recipe != recipe:
            raise ValueError('stale recipe')
        return {'identity': expected_identity, 'photon_layout': {}}

    monkeypatch.setattr(store, 'validate_shared_pole_bank', validate)
    assert _authenticated_constructor_resume(root, identity, recipe, photon=True)
    assert not _authenticated_constructor_resume(
        root, dict(identity, hamiltonian='map:other'), recipe, photon=True)
    assert not _authenticated_constructor_resume(root, identity, {'eta_ev': 0.3}, photon=True)
    receipt['bank_complete'] = False
    receipt_path.write_text(json.dumps(receipt))
    assert not _authenticated_constructor_resume(root, identity, recipe, photon=True)
