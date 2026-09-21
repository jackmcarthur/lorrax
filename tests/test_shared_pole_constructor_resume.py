"""Authenticated constructor-only resume does not accept a partial producer."""
import hashlib
import json

import h5py
import numpy as np

from file_io.commit_state import set_commit_state
from file_io.shared_pole_store import _json as store_json
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
    header=dict(identity=identity, complete=True, final_commit=None)
    header['final_commit']=hashlib.sha256(store_json(header).encode()).hexdigest()
    with h5py.File(root/'bank.h5','w') as h5:
        h5.create_dataset('header_json',data=np.bytes_(store_json(header)))
        set_commit_state(h5,True)


def test_constructor_resume_requires_exact_identity_and_complete_receipts(tmp_path):
    identity={'iteration_id':'sc_0000','hamiltonian':'map:abc'}
    root=tmp_path/'sc_0000_shared_pole'
    _plant(root,identity)
    assert _authenticated_constructor_resume(root,identity)
    assert not _authenticated_constructor_resume(
        root,dict(identity,hamiltonian='map:other'))
    receipt=json.loads((root/'moments_receipt.json').read_text())
    receipt['bank_complete']=False
    (root/'moments_receipt.json').write_text(json.dumps(receipt))
    assert not _authenticated_constructor_resume(root,identity)
