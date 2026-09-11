"""Standalone output parity, including padded centroids and pole columns at P4."""
from pathlib import Path
from types import SimpleNamespace
import json
import os

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from common.collectives import rank0_transaction
from file_io import shared_pole_store as store
from test_shared_pole_store import _device, _model
from test_shared_pole_bank import _bank_fixture, _matrix


def check_outputs(root):
    """Compare every canonical byte and header through the production writers."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    mesh, meta, tables, recipe, identity = _bank_fixture()
    b, packed, poles, counts = _model(meta)
    source = root / "model.h5"
    header = store.write_shared_pole_model(source,
        _device(packed, mesh, P(None, 'x', None, 'y')),
        _device(poles, mesh, P()), counts, q_span=(0, 3),
        meta=meta, tables=tables, recipe=recipe, receipts=dict(identity=identity))
    bank = root / "bank.h5"
    store.initialize_shared_pole_bank(bank, meta=meta, tables=tables,
        recipe=recipe, identity=identity, mesh_xy=mesh)
    for q in range(3):
        for i in range(2):
            value = _matrix(meta, mesh, samples=True, value=q+i)
            store.write_shared_pole_bank(bank, q_span=(q,q+1), sample_span=(i,i+1),
                Wc=value, dWc_ds=-value, meta=meta, expected_identity=identity, mesh_xy=mesh)
        value = _matrix(meta, mesh, samples=False, value=q+3)
        store.write_shared_pole_bank(bank, q_span=(q,q+1), M1=value, M3=2*value,
            meta=meta, expected_identity=identity, mesh_xy=mesh)
    del value, b, packed, poles
    wfn = root / "WFN.h5"
    def make_wfn():
        with h5py.File(wfn, "w") as f:
            g = f.create_group("mf_header")
            g.attrs['provenance'] = 'planted header attribute'
            g.create_dataset("crystal/avec", data=np.eye(3))
    rank0_transaction(wfn, stage="test.output_wfn", write=make_wfn)
    handle = dict(path=str(source), identity=identity, digest=header['digest'])
    from unittest.mock import patch
    with patch.object(store, "SlabIO", wraps=store.SlabIO) as opens, \
            patch.object(store, "validate_shared_pole_bank",
                         wraps=store.validate_shared_pole_bank) as validations:
        outputs = store.export_shared_pole_outputs(handle, meta=meta,
            config=SimpleNamespace(write_poles=True,
                debug=SimpleNamespace(write_w=True)), mesh_xy=mesh,
            source_wfn=wfn, run_dir=root, label="export", tables=tables, print_fn=print)
        destination = str(root / "export_w.h5")
        writer_opens = [call for call in opens.call_args_list
                        if str(call.args[0]) == destination]
        assert [call.kwargs["mode"] for call in writer_opens] == ["w", "a"]
        assert len(validations.call_args_list) == 1  # source bank, once
    store.validate_shared_pole_model(outputs['poles']['path'], expected_identity=identity,
        mesh_xy=mesh, capacity=meta.shared_pole_capacity)
    store.validate_shared_pole_bank(outputs['w']['path'], expected_identity=identity,
        mesh_xy=mesh, require_complete=True)
    # These payloads are tiny planted oracles, not production matrix gathers.
    def compare():
        for kind, original, fields in (
            ('poles', source, ('factor','poles2_ry2','K')),
            ('w', bank, ('Wc','dWc_ds','M1','M3','z_ry','distinct_id'))):
            with h5py.File(original, 'r') as a, h5py.File(outputs[kind]['path'], 'r') as b:
                for field in fields:
                    assert a[field][()].tobytes() == b[field][()].tobytes(), field
                assert b['mf_header'].attrs['provenance'] == 'planted header attribute'
                np.testing.assert_array_equal(b[kind+'_header/centroids/r_mu_fft_idx'][()],
                                              meta.mu_basis.canonical_indices)
                if kind == 'poles':
                    assert b['b'].id == b['factor'].id
        result = dict(status='PASS', collected=1, passed=1,
            scope='P4 complete bank and pole exports, odd centroids and padded columns',
            job_step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),
            outputs=outputs)
        (root/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    rank0_transaction(root, stage="test.output_compare", write=compare)
    with pytest.raises(ValueError, match="export already exists"):
        store.export_shared_pole_outputs(handle, meta=meta,
            config=SimpleNamespace(write_poles=True,
                debug=SimpleNamespace(write_w=True)), mesh_xy=mesh,
            source_wfn=wfn, run_dir=root, label="export", tables=tables, print_fn=print)


def test_shared_pole_outputs(tmp_path):
    check_outputs(tmp_path)


if __name__ == '__main__':
    import sys
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    def main():
        assert jax.process_count() == 4
        check_outputs(sys.argv[1])
        return 0
    run_main_and_finalize(main)
