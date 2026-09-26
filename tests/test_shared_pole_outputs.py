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


def _second_model(root, meta, tables, recipe, identity, mesh):
    """A different valid model (active poles scaled) beside the fixture's: the red twin.

    Inactive columns keep the store's exact sentinel 1.0, which the writer gates.
    """
    b, packed, poles, counts = _model(meta)
    scaled = np.where(np.arange(poles.shape[1]) < counts[:, None], 1.25 * poles, poles)
    path = Path(root) / "model_other.h5"
    header = store.write_shared_pole_model(path,
        _device(packed, mesh, P(None, 'x', None, 'y')),
        _device(scaled, mesh, P()), counts, q_span=(0, 3),
        meta=meta, tables=tables, recipe=recipe, receipts=dict(identity=identity))
    return dict(path=str(path), identity=identity, digest=header['digest'])


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
    # A repeated export of THIS model (same identity and digest) is retained,
    # not rewritten: a restart must not re-export over its own export.
    config = SimpleNamespace(write_poles=True, debug=SimpleNamespace(write_w=True))
    repeated = store.export_shared_pole_outputs(handle, meta=meta, config=config,
        mesh_xy=mesh, source_wfn=wfn, run_dir=root, label="export", tables=tables, print_fn=print)
    assert {kind: row["status"] for kind, row in repeated.items()} == {"poles": "retained", "w": "retained"}
    # Red twin: the same names written from another model still refuse.
    foreign = _second_model(root, meta, tables, recipe, identity, mesh)
    assert foreign["digest"] != handle["digest"]
    with pytest.raises(ValueError, match="export already exists"):
        store.export_shared_pole_outputs(foreign, meta=meta, config=config,
            mesh_xy=mesh, source_wfn=wfn, run_dir=root, label="export", tables=tables, print_fn=print)


def test_shared_pole_outputs(tmp_path):
    check_outputs(tmp_path)


def test_w_export_copies_line_panels(tmp_path):
    """write_w exports a bank as stored: a line support's panels and counts, the dense samples, the moments."""
    from test_shared_pole_bank import _panel
    root = Path(tmp_path)
    mesh, meta, tables, recipe, identity = _bank_fixture()
    recipe.update(
        z_ry=np.asarray([0.3+0.1j, 0.1j, 0.2+0.1j], dtype=np.complex128),
        role=np.asarray([0, 1, 3], dtype=np.int8),
        distinct_id=np.asarray([0, 1, 2], dtype=np.int64),
        held=np.asarray([False, False, True], dtype=np.bool_),
        support_pair=np.asarray([[-1,-1],[-1,-1],[0,1]], dtype=np.int64),
        fit_ids=np.asarray([0, 1], dtype=np.int64),
        held_ids=np.asarray([2], dtype=np.int64))
    from gw.shared_pole_recipe import CapacityLedger
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=1 << 22)
    meta.shared_pole_capacity.reserve("fixture_live_bound", resident_bytes_per_rank=4096, workspace_bytes_per_rank=0)
    meta.shared_pole_capacity.live_stages = ("fixture_live_bound",)
    _, packed, poles, counts = _model(meta)
    source = root / "model.h5"
    header = store.write_shared_pole_model(source,
        _device(packed, mesh, P(None, 'x', None, 'y')),
        _device(poles, mesh, P()), counts, q_span=(0, 3),
        meta=meta, tables=tables, recipe=recipe, receipts=dict(identity=identity))
    bank = root / "bank.h5"
    store.initialize_shared_pole_bank(bank, meta=meta, tables=tables,
        recipe=recipe, identity=identity, mesh_xy=mesh)
    for q in range(3):
        for i in (1, 2):
            value = _matrix(meta, mesh, samples=True, value=q+i)
            store.write_shared_pole_bank(bank, q_span=(q, q+1), sample_span=(i, i+1),
                Wc=value, dWc_ds=-value, meta=meta, expected_identity=identity, mesh_xy=mesh)
        value = _matrix(meta, mesh, samples=False, value=q+3)
        store.write_shared_pole_bank(bank, q_span=(q, q+1), M1=value, M3=2*value,
            meta=meta, expected_identity=identity, mesh_xy=mesh)
    store.write_shared_pole_bank(bank, q_span=(0, 3), line=dict(
        sample=0, panels={"charge": _panel(meta, mesh, fields=5, width=2, value=9, nq=3)},
        counts={"charge": np.asarray([2, 1, 2])}), meta=meta, expected_identity=identity, mesh_xy=mesh)
    wfn = root / "WFN.h5"
    def make_wfn():
        with h5py.File(wfn, "w") as f:
            f.create_group("mf_header").create_dataset("crystal/avec", data=np.eye(3))
    rank0_transaction(wfn, stage="test.output_wfn_line", write=make_wfn)
    handle = dict(path=str(source), identity=identity, digest=header['digest'])
    outputs = store.export_shared_pole_outputs(handle, meta=meta,
        config=SimpleNamespace(write_poles=False, debug=SimpleNamespace(write_w=True)), mesh_xy=mesh,
        source_wfn=wfn, run_dir=root, label="export", tables=tables, print_fn=print)
    exported = store.validate_shared_pole_bank(outputs['w']['path'], expected_identity=identity,
        mesh_xy=mesh, require_complete=True)
    assert exported["line_panels"]["counts"]["charge"] == [[2], [1], [2]]
    assert exported["line_panels"]["width"]["charge"] == [2]
    def compare():
        with h5py.File(bank, 'r') as a, h5py.File(outputs['w']['path'], 'r') as b:
            for field in ('Wc', 'dWc_ds', 'M1', 'M3', store.line_panel_name("charge", 0)):
                assert a[field][()].tobytes() == b[field][()].tobytes(), field
    rank0_transaction(root, stage="test.output_compare_line", write=compare)


def _export_fixture(root):
    """One authenticated model/bank/WFN triple an export can be driven from."""
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
            g.create_dataset("crystal/avec", data=np.eye(3))
    rank0_transaction(wfn, stage="test.retain_wfn", write=make_wfn)
    return (mesh, meta, tables,
            dict(path=str(source), identity=identity, digest=header['digest']),
            _second_model(root, meta, tables, recipe, identity, mesh))


def check_retention(root):
    """N self-consistent maps leave O(1) exports; a one-shot export is inert.

    The model and bank stay the fixture's, so what varies map to map is only
    the export name -- which is exactly what the retention policy keys on.
    """
    root = Path(root)
    mesh, meta, tables, handle, foreign = _export_fixture(root / "sc_0000_shared_pole")
    run_dir = root
    # ``write_w`` is the DEBUG sibling (AWDEBUG): the bank dump is reached
    # through ``config.debug``, while ``write_poles`` stays top-level.
    both = SimpleNamespace(write_poles=True,
                           debug=SimpleNamespace(write_w=True))

    def export(label):
        return store.export_shared_pole_outputs(handle, meta=meta, config=both,
            mesh_xy=mesh, source_wfn=root / "sc_0000_shared_pole" / "WFN.h5",
            run_dir=run_dir, label=label, tables=tables, print_fn=print)

    def managed():
        return sorted(p.name for p in run_dir.iterdir()
                      if p.is_file() and p.name.endswith(".h5"))

    retained = []
    for i in range(4):
        outputs = export(f"sc_{i:04d}")
        assert set(outputs) == {"poles", "w"}
        # Provenance: the surviving names are the CURRENT map's, and the
        # census never grows with the number of maps already run.
        assert managed() == [f"sc_{i:04d}_poles.h5", f"sc_{i:04d}_w.h5"], i
        retained.append(len(managed()))
    assert retained == [2, 2, 2, 2]

    # Nothing outside the managed namespace is eligible, and a one-shot
    # export neither removes nor is removed.
    (run_dir / "keepme_w.h5").touch()
    (run_dir / "sc_00001_w.h5").touch()          # five digits: not managed
    from unittest.mock import patch
    with patch.object(store, "validate_shared_pole_bank",
                      wraps=store.validate_shared_pole_bank) as validations:
        export("oneshot")
        assert len(validations.call_args_list) == 1  # source bank, once
    assert managed() == ["keepme_w.h5", "oneshot_poles.h5", "oneshot_w.h5",
                         "sc_00001_w.h5", "sc_0003_poles.h5", "sc_0003_w.h5"]

    export("sc_0004")
    assert managed() == ["keepme_w.h5", "oneshot_poles.h5", "oneshot_w.h5",
                         "sc_00001_w.h5", "sc_0004_poles.h5", "sc_0004_w.h5"]

    # A repeated label from the same model is retained, and retention keeps it.
    assert {kind: row["status"] for kind, row in export("sc_0004").items()} == \
        {"poles": "retained", "w": "retained"}
    assert managed() == ["keepme_w.h5", "oneshot_poles.h5", "oneshot_w.h5",
                         "sc_00001_w.h5", "sc_0004_poles.h5", "sc_0004_w.h5"]
    # The overwrite guard is untouched for another model: it refuses before
    # anything is released.
    assert foreign["digest"] != handle["digest"]
    with pytest.raises(ValueError, match="export already exists"):
        store.export_shared_pole_outputs(foreign, meta=meta, config=both,
            mesh_xy=mesh, source_wfn=root / "sc_0000_shared_pole" / "WFN.h5",
            run_dir=run_dir, label="sc_0004", tables=tables, print_fn=print)
    assert managed() == ["keepme_w.h5", "oneshot_poles.h5", "oneshot_w.h5",
                         "sc_00001_w.h5", "sc_0004_poles.h5", "sc_0004_w.h5"]

    def receipt():
        (root / 'retention_receipt.json').write_text(json.dumps(dict(
            status='PASS', collected=1, passed=1,
            scope='5 managed maps + 1 one-shot export in one run directory; '
                  'retained file census, overwrite guard, unmanaged names',
            maps=5, retained_after_each_map=retained,
            surviving=managed(),
            job_step=os.environ.get('SLURM_JOB_ID','')+'.'
                     +os.environ.get('SLURM_STEP_ID','')), indent=2)+'\n')
    rank0_transaction(root, stage="test.retention_receipt", write=receipt)


def test_shared_pole_export_retention(tmp_path):
    check_retention(tmp_path)


if __name__ == '__main__':
    import sys
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    def main():
        assert jax.process_count() == 4
        check_outputs(sys.argv[1])
        check_retention(os.path.join(sys.argv[1], "retention"))
        return 0
    run_main_and_finalize(main)
