"""P4 constructor/store integration with an authenticated planted Coulomb resource.

The scratch and compact-model transports are the production store. The response owner reads an exact identity on the planted centroid support
through its authenticated public accessor. No campaign-bank claim is made.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import sys


def run_checks(mesh, directory):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO
    from gw.shared_pole_constructor import construct_shared_poles
    from gw.shared_pole_recipe import ROLE_CODES, RECIPE_HASH, GATE_HASH, CapacityLedger
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from test_shared_pole_store import _fixture

    meta, tables, recipe, identity = _fixture(mesh)
    # Keep the fixture's physical three-parent 3x3x3 geometry. The explicit
    # budget admits fixed native workspace for these tiny GPU matrices without
    # inflating q metadata seen by later Coulomb and store owners.
    meta.n_rmu_padded = meta.mu_basis.n_packed
    recipe.update(recipe_hash=RECIPE_HASH, gate_hash=GATE_HASH,
                  role_codes=ROLE_CODES, fit_ids=[0, 1], held_ids=[2],
                  z_ry=np.asarray([.2j, .2j, .5+.2j, .3+.2j]),
                  role=np.asarray([0, 1, 0, 3], np.int8),
                  distinct_id=np.asarray([0, 0, 1, 2], np.int64),
                  held=np.asarray([False, False, False, True]),
                  support_pair=np.asarray([[-1, -1], [-1, -1], [-1, -1], [0, 1]], np.int64),
                  census=identity, direction_cutoff=1e-4,
                  imaginary_width=2, infinity_width=1,
                  multiplet_relative_tolerance=1e-6, eta_ev=.25)
    meta.shared_pole_recipe = recipe
    meta.shared_pole_capacity = CapacityLedger(
        meta, mesh_xy=mesh, device_budget_bytes=1 << 30)
    meta.shared_pole_capacity.reserve('fixture_bank_inputs',
        resident_bytes_per_rank=4096, workspace_bytes_per_rank=0)
    meta.shared_pole_capacity.live_stages = ('fixture_bank_inputs',)
    path = directory / "bank.h5"
    resident = store.ResidentBankPayload(mesh, carrier=meta.mu_basis.n_canonical,
                                         label="p4 resident twin")
    for handle in (path, resident):
        store.initialize_shared_pole_bank(handle, meta=meta, tables=tables,
                                         recipe=recipe, identity=identity, mesh_xy=mesh)

    def packed(host, spec):
        host = meta.mu_basis.pack_host(host, axis=-2)
        host = meta.mu_basis.pack_host(host, axis=-1)
        return jax.make_array_from_callback(host.shape, NamedSharding(mesh, spec),
                                            lambda index: host[index])

    rng = np.random.default_rng(32009)
    expected = []
    for q, rank in enumerate((3, 5, 4)):
        c = .02*(rng.normal(size=(7, rank)) + 1j*rng.normal(size=(7, rank)))
        t = np.linspace(.2, 1.4, rank)
        expected.append((c, t))
        for handle in (path, resident):
            # Samples 0 (imaginary axis) and 2 (held) are dense; the line sample 1 is planted below.
            for sample, z in ((0, .2j), (2, .3+.2j)):
                w = (c/(z*z-t)) @ c.conj().T
                dw = -(c/(z*z-t)**2) @ c.conj().T
                store.write_shared_pole_bank(handle, q_span=(q, q+1), sample_span=(sample, sample+1),
                    Wc=packed(w[None, None], P(None, None, 'x', 'y')),
                    dWc_ds=packed(dw[None, None], P(None, None, 'x', 'y')),
                    meta=meta, expected_identity=identity, mesh_xy=mesh)
            store.write_shared_pole_bank(handle, q_span=(q, q+1),
                M1=packed((c @ c.conj().T/2)[None], P(None, 'x', 'y')),
                M3=packed(((c*t) @ c.conj().T/2)[None], P(None, 'x', 'y')),
                meta=meta, expected_identity=identity, mesh_xy=mesh)

    def value(z):
        w = np.stack([(c/(z*z-t)) @ c.conj().T for c, t in expected])
        dw = np.stack([-(c/(z*z-t)**2) @ c.conj().T for c, t in expected])
        return packed(w, P(None, 'x', 'y')), packed(dw, P(None, 'x', 'y'))
    from shared_pole_bank_plant import plant_line_samples
    for handle in (path, resident):
        plant_line_samples(handle, value, meta=meta, recipe=recipe, identity=identity, mesh=mesh,
                           nq=3, ordered=False)

    import hashlib
    vpath = directory / "coulomb.h5"
    ncan = meta.mu_basis.n_canonical
    value = np.zeros((3, ncan, ncan), np.complex128)
    value[:, :7, :7] = np.eye(7)
    v = jax.make_array_from_callback(value.shape, NamedSharding(mesh, P(None, 'x', 'y')),
                                      lambda index: value[index])
    with SlabIO(vpath, mode='w', mesh=mesh) as io:
        io.create_dataset('V', shape=(3, 7, 7), dtype=np.complex128)
        io.write_slab('V', v)
    del v
    coulomb = dict(path=str(vpath), dataset='V',
                   sha256=hashlib.sha256(vpath.read_bytes()).hexdigest(),
                   q_irr_full_idx=tables['q_irr_full_idx'].tolist(), basis='canonical')
    meta.shared_pole_capacity.live_stages = ()
    rows = []
    bank = dict(path=path, identity=identity, tables=tables, coulomb=coulomb)
    import contextlib
    from unittest import mock
    import gw.shared_pole_execution as execution_module
    original_route = execution_module.constructor_execution

    def forced_face(*args, **kwargs):
        mode, receipt = original_route(*args, **kwargs)
        return 'face', dict(receipt, reason='test: forced whole-mesh parents', admitted_mode=mode)

    arms = {
        'local': ('local', path, ()),
        # A deck that names the distributed service still runs local parents
        # when they fit (the one-parent-per-round face loop is not forced).
        'distributed': ('distributed', path, ()),
        'face_batched': ('distributed', path,
                         (mock.patch.object(execution_module, 'constructor_execution', forced_face),)),
        'face_single': ('distributed', path,
                        (mock.patch.object(execution_module, 'constructor_execution', forced_face),
                         mock.patch.object(execution_module, 'face_batch_width',
                                           lambda *a, **k: (1, dict(parent_batch=1, reason='test'))))),
        'resident_local': ('local', resident, ()),
    }
    models = {}
    for arm, (layout, source, patches) in arms.items():
        output = directory / f"model_{arm}.h5"
        config = SimpleNamespace(backend=SimpleNamespace(linalg=layout))
        arm_bank = dict(bank, path=source)
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            result = construct_shared_poles(arm_bank, arm_bank, meta, config,
                                            mesh_xy=mesh, output=output)
        header = store.validate_shared_pole_model(output, expected_identity=identity,
            mesh_xy=mesh, capacity=meta.shared_pole_capacity)
        assert header['K'] == [3, 5, 4], header['K']
        errors = []
        with SlabIO(output, mode='r', mesh=mesh) as io:
            for q, (c, t) in enumerate(expected):
                cc, _, pp, _ = store.read_shared_pole_faces(
                    io, (q, q+1), meta=meta, header=header)
                models.setdefault(arm, []).append(
                    (np.asarray(pp)[0, :len(t)].copy(),
                     float(jnp.linalg.norm(jnp.sum(abs(cc)**2, axis=-1)))))
                np.testing.assert_allclose(np.asarray(pp)[0, :len(t)], t, rtol=0, atol=1e-10)
                diagonal = meta.mu_basis.pack_host(np.sum(abs(c)**2, axis=-1), axis=0)[None, :, None]
                target = jax.make_array_from_callback(diagonal.shape,
                    NamedSharding(mesh, P(None, 'x', None)), lambda index: diagonal[index])
                relative = float(jnp.linalg.norm(jnp.sum(abs(cc)**2, axis=-1)-target)/jnp.linalg.norm(target))
                assert relative < 1e-10, relative
                del cc, pp, target
                errors.append(max(result['q_receipts'][q]['constructor']['retained_moment_relative']['M1'] +
                                  result['q_receipts'][q]['constructor']['retained_moment_relative']['M3']))
        assert max(errors) < 1e-10
        assert all(row['status'] != 'FAIL' for receipt in result['q_receipts'] for row in receipt['gates'])
        execution = result['execution']['mode']
        want = 'face' if arm.startswith('face') else 'local'
        assert execution == want, (arm, execution)
        assert result['execution']['requested_layout'] == layout
        if arm == 'face_batched':
            assert result['execution']['face_batch']['parent_batch'] == 3, result['execution']['face_batch']
        expected_ops = {'eigh', 'gemm'} if execution == 'face' else {'eigh'}
        for receipt in result['q_receipts']:
            queries = receipt['constructor']['native_workspace_queries']
            assert {row['op'] for row in queries} == expected_ops
            assert receipt['constructor']['capacity']['execution'] == execution
            # The query list is cumulative across selection, reduction and
            # model phases. ``native_workspace`` is the current phase's live
            # workspace: earlier solve/product scratch is transient and has
            # already been released.
            workspace = receipt['constructor']['capacity']['native_workspace']
            if execution == 'face':
                # The model phase prices GEMM at max(n, round pencil side);
                # a batched face round's side can exceed n.
                current_gemm = [row['bytes_per_rank'] for row in queries
                    if row['op'] == 'gemm'
                    and row['shapes'][0][-1] >= meta.n_rmu_padded]
                assert workspace['gemm'] in current_gemm
            else:
                assert 'gemm' not in workspace
            assert receipt['constructor']['capacity']['price']['phase'] == 'model'
            current_eigh = [row['bytes_per_rank'] for row in queries
                            if row['op'] == 'eigh' and row['shapes'][0][-1] == meta.n_rmu_padded]
            assert workspace['eigh'] == current_eigh[0]
            assert workspace['eigh'] > 0
            assert receipt['constructor']['capacity']['workspace_bytes_per_rank'] == sum(workspace.values())
        if patches:
            rows.append(dict(name='constructor_store_' + arm, layout=layout, status='PASS',
                             K=header['K'], retained_moment_max=max(errors)))
            continue
        # A changed resolved coordinate must refuse before a model write.
        old_z = recipe['z_ry'].copy()
        recipe['z_ry'][2] += .01
        refused = False
        try:
            construct_shared_poles(arm_bank, arm_bank, meta, config,
                                   mesh_xy=mesh, output=directory/f"stale_{arm}.h5")
        except ValueError as error:
            refused = 'changed z' in str(error)
        finally:
            recipe['z_ry'][:] = old_z
        assert refused and not (directory/f"stale_{arm}.h5").exists()
        rows.append(dict(name='constructor_store_ragged_stale', layout=layout, arm=arm, status='PASS',
                         K=header['K'], retained_moment_max=max(errors), constructor=result))
    # Parity. Local parents are one program whatever the requested layout and
    # whatever the bank's residence: bit-identical poles and factor norms.
    # A batched face round and one-parent face rounds agree to round-off.
    parity = {}
    for arm in ('distributed', 'resident_local', 'face_batched'):
        reference = 'face_single' if arm == 'face_batched' else 'local'
        diff = max(max(float(np.max(np.abs(a[0] - b[0]))), abs(a[1] - b[1]) / b[1])
                   for a, b in zip(models[arm], models[reference]))
        bitwise = all(np.array_equal(a[0], b[0]) and a[1] == b[1]
                      for a, b in zip(models[arm], models[reference]))
        parity[arm] = dict(reference=reference, max_rel_or_abs=diff, bitwise=bitwise)
        assert (bitwise if reference == 'local' else diff <= 1e-10), (arm, parity[arm])
    rows.append(dict(name='constructor_route_parity', status='PASS', parity=parity))
    resident.release()
    for layout in ('local', 'distributed'):
        # Independent red map: upstream already owns the full 3U budget.
        # Admission must fail before opening any matrix payload or output.
        ledger = CapacityLedger(meta, mesh_xy=mesh)
        meta.shared_pole_capacity = ledger
        ledger.reserve('planted_upstream',
                       resident_bytes_per_rank=int(ledger.limit_bytes_per_rank),
                       workspace_bytes_per_rank=0)
        ledger.live_stages = ('planted_upstream',)
        config = SimpleNamespace(backend=SimpleNamespace(linalg=layout))
        output = directory/f"capacity_refused_{layout}.h5"
        refused = False
        try:
            construct_shared_poles(bank, {"path": path}, meta, config,
                                   mesh_xy=mesh, output=output)
        except MemoryError as error:
            refused = 'shared_pole_capacity' in str(error)
        assert refused and not output.exists()
        assert ledger.receipt()['entries'][-1]['status'] == 'FAIL'
        rows.append(dict(name='constructor_capacity_refusal', layout=layout, status='PASS',
                         capacity=ledger.receipt()))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from runtime import initialize_communicator_stack, finalize_process
    initialize_communicator_stack()
    import jax
    from common.collectives import resolve_mesh
    mesh = resolve_mesh()
    rows = run_checks(mesh, args.output.parent)
    if jax.process_index() == 0:
        args.output.write_text(json.dumps(dict(status='PASS', checks=rows,
            expected_checks=8, jobid=os.environ['SLURM_JOB_ID'], stepid=os.environ['SLURM_STEP_ID'],
            mesh={axis: int(mesh.shape[axis]) for axis in ('x', 'y')},
            scope='square-mesh constructor and actual scratch/model store plus authenticated response Coulomb accessor; planted positive measure and native workspace queries; device peaks NOT_MEASURED'),
            indent=2, allow_nan=False)+'\n')
    finalize_process()


if __name__ == '__main__':
    main()
