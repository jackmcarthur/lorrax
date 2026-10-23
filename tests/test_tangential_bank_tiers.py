"""Exact public bank/round-reader contract across named and rank-file tiers.

Synthetic identity-grid metadata has 59 parents, 40 samples, and 32 ragged
ordered line panels. Only Gamma and five other parents are written; global
completion stays false. This is storage transport, not a physical Fe model.
"""


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import runtime
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--existing-bank', type=Path)
    options = parser.parse_args()
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.centroid_basis import PackedCentroidBasis
    from common.collectives import device_put_process_local, gather_to_host
    from common.gpu_utils import device_budget_bytes
    from common.grouped_layout import build_square_grouped_shard_layout
    from file_io.tangential_store import (ResidentBankPayload, RoundReader,
        initialize_tangential_bank, tangential_bank_writer,
        validate_tangential_bank, read_round_samples)
    from gw.tangential_recipe import CapacityLedger, ROLE_CODES
    from gw.tangential_screening import _tangential_tables
    from runtime.padding import padded_mu_extent
    from symmetry_maps import SymMaps

    assert mesh.size == 4
    options.out.mkdir(parents=True, exist_ok=True)
    grid, nq, logical = (59, 1, 1), 59, 13
    k = np.zeros((nq, 3)); k[:, 0] = np.arange(nq) / nq
    wfn = SimpleNamespace(trs_holds=False, ntran=1, kgrid=np.asarray(grid),
        nkpts=nq, kpoints=k, sym_matrices=np.eye(3, dtype=np.int64)[None],
        shift=np.zeros(3))
    sym = SymMaps(wfn)
    indices = np.zeros((logical, 3), np.int32); indices[:, 0] = np.arange(logical)
    layout = build_square_grouped_shard_layout([0] * 9 + [1] * 2 + [2] * 2, (2, 2))
    basis = PackedCentroidBasis(mesh, layout, indices, int(padded_mu_extent(logical, mesh)))
    assert not basis.is_identity and np.any(~basis.active_mask)
    meta = SimpleNamespace(nkx=59, nky=1, nkz=1, nk_tot=nq, nspinor=2,
        nspinor_wfnfile=2, nspin=1, n_rmu=logical, n_rmu_padded=basis.n_packed,
        mu_basis=basis, fft_grid=(16, 1, 1), cell_volume=64.)
    ledger = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=int(device_budget_bytes()))
    ledger.live_stages = (); meta.tangential_capacity = ledger
    tables = _tangential_tables(meta, sym, indices)
    assert len(tables['q_irr_full_idx']) == nq
    z = np.asarray([.1 + .02 * s + .2j for s in range(32)]
                   + [.5j, 1j, 2j, 3j, .35 + .2j, .55 + .2j, .7j, 1.4j], complex)
    role = np.asarray([ROLE_CODES['line']] * 32 + [ROLE_CODES['imaginary']] * 4
        + [ROLE_CODES['held_line']] * 2 + [ROLE_CODES['held_imaginary']] * 2, np.int8)
    recipe = dict(version='synthetic-storage-contract-v1', role_codes=dict(ROLE_CODES),
        z_ry=z, role=role, distinct_id=np.arange(40, dtype=np.int64),
        held=np.arange(40) >= 36, support_pair=np.zeros((40, 2), np.int64),
        fit_ids=np.arange(36, dtype=np.int64), held_ids=np.arange(36, 40, dtype=np.int64))
    identity = {name: hashlib.sha256(('synthetic-storage-' + name).encode()).hexdigest()
                for name in ('iteration_id', 'hamiltonian', 'energies', 'occupations',
                             'wavefunctions', 'centroids')}
    square_names = ('Wc', 'dWc_ds', 'M1', 'M3', 'M0', 'M2')
    face4 = NamedSharding(mesh, P(None, None, 'x', 'y'))
    face3 = NamedSharding(mesh, P(None, 'x', 'y'))

    def code(field, parents, samples, rows, cols):
        q = np.asarray(parents)[:, None, None, None]
        s = np.asarray(samples)[None, :, None, None]
        r = np.asarray(rows)[None, None, :, None]
        c = np.asarray(cols)[None, None, None, :]
        epoch = np.where(q == 0, 0, np.where(q == 58, 2, 1))
        return (field * 2**20 + epoch * 2**16 + q * 1024 + s * 16 + r / 4 + c / 128
                + 1j * (-field * 2**18 - epoch * 2**14 + q * 128 - s * 2 + r / 32 - c / 256))

    def square(name, parents, samples=None):
        value = code(square_names.index(name) + 1, parents,
                     [0] if samples is None else samples, range(logical), range(logical))
        value = basis.pack_host(basis.pack_host(value, axis=-2), axis=-1)
        return value[:, 0] if samples is None else value

    def line(sample, parents):
        width = 2 * (sample % 4 + 1)
        counts = (np.asarray(parents) + 2 * sample) % (width + 1)
        value = code(10 + sample, parents, range(9), range(logical), range(width))
        value *= (np.arange(width)[None, None, None, :] < counts[:, None, None, None])
        return basis.pack_host(value, axis=2), counts.astype(np.int64)

    named = options.out / 'named_bank.h5'
    rankfile = ResidentBankPayload(mesh, carrier=basis.n_canonical,
        label=options.out / 'rank_bank.h5', memory_kind='file', root=str(options.out))
    results, held = [], []

    def write_parents(path, lo, hi):
        parents = list(range(lo, hi))
        with tangential_bank_writer(path, meta=meta, expected_identity=identity, mesh_xy=mesh) as (_, _, write):
            write(q_span=(lo, hi), sample_span=(32, 40), **{name:
                device_put_process_local(square(name, parents, range(32, 40)), face4)
                for name in square_names[:2]})
            write(q_span=(lo, hi), **{name:
                device_put_process_local(square(name, parents), face3) for name in square_names[2:]})
            for sample in range(32):
                value, counts = line(sample, parents)
                write(q_span=(lo, hi), line=dict(sample=sample,
                    panels={'charge': device_put_process_local(value, face4)}, counts={'charge': counts}))

    def check(path, header, parents, batch=False, samples=(32, 40)):
        print('BANK_CASE START', str(path), parents, batch, samples, flush=True)
        with RoundReader(path, mesh_xy=mesh) as reader:
            with reader.phase() as io:
                dense, panels = read_round_samples(io, meta=meta, header=header,
                    fields=square_names[:2], sample_span=samples, q_ids=parents,
                    partition_spec=P(('x', 'y'), None, None, None) if batch else None,
                    line_family='charge', line_samples=(0, 15, 31))
            with reader.phase() as io:
                moments, _ = read_round_samples(io, meta=meta, header=header,
                    fields=square_names[2:], q_ids=parents,
                    partition_spec=P(('x', 'y'), None, None) if batch else None)
        for name, value in {**dense, **moments}.items():
            expected = square(name, parents, range(*samples) if name in square_names[:2] else None)
            np.testing.assert_array_equal(np.asarray(gather_to_host(value)), expected)
        for sample, (value, counts) in panels.items():
            expected, expected_counts = line(sample, parents)
            np.testing.assert_array_equal(np.asarray(gather_to_host(value)), expected)
            np.testing.assert_array_equal(counts, expected_counts)
        held.append((parents, samples, dense, moments, panels))
        print('BANK_CASE PASS', parents, batch, samples, flush=True)

    if options.existing_bank is not None:
        header = validate_tangential_bank(options.existing_bank,
            expected_identity=identity, mesh_xy=mesh)
        check(options.existing_bank, header, [58, 0, 3, 58, 2, 4, 1, 0], batch=True)
        result = dict(verdict='PASS', scope='existing named bank sparse/repeated parent regression', all_exact=True)
        (options.out / f'rank{jax.process_index()}.json').write_text(json.dumps(result, indent=2) + '\n')
        print('BANK_SPARSE PASS', jax.process_index(), flush=True)
        return

    try:
        for path in (named, rankfile):
            header = initialize_tangential_bank(path, meta=meta, tables=tables,
                recipe=recipe, identity=identity, mesh_xy=mesh)
            assert header['bank_shape'] == dict(nq=59, nsample=40, d=logical)
            write_parents(path, 0, 1)
            header = validate_tangential_bank(path, expected_identity=identity, mesh_xy=mesh)
            assert not header['complete'] and np.asarray(header['sample_written'])[0].all()
            assert not np.asarray(header['sample_written'])[1:].any()
            refused = False
            try:
                validate_tangential_bank(path, expected_identity=identity, mesh_xy=mesh, require_complete=True)
            except (ValueError, RuntimeError):
                refused = True
            assert refused
            refused = False
            try:
                with RoundReader(path, mesh_xy=mesh) as reader:
                    with reader.phase() as io:
                        read_round_samples(io, meta=meta, header=header, fields=('M1',), q_ids=[1])
            except (ValueError, RuntimeError):
                refused = True
            assert refused
            check(path, header, [0]); check(path, header, [0] * 4, batch=True)
            # Caller bindings are reused; queued staging and prior read outputs must stay valid.
            write_parents(path, 1, 5); write_parents(path, 58, 59)
            header = validate_tangential_bank(path, expected_identity=identity, mesh_xy=mesh)
            assert not header['complete']
            for parents, batch in (([0, 1, 2, 3, 4], False), ([0, 1, 2, 3], True),
                    ([3, 0, 2, 1], True), ([0, 1, 2, 3, 4, 4, 4, 4], True),
                    ([58, 0, 3, 58, 2, 4, 1, 0], True)):
                check(path, header, parents, batch=batch)
            check(path, header, [0, 1, 2, 3], batch=True, samples=(34, 38))
            results.append(dict(tier='named_hdf' if path is named else 'rank_file',
                full_parent_metadata=59, committed_parents=[0, 1, 2, 3, 4, 58], global_complete=False))
        # Recheck arrays held while subsequent reads reused every tier's staging.
        for parents, samples, dense, moments, panels in held:
            for name, value in {**dense, **moments}.items():
                np.testing.assert_array_equal(np.asarray(gather_to_host(value)),
                    square(name, parents, range(*samples) if name in square_names[:2] else None))
            for sample, (value, counts) in panels.items():
                expected, expected_counts = line(sample, parents)
                np.testing.assert_array_equal(np.asarray(gather_to_host(value)), expected)
                np.testing.assert_array_equal(counts, expected_counts)
        assert rankfile.fits and not rankfile._late_host
        stores = {name: store.receipt() for name, store in rankfile._fields.items()}
        assert all(record['tier'] == 'file' for record in stores.values())
        result = dict(verdict='PASS', cases=results, all_exact=True, held_rounds=len(held),
            logical_mu=logical, canonical_mu=basis.n_canonical, packed_mu=basis.n_packed,
            packed_active_mask=basis.active_mask.tolist(), rank_file_stores=stores)
    finally:
        rankfile.release()
    (options.out / f'rank{jax.process_index()}.json').write_text(json.dumps(result, indent=2) + '\n')
    print('BANK_TIERS PASS', jax.process_index(), flush=True)


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
