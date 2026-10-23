"""Compute-node CLI selection gate at the logical FFT-grid capacity."""


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from unittest.mock import patch

    import runtime
    stack = runtime.initialize_communicator_stack()
    import h5py
    import jax
    import numpy as np
    from centroid import kmeans_cli as cli
    from common.collectives import barrier

    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--baseline', action='store_true')
    opts = parser.parse_args()
    work = Path(opts.workdir)
    work.mkdir(parents=True, exist_ok=True)
    rank = jax.process_index()
    stream = (work / f'rank{rank}.jsonl').open('w', buffering=1)
    template = Path(__file__).parent / 'hsuite/fixture_na/WFN.h5'

    for grid in ((2, 2, 2), (3, 3, 1)):
        path = work / ('WFN_' + '_'.join(map(str, grid)) + '.h5')
        if rank == 0:
            # One normalized Gamma plane wave; this tests CLI capacity and
            # sharding, not the accuracy of a material's electronic structure.
            with h5py.File(template, 'r') as source, h5py.File(path, 'w') as out:
                source.copy('mf_header', out)
                def replace(name, value):
                    if name in out:
                        del out[name]
                    out.create_dataset(name, data=value)
                replace('mf_header/gspace/FFTgrid', np.array(grid, np.int32))
                gvec = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.int32)
                replace('mf_header/gspace/components', gvec)
                replace('mf_header/gspace/ng', np.int32(3))
                for name, value in dict(
                        nrk=np.int32(1), mnband=np.int32(1), nspin=np.int32(1),
                        nspinor=np.int32(1), ngk=np.full(1, 3, np.int32),
                        ngkmax=np.int32(3), kgrid=np.ones(3, np.int32),
                        shift=np.zeros(3), w=np.ones(1), rk=np.zeros((1, 3)),
                        ifmin=np.ones((1, 1), np.int32),
                        ifmax=np.ones((1, 1), np.int32),
                        el=np.zeros((1, 1, 1)), occ=np.ones((1, 1, 1))).items():
                    replace('mf_header/kpoints/' + name, value)
                replace('mf_header/symmetry/ntran', np.int32(1))
                replace('mf_header/symmetry/mtrx', np.eye(3, dtype=np.int32)[None])
                replace('mf_header/symmetry/tnp', np.zeros((1, 3)))
                out.create_dataset('wfns/gvecs', data=gvec)
                coeff = np.zeros((1, 1, 3, 2)); coeff[0, 0, 0, 0] = 1.
                out.create_dataset('wfns/coeffs', data=coeff)
        barrier('capacity-fixture-' + str(grid))

    cases = [
        ('identity_full', (2, 2, 2), 7, 1.5, False, False, 8),
        ('literal_padded', (3, 3, 1), 8, 1.5, True, False, 9),
        ('inversion_capped', (3, 3, 1), 8, 3., False, True, 9),
        ('inversion_unchanged', (3, 3, 1), 8, 1.5, False, True, 6),
        ('literal_unchanged', (3, 3, 1), 3, 1.5, True, False, 5),
    ]
    original = cli.weighted_kmeans_jax
    for label, grid, count, over, literal, inversion, expected in cases:
        if opts.baseline and label not in (
                'identity_full', 'inversion_unchanged', 'literal_unchanged'):
            continue
        args = cli.build_parser().parse_args(
            [str(count), '--oversample', str(over), '--fit-window', '0:1,0:1']
            + (['--no-symmetries'] if literal else []))
        args.sigma_ncond = None
        eye = np.eye(3, dtype=np.int32)
        operations = np.stack((eye, -eye)) if inversion else eye[None]
        symmetry = ((operations, operations, np.zeros((2, 3)), 2, True)
                    if inversion else (None, None, None, 1, False))
        calls = []
        def capture(*operands, **kwargs):
            calls.append(int(kwargs['N_c']))
            return original(*operands, **kwargs)
        path = work / ('WFN_' + '_'.join(map(str, grid)) + '.h5')
        barrier('capacity-case-' + label)
        try:
            with patch.object(cli, '_resolve_symmetry', return_value=symmetry), \
                 patch.object(cli, 'weighted_kmeans_jax', side_effect=capture):
                result = cli.select_points(args, path)
        except ValueError as exc:
            if not (opts.baseline and label == 'identity_full'):
                raise
            assert calls == [11] and 'larger sample' in str(exc)
            stream.write(json.dumps(dict(status='PASS', kind='baseline_failure',
                rank=rank, case=label, target=calls[0], error=str(exc))) + '\n')
            continue
        assert not (opts.baseline and label == 'identity_full'), 'baseline must reproduce the failure'
        assert calls == [expected], (label, calls, expected)
        indices = np.asarray(result.indices)
        flat = np.ravel_multi_index(indices.T, grid)
        assert 0 < len(flat) <= count and len(np.unique(flat)) == len(flat)
        assert np.all((indices >= 0) & (indices < np.asarray(grid)))
        if inversion:
            mapped = np.ravel_multi_index(((-indices) % grid).T, grid)
            np.testing.assert_array_equal(np.sort(mapped), np.sort(flat))
        else:
            assert len(flat) == count
        stream.write(json.dumps(dict(status='PASS', rank=rank, case=label,
            target=calls[0], grid=list(grid), count=len(flat),
            sha256=hashlib.sha256(np.ascontiguousarray(indices)).hexdigest())) + '\n')
        result.wfn.close()
    stream.close()
    barrier('capacity-cli-done')
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run this module directly under the configured MPI launcher')


if __name__ == '__main__':
    main()
