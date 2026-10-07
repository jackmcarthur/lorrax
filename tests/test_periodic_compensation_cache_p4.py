"""Collective periodic-cache and two-face moment-action physics oracle."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
from types import SimpleNamespace


def run_oracle(runtime, directory):
    import h5py
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import barrier, agree_io_error, device_put_process_local, gather_to_host
    from file_io.commit_state import COMMIT_STATE
    from isdf.atomic_coulomb import make_periodic_compensation_action
    from isdf.coulomb_fourier_cache import (write_periodic_compensation_cache,
        load_periodic_compensation_cache, _file_digest)
    from runtime.padding import padded_axis, PaddedAxis

    mesh = runtime.mesh
    if mesh.size != 4 or jax.process_count() != 4:
        raise ValueError('Requires real P4 collective cache and GEMM fixture')
    directory = Path(directory)
    error = None
    if jax.process_index() == 0:
        try:
            directory.mkdir()
        except Exception as exc:
            error = exc
    agree_io_error(error, path=directory, stage='periodic fixture fresh directory')
    barrier('periodic_fixture_directory')
    nq, nmom, nmu = 3, 9, 6
    lm = np.asarray([(l, m) for l in range(3) for m in range(-l, l+1)])
    geometry = dict(reciprocal_rows_bohr_inverse=np.eye(3).tolist(),
        cell_volume_bohr3=(2*np.pi)**3, atom_centres_bohr=[[.2, -.1, .3]],
        operator_q_fractional=[[0., 0., 0.], [.5, 0., 0.], [.5, .5, 0.]],
        support_radius_bohr=1.8)
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    moment = padded_axis(nmom, mesh, name='periodic moment', specs=((face.spec, 1), (face.spec, 2)))
    mu = padded_axis(nmu, mesh, name='fixture centroid')
    assert moment.carrier > nmom and mu.carrier > nmu
    # A planted view of the existing packed-axis contract: ghost inside the
    # carrier, not at its suffix. No geometric action is fabricated here.
    active = np.ones(mu.carrier, bool)
    active[3] = False
    active[6] = False
    basis = SimpleNamespace(n_logical=nmu, n_packed=mu.carrier, active_mask=active,
        solve_axis=PaddedAxis('fixture packed solve', mu.carrier, mu.carrier, mu.divisor))
    rng = np.random.default_rng(730904)
    random = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
    x = random((nq, nmom, nmom))*.15
    gram = x.conj().swapaxes(-2, -1) @ x
    rows = random((nq, nmu, nmom))*.3
    padded_gram = np.full((nq, moment.carrier, moment.carrier), np.nan+1j*np.nan)
    padded_gram[:, :nmom, :nmom] = gram
    put = lambda values: device_put_process_local(np.asarray(values, np.complex128), face)
    host = lambda values: np.asarray(gather_to_host(values))
    upstream = directory/'preparation.json'
    if jax.process_index() == 0:
        upstream.write_text(json.dumps(dict(schema='lorrax.test.planted_periodic_preparation.v1',
            scope='Planted finite complex multipole metric; no actual Fourier/tail accuracy claim.',
            shape=list(gram.shape)), sort_keys=True)+'\n')
    barrier('periodic_fixture_upstream')
    preparation = dict(receipt_path=str(upstream), receipt_sha256=_file_digest(upstream),
        payload_sha256=hashlib.sha256(np.ascontiguousarray(gram).tobytes()).hexdigest(),
        producer_sources_sha256={'planted_fixture':_file_digest(__file__)},
        cutoffs=[12., 16.], refinement_max=np.full((nq, 1), 1e-5).tolist())
    cache_path = directory/'periodic.h5'
    written = write_periodic_compensation_cache(cache_path, put(padded_gram),
        mesh=mesh, geometry=geometry, lm=lm, preparation=preparation)
    load = lambda path=cache_path, digest=written['file_sha256'], geo=geometry, channels=lm: (
        load_periodic_compensation_cache(path, mesh=mesh, expected_file_sha256=digest,
            geometry=geo, lm=channels))
    cache = load()
    loaded = host(cache['gram'])
    assert np.array_equal(loaded[:, :nmom, :nmom], gram)
    assert np.count_nonzero(loaded[:, nmom:]) == np.count_nonzero(loaded[:, :, nmom:]) == 0
    action, ledger = make_periodic_compensation_action(mesh, cache,
        centroid_basis=basis, fft_points=125)
    padded_rows = np.zeros((nq, mu.carrier, moment.carrier), np.complex128)
    padded_rows[:, active, :nmom] = rows
    compiled = action.lower(put(padded_rows), cache['gram']).compile()
    memory = compiled.memory_analysis()
    if memory is None:
        raise ValueError('periodic action executable has no memory ledger')
    hlo = compiled.as_text()
    hlo_path = directory/f'action.rank{jax.process_index()}.hlo.txt'
    hlo_path.write_text(hlo)
    actual = host(compiled(put(padded_rows), cache['gram']))
    scale = (125/geometry['cell_volume_bohr3'])**2
    expected = scale*(rows.conj() @ gram @ rows.swapaxes(-2, -1))
    physical = lambda value:value[:, active][:, :, active]
    error = float(np.max(np.abs(physical(actual)-expected)))
    assert np.isfinite(actual).all() and error < 3e-12
    assert np.count_nonzero(actual[:, ~active]) == np.count_nonzero(actual[:, :, ~active]) == 0
    canonical_active = np.arange(mu.carrier) < nmu
    canonical_basis = SimpleNamespace(n_logical=nmu, n_packed=mu.carrier,
        active_mask=canonical_active, solve_axis=mu)
    canonical_action, _ = make_periodic_compensation_action(mesh, cache,
        centroid_basis=canonical_basis, fft_points=125)
    canonical_rows = np.full_like(padded_rows, 1e30*(1+2j))
    canonical_rows[:, :nmu, :nmom] = rows
    canonical_value = host(canonical_action(put(canonical_rows), cache['gram']))
    canonical_error = float(np.max(np.abs(canonical_value[:, :nmu, :nmu]-expected)))
    assert canonical_error < 3e-12
    assert np.count_nonzero(canonical_value[:, nmu:]) == np.count_nonzero(canonical_value[:, :, nmu:]) == 0
    poison_errors = []
    for poison in [np.nan+1j*np.nan, 1e300*(1+2j)]:
        bad_rows = np.full_like(padded_rows, poison)
        bad_rows[:, active, :nmom] = rows
        bad_gram = np.full_like(padded_gram, poison)
        bad_gram[:, :nmom, :nmom] = gram
        poisoned = host(action(put(bad_rows), put(bad_gram)))
        assert np.array_equal(poisoned, actual)
        poison_errors.append(float(np.max(np.abs(poisoned-actual))))
    zero = host(action(put(np.zeros_like(padded_rows)), cache['gram']))
    assert np.count_nonzero(zero) == 0
    missing_bra = scale*(rows @ gram @ rows.swapaxes(-2, -1))
    wrong_scale = rows.conj() @ gram @ rows.swapaxes(-2, -1)
    bra_signal = float(np.max(np.abs(missing_bra-expected)))
    scale_signal = float(np.max(np.abs(wrong_scale-expected)))
    assert bra_signal > .01 and scale_signal > .01
    negatives = []
    def refuse(label, callback):
        try:
            callback()
        except (ValueError, FileExistsError, RuntimeError):
            negatives.append(label)
        else:
            raise AssertionError('expected refusal: '+label)
    refuse('stale_file_pin', lambda: load(digest='0'*64))
    changed = copy.deepcopy(geometry)
    changed['atom_centres_bohr'][0][0] += .01
    refuse('changed_atom_order_or_position', lambda: load(geo=changed))
    changed = copy.deepcopy(geometry)
    changed['operator_q_fractional'][1:] = changed['operator_q_fractional'][1:][::-1]
    refuse('changed_q_order', lambda: load(geo=changed))
    changed = copy.deepcopy(geometry)
    changed['cell_volume_bohr3'] *= 2
    refuse('wrong_cell_volume', lambda: load(geo=changed))
    refuse('harmonic_permutation', lambda: load(channels=lm[::-1]))
    refuse('incomplete_harmonics', lambda: load(channels=lm[:-1]))
    refuse('immutable_writer_destination', lambda: write_periodic_compensation_cache(
        cache_path, put(padded_gram), mesh=mesh, geometry=geometry, lm=lm, preparation=preparation))
    wrong_receipt = dict(preparation, receipt_sha256='0'*64)
    refuse('stale_preparation_receipt', lambda: write_periodic_compensation_cache(
        directory/'wrong_receipt.h5', put(padded_gram), mesh=mesh,
        geometry=geometry, lm=lm, preparation=wrong_receipt))
    # Corrupt a single bounded fixture entry after collective handles close.
    damaged = directory/'changed_payload.h5'
    uncommitted = directory/'uncommitted.h5'
    variable_metadata = directory/'variable_metadata.h5'
    relocated = directory/'relocated.h5'
    if jax.process_index() == 0:
        import shutil
        shutil.copyfile(cache_path, damaged)
        shutil.copyfile(cache_path, uncommitted)
        shutil.copyfile(cache_path, variable_metadata)
        shutil.copyfile(cache_path, relocated)
        with h5py.File(damaged, 'a') as stream:
            stream['periodic_compensation_gram_ry'][0, 0, 0] += .1
        with h5py.File(uncommitted, 'a') as stream:
            stream[COMMIT_STATE][0] = 0
        with h5py.File(variable_metadata, 'a') as stream:
            metadata_text = stream['periodic_compensation_metadata_json'][()].decode()
            del stream['periodic_compensation_metadata_json']
            stream.create_dataset('periodic_compensation_metadata_json', data=metadata_text,
                                  dtype=h5py.string_dtype('utf-8'))
        upstream.rename(directory/'upstream_receipt_retired.json')
    barrier('periodic_fixture_corruptions')
    refuse('changed_payload_file_pin', lambda: load(path=damaged))
    refuse('uncommitted_rehashed_file', lambda: load(path=uncommitted, digest=_file_digest(uncommitted)))
    refuse('unbounded_variable_metadata_rehashed_file',
        lambda: load(path=variable_metadata, digest=_file_digest(variable_metadata)))
    relocated_cache = load(path=relocated)
    assert np.array_equal(host(relocated_cache['gram']), loaded)
    rank_ledger = dict(rank=jax.process_index(), process_count=jax.process_count(),
        mesh_shape=list(mesh.devices.shape),
        hlo=dict(path=str(hlo_path), sha256=_file_digest(hlo_path),
            collective_permute_occurrences=hlo.count('collective-permute'),
            all_gather_occurrences=hlo.count('all-gather'),
            custom_call_occurrences=hlo.count('custom-call')),
        compiled_argument_bytes=memory.argument_size_in_bytes,
        compiled_output_bytes=memory.output_size_in_bytes,
        compiled_temp_bytes=memory.temp_size_in_bytes,
        compiled_alias_bytes=memory.alias_size_in_bytes,
        device_memory_stats=[{k:int(v) for k,v in (d.memory_stats() or {}).items()
            if isinstance(v, (int, np.integer))} for d in jax.local_devices()],
        native_workspace_bytes=ledger['vendor_gemm_workspace_bytes_per_rank'],
        scope='Per-rank compiled action buffers and queried public native GEMM workspace; bounded fixture inputs/results are intentionally tiny replicated host oracles.')
    (directory/f'workspace.rank{jax.process_index()}.json').write_text(json.dumps(rank_ledger, indent=2)+'\n')
    barrier('periodic_fixture_per_rank_ledgers')
    result = dict(status='PASS', owner_sha256=_file_digest(__file__),
        source_owners={name:_file_digest(Path(__file__).resolve().parents[1]/name)
            for name in ['src/isdf/atomic_coulomb.py', 'src/isdf/coulomb_fourier_cache.py']},
        source='Planted complex physical multipole Gram, exact q rows and geometry; no orbital or Fourier preparation.',
        physical_shape=[nq, nmu, nmom], carriers=[mu.carrier, moment.carrier],
        metric_bytes_exact=True, input_ghost_poison_inert=poison_errors,
        cache_relocation_without_upstream_path_passed=True,
        full_complex_action_max_error=error, missing_bra_negative_signal=bra_signal,
        canonical_prefix_action_max_error=canonical_error,
        active_interleaved_slots=np.flatnonzero(active).tolist(),
        missing_grid_scale_negative_signal=scale_signal, zero_moments_exact=True,
        refusals=negatives, workspace_ledger=ledger, cache_sha256=written['file_sha256'],
        scope='Real P4 HDF tile IO and two public distributed N,N GEMMs with transpose partner, logical remainders and typed ghost masks. No actual AgI positive-metric accuracy, spectrum, tail or public attach admission.')
    if jax.process_index() == 0:
        result['per_rank_workspace_ledgers'] = [json.loads((directory/f'workspace.rank{rank}.json').read_text())
            for rank in range(jax.process_count())]
        (directory/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    barrier('periodic_fixture_receipt')
    return 0


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack(platform='gpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    run_main_and_finalize(lambda: run_oracle(runtime, args.directory))
