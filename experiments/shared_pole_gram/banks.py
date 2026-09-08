"""Authenticated campaign banks, with the Run299 congruence adapted to SlabIO."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

S = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')
F = S / 'runs/frequency_integration_sandbox'
ANCHOR = F / '299_shared_residue_ls_20260907/anchor_input.py'
ANCHOR_SHA = '6da657c65c58917b752f1102697b28e065c77491a0ec2ae7442979ee9e390916'
EV = 13.605693122994


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def low_owner(mesh):
    """Keep the frozen conversion/hash guards; replace only I/O and row extent."""
    import jax
    import jax.numpy as jnp
    from file_io.slab_io import SlabIO
    from jax.sharding import NamedSharding, PartitionSpec as P
    face = NamedSharding(mesh, P('x', 'y'))
    stack = NamedSharding(mesh, P(None, 'x', 'y'))
    assert sha(ANCHOR) == ANCHOR_SHA

    def intrinsic_rows(path):
        with SlabIO(path, mode='r', mesh=mesh) as io:
            train = io.read_slab('construction_value_cubic', partition_spec=P(None, 'x', 'y'))
            held = io.read_slab('held_value_cubic', partition_spec=P(None, 'x', 'y'))
        return jnp.concatenate((train, held), axis=0)

    def coulomb_rows(path, slot):
        with SlabIO(path, mode='r', mesh=mesh) as io:
            parents = io.read_small('q_parent_full_rows')
            slab = io.read_slab('V_canonical_qwedge', shape=(1, 896, 896),
                               offset=(slot, 0, 0), partition_spec=P(None, 'x', 'y'))
            v = jax.jit(lambda a: a[0], out_shardings=face)(slab)
        return parents, v

    source = ANCHOR.read_text()
    old = "with h5py.File(path,'r') as f:intrinsic=jnp.asarray(f['construction_value_cubic'][0])"
    assert source.count(old) == 1
    source = source.replace(old, 'intrinsic=intrinsic_rows(path)')
    old = """with h5py.File(COULOMB,'r') as f:
        parents=np.asarray(f['q_parent_full_rows'])
        assert np.array_equal(parents,_receipt['parent_qrows']) and int(parents[slot])==item['q_full']
        v=jnp.asarray(f['V_canonical_qwedge'][slot])"""
    assert source.count(old) == 1
    source = source.replace(old, """parents,v=coulomb_rows(COULOMB,slot)
    assert np.array_equal(parents,_receipt['parent_qrows']) and int(parents[slot])==item['q_full']""")
    # Compile the owner's exact expressions with explicit result layouts.
    # Its eager Hermitian addition otherwise replicates a transposed XY face.
    source = source.replace('ev,u=eig((v+v.conj().T)/2)', 'ev,u=eig(v)')
    owner_half = '(u*jnp.sqrt(ev)[None,:])@u.conj().T'
    owner_congruence = 'vh@intrinsic@vh'
    assert source.count('vh='+owner_half) == 1
    assert source.count('value='+owner_congruence) == 1
    half = jax.jit(eval('lambda u,ev: '+owner_half, {'jnp':jnp}), out_shardings=face)
    convert = jax.jit(eval('lambda vh,intrinsic: '+owner_congruence), out_shardings=stack)
    source = source.replace('vh='+owner_half, 'vh=half(u,ev)')
    source = source.replace('value='+owner_congruence, 'value=convert(vh,intrinsic)')
    namespace = dict(intrinsic_rows=intrinsic_rows, coulomb_rows=coulomb_rows,
                     half=half, convert=convert)
    exec(compile(source, str(ANCHOR), 'exec'), namespace)
    return namespace


def data_metadata(paths, parents):
    """Validate DATA's measured JSON contract without opening its HDF5 banks."""
    banks = []
    for directory in paths:
        directory = Path(directory).resolve()
        receipt_path = directory/'physical_receipt.json'
        launcher_path = directory/'launcher_receipt.json'
        rec = json.loads(receipt_path.read_text())
        launch = json.loads(launcher_path.read_text())
        if rec['status'] != 'COMPLETE' or launch['status'] != 'COMPLETE':
            raise ValueError(f'DATA bank is not COMPLETE in both receipts: {directory}')
        if (str(rec['jobid']), str(rec['stepid'])) != (str(launch['jobid']), str(launch['stepid'])):
            raise ValueError(f'DATA launcher/physical job.step mismatch: {directory}')
        if Path(launch['artifact']).resolve() != receipt_path:
            raise ValueError(f'DATA launcher points to another receipt: {directory}')
        hermite = rec['schema'] == 'lorrax.run307.hermite_cubic_physical_samples.v1'
        if rec['schema'] not in ('lorrax.run307.values_only_cubic_physical_samples.v1', 'lorrax.run307.hermite_cubic_physical_samples.v1') or rec['coordinate'] != 'canonical_physical_Wc':
            raise ValueError(f'Unsupported DATA coordinate/schema: {directory}')
        if rec['units']['z'] != 'Ry' or rec['values_only'] != (not hermite) or rec['geometry']['nmu'] != 896:
            raise ValueError(f'Unsupported DATA units/geometry: {directory}')
        if rec['parent_qrows'] != parents or len(rec['q_receipts']) != 29:
            raise ValueError(f'DATA parent-q contract differs: {directory}')
        if hermite:
            if rec['units'].get('derivative') != 'dWc/d(z_Ry^2)':
                raise ValueError('Unsupported physical derivative units')
            if rec['datasets'].get('training_derivative') != 'construction_derivative_cubic' or rec['datasets'].get('held_derivative') != 'held_derivative_cubic':
                raise ValueError('Unsupported Hermite dataset contract')
            count = sum(len(rec['schedule'][k+'_ev']) for k in ('construction','held'))
            if rec['schedule']['derivative_output_indices'] != list(range(count)):
                raise ValueError('Hermite requires one derivative per construction/held point')
            if len(rec.get('additional_source_artifacts',[])) != 2:
                raise ValueError('Missing Hermite adapter/quadrature source pins')
        source_evidence = {}
        pins = [(key,Path(rec[key+'_path']),rec[key+'_sha256']) for key in ('constructor','owner')]
        if hermite:
            pins += [(f'additional_{i}',Path(item['path']),item['sha256'])
                     for i,item in enumerate(rec['additional_source_artifacts'])]
        for key,source_path,expected_sha in pins:
            if sha(source_path) == expected_sha:
                source_evidence[key] = dict(path=str(source_path), sha256=expected_sha)
                continue
            # Completed banks pin bytes, not the mutable sampler's current HEAD.
            # Inspect at most32 revisions of this named file; the receipt hash
            # must match exact immutable bytes, never a current-code substitute.
            if key == 'owner':
                raise ValueError(f'DATA {key} source hash mismatch: {directory}')
            relative = source_path.relative_to(S).as_posix()
            revisions = subprocess.check_output(
                ['git', '-C', str(S), 'log', '-32', '--format=%H', '--', relative],
                text=True).splitlines()
            revision = None
            for candidate in revisions:
                blob = subprocess.check_output(['git', '-C', str(S), 'show', candidate+':'+relative])
                if hashlib.sha256(blob).hexdigest() == expected_sha:
                    revision = candidate
                    break
            if revision is None:
                raise ValueError(f'DATA immutable constructor bytes absent from bounded file history: {directory}')
            source_evidence[key] = dict(repository=str(S), revision=revision,
                                        path=relative, sha256=expected_sha)
        zs = []
        for kind in ('construction', 'held'):
            pairs = np.asarray(rec['z_Ry'][kind], dtype=float)
            energies = np.asarray(rec['schedule'][kind+'_ev'], dtype=float)
            if pairs.shape != (len(energies), 2) or not np.all(np.isfinite(pairs)):
                raise ValueError(f'DATA z_Ry shape/nonfinite: {directory}')
            actual = pairs[:,0] + 1j*pairs[:,1]
            expected = (energies + 1j*rec['sampling_eta_ev'])/EV
            if not np.allclose(actual, expected, rtol=1e-14, atol=1e-15) or np.any(actual.imag <= 0):
                raise ValueError(f'DATA actual z_Ry disagrees with schedule: {directory}')
            zs.append(actual)
        for slot, item in enumerate(rec['q_receipts']):
            if item['q_wedge'] != slot or item['q_full'] != parents[slot] or not item['shape_verified']:
                raise ValueError(f'DATA q shape/index receipt invalid: {directory}/q{slot:02d}')
        banks.append(dict(hermite=hermite, source_evidence=source_evidence, receipt=rec, receipt_path=str(receipt_path), receipt_sha256=sha(receipt_path),
                          launcher_path=str(launcher_path), launcher_sha256=sha(launcher_path),
                          z=zs[0], zh=zs[1]))
    return banks


def metadata(data_banks=None):
    """Read only JSON schedules here; held points occur once in the bank union."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('anchor_metadata', ANCHOR)
    anchor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(anchor)
    low = json.loads(anchor.OLD.read_text())
    broad = []
    for name in ('01_broad_a', '02_broad_b'):
        path = F / '300_na_shared_ls_inputs_20260907' / name / 'physical_receipt.json'
        rec = json.loads(path.read_text())
        assert rec['status'] == 'COMPLETE' and rec['coordinate'] == 'canonical_physical_Wc'
        assert rec['parent_qrows'] == low['parent_qrows']
        broad.append((path, rec))
    assert broad[0][1]['schedule']['held_ev'] == broad[1][1]['schedule']['held_ev']
    if data_banks:
        low['data_banks'] = data_metadata(data_banks, low['parent_qrows'])
        zs = [bank['z'] for bank in low['data_banks']]
        zhs = [bank['zh'] for bank in low['data_banks']]
        for bank in low['data_banks']:
            if bank['receipt']['input_artifacts']['coulomb_sha256'] != anchor.COULOMB_SHA:
                raise ValueError('DATA Coulomb differs from canonical Run299 dependency')
    else:
        zs = [(np.asarray(low['schedule']['construction_ev']) + .25j) / EV]
        zhs = [(np.asarray(low['schedule']['held_ev']) + .25j) / EV]
    for _, rec in broad:
        zs.append((np.asarray(rec['schedule']['construction_ev']) + 1j*rec['sampling_eta_ev'])/EV)
    rec = broad[0][1]
    zhs.append((np.asarray(rec['schedule']['held_ev']) + 1j*rec['sampling_eta_ev'])/EV)
    # Preserve all existing value indices; derivative twins follow all values.
    z, zh = np.r_[tuple(zs)], np.r_[tuple(zhs)]
    ds, dhs = [], []
    for bank in low.get('data_banks',[]):
        if bank['hermite']:
            ds.append(bank['z']); dhs.append(bank['zh'])
    low['derivative_scale_ry'] = np.r_[np.zeros(len(z)), *[a.imag for a in ds]]
    low['held_derivative_scale_ry'] = np.r_[np.zeros(len(zh)), *[a.imag for a in dhs]]
    return low, broad, np.r_[z,*ds], np.r_[zh,*dhs]


def load(slot, mesh, eig, owner, broad, data_banks=None):
    """Return physical training/held W and V; every spatial face stays XY tiled."""
    import jax.numpy as jnp
    from file_io.slab_io import SlabIO
    from jax.sharding import PartitionSpec as P
    derivative_trains, derivative_helds = [], []
    if data_banks:
        if not owner.get('_data_coulomb_authenticated'):
            assert sha(owner['COULOMB']) == owner['COULOMB_SHA']
            owner['_data_coulomb_authenticated'] = True
        parents, v = owner['coulomb_rows'](owner['COULOMB'], slot)
        assert np.array_equal(parents, data_banks[0]['receipt']['parent_qrows'])
        trains, helds = [], []
        paths = [dict(coulomb_path=str(owner['COULOMB']), coulomb_sha256=owner['COULOMB_SHA'])]
        for bank in data_banks:
            rec = bank['receipt']
            item = rec['q_receipts'][slot]
            path = Path(item['artifact'])
            if sha(path) != item['artifact_sha256']:
                raise ValueError(f'DATA artifact hash mismatch: {path}')
            with SlabIO(path, mode='r', mesh=mesh) as io:
                train = io.read_slab(rec['datasets']['training'], partition_spec=P(None,'x','y'))
                held = io.read_slab(rec['datasets']['held'], partition_spec=P(None,'x','y'))
                if bank['hermite']:
                    import jax
                    from jax.sharding import NamedSharding
                    scale_rows = jax.jit(lambda a,c:a*c[:,None,None],
                        out_shardings=NamedSharding(mesh,P(None,'x','y')))
                    for key,zrows,target in (('training_derivative',bank['z'],derivative_trains),
                                             ('held_derivative',bank['zh'],derivative_helds)):
                        dt = io.read_slab(rec['datasets'][key],partition_spec=P(None,'x','y'))
                        if dt.shape != (len(zrows),896,896) or dt.dtype != jnp.complex128:
                            raise ValueError(f'DATA derivative shape/dtype mismatch: {path}')
                        # Already canonical physical: scale complex rows before H/A.
                        target.append(scale_rows(dt,jnp.asarray(2*zrows.imag*zrows)))
            if train.shape != (len(bank['z']),896,896) or held.shape != (len(bank['zh']),896,896):
                raise ValueError(f'DATA actual dataset shape mismatch: {path}')
            if train.dtype != jnp.complex128 or held.dtype != jnp.complex128:
                raise ValueError(f'DATA dtype is not complex128: {path}')
            trains.append(train)
            helds.append(held)
            paths.append(dict(path=str(path), sha256=item['artifact_sha256'],
                              receipt=bank['receipt_path'], receipt_sha256=bank['receipt_sha256'],
                              launcher=bank['launcher_path'], launcher_sha256=bank['launcher_sha256'],
                              job_step=str(rec['jobid'])+'.'+str(rec['stepid']),
                              source_evidence=bank['source_evidence'],
                              coordinate=rec['coordinate'], band_energy_census=rec['band_energy_census']))
    else:
        low, provenance, v = owner['load_anchor'](slot, eig, jnp, None, return_coulomb=True)
        trains = [low[:12]]
        helds = [low[12:]]
        provenance = dict(provenance, owner_original_row=provenance.get('row'),
                          owner_original_z_ev=provenance.get('z_ev'),
                          row='all construction+held', construction_rows=12, held_rows=int(low.shape[0]-12),
                          conversion_scope='Run299 canonical congruence applied once to all Run216 construction+held rows')
        provenance.pop('z_ev', None)
        paths = [provenance]
    for index, (receipt_path, rec) in enumerate(broad):
        item = rec['q_receipts'][slot]
        path = Path(item['artifact'])
        assert sha(path) == item['artifact_sha256'] and item['q_wedge'] == slot
        with SlabIO(path, mode='r', mesh=mesh) as io:
            trains.append(io.read_slab('construction_value_cubic', partition_spec=P(None, 'x', 'y')))
            if index == 0:
                helds.append(io.read_slab('held_value_cubic', partition_spec=P(None, 'x', 'y')))
        paths.append(dict(path=str(path), sha256=item['artifact_sha256'],
                          receipt=str(receipt_path), receipt_sha256=sha(receipt_path)))
    return jnp.concatenate(trains+derivative_trains), jnp.concatenate(helds+derivative_helds), v, paths


def loss_weights(z):
    """Trapezoid times declared Sigma proxy, normalized separately on each line."""
    weights = np.zeros(len(z))
    for height in np.unique(z.imag):
        indices = np.flatnonzero(z.imag == height)
        indices = indices[np.argsort(z.real[indices])]
        x = z.real[indices] * EV
        if len(x) < 2 or np.any(np.diff(x) <= 0):
            raise ValueError('Each line requires distinct ordered quadrature nodes')
        quad = np.r_[np.diff(x)[0]/2, (x[2:]-x[:-2])/2, np.diff(x)[-1]/2]
        line = quad / (1+(x/20)**2)
        weights[indices] = line / line.sum()
    return weights / len(np.unique(z.imag))
