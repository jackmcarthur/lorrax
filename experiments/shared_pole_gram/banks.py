"""Authenticated campaign banks, with the Run299 congruence adapted to SlabIO."""
from pathlib import Path
import hashlib
import json
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


def metadata():
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
    zs = [(np.asarray(low['schedule']['construction_ev']) + .25j) / EV]
    zhs = [(np.asarray(low['schedule']['held_ev']) + .25j) / EV]
    for _, rec in broad:
        zs.append((np.asarray(rec['schedule']['construction_ev']) + 1j*rec['sampling_eta_ev'])/EV)
    rec = broad[0][1]
    zhs.append((np.asarray(rec['schedule']['held_ev']) + 1j*rec['sampling_eta_ev'])/EV)
    return low, broad, np.r_[tuple(zs)], np.r_[tuple(zhs)]


def load(slot, mesh, eig, owner, broad):
    """Return physical training/held W and V; every spatial face stays XY tiled."""
    import jax.numpy as jnp
    from file_io.slab_io import SlabIO
    from jax.sharding import PartitionSpec as P
    low, provenance, v = owner['load_anchor'](slot, eig, jnp, None, return_coulomb=True)
    trains = [low[:12]]
    helds = [low[12:]]
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
    return jnp.concatenate(trains), jnp.concatenate(helds), v, paths


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
