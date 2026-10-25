"""The real ζ reader pads at its owned SlabIO read, including short q tails."""
from pathlib import Path
from unittest.mock import patch
import math

import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update('jax_enable_x64', True)


def write_zeta_fixture(path, world, logical=13, disk_mu=None):
    """Write a tiny complete file for the real header and slab readers."""
    from file_io.isdf_header import IsdfHeader, write_isdf_header
    from file_io.commit_state import set_commit_state
    path = Path(path)
    side, nq = math.isqrt(int(world)), int(world)+1
    ng = -(-16//side)*side
    valid = np.resize(np.array([7, 11, 9, 6, 8]), nq)
    q, m, g = np.ogrid[:nq, :logical, :ng]
    values = (10000*q+32*m+g + 1j*(q+2*m-g)).astype(np.complex128)
    values = np.where(g < valid[:, None, None], values, 0)
    if disk_mu is not None:
        values = np.pad(values, ((0, 0), (0, int(disk_mu)-logical), (0, 0)))
    fields = {
        'versionnumber': 1, 'flavor': 2,
        'kpoints/nspin': 1, 'kpoints/nspinor': 1, 'kpoints/nrk': nq,
        'kpoints/mnband': 2, 'kpoints/ngkmax': ng, 'kpoints/ecutwfc': 20.,
        'kpoints/kgrid': [nq, 1, 1], 'kpoints/shift': np.zeros(3),
        'kpoints/ngk': valid, 'kpoints/ifmin': np.ones((1, nq), int),
        'kpoints/ifmax': np.ones((1, nq), int), 'kpoints/w': np.ones(nq)/nq,
        'kpoints/rk': np.column_stack((np.arange(nq)/nq, np.zeros((nq, 2)))),
        'kpoints/el': np.zeros((1, nq, 2)), 'kpoints/occ': np.zeros((1, nq, 2)),
        'gspace/ng': ng, 'gspace/ecutrho': 80., 'gspace/FFTgrid': [8, 8, 8],
        'symmetry/ntran': 1, 'symmetry/cell_symmetry': 0,
        'symmetry/mtrx': np.eye(3, dtype=int)[None], 'symmetry/tnp': np.zeros((1, 3)),
        'crystal/celvol': 1., 'crystal/recvol': 1., 'crystal/alat': 1.,
        'crystal/blat': 1., 'crystal/nat': 1, 'crystal/avec': np.eye(3),
        'crystal/bvec': np.eye(3), 'crystal/adot': np.eye(3),
        'crystal/bdot': np.eye(3), 'crystal/atyp': [1], 'crystal/apos': np.zeros((1, 3)),
    }
    with h5py.File(path, 'w') as f:
        for name, value in fields.items():
            f.create_dataset('mf_header/'+name, data=value)
        f.create_dataset('zeta_q_G', data=values)
    rmu = np.array(np.unravel_index(np.arange(logical), (8, 8, 8))).T
    gvec = np.full((nq, 3, ng), -4, np.int32)
    for qi, count in enumerate(valid):
        cols = np.arange(count)
        gvec[qi, :, :count] = np.stack((cols % 3-1, (cols//3) % 3-1, cols//9))
    write_isdf_header(path, IsdfHeader(density='scalar', vertex_mu_L=0,
        r_mu_fft_idx=rmu, r_mu_crystal=rmu/8, zeta_layout='G_flat',
        gvec_components=gvec, ngk_per_q=valid, zeta_cutoff_ry=20.))
    with h5py.File(path, 'a') as f:
        set_commit_state(f, True)


def check_zeta_padded_reads(mesh, path):
    """Read through actual ZetaLoader/SlabIO; compare only owned shard values."""
    from zeta_loader import ZetaLoader
    from file_io.restart_bundle import _ZetaGTiles
    from file_io.slab_io import SlabIO
    world = int(mesh.size)
    records, reads = [], []
    current = {}
    original_read = SlabIO.read_slab

    def owned_read(io, name, *args, **kwargs):
        if name == 'zeta_q_G':
            expected_shape, expected_spec = current['shape'], current['spec']
            assert kwargs['partition_spec'] == expected_spec, 'reader removed an owned axis'
            assert tuple(kwargs['shape']) == expected_shape
            target = NamedSharding(mesh, expected_spec)
            assert not target.is_fully_replicated
            reads.append(dict(shape=list(expected_shape), spec=str(expected_spec)))
        return original_read(io, name, *args, **kwargs)

    with patch.object(SlabIO, 'read_slab', owned_read):
        with ZetaLoader(path, mesh=mesh) as loader:
            logical, ng = int(loader.n_rmu), int(loader.n_G_sph_disk)
            carrier = -(-logical//world)*world
            tiles = _ZetaGTiles(loader, path=str(path), mu_carrier=carrier)
            cases = ((world, 1, P(None, ('x', 'y'), None)),
                     (1, min(3, world), P(None, ('x', 'y'), None)),
                     (world, 1, P(None, 'x', 'y')),
                     (0, world, P(('x', 'y'), None, None)))
            for offset, count, spec in cases:
                shape = (count, carrier, ng)
                current.update(shape=shape, spec=spec)
                result = tiles.read_q_slab(offset, count, sharding=NamedSharding(mesh, spec))
                result.block_until_ready()
                assert result.shape == shape
                assert result.sharding.is_equivalent_to(NamedSharding(mesh, spec), 3)
                largest = 0
                for shard in result.addressable_shards:
                    axes = [np.arange(*index.indices(n)) for index, n in zip(shard.index, shape)]
                    q, m, g = (axes[0][:, None, None]+offset,
                               axes[1][None, :, None], axes[2][None, None, :])
                    want = 10000*q+32*m+g + 1j*(q+2*m-g)
                    want = np.where((m < logical) & (g < loader.ngk_per_q[q]), want, 0)
                    got = np.asarray(shard.data)
                    np.testing.assert_array_equal(got, want)
                    assert np.isfinite(got).all()
                    largest = max(largest, int(got.nbytes))
                assert largest == math.prod(shape)*16//world
                records.append(dict(offset=offset, count=count, shape=list(shape), spec=str(spec),
                                    largest_owned_bytes=largest))
    assert len(reads) == len(records)
    return dict(world=world, logical=logical, carrier=carrier, reads=reads, cases=records)


@pytest.mark.parametrize('side,logical', ((2, 13), (3, 13), (2, 4), (3, 9)))
def test_real_zeta_reader_pads_inside_owned_hyperslabs(tmp_path, side, logical):
    if jax.device_count() < side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))
    path = tmp_path/'zeta.h5'
    write_zeta_fixture(path, int(mesh.size), logical)
    check_zeta_padded_reads(mesh, path)


def test_real_zeta_reader_p64_owned_padding(tmp_path):
    """Fresh process gives the actual reader 64 independent local shard requests."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, XLA_FLAGS='--xla_force_host_platform_device_count=64')
    command = ('import sys; from pathlib import Path; import numpy as np; import jax; '
               'from jax.sharding import Mesh; '
               'from tests.test_zeta_padded_owned_read_cpu import write_zeta_fixture,check_zeta_padded_reads; '
               'p=Path(sys.argv[1]); write_zeta_fixture(p,64); '
               'm=Mesh(np.asarray(jax.devices()).reshape(8,8),("x","y")); '
               'check_zeta_padded_reads(m,p)')
    run = subprocess.run([sys.executable, '-c', command, str(tmp_path/'zeta64.h5')],
                         env=env, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout+run.stderr


def test_header_only_zeta_cannot_enter_the_distributed_slab_path(tmp_path):
    from zeta_loader import ZetaLoader
    from file_io.restart_bundle import _ZetaGTiles
    path = tmp_path/'header_only.h5'
    write_zeta_fixture(path, 4)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
    with ZetaLoader(path) as loader:
        tiles = _ZetaGTiles(loader, path=str(path), mu_carrier=16)
        with patch.object(loader, 'read_zeta_G_local', side_effect=AssertionError('whole slab host read')):
            with pytest.raises(RuntimeError, match='HEADER-ONLY'):
                tiles.read_q_slab(0, 1, sharding=NamedSharding(mesh, P(None, None, None)))
        # Explicit local diagnostics retain the existing tiny read contract.
        got = tiles[0]
        assert got.shape == (16, loader.n_G_sph_disk)
        assert np.isfinite(got).all() and not np.any(got[13:])


@pytest.mark.parametrize('side', (2, 3))
def test_zeta_carrier_is_independent_of_old_processor_padding(tmp_path, side):
    from zeta_loader import ZetaLoader
    from file_io.restart_bundle import _ZetaGTiles
    if jax.device_count() < side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))
    path = tmp_path/'producer64.h5'
    write_zeta_fixture(path, side*side, logical=13, disk_mu=64)
    # Only discarded old processor padding is poisoned. The logical basis
    # and the target's inert carrier rows retain their exact values.
    with h5py.File(path, 'a') as f:
        f['zeta_q_G'][:, 18:] = np.nan
    check_zeta_padded_reads(mesh, path)
    carrier = -(-13//int(mesh.size))*int(mesh.size)
    with ZetaLoader(path) as loader:
        tiles = _ZetaGTiles(loader, path=str(path), mu_carrier=carrier)
        local_read = loader.read_zeta_G_local
        def bounded(key):
            assert key[1].start == 0 and key[1].stop == carrier
            return local_read(key)
        with patch.object(loader, 'read_zeta_G_local', bounded):
            for key in (0, slice(1, 3)):
                value = tiles[key]
                assert value.shape[-2] == carrier and np.isfinite(value).all()
                assert not np.any(value[..., 13:, :])
        with pytest.raises(ValueError, match='below the logical 13'):
            _ZetaGTiles(loader, path=str(path), mu_carrier=12)
