#!/usr/bin/env python3
"""Prepare full-WFN served overlaps on allocated CPU or GPU processes.

Run through lx with --manifest DIR --wfn WFN.h5 --output NEW_DIR. The
manifest must already name authenticated normalized and species served
caches. Its raw-parent artifact may be absent: this tool creates that
artifact and a patch for the complete fitting manifest. Source wavefunctions
are distributed by raw parent; only small overlap tables reach the host.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--wfn', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='new immutable output directory')
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _species_inputs(root):
    """Authenticate preparation inputs without requiring the output artifact."""
    from gw.isdf_augmentation import read_augmentation_manifest

    manifest = read_augmentation_manifest(root, load_raw_parent=False)
    if manifest.get('served_moment_caches') is None:
        raise ValueError('prepare the normalized and species served caches first')
    return manifest, manifest['served_moment_caches']


def _prepare_overlaps(wfn, caches, runtime):
    """Return small unrotated D[parent,band,function] and source identities.

    Every rank reads only its balanced parent block, with all physical bands
    and a common padded G carrier. Projection runs one parent at a time on
    the process-local device. Parent/G coordinates are physical bohr⁻¹;
    served_overlap_table owns the cell-volume and normalized-spinor units.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import (single_device_mesh, device_put_process_local,
        device_put_process_tiles, gather_to_host)
    from isdf.atomic_moments import served_overlap_table
    from wfn_loader import IBZRows

    domain = wfn.symmetry().parent_k_domain
    k = np.asarray(wfn.kvecs(k=domain), np.float64)
    g = np.asarray(wfn.gvecs(k=domain))
    ng = np.asarray(wfn.ngk_valid(k=domain), int)
    nparent, nb = len(k), int(wfn.nbands)
    ranks, rank = int(runtime.process_count), int(runtime.process_index)
    if runtime.mesh.size != ranks or ranks > nparent:
        raise ValueError('use one device per process and no more processes than raw parents')
    q, remainder = divmod(nparent, ranks)
    count = q+(rank < remainder)
    first = rank*q+min(rank, remainder)
    parents = np.arange(first, first+count)
    padded_count = q+bool(remainder)
    centers = (np.asarray(wfn.atom_crys, np.float64)%1.) @ (float(wfn.alat)*np.asarray(wfn.avec))
    atom_types = np.asarray(wfn.atom_types, int)
    if set(atom_types) != set(caches):
        raise ValueError('prepared species disagree with the WFN atoms')
    local = NamedSharding(single_device_mesh(), P())
    project = jax.jit(lambda source, bra: jnp.einsum('nsg,isg->ni', source, bra),
        in_shardings=(local, local), out_shardings=local)
    put = lambda value: device_put_process_local(np.asarray(value), local)
    source = np.asarray(wfn.load_process_local(k=IBZRows(tuple(parents)), bands=(0, nb),
        bispinor=True, bispinor_lift='normalized_rkb')).copy()
    if source.shape != (count, nb, 4, g.shape[1]):
        raise ValueError('source loader did not preserve the common physical-band/G carrier')
    overlaps = [np.zeros((padded_count, nb, len(caches[int(z)]['labels'])), np.complex128)
                for z in atom_types]
    source_sha = np.zeros((padded_count, 32), np.uint8)
    reciprocal = float(wfn.blat)*np.asarray(wfn.bvec)
    oracle = 0.
    for row, parent in enumerate(parents):
        source[row, ..., ng[parent]:] = 0.
        source_sha[row] = np.frombuffer(hashlib.sha256(
            np.ascontiguousarray(source[row]).tobytes()).digest(), np.uint8)
        wavevectors = (g[parent]+k[parent]) @ reciprocal
        wavevectors[ng[parent]:] = 0.
        values = put(source[row])
        for atom, (z, center) in enumerate(zip(atom_types, centers)):
            bra = served_overlap_table(caches[int(z)], wavevectors,
                center_cart=center, cell_volume=float(wfn.cell_volume))
            bra[..., ng[parent]:] = 0.
            overlaps[atom][row] = np.asarray(project(values, put(bra)))
            for band in {0, min(35, nb-1), min(63, nb-1), nb-1}:
                for function in {0, len(bra)//2, len(bra)-1}:
                    oracle = max(oracle, float(abs(overlaps[atom][row, band, function]
                        -np.sum(source[row, band]*bra[function]))))
    if not np.isfinite(oracle) or oracle > 2e-12:
        raise ValueError('bounded independent host overlap oracle failed')
    live = np.concatenate([np.arange(p*padded_count, p*padded_count+q+(p < remainder))
                           for p in range(ranks)])
    def gather_small(values):
        shape = (ranks*padded_count,)+values.shape[1:]
        sharding = NamedSharding(runtime.mesh, P(('x', 'y'), *([None]*(values.ndim-1))))
        def tile(index):
            if (index[0].start != rank*padded_count
                    or index[0].stop != (rank+1)*padded_count):
                raise ValueError('prepared overlap parent ownership changed')
            return values
        return np.asarray(gather_to_host(device_put_process_tiles(shape, sharding, tile)))[live]
    return dict(atom_D=tuple(gather_small(value) for value in overlaps),
        raw_source_sha256=gather_small(source_sha), k_parent_frac=k, gvecs=g, ngk_valid=ng,
        centers_cart=centers, atom_types=atom_types, cell_volume=float(wfn.cell_volume),
        physical_bands=nb, local_parent_rows=parents.tolist(), oracle_maximum=oracle)


def generate(args, runtime):
    from common.collectives import rank0_transaction
    from wfn_loader import WfnLoader
    from isdf.atomic_moments import (raw_parent_moment_binding, write_raw_parent_moments,
        load_raw_parent_moments)
    import numpy as np

    output, root = args.output.resolve(), args.manifest.resolve()
    if output.exists():
        raise FileExistsError('choose a new immutable output directory')
    started = time.perf_counter()
    manifest, caches = _species_inputs(root)
    with WfnLoader(str(args.wfn.resolve()), backend='eager') as wfn:
        prepared = _prepare_overlaps(wfn, caches, runtime)
        binding = raw_parent_moment_binding(wfn,
            **{key: prepared[key] for key in ('k_parent_frac', 'gvecs', 'ngk_valid',
                'centers_cart', 'atom_types', 'cell_volume', 'physical_bands')},
            served_cache_sha256_by_species=manifest['served_moments']['species_sha256'])
        if (manifest.get('overlap', {}).get('mode') == 'full_wfn_lowdin'
                and manifest['overlap'].get('bands') != prepared['physical_bands']):
            raise ValueError('manifest full-WFN window disagrees with the actual source')
        artifact = output/'raw_parent_served_moments.npz'
        def publish():
            output.mkdir(parents=True, exist_ok=False)
            write_raw_parent_moments(artifact, prepared['atom_D'],
                prepared['raw_source_sha256'], binding=binding)
            patch = dict(served_moments=dict(
                raw_parent_file=os.path.relpath(artifact, root), raw_parent_sha256=_sha256(artifact)))
            (output/'manifest_raw_parent_patch.json').write_text(json.dumps(patch, indent=2)+'\n')
        rank0_transaction(artifact, stage='prepare_raw_parent_served_overlaps', write=publish)
        sha = _sha256(artifact)
        loaded = load_raw_parent_moments(artifact, expected_binding=binding, expected_file_sha256=sha)
        for original, restored in zip(prepared['atom_D'], loaded['atom_D']):
            np.testing.assert_array_equal(original, restored)
        record = dict(schema='lorrax.raw_parent_moment_preparation.v1', artifact=str(artifact),
            artifact_sha256=sha, source_manifest_sha256=_sha256(root/'manifest.json'),
            authenticated_preparation_identity=manifest['identity'],
            generation_source_sha256=_sha256(__file__), binding=binding,
            atom_D_shapes=[list(value.shape) for value in prepared['atom_D']],
            rank=int(runtime.process_index), local_parent_rows=prepared['local_parent_rows'],
            independent_host_oracle_maximum=prepared['oracle_maximum'],
            total_seconds=time.perf_counter()-started,
            scope='Unrotated full physical WFN; source parent blocks distributed across all processes. No band crop, Gram pin, extra RKB multiplier or on-demand fitting rebuild.')
        with (output/f'receipt_rank{runtime.process_index:03d}.json').open('x') as stream:
            stream.write(json.dumps(record, indent=2)+'\n')
        if runtime.process_index == 0:
            print(json.dumps(record, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    args = parse_args()
    if 'SLURM_STEP_ID' not in os.environ:
        raise RuntimeError('run this preparation through lx on allocated compute nodes')
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()
    run_main_and_finalize(lambda: generate(args, runtime))
