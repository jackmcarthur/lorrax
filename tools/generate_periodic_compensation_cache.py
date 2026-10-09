#!/usr/bin/env python3
"""Prepare a kernel-bound compact-compensation Gram on allocated compute.

The public Coulomb service supplies every reciprocal weight. The incumbent
Sonine profile supplies the polynomial compensation Fourier transform. No
wavefunction, density, ISDF solve, Hartree source or photon head is rebuilt.
Reciprocal vectors are streamed; moment axes remain on the x/y device face.

Run with --spec SPEC.json --spec-sha256 SHA256 through the normal source-bound
lx GPU route. SPEC binds the WFN, augmentation manifest, source files, ordered
operator q full-grid indices, cutoff ladder in bohr^-1 and fresh output path.
The refinement measurement is a finite Gram comparison, not a rigorous tail
bound or an admission certificate for exchange or GW.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time


SPEC_SCHEMA = 'lorrax.periodic_compensation_preparation_request.v1'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def _interval(part, size):
    if part.step not in (None, 1):
        raise ValueError('periodic preparation requires contiguous shard slices')
    return (0 if part.start is None else part.start,
            size if part.stop is None else part.stop)


def read_spec(path, expected_sha256):
    """Authenticate controls before importing numerical owners or runtime."""
    path = Path(path).resolve()
    if not _sha(expected_sha256) or file_sha256(path) != expected_sha256:
        raise ValueError('periodic compensation preparation request SHA mismatch')
    spec = json.loads(path.read_text())
    names = {'schema', 'producer_sha256', 'source_directory', 'files_sha256',
             'wfn_file', 'augmentation_manifest_file', 'sys_dim',
             'operator_q_fullids', 'geometry', 'cutoffs_bohr_inverse',
             'g_tile', 'refinement_tolerance_ry', 'output_directory'}
    if not isinstance(spec, dict) or set(spec) != names or spec['schema'] != SPEC_SCHEMA:
        raise ValueError('periodic compensation preparation request schema mismatch')
    if file_sha256(__file__) != spec['producer_sha256']:
        raise ValueError('periodic compensation preparation producer SHA mismatch')
    source = Path(spec['source_directory']).resolve()
    if Path.cwd().resolve() != source or Path(os.environ.get('LORRAX_CHECKOUT', '')).resolve() != source:
        raise ValueError('periodic compensation preparation source/cwd binding mismatch')
    files = spec['files_sha256']
    if (not isinstance(files, dict) or not files
            or any(not _sha(value) for value in files.values())):
        raise ValueError('periodic compensation preparation requires exact file pins')
    required = {str(Path(spec[name]).resolve()) for name in
                ('wfn_file', 'augmentation_manifest_file')}
    if not required.issubset(files):
        raise ValueError('periodic compensation preparation lacks physical input pins')
    for filename, digest in files.items():
        if not Path(filename).is_absolute() or file_sha256(filename) != digest:
            raise ValueError(f'periodic compensation input/source SHA mismatch: {filename}')
    if isinstance(spec['sys_dim'], bool) or spec['sys_dim'] not in (2, 3):
        raise ValueError('periodic compensation requires bulk3D or aligned slab2D')
    return spec


def reciprocal_tiles(reciprocal, q, cutoff, tile_size):
    """Stream each integer reciprocal vector in the sphere exactly once.

    A Cartesian sphere lies in this finite Miller-index box by Cauchy--
    Schwarz applied to the columns of B^-1. Even its z-lines are chunked;
    no full rectangular box or reciprocal sphere is materialized.
    """
    import numpy as np

    B, q = np.asarray(reciprocal, float), np.asarray(q, float)
    edge = np.ceil(cutoff*np.linalg.norm(np.linalg.inv(B), axis=0)+np.abs(q)).astype(int)
    parts, count = [], 0
    for i in range(-int(edge[0]), int(edge[0])+1):
        for j in range(-int(edge[1]), int(edge[1])+1):
            for first in range(-int(edge[2]), int(edge[2])+1, tile_size):
                z = np.arange(first, min(first+tile_size, int(edge[2])+1))
                rows = np.column_stack((np.full(len(z), i), np.full(len(z), j), z))
                K = (rows+q) @ B
                rows = rows[np.einsum('gi,gi->g', K, K) <= cutoff*cutoff]
                cursor = 0
                while cursor < len(rows):
                    take = min(tile_size-count, len(rows)-cursor)
                    parts.append(rows[cursor:cursor+take])
                    count += take
                    cursor += take
                    if count == tile_size:
                        yield np.concatenate(parts), tile_size
                        parts, count = [], 0
    if count:
        rows = np.zeros((tile_size, 3), np.int64)
        rows[:count] = np.concatenate(parts)
        yield rows, count


def compensation_fourier(vectors, indices, *, centers, lm, support_radius):
    """Physical Fourier integrals for addressed atom-major unit multipoles."""
    import numpy as np
    from scipy.special import beta, sph_harm_y
    from isdf.augmentation_breit import _sonine_profile_transform

    K = np.asarray(vectors, np.float64)
    magnitude = np.linalg.norm(K, axis=1)
    theta = np.arccos(np.clip(np.divide(K[:, 2], magnitude,
        out=np.ones_like(magnitude), where=magnitude > 0), -1., 1.))
    phi = np.arctan2(K[:, 1], K[:, 0])
    nh, n = len(lm), len(centers)*len(lm)
    result = np.zeros((len(indices), len(K)), np.complex128)
    radial, angular, phase = {}, {}, {}
    for row, index in enumerate(indices):
        if index >= n:  # Mesh carrier tails are exactly inert.
            continue
        atom, harmonic = divmod(int(index), nh)
        l, m = map(int, lm[harmonic])
        if l not in radial:
            radial[l] = (2./beta(l+1.5, 7.)
                *_sonine_profile_transform(l, 6, support_radius, magnitude))
        if harmonic not in angular:
            angular[harmonic] = 4*np.pi*(-1j)**l*sph_harm_y(l, m, theta, phi)
        if atom not in phase:
            phase[atom] = np.exp(-1j*(K @ centers[atom]))
        result[row] = radial[l]*angular[harmonic]*phase[atom]
    return result


def bound_geometry(spec):
    """Derive ordered geometry from actual pinned WFN and atomic controls."""
    import numpy as np
    from wfn_loader import WfnLoader
    from symmetry_maps import bgw_integer_q_to_fractional
    from isdf.coulomb_fourier_cache import (periodic_compensation_geometry,
                                           _periodic_harmonics)

    manifest = json.loads(Path(spec['augmentation_manifest_file']).read_text())
    if manifest.get('charge_metric', {}).get('body_metric') != 'physical_low_local_high':
        raise ValueError('periodic compensation cache requires the positive charge model')
    support = float(manifest['radial']['support_radius'])
    lmax = manifest['angular']['lmax']
    if isinstance(lmax, bool) or not isinstance(lmax, int) or lmax < 0:
        raise ValueError('periodic compensation requires a nonnegative integer density lmax')
    lm = np.asarray([(l, m) for l in range(lmax+1) for m in range(-l, l+1)], np.int64)
    _periodic_harmonics(lm)
    ids = np.asarray(spec['operator_q_fullids'])
    with WfnLoader(str(Path(spec['wfn_file']).resolve()), backend='eager') as wfn:
        kgrid = tuple(map(int, wfn.kgrid))
        if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu'
                or len(np.unique(ids)) != len(ids) or np.any(ids < 0)
                or np.any(ids >= np.prod(kgrid))):
            raise ValueError('invalid ordered full-grid operator q rows')
        lattice = float(wfn.alat)*np.asarray(wfn.avec, np.float64)
        centers = (np.asarray(wfn.atom_crys, np.float64)%1.) @ lattice
        qfull = np.indices(kgrid).reshape(3, -1).T
        geometry = periodic_compensation_geometry(dict(
            reciprocal_rows_bohr_inverse=float(wfn.blat)*np.asarray(wfn.bvec, np.float64),
            cell_volume_bohr3=float(wfn.cell_volume), atom_centres_bohr=centers,
            operator_q_fractional=bgw_integer_q_to_fractional(qfull[ids], kgrid),
            support_radius_bohr=support), sys_dim=spec['sys_dim'])
    if geometry != spec['geometry']:
        raise ValueError('actual WFN/atom/q/support geometry differs from the preparation request')
    return geometry, lm


def _source_binding(spec):
    """Pin literal existing transform/kernel and the selected source files."""
    import numpy as np
    import scipy
    from isdf.augmentation_breit import _sonine_profile_transform
    from isdf.coulomb_fourier_cache import periodic_compensation_geometry
    from symmetry_maps import bgw_integer_q_to_fractional
    from wfn_loader import WfnLoader
    from vcoul import CoulombGeometry, get_kernel, v_qG_table
    from runtime import initialize_communicator_stack, run_main_and_finalize
    from runtime.source_closure import ensure_source_closure

    kernel = get_kernel(spec['sys_dim'])
    owners = (_sonine_profile_transform, v_qG_table, get_kernel, CoulombGeometry,
              type(kernel)._v_bare_per_q)
    sources = {owner.__module__+'.'+owner.__qualname__:
        hashlib.sha256(inspect.getsource(owner).encode()).hexdigest() for owner in owners}
    sources['tools.generate_periodic_compensation_cache'] = file_sha256(__file__)
    # Every imported physical owner must be one of the exact requested files.
    for owner in (*owners, periodic_compensation_geometry,
                  bgw_integer_q_to_fractional, WfnLoader,
                  initialize_communicator_stack, run_main_and_finalize,
                  ensure_source_closure):
        filename = str(Path(inspect.getsourcefile(owner)).resolve())
        if filename not in spec['files_sha256'] or file_sha256(filename) != spec['files_sha256'][filename]:
            raise ValueError(f'unpinned periodic preparation numerical owner: {owner.__qualname__}')
    return sources, dict(numpy_version=np.__version__, scipy_version=scipy.__version__)


def _payload_binding(gram, mesh, runtime):
    """Hash logical addressed shards; gather only bounded digest metadata."""
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_tiles, gather_to_host

    if len(gram.addressable_shards) != 1:
        raise ValueError('periodic preparation requires one addressable device per process')
    shard = gram.addressable_shards[0]
    index = [list(_interval(part, size)) for part, size in zip(shard.index, gram.shape)]
    payload = np.ascontiguousarray(shard.data)
    record = dict(index=index, dtype=payload.dtype.str, shape=list(payload.shape),
                  sha256=hashlib.sha256(memoryview(payload).cast('B')).hexdigest())
    encoded = json.dumps(record, sort_keys=True, separators=(',', ':')).encode('ascii')
    if len(encoded) > 1024:
        raise ValueError('periodic shard digest metadata exceeds its bounded extent')
    local = np.zeros((1, 1024), np.uint8)
    local[0, :len(encoded)] = np.frombuffer(encoded, np.uint8)
    sharding = NamedSharding(mesh, P(('x', 'y'), None))
    def tile(index):
        if _interval(index[0], runtime.process_count) != (runtime.process_index, runtime.process_index+1):
            raise ValueError('periodic digest rank ownership mismatch')
        return local
    packed = device_put_process_tiles((runtime.process_count, 1024), sharding, tile)
    records = [json.loads(bytes(row).rstrip(b'\0'))
               for row in np.asarray(gather_to_host(packed))]
    records.sort(key=lambda row: row['index'])
    encoded = json.dumps(records, sort_keys=True, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(encoded).hexdigest(), records


def generate(spec, spec_sha256, runtime):
    """Accumulate finite-cutoff Grams with bounded, synchronous G tiles."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_tiles, rank0_transaction
    from isdf.coulomb_fourier_cache import write_periodic_compensation_cache
    from runtime.padding import padded_axis
    from vcoul import CoulombGeometry, get_kernel, v_qG_table

    if runtime.n_devices != runtime.process_count or runtime.n_local_devices != 1:
        raise ValueError('periodic preparation requires exactly one device per process')
    cutoffs = np.asarray(spec['cutoffs_bohr_inverse'], np.float64)
    tile_size, tolerance = spec['g_tile'], spec['refinement_tolerance_ry']
    if (cutoffs.ndim != 1 or len(cutoffs) < 2 or not np.isfinite(cutoffs).all()
            or np.any(cutoffs <= 0) or np.any(np.diff(cutoffs) <= 0)
            or isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1
            or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('invalid reciprocal cutoff ladder/tile/refinement tolerance')
    if not jax.config.x64_enabled:
        raise ValueError('periodic compensation preparation requires actual JAX x64')
    output = Path(spec['output_directory']).resolve()
    if output.exists():
        raise FileExistsError('choose a fresh immutable periodic compensation output directory')
    geometry, lm = bound_geometry(spec)
    sources, versions = _source_binding(spec)
    mesh = runtime.mesh
    nq = len(geometry['operator_q_fractional'])
    centers = np.asarray(geometry['atom_centres_bohr'], np.float64)
    reciprocal = np.asarray(geometry['reciprocal_rows_bohr_inverse'], np.float64)
    nmoment = len(centers)*len(lm)
    axis = padded_axis(nmoment, mesh, name='periodic moment',
        specs=((P(None, 'x', 'y'), 1), (P(None, 'x', 'y'), 2)))
    nc, n = len(cutoffs), axis.carrier
    gram_spec = P(None, None, 'x', 'y')
    gram_sharding = NamedSharding(mesh, gram_spec)
    xshard, yshard = NamedSharding(mesh, P('x', None)), NamedSharding(mesh, P('y', None))
    scalar_shard = NamedSharding(mesh, P())
    weights_shard = NamedSharding(mesh, P(None, None))
    gram_shape = (nc, nq, n, n)
    zeros = lambda index: np.zeros(tuple(last-first for part, size in zip(index, gram_shape)
        for first, last in (_interval(part, size),)), np.complex128)
    grams = device_put_process_tiles((nc, nq, n, n), gram_sharding, zeros)

    @jax.jit
    @jax.shard_map(mesh=mesh,
        in_specs=(gram_spec, P('x', None), P('y', None), P(None, None), P()),
        out_specs=gram_spec)
    def accumulate(G, Fx, Fy, weights, qi):
        contribution = jnp.einsum('ig,cg,jg->cij', Fx.conj(), weights, Fy)
        return G.at[:, qi].add(contribution)

    abstract = lambda shape, dtype, sharding: jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    executable = accumulate.lower(
        abstract(grams.shape, np.complex128, gram_sharding),
        abstract((n, tile_size), np.complex128, xshard),
        abstract((n, tile_size), np.complex128, yshard),
        abstract((nc, tile_size), np.float64, weights_shard),
        abstract((), np.int32, scalar_shard)).compile()
    price = executable.memory_analysis()
    if price is None:
        raise ValueError('periodic compensation accumulation AOT has no memory price')
    compiled = {name: int(getattr(price, name)) for name in
        ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
    # A full compiler total added to actual resident leaves is conservative;
    # this is a plan, never an observed allocator peak or whole-GW plan.
    planned_bytes = (16*nc*nq*n*n/mesh.size
        +16*tile_size*n*(1/mesh.shape['x']+1/mesh.shape['y'])+8*nc*tile_size
        +compiled['argument_size_in_bytes']+compiled['output_size_in_bytes']
        +compiled['temp_size_in_bytes']-compiled['alias_size_in_bytes'])
    kernel = get_kernel(spec['sys_dim'])
    coulomb_geometry = CoulombGeometry(reciprocal, geometry['cell_volume_bohr3'])
    counts, tiles, started = [], [], time.perf_counter()
    for qi, q in enumerate(np.asarray(geometry['operator_q_fractional'], np.float64)):
        count, ntiles = 0, 0
        q_device = device_put_process_tiles((), scalar_shard, lambda _: np.asarray(qi, np.int32))
        for miller, valid in reciprocal_tiles(reciprocal, q, float(cutoffs[-1]), tile_size):
            K = (miller+q) @ reciprocal
            weights = np.asarray([v_qG_table(kernel, q[None], miller.T[None],
                geometry=coulomb_geometry, vcoul_cutoff_ry=float(cutoff*cutoff),
                v_head_fn=None)[0] for cutoff in cutoffs])
            weights[:, valid:] = 0.  # finite-q padding must never become a physical q slot.
            if not np.isfinite(weights).all() or np.min(weights) < -1e-12:
                raise ValueError('nonfinite or negative public compensation weights')
            cache = {}
            def fourier_tile(index):
                key = _interval(index[0], n)
                if key not in cache:
                    value = compensation_fourier(K, np.arange(*key), centers=centers, lm=lm,
                        support_radius=geometry['support_radius_bohr'])
                    value[:, valid:] = 0.
                    cache[key] = value
                return cache[key]
            Fx = device_put_process_tiles((n, tile_size), xshard, fourier_tile)
            Fy = device_put_process_tiles((n, tile_size), yshard, fourier_tile)
            W = device_put_process_tiles((nc, tile_size), weights_shard, lambda _: weights)
            grams = executable(grams, Fx, Fy, W, q_device)
            # This drain is unconditional, including host-cache/host-IO cases.
            # It bounds all G-tile inputs, tables, device outputs and host caches.
            grams.block_until_ready()
            del Fx, Fy, W, cache
            count += valid
            ntiles += 1
        counts.append(count)
        tiles.append(ntiles)
        if runtime.process_index == 0:
            print(f'Compensation q {qi+1}/{nq}: {count} reciprocal vectors, {ntiles} tiles', flush=True)
    reduce_sharding = NamedSharding(mesh, P(None, None))
    refinements = jax.jit(lambda G: jnp.max(jnp.abs(G[1:]-G[:-1]), axis=(-2, -1)).T,
        in_shardings=gram_sharding, out_shardings=reduce_sharding)(grams)
    refinement = np.asarray(refinements.addressable_shards[0].data)
    if not np.isfinite(refinement).all() or np.max(refinement[:, -1]) > tolerance:
        raise ValueError(f'periodic compensation final cutoff change exceeds {tolerance} Ry')
    gram = jax.jit(lambda G: G[-1], in_shardings=gram_sharding,
        out_shardings=NamedSharding(mesh, P(None, 'x', 'y')))(grams)
    gram.block_until_ready()
    del grams
    payload_sha, shards = _payload_binding(gram, mesh, runtime)
    receipt = dict(schema='lorrax.periodic_compensation_preparation.v1',
        producer_sha256=file_sha256(__file__), spec_sha256=spec_sha256,
        input_and_source_files_sha256=spec['files_sha256'], geometry=geometry, lm=lm.tolist(),
        q_fullids=spec['operator_q_fullids'], cutoffs_bohr_inverse=cutoffs.tolist(),
        refinement_max_ry=refinement.tolist(), refinement_tolerance_ry=tolerance,
        reciprocal_counts=counts, reciprocal_tiles=tiles, g_tile=tile_size,
        producer_sources_sha256=sources, versions=versions, payload_sha256=payload_sha,
        payload_digest_scope='SHA256 of sorted logical carrier-shard index/dtype/shape/data-SHA records',
        payload_shards=shards, logical_shape=[nq, nmoment, nmoment], moment_carrier=n,
        mesh_shape=list(runtime.mesh_shape), process_count=int(runtime.process_count),
        aot_memory=compiled, planned_bytes_per_rank=int(np.ceil(planned_bytes)),
        allocator_peak_bytes=None, memory_scope='Bounded preparation accumulation only; conservative AOT-plus-leaf plan, not observed peak or full fitting/GW lifetime.',
        accumulated_seconds=time.perf_counter()-started,
        scope='Actual finite-cutoff compact compensation Gram under the public point Coulomb kernel, K=0 excluded. Observed cutoff changes do not rigorously certify the infinite reciprocal tail or any Sigma/GW error. No Hartree or public slab GW admission.')
    receipt_path = output/'preparation_receipt.json'
    def publish_receipt():
        output.mkdir(parents=True, exist_ok=False)
        with receipt_path.open('x') as stream:
            stream.write(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    rank0_transaction(receipt_path, stage='periodic compensation preparation receipt', write=publish_receipt)
    for filename, digest in spec['files_sha256'].items():
        if file_sha256(filename) != digest:
            raise ValueError(f'periodic compensation input changed during preparation: {filename}')
    preparation = dict(receipt_path=str(receipt_path), receipt_sha256=file_sha256(receipt_path),
        payload_sha256=payload_sha, producer_sources_sha256=sources,
        cutoffs=cutoffs.tolist(), refinement_max=refinement.tolist())
    result = write_periodic_compensation_cache(output/'periodic_compensation.h5', gram,
        mesh=mesh, geometry=geometry, lm=lm, preparation=preparation)
    def publish_completion():
        with (output/'completion.json').open('x') as stream:
            stream.write(json.dumps(dict(status='PASS_FINITE_COMPENSATION_PREPARATION',
                artifact=result['path'], artifact_sha256=result['file_sha256'],
                preparation_receipt_sha256=preparation['receipt_sha256'],
                scope=receipt['scope']), indent=2, allow_nan=False)+'\n')
    rank0_transaction(output/'completion.json', stage='periodic compensation completion', write=publish_completion)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--spec-sha256', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if 'SLURM_STEP_ID' not in os.environ:
        raise RuntimeError('run compensation preparation through lx on allocated compute')
    spec = read_spec(args.spec, args.spec_sha256)
    from runtime import initialize_communicator_stack, run_main_and_finalize
    stack = initialize_communicator_stack()
    run_main_and_finalize(lambda: generate(spec, args.spec_sha256, stack))
