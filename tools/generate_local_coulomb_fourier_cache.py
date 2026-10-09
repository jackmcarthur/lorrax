#!/usr/bin/env python3
"""Prepare exact local charge Fourier splines on one allocated CPU rank.

Example through lx: --manifest MANIFEST_DIR --maximum-wavevector 8.944006
--output NEW_DIR. The momentum bound is in bohr^-1 and must cover every
physical consuming q+G. The provider checks that bound again. Preparation
reuses the existing field/factory and preserves its strict error checks.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path,
                        help='authenticated augmentation manifest directory')
    parser.add_argument('--maximum-wavevector', required=True, type=float,
                        help='upper bound for physical |q+G| in bohr^-1')
    parser.add_argument('--output', required=True, type=Path,
                        help='new immutable output directory')
    return parser.parse_args()


def generate(args, runtime):
    import numpy as np
    from gw.isdf_augmentation import read_augmentation_manifest, _radial_grid
    from isdf.atomic_coulomb import atomic_radial_metrics
    from isdf.coulomb_fourier_cache import (build_coulomb_fourier_cache,
        write_coulomb_fourier_cache, load_coulomb_fourier_cache,
        validate_coulomb_fourier_cache, _file_digest)

    if runtime.process_count != 1:
        raise ValueError('local Fourier preparation requires one CPU process')
    if not np.isfinite(args.maximum_wavevector) or args.maximum_wavevector < 0:
        raise ValueError('maximum-wavevector must be finite and nonnegative')
    output, manifest_dir = args.output.resolve(), args.manifest.resolve()
    if output.exists():
        raise FileExistsError('choose a new immutable output directory')
    manifest = read_augmentation_manifest(manifest_dir)
    radial = manifest['radial']
    if 'interpolation_degree' not in radial:
        raise ValueError('prepared Fourier cache requires the physical density interpolant')
    radius, weights, support = _radial_grid(radial)
    lmax = int(manifest['angular']['lmax'])
    if lmax < 0:
        raise ValueError('density angular lmax must be nonnegative')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # Fourier rows are physical radial integrals, independent of Nfft/Omega.
    # Unit metric scale is unused by the cache; the V consumer owns grid/Ry units.
    tables = atomic_radial_metrics(radius, weights, np.arange(lmax+1), support_radius=support,
        fft_points=1, cell_volume=1., interpolation_degree=radial['interpolation_degree'],
        quadrature_order=radial.get('quadrature_order'))
    metrics_seconds = time.perf_counter()-started
    started = time.perf_counter()
    cache = build_coulomb_fourier_cache(tables, args.maximum_wavevector,
                                      radial.get('fourier_points', 4097))
    factory_seconds = time.perf_counter()-started
    artifact = output/'local_coulomb_fourier.npz'
    write_coulomb_fourier_cache(artifact, cache)
    sha = _file_digest(artifact)
    started = time.perf_counter()
    loaded = load_coulomb_fourier_cache(artifact, expected_file_sha256=sha)
    validated = validate_coulomb_fourier_cache(loaded, tables, args.maximum_wavevector,
                                              radial.get('fourier_points', 4097))
    strict_replay_seconds = time.perf_counter()-started
    receipt = dict(schema='lorrax.local_coulomb_fourier_preparation.v1',
        artifact=str(artifact), artifact_sha256=sha, artifact_bytes=artifact.stat().st_size,
        source_manifest=str(manifest_dir/'manifest.json'),
        source_manifest_sha256=_file_digest(manifest_dir/'manifest.json'),
        authenticated_augmentation_identity=manifest['identity'],
        generator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        maximum_wavevector=float(args.maximum_wavevector), radial_points=len(radius),
        density_lmax=lmax, metric_preparation_seconds=metrics_seconds,
        one_time_factory_seconds=factory_seconds, strict_replay_seconds=strict_replay_seconds,
        source_binding=cache['source_binding'], field_binding=cache['field_binding'],
        Fourier_diagnostics={key: validated[key] for key in ('points', 'maximum_wavevector',
            'max_density_validation_error', 'max_compensation_validation_error', 'validation_points')},
        scope='Explicit one-time exact spline preparation. Unit metric scale is unused by the physical radial Fourier rows; the production consumer applies grid/Rydberg factors. No field, quadrature, rank or accuracy threshold is changed.')
    with (output/'preparation_receipt.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    args = parse_args()
    if 'SLURM_STEP_ID' not in os.environ:
        raise RuntimeError('run this preparation through lx on an allocated compute node')
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()
    run_main_and_finalize(lambda: generate(args, runtime))
