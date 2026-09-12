"""Rotate only retained Si support multiplets during a canonical SC replay.

The service outputs retain their physical counts and spectra. Seeded unitary
rotations act inside adjacent relative-gap 1e-6 blocks only. Actual QQ.H
projectors are compared before returning each changed Q to the constructor.
All direction/projector products use the distributed service and x/y faces.
"""


def main():
    import argparse
    from functools import lru_cache
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Driver import owns runtime startup, before instrumentation or JAX use.
    from gw import gw_jax as driver
    import distrib_la
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    import shared_pole_sc_invariants as observer

    mesh = driver.RUNTIME.mesh
    assert jax.process_count() == jax.device_count() == 4
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.output / 'direction_gauge.jsonl'
    seed, tolerance = 2026091028, 1e-6
    bound_entries = 64 * 368**2 // 4
    records = []
    counters = dict(calls=0, changed_calls=0, rotated_clusters=0,
                    rotated_columns=0, max_projector_relative=0.,
                    max_rotation_relative=0., max_host_unitary_entries=0)

    def emit(row):
        records.append(row)
        if jax.process_index() == 0:
            with journal.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print('SC_DIRECTION_GAUGE ' + json.dumps(row, allow_nan=False), flush=True)

    @lru_cache(maxsize=96)
    def kernels(b, n, r):
        rotate_plan = distrib_la.gemm_plan(mesh, m=n, k=r, n=r, nq=b,
                                          dtype=jnp.complex128)
        projector_plan = distrib_la.gemm_plan(mesh, m=n, k=r, n=n, nq=b,
                                             dtype=jnp.complex128)

        @jax.jit
        def rotate(q, u):
            return jax.lax.with_sharding_constraint(rotate_plan(q, u), face)

        @jax.jit
        def check(before, after):
            def projector(q):
                right = jax.lax.with_sharding_constraint(q.conj().swapaxes(-1, -2), face)
                return jax.lax.with_sharding_constraint(projector_plan(q, right), face)
            old = projector(before)
            new = projector(after)
            norm = jnp.linalg.norm(old, axis=(-2, -1))
            defect = jnp.linalg.norm(new-old, axis=(-2, -1))
            motion = jnp.linalg.norm(after-before, axis=(-2, -1))
            qnorm = jnp.linalg.norm(before, axis=(-2, -1))
            return defect/jnp.maximum(norm, 1e-300), motion/jnp.maximum(qnorm, 1e-300)
        return rotate, check

    def clusters(values):
        groups, start = [], 0
        for stop in range(1, len(values)+1):
            boundary = stop == len(values)
            if not boundary:
                a, b = values[stop-1:stop+1]
                boundary = abs(a-b) > tolerance * max(abs(a), abs(b))
            if boundary:
                if stop-start > 1:
                    groups.append((start, stop))
                start = stop
        return groups

    originals = {name: getattr(distrib_la, name) for name in
                 ('right_singular_vectors', 'leading_eigenvectors')}

    def wrapper(name, original):
        def select(*pos, **kwargs):
            q, values = original(*pos, **kwargs)
            assert abs(float(kwargs.get('multiplet_tol', 1e-6))-tolerance) < 1e-15
            batched = q.ndim == 3
            qb = q if batched else q[None]
            spectra = tuple(np.asarray(v) for v in values) if batched else (np.asarray(values),)
            b, n, r = qb.shape
            assert n <= 400 and b <= 4 and r <= n
            assert qb.sharding.is_equivalent_to(face, qb.ndim)
            entries = b*r*r
            assert entries < bound_entries, 'small replicated gauge exceeds literal per-rank object bound'
            assert b*n*n//4 < bound_entries, 'diagnostic projector exceeds per-rank object bound'
            index = counters['calls']
            counters['calls'] += 1
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            rotations = np.broadcast_to(np.eye(r, dtype=np.complex128), (b, r, r)).copy()
            rows = []
            for iq, spectrum in enumerate(spectra):
                group = clusters(spectrum)
                details = []
                for lo, hi in group:
                    m = hi-lo
                    matrix = rng.normal(size=(m, m)) + 1j*rng.normal(size=(m, m))
                    unitary = np.linalg.qr(matrix)[0]
                    rotations[iq, lo:hi, lo:hi] = unitary
                    details.append(dict(start=lo, stop=hi, width=m,
                                        highest=float(spectrum[lo]), lowest=float(spectrum[hi-1])))
                    counters['rotated_clusters'] += 1
                    counters['rotated_columns'] += m
                rows.append(dict(batch_row=iq, retained_rank=len(spectrum), clusters=details))
            changed = any(row['clusters'] for row in rows)
            counters['max_host_unitary_entries'] = max(counters['max_host_unitary_entries'], entries)
            if changed:
                rotate, check = kernels(b, n, r)
                u = device_put_process_local(rotations, face)
                rotated = rotate(qb, u)
                projector_error, motion = [np.asarray(gather_to_host(v)) for v in check(qb, rotated)]
                if not np.all(np.isfinite(projector_error)) or np.max(projector_error) > 1e-11:
                    raise RuntimeError('direction gauge failed actual support-projector preservation')
                assert np.max(motion) > 1e-4, 'declared multiplet rotation was a no-op'
                counters['changed_calls'] += 1
                counters['max_projector_relative'] = max(counters['max_projector_relative'], float(projector_error.max()))
                counters['max_rotation_relative'] = max(counters['max_rotation_relative'], float(motion.max()))
                q = rotated if batched else rotated[0]
            else:
                projector_error = np.zeros(b)
                motion = np.zeros(b)
            emit(dict(kind='direction_selection', call=index, service=name, shape=list(qb.shape),
                      changed=changed, rows=rows, actual_projector_relative=projector_error.tolist(),
                      actual_direction_motion_relative=motion.tolist(),
                      host_unitary_entries=entries, per_rank_projector_entries=b*n*n//4,
                      original_spectra_sha256=hashlib.sha256(b''.join(v.tobytes() for v in spectra)).hexdigest()))
            # Preserve the original spectrum objects, physical ranks and masks.
            return q, values
        return select

    source = Path(__file__).resolve().parents[2]
    binding_paths = [Path(__file__).resolve(), Path(observer.__file__).resolve(),
                     source/'src/gw/shared_pole_constructor.py', source/'src/gw/mpa/sigma.py',
                     source/'services/distrib_la/src/distrib_la/polar.py']
    binding = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
                   file_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binding_paths},
                   job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
                   seed=seed, multiplet_relative_gap=tolerance, literal_bound_complex_entries=bound_entries,
                   scope='Real Si current directions; unitary rotations only within retained multiplets, no physical rank or support changes')
    if jax.process_index() == 0:
        (args.output/'direction_gauge_binding.json').write_text(json.dumps(binding, indent=2)+'\n')
    emit(dict(kind='binding', **binding))
    status = 'RUNNING'
    try:
        for name, original in originals.items():
            setattr(distrib_la, name, wrapper(name, original))
        observer.main()
        status = 'OBSERVED' if counters['changed_calls'] else 'NO_MULTIPLETS_ROTATED'
    except Exception as exc:
        status = 'FAILED'
        emit(dict(kind='failure', exception=type(exc).__name__, message=str(exc)))
        raise
    finally:
        for name, original in originals.items():
            setattr(distrib_la, name, original)
        if jax.process_index() == 0:
            (args.output/'direction_gauge_receipt.json').write_text(json.dumps(
                dict(status=status, binding=binding, counters=counters), indent=2)+'\n')


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
