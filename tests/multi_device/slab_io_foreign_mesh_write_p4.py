"""Foreign-mesh SlabIO write gate, exactly P=4.

Reproduces the writer defect measured on the converged Fe SC runs
(10p/10q/10r, 2026-09-19).  The driver hands ``write_slab`` a per-map artifact
whose sharding lives on a SECOND mesh over the same devices (different axis
names), so the operand spans devices this process cannot address.  The pre-fix
writer either asked JAX for a cross-process different-device-order reshard (a
bare ``AssertionError``, 10p/10q) or fetched the operand with ``np.asarray``
("Fetching value for `jax.Array` that spans non-addressable ... devices", 10r);
in every case ``sigma_mnk.h5`` was written with ``lorrax_io_committed = 0`` and
the run refused publication.

Three arms, all through real SlabIO at P=4:

* ``foreign_mesh_write`` — the payload is sharded on a distinct mesh object
  over the same devices in the same order with different axis names, i.e.
  NOT the writer's layout.  The writer relabels it onto its own mesh (each
  device keeps its shard), never gathers it, and writes the exact bytes.
* ``writer_mesh_write`` — control: the same payload already on the writer's
  mesh, which must keep the historical byte-exact path.
* ``reordered_mesh_refusal`` — the payload on a mesh over the same devices
  in a DIFFERENT order.  Writing it would gather the whole array to every
  host, so it must refuse with ``GATE slab_io_foreign_mesh`` on every rank
  (decisions 2026-08-05).  Red twin: before the refusal, main gathered it.

Exit codes: 0 PASS, 1 FAIL.  Run line:

    <launcher> -n 4 python3 tests/multi_device/slab_io_foreign_mesh_write_p4.py \
      --output <evidence>/foreign_mesh_write_p4.json
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from runtime import initialize_communicator_stack, finalize_process
    initialize_communicator_stack()
    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host, resolve_mesh
    from file_io.slab_io import SlabIO

    mesh = resolve_mesh()
    assert jax.process_count() == 4 and mesh.size == 4, (jax.process_count(), mesh.size)
    devices = np.asarray(mesh.devices)
    assert devices.shape == (2, 2), devices.shape
    # Same devices, different axis names: a second mesh that is NOT the
    # writer's layout, which is the case that reaches the gather branch.
    foreign = Mesh(devices, ('u', 'v'))
    writer_spec = P(('x', 'y'))
    shape = (4, 4, 4)
    rng = np.random.default_rng(20260919)
    reference = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    def on(mesh_obj, spec):
        return jax.make_array_from_callback(
            shape, NamedSharding(mesh_obj, spec), lambda idx: reference[idx])

    payload = on(foreign, P(('u', 'v')))
    assert not payload.is_fully_addressable, (
        'this gate pins the multi-process sharded case; the payload is '
        'fully addressable, so the launch geometry is not P4 one-per-GPU')
    assert payload.sharding.mesh is not mesh

    import file_io._slab_io_ffi as _ffi_backend
    gathers = []
    _real_gather = _ffi_backend.gather_to_host

    def _counting_gather(x):
        gathers.append(tuple(x.shape))
        return _real_gather(x)
    _ffi_backend.gather_to_host = _counting_gather

    path = args.output.parent / 'foreign_mesh_write.h5'
    with SlabIO(path, mode='w', mesh=mesh) as io:
        io.write_slab('foreign_mesh_write', payload)
    foreign_gathers = list(gathers)

    reordered = on(Mesh(devices.T, ('u', 'v')), P(('u', 'v')))
    refusal = ''
    path3 = args.output.parent / 'reordered_mesh_write.h5'
    with SlabIO(path3, mode='w', mesh=mesh) as io:
        try:
            io.write_slab('reordered_mesh_write', reordered)
        except ValueError as exc:
            refusal = str(exc)
    _ffi_backend.gather_to_host = _real_gather
    with SlabIO(path, mode='r', mesh=mesh) as io:
        got_foreign = gather_to_host(
            io.read_slab('foreign_mesh_write', partition_spec=writer_spec))

    control = on(mesh, writer_spec)
    path2 = args.output.parent / 'writer_mesh_write.h5'
    with SlabIO(path2, mode='w', mesh=mesh) as io:
        io.write_slab('writer_mesh_write', control)
    with SlabIO(path2, mode='r', mesh=mesh) as io:
        got_control = gather_to_host(
            io.read_slab('writer_mesh_write', partition_spec=writer_spec))

    rows = [dict(
        name='foreign_mesh_write',
        exact=np.array_equal(got_foreign, reference),
        max_abs_error=float(np.max(np.abs(got_foreign - reference))),
        payload_fully_addressable=bool(payload.is_fully_addressable),
        host_gathers=len(foreign_gathers),
        foreign_mesh_axes=[str(a) for a in foreign.axis_names]), dict(
        name='writer_mesh_write',
        exact=np.array_equal(got_control, reference),
        max_abs_error=float(np.max(np.abs(got_control - reference))),
        payload_fully_addressable=bool(control.is_fully_addressable),
        foreign_mesh_axes=[str(a) for a in mesh.axis_names])]
    rows[0]['exact'] = rows[0]['exact'] and not foreign_gathers
    rows.append(dict(
        name='reordered_mesh_refusal',
        exact='GATE slab_io_foreign_mesh' in refusal,
        refusal=refusal[:160]))
    ok = all(row['exact'] for row in rows)
    result = dict(status='PASS' if ok else 'FAIL', checks=rows, expected_checks=3,
                  job=os.environ.get('SLURM_JOB_ID'),
                  step=os.environ.get('SLURM_STEP_ID'),
                  source_commit=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  scope=('P4 SlabIO foreign-mesh write: one (4,4,4) complex cube '
                         'written from a sharded operand on a renamed mesh with no '
                         'host gather, one control on the writer mesh, one refusal '
                         'on a reordered mesh; byte-exact read-back only, no '
                         'frequency integration and no production deck'))
    if jax.process_index() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    for row in rows:
        assert row['exact'], row
    finalize_process()


if __name__ == '__main__':
    main()
