"""A reader on any node gets a just-written file whole, or a refusal that names the cause.

Compute-node script: run at two nodes on a shared file system,
``srun -N 2 -n 4 --ntasks-per-node 2 ... python tests/test_fresh_file_reads.py
--dir <scratch dir>`` (one node with the default ``--dir $TMPDIR`` runs the
same cases without the cross-node view); the pytest entry skips without the
launcher.

Rank 0 copies the hsuite H2- ``WFN.h5`` into ``--dir`` and cuts it short, the
size view a Lustre client gave the readers of a fresh ``zeta_q.h5`` on
Perlmutter scratch (``truncated file: eof < stored_eof``, the data complete).

(a) The tail comes back 3 s later: every rank's ``WfnLoader`` (a serial
    reader of the route ``file_io.commit_state.open_h5_settled``) opens the
    file once its size is whole and reads the same coefficients as the
    fixture.  Before 2026-11-22 the open raised at once.
(b) The file stays short: every rank's open refuses within the bounded wait
    (``SETTLE_TIMEOUT_S``, 2 s here) with an ``OSError`` that names the cause.
"""


def main():
    import argparse
    import os
    import shutil
    import threading
    import time
    from pathlib import Path
    import runtime
    stack = runtime.initialize_communicator_stack()
    import h5py
    import jax
    import numpy as np
    from common.collectives import barrier
    from file_io import commit_state
    from wfn_loader import WfnLoader

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', default=os.environ.get('TMPDIR', '/tmp'))
    args = p.parse_args()
    mesh = stack.mesh
    rank = jax.process_index()
    fixture = Path(__file__).parent / 'hsuite/fixture/WFN.h5'
    work = Path(args.dir) / (f"lorrax_fresh_reads_{os.environ.get('SLURM_JOB_ID', '0')}_"
                             f"{os.environ.get('SLURM_STEP_ID', '0')}")
    path = work / 'WFN.h5'
    size = os.path.getsize(fixture)
    cut = (size // 2) // 4096 * 4096
    if rank == 0:
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixture, path)
        with open(path, 'rb') as f:
            f.seek(cut)
            tail = f.read()
        os.truncate(path, cut)
    barrier('fresh_reads.cut')
    with h5py.File(fixture, 'r') as f:
        want = f['wfns/coeffs'][:, :, :64, :]

    # (a) the size view comes back
    restore = None
    if rank == 0:
        def put_back():
            time.sleep(3.0)
            with open(path, 'r+b') as f:
                f.seek(cut)
                f.write(tail)
                f.flush()
                os.fsync(f.fileno())
        restore = threading.Thread(target=put_back)
        restore.start()
    t0 = time.monotonic()
    with WfnLoader(path) as wfn:
        waited = time.monotonic() - t0
        nbands = int(wfn.nbands)
    with h5py.File(path, 'r') as f:
        got = f['wfns/coeffs'][:, :, :64, :]
    np.testing.assert_array_equal(got, want)
    if restore is not None:
        restore.join()
    assert waited > 1.0, waited          # the open waited for the tail
    barrier('fresh_reads.restored')

    # (b) the file stays short: a refusal that names the cause
    if rank == 0:
        os.truncate(path, cut)
    barrier('fresh_reads.cut_again')
    saved = commit_state.SETTLE_TIMEOUT_S
    commit_state.SETTLE_TIMEOUT_S = 2.0
    try:
        try:
            WfnLoader(path)
        except OSError as exc:
            text = str(exc)
        else:
            raise AssertionError('a file that stays short opened')
    finally:
        commit_state.SETTLE_TIMEOUT_S = saved
    assert 'truncated file' in text and 'open_h5_settled' in text and 'stale' in text, text
    barrier('fresh_reads.done')
    if rank == 0:
        shutil.rmtree(work, ignore_errors=True)
    print(f'FRESH_FILE_READS PASS rank={rank} P={int(mesh.size)} host={os.uname().nodename} '
          f'waited={waited:.2f}s nbands={nbands}', flush=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run directly under the configured MPI launcher')


if __name__ == '__main__':
    main()
