"""Gate: the collective QP WFN writer on the phdf5 transport, any square P.

The checks of ``tests/test_qp_wfn_sharded_writer.py`` (ragged 4-k fixture,
``ngk`` not divisible by P, two spinors, one-window and several-window arms),
run on the real collective MPI-IO path instead of the emulated serial tier,
plus the mesh-aware WfnLoader read (the phdf5 union read BSE and htransform
use at P>1) compared shard by shard.

    [QPWFN_GATE_DIR=/path/to/evidence] \\
      lx run --pool POOL -N 1 -G 4 -n 4 -- python3 -u tests/multi_device/qp_wfn_sharded_writer_p4.py

No arguments.  Files (a 74 kB source WFN.h5 and two WFN_qp files) go only to
``QPWFN_GATE_DIR``, which must be on a filesystem every rank sees; unset, it is
``$SCRATCH/.qpwfn_gate/<job>.<step>`` (identical on every rank) and is removed
after a pass.  rc=0 and ``QPWFN GATE PASS`` on rank 0 iff every arm round-trips.
"""
from runtime import initialize_communicator_stack

RUNTIME = initialize_communicator_stack()

import os                                                    # noqa: E402
import sys                                                   # noqa: E402

import numpy as np                                           # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import test_qp_wfn_sharded_writer as case                    # noqa: E402
from common.collectives import barrier, process_rank, resolve_mesh  # noqa: E402


def _phdf5_reader_matches(out, want, mesh):
    """Every rank: its shards of the mesh-aware loader's ψ against ``want``."""
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader

    with WfnLoader(str(out), mesh=mesh) as reader:
        assert reader.backend == "phdf5", reader.backend
        psi = reader.load(bands=(0, case.NBANDS), k="ibz")
        full = np.zeros(psi.shape, dtype=np.complex128)
        for ik, w in enumerate(want):
            full[ik, :case.NBANDS, :, :w.shape[-1]] = w
        for shard in psi.addressable_shards:
            np.testing.assert_allclose(np.asarray(shard.data), full[shard.index],
                                       rtol=0, atol=1e-12)
        reader.close()


def main():
    from file_io import qp_wfn

    keep = "QPWFN_GATE_DIR" in os.environ
    root = os.environ.get("QPWFN_GATE_DIR") or os.path.join(
        os.environ.get("SCRATCH") or os.path.expanduser("~"), ".qpwfn_gate",
        f"{os.environ.get('SLURM_JOB_ID', 'nojob')}."
        f"{os.environ.get('SLURM_STEP_ID', os.getpid())}")
    mesh = resolve_mesh()
    p = int(mesh.devices.size)
    src = os.path.join(root, "WFN.h5")
    if process_rank() == 0:
        os.makedirs(root, exist_ok=True)
        case.write_source(src)
    barrier("qpwfn_gate.source")
    coeffs, U, E = case.write_source(None)
    want = case.expected_qp_coeffs(coeffs, U)
    whole = qp_wfn._coefficient_window
    for arm, window in (("one", whole),
                        ("several", lambda mesh, **_: 3 * int(mesh.devices.size))):
        qp_wfn._coefficient_window = window
        out = os.path.join(root, f"WFN_qp_{arm}_p{p}.h5")
        dft = case.write_qp(src, out, mesh, U, E)
        # Every rank checks: jit compiles are agreed across ranks
        # (common.jax_compile_cache), so rank-0-only jax work would hang.
        case.check_file(src, out, want, E, dft)
        barrier(f"qpwfn_gate.{arm}.h5py")
        readers = "h5py layout and eager reader"
        if p > 1:       # one device: the loader's own tier is eager (checked)
            _phdf5_reader_matches(out, want, mesh)
            barrier(f"qpwfn_gate.{arm}.phdf5")
            readers += " and phdf5 reader"
        if process_rank() == 0:
            print(f"QPWFN GATE arm={arm} P={p}: {readers} match", flush=True)
    qp_wfn._coefficient_window = whole
    barrier("qpwfn_gate.done")
    if process_rank() == 0:
        if not keep:
            import shutil
            shutil.rmtree(root, ignore_errors=True)
        print(f"QPWFN GATE PASS P={p}", flush=True)


if __name__ == "__main__":
    main()
