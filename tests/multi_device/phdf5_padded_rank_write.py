"""Gate: the phdf5 collective writer must accept a WHOLLY-PADDED rank slice.

Why this gate exists
--------------------
Sharded producers pad ``mu`` up to a multiple of the world size and pass the
logical prefix as ``valid_shape``; files store the LOGICAL shape.  Rank ``r``
owns local block ``[r*loc, (r+1)*loc)`` with ``loc = mu_padded / P``.  When
``r*loc >= mu`` that rank's block is ENTIRELY padding: C++ clips its file
hyperslab to ``file_count = 0`` and the write is meant to proceed through the
deliberate empty-selection rendezvous (``H5Sselect_none``) so the rank still
joins the collective ``H5Dwrite`` its peers are inside.

``write_ffi.cc`` used to bounds-test ``offset + file_count > extent`` BEFORE
consulting emptiness.  A wholly-padded rank carries an offset already advanced
past the logical extent, so the test rejected a write that would have been a
no-op, the rank returned without entering the collective, and the communicator
stranded.  Reachable only when ``pad >= loc`` -- i.e. when ``mu`` is more than
one rank-slice below its padded extent -- which is why it stayed hidden until
a production transverse channel hit it.

Cost of the miss: two 32-node bispinor GW legs (jobs 7885953 gw_bi and
gw_bi_m20, 2026-08-02) died at
``[SlabIO.close] draining 1 pending writes for zeta_q_mu1.h5`` with NO
traceback and NO diagnostic, because the writer's stderr is lost under
srun+apptainer at teardown (the same effect ``runtime.install_failfast_
excepthook`` documents).  The mechanism was only captured once a probe wrote
each rank's stderr to its own file -- so if this gate ever fails, READ THE
PER-RANK FILES, not the merged log.

Running it
----------
One case per process launch: after a failed collective the MPI/HDF5 state is
not safe to continue from, and the fail-fast excepthook ends the process
anyway.  ``PADRANK_CASE`` selects; each case derives its ``mu`` from the world
size, so the gate is meaningful at any square P.

    PADRANK_CASE=repro   <launcher> -n 4  python3 -m phdf5_padded_rank_write
    PADRANK_CASE=control <launcher> -n 4  python3 -m phdf5_padded_rank_write
    PADRANK_CASE=exact   <launcher> -n 4  python3 -m phdf5_padded_rank_write

    case      mu        wholly-padded ranks
    repro     P + 1     r >= ceil((P+1)/2)      <- the regression
    control   2P - 1    none (last rank owns 1 real row)
    exact     2P        none (no padding at all)

rc=0 iff the file round-trips to the exact reference.  P=1 is skipped: with
one rank there is no padding and nothing to test.
"""
import os
import sys

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402

from common.collectives import (process_count, process_rank,  # noqa: E402
                                resolve_mesh)
from file_io.slab_io import SlabIO                            # noqa: E402
from gw.gw_config import SlabIOBackend                        # noqa: E402

CASE = os.environ.get("PADRANK_CASE", "repro")
N_Q = int(os.environ.get("PADRANK_NQ", "2"))
N_G = int(os.environ.get("PADRANK_NG", "8"))
PATH = os.environ.get("PADRANK_PATH", "padrank_gate.h5")


def _mu_for(case: str, world: int) -> int:
    if case == "repro":
        return world + 1
    if case == "control":
        return 2 * world - 1
    if case == "exact":
        return 2 * world
    raise SystemExit(f"unknown PADRANK_CASE={case!r} "
                     f"(expected repro | control | exact)")


def main():
    rank, world = process_rank(), process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)

    if world < 2:
        p0("[padrank-gate] SKIP: needs P>1 (a single rank never pads)")
        return 0

    mu = _mu_for(CASE, world)
    mesh = resolve_mesh()
    loc = -(-mu // world)
    mu_pad = loc * world
    pure_pad = [r for r in range(world) if r * loc >= mu]

    p0(f"[padrank-gate] case={CASE} world={world} "
       f"mesh={tuple(mesh.devices.shape)} mu={mu} mu_padded={mu_pad} "
       f"loc={loc} pad_rows={mu_pad - mu}")
    p0(f"[padrank-gate] wholly-padded ranks: "
       f"{pure_pad if pure_pad else 'none'}")
    if CASE == "repro" and not pure_pad:
        p0("[padrank-gate] REFUSING: the repro case produced no wholly-padded "
           "rank, so it would gate nothing at this world size.")
        return 2
    if CASE != "repro" and pure_pad:
        p0(f"[padrank-gate] REFUSING: control case unexpectedly has "
           f"wholly-padded ranks {pure_pad}.")
        return 2
    sys.stdout.flush()

    # Distinct value per element, so a mis-placed hyperslab shows up as a
    # wrong value rather than a coincidentally-equal one.  The padded tail is
    # zero, matching the real zeta buffer (L_q's pad block is identity).
    ref = (np.arange(N_Q * mu * N_G, dtype=np.float64).reshape(N_Q, mu, N_G)
           + 1.0).astype(np.complex128)
    ref = ref + 1j * ref
    buf = np.zeros((N_Q, mu_pad, N_G), dtype=np.complex128)
    buf[:, :mu, :] = ref

    A = jax.device_put(jnp.asarray(buf),
                       NamedSharding(mesh, P(None, ('x', 'y'), None)))

    if rank == 0 and os.path.exists(PATH):
        os.remove(PATH)
    jax.experimental.multihost_utils.sync_global_devices("padrank_unlink")

    with SlabIO(PATH, mode="w", mesh=mesh,
                backend=SlabIOBackend.PHDF5_FFI) as io:
        io.create_dataset("zeta_like", shape=(N_Q, mu, N_G),
                          dtype=jnp.complex128)
        io.write_slab("zeta_like", A,
                      offset=(0, 0, 0),
                      global_shape=(N_Q, mu, N_G),
                      valid_shape=(N_Q, mu, N_G))
    jax.experimental.multihost_utils.sync_global_devices("padrank_written")

    ok = 1
    if rank == 0:
        import h5py
        with h5py.File(PATH, "r") as f:
            got = np.asarray(f["zeta_like"])
        if got.shape != ref.shape:
            print(f"[padrank-gate] SHAPE MISMATCH got {got.shape} "
                  f"want {ref.shape}")
            ok = 0
        else:
            bad = int(np.count_nonzero(got != ref))
            print(f"[padrank-gate] round-trip: {bad} mismatched of {ref.size}"
                  f"; max|delta| = {float(np.abs(got - ref).max()):.3e}")
            ok = int(bad == 0)
        print(f"[padrank-gate] VERDICT case={CASE} mu={mu}: "
              f"{'PASS' if ok else 'FAIL'}")
    sys.stdout.flush()
    return 0 if (rank != 0 or ok) else 1


if __name__ == "__main__":
    finalize_process(main())
