"""Gate: the phdf5 collective writer must accept a WHOLLY-PADDED rank slice.

Why this gate exists
--------------------
Sharded producers pad ``mu`` up to a multiple of the world size; files store
the LOGICAL shape, and SlabIO clips the pad rows against it with no argument
from the caller (decisions.md 2026-08-04).  Rank ``r``
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

The three cases added by the 2026-08-04 SlabIO audit
----------------------------------------------------
The write path is only half the contract, and a bounds test that fires on a
PROPER SUBSET of ranks is the same class of defect one step out: the ranks
that do not refuse enter the collective alone and the job hangs with no
message.

    PADRANK_CASE=read_pad   <launcher> -n 4  python3 -m phdf5_padded_rank_write
    PADRANK_CASE=oob_write  <launcher> -n 4  python3 -m phdf5_padded_rank_write
    PADRANK_CASE=oob_read   <launcher> -n 4  python3 -m phdf5_padded_rank_write

    read_pad   mu = P+1.  Read at the PADDED physical shape with
               valid_shape = the logical extent, mu-sharded, so wholly-
               padded ranks take the empty-selection branch of the READ
               collective.  Every rank checks its own block; the padded
               ones must come back EXACTLY zero.  This is the shape
               ``zeta_loader.read_zeta_G_slab`` uses in production.
    oob_write  the LOGICAL slab overruns the dataset extent (offset 1 on
               the mu axis) while wholly-padded ranks exist.  rc=0 iff THIS
               rank refused.  Measured on the unfixed writer at P=4: rank 2
               alone reported oob=1, ranks 0/1/3 entered H5Dwrite, and the
               job died 306 s later at the JAX shutdown barrier deadline
               (job 7888470).
    oob_read   the same for the read path, which before the audit had no
               bounds test at all.  Unfixed at P=4 this HANGS with no
               message on any rank (job 7888470 p4_oobr, killed at 420 s).

The oob cases fail by HANGING, so run them under a wall-clock bound and
read each rank's own rc: the gate is "every rank refused", and a rank that
neither refused nor finished is the failure.  Since decisions.md
2026-08-04 both refusals fire in PYTHON (``_normalize_valid_shape``
bounds-tests the explicit override against the dataset's own extent
before any collective is entered); the C++ tests in ``write_ffi.cc`` /
``read_ffi.cc`` remain as the backstop for direct ``ffi.io`` users and
still take the verdict on the replicated logical slab.

The two cases added when padding became implicit (decisions.md 2026-08-04)
------------------------------------------------------------------------
    PADRANK_CASE=implicit  <launcher> -n 4  python3 -m phdf5_padded_rank_write
    PADRANK_CASE=nodiv     <launcher> -n 4  python3 -m phdf5_padded_rank_write

    implicit  BYTE-IDENTITY.  Write the same padded buffer twice at mu = P+1:
              once in the pre-ruling spelling (explicit ``global_shape`` +
              ``valid_shape``) and once as bare ``write_slab(name, A)``.  The
              two files' dataset bytes must be IDENTICAL, and so must the
              two spellings of the padded read.  This is the gate that
              proves the derived default is the same predicate the call
              sites used to compute by hand — if it ever fails, the sweep
              that deleted those arguments changed physics.
    nodiv     mu = P+1 requested WITHOUT rounding it up: a shape that does
              not divide the mesh under ``P(None,('x','y'),None)``.  The
              question the ruling raised is whether SlabIO should PAD such
              a request internally.  MEASURED (job 7888644, jax 0.9.1): it
              cannot help — ``device_put`` of that (shape, spec) pair
              raises ``IndivisibleError``, so the array the caller asked
              for cannot exist, and reading a rounded-up extent then
              trimming would only replace SlabIO's message with JAX's at
              the same refusal.  The case therefore asserts BOTH: that JAX
              refuses the array, and that ``read_slab`` refuses first, by
              name, on EVERY rank (it is a pure check on replicated
              inputs, so no collective is entered).  This is also what
              settles the audit's item 5: the phdf5_host backend's absent
              divisibility check is not a capability difference, because
              the case it would accept cannot be expressed.
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


_PAD_CASES = ("repro", "read_pad", "oob_write", "oob_read",
              "implicit", "nodiv")
_CONTROL_CASES = ("control", "exact")


def _mu_for(case: str, world: int) -> int:
    if case in _PAD_CASES:
        return world + 1
    if case == "control":
        return 2 * world - 1
    if case == "exact":
        return 2 * world
    raise SystemExit(
        f"unknown PADRANK_CASE={case!r} (expected "
        f"{' | '.join(_PAD_CASES + _CONTROL_CASES)})")


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
    if CASE in _PAD_CASES and not pure_pad:
        p0(f"[padrank-gate] REFUSING: case {CASE} produced no wholly-padded "
           f"rank, so it would gate nothing at this world size.")
        return 2
    if CASE in _CONTROL_CASES and pure_pad:
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

    # ---- oob_write: refuse BEFORE the file is written, so nothing else in
    # this launch depends on the outcome.  The verdict is per-rank: rc=0
    # iff THIS rank refused.  A rank that neither refuses nor returns is
    # the failure the case exists to catch, and it presents as a hang.
    if CASE == "oob_write":
        with SlabIO(PATH, mode="w", mesh=mesh,
                    backend=SlabIOBackend.PHDF5_FFI) as io:
            io.create_dataset("zeta_like", shape=(N_Q, mu, N_G),
                              dtype=jnp.complex128)
            p0(f"[padrank-gate] issuing the overrunning write: "
               f"offset=(0,1,0) valid=(.,{mu},.) vs extent {mu}")
            sys.stdout.flush()
            try:
                io.write_slab("zeta_like", A,
                              offset=(0, 1, 0),
                              global_shape=(N_Q, mu_pad, N_G),
                              valid_shape=(N_Q, mu, N_G))
                io.close()
            except Exception as exc:                          # noqa: BLE001
                print(f"[padrank-gate rank={rank}] REFUSED (correct): "
                      f"{type(exc).__name__}: {exc}", flush=True)
                return 0
        print(f"[padrank-gate rank={rank}] FAIL: the overrunning write was "
              f"ACCEPTED", flush=True)
        return 1

    # ---- implicit: the whole point of decisions.md 2026-08-04.  Two files,
    # same buffer, two spellings; the bytes must not differ.  Run BEFORE the
    # shared write below so this case owns both of its files outright.
    if CASE == "implicit":
        p_old = PATH + ".explicit"
        p_new = PATH + ".implicit"
        for p in (p_old, p_new):
            if rank == 0 and os.path.exists(p):
                os.remove(p)
        jax.experimental.multihost_utils.sync_global_devices("implicit_unlink")

        # (a) the pre-ruling spelling: state the logical extent three times.
        with SlabIO(p_old, mode="w", mesh=mesh,
                    backend=SlabIOBackend.PHDF5_FFI) as io:
            io.create_dataset("zeta_like", shape=(N_Q, mu, N_G),
                              dtype=jnp.complex128)
            io.write_slab("zeta_like", A,
                          offset=(0, 0, 0),
                          global_shape=(N_Q, mu, N_G),
                          valid_shape=(N_Q, mu, N_G))
        # (b) the ruling's spelling: state it once, to create_dataset.
        with SlabIO(p_new, mode="w", mesh=mesh,
                    backend=SlabIOBackend.PHDF5_FFI) as io:
            io.create_dataset("zeta_like", shape=(N_Q, mu, N_G),
                              dtype=jnp.complex128)
            io.write_slab("zeta_like", A)
        jax.experimental.multihost_utils.sync_global_devices("implicit_written")

        # Read both spellings back off the IMPLICIT file, padded + mu-sharded.
        with SlabIO(p_new, mode="r", mesh=mesh,
                    backend=SlabIOBackend.PHDF5_FFI) as io:
            r_exp = io.read_slab("zeta_like", shape=(N_Q, mu_pad, N_G),
                                 dtype=np.complex128, offset=(0, 0, 0),
                                 valid_shape=(N_Q, mu, N_G), mesh=mesh,
                                 partition_spec=P(None, ('x', 'y'), None))
            r_imp = io.read_slab("zeta_like", shape=(N_Q, mu_pad, N_G),
                                 dtype=np.complex128, offset=(0, 0, 0),
                                 mesh=mesh,
                                 partition_spec=P(None, ('x', 'y'), None))
        b_exp = np.asarray(r_exp.addressable_shards[0].data)
        b_imp = np.asarray(r_imp.addressable_shards[0].data)
        read_same = (b_exp.shape == b_imp.shape
                     and b_exp.tobytes() == b_imp.tobytes())
        start = int(r_imp.addressable_shards[0].index[1].start or 0)
        want = buf[:, start:start + b_imp.shape[1], :]
        read_right = (b_imp.shape == want.shape
                      and not int(np.count_nonzero(b_imp != want)))
        print(f"[padrank-gate rank={rank}] read: explicit==implicit="
              f"{read_same} matches_reference={read_right} "
              f"rows [{start},{start + b_imp.shape[1]})", flush=True)

        ok = read_same and read_right
        if rank == 0:
            import hashlib
            import h5py
            digests = {}
            for tag, p in (("explicit", p_old), ("implicit", p_new)):
                with h5py.File(p, "r") as f:
                    arr = np.asarray(f["zeta_like"])
                digests[tag] = (arr.shape,
                                hashlib.sha256(arr.tobytes()).hexdigest())
                if arr.shape != ref.shape or np.count_nonzero(arr != ref):
                    print(f"[padrank-gate] {tag} file does NOT match the "
                          f"reference", flush=True)
                    ok = False
            print(f"[padrank-gate] explicit  {digests['explicit'][0]} "
                  f"sha256={digests['explicit'][1][:16]}", flush=True)
            print(f"[padrank-gate] implicit  {digests['implicit'][0]} "
                  f"sha256={digests['implicit'][1][:16]}", flush=True)
            byte_identical = digests["explicit"] == digests["implicit"]
            print(f"[padrank-gate] VERDICT case=implicit mu={mu}: "
                  f"BYTE-IDENTICAL={byte_identical} "
                  f"{'PASS' if (ok and byte_identical) else 'FAIL'}",
                  flush=True)
            ok = ok and byte_identical
        sys.stdout.flush()
        return 0 if ok else 1

    with SlabIO(PATH, mode="w", mesh=mesh,
                backend=SlabIOBackend.PHDF5_FFI) as io:
        io.create_dataset("zeta_like", shape=(N_Q, mu, N_G),
                          dtype=jnp.complex128)
        io.write_slab("zeta_like", A)
    jax.experimental.multihost_utils.sync_global_devices("padrank_written")

    # ---- nodiv: ask for the LOGICAL mu extent under a spec that cannot
    # shard it.  Two assertions, in order: JAX cannot build the array, and
    # SlabIO refuses first and by name on every rank.
    if CASE == "nodiv":
        if mu % world == 0:
            p0(f"[padrank-gate] REFUSING: mu={mu} divides world={world}, so "
               f"case nodiv would gate nothing at this world size.")
            return 2

        # (1) Can JAX build a non-divisibly-sharded array at all?  This is
        # the measurement that decides whether an internal pad could ever
        # have helped.
        try:
            _probe = jax.device_put(
                jnp.zeros((N_Q, mu, N_G), dtype=jnp.complex128),
                NamedSharding(mesh, P(None, ('x', 'y'), None)))
            _probe.block_until_ready()
            jax_verdict = (f"BUILT (jax {jax.__version__}, local block "
                           f"{_probe.addressable_shards[0].data.shape})")
        except Exception as exc:                              # noqa: BLE001
            jax_verdict = (f"REFUSED (jax {jax.__version__}: "
                           f"{type(exc).__name__}: {str(exc)[:110]})")
        print(f"[padrank-gate rank={rank}] jax.device_put of "
              f"({N_Q},{mu},{N_G}) under P(None,('x','y'),None): "
              f"{jax_verdict}", flush=True)

        # (2) read_slab must refuse first, by name, on THIS rank.
        try:
            with SlabIO(PATH, mode="r", mesh=mesh,
                        backend=SlabIOBackend.PHDF5_FFI) as io:
                io.read_slab("zeta_like",
                             shape=(N_Q, mu, N_G),
                             dtype=np.complex128,
                             offset=(0, 0, 0),
                             mesh=mesh,
                             partition_spec=P(None, ('x', 'y'), None))
        except ValueError as exc:
            named = ("not divisible" in str(exc)
                     and f"{mu_pad}" in str(exc))
            print(f"[padrank-gate rank={rank}] REFUSED (correct), names the "
                  f"rounded-up extent={named}: {exc}", flush=True)
            return 0 if named else 1
        except Exception as exc:                              # noqa: BLE001
            print(f"[padrank-gate rank={rank}] FAIL: refused, but not by "
                  f"SlabIO — {type(exc).__name__}: {exc}", flush=True)
            return 1
        print(f"[padrank-gate rank={rank}] FAIL: the indivisible request "
              f"was ACCEPTED (jax verdict was: {jax_verdict})", flush=True)
        return 1

    # ---- read_pad: the READ-side empty rendezvous.  Physical shape is the
    # padded one, valid_shape is the logical file extent, mu is sharded —
    # so ranks in ``pure_pad`` select nothing and must come back at exactly
    # the zero fill.  Every rank checks its OWN block, so a rank that gets
    # someone else's data fails on that rank.
    if CASE == "read_pad":
        with SlabIO(PATH, mode="r", mesh=mesh,
                    backend=SlabIOBackend.PHDF5_FFI) as io:
            out = io.read_slab("zeta_like",
                               shape=(N_Q, mu_pad, N_G),
                               dtype=np.complex128,
                               offset=(0, 0, 0),
                               valid_shape=(N_Q, mu, N_G),
                               mesh=mesh,
                               partition_spec=P(None, ('x', 'y'), None))
        shard = out.addressable_shards[0]
        blk = np.asarray(shard.data)
        start = int(shard.index[1].start or 0)
        want = buf[:, start:start + blk.shape[1], :]
        bad = (blk.shape != want.shape
               or int(np.count_nonzero(blk != want)))
        print(f"[padrank-gate rank={rank}] block rows [{start},"
              f"{start + blk.shape[1]}) {'(WHOLLY PADDED)' if rank in pure_pad else ''}"
              f" mismatched={bad}", flush=True)
        return 0 if not bad else 1

    if CASE == "oob_read":
        with SlabIO(PATH, mode="r", mesh=mesh,
                    backend=SlabIOBackend.PHDF5_FFI) as io:
            p0(f"[padrank-gate] issuing the overrunning read: "
               f"offset=(0,1,0) valid=(.,{mu},.) vs extent {mu}")
            sys.stdout.flush()
            try:
                io.read_slab("zeta_like",
                             shape=(N_Q, mu_pad, N_G),
                             dtype=np.complex128,
                             offset=(0, 1, 0),
                             valid_shape=(N_Q, mu, N_G),
                             mesh=mesh,
                             partition_spec=P(None, ('x', 'y'), None))
            except Exception as exc:                          # noqa: BLE001
                print(f"[padrank-gate rank={rank}] REFUSED (correct): "
                      f"{type(exc).__name__}: {exc}", flush=True)
                return 0
        print(f"[padrank-gate rank={rank}] FAIL: the overrunning read was "
              f"ACCEPTED past the dataset extent", flush=True)
        return 1

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
