"""Gate: the GW restart reader is a TILE reader, and it is BIT-EXACT.

Why this gate exists
--------------------
``file_io.tagged_arrays.read_restart_state_from_h5`` used to read every
restart dataset with ``[:]`` — the whole ``(nq, μ, μ)`` ``V_qmunu``, the
whole ``S_qmunu``, the whole ``(μ, μ)`` ``V0_noG0_munu``, and the whole
``psi_full_y`` — into one array **on every rank**, and only then applied
``jnp.pad`` and ``with_sharding_constraint``.  Measured at P=4 (job
56389339, MoS2 6x6, N_mu=1496, nq=36): **+1.53 GiB VmHWM per rank**,
silently.  At the design envelope (N_mu=20000, nq=64) the same read is
**381.47 GiB per rank** (CLAIMS 69).

Because of that it was GUARDED OFF above one process — which removed a
capability that had worked at deck scale, and left ``restart = true`` with
no P>1 story at all.  It is now ported to the SlabIO tile path.  This gate
pins the two things that port has to be true about:

1. **REACHABILITY** — the reader runs at P>1 at all.  The old refusal
   raised ``RuntimeError`` naming the process count; if that comes back,
   this fails at ``load``.
2. **BIT EQUALITY** — every element the sharded reader returns is the
   element serial ``h5py`` reads from the same file at the same index.
   Not a tolerance: this is an element-SELECTION change, not a
   reduction-order one, so there is no eps floor to hide behind.  (The
   "~1.5 eps sharded/unsharded floor" quoted in earlier sessions does not
   reproduce; sharding here is bit-exact.)

Coverage is per SPINOR CLASS, because the ψ datasets differ only in the
extent of a replicated axis and a reader that mishandled it would produce
a correctly-shaped, wrongly-indexed array with no error anywhere:

* ``RESTART_NS=1`` — scalar
* ``RESTART_NS=2`` — spinor
* ``RESTART_NS=4`` — bispinor, which additionally writes
  ``psi_full_y_transverse`` at its OWN μ extent (the transverse centroid
  count differs from the charge one, so it carries its own pad).

Running it
----------
``resolve_mesh`` requires a perfect-square device count (1/4/9/16).

    RESTART_NS=2 <launcher> -n 4 python3 -m restart_sharded_parity
    RESTART_NS=4 <launcher> -n 4 python3 -m restart_sharded_parity

Exit 0 = pass.  Every assertion names the dataset and the first
disagreeing index, because "the restart is wrong" is otherwise a
multi-hour bisect.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from runtime import initialize_communicator_stack               # noqa: E402

RUNTIME = initialize_communicator_stack()

import h5py                                                     # noqa: E402
import numpy as np                                              # noqa: E402
import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

from common.collectives import (process_count, process_rank,    # noqa: E402
                                resolve_mesh, barrier)
from file_io.tagged_arrays import (write_restart_state_to_h5,   # noqa: E402
                                   load_restart_state_from_h5)
from runtime.padding import padded_mu_extent                    # noqa: E402

NS = int(os.environ.get("RESTART_NS", "2"))
PATH = os.environ.get("RESTART_PATH", "restart_parity_gate.h5")
NK = int(os.environ.get("RESTART_NK", "3"))
NB = int(os.environ.get("RESTART_NB", "4"))


def _fail(msg: str) -> None:
    sys.stderr.write(f"[restart_parity rank={process_rank()}] FAIL: {msg}\n")
    sys.stderr.flush()
    raise SystemExit(1)


def _cplx(rng, *shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def main() -> int:
    world = process_count()
    mesh = resolve_mesh()
    devices = int(jax.device_count())

    # N_mu deliberately NOT a multiple of the device count: the pad is the
    # thing under test.  ``+1`` guarantees a pad on every square P>1.
    n_mu = devices + 1
    n_mu_T = devices + 2 if NS == 4 else None
    n_q = 2
    n_mu_pad = padded_mu_extent(n_mu, devices)

    rng = np.random.default_rng(20260806 + NS)
    # Producers hand over PADDED buffers whose pad rows are exact zeros;
    # the writer clips them against the logical extent.  Build them that
    # way so this gate exercises the real call pattern.
    V = np.zeros((n_q, n_mu_pad, n_mu_pad), dtype=np.complex128)
    V[:, :n_mu, :n_mu] = _cplx(rng, n_q, n_mu, n_mu)
    V0 = np.zeros((n_mu_pad, n_mu_pad), dtype=np.complex128)
    V0[:n_mu, :n_mu] = _cplx(rng, n_mu, n_mu)
    G0 = np.zeros((n_mu_pad,), dtype=np.complex128)
    G0[:n_mu] = _cplx(rng, n_mu)
    psi = np.zeros((NK, NB, NS, n_mu_pad), dtype=np.complex128)
    psi[..., :n_mu] = _cplx(rng, NK, NB, NS, n_mu)
    enk = rng.standard_normal((NK, NB))

    psi_T = None
    if n_mu_T is not None:
        n_mu_T_pad = padded_mu_extent(n_mu_T, devices)
        psi_T = np.zeros((NK, NB, NS, n_mu_T_pad), dtype=np.complex128)
        psi_T[..., :n_mu_T] = _cplx(rng, NK, NB, NS, n_mu_T)

    if process_rank() == 0:
        print(f"[restart_parity] world={world} devices={devices} "
              f"mesh={mesh.devices.shape} nspinor={NS} n_mu={n_mu} "
              f"n_mu_pad={n_mu_pad} n_mu_T={n_mu_T}", flush=True)

    write_restart_state_to_h5(
        PATH, n_rmu_logical=n_mu,
        V_qmunu=jnp.asarray(V), V0_noG0_munu=jnp.asarray(V0),
        G0_mu_nu=jnp.asarray(G0), psi_full_y=jnp.asarray(psi),
        enk_full=jnp.asarray(enk),
        psi_full_y_transverse=(jnp.asarray(psi_T)
                               if psi_T is not None else None),
        n_rmu_transverse_logical=n_mu_T,
        mesh=mesh, mode="w", kgrid=(n_q, 1, 1),
    )
    barrier("restart_parity_written")

    # ---- (1) REACHABILITY: this is the call that used to refuse --------
    rs = load_restart_state_from_h5(PATH, mesh)

    # ---- (2) BIT EQUALITY against serial h5py on the same file ---------
    # Ground truth is the file, read whole, by a different library.  Every
    # sharded array is materialised here ONLY to compare -- that is what a
    # parity gate is for, and it is why this test uses a tiny n_mu.
    with h5py.File(PATH, "r") as f:
        disk = {k: np.asarray(f[k]) for k in f
                if isinstance(f[k], h5py.Dataset)}

    def check(name, got, want_disk, pad_to=None):
        """Compare THIS RANK'S TILE against the disk bytes it should hold.

        Per-shard on purpose, and not merely to avoid an error: at P>1
        ``jax.device_get`` on these arrays RAISES ("spans non-addressable
        devices"), which is itself the proof that the reader returned a
        distributed array rather than a replicated one. Comparing shard
        by shard also means this gate never materialises the global
        object either -- a parity test that violated the doctrine it is
        checking would be a poor gate.

        ``shard.index`` is a tuple of slices in GLOBAL coordinates, so
        ``want[shard.index]`` is exactly the block this rank must own.
        That is the whole assertion: right elements, right global
        position, bit for bit.
        """
        want = np.zeros(tuple(int(v) for v in got.shape),
                        dtype=want_disk.dtype)
        want[tuple(slice(0, s) for s in want_disk.shape)] = want_disk
        for sh in got.addressable_shards:
            local = np.asarray(sh.data)
            expect = want[sh.index]
            if local.shape != expect.shape:
                _fail(f"{name}: shard at {sh.index} has shape "
                      f"{local.shape}, expected {expect.shape}")
            if not np.array_equal(local, expect):
                bad = np.argwhere(local != expect)
                first = tuple(int(i) for i in bad[0])
                _fail(f"{name}: {len(bad)} element(s) of this rank's tile "
                      f"differ from the serial h5py read; shard index "
                      f"{sh.index}, first local offset {first}: "
                      f"sharded={local[first]!r} serial={expect[first]!r}")
            # The pad must be EXACT zeros, not merely small: downstream
            # code runs on the padded extent and a nonzero pad row is a
            # silent physics error (ROOT_CAUSE.md, device-invariance).
            if pad_to is not None:
                last = sh.index[-1]
                start = 0 if last.start is None else int(last.start)
                keep = max(0, int(pad_to) - start)
                tail = local[..., keep:]
                if tail.size and np.any(tail != 0):
                    _fail(f"{name}: pad rows past global {pad_to} are not "
                          f"exact zeros on this rank (max |pad| = "
                          f"{np.abs(tail).max()!r})")

    check("V_qmunu", rs.V_qmunu, disk["V_qmunu"], pad_to=n_mu)
    check("V0_noG0_munu", rs.V0_noG0_munu, disk["V0_noG0_munu"], pad_to=n_mu)
    check("G0_mu_nu", rs.G0_mu_nu, disk["G0_mu_nu"], pad_to=n_mu)
    check("psi_rmu_Y", rs.psi_rmu_Y, disk["psi_full_y"], pad_to=n_mu)
    check("enk_full", rs.enk_full, disk["enk_full"])

    # ψ spinor axis carried at its on-disk extent, never padded.
    if int(rs.psi_rmu_Y.shape[2]) != NS:
        _fail(f"psi_rmu_Y spinor axis is {rs.psi_rmu_Y.shape[2]}, expected "
              f"{NS} -- the spinor axis must ride through unpadded")

    # The X copy is conj + (nb, s, μ) -> (μ, nb, s); pin it against the
    # SAME disk bytes so a transpose bug cannot hide behind the Y copy.
    psi_disk_pad = np.zeros(tuple(int(v) for v in rs.psi_rmu_Y.shape),
                            dtype=np.complex128)
    psi_disk_pad[..., :disk["psi_full_y"].shape[-1]] = disk["psi_full_y"]
    want_X = np.conj(psi_disk_pad).transpose(0, 3, 1, 2)
    for sh in rs.psi_rmuT_X.addressable_shards:
        local = np.asarray(sh.data)
        expect = want_X[sh.index]
        if not np.array_equal(local, expect):
            bad = np.argwhere(local != expect)
            _fail(f"psi_rmuT_X: this rank's tile differs at shard index "
                  f"{sh.index}, local offset "
                  f"{tuple(int(i) for i in bad[0])}")

    if NS == 4:
        if rs.psi_rmu_Y_transverse is None:
            _fail("bispinor restart: psi_rmu_Y_transverse came back None")
        check("psi_rmu_Y_transverse", rs.psi_rmu_Y_transverse,
              disk["psi_full_y_transverse"], pad_to=n_mu_T)
        if int(rs.n_rmu_transverse_disk) != int(n_mu_T):
            _fail(f"n_rmu_transverse_disk={rs.n_rmu_transverse_disk} != "
                  f"{n_mu_T} -- the LOGICAL transverse extent must survive "
                  f"the port; it is what the loader re-pads from")
        if int(rs.psi_rmu_Y_transverse.shape[2]) != NS:
            _fail("transverse ψ spinor axis was padded")
    elif rs.psi_rmu_Y_transverse is not None:
        _fail(f"nspinor={NS} restart returned a transverse ψ")

    # ---- (3) the tile is a TILE: no rank holds the global array --------
    # A reader that regressed to ``[:]`` would still pass the checks above.
    # The sharding is the distinguishing observable.
    for name, arr in (("V_qmunu", rs.V_qmunu),
                      ("V0_noG0_munu", rs.V0_noG0_munu),
                      ("psi_rmu_Y", rs.psi_rmu_Y)):
        shards = arr.addressable_shards
        if world > 1 and len(shards) == 1:
            local = tuple(int(v) for v in shards[0].data.shape)
            if local == tuple(int(v) for v in arr.shape):
                _fail(f"{name}: this rank's shard is the WHOLE array "
                      f"{local} at world={world} -- the reader materialised "
                      f"the global object, which is the regression this "
                      f"gate exists for")

    barrier("restart_parity_done")
    if process_rank() == 0:
        print(f"[restart_parity] PASS nspinor={NS} world={world} "
              f"(bit-exact vs serial h5py, pad rows exact zero, tiles "
              f"per rank)", flush=True)
        try:
            os.remove(PATH)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
