"""Canonical restart-state I/O for GW/BSE workflows.

This module reads/writes HDF5 restart files in the v2 format used by gw_jax.
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import h5py
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import barrier


def _mu_logical_shape(shape, mu_axes, n_rmu_logical):
    """On-disk (logical) shape for a μ-padded in-memory array: clip the
    ``mu_axes`` extents of ``shape`` to ``n_rmu_logical``.

    Disk contract (SHARDING_RULES §2): files store the LOGICAL μ extent
    so a restart written at any device count re-reads on any other; the
    in-memory pad (``Meta.n_rmu_padded``, zero rows by construction) is
    re-applied on read via ``runtime.padding.padded_mu_extent``.
    """
    out = [int(s) for s in shape]
    for ax in mu_axes:
        out[ax] = min(out[ax], int(n_rmu_logical))
    return tuple(out)


def _restart_write_log_on() -> bool:
    """Gate for the per-dataset liveness telemetry (scorecard AF.4c).

    ON by default, rank 0 only: this module writes tens of GB at
    production scale (~40 GB at MoS2 12x12/c2406) and used to print
    NOTHING while doing it — job 7876423 spent 2 h 55 m in here and was
    wall-clock-killed with no way to tell "slow" from "wedged".  File
    size is the trap AC.3b documents (parallel HDF5 pre-allocates at
    create time, so the file reaches full size before any data lands);
    one line per dataset with its own elapsed time is the cheapest
    thing that makes this stage diagnosable while alive.
    ``LORRAX_RESTART_WRITE_LOG=0`` silences it.
    """
    return (os.environ.get("LORRAX_RESTART_WRITE_LOG", "1")
            not in ("0", "", "false")
            and jax.process_index() == 0)


def _log_restart_write(name, shape, dtype, dt) -> None:
    """One ``[restart_write]`` line for a dataset handed to the writer.

    Single implementation on purpose: the AF.4c line format is
    load-bearing for log diagnosis, and this module used to carry four
    hand-synced copies of it (audit 2026-07-28; QUALITY_PATTERNS #3).
    Gated by :func:`_restart_write_log_on`.

    ``dt`` is the CALLER's elapsed time, and SlabIO's write path returns
    as soon as the tile is queued on its writer thread — so ``dt`` is the
    enqueue, not the transfer, and this line must not present it as one.
    It used to print ``nb/dt`` as a bandwidth, which reports thousands of
    MB/s for a dispatch that has moved nothing.  Worse, the flush it hid
    did not vanish: ``create_dataset`` drains the writer before H5Dcreate
    (the MPI datatype-cache interleave), so each dataset's ``dt`` is
    mostly the PREVIOUS dataset's transfer, and a multi-GB tensor's write
    surfaced against the scalar or header array logged after it, at an
    apparent 0 MB/s.  Both halves of that read as a writer pathology and
    neither is one.  So the size and the rate come from the
    ``SlabIO.close`` drain line, which is the first moment any of the
    bytes are on disk, and ``dt`` is reported here named for what it is.
    """
    if not _restart_write_log_on():
        return
    nb = int(np.prod(shape)) * int(np.dtype(dtype).itemsize)
    print(f"  [restart_write] {name} {tuple(int(v) for v in shape)}"
          f" {nb / 1e9:.2f} GB QUEUED in {dt:.1f} s"
          f" (transfer time and rate: see the SlabIO.close drain line)",
          flush=True)


def write_restart_state_to_h5(
    filename,
    *,
    n_rmu_logical: int,
    V_qmunu=None,
    psi_full_y=None,
    psi_full_y_transverse=None,
    n_rmu_transverse_logical: int | None = None,
    enk_full=None,
    S_qmunu=None,
    V0_noG0_munu=None,
    G0_mu_nu=None,
    W0_qmunu=None,
    init_W0: bool = False,
    mesh=None,
    mode: str = "w",
    kgrid: tuple[int, int, int] | None = None,
    band_slices=None,
):
    """Write (subset of) canonical restart state via SlabIO.

    All array arguments are optional — only the provided ones are
    written, so this function can be called multiple times to flush
    pieces of the restart state as they become available.  With
    ``mode="w"`` the file is truncated first (and the format-version
    attribute written); with ``mode="a"`` the file is opened for
    append / overwrite of the named datasets.

    ``n_rmu_logical`` (= ``meta.n_rmu``) is the μ extent stated on
    disk; SlabIO clips the in-memory pad rows against it.  In-memory arrays
    carry the P-dependent padded extent ``meta.n_rmu_padded`` whose pad
    rows are exact zeros, and persisting them verbatim would make the
    restart file unreadable at a different device count (the
    ROOT_CAUSE.md defect class, one hop downstream).
    ``load_restart_state_from_h5`` re-pads on read.

    ``init_W0=True`` pre-allocates an all-zeros W0_qmunu dataset sized
    from ``V_qmunu``; the ``W0_ready`` attr on that dataset is set to
    False so downstream readers (bse_io) know to treat it as a
    placeholder.  Passing ``W0_qmunu`` directly flips ``W0_ready`` to
    True.

    ``psi_full_y_transverse`` (bispinor only) is the σ^B-side ψ sampled
    at the TRANSVERSE centroid set — the per-channel second ψ dataset
    the bispinor restart round-trip needs.  Its μ axis is clipped by
    ``n_rmu_transverse_logical`` (the transverse centroid count, which
    differs from the charge ``n_rmu_logical``).  The count is also
    stamped as the ``n_rmu_transverse_logical`` dataset:
    ``read_restart_state_from_h5`` cross-checks it against the stored
    dataset's μ extent at load (torn/hand-edited-file guard); the
    loader itself re-pads from the dataset shape
    (``load_restart_state_from_h5`` → ``n_rmu_transverse_disk``).
    """
    from .slab_io import SlabIO

    with SlabIO(filename, mode=mode, mesh=mesh,
) as io:
        if mode == "w":
            io.write_attr("restart_format_version", np.int64(2))
        # kgrid attr lets BSE recover the (nkx,nky,nkz) split from
        # flat-q V_qmunu / W0_qmunu without re-opening the WFN.  Stored
        # as a length-3 int64 dataset (the SlabIO ``write_attr`` path
        # accepts list/tuple).  Optional: callers that don't pass it
        # leave the attr unset; BSE falls back to reading WFN.
        if kgrid is not None and mode == "w":
            io.write_attr("kgrid", np.asarray(kgrid, dtype=np.int64))
        # BAND-WINDOW PROVENANCE.  V_qmunu / psi_full_y / enk_full are all
        # indexed by the band window they were BUILT under; a restart that
        # changes nval/ncond/nband re-reads them under a different window and
        # silently misindexes Sigma -- no crash, just wrong physics (job
        # 7874375: window 70 tensors reused at window 80 gave a QP gap of
        # -135 eV while every stage reported success).  Stamp the window here
        # so :func:`assert_restart_window_matches` can refuse that on load.
        if band_slices is not None and mode == "w":
            io.write_attr("band_window", np.asarray(
                [int(band_slices.b0), int(band_slices.b1),
                 int(band_slices.b2), int(band_slices.b3),
                 int(band_slices.b4)], dtype=np.int64))
        if mode == "w":
            io.write_attr("n_rmu_logical", np.int64(int(n_rmu_logical)))

        # PER-DATASET LIVENESS (scorecard AF.4c): every dataset below
        # emits one [restart_write] line naming its size as it is handed
        # to the writer thread.  The transfer itself is asynchronous and
        # is timed where it completes, on the SlabIO.close drain line —
        # rationale and the LORRAX_RESTART_WRITE_LOG gate live at
        # :func:`_restart_write_log_on` / :func:`_log_restart_write`.

        def _write(name, arr, mu_axes=(), n_logical=None):
            """create+write one dataset, μ axes clipped to ``n_logical``
            (default: the charge ``n_rmu_logical``), with the AF.4c
            telemetry line.  Single write path for every dataset in
            this file, including the transverse ψ and the real W0
            (audit 2026-07-28 — the transverse block used to be an
            inline copy of this helper)."""
            if arr is None:
                return
            n_log = n_rmu_logical if n_logical is None else n_logical
            shape = _mu_logical_shape(arr.shape, mu_axes, n_log)
            _t0 = time.time()
            # The LOGICAL shape is stated once, to create_dataset; the
            # write clips ``arr``'s μ pad rows against it on its own
            # (decisions.md 2026-08-04).
            io.create_dataset(name, shape=shape, dtype=arr.dtype)
            io.write_slab(name, arr)
            _log_restart_write(name, shape, arr.dtype, time.time() - _t0)

        _write("V_qmunu",      V_qmunu,      mu_axes=(-2, -1))
        _write("S_qmunu",      S_qmunu,      mu_axes=(-2, -1))
        _write("V0_noG0_munu", V0_noG0_munu, mu_axes=(-2, -1))
        _write("G0_mu_nu",     G0_mu_nu,     mu_axes=(-1,))
        _write("psi_full_y",   psi_full_y,   mu_axes=(-1,))
        _write("enk_full",     enk_full)

        # Bispinor per-channel ψ: μ axis clipped to the TRANSVERSE
        # logical extent (its own centroid count, not n_rmu_logical).
        if psi_full_y_transverse is not None:
            if n_rmu_transverse_logical is None:
                raise ValueError(
                    "write_restart_state_to_h5: psi_full_y_transverse "
                    "requires n_rmu_transverse_logical (the transverse "
                    "centroid count) to clip its μ axis on disk.")
            n_T = int(n_rmu_transverse_logical)
            _write("psi_full_y_transverse", psi_full_y_transverse,
                   mu_axes=(-1,), n_logical=n_T)
            # Stamped for the load-time extent cross-check in
            # read_restart_state_from_h5.
            io.write_attr("n_rmu_transverse_logical", np.int64(n_T))

        # W0_qmunu: either write the real data or pre-allocate an
        # all-zeros placeholder.
        w0_touched = W0_qmunu is not None or init_W0
        w0_ready = False
        if W0_qmunu is not None:
            _write("W0_qmunu", W0_qmunu, mu_axes=(-2, -1))
            w0_ready = True
        elif init_W0:
            if V_qmunu is None:
                raise ValueError("init_W0=True requires V_qmunu to size the placeholder")
            v_shape = _mu_logical_shape(V_qmunu.shape, (-2, -1), n_rmu_logical)
            v_dtype = V_qmunu.dtype
            _t0 = time.time()
            io.create_dataset("W0_qmunu", shape=v_shape, dtype=v_dtype)
            if _restart_write_log_on():
                # Allocation ONLY -- no data is written here, so this
                # deliberately does NOT use the _log_restart_write
                # completed-write format.  Naming that explicitly
                # matters: under parallel HDF5 this call makes the file
                # jump by the full tensor size, which reads exactly like
                # progress and is not (AC.3b).
                _nb = int(np.prod(v_shape)) * int(np.dtype(v_dtype).itemsize)
                print(f"  [restart_write] W0_qmunu placeholder ALLOCATED "
                      f"{tuple(int(v) for v in v_shape)} {_nb / 1e9:.2f} GB "
                      f"in {time.time() - _t0:.1f} s (no data written)",
                      flush=True)

    # bse_io.py reads W0_ready as an HDF5 attr on the W0_qmunu dataset.
    # Set it rank-0-only after SlabIO has released the file, to stay
    # compatible with that reader.
    if w0_touched and jax.process_index() == 0:
        with h5py.File(filename, "a") as f:
            f["W0_qmunu"].attrs["W0_ready"] = w0_ready
    barrier("restart_W0_ready_flag")


def write_w0_qmunu_to_h5(
    filename, W0_qmunu, *, n_rmu_logical: int, mesh=None,
):
    """Overwrite or append the W0_qmunu dataset in an existing restart file.

    ``n_rmu_logical`` clips the trailing (μ, μ) axes to the logical
    on-disk extent — same contract as ``write_restart_state_to_h5``.
    """
    from .slab_io import SlabIO

    shape = _mu_logical_shape(W0_qmunu.shape, (-2, -1), n_rmu_logical)
    with SlabIO(filename, mode="a", mesh=mesh,
) as io:
        _t0 = time.time()
        io.create_dataset("W0_qmunu", shape=shape, dtype=W0_qmunu.dtype)
        io.write_slab("W0_qmunu", W0_qmunu)
        # Same instrument as ``write_restart_state_to_h5`` (AF.4c).  This
        # is the SECOND (nq, mu, mu) tensor the run writes -- another
        # 13.34 GB at c2406 -- and it had no telemetry at all, so a repeat
        # of the writer pathology would have been invisible here even
        # after AF instrumented its sibling.
        _log_restart_write("W0_qmunu", shape, W0_qmunu.dtype,
                           time.time() - _t0)

    # W0_ready flag is a per-dataset attr read by bse_io.py.
    if jax.process_index() == 0:
        with h5py.File(filename, "a") as f:
            f["W0_qmunu"].attrs["W0_ready"] = True
    barrier("restart_W0_ready_flag")


def write_head_scalars_to_h5(
    filename: str,
    *,
    vhead: complex | None = None,
    whead: np.ndarray | jnp.ndarray | None = None,
    omega_grid: np.ndarray | jnp.ndarray | None = None,
):
    """Persist q=0 Coulomb head scalars to the restart file.

    Stored alongside ``G0_mu_nu``; consumed by ``bse_io._load_ring_subset``
    (and any future Σ-builder) via ``head_correction.apply_q0_head_rank1``.

    - ``vhead``: scalar v(q→0, G=G'=0) in Ry, BGW convention.
    - ``whead``: shape ``(n_omega,)``. Length 1 for static COHSEX,
      length 2 for GN-PPM (static, iω_p).
    - ``omega_grid``: optional ``(n_omega,)`` array of the ω values
      (in Ry) corresponding to ``whead`` — written as an attribute on
      the ``whead`` dataset for consumer interpretation.

    Rank-0-only write (these are tiny; no MPI-IO needed).
    """
    if jax.process_index() != 0:
        barrier("restart_head_scalars")
        return
    with h5py.File(filename, "a") as f:
        if vhead is not None:
            if "vhead" in f:
                del f["vhead"]
            f.create_dataset("vhead", data=np.complex128(vhead))
        if whead is not None:
            if "whead" in f:
                del f["whead"]
            arr = np.asarray(whead, dtype=np.complex128).reshape(-1)
            ds = f.create_dataset("whead", data=arr)
            if omega_grid is not None:
                ds.attrs["omega_grid"] = np.asarray(omega_grid, dtype=np.float64).reshape(-1)
    barrier("restart_head_scalars")


def assert_restart_window_matches(filename, band_slices=None,
                                 n_rmu_logical=None) -> None:
    """Refuse a restart whose tensors were built under a DIFFERENT band
    window or centroid count.

    ``V_qmunu``, ``psi_full_y`` and ``enk_full`` are all indexed by the band
    window in force when they were written.  Reusing them under a changed
    ``nval``/``ncond``/``nband`` misindexes Sigma with no shape error and no
    crash — job 7874375 reused window-70 tensors at window 80 and produced a
    QP gap of -135 eV while every stage reported success.  This turns that
    into a loud, actionable failure naming BOTH windows.

    Files written before this attr existed carry no ``band_window``; those
    are passed through with no check (back-compat), since refusing them would
    strand existing restart files.
    """
    with h5py.File(filename, "r") as f:
        stored_w = np.asarray(f["band_window"]).tolist() if "band_window" in f else None
        stored_mu = (int(np.asarray(f["n_rmu_logical"])[()])
                     if "n_rmu_logical" in f else None)

    if stored_w is not None and band_slices is not None:
        want = [int(band_slices.b0), int(band_slices.b1), int(band_slices.b2),
                int(band_slices.b3), int(band_slices.b4)]
        if [int(x) for x in stored_w] != want:
            raise ValueError(
                f"Restart file {filename} was written under band window "
                f"(b0,b1,b2,b3,b4)={tuple(int(x) for x in stored_w)} but this "
                f"run has {tuple(want)}.  V_qmunu / psi_full_y / enk_full are "
                f"indexed by that window, so reusing them would MISINDEX "
                f"Sigma silently (no crash, wrong QP energies -- see job "
                f"7874375).  Either restore the original nval/ncond/nband, or "
                f"set restart=false to rebuild the tensors for the new window."
            )
    if stored_mu is not None and n_rmu_logical is not None:
        if int(stored_mu) != int(n_rmu_logical):
            raise ValueError(
                f"Restart file {filename} was written with n_rmu={stored_mu} "
                f"but this run has n_rmu={int(n_rmu_logical)}.  The ISDF basis "
                f"differs, so V_qmunu / psi_full_y are not reusable.  Set "
                f"restart=false (or point at the matching centroid file)."
            )


def _munu_slab_request(ds_shape, n_rmu_pad):
    """``(offset, shape, spec)`` for a (…, μ, ν) restart tensor.

    The μ/ν axes are ALWAYS the trailing two — that is the disk contract
    for ``V_qmunu`` / ``S_qmunu`` / ``V0_noG0_munu`` / ``W0_qmunu``, and
    it holds across all three historical layouts because the leading axes
    only ever grew in front:

      * 3-D flat-q ``(nq, μ, ν)``            — what the current pipeline writes
      * 6-D transitional ``(1, npol, npol, nq, μ, ν)``
      * 8-D legacy ``(1, npol, npol, nkx, nky, nkz, μ, ν)``

    So one rule covers all three: shard the last two axes on ('x', 'y'),
    replicate everything in front, and read the leading ``(1, npol, npol)``
    block at index 0 with extent 1 — which is exactly the ``[0, 0, 0]``
    the whole-file reader used to take AFTER materialising the array.

    ``bse_io._resolve_munu_reader`` / ``_MunuSlabPlan`` state the same
    three layouts for the BSE consumer, which additionally needs to
    select a single q; this one never does, so it does not need the
    kgrid.  If a third consumer appears, the layout fact should move here
    (L3) rather than be stated a third time.
    """
    ndim = len(ds_shape)
    if ndim < 2:
        raise ValueError(
            f"restart (μ, ν) tensor has rank {ndim} (shape "
            f"{tuple(ds_shape)}); it must have at least the two trailing "
            f"μ/ν axes.")
    # ONLY the two legacy layouts carry the ``(1, npol, npol)`` prefix that
    # has to be read at index 0.  Everything else -- 2-D ``V0_noG0_munu``,
    # 3-D flat-q ``V_qmunu``, 5-D ``S_qmunu`` -- keeps every leading axis
    # whole, so the rank does not need enumerating and a new leading axis
    # does not need a code change.  (Enumerating it DID cost a bug: the
    # first version of this listed 3/5/6/8 and refused the 2-D
    # ``V0_noG0_munu``, caught by restart_sharded_parity at P=4.)
    lead = 3 if ndim in (6, 8) else 0
    offset = [0] * ndim
    shape = [1] * lead + [int(v) for v in ds_shape[lead:-2]]
    shape += [int(n_rmu_pad), int(n_rmu_pad)]
    spec = P(*([None] * (ndim - 2) + ["x", "y"]))
    return tuple(offset), tuple(shape), spec


def _collapse_leading(A, ds_shape, mesh_xy):
    """Drop the legacy ``(1, npol, npol)`` prefix, flattening q if present.

    Local: the leading axes are extent-1 or replicated and μ/ν keep their
    ('x', 'y') sharding across the reshape, so GSPMD moves nothing.  The
    output sharding is PINNED rather than inferred — an inferred
    resharding of an N_mu²-class object would be a silent all-to-all,
    which is the one thing this reader must never do.  Same reasoning,
    and the same spelling, as ``bse_io._slabio_read_munu``.

    Only the 6-D/8-D legacy layouts reach the reshape at all; the 3-D
    flat-q form the current pipeline writes passes straight through.
    """
    ndim = len(ds_shape)
    if ndim not in (6, 8):            # 3-D flat-q, or S_qmunu's own 5-D
        return A
    mu, nu = int(A.shape[-2]), int(A.shape[-1])
    return jax.jit(
        lambda a: jnp.reshape(a, (-1, mu, nu)),
        out_shardings=NamedSharding(mesh_xy, P(None, "x", "y")))(A)


def _check_nspinor(nspinor: int, where: str) -> int:
    """Gate the ψ spinor axis.  1 (scalar), 2 (spinor), 4 (bispinor).

    THE GATE THE PAD AUDIT ASKED FOR.  The μ axis of every restart tensor
    is padded to a mesh-divisible extent and the pad rows are exact zeros
    by construction; the SPINOR axis is not padded by anything here and
    must not be.  It is read at its on-disk extent and carried through
    replicated, so a 2-component spinor restart and a 4-component
    bispinor restart differ only in that extent.

    The failure this refuses is a file whose spinor axis is neither —
    which, unchecked, would sail through as a perfectly shardable
    replicated axis and misindex every downstream ψ contraction with no
    shape error.  ``nspinor`` is small and replicated, so there is no
    cost to checking it.
    """
    ns = int(nspinor)
    if ns not in (1, 2, 4):
        raise ValueError(
            f"{where}: ψ spinor axis has extent {ns}; expected 1 (scalar), "
            f"2 (spinor) or 4 (bispinor).  The restart file is not one this "
            f"pipeline wrote — regenerate it (restart = false).")
    return ns


def read_restart_state_from_h5(filename, mesh_xy):
    """Read canonical restart state as PER-RANK TILES (restart format v2).

    Returns arrays that are ALREADY sharded on ``mesh_xy`` and ALREADY at
    the padded μ extent.  Nothing larger than one rank's tile is
    materialised at any point, so this works at any process count.

    WHAT THIS REPLACED, AND WHY IT MATTERS
    --------------------------------------
    Until 2026-08-06 this was a full-file reader — every dataset read with
    ``[:]`` into one array on the calling process, ``V_qmunu`` and
    ``S_qmunu`` at ``(nq, μ, μ)`` and ``V0_noG0_munu`` at ``(μ, μ)``, all
    N_mu²-class, whole, on every rank — and it was GUARDED OFF above one
    process with no replacement, which removed a capability that had
    worked at deck scale.  The guard was correct about the cost and wrong
    to be the end of the story: MEASURED at P=4 (job 56389339, MoS2 6x6,
    N_mu=1496, nq=36) the old path cost **+1.53 GiB of VmHWM on every
    rank** (0.95 → 2.47 GiB) and nothing warned; at the design envelope
    (N_mu=20000, nq=64) the same read is **381.47 GiB per rank**
    (CLAIMS 69).

    The port follows ``bse_io.load_bse_data_from_restart_sharded``: read
    the SHAPES and the small replicated metadata with serial h5py, close
    that handle, then move the big tensors through SlabIO under
    collective MPI-IO.  Two live handles on one file is a hazard nobody
    needs, which is why the h5py block closes before SlabIO opens.

    WHAT STAYS ON SERIAL h5py, AND WHY IT IS NOT A LOOPHOLE
    -------------------------------------------------------
    ``enk_full`` ``(nk, nb)``, ``G0_mu_nu`` ``(μ,)``, and the scalar
    stamps.  These are μ-class or smaller — ``G0`` is 320 KB at the
    envelope — and every rank needs all of them.  The doctrine forbids
    materialising an N_mu²-class object, not reading a vector; sharding
    ``G0`` would buy nothing and cost a reshard at every use, and
    ``bse_io`` reads it exactly the same way.

    PADDING.  Disk stores the LOGICAL μ extent so a restart written at
    any device count re-reads at any other (SHARDING_RULES §2).  The
    in-memory convention is ``padded_mu_extent(n_rmu, device_count)``,
    and the pad rows are exact zeros.  Both facts are now enforced by the
    same mechanism: SlabIO is asked for the PADDED shape and zero-fills
    everything past the dataset, so the pad is the read, and there is no
    ``jnp.pad`` and no ``with_sharding_constraint`` applied to an
    already-resident global array.

    SPINOR AND BISPINOR.  ``nspinor`` is read from the ψ dataset and
    carried through as a replicated axis at its on-disk extent — 2 for a
    spinor restart, 4 for a bispinor one — and gated by
    :func:`_check_nspinor`.  It is never padded.  The bispinor
    ``psi_full_y_transverse`` is read at its OWN μ extent (the transverse
    centroid count differs from the charge one) with its own pad.
    """
    from .slab_io import SlabIO
    from runtime.padding import padded_mu_extent
    from common.collectives import device_put_process_local

    divisor = int(jax.device_count())

    # ---- pass 1: geometry + the small replicated arrays, serial h5py ----
    with h5py.File(filename, "r") as f:
        if "psi_full_y" not in f:
            raise ValueError(
                f"Restart file {filename} is missing canonical psi_full_y "
                "dataset. Regenerate restart tensors with current gw_jax.")
        shapes = {k: tuple(int(s) for s in f[k].shape)
                  for k in ("V_qmunu", "S_qmunu", "V0_noG0_munu",
                            "psi_full_y", "psi_full_y_transverse")
                  if k in f}
        dtypes = {k: f[k].dtype for k in shapes}
        enk_full = (np.asarray(f["enk_full"][:]) if "enk_full" in f else None)
        G0_mu_nu = (np.asarray(f["G0_mu_nu"][:]) if "G0_mu_nu" in f else None)
        stored_T = (int(np.asarray(f["n_rmu_transverse_logical"])[()])
                    if "n_rmu_transverse_logical" in f else None)

    n_rmu_disk = int(shapes["V_qmunu"][-1])
    n_rmu_pad = padded_mu_extent(n_rmu_disk, divisor)
    nspinor = _check_nspinor(shapes["psi_full_y"][2],
                             f"read_restart_state_from_h5({filename})")

    # Integrity cross-check of the stamped transverse extent against the
    # dataset it describes (audit 2026-07-28: the stamp used to be
    # write-only shadow metadata, QUALITY_PATTERNS #3 — a mismatch means a
    # torn or hand-edited file and must refuse loudly rather than feed
    # downstream re-padding, #7).  Checked on the SHAPE, before any bytes
    # move, so a bad file costs nothing.
    if "psi_full_y_transverse" in shapes and stored_T is not None:
        disk_T = int(shapes["psi_full_y_transverse"][-1])
        if stored_T != disk_T:
            raise ValueError(
                f"Restart file {filename}: stamped "
                f"n_rmu_transverse_logical={stored_T} does not match the "
                f"psi_full_y_transverse μ extent on disk ({disk_T}).  The "
                f"file is internally inconsistent (torn write or "
                f"hand-edited) — regenerate the restart tensors "
                f"(restart=false).")

    # ---- pass 2: the N_mu²-class and ψ tensors, one tile per rank -------
    psi_spec = P(None, None, None, "y")

    def _read_munu(io, name):
        if name not in shapes:
            return None
        off, shape, spec = _munu_slab_request(shapes[name], n_rmu_pad)
        arr = io.read_slab(name, shape=shape, dtype=dtypes[name],
                           offset=off, mesh=mesh_xy, partition_spec=spec)
        return _collapse_leading(arr, shapes[name], mesh_xy)

    def _read_psi(io, name, n_mu_logical):
        if name not in shapes:
            return None
        ds = shapes[name]
        _check_nspinor(ds[2], f"{name} in {filename}")
        pad = padded_mu_extent(int(n_mu_logical), divisor)
        return io.read_slab(
            name, shape=(int(ds[0]), int(ds[1]), int(ds[2]), int(pad)),
            dtype=dtypes[name], mesh=mesh_xy, partition_spec=psi_spec)

    n_rmu_T_disk = (int(shapes["psi_full_y_transverse"][-1])
                    if "psi_full_y_transverse" in shapes else None)

    with SlabIO(filename, mode="r", mesh=mesh_xy) as io:
        V_qmunu = _read_munu(io, "V_qmunu")
        S_qmunu = _read_munu(io, "S_qmunu")
        V0_noG0_munu = _read_munu(io, "V0_noG0_munu")
        psi_full_y = _read_psi(io, "psi_full_y", n_rmu_disk)
        psi_full_y_transverse = (
            _read_psi(io, "psi_full_y_transverse", n_rmu_T_disk)
            if n_rmu_T_disk is not None else None)

    # G0: μ-class, read whole above.  Collapse a legacy 2-D (nqz, μ) store
    # to its q=0 row, pad to the same in-memory μ extent as everything
    # else, and pin it to the ν axis.  Done HERE so the reader's contract
    # is uniform — every array it returns is padded and sharded — rather
    # than leaving one straggler for the caller to remember.
    if G0_mu_nu is not None:
        if G0_mu_nu.ndim > 1:
            G0_mu_nu = G0_mu_nu[0]
        g0_pad = int(n_rmu_pad) - int(G0_mu_nu.shape[-1])
        if g0_pad > 0:
            G0_mu_nu = np.pad(G0_mu_nu, (0, g0_pad))
        # ``device_put_process_local``, NOT ``jax.device_put`` (AA.1):
        # G0 is host numpy read identically on every rank, and a plain
        # device_put onto a multi-process NamedSharding fires a hidden
        # assert_equal all-gather to prove exactly that.  The old reader
        # used ``with_sharding_constraint`` here, which was only ever
        # exercised at P=1 because the reader was refused above it -- so
        # there was no proven multi-process spelling to inherit.
        G0_mu_nu = device_put_process_local(
            np.ascontiguousarray(G0_mu_nu),
            NamedSharding(mesh_xy, P("y")))
    if enk_full is not None:
        enk_full = device_put_process_local(
            np.ascontiguousarray(enk_full),
            NamedSharding(mesh_xy, P(None, None)))

    del nspinor  # gated above; the extent itself rides on the arrays
    return (V_qmunu, S_qmunu, psi_full_y, enk_full, V0_noG0_munu, G0_mu_nu,
            psi_full_y_transverse, n_rmu_T_disk)


def load_restart_state_from_h5(filename, mesh_xy, band_slices=None,
                              n_rmu_logical=None):
    """Load canonical restart state and reshape wavefunctions into the
    two arrays expected by :func:`gw.wavefunction_bundle.build_wavefunctions`.

    Returns a ``SimpleNamespace`` with fields:

      V_qmunu, S_qmunu, V0_noG0_munu, G0_mu_nu, enk_full
      psi_rmu_Y   (nk, nb, ns, n_rmu)   P(None, None, None, 'y')
                  un-conjugated ψ.
      psi_rmuT_X  (nk, n_rmu, nb, ns)   P(None, 'x', None, None)
                  conjugated ψ* (matches the pair-density convention
                  ``load_centroids_band_chunked`` uses).

    The x-sharded psi copy is derived from the y-sharded one with a
    single y→x all-to-all on the μ axis; this is the only reshard on
    the restart path.
    """
    from types import SimpleNamespace
    # Loud-fail BEFORE any tensor is trusted (see the function's docstring).
    assert_restart_window_matches(filename, band_slices=band_slices,
                                  n_rmu_logical=n_rmu_logical)
    # Everything below arrives ALREADY sharded on mesh_xy and ALREADY at
    # the padded μ extent.  What used to be here — an 8-D/6-D collapse, a
    # ``jnp.pad`` on both μ axes of four tensors, and a
    # ``with_sharding_constraint`` on each — all operated on arrays that
    # were already resident whole on every rank, which is precisely why
    # the reader had to be guarded off above one process.  The pad is now
    # the read (SlabIO zero-fills past the dataset) and the sharding is
    # the read (SlabIO returns the tile), so none of it survives here.
    (V_qmunu, S_qmunu, psi_rmu_Y, enk_full, V0_noG0_munu, G0_mu_nu,
     psi_rmu_Y_T, n_rmu_T_disk) = read_restart_state_from_h5(
        filename, mesh_xy)

    x1_psi_X = NamedSharding(mesh_xy, P(None, "x", None, None))

    # psi_rmuT_X: conj + transpose(nb↔μ) then y→x reshard on μ.  This is
    # the ONLY reshard on the restart path, and it is deliberate: the two
    # ψ copies are what the pair-density contraction needs.
    psi_rmuT_X = jax.lax.with_sharding_constraint(
        jnp.conj(psi_rmu_Y).transpose(0, 3, 1, 2), x1_psi_X)

    # Bispinor transverse ψ: same two-copy derivation as the charge ψ, at
    # the TRANSVERSE μ extent (its own centroid count, its own pad, both
    # already applied by the reader).
    psi_rmuT_X_T = None
    if psi_rmu_Y_T is not None:
        psi_rmuT_X_T = jax.lax.with_sharding_constraint(
            jnp.conj(psi_rmu_Y_T).transpose(0, 3, 1, 2), x1_psi_X)

    return SimpleNamespace(
        V_qmunu=V_qmunu, S_qmunu=S_qmunu, V0_noG0_munu=V0_noG0_munu,
        G0_mu_nu=G0_mu_nu, enk_full=enk_full,
        psi_rmu_Y=psi_rmu_Y, psi_rmuT_X=psi_rmuT_X,
        psi_rmu_Y_transverse=psi_rmu_Y_T,
        psi_rmuT_X_transverse=psi_rmuT_X_T,
        n_rmu_transverse_disk=n_rmu_T_disk,
    )


