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
    """One ``[restart_write]`` line for a completed dataset write.

    Single implementation on purpose: the AF.4c line format is
    load-bearing for log diagnosis, and this module used to carry four
    hand-synced copies of it (audit 2026-07-28; QUALITY_PATTERNS #3).
    Gated by :func:`_restart_write_log_on`.
    """
    if not _restart_write_log_on():
        return
    nb = int(np.prod(shape)) * int(np.dtype(dtype).itemsize)
    print(f"  [restart_write] {name} {tuple(int(v) for v in shape)}"
          f" {nb / 1e9:.2f} GB in {dt:.1f} s"
          f" ({nb / 1e6 / max(dt, 1e-9):.0f} MB/s)", flush=True)


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
    backend=None,
    use_ffi_io: bool | None = None,
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
                backend=backend, use_ffi_io=use_ffi_io) as io:
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

        # PER-DATASET LIVENESS (scorecard AF.4c): every completed write
        # below emits one [restart_write] line — rationale and the
        # LORRAX_RESTART_WRITE_LOG gate live at
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
    backend=None, use_ffi_io: bool | None = None,
):
    """Overwrite or append the W0_qmunu dataset in an existing restart file.

    ``n_rmu_logical`` clips the trailing (μ, μ) axes to the logical
    on-disk extent — same contract as ``write_restart_state_to_h5``.
    """
    from .slab_io import SlabIO

    shape = _mu_logical_shape(W0_qmunu.shape, (-2, -1), n_rmu_logical)
    with SlabIO(filename, mode="a", mesh=mesh,
                backend=backend, use_ffi_io=use_ffi_io) as io:
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


def read_restart_state_from_h5(filename):
    """Read canonical restart state from HDF5 (restart format v2)."""
    with h5py.File(filename, "r") as f:
        if "psi_full_y" not in f:
            raise ValueError(
                f"Restart file {filename} is missing canonical psi_full_y dataset. "
                "Regenerate restart tensors with current gw_jax."
            )

        V_qmunu = jnp.asarray(f["V_qmunu"][:])
        S_qmunu = jnp.asarray(f["S_qmunu"][:]) if "S_qmunu" in f else None
        V0_noG0_munu = jnp.asarray(f["V0_noG0_munu"][:]) if "V0_noG0_munu" in f else None
        G0_mu_nu = jnp.asarray(f["G0_mu_nu"][:]) if "G0_mu_nu" in f else None
        psi_full_y = jnp.asarray(f["psi_full_y"][:])
        enk_full = jnp.asarray(f["enk_full"][:]) if "enk_full" in f else None
        # Bispinor per-channel ψ (transverse centroid set) — optional;
        # absent in scalar restarts and in bispinor restarts written
        # before 2026-07-27.
        psi_full_y_transverse = (
            jnp.asarray(f["psi_full_y_transverse"][:])
            if "psi_full_y_transverse" in f else None)
        # Integrity cross-check of the stamped transverse extent against
        # the dataset it describes (audit 2026-07-28: the stamp used to
        # be write-only shadow metadata, QUALITY_PATTERNS #3 — the
        # loader derives the extent from the dataset shape, so a
        # mismatch means a torn or hand-edited file and must refuse
        # loudly rather than feed downstream re-padding, #7).
        if (psi_full_y_transverse is not None
                and "n_rmu_transverse_logical" in f):
            stored_T = int(np.asarray(f["n_rmu_transverse_logical"])[()])
            disk_T = int(psi_full_y_transverse.shape[-1])
            if stored_T != disk_T:
                raise ValueError(
                    f"Restart file {filename}: stamped "
                    f"n_rmu_transverse_logical={stored_T} does not match "
                    f"the psi_full_y_transverse μ extent on disk "
                    f"({disk_T}).  The file is internally inconsistent "
                    f"(torn write or hand-edited) — regenerate the "
                    f"restart tensors (restart=false).")

    return (V_qmunu, S_qmunu, psi_full_y, enk_full, V0_noG0_munu, G0_mu_nu,
            psi_full_y_transverse)


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
    (V_qmunu, S_qmunu, psi_full_y_raw, enk_full, V0_noG0_munu, G0_mu_nu,
     psi_full_y_T_raw) = read_restart_state_from_h5(filename)

    # V_qmunu is now flat-q ``(nq, μ, μ)``.  Earlier formats had leading
    # ``(1, npol, npol)`` axes (and even earlier, the ``(nkx, nky, nkz)``
    # split); both are gone in the new gw/cohsex pipeline.  μ × μ are
    # still the trailing two axes that carry the (x, y) sharding.
    x1y2_3 = NamedSharding(mesh_xy, P(None, "x", "y"))
    x3y4_5 = NamedSharding(mesh_xy, P(None, None, None, "x", "y"))
    y3_psi_Y = NamedSharding(mesh_xy, P(None, None, None, "y"))
    x1_psi_X = NamedSharding(mesh_xy, P(None, "x", None, None))
    replicated_2 = NamedSharding(mesh_xy, P(None, None))

    # Back-compat: handle restarts written under the old 8-D layout by
    # collapsing leading axes; the new 3-D form passes through.
    if V_qmunu.ndim == 8:
        V_qmunu = jnp.asarray(V_qmunu)[0, 0, 0].reshape(
            -1, V_qmunu.shape[-2], V_qmunu.shape[-1])
    elif V_qmunu.ndim == 6:
        V_qmunu = jnp.asarray(V_qmunu)[0, 0, 0]

    # Disk stores the LOGICAL μ extent (SlabIO clips the writer's pad
    # rows against it); in-memory arrays carry the padded extent
    # ``padded_mu_extent(n_rmu, world_size)`` with exact-zero pad rows.
    # Re-apply the pad here so downstream shapes match ``Meta``.
    # Restart files predating the clip carry an already-padded extent
    # written at the same device count — ``padded_mu_extent`` is then a
    # fixed point and every pad below is a no-op.
    from runtime.padding import padded_mu_extent
    n_rmu_disk = int(V_qmunu.shape[-1])
    mu_pad = padded_mu_extent(n_rmu_disk, int(jax.device_count())) - n_rmu_disk
    if mu_pad > 0:
        V_qmunu = jnp.pad(V_qmunu, ((0, 0), (0, mu_pad), (0, mu_pad)))
        if S_qmunu is not None:
            S_qmunu = jnp.pad(
                S_qmunu,
                [(0, 0)] * (S_qmunu.ndim - 2) + [(0, mu_pad), (0, mu_pad)])
        if V0_noG0_munu is not None:
            V0_noG0_munu = jnp.pad(V0_noG0_munu, ((0, mu_pad), (0, mu_pad)))
        if G0_mu_nu is not None:
            G0_mu_nu = jnp.pad(
                G0_mu_nu, [(0, 0)] * (G0_mu_nu.ndim - 1) + [(0, mu_pad)])
        psi_full_y_raw = jnp.pad(
            psi_full_y_raw,
            [(0, 0)] * (psi_full_y_raw.ndim - 1) + [(0, mu_pad)])

    V_qmunu = jax.lax.with_sharding_constraint(V_qmunu, x1y2_3)
    if S_qmunu is not None:
        S_qmunu = jax.lax.with_sharding_constraint(S_qmunu, x3y4_5)
    if V0_noG0_munu is not None:
        V0_noG0_munu = jax.lax.with_sharding_constraint(V0_noG0_munu, NamedSharding(mesh_xy, P("x", "y")))
    if G0_mu_nu is not None:
        # G0 should be (n_rmu,) for head corrections. If stored as 2D
        # (e.g. (nqz, n_rmu) from an old code version), extract q=0 row.
        if G0_mu_nu.ndim > 1:
            G0_mu_nu = G0_mu_nu[0]
        G0_mu_nu = jax.lax.with_sharding_constraint(G0_mu_nu, NamedSharding(mesh_xy, P("y")))

    # psi_rmu_Y: stored layout (un-conjugated ψ), just pin to Y-sharding.
    psi_rmu_Y = jax.lax.with_sharding_constraint(psi_full_y_raw, y3_psi_Y)
    # psi_rmuT_X: conj + transpose(nb↔μ) then y→x reshard on μ.
    psi_rmuT_X = jax.lax.with_sharding_constraint(
        jnp.conj(psi_rmu_Y).transpose(0, 3, 1, 2),
        x1_psi_X,
    )
    if enk_full is not None:
        enk_full = jax.lax.with_sharding_constraint(enk_full, replicated_2)

    # Bispinor transverse ψ (optional): same re-pad + two-copy derivation
    # as the charge ψ, at the TRANSVERSE μ extent (its own centroid
    # count, its own pad).
    psi_rmu_Y_T = psi_rmuT_X_T = None
    n_rmu_T_disk = None
    if psi_full_y_T_raw is not None:
        n_rmu_T_disk = int(psi_full_y_T_raw.shape[-1])
        mu_pad_T = (padded_mu_extent(n_rmu_T_disk, int(jax.device_count()))
                    - n_rmu_T_disk)
        if mu_pad_T > 0:
            psi_full_y_T_raw = jnp.pad(
                psi_full_y_T_raw,
                [(0, 0)] * (psi_full_y_T_raw.ndim - 1) + [(0, mu_pad_T)])
        psi_rmu_Y_T = jax.lax.with_sharding_constraint(
            psi_full_y_T_raw, y3_psi_Y)
        psi_rmuT_X_T = jax.lax.with_sharding_constraint(
            jnp.conj(psi_rmu_Y_T).transpose(0, 3, 1, 2),
            x1_psi_X,
        )

    return SimpleNamespace(
        V_qmunu=V_qmunu, S_qmunu=S_qmunu, V0_noG0_munu=V0_noG0_munu,
        G0_mu_nu=G0_mu_nu, enk_full=enk_full,
        psi_rmu_Y=psi_rmu_Y, psi_rmuT_X=psi_rmuT_X,
        psi_rmu_Y_transverse=psi_rmu_Y_T,
        psi_rmuT_X_transverse=psi_rmuT_X_T,
        n_rmu_transverse_disk=n_rmu_T_disk,
    )


