"""The post-891047f4 GW bundle reader.

Canonical files carry raw parent faces and logical centroid order. This module
owns format admission, per-rank SlabIO reads, family selection and the single
symmetry-service unfold. Consumers receive arrays, never HDF5 handles.
"""
from __future__ import annotations

import glob
import json
import os
from types import SimpleNamespace
from pathlib import Path
from typing import Optional

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV
import common.timing as timing
from common.collectives import device_put_process_local
from runtime.padding import (authenticate_axis, authenticate_padded_axis,
    mesh_divisor, pad_to_axis, padded_axis, padded_mu_axis, padded_mu_extent,
    pad_axis)
from .tagged_arrays import (BAND_WINDOW_SCHEMA_DATASET, BAND_WINDOW_SCHEMA_VERSION,
    BAND_WINDOW_CARRIER_DATASET, CHARGE_ZETA_IDENTITY_DATASET,
    COULOMB_POLICY_DATASET, DOWNFOLD_PROVENANCE_GROUP,
    _decode_charge_zeta_identity, _loaded_band_axis, _validate_shape_receipt,
    coulomb_policy_from_config, compare_coulomb_policy, parse_coulomb_policy,
    format_coulomb_policy)

_REGENERATE = "regenerate with gwjax at main ≥ 891047f4"


def _require_current(f):
    """Refuse retired full-k bundles once, before any payload read."""
    required = ("psi_parent_y", "psi_parent_y_mun", "psi_parent_k_rows",
                "band_window", "band_window_schema", "enk_full", "kgrid")
    if (any(name not in f for name in required)
            or any(name.startswith("psi_full_") for name in f)
            or int(f["band_window_schema"][()]) != BAND_WINDOW_SCHEMA_VERSION):
        raise ValueError(_REGENERATE)


def _munu_slab_request(ds_shape, n_rmu_pad):
    """Read canonical trailing μ/ν axes with their XY sharding."""
    if len(ds_shape) not in (2, 3, 5):
        raise ValueError(_REGENERATE)
    shape = tuple(ds_shape[:-2]) + (int(n_rmu_pad),) * 2
    return (0,) * len(shape), shape, P(*([None] * (len(shape)-2)), "x", "y")

def read_coulomb_policy_from_h5(filename) -> dict | None:
    """Read the stamp off a restart file with serial h5py; ``None`` if absent.

    Scalar-class metadata, read the same way ``assert_restart_window_matches``
    reads the band window — no SlabIO handle, no collective, safe to call on
    any rank before the tensors move.
    """
    try:
        with h5py.File(filename, "r") as f:
            if COULOMB_POLICY_DATASET not in f:
                return None
            return parse_coulomb_policy(f[COULOMB_POLICY_DATASET][()])
    except (OSError, KeyError):
        return None


def read_downfold_provenance(filename) -> dict | None:
    """The ``downfold_provenance`` group of a restart bundle; ``None`` if absent.

    ``None`` means "natively fitted, as far as this file says" — a bundle
    written by ``gw.gw_jax`` carries no such group, and so does one written
    by a downfold predating the stamp.  Both are read as not-downfolded,
    which is the safe direction: every consumer's existing behaviour is what
    it gets.

    WHY A READER LIVES HERE AT ALL.  A downfolded bundle is deliberately
    indistinguishable from a natively fitted one BY SHAPE — that is what
    makes it a drop-in for ``bse.bse_jax``.  But two facts about it are not
    derivable from shape and are load-bearing for any consumer that has to
    build something NEW in the same ISDF basis rather than only read the
    stored tensors: which centroid table the parent basis came from, and
    which of the parent's centroid rows survived.  ``bse.exciton_bands``
    needs both (its htransform leg fits ψ in the PARENT basis and slices the
    result to the kept rows), and this is the one place either is recorded.

    Serial h5py, no SlabIO handle, no collective — the same contract as
    :func:`read_coulomb_policy_from_h5`, so it is safe to call on any rank
    before the tensors move.

    Returns the group's attributes as a plain dict (bytes decoded to str),
    plus ``keep_idx`` / ``retained_rank_per_q`` as numpy arrays when present.
    """
    try:
        with h5py.File(filename, "r") as f:
            if DOWNFOLD_PROVENANCE_GROUP not in f:
                return None
            g = f[DOWNFOLD_PROVENANCE_GROUP]
            out = {}
            for k, v in g.attrs.items():
                out[k] = v.decode("utf-8") if isinstance(v, bytes) else v
            for name in ("keep_idx", "retained_rank_per_q"):
                if name in g:
                    out[name] = np.asarray(g[name][:])
            return out
    except (OSError, KeyError):
        return None


def describe_coulomb_policy_stamp(filename) -> str:
    """One line naming the Coulomb policy a restart file's tensors carry.

    For readers that CONSUME W rather than rebuild V — the BSE, which by
    its own note "does NOT compute W, it READS it off the GW restart".
    They have no Coulomb config of their own to compare against, so the
    honest disclosure is the stored policy itself, not a match verdict.
    Without this line a BSE run's log has no record of which averaging
    convention its screening was built under, which is exactly the gap
    that made the cross-code residual arguable in the first place.
    """
    stamped = read_coulomb_policy_from_h5(filename)
    if stamped is None:
        return ("  [restart stamp] Coulomb-kernel policy: NOT STAMPED "
                "(GW restart predates the stamp) - the screening in this "
                "file was built under an unrecorded averaging convention.")
    return ("  [restart stamp] screening built under Coulomb policy: "
            + ";".join(f"{k}={v}" for k, v in stamped.items()))


def describe_coulomb_policy_match(filename, cfg, meta=None) -> str:
    """One line for a restart log: matched, mismatched, or legacy-unstamped.

    Returns the text; the caller prints it, so this stays importable from
    the BSE side (which reads the same file and owes the same disclosure)
    without either side owning the other's print function.
    """
    running = coulomb_policy_from_config(cfg, meta)
    stamped = read_coulomb_policy_from_h5(filename)
    if stamped is None:
        return ("  [restart stamp] Coulomb-kernel policy: NOT STAMPED "
                "(file predates the stamp). Read as legacy — the stored V/W "
                "were built under whatever averaging policy that run used, "
                "and this run cannot tell which. Running policy is "
                f"{format_coulomb_policy(running)}")
    diffs = compare_coulomb_policy(stamped, running)
    if not diffs:
        return (f"  [restart stamp] Coulomb-kernel policy matches: "
                f"{format_coulomb_policy(stamped)}")
    detail = "; ".join(f"{k}: file={a!r} run={b!r}" for k, a, b in diffs)
    return (
        "  [restart stamp] WARNING - Coulomb-kernel policy MISMATCH between "
        "this restart file and the running config. The restart reuses "
        "V_qmunu verbatim and never re-runs compute_V_q, so the stored "
        "tensors carry the FILE's policy and every other guard will pass. "
        f"Differences -> {detail}. Rerun with restart = false if the "
        "running policy is the one you meant.")


def assert_restart_window_matches(filename, band_slices=None,
                                 n_rmu_logical=None) -> None:
    """Authenticate a current bundle and its logical band and centroid receipts."""
    with h5py.File(filename, "r") as f:
        _require_current(f)
        from .commit_state import assert_committed
        assert_committed(f, path=filename)
        stored_w = np.asarray(f["band_window"]).tolist() if "band_window" in f else None
        stored_split = (np.asarray(f["band_window_split"]).tolist()
                        if "band_window_split" in f else None)
        stored_schema = (
            int(np.asarray(f[BAND_WINDOW_SCHEMA_DATASET])[()])
            if BAND_WINDOW_SCHEMA_DATASET in f else None)
        stored_carrier = (
            np.asarray(f[BAND_WINDOW_CARRIER_DATASET]).tolist()
            if BAND_WINDOW_CARRIER_DATASET in f else None)
        stored_mu = (int(np.asarray(f["n_rmu_logical"])[()])
                     if "n_rmu_logical" in f else None)

    if stored_schema is not None:
        if stored_schema != BAND_WINDOW_SCHEMA_VERSION:
            raise ValueError(
                f"Restart file {filename} has unsupported band-window schema "
                f"{stored_schema}; this reader supports "
                f"{BAND_WINDOW_SCHEMA_VERSION}.")
        if stored_w is None or stored_carrier is None or stored_split is None:
            raise ValueError(
                f"Restart file {filename} declares band-window schema "
                f"{stored_schema} but is missing band_window, "
                f"band_window_split, or {BAND_WINDOW_CARRIER_DATASET}. The "
                "geometry receipt is torn.")
        if len(stored_w) != 5 or len(stored_carrier) != 5:
            raise ValueError(
                f"Restart file {filename} has malformed band-window receipts: "
                f"logical={stored_w}, carrier={stored_carrier}; expected two "
                "five-edge windows.")
        logical = tuple(int(v) for v in stored_w)
        carrier = tuple(int(v) for v in stored_carrier)
        if logical[:4] != carrier[:4] or carrier[4] < logical[4]:
            raise ValueError(
                f"Restart file {filename} has inconsistent logical/carrier "
                f"band receipts: logical={logical}, carrier={carrier}. The "
                "carrier may add only a zero-pad tail above the logical b4.")

    # THE χ COUNT MUST MATCH; THE Σ COUNT NEED NOT, and the asymmetry is the
    # point.  Every tensor in this file is a function of the SCREENING side or
    # of the loaded extent: ``V_qmunu`` / ``W0_qmunu`` are built from the χ0
    # band sum, ``psi_parent_y`` / ``enk_full`` / ζ span [b0, b4) =
    # max(chi, sigma) — which ``band_window``'s b4 already pins.  NOTHING on
    # disk is a function of ``number_bands_sigma``: Σ slices [0, b4_sigma) out
    # of tensors that already exist.
    #
    # So a Σ-count sweep at fixed χ reuses this file legitimately — which is
    # the case the split exists to make cheap (χ at full bands, Σ short and
    # extrapolated), and it is why this is a targeted check rather than a
    # blanket "no restart under a split".  Changing χ is refused, for exactly
    # the reason the 5-tuple check above exists.
    if band_slices is not None:
        want_b4_logical = int(
            getattr(band_slices, "b4_logical", 0) or band_slices.b4)
        want_split = (
            min(int(band_slices.b4_chi), want_b4_logical),
            min(int(band_slices.b4_sigma), want_b4_logical),
        )
        if stored_split is not None and int(stored_split[0]) != want_split[0]:
            raise ValueError(
                f"Restart file {filename} was written with a chi0/W band sum "
                f"topping out at band {int(stored_split[0])}, but this run "
                f"has number_bands_chi -> band {want_split[0]}.  V_qmunu and "
                f"W0_qmunu ARE the screening, so reusing them would run this "
                f"deck's Sigma against the OTHER deck's W and report rc=0 "
                f"(the same silent-misindex class as the band-window check "
                f"below; see job 7874375).  Either restore the original "
                f"number_bands_chi, or set restart=false.  Note that "
                f"number_bands_SIGMA may be changed freely on a restart: no "
                f"tensor in this file depends on it.")

    if stored_w is not None and band_slices is not None:
        want_b4 = int(band_slices.b4)
        if stored_schema == BAND_WINDOW_SCHEMA_VERSION:
            want_b4 = int(
                getattr(band_slices, "b4_logical", 0) or band_slices.b4)
        want = [int(band_slices.b0), int(band_slices.b1), int(band_slices.b2),
                int(band_slices.b3), want_b4]
        stored_stable = [int(stored_w[index]) for index in (0, 1, 2, 4)]
        want_stable = [want[index] for index in (0, 1, 2, 4)]
        if stored_stable != want_stable:
            raise ValueError(
                f"Restart file {filename} was written under stable restart "
                f"window (b0,b1,b2,b4)={tuple(stored_stable)} but this run "
                f"has {tuple(want_stable)}. V_qmunu / psi_parent_y / enk_full "
                f"are indexed by that window, so reusing them would MISINDEX "
                f"Sigma silently (no crash, wrong QP energies -- see job "
                f"7874375). Either restore the original nval and loaded/chi "
                f"band extent, or set restart=false to rebuild the tensors. "
                f"The Sigma-only b3 edge may change because no restart "
                f"tensor depends on number_bands_sigma."
            )
    if stored_mu is not None and n_rmu_logical is not None:
        if int(stored_mu) != int(n_rmu_logical):
            raise ValueError(
                f"Restart file {filename} was written with n_rmu={stored_mu} "
                f"but this run has n_rmu={int(n_rmu_logical)}.  The ISDF basis "
                f"differs, so V_qmunu / psi_parent_y are not reusable.  Set "
                f"restart=false (or point at the matching centroid file)."
            )


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


def _qirr_wedge_tables(f):
    """Read the symmetry-service tables attached to q-IBZ tensors."""
    from symmetry_maps import read_tables, dataset_q_storage
    return {name: read_tables(f, name)
            for name in ("V_qmunu", "S_qmunu", "V0_noG0_munu", "W0_qmunu")
            if name in f and dataset_q_storage(f[name]) == "ibz"}


def _unfold_wedge(A, tables, n_rmu_pad, mesh_xy):
    """Restore q-IBZ tiles through the producer symmetry action; current full-q tiles pass through."""
    if tables is None:
        return A
    from symmetry_maps import unfold_isdf_operator
    t = tables.padded(int(n_rmu_pad))
    return unfold_isdf_operator(
        A, irr_idx=t.irr_idx_q, sym_idx=t.sym_idx_q, sym_perm=t.sym_perm,
        L_table=t.L_table, q_irr_frac=t.q_irr_frac, mesh_xy=mesh_xy,
        n_sym_spatial=int(t.n_sym_spatial))


def read_restart_state_from_h5(filename, mesh_xy, *, low_mem_bands=False,
                               band_receipt=None, n_band_carrier=None):
    """Read raw parent faces and canonical tensors through SlabIO. Spin is a shape; low_mem_bands selects only face partition specs."""
    from .slab_io import SlabIO
    from common.collectives import device_put_process_local

    # ---- pass 1: geometry + the small replicated arrays, serial h5py ----
    with h5py.File(filename, "r") as f:
        _require_current(f)
        parent_T = "psi_parent_y_transverse" in f
        if parent_T != ("psi_parent_y_transverse_mun" in f):
            raise ValueError("Restart has torn transverse parent faces")
        transverse_name = "psi_parent_y_transverse"
        parent_k_rows = np.asarray(f["psi_parent_k_rows"][()], dtype=np.int64)
        # THE UNFOLD TABLES, while this handle is open and before any tensor
        # bytes move.  Empty on every full-BZ and legacy file, which is what
        # keeps those reads on the byte path they have always had.
        wedge_tables = _qirr_wedge_tables(f)
        shapes = {k: tuple(int(s) for s in f[k].shape)
                  for k in ("V_qmunu", "S_qmunu", "V0_noG0_munu",
                            "psi_parent_y", "psi_parent_y_mun",
                            "psi_parent_y_transverse", "psi_parent_y_transverse_mun")
                  if k in f}
        if parent_T and shapes[transverse_name][0] != shapes["psi_parent_y"][0]:
            raise ValueError("Restart parent-row count differs between charge and current families.")
        dtypes = {k: f[k].dtype for k in shapes}
        for name in shapes:
            _validate_shape_receipt(name, f[name])
        enk_full = (np.asarray(f["enk_full"][:]) if "enk_full" in f else None)
        G0_mu_nu = (np.asarray(f["G0_mu_nu"][:]) if "G0_mu_nu" in f else None)
        stored_T = (int(np.asarray(f["n_rmu_transverse_logical"])[()])
                    if "n_rmu_transverse_logical" in f else None)
        charge_zeta_identity = (
            _decode_charge_zeta_identity(
                f[CHARGE_ZETA_IDENTITY_DATASET][()],
                where=f"Restart file {filename}")
            if CHARGE_ZETA_IDENTITY_DATASET in f else None)

    divisor = mesh_divisor(mesh_xy)
    n_rmu_disk = int(shapes["V_qmunu"][-1])
    mu_axis = padded_mu_axis(n_rmu_disk, divisor)
    n_rmu_pad = mu_axis.carrier
    nspinor = _check_nspinor(
        shapes["psi_parent_y"][2],
        f"read_restart_state_from_h5({filename})")

    # Integrity cross-check of the stamped transverse extent against the
    # dataset it describes (audit 2026-07-28: the stamp used to be
    # write-only shadow metadata, QUALITY_PATTERNS #3 — a mismatch means a
    # torn or hand-edited file and must refuse loudly rather than feed
    # downstream re-padding, #7).  Checked on the SHAPE, before any bytes
    # move, so a bad file costs nothing.
    if transverse_name in shapes and stored_T is not None:
        disk_T = int(shapes[transverse_name][-1])
        if stored_T != disk_T:
            raise ValueError(
                f"Restart file {filename}: stamped "
                f"n_rmu_transverse_logical={stored_T} does not match the "
                f"psi_parent_y_transverse μ extent on disk ({disk_T}).  The "
                f"file is internally inconsistent (torn write or "
                f"hand-edited) — regenerate the restart tensors "
                f"(restart=false).")

    # ---- pass 2: the N_mu²-class and ψ tensors, one tile per rank -------
    from common.wfn_layout import psi_specs
    psi_nmu_spec, psi_mun_spec = psi_specs("face" if low_mem_bands else "axis")

    def _read_munu(io, name):
        if name not in shapes:
            return None
        off, shape, spec = _munu_slab_request(shapes[name], n_rmu_pad)
        with timing.section(
                f"gw_jax.restart.read.{name}", announce=True,
                label=f"restart SlabIO read {name}"):
            arr = io.read_slab(name, shape=shape, dtype=dtypes[name],
                               offset=off, mesh=mesh_xy,
                               partition_spec=spec)
            jax.block_until_ready(arr)
        with timing.section(
                f"gw_jax.restart.wedge_transform.{name}", announce=True,
                label=f"restart wedge transform {name}"):

            authenticate_axis(
                arr, mu_axis, axis=-2,
                where=f"read_restart_state_from_h5 dataset {name!r}")
            authenticate_axis(
                arr, mu_axis, axis=-1,
                where=f"read_restart_state_from_h5 dataset {name!r}")
        # THE UNFOLD, AFTER THE READ AND BEFORE THE CALLER SEES IT.  The
        # request above derives its q extent from the dataset shape, so a
        # wedge simply arrives as (n_q_ibz, mu_pad, nu_pad) on the same spec
        # the unfold takes and returns.  A no-op on every non-wedge file.
            arr = _unfold_wedge(
                arr, wedge_tables.get(name), n_rmu_pad, mesh_xy)
            jax.block_until_ready(arr)
        return arr

    def _read_psi(io, name, n_mu_logical, *, spec, mu_axis=-1,
                  spinor_axis=2, band_axis=1):
        """One direct hyperslab of a ψ dataset, μ padded, at ``spec``.

        ``mu_axis``/``spinor_axis`` default to the legacy/nmu axis order
        (nk, n, s, μ); the mun face (nk, s, μ, n) passes both explicitly
        — its μ is axis -2 and its spinor is axis 1, not axis -1/2.  NO
        RESHARD happens here regardless of ``spec``: this is a straight
        SlabIO hyperslab read, so a face spec costs exactly what the
        legacy spec costs (one direct read), never a transpose collective.
        """
        if name not in shapes:
            return None
        ds = shapes[name]
        _check_nspinor(ds[spinor_axis], f"{name} in {filename}")
        mu_tag = padded_mu_axis(int(n_mu_logical), divisor)
        shape = list(int(s) for s in ds)
        mu_index = int(mu_axis) % len(shape)
        shape[mu_index] = mu_tag.carrier
        b_axis = int(band_axis) % len(shape)
        band_tag = (band_receipt if band_receipt is not None else
                    _loaded_band_axis(int(ds[b_axis]), mesh_xy))
        if n_band_carrier is not None:
            band_tag = authenticate_padded_axis(
                band_tag.logical, int(n_band_carrier), band_tag.divisor,
                name=band_tag.name)
        shape[b_axis] = band_tag.carrier
        with timing.section(
                f"gw_jax.restart.read.{name}", announce=True,
                label=f"restart SlabIO read {name}"):
            arr = io.read_slab(
                name, shape=tuple(shape), dtype=dtypes[name],
                mesh=mesh_xy, partition_spec=spec)
            jax.block_until_ready(arr)
        authenticate_axis(
            arr, mu_tag, axis=mu_index,
            where=f"read_restart_state_from_h5 dataset {name!r}")
        authenticate_axis(
            arr, band_tag, axis=b_axis,
            where=f"read_restart_state_from_h5 dataset {name!r}")
        return arr

    n_rmu_T_disk = (int(shapes[transverse_name][-1])
                    if transverse_name in shapes else None)

    with SlabIO(filename, mode="r", mesh=mesh_xy) as io:
        V_qmunu = _read_munu(io, "V_qmunu")
        S_qmunu = _read_munu(io, "S_qmunu")
        V0_noG0_munu = _read_munu(io, "V0_noG0_munu")
        psi_nmu_parent = None
        psi_mun_parent = None
        psi_nmu_parent_T = psi_mun_parent_T = None
        psi_nmu_parent = _read_psi(io, "psi_parent_y", n_rmu_disk,
                                    spec=psi_nmu_spec)
        psi_mun_parent = _read_psi(io, "psi_parent_y_mun", n_rmu_disk,
                                    spec=psi_mun_spec, mu_axis=-2,
                                    spinor_axis=1, band_axis=-1)
        if n_rmu_T_disk is not None:
            psi_nmu_parent_T = _read_psi(io, transverse_name, n_rmu_T_disk,
                                        spec=psi_nmu_spec)
            psi_mun_parent_T = _read_psi(io, transverse_name + "_mun", n_rmu_T_disk,
                                        spec=psi_mun_spec, mu_axis=-2,
                                        spinor_axis=1, band_axis=-1)

    # G0: μ-class, read whole above.  Collapse a legacy 2-D (nqz, μ) store
    # to its q=0 row, pad to the same in-memory μ extent as everything
    # else, and pin it to the ν axis.  Done HERE so the reader's contract
    # is uniform — every array it returns is padded and sharded — rather
    # than leaving one straggler for the caller to remember.
    if G0_mu_nu is not None:
        if G0_mu_nu.ndim != 1:
            raise ValueError(_REGENERATE)
        G0_mu_nu = np.asarray(pad_to_axis(
            G0_mu_nu, mu_axis, axis=-1))
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
        n_disk_band = int(enk_full.shape[-1])
        band_tag = (band_receipt if band_receipt is not None else
                    _loaded_band_axis(n_disk_band, mesh_xy))
        if n_band_carrier is not None:
            band_tag = authenticate_padded_axis(
                band_tag.logical, int(n_band_carrier), band_tag.divisor,
                name=band_tag.name)
        if band_tag.pad:
            if enk_full.size == 0:
                raise ValueError(
                    "Restart enk_full is empty and cannot define the finite "
                    "energy sentinel needed for band-carrier padding.")
            sentinel = float(np.max(enk_full)) + 1.0
            enk_full = np.asarray(pad_to_axis(
                enk_full, band_tag, axis=-1, fill=sentinel))
        enk_full = device_put_process_local(
            np.ascontiguousarray(enk_full),
            NamedSharding(mesh_xy, P(None, None)))

    del nspinor  # gated above; the extent itself rides on the arrays
    return SimpleNamespace(
        V_qmunu=V_qmunu, S_qmunu=S_qmunu, enk_full=enk_full,
        V0_noG0_munu=V0_noG0_munu, G0_mu_nu=G0_mu_nu,
        n_rmu_transverse_disk=n_rmu_T_disk,
        charge_zeta_identity=charge_zeta_identity,
        psi_nmu_parent=psi_nmu_parent, psi_mun_parent=psi_mun_parent,
        parent_k_rows=parent_k_rows,
        psi_nmu_parent_transverse=psi_nmu_parent_T,
        psi_mun_parent_transverse=psi_mun_parent_T,
        layout="face" if low_mem_bands else "axis",
    )


def read_munu_tensor_from_h5(filename, name, mesh_xy, *, n_rmu_logical=None):
    """Read ONE ``(…, μ, ν)`` restart tensor, sharded, wedge unfolded.

    ``read_restart_state_from_h5`` reads the fixed set of tensors ``gw_init``
    writes on its ``mode="w"`` pass and deliberately does not know about
    ``W0_qmunu`` — W is written later, by a different function, once the
    Dyson solve has produced it.  Every consumer that wants W back has
    therefore had to re-derive the slab request, the legacy-layout collapse
    and the wedge unfold for itself (``bse_io`` does, at its own scale).
    This is that read, named once, on the same three private helpers the
    canonical reader uses, so a fourth consumer does not spell it a fourth
    way.

    Returns ``None`` when the dataset is absent — which is the normal case
    for ``V_qmunu_nohead`` / ``W0_qmunu_nohead``, an opt-in pair nothing
    in-tree writes.  Callers that REQUIRE the tensor say so themselves; a
    reader that raised here could not serve the optional ones.

    Padding, sharding and the wedge follow the canonical reader exactly:
    disk holds the LOGICAL μ extent, memory holds
    ``padded_mu_extent(μ, device_count)`` with zero pad rows, output is
    ``P(None,'x','y')``, and an IBZ-wedge dataset comes back on the FULL BZ
    with the caller none the wiser.

    ``n_rmu_logical`` overrides the μ extent read off the dataset — pass it
    only when the dataset itself is the thing under suspicion.
    """
    from .slab_io import SlabIO

    with h5py.File(filename, "r") as f:
        if name not in f:
            return None
        ds_shape = tuple(int(s) for s in f[name].shape)
        ds_dtype = f[name].dtype
        from symmetry_maps import dataset_q_storage, read_tables
        tables = (read_tables(f, name)
                  if dataset_q_storage(f[name]) == "ibz" else None)

    n_rmu_disk = int(ds_shape[-1] if n_rmu_logical is None else n_rmu_logical)
    mu_tag = padded_mu_axis(n_rmu_disk, mesh_xy)
    n_rmu_pad = mu_tag.carrier
    off, shape, spec = _munu_slab_request(ds_shape, n_rmu_pad)
    with SlabIO(filename, mode="r", mesh=mesh_xy) as io:
        arr = io.read_slab(name, shape=shape, dtype=ds_dtype, offset=off,
                           mesh=mesh_xy, partition_spec=spec)
    authenticate_axis(
        arr, mu_tag, axis=-2,
        where=f"read_munu_tensor_from_h5 dataset {name!r}")
    authenticate_axis(
        arr, mu_tag, axis=-1,
        where=f"read_munu_tensor_from_h5 dataset {name!r}")
    return _unfold_wedge(arr, tables, n_rmu_pad, mesh_xy)


def load_restart_state_from_h5(filename, mesh_xy, band_slices=None,
                              n_rmu_logical=None, low_mem_bands=False):
    """Return the canonical parent state namespace at the requested band layout. All μ axes are canonical and padded; GW packs them in its authenticated basis."""
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
    band_axis = None
    if band_slices is not None:
        n_band_logical = (
            int(getattr(band_slices, "b4_logical", 0) or band_slices.b4)
            - int(band_slices.b0))
        band_axis = _loaded_band_axis(n_band_logical, mesh_xy)
        authenticate_padded_axis(
            band_axis.logical,
            int(band_slices.b4) - int(band_slices.b0),
            band_axis.divisor, name=band_axis.name)

    return read_restart_state_from_h5(
        filename, mesh_xy, low_mem_bands=bool(low_mem_bands),
        band_receipt=band_axis)


def unfold_parent_faces(faces, restart_file, input_file, mesh_xy, *, family="charge"):
    """Authenticate canonical parent faces and unfold only the selected BSE bands."""
    from common.centroid_basis import PackedCentroidBasis
    from common.shard_map import shard_map
    from file_io.centroids import load_centroid_basis
    from file_io.qp_wfn import authenticate_restart_qp_state_source_for_wfn
    from file_io.wfn_basis import centroid_table_md5
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from wfn_loader import WfnLoader

    with h5py.File(restart_file, "r") as f:
        _require_current(f)
        rows = np.asarray(f["psi_parent_k_rows"])
        digest = f.attrs.get({"charge": "centroids_charge_md5",
                              "current": "centroids_transverse_md5"}[family])
    if any(face.shape[0] != len(rows) for face in faces):
        raise ValueError("BSE parent face extent does not match its saved parent rows")
    if input_file is None:
        raise ValueError("BSE parent restart requires input_file for its typed symmetry action")
    from gw.gw_config import LorraxConfig
    cfg = LorraxConfig.from_input_file(input_file)
    wfn = WfnLoader(cfg.paths.wfn_file)
    authenticate_restart_qp_state_source_for_wfn(
        wfn=wfn, state_artifact_path=restart_file, where="BSE parent reader")
    sym = wfn.symmetry()
    path = {"charge": cfg.paths.centroids_file,
            "current": cfg.paths.centroids_file_current}[family]
    centroids = load_centroid_basis(path, wfn.fft_grid, sym=sym)
    idx = centroids.centroid_indices
    if digest != centroid_table_md5(idx):
        raise ValueError("BSE parent restart centroid content does not match its deck")
    if not centroids.orbit_closed or (
            np.array_equal(rows, np.arange(sym.nk_tot)) and sym.nk_red != sym.nk_tot):
        sym = sym.trivial_view()
    basis = PackedCentroidBasis.build(idx, sym, wfn.fft_grid, mesh_xy)
    plan = build_centroid_k_unfold_plan(
        sym, idx, wfn.fft_grid, mesh_xy, nspinor=faces[0].shape[2],
        parent_k_frac=wfn.kvecs(k=sym.parent_k_domain), layout=basis.layout)
    if not np.array_equal(rows, plan.parent_full_rows):
        raise ValueError("BSE parent restart rows do not match the authenticated file wedge")
    spec = P(None, None, None, "x")
    unfold = jax.jit(shard_map(
        lambda a: plan.unfold_face(a, spin_axis=2, mu_axis=3, mesh_axis="x"),
        mesh=mesh_xy, in_specs=spec, out_specs=spec, check_vma=False))
    return tuple(basis.unpack_axis(unfold(basis.pack_axis(a, 3, spec=spec)),
                                   3, spec=spec) for a in faces)




def read_metadata(filename):
    """Return small, replicated bundle facts in canonical file order.

    Energies are (full_k, band) in Ry; the head vector is (centroid,).
    Family shapes describe raw (parent_k, band, spin, centroid) faces.
    """
    with h5py.File(filename, "r") as f:
        _require_current(f)
        def value(name):
            return np.asarray(f[name][()]) if name in f else None
        return dict(
            energies=value("enk_full"), grid=value("kgrid"),
            band_window=value("band_window"), band_split=value("band_window_split"),
            centroid_count=int(f["n_rmu_logical"][()]),
            head_vector=value("G0_mu_nu"), bare_head=value("vhead"),
            screened_head=value("whead"), head_cartesian=value("S_cart_head"),
            screened_ready=("W0_qmunu" in f and bool(f["W0_qmunu"].attrs.get("W0_ready", False))),
            family_shapes={family: tuple(f[name].shape) for family, name in
                (("charge", "psi_parent_y"), ("current", "psi_parent_y_transverse")) if name in f},
            centroid_hashes={family: f.attrs.get(name) for family, name in
                (("charge", "centroids_charge_md5"), ("current", "centroids_transverse_md5"))},
        )


def read_interaction(filename, kind, mesh_xy, *, nohead=False):
    """Return a padded, full-q interaction with P(None, x, y) sharding."""
    names = {"bare": "V_qmunu", "screened": "W0_qmunu"}
    name = names[kind]
    with h5py.File(filename, "r") as f:
        _require_current(f)
        if name not in f or not bool(f[name].attrs.get(
                "W0_ready" if kind == "screened" else "V_ready", kind == "bare")):
            raise ValueError(f"{filename}: {kind} interaction was not persisted")
        if nohead and name + "_nohead" in f:
            name += "_nohead"
    return read_munu_tensor_from_h5(filename, name, mesh_xy)


def read_wavefunctions(filename, input_file, mesh_xy, *, bands=None, family="charge"):
    """Return full-k ψ(k, band, spin, μ_X) in canonical centroid order.

    ``bands`` is a contiguous sequence of indices in the stored band window.
    Spin is the selected family's true extent, without padding or projection.
    """
    from .slab_io import SlabIO
    names = {"charge": "psi_parent_y", "current": "psi_parent_y_transverse"}
    name = names[family]
    with h5py.File(filename, "r") as f:
        _require_current(f)
        shape = tuple(f[name].shape)
        dtype = f[name].dtype
    if bands is None:
        bands = np.arange(shape[1])
    bands = np.asarray(bands, dtype=np.int64)
    if (bands.size == 0 or bands[0] < 0 or bands[-1] >= shape[1]
            or not np.array_equal(bands, np.arange(bands[0], bands[0]+bands.size))):
        raise ValueError("Bundle wavefunction bands must be a nonempty contiguous stored window")
    width = padded_mu_extent(shape[-1], mesh_xy)
    spec = P(None, None, None, "x")
    with SlabIO(filename, mode="r", mesh=mesh_xy) as io:
        face = io.read_slab(name, shape=(shape[0], bands.size, shape[2], width),
            dtype=dtype, offset=(0, int(bands[0]), 0, 0), mesh=mesh_xy,
            partition_spec=spec)
    return unfold_parent_faces((face,), filename, input_file, mesh_xy, family=family)[0]


def read_bse_payload(filename, input_file, mesh_xy, val_indices, cond_indices,
                     *, nohead=False, full_exchange=False):
    """Return canonical BSE ψ faces, Γ exchange and the screened q grid."""
    m = read_metadata(filename)
    v = read_wavefunctions(filename, input_file, mesh_xy, bands=val_indices)
    c = read_wavefunctions(filename, input_file, mesh_xy, bands=cond_indices)
    V = read_interaction(filename, "bare", mesh_xy, nohead=nohead)
    W = read_interaction(filename, "screened", mesh_xy, nohead=nohead)
    grid = tuple(int(n) for n in m["grid"])
    def qgrid(a):
        return jax.jit(lambda x: x.reshape(grid+x.shape[-2:]).transpose(3,4,0,1,2),
            out_shardings=NamedSharding(mesh_xy, P("x", "y", None, None, None)))(a)
    return v, c, V[0], qgrid(W), qgrid(V) if full_exchange else None


def read_coarse_interactions(filename, input_file, mesh_xy):
    """Return canonical coarse ψ host cache and sharded full-q V/W arrays.

    Only one k row is replicated for each host transfer. Interaction arrays
    stay XY sharded and are never restored on a one-device shadow mesh.
    """
    from common.collectives import resolve_mesh
    if mesh_xy is None:
        mesh_xy = resolve_mesh()
    m = read_metadata(filename)
    full = read_wavefunctions(filename, input_file, mesh_xy)
    width = m["centroid_count"]
    host = np.empty(full.shape[:-1]+(width,), dtype=full.dtype)
    replicate = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh_xy, P()))
    for k in range(full.shape[0]):
        host[k] = np.asarray(replicate(full[k, ..., :width]))
    return {"psi": host, "kgrid": m["grid"], "enk": m["energies"],
            "band_window": m["band_window"],
            "Vqmunu": read_interaction(filename, "bare", mesh_xy),
            "W0": read_interaction(filename, "screened", mesh_xy)}


def read_downfold_geometry(filename: str) -> dict:
    """The small, replicated facts, read serially before any tensor moves."""
    with h5py.File(filename, "r") as f:
        _require_current(f)
        psi_key = "psi_parent_y"
        if psi_key not in f:
            raise ValueError(f"downfold: {filename} has no centroid wavefunctions.")
        if "V_qmunu" not in f:
            raise ValueError(
                f"downfold: {filename} has no V_qmunu.")
        if "W0_qmunu" not in f:
            raise ValueError(
                f"downfold: {filename} has no W0_qmunu — the parent GW run "
                f"wrote its V and psi but never got as far as persisting the "
                f"screened interaction.  A downfold of V alone is a "
                f"legitimate thing to want and is not what this driver does; "
                f"finish the parent run first.")
        if not bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            raise ValueError(
                f"downfold: {filename} carries W0_qmunu but its W0_ready "
                f"flag is FALSE — the dataset is the all-zeros PLACEHOLDER "
                f"the writer pre-allocates, not screening.  Downfolding it "
                f"would produce a small bundle full of zeros that every "
                f"shape check passes; the flag exists because that happened "
                f"once already.  Re-run the parent GW to completion.")
        if not bool(f["V_qmunu"].attrs.get("V_ready", True)):
            raise ValueError(
                f"downfold: {filename} says V_ready = False.")
        geom = {
            "n_rmu_logical": (int(np.asarray(f["n_rmu_logical"])[()])
                              if "n_rmu_logical" in f
                              else int(f["V_qmunu"].shape[-1])),
            "kgrid": (tuple(int(v) for v in np.asarray(f["kgrid"])[:])
                      if "kgrid" in f else None),
            "band_window": (np.asarray(f["band_window"])[:].astype(np.int64)
                            if "band_window" in f else None),
            "band_window_split": (
                np.asarray(f["band_window_split"])[:].astype(np.int64)
                if "band_window_split" in f else None),
            "nb": int(f[psi_key].shape[1]),
            "nk": int(f["enk_full"].shape[0]),
            "nspinor": int(f[psi_key].shape[2]),
            "vhead": (np.asarray(f["vhead"])[()] if "vhead" in f else None),
            "whead": (np.asarray(f["whead"][:]) if "whead" in f else None),
            "omega_grid": (np.asarray(f["whead"].attrs["omega_grid"])
                           if "whead" in f and "omega_grid" in f["whead"].attrs
                           else None),
            # The head INTEGRAND's S tensor.  It rides through a downfold
            # untouched for the same reason vhead/whead do: the head channel
            # is a property of the cell and the screening, not of the ISDF
            # basis the tensors were compressed into.  Dropping it would make
            # a downfolded bundle silently unable to densify W (the child
            # would have to rebuild S from dipole.h5, which a bundle-only
            # consumer has no path to).
            "S_cart": (np.asarray(f["S_cart_head"][:])
                       if "S_cart_head" in f else None),
            "centroids_charge_md5": f.attrs.get("centroids_charge_md5"),
            "present": tuple(n for n in ("V_qmunu", "W0_qmunu", "V_qmunu_nohead", "W0_qmunu_nohead") if n in f),
        }
    if geom["kgrid"] is None:
        raise ValueError(
            f"downfold: {filename} carries no kgrid, so the q axis of "
            f"V/W cannot be split into (nkx, nky, nkz).  The Gram build is a "
            f"convolution over k and needs that split; there is no way to "
            f"guess it from the flat q extent.  The bundle predates the "
            f"kgrid stamp — regenerate it with a current gw_jax, or read the "
            f"grid off the WFN the parent run used and note that this "
            f"driver deliberately does not take a WFN (its whole premise is "
            f"that a finished restart is self-describing).")
    return geom



def read_interaction_orbits(src: str, mu_L: int, print_fn):
    """The parent's ``QirrTables``, or ``None`` when it stores the full BZ.

    THE TABLE IS READ, NOT RE-DERIVED, and that is the whole point.  Building
    ``(α, L)`` from geometry needs the parent's symmetry ops, which live on a
    WFN this driver deliberately does not open — and even if it did, a
    second derivation of the permutation would be a second answer to a
    question the parent's own file already answers.  ``qirr_store`` persists
    the tables beside the tensor they deconstructed, so the selection is
    closed under exactly the permutation the parent's wedge storage rests on.

    Returns ``None`` — an ABSENCE, announced as one — when the parent's
    ``V_qmunu`` is stored on the full BZ.  Nothing is wrong in that case; the
    selection is point-granular, closure is unmeasured, and the child cannot
    be wedge-stored because there is no wedge in its lineage to store it on.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import dataset_q_storage, read_tables

    try:
        with h5py.File(src, "r") as f:
            storage = dataset_q_storage(f["V_qmunu"])
        if storage != "ibz":
            print_fn(
                "  [downfold/star] the parent stores V_qmunu on the FULL BZ, "
                "so it carries no centroid source map and this run's "
                "selection is POINT-GRANULAR: orbit closure is UNMEASURED, "
                "which is an absence and not a pass, and the child cannot be "
                "wedge-stored (there is no wedge in its lineage).  Set the "
                "PARENT run's restart_q_storage = auto to get both.")
            return None
        tables = read_tables(src, "V_qmunu")
    except (KeyError, ValueError, OSError) as exc:
        print_fn(
            f"  [downfold/star] the parent's q_irr tables could not be read "
            f"({type(exc).__name__}: {exc}).  Falling back to a "
            f"POINT-GRANULAR selection with closure UNMEASURED — an absence, "
            f"not a pass.")
        return None

    perm = np.asarray(tables.sym_perm)
    if int(perm.shape[1]) < int(mu_L):
        raise ValueError(
            f"downfold: the parent's stored sym_perm describes "
            f"{int(perm.shape[1])} centroids but the bundle declares "
            f"mu_L={mu_L}.  The table and the tensor are not about the same "
            f"centroid set, and selecting orbits against the wrong table "
            f"would produce a child whose 'orbits' are arbitrary index "
            f"classes.")
    print_fn(
        f"  [downfold/star] parent centroid source map read from its own "
        f"q_irr tables: {int(perm.shape[0])} op rows "
        f"({int(tables.n_sym_spatial)} spatial + TRS) over "
        f"{int(perm.shape[1])} centroids, {int(np.asarray(tables.q_irr_frac).shape[0])} "
        f"of {int(np.asarray(tables.irr_idx_q).shape[0])} q on the wedge.")
    return tables



def read_downfold_inputs(filename, input_file, mesh_xy):
    """Return authenticated geometry, full-k faces and linear interactions."""
    if not input_file:
        directory = os.path.dirname(filename)
        if os.path.basename(directory) == "tmp":
            directory = os.path.dirname(directory)
        input_file = os.path.join(directory, "cohsex.in")
    geom = read_downfold_geometry(filename)
    state = load_restart_state_from_h5(filename, mesh_xy)
    psi, = unfold_parent_faces((state.psi_nmu_parent,), filename, input_file, mesh_xy)
    state.psi_rmu_Y = jax.lax.with_sharding_constraint(
        psi, NamedSharding(mesh_xy, P(None, None, None, "y")))
    state.psi_rmuT_X = jax.lax.with_sharding_constraint(
        jnp.conj(psi).transpose(0, 3, 1, 2),
        NamedSharding(mesh_xy, P(None, "x", None, None)))
    state.psi_nmu_parent = state.psi_mun_parent = None
    tensors = {name: state.V_qmunu if name == "V_qmunu" else
               read_munu_tensor_from_h5(filename, name, mesh_xy)
               for name in geom["present"]}
    return geom, state, tensors


def load_dipole_h5(path: str | Path):
    """Load dipole.h5 (psp.get_dipole_mtxels output).

    Returns
    -------
    dipole_cart : (3, nk, nb, nb) complex128 — ⟨mk|v̂_α|nk⟩ (Ry), at the arm
                  the file was built with (its ``prov_vnl_velocity_sign``).
    deltaE      : (nk, nb, nb) float64       — E_b - E_b' (Ry).
    attrs       : dict with ``nbands, nk``.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"dipole file {path!s} not found.  This is a PRODUCED input, not a "
            f"deck file that ships with the system: build it from the deck's "
            f"WFN with\n"
            f"    python3 -u -m psp.get_dipole_mtxels -i <deck>.in --skip-vnl "
            f"--out {Path(path).name}\n"
            f"``--skip-vnl`` writes the momentum operator only, which is the "
            f"arm that matches BerkeleyGW's ``use_momentum``; drop it to get "
            f"the full velocity including the nonlocal commutator.")
    with h5py.File(str(path), "r") as f:
        dipole_cart = np.asarray(f["dipole_cart"][:], dtype=np.complex128)
        deltaE = np.asarray(f["deltaE"][:], dtype=np.float64)
        attrs = dict(f.attrs)
        attrs["nbands"], attrs["nk"] = int(attrs["nbands"]), int(attrs["nk"])
    return dipole_cart, deltaE, attrs



def require_screened_bundle(filename, *, include_w=True, print_fn=print):
    """Authenticate the GW→BSE handoff, including both persistence receipts."""
    with h5py.File(filename, "r") as f:
        _require_current(f)
        for name, ready in (("V_qmunu", "V_ready"), ("W0_qmunu", "W0_ready")):
            if name not in f or not bool(f[name].attrs.get(ready, name == "V_qmunu")):
                raise RuntimeError(f"{filename}: {name} {ready} is false or absent")
    print_fn(f"  BSE restart handoff verified: {os.path.basename(filename)}")


def read_qp_wfn_stamp(path) -> dict | None:
    """Is ``path`` a LORRAX QP WFN?  ``None`` when the file does not say.

    Returns the stamp written by :func:`write_qp_wfn_h5` — ``scheme``,
    ``band_start``, ``band_stop``, ``source`` and optional method provenance
    — or ``None``.

    THREE OUTCOMES, AND ONLY ONE OF THEM IS "NO".  A missing file, an
    unreadable one, and one with no stamp all return ``None``, which means
    **unverifiable**: BerkeleyGW's ``pw2bgw`` output, every WFN.h5 written
    before this stamp, and a QP WFN produced by some other tool are
    indistinguishable here, and a consumer must not read ``None`` as proof
    that the file is mean-field.  A consumer may only use a POSITIVE answer
    to refuse; the absence licenses nothing (``TASTE.md``: an absence is a
    claim about what was searched).

    An unrecognised scheme string is returned as-is rather than mapped onto
    the current one — a caller comparing it against
    :data:`QP_WFN_SCHEME` can then say "this file was written by a different
    version" instead of silently accepting it.
    """
    from .qp_wfn import (
        QP_ENERGY_DEFINITION_ATTR,
        QP_SOLVER_ATTR,
        QP_WFN_ATTR,
        SIGMA_EVAL_PROVENANCE_ATTR,
    )
    try:
        with h5py.File(str(path), "r") as h5:
            raw = h5.attrs.get(QP_WFN_ATTR)
            if raw is None:
                return None
            scheme = raw.decode() if isinstance(raw, bytes) else str(raw)
            def _get(name, default=None):
                v = h5.attrs.get(name, default)
                if isinstance(v, bytes):
                    return v.decode()
                return v
            return {
                "scheme": scheme,
                "band_start": (None if _get("qp_wfn_band_start") is None
                               else int(_get("qp_wfn_band_start"))),
                "band_stop": (None if _get("qp_wfn_band_stop") is None
                              else int(_get("qp_wfn_band_stop"))),
                "source": _get("qp_wfn_source", "") or "",
                "qp_solver": _get(QP_SOLVER_ATTR),
                "qp_energy_definition": _get(QP_ENERGY_DEFINITION_ATTR),
                "sigma_eval_provenance": _get(
                    SIGMA_EVAL_PROVENANCE_ATTR),
            }
    except (OSError, KeyError):
        return None



def read_qp_state_source_provenance(path) -> dict | None:
    """Read a restart source-state record; ``None`` means legacy/unproven."""
    from .qp_wfn import (QP_STATE_SOURCE_DATASET, _validate_qp_state_source)
    try:
        with h5py.File(str(path), "r") as h5:
            if QP_STATE_SOURCE_DATASET not in h5:
                return None
            raw = h5[QP_STATE_SOURCE_DATASET][()]
        payload = raw if isinstance(raw, bytes) else np.asarray(raw).tobytes()
        record = json.loads(payload.decode("utf-8", "strict").rstrip("\x00"))
    except (OSError, KeyError):
        return None
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(
            f"{path}: {QP_STATE_SOURCE_DATASET} is not valid UTF-8 JSON") \
            from exc
    return _validate_qp_state_source(record, path=str(path))



def read_qp_rotations_artifact(h5_path: str) -> dict:
    """Read one complete ``qp_wfn_rotations.h5`` Hamiltonian artifact.

    The physics arrays are unfolded through
    :func:`read_qp_rotations_full_bz`, so wedge and full-BZ storage retain
    one meaning.  The small identity datasets are read here as part of the
    same public format contract rather than independently in every physics
    driver.

    Returns ``U_mnk`` and ``E_qp_nk_rydberg`` on the full BZ together with
    ``band_range``, ``kpoints_crys``, ``kgrid`` and the optional legacy/source
    WFN fingerprint pair.  A partial artifact is refused: a rotation without
    its matched eigenvalues, band labels or k-set cannot define
    ``H_QP = U diag(E_QP) U^H``; one fingerprint attribute without the other
    cannot define which identity scheme was used.
    """
    from .qp_wfn import (
        QP_ROT_METADATA_DATASETS,
        QP_ROT_WFN_FINGERPRINT_ATTR,
        QP_ROT_WFN_FINGERPRINT_SCHEME_ATTR,
        _require_wfn_fingerprint,
    )
    path = os.fspath(h5_path)
    arrays = read_qp_rotations_full_bz(
        path, datasets=("U_mnk", "E_qp_nk_rydberg"))
    missing = [name for name in ("U_mnk", "E_qp_nk_rydberg")
               if name not in arrays]
    with h5py.File(path, "r") as h5:
        missing.extend(name for name in QP_ROT_METADATA_DATASETS
                       if name not in h5)
        if missing:
            raise ValueError(
                f"{os.path.basename(path)} is not a complete QP rotation "
                f"artifact; missing {sorted(set(missing))}.")
        arrays.update({
            "band_range": np.asarray(h5["band_range"][()], dtype=np.int64),
            "kpoints_crys": np.asarray(
                h5["kpoints_crys"][()], dtype=np.float64),
            "kgrid": np.asarray(h5["kgrid"][()], dtype=np.int64),
        })
        if "kirr_to_kfull" not in h5:
            raise ValueError(_REGENERATE)
        arrays["kirr_to_kfull"] = np.asarray(h5["kirr_to_kfull"][()], dtype=np.int64)
        has_scheme = QP_ROT_WFN_FINGERPRINT_SCHEME_ATTR in h5.attrs
        has_fingerprint = QP_ROT_WFN_FINGERPRINT_ATTR in h5.attrs
        if has_scheme != has_fingerprint:
            raise ValueError(
                f"{os.path.basename(path)} has an incomplete source-WFN "
                "identity: fingerprint and scheme attributes must appear "
                "together.")
        if has_fingerprint:
            def _text(value):
                return value.decode("ascii") if isinstance(value, bytes) \
                    else str(value)
            scheme = _text(h5.attrs[QP_ROT_WFN_FINGERPRINT_SCHEME_ATTR])
            fingerprint = _text(h5.attrs[QP_ROT_WFN_FINGERPRINT_ATTR])
            fingerprint = _require_wfn_fingerprint(
                fingerprint,
                where=(f"{os.path.basename(path)} source-WFN fingerprint"))
        else:
            scheme = fingerprint = None
        arrays["source_wfn_fingerprint_scheme"] = scheme
        arrays["source_wfn_fingerprint"] = fingerprint
    return arrays



def read_qp_rotations_full_bz(h5_path: str, datasets=None) -> dict:
    """``qp_wfn_rotations.h5``'s k-indexed arrays, ON THE FULL BZ.

    THE unfolding option the wedge form owes its consumers, and the reason
    the wedge form is safe to write at all: anything that wants the array
    the old writer produced calls this and gets it, wedge-stored file or
    not.  A full-BZ file is read verbatim — the unfold is not attempted,
    because the tables are not there and the rows are not stars.

    Reuses ``kin_ion.read_star_map`` for the stamp contract rather than
    re-implementing it, so the version number, the table names and every
    refusal have ONE definition across ``kin_ion.h5``, ``sigma_mnk.h5`` and
    this file.
    """
    from .qp_wfn import (QP_ROT_K_DATASETS, broadcast_ibz_to_full_bz)
    names = tuple(datasets) if datasets is not None else QP_ROT_K_DATASETS
    star = read_star_map(h5_path, names[0])
    out = {}
    with h5py.File(h5_path, "r") as f:
        for name in names:
            if name not in f:
                continue
            arr = np.asarray(f[name][()])
            out[name] = (arr if star is None
                         else np.asarray(broadcast_ibz_to_full_bz(arr, *star)))
    return out



def read_kirr_to_kfull(rot_file, wfn_kpoints, rot_kpoints, *, artifact=None):
    """The wedge→full-BZ map, READ from the rotation file, never re-derived.

    ``qp_wfn_rotations.h5`` already carries ``kirr_to_kfull``, written
    straight from ``sym.kirr_fullids`` by both producers
    (``gw.gw_jax``/``gw_output.write_results`` and
    ``gw.sc_iteration``).  The symmetry service builds that table by
    EXACT periodic match and RAISES on a miss (``maps.py:1340-1359``),
    and its contract is ``unfolded_kpts[kirr_fullids] == wfn.kpoints``.

    WHAT THIS REPLACED.  This module used to rebuild the same table with
    a ``np.argmin`` over summed coordinate distances at ``tol=1e-6``
    (``find_kpoint_mapping``), taking the nearest full-BZ k to each
    reduced one with no uniqueness check — so two k within tolerance of
    one another resolved silently to whichever came first, and the
    rebuilt table then selected the QP rotation ``U_mnk[ik_full]`` and
    the QP energy for that reduced k.  Those energies reach
    ``eqp{0,1}.dat`` through ``gw.eqp_bgw``, which reads
    ``kirr_to_kfull`` from this very file — so the two disagreeing about
    a k meant the eqp columns and the rotated WFN disagreed too.  There
    is now one table, produced by the service, and this module reads it.

    The coordinates are CHECKED against it rather than searched: the map
    must reproduce ``wfn.kpoints`` from ``kpoints_crys``, which is the
    service's own contract restated at the point of use.
    """
    if artifact is None:
        artifact = read_qp_rotations_artifact(rot_file)
    kirr_to_kfull = np.asarray(artifact["kirr_to_kfull"], dtype=np.int32)

    nk_red = len(wfn_kpoints)
    if kirr_to_kfull.shape != (nk_red,):
        raise ValueError(
            f"kirr_to_kfull has shape {kirr_to_kfull.shape}, expected "
            f"({nk_red},) — the rotation file and {nk_red}-point WFN are "
            f"not the same calculation.")
    if kirr_to_kfull.max(initial=-1) >= len(rot_kpoints):
        raise ValueError(
            f"kirr_to_kfull reaches full-BZ row {int(kirr_to_kfull.max())} "
            f"but kpoints_crys has only {len(rot_kpoints)} rows.")

    # The service's contract, checked here: unfolded_kpts[kirr_fullids]
    # IS wfn.kpoints.  Comparison is modulo a lattice vector because the
    # two files may hold a k in different periodic images.
    d = np.asarray(rot_kpoints)[kirr_to_kfull] - np.asarray(wfn_kpoints)
    d -= np.rint(d)
    worst = float(np.max(np.abs(d))) if d.size else 0.0
    if worst > 1e-6:
        bad = int(np.argmax(np.max(np.abs(d), axis=1)))
        raise ValueError(
            f"kirr_to_kfull[{bad}] = {int(kirr_to_kfull[bad])} points at "
            f"{np.asarray(rot_kpoints)[kirr_to_kfull[bad]].tolist()} but "
            f"WFN reduced k-point {bad} is {np.asarray(wfn_kpoints)[bad].tolist()} "
            f"(worst |Δk| = {worst:.2e}).  The rotation file and the WFN "
            f"disagree about the k-set.")
    return kirr_to_kfull



def _read_eval_energies_ev(
	path: str, *, kirr_to_kfull: np.ndarray, band_start: int, band_stop: int,
) -> np.ndarray:
	"""The IBZ × sigma-window energies (eV) a QP artifact carries.

	Reads either of the two files a run writes its converged spectrum to,
	because both are legitimate answers to "what did Σ get evaluated at"
	and which one is on disk depends on the deck:

	``qp_wfn_rotations.h5`` — ``E_qp_nk_rydberg``, ALREADY on the sigma
	    window (its band axis is ``band_range``), so it is indexed by
	    ``kirr_to_kfull`` and not sliced in bands.  It is read through
	    ``file_io.qp_wfn.read_qp_rotations_full_bz`` because that array may
	    be stored on the file wedge; ``kirr_to_kfull`` is always a full-BZ
	    index, so the indexing below is unchanged either way.  A file with
	    no ``k_storage`` attr comes back verbatim.
	``WFN_qp.h5``           — ``mf_header/kpoints/el``, the WFN file's own
	    k-set (the IBZ) and ALL bands, so it is sliced in bands and not
	    in k.
	"""
	with h5py.File(path, "r") as f:
		has_rot = "E_qp_nk_rydberg" in f
	if has_rot:
		e = np.asarray(read_qp_rotations_full_bz(
			path, datasets=("E_qp_nk_rydberg",))["E_qp_nk_rydberg"],
			dtype=np.float64)
		return e[np.asarray(kirr_to_kfull, dtype=np.int64)] * RYD_TO_EV
	with h5py.File(path, "r") as f:
		if "mf_header/kpoints/el" in f:
			e = np.asarray(f["mf_header/kpoints/el"], dtype=np.float64)
			return e[0][:, int(band_start):int(band_stop)] * RYD_TO_EV
	raise ValueError(
		f"{os.path.basename(path)} carries neither 'E_qp_nk_rydberg' "
		f"(qp_wfn_rotations.h5) nor 'mf_header/kpoints/el' (WFN_qp.h5); it "
		f"cannot say what energies Sigma was evaluated at.")



class BispinorVqReader:
    """Open a bispinor V_q HDF5 written by :func:`compute_V_q_bispinor_g_flat_to_h5`.

    Provides a uniform interface over the 16 (μ_L, ν_L) blocks:

    * ``get_tile(μ_L, ν_L)``        — JAX array on the (None, 'x', 'y')
                                       sharding, materialised on demand.
                                       For zero-by-gauge tiles returns
                                       a zeros array sized appropriately.
                                       For Hermitian-redundant tiles
                                       reads the companion + applies
                                       ``conj(swapaxes(.., -1, -2))``.
    Caller manages the lifecycle (use as a context manager).
    """

    def __init__(self, filename: Path | str, mesh_xy: Mesh, *, mu_bases=None,
                 family_plans=None):
        from gw.v_q_bispinor import (
            UNIQUE_TILES,
            V_QMUNU_DATA_READY_DATASET,
            V_QMUNU_FORMAT,
            V_QMUNU_INVENTORY_DATASET,
            _expected_unique_tile_inventory,
            _validate_unique_tile_datasets,
            tile_dataset_name,
        )
        from file_io.slab_io import SlabIO
        import h5py
        self._filename = Path(filename)
        self._mesh = mesh_xy
        self._mu_bases = mu_bases
        self.q_tables = {}
        self.q_headers = {}

        # Small metadata scalars are written via SlabIO.write_attr (which
        # creates a dataset in the file).  Read them via h5py — every rank
        # opens its own 'r' handle for a few-byte read; broadcast overhead
        # would dominate.  Validate all metadata before opening SlabIO: its
        # constructor opens a collective PhdfCtx which only __exit__ can
        # close, so a constructor-time refusal must happen first.
        with h5py.File(self._filename, "r") as f:
            def _read_scalar(name):
                if name not in f:
                    raise ValueError(
                        f"{self._filename}: required bispinor-V metadata "
                        f"dataset '{name}' is absent.")
                d = f[name]
                if d.shape != () and name != "kgrid":
                    raise ValueError(
                        f"{self._filename}: metadata dataset '{name}' "
                        f"must be scalar, got shape {d.shape}.")
                v = d[()] if d.shape == () else d[:]
                return v
            fmt = _read_scalar("v_qmunu_format")
            if isinstance(fmt, bytes):
                fmt = fmt.decode("utf-8")
            if str(fmt) != V_QMUNU_FORMAT:
                raise ValueError(
                    f"{self._filename}: v_qmunu_format='{fmt}', "
                    f"expected '{V_QMUNU_FORMAT}'.  Wrong file or stale "
                    f"format from a different LORRAX revision."
                )
            self.kgrid = tuple(int(x) for x in _read_scalar("kgrid"))
            self.n_rmu_C = int(_read_scalar("n_rmu_C"))
            self.n_rmu_T = int(_read_scalar("n_rmu_T"))
            self.n_q_total = int(_read_scalar("n_q_total"))
            ready_dataset = f.get(V_QMUNU_DATA_READY_DATASET)
            if (ready_dataset is None or ready_dataset.shape != ()
                    or np.dtype(ready_dataset.dtype) != np.dtype(np.bool_)
                    or not bool(ready_dataset[()])):
                raise ValueError(
                    f"{self._filename}: bispinor-V data_ready receipt is "
                    "absent, malformed, or false; the tile file is not "
                    "certified complete.")
            raw_inventory = _read_scalar(V_QMUNU_INVENTORY_DATASET)
            if isinstance(raw_inventory, bytes):
                raw_inventory = raw_inventory.decode("utf-8")
            try:
                inventory = json.loads(str(raw_inventory))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{self._filename}: malformed bispinor-V unique-tile "
                    "inventory.") from exc
            expected_inventory = _expected_unique_tile_inventory(
                n_q_total=self.n_q_total, n_rmu_C=self.n_rmu_C,
                n_rmu_T=self.n_rmu_T)
            if inventory != expected_inventory:
                raise ValueError(
                    f"{self._filename}: published bispinor-V unique-tile "
                    "inventory does not match the canonical inventory "
                    "derived from the file's logical geometry.")
            _validate_unique_tile_datasets(
                f, filename=self._filename,
                n_q_total=self.n_q_total, n_rmu_C=self.n_rmu_C,
                n_rmu_T=self.n_rmu_T)
            from symmetry_maps import read_tensor, read_tables, QIRR_VERSION_ATTR
            for pair in UNIQUE_TILES:
                name = tile_dataset_name(*pair)
                try:
                    if QIRR_VERSION_ATTR not in f[name].attrs:
                        raise ValueError("Missing q-IBZ format stamp.")
                    _, header = read_tensor(f, name, metadata_only=True)
                    tables = read_tables(f, name)
                    family = int(pair[0] != 0)
                    if (family in self.q_tables and
                            (self.q_tables[family].digest() != tables.digest()
                             or self.q_headers[family].centroid_hash != header.centroid_hash)):
                        raise ValueError("V tiles disagree on their family symmetry tables.")
                    self.q_tables[family] = tables
                    self.q_headers[family] = header
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"{self._filename}: unstamped or torn V tiles; "
                        "legacy full-q files require rerun with restart=false.") from exc
                if (tables.n_q_ibz != self.n_q_total or
                        tables.n_q_full != int(np.prod(self.kgrid))):
                    raise ValueError("Bispinor V q-IBZ tables disagree with its geometry.")

        if family_plans is not None:
            from symmetry_maps import (bgw_integer_q_to_fractional,
                                       verify_centroid_orbit_closure)
            from gw.qgrid_symmetry import qgrid_trs_policy_for
            if mu_bases is None:
                raise ValueError("V family authentication requires the run centroid bases.")
            for family, plan in enumerate(family_plans):
                if plan is None:
                    continue
                basis, sym = mu_bases[family], plan.sym
                policy = qgrid_trs_policy_for(sym=sym, irr_idx_q=sym.irr_idx_q,
                    sym_idx_q=sym.sym_idx_q, kgrid=self.kgrid,
                    n_sym_spatial=plan.n_sym_spatial, context="photon V reader")
                closure = verify_centroid_orbit_closure(
                    basis.canonical_indices / np.asarray(plan.fft_grid),
                    plan.spatial_ops, tnp=plan.translations)
                if self.q_headers[family].centroid_hash != closure.centroid_hash:
                    raise ValueError("Photon V centroid set differs from the run; rerun restart=false.")
                table = self.q_tables[family]
                perm, wraps = basis.pack_tables(table.sym_perm, table.L_table)
                qfrac = bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, self.kgrid)
                if not (np.array_equal(table.q_irr_frac, qfrac)
                        and np.array_equal(table.irr_idx_q, sym.irr_idx_q)
                        and np.array_equal(table.sym_idx_q, policy.unfold_sym_idx)
                        and np.array_equal(perm, plan.sym_perm)
                        and np.array_equal(wraps, plan.L_table)):
                    raise ValueError("Photon V tables differ from the authenticated run; rerun restart=false.")

        self._io = SlabIO(self._filename, mode="r", mesh=mesh_xy)
        self._io.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._io.__exit__(*exc)

    def _zero_tile(self, mu_L: int, nu_L: int) -> jax.Array:
        n_L = self.n_rmu_C if mu_L == 0 else self.n_rmu_T
        n_R = self.n_rmu_C if nu_L == 0 else self.n_rmu_T
        sharding = NamedSharding(self._mesh, P(None, 'x', 'y'))
        n_L_p, n_R_p = self._padded_shape_LR(n_L, n_R)
        return jax.lax.with_sharding_constraint(
            jnp.zeros((self.n_q_total, n_L_p, n_R_p), dtype=jnp.complex128),
            sharding,
        )

    def _tile_shape(self, mu_L: int, nu_L: int) -> tuple[int, int, int]:
        n_L = self.n_rmu_C if mu_L == 0 else self.n_rmu_T
        n_R = self.n_rmu_C if nu_L == 0 else self.n_rmu_T
        return (self.n_q_total, n_L, n_R)

    def _padded_shape_LR(self, n_L: int, n_R: int) -> tuple[int, int]:
        """Round n_L, n_R up to the total mesh-product (``gx*gy``).  This
        mirrors the write-side μ padding in
        ``gw.v_q_g_flat._compute_V_q_g_flat_one_tile`` (its ``_pad``
        helper, which also pads to ``p_x*p_y``) and matches the
        ψ-side μ extent built by ``load_centroids_band_chunked`` — so a
        single pad here makes Σ^B's V tile broadcast against ψ with no
        further padding step in sigma_x_bispinor.

        Padding to ``gx*gy`` (rather than per-axis ``gx``/``gy``) is also
        what makes sharded reads with spec P(None,'x','y') divide
        cleanly under any 2D mesh factorisation.

        Routed through ``runtime.padding.padded_mu_extent`` so the
        test-only LORRAX_EXTRA_MU_PAD knob stays consistent with the
        ψ-side / write-side extents."""
        from runtime.padding import padded_mu_axis
        return (padded_mu_axis(int(n_L), self._mesh).carrier,
                padded_mu_axis(int(n_R), self._mesh).carrier)

    def get_tile(self, mu_L: int, nu_L: int) -> jax.Array:
        """Read one q-IBZ tile and pack its two centroid families at the file boundary."""
        from gw.v_q_bispinor import (HERMITIAN_PAIRS, ZERO_TILES, tile_dataset_name)
        if not (0 <= mu_L <= 3 and 0 <= nu_L <= 3):
            raise ValueError(f"Lorentz indices must be in 0..3; got {(mu_L, nu_L)}.")
        if (mu_L, nu_L) in ZERO_TILES:
            tile = self._zero_tile(mu_L, nu_L)
        else:
            transpose = (mu_L, nu_L) in HERMITIAN_PAIRS
            source = HERMITIAN_PAIRS.get((mu_L, nu_L), (mu_L, nu_L))
            n_L, n_R = self._tile_shape(*source)[1:]
            n_L_p, n_R_p = self._padded_shape_LR(n_L, n_R)
            tile = self._io.read_slab(
                tile_dataset_name(*source),
                shape=(self.n_q_total, n_L_p, n_R_p), mesh=self._mesh,
                partition_spec=P(None, 'y', 'x') if transpose else P(None, 'x', 'y'),
                dtype=jnp.complex128)
            if transpose:
                tile = jnp.conj(jnp.swapaxes(tile, -1, -2))
        if self._mu_bases is not None:
            left = self._mu_bases[int(mu_L != 0)]
            right = self._mu_bases[int(nu_L != 0)]
            if left is right:
                return left.pack_operator(tile)
            tile = left.pack_axis(tile, -2)
            tile = right.pack_axis(tile, -1)
        return tile

    @property
    def filename(self) -> Path:
        return self._filename



def read_photon_charge(filename, mesh_xy):
    """Return the full-q charge block of an authenticated photon artifact."""
    with BispinorVqReader(filename, mesh_xy) as reader:
        tile = reader.get_tile(0, 0)
        return _unfold_wedge(tile, reader.q_tables[0], tile.shape[-1], mesh_xy)


def read_photon_gamma(directory, mesh_xy, bases):
    """Return canonical Gamma one-leg factors, each sharded on μ_X."""
    from .slab_io import SlabIO
    path = os.path.join(directory, "v_q_bispinor.h5")
    with h5py.File(path, "r") as f:
        names = tuple(f"photon_g0_vectors_{channel}" for channel in range(len(bases)))
        if any(name not in f or f[name].shape != (1, basis.n_logical)
               for name, basis in zip(names, bases)):
            raise ValueError(_REGENERATE)
    with SlabIO(path, mode="r", mesh=mesh_xy) as io:
        return tuple(io.read_slab(name, shape=(1,basis.n_canonical),
            partition_spec=P(None,"x"), dtype=jnp.complex128)
            for name,basis in zip(names,bases))


def read_eqp_assembly_receipt(filepath):
	"""Read and validate a v5 charge-only or v6 component-aware receipt.

	No canonical/candidate dataset means a legacy artifact.  A candidate or
	unknown/partial canonical schema refuses rather than falling back to raw.
	v6 additionally authenticates aggregate ``Hdir`` against persisted scalar
	``V_H`` and transverse ``H_T``; v5 refuses those companions entirely.
	"""
	from file_io.sigma_output import (
		EQP_ASSEMBLY_BAND_START_ATTR,
		EQP_ASSEMBLY_BAND_STOP_ATTR,
		EQP_ASSEMBLY_CANDIDATE_DATASET,
		EQP_ASSEMBLY_COMPONENT_SCHEMA_VERSION,
		EQP_ASSEMBLY_C_BASIS_ATTR,
		EQP_ASSEMBLY_C_OMEGA_CANDIDATE_DATASET,
		EQP_ASSEMBLY_C_OMEGA_DATASET,
		EQP_ASSEMBLY_DATASET,
		EQP_ASSEMBLY_DEGENERACY_POLICIES,
		EQP_ASSEMBLY_DEGENERACY_POLICY_ATTR,
		EQP_ASSEMBLY_DEGENERACY_TOL_ATTR,
		EQP_ASSEMBLY_DFT_BAND_BASIS,
		EQP_ASSEMBLY_EXPECTED_DATASET,
		EQP_ASSEMBLY_FILE_ROWS_ATTR,
		EQP_ASSEMBLY_HARTREE_COMPONENTS_CANDIDATE_DATASET,
		EQP_ASSEMBLY_HARTREE_COMPONENTS_DATASET,
		EQP_ASSEMBLY_HARTREE_STATE_ATTR,
		EQP_ASSEMBLY_HARTREE_STATE_LIVE_GSPACE,
		EQP_ASSEMBLY_HX_BASIS_ATTR,
		EQP_ASSEMBLY_K_STORAGE_ATTR,
		EQP_ASSEMBLY_K_STORAGE_WEDGE,
		EQP_ASSEMBLY_SCHEMA_VERSION,
		EQP_ASSEMBLY_SCHEMA_VERSION_ATTR,
		EQP_ASSEMBLY_STATE_ATTR,
		EQP_ASSEMBLY_STATE_READY,
		OMEGA_DATASET,
		SIGMA_DIRECT_COMPONENT_DATASETS,
		SIGMA_EVAL_DATASET,
		SIGMA_OPERATOR_STATE_ATTR,
		SIGMA_OPERATOR_STATE_VERSION_ATTR,
		STACK_H5PY,
		_assert_direct_field_sum,
		_raw_operator_stamp_errors,
		_validate_raw_direct_component_contract,
		open_scope,
	)
	abs_path = os.path.abspath(filepath)

	def _text(value):
		return value.decode("utf-8") if isinstance(value, bytes) else str(value)

	with open_scope(
			abs_path, STACK_H5PY, "r",
			where="read_eqp_assembly_receipt"), \
			h5py.File(abs_path, "r") as h5:
		expected = (h5[EQP_ASSEMBLY_EXPECTED_DATASET][()]
			if EQP_ASSEMBLY_EXPECTED_DATASET in h5 else None)
		canonical_present = EQP_ASSEMBLY_DATASET in h5
		curve_present = EQP_ASSEMBLY_C_OMEGA_DATASET in h5
		components_present = EQP_ASSEMBLY_HARTREE_COMPONENTS_DATASET in h5
		raw_component_present = any(
			name in h5 for name in SIGMA_DIRECT_COMPONENT_DATASETS)
		candidate_names = [name for name in (
			EQP_ASSEMBLY_CANDIDATE_DATASET,
			EQP_ASSEMBLY_C_OMEGA_CANDIDATE_DATASET,
			EQP_ASSEMBLY_HARTREE_COMPONENTS_CANDIDATE_DATASET) if name in h5]
		new_schema = bool(
			expected is not None or canonical_present or curve_present
			or components_present or raw_component_present or candidate_names)
		present, bad = _raw_operator_stamp_errors(h5)
		if new_schema:
			# Every read that claims any new-schema state validates EVERY raw
			# operator cube first.  A complete receipt never blesses a missing,
			# unknown, or partially stamped operator payload.
			if not present or bad:
				raise ValueError(
					f"{os.path.basename(abs_path)} has a new EQP schema but "
					f"partial/unknown raw operator state on {bad or 'no cubes'}.")
		else:
			# New creating writers stamp every raw operator cube.  That state
			# makes the receipt mandatory, so a crash between cube close and the
			# output append refuses instead of masquerading as a legacy file.
			tagged = [name for name in present if (
				SIGMA_OPERATOR_STATE_ATTR in h5[name].attrs
				or SIGMA_OPERATOR_STATE_VERSION_ATTR in h5[name].attrs)]
			if tagged:
				if bad:
					raise ValueError(
						f"{os.path.basename(abs_path)} has partial/unknown raw "
						f"operator-state stamps on {bad}.")
				raise ValueError(
					f"{os.path.basename(abs_path)} is a new raw-operator artifact "
					f"but {EQP_ASSEMBLY_DATASET} is missing; the run stopped "
					"before its live EQP assembly receipt closed.")
			return None
		try:
			expected_version = int(expected)
		except (TypeError, ValueError):
			expected_version = -1
		valid_versions = (
			EQP_ASSEMBLY_SCHEMA_VERSION,
			EQP_ASSEMBLY_COMPONENT_SCHEMA_VERSION,
		)
		if expected is None or expected_version not in valid_versions:
			raise ValueError(
				f"{os.path.basename(abs_path)} has unknown/partial EQP receipt "
				f"expectation {expected!r}; expected "
				f"{valid_versions[0]} or {valid_versions[1]}.")
		component_aware = (
			expected_version == EQP_ASSEMBLY_COMPONENT_SCHEMA_VERSION)
		raw_components = _validate_raw_direct_component_contract(
			h5, required=component_aware)
		if raw_components != component_aware:
			raise ValueError(
				f"schema-v{expected_version} receipt/raw component mismatch: "
				f"raw component-aware={raw_components}.")
		if candidate_names:
			raise ValueError(
				f"{os.path.basename(abs_path)} has an interrupted "
				f"candidate write {candidate_names}; refusing a stale or "
				"partial EQP reconstruction.")
		if (not canonical_present or not curve_present
				or components_present != component_aware):
			missing = [name for name, there in (
				(EQP_ASSEMBLY_DATASET, canonical_present),
				(EQP_ASSEMBLY_C_OMEGA_DATASET, curve_present),
				(EQP_ASSEMBLY_HARTREE_COMPONENTS_DATASET,
				 components_present or not component_aware)) if not there]
			extra = ([] if component_aware or not components_present else
				[EQP_ASSEMBLY_HARTREE_COMPONENTS_DATASET])
			raise ValueError(
				f"{os.path.basename(abs_path)} expects an EQP assembly receipt "
				f"but missing={missing}, unexpected={extra}.")
		ds = h5[EQP_ASSEMBLY_DATASET]
		if not isinstance(ds, h5py.Dataset):
			raise ValueError(f"{EQP_ASSEMBLY_DATASET} is not an HDF5 dataset.")
		required_attrs = (
			EQP_ASSEMBLY_SCHEMA_VERSION_ATTR,
			EQP_ASSEMBLY_STATE_ATTR,
			EQP_ASSEMBLY_HX_BASIS_ATTR,
			EQP_ASSEMBLY_C_BASIS_ATTR,
			EQP_ASSEMBLY_K_STORAGE_ATTR,
			EQP_ASSEMBLY_FILE_ROWS_ATTR,
			EQP_ASSEMBLY_HARTREE_STATE_ATTR,
			EQP_ASSEMBLY_BAND_START_ATTR,
			EQP_ASSEMBLY_BAND_STOP_ATTR,
			EQP_ASSEMBLY_DEGENERACY_POLICY_ATTR,
			EQP_ASSEMBLY_DEGENERACY_TOL_ATTR,
		)
		missing_attrs = [name for name in required_attrs if name not in ds.attrs]
		if missing_attrs:
			raise ValueError(
				f"partial {EQP_ASSEMBLY_DATASET}: missing attrs {missing_attrs}.")
		version = int(ds.attrs[EQP_ASSEMBLY_SCHEMA_VERSION_ATTR])
		state = _text(ds.attrs[EQP_ASSEMBLY_STATE_ATTR])
		hx_basis = _text(ds.attrs[EQP_ASSEMBLY_HX_BASIS_ATTR])
		c_basis = _text(ds.attrs[EQP_ASSEMBLY_C_BASIS_ATTR])
		k_storage = _text(ds.attrs[EQP_ASSEMBLY_K_STORAGE_ATTR])
		hartree_state = _text(ds.attrs[EQP_ASSEMBLY_HARTREE_STATE_ATTR])
		policy = _text(ds.attrs[EQP_ASSEMBLY_DEGENERACY_POLICY_ATTR])
		if version != expected_version:
			raise ValueError(
				f"{EQP_ASSEMBLY_DATASET} schema {version} disagrees with "
				f"expected marker {expected_version}.")
		if (state != EQP_ASSEMBLY_STATE_READY
				or hx_basis != EQP_ASSEMBLY_DFT_BAND_BASIS
				or c_basis != EQP_ASSEMBLY_DFT_BAND_BASIS
				or k_storage != EQP_ASSEMBLY_K_STORAGE_WEDGE
				or hartree_state != EQP_ASSEMBLY_HARTREE_STATE_LIVE_GSPACE
				or policy not in EQP_ASSEMBLY_DEGENERACY_POLICIES):
			raise ValueError(
				f"unsupported {EQP_ASSEMBLY_DATASET} semantics: state={state!r}, "
				f"H/X basis={hx_basis!r}, C basis={c_basis!r}, "
				f"k_storage={k_storage!r}, "
				f"hartree_state={hartree_state!r}, policy={policy!r}.")
		values = np.asarray(ds[()], dtype=np.complex128)
		c_omega = np.asarray(
			h5[EQP_ASSEMBLY_C_OMEGA_DATASET][()], dtype=np.complex128)
		h_components = (np.asarray(
			h5[EQP_ASSEMBLY_HARTREE_COMPONENTS_DATASET][()],
			dtype=np.complex128) if component_aware else None)
		file_rows = np.asarray(
			ds.attrs[EQP_ASSEMBLY_FILE_ROWS_ATTR], dtype=np.int64)
		b0 = int(ds.attrs[EQP_ASSEMBLY_BAND_START_ATTR])
		b1 = int(ds.attrs[EQP_ASSEMBLY_BAND_STOP_ATTR])
		if (b0 < 0 or b1 <= b0 or values.ndim != 3 or values.shape[0] != 3
				or values.shape[2] != b1 - b0):
			raise ValueError(
				f"malformed {EQP_ASSEMBLY_DATASET} shape {values.shape} "
				f"for band window [{b0},{b1}).")
		if not np.all(np.isfinite(values)):
			raise ValueError(f"{EQP_ASSEMBLY_DATASET} contains non-finite values.")
		if np.any(np.imag(values[:2]) != 0.0):
			raise ValueError(
				f"{EQP_ASSEMBLY_DATASET} H/X rows must be exactly real.")
		if component_aware:
			if (h_components.shape != (2,) + values.shape[1:]
					or not np.all(np.isfinite(h_components))):
				raise ValueError(
					f"malformed {EQP_ASSEMBLY_HARTREE_COMPONENTS_DATASET} "
					f"shape {h_components.shape}; expected "
					f"{(2,) + values.shape[1:]}, or non-finite values.")
			if np.any(np.imag(h_components) != 0.0):
				raise ValueError(
					f"{EQP_ASSEMBLY_HARTREE_COMPONENTS_DATASET} must be "
					"exactly real.")
			_assert_direct_field_sum(
				np.real(values[0]), np.real(h_components[0]),
				np.real(h_components[1]), where="persisted EQP receipt")
		if (c_omega.ndim != 3 or c_omega.shape[1:] != values.shape[1:]
				or not np.all(np.isfinite(c_omega))):
			raise ValueError(
				f"malformed {EQP_ASSEMBLY_C_OMEGA_DATASET} shape "
				f"{c_omega.shape} for H/X/C {values.shape}, or non-finite values.")
		omega_rel_ev, omega_reference_ev, omega_reference_provenance = (
			_omega_metadata_from_open_h5(h5))
		if omega_rel_ev is None or omega_rel_ev.shape != (c_omega.shape[0],):
			raise ValueError(
				f"{EQP_ASSEMBLY_C_OMEGA_DATASET} has {c_omega.shape[0]} omega "
				f"rows but {OMEGA_DATASET!r} is absent or has shape "
				f"{None if omega_rel_ev is None else omega_rel_ev.shape}.")
		if (file_rows.ndim != 1 or file_rows.shape[0] != values.shape[1]
				or (file_rows.size and np.min(file_rows) < 0)
				or np.unique(file_rows).size != file_rows.size):
			raise ValueError(
				f"malformed {EQP_ASSEMBLY_FILE_ROWS_ATTR} {file_rows.tolist()} "
				f"for {values.shape[1]} file-wedge rows.")
		tol_ry = float(ds.attrs[EQP_ASSEMBLY_DEGENERACY_TOL_ATTR])
		if not np.isfinite(tol_ry) or tol_ry < 0.0:
			raise ValueError(
				f"malformed {EQP_ASSEMBLY_DEGENERACY_TOL_ATTR}={tol_ry!r}.")
		eval_rel_ev, eval_provenance, eval_coverage = (
			_eval_metadata_from_open_h5(h5))
		if eval_rel_ev is not None and eval_rel_ev.shape != values.shape[1:]:
			# The stamp is on the star wedge (the cube writer put it there);
			# the receipt is on the file wedge.  Pairing them would hand a
			# star parent's evaluation energies to another member of that
			# star, which k_irr_rows_for refuses, so this refuses and names
			# which mismatch it is.
			diagnosis = ""
			nk_receipt, nk_eval = int(values.shape[1]), int(eval_rel_ev.shape[0])
			if nk_eval != nk_receipt:
				diagnosis = (
					f"  The stamp has {nk_eval} k rows against the receipt's "
					f"{nk_receipt}: the stamp is on the cube's STAR wedge and "
					f"the receipt on the FILE wedge, and pairing them needs the "
					f"star substitution this module refuses (k_irr_rows_for).  "
					f"OPEN RULING for a deck whose two wedges differ — stamp "
					f"the evaluation energies on the file wedge, or declare "
					f"them star-invariant and unfold: "
					f"docs/reports/INTEG_CHECKLIST_LANDINGS_2026-08-27.md.")
			raise ValueError(
				f"{SIGMA_EVAL_DATASET} has shape {eval_rel_ev.shape}; expected "
				f"the receipt's file-wedge/window shape {values.shape[1:]}."
				f"{diagnosis}")
		return {
			"hartree_diag_ev": np.real(values[0]),
			"hartree_scalar_diag_ev": (
				None if h_components is None else np.real(h_components[0])),
			"hartree_transverse_diag_ev": (
				None if h_components is None else np.real(h_components[1])),
			"sigma_x_diag_ev": np.real(values[1]),
			"sigma_c_at_dft_diag_ev": values[2],
			"sigma_c_omega_diag_ev": c_omega,
			"omega_rel_ev": omega_rel_ev,
			"omega_reference_ev": omega_reference_ev,
			"omega_reference_provenance": omega_reference_provenance,
			"eval_energies_rel_ev": eval_rel_ev,
			"eval_energies_provenance": eval_provenance,
			"eval_coverage": eval_coverage,
			"band_start": b0,
			"band_stop": b1,
			"file_wedge_full_bz_rows": file_rows,
			"degeneracy_policy": policy,
			"degeneracy_tol_ry": tol_ry,
			"hartree_exchange_basis": hx_basis,
			"correlation_basis": c_basis,
			"schema_version": expected_version,
		}



def read_eval_energies(filepath):
	"""``(eval_rel_ev, provenance, coverage)`` off a ``sigma_mnk.h5``.

	``(None, None, None)`` means the file predates the stamp — NOT that the
	cube was evaluated at E_DFT.  A consumer must treat the two differently:
	the second is a fact it can act on, the first is an absence, and
	collapsing them is exactly how ``make_eqp_bgw`` came to linearize a
	self-consistent cube at E_DFT without saying so.

	``coverage`` is ``{"n_uncovered", "fraction_uncovered", "policy"}`` when
	the writer stamped it, else ``None``.

	The array comes back on the file's OWN k rows (the star wedge when the
	file carries one), like every other dataset here; the caller remaps
	through the same ``k_irr_rows_for`` it uses for the cubes.
	"""
	from file_io.sigma_output import (
		STACK_H5PY,
		open_scope,
	)
	abs_path = os.path.abspath(filepath)
	with open_scope(
			abs_path, STACK_H5PY, "r", where="read_eval_energies"), \
			h5py.File(abs_path, "r") as h5:
		return _eval_metadata_from_open_h5(h5)



def read_omega_reference(filepath):
	"""``(reference_ev, provenance)`` off a ``sigma_mnk.h5``, or ``(None, None)``.

	THE ONE READER OF THE STAMP, so its location is stated once.  ``None``
	means the file predates the stamp (audit A2) — not that its ω axis is
	absolute.  A consumer that cannot tolerate a guess must REFUSE on
	``None`` rather than substitute its own convention; that substitution,
	made silently, is the defect the stamp exists to close.
	"""
	from file_io.sigma_output import (
		STACK_H5PY,
		open_scope,
	)
	abs_path = os.path.abspath(filepath)
	with open_scope(
			abs_path, STACK_H5PY, "r", where="read_omega_reference"), \
			h5py.File(abs_path, "r") as h5:
		_omega, ref, prov = _omega_metadata_from_open_h5(h5)
	return ref, prov



def _find_restart_file(input_file: str) -> str:
    """Locate the unique canonical ISDF restart beside the input deck.

    Centroid counts name different ISDF bases. Neither filename ordering nor
    modification time identifies the basis used for the run's Σ/eqp inputs;
    choosing one can silently change the exciton spectrum.
    """
    input_dir = os.path.dirname(os.path.abspath(input_file))
    candidates = []
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, "tmp", "isdf_tensors_*.h5"))))
    candidates.extend(sorted(glob.glob(os.path.join(input_dir, "isdf_tensors_*.h5"))))
    candidates = [p for p in candidates if os.path.exists(p)]
    if not candidates:
        raise FileNotFoundError(f"Could not find canonical restart file isdf_tensors_*.h5 in {input_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"GATE bse_restart_ambiguous: input_file={input_file!r}; "
            f"got: {len(candidates)} canonical restart bundles: "
            f"{sorted(candidates)}. "
            f"want: exactly one canonical bundle in the run directory "
            f"(including tmp/). why: these bundles can hold different ISDF "
            f"bases; choosing by mtime can silently change the exciton spectrum. "
            f"fix: leave exactly one canonical bundle in {input_dir}, matching "
            f"this run's Σ/eqp inputs; move the other isdf_tensors_*.h5 "
            f"bundles out of the run directory and its tmp/ subdirectory.")
    return candidates[0]



class _ZetaGTiles:
    """The lazy ``(nq, n_mu, ngkmax)`` ζ(G) stack, read through ONE owner.

    ``zeta_q_G`` is 47.8 GB at the converged MoS2 reference and every
    consumer slices it on q, so it is never materialised: this object is
    a HANDLE, not an array.  It replaces the raw ``h5py`` dataset the
    module used to stash in ``zx["ZG"]`` and keeps that dataset's
    indexing surface (``.shape``, ``[q]``, ``[q0:q1]`` → host numpy) so
    no consumer had to change, plus ONE new call that the h5py dataset
    could not express:

    * :meth:`read_q_slab` — the DISTRIBUTED plan.  ``ZetaLoader.load``
      → ``SlabIO.read_slab`` hyperslabs **only this rank's shard** of the
      q-chunk straight into the target sharding.  The call it replaces
      (``device_put_process_local(ds[sl], qb3)``) read the WHOLE chunk
      into host numpy on EVERY rank and then threw away all but its own
      shard: at nq=144 / n_μ=2412 / ngkmax=8603 a 48-q chunk is 15.9 GB,
      per rank.
    * ``__getitem__`` — the LOCAL plan, unchanged in what it returns:
      ``ZetaLoader.read_zeta_G_local(key)`` is the same h5py hyperslab,
      the same host numpy, the same bytes.  It is ALSO the only plan when
      the SlabIO probe declines.

    The layout CONTRACT is read once, from the loader
    (``gvec_components``/``ngk``/``ngkmax``, the sentinel Miller pad,
    ``zeta_q_G[q, :, ngk[q]:] == 0``); this module no longer re-derives
    any of it from raw datasets.  BOTH HANDLES BELONG TO THE LOADER now —
    the collective SlabIO one and the local plan's serial h5py one — and
    :meth:`close` ends them by closing it, instead of this object holding
    a second ``h5py.File`` of its own and the convention that nobody
    drops ``zx`` keeping the rest alive.

    WHY ``__getitem__`` IS NOT A SlabIO READ, AND WHERE THAT NOW LIVES.
    A SlabIO read is COLLECTIVE over the mesh and returns a ``jax.Array``
    whose requested shape must be mesh-divisible under its
    ``partition_spec``: a single-q ``(1, n_mu, ngkmax)`` read cannot be
    q-sharded at all, a replicated one materialises the same bytes on
    every rank plus a device round-trip, and putting a collective behind
    ``ds[q]`` would turn any future rank-0-only diagnostic into a hang
    instead of an error.  That argument is no longer this module's to
    make: it is the documented contract of
    :meth:`zeta_loader.ZetaLoader.read_zeta_G_local` — *local by design,
    per-rank independent, do not make this collective* — because the plan
    it describes moved INTO the door, where the one owner of the file can
    hold it to that promise.  What stays here is the consequence: every
    ``__getitem__`` caller in this module is a replicated host diagnostic
    (``recon``, ``run_gates``, ``run_nulls``); those are the mirrors
    ledger row 64 is about, and replacing them with on-device reductions
    is a diagnostics rewrite, not a transport change.
    """

    def __init__(self, loader: ZetaLoader, *, path: str, distributed: bool):
        self._loader = loader
        self._distributed = bool(distributed)
        if loader.zeta_layout != 'G_flat':
            raise ValueError(
                f"vq_interp needs a G-flat ζ ('zeta_q_G'); {path} has "
                f"zeta_layout={loader.zeta_layout!r}.  Refit with the "
                f"G-flat writer (gw.isdf_fitting).  There is no longer a "
                f"read path for r-space ζ at all: ZetaLoader's disk→G FFT "
                f"+ sphere gather was deleted on 2026-08-07 because no "
                f"writer in the tree emits that layout.")
        self.shape = (int(loader.n_q_on_disk), int(loader.n_rmu_disk),
                      int(loader.n_G_sph_disk))
        self.dtype = np.complex128
        # The header-vs-dataset ngkmax agreement check is GONE from here, not
        # dropped: ``ZetaLoader.__init__`` enforces it at OPEN, which is the
        # only place that can, since BOTH plans that could disagree about
        # ngkmax (header-sized collective, dataset-sized local) are its.

    # -- local plan (host numpy, h5py hyperslab; unchanged semantics) ---
    def __getitem__(self, key):
        """``zeta_q_G[key]`` as host numpy — the loader's serial handle.

        Delegation, not a re-implementation: ``read_zeta_G_local`` returns
        exactly what ``dataset[key]`` returns for any h5py key, and the
        service pins that byte-for-byte against a raw handle.  The
        post-close refusal is the LOADER's now (this used to be a local
        ``self._ds is None`` test with its own message), which is what
        makes "closed" one fact about one owner instead of two objects
        each with a private opinion.
        """
        return self._loader.read_zeta_G_local(key)

    # -- distributed plan (per-rank hyperslab straight into `sharding`) -
    def read_q_slab(self, q_offset: int, q_count: int, *, sharding):
        """``(q_count, n_mu, ngkmax)`` on ``sharding``.

        Distributed: ``ZetaLoader.load`` → ``SlabIO.read_slab`` with a
        per-rank hyperslab.  Local: the h5py chunk placed with
        ``device_put_process_local`` — bit-identical, since both plans
        return the same on-disk elements and neither reduces.
        """
        q_offset, q_count = int(q_offset), int(q_count)
        if self._distributed:
            return self._loader.load(
                q=np.arange(q_offset, q_offset + q_count, dtype=np.int32),
                sharding=sharding.spec)
        return device_put_process_local(
            self[q_offset:q_offset + q_count], sharding)

    # -- ownership -----------------------------------------------------
    def close(self) -> None:
        """Release the ζ handles.  Idempotent; post-close reads REFUSE.

        Both handles are the loader's, so closing it is what ends them —
        and the refusal a later ``ZG[q]`` gets is
        ``ZetaLoader.read_zeta_G_local``'s ("…is closed; its local ζ reads
        are no longer serviceable"), not this class's old private message.
        ``close_zeta_coarse`` still calls ``loader.close()`` right after
        this; ``ZetaLoader.close`` is idempotent, so the second call is a
        no-op and the ownership statement stays true from either end.
        """
        self._loader.close()



def _zeta_mesh_for_loader(mesh, log_fn=print):
    """``(mesh_or_None, distributed)`` — the ζ transport decision, once.

    ``ZetaLoader``'s data path is SlabIO, which REFUSES at open on a
    stack whose phdf5 FFI is absent (``file_io.slab_io``'s module
    docstring: there is one transport and nothing to demote to).  So the
    decision of whether to hand the loader a mesh has to be taken BEFORE
    constructing it, and it is taken here, once, and announced — because
    "which transport ran" is the single most consequential fact about a
    large run's I/O and a silent fallback is indistinguishable from a
    hang (same reasoning, and the same probe, as
    ``bse_io._bse_slabio_usable``).

    Header-only (``None``) does NOT mean "no reader": the loader still
    owns every metadata read on ``zeta_q.h5``.  It means the ζ TILES come
    back through the local h5py plan, which INVARIANTS row 6 licenses as
    the default — the defect that row records is a family with ONLY a
    local plan, and after this change ``vq_interp`` has both.
    """
    if mesh is None:
        return None, False
    from file_io.slab_io import probe_availability
    ok, stage, reason = probe_availability()
    if not ok:
        log_fn(f"  [vq_interp] SlabIO unavailable at probe stage '{stage}' "
               f"({reason}); reading ζ with the local h5py q-hyperslab plan "
               f"— memory-correct (one q-chunk per rank, no allgather) and "
               f"un-sharded, so every rank reads the whole chunk.")
        return None, False
    return mesh, True



def open_zeta(path, **kwargs):
    """Open the canonical ζ service for metadata and sharded q slabs."""
    from zeta_loader import ZetaLoader
    return ZetaLoader(path, **kwargs)


def read_dipole_metadata(path):
    """Return small dipole provenance and coverage attributes."""
    with h5py.File(path, "r") as f:
        return dict(f.attrs)


def read_dipole_parent_window(path, parent_rows, band_start, band_stop, *, nk_full):
    """Return parent-indexed Cartesian velocity blocks (parent, cart, band, band)."""
    with h5py.File(path, "r") as f:
        velocity = f["dipole_cart"]
        if velocity.shape[1] != int(nk_full):
            raise ValueError(
                "dipole_cart must retain its full-BZ file indexing; "
                f"got {velocity.shape[1]} rows, want {nk_full}")
        return np.stack([velocity[:, int(row), band_start:band_stop,
                                  band_start:band_stop] for row in parent_rows])


def load_kin_ion_submatrix(
	h5_path: str,
	band_start: int,
	band_stop: int,
	*,
	mesh: Mesh | None = None,
) -> jax.Array:
	"""Read the [band_start, band_stop) sub-window of ``kin_ion`` replicated.

	Stored dataset is the pristine ``T + V_loc + V_NL`` operator.  Folded
	Hartree files are rejected by :func:`validate_kin_ion_against_run`.

	The full kin_ion sub-block ``(nk, nb, nb)`` fits comfortably on a single
	device — it is loaded **fully replicated** on ``mesh`` so the
	post-self-energy plumbing can operate on replicated arrays uniformly.
	Goes through :class:`SlabIO` for backend parity with the rest of the
	GW input stack (``zeta_q.h5``, ``sigma_omega.h5``).

	THE STAR BROADCAST HAPPENS HERE.  What is on disk is the ``nrk``-row
	block the generator's sweep produced; the ``(nk, nb, nb)`` this returns
	is its unfold, so every caller sees the k-set it always saw and the
	saving is disk and write time rather than a new contract.  The read is
	the stored extent, so the transport moves ``nrk/nk`` of the bytes it
	used to; the broadcast is one gather on the device the slab landed on.

	Parameters
	----------
	h5_path
		Path to ``kin_ion.h5``.
	band_start, band_stop
		0-based half-open band window; ``band_stop > band_start`` and
		``band_stop ≤ nb_total``.
	mesh
		Device mesh.  Required; every slab read is collective over it.
		allgather backend tolerates ``None`` and returns a host-backed
		replicated JAX array.
	backend
		the allgather backend.

	Returns
	-------
	jax.Array, shape ``(nk, nb, nb)``, dtype ``complex128``, replicated.
	"""
	return _load_matrix_submatrix(
		h5_path, "kin_ion", band_start, band_stop, mesh=mesh)



def validate_kin_ion_against_run(
	h5_path: str,
	*,
	expected_bispinor: bool,
	expected_bispinor_gw_mode: str | None = None,
	sys_dim: int | None = None,
	nk: int | None = None,
	band_stop: int | None = None,
	nspinor: int | None = None,
	print_fn=print,
) -> dict:
	"""Validate pristine kinetic+ionic provenance before its slab is read."""
	from file_io.kin_ion import (resolve_four_current_representation)
	attrs = read_kin_ion_provenance(h5_path)
	if bool(attrs.get("has_hartree", False)):
		raise ValueError(
			"kin_ion.h5 has has_hartree=True: this retired format folds V_H "
			"into kin_ion and would double count the mandatory live G-space "
			"Hartree field. Regenerate a pristine kin_ion.h5.")

	stored_bispinor = attrs.get("bispinor")
	if stored_bispinor is None:
		raise ValueError(
			"kin_ion.h5 has no bispinor provenance; regenerate it from the "
			"run's input deck.")
	if bool(stored_bispinor) != bool(expected_bispinor):
		raise ValueError(
			f"kin_ion.h5 has bispinor={bool(stored_bispinor)} but this run "
			f"uses bispinor={bool(expected_bispinor)}; regenerate it.")

	representation = resolve_four_current_representation(
		expected_bispinor, expected_bispinor_gw_mode)
	if expected_bispinor:
		for attr, expected in (
			("charge_representation", representation.charge_representation),
			("spatial_current_representation",
			 representation.spatial_current_representation),
		):
			stored = attrs.get(attr)
			if stored is not None and str(stored) != str(expected):
				raise ValueError(
					f"kin_ion.h5 {attr}={stored!r}, expected {expected!r}.")
		# No per-mode ``bispinor_gw_mode`` check: both shipped values ride
		# the one raw kinetic-balance carrier, so the two representation
		# attrs above ARE the carrier identity.  A file written by one of
		# the two retired carrier-comparison modes (2026-09-01) carries a
		# different ``charge_representation`` and is refused there.

	stored_sys_dim = attrs.get("sys_dim")
	if (sys_dim is not None and stored_sys_dim is not None
			and int(stored_sys_dim) != int(sys_dim)):
		raise ValueError(
			f"kin_ion.h5 has sys_dim={int(stored_sys_dim)} but this run "
			f"uses sys_dim={int(sys_dim)}.")
	stored_nspinor = attrs.get("nspinor")
	if (nspinor is not None and stored_nspinor is not None
			and int(stored_nspinor) != int(nspinor)):
		raise ValueError(
			f"kin_ion.h5 has nspinor={int(stored_nspinor)} but this run "
			f"uses nspinor={int(nspinor)}; regenerate it from this WFN.")
	if nk is not None and int(attrs["_nk_logical"]) != int(nk):
		raise ValueError(
			f"kin_ion.h5 has nk={int(attrs['_nk_logical'])} but the run has "
			f"nk={int(nk)}.")
	if band_stop is not None and int(attrs["_shape"][1]) < int(band_stop):
		raise ValueError(
			f"kin_ion.h5 has {int(attrs['_shape'][1])} bands but the run "
			f"needs {int(band_stop)}.")
	print_fn("  kin_ion: pristine T+V_loc+V_NL; Hartree is built live in G-space.")
	return attrs



def _load_matrix_submatrix(
	h5_path: str,
	dataset: str,
	band_start: int,
	band_stop: int,
	*,
	mesh: Mesh | None,
) -> jax.Array:
	"""Format-owned collective slab read plus authenticated star unfold."""
	from file_io.kin_ion import (SlabIO, _unfold_if_ibz)
	if band_stop <= band_start:
		raise ValueError(f"Invalid band slice [{band_start}, {band_stop})")
	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"kin_ion file not found: {h5_path}")

	# Authenticate the dataset's own k-storage stamp and star tables before
	# issuing any collective payload read.
	star = read_star_map(h5_path, dataset)
	with h5py.File(h5_path, "r") as h5:
		if dataset not in h5:
			raise KeyError(f"Dataset {dataset!r} missing from {h5_path}")
		nk_stored, nb_total, nb_total2 = h5[dataset].shape
	if nb_total != nb_total2:
		raise ValueError(
			f"{dataset} must be square in band axes; "
			f"got {(nk_stored, nb_total, nb_total2)}")
	if band_stop > nb_total:
		raise ValueError(
			f"Requested bands require {band_stop} states but {dataset} only "
			f"has {nb_total}. Regenerate kin_ion.h5 with at least -n "
			f"{band_stop}.")
	nb = band_stop - band_start
	with SlabIO(h5_path, mode="r", mesh=mesh) as io:
		arr = io.read_slab(
			dataset,
			shape=(nk_stored, nb, nb),
			offset=(0, band_start, band_start),
			dtype=jnp.complex128,
			mesh=mesh,
			partition_spec=P(None, None, None),
		)
	return _unfold_if_ibz(arr, star)



def read_kin_ion_provenance(h5_path: str) -> dict:
	"""Return the ``kin_ion`` dataset attributes as a plain dict.

	Missing file or dataset raises; a legacy file simply has fewer keys.
	Values are converted to plain Python/NumPy scalars so callers can
	compare and print them without h5py types leaking out.

	``_shape`` is the STORED shape, so its k extent is ``nrk`` on an
	IBZ-stored file.  ``_nk_logical`` is the k count a consumer will
	actually receive — the two are equal on a full-BZ file and differ by
	the star reduction otherwise, and every check about "does this file
	match my run" wants the logical one.
	"""
	from file_io.kin_ion import (K_STORAGE_FULL, K_STORAGE_IBZ)
	if not os.path.exists(h5_path):
		raise FileNotFoundError(f"kin_ion file not found: {h5_path}")
	star = read_star_map(h5_path, "kin_ion")
	with h5py.File(h5_path, "r") as h5:
		if "kin_ion" not in h5:
			raise KeyError("Dataset 'kin_ion' missing from kin_ion file")
		ds = h5["kin_ion"]
		out = {k: v for k, v in ds.attrs.items()}
		out["_shape"] = tuple(int(s) for s in ds.shape)
		out["_k_storage"] = (K_STORAGE_FULL if star is None
		                     else K_STORAGE_IBZ)
		out["_nk_logical"] = (int(ds.shape[0]) if star is None
		                      else int(star[0].size))
	return out



def read_full_bz_dataset(h5_path: str, dataset: str = "kin_ion"):
	"""One dataset of ``kin_ion.h5``, on the FULL BZ, as a host array.

	The serial-h5py twin of :func:`load_kin_ion_submatrix`, for the
	host-side consumers that read this file with no device mesh to be
	collective over — ``gw.eqp_bgw`` rebuilds eqp{0,1} straight from files
	and has none.  Whole dataset, no band window: those callers slice it
	themselves.

	It exists so "read kin_ion.h5" has ONE meaning in the tree.  The
	alternative is each host-side reader noticing :data:`K_STORAGE_ATTR`
	for itself, and a reader that did not notice would index an
	``(nrk, nb, nb)`` array with full-BZ k — returning another star's
	matrix on every k past the wedge, or an ``IndexError`` on a deck lucky
	enough to be caught.
	"""
	from file_io.kin_ion import (_unfold_if_ibz)
	star = read_star_map(h5_path, dataset)
	with h5py.File(h5_path, "r") as h5:
		if dataset not in h5:
			raise KeyError(f"Dataset {dataset!r} missing from {h5_path}")
		arr = np.asarray(h5[dataset][()])
	return np.asarray(_unfold_if_ibz(arr, star))



def read_star_map(h5_path: str, dataset: str = "kin_ion", *, k_axis: int = 0):
	"""The unfold tables a stored-on-IBZ ``dataset`` needs, or ``None``.

	``None`` means the dataset is stored on the full BZ and must be read
	verbatim — which is what a missing :data:`K_STORAGE_ATTR` means, so
	every file written before this format keeps its meaning exactly.

	Otherwise returns ``(irr_idx_k, sym_idx_k, n_sym_spatial)``.  Every
	way the file can be internally inconsistent raises here rather than
	downstream: a claim with no tables, tables of different lengths, a
	version this reader was not written against, a storage value that is
	neither of the two legal ones, and — the one that matters most — a
	stored k axis that does not match the number of distinct stars the
	tables describe, which is what a truncated or mislabelled slab looks
	like from the outside.

	``k_axis`` names WHICH axis of the stored dataset is the k axis, and
	it exists because ``sigma_mnk.h5``'s dynamic cubes are
	``(n_omega, nk, nb, nb)`` — axis 0 there is frequency, and a check
	that read ``shape[0]`` would compare the star count against the
	frequency count and refuse every correctly-written cube.  It defaults
	to 0, which is every ``kin_ion.h5`` array, so no existing caller
	changes.  ONE implementation of the stamp contract for both files was
	the point: a second copy in ``sigma_output`` would be a second place
	for the version, the table names and the refusals to drift.
	"""
	from file_io.kin_ion import (
		IRR_IDX_DATASET,
		K_STORAGE_ATTR,
		K_STORAGE_FULL,
		K_STORAGE_IBZ,
		K_STORAGE_VALUES,
		K_STORAGE_VERSION,
		K_STORAGE_VERSION_ATTR,
		N_SYM_SPATIAL_ATTR,
		SYM_IDX_DATASET,
	)
	with h5py.File(h5_path, "r") as h5:
		if dataset not in h5:
			raise KeyError(f"Dataset {dataset!r} missing from {h5_path}")
		ds = h5[dataset]
		stored = str(ds.attrs.get(K_STORAGE_ATTR, K_STORAGE_FULL))
		if stored not in K_STORAGE_VALUES:
			raise ValueError(
				f"{os.path.basename(h5_path)}: {dataset}.{K_STORAGE_ATTR} is "
				f"{stored!r}, which is neither {K_STORAGE_IBZ!r} nor "
				f"{K_STORAGE_FULL!r}.  A reader that guessed here would pick "
				f"between reading nrk rows as nk and the reverse.")
		if stored == K_STORAGE_FULL:
			return None
		version = int(ds.attrs.get(K_STORAGE_VERSION_ATTR, -1))
		if version != K_STORAGE_VERSION:
			raise ValueError(
				f"{os.path.basename(h5_path)}: {dataset} is stored on the IBZ "
				f"at format version {version}, but this reader implements "
				f"version {K_STORAGE_VERSION}.  Regenerate kin_ion.h5.")
		if N_SYM_SPATIAL_ATTR not in ds.attrs:
			raise ValueError(
				f"{os.path.basename(h5_path)}: {dataset} claims IBZ storage "
				f"but carries no {N_SYM_SPATIAL_ATTR!r} attr, so the "
				f"conjugation predicate has no threshold to test against.")
		n_sym_spatial = int(ds.attrs[N_SYM_SPATIAL_ATTR])
		missing = [n for n in (IRR_IDX_DATASET, SYM_IDX_DATASET)
				   if n not in h5]
		if missing:
			raise ValueError(
				f"{os.path.basename(h5_path)}: {dataset} claims IBZ storage "
				f"but the file carries no {missing} — the tensor cannot be "
				f"unfolded at all.  A tensor whose reconstruction tables "
				f"live elsewhere is a tensor that silently decays.")
		irr = np.asarray(h5[IRR_IDX_DATASET][()], dtype=np.int32)
		sidx = np.asarray(h5[SYM_IDX_DATASET][()], dtype=np.int32)
		if not (-ds.ndim <= k_axis < ds.ndim):
			raise ValueError(
				f"{os.path.basename(h5_path)}: {dataset} has {ds.ndim} axes, "
				f"so k_axis={k_axis} names no axis of it.")
		nk_stored = int(ds.shape[k_axis])
	if irr.shape != sidx.shape or irr.ndim != 1:
		raise ValueError(
			f"{os.path.basename(h5_path)}: {IRR_IDX_DATASET} {irr.shape} and "
			f"{SYM_IDX_DATASET} {sidx.shape} must both be (nk_full,)")
	# THE NUMBER OF DISTINCT STARS, not ``max + 1``.  The two agree only
	# while the labels are dense, which is exactly what the writers are
	# supposed to guarantee (``file_io.sigma_output.compact_star_tables``,
	# and ``gw.kin_ion_io.star_tables`` through it) — so testing ``max+1``
	# tested the writers' arithmetic instead of their output, and passed
	# on the one shape this refusal exists to catch.  MEASURED 2026-08-17:
	# ``gnppm_debug``'s ``irr_idx_k = [0,2,2,6,8,7,6,7,8]`` gives
	# ``max+1 = 9`` against 9 stored rows and sails through, while the
	# true star count is 5 and four of those rows were in a basis nothing
	# reads.  ``np.unique`` costs a sort of ``nk_full`` int32 once per
	# open and is the property the docstring already claimed.
	n_star = int(np.unique(irr).size)
	if n_star != nk_stored:
		raise ValueError(
			f"{os.path.basename(h5_path)}: {dataset} stores {nk_stored} k "
			f"rows but {IRR_IDX_DATASET} describes {n_star} stars over "
			f"{irr.size} full-BZ k.  The slab and the tables filed with it "
			f"do not describe the same calculation — refusing rather than "
			f"unfolding {min(n_star, nk_stored)} of them.")
	if int(sidx.max(initial=-1)) >= 2 * n_sym_spatial:
		raise ValueError(
			f"{os.path.basename(h5_path)}: {SYM_IDX_DATASET} reaches "
			f"{int(sidx.max())} but the table is 2·{N_SYM_SPATIAL_ATTR} = "
			f"{2 * n_sym_spatial} rows long.")
	return irr, sidx, n_sym_spatial



def _eval_metadata_from_open_h5(h5):
	"""Return the small evaluation-energy stamp from an already-open file."""
	from file_io.sigma_output import (
		OMEGA_COVERAGE_FRAC_ATTR,
		OMEGA_COVERAGE_N_ATTR,
		OMEGA_COVERAGE_POLICY_ATTR,
		SIGMA_EVAL_DATASET,
		SIGMA_EVAL_PROVENANCE_ATTR,
	)
	if SIGMA_EVAL_DATASET not in h5:
		return None, None, None
	ds = h5[SIGMA_EVAL_DATASET]
	arr = np.asarray(ds[()], dtype=np.float64)
	prov = ds.attrs.get(SIGMA_EVAL_PROVENANCE_ATTR, "unstated")
	cov = None
	if OMEGA_COVERAGE_N_ATTR in ds.attrs:
		pol = ds.attrs.get(OMEGA_COVERAGE_POLICY_ATTR, "unstated")
		cov = {
			"n_uncovered": int(ds.attrs[OMEGA_COVERAGE_N_ATTR]),
			"fraction_uncovered": float(
				ds.attrs.get(OMEGA_COVERAGE_FRAC_ATTR, 0.0)),
			"policy": (pol.decode("utf-8")
			           if isinstance(pol, bytes) else str(pol)),
		}
	if isinstance(prov, bytes):
		prov = prov.decode("utf-8")
	return arr, str(prov), cov



def _omega_metadata_from_open_h5(h5):
	"""Return the small omega axis plus its optional reference stamps."""
	from file_io.sigma_output import (
		OMEGA_DATASET,
		OMEGA_REFERENCE_ATTR,
		OMEGA_REFERENCE_PROVENANCE_ATTR,
	)
	if OMEGA_DATASET not in h5:
		return None, None, None
	ds = h5[OMEGA_DATASET]
	omega = np.asarray(ds[()], dtype=np.float64)
	if OMEGA_REFERENCE_ATTR not in ds.attrs:
		return omega, None, None
	ref = float(ds.attrs[OMEGA_REFERENCE_ATTR])
	prov = ds.attrs.get(OMEGA_REFERENCE_PROVENANCE_ATTR, "unstated")
	if isinstance(prov, bytes):
		prov = prov.decode("utf-8")
	return omega, ref, str(prov)
