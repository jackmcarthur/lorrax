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
        # bytes move. Current unreduced tensors have no q-star table.
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

        ``mu_axis``/``spinor_axis`` default to the nmu axis order
        (nk, n, s, μ); the mun face (nk, s, μ, n) passes both explicitly
        — its μ is axis -2 and its spinor is axis 1, not axis -1/2.  NO
        RESHARD happens here regardless of ``spec``: this is a straight
        SlabIO hyperslab read, so a face spec costs exactly what the
        axis spec costs (one direct read), never a transpose collective.
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

    # G0 is the canonical one-dimensional head vector. Pad and shard
    # it on the same centroid extent as the restored tensors.
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

    All callers use the same slab planning, padding and q-star restoration.

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


def read_poles(
    src,
    *,
    pole_slice=None,
    mesh_xy=None,
    unfold=False,
    return_sharded=False,
    to_unit=None,
    allow_partial=False,
    include_odd=False,
    mode="r",
):
    """Read one contiguous pole range with two collective SlabIO reads.

    The leading pole axis is always retained.  ``pole_slice=None`` reads it
    completely; an integer reads a length-one range.  A mesh read is always
    sharded — pass ``return_sharded=True`` with ``mesh_xy``; there is no
    gather path.

    THE SUMMARY LINE DESCRIBES ONE OF TWO BRANCHES, and the contract
    changes with ``mesh_xy``:

    * ``mesh_xy`` given — COLLECTIVE, two ``SlabIO`` reads through
      :func:`open_pole_reader`; ``src`` must be a PATH; returns two sharded
      ``jax.Array``s of ``(n_poles, n_q, n_mu_padded, n_mu_padded)`` on
      ``P(None, None, 'x', 'y')``, where the pad is
      ``runtime.padding.padded_mu_extent``, not ``mesh_divisible_shape``.
    * ``mesh_xy=None`` — SERIAL, rank-local, no SlabIO and nothing
      collective; ``src`` may be a path or an open h5py group; returns two
      HOST numpy arrays at the LOGICAL ``(n_poles, n_q, n_mu, n_mu)``.

    Returns the 2-tuple ``(Omega_p, B_p)`` either way.  This is NOT the
    sole pole reader — :func:`read_fit_block`, :func:`read_fit_tensors` and
    :meth:`PoleReader.read` also read these datasets, and the production Σ
    reader is the last of those; this function has no ``src`` caller.

    ONE range per call, so a caller reading SEVERAL ranges of one file —
    every Σ stage does — opens and closes the store once per range.  Use
    :func:`open_pole_reader` there instead: it holds one collective handle
    across the whole walk and does its h5py reads before that handle
    exists (audit A1).  This function is the single-range door, and is
    that reader with a lifetime of one call.
    """
    from file_io.mpa_store import (
        _finish_pole_read,
        _h5,
        _pole_range,
        _refuse_unfinalized,
        fit_completion_ledger,
    )
    if mesh_xy is not None and not return_sharded:
        raise ValueError(
            "read_poles: got mesh_xy with return_sharded=False; want "
            "return_sharded=True on every mesh read — the collective "
            "read lands sharded and every mesh caller consumes that "
            "layout, so no gather path exists.  Fix: pass "
            "return_sharded=True, or drop mesh_xy for a host-side read.")
    if mesh_xy is not None:
        with open_pole_reader(src, mesh_xy=mesh_xy,
                              allow_partial=allow_partial, mode=mode) as rd:
            return rd.read(pole_slice, unfold=unfold,
                           return_sharded=return_sharded, to_unit=to_unit,
                           include_odd=include_odd)

    with _h5(src, mode) as grp:
        ledger = fit_completion_ledger(grp)
        _refuse_unfinalized(grp, ledger, allow_partial, "read_poles")
        lo, hi = _pole_range(ledger, pole_slice, "read_poles")
        Omega = np.asarray(grp["Omega_p"][lo:hi])
        Bp = np.asarray(grp["B_p"][lo:hi])
        B_odd = (np.asarray(grp["B_odd_p"][lo:hi])
                 if include_odd and ledger["ordered_residues"] else None)
    return _finish_pole_read(
        src, Omega, Bp, ledger, mesh_xy=None, unfold=unfold,
        return_sharded=return_sharded, to_unit=to_unit,
        B_odd=B_odd, include_odd=include_odd)



def open_pole_reader(src, *, mesh_xy, allow_partial=False, mode="r"):
    """A :class:`PoleReader` for one iteration's pole walk.

    COLLECTIVE: this opens a ``SlabIO`` handle, so every rank must call it,
    in the same order, and every rank must close it — use ``with``, or a
    ``close()`` in a ``finally``.  ``src`` must be a PATH.  While the
    reader lives, do not open this path with h5py (see :class:`PoleReader`
    for exactly how much of that the registry enforces).
    """
    return PoleReader(src, mesh_xy=mesh_xy, allow_partial=allow_partial,
                      mode=mode)



class PoleReader:
    """ONE collective handle serving every pole batch of one iteration.

    WHY THIS EXISTS (audit A1 fix 2).  The Σ stage walks the pole axis in
    batches so no complete pole tensor ever exists on host or device, and
    it walks it TWICE per iteration — once for the census that plans the
    windows, once for the spatial executor.  Called through
    :func:`read_poles`, each batch opened and closed its own h5py handle
    (ledger), its own collective ``SlabIO``, and a THIRD h5py handle for
    the unfold tables — and the third one landed *between* the two, so the
    per-batch sequence alternated h5py → FFI → h5py on one file, through
    two independent HDF5 library instances, once per batch.

    This reader collapses that to: read the ledger and the unfold tables
    with h5py FIRST (two opens, or one when the store is not wedge-packed),
    CLOSE h5py, then hold ONE ``SlabIO`` open for every batch of the
    iteration.  Two properties, and the second is the one that matters more
    than the arithmetic:

    * the churn drops from ``3·n_batches`` opens per walk to ``2 + 1``;
    * **no h5py open happens while the collective handle is live**, so
      the alternation the two libraries cannot survive does not occur at
      all inside a Σ stage.

    HOW MUCH OF THAT IS MACHINE-ENFORCED, precisely — because the sentence
    here used to claim all of it.  This reader opens ``SlabIO`` READ-ONLY,
    and :mod:`file_io.hdf5_owner` refuses a cross-stack overlap only when
    one side can WRITE.  So a stray h5py **write** open on this path while
    the reader is alive refuses by name; a stray h5py **read** open is
    ALLOWED BY DESIGN and merely counted.  The no-h5py-while-live property
    above is upheld by this class's own ordering, and by the registry only
    for writers.

    The handle is released in a ``finally`` — use it as a context manager,
    or call :meth:`close` from one.  A refusal raised mid-walk (an
    uncertified fit, a bad pole range) must still release the collective
    handle on every rank, because the next collective call on this mesh
    would otherwise rendezvous against a file this rank never closed.
    """

    def __init__(self, src, *, mesh_xy, allow_partial=False, mode="r"):
        from file_io.mpa_store import (
            _h5,
            _refuse_unfinalized,
            fit_completion_ledger,
        )
        if mesh_xy is None:
            raise ValueError(
                "PoleReader requires mesh_xy: it exists to hold ONE "
                "collective handle open across an iteration's pole "
                "batches, and a mesh-less read has no such handle.  Use "
                "read_poles(src, pole_slice=...) for the host path.")
        self.src = src
        self.mesh_xy = mesh_xy
        # h5py FIRST, and completely, and closed — before any collective
        # handle exists.  Both reads are small and neither is repeated.
        with _h5(src, mode) as grp:
            self.ledger = fit_completion_ledger(grp)
            _refuse_unfinalized(grp, self.ledger, allow_partial, "PoleReader")
        self.tables = (read_fit_unfold_tables(src)
                       if self.ledger["q_storage"] == "ibz" else None)
        self.n_poles = int(self.ledger["n_p"])
        from file_io.slab_io import SlabIO
        self._io = SlabIO(src, mode="r", mesh=mesh_xy)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        io, self._io = getattr(self, "_io", None), None
        if io is not None:
            io.close()

    def read(self, pole_slice=None, *, unfold=False, return_sharded=False,
             to_unit=None, include_odd=False):
        """One contiguous pole range, through the handle already open."""
        from file_io.mpa_store import (
            _finish_pole_read,
            _pole_range,
            _pole_read_shape,
        )
        if self._io is None:
            raise ValueError(
                "PoleReader.read after close(): the collective handle this "
                "reader owns is gone.  Open a new reader rather than "
                "reopening this one — a reader whose handle can be revived "
                "is a reader whose lifetime nobody can read off the code.")
        from jax.sharding import PartitionSpec as P

        lo, hi = _pole_range(self.ledger, pole_slice, "PoleReader.read")
        shape = (hi - lo, *_pole_read_shape(self.ledger, self.mesh_xy))
        Omega = self._io.read_slab(
            "Omega_p", shape=shape, offset=(lo, 0, 0, 0),
            partition_spec=P(None, None, "x", "y"))
        Bp = self._io.read_slab(
            "B_p", shape=shape, offset=(lo, 0, 0, 0),
            partition_spec=P(None, None, "x", "y"))
        B_odd = None
        if include_odd and self.ledger["ordered_residues"]:
            B_odd = self._io.read_slab(
                "B_odd_p", shape=shape, offset=(lo, 0, 0, 0),
                partition_spec=P(None, None, "x", "y"))
        return _finish_pole_read(
            self.src, Omega, Bp, self.ledger, mesh_xy=self.mesh_xy,
            unfold=unfold, return_sharded=return_sharded, to_unit=to_unit,
            tables=self.tables, B_odd=B_odd, include_odd=include_odd)



def read_fit_unfold_tables(src, *, mode="r"):
    """Return the fit store's q-unfold tables, or ``None`` for full BZ."""
    from file_io.mpa_store import (
        FIT_TABLE_OWNER,
        _h5,
        _qs,
    )
    qs = _qs()
    with _h5(src, mode) as grp:
        if FIT_TABLE_OWNER + qs.QIRR_TABLE_SUFFIX not in grp:
            return None
        return qs.read_tables(grp, FIT_TABLE_OWNER, mode=mode)



def read_fit_tensors(src, *, allow_partial=False, mode="r"):
    """The whole ``(Omega_p, B_p, diagnostics, ledger)``.

    For tests and offline inspection.  The Σ stage does NOT read this — it
    streams contiguous pole ranges through :class:`PoleReader` (opened by
    :func:`open_pole_reader`) and never holds the whole tensor.  This used
    to name :func:`read_poles`, which the Σ stage stopped using when
    ``PoleReader`` landed.

    ``Omega_p`` and ``B_p`` come back ``(n_p, n_q, n_mu, n_mu)``
    complex128 and ``diagnostics`` as ``{key: (n_q, n_mu, n_mu) float64}``
    — the whole tensor, on this rank, which is the cost the first
    paragraph is warning about.  Serial; ``src`` is a path or an open
    group.  Same finalize refusal as :func:`read_fit_block`.
    """
    from file_io.mpa_store import (
        _h5,
        _qs,
        _refuse_unfinalized,
        fit_completion_ledger,
    )
    qs = _qs()
    with _h5(src, mode) as grp:
        ledger = fit_completion_ledger(grp)
        _refuse_unfinalized(grp, ledger, allow_partial,
                            "read_fit_tensors")
        Om = np.asarray(grp["Omega_p"][()])
        Bp = np.asarray(grp["B_p"][()])
        diag = {str(k)[len("fit_"):]: np.asarray(grp[k][()])
                for k in grp if str(k).startswith("fit_")}
        return Om, Bp, diag, ledger



def read_fit_block(src, q, mu_cols, *, allow_partial=False, mode="r"):
    """One column block's ``(Omega_p, B_p, diagnostics, ledger)``.

    Refuses an unfinalized store unless ``allow_partial=True``, and
    when partial, refuses the specific columns that are not fitted —
    "the file is incomplete" and "the columns you asked for are
    incomplete" are different facts and a driver resuming a crashed fit
    needs the second one.
    """
    from file_io.mpa_store import (
        _column_span,
        _h5,
        _qs,
        _ranges,
        _refuse_unfinalized,
        fit_completion_ledger,
        normalise_columns,
    )
    qs = _qs()
    with _h5(src, mode) as grp:
        ledger = fit_completion_ledger(grp)
        _refuse_unfinalized(grp, ledger, allow_partial,
                            f"read_fit_block(q={q})")
        iq = int(q)
        if not 0 <= iq < ledger["n_q"]:
            raise IndexError(
                f"read_fit_block: q={iq} is outside [0, "
                f"{ledger['n_q']})")
        cols = normalise_columns(mu_cols, ledger["n_mu"])
        undone = cols[~ledger["blocks_done"][iq, cols]]
        if undone.size:
            raise ValueError(
                f"read_fit_block: q={iq} columns "
                f"{_ranges(undone)} are not fitted.  They read back as "
                f"zeros, which is a converged-looking dark channel and "
                f"not an absent one, so the refusal is on the LEDGER "
                f"and never on the data.")
        lo, hi, sel = _column_span(cols)
        sel = slice(lo, hi) if sel is None else sel
        Om = np.asarray(grp["Omega_p"][:, iq, :, sel])
        Bp = np.asarray(grp["B_p"][:, iq, :, sel])
        stamp = ledger["diagnostic_keys"] or ""
        keys = tuple(key for key in stamp.split(",") if key)
        diag = {}
        for key in keys:
            name = "fit_" + key
            if name not in grp:
                raise ValueError(
                    f"read_fit_block: diagnostic_keys names {name!r}, "
                    "but the dataset is absent")
            diag[key] = np.asarray(grp[name][iq, :, sel])
        return Om, Bp, diag, ledger



def read_fit_io_receipt(src, *, mode="r"):
    """Return the ready body-attempt I/O receipt, or ``None`` if absent."""
    from file_io.mpa_store import (
        _h5,
        _read_fit_io_receipt_group,
    )
    with _h5(src, mode) as grp:
        return _read_fit_io_receipt_group(grp)



def validate_fit_store(src, *, expected_identity=None,
                       expected_screening_diagrams=None, mode="r"):
    """Validate the finalized fit contract before Sigma reads pole bytes.

    ``expected_identity`` may name ``w_grid_hash``, ``w_table_hash`` and
    ``w_centroid_hash`` from the screening object currently in use, plus the
    canonical ``wfn_fingerprint_scheme`` / ``wfn_fingerprint`` copied from
    the W sample.  The fit's own declared ``*_max_allowed`` certification
    thresholds are always enforced against its observed maxima.

    ``expected_screening_diagrams`` is the run's ``screening_diagrams``
    value.  RPA poles and ladder-corrected poles are the same shape, pass
    the same certification and read back equally plausibly, so the only
    thing separating them is the stamp the writer left — which makes this
    the load-time half of QUALITY_PATTERNS #10.  A store with NO stamp is
    refused rather than assumed RPA: "written before the axis existed" and
    "written by the RPA path" are different facts, and silently reading the
    first as the second is how a ladder fit gets consumed as an RPA one.

    Pole bytes are not materialized here.  The finalized datasets themselves
    are nevertheless opened for metadata and must exactly match the ledger's
    logical ``(n_p,n_q,n_mu,n_mu)`` shape and ``complex128`` dtype before a
    collective reader can be opened.  Finiteness is a streamed property: the
    Sigma census reduces each resident pole slab before planning or execution.

    LEGACY POLICY: a caller that supplies no WFN fields may still validate an
    old fit for offline inspection or same-run compatibility.  Explicit reuse
    supplies both canonical fields and therefore refuses every legacy fit that
    predates them; absence is not interpreted as a match.

    Returns the :func:`fit_completion_ledger` dict, which callers use for
    ``n_p``.  Rank-local and serial; ``src`` is a path or an open group.
    """
    from file_io.mpa_store import (
        CERTIFICATION_METRICS,
        FIT_ENERGY_UNITS,
        MPA_IDENTITY_PROVENANCE_KEYS,
        _validate_pole_payload_metadata,
        fit_completion_ledger,
    )
    ledger = fit_completion_ledger(src, mode=mode)
    if expected_screening_diagrams is not None:
        want = str(getattr(expected_screening_diagrams, "value",
                           expected_screening_diagrams))
        got = ledger["provenance"].get("screening_diagrams")
        if got is None:
            raise ValueError(
                f"MPA fit store carries no screening_diagrams stamp, so it "
                f"cannot say whether its poles came from the RPA W or the "
                f"ladder-corrected W; this run is {want!r}.  Regenerate the "
                f"fit with a writer that stamps it (gw.mpa.model."
                f"build_mpa_fit does).")
        if str(got) != want:
            raise ValueError(
                f"MPA fit provenance mismatch: the store was built with "
                f"screening_diagrams = {str(got)!r} and this run is "
                f"{want!r}.  The two produce different W and therefore "
                f"different poles; reusing one for the other is the "
                f"changed-band-window class of silent reuse.")
    if not ledger["complete"]:
        raise ValueError("MPA Sigma requires a finalized pole fit store")
    if ledger["energy_unit"] not in FIT_ENERGY_UNITS:
        raise ValueError("MPA fit store does not declare a supported unit")
    _validate_pole_payload_metadata(src, ledger, mode=mode)
    for key, want in (expected_identity or {}).items():
        if key in MPA_IDENTITY_PROVENANCE_KEYS:
            got = ledger["provenance"].get(key)
        elif key in ("w_grid_hash", "w_table_hash", "w_centroid_hash"):
            got = ledger[key]
        else:
            raise KeyError(f"unknown MPA fit identity field {key!r}")
        if got is None or str(got) != str(want):
            raise ValueError(
                f"MPA fit identity mismatch for {key}: got {got!r}, "
                f"expected {want!r}")
    missing = [key + "_max_allowed" for key in CERTIFICATION_METRICS
               if key + "_max_allowed" not in ledger["certification"]]
    if missing:
        raise ValueError(
            "MPA Sigma requires certified pole fits; the store is missing "
            + ", ".join(missing))
    for metric_key in CERTIFICATION_METRICS:
        key = metric_key + "_max_allowed"
        allowed = ledger["certification"][key]
        if not np.isfinite(float(allowed)) or float(allowed) <= 0.0:
            raise ValueError(
                f"MPA fit has invalid stored certification {key}="
                f"{allowed!r}")
        metric = metric_key + "_max"
        got = ledger[metric]
        if got is None or float(got) > float(allowed):
            raise ValueError(
                f"MPA fit failed its stored certification: {metric}="
                f"{got!r} exceeds {allowed!r}")
    return ledger



def validate_fit_store_for_resume(
        src, *, n_q, n_mu, n_p, energy_unit, grid_hash, table_hash,
        centroid_hash, provenance, ordered_residues, schedule,
        occupation_state=None, mode="r"):
    """Authenticate a partial fit and prove its ledger against the schedule.

    Only whole scheduled ranges can be checkpointed.  The journal, both
    diagnostics vectors and ``blocks_done`` are redundant by design here:
    all three must describe exactly the same non-overlapping union before a
    restart may skip any work.
    """
    from file_io.mpa_store import (
        CERTIFICATION_METRICS,
        MPA_FIT_FORMAT_VERSION,
        PERSISTED_DIAGNOSTICS,
        _occ_stamp_values,
        _validate_pole_payload_metadata,
        _values_equal,
        fit_completion_ledger,
    )
    ledger = fit_completion_ledger(src, mode=mode)
    expected = {
        "format_version": MPA_FIT_FORMAT_VERSION,
        "n_q": int(n_q), "n_mu": int(n_mu), "n_p": int(n_p),
        "energy_unit": str(energy_unit),
        "w_grid_hash": str(grid_hash), "w_table_hash": str(table_hash),
        "w_centroid_hash": str(centroid_hash),
        "ordered_residues": bool(ordered_residues),
        "diagnostic_keys": ",".join(PERSISTED_DIAGNOSTICS),
    }
    faults = [
        f"{key}: stored={ledger.get(key)!r}, expected={want!r}"
        for key, want in expected.items()
        if not _values_equal(ledger.get(key), want)
    ]
    stored_provenance = dict(ledger.get("provenance") or {})
    expected_provenance = dict(provenance or {})
    if (set(stored_provenance) != set(expected_provenance)
            or any(not _values_equal(stored_provenance[key], value)
                   for key, value in expected_provenance.items()
                   if key in stored_provenance)):
        faults.append(
            "provenance differs: stored keys="
            f"{sorted(stored_provenance)}, expected keys="
            f"{sorted(expected_provenance)}")
    stored_occ = read_occupation_stamps(src, mode=mode)
    expected_occ = (
        None if occupation_state is None else _occ_stamp_values(occupation_state))
    if ((stored_occ is None) != (expected_occ is None)
            or (stored_occ is not None and any(
                not _values_equal(stored_occ[key], expected_occ[key])
                for key in expected_occ))):
        faults.append("occupation provenance differs")
    if faults:
        raise ValueError(
            "MPA partial fit store is incompatible:\n  " + "\n  ".join(faults))

    _validate_pole_payload_metadata(src, ledger, mode=mode)
    done = np.asarray(ledger["blocks_done"], dtype=bool)
    if done.shape != (int(n_q), int(n_mu)):
        raise ValueError(
            f"MPA partial fit blocks_done has shape {done.shape}, expected "
            f"{(int(n_q), int(n_mu))}")
    journal = np.asarray(ledger["journal"], dtype=np.int64)
    if journal.ndim != 2 or journal.shape[1:] != (3,):
        raise ValueError(
            f"MPA partial fit journal must have shape (n,3); got {journal.shape}")
    n_records = int(journal.shape[0])
    for key in CERTIFICATION_METRICS:
        values = np.asarray(ledger["block_" + key + "_max"], np.float64)
        if values.shape != (n_records,):
            raise ValueError(
                f"MPA partial fit {key} diagnostics length {values.shape} "
                f"does not equal journal length {n_records}")
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"MPA partial fit {key} diagnostics contain non-finite values")

    schedule = tuple(schedule)
    allowed = {(int(q), int(lo), int(hi)) for q, lo, hi in schedule}
    if len(allowed) != len(schedule):
        raise ValueError("MPA fit schedule itself contains duplicate ranges")
    schedule_union = np.zeros_like(done)
    for q, lo, hi in allowed:
        if not (0 <= q < int(n_q) and 0 <= lo < hi <= int(n_mu)):
            raise ValueError(
                f"MPA fit schedule carries invalid range {(q, lo, hi)}")
        if bool(schedule_union[q, lo:hi].any()):
            raise ValueError(
                f"MPA fit schedule carries overlapping range {(q, lo, hi)}")
        schedule_union[q, lo:hi] = True
    if not bool(schedule_union.all()):
        raise ValueError("MPA fit schedule does not cover every q/column pair")
    union = np.zeros_like(done)
    seen = set()
    for row in journal:
        record = tuple(int(value) for value in row)
        if record not in allowed:
            raise ValueError(
                f"MPA partial fit journal range {record} is outside the "
                "current column schedule")
        if record in seen:
            raise ValueError(
                f"MPA partial fit journal repeats scheduled range {record}")
        seen.add(record)
        q, lo, hi = record
        if bool(union[q, lo:hi].any()):
            raise ValueError(
                f"MPA partial fit journal range {record} overlaps an earlier "
                "committed range")
        union[q, lo:hi] = True
    if not np.array_equal(done, union):
        raise ValueError(
            "MPA partial fit blocks_done does not exactly equal the union of "
            "its validated journal ranges")
    return ledger



def read_head_fit_collective(src, *, mesh_xy, to_unit=None):
    """Collectively read and certify the scalar head fit through SlabIO.

    COLLECTIVE over ``mesh_xy``.  ``src`` must be a PATH.  Every open on
    this path is read-only (h5py ``'r'`` twice, then FFI ``'r'``), which is
    the cross-stack concurrency the one-owner registry allows.

    WHAT IS READ, stated because this is the call that produced failure
    signature S3 (``docs/architecture/slab_io.md#s3``): the four
    ``mpa_head/{sample_z,sample_Wc,Omega_p,B_p}`` vectors, WHOLE, with
    ``partition_spec=P(None)`` and **no offset and no shape** — so the
    extent comes from the dataset and the offset that reaches the FFI is
    zero.  ``sample_z``/``sample_Wc`` are 2·n_p long, ``Omega_p``/``B_p``
    are n_p.  A nonzero ``offset_base`` in a refusal from this call is not
    an arithmetic mistake in this function; it is the marshal.

    RETURNS HOST NUMPY, not sharded arrays, despite taking ``mesh_xy``:
    the four vectors come back ``np.complex128`` via ``as_numpy=True``,
    alongside ``units``, ``diagnostics``, ``provenance``, ``model``,
    ``occupation_stamps`` and ``ready``.

    "CERTIFY" MEANS: :func:`validate_fit_store` on the body, head
    ``format_version == 2``, ``ready``, head-vs-body ``mpa_grid_hash``,
    a known head model, and observed ``condition`` / ``backward_error``
    against the stamped ``*_max_allowed``.  Those two thresholds DEFAULT TO
    INFINITY when absent, so an unstamped head certifies vacuously.
    """
    from file_io.mpa_store import (
        MPA_HEAD_SUFFIX,
        _HEAD_FIT_MODELS,
        _OCC_STAMP_ORDER,
        _h5,
        _open_fit,
        _qs,
        _unit_scale,
    )
    from jax.sharding import PartitionSpec as P

    from file_io.slab_io import SlabIO

    ledger = validate_fit_store(src)
    with _h5(src, "r") as grp:
        _open_fit(grp)
        if MPA_HEAD_SUFFIX not in grp:
            raise ValueError("MPA fit store carries no scalar head")
        head = grp[MPA_HEAD_SUFFIX]
        if int(head.attrs.get("format_version", -1)) != 2:
            raise ValueError("collective scalar-head reader requires format version 2")
        if not bool(head.attrs.get("ready", False)):
            raise ValueError("scalar MPA head is NOT READY")
        source_unit = _qs().qirr_attr_str(head, "frequency_unit")
        model = _qs().qirr_attr_str(head, "model")
        grid_hash = _qs().qirr_attr_str(head, "mpa_grid_hash")
        occupation_stamps = None
        if ("mpa_" + _OCC_STAMP_ORDER[0]) in head.attrs:
            occupation_stamps = {
                "occ_hash": _qs().qirr_attr_str(head, "mpa_occ_hash"),
                "mu_ry": float(head.attrs["mpa_mu_ry"]),
            }
        diagnostics = {
            key: float(head.attrs[key])
            for key in (
                "fit_condition", "fit_backward_error",
                "fit_max_abs_residual")
        }
        provenance = {
            str(key)[len("fit_"):]: head.attrs[key]
            for key in head.attrs if str(key).startswith("fit_")
            and str(key) not in diagnostics
        }
    if str(grid_hash) != str(ledger["w_grid_hash"]):
        raise ValueError("scalar-head/body MPA grid hashes differ")
    if str(model) not in _HEAD_FIT_MODELS:
        raise ValueError(
            f"read_head_fit_collective: stored head model {model!r} is not "
            f"one of {_HEAD_FIT_MODELS}; a consumer must not silently "
            "interpret an unknown fitting protocol")
    condition_limit = float(provenance.get(
        "condition_max_allowed", np.inf))
    backward_limit = float(provenance.get(
        "backward_error_max_allowed", np.inf))
    if diagnostics["fit_condition"] > condition_limit:
        raise ValueError("scalar-head MPA fit exceeds its condition gate")
    if diagnostics["fit_backward_error"] > backward_limit:
        raise ValueError("scalar-head MPA fit exceeds its backward-error gate")

    prefix = MPA_HEAD_SUFFIX + "/"
    with SlabIO(src, mode="r", mesh=mesh_xy) as io:
        z = io.read_slab(
            prefix + "sample_z", partition_spec=P(None), as_numpy=True)
        wc = io.read_slab(
            prefix + "sample_Wc", partition_spec=P(None), as_numpy=True)
        poles = io.read_slab(
            prefix + "Omega_p", partition_spec=P(None), as_numpy=True)
        residues = io.read_slab(
            prefix + "B_p", partition_spec=P(None), as_numpy=True)
    z = np.asarray(z, dtype=np.complex128)
    wc = np.asarray(wc, dtype=np.complex128)
    poles = np.asarray(poles, dtype=np.complex128)
    residues = np.asarray(residues, dtype=np.complex128)
    if z.shape != wc.shape or poles.shape != residues.shape:
        raise ValueError("collective scalar-head payload has inconsistent shapes")
    if not all(np.all(np.isfinite(x)) for x in (z, wc, poles, residues)):
        raise ValueError("collective scalar-head payload is not finite")
    if to_unit is not None:
        # THE SHARED HELPER, not a second copy of the same policy: it
        # refuses an undeclared unit by NAME and prints both spellings,
        # where the inline version said only that the conversion was
        # "unsupported" (audit §E.3 item 12).
        scale = _unit_scale(source_unit, to_unit, "read_head_fit_collective")
        z, poles, residues = z * scale, poles * scale, residues * scale
        source_unit = str(to_unit)
    return {
        "sample_z": z,
        "sample_Wc": wc,
        "Omega_p": poles,
        "B_p": residues,
        "units": {
            "frequency": source_unit,
            "Wc": "a.u.",
            "residue": f"{source_unit}*a.u.",
        },
        "diagnostics": diagnostics,
        "provenance": provenance,
        "model": model,
        "occupation_stamps": occupation_stamps,
        "ready": True,
    }



def read_head_fit(src, *, to_unit=None, mode="r"):
    """Read the complete scalar q->0 MPA fit; refuse absent/partial data."""
    from file_io.mpa_store import (
        MPA_HEAD_SUFFIX,
        _HEAD_FIT_MODELS,
        _h5,
        _open_fit,
        _qs,
        _unit_scale,
    )
    qs = _qs()
    with _h5(src, mode) as grp:
        _open_fit(grp)
        if MPA_HEAD_SUFFIX not in grp:
            raise ValueError("read_head_fit: fit store carries no scalar head")
        head = grp[MPA_HEAD_SUFFIX]
        if int(head.attrs.get("format_version", -1)) != 1:
            raise ValueError("read_head_fit: unsupported scalar-head format")
        if not bool(head.attrs.get("ready", False)):
            raise ValueError("read_head_fit: scalar head is NOT READY")
        source_unit = qs.qirr_attr_str(head, "frequency_unit")
        z = np.asarray(head["sample_z"][()])
        wc = np.asarray(head["sample_Wc"][()])
        poles = np.asarray(head["Omega_p"][()])
        residues = np.asarray(head["B_p"][()])
        diagnostics = {
            key: float(head.attrs[key])
            for key in ("fit_condition", "fit_backward_error",
                        "fit_max_abs_residual")
        }
        units = {
            "frequency": source_unit,
            "Wc": qs.qirr_attr_str(head, "Wc_unit"),
            "residue": qs.qirr_attr_str(head, "residue_unit"),
        }
        model = qs.qirr_attr_str(head, "model")
        if model not in _HEAD_FIT_MODELS:
            raise ValueError(
                f"read_head_fit: got scalar-head model {model!r}; want "
                f"one of {_HEAD_FIT_MODELS} — the only fitting protocols "
                f"this reader knows how to interpret, and a pole set "
                f"whose protocol nobody can name cannot be consumed "
                f"correctly.  Fix: refit the head with a known model, or "
                f"teach _HEAD_FIT_MODELS the new one alongside its "
                f"consumer.")
    if to_unit is not None:
        scale = _unit_scale(source_unit, to_unit, "head read")
        z, poles, residues = z * scale, poles * scale, residues * scale
        units["frequency"] = str(to_unit)
        units["residue"] = f"{to_unit}*a.u."
    return {
        "sample_z": z,
        "sample_Wc": wc,
        "Omega_p": poles,
        "B_p": residues,
        "units": units,
        "diagnostics": diagnostics,
        "model": model,
        "ready": True,
    }



def open_w_column_reader(src, *, mesh_xy, headers):
    """Open the persistent source reader for one fit checkpoint epoch."""
    return WColumnReader(src, mesh_xy=mesh_xy, headers=headers)



class WColumnReader:
    """ONE collective source handle for one bounded fit epoch.

    A fit block still performs one H5Dread per sample component and retains
    exactly one budgeted all-frequency column tile per component.  What this
    object removes is file/dataset discovery churn: the source is opened once
    for the epoch, not once for every ``(q, column)`` block.  At the scalar Bi
    geometry that changes 1,105 source opens to 35 without changing the 1,105
    H5Dreads, their selections, the fit algebra, or device memory.

    ``headers`` is explicit because production authenticates both positive
    and optional negative components before collective I/O begins.  Keeping
    those already-closed serial reads outside this lifetime also prevents an
    h5py metadata open from appearing while the collective handle is live.
    Multiple names share this ONE handle, so ordered fitting reads W(z) and
    W(-z) without independently reopening their common source.

    COLLECTIVE over ``mesh_xy``.  Every rank must construct, read in the same
    order, and close.  Use the context-manager spelling; :meth:`read` refuses
    after close rather than silently reviving a handle whose lifetime would
    no longer be visible in the driver.
    """

    def __init__(self, src, *, mesh_xy, headers):
        if mesh_xy is None:
            raise ValueError(
                "WColumnReader requires mesh_xy: its source handle is "
                "collective")
        rows = {str(name): dict(header)
                for name, header in dict(headers).items()}
        if not rows:
            raise ValueError(
                "WColumnReader requires at least one authenticated header")
        self.src = os.fspath(src)
        self.mesh_xy = mesh_xy
        self.headers = rows
        self.h5d_reads = 0
        from file_io.slab_io import SlabIO
        self._io = SlabIO(self.src, mode="r", mesh=mesh_xy)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        io, self._io = getattr(self, "_io", None), None
        if io is not None:
            io.close()

    def read(
        self,
        name,
        q,
        mu_cols,
        *,
        n_cols_buffer,
        tile_bytes=None,
    ):
        """Read one contiguous column tile through the live handle."""
        from file_io.mpa_store import (
            _column_span,
            _validate_column_request,
        )
        from jax.sharding import PartitionSpec as P

        from file_io.slab_io import mesh_divisible_shape

        if self._io is None:
            raise ValueError(
                "WColumnReader.read after close(): open a new epoch reader")
        key = str(name)
        if key not in self.headers:
            raise KeyError(
                f"WColumnReader has no authenticated header for {key!r}; "
                f"available={sorted(self.headers)}")
        header = self.headers[key]
        n_mu = int(header["n_mu"])
        n_omega = int(header["n_omega"])
        iq, cols, budget = _validate_column_request(
            header, q, mu_cols, tile_bytes,
            f"WColumnReader.read({key!r})")
        lo, _, sel = _column_span(cols)
        if sel is not None:
            raise ValueError(
                "WColumnReader.read requires one contiguous column range; "
                "the production fit schedule emits contiguous blocks")
        width = int(n_cols_buffer)
        if width < int(cols.size) or width > budget:
            raise ValueError(
                f"WColumnReader.read: n_cols_buffer={width}, actual "
                f"width={int(cols.size)}, budget={budget}; require "
                "actual <= buffer <= budget")

        spec = P(None, None, ("x", "y"), None)
        shape = mesh_divisible_shape(
            (n_omega, 1, n_mu, width), self.mesh_xy, spec)
        block = self._io.read_slab(
            key, shape=shape, offset=(0, iq, 0, lo),
            valid_shape=(n_omega, 1, n_mu, int(cols.size)),
            partition_spec=spec)
        self.h5d_reads += 1
        return block



def read_w_columns_collective(
    src,
    name,
    q,
    mu_cols,
    *,
    mesh_xy,
    n_cols_buffer,
    tile_bytes=None,
    header=None,
):
    """One-shot collective read of an all-frequency column tile.

    The returned array is ``(n_omega, 1, n_mu_padded, n_cols_buffer)``
    with ``P(None, None, ('x', 'y'), None)``.  The singleton is the stored
    q axis; retaining it makes the SlabIO offset and the fit-store write
    geometry identical.  A short final column block is zero-filled to the
    fixed buffer width and ``valid_shape`` prevents those zeros from reading
    bytes belonging to the next block.  This is the surgical single-read
    door; a scheduled fit must use :class:`WColumnReader` so its epoch does
    not reopen the same source once per column block.
    """
    hdr = read_w_header(src, name) if header is None else header
    with WColumnReader(
            src, mesh_xy=mesh_xy, headers={name: hdr}) as reader:
        return reader.read(
            name, q, mu_cols, n_cols_buffer=n_cols_buffer,
            tile_bytes=tile_bytes)



def read_w_columns(
    src,
    name,
    q,
    mu_cols,
    *,
    tile_bytes=None,
    n_mu_padded=None,
    out_spec=None,
    require_ready=True,
    mode="r",
):
    """A few ν columns of W_q, ACROSS ALL FREQUENCIES.

    Returns ``(n_omega, N_μ_rows, len(mu_cols))`` complex — the shape
    the per-element plasmon-pole fit consumes.  This is the read the
    leading frequency axis exists for: the fit needs all of ω for one
    (μ, ν) element and never needs all of (μ, ν) for one ω, so the
    frequency axis is the OUTER one on disk and the innermost one in the
    solve.

    THE BUDGET REFUSES BY NAME.  ``len(mu_cols)`` is checked against
    :func:`choose_column_budget` and a request that busts it raises with
    the full arithmetic — the per-column cost, the total, the tile it is
    measured against, the ratio, and the count that would have fit.
    Silently truncating or silently allowing would each defeat the
    constraint the number encodes: a small number of W_q(μ,ν) copies fit
    at once, and this block is priced to be one of them.

    THE SHARDING IS 1-D ON THE ROW AXIS.  ``out_spec`` is checked, not
    applied — the read itself is host-side h5py and the placement is the
    caller's — but a 2-D spec is refused here rather than downstream,
    because by the time it is downstream the column count is no longer
    the number the budget was computed for.  See
    :func:`_refuse_two_dim_sharding`.

    REQUIRES EVERY SLAB.  The block spans the whole frequency axis, so
    every ω must be ready; a partially filled file refuses and names how
    many slabs are missing.  That is stricter than :func:`read_w_slab`
    on purpose — a fit run on the ready half of a grid produces poles
    that are wrong rather than absent.
    """
    from file_io.mpa_store import (
        _column_span,
        _h5,
        _qs,
        _refuse_two_dim_sharding,
        _validate_column_request,
    )
    qs = _qs()
    header = read_w_header(src, name, mode=mode)
    _refuse_two_dim_sharding(out_spec, f"read_w_columns({name!r})")
    n_mu = header["n_mu"]
    iq, cols, _ = _validate_column_request(
        header, q, mu_cols, tile_bytes, f"read_w_columns({name!r})",
        require_ready=require_ready)

    # ONE HYPERSLAB, NOT ONE PER FREQUENCY.  A contiguous run becomes a
    # slice (HDF5 reads it as a single hyperslab); anything else is a
    # point selection on the LAST axis only, which h5py supports and
    # which keeps the row axis whole — the axis the caller shards.
    lo, hi, sel = _column_span(cols)
    with _h5(src, mode) as grp:
        block = grp[name][:, iq, :, slice(lo, hi) if sel is None else sel]
    block = np.asarray(block)

    if n_mu_padded is not None and int(n_mu_padded) != n_mu:
        pad = int(n_mu_padded) - n_mu
        if pad < 0:
            raise ValueError(
                f"read_w_columns({name!r}): the file stores {n_mu} "
                f"logical centroids and the caller asked to pad the row "
                f"axis DOWN to {n_mu_padded}.  The pad only ever grows "
                f"the extent.")
        # ROWS ONLY.  The columns are a selection the caller chose, not
        # an axis with a pad; padding them would invent centroids the
        # caller did not ask for and shift every index in ``mu_cols``.
        block = np.pad(block, ((0, 0), (0, pad), (0, 0)))
    return block



def read_w_tables(src, name, *, mode="r"):
    """The stored unfold tables — ``qirr_store.read_tables``, unchanged.

    Re-exported rather than reimplemented, and named here so a caller
    reading a v2 file does not have to know which module owns the table
    group.  They are ω-INDEPENDENT: one set for the whole frequency
    axis, because the symmetry operation acts on (q, μ, ν).

    THROUGH :func:`_h5`, and that is the whole content of this wrapper.
    ``qirr_store.read_tables`` opens h5py itself, so forwarding ``src``
    to it was an h5py open the ownership registry could not see — the one
    blind spot in this module's one-owner invariant, and not a rare one:
    it runs on EVERY RANK in production (``gw/mpa/fit_driver.py``'s
    unfold-table read).  The door takes the open and hands the already-open
    group on, so the format layer still owns the reading and the registry
    still owns the counting (audit §E.3 item 2).
    """
    from file_io.mpa_store import (
        _h5,
        _qs,
    )
    with _h5(src, mode) as grp:
        return _qs().read_tables(grp, name)



def read_w_slab(
    src,
    name,
    i_omega,
    *,
    q=None,
    unfold=False,
    mesh_xy=None,
    n_mu_padded=None,
    require_ready=True,
    mode="r",
):
    """One frequency slab: ``(n_q, N_μ, N_μ)``, or one q of it.

    THE REMOVABILITY CLAIM, AS A FUNCTION.  What comes back for
    ``unfold=False`` and ``n_mu_padded=None`` is bit-identical to what
    ``qirr_store.read_tensor`` returns from a version-1 file written
    from this slab — same bytes, same wedge, same tables.  That is the
    whole content of "the leading dimension is removable later": the
    axis is a container, not a change of meaning, and dropping it is a
    slice rather than a migration.  ``test_the_leading_axis_is_
    removable`` asserts it attr-for-attr.

    Parameters
    ----------
    q
        Optional q index into the stored wedge.  ``None`` returns every
        stored q at this ω.
    unfold
        Unfold the wedge to the full BZ AT THIS FREQUENCY.  The tables
        are ω-independent, so this is ``unfold_isdf_operator`` on the
        slab — the same call, the same arguments, one frequency at a
        time.  Needs ``mesh_xy``.
    n_mu_padded
        Re-apply a μ pad of the READER's own width.  The file stores the
        LOGICAL extent, so a consumer that wants the padded in-memory
        layout asks for it here rather than finding the writer's pad and
        hoping it matches.
    require_ready
        Refuse when this slab's ledger bit is False.  Default True.
    """
    from file_io.mpa_store import (
        _h5,
        _qs,
    )
    qs = _qs()
    header = read_w_header(src, name, mode=mode)
    i = int(i_omega)
    n_omega = header["n_omega"]
    if not 0 <= i < n_omega:
        raise IndexError(
            f"mpa_store: frequency index {i} is outside [0, {n_omega}) "
            f"for {name!r}.")
    if require_ready and not bool(header["data_ready"][i]):
        raise ValueError(
            f"mpa_store: {name!r} frequency slab {i} (ω = "
            f"{header['omega'][i]}) is PRESENT AND CORRECTLY SHAPED but "
            f"its data_ready bit is False — it is allocated space, not "
            f"data.  {header['n_ready']} of {n_omega} slabs are ready.  "
            f"Reading it would hand the fit a slab of zeros that passes "
            f"every shape check, which is the mechanism behind the "
            f"all-zero-screening incident: a plausible excitonic "
            f"spectrum out of a W that was never written.  A "
            f"frequency-resolved file reaches this state routinely — "
            f"the producer fills ω one line-batched sweep at a time — "
            f"so the ledger is per slab and not per file.  Pass "
            f"require_ready=False to inspect the placeholder "
            f"deliberately.")

    with _h5(src, mode) as grp:
        ds = grp[name]
        raw = ds[i] if q is None else ds[i, int(q)]
    raw = np.asarray(raw)

    # THE TABLES ARE READ ONLY WHEN THEY ARE NEEDED, which is the
    # unfold and the re-pad.  The production per-slab read is neither —
    # a consumer walking ω takes the wedge as stored — and opening the
    # table group on every one of those would be a second file open per
    # frequency for arrays nobody looks at.  Their DIGEST was already
    # checked by ``read_w_header`` above, so this is a saved read and
    # not a skipped check.
    if not unfold and n_mu_padded is None:
        return raw, header

    tables = read_w_tables(src, name, mode=mode)
    can = tables.canonical()
    if n_mu_padded is not None and int(n_mu_padded) != int(can.n_mu):
        pad = int(n_mu_padded) - int(can.n_mu)
        if pad < 0:
            raise ValueError(
                f"mpa_store: {name!r} stores {can.n_mu} logical "
                f"centroids and the caller asked to pad DOWN to "
                f"{n_mu_padded}.  The pad only ever grows the extent; a "
                f"smaller request means the caller and the file "
                f"disagree about the centroid set.")
        widths = [(0, 0)] * (raw.ndim - 2) + [(0, pad), (0, pad)]
        raw = np.pad(raw, widths)
        can = can.padded(int(n_mu_padded))

    if not unfold or header["q_storage"] == "full":
        return raw, header
    if q is not None:
        raise ValueError(
            f"mpa_store: {name!r} cannot unfold a single stored q "
            f"(q={q}).  The unfold gathers every full-BZ row from its "
            f"IBZ parent, so it needs the whole wedge at this ω; ask "
            f"for q=None and index the result.")
    if mesh_xy is None:
        raise ValueError(
            f"mpa_store: {name!r} is stored on the q wedge "
            f"({header['n_q_on_disk']} of {header['n_q_full']} q) and "
            f"unfolding slab {i} needs a mesh; pass mesh_xy= or "
            f"unfold=False to take the wedge.")
    import jax.numpy as jnp
    # THROUGH THE SERVICE'S DOOR, not past it: the top-level package,
    # never ``symmetry_maps.maps``.  Reaching a submodule is what stops
    # a service being replaceable, and ``test_layering`` enforces it.
    from symmetry_maps import unfold_isdf_operator
    # Test seam only — the bulk to-device transfer below has zero
    # production callers; production unfolds sharded in _finish_pole_read.
    full = unfold_isdf_operator(
        jnp.asarray(raw),
        irr_idx=can.irr_idx_q,
        sym_idx=can.sym_idx_q,
        sym_perm=can.sym_perm,
        L_table=can.L_table,
        q_irr_frac=can.q_irr_frac,
        mesh_xy=mesh_xy,
        n_sym_spatial=int(can.n_sym_spatial),
    )
    return full, header



def read_w_header(src, name, *, mode="r"):
    """Everything the file CLAIMS about ``name``, reading no tensor data.

    Returns a plain dict with these keys, every one of which some reader
    below indexes by name: ``format_version``, ``freq_axis``, ``n_omega``,
    ``omega`` ``(n_omega,)`` complex128, ``omega_line`` ``(n_omega,)``
    int32, ``omega_units``, ``sampling``, ``grid_hash``, ``data_ready``
    ``(n_omega,)`` bool, ``n_ready``, ``q_storage``, ``n_q_on_disk``,
    ``n_q_full``, ``n_mu``, ``n_rmu_logical``, ``centroid_hash``,
    ``table_hash``, ``closure_verdict``, ``provenance``.

    Rank-local and serial, but called on every rank from three collective
    functions — a collective caller must invoke it uniformly.  ``src`` is a
    path (this call's ``_h5`` owns the handle) or an already-open group (the
    caller owns it, and ``mode`` is then IGNORED by ``QirrDest``: the
    parameter looks live and is not).

    Every cross-check the format owns runs here,
    so a caller that got a header back has already been told the file is
    self-consistent, and every reader below calls this first rather than
    repeating the checks — one implementation of "what does this file
    say", because a second one is how a reader ends up disagreeing with
    the format about what it is holding.
    """
    from file_io.mpa_store import (
        QIRR_FORMAT_VERSION_FREQ,
        _FREQ_ATTR,
        _MPA_OWNED_ATTRS,
        _SAMPLING_ORDER,
        _canonical_sampling,
        _h5,
        _open_w,
        _qs,
        _refuse_unless_rank_matches_version,
        omega_grid_digest,
    )
    qs = _qs()
    with _h5(src, mode) as grp:
        ds, mgrp = _open_w(grp, name)
        version = _refuse_unless_rank_matches_version(ds, name)
        if version != QIRR_FORMAT_VERSION_FREQ:
            raise ValueError(
                f"mpa_store: {name!r} is format version {version}; the "
                f"frequency-resolved readers are version "
                f"{QIRR_FORMAT_VERSION_FREQ}.  Use "
                f"qirr_store.read_tensor for a version-1 tensor.")

        # THE PARTIAL-STAMP REFUSAL, version 2's half.  The rank check
        # above settles which format this is; this settles whether the
        # format's own record is whole.  Named rather than left to a
        # KeyError deep in the read, because "which attr is missing" is
        # the question a half-written file raises and a traceback
        # through ``ds.attrs[...]`` answers it one attr at a time.
        absent = [a for a in _MPA_OWNED_ATTRS if a not in ds.attrs]
        if absent:
            raise ValueError(
                f"mpa_store: {name!r} is a version "
                f"{QIRR_FORMAT_VERSION_FREQ} tensor missing {absent}.  "
                f"A half-stamped file is refused rather than read: the "
                f"missing half is the sampling protocol, which is what "
                f"says what the ω values MEAN, and a fit against "
                f"abscissae nobody can characterise is a fit nobody can "
                f"reproduce or extend.")

        n_omega = int(ds.shape[0])
        stamped_n = int(ds.attrs["mpa_n_omega"])
        if stamped_n != n_omega:
            raise ValueError(
                f"mpa_store: {name!r} stamps mpa_n_omega={stamped_n} but "
                f"its leading axis is {n_omega}.  The SHAPE is the "
                f"primary discriminant and the attr is its cross-check, "
                f"so a disagreement is a refusal and not a preference.")

        omega = mgrp["omega"][()]
        line = mgrp["omega_line"][()]
        ready = np.asarray(mgrp["data_ready"][()], dtype=bool)
        for label, arr in (("omega", omega), ("omega_line", line),
                           ("data_ready", ready)):
            if int(np.asarray(arr).shape[0]) != n_omega:
                raise ValueError(
                    f"mpa_store: {name!r} has {n_omega} frequency slabs "
                    f"but its {label} is length "
                    f"{int(np.asarray(arr).shape[0])}.  Each of these is "
                    f"one entry per slab; a short one cannot address "
                    f"every slab and a long one addresses slabs that do "
                    f"not exist.")

        # THE SAMPLING ATTRS, through :data:`_SAMPLING_ORDER` and back
        # through ``_canonical_sampling`` — the same coercions the stamp
        # used, run by the one function that owns them.  (The digest
        # check below already re-canonicalises, so this adds no check
        # that did not run before; it only runs one call earlier.)
        sampling, _ = _canonical_sampling({
            key: (qs.qirr_attr_str(ds, "mpa_" + key) if key == "protocol"
                  else ds.attrs["mpa_" + key])
            for key in _SAMPLING_ORDER})
        recomputed = omega_grid_digest(omega, line, sampling)
        stamped_hash = qs.qirr_attr_str(ds, "mpa_grid_hash")
        if stamped_hash != recomputed:
            raise ValueError(
                f"mpa_store: {name!r} ω-grid hash mismatch.  Stamped "
                f"{stamped_hash}, the grid and protocol on disk hash to "
                f"{recomputed}.  The sampling points are not the ones "
                f"this tensor was evaluated at, so every pole fitted "
                f"from it would be fitted against the wrong abscissae.")

        scalar_ready = ds.attrs.get("qirr_data_ready", None)
        if scalar_ready is not None and bool(scalar_ready) != bool(
                ready.all()):
            raise ValueError(
                f"mpa_store: {name!r} stamps qirr_data_ready="
                f"{bool(scalar_ready)} but its per-frequency ledger has "
                f"{int(ready.sum())} of {n_omega} slabs ready.  The "
                f"scalar is the CONSERVATIVE summary any version-1 "
                f"reader will honour, so it must be all(ledger); a "
                f"disagreement is a file claiming readiness it cannot "
                f"support.")

        # The q_irr half: tables, digest, shape-vs-attr — the landed
        # checks, run against the PER-FREQUENCY extents.
        tables = qs.read_tables(grp, name)
        can = tables.canonical()
        if can.digest() != qs.qirr_attr_str(ds, "qirr_table_hash"):
            raise ValueError(
                f"mpa_store: {name!r} table hash mismatch.  The unfold "
                f"tables are not the ones this tensor was written "
                f"against, so every q it reconstructs — at every ω — "
                f"would be a permutation of the wrong centroids.")
        n_q_on_disk = int(ds.shape[1])
        n_mu = int(ds.shape[3])
        shape_says = qs.validate_qirr_tables(can, n_q_on_disk, n_mu)
        attr_says = qs.qirr_attr_str(ds, "q_storage")
        if attr_says != shape_says:
            raise ValueError(
                f"mpa_store: {name!r} shape says q_storage="
                f"{shape_says!r} ({n_q_on_disk} q rows per frequency "
                f"against {can.n_q_full} full-BZ rows in the tables) but "
                f"the attr says {attr_says!r}.  The SHAPE is the primary "
                f"discriminant and the attr is its cross-check, so a "
                f"disagreement is a refusal.")

        prov = {k[len("prov_"):]: v for k, v in ds.attrs.items()
                if str(k).startswith("prov_")}
        sampling_extra = {
            str(k)[len("mpa_prov_"):]: v for k, v in ds.attrs.items()
            if str(k).startswith("mpa_prov_")}
        for key in ("qirr_generator_commit", "qirr_written_utc",
                    "qirr_writer", "mpa_writer"):
            if key in ds.attrs:
                prov[key] = qs.qirr_attr_str(ds, key)
        return {
            "format_version": version,
            "freq_axis": qs.qirr_attr_str(ds, _FREQ_ATTR),
            "n_omega": n_omega,
            "omega": omega,
            "omega_line": line,
            "omega_units": qs.qirr_attr_str(ds, "mpa_omega_units"),
            "sampling": sampling,
            "grid_hash": recomputed,
            "data_ready": ready,
            "n_ready": int(ready.sum()),
            "q_storage": shape_says,
            "n_q_on_disk": n_q_on_disk,
            "n_q_full": can.n_q_full,
            "n_mu": n_mu,
            "dtype": np.dtype(ds.dtype),
            "n_rmu_logical": int(ds.attrs["qirr_n_rmu_logical"]),
            "centroid_hash": qs.qirr_attr_str(ds, "qirr_centroid_hash"),
            "table_hash": can.digest(),
            "closure_verdict": qs.qirr_attr_str(ds, "qirr_closure_verdict"),
            "provenance": prov,
            "sampling_extra": sampling_extra,
        }



def read_w_slab_collective(
    src,
    name,
    i_omega,
    *,
    mesh_xy,
    require_ready=True,
):
    """Read one frequency slab directly into ``P(None,'x','y')``.

    This is the inverse of :func:`write_w_slab_collective`: the file keeps
    the logical centroid extent, SlabIO pads only the two distributed axes,
    and no rank materializes the complete ``(q,mu,nu)`` slab.  The routine
    is valid for any MPA frequency tensor with this layout (in particular
    both ``chi(z)`` and ``Wc(z)``); the historical ``W`` in its name denotes
    the on-disk format, not an extra transport.

    COLLECTIVE over ``mesh_xy``: every rank calls it, in the same order,
    for the same ``i_omega``.  ``src`` must be a PATH (SlabIO).

    RETURNS ``(slab, header)``.  ``slab`` is
    ``(n_q_on_disk, n_mu_padded, n_mu_padded)`` complex128 — the μ extent
    is the canonical ``runtime.padding.padded_mu_extent`` round-up, NOT
    ``header['n_mu']`` and not the weaker per-axis SlabIO round-up.  The
    4-D read is issued at ``P(None, None, 'x', 'y')`` and the returned 3-D
    array therefore carries ``P(None, 'x', 'y')``; that is inferred from the
    leading index, not asserted on the way out.  ``header`` is
    :func:`read_w_header`'s dict, read with h5py on every rank BEFORE the
    collective handle opens (read-only on both stacks, which the one-owner
    registry allows and counts).
    """
    from jax.sharding import PartitionSpec as P
    from file_io.slab_io import SlabIO
    from runtime.padding import padded_mu_extent

    header = read_w_header(src, name)
    i = int(i_omega)
    if not 0 <= i < header["n_omega"]:
        raise IndexError(
            f"read_w_slab_collective: frequency index {i} is outside "
            f"[0,{header['n_omega']}) for {name!r}")
    if require_ready and not bool(header["data_ready"][i]):
        raise ValueError(
            f"read_w_slab_collective: {name!r} slab {i} is allocated but "
            "not ready")

    logical = (1, header["n_q_on_disk"], header["n_mu"], header["n_mu"])
    spec = P(None, None, "x", "y")
    # Both μ axes belong to one canonical in-memory carrier.  Padding each
    # axis only by its own mesh side is legal for this SlabIO read but is not
    # sufficient for downstream product-face/all-to-all consumers.  In
    # particular, P=36 and n_mu=2070 would otherwise reload χ at 2070 while
    # V is correctly carried at padded_mu_extent(2070, 36) == 2088.
    n_mu_padded = int(padded_mu_extent(header["n_mu"], mesh_xy))
    shape = (1, header["n_q_on_disk"], n_mu_padded, n_mu_padded)
    with SlabIO(src, mode="r", mesh=mesh_xy) as io:
        slab = io.read_slab(
            name, shape=shape, offset=(i, 0, 0, 0),
            valid_shape=logical, partition_spec=spec)
    return slab[0], header



def read_occupation_stamps(src, *, mode="r"):
    """Return the fit store's occupation stamps, or None if unstamped."""
    from file_io.mpa_store import (
        _OCC_STAMP_ORDER,
        _h5,
        _qs,
    )
    qs = _qs()
    with _h5(src, mode) as grp:
        if ("mpa_" + _OCC_STAMP_ORDER[0]) not in grp.attrs:
            return None
        return {
            "occ_hash": qs.qirr_attr_str(grp, "mpa_occ_hash"),
            "mu_ry": float(grp.attrs["mpa_mu_ry"]),
            "smearing_family": qs.qirr_attr_str(grp, "mpa_smearing_family"),
            "smearing_width_ry": float(grp.attrs["mpa_smearing_width_ry"]),
            "occ_nelec": float(grp.attrs["mpa_occ_nelec"]),
        }


def _fix_sphere_wrap(zx):
    """(reference ``_fix_sphere_wrap``) Half-boundary wrap disambiguation:
    per q, among the ±1/2 sign candidates keep the one whose sphere fits
    max|q+G|² ≤ cutoff.  No-op on grids without half components."""
    changed = 0
    for q in range(zx["nq"]):
        base = zx["qfr_raw"][q] - np.round(zx["qfr_raw"][q])
        cands = [[]]
        for c in range(3):
            opts = [0.5, -0.5] if abs(abs(base[c]) - 0.5) < 1e-9 else [base[c]]
            cands = [cc + [o] for cc in cands for o in opts]
        n = int(zx["ngk"][q])
        G = zx["gvec"][q][:, :n].astype(np.float64)
        best, bestm = None, None
        for cc in cands:
            qc = np.asarray(cc)
            K = zx["bvec"].T @ (qc[:, None] + G)
            m = float(np.max(np.sum(K * K, axis=0)))
            if bestm is None or m < bestm:
                best, bestm = qc, m
        assert bestm <= zx["zeta_cutoff"] + 1e-9, \
            f"q={q}: no candidate wrap fits the stored sphere"
        if np.max(np.abs(best - zx["qfr"][q])) > 1e-12:
            changed += 1
        zx["qfr"][q] = best
    if changed:
        print(f"  [wrapfix] {changed} of {zx['nq']} q relabeled to the "
              f"sphere-derived center")



def read_vq_payload(restart_file: str, zeta_file: str, *,
                     mesh: Mesh | None = None, log_fn=print, input_file=None,
                     require_slab: bool = True,
                     require_full_bz_zeta: bool = True) -> dict:
    """Load the coarse-grid ζ/ψ/tile data into a plain-dict bundle ``zx``.

    q-LABELING (the two wrap traps, KNOWN_SANDBOX_ERRORS 2026-07-17):
    ``zeta_q.h5 mf_header/kpoints/rk`` stores the UNWRAPPED QE list, while
    the stored ζ spheres are centred on the BGW-WRAPPED q (worth a measured
    155× on the physical interp ladder).  ``np.round`` is round-half-to-even,
    so components at exactly 1/2 need the sphere itself to pick the sign:
    keep the candidate wrap minimising max|q+G|² over the stored sphere
    (``_fix_sphere_wrap``).  Every downstream phase/kernel uses these
    wrapped labels.

    ``require_full_bz_zeta`` is true for the interpolation model, which reads
    every stored ζ tile.  The pure-refit caller sets it false: that route reads
    ζ metadata and the producer's solve provenance but reconstructs ζ(Q) from
    the canonical full-BZ WFN source, so an IBZ-only ζ file is sufficient.

    THE THREE BIG q-STACKS STAY ON DISK.  ``ZG`` (nq, n_μ, ngkmax),
    ``Vqmunu`` and ``W0`` (nq, n_μ, n_μ) are read-only and every consumer
    slices them on the q axis, so they are kept as LAZY handles and
    pulled per q-chunk instead of being materialised per process.  At
    the converged MoS2 reference (nq = 144, n_μ = 2412, ngkmax = 8603) ζ
    alone is **47.8 GB**; four ranks per Perlmutter GPU node (251 GB) cannot
    hold it, which is what confined the exciton driver to ``--vq-mode
    ongrid``.  Lazy, the resident host cost is ψ (3.7 GB) plus one q-chunk.
    Same principle as the ψ(G) host cache in the GW path: large read-only
    caches are pulled per slice, never carried.

    The zeta service owns its metadata and bounded tile transport. Static
    interactions and family wavefunctions use this module's canonical
    parent-face and q-star readers. The returned cache owns one zeta handle,
    released by the consuming VQ cache's close operation.

    ``mesh`` selects the ζ TRANSPORT (see :func:`_zeta_mesh_for_loader`):
    with a mesh whose stack can serve SlabIO, ``prepare_coarse``'s q-chunk
    read becomes a per-rank hyperslab; without one, the local h5py plan
    runs, byte-identical.  It is optional so the host-only diagnostics and
    the fixture tests keep working on a bare checkout.
    """
    from bse.vq_interp import (
        assert_slab_scope,
    )
    zx = {"restart_file": restart_file, "zeta_file": zeta_file}
    zx.update(read_coarse_interactions(restart_file, input_file, mesh))
    _mesh_for_loader, _distributed = _zeta_mesh_for_loader(mesh, log_fn=log_fn)
    zl = open_zeta(zeta_file, mesh=_mesh_for_loader)
    zx["_zeta_loader"] = zl
    zx["zeta_distributed"] = _distributed
    zx["ZG"] = _ZetaGTiles(zl, path=zeta_file, distributed=_distributed)
    # ``ngk_per_q`` is ``isdf_header/ngk`` (the per-q ζ SPHERE size).
    # ``zl.ngk`` is a DIFFERENT array — ``mf_header/kpoints/ngk``, the
    # WFN's per-k G count, bound by ``bind_mf_attrs``.  Reading the wrong
    # one truncates every sphere silently.
    zx["gvec"] = np.asarray(zl.gvec_components).astype(np.int64)
    zx["ngk"] = np.asarray(zl.ngk_per_q).astype(int)
    fg = np.asarray(zl.fft_grid).astype(int)
    qraw = np.array(zl.kpoints, copy=True)
    zx["adot"] = np.asarray(zl.adot)
    blat = float(np.real(zl.blat))
    # BGW stores bvec in units of blat = 2π/alat; physical bohr⁻¹
    # (|bvec^T g|² in Ry) needs the blat factor (measured: 10.4%
    # makeVq-vs-disk residual without it).
    zx["bvec"] = np.asarray(zl.bvec) * blat
    zx["celvol"] = float(np.real(zl.cell_volume))
    rmu_idx = np.asarray(zl.r_mu_fft_idx).astype(int)
    zx["zeta_cutoff"] = float(zl.zeta_cutoff_ry)
    ifmax = np.asarray(zl.ifmax)
    zx["nk"], zx["nb"], zx["ns"], zx["n_mu"] = zx["psi"].shape
    zx["nq"] = zx["ZG"].shape[0]
    zx["ngkmax"] = zx["ZG"].shape[2]
    zx["nx"], zx["ny"], zx["nz"] = [int(x) for x in fg]
    zx["n_rtot"] = zx["nx"] * zx["ny"] * zx["nz"]
    if require_full_bz_zeta and zx["nq"] != zx["nk"]:
        raise ValueError(
            f"vq_interp needs FULL-BZ zeta storage: zeta_q.h5 has nq={zx['nq']} "
            f"but the k-grid has nk={zx['nk']} (IBZ cascade active).  "
            f"Regenerate the fit with full-BZ zeta, or wait for the IBZ-zeta "
            f"unfold (deferred; must route through the one SymMaps sym-action).")
    if not require_full_bz_zeta and zx["nq"] != zx["nk"]:
        log_fn(
            f"  [vq_interp] pure refit accepts IBZ-only ζ metadata: "
            f"nq={zx['nq']}, full-BZ nk={zx['nk']}. Stored ZG is not read; "
            "the symmetry service supplies the full-BZ WFN source and the "
            "restart loader supplies unfolded V_qmunu for the on-grid gate.")
    # ── q LABELS FOR A FULL-BZ ζ WRITTEN FROM A SYMMETRY-REDUCED WFN ──────
    # ``mf_header`` is copied verbatim from the WFN, so ``kpoints/rk`` holds
    # the WFN's k-list — the IBZ when the mean-field run used symmetry.  The ζ
    # historical full-BZ ζ writer stored:
    # ``_bgw_wrap_q(sym.kvecs_asints) / kgrid`` (gw/isdf_fitting.py, the
    # ``q_irr_frac is None`` branch).  On the MoS2 4x4 deck that is 16 ζ tiles
    # against a 10-row ``rk``, and ``qraw[:nq]`` silently returned 10 rows —
    # surfacing three lines later as the misleading "duplicate k labels in rk
    # list" (job 7882499 cell exb64s).  Reconstruct the writer's own list.
    #
    # THIS IS NOT TAKEN ON TRUST.  ``run_gates`` rebuilds V from ζ at EVERY q
    # and compares it against the stored ``V_qmunu[q]`` at 5e-6
    # (``makeVq_vs_disk_Vqmunu_allq_max``).  A permuted or mis-wrapped q list
    # cannot pass that: each q's ζ would be checked against a different q's
    # stored tile.  Do not run this path with ``LORRAX_SKIP_VQ_GATES=1`` until
    # it has passed once on a given deck.
    if qraw.shape[0] < zx["nq"]:
        _kg = np.asarray(zx["kgrid"], dtype=np.float64)
        _idx = np.stack(np.meshgrid(np.arange(_kg[0]), np.arange(_kg[1]),
                                    np.arange(_kg[2]), indexing="ij"),
                        axis=-1).reshape(-1, 3).astype(np.float64)
        _wrapped = np.where(_idx > _kg[None, :] / 2.0, _idx - _kg[None, :], _idx)
        qfull = _wrapped / _kg[None, :]
        # Necessary condition: every k the mean-field header DOES carry must
        # appear in the reconstruction (catches a transposed grid or the wrong
        # wrap convention, both of which would otherwise reach run_gates as a
        # confusing numerical failure).
        _have = {tuple(np.rint(v * _kg).astype(int) % _kg.astype(int))
                 for v in qfull}
        _missing = [tuple(np.rint(v * _kg).astype(int) % _kg.astype(int))
                    for v in qraw if tuple(np.rint(v * _kg).astype(int)
                                           % _kg.astype(int)) not in _have]
        if _missing:
            raise ValueError(
                f"reconstructed full-BZ q list does not contain "
                f"{len(_missing)} of the {qraw.shape[0]} mf_header k-points "
                f"(e.g. {_missing[0]}); the on-disk q ordering is not the "
                f"C-order wrapped {tuple(int(v) for v in zx['kgrid'])} grid "
                f"this reconstruction assumes.")
        try:
            _first = jax.process_index() == 0
        except Exception:
            _first = True
        if _first:
            print(f"  [vq_interp] zeta_q.h5 holds {zx['nq']} full-BZ q but "
                  f"mf_header/kpoints/rk has only {qraw.shape[0]} (the WFN is "
                  f"symmetry-reduced).  q labels reconstructed as the BGW-"
                  f"wrapped C-order {tuple(int(v) for v in zx['kgrid'])} grid; "
                  f"run_gates' per-q makeVq-vs-disk check verifies it.",
                  flush=True)
        qraw = qfull
    zx["qfr_raw"] = qraw[: zx["nq"]]
    zx["qfr"] = zx["qfr_raw"] - np.round(zx["qfr_raw"])  # BGW-wrapped, pre half-fix
    kg = zx["kgrid"]
    zx["k_int"] = np.rint(zx["qfr_raw"] * kg[None, :]).astype(int) % kg[None, :]
    zx["k_lookup"] = {tuple(v): i for i, v in enumerate(zx["k_int"])}
    assert len(zx["k_lookup"]) == zx["nq"], "duplicate k labels in rk list"
    rx = np.arange(zx["nx"]) / zx["nx"]
    ry = np.arange(zx["ny"]) / zx["ny"]
    rz = np.arange(zx["nz"]) / zx["nz"]
    RX, RY, RZ = np.meshgrid(rx, ry, rz, indexing="ij")
    zx["rfrac"] = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)
    dims = np.array([zx["nx"], zx["ny"], zx["nz"]])
    zx["rmu_frac"] = rmu_idx / dims[None, :]     # centroid frac coords s_μ
    zx["rmu_flat"] = ((rmu_idx[:, 0] * zx["ny"]) + rmu_idx[:, 1]) * zx["nz"] \
        + rmu_idx[:, 2]
    zx["nv"] = int(ifmax.ravel()[0])
    assert np.all(ifmax == zx["nv"]), "ifmax not uniform over k"
    _fix_sphere_wrap(zx)
    # SCOPE, before anything expensive.  The stamp is scalar metadata read
    # with serial h5py (safe on every rank, no SlabIO handle), and it is the
    # deck's own record of which Coulomb kernel built ``V_qmunu`` — the one
    # fact this module cannot derive from geometry.  NOT re-announced here:
    # ``bse_io`` already prints ``describe_coulomb_policy_stamp`` once per
    # driver run, and the row-23 log shows that line sitting one screen above
    # the gate failures it explained.  The stamp was never missing; nothing
    # READ it.  So the fix is a refusal that quotes it, not a second copy.
    from file_io import read_coulomb_policy_from_h5
    zx["policy"] = read_coulomb_policy_from_h5(restart_file)
    # ``require_slab=False`` is the REFIT caller, and it is not a bypass of
    # the scope check — it is the observation that the scope check is about a
    # part of this module the refit never touches.  Both slab-only facts
    # (:func:`slab_scope_violations`) are properties of the b26p long-range
    # MODEL and of ``v_slab_on_set``; the refit fits ζ at the target Q and
    # contracts it with the kernel :func:`make_v_on_set` hands it, which on a
    # bulk deck is the producer's own.  The check therefore moved to
    # ``build_vq_evaluator`` — the model build — rather than being weakened,
    # so an interp/both run on a bulk deck refuses exactly as loudly as
    # before, one call later and before anything expensive still.
    if require_slab:
        assert_slab_scope(zx["bvec"], qfr=zx["qfr"], policy=zx["policy"],
                          source=restart_file)
    return zx



def check_dipole_provenance(
    path, *, wfn, nval, ncond, nband,
    bispinor=None, skip_vnl=None, vnl_mode=None, vnl_velocity_sign=None,
    wfn_fingerprint_binding=None,
    print_fn=print,
) -> bool:
    """Does ``path`` match the WFN, window, and requested operator convention?

    Returns True only when a stamp exists AND agrees.  Disagreement goes
    through ``common.sanity.warn`` (the same channel
    ``gw.head_correction`` uses for its coverage check) so a strict run
    turns it into a refusal and a permissive one still prints loudly.
    A MISSING stamp is reported as such and returns False — an
    unstamped file predates this guard and cannot be vouched for.
    """
    from psp.get_dipole_mtxels import (
        WFN_FINGERPRINT_SCHEME,
        _DIPOLE_Q0_OPERATOR_SCHEME,
        _PROV_ATTRS,
        _prov_ne,
        _prov_show,
        wfn_fingerprint,
    )
    from common import sanity
    from common.parallel_transport import fingerprint_from_binding

    try:
        with h5py.File(str(path), "r") as h5:
            attrs = {k: h5.attrs[k] for k in _PROV_ATTRS if k in h5.attrs}
            ncond_mismatch = (
                "prov_ncond" not in attrs
                or _prov_ne(attrs["prov_ncond"], int(ncond)))
            q0_ncond_ok, q0_ncond_detail = (False, "prov_ncond is absent")
            if ncond_mismatch and "prov_ncond" in attrs:
                q0_ncond_ok, q0_ncond_detail = _q0_ncond_coverage(
                    h5, wfn=wfn, ncond=ncond, nband=nband)
    except OSError as exc:
        print_fn(f"  [dipole provenance] cannot open {path} "
                 f"({type(exc).__name__}: {exc})")
        return False

    if "prov_wfn_sha256" not in attrs:
        print_fn(f"  [dipole provenance] {path} carries no provenance stamp "
                 f"(written before the guard existed).  Regenerate with "
                 f"`python -m psp.get_dipole_mtxels` to make it checkable.")
        return False

    got_scheme = attrs.get("prov_wfn_fingerprint_scheme")
    if isinstance(got_scheme, bytes):
        got_scheme = got_scheme.decode()
    fingerprint_checkable = got_scheme == WFN_FINGERPRINT_SCHEME
    if got_scheme is None:
        print_fn(
            "  [dipole provenance] the WFN fingerprint predates the "
            f"location-independent {WFN_FINGERPRINT_SCHEME!r} scheme and "
            "cannot be compared across checkouts; regenerate dipole.h5 with "
            "`python -m psp.get_dipole_mtxels` to make the WFN identity "
            "checkable.")
    elif not fingerprint_checkable:
        print_fn(
            "  [dipole provenance] the WFN fingerprint uses unsupported "
            f"scheme {got_scheme!r}, not {WFN_FINGERPRINT_SCHEME!r}; "
            "regenerate dipole.h5 with `python -m psp.get_dipole_mtxels` "
            "to make it checkable.")
    if not fingerprint_checkable:
        # This is an identity refusal, not evidence that any later field
        # differs.  Preserve that distinction: legacy-fingerprint tests and
        # users must not receive a fabricated DFT/window/operator mismatch
        # merely because a newly required field is also absent.
        return False

    # ``prov_nspinor`` rides the present-key filter below: a legacy file
    # that predates the stamp is accepted (same reading as every other
    # prov_* attr), while a STAMPED mismatch refuses — a dipole.h5 built
    # from an nspinor=1 WFN has the right shape for an nspinor=2 run of
    # the same crystal and vice versa (INVARIANTS row 3: representation).
    want = {"prov_nval": int(nval), "prov_ncond": int(ncond),
            "prov_nband": int(nband),
            "prov_q0_operator_scheme": _DIPOLE_Q0_OPERATOR_SCHEME}
    if fingerprint_checkable:
        want["prov_wfn_sha256"] = (
            wfn_fingerprint(wfn)
            if wfn_fingerprint_binding is None
            else fingerprint_from_binding(wfn_fingerprint_binding, wfn))
    optional = {
        "prov_bispinor": bispinor,
        "prov_skip_vnl": skip_vnl,
        "prov_vnl_mode": vnl_mode,
        "prov_vnl_velocity_sign": vnl_velocity_sign,
    }
    want.update({key: value for key, value in optional.items()
                 if value is not None})
    # An expected operator field that is absent is not a legacy default: it is
    # uncheckable provenance.  The caller choosing that convention must fail
    # closed instead of silently reading whichever operator made the file.
    bad = [(k, attrs.get(k, "<absent>"), v) for k, v in want.items()
           if (k != "prov_ncond" or not q0_ncond_ok)
           and (k not in attrs or _prov_ne(attrs[k], v))]
    # prov_nspinor: required-IF-PRESENT (the comment above this want dict
    # is the contract) — a legacy file that predates the stamp is accepted,
    # a stamped mismatch refuses.
    if "prov_nspinor" in attrs and _prov_ne(attrs["prov_nspinor"],
                                            int(wfn.nspinor)):
        bad.append(("prov_nspinor", attrs["prov_nspinor"],
                    int(wfn.nspinor)))
    if bad:
        detail = "; ".join(f"{k}: file={_prov_show(got)} run={_prov_show(exp)}"
                           for k, got, exp in bad)
        if ncond_mismatch and not q0_ncond_ok:
            detail += f"; q→0 coverage refusal: {q0_ncond_detail}"
        sanity.warn(
            f"{path} was generated from a DIFFERENT DFT solution, spin representation, band "
            f"window, or velocity/representation convention than this run "
            f"({detail}).  dipole.h5 has the right shape either way, so a "
            f"shape-only reader would not notice: the q→0 head S(ω), and "
            f"every Σ_SX/Σ_COH correction built from it, would be assembled "
            f"from incompatible velocity matrix elements.  "
            f"Regenerate it with `python -m psp.get_dipole_mtxels -i <deck>`.",
            print_fn=print_fn)
        return False

    if ncond_mismatch:
        print_fn(
            "  [dipole provenance] producer "
            f"ncond={int(np.asarray(attrs['prov_ncond']))} differs from run "
            f"ncond={int(ncond)}, accepted because {q0_ncond_detail}; the "
            "ordinary payload is the same full-square operator.")
    print_fn(
        f"  dipole.h5 provenance OK (WFN {want['prov_wfn_sha256'][:12]}…, "
        f"window nval={int(nval)} ncond={int(ncond)} nband={int(nband)}"
        + (f", bispinor={bool(bispinor)}" if bispinor is not None else "")
        + (f", vnl_velocity_sign={float(vnl_velocity_sign):+.1f}"
           if vnl_velocity_sign is not None else "")
        + ")")
    return True



def _q0_ncond_coverage(h5, *, wfn, ncond, nband) -> tuple[bool, str]:
    """Can an ``ncond``-mismatched file represent the identical q→0 matrix?"""
    from psp.get_dipole_mtxels import (
        _resolve_dipole_nb_written,
    )
    expected = _resolve_dipole_nb_written(
        wfn, ncond=int(ncond), nband=int(nband))
    if "finite_q" in h5:
        return False, (
            "finite_q/ is present and its stored conduction axis is sized by "
            "the producer's ncond")

    problems = []
    if "prov_nb_written" not in h5.attrs:
        problems.append("prov_nb_written is absent")
    else:
        got = int(np.asarray(h5.attrs["prov_nb_written"]))
        producer_expected = _resolve_dipole_nb_written(
            wfn,
            ncond=int(np.asarray(h5.attrs["prov_ncond"])),
            nband=int(np.asarray(h5.attrs.get("prov_nband", nband))),
        )
        if got != producer_expected:
            problems.append(
                f"prov_nb_written: file={got} producer-resolved="
                f"{producer_expected}")
        if got != expected:
            problems.append(
                f"prov_nb_written: file={got} run-resolved={expected}")

    shapes = {}
    for name, rank in (("dipole_cart", 4), ("deltaE", 3)):
        if name not in h5:
            problems.append(f"{name} is absent")
            continue
        shape = tuple(int(v) for v in h5[name].shape)
        shapes[name] = shape
        if len(shape) != rank or shape[-2:] != (expected, expected):
            problems.append(
                f"{name} shape={shape}, expected square band axes "
                f"({expected},{expected})")
    if (len(shapes.get("dipole_cart", ())) >= 2
            and len(shapes.get("deltaE", ())) >= 1
            and shapes["dipole_cart"][1] != shapes["deltaE"][0]):
        problems.append(
            "dipole_cart and deltaE carry different k extents "
            f"({shapes['dipole_cart'][1]} versus {shapes['deltaE'][0]})")

    return not problems, ("; ".join(problems) if problems
                          else f"identical q→0 extent {expected}")


def read_bgw_eqp(eqp_file: str):
	"""Read a BerkeleyGW ``eqp{0,1}.dat`` — the inverse of :func:`write_bgw_eqp`.

	Returns ``(kpts_irr (nk, 3), e_dft (nk, nb), e_qp (nk, nb),
	band_offset)``; the energies are eV and the k-points are the CRYSTAL
	COORDINATES the block headers carry, on the irreducible wedge the
	writer emitted.

	``band_offset`` is the 0-BASED ABSOLUTE index of column 0, recovered
	from the file's own 1-based ``iband`` labels — the inverse of
	:func:`write_bgw_eqp`'s ``band_offset``.  It is returned rather than
	dropped because a consumer slicing an absolute band window
	(``bandstructure.htransform``) cannot place these columns without it,
	and the previous reader discarded it, which is why that consumer grew
	a second eqp parser instead of using this one.

	The canonical parser is shared by every GW/BSE band consumer.

	Ragged band counts are tolerated (short blocks are NaN-padded to the
	widest), because a caller slicing a band window must be able to see
	which states are absent rather than read a zero.
	"""
	kpts: list[list[float]] = []
	e_dft_blocks: list[list[float]] = []
	e_qp_blocks: list[list[float]] = []
	first_band: int | None = None

	with open(eqp_file) as f:
		while True:
			header = f.readline()
			if not header:
				break
			stripped = header.strip()
			if not stripped:
				break
			if stripped.startswith("#"):
				continue
			parts = header.split()
			if len(parts) < 4:
				break
			kpts.append([float(parts[0]), float(parts[1]), float(parts[2])])
			n_bands = int(parts[3])

			e_dft_k, e_qp_k = [], []
			for _ in range(n_bands):
				cols = f.readline().split()
				# (ispin, iband, E_DFT, E_QP); iband is 1-based ABSOLUTE.
				if first_band is None:
					first_band = int(cols[1])
				e_dft_k.append(float(cols[2]))
				e_qp_k.append(float(cols[3]))
			e_dft_blocks.append(e_dft_k)
			e_qp_blocks.append(e_qp_k)

	if not kpts:
		raise ValueError(
			f"no k-point blocks parsed from {os.path.basename(eqp_file)} — "
			f"expected the BerkeleyGW eqp layout this module writes "
			f"(a '(3f13.9,i8)' k header, then that many '(2i8,2f15.9)' rows)")

	max_band = max(len(b) for b in e_dft_blocks)
	n_kpts = len(kpts)
	e_dft = np.full((n_kpts, max_band), np.nan)
	e_qp = np.full((n_kpts, max_band), np.nan)
	for i in range(n_kpts):
		nb = len(e_dft_blocks[i])
		e_dft[i, :nb] = e_dft_blocks[i]
		e_qp[i, :nb] = e_qp_blocks[i]
	return np.array(kpts), e_dft, e_qp, int(first_band) - 1



def read_eqp_energies(eqp_file: str, sym, band_window: tuple[int, int]) -> jax.Array:
    """Full-BZ QP energies from the IRREDUCIBLE-WEDGE ``eqp{0,1}.dat``.

    Reads the BerkeleyGW-columnar eqp file LORRAX's GW writes — one block
    per ``wfn.kpoints`` entry, the crystal coordinate in the block header,
    energies in eV — and returns ``(nb, nk_full)`` in RYDBERG over
    ``band_window``, which is what this module's DFT path returns.

    THE UNFOLD IS THE SERVICE'S, NOT THIS MODULE'S.  Every IBZ→full-BZ
    map in the tree goes through ``symmetry_maps.star_broadcast``, reached
    here by :func:`symmetry_maps.unfold_file_wedge_to_full_bz` — the FILE
    wedge, ``wfn.kpoints``, which is what ``eqp1.dat`` is indexed by and
    what BerkeleyGW means by the IBZ.  It shares its backend with the
    kin_ion read path; no index table crosses into this module.

    WHAT THIS REPLACED, AND WHY.  It used to require a PRE-UNFOLDED
    full-BZ text file (``nk == sym.nk_tot``, refused otherwise) whose
    ``k-point N:`` blocks it paired to full-BZ k BY POSITION, with no
    coordinate ever read.  Nothing in the tree wrote that file: it came
    from an out-of-tree ``make_eqp_htformat.py`` that joined
    ``eqp_g0w0.dat`` against ``eqp1.dat`` to do the unfold by hand.  That
    is bespoke unfolding one hop upstream, plus a positional pairing that
    a re-ordered file passes silently.  Now the wedge file is read
    directly and the service does the unfold, so the converter has no job
    left and the position never enters.

    The block coordinates are CHECKED against the deck's own wedge
    (``sym.unfolded_kpts[sym.kirr_fullids]``) rather than trusted — a
    file from another deck, or in another order, is refused here instead
    of producing a quasiparticle bandstructure with the energies on the
    wrong k.
    """
    start, end = int(band_window[0]), int(band_window[1])
    nb = int(max(0, end - start))
    if nb == 0:
        raise ValueError("Empty band window requested for EQP override")

    from symmetry_maps import unfold_file_wedge_to_full_bz

    kpts_file, _e_dft_ev, e_qp_ev, band_offset = read_bgw_eqp(eqp_file)
    nk_file, nb_file = e_qp_ev.shape

    # ---- the file must be THIS deck's wedge, in the wedge's order -------
    kirr = np.asarray(sym.unfolded_kpts, dtype=np.float64)[
        np.asarray(sym.kirr_fullids, dtype=np.int64)]
    if nk_file != kirr.shape[0]:
        raise ValueError(
            f"{os.path.basename(eqp_file)} holds {nk_file} k-blocks but this "
            f"deck's irreducible wedge has {kirr.shape[0]} (full BZ "
            f"{int(sym.nk_tot)}).  This reader takes the wedge file LORRAX's "
            f"GW writes; the pre-unfolded full-BZ form is no longer read.")
    # Written by ``%13.9f``, so equality is to that many places; the
    # comparison is modulo a lattice vector because either side may carry
    # a k in a different periodic image.
    dk = np.asarray(kpts_file, dtype=np.float64) - kirr
    dk -= np.rint(dk)
    worst = float(np.max(np.abs(dk))) if dk.size else 0.0
    if worst > 1e-6:
        bad = int(np.argmax(np.max(np.abs(dk), axis=1)))
        raise ValueError(
            f"{os.path.basename(eqp_file)} block {bad} is at "
            f"{np.asarray(kpts_file)[bad].tolist()} but this deck's wedge "
            f"point {bad} is {kirr[bad].tolist()} (worst |Δk| = {worst:.2e} "
            f"over {nk_file} blocks).  The eqp file does not belong to this "
            f"wavefunction, or its k-order differs — either way its energies "
            f"would land on the wrong k.")

    # ---- absolute band window -> the file's columns ---------------------
    lo, hi = start - band_offset, end - band_offset
    if lo < 0 or hi > nb_file:
        raise ValueError(
            f"{os.path.basename(eqp_file)} covers absolute bands "
            f"[{band_offset}, {band_offset + nb_file}) but the requested "
            f"window is [{start}, {end}) — the eqp file does not span the "
            f"htransform sigma window.")
    window_ev = np.asarray(e_qp_ev)[:, lo:hi]
    if np.isnan(window_ev).any():
        n_missing = int(np.isnan(window_ev).sum())
        raise ValueError(
            f"{os.path.basename(eqp_file)} is missing {n_missing} of "
            f"{window_ev.size} (k, band) entries inside the requested window "
            f"[{start}, {end}) — a short block cannot be silently padded.")

    # ---- THE unfold: wedge -> full BZ, through the service --------------
    # The FILE wedge: ``eqp1.dat`` is indexed by ``wfn.kpoints``, which is a
    # different (and on two of three committed decks a different-LENGTH)
    # k-set from the star wedge — see the register.  The call site says
    # which without the reader needing to know what ``trs_reference`` is.
    full_ev = np.asarray(unfold_file_wedge_to_full_bz(sym, window_ev))
    if full_ev.shape[0] != int(sym.nk_tot):
        raise ValueError(
            f"star_broadcast returned {full_ev.shape[0]} k-points, expected "
            f"{int(sym.nk_tot)}")

    # eV on disk (BGW convention); Ry is this module's internal unit.
    return jnp.asarray(full_ev.T / RYD_TO_EV, dtype=jnp.float64)



def apply_eqp_corrections(
    enk_full: np.ndarray,
    eqp_file: str,
    input_file: str,
    ry_to_ev: float = 13.6056980659,
    *,
    state_artifact_path: str | None = None,
) -> np.ndarray:
    """Apply BGW ``eqp{0,1}.dat`` corrections to full-BZ DFT eigenvalues.

    ``eqp_file`` is on the IRREDUCIBLE WEDGE (one block per
    ``wfn.kpoints``, coordinates in the block header) and ``enk_full`` is
    on the full BZ, so this is an UNFOLD — and it goes through the
    symmetry service, like every other unfold in the tree.

    ``input_file`` is REQUIRED.  It used to be optional, and passing
    ``None`` selected a second implementation that matched each full-BZ k
    to a wedge block by comparing MEAN-FIELD ENERGIES to 0.01 eV.  That
    was bespoke unfolding: it happened to be right because E_DFT is
    constant over a symmetry star, but two accidentally-degenerate stars
    alias, and the star it picks is then simply the wrong one — silently,
    with QP energies from another k inside what the caller believes is a
    quasiparticle calculation.  The heuristic existed only because the
    call site believed LORRAX wrote ``eqp1.dat`` on the full BZ and that
    the symmetry-map branch would therefore refuse; it does not and it
    does not (``gw_output.py`` subsets through ``kirr_to_kfull``).  Both
    the belief and the second implementation are gone: there is one path,
    and it asks the service.
    """
    from bse.bse_window import _parse_wfn_path
    if not input_file:
        raise ValueError(
            "apply_eqp_corrections requires input_file: the eqp file is on "
            "the irreducible wedge and the unfold to the full BZ needs this "
            "deck's symmetry tables.  It used to be optional, and omitting "
            "it selected a mean-field-energy nearest-match that silently "
            "took QP shifts from the wrong star whenever two stars were "
            "degenerate; that path is deleted rather than defaulted.")

    # Every public BSE eqp frontend (bse_jax, Haydock, Davidson, the direct
    # restart loader and exciton bands) reaches this one correction owner.
    # Refuse a second DFT-labelled ladder on a positively stamped QP WFN here
    # rather than relying on each CLI to remember the same content contract.
    from file_io.qp_wfn import refuse_conflicting_qp_state_sources
    refuse_conflicting_qp_state_sources(
        wfn_path=_parse_wfn_path(input_file), eqp_file=eqp_file,
        state_artifact_path=state_artifact_path,
        where="BSE diagonal-eqp state")

    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import unfold_file_wedge_to_full_bz
    from wfn_loader import WfnLoader

    # 3-tuple: this consumer's band axis is already LOCAL to the deck's
    # ``b0`` (``enk_full`` spans [b0, b4) and the eqp window starts at b0),
    # so column 0 lines up by construction and the file's absolute band
    # offset is not needed here.
    _kpts_ibz, _e_dft_ibz, e_qp_ibz, _band_offset = read_bgw_eqp(eqp_file)
    nk_ibz, nb_eqp = e_qp_ibz.shape
    nk_full, nb_full = enk_full.shape

    wfn = WfnLoader(_parse_wfn_path(input_file))
    sym = wfn.symmetry()
    if sym.nk_tot != nk_full:
        raise ValueError(
            f"apply_eqp_corrections: enk_full has {nk_full} k-points but the "
            f"deck's symmetry maps describe {sym.nk_tot} — the eigenvalues "
            f"and the WFN in {os.path.basename(input_file)} are not the same "
            f"k-grid.")
    if nk_ibz != sym.nk_red:
        raise ValueError(
            f"apply_eqp_corrections: {os.path.basename(eqp_file)} holds "
            f"{nk_ibz} k-blocks but this deck's irreducible wedge has "
            f"{sym.nk_red}.  This reader expects the wedge file LORRAX's GW "
            f"writes; a full-BZ file ({sym.nk_tot} blocks) is the pre-unfolded "
            f"form that no longer exists.")

    # THE UNFOLD, named for the wedge it is on.  ``eqp1.dat`` is indexed by
    # ``wfn.kpoints`` — the FILE wedge — which on two of the three committed
    # decks is a different LENGTH from the star wedge, so the distinction is
    # not cosmetic.  One backend under both named ops; no index table
    # crosses into this module.
    e_qp_full_ev = unfold_file_wedge_to_full_bz(sym, e_qp_ibz)

    enk_qp = enk_full.copy()
    n_take = min(nb_eqp, nb_full)
    block = np.asarray(e_qp_full_ev)[:, :n_take]
    have = ~np.isnan(block)
    # A band absent from the eqp file keeps its mean-field value; a band
    # present replaces it.  Ry in, Ry out.
    enk_qp[:, :n_take] = np.where(have, block / ry_to_ev, enk_qp[:, :n_take])
    return enk_qp



def read_kin_ion_full_bz(filename):
    """Return the canonical full-zone kinetic plus ionic Hamiltonian."""
    return read_full_bz_dataset(filename, "kin_ion")


def _read_isdf_group(f: h5py.File) -> IsdfHeader:
    from file_io.isdf_header import IsdfHeader, _GROUP
    g = f[_GROUP]
    # Legacy files predate the ``zeta_is_done`` field; treat as ``True``
    # (they were always written atomically at end-of-fit).  Legacy files
    # also predate ``zeta_layout``; treat as ``'r_space'``.  New files
    # carry both fields explicitly.
    zeta_done = (bool(g['zeta_is_done'][()]) if 'zeta_is_done' in g
                 else True)
    zeta_layout = (_decode_isdf_str(g['zeta_layout'][()]) if 'zeta_layout' in g
                   else 'r_space')
    # G-flat metadata (only present when zeta_layout == 'G_flat').
    gv = (np.asarray(g['gvec_components'][:], dtype=np.int32)
          if 'gvec_components' in g else None)
    nk = (np.asarray(g['ngk'][:], dtype=np.int32)
          if 'ngk' in g else None)
    cutoff = (float(g['zeta_cutoff_ry'][()])
              if 'zeta_cutoff_ry' in g else None)
    prov = (_decode_isdf_str(g['fit_provenance'][()])
            if 'fit_provenance' in g else None)
    return IsdfHeader(
        density=_decode_isdf_str(g['density'][()]),
        vertex_mu_L=int(g['vertex_mu_L'][()]),
        r_mu_fft_idx=np.asarray(g['centroids/r_mu_fft_idx'][:], dtype=np.int32),
        r_mu_crystal=np.asarray(g['centroids/r_mu_crystal'][:], dtype=np.float64),
        zeta_is_done=zeta_done,
        zeta_layout=zeta_layout,
        gvec_components=gv,
        ngk_per_q=nk,
        zeta_cutoff_ry=cutoff,
        fit_provenance=prov,
    )


def _decode_isdf_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode('utf-8')
    return str(v)


def read_isdf_header(path: str | Path) -> IsdfHeader:
    """Open ``path`` and return its ``isdf_header`` group."""
    with h5py.File(str(path), 'r') as f:
        return _read_isdf_group(f)


def read_isdf_header_from_file(f: h5py.File) -> IsdfHeader:
    """Same as :func:`read_isdf_header` but operates on an open handle."""
    return _read_isdf_group(f)
