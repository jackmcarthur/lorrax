"""Authenticate the packed-photon body/restart artifact composition.

The record here defines no new physical identity.  It composes the existing
wavefunction-basis receipts, exact zeta fit-provenance strings, canonical
Coulomb policy, and V-format word.  Fresh GW stamps the same small record at
the V-tile and canonical-restart gateways; restart compares them before
opening either distributed payload.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .wfn_basis import WavefunctionBasisReceipt, centroid_table_md5

BISPINOR_VQ_RESTART_BINDING_DATASET = "bispinor_vq_restart_binding"
BISPINOR_VQ_RESTART_BINDING_SCHEMA = "bispinor_vq_restart_binding_v1"


@dataclass(frozen=True)
class BispinorVqRestartBinding:
    """Small host-only composition record shared by V and restart files."""

    v_qmunu_format: str
    zeta_fit_provenance: tuple[str, str, str, str]
    charge_basis_source: dict
    transverse_basis_source: dict
    coulomb_policy: str

    @classmethod
    def from_sources(
        cls,
        *,
        v_qmunu_format: str,
        zeta_fit_provenance: Sequence[str],
        charge_basis_receipt: WavefunctionBasisReceipt,
        transverse_basis_receipt: WavefunctionBasisReceipt,
        coulomb_policy: str,
    ) -> BispinorVqRestartBinding:
        """Compose incumbent identities without rereading source payloads."""
        if not isinstance(charge_basis_receipt, WavefunctionBasisReceipt):
            raise TypeError(
                "bispinor V/restart binding requires a charge "
                "WavefunctionBasisReceipt")
        if not isinstance(transverse_basis_receipt, WavefunctionBasisReceipt):
            raise TypeError(
                "bispinor V/restart binding requires a transverse "
                "WavefunctionBasisReceipt")
        charge = charge_basis_receipt.physical_source_record()
        transverse = transverse_basis_receipt.physical_source_record()
        if charge["role"] != "charge" or transverse["role"] != "transverse":
            raise ValueError(
                "bispinor V/restart binding requires ordered charge and "
                "transverse basis receipts")
        return cls._from_parts(
            v_qmunu_format=v_qmunu_format,
            zeta_fit_provenance=zeta_fit_provenance,
            charge_basis_source=charge,
            transverse_basis_source=transverse,
            coulomb_policy=coulomb_policy,
        )

    @classmethod
    def _from_parts(
        cls,
        *,
        v_qmunu_format,
        zeta_fit_provenance,
        charge_basis_source,
        transverse_basis_source,
        coulomb_policy,
    ) -> BispinorVqRestartBinding:
        fmt = str(v_qmunu_format)
        provenance = tuple(str(value) for value in zeta_fit_provenance)
        if not fmt:
            raise ValueError("bispinor V/restart binding has an empty V format")
        if len(provenance) != 4 or any(not value for value in provenance):
            raise ValueError(
                "bispinor V/restart binding requires four non-empty exact "
                "zeta fit-provenance strings in C,T1,T2,T3 order")
        if not isinstance(charge_basis_source, Mapping):
            raise TypeError("bispinor V/restart charge source is not a record")
        if not isinstance(transverse_basis_source, Mapping):
            raise TypeError(
                "bispinor V/restart transverse source is not a record")
        charge = dict(charge_basis_source)
        transverse = dict(transverse_basis_source)
        if charge.get("role") != "charge" or transverse.get("role") != "transverse":
            raise ValueError(
                "bispinor V/restart basis records have the wrong channel roles")
        policy = str(coulomb_policy)
        if not policy:
            raise ValueError(
                "bispinor V/restart binding has an empty Coulomb-policy stamp")
        return cls(
            v_qmunu_format=fmt,
            zeta_fit_provenance=provenance,
            charge_basis_source=charge,
            transverse_basis_source=transverse,
            coulomb_policy=policy,
        )

    def to_record(self) -> dict:
        return {
            "schema": BISPINOR_VQ_RESTART_BINDING_SCHEMA,
            "v_qmunu_format": self.v_qmunu_format,
            "zeta_fit_provenance": list(self.zeta_fit_provenance),
            "charge_basis_source": dict(self.charge_basis_source),
            "transverse_basis_source": dict(self.transverse_basis_source),
            "coulomb_policy": self.coulomb_policy,
        }

    @classmethod
    def from_record(cls, record) -> BispinorVqRestartBinding:
        if not isinstance(record, Mapping):
            raise TypeError("bispinor V/restart binding payload is not a record")
        required = {
            "schema", "v_qmunu_format", "zeta_fit_provenance",
            "charge_basis_source", "transverse_basis_source", "coulomb_policy",
        }
        if record.get("schema") != BISPINOR_VQ_RESTART_BINDING_SCHEMA:
            raise ValueError(
                "bispinor V/restart binding has unsupported schema "
                f"{record.get('schema')!r}; expected "
                f"{BISPINOR_VQ_RESTART_BINDING_SCHEMA!r}")
        if set(record) != required:
            raise ValueError(
                "bispinor V/restart binding fields differ from its schema: "
                f"got {sorted(record)}, expected {sorted(required)}")
        return cls._from_parts(
            v_qmunu_format=record["v_qmunu_format"],
            zeta_fit_provenance=record["zeta_fit_provenance"],
            charge_basis_source=record["charge_basis_source"],
            transverse_basis_source=record["transverse_basis_source"],
            coulomb_policy=record["coulomb_policy"],
        )

    def encode(self) -> np.ndarray:
        """Return canonical scalar bytes for either metadata gateway."""
        text = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))
        return np.asarray(text.encode("utf-8"), dtype="S")

    def assert_same_composition(
        self, other: BispinorVqRestartBinding, *, where: str,
    ) -> None:
        if not isinstance(other, BispinorVqRestartBinding):
            raise TypeError(
                f"{where}: expected BispinorVqRestartBinding, got "
                f"{type(other).__name__}")
        mine = self.to_record()
        theirs = other.to_record()
        differing = [name for name in mine if mine[name] != theirs[name]]
        if differing:
            raise ValueError(
                "GATE bispinor_packed_restart_binding_mismatch: packed "
                "restart artifacts name different physical sources.\n"
                f"  got:  differing binding fields {differing} at {where}\n"
                "  want: isdf_tensors and v_q_bispinor.h5 written together\n"
                "  why:  same-shaped charge/current tensors from different "
                "WFN, zeta, centroid, or Coulomb sources are silently wrong\n"
                "  fix:  set restart = false and regenerate both artifacts\n"
                "  doc:  docs/input_reference.md, restart.")


def read_bispinor_vq_restart_binding(
    path: str | Path,
) -> BispinorVqRestartBinding | None:
    """Read one small host record; ``None`` means missing/legacy artifact."""
    path = Path(path)
    try:
        with h5py.File(path, "r") as h5:
            if BISPINOR_VQ_RESTART_BINDING_DATASET not in h5:
                return None
            raw = h5[BISPINOR_VQ_RESTART_BINDING_DATASET][()]
    except OSError:
        return None
    try:
        payload = raw if isinstance(raw, bytes) else np.asarray(raw).tobytes()
        record = json.loads(payload.decode("utf-8", "strict").rstrip("\x00"))
        return BispinorVqRestartBinding.from_record(record)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(
            "GATE bispinor_packed_restart_binding_invalid: packed restart "
            "binding metadata is not canonical.\n"
            f"  got:  invalid {BISPINOR_VQ_RESTART_BINDING_DATASET} in {path}\n"
            "  want: a canonical binding written by the current restart writer\n"
            "  why:  corrupt provenance cannot authenticate distributed arrays\n"
            "  fix:  set restart = false and regenerate the artifacts\n"
            "  doc:  docs/input_reference.md, restart.") from exc


def assert_bispinor_vq_restart_binding(
    *, restart_path: str | Path, v_q_path: str | Path,
    where: str = "bispinor restart",
) -> BispinorVqRestartBinding:
    """Refuse a stale/legacy cross-file join before distributed payload I/O."""
    restart = read_bispinor_vq_restart_binding(restart_path)
    photon = read_bispinor_vq_restart_binding(v_q_path)
    missing = []
    if restart is None:
        missing.append(str(restart_path))
    if photon is None:
        missing.append(str(v_q_path))
    if missing:
        raise ValueError(
            "GATE bispinor_packed_restart_binding_missing: packed restart "
            "artifact provenance is missing.\n"
            f"  got:  no {BISPINOR_VQ_RESTART_BINDING_DATASET} in {missing}\n"
            "  want: authenticated isdf_tensors and v_q_bispinor.h5 from one run\n"
            "  why:  a legacy, absent, or partial tile file cannot be joined "
            "safely to the restart carrier\n"
            "  fix:  restore both matching files or set restart = false\n"
            "  doc:  docs/input_reference.md, restart.")
    restart.assert_same_composition(photon, where=where)
    return restart


def authenticate_or_recover_bispinor_vq_restart_binding(
    *,
    restart_path: str | Path,
    v_q_path: str | Path,
    zeta_paths: Sequence[str | Path],
    charge_basis_receipt: WavefunctionBasisReceipt,
    transverse_basis_receipt: WavefunctionBasisReceipt,
    coulomb_policy: str,
    expected_kgrid: Sequence[int],
    expected_v_qmunu_format: str,
) -> tuple[BispinorVqRestartBinding, bool]:
    """Authenticate a current binding or recover a complete pre-schema one.

    Recovery is deliberately narrower than the ordinary reader.  It is
    available only when *both* artifacts predate the binding schema, and only
    when the restart carrier, V-tile metadata, and four complete zeta headers
    reproduce every physical source used by :meth:`from_sources`.  No file is
    modified.  The returned boolean says that the binding was reconstructed
    in memory from those authenticated pre-schema records.

    Parameters
    ----------
    restart_path, v_q_path
        The paired ISDF restart carrier and packed-photon tile file.
    zeta_paths
        Charge followed by the three Cartesian-current zeta files.
    charge_basis_receipt, transverse_basis_receipt
        Canonical receipts constructed from the selected WFN and centroid
        tables after the restart WFN fingerprint has been authenticated.
    coulomb_policy
        Canonical formatted policy of the running deck.
    expected_kgrid
        Full q-grid dimensions used by the tile and zeta files.
    expected_v_qmunu_format
        Current packed V-tile format word.

    Returns
    -------
    binding : BispinorVqRestartBinding
        Authenticated persisted binding or its in-memory pre-schema recovery.
    recovered : bool
        ``True`` only for the complete pre-schema recovery path.
    """
    restart_path = Path(restart_path)
    v_q_path = Path(v_q_path)
    restart = read_bispinor_vq_restart_binding(restart_path)
    photon = read_bispinor_vq_restart_binding(v_q_path)

    # A partially upgraded pair is not a legacy pair.  Reuse the canonical
    # missing-binding refusal rather than treating one record as disposable.
    if (restart is None) != (photon is None):
        assert_bispinor_vq_restart_binding(
            restart_path=restart_path, v_q_path=v_q_path,
            where="bispinor pre-schema recovery")

    if restart is not None:
        restart.assert_same_composition(
            photon, where="bispinor restart artifacts")
        running = BispinorVqRestartBinding.from_sources(
            v_qmunu_format=expected_v_qmunu_format,
            zeta_fit_provenance=restart.zeta_fit_provenance,
            charge_basis_receipt=charge_basis_receipt,
            transverse_basis_receipt=transverse_basis_receipt,
            coulomb_policy=coulomb_policy,
        )
        restart.assert_same_composition(
            running, where="gw_jax running bispinor sources")
        return restart, False

    zeta_paths = tuple(zeta_paths)
    if len(zeta_paths) != 4:
        raise ValueError(
            "GATE bispinor_pre_schema_restart_provenance_missing: legacy "
            "packed restart recovery is incomplete.\n"
            f"  got:  {len(zeta_paths)} zeta paths\n"
            "  want: charge plus three current zeta files\n"
            "  why:  all four exact fit-provenance records are required to "
            "reconstruct the packed source composition\n"
            "  fix:  restore the four zeta files or regenerate the restart\n"
            "  doc:  docs/input_reference.md, restart.")

    from file_io.qp_wfn import read_qp_state_source_provenance
    from file_io.tagged_arrays import (
        format_coulomb_policy,
        read_coulomb_policy_from_h5,
    )
    from zeta_loader import ZetaLoader

    def _missing(fact: str, path: Path) -> ValueError:
        return ValueError(
            "GATE bispinor_pre_schema_restart_provenance_missing: legacy "
            "packed restart recovery is incomplete.\n"
            f"  got:  missing {fact} in {path}\n"
            "  want: the authenticated WFN, centroid, zeta, Coulomb, and "
            "tile stamps written by the historical run\n"
            "  why:  an inferred binding without every source stamp could "
            "join same-shaped tensors from different calculations\n"
            "  fix:  restore the original artifacts or regenerate the restart\n"
            "  doc:  docs/input_reference.md, restart.")

    def _mismatch(fact: str, got, want, path: Path) -> ValueError:
        return ValueError(
            "GATE bispinor_pre_schema_restart_provenance_mismatch: legacy "
            "packed restart sources disagree.\n"
            f"  got:  {fact}={got!r} in {path}\n"
            f"  want: {want!r}\n"
            "  why:  pre-schema recovery is valid only for the exact WFN, "
            "centroid, zeta, Coulomb, and tile composition\n"
            "  fix:  restore one matched artifact family or regenerate it\n"
            "  doc:  docs/input_reference.md, restart.")

    charge = charge_basis_receipt.physical_source_record()
    transverse = transverse_basis_receipt.physical_source_record()
    source_record = read_qp_state_source_provenance(restart_path)
    if source_record is None:
        raise _missing("qp_state_source_provenance", restart_path)
    for name in ("wfn_fingerprint_scheme", "wfn_fingerprint"):
        want = charge[name]
        got = source_record.get(name)
        if got != want or transverse[name] != want:
            raise _mismatch(name, got, want, restart_path)

    try:
        with h5py.File(restart_path, "r") as h5:
            for attr, source in (
                ("centroids_charge_md5", charge),
                ("centroids_transverse_md5", transverse),
            ):
                if attr not in h5.attrs:
                    raise _missing(attr, restart_path)
                got = str(h5.attrs[attr])
                want = source["centroid_table_md5"]
                if got != want:
                    raise _mismatch(attr, got, want, restart_path)
    except OSError as exc:
        raise _missing("readable restart carrier", restart_path) from exc

    stamped_policy = read_coulomb_policy_from_h5(restart_path)
    if stamped_policy is None:
        raise _missing("coulomb_policy", restart_path)
    got_policy = format_coulomb_policy(stamped_policy)
    if got_policy != str(coulomb_policy):
        raise _mismatch(
            "coulomb_policy", got_policy, str(coulomb_policy), restart_path)

    kgrid = tuple(int(value) for value in expected_kgrid)
    n_q = int(np.prod(kgrid))
    try:
        with h5py.File(v_q_path, "r") as h5:
            required = ("v_qmunu_format", "kgrid", "n_rmu_C", "n_rmu_T",
                        "n_q_total")
            for name in required:
                if name not in h5:
                    raise _missing(name, v_q_path)

            raw_format = h5["v_qmunu_format"][()]
            tile_format = (
                raw_format.decode("utf-8", "strict")
                if isinstance(raw_format, bytes) else str(raw_format))
            tile_facts = {
                "v_qmunu_format": tile_format,
                "kgrid": tuple(int(value) for value in h5["kgrid"][...]),
                "n_rmu_C": int(h5["n_rmu_C"][()]),
                "n_rmu_T": int(h5["n_rmu_T"][()]),
                "n_q_total": int(h5["n_q_total"][()]),
            }
    except OSError as exc:
        raise _missing("readable V-tile metadata", v_q_path) from exc
    wanted_tile_facts = {
        "v_qmunu_format": str(expected_v_qmunu_format),
        "kgrid": kgrid,
        "n_rmu_C": int(charge["n_rmu_logical"]),
        "n_rmu_T": int(transverse["n_rmu_logical"]),
        "n_q_total": n_q,
    }
    for name, want in wanted_tile_facts.items():
        got = tile_facts[name]
        if got != want:
            raise _mismatch(name, got, want, v_q_path)

    provenance = []
    expected_sources = (charge, transverse, transverse, transverse)
    for channel, (path, source) in enumerate(zip(zeta_paths, expected_sources)):
        path = Path(path)
        try:
            with ZetaLoader(path) as zeta:
                if zeta.fit_provenance is None:
                    raise _missing("isdf_header/fit_provenance", path)
                if int(zeta.vertex_mu_L) != channel:
                    raise _mismatch(
                        "vertex_mu_L", int(zeta.vertex_mu_L), channel, path)
                if tuple(int(value) for value in zeta.kgrid) != kgrid:
                    raise _mismatch("kgrid", tuple(zeta.kgrid), kgrid, path)
                if zeta.q_layout != "full_bz" or int(zeta.n_q_on_disk) != n_q:
                    raise _mismatch(
                        "q_layout/n_q_on_disk",
                        (zeta.q_layout, int(zeta.n_q_on_disk)),
                        ("full_bz", n_q), path)
                got_md5 = centroid_table_md5(zeta.r_mu_fft_idx)
                want_md5 = source["centroid_table_md5"]
                if got_md5 != want_md5:
                    raise _mismatch(
                        "zeta centroid table", got_md5, want_md5, path)
                if int(zeta.n_rmu) != int(source["n_rmu_logical"]):
                    raise _mismatch(
                        "zeta n_rmu", int(zeta.n_rmu),
                        int(source["n_rmu_logical"]), path)
                provenance.append(str(zeta.fit_provenance))
        except OSError as exc:
            raise _missing("readable complete zeta file", path) from exc

    recovered = BispinorVqRestartBinding.from_sources(
        v_qmunu_format=expected_v_qmunu_format,
        zeta_fit_provenance=provenance,
        charge_basis_receipt=charge_basis_receipt,
        transverse_basis_receipt=transverse_basis_receipt,
        coulomb_policy=coulomb_policy,
    )
    return recovered, True


__all__ = [
    "BISPINOR_VQ_RESTART_BINDING_DATASET",
    "BISPINOR_VQ_RESTART_BINDING_SCHEMA",
    "BispinorVqRestartBinding",
    "authenticate_or_recover_bispinor_vq_restart_binding",
    "assert_bispinor_vq_restart_binding",
    "read_bispinor_vq_restart_binding",
]
