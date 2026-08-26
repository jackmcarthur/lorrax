"""Authenticate the bispinor photon-body/restart artifact composition.

The record here does not define any new physical identity.  It composes the
incumbent wavefunction-basis receipts, the exact zeta fit-provenance strings,
the canonical serialized Coulomb policy, and the V-format word.  Fresh GW
stamps the same record at the two artifact gateways; restart compares those
small host records before opening either distributed payload.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from .wfn_basis import WavefunctionBasisReceipt


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
    ) -> "BispinorVqRestartBinding":
        """Compose existing identities without reading or hashing sources."""
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
    ) -> "BispinorVqRestartBinding":
        fmt = str(v_qmunu_format)
        if not fmt:
            raise ValueError("bispinor V/restart binding has an empty V format")
        raw_provenance = tuple(zeta_fit_provenance)
        if len(raw_provenance) != 4 or any(
                value is None or not str(value) for value in raw_provenance):
            raise ValueError(
                "bispinor V/restart binding requires four non-empty exact "
                "zeta fit-provenance strings in C,T1,T2,T3 order")
        provenance = tuple(str(value) for value in raw_provenance)
        if not isinstance(charge_basis_source, Mapping):
            raise ValueError(
                "bispinor V/restart binding charge basis source is not a record")
        if not isinstance(transverse_basis_source, Mapping):
            raise ValueError(
                "bispinor V/restart binding transverse basis source is not a record")
        charge = dict(charge_basis_source)
        transverse = dict(transverse_basis_source)
        if charge.get("role") != "charge":
            raise ValueError(
                "bispinor V/restart binding charge source has the wrong role")
        if transverse.get("role") != "transverse":
            raise ValueError(
                "bispinor V/restart binding transverse source has the wrong role")
        if coulomb_policy is None or not str(coulomb_policy):
            raise ValueError(
                "bispinor V/restart binding has an empty Coulomb-policy stamp")
        policy = str(coulomb_policy)
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
    def from_record(cls, record) -> "BispinorVqRestartBinding":
        if not isinstance(record, Mapping):
            raise ValueError("bispinor V/restart binding payload is not a record")
        if record.get("schema") != BISPINOR_VQ_RESTART_BINDING_SCHEMA:
            raise ValueError(
                "bispinor V/restart binding has unsupported schema "
                f"{record.get('schema')!r}; expected "
                f"{BISPINOR_VQ_RESTART_BINDING_SCHEMA!r}")
        required = {
            "schema", "v_qmunu_format", "zeta_fit_provenance",
            "charge_basis_source", "transverse_basis_source",
            "coulomb_policy",
        }
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
        """Return canonical scalar bytes for either existing metadata path."""
        text = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))
        return np.asarray(text.encode("utf-8"), dtype="S")

    def assert_same_composition(
        self, other: "BispinorVqRestartBinding", *, where: str,
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
                f"{where}: bispinor V and restart artifacts have different "
                f"physical-source composition fields {differing}. Regenerate "
                "both artifacts together with restart = false.")


def read_bispinor_vq_restart_binding(
    path: str | Path,
) -> BispinorVqRestartBinding | None:
    """Read one small host record; ``None`` means a legacy artifact."""
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
            f"{path}: {BISPINOR_VQ_RESTART_BINDING_DATASET} is not a valid "
            "canonical binding record") from exc


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
            f"{where}: missing {BISPINOR_VQ_RESTART_BINDING_DATASET} in "
            f"{missing}. Legacy artifacts cannot authenticate that the "
            "bispinor photon body and restart carrier share one physical "
            "source; regenerate both with restart = false.")
    restart.assert_same_composition(photon, where=where)
    return restart


__all__ = [
    "BISPINOR_VQ_RESTART_BINDING_DATASET",
    "BISPINOR_VQ_RESTART_BINDING_SCHEMA",
    "BispinorVqRestartBinding",
    "assert_bispinor_vq_restart_binding",
    "read_bispinor_vq_restart_binding",
]
