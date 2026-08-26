"""Host provenance for wavefunctions sampled on an ordered centroid basis.

The receipt defined here is an orchestration object, not a numerical JAX
operand and not an HDF5 payload.  It binds the existing mean-field WFN
fingerprint and restart centroid-table stamp to the transform that produced
the sampled spinors.  Consumers compare it before passing arrays into a
compiled kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib

import numpy as np


#: Existing restart/ISDF centroid-table content identity, now named once.
#: The bytes are unchanged from the historical
#: ``gw.gw_init._centroid_table_md5`` contract: int64, C order, bare MD5
#: hexadecimal digest.
CENTROID_TABLE_FINGERPRINT_SCHEME = "int64-c-order-md5-v1"


def centroid_table_md5(centroid_fft_idx) -> str:
    """Return the canonical restart centroid-table content digest.

    FFT-grid indices, rather than fractional-coordinate text, define the
    sampled basis: distinct text representations that snap to the same grid
    points therefore have one identity.
    """
    try:
        import jax
        values = jax.device_get(centroid_fft_idx)
    except ImportError:  # pragma: no cover - h5py-only inspection installs
        values = centroid_fft_idx
    table = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    if table.ndim != 2 or table.shape[1] != 3:
        raise ValueError(
            "centroid_table_md5 requires FFT-grid indices with shape "
            f"(n_rmu, 3); got {table.shape}")
    return hashlib.md5(table.tobytes()).hexdigest()


def _canonical_band_source_fields(
    *, wfn, wfn_fingerprint_value: str | None, role: str, bispinor: bool,
    band_interval, fft_grid,
) -> dict[str, object]:
    """Build and validate the non-centroid half of one basis receipt."""
    from common.bispinor_init import KINETIC_BALANCE_LIFT_PROVENANCE
    from common.parallel_transport import (
        WFN_FINGERPRINT_SCHEME, wfn_fingerprint)
    from common.wfn_transforms import FULL_BLOCH_TRANSFORM_SCHEME

    start, stop = (int(v) for v in band_interval)
    if start < 0 or stop <= start:
        raise ValueError(
            "WavefunctionBasisReceipt.band_interval must satisfy "
            f"0 <= start < stop; got [{start},{stop})")
    nbands = getattr(wfn, "nbands", None)
    if nbands is not None and stop > int(nbands):
        raise ValueError(
            "WavefunctionBasisReceipt band interval exceeds the source "
            f"WFN: stop={stop}, WFN.nbands={int(nbands)}")
    source_nspinor = int(getattr(wfn, "nspinor", 0))
    use_bispinor = bool(bispinor)
    if use_bispinor and source_nspinor != 2:
        raise ValueError(
            "WavefunctionBasisReceipt bispinor source must carry two "
            f"Pauli spinor components; WFN.nspinor={source_nspinor}")
    if not use_bispinor and source_nspinor <= 0:
        raise ValueError(
            "WavefunctionBasisReceipt source has no positive spinor "
            f"extent; WFN.nspinor={source_nspinor}")
    grid = tuple(int(v) for v in np.asarray(fft_grid).reshape(3))
    if any(v <= 0 for v in grid):
        raise ValueError(
            "WavefunctionBasisReceipt.fft_grid must contain three "
            f"positive integers; got {grid!r}")
    role_value = str(role)
    if role_value not in ("charge", "transverse"):
        raise ValueError(
            "WavefunctionBasisReceipt.role must be 'charge' or "
            f"'transverse'; got {role!r}")
    fingerprint = (
        wfn_fingerprint(wfn)
        if wfn_fingerprint_value is None
        else str(wfn_fingerprint_value))
    if (len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)):
        raise ValueError(
            "WavefunctionBasisReceipt.wfn_fingerprint must be a "
            "64-digit lowercase hexadecimal SHA-256")
    return {
        "role": role_value,
        "wfn_fingerprint_scheme": WFN_FINGERPRINT_SCHEME,
        "wfn_fingerprint": fingerprint,
        "band_interval": (start, stop),
        "fft_grid": grid,
        "source_identity": FULL_BLOCH_TRANSFORM_SCHEME,
        "nspinor_sampled": 4 if use_bispinor else source_nspinor,
        "bispinor_lift_provenance": (
            KINETIC_BALANCE_LIFT_PROVENANCE if use_bispinor else None),
    }


@dataclass(frozen=True)
class WavefunctionBasisReceipt:
    """Immutable identity of psi sampled at one ordered centroid table.

    This host-only receipt deliberately excludes the legacy/face carrier
    layout.  It includes both the full-Bloch transform convention and the
    optional kinetic-balance lift, because a scalar two-spinor carrier and a
    four-component bispinor carrier can originate from the same WFN file,
    band interval, and centroid table while representing different fields.

    ``n_rmu_padded`` is a live carrier fact and can vary with processor count.
    Every other field describes the physical sampled source.
    """

    role: str
    wfn_fingerprint_scheme: str
    wfn_fingerprint: str
    band_interval: tuple[int, int]
    fft_grid: tuple[int, int, int]
    centroid_fingerprint_scheme: str
    centroid_table_md5: str
    n_rmu_logical: int
    n_rmu_padded: int
    source_identity: str
    nspinor_sampled: int
    bispinor_lift_provenance: str | None

    def __post_init__(self) -> None:
        from common.bispinor_init import KINETIC_BALANCE_LIFT_PROVENANCE
        from common.parallel_transport import WFN_FINGERPRINT_SCHEME
        from common.wfn_transforms import FULL_BLOCH_TRANSFORM_SCHEME

        role = str(self.role)
        if role not in ("charge", "transverse"):
            raise ValueError(
                "WavefunctionBasisReceipt.role must be 'charge' or "
                f"'transverse'; got {self.role!r}")
        if self.wfn_fingerprint_scheme != WFN_FINGERPRINT_SCHEME:
            raise ValueError(
                "WavefunctionBasisReceipt requires the canonical WFN "
                f"fingerprint scheme {WFN_FINGERPRINT_SCHEME!r}; got "
                f"{self.wfn_fingerprint_scheme!r}")
        fingerprint = str(self.wfn_fingerprint)
        if (len(fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in fingerprint)):
            raise ValueError(
                "WavefunctionBasisReceipt.wfn_fingerprint must be a "
                "64-digit lowercase hexadecimal SHA-256")
        start, stop = (int(v) for v in self.band_interval)
        if start < 0 or stop <= start:
            raise ValueError(
                "WavefunctionBasisReceipt.band_interval must satisfy "
                f"0 <= start < stop; got [{start},{stop})")
        grid = tuple(int(v) for v in self.fft_grid)
        if len(grid) != 3 or any(v <= 0 for v in grid):
            raise ValueError(
                "WavefunctionBasisReceipt.fft_grid must contain three "
                f"positive integers; got {self.fft_grid!r}")
        if (str(self.centroid_fingerprint_scheme)
                != CENTROID_TABLE_FINGERPRINT_SCHEME):
            raise ValueError(
                "WavefunctionBasisReceipt requires the canonical centroid "
                f"fingerprint scheme {CENTROID_TABLE_FINGERPRINT_SCHEME!r}; "
                f"got {self.centroid_fingerprint_scheme!r}")
        centroid_md5 = str(self.centroid_table_md5)
        if (len(centroid_md5) != 32
                or any(c not in "0123456789abcdef" for c in centroid_md5)):
            raise ValueError(
                "WavefunctionBasisReceipt.centroid_table_md5 must be the "
                "canonical 32-digit lowercase MD5 stamp")
        logical, padded = int(self.n_rmu_logical), int(self.n_rmu_padded)
        if logical <= 0 or padded < logical:
            raise ValueError(
                "WavefunctionBasisReceipt centroid extents must satisfy "
                f"0 < logical <= padded; got {logical}/{padded}")
        if self.source_identity != FULL_BLOCH_TRANSFORM_SCHEME:
            raise ValueError(
                "WavefunctionBasisReceipt requires the canonical full-Bloch "
                f"source identity {FULL_BLOCH_TRANSFORM_SCHEME!r}; got "
                f"{self.source_identity!r}")
        nspinor = int(self.nspinor_sampled)
        lift = self.bispinor_lift_provenance
        if lift is not None:
            lift = str(lift)
        if lift not in (None, KINETIC_BALANCE_LIFT_PROVENANCE):
            raise ValueError(
                "WavefunctionBasisReceipt has an unknown sampled-spinor "
                f"transform {lift!r}")
        if nspinor <= 0 or (lift is not None and nspinor != 4):
            raise ValueError(
                "WavefunctionBasisReceipt sampled spinor extent is "
                f"inconsistent with its transform: nspinor={nspinor}, "
                f"bispinor_lift={lift!r}")

        # Frozen is meaningful only when every nested value is immutable.
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self, "wfn_fingerprint_scheme", str(self.wfn_fingerprint_scheme))
        object.__setattr__(self, "wfn_fingerprint", fingerprint)
        object.__setattr__(self, "band_interval", (start, stop))
        object.__setattr__(self, "fft_grid", grid)
        object.__setattr__(
            self, "centroid_fingerprint_scheme",
            str(self.centroid_fingerprint_scheme))
        object.__setattr__(self, "centroid_table_md5", centroid_md5)
        object.__setattr__(self, "n_rmu_logical", logical)
        object.__setattr__(self, "n_rmu_padded", padded)
        object.__setattr__(self, "source_identity", str(self.source_identity))
        object.__setattr__(self, "nspinor_sampled", nspinor)
        object.__setattr__(self, "bispinor_lift_provenance", lift)

    @classmethod
    def from_source(
        cls,
        *,
        wfn,
        wfn_fingerprint_value: str | None = None,
        role: str,
        bispinor: bool,
        band_interval,
        fft_grid,
        centroid_fft_idx,
        n_rmu_logical: int,
        n_rmu_padded: int,
    ) -> "WavefunctionBasisReceipt":
        """Build the receipt from the canonical WFN/transform/hash owners.

        ``wfn_fingerprint_value`` is only for host orchestration that already
        evaluated :func:`common.parallel_transport.wfn_fingerprint` for this
        exact ``wfn``.  The GW initializer uses it to share one bounded HDF5
        fingerprint scan across the charge/transverse receipts (and the
        restart provenance writer).  Omitting it preserves the self-contained
        constructor used by standalone producers and tests.
        """
        band_fields = _canonical_band_source_fields(
            wfn=wfn, wfn_fingerprint_value=wfn_fingerprint_value,
            role=role, bispinor=bispinor, band_interval=band_interval,
            fft_grid=fft_grid)
        grid = band_fields["fft_grid"]
        import jax
        centroids = np.ascontiguousarray(np.asarray(
            jax.device_get(centroid_fft_idx), dtype=np.int64))
        if centroids.ndim != 2 or centroids.shape[1] != 3:
            raise ValueError(
                "WavefunctionBasisReceipt centroid table must have shape "
                f"(n_rmu,3); got {centroids.shape}")
        logical = int(n_rmu_logical)
        if int(centroids.shape[0]) != logical:
            raise ValueError(
                "WavefunctionBasisReceipt logical centroid extent differs "
                f"from the exact table: {logical} vs {centroids.shape[0]}")
        grid_array = np.asarray(grid, dtype=np.int64)
        if (np.any(centroids < 0)
                or np.any(centroids >= grid_array[None, :])):
            raise ValueError(
                "WavefunctionBasisReceipt centroid indices must lie inside "
                f"fft_grid={grid}")
        return cls(
            **band_fields,
            centroid_fingerprint_scheme=CENTROID_TABLE_FINGERPRINT_SCHEME,
            centroid_table_md5=centroid_table_md5(centroids),
            n_rmu_logical=logical,
            n_rmu_padded=int(n_rmu_padded),
        )

    def assert_matches_source(
        self,
        *,
        wfn,
        role: str,
        bispinor: bool,
        band_interval,
        fft_grid,
        centroid_fft_idx,
        n_rmu_logical: int,
        n_rmu_padded: int,
        where: str,
    ) -> None:
        """Refuse unless current source inputs reproduce this receipt."""
        expected = type(self).from_source(
            wfn=wfn, role=role, bispinor=bispinor,
            band_interval=band_interval, fft_grid=fft_grid,
            centroid_fft_idx=centroid_fft_idx,
            n_rmu_logical=n_rmu_logical, n_rmu_padded=n_rmu_padded)
        differing = [
            item.name for item in fields(self)
            if getattr(self, item.name) != getattr(expected, item.name)]
        if differing:
            raise ValueError(
                f"{where}: supplied WavefunctionBasisReceipt disagrees "
                f"with the canonical source in fields {differing}")

    def assert_matches_band_source(
        self,
        *,
        wfn,
        role: str,
        bispinor: bool,
        band_interval,
        fft_grid,
        wfn_fingerprint_value: str | None = None,
        where: str,
    ) -> None:
        """Authenticate the WFN/band/FFT half of this sampled basis.

        Some operators consume the same loaded band carrier but never sample
        it at centroids.  They must still bind to the orchestration receipt;
        requiring a second centroid argument at such a boundary would make
        an irrelevant table a numerical input.  This owner therefore checks
        exactly the fields fixed before centroid sampling and deliberately
        leaves the receipt's ordered centroid identity untouched.

        ``wfn_fingerprint_value`` is the prepare-time host token for callers
        that already authenticated this exact loaded WFN.  Supplying it
        shares that one bounded HDF5 scan; omitting it keeps standalone
        checks self-contained.
        """
        expected = _canonical_band_source_fields(
            wfn=wfn, wfn_fingerprint_value=wfn_fingerprint_value, role=role,
            bispinor=bispinor, band_interval=band_interval,
            fft_grid=fft_grid)
        differing = [
            name for name, value in expected.items()
            if getattr(self, name) != value]
        if differing:
            raise ValueError(
                f"{where}: supplied WavefunctionBasisReceipt disagrees "
                f"with the canonical band source in fields {differing}")

    def assert_same_source(
        self, other: "WavefunctionBasisReceipt", *, where: str,
    ) -> None:
        """Refuse unless two receipts name one physical sampled basis."""
        if not isinstance(other, WavefunctionBasisReceipt):
            raise TypeError(
                f"{where}: expected WavefunctionBasisReceipt, got "
                f"{type(other).__name__}")
        # The padded extent is the sole runtime-layout field.  Deriving the
        # comparison set from the dataclass schema prevents a newly added
        # physical field from being accidentally omitted here.
        differing = [
            item.name for item in fields(self)
            if item.name != "n_rmu_padded"
            and getattr(self, item.name) != getattr(other, item.name)]
        if differing:
            raise ValueError(
                f"{where}: wavefunction-at-centroids receipts differ in "
                f"physical source fields {differing}")

    def assert_same_carrier(
        self, other: "WavefunctionBasisReceipt", *, where: str,
    ) -> None:
        """Refuse unless physical source and current padded extent agree."""
        self.assert_same_source(other, where=where)
        if int(self.n_rmu_padded) != int(other.n_rmu_padded):
            raise ValueError(
                f"{where}: receipt padded centroid extents differ: "
                f"{int(self.n_rmu_padded)} vs {int(other.n_rmu_padded)}")

    def assert_matches_carrier(self, carrier, *, where: str) -> None:
        """Authenticate one numerical Wavefunctions carrier on the host.

        The receipt deliberately remains outside the carrier's JAX pytree.
        This check is therefore called by the fresh/restart constructors
        before their arrays can be handed to a compiled consumer.
        """
        start, stop = self.band_interval
        slices = carrier.slices
        if start != int(slices.b0) or stop > int(slices.b4):
            raise ValueError(
                f"{where}: basis receipt band interval [{start},{stop}) is "
                f"outside carrier [{int(slices.b0)},{int(slices.b4)})")
        if stop - start > int(carrier.enk.shape[1]):
            raise ValueError(
                f"{where}: basis receipt band width {stop-start} exceeds "
                f"energy carrier width {int(carrier.enk.shape[1])}")
        mu_extents = []
        spin_extents = []
        for value, mu_axis, spin_axis in (
                (carrier.psi_xn, 2, 1), (carrier.psi_xr, 3, 2),
                (carrier.psi_yr, 3, 2), (carrier.psi_yn, 2, 1),
                (carrier.psi_nmu, 3, 2), (carrier.psi_mun, 2, 1)):
            if value is not None:
                mu_extents.append(int(value.shape[mu_axis]))
                spin_extents.append(int(value.shape[spin_axis]))
        if mu_extents and any(
                extent != self.n_rmu_padded for extent in mu_extents):
            raise ValueError(
                f"{where}: carrier centroid extent {mu_extents} disagrees "
                f"with receipt {self.n_rmu_padded}")
        if spin_extents and any(
                extent != self.nspinor_sampled for extent in spin_extents):
            raise ValueError(
                f"{where}: carrier spinor extent {spin_extents} disagrees "
                f"with receipt {self.nspinor_sampled}")


__all__ = [
    "CENTROID_TABLE_FINGERPRINT_SCHEME",
    "WavefunctionBasisReceipt",
    "centroid_table_md5",
]
