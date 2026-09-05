"""Authenticated QE symmetry metadata for a BerkeleyGW WFN header.

``WFN.h5`` retains the Seitz matrices and translations but not QE's
per-operation ``time_reversal`` bit.  This module reads that missing bit from
``data-file-schema.xml`` and binds it only when the schema also describes the
same active operations and stored k rows as the WFN.  A nearby, overwritten
``*.save`` is therefore not accepted merely because its crystal is similar.

The raw matrix boundary is deliberate:

* QE XML ``rotation`` text reshaped in row-major order equals the array h5py
  returns for BGW ``mf_header/symmetry/mtrx``;
* reciprocal k/G actions are formed later as ``mtrx.T``;
* QE's fractional translation is converted to BGW's direct-space carrier as
  ``inv(mtrx) @ tau_qe`` and stored in ``2*pi`` units.

Those are three different operations.  Collapsing any two is the historical
transpose/translation defect this receipt is intended to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


_TWO_PI = 2.0 * np.pi
_MATCH_TOL = 2.0e-7


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _numbers(text: str | None, *, dtype=float) -> np.ndarray:
    if text is None:
        return np.empty((0,), dtype=dtype)
    return np.asarray([dtype(value) for value in text.split()])


def qe_xml_seitz_to_bgw(rotation, fractional_translation) -> np.ndarray:
    """Convert one QE XML translation to BGW ``tnp`` units.

    ``rotation`` is the raw integer matrix in the orientation written by QE
    and retained by BGW.  The returned vector is equivalent modulo ``2*pi``;
    no arbitrary choice between ``+pi`` and ``-pi`` is imposed.
    """
    matrix = np.asarray(rotation, dtype=np.int64)
    tau_qe = np.asarray(fractional_translation, dtype=np.float64)
    if matrix.shape != (3, 3) or tau_qe.shape != (3,):
        raise ValueError(
            "qe_xml_seitz_to_bgw expects rotation (3,3) and translation "
            f"(3,), got {matrix.shape} and {tau_qe.shape}.")
    inverse = np.linalg.inv(matrix.astype(np.float64))
    inverse_integer = np.rint(inverse).astype(np.int64)
    if (not np.array_equal(matrix @ inverse_integer, np.eye(3, dtype=np.int64))
            or not np.array_equal(
                inverse_integer @ matrix, np.eye(3, dtype=np.int64))):
        raise ValueError(
            "QE symmetry rotation has no exact unimodular integer inverse: "
            f"{matrix.tolist()}.")
    return _TWO_PI * (inverse_integer @ tau_qe)


@dataclass(frozen=True)
class QESymmetryReceipt:
    """Small immutable symmetry/k-sampling view of one QE XML schema."""

    schema_path: str
    schema_sha256: str
    kgrid: np.ndarray
    kpoints_crystal: np.ndarray
    nspinor: int
    #: QE's ``<spinorbit>`` (lspinorb).  The ONLY authoritative record of
    #: whether the run used j-resolved projectors; ``noncolin`` does not
    #: imply it.  ``None`` when the schema has no such element.
    spinorbit: bool | None
    do_magnetization: bool | None
    nosym: bool
    noinv: bool
    no_t_rev: bool
    force_symmorphic: bool
    sym_matrices: np.ndarray
    translations: np.ndarray
    antiunitary: np.ndarray


@dataclass(frozen=True)
class QESymmetryBinding:
    """QE receipt after exact compatibility checks against one WFN."""

    schema_path: str
    schema_sha256: str
    antiunitary: np.ndarray
    qe_permitted_pure_time_reversal: bool
    equivalent_schema_paths: tuple[str, ...] = ()
    #: ``QESymmetryReceipt.spinorbit`` of the authenticated schema, so the
    #: WFN loader can expose it (``WfnLoader.spinorbit``) to
    #: ``psp.vnl_ops.resolve_soc_mode`` instead of measuring.
    spinorbit: bool | None = None

    @property
    def n_antiunitary(self) -> int:
        return int(np.count_nonzero(self.antiunitary))


def _bool_text(text: str | None, *, default: bool = False) -> bool:
    if text is None:
        return bool(default)
    value = text.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"QE schema boolean must be true/false, got {text!r}.")


def _schema_cache_key(path: str | Path) -> tuple[str, int, int]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "data-file-schema.xml"
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


def read_qe_symmetry_receipt(path: str | Path) -> QESymmetryReceipt:
    """Read the bounded symmetry/k-point subset of a QE schema.

    The cache key includes size and nanosecond mtime, so an NSCF overwrite in
    the same ``*.save`` directory cannot reuse a stale in-process receipt.
    """
    return _read_qe_symmetry_receipt_cached(*_schema_cache_key(path))


@lru_cache(maxsize=16)
def _read_qe_symmetry_receipt_cached(
    xml_path: str,
    _size: int,
    _mtime_ns: int,
) -> QESymmetryReceipt:
    digest = hashlib.sha256()
    with open(xml_path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    flags: dict[str, bool] = {}
    rotations: list[np.ndarray] = []
    translations_qe: list[np.ndarray] = []
    antiunitary: list[bool] = []
    bvec_rows: dict[str, np.ndarray] = {}
    kpoints_cart_band: list[np.ndarray] = []
    kpoints_cart_input: list[np.ndarray] = []
    kgrid: np.ndarray | None = None
    noncolin = False
    spinorbit: bool | None = None
    do_magnetization: bool | None = None
    current_info_anti = False
    current_info_kind: str | None = None
    current_rotation: np.ndarray | None = None
    current_translation = np.zeros(3, dtype=np.float64)
    declared_nsym: int | None = None
    stack: list[str] = []

    for event, elem in ET.iterparse(xml_path, events=("start", "end")):
        tag = _local(elem.tag)
        if event == "start":
            stack.append(tag)
            if tag == "symmetry":
                current_info_anti = False
                current_info_kind = None
                current_rotation = None
                current_translation = np.zeros(3, dtype=np.float64)
            continue

        parent = stack[-2] if len(stack) >= 2 else ""
        ancestry = tuple(stack)
        if parent == "symmetry_flags" and tag in {
                "nosym", "noinv", "no_t_rev", "force_symmorphic"}:
            flags[tag] = _bool_text(elem.text)
        elif tag == "noncolin" and "band_structure" in ancestry:
            noncolin = _bool_text(elem.text)
        elif tag == "spinorbit" and "band_structure" in ancestry:
            spinorbit = _bool_text(elem.text)
        elif tag == "do_magnetization" and "magnetization" in ancestry:
            do_magnetization = _bool_text(elem.text)
        elif parent == "symmetries" and tag == "nsym":
            declared_nsym = int((elem.text or "").strip())
        elif tag in {"b1", "b2", "b3"} and "reciprocal_lattice" in ancestry:
            bvec_rows[tag] = _numbers(elem.text)
        elif tag == "monkhorst_pack" and "starting_k_points" in ancestry:
            kgrid = np.asarray(
                [int(elem.attrib.get(f"nk{axis}", 1)) for axis in (1, 2, 3)],
                dtype=np.int32)
        elif tag == "k_point" and "ks_energies" in ancestry:
            values = _numbers(elem.text)
            if values.shape == (3,):
                kpoints_cart_band.append(values)
        elif tag == "k_point" and "k_points_IBZ" in ancestry:
            values = _numbers(elem.text)
            if values.shape == (3,):
                kpoints_cart_input.append(values)
        elif parent == "symmetry" and tag == "info":
            current_info_kind = (elem.text or "").strip().lower()
            current_info_anti = (
                elem.attrib.get("time_reversal", "false").strip().lower()
                == "true")
        elif parent == "symmetry" and tag == "rotation":
            values = _numbers(elem.text)
            if values.size != 9:
                raise ValueError(
                    f"QE schema {xml_path}: rotation contains "
                    f"{values.size} values, expected 9.")
            # This raw C reshape is the WFN/HDF5 boundary.  Do not transpose
            # here; reciprocal actions transpose exactly once in SymMaps.
            current_rotation = np.rint(values.reshape(3, 3)).astype(np.int32)
        elif parent == "symmetry" and tag == "fractional_translation":
            values = _numbers(elem.text)
            if values.shape != (3,):
                raise ValueError(
                    f"QE schema {xml_path}: fractional_translation has "
                    f"shape {values.shape}, expected (3,).")
            current_translation = values
        elif tag == "symmetry" and current_rotation is not None:
            # QE emits ``nrot`` lattice candidates after its ``nsym`` active
            # crystal operations.  Only the latter acted on the KS states and
            # can authenticate the WFN header.
            if current_info_kind in {None, "", "crystal_symmetry"}:
                rotations.append(current_rotation)
                translations_qe.append(current_translation.copy())
                antiunitary.append(bool(current_info_anti))

        stack.pop()
        elem.clear()

    if not rotations:
        raise ValueError(f"QE schema {xml_path} contains no symmetry rotations.")
    if declared_nsym is not None and len(rotations) != declared_nsym:
        raise ValueError(
            f"QE schema {xml_path}: declared nsym={declared_nsym} but found "
            f"{len(rotations)} active crystal_symmetry records.")
    if kgrid is None:
        # Older QE schemas with an explicit k-point list carry no MP grid.
        # The WFN still carries it; the exact stored k coordinates below are
        # the stronger binding in that format.
        kgrid = np.zeros(3, dtype=np.int32)
    elif kgrid.shape != (3,) or np.any(kgrid <= 0):
        raise ValueError(f"QE schema {xml_path} has an invalid output kgrid.")
    if set(bvec_rows) != {"b1", "b2", "b3"}:
        raise ValueError(f"QE schema {xml_path} has no complete reciprocal basis.")
    raw_matrices = np.asarray(rotations, dtype=np.int32)
    raw_tau = np.asarray(translations_qe, dtype=np.float64)
    raw_anti = np.asarray(antiunitary, dtype=bool)

    keep = np.ones(raw_matrices.shape[0], dtype=bool)
    if flags.get("nosym", False):
        identity = np.eye(3, dtype=np.int32)
        identity_rows = np.flatnonzero(
            np.all(raw_matrices == identity[None, :, :], axis=(1, 2))
            & ~raw_anti
            & np.all(np.abs(raw_tau - np.rint(raw_tau)) < _MATCH_TOL, axis=1))
        if identity_rows.size != 1:
            raise ValueError(
                f"QE schema {xml_path}: nosym=true but found "
                f"{identity_rows.size} unitary identity rows.")
        keep[:] = False
        keep[int(identity_rows[0])] = True
    else:
        if flags.get("no_t_rev", False):
            keep &= ~raw_anti
        if flags.get("force_symmorphic", False):
            keep &= np.all(
                np.abs(raw_tau - np.rint(raw_tau)) < _MATCH_TOL, axis=1)

    matrices = raw_matrices[keep]
    tau_qe = raw_tau[keep]
    typed = raw_anti[keep]
    tnp = np.stack([
        qe_xml_seitz_to_bgw(matrix, tau)
        for matrix, tau in zip(matrices, tau_qe)
    ])
    bvec = np.stack([bvec_rows[f"b{i}"] for i in (1, 2, 3)])
    kpoints_cart = (
        kpoints_cart_band if kpoints_cart_band else kpoints_cart_input)
    k_cart = np.asarray(kpoints_cart, dtype=np.float64).reshape(-1, 3)
    k_crystal = k_cart @ np.linalg.inv(bvec)

    for array in (kgrid, k_crystal, matrices, tnp, typed):
        array.flags.writeable = False
    return QESymmetryReceipt(
        schema_path=str(xml_path),
        schema_sha256=digest.hexdigest(),
        kgrid=kgrid,
        kpoints_crystal=k_crystal,
        nspinor=2 if noncolin else 1,
        spinorbit=spinorbit,
        do_magnetization=do_magnetization,
        nosym=flags.get("nosym", False),
        noinv=flags.get("noinv", False),
        no_t_rev=flags.get("no_t_rev", False),
        force_symmorphic=flags.get("force_symmorphic", False),
        sym_matrices=matrices,
        translations=tnp,
        antiunitary=typed,
    )


def _periodic_error(actual, expected, *, period: float) -> float:
    delta = (np.asarray(actual, dtype=np.float64)
             - np.asarray(expected, dtype=np.float64)) / float(period)
    delta -= np.rint(delta)
    return float(np.max(np.abs(delta), initial=0.0))


def bind_qe_symmetry_receipt(wfn, receipt: QESymmetryReceipt) -> QESymmetryBinding:
    """Authenticate one QE receipt against a WFN-shaped object."""
    failures: list[str] = []
    wfn_kgrid = np.asarray(wfn.kgrid, dtype=np.int32)
    if np.any(receipt.kgrid > 0) and not np.array_equal(
            receipt.kgrid, wfn_kgrid):
        failures.append(
            f"kgrid QE={receipt.kgrid.tolist()} WFN={wfn_kgrid.tolist()}")
    if int(receipt.nspinor) != int(getattr(wfn, "nspinor", receipt.nspinor)):
        failures.append(
            f"nspinor QE={receipt.nspinor} WFN={int(wfn.nspinor)}")

    nk = int(getattr(wfn, "nkpts", len(wfn.kpoints)))
    wfn_kpoints = np.asarray(wfn.kpoints, dtype=np.float64)[:nk]
    if receipt.kpoints_crystal.shape != wfn_kpoints.shape:
        failures.append(
            f"stored k rows QE={receipt.kpoints_crystal.shape[0]} WFN={nk}")
    elif _periodic_error(receipt.kpoints_crystal, wfn_kpoints, period=1.0) \
            > _MATCH_TOL:
        failures.append("stored k coordinates differ")

    ntran = int(wfn.ntran)
    wfn_matrices = np.asarray(wfn.sym_matrices[:ntran], dtype=np.int32)
    if receipt.sym_matrices.shape != wfn_matrices.shape:
        failures.append(
            f"active operation rows QE={receipt.sym_matrices.shape[0]} "
            f"WFN={ntran}")
    elif not np.array_equal(receipt.sym_matrices, wfn_matrices):
        first = int(np.flatnonzero(np.any(
            receipt.sym_matrices != wfn_matrices, axis=(1, 2)))[0])
        transpose_matches = np.array_equal(
            receipt.sym_matrices.transpose(0, 2, 1), wfn_matrices)
        failures.append(
            f"raw rotation row {first} differs"
            + (" (the transposed stack matches: probable major/minor-axis bug)"
               if transpose_matches else ""))

    wfn_tnp = np.asarray(wfn.translations[:ntran], dtype=np.float64)
    if receipt.translations.shape == wfn_tnp.shape:
        tau_error = _periodic_error(
            receipt.translations, wfn_tnp, period=_TWO_PI)
        if tau_error > _MATCH_TOL:
            failures.append(
                f"Seitz translations differ modulo a lattice vector "
                f"(max fractional error={tau_error:.3e})")
    elif receipt.sym_matrices.shape == wfn_matrices.shape:
        failures.append(
            f"translation rows QE={receipt.translations.shape} "
            f"WFN={wfn_tnp.shape}")

    if failures:
        raise ValueError("; ".join(failures))
    typed = np.asarray(receipt.antiunitary, dtype=bool).copy()
    typed.flags.writeable = False
    # QE's own setup defines the pure k<->-k reduction switch as
    # ``.not. noinv .and. .not. (noncolin .and. domag)``.  The schema's
    # ``magnetization/do_magnetization`` is written directly from ``domag``.
    # ``no_t_rev`` is a different flag: it disables rotation+TR magnetic
    # operations and has already filtered ``typed`` above.  For an older
    # noncollinear schema that omits ``do_magnetization``, fail closed on the
    # pure-TR provenance bit; the measured WFN TR verdict remains separate.
    magnetic_symmetry = (
        int(receipt.nspinor) == 2
        and (receipt.do_magnetization is None
             or bool(receipt.do_magnetization)))
    return QESymmetryBinding(
        schema_path=receipt.schema_path,
        schema_sha256=receipt.schema_sha256,
        antiunitary=typed,
        qe_permitted_pure_time_reversal=(
            not bool(receipt.noinv) and not magnetic_symmetry),
        spinorbit=receipt.spinorbit,
    )


def discover_qe_schema_paths(wfn_path: str | Path) -> tuple[str, ...]:
    """Return a bounded, WFN-anchored set of nearby QE schema candidates."""
    given = Path(wfn_path).expanduser().absolute()
    anchors = [given.parent]
    try:
        resolved_parent = given.resolve().parent
    except OSError:
        resolved_parent = given.parent
    if resolved_parent not in anchors:
        anchors.append(resolved_parent)

    bases: list[Path] = []
    for anchor in anchors:
        for base in (anchor, *tuple(anchor.parents)[:2]):
            if base not in bases:
                bases.append(base)
    candidates: set[Path] = set()
    for base in bases:
        direct = base / "data-file-schema.xml"
        if direct.is_file():
            candidates.add(direct.resolve())
        for relative in (".", "scf", "nscf", "qe/scf", "qe/nscf"):
            directory = base / relative
            if not directory.is_dir():
                continue
            for schema in directory.glob("*.save/data-file-schema.xml"):
                if schema.is_file():
                    candidates.add(schema.resolve())
    return tuple(str(path) for path in sorted(candidates))


def _binding_signature(binding: QESymmetryBinding) -> tuple:
    return (
        tuple(bool(value) for value in binding.antiunitary),
        bool(binding.qe_permitted_pure_time_reversal),
    )


def resolve_qe_symmetry_binding(
    wfn,
    *,
    wfn_path: str | Path,
    schema: str | Path | None = None,
) -> tuple[QESymmetryBinding | None, str]:
    """Resolve an explicit or bounded-auto QE schema for one WFN.

    Explicit schema mismatch is a refusal.  Auto mode returns ``None`` plus
    a detailed diagnostic so :class:`SymMaps` can announce its conservative
    legacy fallback at the exact symmetry-initialization seam.
    """
    explicit = schema is not None
    paths = ((str(Path(schema).expanduser()),) if explicit
             else discover_qe_schema_paths(wfn_path))
    if not paths:
        return None, "no nearby data-file-schema.xml candidate was found"

    bindings: list[QESymmetryBinding] = []
    rejected: list[str] = []
    for path in paths:
        try:
            receipt = read_qe_symmetry_receipt(path)
            bindings.append(bind_qe_symmetry_receipt(wfn, receipt))
        except (OSError, ET.ParseError, ValueError) as exc:
            rejected.append(f"{Path(path).resolve()}: {exc}")
    if explicit and not bindings:
        raise ValueError(
            "Explicit QE symmetry schema does not authenticate this WFN: "
            + (rejected[0] if rejected else str(schema)))
    if not bindings:
        detail = "; ".join(rejected[:3])
        if len(rejected) > 3:
            detail += f"; plus {len(rejected) - 3} more rejected candidate(s)"
        return None, "nearby QE schema candidate(s) did not match the WFN: " + detail

    signatures = {_binding_signature(binding) for binding in bindings}
    if len(signatures) != 1:
        paths_text = ", ".join(binding.schema_path for binding in bindings)
        raise ValueError(
            "Multiple QE schemas authenticate the WFN but disagree on "
            f"antiunitary operation typing: {paths_text}. Pass qe_schema=... "
            "for the WFN-generating NSCF schema.")

    def rank(binding: QESymmetryBinding) -> tuple[int, int, str]:
        path = binding.schema_path
        return (0 if "/nscf/" in path else 1, len(Path(path).parts), path)

    selected = sorted(bindings, key=rank)[0]
    aliases = tuple(
        binding.schema_path for binding in bindings
        if binding.schema_path != selected.schema_path)
    # Re-spelled field by field, this copy silently dropped every binding
    # field added later (spinorbit was the first casualty); replace keeps
    # the whole authenticated record and adds only the aliases.
    import dataclasses as _dc
    selected = _dc.replace(selected, equivalent_schema_paths=aliases)
    diagnostic = (
        f"authenticated {selected.schema_path} "
        f"(sha256={selected.schema_sha256[:12]}, "
        f"antiunitary={selected.n_antiunitary}/{selected.antiunitary.size}, "
        "QE pure-TR reduction flag="
        f"{'enabled' if selected.qe_permitted_pure_time_reversal else 'disabled'})")
    if aliases:
        diagnostic += f"; {len(aliases)} equivalent matching schema(s)"
    return selected, diagnostic


__all__ = [
    "QESymmetryBinding",
    "QESymmetryReceipt",
    "bind_qe_symmetry_receipt",
    "discover_qe_schema_paths",
    "qe_xml_seitz_to_bgw",
    "read_qe_symmetry_receipt",
    "resolve_qe_symmetry_binding",
]
