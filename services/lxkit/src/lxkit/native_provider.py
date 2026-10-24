"""Shared selection and attestation for LORRAX native provider bundles.

``lxkit`` owns policy, never a provider's symbol tables.  The legs
themselves (file name, pin, ABI symbol) are :data:`lxkit.legs.LEGS`; callers
supply their own records over them (build hint and candidates);
this module supplies the one decision procedure used by both
``ffi.common.ffi_loader`` and the independently installable
``distrib_la.loader``.

A sealed deployment is one relocatable directory containing a leg pair (the
host leg and one device leg: CUDA or ROCM) and one ``lorrax_ffi_bundle.json``.
The manifest binds both FFI files, the handler ABI, the source revision and
every private redistributable dependency by size and SHA-256.  MPI, site HDF5,
the device runtime (CUDA or ROCm/HIP) and its collectives (NCCL, RCCL) are
never copied into a bundle; they belong to the machine runtime.

EVERY ROUTE ENFORCES ITS DEPENDENCY CLOSURE.  Each leg's build records the
library set it resolved and verified -- every file of its NEEDED closure by
path, size, mtime and SHA-256, machine libraries (MPI, HDF5, LibSci), the GPU
math/collective libraries the JAX plugin maps (cuBLAS/cuSOLVER/cuFFT/NCCL,
rocBLAS/rocSOLVER/hipBLASLt/RCCL) and the engine-private providers alike: a
checkout leg in ``<leg>.closure.json`` (:data:`CHECKOUT_CLOSURE_SUFFIX`), a
sealed leg in its manifest's ``closures``.  At start-up, before the leg can run
anything, :func:`open_and_attest` checks the record against the leg's bytes
and the recorded files, opens the recorded files by exact path (dependencies
first), opens the leg, and then reads ``/proc/self/maps``: every recorded
family must be mapped exactly once, at the recorded path or byte-identical.  A
missing or stale record, a wrong file and a second copy of a family each refuse
with a ``GATE native-closure`` naming the wrong file, the expected one and the
remedy.  There is no fallback and no warning-and-continue: the launch shell's
``LD_LIBRARY_PATH`` cannot choose a leg's libraries, and nothing else may map a
second copy of them.  :func:`native_closure_identity` is the enforced closure's
identity, the key component of every persistent cache that depends on native
libraries.

Unsealed build-tree legs still print ``LEGACY-UNSEALED`` together with the
actual file hash (the pair is not sealed), but their closure is enforced
exactly as a sealed bundle's.  A manifest is never advisory: once either
selected leg claims one, a partial pin, a mixed provider, a missing file, a
changed byte, a wrong ABI or a wrong ``dladdr`` origin refuses before target
registration.

PINNED PROPERTY: stdlib only.  This module is safe before JAX or MPI startup.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Type

from lxkit.legs import HOST, LEGS

from lxkit.probe import LibraryNotBuilt, LibraryUnusable

__all__ = [
    "BUNDLE_MANIFEST", "BundleAttestation", "CHECKOUT_CLOSURE_SUFFIX",
    "CLOSURE_SCHEMA", "ClosureEntry", "ClosureNotEnforced", "ClosureRecord",
    "NativeAbiMismatch", "check_closure_mapping", "check_record_files",
    "closure_record_digest", "closure_record_path", "expected_closure_identity",
    "library_family", "native_closure_identity",
    "native_closure_identity_or_expected", "native_closure_content_identity_or_expected",
    "parse_closure",
    "read_checkout_closure", "resolve_closure",
    "assert_one_mapped_mpi_runtime", "check_abi", "checkout_build_dir",
    "describe_library",
    "check_launch_libraries", "is_private_redistributable", "locate_library", "open_and_attest",
    "refuse_private_library",
]

BUNDLE_MANIFEST = "lorrax_ffi_bundle.json"
#: 2: the manifest carries each leg's dependency closure (``closures``).
_MANIFEST_SCHEMA = 2
#: Suffix of the record a checkout build writes beside an unsealed leg
#: (``src/ffi/cpp/stage/write_checkout_closure.py``).
CHECKOUT_CLOSURE_SUFFIX = ".closure.json"


def _is_leg_pair(names) -> bool:
    """A sealed bundle binds exactly the host leg and one device leg."""
    names = set(names)
    return (len(names) == 2 and HOST.key in names
            and all(n == HOST.key or (n in LEGS and LEGS[n].accelerator) for n in names))


def _pin_names(names) -> str:
    return " and ".join(LEGS[n].so_env for n in sorted(names) if n in LEGS)
_ANNOUNCED: set[tuple[str, str]] = set()
_PRIVATE_HANDLES: dict[str, tuple[ctypes.CDLL, ...]] = {}


class NativeAbiMismatch(LibraryUnusable):
    """A selected native provider does not speak the caller's handler ABI."""


@dataclass(frozen=True)
class BundleAttestation:
    """A fully verified, relocatable two-leg bundle manifest."""

    manifest_path: Path
    bundle_root: Path
    bundle_id: str
    source_revision: str
    abi: int
    libraries: Mapping[str, Path]
    library_hashes: Mapping[str, str]
    private_libraries: tuple[Path, ...]
    closures: Mapping[str, "ClosureRecord"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_manifest_id(doc: Mapping[str, object]) -> str:
    body = dict(doc)
    body.pop("bundle_id", None)
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def _inside(root: Path, relative: object, *, field: str,
            unusable_cls: Type[OSError]) -> Path:
    text = str(relative)
    rel = Path(text)
    if not text or rel.is_absolute() or ".." in rel.parts:
        raise unusable_cls(
            f"{field} must be a non-empty bundle-relative path without '..'; "
            f"got {text!r}.")
    path = root / rel
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise unusable_cls(
            f"sealed native bundle {field} escapes or is missing: {path}") from exc
    if path.is_symlink() or not resolved.is_file():
        raise unusable_cls(
            f"sealed native bundle {field} must be an immutable regular file, "
            f"not a symlink or special file: {path}")
    return resolved


_RUNTIME_OWNED = (
    re.compile(r"^libmpi(?:_|[.])", re.I),
    re.compile(r"^libmpich", re.I),
    re.compile(r"^libopen-(?:pal|rte)", re.I),
    re.compile(r"^libhdf5", re.I),
    re.compile(r"^libcuda(?:[.]|$)", re.I),
    re.compile(r"^libcudart", re.I),
    re.compile(r"^libcublas(?:Lt)?[.]", re.I),
    re.compile(r"^libcusolver[.]", re.I),
    re.compile(r"^libcufft", re.I),
    re.compile(r"^libcurand", re.I),
    re.compile(r"^libcusparse", re.I),
    re.compile(r"^libnvrtc", re.I),
    re.compile(r"^libnvjitlink", re.I),
    re.compile(r"^libnvidia", re.I),
    re.compile(r"^libnccl", re.I),
    # ROCm: the HIP/HSA runtime, the code-object manager, the vendor math
    # libraries and RCCL belong to the machine's ROCm, as CUDA's do above.
    re.compile(r"^lib(?:amdhip64|hsa-runtime64|hsakmt|amd_comgr|hiprtc)", re.I),
    re.compile(r"^lib(?:hip|roc)(?:blas|blaslt|fft|solver|sparse|rand)", re.I),
    re.compile(r"^librccl", re.I),
    re.compile(r"^libamd_smi", re.I),
    re.compile(r"^ld-linux", re.I),
    re.compile(
        r"^lib(?:c|m|dl|pthread|rt|util|resolv|crypt|gcc_s|stdc[+][+]|gomp|"
        r"atomic)[.]", re.I),
)
_PRIVATE_REDISTRIBUTABLE = re.compile(
    r"^lib(?:cusolverMp|cublasmp|cal|nvshmem(?:_host)?|elpa(?:_openmp)?)[.]", re.I)


def is_private_redistributable(path: str | Path) -> bool:
    """Whether a basename belongs to the approved engine-private closure."""
    return bool(_PRIVATE_REDISTRIBUTABLE.search(Path(path).name))


def _refuse_runtime_owned(path: Path, unusable_cls: Type[OSError]) -> None:
    name = path.name
    if any(rx.search(name) for rx in _RUNTIME_OWNED):
        raise unusable_cls(
            f"sealed native bundle lists runtime-owned {name!r} as a private "
            "library.  MPI, site HDF5, the CUDA or ROCm runtime/driver "
            "libraries and NCCL/RCCL must come from the selected machine "
            "runtime; never copy them into a LORRAX bundle.")


def refuse_private_library(path: str | Path,
                           error_cls: Type[OSError] = LibraryUnusable) -> None:
    """Refuse a process-, site-, or accelerator-runtime-owned private file.

    The bundle sealer and both runtime consumers deliberately share this one
    classification.  Adding an engine-private redistributable dependency is
    allowed; teaching only a deployment script that a runtime provider is
    private is not.
    """
    candidate = Path(path)
    _refuse_runtime_owned(candidate, error_cls)
    if not is_private_redistributable(candidate):
        raise error_cls(
            f"refusing unclassified private library {candidate.name!r}.  A "
            "sealed bundle may carry only the centrally classified LORRAX "
            "engine redistributables (cuSolverMp/cuBLASMp/CAL, ELPA, or "
            "NVSHMEM); process, compiler, site and accelerator "
            "runtime providers remain external.")


def _verify_file(entry: object, root: Path, *, field: str,
                 unusable_cls: Type[OSError]) -> tuple[Path, str]:
    if not isinstance(entry, dict):
        raise unusable_cls(f"sealed native bundle {field} must be an object")
    path = _inside(root, entry.get("path"), field=f"{field}.path",
                   unusable_cls=unusable_cls)
    want_hash = str(entry.get("sha256", ""))
    want_bytes = entry.get("bytes")
    if not re.fullmatch(r"[0-9a-f]{64}", want_hash):
        raise unusable_cls(
            f"sealed native bundle {field}.sha256 is not a full SHA-256")
    try:
        want_bytes_i = int(want_bytes)
    except (TypeError, ValueError) as exc:
        raise unusable_cls(
            f"sealed native bundle {field}.bytes is not an integer") from exc
    have_bytes = path.stat().st_size
    if have_bytes != want_bytes_i:
        raise unusable_cls(
            f"sealed native bundle byte-size mismatch for {path}: manifest "
            f"says {want_bytes_i}, actual file is {have_bytes}")
    have_hash = _sha256(path)
    if have_hash != want_hash:
        raise unusable_cls(
            f"sealed native bundle SHA-256 mismatch for {path}: manifest says "
            f"{want_hash}, actual file is {have_hash}.  The staged bundle was "
            "changed after sealing; refuse it rather than loading stale bytes.")
    return path, have_hash


def _discover_manifest(path: Path) -> Path | None:
    """Find only the two relocatable layouts the sealer writes.

    ``<bundle>/lib/<so>`` is canonical; ``<bundle>/<so>`` is accepted for a
    deliberately flat install.  We never walk ancestors or search the machine.
    """
    for candidate in (path.parent / BUNDLE_MANIFEST,
                      path.parent.parent / BUNDLE_MANIFEST):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _read_bundle(manifest: Path, *, expected_abi: int,
                 unusable_cls: Type[OSError]) -> BundleAttestation:
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise unusable_cls(
            f"native bundle manifest is unreadable or invalid JSON: {manifest}: "
            f"{exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != _MANIFEST_SCHEMA:
        raise unusable_cls(
            f"native bundle manifest {manifest} has schema "
            f"{doc.get('schema') if isinstance(doc, dict) else '<non-object>'}; "
            f"this runtime requires schema {_MANIFEST_SCHEMA}")
    bundle_id = str(doc.get("bundle_id", ""))
    computed_id = _canonical_manifest_id(doc)
    if bundle_id != computed_id:
        raise unusable_cls(
            f"native bundle manifest identity mismatch at {manifest}: recorded "
            f"{bundle_id or '<missing>'}, computed {computed_id}.  The manifest "
            "was changed after sealing.")
    try:
        abi = int(doc.get("ffi_abi"))
    except (TypeError, ValueError) as exc:
        raise unusable_cls(
            f"native bundle manifest {manifest} has no integer ffi_abi") from exc
    if abi != expected_abi:
        raise unusable_cls(
            f"native bundle ABI mismatch before dlopen: {manifest} binds "
            f"abi={abi}, this Python provider speaks abi={expected_abi}.  "
            "Rebuild and reseal both legs as one bundle.")
    source = doc.get("source")
    if not isinstance(source, dict):
        raise unusable_cls(f"native bundle manifest {manifest} has no source object")
    revision = str(source.get("revision", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise unusable_cls(
            f"native bundle manifest {manifest} has no full source revision")
    if source.get("dirty") is not False:
        raise unusable_cls(
            f"native bundle {manifest} was made from a dirty source tree.  A "
            "production bundle must correspond to one immutable commit.")

    records = doc.get("libraries")
    if not isinstance(records, dict) or not _is_leg_pair(records):
        raise unusable_cls(
            f"native bundle manifest {manifest} must bind exactly the host leg "
            f"and one device leg (CUDA or ROCM); got "
            f"{sorted(records) if isinstance(records, dict) else records}")
    root = manifest.parent.resolve()
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for platform in sorted(records):
        paths[platform], hashes[platform] = _verify_file(
            records[platform], root, field=f"libraries.{platform}",
            unusable_cls=unusable_cls)

    private_records = doc.get("private_libraries", [])
    if not isinstance(private_records, list):
        raise unusable_cls(
            f"native bundle manifest {manifest} private_libraries must be a list")
    private: list[Path] = []
    seen_names: set[str] = set()
    for i, entry in enumerate(private_records):
        path, _ = _verify_file(entry, root, field=f"private_libraries[{i}]",
                               unusable_cls=unusable_cls)
        refuse_private_library(path, unusable_cls)
        soname = str(entry.get("soname", "")) if isinstance(entry, dict) else ""
        if (not soname or Path(soname).name != soname
                or path.name != soname):
            raise unusable_cls(
                f"sealed native bundle private_libraries[{i}] must bind one "
                f"plain ELF SONAME equal to its staged filename; path={path.name!r}, "
                f"soname={soname or '<missing>'!r}")
        refuse_private_library(Path(soname), unusable_cls)
        if soname in seen_names:
            raise unusable_cls(
                f"native bundle {manifest} names private provider {soname!r} "
                "more than once; one SONAME must have one origin")
        seen_names.add(soname)
        private.append(path)
    closure_docs = doc.get("closures")
    if not isinstance(closure_docs, dict) or set(closure_docs) != set(paths):
        raise unusable_cls(
            f"native bundle manifest {manifest} must carry one dependency "
            f"closure per leg ({sorted(paths)}); got "
            f"{sorted(closure_docs) if isinstance(closure_docs, dict) else closure_docs}."
            "  Reseal with this tree's seal_bundle.py from legs whose builds "
            "wrote <leg>.closure.json.")
    closures: dict[str, ClosureRecord] = {}
    for platform in sorted(paths):
        rec = parse_closure(closure_docs[platform], origin=manifest,
                            leg=paths[platform], platform=platform,
                            unusable_cls=unusable_cls, bundle_root=root)
        if rec.leg_sha256 != hashes[platform]:
            raise unusable_cls(
                f"native bundle {manifest}: the {platform} closure was recorded "
                f"for leg bytes {rec.leg_sha256[:16]}, the sealed leg is "
                f"{hashes[platform][:16]}; reseal")
        closures[platform] = rec
    return BundleAttestation(
        manifest_path=manifest, bundle_root=root, bundle_id=bundle_id,
        source_revision=revision, abi=abi, libraries=paths,
        library_hashes=hashes, private_libraries=tuple(private),
        closures=closures)


def checkout_build_dir(anchor: str | Path, build_subdir: str) -> Path | None:
    """``<checkout>/src/ffi/cpp/<build_subdir>`` for the nearest LORRAX
    checkout above ``anchor`` (a caller's ``__file__``), or ``None``.

    The one in-tree search every provider loader uses, so the default
    candidate order has one source of truth.  Walk up rather than hard-code
    a depth: the answer must not change if a service directory moves.
    ``None`` means "installed on its own"; then only the env pin (or
    ``sys.path``) can select a library.
    """
    tail = Path("src") / "ffi" / "cpp" / build_subdir
    for parent in Path(anchor).resolve().parents:
        cand = parent / tail
        if cand.is_dir():
            return cand
    return None


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except OSError:
        return False


def locate_library(
    platform: str,
    *,
    specs: Mapping[str, Mapping[str, object]],
    candidates: Mapping[str, Sequence[Path]],
    expected_abi: int,
    environ: Mapping[str, str] | None = None,
    not_built_cls: Type[OSError] = LibraryNotBuilt,
    unusable_cls: Type[OSError] = LibraryUnusable,
) -> Path:
    """Select one leg and enforce a sealed pair before any ``dlopen``.

    Candidate construction remains a table fact owned by each caller.  Every
    consequence of those candidates -- pin refusal, pair coherence, manifest
    identity and file hashes -- lives here.
    """
    env = os.environ if environ is None else environ
    if platform not in specs:
        raise not_built_cls(
            f"no native provider for platform {platform!r} (known: "
            f"{sorted(specs)})")

    selected: dict[str, Path] = {}
    pinned_platforms: set[str] = set()
    for name, spec in specs.items():
        pin_name = str(spec["env"])
        pinned = env.get(pin_name)
        if pinned:
            pinned_platforms.add(name)
            pin = Path(pinned)
            if not pin.is_file():
                raise unusable_cls(
                    f"{pin_name} is set to {pinned!r}, which is not a file.  "
                    f"Refusing to fall back to another {spec['so_name']}: an "
                    "explicit pin that cannot be honored is a refusal, not a hint.")
            selected[name] = pin.resolve()
            continue
        for candidate in candidates.get(name, ()):
            if candidate.is_file():
                selected[name] = candidate.resolve()
                break

    if platform not in selected:
        searched = "\n  ".join(str(p) for p in candidates.get(platform, ()))
        spec = specs[platform]
        raise not_built_cls(
            f"Could not locate {spec['so_name']} (platform={platform}).  Build "
            f"with:\n    {spec['build_hint']}\nPaths searched:\n  "
            f"{searched or '(none)' }" )

    manifests = {name: _discover_manifest(path)
                 for name, path in selected.items()}
    claimed = {m for m in manifests.values() if m is not None}
    if not claimed:
        return selected[platform]
    if len(claimed) != 1:
        raise unusable_cls(
            "mixed native providers: the selected legs claim different "
            f"bundle manifests: {sorted(str(p) for p in claimed)}")
    att = _read_bundle(next(iter(claimed)), expected_abi=expected_abi,
                       unusable_cls=unusable_cls)
    bundle_legs = set(att.libraries)
    if platform not in bundle_legs:
        raise unusable_cls(
            f"sealed bundle {att.manifest_path} binds the {sorted(bundle_legs)} "
            f"legs, not {platform}; select a bundle built for this platform")
    pinned_in_pair = pinned_platforms & bundle_legs
    if pinned_in_pair and pinned_in_pair != bundle_legs:
        raise unusable_cls(
            f"partial sealed-bundle override refused: {_pin_names(bundle_legs)} "
            "must either both be unset or both select the same two-leg bundle; "
            f"pinned platforms={sorted(pinned_platforms)}")
    for name in sorted(bundle_legs):
        chosen = selected.get(name)
        expected = att.libraries[name]
        if chosen is None:
            selected[name] = expected
            chosen = expected
        if not _same_file(chosen, expected):
            raise unusable_cls(
                f"mixed native provider for {name}: selected {chosen}, but "
                f"sealed bundle {att.manifest_path} binds {expected}.  Both legs "
                "must come from one manifest.")
        other_manifest = manifests.get(name)
        if other_manifest is not None and other_manifest != att.manifest_path:
            raise unusable_cls(
                f"mixed native provider for {name}: {chosen} claims "
                f"{other_manifest}, not {att.manifest_path}")
    return selected[platform]


def check_abi(lib: ctypes.CDLL, platform: str, path: str, *,
              expected_abi: int, abi_symbols: Mapping[str, str],
              build_hint: str, strict_unstamped: bool = False,
              mismatch_cls: Type[OSError] = NativeAbiMismatch) -> None:
    """Check the live per-leg ABI symbol before any target registration."""
    symbol = abi_symbols[platform]
    fn = getattr(lib, symbol, None)
    manifest = _discover_manifest(Path(path).resolve())
    if fn is None:
        if strict_unstamped or manifest is not None:
            raise mismatch_cls(
                f"{path} carries no handler-ABI stamp ({symbol} is not exported) "
                f"but this provider requires abi={expected_abi}.  Rebuild: "
                f"{build_hint}")
        key = ("unstamped", str(Path(path).resolve()))
        if key not in _ANNOUNCED:
            _ANNOUNCED.add(key)
            print(
                f"[lorrax native] LEGACY-UNSEALED {path}: no handler-ABI "
                f"stamp; compatibility with abi={expected_abi} cannot be "
                f"proved.  Developer migration only.  Rebuild with "
                f"{build_hint}.", file=sys.stderr, flush=True)
        return
    fn.restype = ctypes.c_int
    fn.argtypes = []
    found = int(fn())
    if found != expected_abi:
        raise mismatch_cls(
            "HANDLER ABI MISMATCH.\n"
            f"  library  {path}\n           speaks abi={found}\n"
            f"  this tree speaks abi={expected_abi}\n"
            "These cannot be paired.  Rebuild and reseal both legs from the "
            f"same source tree: {build_hint}")


class _DlInfo(ctypes.Structure):
    _fields_ = [("dli_fname", ctypes.c_char_p),
                ("dli_fbase", ctypes.c_void_p),
                ("dli_sname", ctypes.c_char_p),
                ("dli_saddr", ctypes.c_void_p)]


def _actual_symbol_origin(lib: ctypes.CDLL, symbol: str) -> Path:
    fn = getattr(lib, symbol)
    addr = ctypes.cast(fn, ctypes.c_void_p).value
    process = ctypes.CDLL(None)
    dladdr = process.dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    dladdr.restype = ctypes.c_int
    info = _DlInfo()
    if not addr or not dladdr(ctypes.c_void_p(addr), ctypes.byref(info)):
        raise LibraryUnusable(
            f"dladdr could not identify the shared object exporting {symbol}")
    if not info.dli_fname:
        raise LibraryUnusable(f"dladdr returned no origin for {symbol}")
    return Path(os.fsdecode(info.dli_fname)).resolve()


# libmpi.so, libmpi_gnu.so (cray-mpich >= 9.1) or libmpi_gnu_<N>.so: the
# pattern of src/ffi/cpp/gate_one_mpi.sh.  Without the bare libmpi_gnu form a
# LibSci 26.03 pulled in beside cray-mpich 9.0.1 mapped a second MPI unseen,
# and a CPU run hung in its second MPI_Init.
_MPI_RUNTIME_BASENAME = re.compile(r"^libmpi(?:_gnu(?:_[0-9]+)?)?[.]so(?:[.]|$)")


def _mapped_paths() -> set[Path]:
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            return {
                Path(os.path.realpath(line.rsplit(maxsplit=1)[-1]))
                for line in maps
                if "/" in line and not line.rstrip().endswith(" (deleted)")
            }
    except OSError:
        return set()


def assert_one_mapped_mpi_runtime(*,
                                  unusable_cls: Type[OSError] = LibraryUnusable
                                  ) -> None:
    paths = {path for path in _mapped_paths()
             if _MPI_RUNTIME_BASENAME.match(path.name)}
    if len(paths) > 1:
        raise unusable_cls(
            "more than one MPI runtime is mapped after loading the native "
            "provider; MPI communicators from different runtimes are "
            f"incompatible: {sorted(str(p) for p in paths)}")
    if ("mpi", "") not in _ANNOUNCED:
        _ANNOUNCED.add(("mpi", ""))
        rank = os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "?"))
        print(f"[lorrax native] rank={rank} mpi="
              f"{next(iter(paths)) if paths else 'none'}",
              file=sys.stderr, flush=True)


def _attest_private_mapping(att: BundleAttestation,
                            unusable_cls: Type[OSError],
                            opened: Sequence[str] = ()) -> None:
    """Every private provider the OPENED legs' closures record is mapped once,
    from the bundle; no other provider of a recorded family is mapped.

    ``opened``: the bundle's platforms open in this process (the one being
    opened included).  A leg preloads only its own closure, so the host leg's
    private providers are not mapped while only the device leg is open.  Empty: all legs.
    """
    mapped = _mapped_paths()
    if not mapped:  # non-Linux diagnostic environment
        return
    by_name: dict[str, set[Path]] = {}
    for path in mapped:
        by_name.setdefault(path.name, set()).add(path)
    legs = [p for p in (opened or att.closures) if p in att.closures]
    wanted = {e.path for p in legs for e in att.closures[p].entries if e.role == "private"}
    for expected in att.private_libraries:
        if expected not in wanted:
            continue
        actual = by_name.get(expected.name, set())
        if expected not in actual or len(actual) != 1:
            raise unusable_cls(
                f"private provider origin mismatch for {expected.name}: sealed "
                f"bundle requires {expected}, live mappings are "
                f"{sorted(str(p) for p in actual) or '<none>'}")
    expected_private = set(att.private_libraries)
    # Only a provider of a family some leg records can win a SONAME a leg binds;
    # another engine-private file (the JAX plugin's own NVSHMEM in a venv with
    # CUDA wheels) cannot, and a second copy of a recorded family is refused by
    # the closure gate already.
    recorded = {e.family for rec in att.closures.values() for e in rec.entries}
    unlisted = sorted(
        str(path) for path in mapped
        if is_private_redistributable(path) and path not in expected_private
        and library_family(_mapped_sonames().get(path, path.name)) in recorded)
    if unlisted:
        raise unusable_cls(
            "sealed native bundle has an engine-private provider mapped "
            "outside its manifest: " + ", ".join(unlisted) + ".  A stale "
            "cuSolverMp/cuBLASMp/CAL/ELPA/NVSHMEM provider can win by "
            "SONAME even when the requested FFI file itself is exact; restart "
            "with only the bundle's declared private closure.")
    for path in mapped:
        try:
            path.relative_to(att.bundle_root)
        except ValueError:
            continue
        _refuse_runtime_owned(path, unusable_cls)


_DT_NEEDED, _DT_SONAME, _DT_RPATH, _DT_RUNPATH = 1, 14, 15, 29


def _elf_dynamic(path: Path) -> dict[int, list[str]]:
    """DT_NEEDED / DT_SONAME / DT_RPATH / DT_RUNPATH strings of an ELF64
    little-endian shared object; empty for anything else."""
    tags: dict[int, list[str]] = {}
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0,
                                           access=mmap.ACCESS_READ) as m:
        if m[:4] != b"\x7fELF" or m[4] != 2 or m[5] != 1:
            return tags
        shoff, = struct.unpack_from("<Q", m, 0x28)
        shentsize, shnum = struct.unpack_from("<HH", m, 0x3A)
        sections = [struct.unpack_from("<4xI16xQQI", m, shoff + i * shentsize)
                    for i in range(shnum)]          # (type, offset, size, link)
        for sh_type, offset, size, link in sections:
            if sh_type != 6:                        # SHT_DYNAMIC
                continue
            strtab = sections[link][1]
            for j in range(size // 16):
                tag, val = struct.unpack_from("<qQ", m, offset + 16 * j)
                if tag == 0:
                    break
                if tag in (_DT_NEEDED, _DT_SONAME, _DT_RPATH, _DT_RUNPATH):
                    end = m.find(b"\0", strtab + val)
                    tags.setdefault(tag, []).append(
                        m[strtab + val:end].decode("utf-8", "replace"))
    return tags


# ---------------------------------------------------------------------------
# The dependency closure: recorded at build, preloaded and enforced at start-up
# ---------------------------------------------------------------------------
#
# A leg's search path is ``$ORIGIN`` alone, a DT_RUNPATH that LD_LIBRARY_PATH
# outranks, so without a record the launch shell decides which file serves each
# SONAME.  Perlmutter's cudatoolkit/13.2 serves cuSOLVERMp 0.8 under the SONAME
# of the staged 0.9.1 the CUDA leg is compiled against, and the site-default
# LibSci/HDF5 map a second cray-mpich.  So every build writes the closure it
# resolved (``src/ffi/cpp/stage/write_checkout_closure.py``; a sealed bundle
# carries the same record in its manifest), and every start-up
#
#   1. checks the record against the leg's bytes and the recorded files'
#      size/mtime (a stale record refuses: rebuild);
#   2. opens every recorded file by exact path, RTLD_GLOBAL, dependencies first,
#      so the recorded file publishes each SONAME before anything searches;
#   3. after the leg's own dlopen, reads /proc/self/maps and requires, for every
#      recorded family, exactly ONE mapped object, at the recorded path or with
#      the recorded SHA-256.  A second copy of a family, a wrong file, or a
#      recorded family that is not mapped refuses with a GATE naming the wrong
#      file, the expected file and the remedy.  There is no fallback.
#
# Start-up only: stat + ELF headers of the mapped objects; no device work, no
# environment, no flags.

CLOSURE_SCHEMA = 2

#: The process, compiler and driver runtime: never recorded, never enforced
#: (the OS image and the GPU driver own them; a leg cannot choose them).
_BASE_RUNTIME = re.compile(
    r"^(?:ld-linux|linux-vdso|lib(?:c|m|dl|pthread|rt|util|resolv|crypt|nsl|anl|"
    r"gcc_s|stdc[+][+]|gomp|atomic|quadmath|gfortran)[.]so)", re.I)
_DRIVER = re.compile(r"^lib(?:cuda|nvidia-[A-Za-z0-9_-]+|nvcuvid)[.]so", re.I)

#: Families a build MUST resolve from a recorded directory.  If one of these
#: is reachable only through the build shell's search path, the writer refuses
#: rather than leaving it to the run shell.
_MUST_PIN = re.compile(
    r"^lib(?:mpi|mpich|hdf5|sci_|cusolver|cublas|cufft|cusparse|curand|cudart|"
    r"nvrtc|nvJitLink|nccl|cal[.]|nvshmem|rocsolver|rocblas|hipblas|rccl|"
    r"amdhip64|rocfft|hipfft|rocsparse|hipsparse|hipsolver|hsa-runtime64|"
    r"amd_comgr|hiprtc|openblas|scalapack|fftw3|elpa)", re.I)

_COMPILERS = r"(?:gnu|cray|intel|nvidia|nv|aocc|amd|pgi)"
#: One family per role, whatever the SONAME spelling: two members of a family
#: in one process are two copies of one library.  Everything else is its own
#: family (the SONAME up to ``.so``), so an auditwheel copy such as h5py's
#: ``libhdf5-9e18f0c6.so.320`` -- a private SONAME that no leg can bind -- is
#: not a second ``libhdf5``.
_FAMILY_RULES = (
    (re.compile(rf"^libmpi(?:_{_COMPILERS}(?:_[0-9]+)?)?[.]so(?:[.]|$)", re.I), "mpi"),
    (re.compile(r"^libmpich[.]so(?:[.]|$)", re.I), "mpi"),
    (re.compile(r"^libsci_[a-z]+(?:_[0-9]+)?_mpi(?:_mp)?[.]so", re.I), "libsci_mpi"),
    (re.compile(r"^libsci_[a-z]+(?:_[0-9]+)?(?:_mp)?[.]so", re.I), "libsci"),
    (re.compile(rf"^libhdf5_hl(?:_parallel)?(?:_{_COMPILERS})?(?:_[0-9]+)?[.]so", re.I),
     "libhdf5_hl"),
    (re.compile(rf"^libhdf5(?:_parallel)?(?:_{_COMPILERS})?(?:_[0-9]+)?[.]so", re.I),
     "libhdf5"),
    # Every OpenBLAS build exports the same BLAS/LAPACK symbols: the threading
    # and integer-width variants (libopenblas_pthreads, _openmp, libopenblasp,
    # libopenblas64_) are one family.  A system libopenblas_pthreads.so.0
    # beside the ELPA OpenBLAS is two BLAS.  (numpy's libscipy_openblas64_
    # prefixes its symbols and is its own family.)
    (re.compile(r"^libopenblas[a-z0-9_]*[.]so", re.I), "libopenblas"),
)

def library_family(soname: str) -> str:
    """The family a SONAME (or file name) belongs to; see :data:`_FAMILY_RULES`."""
    name = Path(str(soname)).name
    for rx, family in _FAMILY_RULES:
        if rx.search(name):
            return family
    return name.split(".so", 1)[0]


class ClosureNotEnforced(RuntimeError):
    """:func:`native_closure_identity` was asked before its leg was enforced."""


@dataclass(frozen=True)
class ClosureEntry:
    """One recorded library of a leg's dependency closure."""

    soname: str
    path: Path          # resolved file (bundle files resolved under the bundle)
    sha256: str
    bytes: int
    mtime_ns: int       # 0: not checked (a sealed bundle file; its hash is)
    role: str           # private | process | pinned | rpath | attest
    family: str
    version: str = ""   # owning wheel "name==version" where the build found one


@dataclass(frozen=True)
class ClosureRecord:
    """A leg's recorded dependency closure (schema :data:`CLOSURE_SCHEMA`)."""

    origin: Path        # <leg>.closure.json, or the bundle manifest
    platform: str
    leg: Path
    leg_sha256: str
    entries: tuple[ClosureEntry, ...]
    digest: str         # sha256 of the canonical record body


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def closure_record_digest(body: Mapping[str, object]) -> str:
    """SHA-256 over what a record binds: schema, platform, the leg's bytes and,
    per library, SONAME, path (bundle-relative inside a bundle), content hash,
    size, role and family.  Diagnostics (search dirs, host, time, externals)
    are not part of it."""
    libs = [{k: lib.get(k) for k in ("soname", "path", "bundle_path", "sha256",
                                      "bytes", "role", "family")}
            for lib in body.get("libraries", [])]                 # type: ignore[union-attr]
    return hashlib.sha256(_canonical({
        "schema": body.get("schema"), "platform": body.get("platform"),
        "leg_sha256": body.get("leg_sha256"), "libraries": libs})).hexdigest()


def _gate(unusable_cls: Type[OSError], got: str, want: str, fix: str) -> OSError:
    return unusable_cls(f"GATE native-closure: got {got}; want {want}; fix: {fix}")


def _rebuild_fix(record_or_leg: object, build_hint: str = "") -> str:
    hint = build_hint or "config/<site>/build_ffi_<leg>.sh"
    return (f"rebuild the leg in the environment you run in ({hint}); the build "
            f"records the closure it resolved and verified")


def parse_closure(body: Mapping[str, object], *, origin: Path, leg: Path,
                  platform: str, unusable_cls: Type[OSError] = LibraryUnusable,
                  bundle_root: Path | None = None) -> ClosureRecord:
    """Validate one record body.  Inside a bundle a private entry names
    ``bundle_path`` (relative to ``bundle_root``) instead of ``path``."""
    if not isinstance(body, Mapping) or body.get("schema") != CLOSURE_SCHEMA:
        schema = body.get("schema") if isinstance(body, Mapping) else None
        raise _gate(unusable_cls, f"closure record {origin} with schema {schema!r}",
                    f"schema {CLOSURE_SCHEMA} (a record that names every library "
                    "the leg maps)", _rebuild_fix(leg))
    if body.get("platform") != platform:
        raise _gate(unusable_cls, f"closure record {origin} for platform "
                    f"{body.get('platform')!r}", f"a record for {platform}",
                    _rebuild_fix(leg))
    leg_hash = str(body.get("leg_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", leg_hash):
        raise _gate(unusable_cls, f"closure record {origin} without leg_sha256",
                    "the SHA-256 of the leg it was written for", _rebuild_fix(leg))
    libs = body.get("libraries")
    if not isinstance(libs, list):
        raise _gate(unusable_cls, f"closure record {origin} without a library list",
                    "libraries[]", _rebuild_fix(leg))
    entries: list[ClosureEntry] = []
    for i, lib in enumerate(libs):
        try:
            if bundle_root is not None and lib.get("bundle_path"):
                path = _inside(bundle_root, lib["bundle_path"],
                               field=f"closure {platform} libraries[{i}]",
                               unusable_cls=unusable_cls)
                mtime = 0
            else:
                path = Path(str(lib["path"]))
                if not path.is_absolute():
                    raise ValueError(f"relative path {path}")
                mtime = int(lib.get("mtime_ns", 0))
            sha = str(lib["sha256"])
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError("sha256 is not a full SHA-256")
            soname = str(lib["soname"])
            entries.append(ClosureEntry(
                soname=soname, path=path, sha256=sha, bytes=int(lib["bytes"]),
                mtime_ns=mtime, role=str(lib.get("role", "")),
                family=str(lib.get("family") or library_family(soname)),
                version=str(lib.get("version") or "")))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise _gate(unusable_cls, f"malformed entry {i} in closure record "
                        f"{origin} ({type(exc).__name__}: {exc})",
                        "soname, path, sha256 and bytes per library",
                        _rebuild_fix(leg)) from exc
    return ClosureRecord(origin=origin, platform=platform, leg=leg,
                         leg_sha256=leg_hash, entries=tuple(entries),
                         digest=closure_record_digest(body))


def closure_record_path(leg: str | Path) -> Path:
    leg = Path(leg)
    return leg.with_name(leg.name + CHECKOUT_CLOSURE_SUFFIX)


def read_checkout_closure(leg: Path, platform: str, *,
                          unusable_cls: Type[OSError] = LibraryUnusable,
                          build_hint: str = "") -> ClosureRecord:
    """The record an unsealed leg's build wrote beside it.  No record, an old
    schema, or a record for other bytes refuses: a leg whose dependencies were
    not recorded cannot be enforced, so it does not load."""
    side = closure_record_path(leg)
    if not side.is_file():
        raise _gate(unusable_cls, f"no closure record for {leg}",
                    f"{side.name} beside it, written by the leg's build",
                    _rebuild_fix(leg, build_hint))
    try:
        body = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _gate(unusable_cls, f"unreadable closure record {side} ({exc})",
                    "a JSON record", _rebuild_fix(leg, build_hint)) from exc
    record = parse_closure(body, origin=side, leg=leg, platform=platform,
                           unusable_cls=unusable_cls)
    actual = _sha256(leg)
    if actual != record.leg_sha256:
        raise _gate(unusable_cls, f"stale closure record {side}: it was written "
                    f"for a leg with SHA-256 {record.leg_sha256[:16]}, {leg} is "
                    f"{actual[:16]}", "a record written by the build that "
                    "produced these bytes", _rebuild_fix(leg, build_hint))
    return record


def check_record_files(record: ClosureRecord, *,
                       unusable_cls: Type[OSError] = LibraryUnusable,
                       build_hint: str = "", hasher=None) -> None:
    """Every recorded file is still the recorded one: present, same size, and
    (when its mtime moved) the same SHA-256.  Otherwise the record is stale."""
    hasher = hasher or _sha256
    for e in record.entries:
        try:
            st = e.path.stat()
        except OSError:
            raise _gate(unusable_cls, f"stale closure record {record.origin}: "
                        f"{e.soname} -> {e.path} no longer exists",
                        "every recorded file in place",
                        _rebuild_fix(record.leg, build_hint)) from None
        if st.st_size != e.bytes:
            raise _gate(unusable_cls, f"stale closure record {record.origin}: "
                        f"{e.path} is {st.st_size} bytes, recorded {e.bytes}",
                        "the recorded file unchanged",
                        _rebuild_fix(record.leg, build_hint))
        if e.mtime_ns and st.st_mtime_ns != e.mtime_ns and hasher(e.path) != e.sha256:
            raise _gate(unusable_cls, f"stale closure record {record.origin}: "
                        f"{e.path} changed content since the build",
                        f"SHA-256 {e.sha256[:16]}",
                        _rebuild_fix(record.leg, build_hint))


def _merge_expectations(records: Sequence[ClosureRecord],
                        unusable_cls: Type[OSError]
                        ) -> dict[str, tuple[ClosureEntry, ClosureRecord]]:
    """family -> the one (entry, record) the process must map, in load order.

    Two legs that record different files for one family refuse."""
    want: dict[str, tuple[ClosureEntry, ClosureRecord]] = {}
    for rec in records:
        for e in rec.entries:
            prior = want.get(e.family)
            if prior is None:
                want[e.family] = (e, rec)
                continue
            pe, prec = prior
            if _same_path(pe.path, e.path) or pe.sha256 == e.sha256:
                continue
            raise _gate(
                unusable_cls,
                f"{rec.leg.name} records {e.family} as {e.path} but "
                f"{prec.leg.name} (opened first) records {pe.path}",
                f"one {e.family} for the whole process",
                "build both legs against the same library set (one module "
                "stack, one venv), then reopen")
    return want


def _same_path(a: Path, b: Path) -> bool:
    return a == b or os.path.realpath(a) == os.path.realpath(b)


_SONAME_CACHE: dict[str, str] = {}


def _mapped_sonames() -> dict[Path, str]:
    """realpath -> DT_SONAME (file name when it has none) of every mapped object."""
    out: dict[Path, str] = {}
    for path in _mapped_paths():
        key = str(path)
        name = _SONAME_CACHE.get(key)
        if name is None:
            try:
                name = (_elf_dynamic(path).get(_DT_SONAME) or [path.name])[0]
            except (OSError, ValueError, struct.error, IndexError):
                name = path.name
            _SONAME_CACHE[key] = name
        out[path] = name
    return out


def check_closure_mapping(records: Sequence[ClosureRecord],
                          mapped: Mapping[Path, str], *,
                          unusable_cls: Type[OSError] = LibraryUnusable,
                          build_hint: str = "", hasher=None
                          ) -> dict[str, tuple[Path, str]]:
    """The gate.  ``mapped``: realpath -> SONAME of every mapped object.

    For every family the records name (merged in load order), exactly one
    object of that family must be mapped, and it must be the recorded file (by
    path, or byte-identical by SHA-256).  Returns family -> (mapped path,
    content SHA-256).  Pure: callers and tests supply the maps.
    """
    hasher = hasher or _sha256
    want = _merge_expectations(records, unusable_cls)
    by_family: dict[str, set[Path]] = {}
    for path, soname in mapped.items():
        family = library_family(soname)
        if family in want:
            by_family.setdefault(family, set()).add(Path(path))
    verified: dict[str, tuple[Path, str]] = {}
    refusals: list[OSError] = []
    for family, (entry, rec) in sorted(want.items()):
        copies = sorted(by_family.get(family, ()), key=str)
        if len(copies) > 1:
            refusals.append(_gate(
                unusable_cls,
                f"{len(copies)} copies of {family} mapped: "
                + ", ".join(str(p) for p in copies),
                f"only {entry.path} (recorded in {rec.origin})",
                f"something in this process loaded a second {family} before or "
                "beside LORRAX (an LD_PRELOAD, a module's library, a Python "
                "package such as mpi4py); remove it, or rebuild the leg against "
                f"that library set ({build_hint or 'config/<site>/build_ffi_<leg>.sh'})"))
            continue
        if not copies and entry.role == "attest":
            continue      # mapped by the JAX plugin later; enforced once it is
        if not copies:
            refusals.append(_gate(unusable_cls, f"no {family} mapped after opening "
                                  f"{rec.leg.name}",
                                  f"{entry.path} (recorded in {rec.origin})",
                                  _rebuild_fix(rec.leg, build_hint)))
            continue
        got = copies[0]
        if _same_path(got, entry.path):
            verified[family] = (got, entry.sha256)
            continue
        actual = hasher(got)
        if actual != entry.sha256:
            refusals.append(_gate(
                unusable_cls,
                f"{got} serving {entry.soname} (SHA-256 {actual[:16]})",
                f"{entry.path} (SHA-256 {entry.sha256[:16]}, recorded in "
                f"{rec.origin})",
                f"the wrong {family} was mapped before the recorded one could "
                "be: remove whatever loads it first (LD_PRELOAD, a module "
                "library, a Python package); LD_LIBRARY_PATH cannot fix this -- "
                "or, if the recorded file is what should change, "
                + _rebuild_fix(rec.leg, build_hint)))
            continue
        verified[family] = (got, actual)
    if len(refusals) == 1:
        raise refusals[0]
    if refusals:
        # every violation, one GATE line each, so the whole wrong set is named
        raise unusable_cls("\n".join(str(r) for r in refusals))
    return verified


def _record_identity(record: ClosureRecord,
                     verified: Mapping[str, tuple[Path, str]] | None) -> str:
    families = sorted({e.family for e in record.entries})
    first: dict[str, tuple[Path, str]] = {}
    for e in record.entries:
        first.setdefault(e.family, (e.path, e.sha256))
    verified = {**first, **(verified or {})}   # unmapped attest family: as recorded
    mapped = [[f, str(verified[f][0]), verified[f][1]] for f in families]
    return hashlib.sha256(_canonical({"record": record.digest,
                                      "mapped": mapped})).hexdigest()


def _record_content_identity(record: ClosureRecord,
                             verified: Mapping[str, tuple[Path, str]] | None) -> str:
    """:func:`_record_identity` without file locations: the record digest (whose
    library paths are bundle-relative inside a bundle) and, per family, the
    content hash of the file mapped (or recorded).  Two copies of one sealed
    bundle at different install paths -- a module candidate and its published
    release -- get the same value; different bytes never do.  This is the key a
    persistent cache needs: a compiled program depends on what the libraries
    are, not on where they live (a path a program does depend on, such as the
    cubin directory a custom call names, is in its HLO and so in JAX's own key)."""
    first: dict[str, str] = {}
    for e in record.entries:
        first.setdefault(e.family, e.sha256)
    shas = {**first, **{f: v[1] for f, v in (verified or {}).items()}}
    return hashlib.sha256(_canonical({
        "record": record.digest,
        "content": [[f, shas[f]] for f in sorted(first)]})).hexdigest()


def _expected_record(leg: Path, platform: str) -> ClosureRecord:
    manifest = _discover_manifest(leg)
    if manifest is not None:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        body = (doc.get("closures") or {}).get(platform)
        return parse_closure(body, origin=manifest, leg=leg, platform=platform,
                             bundle_root=manifest.parent)
    return read_checkout_closure(leg, platform)


def native_closure_content_identity_or_expected(platform: str,
                                                candidates: Sequence[Path] = ()) -> str:
    """The location-free closure identity (:func:`_record_content_identity`) of the
    enforced ``platform`` leg, else of its first existing candidate's record.
    Raises :class:`ClosureNotEnforced` when neither exists."""
    leg = _PLATFORM_LEG.get(platform)
    if leg is not None and leg in _ENFORCED:
        native_closure_identity(platform)        # the one late re-verification
        enf = _ENFORCED[leg]
        return _record_content_identity(enf.record, enf.verified)
    for c in candidates:
        if Path(c).is_file():
            return _record_content_identity(_expected_record(Path(c).resolve(), platform), None)
    raise ClosureNotEnforced(f"no {platform} leg is open and no candidate exists")


def expected_closure_identity(leg: str | Path, platform: str) -> str:
    """The identity a leg will have once enforced, from its record alone (for a
    cache keyed before the leg opens).  Raises when there is no valid record."""
    leg = Path(leg).resolve()
    st = leg.stat()
    manifest = _discover_manifest(leg)
    source = manifest if manifest is not None else closure_record_path(leg)
    try:
        record_bytes = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError:
        record_bytes = ""
    key = (str(leg), platform, st.st_size, st.st_mtime_ns, record_bytes)
    hit = _EXPECTED_CACHE.get(key)
    if hit is not None:
        return hit
    if manifest is not None:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        body = (doc.get("closures") or {}).get(platform)
        record = parse_closure(body, origin=manifest, leg=leg, platform=platform,
                               bundle_root=manifest.parent)
    else:
        record = read_checkout_closure(leg, platform)
    _EXPECTED_CACHE[key] = ident = _record_identity(record, None)
    return ident


_EXPECTED_CACHE: dict[tuple, str] = {}


@dataclass
class _Enforced:
    record: ClosureRecord
    identity: str
    verified: Mapping[str, tuple[Path, str]] = None    # type: ignore[assignment]


_ENFORCED: dict[Path, _Enforced] = {}          # resolved leg -> enforcement
_PLATFORM_LEG: dict[str, Path] = {}            # platform -> resolved leg
_REVERIFIED: list[bool] = [False]
#: Every leg in the order its ``dlopen`` succeeded, appended at that moment by
#: :func:`open_and_attest`, whichever loader (``ffi.common.ffi_loader``,
#: ``distrib_la.loader``) asked.  The SONAME a library answers for belongs to
#: the first object mapped under it, so this, not the order in which
#: enforcements finished (a leg can be dlopened, refused, and enforced on a
#: later call, after a leg opened behind it), is the order closures merge in.
_DLOPEN_ORDER: list[Path] = []


def _in_dlopen_order(records: Sequence[ClosureRecord]) -> list[ClosureRecord]:
    """``records`` in dlopen order; a leg not yet dlopened goes last (it is
    the one being opened)."""
    pos = {leg: i for i, leg in enumerate(_DLOPEN_ORDER)}
    return sorted(records, key=lambda r: pos.get(r.leg, len(pos)))


def _enforced_records() -> list[ClosureRecord]:
    return _in_dlopen_order([e.record for e in _ENFORCED.values()])


def _staged_copy(path: Path) -> Path:
    """The module's node-local copy of a venv file, when the lorrax module staged one
    (``config/perlmutter/module/launch/lorrax_stage_local.py``: ``LORRAX_STAGED_SITE``
    mirrors ``LORRAX_STAGE_VENV_SITE``); else ``path``.  JAX resolves ``nvidia`` to the
    copy, so the closure must open the copy too, or one family is mapped twice.  The
    gate still checks the copy is byte-identical to the recorded file."""
    staged = os.environ.get("LORRAX_STAGED_SITE")
    venv = os.environ.get("LORRAX_STAGE_VENV_SITE")
    if not (staged and venv):
        return path
    try:
        rel = Path(os.path.realpath(path)).relative_to(os.path.realpath(venv))
    except ValueError:
        return path
    cand = Path(staged) / rel
    return cand if cand.is_file() else path


def _preload_closure(record: ClosureRecord, unusable_cls: Type[OSError]) -> None:
    """Open the record's files by exact path, dependencies first, skipping a
    family an already-enforced leg answers for (merged by the same rule the gate
    applies).  Handles stay open for the life of the process."""
    key = f"closure:{record.leg}"
    if key in _PRIVATE_HANDLES:
        return
    _merge_expectations(_in_dlopen_order([*_enforced_records(), record]), unusable_cls)
    prior = _merge_expectations(_enforced_records(), unusable_cls)
    handles: list[ctypes.CDLL] = []
    for e in record.entries:
        if e.family in prior:
            continue      # already open (the same file, or yielded)
        path = _staged_copy(e.path)
        if e.role == "attest" and path == e.path:
            continue      # JAX's to map.  A staged copy is opened here: JAX maps that same
                          # file, and the leg's RUNPATH would otherwise bind the venv's.
        try:
            handles.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
        except OSError as exc:
            raise _gate(unusable_cls, f"{e.path} (recorded for {e.soname} in "
                        f"{record.origin}) failed to load: {exc}",
                        "every recorded library loadable", _rebuild_fix(record.leg)
                        ) from exc
    _PRIVATE_HANDLES[key] = tuple(handles)


def _enforce(record: ClosureRecord, unusable_cls: Type[OSError],
             build_hint: str) -> _Enforced:
    """The gate over every enforced leg plus ``record``, on the live maps."""
    if not sys.platform.startswith("linux"):
        raise _gate(unusable_cls, f"platform {sys.platform}",
                    "Linux (/proc/self/maps)", "LORRAX native legs run on Linux")
    mapped = _mapped_sonames()
    if not mapped:
        raise _gate(unusable_cls, "an unreadable /proc/self/maps",
                    "the process's mapped objects", "run on a Linux kernel with /proc")
    records = _in_dlopen_order(
        [r for r in _enforced_records() if r.leg != record.leg] + [record])
    verified = check_closure_mapping(records, mapped, unusable_cls=unusable_cls,
                                     build_hint=build_hint)
    mine = {e.family for e in record.entries}
    return _Enforced(record=record, identity=_record_identity(record, verified),
                     verified={f: v for f, v in verified.items() if f in mine})


def check_launch_libraries(names: Sequence[str], ld_library_path: str, *,
                           remedy: str = "",
                           unusable_cls: Type[OSError] = LibraryUnusable) -> None:
    """Refuse a launch whose site transport would map a second copy of a
    recorded library.

    ``names`` are the unversioned names the site transport dlopens through
    ``LD_LIBRARY_PATH`` (``site.toml`` ``launch_libraries``; Perlmutter:
    libfabric's ``libcudart.so``).  Such a name matches no loaded SONAME, so
    the loader maps whatever file it finds first; where an enforced leg
    records that library's family, the file must be the recorded one.  Call
    after the legs open and before the first collective.
    """
    recorded = {fam: path for e in _ENFORCED.values()
                for fam, (path, _sha) in (e.verified or {}).items()}
    dirs = [d for d in ld_library_path.split(":") if d]
    for name in names:
        hit = next((Path(d) / name for d in dirs if (Path(d) / name).is_file()), None)
        if hit is None:
            continue                  # runtime.network_env refuses an absent name
        real = hit.resolve()
        family = library_family((_elf_dynamic(real).get(_DT_SONAME) or [real.name])[0])
        want = recorded.get(family)
        if want is None or _same_file(real, Path(want)):
            continue
        raise _gate(unusable_cls,
                    f"{name} on LD_LIBRARY_PATH resolves to {real} ({hit.parent}), which "
                    f"the site transport dlopens: a second copy of {family}",
                    f"{name} -> the recorded {want}",
                    remedy or "put the recorded file under that name first on LD_LIBRARY_PATH")


def native_closure_identity(platform: str | None = None) -> str:
    """Stable SHA-256 over the enforced leg's recorded closure (paths and
    content hashes) and the files verified mapped for it.

    Computed when the leg's enforcement passes and cached; the first call
    re-runs the gate on the live maps once, so a library mapped after the leg
    opened (and before any cache is keyed) is still refused.  ``platform`` is a
    :data:`lxkit.legs.LEGS` key; ``None`` combines every enforced leg.  Raises
    :class:`ClosureNotEnforced` when the leg has not been opened.
    """
    if not _REVERIFIED[0] and _ENFORCED:
        mapped = _mapped_sonames()
        verified = check_closure_mapping(_enforced_records(), mapped)
        _REVERIFIED[0] = True
        for enf in _ENFORCED.values():
            late = {e.family: e for e in enf.record.entries
                    if e.role == "attest" and e.family in verified
                    and e.family not in (enf.verified or {})}
            if late:
                print(f"[lorrax native] closure {enf.record.platform} re-verified: "
                      + " ".join(_identity_item(e, verified[f]) for f, e in sorted(late.items())),
                      file=sys.stderr, flush=True)
    if platform is None:
        if not _ENFORCED:
            raise ClosureNotEnforced("no native leg has been opened and enforced")
        ids = sorted((rec.record.platform, rec.identity) for rec in _ENFORCED.values())
        return hashlib.sha256(_canonical(ids)).hexdigest()
    leg = _PLATFORM_LEG.get(platform)
    if leg is None or leg not in _ENFORCED:
        raise ClosureNotEnforced(
            f"the {platform} leg has not been opened and enforced in this process")
    return _ENFORCED[leg].identity


def native_closure_identity_or_expected(platform: str,
                                        candidates: Sequence[Path] = ()) -> str:
    """:func:`native_closure_identity` if the leg is open, else the identity
    its first existing candidate's record promises.  Raises when neither."""
    try:
        return native_closure_identity(platform)
    except ClosureNotEnforced:
        pass
    for c in candidates:
        if Path(c).is_file():
            return expected_closure_identity(c, platform)
    raise ClosureNotEnforced(f"no {platform} leg is open and no candidate exists")


def resolve_closure(leg: str | Path, *, private_dirs: Sequence[str | Path] = (),
                    process_dirs: Sequence[str | Path] = (),
                    library_dirs: Sequence[str | Path] = (), elf=None
                    ) -> tuple[list[dict], list[dict], list[str]]:
    """The build side: ``(libraries, external, refusals)`` of ``leg``.

    Each NEEDED name is resolved, in order, from ``private_dirs`` (engine-private
    SONAMEs only), ``process_dirs`` (what the JAX GPU plugin maps), the pinned
    ``library_dirs``, then the needing object's own DT_RPATH/DT_RUNPATH.
    Resolved objects are recorded dependency-first and descended into; the
    process/compiler/driver runtime and anything else unresolved is listed as
    external (not enforced).  A :data:`_MUST_PIN` family that resolves nowhere is
    a refusal: the build would leave it to the run shell.
    """
    elf = elf or _elf_dynamic
    leg = Path(leg).resolve()
    pdirs = [Path(d) for d in private_dirs]
    xdirs = [Path(d) for d in process_dirs]
    ldirs = [Path(d) for d in library_dirs]
    libraries: list[dict] = []
    external: list[dict] = []
    refusals: list[str] = []
    seen: set[str] = set()

    def search_dirs(obj: Path) -> tuple[list[Path], list[Path]]:
        """(DT_RPATH dirs, DT_RUNPATH dirs) of ``obj``, $ORIGIN expanded."""
        tags = elf(obj)
        out: tuple[list[Path], list[Path]] = ([], [])
        for tag, dst in ((_DT_RPATH, out[0]), (_DT_RUNPATH, out[1])):
            for entry in tags.get(tag, []):
                for d in entry.split(":"):
                    if d:
                        dst.append(Path(d.replace("${ORIGIN}", str(obj.parent))
                                        .replace("$ORIGIN", str(obj.parent))))
        return out

    def find(name: str, owner: Path, inherited: list[Path]
             ) -> tuple[Path, str] | None:
        order = []
        if is_private_redistributable(name):
            order += [(d, "private") for d in pdirs]
        order += [(d, "process") for d in xdirs] + [(d, "pinned") for d in ldirs]
        rpath, runpath = search_dirs(owner)
        # ld.so: an object without DT_RUNPATH also searches the DT_RPATH of the
        # objects that loaded it (a leg's RPATH serves its libraries' own dependencies).
        own = runpath if runpath else rpath + inherited
        order += [(d, "rpath") for d in own]
        for d, role in order:
            cand = d / name
            if cand.is_file():
                return cand.resolve(), role
        return None

    # Resolve breadth-first, as ld.so does: a name is resolved with the search
    # path of the FIRST object (in load order) that needs it, so a leg's direct
    # NEEDED use the leg's own RPATH even when a dependency's RUNPATH names
    # another copy.  Then record dependency-first (the preload order).
    from collections import deque
    resolved: dict[str, tuple[Path, str] | None] = {}
    leg_rpath, leg_runpath = search_dirs(leg)
    queue = deque((n, leg, [] if leg_runpath else leg_rpath)
                  for n in elf(leg).get(_DT_NEEDED, []))
    while queue:
        name, owner, inherited = queue.popleft()
        if name in seen:
            continue
        seen.add(name)
        if _BASE_RUNTIME.search(name) or _DRIVER.search(name):
            external.append({"soname": name, "why": "process/driver runtime"})
            resolved[name] = None
            continue
        hit = find(name, owner, inherited)
        resolved[name] = hit
        if hit is None:
            if _MUST_PIN.search(name):
                refusals.append(f"{name} (needed by {owner.name})")
            else:
                external.append({"soname": name, "why": "not in a recorded directory"})
            continue
        path = hit[0]
        rpath, runpath = search_dirs(path)
        below = inherited if runpath else rpath + inherited
        queue.extend((d, path, below) for d in elf(path).get(_DT_NEEDED, []))

    done: set[str] = set()

    def emit(name: str) -> None:
        if name in done or resolved.get(name) is None:
            return
        done.add(name)
        path, role = resolved[name]                         # type: ignore[misc]
        for dep in elf(path).get(_DT_NEEDED, []):
            emit(dep)
        if is_private_redistributable(name):
            role = "private"            # whichever directory served it
        st = path.stat()
        soname = (elf(path).get(_DT_SONAME) or [path.name])[0]
        libraries.append({
            "soname": name, "elf_soname": soname, "path": str(path),
            "sha256": _sha256(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
            "role": role, "family": library_family(soname)})

    for name in elf(leg).get(_DT_NEEDED, []):
        emit(name)
    return libraries, external, refusals


def _announce_loaded(path: Path, origin: Path,
                     att: BundleAttestation | None) -> None:
    if att is None:
        digest = _sha256(path)
        key = ("legacy", str(path))
        if key not in _ANNOUNCED:
            _ANNOUNCED.add(key)
            print(
                f"[lorrax native] LEGACY-UNSEALED origin={origin} "
                f"sha256={digest[:16]} pair/private-closure unattested; "
                "developer migration only", file=sys.stderr, flush=True)
        return
    key = ("sealed", att.bundle_id)
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        libs = ",".join(
            f"{name}={att.libraries[name]}#{att.library_hashes[name][:12]}"
            for name in sorted(att.libraries))
        print(
            f"[lorrax native] sealed bundle={att.bundle_id[:16]} "
            f"source={att.source_revision[:12]} abi={att.abi} {libs} "
            f"private={len(att.private_libraries)} actual={origin}",
            file=sys.stderr, flush=True)


def _announce_closure(enforced: "_Enforced") -> None:
    rec = enforced.record
    key = ("closure", str(rec.leg))
    if key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    print(f"[lorrax native] closure {rec.platform} enforced: "
          f"{len(rec.entries)} recorded libraries, one mapped copy each; "
          f"record={rec.digest[:12]} identity={enforced.identity[:12]} "
          f"({rec.origin.name})", file=sys.stderr, flush=True)
    items = []
    for e in rec.entries:
        got = (enforced.verified or {}).get(e.family)
        items.append(_identity_item(e, got) if got else f"{e.family}=<not yet mapped>")
    print(f"[lorrax native] closure {rec.platform} libraries: " + " ".join(items),
          file=sys.stderr, flush=True)


def _identity_item(e: "ClosureEntry", got: tuple[Path, str]) -> str:
    ver = f"[{e.version}]" if e.version else ""
    return f"{e.family}={os.path.realpath(got[0])}#{got[1][:12]}{ver}"


def _prefer_process_hdf5(
    *, unusable_cls: Type[OSError] = LibraryUnusable,
) -> bool:
    """Give an installed h5py's HDF5 symbols global precedence, if present.

    Both native legs are opened ``RTLD_GLOBAL`` and may link site parallel
    HDF5.  Importing h5py afterward can bind its extension against those
    ABI-incompatible symbols.  This lazy best-effort import is deliberately
    owned at the one common pre-dlopen boundary; lxkit still imports with no
    h5py (or JAX) dependency.
    """
    try:
        import h5py  # noqa: F401
    except ModuleNotFoundError as exc:
        # Standalone native consumers may genuinely omit h5py.  A missing
        # dependency *inside* an installed h5py is different: continuing to
        # RTLD_GLOBAL here would hide the broken Python/HDF5 environment and
        # can make the later failure depend on whichever HDF5 SONAME wins.
        if exc.name == "h5py":
            return False
        raise unusable_cls(
            "installed h5py is unusable before the native RTLD_GLOBAL load: "
            f"its dependency {exc.name!r} is missing.  Refusing at the shared "
            "HDF5 load-order boundary rather than allowing an incompatible "
            "site/parallel HDF5 provider to mask the import failure.") from exc
    except Exception as exc:
        raise unusable_cls(
            "installed h5py is unusable before the native RTLD_GLOBAL load: "
            f"{type(exc).__name__}: {exc}.  This commonly indicates a Python/"
            "HDF5 ABI or symbol-closure conflict; repair that environment "
            "before loading a LORRAX native provider.") from exc
    return True


def open_and_attest(
    path: Path,
    *,
    platform: str,
    expected_abi: int,
    abi_symbols: Mapping[str, str],
    build_hint: str,
    strict_unstamped: bool = False,
    unusable_cls: Type[OSError] = LibraryUnusable,
    mismatch_cls: Type[OSError] = NativeAbiMismatch,
) -> tuple[ctypes.CDLL, Path]:
    """Dlopen one preflighted file and attest its live origin and ABI."""
    _prefer_process_hdf5(unusable_cls=unusable_cls)
    selected = path.resolve()
    manifest = _discover_manifest(selected)
    att = (_read_bundle(manifest, expected_abi=expected_abi,
                        unusable_cls=unusable_cls)
           if manifest is not None else None)
    if att is not None and not _same_file(selected, att.libraries[platform]):
        raise unusable_cls(
            f"selected {platform} library {selected} is not the file bound by "
            f"{att.manifest_path}: {att.libraries[platform]}")
    if att is not None:
        record = att.closures[platform]
    else:
        record = read_checkout_closure(selected, platform,
                                       unusable_cls=unusable_cls,
                                       build_hint=build_hint)
    check_record_files(record, unusable_cls=unusable_cls, build_hint=build_hint)
    _preload_closure(record, unusable_cls)
    try:
        lib = ctypes.CDLL(str(selected), mode=ctypes.RTLD_GLOBAL)
        if selected not in _DLOPEN_ORDER:
            _DLOPEN_ORDER.append(selected)
    except OSError as exc:
        raise unusable_cls(
            f"{selected} exists but could not be loaded: {exc}.  This is a "
            "BROKEN BUILD OR ENVIRONMENT, not an absent library; inspect its "
            "runtime dependency closure.") from exc
    enforced = _enforce(record, unusable_cls, build_hint)
    assert_one_mapped_mpi_runtime(unusable_cls=unusable_cls)
    check_abi(lib, platform, str(selected), expected_abi=expected_abi,
              abi_symbols=abi_symbols, build_hint=build_hint,
              strict_unstamped=strict_unstamped or att is not None,
              mismatch_cls=mismatch_cls)
    symbol = abi_symbols[platform]
    origin = selected
    if hasattr(lib, symbol):
        try:
            origin = _actual_symbol_origin(lib, symbol)
        except LibraryUnusable as exc:
            raise unusable_cls(str(exc)) from exc
        if not _same_file(origin, selected):
            raise unusable_cls(
                f"native provider origin mismatch: requested {selected}, but "
                f"the live {symbol} came from {origin}.  A stale provider with "
                "the same SONAME was already mapped; restart with one bundle.")
    if att is not None:
        opened = {platform} | {p for p, leg in _PLATFORM_LEG.items()
                               if p in att.libraries and _same_path(leg, att.libraries[p])}
        _attest_private_mapping(att, unusable_cls, sorted(opened))
    _ENFORCED[selected] = enforced
    _PLATFORM_LEG[platform] = selected
    _announce_loaded(selected, origin, att)
    _announce_closure(enforced)
    return lib, origin


def _closure_note(path: Path) -> str:
    enforced = _ENFORCED.get(path)
    if enforced is None:
        return " | closure not enforced in this process"
    return (f" | closure {enforced.record.digest[:12]} enforced, identity "
            f"{enforced.identity[:12]} ({len(enforced.record.entries)} libraries)")


def describe_library(path: str | Path) -> str:
    """Compact immutable identity for startup reports; never raises."""
    p = Path(path).resolve()
    try:
        manifest = _discover_manifest(p)
        if manifest is not None:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            return (f"{p} | sealed bundle {str(doc.get('bundle_id', '?'))[:16]}"
                    f" | rev {str(doc.get('source', {}).get('revision', '?'))[:12]}"
                    f" | sha {_sha256(p)[:16]}{_closure_note(p)}")
        actual_hash = _sha256(p)
        stamp = p.parent / "PROVENANCE"
        recorded = ""
        if stamp.is_file():
            fields = {}
            for line in stamp.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, _, value = line.partition("=")
                    fields[key.strip()] = value.strip()
            recorded = (f" | recorded-rev {fields.get('git_rev', '?')[:12]}"
                        f" | recorded-sha {fields.get('sha256', '?')[:16]}")
        return (f"{p} | LEGACY-UNSEALED{recorded} | "
                f"actual {p.stat().st_size} bytes sha {actual_hash[:16]}"
                f"{_closure_note(p)}")
    except Exception as exc:                                   # noqa: BLE001
        return f"{p} | native identity unmeasurable ({type(exc).__name__}: {exc})"


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m lxkit.closure verify|identity <leg.so> [platform]``.

    ``verify`` checks a leg's record against its bytes and the recorded files
    (no load) and prints the identity the record promises; exit 3 on a refusal.
    Build-key tooling (e.g. a site's FFI source-key resolver) uses it so a
    build whose recorded library set is gone is never reused.
    """
    import argparse
    ap = argparse.ArgumentParser(prog="python -m lxkit.closure")
    ap.add_argument("action", choices=("verify", "identity"))
    ap.add_argument("leg", type=Path)
    ap.add_argument("platform", nargs="?")
    ns = ap.parse_args(argv)
    leg = ns.leg.resolve()
    platform = ns.platform or next(
        (k for k, spec in LEGS.items() if spec.so_name == leg.name), None)
    if platform is None:
        print(f"cannot infer the platform of {leg.name}; pass it", file=sys.stderr)
        return 2
    try:
        manifest = _discover_manifest(leg)
        if manifest is not None:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            if platform not in (doc.get("closures") or {}):
                raise _gate(LibraryUnusable, f"bundle manifest {manifest} (schema "
                            f"{doc.get('schema')}) with no {platform} closure",
                            "a schema-2 manifest carrying each leg's record",
                            "reseal with this tree's seal_bundle.py from legs whose "
                            "builds wrote <leg>.closure.json")
            record = parse_closure(doc["closures"][platform],
                                   origin=manifest, leg=leg, platform=platform,
                                   bundle_root=manifest.parent)
        else:
            record = read_checkout_closure(leg, platform)
        check_record_files(record)
    except LibraryUnusable as exc:
        print(str(exc), file=sys.stderr)
        return 3
    if ns.action == "identity":
        print(_record_identity(record, None))
    else:
        print(f"OK {platform} record={record.digest} "
              f"identity={_record_identity(record, None)} "
              f"libraries={len(record.entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
