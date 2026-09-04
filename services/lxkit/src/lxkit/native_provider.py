"""Shared selection and attestation for LORRAX native provider bundles.

``lxkit`` owns policy, never a provider's symbol tables.  Callers supply the
two platform records (file name, existing pin, build hint and candidates);
this module supplies the one decision procedure used by both
``ffi.common.ffi_loader`` and the independently installable
``distrib_la.loader``.

A sealed deployment is one relocatable directory containing both platform
libraries and one ``lorrax_ffi_bundle.json``.  The manifest binds both FFI
files, the handler ABI, the source revision and every private redistributable
dependency by size and SHA-256.  MPI, site HDF5, CUDA runtime libraries and
NCCL are deliberately outside that closure.  Their identity belongs to the
machine runtime and is checked live, not copied into a Python package.

Unsealed build-tree artifacts remain loadable during migration, but every
such load prints ``LEGACY-UNSEALED`` together with the actual file hash.  A
manifest is never advisory: once either selected leg claims one, a partial
pin, a mixed provider, a missing file, a changed byte, a wrong ABI or a wrong
``dladdr`` origin refuses before target registration.

PINNED PROPERTY: stdlib only.  This module is safe before JAX or MPI startup.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Type

from lxkit.probe import LibraryNotBuilt, LibraryUnusable

__all__ = [
    "BUNDLE_MANIFEST", "BundleAttestation", "NativeAbiMismatch",
    "assert_one_mapped_mpi_runtime", "check_abi", "describe_library",
    "is_private_redistributable", "locate_library", "open_and_attest",
    "process_can_use_cuda",
    "refuse_private_library",
]

BUNDLE_MANIFEST = "lorrax_ffi_bundle.json"
_MANIFEST_SCHEMA = 1
_PLATFORMS = frozenset(("CUDA", "cpu"))
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
    re.compile(r"^ld-linux", re.I),
    re.compile(
        r"^lib(?:c|m|dl|pthread|rt|util|resolv|crypt|gcc_s|stdc[+][+]|gomp|"
        r"atomic)[.]", re.I),
)
_PRIVATE_REDISTRIBUTABLE = re.compile(
    r"^lib(?:cusolverMp|cublasmp|cal|slate(?:_scalapack_api)?|blaspp|"
    r"lapackpp|nvshmem(?:_host)?)[.]", re.I)


def is_private_redistributable(path: str | Path) -> bool:
    """Whether a basename belongs to the approved engine-private closure."""
    return bool(_PRIVATE_REDISTRIBUTABLE.search(Path(path).name))


def _refuse_runtime_owned(path: Path, unusable_cls: Type[OSError]) -> None:
    name = path.name
    if any(rx.search(name) for rx in _RUNTIME_OWNED):
        raise unusable_cls(
            f"sealed native bundle lists runtime-owned {name!r} as a private "
            "library.  MPI, site HDF5, CUDA runtime/driver libraries and NCCL "
            "must come from the selected machine runtime; never copy them into "
            "a LORRAX bundle.")


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
            "engine redistributables (cuSolverMp/cuBLASMp/CAL, SLATE/BLAS++/"
            "LAPACK++, or NVSHMEM); process, compiler, site and accelerator "
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
    if not isinstance(records, dict) or set(records) != _PLATFORMS:
        raise unusable_cls(
            f"native bundle manifest {manifest} must bind exactly the CUDA and "
            f"cpu legs; got {sorted(records) if isinstance(records, dict) else records}")
    root = manifest.parent.resolve()
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for platform in sorted(_PLATFORMS):
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
    return BundleAttestation(
        manifest_path=manifest, bundle_root=root, bundle_id=bundle_id,
        source_revision=revision, abi=abi, libraries=paths,
        library_hashes=hashes, private_libraries=tuple(private))


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
            "mixed native providers: selected CUDA and cpu legs claim different "
            f"bundle manifests: {sorted(str(p) for p in claimed)}")
    if pinned_platforms and pinned_platforms != _PLATFORMS:
        raise unusable_cls(
            "partial sealed-bundle override refused: LORRAX_FFI_SO and "
            "LORRAX_FFI_HOST_SO must either both be unset or both select the "
            f"same two-leg bundle; pinned platforms={sorted(pinned_platforms)}")

    att = _read_bundle(next(iter(claimed)), expected_abi=expected_abi,
                       unusable_cls=unusable_cls)
    for name in sorted(_PLATFORMS):
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


_MPI_RUNTIME_BASENAME = re.compile(r"^libmpi(?:_gnu_[0-9]+)?[.]so(?:[.]|$)")


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


def _attest_private_mapping(att: BundleAttestation,
                            unusable_cls: Type[OSError]) -> None:
    mapped = _mapped_paths()
    if not mapped:  # non-Linux diagnostic environment
        return
    by_name: dict[str, set[Path]] = {}
    for path in mapped:
        by_name.setdefault(path.name, set()).add(path)
    for expected in att.private_libraries:
        actual = by_name.get(expected.name, set())
        if expected not in actual or len(actual) != 1:
            raise unusable_cls(
                f"private provider origin mismatch for {expected.name}: sealed "
                f"bundle requires {expected}, live mappings are "
                f"{sorted(str(p) for p in actual) or '<none>'}")
    expected_private = set(att.private_libraries)
    unlisted = sorted(
        str(path) for path in mapped
        if is_private_redistributable(path) and path not in expected_private)
    if unlisted:
        raise unusable_cls(
            "sealed native bundle has an engine-private provider mapped "
            "outside its manifest: " + ", ".join(unlisted) + ".  A stale "
            "SLATE/BLAS++/cuSolverMp/cuBLASMp/CAL/NVSHMEM provider can win by "
            "SONAME even when the requested FFI file itself is exact; restart "
            "with only the bundle's declared private closure.")
    for path in mapped:
        try:
            path.relative_to(att.bundle_root)
        except ValueError:
            continue
        _refuse_runtime_owned(path, unusable_cls)


def _preload_private_closure(att: BundleAttestation,
                             unusable_cls: Type[OSError]) -> None:
    """Load exact private providers in manifest order, once per bundle.

    This is what makes the bundle self-localizing without teaching a module or
    run script an ``LD_LIBRARY_PATH``.  The sealer's ``--private-lib`` order is
    dependency-first; each absolute load publishes that provider's SONAME for
    the next library and, finally, the FFI leg.  Handles stay alive for the
    process lifetime.
    """
    if att.bundle_id in _PRIVATE_HANDLES:
        return
    handles: list[ctypes.CDLL] = []
    for path in att.private_libraries:
        try:
            handles.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
        except OSError as exc:
            raise unusable_cls(
                f"sealed private provider {path} could not be loaded: {exc}.  "
                "Private libraries must be listed dependency-first and every "
                "non-machine dependency must be included in the bundle.") from exc
    _PRIVATE_HANDLES[att.bundle_id] = tuple(handles)


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
            for name in ("CUDA", "cpu"))
        print(
            f"[lorrax native] sealed bundle={att.bundle_id[:16]} "
            f"source={att.source_revision[:12]} abi={att.abi} {libs} "
            f"private={len(att.private_libraries)} actual={origin}",
            file=sys.stderr, flush=True)


def _prefer_process_hdf5() -> bool:
    """Give an installed h5py's HDF5 symbols global precedence, if present.

    Both native legs are opened ``RTLD_GLOBAL`` and may link site parallel
    HDF5.  Importing h5py afterward can bind its extension against those
    ABI-incompatible symbols.  This lazy best-effort import is deliberately
    owned at the one common pre-dlopen boundary; lxkit still imports with no
    h5py (or JAX) dependency.
    """
    try:
        import h5py  # noqa: F401
    except Exception:  # h5py is optional for standalone native consumers
        return False
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
    _prefer_process_hdf5()
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
        _preload_private_closure(att, unusable_cls)
    try:
        lib = ctypes.CDLL(str(selected), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise unusable_cls(
            f"{selected} exists but could not be loaded: {exc}.  This is a "
            "BROKEN BUILD OR ENVIRONMENT, not an absent library; inspect its "
            "runtime dependency closure.") from exc
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
        _attest_private_mapping(att, unusable_cls)
    _announce_loaded(selected, origin, att)
    return lib, origin


def process_can_use_cuda(*, environ: Mapping[str, str] | None = None,
                         device_visible=None) -> bool:
    """Whether this process can host a CUDA handler, without importing JAX."""
    env = os.environ if environ is None else environ
    first = env.get("JAX_PLATFORMS", "").split(",")[0].strip().lower()
    if first and first not in ("cuda", "gpu"):
        return False
    cvd = env.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() == "":
        return False
    if device_visible is None:
        device_visible = lambda: (any(Path("/dev").glob("nvidia[0-9]*"))
                                  or Path("/dev/nvidiactl").exists())
    return bool(device_visible())


def describe_library(path: str | Path) -> str:
    """Compact immutable identity for startup reports; never raises."""
    p = Path(path).resolve()
    try:
        manifest = _discover_manifest(p)
        if manifest is not None:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            return (f"{p} | sealed bundle {str(doc.get('bundle_id', '?'))[:16]}"
                    f" | rev {str(doc.get('source', {}).get('revision', '?'))[:12]}"
                    f" | sha {_sha256(p)[:16]}")
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
                f"actual {p.stat().st_size} bytes sha {actual_hash[:16]}")
    except Exception as exc:                                   # noqa: BLE001
        return f"{p} | native identity unmeasurable ({type(exc).__name__}: {exc})"
