#!/usr/bin/env python3
"""Create one immutable, relocatable two-leg LORRAX FFI bundle.

Usage::

    python src/ffi/cpp/stage/seal_bundle.py \
      --cuda path/to/liblorrax_ffi.so \
      --host path/to/liblorrax_ffi_host.so \
      --output path/to/lorrax-ffi-<revision> \
      --private-lib path/to/libblaspp.so.2 \
      --private-lib path/to/libslate.so.2

The output directory must not exist.  Both provider legs and every explicitly
listed private redistributable library are copied as regular files beneath
``lib/``; the JSON manifest is written last and hashes every byte.  MPI, site
HDF5, CUDA runtime/driver libraries and NCCL are refused as private inputs:
those are supplied and attested by the machine runtime, never vendored here.

Each leg's build wrote ``<leg>.closure.json`` (``write_checkout_closure.py``):
the exact library set it resolved and verified.  The manifest carries both
records under ``closures`` (private entries rewritten to the sealed copies,
which must be byte-identical to the recorded ones), so the runtime enforces the
same closure for the sealed pair as for the checkout it came from.

This is intentionally a separate pair-sealing step rather than another CMake
post-build action.  List private libraries dependency-first; the shared
runtime loader opens their exact paths in this order, so no run-script
``LD_LIBRARY_PATH`` is needed.  The CUDA and host legs are often built in
different environments and neither build can certify a pair it cannot see.
Each build's existing acceptance contract remains necessary; this tool makes
the deployable unit only after both accepted artifacts exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[4]
_APPLICATION_SOURCE = _SOURCE_ROOT / "src"
if str(_APPLICATION_SOURCE) not in sys.path:
    sys.path.insert(0, str(_APPLICATION_SOURCE))

from runtime.source_closure import ensure_source_closure  # noqa: E402

ensure_source_closure()

from lxkit import native_provider as _native  # noqa: E402

MANIFEST = "lorrax_ffi_bundle.json"
SCHEMA = 2      # 2: per-leg dependency closures (lxkit.native_provider)
# Search-path directories a published bundle may not name: purge-eligible
# scratch, and the trees one build writes.
_SCRATCH_ROOTS = ("/pscratch", "/scratch", "/tmp", "/var/tmp", "/dev/shm")
_BUILD_PARTS = frozenset({".build", "build", "build_host", "CMakeFiles"})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _entry(path: Path, relative: Path) -> dict[str, object]:
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _elf_dynamic(path: Path) -> tuple[str, tuple[str, ...], list[tuple[str, str]]]:
    """Return the one SONAME, the exact DT_NEEDED names and the
    (DT_RPATH | DT_RUNPATH, path list) entries of an ELF provider."""
    try:
        result = subprocess.run(
            ["readelf", "-d", str(path)], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "readelf is required to seal the native dependency closure") from exc
    if result.returncode:
        raise RuntimeError(
            f"readelf could not inspect native provider {path}: "
            f"{result.stdout.strip()}")
    sonames = re.findall(r"\(SONAME\).*?\[([^]]+)\]", result.stdout)
    if len(sonames) != 1:
        raise RuntimeError(
            f"{path} must carry exactly one ELF SONAME; found {sonames or '<none>'}")
    needed = tuple(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", result.stdout))
    search = re.findall(r"\((RPATH|RUNPATH)\).*?\[([^]]*)\]", result.stdout)
    return sonames[0], needed, search


def _refuse_search_path(name: str, search: list[tuple[str, str]], *,
                        root: Path, leg: bool) -> None:
    """Refuse a search path a published bundle cannot carry.

    Every library: no relative directory (it resolves against the working
    directory), none under scratch (purge-eligible) and none inside a build
    tree or the sealing checkout.  A leg: exactly one DT_RUNPATH, starting with
    $ORIGIN, so its private providers come from the bundle and LD_LIBRARY_PATH
    (the site module) outranks every other entry.
    """
    scratch = [Path(r) for r in (*_SCRATCH_ROOTS, *(os.environ.get(v, "")
               for v in ("SCRATCH", "PSCRATCH", "TMPDIR"))) if r]
    dirs = [d for _, entry in search for d in entry.split(":") if d]
    for d in dirs:
        if d.startswith(("$ORIGIN", "${ORIGIN}")):
            continue
        forms = {Path(d), Path(os.path.realpath(d))}
        if (not Path(d).is_absolute()
                or any(f.is_relative_to(s) for f in forms for s in scratch)
                or any(_BUILD_PARTS & set(f.parts) or f.is_relative_to(root)
                       for f in forms)):
            raise RuntimeError(
                f"{name} searches {d!r}: a relative, scratch or build directory "
                "cannot ship in a bundle; link with $ORIGIN and leave the "
                "machine libraries to the run environment's LD_LIBRARY_PATH")
    if leg and ([tag for tag, _ in search] != ["RUNPATH"]
                or dirs[:1] not in (["$ORIGIN"], ["${ORIGIN}"])):
        raise RuntimeError(
            f"{name} must carry one DT_RUNPATH that starts with $ORIGIN "
            f"(--enable-new-dtags); it has {search or 'none'}")


def _bundle_id(doc: dict[str, object]) -> str:
    body = dict(doc)
    body.pop("bundle_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def _abi_from_header(root: Path) -> int:
    header = root / "src" / "ffi" / "cpp" / "common" / "lorrax_ffi_abi.h"
    match = re.search(
        r"^#define\s+LORRAX_FFI_ABI_VERSION\s+(\d+)",
        header.read_text(encoding="utf-8"), re.M)
    if match is None:
        raise RuntimeError(f"{header} does not define LORRAX_FFI_ABI_VERSION")
    return int(match.group(1))


def _regular_input(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not a regular file: {resolved}")
    return resolved


def _verified_build_identity(path: Path, revision: str) -> None:
    stamp = path.parent / "PROVENANCE"
    if not stamp.is_file():
        raise RuntimeError(
            f"{path} has no adjacent PROVENANCE stamp.  Run the supported "
            "build/acceptance path; an ELF with no source identity cannot enter "
            "a production bundle")
    fields: dict[str, str] = {}
    for line in stamp.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    if fields.get("git_dirty") != "no":
        raise RuntimeError(
            f"{path} provenance says git_dirty={fields.get('git_dirty', '<missing>')}; "
            "rebuild both legs from the clean source revision being sealed")
    if fields.get("git_rev") != revision:
        raise RuntimeError(
            f"{path} provenance says source {fields.get('git_rev', '<missing>')}, "
            f"but the bundle source is {revision}; never mix provider/source builds")
    actual_hash = _sha256(path)
    if fields.get("sha256") != actual_hash:
        raise RuntimeError(
            f"{path} changed after its build receipt: PROVENANCE says "
            f"{fields.get('sha256', '<missing>')}, actual SHA-256 is {actual_hash}")


def _copy_regular(source: Path, destination: Path) -> Path:
    # ``copy2`` follows an input SONAME symlink and writes one regular file at
    # the exact name the dynamic loader asks for.  Runtime attestation refuses
    # mutable symlinks in the sealed output.
    shutil.copy2(source, destination, follow_symlinks=True)
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError(f"sealer did not produce a regular file: {destination}")
    return destination


def _publishable(path: Path, root: Path) -> bool:
    """A machine/process file a published bundle may depend on: absolute, not
    under scratch, not inside a build tree or the sealing checkout."""
    scratch = [Path(r) for r in (*_SCRATCH_ROOTS, *(os.environ.get(v, "")
               for v in ("SCRATCH", "PSCRATCH", "TMPDIR"))) if r]
    forms = {path, Path(os.path.realpath(path))}
    return not (any(f.is_relative_to(s) for f in forms for s in scratch)
                or any(_BUILD_PARTS & set(f.parts) or f.is_relative_to(root)
                       for f in forms))


def _sealed_closure(leg: Path, platform: str, sealed_private: dict[str, Path],
                    root: Path, *, publishable=None) -> dict[str, object]:
    """The leg's build record, rewritten for the bundle (manifest ``closures``)."""
    publishable = publishable or _publishable
    record = _native.read_checkout_closure(leg, platform, unusable_cls=RuntimeError)
    _native.check_record_files(record, unusable_cls=RuntimeError)
    body = json.loads(_native.closure_record_path(leg).read_text(encoding="utf-8"))
    libraries = []
    for lib, entry in zip(body["libraries"], record.entries):
        lib = {k: v for k, v in lib.items() if k != "mtime_ns"}
        if entry.role == "private":
            copy = sealed_private.get(entry.soname)
            if copy is None:
                raise RuntimeError(
                    f"{leg.name}'s recorded closure needs private {entry.soname} "
                    f"({entry.path}), which was not sealed; add --private-lib")
            if _sha256(copy) != entry.sha256:
                raise RuntimeError(
                    f"sealed {copy.name} is not the file {leg.name} was built and "
                    f"verified against ({entry.path}, SHA-256 {entry.sha256[:16]})")
            lib.pop("path", None)
            lib["bundle_path"] = (Path("lib") / copy.name).as_posix()
        elif not publishable(entry.path, root):
            raise RuntimeError(
                f"{leg.name}'s closure maps {entry.path} ({entry.soname}): a "
                "scratch or build-tree file cannot back a published bundle; build "
                "against the published runtime")
        libraries.append(lib)
    return {"schema": body["schema"], "platform": platform,
            "leg_sha256": body["leg_sha256"], "libraries": libraries,
            "external": body.get("external", [])}


def seal(*, root: Path, cuda: Path, host: Path, output: Path,
         private: list[Path]) -> Path:
    root = root.resolve(strict=True)
    cuda = _regular_input(cuda, "CUDA leg")
    host = _regular_input(host, "host leg")
    if cuda.name != "liblorrax_ffi.so":
        raise RuntimeError(
            f"CUDA leg must be named liblorrax_ffi.so, got {cuda.name!r}")
    if host.name != "liblorrax_ffi_host.so":
        raise RuntimeError(
            f"host leg must be named liblorrax_ffi_host.so, got {host.name!r}")
    revision = _git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"git returned a non-full source revision: {revision!r}")
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
    if dirty:
        raise RuntimeError(
            f"refusing to seal from dirty source tree {root}; commit the exact "
            "source that produced both libraries first")
    abi = _abi_from_header(root)
    _verified_build_identity(cuda, revision)
    _verified_build_identity(host, revision)
    cuda_soname, cuda_needed, cuda_search = _elf_dynamic(cuda)
    host_soname, host_needed, host_search = _elf_dynamic(host)
    _refuse_search_path(cuda.name, cuda_search, root=root, leg=True)
    _refuse_search_path(host.name, host_search, root=root, leg=True)
    if cuda_soname != cuda.name or host_soname != host.name:
        raise RuntimeError(
            "provider-leg SONAMEs must equal their canonical filenames: "
            f"CUDA={cuda_soname!r}, host={host_soname!r}")

    private_resolved: list[tuple[Path, str, tuple[str, ...]]] = []
    names = {cuda.name, host.name}
    for candidate in private:
        item = _regular_input(candidate, "private library")
        _native.refuse_private_library(item, RuntimeError)
        soname, needed, search = _elf_dynamic(item)
        _native.refuse_private_library(Path(soname), RuntimeError)
        _refuse_search_path(soname, search, root=root, leg=False)
        if soname in names:
            raise RuntimeError(
                f"duplicate bundle SONAME {soname!r}; one SONAME must have "
                "one provider origin")
        names.add(soname)
        private_resolved.append((item, soname, needed))

    private_names = {soname for _, soname, _ in private_resolved}
    owners = [(cuda.name, cuda_needed), (host.name, host_needed)] + [
        (soname, needed) for _, soname, needed in private_resolved
    ]
    for owner, needed in owners:
        for dependency in needed:
            if (_native.is_private_redistributable(dependency)
                    and dependency not in private_names):
                raise RuntimeError(
                    f"incomplete private dependency closure: {owner} NEEDS "
                    f"{dependency}, but that SONAME was not supplied with "
                    "--private-lib")

    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"immutable bundle output already exists: {output}; choose a new "
            "content/revision-named directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-",
                                  dir=str(output.parent)))
    try:
        libdir = stage / "lib"
        libdir.mkdir()
        cuda_out = _copy_regular(cuda, libdir / cuda.name)
        host_out = _copy_regular(host, libdir / host.name)
        private_out = [
            (_copy_regular(item, libdir / soname), soname, needed)
            for item, soname, needed in private_resolved
        ]
        sealed_private = {soname: out for out, soname, _ in private_out}
        closures = {"CUDA": _sealed_closure(cuda, "CUDA", sealed_private, root),
                    "cpu": _sealed_closure(host, "cpu", sealed_private, root)}
        recorded = {lib["soname"] for body in closures.values()
                    for lib in body["libraries"] if lib.get("bundle_path")}
        unrecorded = sorted(set(sealed_private) - recorded)
        if unrecorded:
            raise RuntimeError(
                f"--private-lib {unrecorded} is in neither leg's recorded closure; "
                "a sealed provider nothing records would never be opened")
        doc: dict[str, object] = {
            "schema": SCHEMA,
            "closures": closures,
            "source": {"revision": revision, "dirty": False},
            "ffi_abi": abi,
            "libraries": {
                "CUDA": {**_entry(cuda_out, Path("lib") / cuda_out.name),
                         "soname": cuda_soname,
                         "needed": list(cuda_needed)},
                "cpu": {**_entry(host_out, Path("lib") / host_out.name),
                        "soname": host_soname,
                        "needed": list(host_needed)},
            },
            "private_libraries": [
                {**_entry(item, Path("lib") / item.name), "soname": soname,
                 "needed": list(needed)}
                for item, soname, needed in private_out
            ],
        }
        doc["bundle_id"] = _bundle_id(doc)
        manifest = stage / MANIFEST
        manifest.write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")

        # Mutation is detected by hashes even by the owner; read-only modes
        # make the intended write-once lifecycle unmistakable to everyone else.
        for item in libdir.iterdir():
            item.chmod(0o444)
        manifest.chmod(0o444)
        libdir.chmod(0o555)
        stage.chmod(0o555)
        os.rename(stage, output)  # same-filesystem atomic publication
    except Exception:
        try:
            stage.chmod(0o755)
            if (stage / "lib").exists():
                (stage / "lib").chmod(0o755)
                for item in (stage / "lib").iterdir():
                    item.chmod(0o644)
            if (stage / MANIFEST).exists():
                (stage / MANIFEST).chmod(0o644)
            shutil.rmtree(stage)
        except OSError:
            pass
        raise
    return output / MANIFEST


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   help="clean LORRAX source root (default: inferred)")
    p.add_argument("--cuda", required=True, type=Path)
    p.add_argument("--host", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--private-lib", action="append", default=[], type=Path,
        help="private redistributable provider, dependency-first (repeatable)")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = _parser().parse_args(argv)
    root = ns.root
    if root is None:
        root = _SOURCE_ROOT
    try:
        manifest = seal(root=root, cuda=ns.cuda, host=ns.host,
                        output=ns.output, private=ns.private_lib)
    except Exception as exc:  # one named refusal, no partial published bundle
        print(f"[seal ffi bundle] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"[seal ffi bundle] {manifest}")
    print(manifest.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
