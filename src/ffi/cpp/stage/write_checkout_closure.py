#!/usr/bin/env python3
"""Write ``<leg>.closure.json``: the dependency closure a leg was built and verified against.

    write_checkout_closure.py <leg.so> [--private-dir DIR ...]
        [--library-dirs A:B:...] [--python VENV_PYTHON] [--process-dir DIR ...]

Every build of every leg writes this record (both machines, both device legs and
the host leg; ``seal_bundle.py`` copies it into a sealed bundle's manifest), and
``lxkit.native_provider.open_and_attest`` enforces it at start-up: the recorded
files are opened by exact path, and every recorded family must then be mapped
exactly once, as the recorded file.  See ``docs/installation/ffi-build.md``.

Each NEEDED name of the leg, transitively, is resolved in this order:

1. ``--private-dir``: the engine-private providers (cuSOLVERMp/cuBLASMp stage),
   by SONAME;
2. the process directories: the RPATH/RUNPATH directories of the leg's JAX
   PJRT plugin (``lxkit.legs.LEGS[...].jax_plugin_prefix``) in ``--python``'s
   site-packages, plus any ``--process-dir``.  The JAX plugin maps its GPU
   math and collective libraries (cuBLAS/cuSOLVER/cuFFT/NCCL, or
   rocBLAS/rocSOLVER/hipBLASLt/RCCL) from there before any leg opens, so the
   leg must bind the same files: one copy per process;
3. ``--library-dirs``: the pinned machine directories the build's verify step
   runs against (MPI, parallel HDF5, LibSci; the toolkit's math libraries where
   the venv carries no GPU wheels, as in the module venv);
4. the needing object's own DT_RPATH/DT_RUNPATH.

The process/compiler/driver runtime (libc, libstdc++, libgomp, libcuda, ...) and
anything outside those directories is listed as external and not enforced.  An
MPI, HDF5, LibSci, GPU math/collective or engine-private library that resolves
from none of them refuses (exit 1): the build would leave it to the run shell.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

_LXKIT = Path(__file__).resolve().parents[4] / "services" / "lxkit" / "src"
if _LXKIT.is_dir() and str(_LXKIT) not in sys.path:
    sys.path.insert(0, str(_LXKIT))      # this tree's lxkit, whatever the venv holds

from lxkit import LEGS, native_provider  # noqa: E402


def _platform_of(leg: Path) -> str:
    for key, spec in LEGS.items():
        if spec.so_name == leg.name:
            return key
    raise SystemExit(f"[write_checkout_closure] REFUSED: {leg.name} is no "
                     f"lxkit.legs.LEGS so_name ({[s.so_name for s in LEGS.values()]})")


def plugin_dirs(python: str, prefix: str) -> list[Path]:
    """Existing RPATH/RUNPATH directories of the JAX PJRT plugin(s)
    ``jax_plugins/<prefix>*/*.so`` in ``python``'s site-packages."""
    if not prefix:
        return []
    out = subprocess.run(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True, capture_output=True, timeout=120)
    if out.returncode:
        raise SystemExit(f"[write_checkout_closure] REFUSED: {python} cannot report "
                         f"its site-packages: {out.stderr.strip()}")
    site = Path(out.stdout.strip())
    dirs: list[Path] = []
    for plugin in sorted((site / "jax_plugins").glob(f"{prefix}*/*.so")):
        tags = native_provider._elf_dynamic(plugin.resolve())
        for tag in (native_provider._DT_RPATH, native_provider._DT_RUNPATH):
            for entry in tags.get(tag, []):
                for d in entry.split(":"):
                    if not d:
                        continue
                    p = Path(d.replace("${ORIGIN}", str(plugin.parent))
                              .replace("$ORIGIN", str(plugin.parent)))
                    if p.is_dir() and p.resolve() not in dirs:
                        dirs.append(p.resolve())
    return dirs


def _versions(libraries: list[dict]) -> None:
    """Best effort: the wheel (``name==version``) that installed each file, from
    the ``RECORD`` of a ``*.dist-info`` in one of its ancestors.  Diagnostic only
    (log identity lines); the record binds content, not names."""
    index: dict[Path, dict[str, str]] = {}
    for lib in libraries:
        path = Path(lib["path"])
        for anc in list(path.parents)[:6]:
            if anc not in index:
                found: dict[str, str] = {}
                for rec in anc.glob("*.dist-info/RECORD"):
                    dist = rec.parent.name[:-len(".dist-info")]
                    name, _, ver = dist.rpartition("-")
                    for line in rec.read_text(encoding="utf-8", errors="replace").splitlines():
                        rel = line.split(",", 1)[0]
                        if ".so" in rel:
                            found[rel] = f"{name}=={ver}"
                index[anc] = found
            hit = index[anc].get(path.relative_to(anc).as_posix())
            if hit:
                lib["version"] = hit
                break


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("leg", type=Path)
    ap.add_argument("--private-dir", action="append", default=[], type=Path)
    ap.add_argument("--library-dirs", default="",
                    help="colon-separated pinned machine library directories")
    ap.add_argument("--python", default=sys.executable,
                    help="the run venv's python (its JAX plugin's directories)")
    ap.add_argument("--process-dir", action="append", default=[], type=Path)
    ap.add_argument("--skip-missing-dirs", action="store_true",
                    help="drop --library-dirs entries that do not exist (for a list "
                         "taken from a verify step's LD_LIBRARY_PATH)")
    args = ap.parse_args(argv)
    leg = args.leg.resolve()
    platform = _platform_of(leg)
    spec = LEGS[platform]
    process = [d.resolve() for d in args.process_dir]
    if spec.accelerator:
        process += [d for d in plugin_dirs(args.python, spec.jax_plugin_prefix)
                    if d not in process]
    lib_dirs = [Path(d).resolve() for d in args.library_dirs.split(":") if d]
    if args.skip_missing_dirs:
        lib_dirs = [d for d in lib_dirs if d.is_dir()]
    for d in [*args.private_dir, *lib_dirs, *process]:
        if not Path(d).is_dir():
            print(f"[write_checkout_closure] REFUSED: directory {d} does not exist",
                  file=sys.stderr)
            return 1
    libraries, external, refusals = native_provider.resolve_closure(
        leg, private_dirs=[d.resolve() for d in args.private_dir],
        process_dirs=process, library_dirs=lib_dirs)
    if refusals:
        print(f"[write_checkout_closure] REFUSED: {leg.name} needs "
              f"{', '.join(refusals)}, found in no recorded directory "
              f"(private={[str(d) for d in args.private_dir]}, process="
              f"{[str(d) for d in process]}, pinned={[str(d) for d in lib_dirs]}).  "
              "Pin its directory; the run shell must not choose it.",
              file=sys.stderr)
        return 1
    # The leg's collective library (NCCL/RCCL) is the JAX plugin's: recorded so
    # the loader attests and enforces it once it is mapped, never preloaded.
    if spec.accelerator and spec.collective_library:
        have = {lib["family"] for lib in libraries}
        fam = f"lib{spec.collective_library}"
        if fam not in have:
            hit = next((f for d in process for f in sorted(d.glob(f"{fam}.so.[0-9]*"))
                        if native_provider.library_family(
                            (native_provider._elf_dynamic(f.resolve()).get(
                                native_provider._DT_SONAME) or [f.name])[0]) == fam), None)
            if hit is None:
                print(f"[write_checkout_closure] REFUSED: no {fam} in the JAX plugin's "
                      f"directories {[str(d) for d in process]}", file=sys.stderr)
                return 1
            path = hit.resolve()
            st_c = path.stat()
            soname = (native_provider._elf_dynamic(path).get(native_provider._DT_SONAME)
                      or [path.name])[0]
            libraries.append({"soname": soname, "elf_soname": soname, "path": str(path),
                              "sha256": native_provider._sha256(path),
                              "bytes": st_c.st_size, "mtime_ns": st_c.st_mtime_ns,
                              "role": "attest", "family": fam})
    _versions(libraries)
    families: dict[str, str] = {}
    for lib in libraries:
        other = families.setdefault(lib["family"], lib["path"])
        if other != lib["path"]:
            print(f"[write_checkout_closure] REFUSED: {leg.name}'s closure holds two "
                  f"{lib['family']}: {other} and {lib['path']}", file=sys.stderr)
            return 1
    st = leg.stat()
    doc = {"schema": native_provider.CLOSURE_SCHEMA, "platform": platform,
           "leg": leg.name, "leg_sha256": native_provider._sha256(leg),
           "leg_bytes": st.st_size, "libraries": libraries, "external": external,
           "search": {"private": [str(d) for d in args.private_dir],
                      "process": [str(d) for d in process],
                      "pinned": [str(d) for d in lib_dirs]},
           "built_on": socket.gethostname(),
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    doc["record_sha256"] = native_provider.closure_record_digest(doc)
    out = native_provider.closure_record_path(leg)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out)
    roles: dict[str, int] = {}
    for lib in libraries:
        roles[lib["role"]] = roles.get(lib["role"], 0) + 1
    print(f"[write_checkout_closure] {out.name}: {len(libraries)} enforced "
          f"libraries {roles}, {len(external)} external; "
          f"record_sha256={doc['record_sha256']}")
    for lib in libraries:
        print(f"[write_checkout_closure]   {lib['soname']:32s} {lib['role']:8s} "
              f"{lib['path']} {lib.get('version', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
