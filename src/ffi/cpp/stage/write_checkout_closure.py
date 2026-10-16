#!/usr/bin/env python3
"""Write ``<leg>.closure.json``: what an unsealed checkout leg was built against.

    write_checkout_closure.py <leg.so> --private-dir DIR [...] --library-dirs A:B:...

A checkout leg's search path is ``$ORIGIN`` alone (a DT_RUNPATH, outranked by
``LD_LIBRARY_PATH``), so without this record the launch shell's modules pick
which file serves each SONAME.  The record names

* ``private_libraries``: every engine-private provider
  (:func:`lxkit.native_provider.is_private_redistributable`) in the leg's NEEDED
  closure, found by SONAME in a ``--private-dir`` (the build stage: the
  cuSOLVERMp/cuBLASMp wheels, the SLATE install);
* ``library_dirs``: the machine library directories the build pinned and its
  verify step ran against (MPI, parallel HDF5, LibSci: the site's
  ``*_RUNTIME_LIB_DIRS``).

``lxkit.native_provider.open_and_attest`` opens those files by exact path,
dependencies first, before the leg itself -- the mechanism a sealed bundle uses
for its manifest -- so a plain clone launch runs the closure that was verified.
The record is a build product beside the leg, like the leg's provenance stamp;
sealing ignores it (a sealed bundle carries its own manifest).

Refuses (exit 1) when a private provider in the NEEDED closure is in no
``--private-dir``: the leg would then resolve it from the search path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lxkit import native_provider


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("leg", type=Path)
    ap.add_argument("--private-dir", action="append", default=[], type=Path)
    ap.add_argument("--library-dirs", default="",
                    help="colon-separated machine library directories")
    args = ap.parse_args(argv)
    leg = args.leg.resolve()
    private_dirs = [d.resolve() for d in args.private_dir]
    found, missing = native_provider.checkout_private_closure(leg, private_dirs)
    if missing:
        print(f"[write_checkout_closure] REFUSED: {leg.name} needs private "
              f"provider(s) {missing} found in no --private-dir "
              f"{[str(d) for d in private_dirs]}", file=sys.stderr)
        return 1
    dirs = [str(Path(d).resolve()) for d in args.library_dirs.split(":") if d]
    for d in dirs:
        if not Path(d).is_dir():
            print(f"[write_checkout_closure] REFUSED: library dir {d} does not exist",
                  file=sys.stderr)
            return 1
    doc = {"schema": 1, "leg": leg.name,
           "private_libraries": [str(p) for p in found],
           "library_dirs": dirs}
    out = leg.with_name(leg.name + native_provider.CHECKOUT_CLOSURE_SUFFIX)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(f"[write_checkout_closure] {out.name}: private="
          f"{[p.name for p in found]} library_dirs={len(dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
