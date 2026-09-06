"""Regenerate only the mini QE reference, in a new maintenance run directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from tests.core.fixtures.build_fixtures import (
    DEFAULT_BGW_BIN, DEFAULT_QE_BIN, _run, _sha256, _validate_tools,
)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qe-bin", type=Path, default=DEFAULT_QE_BIN)
    parser.add_argument("--bgw-bin", type=Path, default=DEFAULT_BGW_BIN)
    args = parser.parse_args()
    tools = _validate_tools(args.qe_bin, args.bgw_bin)
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent / "fixtures" / "H2-screw"
    names = ("scf.in", "nscf.in", "pw2bgw.in", "H.upf")
    for name in names:
        shutil.copy2(source / name, target / name)
    started = time.perf_counter()
    for name in ("scf", "nscf", "pw2bgw"):
        tool = "pw2bgw.x" if name == "pw2bgw" else "pw.x"
        _run([str(tools[tool]), "-in", name + ".in"], cwd=target,
             stdin=None, log=target / (name + ".out"))
        if name in ("scf", "nscf"):
            schema = "scf-schema.xml" if name == "scf" else "data-file-schema.xml"
            shutil.copy2(target / "mini_h2_screw.save" / "data-file-schema.xml",
                         target / schema)
    _run([str(tools["wfn2hdf.x"]), "BIN", "WFN", "WFN.h5"], cwd=target,
         stdin=None, log=target / "wfn2hdf.out")
    # A return code does not certify QE convergence. Retain both XMLs for the
    # sandbox's registered SCF/symmetry parsers before publishing any bytes.
    record = {"status": "built_unvalidated", "jobid": os.environ.get("SLURM_JOB_ID"),
              "stepid": os.environ.get("SLURM_STEP_ID"),
              "seconds": time.perf_counter() - started,
              "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
              "files": {name: _sha256(target / name)
                        for name in (*names, "WFN.h5", "data-file-schema.xml", "scf-schema.xml")}}
    (target / "BUILD.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"QE files built in {target}; certify SCF XML before publishing")


if __name__ == "__main__":
    main()
