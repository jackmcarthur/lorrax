"""psp/nscf_input.py — Parse NSCF input file.

Simple INI-style format:

    [nscf]
    save_dir = path/to/QE.save
    nbnd = 100
    kgrid = 4 4 4
    nosym = false          # default: use crystal symmetries for IBZ reduction
    output = WFN.h5
    tol = 1e-8
    # Optional: read charge density from WFN.h5 instead of QE .save
    rho_from_wfn = false
    wfn_file = WFN_ref.h5  # only used if rho_from_wfn = true
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass


@dataclass
class NSCFInput:
    save_dir: str
    nbnd: int = 100
    kgrid: tuple[int, int, int] = (4, 4, 4)
    nosym: bool = False
    output: str = "WFN.h5"
    tol: float = 1e-8
    rho_from_wfn: bool = False
    wfn_file: str = ""


def read_nscf_input(filename: str) -> NSCFInput:
    """Parse NSCF input file.  Paths are resolved relative to the input file."""
    parser = configparser.ConfigParser()
    parser.read(filename)
    sec = parser["nscf"]
    input_dir = os.path.dirname(os.path.abspath(filename))

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(input_dir, path)

    kgrid_str = sec.get("kgrid", "4 4 4")
    kgrid = tuple(int(x) for x in kgrid_str.split())

    return NSCFInput(
        save_dir=resolve(sec.get("save_dir", ".")),
        nbnd=sec.getint("nbnd", 100),
        kgrid=kgrid,
        nosym=sec.getboolean("nosym", False),
        output=resolve(sec.get("output", "WFN.h5")),
        tol=sec.getfloat("tol", 1e-8),
        rho_from_wfn=sec.getboolean("rho_from_wfn", False),
        wfn_file=resolve(sec.get("wfn_file", "")),
    )
