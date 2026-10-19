"""The ``[epsmat]`` input file of ``gw.epsmat_cli``: schema, defaults, refusals.

The driver reads a finished GW bundle (and, to project, BerkeleyGW's own ε⁻¹
files on the same WFN); it is told where they are, which way to convert, the
BerkeleyGW G cutoff and q → 0 vector when there is no reference file, and the
Coulomb-whitening cut.  Same INI format as every LORRAX deck, one
``[epsmat]`` section; unknown keys REFUSE.  ``docs/input_reference.md``
documents each key; the equations are ``docs/theory/bgw_epsmat_export.md``.
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass

__all__ = ["EPSMAT_DEFAULTS", "EpsmatConfig", "read_epsmat_input"]

#: THE SCHEMA.  The type of the default coerces the deck string; ``None``
#: means required.
EPSMAT_DEFAULTS = {
    # ``export``: LORRAX bundle -> eps0mat.h5 / epsmat.h5.  ``project``:
    # BerkeleyGW eps files -> W_μν(q) written into a copy of the bundle.
    "mode": "export",
    # The finished GW run directory, or its tmp/isdf_tensors_<mu>.h5.
    "source_restart": None,
    # ζ of that bundle; empty = zeta_q.h5 beside the bundle.
    "zeta_file": "",
    # The producer GW deck: WFN, sys_dim and the Coulomb keys V_q was built with.
    "parent_input_file": None,
    # export: where eps0mat.h5 / epsmat.h5 go.  project: the variant run
    # directory (its tmp/ receives the bundle copy); never the source's.
    "output_dir": None,
    # Directory holding BerkeleyGW's eps0mat.h5 / epsmat.h5 on the same WFN.
    # export: q list, q0, ecuts and G order are taken from it.  project: the
    # files projected.
    "bgw_reference": "",
    # export without bgw_reference: BerkeleyGW epsilon_cutoff in Ry.
    "epsilon_cutoff": 0.0,
    # export without bgw_reference: the q -> 0 vector, crystal coordinates.
    "q0_vector": "0.001 0 0",
    # export: which files to write.
    "files": "eps0mat epsmat",
    # Coulomb-whitening cut: keep λ(V_q) > rcond λ_max (amplification <= 1/rcond).
    "rcond": 1.0e-8,
    # ε⁻¹_00(q0): ``s_tensor`` (the bundle's direct q² tensor) or
    # ``unscreened`` (1; debug).
    "q0_head": "s_tensor",
    # q -> 0 wings of eps0mat: ``dipole`` (the bundle's stored ω = 0 head
    # wings L, R: local-field head 1/ε_M, both wings and, in 3D, the body's
    # head-wing term; docs §5 E9–E14) or ``zero`` (wings 0, head from
    # S_cart_head, body without the head-wing term: the behaviour before).
    "q0_wings": "dipole",
    # sys_dim 3 eps0mat with q0_wings = zero: the q -> 0 body omits the O(1) head-wing term
    # ε_b0 ε_00⁻¹ ε_0b.  ``refuse`` or ``omit_head_wing`` (write that body).
    "q0_body_3d": "refuse",
}

_FILES = ("eps0mat", "epsmat")


def read_epsmat_input(filename: str) -> dict:
    """Parse an ``[epsmat]`` file into a dict of typed values; unknown keys refuse."""
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    with open(filename, "r") as fh:
        parser.read_string(fh.read())
    if parser.sections() != ["epsmat"]:
        raise ValueError(
            f"GATE epsmat_deck_section: got sections {parser.sections()} in "
            f"{filename}; want exactly one [epsmat]; why: this driver takes its "
            f"own input, not a GW deck; fix: write an [epsmat] file "
            f"(docs/input_reference.md)")
    named = {k.lower(): v for k, v in parser.items("epsmat")}
    unknown = sorted(set(named) - set(EPSMAT_DEFAULTS))
    if unknown:
        raise ValueError(
            f"GATE epsmat_unknown_key: got {unknown}; want keys of "
            f"{sorted(EPSMAT_DEFAULTS)}; why: a misspelt key would silently keep "
            f"its default; fix: correct the spelling")
    out = {}
    for key, default in EPSMAT_DEFAULTS.items():
        if key not in named:
            out[key] = default
        elif isinstance(default, float):
            out[key] = parser.getfloat("epsmat", key)
        else:
            out[key] = named[key].strip()
    return out


@dataclass(frozen=True)
class EpsmatConfig:
    """The validated ``[epsmat]`` input; every path absolute, every value resolved."""

    mode: str
    source_restart: str
    zeta_file: str
    parent_input_file: str
    output_dir: str
    bgw_reference: str
    epsilon_cutoff: float
    q0_vector: tuple
    files: tuple
    rcond: float
    q0_head: str
    q0_body_3d: str
    q0_wings: str

    @classmethod
    def from_input_file(cls, filename: str) -> "EpsmatConfig":
        p = read_epsmat_input(filename)
        base = os.path.dirname(os.path.abspath(filename))

        def path(key, required):
            raw = str(p[key] or "").strip()
            if not raw:
                if required:
                    raise ValueError(
                        f"GATE epsmat_required_key: got no {key}; want a path; "
                        f"why: {key} has no default; fix: set it in {filename}")
                return ""
            return raw if os.path.isabs(raw) else os.path.normpath(os.path.join(base, raw))

        mode = p["mode"].lower()
        if mode not in ("export", "project"):
            raise ValueError(f"GATE epsmat_mode: got {mode!r}; want export | project; "
                             f"why: the direction of the conversion; fix: set mode")
        files = tuple(p["files"].lower().split())
        if not files or any(f not in _FILES for f in files):
            raise ValueError(f"GATE epsmat_files: got {files}; want a subset of {_FILES}; "
                             f"why: the two BerkeleyGW file kinds; fix: set files")
        q0 = tuple(float(x) for x in p["q0_vector"].split())
        if len(q0) != 3 or max(abs(x) for x in q0) == 0.0:
            raise ValueError(f"GATE epsmat_q0_vector: got {p['q0_vector']!r}; want three "
                             f"numbers, not all zero; why: ε⁻¹ is written at q0 ≠ 0; "
                             f"fix: e.g. q0_vector = 0.001 0 0")
        rcond = float(p["rcond"])
        if not 0.0 < rcond < 1.0:
            raise ValueError(f"GATE epsmat_rcond: got {rcond}; want 0 < rcond < 1; why: a "
                             f"relative cut on λ(V)/λ_max; fix: keep the default 1e-8")
        head = p["q0_head"].lower()
        if head not in ("s_tensor", "unscreened"):
            raise ValueError(f"GATE epsmat_q0_head: got {head!r}; want s_tensor | "
                             f"unscreened; why: the q → 0 head source; fix: set q0_head")
        body = p["q0_body_3d"].lower()
        if body not in ("refuse", "omit_head_wing"):
            raise ValueError(f"GATE epsmat_q0_body_3d_value: got {body!r}; want refuse | "
                             f"omit_head_wing; why: the 3-D q → 0 body without its "
                             f"head–wing term is an approximation taken by name; fix: set "
                             f"q0_body_3d")
        wings = p["q0_wings"].lower()
        if wings not in ("dipole", "zero"):
            raise ValueError(f"GATE epsmat_q0_wings_value: got {wings!r}; want dipole | zero; "
                             f"why: the q → 0 wing source; fix: set q0_wings")
        ref = path("bgw_reference", mode == "project")
        ecut = float(p["epsilon_cutoff"])
        if mode == "export" and not ref and ecut <= 0.0:
            raise ValueError(
                "GATE epsmat_epsilon_cutoff: got no epsilon_cutoff and no "
                "bgw_reference; want one of them; why: the BerkeleyGW G list is "
                "|q+G|² <= ecuts; fix: set epsilon_cutoff (Ry) or bgw_reference")
        return cls(mode=mode, source_restart=path("source_restart", True),
                   zeta_file=path("zeta_file", False),
                   parent_input_file=path("parent_input_file", True),
                   output_dir=path("output_dir", True), bgw_reference=ref,
                   epsilon_cutoff=ecut, q0_vector=q0, files=files, rcond=rcond,
                   q0_head=head, q0_body_3d=body, q0_wings=wings)

    def describe(self) -> str:
        """The resolved input, one key per line, for the run banner."""
        return "\n".join(f"  {k:<18} {getattr(self, k)}" for k in self.__dataclass_fields__)
