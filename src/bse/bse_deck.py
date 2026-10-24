"""BSE deck keys: BerkeleyGW's absorption.x and kernel.x names, read from the deck.

The BSE drivers take their physics from the deck passed with ``-i`` (owner
decision D1 = A, 2026-10-21): the command line keeps only ``-i``,
``--px``/``--py``, ``--report-file`` and ``--out-prefix``.  A key is written
``key = value`` in the deck's section, or, as in BerkeleyGW, as a bare flag on
its own line (``full_bse``) or ``key value`` without ``=``.

Each driver maps the deck keys it reads onto the attributes its solver code
uses (:class:`Key`), so the code below the reader is unchanged.  A key the
deck names but this driver does not read is left alone: one deck serves the
eigensolve, the Haydock driver, the exciton bands and the kernel export.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

#: Spellings of a boolean deck value.  A bare flag line reads as ``true``.
_TRUE = {"", "true", "yes", "on", "1", ".true.", "t"}
_FALSE = {"false", "no", "off", "0", ".false.", "f"}


@dataclass(frozen=True)
class Key:
    """How one deck key fills one driver attribute.

    ``kind``: ``int``, ``float``, ``str``, ``bool``, ``path`` (relative to the
    deck's directory) or ``ints``/``floats`` (a list).  ``choices`` limits a
    ``str``.  ``convert`` maps the typed value to the attribute's value.
    """

    dest: str
    kind: str = "str"
    choices: tuple = ()
    convert: Callable[[Any], Any] | None = None


class DeckError(ValueError):
    """A BSE deck key the driver cannot use as written."""


def as_bool(key: str, raw) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise DeckError(f"deck key `{key}` takes true or false, got {raw!r}")


def typed_value(key: str, spec: Key, raw, deck_dir: str):
    """The deck string ``raw`` of ``key`` as ``spec.kind``."""
    text = str(raw).strip()
    try:
        if spec.kind == "bool":
            return as_bool(key, raw)
        if spec.kind == "int":
            return int(text)
        if spec.kind == "float":
            return float(text.replace("d", "e").replace("D", "e"))
        if spec.kind in ("ints", "floats"):
            parts = text.replace(",", " ").split()
            cast = int if spec.kind == "ints" else (
                lambda s: float(s.replace("d", "e").replace("D", "e")))
            return [cast(p) for p in parts]
        if spec.kind == "path":
            if not text:
                raise DeckError(f"deck key `{key}` needs a file name")
            return text if os.path.isabs(text) else os.path.join(deck_dir, text)
    except DeckError:
        raise
    except ValueError:
        want = {"int": "an integer", "float": "a number", "ints": "integers",
                "floats": "numbers"}.get(spec.kind, spec.kind)
        raise DeckError(f"deck key `{key}` takes {want}, got {raw!r}") from None
    if spec.choices and text.lower() not in spec.choices:
        raise DeckError(f"deck key `{key}` takes one of {', '.join(spec.choices)}; "
                        f"got {raw!r}")
    return text.lower() if spec.choices else text


def deck_params(deck_path: str) -> dict:
    """The deck's parameters (``gw.gw_config.read_lorrax_input``) and the keys it names."""
    from gw.gw_config import _DECK_NAMED_KEYS, read_lorrax_input
    try:
        params = read_lorrax_input(deck_path)
    except ValueError as exc:          # an old or unknown key: say it in one line
        raise DeckError(str(exc)) from None
    return params, set(params.get(_DECK_NAMED_KEYS, ()))


def read_settings(deck_path: str, defaults: argparse.Namespace,
                  keys: Mapping[str, Key], *, required: tuple[str, ...] = (),
                  params: dict | None = None,
                  named: set | None = None) -> tuple[argparse.Namespace, set]:
    """``(settings, named)``: ``defaults`` overlaid with every key of ``keys`` the deck names.

    ``required`` keys must be named.  ``named`` (returned) is the set of deck
    keys the deck names, for the SETTINGS ``deck`` source and the route checks.
    """
    if params is None:
        params, named = deck_params(deck_path)
    named = set(named or ())
    missing = [k for k in required if k not in named]
    if missing:
        raise DeckError("\n".join(f"deck key `{k}` is required" for k in missing))
    settings = argparse.Namespace(**vars(defaults))
    deck_dir = os.path.dirname(os.path.abspath(deck_path))
    for key, spec in keys.items():
        if key not in named:
            continue
        value = typed_value(key, spec, params[key], deck_dir)
        setattr(settings, spec.dest, spec.convert(value) if spec.convert else value)
    return settings, named


def deck_source(named: set) -> Callable[..., str]:
    """``source(*keys)`` for SETTINGS rows: ``deck`` when the deck names one of ``keys``."""
    def source(*keys: str) -> str:
        return "deck" if any(k in named for k in keys) else "default"
    return source


def key_of(keys: Mapping[str, Key]) -> Callable[[str], str]:
    """``name(dest)``: the deck key that fills attribute ``dest`` (for refusals)."""
    by_dest = {}
    for key, spec in keys.items():
        by_dest.setdefault(spec.dest, key)
    return lambda dest: by_dest.get(dest, dest)


def eqp_file(params: dict, named: set, deck_path: str) -> str | None:
    """The QP-energy file of ``eqp_corrections`` / ``eqp_co_corrections``, or None.

    As in BerkeleyGW, ``eqp_co_corrections`` (QP energies on the coarse grid,
    carried onto the fine grid) belongs to a ``kgrid_fine`` run and
    ``eqp_corrections`` to a run without one.  A bare key reads BerkeleyGW's
    file name (``eqp_co.dat`` / ``eqp.dat``) beside the deck; a value names
    the file.  LORRAX's ``eqp1.dat`` is that file's format.
    """
    fine = bool(str(params.get("kgrid_fine", "") or "").strip())
    given = [k for k in ("eqp_corrections", "eqp_co_corrections") if k in named]
    if not given:
        return None
    if len(given) == 2:
        raise DeckError("deck keys `eqp_corrections` and `eqp_co_corrections` both name "
                        "QP energies; keep `eqp_co_corrections` with kgrid_fine, "
                        "`eqp_corrections` without it")
    key = given[0]
    if key == "eqp_co_corrections" and not fine:
        raise DeckError("`eqp_co_corrections` interpolates coarse-grid QP energies onto "
                        "kgrid_fine, which this deck does not set; write `eqp_corrections`")
    if key == "eqp_corrections" and fine:
        raise DeckError("`eqp_corrections` is for a run without kgrid_fine; with "
                        "kgrid_fine the coarse-grid QP energies are `eqp_co_corrections`")
    raw = str(params.get(key, "") or "").strip()
    deck_dir = os.path.dirname(os.path.abspath(deck_path))
    if raw.lower() in _TRUE:
        raw = "eqp_co.dat" if key == "eqp_co_corrections" else "eqp.dat"
    elif raw.lower() in _FALSE:
        return None
    return raw if os.path.isabs(raw) else os.path.join(deck_dir, raw)


def wfn_spin(wfn_path: str) -> tuple[int, int]:
    """``(nspinor, nspin)`` of a BerkeleyGW-format WFN.h5 (``mf_header/kpoints``)."""
    import h5py
    with h5py.File(wfn_path, "r") as f:
        k = f["mf_header/kpoints"]
        return int(k["nspinor"][()]), int(k["nspin"][()])


def spin_kernel(params: dict, named: set, wfn_path: str) -> dict:
    """The BSE kernel the deck's spin keys select on this WFN.

    LORRAX runs full-spinor and non-spinor WFNs.  Spinor: one kernel, D + V - W
    (exchange weight 1); ``spinor = true`` is accepted as an assertion, and
    ``spin_singlet`` / ``spin_triplet`` refuse.  Non-spinor: ``spin_singlet``
    (the default, D + 2V - W) or ``spin_triplet`` (D - W).  The screened direct
    term is always on unless ``local_fields = true`` (D + w V, the RPA with local
    fields), which is never the default.  Returns ``{"include_w", "channel",
    "label"}``; the channel only sets the scalar exchange weight
    (``bse_preconditioner.exchange_spin_weight``), passed explicitly through
    the payload and factory identity.
    """
    def flag(key):
        return key in named and as_bool(key, params[key])

    import os
    if not os.path.isfile(wfn_path):
        # The driver refuses a missing WFN when it loads it; the deck keys are
        # then read as written (a spinor kernel cannot be checked).
        nspinor, nspin = (1 if (flag("spin_singlet") or flag("spin_triplet")) else 2), 1
    else:
        nspinor, nspin = wfn_spin(wfn_path)
    if nspin != 1:
        raise DeckError(f"{wfn_path} is collinear spin-polarized (nspin = {nspin}); the BSE "
                        f"runs spinor or non-spinor WFNs")
    spinor = nspinor == 2
    if "spinor" in named and as_bool("spinor", params["spinor"]) != spinor:
        raise DeckError(f"`spinor = {params['spinor']}` disagrees with {wfn_path}, which is "
                        f"{'a spinor' if spinor else 'a non-spinor'} WFN")
    singlet, triplet, rpa = flag("spin_singlet"), flag("spin_triplet"), flag("local_fields")
    if spinor and (singlet or triplet):
        raise DeckError(f"`{'spin_singlet' if singlet else 'spin_triplet'}` is for a non-spinor "
                        f"WFN; a spinor WFN has one kernel (exchange weight 1): remove it")
    if singlet and triplet:
        raise DeckError("`spin_singlet` and `spin_triplet` name two kernels; keep one")
    if triplet and rpa:
        raise DeckError("`local_fields` keeps only the exchange term, which `spin_triplet` "
                        "removes; keep one")
    channel = "triplet" if triplet else "singlet"
    if spinor:
        label = "spinor kernel D + V - W" if not rpa else "spinor RPA, D + V (local_fields)"
    elif rpa:
        label = "singlet RPA, D + 2V (local_fields)"
    else:
        label = ("triplet D - W (spin_triplet)" if triplet
                 else "singlet D + 2V - W (spin_singlet)")
    return {"include_w": not rpa, "channel": channel, "label": label, "nspinor": nspinor}


def note_extended_kernel(params: dict, named: set, log) -> None:
    """``extended_kernel`` is kernel.x's (bse.bsemat_cli); a solver deck only notes it."""
    if "extended_kernel" in named and as_bool("extended_kernel", params["extended_kernel"]):
        log.warning("extended_kernel has no effect on the BSE solve: LORRAX does not "
                    "interpolate the kernel; full_bse selects the non-TDA problem "
                    "(extended_kernel is read by bse.bsemat_cli)",
                    extra={"lorrax": {"id": "extended_kernel_solver", "kind": "NOTE",
                                      "found": ("extended_kernel", "true"),
                                      "fix": "full_bse = true for the non-TDA BSE"}})


def check_fine_grid_keys(params: dict, named: set) -> None:
    """The fine-grid keys (``kgrid_fine_shift``, ``number_*_bands_coarse``) need
    ``kgrid_fine``: without a fine grid there is no interpolation to configure."""
    if str(params.get("kgrid_fine", "") or "").strip():
        return
    stray = [k for k in ("kgrid_fine_shift", "number_cond_bands_coarse",
                         "number_val_bands_coarse")
             if k in named and str(params.get(k, "") or "").strip()]
    if stray:
        raise DeckError(f"{', '.join(f'`{k}`' for k in stray)} configure{'s' if len(stray) == 1 else ''} "
                        f"the interpolation onto kgrid_fine, which this deck does not set")


def check_dipole_operator(params: dict, named: set, dipole_path: str | None) -> None:
    """``use_momentum`` / ``use_velocity``: assert which operator ``dipole_path`` holds.

    BerkeleyGW picks the operator in absorption.inp; LORRAX picks it when the
    dipole file is made (``psp.get_dipole_mtxels``, ``--skip-vnl`` = p alone).
    The keys are therefore checked against the file's ``skip_vnl`` stamp.
    """
    given = [k for k in ("use_momentum", "use_velocity")
             if k in named and as_bool(k, params[k])]
    if not given:
        return
    if len(given) == 2:
        raise DeckError("deck keys `use_momentum` and `use_velocity` name two operators; "
                        "keep one")
    if dipole_path is None:
        raise DeckError(f"`{given[0]}` names the operator of the dipole file, and this run "
                        f"reads none (set `dipole_file`)")
    import h5py
    with h5py.File(dipole_path, "r") as f:
        skip = bool(f.attrs.get("skip_vnl", False))
    momentum = given[0] == "use_momentum"
    if momentum != skip:
        have = "p alone (momentum)" if skip else "p + i[V_NL, r] (velocity)"
        raise DeckError(f"`{given[0]}` disagrees with {dipole_path}: it holds {have}; "
                        f"make the file with{'' if momentum else 'out'} "
                        f"psp.get_dipole_mtxels --skip-vnl, or change the key")


def omega_count(omega_min_ev: float, omega_max_ev: float, delta_ev: float) -> int:
    """Points of the uniform grid ``[omega_min, omega_max]`` at step ``delta_frequency``."""
    if delta_ev <= 0.0:
        raise DeckError(f"deck key `delta_frequency` must be > 0 eV, got {delta_ev}")
    if omega_max_ev <= omega_min_ev:
        raise DeckError(f"`omega_max_ev` ({omega_max_ev}) must exceed `omega_min_ev` "
                        f"({omega_min_ev})")
    return int(round((omega_max_ev - omega_min_ev) / delta_ev)) + 1
