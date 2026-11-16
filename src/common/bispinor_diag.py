"""The one diagnostic environment switch of the bispinor (four-current) path.

    LORRAX_DIAG_ALPHA_FS_SCALE = s    alpha_fs -> s * alpha_fs in the
                                      kinetic-balance lift and every current
                                      vertex (linearity test: J ~ s,
                                      alpha.A and Sigma^B ~ s^2)

A validation instrument, not a physics key (docs/reference/env_vars.md
section 1b).  It scales a constant at its one home
(``common.bispinor_init.HALFALPHA``), read at import, so every driver of a
workflow sees the same value; a deck key could not reach the import-time
constant of the kmeans, dipole and GW drivers alike.  It is inert when unset
(the factor is exactly 1.0), announces itself once per process when set, is
repeated in every output header (``common.provenance``, field
``bispinor_diag``) and in the run report, and the GW driver refuses it on a
two-component run (:func:`refuse_on_two_component`).

The transverse terms themselves are deck keys since 2026-11-14
(``bispinor_transverse_hartree``, ``bispinor_transverse_exchange``,
``bispinor_transverse_exchange_skip_bands``; docs/input_reference.md).  Their
former environment switches are removed (a set one refuses, naming its key):
one mechanism per quantity.

This module imports nothing from LORRAX so ``common.bispinor_init`` can read
the scale at import.
"""

import math
import os

_ANNOUNCED = set()


def _announce(name: str, value: str) -> None:
    if name not in _ANNOUNCED:
        _ANNOUNCED.add(name)
        print(f"*** LORRAX DIAG: {name}={value}: diagnostic bispinor switch, "
              "not a production result", flush=True)


def alpha_fs_scale() -> float:
    """``LORRAX_DIAG_ALPHA_FS_SCALE``: a finite positive float, 1.0 unset."""
    raw = os.environ.get("LORRAX_DIAG_ALPHA_FS_SCALE", "")
    text = str(raw).strip()
    if not text:
        return 1.0
    try:
        scale = float(text)
    except ValueError:
        scale = math.nan
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(
            f"LORRAX_DIAG_ALPHA_FS_SCALE={raw!r}; expected a finite "
            "positive number (unset = 1)")
    if scale != 1.0:
        _announce("LORRAX_DIAG_ALPHA_FS_SCALE", repr(scale))
    return scale


#: The retired environment switches of 2bde04a7d and the deck keys that
#: replaced them.  A set one refuses (:func:`refuse_retired_switches`): read
#: silently as nothing, it would label a run "off" that still carries the term.
RETIRED_SWITCHES = {
    "LORRAX_DIAG_TT_EXCHANGE": "bispinor_transverse_exchange = false",
    "LORRAX_DIAG_TRANSVERSE_HARTREE": "bispinor_transverse_hartree = false",
    "LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS": "bispinor_transverse_exchange_skip_bands = N",
}


def refuse_retired_switches() -> None:
    """Refuse any set retired switch, naming the input-file key that replaced it."""
    found = [name for name in RETIRED_SWITCHES
             if str(os.environ.get(name, "")).strip()]
    if found:
        raise ValueError(
            "GATE bispinor_diag_switch_retired: a removed environment switch "
            "is set.\n"
            f"  got:  {', '.join(f'{n}={os.environ[n]!r}' for n in found)}\n"
            "  want: the input-file key instead: "
            + "; ".join(f"{n} is now {RETIRED_SWITCHES[n]}" for n in found) + "\n"
            "  why:  the switch is no longer read; the run would carry the "
            "term while you think it is off\n"
            "  doc:  docs/input_reference.md, bispinor_transverse_hartree.")


FOURC_CHARGE_VALUES = ("hartree", "vertex", "both")


def fourc_charge() -> str | None:
    """``LORRAX_DIAG_FOURC_CHARGE``: a literal four-component charge
    (normalized kinetic-balance lift, main 23d904a67's carrier) in the direct
    field (``hartree``), in every charge vertex of zeta / Sigma_x / W / Sigma_c
    (``vertex``), or in both.  Unset: the Pauli carrier, bit-identical.
    Diagnostic of the valence small-component Coulomb double count; never a
    production result."""
    text = str(os.environ.get("LORRAX_DIAG_FOURC_CHARGE", "")).strip().lower()
    if not text:
        return None
    if text not in FOURC_CHARGE_VALUES:
        raise ValueError(
            f"LORRAX_DIAG_FOURC_CHARGE={text!r}; expected one of "
            f"{FOURC_CHARGE_VALUES} (unset = Pauli charge)")
    _announce("LORRAX_DIAG_FOURC_CHARGE", text)
    return text


def fourc_charge_hartree() -> bool:
    return fourc_charge() in ("hartree", "both")


def fourc_charge_vertex() -> bool:
    return fourc_charge() in ("vertex", "both")


def active_switches() -> str | None:
    """One line naming the set switches, or ``None`` when they are inert."""
    scale = alpha_fs_scale()
    parts = [] if scale == 1.0 else [f"LORRAX_DIAG_ALPHA_FS_SCALE={scale!r}"]
    fc = fourc_charge()
    if fc is not None:
        parts.append(f"LORRAX_DIAG_FOURC_CHARGE={fc}")
    return "; ".join(parts) if parts else None


def refuse_on_two_component(*, bispinor: bool) -> None:
    """Refuse a retired switch, and a set switch on a two-component run."""
    refuse_retired_switches()
    active = active_switches()
    if active is not None and not bispinor:
        raise ValueError(
            "GATE bispinor_diag_switch_route: a bispinor diagnostic switch is "
            f"set on a two-component run.\n  got:  {active}, bispinor = false\n"
            "  want: bispinor = true, or the variable unset\n"
            "  why:  the scale acts on four-current terms only")
