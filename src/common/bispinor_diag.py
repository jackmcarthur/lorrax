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
former environment switches are removed: one mechanism per quantity.

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


def active_switches() -> str | None:
    """One line naming the set switch, or ``None`` when it is inert."""
    scale = alpha_fs_scale()
    return None if scale == 1.0 else f"LORRAX_DIAG_ALPHA_FS_SCALE={scale!r}"


def refuse_on_two_component(*, bispinor: bool) -> None:
    """Refuse a set switch on a two-component run, where it scales nothing."""
    active = active_switches()
    if active is not None and not bispinor:
        raise ValueError(
            "GATE bispinor_diag_switch_route: a bispinor diagnostic switch is "
            f"set on a two-component run.\n  got:  {active}, bispinor = false\n"
            "  want: bispinor = true, or the variable unset\n"
            "  why:  the scale acts on four-current terms only")
