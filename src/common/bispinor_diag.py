"""Diagnostic environment switches of the bispinor (four-current) path.

Validation instruments, not physics keys (docs/reference/env_vars.md section
1b).  Each is inert when unset: the code path it guards is then exactly the
one without this module.  A set switch announces itself once per process,
is repeated in every output header (``common.provenance``, field
``bispinor_diag``) and in the startup block, and the GW driver refuses it
outside the route it acts on (:func:`refuse_outside_bare_transverse`).

    LORRAX_DIAG_TT_EXCHANGE = on | off        bare transverse exchange Sigma^B
    LORRAX_DIAG_TRANSVERSE_HARTREE = on | off transverse direct field alpha.A[J]
    LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS = N    drop the N lowest bands of the
                                              Sigma window from Sigma^B's
                                              occupied sum (semicore test)
    LORRAX_DIAG_ALPHA_FS_SCALE = s            alpha_fs -> s * alpha_fs in the
                                              kinetic-balance lift and every
                                              current vertex (linearity test:
                                              J ~ s, alpha.A and Sigma^B ~ s^2)

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


def _on_off(name: str, raw: str) -> bool:
    """Unset, blank or ``on`` keeps the term; ``off`` removes it; else refuse."""
    mode = str(raw).strip().lower()
    if mode in ("", "on"):
        return True
    if mode == "off":
        _announce(name, "off")
        return False
    raise ValueError(f"{name}={raw!r}; expected on or off (unset = on)")


def tt_exchange_enabled() -> bool:
    """False only under ``LORRAX_DIAG_TT_EXCHANGE=off``."""
    return _on_off("LORRAX_DIAG_TT_EXCHANGE",
                   os.environ.get("LORRAX_DIAG_TT_EXCHANGE", ""))


def transverse_hartree_enabled() -> bool:
    """False only under ``LORRAX_DIAG_TRANSVERSE_HARTREE=off``."""
    return _on_off("LORRAX_DIAG_TRANSVERSE_HARTREE",
                   os.environ.get("LORRAX_DIAG_TRANSVERSE_HARTREE", ""))


def tt_exchange_skip_bands() -> int:
    """``LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS``: a non-negative integer, 0 unset."""
    raw = os.environ.get("LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS", "")
    text = str(raw).strip()
    if not text:
        return 0
    try:
        count = int(text)
    except ValueError:
        count = -1
    if count < 0:
        raise ValueError(
            f"LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS={raw!r}; expected a "
            "non-negative integer (unset = 0)")
    if count:
        _announce("LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS", str(count))
    return count


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
    """One line naming every set switch, or ``None`` when all are inert."""
    parts = []
    if not tt_exchange_enabled():
        parts.append("LORRAX_DIAG_TT_EXCHANGE=off")
    if not transverse_hartree_enabled():
        parts.append("LORRAX_DIAG_TRANSVERSE_HARTREE=off")
    skip = tt_exchange_skip_bands()
    if skip:
        parts.append(f"LORRAX_DIAG_TT_EXCHANGE_SKIP_BANDS={skip}")
    scale = alpha_fs_scale()
    if scale != 1.0:
        parts.append(f"LORRAX_DIAG_ALPHA_FS_SCALE={scale!r}")
    return "; ".join(parts) or None


def refuse_outside_bare_transverse(*, bispinor: bool, bare_transverse: bool,
                                   packed_route: bool) -> None:
    """Refuse a set switch on a route it does not reach.

    The two exchange switches act in ``gw.sigma_x_bispinor`` only, which the
    packed photon routes and ``bispinor_gw = full_tangential`` do not use for
    their transverse exchange; accepting them there would label a run "off"
    that still carries the term.  The Hartree and alpha switches act on every
    bispinor route, so they need ``bispinor = true`` and nothing else.
    """
    active = active_switches()
    if active is None:
        return
    if not bispinor:
        raise ValueError(
            "GATE bispinor_diag_switch_route: a bispinor diagnostic switch is "
            f"set on a two-component run.\n  got:  {active}, bispinor = false\n"
            "  want: bispinor = true, or the variable unset\n"
            "  why:  the switches act on four-current terms only")
    exchange = (not tt_exchange_enabled()) or bool(tt_exchange_skip_bands())
    if exchange and (packed_route or not bare_transverse):
        raise ValueError(
            "GATE bispinor_diag_switch_route: an exchange switch is set on a "
            "route whose transverse exchange it does not reach.\n"
            f"  got:  {active}; bare_transverse = {bool(bare_transverse)}, "
            f"packed photon route = {bool(packed_route)}\n"
            "  want: bispinor_gw = bare_transverse off the packed photon "
            "route (compute_mode = tangential, x_only, or a bulk "
            "plasmon-pole run)\n"
            "  why:  only gw.sigma_x_bispinor reads the switch")
