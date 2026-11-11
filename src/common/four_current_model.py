"""One resolver for four-current carrier and artifact representations.

A bispinor run has one four-component carrier for every vertex of the
four-current: the charge density and scalar (CC) screening/exchange/
correlation, and the spatial current channels (transverse Hartree, bare TT
exchange, the packed photon body).  ``bispinor_gw`` selects a model, and this
module is the single place that turns a model name into the carrier
decisions (:class:`FourCurrentRepresentation`), so preprocessing
(``psp.get_dipole_mtxels``, ``gw.kin_ion_io``), the charge and transverse
ISDF fits, the exact Hartree, the scalar head producer and the Sigma
dispatch never derive the representation on their own.

Models -- all three of them (the deck grammar added `full_tangential` in 2026-09;
the two carrier-comparison spellings were retired, ``gw_config``'s
``_RETIRED_BISPINOR_GW_MODES``):

* ``bare_transverse`` (default), ``full_static_cohsex`` (the packed
  static photon mode) and ``full_tangential``: charge and currents on the
  normalized restricted-kinetic-balance (RKB) lift
  ``Psi = [I; X](I + X^dagger X)^(-1/2) Psi_L``, ``X = (alpha_FS/2) sigma.p``,
  and the four-spinor scalar head/dipole artifact
  (``scalar_head_bispinor = True``).
  Theory: ``docs/theory/bispinor-gw.md#lift``.

They resolve to the SAME carrier -- ``bispinor_gw`` selects which Lorentz
blocks are screened, never which four-spinor represents them -- so this
resolver has exactly two outcomes, bispinor and not.  It stays a resolver
rather than collapsing into ``bool(bispinor)`` because the artifact
provenance stamps below are what the zeta and restart-bundle
authenticators compare against, and those need ONE producer.

The representation strings are the provenance stamps written into and
authenticated from the zeta and restart artifacts.  This
module imports nothing from the GW driver so it can be used at artifact
altitude.
"""

import os
from dataclasses import dataclass

NORMALIZED_RKB_FOUR_CURRENT_REPRESENTATION = (
    "normalized_rkb_four_current_v1")
SOURCE_WFN_CHARGE_REPRESENTATION = "source_wfn_normalized_charge_v1"
# Diagnostic carriers (LORRAX_DIAG_BISPINOR_CARRIER, docs/reference/env_vars.md
# section 1b).  Each has its own stamp, so a zeta, restart bundle, SC checkpoint
# or Hall artifact written on another carrier is refused or refit.
DIAG_BISPINOR_CARRIER_ENV = "LORRAX_DIAG_BISPINOR_CARRIER"
DIAG_BISPINOR_CARRIERS = ("normalized_rkb", "raw", "pauli")
RAW_KINETIC_BALANCE_FOUR_CURRENT_DIAG_REPRESENTATION = (
    "raw_kinetic_balance_four_current_diag_v1")
LARGE_BLOCK_PAULI_CHARGE_REPRESENTATION = "large_block_pauli_charge_v1"
_DIAG_ANNOUNCED = set()


def diagnostic_bispinor_carrier() -> str:
    """The bispinor carrier selected by ``LORRAX_DIAG_BISPINOR_CARRIER``.

    Unset or blank is ``normalized_rkb`` (main's carrier, unchanged).
    ``raw``: charge and current on the raw kinetic-balance lift.
    ``pauli``: two-component charge (large block, zero small block) in every
    charge vertex, raw-lift current in every current vertex.  Any other value
    refuses.
    """
    raw = os.environ.get(DIAG_BISPINOR_CARRIER_ENV, "")
    mode = str(raw).strip().lower() or "normalized_rkb"
    if mode not in DIAG_BISPINOR_CARRIERS:
        raise ValueError(
            f"{DIAG_BISPINOR_CARRIER_ENV}={raw!r}; expected one of "
            f"{list(DIAG_BISPINOR_CARRIERS)} (unset = normalized_rkb)")
    if mode != "normalized_rkb" and mode not in _DIAG_ANNOUNCED:
        _DIAG_ANNOUNCED.add(mode)
        print(f"*** LORRAX DIAG: {DIAG_BISPINOR_CARRIER_ENV}={mode}: "
              "diagnostic bispinor carrier, not main's normalized RKB",
              flush=True)
    return mode


@dataclass(frozen=True)
class FourCurrentRepresentation:
    """Resolved carrier choices for one GW model.

    Charge and current body carriers are named separately so each consumer
    reads the one it contracts; today both are the normalized RKB lift.
    ``scalar_head_bispinor`` separately governs the canonical scalar
    dipole/head producer.  Keeping those decisions together is what stops
    preprocessing, ISDF, Hartree, and Sigma from inventing local model maps.
    """

    charge_bispinor: bool
    charge_lift: str | None
    current_bispinor: bool
    current_lift: str | None
    scalar_head_bispinor: bool
    charge_representation: str


def resolve_four_current_representation(
    bispinor: bool,
    model,
) -> FourCurrentRepresentation:
    """Resolve all carrier decisions without importing the GW driver.

    ``model`` is accepted and ignored: all shipped ``bispinor_gw`` values
    ride the same carrier.  The parameter stays so the call
    sites keep naming the mode they resolved -- when a phase-3 mode needs a
    different carrier, this is the one function that has to learn about it.
    """
    from common.bispinor_init import (
        LARGE_BLOCK_LIFT, NORMALIZED_RKB_LIFT, RAW_KINETIC_BALANCE_LIFT)

    if not bool(bispinor):
        return FourCurrentRepresentation(
            charge_bispinor=False,
            charge_lift=None,
            current_bispinor=False,
            current_lift=None,
            scalar_head_bispinor=False,
            charge_representation=SOURCE_WFN_CHARGE_REPRESENTATION,
        )
    diag = diagnostic_bispinor_carrier()
    if diag == "raw":
        return FourCurrentRepresentation(
            charge_bispinor=True,
            charge_lift=RAW_KINETIC_BALANCE_LIFT,
            current_bispinor=True,
            current_lift=RAW_KINETIC_BALANCE_LIFT,
            scalar_head_bispinor=True,
            charge_representation=(
                RAW_KINETIC_BALANCE_FOUR_CURRENT_DIAG_REPRESENTATION),
        )
    if diag == "pauli":
        # The current lift must be raw: the direct field then takes the
        # charge from the leading (unscaled psi_L) block of the current sphere.
        return FourCurrentRepresentation(
            charge_bispinor=True,
            charge_lift=LARGE_BLOCK_LIFT,
            current_bispinor=True,
            current_lift=RAW_KINETIC_BALANCE_LIFT,
            scalar_head_bispinor=True,
            charge_representation=LARGE_BLOCK_PAULI_CHARGE_REPRESENTATION,
        )
    return FourCurrentRepresentation(
        charge_bispinor=True,
        charge_lift=NORMALIZED_RKB_LIFT,
        current_bispinor=True,
        current_lift=NORMALIZED_RKB_LIFT,
        scalar_head_bispinor=True,
        charge_representation=NORMALIZED_RKB_FOUR_CURRENT_REPRESENTATION,
    )
