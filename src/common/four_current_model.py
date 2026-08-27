"""One resolver for four-current carrier and artifact representations."""

from dataclasses import dataclass

PAULI_REFERENCE_BARE_TRANSVERSE_MODEL = "pauli_reference_bare_transverse"
ISOMETRIC_KINETIC_BALANCE_BARE_TRANSVERSE_MODEL = (
    "isometric_kinetic_balance_bare_transverse")
ISOMETRIC_KINETIC_BALANCE_FULL_STATIC_HEADLESS_DIAGNOSTIC_MODEL = (
    "isometric_kinetic_balance_full_static_cohsex_headless_diagnostic")


def is_pauli_reference_model(value) -> bool:
    """Recognize the mixed Pauli-charge/raw4-current model selector."""
    return str(getattr(value, "value", value)).strip().lower() == (
        PAULI_REFERENCE_BARE_TRANSVERSE_MODEL)


def is_isometric_kinetic_balance_model(value) -> bool:
    """Recognize selectors that share the normalized four-current carrier."""
    return str(getattr(value, "value", value)).strip().lower() in {
        ISOMETRIC_KINETIC_BALANCE_BARE_TRANSVERSE_MODEL,
        ISOMETRIC_KINETIC_BALANCE_FULL_STATIC_HEADLESS_DIAGNOSTIC_MODEL,
    }


RAW_KINETIC_BALANCE_CHARGE_REPRESENTATION = (
    "raw_kinetic_balance_identity_charge_v1")
PAULI_TWO_SPINOR_CHARGE_REPRESENTATION = (
    "qe_normalized_pauli_two_spinor_charge_v1")
SOURCE_WFN_CHARGE_REPRESENTATION = "source_wfn_normalized_charge_v1"
RAW_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION = (
    "raw_kinetic_balance_alpha_spatial_current_v1")
ISOMETRIC_KINETIC_BALANCE_CHARGE_REPRESENTATION = (
    "isometric_kinetic_balance_identity_charge_v1")
ISOMETRIC_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION = (
    "isometric_kinetic_balance_alpha_spatial_current_v1")


@dataclass(frozen=True)
class FourCurrentRepresentation:
    """Resolved carrier choices for one GW model.

    Charge and current body carriers are explicit and independent;
    ``scalar_head_bispinor`` separately governs the canonical scalar
    dipole/head producer.  Keeping those two decisions together is what stops
    preprocessing, ISDF, Hartree, and Sigma from inventing local model maps.
    """

    charge_bispinor: bool
    charge_lift: str | None
    current_bispinor: bool
    current_lift: str | None
    scalar_head_bispinor: bool
    charge_representation: str
    spatial_current_representation: str | None


def resolve_four_current_representation(
    bispinor: bool,
    model,
) -> FourCurrentRepresentation:
    """Resolve all carrier decisions without importing the GW driver."""
    from common.bispinor_init import (
        ISOMETRIC_KINETIC_BALANCE_LIFT,
        RAW_KINETIC_BALANCE_LIFT,
    )

    if not bool(bispinor):
        return FourCurrentRepresentation(
            charge_bispinor=False,
            charge_lift=None,
            current_bispinor=False,
            current_lift=None,
            scalar_head_bispinor=False,
            charge_representation=SOURCE_WFN_CHARGE_REPRESENTATION,
            spatial_current_representation=None,
        )
    if is_pauli_reference_model(model):
        return FourCurrentRepresentation(
            charge_bispinor=False,
            charge_lift=None,
            current_bispinor=True,
            current_lift=RAW_KINETIC_BALANCE_LIFT,
            scalar_head_bispinor=False,
            charge_representation=PAULI_TWO_SPINOR_CHARGE_REPRESENTATION,
            spatial_current_representation=(
                RAW_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION),
        )
    if is_isometric_kinetic_balance_model(model):
        return FourCurrentRepresentation(
            charge_bispinor=True,
            charge_lift=ISOMETRIC_KINETIC_BALANCE_LIFT,
            current_bispinor=True,
            current_lift=ISOMETRIC_KINETIC_BALANCE_LIFT,
            scalar_head_bispinor=False,
            charge_representation=(
                ISOMETRIC_KINETIC_BALANCE_CHARGE_REPRESENTATION),
            spatial_current_representation=(
                ISOMETRIC_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION),
        )
    return FourCurrentRepresentation(
        charge_bispinor=True,
        charge_lift=RAW_KINETIC_BALANCE_LIFT,
        current_bispinor=True,
        current_lift=RAW_KINETIC_BALANCE_LIFT,
        scalar_head_bispinor=True,
        charge_representation=RAW_KINETIC_BALANCE_CHARGE_REPRESENTATION,
        spatial_current_representation=(
            RAW_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION),
    )
