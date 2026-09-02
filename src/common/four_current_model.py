"""One resolver for four-current carrier and artifact representations.

A bispinor run carries two independent choices that several stages must
agree on: which four-component carrier represents the *charge* density and
scalar (CC) screening/exchange/correlation, and which represents the
*spatial current* channels (transverse Hartree, bare TT exchange, the packed
photon body).  ``bispinor_gw`` selects a model, and this module is the single
place that turns a model name into those carrier decisions
(:class:`FourCurrentRepresentation`), so preprocessing
(``psp.get_dipole_mtxels``, ``gw.kin_ion_io``), the charge and transverse
ISDF fits, the exact Hartree, the scalar head producer and the Sigma
dispatch never derive the representation split on their own.

Models:

* ``bare_transverse`` (default) and ``full_static_cohsex`` (the packed
  static photon mode): the raw kinetic-balance lift
  ``Psi = (Psi_L, (alpha_FS/2) sigma.p Psi_L)`` for both charge and current,
  and the four-spinor scalar head/dipole artifact
  (``scalar_head_bispinor = True``).
* ``pauli_reference_bare_transverse``: an explicit comparison model -- the
  normalized QE Pauli two-spinor for charge, the raw kinetic-balance current.
* ``isometric_kinetic_balance_bare_transverse``: the pointwise isometry
  ``[I;X](I+X^dagger X)^{-1/2}`` for every finite-q carrier; its scalar head
  remains the canonical two-spinor artifact until normalized endpoint jets
  exist.

The representation strings are the provenance stamps written into and
authenticated from the ``dipole.h5`` / ``kin_ion.h5`` / zeta artifacts.  This
module imports nothing from the GW driver so it can be used at artifact
altitude.
"""

from dataclasses import dataclass

PAULI_REFERENCE_BARE_TRANSVERSE_MODEL = "pauli_reference_bare_transverse"
ISOMETRIC_KINETIC_BALANCE_BARE_TRANSVERSE_MODEL = (
    "isometric_kinetic_balance_bare_transverse")


def is_pauli_reference_model(value) -> bool:
    """Recognize the mixed Pauli-charge/raw4-current model selector."""
    return str(getattr(value, "value", value)).strip().lower() == (
        PAULI_REFERENCE_BARE_TRANSVERSE_MODEL)


def is_isometric_kinetic_balance_model(value) -> bool:
    """Recognize the normalized kinetic-balance comparison selector."""
    return str(getattr(value, "value", value)).strip().lower() == (
        ISOMETRIC_KINETIC_BALANCE_BARE_TRANSVERSE_MODEL)


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
