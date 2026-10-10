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

Models (the deck grammar added `full_shared_pole` in 2026-09;
the two carrier-comparison spellings were retired, ``gw_config``'s
``_RETIRED_BISPINOR_GW_MODES``):

* ``coulomb_only``, ``bare_transverse`` (default), ``full_static_cohsex`` (the packed
  static photon mode) and ``full_shared_pole``: charge and currents on the
  normalized restricted-kinetic-balance (RKB) lift
  ``Psi = [I; X](I + X^dagger X)^(-1/2) Psi_L``, ``X = (alpha_FS/2) sigma.p``,
  and the four-spinor scalar head/dipole artifact
  (``scalar_head_bispinor = True``).
  Theory: ``docs/theory/bispinor-gw.md#lift``.

``coulomb_only`` omits spatial-current interactions but retains the same
normalized four-component charge carrier and scalar-head artifact.

The model names resolve to the same normalized RKB carrier. A bound compact
charge reconstruction may additionally declare ``pauli2embed4`` for a paired
two-component control: its four slots contain ``[psi_Pauli; 0]`` and it admits
only scalar Coulomb fitting with head corrections disabled. This source
transform is stamped separately from RKB in zeta and restart artifacts.

The representation strings are the provenance stamps written into and
authenticated from the zeta and restart artifacts.  This
module imports nothing from the GW driver so it can be used at artifact
altitude.
"""

from dataclasses import dataclass

NORMALIZED_RKB_FOUR_CURRENT_REPRESENTATION = (
    "normalized_rkb_four_current_v1")
SOURCE_WFN_CHARGE_REPRESENTATION = "source_wfn_normalized_charge_v1"
PAULI_ZERO_SMALL_CARRIER = "pauli2embed4"
PAULI_ZERO_SMALL_PROVENANCE = "Psi=[psi_Pauli;0]; zero-small charge carrier without kinetic-balance lift"
PAULI_ZERO_SMALL_CHARGE_REPRESENTATION = "pauli_zero_small_charge_v1"


def charge_carrier_lift_provenance(carrier):
    """Name one source transform without conflating zero-small with RKB."""
    if carrier == PAULI_ZERO_SMALL_CARRIER:
        return PAULI_ZERO_SMALL_PROVENANCE
    from common.bispinor_init import kinetic_balance_lift_provenance
    return kinetic_balance_lift_provenance(carrier)


@dataclass(frozen=True)
class FourCurrentRepresentation:
    """Resolved carrier choices for one GW model.

    Charge and current body carriers are named separately so each consumer
    reads the one it contracts. An explicit zero-small charge control has no
    admitted current or scalar-head carrier.
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
    *, charge_carrier=None,
) -> FourCurrentRepresentation:
    """Resolve all carrier decisions without importing the GW driver.

    All ordinary model names retain their existing RKB decisions. Only a
    bound charge reconstruction passes ``charge_carrier`` explicitly; its
    zero-small Pauli control requires the scalar Coulomb-only model.
    """
    from common.bispinor_init import NORMALIZED_RKB_LIFT

    if charge_carrier == PAULI_ZERO_SMALL_CARRIER:
        if not bool(bispinor) or str(getattr(model, 'value', model)) != 'coulomb_only':
            raise ValueError('The declared Pauli zero-small carrier requires four-slot scalar Coulomb-only charge fitting')
        return FourCurrentRepresentation(charge_bispinor=True,
            charge_lift=PAULI_ZERO_SMALL_CARRIER,current_bispinor=False,
            current_lift=None,scalar_head_bispinor=False,
            charge_representation=PAULI_ZERO_SMALL_CHARGE_REPRESENTATION)
    if charge_carrier not in (None, NORMALIZED_RKB_LIFT):
        raise ValueError('Unknown explicit scalar charge carrier')

    if not bool(bispinor):
        return FourCurrentRepresentation(
            charge_bispinor=False,
            charge_lift=None,
            current_bispinor=False,
            current_lift=None,
            scalar_head_bispinor=False,
            charge_representation=SOURCE_WFN_CHARGE_REPRESENTATION,
        )
    return FourCurrentRepresentation(
        charge_bispinor=True,
        charge_lift=NORMALIZED_RKB_LIFT,
        current_bispinor=True,
        current_lift=NORMALIZED_RKB_LIFT,
        scalar_head_bispinor=True,
        charge_representation=NORMALIZED_RKB_FOUR_CURRENT_REPRESENTATION,
    )
