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

from dataclasses import dataclass

NORMALIZED_RKB_FOUR_CURRENT_REPRESENTATION = (
    "normalized_rkb_four_current_v1")
SOURCE_WFN_CHARGE_REPRESENTATION = "source_wfn_normalized_charge_v1"


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
    from common.bispinor_init import NORMALIZED_RKB_LIFT

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
