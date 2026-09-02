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

Models -- BOTH of them (the deck grammar has two values since 2026-09-01;
the two carrier-comparison spellings were retired, ``gw_config``'s
``_RETIRED_BISPINOR_GW_MODES``):

* ``bare_transverse`` (default) and ``full_static_cohsex`` (the packed
  static photon mode): the raw kinetic-balance lift
  ``Psi = (Psi_L, (alpha_FS/2) sigma.p Psi_L)`` for both charge and current,
  and the four-spinor scalar head/dipole artifact
  (``scalar_head_bispinor = True``).

They resolve to the SAME carrier -- ``bispinor_gw`` selects which Lorentz
blocks are screened, never which four-spinor represents them -- so this
resolver has exactly two outcomes, bispinor and not.  It stays a resolver
rather than collapsing into ``bool(bispinor)`` because the artifact
provenance stamps below are what the ``dipole.h5`` / ``kin_ion.h5`` / zeta
authenticators compare against, and those need ONE producer.

The representation strings are the provenance stamps written into and
authenticated from the ``dipole.h5`` / ``kin_ion.h5`` / zeta artifacts.  This
module imports nothing from the GW driver so it can be used at artifact
altitude.
"""

from dataclasses import dataclass

RAW_KINETIC_BALANCE_CHARGE_REPRESENTATION = (
    "raw_kinetic_balance_identity_charge_v1")
SOURCE_WFN_CHARGE_REPRESENTATION = "source_wfn_normalized_charge_v1"
RAW_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION = (
    "raw_kinetic_balance_alpha_spatial_current_v1")


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
    """Resolve all carrier decisions without importing the GW driver.

    ``model`` is accepted and ignored: both shipped ``bispinor_gw`` values
    ride the raw kinetic-balance carrier.  The parameter stays so the call
    sites keep naming the mode they resolved -- when a phase-3 mode needs a
    different carrier, this is the one function that has to learn about it.
    """
    from common.bispinor_init import RAW_KINETIC_BALANCE_LIFT

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
