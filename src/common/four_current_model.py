"""One resolver for four-current carrier and artifact representations.

A bispinor run has one carrier prescription, the Pauli four-current
(docs/theory/bispinor-gw.md#carrier,
docs/architecture/decisions.md#pauli-four-current):

* every CHARGE vertex -- the Hartree density and its matrix elements, the
  charge zeta, Pi_00, W_C, Sigma^CC and the q->0 charge head -- takes the
  two-component density ``psi_L^dagger psi_L``.  Inside the four-component
  container that is the large-block carrier ``[psi_L; 0]``; the scalar head
  producer (``psp.get_dipole_mtxels``) uses the two-spinor dipole
  (``scalar_head_bispinor = False``).
* every CURRENT vertex -- the transverse Hartree ``alpha.A[J]``, the current
  zeta, the bare and screened TT exchange, CT/TC -- takes the raw
  restricted-kinetic-balance lift ``[psi_L; (alpha_FS/2) sigma.p psi_L]``,
  whose ``c Psi^dagger alpha Psi`` is the Pauli current.

Why: the fully relativistic pseudopotential is generated from a
Dirac-Kohn-Sham atom whose Dirac equation held the Hartree potential of every
electron, and its unscreening removes only the two-component V_H[rho].  A
four-component charge would add the valence-valence Darwin and spin-same-orbit
terms of the Hartree field a second time (bcc Fe: -7 to -9 % on the orbital
moment).  The current terms are in no Dirac-Coulomb pseudopotential, and the
pair (two-component charge, Pauli current) obeys continuity exactly.

``bispinor_gw`` selects which Lorentz blocks are screened, never which carrier
represents them, so this resolver has exactly two outcomes, bispinor and not.
It stays a resolver rather than collapsing into ``bool(bispinor)`` because the
carrier stamp below is what the zeta, restart, SC-checkpoint and tangential
authenticators compare against, and those need ONE producer.  This module
imports nothing from the GW driver so it can be used at artifact altitude.
"""

from dataclasses import dataclass

#: The bispinor carrier stamp written into and authenticated from the restart
#: bundle, the SC checkpoint, the tangential stores, ``sigma_mnk.h5`` and the
#: eqp headers.  A store stamped with an earlier carrier
#: (``normalized_rkb_four_current_v1``, the raw or large-block stamps of
#: 2026-10) refuses by name.
PAULI_FOUR_CURRENT_REPRESENTATION = "pauli_charge_rkb_current_v1"
SOURCE_WFN_CHARGE_REPRESENTATION = "source_wfn_normalized_charge_v1"
#: One line naming the carrier, for the rank-0 startup block.
PAULI_FOUR_CURRENT_SUMMARY = (
    "charge psi_L^dagger psi_L (two-spinor density and q->0 head); current "
    "c Psi^dagger alpha Psi on the raw kinetic-balance lift (Pauli current)")


@dataclass(frozen=True)
class FourCurrentRepresentation:
    """Resolved carrier choices for one GW model.

    Charge and current body carriers are named separately so each consumer
    reads the one it contracts: the large-block carrier for charge, the raw
    kinetic-balance lift for current.  ``scalar_head_bispinor`` governs the
    canonical scalar dipole/head producer (False: the charge head is the
    two-spinor dipole).  Keeping those decisions together is what stops
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
    from common.bispinor_init import LARGE_BLOCK_LIFT, RAW_KINETIC_BALANCE_LIFT

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
        charge_lift=LARGE_BLOCK_LIFT,
        current_bispinor=True,
        current_lift=RAW_KINETIC_BALANCE_LIFT,
        scalar_head_bispinor=False,
        charge_representation=PAULI_FOUR_CURRENT_REPRESENTATION,
    )

