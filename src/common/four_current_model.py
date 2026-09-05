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
blocks are screened, never which four-spinor represents them.  The one
carrier dial is the deck's ``bispinor_current_balance`` (``current_lift``
below): ``kinetic`` keeps ``sigma.p`` for the spatial current, ``velocity``
lifts the CURRENT carrier with ``sigma.v``, ``v = p + dV_NL/dk``, so the
alpha^i vertex is the pseudo-Hamiltonian's velocity at first order in q.
The charge carrier is ``sigma.p`` in every case (its small-component
density is the O(alpha^2) Dirac density, not a current).  So this resolver
has three outcomes: not bispinor, bispinor/kinetic, bispinor/velocity.  It stays a resolver
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
VELOCITY_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION = (
    "velocity_kinetic_balance_per_channel_alpha_spatial_current_v1")


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
    #: The spatial-current lift FAMILY: ``"raw"`` (one sigma.p carrier for
    #: all three channels) or ``"velocity"`` (one carrier per channel).
    current_lift: str | None
    scalar_head_bispinor: bool
    charge_representation: str
    spatial_current_representation: str | None

    def current_lift_for(self, mu_L: int) -> str | None:
        """The loader selector for Lorentz label ``mu_L`` in {1, 2, 3}.

        ``raw`` for the shipped carrier; ``velocity_<mu_L>`` under the
        per-channel velocity balance (``common.bispinor_init``).  This is
        the ONE place the family name becomes a carrier name, so every
        consumer that loads a current carrier asks for its own channel.
        """
        from common.bispinor_init import (
            VELOCITY_KINETIC_BALANCE_LIFT, VELOCITY_KINETIC_BALANCE_LIFTS)
        mu_L = int(mu_L)
        if mu_L not in (1, 2, 3):
            raise ValueError(
                f"current_lift_for: mu_L must be 1, 2 or 3; got {mu_L}")
        if self.current_lift is None:
            return None
        if self.current_lift == VELOCITY_KINETIC_BALANCE_LIFT:
            return VELOCITY_KINETIC_BALANCE_LIFTS[mu_L - 1]
        return self.current_lift

    @property
    def one_current_carrier(self) -> bool:
        """True when all three channels ride the SAME four-spinor."""
        from common.bispinor_init import VELOCITY_KINETIC_BALANCE_LIFT
        return self.current_lift != VELOCITY_KINETIC_BALANCE_LIFT


def resolve_four_current_representation(
    bispinor: bool,
    model,
    *,
    current_lift: str | None = None,
) -> FourCurrentRepresentation:
    """Resolve all carrier decisions without importing the GW driver.

    ``model`` is accepted and ignored: both shipped ``bispinor_gw`` values
    ride the raw kinetic-balance carrier.  The parameter stays so the call
    sites keep naming the mode they resolved -- when a phase-3 mode needs a
    different carrier, this is the one function that has to learn about it.

    ``current_lift`` is the deck's ``bispinor_current_balance`` resolved to a
    lift family (``gw_config.LorraxConfig.bispinor_current_lift``):
    ``None``/``"raw"`` is the shipped ``sigma.p`` carrier for the spatial
    current, ``"velocity"`` the exact per-channel velocity balance
    (``common.bispinor_init.VELOCITY_KINETIC_BALANCE_LIFT``; one carrier
    per Cartesian channel, ``current_lift_for(mu_L)``).  It moves ONLY
    ``current_lift`` and ``spatial_current_representation``: the charge
    carrier and the scalar head/dipole producer stay on kinetic balance by
    design (owner 2026-09-04), so every charge-side stamp is unchanged and
    every current-side artifact (transverse zeta, finite-q alpha vertex,
    transverse basis receipt) authenticates against the new string.
    """
    from common.bispinor_init import (
        RAW_KINETIC_BALANCE_LIFT, VELOCITY_KINETIC_BALANCE_LIFT)

    lift = (RAW_KINETIC_BALANCE_LIFT if current_lift is None
            else str(current_lift).strip().lower())
    if lift not in (RAW_KINETIC_BALANCE_LIFT, VELOCITY_KINETIC_BALANCE_LIFT):
        raise ValueError(
            f"resolve_four_current_representation: current_lift={lift!r} is "
            f"not a spatial-current carrier; expected "
            f"{RAW_KINETIC_BALANCE_LIFT!r} or "
            f"{VELOCITY_KINETIC_BALANCE_LIFT!r}")
    if not bool(bispinor):
        if lift != RAW_KINETIC_BALANCE_LIFT:
            raise ValueError(
                "GATE bispinor_current_balance_requires_bispinor: "
                f"current_lift={lift!r} names a four-spinor spatial-current "
                "carrier, but bispinor = false has no spatial-current "
                "channel.\n"
                "  want: bispinor = true, or bispinor_current_balance = "
                "kinetic (the default)\n"
                "  doc:  docs/input_reference.md, bispinor_current_balance.")
        return FourCurrentRepresentation(
            charge_bispinor=False,
            charge_lift=None,
            current_bispinor=False,
            current_lift=None,
            scalar_head_bispinor=False,
            charge_representation=SOURCE_WFN_CHARGE_REPRESENTATION,
            spatial_current_representation=None,
        )
    spatial = (RAW_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION
               if lift == RAW_KINETIC_BALANCE_LIFT
               else VELOCITY_KINETIC_BALANCE_SPATIAL_CURRENT_REPRESENTATION)
    return FourCurrentRepresentation(
        charge_bispinor=True,
        charge_lift=RAW_KINETIC_BALANCE_LIFT,
        current_bispinor=True,
        current_lift=lift,
        scalar_head_bispinor=True,
        charge_representation=RAW_KINETIC_BALANCE_CHARGE_REPRESENTATION,
        spatial_current_representation=spatial,
    )
