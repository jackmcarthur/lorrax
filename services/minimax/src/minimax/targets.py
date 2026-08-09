"""The vocabulary, as declarative data (design §3.1.4, §4.1).

Two mappings, both importable and enumerable WITHOUT loading a catalog,
reading an ``.npz`` or solving anything — which is the whole point of
``SERVICE_FORM.md:28``: a consumer that wants to know what the service can
be asked for must not have to ask for something to find out.

:data:`FAMILIES` is R4's two-by-two table written down.  A sampling
frequency ``z = ω + iϖ`` has a real part and an imaginary part, each
either zero or not, and each of the four combinations needs a different
family of quadrature rules:

===========  ==========================  =============================
             ω = 0                       ω ≠ 0
===========  ==========================  =============================
ϖ = 0        ``noncrossing``  (1/x)      ``crossing``  (HGL sine sum)
ϖ ≠ 0        ``noncrossing_imag`` /      ``damped_line``  — DOES NOT
             ``complex_laplace``         EXIST (WP9)
===========  ==========================  =============================

Three of those cells have code; the fourth is the line
``gw/screening.py:527-531`` refuses at, and it is the wall the whole MPA
initiative is walking toward.  Writing the table down as data is what lets
:func:`~minimax.door.family_for_character` refuse the empty cell BY NAME
(F6) instead of failing somewhere inside a kernel — and it is what makes
the missing cell a row in a table rather than a new subsystem.

WHY THE CATALOG'S NAMES AND NOT THE DESIGN'S ROUTE NAMES.  The design
brief calls the four cells ``exponential_sum`` / ``sine_sum`` /
``exponential_sum_imag`` / ``damped_line``, which is what they are
mathematically.  The shipped catalog keys on ``noncrossing`` /
``crossing`` / ``noncrossing_imag`` / ``complex_laplace``, which is what
they are called on disk.  A door that renamed them would have to translate
at every lookup and would break byte-comparison against the shipped
bundle, so :class:`FamilySpec` carries BOTH: ``name`` is the catalog key
and the door's argument, ``route`` is the mathematical cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class TargetSpec:
    """What function a rule approximates, and on what domain."""

    name: str
    definition: str
    version: int
    #: ``'real'`` | ``'imag'`` | ``'strip'`` — where the argument lives.
    domain: str


@dataclass(frozen=True)
class FamilySpec:
    """One cell of R4's 2×2, plus what the door needs to serve it."""

    #: The catalog key, and the ``family=`` argument of :func:`minimax.lookup`.
    name: str
    #: The mathematical cell name from the design's R4 table.
    route: str
    #: The target this family approximates (a key of :data:`TARGETS`).
    target: str
    #: The analytic character of ``z`` this cell serves:
    #: ``'static'`` (ω=0, ϖ=0) | ``'real'`` (ω≠0, ϖ=0) |
    #: ``'imag'`` (ω=0, ϖ≠0) | ``'strip'`` (both nonzero).
    character: str
    #: The catalog field the request is quantized against.
    range_param: str
    #: Which catalog file carries this family.
    catalog: str
    #: Does the shipped bundle carry ANY entry for this family?  This is a
    #: measured fact about the bundle in the tree, restated here so that a
    #: consumer can see the hole without opening the catalog — and
    #: ``test_the_family_table_matches_the_shipped_bundle`` is what keeps
    #: it from drifting into a lie.
    shipped: bool
    #: Is the family reachable through :func:`minimax.lookup` today?  A
    #: family can be SHIPPED and not WIRED: ``complex_laplace`` ships
    #: eighteen certified entries whose selection needs a β axis the
    #: selection rule does not have yet, and wiring it wrong would serve a
    #: table fitted to a different function.  That wiring is
    #: ``feat/minimax-beta-selector-2026-08-08``'s commit, not this one.
    wired: bool
    #: The PHYSICS lever a refusal names — how a deck moves a request into
    #: the catalog.  Data, not arithmetic: the service does not own the
    #: map from a deck to a range parameter, it owns the fact that the
    #: parameter has a certified ceiling.
    range_lever: str
    #: The GENERATOR lever a refusal names.
    generator_hint: str
    description: str


#: The target vocabulary.  ``version`` is an integer that bumps when the
#: DEFINITION changes, which is the only thing that can invalidate a
#: shipped table without changing its bytes.
TARGETS: Mapping[str, TargetSpec] = MappingProxyType({
    "inverse": TargetSpec(
        name="inverse",
        definition="1/x on x in [1, R]",
        version=1,
        domain="real"),
    "inverse_imag": TargetSpec(
        name="inverse_imag",
        definition="x/(x^2 + omega_hat^2) on x in [1, R]",
        version=1,
        domain="imag"),
    "hgl": TargetSpec(
        name="hgl",
        definition=("G_hgl(u) on u in [0, A]; "
                    "Im[sqrt(pi/2) exp(-(u+i)^2/2)(1 + i erfi((u+i)/sqrt2))]"),
        version=1,
        domain="real"),
    "fermi": TargetSpec(
        name="fermi",
        definition="G_fermi(u) = tanh-regularized sign target on u in [0, A]",
        version=1,
        domain="real"),
    "complex_laplace": TargetSpec(
        name="complex_laplace",
        definition="1/(u - i*beta) on u in [1, R]",
        version=1,
        domain="strip"),
    "damped_line": TargetSpec(
        name="damped_line",
        definition=("K_z(Delta) = -2 int_0^inf dt e^{-varpi t} e^{i omega t} "
                    "sin(Delta t)  (MPA_THEORY_PLAN section B)"),
        version=1,
        domain="strip"),
})


#: ``target_kind='fermi'`` is REGISTERED AS A HOLE, not as a family.  It is
#: reachable only from the generator (survey §1.6), no shipped table carries
#: it, so every ``fermi`` request is a runtime solve by construction.  Under
#: R1 it becomes a refusal naming an empty family — either tabulate it or
#: delete the vocabulary entry, which is an owner row (design §12) and not
#: this branch's call.  It stays in :data:`TARGETS` so the refusal can name
#: it instead of calling it a typo.
_FERMI_IS_A_HOLE = True


FAMILIES: Mapping[str, FamilySpec] = MappingProxyType({
    "noncrossing": FamilySpec(
        name="noncrossing",
        route="exponential_sum",
        target="inverse",
        character="static",
        range_param="R",
        catalog="catalog.json",
        shipped=True,
        wired=True,
        range_lever=("lower R = x_max/x_min — the band-energy span over the "
                     "gap; R <= 1e5 is certified"),
        generator_hint=("tools/generate_minimax_assets.py "
                        "--family noncrossing "
                        "--r-max <R> --error-tier <eps>   (offline)"),
        description="1/x as a sum of decaying exponentials on [1, R]."),
    "crossing": FamilySpec(
        name="crossing",
        route="sine_sum",
        target="hgl",
        character="real",
        range_param="A_dim",
        catalog="catalog.json",
        shipped=True,
        wired=True,
        range_lever=("lower A_core = 2*omega_max/xi + 2*edge_factor — either "
                     "a narrower Sigma_c omega grid or a larger broadening "
                     "xi; A_dim <= 60 is certified"),
        generator_hint=("tools/generate_minimax_assets.py --family crossing "
                        "--a-max <A> --error-tier <eps>   (~1 core-hour, "
                        "offline)"),
        description="The HGL regularized sign target as a signed sine sum."),
    "noncrossing_imag": FamilySpec(
        name="noncrossing_imag",
        route="exponential_sum_imag",
        target="inverse_imag",
        character="imag",
        range_param="R",
        catalog="catalog.json",
        # MEASURED, and it is R1's structural hole: `catalog.json` carries
        # 31 entries in two families, `{'crossing': 5, 'noncrossing': 26}`.
        # There is no `noncrossing_imag` family at all, which is why
        # `build_imag_quadrature` -- on the path of EVERY GN-PPM run -- is a
        # 100% runtime-solve path with no table behind it.
        shipped=False,
        wired=True,
        range_lever=("lower R = x_max/x_min at fixed "
                     "omega_hat = omega_p/x_min"),
        generator_hint=("tools/generate_imag_minimax_assets.py   (the "
                        "complex_laplace sweep; its real part IS this "
                        "target -- design R4, one campaign not two)"),
        description=("x/(x^2 + omega_hat^2) as a sum of decaying "
                     "exponentials on [1, R].  ZERO shipped tables.")),
    "complex_laplace": FamilySpec(
        name="complex_laplace",
        route="exponential_sum_imag",
        target="complex_laplace",
        character="imag",
        range_param="R",
        catalog="catalog_complex_laplace.json",
        shipped=True,
        # WIRED, and the axis it was waiting for is `minimax.beta_selector`.
        # It stayed unwired while the selection rule had only R, tier and
        # node count -- three axes that round safely -- because beta does
        # not round: the target is a different function at every beta, so a
        # beta-blind match serves a table fitted to something else.  The
        # rule that answers this family is therefore NOT `_catalog`'s; the
        # door routes `complex_laplace` to the beta axis and refuses a
        # lookup that names no beta at all, which is the same safety
        # property the `wired=False` row used to buy, now bought by a rule
        # that can also say yes.
        wired=True,
        range_lever=("lower R, or move beta onto the tabulated ladder"),
        generator_hint=("tools/generate_imag_minimax_assets.py --r <R> "
                        "--beta <beta> --error-tier <eps>"),
        description=("1/(u - i*beta) on [1, R]: the two-dimensional "
                     "(R, beta) envelope whose beta=0 slice is the "
                     "noncrossing family and whose real part is "
                     "noncrossing_imag's target.")),
    "damped_line": FamilySpec(
        name="damped_line",
        route="damped_line",
        target="damped_line",
        character="strip",
        range_param="A_dim",
        catalog="catalog_damped_line.json",
        shipped=False,
        wired=False,
        range_lever=("no lever: the family does not exist.  This is the "
                     "cell gw/screening.py:527-531 refuses at"),
        generator_hint=("nothing to run yet -- WP9 is the campaign that "
                        "creates this family"),
        description=("The MPA kernel on the strip (both parts of z "
                     "nonzero).  R4's empty cell; F6's trigger.")),
})


#: The four analytic characters, in the 2×2's reading order.  A tuple rather
#: than a set so the refusal message is deterministic.
CHARACTERS: tuple[str, ...] = ("static", "real", "imag", "strip")


def families_for_character(character: str) -> tuple[FamilySpec, ...]:
    """Every declared family serving ``character``, wired or not."""
    return tuple(f for f in FAMILIES.values() if f.character == character)
