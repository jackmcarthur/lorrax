"""The vocabulary, as declarative data (design §3.1.4, §4.1).

Two mappings, both importable and enumerable without solving anything
(``SERVICE_FORM.md:28``): a consumer that wants to know what the service
can be asked for must not have to ask for something to find out.

:data:`FAMILIES` is R4's two-by-two table written down.  A sampling
frequency ``z = ω + iϖ`` has a real part and an imaginary part, each
either zero or not, and each of the four combinations needs a different
family of quadrature rules:

===========  ==========================  =============================
             ω = 0                       ω ≠ 0
===========  ==========================  =============================
ϖ = 0        ``noncrossing``  (1/x)      ``crossing``  (HGL sine sum)
ϖ ≠ 0        ``noncrossing_imag`` /      ``damped_line``  — no
             ``complex_laplace``         in-process solver
===========  ==========================  =============================

Three of those cells have solvers.  Writing the table down as data is what
lets :func:`~minimax.door.family_for_character` refuse the empty cell BY
NAME (F6) instead of failing somewhere inside a kernel.

:class:`FamilySpec` carries two names: ``name`` is the door's ``family=``
argument (``noncrossing`` / ``crossing`` / ``noncrossing_imag`` /
``complex_laplace``), and ``route`` is the mathematical cell from the
design's R4 table (``exponential_sum`` / ``sine_sum`` /
``exponential_sum_imag`` / ``damped_line``).
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

    #: The ``family=`` argument of :func:`minimax.serve`.
    name: str
    #: The mathematical cell name from the design's R4 table.
    route: str
    #: The target this family approximates (a key of :data:`TARGETS`).
    target: str
    #: The analytic character of ``z`` this cell serves:
    #: ``'static'`` (ω=0, ϖ=0) | ``'real'`` (ω≠0, ϖ=0) |
    #: ``'imag'`` (ω=0, ϖ≠0) | ``'strip'`` (both nonzero).
    character: str
    #: The name of the request's range parameter.
    range_param: str
    #: Can :func:`minimax.serve` compute this family's rule in process?
    wired: bool
    description: str


#: The target vocabulary.  ``version`` is an integer that bumps when the
#: DEFINITION changes.
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
                    "sin(Delta t)  (THEORY_mpa_implementation.md)"),
        version=1,
        domain="strip"),
})


FAMILIES: Mapping[str, FamilySpec] = MappingProxyType({
    "noncrossing": FamilySpec(
        name="noncrossing",
        route="exponential_sum",
        target="inverse",
        character="static",
        range_param="R",
        wired=True,
        description="1/x as a sum of decaying exponentials on [1, R]."),
    "crossing": FamilySpec(
        name="crossing",
        route="sine_sum",
        target="hgl",
        character="real",
        range_param="A_dim",
        wired=True,
        description="The HGL regularized sign target as a signed sine sum."),
    "noncrossing_imag": FamilySpec(
        name="noncrossing_imag",
        route="exponential_sum_imag",
        target="inverse_imag",
        character="imag",
        range_param="R",
        wired=True,
        description=("x/(x^2 + omega_hat^2) as a sum of decaying "
                     "exponentials on [1, R].")),
    "complex_laplace": FamilySpec(
        name="complex_laplace",
        route="exponential_sum_imag",
        target="complex_laplace",
        character="imag",
        range_param="R",
        # No in-process solver: serve refuses it (UncertifiedSolveRefused).
        wired=False,
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
        wired=False,
        description=("The MPA kernel on the strip (both parts of z "
                     "nonzero).  No quadrature family serves this cell; "
                     "MPA builds its strip rules through the damped_* "
                     "constructors.")),
})


#: The four analytic characters, in the 2×2's reading order.  A tuple rather
#: than a set so the refusal message is deterministic.
CHARACTERS: tuple[str, ...] = ("static", "real", "imag", "strip")


def families_for_character(character: str) -> tuple[FamilySpec, ...]:
    """Every declared family serving ``character``, wired or not."""
    return tuple(f for f in FAMILIES.values() if f.character == character)
