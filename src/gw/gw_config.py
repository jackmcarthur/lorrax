"""Unified configuration for LORRAX GW calculations.

``LorraxConfig`` is built once via :meth:`LorraxConfig.from_input_file`
from the ``[cohsex]`` section of ``cohsex.in`` and threaded through the
entire driver.  Its ~80 input keys are grouped into sub-dataclasses
along the same axes the input file's section comments already use:

    config.head        — q→0 Coulomb-head sources & overrides
    config.minimax     — screening-minimax target error / max nodes / table mode
    config.ppm         — PPM model + sigma quadrature + on-shell σ_c options
    config.sigma_grid  — ω-grid for Σ_c(ω) output
    config.sc          — self-consistency loop knobs (qp_solver = self_consistent)
    config.memory      — chunk sizing
    config.backend     — FFI/linalg backend selection
    config.debug       — debug-only flags & file paths
    config.bse         — BSE interpolation setup (htransform-driven)
    config.paths       — output filenames

The top-level ``LorraxConfig`` retains only system geometry
(``nval`` / ``ncond`` / ``nband`` / ``sys_dim``) and the orthogonal
mode flags (``compute_mode`` / ``qp_solver`` / etc.) that the
driver reads on the fast path.

Derived sub-objects (the math-internal ``MinimaxConfig`` from
``minimax_config.py``, one instance per quadrature consumer) and derived
data (the Σ_c(ω) grid) are constructed on demand via ``LorraxConfig``
properties.
"""

from __future__ import annotations

import configparser
import enum
import os
import re
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path

import numpy as np

from common.units import RYD_TO_EV

# The estimator vocabulary lives with the estimators, not here: a second copy
# of the value list is a second thing to keep in step.  ``band_extrapolation``
# imports only ``common``, so this cannot cycle.
from .band_extrapolation import (
    BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT,
    BAND_EXTRAPOLATION_ESTIMATORS,
    BRACKET_SCHEME_DEFAULT,
    BRACKET_SCHEMES,
)


# ---------------------------------------------------------------------------
#  Environment grammar — ONE boolean vocabulary for the GW init/config layer
# ---------------------------------------------------------------------------
#
# THIS IS THE CANONICAL COPY.  The tree still carries the same recognised
# token set in three other places, each with a reason:
#
#   * ``ffi/common/gate.py::MODE_SPELLINGS`` — three-valued (auto/off/on).
#     Its ``auto`` is load-bearing (a gate may DEMOTE and say so), so it is
#     deliberately NOT folded into a two-valued test; only its on/off halves
#     are the same vocabulary as ours.
#   * ``runtime.__init__._FALSY_TOKENS`` — the sanctioned two-valued
#     falsy-test resolver (the two-resolver doctrine,
#     docs/architecture/layers.md): ``runtime`` must resolve knobs BEFORE
#     jax/config imports are safe, so it keeps its own tiny parser.
# ``isdf/core.py``'s ``_env_bool`` — historically the fourth copy — was
# retired by the P1.3 unification (2026-07-31): ``isdf.core`` now imports
# :func:`env_bool` from here (L1→L1; ``gw/__init__`` pulls only this
# jax-free module, so the import adds nothing and cannot cycle).  The
# remaining copies are pinned set-equal by
# ``tests/test_env_grammar.py::test_defect3_vocabulary_has_not_drifted``,
# which reads them straight out of the source text, without importing jax,
# and fails on any drift.
#
# SEMANTICS:
#   unset or blank      -> the caller's default
#   a truthy spelling   -> True        (case- and whitespace-insensitive)
#   a falsy spelling    -> False
#   anything else       -> False, AND announced once (see ``env_bool``)
#
# Resolving an unrecognised token to something other than False would
# split the grammar between converted and unconverted readers of the same
# knob; adding telemetry does not.

# ONE GRAMMAR, AND IT NO LONGER LIVES HERE (2026-08-22).  The table, the
# announcement and the memo moved DOWN to ``runtime.env_flags`` — L3, no
# jax, no config — because the parsers that still swallowed an
# unrecognised token silently were in ``file_io`` and ``runtime``, which
# are L3 and may not import this module.  A grammar only the top layer can
# reach is a grammar the layers below re-invent, which is exactly what
# ``_slab_io_ffi._env_flag`` and ``runtime._env_falsy`` had done.
#
# The names below are re-exports, kept because this module's four owned
# files, ``isdf.core`` and ``tests/test_env_grammar.py`` all reach for
# ``gw_config.env_bool`` by name.  Behaviour is unchanged in every token.
from runtime.env_flags import (  # noqa: E402
    ANNOUNCED as _ENV_ANNOUNCED,
    ENV_FALSE as _ENV_FALSE,
    ENV_TRUE as _ENV_TRUE,
    env_bool,
    reset_env_announce_state,
)


def env_float(name: str, default: float, *, print_fn=print,
              refuse: bool = False) -> float:
    """Canonical numeric env parse: unset/blank → default, bad → ANNOUNCE
    (or, with ``refuse=True``, RAISE).

    The same defect class as :func:`env_bool`, one type along.  A
    ``try: float(...) except: default`` leaves the user believing a knob is
    in force when it is not — the exact failure the
    ``ISDF_CHUNK_TARGET_UTILIZATION`` parser used to commit.

    ``refuse=True`` is for knobs that GATE correctness rather than tune
    performance (``LORRAX_FH_ORTHO_TOL``): running with the default while
    the user believes a gate threshold is in force is itself the silent
    failure, so garbage refuses loudly, naming the variable — the
    announce-or-refuse doctrine's refuse half.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        if refuse:
            raise ValueError(
                f"{name}={raw!r} is not a number.  Accepted: a float "
                f"(e.g. '1e-6'), or unset/blank for the default "
                f"({default!r}).  Refusing rather than running with a "
                f"value the caller did not choose.") from None
        key = (name, raw)
        if key not in _ENV_ANNOUNCED:
            _ENV_ANNOUNCED.add(key)
            print_fn(f"  *** LORRAX SANITY: {name}={raw!r} is not a number; "
                     f"falling back to {default}.  The knob is NOT in "
                     f"force. ***")
        return default


# ---------------------------------------------------------------------------
#  ζ-truncating env knobs
# ---------------------------------------------------------------------------
#: Env knobs that make the ζ fit stop EARLY and still write a file.
#:
#: ``LORRAX_MAX_RCHUNKS=N`` breaks the r-chunk loop after N chunks
#: (``gw/isdf_fitting.py``), and the writer downstream of the loop still
#: calls ``mark_zeta_done`` — so the truncated ζ is stamped complete.  If
#: ``gw_init`` then stamps ``fit_provenance`` on it, ``_zeta_reuse_ok``
#: will REUSE that ζ in a later production run from the same directory,
#: because provenance records the *configuration*, which is identical.
#: The result is silently wrong physics from a profiling knob.
ZETA_TRUNCATING_ENV_KNOBS = ("LORRAX_MAX_RCHUNKS",)


def active_zeta_truncating_knobs() -> list[tuple[str, str]]:
    """``[(name, raw), ...]`` for every truncating knob currently in force.

    Blank counts as unset (the r-chunk loop's own guard is
    ``if _max_rchunks and ...``, so ``""`` does not truncate).
    """
    out = []
    for name in ZETA_TRUNCATING_ENV_KNOBS:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            out.append((name, raw))
    return out


# ---------------------------------------------------------------------------
#  XLA GPU memory environment — RE-EXPORTED, not defined here
# ---------------------------------------------------------------------------
#
# These four names used to be defined in this file, ~280 lines of them, and
# ``runtime.collect_startup_facts`` imported them from here — an L3 module
# reaching up into the GW driver package for something that knows nothing
# about GW.  They now live in :mod:`runtime.xla_memory`, next to
# ``runtime.set_default_env``, which is the code that decides which of these
# variables LORRAX ships.  See that module's docstring for the four traps it
# encodes and the measurements behind them (jobs 7882443 / 7882447).
#
# The re-export is not a compatibility shim to be swept away: ``gw_init``
# captions the ζ-fit peak and ``gw_output`` prints the startup banner from
# these, and reading them as ``gw_config.<name>`` is how those call sites and
# ``tests/test_env_grammar.py`` are written.  Keeping the alias here costs one
# import and keeps the deck-level vocabulary in one place.
#
# ``runtime`` imports jax only inside function bodies, so this import does NOT
# cost gw_config its jax-free property (the login-node config tests and
# ``gw_output``'s banner both depend on that).
from runtime.xla_memory import (       # noqa: F401
    XlaGpuMemoryEnv,
    XlaPoolReading,
    classify_xla_pool,
    resolve_xla_gpu_memory_env,
)


# ---------------------------------------------------------------------------
#  Enums
# ---------------------------------------------------------------------------

class ComputeMode(str, enum.Enum):
    """The single axis describing what self-energy is computed.

    Orthogonal to ``qp_solver`` (how QP energies are extracted from Σ):
    any mode can be wrapped in the ``self_consistent`` QSGW loop — the
    loop dispatches through the mode-agnostic
    ``sigma_dispatch.compute_sigma_xc`` (COHSEX and GN-PPM verified
    end-to-end; see reports/gw_refactor_map_2026-07-01/
    G0W0_SC_TOGGLE_DESIGN.md §4).

    - ``X_ONLY`` — bare exchange Σ_X = -G·V (no screening, no correlation).
    - ``COHSEX`` — static screened-exchange + Coulomb-hole.
    - ``GN_PPM`` — dynamic Σ_c(ω) via GN plasmon-pole (probe at iω_p).
    - ``HL_PPM`` — dynamic Σ_c(ω) via HL plasmon-pole (probe at real Ω).
    - ``MPA`` — dynamic Σ_c(ω) from an n-pole multipole fit of W on a
      double-parallel sample grid in the complex-ω plane (complex poles
      Ω_p, residues B_p).  **DECLARED, NOT YET RUNNABLE** — see
      :data:`UNIMPLEMENTED_MODES` and
      :func:`refuse_unimplemented_compute_mode` below.

    WHY THE VALUE IS SPELLED ``mpa`` AND NOT ``full_freq``.  Every value
    on this axis names the *ansatz* for W's frequency dependence, not the
    numerical machinery that follows from it: ``cohsex`` is "W at ω = 0",
    ``gn_ppm`` / ``hl_ppm`` are "one plasmon pole, fitted this way".  The
    next member of that series is "n poles, fitted to a sampled W", whose
    name in the literature is the multipole approximation, so ``mpa`` is
    the spelling that keeps the axis reading as one list of ansätze.

    ``full_freq`` was the rejected alternative, and it was rejected for
    two reasons rather than taste.  First, it names a *family* — contour
    deformation, real-axis quadrature and MPA are all "full frequency" —
    so a deck that set it would still have to say which one, which is a
    second axis, which is precisely the thing the "single axis" wording
    at the top of this docstring exists to prevent.  Second, it would
    spend the good name: a genuinely numerical full-frequency Σ (no pole
    model at all) is a plausible future member of this enum, and it
    should be able to be called ``full_freq`` when it arrives instead of
    finding the name already taken by a pole method.  The owner-facing
    shorthand for this work is still "FF"; the deck key is ``mpa``.
    """

    X_ONLY = "x_only"
    COHSEX = "cohsex"
    GN_PPM = "gn_ppm"
    HL_PPM = "hl_ppm"
    MPA = "mpa"

    @property
    def needs_screening(self) -> bool:
        """True for COHSEX / GN-PPM / HL-PPM / MPA; False for bare X."""
        return self is not ComputeMode.X_ONLY

    @property
    def is_dynamic(self) -> bool:
        """True when the mode builds a Σ_c(ω) grid: GN/HL-PPM and MPA.

        The honest reading is "this run has an ω axis", which is what the
        consumers of this property want to know (the σ-cube layout gate,
        ``qp_solver = fixed_point``'s ω-grid requirement, ``GWResults.
        use_ppm``).  It is deliberately NOT the same question as "is this
        a plasmon-pole model" — that one is :attr:`ppm_model`, and the
        two questions differ for exactly one member, ``MPA``.
        """
        return self in (ComputeMode.GN_PPM, ComputeMode.HL_PPM,
                        ComputeMode.MPA)

    @property
    def ppm_model(self) -> str | None:
        """``'gn'`` for GN-PPM, ``'hl'`` for HL-PPM, else None.

        None for MPA as well as for the static modes: MPA is dynamic but
        is not a plasmon-pole model, so any site that means "which of the
        two two-point PPM fits" must ask THIS and handle None, never
        ``is_dynamic`` with an ``else`` that assumes GN.
        """
        return {
            ComputeMode.GN_PPM: "gn",
            ComputeMode.HL_PPM: "hl",
        }.get(self)


#: The LEGACY spellings of the self-energy axis, and the canonical key that
#: replaces each.  ``compute_mode`` / ``qp_solver`` are the load-bearing axes
#: (see :meth:`LorraxConfig.compute_mode` / :meth:`LorraxConfig.qp_solver`);
#: these five booleans/strings are the vocabulary that predates them and that
#: every deck in the tree still writes.
#:
#: They are still parsed and still honored -- a deck that names one keeps
#: running -- but naming one now prints a deprecation note saying what to
#: write instead.  A key honored in silence beside a canonical twin is how a
#: tree ends up with two vocabularies for one axis and no way to tell which
#: one a given run resolved through.
#:
#: RETIREMENT IS A SEPARATE DECISION and has not been taken; this row is the
#: warning stage of it.  The migration shape is the tree's own
#: (``nband`` -> ``number_bands``, ``sigma_band_extrapolation`` ->
#: ``use_band_extrapolation``): note first, remove later.
LEGACY_SIGMA_AXIS_KEYS: dict[str, str] = {
    "do_screened": "compute_mode = x_only | cohsex | gn_ppm | hl_ppm | mpa",
    "use_ppm_sigma": "compute_mode = gn_ppm | hl_ppm",
    "ppm_model": "compute_mode = gn_ppm | hl_ppm",
    "self_consistent": "qp_solver = self_consistent",
    "sigma_at_dft_energies": "qp_solver = one_shot_dft (now the default)",
}


def announce_legacy_sigma_axis_keys(named_keys, resolved_mode, resolved_solver,
                                    *, print_fn=print) -> tuple[str, ...]:
    """Print one deprecation note per LEGACY self-energy-axis key the deck named.

    Returns the keys announced, so a caller (or a test) can assert on them
    rather than scraping the log.  Nothing is refused and nothing resolves
    differently: this is the warning stage of the migration described on
    :data:`LEGACY_SIGMA_AXIS_KEYS`.
    """
    named = frozenset(str(k).strip().lower() for k in (named_keys or ()))
    hit = tuple(k for k in LEGACY_SIGMA_AXIS_KEYS if k in named)
    if not hit:
        return ()
    mode = getattr(resolved_mode, "value", resolved_mode)
    solver = getattr(resolved_solver, "value", resolved_solver)
    print_fn(
        f"  [config provenance] this deck names {len(hit)} LEGACY "
        f"self-energy-axis key(s); they are honored, and the canonical axes "
        f"resolved to compute_mode = {mode}, qp_solver = {solver}.")
    for key in hit:
        print_fn(f"    {key} -> write {LEGACY_SIGMA_AXIS_KEYS[key]} instead")
    return hit


class SigmaChannel(str, enum.Enum):
    """One term of Σ that a compute mode either builds or does not.

    These are the channels the driver's outputs are written FROM — the
    names on ``sigma_dispatch.SigmaResult`` and the operands of the QP
    ladders in ``gw_output`` — not every intermediate a kernel touches.

    - ``X`` — bare exchange Σ_x = −G·V.  Built by every mode; it needs no
      screening and every output that reports a Σ decomposition wants it.
    - ``SX`` — static screened exchange Σ_SX = −G·W(0).
    - ``COH`` — the Coulomb hole Σ_COH.  SX and COH are one pair in
      practice (a mode that builds one builds the other) but they are two
      datasets and two columns, so they are two channels here.
    - ``C_OMEGA`` — dynamic correlation Σ_c(ω) on an ω grid, whatever
      analytic model produced it.
    """

    X = "x"
    SX = "sx"
    COH = "coh"
    C_OMEGA = "c_omega"

    @property
    def label(self) -> str:
        """How the channel is spelled in prose and in operator messages.

        The enum VALUE stays a lowercase identifier because it is data —
        it keys tables and appears in tests.  Messages an operator reads
        want the physics spelling, and having both means neither has to
        compromise.
        """
        return {
            SigmaChannel.X: "Σ_X",
            SigmaChannel.SX: "Σ_SX",
            SigmaChannel.COH: "Σ_COH",
            SigmaChannel.C_OMEGA: "Σ_c(ω)",
        }[self]


#: WHICH Σ CHANNELS EACH MODE ACTUALLY BUILDS — the table the writers ask
#: instead of hand-checking mode strings.
#:
#: This formalises a rule the tree was already obeying by hand.  The QSGW
#: plotting appendix (``gw_output.write_qsgw_qp_ladders``, landed
#: 2026-08-08) omits ``qp_static_cohsex_ev`` on a run that did not build
#: Σ_SX and Σ_COH and says so in one rank-0 line, because the alternative
#: — putting a different operator under that dataset's name — is worse
#: than the dataset's absence.  That judgement is right and it is not
#: specific to that writer or to those two channels, so the fact it turns
#: on lives here, once, and the writer reads it.
#:
#: THE RULE FOR ADDING A MODE: give it a row.  Every member of
#: :class:`ComputeMode` must appear (``tests/test_ff_compute_mode.py``
#: fails otherwise), and :func:`sigma_channels_for` refuses by name for a
#: mode with no row rather than returning an empty set — an empty set
#: would read as "this mode builds nothing", which is a legible answer,
#: and a legible wrong answer is the failure this table exists to stop.
#:
#: MPA's row is what the fit stage WILL build: Σ_x as usual, plus Σ_c(ω)
#: evaluated from the complex poles (Ω_p, B_p) instead of from a
#: two-point plasmon-pole fit.  Same channels as the PPM modes, different
#: producer — which is exactly why the table alone cannot be the safety
#: net, and why the mode also refuses to run (below).
MODE_SIGMA_CHANNELS: dict[ComputeMode, frozenset[SigmaChannel]] = {
    ComputeMode.X_ONLY: frozenset({SigmaChannel.X}),
    ComputeMode.COHSEX: frozenset({SigmaChannel.X, SigmaChannel.SX,
                                   SigmaChannel.COH}),
    ComputeMode.GN_PPM: frozenset({SigmaChannel.X, SigmaChannel.C_OMEGA}),
    ComputeMode.HL_PPM: frozenset({SigmaChannel.X, SigmaChannel.C_OMEGA}),
    ComputeMode.MPA: frozenset({SigmaChannel.X, SigmaChannel.C_OMEGA}),
}


def coerce_compute_mode(mode) -> ComputeMode:
    """Accept a :class:`ComputeMode`, its ``.value``, or a bare string.

    The writers reach this table holding whatever their caller handed
    them — a resolved enum from ``config.compute_mode`` in the driver, a
    plain string in a deck-echo path, an object carrying ``.value`` in a
    unit test's stand-in config.  Normalising in ONE place is what lets
    the table be the single answer rather than the third mode-string
    hand-check in the tree.

    An unrecognised spelling raises the same ValueError shape the config
    parser raises, naming the legal set — a typo never resolves to a
    default.
    """
    if isinstance(mode, ComputeMode):
        return mode
    raw = getattr(mode, "value", mode)
    try:
        return ComputeMode(str(raw).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"compute_mode={raw!r} is not a known mode; expected one of: "
            f"{', '.join(m.value for m in ComputeMode)}."
        ) from exc


class HeadCorrection(str, enum.Enum):
    """Finite-grid treatment of the singular macroscopic ``q -> 0`` head.

    ``FULL`` is the physical default: an irreducible direct response is
    completed with its microscopic head/body wings exactly once, while an
    already micro-reducible response (the BSE resolvent) is used as-is.
    ``NO_LOCAL_FIELDS`` is the explicitly diagnostic epsilon-head value, and
    ``OFF`` removes the special Gamma-cell contribution so brute-force k-grid
    convergence can be studied.  The diagram choice remains the orthogonal
    :class:`ScreeningDiagrams` axis.
    """

    FULL = "full"
    NO_LOCAL_FIELDS = "no_local_fields"
    OFF = "off"


def coerce_head_correction(value) -> HeadCorrection:
    """Normalize the public head policy without a silent fallback."""
    if isinstance(value, HeadCorrection):
        return value
    raw = getattr(value, "value", value)
    try:
        return HeadCorrection(str(raw).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"head_correction={raw!r} is not a known policy; expected one "
            f"of: {', '.join(v.value for v in HeadCorrection)}."
        ) from exc


class ScreeningDiagrams(str, enum.Enum):
    """WHICH DIAGRAMS build the W that Σ consumes — the screening axis.

    Orthogonal to :class:`ComputeMode` (which Σ *ansatz* is evaluated) and
    to ``screening_method`` (how the χ₀ frequency integral is done).  Those
    two say *at which frequencies* W is wanted and *how the quadrature is
    taken*; this one says *which series* W sums.

    - ``W_RPA`` — the random-phase approximation, ``W = (1 − Vχ₀)⁻¹V``.
      The only screening LORRAX had before 2026-08-15 and the default, so
      a deck that does not name this key is bit-identical to every deck
      written before it.
    - ``W_BSE`` — ladder-corrected W: ``W(ω) − v = v (ω − H)⁻¹ v`` with the
      statically screened direct rung ``−W(0)`` in the kernel of ``H``.
      Two-stage by construction — the RPA ``W(0)`` of the first stage IS
      the ``W_R`` the ladder kernel consumes — which is why this value
      changes the dataflow rather than one solver call.

    WHY AN ENUM AND NOT A BOOL.  ``ladder_screening = true`` would name the
    one alternative that exists today and spend the axis: the resolvent
    formalism admits more than one diagram set (TDA vs full symplectic,
    test-charge vs test-electron), and each is a *value* on this axis, not
    a second boolean beside it.  The same reasoning that spelled
    ``compute_mode`` as an enum of ansätze rather than ``use_ppm_sigma``
    (see :class:`ComputeMode`'s docstring) applies here.

    NOT EVERY COMBINATION IS SUPPORTED.  ``w_bse`` is refused at parse
    time against ``x_only``, ``hl_ppm``, the self-consistent QP solver,
    ``mc_average_placement != off`` and a declared metal
    (``mpa_material_class = metal``) — see
    :func:`refuse_unsupported_screening_diagrams`, which carries the
    reason for each.  INSULATORS ONLY is the one of those that a deck key
    cannot always express: a metallic WFN on a deck that declares nothing
    is refused at the stage instead, on the occupations themselves
    (``gw.screening_bse``, the same ``w_bse_insulators_only`` id).
    """

    W_RPA = "w_rpa"
    W_BSE = "w_bse"


def coerce_screening_diagrams(value) -> ScreeningDiagrams:
    """Accept a :class:`ScreeningDiagrams`, its ``.value``, or a string.

    Same shape and same reason as :func:`coerce_compute_mode`: the parser,
    a hand-built stub config and a deck-echo path all reach the axis
    holding different spellings of the same request, and normalising in
    ONE place is what keeps the dispatch a single answer.  A typo raises
    naming the legal set — it never resolves to the default.
    """
    if isinstance(value, ScreeningDiagrams):
        return value
    raw = getattr(value, "value", value)
    try:
        return ScreeningDiagrams(str(raw).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"screening_diagrams={raw!r} is not a known screening diagram "
            f"set; expected one of: "
            f"{', '.join(d.value for d in ScreeningDiagrams)}."
        ) from exc


#: WHICH COMBINATIONS OF ``screening_diagrams = w_bse`` v1 REFUSES, and why.
#:
#: Each entry is ``rule_id -> (predicate, got, want, fix, doc)``; the
#: refusal text is assembled from the five so every message has the same
#: five parts and no rule can be added without answering all of them.
#: Predicates take the resolved :class:`LorraxConfig`.
#:
#: SUPPORTED, deliberately absent from this table: ``cohsex``, ``gn_ppm``,
#: ``mpa`` and ``qp_solver = one_shot_dft`` / ``fixed_point``.
_W_BSE_REFUSALS: tuple[tuple[str, object, object, str, str, str], ...] = (
    (
        "w_bse_needs_a_screened_mode",
        lambda cfg: cfg.compute_mode is ComputeMode.X_ONLY,
        lambda cfg: f"compute_mode = {cfg.compute_mode.value}",
        "a mode that consumes W: cohsex, gn_ppm, or mpa",
        "drop screening_diagrams (or set it to w_rpa) for a bare-exchange "
        "run, or pick a screened compute_mode",
        "x_only builds no W at all, so a ladder-corrected W would be "
        "computed and discarded.  A key that parses and changes nothing is "
        "the ctsp defect (docs/architecture/decisions.md, 2026-08-06 "
        "ruling), and it is refused here rather than repeated",
    ),
    (
        "w_bse_hl_ppm_broadening_unimplemented",
        lambda cfg: cfg.compute_mode is ComputeMode.HL_PPM,
        lambda cfg: f"compute_mode = {cfg.compute_mode.value}",
        "cohsex (ladder W(0)), gn_ppm (ladder W(0) + W(i*omega_p)), or mpa",
        "use gn_ppm, whose probe sits on the imaginary axis where the "
        "resolvent needs no broadening policy, or keep w_rpa for hl_ppm",
        "the HL probe is a REAL-axis frequency, and (z - H)^-1 on the real "
        "axis needs a broadening (eta / xi) policy that this tree does not "
        "have a single answer for: GN-PPM and MPA already evaluate Sigma at "
        "silently different broadenings on the same deck "
        "(KNOWN_LORRAX_ISSUES.md:131 and :134 -- a 5.7x gap, floor applied "
        "by one side only).  Improvising a third convention for the ladder "
        "on top of that is how the discrepancy would become permanent, so "
        "v1 refuses instead.  Same shape as the wired-but-refused "
        "schur_avg placement (src/gw/head_channel.py:153)",
    ),
    (
        "w_bse_self_consistency_unimplemented",
        lambda cfg: cfg.qp_solver is QPSolver.SELF_CONSISTENT,
        lambda cfg: f"qp_solver = {cfg.qp_solver.value}",
        "qp_solver = one_shot_dft (the default) or fixed_point",
        "run w_bse single-shot; for a QSGW loop keep screening_diagrams = "
        "w_rpa until the per-iteration cycle lands",
        "the ladder reads its kernel W_R back out of the restart file the "
        "same run just wrote, so every SC iteration needs its own "
        "persist/reload cycle with its own provenance.  That cycle is not "
        "built; running without it would feed iteration N's ladder the "
        "iteration-1 W(0) and report the result as QSGW-hat.  Named "
        "deferral, DESIGN_2026-08-15.md section 1",
    ),
    (
        # ``mpa_material_class`` is the authority for which MPA material
        # formulation the deck requests.  The metal branch now also requires
        # explicit smearing metadata, but that metadata does not weaken this
        # rule: every certified w_bse path is insulating, so the material
        # class alone is the complete parse-time predicate.  A fractional
        # WFN whose deck never declares metal is caught by the runtime
        # occupation gate (gw/screening_bse.py, the same rule id).
        "w_bse_insulators_only",
        lambda cfg: (str(getattr(cfg.mpa, "material_class", "insulator"))
                     .strip().lower() != "insulator"),
        lambda cfg: f"mpa_material_class = {cfg.mpa.material_class}",
        "mpa_material_class = insulator (the default) -- w_bse v1 serves "
        "insulating systems only",
        "drop mpa_material_class (or set it to insulator) for a gapped "
        "system, or keep screening_diagrams = w_rpa for a metal",
        "the ladder operator, its TRS-gauge machinery and every "
        "certification this feature has are INSULATOR-DERIVED: integer "
        "occupations and a gapped D throughout (the pair basis is a "
        "band-index cut at nelec, and the resolvent's poles are the "
        "transition energies that cut produces).  Partial occupations "
        "enter BOTH -- the pair basis gains partially-blocked transitions "
        "and (z - H)^-1 gains poles at ~0 -- in ways nothing here has "
        "measured, so the run would produce a complete, plausible W under "
        "a diagram set that was never verified for it.  The metallic MPA "
        "screening/Sigma pipeline is live under screening_diagrams = w_rpa; "
        "that is the supported alternative, not evidence that the distinct "
        "ladder operator has acquired fractional-occupation semantics",
    ),
    (
        "w_bse_head_placement_unimplemented",
        lambda cfg: str(cfg.head.mc_average_placement) != "off",
        lambda cfg: f"mc_average_placement = {cfg.head.mc_average_placement}",
        "mc_average_placement = off (the default)",
        "leave mc_average_placement at off under w_bse, or keep "
        "screening_diagrams = w_rpa to use the BGW head placement",
        "head_correction=full obtains q=0 from the micro-reducible ladder "
        "resolvent itself. mc_average_placement moves a finite-q W head "
        "scalar AFTER the Dyson solve, which is a second post-solve policy "
        "for the same singular channel; composing the two has not been "
        "derived or certified, so it remains a named refusal",
    ),
)


def refuse_unsupported_screening_diagrams(config) -> None:
    """Refuse the ``w_bse`` combinations v1 does not serve, at PARSE time.

    Called from :meth:`LorraxConfig.from_input_file` once the record
    exists, because every predicate here reads a RESOLVED axis
    (``compute_mode`` and ``qp_solver`` are properties that fold in the
    legacy flags) and re-deriving them beside the parse would be a second
    opinion about the same question -- the shadow-accounting failure
    class, QUALITY_PATTERNS #3.

    NO-OP FOR ``w_rpa``, evaluated first and returning before any property
    is touched: a default deck must not acquire a new parse-time
    resolution -- and hence a new possible refusal -- from this function
    existing.
    """
    diagrams = coerce_screening_diagrams(
        getattr(config.screening, "diagrams", ScreeningDiagrams.W_RPA))
    if diagrams is not ScreeningDiagrams.W_BSE:
        return
    for rule_id, predicate, got, want, fix, doc in _W_BSE_REFUSALS:
        if not predicate(config):
            continue
        raise ValueError(
            f"GATE {rule_id}: screening_diagrams = w_bse is refused with "
            f"{got(config)}.\n"
            f"  got:  screening_diagrams = w_bse, {got(config)}\n"
            f"  want: {want}\n"
            f"  fix:  {fix}\n"
            f"  why:  {doc}.\n"
            f"  doc:  docs/input_reference.md '## Screening', "
            f"screening_diagrams.")


def sigma_channels_for(mode) -> frozenset[SigmaChannel]:
    """The Σ channels ``mode`` builds, per :data:`MODE_SIGMA_CHANNELS`."""
    resolved = coerce_compute_mode(mode)
    try:
        return MODE_SIGMA_CHANNELS[resolved]
    except KeyError as exc:
        raise KeyError(
            f"compute_mode={resolved.value!r} has no row in "
            f"MODE_SIGMA_CHANNELS.  Every ComputeMode member needs one: "
            f"add the row beside the enum in gw_config.py saying which of "
            f"{', '.join(c.value for c in SigmaChannel)} this mode builds."
        ) from exc


def mode_builds_channels(mode, *channels: SigmaChannel) -> bool:
    """True when ``mode`` builds ALL of ``channels``."""
    built = sigma_channels_for(mode)
    return all(c in built for c in channels)


def explain_missing_channels(mode, *channels: SigmaChannel) -> str:
    """The named-omission clause for channels ``mode`` does not build.

    Phrased as a fragment so a writer can put it in parentheses after the
    name of whatever it is declining to write, which is the shape the
    QSGW appendix's line already had.
    """
    resolved = coerce_compute_mode(mode)
    built = sigma_channels_for(resolved)
    absent = [c for c in channels if c not in built]
    if not absent:
        return f"nothing missing: compute_mode = {resolved.value} builds them"
    return (f"{'/'.join(c.label for c in absent)} "
            f"{'are' if len(absent) > 1 else 'is'} not built by "
            f"compute_mode = {resolved.value}")


# ---------------------------------------------------------------------------
#  Modes that are DECLARED but do not yet run
# ---------------------------------------------------------------------------
#
# A mode lands on this axis before its Σ stage lands, deliberately: the
# value, the parser, the channel row and every dispatch site are the part
# that has to be right BEFORE any kernel exists, because that is the part
# a half-finished mode silently falls through.  What this dict buys is the
# guarantee that "declared" never means "quietly ran as something else".
#
# The driver calls ``refuse_unimplemented_compute_mode`` at entry, before
# any heavy stage, so the operator learns in the first second of the run
# rather than after the ζ fit.  Every mode-dispatch site downstream ALSO
# refuses MPA by name rather than relying on that entry check — the entry
# check is the courtesy, the site-level refusals are the safety.
#
# REMOVING A ROW IS THE LANDING GESTURE.  When the MPA fit stage lands,
# its author deletes this entry and the suite that pins the refusal fails
# loudly until it is rewritten to pin the new behaviour.
UNIMPLEMENTED_MODES: dict[ComputeMode, str] = {}


def refuse_unimplemented_compute_mode(mode, *, context: str = "this run"):
    """Refuse a declared-but-not-yet-built compute mode, by name.

    No-op for every mode whose Σ stage exists, so the call is free to sit
    on the driver's fast path.  Raises :class:`NotImplementedError` —
    distinct from the ``ValueError`` a *typo* gets from the parser,
    because the two are different operator mistakes and deserve different
    words: ``compute_mode = mpaa`` is "no such mode", ``compute_mode =
    mpa`` is "that mode, not yet".
    """
    resolved = coerce_compute_mode(mode)
    reason = UNIMPLEMENTED_MODES.get(resolved)
    if reason is None:
        return resolved
    raise NotImplementedError(
        f"compute_mode = {resolved.value} is declared but not yet "
        f"implemented, so {context} refuses rather than running a "
        f"different self-energy ansatz under that name: {reason}")


class QPSolver(str, enum.Enum):
    """How QP energies are extracted from Σ — orthogonal to ``compute_mode``.

    The three states are mutually exclusive answers to the same physics
    question, each naming a standard method:

    - ``ONE_SHOT_DFT`` — textbook G0W0 (THE DEFAULT).  Σ is built once
      from the DFT inputs and *everything* is evaluated at E_DFT: the
      eqp0/eqp1 text outputs (at-DFT Newton + Z-linearization, as always)
      AND the QSGW-symmetrised Σ_xc whose eigh produces ``E_qp_ry`` /
      ``qp_wfn_rotations.h5`` / ``WFN_qp.h5``.  No iteration of any kind.
    - ``FIXED_POINT`` — one-shot Σ + diagonal on-shell solve
      E = h0 + ReΣ(E) for the QSGW-build evaluation energies
      (eigenvalue-only; Σ is never rebuilt).  Dynamic modes only — static
      Σ has no ω-grid to solve on.  ``sigma.sigma_at_dft_extrapolate`` is a
      sub-knob of this state (scissor for out-of-grid bands).
    - ``SELF_CONSISTENT`` — full QSGW loop (:mod:`gw.sc_iteration`):
      Σ rebuilt each iteration from rotated ψ + the previous iteration's
      E.  Loop knobs live in :class:`SCConfig` (``config.sc``).

    eqp0.dat / eqp1.dat keep the same formula in all three states; only
    the provenance of Σ changes under ``SELF_CONSISTENT`` (converged Σ,
    still evaluated at E_DFT — one more at-DFT Newton step from the SC
    fixed point).
    """

    ONE_SHOT_DFT = "one_shot_dft"
    FIXED_POINT = "fixed_point"
    SELF_CONSISTENT = "self_consistent"



#: The two W Dyson plans (``gw/w_isdf.py``) — the ONLY legal resolved
#: values of the ``w_dyson_solver`` input key.
_W_DYSON_PLANS = ("local", "distributed")


def normalize_w_dyson_solver(value) -> str:
    """Normalise a ``w_dyson_solver`` spelling to one of the TWO plans.

    Single source of the vocabulary — the parser and
    ``w_isdf._resolve_w_solve_fn`` both call this, so a spelling cannot
    mean different things at parse time and solve time.

    - ``local`` / ``auto`` / None → ``"local"`` (the q-parallel per-q
      dense LU; ``auto`` is a permanent back-compat alias).
    - ``distributed`` → ``"distributed"`` (the 2-D-sharded stacked-GEMM
      backsolve through the distrib_la plan door).
    - ``lu`` → ``"local"`` with a DeprecationWarning (it was the same
      route under its old name).
    - ``lstsq`` → ``ValueError``: the SVD min-norm inner solve was
      REMOVED in the two-plan cleanup (2026-07-27) — old decks fail
      informatively instead of silently rerouting.
    """
    s = ("auto" if value is None else str(value)).strip().lower()
    if s == "lu":
        import warnings
        warnings.warn(
            "w_dyson_solver = lu is deprecated: the per-q pivoted LU is "
            "now spelled 'local' (and is the default).  Update the deck "
            "to w_dyson_solver = local.",
            DeprecationWarning, stacklevel=2)
        s = "local"
    if s == "lstsq":
        raise ValueError(
            "w_dyson_solver = lstsq was REMOVED (two-plan W cleanup, "
            "2026-07-27).  The two plans are 'local' (per-q pivoted LU, "
            "default) and 'distributed' (2-D-sharded ScaLAPACK/cuSOLVERMp "
            "backsolve).  lstsq existed as a rank-deficiency fallback; a "
            "rank-deficient A = 1 - V·chi0 means the centroid basis has "
            "over-completed the pair-density rank — reduce n_mu (fewer "
            "centroids) or raise zeta_rcond instead of masking it with a "
            "min-norm solve.")
    if s == "auto":
        return "local"
    if s not in _W_DYSON_PLANS:
        raise ValueError(
            f"w_dyson_solver={value!r} invalid; expected "
            f"local (default; auto is an alias) or distributed.")
    return s


#: WHERE the last :func:`eigh_backend_choices` answer came from:
#: ``"distrib_la.BACKEND_CHOICES"`` (the live import) or ``"fallback (…)"``
#: naming the ImportError that forced the literal.  A returned tuple cannot
#: say which, and the two are EQUAL today — so without this the one test
#: that guards the drift passes identically in both worlds.
EIGH_CHOICES_SOURCE = "not called"


def eigh_backend_choices() -> tuple:
    """The legal ``eigh_backend`` spellings — the RESOLVER's own list.

    Read from :data:`distrib_la.BACKEND_CHOICES` so the parser and the
    thing that actually dispatches cannot drift.  They HAD drifted:
    this parser accepted only ``auto|off|cusolvermp|slate`` while the
    resolver had grown ``distributed`` (the portable "spread ONE tile over
    the mesh" spelling, and the ONLY eigh backend that exists on a host
    mesh, where it means ScaLAPACK ``pzheevd``) and ``scalapack``.  The
    effect was that the low-memory eigh could not be requested at all
    through a GW input file on CPU — the very platform it is needed on.

    ``BACKEND_CHOICES`` is importable with NO ``.so`` on the machine — that
    is a distrib_la door promise, precisely so a deck parser never needs
    the FFI layer.  The literal fallback below covers the remaining case,
    a tree whose ``services/`` is not on the path at all; it is pinned
    equal to the door's list by ``tests/test_bse_setup_qchunk.py``.

    :data:`EIGH_CHOICES_SOURCE` records WHICH of the two answered, because
    a test comparing the two lists cannot: they are equal today, so the
    comparison passes whether the live import ran or the except branch
    caught it, and the drift this function exists to prevent would recur
    with no signal at all.
    """
    global EIGH_CHOICES_SOURCE
    try:
        from ffi import _services
        _services.ensure_on_path()
        from distrib_la import BACKEND_CHOICES
    except ImportError as exc:
        # NARROW, and only around the two IMPORTS.  This used to be a bare
        # ``except Exception`` wrapped around three failure points AND the
        # dict subscript, so a KeyError from a door that renamed the op --
        # a CONTRACT break -- was indistinguishable from "this tree has no
        # services/ directory" and was answered with a frozen literal.  The
        # subscript is outside the guard now: if the door stops publishing
        # an ``eigh`` row, that raises here rather than being papered over.
        EIGH_CHOICES_SOURCE = f"fallback ({type(exc).__name__}: {exc})"
        return ("auto", "off", "distributed", "cusolvermp", "slate",
                "scalapack")
    EIGH_CHOICES_SOURCE = "distrib_la.BACKEND_CHOICES"
    return tuple(BACKEND_CHOICES["eigh"])


def distrib_la_batched_route_choices() -> tuple[str, ...]:
    """User-facing batch-route vocabulary from the ``distrib_la`` door.

    ``auto`` preserves the resolved backend's established scan/stacked-FFI
    route.  ``batch_reshard`` moves the batch axis onto the device mesh and
    runs the service's local JAX kernel on whole per-device matrices.  Keep
    this resolver beside :func:`eigh_backend_choices`: deck and CLI parsers
    must not grow frozen copies of a service-owned vocabulary.
    """
    try:
        from ffi import _services
        _services.ensure_on_path()
        from distrib_la import BATCHED_ROUTE_CHOICES
    except ImportError:
        # A source-only install can parse decks before services/ is installed.
        # The service contract test pins this fallback equal to the live list.
        return ("auto", "batch_reshard")
    return tuple(BATCHED_ROUTE_CHOICES)


def resolve_distrib_la_batched_route(
        params, *, override: str | None = None) -> str:
    """Resolve the universal distrib_la batch schedule from deck/CLI input.

    ``use_low_mem_eigh`` says that one complete eigensolver tile cannot fit
    on a device; ``batch_reshard`` requires exactly that residency.  Refuse
    the contradictory pair here, before either the GW or standalone BSE
    drivers allocate a matrix.
    """
    raw = (override if override is not None else
           (params.get("distrib_la_batched_route", "auto")
            if hasattr(params, "get") else "auto"))
    route = str("auto" if raw is None else raw).strip().lower()
    choices = distrib_la_batched_route_choices()
    if route not in choices:
        raise ValueError(
            f"distrib_la_batched_route={route!r} invalid; expected "
            f"{' / '.join(choices)}.")
    if (route == "batch_reshard" and hasattr(params, "get")
            and bool(params.get("use_low_mem_eigh", False))):
        raise ValueError(
            "distrib_la_batched_route='batch_reshard' contradicts "
            "use_low_mem_eigh=true: batch_reshard places one complete "
            "matrix (and its result/workspace) on each participating "
            "device, while use_low_mem_eigh asserts that whole-tile "
            "residency is not safe.  Keep distrib_la_batched_route=auto "
            "for the robust face-sharded eigensolver route, or disable "
            "use_low_mem_eigh only when a full tile fits.")
    return route


def resolve_eigh_backend(params, *, override: str | None = None) -> str:
    """``(eigh_backend, use_low_mem_eigh)`` → ONE backend string.

    THE single place the two spellings of one axis are combined, so a
    driver that reads the raw params dict (``bandstructure.htransform``,
    ``bse.exciton_bands``) gets the same answer as ``LorraxConfig``.

    ``override`` is a CLI ``--eigh-backend`` value, or ``None`` for "the
    deck decides".  It replaces the deck's ``eigh_backend`` and then goes
    through the same combination, which is the only spelling of the
    precedence that cannot drift: the flag names the LIBRARY, the deck key
    ``use_low_mem_eigh`` names the INTENT, and one axis has one resolver.
    The two CLI drivers used to do `args.X if args.X is not None else
    params.get(...)` inline and never called this function at all, so
    ``use_low_mem_eigh`` was parsed, stored, and read by nobody on either
    of them.

    * ``use_low_mem_eigh`` unset/false → ``eigh_backend`` verbatim.
    * true + ``auto`` → ``"distributed"`` (the platform's distributed
      library; ScaLAPACK on a host mesh, cuSOLVERMp on CUDA).
    * true + an explicit library name → that name (it already IS the
      distributed path).
    * true + ``off`` → ``ValueError``.  ``off`` pins the q-batched native
      eigh, which needs a WHOLE ``(rank, rank)`` matrix per device — the
      one thing the flag says is unaffordable.  Refusing at parse time is
      the doctrine: an explicit request that cannot be honoured never
      silently becomes its opposite.

    Vocabulary is checked here too, so an unknown spelling fails at parse
    time rather than at the first eigh.
    """
    if override is not None:
        raw = override
    else:
        raw = (params.get("eigh_backend", "auto")
               if hasattr(params, "get") else "auto")
    backend = str("auto" if raw is None else raw).strip().lower()
    choices = eigh_backend_choices()
    if backend not in choices:
        raise ValueError(
            f"eigh_backend={backend!r} invalid; expected "
            f"{' / '.join(choices)}.")
    low_mem = bool(params.get("use_low_mem_eigh", False)
                   if hasattr(params, "get") else False)
    if not low_mem:
        return backend
    if backend in ("auto", "native"):
        return "distributed"
    if backend == "off":
        raise ValueError(
            "use_low_mem_eigh = true with eigh_backend = off is a "
            "contradiction: 'off' pins the q-batched NATIVE eigh, which is "
            "the path that needs one whole (rank, rank) matrix per device — "
            "exactly what the low-memory flag says will not fit.  Either "
            "drop use_low_mem_eigh, or set eigh_backend = auto (resolves to "
            "'distributed') or name a library "
            "(distributed|cusolvermp|slate|scalapack).")
    return backend


# ---------------------------------------------------------------------------
#  Defaults — single source of truth for every input key
# ---------------------------------------------------------------------------

#: The production ζ rank-truncation cutoff.  ONE copy, importable.  Until
#: 2026-08-07 this number lived six times — here in ``_DEFAULTS`` and as
#: the signature default of five producer-side functions (one in
#: ``gw/isdf_fitting.py``, four in ``isdf/core.py``) — "mirrored", i.e.
#: kept equal by a comment, after a history of moving three times in one
#: day (1e-10 → 1e-6 → 1e-8, all 2026-07-21, one move re-freezing a
#: gate).  The five signature sites now import THIS name.  The measured
#: rationale for the VALUE is at the ``"zeta_rcond"`` deck entry below;
#: the value itself is owner-scoped (R19: lowering it on a noise-floor
#: argument would have cost a 5000 eV error) and did not change here.
ZETA_RCOND_DEFAULT: float = 1e-8

#: The production TRANSVERSE ζ rank-truncation cutoff τ (relative to
#: ``|λ|_max``, per q).  Same one-copy rule as :data:`ZETA_RCOND_DEFAULT`
#: above, and for the same reason: until 2026-08-22 this number lived THREE
#: times — here in ``_DEFAULTS`` and as the signature default of
#: ``gw/isdf_fitting.fit_zeta_to_h5`` and ``isdf/core.factor_c_q`` — so a
#: producer-side caller that omitted the argument silently picked up a
#: literal nobody would have thought to change.  The two signature sites now
#: import THIS name.  The measured rationale is at the
#: ``"transverse_zeta_rcond"`` deck entry below.
#:
#: NOTE what this value implies under the truncation policy: κ_cap = 1e10,
#: which is ABOVE the 1e8 certified for a PSD overlap Gram.  That is not an
#: oversight — the transverse channel is a different (indefinite) operator
#: with no production-deck measurement behind it, so its site is
#: UNCERTIFIED and warns rather than refusing
#: (``docs/dev/rank_truncation_policy.md`` §6, §9).
TRANSVERSE_ZETA_RCOND_DEFAULT: float = 1e-10

_DEFAULTS = {
    # System geometry
    "nval": 5,
    "ncond": 5,
    # ── THE BAND-COUNT FAMILY ───────────────────────────────────────────
    # FOUR keys, ONE resolver (:func:`resolve_band_counts`), and this dict
    # is the only place a NUMBER lives.
    #
    #   number_bands         the umbrella.  Sizes BOTH consumers.  100.
    #   number_bands_chi     χ0/W screening band count.   None = follow it.
    #   number_bands_sigma   Σ band-sum count.            None = follow it.
    #   nband                TRANSITIONAL ALIAS of ``number_bands``.  None.
    #
    # WHY THE SPLIT (2026-08-16, owner request).  ``number_bands`` sized two
    # convergence behaviours that are not the same behaviour, measured on the
    # Si 4×4×4 SOC deck:
    #
    #   * The **Σ** band sum extrapolates.  ``sigma_band_extrapolation``
    #     fits S_∞ + A/N from three partial sums and takes the truncation
    #     error from 106.2 meV MAE raw to 18.3 meV at 248 bands — i.e. Σ can
    #     be run at FEWER bands and corrected.
    #   * The **χ** band count does not.  Holding Σ fixed and sweeping only
    #     the screening's band count 40 → 248 moves band-edge Σ_CH by
    #     50–222 meV, and the last rung 224 → 248 still moves the median
    #     state by 40.7 meV NON-MONOTONICALLY, so there is no 1/N to fit.
    #
    # With one key the two cannot be configured apart, and the study above
    # had to vary BerkeleyGW's ``epsilon`` count to isolate the W side at
    # all.  "χ at full bands, Σ at fewer plus extrapolation" is both the
    # physically right configuration and the cheap one, and this is the pair
    # of keys that expresses it.
    #
    # WHY ``number_bands`` OWNS THE 100 AND ``nband`` IS None.  A default is
    # a number, and a number must live in exactly one entry or the two spell
    # different runs the day one of them moves.  ``number_bands`` is the
    # canonical spelling going forward (owner ruling 2026-08-16: the rest of
    # the deck migrates to ``number_bands_*`` shortly), so it holds the
    # value; ``nband`` is ``None`` = "the deck did not say", which is the
    # only way to tell a deck that pinned the alias from one that never
    # mentioned it.  Both spellings run; naming BOTH with DIFFERENT values
    # is a refusal, not a precedence puzzle.
    #
    # WHAT THE ISDF ζ FIT GETS: ``max(chi, sigma)``.  The interpolation basis
    # has to span the pair densities of whichever consumer reaches higher, so
    # the ψ this run loads spans ``[b0, b4)`` with ``b4`` built from the max
    # and the SMALLER consumer takes a narrower window inside it.  The winner
    # is logged (``BandCounts.describe``); a silent ``max`` is a day of
    # mis-debugging.
    "number_bands": 100,
    "number_bands_chi": None,
    "number_bands_sigma": None,
    "nband": None,
    # ζ-FIT BAND-WINDOW TOP, decoupled from the χ0/Σ band-sum top
    # (2026-08-11).  ``None`` (the default) means "follow ``nband``", which
    # is what this axis did for its whole history and what keeps every
    # existing deck bit-identical — the ζ fit's right band range is then
    # ``(b1, b4)`` exactly as before, PADDED extent and all.
    #
    # WHY IT EXISTS.  ``nband`` served two unrelated jobs: the top of the
    # χ0/Σ band sum (``b4``) and the top of the window ζ is fitted on.  The
    # ζ fit wants a NARROW window on an over-complete centroid set — the
    # htransform Galerkin leg the BSE's per-Q refit runs needs
    # ``n_μ·n_s ≥ nk·nb``, and on the Si 4×4×4 / 2628-centroid lineage the
    # measured capacity point is nb ≈ 52 — while the band sum wants a WIDE
    # one.  With one key for both, narrowing the fit window truncated the
    # band sum by the same eight bands and moved every quasiparticle level
    # by ~222 meV (median over the 4v8c window; 48 meV in the direct gap),
    # which is not a ζ-basis effect at all.  Set this instead and the band
    # sum keeps its bands.
    #
    # It only ever NARROWS: a ``zeta_nband`` above the ISDF fit's own top —
    # ``max(number_bands_chi, number_bands_sigma)`` since 2026-08-16 — is
    # refused, because the centroid ψ is loaded once over ``[b0, b4)`` and
    # there is nothing above b4 to fit.  (It narrows the fit INSIDE that
    # window; the χ/Σ split narrows the two band SUMS inside the same
    # window.  The three are independent and compose.)
    # Its edge takes a STRICT ``band_degeneracy`` check (an
    # explicit request is a new deck, so there is no census to grandfather):
    # a ζ-fit window that splits a multiplet fits half of an irrep, and ζ is
    # what the IBZ cascade unfolds.  See the ``nband`` entry in
    # docs/input_reference.md and gw.gw_init.fit_zeta.
    "zeta_nband": None,
    "sys_dim": 2,
    # Rebuild V_H from the CURRENT orbitals each self-consistent iteration
    # instead of rotating the fixed DFT one into the QP basis.  Off keeps
    # QSGW fixed-density, which is what every result before 2026-08-04 was.
    "density_self_consistent": False,
    # Run the SC loop's H / E / U on the STAR wedge, broadcasting back at
    # the boundary.  Sigma stays on the full BZ -- it is an FFT over the
    # k-grid.  Off keeps the loop entirely full-BZ.
    #
    # DEFAULT FLIPPED False -> True, 2026-08-15, on the owner's standing
    # directive that H^QP be built and eigh'd only on symmetry-reduced
    # k-points.  Two measurements paid for the flip, both on
    # ``gnppm_debug`` (file wedge 9, star wedge 5, so the two differ):
    #
    #   1. EQUIVALENCE, with ``sc_accelerator = linear``, ``sc_mixing = 1``:
    #      the on and off arms agree to **1e-6 meV** -- the ``%15.9f``
    #      print floor -- on E_QP at EVERY iterate and in the final
    #      eqp0/eqp1, with identical k coordinates.  The map is exactly
    #      k-set invariant, so the two arms have the same fixed point.
    #
    #   2. THE TRAJECTORY IS NOT, under the DEFAULT accelerator.  With
    #      rCROP the same pair diverges to 24.45 meV by map call 5 and
    #      113.3 meV in the final eqp0.  That is not a defect: rCROP's
    #      least-squares mixing minimises a residual norm summed over the
    #      loop's OWN k-set, so on the star wedge each orbit is counted
    #      once and on the full BZ with its multiplicity.  Different
    #      weights, different coefficients, same fixed point.  It does mean
    #      an UNCONVERGED rCROP run's iterates move when this flag moves --
    #      relevant given that the accepted Si QSGW run is on record as not
    #      converged.
    #
    # This had rotted invisibly (crash at the eqp writer, the two wedges
    # conflated) because no committed deck ran the SC path at all.  One now
    # does: tests/regression/gnppm_debug/gnppm_sc.in.
    "sc_on_ibz": True,
    # Update the q->0 head from the current QSGW Hamiltonian through saved
    # nearest-neighbour parallel-transport links.  Explicit opt-in preserves
    # every historical deck and makes a missing/stale artifact a refusal.
    # ``dft_velocity`` runs the same head chain on the artifact's exact DFT
    # p-matrix velocity stage only, without the links.
    "sc_head_update": "off",       # off | parallel_transport | dft_velocity
    "parallel_transport_file": "parallel_transport.h5",
    # Density-grid cutoff (Ry) for the psp matrix-element tools (kin_ion /
    # dipole).  None → the consumer defaults it to the WFN's own ``ecutwfc``.
    "ecutrho": None,
    # File paths
    "wfn_file": "WFN.h5",
    "centroids_file": "centroids_frac.txt",
    # Optional second centroid file used by the bispinor pipeline:
    # μ_L=1,2,3 (transverse) ζ-fits use Gordon-current-density centroids
    # rather than the charge-density centroids in ``centroids_file``.
    # Empty string == "not set" (cfg.centroids_file_current is None then).
    "centroids_file_current": "",
    "kin_ion_file": "kin_ion.h5",
    # Where H0's mean-field Hartree term comes from.  H0 = kin_ion + V_H is
    # a ~500 eV cancellation, so this is an explicit, validated choice
    # rather than something inferred from what happens to be on disk.
    #   auto   — stored 'v_hartree' array in kin_ion.h5 if present, else
    #            the legacy folded file if that is what it is, else isdf
    #   stored — require the exact array in kin_ion.h5 (raises if absent)
    #   isdf   — the ISDF V_q[0] tile (cohsex_sigma's Hartree kernel);
    #            distributed and in-loop capable, centroid-count dependent
    #   gspace — rebuild the exact FFT-grid matrix on the fly this run
    # See file_io/kin_ion.py's module docstring for the full contract and
    # the scorecard's S.5 table for the accuracy each buys.
    "hartree_source": "auto",
    # Three human-readable text outputs (always written), plus one opt-in
    # fixed-Sigma eigenvalue-self-consistent QP ladder:
    #   sigma_diag.dat — LORRAX-native per-(k,n) Σ-decomposition dump.
    #   eqp0.dat       — BGW-format zeroth-order QP energies.
    #   eqp1.dat       — BGW-format Z-linearized QP energies (Z=1 in
    #                    static COHSEX, central-difference Z in PPM).
    # The legacy ``output_file`` key (LORRAX-native eqp0.dat) and
    # ``eqp_output_file`` (unused) were dropped 2026-05-04; setting
    # them in cohsex.in now logs a deprecation warning and is ignored.
    "sigma_diag_file": "sigma_diag.dat",
    "eqp0_file": "eqp0.dat",
    "eqp1_file": "eqp1.dat",
    "eqp2_file": "eqp2.dat",
    # Rank-zero, human-readable calculation report.  This is the clean
    # application output; launcher placement and rank-binding diagnostics
    # remain in the launcher's own log.
    "report_file": "gwjax.out",
    "sigma_omega_h5_file": "sigma_mnk.h5",
    # Core flags
    "restart": True,
    # ``write_restart_tensors``: does this run PERSIST tmp/isdf_tensors_*.h5
    # at all?  DEFAULT true — today's behaviour, unchanged, until the owner
    # rules otherwise (SPEC_qirr_restart_tensors.md §7 and
    # DESIGN_symmetry_restart_followup.md, "Owner decisions carried").
    #
    # A COMPLEMENT, NEVER AN ALTERNATIVE, TO q_irr STORAGE.  The two answer
    # different questions and the second does not retire the first:
    #   * this key is for a run that DISCARDS the artifact.  Nothing in
    #     ``gw_jax`` reads ``W0_qmunu``/``V_qmunu`` back, so a GW run with no
    #     BSE downstream spends its restart-write time (MEASURED on the Si
    #     production deck: 4.5 s of a 19.4-22.6 s warm wall, ~21%, and 2.01
    #     GB of disk) on bytes nobody opens.  Setting this false buys all of
    #     that back and buys nothing else.
    #   * q_irr storage is for a run that KEEPS the artifact and wants it 8x
    #     smaller and 8x faster to write, WITH the fold/unfold machinery
    #     exercised on every production restart load.  A run that feeds BSE
    #     needs the file; this key cannot help it, and the format work is
    #     what does.
    # Reading this as "we can skip the write instead of doing the format
    # work" would be the wrong trade: it optimises the runs that do not need
    # the tensor and leaves the runs that do exactly where they were.
    #
    # WHAT GUARDS THE DOWNSTREAM.  Nothing new.  A BSE run pointed at a
    # directory whose GW leg was told not to write refuses on the paths that
    # already existed: ``bse_io._find_restart_file`` raises FileNotFoundError
    # naming ``isdf_tensors_*.h5``, and every W0 consumer gates on the
    # ``W0_ready`` attr rather than on the dataset's presence (the April
    # all-zero-screening mechanism; see tests/test_bse_w0_ready_gate.py).
    # Suppressing the write makes the file ABSENT, which is the loudest of
    # the states those guards distinguish.
    "write_restart_tensors": True,
    # ``write_qsgw_datasets``: does this run add the QSGW / QP-ladder
    # appendix to sigma_mnk.h5?  DEFAULT false, and false is exactly
    # today's file — these four datasets have had no producer since
    # 2026-04-11, when the QP/output rewrite deleted the block that wrote
    # them along with a pile of genuinely dead code beside it.  Turning it
    # on restores them (owner ruling 2026-08-08: "gated by input file but
    # exist; a lot of people will want to plot that").
    #
    # WHAT IT ADDS, and where each one comes from:
    #   sigma_xc_qsgw_kij_ev       (nk, nb, nb) complex — the static
    #       Hermitian Σ_xc the QSGW ansatz builds, written in the basis
    #       the rest of the file's cubes are in (the QP basis under
    #       self-consistency, the DFT basis one-shot).
    #   qp_omega0_ev               (nk, nb) real — the QP ladder of
    #       H₀ + Σ_x + Σ_c(ω≈0), one extra eigh of an (nk, nb, nb).
    #   qp_diag_self_consistent_ev (nk, nb) real — the diagonal on-shell
    #       fixed point E = h₀ + ReΣ(E), host-side, ~100 iterations of a
    #       (nk, nb) map.
    #   qp_static_cohsex_ev        (nk, nb) real — the static COHSEX
    #       ladder, H₀ + Σ_SX + Σ_COH.  Written only by a run that BUILT
    #       those two channels, which today means compute_mode = cohsex;
    #       a PPM run says so in one line rather than putting a different
    #       operator under that name.  See ``gw.gw_output``.
    #
    # NOT A DEBUG FLAG, which is why it is here and not in ``debug``.  The
    # appendix is a supported output with a stable format: same k-set,
    # same k_storage stamp and the same four star-spread numbers as every
    # other cube in the file, so a plotting script reads it through the
    # same reader path (``file_io.read_star_map`` → the landed unfold) it
    # already uses for Σ_c.
    #
    # THE COST IS A WRITE AND TWO SMALL SOLVES, never a Σ kernel.  Nothing
    # behind this key recomputes screening or self-energy; if a quantity
    # was not built by the run's compute mode it is omitted and named,
    # rather than manufactured to fill a dataset slot.
    "write_qsgw_datasets": False,
    # ``restart_q_storage``: on WHICH q-set are V_qmunu / W0_qmunu stored?
    # DEFAULT auto — storage FOLLOWS THE WFN FILE.  This is the end state the
    # owner ruled for on 2026-08-08 ~13:20, quoted below, and it became
    # reachable on 2026-08-15 when the last reader learned to unfold.
    #   auto — the DEFAULT.  Store the pre-unfold IBZ wedge when the deck's
    #          centroid set is orbit-closed AND this run's q path actually
    #          reduced; the full BZ otherwise.  On a non-closed set (the Si
    #          production 960-centroid deck: 47 of 48 ops violating) this is
    #          byte-for-byte the old file.  On a CLOSED set it is ~8x smaller
    #          — MEASURED on ``si_bse_debug`` GW+BSE end to end at P=4:
    #          ``isdf_tensors_480.h5`` 541,335,584 -> 130,299,936 B (4.15x),
    #          BSE lowest-8 eigenvalues bit-identical, eqp0/eqp1/sigma_diag
    #          byte-identical.
    #   full — preserve the old bytes exactly, unconditionally.  The escape
    #          hatch for a frozen reference or a stale out-of-tree consumer,
    #          and the control arm of any A/B: it does not ask the closure
    #          question at all, so it cannot be changed by the answer.
    #   ibz  — REFUSE on a set that is not storable, naming which of the two
    #          conditions failed.  For a deck that believes it is closed and
    #          wants to be told the day it stops being.
    #
    # WHY THE DEFAULT MOVED, AND WHAT HAD TO BE TRUE FIRST.  It shipped
    # briefly defaulting to ``auto`` and the 2026-08-08 landing census
    # measured nine red cells across the GW and BSE restart paths — because
    # at that date THE READERS DID NOT UNFOLD.  That is the sentence this
    # comment used to carry as a standing warning, and it stopped being true:
    # the GW restart reader has unfolded since ``536cbac9``
    # (``file_io.tagged_arrays._unfold_wedge``, applied at ``:1098`` and
    # ``:1197``), and ``bse._MunuSlabPlan``'s refusal was lifted 2026-08-15
    # (``bse_loading.py:707-711`` unfolds through the SAME function).  The
    # cost argument behind that refusal was measured and did not survive:
    # 57.4 GiB/s unfold against 2.919 GiB/s disk, a 6-17x win at every size
    # tested, with mu^2 cancelling out of the comparison entirely.
    #
    # THE RULING THIS IMPLEMENTS, verbatim: "symmetries should not need an
    # auto mode — if symmetries are not to be used, the wavefunction file
    # should've been generated with no symmetries."  The WFN file already
    # answers the question this key asks.  ``auto`` IS "follow the file";
    # the key survives only as the escape hatch and the A/B control, and is
    # still slated for deletion by the GW+BSE restart consolidation
    # registered in tests/KNOWN_FAILURES.md.  Do not build on it.
    # See gw/restart_q_storage.py for the resolution and the seam, and
    # DESIGN_symmetry_restart_followup.md for the pre-unfold-persistence
    # decision this key selects.
    "restart_q_storage": "auto",
    # ``qp_rotations_k_storage``: on WHICH k-set is qp_wfn_rotations.h5
    # stored?  Same three words as ``restart_q_storage``, and DEFAULT auto
    # for the same reason — storage follows the WFN file.  The wedge here is
    # the FILE wedge (``wfn.kpoints``, ``sym.nk_red`` rows), which is the
    # k-set ``kirr_to_kfull`` already addressed, so a wedge-stored file needs
    # no change at all in ``postprocess.rotate_wfn_to_qp`` or ``gw.eqp_bgw``.
    #
    # ``auto`` IS NOT "reduce whenever symmetry allows".  ``U_mnk`` is a
    # stack of eigenvectors, defined up to a phase and up to a unitary
    # mixing inside a degenerate multiplet, so whether the off-wedge rows
    # are redundant depends on how THIS RUN made them: the SC loop under
    # ``sc_on_ibz`` broadcasts them from the wedge (redundant), while the
    # one-shot path runs an independent ``eigh`` at every full-BZ k (NOT
    # redundant).  ``file_io.qp_wfn.write_qp_rotations_h5`` therefore runs
    # the reader's own round trip on the arrays in hand and keeps the wedge
    # only when it reproduces them exactly; ``auto`` falls back to full-BZ
    # storage and says which array failed and by how much, and ``ibz``
    # refuses instead of falling back.
    "qp_rotations_k_storage": "auto",
    # ``compute_mode`` is the single axis describing the self-energy ansatz.
    # ``"auto"`` infers from the legacy ``do_screened`` / ``use_ppm_sigma`` /
    # ``ppm_model`` flags so existing input files keep working unchanged.
    # New input files should set ``compute_mode`` explicitly:
    #   "x_only" | "cohsex" | "gn_ppm" | "hl_ppm" | "mpa".
    #
    # ``mpa`` — the multipole-W ansatz, the owner's "FF" — PARSES TODAY AND
    # REFUSES TO RUN TODAY.  Its Σ stage has not landed, so the driver
    # stops at entry naming the mode rather than falling through to a
    # plasmon-pole run; ``auto`` never infers it, and no legacy flag
    # combination reaches it.  See ``UNIMPLEMENTED_MODES`` beside the enum
    # for why the value ships ahead of the kernels, and the ``ComputeMode``
    # docstring for why it is spelled ``mpa`` rather than ``full_freq``.
    "compute_mode": "auto",
    # ``qp_solver`` is the orthogonal axis describing how QP energies are
    # extracted from Σ (see the ``QPSolver`` enum).  ``"auto"`` resolves
    # from the deprecated ``self_consistent`` key (true → self_consistent)
    # and otherwise defaults to "one_shot_dft" (standard G0W0).  New input
    # files should set it explicitly:
    #   "one_shot_dft" | "fixed_point" | "self_consistent".
    "qp_solver": "auto",
    "do_screened": True,
    "bispinor": False,
    # The relative sign of the i[r, V_NL] commutator in the assembled
    # velocity, read by ``psp.get_dipole_mtxels`` and passed to
    # ``common.mtxel_sweep.dipole_operator``.  ``-1`` is the shipped
    # assembly and ``+1`` the arm that reproduces BerkeleyGW's q -> 0
    # head; the words "shipped" / "flipped" spell the same two.  Empty is
    # NOT DECLARED and resolves to the shipped sign -- and for that
    # reason it must be a STRING default: a float default would make
    # "unset" and an explicit "-1" indistinguishable, and the whole point
    # of the stamp this feeds is to say which arm a dipole.h5 was built
    # with.
    #
    # IT HAS TO BE HERE.  ``read_lorrax_input`` builds ``params`` from
    # this table alone, so a key absent from it is parsed, reported as
    # unrecognized and dropped -- the producer then reads its own default
    # and the run is the other arm.  Measured, not argued: the first
    # flipped-arm dipole.h5 came back stamped ``-1.0`` with
    # "1 unrecognized deck key(s)" in the log, which is this project's
    # named failure mode reproduced in one line of a deck.
    "vnl_velocity_sign": "",
    "do_G0": True,
    # Deprecated (2026-07-08): ``self_consistent = true`` is honored as an
    # alias for ``qp_solver = self_consistent`` via auto-resolution.  SC is
    # wired for ALL modes (mode-agnostic sigma_dispatch), not just COHSEX.
    "self_consistent": False,
    # Self-consistency loop knobs (read only when qp_solver=self_consistent).
    # Promoted from the LORRAX_SC_* env vars (2026-07-08); the envs are
    # still honored as deprecated overrides.
    "sc_max_iter": 20,
    "sc_tol_ev": 1.0e-4,
    "sc_accelerator": "rcrop",   # rcrop | linear
    "sc_history_depth": 5,       # rCROP history depth
    "sc_mixing": 1.0,            # linear-mixing α (accelerator=linear only)
    "sc_dump_dir": "",           # E-history npy dump dir ("" = off)
    "sc_eigh": "auto",           # auto | native | distributed (per-iteration
                                 # eigh of the (nk, nb, nb) carry; a LAYOUT
                                 # choice, independent of the physics knobs)
    # Optional fourth text output beside the ordinary one-shot eqp0/eqp1
    # pair.  This iterates ONLY the eigenvalues/eigenvectors against the
    # already-computed full Sigma_c(omega) table: W, screening, and Sigma
    # diagrams are never recomputed.  The 1 meV default is a max|dE| test.
    "write_eqp2": False,
    "eqp2_tol_ev": 1.0e-3,
    "eqp2_max_iter": 20,
    "eqp2_accelerator": "rcrop",  # rcrop | linear (Picard)
    "eqp2_history_depth": 5,
    "use_ppm_sigma": False,
    # BGW-style averaging of diagonal Σ within degenerate sets (mirrors
    # ``Sigma/shiftenergy.f90`` band-averaging).  ``no_degen_averaging =
    # true`` disables it and emits the raw QE-basis-dependent diagonals.
    # ``degen_avg_tol_ry`` matches BGW's ``TOL_Degeneracy = 1e-6 Ry``.
    "no_degen_averaging": False,
    "degen_avg_tol_ry": 1.0e-6,
    # NOTE: ``slab_io`` and ``use_ffi_io`` were REMOVED as deck keys on
    # 2026-08-06 — see ``_LEGACY_DECK_KEYS``.  There is one sharded-slab
    # transport and the deck does not choose it.
    # ``accumulate_rchunk_to_gflat`` flat-axis chunker.  Bounds the
    # per-scan-iter FFT box ``chunk_size · n_rtot``.
    # 0 (default) = one-shot; the gflat memory model overrides this
    # at runtime when its planner picks a smaller value, but cohsex.in
    # > 0 wins over the planner.
    "gflat_chunk_size": 0,
    # V_q inner G-axis GEMM chunk size.  Bounds the per-q ``lax.scan``
    # working set inside the per-q V_q kernel.
    # 0 (default) = auto (``_pick_g_chunk(ngkmax)`` → largest divisor
    # of ngkmax ≤ 4096).
    "vq_g_chunk_size": 0,
    # ζ-fit solver path overrides (3-state).  Default ``auto`` picks
    # cuSolverMp on true 2D meshes (p_x ≥ 2 AND p_y ≥ 2) and the
    # JAX/CUDA fallback otherwise.  Force a path with ``on`` / ``off``.
    # Distributed dense-linalg backends (block-cyclic).  Portable axes —
    # the values name LIBRARIES, not vendors' key names:
    #   distributed_cholesky = auto | off | cusolvermp | slate
    #       charge-channel ζ-fit Cholesky.  auto → cusolvermp on true-2D
    #       GPU meshes, in-tree sharded_cholesky otherwise.  slate is the
    #       portability path (Frontier/Aurora); explicit request fails
    #       loudly if the FFI/library is absent (optional dependency).
    #   distributed_lu = auto | off | cusolvermp | scalapack
    #       transverse-channel LU.  scalapack = the host/CPU-backend
    #       backend (Cray LibSci pXgetrf+pXgetrs via liblorrax_ffi_host);
    #       explicit, never auto-picked.  (SLATE getrf not yet written.)
    "distributed_cholesky": "auto",
    "distributed_lu":       "auto",
    # Universal schedule for every array-returning ``distrib_la.Plan.batched``
    # call.  ``auto`` preserves the robust distributed/backend-batched route
    # selected by each plan.  ``batch_reshard`` is the small-matrix/high-HBM
    # opt-in: P(None,'x','y') -> P(('x','y'),None,None), local JAX linalg on
    # each device's whole matrices, then the inverse reshard.  Backend
    # resolution and the Plan I/O contract stay unchanged; this explicit
    # route runs the local JAX kernel in place of that backend for the call.
    "distrib_la_batched_route": "auto",
    #   eigh_backend = auto | off | distributed | cusolvermp | slate
    #                | scalapack
    #       Hermitian eigensolver for the BSE/htransform distributed-eigh
    #       sites (bse_setup fH_q, vq_interp coarse C_q tiles).  auto|off =
    #       the q-BATCHED native jnp.linalg.eigh (the measured default at
    #       every production tile size); the rest spread ONE tile over the
    #       whole mesh via the distributed-linalg FFI — the wide-band-window
    #       regime where a single matrix no longer fits on one device (square
    #       mesh + one process per device required; all guards fire at
    #       resolve time — see services/distrib_la +
    #       docs/services/distrib_la.md).
    #       ``distributed`` = the PLATFORM's distributed library (ScaLAPACK
    #       pzheevd on a host mesh, cuSOLVERMp on CUDA) and is the spelling
    #       that ports; the vocabulary is distrib_la's own, checked
    #       against it at parse time so the two can never drift.  The
    #       --eigh-backend CLI flag of htransform / exciton_bands OVERRIDES
    #       this key.
    "eigh_backend":         "auto",
    #   use_low_mem_eigh = true | false   (default false)
    #       The SAME axis named by INTENT instead of by library: "one whole
    #       (rank, rank) matrix does not fit on a rank, keep it spread over
    #       the mu x nu face".  true + eigh_backend=auto  =>  'distributed'.
    #       true + an explicit library name keeps that name.  true +
    #       eigh_backend=off is a CONTRADICTION and is refused at parse time,
    #       as is a true that cannot be honoured on this mesh — never a
    #       silent fall back to the whole-matrix path the flag exists to
    #       avoid.  See bandstructure.bse_setup.compute_wfns_fi.
    "use_low_mem_eigh":     False,
    # Charge ζ-fit OPT-IN Tikhonov ridge ε (added ON TOP of the fixed
    # 1e-14·|tr| non-singularity floor, as a fraction of the mean CCT
    # diagonal tr(C)/n): C_q ← C_q + [1e-14·|tr| + ε·|tr|/n]·I before the
    # replicated Cholesky.  A per-q SCALAR, so mesh-invariant.  Default 0.0
    # ⇒ bit-identical to the historical factor (frozen-golden contract).  A
    # POSITIVE ε conditions a NEAR-SINGULAR CCT (n_μ over-complete for the
    # system's pair-density rank) so ζ = (C+εI)⁻¹Z stops amplifying the
    # ULP-level, mesh-dependent pair-density (cuBLAS-GEMM-per-shard-dim)
    # roundoff into a grid-dependent V_q.  MoS2 6×6 (n_μ=1600) needs ε≈1e-4
    # to bring cross-grid Re Σ_c agreement from O(10 eV) to ~10 meV.  It
    # PERTURBS the physical result (the regularised answer is ε-dependent on
    # this ill-posed fit) — hence opt-in, a physics call.  Env override
    # LORRAX_ZETA_RIDGE.  See reports/gw_zeta_mesh_invariance_2026-07-20.
    "zeta_ridge":           0.0,
    # Charge ζ-solve conditioner (μ_L=0 channel only).  "rank_truncate"
    # (DEFAULT) = rank-revealing eigh pseudo-inverse: drop eigenvalues
    # < zeta_rcond·λ_max before inverting, so the near-null directions of
    # the over-complete charge CCT (n_μ > pair-density rank, κ~1e13) are
    # removed at the source instead of amplified by plain Cholesky into
    # O(1) V_q errors that GN-PPM magnifies to tens of eV (the conduction
    # Σc blow-up / device-count / nband instability).  "cholesky" = the
    # historical replicated/cuSolverMp Cholesky path (bit-identical to the
    # pre-feature behavior; the selectable alternative).
    "charge_zeta_solve":    "rank_truncate",   # rank_truncate | cholesky
    # ζ BACK-SOLVE TIER — how much of the (nq, μ, μ) charge factor is ever
    # replicated.  The first three tiers below are numerically free (same
    # per-q arithmetic, only the gathered extent differs); `distributed`
    # replaces the factorization as well and is the only one that scales.
    #   replicated  = today's path: gather the whole (q_batch, μ, μ) stack
    #                 onto every rank, nq·μ²·16 B, re-gathered per r-chunk
    #                 (18.9 GB/rank at MoS2 12×12 / μ=1998).
    #   per_q       = gather ONE (μ, μ) tile at a time, loop q — the slice
    #                 is taken inside a shard_map so the partitioner cannot
    #                 turn it back into the full-stack gather (it did until
    #                 workstream AA; scorecard Y.2).  μ²·(1+1/p_y)·16 per
    #                 execution and nq executions per r-chunk, so its TOTAL
    #                 per-r-chunk traffic is ≈ the replicated tier's while
    #                 its LIVE gather is nq× smaller: use it when memory,
    #                 not bandwidth, is the binder and the mesh is not
    #                 square enough for `distributed`.
    #   distributed = NOTHING O(μ²) is replicated.  Distributed eigh
    #                 (ScaLAPACK pzheevd), truncation on the replicated
    #                 spectrum, 2D-sharded C⁺, and a stacked GEMM C⁺@Z with
    #                 both operands 2D-sharded.  The ONLY tier whose
    #                 FACTORIZATION divides by P — the other two run one
    #                 dense eigh per q redundantly on every rank, O(nq·μ³)
    #                 with no P-scaling (~86 h at μ=10k).  EXPLICIT opt-in:
    #                 a block-cyclic eigh picks a different (equally valid)
    #                 gauge, so ζ matches the other tiers to ~κ·ε, not
    #                 bit-exactly.  Needs charge_zeta_solve='rank_truncate'
    #                 and a SQUARE or 1-D mesh (pXheevd descriptor rule);
    #                 refuses at resolve time otherwise.  On the transverse
    #                 channels it resolves to per_q (indefinite CCT — its
    #                 distributed route is distributed_lu=scalapack).
    #   auto (DEFAULT) = replicated while the gather fits under
    #                 LORRAX_ZETA_GATHER_CAP_GIB (4 GiB), per_q above it.
    #                 Never `distributed`.  Fixture-scale stacks stay on
    #                 replicated, so the default path is bit-identical to
    #                 the pre-feature one.
    # A SEPARATE env bound governs the `distributed` tier's TRANSPORT, not
    # its memory: LORRAX_COLLECTIVE_CHUNK_MB (128 MB) caps ONE emitted
    # collective's payload.  The 4 GiB gather cap was satisfied when job
    # 7876062 died at P=144 inside a single 1.15 GB Gloo AllGather; see
    # isdf/core.py's "COLLECTIVE PAYLOAD CHUNKING" note and scorecard AF.
    "distributed_zeta_solve": "auto",  # auto | replicated | per_q | distributed
    # Rank-truncation cutoff (relative to λ_max, per q).  DEFAULT 1e-8 —
    # the LOW end of the over-complete recovery plateau.  An over-complete
    # basis needs it: at MoS2 4×4/1204c, 1e-10 only partially recovers (MAE
    # 1.4 eV vs BGW) while the whole 1e-8…1e-4 plateau collapses to ~0.04 eV
    # — so pick the plateau's low end, because truncation is NOT free on a
    # well-conditioned basis: bulk Si 4×4×4/960c (the BGW-anchored
    # si_cohsex_3d gate) does have eigenvalues below the cut, and 1e-6 drifts
    # its sigTOT by 1.021 meV where 1e-8 costs only 0.054 meV — the identical
    # over-complete cure at ~20× less drift (sweep table in
    # docs/docs_gwjax/COHSEX_INPUT.md).  Env override LORRAX_ZETA_RCOND.
    # The isdf/core.py + gw/isdf_fitting.py signature defaults IMPORT
    # ZETA_RCOND_DEFAULT (defined above _DEFAULTS) — one copy, no mirrors.
    # reports/gw_rank_truncation_2026-07-20 + gw_bandrange_centroids_2026-07-21.
    "zeta_rcond":           ZETA_RCOND_DEFAULT,
    # Transverse ζ-solve family (bispinor μ_L=1,2,3 channels only; inert
    # otherwise).  "ridge" (DEFAULT) = the historical hoisted pivoted-LU
    # with the 1e-12·|tr|/n diagonal ridge — byte-identical to the
    # pre-feature behavior.  "rank_truncate" = per-q eigh pseudo-inverse
    # of the Hermitian INDEFINITE transverse CCT with an |λ| cut (drop
    # |λ| < transverse_zeta_rcond·|λ|_max): the charge channel's
    # conditioning cure ported to the transverse channel — TRS-paired
    # near-null current modes are REMOVED instead of inverted through at
    # the ridge floor (κ~1e12), and the per-q n_keep log doubles as the
    # transverse basis-adequacy instrument.  Grammar mirrors
    # charge_zeta_solve.  Its LOCAL plan (replicated whole-tile eigh,
    # q-parallel at P>1) runs at ANY centroid count on ANY mesh; its
    # DISTRIBUTED plan is selected by distributed_zeta_solve=distributed
    # (pzheevd at the padded extent — the mesh-divisibility constraint of
    # distributed_lu=scalapack does not apply).  distributed_lu is an LU
    # backend key and conflicts with this family: explicit
    # scalapack/cusolvermp + rank_truncate refuses at parse time.
    "transverse_zeta_solve": "ridge",   # ridge | rank_truncate
    # Transverse rank-truncation cutoff τ (relative to |λ|_max, per q).
    # Only read by transverse_zeta_solve=rank_truncate.  Default from the
    # 2026-08 MoS2 4×4 bispinor calibration ladder (eqp drift vs the
    # ridge control monotone in τ and within the 1e-4 eV gauge tolerance
    # across transverse set sizes).  No env twin (scorecard AV: policy
    # knobs live in the deck).  TRANSVERSE_ZETA_RCOND_DEFAULT (defined above
    # _DEFAULTS) — one copy, no mirrors.
    "transverse_zeta_rcond": TRANSVERSE_ZETA_RCOND_DEFAULT,
    # γ̃-double-contract kernel variant inside the monolithic pair
    # pipeline (see ``common.gamma_matrices.gamma_double_contract``).
    # Math identical across all three; differ in HLO structure.
    #   "take"   – jnp.take + element-wise phase mul (default).
    #   "einsum" – materialise the sparse γ̃ and contract via einsum.
    #   "scan"   – lax.scan over the (a, b) spin axis pairs.
    "gamma_contract_mode": "take",
    # Memory / chunking
    "memory_per_device_gb": 0.0,  # 0 = auto-detect
    # Owner-selected no-key policy.  Its pre-AOT P=4 premise (Si 80 Ry:
    # bc16 33 ms versus full-window 46 ms) reversed after the SM80 AOT merge
    # (31 versus 21 ms), so 16 is not asserted to be the faster universal
    # choice.  Explicit 0 opts into the full-window-first planner ladder; any
    # positive value remains an override.  The planner mesh-rounds and caps
    # all three forms at the logical zeta window.
    "band_chunk_size": 16,
    "r_chunk_size": 0,
    # Two-face 2-D-sharded ψ carrier (gw.wavefunction_bundle
    # layout="face") in place of the legacy four single-axis copies:
    # 2*S/(Px*Py) per-rank psi residency instead of 2*S/Px + 2*S/Py.
    # Default false = layout="legacy", the exact existing construction
    # path, bit-identical.  NOT an env var (decisions.md: physics- and
    # routing-relevant choices are declared inputs, not environment).
    # Narrow envelope while G/Sigma/head/rotation/exact-response
    # consumers are ported one at a time (see
    # reports/gwjax_low_mem_bands_audit_2026-08-22/report.md §6); an
    # unsupported combination refuses by name rather than silently
    # falling back to legacy.
    "low_mem_bands": False,
    # ISDF
    # Which of the TWO W Dyson plans solves A·W = V, A = (1 - Vχ₀):
    #   local (default; auto is an alias)
    #                per-q pivoted LU inside the q-parallel shard_map.
    #   distributed  2-D-sharded stacked-GEMM backsolve through the
    #                distrib_la plan door (ScaLAPACK on CPU meshes,
    #                cuSOLVERMp on CUDA); no rank ever materialises a
    #                full (μ, μ) tile.  Refuses loudly at resolve time
    #                on an unsupported mesh/build — never silently
    #                downgrades to local.
    # (lu → local with a DeprecationWarning; lstsq was removed.)
    "w_dyson_solver": "auto",
    "mc_average_vcoul_body": True,
    # One explicit BerkeleyGW-emulation contract for the metallic q=0 cell.
    # ``exact`` is the shipped LORRAX limit.  ``bgw_q0shift`` bundles the
    # three BGW conventions that must move together: no finite-q body-slot
    # averaging, the analytic-sphere q=0 bare-head estimator, and screening
    # sampled at a finite shifted q0.  It deliberately does not select an
    # occupation/spectral broadening or an MPA quadrature.
    "bgw_metal_q0_treatment": "exact",
    # Reduced reciprocal coordinates of BGW's epsilon q0 sample.  The
    # shipping comparison grid is 8x8x8, so (0,0,1/8) is one grid step.
    "bgw_metal_q0_vector": "0 0 0.125",
    # WHERE the q != 0 mini-BZ Coulomb average is APPLIED.  Orthogonal to
    # ``mc_average_vcoul_body``, which decides WHETHER an average is computed.
    #   "off"  (default) -- today's placement: <v> is substituted into the
    #          argmin |q+G| slots of the one production V tile, which is then
    #          both the Dyson operator and the Dyson right-hand side.  Every
    #          existing deck reproduces bit-identically here.
    #   "bgw"  -- BerkeleyGW parity: the average is applied to W's HEAD
    #          CHANNEL as a scalar per q-cell AFTER the Dyson solve, i.e.
    #          W_head = eps_c^-1 <v> with eps_c built from the bare v.  This
    #          is the placement BGW's Sigma and BSE both use
    #          (mtxel_cor.f90:1659-1662, intkernel.f90:887) and it is the
    #          EXACT cell average of the screened head under the f-sum-rule
    #          scaling chi ~ q^2.  Gamma is untouched.
    #   "schur_avg" -- the derivable target <W>_C = <v eps^-1>_C.  Wired and
    #          REFUSED; it needs chi inside the cell, which is an open
    #          question (COULOMB_AVG_ARCHITECTURE.md section 2.4).
    # See gw/head_channel.py for the derivation and the seam analysis.
    "mc_average_placement": "off",
    # Optional BerkeleyGW ``write_vcoul`` dump to source the mini-BZ
    # enhancement <v>/v_c from, instead of LORRAX's own estimator, when
    # mc_average_placement = "bgw".  Same override pattern as ``vhead`` /
    # ``whead_0freq``: it pins a cross-code comparison to BGW's byte values
    # so the residual left over cannot be a difference of Monte-Carlo
    # estimators.  Empty = use LORRAX's own <v> (which already agrees with
    # BGW's to 7e-4 - 2e-3 relative on every shell).
    "mc_average_placement_vcoul": "",
    # Per-Q mini-BZ Coulomb head cell-averaging (BGW minibzaverage_3d/2d).
    # False (default) = current behavior, BIT-IDENTICAL: the q→0 head is the
    # pure-Sobol mini-BZ mean and every finite-Q exchange head is the analytic
    # POINT value v(Q+G*).  True routes the head through
    # ``gw.coulomb.base.minibz_average``: the q→0 3D head gains the analytic
    # Baldereschi-Tosatti sphere term (seed-independent), the Voronoi fold
    # widens (nmax 1→3), and the BSE arbitrary-Q ``eval_vq`` head becomes the
    # mini-BZ CELL AVERAGE ``<v_LR(Q+G*)>_mBZ`` (fixes the 4-13% near-Γ /
    # zone-boundary point-vs-cell-average error, arbitrary_q_bse.md §16.4).
    # The winding (2D e^{-i2θ}) is unaffected — only the head magnitude is
    # averaged; the phase-factored ζ̃ rank-1 structure carries the direction.
    "head_minibz_average": False,
    # Singular Gamma-cell policy.  ``full`` is the shipping macroscopic W
    # head (microscopic local fields folded exactly once); the other two are
    # convergence/diagnostic arms, not alternative diagram sets.
    "head_correction": "full",
    # Opt-in finite-q W-av preprocessing.  The flags select the first and
    # second reciprocal-grid stencil shells written beside the PT data; they
    # do not activate the not-yet-complete metallic finite-q screening path.
    "w_av_first_neighbors": False,
    "w_av_second_neighbors": False,

    # BSE fine-grid densification.  When set to "NX NY NZ" (or "NX,NY,NZ") and
    # DIFFERENT from the coarse restart/WFN grid, the GENERAL BSE init
    # (``bse_io.load_bse_data_from_restart_sharded``) interpolates the ENTIRE
    # BSE problem — ψ, QP ε (htransform fH), V_Q exchange (vq_interp), and the
    # W direct term (zero-pad in R) — from the coarse grid onto this fine grid
    # BEFORE any solve, so EVERY BSE solver (exciton_bands / feast / nontda /
    # kpm / resolvent) transparently runs on the fine grid.  Each fine length
    # must be at least the matching coarse length; integer nesting is not
    # required (8x8x1→12x12x1 evaluates the same coarse Fourier polynomial).
    # Empty (default) or == the coarse grid → the coarse ``data`` bundle is
    # returned byte-identically (fast path untouched).  This is the native-
    # coarse route for a nonnested target; exciton_bands ``--w-coarse-grid``
    # instead decimates a native fine W and therefore remains nested-only.
    "bse_k_grid": "",
    "bare_coulomb_cutoff": None,
    # ζ-sphere cutoff (Ry).  When the writer emits zeta_q_G with per-q
    # WFN.h5-style spheres, this is the cutoff used to define the per-q
    # G-list.  Defaults to ecutwfc (mirrors the bare-Coulomb default);
    # max value is ecutrho.  Must be ≥ bare_coulomb_cutoff (V_q can't
    # use ζ̃(q+G) at G's the writer didn't store).
    "zeta_cutoff": None,
    # BGW vcoul override (for diagnostic BGW-vs-LORRAX comparison)
    "use_bgw_vcoul": False,
    "bgw_vcoul_file": "",
    # Aux WFN for pulling the 48-op crystal symmetry group when the main
    # WFN is nosym (its mf_header/symmetry/mtrx is truncated to identity).
    # Used only to fold LORRAX full-BZ q's onto BGW's IBZ q-list.
    "bgw_vcoul_sym_wfn": "",
    # Coulomb head
    "wcoul0_source": "s_tensor",
    "wcoul0_eta": 0.0,
    "vhead": None,
    "whead_0freq": None,
    "whead_imfreq": None,
    # Screening / minimax
    "screening_method": "minimax",
    # WHICH DIAGRAMS W sums.  Orthogonal to screening_method (how the chi0
    # frequency integral is taken) and to compute_mode (which Sigma ansatz
    # asks for W):
    #   "w_rpa" (DEFAULT) — W = (1 - V chi0)^-1 V, every deck written
    #        before 2026-08-15, bit-identical.
    #   "w_bse" — ladder-corrected W(omega) - v = v (omega - H)^-1 v with
    #        the statically screened direct rung -W(0) in H's kernel.  The
    #        RPA W(0) is still computed and persisted first; it IS the
    #        ladder kernel's W_R.  Refused against x_only / hl_ppm /
    #        qp_solver = self_consistent / mc_average_placement != off.
    "screening_diagrams": "w_rpa",
    # w_bse only: probe columns of the mu^2 ladder tile solved per block.
    # 0 (default) = the whole padded centroid basis in one block — the
    # historical behaviour, bit-identical for every existing deck.  A
    # positive value bounds the per-block solve memory; the facade rounds
    # it up to the mesh 'y'-axis multiple the reduce-scatter snapshot
    # tiles by.  This is DISPATCH GRANULARITY, not physics: chunking
    # changes when columns are solved, never their values.  Measured
    # motivation: with the knob unreachable, the fully relativistic LiF
    # 666-centroid lift solve attempted one 77.83-GiB block allocation
    # (JID 57288835), while the same deck at probe_chunk=64 passed every
    # production-W gate (JID 57280453).
    "ladder_probe_chunk": 0,
    # BerkeleyGW-compatible first-order Methfessel-Paxton width, in eV.
    # Zero preserves the historical step-occupation path.  The first
    # consumer is the per-iteration QSGW parallel-transport head; the same
    # key is intentionally reserved for later Green-function/screening
    # consumers so there is one occupation convention for the whole code.
    "occ_broadening": 0.0,
    "minimax_target_error": 1.0e-6,
    "minimax_max_nodes": 64,
    "regenerate_minimax_tables": False,
    "minimax_energy_reference": "midgap",
    # PPM
    # ppm_model picks the two-point pole-fit ansatz:
    #   "gn" — Godby-Needs: second probe at ω = i·ppm_omega_p (imaginary,
    #          ppm_omega_p ≈ 2 Ry by default).
    #   "hl" — Hybertsen-Louie: second probe at ω = ppm_omega_p (real,
    #          chosen above all transition energies; default 200 Ry).
    "ppm_model": "gn",
    "ppm_omega_p": 2.0,
    "ppm_fallback_omega": 2.0,
    # Override the head pole frequency Ω_h directly (Ry).  Useful for
    # testing against BGW's analytic head — set to BGW's
    # √(ω_p²/(1−ε_head⁻¹)) value to remove the LORRAX-vs-BGW
    # ε_head averaging convention as a source of disagreement.
    # None = compute Ω_h normally (analytic for HL, 2-pt fit for GN).
    "ppm_head_omega_h_ry": None,
    # Probe-χ₀ reuse (GN model only).  The probe-ω screening pass rebuilds
    # χ₀ with its own imaginary-axis minimax nodes — a second full τ sweep
    # (Gv/Gc build + FFTs + contraction per node) costing nearly as much
    # as the static pass (scorecard BC: 9.6 s vs 9.1 s at b300/P=16).
    #   "off"  (default) — dedicated probe quadrature, today's exact path.
    #   "auto" — represent the probe integrand x/(x²+ωp²) on the STATIC
    #        pass's τ nodes plus the MINIMAL augmentation from the
    #        dedicated quadrature's node set (Lawson-weighted fits;
    #        minimax_screening.refit_imag_alpha_augmented) at an error no
    #        worse than max(dedicated err, target_error); the probe χ₀
    #        then accumulates as a second weighted sum inside ONE fused τ
    #        sweep — shared nodes' tensors are computed once and only the
    #        k extras cost new compute.  Guaranteed fallback: with every
    #        extra node in, the exact dedicated weights are installed.
    # Numerics: same quadrature-error contract, different bits — NOT
    # bit-identical to "off" (pinned-baseline decks must keep "off" until
    # their references are re-pinned).  HL probes (real axis) always take
    # the dedicated path.
    "ppm_probe_chi_reuse": "off",
    "ppm_sigma_target_error": 1.0e-6,
    "ppm_sigma_max_nodes": 64,
    # Multipole W sampling / bounded Sigma consumption.
    "mpa_n_poles": 8,
    "mpa_material_class": "insulator",
    "mpa_sampling_alpha": 1,
    # The incumbent nested extension and the literal Leon/Yambo qPPS
    # continuation agree through eight poles and differ from nine onward.
    "mpa_sampling_schedule": "nested",
    # Pole identification only; residue fitting and guards remain common.
    "mpa_pole_solver": "loewner",
    "mpa_varpi_near_ry": 0.2,
    "mpa_varpi_far_ry": 2.0,
    # Height (Ry) of the metal near line's FIRST sample, z_1^1 = i*shift --
    # the published stability dodge around zero-energy intraband
    # transitions, NOT a broadening.  Unset (the default) = the published
    # 1e-5 Ha = 2e-5 Ry constant in ``gw.mpa.sampling._METAL_ORIGIN_SHIFT``,
    # which is bit-for-bit every grid built before this key existed.  A
    # METAL-ONLY key: an insulating deck's first sample is z = 0 exactly, so
    # setting it there is refused rather than ignored.  Ry like every deck
    # key, and therefore TWICE the Ha value the papers quote.
    "mpa_metal_origin_shift_ry": None,
    "mpa_pole_batch_size": 4,
    # Both targets bound the same dimensionless relative residual
    # |1-d Q(d)|.  They are separate because the positive crossing rule is
    # demonstrably less observable-sensitive than the support-selected
    # sign-definite rules on the validated MPA Sigma gate.
    "mpa_sigma_sector_target_error": 6.5e-4,
    "mpa_sigma_crossing_target_error": 2.0e-3,
    "mpa_sigma_max_nodes": 96,
    # Σ planner ω-clustering gap (Ry): with a gapped (patched) ω grid the
    # crossing core decomposes per cluster; a contiguous grid is always
    # one cluster and keeps the incumbent plan bit-for-bit.
    "mpa_sigma_omega_cluster_gap_ry": 1.0,
    # OCCUPANCY at which a band leaves a metallic Green's-function branch.
    # The Σ planner cuts on the branch WEIGHT (f on val, 1−f on cond), so
    # the applied floor is 1 − threshold: 0.995 ⇒ |weight| > 0.005.  It is
    # a magnitude, because MP1 f_kn is never clipped and a wrong-side band
    # carries a NEGATIVE weight that must be kept.  1.0 reproduces the
    # historical exact `weight != 0` rule bit-for-bit.  Metal-only: an
    # insulating branch has no weight and is untouched.
    "occupation_window_threshold": 0.995,
    # Sigma frequency grid
    "sigma_omega_min_ev": -5.0,
    "sigma_omega_max_ev": 5.0,
    "sigma_omega_step_ev": 0.25,
    # "" = the contiguous [min, max] grid.  "lo:hi, lo:hi" (eV) = a union
    # of uniform patches at sigma_omega_step_ev — the semicore dynamic-
    # range spelling (docs/dev/crossing-rule-cost-law.md).
    "sigma_omega_patches_ev": "",
    "sigma_regularization_ev": 0.25,
    # Effective-xi FLOOR, in eV.  "auto" (default) = the ansatz's own
    # conditioning floor: for the HGL plasmon-pole crossing quadrature that
    # is `ppm_windows.crossing_regularization_floor` = 2*omega_max/(24 -
    # 2*edge), which is why a GN-PPM run on a +/-5 eV grid at edge 1.5
    # silently ran at 0.4762 eV where the deck said 0.25; for MPA it is 0,
    # because MPA's crossing family is a positive real-time rule with its
    # own node ceiling and the HGL bandwidth derivation says nothing about
    # it.  A FLOAT is an explicit floor applied to EVERY ansatz -- the knob
    # that equalises xi across a cross-ansatz comparison, which is otherwise
    # confounded (1.90x apart on the sodium 48b deck, 5.7x on a +/-15 eV
    # window).  0 is legal and means "do not raise"; on an HGL ansatz that
    # re-opens the ill-conditioned regime the floor exists for, so it is
    # spellable on purpose and stamped so it cannot happen by accident.
    # Resolved ONCE by `ppm_windows.resolve_sigma_regularization` and
    # stamped into sigma_mnk.h5 beside the requested value.
    "sigma_regularization_floor_ev": "auto",
    "sigma_window_edge_factor": 1.5,
    # Σ_c(ω,k,m,n) end-of-stage layout (wk_REL ω-cube sharding workstream):
    #   "replicated" (default) — today's path: the per-rank (m_X, n_Y) host
    #       tiles are gathered into the FULL cube on EVERY rank
    #       (n_ω·nk·nb²·16 B replicated; 2751 MB/rank at nb=512).
    #   "sharded" — the tiles stay where the stacked psum_scatter left them
    #       on the existing 2-D mesh; consumers (head injection, diag/eqp
    #       interpolation, QSGW build, sigma_mnk.h5 SlabIO write) read the
    #       P(None,None,'x','y')-sharded cube directly.  Outputs are
    #       bit-identical to "replicated" (movement-only; A/B gated by
    #       tests/multi_device/sigma_omega_layout_ab.py under BOTH
    #       one_shot_dft and self_consistent).
    #       Refusal (at driver resolve time, never mid-run): an
    #       indivisible σ band window.  The second refusal here used to be
    #       slab_io=h5py_allgather at P>1; that tier no longer exists
    #       (2026-08-06), so the condition cannot arise.  There is no
    #       qp_solver refusal: the SC loop never rotates the cube (it is
    #       absent from the finalize `replace` at sc_iteration.py:1321),
    #       so there is no rotation seam to port, and the two layouts
    #       measure bit-identical under SC (jobs 7889782/7889789).
    "sigma_omega_layout": "replicated",
    # PPM sigma options
    # PPM invalid-pole treatment (BGW invalid_gpp_mode). 'zero' drops Omega^2<0
    # poles (BGW mode 0); '2ry' keeps the fit's fallback pole (BGW mode 2);
    # 'static_limit' (default, matching BGW's default mode 3) drops the
    # dynamical pole and adds the analytic static-COHSEX term for those
    # modes — see ppm_sigma._compute_invalid_static_sigma.
    "ppm_invalid_mode": "static_limit",
    # Band-convergence extrapolation of Sigma_c (gw.band_extrapolation).
    # ON by default since 2026-08-16 (owner ruling).  ON evaluates the
    # Sigma_c band sum at THREE band counts in one pass.  The default
    # bracket scheme is 80 %, 90 % and 100 % of the TOTAL **SIGMA** band
    # count (``number_bands_sigma``, NOT ``number_bands_chi``); the explicit
    # ``band_extrapolation_bracket_scheme`` below can instead select the
    # conduction-half / k-mean-energy-midpoint geometry.  Both prefer
    # degeneracy-clean interior boundaries, build three DISJOINT
    # band-bracket Green's functions per tau against one W(tau), and USE the
    # extrapolated Sigma_c as the E_nk that self-consistent iterations see.
    # Costs ~3x the Sigma tau-loop wall: G(tau) lives in the centroid basis,
    # so the FFT chain and the psi projection are paid once per bracket
    # regardless of how the bands are split.
    #
    # HARD GATE: REFUSES BY NAME when the Sigma band sum has n_cond < n_occ,
    # i.e. number_bands_sigma < 2*n_occ.  The owner's form is
    # "nband >= 2*N_electrons"; n_occ is the spin-convention-independent way
    # to write it (SOC/noncolin n_occ = N_electrons, otherwise
    # N_electrons/2).  THE COUNT IS THE SIGMA COUNT (merge ruling
    # 2026-08-16): the gate is a statement about the sum being fitted, and
    # ``number_bands_chi`` is never read by the planner -- raising it does
    # not clear the refusal, and the message says so.  See
    # gw.band_extrapolation.plan_band_brackets.
    #
    # PER-STAGE AUTO-DISABLE: the 1/N -> 0 limit is MODE-DEPENDENT and is
    # WRONG for a static Coulomb hole (measured: static COHSEX MAE 94.9 ->
    # 288.2 meV as nband 60 -> 124, ANTI-converging, against GN-PPM 171.3 ->
    # 32.8).  So on a non-PPM compute_mode this key DISABLES ITSELF with a
    # recorded note rather than refusing the run -- but only when it is at
    # its default.  A deck that names it explicitly on a non-PPM mode still
    # REFUSES, because an explicitly requested knob that silently does
    # nothing is how a green A/B comes to measure nothing.
    # THE DEFAULT IS TRUE -- see USE_BAND_EXTRAPOLATION_DEFAULT.  The entry
    # here is None ("no deck named it") because the resolver has to tell
    # "the deck said false" from "the deck said nothing": only the former
    # turns the feature off, and only the former makes a non-PPM mode refuse
    # instead of auto-disabling.  Both keys are tri-state for that reason.
    "use_band_extrapolation": None,
    # DEPRECATED ALIAS (transitional, 2026-08-16) for use_band_extrapolation.
    # Kept so committed decks and fixtures do not break.  Naming BOTH keys
    # with DIFFERENT values REFUSES BY NAME rather than picking a winner --
    # same migration shape as nband -> number_bands.  Its default is None
    # ("not named"), not False: the resolver distinguishes "the deck said
    # false" from "the deck said nothing", and only the former can turn the
    # feature off.
    "sigma_band_extrapolation": None,
    # WHICH band-convergence estimator consumes the three bracket sums.
    # Both read the SAME three points; they differ only in what they do with
    # them, so this key costs nothing and changes no compute.
    #
    #   spectral_shell   DEFAULT since 2026-08-17.  Solves one decay exponent
    #                    PER EXTERNAL STATE from the ratio of the two observed
    #                    shell increments against spectral moments of the DFT
    #                    eigenvalues, then integrates the remaining tail to
    #                    the finite plane-wave basis N_PW = min(ngk)*nspinor.
    #                    Held out against a MEASURED S(508) on the Si 50 Ry
    #                    508-band arm its median error is 4.7 / 14.7 / 12.5 /
    #                    0.7 / 0.0 meV at N_max = 152 / 204 / 260 / 296 / 396,
    #                    against 45.8 / 29.7 / 17.5 / 12.6 / 3.5 for the
    #                    incumbent.  REFUSES BY NAME on a state whose two
    #                    shells disagree in sign or whose exponent has no
    #                    bracketed root -- it never clips and never
    #                    substitutes.
    #   band_index_only  The incumbent two-parameter S_inf + A/N least
    #                    squares, under its honest name: the band INDEX is the
    #                    only thing it looks at.  Kept, selectable, and
    #                    bit-for-bit unchanged -- it is the same code path.
    #
    # See gw.band_extrapolation's module docstring for both derivations, the
    # held-out table and the owner rulings (beta is per-state and is never
    # pooled; the ladder comes from the DFT eigenvalues only).
    "band_extrapolation_estimator": BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT,
    # WHICH three compile-time band brackets feed the estimator.  Preserve
    # the incumbent total-band 80/90/100 geometry unless a deck explicitly
    # selects a conduction-coordinate experiment.  This is a COMPUTE
    # choice, unlike band_extrapolation_estimator: changing it recompiles and
    # re-runs Sigma.
    #
    # THE COORDINATE THE FRACTIONS ARE IN IS PART OF THIS KEY.
    # ``total_fractions`` reads 0.80 as round(0.80*N_max);
    # ``conduction_fractions`` reads it as n_occ + round(0.80*(N_max-n_occ)),
    # which is the coordinate the owner named on 2026-08-18.  The two are one
    # band apart on Si (n_occ=8) and tens of bands apart on a CrI3-scale
    # occupied manifold, so the reading cannot be left implicit -- but it is
    # also not a numerical ruling: reinterpreting 80/90 as conduction
    # fractions WORSENED the dense-Si control's N=180..396 indirect-gap
    # spread/max from 29.79/17.29 to 38.12/25.63 meV (JID 57267197).  Hence a
    # named spelling with the incumbent as the default, never an inference.
    "band_extrapolation_bracket_scheme": BRACKET_SCHEME_DEFAULT,
    "fermi_reference": "midgap",
    # DFT occupation smearing of the starting point.  REQUIRED as a pair
    # when ``mpa_material_class = metal``; refused under insulator.  These
    # are deck keys because WFN.h5 does NOT carry them: mf_header stores
    # el/occ/w but no smearing family and no degauss (verified 2026-08-15
    # on the canonical Na deck's WFN.h5 — zero attrs anywhere in mf_header).
    "occ_smearing_family": None,
    "occ_smearing_width_ry": None,
    # Far-tail clamp on the MP1 occupation table, applied AT EVALUATION
    # (inside the fixed-N root), so mu is solved for the clamped table and
    # the fixed-N invariant is asserted against what consumers receive.
    # |f| < tol -> 0, |1-f| < tol -> 1.  Nothing near the Fermi surface is
    # touched: gw.efermi.occupation_clamp_tol caps this 35x below MP1's
    # overshoot extremum 0.0354579, which is configured quadrature.  0.0
    # disables it bit-for-bit.  Insulating decks are unaffected by
    # construction (their table is already exactly 0/1).
    "occupation_clamp_tol": 1e-8,
    "sigma_at_dft_extrapolate": False,
    # Deprecated (2026-07-08): ``sigma_at_dft_energies = true`` is honored
    # as an alias for ``qp_solver = one_shot_dft`` — which is now the
    # default — via auto-resolution.  (The key was parsed-but-unread for
    # its whole life; its intended meaning, authoritative at-DFT QP
    # evaluation, is exactly QPSolver.ONE_SHOT_DFT.)
    "sigma_at_dft_energies": False,
    # Debug
    "sigma_freq_debug_output": False,
    "sigma_freq_debug_file": "sigma_freq_debug.dat",
    # QP wavefunction file dump.  Default True: end-of-run write of
    # ``WFN_qp.h5`` (BGW format, ψ rotated by the final U, energies
    # replaced by E_QP).  Fires for both one-shot and SC; set False to
    # skip the ~10s of MB write when only eqp.dat is wanted.
    "write_wfn_h5": True,
    # BSE interpolation setup (htransform-driven fine-k wfn recovery; see
    # ``bandstructure.bse_setup.compute_wfns_fi``).
    # 0 preserves the exact numerical-rank htransform span.  A value >= 1 is
    # an explicit cross-k model-order approximation: retain approximately
    # multiplier * (bands in the htransform window) shared alpha directions,
    # then restore the per-k row-isometry required by f(H).  This is NOT an
    # rtol alias and must be converged on the final observable.
    "htransform_rank_multiplier": 0.0,
    "get_centroids_fi": False,   # Gate; if True, compute fine-grid wfns at coarse centroids.
    "wfn_fi_min": 0,             # Sub-window of htransform band axis (0-based).
    "wfn_fi_max": 0,             # Exclusive upper end. wfn_fi_max==0 → use full window.
    "kgrid_fi": "",              # "nx ny nz" or "nx,ny,nz". Empty → no fine grid.
    # Fine-grid q CHUNK: how many q-points of the fine set have their
    # f(H(q)) built and decomposed at once.  0 (DEFAULT) = N_q_co, the
    # COARSE k-point count prod(kgrid_co) — not a bare constant.  fH_R is
    # (nk_co, rank, rank) face-sharded, so one chunk of N_q_co q-points is
    # byte-for-byte the same per-rank residency as fH_R itself: a deck that
    # could build fH_R at all can afford exactly one such chunk, and the
    # fine-grid pass then completes for ANY N_q_fi.  Raise it to trade
    # memory for fewer collectives, lower it on a memory-tight rank.  It is
    # a FLOOR: rounded up to a multiple of the device count so the q axis
    # stays shardable (bse_setup pads with sharding_fit.padded_extent).
    # Ignored by the distributed-eigh path, whose chunk is 1 by
    # construction.  See bandstructure.bse_setup.compute_wfns_fi.
    "wfn_fi_q_chunk": 0,

    # Deck hygiene.  False (DEFAULT): a key that is not in _DEFAULTS and
    # not covered by a legacy/deprecation branch is reported in one
    # aggregated rank-0 warning and ignored.  True: the same condition
    # raises ValueError naming every unknown key — for CI decks and fresh
    # runs where a typo must not silently drop a knob.
    "strict_keys": False,
}

# Deck keys REMOVED from _DEFAULTS but still handled by an explicit
# legacy/deprecation branch in ``read_lorrax_input`` (raise or dedicated
# DeprecationWarning).  The unknown-key check skips these so one deck key
# never draws two messages.  Keys that are deprecated but still in
# _DEFAULTS (self_consistent, sigma_at_dft_energies) need no entry here.
_LEGACY_DECK_KEYS = frozenset({
    "use_shipped_minimax_tables",   # refused with replacement named
    "chunk_size",                   # warn-and-ignore (planner-owned)
    "output_file",                  # warn-and-ignore (sigma_diag_file)
    "eqp_output_file",              # warn-and-ignore (auto eqp0/eqp1)
    # Removed transport selectors.  Keep only these parse-time tombstones so
    # an old deck cannot silently appear to select a deleted HDF5 path.
    "slab_io",                      # refused (one transport now)
    "use_ffi_io",                   # refused (one transport now)
    "gspace_mode",                  # refused (one ψ(G) lifecycle now)
    # 2026-08-14: host-tile accumulation is the only Σ(ω) accumulation
    # mode, so the key steered nothing.  The removed ``kij_stream`` VALUE
    # keeps its dedicated parse refusal; other values warn-and-ignore.
    "sigma_omega_accumulation",
})

# Keys whose string values should be lowercased and stripped
_NORMALIZE_STR = {
    "compute_mode",
    "qp_solver",
    "sc_accelerator",
    "sc_eigh",
    "sc_head_update",
    "wcoul0_source", "head_correction", "screening_method",
    "screening_diagrams",
    "minimax_energy_reference",
    "sigma_omega_layout", "fermi_reference",
    "band_extrapolation_estimator",
    "band_extrapolation_bracket_scheme",
    "occ_smearing_family",
    "w_dyson_solver",
    "ppm_invalid_mode",
    "ppm_model",
    "ppm_probe_chi_reuse",
    # ``restart_q_storage`` normalises here and is VALIDATED at parse time
    # against RESTART_Q_STORAGE, the same shape ``hartree_source`` uses: a
    # key whose wrong value would otherwise surface as a refusal deep in the
    # restart write, after the compute.
    "restart_q_storage",
    # ``qp_rotations_k_storage`` normalises and validates the same way, for
    # the same reason: its wrong value would otherwise surface as a refusal
    # in the post-SC artifact dump, after the whole self-consistency.
    "qp_rotations_k_storage",
    # distributed-linalg backend axes (consumed both via LorraxConfig and
    # directly from the params dict by htransform / exciton_bands).
    "eigh_backend",
    "distrib_la_batched_route",
}

# Tri-state booleans: _DEFAULTS value is None (= unset), an explicit
# input-file value parses as bool.  The parse loop needs the set because
# ``default is None`` otherwise means "nullable float".
_NULLABLE_BOOL = frozenset({
    # Both halves of the 2026-08-16 band-extrapolation migration.  Tri-state
    # so ``resolve_band_extrapolation`` can distinguish a deck that SAID
    # false from one that said nothing -- the difference between "turn it
    # off" and "let the default stand", and the difference between refusing
    # a non-PPM mode and auto-disabling on it.
    "use_band_extrapolation",
    "sigma_band_extrapolation",
})

#: Keys whose default is ``None`` ("unset") but whose explicit value is an
#: INTEGER.  Same role ``_NULLABLE_BOOL`` plays for tri-state booleans: the
#: parse loop's ``default is None`` branch otherwise means "nullable float",
#: and a band edge that parsed as ``52.0`` — or, worse, accepted ``52.5``
#: and silently truncated it — is a band edge nobody can reason about.  A
#: non-integral value raises out of ``configparser.getint`` by name.
_NULLABLE_INT = frozenset({
    "zeta_nband",
    # The band-count family's three "the deck did not say" slots.  They are
    # nullable for the same reason ``zeta_nband`` is — ``None`` has to be
    # distinguishable from any integer a deck could write — and integer for
    # the same reason too: a band edge that parsed as ``248.0``, or worse
    # accepted ``248.5`` and truncated it, is a band edge nobody can reason
    # about.  See ``resolve_band_counts``.
    "number_bands_chi",
    "number_bands_sigma",
    "nband",
})

#: Keys whose default is None but whose explicit value is a STRING — the
#: bare ``default is None`` parser branch is the nullable-float one.
_NULLABLE_STR = frozenset({"occ_smearing_family"})

#: Reserved slot in the params dict holding the set of deck keys the DECK
#: named.  Leading underscore because it is not a deck key and must never
#: match one: ``read_lorrax_input`` builds params from ``_DEFAULTS.items()``
#: and every real key comes from there, so a name that cannot appear in
#: ``_DEFAULTS`` cannot collide.  Read once, into ``GWConfig.raw_input_keys``.
_DECK_NAMED_KEYS = "_deck_named_keys"

#: Reserved slot holding the resolved :class:`BandCounts`.  Same convention
#: and the same reason as ``_DECK_NAMED_KEYS``: resolution happens ONCE, in
#: ``read_lorrax_input``, and the answer travels in the params dict rather
#: than being re-derived by every consumer.  Re-deriving it would not even be
#: possible after the fact — ``read_lorrax_input`` MIRRORS the resolved load
#: top back onto ``params["nband"]`` for the tools that read the dict
#: directly, which erases the distinction a second resolution would need.
_BAND_COUNTS = "_band_counts"


# ---------------------------------------------------------------------------
#  Band counts: one resolver, one precedence, four keys
# ---------------------------------------------------------------------------

class BandCountConflict(ValueError):
    """Two band-count keys were set and they disagree.

    Refusal, not coercion.  Every silent resolution of this case is wrong for
    somebody: picking the umbrella throws away the specific request the deck
    took the trouble to write, picking the specific makes the umbrella a lie
    for the OTHER consumer, and picking the max or the min invents a run
    nobody asked for.  So it is named, with both values quoted and the edit
    that fixes it spelled out.
    """


@dataclass(frozen=True)
class BandCounts:
    """The resolved χ and Σ band counts, and what the ISDF fit is sized by.

    Constructed exactly once per run, by :func:`resolve_band_counts`.  The
    only three numbers below this point are :attr:`chi`, :attr:`sigma` and
    :attr:`isdf`; nothing downstream re-reads a deck key to get a band count.

    Attributes
    ----------
    chi, sigma : int
        The χ0/W band count and the Σ band-sum count, both fully resolved
        (never ``None``): a deck that names only the umbrella gets them equal
        to it, which is the whole of the bit-identity claim.
    isdf : int
        ``max(chi, sigma)`` — the top of the band window the ψ is loaded over
        and therefore the window the ISDF ζ fit is built for.  The
        interpolation basis has to span the pair densities of whichever
        consumer reaches higher; sizing it by the smaller one would leave the
        larger consumer extrapolating in the ζ basis.
    named : frozenset[str]
        Which of the four keys the DECK itself wrote.  Kept so consumers can
        distinguish "asked for this edge by name" (→ strict degeneracy check,
        the ``zeta_nband`` precedent) from "inherited it" (→ the grandfather
        clause), without re-parsing the deck.
    """

    chi: int
    sigma: int
    named: frozenset = frozenset()

    @property
    def isdf(self) -> int:
        """Band-window top the ISDF ζ fit is built for: ``max(chi, sigma)``."""
        return max(self.chi, self.sigma)

    @property
    def isdf_source(self) -> str:
        """Which count won the ``max``: ``"chi"``, ``"sigma"`` or ``"tied"``."""
        if self.chi > self.sigma:
            return "chi"
        if self.sigma > self.chi:
            return "sigma"
        return "tied"

    @property
    def split(self) -> bool:
        """True when the two consumers were given DIFFERENT counts."""
        return self.chi != self.sigma

    def describe(self, zeta_fit_edge: int | None = None) -> str:
        """The one line a run logs so the ``max`` is never silent.

        Named in the brief that asked for the split: "log which count won the
        ``max`` and what the fit was built for.  A silent ``max`` is the kind
        of thing that gets mis-debugged for a day."

        ``zeta_fit_edge`` IS THE RESOLVED EDGE, NOT THE DECK KEY.  Pass
        ``gw.gw_init.resolve_zeta_fit_edge(band_slices, config.zeta_nband)``
        — the same value the fit, the window gates and the memory planner
        act on.  ``None`` means "nothing narrows it", i.e. the fit really is
        sized by :attr:`isdf`.

        A BANNER PRINTS RESOLVED VALUES ONLY.  With ``nband=700`` and
        ``zeta_nband=160`` this line used to say "ISDF zeta fit sized for 700
        bands ... the fit spans both" while the resolver and the memory
        planner were both acting on 160 (the CrI3 rank floor fell 180 -> 84
        on that key).  A startup-only run was then left with a materially
        false provenance line and no way to tell.  Perlmutter smoke step
        57236676.2,
        ``runs/CrI3/00_fm_331_991_700b_qsgw_gnppm_20260818/00_lorrax_smoke_p4/``.
        """
        if zeta_fit_edge is not None and int(zeta_fit_edge) != int(self.isdf):
            fit = (f"ISDF zeta fit sized for {int(zeta_fit_edge)} bands, "
                   f"NARROWED from {self.isdf} by deck key zeta_nband")
        else:
            fit = f"ISDF zeta fit sized for {self.isdf} bands"
        if not self.split:
            return (f"band counts: chi = sigma = {self.chi}; {fit} (the two "
                    f"counts are TIED, so the band sums are the same window)")
        return (f"band counts: chi = {self.chi}, sigma = {self.sigma}; {fit}"
                f" — the max is SET BY number_bands_{self.isdf_source} (the "
                f"larger); the smaller count ({min(self.chi, self.sigma)}) "
                f"does NOT size the fit")


#: The umbrella key and its transitional alias, canonical spelling first.
_UMBRELLA_KEYS = ("number_bands", "nband")

#: Consumer key -> the attribute it resolves into.
_SPECIFIC_KEYS = {"number_bands_chi": "chi", "number_bands_sigma": "sigma"}


def resolve_band_counts(params: dict, deck_named=None) -> BandCounts:
    """Resolve the four band-count keys into one :class:`BandCounts`.

    **THE ONLY PLACE THIS PRECEDENCE EXISTS.**  Four keys with two spellings
    of the umbrella is exactly the shape that grows a second, disagreeing
    resolution in a consumer six weeks later, so there is one function, it is
    pure, and it is directly testable without a deck, a WFN or jax.

    PRECEDENCE, in order:

    1. ``nband`` is a TRANSITIONAL ALIAS of ``number_bands``.  Either
       spelling sets the umbrella.  Both set to DIFFERENT values → refuse
       (:class:`BandCountConflict`).
    2. The umbrella supplies BOTH consumers.  A deck that names only it —
       every deck in the tree today — gets ``chi == sigma == umbrella``, and
       that is the bit-identity claim.
    3. ``number_bands_chi`` / ``number_bands_sigma`` override their own
       consumer and nothing else.
    4. Naming the umbrella AND a specific key with DIFFERENT values → refuse.
       "The umbrella overrides both" and "a specific key overrides its
       consumer" are both true and they contradict each other exactly here;
       this codebase has been bitten repeatedly by silent coercion, so the
       contradiction is reported rather than broken by fiat.  Naming them
       with the SAME value is redundant, not wrong, and is accepted.

    Parameters
    ----------
    params : dict
        A params dict from :func:`read_lorrax_input`, or any dict with the
        same keys.  Missing keys fall back to ``_DEFAULTS``.
    deck_named : iterable of str, optional
        The keys the deck itself wrote.  Defaults to
        ``params[_DECK_NAMED_KEYS]`` and then to "every key whose value is
        not None", so a hand-made dict behaves sensibly.  This is what
        separates "set to 100" from "defaulted to 100": without it a deck
        that pinned ``number_bands = 100`` beside ``number_bands_chi = 248``
        would be indistinguishable from one that pinned neither, and rule 4
        could not fire.
    """
    if deck_named is None:
        deck_named = params.get(_DECK_NAMED_KEYS)
    if deck_named is None:
        # No provenance supplied (a hand-made params dict).  "Present with a
        # non-None value" is the best available proxy — and it is read WITHOUT
        # the ``_DEFAULTS`` fallback, so a dict that simply omits a key is not
        # credited with naming it.
        deck_named = {k for k in (_UMBRELLA_KEYS + tuple(_SPECIFIC_KEYS))
                      if params.get(k) is not None}
    named = frozenset(str(k).lower() for k in deck_named)

    def _val(key):
        v = params.get(key, _DEFAULTS.get(key))
        return None if v is None or v == "" else int(v)

    # --- 1. umbrella, from either spelling -----------------------------
    # A key's VALUE is only consulted when the deck NAMED it.  Reading
    # ``number_bands`` unconditionally would let its default (100) outrank an
    # explicit ``nband = 248``, which is the alias silently not working —
    # exactly the failure the alias exists to prevent.
    canonical, alias = _UMBRELLA_KEYS
    u_canonical = _val(canonical) if canonical in named else None
    u_alias = _val(alias) if alias in named else None
    if (u_canonical is not None and u_alias is not None
            and u_canonical != u_alias):
        raise BandCountConflict(
            f"the deck sets BOTH `{canonical} = {u_canonical}` and the "
            f"transitional alias `{alias} = {u_alias}`, and they disagree.  "
            f"`{alias}` is an accepted spelling of `{canonical}`, not a "
            f"second axis, so there is no rule that makes one of these win "
            f"without discarding the other.  Delete one of the two lines "
            f"(keep `{canonical}` — `{alias}` is transitional).")
    umbrella = u_canonical if u_canonical is not None else u_alias
    umbrella_named = umbrella is not None
    if umbrella is None:
        umbrella = int(_DEFAULTS[canonical])

    # --- 2/3/4. the two consumers --------------------------------------
    resolved = {}
    for key, attr in _SPECIFIC_KEYS.items():
        v = _val(key)
        if key not in named or v is None:
            resolved[attr] = umbrella          # rule 2
            continue
        if umbrella_named and v != umbrella:   # rule 4
            which = canonical if canonical in named else alias
            other = "sigma" if attr == "chi" else "chi"
            raise BandCountConflict(
                f"the deck sets the umbrella `{which} = {umbrella}` AND "
                f"`{key} = {v}`, and they disagree.  `{which}` sets BOTH "
                f"band counts, so this deck says the {attr} count is "
                f"{umbrella} and also that it is {v}.  Pick one of:\n"
                f"    - drop `{which}` and set `number_bands_chi` / "
                f"`number_bands_sigma` explicitly (both of them — whichever "
                f"you leave out falls back to the umbrella default "
                f"{int(_DEFAULTS[canonical])}, which is almost certainly not "
                f"what you meant);\n"
                f"    - drop `{key}` and run both consumers at {umbrella};\n"
                f"    - keep `{which} = {v}` if {v} is what you meant for "
                f"the {other} count too.\n"
                f"Nothing is coerced here: silently preferring either value "
                f"would run a calculation this deck did not describe.")
        resolved[attr] = v                     # rule 3

    for attr, v in resolved.items():
        if v < 1:
            raise ValueError(
                f"number_bands_{attr} = {v} is not a band count; it must be "
                f">= 1.")
    return BandCounts(chi=int(resolved["chi"]), sigma=int(resolved["sigma"]),
                      named=named & (set(_UMBRELLA_KEYS) | set(_SPECIFIC_KEYS)))


#: What ``use_band_extrapolation`` means when no deck names either spelling.
#: TRUE since 2026-08-16 (owner ruling): the band-convergence extrapolation of
#: Σ_c runs by default and its extrapolated result is the E_nk that
#: self-consistent iterations see.  Lives here as a named constant rather than
#: inline in ``_DEFAULTS`` because ``_DEFAULTS`` has to carry ``None`` for the
#: tri-state resolution below, and a reader looking up "what is the default"
#: must not find ``None`` and conclude "off".
USE_BAND_EXTRAPOLATION_DEFAULT = True

#: The current key and its deprecated alias, in message order.
_BAND_EXTRAP_KEYS = ("use_band_extrapolation", "sigma_band_extrapolation")


def resolve_band_extrapolation(use_val, alias_val, *, print_fn=None) -> tuple:
    """Resolve the two spellings into ``(enabled, explicit)``.

    ``use_band_extrapolation`` is the key; ``sigma_band_extrapolation`` is a
    TRANSITIONAL alias kept so committed decks and fixtures do not break.
    Both arrive tri-state: ``None`` means the deck did not name that spelling.

    Returns
    -------
    enabled : bool
        Whether the feature is on.
    explicit : bool
        Whether a deck NAMED either spelling.  This is not decoration -- it
        selects between two different behaviours on a non-PPM ``compute_mode``
        (``gw.sigma_dispatch``): a defaulted-on key AUTO-DISABLES with a
        recorded note so that staged / static runs stay usable, while an
        explicitly-named one REFUSES.  Silently ignoring a knob the operator
        wrote down is how a green A/B comes to measure nothing.

    Raises
    ------
    ValueError
        When both spellings are named and they DISAGREE.  Refusing by name
        rather than picking a winner: whichever precedence we chose, half the
        decks that hit it would silently get the other one, and the operator
        would have no signal.  Same migration shape as ``nband`` ->
        ``number_bands``.
    """
    named = {k: v for k, v in zip(_BAND_EXTRAP_KEYS, (use_val, alias_val))
             if v is not None}
    if len(named) == 2 and bool(use_val) != bool(alias_val):
        raise ValueError(
            f"Deck names BOTH band-extrapolation keys and they DISAGREE: "
            f"use_band_extrapolation = {bool(use_val)!r} against the "
            f"deprecated alias sigma_band_extrapolation = "
            f"{bool(alias_val)!r}.\n"
            f"  These are the same switch under two names, so there is no "
            f"consistent run to produce and no winner will be picked for "
            f"you.  Remove ONE of them -- keep `use_band_extrapolation`, "
            f"which is the current spelling; `sigma_band_extrapolation` is "
            f"deprecated (2026-08-16) and is honored only for decks written "
            f"before the rename.")
    if not named:
        return bool(USE_BAND_EXTRAPOLATION_DEFAULT), False
    if alias_val is not None and print_fn is not None:
        print_fn(
            f"  *** DEPRECATED KEY: `sigma_band_extrapolation` is "
            f"transitional and will be removed; it is honored here as "
            f"`use_band_extrapolation = {bool(alias_val)}`.  Rename it in "
            f"this deck.")
    # Both named and AGREEING is fine, and lands here on either value.
    return bool(next(iter(named.values()))), True


def sigma_stage_modes(config, fallback=None) -> tuple:
    """Every :class:`ComputeMode` this RUN will dispatch a Σ under, in order.

    **THIS FUNCTION EXISTS BECAUSE A PREVIOUS ANALYSIS WAS WRONG, and the
    correction is worth stating rather than silently applying.**  The
    2026-08-16 SC-wiring branch concluded from a ``git log --all`` /
    ``git grep --all`` search that ``sc_stage_N_type`` "does not exist on any
    branch", and mapped per-stage behaviour onto ``compute_mode`` as the only
    available proxy.  The search was run in a single-branch checkout, where
    ``--all`` covers only FETCHED refs, so the null was a statement about that
    checkout's remotes.  The keys are real: ``origin/feat/staged-sc-2026-08-15``
    (98289d77) carries ``SC_STAGE_TYPES`` (``none | cohsex | gnppm | mpa``),
    ``SCStage(mode, cutoff_ev, max_iter)``, ``default_sc_ladder`` and
    ``resolve_sc_stages`` in this file, plus ``SCConfig.stages`` and
    ``run_staged_self_consistency`` in ``gw.sc_iteration``.

    WHAT THE REAL INTERFACE CHANGES, AND WHAT IT DOES NOT.

    * It does NOT invalidate the ``compute_mode`` seam.  Read against the real
      branch, ``run_staged_self_consistency`` rebuilds each stage's inputs with
      ``dataclasses.replace(config, compute_mode_raw=stage.mode.value)`` and
      passes ``stage.mode`` into ``compute_sigma_xc``, so during a stage the
      dispatched ``mode`` **is** that stage's mode.  A per-stage guard written
      against ``compute_mode`` therefore fires per stage already.  That part of
      the SC branch was accidentally right.
    * It DOES invalidate the REFUSAL.  A refusal is a statement about the whole
      RUN, and under a ladder the stage in front of you is not the run.  With
      the guard written per-stage, an explicitly-named key would kill:
      ``sc_stage_1_type = cohsex, sc_stage_2_type = gnppm`` at stage 1 (before
      reaching the very stage that consumes the key), and the SHIPPED DEFAULT
      LADDER for ``compute_mode = mpa`` — ``(GN_PPM @5 meV, MPA @2 meV)`` — at
      stage 2, after paying for a full GN-PPM stage.  Both are runs that must
      work.  Hence this function, and hence the refusal below is asked about
      the LADDER rather than about one stage.

    Parameters
    ----------
    config
        A :class:`LorraxConfig`, or anything shaped like one.  Read entirely
        through ``getattr`` so it is correct **before** the staged-SC branch
        merges (no ``config.sc.stages`` → the deck's single ``compute_mode``)
        and **after** it merges (the resolved ladder), with no edit here.
    fallback
        Mode to report when the config exposes neither a ladder nor a
        ``compute_mode`` — a hand-made namespace in a unit test, or a config
        whose ``compute_mode`` property refuses.  Callers pass the mode they
        are currently dispatching, which is the only honest answer available.
    """
    stages = getattr(getattr(config, "sc", None), "stages", None) or ()
    modes = []
    for stage in stages:
        raw = getattr(stage, "mode", stage)
        try:
            modes.append(coerce_compute_mode(raw))
        except (ValueError, TypeError):
            continue
    if not modes:
        try:
            modes = [coerce_compute_mode(config.compute_mode)]
        except (AttributeError, ValueError, TypeError):
            modes = []
    if not modes and fallback is not None:
        try:
            modes = [coerce_compute_mode(fallback)]
        except (ValueError, TypeError):
            modes = []
    # Order-preserving dedupe: the ladder is short and a reader of the refusal
    # wants "which schemes does this run use", not a repetition count.
    seen, out = set(), []
    for m in modes:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return tuple(out)


def band_extrapolation_is_consumable(modes) -> bool:
    """Does ANY stage of this run reach the kernel that reads the key?

    ``ppm_model is not None`` is the exact predicate: the extrapolation is
    wired into the two-point GN/HL plasmon-pole Σ_c kernel and nothing else.
    Deliberately NOT ``is_dynamic`` — that is True for MPA, which is dynamic
    and still does not consume this key.
    """
    return any(getattr(m, "ppm_model", None) is not None for m in modes)


# ---------------------------------------------------------------------------
#  Input file parser
# ---------------------------------------------------------------------------

def _deck_key_line(lines, start, end, key) -> str:
    """Locate ``key`` in the ``[cohsex]`` section; return ``"line N"``.

    Returns ``"line ?"`` when the key cannot be found on a line of its own
    (it can still have been parsed — configparser accepts continuations).
    """
    lineno = next(
        (i + 1 for i in range(start, end)
         if re.match(rf"\s*{re.escape(key)}\s*[=:]", lines[i], re.IGNORECASE)),
        None)
    return f"line {lineno}" if lineno is not None else "line ?"


def _print_deck_report(msg: str) -> None:
    """Print one deck-hygiene report on rank 0.

    ``process_rank`` is jax-free-safe (lazy jax import inside, falls back
    to 0 when jax is absent or uninitialised) — a downhill L1→L3 import,
    function-scoped so this parser stays importable without the common
    package fully initialised.
    """
    try:
        from common.collectives import process_rank
        rank = process_rank()
    except Exception:                                  # noqa: BLE001
        rank = 0
    if rank == 0:
        print(msg)


def read_lorrax_input(filename: str) -> dict:
    """Parse a LORRAX input file ([cohsex] section) into a params dict.

    Handles the QE-style K_POINTS block and strips it before INI parsing.
    All keys use ``_DEFAULTS`` for fallback values — no duplicate definitions.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Locate [cohsex] section
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith('[cohsex]'):
            start = i
            break
    if start is None:
        for i, line in enumerate(lines):
            if re.match(r"\s*\[.*\]", line):
                start = i
                break
    end = len(lines)

    # Locate optional K_POINTS block
    kp_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("k_points"):
            kp_idx = i
            break
    kp_end = None
    if kp_idx is not None and kp_idx + 1 < len(lines):
        try:
            seg_count = int(lines[kp_idx + 1].strip().split()[0])
        except Exception:
            seg_count = 0
        kp_end = min(len(lines), kp_idx + 2 + max(seg_count, 0))

    if start is not None:
        for j in range(start + 1, len(lines)):
            if re.match(r"\s*\[.*\]", lines[j]):
                end = j
                break
        # Strip K_POINTS from INI text
        if kp_idx is not None and start <= kp_idx < end:
            section_lines = lines[start:kp_idx] + lines[(kp_end or kp_idx + 1):end]
        else:
            section_lines = lines[start:end]

        # inline_comment_prefixes so 'key = off  # note' parses to 'off', not
        # 'off  # note' (the latter silently voided flags — a real footgun).
        parser = configparser.ConfigParser(inline_comment_prefixes=('#',))
        parser.read_string(''.join(section_lines))
        section = parser["cohsex"] if "cohsex" in parser else parser[parser.sections()[0]]

        # Legacy key check
        if section.get("use_shipped_minimax_tables", fallback=None) is not None:
            raise ValueError(
                "Input key 'use_shipped_minimax_tables' is no longer supported. "
                "Use 'regenerate_minimax_tables = true/false' instead.")

        # RETIRED-KEY REPORT.  A key with an explicit legacy branch is
        # exempt from the unknown-key check below so one deck key never
        # draws two messages — but that exemption left
        # ``warnings.warn(..., DeprecationWarning)`` as the ONLY report,
        # and Python's default filter hides DeprecationWarning outside
        # ``__main__``.  A retired key was therefore parsed, matched,
        # ignored, and announced to nobody, which is exactly the failure
        # the unknown-key check exists to prevent.  Collect every hit and
        # print it through the same rank-0 reporter, in wording that keeps
        # "retired" (the key was real once, and here is what replaced it)
        # distinct from "unrecognized" (nothing ever read this).  The
        # DeprecationWarnings stay — they are what a library consumer
        # filters on.  This report never REFUSES: ``strict_keys`` governs
        # unknown keys, and whether a retired key should be fatal is the
        # deck owner's call, not the parser's.
        retired = []                         # (key, what the run does with it)

        # ``chunk_size`` (legacy band-chunk knob) was a no-op: its only
        # consumer wrote ``meta.chunk_size``, which nothing ever read —
        # chunk sizing is owned by the gflat planner.  Dropped 2026-07-09.
        if section.get("chunk_size", fallback=None) is not None:
            import warnings
            warnings.warn(
                "Input key 'chunk_size' is no longer supported and will be "
                "ignored (it was a no-op; chunk sizing is planner-owned — "
                "see 'gflat_chunk_size' / 'band_chunk_size').",
                DeprecationWarning, stacklevel=2,
            )
            retired.append((
                "chunk_size",
                "IGNORED — it was a no-op; chunk sizing is planner-owned "
                "(see 'gflat_chunk_size' / 'band_chunk_size')"))
        for legacy_key in ("output_file", "eqp_output_file"):
            if section.get(legacy_key, fallback=None) is not None:
                import warnings
                warnings.warn(
                    f"Input key '{legacy_key}' is no longer supported and "
                    f"will be ignored.  ``output_file`` (LORRAX-native eqp0) "
                    f"is now ``sigma_diag_file`` (defaults to "
                    f"``sigma_diag.dat``); BGW-format ``eqp0.dat`` and "
                    f"``eqp1.dat`` (with Z-linearization) are written "
                    f"automatically.  Remove '{legacy_key}' from your "
                    f"input file.",
                    DeprecationWarning, stacklevel=2,
                )
                retired.append((
                    legacy_key,
                    "IGNORED — the LORRAX-native eqp0 filename is now "
                    "'sigma_diag_file'; eqp0.dat / eqp1.dat are written "
                    "automatically"))
        # There is one sharded-slab transport and the deck does not select
        # it.  Refuse the deleted selectors by name: accepting and ignoring
        # them made stale decks look as though their requested HDF5 route was
        # still active.  The tombstones stay in ``_LEGACY_DECK_KEYS`` only so
        # strict unknown-key handling does not mask this specific message.
        for legacy_key in ("slab_io", "use_ffi_io"):
            if section.get(legacy_key, fallback=None) is not None:
                raise ValueError(
                    f"Input key '{legacy_key}' is no longer supported and "
                    f"must be removed: there is one sharded-slab transport "
                    f"and the deck does not select an HDF5 implementation."
                )
        if section.get("gspace_mode", fallback=None) is not None:
            raise ValueError(
                "Input key 'gspace_mode' is no longer supported and must "
                "be removed: the fit builds one all-rank-sharded ψ(r) "
                "cache, then releases the host ψ(G) tiles before the "
                "r-chunk loop.  The former file_reread mode no longer "
                "described a distinct execution path."
            )
        # ``sigma_omega_accumulation`` was REMOVED (2026-08-14): host-tile
        # accumulation is the only mode, so the key steered nothing.  The
        # long-removed ``kij_stream`` VALUE keeps its dedicated refusal.
        _acc = section.get("sigma_omega_accumulation", fallback=None)
        if _acc is not None:
            if _acc.strip().strip('"\'').lower() == "kij_stream":
                raise ValueError(
                    "sigma_omega_accumulation = kij_stream was REMOVED; "
                    "host-tile accumulation is the only mode "
                    "(sigma_omega_layout selects the end-of-stage layout)")
            import warnings
            warnings.warn(
                "Input key 'sigma_omega_accumulation' is no longer "
                "supported and will be ignored (host-tile accumulation is "
                "the only mode).  Remove it from your input file.",
                DeprecationWarning, stacklevel=2,
            )
            retired.append((
                "sigma_omega_accumulation",
                "IGNORED — host-tile accumulation is the only mode"))
        # Deprecated qp_solver aliases (still honored via auto-resolution;
        # see ``LorraxConfig.qp_solver``).
        for legacy_key, replacement in (
            ("self_consistent", "qp_solver = self_consistent"),
            ("sigma_at_dft_energies", "qp_solver = one_shot_dft (the default)"),
        ):
            if section.get(legacy_key, fallback=None) is not None:
                import warnings
                warnings.warn(
                    f"Input key '{legacy_key}' is deprecated; it is honored "
                    f"via ``qp_solver = auto`` resolution.  Set "
                    f"'{replacement}' instead.",
                    DeprecationWarning, stacklevel=2,
                )
                retired.append((
                    legacy_key,
                    f"deprecated but still HONORED via 'qp_solver = auto' "
                    f"resolution; set '{replacement}' instead"))

        if retired:
            _print_deck_report(
                f"read_lorrax_input: {len(retired)} retired deck key(s) in "
                f"{filename}:\n"
                + "\n".join(
                    f"    {key} "
                    f"({_deck_key_line(lines, start, end, key)}): {note}"
                    for key, note in retired))


        # REMOVED keys (owner-approved deletions, 2026-07-31; these behave
        # like any other unknown deck key — reported by the unknown-key
        # check below, never steering anything): ``isdf_memory_mode``
        # (two-plan W cleanup — the W Dyson solve is selected by
        # w_dyson_solver=local|distributed) and the legacy aliases
        # ``cusolvermp_charge``/``cusolvermp_lu`` (use distributed_cholesky
        # / distributed_lu).

        # --- Unknown-key check -----------------------------------------
        # Every key in the deck that is neither in ``_DEFAULTS`` nor
        # handled by one of the explicit legacy branches above is reported
        # in ONE aggregated rank-0 warning (key, line number, "ignored").
        # Silently dropping unknown keys turned every typo and every stale
        # doc into silent wrong physics.  Warn, don't refuse — archived
        # decks carry dead keys — unless the deck opts in via
        # ``strict_keys = true``, which upgrades the warning to a
        # ValueError naming all unknown keys at once.
        # Retired keys are exempt (they got their own report above).
        # configparser lower-cases option names (``optionxform = str.lower``),
        # so iterating ``section`` yields ``do_g0`` for a deck that writes
        # the documented ``do_G0`` -- the ONE non-lower-case key among the
        # 99 in _DEFAULTS.  Comparing the two raw made that key BOTH
        # honoured and unrecognised at the same time: ``section.get`` folds
        # the LOOKUP too, so ``do_G0 = false`` really did steer the run,
        # while this check reported it as an unknown key -- and, under
        # ``strict_keys``, REFUSED a valid deck outright.  Fold both sides
        # so recognition matches the lookup that already happens.
        _known = ({k.lower() for k in _DEFAULTS}
                  | {k.lower() for k in _LEGACY_DECK_KEYS})
        unknown = [k for k in section if k.lower() not in _known]
        if unknown:
            located = [f"{key} ({_deck_key_line(lines, start, end, key)})"
                       for key in unknown]
            if section.getboolean(
                    "strict_keys",
                    fallback=bool(_DEFAULTS["strict_keys"])):
                raise ValueError(
                    f"strict_keys = true: {len(unknown)} unknown deck "
                    f"key(s) in {filename}:\n"
                    + "\n".join(f"    {loc}: not a recognized deck key"
                                for loc in located))
            _print_deck_report(
                f"read_lorrax_input: {len(unknown)} unrecognized deck "
                f"key(s) in {filename}:\n"
                + "\n".join(
                    f"    {loc}: ignored — not a recognized deck key"
                    for loc in located))

        # Build params from _DEFAULTS, overriding with parsed values
        params = {}
        # WHICH KEYS THE DECK ITSELF NAMED.  ``params`` cannot answer this
        # afterwards — a deck pinning a key to its default and a deck that
        # never mentions it produce the identical entry — and the difference
        # matters to anything that must speak only to decks that opted in.
        # Its first consumer is the ``restart_q_storage`` deprecation notice
        # (owner ruling 2026-08-08: the key is scheduled for deletion), which
        # must fire for a deck that pins it and stay silent for the other
        # ~forty, or it is noise nobody reads.  Recorded here, where the
        # answer is free, rather than re-parsed by each consumer.
        named = set()
        for key, default in _DEFAULTS.items():
            raw = section.get(key, fallback=None)
            if raw is not None:
                named.add(key)
            if raw is None:
                params[key] = default
            elif key in _NULLABLE_BOOL:
                # Tri-state boolean (default None = unset); an explicit
                # value parses as bool.
                params[key] = section.getboolean(key)
            elif key in _NULLABLE_INT:
                params[key] = section.getint(key)
            elif key in _NULLABLE_STR:
                params[key] = str(raw)
            elif isinstance(default, bool):
                params[key] = section.getboolean(key)
            elif isinstance(default, int):
                params[key] = section.getint(key)
            elif isinstance(default, float):
                params[key] = section.getfloat(key)
            elif default is None:
                # Nullable float (vhead, whead_0freq, etc.)
                params[key] = section.getfloat(key, fallback=None)
            else:
                params[key] = str(raw)
            if key in _NORMALIZE_STR and isinstance(params[key], str):
                params[key] = params[key].strip().lower()
        params[_DECK_NAMED_KEYS] = frozenset(named)
    else:
        params = dict(_DEFAULTS)
        params[_DECK_NAMED_KEYS] = frozenset()

    # --- Band counts: resolve ONCE, here ------------------------------
    # ``number_bands`` / ``number_bands_chi`` / ``number_bands_sigma`` /
    # ``nband`` collapse into two numbers plus their max, and this is the
    # only call to the resolver on the deck path.  Resolving here rather
    # than in ``LorraxConfig`` is what lets the params dict stay honest for
    # the tools that read it directly (``bandstructure.htransform``,
    # ``psp.get_DFT_mtxels``, ``gw.kin_ion_io``, ``file_io.epsreader``):
    # they ask for ``params["nband"]`` and must get the LOADED band extent,
    # which after the split is ``max(chi, sigma)`` — the same number they
    # always got on an unsplit deck.
    #
    # The mirror is why this is not idempotent and why the answer is
    # cached in ``params[_BAND_COUNTS]`` instead of being re-derived: after
    # the write-back, ``nband`` no longer says what the DECK said, so a
    # second ``resolve_band_counts`` on this dict would see an umbrella that
    # the deck never wrote.
    _counts = resolve_band_counts(params, deck_named=params[_DECK_NAMED_KEYS])
    params[_BAND_COUNTS] = _counts
    params["number_bands_chi"] = _counts.chi
    params["number_bands_sigma"] = _counts.sigma
    params["number_bands"] = _counts.isdf
    params["nband"] = _counts.isdf

    # Parse optional QE K_POINTS block
    if kp_idx is not None:
        j = kp_idx + 1
        try:
            nseg = int(lines[j].strip().split()[0])
        except Exception:
            nseg = 0
        segments = []
        for k in range(nseg):
            row_idx = j + 1 + k
            if row_idx >= len(lines):
                break
            row_full = lines[row_idx].rstrip('\n')
            label = None
            for marker in ('#', '!', ';'):
                if marker in row_full:
                    label = row_full.split(marker, 1)[1].strip() or None
                    row_full = row_full.split(marker, 1)[0]
                    break
            row = row_full.strip()
            if not row:
                continue
            parts = row.split()
            if len(parts) < 3:
                continue
            segments.append({
                "k": [float(parts[0]), float(parts[1]), float(parts[2])],
                "n": int(parts[3]) if len(parts) >= 4 else 1,
                "label": label,
            })
        if segments:
            params["kpoints_crystal_b"] = {"segments": segments}

    return params


# Backward-compatible alias
read_cohsex_input = read_lorrax_input


# ---------------------------------------------------------------------------
#  LorraxConfig
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Sub-dataclasses (each frozen, attribute-accessed via ``config.<group>.X``)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilePaths:
    """Output filenames + non-WFN inputs.  Resolved to absolute paths."""
    wfn_file: str
    centroids_file: str
    # Bispinor: optional Gordon-current-density centroid file for μ_L=1,2,3.
    # ``None`` falls back to the scalar charge-only path (CC tile only).
    centroids_file_current: str | None
    kin_ion_file: str
    parallel_transport_file: str
    sigma_diag_file: str
    eqp0_file: str
    eqp1_file: str
    eqp2_file: str
    report_file: str
    sigma_omega_h5_file: str


def _normalize_placement(value):
    """Canonicalise ``mc_average_placement`` at deck-parse time.

    Delegates to :func:`gw.head_channel.normalize_placement` so the deck
    parser and the consumer cannot drift on what the mode names are, and so
    a typo is a refusal at config time (with the valid list in the message)
    rather than a silent ``off`` two stages later.  Imported lazily: this
    module is imported by the CLI before jax is configured, and
    ``head_channel`` keeps its jax imports function-local for the same
    reason, so the cost is one numpy-only module.
    """
    from .head_channel import normalize_placement
    return normalize_placement(value)


BGW_METAL_Q0_TREATMENTS = ("exact", "bgw_q0shift")


def _normalize_bgw_metal_q0_treatment(value) -> str:
    mode = str(value or "exact").strip().lower()
    if mode not in BGW_METAL_Q0_TREATMENTS:
        raise ValueError(
            f"bgw_metal_q0_treatment = {value!r} is not recognised; expected "
            "'exact' or 'bgw_q0shift'.")
    return mode


def refuse_unsupported_bgw_metal_q0_treatment(config) -> None:
    """Refuse a BGW q0 bundle whose finite-q W channel is not consumed."""
    if not bool(config.head.uses_bgw_metal_q0shift):
        return
    mode = config.compute_mode
    if mode is ComputeMode.MPA:
        return
    raise ValueError(
        "GATE bgw_q0shift_requires_mpa: "
        "bgw_metal_q0_treatment = bgw_q0shift is refused with "
        f"compute_mode = {mode.value}.\n"
        f"  got:  bgw_metal_q0_treatment = bgw_q0shift, "
        f"compute_mode = {mode.value}\n"
        "  want: compute_mode = mpa\n"
        "  fix:  use compute_mode = mpa to consume the matching finite-q0 "
        "epsilon-inverse W head/wings, or set "
        "bgw_metal_q0_treatment = exact\n"
        "  why:  cohsex, gn_ppm, hl_ppm, and x_only do not consume the "
        "finite-q0 epsilon-inverse channel; allowing the key would pair "
        "BerkeleyGW's analytic-sphere v0 with a different W0\n"
        "  doc:  docs/input_reference.md '## Screening', "
        "bgw_metal_q0_treatment.")


#: WHICH COMBINATIONS OF ``low_mem_bands = true`` v1 REFUSES, and why.
#:
#: Same shape as ``_W_BSE_REFUSALS`` above: ``rule_id -> (predicate, got,
#: want, fix, doc)``, assembled into one five-part message so a rule cannot
#: be added without answering all five.  Predicates take the resolved
#: :class:`LorraxConfig` and check only their OWN axis — the caller,
#: :func:`refuse_unsupported_low_mem_bands`, has already established
#: ``low_mem_bands = true`` before the loop runs.
#:
#: SUPPORTED, deliberately absent from this table (guide
#: ``reports/gwjax_low_mem_bands_audit_2026-08-22/report.md`` §6):
#: scalar/spinor, one-shot insulator (``qp_solver`` = ``one_shot_dft`` or
#: ``fixed_point``), ``head_correction`` = ``off`` | ``no_local_fields``,
#: standard chi0, COHSEX / GN-PPM / HL-PPM / insulating MPA, restart
#: read/write.
#:
#: A FIFTH combination the guide names — an explicit dense ``Gij`` operand
#: — has no deck key (every shipped driver call site leaves it at its
#: ``None`` default; see ``cohsex_sigma._resolve_Gij``), so it cannot be a
#: row in a config-resolution table keyed on parsed values.  It is guarded
#: separately by :func:`refuse_explicit_gij_under_low_mem_bands`, called
#: from ``compute_sigma_xc`` at the one seam that ever sees both operands
#: together.
_LOW_MEM_BANDS_REFUSALS: tuple[tuple[str, object, object, str, str, str], ...] = (
    # LIFTED 2026-08-23 (feat/qsgw-face-rotations-2026-08-23) per this row's
    # own recorded lift condition: rotate_wavefunctions now dispatches on
    # wfns_dft.layout and routes layout='face' through
    # wavefunction_bundle._rotate_wavefunctions_face (two planned
    # distrib_la.gemm_plan N,N GEMMs -- U^T @ psi_nmu, psi_mun @ U -- against
    # a block-embedded U rather than a sliced ψ; see wavefunction_bundle
    # ._face_rotate_kernel/._face_embed_active_U).  sc_iteration.py:1753
    # needed NO change: it already calls rotate_wavefunctions(inputs.
    # wfns_dft, ...), and the dispatch reads wfns_dft.layout, not a
    # call-site flag.  Gated: real 4-rank CUDA algebra parity vs legacy
    # (tests/test_qsgw_rotate_face_parity.py, U from a REAL small eigh —
    # ns=1/ns=2, default AND offset active windows — 3/3 PASS, max relative
    # diff ~1e-16..2e-16); a real end-to-end MoS2 k6_c50 compute_mode=
    # gn_ppm head_correction=full qp_solver=self_consistent (3 iterations)
    # leg, face vs legacy — see this session's CLAIMS.md row for job id and
    # measured tolerances.  ``head_wings_sharded``'s own consumer,
    # ``build_iteration_head_response`` (qsgw_head.py), needed NO change
    # either: it treats ``wfns_qp`` opaquely (no direct psi_* field access)
    # and forwards it to the already layout-dispatching wing kernels; the
    # ONLY reason it read as "still legacy-only" before this session is
    # that its sole producer, rotate_wavefunctions, had no face arm.
    (
        # PORTED 2026-08-23 (feat/metal-response-face-2026-08-23,
        # docs/architecture/fractional_chi0_response_face.md): the exact
        # finite-occupation chi0 the census row named --
        # w_isdf._fractional_pair_scan's ordered-pair kernel (Gamma static
        # body + finite-q/finite-z direct kernel) AND the separate
        # fractional/contour kernel -- now both dispatch on wfns.layout
        # and route through a genuinely new 2-D band-pair mechanism
        # (masked-gather + psum on BOTH mesh axes, isdf.core._z_q_face's
        # idiom generalized) for the ordered-pair half, and a plain
        # build_G_tau(layout='face', ...) substitution for the
        # fractional/contour half (its two Green's functions are each
        # one-particle).  Gated: real 4-rank CUDA algebra parity
        # (tests/test_chi0_fractional_face_parity.py), a structural
        # no-single-axis-psi proof, and a production-shape Na-deck
        # harness (face-vs-legacy, NOT a full gw_jax driver run -- see
        # below for why).
        #
        # THE ROW STAYS REFUSING ANYWAY, narrowed to name the REAL
        # remaining blocker rather than lifted: _validate_metal_compute_
        # mode (this file, below) REQUIRES compute_mode = mpa whenever
        # mpa_material_class = metal, and compute_mode = mpa is
        # UNCONDITIONALLY refused under low_mem_bands = true by the
        # SEPARATE low_mem_bands_dynamic_ppm_unported row (gw.mpa.sigma's
        # own Sigma_c(omega) executor -- a different subsystem entirely,
        # frequency-domain self-energy, not this row's chi0/response
        # subject, and outside this session's charter). So no deck
        # combination can reach this session's ported kernels without
        # ALSO tripping that other row first if this one were removed:
        # lifting this row alone would not unblock a single live
        # low_mem_bands=true metal deck, only change WHICH rule id it
        # refuses under. Per the design doc's own closing section, this
        # row is kept and will be lifted for real once
        # low_mem_bands_dynamic_ppm_unported is ALSO lifted for MPA (a
        # separate, unscoped port) -- at that point this row's own
        # predicate can be dropped outright, since nothing else in the
        # census blocks a metal deck once both are ported.
        "low_mem_bands_metal_material_class_unported",
        lambda cfg: (str(getattr(cfg.mpa, "material_class", "insulator"))
                     .strip().lower() == "metal"),
        lambda cfg: f"mpa_material_class = {cfg.mpa.material_class}",
        "mpa_material_class = insulator (the default)",
        "set mpa_material_class = insulator (or drop the key), or set "
        "low_mem_bands = false for a metallic run",
        "the exact finite-occupation chi0 response is now ported to the "
        "face layout (2026-08-23, docs/architecture/"
        "fractional_chi0_response_face.md) and gated on real 4-rank CUDA, "
        "but mpa_material_class = metal unconditionally requires "
        "compute_mode = mpa (_validate_metal_compute_mode below), which "
        "the SEPARATE low_mem_bands_dynamic_ppm_unported row still "
        "refuses (gw.mpa.sigma's own Sigma_c(omega) executor is a "
        "different, unrelated, unfinished port) -- so this row stays "
        "refusing to give an accurate rule id until BOTH are lifted "
        "together, rather than letting a cleared deck fall through to a "
        "less specific refusal one row later",
    ),
    # LIFTED for FRESH-FIT decks (2026-08-23,
        # feat/transverse-zeta-face-2026-08-23) — the row's LAST gap
        # closed.  Both halves the previous session's comment named as
        # missing are now ported and gated:
        #
        # * Sigma^B/vertex insertion (sigma_x_bispinor.py's G-build side)
        #   — gw.wavefunction_bundle.with_lorentz_vertices, a
        #   representation-aware bundle operation folding γ̃ into whichever
        #   pair of fields plays the G-build's direct/conjugated role
        #   (psi_xn/psi_yr legacy, psi_mun/psi_nmu face) — landed
        #   2026-08-23 (feat/bispinor-face-2026-08-23), gated on real
        #   4-rank CUDA with a genuine ns=4 fixture (5/5 Lorentz pairs,
        #   ~1e-16 relative; tests/multi_device/
        #   bispinor_transverse_vertex_face_gate.py).
        # * The transverse ζ-FIT's face path — ``isdf.core.
        #   c_q_from_psi_sm(layout='face')``/``z_q_from_psi_sm(layout=
        #   'face')`` now accept non-identity ``gamma_L``/``gamma_R`` via
        #   psi-ENDPOINT application (mirroring
        #   ``with_lorentz_vertices``'s own field/axis table, folded in
        #   BEFORE the band GEMM / masked-gather rather than at
        #   ``gamma_double_contract``'s post-IFFT step — see
        #   ``docs/architecture/zeta_fit_face_psi_cct.md``'s "γ̃ VERTEX"
        #   sections for the derivation and its conjugation-convention
        #   correction, found by reading ``greens_function_kernel.
        #   _build_G_face`` directly: CCT's psi_mun is the CONJUGATED
        #   operand, the OPPOSITE role from the G-build's own psi_mun).
        #   Gated on real 4-rank CUDA, ALL 15 non-identity
        #   ``(mu_L, nu_L)`` Lorentz-index pairs at ns=4 (the
        #   discriminating cases — an identity vertex passes trivially
        #   and proves nothing): ``tests/test_isdf_cq_face_parity.py``
        #   18/18 PASS, max relative diff ~6e-16; ``tests/
        #   test_isdf_zq_face_parity.py`` 18/18 PASS on real CUDA
        #   (mostly bit-exact, max relative diff ~4e-16 where not — the
        #   masked-``psum`` mechanism is a select, immune to summation-
        #   order noise, same as its own identity-channel result).
        #   ``gw.isdf_fitting.fit_zeta_to_h5``'s ``vertex_mu_L != 0``
        #   refusal under ``low_mem_bands`` is dropped;
        #   ``isdf.core._make_fit_one_rchunk_kernel``'s matching refusal
        #   too.  ``gw.gw_init.fit_zeta`` builds the TRANSVERSE
        #   centroid set's OWN face carrier via the SAME
        #   ``PSI_MUN_SPEC``/``PSI_NMU_SPEC`` build path the charge
        #   channel already uses (not a fork), reused for both the ζ_T
        #   fit and the post-fit Σ^B bundle.
        #
        # End-to-end: MoS2 3×3 bispinor GN-PPM fixture
        # (tests/regression/bispinor_debug/bispinor_test.in,
        # head_correction=off, restart=false), real 4-rank CUDA,
        # low_mem_bands=true vs low_mem_bands=false — see
        # runs/MoS2/90_bispinor_lowmem_smoke_2026-08-23/ for the
        # artifacts and claims/ for the numbers.
        #
        # DELETED, not narrowed (same precedent as
        # ``low_mem_bands_self_consistent_unported``'s own 2026-08-23
        # lift): the census row's gap is closed for every combination
        # bispinor ITSELF supports.  ``restart = true`` + ``bispinor =
        # true`` was ALREADY refused before this session and independent
        # of ``low_mem_bands`` — ``gw.gw_init.prepare_isdf_and_
        # wavefunctions``'s restart-read path raises loudly whenever a
        # restart file has no ``psi_full_y_transverse`` dataset, which is
        # every file (this predates low_mem_bands entirely; see
        # tests/regression/bispinor_debug/README.md: "bispinor restart
        # is not yet supported").  A ``write_restart_tensors = true``
        # fresh run (the default) is harmless under low_mem_bands too:
        # the low_mem_bands restart WRITE branch simply omits the
        # transverse-centroid face carrier (no dataset for it yet), and
        # the SAME pre-existing read-side check refuses loudly if
        # anyone later tries to restart from that file with
        # ``bispinor = true`` — no silent data loss, in either layout.
        # This row is deleted outright rather than re-pointed at that
        # pre-existing, low_mem_bands-independent limitation.
    (
        # LIFTED for GN_PPM/HL_PPM 2026-08-22 (feat/dynamic-sigma-face-
        # port-2026-08-22); the row is KEPT, narrowed to MPA only.  History:
        # DISCOVERED on real 4-rank CUDA (tests/multi_device/
        # low_mem_bands_one_shot_insulating_envelope_gate.py, MoS2 k6_c50,
        # 2026-08-22): a low_mem_bands=true, compute_mode=gn_ppm deck (the
        # supported-table's own claimed envelope) ran ISDF fit + chi0/W
        # construction to completion under layout='face', then died in
        # ``ppm_tau_kernel.precompile_sigma`` at the FIRST legacy accessor
        # call (``wfns.xn(s.full)``) with the carrier's own named
        # ``_require_legacy`` ValueError -- not a clean parse-time refusal.
        # The dynamic two-point plasmon-pole Sigma_c(omega) pipeline
        # (ppm_tau_kernel.py's ``precompile_sigma``/``_get_sigma_tau_kernel``
        # /``_get_sigma_kij_kernel``, ``common.contract_bands``'s
        # channels="split_reim" face arm, and ppm_sigma.py's per-branch
        # sigma builders + invalid-pole static-limit term) now dispatches
        # on ``wfns.layout`` and routes through
        # ``greens_function_kernel.build_G_tau(layout='face', gemm=...)``/
        # ``contract_bands_block_reshard(layout='face', channels=...)`` —
        # the SAME canonical owners the static COHSEX channels already
        # used, extended rather than forked (report §5).  Gated: real
        # 4-rank CUDA algebra parity (legacy vs face, identity + real tau
        # weights, ns=1/ns=2, a non-mesh-divisible sigma window,
        # tests/test_ppm_tau_kernel_face_parity.py, 5/5 PASS), a real
        # end-to-end MoS2 k6_c50 leg at compute_mode=gn_ppm
        # head_correction=full matching the legacy gn_ppm reference to
        # ~1e-5 eV, and tests/test_zeta_mesh_invariance.py 7/7 unaffected
        # (claims/0435.md).  A LARGER k6_c600 (mu=5282) confirmation of
        # the same combination could NOT be completed this session: it
        # dies in the already-registered, pre-existing qsgw_head.py
        # head-response OOM (KNOWN_LORRAX_ISSUES.md's
        # src/gw/qsgw_head.py:250-256 row; third independent
        # reproduction, claims/0436.md) before ever reaching this
        # pipeline's own code — that defect is inherited from this
        # branch's base and is unrelated to this port (it also blocks
        # head_correction=full under low_mem_bands=true for COHSEX at
        # that scale).  Production-scale confirmation of THIS port
        # remains open follow-up work, not claimed here.  ``mpa/sigma.py``
        # (insulating MPA's own executor,
        # ``_integrate_sigma_batches``) was mechanically ported the SAME
        # session, sharing this now-gated tau-kernel/projector infra, but
        # was NOT itself run end to end this session (its own
        # sharded-output final layout has an additional named gap for a
        # split Σ window — see that file's own comment) — kept refused
        # below pending its own gate, rather than lifted on the strength
        # of a shared-code argument alone.
        "low_mem_bands_dynamic_ppm_unported",
        lambda cfg: cfg.compute_mode is ComputeMode.MPA,
        lambda cfg: f"compute_mode = {cfg.compute_mode.value}",
        "compute_mode = x_only, cohsex, gn_ppm, or hl_ppm",
        "set compute_mode = gn_ppm/hl_ppm/cohsex/x_only, or "
        "low_mem_bands = false for an MPA run",
        "gw.mpa.sigma's executor was mechanically ported to layout='face' "
        "the same session as ppm_sigma.py's but has no end-to-end gate of "
        "its own yet, and its sharded-output tail additionally refuses a "
        "split Σ window under this layout by name (see "
        "gw.mpa.sigma._integrate_sigma_batches) — kept refused pending "
        "that gate rather than lifted on ppm_tau_kernel's shared "
        "infrastructure alone",
    ),
)


def refuse_unsupported_low_mem_bands(config) -> None:
    """Refuse the ``low_mem_bands = true`` combinations v1 does not serve.

    BEFORE ALLOCATION.  Called from :meth:`LorraxConfig.from_input_file`
    once the resolved record exists (predicates read ``compute_mode`` /
    ``qp_solver``, which fold in the legacy flags — the same reason
    :func:`refuse_unsupported_screening_diagrams` is called there and not
    re-derived), so a doomed deck refuses in the first second of a run
    rather than after the chi0 build or the ISDF fit that would otherwise
    silently rebuild a one-axis replica to serve it.  Also called from
    ``gw.gw_init.prepare_isdf_and_wavefunctions`` at entry, mirroring
    :func:`refuse_unimplemented_compute_mode`'s two call sites: the parser
    call is what saves the operator's allocation on the production path,
    the driver-entry call is what makes a hand-built config (a test
    harness, a future direct caller) safe without having to remember the
    parser check.

    NO-OP FOR ``low_mem_bands = false`` (the default), evaluated first and
    returning before any predicate is touched: a default deck must not
    acquire a new parse-time resolution — and hence a new possible
    refusal — from this feature existing.
    """
    if not bool(config.memory.low_mem_bands):
        return
    for rule_id, predicate, got, want, fix, doc in _LOW_MEM_BANDS_REFUSALS:
        if not predicate(config):
            continue
        raise ValueError(
            f"GATE {rule_id}: low_mem_bands = true is refused with "
            f"{got(config)}.\n"
            f"  got:  low_mem_bands = true, {got(config)}\n"
            f"  want: {want}\n"
            f"  fix:  {fix}\n"
            f"  why:  {doc}.\n"
            f"  doc:  docs/input_reference.md '## ISDF / zeta', "
            f"low_mem_bands.")


def refuse_explicit_gij_under_low_mem_bands(config, Gij) -> None:
    """Refuse an explicit dense ``Gij`` operand under ``low_mem_bands = true``.

    THE ONE ROW ``_LOW_MEM_BANDS_REFUSALS`` CANNOT HOLD.  Every other
    combination in the envelope is a deck key readable at parse time; an
    explicit ``Gij`` is a keyword-only Python parameter of
    ``compute_sigma_xc`` / ``compute_cohsex_sigma`` that every shipped
    driver call site (``gw_jax.py``, ``sc_iteration.py``) leaves at its
    ``None`` default — ``cohsex_sigma._resolve_Gij``'s docstring names the
    caller this guards, the SC-COHSEX loop iterating on its own projector.
    No deck-resolution point ever sees the value, so this is called from
    ``compute_sigma_xc`` at entry instead, before any Gij-dependent
    allocation (``build_Gij`` / the dense band-matrix contract) — the one
    seam that ever sees both ``low_mem_bands`` and a live ``Gij`` operand
    together.

    NO-OP for ``Gij is None`` (every production call today) or
    ``low_mem_bands = false``, so this feature existing changes nothing for
    the vastly more common calls that never touch either axis.
    """
    if not bool(config.memory.low_mem_bands) or Gij is None:
        return
    raise ValueError(
        "GATE low_mem_bands_explicit_gij_unported: "
        "low_mem_bands = true is refused with an explicit Gij operand.\n"
        "  got:  low_mem_bands = true, Gij is not None (explicit dense "
        "band-space occupation projector)\n"
        "  want: Gij = None (the standard occupation_state path)\n"
        "  fix:  do not pass an explicit Gij under low_mem_bands = true — "
        "the occupation_state argument already builds one — or set "
        "low_mem_bands = false\n"
        "  why:  cohsex_sigma.build_Gij returns a fully replicated dense "
        "(nk, nb_sigma, nb_sigma) array; under face psi, G and the "
        "Hartree/exchange projection need a face-sharded band-matrix "
        "contract that has not been ported (obstacle #4, 'treat production "
        "Gij as diagonal data')\n"
        "  doc:  docs/input_reference.md '## ISDF / zeta', low_mem_bands.")


def _parse_bgw_metal_q0_vector(value) -> tuple[float, float, float]:
    """Parse one reduced-coordinate q0 vector without guessing its units."""
    raw = str(value or "").replace(",", " ").split()
    if len(raw) != 3:
        raise ValueError(
            "bgw_metal_q0_vector must contain exactly three reduced "
            f"reciprocal coordinates; got {value!r}.")
    try:
        q0 = tuple(float(component) for component in raw)
    except ValueError as exc:
        raise ValueError(
            "bgw_metal_q0_vector must contain three finite numbers; "
            f"got {value!r}.") from exc
    if not all(np.isfinite(component) for component in q0):
        raise ValueError(
            "bgw_metal_q0_vector must contain three finite numbers; "
            f"got {value!r}.")
    if max(abs(component) for component in q0) <= 1.0e-14:
        raise ValueError(
            "bgw_metal_q0_vector must be nonzero: BerkeleyGW's metallic "
            "epsilon q0 sample is a shifted grid point.")
    return q0


@dataclass(frozen=True)
class HeadConfig:
    """q→0 Coulomb-head sources, BGW vcoul override, bare-cutoff knobs.

    All Coulomb-at-small-q tweaks live here.  Σ head plumbing
    (``wcoul0_*``, ``vhead``/``whead_*``) is consumed by
    :class:`gw.head_correction.HeadResolver`; the BGW vcoul override is
    purely diagnostic (matches BGW's per-G mini-BZ averaging exactly for
    bit-reproducible comparisons).
    """
    correction: HeadCorrection    # full | no_local_fields | off
    wcoul0_source: str            # "s_tensor" | "epshead"
    wcoul0_eta: float
    vhead: float | None           # explicit override v_h[ω=0]
    whead_0freq: float | None     # explicit override W_h[ω=0]
    whead_imfreq: float | None    # explicit override W_h[iω_p]
    mc_average_vcoul_body: bool
    bgw_metal_q0_treatment: str   # "exact" | "bgw_q0shift"
    bgw_metal_q0_vector: tuple[float, float, float]
    mc_average_placement: str      # "off" (default) | "bgw" | "schur_avg"
    mc_average_placement_vcoul: str | None   # BGW vcoul dump for byte-sourced <v>
    head_minibz_average: bool      # per-Q mini-BZ head cell-average (default off)
    w_av_first_neighbors: bool
    w_av_second_neighbors: bool
    bare_coulomb_cutoff: float | None
    zeta_cutoff: float | None
    use_bgw_vcoul: bool
    bgw_vcoul_file: str | None
    bgw_vcoul_sym_wfn: str | None

    @property
    def uses_bgw_metal_q0shift(self) -> bool:
        return self.bgw_metal_q0_treatment == "bgw_q0shift"

    @property
    def analytic_q0_sphere(self) -> bool:
        """Whether the q=0 bare head uses the analytic-sphere split."""
        return self.head_minibz_average or self.uses_bgw_metal_q0shift


@dataclass(frozen=True)
class ScreeningConfig:
    """χ₀ / W screening: method choice + minimax-quadrature knobs.

    ``method`` selects the chi0 frequency treatment, and minimax is the
    ONLY one LORRAX implements (owner ruling 2026-08-06).  Nothing
    downstream branches on this field, and that is deliberate -- there is
    no second branch to take.  Its whole job is the ``__post_init__``
    check below, which is what makes it honest: before that check the
    field was pure decoration, so ``screening_method = ctsp`` parsed,
    normalised, and ran minimax without a word.

    ``diagrams`` is a DIFFERENT axis and it does have a second branch:
    ``method`` says how the chi0 frequency integral is taken, ``diagrams``
    says which series W sums (RPA, or the BSE ladder).  The fork lives in
    ``gw.screening.compute_screening_model`` and nowhere else.  Its
    default is spelled here as well as in ``_DEFAULTS`` so a hand-built
    config -- a tool, a test stub -- takes the SAME decision the parser
    would; a fallback that disagreed with the registered default is the
    defect the ``restart_q_storage`` note above describes.
    """
    method: str                   # "minimax" -- the only supported value
    occ_broadening_ev: float      # BGW MP1 width; 0 keeps step occupations
    minimax_target_error: float
    minimax_max_nodes: int
    regenerate_minimax_tables: bool
    minimax_energy_reference: str  # "midgap" | "vbm"
    diagrams: ScreeningDiagrams = ScreeningDiagrams.W_RPA
    # w_bse only — ``bse.w_ladder.compute_wc_qwedge``'s public
    # ``probe_chunk`` memory knob, threaded through the facade
    # (``gw.screening_bse._ladder_wedge``).  0 = whole padded basis in
    # one block (historical).  Deck key: ``ladder_probe_chunk``.
    ladder_probe_chunk: int = 0

    def __post_init__(self):
        if self.occ_broadening_ev < 0.0:
            raise ValueError("occ_broadening must be >= 0 eV.")
        if int(self.ladder_probe_chunk) < 0:
            raise ValueError(
                f"ladder_probe_chunk = {self.ladder_probe_chunk} must be "
                f">= 0 (0 = the whole padded centroid basis in one probe "
                f"block; a positive value bounds the per-block ladder "
                f"solve memory).")
        # REFUSE, naming what IS supported, instead of silently resolving
        # to it.  ``ctsp`` (contour-tail / separable-pole terminology from
        # the pre-minimax era) was accepted here for months and always ran
        # minimax, because no reader of this field has ever existed -- so
        # tests/regression/cohsex_debug/cohsex_test_ctsp_compare.in was a
        # "comparison" fixture comparing minimax against minimax.  A
        # silent downgrade of an EXPLICIT request is the failure mode
        # ffi/gate.py exists to prevent; the same rule applies to a
        # synonym that resolves to the supported method by accident
        # rather than by design.
        if self.method != "minimax":
            raise ValueError(
                f"screening_method = {self.method!r} is not supported.  "
                f"'minimax' (minimax quadrature for chi0 on the imaginary "
                f"axis) is the only screening method LORRAX implements.  "
                f"Note for decks carrying 'ctsp': that spelling was "
                f"accepted historically and SILENTLY RAN MINIMAX -- it "
                f"never selected a different method, so replacing it with "
                f"'screening_method = minimax' (or deleting the key, "
                f"which defaults to minimax) changes no result.")
        # REFUSE a spelling this axis does not have, naming the two it
        # does.  The parser already normalises through
        # ``coerce_screening_diagrams``; this is the guard for every OTHER
        # constructor of this record (tools, test stubs, a future reader),
        # so the axis cannot acquire a third value by assignment.
        if not isinstance(self.diagrams, ScreeningDiagrams):
            raise ValueError(
                f"screening_diagrams = {self.diagrams!r} is not supported.  "
                f"The legal set is exactly "
                f"{{{', '.join(d.value for d in ScreeningDiagrams)}}}: "
                f"'w_rpa' (default) sums the random-phase series "
                f"W = (1 - V chi0)^-1 V, and 'w_bse' sums the ladder series "
                f"W(w) - v = v (w - H)^-1 v with the statically screened "
                f"direct rung in H.  Pass a ScreeningDiagrams member or run "
                f"the value through gw_config.coerce_screening_diagrams, "
                f"which raises with this same set for a typo.  This axis is "
                f"ORTHOGONAL to screening_method and to compute_mode; see "
                f"docs/input_reference.md '## Screening'.")


@dataclass(frozen=True)
class DynamicSigmaConfig:
    """Ansatz-neutral real-frequency Sigma grid and output policy."""
    omega_min_ev: float
    omega_max_ev: float
    omega_step_ev: float
    regularization_ev: float
    window_edge_factor: float
    omega_layout: str
    fermi_reference: str
    sigma_at_dft_extrapolate: bool
    sigma_at_dft_energies: bool
    #: ``sigma_regularization_floor_ev``: "auto" or a float in eV.  See
    #: :func:`gw.ppm_windows.resolve_sigma_regularization`, which is the
    #: ONLY place this is interpreted -- the drivers and the sigma_mnk.h5
    #: writer all call it, so the stamped xi and the xi the kernel ran at
    #: cannot disagree.
    regularization_floor_ev: str | float = "auto"
    #: ``sigma_omega_patches_ev``: "" (default, the contiguous
    #: [min, max] grid) or "lo:hi, lo:hi, ..." — a union of uniform
    #: patches at ``omega_step_ev``, replacing the contiguous grid.  This
    #: is how a semicore run buys its dynamic range: dense points near
    #: the valence window and near each semicore QP cluster, NO points
    #: in the empty gap between them, so the MPA crossing rule never has
    #: to resolve the gap (docs/dev/crossing-rule-cost-law.md).  The
    #: Σ(ω)→E interpolation is searchsorted piecewise-linear and needs no
    #: uniformity; solved QP energies landing inside a hole are refused
    #: at the QSGW seam (gw.qsgw_utils.assert_omega_grid_covers).
    omega_patches_ev: str = ""
    #: Band-convergence extrapolation of Sigma_c, resolved from
    #: ``use_band_extrapolation`` (default TRUE) and its deprecated alias
    #: ``sigma_band_extrapolation`` by
    #: :func:`resolve_band_extrapolation`.  Applied by the GN/HL-PPM pipeline
    #: (``gw.ppm_pipeline``); on a non-PPM ``compute_mode`` it is turned OFF
    #: with a recorded note (or refused -- see ``band_extrapolation_explicit``)
    #: by ``gw.sigma_dispatch``, because the 1/N limit is mode-dependent and
    #: wrong for a static Coulomb hole.
    band_extrapolation: bool = USE_BAND_EXTRAPOLATION_DEFAULT
    #: Did a deck NAME either spelling?  Selects between auto-disabling and
    #: refusing on a non-PPM mode; see :func:`resolve_band_extrapolation`.
    band_extrapolation_explicit: bool = False
    #: WHICH estimator consumes the three bracket sums --
    #: ``spectral_shell`` (default) or ``band_index_only``.  Read once, by
    #: ``gw.ppm_pipeline``.  It selects nothing about the COMPUTE: both
    #: estimators read the same three points the same brackets produce, so
    #: this is a post-processing choice and switching it re-does no Sigma.
    band_extrapolation_estimator: str = BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT
    #: WHICH compile-time cuts produce the three sums.  The legacy
    #: ``total_fractions`` geometry remains the default; the explicit
    #: ``conduction_fractions`` arm reads the SAME fractions in the
    #: conduction coordinate (``n_occ + f*(N_max - n_occ)``), and
    #: ``conduction_energy_midpoint`` places N1 at half the included
    #: conduction manifold and N2 at the k-mean DFT-energy midpoint.
    band_extrapolation_bracket_scheme: str = BRACKET_SCHEME_DEFAULT
    #: Did the deck name the scheme?  An explicitly named compute control may
    #: not be silently ignored when extrapolation or all PPM stages are off.
    band_extrapolation_bracket_scheme_explicit: bool = False

    def __post_init__(self):
        if self.omega_step_ev <= 0.0:
            raise ValueError("sigma_omega_step_ev must be > 0.")
        if self.omega_max_ev < self.omega_min_ev:
            raise ValueError("sigma_omega_max_ev must be >= sigma_omega_min_ev.")
        self.parsed_omega_patches_ev()
        if self.fermi_reference not in ("vbm", "midgap", "mp1_fixed_n"):
            raise ValueError(
                "fermi_reference must be 'vbm', 'midgap' or 'mp1_fixed_n'.")
        if self.omega_layout not in ("replicated", "sharded"):
            raise ValueError(
                "sigma_omega_layout must be 'replicated' or 'sharded'.")
        # 'auto' or a non-negative float in eV.  A TYPO must refuse here,
        # not resolve to 'auto' -- a floor key that silently defaults is the
        # confound the key was added to remove.
        _floor = self.regularization_floor_ev
        if not (isinstance(_floor, str)
                and _floor.strip().lower() == "auto"):
            try:
                _floor_f = float(_floor)
            except (TypeError, ValueError):
                raise ValueError(
                    f"sigma_regularization_floor_ev must be 'auto' or a "
                    f"number of eV; got {_floor!r}.") from None
            if not (_floor_f >= 0.0):
                raise ValueError(
                    f"sigma_regularization_floor_ev must be >= 0; got "
                    f"{_floor_f!r}.")
        # REFUSE an unrecognised estimator, naming both, rather than falling
        # back to the default.  A misspelling that silently ran the default
        # would be an A/B measuring nothing -- the same rule
        # ``screening_method`` states above, for the same reason.
        if self.band_extrapolation_estimator not in (
                BAND_EXTRAPOLATION_ESTIMATORS):
            raise ValueError(
                f"band_extrapolation_estimator = "
                f"{self.band_extrapolation_estimator!r} is not a known "
                f"band-convergence estimator.  The two are: 'spectral_shell' "
                f"(the DEFAULT -- one decay exponent per external state from "
                f"the two shell increments against spectral moments of the "
                f"DFT eigenvalues, tail integrated to the finite plane-wave "
                f"basis) and 'band_index_only' (the incumbent two-parameter "
                f"S_inf + A/N least squares).  Both consume the SAME three "
                f"bracket sums; neither changes what is computed.")
        if self.band_extrapolation_bracket_scheme not in BRACKET_SCHEMES:
            raise ValueError(
                f"band_extrapolation_bracket_scheme = "
                f"{self.band_extrapolation_bracket_scheme!r} is not known; "
                f"choose one of {BRACKET_SCHEMES}.  This key selects the "
                f"three band sums that are computed, so no fallback is "
                f"safe.")
        if (self.band_extrapolation_bracket_scheme_explicit
                and not self.band_extrapolation):
            raise ValueError(
                "band_extrapolation_bracket_scheme was explicitly named, "
                "but use_band_extrapolation = false, so no bracket planner "
                "would consume it.  Remove the scheme key or enable band "
                "extrapolation.")

    def parsed_omega_patches_ev(self):
        """The validated ``[(lo, hi), ...]`` patch list, or ``[]``.

        Parsed from ``"lo:hi, lo:hi"``.  Patches must be well-formed
        (hi > lo), ascending, and separated by at least one step —
        overlapping or touching patches are a deck typo, refused rather
        than silently merged.
        """
        text = str(self.omega_patches_ev or "").strip()
        if not text:
            return []
        patches = []
        for piece in text.split(","):
            piece = piece.strip()
            if not piece:
                continue
            parts = piece.split(":")
            try:
                lo, hi = (float(parts[0]), float(parts[1])) \
                    if len(parts) == 2 else (np.nan, np.nan)
            except ValueError:
                lo = hi = np.nan
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                raise ValueError(
                    "sigma_omega_patches_ev must be 'lo:hi, lo:hi, ...' "
                    f"with hi > lo in eV; could not parse {piece!r}")
            patches.append((lo, hi))
        for (l0, h0), (l1, h1) in zip(patches, patches[1:]):
            if l1 < h0 + self.omega_step_ev:
                raise ValueError(
                    "sigma_omega_patches_ev patches must be ascending and "
                    f"separated by at least one step; [{l0}:{h0}] then "
                    f"[{l1}:{h1}] at step {self.omega_step_ev}. Merge "
                    "them into one patch instead.")
        return patches


@dataclass(frozen=True)
class PPMConfig:
    """Parameters specific to the two-point plasmon-pole ansatz."""
    # --- Model selection ---
    model: str                    # "gn" | "hl" — picked by ComputeMode usually
    omega_p: float                # probe ω (Ry); imag for GN, real for HL
    fallback_omega: float
    head_omega_h_ry: float | None # override Ω_h directly (BGW comparisons)
    #: Probe-χ₀ reuse: "off" (dedicated probe quadrature, exact historical
    #: path) | "auto" (weights-only refit on the static τ nodes, probe χ₀
    #: folded into the static sweep when the error gate passes — see
    #: _DEFAULTS["ppm_probe_chi_reuse"]).
    probe_chi_reuse: str

    # --- σ-quadrature minimax ---
    sigma_target_error: float
    sigma_max_nodes: int

    invalid_mode: str             # "zero" | "2ry" | "static_limit" | "infinity"(alias)

    def __post_init__(self):
        # Validate scalar knobs once, at the parse site (values are already
        # normalized in ``from_input_file``).  Capability gating for
        # invalid_mode ('imaginary' → NotImplementedError, needs a
        # complex-Ω path) stays in the Σ^c kernel — this checks only
        # that the *value* is recognized.
        if self.invalid_mode not in (
            "zero", "skip", "2ry", "static_limit", "infinity", "imaginary"
        ):
            raise ValueError(
                f"ppm.invalid_mode: unknown value {self.invalid_mode!r}")
        if self.probe_chi_reuse not in ("off", "auto"):
            raise ValueError(
                "ppm_probe_chi_reuse must be 'off' or 'auto'; "
                f"got {self.probe_chi_reuse!r}.")


@dataclass(frozen=True)
class MPAConfig:
    """Multipole sample geometry and bounded pole-consumption policy."""

    n_poles: int
    material_class: str
    sampling_alpha: int
    sampling_schedule: str
    pole_solver: str
    varpi_near_ry: float
    varpi_far_ry: float
    #: Metal near-line origin shift in Ry; ``None`` = the published
    #: ``sampling._METAL_ORIGIN_SHIFT`` default (2e-5 Ry = 1e-5 Ha).
    metal_origin_shift_ry: float | None
    pole_batch_size: int
    sigma_sector_target_error: float
    sigma_crossing_target_error: float
    sigma_max_nodes: int
    #: ``mpa_sigma_omega_cluster_gap_ry``: the Σ planner splits each
    #: branch's |ω| evaluation values into clusters at gaps larger than
    #: this, decomposing the crossing core per cluster (shell + Laplace
    #: slabs) so the eta-resolved rule bandwidth is set by the cluster
    #: span, not the dynamic range.  A contiguous grid is always one
    #: cluster (the incumbent plan, bit-for-bit); pair with
    #: ``sigma_omega_patches_ev`` to actually gap the grid.
    sigma_omega_cluster_gap_ry: float = 1.0
    #: ``occupation_window_threshold``: the OCCUPANCY at which a band stops
    #: counting toward a metallic Green's-function branch.  The Σ planner's
    #: cut is on the branch WEIGHT (``f`` on val, ``1 − f`` on cond), so the
    #: floor it applies is ``1 − threshold`` — 0.995 ⇒ ``|weight| > 0.005``
    #: — and it is a MAGNITUDE because MP1 occupations are never clipped.
    #: 1.0 recovers the historical exact ``weight != 0`` rule.  Validated at
    #: its consumer (``gw.mpa.sigma_windows._weight_floor``) per the ruling
    #: below.
    occupation_window_threshold: float = 0.995

    # pole_batch_size and the two Sigma target errors are validated at their
    # consumers (gw.mpa.sigma / gw.mpa.sigma_windows) — single owner.
    def __post_init__(self):
        if not 1 <= self.n_poles <= 16:
            raise ValueError("mpa_n_poles must be in [1, 16]")
        if self.material_class not in ("insulator", "metal"):
            raise ValueError(
                "mpa_material_class must be 'insulator' or 'metal'; "
                f"got {self.material_class!r}")
        if self.sampling_alpha not in (1, 2):
            raise ValueError("mpa_sampling_alpha must be 1 or 2")
        if self.sampling_schedule not in ("nested", "leon"):
            raise ValueError(
                "mpa_sampling_schedule must be 'nested' or 'leon'; "
                f"got {self.sampling_schedule!r}")
        if self.pole_solver not in ("loewner", "companion", "thiele"):
            raise ValueError(
                "mpa_pole_solver must be 'loewner', 'companion', or "
                "'thiele'; "
                f"got {self.pole_solver!r}")
        if not (0.0 < self.varpi_near_ry < self.varpi_far_ry):
            raise ValueError(
                "MPA line heights must satisfy 0 < near < far")
        # The origin shift is metal-only geometry: an insulating plan puts
        # its first near-line sample at z = 0 exactly, so a shift there is
        # an off-dial that must refuse rather than be ignored.
        if self.metal_origin_shift_ry is not None:
            if self.material_class != "metal":
                raise ValueError(
                    "mpa_metal_origin_shift_ry is a metal-only key "
                    f"(mpa_material_class = {self.material_class} here): it "
                    "moves the METAL near line's first sample off zero, and "
                    "an insulating plan samples z = 0 exactly. Remove it, or "
                    "set mpa_material_class = metal.")
            if not (0.0 < self.metal_origin_shift_ry < self.varpi_near_ry):
                raise ValueError(
                    "mpa_metal_origin_shift_ry must satisfy 0 < shift < "
                    f"mpa_varpi_near_ry; got shift = "
                    f"{self.metal_origin_shift_ry!r} Ry against "
                    f"mpa_varpi_near_ry = {self.varpi_near_ry!r} Ry. The "
                    "shift dodges the zero-energy intraband pile-up without "
                    "climbing off the near line it sits on. NOTE the unit: "
                    "this key is Ry like every deck key, so it is TWICE the "
                    "Hartree value the multipole papers quote (published "
                    "default 1e-5 Ha = 2e-5 Ry).")
        if self.sigma_max_nodes < 2:
            raise ValueError("mpa_sigma_max_nodes must be at least two")
        if not (np.isfinite(self.sigma_omega_cluster_gap_ry)
                and self.sigma_omega_cluster_gap_ry > 0.0):
            raise ValueError(
                "mpa_sigma_omega_cluster_gap_ry must be finite and "
                "positive (Ry); a huge value disables clustering")

    def sample_plan(self, omega_m_ry):
        """Return the configured double-parallel frequency plan in Ry.

        This is sampling geometry only.  In particular, constructing a
        metallic plan does not claim that the occupation-weighted χ/Σ
        evaluators needed to consume it have landed.
        """
        from .mpa.sample_plan import mpa_plan

        return mpa_plan(
            self.n_poles, omega_m_ry,
            material_class=self.material_class,
            alpha=self.sampling_alpha,
            schedule=self.sampling_schedule,
            varpi_near=self.varpi_near_ry,
            varpi_far=self.varpi_far_ry,
            origin_shift=self.metal_origin_shift_ry,
            energy_unit="Ry",
        )


#: The ``sc_head_update`` values that rebuild the q->0 head every QSGW
#: iteration, i.e. the ones a fractionally occupied deck may choose.  One
#: tuple, so the vocabulary, the mandatory-metal rule and the driver's
#: dispatch cannot disagree about what "a metal head mode" is.
METAL_HEAD_UPDATES = ("parallel_transport", "dft_velocity")


@dataclass(frozen=True)
class SCConfig:
    """Self-consistency loop knobs (read only when qp_solver=self_consistent).

    Promoted from the ``LORRAX_SC_*`` env vars (NEXT_TARGETS #11); the
    envs are still honored as deprecated overrides at config construction
    (``from_input_file`` prints a note when one is active).

    - ``max_iter`` / ``tol_ev``: loop length and RMS-ΔE convergence (eV).
    - ``accelerator``: ``"rcrop"`` (Anderson-style restart-CROP, default —
      required for QSGW's typical 2-cycle Jacobian) or ``"linear"``
      (plain α-mixing, diagnostic).  rCROP makes TWO ``gw_iteration_map``
      calls per accelerator iteration (trial + residual).
    - ``history_depth``: rCROP history (m=5 is BGW's QSGW default).
    - ``mixing``: linear-mixing α (``accelerator="linear"`` only).
    - ``dump_dir``: per-iteration E-history .npy dump dir (None = off).
    - ``eigh``: which eigh diagonalises the ``(nk, nb, nb)`` carry each
      iteration — ``"native"`` (k-sharded batch: one WHOLE ``(nb, nb)``
      tile per device), ``"distributed"`` (one tile spread over the mesh),
      or ``"auto"``.  A LAYOUT choice: it does not change the physics and
      it is deliberately not a side effect of ``density_self_consistent``,
      which is what used to select it.  Resolution lives in
      ``sc_iteration._resolve_sc_eigh``.
    """
    max_iter: int
    tol_ev: float
    accelerator: str      # "rcrop" | "linear"
    history_depth: int
    mixing: float
    dump_dir: str | None
    eigh: str = "auto"    # "auto" | "native" | "distributed"
    #: "off" | "parallel_transport" | "dft_velocity".  The two non-off
    #: values are the METAL head modes: both run the per-iteration head
    #: chain, and they differ only in the velocity operator they feed it —
    #: ``parallel_transport`` adds the fourth-order finite-link covariant DΔH
    #: correction from saved neighbour overlaps, ``dft_velocity`` uses the
    #: exact DFT p-matrix
    #: velocity alone.  ``METAL_HEAD_UPDATES`` is the vocabulary consumers
    #: test against; do not spell the pair out a second time.
    head_update: str = "off"

    def __post_init__(self):
        if self.max_iter < 1:
            raise ValueError("sc_max_iter must be >= 1.")
        if self.tol_ev <= 0.0:
            raise ValueError("sc_tol_ev must be > 0.")
        if self.accelerator not in ("rcrop", "linear"):
            raise ValueError(
                f"sc_accelerator must be 'rcrop' or 'linear'; "
                f"got {self.accelerator!r}.")
        if self.history_depth < 1:
            raise ValueError("sc_history_depth must be >= 1.")
        if not (0.0 < self.mixing <= 1.0):
            raise ValueError("sc_mixing must be in (0, 1].")
        if self.eigh not in ("auto", "native", "distributed"):
            raise ValueError(
                f"sc_eigh must be 'auto', 'native' or 'distributed'; "
                f"got {self.eigh!r}.")
        if self.head_update not in ("off",) + METAL_HEAD_UPDATES:
            raise ValueError(
                "sc_head_update must be 'off', "
                + " or ".join(repr(v) for v in METAL_HEAD_UPDATES)
                + f"; got {self.head_update!r}.")


@dataclass(frozen=True)
class EQP2Config:
    """Fixed-Sigma eigenvalue self-consistency for the opt-in eqp2 file.

    This is deliberately separate from :class:`SCConfig`: it does not
    rebuild G, chi0, W, or Sigma.  It repeatedly evaluates and rotates the
    one-shot full-matrix Sigma(omega) table, diagonalizes the resulting QP
    Hamiltonian, and tests the worst eigenvalue change in eV.
    """

    enabled: bool = False
    tol_ev: float = 1.0e-3
    max_iter: int = 20
    accelerator: str = "rcrop"
    history_depth: int = 5

    def __post_init__(self):
        if self.tol_ev <= 0.0:
            raise ValueError("eqp2_tol_ev must be > 0.")
        if self.max_iter < 1:
            raise ValueError("eqp2_max_iter must be >= 1.")
        if self.accelerator not in ("rcrop", "linear"):
            raise ValueError(
                "eqp2_accelerator must be 'rcrop' or 'linear'; got "
                f"{self.accelerator!r}.")
        if self.history_depth < 1:
            raise ValueError("eqp2_history_depth must be >= 1.")


@dataclass(frozen=True)
class MemoryConfig:
    """Per-device memory budget + chunk sizing + AOT chunk-chooser flag.

    ``memory_per_device_gb=0`` triggers GPU auto-detection at config
    construction time.  ``chunk_target_utilization=0`` is the auto sentinel;
    a positive ``ISDF_CHUNK_TARGET_UTILIZATION`` value overrides the
    planner's spin-aware default after clamping to ``[0.85, 1.0]``.
    """
    per_device_gb: float
    chunk_target_utilization: float
    band_chunk_size: int
    r_chunk_override: int         # 0 = auto
    gflat_chunk_size: int         # 0 = planner-picked
    vq_g_chunk_size: int          # 0 = auto _pick_g_chunk(ngkmax)
    low_mem_bands: bool           # gw.wavefunction_bundle layout="face"


@dataclass(frozen=True)
class BackendConfig:
    """FFI/linalg backend selection resolved once at startup."""
    w_dyson_solver: str  # "local" | "distributed" (normalized; W Dyson plan)
    distributed_cholesky: str  # "auto" | "off" | "cusolvermp" | "slate"
    distributed_lu: str        # "auto" | "off" | "cusolvermp" | "scalapack"
    distrib_la_batched_route: str  # "auto" | "batch_reshard"
    eigh_backend: str          # resolved: auto|off|distributed|cusolvermp|
                               #           slate|scalapack (use_low_mem_eigh
                               #           already folded in)
    use_low_mem_eigh: bool     # what the deck ASKED for, kept for the banner
    zeta_ridge: float          # charge-CCT Tikhonov ridge ε (rel. to tr/n)
    charge_zeta_solve: str     # "rank_truncate" | "cholesky"
    distributed_zeta_solve: str  # "auto"|"replicated"|"per_q"|"distributed"
    zeta_rcond: float          # rank-truncation cutoff (·λ_max)
    transverse_zeta_solve: str  # "ridge" | "rank_truncate" (bispinor ζ_T)
    transverse_zeta_rcond: float  # transverse cut τ (·|λ|_max)
    gamma_contract_mode: str  # "take" | "einsum" | "scan"

    def summary(self) -> str:
        """One-line "what's active" for the run banner."""
        return (
            "backend: "
            + (f"w_dyson_solver={self.w_dyson_solver}, "
               if self.w_dyson_solver != "local" else "")
            + f"distributed_cholesky={self.distributed_cholesky}, "
            f"distributed_lu={self.distributed_lu}, "
            + (f"distrib_la_batched_route={self.distrib_la_batched_route}, "
               if self.distrib_la_batched_route != "auto" else "")
            + (f"eigh_backend={self.eigh_backend}"
               + (" (use_low_mem_eigh)" if self.use_low_mem_eigh else "")
               + ", " if self.eigh_backend != "auto" else "")
            + f"charge_zeta_solve={self.charge_zeta_solve}"
            + (f"(rcond={self.zeta_rcond:g})"
               if self.charge_zeta_solve == 'rank_truncate' else '')
            + (f", zeta_ridge={self.zeta_ridge:g}"
               if self.zeta_ridge else '')
            + (f", distributed_zeta_solve={self.distributed_zeta_solve}"
               if self.distributed_zeta_solve != 'auto' else '')
            + (f", transverse_zeta_solve={self.transverse_zeta_solve}"
               f"(rcond={self.transverse_zeta_rcond:g})"
               if self.transverse_zeta_solve != 'ridge' else '')
            + f", gamma_contract={self.gamma_contract_mode}"
        )


@dataclass(frozen=True)
class DebugConfig:
    """Debug-only flags + auxiliary output filenames."""
    sigma_freq_debug_output: bool
    sigma_freq_debug_file: str
    write_wfn_h5: bool


@dataclass(frozen=True)
class BSEConfig:
    """BSE interpolation setup (htransform-driven fine-k wfn recovery).

    See ``bandstructure.bse_setup.compute_wfns_fi``.  ``get_centroids_fi``
    is the master gate; if False the rest is unused.
    """
    get_centroids_fi: bool
    wfn_fi_min: int
    wfn_fi_max: int
    kgrid_fi: str
    wfn_fi_q_chunk: int   # 0 = N_q_co (prod(kgrid_co)); see compute_wfns_fi


#: Relative tolerance on the ``occ_smearing_width_ry`` / ``occ_broadening``
#: agreement.  Loose enough that the eV<->Ry round trip cannot trip it (a
#: deck written with CODATA 13.605693122994 eV/Ry and read back through
#: ``common.units.RYD_TO_EV`` = 13.6056980659 differs by 3.6e-7 relative),
#: tight enough that a CONVENTION error — the factor of two between the QE
#: degauss and BerkeleyGW's half-width — is three orders of magnitude clear
#: of it.
_OCC_WIDTH_RTOL = 1.0e-4


def _validate_occupation_smearing(mpa, sigma, screening, family, width_ry):
    """Cross-key deck validation for the metallic occupation model.

    Metal decks must declare the smearing pair and the two Sigma keys the
    metallic MPA path actually honors; insulating decks must not carry the
    smearing pair at all (an off-dial refusal, never a silent ignore).
    Module-level so tests exercise it without a full parse.

    THE WIDTH CONVENTION, stated once, here, because two keys carry it.
    ``occ_smearing_width_ry`` and ``occ_broadening`` are the SAME width in
    different units: BerkeleyGW's ``occ_broadening``, whose MP1 argument is
    ``(E - mu) / (2 * width)``.  The QE ``degauss`` is TWICE it.  A deck
    that sets both and disagrees is refused below rather than silently
    resolved, because the two ways of being wrong (halving or doubling the
    smearing) are indistinguishable in the output.
    """
    if mpa.material_class == "metal":
        if family is None or width_ry is None:
            raise ValueError(
                "mpa_material_class = metal requires the occupation "
                "smearing pair: set occ_smearing_family = mp1 and "
                "occ_smearing_width_ry = <BerkeleyGW occ_broadening, in "
                "Ry> = <the QE degauss, in Ry> / 2. "
                "They cannot be derived from WFN.h5 (mf_header carries "
                "no smearing metadata).")
        if family != "mp1":
            raise ValueError(
                "occ_smearing_family supports only 'mp1' "
                f"(Methfessel-Paxton order 1); got {family!r}. Other "
                "families need their own occupation solve and error "
                "certificates before they can be honored.")
        if not width_ry > 0.0:
            raise ValueError(
                "occ_smearing_width_ry must be > 0 for a metal; got "
                f"{width_ry!r}")
        if sigma.fermi_reference != "mp1_fixed_n":
            raise ValueError(
                "mpa_material_class = metal requires fermi_reference = "
                f"mp1_fixed_n (got {sigma.fermi_reference!r}): the metallic "
                "Sigma measures energies against the fixed-N MP1 chemical "
                "potential, not a gap-derived reference.")
        if sigma.omega_layout != "sharded":
            raise ValueError(
                "mpa_material_class = metal requires sigma_omega_layout = "
                f"sharded (got {sigma.omega_layout!r}): the MPA Sigma "
                "emits the mesh-sharded omega cube only.")
    else:
        if family is not None or width_ry is not None:
            raise ValueError(
                "occ_smearing_family / occ_smearing_width_ry are metal-only "
                "keys (mpa_material_class = insulator here). Remove them, "
                "or set mpa_material_class = metal.")
        if sigma.fermi_reference == "mp1_fixed_n":
            raise ValueError(
                "fermi_reference = mp1_fixed_n requires "
                "mpa_material_class = metal (it names the fixed-N MP1 "
                "chemical potential, which insulating decks do not solve).")

    # Cross-key width agreement, checked last so the class-level off-dial
    # refusals above own their own messages.  ``occ_broadening = 0`` is the
    # documented "step occupations" dial, not a width, so it is never a
    # disagreement — the b24/b40 step-occupation control arms deliberately
    # set exactly that beside a live smearing pair.
    broadening_ev = float(screening.occ_broadening_ev)
    if width_ry is not None and broadening_ev > 0.0:
        broadening_ry = broadening_ev / RYD_TO_EV
        if abs(width_ry - broadening_ry) > _OCC_WIDTH_RTOL * abs(broadening_ry):
            raise ValueError(
                "occ_smearing_width_ry and occ_broadening are the SAME "
                "width in different units, and this deck's two values "
                f"disagree: occ_smearing_width_ry = {width_ry!r} Ry vs "
                f"occ_broadening = {broadening_ev!r} eV = "
                f"{broadening_ry:.12g} Ry (RYD_TO_EV = {RYD_TO_EV}). "
                "Both are BerkeleyGW's occ_broadening, whose MP1 argument "
                "is (E-mu)/(2*width); the QE degauss is TWICE either of "
                f"them, so this deck implies degauss = {2.0 * width_ry:.12g}"
                f" Ry from the first key and {2.0 * broadening_ry:.12g} Ry "
                "from the second. Set occ_smearing_width_ry = degauss/2 "
                "and occ_broadening = occ_smearing_width_ry * "
                f"{RYD_TO_EV} eV/Ry, or remove one of them.")


def _validate_metal_compute_mode(config):
    """Bind the declared metal formulation to its occupation-aware mode."""
    if config.mpa.material_class != "metal":
        return
    mode = config.compute_mode
    if mode is ComputeMode.MPA:
        return

    raw_mode = (config.compute_mode_raw or "auto").strip().lower()
    got_mode = (
        f"compute_mode = auto (resolved to {mode.value})"
        if raw_mode == "auto" else f"compute_mode = {mode.value}")
    raise ValueError(
        "GATE metal_material_class_requires_mpa: "
        "mpa_material_class = metal is refused outside the "
        "occupation-aware MPA path.\n"
        f"  got: mpa_material_class = metal; {got_mode}\n"
        "  want: mpa_material_class = metal with compute_mode = mpa\n"
        "  fix: set compute_mode = mpa for a metal, or set "
        "mpa_material_class = insulator for an x_only / cohsex / gn_ppm / "
        "hl_ppm run\n"
        f"  why: the compute_mode = {mode.value} head route accepts no "
        "occupation_state and uses step occupations from the dipole/static "
        "head path; it cannot honor the fixed-N MP1 mu, fractional "
        "occupations, Drude term, or Thomas-Fermi limit declared by the "
        "metal key\n"
        "  doc: docs/theory/metallic-mpa-screening.md")


@dataclass(frozen=True)
class LorraxConfig:
    """Unified, immutable configuration for a LORRAX GW calculation.

    Created once via :meth:`from_input_file` and threaded through the
    entire driver.  Top-level fields are ``hot-path`` reads (system
    geometry + the orthogonal mode flags); group sub-dataclasses
    organise the remaining ~70 input keys along the same axes the
    input file's section comments already use.

    Access pattern::

        config.compute_mode           # -> ComputeMode enum
        config.head.wcoul0_source     # head plumbing
        config.ppm.omega_p            # PPM probe ω
        config.sigma.omega_layout     # shared dynamic-Sigma output policy
        config.debug.sigma_freq_debug_output

    See module docstring for the full grouping.  ``cohsex.in`` keys
    are unchanged — input files written for prior versions still parse
    (the factory unflattens the dict into sub-dataclasses).
    """

    # --- System geometry (top-level; hot path) ---
    nval: int
    ncond: int
    #: The LOADED band extent = ``bands.isdf`` = ``max(chi, sigma)``.  On
    #: every deck that names only the umbrella (or only the transitional
    #: ``nband`` alias) this is exactly the umbrella, which is why the whole
    #: tree reads unchanged.  On a SPLIT deck it is the larger of the two
    #: counts: the ψ is loaded once over ``[b0, b4)`` and the smaller
    #: consumer takes a narrower window inside it.  A consumer that wants
    #: "how many bands does χ0 sum" or "how many does Σ sum" must ask
    #: ``bands.chi`` / ``bands.sigma``, never this field.
    nband: int
    #: The resolved band-count family.  See :func:`resolve_band_counts` for
    #: the precedence and :meth:`BandCounts.describe` for the log line.
    bands: BandCounts
    #: ζ-fit band-window top.  ``None`` == follow ``nband`` (every deck
    #: written before 2026-08-11); an int NARROWS the ζ fit's band ranges
    #: without touching ``b4``, the χ0/Σ band-sum top.  See ``_DEFAULTS``.
    zeta_nband: int | None
    sys_dim: int
    density_self_consistent: bool
    sc_on_ibz: bool
    #: auto | stored | isdf | gspace — see HARTREE_SOURCES.
    hartree_source: str
    #: DFT occupation smearing of the starting point ("mp1" = Methfessel-
    #: Paxton order 1, the only certified family).  REQUIRED as a pair when
    #: ``mpa_material_class = metal``; refused under insulator.  Deck keys,
    #: not derived: WFN.h5's mf_header carries el/occ/w but no smearing
    #: metadata (see ``_DEFAULTS``).  Validated by
    #: ``_validate_occupation_smearing``.
    #:
    #: ``occ_smearing_width_ry`` IS ``occ_broadening`` in Ry — BerkeleyGW's
    #: convention, MP1 argument ``(E-mu)/(2*width)`` — and therefore HALF
    #: the QE ``degauss``.  It is the metal path's single width source; see
    #: :attr:`occ_broadening_ry`.
    occ_smearing_family: str | None
    occ_smearing_width_ry: float | None
    #: ``occupation_clamp_tol``: the distance from 0 or 1 within which an
    #: MP1 occupation is snapped to EXACTLY 0 or 1, applied at the point
    #: the occupations are evaluated and therefore inside the fixed-N root
    #: (``gw.efermi.clamp_occupation_tail``).  Distinct from
    #: ``occupation_window_threshold``, which decides band membership of a
    #: Green's-function branch and is not replaced by this.  Validated at
    #: its consumer (``gw.efermi.occupation_clamp_tol``), the same shape
    #: ``occupation_window_threshold`` uses.
    occupation_clamp_tol: float

    # --- Core mode flags (top-level; hot path) ---
    restart: bool
    #: Persist ``tmp/isdf_tensors_*.h5`` at all.  True preserves today's
    #: behaviour; see ``_DEFAULTS["write_restart_tensors"]`` for why this is
    #: a COMPLEMENT to q_irr storage and not an alternative to it.
    write_restart_tensors: bool
    #: Add the QSGW Σ_xc cube and the QP energy ladders to ``sigma_mnk.h5``.
    #: False (the default) is byte-for-byte today's file; see
    #: ``_DEFAULTS["write_qsgw_datasets"]`` for what each dataset is and
    #: which compute mode produces it.
    write_qsgw_datasets: bool
    #: RAW ``restart_q_storage`` request — "full" (the default) | "auto" |
    #: "ibz".  Validated at parse time, resolved LATE
    #: (``gw.restart_q_storage``): ``auto``'s answer depends on the run's
    #: centroid set, which does not exist yet here.  ``full`` needs no
    #: resolution but still goes through the same seam, so there is one
    #: resolution point rather than a fast path beside it.  Same ``_raw``
    #: convention as ``compute_mode_raw``.
    restart_q_storage_raw: str
    #: ``qp_rotations_k_storage`` — "auto" (the default) | "full" | "ibz".
    #: NOT a ``_raw``: unlike ``restart_q_storage`` there is nothing to
    #: resolve late, because the question it asks ("do these rows unfold
    #: back to what I hold?") is answered by the arrays themselves at the
    #: writer, not by a centroid set that does not exist yet.
    qp_rotations_k_storage: str
    #: The deck keys THIS DECK NAMED, as opposed to inherited from
    #: ``_DEFAULTS``.  Empty for a config built from a hand-made params dict,
    #: which is why every consumer must treat "absent" as "did not ask" — see
    #: ``gw.restart_q_storage._deck_named_the_key``.  Exists so a key that is
    #: on its way out can speak to the decks that pin it without speaking to
    #: every other deck in the tree.
    raw_input_keys: frozenset
    compute_mode_raw: str         # "auto" | one of ComputeMode.value strings
    qp_solver_raw: str            # "auto" | one of QPSolver.value strings
    do_screened: bool
    bispinor: bool
    do_G0: bool
    self_consistent: bool         # deprecated alias; ``qp_solver`` is canonical
    use_ppm_sigma: bool           # legacy mirror; ``compute_mode`` is canonical
    no_degen_averaging: bool
    degen_avg_tol_ry: float

    # --- Sub-dataclass groups (everything else) ---
    paths: FilePaths
    head: HeadConfig
    screening: ScreeningConfig
    sigma: DynamicSigmaConfig
    ppm: PPMConfig
    mpa: MPAConfig
    sc: SCConfig
    eqp2: EQP2Config
    memory: MemoryConfig
    backend: BackendConfig
    debug: DebugConfig
    bse: BSEConfig

    # --- Optional parsed blocks ---
    kpoints_crystal_b: dict | None = None

    # --- Input directory (for resolving relative paths at runtime) ---
    input_dir: str = ""
    # --- The deck this config was parsed from -------------------------
    # ``input_dir``'s missing half.  A stage that has to hand the DECK to
    # another component -- not a path resolved out of it -- had no way to
    # name it: the driver's ``args.input`` died at ``from_input_file``.  The
    # first such consumer is the ``screening_diagrams = w_bse`` handoff:
    # ``bse.w_ladder.compute_wc_qwedge`` needs the deck because the
    # irreducible q wedge comes from ``SymMaps``, which is built from the
    # WFN the DECK names, and no restart file records it.  Empty for a
    # config built by hand, and the consumer refuses on empty rather than
    # guessing a deck beside the restart.
    input_file: str = ""

    def __post_init__(self):
        """Refuse metallic and head settings outside their landed scope."""
        _validate_metal_compute_mode(self)
        if self.eqp2.enabled:
            if self.qp_solver is not QPSolver.ONE_SHOT_DFT:
                raise ValueError(
                    "write_eqp2=true is an additional fixed-Sigma result "
                    "for qp_solver=one_shot_dft; it cannot be combined with "
                    f"qp_solver={self.qp_solver.value}.  Use one_shot_dft, "
                    "or disable write_eqp2 and choose fixed_point / "
                    "self_consistent as the primary QP treatment.")
            if not self.compute_mode.is_dynamic:
                raise ValueError(
                    "write_eqp2=true requires a dynamic full-matrix "
                    "Sigma_c(omega) table; choose compute_mode=gn_ppm, "
                    "hl_ppm, or mpa, or disable write_eqp2.")
        if (self.qp_solver is QPSolver.SELF_CONSISTENT
                and self.head.correction is HeadCorrection.OFF
                and self.sc.head_update != "off"):
            raise ValueError(
                "sc_head_update requests a QSGW velocity/head rebuild, but "
                "head_correction=off removes that channel. The update flag "
                "does not override the head policy; set sc_head_update=off "
                "or choose head_correction=full/no_local_fields.")
        if self.screening.occ_broadening_ev == 0.0:
            return
        if self.qp_solver is not QPSolver.SELF_CONSISTENT:
            raise ValueError(
                "occ_broadening > 0 is currently implemented only for "
                "qp_solver=self_consistent.")
        if self.sc.head_update not in METAL_HEAD_UPDATES:
            raise ValueError(
                "occ_broadening > 0 currently updates only the QSGW head; "
                "set sc_head_update to one of "
                + ", ".join(METAL_HEAD_UPDATES)
                + ".")
        # rCROP is legal on metallic decks since the ENTRY-solve rule
        # (2026-08-15): gw_iteration_map solves its MP1 occupation state
        # from the spectrum of the H it is handed, every call, so F(H) is
        # a self-map of H alone and any accelerator trajectory (trial or
        # accepted iterates) gets occupations consistent with its own H
        # by construction.  The refusal this replaces guarded the old
        # END-of-iteration carry, which was exact only on the mixing=1
        # linear trajectory.

    # ------------------------------------------------------------------
    #  Derived config objects
    # ------------------------------------------------------------------

    @property
    def occ_broadening_ry(self) -> float:
        """THE occupation-smearing width consumed at runtime, in Ry.

        One width, one owner.  Every MP1 solve in the driver reads this
        and nothing else, so the two deck keys that carry the width can
        no longer feed different numbers into different stages.

        CONVENTION — BerkeleyGW's, not QE's.  ``gw.efermi``'s MP1
        argument is ``(E - mu) / (2 * width)`` (``_mp1_values``), the same
        form BerkeleyGW uses (``Common/input_utils.f90:380``), so this
        width is HALF the QE ``degauss``.  Measured, not asserted: at
        ``degauss = 0.02 Ry`` the sodium SOC deck's BGW arm reproduces
        QE's own stored occupations to 7.1e-12 with ``occ_broadening =
        0.13605693122994 eV = 0.01 Ry`` (CLAIMS 185), and LORRAX's mu
        lands 6.2e-7 eV from QE's E_F at the same width (CLAIMS 180).
        ``OccupationState.smearing_width_ry`` — the field this feeds and
        the one stamped into the MPA fit store — is the same quantity
        under the same name.

        SOURCE.  ``occ_smearing_width_ry`` when the deck declares it (the
        metal path); otherwise ``occ_broadening`` converted from eV, which
        is every insulating and pre-metal deck and is bit-for-bit what
        those decks used before this key existed.  When both are set
        ``_validate_occupation_smearing`` has already refused any
        disagreement beyond ``_OCC_WIDTH_RTOL``, so the branch cannot
        change the physics of a deck that carries both — it only decides
        which of two agreeing numbers is the exact one, and the deck's own
        Ry value is the one that did not make a round trip through eV.

        NOT A DIAL.  ``occ_broadening == 0`` remains the switch that
        selects step occupations (``sc_iteration._solve_head_occupations``
        and the metal V_H rebuild both read it as such); this property
        answers "how wide", never "whether".
        """
        if self.occ_smearing_width_ry is not None:
            return float(self.occ_smearing_width_ry)
        return float(self.screening.occ_broadening_ev) / RYD_TO_EV

    @property
    def compute_mode(self) -> ComputeMode:
        """Resolve ``compute_mode`` from explicit input or legacy flags.

        ``compute_mode = auto`` (the default) infers from
        ``do_screened`` / ``use_ppm_sigma`` / ``ppm.model``.  An explicit
        setting overrides them; the legacy fields are still parsed for
        back-compat but the enum is the load-bearing axis the driver
        pivots on.

        RESOLVING IS NOT PERMITTING.  This property answers "which mode
        did the deck ask for", and it answers it for every member of the
        enum including the ones whose Σ stage has not landed — the
        refusal for those is
        :func:`refuse_unimplemented_compute_mode`, called at driver
        entry, so that config-only consumers (the deck echo, the layering
        tests, an operator reading a config back) can name the mode
        without tripping over it.  ``auto`` never infers an unimplemented
        mode: the legacy flags it reads predate all of them.
        """
        raw = (self.compute_mode_raw or "auto").strip().lower()
        if raw == "auto":
            if self.use_ppm_sigma:
                if not self.do_screened:
                    raise ValueError(
                        "use_ppm_sigma=true requires do_screened=true."
                    )
                return (
                    ComputeMode.HL_PPM
                    if str(self.ppm.model).strip().lower() == "hl"
                    else ComputeMode.GN_PPM
                )
            return ComputeMode.COHSEX if self.do_screened else ComputeMode.X_ONLY
        try:
            explicit = ComputeMode(raw)
        except ValueError as exc:
            raise ValueError(
                f"compute_mode={raw!r} invalid; expected one of: "
                f"{', '.join(m.value for m in ComputeMode)}, or 'auto'."
            ) from exc
        # The enum is load-bearing: an explicit screened mode contradicts
        # the legacy ``do_screened = false``.  (Explicit ``x_only`` simply
        # wins over the do_screened default — the driver derives its
        # screening entirely from the mode.)
        if explicit is not ComputeMode.X_ONLY and not self.do_screened:
            raise ValueError(
                f"compute_mode={raw!r} requires screening, but the legacy "
                f"flag do_screened=false was also set. Remove one of the two."
            )
        return explicit

    @property
    def qp_solver(self) -> QPSolver:
        """Resolve ``qp_solver`` from explicit input or legacy flags.

        ``qp_solver = auto`` (the default) resolves:

        1. ``self_consistent = true`` → ``SELF_CONSISTENT`` (deprecated
           key, still honored);
        2. else → ``ONE_SHOT_DFT`` — standard G0W0 is the default.
           (The deprecated ``sigma_at_dft_energies = true`` alias also
           lands here: its intended meaning — authoritative at-DFT QP
           evaluation — IS the default.)

        An explicit setting overrides the legacy flags, mirroring how
        ``compute_mode`` absorbs ``do_screened`` / ``use_ppm_sigma``.

        Validation (mutually inconsistent axis combinations):

        - ``fixed_point`` × static mode → error (no ω-grid to solve on;
          a silent no-op would blur the axis).
        """
        raw = (self.qp_solver_raw or "auto").strip().lower()
        if raw == "auto":
            solver = (QPSolver.SELF_CONSISTENT if self.self_consistent
                      else QPSolver.ONE_SHOT_DFT)
        else:
            try:
                solver = QPSolver(raw)
            except ValueError as exc:
                raise ValueError(
                    f"qp_solver={raw!r} invalid; expected one of: "
                    f"{', '.join(s.value for s in QPSolver)}, or 'auto'."
                ) from exc
        mode = self.compute_mode
        if solver is QPSolver.FIXED_POINT and not mode.is_dynamic:
            # The list of dynamic modes is read off the enum, not typed
            # here: the day a mode joins them this message says so without
            # anyone remembering to come back and edit it.  Modes that
            # refuse to run are left out — advice has to be followable.
            _dynamic = " / ".join(
                m.value for m in ComputeMode
                if m.is_dynamic and m not in UNIMPLEMENTED_MODES)
            raise ValueError(
                f"qp_solver=fixed_point requires a dynamic compute_mode "
                f"({_dynamic}); static Σ ({mode.value}) has no ω-grid "
                f"to solve E = h0 + ReΣ(E) on.  Use one_shot_dft (identical "
                f"physics for static Σ) or self_consistent.")
        return solver

    @property
    def minimax_config(self):
        """Math-internal :class:`gw.minimax_config.MinimaxConfig` for χ₀."""
        from .minimax_config import MinimaxConfig
        return MinimaxConfig(
            target_error=self.screening.minimax_target_error,
            max_nodes=self.screening.minimax_max_nodes,
            regenerate_tables=self.screening.regenerate_minimax_tables,
            energy_reference=self.screening.minimax_energy_reference,
        )

    @property
    def sigma_quadrature_config(self):
        """Math-internal :class:`gw.minimax_config.MinimaxConfig` for Σ^c."""
        from .minimax_config import MinimaxConfig
        return MinimaxConfig(
            target_error=self.ppm.sigma_target_error,
            max_nodes=self.ppm.sigma_max_nodes,
            crossing_max_nodes=max(500, self.ppm.sigma_max_nodes),
            crossing_eps_q=1.0e-3,
            regenerate_tables=self.screening.regenerate_minimax_tables,
        )

    @property
    def omega_grid_ev(self):
        """Σ_c(ω) frequency grid in eV (length-stable single formula).

        ``n = floor((max−min)/step + 0.5) + 1`` — the Ry grid is derived
        from this one by division so the two can never disagree in length
        or accumulate independent float-step rounding.

        With ``sigma_omega_patches_ev`` set, the grid is the union of the
        patches, each built by the SAME length-stable formula, ascending
        by the patch validation.  ``sigma_omega_min/max_ev`` are ignored
        then — the patches ARE the grid.
        """
        p = self.sigma
        patches = p.parsed_omega_patches_ev()
        if not patches:
            n = int(np.floor(
                (p.omega_max_ev - p.omega_min_ev) / p.omega_step_ev
                + 0.5)) + 1
            return (p.omega_min_ev
                    + p.omega_step_ev * np.arange(n, dtype=np.float64))
        pieces = []
        for lo, hi in patches:
            n = int(np.floor((hi - lo) / p.omega_step_ev + 0.5)) + 1
            pieces.append(lo + p.omega_step_ev * np.arange(
                n, dtype=np.float64))
        grid = np.concatenate(pieces)
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError(
                "sigma_omega_patches_ev produced a non-increasing grid; "
                "patches must be ascending and disjoint")
        return grid

    @property
    def omega_grid_ry(self):
        """Σ_c(ω) frequency grid in Rydberg (derived from the eV grid)."""
        return self.omega_grid_ev / RYD_TO_EV

    # ------------------------------------------------------------------
    #  Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_input_file(cls, filename: str, *, print_fn=print) -> LorraxConfig:
        """Parse input file and resolve runtime settings (memory, env vars).

        Replaces ``read_cohsex_input`` + ``resolve_runtime_config`` +
        path resolution in one call.  Returns a ``LorraxConfig`` with
        sub-dataclasses fully populated.
        """
        from file_io import resolve_input_paths

        params = read_lorrax_input(filename)
        input_dir = os.path.dirname(os.path.abspath(filename))
        resolve_input_paths(params, input_dir)

        # --- Memory auto-detection ---
        memory_per_device_gb = float(params.get("memory_per_device_gb", 0.0))
        if memory_per_device_gb <= 0:
            from common.gpu_utils import get_device_memory_gb
            memory_per_device_gb = get_device_memory_gb()
            print_fn(
                f"  Auto-detected memory budget: {memory_per_device_gb:.2f} GB/device"
            )

        # --- Chunk utilization from env ---
        # 0.0 (default) = auto: the planner uses its ns²-aware default
        # (higher for scalar, lower for bispinor's 4× pair density).  A
        # positive env value overrides it, clamped to [0.85, 1.0].
        # ``env_float`` announces a non-numeric value instead of swallowing
        # it — the bare ``except Exception`` here left the user believing a
        # utilization was in force when it was not.
        chunk_utilization = env_float("ISDF_CHUNK_TARGET_UTILIZATION", 0.0,
                                      print_fn=print_fn)
        if chunk_utilization > 0:
            chunk_utilization = max(0.85, min(1.0, chunk_utilization))

        def _g(key):
            return params.get(key, _DEFAULTS.get(key))

        # Resolve the bundled metallic q0 contract before constructing any
        # typed group.  ``mc_average_vcoul_body`` defaults to true for every
        # historical deck, but BGW's noavg metal comparison requires false.
        # Only an EXPLICIT contradictory value refuses: an absent key is the
        # compatibility case this bundle exists to override, while an
        # explicit false is already compatible and remains visible in the
        # provenance line.
        _bgw_q0_mode = _normalize_bgw_metal_q0_treatment(
            _g("bgw_metal_q0_treatment"))
        _bgw_q0_vector = _parse_bgw_metal_q0_vector(
            _g("bgw_metal_q0_vector"))
        if _bgw_q0_mode == "bgw_q0shift" and int(_g("sys_dim")) != 3:
            raise ValueError(
                "bgw_metal_q0_treatment = bgw_q0shift is defined only for "
                "3-D metals (sys_dim = 3); this deck sets "
                f"sys_dim = {int(_g('sys_dim'))}.")
        _named_keys = frozenset(params.get(_DECK_NAMED_KEYS, ()))
        _effective_named_keys = set(_named_keys)
        if _bgw_q0_mode == "exact":
            # An explicit spelling of the shipping default must serialize to
            # the same LorraxConfig as an absent key.  ``raw_input_keys`` is
            # otherwise the one field that would distinguish them.
            _effective_named_keys.discard("bgw_metal_q0_treatment")
            if _bgw_q0_vector == _parse_bgw_metal_q0_vector(
                    _DEFAULTS["bgw_metal_q0_vector"]):
                _effective_named_keys.discard("bgw_metal_q0_vector")
        _mc_average_vcoul_body = bool(_g("mc_average_vcoul_body"))
        if _bgw_q0_mode == "bgw_q0shift":
            if ("mc_average_vcoul_body" in _named_keys
                    and _mc_average_vcoul_body):
                raise ValueError(
                    "contradictory deck settings: "
                    "bgw_metal_q0_treatment = bgw_q0shift requires "
                    "mc_average_vcoul_body = false, but the deck explicitly "
                    "sets mc_average_vcoul_body = true. Remove that key or "
                    "set it to false.")
            _mc_origin = (
                "explicit compatible mc_average_vcoul_body=false"
                if "mc_average_vcoul_body" in _named_keys
                else "inherited mc_average_vcoul_body=true")
            print_fn(
                "  [config provenance] bgw_metal_q0_treatment="
                "bgw_q0shift: overriding mc_average_vcoul_body -> false "
                f"({_mc_origin}); q0 reduced vector="
                f"{_bgw_q0_vector}. Analytic-sphere v-head and finite-q0 "
                "W head/wings are enabled; eta/broadening and MPA "
                "quadrature are unchanged.")
            _mc_average_vcoul_body = False

        # --- Build sub-dataclasses ---
        cents_curr = _g("centroids_file_current")
        cents_curr_resolved = str(cents_curr) if cents_curr else None
        paths = FilePaths(
            wfn_file=str(_g("wfn_file")),
            centroids_file=str(_g("centroids_file")),
            centroids_file_current=cents_curr_resolved,
            kin_ion_file=str(_g("kin_ion_file")),
            parallel_transport_file=str(_g("parallel_transport_file")),
            sigma_diag_file=str(_g("sigma_diag_file")),
            eqp0_file=str(_g("eqp0_file")),
            eqp1_file=str(_g("eqp1_file")),
            eqp2_file=str(_g("eqp2_file")),
            report_file=str(_g("report_file")),
            sigma_omega_h5_file=str(_g("sigma_omega_h5_file")),
        )
        _head_correction = coerce_head_correction(_g("head_correction"))
        _legacy_do_g0 = bool(_g("do_G0"))
        if "do_G0" in _named_keys:
            legacy_policy = (
                HeadCorrection.FULL if _legacy_do_g0
                else HeadCorrection.OFF)
            if ("head_correction" in _named_keys
                    and ((_head_correction is HeadCorrection.OFF)
                         != (legacy_policy is HeadCorrection.OFF))):
                raise ValueError(
                    "contradictory deck settings: legacy do_G0 and "
                    "head_correction request opposite Gamma-head policies. "
                    "Remove do_G0 and use head_correction = full | "
                    "no_local_fields | off.")
            if "head_correction" not in _named_keys:
                _head_correction = legacy_policy
                print_fn(
                    "  [config provenance] legacy do_G0 explicitly set: "
                    f"mapping it to head_correction = "
                    f"{_head_correction.value}. Prefer the named policy in "
                    "new decks.")
        _resolved_do_g0 = _head_correction is not HeadCorrection.OFF

        head = HeadConfig(
            correction=_head_correction,
            wcoul0_source=str(_g("wcoul0_source")).strip().lower(),
            wcoul0_eta=float(_g("wcoul0_eta") or 0.0),
            vhead=_g("vhead"),
            whead_0freq=_g("whead_0freq"),
            whead_imfreq=_g("whead_imfreq"),
            mc_average_vcoul_body=_mc_average_vcoul_body,
            bgw_metal_q0_treatment=_bgw_q0_mode,
            bgw_metal_q0_vector=_bgw_q0_vector,
            mc_average_placement=_normalize_placement(
                _g("mc_average_placement")),
            mc_average_placement_vcoul=(
                str(_g("mc_average_placement_vcoul") or "") or None),
            head_minibz_average=bool(_g("head_minibz_average")),
            w_av_first_neighbors=bool(_g("w_av_first_neighbors")),
            w_av_second_neighbors=bool(_g("w_av_second_neighbors")),
            bare_coulomb_cutoff=_g("bare_coulomb_cutoff"),
            zeta_cutoff=_g("zeta_cutoff"),
            use_bgw_vcoul=bool(_g("use_bgw_vcoul")),
            bgw_vcoul_file=(str(_g("bgw_vcoul_file")) or None),
            bgw_vcoul_sym_wfn=(str(_g("bgw_vcoul_sym_wfn")) or None),
        )
        screening = ScreeningConfig(
            method=str(_g("screening_method")).strip().lower(),
            occ_broadening_ev=float(_g("occ_broadening")),
            minimax_target_error=float(_g("minimax_target_error")),
            minimax_max_nodes=int(_g("minimax_max_nodes")),
            regenerate_minimax_tables=bool(_g("regenerate_minimax_tables")),
            minimax_energy_reference=str(_g("minimax_energy_reference")).strip().lower(),
            diagrams=coerce_screening_diagrams(_g("screening_diagrams")),
            ladder_probe_chunk=int(_g("ladder_probe_chunk")),
        )
        ppm = PPMConfig(
            model=str(_g("ppm_model")).strip().lower(),
            omega_p=float(_g("ppm_omega_p")),
            fallback_omega=float(_g("ppm_fallback_omega")),
            head_omega_h_ry=(
                float(_g("ppm_head_omega_h_ry"))
                if _g("ppm_head_omega_h_ry") is not None else None),
            probe_chi_reuse=str(_g("ppm_probe_chi_reuse")).strip().lower(),
            sigma_target_error=float(_g("ppm_sigma_target_error")),
            sigma_max_nodes=int(_g("ppm_sigma_max_nodes")),
            invalid_mode=str(_g("ppm_invalid_mode") or "static_limit").strip().lower(),
        )
        mpa = MPAConfig(
            n_poles=int(_g("mpa_n_poles")),
            material_class=str(_g("mpa_material_class")).strip().lower(),
            sampling_alpha=int(_g("mpa_sampling_alpha")),
            sampling_schedule=str(
                _g("mpa_sampling_schedule")).strip().lower(),
            pole_solver=str(_g("mpa_pole_solver")).strip().lower(),
            varpi_near_ry=float(_g("mpa_varpi_near_ry")),
            varpi_far_ry=float(_g("mpa_varpi_far_ry")),
            metal_origin_shift_ry=(
                float(_g("mpa_metal_origin_shift_ry"))
                if _g("mpa_metal_origin_shift_ry") is not None else None),
            pole_batch_size=int(_g("mpa_pole_batch_size")),
            sigma_sector_target_error=float(
                _g("mpa_sigma_sector_target_error")),
            sigma_crossing_target_error=float(
                _g("mpa_sigma_crossing_target_error")),
            sigma_max_nodes=int(_g("mpa_sigma_max_nodes")),
            sigma_omega_cluster_gap_ry=float(
                _g("mpa_sigma_omega_cluster_gap_ry")),
            occupation_window_threshold=float(
                _g("occupation_window_threshold")),
        )
        _eqp2_enabled = bool(_g("write_eqp2"))
        _sigma_omega_layout = str(
            _g("sigma_omega_layout")).strip().lower()
        if _eqp2_enabled and _sigma_omega_layout != "sharded":
            if "sigma_omega_layout" in _named_keys:
                raise ValueError(
                    "write_eqp2=true requires sigma_omega_layout=sharded; "
                    "the full Sigma(omega,k,m,n) cube is the fixed operand "
                    "of every eqp2 iteration and may not be replicated on "
                    "each rank.  Set sigma_omega_layout=sharded or disable "
                    "write_eqp2.")
            _sigma_omega_layout = "sharded"
            print_fn(
                "  [config provenance] write_eqp2=true: resolving the "
                "unnamed sigma_omega_layout default replicated -> sharded; "
                "the fixed full Sigma(omega) cube stays distributed through "
                "every eigenvalue-consistency iteration.")
        sigma = DynamicSigmaConfig(
            omega_min_ev=float(_g("sigma_omega_min_ev")),
            omega_max_ev=float(_g("sigma_omega_max_ev")),
            omega_step_ev=float(_g("sigma_omega_step_ev")),
            regularization_ev=float(_g("sigma_regularization_ev")),
            window_edge_factor=float(_g("sigma_window_edge_factor")),
            omega_layout=_sigma_omega_layout,
            fermi_reference=str(_g("fermi_reference")).strip().lower(),
            regularization_floor_ev=_g("sigma_regularization_floor_ev"),
            sigma_at_dft_extrapolate=bool(_g("sigma_at_dft_extrapolate")),
            sigma_at_dft_energies=bool(_g("sigma_at_dft_energies")),
            omega_patches_ev=str(_g("sigma_omega_patches_ev")).strip(),
            band_extrapolation_estimator=str(
                _g("band_extrapolation_estimator")
                or BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT).strip().lower(),
            band_extrapolation_bracket_scheme=str(
                _g("band_extrapolation_bracket_scheme")
                or BRACKET_SCHEME_DEFAULT).strip().lower(),
            band_extrapolation_bracket_scheme_explicit=(
                "band_extrapolation_bracket_scheme" in _named_keys),
            **dict(zip(
                ("band_extrapolation", "band_extrapolation_explicit"),
                resolve_band_extrapolation(
                    _g("use_band_extrapolation"),
                    _g("sigma_band_extrapolation"),
                    print_fn=_print_deck_report))),
        )
        # With patches, omega_min/max_ev ARE the patch hull.  Consumers
        # read these fields as "the Σ grid's reach" (the SC partition's
        # in-grid classification above all); leaving them at the deck
        # defaults silently scissored every band outside [-5, +5] on the
        # first patched run — Σ was computed on the deep clusters and
        # then never consulted (measured: arm 21, SC partition 2/48).
        _patches = sigma.parsed_omega_patches_ev()
        if _patches:
            sigma = _dc_replace(
                sigma, omega_min_ev=float(_patches[0][0]),
                omega_max_ev=float(_patches[-1][1]))
        _occ_family = _g("occ_smearing_family")
        _occ_family = (
            str(_occ_family).strip().lower()
            if _occ_family is not None else None)
        _occ_width = _g("occ_smearing_width_ry")
        _occ_width = float(_occ_width) if _occ_width is not None else None
        _validate_occupation_smearing(
            mpa, sigma, screening, _occ_family, _occ_width)
        # SC loop knobs.  The LORRAX_SC_* env vars are deprecated overrides
        # of the sc_* input keys (kept so existing sweep scripts run
        # unchanged); a note is printed whenever one is active.
        def _sc_env(env_key: str, cast, file_val, input_key: str):
            raw_env = os.environ.get(env_key)
            if raw_env is None or raw_env == "":
                return file_val
            val = cast(raw_env)
            print_fn(
                f"  [config] {env_key}={raw_env} (deprecated env override; "
                f"set '{input_key} = {raw_env}' in cohsex.in instead)")
            return val

        sc = SCConfig(
            max_iter=_sc_env(
                "LORRAX_SC_MAX_ITER", int, int(_g("sc_max_iter")),
                "sc_max_iter"),
            tol_ev=_sc_env(
                "LORRAX_SC_TOL_EV", float, float(_g("sc_tol_ev")),
                "sc_tol_ev"),
            accelerator=_sc_env(
                "LORRAX_SC_ACCEL", lambda s: str(s).strip().lower(),
                str(_g("sc_accelerator")).strip().lower(), "sc_accelerator"),
            history_depth=_sc_env(
                "LORRAX_SC_DEPTH", int, int(_g("sc_history_depth")),
                "sc_history_depth"),
            mixing=_sc_env(
                "LORRAX_SC_MIXING", float, float(_g("sc_mixing")),
                "sc_mixing"),
            dump_dir=_sc_env(
                "LORRAX_SC_DUMP_DIR", str, str(_g("sc_dump_dir") or ""),
                "sc_dump_dir") or None,
            # No env override: the LORRAX_SC_* envs are deprecated and a
            # new knob must not add one.
            eigh=str(_g("sc_eigh")).strip().lower(),
            head_update=str(_g("sc_head_update")).strip().lower(),
        )
        eqp2 = EQP2Config(
            enabled=_eqp2_enabled,
            tol_ev=float(_g("eqp2_tol_ev")),
            max_iter=int(_g("eqp2_max_iter")),
            accelerator=str(_g("eqp2_accelerator")).strip().lower(),
            history_depth=int(_g("eqp2_history_depth")),
        )
        memory = MemoryConfig(
            per_device_gb=memory_per_device_gb,
            chunk_target_utilization=chunk_utilization,
            band_chunk_size=int(_g("band_chunk_size")),
            r_chunk_override=int(_g("r_chunk_size")),
            gflat_chunk_size=int(_g("gflat_chunk_size")),
            vq_g_chunk_size=int(_g("vq_g_chunk_size")),
            low_mem_bands=bool(_g("low_mem_bands")),
        )
        # SlabIO routing + auto-route GPU FFIs off on the CPU backend.
        # cuSOLVERMp / cuBLASMp are GPU-only.  The phdf5 FFI is NOT: both
        # its read and its write core compile CUDA-free into
        # liblorrax_ffi_host.so (``LORRAX_FFI_NO_CUDA``; see
        # ``ffi/cpp/phdf5/platform_seam.h``), so on CPU it is preferred
        # whenever the deployed lib exports the handler.
        #
        #   * the sharded-slab transport is NOT selected here any
        #     more.  There is one, it is capability-checked at the file
        #     open by ``file_io.slab_io.assert_available``, and a stack
        #     that cannot serve it refuses there naming the probe that
        #     declined.  Removing the deck key removed the router, the
        #     three tiers and seven refusals with it (2026-08-06).
        #   * on CPU, explicit ``cusolvermp`` is
        #     REFUSED at parse time (CUDA-only backend; doctrine 3);
        #     ``distributed_lu = auto`` demotes to ``"off"`` (in-tree
        #     per-q ``jnp.linalg.solve``) with an announcement.
        #
        # User-facing: same ``cohsex.in`` works on both backends.
        # Distributed-linalg axes.
        _dist_chol = str(_g("distributed_cholesky")).strip().lower()
        _dist_lu = str(_g("distributed_lu")).strip().lower()
        _distrib_la_batched_route = resolve_distrib_la_batched_route(params)
        if _dist_chol not in ("auto", "off", "cusolvermp", "slate"):
            raise ValueError(
                f"distributed_cholesky={_dist_chol!r} invalid; expected "
                f"auto / off / cusolvermp / slate.")
        if _dist_lu not in ("auto", "off", "cusolvermp", "scalapack"):
            raise ValueError(
                f"distributed_lu={_dist_lu!r} invalid; expected auto / off "
                f"/ cusolvermp / scalapack (a SLATE getrf wrapper does not "
                f"exist yet; scalapack is the host/CPU-backend option).")
        # eigh_backend + use_low_mem_eigh are ONE axis; ``resolve_eigh_backend``
        # is the single place they combine (the raw-params drivers call the
        # same function).  It also owns the vocabulary check, read from
        # distrib_la so parser and dispatcher cannot drift.
        _use_low_mem_eigh = bool(_g("use_low_mem_eigh"))
        _eigh_backend = resolve_eigh_backend({
            "eigh_backend": _g("eigh_backend"),
            "use_low_mem_eigh": _use_low_mem_eigh,
        })
        # No CPU rewriting for eigh_backend: an explicit FFI request keeps
        # the fails-loudly semantics — distrib_la.resolve_backend rejects
        # cusolvermp on a CPU mesh (and a slate-less build) at resolve time.
        _charge_zeta_solve = str(_g("charge_zeta_solve")).strip().lower()
        if _charge_zeta_solve not in ("rank_truncate", "cholesky"):
            raise ValueError(
                f"charge_zeta_solve={_charge_zeta_solve!r} invalid; expected "
                f"rank_truncate / cholesky.")
        # Normalised to one of the TWO plans at PARSE time (fails loudly
        # here on removed spellings, not 20 minutes into the run).
        _w_dyson_solver = normalize_w_dyson_solver(_g("w_dyson_solver"))
        _dist_zeta_solve = str(_g("distributed_zeta_solve")).strip().lower()
        if _dist_zeta_solve not in (
                "auto", "replicated", "per_q", "distributed"):
            raise ValueError(
                f"distributed_zeta_solve={_dist_zeta_solve!r} invalid; "
                f"expected auto / replicated / per_q / distributed.")
        _transverse_zeta_solve = str(
            _g("transverse_zeta_solve")).strip().lower()
        if _transverse_zeta_solve not in ("ridge", "rank_truncate"):
            raise ValueError(
                f"transverse_zeta_solve={_transverse_zeta_solve!r} invalid; "
                f"expected ridge / rank_truncate.")
        if (_transverse_zeta_solve == "rank_truncate"
                and _dist_lu in ("scalapack", "cusolvermp", "on")):
            # Same refusal isdf/core._resolve_solver_kind_transverse makes
            # at resolve time, surfaced at PARSE time so a doomed bispinor
            # deck refuses in seconds, not after the charge fit.
            raise ValueError(
                f"transverse_zeta_solve=rank_truncate selects the eigh "
                f"pseudo-inverse family (distributed plan via "
                f"distributed_zeta_solve=distributed), but "
                f"distributed_lu={_dist_lu!r} explicitly requests an LU "
                f"backend that family does not run.  Leave distributed_lu "
                f"at auto/off, or keep transverse_zeta_solve=ridge.")
        _transverse_zeta_rcond = float(_g("transverse_zeta_rcond"))
        if not (0.0 < _transverse_zeta_rcond < 1.0):
            raise ValueError(
                f"transverse_zeta_rcond={_transverse_zeta_rcond!r} must be "
                f"a relative cutoff in (0, 1).")
        try:
            import jax as _jax
            _is_cpu_backend = _jax.default_backend() == "cpu"
        except Exception:
            _is_cpu_backend = False
        if _is_cpu_backend:
            # Doctrine 3 (audit fix/zq 2026-07-28): an EXPLICIT
            # ``cusolvermp`` on a CPU JAX backend REFUSES at parse time —
            # matching the scalapack-on-GPU refusal below and
            # eigh_backend's fails-loudly contract — instead of being
            # rewritten to 'off' (which silently ran a different solver
            # than the input file names).  Only 'auto' may demote, with an
            # announcement.
            #
            # slate / scalapack pass through: host-platform FFIs
            # (liblorrax_ffi_host.so) with explicit-request-fails-loudly
            # semantics at their own resolve/call sites.
            #
            # ``distributed_cholesky = auto`` ALSO passes through.  It
            # used to be forced to 'off' here, but 'off' is an *override*
            # that short-circuits the whole route policy in isdf/core.py
            # straight to ``sharded_cholesky`` -- and the replicated route
            # it thereby skips is the ONLY one that carries the charge
            # ζ-solve rank-truncation cure
            # (charge_zeta_solve='rank_truncate').  That route is
            # replicated dense JAX with no FFI, so it is perfectly valid
            # on CPU; only the ABOVE-cap cuSOLVERMp branch is CUDA-only,
            # and isdf/core.py declines that on a CPU mesh.  Forcing 'off'
            # here silently produced a non-rank-conditioned ζ on CPU
            # (garbage V_q -> inverted QP gap) with no warning.
            if _dist_chol == "cusolvermp":
                raise ValueError(
                    "distributed_cholesky = cusolvermp is CUDA-only but "
                    "the JAX backend is CPU; use distributed_cholesky = "
                    "auto|off|slate on CPU runs (auto keeps the replicated "
                    "rank-truncation route; slate is the host FFI).")
            if _dist_lu == "cusolvermp":
                raise ValueError(
                    "distributed_lu = cusolvermp is CUDA-only but the JAX "
                    "backend is CPU; use distributed_lu = "
                    "auto|off|scalapack on CPU runs (scalapack is the "
                    "ScaLAPACK host FFI).")
            if _dist_lu == "auto":
                # 'auto' demote, announced: auto never picks an FFI LU on
                # a CPU backend (cuSOLVERMp is CUDA-only and auto never
                # selects ScaLAPACK), so 'off' (in-tree per-q
                # jnp.linalg.solve) is the same route auto would resolve
                # to — made explicit here, and said out loud.
                print_fn(
                    "  [config] distributed_lu=auto on CPU backend: auto "
                    "never picks an FFI LU here (cuSOLVERMp is CUDA-only; "
                    "ScaLAPACK is explicit-only).  Demoting to 'off' "
                    "(in-tree per-q jnp.linalg.solve).  The ScaLAPACK "
                    "host FFI is available via explicit "
                    "distributed_lu = scalapack."
                )
                _dist_lu = "off"
        elif _dist_lu == "scalapack":
            # Host-only backend on a non-CPU JAX backend: reject at parse
            # time — the alternative is a ValueError hours later at the
            # first transverse solve, after the C_q build.
            raise ValueError(
                "distributed_lu=scalapack is host-only (Cray LibSci) but "
                "the JAX backend is not CPU; use distributed_lu = "
                "auto|off|cusolvermp on GPU runs.")
        backend = BackendConfig(
            w_dyson_solver=_w_dyson_solver,
            distributed_cholesky=_dist_chol,
            distributed_lu=_dist_lu,
            distrib_la_batched_route=_distrib_la_batched_route,
            eigh_backend=_eigh_backend,
            use_low_mem_eigh=_use_low_mem_eigh,
            zeta_ridge=float(_g("zeta_ridge")),
            charge_zeta_solve=_charge_zeta_solve,
            distributed_zeta_solve=_dist_zeta_solve,
            zeta_rcond=float(_g("zeta_rcond")),
            transverse_zeta_solve=_transverse_zeta_solve,
            transverse_zeta_rcond=_transverse_zeta_rcond,
            gamma_contract_mode=str(_g("gamma_contract_mode")).strip().lower(),
        )
        # Validate the V_H source at PARSE time, not at the read that
        # would otherwise fail 20 minutes into a 40-node run.
        from file_io.kin_ion import HARTREE_SOURCES
        _hartree_source = str(_g("hartree_source") or "auto").strip().lower()
        if _hartree_source not in HARTREE_SOURCES:
            raise ValueError(
                f"hartree_source={_hartree_source!r} is not one of "
                f"{HARTREE_SOURCES}.  H0 = kin_ion + V_H is a ~500 eV "
                "cancellation; this key is not guessed.")
        # Same treatment, same reason, for the restart q-set.  Validated
        # here and NOT resolved here: ``auto`` resolves against the closure
        # answer, which needs the run's centroid set and its symmetry
        # tables, so the field below is the RAW request and
        # ``gw.restart_q_storage.resolve_restart_q_storage`` turns it into a
        # mode once those exist.  (``hartree_source`` can be stored resolved
        # because its ``auto`` resolves against a file already on disk; this
        # one cannot, and the ``_raw`` suffix says which kind it is — the
        # same convention ``compute_mode_raw`` / ``qp_solver_raw`` use.)
        from gw.restart_q_storage import RESTART_Q_STORAGE
        # The ``or`` fallback must agree with ``_DEFAULTS`` — it is reached
        # only by a caller that built the params dict by hand and left the
        # key out, and a fallback that disagreed with the registered default
        # would make THAT caller silently take a different storage decision.
        _restart_q_storage = str(
            _g("restart_q_storage") or "auto").strip().lower()
        if _restart_q_storage not in RESTART_Q_STORAGE:
            raise ValueError(
                f"restart_q_storage={_restart_q_storage!r} is not one of "
                f"{RESTART_Q_STORAGE}.  This key selects the q-set the "
                "restart tensors are STORED on; a value nobody recognises "
                "is not silently read as the default.")

        from file_io.qp_wfn import QP_ROTATIONS_K_STORAGE
        # Same ``or`` caveat as above: this fallback is reached only by a
        # hand-built params dict and must agree with ``_DEFAULTS``.
        _qp_rot_k_storage = str(
            _g("qp_rotations_k_storage") or "auto").strip().lower()
        if _qp_rot_k_storage not in QP_ROTATIONS_K_STORAGE:
            raise ValueError(
                f"qp_rotations_k_storage={_qp_rot_k_storage!r} is not one "
                f"of {QP_ROTATIONS_K_STORAGE}.  This key selects the k-set "
                "qp_wfn_rotations.h5 is STORED on; a value nobody "
                "recognises is not silently read as the default.")

        debug = DebugConfig(
            sigma_freq_debug_output=bool(_g("sigma_freq_debug_output")),
            sigma_freq_debug_file=str(_g("sigma_freq_debug_file")),
            write_wfn_h5=bool(_g("write_wfn_h5")),
        )
        bse = BSEConfig(
            get_centroids_fi=bool(_g("get_centroids_fi")),
            wfn_fi_min=int(_g("wfn_fi_min")),
            wfn_fi_max=int(_g("wfn_fi_max")),
            kgrid_fi=str(_g("kgrid_fi") or ""),
            wfn_fi_q_chunk=int(_g("wfn_fi_q_chunk")),
        )
        if bse.wfn_fi_q_chunk < 0:
            raise ValueError(
                f"wfn_fi_q_chunk={bse.wfn_fi_q_chunk} invalid; expected >= 1, "
                f"or 0 for the default (= N_q_co, the coarse k-point count).")

        # BAND COUNTS.  ``read_lorrax_input`` already resolved them (once) and
        # left the answer in the params dict; a hand-made dict that never went
        # through the parser gets resolved here instead.  Either way there is
        # exactly one ``resolve_band_counts`` call per config.
        _bands = params.get(_BAND_COUNTS)
        if not isinstance(_bands, BandCounts):
            _bands = resolve_band_counts(params)

        # ζ-fit window top.  Empty / unset collapse to None — "follow the
        # loaded window".  An EXPLICIT value is stored verbatim, INCLUDING one
        # that equals ``bands.isdf``.
        #
        # WHY IT IS NO LONGER ERASED HERE (2026-08-22).  This used to rewrite
        # ``zeta_nband == bands.isdf`` to None, reasoning that a redundant
        # restatement of the default must take the default path "pad and all".
        # It is not redundant, because ``bands.isdf`` is the LOGICAL count and
        # the edge the fit actually gets is ``BandSlices.b4`` — that count
        # ROUNDED UP to the world size.  On P=4 a scalar-Si deck with
        # ``nband = zeta_nband = 14`` silently fitted [0,16) and then refused,
        # correctly, because band 16 cuts a multiplet; the deck had asked for
        # 14 and no banner ever said otherwise (JID 57152792,
        # runs/Si_scalar/11_scalar_v_rootcause_20260817/).
        #
        # The collapse still exists — it just happens where the padded edge is
        # known, in ``gw.gw_init.resolve_zeta_fit_edge``, which is also the one
        # place the banner and the three fit-window consumers read.  A deck
        # whose ``nband`` already divides the world size is unchanged.
        _zeta_nband_raw = _g("zeta_nband")
        if _zeta_nband_raw in (None, ""):
            _zeta_nband = None
        else:
            _zeta_nband = int(_zeta_nband_raw)
            if _zeta_nband < 1 or _zeta_nband > _bands.isdf:
                raise ValueError(
                    f"zeta_nband={_zeta_nband} must be in [1, {_bands.isdf}] "
                    f"— the ISDF fit's band-window top, which is "
                    f"max(number_bands_chi={_bands.chi}, "
                    f"number_bands_sigma={_bands.sigma}).  zeta_nband NARROWS "
                    f"the band window ζ is fitted on; it cannot widen it, "
                    f"because the centroid ψ this run loads spans [b0, b4) "
                    f"and there are no bands above b4 to fit.  Raise "
                    f"number_bands (or whichever of number_bands_chi / "
                    f"number_bands_sigma is the larger) if you want more "
                    f"bands in the fit AND in the band sum that owns them.")

        resolved = cls(
            # Top-level: system + mode flags
            nval=int(_g("nval")),
            ncond=int(_g("ncond")),
            nband=int(_bands.isdf),
            bands=_bands,
            zeta_nband=_zeta_nband,
            sys_dim=int(_g("sys_dim")),
            density_self_consistent=bool(_g("density_self_consistent")),
            sc_on_ibz=bool(_g("sc_on_ibz")),
            hartree_source=_hartree_source,
            occ_smearing_family=_occ_family,
            occ_smearing_width_ry=_occ_width,
            occupation_clamp_tol=float(_g("occupation_clamp_tol")),
            restart=bool(_g("restart")),
            write_restart_tensors=bool(_g("write_restart_tensors")),
            write_qsgw_datasets=bool(_g("write_qsgw_datasets")),
            restart_q_storage_raw=_restart_q_storage,
            qp_rotations_k_storage=_qp_rot_k_storage,
            # Build from a stable sequence.  Equal sets reached through an
            # absent key versus an explicit default can retain different
            # hash-table histories; pickling those frozensets then need not
            # be byte-identical even though the typed values compare equal.
            raw_input_keys=frozenset(sorted(_effective_named_keys)),
            compute_mode_raw=str(_g("compute_mode") or "auto").strip().lower(),
            qp_solver_raw=str(_g("qp_solver") or "auto").strip().lower(),
            do_screened=bool(_g("do_screened")),
            bispinor=bool(_g("bispinor")),
            # Compatibility mirror only.  Every new head decision reads the
            # enum above; keeping this resolved bool prevents old consumers
            # from disagreeing with ``head_correction = off``.
            do_G0=_resolved_do_g0,
            self_consistent=bool(_g("self_consistent")),
            use_ppm_sigma=bool(_g("use_ppm_sigma")),
            no_degen_averaging=bool(_g("no_degen_averaging")),
            degen_avg_tol_ry=float(_g("degen_avg_tol_ry")),
            # Sub-dataclass groups
            paths=paths,
            head=head,
            screening=screening,
            sigma=sigma,
            ppm=ppm,
            mpa=mpa,
            sc=sc,
            eqp2=eqp2,
            memory=memory,
            backend=backend,
            debug=debug,
            bse=bse,
            # Parsed blocks
            kpoints_crystal_b=params.get("kpoints_crystal_b"),
            input_dir=input_dir,
            input_file=os.path.abspath(filename),
        )
        # CROSS-KEY, and therefore after the record exists: the w_bse
        # refusals read resolved axes (compute_mode / qp_solver fold in the
        # legacy flags), and the honest way to ask which mode a deck chose
        # is to ask the resolver, not to re-derive it here.  A w_rpa deck
        # returns from this call before either property is touched.
        refuse_unsupported_bgw_metal_q0_treatment(resolved)
        refuse_unsupported_screening_diagrams(resolved)
        # Same position/reason as the two calls above: low_mem_bands=false
        # (the default) returns before any predicate is touched, so this
        # adds no new resolution to a default deck.  See
        # refuse_unsupported_low_mem_bands's docstring for why this call
        # (parser altitude) and the mirrored call in
        # gw.gw_init.prepare_isdf_and_wavefunctions (driver-entry altitude,
        # for hand-built configs) both exist.
        refuse_unsupported_low_mem_bands(resolved)
        # ONE CANONICAL VOCABULARY FOR THE SELF-ENERGY AXIS, and a note for
        # the other one.  Same position and same reason as the two refusals
        # above: the announcement quotes the RESOLVED axes, which only the
        # record can answer.  Honoring a legacy key in silence beside a
        # canonical twin is how a tree ends up with two vocabularies for one
        # axis and no way to tell which one a run went through.
        announce_legacy_sigma_axis_keys(
            _named_keys, resolved.compute_mode, resolved.qp_solver,
            print_fn=print_fn)
        return resolved
