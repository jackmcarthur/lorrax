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
    config.backend     — FFI/IO backend selection (gspace_io / w_dyson_solver)
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.units import RYD_TO_EV


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

_ENV_TRUE = ("1", "on", "true", "yes")
_ENV_FALSE = ("0", "off", "false", "no")

#: (name, raw value) pairs already announced, so a knob read once per
#: r-chunk cannot spam a multi-hour log.
_ENV_ANNOUNCED: set = set()


def reset_env_announce_state() -> None:
    """Forget which grammar errors have been announced (tests only)."""
    _ENV_ANNOUNCED.clear()


def env_bool(name: str, default: bool, *, print_fn=print) -> bool:
    """Canonical boolean env parse for the GW init/config layer.

    Parameters
    ----------
    name : str
        Environment variable, e.g. ``"LORRAX_MEM_DEBUG"``.
    default : bool
        Value when the variable is unset or blank.  This is the knob's
        DOCUMENTED default, not a guess — ``docs/dev/env_vars.md`` is the
        table it has to agree with.

    Notes
    -----
    The three bugs this replaces, all in the four files this layer owns:

    * ``if os.environ.get("LORRAX_EXIT_AFTER_ZETA"):`` — a bare presence
      test, so ``=0`` ended a production run with ``SystemExit(0)``;
    * ``not in ("0", "off", "false")`` — case-SENSITIVE, so
      ``LORRAX_MALLOC_TRIM=OFF`` left the trim hook on while the
      documented sibling ``LORRAX_MALLOC_TUNE=OFF`` correctly turned off;
    * ``bool(os.environ.get(...))`` — same presence-test class.

    An unrecognised token resolves False (see the module comment) but is
    ANNOUNCED, once per (name, value), with the project's grep-able
    ``*** LORRAX SANITY`` marker.  Silently resolving a typo in either
    direction is the failure mode this whole helper exists to remove.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    tok = raw.strip().lower()
    if tok in _ENV_TRUE:
        return True
    if tok in _ENV_FALSE:
        return False
    key = (name, raw)
    if key not in _ENV_ANNOUNCED:
        _ENV_ANNOUNCED.add(key)
        print_fn(
            f"  *** LORRAX SANITY: {name}={raw!r} is not a recognised "
            f"boolean.  Accepted: {'/'.join(_ENV_TRUE)} (on), "
            f"{'/'.join(_ENV_FALSE)} (off), unset/blank (default="
            f"{'on' if default else 'off'}).  Treating it as OFF. ***")
    return False


def env_float(name: str, default: float, *, print_fn=print,
              refuse: bool = False) -> float:
    """Canonical numeric env parse: unset/blank → default, bad → ANNOUNCE
    (or, with ``refuse=True``, RAISE).

    The same defect class as :func:`env_bool`, one type along.  A
    ``try: float(...) except: default`` leaves the user believing a knob is
    in force when it is not — the exact failure this file's
    ``ISDF_ZCT_STAGE_CAP_GB`` handler already carries a comment about
    ("an OOM later, with no clue"), and which its sibling
    ``ISDF_CHUNK_TARGET_UTILIZATION`` was still committing.

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


def resolve_zct_stage_cap(cap_raw, frac_raw, *, per_device_gb: float,
                          total_gb: float, print_fn=print):
    """Resolve the ζCᵀ stage cap in GB, or ``None`` when no cap applies.

    Pure — the caller supplies ``total_gb`` (the physical card, ``0.0``
    when there is no device to take a fraction of, i.e. the CPU backend).

    Every path that ends in "no cap" says why.  The CPU path used to be
    silent: the fraction branch sat behind
    ``jax.default_backend() in ("gpu", "cuda")``, so a user who exported
    ``ISDF_ZCT_STAGE_CAP_FRAC`` on a CPU run got no cap, no clamp and no
    message — indistinguishable from the knob working.
    """
    if cap_raw and str(cap_raw).strip():
        try:
            return min(float(per_device_gb),
                       max(0.0, float(cap_raw)))
        except ValueError:
            # Deliberately does NOT fall through to the _FRAC branch: an
            # explicit absolute cap that cannot be parsed is a user error,
            # and quietly substituting a different knob's answer for it
            # would hide the typo behind a plausible number.
            print_fn(f"  *** LORRAX SANITY: ISDF_ZCT_STAGE_CAP_GB="
                     f"{cap_raw!r} is not a number; NO stage cap is in "
                     f"force (ISDF_ZCT_STAGE_CAP_FRAC is not consulted as "
                     f"a fallback for a malformed explicit cap). ***")
            return None
    if not (frac_raw and str(frac_raw).strip()):
        return None
    if float(total_gb) <= 0:
        print_fn(f"  *** LORRAX SANITY: ISDF_ZCT_STAGE_CAP_FRAC="
                 f"{frac_raw!r} is a fraction of the DEVICE's physical "
                 f"memory, and this backend reports none (total_gb=0) — "
                 f"so NO stage cap is in force.  Use "
                 f"ISDF_ZCT_STAGE_CAP_GB for an absolute cap. ***")
        return None
    try:
        frac = max(0.10, min(0.95, float(frac_raw)))
    except ValueError:
        print_fn(f"  *** LORRAX SANITY: ISDF_ZCT_STAGE_CAP_FRAC="
                 f"{frac_raw!r} is not a number; the stage cap is "
                 f"NOT set. ***")
        return None
    return min(float(per_device_gb), frac * float(total_gb))


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
# REMOVING A ROW IS THE LANDING GESTURE.  When a mode's Σ stage lands, its
# author deletes the entry and the suite that pins the refusal fails loudly
# until it is rewritten to pin the new behaviour.
#
# THE DICT IS EMPTY, AND THAT IS A STATE AND NOT AN OMISSION.  ``mpa`` was
# the last row and it came out when the driver seam landed: the fit stage,
# the Σ pass loop (``gw.mpa.sigma_pass.compute_mpa_sigma_c_omega_grid``,
# gated against the fused all-pole closed form at 6.8e-11), the q → 0 head
# (``gw.mpa.sigma_head``) and the pipeline that sequences them
# (``gw.mpa_pipeline``) are all in the tree, and ``sigma_dispatch`` has a
# branch for the mode.
#
# WHAT USED TO BE HERE IS NOW REFUSED WHERE IT IS MET, which is the whole
# reason the row could come out.  The two properties of the STORE that
# blocked the mode are still refused BY NAME, by the code that reads each
# of them: a fit store fitted on a DIFFERENT zone than this run's cannot be
# summed by a kernel whose k-q sums index this one
# (``sigma_pass.resolve_pole_q_axis``; a wedge OF this zone is served to
# the full BZ by the certified pole unfold and is no longer refused), and
# a store
# with no ``__mpahead`` axis has no q → 0 head
# (``mpa_store.read_head_poles``), which is not a correction that can be
# omitted and noticed later.  A deck that names no store at all is refused
# by :func:`refuse_missing_mpa_fit_store` at driver entry.  Those are
# refusals about a FILE, which is why they belong at the file and not on
# an axis of modes.
#
# Keeping the dict (and every site that consults it) after it emptied is
# deliberate: the machinery is what makes the next mode's landing a
# one-line edit instead of a re-derivation, and an empty dict is the
# cheapest possible fast-path check.
UNIMPLEMENTED_MODES: dict[ComputeMode, str] = {}


#: The energy units a multipole fit store's pole axis can be stated in.
#: The FACTORS live in ``gw.mpa.sigma_head.POLE_ENERGY_UNITS``, which is
#: the module that applies them; this is only the parser's legal set, kept
#: here so a deck typo is caught at parse time without the config layer
#: importing the Σ stage.  ``test_mpa_sigma_head`` asserts the two agree.
POLE_ENERGY_UNIT_SPELLINGS = ("Ry", "Ha")


def coerce_pole_energy_unit(value) -> str:
    """Normalise ``mpa_pole_energy_unit``; refuse an unknown spelling.

    Case-insensitive on input and canonical on output, so a deck may
    write ``ry`` / ``HA`` and every consumer downstream compares against
    one spelling.  A typo raises rather than defaulting: the default is
    the thing being chosen, and silently falling back to it would scale
    every head pole by two with no other symptom.
    """
    raw = str(value if value is not None else "Ry").strip()
    for known in POLE_ENERGY_UNIT_SPELLINGS:
        if raw.lower() == known.lower():
            return known
    raise ValueError(
        f"mpa_pole_energy_unit={value!r} is not a known energy unit; "
        f"expected one of: {', '.join(POLE_ENERGY_UNIT_SPELLINGS)}.")


def refuse_missing_mpa_fit_store(config, *, context: str = "this run") -> str:
    """The MPA fit store path, or a refusal that names the missing key.

    ``compute_mode = mpa`` reads its screening from a staged (B_p, Ω_p)
    store and from nowhere else — :func:`gw.screening.screening_requests_for`
    returns NO requests for it, so there is no W anywhere in the run to
    fall back to.  A deck that selects the mode and names no store is
    therefore not under-specified in a way the driver could paper over; it
    has asked for a screened self-energy and supplied no screening.

    Returns the raw (already input-dir-resolved) path for every other
    mode's benefit too, so a caller need not branch on the mode to ask.
    """
    resolved = coerce_compute_mode(config.compute_mode)
    path = getattr(config.paths, "mpa_fit_file", None)
    if resolved is not ComputeMode.MPA:
        return path
    if path:
        return path
    raise ValueError(
        f"compute_mode = mpa but the deck names no 'mpa_fit_file', so "
        f"{context} has no screening at all: the multipole Σ_c is built "
        f"from the complex poles (B_p, Ω_p) in a staged fit store, and "
        f"screening_requests_for(mpa) deliberately asks the screening "
        f"stage for nothing because that store already holds the answer.  "
        f"Set mpa_fit_file to the store written by the MPA fit driver "
        f"(file_io.mpa_store); it must carry both a full-BZ pole axis and "
        f"a __mpahead head axis, and the readers refuse it by name when it "
        f"does not.")


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
      Σ has no ω-grid to solve on.  ``ppm.sigma_at_dft_extrapolate`` is a
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



class GspaceIO(str, enum.Enum):
    """How ψ(G) is moved into the ISDF r-chunk loop.

    Both modes keep ψ(G) on host in per-rank band-sharded layout and
    pull one band-chunk at a time into the jit via io_callback — never
    more than one bc on device at a time.

    - ``HOST_CACHE`` — read ψ(G) once at startup, keep resident in host
      RAM for the full run.  Default; fastest.
    - ``FILE_REREAD`` — rebuild the host buffer at each r-chunk via
      phdf5 collective read; drop between r-chunks.  Zero persistent
      host residency (needed for huge systems where host RAM can't
      hold ψ(G)).
    """
    HOST_CACHE = "host_cache"
    FILE_REREAD = "file_reread"


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

_DEFAULTS = {
    # System geometry
    "nval": 5,
    "ncond": 5,
    "nband": 100,
    "sys_dim": 2,
    # Rebuild V_H from the CURRENT orbitals each self-consistent iteration
    # instead of rotating the fixed DFT one into the QP basis.  Off keeps
    # QSGW fixed-density, which is what every result before 2026-08-04 was.
    "density_self_consistent": False,
    # Run the SC loop's H / E / U on the IBZ, broadcasting back at the
    # boundary.  Sigma stays on the full BZ -- it is an FFT over the
    # k-grid.  Off keeps the loop entirely full-BZ.
    "sc_on_ibz": False,
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
    # The staged multipole (B_p, Ω_p) fit store — ``compute_mode = mpa``
    # reads its ENTIRE screening from this file and asks the screening
    # stage for nothing, so a deck that selects that mode without naming
    # one is refused by ``refuse_missing_mpa_fit_store`` rather than run
    # against the bare Coulomb.  Empty string == "not set"; resolved
    # relative to the deck's directory like every other path key.  Written
    # by ``gw.mpa.fit_driver`` / ``file_io.mpa_store``; the file must carry
    # BOTH a full-BZ pole axis and a ``__mpahead`` head axis, and the two
    # readers refuse it by name when it does not.
    "mpa_fit_file": "",
    # What energy unit a LEGACY head set's poles are stated in.  SCOPE
    # NARROWED 2026-08-09: the store now declares its own units — the
    # BODY pole axis via ``mpa_fit_energy_unit`` (required at allocate;
    # the fit driver inherits it from the W store's ``mpa_omega_units``
    # stamp, and the body readers refuse an undeclared store by name —
    # ``mpa_store.declare_fit_energy_unit`` is the one-time migration
    # stamp), and a head set via its own attr when written with
    # ``energy_unit=``.  A DECLARED head set comes back from
    # ``read_head_poles`` already converted to Ry and this key is not
    # consulted for it.  What remains this key's to say is the unit of a
    # head set written BEFORE the declaration existed — the head has a
    # working caller-owned conversion (``gw.mpa.sigma_head``) that
    # predates the stamp, so legacy sets read rather than refuse, and
    # this key is their unit.  The first-light heads were fitted on the
    # W store's Hartree abscissae, so decks reading them say "Ha";
    # both pole arrays scale by the same factor (the residue carries one
    # power of energy in the numerator); nothing guesses.
    "mpa_pole_energy_unit": "Ry",
    # WHICH q → 0 HEAD SET, when the store carries more than one.  A fit
    # store's ``__mpahead`` group may be joined by ``__mpahead__<label>``
    # siblings (``mpa_store.head_group_name``), and the reason they exist
    # is an OPEN OWNER DECISION: the sign of the nonlocal velocity
    # commutator in ``common.mtxel_sweep.dipole_operator`` moves the head
    # pole by ~3 eV, so both conventions are fitted into ONE store as two
    # named sets and the comparison table carries both columns without two
    # 23 GB stores.  ``as_shipped`` is the bare group, which is also what a
    # store written before labels existed holds — so the default reads
    # every store there is.  The RESOLVED label is announced on the run's
    # Σ head line: a head number that cannot be attributed to a convention
    # is precisely what this arrangement exists to prevent.
    "mpa_head_label": "as_shipped",
    # SPREADING THE POLE PASSES OVER DEVICES.  The pass loop is a sum
    # over the store's poles, one term per pass, and on a production
    # field one term costs hours — so the three keys below let one
    # process walk SOME of the poles and write its partial cube, and a
    # later process sum the partials.  They change no physics and no
    # quadrature: the recombination happens before the q→0 head, before
    # the at-DFT interpolation and before the writer, so steps 3–5 are
    # reached by the same object whether the run was split or whole.
    #
    # ``mpa_pole_subset`` — comma-separated pole indices this process
    # integrates ("" = every pole, the single-process production route).
    # The poles are walked in the store's PINNED ascending order however
    # they are typed; the key names which, never in what order.
    # ``mpa_pass_partial_out`` — where this process writes its partial
    # Σ_c cube.  A run with this set announces, loudly, that its eqp
    # numbers are computed from a partial sum and are not a self-energy.
    # ``mpa_pass_partial_in`` — a file, glob or directory of partial
    # cubes to SUM, in the pinned ascending order, instead of
    # integrating anything.  The combiner refuses a stack that does not
    # cover the store's pass order exactly once, because a missing or
    # doubled pole returns a finite, smooth, plausible Σ that differs by
    # tens of meV — the size of the effect these runs measure.
    "mpa_pole_subset": "",
    "mpa_pass_partial_out": "",
    "mpa_pass_partial_in": "",
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
    # Three human-readable text outputs (always written):
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
    # DEFAULT full — the q_irr wedge is OPT-IN PER DECK, which is what
    # DESIGN_symmetry_restart_followup.md ruled and what this key was built
    # to obey: "the deck key keeps full-BZ storage as the default until the
    # owner rules on centroid regeneration, and the q_irr path is opt-in per
    # deck."  It shipped briefly defaulting to ``auto`` and the 2026-08-08
    # landing census measured what that cost — nine red cells across the GW
    # and BSE restart paths on the two decks whose centroid sets are already
    # orbit-closed.  A default that moves bytes is a default that has to be
    # asked for.
    #   full — the DEFAULT.  Preserve today's bytes exactly, unconditionally.
    #          Does not ask the closure question, so it is also the control
    #          arm of any A/B.
    #   auto — store the pre-unfold IBZ wedge when the deck's centroid set is
    #          orbit-closed AND this run's q path actually reduced; the full
    #          BZ otherwise.  On a non-closed set (today's Si production
    #          960-centroid deck: 47 of 48 ops violating) this is byte-for-byte
    #          today's file.  On a CLOSED set it is ~8x smaller.
    #   ibz  — REFUSE on a set that is not storable, naming which of the two
    #          conditions failed.  For a deck that believes it is closed and
    #          wants to be told the day it stops being.
    # BEFORE YOU SET auto OR ibz ON A CLOSED-SET DECK: the readers do not
    # unfold yet.  ``bse_io._MunuSlabPlan`` refuses a wedge outright (at every
    # process count, not only P>1), and the GW restart reader refuses it too
    # since the same landing — see ``file_io.tagged_arrays``.  The wedge is
    # therefore usable today by runs that DISCARD the restart artifact or read
    # it back through the serial h5py path.
    #
    # ⚠ THIS WHOLE KEY IS TRANSITIONAL — OWNER RULING 2026-08-08 ~13:20.
    # `full` is where it rests until the key is DELETED, not a permanent
    # setting to tune.  The owner's ruling is that symmetry should never have
    # needed a mode switch at all: "symmetries should not need an auto mode —
    # if symmetries are not to be used, the wavefunction file should've been
    # generated with no symmetries."  The WFN file already answers the
    # question this key asks, so the end state is storage that FOLLOWS the
    # file — the wedge whenever the deck carries symmetries, readers that
    # always unfold — with `restart_q_storage` retired entirely rather than
    # defaulted differently.  That work is the GW+BSE restart consolidation
    # registered in tests/KNOWN_FAILURES.md; do not build on this key.
    # See gw/restart_q_storage.py for the resolution and the seam, and
    # DESIGN_symmetry_restart_followup.md for the pre-unfold-persistence
    # decision this key selects.
    "restart_q_storage": "full",
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
    # j-RESOLVED (True) vs j-AVERAGED (False) nonlocal projectors, i.e.
    # QE's ``lspinorb``.  ``None`` means "not declared", which is the
    # historical state and keeps ``vnl_ops.resolve_soc_mode``'s announced
    # j-resolved default and its banner.  It is a deck key because
    # nothing in a BerkeleyGW WFN.h5 records lspinorb -- ``mf_header``
    # carries ``nspinor``, and ``noncolin`` is not ``lspinorb`` -- so for
    # a BGW-fed run the deck is the only place the answer can come from.
    # A QE ``.save``-fed run reads ``<spinorbit>`` and does not need it.
    # The default is the EMPTY STRING and not ``False``: the reader types
    # each key from its default, so a bool default would make "unset" and
    # "soc = false" the same request, and they are opposite ones.
    "soc": "",
    # The relative sign of the i[r, V_NL] commutator in the assembled
    # velocity, read by ``psp.get_dipole_mtxels`` and passed to
    # ``common.mtxel_sweep.dipole_operator``.  ``-1`` is the shipped
    # assembly and ``+1`` the arm that reproduces BerkeleyGW's q -> 0
    # head; the words "shipped" / "flipped" spell the same two.  Empty is
    # NOT DECLARED and resolves to the shipped sign, on the same reading
    # as ``soc`` above -- and for the same reason it must be a string
    # default: a float default would make "unset" and an explicit "-1"
    # indistinguishable, and the whole point of the stamp this feeds is
    # to say which arm a dipole.h5 was built with.
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
    # ψ(G) source for the ISDF r-chunk loop.  Both modes keep ψ(G) on
    # the HOST in per-rank band-sharded layout and pull one band-chunk
    # at a time into the jit via io_callback — never more than one bc
    # on device at a time.  Modes differ in host-side lifecycle:
    #   "host_cache"  – read once at startup, keep resident in host
    #                   RAM for the full run (default; fastest).
    #   "file_reread" – rebuild the host buffer at each r-chunk via
    #                   phdf5 collective read; drop between r-chunks.
    #                   Zero persistent host residency (needed for
    #                   huge systems where host RAM can't hold ψ(G)).
    "gspace_mode": "host_cache",
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
    # knobs live in the deck).
    "transverse_zeta_rcond": 1e-10,
    # γ̃-double-contract kernel variant inside the monolithic pair
    # pipeline (see ``common.gamma_matrices.gamma_double_contract``).
    # Math identical across all three; differ in HLO structure.
    #   "take"   – jnp.take + element-wise phase mul (default).
    #   "einsum" – materialise the sparse γ̃ and contract via einsum.
    #   "scan"   – lax.scan over the (a, b) spin axis pairs.
    "gamma_contract_mode": "take",
    # Memory / chunking
    "memory_per_device_gb": 0.0,  # 0 = auto-detect
    "band_chunk_size": 16,
    "r_chunk_size": 0,
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
    # BSE fine-grid densification.  When set to "NX NY NZ" (or "NX,NY,NZ") and
    # DIFFERENT from the coarse restart/WFN grid, the GENERAL BSE init
    # (``bse_io.load_bse_data_from_restart_sharded``) interpolates the ENTIRE
    # BSE problem — ψ, QP ε (htransform fH), V_Q exchange (vq_interp), and the
    # W direct term (zero-pad in R) — from the coarse grid onto this fine grid
    # BEFORE any solve, so EVERY BSE solver (exciton_bands / feast / nontda /
    # kpm / resolvent) transparently runs on the fine grid.  Each fine length
    # must be a positive multiple of the matching coarse length (coarse BZ ⊂
    # fine BZ).  Empty (default) or == the coarse grid → the coarse ``data``
    # bundle is returned byte-identically (fast path untouched).  Subsumes the
    # exciton_bands ``--w-coarse-grid`` W-only flag for the direct term.
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
    # Sigma frequency grid
    "sigma_omega_min_ev": -5.0,
    "sigma_omega_max_ev": 5.0,
    "sigma_omega_step_ev": 0.25,
    "sigma_regularization_ev": 0.25,
    "sigma_window_edge_factor": 1.5,
    "sigma_omega_batch_size": 4,
    "sigma_omega_accumulation": "auto",
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
    "fermi_reference": "midgap",
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
    # 2026-08-06: the sharded-slab transport is no longer selectable from
    # a deck.  There was one correct value and three wrong ones, two of
    # which named tiers that have been deleted; a key whose only legal
    # setting is the default is not configuration.  Both are
    # warn-and-ignore rather than refuse: they never changed a NUMBER,
    # only which library moved the bytes, so an old deck should keep
    # running and say what it lost.  See file_io/slab_io.py.
    "slab_io",                      # warn-and-ignore (one transport now)
    "use_ffi_io",                   # warn-and-ignore (deprecated 2026-07-27)
})

# Keys whose string values should be lowercased and stripped
_NORMALIZE_STR = {
    "compute_mode",
    "qp_solver",
    "sc_accelerator",
    "sc_eigh",
    "wcoul0_source", "screening_method", "minimax_energy_reference",
    "sigma_omega_accumulation", "sigma_omega_layout", "fermi_reference",
    "w_dyson_solver",
    "ppm_invalid_mode",
    "ppm_model",
    "ppm_probe_chi_reuse",
    # ``restart_q_storage`` normalises here and is VALIDATED at parse time
    # against RESTART_Q_STORAGE, the same shape ``hartree_source`` uses: a
    # key whose wrong value would otherwise surface as a refusal deep in the
    # restart write, after the compute.
    "restart_q_storage",
    # distributed-linalg backend axes (consumed both via LorraxConfig and
    # directly from the params dict by htransform / exciton_bands).
    "eigh_backend",
}

# Tri-state booleans: _DEFAULTS value is None (= unset), an explicit
# input-file value parses as bool.  The parse loop needs the set because
# ``default is None`` otherwise means "nullable float".
_NULLABLE_BOOL = frozenset()

#: Reserved slot in the params dict holding the set of deck keys the DECK
#: named.  Leading underscore because it is not a deck key and must never
#: match one: ``read_lorrax_input`` builds params from ``_DEFAULTS.items()``
#: and every real key comes from there, so a name that cannot appear in
#: ``_DEFAULTS`` cannot collide.  Read once, into ``GWConfig.raw_input_keys``.
_DECK_NAMED_KEYS = "_deck_named_keys"


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
        # ``slab_io`` / ``use_ffi_io`` had no branch at all: named in
        # ``_LEGACY_DECK_KEYS`` (so skipped by the unknown-key check) and
        # then handled nowhere, which is the silent case in its purest
        # form.  There is one sharded-slab transport and the deck does not
        # select it; see file_io/slab_io.py.
        for legacy_key in ("slab_io", "use_ffi_io"):
            if section.get(legacy_key, fallback=None) is not None:
                import warnings
                warnings.warn(
                    f"Input key '{legacy_key}' is no longer supported and "
                    f"will be ignored: there is one sharded-slab transport "
                    f"and the deck does not select it.  It never changed a "
                    f"number, only which library moved the bytes.  Remove "
                    f"'{legacy_key}' from your input file.",
                    DeprecationWarning, stacklevel=2,
                )
                retired.append((
                    legacy_key,
                    "IGNORED — one sharded-slab transport, not deck-selectable "
                    "(it never changed a number, only which library moved "
                    "the bytes)"))
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
                # value parses as bool.  Currently EMPTY — the last member
                # (``use_ffi_io``) became a legacy key on 2026-08-06.
                params[key] = section.getboolean(key)
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
    sigma_diag_file: str
    eqp0_file: str
    eqp1_file: str
    sigma_omega_h5_file: str
    #: Staged multipole fit store for ``compute_mode = mpa``.  ``None``
    #: means the deck did not name one, which is legal for every other
    #: mode and refused for that one — see
    #: :func:`refuse_missing_mpa_fit_store`.
    mpa_fit_file: str | None = None


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


@dataclass(frozen=True)
class HeadConfig:
    """q→0 Coulomb-head sources, BGW vcoul override, bare-cutoff knobs.

    All Coulomb-at-small-q tweaks live here.  Σ head plumbing
    (``wcoul0_*``, ``vhead``/``whead_*``) is consumed by
    :class:`gw.head_correction.HeadResolver`; the BGW vcoul override is
    purely diagnostic (matches BGW's per-G mini-BZ averaging exactly for
    bit-reproducible comparisons).
    """
    wcoul0_source: str            # "s_tensor" | "epshead"
    wcoul0_eta: float
    vhead: float | None           # explicit override v_h[ω=0]
    whead_0freq: float | None     # explicit override W_h[ω=0]
    whead_imfreq: float | None    # explicit override W_h[iω_p]
    mc_average_vcoul_body: bool
    mc_average_placement: str      # "off" (default) | "bgw" | "schur_avg"
    mc_average_placement_vcoul: str | None   # BGW vcoul dump for byte-sourced <v>
    head_minibz_average: bool      # per-Q mini-BZ head cell-average (default off)
    bare_coulomb_cutoff: float | None
    zeta_cutoff: float | None
    use_bgw_vcoul: bool
    bgw_vcoul_file: str | None
    bgw_vcoul_sym_wfn: str | None


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
    """
    method: str                   # "minimax" -- the only supported value
    minimax_target_error: float
    minimax_max_nodes: int
    regenerate_minimax_tables: bool
    minimax_energy_reference: str  # "midgap" | "vbm"

    def __post_init__(self):
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


@dataclass(frozen=True)
class PPMConfig:
    """Plasmon-pole model + Σ_c(ω) output grid + on-shell options.

    Single grouped home for everything PPM/Σ_c-related: the pole-fit
    ansatz, the probe-ω choice, the analytic head-pole override, the
    σ-quadrature minimax tolerances, the ω-grid for the output, and
    the post-hoc on-shell evaluation knobs.
    """
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

    # --- ω-grid for Σ_c(ω) output (eV) ---
    omega_min_ev: float
    omega_max_ev: float
    omega_step_ev: float
    regularization_ev: float
    window_edge_factor: float
    omega_batch_size: int
    omega_accumulation: str       # "auto" | "kij"
    #: Σ_c(ω) end-of-stage layout: "replicated" gathers the full cube onto
    #: every rank (historical path); "sharded" keeps it (m_X, n_Y)-tiled on
    #: the existing mesh and consumers read the tiles directly (movement-only,
    #: bit-identical outputs; see _DEFAULTS["sigma_omega_layout"]).
    omega_layout: str             # "replicated" | "sharded"

    # --- on-shell evaluation knobs ---
    invalid_mode: str             # "zero" | "2ry" | "static_limit" | "infinity"(alias)
    fermi_reference: str          # "midgap" | "vbm"
    sigma_at_dft_extrapolate: bool
    sigma_at_dft_energies: bool

    def __post_init__(self):
        # Validate scalar knobs once, at the parse site (values are already
        # normalized in ``from_input_file``).  Capability gating for
        # invalid_mode ('imaginary' → NotImplementedError, needs a
        # complex-Ω path) stays in the Σ^c kernel — this checks only
        # that the *value* is recognized.
        if self.omega_step_ev <= 0.0:
            raise ValueError("ppm.omega_step_ev must be > 0.")
        if self.omega_max_ev < self.omega_min_ev:
            raise ValueError("ppm.omega_max_ev must be >= ppm.omega_min_ev.")
        if self.fermi_reference not in ("vbm", "midgap"):
            raise ValueError("ppm.fermi_reference must be 'vbm' or 'midgap'.")
        if self.omega_accumulation == "kij_stream":
            # REMOVED mode (2026-07-31): the single-process streamed-h5
            # accumulator is gone.  Refuse the removed VALUE of a known
            # key rather than silently rerouting an old deck.
            raise ValueError(
                "sigma_omega_accumulation = kij_stream was REMOVED: the "
                "single-process streamed-h5 accumulator is gone "
                "(superseded by host-tile accumulation and "
                "sigma_omega_layout=sharded for cubes that do not fit).  "
                "Use 'kij' or 'auto'.")
        if self.omega_accumulation not in ("auto", "kij"):
            raise ValueError(
                "ppm.omega_accumulation must be auto/kij.")
        if self.omega_layout not in ("replicated", "sharded"):
            raise ValueError(
                "sigma_omega_layout must be 'replicated' or 'sharded'; "
                f"got {self.omega_layout!r}.")
        if self.invalid_mode not in (
            "zero", "skip", "2ry", "static_limit", "infinity", "imaginary"
        ):
            raise ValueError(
                f"ppm.invalid_mode: unknown value {self.invalid_mode!r}")
        if self.omega_batch_size < 1:
            raise ValueError("ppm.omega_batch_size must be >= 1.")
        if self.probe_chi_reuse not in ("off", "auto"):
            raise ValueError(
                "ppm_probe_chi_reuse must be 'off' or 'auto'; "
                f"got {self.probe_chi_reuse!r}.")


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


@dataclass(frozen=True)
class MemoryConfig:
    """Per-device memory budget + chunk sizing + AOT chunk-chooser flag.

    ``memory_per_device_gb=0`` triggers GPU auto-detection at config
    construction time.  ``chunk_target_utilization`` is sourced from the
    ``ISDF_CHUNK_TARGET_UTILIZATION`` env var (default 0.97).
    ``zct_stage_cap_gb`` similarly from
    ``ISDF_ZCT_STAGE_CAP_GB`` / ``ISDF_ZCT_STAGE_CAP_FRAC``.
    """
    per_device_gb: float
    chunk_target_utilization: float
    band_chunk_size: int
    r_chunk_override: int         # 0 = auto
    zct_stage_cap_gb: float | None
    gflat_chunk_size: int         # 0 = one-shot (or planner-picked)
    vq_g_chunk_size: int          # 0 = auto _pick_g_chunk(ngkmax)


@dataclass(frozen=True)
class BackendConfig:
    """Three-axis backend selection: I/O + ψ(G) lifecycle + W Dyson plan.

    All three knobs were previously orthogonal-sounding boolean/string
    flags in different namespaces (``gspace_mode`` /
    ``isdf_memory_mode``) that secretly toggled FFI paths.  Grouped here
    so :meth:`summary` can print one line at startup describing what's
    actually active per channel.
    """
    gspace_io: GspaceIO
    w_dyson_solver: str  # "local" | "distributed" (normalized; W Dyson plan)
    distributed_cholesky: str  # "auto" | "off" | "cusolvermp" | "slate"
    distributed_lu: str        # "auto" | "off" | "cusolvermp" | "scalapack"
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
            f"backend: gspace_io={self.gspace_io.value}, "
            + (f"w_dyson_solver={self.w_dyson_solver}, "
               if self.w_dyson_solver != "local" else "")
            + f"distributed_cholesky={self.distributed_cholesky}, "
            f"distributed_lu={self.distributed_lu}, "
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
        config.debug.sigma_freq_debug_output

    See module docstring for the full grouping.  ``cohsex.in`` keys
    are unchanged — input files written for prior versions still parse
    (the factory unflattens the dict into sub-dataclasses).
    """

    # --- System geometry (top-level; hot path) ---
    nval: int
    ncond: int
    nband: int
    sys_dim: int
    density_self_consistent: bool
    sc_on_ibz: bool
    #: auto | stored | isdf | gspace — see HARTREE_SOURCES.
    hartree_source: str

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
    #: Energy unit of the multipole fit store's poles and residues.  See
    #: ``_DEFAULTS["mpa_pole_energy_unit"]`` for why this is a deck key
    #: and not a constant, and ``gw.mpa.sigma_head`` for the conversion.
    mpa_pole_energy_unit: str
    #: Which named q→0 head set in the fit store to read.  See
    #: ``_DEFAULTS["mpa_head_label"]``; ``as_shipped`` is the bare
    #: ``__mpahead`` group and therefore also every pre-label store.
    mpa_head_label: str
    #: Which poles this process integrates, comma separated; "" is every
    #: pole.  See ``_DEFAULTS["mpa_pole_subset"]``.
    mpa_pole_subset: str
    #: Where this process writes its partial Σ_c cube (""= not a partial
    #: run).  See ``_DEFAULTS["mpa_pass_partial_out"]``.
    mpa_pass_partial_out: str
    #: A file, glob or directory of partial Σ_c cubes to sum in the pinned
    #: order instead of integrating ("" = integrate).  See
    #: ``_DEFAULTS["mpa_pass_partial_in"]``.
    mpa_pass_partial_in: str

    # --- Sub-dataclass groups (everything else) ---
    paths: FilePaths
    head: HeadConfig
    screening: ScreeningConfig
    ppm: PPMConfig
    sc: SCConfig
    memory: MemoryConfig
    backend: BackendConfig
    debug: DebugConfig
    bse: BSEConfig

    # --- Optional parsed blocks ---
    kpoints_crystal_b: dict | None = None

    # --- Input directory (for resolving relative paths at runtime) ---
    input_dir: str = ""

    # ------------------------------------------------------------------
    #  Derived config objects
    # ------------------------------------------------------------------

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
        """
        p = self.ppm
        n = int(np.floor(
            (p.omega_max_ev - p.omega_min_ev) / p.omega_step_ev + 0.5)) + 1
        return p.omega_min_ev + p.omega_step_ev * np.arange(n, dtype=np.float64)

    @property
    def omega_grid_ry(self):
        """Σ_c(ω) frequency grid in Rydberg (derived from the eV grid)."""
        return self.omega_grid_ev / RYD_TO_EV

    # ------------------------------------------------------------------
    #  Back-compat aliases — the FFI/IO group changed semantics (bool /
    #  string → enum), so callers that still want the old names get
    #  coerced views.  New code should use ``config.backend.<field>`` /
    #  ``config.memory.<field>`` etc. directly.
    # ------------------------------------------------------------------

    # ``use_ffi_io`` (the property) is GONE with the tiers it
    # distinguished.  It answered "is this a per-rank-parallel writer?",
    # which now has one answer for every run, so a caller branching on it
    # was branching on a constant.  2026-08-06.

    @property
    def gspace_mode(self) -> str:
        """Legacy ``gspace_mode: str`` view of ``backend.gspace_io``."""
        return self.backend.gspace_io.value

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

        # --- ZCT stage cap from env ---
        # ``total_gb`` is the PHYSICAL card; 0.0 when the backend has no
        # device memory (CPU).  Passing it in keeps the decision — and
        # every "no cap, and here is why" announcement — in one pure,
        # testable place instead of behind a backend guard that used to
        # skip the fraction branch in silence.
        import jax
        _zct_total_gb = 0.0
        if jax.default_backend() in ("gpu", "cuda"):
            from common.gpu_utils import get_device_memory_info
            _zct_total_gb = float(
                get_device_memory_info().get("total_gb", 0.0))
        zct_stage_cap_gb = resolve_zct_stage_cap(
            os.environ.get("ISDF_ZCT_STAGE_CAP_GB"),
            os.environ.get("ISDF_ZCT_STAGE_CAP_FRAC"),
            per_device_gb=memory_per_device_gb,
            total_gb=_zct_total_gb,
            print_fn=print_fn)

        def _g(key):
            return params.get(key, _DEFAULTS.get(key))

        # --- Build sub-dataclasses ---
        cents_curr = _g("centroids_file_current")
        cents_curr_resolved = str(cents_curr) if cents_curr else None
        paths = FilePaths(
            wfn_file=str(_g("wfn_file")),
            centroids_file=str(_g("centroids_file")),
            centroids_file_current=cents_curr_resolved,
            kin_ion_file=str(_g("kin_ion_file")),
            sigma_diag_file=str(_g("sigma_diag_file")),
            eqp0_file=str(_g("eqp0_file")),
            eqp1_file=str(_g("eqp1_file")),
            sigma_omega_h5_file=str(_g("sigma_omega_h5_file")),
            # Empty string is "not set", the same convention
            # ``centroids_file_current`` uses — and the same reason: an
            # absent optional path must be distinguishable from a path
            # that happens to resolve to the deck's own directory.
            mpa_fit_file=(str(_g("mpa_fit_file")) or None),
        )
        head = HeadConfig(
            wcoul0_source=str(_g("wcoul0_source")).strip().lower(),
            wcoul0_eta=float(_g("wcoul0_eta") or 0.0),
            vhead=_g("vhead"),
            whead_0freq=_g("whead_0freq"),
            whead_imfreq=_g("whead_imfreq"),
            mc_average_vcoul_body=bool(_g("mc_average_vcoul_body")),
            mc_average_placement=_normalize_placement(
                _g("mc_average_placement")),
            mc_average_placement_vcoul=(
                str(_g("mc_average_placement_vcoul") or "") or None),
            head_minibz_average=bool(_g("head_minibz_average")),
            bare_coulomb_cutoff=_g("bare_coulomb_cutoff"),
            zeta_cutoff=_g("zeta_cutoff"),
            use_bgw_vcoul=bool(_g("use_bgw_vcoul")),
            bgw_vcoul_file=(str(_g("bgw_vcoul_file")) or None),
            bgw_vcoul_sym_wfn=(str(_g("bgw_vcoul_sym_wfn")) or None),
        )
        screening = ScreeningConfig(
            method=str(_g("screening_method")).strip().lower(),
            minimax_target_error=float(_g("minimax_target_error")),
            minimax_max_nodes=int(_g("minimax_max_nodes")),
            regenerate_minimax_tables=bool(_g("regenerate_minimax_tables")),
            minimax_energy_reference=str(_g("minimax_energy_reference")).strip().lower(),
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
            omega_min_ev=float(_g("sigma_omega_min_ev")),
            omega_max_ev=float(_g("sigma_omega_max_ev")),
            omega_step_ev=float(_g("sigma_omega_step_ev")),
            regularization_ev=float(_g("sigma_regularization_ev")),
            window_edge_factor=float(_g("sigma_window_edge_factor")),
            omega_batch_size=int(_g("sigma_omega_batch_size")),
            omega_accumulation=str(_g("sigma_omega_accumulation")).strip().lower(),
            omega_layout=str(_g("sigma_omega_layout")).strip().lower(),
            invalid_mode=str(_g("ppm_invalid_mode") or "static_limit").strip().lower(),
            fermi_reference=str(_g("fermi_reference")).strip().lower(),
            sigma_at_dft_extrapolate=bool(_g("sigma_at_dft_extrapolate")),
            sigma_at_dft_energies=bool(_g("sigma_at_dft_energies")),
        )
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
        )
        memory = MemoryConfig(
            per_device_gb=memory_per_device_gb,
            chunk_target_utilization=chunk_utilization,
            band_chunk_size=int(_g("band_chunk_size")),
            r_chunk_override=int(_g("r_chunk_size")),
            zct_stage_cap_gb=zct_stage_cap_gb,
            gflat_chunk_size=int(_g("gflat_chunk_size")),
            vq_g_chunk_size=int(_g("vq_g_chunk_size")),
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
            gspace_io=GspaceIO(str(_g("gspace_mode")).strip().lower()),
            w_dyson_solver=_w_dyson_solver,
            distributed_cholesky=_dist_chol,
            distributed_lu=_dist_lu,
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
            _g("restart_q_storage") or "full").strip().lower()
        if _restart_q_storage not in RESTART_Q_STORAGE:
            raise ValueError(
                f"restart_q_storage={_restart_q_storage!r} is not one of "
                f"{RESTART_Q_STORAGE}.  This key selects the q-set the "
                "restart tensors are STORED on; a value nobody recognises "
                "is not silently read as the default.")

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

        return cls(
            # Top-level: system + mode flags
            nval=int(_g("nval")),
            ncond=int(_g("ncond")),
            nband=int(_g("nband")),
            sys_dim=int(_g("sys_dim")),
            density_self_consistent=bool(_g("density_self_consistent")),
            sc_on_ibz=bool(_g("sc_on_ibz")),
            hartree_source=_hartree_source,
            restart=bool(_g("restart")),
            write_restart_tensors=bool(_g("write_restart_tensors")),
            write_qsgw_datasets=bool(_g("write_qsgw_datasets")),
            restart_q_storage_raw=_restart_q_storage,
            raw_input_keys=frozenset(params.get(_DECK_NAMED_KEYS, ())),
            compute_mode_raw=str(_g("compute_mode") or "auto").strip().lower(),
            qp_solver_raw=str(_g("qp_solver") or "auto").strip().lower(),
            do_screened=bool(_g("do_screened")),
            bispinor=bool(_g("bispinor")),
            do_G0=bool(_g("do_G0")),
            self_consistent=bool(_g("self_consistent")),
            use_ppm_sigma=bool(_g("use_ppm_sigma")),
            no_degen_averaging=bool(_g("no_degen_averaging")),
            degen_avg_tol_ry=float(_g("degen_avg_tol_ry")),
            mpa_pole_energy_unit=coerce_pole_energy_unit(
                _g("mpa_pole_energy_unit")),
            mpa_head_label=str(_g("mpa_head_label") or "").strip(),
            mpa_pole_subset=str(_g("mpa_pole_subset") or "").strip(),
            mpa_pass_partial_out=str(
                _g("mpa_pass_partial_out") or "").strip(),
            mpa_pass_partial_in=str(_g("mpa_pass_partial_in") or "").strip(),
            # Sub-dataclass groups
            paths=paths,
            head=head,
            screening=screening,
            ppm=ppm,
            sc=sc,
            memory=memory,
            backend=backend,
            debug=debug,
            bse=bse,
            # Parsed blocks
            kpoints_crystal_b=params.get("kpoints_crystal_b"),
            input_dir=input_dir,
        )
