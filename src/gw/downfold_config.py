"""The downfold driver's OWN input file — schema, defaults and refusals.

The downfold is not a GW run and it does not read a GW deck.  It reads a
finished GW calculation's restart bundle off disk and writes a smaller one,
so the only things it needs to be told are where the parent is, which bands
it must keep faithful, how small the answer should be, how much round-off
amplification is authorised, and where to put the result.  Six facts.  A GW
deck carries a hundred and forty, and pointing this driver at one would make
every one of them look like an input to a compression it has no bearing on.

FORMAT.  The same INI-ish text every LORRAX deck uses — one ``[downfold]``
section, ``key = value``, ``#`` comments — so a user who can read a GW deck
can read this one.  Nothing else is shared with ``gw_config``: the key set
below is the whole schema.

UNKNOWN KEYS ARE REFUSED, not ignored.  ``gw_config`` warns instead, and it
is right to, because it carries a decade of decks that must keep loading.
This schema has no history to protect, and a misspelt ``downfold_rcond`` that
silently keeps the default is exactly the class of quiet wrong answer the
rest of this initiative is built to prevent.

WHY THE WINDOW IS THE MOST IMPORTANT KEY IN THIS FILE.  The retained band
window is what the small basis is SELECTED against, and
``docs/dev/isdf_basis_adequacy_at_large_nband.md`` records what happens when
a basis is selected against one window and consumed on another: a GW run at
nband=1024 whose centroids had been pruned on a 26x52 window produced a QP
gap of 0.36 eV where the answer is ~3.1-3.7 eV, with a negative ``eqp1``, and
passed every gate in the suite.  Rebuilding at identical everything and
changing only the prune window moved ``eqp0`` from 0.3645 to 3.1350 to 3.7227
eV, monotone in window width.  A downfold is that same operation done on
purpose; the window key is where the purpose is stated.
"""
from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass

from gw.downfold import DEFAULT_RCOND

__all__ = [
    "DOWNFOLD_DEFAULTS",
    "DownfoldConfig",
    "read_downfold_input",
]


# ---------------------------------------------------------------------------
# THE SCHEMA.  One dict, each key documented where it lives, and the type of
# the default is what coerces the deck's string.  ``None`` means "no default
# — the deck must say", and the refusal names what to say.
# ---------------------------------------------------------------------------

DOWNFOLD_DEFAULTS = {
    # ---- what to read, what to write ---------------------------------
    # ``source_restart``: the FINISHED GW calculation to compress.  Either
    # the run directory (the driver then finds ``tmp/isdf_tensors_*.h5``
    # inside it, the same lookup ``bse_io._find_restart_file`` does and with
    # the same loud refusal when more than one centroid count is present) or
    # the ``.h5`` file itself when a directory holds several.  Relative
    # paths resolve against the directory of THIS input file, so a downfold
    # deck sitting beside the GW run reads ``source_restart = .``.
    "source_restart": None,
    # ``output_restart``: where the small bundle goes.  A DIRECTORY; the
    # driver creates ``<dir>/tmp/isdf_tensors_<mu_S>.h5`` inside it, which
    # is the layout every BSE consumer already globs for.  It must not be
    # the source directory: two ``isdf_tensors_*.h5`` at different centroid
    # counts in one directory is the ambiguity ``_find_restart_file`` warns
    # about, and the downfold is not going to manufacture it.
    "output_restart": None,
    # ``parent_centroids_file``: OPTIONAL.  The parent run's centroid
    # coordinate table.  When given, the driver writes the kept rows out as
    # a sibling centroid file and stamps its md5 into the small bundle, so
    # the small basis can be handed to a fresh GW run.  When absent the
    # bundle is still complete for every BSE consumer — the coordinates are
    # not in the bundle format at all, only their hash is — and the driver
    # says so rather than stamping the parent's hash, which would be a lie
    # about which points the tensors describe.
    "parent_centroids_file": "",

    # ---- the retained band window ------------------------------------
    # THE FIT WINDOW.  Two spellings, and exactly one of them may be used.
    #
    #   n_val / n_cond          -> left = (0, n_val), right = (n_val,
    #                              n_val + n_cond).  The legacy
    #                              valence/conduction spelling, identical to
    #                              ``centroid.pivoted_cholesky``'s.
    #   band_range_left/right   -> "lo:hi", half-open, ABSOLUTE band indices
    #                              into psi_full_y / enk_full.
    #
    # Stage 1 serves the BSE, so the window is expected to be SYMMETRIC
    # (left == right): the BSE's direct and exchange kernels both contract ψ
    # legs that lie inside the retained window, so a symmetric fit covers
    # them exactly.  An ASYMMETRIC window (retained x all-bands) is what
    # Sigma would need, because Sigma's internal band sum runs over the full
    # window while its outer projection does not — and the probe measures
    # that at about 2x the mu_S, not the order of magnitude the design
    # feared.  Stage 1 ACCEPTS an asymmetric window and says loudly that it
    # is unvalidated; it does not refuse it, because the algebra is the same
    # and the measurement that would validate it is a later stage.
    "n_val": None,
    "n_cond": None,
    "band_range_left": "",
    "band_range_right": "",

    # ---- the size of the answer --------------------------------------
    # ``mu_small``: the small centroid count.  NO DEFAULT, deliberately —
    # it is a physics choice and the number that is right depends on the
    # window, which the driver cannot guess.  ``auto`` means "set it to the
    # number of independent pair-density directions the window holds at
    # ``downfold_rcond``", which on the rank probe's evidence is the only
    # value a user can pick without having read the probe.
    #
    # The refusal when it is too large is not a failure: it is this
    # initiative's whole point arriving early and cheaply.  A 20-band window
    # holds about 190 directions on both decks measured; a nominal
    # "500-centroid small basis" on such a window is a fiction in which two
    # thirds of the basis is truncated away by the solve that consumes it.
    "mu_small": None,

    # ---- conditioning -------------------------------------------------
    # ``downfold_rcond``: 1/kappa_cap on the truncated pseudo-inverse of
    # S_SS.  The retained set is {i : lambda_i > rcond * lambda_max}, so the
    # achieved amplification is at most 1/rcond BY CONSTRUCTION.  It is a
    # CAP ON AMPLIFICATION, not a gap-finder — ISDF pair-density spectra are
    # smooth and have no knee, elbow or plateau to cut at, and every
    # criterion phrased as "cut at the separation" is inapplicable here
    # (``common/rank_criterion.py`` carries the derivation and the
    # measurements that refute the discrepancy principle, the L-curve and
    # GCV against this exact failure mode).
    #
    # DEFAULT 1.1e-6, and it is a measured number rather than a round one.
    # ``DOWNFOLD_RANK_PROBE.md`` §1: at a 20-band window si_bse_debug holds
    # 196 independent directions at rcond 1e-6 and hbn_cohsex_debug 185,
    # both stable to 2 % as the candidate pool grows 9-12x — a real ceiling.
    # At 1e-8 the same window reports 693 and at 1e-10 1208, and neither is
    # a ceiling: they keep climbing with the size of the pool you were
    # willing to pay for.  Buying mu_S = 500 on a 20-band window means
    # keeping directions at lambda/lambda_max = 3.5e-8, and the R19 anchor
    # says what that costs.
    #
    # The only admissible evidence for changing this is OBSERVABLE
    # convergence — sweep it and take the plateau in the energy, not in the
    # spectrum.  That sweep is a later stage; this default is where it
    # starts.
    "downfold_rcond": DEFAULT_RCOND,
    # ``downfold_select_tol``: pivoted Cholesky's stopping tolerance,
    # relative to the largest initial Gram diagonal.  Empty = sqrt(eps),
    # the kernel's own default.  IT IS NOT THE SAME KNOB AS
    # ``downfold_rcond`` and does not produce the same rank: measured on one
    # Gram, the selection rank runs about 3x the eigenvalue truncation rank
    # at the same nominal number, because pivoted Cholesky stops on a
    # residual Schur diagonal and the truncation stops on an eigenvalue.
    # Setting both to the same value is the most natural mistake available
    # here, and the driver prints the two ranks side by side, labelled
    # differently, so it cannot be made silently.
    "downfold_select_tol": None,

    # ---- how it runs ---------------------------------------------------
    # ``mode``: ``cur`` selects the small basis as a SUBSET of the parent's
    # centroids and builds the transfer from the resulting submatrices —
    # both fit operands are then submatrices of one object, no second zeta
    # fit is needed anywhere, the new psi-at-centroids is a literal column
    # slice, and the exact-reproduction gate becomes an algebraic identity
    # rather than a hopeful tolerance.  ``refit`` — a fresh narrow-window
    # k-means plus a fresh zeta fit — is REFUSED in stage 1 and the refusal
    # says why: on the probe's evidence the parent's own k-means set already
    # certifies 194 of the 196 directions a 20-band window contains, so a
    # re-fit would buy at most 1 % more rank for a second full zeta fit.
    # The key exists so the door stays open and so an owner who wants it can
    # ask for it by name.
    "mode": "cur",
    # ``plan``: ``auto`` and ``local`` both mean the GSPMD/local plan, which
    # is what stage 1 implements.  ``distributed`` (mu tiled over a 2-D
    # process grid through ``distrib_la``) is a later stage and is refused
    # rather than silently demoted, because a demotion changes the numerical
    # gauge and this codebase's contract is to announce demotions and refuse
    # explicit requests it cannot honour.
    "plan": "auto",
    # ``report_residual``: compute and print the per-q Pythagorean error bar
    # eps_W.  Default TRUE and it should stay true: it is the only number
    # that answers "did the downfold work" without a reference calculation,
    # and its cost is two GEMMs at mu_L per q — the same cost class as the
    # congruence.  Turning it off also drops the mu_L^2 parent Gram, which
    # is the largest object on this path; that is the trade, and it is the
    # caller's to make explicitly.
    "report_residual": True,
    # ``residual_refuse_above``: refuse to WRITE the small bundle when the
    # worst-q eps_W exceeds this.  Empty = report only, never refuse, which
    # is the stage-1 default because nobody has yet measured what eps_W a
    # good downfold has on a production deck.  Set it once you know.
    "residual_refuse_above": None,
}

#: Keys whose value is lower-cased and stripped before coercion.
_NORMALIZE_STR = ("mode", "plan", "mu_small")

_RANGE_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")


def _parse_range(raw: str, key: str) -> tuple[int, int]:
    m = _RANGE_RE.match(raw)
    if not m:
        raise ValueError(
            f"downfold: {key}={raw!r} is not a band range.  Spell it "
            f"'lo:hi' — half-open, ABSOLUTE band indices into the restart's "
            f"psi_full_y / enk_full, counting from 0.  Example: a 20-band "
            f"window over the lowest 20 bands is '0:20'.")
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo >= hi:
        raise ValueError(
            f"downfold: {key}={raw!r} is empty (lo >= hi).  The range is "
            f"half-open: '0:20' means bands 0..19.")
    return lo, hi


def read_downfold_input(filename: str) -> dict:
    """Parse a ``[downfold]`` input file into a plain dict of typed values.

    Type-driven coercion off :data:`DOWNFOLD_DEFAULTS`, exactly as
    ``gw_config.read_lorrax_input`` does: the type of the default decides how
    the deck's string is read.  Unknown keys REFUSE.
    """
    with open(filename, "r") as fh:
        text = fh.read()

    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    try:
        parser.read_string(text)
    except configparser.Error as exc:
        raise ValueError(
            f"downfold: could not parse {filename} as an INI-style input "
            f"file ({exc}).  The format is one '[downfold]' section header "
            f"followed by 'key = value' lines; '#' starts a comment.") from exc

    if parser.has_section("downfold"):
        section = "downfold"
    elif parser.sections():
        section = parser.sections()[0]
    else:
        raise ValueError(
            f"downfold: {filename} has no section header.  The first "
            f"non-comment line must be '[downfold]'.  If you pointed this "
            f"driver at a GW deck ('[cohsex]'), that is the mistake this "
            f"driver exists not to make: the downfold takes its OWN input "
            f"file naming a finished GW calculation, not the deck that "
            f"produced it.")
    if section != "downfold":
        raise ValueError(
            f"downfold: {filename} declares section [{section}], not "
            f"[downfold].  A GW deck ('[cohsex]') is not a downfold input "
            f"file — see docs/downfold.md for the six keys this driver "
            f"takes.")

    named = {k.lower(): v for k, v in parser.items(section)}
    unknown = sorted(set(named) - {k.lower() for k in DOWNFOLD_DEFAULTS})
    if unknown:
        raise ValueError(
            f"downfold: {filename} names {len(unknown)} key(s) this driver "
            f"does not have: {unknown}.  Unknown keys are REFUSED rather "
            f"than ignored — a misspelt 'downfold_rcond' that silently keeps "
            f"the default is exactly the quiet wrong answer this initiative "
            f"is built to prevent.  The whole schema is "
            f"{sorted(DOWNFOLD_DEFAULTS)}; docs/downfold.md documents each "
            f"one.")

    out: dict = {}
    for key, default in DOWNFOLD_DEFAULTS.items():
        if key not in named:
            out[key] = default
            continue
        raw = named[key]
        if key in _NORMALIZE_STR:
            raw = raw.strip().lower()
        if isinstance(default, bool):
            out[key] = parser.getboolean(section, key)
        elif isinstance(default, int) and not isinstance(default, bool):
            out[key] = parser.getint(section, key)
        elif isinstance(default, float):
            out[key] = parser.getfloat(section, key)
        elif default is None:
            out[key] = raw.strip()
        else:
            out[key] = raw.strip()
    out["_named_keys"] = frozenset(named)
    return out


@dataclass(frozen=True)
class DownfoldConfig:
    """The parsed, validated downfold input.  Every field is resolved."""

    source_restart: str
    output_restart: str
    parent_centroids_file: str
    band_range_left: tuple[int, int]
    band_range_right: tuple[int, int]
    window_spelling: str
    mu_small: int | str
    downfold_rcond: float
    downfold_select_tol: float | None
    mode: str
    plan: str
    report_residual: bool
    residual_refuse_above: float | None
    input_path: str

    # -- construction -----------------------------------------------------

    @classmethod
    def from_input_file(cls, filename: str, *, print_fn=print
                        ) -> "DownfoldConfig":
        params = read_downfold_input(filename)
        input_dir = os.path.dirname(os.path.abspath(filename)) or "."

        def _path(key, *, required):
            raw = str(params.get(key) or "").strip()
            if not raw:
                if required:
                    raise ValueError(
                        f"downfold: '{key}' has no default and this input "
                        f"file does not set it.  {_KEY_HELP[key]}")
                return ""
            return raw if os.path.isabs(raw) else os.path.normpath(
                os.path.join(input_dir, raw))

        source = _path("source_restart", required=True)
        output = _path("output_restart", required=True)
        centroids = _path("parent_centroids_file", required=False)

        # -- the window, in exactly one spelling --------------------------
        n_val = params["n_val"]
        n_cond = params["n_cond"]
        n_val = int(n_val) if str(n_val).strip() not in ("", "None") else None
        n_cond = int(n_cond) if str(n_cond).strip() not in ("", "None") else None
        br_l = str(params["band_range_left"] or "").strip()
        br_r = str(params["band_range_right"] or "").strip()
        by_counts = n_val is not None or n_cond is not None
        by_ranges = bool(br_l) or bool(br_r)
        if by_counts and by_ranges:
            raise ValueError(
                "downfold: the retained band window is given TWICE — both "
                "n_val/n_cond and band_range_left/right are set.  Use one "
                "spelling.  n_val/n_cond is the valence/conduction shorthand "
                "for left=(0, n_val), right=(n_val, n_val+n_cond); "
                "band_range_* states absolute half-open ranges directly and "
                "is the only spelling that can express an asymmetric window.")
        if not by_counts and not by_ranges:
            raise ValueError(
                "downfold: no retained band window.  This is the key that "
                "decides what the compression is faithful TO, so it has no "
                "default: set n_val and n_cond, or band_range_left and "
                "band_range_right ('lo:hi', absolute, half-open).  A basis "
                "selected against one window and consumed on another is the "
                "measured 2.8 eV failure recorded in "
                "docs/dev/isdf_basis_adequacy_at_large_nband.md.")
        if by_counts:
            if n_val is None or n_cond is None:
                raise ValueError(
                    f"downfold: the n_val/n_cond spelling needs BOTH "
                    f"(got n_val={n_val!r}, n_cond={n_cond!r}).")
            left = (0, int(n_val))
            right = (int(n_val), int(n_val) + int(n_cond))
            spelling = f"n_val={n_val}, n_cond={n_cond}"
        else:
            if not br_l or not br_r:
                raise ValueError(
                    "downfold: the band_range spelling needs BOTH "
                    "band_range_left and band_range_right.  They are the two "
                    "legs of the pair density rho_mn = conj(psi_m) psi_n; a "
                    "symmetric BSE window sets them equal.")
            left = _parse_range(br_l, "band_range_left")
            right = _parse_range(br_r, "band_range_right")
            spelling = f"band_range_left={br_l}, band_range_right={br_r}"

        # -- mu_small ------------------------------------------------------
        raw_mu = str(params["mu_small"] or "").strip().lower()
        if not raw_mu:
            raise ValueError(
                "downfold: 'mu_small' has no default and this input file "
                "does not set it.  " + _KEY_HELP["mu_small"])
        if raw_mu == "auto":
            mu_small: int | str = "auto"
        else:
            try:
                mu_small = int(raw_mu)
            except ValueError as exc:
                raise ValueError(
                    f"downfold: mu_small={raw_mu!r} is neither an integer nor "
                    f"'auto'.") from exc

        # -- conditioning --------------------------------------------------
        rcond = float(params["downfold_rcond"])
        if not (0.0 < rcond < 1.0):
            raise ValueError(
                f"downfold: downfold_rcond={rcond!r} must lie in (0, 1).  It "
                f"is a RELATIVE threshold on lambda/lambda_max — the "
                f"reciprocal of the amplification cap the truncated "
                f"pseudo-inverse may reach — not an absolute eigenvalue.")
        raw_tol = str(params["downfold_select_tol"] or "").strip()
        select_tol = float(raw_tol) if raw_tol else None
        if select_tol is not None and not (0.0 < select_tol < 1.0):
            raise ValueError(
                f"downfold: downfold_select_tol={select_tol!r} must lie in "
                f"(0, 1); it is relative to the largest initial Gram "
                f"diagonal.")
        raw_refuse = str(params["residual_refuse_above"] or "").strip()
        refuse_above = float(raw_refuse) if raw_refuse else None

        # -- mode / plan ---------------------------------------------------
        mode = str(params["mode"]).strip().lower()
        if mode == "refit":
            raise ValueError(
                "downfold: mode='refit' is not implemented in stage 1, and "
                "it is refused rather than quietly demoted to 'cur' so that "
                "nobody reads a CUR result as a refit one.\n"
                "  WHAT 'refit' WOULD MEAN: run k-means and pivoted Cholesky "
                "afresh against the narrow window to get centroid POSITIONS "
                "that are optimal for it, then run a second full zeta fit to "
                "produce zeta^S, then rebuild psi at the new points from "
                "WFN.h5.\n"
                "  WHY 'cur' SERVES THE REQUEST: the owner's phrasing — call "
                "the centroid code with however many centroids are requested, "
                "do the zeta fitting on those centroids — is served by CUR "
                "selection plus the transfer solve.  Pivoted Cholesky IS the "
                "centroid code, run here against the retained window; the "
                "least-squares transfer T = S_SS^-1 S_cross IS the fit, done "
                "in the pair-density metric that preserves the observable "
                "rather than in the zeta metric that preserves the field "
                "uniformly over all of space, including the vacuum the "
                "retained bands have no amplitude in.  The measured case for "
                "it: the parent's own 480-point k-means set certifies 194 of "
                "the 196 directions a 20-band window contains, so a re-fit "
                "would buy at most 1 % more rank for the price of a second "
                "zeta fit.\n"
                "  If you want 'refit' anyway, that is a real and buildable "
                "stage and this key is where it lands — say so and it gets "
                "scoped.")
        if mode != "cur":
            raise ValueError(
                f"downfold: mode={mode!r} is not one of 'cur' (the "
                f"recommended and only implemented route) or 'refit' "
                f"(reserved, refused in stage 1).")
        plan = str(params["plan"]).strip().lower()
        if plan == "distributed":
            raise ValueError(
                "downfold: plan='distributed' — mu tiled over a 2-D process "
                "grid through distrib_la — is a later stage.  It is refused "
                "rather than demoted to 'local', because a block-cyclic "
                "factorisation is a different (equally valid) numerical "
                "gauge and silently changing gauge under an explicit request "
                "is the behaviour this repo's demotion doctrine forbids.  "
                "The local plan emits ZERO collectives in its linear algebra "
                "and is bit-identical across process grids, which is what "
                "stage 1's gates are built on.")
        if plan not in ("auto", "local"):
            raise ValueError(
                f"downfold: plan={plan!r} is not one of 'auto', 'local' "
                f"(both = the local/GSPMD plan in stage 1) or 'distributed' "
                f"(a later stage).")

        if os.path.abspath(source) == os.path.abspath(output):
            raise ValueError(
                f"downfold: source_restart and output_restart are the same "
                f"path ({source}).  The small bundle is named by its centroid "
                f"count, so writing it beside the parent would leave two "
                f"isdf_tensors_*.h5 in one directory and every consumer that "
                f"globs for one would have to guess.  Give the output its own "
                f"directory.")

        cfg = cls(
            source_restart=source, output_restart=output,
            parent_centroids_file=centroids,
            band_range_left=left, band_range_right=right,
            window_spelling=spelling, mu_small=mu_small,
            downfold_rcond=rcond, downfold_select_tol=select_tol,
            mode=mode, plan=plan,
            report_residual=bool(params["report_residual"]),
            residual_refuse_above=refuse_above,
            input_path=os.path.abspath(filename),
        )
        return cfg

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        tol = ("sqrt(eps) (kernel default)"
               if self.downfold_select_tol is None
               else f"{self.downfold_select_tol:g}")
        lines = [
            f"  input file        {self.input_path}",
            f"  source_restart    {self.source_restart}",
            f"  output_restart    {self.output_restart}",
            f"  retained window   {self.window_spelling}"
            f"  ->  left [{self.band_range_left[0]}, "
            f"{self.band_range_left[1]}) x right "
            f"[{self.band_range_right[0]}, {self.band_range_right[1]})",
            f"  mu_small          {self.mu_small}",
            f"  downfold_rcond    {self.downfold_rcond:g}"
            f"  (kappa_cap = {1.0 / self.downfold_rcond:.3e})",
            f"  select_tol        {tol}   <- NOT the same knob as rcond",
            f"  mode / plan       {self.mode} / {self.plan}",
            f"  report_residual   {self.report_residual}",
        ]
        if self.parent_centroids_file:
            lines.append(f"  parent centroids  {self.parent_centroids_file}")
        if self.residual_refuse_above is not None:
            lines.append(
                f"  refuse eps_W >    {self.residual_refuse_above:g}")
        return "\n".join(lines)


_KEY_HELP = {
    "source_restart": (
        "Point it at the finished GW run: either the run directory (the "
        "driver finds tmp/isdf_tensors_*.h5 inside it) or that .h5 file "
        "directly."),
    "output_restart": (
        "Name a directory for the small bundle; the driver writes "
        "<dir>/tmp/isdf_tensors_<mu_S>.h5, which is the layout every BSE "
        "consumer already looks for."),
    "mu_small": (
        "Set it to an integer, or to 'auto' meaning 'as many centroids as "
        "the retained window has independent pair-density directions at "
        "downfold_rcond'.  It has no default because the right number "
        "depends on the window, and a 20-band window holds about 190 "
        "directions on every deck measured so far — not the ~500 the "
        "original design assumed."),
}
