"""Pure formatting helpers for the production centroid coordinate file and
the ``kmeans.out`` main log (SETTINGS rows and the RESULTS block)."""

from __future__ import annotations

import math
import os

from common.preprocessing_output import results_lines
from common.report_blocks import Setting
from common.scientific_output import band_range

#: The kmeans TIMING table: ``(label, timing sections...)``.
KMEANS_STAGES = (
    ("wavefunction setup", "setup.wfn_io"),
    ("feature weight", "setup.weight"),
    ("weighted k-means", "kmeans"),
    ("orbit snap and unfold", "snap_unfold"),
    ("Lloyd-state release", "release_before_prune"),
    ("Gram and pivoted Cholesky", "prune"),
)


def validate_mode_policy(args) -> None:
    """Refuse current-mode switches that change its physical metric."""
    if int(getattr(args, "prune_q_shells", 0)) < 0:
        raise ValueError("--prune-q-shells must be >= 0")
    if float(getattr(args, "weight_power", 0.5)) <= 0.0:
        raise ValueError("--weight-power must be positive")
    if float(getattr(args, "prune_normalize", 0.0)) < 0.0:
        raise ValueError("--prune-normalize must be >= 0")
    if args.density_mode == "current":
        if int(getattr(args, "prune_q_shells", 0)) > 0:
            raise ValueError(
                "--prune-q-shells is a charge-Gram option; current-mode "
                "selection prunes on the q = 0 transverse Gram")
        if args.no_symmetries:
            raise ValueError(
                "current centroid selection requires the crystal's "
                "symmetries; --no-symmetries is not a physical current-mode "
                "policy")
        if float(args.oversample) <= 1.0:
            raise ValueError(
                "current centroid selection requires transverse-Gram pruning; "
                "--oversample must be greater than 1")


def prune_band_ranges(args, n_val: int, n_cond: int):
    """Resolved pair-density windows, shared by pruning and provenance.

    The windows follow the owner's rule (PAIRS audit, 2026-09-26): left =
    every occupied band plus the deck's Σ conduction window
    ``[0, n_val + ncond)``, right = every band ``[0, n_val + n_cond)``.
    ``args.sigma_ncond`` carries the deck's ``ncond``; when it is ``None``
    (no deck given) the Σ window is unknown and the left leg widens to the
    right one, the full square, a safe superset.  ``--fit-window`` is the one
    override (the ``--prune-window`` choices were removed 2026-10-21; this
    was their default, ``v_x_vc``).  A window that drops an occupied band
    refuses by name.
    """
    top = int(n_val) + int(n_cond)
    fit_window = getattr(args, "fit_window", None)
    if fit_window is not None:
        try:
            fields = str(fit_window).split(",")
            if len(fields) != 2:
                raise ValueError
            left = tuple(int(v) for v in fields[0].split(":"))
            right = tuple(int(v) for v in fields[1].split(":"))
            if len(left) != 2 or len(right) != 2:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--fit-window must be L0:L1,R0:R1 using integer, 0-based "
                "half-open bounds") from exc
        if any(lo < 0 or hi <= lo or hi > top
               for lo, hi in (left, right)):
            raise ValueError(
                f"--fit-window ranges must be nonempty and lie in [0,{top}); "
                f"got left={left}, right={right}")
        from file_io.centroids import assert_windows_keep_occupied
        assert_windows_keep_occupied(
            left, right, int(n_val), where="kmeans --fit-window")
        return left, right, "explicit feature pair"
    sigma_ncond = getattr(args, "sigma_ncond", None)
    if sigma_ncond is None:
        return ((0, top), (0, top),
                "full protected square: no deck, Sigma window unknown")
    left_top = min(top, int(n_val) + int(sigma_ncond))
    return ((0, left_top), (0, top),
            "(occupied + Sigma conduction) x all bands")


def resolve_sigma_ncond(args, wfn, print_fn=print) -> None:
    """Set ``args.sigma_ncond`` from the input file's parameters (``args.sigma_deck``).

    Its ``ncond``, or ``number_bands_protected`` resolved against the WFN as
    gw resolves it (``gw.gw_config.resolve_band_request``); left as is when
    no input file was given.
    """
    deck = getattr(args, "sigma_deck", None)
    if deck is not None:
        from gw.gw_config import resolve_band_request
        args.sigma_ncond = int(resolve_band_request(deck, wfn, print_fn)["ncond"])


class CentroidRankNotEstablished(RuntimeWarning):
    """The point budget stopped the select before the pool was spent."""


def rank_law_estimate(*, left, right, nspinor: int, kgrid, ng: int,
                      ecutwfc: float, ecutrho: float):
    """``(N_mu, L, B, terms)`` of the charge pair-set rank law, or ``None``.

    N_mu(1 meV RMS Σ_x) ≈ 0.5·min(P, 445 L^0.107 B^0.322, 0.97 N_G(4E_wfc))
    (docs/theory/isdf-exchange-accuracy.md#rank-law).  L and B are the left
    and right window widths, per Kramers pair on a spinor WFN; P = N_k·L·B
    bounds the pair functions; N_G(4E_wfc) scales the WFN's density sphere
    ``ng`` at ``ecutrho``.  A Si calibration checked on Fe and Ni 20³.
    """
    per = 2.0 if int(nspinor) == 2 else 1.0
    L = (int(left[1]) - int(left[0])) / per
    B = (int(right[1]) - int(right[0])) / per
    if not (L > 0 and B > 0 and float(ecutrho) > 0 and int(ng) > 0):
        return None
    terms = (math.prod(int(n) for n in kgrid) * L * B,
             445.0 * L ** 0.107 * B ** 0.322,
             0.97 * int(ng) * (4.0 * float(ecutwfc) / float(ecutrho)) ** 1.5)
    return 0.5 * min(terms), L, B, terms


def pool_rank_warning(*, pool: int, written: int, unpicked: float | None,
                      rank_law=None) -> str | None:
    """The warning for a select the point budget stopped, else ``None``.

    ``unpicked`` is the largest never-picked residual over the floor
    (``prune_candidates_by_pivoted_cholesky``); above 1 the pool still held
    directions when the budget ran out, so the pair-space rank is unmeasured.
    """
    if unpicked is None or float(unpicked) <= 1.0:
        return None
    law = ("" if rank_law is None else
           f"  The rank law puts N_mu for 1 meV RMS Sigma_x near "
           f"{rank_law[0]:.0f} (written/estimate = "
           f"{int(written) / rank_law[0]:.2f}).")
    return (f"pair-space rank not established: the point budget stopped the "
            f"pivoted Cholesky at {int(written)} points while unpicked "
            f"candidates of the {int(pool)}-point pool still carry "
            f"{float(unpicked):.3g}x its floor, so {int(written)} centroids "
            f"are not a complete ISDF basis.{law}  Raise the request or the "
            f"pool (--oversample) until the header reads 'stop: pool spent' "
            f"to measure the rank.")


def format_centroid_header(*, feature_fit: str, source_wfn: str,
                           weight_label: str, num_electrons: float,
                           occupied_boundary: int, fft_grid, kgrid, shift,
                           seed: int, requested: int,
                           candidates: int, written: int, pruning: str,
                           prune_rank: int | None, prune_left, prune_right,
                           prune_label: str, orbit_aware: bool, n_sym: int,
                           density_mode: str, pool: int,
                           unpicked: float | None = None,
                           rank_law=None) -> str:
    """Centroid-table provenance, kept pure so the file contract is gated.

    ``rho_power=1`` stays in the ``selection:`` line although ``--rho-power``
    was removed (2026-10-21): the k-means mass is always the measured feature
    norm, and the line keeps its byte layout for existing tables and the
    hsuite reference.
    """
    rank_note = ""
    if prune_rank is not None:
        stop = (f"stop: pool spent, largest unpicked residual "
                f"{float(unpicked):.3g}x the floor; the pair-space rank is at "
                f"least {int(prune_rank)} and a larger pool can raise it"
                if float(unpicked) <= 1.0 else
                f"stop: point budget, largest unpicked residual "
                f"{float(unpicked):.3g}x the floor; the pair-space rank "
                f"exceeds {int(prune_rank)} and is not measured")
        rank_note = (f"\npool rank={int(prune_rank)} of {int(written)} kept "
                     f"pivots on a {int(pool)}-point pool; {stop}")
    if rank_law is not None:
        n_mu, L, B, terms = rank_law
        rank_note += (
            f"\nrank law: N_mu(1 meV RMS Sigma_x) ~ 0.5 min(P, 445 L^0.107 "
            f"B^0.322, 0.97 N_G(4E_wfc)) = 0.5 min({terms[0]:.4g}, "
            f"{terms[1]:.0f}, {terms[2]:.0f}) = {n_mu:.0f} at L={L:g}, "
            f"B={B:g}; written/estimate={int(written) / n_mu:.2f}")
    channel = ("gamma^0 (charge) ISDF" if density_mode == "scalar" else
               "gamma^{1,2,3} (current) ISDF")
    return (
        "LORRAX ISDF centroid coordinates (fractional crystal units)\n"
        f"feature fit: {feature_fit}\n"
        f"source wavefunctions: {os.path.abspath(source_wfn)}\n"
        f"selection metric: {weight_label}\n"
        f"electrons: {float(num_electrons):.8g}; "
        f"occupied-band boundary: {int(occupied_boundary)}\n"
        f"FFT grid: {tuple(fft_grid)}; k grid: {tuple(kgrid)}; "
        f"shift: {tuple(shift)}\n"
        f"selection: weighted k-means; seed={int(seed)}; "
        f"rho_power=1; requested={int(requested)}; "
        f"candidates={int(candidates)}; written={int(written)}\n"
        f"pruning: {pruning}; pair-density windows "
        f"left={tuple(prune_left)}, right={tuple(prune_right)} "
        f"({prune_label}){rank_note}\n"
        f"symmetry closure: {'orbit-aware' if orbit_aware else 'literal points'}; "
        f"spatial operations={int(n_sym)}\n"
        f"coordinates: x y z snapped to FFT grid; {int(written)} unique points\n"
        f"intended channels: {channel}"
    )


def _windows_text(left, right) -> str:
    """0-based half-open band windows as 1-based bands: ``left 1-3 x right 1-7``."""
    return (f"left {band_range(int(left[0]), int(left[1]))} x right "
            f"{band_range(int(right[0]), int(right[1]))}")


def kmeans_settings_rows(args, *, source) -> list:
    """The kmeans SETTINGS rows from the parsed command line; ``source(*flags)``
    says whether a flag was given (``ProductionReport.flag_source``)."""
    oversample = float(args.oversample)
    n_c = int(args.n_centroids)
    m_cand = int(math.ceil(n_c * oversample)) if oversample > 1.0 else n_c
    current = args.density_mode == "current"
    rows = [
        Setting("n_centroids", str(n_c), "cli",
                "point budget: the centroids requested"),
        Setting("oversample", f"{oversample:g}", source("--oversample"),
                f"{m_cand} k-means candidates, pruned by pivoted Cholesky"
                if oversample > 1.0 else "no pivoted-Cholesky pruning"),
        Setting("density_mode", str(args.density_mode), source("--density-mode"),
                "transverse current Gram (gamma^{1,2,3} ISDF)" if current
                else "charge Gram (gamma^0 ISDF)"),
        Setting("seed", str(int(args.seed)), source("--seed"),
                "k-means start; same seed and ranks, same points"),
    ]
    if args.fit_window is not None:
        rows.append(Setting("fit_window", str(args.fit_window), "cli",
                            "left,right band windows, 0-based, half-open"))
    else:
        rows.append(Setting("band windows",
                            "deck ncond" if args.input is not None else "all bands",
                            "deck" if args.input is not None else "default",
                            "left: occupied + deck ncond; right: every WFN band"
                            if args.input is not None else
                            "no deck: every WFN band on both legs"))
    rows.append(Setting("symmetries",
                        "off" if args.no_symmetries else "on",
                        source("--no-symmetries"),
                        "literal points" if args.no_symmetries else
                        "closed under the decorated-atom space group"))
    shells = int(getattr(args, "prune_q_shells", 0))
    if shells:
        rows.append(Setting("prune_q_shells", str(shells), "cli",
                            f"prune Gram summed over q = 0 and {shells} "
                            "nonzero q shells"))
    power = float(getattr(args, "weight_power", 0.5))
    if power != 0.5:
        rows.append(Setting("weight_power", f"{power:g}", "cli",
                            "k-means mass = (diag q=0 Gram)^P"))
    normalize = float(getattr(args, "prune_normalize", 0.0))
    if normalize:
        rows.append(Setting("prune_normalize", f"{normalize:g}", "cli",
                            "prune on D^-G Gram D^-G"))
    return rows


def rank_text(*, prune_rank, written: int, pool: int, unpicked) -> str:
    """What the pivoted Cholesky measured, in one row value."""
    if prune_rank is None:
        return "not measured (no pivoted-Cholesky pruning)"
    if unpicked is not None and float(unpicked) <= 1.0:
        return (f"at least {int(prune_rank)} (pool spent: largest unpicked residual "
                f"{float(unpicked):.3g} x the floor); {int(prune_rank)} of "
                f"{int(written)} points are independent directions")
    return (f"NOT established: the point budget stopped the select at "
            f"{int(written)} points of the {int(pool)}-point pool (largest "
            f"unpicked residual {float(unpicked):.3g} x the floor); "
            f"{int(prune_rank)} kept pivots")


def kmeans_results_lines(*, requested: int, candidates: int, pool: int,
                         written: int, out_file: str, orbit_aware: bool,
                         n_sym: int, windows, window_label: str,
                         density_mode: str, prune_rank, unpicked,
                         rank_law=None) -> list[str]:
    """The kmeans RESULTS block: what was written, the selection funnel, the
    symmetries, the sampling metric and the measured pair-space rank."""
    left, right = windows
    metric = ("transverse current" if density_mode == "current" else "charge")
    rows = [
        ("centroids", f"{int(written)} written to {os.path.basename(str(out_file))} "
                      f"({int(requested)} requested)"),
        ("selection", f"{int(requested)} requested -> {int(candidates)} k-means "
                      f"candidates -> {int(pool)}-point pool -> {int(written)} written"),
        ("symmetries", f"closed under {int(n_sym)} spatial operations "
                       "(decorated atoms); snapped to the FFT grid"
                       if orbit_aware else
                       "none imposed, literal points; snapped to the FFT grid"),
        ("sampling metric", f"sqrt(diag q=0 {metric} Gram) on "
                            f"{_windows_text(left, right)} ({window_label}), "
                            "unit band weights"),
        ("pair-space rank", rank_text(prune_rank=prune_rank, written=written,
                                      pool=pool, unpicked=unpicked)),
    ]
    if rank_law is not None:
        n_mu, L, B, _ = rank_law
        rows.append(("rank law", f"N_mu(1 meV RMS Sigma_x) ~ {n_mu:.0f} at L={L:g}, "
                                 f"B={B:g}; written/estimate {int(written) / n_mu:.2f}"))
    return results_lines(rows)


__all__ = [
    "CentroidRankNotEstablished",
    "format_centroid_header",
    "KMEANS_STAGES",
    "kmeans_results_lines",
    "kmeans_settings_rows",
    "rank_text",
    "pool_rank_warning",
    "prune_band_ranges",
    "rank_law_estimate",
    "validate_mode_policy",
]
