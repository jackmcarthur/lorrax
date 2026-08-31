"""The runtime surface: look a rule up, or serve one and say what it is.

R1, in one sentence: *the door serves certified shipped tables and refuses
gaps by name; runtime solving is not a service path.*

R1 SHIPS IN TWO STAGES, and this module is stage 1.  The reason is a
measurement, not a preference: the imaginary-axis family has **no shipped
tables at all** — ``catalog.json`` carries 31 entries in two families,
``{'crossing': 5, 'noncrossing': 26}``, and ``build_imag_quadrature``, on
the path of every GN-PPM run there has ever been, is a 100% runtime-solve
path with nothing behind it.  Applied literally today, lookup-and-refuse
refuses the default production deck.  So:

* **Stage 1 (this commit).**  The refusal machinery, the provenance
  record and the certified-table path all ship.  The
  ``LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE`` env key
  gates the in-process solve and **defaults to on**, so no deck changes
  behaviour in a refactor commit — but every solve now prints one loud
  line naming the request, the achieved error, the measured Σ|w| and κ₀,
  and the words *uncertified, not reproducible across hosts*.
* **Stage 2 (after the catalog closes).**  The default flips to refuse.
  The gate is the WP1 request census, which has now been taken: 54
  distinct requests, 51 served, **3 refusals, all closed by ~18
  noncrossing_imag entries.**  The flip is a one-line default change and
  is deliberately NOT in this branch.

The design's own risk register calls the comfortable failure mode by name:
the hatch stays on forever and we get a better-logged version of today.
The defence is that it announces on every distinct use, so "we never armed
stage 2" is visible in every log rather than discoverable only by reading
a design document.
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Any

import numpy as np

from minimax import _catalog as _cat
from minimax.records import Quadrature, runtime_provenance
from minimax.refusals import (
    AmplificationCap,
    NoCertifiedTable,
    SamplingUnsupported,
    UncertifiedSolveRefused,
    UnknownTarget,
    no_certified_table_text,
)
from minimax.targets import (CHARACTERS, FAMILIES, TARGETS,
                             families_for_character)

#: The R1 stage-1 escape hatch.  Default ON.  Stage 2 flips this default;
#: nothing else about the service changes when it does, which is what
#: "the staging is a default value, not an architecture" means.
RUNTIME_SOLVE_ENV = "LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE"

_SERVE_ANNOUNCED: set[str] = set()


def runtime_solve_allowed() -> bool:
    """Is the R1 escape hatch open?  Stage 1 default: yes."""
    v = os.environ.get(RUNTIME_SOLVE_ENV, "1").strip().lower()
    return v not in {"0", "false", "no", "off", ""}


def reset_announcements() -> None:
    """Test hook: forget what has already been announced."""
    _SERVE_ANNOUNCED.clear()


# ---------------------------------------------------------------------------
#  Vocabulary checks — F3, and the family/target compatibility rule
# ---------------------------------------------------------------------------

def _resolve(family: str, target: str) -> tuple[Any, str | None]:
    """``(FamilySpec, target_kind)``, or F3.

    ``target_kind`` is the catalog's discriminant for the crossing family
    (``'hgl'`` / ``'fermi'``) and ``None`` everywhere else, because no
    other family's rows carry one.  Getting that mapping wrong would make
    every crossing lookup match the first row of the wrong regularization,
    which is precisely the class of bug a declared vocabulary exists to
    make impossible.
    """
    spec = FAMILIES.get(family)
    if spec is None:
        raise UnknownTarget(
            f"minimax: unknown family {family!r}.  Declared families: "
            f"{sorted(FAMILIES)}.")
    if target not in TARGETS:
        raise UnknownTarget(
            f"minimax: unknown target {target!r} for family {family!r}.  "
            f"Declared targets: {sorted(TARGETS)}.")
    if spec.name == "crossing":
        if target not in ("hgl", "fermi"):
            raise UnknownTarget(
                f"minimax: family 'crossing' regularizes with target_kind "
                f"'hgl' or 'fermi'; {target!r} is neither.")
        return spec, target
    if target != spec.target:
        raise UnknownTarget(
            f"minimax: family {family!r} approximates {spec.target!r} "
            f"({TARGETS[spec.target].definition}); it cannot serve "
            f"{target!r}.")
    return spec, None


def family_for_character(character: str) -> str:
    """The family serving an analytic character of ``z``, or F6.

    R4's 2×2 dispatch, over declarative data.  Three cells resolve; the
    strip cell — both parts of ``z`` nonzero, which is where MPA lives —
    refuses, because ``damped_line`` does not exist yet.  That refusal is
    ``gw/screening.py:527-531`` moved to the place that can name what is
    missing and what would create it.
    """
    if character not in CHARACTERS:
        raise SamplingUnsupported(
            f"minimax: {character!r} is not an analytic character of z.  "
            f"The four cells are {list(CHARACTERS)} "
            f"(z = omega + i*varpi; each part is zero or not).")
    live = [f for f in families_for_character(character) if f.wired]
    if not live:
        declared = families_for_character(character)
        raise SamplingUnsupported(
            f"minimax: no live quadrature family serves character "
            f"{character!r}.  Declared for that cell: "
            f"{[f.name for f in declared] or 'nothing'}.  "
            + (f"{declared[0].generator_hint}" if declared else ""))
    return live[0].name


# ---------------------------------------------------------------------------
#  The catalog, enumerable
# ---------------------------------------------------------------------------

def catalog(name: str = "catalog.json"):
    """The shipped bundle as an enumerable, assertable view.  No solve."""
    return _cat.catalog_view(name)


# ---------------------------------------------------------------------------
#  F1/F2/F4 — lookup, and only lookup
# ---------------------------------------------------------------------------

def _check_amplification(entry, view) -> None:
    """F2 — the declared cap, read off the catalog's own shipping rule.

    The number is DATA and it lives in the artifact: the theory plan owns
    it (κ₀ ≤ 2 normal, 2–4 a versioned exception, > 4 rejected) and the
    generator stamps it, so a constant in this file would be a second
    opinion nobody asked for.  A v1 entry declares no κ₀ and no shipping
    rule; there is nothing to check and the door says so in the
    provenance rather than inventing a number.
    """
    if entry.kappa0 is None:
        return
    rejected_above = view.shipping_rule.get("rejected_above")
    if rejected_above is None:
        return
    if entry.kappa0 > float(rejected_above):
        raise AmplificationCap(
            f"minimax: shipped table {entry.file!r} declares kappa0 = "
            f"{entry.kappa0:.4g}, above the catalog's own rejection "
            f"threshold of {float(rejected_above):.4g} "
            f"({view.shipping_rule.get('kappa0_definition', 'amp')})."
            f"  A rule whose weights amplify that hard is not servable: the "
            f"quadrature error is not the error you get.  Regenerate the "
            f"entry under the cap, or take the versioned exception "
            f"explicitly.")


def _lookup_beta_axis(spec, target: str, *, range_value: float,
                      error_bound: float, n_max: int,
                      beta: float, beta_clause: str) -> Quadrature:
    """``complex_laplace``, through the axis that does not round.

    THIS FAMILY DOES NOT USE ``_catalog.select_entry`` AND THAT IS THE
    POINT.  The generic rule matches on three axes that all round safely —
    ``R`` up, the tier down, the node count as a ceiling — and ``beta``
    rounds neither way, because ``1/(u - i beta)`` is a different function
    at every ``beta``.  So the rule for this family lives in
    :mod:`minimax.beta_selector`, which matches ``beta`` against each
    entry's own measured tolerance band and refuses when nothing covers
    the request.  Routing here is what makes ``wired=True`` safe: the
    family is selectable, and it is still impossible to select one of its
    tables without saying which ``beta`` and which clause you meant.
    """
    from minimax import beta_selector as _beta          # noqa: PLC0415

    picked = _beta.select(range_value=float(range_value), beta=float(beta),
                          beta_clause=str(beta_clause),
                          target_error=float(error_bound),
                          max_nodes=int(n_max))
    if isinstance(picked, _beta.TableRefusal):
        # The selector's refusals already carry what F1 promises: the
        # request, the nearest certified beta and its band, and the two
        # levers.  Re-wording them here would be a second, worse copy.
        raise NoCertifiedTable(picked.message)

    entry = picked.entry
    typed = _cat.CatalogEntry(
        index=-1, family=spec.name, range_max=float(entry["range_max"]),
        error_bound=float(entry["error_bound"]),
        node_count=int(entry["node_count"]), file=str(entry["file"]),
        range_param=spec.range_param, target_kind=None, eps_q=None,
        claimed_max_error=None, kappa0=entry.get("kappa0"),
        certified=bool(entry.get("certified", False)),
        catalog_name=spec.catalog, raw=entry)
    prov = _cat.provenance_for(
        typed, f"sha256:{str(entry.get('payload_sha256', 'unrecorded'))}",
        _cat.load_catalog_dict(spec.catalog))
    quad = Quadrature(
        nodes=picked.tau, weights=picked.alpha, family=spec.name,
        target=target, range_param=spec.range_param,
        range_value=float(range_value), error_bound=float(error_bound),
        # The error MEASURED at the request's beta, not the entry's own
        # stamp: a number measured somewhere else is not a measurement of
        # this request.  `beta_selector.TableSelection` says the same.
        max_error=float(picked.modulus_error),
        kappa0=(float(entry["kappa0"]) if entry.get("kappa0") is not None
                else None),
        kappa1=None, provenance=prov)
    _check_amplification(typed, catalog(spec.catalog))
    _announce(quad)
    return quad


def lookup(*, family: str, target: str, range_value: float,
           error_bound: float, n_max: int, **family_kw) -> Quadrature:
    """The whole runtime surface.  Refuses (F1–F4) or returns.  NEVER solves.

    ``family_kw`` carries the family's own discriminants — ``eps_q`` for
    the crossing family, ``beta``/``beta_clause`` for ``complex_laplace``.
    Unknown keywords are refused rather than ignored: a silently-dropped
    selector is how you serve a table fitted to a different function.
    """
    spec, target_kind = _resolve(family, target)

    if spec.name == "complex_laplace":
        beta = family_kw.pop("beta", None)
        beta_clause = family_kw.pop("beta_clause", None)
        if family_kw:
            raise UnknownTarget(
                f"minimax: family {family!r} takes no {sorted(family_kw)} "
                f"selector.  Silently ignoring one would serve a table "
                f"fitted to a different function.")
        if beta is None or beta_clause is None:
            raise UnknownTarget(
                f"minimax: family {family!r} is selected on (R, beta, "
                f"clause, tier, nodes) and this request names "
                + ("no beta" if beta is None else "no beta_clause") + ".\n"
                "  beta does not round -- the target is a different "
                "function at every beta -- and the two clauses of the "
                "envelope (width = Gamma_p/x_min, height = varpi/x_min) "
                "OVERLAP near 0.6, so neither can be inferred from the "
                "other or from the number alone.  The caller says which, "
                "because only the caller knows what its numerator was.")
        return _lookup_beta_axis(
            spec, target, range_value=range_value, error_bound=error_bound,
            n_max=n_max, beta=beta, beta_clause=beta_clause)

    eps_q = family_kw.pop("eps_q", None)
    if family_kw:
        raise UnknownTarget(
            f"minimax: family {family!r} takes no {sorted(family_kw)} "
            f"selector.  Silently ignoring one would serve a table fitted "
            f"to a different function.")

    if not spec.wired:
        raise NoCertifiedTable(
            f"minimax: family {family!r} is declared but NOT WIRED into the "
            f"selection rule"
            + (" (its entries are staged in the bundle; the selection rule "
               "has no beta axis yet, and a beta-blind match would serve a "
               "table fitted to a different function)"
               if spec.shipped else " and ships no tables")
            + f".  {spec.generator_hint}")

    view = catalog(spec.catalog)
    entry = _cat.select_entry(
        view.entries, family,
        range_value=range_value, target_error=error_bound,
        max_nodes=n_max, target_kind=target_kind, eps_q=eps_q)

    if entry is None:
        below = _cat.nearest_below(
            view.entries, family, range_value=range_value,
            target_error=error_bound, target_kind=target_kind, eps_q=eps_q)
        extra = ""
        if not spec.shipped:
            extra = (f"the {family!r} family has ZERO shipped entries — this "
                     f"is a structural hole, not a straggler")
        raise NoCertifiedTable(no_certified_table_text(
            family=family, target=target, range_param=spec.range_param,
            range_value=range_value, error_bound=error_bound, n_max=n_max,
            nearest_below=below, range_lever=spec.range_lever,
            generator_hint=spec.generator_hint, extra=extra))

    _check_amplification(entry, view)
    tau, alpha, max_error, kappa0, table_hash = _cat.load_table(entry)
    prov = _cat.provenance_for(entry, table_hash,
                               _cat.load_catalog_dict(spec.catalog))
    quad = Quadrature(
        nodes=tau, weights=alpha, family=family, target=target,
        range_param=spec.range_param, range_value=float(range_value),
        error_bound=float(error_bound), max_error=float(max_error),
        kappa0=kappa0, kappa1=None, provenance=prov)
    _announce(quad)
    return quad


def nearest_certified(*, family: str, target: str, range_value: float,
                      error_bound: float, n_max: int,
                      **family_kw) -> Quadrature | None:
    """What a refusal offers.  Pure catalog algebra; NEVER raises."""
    try:
        spec, target_kind = _resolve(family, target)
        view = catalog(spec.catalog)
        entry = _cat.nearest_below(
            view.entries, family, range_value=range_value,
            target_error=error_bound, target_kind=target_kind,
            eps_q=family_kw.get("eps_q"))
        if entry is None:
            return None
        tau, alpha, max_error, kappa0, table_hash = _cat.load_table(entry)
        return Quadrature(
            nodes=tau, weights=alpha, family=family, target=target,
            range_param=spec.range_param, range_value=entry.range_max,
            error_bound=entry.error_bound, max_error=float(max_error),
            kappa0=kappa0, kappa1=None,
            provenance=_cat.provenance_for(
                entry, table_hash, _cat.load_catalog_dict(spec.catalog)))
    except Exception:                                  # noqa: BLE001
        # DELIBERATELY BROAD, and it is the one broad catch in this
        # service.  The contract is "never raises", because this function's
        # only caller is a refusal message: an exception raised while
        # building the explanation for another exception replaces a useful
        # refusal with a confusing one.  Every failure mode it can swallow
        # is one the ORIGINAL refusal already reported.
        return None


# ---------------------------------------------------------------------------
#  R1 stage 1 — serve: look up, or solve behind the announced escape hatch
# ---------------------------------------------------------------------------

def _sum_abs_w(w: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(w))))


def noncrossing_kappa0(
    tau: np.ndarray,
    weights: np.ndarray,
    range_value: float,
    *,
    n_eval: int = 512,
) -> float:
    """Return the service-owned noncrossing amplification functional."""
    tau = np.asarray(tau, dtype=np.float64)
    weights = np.asarray(weights)
    upper = max(float(range_value), 1.0 + 1.0e-12)
    u = np.geomspace(1.0, upper, int(n_eval))
    envelope = u * np.sum(
        np.abs(weights)[None, :] * np.exp(-tau[None, :] * u[:, None]),
        axis=1,
    )
    return float(np.max(envelope))


def _kappa0(family: str, tau: np.ndarray, w: np.ndarray,
            range_value: float) -> float:
    """The amplification metric, per family.

    For the exponential-sum families this is the catalog's own definition
    (``max over u in [1,R] of u * sum_l |w_l| e^{-t_l u}``, normalised
    against the pure-damping envelope 1/u).  For the crossing family it is
    Σ|α| — which is the quantity survey §2.4 measured moving by three
    orders of magnitude between hosts, and the quantity the shipped
    crossing tables are compared on.  One name, two definitions, stated
    rather than blurred.
    """
    w = np.asarray(w)
    if family == "crossing":
        return _sum_abs_w(w)
    return noncrossing_kappa0(tau, w, range_value)


def _announce(quad: Quadrature) -> None:
    """R2: every table served announces its origin ONCE.

    Once per distinct (request, source) — not once per call.  A quadrature
    request repeats per q-block per SCF iteration per rank, and an
    announcement nobody can read is the same as no announcement.
    """
    key = (f"{quad.family}|{quad.target}|{quad.range_value!r}|"
           f"{quad.error_bound!r}|{quad.provenance.source}")
    if key in _SERVE_ANNOUNCED:
        return
    _SERVE_ANNOUNCED.add(key)
    warnings.warn(quad.one_line(), RuntimeWarning, stacklevel=3)


def _announce_uncertified(quad: Quadrature, sum_abs_w: float,
                          n_max: int) -> None:
    """The escape hatch, said out loud.  Once per distinct request."""
    key = (f"SOLVE|{quad.family}|{quad.target}|"
           f"{quad.range_value!r}|{quad.error_bound!r}")
    if key in _SERVE_ANNOUNCED:
        return
    _SERVE_ANNOUNCED.add(key)
    kappa = "unrecorded" if quad.kappa0 is None else f"{quad.kappa0:.4g}"
    warnings.warn(
        f"minimax: UNCERTIFIED SOLVE {quad.family}/{quad.target} "
        f"{quad.range_param}={quad.range_value:g} target "
        f"{quad.error_bound:.0e} n_max={int(n_max)} -> "
        f"{quad.node_count} nodes, max_err {quad.max_error:.4g}, "
        f"sum|w| {sum_abs_w:.4g}, kappa0 {kappa} | "
        f"{quad.provenance.one_line()} -- UNCERTIFIED, NOT REPRODUCIBLE "
        f"ACROSS HOSTS.  No shipped table matched this request, so the rule "
        f"was computed in-process by a SciPy optimiser whose answer depends "
        f"on this machine's LAPACK (three hosts, three answers, kappa0 "
        f"varying by 900x -- survey section 2.4).  {RUNTIME_SOLVE_ENV}=0 "
        f"makes this a refusal instead.",
        RuntimeWarning, stacklevel=3)


def serve(*, family: str, target: str, range_value: float,
          error_bound: float, n_max: int, use_shipped: bool = True,
          **family_kw) -> Quadrature:
    """Look up; on a miss, take the R1 stage-1 escape hatch or refuse (F5).

    This is what production calls.  ``use_shipped=False`` is the deck key
    ``regenerate_minimax_tables`` arriving here — an explicit request for
    the uncertified path, which still announces.
    """
    spec, target_kind = _resolve(family, target)
    eps_q = family_kw.get("eps_q")
    omega_hat = family_kw.get("omega_hat")
    beta_kw = {k: family_kw[k] for k in ("beta", "beta_clause")
               if k in family_kw}

    if use_shipped and spec.wired:
        try:
            return lookup(family=family, target=target,
                          range_value=range_value, error_bound=error_bound,
                          n_max=n_max,
                          **({"eps_q": eps_q} if eps_q is not None else {}),
                          **beta_kw)
        except NoCertifiedTable as miss:
            reason = str(miss)
    else:
        reason = (
            f"the caller disabled the shipped-table path "
            f"(use_shipped={use_shipped})" if not use_shipped
            else f"family {family!r} is not wired into the rule")

    if not runtime_solve_allowed():
        raise UncertifiedSolveRefused(
            f"minimax: {RUNTIME_SOLVE_ENV}=0 and no certified table serves "
            f"this request, so the service refuses rather than computing an "
            f"uncertified rule in-process.\n{reason}")

    return solve_uncertified(
        family=family, target=target, range_value=range_value,
        error_bound=error_bound, n_max=n_max,
        eps_q=eps_q, omega_hat=omega_hat)


# ---------------------------------------------------------------------------
#  The offline solvers, reached only through the hatch
# ---------------------------------------------------------------------------
#  The three wrappers below are carried VERBATIM from
#  `gw.minimax_screening._solve_*_scaled_cached`: the same `lru_cache`
#  sizes, the same key rounding, the same disk-cache payload dicts.  That is
#  not tidiness — the payload dict IS the legacy cache key, so any change
#  here would silently invalidate every warm cache in the fleet and move
#  numbers (including the frozen G2 reference's) inside a refactor commit.

@lru_cache(maxsize=64)
def _solve_noncrossing_scaled_cached(logR_key: float, target_key: float,
                                     max_nodes: int):
    payload = {"solver": "noncrossing", "logR_key": float(logR_key),
               "target_key": float(target_key), "max_nodes": int(max_nodes)}
    from minimax import cache as _cache                # noqa: PLC0415
    cached = _cache.load("noncrossing", payload)
    if cached is not None:
        return cached
    from minimax import solver as _solver              # noqa: PLC0415
    tau, w, _n, err = _solver.noncrossing_grids(
        float(np.exp(logR_key)), float(target_key), N_start=2,
        N_max=max_nodes)
    tau = np.asarray(tau, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    err = float(err)
    _cache.store("noncrossing", payload, tau, w, err)
    return tau, w, err, runtime_provenance(
        _cache._payload_hash(tau, w), _cache.backend_tag())


@lru_cache(maxsize=64)
def _solve_noncrossing_imag_scaled_cached(logR_key: float,
                                          omega_hat_key: float,
                                          target_key: float,
                                          max_nodes: int):
    payload = {"solver": "noncrossing_imag", "logR_key": float(logR_key),
               "omega_hat_key": float(omega_hat_key),
               "target_key": float(target_key), "max_nodes": int(max_nodes)}
    from minimax import cache as _cache                # noqa: PLC0415
    cached = _cache.load("noncrossing_imag", payload)
    if cached is not None:
        return cached
    from minimax import solver as _solver              # noqa: PLC0415
    tau, w, _n, err = _solver.noncrossing_imag_grids(
        float(np.exp(logR_key)), float(omega_hat_key), float(target_key),
        N_start=2, N_max=max_nodes)
    tau = np.asarray(tau, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    err = float(err)
    _cache.store("noncrossing_imag", payload, tau, w, err)
    return tau, w, err, runtime_provenance(
        _cache._payload_hash(tau, w), _cache.backend_tag())


@lru_cache(maxsize=128)
def _solve_crossing_scaled_cached(A_key: float, target_key: float,
                                  max_nodes: int, eps_q_key: float,
                                  target_kind: str):
    payload = {"solver": "crossing", "A_key": float(A_key),
               "target_key": float(target_key), "max_nodes": int(max_nodes),
               "eps_q_key": float(eps_q_key), "target_kind": str(target_kind)}
    from minimax import cache as _cache                # noqa: PLC0415
    cached = _cache.load("crossing", payload)
    if cached is not None:
        return cached
    from minimax import solver as _solver              # noqa: PLC0415
    if target_kind == "hgl":
        G_func, tau_max_func = _solver.G_hgl, _solver.tau_max_hgl
    elif target_kind == "fermi":
        G_func, tau_max_func = _solver.G_fermi, _solver.tau_max_fermi
    else:
        raise UnknownTarget(
            f"minimax: unknown crossing target_kind={target_kind!r}.")
    tau, w, _n, err = _solver.crossing_grids(
        float(A_key), float(target_key), G_func, tau_max_func,
        eps_q=float(eps_q_key), N_max=max_nodes)
    tau = np.asarray(tau, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    err = float(err)
    _cache.store("crossing", payload, tau, w, err)
    return tau, w, err, runtime_provenance(
        _cache._payload_hash(tau, w), _cache.backend_tag())


def _tolerance_key(error_bound: float) -> float:
    """Round a tolerance for the cache key WITHOUT rounding it away.

    ``round(x, 14)`` is the key rounding carried verbatim from the original
    call sites, and it is kept so every stored entry still hits.  But a
    tolerance is a positive number that may legitimately be far below 1e-14
    once rescaled into ``[1, R]`` units (the Laplace bound asks the service
    for ``eps_phys * x_min``), and rounding such a value to 14 decimals
    yields exactly 0.0.

    A zero tolerance is not a tight request, it is an UNSATISFIABLE one:
    ``noncrossing_grids`` exits early on ``err < eps``, which no rule can
    meet at eps=0, so it silently runs the entire N ladder to N_max and
    returns the last rule -- measured at 18+ minutes on the sodium SC deck,
    where it looked like a hang.

    So: keep the decimal key where it is faithful, and fall back to a
    significant-figure key only where it would underflow. Existing cache
    entries are unaffected.
    """
    x = float(error_bound)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(
            f"minimax: tolerance must be finite and positive; got {x!r}. "
            "A non-positive tolerance cannot be met by any rule and would "
            "run the solver's full node ladder for nothing.")
    key = round(x, 14)
    if key <= 0.0:
        key = float(f"{x:.12e}")
    return key


def solve_uncertified(*, family: str, target: str, range_value: float,
                      error_bound: float, n_max: int,
                      eps_q: float | None = None,
                      omega_hat: float | None = None) -> Quadrature:
    """Run the offline solver in-process, and SAY SO.

    The rounding of every cache key below is carried verbatim from the
    pre-extraction call sites, for the reason given above the wrappers.
    """
    spec, target_kind = _resolve(family, target)
    if family == "noncrossing":
        tau, w, err, prov = _solve_noncrossing_scaled_cached(
            round(float(np.log(float(range_value))), 12),
            _tolerance_key(error_bound), int(n_max))
    elif family == "noncrossing_imag":
        if omega_hat is None:
            raise UnknownTarget(
                "minimax: family 'noncrossing_imag' needs omega_hat.")
        tau, w, err, prov = _solve_noncrossing_imag_scaled_cached(
            round(float(np.log(float(range_value))), 12),
            round(float(omega_hat), 12),
            _tolerance_key(error_bound), int(n_max))
    elif family == "crossing":
        tau, w, err, prov = _solve_crossing_scaled_cached(
            round(float(range_value), 12), _tolerance_key(error_bound),
            int(n_max),
            round(float(eps_q if eps_q is not None else 1.0e-3), 12),
            str(target_kind))
    else:
        raise UncertifiedSolveRefused(
            f"minimax: family {family!r} has no in-process solver.  "
            f"{spec.generator_hint}")

    quad = Quadrature(
        nodes=tau, weights=w, family=family, target=target,
        range_param=spec.range_param, range_value=float(range_value),
        error_bound=float(error_bound), max_error=float(err),
        kappa0=_kappa0(family, tau, w, range_value), kappa1=None,
        provenance=prov)
    _announce_uncertified(quad, _sum_abs_w(w), int(n_max))
    return quad


def cached_solve_payload(family: str, **kw) -> dict[str, Any]:
    """The disk-cache payload dict for a request.  Test/diagnostic surface."""
    if family == "noncrossing":
        return {"solver": "noncrossing", "logR_key": float(kw["logR_key"]),
                "target_key": float(kw["target_key"]),
                "max_nodes": int(kw["max_nodes"])}
    if family == "noncrossing_imag":
        return {"solver": "noncrossing_imag",
                "logR_key": float(kw["logR_key"]),
                "omega_hat_key": float(kw["omega_hat_key"]),
                "target_key": float(kw["target_key"]),
                "max_nodes": int(kw["max_nodes"])}
    if family == "crossing":
        return {"solver": "crossing", "A_key": float(kw["A_key"]),
                "target_key": float(kw["target_key"]),
                "max_nodes": int(kw["max_nodes"]),
                "eps_q_key": float(kw["eps_q_key"]),
                "target_kind": str(kw["target_kind"])}
    raise UnknownTarget(f"minimax: no solver payload shape for {family!r}.")
