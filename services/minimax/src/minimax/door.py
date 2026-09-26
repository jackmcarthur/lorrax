"""The runtime surface: compute a rule and say what it is.

Every rule is solved in process at run time (owner, 2026-09-16).
:func:`serve` checks the request against the declared vocabulary, calls
:func:`solve_uncertified`, and announces each distinct request once with
its node count, achieved error, Σ|w| and κ₀.  Node positions can differ
in the last digits between hosts, because the solve runs on the host's
LAPACK; two rules are compared by node count and error, not byte for byte.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import numpy as np

from minimax.records import Quadrature, runtime_provenance
from minimax.refusals import (
    SamplingUnsupported,
    UncertifiedSolveRefused,
    UnknownTarget,
)
from minimax.targets import (CHARACTERS, FAMILIES, TARGETS,
                             families_for_character)

_SERVE_ANNOUNCED: set[str] = set()


def reset_announcements() -> None:
    """Test hook: forget what has already been announced."""
    _SERVE_ANNOUNCED.clear()


# ---------------------------------------------------------------------------
#  Vocabulary checks — F3, and the family/target compatibility rule
# ---------------------------------------------------------------------------

def _resolve(family: str, target: str) -> tuple[Any, str | None]:
    """``(FamilySpec, target_kind)``, or F3.

    ``target_kind`` is the crossing family's regularization discriminant
    (``'hgl'`` / ``'fermi'``) and ``None`` everywhere else.  It selects the
    target function the crossing solver fits, so a wrong mapping would fit
    the wrong regularization.
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
            + (f"{declared[0].description}" if declared else ""))
    return live[0].name


# ---------------------------------------------------------------------------
#  serve: every rule is computed here, at run time
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

    For the exponential-sum families it is
    ``max over u in [1,R] of u * sum_l |w_l| e^{-t_l u}``, normalised
    against the pure-damping envelope 1/u.  For the crossing family it is
    Σ|α|, the quantity measured moving by three orders of magnitude between
    hosts.  One name, two definitions, stated rather than blurred.
    """
    w = np.asarray(w)
    if family == "crossing":
        return _sum_abs_w(w)
    return noncrossing_kappa0(tau, w, range_value)


def _announce_solved(quad: Quadrature, sum_abs_w: float,
                     n_max: int) -> None:
    """Every rule this service serves, said out loud once per request."""
    key = (f"SOLVE|{quad.family}|{quad.target}|"
           f"{quad.range_value!r}|{quad.error_bound!r}")
    if key in _SERVE_ANNOUNCED:
        return
    _SERVE_ANNOUNCED.add(key)
    kappa = "unrecorded" if quad.kappa0 is None else f"{quad.kappa0:.4g}"
    met = (quad.max_error is not None and quad.error_bound is not None
           and float(quad.max_error) <= float(quad.error_bound))
    verdict = ("met its target" if met else
               f"MISSED its target: the n_max={int(n_max)} rule's max_err "
               f"exceeds {quad.error_bound:.0e}")
    warnings.warn(
        f"minimax: SOLVED {quad.family}/{quad.target} "
        f"{quad.range_param}={quad.range_value:g} target "
        f"{quad.error_bound:.0e} n_max={int(n_max)} -> "
        f"{quad.node_count} nodes, max_err {quad.max_error:.4g}, "
        f"sum|w| {sum_abs_w:.4g}, kappa0 {kappa} | "
        f"{quad.provenance.one_line()}.  Solved here at run time; it "
        f"{verdict}.  Node positions can differ in the last digits between "
        f"hosts because the solve goes through this machine's LAPACK: "
        f"compare two rules by their node count and error, not byte for "
        f"byte.",
        RuntimeWarning, stacklevel=3)


def serve(*, family: str, target: str, range_value: float,
          error_bound: float, n_max: int, **family_kw) -> Quadrature:
    """Compute the rule this request asks for, here, now.

    Every placement is computed at run time (owner, 2026-09-16).  A
    ``noncrossing`` solve costs 27-129 ms (median 44) per request.
    """
    _resolve(family, target)                      # refuses an unknown request
    unknown = sorted(set(family_kw) - {"eps_q", "omega_hat"})
    if unknown:
        # A selector this door does not understand is refused, never dropped
        # (TASTE 13); the retired `use_shipped` selector refuses by name.
        raise UnknownTarget(
            f"minimax: serve() takes no {unknown} selector; `use_shipped` is "
            f"retired: every rule is computed at run time.")
    return solve_uncertified(
        family=family, target=target, range_value=range_value,
        error_bound=error_bound, n_max=n_max,
        eps_q=family_kw.get("eps_q"), omega_hat=family_kw.get("omega_hat"))


# ---------------------------------------------------------------------------
#  The in-process solvers
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
    # The levelled Remez rule: smallest N whose best N-term error meets
    # the target, certified by alternation, in milliseconds.  It replaced a
    # VarPro+Lawson ladder that measured 2-4x above the best error at its
    # N, used 1-2 extra nodes at eps >= 1e-8 and up to 24 extra (negative
    # weights, kappa0 4.8e3) at 1e-10, and took 0.1-45 s per request; that
    # ladder is deleted (runs/DEV/326).  The cache payload is unchanged, so
    # a warm cache keeps serving old entries until it is cleared.
    from minimax import levelled as _levelled          # noqa: PLC0415
    tau, w, _n, err = _levelled.noncrossing_levelled(
        float(np.exp(logR_key)), float(target_key), N_max=max_nodes)
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

    A zero tolerance is not a tight request, it is an UNSATISFIABLE one.
    An N-ladder solver exits early on ``err < eps``, which no rule can meet
    at eps=0, so it silently runs the whole ladder to N_max and returns the
    last rule -- measured at 18+ minutes on the sodium SC deck, where it
    looked like a hang.

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
            f"minimax: family {family!r} has no in-process solver.")

    quad = Quadrature(
        nodes=tau, weights=w, family=family, target=target,
        range_param=spec.range_param, range_value=float(range_value),
        error_bound=float(error_bound), max_error=float(err),
        kappa0=_kappa0(family, tau, w, range_value), kappa1=None,
        provenance=prov)
    _announce_solved(quad, _sum_abs_w(w), int(n_max))
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
