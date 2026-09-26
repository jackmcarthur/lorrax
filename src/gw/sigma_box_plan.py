"""Measure-independent denominator-box quadrature for dynamic Sigma(omega).

The public path in this module is deliberately short:

``physical product window -> denominator box -> rule -> executor nodes``.

Pole fields remain distributed.  MPA supplies the bounded per-pole extrema
returned by :func:`gw.mpa.sigma_windows.summarize_sigma_poles`; PPM supplies
the scalar extrema of each exact ``(q, mu, nu)`` pane.  No residue histogram,
sampled lattice, error apportionment, or campaign-wide selection enters the
quadrature.  The box construction, lower-half-plane conjugation, cache policy,
fit guards, and conversion to executor ``(t, alpha)`` live here once for both
routes.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import pickle
import socket
import time
import uuid
import zipfile

from ffi import _services

_services.ensure_on_path()

import jax.numpy as jnp
import numpy as np

from common import timing
from common.collectives import (all_gather_processes, gather_to_host,
                                process_count, process_rank)
from common.units import RYD_TO_EV
from gw.minimax_screening import MinimaxNodes
from gw.mpa.sigma_windows import SharedSigmaWindow, sigma_pole_edges
from gw.ppm_windows import _SigmaWindow
from gw.scissor import sc_state_pad_ev
from minimax import (
    analytic_line_box_rule,
    UniformRule,
    boundary_samples,
    build_uniform_rule,
    rule_roundoff_amplification,
    uniform_rule_solver_identity,
)


_FACTOR_GROWTH_CAP = 30.0
_RUNTIME_NOISE_EPSILON = 6.0e-8
#: Absolute budget on a rule's runtime-noise bound (roundoff amplification x
#: _RUNTIME_NOISE_EPSILON, :func:`_accept_rule`), in the certificate's own
#: currency.  Roundoff is set by the executor's arithmetic, not by the
#: quadrature, so its budget does not shrink with eps: production's
#: 0.05 x 1e-4 (the same double), which lets eps below 2e-5 certify.
_RUNTIME_NOISE_BUDGET = 5.0e-6
_SC_POLE_PAD_FRACTION = 0.10
_BOX_SIGN_FRACTION = 0.7
#: Pad toward zero for a sign-definite SC window: 0.5 of its distance escaped by 1.6% on TaAs 8^3
#: map 1 and 0.25 again at map 2 (semimetal valence state 30 -> 15 -> 6 meV from E_F); 0.05 floors it below 1 meV.
_SC_ZERO_SIDE_CAP = 0.05
#: v5: rules are built on outward-snapped boxes (_build_box); v4 entries were
#: built on the raw request and are not served, so a warm cache equals a cold one.
_RULE_CACHE_SCHEMA = "sigma-box-ry-v5"
#: The run-independent rule table's entry format and key definition
#: (:func:`_rule_table_key`); a new value opens a new namespace.
_RULE_TABLE_FORMAT = "sigma-box-table-v1"


def _rule_digest(rule, noise_amplification):
    """Authenticate the certificate and its complex128 Ry-inverse nodes."""
    identity = hashlib.sha256(json.dumps(
        [_RULE_CACHE_SCHEMA, list(rule.box), float(rule.eps),
         bool(rule.relative)]).encode())
    for array in (rule.times, rule.weights):
        identity.update(np.asarray(array, dtype="<c16").tobytes())
    identity.update(np.asarray(
        [rule.sup_error, rule.kappa_max, noise_amplification,
         rule.theta_deg, rule.rank], dtype="<f8").tobytes())
    return identity.hexdigest()


#: The self-consistent identity spells its Hamiltonian
#: ``sc_map_{iteration}:{occ_hash}`` (gw/shared_pole_recipe.py), so the map
#: label lives inside a value, not only in the ``iteration_id`` key.
_SC_MAP_LABEL = re.compile(r"^sc_map_\d+:")


def _map_invariant_identity(identity):
    """The identity with its SC map label removed, physics intact.

    Stripping only ``iteration_id`` left the map number in ``hamiltonian``, so
    every SC map opened a fresh request namespace and could never serve a
    containment-compatible rule stored by an earlier map. What remains is the
    physical input the rule depends on; every hit is still re-checked for box
    containment and error currency before use.
    """
    return {key: (_SC_MAP_LABEL.sub("", value) if isinstance(value, str) else value)
            for key, value in identity.items() if key != "iteration_id"}


def _receipt_json(receipt):
    """Serialize the durable quadrature receipt as strict JSON.

    Product windows carry open endpoints (a state tail to ``+inf``, a resonant
    window from ``-inf``); ``json.dumps`` would spell them ``Infinity``, which
    no strict parser reads. An unbounded endpoint is ``null`` and its side is
    its position in the interval pair; any other non-finite value refuses.
    """
    def finite(value):
        if isinstance(value, float) and not np.isfinite(value):
            if np.isnan(value):
                raise ValueError("Sigma quadrature receipt: NaN is not an unbounded edge")
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(item) for item in value]
        return value
    return json.dumps(finite(receipt), sort_keys=True, allow_nan=False)


def sigma_rule_request_cache(directory, identity, poles2, counts, *, eta, eps):
    """Scope shared-pole rules to authenticated current-map physical inputs.

    ``identity`` is the model's energy/occupation/recipe provenance;
    ``poles2`` [Nq,K] in Ry**2 and ``counts`` [Nq] are the small host census.
    Map labels are excluded (:func:`_map_invariant_identity`): equal physical
    inputs on restart and across SC maps share rules, but changed spectra or
    occupations cannot inherit a previous map's plan.
    Domain containment and the executor noise/growth gates still run on hits.
    """
    if directory is None:
        return None
    inputs = _map_invariant_identity(identity)
    digest = hashlib.sha256(json.dumps(
        [_RULE_CACHE_SCHEMA, inputs, float(eta), float(eps)],
        sort_keys=True).encode())
    for row, count in zip(poles2, counts):
        digest.update(np.asarray([count], dtype="<i8").tobytes())
        digest.update(np.asarray(row[:int(count)], dtype="<f8").tobytes())
    return os.path.join(directory, "request_" + digest.hexdigest())


def resolve_sigma_box_cache_dir(setting, input_dir):
    """Resolve the deck's uniform-rule cache spelling beside its input.

    ``"auto"`` selects ``<input_dir>/tmp/sigma_quadrature_rules``;
    ``"off"`` disables the acceleration, including the run-independent rule
    table (:func:`resolve_sigma_rule_table_dir`); any other relative path is
    resolved against ``input_dir``.  A cache is not an accuracy path: every
    loaded rule is still checked for box containment and the requested error
    currency.
    """
    raw = str(setting).strip()
    if raw.lower() == "off":
        return None
    root = os.path.abspath(input_dir)
    if raw.lower() == "auto":
        return os.path.join(root, "tmp", "sigma_quadrature_rules")
    expanded = os.path.expanduser(raw)
    return (expanded if os.path.isabs(expanded)
            else os.path.join(root, expanded))


def _resolve_uniform_rule_trace():
    """Return whether the debug-only box trace was requested."""
    return bool(os.environ.get("LORRAX_UNIFORM_RULE_TRACE"))


def _live_states(branch):
    """Return the executor's exact live state support on the host."""
    energy = np.asarray(gather_to_host(branch.E_A), dtype=np.float64)
    base = np.asarray(gather_to_host(branch.base_mask_A), dtype=bool)
    if base.shape != energy.shape:
        base = np.reshape(base, energy.shape)
    live = base & np.isfinite(energy)
    if branch.band_weight is not None:
        weight = np.asarray(
            gather_to_host(branch.band_weight), dtype=np.float64
        ).reshape(energy.shape)
        # The branch builder has already applied the deck's occupation-window
        # threshold.  This final nonzero test mirrors the multiplicative
        # executor and prevents an exactly absent state from widening a box.
        live &= np.isfinite(weight) & (np.abs(weight) > 0.0)
    indices = np.flatnonzero(live.reshape(-1)).astype(np.int32)
    if not indices.size:
        raise ValueError(f"Sigma box branch {branch.tag!r} has no live states")
    return energy, energy.reshape(-1)[indices], indices


def _product_geometry(branches, eta, edge_factor):
    omega_max = max(
        (float(np.max(branch.omega_abs)) for branch in branches
         if branch.omega_abs.size), default=0.0)
    excursion = 0.0
    state_rows = []
    for branch in branches:
        shape, energy, indices = _live_states(branch)
        excursion = max(excursion, -min(float(np.min(energy)), 0.0))
        state_rows.append((shape, energy, indices))
    state_edge = float(edge_factor) * eta
    edges = sigma_pole_edges(branches, state_edge, excursion)
    return state_rows, {
        "omega_max_ry": omega_max,
        "state_edge_ry": state_edge,
        "pole_edges_ry": edges,
        "omega_cut_ry": edges["near"],
        "negative_state_excursion_ry": excursion,
        "edge_factor": float(edge_factor),
    }


def _state_products(branch, state_edge, edges):
    """The sole owner of the product partition of one branch.

    Rows are ``(name, state_lo, state_hi, pole_selector, pole_lo, pole_hi,
    omega_lo, omega_hi)``: half-open ``(lo, hi]`` state and pole intervals and
    ``[lo, hi)`` in ``|ω|``.  A crossing branch (``ω≥E_F cond``, ``ω<E_F val``)
    splits states and poles at its half's edge.  A non-crossing branch keeps
    its bulk (``E > edge``) whole; its resonant-side states split at the
    ``|ω|`` cut ``near``: above it every denominator is at least ``edge`` from
    zero (``omega_tail``, one sign-definite window over all poles), below it
    the poles split at ``near``.

    Tempting, and why not: one resonant window over every ``|ω|`` of a
    non-crossing branch.  Its long side is ``ω_max + Ω_max``, a crossing rule
    over the cover-grown range for a sliver of excursion states near
    ``ω = 0`` (Na 8^3 map 0: 1682 -> 39 pairs, fit 477 -> 2 s, claim 2821).
    """
    crossing = ((branch.space == "cond" and not branch.neg_omega_half)
                or (branch.space == "val" and branch.neg_omega_half))
    inf = np.inf
    if crossing:
        edge = edges["neg" if branch.neg_omega_half else "pos"]
        half = "neg" if branch.neg_omega_half else "pos"
        return (
            ("resonant", -inf, edge, f"shallow:{half}", 0.0, edge, 0.0, inf),
            ("state_tail", edge, inf, f"shallow:{half}", 0.0, edge, 0.0, inf),
            ("pole_tail", -inf, inf, f"deep:{half}", edge, inf, 0.0, inf),
        )
    near = edges["near"]
    return (
        ("bulk", state_edge, inf, "all", 0.0, inf, 0.0, inf),
        ("resonant", -inf, state_edge, "shallow:near", 0.0, near, 0.0, near),
        ("pole_tail", -inf, state_edge, "deep:near", near, inf, 0.0, near),
        ("omega_tail", -inf, state_edge, "all", 0.0, inf, near, inf),
    )


def _pole_rows(summaries, selector):
    indices, stats = [], []
    for pole, evidence in summaries:
        row = evidence[selector]
        if row is not None:
            indices.append(int(pole))
            stats.append(tuple(float(value) for value in row))
    return np.asarray(indices, dtype=np.int32), stats


def _pole_bounds(count, lower, upper):
    bounds = np.asarray(
        (lower, upper, -np.inf, -np.inf, np.inf, np.inf),
        dtype=np.float64)
    return np.broadcast_to(bounds, (int(count), 6)).copy()


def _box(real_lo, real_hi, gamma_lo, gamma_hi, eta):
    """Pad real support by 2%, without changing its sign topology."""
    pad = 0.02 * max(real_hi - real_lo, eta)
    lo = real_lo - pad if real_lo <= 0.0 else max(real_lo - pad,
                                                   _BOX_SIGN_FRACTION * real_lo)
    hi = real_hi + pad if real_hi >= 0.0 else min(real_hi + pad,
                                                   _BOX_SIGN_FRACTION * real_hi)
    return (float(lo), float(hi),
            float(gamma_lo + eta), float(gamma_hi + eta))


def _box_for_window(frequencies, states, pole_stats, pole_sign, eta):
    a_lo = min(row[0] for row in pole_stats)
    a_hi = max(row[1] for row in pole_stats)
    gamma_lo = min(row[2] for row in pole_stats)
    gamma_hi = max(row[3] for row in pole_stats)
    corners = [
        frequency - pole_sign * (state + pole)
        for frequency in (float(np.min(frequencies)),
                          float(np.max(frequencies)))
        for state in (float(np.min(states)), float(np.max(states)))
        for pole in (a_lo, a_hi)
    ]
    raw_lo, raw_hi = float(min(corners)), float(max(corners))
    return (_box(raw_lo, raw_hi, gamma_lo, gamma_hi, eta),
            (raw_lo, raw_hi), (a_lo, a_hi, gamma_lo, gamma_hi))


def make_sigma_box_spec(
    *, name, frequencies, states, pole_stats, pole_sign, eta_ry,
):
    """Construct one route-neutral denominator-box fit specification.

    Parameters
    ----------
    name
        Stable diagnostic identity for the physical product window.
    frequencies
        External frequencies owned by the window, shape ``(nomega,)`` in Ry.
    states
        Exact live intermediate-state energies, shape ``(nstate,)`` in Ry.
    pole_stats
        Per-pole or per-pane ``(real_min, real_max, gamma_min, gamma_max)``
        rows in Ry.  These are scalar extrema, never histogram weights.
    pole_sign
        ``+1`` for conduction denominators and ``-1`` for valence.
    eta_ry
        Positive retarded broadening in Ry.

    Returns
    -------
    dict
        Box, raw support, fit currency, conjugation, and factor references.
        Route-specific selector metadata may be added by the caller.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    states = np.asarray(states, dtype=np.float64).reshape(-1)
    rows = tuple(tuple(float(value) for value in row) for row in pole_stats)
    sign, eta = float(pole_sign), float(eta_ry)
    if not frequencies.size or not np.all(np.isfinite(frequencies)):
        raise ValueError(f"Sigma box window {name!r} has no finite frequencies")
    if not states.size or not np.all(np.isfinite(states)):
        raise ValueError(f"Sigma box window {name!r} has no finite states")
    if not rows or any(len(row) != 4 for row in rows):
        raise ValueError(
            f"Sigma box window {name!r} needs four pole extrema per row")
    if sign not in (-1.0, 1.0):
        raise ValueError("Sigma box pole_sign must be +1 or -1")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    box, raw_real, pole_extent = _box_for_window(
        frequencies, states, rows, sign, eta)
    kind = ("sign_definite_positive" if box[0] > 0.0 else
            "sign_definite_negative" if box[1] < 0.0 else
            "crossing")
    e_ref_a, e_ref_b = _factor_references(kind, sign, states, rows)
    return {
        "name": str(name), "frequencies": frequencies, "states": states,
        "pole_stats": rows, "pole_sign": sign,
        "raw_real_support": raw_real, "box": box, "kind": kind,
        "pole_extent": pole_extent, "conjugate": sign < 0.0,
        "E_ref_A": e_ref_a, "E_ref_B": e_ref_b,
    }


def _rule_cache_lookup(
    directory, box, eps, relative, *, noise_amplification_cap,
):
    """Return the smallest compatible rule plus any unreadable-path warnings.

    Only ``_RULE_CACHE_SCHEMA`` entries are served, and each is authenticated
    against its stored digest before any compatibility filter reads it.
    Other schemas (including clock-reduced entries written before
    2026-09-11) are ignored with one warning, not deleted.
    """
    if directory is None:
        return None, ()
    warnings = []
    try:
        entries = [name for name in os.listdir(directory)
                   if name.startswith("rule_") and name.endswith(".npz")]
        names = [name for name in entries
                 if name.startswith(f"rule_{_RULE_CACHE_SCHEMA}_")]
        stale = sorted(set(entries) - set(names))
        if stale:
            warnings.append(
                "WARNING sigma quadrature cache schema migration: "
                f"path={os.path.abspath(directory)}; schema={_RULE_CACHE_SCHEMA}; "
                f"ignored {len(stale)} rule file(s) of another schema, first={stale[0]}; "
                "affected windows will be rebuilt. The files are retained "
                "as prior-run evidence.")
    except FileNotFoundError:
        # A fresh request scope: nothing stored yet is an empty cache. The
        # plan's builds create the directory when they are stored.
        return None, ()
    except OSError as exc:
        path = os.path.abspath(directory)
        warnings.append(
            "WARNING sigma quadrature cache lookup failed; rules will be "
            "rebuilt in memory: "
            f"path={path} error={type(exc).__name__}: {exc}")
        return None, tuple(warnings)
    best = None
    for name in sorted(names):
        path = os.path.abspath(os.path.join(directory, name))
        try:
            with np.load(path) as data:
                # Authenticate the stored object before compatibility filtering.
                # A nearby requested eps may reuse this certificate, but is
                # never substituted into the digest of its immutable identity.
                cached_box = tuple(float(value) for value in data["box"])
                rule = UniformRule(
                    times=np.asarray(data["times"]),
                    weights=np.asarray(data["weights"]),
                    box=cached_box, eps=float(data["eps"]),
                    relative=bool(data["relative"]),
                    theta_deg=float(data["theta_deg"]),
                    rank=int(data["rank"]),
                    sup_error=float(data["sup_error"]),
                    kappa_max=float(data["kappa_max"]), seconds=0.0)
                amplification = float(data["roundoff_amplification"])
                if (str(data["schema"]) != _RULE_CACHE_SCHEMA
                        or not _rule_is_certified(rule, rule.eps)
                        or rule.times.ndim != 1 or rule.weights.ndim != 1
                        or not np.isfinite(amplification)
                        or str(data["digest"]) != _rule_digest(rule, amplification)):
                    raise ValueError("GATE sigma_rule_integrity: certificate digest/schema mismatch")
                # A cached certificate above eps, or one built for a looser
                # noise consumer, is not a rule for this request whatever its
                # node count (Na pole-tail, 2026-09-05).
                if (abs(rule.eps - eps) > 1.0e-12 * eps
                        or rule.relative != relative
                        or amplification > noise_amplification_cap
                        or rule.sup_error > eps):
                    continue
                if not (cached_box[0] <= box[0]
                        and cached_box[1] >= box[1]
                        and cached_box[2] <= box[2]
                        and cached_box[3] >= box[3]):
                    continue
                if best is None or rule.node_count < best[0].node_count:
                    best = (rule, name)
        except (zipfile.BadZipFile, EOFError) as exc:
            # A CORRUPT archive (a torn write, a quota-truncated copy) is a
            # miss, never a refusal: zipfile.BadZipFile is not an OSError, so
            # it used to escape this loop and refuse every later run in the
            # scope.  Rank 0 deletes it so the scope heals; a peer reading
            # it concurrently also misses (or finds it gone: an OSError).
            removed = ""
            if process_rank() == 0:
                try:
                    os.unlink(path)
                    removed = "; deleted by rank 0"
                except OSError as unlink_exc:
                    removed = f"; delete failed: {unlink_exc}"
            warnings.append(
                "WARNING sigma quadrature cache entry is corrupt and was "
                f"treated as a miss: path={path} "
                f"error={type(exc).__name__}: {exc}{removed}")
        except (OSError, KeyError, ValueError) as exc:
            warnings.append(
                "WARNING sigma quadrature cache entry is unreadable and "
                "will not be used: "
                f"path={path} error={type(exc).__name__}: {exc}")
    return best, tuple(warnings)


def _rule_is_certified(rule, eps) -> bool:
    """A rule is a rule only if every number in it is finite and sup <= eps.

    A finite ``sup_error`` beside NaN nodes or weights is not a certificate
    (lane QUADCHECK's boundary probe, JID 57930535.148, 2026-09-05): the
    executor would multiply NaN into every Sigma element and the noise gate
    compares NaN with a ceiling, which never refuses.
    """
    times = np.asarray(rule.times)
    weights = np.asarray(rule.weights)
    return bool(
        times.size > 0 and weights.size == times.size
        and np.all(np.isfinite(times)) and np.all(np.isfinite(weights))
        and np.isfinite(float(rule.sup_error))
        and float(rule.sup_error) <= float(eps)
        and np.isfinite(float(rule.kappa_max)))


def _rule_cache_store(directory, rule, noise_amplification):
    """Atomically store one immutable box certificate, or return a warning."""
    if directory is None:
        return None
    if not (_rule_is_certified(rule, float(rule.eps))
            and np.isfinite(float(noise_amplification))):
        return ("WARNING sigma quadrature cache store refused an uncertified "
                "or non-finite rule (nothing written)")
    digest = _rule_digest(rule, noise_amplification)
    path = os.path.abspath(os.path.join(
        directory, f"rule_{_RULE_CACHE_SCHEMA}_{digest}.npz"))
    temporary = None
    try:
        os.makedirs(directory, exist_ok=True)
        temporary = _temporary_name(path)
        _write_rule_archive(temporary, rule, noise_amplification, digest)
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        # A cache is an acceleration, never a second correctness path, so the
        # accepted rule remains usable. Silence is still wrong: it turns every
        # restart into a cold 100+s plan with no explanation.
        return (
            "WARNING sigma quadrature cache write failed; this accepted rule "
            "will be rebuilt on a later run: "
            f"path={path} error={type(exc).__name__}: {exc}")
    return None


def _temporary_name(path):
    """A sibling name no other node, rank or process can pick for ``path``."""
    return (f"{path}.{socket.gethostname()}.{process_rank()}.{os.getpid()}."
            f"{uuid.uuid4().hex[:12]}.tmp")


def _write_rule_archive(path, rule, noise_amplification, digest, **extra):
    """Write one rule archive at ``path`` and make it durable."""
    with open(path, "wb") as handle:
        np.savez(
            handle, schema=_RULE_CACHE_SCHEMA, digest=digest,
            box=np.asarray(rule.box, np.float64),
            eps=float(rule.eps), relative=bool(rule.relative),
            times=rule.times, weights=rule.weights,
            sup_error=float(rule.sup_error),
            kappa_max=float(rule.kappa_max),
            roundoff_amplification=float(noise_amplification),
            theta_deg=float(rule.theta_deg), rank=int(rule.rank),
            seconds=float(rule.seconds), **extra)
        # Durable before it becomes visible: without this a node loss
        # after the rename can leave a torn archive under the final name.
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------- rule table
# The run-local request scope above is a SERVING policy: containment inside
# one physical scope, so sector calls and restarts share rules. The table
# below is a MEMO of the builder: ``build_uniform_rule(build_box, eps,
# kappa_cap)`` reads no clock and pins its BLAS threads, so its result is a
# function of the snapped build box, the currency and the solver identity
# (claim 2737). A hit returns the bytes a cold build returns, so a warm run
# equals the cold run that wrote the table, and no run's answer depends on
# which other decks wrote it. Serving across runs by containment would.

def resolve_sigma_rule_table_dir(cache_dir):
    """The run-independent rule table, or ``None`` when caching is off.

    ``$SCRATCH/.cache/lorrax/sigma_box_rules``, beside the compile caches
    (:func:`common.jax_compile_cache.default_cache_root`); no knob.
    ``sigma_quadrature_cache_dir = off`` (``cache_dir is None``) turns it
    off with the run-local scope. ``LORRAX_SIGMA_RULE_TABLE_TEST_DIR`` is
    the suite's private table (``tests/conftest.py``): tests patch the
    builder, and a fake rule must never reach the user's table.
    """
    if cache_dir is None:
        return None
    private = os.environ.get("LORRAX_SIGMA_RULE_TABLE_TEST_DIR", "").strip()
    if private:
        return private
    from common.jax_compile_cache import default_cache_root
    return os.path.join(str(default_cache_root().parent), "sigma_box_rules")


def _rule_table_key(build_box, eps, relative, kappa_cap):
    """Everything a builder call's result depends on, JSON-ready.

    ``builder`` names the function that answers: a patched builder (a test
    fake) opens its own namespace and can never answer a production key.
    """
    return {
        "format": _RULE_TABLE_FORMAT, "schema": _RULE_CACHE_SCHEMA,
        "builder": (f"{build_uniform_rule.__module__}."
                    f"{build_uniform_rule.__qualname__}"),
        "box": [float(value) for value in build_box], "eps": float(eps),
        "relative": bool(relative),
        "kappa_cap": None if kappa_cap is None else float(kappa_cap),
        "solver": uniform_rule_solver_identity(),
    }


def _rule_table_path(root, key):
    """``(path, key_json)``: one namespace per format and solver identity.

    JSON spells a float by its shortest round-trip repr, so equal keys are
    equal bit for bit."""
    blob = json.dumps(key, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode()).hexdigest()
    solver = hashlib.sha256(json.dumps(
        [key["builder"], key["solver"]], sort_keys=True).encode()).hexdigest()[:16]
    return (os.path.join(root, f"{key['format']}_{solver}", digest[:2],
                         f"rule_{digest}.npz"), blob)


def _rule_table_lookup(root, key):
    """``((rule, amplification, digest), None)`` on a hit, else ``(None, why)``.

    ``why`` is ``None`` for an absent entry and a named warning for one that
    exists but is not served: another format or schema, another key, or a
    certificate whose digest does not authenticate.
    """
    path, blob = _rule_table_path(root, key)
    try:
        with np.load(path, allow_pickle=False) as data:
            stored = (str(data["table_format"]), str(data["schema"]))
            if stored != (_RULE_TABLE_FORMAT, _RULE_CACHE_SCHEMA):
                raise ValueError(
                    f"schema mismatch: entry {stored[0]}/{stored[1]}, "
                    f"this build reads {_RULE_TABLE_FORMAT}/{_RULE_CACHE_SCHEMA}")
            box = tuple(float(value) for value in data["box"])
            if str(data["key"]) != blob or list(box) != key["box"]:
                raise ValueError("key mismatch: the entry answers another request")
            rule = UniformRule(
                times=np.asarray(data["times"]),
                weights=np.asarray(data["weights"]),
                box=box, eps=float(data["eps"]),
                relative=bool(data["relative"]),
                theta_deg=float(data["theta_deg"]), rank=int(data["rank"]),
                sup_error=float(data["sup_error"]),
                kappa_max=float(data["kappa_max"]), seconds=0.0)
            amplification = float(data["roundoff_amplification"])
            digest = _rule_digest(rule, amplification)
            if (str(data["digest"]) != digest
                    or not _rule_is_certified(rule, rule.eps)
                    or rule.times.ndim != 1 or rule.weights.ndim != 1
                    or not np.isfinite(amplification)):
                raise ValueError("certificate digest mismatch")
    except FileNotFoundError:
        return None, None
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        return None, (
            "WARNING sigma rule table entry not served (a miss; this run "
            f"builds the rule and replaces the entry): path={path} "
            f"error={type(exc).__name__}: {exc}")
    return (rule, amplification, digest), None


def _rule_table_store(root, key, rule, noise_amplification):
    """Publish one built rule; the first writer wins. Returns a warning or ``None``.

    The archive is written and synced under a private name, then hard-linked
    to its key: ``link`` fails when the key exists, so a published entry
    never changes and readers never see a partial file. A published entry
    whose rule differs from this build is a DETERMINISM warning (the builder
    is meant to be a function of the key); the table keeps the first. An
    unservable entry is replaced by rename. Where the filesystem has no hard
    links, rename publishes (the last writer wins, still whole).
    """
    path, blob = _rule_table_path(root, key)
    digest = _rule_digest(rule, noise_amplification)
    temporary = None
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = _temporary_name(path)
        _write_rule_archive(temporary, rule, noise_amplification, digest,
                            table_format=_RULE_TABLE_FORMAT, key=blob)
        try:
            os.link(temporary, path)
            return None
        except FileExistsError:
            pass
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV,
                                 errno.EMLINK, errno.ENOSYS):
                raise
            os.replace(temporary, path)
            temporary = None
            return None
        existing, problem = _rule_table_lookup(root, key)
        if existing is None:
            os.replace(temporary, path)
            temporary = None
            return None if problem is None else (
                problem + "; replaced by this run's build")
        if existing[2] != digest:
            return (
                "WARNING sigma rule table DETERMINISM: this run built a "
                "different rule for a key already in the table (the table "
                f"keeps the first; this run used its own build): path={path} "
                f"table digest={existing[2][:16]} build digest={digest[:16]}")
        return None
    except OSError as exc:
        return ("WARNING sigma rule table write failed; this rule will be "
                f"rebuilt by a later run: path={path} "
                f"error={type(exc).__name__}: {exc}")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


#: Relative cell of the logarithmic grid every build box is snapped outward to.
_BUILD_GRID_STEP = 1.0e-4


def snap_outward(x, scale, outward):
    """``x`` moved outward (``outward`` = +1 up, -1 down) to the nearest point
    of ``sign(x) * scale * (1 + _BUILD_GRID_STEP)**k``, k integer; zero stays
    zero and a nonzero edge never changes sign (a sign-definite box stays so).

    The one grid for every quantity a rule is built from or a reuse decision
    compares: Sigma build boxes (scale eta), the chi response rule support and
    the SC shared-pole sampling envelope (scale 1). A round-off-perturbed
    input lands on the same grid point unless it straddles a cell edge."""
    if x == 0.0:
        return 0.0
    k = np.log(abs(x) / scale) / np.log1p(_BUILD_GRID_STEP)
    k = np.ceil(k) if outward * np.sign(x) > 0 else np.floor(k)
    return float(np.sign(x) * scale * np.exp(k * np.log1p(_BUILD_GRID_STEP)))


def _build_box(box, eta, *, widen):
    """The box a rule is built on: the request, optionally widened, snapped outward.

    ``widen`` adds 1% of the width to the far edges (|x| > 3 eta) so nearby
    SC maps and sector calls hit by containment. The snap makes the rule a
    function of a grid cell rather than of the exact request: the builder is
    a nonlinear fit with many certified local solutions, so a request moved
    by round-off (extreme shared-pole edges differ 1e-9-4e-8 relative between
    two exact GEMM orders) otherwise lands on a different rule. Every Fe 4^3
    bispinor window rule differed between the face and band-complete ψ
    contraction orders, and
    eqp1 by 0.32 meV (P2-E, 2026-09-24). On the 1e-4 grid a perturbed
    request maps to the same build box, hence the same rule bit for bit,
    unless it straddles a cell edge (probability ~ perturbation / 1e-4).
    The cell is kept that fine because the fixed-N bracket accepts node
    counts in 10% steps: a marginal certification flips when its box grows,
    and 0.1% cells added 5% tau pairs on CrI3 8x8 SC (1e-4 cells: see the
    commit).
    """
    if widen:
        extra = 0.01 * max(box[1] - box[0], eta)
        near = 3.0 * eta
        box = (box[0] - extra if box[0] < -near else box[0],
               box[1] + extra if box[1] > near else box[1],
               box[2], box[3] * 1.01)
    return (snap_outward(box[0], eta, -1), snap_outward(box[1], eta, +1),
            snap_outward(box[2], eta, -1), snap_outward(box[3], eta, +1))


def _factor_references(kind, pole_sign, states, pole_stats):
    if kind == "crossing":
        return float(np.min(states)), 0.0
    table_sign = 1.0 if kind == "sign_definite_positive" else -1.0
    endpoint = np.max if pole_sign * table_sign > 0.0 else np.min
    pole_real = np.asarray(
        [value for row in pole_stats for value in row[:2]], np.float64)
    return float(endpoint(states)), float(endpoint(pole_real))


def _factor_growth(times, pole_sign, states, pole_stats, e_ref_a, e_ref_b):
    """Worst log growth of the executor's two separately factored terms."""
    times_exec = pole_sign * np.asarray(times, np.complex128).reshape(-1)
    green = float(np.max(np.real(
        -1.0j * (states[:, None] - e_ref_a) * times_exec[None, :])))
    pole_corners = np.asarray([
        real - 1.0j * gamma
        for row in pole_stats
        for real in row[:2]
        for gamma in row[2:]
    ], dtype=np.complex128)
    screened = float(np.max(np.real(
        -1.0j * (pole_corners[:, None] - e_ref_b)
        * times_exec[None, :])))
    return green, screened


def _noise_amplification_cap():
    """Largest roundoff amplification the executor's noise budget admits."""
    return _RUNTIME_NOISE_BUDGET / _RUNTIME_NOISE_EPSILON


def _fit_rule(spec, eps, cache_dir, eta, *, cache_build_widen=True,
              attempts=None):
    """Look up or build one window's rule and accept it; never write the cache.

    The lookup reads only certificates written before this plan: the plan's
    own builds are stored after every rank has looked up (see
    :func:`fit_sigma_box_specs`), so no window's choice depends on how far
    another rank has got. ``attempts`` restricts a crossing build to one
    range of its fixed-N bracket (``build_uniform_rule``); ``None`` is
    returned when that range does not certify.
    """
    requested_box = spec["box"]
    # This is exactly the builder's default currency predicate.  It is used
    # here only to search cache metadata; cache misses still leave the choice
    # to build_uniform_rule(relative=None).
    relative = requested_box[0] > 0.0 or requested_box[1] < 0.0
    noise_amplification_cap = _noise_amplification_cap()
    analytic_line = (bool(spec.get("analytic_line")) and not relative
                     and requested_box[2] == requested_box[3]
                     and spec["pole_extent"][2:] == (0.0, 0.0))
    built = False
    rule_table, table_key = "none", None
    if analytic_line:
        # PPM's real poles make Im(d)=eta exactly.  Ask the analytic service
        # for that line; a cached rectangle rule cannot silently preempt it.
        rule = analytic_line_box_rule(requested_box, eps)
        cache_lookup_warnings = ()
        cache_status = "analytic-line"
    else:
        cached, cache_lookup_warnings = _rule_cache_lookup(
            cache_dir, requested_box, eps, relative,
            noise_amplification_cap=noise_amplification_cap)
        if cached is not None:
            rule, cache_name = cached
            cache_status = f"hit:{cache_name}"
        else:
            build_box = _build_box(requested_box, eta, widen=(
                cache_dir is not None and cache_build_widen))
            build_kwargs = {}
            if relative:
                # For a sign-definite rule the service's kappa is
                # sum|term|/|Q|, while Sigma's noise amplification is
                # |d|*sum|term|.  The certified relative sup error gives
                # |d Q(d)| <= 1 + eps, so this cap is sufficient for the
                # executor's absolute noise condition.  Crossing
                # rules use peak-relative term mass instead and retain the
                # service's ordinary cancellation cap.
                build_kwargs["kappa_cap"] = (
                    noise_amplification_cap / (1.0 + eps))
            # The table memoizes this builder call: the attempts range is not
            # in the key because the stored rule is the window's decided one.
            table = resolve_sigma_rule_table_dir(cache_dir)
            entry = None
            if table is not None:
                table_key = _rule_table_key(
                    build_box, eps, build_box[0] > 0.0 or build_box[1] < 0.0,
                    build_kwargs.get("kappa_cap"))
                entry, table_warning = _rule_table_lookup(table, table_key)
                if table_warning is not None:
                    cache_lookup_warnings += (table_warning,)
            if entry is not None:
                rule, rule_table = entry[0], "hit"
            else:
                if attempts is not None and not relative:
                    build_kwargs["attempts"] = attempts
                rule = build_uniform_rule(build_box, eps, **build_kwargs)
                if rule is None:
                    return None
                rule_table = "off" if table is None else "built"
            # A table hit is this plan's build in every later step (the
            # request-scope store and _serve_from_plan), so warm equals cold.
            built = True
            cache_status = "miss" if cache_dir is not None else "off"
        # There is no retry.  The builder takes no clock and no pass count,
        # so a second call with the same inputs returns the same rule; the
        # old 5x-budget retry existed only because the first attempt could
        # have been cut short by a deadline, and there is no deadline to
        # lengthen.  A refusal here is now a statement about the box.
    fit = _accept_rule(spec, rule, eps, cache_status=cache_status,
                       cache_dir=cache_dir)
    fit.update(built=built, analytic_line=analytic_line,
               cache_lookup_warnings=tuple(cache_lookup_warnings),
               rule_table=rule_table, rule_table_key=table_key)
    return fit


def _accept_rule(spec, rule, eps, *, cache_status, cache_dir):
    """Accept one rule for one window, or refuse; return its executor receipt."""
    noise_budget = _RUNTIME_NOISE_BUDGET
    # ONE ACCEPTANCE ON EVERY PATH.  One-shot, fixed-SC initialization and
    # its rebuilds all require the certified sup error at or below eps; the
    # fixed-SC bypass (enforce_sup_error=False, 2026-09-03) let Na retain a
    # conduction pole-tail rule at 400 x eps in every self-consistent arm.
    if not _rule_is_certified(rule, eps):
        cache_note = ("" if cache_dir is None
                      else f", cache directory {os.path.abspath(cache_dir)}")
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused: rule sup error "
            f"{float(rule.sup_error):.6g} exceeds eps={eps:.6g} or the rule "
            f"is not finite ({int(np.asarray(rule.times).size)} nodes on box "
            f"{tuple(round(float(v), 6) for v in rule.box)}, kind "
            f"{spec.get('kind', '?')}, cache={cache_status}{cache_note}"
            f"). Remedy: a sign-preserving or split product window (the SC "
            f"pad now keeps sign-definite supports sign-definite), or a "
            f"certified crossing rule; do not loosen sigma_quadrature_eps to "
            f"admit this rule.")
    # Runtime perturbations must be bounded in the SAME currency as the
    # approximation.  ``kappa = sum|term|/|Q|`` is already relative for a
    # sign-definite box, but it overstates a crossing box's peak-relative
    # error by ~|d|/eta at its far edge.  Measure rho*sum|term| directly.
    # the noise mass has a subharmonic logarithm, so its box maximum lies on
    # the boundary: sample the edges at the rule's own horizon
    noise_cloud = boundary_samples(
        rule.box, rule.theta_deg, float(np.max(np.abs(rule.times))), eps)
    noise_rho = (np.abs(noise_cloud) if rule.relative
                 else float(np.min(noise_cloud.imag)))
    noise_amplification = rule_roundoff_amplification(
        rule.times, rule.weights, noise_cloud, noise_rho)
    noise_bound = noise_amplification * _RUNTIME_NOISE_EPSILON
    if not np.isfinite(noise_bound) or noise_bound > noise_budget:
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused: runtime-noise "
            f"bound {noise_bound:.6g} exceeds {noise_budget:.6g}")

    times = np.asarray(rule.times, np.complex128)
    weights = np.asarray(rule.weights, np.complex128)
    if spec["conjugate"]:
        # 1/conj(d) = conj(1/d): one upper-half-plane build serves the
        # lower-half-plane causal branch exactly.
        times, weights = -np.conj(times), np.conj(weights)
    growth = _factor_growth(
        times, spec["pole_sign"], spec["states"], spec["pole_stats"],
        spec["E_ref_A"], spec["E_ref_B"])
    if max(growth) > _FACTOR_GROWTH_CAP:
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused: factored log "
            f"growth {max(growth):.6g} exceeds {_FACTOR_GROWTH_CAP:g}")
    node_digest = hashlib.sha256(
        np.ascontiguousarray(times).view(np.uint8).tobytes()
        + np.ascontiguousarray(weights).view(np.uint8).tobytes()
    ).hexdigest()[:16]
    return {
        "times": times, "weights": weights,
        "node_count": int(times.size), "rule_box": tuple(rule.box),
        "relative": bool(rule.relative), "sup_error": float(rule.sup_error),
        "kappa_max": float(rule.kappa_max), "theta_deg": float(rule.theta_deg),
        "rank": int(rule.rank), "seconds": float(rule.seconds),
        "cache_status": cache_status, "factor_growth": growth,
        "noise_bound": noise_bound, "noise_budget": noise_budget,
        "roundoff_amplification": noise_amplification,
        "node_digest": node_digest,
        "rule": rule, "rule_digest": _rule_digest(rule, noise_amplification),
        "cache_write_warning": None, "cache_lookup_warnings": (),
        "rule_table": "none", "rule_table_key": None,
        "rule_table_warning": None,
        "one_line": (f"analytic line: {rule.node_count} nodes, "
                     f"sup {rule.sup_error:.2e} (eps {eps:g})"
                     if cache_status == "analytic-line" else rule.one_line()),
    }


def _serve_from_plan(specs, fits, eps, cache_dir):
    """Give each window the smallest compatible rule of this whole plan.

    Resolution runs after every miss is built, on the replicated receipts, in
    a fixed order: candidates are the window's own rule (a pre-plan cache hit
    or its build) and every rule this plan built, ranked by (node count,
    certificate digest), the same key a later cache lookup uses. The result
    is the rule a warm rerun would pick and does not depend on rank timing:
    before this, whether a window saw another window's fresh rule depended on
    how far the other rank had got (Si shared-pole ``cond:pole_tail`` took
    the 9-node own rule or the 7-node ``cond:bulk`` one, eqp1 0.80 ueV apart;
    KNOWN_LORRAX_ISSUES 2026-09-24).
    """
    fresh = sorted((fit for fit in fits if fit["built"]),
                   key=lambda fit: (fit["node_count"], fit["rule_digest"]))
    cap = _noise_amplification_cap()
    served = []
    for spec, own in zip(specs, fits):
        chosen = own
        if not own["analytic_line"]:
            box = spec["box"]
            relative = box[0] > 0.0 or box[1] < 0.0
            own_key = (own["node_count"], own["rule_digest"])
            for other in fresh:
                if (other["node_count"], other["rule_digest"]) >= own_key:
                    break
                rule = other["rule"]
                if (abs(rule.eps - eps) > 1.0e-12 * eps
                        or bool(rule.relative) != relative
                        or other["roundoff_amplification"] > cap
                        or not _box_contains(tuple(rule.box), box)):
                    continue
                try:
                    chosen = _accept_rule(
                        spec, rule, eps, cache_dir=cache_dir,
                        cache_status=f"plan:{other['rule_digest'][:16]}")
                except RuntimeError:
                    continue
                chosen.update(built=False, analytic_line=False,
                              cache_lookup_warnings=own["cache_lookup_warnings"])
                break
        served.append(chosen)
    return served


def _fit_cost(spec, eta):
    """Predicted builder cost, only to balance ranks: a crossing rule's node
    count follows its short side in units of eta (CrI3 8x8: 113-129 nodes,
    4-9 s); a sign-definite rule is a few nodes (0.5-1.5 s)."""
    if spec["kind"] != "crossing":
        return 1.0
    return 1.0 + min(-float(spec["box"][0]), float(spec["box"][1])) / eta


def _rank_assignment(costs, world):
    """Longest predicted fit first, each to the least-loaded rank.

    Equal costs reduce to round-robin. The assignment is a function of the
    specs, so every rank computes the same one, and a rule does not depend on
    the rank that fits it."""
    load, owned = [0.0] * world, [[] for _ in range(world)]
    for index in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        target = min(range(world), key=lambda r: (load[r], r))
        owned[target].append(index)
        load[target] += costs[index]
    return owned


def _bracket_tasks(specs, costs, world):
    """``(window, attempts, cost)`` tasks: crossing builds split across ranks.

    A crossing rule is a fixed-N bracket of 2-3 solves of 3-10 s each after a
    3-6 s setup (CrI3 8x8 SC, val:resonant at 248 and 269 nodes), serial on
    one rank while the other ranks finish their sign-definite windows in a
    few seconds. A window whose predicted cost is a share ``f`` of the plan
    gets ``floor(f * world)`` tasks, and at least two once ``f * world >= 1``:
    single attempts ``0, 1, ...`` and the tail (the remaining attempts and
    the fallback). One dominant window takes three of four ranks and leaves
    one for the sign-definite windows; two (the SC map-0 one-shot and padded
    rules) take two each. Rounding half up gave the lone window a fourth,
    wasted attempt and stacked the small windows on its ranks (CrI3 8x8 SC
    map 1: 15.7 s against 12.2 s).
    Every task repeats the setup; the window's wall becomes one setup plus
    its longest attempt instead of the sum. Analytic lines and sign-definite
    windows stay whole.
    """
    total = float(sum(costs)) or 1.0
    tasks = []
    for index, (spec, cost) in enumerate(zip(specs, costs)):
        share = world * cost / total
        split = (1 if world < 2 or share < 1.0 or spec.get("analytic_line")
                 or spec["kind"] != "crossing" else max(2, int(share)))
        if split < 2:
            tasks.append((index, None, cost))
            continue
        ranges = [(j, j + 1) for j in range(split - 1)] + [(split - 1, None)]
        tasks.extend((index, attempts, cost / split) for attempts in ranges)
    return tasks


def _parallel_fits(specs, worker, costs):
    """Fit independent windows once across ranks and replicate small rules.

    A crossing build may run as several attempt ranges
    (:func:`_bracket_tasks`); its window takes the first range, in attempt
    order, that returns a rule or refuses, which is the serial build's
    result: the node-count sequence does not depend on the solves and every
    rank runs the same BLAS configuration. ``worker(index, attempts)``.
    """
    rank, world = int(process_rank()), int(process_count())
    tasks = _bracket_tasks(specs, costs, world)
    local = []
    for task in _rank_assignment([cost for _, _, cost in tasks], world)[rank]:
        index, attempts, _cost = tasks[task]
        started = time.perf_counter()
        try:
            value = worker(index, attempts)
            error = None
        except Exception as exc:  # refusals cross ranks as data, then raise
            value = None
            error = f"{type(exc).__name__}: {exc}"
        local.append({
            "index": index, "task": task, "source_rank": rank, "value": value,
            "error": error, "wall_seconds": time.perf_counter() - started,
        })
    if world == 1:
        shards = [local]
    else:
        payload = np.frombuffer(
            pickle.dumps(local, protocol=pickle.HIGHEST_PROTOCOL),
            dtype=np.uint8)
        lengths = np.asarray(all_gather_processes(
            np.asarray(payload.size, np.int32)), dtype=np.int64).reshape(-1)
        # A power-of-two carrier: the gather's shape is part of its compile
        # key, and the pickled receipts carry the run's rule-cache path, so
        # an exact width gave every run directory its own executable.
        width = 1 << max(0, int(np.max(lengths)) - 1).bit_length()
        padded = np.zeros(width, np.uint8)
        padded[:payload.size] = payload
        gathered = np.asarray(all_gather_processes(padded), np.uint8)
        shards = [pickle.loads(np.ascontiguousarray(
            gathered[source, :int(length)]).tobytes())
                  for source, length in enumerate(lengths)]
    done = sorted((row for shard in shards for row in shard),
                  key=lambda row: row["task"])
    if [row["task"] for row in done] != list(range(len(tasks))):
        raise RuntimeError("Sigma box planner did not gather every window")
    rows = []
    for index in range(len(specs)):
        ranges = [row for row in done if row["index"] == index]
        # Tasks are in attempt order; the tail always decides.
        decided = next(row for row in ranges
                       if row["error"] is not None or row["value"] is not None)
        rows.append(dict(decided, wall_seconds=max(
            row["wall_seconds"] for row in ranges)))
    refusal = next((row for row in rows if row["error"] is not None), None)
    if refusal is not None:
        raise RuntimeError(refusal["error"])
    return [row["value"] for row in rows], rows


def fit_sigma_box_specs(
    specs, eta_ry, *, eps, cache_dir, cache_build_widen=True,
):
    """One plan: :func:`fit_sigma_box_spec_groups` with a single group."""
    (fits, fit_rows), = fit_sigma_box_spec_groups(
        [(specs, cache_build_widen)], eta_ry, eps=eps, cache_dir=cache_dir)
    return fits, fit_rows


def fit_sigma_box_spec_groups(groups, eta_ry, *, eps, cache_dir):
    """Fit independent route-neutral box specifications across processes.

    The input rows must come from :func:`make_sigma_box_spec`.  This function
    owns the shared cache lookup/build, rule acceptance, lower-half-plane
    conjugation, runtime-noise guard, and factored-growth guard.  It returns
    only small replicated rule receipts; route-specific physical selectors
    stay with the caller.  With a cache, every window is looked up against
    the certificates present before the plan, every miss is built, the
    builds are stored, and each window is then served the smallest
    compatible rule of the plan in a fixed order (:func:`_serve_from_plan`),
    so the result does not depend on rank timing.

    ``groups`` is a list of ``(specs, cache_build_widen)``; each group is one
    plan, looked up against the cache as it was before this call and served
    only from its own builds, but all groups share one balanced parallel pass
    (the SC map-0 one-shot and padded sets, P2-E 2026-09-24). Returns one
    ``(fits, fit_rows)`` per group.
    """
    rows, widen, bounds = [], [], []
    for specs, group_widen in groups:
        start = len(rows)
        rows.extend(specs)
        widen.extend([bool(group_widen)] * (len(rows) - start))
        bounds.append((start, len(rows)))
    eta, tolerance = float(eta_ry), float(eps)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("sigma_quadrature_eps must lie in (0, 1)")
    fits, fit_rows = _parallel_fits(
        rows, lambda index, attempts: _fit_rule(
            rows[index], tolerance, cache_dir, eta,
            cache_build_widen=widen[index], attempts=attempts),
        [_fit_cost(spec, eta) for spec in rows])
    if cache_dir is None:
        return [(fits[lo:hi], fit_rows[lo:hi]) for lo, hi in bounds]
    # Every rank has looked up by now (the gather above), so writing the
    # plan's builds cannot change any choice made in it. One writer: the
    # replicated receipts already hold every build.
    table = resolve_sigma_rule_table_dir(cache_dir)
    if process_rank() == 0:
        stored, published = {}, {}
        for fit in fits:
            if fit["built"] and fit["rule_digest"] not in stored:
                stored[fit["rule_digest"]] = _rule_cache_store(
                    cache_dir, fit["rule"], fit["roundoff_amplification"])
            if fit["built"]:
                fit["cache_write_warning"] = stored[fit["rule_digest"]]
            if fit["rule_table"] == "built" and table is not None:
                # Every builder call of the plan, served or not, is memoized.
                key = json.dumps(fit["rule_table_key"], sort_keys=True)
                if key not in published:
                    published[key] = _rule_table_store(
                        table, fit["rule_table_key"], fit["rule"],
                        fit["roundoff_amplification"])
                fit["rule_table_warning"] = published[key]
    if process_count() > 1 and any(fit["built"] for fit in fits):
        # ORDER the store before any rank's NEXT lookup.  Sector Sigma calls
        # share one scope (eb19474d): without this a peer can look up the TT
        # or CT windows before rank 0 has stored CC's builds, miss, and fit a
        # different (within-eps) rule, so the plan would depend on rank
        # timing.  ``fits`` is replicated, so every rank takes this or none.
        all_gather_processes(np.asarray(0, np.int32))
    return [(_serve_from_plan(rows[lo:hi], fits[lo:hi], tolerance, cache_dir),
             fit_rows[lo:hi]) for lo, hi in bounds]


def _box_contains(outer, inner):
    """Return whether one certified denominator box contains another."""
    return (outer[0] <= inner[0] and outer[1] >= inner[1]
            and outer[2] <= inner[2] and outer[3] >= inner[3])


def _box_escape_reasons(outer, inner):
    """Describe every edge by which ``inner`` escapes ``outer``."""
    labels = ("real_lo", "real_hi", "imag_lo", "imag_hi")
    escaped = (
        inner[0] < outer[0], inner[1] > outer[1],
        inner[2] < outer[2], inner[3] > outer[3],
    )
    return [
        f"{label}: current={inner[index]:.12g} Ry, "
        f"fixed={outer[index]:.12g} Ry"
        for index, (label, is_outside) in enumerate(zip(labels, escaped))
        if is_outside
    ]


def _sc_padded_box_spec(spec, eta):
    """Return the frozen SC certificate box for one product window.

    Every state may drift by its own classification pad ``sc_state_pad_ev``
    before it is reclassified, so the corners are taken over the padded state
    interval ``[min(E - pad(E)), max(E + pad(E))]``: each real edge moves by
    the pad of the state that sets it. Branch state energies already measure
    distance from mu. Pole drift is covered independently by widening every
    real and imaginary pole extent by ten percent. Nothing else is added; a
    map that leaves this box is an escape and refits the set.

    Tempting, and why not: pad every edge by the window's largest pad. A
    crossing rule's node count follows twice its SHORT side, which the
    frontier state sets, while the largest pad belongs to the farthest state:
    on the Si 101 map-0 boxes the two crossing rules certify at 127 + 135
    nodes that way against 108 + 94 per state (one-shot 86 + 93; the retired
    flat +-2 eV pad on top gave 167 + 161).
    runs/runtime/sigma_quad_20260924/m3_planner.
    """
    a_lo, a_hi, gamma_lo, gamma_hi = spec["pole_extent"]
    frac = _SC_POLE_PAD_FRACTION
    padded_poles = [(
        a_lo - frac * abs(a_lo),
        a_hi + frac * abs(a_hi),
        max(0.0, gamma_lo - frac * abs(gamma_lo)),
        gamma_hi + frac * abs(gamma_hi),
    )]
    # A treatment ceiling is a fixed-domain contract, not a live pole. Keep
    # its declared support separate from the current pole statistics and
    # factor references, and use it only to size the frozen certificate.
    if "sc_support_pole_extent" in spec:
        padded_poles.append(tuple(spec["sc_support_pole_extent"]))
    states = np.asarray(spec["states"], dtype=np.float64)
    pad_ry = sc_state_pad_ev(states * RYD_TO_EV) / RYD_TO_EV
    low, high = int(np.argmin(states - pad_ry)), int(np.argmax(states + pad_ry))
    # Membership is re-selected on every map: a state past the window's own
    # selector bound belongs to the neighbouring window, whose certificate
    # covers it. Padding across the bound only drags a sign-definite edge
    # toward zero until the zero-side cap stops it (TaAs 4^3 metal SC,
    # 2026-09-24: val:bulk at 0.0013 Ry against a 0.0276 Ry selector edge,
    # a 14.6 Ry-tall relative box that certified with 0.04% margin and was
    # refused on refit).
    state_lo, state_hi = spec.get("state_interval", (-np.inf, np.inf))
    padded_states = np.asarray(
        [max(states[low] - pad_ry[low], state_lo),
         min(states[high] + pad_ry[high], state_hi)])
    pole_box, _, _ = _box_for_window(
        spec["frequencies"], padded_states, padded_poles,
        spec["pole_sign"], eta)
    box = [
        min(spec["box"][0], pole_box[0]),
        max(spec["box"][1], pole_box[1]),
        min(spec["box"][2], pole_box[2]),
        max(spec["box"][3], pole_box[3]),
    ]
    # A SIGN-DEFINITE SUPPORT STAYS SIGN-DEFINITE.  The pads above can
    # push the zero-side edge of a strictly negative (or positive) support
    # across zero, which turns an easy relative rule into a crossing rule
    # the builder cannot certify: the Na conduction pole-tail window was
    # retained at sup=0.04 against eps=1e-4 with 906 nodes, while its actual
    # support has a 24-node rule at eps (lane QUADCHECK, 2026-09-05).  The
    # pad toward zero stops the zero-side edge at ``_SC_ZERO_SIDE_CAP`` of
    # its distance to zero; a support that really crosses later is a box
    # escape and rebuilds.
    if spec["kind"] == "sign_definite_negative":
        box[1] = min(box[1], _SC_ZERO_SIDE_CAP * spec["box"][1])
    elif spec["kind"] == "sign_definite_positive":
        box[0] = max(box[0], _SC_ZERO_SIDE_CAP * spec["box"][0])
    # Membership can change without appreciable state motion: a state just
    # outside a tail at map 0 can enter it at map 1. Cover the selector's
    # guaranteed sign gap, not the accidental nearest initial sample.
    if "sc_selector_gap_ry" in spec:
        gap = float(spec["sc_selector_gap_ry"])
        if spec["kind"] == "sign_definite_negative":
            box[1] = max(box[1], -gap)
        elif spec["kind"] == "sign_definite_positive":
            box[0] = min(box[0], gap)
    padded = dict(spec)
    padded["box"] = tuple(float(value) for value in box)
    padded["kind"] = (
        "sign_definite_positive" if box[0] > 0.0 else
        "sign_definite_negative" if box[1] < 0.0 else "crossing")
    padded["sc_unpadded_box"] = tuple(spec["box"])
    padded["sc_state_pad_ev"] = (float(pad_ry[low] * RYD_TO_EV),
                                 float(pad_ry[high] * RYD_TO_EV))
    padded["sc_pole_pad_fraction"] = _SC_POLE_PAD_FRACTION
    if not _box_contains(padded["box"], spec["box"]):
        raise RuntimeError(
            f"SC fixed-rule padding failed to contain {spec['name']!r}")
    return padded


class _RuleValidityFailure(RuntimeError):
    """The frozen rule is numerically invalid on the current map."""


def _fixed_fit_for_spec(entry, spec):
    """Reuse one immutable rule and recheck current factor growth."""
    fit = dict(entry["fit"])
    growth = _factor_growth(
        fit["times"], spec["pole_sign"], spec["states"],
        spec["pole_stats"], spec["E_ref_A"], spec["E_ref_B"])
    if max(growth) > _FACTOR_GROWTH_CAP:
        raise _RuleValidityFailure(
            f"GATE sc_fixed_rule_validity: Sigma box window {spec['name']!r} "
            f"refused while reusing its fixed SC rule: factored log growth "
            f"{max(growth):.6g} exceeds {_FACTOR_GROWTH_CAP:g}")
    fit["factor_growth"] = growth
    fit["cache_status"] = "hit:sc-fixed"
    fit["rule_table"] = "none"
    fit["seconds"] = 0.0
    # A failed write was announced on the iteration that attempted it. Reusing
    # the in-memory fixed rule must not repeat the old warning every SC map.
    fit["cache_write_warning"] = None
    fit["rule_table_warning"] = None
    fit["cache_lookup_warnings"] = ()
    return fit


def _fit_fixed_sc_rules(
    specs, eta, *, eps, cache_dir, session, material_class=None,
):
    """One frozen, padded rule set from the first map; map 0 itself one-shot.

    The first call serves its own map with the ordinary one-shot rules, so SC
    map 0 equals the one-shot G0W0 bit for bit, and on the same call freezes
    a rule set on those boxes padded by :func:`_sc_padded_box_spec`. Every
    later map reuses those tau nodes while they hold. The one-shot rules are
    served only from their own plan, so no padded certificate reaches map 0;
    both sets are fitted in one balanced parallel pass.
    Freezing at map 0 rather than map 2 (P2-E, 2026-09-24): Fe 4^3 bispinor
    SC plan 48.6 -> ~28 s over 4 maps, since a metal's windows barely move and
    maps 1-2 had refit all 12; on CrI3 8x8, whose gap opens at map 1, the map-1
    escapes refit around the new boxes, as the map-2 freeze did. Three
    events refit:

    * a window that escapes its certificate box, changes error currency, or
      did not exist when the set froze is refit on this map, alone (owner
      2026-09-22, TaAs semimetal SC); the receipt names each refit window
      with its reason and keeps the freezing map's pair cost;
    * a metal<->insulator flip re-initializes the set under the new class;
    * a rule-validity failure during reuse (factored-log growth above the
      cap) refits that one window.
    """
    rows = list(specs)
    iteration = int(session.get("call_count", 0)) + 1
    session["call_count"] = iteration
    named_class = None if material_class is None else str(material_class)
    if named_class is not None:
        previous_class = session.get("material_class")
        if previous_class is None:
            session["material_class"] = named_class
        elif str(previous_class) != named_class:
            session["material_class"] = named_class
            session.pop("rules", None)
            session["class_flip"] = f"{previous_class}->{named_class}"
    if "rules" in session and (
            float(session["eta_ry"]) != float(eta)
            or float(session["eps"]) != float(eps)):
        raise ValueError(
            "SC fixed quadrature session changed currency: "
            f"eta {session['eta_ry']!r}->{eta!r}, "
            f"eps {session['eps']!r}->{eps!r}")
    if "rules" not in session:
        session["eta_ry"] = float(eta)
        session["eps"] = float(eps)
        padded = [_sc_padded_box_spec(spec, eta) for spec in rows]
        (served, fit_rows), (fits, padded_rows) = fit_sigma_box_spec_groups(
            [(rows, True), (padded, False)], eta, eps=eps, cache_dir=cache_dir)
        session["rules"] = {
            spec["name"]: {
                "fit": dict(fit, cache_status=f"init:{fit['cache_status']}"),
                "padded_box": tuple(padded_spec["box"]),
                "initial_box": tuple(spec["box"]),
            } for spec, padded_spec, fit in zip(rows, padded, fits)}
        session["initial_window_tau_pairs"] = int(sum(
            fit["node_count"] for fit in fits))
        return served, list(fit_rows) + list(padded_rows), {
            "iteration": iteration, "mode": "one-shot", "initialized": True,
            "rebuilt": (), "recompute_reasons": (), "escaped": 0,
            "rebuild_count_total": int(session.get("rebuild_count", 0)),
            "material_class": session.get("material_class"),
            "class_flip": session.pop("class_flip", None),
        }

    rules = session["rules"]
    # Owner 2026-09-22 (TaAs semimetal SC): a window that escapes its frozen
    # box, or a new window, is refit on this map instead of refusing. Only
    # those windows are refit; every other window keeps its frozen nodes
    # (whole-set refits measured 9/9 windows per escape on the CrI3 8x8 SC
    # gate, 2026-09-24).
    escape_reasons = {}
    for spec in rows:
        entry = rules.get(spec["name"])
        if entry is None:
            escape_reasons[spec["name"]] = "absent when the rules froze"
            continue
        reasons = _box_escape_reasons(entry["fit"]["rule_box"], spec["box"])
        if bool(entry["fit"]["relative"]) != (spec["kind"] != "crossing"):
            reasons.append("absolute/relative error currency changed")
        if reasons:
            escape_reasons[spec["name"]] = "escape: " + "; ".join(reasons)
    fit_rows = []
    if escape_reasons:
        escaped = [spec for spec in rows if spec["name"] in escape_reasons]
        padded = [_sc_padded_box_spec(spec, eta) for spec in escaped]
        # Its own stage: a refit is host work between the W response and the
        # Sigma tau sweep, 2-27 s per CrI3 8x8 SC map (P2-S, 2026-09-25).
        with timing.section("sigma.rule_refit", announce=True,
                            label=f"Sigma rule refit ({len(padded)} escaped windows)"):
            new_fits, fit_rows = fit_sigma_box_specs(
                padded, eta, eps=eps, cache_dir=cache_dir,
                cache_build_widen=False)
        for spec, padded_spec, fit in zip(escaped, padded, new_fits):
            rules[spec["name"]] = {
                "fit": dict(fit, cache_status=f"rebuild:sc-fixed:{fit['cache_status']}"),
                "padded_box": tuple(padded_spec["box"]),
                "initial_box": tuple(spec["box"]),
                "rebuilt_at_iteration": iteration,
                "rebuild_reason": escape_reasons[spec["name"]],
            }
    # A product window may temporarily have no live state/pole tuples.  Keep
    # its frozen receipt in ``rules`` and simply omit its zero
    # contribution from this map; if it reappears, the containment check
    # above applies to it again.
    recomputed = dict(escape_reasons)
    fits = []
    for spec in rows:
        entry = rules[spec["name"]]
        if spec["name"] in escape_reasons:
            fits.append(dict(entry["fit"]))
            continue
        try:
            fits.append(_fixed_fit_for_spec(entry, spec))
        except _RuleValidityFailure as exc:
            padded_spec = _sc_padded_box_spec(spec, eta)
            with timing.section("sigma.rule_refit", announce=True,
                                label="Sigma rule refit (validity)"):
                new_fits, new_rows = fit_sigma_box_specs(
                    [padded_spec], eta, eps=eps,
                    cache_dir=cache_dir, cache_build_widen=False)
            rebuilt = dict(new_fits[0])
            rebuilt["cache_status"] = "rebuild:sc-fixed-validity"
            rules[spec["name"]] = {
                "fit": rebuilt,
                "padded_box": tuple(padded_spec["box"]),
                "initial_box": tuple(spec["box"]),
                "rebuilt_at_iteration": iteration,
                "rebuild_reason": f"validity: {exc}",
            }
            recomputed[spec["name"]] = f"validity: {exc}"
            fit_rows.extend(new_rows)
            fits.append(dict(rebuilt))
    if recomputed:
        session["rebuild_count"] = int(
            session.get("rebuild_count", 0)) + len(recomputed)
        if process_rank() == 0:
            for name, reason in recomputed.items():
                print(f"  [sc-fixed] iteration {iteration}: recomputed the "
                      f"rule for {name!r} ({reason})")
    return fits, fit_rows, {
        "iteration": iteration, "mode": "frozen", "initialized": False,
        "rebuilt": tuple(name for name in (spec["name"] for spec in rows)
                         if name in recomputed),
        "recompute_reasons": tuple(sorted(recomputed.items())),
        "escaped": len(escape_reasons),
        "rebuild_count_total": int(session.get("rebuild_count", 0)),
        "material_class": session.get("material_class"),
        "class_flip": session.pop("class_flip", None),
    }


def sigma_box_executor_nodes(
    fit, pole_sign, eta_ry, *, one_sided_hermitian=False,
):
    """Convert one accepted box rule to the dynamic-Sigma executor contract.

    The shared physical convention is

    ``time_exec = pole_sign * time`` and
    ``alpha_exec = weight * exp(-eta * time_exec)``.

    The fitted lower-half-plane valence rule has already received the exact
    ``time=-conj(time), weight=conj(weight)`` transformation in
    :func:`fit_sigma_box_specs`.  ``one_sided_hermitian`` retains PPM's
    crossing channel and its global completion ``(Z-Z^dagger)/(2i)``.  In
    that contract the coefficient is multiplied by ``i`` so completion is
    exactly the Hermitian part of the full causal box sum:

    ``(i Q - (i Q)^dagger)/(2i) = (Q + Q^dagger)/2``.

    No channel is collapsed and the equality holds for general complex box
    times and weights, not only a real one-sided sine grid.
    """
    sign, eta = float(pole_sign), float(eta_ry)
    if sign not in (-1.0, 1.0):
        raise ValueError("Sigma box pole_sign must be +1 or -1")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    time_exec = sign * np.asarray(fit["times"], np.complex128)
    alpha_exec = (np.asarray(fit["weights"], np.complex128)
                  * np.exp(-eta * time_exec))
    if one_sided_hermitian:
        alpha_exec = 1.0j * alpha_exec
    return MinimaxNodes(
        t=jnp.asarray(time_exec, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_exec, dtype=jnp.complex128))


def plan_sigma_windows(
    pole_summaries,
    branches,
    omega_ry,
    eta_ry,
    *,
    eps,
    cache_dir,
    print_fn=print,
    edge_factor=1.5,
    fixed_rule_session=None,
    analytic_line=False,
    material_class=None,
    fixed_pole_support_ry=None,
    certificate_pole_summaries=None,
):
    """Build the complete MPA Sigma quadrature from raw support boxes.

    ``analytic_line`` asks the service for its fixed-height rule only when
    the box crosses zero and all poles are real. PPM supplies that request;
    generic MPA and shared-pole W keep their box-rule contracts.

    Parameters
    ----------
    pole_summaries
        Concatenated output of ``summarize_sigma_poles``.  Each row contains
        only live per-pole extrema for the all/shallow/deep selectors.
    branches
        Causal ``_SigmaBranch`` rows.  Their masks and optional occupation
        weights are exactly the state support the executor will consume.
    omega_ry
        Requested external frequency grid in Ry.  Used to verify each
        branch's global indices before constructing its denominator corners.
    eta_ry
        Positive retarded broadening in Ry.  It enters both the box's
        imaginary extent and, exactly once, the executor weights.
    eps
        Per-window uniform sup ceiling.  The rule builder certifies directly
        at this value, using relative error on sign-definite boxes and
        peak-relative error on crossing boxes; this matches the measured
        Sigma error currency.
    cache_dir
        Directory for immutable box-rule certificates, or ``None``.
    fixed_rule_session
        Mutable run-local receipt used only by a multi-map SC calculation.
        Its first call serves the one-shot rules and certifies the same
        boxes padded by the fixed SC policy; every later map reuses those
        nodes by containment.  ``None``
        preserves the ordinary one-shot planner byte-for-byte.
    fixed_pole_support_ry
        Optional positive real-pole endpoint that the frozen fixed rule
        must cover. The declared ``[0, endpoint]`` interval is intersected
        with each existing pole selector only for the SC certificate; live
        pole statistics, references and executor intervals remain unchanged.
    certificate_pole_summaries
        Optional ``summarize_*`` rows of a pole census that contains
        ``pole_summaries`` (a shared-pole map's union over its sectors).
        They set only each window's certificate box, so every sector call
        of one map requests the same boxes and reuses one set of fits; the
        executor's pole selection, factor references and window kind stay
        this call's own.  A window whose union box would change kind keeps
        its own box.

    Returns
    -------
    windows, geometry
        Executable ``SharedSigmaWindow`` rows and a JSON-compatible planning
        report.

    Notes
    -----
    Tempting, and why not:

    * Histogram-weight the support: that made a low-mass Fermi state 0.95 meV
      wrong on Na.  Every live tuple gets the same box certificate instead.
    * Merge a whole branch: it replaces three cheap sign-aware boxes by one
      wide crossing box and can silently reintroduce measure dependence.
    * Use peak-relative sup on sign-definite tails: a semicore term at
      ``|d|/eta ~ 200`` then spends about 200 times the intended relative
      error; the builder's relative currency measured 0.1 rather than 4 meV.
    * Retry at tighter ``eps``: a sup, noise, growth, or resource refusal is
      already about this box; a hidden retry is a second accuracy policy.
    * Widen near-zero edges for cache hits: those edges set crossing rank and
      do not drift; measured 3% all-edge widening added 67 pairs on Na.
    * Reserve another factor for the number of product windows: the windows
      partition the causal ``(state, pole, omega-sign)`` tuples, so every
      denominator error enters Sigma exactly once.  If a box certificate is
      ``|Q(d)-1/d| <= eps/eta`` (and ``<= eps/|d|`` on a sign-definite box),
      then one state's error obeys
      ``|delta Sigma_n| <= sum_p |M_np| eps/eta``.  There is no window-count
      factor.  Lane E's blanket 0.1 reserve raised the Si/Na pair counts from
      551/579 to 690/831 (+25.23%/+43.52%) without measurable accuracy gain.
    """
    started = time.perf_counter()
    eta, tolerance = float(eta_ry), float(eps)
    edge = float(edge_factor)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("sigma_quadrature_eps must lie in (0, 1)")
    if not np.isfinite(edge) or edge < 0.0:
        raise ValueError("sigma_window_edge_factor must be nonnegative")
    fixed_pole_support = None
    if fixed_pole_support_ry is not None:
        fixed_pole_support = float(fixed_pole_support_ry)
        if (fixed_rule_session is None or not np.isfinite(fixed_pole_support)
                or fixed_pole_support <= 0.0):
            raise ValueError(
                "fixed_pole_support_ry requires a fixed SC session and a "
                "finite positive endpoint")
        previous = fixed_rule_session.setdefault(
            "pole_support_ry", fixed_pole_support)
        if float(previous) != fixed_pole_support:
            raise ValueError(
                "fixed SC pole support changed after initialization: "
                f"{previous!r}->{fixed_pole_support!r} Ry")
    branch_rows = list(branches)
    summaries = tuple(pole_summaries)
    if not summaries:
        raise ValueError("Sigma box planning needs at least one pole summary")
    omega_grid = np.asarray(omega_ry, dtype=np.float64)
    state_rows, geometry = _product_geometry(branch_rows, eta, edge)

    specs, branch_reports = [], []
    for branch, (state_shape, raw_energy, flat_indices) in zip(
            branch_rows, state_rows):
        positions = np.asarray(branch.omega_idx, dtype=np.int64)
        frequencies = omega_grid[positions]
        expected = (-np.asarray(branch.omega_abs)
                    if branch.neg_omega_half else np.asarray(branch.omega_abs))
        if not np.allclose(frequencies, expected, rtol=0.0, atol=1.0e-13):
            raise ValueError(f"Sigma branch {branch.tag!r} indices disagree")
        pole_sign = 1.0 if branch.space == "cond" else -1.0
        report = {
            "tag": branch.tag, "space": branch.space,
            "negative_frequency_half": bool(branch.neg_omega_half),
            "live_state_count": int(raw_energy.size),
            "plan_start": len(specs), "windows": [],
        }
        omega_abs = np.asarray(branch.omega_abs, np.float64)
        for (name, state_lo, state_hi, selector, pole_lo, pole_hi,
             omega_lo, omega_hi) in _state_products(
                 branch, geometry["state_edge_ry"], geometry["pole_edges_ry"]):
            local = np.nonzero(
                (raw_energy > state_lo) & (raw_energy <= state_hi))[0]
            owned = np.nonzero(
                (omega_abs >= omega_lo) & (omega_abs < omega_hi))[0]
            pole_indices, pole_stats = _pole_rows(summaries, selector)
            if not local.size or not owned.size or not pole_indices.size:
                continue
            states = raw_energy[local]
            spec = make_sigma_box_spec(
                name=f"{branch.tag}:{name}", frequencies=frequencies[owned],
                states=states, pole_stats=pole_stats,
                pole_sign=pole_sign, eta_ry=eta)
            if certificate_pole_summaries is not None:
                _, union_stats = _pole_rows(certificate_pole_summaries, selector)
                union = (make_sigma_box_spec(
                    name=spec["name"], frequencies=frequencies[owned],
                    states=states,
                    pole_stats=union_stats, pole_sign=pole_sign, eta_ry=eta)
                    if union_stats else None)
                if (union is not None and union["kind"] == spec["kind"]
                        and _box_contains(union["box"], spec["box"])):
                    spec.update(box=union["box"],
                                raw_real_support=union["raw_real_support"],
                                pole_extent=union["pole_extent"])
            spec["analytic_line"] = bool(analytic_line)
            if fixed_pole_support is not None:
                support_lo = max(0.0, float(pole_lo))
                support_hi = min(fixed_pole_support, float(pole_hi))
                if support_hi > support_lo:
                    spec["sc_support_pole_extent"] = (
                        support_lo, support_hi, 0.0, 0.0)
            if (fixed_rule_session is not None
                    and name in ("bulk", "state_tail", "pole_tail",
                                 "omega_tail")
                    and geometry["state_edge_ry"] > 0.0
                    and (fixed_pole_support is not None or all(
                        lo >= 0.0 and gamma_lo == gamma_hi == 0.0
                        for lo, _, gamma_lo, gamma_hi in pole_stats))):
                # The selectors guarantee this gap for positive real poles,
                # including scalar W without a sector treatment ceiling
                # (omega_tail: its |omega| cut follows this map's excursion).
                # Cover future selector members, not only initial samples.
                spec["sc_selector_gap_ry"] = (
                    _BOX_SIGN_FRACTION * geometry["state_edge_ry"])
            spec.update({
                "branch": branch,
                "state_indices": flat_indices[local],
                "state_shape": state_shape.shape,
                "state_interval": (float(state_lo), float(state_hi)),
                "pole_indices": pole_indices,
                "pole_bounds": (float(pole_lo), float(pole_hi)),
                "omega_interval": (float(omega_lo), float(omega_hi)),
                "omega_abs": omega_abs[owned],
                "omega_idx": positions[owned],
                "branch_report": report,
            })
            specs.append(spec)
        report["plan_stop"] = len(specs)
        report["window_count"] = report["plan_stop"] - report["plan_start"]
        branch_reports.append(report)

    fixed_receipt = None
    if fixed_rule_session is None:
        fits, fit_rows = fit_sigma_box_specs(
            specs, eta, eps=tolerance, cache_dir=cache_dir)
    else:
        fits, fit_rows, fixed_receipt = _fit_fixed_sc_rules(
            specs, eta, eps=tolerance,
            cache_dir=cache_dir, session=fixed_rule_session,
            material_class=material_class)
    if process_rank() == 0:
        announced = set()
        for fit in fits:
            warnings = tuple(fit.get("cache_lookup_warnings", ()))
            for write_warning in (fit.get("cache_write_warning"),
                                  fit.get("rule_table_warning")):
                if write_warning:
                    warnings += (write_warning,)
            for warning in warnings:
                if warning not in announced:
                    print_fn(warning)
                    announced.add(warning)
    # The (window, tau) pair count is reported, never refused on: the owner
    # eliminated the pair ceiling (2026-09-02).  A count above what a deck
    # can afford is a planning question answered by eps and the window
    # geometry, not a runtime refusal (TASTE 70).
    pairs = sum(row["node_count"] for row in fits)
    frozen = fixed_receipt is not None and fixed_receipt["mode"] == "frozen"

    output = []
    for spec, fit in zip(specs, fits):
        mask = np.zeros(int(np.prod(spec["state_shape"])), dtype=bool)
        mask[np.asarray(spec["state_indices"], np.int64)] = True
        external_sign = -1 if spec["branch"].neg_omega_half else 1
        window = _SigmaWindow(
            name=spec["name"],
            nodes=sigma_box_executor_nodes(
                fit, spec["pole_sign"], eta),
            mask_A=mask.reshape(spec["state_shape"]),
            E_ref_A=spec["E_ref_A"], E_ref_B=spec["E_ref_B"],
            omega_sign=int(spec["pole_sign"]) * external_sign,
            project="full", prefactor=-1.0,
            max_error=fit["sup_error"],
            provenance=(
                f"uniform denominator box {spec['box']}; "
                f"{fit['one_line']}; cache={fit['cache_status']}; "
                f"factor_growth={fit['factor_growth']}"))
        output.append(SharedSigmaWindow(
            window=window, E_A=spec["branch"].E_A,
            omega_abs=spec["omega_abs"], omega_idx=spec["omega_idx"],
            pole_indices=spec["pole_indices"],
            bounds=_pole_bounds(
                len(spec["pole_indices"]), *spec["pole_bounds"]),
            phase_real=np.zeros(len(spec["pole_indices"]), dtype=bool),
            band_weight=spec["branch"].band_weight,
            space=spec["branch"].space))
        spec["branch_report"]["windows"].append({
            "name": spec["name"], "kind": spec["kind"],
            "state_interval_ry": list(spec["state_interval"]),
            "pole_interval_ry": list(spec["pole_bounds"]),
            "omega_abs_interval_ry": list(spec["omega_interval"]),
            "pole_indices": spec["pole_indices"].tolist(),
            "raw_real_support_ry": list(spec["raw_real_support"]),
            "box_ry": list(spec["box"]), "rule_box_ry": list(fit["rule_box"]),
            "node_count": fit["node_count"],
            "node_digest": fit["node_digest"],
            "criterion": ("relative" if fit["relative"]
                          else "peak-relative"),
            "sup_error": fit["sup_error"], "eps": tolerance,
            "requested_eps": tolerance,
            "kappa_max": fit["kappa_max"],
            "roundoff_amplification": fit["roundoff_amplification"],
            "runtime_noise_bound": fit["noise_bound"],
            "runtime_noise_budget": fit["noise_budget"],
            "factor_growth": list(fit["factor_growth"]),
            "cache_status": fit["cache_status"],
            "rule_table": fit["rule_table"],
            "fit_seconds": fit["seconds"],
            "sc_fixed_rule": frozen,
            "sc_fixed_padded_box_ry": (
                list(fixed_rule_session["rules"][spec["name"]]["padded_box"])
                if frozen else None),
        })
        if _resolve_uniform_rule_trace() and process_rank() == 0:
            print_fn(
                f"[uniform-box] {spec['name']} "
                f"support_re={spec['raw_real_support']} box={spec['box']} (Ry)")

    distinct = sum(len({
        (float(value.real), float(value.imag))
        for row in output[report["plan_start"]:report["plan_stop"]]
        for value in np.asarray(row.window.nodes.t)
    }) for report in branch_reports)
    geometry.update({
        "planner": "uniform_denominator_boxes",
        "eta_ry": eta, "eps": tolerance,
        "rule_eps": tolerance,
        "cache_dir": cache_dir, "rule_cache_schema": _RULE_CACHE_SCHEMA,
        "rule_table_dir": resolve_sigma_rule_table_dir(cache_dir),
        "rule_table_format": _RULE_TABLE_FORMAT,
        "rule_table_lookups": {
            status: sum(1 for row in fit_rows
                        if (row.get("value") or {}).get("rule_table") == status)
            for status in ("hit", "built")},
        "n_windows": len(output),
        "window_tau_pairs": pairs, "distinct_tau_count": distinct,
        "plan_seconds": time.perf_counter() - started,
        "planning_process_count": int(process_count()),
        "critical_fit_wall_seconds": max(
            (row["wall_seconds"] for row in fit_rows), default=0.0),
        "branches": branch_reports,
    })
    if fixed_rule_session is not None:
        geometry.update({
            "sc_fixed_quadrature": True,
            "sc_rule_mode": fixed_receipt["mode"],
            "sc_fixed_iteration": int(fixed_receipt["iteration"]),
            "sc_fixed_initialized": bool(fixed_receipt["initialized"]),
            "sc_fixed_rebuilds_this_iteration": len(
                fixed_receipt.get("rebuilt", ())),
            "sc_fixed_rebuilt_windows": list(fixed_receipt.get("rebuilt", ())),
            "sc_fixed_escaped_windows": int(fixed_receipt.get("escaped", 0)),
            "sc_fixed_total_rebuild_count": int(
                fixed_receipt.get("rebuild_count_total", 0)),
            "sc_fixed_initial_window_tau_pairs": fixed_rule_session.get(
                "initial_window_tau_pairs"),
            "sc_fixed_material_class": fixed_receipt.get("material_class"),
            "sc_fixed_recompute_reasons": dict(
                fixed_receipt.get("recompute_reasons", ())),
            "sc_fixed_class_flip": fixed_receipt.get("class_flip"),
            "sc_state_edge_padding_ev": max(
                (float(np.max(sc_state_pad_ev(spec["states"] * RYD_TO_EV)))
                 for spec in specs), default=0.0),
            "sc_pole_extent_padding_fraction": _SC_POLE_PAD_FRACTION,
            "sc_fixed_pole_support_ry": fixed_pole_support,
        })
    else:
        geometry["sc_fixed_quadrature"] = False
    # Keep the accepted rule identity in the normal scientific report,
    # including cache-off and repeated SC planning calls.
    if process_rank() == 0:
        print_fn("Sigma quadrature receipt: " + _receipt_json(geometry))
    return output, geometry


__all__ = [
    "fit_sigma_box_specs",
    "make_sigma_box_spec",
    "plan_sigma_windows",
    "resolve_sigma_box_cache_dir",
    "sigma_box_executor_nodes",
]
