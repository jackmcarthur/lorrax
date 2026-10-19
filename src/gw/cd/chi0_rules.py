"""chi0 rules of the contour-deformation route: the shared-pole bank's response quadrature on the CD
memory windows, built in the job or precomputed on a CPU node.

Where this sits (docs/dev/cd_frequency_integration.md): :func:`gw.cd.sigma.bank_chi0` evaluates chi0 at
every CD point (imaginary nodes ``i u``; real knots ``j h + i eta`` with their FD-9 ghosts) through
``response_bank.integrate_response_group``.  Its rules come from here, one rule request per CD memory
window (:func:`cd_windows`: the chunks the CD streams evaluate, in stream order), so a window's
Green-pair sweep is that of its own members:

* a window whose points all have ``|Re z| < lo`` (an insulator's imaginary nodes): the bank's
  noncrossing ``minimax.response_laplace_rule`` (ordered, reference ``lo``): positive real times,
  certified on the rounded arrays, independent of Im z.  :func:`_laplace_group` puts its rows on the
  direct stream's slots: forward = even + odd fits ``1/(d - z)``, reverse = even - odd fits
  ``1/(d + z)``.  A point the certificate refuses goes to the bank tree when its pencil is small
  (:data:`CD_FALLBACK_PENCIL_SIZE`; the large-u nodes), else, on a deck whose symmetry service
  authenticated time reversal (the two orientations equal, the odd channel exactly zero), to the
  even-only rule; otherwise the run refuses (GATE cd_chi0_rule_unfit);
* every other window (the real knots): ``response_bank.response_quadrature`` with that window as its
  request and the CD rule options -- ``minimax.ResponseGroupTree`` at :data:`CD_RULE_CAPACITY` nodes
  and the mass gate :func:`cd_max_kappa`, on the host pencils -- halved where a window does not fit.

Sampled errors are accepted against the tolerance exactly as in the bank; only the node capacity and
the coefficient-mass gate are the CD's own (the bank's defaults are untouched).

Precompute: ``python -m gw.cd.chi0_rules`` builds the same rules from the energy table and the CD
geometry and writes them with their identity (:func:`rule_identity`); a CD run given
``cd_chi0_rules_file`` loads them and refuses any mismatch (GATE cd_chi0_rules_identity).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

from common.units import RYD_TO_EV

#: Layout version of the rule file and its identity.
RULES_VERSION = "cd_chi0_rules_v1"
#: Default largest node count of one CD shared set (the host pencil's rank; rows padded to twice it).
#: The bank's own is ``minimax.RESPONSE_RULE_CAPACITY`` = 192; the ZnO pricing note (v3) projects
#: 669 (408-1060) nodes for a 34 eV top knot at eta = 0.25 eV.  Deck key ``cd_chi0_rule_capacity``.
CD_RULE_CAPACITY = 1088
#: The CD coefficient-mass gate admits rounding up to this share of the tolerance: kappa * eps <=
#: share * tol (the bank's 5000 is kappa * eps = 1.1e-12, i.e. 1e-4 of its production 1e-8).
CD_KAPPA_ROUNDING_SHARE = 1e-2
#: A point the Laplace certificate refuses takes the bank tree when its pencil has at most this many
#: samples (the real-knot windows' pencils are 16,619 on the ZnO deck: ~44 GB of host memory).
CD_FALLBACK_PENCIL_SIZE = 20000
_PLAN_KEYS = ("lo", "hi", "decay_rate", "amplitude", "metallic", "reference")
_SLICED = ("value", "derivative", "sampled_error", "coefficient_mass")


def cd_max_kappa(tolerance):
    """The CD mass gate: the bank's ``minimax.RESPONSE_MAX_KAPPA``, or larger where kappa * eps <= 1 % of tol."""
    import minimax
    return max(float(minimax.RESPONSE_MAX_KAPPA),
               CD_KAPPA_ROUNDING_SHARE * float(tolerance) / float(np.finfo(np.float64).eps))


def cd_rule_options(tolerance, capacity=None):
    """``minimax.ResponseGroupTree`` options of the CD real-knot windows."""
    return dict(capacity=int(CD_RULE_CAPACITY if capacity is None else capacity),
                max_kappa=cd_max_kappa(tolerance))


# ------------------------------------------------------------------------------- CD points and windows
def cd_windows(u_ry, h_ry, eta_ry, J_pad, chunk_i, chunk_r, ghosts):
    """The point arrays of every CD chunk, exactly as ``gw.cd.sigma`` evaluates them.

    ``u_ry`` is the padded imaginary-node table (``n_imag_pad`` entries).  Returns ``(imag, real)``:
    ``imag[c] = 1j * u_ry[c0:c0 + chunk_i]``; ``real[0]`` the knots ``-ghosts .. chunk_r + ghosts - 1``
    and ``real[c]`` (c > 0) the knots ``j0 + ghosts .. j0 + chunk_r + ghosts - 1`` (the rolling
    FD-9 ghosts), each ``h_ry * j + 1j * eta_ry``.
    """
    u_ry = np.asarray(u_ry, dtype=np.float64)
    imag = [1j * u_ry[c0:c0 + int(chunk_i)] for c0 in range(0, u_ry.size, int(chunk_i))]
    real = []
    for j0 in range(0, int(J_pad), int(chunk_r)):
        first = j0 - ghosts if j0 == 0 else j0 + ghosts
        real.append(float(h_ry) * np.arange(first, j0 + int(chunk_r) + ghosts) + 1j * float(eta_ry))
    return imag, real


def window_members(imag, real):
    """Distinct points (first-seen order) and each window's members (indices into the points).

    A point repeated inside or across windows (the padded imaginary nodes) belongs to the first
    window that evaluates it; a window left without members is dropped.
    """
    index, windows = {}, []
    for kind, chunks in (("imag", imag), ("real", real)):
        for c, chunk in enumerate(chunks):
            members = []
            for z in np.asarray(chunk, dtype=np.complex128).ravel().tolist():
                if complex(z) not in index:
                    index[complex(z)] = len(index)
                    members.append(index[complex(z)])
            if members:
                windows.append(dict(kind=kind, chunk=c, members=members))
    return np.asarray(list(index), dtype=np.complex128), windows


# ------------------------------------------------------------------------------- identity
def _star_sha(table):
    """sha256 of the distinct k rows: equal for a wedge table and its star-unfolded copy."""
    rows = np.unique(np.ascontiguousarray(np.asarray(table, dtype=np.float64)), axis=0)
    return hashlib.sha256(rows.tobytes()).hexdigest()


def sample_plan(wfns, meta, points, *, mu_ry, tolerance):
    """The bank sample plan of the CD points, and the census tables it is built from.

    ``partial_at_mu`` follows ``shared_pole_recipe.bind_shared_pole_census``: some partially occupied
    band crosses mu across the k census.
    """
    from common.collectives import gather_to_host
    stop = int(wfns.slices.b4_logical) - int(wfns.slices.b0)
    energy = np.asarray(gather_to_host(wfns.enk), dtype=np.float64)[:, :stop]
    occupied = np.asarray(gather_to_host(wfns.occ), dtype=np.float64)[:, :stop]
    crossing = (energy.min(axis=0) <= mu_ry) & (energy.max(axis=0) >= mu_ry)
    partial = np.any((occupied > 0.0) & (occupied < 1.0), axis=0)
    count = min(int(meta.b_id_4_chi_user), int(wfns.slices.b4_logical)) - int(wfns.slices.b0)
    plan = dict(z_ry=points, distinct_id=np.arange(points.size), held=np.zeros(points.size, dtype=bool),
                role=np.full(points.size, "cd_point"), bank_rule_tolerance=float(tolerance),
                census=dict(mu_ry=float(mu_ry), partial_at_mu=bool(np.any(partial & crossing))))
    return plan, energy[:, :count], occupied[:, :count]


def plan_identity(support, metallic):
    """The rule plan's scalars as ``response_bank.response_quadrature`` snaps them (no held session)."""
    from gw.sigma_box_plan import snap_outward
    lo, hi = snap_outward(support["lo"] - 0., 1., -1), snap_outward(support["hi"] + 0., 1., +1)
    decay = snap_outward(support["decay_rate"], 1., -1)
    return dict(lo=float(lo), hi=float(hi), decay_rate=float(decay),
                amplitude=float(snap_outward(support["amplitude"], 1., +1)), metallic=bool(metallic),
                reference=0. if decay else float(lo))


def rule_identity(energy, occupied, points, windows, plan, tolerance, options, route):
    """What the rules are a function of; a loaded file must match every entry."""
    return dict(version=RULES_VERSION, route=route, tolerance=float(tolerance),
                options={k: float(v) for k, v in sorted((options or {}).items())},
                plan={k: plan[k] for k in _PLAN_KEYS}, points=int(points.size),
                points_sha256=hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest(),
                energies_sha256=_star_sha(energy), occupations_sha256=_star_sha(occupied),
                windows=[[w["kind"], int(w["chunk"]), [int(m) for m in w["members"]]] for w in windows])


def check_identity(got, want):
    """Refuse a rule file built for another deck, tolerance, z set or code layout."""
    bad = [k for k in want if got.get(k) != want[k]]
    if bad:
        detail = "; ".join(f"{k}: file {str(got.get(k))[:80]} vs run {str(want[k])[:80]}" for k in bad)
        raise ValueError(f"GATE cd_chi0_rules_identity: got: a rule file whose {', '.join(bad)} differ ({detail}); "
                         "want: the rules of this run's energies, tolerance, CD points and windows; why: the "
                         "nodes are a function of all of them (rebuild with python -m gw.cd.chi0_rules)")


# ------------------------------------------------------------------------------- Laplace windows
def _laplace_group(points, members, plan, rel_tol, node_capacity, ordered=True):
    """``minimax.response_laplace_rule`` on ``members``, as one direct-stream group (see module doc)."""
    import minimax
    z = points[list(members)]
    rule = minimax.response_laplace_rule(plan["lo"], plan["hi"], z, rel_tol=rel_tol, ordered=ordered,
                                         reference_ry=plan["reference"])
    t = np.asarray(rule["t"], dtype=np.float64)
    if t.size > node_capacity:
        raise ValueError(f"Laplace rule of {t.size} nodes exceeds the node capacity {node_capacity}")
    even, deven = np.asarray(rule["projection_value"]), np.asarray(rule["projection_derivative"])
    odd = np.asarray(rule["odd_projection_value"]) if ordered else np.zeros_like(even)
    dodd = np.asarray(rule["odd_projection_derivative"]) if ordered else np.zeros_like(deven)
    shape = (len(members), 2, int(node_capacity))
    value, derivative = np.zeros(shape, complex), np.zeros(shape, complex)
    value[:, 0, :t.size], value[:, 1, :t.size] = even + odd, even - odd
    derivative[:, 0, :t.size], derivative[:, 1, :t.size] = deven + dodd, deven - dodd
    times = np.zeros(int(node_capacity), complex)
    times[:t.size] = t
    bound = float(rule["certificate"]["maximum_bound"])
    mass = np.stack([z.imag[:, None] * np.abs(value).sum(-1), z.imag[:, None] ** 3 * np.abs(derivative).sum(-1)],
                    axis=-1)
    return dict(t=times, value=value, derivative=derivative, count=int(t.size),
                sampled_error=np.full((len(members), 2, 2), bound), coefficient_mass=mass,
                members=[int(m) for m in members], reference_ry=plan["reference"],
                provider="laplace" if ordered else "laplace_even", node_digest=rule.get("node_digest"))


def _laplace_cover(points, members, plan, rel_tol, node_capacity, ordered=True):
    """Laplace groups over ``members``, halved where the certificate refuses; also the refused points."""
    try:
        return [_laplace_group(points, members, plan, rel_tol, node_capacity, ordered)], []
    except (RuntimeError, ValueError, ArithmeticError):
        if len(members) == 1:
            return [], list(members)
        half = len(members) // 2
        a, ra = _laplace_cover(points, members[:half], plan, rel_tol, node_capacity, ordered)
        b, rb_ = _laplace_cover(points, members[half:], plan, rel_tol, node_capacity, ordered)
        return a + b, ra + rb_


# ------------------------------------------------------------------------------- build / stream
def build_rules(wfns, meta, points, windows, *, mu_ry, tolerance, slots, capacity=None,
                time_reversal=False, route="cd", serial=False, build=None, print_fn=print):
    """The rule payload of ``windows`` (see module doc) and the bank support it was built on.

    ``route="bank"`` is the bank's own single request over every point (no Laplace windows, the bank's
    capacity and gate).  ``serial`` builds the pencil windows one request at a time (per-window wall
    clock for the census) instead of in one ``_gather_group_rules`` round over every process.
    ``build`` (indices into ``windows``; default all) restricts the build to those windows, a partial
    (census) payload; the identity always lists every window.
    """
    import minimax
    from gw import response_bank as rb
    EV = float(RYD_TO_EV)
    points = np.asarray(points, dtype=np.complex128)
    if not points.size or not np.all(np.isfinite(points)) or not np.all(points.imag > 0):
        raise ValueError("GATE cd_chi0_points: got: an empty, nonfinite or non-causal CD point set; want: "
                         "finite points with Im z > 0; why: the bank rule fits upper-half-plane evaluations")
    if getattr(meta, "shared_pole_response_rules", None) is not None:
        raise ValueError("GATE cd_chi0_session: got: a held shared-pole rule session on meta; want: none; "
                         "why: the CD rule must neither reuse nor overwrite the bank's held-map rules")
    plan, energy, occupied = sample_plan(wfns, meta, points, mu_ry=mu_ry, tolerance=tolerance)
    receipt = {"correlation_count": 0}
    support = rb.response_support(wfns, meta, plan, receipt, print_fn=print_fn)
    pid = plan_identity(support, plan["census"]["partial_at_mu"])
    options = {} if route == "bank" else cd_rule_options(tolerance, capacity)
    node_capacity = 2 * int(options.get("capacity", minimax.RESPONSE_RULE_CAPACITY))
    rel_tol = float(tolerance) / pid["amplitude"]
    ident = rule_identity(energy, occupied, points, windows, pid, tolerance, options, route)
    laplace_ok = route == "cd" and pid["decay_rate"] == 0.0 and pid["lo"] > 0.0
    built = list(range(len(windows))) if build is None else sorted(int(i) for i in build)
    chosen = [windows[i] for i in built]
    lap = [w for w in chosen if laplace_ok and np.all(np.abs(points[w["members"]].real) < pid["lo"])]
    pen = [w for w in chosen if w not in lap]
    seconds = {}
    groups, fallback, even, refused = [], [], [], []
    started = time.monotonic()
    for w in lap:
        got, rest = _laplace_cover(points, list(w["members"]), pid, rel_tol, node_capacity, True)
        groups += got
        for m in rest:
            size = minimax.ResponseGroupTree(pid["lo"], pid["hi"], points[[m]], rel_tol=rel_tol,
                                             decay_rate=pid["decay_rate"]).size((0,))
            (fallback if size <= CD_FALLBACK_PENCIL_SIZE else even).append(m)
    if even:
        if not time_reversal:
            refused = even
        else:
            got, refused = _laplace_cover(points, even, pid, rel_tol, node_capacity, False)
            groups += got
    if refused:
        zs = ", ".join(f"{complex(points[m]) * EV:.4g}" for m in refused[:8])
        raise ValueError(
            f"GATE cd_chi0_rule_unfit: got: {len(refused)} CD point(s) (z/eV: {zs}{' ...' if len(refused) > 8 else ''}) "
            f"that the ordered Laplace rule refuses at tolerance {tolerance:g} and whose pencils exceed "
            f"{CD_FALLBACK_PENCIL_SIZE} samples; time reversal {'authenticated' if time_reversal else 'not authenticated'} "
            "(the even-only rule needs it); want: a certified bank rule at every point; why: chi0 has one provider, "
            "the bank quadrature (decide: even-only rule, a floored u, or a remapped imaginary rule)")
    seconds["laplace"] = time.monotonic() - started
    requests = [list(w["members"]) for w in pen] + ([fallback] if fallback else [])
    pencil_groups, rules = [], None
    started = time.monotonic()
    if route == "bank":
        batches = [None]                       # the bank's own request: response_groups over every point
    else:
        batches = [[r] for r in requests] if serial else ([requests] if requests else [])
    for i, batch in enumerate(batches):
        t0 = time.monotonic()
        try:
            rules = rb.response_quadrature(meta, plan, receipt, support, group_size=int(slots),
                                           print_fn=print_fn, requests=batch, rule_options=options)
        except ValueError as exc:
            raise ValueError(
                f"GATE cd_chi0_rule: got: the bank response rule refused a CD window ({exc}); interval "
                f"[{pid['lo'] * EV:.4g}, {pid['hi'] * EV:.4g}] eV, tolerance {tolerance:g}, options {options}; "
                "want: a rule meeting cd_chi0_rule_tolerance at every point; why: chi0 has one provider, the "
                "bank quadrature (its pencil grows like (hi - lo) / min Im z)") from exc
        if serial:
            seconds[f"request_{i}"] = time.monotonic() - t0
        pencil_groups += [dict(g, provider="pencil") for g in rules["plan"]["rule_groups"]]
        drift = [k for k in _PLAN_KEYS if rules["plan"][k] != pid[k]]
        if drift:
            raise ValueError(f"GATE cd_chi0_rules_plan: response_quadrature's plan {drift} differs from "
                             "chi0_rules.plan_identity; want: one snapping rule (update plan_identity)")
    seconds["pencil"] = time.monotonic() - started
    groups = pencil_groups + groups
    covered = sorted(m for g in groups for m in g["members"])
    if covered != sorted(int(m) for w in chosen for m in w["members"]):
        raise ValueError("GATE cd_chi0_rules_cover: the rule groups do not cover every built point exactly once")
    payload = dict(identity=ident, plan=pid, rule_groups=groups, seconds=seconds,
                   requests=[[int(m) for m in r] for r in requests], built=built,
                   partial=len(built) != len(windows))
    payload["census"] = census_rows(points, windows, groups, seconds if serial else None, requests)
    return payload, support


def stream_rules(payload, support, slots):
    """The ``rules`` dict ``integrate_response_group`` reads (``response_quadrature``'s return layout)."""
    refs = np.array(support["refs"], dtype=np.float64, copy=True)
    if payload["plan"]["decay_rate"]:
        refs[:] = support["mu"]
    return dict(plan=dict(payload["plan"]), slots=int(slots), f=support["f"], u=support["u"], refs=refs,
                band_ranges=support["band_ranges"])


def census_rows(points, windows, groups, seconds=None, requests=None):
    """One row per rule group: window, provider, points, z ranges (eV), nodes, sampled error, kappa."""
    EV = float(RYD_TO_EV)
    label = {}
    for w in windows:
        for m in w["members"]:
            label[int(m)] = f"{w['kind']}:{w['chunk']}"
    wall = {}
    if seconds and requests:
        for i, r in enumerate(requests):
            for m in r:
                wall[int(m)] = seconds.get(f"request_{i}")
    rows = []
    for g in groups:
        z = points[list(g["members"])]
        rows.append(dict(window=label.get(int(g["members"][0]), "?"), provider=g.get("provider", "pencil"),
                         points=len(g["members"]), re_ev_min=float(z.real.min() * EV),
                         re_ev_max=float(z.real.max() * EV), im_ev_min=float(z.imag.min() * EV),
                         im_ev_max=float(z.imag.max() * EV), nodes=int(g["count"]),
                         sampled_error=float(np.max(g["sampled_error"])),
                         kappa=float(np.max(g["coefficient_mass"])),
                         request_wall_s=wall.get(int(g["members"][0]))))
    return rows


# ------------------------------------------------------------------------------- files
def save_rules(path, payload):
    """Write the payload (pickle) and its census beside it (``.census.csv``); returns the file sha256."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=4)
    os.replace(tmp, path)
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(path + ".sha256", "w") as fh:
        fh.write(sha + "\n")
    rows = payload.get("census") or []
    if rows:
        with open(path + ".census.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return sha


def load_rules(path, *, allow_partial=False):
    """Read a complete rule file (sha256 sidecar checked when present).

    ``allow_partial`` admits a partial (``--windows`` census) build; only an instrument run of
    ``LORRAX_CD_STOP_AFTER`` passes it, and ``gw.cd.sigma.bank_chi0`` then refuses any point of an
    unbuilt window (GATE cd_chi0_rule_unbuilt).
    """
    with open(path, "rb") as fh:
        data = fh.read()
    side = path + ".sha256"
    if os.path.exists(side):
        want = open(side).read().split()[0]
        if hashlib.sha256(data).hexdigest() != want:
            raise ValueError(f"GATE cd_chi0_rules_file: {path} does not match its sha256 sidecar")
    payload = pickle.loads(data)
    if payload.get("identity", {}).get("version") != RULES_VERSION:
        raise ValueError(f"GATE cd_chi0_rules_file: {path} is not a {RULES_VERSION} rule file")
    if payload.get("partial") and not allow_partial:
        raise ValueError(f"GATE cd_chi0_rules_file: {path} is a partial (census) build; merge every window first")
    return payload


def merge_payloads(parts):
    """One complete payload from partial builds of the same identity (``--windows`` subsets)."""
    base = parts[0]["identity"]
    for p in parts[1:]:
        check_identity(p["identity"], base)
    groups, built, census, seconds = [], set(), [], {}
    for p in parts:
        groups += p["rule_groups"]
        built |= set(p["built"])
        census += p["census"]
        seconds.update({f"{k}@{len(seconds)}": v for k, v in p["seconds"].items()})
    covered = sorted(m for g in groups for m in g["members"])
    if covered != list(range(base["points"])) or built != set(range(len(base["windows"]))):
        raise ValueError("GATE cd_chi0_rules_cover: the merged parts do not cover every CD window exactly once")
    return dict(identity=base, plan=parts[0]["plan"], rule_groups=groups, seconds=seconds,
                built=sorted(built), census=census, requests=[], partial=False)


# ------------------------------------------------------------------------------- precompute CLI
def _energies(args):
    if args.energies:
        energy = np.load(args.energies)
    else:
        import h5py
        with h5py.File(args.wfn, "r") as fh:
            energy = np.asarray(fh["mf_header/kpoints/el"])[0]
    energy = np.asarray(energy, dtype=np.float64)
    if args.occupations:
        occ = np.asarray(np.load(args.occupations), dtype=np.float64)
    elif args.nocc:
        occ = np.zeros_like(energy)
        occ[:, :int(args.nocc)] = 1.0
    else:
        import h5py
        with h5py.File(args.wfn, "r") as fh:
            occ = np.asarray(fh["mf_header/kpoints/occ"], dtype=np.float64)[0]
    return energy, occ


def precompute(argv=None):
    """CLI: build (or merge) the CD chi0 rules of one deck geometry; see ``--help``."""
    ap = argparse.ArgumentParser(prog="python -m gw.cd.chi0_rules", description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--energies", help="energy table [nk, nb] (Ry), .npy: the WFN wedge or the full BZ")
    src.add_argument("--wfn", help="WFN .h5 (mf_header/kpoints/el and occ, spin 0)")
    ap.add_argument("--occupations", help="occupation table [nk, nb], .npy (default: the WFN's, or --nocc)")
    ap.add_argument("--nocc", type=int, default=0, help="step occupations: the lowest NOCC bands full")
    ap.add_argument("--nb-chi", type=int, required=False, help="screening bands (b_id_4_chi_user)")
    ap.add_argument("--b0", type=int, default=0)
    ap.add_argument("--b4-logical", type=int, default=0, help="logical band count (default: the table's)")
    ap.add_argument("--mu-ev", type=float, default=None, help="chemical potential (default: midgap)")
    ap.add_argument("--eta-ev", type=float, default=0.25, help="sigma_regularization_ev")
    ap.add_argument("--knot-step-ev", type=float, default=0.0625, help="cd_knot_step_ev")
    ap.add_argument("--imaginary-nodes", type=int, default=96, help="cd_imaginary_nodes")
    ap.add_argument("--knots", type=int, default=0, help="J, the live knot count of the run's geometry line")
    ap.add_argument("--reach-ev", type=float, default=0.0, help="cd_residue_reach_ev (J = ceil(reach/h) + 2)")
    ap.add_argument("--ndev", type=int, default=0, help="P, the run's device count")
    ap.add_argument("--nq-irr", type=int, default=0)
    ap.add_argument("--nk", type=int, default=0, help="full k count")
    ap.add_argument("--n-packed", type=int, default=0, help="packed centroid count")
    ap.add_argument("--knot-chunk", type=int, default=0, help="cd_knot_chunk (0 = the tile planner)")
    ap.add_argument("--tolerance", type=float, default=1e-8, help="cd_chi0_rule_tolerance")
    ap.add_argument("--capacity", type=int, default=CD_RULE_CAPACITY, help="cd_chi0_rule_capacity")
    ap.add_argument("--time-reversal", action="store_true", help="the deck's SymMaps.trs_allowed")
    ap.add_argument("--windows", default="all", help="all, or a comma list of imag:C / real:C / real:-1 (census)")
    ap.add_argument("--merge", nargs="+", help="merge partial rule files into --out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.merge:
        parts = [pickle.load(open(p, "rb")) for p in args.merge]
        payload = merge_payloads(parts)
        print(json.dumps(dict(out=args.out, sha256=save_rules(args.out, payload), groups=len(payload["rule_groups"]))))
        return 0
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", True)
    from types import SimpleNamespace
    from gw.cd.rule import tan_mapped_gauss_rule
    from gw.cd.sigma import GH, cd_chunk_plan
    EV = float(RYD_TO_EV)
    energy, occ = _energies(args)
    nb = energy.shape[1]
    b4 = int(args.b4_logical or nb)
    nb_chi = int(args.nb_chi or b4)
    occ_band = occ[:, :b4]
    if args.mu_ev is None:
        vbm = float(energy[:, :b4][occ_band > 0.5].max())
        cbm = float(energy[:, :b4][occ_band <= 0.5].min())
        mu = 0.5 * (vbm + cbm)
    else:
        mu = float(args.mu_ev) / EV
    h, eta = float(args.knot_step_ev) / EV, float(args.eta_ev) / EV
    if args.knots:
        J = int(args.knots)
    elif args.reach_ev > 0:
        J = int(np.ceil(float(args.reach_ev) / EV / h)) + 2
    else:
        ap.error("give --knots (the run's J) or --reach-ev")
    if not (args.ndev and args.nq_irr):
        ap.error("give --ndev and --nq-irr (the chunk is rounded to the (q, z) divisibility of P)")
    nq_pad = -(-int(args.nk or args.nq_irr) // int(args.ndev)) * int(args.ndev)
    chunk_i, chunk_r, n_imag_pad, J_pad = cd_chunk_plan(int(args.n_packed or 0), int(args.nq_irr), nq_pad,
                                                        int(args.ndev), int(args.imaginary_nodes), J,
                                                        int(args.knot_chunk))
    u_all = np.asarray(tan_mapped_gauss_rule(int(args.imaginary_nodes), eta * EV)[0]) / EV
    u = np.concatenate([u_all, np.full(n_imag_pad - u_all.size, u_all[-1])])
    imag, real = cd_windows(u, h, eta, J_pad, chunk_i, chunk_r, GH)
    points, windows = window_members(imag, real)
    pick = list(range(len(windows)))
    if args.windows != "all":
        keys = {f"{w['kind']}:{w['chunk']}": i for i, w in enumerate(windows)}
        last = max(w["chunk"] for w in windows if w["kind"] == "real")
        keys["real:-1"] = keys[f"real:{last}"]
        pick = sorted({keys[k.strip()] for k in args.windows.split(",")})
    wfns = SimpleNamespace(enk=jax.numpy.asarray(energy), occ=jax.numpy.asarray(occ),
                           slices=SimpleNamespace(b0=int(args.b0), b4_logical=b4))
    meta = SimpleNamespace(b_id_4_chi_user=nb_chi)
    slots = max(chunk_i, chunk_r + 2 * GH)
    sub = [windows[i] for i in pick]
    print(json.dumps(dict(J=J, J_pad=J_pad, chunk_imag=chunk_i, chunk_real=chunk_r, n_imag_pad=n_imag_pad,
                          points=int(points.size), windows=len(windows), building=[f"{w['kind']}:{w['chunk']}" for w in sub],
                          mu_ev=mu * EV, tolerance=args.tolerance, capacity=args.capacity)), flush=True)
    payload, _ = build_rules(wfns, meta, points, windows, mu_ry=mu, tolerance=float(args.tolerance), slots=slots,
                             capacity=int(args.capacity), time_reversal=bool(args.time_reversal), serial=True,
                             build=pick)
    sha = save_rules(args.out, payload)
    for row in payload["census"]:
        print(json.dumps(row), flush=True)
    print(json.dumps(dict(out=args.out, sha256=sha, partial=payload["partial"], groups=len(payload["rule_groups"]),
                          seconds=payload["seconds"])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(precompute())
