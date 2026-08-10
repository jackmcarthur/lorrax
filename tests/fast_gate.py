"""The DEFAULT gate: the Si test calculation, for the drivers a branch touched.

THE FOUR-GPU RULE — EVERY GPU LEG RUNS AT P=4
---------------------------------------------
Whichever tier selects them, **every GPU verification leg runs at P=4**.  A
P=1-only verification is never sufficient for landing; unit and CPU cells are
exempt.  The owner's rationale, verbatim:

    "use four gpus for 100% of all testing so that never ever do we run
     something on one GPU and then learn it doesn't generalize later"

This module chooses WHICH cells run, never how many devices they run on --
see ``AGENT_PREAMBLE.md`` at the repository root for the rule itself.

WHY THIS FILE EXISTS (the owner, 2026-08-09)
--------------------------------------------
    "the test suite for lorrax became a super clodgy mess because of the
     llm test-for-everything habit; really we should run that Si test
     calculation (granted for all drivers that were touched since last ran)
     and the tests for the services and have that basically be it."

So there are exactly TWO tiers, and this module is the roster and the
decision procedure for the first one:

* **DEFAULT** — ``pytest`` with no arguments.  The Si end-to-end test
  calculation for each driver the branch TOUCHED, plus every service's own
  suite (``services/*/tests``), unchanged.  Minutes.

* **CENSUS** — ``pytest --census`` (equivalently ``pytest -m census``).
  Byte-for-byte the run that ``pytest`` used to be: the same collected set,
  the same ``addopts = "-m 'not extra'"``, the same meaning.
  ``tests/KNOWN_FAILURES.md`` accounts for THE CENSUS RUN and nothing about
  that accounting changes.

Nothing was deleted.  The per-fix unit-cell zoo — every red-twin and gate
cell written to pin one past bug — still exists, still runs, and still has
to be green; it now runs when you ask for the census rather than on every
invocation.

WHAT THIS MODULE IS NOT
-----------------------
It is not a second selection language.  It is a ROSTER (which cells are the
Si smoke, per driver), a MAP (which source paths can affect which driver),
and two pure functions over them.  The pytest wiring is twenty lines in
``tests/conftest.py``; the decisions are here so they are falsifiable
without a pytest session, the same convention as ``harness.pin_one_gpu``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Sentinel for "every driver" — a conservative map entry, or a diff we
#: could not read.  Never spell this as the full set: the point of the
#: sentinel is that it survives a driver being ADDED to the roster.
ALL = "ALL"


# ---------------------------------------------------------------------------
# THE DRIVERS.  docs/drivers.md is the source of truth for this list; it is
# restated here (rather than parsed) because a gate that reads its own roster
# out of prose fails in the direction that looks green.
# ---------------------------------------------------------------------------
DRIVERS = (
    "centroid.kmeans_cli",
    "psp.get_dipole_mtxels",
    "gw.kin_ion_io",
    "gw.gw_jax",
    "bandstructure.htransform",
    "bse.bse_jax",
    "bse.exciton_bands",
)


# ---------------------------------------------------------------------------
# THE SMOKE ROSTER — existing in-tree cells, WRAPPED, NEVER AUTHORED.
#
# Every node id below already existed and already ran the driver end to end
# against the deck's own pinned numbers at the deck's own tolerances.  This
# file selects them; it does not restate a single tolerance, and there is no
# new physics deck anywhere in this change.
# ---------------------------------------------------------------------------
SI_SMOKE: dict[str, tuple[str, ...]] = {
    # tests/regression/si_cohsex_debug — Si 4x4x4 COHSEX.  Three cells over
    # TWO runs of the same fixture dir (conftest's ``si_session`` and
    # ``si_fast_session``): the production deck is the suite's only place
    # where BerkeleyGW enters the loop, and the fast deck (20 bands, 144
    # centroids) is the cheap self-freeze that moves first when code drifts.
    "gw.gw_jax": (
        "tests/test_gw_jax_regression.py::test_si_fast_matches_frozen_reference",
        "tests/test_gw_jax_regression.py::test_si_production_matches_frozen_reference",
        "tests/test_gw_jax_regression.py::test_si_production_matches_berkeleygw",
    ),
    # tests/regression/si_bse_debug — Si BSE.  ONE cell that runs BOTH
    # stages (``gw.gw_jax`` then ``bse.bse_jax``) on the same run dir and
    # checks the lowest 20 excitons against the frozen LORRAX pin AND the
    # BerkeleyGW band.
    "bse.bse_jax": (
        "tests/test_bse_bgw_regression.py::test_bse_matches_frozen_and_bgw",
    ),
    # NOT SI, AND THAT IS THE POINT.  ``psp.get_dipole_mtxels`` needs the
    # deck's ``*.upf`` and Si does not ship them (tests/regression/
    # si_cohsex_debug carries no pseudopotentials — the 2026-08-09 defect
    # was exactly that a pseudo-less deck reached the sweep).  cohsex_debug
    # is the ONE regression deck that carries its own UPFs, so it is the
    # only end-to-end run of this driver a clean checkout can do.  Listed
    # here because the alternative is a touched driver with no smoke at all.
    "psp.get_dipole_mtxels": (
        "tests/test_dipole_regeneration_gate.py::"
        "test_the_default_analytic_sweep_writes_a_valid_dipole_h5",
    ),
}


# ---------------------------------------------------------------------------
# THE CONTRACT DECKS — the same smoke, expressed as a LAUNCH instead of a
# node id, because the JAX CACHE CONTRACT has to run the driver as four
# PROCESSES and a pytest node id runs it as one.
#
# Same rule as SI_SMOKE above: existing decks, existing arguments, nothing
# authored.  Each entry is the ordered list of STAGES that make one run of
# that driver's smoke deck; every stage is launched at P=4 and every stage
# is held to the contract, because a two-stage deck that is symmetric in
# its second stage and divergent in its first is exactly the state a
# single end-of-run check cannot see.
#
# ``tests/test_jax_cache_contract.py`` consumes this; a gate in that file
# asserts CONTRACT_DECKS and SI_SMOKE cover the same drivers, so the two
# rosters cannot drift apart silently.
# ---------------------------------------------------------------------------
CONTRACT_DECKS: dict[str, tuple[dict, ...]] = {
    # The FAST Si deck (20 bands, 144 centroids, ~12 s at P=1), not the
    # production one: the contract is about cache keys, not about physics
    # numbers, and the production deck's extra minutes buy this gate
    # nothing it can measure.
    "gw.gw_jax": (
        {"stage": "gw", "deck": "si_cohsex_debug", "module": "gw.gw_jax",
         "argv": ("-i", "cohsex_si_fast.in")},
    ),
    # TWO stages on ONE run dir, exactly as
    # ``test_bse_bgw_regression.test_bse_matches_frozen_and_bgw`` does it:
    # the BSE direct term reads W0_qmunu back from what the GW stage
    # persisted, so the stages are not independent and cannot be split.
    # ``--band-degeneracy off`` and the 4v4c window are that cell's pins,
    # restated here for the same reason it states them.
    "bse.bse_jax": (
        {"stage": "gw", "deck": "si_bse_debug", "module": "gw.gw_jax",
         "argv": ("-i", "bse_si_test.in")},
        {"stage": "bse", "deck": "si_bse_debug", "module": "bse.bse_jax",
         "argv": ("-i", "bse_si_test.in", "--bse", "--lanczos", "--tda",
                  "--matvec-kind=ring", "--n-val", "4", "--n-cond", "4",
                  "--n-occ", "4", "--band-degeneracy", "off",
                  "--n-reorth", "-1", "--max-lanczos-iter", "60",
                  "--n-eig", "8", "--px", "2", "--py", "2")},
    ),
    # NOT Si, for the reason SI_SMOKE gives above: this driver needs the
    # deck's ``*.upf`` and ``cohsex_debug`` is the only deck that ships
    # them.  It also needs the host FFI library, so the cell skips with
    # that reason rather than failing when the .so is absent.
    "psp.get_dipole_mtxels": (
        {"stage": "dipole", "deck": "cohsex_debug",
         "module": "psp.get_dipole_mtxels",
         "argv": ("-i", "cohsex_test.in", "--out", "dipole_regen.h5")},
    ),
}

#: Drivers with NO end-to-end deck a clean checkout can run, and why.
#: Reported by ``pytest_report_header`` whenever one of them is touched, so
#: a branch that changes an undecked driver is told that the default gate
#: says nothing about it rather than being handed a quiet green.
UNDECKED: dict[str, str] = {
    "centroid.kmeans_cli":
        "no deck: tests/test_kmeans_smoke.py runs the kernel on a synthetic "
        "hex cell, not the CLI on a fixture (kmeans is a fixture-generation "
        "tool; breakage surfaces loudly at regen)",
    "gw.kin_ion_io":
        "no deck: nothing in-tree runs this driver end to end.  Regenerating "
        "kin_ion.h5 needs the deck's *.upf and no Si fixture ships them",
    "bandstructure.htransform":
        "no deck: tests/test_htransform_kpath_gates.py calls h_transform as a "
        "library function on synthetic input; there is no ht.in in "
        "tests/regression",
    "bse.exciton_bands":
        "no in-tree deck: tests/test_exciton_bands.py is a real driver smoke "
        "but stages an out-of-repo Perlmutter fixture "
        "(/pscratch/.../05_lorrax_cohsex_native) and skips without it",
}


# ---------------------------------------------------------------------------
# THE FILE -> DRIVER MAP.  Deliberately COARSE and deliberately BIASED
# toward running more: a path that maps to too many drivers costs a Si run,
# and a path that maps to too few costs a missed regression.  Longest prefix
# wins; anything unlisted means ALL.
# ---------------------------------------------------------------------------
PATH_DRIVERS: tuple[tuple[str, object], ...] = (
    # --- driver sources: the narrow entries, and the only ones ---
    ("src/bse/", ("bse.bse_jax", "bse.exciton_bands")),
    ("src/gw/", ("gw.gw_jax", "gw.kin_ion_io")),
    ("src/bandstructure/", ("bandstructure.htransform",)),
    ("src/centroid/", ("centroid.kmeans_cli",)),
    # src/psp and src/common are ALL by the owner's own instruction, and the
    # instruction is right: psp holds the pseudopotential/velocity operators
    # every driver's matrix elements go through, and common holds the shared
    # contracts.  src/isdf, src/solvers, src/file_io, src/runtime, src/ffi,
    # src/mixing, src/postprocess are ALL for the same reason and are simply
    # left unlisted -> ALL.
    ("src/psp/", ALL),
    ("src/common/", ALL),

    # --- test-side inputs ---
    # A regression FIXTURE change (deck, reference, WFN) can move any gate
    # that reads it, and the gates are cheap relative to being wrong.
    ("tests/regression/", ALL),
    # Everything else under tests/ is the suite's own machinery.
    ("tests/", ALL),

    # --- things that cannot move a driver ---
    ("docs/", ()),
    ("manual/", ()),
    ("reports/", ()),
    ("misc/", ()),
    ("bench/", ()),
    (".github/", ()),
)

#: ``services/<name>/`` -> the drivers that consume that service.  The
#: service's OWN suite always runs (it is the other half of the default
#: tier), so these entries are about the DEPENDENTS: what else has to be
#: re-smoked when the service moves.  A service not listed here maps to ALL.
SERVICE_DEPENDENTS: dict[str, object] = {
    "distrib_la": ALL,          # the linear-algebra floor under everything
    "wfn_loader": ALL,          # every driver reads WFN.h5 through it
    "zeta_loader": ALL,         # zeta_q.h5 feeds GW and BSE alike
    "symmetry_maps": ALL,
    "vcoul": ALL,
    "minimax": ("gw.gw_jax",),  # imaginary-axis screening: the GW driver
    "lxkit": (),                # test/probe helpers; no driver consumes it
}


# ---------------------------------------------------------------------------
# Pure decisions
# ---------------------------------------------------------------------------
def drivers_for_path(rel: str) -> object:
    """The drivers a single repo-relative path can affect, or :data:`ALL`."""
    rel = rel.replace(os.sep, "/").lstrip("./")
    if rel.startswith("services/"):
        parts = rel.split("/")
        if len(parts) >= 2:
            return SERVICE_DEPENDENTS.get(parts[1], ALL)
        return ALL
    best: object = None
    best_len = -1
    for prefix, drivers in PATH_DRIVERS:
        if rel.startswith(prefix) and len(prefix) > best_len:
            best, best_len = drivers, len(prefix)
    # UNMAPPED -> ALL.  pyproject.toml, uv.lock, AGENTS.md, tools/, config/,
    # scripts/ and anything invented after today all land here, which is the
    # fail-safe direction.
    return ALL if best is None else best


def drivers_for_paths(paths) -> set:
    """Union over ``paths``; :data:`ALL` absorbs everything."""
    out: set = set()
    for rel in paths:
        d = drivers_for_path(rel)
        if d is ALL or d == ALL:
            return set(DRIVERS)
        out.update(d)
    return out


def _git(root: Path, *args: str):
    try:
        proc = subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def changed_paths(root: Path = None, ref: str = None):
    """Repo-relative paths that differ from ``ref``, or ``None`` if unknown.

    ``None`` is NOT "nothing changed" — it is "this is not a git checkout, or
    the ref does not exist here", and the caller must treat it as ALL.  The
    two are kept apart because they are opposite verdicts and conflating
    them is how a gate goes quietly empty.

    Compares against the MERGE-BASE with ``ref`` (default ``origin/main``,
    override with ``LX_GATE_REF``), so a branch is judged on what IT changed
    and not on what main gained underneath it.  Committed, staged, unstaged
    and untracked changes all count.
    """
    root = Path(root or REPO_ROOT)
    ref = ref or os.environ.get("LX_GATE_REF") or "origin/main"
    base = _git(root, "merge-base", ref, "HEAD")
    if base is None:
        base = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}")
        if base is None:
            return None
    base = base.strip()
    diff = _git(root, "diff", "--name-only", base, "--")
    if diff is None:
        return None
    untracked = _git(root, "ls-files", "--others", "--exclude-standard") or ""
    return sorted({ln.strip() for ln in (diff + "\n" + untracked).splitlines()
                   if ln.strip()})


def selected_drivers(root: Path = None, ref: str = None):
    """``(drivers, why)`` — the drivers whose Si smoke the default run does.

    ``LX_GATE_DRIVERS`` overrides the diff entirely: a comma-separated list
    of driver module names, or ``all`` / ``none``.
    """
    override = (os.environ.get("LX_GATE_DRIVERS") or "").strip()
    if override:
        low = override.lower()
        if low == "all":
            return set(DRIVERS), "LX_GATE_DRIVERS=all"
        if low in ("none", "-"):
            return set(), "LX_GATE_DRIVERS=none"
        names = {n.strip() for n in override.split(",") if n.strip()}
        unknown = names - set(DRIVERS)
        if unknown:
            raise ValueError(
                f"LX_GATE_DRIVERS names {sorted(unknown)}, which are not "
                f"drivers.  Known: {sorted(DRIVERS)}")
        return names, f"LX_GATE_DRIVERS={override}"

    paths = changed_paths(root, ref)
    if paths is None:
        return set(DRIVERS), "no readable diff (not a git checkout, or no ref)"
    if not paths:
        # A tree identical to its merge-base has nothing to judge, so the
        # gate does not narrow.  This also keeps a bare ``pytest`` on a
        # fresh clone meaningful instead of vacuous.
        return set(DRIVERS), "no changes vs merge-base"
    return drivers_for_paths(paths), f"{len(paths)} changed path(s) vs merge-base"


def smoke_node_ids(drivers) -> set:
    """The roster node ids for ``drivers`` (drivers with no deck contribute
    nothing — see :data:`UNDECKED`)."""
    out: set = set()
    for d in drivers:
        out.update(SI_SMOKE.get(d, ()))
    return out


def matches(nodeid: str, roster: set) -> bool:
    """Roster membership, tolerant of a cell later gaining parameters."""
    if nodeid in roster:
        return True
    head = nodeid.split("[", 1)[0]
    return head in roster


def is_service_item(nodeid: str) -> bool:
    """Service suites are the other half of the default tier, verbatim."""
    return nodeid.startswith("services/")
