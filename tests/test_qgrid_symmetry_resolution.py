"""The q-grid closure fallback is LOUD, and it is announced in one place.

WHAT WENT WRONG, AND WHERE.  ``centroid_source_map_and_wrap`` refuses on a
centroid set that is not orbit-closed — correctly.  But a refusal is only
as loud as the ``except`` that receives it, and this tree had three of them:

* ``gw/v_q_g_flat.py:_resolve_ibz_q_list`` caught it, set the tables to
  ``None``, and printed a line only if its ``verbose`` argument was true.
  ``gw/screening.py`` called it with ``verbose=False``, so the W Dyson
  solve silently widened from ``n_q_ibz`` blocks to ``n_q_full``.
* ``gw/v_q_bispinor.py`` called that helper twice and printed its own
  second wording, also behind ``verbose``.
* ``gw/isdf_fitting.py`` called the table builder directly, for its
  exception, to decide ``write_ibz_only``.

On the production 960-centroid Si deck the refusal is the NORMAL path (47
of 48 ops violating), so this was not a corner: it is what every run did,
and the cost — an ~8× wider q axis, restart tensors 8× larger than the
q_irr design intends, a 16.9 meV Σ star spread against 0.7 on a closed
set — appeared in no log line.

WHAT IS PINNED HERE.  Three cells, and they are the three ways this could
come back:

1. On a non-closed deck the announcement fires EXACTLY ONCE, however many
   times the run resolves the same centroid set (V_q, the W solve, every
   self-consistency iteration).  A per-call line is a line nobody reads.
2. On a closed deck it does NOT fire.  An announcement that is always
   present carries no information.
3. No module under ``src/`` reaches ``centroid_source_map_and_wrap`` behind
   the resolution point's back.  That is the ratchet: the consolidation is
   only worth anything for as long as it is the only door.

The deck header and centroid file are read ``'r'``; nothing here writes a
fixture.
"""

from __future__ import annotations

import ast
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "src")
_REG = os.path.join(_HERE, "regression")

#: ``(deck, wfn, centroid file)``.  MEASURED 2026-08-08: the 960-set is the
#: production Si set and is NOT closed (47/48 ops, worst 1.318e-01); the
#: 144-set is the one orbit-closed 48-op set in the tree (0/48, 1.0e-06).
#: Both live in the same deck, so the pair differs in exactly the thing
#: under test.
_DECK = ("si_cohsex_debug", "WFN.h5")
_OPEN_SET = "centroids_frac_960.txt"
_CLOSED_SET = "centroids_frac_144.txt"


def _q_receipt_sym(source="test:inversion"):
    return SimpleNamespace(
        q_symmetry_source=source,
        trs_allowed=False,
        sym_matrices=np.asarray((np.eye(3), -np.eye(3)), dtype=np.int32),
        translations=np.zeros((2, 3)),
        irr_idx_q=np.asarray((0, 1), dtype=np.int32),
        sym_idx_q=np.asarray((0, 1), dtype=np.int32),
        q_irr_kgrid_int=np.asarray(((0, 0, 0),), dtype=np.int32),
        q_irr_full_idx=np.asarray((0,), dtype=np.int32),
    )


def test_zeta_q_receipt_matches_current_effective_group_across_channels():
    from gw.qgrid_symmetry import require_zeta_q_symmetry
    from symmetry_maps import q_symmetry_receipt_json

    sym = _q_receipt_sym()
    receipt = q_symmetry_receipt_json(sym)
    loaders = tuple(SimpleNamespace(q_symmetry_receipt=receipt)
                    for _ in range(4))
    require_zeta_q_symmetry(sym=sym, loaders=loaders)

    mismatched = json.loads(receipt)
    mismatched["q_irr_full_idx"] = [1]
    loaders = loaders[:2] + (SimpleNamespace(q_symmetry_receipt=json.dumps(
        mismatched, sort_keys=True, separators=(",", ":"))),) + loaders[3:]
    with pytest.raises(ValueError, match="channel 2"):
        require_zeta_q_symmetry(sym=sym, loaders=loaders)


def test_zeta_q_receipt_preserves_historical_wfn_path():
    from gw.qgrid_symmetry import require_zeta_q_symmetry

    require_zeta_q_symmetry(
        sym=SimpleNamespace(q_symmetry_source="wfn"),
        loaders=(SimpleNamespace(q_symmetry_receipt=None),))
    with pytest.raises(ValueError, match="channel 0"):
        require_zeta_q_symmetry(
            sym=SimpleNamespace(q_symmetry_source="wfn"),
            loaders=(SimpleNamespace(q_symmetry_receipt="{}"),))


def _deck_file(*parts):
    return os.path.join(_REG, _DECK[0], *parts)


def _needs_deck(name):
    if not (os.path.isfile(_deck_file(_DECK[1]))
            and os.path.isfile(_deck_file(name))):
        pytest.skip(f"{_DECK[0]}/{name} not in this tree")
    pytest.importorskip("h5py")


def _sym_stub_and_centroids(name):
    """A ``SymMaps``-shaped stub plus the centroid index table.

    ``resolve_qgrid_symmetry_tables`` reads exactly two attributes off the
    symmetry object — ``sym_matrices`` (BGW ``mtrx``) and ``translations``
    (BGW ``tnp`` = 2π·τ) — so the stub carries those and nothing else.
    Building the real ``SymMaps`` would drag the k-grid reduction in and
    make this cell a test of the loader.
    """
    import types

    import h5py

    with h5py.File(_deck_file(_DECK[1]), "r") as f:
        g = f["mf_header"]["symmetry"]
        n = int(g["ntran"][()])
        S = np.asarray(g["mtrx"][:n])
        tnp = np.asarray(g["tnp"][:n])
        fft = np.asarray(f["mf_header"]["gspace"]["FFTgrid"][:],
                         dtype=np.int64)
    frac = np.loadtxt(_deck_file(name))
    idx = (np.rint(frac * fft[None, :]).astype(np.int64)
           % fft[None, :]).astype(np.int32)
    return types.SimpleNamespace(sym_matrices=S, translations=tnp), idx, fft


def _resolve_n_times(name, n, context="unit test"):
    """Resolve the same centroid set ``n`` times, as a real run does."""
    from gw.qgrid_symmetry import resolve_qgrid_symmetry_tables

    sym, idx, fft = _sym_stub_and_centroids(name)
    out = []
    for _ in range(n):
        out.append(resolve_qgrid_symmetry_tables(
            sym=sym, centroid_indices=idx, fft_grid=fft, context=context))
    return out


@pytest.fixture()
def clean_announcements():
    """Forget every memoized announcement, before AND after.

    ``ffi.gate`` dedupes per process for the whole run, which is the
    behaviour under test — so a cell that did not reset would measure
    whichever cell ran first, and one that did not reset afterwards would
    silence the next.
    """
    from ffi.gate import reset_gate_state

    reset_gate_state()
    yield
    reset_gate_state()


# ---------------------------------------------------------------------------
# 1 + 2: the announcement, and its red twin
# ---------------------------------------------------------------------------

def test_the_fallback_announcement_fires_exactly_once_per_run(
        capsys, clean_announcements):
    """Four resolutions of one non-closed set, ONE line.

    Four is not arbitrary: a scalar production run resolves the charge
    centroid set at the ζ̃ IBZ write, at the V_q pass, at the static-W
    Dyson solve, and again at each self-consistency iteration.  Before the
    consolidation those sites printed zero, one or two lines depending on
    a ``verbose`` argument; the number that is correct is one.
    """
    _needs_deck(_OPEN_SET)
    res = _resolve_n_times(_OPEN_SET, 4, context="V_q / W q-grid reduction")
    assert all(not r.use_ibz for r in res)
    out = capsys.readouterr().out
    n = out.count("LORRAX q-grid symmetry: FALLBACK")
    assert n == 1, (
        f"the fallback announced {n} times across 4 resolutions of the "
        f"same centroid set; exactly one is the contract.  Captured:\n{out}")
    # And the one line has to be worth printing.
    assert "V_q / W q-grid reduction" in out
    assert "q-grid symmetry reduction disabled" in out
    assert "solving on the full BZ" in out
    assert "restart tensors stay full-BZ" in out
    assert "verify_centroid_orbit_closure" in out
    worst = res[0].verdict.worst_residual
    assert f"{worst:.3e}" in out, out
    assert f"op {res[0].verdict.worst_op}" in out, out


def test_a_closed_deck_announces_nothing_at_all(capsys, clean_announcements):
    """RED TWIN.  Same code path, orbit-closed set, silence.

    Without this cell "announce the fallback" is satisfied by a print
    statement with no condition on it, which would be the same silence
    inverted: a line on every run tells an operator nothing about this
    run.
    """
    _needs_deck(_CLOSED_SET)
    res = _resolve_n_times(_CLOSED_SET, 4)
    assert all(r.use_ibz for r in res)
    out = capsys.readouterr().out
    assert "LORRAX q-grid symmetry" not in out, out
    assert out.strip() == "", f"the closed path printed:\n{out}"


def test_two_different_centroid_sets_are_two_facts(capsys,
                                                   clean_announcements):
    """Dedup is on the SET, not on the process.

    A bispinor deck carries a charge and a transverse centroid set whose
    closure can genuinely differ; collapsing them to one announcement
    would report one channel's health as both.  Here the two sets are the
    960 (open) and 144 (closed) tables of one deck, so exactly one line
    must appear — and adding a second OPEN set would add a second line.
    """
    _needs_deck(_OPEN_SET)
    _needs_deck(_CLOSED_SET)
    _resolve_n_times(_OPEN_SET, 2, context="charge centroids")
    _resolve_n_times(_CLOSED_SET, 2, context="transverse centroids")
    out = capsys.readouterr().out
    assert out.count("LORRAX q-grid symmetry: FALLBACK") == 1, out
    assert "charge centroids" in out
    assert "transverse centroids" not in out


def test_the_transverse_refusal_path_does_not_announce_a_fallback(
        capsys, clean_announcements):
    """``announce_fallback=False`` is the one suppression, and it is right.

    ``gw/isdf_fitting.py``'s transverse ζ̃_T write RAISES on a non-closed
    transverse set — the V_q orchestrator assumes an IBZ ζ̃_T and there is
    nothing to degrade to.  Printing "solving on the full BZ" beside that
    refusal would describe a run that is not happening.  The resolution
    still comes back ``full_bz``; only the announcement is withheld.
    """
    _needs_deck(_OPEN_SET)
    from gw.qgrid_symmetry import resolve_qgrid_symmetry_tables

    sym, idx, fft = _sym_stub_and_centroids(_OPEN_SET)
    res = resolve_qgrid_symmetry_tables(
        sym=sym, centroid_indices=idx, fft_grid=fft,
        context="bispinor transverse ζ̃_T IBZ write",
        announce_fallback=False)
    assert not res.use_ibz
    assert res.reason
    assert capsys.readouterr().out.strip() == ""


# ---------------------------------------------------------------------------
# 3: the ratchet
# ---------------------------------------------------------------------------

#: The only module under ``src/`` that may name
#: the table builder.
#: It is the wave-1 re-export shim (``symmetry_maps.orbit_syms`` moved into
#: the service); it forwards the name and never calls it.  The shim is
#: deleted by the phase-wide cleanup commit, at which point this set empties.
_PERM_NAME_ALLOWED = {os.path.join("centroid", "orbit_syms.py")}

#: BOTH SPELLINGS OF THE TABLE BUILDER, and this set is why the ratchet
#: survived the 2026-08-08 rename sweep.  The name is matched here as a
#: STRING — in the source-text prefilter and against the AST's identifier
#: set — so an AST-aware rename tool updates every call site in the tree
#: and leaves this gate matching a name nobody uses any more.  It would
#: then pass, forever, while a new ``src/`` module called the new name
#: straight past the resolution point.  The old spelling stays until the
#: compat aliases are retired (``symmetry_maps._compat``): as long as the
#: alias resolves, calling it is exactly the bypass this cell forbids.
_PERM_NAMES = {"centroid_source_map_and_wrap", "compute_centroid_sym_perm"}

#: The only module under ``src/`` allowed to reach the service's resolution
#: door.  Everything in ``gw/`` goes through this one adapter, which is what
#: makes "exactly one announcement" a property of the tree and not of a
#: convention.
_RESOLVE_ALLOWED = {os.path.join("gw", "qgrid_symmetry.py")}

#: Modules whose closure handling was consolidated, and which must therefore
#: contain no ``try`` around the resolution any more.  Naming them
#: explicitly (rather than scanning all of ``src/``) is what makes a NEW
#: file with a new ``except`` show up as an unlisted caller in the first
#: check above rather than slipping past this one.
_CONSOLIDATED = (
    os.path.join("gw", "v_q_g_flat.py"),
    os.path.join("gw", "isdf_fitting.py"),
    os.path.join("gw", "v_q_bispinor.py"),
    os.path.join("gw", "screening.py"),
)


def _src_modules():
    for root, _dirs, files in os.walk(_SRC):
        if "__pycache__" in root:
            continue
        for fn in sorted(files):
            if fn.endswith(".py"):
                path = os.path.join(root, fn)
                yield os.path.relpath(path, _SRC), path


def _names_used(tree):
    """Every identifier the module NAMES — imported, called or referenced.

    Attribute access counts (``orbit_syms.centroid_source_map_and_wrap``) and
    so does an aliased import (``... as _check_perm``, which is how one of
    the three call sites spelled it), because a ratchet that only matched
    the bare name would have missed the site it was written for.
    """
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                out.add(a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name.split(".")[-1])
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.Name):
            out.add(n.id)
    return out


def test_no_module_bypasses_the_resolution_point():
    """The ratchet.  One door to the table builder, one door to the service.

    Both halves are needed.  Pinning only the table builder
    would let a new site call ``resolve_qgrid_symmetry`` directly and skip
    the announcement; pinning only the service door would let one call the
    table builder and go back to catching exceptions.
    """
    offenders_perm, offenders_resolve = [], []
    for rel, path in _src_modules():
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if not any(n in src for n in _PERM_NAMES) \
                and "resolve_qgrid_symmetry" not in src:
            continue
        names = _names_used(ast.parse(src, filename=path))
        if (names & _PERM_NAMES) and rel not in _PERM_NAME_ALLOWED:
            offenders_perm.append(rel)
        if "resolve_qgrid_symmetry" in names and rel not in _RESOLVE_ALLOWED:
            offenders_resolve.append(rel)

    assert not offenders_perm, (
        f"{offenders_perm} reach the centroid table builder directly.  "
        f"The closure decision is taken once, in "
        f"``symmetry_maps.resolve_qgrid_symmetry``, and announced once by "
        f"``gw.qgrid_symmetry.resolve_qgrid_symmetry_tables``; a site that "
        f"builds the table itself is a site that will catch the refusal "
        f"itself and degrade the run in silence again.")
    assert not offenders_resolve, (
        f"{offenders_resolve} call the service's resolution door directly, "
        f"bypassing ``gw.qgrid_symmetry`` — which is where rank 0 and the "
        f"once-per-run memory live.  Route through the adapter.")


def test_the_consolidated_sites_no_longer_catch_the_refusal():
    """No ``try`` wraps the resolution at the four sites it was removed from.

    The consolidation's whole content is that the answer comes back as a
    value.  A ``try`` around it would mean somebody re-introduced a path
    that turns the answer back into an exception, and the ``except`` would
    be free to be silent again.
    """
    bad = []
    for rel in _CONSOLIDATED:
        path = os.path.join(_SRC, rel)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                f = sub.func
                nm = (f.id if isinstance(f, ast.Name)
                      else f.attr if isinstance(f, ast.Attribute) else "")
                if nm in ("resolve_qgrid_symmetry",
                          "resolve_qgrid_symmetry_tables",
                          "_resolve_ibz_q_list"):
                    bad.append(f"{rel}:{node.lineno} wraps {nm}")
    assert not bad, (
        "the q-grid resolution is caught again at: " + "; ".join(bad)
        + ".  It returns a resolution; there is nothing to catch, and an "
          "``except`` here is how the silent degradation comes back.")


def test_resolve_ibz_q_list_has_no_verbose_knob():
    """``verbose`` gated whether a degradation was visible.  It is gone.

    ``gw/screening.py`` passed ``verbose=False`` into this helper, which is
    precisely why the W Dyson solve could widen to the full BZ without a
    word.  A knob whose only effect is to hide a degradation is not a
    verbosity knob, and re-adding one here would restore the defect
    without restoring the ``except``.
    """
    path = os.path.join(_SRC, "gw", "v_q_g_flat.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and \
                node.name == "_resolve_ibz_q_list":
            args = ([a.arg for a in node.args.args]
                    + [a.arg for a in node.args.kwonlyargs])
            assert "verbose" not in args, (
                "_resolve_ibz_q_list took a ``verbose`` argument again")
            return
    raise AssertionError("gw.v_q_g_flat._resolve_ibz_q_list not found")
