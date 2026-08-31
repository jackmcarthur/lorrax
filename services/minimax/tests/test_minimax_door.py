"""The door: its surface, its announcements, and R1's staging.

Three subjects, and they are the three things the extraction actually
changed about behaviour rather than about location.

1. **Door reachability.**  Everything a consumer needs is a top-level
   name, and the solver half is on the door LAZILY.  A cell that only
   checked ``hasattr`` would be satisfied by an eager import, so the lazy
   half is checked for deferral as well as for presence.
2. **Provenance, announced once.**  Every table served says where it came
   from, once per distinct request — not once per call, because a
   quadrature request repeats per q-block per SCF iteration per rank and
   an announcement nobody can read is the same as no announcement.
3. **R1 stage 1.**  The escape hatch defaults OPEN, so no deck changes
   behaviour in a refactor commit; closing it makes the same request a
   refusal.  Both directions are cells, because "the default is on" and
   "the flag does something" are different claims and only the pair
   distinguishes a staged rollout from a decoration.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import minimax as M


# ---------------------------------------------------------------------------
#  1.  The door
# ---------------------------------------------------------------------------

def test_every_name_on_the_door_resolves():
    for name in M.__all__:
        assert hasattr(M, name), name


def test_the_solver_half_is_deferred_until_it_is_named():
    """The lazy door, from inside a process that has already imported the
    package.  ``minimax.solver`` must not be in ``sys.modules`` merely
    because ``minimax`` is — the in-process half of the claim the
    isolation suite measures in a child."""
    pytest.importorskip("scipy")
    import subprocess                              # noqa: PLC0415
    import sys                                     # noqa: PLC0415
    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src")
    probe = (
        "import sys; sys.path.insert(0, %r)\n"
        "import minimax\n"
        "assert 'minimax.solver' not in sys.modules, 'eager'\n"
        "minimax.G_hgl\n"
        "assert 'minimax.solver' in sys.modules, 'never loaded'\n"
        "print('OK')\n" % (src,))
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True)
    assert out.stdout.strip().endswith("OK"), (out.stdout, out.stderr)


def test_the_lazy_door_refuses_a_name_it_does_not_have():
    """``__getattr__`` must not become a hole that answers anything."""
    with pytest.raises(AttributeError):
        M.not_a_solver_name           # noqa: B018


def test_the_declared_families_match_the_shipped_bundle():
    """:data:`minimax.FAMILIES` claims which families ship tables.  That is
    a MEASURED fact about the bundle restated as data, and a restatement
    is exactly the kind of thing that rots — so it is checked against the
    bundle rather than trusted.

    This cell is also where R1's central finding is pinned: the
    imaginary-axis family really does ship nothing.
    """
    shipped_families = set(M.catalog().families())
    shipped_families |= set(
        M.catalog_view("catalog_complex_laplace.json").families())
    for name, spec in M.FAMILIES.items():
        assert spec.shipped == (name in shipped_families), (
            f"FAMILIES says {name} shipped={spec.shipped}, bundle says "
            f"{name in shipped_families}")
    assert M.FAMILIES["noncrossing_imag"].shipped is False
    assert M.FAMILIES["damped_line"].shipped is False


def test_the_catalog_view_is_enumerable_without_solving_anything():
    """31 entries, two families, no optimiser.

    The "no optimiser" half is NOT asserted here with
    ``'minimax.solver' not in sys.modules`` — in a shared session another
    cell may legitimately have imported it already, and a cell whose
    verdict depends on collection order is worse than no cell.  The
    deferral is measured in a fresh process by
    :func:`test_the_solver_half_is_deferred_until_it_is_named` and in a
    scrubbed one by the isolation suite.
    """
    view = M.catalog()
    assert len(view) == len(view.entries) == 31
    assert len(view.for_family("crossing")) == 5
    assert len(view.for_family("noncrossing")) == 26


# ---------------------------------------------------------------------------
#  2.  Provenance, and the announcement
# ---------------------------------------------------------------------------

def test_a_served_table_announces_its_origin_once():
    """R2's headline.  The first serve of a request announces; the second
    of the SAME request does not."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                 error_bound=1.0e-6, n_max=64)
        M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                 error_bound=1.0e-6, n_max=64)
    lines = [str(w.message) for w in caught
             if str(w.message).startswith("minimax: served")]
    assert len(lines) == 1, lines
    assert "shipped noncrossing/noncrossing_R_10p000000" in lines[0]
    assert "sha256:" in lines[0]
    assert "UNCERTIFIED" in lines[0]


def test_a_different_request_announces_separately():
    """RED TWIN for the once-only rule.  Announce-once must be keyed on the
    REQUEST; a global "announced already" flag would silence the second
    table entirely, which is the failure mode that makes a log look clean
    while two different artifacts are in play."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                 error_bound=1.0e-6, n_max=64)
        M.lookup(family="noncrossing", target="inverse", range_value=1000.0,
                 error_bound=1.0e-6, n_max=64)
    lines = [str(w.message) for w in caught
             if str(w.message).startswith("minimax: served")]
    assert len(lines) == 2, lines
    assert lines[0] != lines[1]


def test_the_announcement_reset_is_not_a_no_op():
    """RED TWIN for the conftest fixture.

    Every announcement cell in this suite depends on the autouse reset
    actually clearing state.  If it silently did nothing, the cells would
    still pass whenever they happened to run first — so the reset itself
    is measured.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                 error_bound=1.0e-6, n_max=64)
        M.reset_announcements()
        M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                 error_bound=1.0e-6, n_max=64)
    lines = [str(w.message) for w in caught
             if str(w.message).startswith("minimax: served")]
    assert len(lines) == 2, lines


def test_the_provenance_one_liner_carries_every_field_a_reader_needs():
    q = M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                 error_bound=1.0e-6, n_max=64)
    line = q.provenance.one_line()
    assert line.startswith("shipped noncrossing/")
    assert "sha256:" in line
    assert "gen unrecorded (catalog schema v1)" in line
    assert "backend unrecorded (catalog schema v1)" in line
    assert line.endswith("UNCERTIFIED")


def test_a_v2_entry_carries_a_real_generator_stamp():
    """The contrast that makes the v1 'unrecorded' meaningful.

    The complex_laplace bundle records its tool, its numpy and its scipy,
    so the same one-liner says something specific — which is what the v1
    rows will say once WP6's certification tier has been through them.
    """
    from minimax import _catalog as C                # noqa: PLC0415
    view = C.catalog_view("catalog_complex_laplace.json")
    entry = view.entries[0]
    _t, _a, _e, _k, h = C.load_table(entry)
    prov = C.provenance_for(
        entry, h, C.load_catalog_dict("catalog_complex_laplace.json"))
    assert prov.certified is True
    assert "generate_imag_minimax_assets.py@" in prov.generator_commit
    assert "numpy-" in prov.generation_backend
    assert "scipy-" in prov.generation_backend
    assert prov.one_line().endswith("CERTIFIED")


def test_certified_and_shipped_are_different_claims():
    """The distinction the whole provenance record exists to keep.

    ``source`` says which ARTIFACT answered; ``certified`` says whether
    that artifact carries a measured certification record.  Every v1 table
    is a real shipped artifact whose claim about itself has never been
    checked, and printing that on every serve is what makes WP6's absence
    visible in a log instead of only in a design document.
    """
    q = M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                 error_bound=1.0e-6, n_max=64)
    assert q.provenance.source == "shipped"
    assert q.provenance.certified is False
    assert "shipped" in q.provenance.one_line()
    assert "UNCERTIFIED" in q.provenance.one_line()


def test_mixed_v1_certified_entry_is_complete_and_error_bound_to_payload(
        tmp_path, monkeypatch):
    """A certified append cannot borrow legacy absence or a catalog claim."""
    from minimax import _catalog as C                # noqa: PLC0415

    tau = np.array([0.5], dtype=np.float64)
    alpha = np.array([1.0], dtype=np.float64)
    provenance = {
        "tool": "tools/generate_minimax_assets.py",
        "tool_sha256": "1" * 64,
        "generator_commit": "2" * 40,
        "backend_sha256": "3" * 64,
    }
    entry = {
        "family": "noncrossing", "range_max": 2.0,
        "error_bound": 1.0e-6, "node_count": 1,
        "file": "noncrossing/certified.npz", "max_error": 1.0e-8,
        "kappa0": 1.0, "payload_sha256": M.payload_sha256(tau, alpha),
        "certification": {"checks": ["refined_error"]},
        "provenance": provenance, "certified": True,
    }
    for missing in ("payload_sha256", "kappa0", "certification",
                    "provenance"):
        incomplete = dict(entry)
        incomplete.pop(missing)
        with pytest.raises(M.CatalogCorrupt, match="incomplete"):
            M.parse_catalog(
                {"schema_version": 1, "tables": [incomplete]},
                catalog_name="incomplete.json")

    for invalid_error in (-1.0, np.inf, 2.0e-6):
        invalid = dict(entry, max_error=invalid_error)
        with pytest.raises(M.CatalogCorrupt, match="within"):
            M.parse_catalog(
                {"schema_version": 1, "tables": [invalid]},
                catalog_name="invalid-error.json")

    root = tmp_path / "minimax_assets"
    path = root / entry["file"]
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path, tau=tau, alpha=alpha,
        max_error=np.asarray(2.0e-8, dtype=np.float64))
    monkeypatch.setattr(C, "_asset_root", lambda: root)
    C.clear_caches()
    parsed = M.parse_catalog(
        {"schema_version": 1, "tables": [entry]},
        catalog_name="mismatched.json")[0]
    with pytest.raises(M.TableUnreadable, match="differs bit-exactly"):
        C.load_table(parsed)
    np.savez_compressed(
        path, tau=tau, alpha=alpha,
        max_error=np.asarray(np.inf, dtype=np.float64))
    C.clear_caches()
    with pytest.raises(M.TableUnreadable, match="payload max_error=.*within"):
        C.load_table(parsed)


@pytest.mark.parametrize("tau,alpha,node_count,match", [
    (np.array([[0.5]]), np.array([1.0]), 1, "equal-length 1-D"),
    (np.array([0.5]), np.array([1.0]), 2, "matching node_count"),
    (np.array([np.nan]), np.array([1.0]), 1, "non-finite"),
    (np.array([0.5]), np.array([-1.0]), 1, "positive real"),
])
def test_certified_noncrossing_payload_shape_and_values_refuse(
        tmp_path, monkeypatch, tau, alpha, node_count, match):
    """Selector metadata cannot outrun the certified numerical payload."""
    from minimax import _catalog as C                # noqa: PLC0415

    root = tmp_path / "minimax_assets"
    rel = "noncrossing/certified.npz"
    path = root / rel
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path, tau=tau, alpha=alpha,
        max_error=np.asarray(1.0e-8, dtype=np.float64))
    raw = {
        "family": "noncrossing", "range_max": 2.0,
        "error_bound": 1.0e-6, "node_count": node_count, "file": rel,
        "max_error": 1.0e-8, "kappa0": 1.0,
        "payload_sha256": M.payload_sha256(tau, alpha),
        "certification": {"checks": ["refined_error"]},
        "provenance": {
            "tool": "tools/generate_minimax_assets.py",
            "tool_sha256": "1" * 64, "generator_commit": "2" * 40,
            "backend_sha256": "3" * 64,
        },
        "certified": True,
    }
    entry = M.parse_catalog(
        {"schema_version": 1, "tables": [raw]},
        catalog_name="bad-payload.json")[0]
    monkeypatch.setattr(C, "_asset_root", lambda: root)
    C.clear_caches()
    with pytest.raises(M.TableUnreadable, match=match):
        C.load_table(entry)


def test_final_run33_two_pane_requests_are_publicly_certified():
    """The final equal-range plan is closed; obsolete percentile panes are not."""
    low = M.lookup(
        family="noncrossing", target="inverse",
        range_value=212.23793639387773,
        error_bound=3.4533298639725701e-8, n_max=64)
    assert low.provenance.catalog_entry.endswith(
        "noncrossing_R_256p000000_eps_3p0em08.npz")
    assert low.provenance.certified is True
    assert low.max_error == 1.8089704512114224e-8
    assert low.kappa0 == 1.0000026652201994
    assert "tools/generate_minimax_assets.py@c55621b2" in (
        low.provenance.generator_commit)
    assert "id-" in low.provenance.generation_backend
    assert low.provenance.one_line().endswith("CERTIFIED")

    high = M.lookup(
        family="noncrossing", target="inverse",
        range_value=212.32817285287737,
        error_bound=4.9322777100153476e-7, n_max=64)
    assert high.provenance.catalog_entry.endswith(
        "noncrossing_R_215p443469_eps_2p0em07.npz")
    assert high.provenance.certified is True
    assert high.max_error == 5.882476630022365e-8
    assert high.kappa0 == 1.0000064579102943
    assert high.provenance.one_line().endswith("CERTIFIED")


# ---------------------------------------------------------------------------
#  3.  R1 staging — the escape hatch
# ---------------------------------------------------------------------------

def test_the_escape_hatch_defaults_open(monkeypatch):
    """STAGE 1's whole content: no deck changes behaviour in a refactor.

    The refusal machinery ships, the announcements ship, and the default
    is such that a run which worked yesterday works today — because the
    imaginary-axis family has no shipped tables and arming the refusal
    before generating it would have made the extraction a
    physics-stopping commit.
    """
    monkeypatch.delenv(M.RUNTIME_SOLVE_ENV, raising=False)
    assert M.runtime_solve_allowed() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_the_escape_hatch_closes_on_every_spelling(monkeypatch, value):
    monkeypatch.setenv(M.RUNTIME_SOLVE_ENV, value)
    assert M.runtime_solve_allowed() is False


def test_f5_a_miss_with_the_hatch_closed_is_a_refusal(monkeypatch,
                                                      isolated_cache):
    """STAGE 2, exercised without arming it.

    The flip is one default value, so the behaviour it produces can be
    tested today by setting the flag — which is what makes "the staging is
    a default, not an architecture" a checkable statement rather than a
    reassuring one.  The refusal carries the ORIGINAL miss inside it, so
    the reader learns which table was wanted and not merely that something
    was refused.
    """
    monkeypatch.setenv(M.RUNTIME_SOLVE_ENV, "0")
    with pytest.raises(M.UncertifiedSolveRefused) as excinfo:
        M.serve(family="crossing", target="hgl", range_value=83.0,
                error_bound=1.0e-6, n_max=500, eps_q=1.0e-3)
    text = str(excinfo.value)
    assert M.RUNTIME_SOLVE_ENV in text
    assert "no certified crossing table for A_dim=83" in text
    assert "nearest certified below: A_dim=60" in text


def test_f5_an_exhausted_solver_ladder_is_a_refusal(monkeypatch):
    """The last rule is not a solution when its measured error misses."""
    from minimax import door

    provenance = M.runtime_provenance("fake", "test")
    monkeypatch.setattr(
        door, "_solve_noncrossing_scaled_cached",
        lambda *_args: (np.array([1.0]), np.array([1.0]), 2.0e-4,
                        provenance))

    with pytest.raises(M.UncertifiedSolveRefused, match="exhausted"):
        M.solve_uncertified(
            family="noncrossing", target="inverse", range_value=10.0,
            error_bound=1.0e-6, n_max=2)


def test_f5_false_case_with_the_hatch_open_it_solves_and_says_so(
        monkeypatch, isolated_cache):
    """The FALSE case for F5, and the loudest line in the service.

    A solve must name the request, the achieved error, the measured Σ|w|
    and κ₀, and the words *uncertified, not reproducible across hosts* —
    because the comfortable failure mode of this whole design is that the
    hatch stays open forever and nobody notices.  The defence is that
    every log says so.

    A_dim = 20 rather than 83: the point of this cell is the ANNOUNCEMENT,
    and a small bandwidth solves in a second where the G2 gate's 83 takes
    the better part of a minute.
    """
    pytest.importorskip("scipy")
    monkeypatch.setenv(M.RUNTIME_SOLVE_ENV, "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q = M.serve(family="crossing", target="hgl", range_value=20.0,
                    error_bound=1.0e-6, n_max=60, eps_q=1.0e-3,
                    use_shipped=False)
    assert q.provenance.source == "runtime-uncertified"
    assert q.provenance.certified is False
    assert q.kappa0 is not None
    lines = [str(w.message) for w in caught
             if "UNCERTIFIED SOLVE" in str(w.message)]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "crossing/hgl A_dim=20" in line
    assert "sum|w|" in line and "kappa0" in line
    assert "NOT REPRODUCIBLE ACROSS HOSTS" in line
    assert M.RUNTIME_SOLVE_ENV in line


def test_serve_prefers_the_certified_table_over_the_hatch(isolated_cache):
    """The ordering that makes stage 1 worth having at all: the hatch is a
    FALLBACK, not a parallel path.  A request the catalog covers must
    never reach the solver."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q = M.serve(family="crossing", target="hgl", range_value=40.0,
                    error_bound=1.0e-6, n_max=500, eps_q=1.0e-3)
    assert q.provenance.source == "shipped"
    assert not [w for w in caught if "UNCERTIFIED SOLVE" in str(w.message)]


def test_use_shipped_false_is_an_explicit_request_for_the_hatch(
        monkeypatch, isolated_cache):
    """``regenerate_minimax_tables`` arriving at the door.

    It is an EXPLICIT request for the uncertified path, so it bypasses the
    catalog — and it still announces, because "the user asked for it" is a
    reason not to refuse and not a reason to go quiet.
    """
    pytest.importorskip("scipy")
    monkeypatch.setenv(M.RUNTIME_SOLVE_ENV, "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q = M.serve(family="noncrossing", target="inverse", range_value=10.0,
                    error_bound=1.0e-6, n_max=64, use_shipped=False)
    assert q.provenance.source in ("runtime-uncertified", "cache")
    assert [w for w in caught if "UNCERTIFIED SOLVE" in str(w.message)]


def test_a_family_with_no_in_process_solver_refuses_rather_than_hanging(
        monkeypatch):
    """``complex_laplace`` has no in-process solver, so the hatch cannot
    rescue a request no shipped table covers.  The refusal names the
    generator instead of the flag, because the flag is not the fix here.

    The request is fully specified and OFF the tabulated β ladder: since
    the beta axis landed, an under-specified request refuses earlier and
    for a different reason (``UnknownTarget``, not the hatch), so it would
    no longer exercise this path at all.
    """
    monkeypatch.setenv(M.RUNTIME_SOLVE_ENV, "1")
    with pytest.raises(M.UncertifiedSolveRefused) as excinfo:
        M.serve(family="complex_laplace", target="complex_laplace",
                range_value=21.544346900318832, error_bound=1.0e-6, n_max=64,
                beta=7.5, beta_clause=M.beta_selector.HEIGHT)
    assert "generate_imag_minimax_assets.py" in str(excinfo.value)
