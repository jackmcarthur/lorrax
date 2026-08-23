"""``low_mem_bands = true`` — the named-refusal envelope.

WHAT IS BEING PINNED.  Guide: ``reports/gwjax_low_mem_bands_audit_2026-08-22/
report.md`` §6, "Unsupported combinations must refuse before allocation".
Five of the six unsupported combinations are deck keys and refuse AT PARSE
TIME through ``gw_config.refuse_unsupported_low_mem_bands``, called from
``LorraxConfig.from_input_file`` (and again, for a hand-built config, from
``gw.gw_init.prepare_isdf_and_wavefunctions`` at entry).  The sixth — an
explicit dense ``Gij`` operand — has no deck key (it is a keyword-only
Python parameter every shipped driver call site leaves at its ``None``
default), so it is guarded separately by
``gw_config.refuse_explicit_gij_under_low_mem_bands``, called from
``compute_sigma_xc`` at entry, the one seam that ever sees both operands
together.

THE ``low_mem_bands_dynamic_ppm_unported`` ROW (added 2026-08-22, after the
other five) is a CORRECTION, not part of the original guide text: the guide's
own §6 envelope listed ``compute_mode = gn_ppm`` as an example of a
*supported* combination, and the sibling wave that ported G/projection/
Hartree only ported the STATIC Σ channels (x_only, COHSEX's sigma_sx/
sigma_coh/hartree) — the dynamic two-point-PPM/MPA Σ_c(ω) pipeline
(``ppm_tau_kernel.py``, ``ppm_sigma.py``, ``mpa/sigma.py``) still reads the
legacy ``wfns.xn/xr/yr/yn`` accessors directly.  This was discovered on real
4-rank CUDA (`tests/multi_device/
low_mem_bands_one_shot_insulating_envelope_gate.py`, ``claims/0429.md``): a
``compute_mode = gn_ppm`` deck ran the ISDF fit and chi0/W screening to
completion under ``layout='face'`` and then died inside
``ppm_tau_kernel.precompile_sigma`` with the carrier's own named
``_require_legacy`` ``ValueError`` — a real crash, not a clean parse-time
refusal.  This row closes that gap.

Same shape as ``tests/test_screening_diagrams_config.py``'s w_bse refusal
matrix: a RED TWIN per rule (it actually fires, names its rule id, carries
all five message parts) plus a POSITIVE TWIN (the supported envelope does
NOT refuse), plus a ratchet on the table's own shape.  Pure config-parsing
and one small function call — ``gw.gw_config`` is deliberately jax-free
(see its own comment beside ``prepare_isdf_and_wavefunctions``'s import),
so this file needs no mesh, no device, and runs on a login node.
"""
from __future__ import annotations

import pathlib

import pytest

from gw.gw_config import (
    LorraxConfig,
    QPSolver,
    refuse_explicit_gij_under_low_mem_bands,
    refuse_unsupported_low_mem_bands,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]

_BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""

#: The companion keys ``mpa_material_class = metal`` requires at parse time
#: (``_validate_metal_compute_mode`` / ``_validate_occupation_smearing``) —
#: identical block to ``test_screening_diagrams_config.py``'s ``_METAL_KEYS``,
#: so a metal deck under test here satisfies the SAME prerequisite gate a
#: production metal deck would.
_METAL_KEYS = """\
mpa_material_class = metal
compute_mode = mpa
occ_smearing_family = mp1
occ_smearing_width_ry = 0.02
fermi_reference = mp1_fixed_n
sigma_omega_layout = sharded
"""


def _config(tmp_path, extra="", name="low_mem_bands.in"):
    path = tmp_path / name
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# 1. The refusal matrix — every row refuses at PARSE time, by rule id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule_id, extra", [
    ("low_mem_bands_self_consistent_unported",
     "head_correction = off\nqp_solver = self_consistent\n"),
    ("low_mem_bands_metal_material_class_unported",
     "head_correction = off\n" + _METAL_KEYS),
    ("low_mem_bands_bispinor_unported",
     "head_correction = off\nbispinor = true\n"),
    ("low_mem_bands_dynamic_ppm_unported",
     "head_correction = off\ncompute_mode = gn_ppm\n"),
    ("low_mem_bands_dynamic_ppm_unported",
     "head_correction = off\ncompute_mode = hl_ppm\n"),
    ("low_mem_bands_dynamic_ppm_unported",
     "head_correction = off\ncompute_mode = mpa\n"),   # insulating MPA
    ("low_mem_bands_dynamic_ppm_unported",
     "head_correction = off\nqp_solver = fixed_point\ncompute_mode = gn_ppm\n"),
])
def test_each_unsupported_combination_refuses_at_parse_time(
        tmp_path, rule_id, extra):
    """AT PARSE TIME, before the ISDF fit or any device allocation.

    The message carries got / want / fix / why / doc so an operator does
    not have to come to this file to know what to change.
    """
    with pytest.raises(ValueError) as exc:
        _config(tmp_path, "low_mem_bands = true\n" + extra)
    message = str(exc.value)
    assert rule_id in message
    for part in ("got:", "want:", "fix:", "why:", "doc:"):
        assert part in message, f"{rule_id} refusal is missing '{part}'"


def test_head_correction_full_is_lifted_on_the_bare_default():
    """head_correction=full (the shipping default) is PORTED for the face
    layout (feat/head-wings-face-port-2026-08-22): the wing kernels read
    wfns.psi_nmu/psi_mun, so the former
    low_mem_bands_head_correction_full_unported row is deleted per its own
    recorded lift condition.  A bare ``low_mem_bands = true`` deck with a
    STATIC Sigma channel must resolve with no head refusal; a dynamic
    compute_mode still refuses, but via the DYNAMIC row, not a head row."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(pathlib.Path(d), "low_mem_bands = true\ncompute_mode = cohsex\n")
        assert cfg.head.correction.value == "full"
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="low_mem_bands_dynamic_ppm_unported"):
            _config(pathlib.Path(d), "low_mem_bands = true\ncompute_mode = gn_ppm\n")


# ---------------------------------------------------------------------------
# 2. The positive twin — the supported envelope does NOT refuse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extra", [
    "head_correction = off\n",
    "head_correction = no_local_fields\n",
    "head_correction = off\nqp_solver = one_shot_dft\n",
    "head_correction = off\ncompute_mode = x_only\n",
    "head_correction = off\ncompute_mode = cohsex\n",
    "head_correction = off\nbispinor = false\n",
    "head_correction = off\nrestart = true\n",
])
def test_the_supported_envelope_parses_and_does_not_refuse(tmp_path, extra):
    """The other half of a refusal matrix: what it does NOT refuse.

    scalar/spinor, one-shot insulator (qp_solver=one_shot_dft only —
    fixed_point structurally requires a dynamic compute_mode, which the
    low_mem_bands_dynamic_ppm_unported row now refuses), head_correction=
    off|no_local_fields, standard chi0, the STATIC Sigma channels only
    (x_only, COHSEX), restart — none of these should trip a rule the
    moment low_mem_bands=true is added beside them.

    GN-PPM/HL-PPM/insulating MPA and qp_solver=fixed_point used to be
    listed here.  A real 4-rank CUDA gate found compute_mode=gn_ppm
    crashing mid-run (claims/0429.md); those cases moved to the red-twin
    matrix above under low_mem_bands_dynamic_ppm_unported.
    """
    config = _config(tmp_path, "low_mem_bands = true\n" + extra)
    assert config.memory.low_mem_bands is True
    refuse_unsupported_low_mem_bands(config)   # must not raise (again)


def test_low_mem_bands_false_decks_are_untouched_by_every_rule(tmp_path):
    """FROZEN-PIN CELL.  Not one row fires under the default.

    The refusal function returns before touching a single predicate on a
    ``low_mem_bands = false`` deck, so a default deck cannot acquire a NEW
    parse-time resolution -- and therefore a new possible error -- from
    this feature existing.  Every combination that DOES refuse above is
    tried here with the key left at its default (or explicitly false).
    """
    for extra in (
            "",
            "qp_solver = self_consistent\n",
            _METAL_KEYS,   # already names compute_mode = mpa
            "bispinor = true\n",
    ):
        config = _config(tmp_path, extra, name="lmb_false.in")
        assert config.memory.low_mem_bands is False
        refuse_unsupported_low_mem_bands(config)   # must not raise


def test_the_key_is_registered_so_it_is_not_an_unknown_key(tmp_path):
    """``strict_keys = true`` must ACCEPT it — the registry test."""
    config = _config(
        tmp_path, "strict_keys = true\nlow_mem_bands = true\n"
                  "head_correction = off\n")
    assert config.memory.low_mem_bands is True


# ---------------------------------------------------------------------------
# 3. The table ratchet — every row has all five parts and a unique id
# ---------------------------------------------------------------------------

def test_every_rule_has_all_five_parts_and_a_unique_id():
    """A rule added without a ``fix`` or a ``why`` stops a run without
    telling anyone what to do about it — the shape this table exists to
    make impossible."""
    from gw.gw_config import _LOW_MEM_BANDS_REFUSALS

    ids = [row[0] for row in _LOW_MEM_BANDS_REFUSALS]
    assert len(ids) == len(set(ids)), f"duplicate rule id in {ids}"
    assert len(ids) == 4, (
        "the table grew or shrank -- update this test AND the docs "
        "envelope table in docs/input_reference.md together")
    for row in _LOW_MEM_BANDS_REFUSALS:
        rule_id, predicate, got, want, fix, doc = row
        assert callable(predicate) and callable(got)
        for text, part in ((rule_id, "id"), (want, "want"), (fix, "fix"),
                           (doc, "why")):
            assert isinstance(text, str) and len(text) > 8, (
                f"{rule_id}: {part} is missing or too short to be advice")


# ---------------------------------------------------------------------------
# 4. The fifth row — explicit dense Gij (no deck key)
# ---------------------------------------------------------------------------

def test_an_explicit_gij_under_low_mem_bands_refuses(tmp_path):
    """RED TWIN.  A live Gij operand alongside low_mem_bands=true refuses,
    naming the rule id and all five message parts, exactly like the four
    deck-key rows above."""
    config = _config(tmp_path, "low_mem_bands = true\nhead_correction = off\n")
    with pytest.raises(ValueError) as exc:
        refuse_explicit_gij_under_low_mem_bands(config, "not-actually-an-array")
    message = str(exc.value)
    assert "low_mem_bands_explicit_gij_unported" in message
    for part in ("got:", "want:", "fix:", "why:", "doc:"):
        assert part in message, f"Gij refusal is missing '{part}'"


def test_gij_none_under_low_mem_bands_is_the_positive_twin(tmp_path):
    """POSITIVE TWIN.  ``Gij = None`` is every production call today —
    this must never raise, low_mem_bands or not."""
    on = _config(tmp_path, "low_mem_bands = true\nhead_correction = off\n")
    off = _config(tmp_path, "", name="lmb_off.in")
    refuse_explicit_gij_under_low_mem_bands(on, None)    # must not raise
    refuse_explicit_gij_under_low_mem_bands(off, None)   # must not raise
    refuse_explicit_gij_under_low_mem_bands(               # must not raise
        off, "an explicit Gij is FINE under low_mem_bands=false")


# ---------------------------------------------------------------------------
# 5. Driver-entry mirror — gw_init calls the same function on a hand-built
#    config, not a second ad hoc guard
# ---------------------------------------------------------------------------

def test_gw_init_calls_the_canonical_envelope_function_once():
    """``prepare_isdf_and_wavefunctions`` no longer carries its own
    bispinor-only guards (one per fresh/restart branch) -- it calls the
    ONE table-driven function, which covers all four deck-key rows, before
    either branch runs.  Guards against the exact duplication the sibling
    carrier branch flagged as a remaining risk in its own report."""
    import inspect
    from gw import gw_init

    src = inspect.getsource(gw_init.prepare_isdf_and_wavefunctions)
    assert src.count("refuse_unsupported_low_mem_bands(cfg)") == 1
    assert "does not support bispinor" not in src, (
        "an ad hoc bispinor-only guard survived beside the canonical "
        "table-driven refusal -- these should not both exist")
    # The call must precede BOTH branches it used to guard separately.
    order = [src.index(name) for name in (
        "refuse_unsupported_low_mem_bands(cfg)",
        "if not cfg.restart:")]
    assert order == sorted(order), (
        "the envelope refusal must run before the fresh/restart branch, "
        "not after either one has started allocating")


def test_compute_sigma_xc_checks_the_gij_row_before_any_kernel():
    """The Gij row runs at the top of the dispatch, beside the mode-
    unimplemented check it is modeled on -- before either the COHSEX/PPM
    kernel imports execute or any Gij-dependent allocation."""
    import inspect
    from gw import sigma_dispatch

    src = inspect.getsource(sigma_dispatch.compute_sigma_xc)
    order = [src.index(name) for name in (
        "refuse_unimplemented_compute_mode(",
        "refuse_explicit_gij_under_low_mem_bands(",
        "W_static = W_by_role.get(")]
    assert order == sorted(order), (
        "compute_sigma_xc no longer checks the Gij envelope row before "
        "the static-Sigma kernel dispatch")


# ---------------------------------------------------------------------------
# 6. Docs
# ---------------------------------------------------------------------------

def test_the_docs_envelope_table_names_every_rule_id():
    """docs/input_reference.md is hand-maintained; the table must name
    every rule id in the code, or the two will drift apart silently."""
    page = (_REPO / "docs" / "input_reference.md").read_text()
    assert "low_mem_bands = true` envelope" in page
    from gw.gw_config import _LOW_MEM_BANDS_REFUSALS

    for rule_id, *_ in _LOW_MEM_BANDS_REFUSALS:
        assert rule_id in page, f"{rule_id} is not named in the docs table"
    assert "low_mem_bands_explicit_gij_unported" in page


def test_qp_solver_enum_still_has_exactly_the_members_the_row_assumes():
    """Pin the axis this row reads, the same way the w_bse suite pins
    ``ScreeningDiagrams``.  A fourth ``QPSolver`` member would need its
    own supported/refused decision, not silent inheritance."""
    assert {m.value for m in QPSolver} == {
        "one_shot_dft", "fixed_point", "self_consistent"}
