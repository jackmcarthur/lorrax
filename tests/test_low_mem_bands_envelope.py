"""``low_mem_bands = true`` — supported routes and the live Gij refusal.

WHAT IS BEING PINNED.  Guide: ``reports/gwjax_low_mem_bands_audit_2026-08-22/
report.md`` §6, "Unsupported combinations must refuse before allocation".
All former deck-key refusal rows were lifted by 2026-08-23.  Their empty
table and no-op parser/driver calls were removed by the 2026-09-02 gate
survey.  The remaining explicit dense ``Gij`` operand has no deck key (it is
a keyword-only
Python parameter every shipped driver call site leaves at its ``None``
default), so it is guarded separately by
``gw_config.refuse_explicit_gij_under_low_mem_bands``, called from
``compute_sigma_xc`` at entry, the one seam that ever sees both operands
together.

THE ``low_mem_bands_dynamic_ppm_unported`` ROW (added 2026-08-22, after the
other five, NARROWED to MPA-only the same day) is a CORRECTION, not part of
the original guide text: the guide's own §6 envelope listed
``compute_mode = gn_ppm`` as an example of a *supported* combination, and
the sibling wave that ported G/projection/Hartree only ported the STATIC Σ
channels (x_only, COHSEX's sigma_sx/sigma_coh/hartree) — the dynamic
two-point-PPM/MPA Σ_c(ω) pipeline (``ppm_tau_kernel.py``, ``ppm_sigma.py``,
``mpa/sigma.py``) still read the legacy ``wfns.xn/xr/yr/yn`` accessors
directly.  This was discovered on real 4-rank CUDA (`tests/multi_device/
low_mem_bands_one_shot_insulating_envelope_gate.py`, ``claims/0429.md``): a
``compute_mode = gn_ppm`` deck ran the ISDF fit and chi0/W screening to
completion under ``layout='face'`` and then died inside
the former PPM tau precompile with the carrier's own named
``_require_legacy`` ``ValueError`` — a real crash, not a clean parse-time
refusal.

LIFTED for ``gn_ppm``/``hl_ppm`` 2026-08-22 (feat/dynamic-sigma-face-port-
2026-08-22): both now dispatch on ``wfns.layout`` and route through the
SAME canonical ``build_G_tau(layout='face', gemm=...)``/
``contract_bands_block_reshard(layout='face', channels=...)`` owners the
static COHSEX channels already used.  Gated on real 4-rank CUDA (algebra
parity, ``tests/test_ppm_tau_kernel_face_parity.py``) and a real
end-to-end MoS2 k6_c50 ``compute_mode = gn_ppm head_correction = full``
leg matching the legacy gn_ppm reference to ~1e-5 eV — see
``gw.gw_config``'s own row comment and the session's ``CLAIMS.md`` rows
for job ids.  A larger k6_c600 (mu=5282) confirmation of the same
combination hit a pre-existing, unrelated ``qsgw_head.py`` OOM
(``claims/0436.md``) before reaching this port's own code; production-
scale confirmation remains open follow-up work.  ``mpa/sigma.py``
(insulating MPA) was mechanically ported the same session but was NOT
end-to-end gated then, so the row STAYED for ``compute_mode = mpa``.

**The row is now DELETED (2026-08-23, feat/mpa-executor-face-gate-
2026-08-23, ``claims/0443.md``), not narrowed further**: ``mpa/sigma.
_integrate_sigma_batches``' own named gap (a split Σ window, ``nb_sigma
!= nb_full``) is fixed — ``strip_sigma_window`` gained a device-array arm
reusing ``wavefunction_bundle.pack_band_window``'s slice+reshard
mechanism on Σ_c's own trailing axes — and gated end to end (real 4-rank
CUDA mesh-aware parity, ``tests/test_mpa_sigma_split_window_strip.py``,
5/5 bit-exact; a real fresh one-shot insulating MPA leg with a genuine
split window, eqp0/eqp1 max|dE_QP| 6.5e-5/8.6e-5 eV legacy-vs-face).
Insulating ``compute_mode = mpa`` moves to the positive-twin matrix
below.  ``mpa_material_class = metal`` remains refused, but by
``low_mem_bands_metal_material_class_unported`` ALONE — a metal-only-
narrowed dynamic_ppm row was drafted and then deleted rather than kept,
because a metal deck's predicate is a strict subset of (and, given
``_validate_metal_compute_mode``'s standing invariant, logically
equivalent to) the metal row's own predicate, which appears earlier in
the table and always fires first: a second row with an unreachable
predicate is exactly the "gate that cannot fail" shape ``TASTE.md``
warns against, so it was not shipped.  See ``gw_config.py``'s own comment
at the deletion site, and the metal row's own updated comment for why
metal MPA itself stays refused (three named infra obstacles this
session, none of them in ``gw.mpa.sigma``'s own code).

``low_mem_bands_self_consistent_unported`` — LIFTED 2026-08-23 (feat/qsgw-
face-rotations-2026-08-23), same shape as the ``head_correction=full`` lift
above: ``wavefunction_bundle.rotate_wavefunctions`` now dispatches on
``wfns_dft.layout`` and routes ``layout='face'`` through
``rotate_wavefunctions`` — two planned ``distrib_la.gemm_plan`` N,N
GEMMs (``U^T @ psi_nmu``, ``psi_mun @ U``) against a block-embedded U
rather than a sliced ψ.  ``sc_iteration.py:1753`` needed no change.  Gated
on real 4-rank CUDA algebra parity (``tests/test_qsgw_rotate_face_parity.py``
— U from a REAL small eigh, ns=1/ns=2, default AND offset active windows,
3/3 PASS, max relative diff ~1e-16..2e-16) and a real end-to-end MoS2 k6_c50
``compute_mode=gn_ppm head_correction=full`` leg (3 SC iterations) vs the
``low_mem_bands=false`` reference — see ``gw_config._LOW_MEM_BANDS_
REFUSALS``' own history comment and this session's CLAIMS.md row for the
job id and measured tolerances.  The row is DELETED (not narrowed), same
as the ``head_correction=full`` row's own precedent: every SC combination
this project supports today is covered.

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
)

_REPO = pathlib.Path(__file__).resolve().parents[1]

_BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""

#: Metal runtime prerequisites; material class itself comes from WFN
#: occupations and the Sigma cube layout is no longer a deck dial.
_METAL_KEYS = """\
compute_mode = mpa
occ_smearing_family = mp1
occ_smearing_width_ry = 0.02
fermi_reference = mp1_fixed_n
"""


def _config(tmp_path, extra="", name="low_mem_bands.in"):
    path = tmp_path / name
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: None)


def test_full_q_storage_selection_refuses_at_parse_time(tmp_path):
    """Explicit full-q storage cannot override the WFN's parent representation."""
    with pytest.raises(ValueError, match="restart_q_storage='full'"):
        _config(tmp_path, "restart_q_storage = full\n")


def test_head_correction_full_is_lifted_on_the_bare_default():
    """head_correction=full (the shipping default) is PORTED for the face
    layout (feat/head-wings-face-port-2026-08-22): the wing kernels read
    wfns.psi_nmu/psi_mun, so the former
    low_mem_bands_head_correction_full_unported row is deleted per its own
    recorded lift condition.  A bare ``low_mem_bands = true`` deck with a
    STATIC Sigma channel must resolve with no head refusal.  GN-PPM and
    insulating MPA at the bare default (head_correction=full) are BOTH now
    fully-supported shipping-default decks under low_mem_bands
    (feat/dynamic-sigma-face-port-2026-08-22 and feat/mpa-executor-
    face-gate-2026-08-23 -- both gated end to end, see this file's module
    docstring); only mpa_material_class = metal still refuses, via the
    METAL row, not a head or dynamic-mode row (that row was deleted --
    see gw_config.py's own comment at the deletion site)."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(pathlib.Path(d), "low_mem_bands = true\ncompute_mode = cohsex\n")
        assert cfg.head.correction.value == "full"
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(pathlib.Path(d), "low_mem_bands = true\ncompute_mode = gn_ppm\n")
        assert cfg.head.correction.value == "full"
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(pathlib.Path(d), "low_mem_bands = true\ncompute_mode = mpa\n")
        assert cfg.head.correction.value == "full"
    with tempfile.TemporaryDirectory() as d:
        # LIFTED 2026-08-23: a metal MPA deck now parses under
        # low_mem_bands=true (the mpa_z status-line NameError that made
        # the SC driver unreachable is fixed on the same branch).
        cfg = _config(pathlib.Path(d), "low_mem_bands = true\n" + _METAL_KEYS)
        from gw.gw_config import validate_material_inputs
        validate_material_inputs(cfg, "metal")


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
    # bispinor, LIFTED 2026-08-23 (feat/transverse-zeta-face-2026-08-23,
    # row DELETED per the same "delete, don't narrow" precedent as
    # qp_solver=self_consistent's own lift): isdf.core.c_q_from_psi_sm/
    # z_q_from_psi_sm(layout='face') now accept non-identity gamma_L/
    # gamma_R via psi-endpoint application, gated on real 4-rank CUDA
    # with all 15 non-identity Lorentz-index pairs at ns=4 (tests/
    # test_isdf_cq_face_parity.py, tests/test_isdf_zq_face_parity.py),
    # plus a real end-to-end MoS2 3x3 bispinor GN-PPM leg vs the
    # low_mem_bands=false reference -- see gw_config._LOW_MEM_BANDS_
    # REFUSALS' own history comment.
    "head_correction = off\nbispinor = true\n",
    # GN-PPM/HL-PPM, LIFTED 2026-08-22 (feat/dynamic-sigma-face-port-
    # 2026-08-22) -- gated end to end on real 4-rank CUDA (algebra parity
    # + a real MoS2 k6_c50 leg vs the legacy gn_ppm reference; a larger
    # k6_c600 confirmation hit an unrelated pre-existing OOM), see this
    # file's module docstring.  head_correction is left at its bare
    # default (full) here deliberately: that is now the fully-supported
    # SHIPPING-DEFAULT deck under low_mem_bands, not merely a case that
    # happens to also work.
    "compute_mode = gn_ppm\n",
    "compute_mode = hl_ppm\n",
    "compute_mode = gn_ppm\nqp_solver = fixed_point\n",
    # qp_solver=self_consistent, LIFTED 2026-08-23 (feat/qsgw-face-
    # rotations-2026-08-23): rotate_wavefunctions now dispatches on
    # wfns_dft.layout (wavefunction_bundle.rotate_wavefunctions) --
    # gated on real 4-rank CUDA algebra parity (tests/
    # test_qsgw_rotate_face_parity.py) and a real end-to-end MoS2 k6_c50
    # compute_mode=gn_ppm head_correction=full qp_solver=self_consistent
    # (3 SC iterations) leg vs the low_mem_bands=false reference; see
    # gw_config._LOW_MEM_BANDS_REFUSALS' own history comment and this
    # session's CLAIMS.md row.  Same shape as the fixed_point row above,
    # covering the bare shipping default (head_correction left at 'full').
    "compute_mode = gn_ppm\nqp_solver = self_consistent\n",
    # Insulating compute_mode = mpa, LIFTED 2026-08-23 (feat/mpa-executor-
    # face-gate-2026-08-23, low_mem_bands_dynamic_ppm_unported DELETED):
    # gw.mpa.sigma._integrate_sigma_batches' named split-Sigma-window gap
    # (nb_sigma != nb_full, the ordinary case) is fixed --
    # strip_sigma_window's new mesh-aware device-array arm reuses
    # wavefunction_bundle.pack_band_window's own slice+reshard mechanism
    # on Sigma_c's own trailing axes.  Gated on real 4-rank CUDA (mesh-
    # aware strip_sigma_window parity, tests/
    # test_mpa_sigma_split_window_strip.py, 5/5 bit-exact) and a real
    # end-to-end fresh one-shot Si_scalar MPA leg with a genuine split
    # window (nb_sigma=8 < nb_full=20): eqp0.dat max|dE_QP|=6.510e-05 eV,
    # eqp1.dat max|dE_QP|=8.575e-05 eV, max|dE_DFT|=0.0 eV both, legacy
    # vs face (claims/0443.md).  head_correction left at its bare default
    # (full) here deliberately, same reasoning as the gn_ppm row above.
    "compute_mode = mpa\n",
])
def test_the_supported_envelope_parses_and_does_not_refuse(tmp_path, extra):
    """The other half of a refusal matrix: what it does NOT refuse.

    scalar/spinor, one-shot insulator (qp_solver=one_shot_dft) OR
    qp_solver=fixed_point with a dynamic compute_mode, head_correction=
    off|no_local_fields|full (the bare default), standard chi0, EVERY
    Sigma channel including insulating MPA (x_only, COHSEX, GN-PPM,
    HL-PPM, MPA), restart — none of these should trip a rule the moment
    low_mem_bands=true is added beside them.

    Metal MPA is LIFTED too (2026-08-23): the envelope is empty, so
    every combination the legacy path serves parses under
    low_mem_bands=true.
    """
    config = _config(tmp_path, "low_mem_bands = true\n" + extra)
    assert config.memory.low_mem_bands is True


def test_the_key_is_registered_so_it_is_not_an_unknown_key(tmp_path):
    """Always-strict parsing must accept the registered boolean dial."""
    config = _config(
        tmp_path, "low_mem_bands = true\n"
                  "head_correction = off\n")
    assert config.memory.low_mem_bands is True


# ---------------------------------------------------------------------------
# 3. The remaining row — explicit dense Gij (no deck key)
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
    with pytest.raises(ValueError, match="explicit_gij_unported"):
        refuse_explicit_gij_under_low_mem_bands(off, "explicit dense Gij")


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

def test_the_docs_name_the_live_gij_refusal():
    """The input reference must name the one live low-memory refusal."""
    page = (_REPO / "docs" / "input_reference.md").read_text()
    assert "Raw-parent GW" in page
    assert "low_mem_bands_explicit_gij_unported" in page


def test_qp_solver_enum_still_has_exactly_the_members_the_row_assumes():
    """Pin the axis this row reads, the same way the w_bse suite pins
    ``ScreeningDiagrams``.  A fourth ``QPSolver`` member would need its
    own supported/refused decision, not silent inheritance."""
    assert {m.value for m in QPSolver} == {
        "one_shot_dft", "fixed_point", "self_consistent"}
