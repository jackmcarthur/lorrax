"""``bispinor_tt_head_correction`` — the bare TT (transverse-transverse)
q=Γ, G=0 mini-BZ head correction.

CPU-only (no GPU/SlabIO required): everything exercised here is host-side
numpy/jax algebra (``vcoul``'s mini-BZ sampler, ``gw.v_q_bispinor``'s
per-q ``v(q+G)`` builder, and ``gw.gw_config``'s parse-time refusal) — no
HDF5/FFI write path, mirroring ``tests/test_sigma_x_bispinor.py``'s own
"CPU-only" scope note.

WHAT IS BEING PINNED.  Design note:
``docs/bispinor_tt_head_correction_2026-08-23.md``.  Physics source:
``docs/BISPINOR_DHFB_DESIGN.md`` §11 (the bi4/MoS2 4×4 measurement) and
``KNOWN_LORRAX_ISSUES.md``'s bispinor row (claim 41, job 7885325).

Four things are checked, each an observable that FAILS if the physics or
the wiring is wrong (TASTE.md pattern 1):

1. Default OFF is byte-identical to the pre-existing q=Γ, G=0 zero slot
   (``test_tt_head_correction_off_leaves_gamma_slot_zero``).
2. The injected value at ON matches the SAME tensor
   ``_tt_head_tensor`` computes and lands ONLY at the (q=Γ, G=0) slot —
   every other (q, G) entry is untouched (``test_tt_head_correction_on_
   injects_only_at_gamma_g0``).
3. An algebraic invariant the estimator cannot satisfy by accident:
   ``tr(T) == 2 * vc0`` exactly, because ``tr(t_ij(q̂)) = 2`` at every
   point on the unit sphere regardless of direction — this is a RED TWIN
   for a wrong index, a wrong sign, or a wrong projector formula
   (``test_tt_head_tensor_trace_identity_matches_2_vc0``).
4. The measured slab reference ratio ``diag(1/2, 1/2, 1)`` for an
   in-plane-isotropic mini-BZ, reproducing ``docs/BISPINOR_DHFB_DESIGN.
   md``'s bi4 measurement (0.4993, 0.5007, 1.0000) independently, on a
   different synthetic cell (``test_tt_head_tensor_matches_measured_
   slab_reference_ratio``).

Plus the parse-time refusal envelope (rule id, all five message parts,
no-op at the default) mirroring ``tests/test_low_mem_bands_envelope.py``'s
own shape.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pathlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1-4: builder / tensor correctness (no SlabIO, no HDF5 write)
# ---------------------------------------------------------------------------

# Synthetic slab geometry: in-plane-isotropic (square) lattice, so the
# theoretical prediction <P_T> = diag(1/2, 1/2, 1) applies exactly (up to
# Monte-Carlo noise) -- see reports/bispinor_screened_wings_q0_audit_2026-
# 08-22/report.md's executive answer.
_A = 12.0                                   # bohr, real-space in-plane a
_C = 30.0                                   # bohr, real-space out-of-plane c
_BVEC = np.diag([2.0 * np.pi / _A, 2.0 * np.pi / _A, 2.0 * np.pi / _C])
_CELL_VOLUME = _A * _A * _C
_KGRID = (4, 4, 1)
_SYS_DIM = 2


def _bare_vc0() -> complex:
    """``⟨v(q)⟩_mBZ`` at q=Γ, BARE units, on the SAME draw
    ``_tt_head_tensor``/``q0_average_transverse_tensor`` uses (``nmax=1``,
    ``is_2d=True``, ``kind='slab'``) -- independent of
    ``vcoul.Slab2D.q0_average``'s own screened-head branches (S_cart /
    epshead), which this module has no reason to exercise."""
    from ffi import _services
    _services.ensure_on_path()
    from vcoul.minibz import (minibz_average, minibz_inscribed_sphere_r2,
                               sample_minibz_qpoints)
    from vcoul.geometry import CoulombGeometry

    geometry = CoulombGeometry(bvec=_BVEC, cell_volume=_CELL_VOLUME)
    batches = sample_minibz_qpoints(
        geometry, _KGRID, nsamples=2**18, method="sobol", qmc_reps=10,
        nmax=1, is_2d=True)
    q0sph2 = minibz_inscribed_sphere_r2(_BVEC, _KGRID, is_2d=True)
    zc = float(np.pi / _BVEC[2, 2])
    return minibz_average(
        np.zeros(3), [np.asarray(b) for b in batches], kind="slab",
        celvol=float(_CELL_VOLUME), n_kpts=int(np.prod(_KGRID)),
        q0sph2=q0sph2, zc=zc, analytic_sphere=False, adaptive=True)


def _one_q_gamma_g0_table():
    """One q (Γ) with 3 G's, the middle one being G=0 -- the minimal
    fixture that exercises the K=0 detection without needing the full
    ``compute_per_q_bare_coulomb_components`` sphere machinery.

    ``gvec_components`` is ``(n_q, 3, ngkmax)`` -- component-major, per
    ``isdf_header/gvec_components`` -- so a ``(ngkmax, 3)`` list of
    Miller-index ROWS is built first and transposed once.
    """
    q_irr_frac = np.zeros((1, 3), dtype=np.float64)          # q = Γ
    g_rows = np.array([[1, 0, 0], [0, 0, 0], [-1, 1, 0]],
                       dtype=np.float64)                      # (3 G's, xyz)
    gvec_components = g_rows.T[None, :, :]                    # (1, 3, 3)
    assert np.all(gvec_components[0, :, 1] == 0.0)             # g_idx=1 is G=0
    return q_irr_frac, gvec_components


def test_tt_head_correction_off_leaves_gamma_slot_zero():
    """Default (tt_head_correction=False): the q=Γ, G=0 slot of a TT tile
    is exactly 0, same as before this feature existed."""
    from gw.v_q_bispinor import _make_per_q_v_builder_for_tile

    q_irr_frac, gvec_components = _one_q_gamma_g0_table()
    builder = _make_per_q_v_builder_for_tile(
        mu_L=1, nu_L=1, bvec=_BVEC, cell_volume=_CELL_VOLUME,
        sys_dim=_SYS_DIM, vcoul_cutoff_ry=None)
    out = np.asarray(builder(q_irr_frac, gvec_components))
    assert out.shape == (1, 3)
    np.testing.assert_array_equal(out[0, 1], 0.0 + 0.0j)      # G=0 slot


def test_tt_head_correction_on_injects_only_at_gamma_g0():
    """tt_head_correction=True: the q=Γ, G=0 slot equals
    ``-_tt_head_tensor(...)[i, j] / cell_volume`` and nothing else moves."""
    from gw.v_q_bispinor import (
        _make_per_q_v_builder_for_tile, _tt_head_tensor)

    q_irr_frac, gvec_components = _one_q_gamma_g0_table()
    T = _tt_head_tensor(
        bvec=_BVEC, cell_volume=_CELL_VOLUME, sys_dim=_SYS_DIM,
        kgrid=_KGRID)
    assert T.shape == (3, 3)
    assert np.all(np.isfinite(T))

    for (mu_L, nu_L, i, j) in [(1, 1, 0, 0), (2, 2, 1, 1), (1, 2, 0, 1)]:
        b_off = _make_per_q_v_builder_for_tile(
            mu_L=mu_L, nu_L=nu_L, bvec=_BVEC, cell_volume=_CELL_VOLUME,
            sys_dim=_SYS_DIM, vcoul_cutoff_ry=None)
        b_on = _make_per_q_v_builder_for_tile(
            mu_L=mu_L, nu_L=nu_L, bvec=_BVEC, cell_volume=_CELL_VOLUME,
            sys_dim=_SYS_DIM, vcoul_cutoff_ry=None,
            kgrid=_KGRID, tt_head_correction=True)
        out_off = np.asarray(b_off(q_irr_frac, gvec_components))
        out_on = np.asarray(b_on(q_irr_frac, gvec_components))
        expected = -complex(T[i, j] / _CELL_VOLUME)
        np.testing.assert_allclose(out_on[0, 1], expected, rtol=1e-12)
        # Every OTHER (q, G) slot is untouched by the correction.
        untouched = np.array([0, 2])
        np.testing.assert_array_equal(out_on[0, untouched], out_off[0, untouched])


def test_tt_head_correction_requires_kgrid():
    from gw.v_q_bispinor import _make_per_q_v_builder_for_tile
    with pytest.raises(ValueError, match="kgrid"):
        _make_per_q_v_builder_for_tile(
            mu_L=1, nu_L=1, bvec=_BVEC, cell_volume=_CELL_VOLUME,
            sys_dim=_SYS_DIM, vcoul_cutoff_ry=None,
            tt_head_correction=True)                          # kgrid=None


def test_tt_head_correction_cc_tile_is_a_no_op():
    """tt_head_correction is a TT-only knob; the CC tile is unaffected
    even when the flag is on (is_CC short-circuits before any vcoul
    call in _make_per_q_v_builder_for_tile)."""
    from gw.v_q_bispinor import _make_per_q_v_builder_for_tile

    q_irr_frac, gvec_components = _one_q_gamma_g0_table()
    b_off = _make_per_q_v_builder_for_tile(
        mu_L=0, nu_L=0, bvec=_BVEC, cell_volume=_CELL_VOLUME,
        sys_dim=_SYS_DIM, vcoul_cutoff_ry=None)
    b_on = _make_per_q_v_builder_for_tile(
        mu_L=0, nu_L=0, bvec=_BVEC, cell_volume=_CELL_VOLUME,
        sys_dim=_SYS_DIM, vcoul_cutoff_ry=None,
        kgrid=_KGRID, tt_head_correction=True)
    np.testing.assert_array_equal(
        np.asarray(b_off(q_irr_frac, gvec_components)),
        np.asarray(b_on(q_irr_frac, gvec_components)))


def test_tt_head_tensor_refuses_box_truncation():
    from gw.v_q_bispinor import _tt_head_tensor
    with pytest.raises(ValueError, match="sys_dim"):
        _tt_head_tensor(
            bvec=_BVEC, cell_volume=_CELL_VOLUME, sys_dim=0, kgrid=_KGRID)


def test_tt_head_tensor_trace_identity_matches_2_vc0():
    """tr(t_ij(q̂)) = 2 at every point on the unit sphere (delta_ii=3
    minus |q̂|²=1), so tr(⟨v(q) t_ij(q̂)⟩_mBZ) == 2⟨v(q)⟩_mBZ EXACTLY,
    for any cell shape -- the same estimator, the same draws, just a
    different weight.  A wrong index order, a wrong sign in t_ij, or an
    off-by-factor in the volume convention would break this identity;
    it is insensitive to nothing about the projector's actual formula
    except the trace of a rank-2 projector being d-1 in d=3."""
    from gw.v_q_bispinor import _tt_head_tensor

    vc0_mean = _bare_vc0()
    T = _tt_head_tensor(
        bvec=_BVEC, cell_volume=_CELL_VOLUME, sys_dim=_SYS_DIM,
        kgrid=_KGRID)
    np.testing.assert_allclose(
        np.trace(T), 2.0 * float(np.real(vc0_mean)), rtol=1e-10)


def test_tt_head_tensor_matches_measured_slab_reference_ratio():
    """In-plane-isotropic slab cell: T/vc0 -> diag(1/2, 1/2, 1), matching
    docs/BISPINOR_DHFB_DESIGN.md §11's bi4 measurement (0.4993, 0.5007,
    1.0000) to Monte-Carlo tolerance on an UNRELATED synthetic cell --
    the shape is a property of the physics (the in-plane mini-BZ angular
    average of the transverse projector), not of one deck's geometry."""
    from gw.v_q_bispinor import _tt_head_tensor

    vc0_mean = _bare_vc0()
    T = _tt_head_tensor(
        bvec=_BVEC, cell_volume=_CELL_VOLUME, sys_dim=_SYS_DIM,
        kgrid=_KGRID)
    ratio = T / float(np.real(vc0_mean))
    np.testing.assert_allclose(np.diag(ratio), [0.5, 0.5, 1.0], atol=5e-3)
    off_diag_mask = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(ratio[off_diag_mask], 0.0, atol=5e-3)


# ---------------------------------------------------------------------------
# Refusal envelope (parse time), mirroring test_low_mem_bands_envelope.py
# ---------------------------------------------------------------------------

_BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""


def _config(tmp_path, extra="", name="bispinor_tt_head.in"):
    from gw.gw_config import LorraxConfig
    path = tmp_path / name
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: None)


def test_default_is_off_and_never_refuses(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.head.bispinor_tt_head_correction is False


def test_refuses_without_bispinor(tmp_path):
    with pytest.raises(ValueError) as exc:
        _config(tmp_path, "bispinor_tt_head_correction = true\n")
    message = str(exc.value)
    assert "bispinor_tt_head_unsupported" in message
    for part in ("got:", "want:", "fix:", "why:", "doc:"):
        assert part in message, f"refusal is missing '{part}'"


def test_refuses_sys_dim_zero(tmp_path):
    with pytest.raises(ValueError) as exc:
        _config(
            tmp_path,
            "bispinor = true\nbispinor_tt_head_correction = true\n"
            "sys_dim = 0\n")
    assert "bispinor_tt_head_unsupported" in str(exc.value)


def test_supported_combination_parses_without_refusing(tmp_path):
    cfg = _config(
        tmp_path,
        "bispinor = true\nbispinor_tt_head_correction = true\n"
        "sys_dim = 2\n")
    assert cfg.bispinor is True
    assert cfg.head.bispinor_tt_head_correction is True


def test_bispinor_false_decks_are_untouched(tmp_path):
    """The flag existing must not change any default-deck resolution."""
    cfg = _config(tmp_path)
    assert cfg.bispinor is False
    assert cfg.head.bispinor_tt_head_correction is False


def test_gw_init_calls_the_canonical_refusal():
    """The driver-entry mirror call exists (parser-altitude coverage
    duplicated for a hand-built cfg), same shape as low_mem_bands's own
    ``test_gw_init_calls_the_canonical_envelope_function_once``."""
    import inspect
    from gw import gw_init
    src = inspect.getsource(gw_init.prepare_isdf_and_wavefunctions)
    assert "refuse_unsupported_bispinor_tt_head_correction(cfg)" in src


def test_the_docs_row_names_the_key():
    repo = pathlib.Path(__file__).resolve().parents[1]
    text = (repo / "docs" / "input_reference.md").read_text()
    assert "bispinor_tt_head_correction" in text
    assert "bispinor_tt_head_unsupported" in text


def test_refusal_doc_pointer_names_a_section_that_actually_exists(tmp_path):
    """Both refusal messages cite a ``docs/input_reference.md`` section by
    its ``## `` heading -- catches the class of bug where the row lands
    under one heading (here, ``## Screening``, beside its sibling
    ``head_minibz_average``) but the "doc:" pointer in the error text
    names a different, nonexistent one (a prior version of this refusal
    pointed at a never-added ``## Bispinor`` heading)."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    headings = {
        line.strip() for line in
        (repo / "docs" / "input_reference.md").read_text().splitlines()
        if line.startswith("## ")}

    with pytest.raises(ValueError) as exc:
        _config(tmp_path, "bispinor_tt_head_correction = true\n")
    message = str(exc.value)
    doc_line = next(ln for ln in message.splitlines() if ln.strip().startswith("doc:"))
    cited_section = doc_line.split("'")[1]
    assert cited_section in headings, (
        f"refusal cites {cited_section!r}, which is not a heading in "
        f"docs/input_reference.md (have: {sorted(headings)})")
