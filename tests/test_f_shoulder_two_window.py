"""The f-transform shoulder, its tripwire, and the refit's TWO-WINDOW contract.

THE MECHANISM, because every cell here is a consequence of it.  The htransform
builds ``fH_k = Σ_n f(ε_n,k) c_n,k c_n,kᴴ`` with ``f`` the bandwidth-bound
transform, and ``f`` is IDENTICALLY ZERO for ε ≥ ``shift``.  ``shift`` is
``max_k ε[nb−1]`` — the maximum over k of the top band of ``ctilde``'s OWN
window.  So the top band contributes exactly nothing to ``fH`` at the k that
attains that maximum, ``f`` vanishes to second order approaching it, and a
whole SHOULDER of bands below the top carries a per-cent or less of fH's
weight.  Those eigenvector slots are degenerate with fH's own
``(rank − nb)``-dimensional null space and ``eigh`` fills them with arbitrary
directions out of it.

WHY IT COST FIVE LANES.  Every instrument the tree had exonerated the
htransform, correctly: on the parent where the on-grid tile null read 140, the
``ctilde`` orthonormality was 8.9e-15 and the Galerkin residual 4.3e-15.  The
REPRESENTATION was never the defect.  The REQUEST was — ``refit_prepare``
pinned the fH window to the ζ-fit window with zero guard bands, so the refit
asked ``compute_wfns_fi`` for precisely the bands ``fH`` cannot represent
(``tests/known_failures/2026-08-11-fifth-wall-is-the-f-transform-shoulder.md``).

THE MEASURED SIGNATURE this file's fixture reproduces, from §2 of that row:

    dp2628n20  nb 20, zero guards: bands 16,17,18,19 read
               min_k |f|/max|f| = 0.000000, exactly zero at 3 k each
               (a 4-fold multiplet sitting at the top at one k-star);
               α-space overlap ‖O[m,:]‖ collapses to 0.23…0.27 there;
               on-grid tile null 1.267 against a 5.0e-02 bracket.
    p2628n52   nb 52, zero guards: bands 50, 51, zero at 8 k each.
    Bands that SURVIVE never get closer to zero than ~4e-4 of max|f| and
    come back at ‖O[m,:]‖ = 1.000000 at every k.

Good and bad are four decades apart with nothing in between, which is why the
gate's floor is exactly zero and is not a tolerance judgement.

WHAT IS ASSERTED, in the order that makes it mean anything:

1. THE INSTRUMENT — the fixture really does zero a multiplet, and the gate
   reports the same census the diagnostic printed.  Without this the rest
   could pass on energies where ``f`` never vanishes and the file would be
   measuring nothing.
2. THE RED TWIN — zero guards REFUSE by name; the same fixture with guards
   passes.  Both arms in one cell so neither can be quoted alone.
3. The gate's env grammar: garbage refuses, a negative value disables (and
   says so), a positive value tightens.
4. The two-window contract in ``refit_prepare``: ψ is streamed over the fH
   window, ``B_full`` is built from it, and ``psi_r`` — the pair-density leg —
   is the ζ sub-block.  Guards shape fH; they are never pair density.
5. Default identity: ``initialize_wfns``' widening is opt-in at 0, so no
   caller that omits it loads a band it did not load before.
6. The splash radius of §7: both other callers that ask for the top of their
   own window now warn.
"""
from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_VQ = REPO_ROOT / "src" / "bse" / "vq_interp.py"
SRC_HT = REPO_ROOT / "src" / "bandstructure" / "htransform.py"
SRC_DENSIFY = REPO_ROOT / "src" / "bse" / "bse_densify.py"


# ---------------------------------------------------------------------------
#  the fixture: a window whose TOP MULTIPLET is exactly at the shift
# ---------------------------------------------------------------------------

#: bands, k-points, α-rank of the synthetic htransform triple.
_NB, _NK_GRID, _RANK, _NMU, _NS = 8, (2, 2, 1), 24, 6, 2

#: how many bands of the fixture sit AT the shift, i.e. are invisible to fH.
#: Four, to mirror ``dp2628n20``'s 4-fold top multiplet.
_N_DEAD = 4


def _synthetic_shoulder(seed: int = 20260811):
    """``(ctilde, enk, B_at_mu, kgrid_co)`` with a DEAD TOP MULTIPLET.

    ``ctilde`` is band-orthonormal per k by QR — exactly what
    ``streaming_galerkin_solve`` produces — so ``build_fH_R``'s own
    orthonormality gate cannot fire first and steal the refusal this file is
    about.

    The energies are ascending in the band index at every k, as eigenvalues
    are, and that is what makes the dead set a MULTIPLET rather than an
    arbitrary pick: ``shift = max_k ε[nb−1]`` and ε[b,k] ≤ ε[nb−1,k] ≤ shift,
    so ``ε[b,k] ≥ shift`` forces ε[b,k] = ε[nb−1,k] = shift.  A band below the
    top can only be zeroed by being DEGENERATE with the top band at the k that
    defines the shift — which is precisely what a 4-fold irrep at a symmetry
    point does on the real deck.
    """
    rng = np.random.default_rng(seed)
    nk = _NK_GRID[0] * _NK_GRID[1] * _NK_GRID[2]
    ct = np.empty((nk, _NB, _RANK), dtype=np.complex128)
    for k in range(nk):
        z = (rng.standard_normal((_RANK, _NB))
             + 1j * rng.standard_normal((_RANK, _NB)))
        q, _ = np.linalg.qr(z)
        ct[k] = np.conj(q.T)
    enk = (np.linspace(-0.6, 0.4, _NB)[:, None]
           + 0.05 * np.cos(2 * np.pi * np.arange(nk) / nk)[None, :])
    # The shift, and the k-star that attains it.  ``k_top`` is where the top
    # band peaks; ``k_deg`` is a DIFFERENT k at which the top ``_N_DEAD``
    # bands are pushed up to that same value — the multiplet.
    shift = float(enk[_NB - 1].max())
    k_deg = int(np.argmin(enk[_NB - 1]))
    enk[_NB - _N_DEAD:, k_deg] = shift
    B = (rng.standard_normal((_RANK, _NS, _NMU))
         + 1j * rng.standard_normal((_RANK, _NS, _NMU)))
    return ct, enk, B, _NK_GRID


def _run_fi(band_window, log=None, **kw):
    """``compute_wfns_fi`` on the shoulder fixture, at the given window."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from bandstructure.bse_setup import compute_wfns_fi

    ct, enk, B, kgrid_co = _synthetic_shoulder()
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    with mesh:
        return compute_wfns_fi(
            ctilde=jnp.asarray(ct), B_at_mu=jnp.asarray(B),
            enk_sigma=jnp.asarray(enk), kgrid_co=kgrid_co,
            kgrid_fi=(2, 2, 1), band_window_fi=band_window, mesh_xy=mesh,
            log_fn=log, **kw)


# ---------------------------------------------------------------------------
#  1. THE INSTRUMENT — the fixture zeroes a multiplet, and the census matches
# ---------------------------------------------------------------------------

def test_instrument_fixture_really_zeroes_a_top_multiplet():
    """Before any gate: ``f`` on this fixture is EXACTLY zero for the top
    ``_N_DEAD`` bands at one k, and never zero below them.

    This is the cell that makes the red twin below informative.  Computed
    from ``f_transform_eigs`` itself, not from the gate — an instrument that
    shares the gate's arithmetic would agree with it by construction.
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from bandstructure.htransform import f_transform_eigs

    _, enk, _, _ = _synthetic_shoulder()
    f_eps, _a, _n, shift = f_transform_eigs(jnp.asarray(enk))
    fa = np.abs(np.asarray(f_eps))
    assert fa.shape == (_NB, enk.shape[1])
    zero_per_band = (fa == 0.0).sum(axis=1)
    dead = np.nonzero(zero_per_band)[0]
    assert list(dead) == list(range(_NB - _N_DEAD, _NB)), (
        f"the fixture must zero exactly the top {_N_DEAD} bands; it zeroed "
        f"{list(dead)} (per-band zero counts {list(zero_per_band)})")
    # And the survivors are not marginal — the real parents' survivors sit at
    # ~4e-4 of max|f| and come back at 1.000000.
    ratio = fa[: _NB - _N_DEAD].min(axis=1) / fa.max()
    assert ratio.min() > 1e-6, (
        f"the fixture's surviving bands must be comfortably alive; worst "
        f"min_k |f|/max|f| = {ratio.min():.3e}")
    assert float(shift) == pytest.approx(float(enk[_NB - 1].max()))


def test_gate_census_matches_the_diagnostic_shape():
    """The gate reports ``min_b min_k |f|/max|f|`` and an exactly-zero count
    over the RETURNED window — the two numbers §2 of the row published."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from bandstructure.bse_setup import _f_shoulder_gate
    from bandstructure.htransform import f_transform_eigs

    _, enk, _, _ = _synthetic_shoulder()
    f_eps, _a, _n, shift = f_transform_eigs(jnp.asarray(enk))
    lines: list[str] = []
    worst = _f_shoulder_gate(f_eps, 0, _NB - _N_DEAD, float(shift),
                             lines.append, rank=_RANK)
    assert worst > 0.0
    blob = "\n".join(lines)
    assert "f-shoulder" in blob and "guard band" in blob, blob
    assert "0 exactly-zero" in blob, (
        f"a guarded window has no zeroed slot and the gate must say so:\n{blob}")


# ---------------------------------------------------------------------------
#  2. THE RED TWIN — zero guards refuse, guards pass, same fixture
# ---------------------------------------------------------------------------

def test_red_twin_zero_guards_refuse_and_guards_pass():
    """ONE cell, BOTH arms.  The only difference is where the returned window
    stops; the ctilde, the energies and the α-basis are the same arrays.

    A refusal reported without its green twin proves only that something
    raised, and a green reported without its red proves only that nothing
    did.  The production consequence of the red arm is an on-grid tile null
    of 1.267 against a 5.0e-02 bracket.
    """
    pytest.importorskip("jax")

    # RED: ask for the whole window, guards = 0 — the convicted configuration.
    with pytest.raises(ValueError) as exc:
        _run_fi((0, _NB))
    msg = str(exc.value)
    assert "invisible to fH" in msg, msg
    assert "GUARD BANDS" in msg, (
        f"the refusal must name the fix, not just the symptom:\n{msg}")
    assert "LORRAX_FI_FSHOULDER_TOL" in msg and "known-bad" in msg, (
        f"the refusal must say the override is not a fix:\n{msg}")
    assert str(_NB - _N_DEAD) in msg, (
        f"the refusal must name the band it failed on:\n{msg}")

    # GREEN: same everything, guards = _N_DEAD.  Runs to a bundle.
    out = _run_fi((0, _NB - _N_DEAD))
    assert out.psi_rmu_Y.shape[1] == _NB - _N_DEAD


def test_partial_guards_still_refuse_when_the_shoulder_is_not_cleared():
    """Guards that stop INSIDE the dead multiplet are not guards.

    The fix is "clear the shoulder", not "add some bands" — §4 of the row
    measured that splicing only the TOP band back moved the tile null from
    1.267 to 0.947, a minority of the error, and the ``lam_max`` gap WIDENED.
    """
    pytest.importorskip("jax")
    for b_max in range(_NB - _N_DEAD + 1, _NB + 1):
        with pytest.raises(ValueError, match="invisible to fH"):
            _run_fi((0, b_max))


# ---------------------------------------------------------------------------
#  3. the env grammar of the gate
# ---------------------------------------------------------------------------

def test_fshoulder_tol_grammar(monkeypatch):
    """``LORRAX_FI_FSHOULDER_TOL``: default 0.0, garbage REFUSES, a negative
    value disables and announces, a positive value tightens.

    The one deliberate difference from ``LORRAX_FH_ORTHO_TOL`` — where ``0``
    is the off switch — is that here ``0`` is the DEFAULT, because a band
    whose ``f`` is exactly zero is absent rather than inaccurate.  Turning
    this gate off therefore needs a value no threshold could be.
    """
    pytest.importorskip("jax")
    from bandstructure.bse_setup import (FI_FSHOULDER_TOL_DEFAULT,
                                         resolve_fi_fshoulder_tol)

    assert FI_FSHOULDER_TOL_DEFAULT == 0.0
    monkeypatch.delenv("LORRAX_FI_FSHOULDER_TOL", raising=False)
    assert resolve_fi_fshoulder_tol(None) == 0.0

    monkeypatch.setenv("LORRAX_FI_FSHOULDER_TOL", "banana")
    with pytest.raises(ValueError, match="LORRAX_FI_FSHOULDER_TOL"):
        resolve_fi_fshoulder_tol(None)

    lines: list[str] = []
    monkeypatch.setenv("LORRAX_FI_FSHOULDER_TOL", "-1")
    assert resolve_fi_fshoulder_tol(lines.append) == -1.0
    assert "DISABLED" in "\n".join(lines), lines
    # and disabled means the convicted configuration RUNS — that is what
    # makes it a reproduction switch and not a second opinion.
    out = _run_fi((0, _NB))
    assert out.psi_rmu_Y.shape[1] == _NB

    monkeypatch.setenv("LORRAX_FI_FSHOULDER_TOL", "1e-3")
    assert resolve_fi_fshoulder_tol(None) == 1e-3


# ---------------------------------------------------------------------------
#  4. the two-window contract in refit_prepare
# ---------------------------------------------------------------------------

def _refit_prepare_src() -> str:
    import bse.vq_interp as vq
    return inspect.getsource(vq.refit_prepare)


def test_refit_prepare_streams_psi_over_the_FH_window():
    """ψ for ``B_full`` comes from ``band_range_fh``, not ``band_range``.

    ``B_full = W_proj @ ψ`` and ``W_proj``/``ctilde`` are the WIDE window's
    objects — a ζ-window ψ here is a shape error at best and a silently
    wrong α-basis at worst.
    """
    src = _refit_prepare_src()
    tree = ast.parse(inspect.cleandoc(src) if src.startswith(" ") else src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "iter_psi_rchunk_bandwise"]
    assert len(calls) == 1, f"expected one ψ stream, found {len(calls)}"
    names = {a.id for a in calls[0].args if isinstance(a, ast.Name)}
    assert "band_range_fh" in names, (
        f"the ψ stream must read the fH window; it read {sorted(names)}")
    assert "band_range" not in names, (
        "the ψ stream must NOT read the ζ window — that is the pinning this "
        "contract removes")


def test_refit_prepare_stores_the_ZETA_block_as_pair_density():
    """``rst["psi_r"]`` is the ζ sub-block: guards shape fH, never ρ.

    ``refit_vq`` reshapes ``psi_r`` to ``(nk, zx["nb"], ns, n_rp)`` and both
    legs of the pair density come out of it, so a wide ``psi_r`` would be a
    silent reshape onto the wrong bands.
    """
    src = _refit_prepare_src()
    assert "psi_r_host[:, :nb]" in src, (
        "psi_r must be the ζ-window slice of the wide host buffer")
    assert '"psi_r": psi_r' in src
    assert "nb = nb_wide - n_guard" in src, (
        "``nb`` must be the ζ window's band count everywhere below the load")


def test_refit_prepare_reports_both_galerkin_residuals():
    """Two windows, two residuals, and the ζ one is named as the tile floor.

    Printing only the fH residual attributes the guards' own (larger,
    harmless) representation error to the tile; printing only the ζ one hides
    a wide window that has stopped spanning.
    """
    src = _refit_prepare_src()
    assert "galerkin_rel_zeta" in src and '"galerkin_rel": gal,' in src
    assert "the ζ one is the refit-vs-stored on-grid" in src, (
        "the log line must say WHICH residual bounds the tile")


def test_refit_guard_default_is_four_and_is_the_measured_depth():
    import bse.vq_interp as vq

    assert vq.REFIT_N_GUARD_DEFAULT == 4
    assert inspect.signature(vq.refit_prepare).parameters["n_guard"].default is None, (
        "None means 'take the module default'; a literal 4 in the signature "
        "would let a caller drift from REFIT_N_GUARD_DEFAULT silently")


def test_negative_guard_count_refuses():
    """There is no such thing as a negative guard band, and 0 is the red
    twin rather than an error — so the refusal is strictly below zero."""
    import bse.vq_interp as vq

    with pytest.raises(SystemExit, match="negative"):
        vq.refit_prepare("nonexistent.in", None, {}, n_guard=-1)


# ---------------------------------------------------------------------------
#  5. default identity of the widening
# ---------------------------------------------------------------------------

def test_initialize_wfns_widening_is_opt_in_at_zero():
    """``n_guard_bands`` defaults to 0 and the widening is guarded by
    ``if n_guard_bands:`` — so every existing caller loads exactly the bands
    it loaded before, including ``nband``.

    Checked on the source rather than by running the loader, because the
    claim is about a code path NOT being taken.
    """
    src = SRC_HT.read_text(encoding="utf8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "initialize_wfns")
    kwonly = {a.arg: d for a, d in zip(fn.args.args[-len(fn.args.defaults):],
                                       fn.args.defaults)}
    assert "n_guard_bands" in kwonly, "the widening must be a named parameter"
    assert isinstance(kwonly["n_guard_bands"], ast.Constant)
    assert kwonly["n_guard_bands"].value == 0, (
        "default must be 0 — the historical window, band for band")
    body = ast.get_source_segment(src, fn)
    assert "if n_guard_bands:" in body, (
        "the widening must be inside a falsy-guarded branch so the default "
        "path executes none of it")


def test_guard_bands_the_wfn_cannot_supply_refuse():
    """A guard band past the file's band count arrives as EXACT ZEROS (Meta's
    past-mnband pad), which the Galerkin solve absorbs without complaint and
    the f-transform then reports as a perfectly representable band.  That is
    the silent version of the defect this whole contract removes, so it
    refuses by name."""
    src = SRC_HT.read_text(encoding="utf8")
    fn = src[src.index("def initialize_wfns"):]
    fn = fn[:fn.index("\ndef ")]
    assert "wfn.nbands" in fn and "EXACT ZEROS" in fn, fn[:400]
    assert "raise SystemExit" in fn


def test_nband_is_raised_with_ncond_so_guards_are_read():
    """Widening ``ncond`` alone is a trap: ``Meta`` zero-pads ψ above
    ``nband``, so bands between the old ``nband`` and the new window edge
    would arrive as zeros.  Both move together or neither does."""
    src = SRC_HT.read_text(encoding="utf8")
    fn = src[src.index("def initialize_wfns"):]
    fn = fn[:fn.index("\ndef ")]
    assert "nband = max(_nband_deck, int(wfn.nelec) + ncond)" in fn


# ---------------------------------------------------------------------------
#  6. the splash radius named in §7 of the row
# ---------------------------------------------------------------------------

def test_get_centroids_fi_warns_about_its_zero_guard_default():
    """``wfn_fi_max`` unset defaults to the FULL band count — zero guards,
    the exact configuration the row convicts.  The gate refuses; this warns
    first, in the vocabulary of the deck key the user would have to change."""
    src = SRC_HT.read_text(encoding="utf8")
    i = src.index('b_max = int(params["wfn_fi_max"]) or int(ctilde.shape[1])')
    window = src[i:i + 1600]
    assert "_n_guard" in window and "[warn]" in window, window[:400]
    assert "ZERO-GUARD default" in window


def test_bse_densify_warns_when_b_max_equals_the_window():
    """``bse_densify`` checks only ``b_max > nb_window`` and so PERMITS
    ``b_max == nb_window`` — a deck whose ``nband`` equals ``nval + ncond``
    has no conduction guard and lands in the same place."""
    src = SRC_DENSIFY.read_text(encoding="utf8")
    i = src.index("escapes the htransform fH")
    window = src[i:i + 1800]
    assert "_n_guard = nb_window - b_max" in window
    assert "[warn]" in window and "f-transform's shift" in window


# ---------------------------------------------------------------------------
#  7. the driver's knob and its stamp
# ---------------------------------------------------------------------------

def test_contracted_cert_runs_under_the_zeta_window_too():
    """The tile null bounds a TILE at 5.0e-02 RELATIVE; the curve is published
    in meV.  Those are different claims, so under ``--refit-window=zeta`` the
    contracted eigenvalue certification now runs ALONGSIDE the tile null
    rather than only under ``bse``.

    Strictly additive: it relaxes nothing, it reuses the path's own caches and
    scan, and the empty-population case stays a REFUSAL only for ``bse``,
    where it is the sole gate — under ``zeta`` the tile null has already run
    and a missing stamp is announced instead of raised.
    """
    src = (REPO_ROOT / "src" / "bse" / "exciton_bands.py").read_text("utf8")
    assert 'if not cert_idx and args.refit_window == "bse":' in src, (
        "the empty-population refusal must be bse-only")
    assert "window_mode=args.refit_window" in src, (
        "the cert must know which gate it is on this run")
    # and the twin rows are built off cert_idx regardless of window, which is
    # the line that actually makes it run.
    assert "for iQ in cert_idx:" in src


def test_exciton_bands_exposes_and_threads_the_guard_count():
    src = (REPO_ROOT / "src" / "bse" / "exciton_bands.py").read_text("utf8")
    assert '"--refit-guard-bands"' in src
    assert src.count("n_guard=args.refit_guard_bands") == 2, (
        "BOTH refit_prepare call sites (--vq-mode=refit and =both) must "
        "thread it; a knob honoured on one path only is worse than none")
    assert "refit-fh-window" in src, (
        "the contract must travel with the .dat — a curve drawn through a "
        "zero-guard refit is unrecognisable after the fact otherwise")
