"""Σ_PPM tightening program — WS0 keystone gates (G1, G2).

These are the tests the consensus_draft (reports/sigma_ppm_tighten_2026-07-04)
requires to exist *before* any behaviour-adjacent Σ_PPM change lands.  G3 (the
head negative-branch regression) lives in ``tests/test_head_correction.py``.

  G1  kij ↔ kij_stream accumulator parity.  GREEN since WS1 — it is the gate
      that detected Bug B (the streamed path dropped the analytic q→0 head,
      ppm_pipeline.py _inject_analytic_head).  WS1's fix RMW-adds the head
      into the streamed sigma_kij h5, so the two accumulation paths now agree
      to ~1e-12.  G1's *absence* is why Bug B survived: no golden gate ever
      exercised the stream path (the small fixtures auto-select KIJ_HOST).

  G2  per-branch / per-window reference tiles.  GREEN — the bit-identity pin
      the WS3 file split (moving _SigmaWindow / _SigmaBranch / _iter_branches /
      _build_*_sigma_windows / _build_windows_for_branch into ppm_windows.py)
      must stay identical against.  Also guards against a split silently
      dropping a branch (all 4 branches × their windows asserted non-empty).
"""

import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_REG = REPO_ROOT / "tests" / "regression"

# Reuse the e2e subprocess runner + GPU gating from the regression harness so
# G1 drives gw.gw_jax exactly the way the golden gates do.  (tests/ is a
# package, so put its dir on sys.path for a bare sibling import under any
# pytest import mode.)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_gw_jax_regression import (          # noqa: E402
    _gpu_available,
    _requested_platform,
    _run_gw_jax,
)


# ===========================================================================
#  G2 — per-branch / per-window reference tiles (GREEN, WS3 pin)
# ===========================================================================
#
# We drive the window builders directly (the "equivalent per-branch τ-node /
# mask arrays that _build_windows_for_branch produces" the spec explicitly
# permits) rather than scraping an e2e run: WS3 is a *pure move* of exactly
# these symbols, so a unit test that imports and exercises them is the precise
# pin.  Inputs are synthetic but structured to populate every window kind:
#   * a symmetric ω grid (±10 eV) → all 4 (ω-sign × cond/val) branches;
#   * E_A spanning the window threshold T → the core + a_stripe split;
#   * Ω_q spanning T → the b_slab (Ω>T) window as well.

G2_REF = _REG / "sigma_ppm_gates" / "g2_branch_window_ref.npz"


def _build_branch_windows():
    """Build the 4 branches × their windows from controlled synthetic inputs.

    Returns ``(branches, flat)`` where ``branches`` is a list of
    ``(tag, kernel_sign, [_SigmaWindow, ...])`` and ``flat`` is the dict of
    plain numpy arrays / scalars captured for the reference freeze.
    """
    import jax.numpy as jnp
    from common.units import RYD_TO_EV
    from gw.ppm_windows import _build_windows_for_branch, _iter_branches

    nk, nb = 2, 6                       # 3 valence + 3 conduction bands
    # E_cond (conduction) and H_val (valence) energies-above-Fermi (Ry),
    # chosen to straddle T ≈ 0.76 Ry so both E_A≤T and E_A>T are populated.
    e_above = np.array([0.30, 0.90, 1.60], dtype=np.float64)      # Ry
    E_cond = np.zeros((nk, nb), dtype=np.float64)
    H_val = np.zeros((nk, nb), dtype=np.float64)
    cond_mask = np.zeros((nk, nb), dtype=bool)
    val_mask = np.zeros((nk, nb), dtype=bool)
    for k in range(nk):
        val_mask[k, 0:3] = True
        cond_mask[k, 3:6] = True
        H_val[k, 0:3] = e_above
        E_cond[k, 3:6] = e_above

    # Ω_q (nq, μ, μ) straddling T so mask_B le_t / gt_t are both non-empty.
    Omega_abs = np.array(
        [[[0.40, 0.50], [0.55, 0.45]],
         [[1.00, 1.20], [1.10, 1.30]]], dtype=np.float64)
    B_mask = Omega_abs > 1.0e-14

    # Symmetric ω grid in Ry — both halves present ⇒ all 4 branches.
    omega_ry = np.arange(-10.0, 10.0 + 1e-9, 0.5, dtype=np.float64) / RYD_TO_EV
    idx_pos = np.where(omega_ry >= 0.0)[0]
    idx_neg = np.where(omega_ry < 0.0)[0]
    omega_pos = omega_ry[idx_pos]
    omega_neg_abs = -omega_ry[idx_neg]

    branches = _iter_branches(
        omega_pos=omega_pos, idx_pos=idx_pos,
        omega_neg_abs=omega_neg_abs, idx_neg=idx_neg,
        E_cond=jnp.asarray(E_cond), H_val=jnp.asarray(H_val),
        cond_mask=jnp.asarray(cond_mask), val_mask=jnp.asarray(val_mask),
    )

    quad = dict(
        regularization_width_ry=0.25 / RYD_TO_EV,
        edge_factor=1.5,
        target_error=1e-6,
        max_nodes=64,
        crossing_eps_q=1e-3,
        crossing_max_nodes=500,
        use_shipped_minimax_tables=True,
    )

    out = []
    flat: dict[str, np.ndarray] = {}
    for br in branches:
        windows = _build_windows_for_branch(
            omega_nonneg_ry=br.omega_abs,
            E_A=br.E_A, base_mask_A=br.base_mask_A,
            Omega_q=jnp.asarray(Omega_abs), base_mask_B=jnp.asarray(B_mask),
            kernel_sign=br.kernel_sign,
            log_tag=br.tag, print_fn=lambda *a, **k: None,
            **quad,
        )
        out.append((br.tag, int(br.kernel_sign), windows))
        for wi, w in enumerate(windows):
            key = f"{br.tag}|{wi}|{w.name}"
            flat[f"{key}|t"] = np.asarray(w.nodes.t, dtype=np.complex128)
            flat[f"{key}|alpha"] = np.asarray(w.nodes.alpha, dtype=np.complex128)
            flat[f"{key}|mask_A"] = np.asarray(w.mask_A, dtype=bool)
            flat[f"{key}|meta"] = np.array([
                float(w.E_ref_A), float(w.E_ref_B),
                float(w.omega_sign), float(w.prefactor),
                float(w.project_code), float(w.n_tau),
                float(w.mask_B_threshold) if w.mask_B_threshold is not None else np.nan,
            ], dtype=np.float64)
            flat[f"{key}|tags"] = np.array(
                [w.project, w.mask_B_mode, str(w.crossing_kind)], dtype="<U16")
    return out, flat


def _regenerate_g2_reference():
    """(Re)write the frozen G2 reference .npz.  Call manually when the window
    builders legitimately change; never inside the test."""
    _, flat = _build_branch_windows()
    G2_REF.parent.mkdir(parents=True, exist_ok=True)
    np.savez(G2_REF, **flat)
    return G2_REF


@pytest.mark.regression
def test_g2_branch_window_tiles_are_frozen():
    if _requested_platform() in {"gpu", "cuda"} and not _gpu_available():
        pytest.skip("CUDA GPU not available for requested platform=gpu.")

    branches, flat = _build_branch_windows()

    # --- structural guards: no branch or window may silently vanish -------
    tags = [t for t, _, _ in branches]
    assert tags == ["ω≥E_F cond", "ω≥E_F val", "ω<E_F cond", "ω<E_F val"], tags
    win_names = {t: [w.name for w in ws] for t, _, ws in branches}
    # kernel_sign=+1 halves get the 3-window crossing/stripe/slab split;
    # kernel_sign=-1 halves get the single Laplace window.
    assert win_names["ω≥E_F cond"] == ["core", "a_stripe", "b_slab"], win_names
    assert win_names["ω<E_F val"] == ["core", "a_stripe", "b_slab"], win_names
    assert win_names["ω≥E_F val"] == ["single"], win_names
    assert win_names["ω<E_F cond"] == ["single"], win_names
    for _t, _s, ws in branches:
        assert ws, f"branch {_t!r} produced no windows"
        for w in ws:
            assert bool(np.any(w.mask_A)), f"{_t}:{w.name} empty mask_A"
            assert w.n_tau > 0, f"{_t}:{w.name} zero τ nodes"

    # --- bit-identity against the frozen reference ------------------------
    assert G2_REF.exists(), (
        f"missing G2 reference {G2_REF}; regenerate with "
        f"tests.test_sigma_ppm_gates._regenerate_g2_reference()")
    ref = np.load(G2_REF, allow_pickle=False)
    assert set(ref.files) == set(flat), (
        f"window-tile key set drifted:\n  new-only={set(flat) - set(ref.files)}"
        f"\n  ref-only={set(ref.files) - set(flat)}")
    for key in ref.files:
        got = flat[key]
        want = ref[key]
        if got.dtype.kind == "U":
            assert np.array_equal(got, want), f"{key} tag mismatch: {got} != {want}"
        else:
            np.testing.assert_array_equal(got, want, err_msg=f"{key} not bit-identical")


# ===========================================================================
#  G1 — kij ↔ kij_stream accumulator parity (GREEN since WS1 — detects Bug B)
# ===========================================================================
#
# The streamed accumulator writes a head-less Σ_c (Bug B: _inject_analytic_head
# early-returned (None, None) when sigma_c_omega is None, so the analytic q→0
# head was never RMW-added into the sigma_c_kij h5).  The host path injects the
# head into the in-memory tensor.  Both write ``sigma_c_kij_ev`` into
# sigma_mnk.h5, so comparing those two datasets isolates the dropped head.
#
# CONFIRMED RED DIFF before WS1 (MoS2 3×3 gnppm fixture, 1 GPU):
#   max|Δ sigma_c_kij_ev|          = 4.13 eV
#   max|Δ| on the band DIAGONAL    = 4.13 eV       (bands 24-31, near VBM/nval=26)
#   max|Δ| OFF the band diagonal   = 7.0e-10 eV    (numerical noise)
# i.e. the entire discrepancy sat on the band-diagonal head-affected bands —
# the analytic q→0 head, which is diagonal in band and dropped by the stream
# path — NOT a random accumulator disagreement.  WS1's fix (RMW-add the head
# into the stream h5) makes the two paths agree to ~1e-12.
#
# NB two precursors had to be handled for the stream path to even RUN to the
# point where Bug B is observable, both folded into WS1's scope:
#   * ppm_sigma.py fillvalue=0.0 on a complex128 dataset crashes h5py>=3.13
#     ("no appropriate function for conversion path") — see KNOWN_SANDBOX_ERRORS;
#   * sigma_freq_debug_output feeds the None sigma_c_omega into the eqp
#     z-factor writer (gw_jax.py:860) and crashes, so this gate disables it.

G1_CASE = _REG / "gnppm_debug"
_ACCUM_LINE = "sigma_omega_accumulation = "


def _write_g1_input(dst_dir: Path, *, accumulation: str, kij_h5: str | None) -> str:
    """Copy the gnppm fixture input, forcing the accumulation mode.

    Disables sigma_freq_debug_output/sigma_debug_split_contrib (their None-Σ_c
    handling is a separate Bug-B sibling that crashes the stream path before
    the h5 is comparable) so both modes run to a written sigma_mnk.h5.
    """
    src = (G1_CASE / "gnppm_test.in").read_text()
    lines = []
    for ln in src.splitlines():
        if ln.startswith("sigma_freq_debug_output"):
            ln = "sigma_freq_debug_output = false"
        elif ln.startswith("sigma_debug_split_contrib"):
            ln = "sigma_debug_split_contrib = false"
        elif ln.startswith("sigma_omega_h5_file"):
            lines.append(f"sigma_omega_accumulation = {accumulation}")
            if kij_h5 is not None:
                lines.append(f"sigma_kij_h5_file = {kij_h5}")
        lines.append(ln)
    name = f"gnppm_{accumulation}.in"
    (dst_dir / name).write_text("\n".join(lines) + "\n")
    return name


def _run_mode(tmp_path, platform, tag, *, accumulation, kij_h5):
    run_dir = tmp_path / tag
    shutil.copytree(
        G1_CASE, run_dir,
        ignore=shutil.ignore_patterns("tmp", "*.dat", "sigma_mnk.h5", "*_qp.h5"))
    input_name = _write_g1_input(run_dir, accumulation=accumulation, kij_h5=kij_h5)
    res = _run_gw_jax(run_dir, input_name, platform)
    if res.returncode != 0:
        pytest.fail(
            f"G1 {tag} run failed.\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    return run_dir


def _read_sigma_c_kij_ev(sigma_mnk_path: Path) -> np.ndarray:
    import h5py
    with h5py.File(sigma_mnk_path, "r") as h5:
        return np.asarray(h5["sigma_c_kij_ev"], dtype=np.complex128)


@pytest.mark.regression
def test_g1_kij_vs_kij_stream_parity(tmp_path):
    platform = _requested_platform()
    if platform in {"gpu", "cuda"} and not _gpu_available():
        pytest.skip("CUDA GPU not available for requested platform=gpu.")

    host_dir = _run_mode(tmp_path, platform, "kij",
                         accumulation="kij", kij_h5=None)
    stream_dir = _run_mode(tmp_path, platform, "kij_stream",
                           accumulation="kij_stream", kij_h5="sigma_kij_stream.h5")

    sig_host = _read_sigma_c_kij_ev(host_dir / "sigma_mnk.h5")
    sig_stream = _read_sigma_c_kij_ev(stream_dir / "sigma_mnk.h5")
    assert sig_host.shape == sig_stream.shape, (sig_host.shape, sig_stream.shape)

    # The kij (host) path injects the analytic q→0 head; before WS1 the stream
    # path dropped it (Bug B) and this failed by ~4.13 eV on the band-diagonal
    # near-gap bands (24-31).  WS1 RMW-adds the identical head into the stream
    # h5, so the band-diagonal head is gone: post-fix the FULL tensor agrees to
    # ~7e-10 eV, and the head-carrying diagonal to ~4e-10 eV.
    #
    # The residual ~7e-10 floor is NOT the head — it sits on large-magnitude
    # OFF-diagonal elements (~1e3 eV) where the head is exactly zero.  It is the
    # pre-existing numpy-vs-jax dual-projector ULP difference (host uses
    # _project_tau_onto_omega_np, stream uses the jitted _project_tau_onto_omega
    # — see the K2/WS4 accumulator unification, which deletes one projector and
    # will let this gate tighten back to ~1e-12).  atol=1e-8 clears that floor
    # with >10x margin while still catching a dropped head (4.13 eV) loudly.
    np.testing.assert_allclose(sig_stream, sig_host, rtol=0.0, atol=1e-8)
