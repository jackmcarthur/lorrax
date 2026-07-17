"""Γ-on-site BSE degeneracy gate — screening-window closure + tile covariance.

Root cause (reports/bse_refactor_map_2026-07-15 PHASE2_LOG "Si BSE symmetry —
Round 2"; FINDINGS2 Task 1/4): splitting a degenerate multiplet at the SCREENING
ISDF fit window top makes the fitted ζ̃ — and every V_q/W_q tile built from it —
non-covariant under the crystal symmetry, which splits Γ-on-site exciton
multiplets that the crystal symmetry forces to be degenerate.  The
screening-window degeneracy fix (``common/meta.py``
``round_band_window_to_closed_shell``) closes that window; this gate guards the
downstream invariant.

Construction: over an auto-detected degeneracy-closed ``(nv, nc)`` window at Γ,
build the Γ-on-site block ``H_Γ = D + Kx − Kd`` (pure numpy ``eigvalsh`` on a
``(nc·nv)²`` matrix — ms, no second GW run) from BOTH the production q=0 tiles
and their little-group-symmetrized (exactly covariant) counterparts.

Two invariants:
  1. Little-group symmetrization must not move the Γ-on-site spectrum — i.e. the
     production tiles ARE covariant (``max|λ_raw − λ_cov|`` small).  A regression
     in screening-window closure or ζ-fit covariance breaks this.
  2. Any covariant-spectrum multiplet (crystal-symmetry-forced degeneracy) must
     be degenerate under the covariant tiles (``< MULT_TOL``) and split only
     within tolerance under the raw tiles.

Calibration (2026-07-16, committed gnppm MoS2 fixture, ntran=2): the tiles are
already covariant (little-group symmetrization moves V0/W0 by ~2.8e-9 and the
Γ-on-site spectrum by ~4e-4 μeV); the low-symmetry fixture has NO Γ exciton
multiplets, so invariant (2) is vacuous there and invariant (1) carries the
gate.  The TIGHT tier is active because the fix lands together with this gate;
the LOOSE tier documents the Si Γ-block raw-tile floor (36 μeV, FINDINGS2 Task 1)
a gross regression would cross.

Piggybacks the session-scoped ``gnppm_session`` GW run — no second GW run.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

RY = 13.6056980659

# Two-tier thresholds (μeV).  See module docstring for calibration.
TIER_LOOSE_UEV = 50.0    # gross-regression guard (Si Γ-block raw floor 36 μeV)
TIER_TIGHT_UEV = 5.0     # post-fix covariant-tile level (active with the fix)
MULT_TOL_UEV = 1.0       # covariant-tile multiplet must be degenerate to this
_WINDOW_TARGET = 4       # target half-window (bands) before degeneracy rounding


def _build_gamma_H(pc, pv, ec, ev, V0, W0):
    """Γ-on-site exciton block H = D + Kx − Kd (single k, no 1/Nk — a uniform
    scale that leaves the degeneracy structure invariant).

    pc/pv : (nc/nv, s, μ) ψ at Γ; ec/ev : (nc/nv,) DFT energies (Ry);
    V0/W0 : (μ, μ) bare/screened q=0 tiles.  Kx = ⟨cv|v|c'v'⟩,
    Kd = ⟨cc'|W|vv'⟩ (Henneke exchange/direct at q=0).
    """
    nc = pc.shape[0]; nv = pv.shape[0]
    M = np.einsum("csm,vsm->cvm", np.conj(pc), pv, optimize=True)
    Kx = np.einsum("cvm,mn,CVn->cvCV", np.conj(M), V0, M, optimize=True)
    Pc = np.einsum("csm,Csm->cCm", np.conj(pc), pc, optimize=True)
    Pv = np.einsum("vsn,Vsn->vVn", pv, np.conj(pv), optimize=True)
    Kd = np.einsum("cCn,vVn->cvCV",
                   np.einsum("cCm,mn->cCn", Pc, W0, optimize=True), Pv,
                   optimize=True)
    D = np.zeros((nc, nv, nc, nv), dtype=np.complex128)
    for c in range(nc):
        for v in range(nv):
            D[c, v, c, v] = ec[c] - ev[v]
    H = (D + Kx - Kd).reshape(nc * nv, nc * nv)
    return 0.5 * (H + H.conj().T)


def _symmetrize_q0(T0, alpha, ntran):
    """Little-group-symmetrize a q=0 tile: at q=0 the stabilizer is every
    spatial op and the umklapp phase is 1, so the covariant tile is the mean
    of the centroid-permuted tile over all ops (FINDINGS2; closed_window.py)."""
    nmu = T0.shape[0]
    acc = np.zeros((nmu, nmu), dtype=np.complex128)
    for s in range(ntran):
        a = alpha[s]
        acc += T0[np.ix_(a, a)]
    return acc / ntran


def _covariant_multiplets(ev_cov, tol_uev):
    """Contiguous degenerate groups (size > 1) of the covariant spectrum."""
    grp = [[0]]
    for i in range(1, len(ev_cov)):
        if (ev_cov[i] - ev_cov[i - 1]) * 1e6 > tol_uev:
            grp.append([i])
        else:
            grp[-1].append(i)
    return [g for g in grp if len(g) > 1]


@pytest.mark.gpu
def test_gamma_onsite_degeneracy(gnppm_session):
    """Γ-on-site BSE spectrum is invariant under little-group tile
    symmetrization (tiles covariant), and any covariant multiplet stays
    degenerate — the screening-window closure / tile-covariance guard."""
    harness.skip_unless_gpu(pytest)
    import h5py
    from bse import bse_io
    from file_io.wfn_loader import WfnLoader
    from centroid.orbit_syms import compute_centroid_sym_perm
    from gw.degen_average import (
        round_band_window_to_closed_shell, TOL_DEGENERACY_RY)

    run_dir = gnppm_session.run_dir
    inp = str(run_dir / gnppm_session.input_name)
    restart = bse_io._find_restart_file(inp)

    with h5py.File(restart, "r") as f:
        enk = np.asarray(f["enk_full"][:])          # (nk, nb) Ry
    nk, nb = enk.shape
    n_occ = bse_io.resolve_n_occ(enk, input_file=inp)

    # ---- auto-detect a degeneracy-closed (nv, nc) window at Γ (k-index 0) ----
    # Reuse the fix's helper with a single Γ energy row so min-over-k == the Γ
    # gap: raise the valence bottom, lower the conduction top to closed shells.
    e_g = enk[0:1, :]
    b_v = round_band_window_to_closed_shell(
        e_g, max(0, n_occ - _WINDOW_TARGET), TOL_DEGENERACY_RY, "up")
    b_c = round_band_window_to_closed_shell(
        e_g, min(n_occ + _WINDOW_TARGET, nb), TOL_DEGENERACY_RY, "down")
    nv = n_occ - b_v
    nc = b_c - n_occ
    assert nv > 0 and nc > 0, f"empty Γ window (nv={nv}, nc={nc})"
    # boundaries must be genuinely closed at Γ (construction sanity)
    if b_v > 0:
        assert float(enk[0, b_v] - enk[0, b_v - 1]) > TOL_DEGENERACY_RY
    if b_c < nb:
        assert float(enk[0, b_c] - enk[0, b_c - 1]) > TOL_DEGENERACY_RY

    # ---- production (head-injected) q=0 tiles for this window ----
    data = bse_io._load_ring_subset(
        restart, n_val=nv, n_cond=nc, px=1, py=1, input_file=inp)
    pc = np.asarray(data["psi_c"])[0]           # (c, s, μ) at Γ
    pv = np.asarray(data["psi_v"])[0]
    ec = np.asarray(data["eps_c"])[0]
    ev_ = np.asarray(data["eps_v"])[0]
    V0 = np.asarray(data["V_q0"])               # (μ, μ)
    W0 = np.asarray(data["W_q"])[:, :, 0, 0, 0]  # q=0 slice (μ, μ)

    # ---- little-group-symmetrized (exactly covariant) q=0 tiles ----
    wfn = WfnLoader(str(run_dir / "WFN.h5"), backend="eager")
    ntran = int(wfn.ntran)
    fft = np.asarray(wfn.fft_grid, dtype=np.int64)
    cfrac = np.loadtxt(str(run_dir / "centroids_frac_399.txt"))
    ridx = np.rint(cfrac * fft[None]).astype(np.int64) % fft[None]
    alpha, _ = compute_centroid_sym_perm(
        ridx, wfn.sym_matrices[:ntran], wfn.translations[:ntran], fft,
        validate=True)
    n_rmu_pad = V0.shape[0]
    # alpha permutes the n_rmu logical centroids; pad to the loaded μ extent
    # (identity on the zero pad rows) so it indexes the padded tiles.
    n_log = alpha.shape[1]
    if n_rmu_pad > n_log:
        pad = np.tile(np.arange(n_log, n_rmu_pad)[None, :], (alpha.shape[0], 1))
        alpha = np.concatenate([alpha, pad], axis=1)
    V0s = _symmetrize_q0(V0, alpha, ntran)
    W0s = _symmetrize_q0(W0, alpha, ntran)

    ev_raw = np.sort(np.linalg.eigvalsh(
        _build_gamma_H(pc, pv, ec, ev_, V0, W0))) * RY
    ev_cov = np.sort(np.linalg.eigvalsh(
        _build_gamma_H(pc, pv, ec, ev_, V0s, W0s))) * RY

    # ---- invariant 1: covariant tiles reproduce the production spectrum ----
    spectrum_shift_uev = float(np.max(np.abs(ev_raw - ev_cov)) * 1e6)
    assert spectrum_shift_uev < TIER_TIGHT_UEV, (
        f"Γ-on-site spectrum moved {spectrum_shift_uev:.3f} μeV under "
        f"little-group tile symmetrization (tiles non-covariant); "
        f"tier {TIER_TIGHT_UEV} μeV.  (nv={nv} nc={nc} ntran={ntran})")

    # ---- invariant 2: covariant multiplets stay degenerate ----
    for g in _covariant_multiplets(ev_cov, MULT_TOL_UEV):
        gi = np.array(g)
        cov_split = float((ev_cov[gi].max() - ev_cov[gi].min()) * 1e6)
        raw_split = float((ev_raw[gi].max() - ev_raw[gi].min()) * 1e6)
        assert cov_split < MULT_TOL_UEV, (
            f"covariant-tile multiplet (size {len(g)}) split {cov_split:.3f} "
            f"μeV > {MULT_TOL_UEV} μeV")
        assert raw_split < TIER_TIGHT_UEV, (
            f"raw-tile multiplet (size {len(g)}) split {raw_split:.3f} μeV > "
            f"tier {TIER_TIGHT_UEV} μeV")
