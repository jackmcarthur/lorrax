"""tdagw/td_system.py — the TD-aGW system of a GW restart bundle, without TDGW.

TDGW builds its header (energies, k grid, dipole) from a ``WFN.h5`` and an
``eqp.dat``.  This module builds the same header from the LORRAX restart
bundle that also supplies ΔΣ, in ONE :func:`tdagw.model.load_td_model` call,
so the energies, the occupations, the position operator and the ΔΣ model all
refer to the same ψ:

* **Energies.**  The window's energies of the loader (``model["window"]``):
  quasiparticle when an ``eqp_file`` is given — re-sliced by DFT index on a
  coarse deck exactly as ``bse_jax eqp_corrections`` does, the eigenvalues of the
  htransform's QP f(H) on a ``kgrid_fine`` deck — and DFT otherwise.
  Occupations by window position: the ``n_val`` valence rows are full.
* **Position operator.**  ``r_nm = v_nm / (i (E_n − E_m))`` from the bundle's
  own velocity ``v_nm = ⟨n|v̂|m⟩`` (``v̂ = i[H, r̂]``), with the velocity's own
  (DFT) energies: the DFT-velocity approximation, under which r is
  scissor-invariant, the convention of the BSE dipoles
  (``absorption_common.slice_dipole_to_bse_window``).  Coarse deck: the
  coarse WFN's ``dipole.h5`` sliced over the WHOLE window (vv', cc' and cv);
  ``kgrid_fine`` deck: the velocity carried through the htransform into the
  fine states (``bse_densify.densified_dipoles(full_window=True)``).
* **Pairs.**  ``n = m`` is excluded: a velocity gives no diagonal position, so
  the system has no intraband commutator ``E·[r^intra, ρ]`` — TDGW's
  ``use_vmtxel`` route (dipole ``itype`` 1), not its finite-difference route
  (``itype`` 2, ``TDGW_RINTRA_IN_EOM``).  A c↔v pair must have a positive gap
  above ``gap``; same-occupation pairs (vv', cc') follow ``pair_rule``:
  ``mask`` (default) multiplies r by ``1 − exp(−ΔE²/(2 gap²))``, the energy
  scale TDGW's finite-difference route uses (``esmear``) to hand
  near-degenerate pairs to the intraband term, which this route does not
  have; ``cut`` zeroes pairs with ``|ΔE| < gap``; ``cv`` keeps the c↔v blocks
  only, the structure of TDGW's ``use_vmtxel`` header.

The header has the layout of :func:`tdagw.tdgw_io.read_header` (dipole
``itype`` 1), so :func:`tdagw.tdgw_io.header_operators` and the output writer
take it unchanged.  :func:`bse_dipole_receipt` is the gauge check: the BSE's
own ``⟨0|r̂|S⟩`` (``eigenvectors.h5``) reproduced from this header's r and the
BSE eigenvectors.
"""
from __future__ import annotations

import os
from typing import NamedTuple, Optional

import numpy as np

from common.units import RYD_TO_EV

__all__ = ["PAIR_RULES", "TdBundleSystem", "position_from_velocity", "position_from_window_r",
           "window_energies", "bundle_header", "exciton_dipoles",
           "bse_dipole_receipt", "build_td_system"]

PAIR_RULES = ("mask", "cut", "cv")
#: A velocity whose anti-Hermitian part exceeds this (relative) refuses.
VELOCITY_HERMITIAN_RTOL = 1e-8
#: |E_n - E_m| below this (Ry) is an exact degeneracy: no r at all.
EXACT_DEGENERACY_RY = 1e-12


class TdBundleSystem(NamedTuple):
    """``hdr`` (tdgw_io.read_header layout), ``model`` (load_td_model), receipts."""
    hdr: dict
    model: dict
    receipts: dict


def _gate(gid, got, want, fix):
    return ValueError(f"GATE {gid}: got: {got}; want: {want}; fix: {fix}")


# ---------------------------------------------------------------------------
# Pure pieces (host numpy)
# ---------------------------------------------------------------------------

def position_from_velocity(v, e_ry, n_val: int, *, rule: str = "mask",
                           gap_ry: float) -> tuple[np.ndarray, dict]:
    """``r_nm = v_nm / (i (E_n − E_m))`` over every window pair.

    Parameters
    ----------
    v : (3, nk, nb, nb) complex — ``⟨nk|v̂_α|mk⟩`` (Ry·bohr), bra n, ket m.
    e_ry : (nk, nb) float — the velocity's own band energies (Ry).
    n_val : int — the leading occupied rows.
    rule : ``mask`` | ``cut`` | ``cv`` — same-occupation pairs (module doc).
    gap_ry : float — the energy scale of ``rule`` and the smallest c↔v gap.

    Returns
    -------
    r : (3, nk, nb, nb) complex — ``⟨nk|r̂_α|mk⟩`` (bohr), Hermitian, zero
        diagonal.
    receipt : dict — pair counts and gaps (eV) per block.
    """
    if rule not in PAIR_RULES:
        raise _gate("td_pair_rule", repr(rule), " | ".join(PAIR_RULES),
                    "set [lorrax] pair_rule")
    v = np.asarray(v, dtype=np.complex128)
    e = np.asarray(e_ry, dtype=np.float64)
    nk, nb = e.shape
    if v.shape != (3, nk, nb, nb):
        raise _gate("td_velocity_shape", v.shape, (3, nk, nb, nb),
                    "pass the velocity of the window the energies describe")
    if not 0 < n_val < nb:
        raise _gate("td_window_occupation", f"n_val = {n_val} of {nb} bands",
                    "0 < n_val < nb", "give a window that straddles the gap")
    scale = float(np.abs(v).max()) or 1.0
    vh = np.conj(np.swapaxes(v, -1, -2))
    herm = float(np.abs(v - vh).max()) / scale
    if herm > VELOCITY_HERMITIAN_RTOL:
        raise _gate("td_velocity_hermitian", f"max|v - v^H|/max|v| = {herm:.3e}",
                    f"<= {VELOCITY_HERMITIAN_RTOL:.0e}",
                    "the velocity file is not a Hermitian band operator")
    v = 0.5 * (v + vh)
    de = e[:, :, None] - e[:, None, :]                  # (nk, n, m) = E_n - E_m
    occ = np.arange(nb) < n_val
    cv = occ[:, None] != occ[None, :]                   # (nb, nb) c<->v pairs
    off = ~np.eye(nb, dtype=bool)
    same = off & ~cv
    de_cv = de[:, n_val:, :n_val]                       # E_c - E_v
    min_cv = float(de_cv.min())
    if not min_cv > gap_ry:
        raise _gate("td_cv_gap", f"min (E_c - E_v) = {min_cv * RYD_TO_EV:.6g} eV in "
                    "the velocity's energies", f"> {gap_ry * RYD_TO_EV:.3g} eV "
                    "at every k", "r = v/(E_c - E_v) is undefined at a band inversion; "
                    "use a window whose conduction rows lie above its valence rows")
    exact = np.abs(de) < EXACT_DEGENERACY_RY
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((off[None] & ~exact)[None], v / (1j * de[None]), 0.0)
    near = same[None] & (np.abs(de) < gap_ry)           # (nk, nb, nb)
    if rule == "mask":
        factor = np.where(same[None], -np.expm1(-de ** 2 / (2.0 * gap_ry ** 2)), 1.0)
        r = r * factor[None]
    elif rule == "cut":
        r = np.where(near[None], 0.0, r)
    else:                                               # "cv": TDGW's vmtxel layout
        r = np.where(same[None, None], 0.0, r)
    r[:, :, np.arange(nb), np.arange(nb)] = 0.0
    r = 0.5 * (r + np.conj(np.swapaxes(r, -1, -2)))

    def _block(mask):
        sel = np.triu(mask, 1)
        gaps = np.abs(de)[:, sel]
        return dict(pairs=int(sel.sum()) * nk,
                    below_gap=int((gaps < gap_ry).sum()),
                    exact=int((gaps < EXACT_DEGENERACY_RY).sum()),
                    min_gap_ev=float(gaps.min() * RYD_TO_EV) if gaps.size else None)

    vv = np.zeros((nb, nb), bool)
    vv[:n_val, :n_val] = True
    cc = np.zeros((nb, nb), bool)
    cc[n_val:, n_val:] = True
    receipt = dict(rule=rule, gap_ev=float(gap_ry * RYD_TO_EV),
                   velocity_hermitian_dev=herm, min_cv_gap_ev=min_cv * RYD_TO_EV,
                   vv=_block(vv & off), cc=_block(cc & off),
                   cv_pairs=int(nk * n_val * (nb - n_val)),
                   max_abs_r_bohr=float(np.abs(r).max()))
    return r, receipt


#: the gap of the mask Route B applies to r_window (bse_densify, pair rule mask)
ROUTE_B_MASK_GAP_EV = 0.05


def position_from_window_r(r, e_ry, n_val: int, *, rule: str = "mask",
                           gap_ry: float) -> tuple[np.ndarray, dict]:
    """The window position Route B supplies (``r_window``), with this module's rules.

    ``r`` (3, nk, nb, nb) is ``<i|r|j>`` (bohr) in the fine states, already
    masked by Route B's own ``mask`` rule on same-occupation pairs; ``cut``
    and ``cv`` are applied on top of it here, ``mask`` is not applied twice.
    ``e_ry`` are the velocity's band energies (receipt counts and the c-v
    gate only; no division).
    """
    if rule not in PAIR_RULES:
        raise _gate("td_pair_rule", repr(rule), " | ".join(PAIR_RULES), "set [lorrax] pair_rule")
    if rule == "mask" and abs(gap_ry * RYD_TO_EV - ROUTE_B_MASK_GAP_EV) > 1e-9:
        raise _gate("td_pair_gap_route_b", f"pair rule mask at {gap_ry * RYD_TO_EV:g} eV",
                    f"Route B's mask scale {ROUTE_B_MASK_GAP_EV:g} eV (applied upstream)",
                    "set pair_gap_ev = 0.05, or pair_rule = 'cut' | 'cv'")
    r = np.asarray(r, dtype=np.complex128)
    e = np.asarray(e_ry, dtype=np.float64)
    nk, nb = e.shape
    if r.shape != (3, nk, nb, nb):
        raise _gate("td_window_position_shape", r.shape, (3, nk, nb, nb),
                    "pass Route B's r_window of the window the energies describe")
    scale = float(np.abs(r).max()) or 1.0
    rh = np.conj(np.swapaxes(r, -1, -2))
    herm = float(np.abs(r - rh).max()) / scale
    if herm > VELOCITY_HERMITIAN_RTOL:
        raise _gate("td_position_hermitian", f"max|r - r^H|/max|r| = {herm:.3e}",
                    f"<= {VELOCITY_HERMITIAN_RTOL:.0e}", "r_window is not a Hermitian band operator")
    r = 0.5 * (r + rh)
    de = e[:, :, None] - e[:, None, :]
    occ = np.arange(nb) < n_val
    cv = occ[:, None] != occ[None, :]
    off = ~np.eye(nb, dtype=bool)
    same = off & ~cv
    min_cv = float(de[:, n_val:, :n_val].min())
    if not min_cv > gap_ry:
        raise _gate("td_cv_gap", f"min (E_c - E_v) = {min_cv * RYD_TO_EV:.6g} eV",
                    f"> {gap_ry * RYD_TO_EV:.3g} eV at every k",
                    "use a window whose conduction rows lie above its valence rows")
    if rule == "cut":
        r = np.where((same[None] & (np.abs(de) < gap_ry))[None], 0.0, r)
    elif rule == "cv":
        r = np.where(same[None, None], 0.0, r)
    r[:, :, np.arange(nb), np.arange(nb)] = 0.0

    def _block(mask):
        sel = np.triu(mask, 1)
        gaps = np.abs(de)[:, sel]
        return dict(pairs=int(sel.sum()) * nk, below_gap=int((gaps < gap_ry).sum()),
                    exact=int((gaps < EXACT_DEGENERACY_RY).sum()),
                    min_gap_ev=float(gaps.min() * RYD_TO_EV) if gaps.size else None)

    vv = np.zeros((nb, nb), bool); vv[:n_val, :n_val] = True
    cc = np.zeros((nb, nb), bool); cc[n_val:, n_val:] = True
    receipt = dict(rule=rule + (" (Route B mask upstream)" if rule == "mask" else " on Route B's masked r"),
                   gap_ev=float(gap_ry * RYD_TO_EV), position_hermitian_dev=herm,
                   min_cv_gap_ev=min_cv * RYD_TO_EV, vv=_block(vv & off), cc=_block(cc & off),
                   cv_pairs=int(nk * n_val * (nb - n_val)), max_abs_r_bohr=float(np.abs(r).max()),
                   source="r_window (exact position of the fine states)")
    return r, receipt


def window_energies(window: dict) -> tuple[np.ndarray, int]:
    """``(e (nk, nb) Ry, n_val)`` of ``load_td_model``'s ``window`` (pads stripped)."""
    from common.collectives import gather_to_host
    nv, nc = int(window["n_val"]), int(window["n_cond"])
    ev = np.asarray(gather_to_host(window["eps_v"]))[:, :nv]
    ec = np.asarray(gather_to_host(window["eps_c"]))[:, :nc]
    return np.concatenate([ev, ec], axis=1).astype(np.float64), nv


def _structure(input_file: str) -> dict:
    """rprim (bohr, rows), atoms, FFT grid, cell volume from the deck's WFN header."""
    import h5py
    from gw.gw_config import LorraxConfig
    path = LorraxConfig.from_input_file(input_file).paths.wfn_file
    with h5py.File(path, "r") as f:
        cr = f["mf_header/crystal"]
        alat = float(cr["alat"][()])
        avec = np.asarray(cr["avec"][()], dtype=np.float64)
        apos = np.asarray(cr["apos"][()], dtype=np.float64)
        return dict(wfn=os.path.abspath(path), rprim=alat * avec,
                    atomic_numbers=np.asarray(cr["atyp"][()], dtype=np.int64),
                    atomic_positions_red=apos @ np.linalg.inv(avec),
                    fftgrid=np.asarray(f["mf_header/gspace/FFTgrid"][()], np.int32),
                    cell_volume=float(cr["celvol"][()]))


def bundle_header(*, structure: dict, k_frac, kgrid, bands: tuple, nspinor: int,
                  e_qp_ry, n_val: int, r) -> dict:
    """The :func:`tdagw.tdgw_io.read_header` dict of a bundle system (itype 1).

    ``bands`` is the absolute 0-based window ``[b_lo, b_hi)``; the header keeps
    TDGW's 1-based ``(start, stop, step)``.  Energies in Ha by way of eV with
    TDGW's Hartree (``tdgw_io.HA_IN_EV``): the eqp files are in eV and LORRAX
    reads them with BerkeleyGW's Rydberg, so Ry -> eV -> Ha reproduces TDGW's
    own ``e_kn`` (halving would leave the 0.36 ppm between the two
    constants).  Occupations by window position, uniform k weights.
    """
    from tdagw.tdgw_io import HA_IN_EV
    b_lo, b_hi = (int(b) for b in bands)
    e = np.asarray(e_qp_ry, dtype=np.float64) * RYD_TO_EV / HA_IN_EV
    nk, nb = e.shape
    f = np.zeros((nk, nb))
    f[:, :n_val] = 1.0
    return dict(path=None, rprim=np.asarray(structure["rprim"]),
                atomic_numbers=np.asarray(structure["atomic_numbers"]),
                atomic_positions_red=np.asarray(structure["atomic_positions_red"]),
                kpts=np.asarray(k_frac, dtype=np.float64),
                kgrid=np.asarray(kgrid, dtype=np.int64), kshift=np.zeros(3),
                bands=(b_lo + 1, b_hi + 1, 1), nspinor=int(nspinor),
                fftgrid=np.asarray(structure["fftgrid"]), e_kn=e, f_kn=f,
                w_k=np.full(nk, 1.0 / nk), itype=1,
                rinter=np.asarray(r, dtype=np.complex128), rintra=None,
                bvecs=None, bvecs_weight=None, is_patched=False, olp=None)


def exciton_dipoles(r, A, n_val: int) -> np.ndarray:
    """``⟨0|r̂_α|S⟩ = Σ_{kcv} A^S_{kcv} ⟨vk|r̂_α|ck⟩``, (N, 3).

    ``A`` is ``(N, nk, nc, nv)`` (``absorption_common.load_eigenvectors_h5``:
    valence lowest first, the window's own order); ``r`` the header's.
    """
    r = np.asarray(r)
    nc, nv = int(A.shape[2]), int(A.shape[3])
    if nv != n_val or r.shape[-1] != nv + nc:
        raise _gate("td_projection_window", f"eigenvectors {nc}c x {nv}v",
                    f"the TD window {r.shape[-1] - n_val}c x {n_val}v",
                    "solve the BSE on the TD window")
    r_vc = r[:, :, :n_val, n_val:]                      # (3, nk, nv, nc)
    return np.einsum("skcv,xkvc->sx", np.asarray(A), r_vc)


def eigenvector_kpts(params: dict, nk: int) -> np.ndarray:
    """The k points of a ``load_eigenvectors_h5`` params dict as (nk, 3).

    ``bse_jax`` writes ``exciton_header/kpoints/kpts`` in BerkeleyGW's layout,
    (nk, 3) as h5py reads it; (3, nk), the layout files written before
    2026-10-22 carry, is accepted too.
    """
    kp = np.asarray(params["kpts"], dtype=np.float64)
    if kp.shape == (3, nk):
        return kp.T.copy()
    if kp.shape == (nk, 3):
        return kp
    raise _gate("td_eigenvector_kpts", f"kpts of shape {kp.shape}", f"(3, {nk}) or ({nk}, 3)",
                "pass a bse_jax write_eigenvectors file of the same k grid")


def bse_dipole_receipt(r, eigenvectors_file: str, n_val: int, kpts, *,
                       n_states: Optional[int] = None) -> dict:
    """The BSE's stored ``⟨0|r̂|S⟩`` reproduced from this header's r.

    ``bse_jax dipole_file`` stores ``Σ_t A^S_t conj(d_t)`` with
    ``d = ⟨c|v̂|v⟩/ΔE = i ⟨c|r̂|v⟩``, i.e. ``−i ⟨0|r̂|S⟩``; the receipt compares
    :func:`exciton_dipoles` with ``i × stored`` and the in-plane helicity
    strengths ``I± = |D_x ∓ i D_y|²/2`` of both.
    """
    from bse.absorption_common import load_eigenvectors_h5, load_exciton_dipoles_h5
    _, A, params = load_eigenvectors_h5(eigenvectors_file)
    _, stored = load_exciton_dipoles_h5(eigenvectors_file)
    n = A.shape[0] if n_states is None else min(int(n_states), A.shape[0])
    kpts = np.asarray(kpts, dtype=np.float64)
    dk = eigenvector_kpts(params, kpts.shape[0]) - kpts
    if dk.shape != (A.shape[1], 3) or np.abs(dk - np.rint(dk)).max() > 1e-6:
        raise _gate("td_projection_kpoints", "eigenvector k list differs from the header's",
                    "the same k points in the same order",
                    "solve the BSE on the bundle the TD system is built from")
    mine = exciton_dipoles(r, A[:n], n_val)
    ref = 1j * stored[:n]
    i_plus = lambda d: 0.5 * np.abs(d[:, 0] - 1j * d[:, 1]) ** 2   # noqa: E731
    i_minus = lambda d: 0.5 * np.abs(d[:, 0] + 1j * d[:, 1]) ** 2  # noqa: E731
    scale = float(np.abs(ref).max()) or 1.0
    return dict(n_states=int(n), eigenvectors=os.path.abspath(eigenvectors_file),
                dipole_rel_dev=float(np.abs(mine - ref).max() / scale),
                i_plus_rel_dev=float(np.abs(i_plus(mine) - i_plus(ref)).max()
                                     / max(float(i_plus(ref).max()), 1e-300)),
                i_minus_rel_dev=float(np.abs(i_minus(mine) - i_minus(ref)).max()
                                      / max(float(i_minus(ref).max()), 1e-300)))


# ---------------------------------------------------------------------------
# The velocity of either route
# ---------------------------------------------------------------------------

def _coarse_velocity(model: dict, input_file: str, dipole_file: str, *, bands,
                     e_dft_ry) -> tuple[np.ndarray, np.ndarray, dict]:
    """The coarse WFN's dipole.h5 over the whole window, checked against the restart."""
    from bse.bse_window import _parse_wfn_path
    from file_io.dipole import require_dipole_k_grid
    from file_io.restart_bundle import load_dipole_h5, read_dipole_metadata
    b_lo, b_hi = bands
    stored = int(read_dipole_metadata(dipole_file)["stored_bands"])
    if stored < b_hi:
        raise _gate("td_dipole_bands", f"{dipole_file} has {stored} bands",
                    f">= {b_hi}", "rerun psp.get_dipole_mtxels with more bands")
    dipole_cart, delta_e, attrs = load_dipole_h5(
        dipole_file, wfn_path=_parse_wfn_path(input_file),
        mesh_xy=model["psi_X"].sharding.mesh,
        band_rows=(b_lo, b_hi), band_cols=(b_lo, b_hi))
    nk = int(np.prod(model["kgrid"]))
    require_dipole_k_grid(attrs, nk, dipole_path=dipole_file)
    # The TD position model is explicitly host-owned over its active window;
    # this is a bounded-window handoff, never a read of the full file's bands.
    from common.collectives import gather_to_host
    nb = b_hi - b_lo
    v = np.asarray(gather_to_host(dipole_cart))[:, :, :nb, :nb]
    de_file = np.asarray(gather_to_host(delta_e))[:, :nb, :nb]
    de_restart = e_dft_ry[:, :, None] - e_dft_ry[:, None, :]
    dev = float(np.abs(de_file - de_restart).max())
    if dev > 1e-8:
        raise _gate("td_dipole_energies", f"max |dE(dipole.h5) - dE(restart)| = "
                    f"{dev * RYD_TO_EV:.3e} eV", "the restart's DFT energies",
                    "build dipole.h5 from the restart's WFN")
    return v, e_dft_ry, dict(route="coarse dipole.h5 (whole window)",
                             source=os.path.abspath(dipole_file),
                             velocity=str(attrs.get("operator", attrs.get("arm", ""))),
                             energies_dev_ev=dev * RYD_TO_EV)


def _densified_velocity(model: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """``window["dipoles"]`` of a kgrid_fine bundle: fine velocity + its DFT energies."""
    from tdagw.model import flat_k_rows
    dip = model["window"]["dipoles"]
    if dip is None or "v_window" not in dip:
        raise _gate("td_densified_velocity", "a kgrid_fine window without v_window",
                    "the velocity over every window pair",
                    "load with velocity_file and velocity_full_window=True")
    rows = np.asarray(flat_k_rows(np.asarray(dip["q_frac"]), model["kgrid"]))
    if not np.array_equal(rows, np.arange(rows.size)):
        raise _gate("td_densified_k_order", "the fine q list is not the C-order grid",
                    "the model's flat k rows", "report it: a bookkeeping defect")
    r_win = dip.get("r_window")
    return (np.asarray(dip["v_window"]), np.asarray(dip["e_dft_window"], np.float64),
            dict(route=dip["route"] + ", full window" + (", exact r_window" if r_win is not None else ""),
                 source=dip["source"], velocity=dip["velocity"], on_grid_rel=dip["on_grid_rel"]),
            None if r_win is None else np.asarray(r_win))


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------

def build_td_system(input_file: str, mesh_xy, *, bands, dipole_file: str,
                    eqp_file: Optional[str] = None, n_occ: Optional[int] = None,
                    pair_rule: str = "mask", gap_ev: float = 0.05,
                    zeta_file: Optional[str] = None,
                    degeneracy_mode: Optional[str] = None,
                    eigenvectors_file: Optional[str] = None,
                    position_fix: bool = False,
                    log=print) -> TdBundleSystem:
    """Header + ΔΣ model of a GW restart bundle from one ``load_td_model`` call.

    Parameters
    ----------
    input_file : the GW deck of the restart (a ``kgrid_fine`` deck densifies).
    bands : (b_lo, b_hi) — absolute 0-based TD window, straddling ``n_occ``.
    dipole_file : the coarse WFN's ``dipole.h5`` (``psp.get_dipole_mtxels``).
    eqp_file : the BSE run's ``eqp1.dat`` (QP energies), or None (DFT).
    pair_rule, gap_ev : same-occupation pairs (module doc); ``gap_ev`` is also
        the smallest accepted c↔v gap.
    eigenvectors_file : a ``bse_jax write_eigenvectors and dipole_file`` file on the same
        bundle and window: adds the ⟨0|r|S⟩ gauge receipt.
    position_fix : densified deck with ``eqp_file``: the loader's opt-in exact
        position of the fine QP states (``lorrax.dipole_position_fix``,
        ``bse_jax dipole_position_fix``); off, ``v/ΔE_RQ``.
    """
    from file_io.restart_bundle import _find_restart_file, read_metadata
    from tdagw.model import _deck_densifies, load_td_model
    b_lo, b_hi = (int(b) for b in bands)
    restart = _find_restart_file(input_file)
    densified = _deck_densifies(input_file, read_metadata(restart))
    model = load_td_model(
        input_file, mesh_xy, bands=(b_lo, b_hi), n_occ=n_occ, zeta_file=zeta_file,
        degeneracy_mode=degeneracy_mode, eqp_file=eqp_file,
        velocity_file=dipole_file if densified else None,
        velocity_full_window=densified, velocity_position_fix=bool(position_fix),
        keep_window=True)
    window = model["window"]
    e_qp, n_val = window_energies(window)
    if densified:
        v, e_dft, vrec, r_win = _densified_velocity(model)
    else:
        enk = np.asarray(read_metadata(restart)["energies"], dtype=np.float64)
        r_win = None
        v, e_dft, vrec = _coarse_velocity(model, input_file, dipole_file,
                                          bands=(b_lo, b_hi), e_dft_ry=enk[:, b_lo:b_hi])
    if r_win is not None:                    # Route B's exact position (QP-mixed fine states)
        r, prec = position_from_window_r(r_win, e_dft, n_val, rule=pair_rule,
                                         gap_ry=float(gap_ev) / RYD_TO_EV)
    else:
        r, prec = position_from_velocity(v, e_dft, n_val, rule=pair_rule,
                                         gap_ry=float(gap_ev) / RYD_TO_EV)
    structure = _structure(input_file)
    hdr = bundle_header(structure=structure, k_frac=model["k_frac"], kgrid=model["kgrid"],
                        bands=(b_lo, b_hi), nspinor=int(model["nspinor"]),
                        e_qp_ry=e_qp, n_val=n_val, r=r)
    receipts = dict(input_file=os.path.abspath(input_file), bands=(b_lo, b_hi),
                    n_val=n_val, n_cond=int(window["n_cond"]), n_occ=int(window["n_occ"]),
                    densified=bool(window["densified"]),
                    energy_source=window["energy_source"], eqp_file=window["eqp_file"],
                    velocity=vrec, pairs=prec, wfn=structure["wfn"],
                    cell_volume_bohr3=structure["cell_volume"],
                    qp_minus_dft_gap_ev=float((e_qp[:, n_val:].min() - e_qp[:, :n_val].max()
                                               - e_dft[:, n_val:].min() + e_dft[:, :n_val].max())
                                              * RYD_TO_EV))
    if eigenvectors_file:
        receipts["bse_dipoles"] = bse_dipole_receipt(r, eigenvectors_file, n_val, hdr["kpts"])
    hdr["td_system"] = receipts
    log(f"[td_system] {'kgrid_fine' if densified else 'coarse'} bundle {input_file}: "
        f"{hdr['kpts'].shape[0]} k x {e_qp.shape[1]} bands [{b_lo},{b_hi}) "
        f"({n_val}v), energies {window['energy_source']}, r from {vrec['route']}, "
        f"pair rule {pair_rule} at {gap_ev:g} eV (vv' below: {prec['vv']['below_gap']}, "
        f"cc' below: {prec['cc']['below_gap']}), min c-v gap {prec['min_cv_gap_ev']:.4f} eV")
    if "bse_dipoles" in receipts:
        b = receipts["bse_dipoles"]
        log(f"[td_system] BSE <0|r|S> from header + eigenvectors ({b['n_states']} states): "
            f"rel dev {b['dipole_rel_dev']:.2e}, I+ {b['i_plus_rel_dev']:.2e}, "
            f"I- {b['i_minus_rel_dev']:.2e}")
    return TdBundleSystem(hdr=hdr, model=model, receipts=receipts)
