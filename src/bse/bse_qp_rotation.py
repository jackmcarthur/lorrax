"""The quasiparticle-rotated BSE transition basis (``bse_jax --qp-rotations``).

A GW run diagonalizes its Hermitian effective Hamiltonian in the Σ window
``[q0, q1)`` of the deck WFN's DFT bands,

    H_QP(k) = U(k) diag(E_QP(k)) U(k)^H,        U_mn(k) = <m_DFT k | n_QP k>,   (1)

and stores the pair in ``qp_wfn_rotations.h5``
(:func:`file_io.qp_wfn.write_qp_rotations_h5`).  The BSE in that basis changes
three things and nothing else.

* The states at the ISDF centroids,

      ψ'_nk(μ) = Σ_{m=q0}^{q1-1} ψ_mk(μ) U_mn(k),                                (2)

  over the COMPLETE block.  A sliced eigenvector matrix is not unitary, so
  the block must lie inside the restart's ζ-training window
  (``GATE bse_qp_rotation_block``), the rule the htransform consumer applies
  (:func:`bandstructure.fh_interp.resolve_qp_hamiltonian_state`).
* The transition energies, ``ε_nk = E_QP(n, k)``.
* The dipoles, :func:`qp_window_dipoles`.

``V_μν`` and ``W_μν`` live in the centroid basis and do not depend on the band
basis, so the restart's V and W0 are used unchanged.

The window is chosen in QP order: ``--n-val``/``--n-cond`` count states of the
artifact's ascending ladder below/above ``n_occ``, and the degeneracy guard
runs on that ladder.  A QP state of the window may be dominated by a DFT band
outside the DFT-index window of the same counts; (2) runs over the whole
block, so that is physics, not truncation.  The DFT-index leakage
``1 - Σ_{m∈[lo,hi)} |U_mn|²`` is printed to show how far the two windows
differ.

THE STAR GAUGE.  :func:`file_io.restart_bundle.read_qp_rotations_artifact`
unfolds a wedge-stored U through ``symmetry_maps.star_broadcast``: for a
full-BZ ``k = g p`` of the WFN parent ``p``, ``U(k) = U(p)`` when g is unitary
and ``U(p)*`` when g is antiunitary.  That holds in the restart's gauge only if
the restart's ψ obeys ``ψ_k = g ψ_p``, so that ``H_QP(k)`` in the ψ_k basis is
``H_QP(p)`` or ``H_QP(p)*``.  :func:`certify_star_gauge` measures this on the
restart's own ψ, with the band matrices of 1 and σ_z sampled at the centroids,

    S_k,mn = Σ_{μ,s} ψ*_mks(μ) ψ_nks(μ),     S^z_k,mn = Σ_{μ,s} σ_s ψ*_mks(μ) ψ_nks(μ),

which must equal ``S_p`` (unitary g) or ``S_p*`` (antiunitary g); the opposite
rule's residual is printed as the check's sensitivity.  The artifact's own
``H_QP(k)`` must obey the same rule (gauge-invariant, so it holds for wedge
storage and for per-k diagonalizations alike).  A restart that stores ψ on
every full-BZ k makes no star claim at the centroids and skips the ψ check.
"""
from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.band_degeneracy import resolve_band_window
from common.units import RYD_TO_EV
from common.log import get_logger

log = get_logger(__name__)

#: Relative Frobenius tolerance of the star relation on the artifact's
#: ``H_QP(k)`` and on the restart's centroid band matrices.  Both are exact
#: transformations of stored float64 data (a gather, a conjugation, a centroid
#: permutation and a spinor rotation), so the residual is rounding, ~1e-15.
STAR_RELATION_RTOL = 1.0e-10


def star_parents(irr_idx_k, sym_idx_k, n_sym_spatial, kirr_fullids):
    """Full-BZ parent row and antiunitary flag of every full-BZ k.

    Parameters
    ----------
    irr_idx_k : (nk,) int — file-wedge (WFN IBZ) index of each full-BZ k.
    sym_idx_k : (nk,) int — TRS-augmented symmetry row mapping that parent
        to k; rows ``>= n_sym_spatial`` are antiunitary.
    n_sym_spatial : int
    kirr_fullids : (nk_red,) int — full-BZ row of each WFN IBZ point.

    Returns
    -------
    parent : (nk,) int — full-BZ row ``p`` with ``k = g p``.
    antiunitary : (nk,) bool — ``g`` carries complex conjugation.

    This is the ``trs_reference="ibz_slab"`` rule of
    ``symmetry_maps.star_broadcast``: the stored row is the untransformed WFN
    state, so the predicate is the member's own symmetry row.
    """
    irr = np.asarray(irr_idx_k, dtype=np.int64)
    parent = np.asarray(kirr_fullids, dtype=np.int64)[irr]
    antiunitary = np.asarray(sym_idx_k, dtype=np.int64) >= int(n_sym_spatial)
    return parent, antiunitary


def star_relation_residuals(M, parent, antiunitary):
    """Relative residuals of ``M_k = M_p`` (unitary) / ``M_p*`` (antiunitary).

    Parameters
    ----------
    M : (nk, nb, nb) complex — a k-diagonal band matrix on the full BZ.
    parent, antiunitary : from :func:`star_parents`.

    Returns
    -------
    (residual, residual_other) : two (nk,) float arrays,
        ``||M_k - R(M_p)||_F / ||M_p||_F`` under the stated rule ``R`` and
        under the opposite rule.  The second is the check's sensitivity: a
        rule the data cannot tell apart from its opposite certifies nothing.
    """
    M = np.asarray(M)
    Mp = M[np.asarray(parent)]
    conj = np.asarray(antiunitary, dtype=bool)[:, None, None]
    stated = np.where(conj, np.conj(Mp), Mp)
    other = np.where(conj, Mp, np.conj(Mp))
    norm = np.maximum(np.linalg.norm(Mp, axis=(1, 2)), np.finfo(float).tiny)
    return (np.linalg.norm(M - stated, axis=(1, 2)) / norm,
            np.linalg.norm(M - other, axis=(1, 2)) / norm)


def qp_window_from_artifact(U, E, band_range, *, n_occ, n_val, n_cond,
                            degeneracy_mode, degeneracy_tol_ry,
                            log=log.info):
    """The BSE window in QP order and its rotation columns.

    Parameters
    ----------
    U : (nk, nb, nb) complex — full-BZ ``U_mn(k)``, eq. (1), ``nb = q1 - q0``.
    E : (nk, nb) float — the matched QP energies (Ry), ascending at each k.
    band_range : (q0, q1) — absolute DFT band range of the block.
    n_occ : int — occupied count; QP states ``[q0, n_occ)`` are occupied.
    n_val, n_cond : int — requested QP valence/conduction counts.
    degeneracy_mode, degeneracy_tol_ry : the ``--band-degeneracy`` guard,
        applied to the QP ladder (``common.band_degeneracy``).

    Returns
    -------
    dict with ``n_val``, ``n_cond`` (after the guard), ``U_val`` (nk, nb, nv),
    ``U_cond`` (nk, nb, nc), ``eps_v`` (nk, nv), ``eps_c`` (nk, nc) in Ry, and
    ``diagnostics``: the DFT-index leakage of the window, the cross-gap mixing
    ``θ = max_k ||U_VC(k)||_F`` (DFT valence rows, QP conduction columns of the
    window) and its transpose, and the largest single-state cross-gap weight.
    """
    q0, q1 = (int(b) for b in band_range)
    n_occ, n_val, n_cond = int(n_occ), int(n_val), int(n_cond)
    U = np.asarray(U)
    E = np.asarray(E, dtype=np.float64)
    nb = q1 - q0
    if U.shape[1:] != (nb, nb) or E.shape != (U.shape[0], nb):
        raise ValueError(
            f"qp_window_from_artifact: U {U.shape} / E {E.shape} do not "
            f"describe the block [{q0}, {q1}).")
    lo, hi = n_occ - n_val, n_occ + n_cond
    if not (q0 <= lo and hi <= q1 and q0 < n_occ < q1):
        raise ValueError(
            f"GATE bse_qp_window_outside_block: got BSE window [{lo}, {hi}) "
            f"(n_occ {n_occ}); want it inside the QP rotation block "
            f"[{q0}, {q1}) with the occupied boundary strictly inside.  why: "
            f"outside the block there are no QP states, only DFT ones.  fix: "
            f"narrow --n-val/--n-cond, or rerun the GW with a wider Σ window.")
    n_val, n_cond = resolve_band_window(
        E, n_occ - q0, n_val, n_cond, tol_ry=degeneracy_tol_ry,
        mode=degeneracy_mode, where="bse --qp-rotations (QP ladder)", log=log)
    lo, hi = n_occ - n_val, n_occ + n_cond
    if not (q0 <= lo and hi <= q1):
        raise ValueError(
            f"GATE bse_qp_window_outside_block: the degeneracy guard widened "
            f"the window to [{lo}, {hi}), outside the QP block [{q0}, {q1}).")
    if hi == q1:
        log(f"  [qp-rotations] the conduction edge {hi} is the Σ-window edge: "
            f"closure against bands above the QP block cannot be checked.")
    val = np.arange(lo, n_occ) - q0
    cond = np.arange(n_occ, hi) - q0
    weight = np.abs(U) ** 2
    window = np.concatenate([val, cond])
    leak = 1.0 - weight[:, lo - q0:hi - q0, :][:, :, window].sum(axis=1)
    k_leak, i_leak = np.unravel_index(int(np.argmax(leak)), leak.shape)
    split = n_occ - q0
    theta_vc = np.linalg.norm(U[:, :split, :][:, :, cond], axis=(1, 2))
    theta_cv = np.linalg.norm(U[:, split:, :][:, :, val], axis=(1, 2))
    cross = np.concatenate([weight[:, :split, :][:, :, cond].sum(axis=1),
                            weight[:, split:, :][:, :, val].sum(axis=1)],
                           axis=1)
    return {
        "n_val": n_val, "n_cond": n_cond,
        "U_val": U[:, :, val], "U_cond": U[:, :, cond],
        "eps_v": E[:, val], "eps_c": E[:, cond],
        "diagnostics": {
            "dft_window_leakage_max": float(leak.max()),
            "dft_window_leakage_at": (int(k_leak), int(window[i_leak] + q0)),
            "theta_vc_max": float(theta_vc.max()),
            "theta_cv_max": float(theta_cv.max()),
            "cross_gap_weight_max": float(cross.max()),
        },
    }


def certify_star_gauge(S, Sz, parent, antiunitary, *, where, log=log.info):
    """Refuse a restart ψ whose star relation is not the artifact's.

    ``S``, ``Sz`` (nk, nb, nb): the centroid band matrices of 1 and σ_z over
    the rotation block (module docstring).  Raises ``GATE bse_qp_star_gauge``
    when either misses the ``ibz_slab`` rule by more than
    :data:`STAR_RELATION_RTOL`; logs both residuals and the opposite rule's
    smallest residual over the antiunitary k (the sensitivity).  Returns
    ``[(name, max residual, worst k, sensitivity), ...]`` for the run report.
    """
    anti = np.asarray(antiunitary, dtype=bool)
    rows = []
    for name, M in (("1", S), ("sigma_z", Sz)):
        res, other = star_relation_residuals(M, parent, anti)
        k_bad = int(np.argmax(res))
        sens = float(other[anti].min()) if anti.any() else float("nan")
        rows.append((name, float(res.max()), k_bad, sens))
        log(f"  [qp-rotations] star gauge of the restart psi, band matrix of "
            f"{name}: max residual {res.max():.3e} (k {k_bad}); opposite "
            f"(anti)unitary rule misses by >= {sens:.3e} over "
            f"{int(anti.sum())} antiunitary k")
    bad = [r for r in rows if not r[1] <= STAR_RELATION_RTOL]
    if bad:
        name, res, k_bad, _ = bad[0]
        raise ValueError(
            f"GATE bse_qp_star_gauge: {where}: got a centroid band matrix of "
            f"{name} whose star image misses by {res:.3e} at full-BZ k "
            f"{k_bad} (parent {int(parent[k_bad])}, "
            f"{'antiunitary' if anti[k_bad] else 'unitary'}); want <= "
            f"{STAR_RELATION_RTOL:.0e}.  why: the QP rotation's off-wedge rows "
            f"are U(p) or U(p)* by that rule, which is the restart's gauge "
            f"only when psi_k = g psi_p.  fix: a restart whose centroids are "
            f"orbit-closed under the WFN's magnetic group, or a QP rotation "
            f"artifact diagonalized in this restart's own full-BZ gauge.")
    return rows


@functools.partial(jax.jit, static_argnames=("replicated",))
def _source_band_matrices(psi_v, psi_c, *, replicated):
    """S and S^z of the rotation block, replicated: (nk, nb, nb) each."""
    psi = jnp.concatenate([psi_v, psi_c], axis=1)
    sigma = (jnp.asarray([1.0, -1.0]) if psi.shape[2] == 2
             else jnp.ones((psi.shape[2],)))
    s = jnp.einsum("kmsu,knsu->kmn", jnp.conj(psi), psi)
    s_z = jnp.einsum("kmsu,s,knsu->kmn", jnp.conj(psi), sigma, psi)
    return (jax.lax.with_sharding_constraint(s, replicated),
            jax.lax.with_sharding_constraint(s_z, replicated))


@functools.partial(jax.jit, static_argnames=("sharding",))
def _rotate_block(psi_v, psi_c, u_val, u_cond, *, sharding):
    """Eq. (2) for the window's valence and conduction columns."""
    psi = jnp.concatenate([psi_v, psi_c], axis=1)

    def rotate(u):
        return jax.lax.with_sharding_constraint(
            jnp.einsum("kmsu,kmn->knsu", psi, u), sharding)
    return rotate(u_val), rotate(u_cond)


def rotate_window_psi(psi_v_src, psi_c_src, qp, mesh_xy, *, log=log.info):
    """ψ' of the QP window, eq. (2), after the star-gauge certificate.

    Parameters
    ----------
    psi_v_src, psi_c_src : (nk, n_occ - q0, ns, μ_pad) and (nk, q1 - n_occ,
        ns, μ_pad) complex, P(None, None, None, 'x') — the restart's DFT ψ of
        the whole block, as ``read_bse_payload`` returns it.
    qp : dict from :func:`load_qp_rotation`.

    Returns
    -------
    (psi_v, psi_c) : (nk, nv, ns, μ_pad) and (nk, nc, ns, μ_pad) complex,
        P(None, None, None, 'x').  The band contraction is local on every
        device; the replicated U columns are ``nk·nb·(nv+nc)·16`` bytes.
    """
    from common.collectives import device_put_process_local, gather_to_host
    replicated = NamedSharding(mesh_xy, P())
    if qp["certify_psi_star"]:
        S, Sz = _source_band_matrices(psi_v_src, psi_c_src,
                                      replicated=replicated)
        qp["diagnostics"]["psi_star"] = certify_star_gauge(
            gather_to_host(S), gather_to_host(Sz), qp["parent"],
            qp["antiunitary"], where=os.path.basename(qp["path"]), log=log)
    else:
        qp["diagnostics"]["psi_star"] = None
        log("  [qp-rotations] the restart stores psi on every full-BZ k: no "
            "star relation at the centroids to certify.")
    u_val, u_cond = (device_put_process_local(
        np.asarray(qp[name], dtype=np.complex128), replicated)
        for name in ("U_val", "U_cond"))
    return _rotate_block(
        psi_v_src, psi_c_src, u_val, u_cond,
        sharding=NamedSharding(mesh_xy, P(None, None, None, "x")))


def summary_text(qp) -> str:
    """One stage row of the QP-rotated basis (the full block,
    :func:`report_lines`, goes to the JSON sidecar)."""
    q0, q1 = qp["band_range"]
    d = qp["diagnostics"]
    gap = (np.min(qp["eps_c"]) - np.max(qp["eps_v"])) * RYD_TO_EV
    return (f"{os.path.basename(str(qp['path']))}: DFT bands {int(q0) + 1}-{int(q1)} "
            f"rotated whole, QP gap {gap:.4f} eV; cross-gap mixing theta_vc "
            f"{d['theta_vc_max']:.4f}, theta_cv {d['theta_cv_max']:.4f}")


def report_lines(qp):
    """The run report's QP-rotated-basis block: inputs, window, certificates."""
    q0, q1 = qp["band_range"]
    lo, hi = qp["n_occ"] - qp["n_val"], qp["n_occ"] + qp["n_cond"]
    d = qp["diagnostics"]
    star = d.get("psi_star")
    psi_line = ("not applicable (the restart stores psi on every full-BZ k)"
                if star is None else "; ".join(
                    f"{name} {res:.2e} (opposite rule >= {sens:.2e})"
                    for name, res, _, sens in star))
    gap = (np.min(qp["eps_c"]) - np.max(qp["eps_v"])) * RYD_TO_EV
    return (
        f"QP rotations   : {qp['path']}",
        f"QP block       : DFT bands [{q0}, {q1}) rotated whole; window "
        f"[{lo}, {hi}) in QP order ({qp['n_val']}v x {qp['n_cond']}c); "
        f"band-agnostic QP gap {gap:.4f} eV",
        f"Star gauge     : artifact H_QP {d['star_h_residual']:.2e} (opposite "
        f"rule >= {d['star_h_sensitivity']:.2e}); restart psi band matrices "
        f"{psi_line}; {d['n_antiunitary']} antiunitary k; tolerance "
        f"{STAR_RELATION_RTOL:.0e}",
        f"DFT-window leak: {d['dft_window_leakage_max']:.4f} at k "
        f"{d['dft_window_leakage_at'][0]}, QP band "
        f"{d['dft_window_leakage_at'][1]} (weight outside DFT bands "
        f"[{lo}, {hi}); the rotation itself is exact over the block)",
        f"Cross-gap mix  : theta_vc {d['theta_vc_max']:.4f}, theta_cv "
        f"{d['theta_cv_max']:.4f}, largest single state "
        f"{d['cross_gap_weight_max']:.4f}",
        "Dipoles        : interband-commutator QP velocity v_DFT + [dH, W] "
        "(qp_window_dipoles); the omitted explicit d_k dH is first order "
        "in theta",
    )


def qp_window_dipoles(dipole_cart, delta_e, qp):
    """Per-transition dipoles of the QP window, ``(3, nk, nc, nv)``.

    The QP velocity is the interband-commutator one of
    ``gw.qsgw_head.interband_commutator_velocity``,

        v^QP ≈ v^DFT + [ΔH, 𝒲],   𝒲_mn = v_mn / (E_m - E_n) on cross-gap
        pairs (m < n_occ <= n or n < n_occ <= m), 𝒲 = 0 within each class,

    where ``ΔH = H_QP - diag(E^DFT)``.  With ``v^DFT = [E^DFT, 𝒲] + v^same``
    (``v^same`` its same-class block) this is ``[H_QP, 𝒲] + v^same``, so in
    the QP basis, ``d = v^QP_{c'v'} / (E'_c' - E'_v')`` is exactly

        d_{c'v'} = (U_c^H 𝒲 U_v)_{c'v'}
                   + (U_c^H v^same U_v)_{c'v'} / (E'_c' - E'_v').             (3)

    At ``U = I`` the second term is exactly zero and the first is the
    ``--eqp`` dipole ``v_cv / ΔE_cv`` (``slice_dipole_to_bse_window``),
    bit for bit.  Omitted: the explicit ``∂_k ΔH`` of the covariant
    derivative ``D_k ΔH`` (the ``parallel_transport`` velocity), which enters
    only through the cross-gap mixing ``θ`` (``qp["diagnostics"]``) and is
    first order in it.  The pure DFT velocity ``U^H v^DFT U / ΔE'`` would also
    drop ``[ΔH, 𝒲]`` and with it the scissor renormalization that keeps
    ``r^QP = r^DFT`` for a diagonal ΔH.

    Parameters
    ----------
    dipole_cart : (3, nk, nb_file, nb_file) complex — ``<mk|v|nk>`` of
        ``load_dipole_h5`` (DFT basis, full BZ, the restart's gauge).
    delta_e : (nk, nb_file, nb_file) float — ``E_m - E_n`` (Ry, DFT).
    qp : dict from :func:`load_qp_rotation`.
    """
    q0, q1 = (int(b) for b in qp["band_range"])
    split = int(qp["n_occ"]) - q0
    v = np.asarray(dipole_cart[:, :, q0:q1, q0:q1])
    de = np.asarray(delta_e[:, q0:q1, q0:q1])
    W = np.zeros_like(v)
    W[:, :, split:, :split] = v[:, :, split:, :split] / de[None, :, split:, :split]
    W[:, :, :split, split:] = v[:, :, :split, split:] / de[None, :, :split, split:]
    same = v.copy()
    same[:, :, split:, :split] = 0.0
    same[:, :, :split, split:] = 0.0
    uc_h = np.conj(np.swapaxes(np.asarray(qp["U_cond"]), -1, -2))[None]
    uv = np.asarray(qp["U_val"])[None]
    de_qp = (np.asarray(qp["eps_c"])[:, :, None]
             - np.asarray(qp["eps_v"])[:, None, :])
    return uc_h @ W @ uv + (uc_h @ same @ uv) / de_qp[None]


def load_qp_rotation(restart_file, input_file, qp_rotations_file, *, header,
                     n_occ, n_val, n_cond, degeneracy_mode, degeneracy_tol_ry,
                     log=log.info):
    """Read, authenticate and window one ``qp_wfn_rotations.h5`` for the BSE.

    Refuses: ``--eqp`` beside it, a QP WFN deck, a restart from another WFN
    (``file_io.qp_wfn.refuse_conflicting_qp_state_sources``); another WFN's
    artifact (``authenticate_qp_rotations_source_wfn``); a k frame or a
    non-unitary U (``validate_qp_rotations_frame``); a block outside the
    restart's ζ training (``GATE bse_qp_rotation_block``); stored star tables
    that are not the WFN's (``GATE bse_qp_star_tables``); an artifact
    ``H_QP(k)`` that is not star-covariant (``GATE bse_qp_star_gauge``); a
    window outside the block or cut in QP order (:func:`qp_window_from_artifact`).

    Returns the window dict of :func:`qp_window_from_artifact` plus ``path``,
    ``band_range``, ``n_occ``, the star ``parent``/``antiunitary`` tables and
    ``certify_psi_star`` (the restart stores ψ on the wedge only).
    """
    from ffi import _services
    _services.ensure_on_path()
    import symmetry_maps
    from wfn_loader import WfnLoader
    from file_io.qp_wfn import (authenticate_qp_rotations_source_wfn,
                                refuse_conflicting_qp_state_sources)
    from file_io.restart_bundle import (read_qp_rotations_artifact,
                                        read_star_map,
                                        validate_qp_rotations_frame)
    from .bse_window import (BseWindowOutsideZetaTrainingError,
                             _generate_kpts_grid, _parse_wfn_path,
                             assert_bse_window_in_zeta_training)

    path = os.path.abspath(os.fspath(qp_rotations_file))
    wfn_path = _parse_wfn_path(input_file)
    refuse_conflicting_qp_state_sources(
        wfn_path=wfn_path, qp_rotations_file=path,
        state_artifact_path=restart_file, where="bse --qp-rotations")
    artifact = read_qp_rotations_artifact(path)
    wfn = WfnLoader(wfn_path)
    authenticate_qp_rotations_source_wfn(artifact, wfn, artifact_path=path)
    sym = wfn.symmetry()
    # The BSE's k rows are the C-order grid; the star tables below index
    # the symmetry service's full-BZ list, so both must be that grid.
    grid_k = _generate_kpts_grid(*(int(n) for n in header["grid"]))
    dk = np.asarray(sym.unfolded_kpts, dtype=np.float64) - grid_k
    if dk.shape != grid_k.shape or np.abs(dk - np.rint(dk)).max() > 1e-8:
        raise ValueError(
            f"GATE bse_qp_k_order: got a symmetry-service full-BZ list that "
            f"is not the BSE's C-order {tuple(header['grid'])} grid; want "
            f"the same rows.  why: U, E and the star tables are indexed by "
            f"that list, the restart psi by the grid.")
    U, E, (q0, q1) = validate_qp_rotations_frame(
        artifact, kgrid=header["grid"], kpoints_crys=grid_k,
        artifact_path=path)
    try:
        assert_bse_window_in_zeta_training(header, q0, q1)
    except BseWindowOutsideZetaTrainingError as exc:
        raise ValueError(
            f"GATE bse_qp_rotation_block: got QP block [{q0}, {q1}) from "
            f"{os.path.basename(path)}; want it inside the restart's ζ "
            f"training.  why: eq. (2) needs every DFT band of the block, and a "
            f"sliced eigenvector matrix is not unitary.  fix: rerun the GW "
            f"with nval/ncond covering the block.  ({exc})") from exc
    if int(header["logical_band_count"]) < q1:
        raise ValueError(
            f"GATE bse_qp_rotation_block: got QP block [{q0}, {q1}); the "
            f"restart stores psi for {int(header['logical_band_count'])} "
            f"bands only.")
    irr, sidx, nss = symmetry_maps.star_tables_of(sym)
    kirr = np.asarray(sym.kirr_fullids, dtype=np.int64)
    stored = read_star_map(path, "U_mnk")
    if stored is not None and not (
            np.array_equal(stored[0], irr) and np.array_equal(stored[1], sidx)
            and int(stored[2]) == int(nss)
            and np.array_equal(np.asarray(artifact["kirr_to_kfull"]), kirr)):
        raise ValueError(
            f"GATE bse_qp_star_tables: got {os.path.basename(path)} unfold "
            f"tables that differ from the WFN symmetry service's; want the "
            f"same (irr_idx_k, sym_idx_k, n_sym_spatial, kirr_to_kfull).  "
            f"why: its off-wedge rows are star images by its own tables.")
    parent, anti = star_parents(irr, sidx, nss, kirr)
    H = np.einsum("kmn,kn,kln->kml", U, E, np.conj(U))
    h_res, h_other = star_relation_residuals(H, parent, anti)
    if not float(h_res.max()) <= STAR_RELATION_RTOL:
        k_bad = int(np.argmax(h_res))
        raise ValueError(
            f"GATE bse_qp_star_gauge: got H_QP(k) = U E U^H from "
            f"{os.path.basename(path)} missing its star image by "
            f"{h_res.max():.3e} at full-BZ k {k_bad}; want <= "
            f"{STAR_RELATION_RTOL:.0e}.  why: a star-covariant Σ gives "
            f"H_QP(gp) = H_QP(p) or H_QP(p)* in the unfolded gauge.")
    window = qp_window_from_artifact(
        U, E, (q0, q1), n_occ=n_occ, n_val=n_val, n_cond=n_cond,
        degeneracy_mode=degeneracy_mode, degeneracy_tol_ry=degeneracy_tol_ry,
        log=log)
    rows = header.get("parent_k_rows")
    window.update(
        path=path, band_range=(q0, q1), n_occ=int(n_occ), parent=parent,
        antiunitary=anti,
        certify_psi_star=(rows is not None
                          and int(np.size(rows)) < int(sym.nk_tot)))
    d = window["diagnostics"]
    d["star_h_residual"] = float(h_res.max())
    d["star_h_sensitivity"] = (float(h_other[anti].min()) if anti.any()
                               else float("nan"))
    d["n_antiunitary"] = int(np.sum(anti))
    lo, hi = int(n_occ) - window["n_val"], int(n_occ) + window["n_cond"]
    log(f"  [qp-rotations] {path}: block [{q0}, {q1}), window [{lo}, {hi}) "
        f"in QP order ({window['n_val']}v x {window['n_cond']}c), "
        f"{int(np.sum(anti))}/{anti.size} full-BZ k antiunitary; artifact "
        f"H_QP star residual {h_res.max():.2e}")
    log(f"  [qp-rotations] DFT-index leakage of the QP window "
        f"{d['dft_window_leakage_max']:.4f} (k {d['dft_window_leakage_at'][0]}"
        f", QP band {d['dft_window_leakage_at'][1]}); cross-gap mixing "
        f"theta_vc {d['theta_vc_max']:.4f}, theta_cv {d['theta_cv_max']:.4f}, "
        f"largest single-state cross-gap weight "
        f"{d['cross_gap_weight_max']:.4f}; band-agnostic QP gap "
        f"{(window['eps_c'].min() - window['eps_v'].max()) * RYD_TO_EV:.4f} eV")
    return window
