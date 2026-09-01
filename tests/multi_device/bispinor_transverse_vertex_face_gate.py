"""Real 4-rank CUDA gate: bispinor Σ^B transverse-vertex insertion —
``gw.wavefunction_bundle.with_lorentz_vertices`` — against an independent
4x4-γ̃-matmul NumPy reference, AND legacy-vs-face parity through the exact
kernel chain ``gw.sigma_x_bispinor.compute_sigma_x_bispinor`` uses
(``gw.cohsex_sigma._make_cohsex_kernels``'s layout-dispatched ``sigma_sx``).

Guide: reports/gwjax_low_mem_bands_audit_2026-08-22/report.md, census row
"Bispinor transverse exchange" — this file is the "transverse-exchange
algebra parity legacy-vs-face on real 4-rank CUDA with a genuine 4-spinor
fixture" gate that row's guide names.

WHY REAL CUDA, NOT EMULATED.  The face-layout leg of every check below
routes ``build_G``/the band projector through ``distrib_la.gemm_plan``
(cuBLASMp, CUDA-only, refuses on any non-CUDA mesh by construction) — same
reason ``low_mem_bands_g_projection_hartree_gate.py`` (this file's sibling
and structural template) needs real hardware.  A CPU-emulated leg would
silently pass NOTHING for the face side.

SAME ψ, EVERY REPRESENTATION.  Exactly ``low_mem_bands_g_projection_
hartree_gate.py``'s own convention: ONE host NumPy ψ array (here ns=4, a
GENUINE bispinor/4-spinor shape — γ̃^i is a real 4x4 matrix, not the
ns<4 no-op every OTHER face gate in this tree happens to exercise), every
legacy/face copy derived from it by pure reshape/transpose.

Checks:
  1. ``with_lorentz_vertices`` + ``build_G`` — for (mu_L, nu_L) in
     {(0,0), (1,1), (1,2), (2,3), (3,1)} (identity, a diagonal transverse
     tile, and the three off-diagonal upper/lower pairs Σ^B actually
     sums), BOTH layouts against an INDEPENDENT NumPy reference built
     from the FULL 4x4 γ̃ matrix (``common.gamma_matrices.gamma0..3``),
     not the perm/phase gather this function itself uses — a genuinely
     different mechanism for the same claim (the module's own docstring:
     "Replaces a 4x4 matmul ... with a gather + multiply").
  2. The full ``sigma_sx`` chain (``build_G`` -> FFT convolve -> band
     projection) as ``compute_sigma_x_bispinor`` actually calls it via
     ``_make_cohsex_kernels``, legacy vs face, on the SAME vertex-
     inserted ψ and the SAME (Hermitian, physical) V tile — this is
     ``compute_sigma_x_bispinor``'s own per-tile mechanism, minus the
     HDF5 reader / 9-tile Python loop (pure orchestration, not new
     physics).  The same fixture also checks that the packed-photon block
     kernel's dynamic X/SX selector reproduces V and 2V algebra through
     one compiled executable, and that its full-photon caller can request
     Hartree without running the historical bare-X path.  ``mu_L == 0``
     also exercises the no-op short-circuit path
     (``with_lorentz_vertices`` returns the SAME bundle object).
  3. The static four-current bubble for all 16 Lorentz blocks against a
     literal ordered-band-pair NumPy oracle on deterministic complex
     broken-TR states.  Longitudinal and transverse centroid extents differ,
     so CT/TC reverse axes cannot pass by an accidental square transpose.
     The combined tensor is Hermitian per q and exactly q-reciprocal; a
     distinct-but-value-identical CC endpoint matches scalar charge.

Run:
    lx run -N 1 -G 4 -n 4 bash <pythonpath-wrapper> python3 -u \\
        tests/multi_device/bispinor_transverse_vertex_face_gate.py \\
        --mesh 2x2
"""
from __future__ import annotations

import argparse
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TESTS)
if os.path.join(_REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "src"))

if __name__ == "__main__":
    from runtime import initialize_communicator_stack
    _RUNTIME = initialize_communicator_stack(platform="gpu")

import numpy as np                                              # noqa: E402

RTOL = 1e-10

#: The 5 Lorentz-index pairs exercised: identity, one diagonal transverse
#: tile, and the three distinct (unordered) off-diagonal transverse pairs
#: Σ^B actually sums over (each seen once as (i,j) and once as its
#: transpose (j,i), since gamma_apply is NOT assumed symmetric under
#: swapping which side gets which index).
_LORENTZ_PAIRS = ((0, 0), (1, 1), (1, 2), (2, 3), (3, 1))


# ---------------------------------------------------------------------------
# Shared operand / measurement helpers — same idiom as
# low_mem_bands_g_projection_hartree_gate.py / test_distrib_la_multiproc.py.
# ---------------------------------------------------------------------------

def _rng_mat(rng, shape, dtype=np.complex128):
    a = rng.standard_normal(shape)
    if np.dtype(dtype).kind == "c":
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


def _put(np_arr, mesh, spec):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(np.asarray(np_arr), NamedSharding(mesh, P(*spec)))


def _gather(x):
    import jax
    if jax.process_count() == 1:
        return np.asarray(x)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-300)


def _gamma_full(mu_L: int) -> np.ndarray:
    """The FULL 4x4 γ̃^{mu_L} table, straight off
    ``common.gamma_matrices`` — independent of
    ``gamma_perm_phase``/``gamma_apply``'s gather+multiply decomposition
    (different code path, same module's own literal tables)."""
    import jax
    from common.gamma_matrices import gamma0, gamma1, gamma2, gamma3
    return np.asarray(jax.device_get([gamma0, gamma1, gamma2, gamma3][mu_L]))


# ---------------------------------------------------------------------------
# 1. with_lorentz_vertices + build_G vs an independent 4x4-matmul reference.
# ---------------------------------------------------------------------------

def check_vertex_build_g(mesh, dtype="complex128", *, mu_L, nu_L, ns=4,
                         mu=8, nb=6, nk=2):
    from gw.wavefunction_bundle import (
        BandSlices, Wavefunctions, PSI_XN_SPEC, PSI_YR_SPEC,
        PSI_MUN_SPEC, PSI_NMU_SPEC, with_lorentz_vertices)
    from gw.greens_function_kernel import build_G
    from distrib_la import gemm_plan

    rng = np.random.default_rng(2026082301 + 17 * mu_L + nu_L)
    psi_np = _rng_mat(rng, (nk, nb, ns, mu), dtype)   # (nk, n, s, mu)
    psi_band_last = psi_np.transpose(0, 2, 3, 1)       # (nk, s, mu, n)
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)
    rep2 = _put(np.zeros((nk, nb)), mesh, (None, None))

    wfns_legacy = Wavefunctions(
        psi_xn=_put(psi_band_last, mesh, PSI_XN_SPEC),
        psi_yr=_put(psi_np, mesh, PSI_YR_SPEC),
        enk=rep2, occ=rep2, slices=slices,
    )
    wfns_face = Wavefunctions(
        psi_mun=_put(psi_band_last, mesh, PSI_MUN_SPEC),
        psi_nmu=_put(psi_np, mesh, PSI_NMU_SPEC),
        enk=rep2, occ=rep2, slices=slices, layout="face",
    )

    wfns_legacy_v = with_lorentz_vertices(wfns_legacy, mu_L, nu_L)
    wfns_face_v = with_lorentz_vertices(wfns_face, mu_L, nu_L)
    if mu_L == 0 and nu_L == 0:
        assert wfns_legacy_v is wfns_legacy, (
            "with_lorentz_vertices(0,0) must be a true no-op (same object)")
        assert wfns_face_v is wfns_face, (
            "with_lorentz_vertices(0,0) must be a true no-op (same object)")

    plan = gemm_plan(mesh, m=mu * ns, k=nb, n=mu * ns, nq=nk, dtype=dtype)
    G_legacy = build_G(wfns_legacy_v.psi_xn, wfns_legacy_v.psi_yr,
                       layout="legacy")
    G_face = build_G(wfns_face_v.psi_mun, wfns_face_v.psi_nmu,
                     layout="face", gemm=plan)

    # Independent reference: FULL 4x4 gamma matmul on the spin axis of
    # the direct operand (psi_np transposed to band-last) and the
    # conjugated operand (psi_np as-is), THEN build_G's own plain
    # contraction formula -- no gather/perm/phase anywhere in this path.
    Gam_mu = _gamma_full(mu_L)
    Gam_nu = _gamma_full(nu_L)
    psi_direct_ref = np.einsum("ab,kbxn->kaxn", Gam_mu, psi_band_last,
                               optimize=True)          # (nk,s,mu,n)
    psi_conj_ref = np.einsum("ab,knbx->knax", Gam_nu, psi_np,
                             optimize=True)             # (nk,n,s,mu)
    want = np.einsum("ksxn,knty->ksxty", psi_direct_ref,
                     np.conj(psi_conj_ref), optimize=True)

    r_legacy = _rel(_gather(G_legacy), want)
    r_face = _rel(_gather(G_face), want)
    assert r_legacy < RTOL, (
        f"legacy vertex+G rel err {r_legacy:.3e} (mu_L={mu_L}, nu_L={nu_L})")
    assert r_face < RTOL, (
        f"face vertex+G rel err {r_face:.3e} (mu_L={mu_L}, nu_L={nu_L})")
    return {"legacy": r_legacy, "face": r_face}


# ---------------------------------------------------------------------------
# 2. Full sigma_sx chain (build_G -> convolve -> project), legacy vs face,
#    exactly compute_sigma_x_bispinor's own per-tile mechanism.
# ---------------------------------------------------------------------------

def check_sigma_sx_chain_face_matches_legacy(
        mesh, dtype="complex128", *, mu_L, nu_L, ns=4, mu=8, nb_full=8,
        nb_sigma=5, nk=2):
    from gw.cohsex_sigma import _make_cohsex_kernels
    from gw.photon_sigma import (
        _TERM_SX, _TERM_X, _make_photon_static_block_kernel)
    from gw.wavefunction_bundle import (
        BandSlices, Wavefunctions, PSI_XN_SPEC, PSI_XR_SPEC, PSI_YR_SPEC,
        PSI_YN_SPEC, PSI_MUN_SPEC, PSI_NMU_SPEC, with_lorentz_vertices)

    rng = np.random.default_rng(2026082310 + 17 * mu_L + nu_L)
    psi_np = _rng_mat(rng, (nk, nb_full, ns, mu), dtype)   # (nk,n,s,mu)
    psi_band_last = psi_np.transpose(0, 2, 3, 1)
    enk_np = np.sort(rng.standard_normal((nk, nb_full)), axis=1)
    f_np = rng.uniform(0.05, 0.95, size=(nk, nb_sigma))
    Gij_np = np.zeros((nk, nb_sigma, nb_sigma), dtype=complex)
    idx = np.arange(nb_sigma)
    Gij_np[:, idx, idx] = f_np
    V0_np = _rng_mat(rng, (mu, mu), dtype)
    V0_np = 0.5 * (V0_np + np.conj(V0_np.T))     # Hermitian, physical V(q)
    # sigma_sx (unlike hartree) runs V through the flat-k FFT convolve
    # (_convolve -> make_flat_k_ifftn), which needs the FULL nkx*nky*nkz
    # leading extent -- one (Hermitian) V slice per k, not a single q=0
    # slot the way check_hartree's own V_q_np gets away with.
    V_q_np = np.tile(V0_np[None], (nk, 1, 1))
    kgrid = (nk, 1, 1)
    slices = BandSlices.from_band_edges(0, 0, 2, nb_sigma, nb_full)

    wfns_legacy = Wavefunctions(
        psi_xn=_put(psi_band_last, mesh, PSI_XN_SPEC),
        psi_xr=_put(psi_np, mesh, PSI_XR_SPEC),
        psi_yr=_put(psi_np, mesh, PSI_YR_SPEC),
        psi_yn=_put(psi_band_last, mesh, PSI_YN_SPEC),
        enk=_put(enk_np, mesh, (None, None)),
        occ=_put(np.zeros_like(enk_np), mesh, (None, None)),
        slices=slices,
    )
    wfns_face = Wavefunctions(
        psi_nmu=_put(psi_np, mesh, PSI_NMU_SPEC),
        psi_mun=_put(psi_band_last, mesh, PSI_MUN_SPEC),
        enk=_put(enk_np, mesh, (None, None)),
        occ=_put(np.zeros_like(enk_np), mesh, (None, None)),
        slices=slices, layout="face",
    )
    Gij_legacy = _put(Gij_np, mesh, (None, None, None))
    Gij_face = _put(Gij_np, mesh, (None, None, None))
    V_q_legacy = _put(V_q_np, mesh, (None, None, None))
    V_q_face = _put(V_q_np, mesh, (None, None, None))

    sigma_sx_legacy, _ = _make_cohsex_kernels(mesh, kgrid, nk,
                                              layout="legacy")
    sigma_sx_face, _ = _make_cohsex_kernels(
        mesh, kgrid, nk, layout="face",
        face_shape=(nk, nb_full, mu, ns))

    wfns_legacy_v = with_lorentz_vertices(wfns_legacy, mu_L, nu_L)
    wfns_face_v = with_lorentz_vertices(wfns_face, mu_L, nu_L)

    # Legacy: with_lorentz_vertices only ever touches psi_xn/psi_yr (the
    # G-build's two fields), leaving psi_xr/psi_yn (the projection's own
    # two fields) untouched — so the single-argument call already keeps
    # the outer projection bra/ket un-rotated, exactly what the physics
    # requires (module docstring of gw.sigma_x_bispinor).
    got_legacy = _gather(sigma_sx_legacy(wfns_legacy_v, Gij_legacy, V_q_legacy))
    # Face: the two-array carrier serves BOTH roles, so the vertex-
    # inserted bundle must be threaded through ONLY as wfns_g (the
    # G-build operand); wfns_face itself (untouched) supplies the
    # projection's bra/ket.  See _make_cohsex_kernels_face's own
    # sigma_sx docstring for the derivation and the O(1) discrepancy
    # this parameter exists to fix (found by an earlier version of THIS
    # gate, before wfns_g existed).
    got_face_full = _gather(sigma_sx_face(wfns_face, Gij_face, V_q_face,
                                          wfns_g=wfns_face_v))
    # Mirrors compute_sigma_x_bispinor's OWN gather-then-window step for
    # layout='face' (its return contract: (nk, nb_sigma, nb_sigma) in
    # EITHER layout).
    got_face = got_face_full[:, :nb_sigma, :nb_sigma]

    assert got_legacy.shape == got_face.shape, (
        f"shape mismatch: legacy={got_legacy.shape} face={got_face.shape}")
    r = _rel(got_face, got_legacy)
    assert r < RTOL, (
        f"sigma_sx chain face-vs-legacy rel err {r:.3e} "
        f"(mu_L={mu_L}, nu_L={nu_L})")

    # The coupled photon path must evaluate X[V_packed] and SX[W_packed]
    # through the same Green/convolution/projector graph and one compiled
    # executable.  Use W=2V so both selector branches have an exact algebraic
    # reference, while keeping V/W in their production 2-D packed sharding.
    V_q_packed = _put(V_q_np, mesh, (None, "x", "y"))
    W_q_packed = 2.0 * V_q_packed
    photon_block = _make_photon_static_block_kernel(
        mesh, kgrid, nk, wfns_face, wfns_face)
    left_g = with_lorentz_vertices(wfns_face, mu_L, 0)
    right_g = with_lorentz_vertices(wfns_face, 0, nu_L)
    photon_x = photon_block(
        wfns_face, wfns_face, left_g, right_g, Gij_face,
        W_q_packed, V_q_packed, np.asarray(_TERM_X, dtype=np.int32))
    photon_x.block_until_ready()
    photon_sx = photon_block(
        wfns_face, wfns_face, left_g, right_g, Gij_face,
        W_q_packed, V_q_packed, np.asarray(_TERM_SX, dtype=np.int32))
    photon_sx.block_until_ready()
    r_photon_x = _rel(_gather(photon_x), got_face_full)
    r_photon_sx = _rel(_gather(photon_sx), 2.0 * got_face_full)
    assert r_photon_x < RTOL, (
        f"photon X[V] rel err {r_photon_x:.3e} "
        f"(mu_L={mu_L}, nu_L={nu_L})")
    assert r_photon_sx < RTOL, (
        f"photon SX[2V] rel err {r_photon_sx:.3e} "
        f"(mu_L={mu_L}, nu_L={nu_L})")
    cache_size = photon_block._cache_size()
    assert cache_size == 1, (
        "dynamic photon X/SX selector compiled more than one executable: "
        f"cache_size={cache_size}")

    return {
        "face_vs_legacy": r,
        "photon_x_vs_v": r_photon_x,
        "photon_sx_vs_2v": r_photon_sx,
        "photon_kernel_cache_size": cache_size,
        "skipped_x_max_abs": max_skipped_x,
    }


# ---------------------------------------------------------------------------
# 3. Static four-current ordered-pair response vs literal band sum.
# ---------------------------------------------------------------------------

def check_four_current_ordered_pair_all16(
        mesh, dtype="complex128", *, nk=3, nb=4, n_c=6, n_t=8,
        spin_pair_stream=None):
    """Discriminate AB/BA, q/-q, and rectangular CT/TC orientations."""
    import jax
    from types import SimpleNamespace

    from symmetry_maps import q_negation_index
    from gw import w_isdf
    from gw.wavefunction_bundle import (
        BandSlices, Wavefunctions, PSI_MUN_SPEC, PSI_NMU_SPEC)

    if dtype != "complex128":
        raise ValueError("four-current ordered-pair gate requires complex128")
    rng = np.random.default_rng(2026082517)
    psi_c = _rng_mat(rng, (nk, nb, 4, n_c), np.complex128)
    psi_t = _rng_mat(rng, (nk, nb, 4, n_t), np.complex128)
    enk = np.tile(np.asarray([-0.5, -0.5, 0.5, 0.5]), (nk, 1))
    occ = np.tile(np.asarray([1.0, 1.0, 0.0, 0.0]), (nk, 1))
    slices = BandSlices.from_band_edges(0, 0, 2, 4, 4)

    def bundle(psi):
        return Wavefunctions(
            psi_mun=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_MUN_SPEC),
            psi_nmu=_put(psi, mesh, PSI_NMU_SPEC),
            enk=_put(enk, mesh, (None, None)),
            occ=_put(occ, mesh, (None, None)),
            slices=slices, layout="face")

    wfns_c = bundle(psi_c)
    wfns_t = bundle(psi_t)
    families = (wfns_c, wfns_t, wfns_t, wfns_t)
    psi_families = (psi_c, psi_t, psi_t, psi_t)
    extents = (n_c, n_t, n_t, n_t)
    offsets = np.cumsum((0,) + extents)
    quad = SimpleNamespace(
        tau=np.asarray([0.0]), alpha=np.asarray([1.0]))
    meta = SimpleNamespace(nkx=nk, nky=1, nkz=1, nk_tot=nk)

    got_blocks = {}
    worst_oracle = 0.0
    for A in range(4):
        gamma_a = _gamma_full(A)
        psi_a = psi_families[A]
        for B in range(4):
            gamma_b = _gamma_full(B)
            psi_b = psi_families[B]
            got = _gather(w_isdf.compute_no_pair_dirac_current_block(
                families[A], families[B], quad, meta, mesh,
                vertex_left=A, vertex_right=B,
                spin_pair_stream=spin_pair_stream))
            want = np.zeros(
                (nk, extents[A], extents[B]), dtype=np.complex128)
            for q in range(nk):
                for k in range(nk):
                    kmq = (k - q) % nk
                    for v in (0, 1):
                        for c in (2, 3):
                            # Occupied-to-empty ordered transition.
                            left_vc = np.einsum(
                                "am,aA,Am->m", psi_a[k, v], gamma_a,
                                np.conj(psi_a[kmq, c]), optimize=True)
                            right_vc = np.einsum(
                                "bn,Bb,Bn->n", np.conj(psi_b[k, v]),
                                gamma_b, psi_b[kmq, c], optimize=True)
                            # Empty-to-occupied ordered transition.  This is
                            # F_BA(-q)^dagger after relabelling k; spelling it
                            # directly keeps the oracle independent of the
                            # production R-space completion.
                            left_cv = np.einsum(
                                "am,aA,Am->m", psi_a[k, c], gamma_a,
                                np.conj(psi_a[kmq, v]), optimize=True)
                            right_cv = np.einsum(
                                "bn,Bb,Bn->n", np.conj(psi_b[k, c]),
                                gamma_b, psi_b[kmq, v], optimize=True)
                            want[q] -= (
                                left_vc[:, None] * right_vc[None, :]
                                + left_cv[:, None] * right_cv[None, :]
                            ) / np.sqrt(float(nk))
            err = _rel(got, want)
            worst_oracle = max(worst_oracle, err)
            assert err < RTOL, (
                f"four-current ({A},{B}) ordered-pair rel err {err:.3e}")
            got_blocks[A, B] = got

    combined = np.zeros(
        (nk, int(offsets[-1]), int(offsets[-1])), dtype=np.complex128)
    for A in range(4):
        for B in range(4):
            combined[:, offsets[A]:offsets[A + 1],
                     offsets[B]:offsets[B + 1]] = got_blocks[A, B]
    scale = max(float(np.max(np.abs(combined))), 1e-300)
    herm = max(float(np.max(np.abs(row - row.conj().T)))
               for row in combined) / scale
    neg = q_negation_index((nk, 1, 1))
    reciprocity = float(np.max(
        np.abs(combined - np.conj(combined[neg])))) / scale
    assert herm < RTOL, f"combined current chi Hermiticity {herm:.3e}"
    assert reciprocity < RTOL, (
        f"combined current chi q reciprocity {reciprocity:.3e}")

    # Object identity is deliberately absent from the physics dispatch.
    # A separately allocated but value-identical right endpoint must run the
    # same CC contraction and equal the scalar charge SSOT bit for bit.
    wfns_c_clone = bundle(psi_c.copy())
    assert wfns_c_clone is not wfns_c
    cc_distinct = _gather(w_isdf.compute_no_pair_dirac_current_block(
        wfns_c, wfns_c_clone, quad, meta, mesh,
        vertex_left=0, vertex_right=0,
        spin_pair_stream=spin_pair_stream))
    cc_charge = _gather(w_isdf.compute_chi0(
        wfns_c, quad, meta, mesh))
    cc_rel = _rel(cc_distinct, cc_charge)
    if spin_pair_stream:
        assert cc_rel < RTOL, (
            "spin-pair streamed CC endpoints differ from charge SSOT: "
            f"rel={cc_rel:.3e}")
    else:
        assert np.array_equal(cc_distinct, cc_charge), (
            "distinct value-identical CC endpoints differ from charge SSOT: "
            f"rel={cc_rel:.3e}")

    return {
        "all16_worst_rel": worst_oracle,
        "combined_hermiticity": herm,
        "q_reciprocity": reciprocity,
        "distinct_cc_bit_equal": bool(np.array_equal(cc_distinct, cc_charge)),
        "distinct_cc_rel": cc_rel,
    }


# ---------------------------------------------------------------------------
# CLI mode.
# ---------------------------------------------------------------------------

_CLI_CELLS = (
    [(f"vertex_g_{mu_L}{nu_L}",
      (lambda mesh, dt, mu_L=mu_L, nu_L=nu_L:
       check_vertex_build_g(mesh, dt, mu_L=mu_L, nu_L=nu_L)))
     for (mu_L, nu_L) in _LORENTZ_PAIRS]
    + [(f"sigma_sx_chain_{mu_L}{nu_L}",
        (lambda mesh, dt, mu_L=mu_L, nu_L=nu_L:
         check_sigma_sx_chain_face_matches_legacy(
             mesh, dt, mu_L=mu_L, nu_L=nu_L)))
       for (mu_L, nu_L) in _LORENTZ_PAIRS]
    + [("four_current_ordered_pair_all16",
        check_four_current_ordered_pair_all16),
       ("four_current_spin_pair_stream_all16",
        (lambda mesh, dt: check_four_current_ordered_pair_all16(
            mesh, dt, spin_pair_stream=True)))]
)


def _mesh_from_arg(spec):
    import jax
    from jax.sharding import Mesh
    px, py = (int(v) for v in spec.lower().split("x"))
    return Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))


def _cli_main():
    import jax

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--dtype", default="complex128")
    args = ap.parse_args()
    mesh = _mesh_from_arg(args.mesh)
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()}",
       flush=True)

    failures, ran = 0, 0
    for name, fn in _CLI_CELLS:
        if args.only and args.only not in name:
            continue
        tag = f"{name}[{args.mesh},{args.dtype}]"
        try:
            out = fn(mesh, args.dtype)
            ran += 1
            p0(f"PASS {tag} {out if out is not True else ''}", flush=True)
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {tag}: {exc}", flush=True)
        except Exception as exc:                                # noqa: BLE001
            failures += 1
            p0(f"ERROR {tag}: {type(exc).__name__}: "
               f"{' '.join(str(exc).split())[:600]}", flush=True)
    p0(f"done: {ran} cells ran, {failures} failures", flush=True)
    return 1 if (failures or ran == 0) else 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_cli_main)
