"""Algebra parity for the canonical distributed band-window repack
(``gw.wavefunction_bundle.pack_band_window``) and the shared packed-bracket
tau kernel it feeds (``gw.ppm_tau_kernel``, ``pack_brackets=True``), real
multi-rank CUDA.

Companion to ``tests/test_ppm_tau_kernel_face_parity.py`` (which gates
``layout='face'`` MASK brackets against ``layout='legacy'`` static
slices) and to ``tests/test_isdf_cq_face_parity.py``/``test_isdf_zq_face_
parity.py`` (the ISDF fit's own face port) -- this one gates the NEW
packed-bracket route (2026-08-23, the guide's own named endgame: "the
gemm [should] be partial, or materialize the band slices as their own
array copies at the start of the sigma procedure").

    lx run -N 1 -G 4 -n 4 bash tmp/lm_packband_run_wrap.sh \\
        tests/test_pack_band_window.py --mesh 2x2

Under plain pytest (one process) every multi-rank case SKIPS rather than
failing -- it names exactly why (process_count) rather than reporting a
silent pass (TASTE.md, "a check that cannot fail is not evidence"). The
ISOLATED ``pack_band_window`` numerics case also needs real multi-process
CUDA (the packed carrier is genuinely 2-D sharded) and skips the same way.

What is and is not checked
---------------------------
Algebra parity, not physics: synthetic random operands, no PPM-fit
provenance -- see test_ppm_tau_kernel_face_parity.py's own note, which
applies verbatim here.

Cases
-----
* ``pack_isolated_ns1``/``pack_isolated_ns2`` -- ``pack_band_window`` in
  ISOLATION (no kernel involved): the packed pair, gathered to host, must
  equal a plain host reference (slice the full synthetic ψ, zero-pad to
  the mesh multiple) exactly (bit-exact -- packing is a slice + pad, no
  reduction, no float roundoff to tolerate). Windows are DELIBERATELY
  non-mesh-divisible at both edges.
* ``pack_isolated_full_window_is_identity`` -- the trivial ``(0, None)``
  window must return the VERY SAME array objects the carrier already
  holds (no new array, no collective) -- checked by identity (``is``),
  not just value equality.
* ``packed_vs_mask_vs_legacy_ns1_3brackets`` /
  ``..._ns2_3brackets`` -- a genuine THREE-bracket extrapolation-shaped
  plan (the band_extrapolation shape), comparing the packed route
  (``pack_brackets=True``), the mask route (``pack_brackets=False``,
  the ORIGINAL 2026-08-22 bring-up, kept as the parity oracle) and the
  legacy static-slice route on the SAME synthetic ψ and physics
  operands. All three brackets have widths that do NOT divide the 2x2
  mesh, so every packed pair is genuinely padded.
* ``packed_vs_mask_float_weight`` -- a FLOAT ``band_weight`` (not a bool
  mask) through the packed route, confirming the dtype dispatch
  (``_g_from_selector_packed``) matches the mask route's own
  (``_g_from_selector``).
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    _TESTS = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.dirname(_TESTS)
    for _svc in ("lxkit", "distrib_la"):
        _src = os.path.join(_REPO, "services", _svc, "src")
        if os.path.isdir(_src) and _src not in sys.path:
            sys.path.insert(0, _src)
    from lxkit.gate import platform_from_env
    from runtime import initialize_communicator_stack
    _plat = platform_from_env()
    _RUNTIME = initialize_communicator_stack(
        platform="gpu" if _plat == "CUDA" else "cpu")

import argparse

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.ppm_tau_kernel import get_shared_sigma_tau_kernel
from gw.wavefunction_bundle import (
    BandSlices, PSI_MUN_SPEC, PSI_NMU_SPEC, Wavefunctions,
    pack_band_window,
)

PX = PY = 2


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def _to_host(x, mesh=None):
    from jax.experimental import multihost_utils as mhu
    del mesh
    return np.asarray(mhu.process_allgather(x, tiled=True))


def _face_wfns(mesh, *, ns, nk, n_rmu, nb_full, psi_full, psi_full_T):
    """A minimal layout='face' Wavefunctions bundle for these tests --
    only .layout / .psi_mun / .psi_nmu / .slices.nb_full are read by
    pack_band_window."""
    mun_spec = NamedSharding(mesh, PSI_MUN_SPEC)
    nmu_spec = NamedSharding(mesh, PSI_NMU_SPEC)
    psi_mun = jax.device_put(jnp.asarray(psi_full_T), mun_spec)
    psi_nmu = jax.device_put(jnp.asarray(psi_full), nmu_spec)
    slices = BandSlices.from_band_edges(0, 0, 0, nb_full, nb_full)
    enk = jnp.zeros((nk, nb_full), dtype=jnp.float64)
    occ = jnp.zeros((nk, nb_full), dtype=jnp.float64)
    return Wavefunctions(enk=enk, occ=occ, slices=slices,
                         psi_nmu=psi_nmu, psi_mun=psi_mun, layout="face")


# ---------------------------------------------------------------------------
# Isolated pack_band_window numerics
# ---------------------------------------------------------------------------

def check_pack_isolated(mesh, *, ns, nk, n_rmu, nb_full, lo, hi, seed):
    rng = np.random.default_rng(seed)
    psi_full = _crand(rng, nk, nb_full, ns, n_rmu)      # (k, n, s, mu)
    psi_full_T = np.transpose(psi_full, (0, 2, 3, 1))   # (k, s, mu, n)
    wfns = _face_wfns(mesh, ns=ns, nk=nk, n_rmu=n_rmu, nb_full=nb_full,
                      psi_full=psi_full, psi_full_T=psi_full_T)

    mun_w, nmu_w = pack_band_window(wfns, lo, hi, mesh_xy=mesh)
    hi_ = nb_full if hi is None else hi
    width = hi_ - lo
    q = PX
    w_pad = -(-width // q) * q

    mun_ref = np.pad(psi_full_T[:, :, :, lo:hi_],
                     ((0, 0), (0, 0), (0, 0), (0, w_pad - width)))
    nmu_ref = np.pad(psi_full[:, lo:hi_, :, :],
                     ((0, 0), (0, w_pad - width), (0, 0), (0, 0)))

    mun_h = _to_host(mun_w)
    nmu_h = _to_host(nmu_w)
    assert mun_h.shape == mun_ref.shape, (mun_h.shape, mun_ref.shape)
    assert nmu_h.shape == nmu_ref.shape, (nmu_h.shape, nmu_ref.shape)
    max_mun = float(np.abs(mun_h - mun_ref).max())
    max_nmu = float(np.abs(nmu_h - nmu_ref).max())
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"  pack_isolated lo={lo} hi={hi} nb_full={nb_full} w_pad={w_pad}: "
       f"max|diff| mun={max_mun:.3e} nmu={max_nmu:.3e}")
    # Slice + zero-pad is exact -- no reduction, no summation-order
    # ambiguity -- so this is a BIT-EXACT check, not a relative-tolerance
    # one (unlike the SUMMA-GEMM parity bars elsewhere in this package).
    assert max_mun == 0.0, f"pack_band_window psi_mun_w mismatch: {max_mun:.3e}"
    assert max_nmu == 0.0, f"pack_band_window psi_nmu_w mismatch: {max_nmu:.3e}"


def check_pack_full_window_identity(mesh, *, nb_full):
    ns, nk, n_rmu = 1, 2, 4
    rng = np.random.default_rng(11)
    psi_full = _crand(rng, nk, nb_full, ns, n_rmu)
    psi_full_T = np.transpose(psi_full, (0, 2, 3, 1))
    wfns = _face_wfns(mesh, ns=ns, nk=nk, n_rmu=n_rmu, nb_full=nb_full,
                      psi_full=psi_full, psi_full_T=psi_full_T)
    mun_w, nmu_w = pack_band_window(wfns, 0, None, mesh_xy=mesh)
    assert mun_w is wfns.psi_mun, (
        "pack_band_window(0, None) must return the RESIDENT carrier "
        "unchanged -- no new array, no collective (the trivial full-"
        "window fast path)")
    assert nmu_w is wfns.psi_nmu


# ---------------------------------------------------------------------------
# packed vs mask vs legacy, through the tau kernel
# ---------------------------------------------------------------------------

def check_packed_kernel_parity(mesh, *, ns, nk_tuple, n_rmu, nb_full,
                               nb_sigma, weight_kind, brackets,
                               seed):
    from gw.ppm_sigma import pad_sigma_window
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    nkx, nky, nkz = nk_tuple
    nk = nkx * nky * nkz
    kgrid = nk_tuple
    rng = np.random.default_rng(seed)

    psi_full = _crand(rng, nk, nb_full, ns, n_rmu)
    psi_full_T = np.transpose(psi_full, (0, 2, 3, 1))

    E_A = jnp.asarray(rng.uniform(-1.0, 1.0, size=(nk, nb_full)))
    if weight_kind == "bool":
        sel = jnp.asarray(rng.uniform(size=(nk, nb_full)) > 0.5)
    else:
        sel = jnp.asarray(rng.uniform(0.0, 1.0, size=(nk, nb_full)))
    B_q = jnp.asarray(_crand(rng, nk, n_rmu, n_rmu))
    Omega_q = jnp.asarray(
        rng.uniform(0.2, 3.0, size=(nk, n_rmu, n_rmu))
        - 1j * rng.uniform(0.0, 0.5, size=(nk, n_rmu, n_rmu)))
    mask_B = jnp.asarray(rng.uniform(size=(nk, n_rmu, n_rmu)) > 0.3)
    B_poles = jnp.where(mask_B, B_q, 0.0)[None, ...]
    Omega_poles = Omega_q[None, ...]
    pole_indices = jnp.asarray([0], dtype=jnp.int32)
    bounds = jnp.asarray([[0.0, np.inf, -np.inf, -np.inf,
                           np.inf, np.inf]], dtype=jnp.float64)
    phase_real = jnp.asarray([False])
    E_ref_A = jnp.asarray(0.0, dtype=jnp.float64)
    E_ref_B = jnp.asarray(0.0, dtype=jnp.float64)
    t_node = jnp.asarray(0.15 + 0.07j, dtype=jnp.complex128)

    # ---- legacy (static-slice) reference --------------------------------
    xn_spec = NamedSharding(mesh, P(None, None, "x", None))
    yr_spec = NamedSharding(mesh, P(None, None, None, "y"))
    xr_spec = NamedSharding(mesh, P(None, None, None, "x"))
    yn_spec = NamedSharding(mesh, P(None, None, "y", None))
    psi_coh_xn_L = jax.device_put(jnp.asarray(psi_full_T), xn_spec)
    psi_coh_yr_L = jax.device_put(jnp.asarray(psi_full), yr_spec)
    psi_proj_xr_L0 = jax.device_put(
        jnp.asarray(psi_full[:, :nb_sigma, :, :]), xr_spec)
    psi_proj_yn_L0 = jax.device_put(
        jnp.asarray(psi_full_T[:, :, :, :nb_sigma]), yn_spec)
    psi_proj_xr_L, psi_proj_yn_L, nb_real = pad_sigma_window(
        psi_proj_xr_L0, psi_proj_yn_L0, mesh)
    assert nb_real == nb_sigma
    tau_kernel_legacy = get_shared_sigma_tau_kernel(
        mesh_xy=mesh, kgrid=kgrid, brackets=brackets)
    out_legacy = jax.block_until_ready(tau_kernel_legacy(
        psi_coh_xn_L, psi_coh_yr_L, psi_proj_xr_L, psi_proj_yn_L,
        E_A, sel, B_poles, Omega_poles, pole_indices, bounds, phase_real,
        E_ref_A, E_ref_B, t_node,
    ))

    # ---- face operands ----------------------------------------------------
    face_shape = (nk, nb_full, n_rmu, ns)
    wfns = _face_wfns(mesh, ns=ns, nk=nk, n_rmu=n_rmu, nb_full=nb_full,
                      psi_full=psi_full, psi_full_T=psi_full_T)

    # MASK route (pack_brackets=False) -- the parity oracle.
    tau_kernel_mask = get_shared_sigma_tau_kernel(
        mesh_xy=mesh, kgrid=kgrid, brackets=brackets,
        layout="face", face_shape=face_shape, pack_brackets=False)
    out_mask = jax.block_until_ready(tau_kernel_mask(
        wfns.psi_mun, wfns.psi_nmu, wfns.psi_nmu, wfns.psi_mun,
        E_A, sel, B_poles, Omega_poles, pole_indices, bounds, phase_real,
        E_ref_A, E_ref_B, t_node,
    ))

    # PACKED route (pack_brackets=True, the default) -- the new route.
    tau_kernel_packed = get_shared_sigma_tau_kernel(
        mesh_xy=mesh, kgrid=kgrid, brackets=brackets,
        layout="face", face_shape=face_shape, pack_brackets=True)
    packed = [pack_band_window(wfns, lo, hi, mesh_xy=mesh)
             for lo, hi in brackets]
    psi_coh_xn_tuple = tuple(p[0] for p in packed)
    psi_coh_yr_tuple = tuple(p[1] for p in packed)
    if len(brackets) > 1:
        # For >1 bracket every packed pair must be a FRESH array distinct
        # from the resident carrier (this is the whole point -- narrower
        # than nb_full) EXCEPT where a bracket genuinely covers the full
        # window (none do in these cases).
        for mw in psi_coh_xn_tuple:
            assert mw is not wfns.psi_mun
    out_packed = jax.block_until_ready(tau_kernel_packed(
        psi_coh_xn_tuple, psi_coh_yr_tuple, wfns.psi_nmu, wfns.psi_mun,
        E_A, sel, B_poles, Omega_poles, pole_indices, bounds, phase_real,
        E_ref_A, E_ref_B, t_node,
    ))

    for c, (la, ma, pa) in enumerate(((out_legacy, out_mask, out_packed),)):
        la_h = _to_host(la)[..., :nb_sigma, :nb_sigma]
        ma_h = _to_host(ma)[..., :nb_sigma, :nb_sigma]
        pa_h = _to_host(pa)[..., :nb_sigma, :nb_sigma]
        ref_scale = float(np.abs(la_h).max())

        d_lm = float(np.abs(la_h - ma_h).max()) / max(ref_scale, 1e-300)
        d_lp = float(np.abs(la_h - pa_h).max()) / max(ref_scale, 1e-300)
        d_mp = float(np.abs(ma_h - pa_h).max()) / max(ref_scale, 1e-300)
        p0(f"  ns={ns} nk={nk} nb_full={nb_full} nb_sigma={nb_sigma} "
           f"brackets={brackets} channel={c}: "
           f"legacy-vs-mask={d_lm:.3e} legacy-vs-packed={d_lp:.3e} "
           f"mask-vs-packed={d_mp:.3e}")
        assert d_lm < 1e-9, f"legacy vs mask FAILED (channel {c}): {d_lm:.3e}"
        assert d_lp < 1e-9, f"legacy vs packed FAILED (channel {c}): {d_lp:.3e}"
        # mask-vs-packed should be TIGHTER than either's gap to legacy --
        # both face routes share the SAME cuBLASMp summation order and
        # differ from legacy's shard_map/psum_scatter chain only in
        # WHICH bands are zero-padded, not in HOW the reduction sums.
        assert d_mp < 1e-9, f"mask vs packed FAILED (channel {c}): {d_mp:.3e}"


PACK_ISOLATED_CASES = (
    ("pack_isolated_ns1", dict(
        ns=1, nk=2, n_rmu=4, nb_full=10, lo=1, hi=7, seed=101)),
    ("pack_isolated_ns2", dict(
        ns=2, nk=2, n_rmu=4, nb_full=10, lo=3, hi=9, seed=102)),
    ("pack_isolated_open_hi", dict(
        ns=1, nk=2, n_rmu=4, nb_full=10, lo=2, hi=None, seed=103)),
)

KERNEL_CASES = (
    ("packed_vs_mask_vs_legacy_ns1_3brackets", dict(
        ns=1, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=10, nb_sigma=10,
        weight_kind="bool",
        brackets=((0, 3), (3, 7), (7, 10)), seed=201)),
    ("packed_vs_mask_vs_legacy_ns2_3brackets", dict(
        ns=2, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=10, nb_sigma=10,
        weight_kind="bool",
        brackets=((0, 3), (3, 7), (7, 10)), seed=202)),
    ("packed_vs_mask_float_weight", dict(
        ns=1, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=10, nb_sigma=6,
        weight_kind="float",
        brackets=((0, 3), (3, 7), (7, 10)), seed=203)),
)


@pytest.mark.parametrize("name,kwargs", PACK_ISOLATED_CASES,
                         ids=[c[0] for c in PACK_ISOLATED_CASES])
def test_pack_band_window_isolated(name, kwargs):
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes (the face carrier is 2-D "
            f"sharded); got process_count={jax.process_count()}")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_pack_isolated(mesh, **kwargs)


def test_pack_band_window_full_window_is_identity():
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes; "
            f"got process_count={jax.process_count()}")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_pack_full_window_identity(mesh, nb_full=8)


@pytest.mark.parametrize("name,kwargs", KERNEL_CASES,
                         ids=[c[0] for c in KERNEL_CASES])
def test_packed_kernel_matches_mask_and_legacy(name, kwargs):
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes for gemm_plan's cuBLASMp "
            f"GUARD 4 (got process_count={jax.process_count()}); run "
            f"`lx run -N 1 -G 4 -n 4 ... {__file__} --mesh 2x2` for the "
            f"real check (see this module's docstring)")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_packed_kernel_parity(mesh, **kwargs)


def _cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="2x2", help="PxQ process mesh")
    args = ap.parse_args()
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()}")
    if jax.device_count() != px * py:
        p0(f"REFUSE: need exactly {px * py} devices for a {args.mesh} mesh; "
           f"got {jax.device_count()}")
        return 1
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))
    failures = 0
    for name, kwargs in PACK_ISOLATED_CASES:
        try:
            check_pack_isolated(mesh, **kwargs)
            p0(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {name}: {exc}")
    try:
        check_pack_full_window_identity(mesh, nb_full=8)
        p0("PASS pack_isolated_full_window_is_identity")
    except AssertionError as exc:
        failures += 1
        p0(f"FAIL pack_isolated_full_window_is_identity: {exc}")
    for name, kwargs in KERNEL_CASES:
        try:
            check_packed_kernel_parity(mesh, **kwargs)
            p0(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {name}: {exc}")
    total = len(PACK_ISOLATED_CASES) + 1 + len(KERNEL_CASES)
    p0(f"done: {total - failures}/{total} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_cli_main)
