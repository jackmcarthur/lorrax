"""Algebra parity: the shared dynamic Sigma_c(tau) kernel
(``gw.ppm_tau_kernel.get_shared_sigma_tau_kernel``), ``layout='legacy'`` vs
``layout='face'``, on the SAME synthetic operands, real multi-rank CUDA.

Companion to ``tests/test_isdf_cq_face_parity.py``/``test_isdf_zq_face_
parity.py`` (feat/dynamic-sigma-face-port-2026-08-22): those gate the ISDF
fit's own face port; this one gates the dynamic PPM/MPA Sigma pipeline's
face port (``docs`` register: ``low_mem_bands_dynamic_ppm_unported``).
Same reason for real multi-process CUDA -- ``layout='face'`` needs a real
``distrib_la.gemm_plan`` (GUARD 4: cuBLASMp refuses whenever
``jax.process_count() < mesh.devices.size``) -- and the same CLI-vs-pytest
dual entry point.

    lx run -N 1 -G 4 -n 4 bash <wrapper.sh> \\
        tests/test_ppm_tau_kernel_face_parity.py --mesh 2x2

Under plain pytest (one process) every case SKIPS rather than failing --
it names exactly why (process_count) rather than reporting a silent pass
(TASTE.md, "a check that cannot fail is not evidence").

What is and is not checked
---------------------------
This is an ALGEBRA-PARITY test, not a physics test: B_q/Omega_q/E_A/mask_A
are synthetic random operands with no PPM-fit provenance.  What it proves
is that ``layout='face'`` computes the SAME sigma^tau(k,m,n) tile the
established ``layout='legacy'`` body does, given the SAME logical psi and
the SAME physics operands -- not that either is physically correct PPM
Sigma_c (that is what the k6_c600 end-to-end leg checks, against a
BerkeleyGW/legacy-gn_ppm reference).

Five cases cover the matrix the port's own design points name:
  * ``identity_ns1``/``identity_ns2`` -- the boolean band-identity mask
    path (``build_G_tau``'s ``mask=`` seam), ns=1 and ns=2 (the GEMM-seam
    spin-merge order), sigma window == nb_full (mesh-divisible, the
    trivial/no-pad case).
  * ``real_weight_ns1_nondivisible`` -- a FLOAT ``band_weight`` (fractional
    occupation-shaped, not 0/1) through ``build_G_tau``'s ``band_weight=``
    seam, AND a sigma window that does NOT divide the mesh (nb_sigma=5 on
    a 2x2 mesh) -- legacy's ``pad_sigma_window``/``strip_sigma_window``
    round trip vs face's full-nb_full-then-late-slice design (this
    shared executor's full complex carrier.
  * ``real_weight_ns2`` -- ns=2 through the float band_weight seam, the
    single-complex-chain plan.
  * ``brackets_ns1`` -- a genuine THREE-bracket plan (the band-
    extrapolation shape), exercising ``_bracketed_face``'s mask-
    intersection loop for real (not the single-bracket ((0,None),) shape
    every other case here uses, which is a no-op intersection).
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
from gw.ppm_sigma import pad_sigma_window

PX = PY = 2


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


#: (ns, nk_tuple, n_rmu, nb_full, nb_sigma, weight_kind, brackets,
#:  seed) -- see module docstring for what each case exercises.
#: weight_kind: 'bool' -> build_G_tau's mask= seam; 'float' -> band_weight=.
_CASES = (
    ("identity_ns1", dict(
        ns=1, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8, nb_sigma=8,
        weight_kind="bool", brackets=((0, None),), seed=1)),
    ("identity_ns2", dict(
        ns=2, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8, nb_sigma=8,
        weight_kind="bool", brackets=((0, None),), seed=2)),
    ("real_weight_ns1_nondivisible", dict(
        ns=1, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8, nb_sigma=5,
        weight_kind="float", brackets=((0, None),), seed=3)),
    ("real_weight_ns2", dict(
        ns=2, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8, nb_sigma=8,
        weight_kind="float", brackets=((0, None),), seed=4)),
    ("brackets_ns1", dict(
        ns=1, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8, nb_sigma=8,
        weight_kind="bool",
        brackets=((0, 3), (3, 6), (6, None)), seed=5)),
)


def _to_host(x, mesh):
    from jax.experimental import multihost_utils as mhu
    del mesh
    return np.asarray(mhu.process_allgather(x, tiled=True))


def check_tau_kernel_face_parity(mesh, *, ns, nk_tuple, n_rmu, nb_full,
                                 nb_sigma, weight_kind, brackets,
                                 seed):
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    nkx, nky, nkz = nk_tuple
    nk = nkx * nky * nkz
    kgrid = nk_tuple
    rng = np.random.default_rng(seed)

    # ---- one logical psi, two axis orders (module docstring) ---------
    # psi_full[k, n(band), s, mu]: the (n,s,mu) order -- IS psi_xr/psi_yr
    # (legacy) and psi_nmu (face), unchanged.
    psi_full = _crand(rng, nk, nb_full, ns, n_rmu)
    # (k, s, mu, n) order -- IS psi_xn/psi_yn (legacy) and psi_mun (face).
    psi_full_T = np.transpose(psi_full, (0, 2, 3, 1))

    # ---- shared, layout-agnostic physics operands ---------------------
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

    tau_kernel_legacy = get_shared_sigma_tau_kernel(
        mesh_xy=mesh, kgrid=kgrid, brackets=brackets)

    # ---- legacy operands ----------------------------------------------
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

    out_legacy = jax.block_until_ready(tau_kernel_legacy(
        psi_coh_xn_L, psi_coh_yr_L, psi_proj_xr_L, psi_proj_yn_L,
        E_A, sel, B_poles, Omega_poles, pole_indices, bounds, phase_real,
        E_ref_A, E_ref_B, t_node,
    ))

    # ---- face operands ---------------------------------------------------
    mun_spec = NamedSharding(mesh, P(None, None, "x", "y"))
    nmu_spec = NamedSharding(mesh, P(None, "x", None, "y"))
    psi_mun = jax.device_put(jnp.asarray(psi_full_T), mun_spec)
    psi_nmu = jax.device_put(jnp.asarray(psi_full), nmu_spec)
    face_shape = (nk, nb_full, n_rmu, ns)

    # pack_brackets=False: this module gates the MASK bracket loop
    # specifically (module docstring, ``brackets_ns1``'s own case
    # comment -- "exercising _bracketed_face's mask-intersection loop for
    # real").  ``pack_brackets`` now defaults True at the multi-bracket
    # kernel factory (2026-08-23), which would otherwise silently route
    # this SAME case through the packed loop instead -- a different
    # calling convention (tuples, not bare arrays) that this test's own
    # operands below do not use.  Pin the mask route explicitly rather
    # than ride whatever the current default happens to be; the packed
    # route has its own dedicated gate,
    # tests/test_pack_band_window.py, which cross-checks packed vs mask
    # vs legacy on the identical three-bracket shape.
    tau_kernel_face = get_shared_sigma_tau_kernel(
        mesh_xy=mesh, kgrid=kgrid, brackets=brackets,
        layout="face", face_shape=face_shape, pack_brackets=False)

    out_face = jax.block_until_ready(tau_kernel_face(
        psi_mun, psi_nmu, psi_nmu, psi_mun,
        E_A, sel, B_poles, Omega_poles, pole_indices, bounds, phase_real,
        E_ref_A, E_ref_B, t_node,
    ))

    for c, (la, fa) in enumerate(((out_legacy, out_face),)):
        la_h = _to_host(la, mesh)[..., :nb_sigma, :nb_sigma]
        fa_h = _to_host(fa, mesh)[..., :nb_sigma, :nb_sigma]
        assert la_h.shape == fa_h.shape, (
            f"channel {c}: shape mismatch legacy={la_h.shape} "
            f"face={fa_h.shape}")
        absdiff = np.abs(la_h - fa_h)
        ref_scale = float(np.abs(la_h).max())
        max_abs = float(absdiff.max())
        max_rel = max_abs / max(ref_scale, 1e-300)
        p0(f"  ns={ns} nk={nk} nb_full={nb_full} nb_sigma={nb_sigma} "
           f"brackets={brackets} channel={c}: "
           f"max|diff|={max_abs:.3e} (ref scale {ref_scale:.3e}) "
           f"max|rel diff|={max_rel:.3e}")
        # Engine-parity bar (this repo's convention, same as
        # test_isdf_cq_face_parity's): relative, never bit-exact -- a
        # SUMMA-distributed cuBLASMp GEMM and a shard_map/psum_scatter
        # chain sum in different orders.
        assert max_rel < 1e-9, (
            f"tau kernel layout='face' vs 'legacy' parity FAILED "
            f"(channel {c}): max relative diff {max_rel:.3e}")


@pytest.mark.parametrize("name,kwargs", _CASES, ids=[c[0] for c in _CASES])
def test_tau_kernel_face_matches_legacy(name, kwargs):
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes for gemm_plan's cuBLASMp "
            f"GUARD 4 (got process_count={jax.process_count()}); run "
            f"`lx run -N 1 -G 4 -n 4 ... {__file__} --mesh 2x2` for the "
            f"real check (see this module's docstring)")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_tau_kernel_face_parity(mesh, **kwargs)


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
    for name, kwargs in _CASES:
        try:
            check_tau_kernel_face_parity(mesh, **kwargs)
            p0(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {name}: {exc}")
    p0(f"done: {len(_CASES) - failures}/{len(_CASES)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_cli_main)
