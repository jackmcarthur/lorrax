"""Algebra parity: ``isdf.core.c_q_from_psi_sm(layout='legacy')`` vs
``layout='face'`` on the SAME synthetic ψ data, real multi-rank CUDA.

``layout='face'`` needs a real :func:`distrib_la.gemm_plan`, which needs
REAL multi-process CUDA (GUARD 4: cuBLASMp/cuSOLVERMp refuse whenever
``jax.process_count() < mesh.devices.size`` -- see
``services/distrib_la/matmul_plan.py``).  A plain ``pytest`` invocation is
one process with (at most) emulated/visible devices, so the real check
below only ever runs under the CLI multi-rank invocation; this mirrors
``services/distrib_la/tests/test_distrib_la_multiproc.py``'s own
ONE-SET-OF-CHECK-BODIES-TWO-CALLERS convention rather than inventing a
second one.

    lx run -N 1 -G 4 -n 4 bash <wrapper.sh> \\
        tests/test_isdf_cq_face_parity.py --mesh 2x2

(``<wrapper.sh>`` -- any script that pins PYTHONPATH to this checkout and
LORRAX_FFI_SO before ``exec python3 -u "$@"`` -- is required; see AGENTS.md's
`lx run` section.)  Real 4-rank result, 2026-08-22 (this file's introducing commit):
3/3 cases PASS, max relative diff 2.1e-15 (float64 noise from a different
summation order -- SUMMA-distributed cuBLASMp GEMM vs a rank-local
``jnp.einsum`` -- not a discrepancy), covering ns=1, ns=2 (the GEMM-seam
spin-merge order) and an L/R band window whose lower edges also differ
(not just its upper edge) from ``band_range_full``'s own origin.  See
``docs/architecture/zeta_fit_face_psi_cct.md``.

**γ̃ vertex extension (2026-08-23, feat/transverse-zeta-face-2026-08-23)**:
``_GAMMA_CASES`` adds all 15 non-identity ``(mu_L, nu_L)`` Lorentz-index
pairs at ns=4 -- the DISCRIMINATING cases (an identity vertex passes
trivially: `gamma_apply` is a no-op, so the endpoint-application code below
is never actually exercised by the three cases above).  These gate
``isdf.core._c_q_face``'s endpoint-application vertex insertion against
``_c_q_legacy``'s existing post-IFFT ``gamma_double_contract`` -- same
tolerance (``1e-10`` relative), same mechanism as the identity cases.

Under plain pytest (one process), the cell below SKIPS rather than
failing -- it is not lying about the property it did not check (TASTE.md,
"a check that cannot fail is not evidence" family): it names exactly why
(process_count) rather than reporting a silent pass.
"""
from __future__ import annotations

import os
import sys

# CLI multi-rank mode uses the same runtime boundary as production drivers
# -- mirrors test_distrib_la_multiproc.py exactly (init BEFORE importing
# jax/distrib_la, guarded by __main__, so a plain pytest collection never
# pays or triggers this).
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

from isdf.core import c_q_from_psi_sm
from distrib_la import gemm_plan
from common.gamma_matrices import gamma_perm_phase

PX = PY = 2

#: (ns, nk_tuple, n_rmu, nb_full, l_range, r_range, seed) -- see
#: check_cq_face_parity's docstring for what each case exercises.
_CASES = (
    ("ns1_asym_upper", dict(ns=1, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8,
                             l_range=(0, 5), r_range=(0, 8), seed=1)),
    ("ns2_spinor",     dict(ns=2, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8,
                             l_range=(0, 5), r_range=(0, 8), seed=2)),
    ("ns1_asym_lower", dict(ns=1, nk_tuple=(1, 2, 1), n_rmu=4, nb_full=8,
                             l_range=(2, 8), r_range=(0, 8), seed=3)),
)

#: γ̃-VERTEX cases (2026-08-23, feat/transverse-zeta-face-2026-08-23) — the
#: DISCRIMINATING cases the task asks for: identity-vertex agreement (the
#: three cases above, unchanged) proves nothing about a non-identity γ̃, since
#: an identity perm/phase makes the endpoint-application code below a no-op
#: byte-identical to the pre-vertex path.  ns=4 (a genuine bispinor fixture,
#: not merely ns=2), ALL (mu_L, nu_L) pairs in {0,1,2,3}^2 EXCEPT (0,0)
#: (already covered by the identity cases above) -- 15 cases, each its own
#: seed so no two cases share random ψ.
_GAMMA_CASES = tuple(
    (f"gamma_mu{mu_l}_nu{nu_l}",
     dict(ns=4, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8,
          l_range=(0, 5), r_range=(0, 8), seed=100 + 4 * mu_l + nu_l,
          gamma_mu_L=mu_l, gamma_nu_L=nu_l))
    for mu_l in range(4) for nu_l in range(4)
    if not (mu_l == 0 and nu_l == 0)
)


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def check_cq_face_parity(mesh, *, ns, nk_tuple, n_rmu, nb_full, l_range,
                         r_range, seed, gamma_mu_L=0, gamma_nu_L=0):
    """Build the SAME random ψ, feed it to both layouts, diff C_q.

    ``l_range``/``r_range`` need not be mesh-divisible (the face path's
    "weight, don't window" design point -- band_range_left/right are
    sigma-window edges, not mesh-aligned in general); only
    ``full = (min(l0,r0), max(l1,r1))`` must equal ``(0, nb_full)`` here,
    matching ``fit_zeta_to_h5``'s own ``band_range_full`` convention.

    ``gamma_mu_L``/``gamma_nu_L`` (Lorentz index, 0-3): 0 means identity
    (``gamma_L``/``gamma_R=None``, the pre-vertex charge channel); 1-3
    build the ``(perm, phase)`` tuple for γ̃^{mu_L}/γ̃^{nu_L}
    (:func:`common.gamma_matrices.gamma_perm_phase`) and feed the SAME
    tuple to BOTH the legacy and face calls below -- the discriminating
    check this file's own module docstring update explains.
    """
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    nkx, nky, nkz = nk_tuple
    nk = nkx * nky * nkz
    rng = np.random.default_rng(seed)
    # psi_full[k, n(band), s(spin), mu] -- raw, un-conjugated ground truth.
    psi_full = _crand(rng, nk, nb_full, ns, n_rmu)

    # The loader's convention: psi_rmu_Y is raw psi;
    # psi_rmuT_X is ALREADY psi* (conjugated), transposed to (k, mu, n, s).
    psi_rmu_Y_loaded = psi_full
    psi_rmuT_X_loaded = np.conj(psi_full).transpose(0, 3, 1, 2)

    l0, l1 = l_range
    r0, r1 = r_range
    full0, full1 = min(l0, r0), max(l1, r1)
    if (full0, full1) != (0, nb_full):
        raise ValueError("test setup: full range must equal (0, nb_full)")

    # ---- legacy path -------------------------------------------------
    x1_4 = NamedSharding(mesh, P(None, "x", None, None))
    y3_4 = NamedSharding(mesh, P(None, None, None, "y"))
    psi_l_X_d = jax.device_put(
        jnp.asarray(psi_rmuT_X_loaded[:, :, l0:l1, :]), x1_4)
    psi_l_Y_d = jax.device_put(
        jnp.asarray(psi_rmu_Y_loaded[:, l0:l1, :, :]), y3_4)
    psi_r_X_d = jax.device_put(
        jnp.asarray(psi_rmuT_X_loaded[:, :, r0:r1, :]), x1_4)
    psi_r_Y_d = jax.device_put(
        jnp.asarray(psi_rmu_Y_loaded[:, r0:r1, :, :]), y3_4)

    gamma_L = None if gamma_mu_L == 0 else gamma_perm_phase(gamma_mu_L)
    gamma_R = None if gamma_nu_L == 0 else gamma_perm_phase(gamma_nu_L)

    C_legacy = jax.block_until_ready(c_q_from_psi_sm(
        psi_l_X_d, psi_l_Y_d, psi_r_X_d, psi_r_Y_d,
        gamma_L=gamma_L, gamma_R=gamma_R, kgrid=nk_tuple, mesh_xy=mesh,
        layout="legacy"))

    # ---- face path -----------------------------------------------------
    # Exactly gw.gw_init.prepare_isdf_and_wavefunctions's own formula for
    # psi_mun_fresh/psi_nmu_fresh -- not re-derived here.
    psi_mun_np = np.conj(psi_rmuT_X_loaded).transpose(0, 3, 1, 2)  # (k,s,mu,n)
    psi_nmu_np = psi_rmu_Y_loaded                                   # (k,n,s,mu)
    psi_mun_d = jax.device_put(
        jnp.asarray(psi_mun_np), NamedSharding(mesh, P(None, None, "x", "y")))
    psi_nmu_d = jax.device_put(
        jnp.asarray(psi_nmu_np), NamedSharding(mesh, P(None, "x", None, "y")))

    idx = np.arange(nb_full)
    w_l = np.where((idx >= l0) & (idx < l1), 1.0, 0.0)
    w_r = np.where((idx >= r0) & (idx < r1), 1.0, 0.0)

    plan = gemm_plan(mesh, m=n_rmu * ns, k=nb_full, n=n_rmu * ns, nq=nk,
                      dtype=jnp.complex128)
    p0(f"  {plan.describe()}")

    C_face = jax.block_until_ready(c_q_from_psi_sm(
        kgrid=nk_tuple, mesh_xy=mesh, layout="face",
        psi_mun=psi_mun_d, psi_nmu=psi_nmu_d,
        gamma_L=gamma_L, gamma_R=gamma_R,
        weight_l=jnp.asarray(w_l), weight_r=jnp.asarray(w_r), gemm=plan))

    # Both C_legacy/C_face are genuinely multi-process sharded
    # (P(None,'x','y')) -- gather with process_allgather, not device_get.
    from jax.experimental import multihost_utils as mhu
    C_legacy_host = np.asarray(mhu.process_allgather(C_legacy, tiled=True))
    C_face_host = np.asarray(mhu.process_allgather(C_face, tiled=True))

    assert C_legacy_host.shape == C_face_host.shape, (
        f"shape mismatch: legacy={C_legacy_host.shape} "
        f"face={C_face_host.shape}")
    absdiff = np.abs(C_legacy_host - C_face_host)
    ref_scale = float(np.abs(C_legacy_host).max())
    max_abs = float(absdiff.max())
    max_rel = max_abs / max(ref_scale, 1e-300)
    p0(f"  ns={ns} nk={nk} n_rmu={n_rmu} nb_full={nb_full} l={l_range} "
       f"r={r_range} gamma=(mu_L={gamma_mu_L},nu_L={gamma_nu_L}): "
       f"max|diff|={max_abs:.3e} (ref scale {ref_scale:.3e}) "
       f"max|rel diff|={max_rel:.3e}")
    # Engine-parity bar (C8, this repo's convention): relative, never
    # bit-exact -- a distributed SUMMA GEMM and a rank-local einsum are
    # different reductions in a different order.
    assert max_rel < 1e-10, (
        f"c_q_from_psi_sm layout='face' vs 'legacy' parity FAILED: "
        f"max relative diff {max_rel:.3e} (case ns={ns}, l={l_range}, "
        f"r={r_range}, gamma=(mu_L={gamma_mu_L},nu_L={gamma_nu_L}))")


_ALL_CASES = _CASES + _GAMMA_CASES


@pytest.mark.parametrize("name,kwargs", _ALL_CASES, ids=[c[0] for c in _ALL_CASES])
def test_cq_face_layout_matches_legacy(name, kwargs):
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes for gemm_plan's cuBLASMp "
            f"GUARD 4 (got process_count={jax.process_count()}); run "
            f"`lx run -N 1 -G 4 -n 4 ... {__file__} --mesh 2x2` for the "
            f"real check (see this module's docstring)")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_cq_face_parity(mesh, **kwargs)


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
    for name, kwargs in _ALL_CASES:
        try:
            check_cq_face_parity(mesh, **kwargs)
            p0(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {name}: {exc}")
    p0(f"done: {len(_ALL_CASES) - failures}/{len(_ALL_CASES)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_cli_main)
