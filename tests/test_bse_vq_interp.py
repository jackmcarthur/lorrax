"""Gates for ``bse.vq_interp`` — the arbitrary-Q exchange-tile backend.

Fixture: the MoS2 3×3 640-centroid COHSEX fixture (full-BZ ζ storage) —
the exact dataset the reference implementation's pinned acceptance numbers
were measured on (``REFERENCE_arbitrary_q_vq.py`` / arbitrary_q_bse.md
§13.6).  Skipped when the sandbox fixture is absent (tests stay runnable
on a bare checkout; the machine-level gates + nulls inside vq_interp are
fixture-agnostic and re-run wherever the pipeline runs).

1. **Machine gates + nulls** (hard): recon round-trip, makeVq-vs-disk,
   X^H X == C_q, kernel-split identity, exact-stencil nulls, F-rebuild.
2. **LOO accuracy** vs the reference e2e thresholds (baseline B med
   1.409e-2 / max 3.553e-2, exc 0.642/2.542 meV; thresholds ~1.5× above —
   a formalism regression moves B by 10-100×).
3. **Jitted evaluator**: parity with the host evaluator at on-grid and
   off-grid Q (≤1e-9), output sharding ``P('x','y')``, and the compile
   census — ONE jit cache entry serves every Q (per-q-recompile lesson).

All 1-GPU (no 16-GPU gating).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

FX = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/"
      "05_lorrax_cohsex_native/tmp")

pytestmark = pytest.mark.skipif(
    not os.path.exists(f"{FX}/zeta_q.h5"),
    reason="MoS2 3x3 640-centroid fixture not present")

THRESH = {"B_med": 2.2e-2, "B_max": 5.5e-2, "exc_med": 1.00, "exc_max": 4.00}


@pytest.fixture(scope="module")
def vq_state():
    harness.skip_unless_gpu(pytest)
    from bse import vq_interp as vqi
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1),
                axis_names=("x", "y"))
    zx = vqi.load_zeta_coarse(f"{FX}/isdf_tensors_640.h5", f"{FX}/zeta_q.h5")
    C_q = vqi.build_cq(zx, mesh)
    vqi.run_gates(zx, C_q)                 # hard machine-level gates
    with mesh:
        prep = vqi.prepare_coarse(zx, C_q, mesh)
    des = vqi.lr_design_blocks(zx, prep)
    coeffs = vqi.fit_lr_model(des)
    vqi.run_nulls(zx, prep, des, coeffs)   # hard machine-level nulls
    return dict(vqi=vqi, mesh=mesh, zx=zx, prep=prep, des=des, coeffs=coeffs)


@pytest.mark.gpu
def test_loo_accuracy_vs_reference_thresholds(vq_state):
    """Honest LOO at every coarse q: B + exciton swap vs the stored-fit
    truth, within the reference e2e thresholds."""
    vqi, zx = vq_state["vqi"], vq_state["zx"]
    prep, des = vq_state["prep"], vq_state["des"]
    Bs, excs = [], []
    for q0 in range(zx["nq"]):
        train = [q for q in range(zx["nq"]) if q != q0]
        C_loo = vqi.fit_lr_model(des, exclude=q0)
        Vp = vqi.eval_vq_host(zx, prep, des, C_loo, zx["qfr"][q0], train=train)
        x = vqi.gap_window_pairs(zx, q0)
        B_true = vqi.b_block(x, vqi.make_vq(zx, zx["ZG"][q0], q0))
        Bs.append(vqi.relF(vqi.b_block(x, Vp), B_true))
        D, Hdir = vqi.build_hdir(zx, q0)
        dev = np.abs(vqi.exciton_evs(zx, D, Hdir, vqi.b_block(x, Vp))
                     - vqi.exciton_evs(zx, D, Hdir, B_true))
        excs.append(float(np.max(dev) * vqi.RY2MEV))
    assert float(np.median(Bs)) <= THRESH["B_med"]
    assert float(np.max(Bs)) <= THRESH["B_max"]
    assert float(np.median(excs)) <= THRESH["exc_med"]
    assert float(np.max(excs)) <= THRESH["exc_max"]


@pytest.mark.gpu
def test_eval_jit_parity_sharding_census(vq_state):
    """Jitted evaluator == host evaluator; tile sharded P('x','y'); ONE
    compile serves on-grid and off-grid Q alike."""
    vqi, zx, mesh = vq_state["vqi"], vq_state["zx"], vq_state["mesh"]
    prep, des, coeffs = vq_state["prep"], vq_state["des"], vq_state["coeffs"]
    with mesh:
        eval_jit = vqi.make_eval_vq(zx, prep, des, mesh, None)
        pinvF = jnp.asarray(vqi.stencil_pinv(zx["qfr"], vqi.stencil_r7(zx)))
        cpk = vqi.pack_coeffs(des, coeffs)
        tests = [zx["qfr"][1], np.array([0.11, 0.07, 0.0]),
                 np.array([1.0 / 6.0, 0.0, 0.0])]
        for qf in tests:
            Vj = eval_jit(jnp.asarray(qf), prep["V_SRc"], pinvF, cpk)
            assert Vj.sharding.spec == P("x", "y")
            Vh = vqi.eval_vq_host(zx, prep, des, coeffs, qf)
            assert vqi.relF(np.asarray(jax.device_get(Vj)), Vh) <= 1e-9
        assert eval_jit._cache_size() == 1


@pytest.mark.gpu
def test_eval_on_grid_is_stencil_exact(vq_state):
    """With a grid-RESOLVING R set the trig stencil is a delta: the SR
    part returns a training q's own cleaned tile exactly (the reference
    run_nulls exact-stencil identity, pytest-visible).  The production
    nR7 stencil is a least-squares fit — NOT a delta at full training —
    so exactness is asserted on the full R lattice only."""
    vqi, zx = vq_state["vqi"], vq_state["zx"]
    prep, des, coeffs = vq_state["prep"], vq_state["des"], vq_state["coeffs"]
    q0 = 2
    kg = zx["kgrid"]
    Rfull = np.array([[i - kg[0] // 2, j - kg[1] // 2, 0]
                      for i in range(kg[0]) for j in range(kg[1])])
    pinvF = vqi.stencil_pinv(zx["qfr"], Rfull)
    w = np.exp(-2j * np.pi * (zx["qfr"][q0] @ Rfull.T)) @ pinvF
    V_SR = np.tensordot(w, prep["V_SRc_np"], axes=(0, 0))
    assert vqi.relF(V_SR, prep["V_SRc_np"][q0]) <= 1e-9
    V = vqi.eval_vq_host(zx, prep, des, coeffs, zx["qfr"][q0])
    assert np.isfinite(np.linalg.norm(V))
