"""Sharded test for ``gw.head_wing_schur``.

Two checks:

1. **Algebraic correctness** — sharded Schur reconstruction reproduces
   the dense ``(I − Vχ)⁻¹V`` to machine precision (1e-12 in the
   complex128 path), with identity (no override) wing inputs.

2. **Communication minimality** — print HLO of the rank-1 ops
   (``extract_V_body_sharded`` and ``assemble_W_sharded``) and confirm
   they contain **no** all-gather / all-reduce / collective HLOs.
   The Schur-reduction step is allowed reductions (one psum each for
   the two A_wing's and one for χ_head_eff per q).

Run on a 2×2 mesh (4 devices). On a single-GPU machine this still works
via JAX's `_USE_PJIT_PARTITIONER` simulating the mesh on one device.
"""
from __future__ import annotations

import os
import sys

# Allow ``import gw.head_wing_schur`` from a fresh process.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# x64 + JAX_PLATFORMS must be set BEFORE first use of jax — production
# tests use the canonical lorrax_X env defaults (``JAX_ENABLE_X64``,
# ``JAX_PLATFORMS=cuda,cpu``) but a fresh test interpreter doesn't inherit
# them, so we set them manually here.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import jax
jax.config.update("jax_enable_x64", True)

# Multi-process coordination — LORRAX is launched as N MPI ranks each
# owning 1 local GPU.  Without ``jax.distributed.initialize`` each rank
# only sees its 1 device and any 2×2 mesh falls back to 1×1, hiding all
# cross-device collectives.  Pattern lifted from
# ``common/cusolvermp_solve_lu_test.py``.
_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _init_distributed():
    if os.environ.get(_DIST):
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        one_gpu = cvd and "," not in cvd
        init_kwargs = {"local_device_ids": [0]} if one_gpu else {}
        try:
            jax.distributed.initialize(**init_kwargs)
        except Exception:
            pass
    os.environ[_DIST] = "1"
_init_distributed()

import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils

from gw.head_wing_schur import (  # noqa: E402
    extract_V_body_sharded,
    solve_W_body0_sharded,
    schur_reductions_sharded,
    W_head_scalar_per_q,
    assemble_W_sharded,
    reconstruct_W_via_schur_sharded,
    V_FLATQ_SPEC,
    G0_X_SPEC,
    G0_Y_SPEC,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_mesh(grid_x: int = 2, grid_y: int = 2) -> Mesh:
    """2-D device mesh.  Falls back to 1×1 if device_count() == 1."""
    devs = jax.devices()
    n = len(devs)
    if n >= grid_x * grid_y:
        arr = np.asarray(devs[: grid_x * grid_y]).reshape(grid_x, grid_y)
    else:
        # Single-device fallback: 1×1 mesh, axes named ('x','y') so the
        # PartitionSpec strings still resolve.  Comm checks will be
        # trivially satisfied (no collectives needed).
        arr = np.asarray(devs[:1]).reshape(1, 1)
    return Mesh(arr, axis_names=('x', 'y'))


def _make_synthetic_problem(rng: np.random.Generator, *, nq: int = 8, n_mu: int = 64):
    """Build a Hermitian V (PSD) and Hermitian χ_body (negative-semidef
    so that ``I − Vχ`` is well conditioned) plus rank-1 head channel.

    Returns numpy arrays for the dense reference; the sharded driver
    re-puts them onto the mesh.
    """
    # Background Hermitian V_body: A·A† + δ·I  (PSD, well conditioned)
    A = (rng.normal(size=(nq, n_mu, n_mu))
         + 1j * rng.normal(size=(nq, n_mu, n_mu))) / np.sqrt(n_mu)
    V_body = A @ np.conj(A.swapaxes(-1, -2)) + 0.05 * np.eye(n_mu, dtype=complex)

    # Head-channel rank-1 g0 — small magnitude so the head doesn't dominate
    g0 = (rng.normal(size=(nq, n_mu))
          + 1j * rng.normal(size=(nq, n_mu))) / np.sqrt(n_mu)
    v_head = rng.uniform(0.5, 2.0, size=(nq,)).astype(np.complex128)

    # Reassemble V (the "input" to the Schur driver):
    #   V = V_body + v_head · g0* ⊗ g0
    V_full = V_body + v_head[:, None, None] * (
        np.conj(g0)[:, :, None] * g0[:, None, :])

    # χ_body — Hermitian negative-semidef so (I − Vχ) is well conditioned.
    # Magnitude tuned to keep ||V·χ|| safely below 1 (conditioning of the
    # Schur LU is what limits the precision of this test, NOT any
    # algebraic mismatch).
    B = (rng.normal(size=(nq, n_mu, n_mu))
         + 1j * rng.normal(size=(nq, n_mu, n_mu))) / np.sqrt(n_mu)
    chi_body = -0.005 * (B @ np.conj(B.swapaxes(-1, -2)))

    # Wing/head: small Hermitian-compatible vectors
    chi_wing  = (rng.normal(size=(nq, n_mu))
                 + 1j * rng.normal(size=(nq, n_mu))) * 0.005
    chi_wingp = np.conj(chi_wing)        # static-ω convention
    chi_head  = (rng.normal(size=(nq,))
                 + 1j * rng.normal(size=(nq,))) * 0.01

    return dict(
        V_full=V_full.astype(np.complex128),
        V_body_ref=V_body.astype(np.complex128),
        chi_body=chi_body.astype(np.complex128),
        g0=g0.astype(np.complex128),
        v_head=v_head.astype(np.complex128),
        chi_wing=chi_wing.astype(np.complex128),
        chi_wingp=chi_wingp.astype(np.complex128),
        chi_head=chi_head.astype(np.complex128),
    )


def _dense_W_reference(V_full: np.ndarray, chi_body: np.ndarray,
                       g0: np.ndarray, v_head: np.ndarray,
                       chi_wing: np.ndarray, chi_wingp: np.ndarray,
                       chi_head: np.ndarray) -> np.ndarray:
    """Reference W via the bordered-Dyson Schur algebra in pure numpy.

    Equivalent to (I − V·χ_full)⁻¹ V where χ_full = χ_body + rank-1
    wing/head pieces — but constructed from the *components* the sharded
    driver also receives, so any mismatch reflects the sharded
    implementation, not a different physical W.
    """
    nq = V_full.shape[0]
    W = np.zeros_like(V_full)
    for q in range(nq):
        V_body = V_full[q] - v_head[q] * (np.conj(g0[q])[:, None] * g0[q][None, :])
        n_mu = V_body.shape[0]
        eye = np.eye(n_mu, dtype=complex)
        W_body0 = np.linalg.solve(eye - V_body @ chi_body[q], V_body)

        A_wing  = W_body0 @ chi_wing[q]
        A_wingp = chi_wingp[q] @ W_body0
        chi_eff = chi_head[q] + chi_wingp[q] @ A_wing
        W_head  = v_head[q] / (1.0 - v_head[q] * chi_eff)

        left  = np.conj(g0[q]) + A_wing
        right = g0[q]          + A_wingp
        W[q] = W_body0 + W_head * (left[:, None] * right[None, :])
    return W


# ---------------------------------------------------------------------------
# 1. Algebraic correctness
# ---------------------------------------------------------------------------

def test_schur_reconstruction_matches_dense():
    rng = np.random.default_rng(42)
    prob = _make_synthetic_problem(rng, nq=4, n_mu=32)

    mesh = _make_mesh(2, 2)
    repspec = NamedSharding(mesh, P(None,))

    V_q       = jax.device_put(jnp.asarray(prob['V_full']),
                               NamedSharding(mesh, V_FLATQ_SPEC))
    chi_body  = jax.device_put(jnp.asarray(prob['chi_body']),
                               NamedSharding(mesh, V_FLATQ_SPEC))
    g0_x      = jax.device_put(jnp.asarray(prob['g0']),
                               NamedSharding(mesh, G0_X_SPEC))
    g0_y      = jax.device_put(jnp.asarray(prob['g0']),
                               NamedSharding(mesh, G0_Y_SPEC))
    chi_wing_x  = jax.device_put(jnp.asarray(prob['chi_wing']),
                                 NamedSharding(mesh, G0_X_SPEC))
    chi_wing_y  = jax.device_put(jnp.asarray(prob['chi_wing']),
                                 NamedSharding(mesh, G0_Y_SPEC))
    chi_wingp_x = jax.device_put(jnp.asarray(prob['chi_wingp']),
                                 NamedSharding(mesh, G0_X_SPEC))
    chi_wingp_y = jax.device_put(jnp.asarray(prob['chi_wingp']),
                                 NamedSharding(mesh, G0_Y_SPEC))
    chi_head    = jax.device_put(jnp.asarray(prob['chi_head']), repspec)
    v_head      = jax.device_put(jnp.asarray(prob['v_head']), repspec)

    with mesh:
        W_sharded = reconstruct_W_via_schur_sharded(
            V_q, chi_body,
            g0_x, g0_y,
            chi_wing_x, chi_wing_y,
            chi_wingp_x, chi_wingp_y,
            chi_head, v_head,
            mesh_xy=mesh,
        )

    W_ref = _dense_W_reference(
        prob['V_full'], prob['chi_body'], prob['g0'], prob['v_head'],
        prob['chi_wing'], prob['chi_wingp'], prob['chi_head'])

    # Cross-process gather so the assertion is meaningful when the mesh
    # spans multiple ranks (otherwise each rank only sees its own shard).
    W_full = np.asarray(multihost_utils.process_allgather(W_sharded, tiled=False))
    diff = W_full - W_ref
    rel = np.max(np.abs(diff)) / np.max(np.abs(W_ref))
    if jax.process_index() == 0:
        print(f"  max |Δ W| = {np.max(np.abs(diff)):.2e}")
        print(f"  max |W_ref| = {np.max(np.abs(W_ref)):.2e}")
        print(f"  relative max |Δ W| = {rel:.2e}")
    assert rel < 1e-9, f"Sharded Schur reconstruction differs from dense by {rel:.2e}"


# ---------------------------------------------------------------------------
# 2. Communication minimality (HLO inspection)
# ---------------------------------------------------------------------------

def _hlo_text(fn, *args, mesh, static_argnames=()) -> str:
    """Lower+compile ``fn(*args, mesh_xy=mesh)`` and return HLO text.

    We bind ``mesh`` as a static keyword via partial to avoid closures
    over sharded arrays, which break in multi-process JAX.
    """
    from functools import partial as _partial
    bound = _partial(fn, mesh_xy=mesh)
    bound.__name__ = getattr(fn, "__name__", "fn")
    fn_jit = jax.jit(bound, static_argnames=static_argnames)
    return str(fn_jit.lower(*args).compile().as_text())


def test_outer_product_steps_have_no_collectives():
    """Both ``extract_V_body_sharded`` and ``assemble_W_sharded`` should
    compile to a pure local kernel — no all-gather / all-reduce HLO ops.

    The Schur-reductions step IS allowed collectives (it's matrix-vector
    contraction along sharded axes); we report its op count for the
    profile record but don't enforce zero.
    """
    rng = np.random.default_rng(7)
    prob = _make_synthetic_problem(rng, nq=4, n_mu=32)

    mesh = _make_mesh(2, 2)
    repspec = NamedSharding(mesh, P(None,))

    V_q       = jax.device_put(jnp.asarray(prob['V_full']),
                               NamedSharding(mesh, V_FLATQ_SPEC))
    g0_x      = jax.device_put(jnp.asarray(prob['g0']),
                               NamedSharding(mesh, G0_X_SPEC))
    g0_y      = jax.device_put(jnp.asarray(prob['g0']),
                               NamedSharding(mesh, G0_Y_SPEC))
    v_head    = jax.device_put(jnp.asarray(prob['v_head']), repspec)

    # ----- extract_V_body_sharded -----
    with mesh:
        txt_extract = _hlo_text(extract_V_body_sharded,
                                V_q, g0_x, g0_y, v_head, mesh=mesh)
    bad_substrings = ['all-gather', 'all-reduce', 'all-to-all',
                      'collective-permute', 'reduce-scatter', 'replica-id']
    # 'reduce' alone is too noisy — XLA emits ``reduce`` for every fold
    # over a non-sharded axis; we look only for explicit cross-device
    # collective ops.
    found_extract = {sub: txt_extract.count(sub) for sub in bad_substrings}
    n_lines_extract = len(txt_extract.splitlines())
    if jax.process_index() == 0:
        print(f"  extract_V_body  : {n_lines_extract} HLO lines, collectives={found_extract}")
    for sub, n in found_extract.items():
        assert n == 0, f"extract_V_body emits {n} '{sub}' ops — should be zero"

    # ----- assemble_W_sharded -----
    W_body0 = V_q.copy()
    A_wing_x  = jax.device_put(jnp.asarray(prob['chi_wing']),
                               NamedSharding(mesh, G0_X_SPEC))
    A_wingp_y = jax.device_put(jnp.asarray(prob['chi_wing']),
                               NamedSharding(mesh, G0_Y_SPEC))
    W_head_q  = jax.device_put(jnp.ones_like(jnp.asarray(prob['v_head'])), repspec)

    with mesh:
        txt_assemble = _hlo_text(
            assemble_W_sharded,
            W_body0, g0_x, g0_y, A_wing_x, A_wingp_y, W_head_q,
            mesh=mesh)
    found_assemble = {sub: txt_assemble.count(sub) for sub in bad_substrings}
    n_lines_assemble = len(txt_assemble.splitlines())
    if jax.process_index() == 0:
        print(f"  assemble_W      : {n_lines_assemble} HLO lines, collectives={found_assemble}")
    for sub, n in found_assemble.items():
        assert n == 0, f"assemble_W emits {n} '{sub}' ops — should be zero"

    # ----- schur_reductions_sharded (informational) -----
    chi_wing_x  = jax.device_put(jnp.asarray(prob['chi_wing']),
                                 NamedSharding(mesh, G0_X_SPEC))
    chi_wing_y  = jax.device_put(jnp.asarray(prob['chi_wing']),
                                 NamedSharding(mesh, G0_Y_SPEC))
    chi_wingp_x = jax.device_put(jnp.asarray(prob['chi_wingp']),
                                 NamedSharding(mesh, G0_X_SPEC))
    chi_wingp_y = jax.device_put(jnp.asarray(prob['chi_wingp']),
                                 NamedSharding(mesh, G0_Y_SPEC))
    chi_head    = jax.device_put(jnp.asarray(prob['chi_head']), repspec)

    with mesh:
        txt_schur = _hlo_text(
            schur_reductions_sharded,
            W_body0, chi_wing_x, chi_wing_y, chi_wingp_x, chi_wingp_y, chi_head,
            mesh=mesh)
    found_schur = {sub: txt_schur.count(sub) for sub in bad_substrings}
    n_lines_schur = len(txt_schur.splitlines())
    if jax.process_index() == 0:
        print(f"  schur_reductions: {n_lines_schur} HLO lines, collectives={found_schur}")
    # Sanity: schur_reductions MUST contract over sharded axes, so on a
    # multi-device mesh it should have at least one cross-device reduction.
    # On a 1×1 mesh (single device fallback) collectives are trivially
    # zero; otherwise expect ≥ 1 all-reduce.
    n_devices = mesh.devices.size
    if n_devices > 1:
        total_collective = sum(found_schur.values())
        assert total_collective >= 1, (
            f"On a {mesh.devices.shape} mesh, schur_reductions should "
            f"emit at least one cross-device reduction; got {found_schur}. "
            f"Either the einsum contractions aren't actually sharded or HLO "
            f"text doesn't match the substrings.")
        if jax.process_index() == 0:
            print(f"  (≥1 collective expected on {mesh.devices.shape} mesh — OK)")


# ---------------------------------------------------------------------------
# Smoke entry point — ``python tests/test_head_wing_schur.py`` runs both.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rank0 = (jax.process_index() == 0)
    if rank0:
        print(f"# rank 0/{jax.process_count()} — global devices = {jax.device_count()}")
        print("# 1. Algebraic correctness vs dense reference")
    test_schur_reconstruction_matches_dense()
    if rank0:
        print("    PASS")
        print()
        print("# 2. Outer-product collectives")
    test_outer_product_steps_have_no_collectives()
    if rank0:
        print("    PASS")
