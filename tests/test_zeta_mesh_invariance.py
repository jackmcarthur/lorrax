"""Portable cross-mesh invariance gate for the charge ζ-fit factor+solve.

THE regression guard for the 2026-07-20 device-count-correctness bug: the
charge-channel ζ-fit's distributed cuSolverMp Cholesky is block-cyclic, so
its partial-sum regrouping depends on the process grid ``(px, py)``.  At
large, mildly rank-deficient n_μ (MoS2 6×6, 1600 centroids) the factor
``L_q = chol(C_q)`` drifted ~0.3% between a 2×2 and a 4×4 grid, and the
GN-PPM pole construction amplified that into tens-of-eV Σ_c garbage on
non-16-GPU meshes (reports/gw_zeta_mesh_invariance_2026-07-20).

The fix routes fit-size charge tiles through a fully-REPLICATED dense
``jnp.linalg.cholesky`` (``isdf.core._factor_c_q_replicated``), which runs on
the whole matrix on every device and is therefore bit-identical across
device counts and process grids.  This gate locks that property in: it
factors + back-solves a FIXED synthetic SPD CCT on several CPU meshes
(1×1, 1×2, 2×1, 2×2 — via ``--xla_force_host_platform_device_count``) and
asserts the resulting L_q and ζ agree to the ULP floor.  Portable: CPU-only,
no GPU, no cuSolverMp — it guards the *auto-resolved* path (which must be the
mesh-invariant replicated Cholesky for fit-size stacks) so any future change
that reintroduces a grid-dependent charge factor as the default trips it.

The full end-to-end complement (MoS2 6×6 GN-PPM at 1×1 vs 2×2, asserting
|Δ Re Σ_c(VBM)| < few meV) lives in
``tests/multi_device/eqp_invariance_cross_p.py`` + the report harness; it
needs a multi-GPU allocation and is not in the default suite.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

# Meshes exercised on the CPU host-device pool and the ULP-floor tolerance.
# 4 host devices → {1×1, 1×2, 2×1, 2×2, 1×4, 4×1}.  A grid-dependent factor
# (the pre-fix cuSolverMp policy) would show ~1e-3 frob-rel here; the
# replicated dense factor is bit-identical, so 1e-10 cleanly separates them.
_NDEV = 4
_TOL = 1.0e-10


def _worker() -> int:
    """Child process: build a fixed SPD CCT, factor + solve on every mesh,
    print the worst cross-mesh frob-rel for L_q and ζ as JSON."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from isdf import factor_c_q, solve_zeta
    from isdf.core import _resolve_solver_kind

    devs = jax.devices()
    if len(devs) < _NDEV:
        print(json.dumps({"skip": f"only {len(devs)} devices"}))
        return 0

    # Fixed, deterministic, well-posed SPD charge CCT + RHS.  n_μ=64 is
    # divisible by every mesh axis (no μ-pad) and the stack is tiny, so the
    # auto resolver must pick the replicated dense Cholesky.
    rng = np.random.default_rng(20260720)
    nq, n_mu, n_rhs = 4, 64, 24
    A = (rng.standard_normal((nq, n_mu, n_mu + 8))
         + 1j * rng.standard_normal((nq, n_mu, n_mu + 8)))
    C = A @ np.conj(np.transpose(A, (0, 2, 1)))          # SPD, Hermitian
    C = C + n_mu * np.eye(n_mu)[None]                    # well-conditioned
    C = 0.5 * (C + np.conj(np.transpose(C, (0, 2, 1))))  # kill fp asymmetry
    C = C.astype(np.complex128)
    Zrhs = (rng.standard_normal((nq, n_mu, n_rhs))
            + 1j * rng.standard_normal((nq, n_mu, n_rhs))).astype(np.complex128)

    mesh_shapes = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (4, 1)]
    L_ref = zeta_ref = None
    worst_L = worst_z = 0.0
    kinds = {}
    for (px, py) in mesh_shapes:
        mesh = Mesh(np.asarray(devs[: px * py]).reshape(px, py), ('x', 'y'))
        kind = _resolve_solver_kind(mesh, 0, 'auto', n_rmu=n_mu, nq=nq)
        kinds[f"{px}x{py}"] = kind
        assert kind == 'replicated_cholesky', (
            f"auto resolver picked {kind!r} on {px}x{py} for fit-size n_μ; "
            f"expected 'replicated_cholesky' (mesh-invariant)")
        in_sh = NamedSharding(mesh, P(None, 'x', 'y'))
        C_dev = jax.device_put(jnp.asarray(C), in_sh)
        Z_dev = jax.device_put(jnp.asarray(Zrhs), in_sh)
        L = factor_c_q(C_dev, mesh, vertex_mu_L=0,
                       n_rmu_logical=n_mu, solver_kind=kind)
        zeta = solve_zeta(L, Z_dev, mesh, q_chunk_size=nq,
                          solver_kind=kind, n_rmu_logical=n_mu)
        L_np = np.asarray(jax.device_get(L))
        z_np = np.asarray(jax.device_get(zeta))
        if L_ref is None:
            L_ref, zeta_ref = L_np, z_np
        else:
            worst_L = max(worst_L, float(
                np.linalg.norm(L_np - L_ref) / max(np.linalg.norm(L_ref), 1e-300)))
            worst_z = max(worst_z, float(
                np.linalg.norm(z_np - zeta_ref) / max(np.linalg.norm(zeta_ref), 1e-300)))
    print(json.dumps({"worst_L": worst_L, "worst_zeta": worst_z,
                      "kinds": kinds}))
    return 0


def test_zeta_fit_charge_factor_solve_is_mesh_invariant():
    """factor_c_q + solve_zeta (charge, auto) give bit-identical L_q and ζ
    across CPU meshes {1×1, 1×2, 2×1, 2×2, 1×4, 4×1}."""
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    # Append so any pre-existing XLA_FLAGS survive.
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + f" --xla_force_host_platform_device_count={_NDEV}").strip()
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "worker"],
        env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, (
        f"worker failed rc={res.returncode}\nSTDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}")
    line = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert line, f"no JSON from worker.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    out = json.loads(line[-1])
    if "skip" in out:
        pytest.skip(f"cross-mesh gate: {out['skip']}")
    assert out["worst_L"] <= _TOL, (
        f"charge Cholesky L_q drifts across meshes: worst frob-rel "
        f"{out['worst_L']:.3e} > {_TOL:g} (solver picks: {out['kinds']})")
    assert out["worst_zeta"] <= _TOL, (
        f"charge ζ drifts across meshes: worst frob-rel "
        f"{out['worst_zeta']:.3e} > {_TOL:g} (solver picks: {out['kinds']})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sys.exit(_worker())
    sys.exit(test_zeta_fit_charge_factor_solve_is_mesh_invariant() or 0)
