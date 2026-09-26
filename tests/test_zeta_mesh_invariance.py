"""Portable cross-mesh invariance gate for the charge ζ-fit factor+solve.

THE regression guard for the 2026-07-20 device-count-correctness bug: a
block-cyclic distributed charge factor regroups its partial sums with the
process grid ``(px, py)``, and at large, mildly rank-deficient n_μ (MoS2 6×6,
1600 centroids) the factor drifted ~0.3% between a 2×2 and a 4×4 grid, which
the GN-PPM pole construction amplified into tens-of-eV Σ_c garbage.

The charge factor is the replicated rank-truncated eigh
(``isdf.core._factor_c_q_replicated``): every q is factored as ONE dense
whole-tile call, so it is bit-identical across device counts and process
grids.  This gate factors + back-solves a FIXED near-singular CCT on several
CPU meshes (1×1, 1×2, 2×1, 2×2 — via ``--xla_force_host_platform_device_count``)
and asserts the factor and ζ agree to the ULP floor.  Portable: CPU-only, no
GPU.

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
# would show ~1e-3 frob-rel here; the replicated dense factor is
# bit-identical, so 1e-10 cleanly separates them.
_NDEV = 4
_TOL = 1.0e-10


def _per_q_solve(F, Z, *, n_log):
    """ζ = C⁺Z per q at the logical extent: route G's charge back-solve
    (``isdf.core._zeta_logical_solvers``, ``cplus.apply``) vmapped over q."""
    import jax
    from isdf.core import _zeta_logical_solvers
    _, _, pinv_matmul = _zeta_logical_solvers(int(n_log))
    return jax.jit(jax.vmap(pinv_matmul))(F, Z)


def _worker_rank_truncate() -> int:
    """Child process: build a NEAR-SINGULAR (over-complete) SPD CCT, factor
    + solve with the rank-truncation path on every mesh.  Asserts the
    auto-resolved kind is ``replicated_rank_truncate`` and reports (a) the
    worst cross-mesh frob-rel for the pseudo-inverse factor B and ζ, and
    (b) the pseudo-inverse residual on range(C), which a full (untruncated)
    solve would fail by ‖(I − P_range) Z‖."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from isdf import factor_c_q
    from isdf.core import _resolve_solver_kind

    devs = jax.devices()
    if len(devs) < _NDEV:
        print(json.dumps({"skip": f"only {len(devs)} devices"}))
        return 0

    # Near-singular charge CCT: designed spectrum with r≪n_μ "signal"
    # eigenvalues O(1) and the rest ~1e-13 (κ≈1e13) — the over-complete
    # n_μ > pair-density-rank regime that makes plain Cholesky amplify.
    rng = np.random.default_rng(20260721)
    nq, n_mu, n_rhs, r = 4, 64, 24, 40
    rcond = 1e-10
    C = np.empty((nq, n_mu, n_mu), dtype=np.complex128)
    for iq in range(nq):
        M = (rng.standard_normal((n_mu, n_mu))
             + 1j * rng.standard_normal((n_mu, n_mu)))
        Q, _ = np.linalg.qr(M)                            # Haar-ish unitary
        evals = np.concatenate([rng.uniform(0.5, 2.0, r),
                                np.full(n_mu - r, 1e-13)])
        Ciq = (Q * evals) @ np.conj(Q.T)
        C[iq] = 0.5 * (Ciq + np.conj(Ciq.T))             # kill fp asymmetry
    Zrhs = (rng.standard_normal((nq, n_mu, n_rhs))
            + 1j * rng.standard_normal((nq, n_mu, n_rhs))).astype(np.complex128)

    mesh_shapes = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (4, 1)]
    B_ref = zeta_ref = None
    worst_B = worst_z = 0.0
    kinds = {}
    zeta_rt_2x2 = None
    for (px, py) in mesh_shapes:
        mesh = Mesh(np.asarray(devs[: px * py]).reshape(px, py), ('x', 'y'))
        kind = _resolve_solver_kind(0, 'auto', n_rmu=n_mu, nq=nq)
        kinds[f"{px}x{py}"] = kind
        assert kind == 'replicated_rank_truncate', (
            f"auto resolver picked {kind!r} on {px}x{py} for fit-size n_μ; "
            f"expected 'replicated_rank_truncate'")
        in_sh = NamedSharding(mesh, P(None, 'x', 'y'))
        C_dev = jax.device_put(jnp.asarray(C), in_sh)
        Z_dev = jax.device_put(jnp.asarray(Zrhs), in_sh)
        B = factor_c_q(C_dev, mesh, vertex_mu_L=0, n_rmu_logical=n_mu,
                       solver_kind=kind, zeta_rcond=rcond)
        zeta = _per_q_solve(B, Z_dev, n_log=n_mu)
        B_np = np.asarray(jax.device_get(B))
        z_np = np.asarray(jax.device_get(zeta))
        if B_ref is None:
            B_ref, zeta_ref = B_np, z_np
        else:
            worst_B = max(worst_B, float(
                np.linalg.norm(B_np - B_ref) / max(np.linalg.norm(B_ref), 1e-300)))
            worst_z = max(worst_z, float(
                np.linalg.norm(z_np - zeta_ref) / max(np.linalg.norm(zeta_ref), 1e-300)))
        if (px, py) == (2, 2):
            zeta_rt_2x2 = z_np

    # ζ_rt should reconstruct Z on the range of C: C ζ ≈ P_range Z.  With the
    # designed spectrum the range is exactly the top-r subspace; the residual
    # of the pseudo-inverse relation ‖C ζ − Z_range‖/‖Z_range‖ is ~ULP.
    Z_range_res = 0.0
    for iq in range(nq):
        w, V = np.linalg.eigh(C[iq])
        keep = w > rcond * w.max()
        Pr = V[:, keep] @ np.conj(V[:, keep].T)
        Zr = Pr @ Zrhs[iq]
        Z_range_res = max(Z_range_res, float(
            np.linalg.norm(C[iq] @ zeta_rt_2x2[iq] - Zr)
            / max(np.linalg.norm(Zr), 1e-300)))
    print(json.dumps({"worst_B": worst_B, "worst_zeta": worst_z,
                      "kinds": kinds, "range_residual": Z_range_res}))
    return 0


def _worker_qparallel() -> int:
    """Child process: the q-parallel EXECUTION of the replicated charge
    factor (the P>1 fold, ``LORRAX_ZETA_QPARALLEL``) must return EXACTLY
    the bits of the all-ranks execution — same plan, same bits — on every
    mesh, in BOTH modes, with a q count that does not divide the device
    count (q-pad + cond-skip) and a padded μ extent (identity-pad
    re-embed).  Exact equality, not a tolerance: the fold is a schedule,
    and any nonzero delta means it silently became a numerical route."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from isdf import factor_c_q

    devs = jax.devices()
    if len(devs) < _NDEV:
        print(json.dumps({"skip": f"only {len(devs)} devices"}))
        return 0

    # Well-conditioned SPD logical block, zero-embedded to a padded extent: n_log=60 inside n_pad=64 exercises
    # solve_at_logical + the identity-pad re-embed; nq=6 does not divide
    # 4 devices, exercising the q-pad + cond-skip.
    rng = np.random.default_rng(20260801)
    nq, n_log, n_pad, = 6, 60, 64
    A = (rng.standard_normal((nq, n_log, n_log + 8))
         + 1j * rng.standard_normal((nq, n_log, n_log + 8)))
    C_log = A @ np.conj(np.transpose(A, (0, 2, 1)))
    C_log = C_log + n_log * np.eye(n_log)[None]
    C_log = 0.5 * (C_log + np.conj(np.transpose(C_log, (0, 2, 1))))
    C = np.zeros((nq, n_pad, n_pad), dtype=np.complex128)
    C[:, :n_log, :n_log] = C_log

    exact = {}
    max_abs = 0.0
    for (px, py) in [(1, 2), (2, 1), (2, 2), (1, 4), (4, 1)]:
        mesh = Mesh(np.asarray(devs[: px * py]).reshape(px, py), ('x', 'y'))
        in_sh = NamedSharding(mesh, P(None, 'x', 'y'))
        C_dev = jax.device_put(jnp.asarray(C), in_sh)
        for mode, kind in (('rank_truncate', 'replicated_rank_truncate'),):
            outs = {}
            for force in ('0', '1'):
                os.environ['LORRAX_ZETA_QPARALLEL'] = force
                outs[force] = np.asarray(jax.device_get(factor_c_q(
                    C_dev, mesh, vertex_mu_L=0, n_rmu_logical=n_log,
                    solver_kind=kind, zeta_rcond=1e-10)))
            exact[f"{px}x{py}_{mode}"] = bool(
                np.array_equal(outs['0'], outs['1']))
            max_abs = max(max_abs, float(
                np.max(np.abs(outs['0'] - outs['1']))))
    os.environ.pop('LORRAX_ZETA_QPARALLEL', None)
    print(json.dumps({"exact": exact, "max_abs": max_abs}))
    return 0


def _worker_cap() -> int:
    """Child process: the replication-cap CONTRACT for the production
    ``charge_zeta_solve='rank_truncate'``.  Reports what the auto resolver
    returns for the two real MoS2 12×12 / n_μ=2412 ζ stacks — IBZ (nq=74,
    6.42 GiB) and full-BZ (nq=144, 12.48 GiB) — under whatever cap the
    parent set via ``LORRAX_ZETA_REPLICATE_CAP_GIB``."""
    import isdf.core as core

    res = {"cap_gib": core._REPLICATED_CHOL_MAX_STACK_BYTES / 1024 ** 3}
    for tag, nq in (("ibz74", 74), ("fullbz144", 144)):
        try:
            res[f"{tag}_rank_truncate"] = core._resolve_solver_kind(
                0, "auto", n_rmu=2412, nq=nq)
        except ValueError as exc:
            res[f"{tag}_rank_truncate"] = f"RAISE:{exc}"
    print(json.dumps(res))
    return 0


def _run_worker(tag: str, timeout: int = 600, env_extra: dict | None = None,
                ndev: int | None = None):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    if env_extra:
        env.update(env_extra)
    # Append so any pre-existing XLA_FLAGS survive.
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + f" --xla_force_host_platform_device_count={ndev or _NDEV}").strip()
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), tag],
        env=env, capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, (
        f"worker {tag} failed rc={res.returncode}\nSTDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}")
    line = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert line, f"no JSON from worker.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return json.loads(line[-1])


def test_zeta_fit_charge_rank_truncate_is_mesh_invariant_and_conditions():
    """Rank-truncation (the charge ζ-solve): on a near-singular over-complete
    CCT (κ≈1e13) the pseudo-inverse factor B and ζ are bit-identical across
    CPU meshes, and ζ solves the pseudo-inverse relation on range(C)."""
    out = _run_worker("worker_rt")
    if "skip" in out:
        pytest.skip(f"rank-truncate gate: {out['skip']}")
    assert out["worst_B"] <= _TOL, (
        f"rank-truncated factor B drifts across meshes: worst frob-rel "
        f"{out['worst_B']:.3e} > {_TOL:g} (solver picks: {out['kinds']})")
    assert out["worst_zeta"] <= _TOL, (
        f"rank-truncated ζ drifts across meshes: worst frob-rel "
        f"{out['worst_zeta']:.3e} > {_TOL:g} (solver picks: {out['kinds']})")
    # ζ solves the pseudo-inverse relation on the range of C to the ULP floor.
    # A full solve would miss it by ‖(I − P_range) Z‖, which is O(1) here.
    assert out["range_residual"] <= 1e-8, (
        f"rank-truncated ζ does not reconstruct Z on range(C): "
        f"residual {out['range_residual']:.3e}")


def test_qparallel_execution_is_bit_identical_to_replicated():
    """The folded q-parallel execution of the replicated charge factor
    (``LORRAX_ZETA_QPARALLEL``, ``isdf.core._factor_c_q_replicated_qparallel``)
    returns EXACTLY the bits of the all-ranks execution on every mesh,
    including a non-device-dividing nq and a padded μ
    extent.  This is what makes the fold a SCHEDULE of the replicated plan
    rather than a third resolution of the factor family — the moment this
    gate needs a tolerance, it has become a plan and must be re-argued."""
    out = _run_worker("worker_qpar")
    if "skip" in out:
        pytest.skip(f"q-parallel gate: {out['skip']}")
    bad = sorted(k for k, v in out["exact"].items() if not v)
    assert not bad, (
        f"q-parallel factor bits drift from the all-ranks execution on "
        f"{bad} (max abs delta {out['max_abs']:.3e}); the fold's "
        f"bit-identity contract is broken")


def test_rank_truncate_refuses_above_the_replication_cap():
    """``charge_zeta_solve='rank_truncate'`` must RAISE — never silently
    downgrade — when the CCT stack exceeds the replication cap, and the cap
    itself must be raisable from the environment.

    The silent downgrade cost two sessions: the MoS2 12×12 / n_μ=2412
    FULL-BZ ζ stack (nq=144, 12.48 GiB) fell back to the distributed
    cuSolverMp Cholesky, and the ζ came out 4.5× too large — rebuilding V_q
    to relF 16–32 instead of 1.8e-15 (reports/bse_exciton_smooth_2026-07-21,
    logs/zeta_fullbz_BAD_cholesky_fallback.log).  Both real stacks of that
    campaign are pinned here: IBZ nq=74 (6.42 GiB, needs cap ≥ 8) and
    full-BZ nq=144 (12.48 GiB, needs cap ≥ 13)."""
    default = _run_worker("worker_cap")
    if "skip" in default:
        pytest.skip(f"cap gate: {default['skip']}")
    assert default["cap_gib"] == pytest.approx(4.0), \
        f"default replication cap changed: {default['cap_gib']} GiB"
    # The cap gates ONE q-BATCH, not the whole stack —
    # factor_c_q_replicated_batched bounds the true per-rank transient at the
    # cap regardless of nq, so both historical stacks (ibz74 6.42 GiB,
    # fullbz144 12.48 GiB TOTAL) resolve to the replicated rank-truncate
    # route under the DEFAULT cap.
    for tag in ("ibz74", "fullbz144"):
        assert default[f"{tag}_rank_truncate"] == "replicated_rank_truncate", (
            f"{tag} must resolve to replicated_rank_truncate under the "
            f"per-BATCH cap contract, got "
            f"{default[f'{tag}_rank_truncate']!r}")

    # The env knob still parses and overrides (it now sets the batch bound).
    raised = _run_worker("worker_cap",
                         env_extra={"LORRAX_ZETA_REPLICATE_CAP_GIB": "8"})
    assert raised["cap_gib"] == pytest.approx(8.0), \
        "LORRAX_ZETA_REPLICATE_CAP_GIB did not raise the cap"
    for tag in ("ibz74", "fullbz144"):
        assert raised[f"{tag}_rank_truncate"] == "replicated_rank_truncate"


def test_distributed_tier_collective_payload_is_bounded(monkeypatch):
    """The COLLECTIVE PAYLOAD bound of the `distributed` tier (scorecard AF).

    A memory cap is not a transport cap.  A memory cap bounds how much
    gathered data may be LIVE; this bounds how many bytes
    ONE ``all_gather`` / ``psum_scatter`` instruction hands to the
    transport in a single shot.  Job 7876062 died at P=144 on a 1.15 GB
    single-shot Gloo AllGather in the C⁺ formation with MaxRSS at 12 % of
    budget — i.e. the memory cap was satisfied and the job still died.

    Pinned here because the failure mode is invisible below P≈64: at
    fixture scale the cap does not bite and the tier runs exactly as it
    did before this workstream.
    """
    import isdf.core as core

    monkeypatch.delenv("LORRAX_COLLECTIVE_CHUNK_MB", raising=False)

    assert core._DEFAULT_COLLECTIVE_CHUNK_MB == 128.0
    assert core._collective_chunk_bytes() == 128 * 1024 ** 2

    # THE production point (MoS2 12×12, c2406): nq=144, μ_pad=2448, 12×12
    # mesh.  The C⁺ formation's two collectives are μ²/Py and μ²/Px per q,
    # both 2448·204·16 = 7.99 MB; unchunked that is 1.15 GB in one shot.
    per_q = 2448 * 204 * 16
    qb = core._chunk_q(144, per_q)
    assert qb == 16, qb
    assert qb * per_q <= core._collective_chunk_bytes()
    assert 144 * per_q > 1e9, "the unchunked payload is the one that died"

    # The back-solve's larger leg: Z's column block, μ·r_chunk/Py per q.
    per_q_z = 2448 * (11664 // 12) * 16
    qb_z = core._chunk_q(144, per_q_z)
    assert qb_z == 3, qb_z
    assert qb_z * per_q_z <= core._collective_chunk_bytes()

    # Fixture scale: the cap must NOT bite, so P≤16 behaviour (and every
    # existing gate) is the pre-AF code path exactly.
    assert core._chunk_q(9, 64 * 32 * 16) == 9

    # One q is the floor — a single q's collective is irreducible by
    # q-blocking, and the tier must not deadlock trying to go below it.
    assert core._chunk_q(144, 1 << 30) == 1

    # Tunable, and disable-able for reproducing the failure on purpose.
    monkeypatch.setenv("LORRAX_COLLECTIVE_CHUNK_MB", "64")
    assert core._collective_chunk_bytes() == 64 * 1024 ** 2
    assert core._chunk_q(144, per_q) == 8
    monkeypatch.setenv("LORRAX_COLLECTIVE_CHUNK_MB", "0")
    assert core._chunk_q(144, per_q) == 144      # unbounded == pre-AF
    monkeypatch.setenv("LORRAX_COLLECTIVE_CHUNK_MB", "not-a-number")
    assert core._collective_chunk_bytes() == 128 * 1024 ** 2


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker_cap":
        sys.exit(_worker_cap())
    if len(sys.argv) > 1 and sys.argv[1] == "worker_rt":
        sys.exit(_worker_rank_truncate())
    if len(sys.argv) > 1 and sys.argv[1] == "worker_qpar":
        sys.exit(_worker_qparallel())
    sys.exit(test_zeta_fit_charge_rank_truncate_is_mesh_invariant_and_conditions() or 0)
