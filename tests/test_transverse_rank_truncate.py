"""Gates for the TRANSVERSE rank-truncating ζ solve (2026-08).

``transverse_zeta_solve = 'rank_truncate'`` ports the charge channel's
rank-truncating ζ solve to the bispinor transverse channels: per-q eigh
of the Hermitian INDEFINITE transverse CCT, drop |λ| < τ·|λ|_max
(τ = ``transverse_zeta_rcond``), store the EXPLICIT truncated
pseudo-inverse C⁺, apply ONE GEMM per r-chunk.  The factor stage is
HOISTED through the charge replicated scaffolding
(``_charge_factor_math`` mode ``'transverse_rank_truncate'`` inside
``factor_c_q_replicated_batched``), so it inherits the q-parallel fold
and its bit-identity contract.

Gates:

* ``test_local_family_is_bit_identical_across_schedules`` — within the
  family, EXACT bit equality across CPU meshes 1x1/2x2/1x4, both factor
  schedules (``LORRAX_ZETA_QPARALLEL`` 0/1), both back-solve gather
  tiers (replicated/per_q), two q-chunkings, and two r-chunks against
  ONE factor.  Fixture: indefinite spectrum with TRS-paired near-null
  modes, non-dividing nq (q-pad + cond-skip), padded mu (identity
  re-embed + logical-extent slicing), and a spinor²-blocked RHS column
  count.  The moment this needs a tolerance, the schedule fold has
  become a numerical route and must be re-argued.
* the same worker checks ζ against an independent numpy truncated
  pseudo-inverse (~1e-12 relative — same arithmetic, different library)
  and that ζ pad rows are exactly zero.
* ``test_truncation_drops_the_trs_near_null_pair`` — n_keep (via the
  rank of C⁺) is exactly n_log - 2 on the fixture: the |λ| cut removes
  the TRS pair the LU ridge merely lifts.
* ``test_ridge_vs_rank_truncate_agree_on_kept_subspace`` — the two
  FAMILIES are different algorithms (not bit-comparable): on the kept
  subspace they agree to ~ridge-level; the ridge solution additionally
  carries the huge near-null components truncation removes.  This is
  the unit-level statement of the production gauge-tolerance A/B.
* ``test_resolver_semantics`` — family selection, any-count resolve (no
  divisibility raise on the rank_truncate family), the explicit
  ``distributed_lu`` conflict refusal, and untouched ridge-family
  defaults.

The DISTRIBUTED plan (pzheevd at the padded extent, indefinite mode)
needs one JAX process per device + the host FFI .so, so its parity leg
lives in the srun harness / CLAIMS ledger, exactly like the hoist
gate's scalapack twin.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_NDEV = 4


def _fixture(rng, nq, n_log, n_pad, n_z, n_z2):
    """Hermitian INDEFINITE per-q logical blocks with a TRS-paired
    near-null mode pair each, embedded at padded extent; two RHS
    r-chunks with exactly-zero pad rows.  Returns (C, Z, Z2, spectra)
    with spectra = per-q (lam, V) for the numpy reference."""
    import numpy as np

    C = np.zeros((nq, n_pad, n_pad), dtype=np.complex128)
    spectra = []
    for q in range(nq):
        A = (rng.standard_normal((n_log, n_log))
             + 1j * rng.standard_normal((n_log, n_log)))
        H = 0.5 * (A + A.conj().T)
        lam, V = np.linalg.eigh(H)
        lam = lam - np.median(lam)          # both signs
        lam[0] = 1e-14                      # TRS near-null pair
        lam[1] = -1e-14
        C[q, :n_log, :n_log] = (V * lam[None, :]) @ V.conj().T
        spectra.append((lam, V))
    Z = np.zeros((nq, n_pad, n_z), dtype=np.complex128)
    Z[:, :n_log, :] = (rng.standard_normal((nq, n_log, n_z))
                       + 1j * rng.standard_normal((nq, n_log, n_z)))
    Z2 = np.zeros((nq, n_pad, n_z2), dtype=np.complex128)
    Z2[:, :n_log, :] = (rng.standard_normal((nq, n_log, n_z2))
                        + 1j * rng.standard_normal((nq, n_log, n_z2)))
    return C, Z, Z2, spectra


def _np_pinv_solve(spectra, Z_log, tau):
    """Independent truncated-pseudo-inverse reference: |λ| cut at
    τ·|λ|_max, ζ = Σ_keep v (1/λ) vᴴ Z."""
    import numpy as np

    out = np.zeros_like(Z_log)
    for q, (lam, V) in enumerate(spectra):
        sig = np.abs(lam)
        keep = sig > (tau * sig.max())
        inv = np.where(keep, 1.0 / np.where(keep, lam, 1.0), 0.0)
        out[q] = (V * inv[None, :]) @ (V.conj().T @ Z_log[q])
    return out


def _worker_rt() -> int:
    """Child: local-family bit identity + numpy reference + pad rows."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from isdf import factor_c_q, solve_zeta
    import isdf.core as core

    devs = jax.devices()
    if len(devs) < _NDEV:
        print(json.dumps({"skip": f"only {len(devs)} devices"}))
        return 0

    rng = np.random.default_rng(20260801)
    # nq=6 does not divide 4 devices (q-pad + cond-skip in the fold);
    # n_z=32 is the bispinor spinor^2 column blocking (ns^2=16 x r=2);
    # n_z2=30 does NOT divide the device count (needs_padding path).
    nq, n_log, n_pad, n_z, n_z2 = 6, 60, 64, 32, 30
    tau = 1e-8
    C, Z, Z2, spectra = _fixture(rng, nq, n_log, n_pad, n_z, n_z2)
    ref1 = _np_pinv_solve(spectra, Z[:, :n_log, :], tau)
    ref2 = _np_pinv_solve(spectra, Z2[:, :n_log, :], tau)

    exact = {}
    ref_ok = True
    pad_ok = True
    max_ref_rel = 0.0
    got_first = {}
    for (px, py) in [(1, 1), (2, 2), (1, 4)]:
        mesh = Mesh(np.asarray(devs[: px * py]).reshape(px, py), ('x', 'y'))
        in_sh = NamedSharding(mesh, P(None, 'x', 'y'))
        C_dev = jax.device_put(jnp.asarray(C), in_sh)
        for force in (('0',) if px * py == 1 else ('0', '1')):
            os.environ['LORRAX_ZETA_QPARALLEL'] = force
            core._replicated_chol_cache.clear()
            core._qparallel_factor_cache.clear()
            Cp, piv = factor_c_q(
                C_dev, mesh, vertex_mu_L=1, n_rmu_logical=n_log,
                solver_kind='transverse_rank_truncate',
                transverse_zeta_rcond=tau)
            assert piv is None, "rank_truncate family returns piv=None"
            for gather in ('replicated', 'per_q'):
                for q_chunk in (2, nq):
                    # ONE hoisted factor, TWO r-chunks (the reuse).
                    got1 = np.asarray(jax.device_get(solve_zeta(
                        Cp, jax.device_put(jnp.asarray(Z), in_sh),
                        mesh, q_chunk, vertex_mu_L=1,
                        solver_kind='transverse_rank_truncate',
                        n_rmu_logical=n_log, zeta_gather=gather)))
                    got2 = np.asarray(jax.device_get(solve_zeta(
                        Cp, jax.device_put(jnp.asarray(Z2), in_sh),
                        mesh, q_chunk, vertex_mu_L=1,
                        solver_kind='transverse_rank_truncate',
                        n_rmu_logical=n_log, zeta_gather=gather)))
                    tag = f"{px}x{py}_qp{force}_{gather}_qc{q_chunk}"
                    if not got_first:
                        got_first = {"z1": got1, "z2": got2}
                        exact[tag] = True
                    else:
                        # Bit identity across meshes, schedules, gather
                        # tiers and q-chunkings WITHIN the family.
                        exact[tag] = bool(
                            np.array_equal(got_first["z1"], got1)
                            and np.array_equal(got_first["z2"], got2))
                    # Independent numpy reference (different library:
                    # tight allclose, not bit equality).
                    r1 = float(np.max(np.abs(got1[:, :n_log, :] - ref1))
                               / np.max(np.abs(ref1)))
                    r2 = float(np.max(np.abs(got2[:, :n_log, :] - ref2))
                               / np.max(np.abs(ref2)))
                    max_ref_rel = max(max_ref_rel, r1, r2)
                    ref_ok = ref_ok and (r1 < 1e-10) and (r2 < 1e-10)
                    # ζ pad rows exactly zero.
                    pad_ok = pad_ok and bool(
                        np.all(got1[:, n_log:, :] == 0.0)
                        and np.all(got2[:, n_log:, :] == 0.0))
    os.environ.pop('LORRAX_ZETA_QPARALLEL', None)

    # Truncation really drops the TRS pair: rank(C⁺ logical block).
    Cp_log = np.asarray(jax.device_get(Cp))[:, :n_log, :n_log]
    ranks = [int(np.linalg.matrix_rank(Cp_log[q], tol=1e-6))
             for q in range(nq)]
    print(json.dumps({
        "exact": exact, "ref_ok": ref_ok, "max_ref_rel": max_ref_rel,
        "pad_ok": pad_ok, "ranks": ranks, "n_log": n_log}))
    return 0


def _worker_gauge() -> int:
    """Child: ridge vs rank_truncate — kept-subspace agreement, and the
    near-null content only the ridge solution carries."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from isdf import factor_c_q, solve_zeta

    devs = jax.devices()
    rng = np.random.default_rng(20260802)
    nq, n_log, n_pad, n_z = 3, 48, 48, 16
    tau = 1e-8
    C, Z, _, spectra = _fixture(rng, nq, n_log, n_pad, n_z, n_z)

    mesh = Mesh(np.asarray(devs[:1]).reshape(1, 1), ('x', 'y'))
    in_sh = NamedSharding(mesh, P(None, 'x', 'y'))
    C_dev = jax.device_put(jnp.asarray(C), in_sh)
    Z_dev = jax.device_put(jnp.asarray(Z), in_sh)

    LU, piv = factor_c_q(C_dev, mesh, vertex_mu_L=1, n_rmu_logical=n_log,
                         solver_kind='lu')
    z_lu = np.asarray(jax.device_get(solve_zeta(
        LU, Z_dev, mesh, nq, vertex_mu_L=1, solver_kind='lu',
        n_rmu_logical=n_log, lu_piv=piv)))
    Cp, _ = factor_c_q(C_dev, mesh, vertex_mu_L=1, n_rmu_logical=n_log,
                       solver_kind='transverse_rank_truncate',
                       transverse_zeta_rcond=tau)
    z_rt = np.asarray(jax.device_get(solve_zeta(
        Cp, Z_dev, mesh, nq, vertex_mu_L=1,
        solver_kind='transverse_rank_truncate', n_rmu_logical=n_log)))

    kept_rel = 0.0
    null_ridge = 0.0
    null_rt = 0.0
    for q, (lam, V) in enumerate(spectra):
        keep = np.abs(lam) > (tau * np.abs(lam).max())
        Vk = V[:, keep]
        Vn = V[:, ~keep]
        d = Vk.conj().T @ (z_lu[q, :n_log, :] - z_rt[q, :n_log, :])
        kept_rel = max(kept_rel,
                       float(np.max(np.abs(d))
                             / np.max(np.abs(Vk.conj().T
                                             @ z_rt[q, :n_log, :]))))
        null_ridge = max(null_ridge, float(np.max(np.abs(
            Vn.conj().T @ z_lu[q, :n_log, :]))))
        null_rt = max(null_rt, float(np.max(np.abs(
            Vn.conj().T @ z_rt[q, :n_log, :]))))
    print(json.dumps({
        "kept_rel": kept_rel,          # families agree here (~ridge level)
        "null_ridge": null_ridge,      # 1/ridge-scale amplification
        "null_rt": null_rt,            # exactly what truncation removes
    }))
    return 0


def _worker_resolver() -> int:
    """Child: resolver semantics of the new family."""
    import numpy as np
    import jax
    from jax.sharding import Mesh

    from isdf.core import (
        _resolve_solver_kind_transverse,
        _resolve_zeta_gather,
    )

    devs = jax.devices()
    if len(devs) < _NDEV:
        print(json.dumps({"skip": f"only {len(devs)} devices"}))
        return 0
    mesh22 = Mesh(np.asarray(devs[:4]).reshape(2, 2), ('x', 'y'))

    out = {}
    # Any count resolves on the rank_truncate family — 135 does not
    # divide a 2x2 mesh, which demotes/refuses on the LU family.
    out["rt_any_count"] = _resolve_solver_kind_transverse(
        mesh22, "auto", n_rmu_logical=135,
        transverse_zeta_solve="rank_truncate")
    out["rt_off"] = _resolve_solver_kind_transverse(
        mesh22, "off", n_rmu_logical=135,
        transverse_zeta_solve="rank_truncate")
    # Explicit LU backend + rank_truncate is a CONFLICT: refuse.
    try:
        _resolve_solver_kind_transverse(
            mesh22, "scalapack", n_rmu_logical=136,
            transverse_zeta_solve="rank_truncate")
        out["conflict_raises"] = False
    except ValueError:
        out["conflict_raises"] = True
    # Ridge family untouched: auto on a CPU mesh keeps the local LU.
    out["ridge_auto"] = _resolve_solver_kind_transverse(
        mesh22, "auto", n_rmu_logical=135,
        transverse_zeta_solve="ridge")
    # Bad family name refuses.
    try:
        _resolve_solver_kind_transverse(
            mesh22, "auto", transverse_zeta_solve="svd")
        out["bad_family_raises"] = False
    except ValueError:
        out["bad_family_raises"] = True
    # The back-solve tier key: 'distributed' on a RIDGE-family transverse
    # channel still resolves to per_q (the documented one-key-two-
    # channels demotion).
    out["ridge_dist_tier"] = _resolve_zeta_gather(
        "distributed", n_rmu=64, nq=4, mesh_xy=mesh22, vertex_mu_L=1,
        charge_zeta_solve="rank_truncate", transverse_zeta_solve="ridge")
    print(json.dumps(out))
    return 0


def _run_worker(tag: str, timeout: int = 900):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + f" --xla_force_host_platform_device_count={_NDEV}"
                        ).strip()
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), tag],
        env=env, capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, (
        f"worker {tag} failed rc={res.returncode}\nSTDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}")
    line = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert line, f"no JSON from worker.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return json.loads(line[-1])


def test_local_family_is_bit_identical_across_schedules():
    """Within the rank_truncate family: exact bit equality across
    meshes, factor schedules, gather tiers, q-chunkings and r-chunks;
    tight agreement with an independent numpy pseudo-inverse; ζ pad
    rows exactly zero."""
    out = _run_worker("worker_rt")
    if "skip" in out:
        pytest.skip(f"rank_truncate gate: {out['skip']}")
    bad = sorted(k for k, v in out["exact"].items() if not v)
    assert not bad, (
        f"transverse rank_truncate drifts across schedules on {bad}; "
        f"the family's bit-identity contract is broken")
    assert out["ref_ok"], (
        f"numpy pseudo-inverse reference disagrees "
        f"(max rel {out['max_ref_rel']:.3e})")
    assert out["pad_ok"], "zeta pad rows are not exactly zero"


def test_truncation_drops_the_trs_near_null_pair():
    """n_keep == n_log - 2 on the fixture (both TRS modes cut)."""
    out = _run_worker("worker_rt")
    if "skip" in out:
        pytest.skip(f"rank_truncate gate: {out['skip']}")
    want = out["n_log"] - 2
    assert all(r == want for r in out["ranks"]), (
        f"C+ ranks {out['ranks']} != {want}: the |lambda| cut did not "
        f"remove exactly the TRS near-null pair")


def test_ridge_vs_rank_truncate_agree_on_kept_subspace():
    """Different algorithms, not bit-comparable: kept-subspace agreement
    at ~ridge level; the near-null amplification exists only on the
    ridge side (it is what truncation removes)."""
    out = _run_worker("worker_gauge")
    assert out["kept_rel"] < 1e-6, (
        f"families disagree on the KEPT subspace ({out['kept_rel']:.3e}) "
        f"— that is an algorithm bug, not a gauge difference")
    assert out["null_rt"] < 1e-8, (
        f"rank_truncate zeta carries near-null content "
        f"({out['null_rt']:.3e}); the cut is not being applied")
    assert out["null_ridge"] > 1e2 * max(out["null_rt"], 1e-300), (
        f"fixture failed to exhibit the ridge-side near-null "
        f"amplification (ridge {out['null_ridge']:.3e} vs rt "
        f"{out['null_rt']:.3e}) — the gauge comparison is vacuous")


def test_resolver_semantics():
    """Family selection + conflicts + untouched ridge defaults."""
    out = _run_worker("worker_resolver")
    if "skip" in out:
        pytest.skip(f"resolver gate: {out['skip']}")
    assert out["rt_any_count"] == "transverse_rank_truncate"
    assert out["rt_off"] == "transverse_rank_truncate"
    assert out["conflict_raises"], (
        "explicit distributed_lu + rank_truncate must refuse")
    assert out["ridge_auto"] == "lu"
    assert out["bad_family_raises"]
    assert out["ridge_dist_tier"] == "per_q"


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    if tag == "worker_rt":
        sys.exit(_worker_rt())
    if tag == "worker_gauge":
        sys.exit(_worker_gauge())
    if tag == "worker_resolver":
        sys.exit(_worker_resolver())
    print(f"unknown worker tag {tag!r}", file=sys.stderr)
    sys.exit(2)
