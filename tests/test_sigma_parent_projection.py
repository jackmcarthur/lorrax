"""The raw-parent Σ route equals the full-k face route, and its antiunitary
rule is the transpose.

Two seams of ``gw.wavefunction_bundle.ParentSigmaRoute`` are checked on the
synthetic symmetric fixture of ``tests/test_isdf_zq_parent_parity.py`` (a
glide, a genuine k reduction, a time-reversed row, SU(2) mixing):

* G: ``build_G`` on the PACKED parent faces with ``k_unfold_plan`` equals
  ``build_G`` on the full-k faces, with complex (real-time-like) band phases
  so the antiunitary rule of the operator transport is live.
* Parents-only bundle: a face bundle with BOTH full-k faces ``None`` and
  the carrier as its only ψ names the full-k face shape and the route's
  parent shapes to the kernel factories (``sigma_face_kernel_kwargs``).
* Σ projection: ``ppm_tau_kernel._make_project_ri_reduce_scatter(parent_route)``
  — select the parents' rows of a full-k operator, project with the
  CANONICAL parent faces, broadcast the band matrix — equals the full-k face
  projector on an operator that transforms like a Green function
  (``plan.finish_green`` of a NON-Hermitian parent operator).  The
  ``conj`` broadcast rule is shown to be wrong on the time-reversed row and
  right on the unitary rows, which is why the route names ``transpose``.

Runs in a CPU subprocess with four emulated devices on a 2x2 mesh; the
distributed GEMM plan is replaced by a local einsum exactly as
``tests/test_centroid_k_unfold.py`` does.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_TOL = 1.0e-11


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype("complex128")


def _worker() -> int:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from types import SimpleNamespace

    import distrib_la

    def _local_gemm_plan(_mesh, **_kwargs):
        return lambda a, b: jnp.einsum("qmk,qkn->qmn", a, b, optimize=True)

    distrib_la.gemm_plan = _local_gemm_plan

    from common.contract_bands import contract_bands_block_reshard
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.greens_function_kernel import build_G
    from gw.ppm_tau_kernel import _make_project_ri_reduce_scatter
    from gw.wavefunction_bundle import ParentSigmaRoute
    from symmetry_maps import (
        centroid_source_map_and_wrap, spinor_rotation_for_sym_row,
        unfold_file_wedge_band_operator)

    devs = jax.devices()
    PX = PY = 2
    if len(devs) < PX * PY:
        print(json.dumps({"skip": f"only {len(devs)} devices (<{PX*PY})"}))
        return 0
    mesh = Mesh(np.asarray(devs[:PX * PY]).reshape(PX, PY), ("x", "y"))
    rng = np.random.default_rng(20260905)

    fft_grid = (4, 4, 4)
    kgrid = (2, 2, 1)
    nk = 4
    ns, nb = 2, 8
    swap = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    ops = np.stack([np.eye(3, dtype=np.int64), swap])
    tnp = np.asarray([[0.0, 0.0, 0.0], [np.pi, np.pi, 0.0]])
    n_tran = 2
    kfrac = np.asarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]) / np.asarray(
        kgrid, dtype=np.float64)
    # Raw rows k0, k1, k3 are their own full-k rows (identity, as
    # SymMaps.kirr_fullids guarantees); k2 = Θ·σ_xy k1 is the one child and
    # carries the antiunitary row, with a nontrivial spatial part.
    irr = np.asarray([0, 1, 1, 2], dtype=np.int32)
    sym = np.asarray([0, 0, n_tran + 1, 0], dtype=np.int32)
    parent_rows = np.asarray([0, 1, 3], dtype=np.int32)   # full rows of raw rows
    parent_k = kfrac[parent_rows]
    n_parent = 3
    theta = 0.7
    U1 = np.asarray([[np.cos(theta), -1j * np.sin(theta)],
                     [-1j * np.sin(theta), np.cos(theta)]])
    U_spatial = np.stack([np.eye(2, dtype=np.complex128), U1])

    def spinor_action(rows, *, nspinor):
        return spinor_rotation_for_sym_row(
            U_spatial, np.asarray(rows), n_tran, nspinor=nspinor)

    sym_mats_k = np.concatenate([ops.transpose(0, 2, 1), -ops.transpose(0, 2, 1)])
    sym_fx = SimpleNamespace(
        sym_matrices=ops, translations=tnp, sym_mats_k=sym_mats_k,
        irr_idx_k=irr, sym_idx_k=sym, spinor_action=spinor_action,
        unfolded_kpts=kfrac, kirr_fullids=parent_rows, nk_red=n_parent,
        nk_tot=nk)

    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in fft_grid), indexing="ij")
    grid_pts = np.stack(
        [ix.reshape(-1), iy.reshape(-1), iz.reshape(-1)], axis=1).astype(np.int32)
    perm_g, _ = centroid_source_map_and_wrap(
        grid_pts, ops, tnp, fft_grid, extend_trs=True)
    cent_flat = []
    for seed in (0, 5, 22, 27, 41, 50, 60, 63, 9, 14):
        orbit = sorted({int(perm_g[s, seed]) for s in range(2 * n_tran)})
        if len(cent_flat) + len(orbit) > 8 or any(c in cent_flat for c in orbit):
            continue
        cent_flat.extend(orbit)
        if len(cent_flat) == 8:
            break
    cent_flat = np.asarray(sorted(cent_flat))
    cent_idx = grid_pts[cent_flat]
    n_mu = 8
    perm_c, L_c = centroid_source_map_and_wrap(
        cent_idx, ops, tnp, fft_grid, extend_trs=True)

    plan = build_centroid_k_unfold_plan(
        sym_fx, cent_idx, fft_grid, mesh, nspinor=ns,
        parent_k_frac=parent_k, canonical_centroid_extent=n_mu)
    assert np.array_equal(plan.parent_full_rows, parent_rows)

    # ---- parents at the centroids and their typed-action children -------
    psi_parent = _crand(rng, n_parent, nb, ns, n_mu)
    psi_full = np.empty((nk, nb, ns, n_mu), dtype=np.complex128)
    for k in range(nk):
        p, s = int(irr[k]), int(sym[k])
        U_eff = spinor_action(np.asarray([s]), nspinor=ns)[0]
        val = psi_parent[p][:, :, perm_c[s]] * np.exp(
            2j * np.pi * (L_c[s].astype(np.float64) @ parent_k[p]))[None, None, :]
        if s >= n_tran:
            val = np.conj(val)
        psi_full[k] = np.einsum("ac,ncr->nar", U_eff, val, optimize=True)

    mun_sh = NamedSharding(mesh, P(None, None, "x", "y"))
    nmu_sh = NamedSharding(mesh, P(None, "x", None, "y"))
    psi_nmu_full = jax.device_put(jnp.asarray(psi_full), nmu_sh)
    psi_mun_full = jax.device_put(
        jnp.asarray(psi_full.transpose(0, 2, 3, 1)), mun_sh)
    psi_nmu_par = jax.device_put(jnp.asarray(psi_parent), nmu_sh)
    psi_mun_par = jax.device_put(
        jnp.asarray(psi_parent.transpose(0, 2, 3, 1)), mun_sh)
    with mesh:
        psi_nmu_pk, psi_mun_pk = plan.pack_face_pair(psi_nmu_par, psi_mun_par)

    # ---- G: parent contraction + plan == full-k contraction --------------
    phases_parent = _crand(rng, n_parent, nb)
    phases_full = phases_parent[irr]
    gemm = _local_gemm_plan(mesh)
    G_full = np.asarray(jax.block_until_ready(build_G(
        psi_mun_full, psi_nmu_full, phases=jnp.asarray(phases_full),
        layout="face", gemm=gemm)))
    G_par = np.asarray(jax.block_until_ready(build_G(
        psi_mun_pk, psi_nmu_pk, phases=jnp.asarray(phases_parent),
        layout="face", gemm=gemm, k_unfold_plan=plan)))
    g_rel = float(np.max(np.abs(G_par - G_full))) / float(np.max(np.abs(G_full)))

    # ---- a full-k operator that transforms like a Green function --------
    O_parent = _crand(rng, n_parent, ns, plan.n_centroid_packed, ns,
                      plan.n_centroid_packed)
    O_parent = jax.device_put(
        jnp.asarray(O_parent),
        NamedSharding(mesh, P(None, None, "x", None, "y")))
    with mesh:
        sigma_full = jax.block_until_ready(plan.finish_green(O_parent))
    assert sigma_full.shape == (nk, ns, n_mu, ns, n_mu)

    full_project = contract_bands_block_reshard(
        mesh, layout="face", face_shape=(nk, nb, n_mu, ns))
    ref = np.asarray(jax.block_until_ready(
        full_project(psi_nmu_full, sigma_full, psi_mun_full)))

    route = ParentSigmaRoute(
        plan=plan, k_rows=parent_rows, sym=sym_fx,
        g_face_shape=(n_parent, nb, plan.n_centroid_packed, ns),
        proj_face_shape=(n_parent, nb, n_mu, ns))
    project = _make_project_ri_reduce_scatter(
        mesh, merged_x=True, layout="face", face_shape=(nk, nb, n_mu, ns),
        parent_route=route)
    got = np.asarray(jax.block_until_ready(jax.jit(project)(
        psi_nmu_par, sigma_full, psi_mun_par)))
    scale = float(np.max(np.abs(ref)))
    proj_rel = float(np.max(np.abs(got - ref))) / scale

    # The discriminator: the conj rule on the time-reversed row (k3).
    inner = contract_bands_block_reshard(
        mesh, layout="face", face_shape=(n_parent, nb, n_mu, ns))
    on_parents = inner(psi_nmu_par, sigma_full[parent_rows], psi_mun_par)
    conj_rule = np.asarray(jax.block_until_ready(
        unfold_file_wedge_band_operator(sym_fx, on_parents, trs_rule="conj")))
    trs_rows = np.flatnonzero(sym >= n_tran)
    uni_rows = np.flatnonzero(sym < n_tran)
    conj_rel_trs = float(np.max(np.abs(conj_rule[trs_rows] - ref[trs_rows]))) / scale
    conj_rel_uni = float(np.max(np.abs(conj_rule[uni_rows] - ref[uni_rows]))) / scale

    # A parents-only bundle -- face layout, BOTH full-k faces None, the
    # carrier its only ψ -- must name the full-k extents to every kernel
    # factory exactly as a full-k bundle does (gw_init parents-only storage).
    from gw.wavefunction_bundle import (
        BandSlices, ParentGreenCarrier, Wavefunctions, padded_centroid_extent,
        sigma_face_kernel_kwargs)
    zeros_full = jnp.zeros((nk, nb))
    carrier = ParentGreenCarrier(
        psi_nmu=psi_nmu_pk, psi_mun=psi_mun_pk,
        enk=jnp.zeros((n_parent, nb)), occ=jnp.zeros((n_parent, nb)),
        plan=plan, psi_nmu_canonical=psi_nmu_par, psi_mun_canonical=psi_mun_par)
    bare = Wavefunctions(
        enk=zeros_full, occ=zeros_full,
        slices=BandSlices.from_band_edges(0, 0, nb // 2, nb, nb),
        layout="face", green_parent=carrier)
    kw = sigma_face_kernel_kwargs(bare)
    parents_only_ok = bool(
        tuple(kw["face_shape"]) == (nk, nb, n_mu, ns)
        and padded_centroid_extent(bare) == n_mu
        and tuple(kw["parent_route"].g_face_shape)
        == (n_parent, nb, plan.n_centroid_packed, ns)
        and tuple(kw["parent_route"].proj_face_shape) == (n_parent, nb, n_mu, ns))

    print(json.dumps({
        "g_rel": g_rel, "proj_rel": proj_rel,
        "conj_rule_rel_on_tr_rows": conj_rel_trs,
        "conj_rule_rel_on_unitary_rows": conj_rel_uni,
        "parents_only_bundle_names_full_k_shapes": parents_only_ok,
    }))
    return 0


def _run_worker(ndev: int = 4, timeout: int = 600):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["LORRAX_CONV_KPAIR_FFI"] = "off"
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={ndev}").strip()
    _repo_src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env["PYTHONPATH"] = _repo_src + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "worker"],
        env=env, capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, (
        f"worker failed rc={res.returncode}\nSTDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}")
    lines = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, f"no JSON from worker.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return json.loads(lines[-1])


def test_parent_sigma_route_matches_full_k_and_uses_the_transpose_rule():
    out = _run_worker()
    if "skip" in out:
        pytest.skip(f"parent Σ route parity: {out['skip']}")
    assert out["g_rel"] < _TOL, f"G on parents vs full k: {out}"
    assert out["proj_rel"] < _TOL, f"Σ projection on parents vs full k: {out}"
    # The conj rule is right on unitary rows and wrong on the antiunitary
    # one: the transpose is the discriminating choice, not a convention.
    assert out["conj_rule_rel_on_unitary_rows"] < _TOL, out
    assert out["conj_rule_rel_on_tr_rows"] > 0.1, out
    assert out["parents_only_bundle_names_full_k_shapes"] is True, out


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        sys.exit(_worker())
    raise SystemExit("usage: python test_sigma_parent_projection.py worker")
