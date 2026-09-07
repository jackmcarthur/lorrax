"""The raw-parent Σ route equals the full-k face route, and its antiunitary
rule is the transpose.

Two seams of ``gw.centroid_k_unfold.CentroidKUnfoldPlan`` are checked on the
synthetic symmetric fixture of ``tests/test_isdf_zq_parent_parity.py`` (a
glide, a genuine k reduction, a time-reversed row, SU(2) mixing):

* G: ``build_G`` on the PACKED parent faces with ``k_unfold_plan`` equals
  ``build_G`` on the full-k faces, with complex (real-time-like) band phases
  so the antiunitary rule of the operator transport is live.
* Parents-only bundle: a face bundle with BOTH full-k faces ``None`` and
  the carrier as its only ψ names the full-k face shape and the route's
  parent shapes to the kernel factories (``sigma_face_kernel_kwargs``).
* Fractional-occupation contour χ0 (``w_isdf.compute_chi0_contour_fractional``)
  on the parents-only bundle equals the full-k face bundle (CPU steps
  without the host FFT FFI report partial scope; run the worker on GPUs).
* Fractional static-Γ and direct-q pair scans on the parents-only bundle
  equal the full-k face bundle: each band tile is unfolded from the packed
  parents inside the scan (``symmetry_maps.unfold_wavefunction_local``).
* q→0 head wings (``qsgw_head.head_wings_sharded`` and the static wings) on
  the parents-only bundle equal the full-k face bundle: the children are
  streamed one parent star at a time from the packed parents
  (``w_isdf.iter_parent_children_faces``).
* SC rotation: ``wavefunction_bundle.rotate_wavefunctions`` on the
  parents-only bundle rotates the carrier's faces with U at the parents'
  rows and equals the rotated full-k faces read at those rows.
* Σ projection: ``ppm_tau_kernel._make_project_ri_reduce_scatter(k_unfold_plan)``
  — select the parents' rows of a full-k operator, project with the parent
  faces, then unfold the final band matrix — equals the full-k face projector on an
  operator that transforms like a Green function (``plan.unfold_operator``
  of a NON-Hermitian parent operator).  Every face is in the run's packed
  centroid order, the one in-memory order.  The
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
        gemm = lambda a, b: jnp.einsum("qmk,qkn->qmn", a, b, optimize=True)
        gemm.mesh = _mesh
        gemm.in_sharding_a = NamedSharding(_mesh, P(None, "x", "y"))
        gemm.in_sharding_b = gemm.in_sharding_a
        return gemm

    distrib_la.gemm_plan = _local_gemm_plan

    from common.contract_bands import contract_bands_block_reshard
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.greens_function_kernel import build_G
    from gw.ppm_tau_kernel import _make_project_ri_reduce_scatter
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
        parent_k_frac=parent_k)
    assert np.array_equal(plan.parent_full_rows, parent_rows)
    n_pk = int(plan.n_centroid_packed)
    pack = plan.layout.axis.pack_host   # stands in for the loader's sampling

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

    # Every face is in the run's PACKED centroid order (the one in-memory
    # order); the full-k reference route runs in it too.
    mun_sh = NamedSharding(mesh, P(None, None, "x", "y"))
    nmu_sh = NamedSharding(mesh, P(None, "x", None, "y"))
    psi_nmu_full = jax.device_put(jnp.asarray(pack(psi_full, axis=3)), nmu_sh)
    psi_mun_full = jax.device_put(
        jnp.asarray(pack(psi_full.transpose(0, 2, 3, 1), axis=2)), mun_sh)
    psi_nmu_pk = jax.device_put(jnp.asarray(pack(psi_parent, axis=3)), nmu_sh)
    psi_mun_pk = jax.device_put(
        jnp.asarray(pack(psi_parent.transpose(0, 2, 3, 1), axis=2)), mun_sh)

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
        sigma_full = jax.block_until_ready(plan.unfold_operator(O_parent))
    assert sigma_full.shape == (nk, ns, n_pk, ns, n_pk)

    full_project = contract_bands_block_reshard(
        mesh, layout="face", face_shape=(nk, nb, n_pk, ns))
    ref = np.asarray(jax.block_until_ready(
        full_project(psi_nmu_full, sigma_full, psi_mun_full)))

    project = _make_project_ri_reduce_scatter(
        mesh, merged_x=True, layout="face", face_shape=(nk, nb, n_pk, ns),
        k_unfold_plan=plan)
    projected_parents = jax.jit(project)(psi_nmu_pk, sigma_full, psi_mun_pk)
    assert projected_parents.shape == (n_parent, nb, nb)
    got = np.asarray(jax.block_until_ready(unfold_file_wedge_band_operator(
        sym_fx, projected_parents, trs_rule="transpose")))
    scale = float(np.max(np.abs(ref)))
    proj_rel = float(np.max(np.abs(got - ref))) / scale

    # The discriminator: the conj rule on the time-reversed row (k3).
    inner = contract_bands_block_reshard(
        mesh, layout="face", face_shape=(n_parent, nb, n_pk, ns))
    on_parents = inner(psi_nmu_pk, sigma_full[parent_rows], psi_mun_pk)
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
        plan=plan)
    bare = Wavefunctions(
        enk=zeros_full, occ=zeros_full,
        slices=BandSlices.from_band_edges(0, 0, nb // 2, nb, nb),
        layout="face", green_parent=carrier)
    kw = sigma_face_kernel_kwargs(bare)
    parents_only_ok = bool(
        tuple(kw["face_shape"]) == (nk, nb, n_pk, ns)
        and padded_centroid_extent(bare) == n_pk
        and kw["k_unfold_plan"] is plan)

    # Fractional-occupation CONTOUR χ0 on the parents-only bundle equals the
    # full-k face bundle (its two G's ride the same parent transport).
    from gw.efermi import mp1_occupations
    from gw.w_isdf import compute_chi0_contour_fractional
    enk_parent = rng.uniform(-1.0, 1.0, size=(n_parent, nb))
    enk_parent[:, :2] -= 40.0
    enk_parent[:, -2:] += 40.0
    enk_full_np = enk_parent[irr]                      # star-invariant table
    enk_j = jnp.asarray(enk_full_np)
    mu_f = float(np.median(enk_full_np))
    f_kn = mp1_occupations(enk_j, mu_f, 0.15)
    slices = BandSlices.from_band_edges(0, 0, nb // 2, nb, nb)
    wfns_full = Wavefunctions(
        psi_nmu=psi_nmu_full, psi_mun=psi_mun_full, enk=enk_j, occ=f_kn,
        slices=slices, layout="face")
    carrier_e = ParentGreenCarrier(
        psi_nmu=psi_nmu_pk, psi_mun=psi_mun_pk,
        enk=jnp.asarray(enk_parent), occ=jnp.asarray(np.asarray(f_kn)[parent_rows]),
        plan=plan)
    wfns_par = Wavefunctions(
        enk=enk_j, occ=f_kn, slices=slices, layout="face",
        green_parent=carrier_e)
    meta = SimpleNamespace(nk_tot=nk, nkx=kgrid[0], nky=kgrid[1], nkz=kgrid[2])
    t_nodes = np.asarray([0.1, 0.5, 1.3])
    w_rows = np.asarray([0.4, 0.3, 0.2])
    z_c = np.asarray([0.05 + 0.2j])
    try:
        chi_full = np.asarray(jax.block_until_ready(compute_chi0_contour_fractional(
            wfns_full, t_nodes, w_rows, z_c, meta, mesh,
            occupations=f_kn, energy_reference=mu_f)))
        chi_par = np.asarray(jax.block_until_ready(compute_chi0_contour_fractional(
            wfns_par, t_nodes, w_rows, z_c, meta, mesh,
            occupations=f_kn, energy_reference=mu_f)))
        frac_contour = float(np.max(np.abs(chi_par - chi_full))) / float(
            np.max(np.abs(chi_full)))
    except RuntimeError as exc:
        if "FFTW3-ABI host backend is unavailable" not in str(exc):
            raise
        frac_contour = "skip: host FFT FFI backend unavailable"

    # The two fractional PAIR SCANS (static Γ, direct q) need ψ itself at
    # every k: on the parents-only bundle each band tile is unfolded from
    # the packed parents inside the scan (symmetry_maps.
    # unfold_wavefunction_local).  No FFT is involved, so this runs on CPU.
    from gw.efermi import OccupationState, mp1_negative_derivative
    from gw.w_isdf import (
        compute_chi0_direct_fractional, compute_chi0_static_fractional_gamma)
    surf = mp1_negative_derivative(enk_j, mu_f, 0.15)
    g_full = np.asarray(jax.block_until_ready(compute_chi0_static_fractional_gamma(
        wfns_full, enk_j, f_kn, surf, meta, mesh, nb_logical=nb - 1)))
    g_par = np.asarray(jax.block_until_ready(compute_chi0_static_fractional_gamma(
        wfns_par, enk_j, f_kn, surf, meta, mesh, nb_logical=nb - 1)))
    frac_gamma = float(np.max(np.abs(g_par - g_full))) / float(np.max(np.abs(g_full)))
    kfrac_i = np.rint(kfrac * np.asarray(kgrid)).astype(int)
    kminq = np.asarray([[2 * ((kfrac_i[k, 0] - kfrac_i[q, 0]) % 2)
                         + ((kfrac_i[k, 1] - kfrac_i[q, 1]) % 2)
                         for k in range(nk)] for q in range(nk)], dtype=np.int32)
    occ_state = OccupationState(
        f_kn=f_kn, mu_ry=mu_f, smearing_family="mp1", smearing_width_ry=0.15,
        n_electrons=float(np.sum(np.asarray(f_kn))))
    z_direct = np.asarray([0.03 + 0.1j, 0.2 + 0.05j])
    d_full = np.asarray(jax.block_until_ready(compute_chi0_direct_fractional(
        wfns_full, z_direct, meta, mesh, occupation_state=occ_state,
        kminq_rows=kminq, nb_logical=nb - 1)))
    d_par = np.asarray(jax.block_until_ready(compute_chi0_direct_fractional(
        wfns_par, z_direct, meta, mesh, occupation_state=occ_state,
        kminq_rows=kminq, nb_logical=nb - 1)))
    frac_direct = float(np.max(np.abs(d_par - d_full))) / float(np.max(np.abs(d_full)))
    # The q->0 head wings on the parents-only bundle equal the full-k face
    # bundle: the children are streamed one parent star at a time
    # (w_isdf.iter_parent_children_faces) with the velocity read at every k.
    from gw.qsgw_head import head_wings_sharded, static_head_wings_sharded
    v_np = _crand(rng, 3, nk, nb, nb)
    v_np = 0.5 * (v_np + np.conj(np.swapaxes(v_np, -1, -2)))
    omegas = np.asarray([0.0, 0.3 + 0.05j])
    wing_kw = dict(mesh=mesh, nb_logical=nb - 1, nk_tot=nk, nspin=1,
                   nspinor=ns, eta_ry=0.02, surface_weight_kn=surf)
    Yf, Zf = head_wings_sharded(jnp.asarray(v_np), wfns_full, enk_j, f_kn,
                                omegas, **wing_kw)
    Yp, Zp = head_wings_sharded(jnp.asarray(v_np), wfns_par, enk_j, f_kn,
                                omegas, **wing_kw)
    jax.block_until_ready((Yf, Zf, Yp, Zp))
    wings_rel = max(
        float(np.max(np.abs(np.asarray(Yp) - np.asarray(Yf)))) / float(np.max(np.abs(np.asarray(Yf)))),
        float(np.max(np.abs(np.asarray(Zp) - np.asarray(Zf)))) / float(np.max(np.abs(np.asarray(Zf)))))
    sf = static_head_wings_sharded(wfns_full, surf, mesh=mesh, nb_logical=nb - 1,
                                   nk_tot=nk, nspin=1, nspinor=ns)
    sp = static_head_wings_sharded(wfns_par, surf, mesh=mesh, nb_logical=nb - 1,
                                   nk_tot=nk, nspin=1, nspinor=ns)
    jax.block_until_ready((sf, sp))
    static_wings_rel = max(
        float(np.max(np.abs(np.asarray(sp[i]) - np.asarray(sf[i])))) / float(np.max(np.abs(np.asarray(sf[i]))))
        for i in range(2))
    # The self-consistent map's rotation on a parents-only bundle: rotating
    # the carrier with U on the parents' rows equals rotating the full-k
    # faces and reading the parents' rows back (children share the parent's
    # U in the transported gauge, conjugated on antiunitary rows).
    from gw.wavefunction_bundle import rotate_wavefunctions
    U_par_np = np.stack([np.linalg.qr(_crand(rng, nb, nb))[0] for _ in range(n_parent)])
    U_full_np = np.stack([np.conj(U_par_np[irr[k]]) if sym[k] >= n_tran else U_par_np[irr[k]]
                          for k in range(nk)])
    e_new_par = rng.uniform(-1.0, 1.0, size=(n_parent, nb))
    e_new_full = jnp.asarray(e_new_par[irr])
    rot_full = rotate_wavefunctions(
        wfns_full, jnp.asarray(U_full_np), enk_active_new=e_new_full, efermi=None,
        mesh_xy=mesh, active_slice=slice(0, nb))
    rot_par = rotate_wavefunctions(
        wfns_par, jnp.asarray(U_full_np), enk_active_new=e_new_full, efermi=None,
        mesh_xy=mesh, active_slice=slice(0, nb))
    rc = rot_par.green_parent
    full_nmu = np.asarray(rot_full.psi_nmu)[parent_rows]
    full_mun = np.asarray(rot_full.psi_mun)[parent_rows]
    rot_rel = max(
        float(np.max(np.abs(np.asarray(rc.psi_nmu) - full_nmu))) / float(np.max(np.abs(full_nmu))),
        float(np.max(np.abs(np.asarray(rc.psi_mun) - full_mun))) / float(np.max(np.abs(full_mun))))

    print(json.dumps({
        "sc_rotation_parents_vs_full_rel": rot_rel,
        "head_wings_parents_vs_full_rel": wings_rel,
        "static_head_wings_parents_vs_full_rel": static_wings_rel,
        "g_rel": g_rel, "proj_rel": proj_rel,
        "conj_rule_rel_on_tr_rows": conj_rel_trs,
        "conj_rule_rel_on_unitary_rows": conj_rel_uni,
        "parents_only_bundle_names_full_k_shapes": parents_only_ok,
        "fractional_contour_parents_vs_full_rel": frac_contour,
        "fractional_static_gamma_parents_vs_full_rel": frac_gamma,
        "fractional_direct_q_parents_vs_full_rel": frac_direct,
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


_WORKER_RESULT: dict = {}


def _worker_result():
    """One CPU worker per module; every cell below reads its JSON."""
    if not _WORKER_RESULT:
        _WORKER_RESULT.update(_run_worker())
    out = _WORKER_RESULT
    if "skip" in out:
        pytest.skip(f"parent Σ route parity: {out['skip']}")
    return out


def test_parent_sigma_route_matches_full_k_and_uses_the_transpose_rule():
    out = _worker_result()
    assert out["g_rel"] < _TOL, f"G on parents vs full k: {out}"
    assert out["proj_rel"] < _TOL, f"Σ projection on parents vs full k: {out}"
    # The conj rule is right on unitary rows and wrong on the antiunitary
    # one: the transpose is the discriminating choice, not a convention.
    assert out["conj_rule_rel_on_unitary_rows"] < _TOL, out
    assert out["conj_rule_rel_on_tr_rows"] > 0.1, out
    assert out["parents_only_bundle_names_full_k_shapes"] is True, out
    assert out["fractional_static_gamma_parents_vs_full_rel"] < 1.0e-10, out
    assert out["sc_rotation_parents_vs_full_rel"] < 1.0e-10, out
    assert out["fractional_direct_q_parents_vs_full_rel"] < 1.0e-10, out
    assert out["head_wings_parents_vs_full_rel"] < 1.0e-10, out
    assert out["static_head_wings_parents_vs_full_rel"] < 1.0e-10, out


def test_fractional_contour_chi0_on_parents_matches_full_k():
    """Its own cell: without the host FFT FFI this arm is SKIPPED, not
    folded into a passing test (the P4 GPU legs cover it)."""
    fc = _worker_result()["fractional_contour_parents_vs_full_rel"]
    if isinstance(fc, str):
        pytest.skip("fractional contour arm: " + fc)
    assert fc < 1.0e-10, f"fractional contour χ0 on parents vs full k: {fc}"


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        sys.exit(_worker())
    raise SystemExit("usage: python test_sigma_parent_projection.py worker")
