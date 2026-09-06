"""Algebra parity: the raw-parent ζ-fit kernels against the full-k face route.

``isdf.core._z_q_face_parent`` builds the pair-density RHS ``Z_q(mu, r)`` on
one orbit-closed real-grid tile from the raw WFN parents only and transports
the completed projector to full k through ``symmetry_maps``;
``isdf.core._c_q_face_parent`` does the same for the CCT Gram.  Their claim is
the identity the design brief names decisive:

    contract(unfold ψ) == unfold(contract ψ)

so the fixture is a full-k wavefunction set GENERATED from raw parents by
the typed action (spatial source map + lattice-wrap Bloch phase, spinor
rotation, complex conjugation on the antiunitary rows), a real-space grid
group with a nonsymmorphic glide ({E, σ_xy | (½,½,0)} on a 4x4x4 box) that
gives a genuine k reduction (k1 ↔ k2 on the 2x2x1 grid), a time-reversed row
(k3), and two-component spinors with a non-diagonal SU(2) action.  The
established full-k face kernels then see the children, the parent kernels see
only the parents, and the two must agree at floating-point noise on every
tile slot; pad slots must be exactly zero.  Every symmetry table comes from
the service (``centroid_source_map_and_wrap``, ``spinor_rotation_for_sym_row``)
— the test derives no convention of its own.

Runs in a CPU subprocess with four emulated devices on a 2x2 mesh, exactly as
``tests/test_isdf_zq_face_parity.py`` does; no GPU and no ``lx run`` needed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_TOL = 1.0e-10

_CASES = (
    ("ns1_scalar", dict(ns=1, seed=11)),
    ("ns2_spinor", dict(ns=2, seed=12)),
    ("ns4_charge", dict(ns=4, seed=13)),
    ("ns4_current1", dict(ns=4, seed=13, vertex=1)),
    ("ns4_current2", dict(ns=4, seed=13, vertex=2)),
    ("ns4_current3", dict(ns=4, seed=13, vertex=3)),
)
_CASES_BY_NAME = dict(_CASES)


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype("complex128")


def _worker(case_name: str) -> int:
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from types import SimpleNamespace

    from isdf.core import (
        build_psi_r_cache_sm, c_q_from_psi_sm, z_q_from_psi_sm)
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from symmetry_maps import (
        centroid_source_map_and_wrap, spinor_rotation_for_sym_row)

    case = _CASES_BY_NAME[case_name]
    ns = int(case["ns"])
    vertex = int(case.get("vertex", 0))
    from common.gamma_matrices import gamma_perm_phase
    gamma = None if vertex == 0 else gamma_perm_phase(vertex)
    rng = np.random.default_rng(int(case["seed"]))

    devs = jax.devices()
    PX = PY = 2
    if len(devs) < PX * PY:
        print(json.dumps({"skip": f"only {len(devs)} devices (<{PX*PY})"}))
        return 0
    mesh = Mesh(np.asarray(devs[:PX * PY]).reshape(PX, PY), ("x", "y"))

    # ---- geometry: a real space group of order two with a glide ---------
    fft_grid = (4, 4, 4)
    n_rtot = 64
    kgrid = (2, 2, 1)
    nk = 4
    swap = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    ops = np.stack([np.eye(3, dtype=np.int64), swap])
    tnp = np.asarray([[0.0, 0.0, 0.0], [np.pi, np.pi, 0.0]])   # tau = (½,½,0)
    n_tran = 2
    kints = np.asarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    kfrac = kints / np.asarray(kgrid, dtype=np.float64)
    # Raw rows k0, k1, k3.  k2 = swap·k1 is the one genuine child; k3 is
    # its own image through time reversal composed with the identity.
    irr = np.asarray([0, 1, 1, 2], dtype=np.int32)
    sym = np.asarray([0, 0, 1, n_tran + 0], dtype=np.int32)
    parent_k = kfrac[[0, 1, 3]]
    n_parent = 3
    if ns in (2, 4):
        theta = 0.7
        U1 = np.asarray([[np.cos(theta), -1j * np.sin(theta)],
                         [-1j * np.sin(theta), np.cos(theta)]])
        U_spatial = np.stack([np.eye(2, dtype=np.complex128), U1])
    else:
        U_spatial = np.ones((2, 1, 1), dtype=np.complex128)

    def spinor_action(rows, *, nspinor):
        return spinor_rotation_for_sym_row(
            U_spatial, np.asarray(rows), n_tran, nspinor=nspinor, R_cart=ops)

    sym_fx = SimpleNamespace(
        sym_matrices=ops, translations=tnp,
        irr_idx_k=irr, sym_idx_k=sym, spinor_action=spinor_action,
        unfolded_kpts=kfrac, kirr_fullids=np.asarray([0, 1, 3]))

    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in fft_grid), indexing="ij")
    grid_pts = np.stack(
        [ix.reshape(-1), iy.reshape(-1), iz.reshape(-1)], axis=1).astype(np.int32)
    perm_g, L_g = centroid_source_map_and_wrap(
        grid_pts, ops, tnp, fft_grid, extend_trs=True)   # (4, 64), (4, 64, 3)
    x_frac = grid_pts / np.asarray(fft_grid, dtype=np.float64)

    # ---- parents on the grid, children by the typed action --------------
    nb_full = 8
    l_range, r_range = (0, 5), (2, 8)
    psi_parent = _crand(rng, n_parent, nb_full, ns, n_rtot)
    psi_full = np.empty((nk, nb_full, ns, n_rtot), dtype=np.complex128)
    for k in range(nk):
        p, s = int(irr[k]), int(sym[k])
        U_eff = spinor_action(np.asarray([s]), nspinor=ns)[0]
        val = psi_parent[p][:, :, perm_g[s]] * np.exp(
            2j * np.pi * (L_g[s].astype(np.float64) @ parent_k[p]))[None, None, :]
        if s >= n_tran:
            val = np.conj(val)
        psi_full[k] = np.einsum("ac,ncr->nar", U_eff, val, optimize=True)

    # ---- an orbit-closed centroid set of eight grid points --------------
    cent_flat = []
    for seed in (0, 5, 22, 27, 41, 50, 60, 63, 9, 14):
        orbit = sorted({int(perm_g[s, seed]) for s in range(2 * n_tran)})
        if len(cent_flat) + len(orbit) > 8:
            continue
        if any(c in cent_flat for c in orbit):
            continue
        cent_flat.extend(orbit)
        if len(cent_flat) == 8:
            break
    assert len(cent_flat) == 8, cent_flat
    cent_flat = np.asarray(sorted(cent_flat))
    cent_idx = grid_pts[cent_flat]

    plan = build_centroid_k_unfold_plan(
        sym_fx, cent_idx, fft_grid, mesh, nspinor=ns,
        parent_k_frac=parent_k)
    tiles = plan.real_grid_tiles(target_width=32)
    assert tiles.width <= 32 and tiles.n_tiles >= 2, (tiles.width, tiles.n_tiles)

    # ---- centroid faces, both in the run's PACKED order ------------------
    # (the loader samples the packed table; the host packer stands in here)
    pack = plan.layout.axis.pack_host
    mun_sh = NamedSharding(mesh, P(None, None, "x", "y"))
    nmu_sh = NamedSharding(mesh, P(None, "x", None, "y"))
    psi_mun_full = jax.device_put(jnp.asarray(pack(
        psi_full[:, :, :, cent_flat].transpose(0, 2, 3, 1), axis=2)), mun_sh)
    psi_nmu_full = jax.device_put(jnp.asarray(pack(
        psi_full[:, :, :, cent_flat], axis=3)), nmu_sh)
    psi_nmu_pk = jax.device_put(jnp.asarray(pack(
        psi_parent[:, :, :, cent_flat], axis=3)), nmu_sh)
    psi_mun_pk = jax.device_put(jnp.asarray(pack(
        psi_parent[:, :, :, cent_flat].transpose(0, 2, 3, 1), axis=2)), mun_sh)
    jax.block_until_ready((psi_nmu_pk, psi_mun_pk))

    # ---- ψ(G) mock stores: the identity FFT box ----------------------------
    def _coeffs(psi_r, kvecs):
        u = psi_r * np.exp(-2j * np.pi * (x_frac @ kvecs.T).T)[:, None, None, :]
        box = u.reshape(*u.shape[:3], *fft_grid)
        return np.fft.fftn(box, axes=(-3, -2, -1), norm="ortho").reshape(
            *u.shape[:3], n_rtot)

    band_chunk_ranges = ((0, nb_full),)

    class _MeshStore:
        def __init__(self, psi_G, kvecs):
            nrows = psi_G.shape[0]
            self._psi_G = psi_G
            self._py = PY
            self.band_chunk_ranges = band_chunk_ranges
            self._bpd_max = nb_full // (PX * PY)
            self.meta = SimpleNamespace(
                fft_grid=fft_grid, nk_tot=nrows, nspinor=ns)
            g_index = np.broadcast_to(
                np.arange(n_rtot, dtype=np.int32).reshape(fft_grid),
                (nrows,) + fft_grid)
            self._g = jax.device_put(
                jnp.asarray(g_index), NamedSharding(mesh, P(None, None, None, None)))
            self._k = jax.device_put(
                jnp.asarray(kvecs), NamedSharding(mesh, P(None, None)))
            self._per_rank_shape = (nrows, self._bpd_max, ns, n_rtot)

        def read_local_band_chunk(self, x_idx, y_idx, bc_idx):
            r = int(x_idx) * self._py + int(y_idx)
            bpd = self._bpd_max
            out = np.zeros(self._per_rank_shape, dtype=np.complex128)
            out[:, :bpd] = self._psi_G[:, r * bpd:(r + 1) * bpd]
            return out

        def _slice_local_tile_bc(self, x_idx, y_idx, bc_idx):
            return self.read_local_band_chunk(x_idx, y_idx, bc_idx)

        @property
        def local_band_chunk_shape(self):
            return self._per_rank_shape

        @property
        def band_chunk_carrier(self):
            return self._bpd_max * PX * PY

        @property
        def g_index(self):
            return self._g

        @property
        def kvecs_frac(self):
            return self._k

    full_store = _MeshStore(_coeffs(psi_full, kfrac), kfrac)
    parent_store = _MeshStore(_coeffs(psi_parent, parent_k), parent_k)

    idx = np.arange(nb_full)
    w_l = jnp.asarray(np.where((idx >= l_range[0]) & (idx < l_range[1]), 1.0, 0.0))
    w_r = jnp.asarray(np.where((idx >= r_range[0]) & (idx < r_range[1]), 1.0, 0.0))

    # ---- reference: the established full-k face kernel on the whole grid --
    Z_face = np.asarray(jax.block_until_ready(z_q_from_psi_sm(
        psi_G_store=full_store, psi_r_cache=None,
        band_chunk_ranges=band_chunk_ranges,
        r_start_dyn=0, r_chunk_size=n_rtot,
        kgrid=kgrid, mesh_xy=mesh, layout="face",
        psi_mun=psi_mun_full, weight_l=w_l, weight_r=w_r,
        cache_face_y_blocks=True, gamma_L=gamma, gamma_R=gamma)))               # (nk, mu_pk, 64)
    # both routes run in the packed order; compare in canonical rows
    Z_face = plan.layout.axis.unpack_host(Z_face, axis=1)   # (nk, 8, 64)

    # ---- parent route, both ψ(r) sources, every tile -------------------
    parent_cache = jax.block_until_ready(
        build_psi_r_cache_sm(parent_store, mesh_xy=mesh))
    scale = float(np.max(np.abs(Z_face)))
    max_rel = 0.0
    max_pad = 0.0
    for t in range(tiles.n_tiles):
        local_perm, wraps = tiles.source_tables(t)
        r_index = tiles.r_index[t]
        active = r_index >= 0
        for cache in (None, parent_cache):
            Z_pk = z_q_from_psi_sm(
                psi_G_store=parent_store, psi_r_cache=cache,
                band_chunk_ranges=band_chunk_ranges,
                r_start_dyn=0, r_chunk_size=tiles.width,
                kgrid=kgrid, mesh_xy=mesh, layout="face",
                psi_mun=psi_mun_pk, weight_l=w_l, weight_r=w_r,
                k_unfold_plan=plan, gamma_L=vertex, gamma_R=vertex,
                tile_r_index=jnp.asarray(r_index, dtype=jnp.int32),
                tile_local_perm=jnp.asarray(local_perm),
                tile_wraps=jnp.asarray(wraps))
            if vertex and cache is not None:
                from isdf.core import _z_q_face_coupled_mu123
                coupled = _z_q_face_coupled_mu123(
                    psi_mun_pk, parent_store, cache, w_l, w_r,
                    band_chunk_ranges=band_chunk_ranges,
                    r_start_dyn=0, r_chunk_size=tiles.width,
                    kgrid=kgrid, mesh_xy=mesh, k_unfold_plan=plan,
                    tile_r_index=jnp.asarray(r_index),
                    tile_local_perm=jnp.asarray(local_perm), tile_wraps=jnp.asarray(wraps))
                np.testing.assert_allclose(np.asarray(coupled[vertex - 1]),
                                           np.asarray(Z_pk), rtol=1e-13, atol=1e-11)
            Z_can = plan.layout.axis.unpack_host(
                np.asarray(jax.block_until_ready(Z_pk)), axis=1)  # (nk, 8, width)
            assert Z_can.shape == (nk, 8, tiles.width), Z_can.shape
            diff = Z_can[:, :, active] - Z_face[:, :, r_index[active]]
            max_rel = max(max_rel, float(np.max(np.abs(diff))) / scale)
            if np.any(~active):
                max_pad = max(max_pad, float(np.max(np.abs(Z_can[:, :, ~active]))))

    # ---- CCT: the same claim for the square projector -----------------
    def _local_gemm(a, b):
        return jnp.einsum("qmk,qkn->qmn", a, b, optimize=True)

    C_face = np.asarray(jax.block_until_ready(c_q_from_psi_sm(
        kgrid=kgrid, mesh_xy=mesh, layout="face",
        psi_mun=psi_mun_full, psi_nmu=psi_nmu_full,
        weight_l=w_l, weight_r=w_r, gemm=_local_gemm,
        gamma_L=gamma, gamma_R=gamma)))
    C_parent = np.asarray(jax.block_until_ready(c_q_from_psi_sm(
        kgrid=kgrid, mesh_xy=mesh, layout="face",
        psi_mun=psi_mun_pk, psi_nmu=psi_nmu_pk,
        weight_l=w_l, weight_r=w_r, gemm=_local_gemm,
        k_unfold_plan=plan, gamma_L=vertex, gamma_R=vertex)))
    c_rel = float(np.max(np.abs(C_parent - C_face))) / float(np.max(np.abs(C_face)))

    current_c_rel = None
    if ns == 4:
        from common.gamma_matrices import gamma_perm_phase
        C_current = np.asarray(jax.block_until_ready(c_q_from_psi_sm(
            kgrid=kgrid, mesh_xy=mesh, layout="face",
            psi_mun=psi_mun_full, psi_nmu=psi_nmu_full,
            gamma_L=gamma_perm_phase(1), gamma_R=gamma_perm_phase(1),
            weight_l=w_l, weight_r=w_r, gemm=_local_gemm)))
        C_no_vertex = np.asarray(jax.block_until_ready(c_q_from_psi_sm(
            kgrid=kgrid, mesh_xy=mesh, layout="face",
            psi_mun=psi_mun_pk, psi_nmu=psi_nmu_pk,
            weight_l=w_l, weight_r=w_r, gemm=_local_gemm, k_unfold_plan=plan)))
        current_c_rel = float(np.max(np.abs(C_no_vertex - C_current))) / float(np.max(np.abs(C_current)))

    # The children are NOT trivially the parents: a route that forgot the
    # symmetry action entirely must be visibly wrong on this fixture.
    naive = float(np.max(np.abs(psi_full[[0, 1, 2, 3]] - psi_parent[irr])))
    print(json.dumps({
        "max_rel": max_rel, "max_pad": max_pad, "c_rel": c_rel,
        "current_c_without_vertex_relative_difference": current_c_rel,
        "n_tiles": int(tiles.n_tiles), "width": int(tiles.width),
        "naive_child_parent_gap": naive / float(np.max(np.abs(psi_full))),
    }))
    return 0


def _run_worker(case_name: str, ndev: int = 4, timeout: int = 600):
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
        [sys.executable, os.path.abspath(__file__), "worker", case_name],
        env=env, capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, (
        f"worker {case_name} failed rc={res.returncode}\nSTDOUT:\n"
        f"{res.stdout}\nSTDERR:\n{res.stderr}")
    lines = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, f"no JSON from worker.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return json.loads(lines[-1])


@pytest.mark.parametrize("name", [c[0] for c in _CASES])
def test_parent_zq_and_cq_match_the_full_k_face_route(name):
    out = _run_worker(name)
    if "skip" in out:
        pytest.skip(f"parent ζ-fit parity: {out['skip']}")
    assert out["naive_child_parent_gap"] > 0.1, out
    assert out["max_rel"] < _TOL, f"Z_q parent vs face: {out}"
    assert out["max_pad"] == 0.0, f"tile pad slots must be exactly zero: {out}"
    assert out["c_rel"] < _TOL, f"C_q parent vs face: {out}"


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "worker":
        sys.exit(_worker(sys.argv[2]))
    raise SystemExit("usage: python test_isdf_zq_parent_parity.py worker <case>")
