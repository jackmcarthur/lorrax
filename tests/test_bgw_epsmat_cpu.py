"""BerkeleyGW ε⁻¹ export and projection (``gw.epsmat_export``) on a synthetic slab, CPU 2x2.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``
(CPU clause of the four-GPU rule: device-count logic only).  A toy ISDF slab
(non-orthogonal ζ̃_μ(q+G), the Ismail-Beigi kernel, χ₀ = −CCᴴ) is screened in
μ space; the exported block is checked against an INDEPENDENT dense G-space RPA
inverse ε⁻¹ = (1 − v χ₀(G,G'))⁻¹ that shares no code with the export.  Each
check carries a red twin: the case in which the property is false.
"""
import functools
import os

import h5py
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

A_LAT, L_Z, SIG, N_MU, N_MU_PAD = 6.0, 18.0, 0.7, 10, 12
BVEC = 2 * np.pi * np.array([[1 / A_LAT, 1 / (np.sqrt(3) * A_LAT), 0.0],
                             [0.0, 2 / (np.sqrt(3) * A_LAT), 0.0],
                             [0.0, 0.0, 1 / L_Z]])
OMEGA = A_LAT ** 2 * np.sqrt(3) / 2 * L_Z
BDOT = BVEC @ BVEC.T
RNG = np.random.default_rng(11)
R_MU = RNG.random((N_MU, 3)) @ (2 * np.pi * np.linalg.inv(BVEC).T)
C_RESP = RNG.standard_normal((N_MU, 5)) + 1j * RNG.standard_normal((N_MU, 5))
MILLER = np.array([[h, k, l] for h in range(-4, 5) for k in range(-4, 5)
                   for l in range(-12, 13)])


def _mesh():
    # The CPU clause: an emulated 2x2 HOST mesh, also inside a GPU process (`lx test`), where
    # jax.devices() are CUDA devices and SlabIO's serial (emulation) tier refuses them.
    cpu = jax.devices("cpu")
    if len(cpu) < 4:
        pytest.skip("needs 4 host CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(cpu[:4]).reshape(2, 2), ("x", "y"))


def _cpu_mesh():
    """``_mesh()`` for a test that writes through SlabIO's serial tier: CPU only.

    The serial tier (``file_io._slab_io_serial``) refuses a non-``cpu`` mesh
    platform by design, so on a GPU node (``lx test``: four CUDA devices in
    one process) such a test skips by name instead of failing.
    """
    mesh = _mesh()
    platform = mesh.devices.flat[0].platform
    if platform != "cpu":
        pytest.skip(f"CPU only: SlabIO's serial tier refuses a {platform!r} mesh "
                    "platform (file_io._slab_io_serial); run with JAX_PLATFORMS=cpu "
                    "XLA_FLAGS=--xla_force_host_platform_device_count=4")
    return mesh


def v_slab(q, G):
    """Ismail-Beigi slab Coulomb over Ω (Ry), zero at q+G = 0; written independently."""
    K = (q[None, :] + G) @ BVEC
    k2 = np.sum(K * K, 1)
    zc = np.pi / np.linalg.norm(BVEC[2])
    f = 1 - np.exp(-zc * np.hypot(K[:, 0], K[:, 1])) * np.cos(K[:, 2] * zc)
    return np.where(k2 < 1e-12, 0.0, 8 * np.pi / np.where(k2 < 1e-12, 1, k2) * f) / OMEGA


@functools.lru_cache(maxsize=None)
def toy(qkey, e_zeta=5.0):
    """Sphere, ζ̃ (G, μ), v, χ₀, V, W at one q (prototype normalisation)."""
    q = np.asarray(qkey)
    K = (q[None, :] + MILLER) @ BVEC
    sphere = MILLER[np.sum(K * K, 1) <= e_zeta]
    Ks = (q[None, :] + sphere) @ BVEC
    Z = np.exp(-0.5 * SIG ** 2 * np.sum(Ks * Ks, 1))[:, None] * np.exp(-1j * Ks @ R_MU.T)
    Z = (Z + 0.05 * RNG.standard_normal(Z.shape)) / np.sqrt(OMEGA)
    v = v_slab(q, sphere)
    V = (Z.conj().T * v) @ Z
    chi0 = -(C_RESP @ C_RESP.conj().T)
    chi0 *= 0.6 / np.linalg.norm(np.sqrt(v)[:, None] * (Z @ chi0 @ Z.conj().T) * np.sqrt(v)[None, :], 2)
    W = np.linalg.solve(np.eye(N_MU) - V @ chi0, V)
    return sphere, Z, v, chi0, V, W


def dense_epsinv(Z, v, chi0):
    """The third evaluation: G-space RPA ε⁻¹ = (1 − diag(v) Z χ₀ Zᴴ)⁻¹ on the whole sphere."""
    return np.linalg.inv(np.eye(Z.shape[0]) - v[:, None] * (Z @ chi0 @ Z.conj().T))


def bgw_list(q, ecuts, is_q0=False):
    from file_io import bgw_epsmat as be
    rho = MILLER[np.sum((MILLER @ BVEC) ** 2, 1) <= 30.0]
    return rho, be.q_gspace(rho, BDOT, q, ecuts, is_q0=is_q0)


FACE_SPEC = P(None, "x", "y")


def faces(mesh, Z, V, W, rcond=1e-10):
    """ζ̃ μ-sharded and the toy's μ-space faces (V, W − V, B, K, χ) through the production path."""
    from gw import epsmat_export as ee
    ngk = Z.shape[0]
    zt = np.zeros((1, N_MU_PAD, ngk + 3), np.complex128)
    zt[0, :N_MU, :ngk] = Z.T
    zeta = jax.device_put(zt, NamedSharding(mesh, P(None, ("x", "y"), None)))
    face = NamedSharding(mesh, FACE_SPEC)
    pad = ((0, N_MU_PAD - N_MU), (0, N_MU_PAD - N_MU))
    Vf = jax.device_put(np.pad(V, pad)[None], face)
    Wcf = jax.device_put(np.pad(W - V, pad)[None], face)
    plan = ee.eigh_plan(mesh, N_MU_PAD)
    B, _ = ee.coulomb_whitener(Vf, rcond=rcond, plan=plan)
    K = ee.whitened_correlation(Wcf, B, mesh=mesh)
    chi = ee.response(B, K, mesh=mesh)
    return dict(zeta=zeta, Vf=Vf, Wcf=Wcf, B=B, K=K, chi=chi, plan=plan)


def export(mesh, q, ecuts, *, is_q0=False, head=None, rcond=1e-10):
    from gw import epsmat_export as ee
    sphere, Z, v, chi0, V, W = toy(tuple(q))
    rho, qg = bgw_list(np.asarray(q), ecuts, is_q0)
    miller = rho[qg.order[: qg.nmtx]]
    f = faces(mesh, Z, V, W, rcond)
    slots = ee.zeta_slots(np.vstack([sphere, np.zeros((3, 3), int)]), sphere.shape[0], miller,
                          fill=f["zeta"].shape[2])
    v_eps = v_slab(np.asarray(q), miller)
    n_pad = ee._pad_to(qg.nmtx, 4)
    Zf = ee.zeta_face(f["zeta"], slots, mesh=mesh, n_pad=n_pad)
    blk6, diag = ee.epsinv_block(Zf, f["chi"], v_eps, mesh=mesh, n_valid=qg.nmtx,
                                 q0_head_value=head)
    return dict(sphere=sphere, Z=Z, v=v, chi0=chi0, V=V, W=W, miller=miller, qg=qg, **f,
                Zf=Zf, slots=slots, v_eps=v_eps, n_pad=n_pad, blk6=blk6,
                blk=np.asarray(blk6)[0, 0, 0], diag=np.asarray(diag), n=qg.nmtx)


def rows_in_sphere(sphere, miller):
    idx = {tuple(g): i for i, g in enumerate(sphere)}
    return np.array([idx[tuple(g)] for g in miller])


# ---------------------------------------------------------------------------

def test_gcutoff_is_bgw_bisection_with_its_two_edges():
    from file_io.bgw_epsmat import gcutoff
    rng = np.random.default_rng(3)
    for _ in range(500):
        e = np.sort(rng.random(rng.integers(2, 40)) * 10)
        ec = rng.uniform(e[0], e[-1] - 1e-9)
        assert gcutoff(e, ec) == int(np.sum(e <= ec))
    e = np.array([0.0, 1.0, 2.0, 3.0])
    assert gcutoff(e, 3.0) == 3            # BerkeleyGW: whole list only when e[-1] < ecuts
    assert gcutoff(e, 3.5) == 4
    assert gcutoff(e + 1.0, 0.5) == 1      # never below 1
    # Red twin: a plain count differs at the edge this port keeps.
    assert int(np.sum(e <= 3.0)) != gcutoff(e, 3.0)


def test_gvector_order_is_the_sortrx_comparator():
    from file_io.bgw_epsmat import TOL_ZERO, gvector_order, kinetic_energies
    q = np.array([0.0, 1 / 3, 0.0])
    rho, _ = bgw_list(q, 4.0)
    ek = kinetic_energies(rho, BDOT, q)
    eff = rho.max(0) - rho.min(0) + 1
    gk = rho[:, 2] + eff[2] * (rho[:, 1] + eff[1] * rho[:, 0])

    def is_greater(i, j):
        return ek[i] - ek[j] > TOL_ZERO or (ek[i] - ek[j] > -TOL_ZERO and gk[i] > gk[j])

    ref = sorted(range(len(ek)), key=functools.cmp_to_key(
        lambda i, j: 1 if is_greater(i, j) else (-1 if is_greater(j, i) else 0)))
    got = gvector_order(rho, ek)
    assert np.array_equal(got, ref)
    # Red twin: energy alone (no GK tie-break) is a different order on this list.
    assert not np.array_equal(np.argsort(ek, kind="stable"), ref)


def test_block_equals_dense_rpa_inverse_and_red_twins():
    mesh = _mesh()
    q = (1 / 6, 1 / 3, 0.0)
    r = export(mesh, q, ecuts=2.5)
    n = r["n"]
    E = r["blk"][:n, :n, 0] + 1j * r["blk"][:n, :n, 1]        # E[a, b] = ε⁻¹(G_b, G_a)
    rows = rows_in_sphere(r["sphere"], r["miller"])
    ref = dense_epsinv(r["Z"], r["v"], r["chi0"])[np.ix_(rows, rows)]
    err = np.linalg.norm(E.T - ref) / np.linalg.norm(ref - np.eye(n))
    assert err < 1e-10, err
    assert np.allclose(r["diag"][:n], np.diag(ref), rtol=0, atol=1e-10)
    # Red twin 1: the untransposed block is not ε⁻¹ (ε⁻¹ is not symmetric).
    assert np.linalg.norm(E - ref) / np.linalg.norm(ref - np.eye(n)) > 1e-2
    # Red twin 2: the naive W(G,G') = Z W Zᴴ, read as ε⁻¹ = W / v(G'), is not ε⁻¹.
    Zr, W, v = r["Z"][rows], r["W"], r["v"][rows]
    naive = (Zr @ W @ Zr.conj().T) / v[None, :]
    assert np.linalg.norm(naive - ref) / np.linalg.norm(ref - np.eye(n)) > 1.0


def test_epsmat_ecut_is_the_leading_sub_block_not_a_reinversion():
    """``epsmat_ecut`` keeps the leading gcutoff G of the full order: the written block is the
    sub-block of the converged ε⁻¹ (docs/theory/bgw_epsmat_export.md#ecut).  Control
    variant: the inverse of the dielectric matrix restricted to the cut G set, which is what a
    fresh epsilon run at the smaller cutoff would write."""
    from gw import epsmat_export as ee
    from file_io import bgw_epsmat as be
    mesh = _mesh()
    q = (1 / 6, 1 / 3, 0.0)
    assert ee.resolve_ecut(0.0, 5.0, source="t")[0] == 5.0          # 0 = the full sphere
    e, _ = ee.resolve_ecut(2.5, 5.0, source="t")
    with pytest.raises(ValueError, match="GATE epsmat_ecut_beyond_sphere"):
        ee.resolve_ecut(5.5, 5.0, source="t")
    full = export(mesh, q, ecuts=5.0)
    # The prefix of the full order is BerkeleyGW's own list at the smaller cutoff.
    qg = full["qg"]
    pre = qg._replace(nmtx=be.gcutoff(qg.ekin[qg.order], e))
    _, direct = bgw_list(np.asarray(q), e)
    assert pre.nmtx == direct.nmtx and np.array_equal(pre.order[:pre.nmtx], direct.order[:direct.nmtx])
    cut = export(mesh, q, ecuts=e)
    n, N = cut["n"], full["n"]
    assert 0 < n < N
    Ec = cut["blk"][:n, :n, 0] + 1j * cut["blk"][:n, :n, 1]
    Ef = full["blk"][:N, :N, 0] + 1j * full["blk"][:N, :N, 1]
    one = np.linalg.norm(Ef[:n, :n] - np.eye(n))
    assert np.linalg.norm(Ec - Ef[:n, :n]) / one < 1e-12
    assert np.allclose(cut["diag"][:n], full["diag"][:n], rtol=0, atol=1e-12)
    rows = rows_in_sphere(cut["sphere"], cut["miller"])
    eps = np.eye(cut["Z"].shape[0]) - cut["v"][:, None] * (cut["Z"] @ cut["chi0"] @ cut["Z"].conj().T)
    assert np.linalg.norm(Ec.T - np.linalg.inv(eps)[np.ix_(rows, rows)]) / one < 1e-10
    reinv = np.linalg.inv(eps[np.ix_(rows, rows)])
    assert np.linalg.norm(Ec.T - reinv) / one > 1e-3, "the cut must not be a re-inversion"


def test_projection_round_trip_and_its_gate():
    from gw import epsmat_export as ee
    mesh = _mesh()
    q = (0.0, 1 / 6, 0.0)
    r = export(mesh, q, ecuts=5.0)                             # eps list = the ζ sphere
    n = r["n"]
    blk6 = r["blk6"]
    proj = ee.project_block(blk6, r["Zf"], r["v_eps"], mesh=mesh, n_valid=n, drop_head=False)
    Wc = r["W"] - r["V"]
    assert np.linalg.norm(np.asarray(proj)[0, :N_MU, :N_MU] - Wc) / np.linalg.norm(Wc) < 1e-10
    vv = np.zeros(r["n_pad"]); vv[:n] = r["v_eps"]
    V_e = ee.coulomb_gram(r["Zf"], vv, mesh=mesh)
    err, rt = ee._check_projection(proj, V_e, r["chi"], r["Wcf"], 0, "test", mesh)
    assert err < 1e-10 and rt < 1e-10
    # Red twin: the naive block (ε⁻¹ − 1 = Z (W − V) Zᴴ / v) projects far from W − V,
    # and the projection gate refuses it.
    rows = rows_in_sphere(r["sphere"], r["miller"])
    Zr, v = r["Z"][rows], r["v"][rows]
    naive = np.eye(n) + (Zr @ Wc @ Zr.conj().T) / np.where(v > 0, v, 1)[None, :]
    nb = np.zeros_like(r["blk"])
    nb[:n, :n, 0], nb[:n, :n, 1] = naive.T.real, naive.T.imag
    nb6 = jax.device_put(nb.reshape(blk6.shape), blk6.sharding)
    bad = ee.project_block(nb6, r["Zf"], r["v_eps"], mesh=mesh, n_valid=n, drop_head=False)
    assert np.linalg.norm(np.asarray(bad)[0, :N_MU, :N_MU] - Wc) / np.linalg.norm(Wc) > 0.5
    with pytest.raises(ValueError, match="GATE epsmat_block_projection"):
        ee._check_projection(bad, V_e, r["chi"], r["Wcf"], 0, "red twin", mesh)


def test_q0_block_head_wings_and_body():
    mesh = _mesh()
    head = 0.8125 + 0.0j
    r = export(mesh, (0.0, 0.0, 0.0), ecuts=2.5, is_q0=True, head=head)
    n = r["n"]
    assert np.all(r["miller"][0] == 0)
    E = (r["blk"][:n, :n, 0] + 1j * r["blk"][:n, :n, 1]).T    # ε⁻¹(G, G')
    assert E[0, 0] == head and np.all(E[0, 1:] == 0) and np.all(E[1:, 0] == 0)
    rows = rows_in_sphere(r["sphere"], r["miller"])
    ref = dense_epsinv(r["Z"], r["v"], r["chi0"])[np.ix_(rows, rows)]   # v(0) = 0: body is D⁻¹
    body = np.linalg.norm(E[1:, 1:] - ref[1:, 1:]) / np.linalg.norm(ref[1:, 1:] - np.eye(n - 1))
    assert body < 1e-10, body
    assert r["diag"][0] == head
    # Red twin: the raw column ε̄⁻¹(G, 0) is a nonzero wing, which the q0 file must not carry.
    assert np.linalg.norm(ref[1:, 0]) > 1e-6


def test_q0_head_is_the_direct_s_tensor_model():
    """ε⁻¹_00(q0) = 1 / (1 − v q0ᵀSq0) with v the truncated 8π/|q0|² in Ry bohr³.

    Pinned to the owner's convention (``gw.head_densify``) through the
    production path ``_q0_head_value``.  Red twin: v/Ω (Ry), the convention
    the first CrI3 export used, gives a head weaker by Ω — 0.9999974 against
    BerkeleyGW's 0.98423 on the 6x6 slab.
    """
    from types import SimpleNamespace
    from vcoul import CoulombGeometry
    from gw import epsmat_export as ee
    S = -np.diag([0.4, 0.4, 0.02]).astype(complex)
    q0 = np.array([0.001, 0.0, 0.0])
    src = SimpleNamespace(
        geometry=CoulombGeometry(bvec=BVEC, cell_volume=OMEGA), head={"S_cart": S}, kgrid=(6, 6, 1),
        config=SimpleNamespace(sys_dim=2, head=SimpleNamespace(bare_coulomb_cutoff=None,
                                                               mc_average_vcoul_body=False)))
    h = ee._q0_head_value(SimpleNamespace(q0_head="s_tensor"), src, q0)
    qc = q0 @ BVEC
    v_bgw = float(v_slab(q0, np.zeros((1, 3)))[0]) * OMEGA      # Ry bohr³
    want = 1 / (1 + v_bgw * 0.4 * np.dot(qc, qc))
    assert abs(h - want) < 1e-12 and 0 < h.real < 1
    per_omega = 1 / (1 + v_bgw / OMEGA * 0.4 * np.dot(qc, qc))
    assert abs(per_omega - h) > 100 * abs(1 - h) * 1e-3 and abs(1 - per_omega) < abs(1 - h) / 100
    assert ee._q0_head_value(SimpleNamespace(q0_head="unscreened"), src, q0) == 1


def test_file_layout_reads_back_as_epsinv(tmp_path):
    import h5py
    from file_io import bgw_epsmat as be
    from file_io.epsreader import EPSReader
    from file_io.slab_io import SlabIO
    mesh = _cpu_mesh()
    # ecuts 2.8: nmtx 43 under a 44-row mesh pad, so the write must clip.
    q, ecuts = (1 / 6, 1 / 3, 0.0), 2.8
    r = export(mesh, q, ecuts=ecuts)
    n = r["n"]
    assert r["blk"].shape[0] > n
    rho, qg = bgw_list(np.asarray(q), ecuts)
    wfn = tmp_path / "WFN.h5"
    with h5py.File(wfn, "w") as f:
        f["mf_header/gspace/components"] = rho.astype(np.int32)
        f["mf_header/gspace/ng"] = np.int32(rho.shape[0])
        f["mf_header/crystal/bdot"] = BDOT
        f["mf_header/crystal/celvol"] = OMEGA
    path = str(tmp_path / "epsmat.h5")
    be.create_epsmat_file(path, wfn_file=str(wfn), qpts=[q], qgrid=(6, 6, 1), nmtx=[n],
                          ecuts=ecuts, nband=8, efermi_ry=0.0, icutv=6)
    vb = v_slab(np.asarray(q), rho[qg.order[:n]]) * OMEGA
    be.write_q_gspace(path, 0, qg, vb)
    offset, _ = be.matrix_block_window(0, 0, r["blk"].shape[0])
    blk6 = r["blk6"]
    with SlabIO(path, mode="a", mesh=mesh) as io:
        be.register_matrix(io, 1, n)
        io.write_slab(be.MATRIX, blk6, offset=offset)
    be.write_q_diagonal(path, 0, r["diag"][:n])
    with EPSReader(path) as eps:
        E = eps.get_eps_matrix(0)
        assert int(eps.nmtx[0]) == n and np.all(eps.qpt_done == 1)
        assert np.array_equal(eps.gind_eps2rho[0][:n], qg.order[:n])
        assert np.allclose(eps.get_eps_diagonal(0), np.diag(E))
        vc = eps.vcoul[0, :n]
    rows = rows_in_sphere(r["sphere"], r["miller"])
    ref = dense_epsinv(r["Z"], r["v"], r["chi0"])[np.ix_(rows, rows)]
    assert np.linalg.norm(E - ref) / np.linalg.norm(ref - np.eye(n)) < 1e-10
    # W = ε⁻¹ v is Hermitian in the reader's orientation; the transposed one is not.
    W = E * vc[None, :]
    assert np.linalg.norm(W - W.conj().T) / np.linalg.norm(W) < 1e-10
    Wt = E.T * vc[None, :]
    assert np.linalg.norm(Wt - Wt.conj().T) / np.linalg.norm(Wt) > 1e-3
    # Red twin: the same padded write on a handle that did not register the
    # skeleton's dataset refuses (reuse-or-refuse, both SlabIO tiers).
    with pytest.raises(ValueError, match="the dataset exists"):
        with SlabIO(path, mode="a", mesh=mesh) as io:
            io.write_slab(be.MATRIX, blk6, offset=offset)
    with h5py.File(path, "r") as f:
        assert f[be.MATRIX].chunks is None and int(f["eps_header/params/icutv"][()]) == 6


def test_v_gram_reproduces_v_and_catches_a_rescaled_zeta():
    from gw import epsmat_export as ee
    mesh = _mesh()
    sphere, Z, v, chi0, V, W = toy((0.0, 1 / 3, 0.0))
    zeta = faces(mesh, Z, V, W)["zeta"]
    ngk = Z.shape[0]
    n_s = ee._pad_to(ngk, 2)
    vs = np.zeros(n_s); vs[:ngk] = v
    Vrec = np.asarray(ee.coulomb_gram(ee.zeta_face(zeta, np.arange(ngk), mesh=mesh, n_pad=n_s),
                                      vs, mesh=mesh))[0, :N_MU, :N_MU]
    assert np.linalg.norm(Vrec - V) / np.linalg.norm(V) < 1e-12
    # Red twin: a ζ normalized differently by 0.1 % is caught at the gate tolerance.
    bad = jax.device_put(1.001 * np.asarray(zeta), zeta.sharding)
    Vbad = np.asarray(ee.coulomb_gram(ee.zeta_face(bad, np.arange(ngk), mesh=mesh, n_pad=n_s),
                                      vs, mesh=mesh))[0, :N_MU, :N_MU]
    assert np.linalg.norm(Vbad - V) / np.linalg.norm(V) > ee.V_RECONSTRUCTION_TOL


def test_umklapp_match_and_slots():
    from gw.epsmat_export import match_q_umklapp, zeta_slots
    qs = np.array([[0, 0, 0], [0, 1 / 3, 0], [1 / 2, 1 / 2, 0]])
    idx, g0 = match_q_umklapp(qs, [[0, -2 / 3, 0], [1 / 2, -1 / 2, 0], [0.25, 0, 0]])
    assert idx.tolist() == [1, 2, -1] and g0[:2].tolist() == [[0, -1, 0], [0, -1, 0]]
    s = zeta_slots(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]]), 3,
                   np.array([[0, 1, 0], [9, 9, 9], [0, 0, 0]]), fill=4)
    assert s.tolist() == [2, 4, 0]            # the pad row (9,9,9) is not physical


def test_deck_refusals(tmp_path):
    from gw.epsmat_config import EpsmatConfig
    good = "[epsmat]\nsource_restart = r\nparent_input_file = d.in\noutput_dir = o\nepsmat_ecut = 10\n"
    (tmp_path / "a.in").write_text(good)
    cfg = EpsmatConfig.from_input_file(str(tmp_path / "a.in"))
    assert cfg.mode == "export" and cfg.files == ("eps0mat", "epsmat") and cfg.rcond == 1e-8
    assert cfg.epsmat_ecut == 10.0
    (tmp_path / "a.in").write_text(good.replace("epsmat_ecut = 10\n", ""))
    assert EpsmatConfig.from_input_file(str(tmp_path / "a.in")).epsmat_ecut == 0.0   # full sphere
    for bad, gate in ((good + "epsmat_ecu = 3\n", "epsmat_unknown_key"),
                      (good.replace("epsmat_ecut = 10", "epsilon_cutoff = 10"), "epsmat_retired_key"),
                      (good.replace("epsmat_ecut = 10", "epsmat_ecut = -1"), "epsmat_ecut_value"),
                      (good + "mode = both\n", "epsmat_mode"),
                      (good.replace("[epsmat]", "[cohsex]"), "epsmat_deck_section")):
        (tmp_path / "b.in").write_text(bad)
        with pytest.raises(ValueError, match=f"GATE {gate}"):
            EpsmatConfig.from_input_file(str(tmp_path / "b.in"))


def test_efermi_is_bgw_midgap(tmp_path):
    """``efermi`` is find_efermi's (max VBM + min CBM)/2 over every k and spin.

    Two k points with the VBM at one and the CBM at the other (an indirect
    gap) and spin-dependent occupation; red twin: the off-by-one band index
    (``el[ifmax]`` as the VBM) gives a different energy.
    """
    import h5py
    from file_io.bgw_epsmat import midgap_ry
    el = np.array([[[-1.0, 0.20, 0.60, 0.90], [-1.1, 0.30, 0.50, 1.20]],
                   [[-0.9, 0.25, 0.70, 1.00], [-1.2, 0.10, 0.55, 0.80]]])
    ifmax = np.array([[2, 2], [2, 2]], dtype=np.int32)
    path = tmp_path / "WFN.h5"
    with h5py.File(path, "w") as f:
        f["mf_header/kpoints/el"] = el
        f["mf_header/kpoints/ifmax"] = ifmax
    vbm, cbm = 0.30, 0.50
    assert midgap_ry(str(path), 4) == pytest.approx(0.5 * (vbm + cbm), abs=1e-15)
    assert midgap_ry(str(path), 3) == pytest.approx(0.40, abs=1e-15)
    off_by_one = 0.5 * (max(el[s, k, ifmax[s, k]] for s in range(2) for k in range(2)) + cbm)
    assert abs(off_by_one - midgap_ry(str(path), 4)) > 0.1



def test_matrices_are_faces_and_only_vectors_replicate():
    """Every N_μ²- and N_G²-class array is a P(None,'x','y') face; vectors are replicated.

    The audit rule: no N_μ² object on fewer than all P devices.  V, W − V, B, K, χ,
    V_ε, the projection and the block stay tiled; the diagonal (read on the host by
    every process) and the eigenvalues are the only replicated arrays.  Red twin:
    the replicated μ-space plan this replaced — B gathered whole — is not a face.
    """
    from gw import epsmat_export as ee
    mesh = _mesh()
    r = export(mesh, (1 / 6, 1 / 3, 0.0), ecuts=2.5)
    proj = ee.project_block(r["blk6"], r["Zf"], r["v_eps"], mesh=mesh, n_valid=r["n"],
                            drop_head=False)
    vv = np.zeros(r["n_pad"]); vv[: r["n"]] = r["v_eps"]
    gram = ee.coulomb_gram(r["Zf"], vv, mesh=mesh)
    for name, a in (("V", r["Vf"]), ("W-V", r["Wcf"]), ("B", r["B"]), ("K", r["K"]),
                    ("chi", r["chi"]), ("Zf", r["Zf"]), ("proj", proj), ("V_eps", gram)):
        assert a.sharding.spec == FACE_SPEC and not a.is_fully_replicated, name
    assert r["blk6"].sharding.spec == P(None, None, None, "x", "y", None)
    assert not r["blk6"].is_fully_replicated
    _, diag = ee.epsinv_block(r["Zf"], r["chi"], r["v_eps"], mesh=mesh, n_valid=r["n"])
    assert diag.is_fully_replicated
    whole = jax.device_put(r["B"], NamedSharding(mesh, P()))
    assert whole.is_fully_replicated and whole.sharding.spec != FACE_SPEC


def test_eigh_plan_requests_the_distributed_solver_on_many_processes(monkeypatch):
    """One process per device -> distrib_la 'distributed'; one process -> native.

    Red twin: distrib_la's 'auto' is native everywhere and would solve a gathered
    matrix on every rank, so the plan must never ask for it.
    """
    import distrib_la as dla
    from gw import epsmat_export as ee
    mesh = _mesh()
    asked = []
    monkeypatch.setattr(dla, "plan", lambda op, m, *, backend, n: asked.append((op, backend, n)))
    ee.eigh_plan(mesh, 12)
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    ee.eigh_plan(mesh, 12)
    assert asked == [("eigh", "native", 12), ("eigh", "distributed", 12)]
    assert all(b != "auto" for _, b, _ in asked)


FIXTURE_BGW = os.path.join(os.path.dirname(__file__), "fixtures", "bgw_epsmat_corner48.h5")


def test_epsreader_orientation_on_a_berkeleygw_written_block():
    """EPSReader returns ε⁻¹(G, G') with rows G, columns G' on BerkeleyGW's own output.

    The fixture is the 48 × 48 low-|q+G| corner of BerkeleyGW 4.0 epsilon.cplx.x's
    epsmat.h5 at q = (0, 1/3, 0) (CrI3 6×6, run 516), stored values verbatim
    (tests/fixtures/make_corner48.py).  χ is Hermitian, so (ε⁻¹ − 1)/v(q+G),
    rows scaled, is Hermitian in the right orientation only: 1e-15 for
    ``get_eps_matrix``; the stored block untransposed gives 0.78 (red twin).
    ``get_eps_diagonal`` reads ``mats/matrix-diagonal`` ``(nq, nmtx, 2)`` and is
    the matrix diagonal bit for bit.
    """
    from file_io.epsreader import EPSReader
    r = EPSReader(FIXTURE_BGW)
    n = int(r.nmtx[0])
    M = r.get_eps_matrix(0)
    v = np.asarray(r.vcoul[0, :n], dtype=np.float64)

    def herm(m):
        x = (m - np.eye(n)) / v[:, None]
        return np.linalg.norm(x - x.conj().T) / np.linalg.norm(x)

    assert M.shape == (n, n) == (48, 48)
    assert herm(M) < 1e-12
    assert herm(M.T) > 0.5                   # the stored block as it lies on disk
    assert np.array_equal(r.get_eps_diagonal(0), np.diag(M))
    with h5py.File(FIXTURE_BGW, "r") as f:
        assert f["mats/matrix-diagonal"].shape == (1, n, 2)


@pytest.mark.parametrize("files,sys_dim,flag,want", [
    (("eps0mat", "epsmat"), 2, "refuse", "q -> 0 limit"),
    (("epsmat",), 3, "refuse", "no eps0mat"),
    (("eps0mat",), 3, "omit_head_wing", "omit_head_wing"),
])
def test_q0_body_policy_writes(files, sys_dim, flag, want):
    from gw.epsmat_export import q0_body_policy
    assert q0_body_policy(files, sys_dim, flag).startswith(want)


def test_q0_body_policy_refuses_a_3d_eps0mat_without_the_flag():
    """A 3-D q → 0 body without its O(1) head–wing term is written only by name."""
    from gw.epsmat_export import q0_body_policy
    with pytest.raises(ValueError, match="GATE epsmat_q0_body_3d"):
        q0_body_policy(("eps0mat", "epsmat"), 3, "refuse")
    with pytest.raises(ValueError, match="GATE epsmat_q0_body_sys_dim"):
        q0_body_policy(("eps0mat",), 1, "omit_head_wing")


def test_q0_body_3d_key_is_validated(tmp_path):
    from gw.epsmat_config import EpsmatConfig
    deck = tmp_path / "e.in"
    base = ("[epsmat]\nsource_restart = /x\nparent_input_file = /x/g.in\n"
            "output_dir = /x/o\nepsmat_ecut = 5.0\n")
    deck.write_text(base)
    assert EpsmatConfig.from_input_file(str(deck)).q0_body_3d == "refuse"
    deck.write_text(base + "q0_body_3d = omit_head_wing\n")
    assert EpsmatConfig.from_input_file(str(deck)).q0_body_3d == "omit_head_wing"
    deck.write_text(base + "q0_body_3d = yes\n")
    with pytest.raises(ValueError, match="GATE epsmat_q0_body_3d_value"):
        EpsmatConfig.from_input_file(str(deck))
