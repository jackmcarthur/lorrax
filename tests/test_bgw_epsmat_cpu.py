"""BerkeleyGW ε⁻¹ export and projection (``gw.epsmat_export``) on a synthetic slab, CPU 2x2.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``
(CPU clause of the four-GPU rule: device-count logic only).  A toy ISDF slab
(non-orthogonal ζ̃_μ(q+G), the Ismail-Beigi kernel, χ₀ = −CCᴴ) is screened in
μ space; the exported block is checked against an INDEPENDENT dense G-space RPA
inverse ε⁻¹ = (1 − v χ₀(G,G'))⁻¹ that shares no code with the export.  Each
check carries a red twin: the case in which the property is false.
"""
import functools

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
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


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


def sharded_inputs(mesh, Z, V, W, rcond=1e-10):
    from gw import epsmat_export as ee
    ngk = Z.shape[0]
    zt = np.zeros((1, N_MU_PAD, ngk + 3), np.complex128)
    zt[0, :N_MU, :ngk] = Z.T
    zeta = jax.device_put(zt, NamedSharding(mesh, P(None, ("x", "y"), None)))
    B, _ = ee.coulomb_whitener(jnp.asarray(0.5 * (V + V.conj().T)), rcond=rcond)
    K = ee.whitened_correlation(jnp.asarray(W - V), B)
    pad = N_MU_PAD - N_MU
    return zeta, jnp.pad(B, ((0, pad), (0, pad))), jnp.pad(K, ((0, pad), (0, pad))), B, K


def export(mesh, q, ecuts, *, is_q0=False, head=None, rcond=1e-10):
    from gw import epsmat_export as ee
    sphere, Z, v, chi0, V, W = toy(tuple(q))
    rho, qg = bgw_list(np.asarray(q), ecuts, is_q0)
    miller = rho[qg.order[: qg.nmtx]]
    zeta, Bp, Kp, B, K = sharded_inputs(mesh, Z, V, W, rcond)
    slots = ee.zeta_slots(np.vstack([sphere, np.zeros((3, 3), int)]), sphere.shape[0], miller,
                          fill=zeta.shape[2])
    v_eps = v_slab(np.asarray(q), miller)
    blk, diag = ee.epsinv_block(zeta, slots, Bp, Kp, v_eps, mesh=mesh, n_valid=qg.nmtx,
                                q0_head_value=head)
    return dict(sphere=sphere, Z=Z, v=v, chi0=chi0, V=V, W=W, miller=miller, qg=qg,
                zeta=zeta, Bp=Bp, Kp=Kp, B=B, K=K, slots=slots, v_eps=v_eps,
                blk=np.asarray(blk), diag=np.asarray(diag), n=qg.nmtx)


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


def test_projection_round_trip_and_its_gate():
    from gw import epsmat_export as ee
    mesh = _mesh()
    q = (0.0, 1 / 6, 0.0)
    r = export(mesh, q, ecuts=5.0)                             # eps list = the ζ sphere
    n = r["n"]
    blk6 = jax.device_put(r["blk"].reshape(1, 1, 1, *r["blk"].shape),
                          NamedSharding(mesh, P(None, None, None, "x", "y", None)))
    proj = np.asarray(ee.project_block(blk6, r["zeta"], r["slots"], r["v_eps"], mesh=mesh,
                                       n_valid=n, drop_head=False))[:N_MU, :N_MU]
    Wc = r["W"] - r["V"]
    assert np.linalg.norm(proj - Wc) / np.linalg.norm(Wc) < 1e-10
    V_e = np.asarray(ee.coulomb_gram(r["zeta"], r["slots"], r["v_eps"], mesh=mesh))[:N_MU, :N_MU]
    err, rt = ee._check_projection(jnp.asarray(proj), jnp.asarray(V_e), r["Bp"], r["Kp"],
                                   jnp.asarray(Wc), N_MU, 0, "test")
    assert err < 1e-10 and rt < 1e-10
    # Red twin: the naive block (ε⁻¹ − 1 = Z (W − V) Zᴴ / v) projects far from W − V,
    # and the projection gate refuses it.
    rows = rows_in_sphere(r["sphere"], r["miller"])
    Zr, v = r["Z"][rows], r["v"][rows]
    naive = np.eye(n) + (Zr @ Wc @ Zr.conj().T) / np.where(v > 0, v, 1)[None, :]
    nb = np.zeros_like(r["blk"])
    nb[:n, :n, 0], nb[:n, :n, 1] = naive.T.real, naive.T.imag
    nb6 = jax.device_put(nb.reshape(blk6.shape), blk6.sharding)
    bad = ee.project_block(nb6, r["zeta"], r["slots"], r["v_eps"], mesh=mesh, n_valid=n,
                           drop_head=False)[:N_MU, :N_MU]
    assert np.linalg.norm(np.asarray(bad) - Wc) / np.linalg.norm(Wc) > 0.5
    with pytest.raises(ValueError, match="GATE epsmat_block_projection"):
        ee._check_projection(bad, jnp.asarray(V_e), r["Bp"], r["Kp"], jnp.asarray(Wc),
                             N_MU, 0, "red twin")


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
    mesh = _mesh()
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
    offset, shape = be.matrix_block_window(0, 0, r["blk"].shape[0])
    blk6 = jax.device_put(r["blk"].reshape(shape),
                          NamedSharding(mesh, P(None, None, None, "x", "y", None)))
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
    zeta, *_ = sharded_inputs(mesh, Z, V, W)
    ngk = Z.shape[0]
    Vrec = np.asarray(ee.coulomb_gram(zeta, np.arange(ngk), v, mesh=mesh))[:N_MU, :N_MU]
    assert np.linalg.norm(Vrec - V) / np.linalg.norm(V) < 1e-12
    # Red twin: a ζ normalized differently by 0.1 % is caught at the gate tolerance.
    Vbad = np.asarray(ee.coulomb_gram(jax.device_put(1.001 * np.asarray(zeta), zeta.sharding),
                                      np.arange(ngk), v, mesh=mesh))[:N_MU, :N_MU]
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
    good = "[epsmat]\nsource_restart = r\nparent_input_file = d.in\noutput_dir = o\nepsilon_cutoff = 10\n"
    (tmp_path / "a.in").write_text(good)
    cfg = EpsmatConfig.from_input_file(str(tmp_path / "a.in"))
    assert cfg.mode == "export" and cfg.files == ("eps0mat", "epsmat") and cfg.rcond == 1e-8
    for bad, gate in ((good + "epsilon_cutof = 3\n", "epsmat_unknown_key"),
                      (good.replace("epsilon_cutoff = 10\n", ""), "epsmat_epsilon_cutoff"),
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



def test_host_fetched_outputs_are_replicated():
    """Every array the drivers fetch to the host leaves its program replicated.

    On P processes a ``P(('x','y'))`` array spans other processes' devices and
    ``jax.device_get`` refuses it (the CrI3 P=4 export died there while every
    single-process check passed).  Red twin: the block, which stays tiled, is
    not replicated, and neither is a ``P(('x','y'))`` copy of the diagonal.
    """
    from gw import epsmat_export as ee
    mesh = _mesh()
    r = export(mesh, (1 / 6, 1 / 3, 0.0), ecuts=2.5)
    blk, diag = ee.epsinv_block(r["zeta"], r["slots"], r["Bp"], r["Kp"], r["v_eps"],
                                mesh=mesh, n_valid=r["n"])
    blk6 = jax.device_put(np.asarray(blk).reshape((1, 1, 1) + tuple(blk.shape)),
                          NamedSharding(mesh, P(None, None, None, "x", "y", None)))
    proj = ee.project_block(blk6, r["zeta"], r["slots"], r["v_eps"], mesh=mesh,
                            n_valid=r["n"], drop_head=False)
    gram = ee.coulomb_gram(r["zeta"], r["slots"], r["v_eps"], mesh=mesh)
    assert diag.is_fully_replicated and proj.is_fully_replicated and gram.is_fully_replicated
    assert not blk.is_fully_replicated
    assert not jax.device_put(diag, NamedSharding(mesh, P(("x", "y")))).is_fully_replicated
