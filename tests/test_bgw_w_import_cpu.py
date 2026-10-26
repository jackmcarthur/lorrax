"""BerkeleyGW W → ISDF import (``gw.epsmat_export`` projection, ``screening_source = bgw_epsmat``), CPU 2x2.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``
(CPU clause of the four-GPU rule: device-count logic only).  The projection of
a symmetry-image q (§4, E7a) is checked against a dense NumPy evaluation of
Z_cᴴ (ε⁻¹_c − 1) v Z_c that shares no code with it, with the phase and the
antiunitary conjugation each removed in a red twin; the frequency rules and
the GW driver's reader refuse what they must.
"""
import numpy as np
import pytest
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

from tests.test_bgw_epsmat_cpu import (N_MU, N_MU_PAD, _cpu_mesh, _mesh, bgw_list, dense_epsinv,
                                 rows_in_sphere, toy, v_slab)

jax.config.update("jax_enable_x64", True)


def _sym():
    from symmetry_maps import SymMaps
    sym = SymMaps.__new__(SymMaps)
    spatial = np.array([np.eye(3, dtype=int), np.diag([-1, -1, 1])])
    sym.sym_matrices = spatial
    sym.sym_mats_k = np.concatenate([spatial, -spatial])
    sym.translations = np.array([[0., 0., 0.], [np.pi, np.pi / 3, 0.]])
    sym.trs_allowed = True
    return sym


def test_symmetry_image_projection_equals_dense_and_red_twins():
    """A stored q reached from a file q by an antiunitary, translated row projects to
    Z_cᴴ(ε⁻¹_c − 1)vZ_c with ε⁻¹_c = φ conj(ε⁻¹_f) φ̄ (E7a); without φ or the
    conjugation it does not."""
    from types import SimpleNamespace
    from gw import epsmat_export as ee
    mesh = _mesh()
    qf = np.array([1 / 6, 1 / 3, 1 / 8])            # the file's (BerkeleyGW's) q
    qs = np.array([1 / 6, 1 / 3, 7 / 8])            # LORRAX's stored q = S qf, S = -diag(-1,-1,1)
    sym = _sym()
    src = SimpleNamespace(sym=sym, q_stored=qs[None])
    heads = {"epsmat": {"qpts": qf[None]}}
    plan = ee.import_q_plan(SimpleNamespace(sym=sym, q_stored=np.vstack([np.zeros(3), qs])), heads)
    assert plan[0].name == "eps0mat" and plan[1].sym_row != 0
    row = plan[1]._replace(iq_store=0)
    rho, qg = bgw_list(qf, 2.5)
    n = qg.nmtx
    g_f = rho[qg.order[:n]]
    g_s, phase, anti = ee.import_gvectors(src, row, qf, g_f)
    S = sym.sym_mats_k[row.sym_row]
    assert anti and np.max(np.abs((qf + g_f) @ S.T - (qs + g_s))) < 1e-12
    # The file block: the dense RPA ε⁻¹ at qf on its G list, file orientation [a, b] = ε⁻¹(G_b, G_a).
    sph_f, Z_f, v_f, chi_f, _, _ = toy(tuple(qf))
    E_f = dense_epsinv(Z_f, v_f, chi_f)[np.ix_(rows_in_sphere(sph_f, g_f), rows_in_sphere(sph_f, g_f))]
    n_pad = ee._pad_to(n, 4)
    blk = np.zeros((1, 1, 1, n_pad, n_pad, 2))
    blk[0, 0, 0, :n, :n, 0], blk[0, 0, 0, :n, :n, 1] = E_f.T.real, E_f.T.imag
    blk = jax.device_put(blk, NamedSharding(mesh, P(None, None, None, "x", "y", None)))
    # The stored q's own ζ̃ (an unrelated toy basis: the identity holds for any ζ̃).
    sph_s, Z_s, _, _, _, _ = toy(tuple(qs))
    slots = rows_in_sphere(sph_s, g_s)
    zt = np.zeros((1, N_MU_PAD, Z_s.shape[0] + 3), np.complex128)
    zt[0, :N_MU, :Z_s.shape[0]] = Z_s.T
    zeta = jax.device_put(zt, NamedSharding(mesh, P(None, ("x", "y"), None)))
    v_s = v_slab(qs, g_s)

    def imported(ph, conj):
        Zf = ee.import_zeta_face(zeta, slots, ph, conj, mesh=mesh, n=n, n_pad=n_pad)
        Wc = ee.project_block(blk, Zf, v_s, mesh=mesh, n_valid=n, drop_head=False)
        return np.asarray(jnp_conj(Wc) if conj else Wc)[0, :N_MU, :N_MU]

    E_c = phase[:, None] * E_f.conj() * phase.conj()[None, :]
    Zc = Z_s[slots]
    truth = Zc.conj().T @ ((E_c - np.eye(n)) * v_s[None, :]) @ Zc
    scale = np.linalg.norm(truth)
    assert np.linalg.norm(imported(phase, True) - truth) / scale < 1e-10
    assert np.linalg.norm(imported(None, True) - truth) / scale > 1e-2     # red twin: no φ
    assert np.linalg.norm(imported(phase, False) - truth) / scale > 1e-2   # red twin: no TR conj


def jnp_conj(a):
    import jax.numpy as jnp
    return jnp.conj(a)


def test_q_coverage_and_frequency_rules():
    """Every stored q needs a file row; only ω = 0 (and the GN iω_p) are read."""
    from types import SimpleNamespace
    from gw import epsmat_export as ee
    sym = _sym()
    sym.trs_allowed = False                          # no antiunitary rows: qs is unreachable
    src = SimpleNamespace(sym=sym, q_stored=np.array([[0., 0., 0.], [1 / 6, 1 / 3, 7 / 8]]))
    with pytest.raises(ValueError, match="GATE epsmat_project_q_coverage"):
        ee.import_q_plan(src, {"epsmat": {"qpts": np.array([[1 / 6, 1 / 3, 1 / 8]])}})
    static = {"freqs/freq_dep": 0, "freqs/freqs": np.zeros((1, 2))}
    gn = {"freqs/freq_dep": 3, "freqs/freqs": np.array([[0., 0.], [0., 27.2114]])}
    rows = ee.projection_frequencies({"eps0mat": gn, "epsmat": gn})
    assert [r.dataset for r in rows] == ["W0_qmunu", "Wp_qmunu"]
    assert abs(rows[1].omega_ry - 2j) < 1e-5
    assert [r.dataset for r in ee.projection_frequencies({"eps0mat": static})] == ["W0_qmunu"]
    with pytest.raises(ValueError, match="GATE epsmat_project_freqs_differ"):
        ee.projection_frequencies({"eps0mat": gn, "epsmat": static})
    with pytest.raises(ValueError, match="GATE epsmat_project_freq_dep"):
        ee.projection_frequencies({"eps0mat": {"freqs/freq_dep": 2, "freqs/freqs": np.zeros((5, 2))}})
    real_axis = {"freqs/freq_dep": 3, "freqs/freqs": np.array([[0., 0.], [27.2114, 0.]])}
    with pytest.raises(ValueError, match="GATE epsmat_project_freqs"):
        ee.projection_frequencies({"eps0mat": real_axis})


def test_reader_serves_projected_rows_and_refuses_others(tmp_path, monkeypatch):
    """``read_projected_screening`` returns the stamped rows and heads; another W0 source,
    another frequency or a missing probe refuses by name."""
    import h5py
    from file_io import restart_bundle as rb
    mesh = _cpu_mesh()
    monkeypatch.setattr(rb, "_require_current", lambda f: None)
    rng = np.random.default_rng(3)
    nq, nmu = 3, 6
    W0 = rng.standard_normal((nq, nmu, nmu)) + 1j * rng.standard_normal((nq, nmu, nmu))
    Wp = rng.standard_normal((nq, nmu, nmu)) + 0j
    path = tmp_path / "isdf_tensors_6.h5"
    with h5py.File(path, "w") as f:
        f["V_qmunu"] = np.zeros((nq, nmu, nmu), np.complex128)
        for name, arr, om in (("W0_qmunu", W0, 0.0), ("Wp_qmunu", Wp, 2.0)):
            f[name] = arr
            f[name].attrs.update({"W0_source": "bgw_epsmat", "omega_ry_re": 0.0,
                                  "omega_ry_im": om, "bgw_reference": "/x", "bgw_files": "[]"})
        d = f.create_dataset("bgw_epsinv_head", data=np.array([0.08 + 0j, 0.5 + 0j]))
        d.attrs["omega_ry_re"] = np.zeros(2)
        d.attrs["omega_ry_im"] = np.array([0.0, 2.0])
    V = jax.device_put(np.zeros((nq, 8, 8), np.complex128), NamedSharding(mesh, P(None, "x", "y")))
    W, heads, _ = rb.read_projected_screening(str(path), {"static": 0j, "probe": 2j}, V, mesh)
    assert np.allclose(np.asarray(W["static"].values)[:, :nmu, :nmu], W0)
    assert np.allclose(np.asarray(W["probe"].values)[:, :nmu, :nmu], Wp)
    assert W["static"].is_whole_zone() and heads == {"static": 0.08 + 0j, "probe": 0.5 + 0j}
    with pytest.raises(ValueError, match="GATE screening_source_frequency"):
        rb.read_projected_screening(str(path), {"probe": 2.5j}, V, mesh)
    with h5py.File(path, "a") as f:
        f["W0_qmunu"].attrs["W0_source"] = "lorrax"
        del f["Wp_qmunu"]
    with pytest.raises(ValueError, match="GATE screening_source_provenance"):
        rb.read_projected_screening(str(path), {"static": 0j}, V, mesh)
    with pytest.raises(ValueError, match="GATE screening_source_dataset"):
        rb.read_projected_screening(str(path), {"probe": 2j}, V, mesh)
