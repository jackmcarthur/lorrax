"""The q → 0 dipole wings of the BerkeleyGW ε⁻¹ export (docs/theory/bgw_epsmat_export.md §5, E9–E14), CPU 2x2.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``
(CPU clause of the four-GPU rule: device-count logic only).

1. The wing L that GW stores is the production head-wing kernel
   (``qsgw_head.head_wings_sharded``).  Contracted with the density jets
   D = v/ΔE in place of the centroid pair products it must return −Ω S, S the
   production S tensor (``qsgw_head.head_s_tensor_sharded``): the identity
   that fixes g, the sign and the conjugation of (E9).
2. The q0 block written from the body and the wings (``epsmat_export.
   q0_dipole_terms`` + ``epsinv_block``) equals an INDEPENDENT dense G-space
   inverse ε⁻¹ = (1 − v χ₀)⁻¹ of a 3-D toy whose χ₀ carries a q-linear G = 0
   row and column: head (local fields), both wings and the O(1) body term.
Each check carries a red twin.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

A_LAT = 6.0
BVEC = 2 * np.pi / A_LAT * np.eye(3)
OMEGA = A_LAT ** 3
RNG = np.random.default_rng(23)


def _mesh():
    # The CPU clause: an emulated 2x2 HOST mesh, also inside a GPU process (`lx test`), where
    # jax.devices() are CUDA devices and SlabIO's serial (emulation) tier refuses them.
    cpu = jax.devices("cpu")
    if len(cpu) < 4:
        pytest.skip("needs 4 host CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(cpu[:4]).reshape(2, 2), ("x", "y"))


# ---------------------------------------------------------------------------
# 1. L contracted with the dipole jets is −Ω S
# ---------------------------------------------------------------------------

def _two_band_head(mesh, conj_pair=True):
    """Velocity (3, nk, 4, 4) with one active pair (c=2, v=0) per k, energies, occupations,
    and centroid faces whose pair product conj(ψ_c(μ)) ψ_v(μ) is the jet D^μ = v^μ_cv/ΔE
    (μ = x, y, z; a fourth centroid is zero).  ``conj_pair=False`` is the red twin."""
    from common.wfn_layout import psi_specs
    nk, nb, nmu = 6, 4, 4
    d = RNG.standard_normal((3, nk)) + 1j * RNG.standard_normal((3, nk))
    e = np.zeros((nk, nb)); e[:, 0] = -0.40 - 0.02 * np.arange(nk); e[:, 1] = -0.9
    e[:, 2] = 0.35 + 0.03 * np.arange(nk); e[:, 3] = 1.4
    f = np.zeros((nk, nb)); f[:, :2] = 1.0
    v = np.zeros((3, nk, nb, nb), np.complex128)
    v[:, :, 2, 0] = d; v[:, :, 0, 2] = np.conj(d)
    dE = e[:, 2] - e[:, 0]
    D = d / dE[None, :]                                   # density jet per k, (3, nk)
    psi = np.zeros((nk, 1, nmu, nb), np.complex128)       # (k, s, μ, n): ψ_n(r_μ)
    psi[:, 0, :3, 0] = 1.0
    psi[:, 0, :3, 2] = (np.conj(D) if conj_pair else D).T
    nmu_spec, mun_spec = psi_specs("face")
    put = lambda a, sp: jax.device_put(a, NamedSharding(mesh, sp))
    wfns = SimpleNamespace(layout="face", psi_mun=put(psi, mun_spec),
                           psi_nmu=put(np.transpose(psi, (0, 3, 1, 2)), nmu_spec),
                           enk=e, occ=f, slices=("one",))
    return v, e, f, wfns, nk


def test_wing_with_dipole_jets_is_minus_omega_S_and_red_twin():
    from gw.qsgw_head import head_s_tensor_sharded, head_wings_sharded
    mesh = _mesh()
    kw = dict(mesh=mesh, nb_logical=4, nspin=1, nspinor=1)
    for conj_pair in (True, False):
        v, e, f, wfns, nk = _two_band_head(mesh, conj_pair)
        vs = jax.device_put(v, NamedSharding(mesh, P(None, None, "x", "y")))
        S = np.asarray(head_s_tensor_sharded(vs, e, f, np.zeros(1), cell_volume=OMEGA,
                                             nk_tot=nk, **kw))[0]
        Y, Z = head_wings_sharded(vs, wfns, e, f, np.zeros(1), nk_tot=nk, **kw)
        L, R = np.asarray(Y)[0][:, :3], np.asarray(Z)[0][:3, :]
        err = np.linalg.norm(L + OMEGA * S) / np.linalg.norm(OMEGA * S)
        if conj_pair:
            # (E9): q·L·q = −Ω qᵀSq for every q, and R = Lᴴ at ω = 0.
            assert err < 1e-12, err
            assert np.allclose(R, np.conj(L).T, atol=1e-12 * np.abs(L).max())
            q = RNG.standard_normal(3)
            assert abs(q @ L @ q + OMEGA * (q @ S @ q)) < 1e-12 * abs(OMEGA * (q @ S @ q))
            assert np.all(np.linalg.eigvalsh(0.5 * (S + S.conj().T)).real < 0)   # χ₀₀ < 0
        else:
            # Red twin: the unconjugated pair product is not the density jet.
            assert err > 1e-2, err


# ---------------------------------------------------------------------------
# 2. The q0 block from body + wings equals the dense 3-D inverse
# ---------------------------------------------------------------------------

N_MU, N_MU_PAD, N_T = 10, 12, 7
MILLER = np.array([[h, k, l] for h in range(-2, 3) for k in range(-2, 3) for l in range(-2, 3)])


def _q0_toy(q0_frac=(1.0e-3, 0.0, 0.0)):
    """3-D toy at q0: G list (G = 0 first), ζ̃ (G, μ), v/Ω, and χ₀ = −Σ_t ρ_t ρ_tᴴ / Ω with
    ρ_t(G ≠ 0) = (Z C)_{G t} and ρ_t(0) = q0·m_t (the q-linear long-wavelength density):
    stable (χ₀ ≼ 0) and q-linear in its G = 0 row and column by construction."""
    q0 = np.asarray(q0_frac) @ BVEC
    K = (np.asarray(q0_frac)[None, :] + MILLER) @ BVEC
    k2 = np.sum(K * K, 1)
    keep = np.sum((MILLER @ BVEC) ** 2, 1) <= 2.2 * (2 * np.pi / A_LAT) ** 2
    order = np.argsort(np.sum((MILLER[keep] @ BVEC) ** 2, 1), kind="stable")
    G = MILLER[keep][order]
    assert np.all(G[0] == 0)
    Kg = (np.asarray(q0_frac)[None, :] + G) @ BVEC
    v = 8 * np.pi / np.sum(Kg * Kg, 1) / OMEGA                 # v(q0 + G)/Ω, Ry; v[0] = v(q0)/Ω
    Z = np.exp(-0.3 * np.sum((G @ BVEC) ** 2, 1))[:, None] * np.exp(
        -1j * (G @ BVEC) @ (RNG.random((N_MU, 3)) * A_LAT).T)
    Z = (Z + 0.1 * RNG.standard_normal(Z.shape)) / np.sqrt(OMEGA)
    C = RNG.standard_normal((N_MU, N_T)) + 1j * RNG.standard_normal((N_MU, N_T))
    m = 0.6 * (RNG.standard_normal((3, N_T)) + 1j * RNG.standard_normal((3, N_T)))
    vb = v.copy(); vb[0] = 0.0
    # scale the body so that ‖v^½ χ v^½‖ ~ 0.6 (static, passive)
    body = -(C @ C.conj().T)
    s = 0.6 / np.linalg.norm(np.sqrt(vb)[:, None] * (Z @ body @ Z.conj().T) * np.sqrt(vb)[None, :], 2)
    C *= np.sqrt(s)
    rho = np.vstack([q0 @ m, Z[1:] @ C])                        # (nG, N_T)
    chi0_G = -(rho @ rho.conj().T)                              # full G-space χ₀ with head and wings
    # μ-space objects of the export (normalisation of bgw_epsmat_export §1, (E9)):
    chi0_mu = -(C @ C.conj().T)                                 # body
    L = -(m @ C.conj().T)                                       # (3, μ):  χ₀(0,G) = Σ (q·L)_μ conj(Z_Gμ)
    R = -(C @ m.conj().T)                                       # (μ, 3):  χ₀(G,0) = Σ Z_Gμ (R q)_μ
    S_direct = -(np.conj(m) @ m.T) / OMEGA                       # χ₀₀ = Ω qᵀ S q  (S_ab ∝ conj(m_a) m_b)
    Zb = Z.copy(); Zb[0] = 0.0                                  # the bundle's Γ body: v(G = 0) = 0
    V = (Zb.conj().T * vb) @ Zb
    W = np.linalg.solve(np.eye(N_MU) - V @ chi0_mu, V)
    return dict(q0=q0, G=G, v=v, Z=Z, chi0_G=chi0_G, chi0_mu=chi0_mu, L=L, R=R, S=S_direct,
                V=V, W=W)


def _exported_q0_block(mesh, t, *, wings=True, swap=False):
    from gw import epsmat_export as ee
    n = t["G"].shape[0]
    face = NamedSharding(mesh, P(None, "x", "y"))
    pad = ((0, N_MU_PAD - N_MU), (0, N_MU_PAD - N_MU))
    Vf = jax.device_put(np.pad(t["V"], pad)[None], face)
    Wf = jax.device_put(np.pad(t["W"], pad)[None], face)
    Wcf = jax.device_put(np.pad(t["W"] - t["V"], pad)[None], face)
    plan = ee.eigh_plan(mesh, N_MU_PAD)
    B, _ = ee.coulomb_whitener(Vf, rcond=1e-12, plan=plan)
    chi = ee.response(B, ee.whitened_correlation(Wcf, B, mesh=mesh), mesh=mesh)
    zt = np.zeros((1, N_MU_PAD, n + 1), np.complex128); zt[0, :N_MU, :n] = t["Z"].T
    zeta = jax.device_put(zt, NamedSharding(mesh, P(None, ("x", "y"), None)))
    n_pad = ee._pad_to(n, 4)
    Zf = ee.zeta_face(zeta, np.arange(n), mesh=mesh, n_pad=n_pad)
    v_eps = t["v"].copy(); v_eps[0] = 0.0                      # the export's q0 rows
    q0 = t["q0"]
    if not wings:
        chi00 = OMEGA * (q0 @ t["S"] @ q0)
        head = 1.0 / (1.0 - t["v"][0] * chi00)
        blk, _ = ee.epsinv_block(Zf, chi, v_eps, mesh=mesh, n_valid=n, q0_head_value=head)
    else:
        L = np.zeros((3, N_MU_PAD), complex); L[:, :N_MU] = t["L"]
        R = np.zeros((N_MU_PAD, 3), complex); R[:N_MU] = t["R"]
        wl, wr = q0 @ L, R @ q0
        if swap:                                               # red twin: wings' conjugation swapped
            wl, wr = np.conj(wl), np.conj(wr)
        chi_q0, head, w, fig = ee.q0_dipole_terms(Zf, chi, Vf, Wf, wl, wr, v_eps, mesh=mesh,
                                                  v0=t["v"][0], chi00=OMEGA * (q0 @ t["S"] @ q0))
        blk, _ = ee.epsinv_block(Zf, chi_q0, v_eps, mesh=mesh, n_valid=n, q0_head_value=head,
                                 q0_wings=w)
    b = np.asarray(blk)[0, 0, 0][:n, :n]
    return (b[..., 0] + 1j * b[..., 1]).T                       # ε⁻¹(G, G')


def _parts(E, ref):
    n = E.shape[0]
    rel = lambda a, b: np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)
    return dict(head=abs(E[0, 0] - ref[0, 0]) / abs(1 - ref[0, 0]),
                row=rel(E[0, 1:], ref[0, 1:]), col=rel(E[1:, 0], ref[1:, 0]),
                body=np.linalg.norm(E[1:, 1:] - ref[1:, 1:]) / np.linalg.norm(ref[1:, 1:] - np.eye(n - 1)))


def test_q0_block_from_dipole_wings_equals_dense_3d_inverse_and_red_twins():
    mesh = _mesh()
    t = _q0_toy()
    n = t["G"].shape[0]
    ref = np.linalg.inv(np.eye(n) - t["v"][:, None] * t["chi0_G"])   # BerkeleyGW's ε⁻¹ = (1 − vχ₀)⁻¹
    # the toy is a 3-D semiconductor at q0: an O(1) LF head, O(1/q) row and O(q) column wings
    assert 0.02 < ref[0, 0].real < 0.9
    assert np.abs(ref[0, 1:]).max() > 1e2 * np.abs(ref[1:, 0]).max()
    got = _parts(_exported_q0_block(mesh, t), ref)
    for k, e in got.items():
        assert e < 1e-9, (k, e)
    # Red twin 1: the zero-wing file (head from the direct S, no body term) misses the
    # local-field head, both wings and the O(1) 3-D body term.
    zero = _parts(_exported_q0_block(mesh, t, wings=False), ref)
    assert zero["head"] > 1e-3 and zero["row"] == 1.0 and zero["col"] == 1.0
    assert zero["body"] > 1e-3, zero
    # Red twin 2: swapping the wings' conjugation breaks the wings and the body term.
    sw = _parts(_exported_q0_block(mesh, t, swap=True), ref)
    assert sw["row"] > 1e-3 and sw["col"] > 1e-3 and sw["body"] > 1e-4, sw


def test_q0_body_policy_with_dipole_wings_writes_3d():
    from gw.epsmat_export import q0_body_policy
    assert q0_body_policy(("eps0mat",), 3, "refuse", "dipole").startswith("complete")
    assert q0_body_policy(("eps0mat",), 2, "refuse", "dipole").startswith("complete")
    with pytest.raises(ValueError, match="epsmat_q0_body_3d"):
        q0_body_policy(("eps0mat",), 3, "refuse", "zero")


def test_q0_wings_key_is_validated(tmp_path):
    from gw.epsmat_config import EpsmatConfig
    base = ("[epsmat]\nsource_restart = r.h5\nparent_input_file = gw.in\noutput_dir = out\n"
            "epsilon_cutoff = 5\n")
    p = tmp_path / "a.in"; p.write_text(base)
    assert EpsmatConfig.from_input_file(str(p)).q0_wings == "dipole"
    p.write_text(base + "q0_wings = zero\n")
    assert EpsmatConfig.from_input_file(str(p)).q0_wings == "zero"
    p.write_text(base + "q0_wings = maybe\n")
    with pytest.raises(ValueError, match="epsmat_q0_wings_value"):
        EpsmatConfig.from_input_file(str(p))
