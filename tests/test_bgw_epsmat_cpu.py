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

