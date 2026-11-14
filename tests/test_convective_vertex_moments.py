r"""Continuity moments of the convective current vertex against the commutator i[H, F].

CPU, pytest, no solid and no GPU.  One fully relativistic hydrogen
pseudopotential (``tests/hsuite/fixture/H.upf``, j-dependent D on s and p) in
a cubic box, plane waves on the whole FFT box, three random localized
two-component states.  The carriers are the production ones:
``common.bispinor_init.lift_to_4spinor`` (raw) and
``convective_carrier`` with ``psp.vnl_ops.convective_velocity_field``.

1. **L = 1, 2 in real space, to round-off.**  With the moment projectors
   $Y_a = (x_a - \tau_a)\beta$ formed on the grid (``dZ = -iY``), the nonlocal
   part of the channel-a pair current $j^a = \Psi^{(a)\dagger}\alpha^a\Psi^{(a)}$
   satisfies $\sum_a\int j^a_{\rm nl}\,\partial_aF = {\tt HALFALPHA}\,
   \langle m|i[V_{\rm NL},F]|n\rangle$ (Ry) for $F = x_b$ and $x_bx_c$: the
   first by definition, the second by the Jacobi identity
   ($[x_c,[V,x_b]] = [x_b,[V,x_c]]$), which a wrong placement of
   $\sigma^a$ or a wrong scale breaks.  $F = x^3$, $xyz$ leave a defect, so
   the check can fail.  The kinetic vertex (``kinetic``) misses exactly
   ${\tt HALFALPHA}\langle i[V_{\rm NL},F]\rangle$, the known amount, and its
   carrier is the raw lift itself.  (The kinetic current's own moments,
   $\{p_a,\partial_aF\} = i[p^2,F]$, are a continuum identity common to both
   vertices.)
2. **L = 1 of the Bloch vertex with the production dZ.**  At a general k on a
   sphere, $\langle\Psi^{(a)}_m|\alpha^a|\Psi^{(a)}_n\rangle = {\tt HALFALPHA}
   \langle m|2(k+G)_a + \partial_aV_{\rm NL}|n\rangle$ to round-off, the
   analytic $\partial_aV_{\rm NL}$ agrees with the finite difference of
   $V_{\rm NL}(k)$ to the radial-table accuracy (6e-5 here, printed), and
   the grid moment projectors reproduce the analytic ones (discretization
   level, printed).
3. **Linearity in alpha.**  ``LORRAX_DIAG_ALPHA_FS_SCALE = s`` scales the
   convective increment by exactly s (fresh interpreter).
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "hsuite" / "fixture"
A_BOX, N_GRID = 10.0, 20


def _setup():
    import jax
    jax.config.update("jax_enable_x64", True)
    pytest.importorskip("xsdata")
    from psp.pseudos import load_pseudopotentials
    from psp.vnl_ops import build_vnl_setup
    pseudos = load_pseudopotentials(str(FIXTURE_DIR))
    crystal = SimpleNamespace(
        blat=2.0 * np.pi / A_BOX, bvec=np.eye(3), cell_volume=A_BOX ** 3,
        atom_crys=np.array([[0.5, 0.5, 0.5]]), atom_types=np.array([1]),
        spinorbit=True, nspinor=2)
    qmax = np.sqrt(3.0) * np.pi * N_GRID / A_BOX
    setup = build_vnl_setup(crystal, pseudos=pseudos, nspinor=2, q_max=qmax,
                            print_fn=lambda *_a: None)
    assert setup.soc and setup.total_R > 0
    return setup


def _box_basis():
    """Miller indices of the whole FFT box, in the C order of the grid."""
    m = np.fft.fftfreq(N_GRID, 1.0 / N_GRID).astype(int)
    G = np.stack(np.meshgrid(m, m, m, indexing="ij"), -1).reshape(-1, 3)
    return G


def _to_grid(c):
    """Unitary: G coefficients (..., N) -> grid values (..., N)."""
    shape = c.shape[:-1] + (N_GRID,) * 3
    return (np.fft.ifftn(c.reshape(shape), axes=(-3, -2, -1)) * np.sqrt(N_GRID ** 3)
            ).reshape(c.shape)


def _to_G(f):
    shape = f.shape[:-1] + (N_GRID,) * 3
    return (np.fft.fftn(f.reshape(shape), axes=(-3, -2, -1)) / np.sqrt(N_GRID ** 3)
            ).reshape(f.shape)


def _displacements():
    """x - tau on the grid (atom at the box centre), bohr, (3, N)."""
    h = A_BOX / N_GRID
    x = np.arange(N_GRID) * h - 0.5 * A_BOX
    return np.stack(np.meshgrid(x, x, x, indexing="ij"), 0).reshape(3, -1)


def _states(rng, nb=3):
    x0 = _displacements()
    psi = np.zeros((nb, 2, x0.shape[1]), complex)
    for n in range(nb):
        x = x0 - rng.normal(scale=0.4, size=3)[:, None]           # off-centre: no accidental zeros
        r2 = np.sum(x ** 2, axis=0)
        for s in range(2):
            c = rng.normal(size=4) + 1j * rng.normal(size=4)
            psi[n, s] = (c[0] + c[1] * x[0] + c[2] * x[1] + c[3] * x[2] * x[0]) * np.exp(
                -r2 / (2.0 * (0.9 + 0.2 * n) ** 2))
        psi[n] /= np.sqrt(np.sum(np.abs(psi[n]) ** 2))
    return psi


def _carriers(psi_G, sigma_gamma, gvecs, vertex):
    import jax.numpy as jnp
    from common.bispinor_init import convective_carrier, lift_to_4spinor
    B = np.eye(3) * 2.0 * np.pi / A_BOX
    raw = lift_to_4spinor(jnp.asarray(psi_G)[None], jnp.asarray(gvecs, float)[None],
                          jnp.zeros((1, 3)), jnp.asarray(B), representation="raw")
    return raw, [convective_carrier(raw, jnp.asarray(sigma_gamma)[:, None], a, vertex=vertex)
                 for a in (1, 2, 3)]


def test_real_space_moments_equal_the_commutator():
    import jax.numpy as jnp
    from common.bispinor_init import HALFALPHA
    from common.gamma_matrices import gamma1, gamma2, gamma3
    from psp.vnl_ops import apply_vnl, build_vnl_kdata_from_kvec, convective_velocity_field
    setup = _setup()
    G = _box_basis()
    kd = build_vnl_kdata_from_kvec(np.zeros(3), G, setup, compute_dZ=True)
    Z, E = np.asarray(kd.Z), setup.E_super
    x = _displacements()
    Y = _to_G(x[:, None, :] * _to_grid(Z)[None])                 # (3, R, N): <G|(x-tau)beta>
    dZ_grid = -1j * Y
    psi_r = _states(np.random.default_rng(7))
    psi_G = _to_G(psi_r)
    sg = np.asarray(convective_velocity_field(jnp.asarray(psi_G), jnp.asarray(Z),
                                              jnp.asarray(dZ_grid), E))
    raw, conv = _carriers(psi_G, sg, G, "convective")
    raw_k, kin = _carriers(psi_G, sg, G, "kinetic")
    assert all(kin[a] is raw_k for a in range(3))                  # the raw lift itself
    alphas = (gamma1, gamma2, gamma3)

    def pair_current(carrier, alpha):
        f = _to_grid(np.asarray(carrier)[0])                       # (nb, 4, N)
        return np.einsum("msr,st,ntr->mnr", np.conj(f), np.asarray(alpha), f)

    j_nl = np.stack([pair_current(conv[a], alphas[a]) - pair_current(raw, alphas[a])
                     for a in range(3)])                           # (3, nb, nb, N)
    j_kin_nl = np.stack([pair_current(kin[a], alphas[a]) - pair_current(raw, alphas[a])
                         for a in range(3)])
    assert not np.any(j_kin_nl)
    Vpsi_r = _to_grid(np.asarray(apply_vnl(jnp.asarray(psi_G), jnp.asarray(Z), E)))

    def commutator(F):
        # <m| i(V F - F V) |n>, F real and diagonal on the grid, V Hermitian.
        return 1j * (np.einsum("msr,nsr->mn", np.conj(Vpsi_r), F * psi_r)
                     - np.einsum("msr,nsr->mn", np.conj(F * psi_r), Vpsi_r))

    def moment(grad):
        return np.einsum("amnr,ar->mn", j_nl, grad)

    one = np.ones(x.shape[1])
    zero = np.zeros(x.shape[1])
    cases = {}
    for b in range(3):
        grad = np.stack([one if a == b else zero for a in range(3)])
        cases[f"L1 x{b}"] = (x[b], grad, 1)
        for c in range(b, 3):
            grad = np.stack([(x[c] if a == b else zero) + (x[b] if a == c else zero)
                             for a in range(3)])
            cases[f"L2 x{b}x{c}"] = (x[b] * x[c], grad, 2)
    cases["L3 x0^3"] = (x[0] ** 3, np.stack([3 * x[0] ** 2, zero, zero]), 3)
    cases["L3 x0x1x2"] = (x[0] * x[1] * x[2], np.stack([x[1] * x[2], x[0] * x[2], x[0] * x[1]]), 3)
    worst, defect = 0.0, np.inf
    wants = {name: HALFALPHA * commutator(F) for name, (F, _, _) in cases.items()}
    # One scale per order L: the largest |HALFALPHA <i[V,F]>| of that order.
    scale = {L: max(float(np.abs(wants[n]).max()) for n, c in cases.items() if c[2] == L)
             for L in (1, 2, 3)}
    for name, (F, grad, L) in cases.items():
        want = wants[name]
        got = moment(grad)
        rel = float(np.abs(got - want).max() / scale[L])
        print(f"{name}: |moment - HALFALPHA<i[V,F]>| / scale_L = {rel:.2e}; the kinetic vertex "
              f"misses {float(np.abs(want).max()):.3e} (= max |HALFALPHA<i[V,F]>|)")
        if L <= 2:
            worst = max(worst, rel)
        else:
            defect = min(defect, rel)
    assert worst < 1e-12, worst
    assert defect > 1e-3, f"L = 3 moments agree ({defect}): the check is blind"


def test_bloch_vertex_with_the_production_derivative():
    import jax.numpy as jnp
    from common.bispinor_init import HALFALPHA
    from common.gamma_matrices import gamma1, gamma2, gamma3
    from psp.vnl_ops import (build_vnl_kdata_from_kvec, convective_velocity_field,
                             vnl_matrix, vnl_velocity_finite_difference, vnl_velocity_matrix)
    setup = _setup()
    rng = np.random.default_rng(19)
    B = np.eye(3) * 2.0 * np.pi / A_BOX
    k = np.array([0.13, -0.21, 0.07])
    G_all = _box_basis()
    sphere = G_all[np.sum(((G_all + k) @ B) ** 2, axis=1) < 12.0]
    kd = build_vnl_kdata_from_kvec(k, sphere, setup, compute_dZ=True)
    Z, dZ, E = kd.Z, kd.dZ, setup.E_super
    nb, ng = 4, sphere.shape[0]
    psi = rng.normal(size=(nb, 2, ng)) + 1j * rng.normal(size=(nb, 2, ng))
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2, axis=(1, 2)))[:, None, None]
    psi = jnp.asarray(psi)
    sg = convective_velocity_field(psi, Z, dZ, E)
    from common.bispinor_init import convective_carrier, lift_to_4spinor
    raw = lift_to_4spinor(psi[None], jnp.asarray(sphere, float)[None], jnp.asarray(k)[None],
                          jnp.asarray(B), representation="raw")
    v_nl = np.asarray(vnl_velocity_matrix(psi, Z, dZ, E))                  # (3, nb, nb), Ry bohr
    K = (sphere + k) @ B
    p2 = np.einsum("msg,ga,nsg->amn", np.conj(np.asarray(psi)), 2.0 * K, np.asarray(psi))
    for a, alpha in zip((1, 2, 3), (gamma1, gamma2, gamma3)):
        c = np.asarray(convective_carrier(raw, sg[:, None], a, vertex="convective")[0])
        r = np.asarray(raw[0])
        vertex = np.einsum("msg,st,ntg->mn", np.conj(c), np.asarray(alpha), c)
        kinetic = np.einsum("msg,st,ntg->mn", np.conj(r), np.asarray(alpha), r)
        want = HALFALPHA * (p2[a - 1] + v_nl[a - 1])
        np.testing.assert_allclose(vertex, want, rtol=0, atol=1e-13 * np.abs(want).max())
        np.testing.assert_allclose(want - kinetic, HALFALPHA * v_nl[a - 1],
                                   rtol=0, atol=1e-13 * np.abs(want).max())

    def matrix_at(kc):
        z = build_vnl_kdata_from_kvec(np.asarray(kc), sphere, setup).Z
        return np.asarray(vnl_matrix(psi, z, E))
    fd = vnl_velocity_finite_difference(matrix_at, k, sphere, B, h=1e-3, h_rel=0.0,
                                        scheme="richardson")
    rel_fd = float(np.abs(fd - v_nl).max() / np.abs(v_nl).max())
    print(f"analytic d_a V_NL against the finite difference of V_NL(k): {rel_fd:.2e}")
    # Linear interpolation of the radial table (dq ~ 5e-4) against its analytic
    # derivative table: agreement at the table accuracy, not round-off.
    assert rel_fd < 1e-3, rel_fd

    # The grid moment projectors of test 1 against the analytic ones (k = 0, whole box).
    G = _box_basis()
    k0 = build_vnl_kdata_from_kvec(np.zeros(3), G, setup, compute_dZ=True)
    Zb = np.asarray(k0.Z)
    Y = _to_G(_displacements()[:, None, :] * _to_grid(Zb)[None])
    psi_G = jnp.asarray(_to_G(_states(np.random.default_rng(7))))
    grid = np.asarray(vnl_velocity_matrix(psi_G, k0.Z, jnp.asarray(-1j * Y), E))
    analytic = np.asarray(vnl_velocity_matrix(psi_G, k0.Z, k0.dZ, E))
    rel_grid = float(np.abs(grid - analytic).max() / np.abs(analytic).max())
    print(f"grid moment projectors against the analytic dZ (box {A_BOX} bohr, {N_GRID}^3): "
          f"{rel_grid:.2e}")
    assert rel_grid < 1e-2, rel_grid


_LINEARITY = r"""
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from common.bispinor_init import HALFALPHA, convective_carrier, lift_to_4spinor
rng = np.random.default_rng(5)
psi = jnp.asarray(rng.normal(size=(1, 2, 2, 9)) + 1j * rng.normal(size=(1, 2, 2, 9)))
sg = jnp.asarray(rng.normal(size=(3, 1, 2, 2, 9)) + 1j * rng.normal(size=(3, 1, 2, 2, 9)))
raw = lift_to_4spinor(psi, jnp.asarray(rng.integers(-2, 3, (1, 9, 3)).astype(float)),
                      jnp.zeros((1, 3)), jnp.eye(3), representation="raw")
inc = np.concatenate([np.asarray(convective_carrier(raw, sg, a, vertex="convective") - raw).ravel()
                      for a in (1, 2, 3)])
np.save("OUT", inc)
print(repr(HALFALPHA))
"""


def test_increment_is_linear_in_alpha(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env["JAX_PLATFORMS"] = "cpu"
    out = {}
    for s in ("", "0.5"):
        env.pop("LORRAX_DIAG_ALPHA_FS_SCALE", None)
        if s:
            env["LORRAX_DIAG_ALPHA_FS_SCALE"] = s
        path = tmp_path / f"inc{s or '1'}.npy"
        done = subprocess.run([sys.executable, "-c", _LINEARITY.replace("OUT", str(path))],
                              env=env, capture_output=True, text=True, check=True)
        out[s] = (float(done.stdout.strip().splitlines()[-1]), np.load(path))
    (h1, inc1), (h2, inc2) = out[""], out["0.5"]
    assert h2 == 0.5 * h1
    assert np.max(np.abs(inc1)) > 0
    np.testing.assert_allclose(inc2, 0.5 * inc1, rtol=0, atol=1e-15 * np.max(np.abs(inc1)))
