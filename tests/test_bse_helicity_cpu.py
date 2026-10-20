"""In-plane helicity of BSE states: the campaign convention on a C3 toy, and
one table on the native and ``bse_k_grid`` routes.

σ+ is the field E₀(cos ωt, sin ωt, 0).  A state whose orbital angular momentum
about +z is one unit above the ground state's absorbs σ+ only.  The toy is a
triangular lattice of s, p_x, p_y orbitals (two s guards above) with C3
Slater-Koster hoppings and an orbital Zeeman term B L_z on the p shell: at Γ
the conduction states are p± = (p_x ± i p_y)/√2 with ⟨L_z⟩ = ±1, which names
each transition's helicity without the formula under test.  The red twins flip
the convention (I+ <-> I-, or the conjugated ⟨S|r|0⟩ spelling of D) and must
fail the same check.

CPU, 2x2 host mesh: XLA_FLAGS=--xla_force_host_platform_device_count=4.
"""
import ast
import os
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

import bse.absorption_common as absorption_common  # noqa: E402
from bandstructure.bse_setup import compute_wfns_fi  # noqa: E402
from bse.absorption_common import (exciton_dipole_projections,  # noqa: E402
                                   helicity_table_lines, inplane_helicity,
                                   slice_dipole_to_bse_window)
from bse.bse_densify import densified_dipoles  # noqa: E402
from ffi.fft import xla_reference  # noqa: E402
from isdf.galerkin import NodalInterp  # noqa: E402

A1, A2 = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3.0) / 2.0])
HALF_NN = ((1, 0), (0, 1), (-1, 1))            # the other three are -n
ONSITE = np.array([-2.0, -1.0, 1.0, 1.0, 4.0, 4.6])   # s_a s_b p_x p_y s_c s_d
P_ROWS = (2, 3)
B_Z = 0.15                                     # orbital Zeeman on the p shell
NB, B_MIN, B_MAX, N_VAL, N_OCC = 6, 0, 4, 2, 2  # window [0,4): 2 v, 2 c
KC, KF = (6, 6, 1), (9, 9, 1)                  # Γ, K, K' and 6 more shared
L_Z = np.zeros((6, 6), complex)
L_Z[2, 3], L_Z[3, 2] = -1j, 1j                 # L_z on (p_x, p_y)


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _lattice():
    """{n: H_n}: H(k) = Σ_n H_n exp(2πi k·n), n in lattice coordinates."""
    t_ss = {(0, 0): 0.05, (1, 1): -0.06, (0, 1): 0.02, (4, 4): 0.04,
            (5, 5): -0.03, (4, 5): 0.01, (1, 4): 0.02}
    t_sp = {0: 0.10, 1: 0.18, 4: 0.12, 5: 0.07}
    t_sigma, t_pi = 0.20, -0.06
    h0 = np.diag(ONSITE).astype(complex) + B_Z * L_Z
    hop = {(0, 0): h0}
    for n in HALF_NN:
        d = n[0] * A1 + n[1] * A2
        u = d / np.linalg.norm(d)
        h = np.zeros((6, 6), complex)
        for (i, j), t in t_ss.items():
            h[i, j] = h[j, i] = t
        for i, t in t_sp.items():               # Slater-Koster s-p: odd in u
            for a, p in enumerate(P_ROWS):
                h[i, p], h[p, i] = t * u[a], -t * u[a]
        for a, p in enumerate(P_ROWS):
            for b, q in enumerate(P_ROWS):
                h[p, q] = (t_sigma * u[a] * u[b]
                           + t_pi * (float(a == b) - u[a] * u[b]))
        hop[n], hop[(-n[0], -n[1])] = h, h.conj().T
    return hop


def _grid(n):
    axes = [np.arange(v) / v for v in n]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return (g + 0.5) % 1.0 - 0.5


def _states(kpts, hop=None):
    """C (nk, NB, 6) = ⟨orbital|ψ⟩ᵀ, E (nk, NB), v (nk, 3, NB, NB) =
    ⟨m|∂H/∂k|n⟩ with Cartesian x, y (z = 0), and ⟨L_z⟩ (nk, NB)."""
    hop = _lattice() if hop is None else hop
    cs, es, vs, lz = [], [], [], []
    for k in kpts:
        ph = {n: np.exp(2j * np.pi * (k[0] * n[0] + k[1] * n[1])) for n in hop}
        hk = sum(ph[n] * h for n, h in hop.items())
        dk = [sum(1j * (n[0] * A1 + n[1] * A2)[a] * ph[n] * h
                  for n, h in hop.items()) for a in (0, 1)]
        e, u = np.linalg.eigh(hk)
        cs.append(u.T)
        es.append(e)
        vs.append(np.stack([u.conj().T @ d @ u for d in dk]
                           + [np.zeros((NB, NB))]))
        lz.append(np.real(np.einsum("an,ab,bn->n", u.conj(), L_Z, u)))
    return np.array(cs), np.array(es), np.array(vs), np.array(lz)


def _window_dipoles(e, v):
    """Native route: the dipole.h5 slice, d_t = ⟨ck|v̂|vk⟩/ΔE, (3, nk, nc, nv)."""
    cart = np.moveaxis(v, 1, 0)                          # (3, nk, nb, nb)
    delta = e[:, :, None] - e[:, None, :]
    return slice_dipole_to_bse_window(cart, delta, N_OCC, N_VAL,
                                      B_MAX - B_MIN - N_VAL)[0]


def _transition_dipoles(d_alpha):
    """⟨0|r|S⟩ of every single transition, through THE contraction."""
    _, nk, nc, nv = d_alpha.shape
    eye = np.eye(nk * nc * nv).reshape(nk * nc * nv, nk, nc, nv)
    return exciton_dipole_projections(eye, d_alpha)      # (N, 3)


def _assert_named_helicity(eta, lz_c):
    """Each Γ transition into p± is pure σ±: η = sign⟨L_z⟩ of its c state."""
    assert np.all(np.abs(np.abs(lz_c) - 1.0) < 1e-12), lz_c
    np.testing.assert_allclose(eta, np.sign(lz_c), atol=1e-10)


def _gamma():
    _, e, v, lz = _states(np.zeros((1, 3)))
    d = _window_dipoles(e, v)                             # (3, 1, 2, 2)
    lz_c = np.repeat(lz[0, N_OCC:B_MAX], N_VAL)           # per (c, v), C order
    return _transition_dipoles(d), lz_c


def test_c3_toy_states_take_the_field_defined_labels():
    dip, lz_c = _gamma()
    i_plus, i_minus, eta = inplane_helicity(dip, 1)
    _assert_named_helicity(eta, lz_c)
    np.testing.assert_allclose(i_plus + i_minus,
                               (np.abs(dip[:, :2]) ** 2).sum(1), rtol=1e-12)
    assert np.all(i_plus[lz_c > 0] > 1e-3) and np.all(i_minus[lz_c < 0] > 1e-3)


def test_flipped_conventions_fail_the_c3_check():
    """Red twins: I+ and I- swapped, or D conjugated (the ⟨S|r|0⟩ spelling)."""
    dip, lz_c = _gamma()
    swapped = inplane_helicity(dip, 1)[2] * -1.0
    conjugated = inplane_helicity(np.conj(dip), 1)[2]
    for eta in (swapped, conjugated):
        with pytest.raises(AssertionError):
            _assert_named_helicity(eta, lz_c)


def test_phase_gauge_and_n_k_normalisation():
    """The velocity form's global -i, a band gauge carried by d and A alike,
    and the zone size do not move the per-cell table."""
    kc = _grid(KC)
    _, e, v, _ = _states(kc)
    d = _window_dipoles(e, v)                             # (3, nk, nc, nv)
    rng = np.random.default_rng(5)
    a = rng.standard_normal((5,) + d.shape[1:]) + 1j * rng.standard_normal(
        (5,) + d.shape[1:])
    a /= np.linalg.norm(a.reshape(5, -1), axis=1)[:, None, None, None]
    ref = inplane_helicity(exciton_dipole_projections(a, d), d.shape[1])
    theta = rng.uniform(0, 2 * np.pi, (d.shape[1], NB))
    th_c, th_v = theta[:, N_OCC:B_MAX], theta[:, B_MIN:N_OCC]
    gauge = np.exp(-1j * th_c[:, :, None] + 1j * th_v[:, None, :])
    for dd, aa in ((-1j * d, a), (d * gauge[None], a * gauge[None])):
        got = inplane_helicity(exciton_dipole_projections(aa, dd), d.shape[1])
        for x, y in zip(got, ref):
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-15)
    # One flat transition spread over the zone, d = ⟨c|v|v⟩/ΔE ∝ (1, -i): the
    # s -> p+ form (⟨c|r|v⟩ ∝ (1, -i)·i, so D ∝ (1, i)); per cell it is N_k-free.
    flat = np.zeros((1,) + d.shape[1:], complex)
    flat[0, :, 1, 1] = 1.0 / np.sqrt(d.shape[1])
    d_flat = np.zeros_like(d)
    d_flat[:, :, 1, 1] = np.array([1.0, -1j, 0.0])[:, None]
    ip, im, eta = inplane_helicity(exciton_dipole_projections(flat, d_flat),
                                   d.shape[1])
    np.testing.assert_allclose([ip[0], im[0], eta[0]], [2.0, 0.0, 1.0],
                               atol=1e-12)


def test_native_and_densified_routes_give_one_table():
    """Route B (coarse velocity through the htransform) against the exact fine
    states: identical where the grids share k (Γ, K, K', ...), and within the
    densifier's own interpolation error elsewhere."""
    mesh = _mesh()
    kc = _grid(KC)
    c, e, v, _ = _states(kc)
    nk = len(kc)
    sym = SimpleNamespace(
        irr_idx_k=np.arange(nk), sym_idx_k=np.zeros(nk, np.int32),
        sym_mats_k=np.zeros((2, 3, 3)), nk_red=nk, nk_tot=nk,
        unfolded_kpts=np.asarray(kc),
        cartesian_action=lambda rows, axial, time_odd: np.broadcast_to(
            np.eye(3), (len(np.asarray(rows)), 3, 3)))
    # The toy's orbitals as co-located nodes; their nodal values are ``c``.
    nodes = NodalInterp(jnp.asarray(c), np.zeros((c.shape[2], 3)),
                        np.diag([1.0, 1.0, 10.0]), 2, 1)
    with xla_reference():
        fine = compute_wfns_fi(
            ctilde=jnp.asarray(c), B_at_mu=nodes,
            enk_sigma=jnp.asarray(e.T), kgrid_co=KC, kgrid_fi=KF,
            band_window_fi=(B_MIN, B_MAX), mesh_xy=mesh, return_coeffs=True)
        dip = densified_dipoles(
            (v, e, "toy dH/dk", "toy_c3.h5"), sym=sym, fh_energies=e,
            basis=SimpleNamespace(ctilde=jnp.asarray(c), band_range=(0, NB),
                                  basis_at_nodes=nodes),
            fine=fine, coarse_grid=KC, n_val=N_VAL, window_start=B_MIN,
            a_band_index=None, mesh_xy=mesh, log_fn=lambda *a: None)
    q = np.asarray(fine.q_frac)
    _, e_f, v_f, _ = _states(q)
    nq = len(q)
    eta_b = inplane_helicity(_transition_dipoles(dip["d_alpha"]), nq)
    eta_n = inplane_helicity(_transition_dipoles(_window_dipoles(e_f, v_f)), nq)
    shared = np.array([np.any(np.all(np.abs((k - kc + 0.5) % 1.0 - 0.5)
                                     < 1e-9, axis=1)) for k in q])
    per_q = lambda x: x.reshape(nq, -1)                    # (q, c*v)
    for xb, xn in zip(eta_b[:2], eta_n[:2]):               # I+, I-
        scale = per_q(xn).max()
        np.testing.assert_allclose(per_q(xb)[shared], per_q(xn)[shared],
                                   rtol=0, atol=1e-12 * scale)
        # measured 6e-4 of the largest strength off the shared k (rms 7e-5)
        assert np.abs(per_q(xb) - per_q(xn)).max() < 3e-3 * scale
    on = per_q(eta_n[2])[shared]
    np.testing.assert_allclose(per_q(eta_b[2])[shared], on, atol=1e-10)
    gamma = int(np.flatnonzero(np.all(np.abs(q) < 1e-12, axis=1))[0])
    np.testing.assert_allclose(np.sort(per_q(eta_b[2])[gamma]),
                               [-1.0, -1.0, 1.0, 1.0], atol=1e-8)


def test_helicity_table_lines_state_the_convention():
    dip, lz_c = _gamma()
    lines = helicity_table_lines(np.array([1.5, 1.6, 2.0, 2.1]), dip, 1)
    text = "\n".join(lines)
    for phrase in ("E(t) = E₀ (cos ωt, sin ωt, 0)", "counter-clockwise",
                   "D = ⟨0|r|S⟩", "I± = |D_x ∓ i D_y|² / (2 N_k)",
                   "N_k = 1"):
        assert phrase in text, phrase
    rows = [line for line in lines if line.strip().startswith("S0")]
    assert len(rows) == 4
    for row, lz in zip(rows, lz_c):
        assert row.rstrip().endswith("+1.0000" if lz > 0 else "-1.0000"), row


def test_bse_jax_reads_the_flag_on_the_lanczos_route():
    """The flag is parsed and on the Lanczos route's read set (bse_jax's
    module import runs the runtime start-up, so the source is read)."""
    path = os.path.join(os.path.dirname(absorption_common.__file__),
                        "bse_jax.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    flags = {node.args[0].value for node in ast.walk(tree)
             if isinstance(node, ast.Call) and getattr(node.func, "attr", "")
             == "add_argument" and node.args
             and isinstance(node.args[0], ast.Constant)}
    assert "--helicity-table" in flags
    routes = next(node.value for node in ast.walk(tree)
                  if isinstance(node, ast.Assign)
                  and any(getattr(t, "id", "") == "_ROUTE_DESTS"
                          for t in node.targets))
    lanczos = routes.values[[k.value for k in routes.keys].index("lanczos")]
    assert "helicity_table" in {e.value for e in lanczos.elts}
