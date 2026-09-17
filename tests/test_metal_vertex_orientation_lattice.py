"""Four-current stream against a fractional, TR-broken supercell Kubo sum.

The small CPU plant uses the same explicitly declared FFT/GEMM stand-ins as
the charge orientation plant. It tests nonzero real times and all sixteen
vertex blocks, including a conjugation-error red twin.
"""
import numpy as np
import pytest
from types import SimpleNamespace

from test_metal_chi0_orientation_lattice import _Lattice, cpu_standins

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def check_fractional_vertex_stream_supercell(mesh, put):
    from common.collectives import gather_to_host
    from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC
    from common.gamma_matrices import _gamma_tables
    from gw.w_isdf import _get_chi_fractional_contour_kernel_face

    lat = _Lattice(n1=3, n2=1, n_sites=8)
    # Two physical sites, each carrying four spinor components at one point.
    lat.tau = np.repeat(lat.tau[::4], 4, axis=0)
    e, u = lat.bands()
    psi = lat.bloch_states(u).reshape(lat.nk, 8, lat.nk, 2, 4)
    mu = np.median(e[:, 3])
    f = 1 / (1 + np.exp((e - mu) / 0.7))
    assert np.any((f > 0.05) & (f < 0.95))
    assert np.max(np.abs(e - e[lat.minus()])) > 0.1
    times = np.array([0.37, 1.13, 2.9])

    # rho[a,b,R,mu,A] = <a|J_A(R,mu)|b>. The supercell sum is
    # -i sum_ab (f_a-f_b) exp(i(E_a-E_b)t) rho_ab(x) rho_ba(y).
    states = psi.reshape(-1, lat.nk, 2, 4)
    rho = np.einsum("armi, Aij, brmj->abrmA", states.conj(),
                    np.asarray(_gamma_tables), states)
    delta = e.reshape(-1)[:, None] - e.reshape(-1)[None, :]
    df = f.reshape(-1)[:, None] - f.reshape(-1)[None, :]
    expected = np.empty((lat.nk, len(times), 8, 8), complex)
    for it, t in enumerate(times):
        chi = -1j * np.einsum("ab,abmA,abrnB->mArnB",
                              df * np.exp(1j * delta * t),
                              rho[:, :, 0], rho.conj())
        for iq, q in enumerate(lat.kfrac):
            phase = np.exp(2j * np.pi * (lat.cells @ q))
            expected[iq, it] = np.einsum("mArnB,r->AmBn", chi, phase).reshape(8, 8)

    cell = np.sqrt(lat.nk) * psi[:, :, 0]
    # Vertex is applied to both endpoint carriers of G^>, before build_G.
    bare = np.concatenate([cell] * 4, axis=2)
    current = np.concatenate([
        np.einsum("ij,knmj->knmi", gamma, cell)
        for gamma in _gamma_tables], axis=2)
    mun = tuple(put(x.transpose(0, 3, 2, 1), PSI_MUN_SPEC) for x in (bare, current))
    nmu = tuple(put(x.transpose(0, 1, 3, 2), PSI_NMU_SPEC) for x in (bare, current))
    kernel = _get_chi_fractional_contour_kernel_face(
        mesh, (lat.n1, lat.n2, 1), len(times), (lat.nk, 8, 8, 4),
        ordered=True, vertex=True)
    def evaluate(left, right):
        values = kernel(put(times), put(np.eye(len(times), dtype=complex)),
                        left, right, put(e), put(f.astype(complex)),
                        put((1-f).astype(complex)), put(0.21))
        return np.stack([np.asarray(gather_to_host(v)) for v in values], axis=1) / np.sqrt(lat.nk)

    got = evaluate(mun, nmu)
    for A in range(4):
        for B in range(4):
            block = (slice(None), slice(None), slice(2*A, 2*A+2), slice(2*B, 2*B+2))
            np.testing.assert_allclose(got[block], expected[block], rtol=2e-12, atol=2e-13)

    # Red twins: wrong q orientation, and conjugating alpha_y on just one
    # endpoint. Each must be rejected by the same known-answer comparison.
    assert np.linalg.norm(got[lat.minus()] - expected) / np.linalg.norm(expected) > 1e-3
    wrong_current = current.copy()
    wrong_current[:, :, 4:6] *= -1
    red = evaluate(mun, (nmu[0], put(wrong_current.transpose(0, 1, 3, 2), PSI_NMU_SPEC)))
    assert np.linalg.norm(red - expected) / np.linalg.norm(expected) > 1e-3

    # The exact coefficients reuse this stream with energy-power weights.
    # These t=0 moment checks supplement, and never replace, the test above.
    from gw.response_bank import exact_bare_moments
    wfns = SimpleNamespace(enk=put(e), occ=put(f), green_parent=None, layout="face",
        slices=SimpleNamespace(b0=0, b4_logical=8, nb_full=8))
    meta = SimpleNamespace(nk_tot=lat.nk, nkx=lat.n1, nky=lat.n2, nkz=1,
        b_id_4_chi_user=8, nspin=1, nspinor=4, nspinor_wfnfile=2,
        mu_basis=SimpleNamespace(n_packed=8))
    a0, a1, o0, o1, _ = exact_bare_moments(wfns, meta, mesh_xy=mesh,
        q_ids=tuple(range(lat.nk)), execute=lambda kernel,args,label: kernel(*args),
        vertex=(mun, nmu, put(e)))
    for power, value in enumerate((o0, a0, o1, a1)):
        coefficient = np.einsum("ab,abmA,abrnB->mArnB",
            df*(-delta)**power, rho[:, :, 0], rho.conj())
        reference = []
        for q in lat.kfrac:
            phase = np.exp(2j*np.pi*(lat.cells @ q))
            reference.append(np.einsum("mArnB,r->AmBn", coefficient, phase).reshape(8, 8))
        np.testing.assert_allclose(gather_to_host(value), reference, rtol=3e-11, atol=3e-11)


def test_fractional_vertex_stream_supercell(cpu_standins):
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    def put(x, spec=P()):
        return jax.device_put(np.asarray(x), NamedSharding(mesh, spec))
    check_fractional_vertex_stream_supercell(mesh, put)
