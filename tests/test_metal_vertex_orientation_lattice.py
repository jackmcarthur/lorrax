"""Four-current stream against a fractional, TR-broken supercell Kubo sum.

The small CPU plant uses the same explicitly declared FFT/GEMM stand-ins as
the charge orientation plant. It tests nonzero real times and all sixteen
vertex blocks: the stream builds each (charge, current) family pair's Green
and applies the Dirac vertices on its spin indices.  Red twins: the wrong q
orientation, and a current family that is not the one the T channels read.
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
    from gw.photon_layout import PhotonBasisLayout, PhotonFamilies
    from gw.response_bank import PhotonEndpoints
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
    # One site pair serves both centroid families (charge, current); the
    # packed photon order on a 1x1 mesh is C, T1, T2, T3.
    layout = PhotonBasisLayout.from_centroid_extents(2, 2, mesh, packed=True)
    families = PhotonFamilies(plans=(None, None), packed_layout=layout, layout=layout)
    mun = tuple(put(x.transpose(0, 3, 2, 1), PSI_MUN_SPEC) for x in (cell, cell))
    nmu = tuple(put(x.transpose(0, 1, 3, 2), PSI_NMU_SPEC) for x in (cell, cell))
    kernel = _get_chi_fractional_contour_kernel_face(
        mesh, (lat.n1, lat.n2, 1), len(times), (lat.nk, 8, 8, 4),
        ordered=True, vertex=families)
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

    # Red twins: wrong q orientation, and a current family whose two sites
    # are exchanged on one endpoint (the T channels must read the current
    # family, the C channel the charge family). Each must be rejected by the
    # same known-answer comparison.
    assert np.linalg.norm(got[lat.minus()] - expected) / np.linalg.norm(expected) > 1e-3
    swapped = put(cell[:, :, ::-1].transpose(0, 1, 3, 2), PSI_NMU_SPEC)
    red = evaluate(mun, (nmu[0], swapped))
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
        vertex=PhotonEndpoints(families, mun, nmu, put(e)))
    for power, value in enumerate((o0, a0, o1, a1)):
        coefficient = np.einsum("ab,abmA,abrnB->mArnB",
            df*(-delta)**power, rho[:, :, 0], rho.conj())
        reference = []
        for q in lat.kfrac:
            phase = np.exp(2j*np.pi*(lat.cells @ q))
            reference.append(np.einsum("mArnB,r->AmBn", coefficient, phase).reshape(8, 8))
        np.testing.assert_allclose(gather_to_host(value), reference, rtol=3e-11, atol=3e-11)

    # Remote windows retain independent even/odd orientations with fractional
    # occupations on BOTH sides. The vertex stays on the upper endpoint in
    # the reverse occupation term; swapping it conjugates the wrong channel.
    lower, upper = e < mu, e >= mu
    refs = np.array([e[lower].max(), e[upper].min()])
    tau = np.array([0.23, 0.71])
    projections = np.zeros((8, 2), complex)
    projections[:2] = np.eye(2)
    projections[6:] = np.eye(2)
    laplace = _get_chi_fractional_contour_kernel_face(mesh, (lat.n1, lat.n2, 1),
        4, (lat.nk, 8, 8, 4), selected_q=tuple(range(lat.nk)),
        pair_mode="laplace_ordered", vertex=families)
    result = laplace(put(tau), put(projections), mun, nmu, put(e),
        put(np.stack([f*lower, (1-f)*lower])),
        put(np.stack([(1-f)*upper, f*upper])), put(refs))
    reference = np.empty((lat.nk, 4, 8, 8), complex)
    flat_f = f.reshape(-1)
    forward = flat_f[:, None]*(1-flat_f[None, :])
    backward = (1-flat_f[:, None])*flat_f[None, :]
    pairs = lower.reshape(-1)[:, None] & upper.reshape(-1)[None, :]
    for it, t in enumerate(tau):
        for offset, weights, parity in ((0, forward-backward, 1), (2, forward+backward, -1)):
            weights = weights*pairs*np.exp((delta+refs[1]-refs[0])*t)
            c = np.einsum("ab,abmA,abrnB->mArnB", weights, rho[:, :, 0], rho.conj())
            c = c + parity*c.conj()
            for iq, q in enumerate(lat.kfrac):
                phase = np.exp(2j*np.pi*(lat.cells @ q))
                reference[iq, offset+it] = np.einsum("mArnB,r->AmBn", c, phase).reshape(8, 8)
    np.testing.assert_allclose(np.asarray(gather_to_host(result))/np.sqrt(lat.nk),
                               reference, rtol=3e-12, atol=3e-13)

    # Static FD reference at q=0. The same stream includes the -f' diagonal
    # contribution; removing it yields the finite-grid interband Pi_grid.
    import minimax
    rule = minimax.matsubara_response_rule(1/0.7, float(e.max()-e.min()), (0,))
    static = _get_chi_fractional_contour_kernel_face(mesh, (lat.n1, lat.n2, 1),
        1, (lat.nk, 8, 8, 4), selected_q=(0,), pair_mode="kms_static",
        ordered=True, vertex=families)
    result = static(put(rule["t"]), put(rule["weights"]), mun, nmu, put(e),
                    put(np.ones_like(f)), put(np.ones_like(f)), put(np.array([1/0.7, mu])))
    weights = np.divide(df, delta, out=np.zeros_like(df), where=delta != 0)
    np.fill_diagonal(weights, -f.reshape(-1)*(1-f.reshape(-1))/0.7)
    correlation = np.einsum("ab,abmA,abrnB->mArnB", weights, rho[:, :, 0], rho.conj())
    reference = np.sum(correlation, axis=2).transpose(1, 0, 3, 2).reshape(8, 8)
    np.testing.assert_allclose(np.asarray(gather_to_host(result))[0, 0]/np.sqrt(lat.nk),
                               reference, rtol=2e-8, atol=2e-9)


def test_fractional_vertex_stream_supercell(cpu_standins):
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    def put(x, spec=P()):
        return jax.device_put(np.asarray(x), NamedSharding(mesh, spec))
    check_fractional_vertex_stream_supercell(mesh, put)
