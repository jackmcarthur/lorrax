"""The ordered-orientation (time-reversal-odd) four-current vertex-pair kernel.

Every expected value here is the EXACT independent-particle response written
out in the test, on a planted system, never a second call into the code under
test.  DERIVATION_gnppm_nonhermitian.md section 1 at one k:

    chi_AB(i w) = sum_t [ P_t / (i w - D) - conj(P_t) / (i w + D) ],
    P_t[m, n]  = rho^A_t(m) conj(rho^B_t(n)),
    rho^A_t(m) = psi_v(m)^H gammatilde^A psi_c(m).

At one tau node (tau = 0) the kernel returns ``g * own + h * partner``, where
``own`` is its (-D)-pole orientation and ``partner`` the (+D)-pole one.  So a
planted weight ``gamma = -1 / (D + i w)`` on the ordered route (``h =
conj(gamma)``) must reproduce chi_AB(i w) exactly, and a planted real weight
``alpha = -D / (w^2 + D^2)`` on the even route (``h = g``) must reproduce the
EVEN-in-omega part of it.  That pins the ordering convention analytically: a
kernel that put gamma on the wrong orientation returns chi_AB(-i w) and fails.

Two plants.  A Theta-CLOSED Kramers set {v, Theta v} -> {c, Theta c} with
Theta = diag(i sigma_y, i sigma_y) K, where charge is Theta-even and alpha^i is
Theta-odd; and a Theta-BROKEN control {v, w} -> {c, d}.  On the closed set the
exact charge-current block is purely ODD in omega and zero at omega = 0 — the
continuity channel, which exists with or without time reversal — so the even
rule deletes it there while the ordered rule keeps it.  The numpy twin of these
statements is runs/frequency_integration_sandbox/421_bispw_20260915/harness/
kramers_lehmann.py.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

DELTA = 2.0          # every planted transition: occupied at -1, empty at +1
N_MU = 4
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
THETA_U = np.block([[1j * SIGMA_Y, np.zeros((2, 2))],
                    [np.zeros((2, 2)), 1j * SIGMA_Y]])
ALL_CLASSES = tuple((left, right)
                    for left in ((0,), (1, 2, 3))
                    for right in ((0,), (1, 2, 3)))


def _bands(kind, seed=84):
    """``(n_band=4, spin=4, mu)`` bands ordered occupied, occupied, empty, empty."""
    rng = np.random.default_rng(seed)

    def draw():
        return rng.normal(size=(4, N_MU)) + 1j * rng.normal(size=(4, N_MU))

    def partner(psi):
        return np.einsum("ab,bm->am", THETA_U, psi.conj())

    if kind == "kramers":
        v, c = draw(), draw()
        return np.stack([v, partner(v), c, partner(c)])
    if kind == "broken":
        return np.stack([draw(), draw(), draw(), draw()])
    raise ValueError(kind)


def _vertex_matrix(vertex):
    from common.gamma_matrices import gamma_perm_phase
    perm, phase = gamma_perm_phase(vertex)
    return np.asarray(phase)[:, None] * np.eye(4)[np.asarray(perm)]


def _operands(pairs):
    from common.gamma_matrices import gamma_perm_phase
    out = []
    for A, B in pairs:
        row = []
        for vertex in (A, B):
            perm, phase = gamma_perm_phase(vertex)
            row.extend((perm, jnp.conj(phase)))
        out.append(tuple(row))
    return tuple(out)


def _exact(bands, A, B, omega):
    """chi_AB(i omega), literal Lehmann sum over occupied x empty."""
    z = 1j * omega
    GA, GB = _vertex_matrix(A), _vertex_matrix(B)
    total = np.zeros((N_MU, N_MU), dtype=complex)
    for v in bands[:2]:
        for c in bands[2:]:
            rho_a = np.einsum("am,ab,bm->m", v.conj(), GA, c)
            rho_b = np.einsum("am,ab,bm->m", v.conj(), GB, c)
            Pt = rho_a[:, None] * rho_b.conj()[None, :]
            total += Pt / (z - DELTA) - Pt.conj() / (z + DELTA)
    return total


def _kernel(monkeypatch, bands):
    """The production vertex-pair kernel on the plant, FFT and GEMM as identities."""
    import common.fft_helpers as fft_helpers
    import distrib_la
    from gw.w_isdf import _get_chi_minimax_kernel, MinimaxNodes

    monkeypatch.setattr(fft_helpers, "make_flat_k_fftn",
                        lambda *a, **k: lambda x: x)

    def local_gemm_plan(mesh, **kwargs):
        gemm = lambda x, y: x @ y
        gemm.mesh = mesh
        gemm.in_sharding_a = NamedSharding(mesh, P(None, "x", "y"))
        gemm.in_sharding_b = gemm.in_sharding_a
        return gemm

    monkeypatch.setattr(distrib_la, "gemm_plan", local_gemm_plan)
    side = 2 if len(jax.devices()) >= 4 else 1
    mesh = Mesh(np.asarray(jax.devices()[:side * side]).reshape(side, side),
                ('x', 'y'))
    put = lambda x, spec: jax.device_put(jnp.asarray(x), NamedSharding(mesh, spec))
    mun = put(bands.transpose(1, 2, 0)[None], P(None, None, 'x', 'y'))
    nmu = put(bands[None], P(None, 'x', None, 'y'))
    energies = jnp.asarray([[-1.0, -1.0, 1.0, 1.0]])
    shape = (1, bands.shape[0], N_MU, 4)

    def run(weight, pairs, ordered):
        nodes = MinimaxNodes(t=jnp.asarray([0.], jnp.complex128),
                             alpha=jnp.asarray([weight], jnp.complex128))
        kernel = _get_chi_minimax_kernel(
            mesh, (1, 1, 1), layout='face', face_shape=shape,
            right_face_shape=shape, vertex_pairs=pairs,
            ordered_orientations=ordered)
        out = kernel(nodes, mun, nmu, energies < 0, energies > 0, energies,
                     jnp.asarray(-1.), jnp.asarray(1.), _operands(pairs))
        return [np.asarray(block)[0] for block in out]

    return run


def _assert_close(got, want, label, scale):
    """Absolute agreement at 1e-12 of the PLANT's scale.

    ``scale`` is the size of the plant's charge block, not of ``want``: several
    exact blocks are identically zero (the charge-current block of a
    Theta-closed set at omega = 0 and its even part at any omega), and a
    tolerance relative to an exact zero is a tolerance of zero.
    """
    np.testing.assert_allclose(
        got, want, rtol=0.0, atol=1e-12 * scale,
        err_msg=f"{label}: max|got - want| / plant scale = "
                f"{float(np.max(np.abs(got - want))) / scale:.3e}")


def _plant_scale(bands):
    return float(np.max(np.abs(_exact(bands, 0, 0, 0.0))))


@pytest.mark.parametrize("plant", ("kramers", "broken"))
@pytest.mark.parametrize("left_vertices,right_vertices", ALL_CLASSES)
@pytest.mark.parametrize("omega", (0.0, 0.7))
def test_ordered_kernel_is_the_exact_lehmann_response(
        monkeypatch, plant, left_vertices, right_vertices, omega):
    """G-B1.  gamma = -1/(D + i w) on the ordered route gives chi_AB(i w) EXACTLY.

    All sixteen blocks, both plants, at omega = 0 and at finite omega.  This is
    the analytic statement of the ordering convention: the kernel's own
    orientation is the (-D)-pole object and receives gamma.  Swapping gamma
    onto the partner returns chi_AB(-i w), whose odd part has the opposite sign.
    """
    bands = _bands(plant)
    run = _kernel(monkeypatch, bands)
    pairs = tuple((A, B) for A in left_vertices for B in right_vertices)
    blocks = run(-1.0 / (DELTA + 1j * omega), pairs, ordered=True)
    for (A, B), got in zip(pairs, blocks):
        _assert_close(got, _exact(bands, A, B, omega),
                      f"{plant} ordered chi_{A}{B}(i*{omega})",
                      _plant_scale(bands))


@pytest.mark.parametrize("plant", ("kramers", "broken"))
@pytest.mark.parametrize("left_vertices,right_vertices", ALL_CLASSES)
def test_even_kernel_is_the_even_in_omega_part(
        monkeypatch, plant, left_vertices, right_vertices):
    """G-B2.  The shipped even rule returns exactly the even-in-omega part.

    With a real weight both orientations are weighted alike, so the kernel can
    only return the part of chi_AB(i w) that is even in omega.  Pinned against
    ``[chi(i w) + chi(-i w)] / 2`` from the literal sum.
    """
    omega = 0.7
    bands = _bands(plant)
    run = _kernel(monkeypatch, bands)
    pairs = tuple((A, B) for A in left_vertices for B in right_vertices)
    blocks = run(-DELTA / (omega ** 2 + DELTA ** 2), pairs, ordered=False)
    for (A, B), got in zip(pairs, blocks):
        want = 0.5 * (_exact(bands, A, B, omega) + _exact(bands, A, B, -omega))
        _assert_close(got, want, f"{plant} even chi_{A}{B}",
                      _plant_scale(bands))


@pytest.mark.parametrize("B", (1, 2, 3))
def test_even_rule_deletes_the_charge_current_block_under_time_reversal(
        monkeypatch, B):
    """G-B3.  The control that decides the policy.

    On the Theta-closed plant the exact chi_0B(i w) is nonzero, purely odd in
    omega, and anti-Hermitian with its partner (chi_B0 = -chi_0B^H); the even
    rule returns ZERO for it.  So the ordered orientations are needed for the
    charge-current blocks at imaginary frequency on EVERY deck, not only on a
    deck whose time-reversal verdict is false — unlike the charge block, whose
    odd part does vanish under Theta.
    """
    omega = 0.7
    bands = _bands("kramers")
    run = _kernel(monkeypatch, bands)
    exact_ct = _exact(bands, 0, B, omega)
    exact_tc = _exact(bands, B, 0, omega)
    exact_ct_even = 0.5 * (exact_ct + _exact(bands, 0, B, -omega))
    scale = float(np.max(np.abs(exact_ct)))
    assert scale > 1e-8, "planted CT is zero; nothing is tested"
    assert float(np.max(np.abs(exact_ct_even))) < 1e-12 * scale
    assert float(np.max(np.abs(exact_tc + exact_ct.conj().T))) < 1e-12 * scale
    assert float(np.max(np.abs(_exact(bands, 0, B, 0.0)))) < 1e-12 * scale
    even_ct = run(-DELTA / (omega ** 2 + DELTA ** 2), ((0, B),),
                  ordered=False)[0]
    ordered_ct = run(-1.0 / (DELTA + 1j * omega), ((0, B),), ordered=True)[0]
    assert float(np.max(np.abs(even_ct))) < 1e-12 * scale
    _assert_close(ordered_ct, exact_ct, f"kramers ordered chi_0{B}",
                  _plant_scale(bands))


@pytest.mark.parametrize("left_vertices,right_vertices", ALL_CLASSES)
def test_ordered_with_real_weight_reproduces_the_even_route(
        monkeypatch, left_vertices, right_vertices):
    """G-B4.  With a real weight the two orientations coincide.

    SCOPE: agreement to roundoff, not bit identity — the even route applies one
    weight to the summed orientations and the ordered route two weights to two
    terms.  Bit identity of the PRODUCTION even path rests on its branch being
    untouched, and is measured end to end (MoS2 legs M0/M1).
    """
    bands = _bands("broken")
    run = _kernel(monkeypatch, bands)
    pairs = tuple((A, B) for A in left_vertices for B in right_vertices)
    for a, b in zip(run(-0.4, pairs, ordered=False),
                    run(-0.4, pairs, ordered=True)):
        _assert_close(b, a, "ordered(real weight) vs even",
                      _plant_scale(bands))


def test_ordered_route_refuses_without_odd_quadrature_weights():
    """The named refusal: an ordered build with no odd weights would return the
    even answer with the channel silently deleted."""
    from gw.w_isdf import _refuse_missing_odd_kernel, _ordered_node_weights

    class _Quad:
        tau = np.asarray([0.5, 1.5])
        alpha = np.asarray([0.25, 0.75])
        alpha_odd = None

    quad = _Quad()
    with pytest.raises(ValueError, match="chi0_imag_ordered_needs_odd_kernel"):
        _refuse_missing_odd_kernel(quad)
    quad.alpha_odd = np.asarray([0.1, -0.2])
    tau, gamma = _ordered_node_weights(quad, 0.0)
    np.testing.assert_allclose(gamma, -(quad.alpha - 1j * quad.alpha_odd))
    quad.alpha_odd = np.asarray([0.1])
    with pytest.raises(ValueError, match="must share one"):
        _ordered_node_weights(quad, 0.0)


def test_ordered_flag_is_refused_where_it_has_no_meaning():
    """The scalar contour route completes at -q through its own gather; the
    ordered vertex-pair completion is not a second way to spell it."""
    from gw.w_isdf import _get_chi_minimax_kernel

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
    with pytest.raises(ValueError, match="ordered_orientations"):
        _get_chi_minimax_kernel(mesh, (1, 1, 1), layout='face',
                                face_shape=(1, 4, 4, 4), vertex_pairs=None,
                                ordered_orientations=True)
    with pytest.raises(ValueError, match="ordered_orientations"):
        _get_chi_minimax_kernel(mesh, (1, 1, 1), layout='face',
                                face_shape=(1, 4, 4, 4),
                                vertex_pairs=((0, 1),), complex_contour=True,
                                ordered_orientations=True)


@pytest.mark.parametrize("trs_allowed", (True, False))
def test_packed_imaginary_roles_are_ordered_on_every_deck(monkeypatch, trs_allowed):
    """The policy G-B3 forces: the packed current blocks take the ordered
    orientations at imaginary frequency whatever the time-reversal verdict,
    because their odd-in-omega part (the continuity channel) survives time
    reversal.  A real-axis role cannot carry the odd residue and stays even."""
    from types import SimpleNamespace
    import gw.minimax_screening as ms
    from gw.screening import ScreeningRequest, photon_role_quadratures

    calls = []

    def fake_imag(quad, omega, cfg, *, print_fn=None, with_odd_kernel=False):
        calls.append(("imag", with_odd_kernel))
        return SimpleNamespace(tau=[0.0], alpha=[1.0],
                               alpha_odd=[0.5] if with_odd_kernel else None,
                               max_error=1e-9)

    def fake_real(quad, omega, cfg, *, print_fn=None):
        calls.append(("real", None))
        return SimpleNamespace(tau=[0.0], alpha=[1.0], alpha_odd=None,
                               max_error=1e-9)

    monkeypatch.setattr(ms, "build_imag_quadrature", fake_imag)
    monkeypatch.setattr(ms, "build_real_quadrature", fake_real)
    import gw.quadrature_log as qlog
    monkeypatch.setattr(qlog, "record_minimax", lambda *a, **k: None)
    config = SimpleNamespace(minimax_config=SimpleNamespace(target_error=1e-7))
    requests = [ScreeningRequest(0j, "static"),
                ScreeningRequest(0.1j, "probe0"),
                ScreeningRequest(0.2 + 0j, "probe1")]
    plan = photon_role_quadratures(
        requests, quad=SimpleNamespace(), config=config,
        sym=SimpleNamespace(trs_allowed=trs_allowed),
        print_fn=lambda *a, **k: None)
    ordered = {role: flag for role, _quad, flag, _contact in plan}
    assert ordered == {"static": False, "probe0": True, "probe1": False}
    assert ("imag", True) in calls
