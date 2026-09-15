"""The time-reversal-odd (ordered-orientation) vertex-pair kernel.

CT is the magnetisation-odd channel of the four-current response.  The even
vertex-pair rule weights BOTH particle-hole orientations with one real alpha and
therefore deletes that channel; the ordered rule weights the kernel's own
(-Delta-pole) orientation with gamma and its partner with conj(gamma).  See
``docs/dev/notes/DERIVATION_gnppm_nonhermitian.md`` sections 1-2, whose scalar
form these tests are the vertex-pair twin of.

Every case here is PLANTED: the expected value is a literal Lehmann sum written
out in the test, not a second call into the code under test.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _planted_kernel(monkeypatch, *, ordered, real_wavefunctions=False,
                    seed=84):
    """One tiny four-spinor pair at a single k, with the FFT and the GEMM
    replaced by identities so the R-space contraction is readable.

    Returns ``(run, left, right)`` where ``run(gamma, pairs)`` evaluates the
    kernel for one family class at one node.
    """
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
    rng = np.random.default_rng(seed)
    if real_wavefunctions:
        # The Theta-symmetric plant: real psi at a TRIM.  The partner
        # orientation then equals the kernel's own, so the odd bracket of
        # DERIVATION section 1 is identically zero and no choice of beta can
        # make it move.
        left = rng.normal(size=(4, 4, 4)) + 0j
        right = rng.normal(size=(4, 4, 6)) + 0j
    else:
        left = rng.normal(size=(4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4))
        right = rng.normal(size=(4, 4, 6)) + 1j * rng.normal(size=(4, 4, 6))
    put = lambda x, spec: jax.device_put(jnp.asarray(x), NamedSharding(mesh, spec))
    mun = put(left.transpose(1, 2, 0)[None], P(None, None, 'x', 'y'))
    nmu = put(right[None], P(None, 'x', None, 'y'))
    energies = jnp.asarray([[-1., 1., 1., 1.]])

    def run(gamma, pairs, operands):
        nodes = MinimaxNodes(t=jnp.asarray([0.], jnp.complex128),
                             alpha=jnp.asarray([gamma], jnp.complex128))
        kernel = _get_chi_minimax_kernel(
            mesh, (1, 1, 1), layout='face', face_shape=(1, 4, 4, 4),
            right_face_shape=(1, 4, 6, 4), vertex_pairs=pairs,
            ordered_orientations=ordered)
        return kernel(nodes, mun, nmu, energies < 0, energies > 0, energies,
                      jnp.asarray(-1.), jnp.asarray(1.), tuple(operands))

    return run, left, right


def _class_operands(pairs):
    """``(operands, gamma_matrices)`` for one family class, from the module's
    own permutation/phase tables."""
    from common.gamma_matrices import gamma_perm_phase
    operands, matrices = [], []
    for A, B in pairs:
        vertex_operands, pair_matrices = [], []
        for vertex in (A, B):
            perm, phase = gamma_perm_phase(vertex)
            vertex_operands.extend((perm, jnp.conj(phase)))
            pair_matrices.append(
                np.asarray(phase)[:, None] * np.eye(4)[np.asarray(perm)])
        operands.append(tuple(vertex_operands))
        matrices.append(pair_matrices)
    return operands, matrices


def _forward_orientation(left, right, pair_matrices):
    """The kernel's OWN orientation, written out: the -Delta-pole object of
    DERIVATION section 1 at tau = 0, summed over the planted transitions."""
    forward = np.zeros((4, 6), dtype=complex)
    for occupied, empty in ((0, 1), (0, 2), (0, 3)):
        lvc = np.einsum('am,ab,bm->m', left[occupied].conj(),
                        pair_matrices[0], left[empty])
        rvc = np.einsum('am,ab,bm->m', right[occupied].conj(),
                        pair_matrices[1], right[empty])
        forward += lvc[:, None] * rvc.conj()[None, :]
    return forward


ALL_CLASSES = tuple((left, right)
                    for left in ((0,), (1, 2, 3))
                    for right in ((0,), (1, 2, 3)))


@pytest.mark.parametrize("left_vertices,right_vertices", ALL_CLASSES)
def test_ordered_vertex_pair_matches_literal_ordered_lehmann(
        monkeypatch, left_vertices, right_vertices):
    """G-B1.  gamma on the kernel's own orientation, conj(gamma) on its partner.

    This is the whole content of the odd kernel, and it is planted: a complex
    gamma with a genuinely nonzero odd part (Im gamma) against a literal sum.
    All sixteen blocks are covered by the four family classes.
    """
    gamma = -0.7 + 0.35j
    pairs = tuple((A, B) for A in left_vertices for B in right_vertices)
    operands, matrices = _class_operands(pairs)
    run, left, right = _planted_kernel(monkeypatch, ordered=True)
    results = run(gamma, pairs, operands)
    assert len(results) == len(pairs)
    for result, pair_matrices in zip(results, matrices):
        forward = _forward_orientation(left, right, pair_matrices)
        expected = gamma * forward + np.conj(gamma) * forward.conj()
        np.testing.assert_allclose(np.asarray(result)[0], expected,
                                   rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("left_vertices,right_vertices", ALL_CLASSES)
def test_ordered_with_real_gamma_reproduces_the_even_completion(
        monkeypatch, left_vertices, right_vertices):
    """G-B2.  Real gamma makes the two node weights coincide, and the ordered
    route returns the even route's value.

    SCOPE: this is agreement to roundoff, not bit identity — the even route
    applies one weight to the summed orientations and the ordered route applies
    two weights to the two terms.  Bit identity of the PRODUCTION path is a
    different statement and rests on the even branch's arithmetic being
    untouched: ``ordered_orientations`` defaults to false and that branch of
    ``_contract_chi_vertices`` is the incumbent expression, which
    ``test_photon_chi_vertices.py`` still pins.
    """
    pairs = tuple((A, B) for A in left_vertices for B in right_vertices)
    operands, _ = _class_operands(pairs)
    even_run, _, _ = _planted_kernel(monkeypatch, ordered=False)
    odd_run, _, _ = _planted_kernel(monkeypatch, ordered=True)
    even = even_run(-1.0 + 0.0j, pairs, operands)
    ordered = odd_run(-1.0 + 0.0j, pairs, operands)
    for a, b in zip(even, ordered):
        scale = float(np.max(np.abs(np.asarray(a))))
        np.testing.assert_allclose(np.asarray(b), np.asarray(a),
                                   rtol=0.0, atol=1e-14 * max(scale, 1.0))


@pytest.mark.parametrize("left_vertices,right_vertices", ALL_CLASSES)
def test_odd_channel_vanishes_on_a_time_reversal_symmetric_plant(
        monkeypatch, left_vertices, right_vertices):
    """G-B3.  The TRS control at kernel level: with a Theta-symmetric plant the
    odd weight beta cannot move the answer.

    Under Theta the partner orientation equals the kernel's own, so the odd
    bracket of DERIVATION section 1 is zero and the response is independent of
    Im(gamma).  A sign or ordering error in the odd kernel breaks exactly this.

    SCOPE: kernel level, single k.  It tests that the ODD CHANNEL is zero where
    it must be.  It does not test the k-summed cancellation that makes the whole
    CT BLOCK vanish on a bulk time-reversal-symmetric deck; that is a deck-level
    control.
    """
    pairs = tuple((A, B) for A in left_vertices for B in right_vertices)
    operands, _ = _class_operands(pairs)
    run, _, _ = _planted_kernel(monkeypatch, ordered=True,
                                real_wavefunctions=True)
    even_only = run(-0.7 + 0.0j, pairs, operands)
    with_odd = run(-0.7 + 0.35j, pairs, operands)
    for a, b in zip(even_only, with_odd):
        scale = float(np.max(np.abs(np.asarray(a))))
        assert scale > 0.0, "planted block is identically zero; nothing tested"
        np.testing.assert_allclose(np.asarray(b), np.asarray(a),
                                   rtol=0.0, atol=1e-13 * scale)


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

    side = 1
    mesh = Mesh(np.asarray(jax.devices()[:side * side]).reshape(side, side),
                ('x', 'y'))
    with pytest.raises(ValueError, match="ordered_orientations"):
        _get_chi_minimax_kernel(mesh, (1, 1, 1), layout='face',
                                face_shape=(1, 4, 4, 4),
                                vertex_pairs=None,
                                ordered_orientations=True)
    with pytest.raises(ValueError, match="ordered_orientations"):
        _get_chi_minimax_kernel(mesh, (1, 1, 1), layout='face',
                                face_shape=(1, 4, 4, 4),
                                vertex_pairs=((0, 1),),
                                complex_contour=True,
                                ordered_orientations=True)
