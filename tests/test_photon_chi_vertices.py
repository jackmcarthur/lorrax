"""Physical ψ†Γψ vertices at the conjugated Green-pair boundary."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def test_current_chi_matches_literal_hermitian_density(monkeypatch):
    """The Lehmann pair uses ψ†Γψ at each endpoint, including imaginary Γ2."""
    import common.fft_helpers as fft_helpers
    import distrib_la
    from common.gamma_matrices import gamma_perm_phase
    from gw.w_isdf import _get_chi_minimax_kernel, MinimaxNodes

    monkeypatch.setattr(fft_helpers, "make_flat_k_fftn", lambda *a, **k: lambda x: x)
    def local_gemm_plan(mesh, **kwargs):
        gemm = lambda x, y: x @ y
        gemm.mesh = mesh
        return gemm

    monkeypatch.setattr(distrib_la, "gemm_plan", local_gemm_plan)
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x', 'y'))
    rng = np.random.default_rng(84)
    left = rng.normal(size=(4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4))
    right = rng.normal(size=(4, 4, 6)) + 1j * rng.normal(size=(4, 4, 6))
    put = lambda x, spec: jax.device_put(jnp.asarray(x), NamedSharding(mesh, spec))
    mun = put(left.transpose(1, 2, 0)[None], P(None, None, 'x', 'y'))
    nmu = put(right[None], P(None, 'x', None, 'y'))
    nodes = MinimaxNodes(t=jnp.asarray([0.], jnp.complex128),
                         alpha=jnp.asarray([-1.], jnp.complex128))
    energies = jnp.asarray([[-1., 1., 1., 1.]])
    kernels = []
    for A, B in ((0, 2), (2, 0), (1, 2), (2, 3)):
        matrices = []
        operands = []
        for vertex in (A, B):
            perm, phase = gamma_perm_phase(vertex)
            operands.extend((perm, jnp.conj(phase)))
            matrices.append(np.asarray(phase)[:, None] * np.eye(4)[np.asarray(perm)])
        expected = np.zeros((4, 6), dtype=complex)
        for occupied, empty in ((0, 1), (0, 2), (0, 3)):
            lvc = np.einsum('am,ab,bm->m', left[occupied].conj(), matrices[0], left[empty])
            rvc = np.einsum('am,ab,bm->m', right[occupied].conj(), matrices[1], right[empty])
            pair = lvc[:, None] * rvc.conj()[None, :]
            expected -= pair + pair.conj()
        kernel = _get_chi_minimax_kernel(mesh, (1, 1, 1), layout='face',
            face_shape=(1, 4, 4, 4), right_face_shape=(1, 4, 6, 4), vertex_pair=(A, B))
        kernels.append(kernel)
        result = kernel(nodes, mun, nmu, energies < 0, energies > 0, energies,
                        jnp.asarray(-1.), jnp.asarray(1.), tuple(operands))
        np.testing.assert_allclose(np.asarray(result)[0], expected, rtol=2e-13, atol=2e-13)
    assert kernels[2] is kernels[3]
