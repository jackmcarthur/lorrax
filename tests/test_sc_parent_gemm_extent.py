"""SC parent rotation and screening/Sigma plans share the carrier's k extent.

Run on a 2x2 emulated CPU mesh. A shape-checking einsum replaces only the
native GEMM provider; FFT factories are stubbed during plan construction.
The real GPU contraction and FFT execution are covered by the Na P4 gate.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@pytest.mark.parametrize("layout", ["face", "axis"])
def test_rotated_parent_screening_and_static_sigma_plan_extent(monkeypatch, layout):
    from lxkit.testing import require_devices
    require_devices(4, "cpu")
    import distrib_la
    from common import fft_helpers
    from gw import wavefunction_bundle as wb
    from gw.cohsex_sigma import _make_cohsex_kernels_face
    from gw.ppm_sigma import _face_g_plan
    from gw.w_isdf import _get_chi_fractional_contour_kernel_face

    mesh = Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))
    plans = []

    def planned_einsum(mesh, *, nq, m, k, n, **kwargs):
        def gemm(a, b):
            assert a.shape == (nq, m, k)
            assert b.shape == (nq, k, n)
            return jnp.einsum("qmk,qkn->qmn", a, b)
        gemm.nq = nq
        gemm.mesh = mesh
        gemm.in_sharding_a = NamedSharding(mesh, P(None, "x", "y"))
        gemm.in_sharding_b = gemm.in_sharding_a
        plans.append(gemm)
        return gemm

    monkeypatch.setattr(distrib_la, "gemm_plan", planned_einsum)
    monkeypatch.setattr(fft_helpers, "make_flat_k_fftn",
                        lambda *args, **kwargs: lambda value: value)
    wb._FACE_ROTATE_CACHE.clear()
    # Three raw parents on four full-k rows: replicated k must not be
    # padded or replaced by the output FFT grid's extent.
    nk, npk, nb, ns, mu = 4, 3, 4, 1, 8
    rows = np.asarray([0, 1, 3])
    plan = SimpleNamespace(
        n_parent=npk, n_full=nk, nspinor=ns, n_centroid_packed=mu,
        parent_full_rows=rows, sym=object(),
        parent_rows=lambda value: value[rows])
    rng = np.random.default_rng(20260907)
    psi = rng.normal(size=(npk, nb, ns, mu)) + 1j * rng.normal(size=(npk, nb, ns, mu))
    energies = jnp.zeros((nk, nb))
    bare = wb.Wavefunctions(
        enk=energies, occ=energies,
        slices=wb.BandSlices.from_band_edges(0, 0, 2, nb, nb), layout=layout)
    parent = wb.attach_packed_parent_green_carrier(
        bare, jnp.asarray(psi), jnp.asarray(psi.transpose(0, 2, 3, 1)),
        plan=plan, mesh_xy=mesh)
    u = np.stack([np.linalg.qr(rng.normal(size=(nb, nb)))[0] for _ in range(nk)])
    rotated = wb.rotate_wavefunctions(
        parent, jnp.asarray(u), enk_active_new=energies, efermi=None, mesh_xy=mesh)
    assert rotated.green_parent.plan is plan
    assert rotated.layout == rotated.green_parent.layout == layout
    np.testing.assert_allclose(
        np.asarray(rotated.green_parent.psi_nmu),
        np.einsum("kmn,kmsr->knsr", u[rows], psi), atol=1e-12)

    green = wb.green_face_kernel_kwargs(rotated)
    sigma = wb.sigma_face_kernel_kwargs(rotated)
    assert green["face_shape"][0] == npk
    assert sigma["face_shape"][0] == nk
    assert green["k_unfold_plan"] is sigma["k_unfold_plan"] is plan
    plans.clear()
    _get_chi_fractional_contour_kernel_face(mesh, (2, 2, 1), 1, **green)
    assert len(plans) == 1
    assert plans[0].nq == npk
    plans.clear()
    _make_cohsex_kernels_face(mesh, _convolve=lambda *args: None, **sigma)
    assert plans and all(p.nq == npk for p in plans)
    ppm_plan = _face_g_plan(mesh, green["face_shape"], layout=layout)
    assert ppm_plan.nq == npk
    wb._FACE_ROTATE_CACHE.clear()
