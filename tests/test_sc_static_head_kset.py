"""The SC map's static head terms leave its k-set seam on the loop k-set (CPU 2x2 mesh).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
On a symmetry-reduced grid the map builds the static head terms on the full BZ
(Sigma needs them there) and its one output seam
(``_sc_output_tables_on_loop_kset``) moves Sigma and U to the star wedge.  The
terms must move with them: the final diagnostics under
``sigma_freq_debug_output`` rotate them with that U, and at
``sc_max_iter = 1`` that rotation is the first consumer to meet the k axes.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

jax.config.update("jax_enable_x64", True)


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def test_one_map_static_head_terms_rotate_on_the_loop_kset():
    from symmetry_maps import KStarMap
    from gw import sc_iteration as sc
    from gw.head_correction import compute_static_head_terms
    from gw.sigma_dispatch import SigmaResult

    mesh = _mesh()
    # k = 0, 1/3, 2/3 on a 3-point line: 1/3 and 2/3 are time-reversal
    # partners, so the loop k-set (star wedge) has two rows.
    ks = KStarMap([0, 1, 1], [0, 0, 1], 1)
    nb = 4
    rng = np.random.default_rng(3)
    a = rng.normal(size=(ks.nk_irr, nb, nb)) + 1j * rng.normal(size=(ks.nk_irr, nb, nb))
    U_loop_in = np.linalg.qr(a)[0]
    U_full = ks.broadcast(U_loop_in)
    # Star-invariant fractional occupations, different between the stars.
    occ_loop = np.array([[1.0, 1.0, 0.5, 0.0], [1.0, 0.8, 0.1, 0.0]])
    occ_full = np.asarray(ks.broadcast(occ_loop)).real
    head_full = compute_static_head_terms(
        vc0=2.0, wcoul0_static=0.5, occ=occ_full, cell_volume=10.0,
        nk_tot=ks.nk_full, source="test")
    tables = jnp.asarray(rng.normal(size=(ks.nk_full, nb, nb)) + 0j)
    sigma_full = SigmaResult(
        v_h_kij_ry=tables, sigma_x_kij_ry=tables, sigma_xc_kij_ry=tables)

    _, U, _, _, _, head = sc._sc_output_tables_on_loop_kset(
        sigma_full, tables, None, U_full, None, head_full, ks)
    # run_sc_driver's final diagnostics, as the debug flag runs them.
    U = sc._place(U, mesh, sc._band_rotation_spec())
    head_dft = sc._rotate_static_head_terms_to_dft(
        head, U, mesh=mesh, weight_dft_qp=jnp.real(U * jnp.conj(U)))

    weight = np.abs(U_loop_in) ** 2
    for name in ("sigma_x_diag", "sigma_sx_diag", "sigma_sx_minus_x_diag",
                 "sigma_coh_diag"):
        want = np.einsum("kmn,kn->km", weight,
                         np.asarray(getattr(head_full, name))[ks.rows])
        np.testing.assert_allclose(np.asarray(getattr(head_dft, name)), want,
                                   rtol=0, atol=1e-14)

    # A k-independent (nb,) seed head passes the seam unchanged.
    seed = compute_static_head_terms(
        vc0=2.0, wcoul0_static=0.5, occ=occ_loop[0], cell_volume=10.0,
        nk_tot=ks.nk_full, source="seed")
    *_, seed_out = sc._sc_output_tables_on_loop_kset(
        sigma_full, tables, None, U_full, None, seed, ks)
    assert seed_out.sigma_x_diag is seed.sigma_x_diag
