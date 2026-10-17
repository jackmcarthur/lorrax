"""The SC final head rotation on a band-padded U (CPU 2x2 mesh).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
A static Sigma (COHSEX, x-only) carries no band axis, so ``run_sc_driver``'s
final rotation pads U to the mesh carrier when the band count does not divide
the mesh.  Under ``sigma_freq_debug_output`` the same U rotates the head
diagonals, which stay logical-width; the weight must come from U's logical
block.
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


def test_final_head_terms_rotate_with_a_padded_static_U():
    from runtime.padding import pad_square, padded_axis
    from symmetry_maps import KStarMap
    from gw import sc_iteration as sc
    from gw.head_correction import compute_static_head_terms
    from gw.sigma_dispatch import SigmaResult

    mesh = _mesh()
    # Star-wedge loop k-set of a 3-point line (1/3 and 2/3 are time-reversal
    # partners); five SC bands, which the 2x2 mesh's band divisor 2 pads to 6.
    ks = KStarMap([0, 1, 1], [0, 0, 1], 1)
    nb = 5
    rng = np.random.default_rng(5)
    a = rng.normal(size=(ks.nk_irr, nb, nb)) + 1j * rng.normal(size=(ks.nk_irr, nb, nb))
    U_loop_in = np.linalg.qr(a)[0]
    occ_loop = np.array([[1.0, 1.0, 1.0, 0.5, 0.0], [1.0, 1.0, 0.8, 0.1, 0.0]])
    head_full = compute_static_head_terms(
        vc0=2.0, wcoul0_static=0.5, occ=np.asarray(ks.broadcast(occ_loop)).real,
        cell_volume=10.0, nk_tot=ks.nk_full, source="test")
    tables = jnp.asarray(rng.normal(size=(ks.nk_full, nb, nb)) + 0j)
    sigma_full = SigmaResult(
        v_h_kij_ry=tables, sigma_x_kij_ry=tables, sigma_xc_kij_ry=tables)
    _, U, _, _, _, head = sc._sc_output_tables_on_loop_kset(
        sigma_full, tables, None, ks.broadcast(U_loop_in), None, head_full, ks)

    # run_sc_driver's static final rotation pad, as it builds it.
    spec = sc._band_rotation_spec()
    static_axis = padded_axis(
        nb, mesh, name="static Sigma final rotation", specs=((spec, 1), (spec, 2)))
    assert static_axis.pad == 1
    U = sc._place(pad_square(U, static_axis, pad_diagonal=1.0), mesh, spec)
    head_dft = sc._final_head_terms_to_dft(head, U, mesh=mesh, pad_axis=static_axis)

    weight = np.abs(U_loop_in) ** 2
    for name in ("sigma_x_diag", "sigma_sx_diag", "sigma_sx_minus_x_diag",
                 "sigma_coh_diag"):
        want = np.einsum("kmn,kn->km", weight,
                         np.asarray(getattr(head_full, name))[ks.rows])
        np.testing.assert_allclose(np.asarray(getattr(head_dft, name)), want,
                                   rtol=0, atol=1e-14)
