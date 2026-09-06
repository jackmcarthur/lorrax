"""Small, falsifiable gates used by the mini integration runner."""
import numpy as np


EQP_ATOL_EV = 2e-5


def require_p4(runtime):
    assert runtime.process_count == runtime.n_devices == 4, (
        "mini requires four processes with one GPU each; use lx run -N 1 -G 4 -n 4")
    assert runtime.n_local_devices == 1 and runtime.mesh_shape == (2, 2), (
        "mini requires one local GPU and a square 2x2 process mesh")


def check_results(eqp, sigma, *, reference=None):
    """Require all 9 irreducible k × 3 band rows and COHSEX's additive identity."""
    assert eqp.shape == (27,) and np.isfinite(eqp).all(), eqp.shape
    assert sigma.shape == (27, 7) and np.isfinite(sigma).all(), sigma.shape
    assert len(np.unique(sigma[:, :2], axis=0)) == 27, "duplicate k/band rows"
    _, counts = np.unique(sigma[:, 0], return_counts=True)
    assert len(counts) == 9 and np.all(counts == 3), "incomplete k/band coverage"
    # The output rounds each summand to six decimals independently.
    np.testing.assert_allclose(sigma[:, 2] + sigma[:, 3], sigma[:, 4],
                               rtol=0, atol=1.6e-6)
    if reference is not None:
        ref_eqp, ref_sigma = reference
        np.testing.assert_array_equal(sigma[:, :2], ref_sigma[:, :2])
        np.testing.assert_allclose(eqp, ref_eqp, rtol=0, atol=EQP_ATOL_EV)
        np.testing.assert_allclose(sigma[:, 2:], ref_sigma[:, 2:],
                                   rtol=0, atol=EQP_ATOL_EV)
