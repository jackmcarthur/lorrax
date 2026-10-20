"""Magnus stability probes distinguish a physical HF quench from seed growth."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh

from tdagw import propagate as tp
from tdagw.self_energy import SelfEnergyModel

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("n_trials", [1, 3])
@pytest.mark.parametrize("method,frame", [("cfm4", "interaction"),
                                         ("magnus2", "whole_rho")])
def test_quench_probe_tracks_reference_trajectory(n_trials, method, frame):
    mesh = Mesh(np.asarray(jax.devices()).reshape(1, -1), ("x", "y"))
    ops, nk_pad = tp.place_operators(dict(
        e_qp=np.array([[0.0, 1.0]]), f_kn=np.array([[1.0, 0.0]]),
        r_inter=np.zeros((3, 1, 2, 2), complex), gamma=np.zeros((2, 2)),
        w_k=np.ones(1), v0_inter=np.zeros((3, 1, 2, 2), complex),
        v0_intra=None, intra=None), mesh)
    baseline = np.zeros((n_trials, nk_pad, 2, 2), complex)
    baseline[:, 0] = np.array([[0.0, 0.01], [0.01, 0.0]])
    baseline = tp.place_state(baseline, mesh)

    def apply(params, rho, t):
        del t
        # .1*rho commutes with rho, so the exact density motion is unitary
        # under the constant Hamiltonian, while its coupling radius is .1.
        return params + 0.1 * rho

    model = SelfEnergyModel(apply, baseline, True, "absolute HF quench")
    cfg = tp.PropagatorConfig(method, 0.0, 1.0, 1.0, 0.001, 2,
                             1e-9, 1e-9, np.inf, 1, False, frame=frame)
    sigma, _ = tp.coupling_radius(None, ops, model,
        lambda t: jnp.zeros((n_trials, 3)), mesh, t0=0.0, n_iter=4,
        n_trials=n_trials)
    np.testing.assert_allclose(sigma, 0.1, rtol=1e-10)
    original, _ = tp.step_growth(cfg, ops, model, mesh, h=0.001, t0=0.0,
                                 n_trials=n_trials)
    assert original > 1.0 + tp.PROBE_TOLERANCE
    growth, norms = tp.step_growth(cfg, ops, model, mesh, h=0.001, t0=0.0,
        tracking_reference=True, n_trials=n_trials)
    np.testing.assert_allclose(growth, 1.0, atol=2e-6)
    np.testing.assert_allclose(norms[-1], norms[23], rtol=5e-5)
