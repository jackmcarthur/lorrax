"""``qsgw_utils.sigma_grid_edge_ambiguity`` -- the Sigma(0) out-of-grid rule's
discontinuity makes the SC fixed point path-dependent within the jump.

Synthetic diagonal Sigma_c(omega) on a [-12, +10] eV grid, built so the
top-edge jump Sigma(0) - Sigma(+10) is +1.87 eV (the measured Fe 4^3
bispinor H-point value, sandbox claim 2688) and the bottom-edge jump is
-0.5 eV.  No device, no mesh.
"""
import numpy as np

from gw.qsgw_utils import sigma_grid_edge_ambiguity


def _grid_and_sigma(nk=2, nb=6):
    omega = np.arange(-12.0, 10.0 + 1e-9, 0.25)
    # Sigma_c(omega) = a + s*omega: Sigma(0) = a, Sigma(+10) = a + 10 s,
    # Sigma(-12) = a - 12 s.  s = -0.187 gives jumps +1.87 (top), -2.244
    # (bottom); override the bottom sample for a -0.5 eV bottom jump.
    a, s = 0.3, -0.187
    sig = (a + s * omega)[:, None, None] * np.ones((1, nk, nb))
    sig[0] = a + 0.5          # Sigma(0) - Sigma(-12) = -0.5 eV
    return omega, sig


def test_both_branches_of_the_measured_fe_case_are_flagged():
    omega, sig = _grid_and_sigma()
    e = np.array([[9.83, 11.70, 12.50, 5.00, -11.80, -12.70],
                  [0.00, 7.90, 10.00, 11.86, 11.88, -13.00]])
    amb, jump = sigma_grid_edge_ambiguity(sig, omega, e)
    np.testing.assert_allclose(jump[0, :4], 1.87, atol=1e-9)
    # +9.83 (inside by 0.17) and +11.70 (outside by 1.70): the two measured
    # branches, both within the 1.87 eV jump.
    assert amb[0, 0] and amb[0, 1]
    # 2.50 eV beyond the edge is beyond the jump; +5 is far inside.
    assert not amb[0, 2] and not amb[0, 3]
    # Bottom edge, jump -0.5 eV: 0.2 inside and 0.7 outside.
    assert amb[0, 4] and not amb[0, 5]
    np.testing.assert_allclose(jump[0, 4:], -0.5, atol=1e-9)
    # The band is strict and centred on the edge: +10.00 (on it) and
    # +11.86 (1.86 < 1.87) are in, +11.88 is out; +7.90 (2.10 inside) is out.
    assert amb[1, 2] and amb[1, 3] and not amb[1, 4]
    assert not amb[1, 0] and not amb[1, 1] and not amb[1, 5]


def test_an_inward_jump_leaves_the_inside_state_unique():
    omega = np.arange(-12.0, 10.0 + 1e-9, 0.25)
    sig = np.zeros((omega.size, 1, 2))
    sig[-1] = 1.0                                   # Sigma(0) - Sigma(+10) = -1
    amb, jump = sigma_grid_edge_ambiguity(sig, omega, np.array([[9.5, 10.5]]))
    np.testing.assert_allclose(jump, -1.0, atol=1e-12)
    assert not amb.any()


def test_continuous_sigma_across_the_edge_flags_nothing():
    omega = np.arange(-12.0, 10.0 + 1e-9, 0.25)
    sig = np.full((omega.size, 1, 3), 0.7)       # Sigma(0) == Sigma(edges)
    e = np.array([[9.99, 10.01, -12.01]])
    amb, jump = sigma_grid_edge_ambiguity(sig, omega, e)
    assert not amb.any()
    np.testing.assert_allclose(jump, 0.0, atol=1e-12)


def test_the_verdict_says_not_unique_and_never_hides_it():
    from gw.sc_iteration import ConvergenceVerdict
    v = ConvergenceVerdict(True, 4e-4, 1e-4, 1e-4, 26, 26, 3, 14, 5e-4,
                           edge_ambiguous=3,
                           edge_ambiguous_detail="largest jump +1.870 eV")
    text = v.summary()
    assert "CONVERGED" in text and "fixed point NOT UNIQUE: 3 state(s)" in text
    plain = ConvergenceVerdict(True, 4e-4, 1e-4, 1e-4, 26, 26, 3, 14, 5e-4)
    assert "NOT UNIQUE" not in plain.summary()
