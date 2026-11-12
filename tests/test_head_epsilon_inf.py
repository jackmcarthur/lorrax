"""head_epsilon_inf: the static-head pin and its parse gates."""
import numpy as np
import pytest

from gw.gw_config import LorraxConfig, parse_head_epsilon_inf
from gw.head_correction import pin_static_head


def _S(omegas, rng):
    """A symmetric anisotropic head with a z-dependent shape, S(0) negative definite."""
    A = rng.normal(size=(3, 3))
    base = -(A @ A.T + 3.0 * np.eye(3)) / (8.0 * np.pi)
    B = rng.normal(size=(3, 3))
    shape = 0.1 * (B + B.T)
    return np.stack([base / (1.0 - 0.3 * z) + shape * z for z in omegas]).astype(complex)


def test_pin_static_head_identities():
    rng = np.random.default_rng(7)
    z = np.array([0.0, 0.5j, 1.3j, 0.2 + 0.4j, 4.0j])
    S = _S(z, rng)
    target = np.array([[11.7, 0.4, 0.0], [0.4, 9.0, 0.2], [0.0, 0.2, 13.1]])
    pinned, M, eps0 = pin_static_head(S, z, target.reshape(-1))
    # The static row is the target exactly; the computed eps(0) is reported.
    np.testing.assert_allclose(np.eye(3) - 8 * np.pi * pinned[0].real, target, atol=1e-12)
    np.testing.assert_allclose(eps0, np.eye(3) - 8 * np.pi * S[0].real, atol=1e-12)
    np.testing.assert_allclose(pinned, np.swapaxes(pinned, -1, -2), atol=1e-14)
    # The code's own value is the identity to round-off.
    same, M1, _ = pin_static_head(S, z, eps0.reshape(-1))
    np.testing.assert_allclose(M1, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(same, S, rtol=1e-12, atol=1e-16)
    # Cubic: one scale r at every node.
    Sc = np.stack([(-1.0 / (8 * np.pi)) * (9.5 - 1.0) / (1 - 0.2 * w) * np.eye(3) for w in z])
    cubic, Mc, _ = pin_static_head(Sc, z, np.diag([11.7] * 3).reshape(-1))
    r = (11.7 - 1.0) / (9.5 - 1.0)
    np.testing.assert_allclose(cubic, r * Sc, rtol=1e-13)
    # No static node refuses.
    with pytest.raises(ValueError, match="GATE head_epsilon_inf_static_row"):
        pin_static_head(S[1:], z[1:], target.reshape(-1))


def _config(tmp_path, extra):
    path = tmp_path / "gw.in"
    path.write_text("[cohsex]\nsys_dim = 3\nhead_correction = full\nnumber_bands = 30\n"
                    "nval = 4\nncond = 4\n" + extra)
    return LorraxConfig.from_input_file(str(path), resolve_hardware=False,
                                       runtime_platform="cpu", print_fn=lambda *_a: None)


def test_head_epsilon_inf_parse_and_scope(tmp_path):
    assert parse_head_epsilon_inf("") is None
    np.testing.assert_allclose(parse_head_epsilon_inf("11.7"), np.diag([11.7] * 3).reshape(-1))
    np.testing.assert_allclose(parse_head_epsilon_inf("5, 6 7"), np.diag([5, 6, 7.0]).reshape(-1))
    for bad in ("1 2", "0.5", "1 2 3 4 5 6 7 8 9", "x"):
        with pytest.raises(ValueError, match="head_epsilon_inf"):
            parse_head_epsilon_inf(bad)
    cfg = _config(tmp_path, "compute_mode = tangential\nhead_epsilon_inf = 11.7\n")
    assert cfg.head.epsilon_inf[0] == 11.7
    assert _config(tmp_path, "compute_mode = tangential\n").head.epsilon_inf is None
    for extra in ("compute_mode = gn_ppm\n",
                  "compute_mode = tangential\nscreening_diagrams = w_bse\n",
                  "compute_mode = tangential\nwhead_0freq = 0.0\n"):
        with pytest.raises(ValueError, match="GATE head_epsilon_inf_scope"):
            _config(tmp_path, extra + "head_epsilon_inf = 11.7\n")
