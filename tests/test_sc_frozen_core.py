"""sc_frozen_core_bands: the lowest N bands keep their DFT Hamiltonian block."""
import numpy as np
import jax.numpy as jnp
import pytest

from gw.sc_iteration import _freeze_core_block_to_dft


def test_frozen_block_is_dft_diagonal_and_decoupled():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 5, 5)) + 1j * rng.normal(size=(2, 5, 5))
    H = jnp.asarray(a + np.conj(np.swapaxes(a, 1, 2)))
    e_dft = jnp.asarray(rng.normal(size=(2, 4)))      # logical 4 < carrier 5
    out = np.asarray(_freeze_core_block_to_dft(H, e_dft, 2))
    for k in range(2):
        np.testing.assert_array_equal(out[k, :2, :2],
                                      np.diag(np.asarray(e_dft)[k, :2]))
        assert not out[k, :2, 2:].any() and not out[k, 2:, :2].any()
        np.testing.assert_array_equal(out[k, 2:, 2:], np.asarray(H)[k, 2:, 2:])
        w = np.linalg.eigvalsh(out[k])
        for e in np.asarray(e_dft)[k, :2]:
            assert np.min(np.abs(w - e)) < 1e-12


def test_frozen_core_config(tmp_path):
    from test_qp_solver_config import _config
    assert _config(tmp_path).sc.frozen_core_bands == 0
    assert _config(tmp_path, "sc_frozen_core_bands = 88\n").sc.frozen_core_bands == 88
    with pytest.raises(ValueError, match="sc_frozen_core_bands"):
        _config(tmp_path, "sc_frozen_core_bands = -1\n")
