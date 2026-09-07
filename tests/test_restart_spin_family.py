"""The BSE time-reversal carrier preserves every supported restart family."""
import numpy as np
import pytest


@pytest.mark.parametrize("ns", [1, 2, 4])
def test_charge_family_trs_preserves_span_and_pairs(ns):
    """Time reversal squares to +1 for scalar and -1 for both fermion families."""
    from bse.bse_w_exact import _trs_fix_band_array

    rng = np.random.default_rng(617)
    if ns == 1:
        rotation = np.eye(1)
        basis, _ = np.linalg.qr(rng.normal(size=(4, 2)))
        pair = basis.T.reshape(2, ns, 4).astype(complex)
    else:
        rotation = np.kron(np.eye(ns // 2), [[0., 1.], [-1., 0.]])
        state = rng.normal(size=(ns, 4)) + 1j * rng.normal(size=(ns, 4))
        state /= np.linalg.norm(state)
        pair = np.stack([state, rotation @ state.conj()])
    psi = np.tile(pair[None], (3, 1, 1, 1))
    mixing, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    psi = np.einsum("ab,kbsm->kasm", mixing, psi)
    energies = np.zeros((3, 2))
    fixed, fixed_energies = _trs_fix_band_array(psi, energies, (3, 1, 1), label="charge")
    np.testing.assert_array_equal(fixed_energies, energies)
    np.testing.assert_allclose(fixed[2], np.einsum("st,btm->bsm", rotation, fixed[1].conj()), atol=1e-12)
    for before, after in zip(psi, fixed):
        before, after = before.reshape(2, -1), after.reshape(2, -1)
        np.testing.assert_allclose(after.conj().T @ after, before.conj().T @ before, atol=1e-12)
    broken = energies.copy()
    broken[2] += 1e-3
    with pytest.raises(ValueError, match="trs_gauge_energies_disagree"):
        _trs_fix_band_array(psi, broken, (3, 1, 1), label="charge")
