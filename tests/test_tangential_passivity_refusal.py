"""``GATE tangential_passivity`` names its numbers (injected violation).

CPU, pytest.  The refusal used to say only "failed at q"; a Fe 8^3 SC run
died on it twice with no way to tell which bound broke.  An over-unity
whitened response (the suspected case) and a negative one go through the
real check (:func:`gw.tangential_gates._passivity_response_checks`) and the
refusal text must carry lambda_min, lambda_max, the anti-Hermitian norm, the
broken bound and the receipt.  The tolerance is unchanged.
"""

import numpy as np


def test_passivity_refusal_prints_min_max_antihermitian_and_receipt():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from gw.tangential_gates import _passivity_response_checks, passivity_refusal
    from gw.tangential_recipe import shared_real_pole_gates_v1_r3b as gates, table_hash

    def eigh(h, cut):
        del cut
        return jnp.linalg.eigvalsh(h), None

    n = 4
    rng = np.random.default_rng(3)
    u, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    over = u @ np.diag([0.1, 0.5, 0.9, 1.0 + 3e-6]) @ u.conj().T       # upper bound
    under = u @ np.diag([-2e-9, 0.2, 0.4, 0.6]) @ u.conj().T            # lower bound
    good = u @ np.diag([0.0, 0.2, 0.4, 0.6]) @ u.conj().T
    passive = _passivity_response_checks(jnp.asarray(np.stack([over, under, good])),
                                         eigh=eigh, gates=gates)
    assert np.asarray(passive["passivity"]).tolist() == [False, False, True]
    text = passivity_refusal(passive, 0, 53, gates=gates, eta_ev=0.1, ordered=False,
                             gate_hash=table_hash(gates))
    assert text.startswith("GATE tangential_passivity:") and "q=53" in text
    assert "lambda_max - 1 = 3.0" in text and "lambda_min =" in text
    assert "anti-Hermitian relative =" in text and table_hash(gates) in text
    assert all(part in text for part in ("got:", "want:", "receipt:", "why:"))
    low = passivity_refusal(passive, 1, 9, gates=gates, eta_ev=0.1, ordered=True,
                            gate_hash="h")
    assert "broken: lambda_min -2.0" in low and "ordered" in low
