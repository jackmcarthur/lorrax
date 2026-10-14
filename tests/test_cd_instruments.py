"""CPU pins of two CD instruments in gw.cd.sigma: the Sigma_x identity gate and the leak ledger.

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE: unit and CPU cells; no device logic here).
What is pinned: ``GATE cd_sigma_x_identity`` admits 0.9e-3 meV and refuses 1.1e-3 meV and NaN (design
target 1e-3 meV); ``ledger_sample`` reads the pool high-water WITHOUT a reset, so it no longer truncates
the per-stage device peak that ``common.timing`` charges at section boundaries.
"""
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(r'''
    import numpy as np, jax
    jax.config.update("jax_enable_x64", True)
    import runtime.xla_memory as xm
    from gw.cd import sigma as cds
    calls = []
    def fake_high_water(*, reset=False):
        calls.append(reset)
        return 123456789
    xm.pool_high_water = fake_high_water
    row = cds.ledger_sample("real", 3, 0.5)
    row2 = cds.ledger_sample("real", 4, 0.5)
    assert calls == [False, False], calls
    assert row["pool_high_water"] == 123456789 and row2["chunk"] == 4, (row, row2)
    assert cds.SIGMA_X_IDENTITY_TOL_MEV == 1e-3
    per_mev = 1.0 / (float(cds.RYD_TO_EV) * 1e3)
    ref = np.array([[-1.25, -0.5, 0.25]]) + 0j
    dmax, note = cds.sigma_x_identity_check(ref + 0.9e-3 * per_mev, ref)
    assert abs(dmax - 0.9e-3) < 1e-9 and "meV" in note, (dmax, note)
    for bad in (1.1e-3, float("nan")):
        try:
            cds.sigma_x_identity_check(ref + bad * per_mev, ref)
        except ValueError as exc:
            assert "GATE cd_sigma_x_identity" in str(exc) and "<= 0.001 meV" in str(exc), str(exc)
        else:
            raise AssertionError(f"{bad} meV passed GATE cd_sigma_x_identity")
    print("CDINSTRUMENTS ok")
''')


def test_cd_sigma_x_identity_gate_and_ledger_no_reset():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0 and "CDINSTRUMENTS ok" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]
