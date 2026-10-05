"""Mode 7's run-time split-arm scratch is priced (CPU, seconds).

The Σ k-convolution (mathdx mode 7) takes a split arm when one block cannot
hold a spin group of two or more columns, and that arm draws up to 1 GiB of
XLA scratch that the compiled figure does not see.  ``ffi.fft.
klead_unfold_scratch_bytes`` mirrors the handler's rule; ``sigma_spin_block``
prices it, and the Σ τ window's checks (``SynthesisTau``) add it to the native
workspace.  A100 opt-in shared memory: 166912 B.
"""
from types import SimpleNamespace

OPTIN = 166912
GIB = 1 << 30


def test_split_arm_rule_and_bound():
    from ffi.fft import klead_unfold_scratch_bytes as scratch

    assert scratch((4, 4, 4), 2, 10**6, optin=OPTIN) == 0       # single arm
    assert scratch((12, 12, 12), 2, 10**6, optin=OPTIN) == 0    # one group of 4 columns fits
    assert scratch((12, 12, 12), 1, 10**6, optin=OPTIN) == 0    # four scalar columns fit
    per_pair = 20 ** 3 * 4 * 16
    assert scratch((20, 20, 20), 2, 10, optin=OPTIN) == 10 * per_pair
    assert scratch((20, 20, 20), 2, 10**6, optin=OPTIN) == (GIB // per_pair) * per_pair
    assert scratch((14, 14, 14), 1, 10**6, optin=OPTIN) > 0     # one scalar column per block
    assert scratch((14, 14, 14), 2, 1, optin=OPTIN) == 14 ** 3 * 4 * 16


def test_sigma_price_and_window_checks_carry_the_scratch(monkeypatch):
    from ffi import fft as F
    from gw.greens_function_kernel import sigma_spin_block
    from gw.mpa.sigma import SynthesisTau

    monkeypatch.setattr(F, "kconv_backend", lambda mesh: "mathdx")
    monkeypatch.setattr(F, "_optin_smem_bytes", lambda ordinal=0: OPTIN)
    mesh = SimpleNamespace(shape={"x": 8, "y": 8})
    plan = {}
    sigma_spin_block(n_parent=4, n_rmu=1024, ns=2, n_full=8000, n_band=40, mesh=mesh,
                     partner_tiles=1, kgrid=(20, 20, 20), plan=plan)
    assert plan["scratch"] == (GIB // (8000 * 4 * 16)) * 8000 * 4 * 16
    plan_small = {}
    sigma_spin_block(n_parent=4, n_rmu=1024, ns=2, n_full=64, n_band=40, mesh=mesh,
                     partner_tiles=1, kgrid=(4, 4, 4), plan=plan_small)
    assert plan_small["scratch"] == 0
    assert plan["new"] - plan["scratch"] > 0

    spatial = lambda *a: None
    spatial.price = dict(plan)
    tau = SynthesisTau(spatial, None, None, None, 1000, "toy", None, "key", ())
    assert tau._native == 1000 + plan["scratch"]
