"""Mode 7's run-time split-arm scratch is priced (CPU, seconds).

The Σ k-convolution (mathdx mode 7) takes a split arm when one block cannot
hold a spin group of two or more columns, and that arm draws up to 1 GiB of
XLA scratch that the compiled figure does not see.  ``ffi.fft.
klead_unfold_scratch_bytes`` mirrors the handler's rule, including the m7s passes'
pair-major W_R (``16·N_k`` B a pair at ``n_s = 2`` on arm 2); ``sigma_spin_block``
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
    # n_s = 2 on arm 2 is m7s: the intermediate plus the chunk's pair-major W_R
    per_pair = 20 ** 3 * 4 * 16 + 20 ** 3 * 16
    assert scratch((20, 20, 20), 2, 10, optin=OPTIN) == 10 * per_pair
    assert scratch((20, 20, 20), 2, 10**6, optin=OPTIN) == (GIB // per_pair) * per_pair
    assert scratch((14, 14, 14), 2, 1, optin=OPTIN) == 14 ** 3 * 4 * 16 + 14 ** 3 * 16
    # n_s = 1 on arm 2 is the plain arm 2 (no m7s at n_s = 1): the intermediate only
    assert scratch((14, 14, 14), 1, 10, optin=OPTIN) == 10 * 14 ** 3 * 16
    # arm 1 (one padded column exceeds a block, 22^3 on an A100): the intermediate only
    assert scratch((22, 22, 22), 2, 10, optin=OPTIN) == 10 * 22 ** 3 * 4 * 16


def test_sigma_price_and_window_checks_carry_the_scratch(monkeypatch):
    from ffi import fft as F
    from gw.greens_function_kernel import sigma_spin_block
    from gw.mpa.sigma import SynthesisTau

    from common import fft_helpers
    # The Green kernel reaches the router through common.fft_helpers.
    monkeypatch.setattr(fft_helpers, "kconv_backend", lambda mesh, *a, **k: "mathdx")
    monkeypatch.setattr(F, "_optin_smem_bytes", lambda ordinal=0: OPTIN)
    mesh = SimpleNamespace(shape={"x": 8, "y": 8})
    plan = {}
    sigma_spin_block(n_parent=4, n_rmu=1024, ns=2, n_full=8000, n_band=40, mesh=mesh,
                     partner_tiles=1, kgrid=(20, 20, 20), plan=plan)
    per_pair = 8000 * 4 * 16 + 8000 * 16                         # m7s at 20^3, n_s = 2
    assert plan["scratch"] == (GIB // per_pair) * per_pair
    plan_small = {}
    sigma_spin_block(n_parent=4, n_rmu=1024, ns=2, n_full=64, n_band=40, mesh=mesh,
                     partner_tiles=1, kgrid=(4, 4, 4), plan=plan_small)
    assert plan_small["scratch"] == 0
    assert plan["new"] - plan["scratch"] > 0

    spatial = lambda *a: None
    spatial.price = dict(plan)
    tau = SynthesisTau(spatial, None, None, None, 1000, "toy", None, "key", ())
    assert tau._native == 1000 + plan["scratch"]
