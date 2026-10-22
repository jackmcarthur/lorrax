"""The TD refit's chi0 in four (hole x electron) blocks, on the synthetic GW bundle (CPU, 2x2 mesh).

``gw.response_bank`` with ``meta.tangential_response_blocks`` (``response_blocks``,
``_block_quadrature``, ``_integrate_block_passes``) against the exact ordered-pair scan
``gw.w_isdf.compute_chi0_direct_fractional`` at complex z down to Im z = 0.1 eV, with random
fractional populations on the active bands; the one-rule path on the same carrier as the
convention check; the blocks sum to all pairs; the gates.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python -m pytest -q tests/test_response_bank_blocks_cpu.py
"""
import os
import sys
from types import SimpleNamespace

import runtime  # noqa: F401  (before numpy: the repo's bootstrap)
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_tdagw_dynsig_sigma as FX  # noqa: E402  (the synthetic GW bundle)

RY = 13.605693122994
#: bands 0, 1 frozen occupied; A = [2, 7) the active bands; 7, 8 frozen empty (NBF = 9).
#: A spans at most 0.5 Ry: at Im z = 0.1 eV the AA closed form needs n ~ 480 > 384 nodes
#: (two panels), with an affordable host pencil (~1300 samples) tried first.
ACTIVE = (2, 7)
Z_EV = (0.1j, 0.5j, 2.0j, 1.0 + 1.0j, 3.0 + 0.5j)
TOL = 1e-8


def _energies(rng, nk):
    e = np.empty((nk, FX.NBF))
    for k in range(nk):
        e[k, 0:2] = np.sort(rng.uniform(-1.4, -1.0, 2))
        e[k, 2:5] = np.sort(rng.uniform(-0.25, -0.05, 3))
        e[k, 5:7] = np.sort(rng.uniform(0.05, 0.25, 2))
        e[k, 7:9] = np.sort(rng.uniform(0.9, 1.4, 2))
    return e


def _occupations(rng, nk, fractional):
    f = np.zeros((nk, FX.NBF))
    f[:, :5] = 1.0
    if fractional:
        f[:, 2:7] = np.clip(f[:, 2:7] + rng.uniform(-0.3, 0.3, (nk, 5)) * np.array([1, 1, 1, 0, 0])
                            + rng.uniform(0.0, 0.3, (nk, 5)) * np.array([0, 0, 0, 1, 1]), 0.0, 1.0)
    return f


def _mesh():
    import jax
    from jax.sharding import Mesh
    devs = np.asarray(jax.devices())
    if devs.size < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(devs[:4].reshape(2, 2), ("x", "y"))


def system(mesh, seed=20261019, fractional=True):
    """(wfns, meta, kmq, f) of the synthetic bundle with gapped F | A | F energies."""
    rng = np.random.default_rng(seed)
    nk = int(np.prod(FX.KGRID))
    psi = 0.6 * (rng.standard_normal((nk, FX.NBF, FX.NS, FX.NMU))
                 + 1j * rng.standard_normal((nk, FX.NBF, FX.NS, FX.NMU)))
    e = _energies(rng, nk)
    f = _occupations(rng, nk, fractional)
    kf = np.stack(np.unravel_index(np.arange(nk), FX.KGRID), axis=1) / np.asarray(FX.KGRID, float)
    kplan, perm = FX._plan(mesh, kf, rng)
    psi = psi[..., perm]
    wfns, meta = FX._gw_bundle(mesh, psi, e, f, kplan)
    meta.mu_basis = SimpleNamespace(n_packed=FX.NMU, n_logical=FX.NMU)
    if not hasattr(meta, "b_id_4_chi_user"):
        meta.b_id_4_chi_user = FX.NBF
    return wfns, meta, FX._kmq(), f


def sample_plan(z_ry, mu_ry, tol=TOL):
    n = len(z_ry)
    return dict(z_ry=np.asarray(z_ry, np.complex128), distinct_id=np.arange(n), held=np.zeros(n, bool),
                role=np.zeros(n, np.int64), census=dict(mu_ry=float(mu_ry), partial_at_mu=False),
                bank_rule_tolerance=tol)


def bank_chi(wfns, meta, mesh, z_ry, *, blocks=None):
    """The bank's value rows per sample, [n_z, n_q, mu, nu], the rules and the receipt (one stream program)."""
    import jax
    from gw import response_bank as RB
    if blocks is not None:
        meta.tangential_response_blocks = dict(active=tuple(blocks), held={})
    elif hasattr(meta, "tangential_response_blocks"):
        del meta.tangential_response_blocks
    plan = sample_plan(z_ry, 0.0)
    receipt = dict(correlation_count=0)
    say = lambda *a, **k: None
    support = RB.response_support(wfns, meta, plan, receipt, print_fn=say)
    rules = RB.response_quadrature(meta, plan, receipt, support, group_size=len(z_ry), print_fn=say)
    execute = lambda kernel, args, name, runtime_bytes=0: kernel(*args)
    nq = int(meta.nk_tot)
    out = np.zeros((len(z_ry), nq, FX.NMU, FX.NMU), np.complex128)
    for group in rules["groups"]:
        raw = RB.integrate_response_group(wfns, meta, mesh, rules, group, q_ids=list(range(nq)),
                                          execute=execute, receipt=receipt, ordered=True)
        raw = np.asarray(jax.device_get(raw))
        for slot, m in enumerate(group["members"]):
            out[int(m)] = raw[2 * slot]
    return out, rules, receipt


def exact_chi(wfns, meta, mesh, z_ry, f, kmq):
    from gw.w_isdf import compute_chi0_direct_fractional
    nb_pad = int(wfns.slices.nb_full)
    f_p = np.zeros((f.shape[0], nb_pad)); f_p[:, :f.shape[1]] = f
    v = compute_chi0_direct_fractional(wfns, np.asarray(z_ry), meta, mesh,
                                       occupation_state=SimpleNamespace(f_kn=f_p), kminq_rows=kmq,
                                       ordered=True)
    v = np.asarray(v)
    return v if len(z_ry) > 1 else v[None]


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _relative_per_z(a, b):
    return [float(np.max(np.abs(x - y)) / np.max(np.abs(y))) for x, y in zip(a, b)]


def test_four_blocks_match_the_exact_scan():
    """The one-rule bank at a step state, then four blocks at fractional active populations with z
    down to 0.1 eV, against the exact ordered-pair scan."""
    from ffi import fft as F
    mesh = _mesh()

    # the one-rule bank at a step state equals the exact scan (fixes the comparison's convention)
    wfns, meta, kmq, f = system(mesh, fractional=False)
    z = np.asarray(Z_EV[1:]) / RY                 # the one-rule path builds these
    with F.xla_reference():
        bank, _, _ = bank_chi(wfns, meta, mesh, z)
        exact = exact_chi(wfns, meta, mesh, z, f, kmq)
    rel = _relative_per_z(bank, exact)
    print("one-rule vs exact (step):", ["%.2e" % r for r in rel])
    assert max(rel) <= 50 * TOL

    # fractional active populations, z down to 0.1 eV: four blocks vs the exact ordered-pair scan
    wfns, meta, kmq, f = system(mesh, fractional=True)
    z = np.asarray(Z_EV) / RY
    with F.xla_reference():
        bank, rules, receipt = bank_chi(wfns, meta, mesh, z, blocks=ACTIVE)
        exact = exact_chi(wfns, meta, mesh, z, f, kmq)
    rel = _relative_per_z(bank, exact)
    blocks = receipt["rule"]["blocks"]
    print("blocks:", [(b["name"], ["%.4g" % (x * RY) for x in b["interval_ry"]], b["nodes"]) for b in blocks])
    print("four blocks vs exact:", ["%.2e" % r for r in rel], "nodes", receipt["nodes"])
    names = [b["name"] for b in blocks]
    assert names == ["FF", "FA", "AF", "AA"]
    lo = {b["name"]: b["interval_ry"][0] for b in blocks}
    assert lo["FF"] > 0 and lo["FA"] > 0 and lo["AF"] > 0 and lo["AA"] < 0
    aa = next(b for b in blocks if b["name"] == "AA")
    assert any(g.get("panel") for g in aa["groups"])          # the 0.1 eV sample took panels
    assert max(rel) <= 50 * TOL
    # held: a second quadrature on the same spec reuses every block's rules
    from gw import response_bank as RB
    receipt2 = dict(correlation_count=0)
    plan = sample_plan(z, 0.0)
    support = RB.response_support(wfns, meta, plan, receipt2, print_fn=lambda *a, **k: None)
    RB.response_quadrature(meta, plan, receipt2, support, group_size=len(z), print_fn=lambda *a, **k: None)
    assert receipt2["rule"]["reused"] is True

