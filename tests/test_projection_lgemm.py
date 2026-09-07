"""HLO pin + parity gates for the L-GEMM f64-split projection relowering.

Regression tests for wk_REL/RESHARD_OVERHEAD_MEMO.md Sec. 4.4 (2026-07-28):
XLA:CPU used to PROMOTE the f64 channel operands of the ψ†σψ projection
einsum to c128 (a ~400 MB ``convert`` materialization per channel at
nb=128/P=64) and issue Eigen zgemm at 2× the mathematically required
flops — measured 295 GF/s where the same machine's BLAS runs the identical
contraction at 1263 GF/s.  The fix (movement-only, owner channel algebra
untouched) lowers the TWO-CHANNEL body's (``_project_ri_local``) large
right-einsums as pure f64 dgemms + a ``lax.complex`` recombine.  The
MERGED body (``_project_x_local``) deliberately keeps its single complex
chain: it has no promotion pathology (complex × complex at minimal flops)
and the f64 split measured as a REGRESSION there (job 7878942 — Eigen
dgemm ~172 GF/s is per-flop below its zgemm's 295; refutation recorded in
the body docstring, patch in wk_REL/lgemm_full_2026-07-28.patch).

These tests compile the production reduce-scatter projectors on a 2×2
emulated-device mesh (the ``test_sanity_gates_jax.py`` 4-device pattern)
and assert on the optimized HLO — the only ground truth for lowering
claims (QUALITY_PATTERNS #4):

1. NO rank≥2 f64→c128 ``convert`` exists in either projection module
   (the promotion copies are gone; in the merged module this also guards
   against a future PARTIAL split re-introducing a mixed-dtype dot);
2. two-channel: the LARGE (μ-contracting right) dots are f64 — the only
   c128 dots left are the small post-scatter left dots at their exact
   shape; merged: only the genuine complex right + left dots exist;
3. the collective contract is untouched: exactly two c128
   ``reduce-scatter`` ops per module at the exact pre-relowering payload
   shapes (stacked leading 2 for the two-channel plan, no leading 2 for
   the merged plan) — the relowering is movement-only INSIDE the rank,
   collectives byte-identical;
4. numerics: both projectors match an independent numpy reference at
   1e-12 (value-level gate — the dgemm split reorders sums vs the
   promoted zgemm, so bit-equality is deliberately NOT claimed).

Run inside the container (login python has no jax), e.g.::

    /scratch2/08271/jackmc/lorrax_setup/alloc_run.sh 1 1 \
        /work2/08271/jackmc/frontera/lorrax/src \
        /work2/08271/jackmc/frontera/lorrax \
        python -u tests/test_projection_lgemm.py

or under pytest with the venv on PATH.  One process, no GPU, seconds.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
# LORRAX_BANDS_GEMM_FFI is deliberately NOT set here — see the
# `_xla_plan_dial` fixture below (P19, the second instance of the same
# collection-time leak).  The knobs set at module scope in this file are
# only the ones jax/XLA read at IMPORT time.
#
# Four emulated host devices: the projector's shard_map/psum_scatter path
# is identical to production; multi-process collectives are covered by the
# restart-gated P=64 A/B (wk_REL/lgemm_ab.sbatch).
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        f"{_xla_flags} --xla_force_host_platform_device_count=4".strip())

import pytest                                        # noqa: E402
import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402


@pytest.fixture(autouse=True)
def _xla_plan_dial(monkeypatch):
    """Pin the GEMM dial OFF, PER TEST (was a module-scope setdefault).

    This file pins the DEFAULT XLA GEMM lowering of the projector (dot
    classes, converts).  The dial defaults ON since the FFI-required ruling
    (decisions.md 2026-08-01), which would replace the pinned dots with
    mklblas custom-calls, so it has to be off while these pins run — but at
    module scope that pin runs at COLLECTION time and never unwinds, so it
    reconfigures every other test file in the session.  See
    ``tests/test_contract_bands.py::_xla_plan_dial`` for the gate that was
    measurably propped up by exactly this leak (P19).  The FFI plan has its
    own pins in tests/test_contract_bands.py.
    """
    monkeypatch.setenv("LORRAX_BANDS_GEMM_FFI", "0")

from common.contract_bands import contract_bands_block_reshard  # noqa: E402


# Test dims (mirror wk_REL/check_channel_hermiticity.py stage p4): global
# μ=16 → μ_loc=8 per mesh axis, m=n=8 → m_loc=n_loc=4 on the 2×2 mesh.
NK, NS, MU, MN = 4, 2, 16, 8
PX = PY = 2
TOL = 1e-12

_DOT_RE = re.compile(r"=\s+(f64|c128)\[([\d,]*)\]\S*\s+dot\(")
_CONVERT_RE = re.compile(r"=\s+c128\[([\d,]*)\]\S*\s+convert\(f64\[")
_RS_RE = re.compile(r"=\s+(\w+)\[([\d,]*)\]\S*\s+reduce-scatter\(")


def _mesh():
    import pytest
    n_dev = len(jax.devices())
    if n_dev < 4:
        # skip, not assert — see test_contract_bands._mesh (the 4-device
        # leg reruns these under xla_force_host_platform_device_count=4).
        pytest.skip(
            f"needs 4 (emulated) devices, got {n_dev}; set "
            f"XLA_FLAGS=--xla_force_host_platform_device_count=4")
    return Mesh(np.asarray(jax.devices()[:4]).reshape(PX, PY), ("x", "y"))


def _inputs(mesh):
    rng = np.random.default_rng(23)

    def crand(*shape):
        return (rng.standard_normal(shape)
                + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

    psi_xr = crand(NK, MN, NS, MU)          # (nk, m, s, μ)
    sigma_k = crand(NK, NS, MU, NS, MU)     # (nk, s, μ, s', μ')
    psi_yn = crand(NK, NS, MU, MN)          # (nk, s', μ', n)
    dev = (
        jax.device_put(jnp.asarray(psi_xr),
                       NamedSharding(mesh, P(None, None, None, "x"))),
        jax.device_put(jnp.asarray(sigma_k),
                       NamedSharding(mesh, P(None, None, "x", None, "y"))),
        jax.device_put(jnp.asarray(psi_yn),
                       NamedSharding(mesh, P(None, None, "y", None))),
    )
    return (psi_xr, sigma_k, psi_yn), dev


def _compiled_text(merged_x):
    mesh = _mesh()
    proj = contract_bands_block_reshard(
        mesh, channels="none" if merged_x else "split_reim")
    _, dev = _inputs(mesh)
    return jax.jit(proj).lower(*dev).compile().as_text()


def _assert_lowering(txt, *, tag, expect_rs_shapes, allowed_c128_dots,
                     min_f64_dots):
    # (1) The promotion copies are gone: no rank≥2 f64→c128 convert
    # anywhere (fusion bodies included).  Scalars/rank-1 are permitted.
    offenders = [m.group(0) for m in _CONVERT_RE.finditer(txt)
                 if "," in m.group(1)]
    assert not offenders, (
        f"[{tag}] f64→c128 promotion convert(s) re-grew:\n"
        + "\n".join(offenders))

    # (2) Dots: only the expected genuinely-complex contractions may be
    # c128; the two-channel body's large right contractions must be f64.
    dots = _DOT_RE.findall(txt)
    assert dots, f"[{tag}] no dot ops found — parsing bug?"
    f64_dots = [s for t, s in dots if t == "f64"]
    c128_dots = [s for t, s in dots if t == "c128"]
    assert len(f64_dots) >= min_f64_dots, (
        f"[{tag}] expected ≥{min_f64_dots} f64 dgemms (the split "
        f"right-einsums), got {len(f64_dots)}: {dots}")
    bad = [s for s in c128_dots if s not in allowed_c128_dots]
    assert not bad, (
        f"[{tag}] c128 dot(s) outside the allowed genuinely-complex set "
        f"{sorted(allowed_c128_dots)}: {bad} — a promoted complex GEMM "
        f"re-grew")

    # (3) Collective contract byte-untouched: exactly two c128
    # reduce-scatters at the exact pre-relowering payload shapes.
    rs = _RS_RE.findall(txt)
    assert len(rs) == 2, f"[{tag}] expected 2 reduce-scatters, got {rs}"
    assert all(t == "c128" for t, _ in rs), f"[{tag}] rs dtype changed: {rs}"
    got = sorted(s for _, s in rs)
    assert got == sorted(expect_rs_shapes), (
        f"[{tag}] reduce-scatter payload shapes changed: {got} vs "
        f"expected {sorted(expect_rs_shapes)} — collectives are a "
        f"separate owner-gated lever and must not move here")


def test_two_channel_hlo_and_parity():
    """Two-channel body: 4 f64 right dgemms, no promotion, rs unchanged."""
    txt = _compiled_text(False)
    _assert_lowering(
        txt, tag="two-channel",
        # 'y' rs output (2, nk, s, μ_X_loc, n/p_y); 'x' rs output
        # (2, nk, m/p_x, n/p_y) — stacked leading 2 (AK.9), as before.
        expect_rs_shapes=[
            f"2,{NK},{NS},{MU // PX},{MN // PY}",
            f"2,{NK},{MN // PX},{MN // PY}",
        ],
        # Only the small post-'y'-scatter left dots may be complex.
        allowed_c128_dots={f"{NK},{MN},{MN // PY}"},
        min_f64_dots=4)

    mesh = _mesh()
    proj = contract_bands_block_reshard(mesh, channels="split_reim")
    (psi_xr, sigma_k, psi_yn), dev = _inputs(mesh)
    s_r, s_i = jax.jit(proj)(*dev)
    s_r, s_i = np.asarray(s_r), np.asarray(s_i)
    ref_r = np.einsum("kmsx,ksxty,ktyn->kmn", np.conj(psi_xr),
                      np.real(sigma_k).astype(np.complex128), psi_yn,
                      optimize=True)
    ref_i = np.einsum("kmsx,ksxty,ktyn->kmn", np.conj(psi_xr),
                      np.imag(sigma_k).astype(np.complex128), psi_yn,
                      optimize=True)
    scale = max(np.max(np.abs(ref_r)), np.max(np.abs(ref_i)))
    err = max(np.max(np.abs(s_r - ref_r)), np.max(np.abs(s_i - ref_i)))
    assert err / scale <= TOL, f"two-channel parity {err / scale:.3e} > 1e-12"


def test_merged_hlo_and_parity():
    """Merged body: genuine complex dots only, no promotion, halved rs."""
    txt = _compiled_text(True)
    _assert_lowering(
        txt, tag="merged",
        # Merged plan: NO stacked leading 2 — payload halved by design
        # (laplace_merge_notes.md colltable check), unchanged here.
        expect_rs_shapes=[
            f"{NK},{NS},{MU // PX},{MN // PY}",
            f"{NK},{MN // PX},{MN // PY}",
        ],
        # The merged body is genuinely complex × complex: its large right
        # dot (either unmerged (nk,s,μ_X_loc,n) or with the s,μ free dims
        # merged) AND the small left dot are the allowed c128 dots — the
        # f64 split was REFUTED here (see module docstring).
        allowed_c128_dots={
            f"{NK},{NS},{MU // PX},{MN}",
            f"{NK},{NS * (MU // PX)},{MN}",
            f"{NK},{MN},{MN // PY}",
        },
        min_f64_dots=0)

    mesh = _mesh()
    proj = contract_bands_block_reshard(mesh, channels="none")
    (psi_xr, sigma_k, psi_yn), dev = _inputs(mesh)
    x = np.asarray(jax.jit(proj)(*dev))
    ref = np.einsum("kmsx,ksxty,ktyn->kmn", np.conj(psi_xr), sigma_k,
                    psi_yn, optimize=True)
    scale = np.max(np.abs(ref))
    err = np.max(np.abs(x - ref))
    assert err / scale <= TOL, f"merged parity {err / scale:.3e} > 1e-12"


# ---------------------------------------------------------------------------

def _main():
    # The pytest path gets this from the autouse `_xla_plan_dial` fixture,
    # which does not run here because this branch calls the test functions
    # directly.  Safe under __main__: pytest did not start this process.
    os.environ.setdefault("LORRAX_BANDS_GEMM_FFI", "0")
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
        except Exception as exc:               # noqa: BLE001 - test runner
            import traceback
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
