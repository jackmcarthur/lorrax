"""HLO pin + parity + refusal gates for contract_bands_block_reshard.

The shared band projection+reshard primitive
(``common.contract_bands.contract_bands_block_reshard``, owner directive
2026-07-28) subsumes the two Σ projection tails and is slated for the BSE
matvec decode.  These tests compile it on a 2×2 emulated-device mesh (the
``test_sanity_gates_jax.py`` 4-device pattern) and assert on the optimized
HLO — the only ground truth for lowering claims (QUALITY_PATTERNS #4):

1. identity vs an independent numpy reference einsum at 1e-14 (relative)
   for every plan: single complex chain, split_reim two-channel, BOTH
   extra-stack orders (leading AND minor), and the real-operand
   de-promoted chain;
2. collective contract: exactly two c128 ``reduce-scatter`` ops per
   module at the exact expected payload shapes (channel/extra stack
   riding ONE collective per mesh axis — AK.9);
3. de-promotion: NO rank≥2 f64→c128 ``convert`` anywhere a real operand
   enters (the ~400 MB promotion-copy class the memo priced);
4. refusals are actionable: divisibility, inverted mesh axis order,
   split_reim×extra, split_reim on a real O;
5. (when the host .so with the mklblas handler is reachable) the
   LORRAX_BANDS_GEMM_FFI plan: parity at 1e-14, the right-GEMM
   custom-calls present at the expected count, reduce-scatter payloads
   byte-identical to the XLA plan, and the quiet extra="minor" XLA
   fallback (structural — the contracted axis is not GEMM-reachable).

The XLA-plan pins (1-4) run with the dial pinned =0 (the announced debug
opt-out): since the FFI-required ruling (decisions.md 2026-08-01) the
dial defaults ON and a missing handler REFUSES at the factory instead of
demoting, so an unpinned run of these pins would either take the GEMM
plan or die, depending on whether the .so is reachable.

Run inside the container (login python has no jax), e.g.::

    /scratch2/08271/jackmc/lorrax_setup/alloc_run.sh 1 1 \
        /work2/08271/jackmc/frontera/lorrax/src \
        /work2/08271/jackmc/frontera/lorrax \
        python -u tests/test_contract_bands.py

One process, no GPU, seconds.  The multi-process collective behavior
(impl=mpi warm-up contract included) is covered by the restart-gated P=64
A/B (wk_REL/cbands_ab.sbatch).
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_FLAGS",
                      "--xla_force_host_platform_device_count=4")
# The XLA-plan pins below assert the NATIVE lowering; the dial defaults ON
# since the FFI-required ruling, so pin it to the announced debug opt-out.
# The FFI-plan tests set =1 (and the default test unsets) explicitly.
os.environ.setdefault("LORRAX_BANDS_GEMM_FFI", "0")

import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from common.contract_bands import (                  # noqa: E402
    contract_bands_block_reshard, bands_gemm_ffi_enabled,
    bands_gemm_ffi_mode)


NK, NS, MU, MN, E = 4, 2, 16, 8, 3
PX = PY = 2
TOL = 1e-14

_CONVERT_RE = re.compile(r"=\s+c128\[([\d,]*)\]\S*\s+convert\(f64\[")
_RS_RE = re.compile(r"=\s+(\w+)\[([\d,]*)\]\S*\s+reduce-scatter\(")
_CC_RE = re.compile(r"custom_call_target=\"lorrax_mklblas_gemm_batch\"")
_F64DOT_RE = re.compile(r"=\s+f64\[[\d,]*\]\S*\s+dot\(")


def _mesh():
    import pytest
    n_dev = len(jax.devices())
    if n_dev < 4:
        # skip, not assert: a plain 1-device invocation is a legitimate way
        # to run the suite (the 4-device leg reruns these under
        # XLA_FLAGS=--xla_force_host_platform_device_count=4) — same
        # convention as test_staged_reshard / test_sharding_fit.
        pytest.skip(
            f"needs 4 (emulated) devices, got {n_dev}; set "
            f"XLA_FLAGS=--xla_force_host_platform_device_count=4")
    return Mesh(np.asarray(jax.devices()[:4]).reshape(PX, PY), ("x", "y"))


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def _operands(mesh, *, extra="none", real_o=False, seed=37):
    rng = np.random.default_rng(seed)
    psi_l = _crand(rng, NK, MN, NS, MU)
    psi_r = _crand(rng, NK, NS, MU, MN)
    if extra == "leading":
        o = _crand(rng, E, NK, NS, MU, NS, MU)
        o_spec = P(None, None, None, "x", None, "y")
    elif extra == "minor":
        o = _crand(rng, NK, NS, MU, NS, MU, E)
        o_spec = P(None, None, "x", None, "y", None)
    else:
        o = _crand(rng, NK, NS, MU, NS, MU)
        o_spec = P(None, None, "x", None, "y")
    if real_o:
        o = np.real(o).copy()
    dev = (
        jax.device_put(jnp.asarray(psi_l),
                       NamedSharding(mesh, P(None, None, None, "x"))),
        jax.device_put(jnp.asarray(o), NamedSharding(mesh, o_spec)),
        jax.device_put(jnp.asarray(psi_r),
                       NamedSharding(mesh, P(None, None, "y", None))),
    )
    return (psi_l, o, psi_r), dev


def _ref(psi_l, o, psi_r, *, extra="none"):
    sub = {"none": "kmsx,ksxty,ktyn->kmn",
           "leading": "kmsx,eksxty,ktyn->ekmn",
           "minor": "kmsx,ksxtye,ktyn->kmne"}[extra]
    return np.einsum(sub, np.conj(psi_l), o.astype(np.complex128), psi_r,
                     optimize=True)


def _relerr(a, b):
    scale = max(np.max(np.abs(b)), 1e-300)
    return np.max(np.abs(np.asarray(a) - b)) / scale


def _assert_rs(txt, tag, expect_shapes):
    rs = _RS_RE.findall(txt)
    assert len(rs) == 2, f"[{tag}] expected 2 reduce-scatters, got {rs}"
    assert all(t == "c128" for t, _ in rs), f"[{tag}] rs dtype changed: {rs}"
    got = sorted(s for _, s in rs)
    assert got == sorted(expect_shapes), (
        f"[{tag}] reduce-scatter payload shapes {got} != expected "
        f"{sorted(expect_shapes)}")


def _assert_no_promotion(txt, tag):
    offenders = [m.group(0) for m in _CONVERT_RE.finditer(txt)
                 if "," in m.group(1)]
    assert not offenders, (
        f"[{tag}] rank>=2 f64->c128 promotion convert(s) present:\n"
        + "\n".join(offenders))


def _compile_text(proj, dev):
    return jax.jit(proj).lower(*dev).compile().as_text()


# ---------------------------------------------------------------------------
# XLA-plan parity + HLO pins
# ---------------------------------------------------------------------------

def test_none_complex():
    mesh = _mesh()
    proj = contract_bands_block_reshard(mesh, channels="none")
    (psi_l, o, psi_r), dev = _operands(mesh)
    out = jax.jit(proj)(*dev)
    err = _relerr(out, _ref(psi_l, o, psi_r))
    assert err <= TOL, f"none/complex parity {err:.3e} > 1e-14"
    txt = _compile_text(proj, dev)
    _assert_no_promotion(txt, "none/complex")
    _assert_rs(txt, "none/complex", [
        f"{NK},{NS},{MU // PX},{MN // PY}", f"{NK},{MN // PX},{MN // PY}"])


def test_split_reim():
    mesh = _mesh()
    proj = contract_bands_block_reshard(mesh, channels="split_reim")
    (psi_l, o, psi_r), dev = _operands(mesh)
    s_r, s_i = jax.jit(proj)(*dev)
    ref_r = _ref(psi_l, np.real(o), psi_r)
    ref_i = _ref(psi_l, np.imag(o), psi_r)
    err = max(_relerr(s_r, ref_r), _relerr(s_i, ref_i))
    assert err <= TOL, f"split_reim parity {err:.3e} > 1e-14"
    txt = _compile_text(proj, dev)
    _assert_no_promotion(txt, "split_reim")
    # Stacked leading 2 (AK.9), exactly the historical two-channel payloads.
    _assert_rs(txt, "split_reim", [
        f"2,{NK},{NS},{MU // PX},{MN // PY}",
        f"2,{NK},{MN // PX},{MN // PY}"])
    n_f64 = len(_F64DOT_RE.findall(txt))
    assert n_f64 >= 4, (
        f"[split_reim] expected >=4 f64 dgemms (de-promoted right "
        f"contractions), got {n_f64}")


def test_extra_leading():
    mesh = _mesh()
    proj = contract_bands_block_reshard(mesh, extra="leading")
    (psi_l, o, psi_r), dev = _operands(mesh, extra="leading")
    out = jax.jit(proj)(*dev)
    err = _relerr(out, _ref(psi_l, o, psi_r, extra="leading"))
    assert err <= TOL, f"extra=leading parity {err:.3e} > 1e-14"
    txt = _compile_text(proj, dev)
    _assert_no_promotion(txt, "extra=leading")
    _assert_rs(txt, "extra=leading", [
        f"{E},{NK},{NS},{MU // PX},{MN // PY}",
        f"{E},{NK},{MN // PX},{MN // PY}"])


def test_extra_minor():
    mesh = _mesh()
    proj = contract_bands_block_reshard(mesh, extra="minor")
    (psi_l, o, psi_r), dev = _operands(mesh, extra="minor")
    out = jax.jit(proj)(*dev)
    err = _relerr(out, _ref(psi_l, o, psi_r, extra="minor"))
    assert err <= TOL, f"extra=minor parity {err:.3e} > 1e-14"
    txt = _compile_text(proj, dev)
    _assert_no_promotion(txt, "extra=minor")
    _assert_rs(txt, "extra=minor", [
        f"{NK},{NS},{MU // PX},{MN // PY},{E}",
        f"{NK},{MN // PX},{MN // PY},{E}"])


def test_real_operand_depromotion():
    """Real O (the stacked-channel class): pure-f64 right dgemms, no
    promotion converts, complex output via lax.complex."""
    mesh = _mesh()
    proj = contract_bands_block_reshard(mesh, extra="leading")
    (psi_l, o, psi_r), dev = _operands(mesh, extra="leading", real_o=True)
    out = jax.jit(proj)(*dev)
    err = _relerr(out, _ref(psi_l, o, psi_r, extra="leading"))
    assert err <= TOL, f"real-O parity {err:.3e} > 1e-14"
    txt = _compile_text(proj, dev)
    _assert_no_promotion(txt, "real-O")
    n_f64 = len(_F64DOT_RE.findall(txt))
    assert n_f64 >= 2, (
        f"[real-O] expected >=2 f64 dgemms (Re-psi/Im-psi split), got "
        f"{n_f64}")
    _assert_rs(txt, "real-O", [
        f"{E},{NK},{NS},{MU // PX},{MN // PY}",
        f"{E},{NK},{MN // PX},{MN // PY}"])


# ---------------------------------------------------------------------------
# Refusals (must be actionable, not cryptic psum_scatter crashes)
# ---------------------------------------------------------------------------

def test_refusals():
    mesh = _mesh()
    # (a) divisibility: m=6 not divisible by p_x=2? use m=7.
    proj = contract_bands_block_reshard(mesh)
    rng = np.random.default_rng(5)
    psi_l = jnp.asarray(_crand(rng, NK, 7, NS, MU))
    psi_r = jnp.asarray(_crand(rng, NK, NS, MU, MN))
    o = jnp.asarray(_crand(rng, NK, NS, MU, NS, MU))
    try:
        proj(psi_l, o, psi_r)
        raise AssertionError("divisibility refusal did not fire")
    except ValueError as exc:
        assert "pad_sigma_window" in str(exc) or "INDEPENDENTLY" in str(exc)

    # (b) inverted axes: large payload would ride the strided axis.
    try:
        contract_bands_block_reshard(mesh, axes=("y", "x"))
        raise AssertionError("inverted-axes refusal did not fire")
    except ValueError as exc:
        assert "minor axis" in str(exc)

    # (c) split_reim + extra stack.
    try:
        contract_bands_block_reshard(mesh, channels="split_reim",
                                     extra="leading")
        raise AssertionError("split_reim+extra refusal did not fire")
    except ValueError as exc:
        assert "split_reim" in str(exc)

    # (d) split_reim on a real O.
    proj2 = contract_bands_block_reshard(mesh, channels="split_reim")
    o_real = jnp.asarray(np.real(np.asarray(o)))
    psi_l8 = jnp.asarray(_crand(rng, NK, MN, NS, MU))
    try:
        proj2(psi_l8, o_real, psi_r)
        raise AssertionError("split_reim real-O refusal did not fire")
    except TypeError as exc:
        assert "complex" in str(exc)


# ---------------------------------------------------------------------------
# REQUIRED default (decisions.md 2026-08-01): the dial defaults ON; =0 is
# an announced debug opt-out; a stale =auto (the deleted capability-
# detection mode) resolves to the default with a grammar note; a missing
# handler REFUSES at the factory instead of demoting.  extra="minor"
# quietly keeps the XLA plan under every mode (structural).
# ---------------------------------------------------------------------------

def test_required_default():
    ok, reason = _ffi_available()
    prev = os.environ.pop("LORRAX_BANDS_GEMM_FFI", None)
    try:
        assert bands_gemm_ffi_mode() == "on", (
            "unset must resolve ON — the FFI layer is required")
        assert bands_gemm_ffi_enabled()
        os.environ["LORRAX_BANDS_GEMM_FFI"] = "auto"    # deleted mode
        assert bands_gemm_ffi_mode() == "on", (
            "a stale =auto must resolve to the default (announced grammar "
            "note), never silently to off")
        os.environ["LORRAX_BANDS_GEMM_FFI"] = "0"
        assert bands_gemm_ffi_mode() == "off"
        assert not bands_gemm_ffi_enabled(), "explicit =0 must opt out"
        del os.environ["LORRAX_BANDS_GEMM_FFI"]

        mesh = _mesh()
        if ok:
            # required-ON: the leading plan carries the GEMM custom-call ...
            proj = contract_bands_block_reshard(mesh, extra="leading")
            (psi_l, o, psi_r), dev = _operands(mesh, extra="leading")
            out = jax.jit(proj)(*dev)
            err = _relerr(out, _ref(psi_l, o, psi_r, extra="leading"))
            assert err <= TOL, f"[req leading] parity {err:.3e} > 1e-14"
            txt = _compile_text(proj, dev)
            assert len(_CC_RE.findall(txt)) == 1, (
                "[req leading] expected the FFI custom-call by default")
            # ... while extra="minor" quietly keeps the XLA plan
            # (structural, not a demotion) ...
            proj_m = contract_bands_block_reshard(mesh, extra="minor")
            (psi_l, o, psi_r), dev = _operands(mesh, extra="minor")
            out = jax.jit(proj_m)(*dev)
            err = _relerr(out, _ref(psi_l, o, psi_r, extra="minor"))
            assert err <= TOL, f"[req minor] parity {err:.3e} > 1e-14"
            txt = _compile_text(proj_m, dev)
            assert len(_CC_RE.findall(txt)) == 0, (
                "[req minor] must keep the XLA plan quietly, not refuse")
            # ... while complex64 RIDES the handler (all four precisions).
            proj_c = contract_bands_block_reshard(mesh, channels="none")
            (psi_l, o, psi_r), dev = _operands(mesh)
            dev64 = tuple(jnp.asarray(d, dtype=jnp.complex64) for d in dev)
            txt = jax.jit(proj_c).lower(*dev64).compile().as_text()
            assert len(_CC_RE.findall(txt)) == 1, (
                "[req c64] complex64 must use the cgemm custom-call")
            print("  [required] default ON engages; minor keeps XLA "
                  "quietly, c64 rides the handler", flush=True)
        else:
            # REQUIRED: a missing handler refuses at the factory, naming
            # the library — never a silent demotion to the XLA plan.
            try:
                contract_bands_block_reshard(mesh, extra="leading")
                raise AssertionError(
                    "required-handler refusal did not fire with the "
                    "handler unavailable")
            except RuntimeError as exc:
                assert "REQUIRED" in str(exc), str(exc)
            print(f"  [required] handler unavailable -> factory refusal "
                  f"({reason})", flush=True)
    finally:
        if prev is None:
            os.environ.pop("LORRAX_BANDS_GEMM_FFI", None)
        else:
            os.environ["LORRAX_BANDS_GEMM_FFI"] = prev


# ---------------------------------------------------------------------------
# Gated FFI MKL GEMM plan (runs only when the host .so is reachable)
# ---------------------------------------------------------------------------

def _ffi_available():
    try:
        from ffi.common import ffi_loader
        ok, reason = ffi_loader.probe_target("lorrax_mklblas_gemm_batch",
                                             "cpu")
        return ok, reason
    except Exception as exc:                       # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Single precision (owner order 2026-07-29): the handler serves all four
# BLAS precisions, so c64 (the BSE fp32-GMRES class) and f32 must both use
# the custom-call AND agree with the reference.
#
# TOLERANCE JUSTIFICATION.  Unit roundoff for binary32 is u = 2^-24 =
# 5.96e-8.  The worst-case relative error of a length-K dot product is
# ~K*u; here the contracted length is K = NS*MU = 32, so K*u = 1.9e-6.  We
# compare TWO f32 evaluations with different summation orders (the vendor
# GEMM vs the XLA/Eigen dot), each carrying that bound, so the difference
# can reach ~2*K*u = 3.8e-6.  TOL32 = 1e-5 is therefore the smallest round
# number strictly ABOVE the worst case.  A 1e-6 tolerance would sit BELOW
# the theoretical bound and be flaky by construction — the measured error
# is printed so the real margin is visible rather than assumed.
# ---------------------------------------------------------------------------

TOL32 = 1e-5


def test_single_precision_ffi():
    ok, reason = _ffi_available()
    if not ok:
        print(f"  [f32] SKIP (handler unavailable: {reason})", flush=True)
        return
    mesh = _mesh()
    (psi_l, o, psi_r), dev = _operands(mesh)
    exact = _ref(psi_l, o, psi_r)          # float64/complex128 reference

    def _run(dtype, gemm):
        os.environ["LORRAX_BANDS_GEMM_FFI"] = gemm
        try:
            proj = contract_bands_block_reshard(mesh, channels="none")
            dsm = tuple(jnp.asarray(d, dtype=dtype) for d in dev)
            out = jax.jit(proj)(*dsm)
            txt = jax.jit(proj).lower(*dsm).compile().as_text()
            return np.asarray(out), len(_CC_RE.findall(txt))
        finally:
            os.environ["LORRAX_BANDS_GEMM_FFI"] = "0"

    # --- complex64: the BSE fp32-GMRES class -------------------------
    on_c, ncc_on = _run(jnp.complex64, "1")
    off_c, ncc_off = _run(jnp.complex64, "0")
    assert ncc_on == 1, f"[f32 c64] expected 1 cgemm custom-call, got {ncc_on}"
    assert ncc_off == 0, f"[f32 c64] dial=0 must not emit one, got {ncc_off}"
    d_hh = _relerr(on_c, off_c)            # handler vs XLA, both f32
    d_he = _relerr(on_c, exact)            # handler vs exact
    assert d_hh <= TOL32, f"[f32 c64] handler-vs-XLA {d_hh:.3e} > {TOL32}"
    assert d_he <= TOL32, f"[f32 c64] handler-vs-exact {d_he:.3e} > {TOL32}"

    # --- float32: all-real operands take the sgemm path --------------
    # NB: _operands(real_o=True) realifies only O — ψ stays complex.  The
    # f32 GEMM path needs ALL THREE real, and the reference must be taken
    # from the SAME real operands.  (First version compared a real-operand
    # run against a complex-ψ reference and "failed" at 6.8e-1: a test
    # bug, not a handler bug.  Kept as a comment so it is not reintroduced.)
    (cpsi_l, co, cpsi_r), rdev = _operands(mesh, real_o=True)
    rl = np.real(cpsi_l).copy()
    ro = np.real(co).copy()
    rr = np.real(cpsi_r).copy()
    r_exact = np.real(_ref(rl, ro, rr))
    rdev32 = tuple(jax.device_put(jnp.asarray(v, dtype=jnp.float32),
                                  d.sharding)
                   for v, d in zip((rl, ro, rr), rdev))
    os.environ["LORRAX_BANDS_GEMM_FFI"] = "1"
    try:
        proj = contract_bands_block_reshard(mesh, channels="none")
        out32 = np.asarray(jax.jit(proj)(*rdev32))
        txt = jax.jit(proj).lower(*rdev32).compile().as_text()
        ncc32 = len(_CC_RE.findall(txt))
    finally:
        os.environ["LORRAX_BANDS_GEMM_FFI"] = "0"
    assert ncc32 == 1, f"[f32 f32] expected 1 sgemm custom-call, got {ncc32}"
    d32 = _relerr(out32, r_exact)
    assert d32 <= TOL32, f"[f32 f32] handler-vs-exact {d32:.3e} > {TOL32}"

    print(f"  [f32] c64 handler-vs-XLA {d_hh:.2e}, handler-vs-exact "
          f"{d_he:.2e}; f32 handler-vs-exact {d32:.2e}  (tol {TOL32:g} = "
          f"2*K*u with K={NS*MU}, u=2^-24)", flush=True)
    print("  [f32] all four precisions served — c64/f32 custom-calls fire",
          flush=True)


def test_ffi_gemm_plan():
    ok, reason = _ffi_available()
    if not ok:
        print(f"  [ffi] SKIP (handler unavailable: {reason})", flush=True)
        return
    os.environ["LORRAX_BANDS_GEMM_FFI"] = "1"
    try:
        assert bands_gemm_ffi_enabled()
        mesh = _mesh()

        # none/complex → ONE zgemm custom-call; rs payloads unchanged.
        proj = contract_bands_block_reshard(mesh, channels="none")
        (psi_l, o, psi_r), dev = _operands(mesh)
        out = jax.jit(proj)(*dev)
        err = _relerr(out, _ref(psi_l, o, psi_r))
        assert err <= TOL, f"[ffi none] parity {err:.3e} > 1e-14"
        txt = _compile_text(proj, dev)
        ncc = len(_CC_RE.findall(txt))
        assert ncc == 1, f"[ffi none] expected 1 gemm custom-call, got {ncc}"
        _assert_rs(txt, "ffi none", [
            f"{NK},{NS},{MU // PX},{MN // PY}",
            f"{NK},{MN // PX},{MN // PY}"])

        # split_reim → FOUR dgemm custom-calls (2 channels × Re/Im ψ).
        proj = contract_bands_block_reshard(mesh, channels="split_reim")
        s_r, s_i = jax.jit(proj)(*dev)
        err = max(_relerr(s_r, _ref(psi_l, np.real(o), psi_r)),
                  _relerr(s_i, _ref(psi_l, np.imag(o), psi_r)))
        assert err <= TOL, f"[ffi split] parity {err:.3e} > 1e-14"
        txt = _compile_text(proj, dev)
        ncc = len(_CC_RE.findall(txt))
        assert ncc == 4, f"[ffi split] expected 4 gemm custom-calls, got {ncc}"
        _assert_no_promotion(txt, "ffi split")
        _assert_rs(txt, "ffi split", [
            f"2,{NK},{NS},{MU // PX},{MN // PY}",
            f"2,{NK},{MN // PX},{MN // PY}"])

        # extra=leading (broadcast batch) → ONE custom-call.
        proj = contract_bands_block_reshard(mesh, extra="leading")
        (psi_l, o, psi_r), dev = _operands(mesh, extra="leading")
        out = jax.jit(proj)(*dev)
        err = _relerr(out, _ref(psi_l, o, psi_r, extra="leading"))
        assert err <= TOL, f"[ffi leading] parity {err:.3e} > 1e-14"
        txt = _compile_text(proj, dev)
        ncc = len(_CC_RE.findall(txt))
        assert ncc == 1, (
            f"[ffi leading] expected 1 gemm custom-call, got {ncc}")

        # extra=minor quietly keeps the XLA plan even at =1 (structural:
        # the contracted axis is not GEMM-reachable; not a demotion).
        proj_m = contract_bands_block_reshard(mesh, extra="minor")
        (psi_l, o, psi_r), dev = _operands(mesh, extra="minor")
        out = jax.jit(proj_m)(*dev)
        err = _relerr(out, _ref(psi_l, o, psi_r, extra="minor"))
        assert err <= TOL, f"[ffi minor] parity {err:.3e} > 1e-14"
        txt = _compile_text(proj_m, dev)
        assert len(_CC_RE.findall(txt)) == 0, (
            "[ffi minor] must keep the XLA plan (0 custom-calls)")
        print("  [ffi] all FFI-plan gates PASS", flush=True)
    finally:
        os.environ["LORRAX_BANDS_GEMM_FFI"] = "0"


# ---------------------------------------------------------------------------

def _main():
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
