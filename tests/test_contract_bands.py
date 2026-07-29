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
   byte-identical to the XLA plan, and the extra="minor" refusal.

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
    n_dev = len(jax.devices())
    assert n_dev >= 4, (
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
# AUTO default (owner order 2026-07-29): ON iff CPU platform + handler in
# the host .so; explicit 0 disables; auto quietly keeps the XLA plan where
# the dial cannot apply (extra="minor", non-f64/c128 dtypes).
# ---------------------------------------------------------------------------

def test_auto_default():
    ok, reason = _ffi_available()
    prev = os.environ.pop("LORRAX_BANDS_GEMM_FFI", None)
    try:
        assert bands_gemm_ffi_mode() == "auto"
        auto_on = bands_gemm_ffi_enabled()
        assert auto_on == ok, (
            f"auto verdict {auto_on} != handler availability {ok} "
            f"({reason})")
        os.environ["LORRAX_BANDS_GEMM_FFI"] = "auto"
        assert bands_gemm_ffi_enabled() == ok
        os.environ["LORRAX_BANDS_GEMM_FFI"] = "0"
        assert bands_gemm_ffi_mode() == "off"
        assert not bands_gemm_ffi_enabled(), "explicit =0 must disable auto"
        del os.environ["LORRAX_BANDS_GEMM_FFI"]

        mesh = _mesh()
        if ok:
            # auto-ON: the leading plan carries the GEMM custom-call ...
            proj = contract_bands_block_reshard(mesh, extra="leading")
            (psi_l, o, psi_r), dev = _operands(mesh, extra="leading")
            out = jax.jit(proj)(*dev)
            err = _relerr(out, _ref(psi_l, o, psi_r, extra="leading"))
            assert err <= TOL, f"[auto leading] parity {err:.3e} > 1e-14"
            txt = _compile_text(proj, dev)
            assert len(_CC_RE.findall(txt)) == 1, (
                "[auto leading] expected the FFI custom-call under auto-ON")
            # ... while extra="minor" quietly keeps the XLA plan (the
            # explicit-mode refusal must NOT fire under auto) ...
            proj_m = contract_bands_block_reshard(mesh, extra="minor")
            (psi_l, o, psi_r), dev = _operands(mesh, extra="minor")
            out = jax.jit(proj_m)(*dev)
            err = _relerr(out, _ref(psi_l, o, psi_r, extra="minor"))
            assert err <= TOL, f"[auto minor] parity {err:.3e} > 1e-14"
            txt = _compile_text(proj_m, dev)
            assert len(_CC_RE.findall(txt)) == 0, (
                "[auto minor] must fall back to the XLA plan, not refuse")
            # ... and non-f64/c128 dtypes (fp32 BSE class) fall back too.
            proj_c = contract_bands_block_reshard(mesh, channels="none")
            (psi_l, o, psi_r), dev = _operands(mesh)
            dev64 = tuple(jnp.asarray(d, dtype=jnp.complex64) for d in dev)
            txt = jax.jit(proj_c).lower(*dev64).compile().as_text()
            assert len(_CC_RE.findall(txt)) == 0, (
                "[auto c64] complex64 operands must keep the XLA lowering")
            print("  [auto] AUTO-ON engages; minor + c64 fall back to XLA",
                  flush=True)
        else:
            proj = contract_bands_block_reshard(mesh, extra="leading")
            (psi_l, o, psi_r), dev = _operands(mesh, extra="leading")
            txt = _compile_text(proj, dev)
            assert len(_CC_RE.findall(txt)) == 0
            print(f"  [auto] handler unavailable -> auto-OFF ({reason})",
                  flush=True)
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

        # extra=minor must REFUSE under the FFI dial.
        try:
            contract_bands_block_reshard(mesh, extra="minor")
            raise AssertionError("ffi extra=minor refusal did not fire")
        except RuntimeError as exc:
            assert "minor" in str(exc)
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
