"""Numerical gate for the flat-k FFT host handler (``lorrax_mklfft_flat_k``).

WRITTEN BEFORE THE FFTW3 BACKEND EXISTS, ON PURPOSE.  A subtly wrong FFT
does not crash: it returns plausible numbers and silently corrupts every
Sigma downstream.  So the contract gets pinned against an independent
reference (numpy) first, and the implementation is then made to satisfy it.

WHAT IS ACTUALLY BEING CHECKED
------------------------------
The handler computes a 3-D FFT over the LEADING flat-k axis of a
``(nk, *trail)`` c128 array, nk == nkx*nky*nkz, batched over the trail.  The
array is k-MAJOR: element (k, t) lives at ``x[k*T + t]`` with T = prod(trail).
That is the whole point of the handler — the transform reads the k axis at
stride T instead of demanding a k-minor transpose first.

So the reference is:  reshape (nk, *trail) -> (nkx, nky, nkz, *trail),
run numpy's fftn/ifftn over axes (0,1,2), reshape back.  If the handler's
batch descriptor is wrong (wrong stride, wrong distance, wrong embed), the
result is a well-formed array of wrong numbers, and ONLY a comparison like
this one catches it.

THE FOUR HAZARD CLASSES, AND WHICH TEST COVERS EACH
---------------------------------------------------
1. plain correctness + norm conventions   -> test_matches_numpy_*
2. STRIDED/BATCHED descriptor (istride=T, -> test_batched_trail_* (T > 1);
   idist=1) -- a plan-descriptor error        this is where a transposed or
   hides here and NOWHERE else                mis-strided plan shows up
3. in-place aliasing (XLA may hand the    -> test_inplace_alias_matches
   same buffer as in and out)
4. howmany > 1 with a k extent that is    -> test_nonuniform_kgrid_batched
   NOT contiguous in memory (nkx!=nky!=nkz,   (the uniform-batch assumption
   T > 1) -- the uniform-batch assumption      is the thing most likely to
                                               be wrong on a new machine

Run (host FFI must be built and on LORRAX_FFI_HOST_SO)::

    JAX_PLATFORMS=cpu srun --mpi=pmi2 -n1 python -m pytest \
        tests/test_fft_flat_k_numerics.py -v
"""
from __future__ import annotations

import os

# MUST precede the jax import.  The handler is complex128-only and refuses
# anything else; without x64 JAX silently downcasts every literal to
# complex64 and the refusal fires before a single number is compared.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

from ffi.common import ffi_loader  # noqa: E402
from ffi.fft import FLAT_K_TARGET, make_flat_k_fft_ffi  # noqa: E402

# PARITY CLASS, STATED EXPLICITLY (TASTE rule 15).
#
# Value-level, NOT bit-exact.  Per docs/dev/flat_k_fft_service.md §7 and
# docs/architecture/ffi_layout.md §7: comparing against a different FFT
# engine is a ~1e-15 value-level agreement, and the Sigma path is gated at
# relative 1e-12.  Two things this deliberately is NOT:
#
#   * NOT bit-exactness.  wk_REL/gatecheck.py cells E/E2 assert
#     np.array_equal only between two callers of the SAME handler.  Swapping
#     DFTI for FFTW changes the engine, where bit equality is not promised
#     and asserting it would be a false gate.
#   * NOT 1e-16.  Those figures in the evidence index are MEASURED unit
#     residuals (max 3.7e-16) sitting at the c128 ULP; a threshold there
#     tests nothing.  ffi_gate_contract.md §3.5 records that the gate's
#     first run found two cells perturbing below the ULP and therefore
#     asserting nothing at all.
#
# So: 1e-12 absolute on unit-scale data / 1e-11 relative — inside the stated
# class, loose enough to be engine-independent, tight enough that a wrong
# stride, distance or axis order cannot pass.
ATOL = 1e-12
RTOL = 1e-11


def _probe() -> tuple[bool, str]:
    ok, reason = ffi_loader.probe_target(FLAT_K_TARGET, "cpu")
    return bool(ok), str(reason)


_OK, _REASON = _probe()

# SKIP vs FAIL is a deliberate distinction, and it is the whole reason this
# module does not just use the repo's usual `needs_host_ffi` skipif.
#
#   library absent / unloadable  -> SKIP.  That is an ENVIRONMENT fact (no
#       host build here, wrong LD_LIBRARY_PATH); nothing about the FFT is
#       being asserted, so skipping is honest.
#   library loads, handler missing -> FAIL.  That is a BUILD DEFECT.  The
#       host lib exists and was configured without an FFT backend, which
#       means every CPU chain driver refuses at startup
#       (runtime._enforce_required_ffi).  A skip here would report green on
#       a build that cannot run, which is exactly the "scoped check reported
#       as an unscoped all-clear" failure this suite exists to prevent.
_LIB_LOADED = _OK or ("does not export" in _REASON) or ("unknown target" in _REASON)

if _LIB_LOADED and not _OK:
    def test_flat_k_fft_handler_must_be_built():
        pytest.fail(
            "The host FFI library loaded but exports NO flat-k FFT handler:\n"
            f"  {_REASON}\n"
            "Every CPU-mesh chain driver refuses at startup without it "
            "(LORRAX_FFT_FFI defaults ON and off_policy='refuse' — the XLA "
            "flat-k arm was deleted under the FFI-required ruling). "
            "Build the host lib with an FFT backend: cray-fftw / FFTW3 via "
            "config/perlmutter/build_ffi_host.sh, or MKL DFTI elsewhere.")

needs_fft_ffi = pytest.mark.skipif(
    not _LIB_LOADED,
    reason=f"host FFI library not loadable (environment): {_REASON}",
)


def _mesh() -> Mesh:
    """1x1 CPU mesh — the handler is rank-local, so one device exercises the
    whole descriptor path.  Multi-rank adds no new FFT behaviour (the k axes
    must be replicated; validate_flat_spec enforces it)."""
    dev = jax.devices("cpu")[:1]
    return Mesh(np.array(dev).reshape(1, 1), ("x", "y"))


def _reference(x_flat: np.ndarray, kgrid, kind: str, norm) -> np.ndarray:
    """Independent numpy reference on the k-MAJOR flat layout."""
    nkx, nky, nkz = kgrid
    trail = x_flat.shape[1:]
    x3 = x_flat.reshape(nkx, nky, nkz, *trail)
    fn = np.fft.fftn if kind == "fftn" else np.fft.ifftn
    out = fn(x3, axes=(0, 1, 2), norm=norm)
    return np.asarray(out).reshape(x_flat.shape)


def _run(x_flat: np.ndarray, kgrid, kind: str, norm):
    """Drive the real FFI call path (make_flat_k_fft_ffi -> shard_map ->
    jax.ffi.ffi_call), not a hand-rolled ffi_call, so the wrapper's own
    attribute packing and scale folding are covered too.

    `spec` describes the 3-D form (nkx, nky, nkz, *trail), so it carries
    3 + (x.ndim - 1) axes; validate_flat_spec collapses the leading three
    into the single flat-k axis."""
    mesh = _mesh()
    n_trail = x_flat.ndim - 1
    spec = P(None, None, None, *([None] * n_trail))
    fn = make_flat_k_fft_ffi(mesh, kgrid, spec, kind=kind, norm=norm,
                             out_spec=None)
    return np.asarray(jax.jit(fn)(jnp.asarray(x_flat)))


def _rand(shape, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) +
            1j * rng.standard_normal(shape)).astype(np.complex128)


# ---------------------------------------------------------------------------
# HAZARD 1 — plain correctness and the three norm conventions.
# ---------------------------------------------------------------------------
@needs_fft_ffi
@pytest.mark.parametrize("kind", ["fftn", "ifftn"])
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_matches_numpy_scalar_trail(kind, norm):
    """T == 1: the degenerate batch.  Catches sign convention and norm."""
    kgrid = (4, 4, 4)
    x = _rand((64, 1), seed=11)
    got = _run(x, kgrid, kind, norm)
    want = _reference(x, kgrid, kind, norm)
    np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# HAZARD 2 — the STRIDED/BATCHED descriptor.  istride = T, idist = 1.
# If the plan is built with istride/idist swapped (the classic error) the
# T == 1 test above still passes and every one of these fails.
# ---------------------------------------------------------------------------
@needs_fft_ffi
@pytest.mark.parametrize("trail", [(2,), (7,), (3, 5), (2, 2, 2)])
@pytest.mark.parametrize("kind", ["fftn", "ifftn"])
def test_batched_trail_matches_numpy(trail, kind):
    kgrid = (4, 4, 4)
    x = _rand((64, *trail), seed=23)
    got = _run(x, kgrid, kind, "backward")
    want = _reference(x, kgrid, kind, "backward")
    np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL)


@needs_fft_ffi
def test_batched_trail_is_not_accidentally_broadcast():
    """Negative control for HAZARD 2.

    If the handler ignored the trail and transformed only the first column,
    broadcasting it, the previous test would still pass on data whose trail
    columns happen to be similar.  Here every trail column is a DIFFERENT
    signal, so a broadcast or a stride collapse is unmissable: each column's
    transform must differ from every other column's."""
    kgrid = (4, 4, 4)
    T = 5
    x = np.zeros((64, T), dtype=np.complex128)
    for t in range(T):
        x[:, t] = _rand((64,), seed=100 + t)
    got = _run(x, kgrid, "fftn", "backward")
    want = _reference(x, kgrid, "fftn", "backward")
    np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL)
    for t in range(1, T):
        assert not np.allclose(got[:, 0], got[:, t], atol=1e-8), (
            f"trail column {t} equals column 0 — the batch dimension "
            f"collapsed (stride/distance descriptor error)")


# ---------------------------------------------------------------------------
# HAZARD 4 — howmany > 1 AND a non-cubic k grid.  With nkx != nky != nkz the
# three k strides are all different, so a descriptor that assumes a cubic or
# contiguous k block produces a wrong-but-plausible answer.
# ---------------------------------------------------------------------------
@needs_fft_ffi
@pytest.mark.parametrize("kgrid", [(2, 3, 4), (4, 3, 2), (3, 5, 2), (6, 1, 2)])
@pytest.mark.parametrize("kind", ["fftn", "ifftn"])
def test_nonuniform_kgrid_batched(kgrid, kind):
    nk = kgrid[0] * kgrid[1] * kgrid[2]
    x = _rand((nk, 3), seed=37)
    got = _run(x, kgrid, kind, "backward")
    want = _reference(x, kgrid, kind, "backward")
    np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL)


@needs_fft_ffi
def test_k_axis_order_is_row_major_xyz():
    """Pins the k-index convention: flat index k == (kx*nky + ky)*nkz + kz.

    A handler that transposed the k axes (e.g. treated the grid as z-major)
    would still return the right MULTISET of values for a cubic grid, which
    no norm or round-trip check can see.  A non-cubic grid plus an impulse
    at a known (kx,ky,kz) pins the order exactly."""
    kgrid = (2, 3, 4)
    nkx, nky, nkz = kgrid
    nk = nkx * nky * nkz
    x = np.zeros((nk, 1), dtype=np.complex128)
    kx, ky, kz = 1, 2, 3
    x[(kx * nky + ky) * nkz + kz, 0] = 1.0
    got = _run(x, kgrid, "fftn", "backward")
    want = _reference(x, kgrid, "fftn", "backward")
    np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Round trip — independent of the reference, so it survives even if the
# numpy convention argument above is somehow mis-stated.
# ---------------------------------------------------------------------------
@needs_fft_ffi
@pytest.mark.parametrize("kgrid", [(4, 4, 4), (2, 3, 4)])
@pytest.mark.parametrize("trail", [(1,), (4,)])
def test_round_trip_is_identity(kgrid, trail):
    nk = kgrid[0] * kgrid[1] * kgrid[2]
    x = _rand((nk, *trail), seed=5)
    fwd = _run(x, kgrid, "fftn", "backward")
    back = _run(fwd, kgrid, "ifftn", "backward")
    np.testing.assert_allclose(back, x, atol=1e-11, rtol=1e-10)


@needs_fft_ffi
def test_ortho_round_trip_preserves_norm():
    """Parseval under 'ortho' — a scale error that a round trip would cancel
    (because fwd and back scales multiply to 1) shows up here."""
    kgrid = (4, 4, 4)
    x = _rand((64, 3), seed=9)
    fwd = _run(x, kgrid, "fftn", "ortho")
    assert np.isclose(np.linalg.norm(fwd), np.linalg.norm(x),
                      rtol=1e-11, atol=1e-12)


# ---------------------------------------------------------------------------
# HAZARD 3 — in-place aliasing.  make_flat_k_fft_ffi declares
# input_output_aliases={0: 0}, so when the operand is dead XLA hands the
# handler the SAME buffer for input and output.  A handler that writes its
# output while still reading the input produces garbage ONLY in this mode.
# ---------------------------------------------------------------------------
@needs_fft_ffi
@pytest.mark.parametrize("trail", [(1,), (5,)])
def test_inplace_alias_matches_out_of_place(trail):
    """Force the alias by donating the operand, and compare against the
    non-donated result computed from an independent copy."""
    kgrid = (4, 4, 4)
    x = _rand((64, *trail), seed=77)
    want = _reference(x, kgrid, "fftn", "backward")

    mesh = _mesh()
    spec = P(None, None, None, *([None] * len(trail)))
    fn = make_flat_k_fft_ffi(mesh, kgrid, spec, kind="fftn", norm="backward",
                             out_spec=None)
    donated = jax.jit(fn, donate_argnums=0)
    got = np.asarray(donated(jnp.asarray(x)))
    np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Contract refusals — the handler must reject what it cannot do, loudly.
# ---------------------------------------------------------------------------
@needs_fft_ffi
def test_refuses_wrong_leading_extent():
    kgrid = (4, 4, 4)
    x = _rand((63, 1), seed=3)          # 63 != 4*4*4
    with pytest.raises(ValueError, match="leading extent"):
        _run(x, kgrid, "fftn", "backward")


@needs_fft_ffi
def test_refuses_non_complex128():
    mesh = _mesh()
    fn = make_flat_k_fft_ffi(mesh, (4, 4, 4), P(None, None, None, None),
                             kind="fftn", norm="backward", out_spec=None)
    with pytest.raises(TypeError, match="complex128"):
        fn(jnp.zeros((64, 1), dtype=jnp.complex64))
