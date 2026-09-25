"""``common.fourier_plan.LocalFourierPlan`` against numpy's FFT.

Runs on the default backend (CPU or one GPU).  Both per-axis backends are
exercised on every platform: ``device_kind='__gemm__'`` forces the stored-matrix
GEMM on every axis, ``'__fft__'`` (an unknown device) the library FFT with its
embedding gather and restriction take.  The contract is value-level:
relative error ≤ 1e-12 in complex128 against ``np.fft`` on the embedded input.
"""

import zlib

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from common import fourier_plan
from common.fourier_plan import LocalFourierPlan, dft_matrix

RTOL = 1e-12
SIZES = list(range(2, 33)) + [36, 45, 48, 54, 60, 64, 72, 75, 80, 96, 100, 125, 128,
                              150, 180, 216, 256]
BACKENDS = ("__gemm__", "__fft__")


LEGS = ("xla", "ffi") if jax.default_backend() == "gpu" else ("xla",)


@pytest.fixture(autouse=True, params=LEGS)
def leg(request, monkeypatch):
    """Every test runs on each leg the platform has: XLA ops, and on CUDA the
    one-custom-call leg (the bundle's ``lorrax_fourier_plan_mathdx``; a library
    without it fails here, never skips)."""
    if request.param == "ffi":
        from ffi.common import ffi_loader
        ok, why = ffi_loader.probe_target("lorrax_fourier_plan_mathdx", "CUDA")
        assert ok, why
    monkeypatch.setattr(fourier_plan, "_default_leg", lambda: request.param)
    return request.param


@pytest.fixture(autouse=True)
def _forced_gemm_kind(monkeypatch):
    monkeypatch.setitem(fourier_plan.GEMM_CROSSOVER, "__gemm__", (range(1 << 30),) * 2)


def _rng(*key):
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def _crandn(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _reference(x_c, extents, axes, sign, norm, in_sup, out_sup):
    """Embed, ``np.fft.fftn``/``ifftn``, restrict — independent of the plan."""
    ndim = x_c.ndim
    axes_p = [a % ndim for a in axes]
    shape = list(x_c.shape)
    for n, a in zip(extents, axes_p):
        shape[a] = n
    full = np.zeros(shape, np.complex128)
    grids = []
    for n, a, a_raw in zip(extents, axes_p, axes):
        grids.append(np.asarray(in_sup.get(a_raw, np.arange(n))) % n)
    # scatter-add so a repeated input index sums, as the embedding does
    ix = np.ix_(*[np.arange(s) if a not in axes_p else grids[axes_p.index(a)]
                  for a, s in enumerate(x_c.shape)])
    np.add.at(full, ix, x_c)
    f = np.fft.fftn if sign < 0 else np.fft.ifftn
    y = f(full, axes=axes_p, norm=norm)
    for n, a, a_raw in zip(extents, axes_p, axes):
        if a_raw in out_sup:
            y = np.take(y, np.asarray(out_sup[a_raw]) % n, axis=a)
    return y


def _check(x_c, extents, axes, *, sign, norm="backward", in_sup=None, out_sup=None,
           kind="__fft__", jit=True):
    in_sup, out_sup = in_sup or {}, out_sup or {}
    plan = LocalFourierPlan(extents, axes, sign=sign, norm=norm, in_support=in_sup,
                            out_support=out_sup, device_kind=kind)
    run = jax.jit(plan) if jit else plan
    y = np.asarray(run(jnp.asarray(x_c)))
    ref = _reference(x_c, extents, axes, sign, norm, in_sup, out_sup)
    assert y.shape == ref.shape
    err = np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-300)
    assert err <= RTOL, (extents, axes, sign, norm, kind, err)
    return plan


@pytest.mark.parametrize("kind", BACKENDS)
@pytest.mark.parametrize("n", SIZES)
def test_full_1d_every_size(n, kind):
    rng = _rng("1d", n)
    for sign in (-1, 1):
        _check(_crandn(rng, (7, n)), (n,), (-1,), sign=sign, kind=kind)
        _check(_crandn(rng, (n, 5)), (n,), (0,), sign=sign, kind=kind)


@pytest.mark.parametrize("kind", BACKENDS)
@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_norm_and_sign_match_jnp_fft(sign, norm, kind):
    rng = _rng("norm", sign, str(norm))
    x = _crandn(rng, (3, 6, 10, 9))
    _check(x, (6, 10, 9), (1, 2, 3), sign=sign, norm=norm, kind=kind)
    jf = jnp.fft.fftn if sign < 0 else jnp.fft.ifftn
    plan = LocalFourierPlan((6, 10, 9), (1, 2, 3), sign=sign, norm=norm, device_kind=kind)
    ref = np.asarray(jf(jnp.asarray(x), axes=(1, 2, 3), norm=norm))
    assert np.linalg.norm(np.asarray(plan(jnp.asarray(x))) - ref) <= RTOL * np.linalg.norm(ref)


@pytest.mark.parametrize("kind", BACKENDS)
@pytest.mark.parametrize("extents", [(8, 8, 8), (3, 5, 7), (11, 13, 17), (24, 24, 24),
                                     (6, 6, 1), (19, 23), (54, 54), (72, 80), (45, 60)])
def test_full_composites(extents, kind):
    rng = _rng("comp", extents)
    d = len(extents)
    axes = tuple(range(-d, 0))
    for sign in (-1, 1):
        _check(_crandn(rng, (3,) + extents), extents, axes, sign=sign, kind=kind)
    # transformed axes not trailing, batch axes on both sides
    x = _crandn(rng, (2,) + extents + (3,))
    _check(x, extents, tuple(range(1, d + 1)), sign=-1, norm="ortho", kind=kind)


def _supports(n, rng):
    """Contiguous, centred-wrap (the G-sphere projection) and random sets."""
    k = max(1, n // 2)
    half = k // 2
    return {
        "contiguous": np.arange(1, 1 + k) % n,
        "centred_wrap": np.arange(-half, k - half) % n,
        "random": np.sort(rng.choice(n, size=k, replace=False)),
        "single": np.array([n - 1]),
    }


@pytest.mark.parametrize("kind", BACKENDS)
@pytest.mark.parametrize("extents", [(12, 10), (54, 54), (17, 23), (24, 24, 24), (9, 16, 25),
                                     (80, 72)])
def test_sparse_supports(extents, kind):
    rng = _rng("sparse", extents)
    d = len(extents)
    axes = tuple(range(-d, 0))
    for label in ("contiguous", "centred_wrap", "random", "single"):
        sups = [_supports(n, rng)[label] for n in extents]
        in_sup = {a: s for a, s in zip(axes, sups)}
        x = _crandn(rng, (4,) + tuple(len(s) for s in sups))
        for sign in (-1, 1):
            # sphere -> box (input restricted on every axis)
            _check(x, extents, axes, sign=sign, in_sup=in_sup, kind=kind)
            # box -> sphere (output restricted on every axis)
            _check(_crandn(rng, (4,) + extents), extents, axes, sign=sign,
                   out_sup=in_sup, kind=kind)
        # mixed: first axis restricted on input, last on output, different sets
        out_last = {axes[-1]: _supports(extents[-1], rng)["random"]}
        x_m = _crandn(rng, (3, len(sups[0])) + extents[1:])
        _check(x_m, extents, axes, sign=-1, in_sup={axes[0]: sups[0]}, out_sup=out_last,
               kind=kind)


@pytest.mark.parametrize("kind", BACKENDS)
def test_single_plane_waves(kind):
    n = (15, 16, 21)
    rng = _rng("pw")
    sup = {a: _supports(m, rng)["centred_wrap"] for a, m in zip((0, 1, 2), n)}
    shape = tuple(len(s) for s in sup.values())
    plan = LocalFourierPlan(n, (0, 1, 2), sign=1, norm="forward", in_support=sup,
                            device_kind=kind)
    r = np.meshgrid(*[np.arange(m) for m in n], indexing="ij")
    for _ in range(4):
        g = tuple(int(rng.integers(s)) for s in shape)
        x = np.zeros(shape, np.complex128)
        x[g] = 1.0
        y = np.asarray(plan(jnp.asarray(x)))
        G = [int(sup[a][g[a]]) for a in range(3)]
        expect = np.exp(2j * np.pi * sum(G[a] * r[a] / n[a] for a in range(3)))
        assert np.max(np.abs(y - expect)) <= RTOL


@pytest.mark.parametrize("kind", BACKENDS)
def test_round_trip_sphere_box_sphere(kind):
    n = (24, 20, 27)
    rng = _rng("rt")
    sup = {a: _supports(m, rng)["centred_wrap"] for a, m in zip((1, 2, 3), n)}
    x = _crandn(rng, (5,) + tuple(len(s) for s in sup.values()))
    to_r = LocalFourierPlan(n, (1, 2, 3), sign=1, norm="backward", in_support=sup,
                            device_kind=kind)
    to_g = LocalFourierPlan(n, (1, 2, 3), sign=-1, norm="backward", out_support=sup,
                            device_kind=kind)
    back = np.asarray(jax.jit(lambda v: to_g(to_r(v)))(jnp.asarray(x)))
    assert np.linalg.norm(back - x) <= RTOL * np.linalg.norm(x)


@pytest.mark.parametrize("kind", BACKENDS)
def test_exact_output_order(kind):
    n = 18
    rng = _rng("order")
    idx = np.array([5, 0, 17, 5, 9, -1, 36])     # unsorted, repeated, negative, wrapped
    x = _crandn(rng, (6, n))
    plan = _check(x, (n,), (1,), sign=-1, out_sup={1: idx}, kind=kind)
    assert plan.stages == [(1, kind.strip("_"), n, idx.size)]
    y = np.asarray(plan(jnp.asarray(x)))
    np.testing.assert_allclose(y, np.fft.fft(x, axis=1)[:, idx % n], rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", BACKENDS)
@pytest.mark.parametrize("batch", [1, 2, 3, 7, 13, 64, 257])
def test_batch_tails(batch, kind):
    rng = _rng("tail", batch)
    sup = {-2: np.arange(-3, 4) % 16}
    x = _crandn(rng, (batch, 7, 9))
    _check(x, (16, 9), (-2, -1), sign=1, in_sup=sup, kind=kind)
    _check(_crandn(rng, (batch, 16, 9)), (16, 9), (-2, -1), sign=-1, kind=kind, jit=False)


def test_stage_order_and_backend_choice(monkeypatch):
    sup_in, sup_out = np.arange(4), np.arange(3)
    kw = dict(sign=-1, in_support={0: sup_in}, out_support={2: sup_out})
    plan = LocalFourierPlan((8, 10, 12), (0, 1, 2), device_kind="__gemm__", **kw)
    assert plan.stages == [(2, "gemm", 12, 3), (1, "gemm", 10, 10), (0, "gemm", 4, 8)]
    plan = LocalFourierPlan((8, 10, 12), (0, 1, 2), device_kind="__fft__", **kw)
    assert plan.stages == [(0, "fft", 4, 8), (1, "fft", 10, 10), (2, "fft", 12, 3)]
    assert [op[0] for op in plan._ops] == ["embed", "fft", "take"]
    # a mixed table: supported axes up to 8 on the GEMM, full axes on the FFT
    monkeypatch.setitem(fourier_plan.GEMM_CROSSOVER, "__mixed__", (range(0), range(9)))
    plan = LocalFourierPlan((8, 10, 12), (0, 1, 2), device_kind="__mixed__", **kw)
    assert plan.stages == [(1, "fft", 10, 10), (2, "fft", 12, 3), (0, "gemm", 4, 8)]
    assert fourier_plan.gemm_crossover("an unknown accelerator") == (range(0), range(0), 1.0, range(0))
    # the A100 row: supported plane axes on the GEMM, a full axis on the FFT
    sup = {1: np.arange(-13, 14) % 54, 2: np.arange(27)}
    x = _crandn(_rng("a100"), (2, 27, 27, 9))
    plan = _check(x, (54, 54, 9), (1, 2, 3), sign=-1, in_sup=sup,
                  kind="NVIDIA A100-SXM4-40GB")
    assert plan.stages == [(3, "fft", 9, 9), (2, "gemm", 27, 54), (1, "gemm", 27, 54)]
    # a support that lists the whole axis in order is a full axis (the FFT on A100)
    full = LocalFourierPlan((54, 80), (-2, -1), sign=-1, out_support={-2: np.arange(54), -1: np.arange(80)},
                            device_kind="NVIDIA A100-SXM4-40GB")
    assert [s[1] for s in full.stages] == ["fft", "fft"]
    # a supported axis takes the A100 GEMM only while n_in·n_out/N² ≤ 0.54
    frac = LocalFourierPlan((80, 80), (-2, -1), sign=+1, in_support={-2: np.arange(43), -1: np.arange(44)},
                            device_kind="NVIDIA A100-SXM4-40GB")
    assert sorted((s[0], s[1]) for s in frac.stages) == [(-2, "gemm"), (-1, "fft")]
    # a 1- or 2-axis plan's supported axis takes the A100 GEMM from N = 32; a 3-axis one from 16
    small = {-2: np.arange(12), -1: np.arange(12)}
    two = LocalFourierPlan((24, 24), (-2, -1), sign=-1, out_support=small,
                           device_kind="NVIDIA A100-SXM4-40GB")
    three = LocalFourierPlan((24, 24, 24), (-3, -2, -1), sign=-1, out_support={**small, -3: np.arange(12)},
                             device_kind="NVIDIA A100-SXM4-40GB")
    assert [s[1] for s in two.stages] == ["fft", "fft"]
    assert [s[1] for s in three.stages] == ["gemm"] * 3


@pytest.mark.parametrize("kind", BACKENDS)
def test_out_perm_and_transpose_free_route(kind):
    """``out_perm`` transposes the result; from a (Kc, B, Kb) input the GEMM
    chain writes (B, n_b, n_c) directly, with no transpose at all."""
    n_b, n_c = 18, 20
    sb, sc = np.arange(-4, 5) % n_b, np.arange(-5, 5) % n_c
    rng = _rng("perm", kind)
    x = _crandn(rng, (sc.size, 6, sb.size))                     # (Kc, B, Kb)
    plan = LocalFourierPlan((n_b, n_c), (2, 0), sign=-1, in_support={2: sb, 0: sc},
                            out_perm=(1, 2, 0), device_kind=kind)
    y = np.asarray(jax.jit(plan)(jnp.asarray(x)))
    ref = _reference(x, (n_b, n_c), (2, 0), -1, "backward", {2: sb, 0: sc}, {})
    ref = np.transpose(ref, (1, 2, 0))
    assert y.shape == (6, n_b, n_c)
    assert np.linalg.norm(y - ref) <= RTOL * np.linalg.norm(ref)
    if kind == "__gemm__":
        assert plan._route(x.shape, [1, 2, 0])[0] == 0
        # from the natural (B, Kb, Kc) layout the cheapest route still moves data
        plan2 = LocalFourierPlan((n_b, n_c), (1, 2), sign=-1, in_support={1: sb, 2: sc},
                                 device_kind=kind)
        assert 0 < plan2._route((6, sb.size, sc.size), [0, 1, 2])[0] < 6 * n_b * n_c


def test_plan_inside_shard_map():
    """The plan is a local kernel: constants only, no collectives, any mesh."""
    from jax.sharding import Mesh, PartitionSpec as P
    from common.shard_map import shard_map
    devs = np.array(jax.devices())
    mesh = Mesh(devs, ("b",))
    n, sup = (12, 10), {-1: np.arange(-2, 3) % 10}
    rng = _rng("shmap")
    x = _crandn(rng, (2 * devs.size, 12, 5))
    for kind in BACKENDS:
        plan = LocalFourierPlan(n, (-2, -1), sign=1, norm="ortho", in_support=sup,
                                device_kind=kind)
        f = jax.jit(shard_map(plan, mesh=mesh, in_specs=P("b"), out_specs=P("b")))
        y = np.asarray(f(jnp.asarray(x)))
        ref = _reference(x, n, (-2, -1), 1, "ortho", sup, {})
        assert np.linalg.norm(y - ref) <= RTOL * np.linalg.norm(ref)


def test_dft_matrix_exact_phase_reduction():
    n = 97
    A = dft_matrix(n, np.arange(n), np.arange(n), sign=-1)
    ref = np.fft.fft(np.eye(n), axis=0)
    assert np.max(np.abs(A - ref)) <= 1e-13
    # huge indices reduce exactly: (j + 10^12 n) behaves as j
    B = dft_matrix(n, np.arange(n) + 10**12 * n, np.arange(n), sign=-1)
    assert np.array_equal(A, B)


def test_refusals():
    plan = LocalFourierPlan((8,), (-1,), sign=-1, in_support={-1: np.arange(3)},
                            device_kind="__fft__")
    with pytest.raises(ValueError, match="extent"):
        plan(jnp.zeros((2, 8), jnp.complex128))
    with pytest.raises(TypeError, match="dtype"):
        plan(jnp.zeros((2, 3), jnp.complex64))
    with pytest.raises(ValueError, match="keys"):
        LocalFourierPlan((8,), (-1,), sign=-1, out_support={0: np.arange(2)})
    with pytest.raises(ValueError, match="sign"):
        LocalFourierPlan((8,), (-1,), sign=0)
    with pytest.raises(ValueError, match="repeats"):
        LocalFourierPlan((8,), (-1,), sign=-1, in_support={-1: np.array([1, 9])})


def test_in_gather_plane_from_cylinder():
    """The cylinder-plane form (the route-G ζ planes): x holds the occupied
    cells of an (n_b, n_c) plane; plain and slab forms against numpy."""
    from jax.sharding import Mesh
    nb, nc = 24, 20
    h = lambda n: np.where(np.arange(n) < n // 2, np.arange(n), np.arange(n) - n)
    occ = (h(nb)[:, None] ** 2 + h(nc)[None, :] ** 2 < 40).ravel()
    n_col = int(occ.sum())
    pfc = np.full(nb * nc, n_col, np.int32)
    pfc[occ] = np.arange(n_col)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    plan = LocalFourierPlan((nb, nc), (-2, -1), sign=-1, norm="backward",
                            in_gather=(pfc, n_col), mesh=mesh)
    rng = _rng("gather")
    Fa = _crandn(rng, (2, 5, 3, n_col))
    plane = np.where(pfc < n_col, Fa[..., np.clip(pfc, 0, n_col - 1)], 0)
    ref = np.fft.fftn(plane.reshape(Fa.shape[:-1] + (nb, nc)), axes=(-2, -1))
    y = np.asarray(jax.jit(plan)(jnp.asarray(Fa)))
    assert np.linalg.norm(y - ref) <= RTOL * np.linalg.norm(ref)
    ys = np.asarray(jax.jit(lambda a, s: plan(a, s, 2))(jnp.asarray(Fa), jnp.int32(2)))
    assert np.linalg.norm(ys - ref[:, 2:4]) <= RTOL * np.linalg.norm(ref[:, 2:4])
    with pytest.raises(ValueError, match="in_gather"):
        LocalFourierPlan((nb, nc), (-2, -1), sign=1, in_gather=(pfc, n_col), mesh=mesh)


# ---- audit 2026-09-25 (reports/new_fft_kernel/audit_portability_safety.md) ----

def test_contract_refused_by_name():
    """complex128 over 1..3 axes on every platform (finding 6): the CUDA leg serves
    exactly that, so a plan outside it refuses at construction, not at run time."""
    with pytest.raises(ValueError, match="GATE fourier-plan-contract"):
        LocalFourierPlan((8,), (-1,), sign=-1, dtype=jnp.complex64)
    with pytest.raises(ValueError, match="GATE fourier-plan-contract"):
        LocalFourierPlan((2, 2, 2, 2), (0, 1, 2, 3), sign=-1)


def test_plan_loads_its_cuda_target(monkeypatch):
    """A plan built where CUDA can lower it probes (loads) lorrax_fourier_plan_mathdx itself
    (finding 5): standalone use must not depend on another factory having loaded it."""
    import ffi.fft as F
    seen = []
    monkeypatch.setattr(fourier_plan, "_cuda_present", lambda mesh: True)
    monkeypatch.setattr(F, "_require_target", lambda target, platform: seen.append((target, platform)))
    LocalFourierPlan((8,), (-1,), sign=-1, device_kind="__fft__")
    assert seen == [(F.FOURIER_PLAN_TARGET, "CUDA")], seen


@pytest.mark.parametrize("sign", (-1, 1))
def test_fft_axes_around_a_gemm_axis(sign):
    """A GEMM axis between two FFT axes (finding 2): the A100 row takes the
    supported middle axis (N=32) as a GEMM and the full outer axes (N=8) as FFTs,
    so the CUDA leg's FFT group is two runs."""
    rng = _rng("noncontig", sign)
    sup = {1: np.arange(-5, 6) % 32}
    plan = _check(_crandn(rng, (3, 8, 11, 8)), (8, 32, 8), (1, 2, 3), sign=sign,
                  in_sup={2: sup[1]}, kind="NVIDIA A100-SXM4-40GB")
    assert [(ax, how) for ax, how, *_ in plan.stages] == [(1, "fft"), (3, "fft"), (2, "gemm")], plan.stages


def test_cpu_operand_in_a_gpu_process(monkeypatch):
    """The leg follows the platform the call is lowered for (finding 4): a CPU
    operand in a GPU process takes the XLA leg instead of failing NOT_FOUND."""
    if jax.default_backend() != "gpu":
        pytest.skip("needs a GPU process")
    monkeypatch.setattr(fourier_plan, "_default_leg", lambda: None)
    cpu = jax.devices("cpu")[0]
    rng = _rng("cpuop")
    x = _crandn(rng, (4, 16, 12))
    plan = LocalFourierPlan((16, 12), (-2, -1), sign=-1, device_kind="__fft__")
    y = jax.jit(plan)(jax.device_put(jnp.asarray(x), cpu))
    assert list(y.devices())[0].platform == "cpu"
    np.testing.assert_allclose(np.asarray(y), np.fft.fftn(x, axes=(-2, -1)), rtol=0, atol=1e-11)


def _device_free_bytes():
    import ctypes
    rt = ctypes.CDLL("libcudart.so.13")
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    assert rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0
    return int(free.value)


def test_cufft_work_area_comes_from_the_pool(leg):
    """cuFFT plans cached per batch size must not hold work areas outside XLA's
    pool (finding 3): a Bluestein length (4099) at batch ~300 held ~84 MB per new
    batch size.  Three new batch sizes may not take more than 16 MB of free
    device memory."""
    if leg != "ffi":
        pytest.skip("the CUDA leg's cuFFT plans")
    plan = LocalFourierPlan((4099,), (-1,), sign=-1, device_kind="__fft__")
    run = jax.jit(plan)
    rng = _rng("bluestein")
    jax.block_until_ready(run(jnp.asarray(_crandn(rng, (300, 4099)))))
    before = _device_free_bytes()
    for b in (301, 302, 303):
        x = _crandn(rng, (b, 4099))
        y = np.asarray(jax.block_until_ready(run(jnp.asarray(x))))
        assert np.max(np.abs(y - np.fft.fft(x, axis=-1))) <= 1e-9 * np.max(np.abs(y))
    assert before - _device_free_bytes() <= 16 << 20, (before, _device_free_bytes())
