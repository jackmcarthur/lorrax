"""The k-convolution router: one factory per k-axis operation, one backend per platform.

Every k-axis convolution and k-axis transform the physics needs is asked for
through a factory here (or its ``common.fft_helpers`` re-export).  The factory
chooses the backend from the mesh's device vendor only, never from an
environment variable (``docs/architecture/decisions.md#xla-reference``):

    CUDA   nvidia-mathdx: cuFFTDx thread FFTs inside one fused shared-memory
           pass per k-row, NVRTC-built per (mode, k-grid) and disk-cached
           (``cpp/cufft/kconv_mathdx_cuda_ffi.cc``).  Kept because it is
           decisive on memory (``docs/architecture/kconv.md#why-fused``).
    cpu    the plan backend: the reference composition (the same unfolds,
           products and spin sums in XLA) with the host FFTW3-ABI flat-k
           transforms and fused Σ convolution (``cpp/fftw/``).  Kept because it
           is 3.8-7.4x faster than ``jnp.fft`` at a CrI3 rank tile.
    other  the XLA backend: the same composition with ``jnp.fft`` along the k
           axes.  It is the reference every vendor route is gated against on
           the same device (:func:`xla_reference`, ``tests/test_kconv_xla_gate.py``).

    factory                   layout         operation
    ------------------------  -------------  ----------------------------------------
    make_fused_conv_kpair     3-D leading    ISDF CCT/ZCT post-pair convolution
    make_fused_conv_kparent   parent tables  the same with the typed parent load
    make_fused_conv_kplane    route-G planes the same read from the D-plane FFT output,
                                             Bloch phase and L/R split applied on load
    make_kconv_klead          flat leading   Σ / COHSEX  fftn(ifftn(T)·ifftn(W))
    make_kconv_klead_unfold   parent Green   the Σ one read from the raw-parent G with
                                             the typed unfold and spin action on load,
                                             stored at the caller's k rows only
    make_kconv_lorentz_unfold parent G + W   the four-current Σ: the same load for G and
                                             for W's irreducible-q parent tile, then the
                                             γ_i Ĝ γ_j† · Ŵ_ij block sum in R space
    make_kfft_klead_unfold    wedge interaction  make_kconv_klead's prep (ifftn into R
                                             space) read from the q wedge
    make_kconv_kminor         trailing       BSE rung    fftn(ifftn(X)·K_R)
    make_kfft_klead / _local  flat leading   one transform
    make_kfft_kminor / _local trailing       one transform

Pick the factory whose k position matches the tile you already hold; a caller
does not transpose to reach another.  On CUDA a grid the mathdx family cannot
serve on this device (:func:`mathdx_refusal`: an axis above ``KCONV_AXIS_MAX``,
a split-arm plane tile beyond the opt-in shared memory, a failed probe compile)
takes the XLA backend, with one warning.
Contract: ``docs/architecture/kconv.md``.
"""

from __future__ import annotations

import contextlib
import math
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

# ``common.fft_helpers`` is imported inside the
# function bodies that use them: importing any ``common`` submodule runs
# ``common/__init__.py``, which imports ``common.fft_helpers``, which imports
# this module, so a module-scope import here would close an import cycle.

__all__ = [
    "ffi_fft_scale", "validate_flat_spec", "xla_reference",
    # The k-convolution router: CUDA -> nvidia-mathdx, cpu -> the plan backend, else XLA.
    "KCONV_PAIR_TARGET", "KCONV_PARENT_TARGET", "KCONV_KLEAD_TARGET",
    "KFFT_KLEAD_TARGET", "KCONV_KMINOR_TARGET", "KFFT_KMINOR_TARGET",
    "KCONV_TARGETS", "KCONV_AXIS_MAX",
    "kconv_backend", "require_kconv", "mathdx_root", "cubin_cache_dir", "conv_kpair_scale",
    "make_fused_conv_kpair", "make_fused_conv_kparent", "make_fused_conv_kplane",
    "KCONV_PLANE_TARGET",
    "KConvStored", "make_kconv_klead", "make_kconv_klead_unfold", "KCONV_KLEAD_UNFOLD_TARGET",
    "klead_unfold_scratch_bytes",
    "make_kconv_lorentz_unfold", "KCONV_KLEAD_LORENTZ_TARGET",
    "make_kfft_klead_unfold", "KFFT_KLEAD_UNFOLD_TARGET", "live_row_mask",
    "make_kconv_chi_unfold", "KCONV_CHI_UNFOLD_TARGET",
    "chi_unfold_scratch_bytes", "make_kconv_chi_vertex", "KCONV_CHI_VERTEX_TARGET",
    "make_kconv_kminor", "kconv_kminor_out_shape",
    "make_kfft_klead", "make_kfft_kminor",
    "make_local_kfft_klead", "make_local_kfft_kminor", "make_local_kconv_kminor",
    "make_local_kconv_klead",
    "KCONV_KLEAD_OUTER_TARGET", "klead_outer_refusal", "make_local_kconv_klead_outer",
    "KCONV_KLEAD_OUTER_DECODE_TARGET", "klead_outer_decode_refusal", "make_local_kconv_klead_outer_decode",
    "PLANE_FFT_GATHER_TARGET", "plane_fft_split", "plane_resident_bytes", "make_plane_fft_gather",
]

#: The NVIDIA k-convolution family on nvidia-mathdx (the router's CUDA leg).
KCONV_PAIR_TARGET = "lorrax_mathdx_kconv_pair"
KCONV_PARENT_TARGET = "lorrax_mathdx_kconv_parent"
KCONV_PLANE_TARGET = "lorrax_mathdx_kconv_plane"
KCONV_KLEAD_TARGET = "lorrax_mathdx_kconv_klead"
#: Mode 2 with its T formed on the load as a rank-K outer-product sum (the BSE W term's encode):
#: :func:`make_local_kconv_klead_outer`, cpp/cufft/kconv_outer_cuda_ffi.cc.
KCONV_KLEAD_OUTER_TARGET = "lorrax_mathdx_kconv_klead_outer"
#: The outer load with its K-sum pipe chosen (``LORRAX_BSE_OUTER_KSUM=fma``, A/B; additive).
KCONV_KLEAD_OUTER_KSUM_TARGET = "lorrax_mathdx_kconv_klead_outer_ksum"
#: The same load with the BSE decode's (t, μ) contraction fused into the store (U never stored):
#: :func:`make_local_kconv_klead_outer_decode`.
KCONV_KLEAD_OUTER_DECODE_TARGET = "lorrax_mathdx_kconv_klead_outer_decode"
#: Modes 7/8 on the parent rows, with the conj-on-load partner and (mode 7) the output spin block.
KCONV_KLEAD_UNFOLD_TARGET = "lorrax_mathdx_kconv_klead_unfold_xblock"
#: Mode 8 with W read from its irreducible-q parent tile on the load (the library keeps the V_R
#: targets for older trees; nothing here calls them).
KCONV_KLEAD_LORENTZ_TARGET = "lorrax_mathdx_kconv_klead_lorentz_wparent"
KFFT_KLEAD_UNFOLD_TARGET = "lorrax_mathdx_kfft_klead_unfold"
#: Mode 11, the chi0 pass read from the raw-parent Green pair (:func:`make_kconv_chi_unfold`).
KCONV_CHI_UNFOLD_TARGET = "lorrax_mathdx_kconv_chi_unfold"
#: Mode 11 with the four-current channel vertices (:func:`make_kconv_chi_vertex`).
KCONV_CHI_VERTEX_TARGET = "lorrax_mathdx_kconv_chi_vertex"
KFFT_KLEAD_TARGET = "lorrax_mathdx_kfft_klead"
KCONV_KMINOR_TARGET = "lorrax_mathdx_kconv_kminor"
KFFT_KMINOR_TARGET = "lorrax_mathdx_kfft_kminor"
#: Mode 10, the route-G plane FFT with gather-on-load (:func:`make_plane_fft_gather`).
PLANE_FFT_GATHER_TARGET = "lorrax_mathdx_plane_fft_gather"
#: Every mathdx target; ``require_kconv`` checks them all at startup.
KCONV_TARGETS = (KCONV_PAIR_TARGET, KCONV_PARENT_TARGET, KCONV_PLANE_TARGET, KCONV_KLEAD_TARGET,
                 KCONV_KLEAD_OUTER_TARGET, KCONV_KLEAD_OUTER_DECODE_TARGET,
                 KCONV_KLEAD_UNFOLD_TARGET, KCONV_KLEAD_LORENTZ_TARGET,
                 KFFT_KLEAD_TARGET, KFFT_KLEAD_UNFOLD_TARGET, KCONV_CHI_UNFOLD_TARGET,
                 KCONV_KMINOR_TARGET, KFFT_KMINOR_TARGET, PLANE_FFT_GATHER_TARGET)

def ffi_fft_scale(kind: str, norm: str | None, nk: int) -> float:
    """Total scale matching jnp.fft's norm conventions exactly:
    ifftn: backward/None -> 1/N, ortho -> 1/sqrt(N), forward -> 1;
    fftn : backward/None -> 1,  ortho -> 1/sqrt(N), forward -> 1/N.

    Computed HERE, in Python, and shipped to the handler as a plain scale —
    the handlers implement no norm convention of their own."""
    if norm == 'ortho':
        return 1.0 / math.sqrt(float(nk))
    if norm in (None, 'backward'):
        return 1.0 / float(nk) if kind == 'ifftn' else 1.0
    if norm == 'forward':
        return 1.0 if kind == 'ifftn' else 1.0 / float(nk)
    raise ValueError(f"Unsupported FFT norm={norm!r}")


def validate_flat_spec(spec: P, what: str) -> P:
    """FFT axes (leading three of the 3-D form) must be replicated; return
    the equivalent flat-form spec (nk axis replicated + original trail)."""
    axes = tuple(spec)
    if len(axes) < 3 or any(ax is not None for ax in axes[:3]):
        raise ValueError(
            f"FFI flat-k backend needs the three k axes of {what} replicated "
            f"(spec {spec}); sharded FFT axes are unsupported (same contract "
            f"as the XLA-path helpers).")
    return P(None, *axes[3:])


# ===========================================================================
# THE k-CONVOLUTION ROUTER — the ISDF pair convolution
# ===========================================================================
# CUDA -> the nvidia-mathdx family; cpu -> the plan backend; every other
# platform -> the XLA backend.  All return the SAME callable contract, so a
# consumer never branches on the backend.

#: cuFFTDx fp64 thread-FFT limit: a k-grid with a longer axis takes the XLA backend.
KCONV_AXIS_MAX = 40


def conv_kpair_scale(norm: str | None, nk: int, mult: float = 1.0) -> float:
    """Fold two inverse norms and one forward norm into one scalar."""
    si = ffi_fft_scale("ifftn", norm, nk)
    return si * si * ffi_fft_scale("fftn", norm, nk) * float(mult)


def _conv_kpair_phase_codes(phase, ns: int, label: str) -> np.ndarray:
    """Encode exact monomial phases as 0:+1, 1:+i, 2:-1, 3:-i."""
    values = np.asarray(phase, dtype=np.complex128).reshape(-1)
    if values.size != ns:
        raise ValueError(f"k-conv {label} phase has {values.size} entries; ns={ns}")
    quadrants = np.asarray([1, 1j, -1, -1j], dtype=np.complex128)
    codes = np.empty(ns, dtype=np.int64)
    for i, value in enumerate(values):
        hits = np.flatnonzero(np.abs(quadrants - value) <= 1e-14)
        if hits.size != 1:
            raise ValueError(
                f"k-conv {label} phase[{i}]={value!r} is not in {{+1,+i,-1,-i}}")
        codes[i] = int(hits[0])
    return codes


def _check_perm(perm, ns: int, label: str) -> np.ndarray:
    p = np.asarray(perm, dtype=np.int64).reshape(-1)
    if p.size != ns or sorted(int(v) for v in p) != list(range(ns)):
        raise ValueError(f"k-conv {label} perm {p.tolist()} is not a permutation of range({ns})")
    return p


def mathdx_root() -> str:
    """The installed nvidia-mathdx wheel's ``nvidia/mathdx`` directory, or a GATE refusal.

    Found from the Python package spec, never from an environment variable;
    NVRTC includes ``include/`` and ``external/cutlass/include`` beneath it.
    """
    import importlib.util
    import os
    try:
        spec = importlib.util.find_spec("nvidia.mathdx")
    except ModuleNotFoundError:
        spec = None
    roots = list(spec.submodule_search_locations) if spec is not None else []
    for root in roots:
        if os.path.isfile(os.path.join(root, "include", "cufftdx.hpp")):
            return root
    raise RuntimeError(
        "GATE mathdx-headers: got no importable nvidia.mathdx with "
        "include/cufftdx.hpp; want the nvidia-mathdx wheel, the only supported "
        "k-convolution backend on NVIDIA GPUs (docs/architecture/kconv.md#router); why: "
        "the fused k-convolution kernels are compiled at run time from its "
        "cuFFTDx headers; fix: pip install nvidia-mathdx.")


#: Non-empty inside :func:`xla_reference`.
_XLA_REFERENCE: list = []


@contextlib.contextmanager
def xla_reference():
    """Build every factory inside this block on the XLA backend, on any platform.

    The reference a kept vendor route is gated against on the same device
    (``tests/test_kconv_xla_gate.py``) and timed against.  A factory decides
    its backend when it is built, so the callables it returns keep it after
    the block ends.  It is a Python argument of the caller, not an environment
    variable, so no deployment can select it by accident.
    """
    _XLA_REFERENCE.append(True)
    try:
        yield
    finally:
        _XLA_REFERENCE.pop()


#: Why the mathdx family cannot run on this job's devices: set once by
#: :func:`require_kconv` when its probe compile fails on any process.
_MATHDX_DOWN: list = []


def mathdx_refusal(kgrid, *, kminor: bool = False, optin: int | None = None) -> str:
    """Why the mathdx family cannot serve ``kgrid`` on this device ("" when it can).

    Decided from the grid and the device's opt-in shared memory per block
    (``optin``, default the attribute), so every rank of a job decides alike:
    an axis above :data:`KCONV_AXIS_MAX`; a split-arm plane tile of 16
    columns (every k-box mode's floor at ns <= 4) beyond the opt-in memory;
    for the k-minor modes 4/5, which have no split arm, one k-box column
    beyond it; or a failed probe compile at startup.
    """
    if _MATHDX_DOWN:
        return _MATHDX_DOWN[0]
    nx, ny, nz = (int(v) for v in kgrid)
    if max(nx, ny, nz) > KCONV_AXIS_MAX:
        return f"an axis above {KCONV_AXIS_MAX}, the fp64 cuFFTDx thread-FFT limit"
    have = _optin_smem_bytes() if optin is None else int(optin)
    if have is None:
        return ""
    plane, column = 16 * 16 * ((ny * (nz | 1)) | 1), 16 * ((nx * ny * (nz | 1)) | 1)
    # ponytail: one floor for every k-box mode; a single-pass mode-2/3 tile could still fit a
    # grid with nx < 8 that this sends to XLA.
    if kminor and column > have:
        return (f"one k-box column needs {column} B > {have} B of opt-in shared memory "
                "(the k-minor modes 4 and 5 have no split arm)")
    if plane > have:
        return f"a split-arm plane tile needs {plane} B > {have} B of opt-in shared memory"
    return ""


def kconv_backend(mesh: Mesh, kgrid=None, *, kminor: bool = False) -> str:
    """``'mathdx'`` on a CUDA mesh, ``'plan'`` on a cpu mesh, ``'xla'`` on every
    other platform and inside :func:`xla_reference`.  With ``kgrid`` (and
    ``kminor`` for modes 4/5), a CUDA mesh takes ``'xla'`` where
    :func:`mathdx_refusal` names a reason, announced once per grid as a warning."""
    from ffi.gate import mesh_ffi_platform
    if _XLA_REFERENCE:
        return "xla"
    backend = {"CUDA": "mathdx", "cpu": "plan"}.get(mesh_ffi_platform(mesh), "xla")
    if backend != "mathdx" or (kgrid is None and not _MATHDX_DOWN):
        return backend
    why = _MATHDX_DOWN[0] if kgrid is None else mathdx_refusal(kgrid, kminor=kminor)
    if not why:
        return backend
    from ffi.gate import announce_once
    grid = "" if kgrid is None else f" for k-grid {tuple(int(v) for v in kgrid)}"
    msg = (f"[kconv] k-convolution router: CUDA -> XLA (jnp.fft){grid}"
           f"{' (k-minor)' if kminor else ''}, not nvidia-mathdx: {why}")
    if announce_once(("kconv", "xla", None if kgrid is None else tuple(kgrid), kminor), msg):
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return "xla"


def require_kconv(mesh: Mesh, *, announce: bool = True) -> str:
    """Startup check of the router's backend on this mesh; returns it or refuses.

    CUDA: the nvidia-mathdx wheel, every family target and one probe compile;
    cpu: the two host plan targets.  The XLA backend needs no library.
    """
    from ffi.gate import announce_once, mesh_ffi_platform
    backend = kconv_backend(mesh)
    if backend == "mathdx":
        root = mathdx_root()
        for target in KCONV_TARGETS:
            _require_target(target, "CUDA")
        try:
            _probe_kconv_compile(mesh)
            why = ""
        except RuntimeError as e:
            why = str(e)
        # Every process takes the same route: one failed probe sends all to XLA.
        from jax.experimental import multihost_utils
        if int(np.max(multihost_utils.process_allgather(np.int32(bool(why))))):
            _MATHDX_DOWN[:] = [why or "a peer process failed the mathdx probe compile"]
            return kconv_backend(mesh)
        announce_once(("kconv", "backend", backend),
                      f"[kconv] k-convolution router: CUDA -> nvidia-mathdx ({root}); "
                      f"cubin cache {_cubin_cache_summary()}",
                      scope="rank0", emit=announce)
    elif backend == "plan":
        for target in (FLAT_K_TARGET, GW_CONV_TARGET):
            _require_target(target, "cpu")
        announce_once(("kconv", "backend", backend),
                      "[kconv] k-convolution router: cpu -> FFTW3-ABI host plan route",
                      scope="rank0", emit=announce)
    else:
        announce_once(("kconv", "backend", backend),
                      f"[kconv] k-convolution router: {mesh_ffi_platform(mesh)} -> XLA "
                      f"(jnp.fft k-axis transforms)", scope="rank0", emit=announce)
    return backend


def _probe_kconv_compile(mesh: Mesh) -> None:
    """Compile and run one tiny mathdx kernel (mode 3, k-grid 2x1x1) on this
    process's first mesh device (else its first local device), so a device the installed cuFFTDx cannot
    compile for refuses at startup, naming its compute capability, rather than
    at the first k-convolution.  The cubin is disk-cached like every other."""
    local = [d for d in mesh.devices.flat if d.process_index == jax.process_index()]
    dev = local[0] if local else jax.local_devices()[0]
    try:
        with jax.default_device(dev):
            jax.block_until_ready(jax.jit(lambda: _rows_kfft_call(
                KFFT_KLEAD_TARGET, jnp.zeros((2, 1), jnp.complex128), (2, 1, 1),
                forward=True, scale=1.0))())
    except Exception as e:                                          # noqa: BLE001
        from importlib import metadata
        try:
            wheel = metadata.version("nvidia-mathdx")
        except metadata.PackageNotFoundError:
            wheel = "?"
        cc = getattr(dev, "compute_capability", "?")
        raise RuntimeError(
            f"GATE mathdx-probe: got a probe compile failure on {dev.device_kind} (compute "
            f"capability {cc}) with nvidia-mathdx {wheel}; want every mathdx kernel to compile "
            "for this device; why: cuFFTDx defines a fixed list of SM targets and the kernels "
            "are built by NVRTC for the device's own; fix: a nvidia-mathdx wheel that supports "
            f"this architecture. Cause: {e}") from e


def _cubin_cache_summary() -> str:
    """``<dir>: N images, X MB`` for the startup line (one flat directory, no walk)."""
    import os
    d = cubin_cache_dir()
    try:
        sizes = [e.stat().st_size for e in os.scandir(d)
                 if e.is_file() and e.name.endswith(".cubin")]
    except FileNotFoundError:
        sizes = []
    linked = seed_cubin_cache(str(CUBIN_STORE), d)
    return (f"{d}: {len(sizes)} images, {sum(sizes) / 1e6:.1f} MB"
            + (f" ({linked} linked from the store {CUBIN_STORE})" if CUBIN_STORE.is_dir() else ""))


def _require_target(target: str, platform: str) -> None:
    from ffi.common import ffi_loader
    ok, why = ffi_loader.probe_target(target, platform)
    if not ok:
        raise RuntimeError(
            f"GATE kconv-target: got a liblorrax_ffi without {target} on "
            f"{platform} ({why}); want the handler this router selects; fix: "
            "rebuild the native library (Perlmutter: "
            "config/perlmutter/build_ffi_cuda.sh) from this tree.")


def _mathdx_attrs(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale) -> dict:
    kg = _check_kgrid(kgrid, "mathdx")
    return dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                scale=np.float64(scale),
                perm_l=_check_perm(perm_l, ns, "left"),
                phase_l=_conv_kpair_phase_codes(phase_l, ns, "left"),
                perm_r=_check_perm(perm_r, ns, "right"),
                phase_r=_conv_kpair_phase_codes(phase_r, ns, "right"),
                **_mathdx_common())


def _check_kgrid(kgrid, backend: str) -> tuple[int, int, int]:
    """Three positive k axes; on the ``mathdx`` leg also at most ``KCONV_AXIS_MAX``.

    The cap is the fp64 cuFFTDx thread-FFT limit, a mathdx constraint: the XLA
    backend has none.
    """
    kg = tuple(int(v) for v in kgrid)
    if len(kg) != 3 or min(kg) < 1:
        raise RuntimeError(
            f"GATE kconv-kgrid: got k-grid {kg}; want three positive axes; fix: "
            "pass the (nkx, nky, nkz) of the run's k-grid.")
    if backend == "mathdx" and max(kg) > KCONV_AXIS_MAX:
        raise RuntimeError(
            f"GATE mathdx-kconv-axis: got k-grid {kg}; want every axis in "
            f"[1,{KCONV_AXIS_MAX}]; why: the fp64 cuFFTDx thread-FFT limit; fix: "
            "a smaller k-grid.")
    return kg


def _mathdx_common() -> dict:
    """The two string attributes every mathdx handler takes: the wheel root and
    the disk cubin cache directory (:func:`cubin_cache_dir`)."""
    return dict(mathdx_root=mathdx_root(), cubin_dir=cubin_cache_dir())


def cubin_cache_dir() -> str:
    """Where the compiled mathdx kernels are kept: ``$SCRATCH/.cache/lorrax/kconv_mathdx``,
    or ``~/.cache/lorrax/kconv_mathdx`` where the site defines no ``SCRATCH``.

    Always on, and separate from the XLA compile cache
    (``common.jax_compile_cache``, one namespace per release): this store is small
    and content-addressed — each image is keyed by the full hash of its source,
    NVRTC options, wheel version and NVRTC version, written by tmp+rename and
    re-hashed on read — so reusing it can never change a result, while
    rebuilding it costs about 6 s per (mode, k-grid) per process.  One
    directory for every world size: an image depends on the device and the
    wheel, not on P.  No knob.  The first call of a process seeds it from the
    release's read-only store (:func:`seed_cubin_cache`).
    """
    from lxkit import user_cache_dir
    d = str(user_cache_dir("kconv_mathdx"))
    seed_cubin_cache(str(CUBIN_STORE), d)
    return d


#: The release's read-only image store: ``<source root>/cubin_store``, built once per
#: architecture at install (``scripts/build_cubin_store.py``); absent in a checkout.
CUBIN_STORE = Path(__file__).resolve().parents[2] / "cubin_store"


@lru_cache(maxsize=None)
def seed_cubin_cache(store: str, cache: str) -> int:
    """Link every store image the per-user cache lacks into it, once per process; the count.

    The cache path, which every mathdx call carries as its ``cubin_dir`` attribute (and so
    in its JAX compile key), never changes: the store only feeds it.  Each link is made
    under a temporary name and renamed into place, so concurrent ranks cannot tear one; a
    file already present is kept, and a dangling link (a retired release) is replaced.  The
    native reader checks each image's key and hash as for any cached file
    (``common/nvrtc_build.h``), so a torn or foreign store file is rebuilt into the cache.
    """
    import os
    import warnings
    try:
        names = [e.name for e in os.scandir(store) if e.name.endswith(".cubin")]
        os.makedirs(cache, exist_ok=True)
    except OSError:
        return 0
    n = 0
    for name in names:
        dst = os.path.join(cache, name)
        if os.path.exists(dst):
            continue
        tmp = f"{dst}.link.{os.getpid()}"
        try:
            os.symlink(os.path.join(store, name), tmp)
            os.replace(tmp, dst)
            n += 1
        except OSError as e:
            warnings.warn(f"[kconv] cubin store image {name} not linked into {cache}: {e}; "
                          "it compiles once into the cache instead", RuntimeWarning, stacklevel=2)
    return n


# ---- the plan (cpu) and XLA backends: one composition, two transform engines ----

#: The host FFTW3-ABI flat-k transform and fused Σ convolution of the plan backend.
FLAT_K_TARGET = "lorrax_mklfft_flat_k"
GW_CONV_TARGET = "lorrax_mklfft_gw_conv"


# Kernel lessons: the plan backend's host FFTW3-ABI handlers (numbers: sandbox claim ids).
# Over plain JAX: one CrI3 6x6 P4 rank tile (6x6x1 k, 489 x 489 centroids, ns 2) on 128 Milan
#   cores, the Sigma tau convolution 0.16 s against 1.20 s for jnp.fft and the chi0 transform
#   pair 0.34 s against 1.32 s; peak RSS 5.6 against 4.2 GB (3979).
def _host_flat_k(x_flat, kgrid, kind: str, scale: float = 1.0):
    """``scale·FFT^±`` of the leading flat-k axis on the host FFTW3-ABI handler (complex128)."""
    _require_target(FLAT_K_TARGET, "cpu")
    kg = tuple(int(v) for v in kgrid)
    if x_flat.dtype != jnp.complex128:
        raise TypeError(f"the host flat-k transform is complex128 only, got {x_flat.dtype}")
    if int(x_flat.shape[0]) != kg[0] * kg[1] * kg[2]:
        raise ValueError(f"flat-k input leading extent {x_flat.shape[0]} != prod({kg})")
    return jax.ffi.ffi_call(
        FLAT_K_TARGET, jax.ShapeDtypeStruct(x_flat.shape, x_flat.dtype),
        input_output_aliases={0: 0},
    )(x_flat, nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
      forward=np.int64(0 if kind == "ifftn" else 1), scale=np.float64(scale))


def _host_gw_conv_local(kgrid, norm: str | None, mult: float) -> Callable:
    """Rank-local ``fn(G, W) -> sigma`` on the host gw_conv handler (the plan backend):
    ``sigma = fftn(ifftn(G) * ifftn(W)[:, None, :, None, :] * mult)``, all three
    transforms and the product in one chunked call, so the R-space G tile never
    materialises.  ``W`` stays in k space (the plan backend's ``prep`` is the identity)."""
    _require_target(GW_CONV_TARGET, "cpu")
    kg = tuple(int(v) for v in kgrid)
    nk = kg[0] * kg[1] * kg[2]
    attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                 scale_i=np.float64(ffi_fft_scale('ifftn', norm, nk)),
                 scale_f=np.float64(ffi_fft_scale('fftn', norm, nk) * float(mult)))

    def _local(g_local, w_local):
        return jax.ffi.ffi_call(GW_CONV_TARGET, jax.ShapeDtypeStruct(g_local.shape, g_local.dtype),
                                input_output_aliases={0: 0})(g_local, w_local, **attrs)
    return _local


def _kfft(x_flat, kgrid, kind: str, backend: str):
    """Unnormalised transform of the leading flat-k axis: the host handler on the
    plan backend, ``jnp.fft`` on the XLA backend."""
    if backend == "plan":
        return _host_flat_k(x_flat, kgrid, kind)
    from common.fft_helpers import local_fftn3, local_ifftn3   # see the import-cycle note
    kg = tuple(int(v) for v in kgrid)
    norm = "forward" if kind == "ifftn" else "backward"          # both unnormalised
    f = local_ifftn3 if kind == "ifftn" else local_fftn3
    y = f(x_flat.reshape(kg + tuple(x_flat.shape[1:])), axes=(0, 1, 2), norm=norm)
    return y.reshape(x_flat.shape)


def _pair_tail(P_l, P_r, kgrid, perm_l, phase_l, perm_r, phase_r, scale, backend):
    """``s·FFT_k Σ_ab phase_l[a]·phase_r[b]·conj(IFFT_k P_l[:,a,…,b])·IFFT_k P_r[:,π_l a,…,π_r b]``.

    ``P_l``/``P_r`` are flat-k open-spin ``(nk, ns, *rows, ns)``; returns ``(nk, *rows)``.
    """
    ns = int(P_l.shape[1])
    I_l = jnp.conj(_kfft(P_l, kgrid, "ifftn", backend))
    I_r = _kfft(P_r, kgrid, "ifftn", backend)
    Z = 0
    for a in range(ns):
        for b in range(ns):
            w = complex(phase_l[a]) * complex(phase_r[b])
            Z = Z + w * I_l[:, a, ..., b] * I_r[:, int(perm_l[a]), ..., int(perm_r[b])]
    return _kfft(Z, kgrid, "fftn", backend) * scale


def _parent_open_spin(D, tables, right: bool):
    """The typed parent load of every full-k child: ``(nk, ns, mu, nu, ns)``.

    The same map as the mathdx kernel's load: umklapp phases of both
    endpoints, antiunitary conjugation, then ``P = conj(Σ_cd coef·T(D_cd))``.
    """
    irr, sym, left, rightp, L, R, q, trs, coef_l, coef_r = tables
    ns = int(D.shape[1])
    coef = (coef_r if right else coef_l).reshape(-1, ns, ns, ns, ns)   # (k, a, b, c, e)
    lm = jnp.take(left, sym, axis=0)                                    # (k, mu)
    rn = jnp.take(rightp, sym, axis=0)                                  # (k, nu)
    qp = jnp.take(q, irr, axis=0)                                       # (k, 3)
    pl = jnp.exp(2j * jnp.pi * jnp.einsum('ki,kmi->km', qp, jnp.take(L, sym, axis=0)))
    pr = jnp.exp(-2j * jnp.pi * jnp.einsum('ki,kni->kn', qp, jnp.take(R, sym, axis=0)))
    G = jnp.take(D, irr, axis=0)                                        # (k, c, mu, e, nu)
    G = jnp.take_along_axis(G, lm[:, None, :, None, None], axis=2)
    G = jnp.take_along_axis(G, rn[:, None, None, None, :], axis=4)
    V = pl[:, None, :, None, None] * G * pr[:, None, None, None, :]
    V = jnp.where((trs != 0)[:, None, None, None, None], jnp.conj(V), V)
    return jnp.conj(jnp.einsum('kabce,kcmen->kamnb', coef, V))   # spin axes 1 and -1


def pair_resident_refusal(kgrid, *, optin=None) -> str:
    """Why modes 0/1/6 need the staged route; same three-bank rule as CUDA.

    The decision changes execution only. The staged route uses the existing
    k-axis transform factory on each complete-P spatial tile, never a full-spin
    unfolded bank. Both routes have O(nk log(nk) mu nu / P) work and O(nk mu nu / P)
    scratch for bounded spin dimension (2/4); no new vendor or driver route.
    """
    kg = _check_kgrid(kgrid, "mathdx")
    have = _optin_smem_bytes() if optin is None else int(optin)
    need = 3 * 16 * (int(np.prod(kg)) | 1)
    if have is None:
        return "no CUDA driver to read the opt-in shared memory"
    why = f"resident pair row needs {need} B > {have} B opt-in shared memory" if need > have else ""
    if why:
        from ffi.gate import announce_once
        announce_once(("kconv", "pair-staged", kg, have),
                      f"[kconv] pair convolution -> staged spin/spatial tiles through the native k-axis FFT: {why}; "
                      "scratch columns<=2048, no full-spin unfolded banks", scope="rank0")
    return why


def _stream_pair_components(kgrid, ns, perm_l, phase_l, perm_r, phase_r,
                            scale, left, right, shape, fft, ifft):
    """Sum open-spin components with only scalar-spin k tiles live.

    ``left(a,b)``/``right(a,b)`` return (nk,mu,nu); their typed load owns
    symmetry, conjugation and phases. A loop keeps XLA from constructing
    the ns² full-k transform banks which the resident kernel avoids.
    """
    pl = jnp.asarray(_check_perm(perm_l, ns, "left"))
    pr = jnp.asarray(_check_perm(perm_r, ns, "right"))
    phl = jnp.asarray(np.asarray(phase_l, np.complex128).reshape(ns))
    phr = jnp.asarray(np.asarray(phase_r, np.complex128).reshape(ns))

    def add(i, total):
        a, b = i // ns, i % ns
        # Seal counter-indexed loads against post-loop-write rematerialization.
        L = jax.lax.optimization_barrier(left(a, b))
        R = jax.lax.optimization_barrier(right(pl[a], pr[b]))
        weight = jax.lax.optimization_barrier(phl[a] * phr[b])
        return total + weight * jnp.conj(ifft(L)) * ifft(R)

    z = jax.lax.fori_loop(0, ns * ns, add, jnp.zeros(shape, jnp.complex128), unroll=False)
    return fft(z) * scale


def _staged_pair_ffts(mesh, kgrid):
    """Existing NVIDIA k-box/CPU transform factories, with unnormalized transforms."""
    return (make_local_kfft_klead(mesh, kgrid, kind="fftn", norm="backward"),
            make_local_kfft_klead(mesh, kgrid, kind="ifftn", norm="forward"))


def _parent_spin_component(D, tables, right, a, b, mu_start=0, mu_size=None, nu_start=0, nu_size=None):
    """One component of _parent_open_spin, without its ns² full-k bank."""
    irr, sym, left, rightp, L, R, q, trs, coef_l, coef_r = tables
    ns = int(D.shape[1])
    coef = (coef_r if right else coef_l).reshape(-1, ns, ns, ns, ns)
    mu_size = int(D.shape[2]) if mu_size is None else int(mu_size)
    nu_size = int(D.shape[4]) if nu_size is None else int(nu_size)
    lm = jax.lax.dynamic_slice_in_dim(jnp.take(left, sym, axis=0), mu_start, mu_size, axis=1)
    rn = jax.lax.dynamic_slice_in_dim(jnp.take(rightp, sym, axis=0), nu_start, nu_size, axis=1)
    Lv = jax.lax.dynamic_slice_in_dim(jnp.take(L, sym, axis=0), mu_start, mu_size, axis=1)
    Rv = jax.lax.dynamic_slice_in_dim(jnp.take(R, sym, axis=0), nu_start, nu_size, axis=1)
    qp = jnp.take(q, irr, axis=0)
    pl = jnp.exp(2j * jnp.pi * jnp.einsum('ki,kmi->km', qp, Lv))
    pr = jnp.exp(-2j * jnp.pi * jnp.einsum('ki,kni->kn', qp, Rv))

    def add(i, total):
        c, e = i // ns, i % ns
        # One gather of the requested spatial tile. Taking all k rows before
        # slicing endpoints would secretly rebuild the full scalar-spin bank.
        G = D[irr[:, None, None], c, lm[:, :, None], e, rn[:, None, :]]
        G = jax.lax.optimization_barrier(G)
        V = pl[:, :, None] * G * pr[:, None, :]
        V = jnp.where((trs != 0)[:, None, None], jnp.conj(V), V)
        w = jax.lax.optimization_barrier(coef[:, a, b, c, e])
        return total + w[:, None, None] * V

    out = jax.lax.fori_loop(0, ns * ns, add,
                           jnp.zeros((irr.shape[0], mu_size, nu_size), D.dtype),
                           unroll=False)
    return jnp.conj(out)


def _stream_pair_tiles(kgrid, ns, perm_l, phase_l, perm_r, phase_r,
                       scale, left, right, shape, fft, ifft):
    """Complete-P output, with <=2048 spatial columns of scalar-spin scratch."""
    nk, mu, nu = shape
    mt, nt = min(mu, 32), min(nu, 64)
    nm, nn = (mu + mt - 1) // mt, (nu + nt - 1) // nt

    def tile(i, out):
        m = jax.lax.optimization_barrier(jnp.minimum((i // nn) * mt, mu - mt))
        n = jax.lax.optimization_barrier(jnp.minimum((i % nn) * nt, nu - nt))
        value = _stream_pair_components(
            kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale,
            lambda a, b: left(a, b, m, mt, n, nt),
            lambda a, b: right(a, b, m, mt, n, nt),
            (nk, mt, nt), fft, ifft)
        return jax.lax.dynamic_update_slice(out, value, (0, m, n))

    return jax.lax.fori_loop(0, nm * nn, tile, jnp.zeros(shape, jnp.complex128), unroll=False)


def _staged_kparent(mesh, kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale):
    fft, ifft = _staged_pair_ffts(mesh, kgrid)

    def apply(D_l, D_r, tables):
        _check_parent_operands(D_l, D_r, ns)
        return _stream_pair_tiles(
            kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale,
            lambda a, b, m, mt, n, nt: _parent_spin_component(D_l, tables, False, a, b, m, mt, n, nt),
            lambda a, b, m, mt, n, nt: _parent_spin_component(D_r, tables, True, a, b, m, mt, n, nt),
            (int(np.prod(kgrid)), int(D_l.shape[2]), int(D_l.shape[4])), fft, ifft)
    return apply


# ---- the router's pair-convolution factories --------------------------------

# Kernel lessons: the zeta-fit pair convolution and plane FFT, modes 0, 1, 6 and 10 (numbers:
# sandbox claim ids).
# Over plain JAX: mode 10 against run concat + jnp.fft.fftn, 1.49-2.33x, production planes 54^2
#   1.69x and 80^2 1.53x (2749), zeta charge loop -18.7% / -20.6% (2750); mode 6 (Bloch phase and
#   L/R split on load) against the XLA moveaxis/phase/split + mode 1, route-G device time -30.7%
#   (2704).
# Paid: mode 10 at 512 threads, 80^2 3.26 -> 2.60 ms; its cp.async gather 1.11-1.37x (2785).
# Did not pay: mode 1's typed parent load, 0.84-0.95x of XLA at VI3 12x12x1, 1.2-1.9x on TaAs
#   (2650); a DFT GEMM on a full axis, never faster than cuFFT (2748); the CUDA leg on FFT-only
#   plans, 1.20-1.33x slower than the XLA leg (2786); plan supports for density, mtxel and the
#   zeta cylinder, <= 2% of FFT time (2785); one plane per block at N <= 40 (2749).
# Decides it: HBM passes (mode 10 writes each plane once; route G runs at 64% of HBM) until
#   residency binds: mode 6 at Fe 8^3 is 1 block per SM with 62% excess bank wavefronts (2796).
def make_fused_conv_kpair(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    *,
    perm_l,
    phase_l,
    perm_r,
    phase_r,
    norm: str | None = "forward",
    mult: float = 1.0,
) -> Callable:
    """The ISDF CCT/ZCT post-pair convolution ``fn(A, B) -> U``, routed by platform.

    ``A``/``B`` ``(nkx,nky,nkz, ns, col, mu, ns)`` c128, device-local inside
    the caller's ``shard_map``; ``U`` ``(nkx,nky,nkz, col, mu)``::

        U = s·FFT_k Σ_ab phase_l[a]·phase_r[b]·conj(IFFT_k A[…,a,…,b])·IFFT_k B[…,π_l a,…,π_r b]

    CUDA: the nvidia-mathdx family; cpu: the plan backend; elsewhere the XLA
    backend; all return the same contract.
    """
    ns = int(np.asarray(perm_l).size)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    scale = conv_kpair_scale(norm, nk, mult)
    backend = kconv_backend(mesh, kgrid)
    if backend == "mathdx" and pair_resident_refusal(kgrid):
        fft, ifft = _staged_pair_ffts(mesh, kgrid)

        def _staged(A, B):
            _check_pair_operands(A, B, (nkx, nky, nkz), ns)
            Af, Bf = A.reshape((nk,) + A.shape[3:]), B.reshape((nk,) + B.shape[3:])
            def component(X, a, b, m, mt, n, nt):
                return jax.lax.dynamic_slice(X, (0, a, m, n, b), (nk, 1, mt, nt, 1))[:, 0, :, :, 0]
            U = _stream_pair_tiles(
                kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale,
                lambda a, b, m, mt, n, nt: component(Af, a, b, m, mt, n, nt),
                lambda a, b, m, mt, n, nt: component(Bf, a, b, m, mt, n, nt),
                (nk,) + A.shape[4:6], fft, ifft)
            return U.reshape(A.shape[:3] + A.shape[4:6])
        return _staged
    if backend == "mathdx":
        _require_target(KCONV_PAIR_TARGET, "CUDA")
        attrs = _mathdx_attrs(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale)

        def _mathdx(A, B):
            _check_pair_operands(A, B, (nkx, nky, nkz), ns)
            out = jax.ShapeDtypeStruct(A.shape[:3] + A.shape[4:6], A.dtype)
            return jax.ffi.ffi_call(KCONV_PAIR_TARGET, out)(A, B, **attrs)
        return _mathdx

    return _kpair(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale, backend)


def make_fused_conv_kparent(mesh, kgrid, ns, trailing_shape, *,
                            perm_l, phase_l, perm_r, phase_r, centroid_major=False) -> Callable:
    """The ISDF parent-load pair convolution ``fn(D_l, D_r, tables) -> U``, routed by platform.

    ``D_l``/``D_r`` ``(n_parent, ns, mu, ns, nu)`` c128 raw-parent projectors
    and the ten typed tables of ``isdf.core._parent_conv_tables_local``;
    ``U`` ``(nk, mu, nu)``.  ``centroid_major`` states the physical layout of
    the D operands (CCT) for the CUDA handler; the logical contract is
    unchanged.  ``trailing_shape`` is the caller's ``(mu, nu)`` tile, kept for
    the seam's signature.  CUDA: nvidia-mathdx; cpu: the plan backend; elsewhere XLA.
    """
    del trailing_shape
    ns = int(ns)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    scale = conv_kpair_scale("forward", nk, 1.0)
    backend = kconv_backend(mesh, kgrid)
    if backend == "mathdx" and pair_resident_refusal(kgrid):
        return _staged_kparent(mesh, kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale)
    if backend == "mathdx":
        _require_target(KCONV_PARENT_TARGET, "CUDA")
        attrs = _mathdx_attrs(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale)
        attrs["centroid_major"] = np.int64(bool(centroid_major))
        layout = (0, 4, 3, 2, 1) if centroid_major else None

        def _mathdx(D_l, D_r, tables):
            _check_parent_operands(D_l, D_r, ns)
            out = jax.ShapeDtypeStruct((nk, D_l.shape[2], D_l.shape[4]), D_l.dtype)
            return jax.ffi.ffi_call(
                KCONV_PARENT_TARGET, out,
                input_layouts=(layout, layout, *(None for _ in tables)),
            )(D_l, D_r, *tables, **attrs)
        return _mathdx

    return _kparent(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale, backend)


def make_fused_conv_kplane(mesh, kgrid, ns, *, perm_l, phase_l, perm_r, phase_r) -> Callable:
    """The route-G pair convolution ``fn(D, F) -> U`` read from the D-plane FFT output.

    ``D`` ``(nk, g, ns, 2c, ns, p)`` c128 is the plane transform exactly as
    ``isdf.zeta_mubatch`` leaves it (``g`` planes of ``p`` points; slots
    ``[0, c)`` of the ``2c`` axis are the L projector, ``[c, 2c)`` the R one),
    ``F`` ``(nk, g, p)`` its Bloch phase; ``U`` ``(nk, c, g·p)``::

        P^X_{k,ab}(m, (g,p)) = conj(F[k,g,p] · D[k,g,a,X+m,b,p])      X = 0 | c
        U = s·FFT_k Σ_ab phase_l[a]·phase_r[b]·conj(IFFT_k P^L_ab)·IFFT_k P^R_{π_l a, π_r b}

    — :func:`make_fused_conv_kparent` on the identity plan, without the
    transposed, phased and split copy of ``D`` that its operand layout needs.
    CUDA: nvidia-mathdx mode 6 (the phase and the split are applied on load);
    the plan and XLA backends run the same composition.  ``s`` is the forward-norm
    pair scale of the parent factory.
    """
    ns = int(ns)
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    scale = conv_kpair_scale("forward", nk, 1.0)
    backend = kconv_backend(mesh, kgrid)
    if backend == "mathdx" and pair_resident_refusal(kgrid):
        fft, ifft = _staged_pair_ffts(mesh, kgrid)

        def _staged(D, F):
            c = _check_plane_operands(D, F, nk, ns)
            def component(a, b, offset, m, mt, n, nt):
                # Gather only this spatial tile, including flattened(g,p) columns.
                # A moveaxis/reshape of all g planes before slicing could copy a
                # complete scalar-spin bank, defeating the scratch bound.
                cols = n + jnp.arange(nt)
                gv, pv = cols // D.shape[5], cols % D.shape[5]
                rows = offset + m + jnp.arange(mt)
                X = D[jnp.arange(nk)[:, None, None], gv[None, None, :], a,
                      rows[None, :, None], b, pv[None, None, :]]
                phase = F[:, gv, pv]
                return jnp.conj(X * phase[:, None, :])
            return _stream_pair_tiles(
                kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale,
                lambda a, b, m, mt, n, nt: component(a, b, 0, m, mt, n, nt),
                lambda a, b, m, mt, n, nt: component(a, b, c, m, mt, n, nt),
                (nk, c, D.shape[1] * D.shape[5]), fft, ifft)
        return _staged
    if backend == "mathdx":
        _require_target(KCONV_PLANE_TARGET, "CUDA")
        attrs = _mathdx_attrs(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale)

        def _mathdx(D, F):
            c = _check_plane_operands(D, F, nk, ns)
            out = jax.ShapeDtypeStruct((nk, c, D.shape[1] * D.shape[5]), D.dtype)
            return jax.ffi.ffi_call(KCONV_PLANE_TARGET, out)(D, F, **attrs)
        return _mathdx

    pl, pr = _check_perm(perm_l, ns, "left"), _check_perm(perm_r, ns, "right")
    phl = np.asarray(phase_l, np.complex128).reshape(-1)
    phr = np.asarray(phase_r, np.complex128).reshape(-1)

    def _xla(D, F):
        c = _check_plane_operands(D, F, nk, ns)
        X = jnp.moveaxis(D * F[:, :, None, None, None, :], 1, 4)    # (k, a, 2c, b, g, p)
        X = jnp.conj(jnp.moveaxis(X.reshape(nk, ns, 2 * c, ns, -1), 3, 4))
        return _pair_tail(X[:, :, :c], X[:, :, c:], kgrid, pl, phl, pr, phr, scale,
                          backend)
    return _xla


def _check_plane_operands(D, F, nk: int, ns: int) -> int:
    """Shape/dtype contract of :func:`make_fused_conv_kplane`; returns ``c``."""
    if (D.ndim != 6 or F.ndim != 3 or D.dtype != jnp.complex128 or F.dtype != jnp.complex128
            or int(D.shape[0]) != nk or int(D.shape[2]) != ns or int(D.shape[4]) != ns
            or int(D.shape[3]) % 2 or tuple(F.shape) != (nk, D.shape[1], D.shape[5])):
        raise ValueError(
            f"k-conv plane expects c128 D (nk={nk}, g, ns={ns}, 2c, ns, p) and F (nk, g, p); "
            f"got {D.shape} {D.dtype} / {F.shape} {F.dtype}")
    return int(D.shape[3]) // 2


def _kpair(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale, backend) -> Callable:
    """The plan/XLA-backend ``fn(A, B) -> U`` of :func:`make_fused_conv_kpair`."""
    kg = tuple(int(v) for v in kgrid)
    nk = kg[0] * kg[1] * kg[2]
    pl, pr = _check_perm(perm_l, ns, "left"), _check_perm(perm_r, ns, "right")
    phl = np.asarray(phase_l, np.complex128).reshape(-1)
    phr = np.asarray(phase_r, np.complex128).reshape(-1)

    def _xla(A, B):
        _check_pair_operands(A, B, kg, ns)
        flat = lambda X: X.reshape((nk,) + tuple(X.shape[3:]))
        U = _pair_tail(flat(A), flat(B), kg, pl, phl, pr, phr, scale, backend)
        return U.reshape(A.shape[:3] + A.shape[4:6])
    return _xla


def _kparent(kgrid, ns, perm_l, phase_l, perm_r, phase_r, scale, backend) -> Callable:
    """The plan/XLA-backend ``fn(D_l, D_r, tables) -> U`` of :func:`make_fused_conv_kparent`."""
    pl, pr = _check_perm(perm_l, ns, "left"), _check_perm(perm_r, ns, "right")
    phl = np.asarray(phase_l, np.complex128).reshape(-1)
    phr = np.asarray(phase_r, np.complex128).reshape(-1)

    def _xla(D_l, D_r, tables):
        _check_parent_operands(D_l, D_r, ns)
        return _pair_tail(_parent_open_spin(D_l, tables, False),
                          _parent_open_spin(D_r, tables, True),
                          kgrid, pl, phl, pr, phr, scale, backend)
    return _xla


def _check_pair_operands(A, B, kg, ns) -> None:
    if A.dtype != jnp.complex128 or B.dtype != jnp.complex128:
        raise TypeError(f"k-conv pair is complex128 only; got {A.dtype}/{B.dtype}")
    if A.ndim != 7 or A.shape != B.shape or tuple(int(v) for v in A.shape[:3]) != kg \
            or int(A.shape[3]) != ns or int(A.shape[6]) != ns:
        raise ValueError(
            f"k-conv pair expects equal (nkx,nky,nkz,ns,col,mu,ns) operands with "
            f"k-grid {kg}, ns={ns}; got {A.shape}/{B.shape}")


def _check_parent_operands(D_l, D_r, ns) -> None:
    if (D_l.ndim != 5 or D_l.shape != D_r.shape or D_l.shape[1] != ns or D_l.shape[3] != ns
            or D_l.dtype != jnp.complex128 or D_r.dtype != jnp.complex128):
        raise ValueError("k-conv parent requires matching c128 (parent,ns,mu,ns,nu) operands")


# ===========================================================================
# THE k-CONVOLUTION ROUTER — stored-kernel convolutions and k-axis transforms
# ===========================================================================
# Same routing as the pair family above: CUDA -> nvidia-mathdx (modes 2-5 of
# cpp/cufft/kconv_mathdx_cuda_ffi.cc), cpu -> the plan backend, other -> XLA.
# The k axis is either LEADING (the Σ/COHSEX dot layout, flat k first)
# or MINOR (the BSE ring layout, the three k axes last); a caller asks for the
# factory that matches the tile it holds, and never transposes to reach another.
#
#     make_kconv_klead   Σ, COHSEX   U = mult·fftn(ifftn(T)·ifftn(W)[:,None,:,None,:])
#     make_kconv_kminor  BSE rung    U = mult·fftn_k(ifftn_k(X)·K_R)   (K_R already R space)
#     make_kfft_klead    flat-k      Y = fftn|ifftn over the leading k axis
#     make_kfft_kminor   BSE         Y = fftn|ifftn over the three trailing k axes
#
# The *_local variants are the same callables for code already inside a
# shard_map (which cannot nest); the plain ones wrap their own.

class KConvStored(NamedTuple):
    """A stored-kernel k-leading convolution, split at its one W-only seam.

    ``prep(W) -> W_prep`` is everything that depends on W alone, paid once per
    W; ``apply(T, W_prep) -> U`` is the rest, paid per T.  ``W_prep`` is the
    R-space form: pass it only to the ``apply`` of the same pair.
    """
    prep: Callable
    apply: Callable


def _rows_kfft_call(target, x2, kg, *, forward: bool, scale: float):
    """One mathdx transform call on a 2-D (nk, rows) or (rows, nk) tile."""
    attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                 scale=np.float64(scale), forward=np.int64(1 if forward else 0),
                 **_mathdx_common())
    return jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(x2.shape, x2.dtype),
                            input_output_aliases={0: 0})(x2, **attrs)


def _check_complex(*xs):
    """Every operand complex128, or every operand complex64 (the fp32-GMRES BSE
    arm; mathdx modes 2-5 compile a single-precision image for it).  Never cast."""
    dts = {jnp.dtype(x.dtype) for x in xs}
    if len(dts) != 1 or dts.pop() not in (jnp.dtype(jnp.complex128), jnp.dtype(jnp.complex64)):
        raise TypeError(f"the k-convolution router takes all-complex128 or all-complex64 "
                        f"operands; got {[str(x.dtype) for x in xs]} (it never casts)")


def make_local_kfft_klead(mesh: Mesh, kgrid, *, kind: str, norm: str | None) -> Callable:
    """Rank-local ``fn(X) -> Y`` over the LEADING flat-k axis of ``X (nk, *trail)``."""
    if kind not in ("ifftn", "fftn"):
        raise ValueError(f"kind must be 'ifftn' or 'fftn', got {kind!r}")
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid))
    nk = kg[0] * kg[1] * kg[2]
    scale = ffi_fft_scale(kind, norm, nk)
    if kconv_backend(mesh, kgrid) == "mathdx":
        _require_target(KFFT_KLEAD_TARGET, "CUDA")

        def _mathdx(x):
            _check_complex(x)
            y = _rows_kfft_call(KFFT_KLEAD_TARGET, x.reshape(nk, -1), kg,
                                forward=kind == "fftn", scale=scale)
            return y.reshape(x.shape)
        return _mathdx

    if kconv_backend(mesh, kgrid) == "plan":
        return lambda x: _host_flat_k(x, kg, kind, scale)

    def _xla(x):
        _check_complex(x)
        return _kfft(x, kg, kind, "xla") * scale
    return _xla


def make_local_kfft_kminor(mesh: Mesh, kgrid, *, kind: str, norm: str | None) -> Callable:
    """Rank-local ``fn(X) -> Y`` over the three TRAILING k axes of ``X (..., nkx, nky, nkz)``."""
    if kind not in ("ifftn", "fftn"):
        raise ValueError(f"kind must be 'ifftn' or 'fftn', got {kind!r}")
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid, kminor=True))
    nk = kg[0] * kg[1] * kg[2]
    scale = ffi_fft_scale(kind, norm, nk)
    backend = kconv_backend(mesh, kgrid, kminor=True)
    mathdx = backend == "mathdx"
    if mathdx:
        _require_target(KFFT_KMINOR_TARGET, "CUDA")

    def _kminor(x):
        _check_complex(x)
        if tuple(int(v) for v in x.shape[-3:]) != kg:
            raise ValueError(f"k-minor transform expects trailing k axes {kg}; got {x.shape}")
        if mathdx:
            y = _rows_kfft_call(KFFT_KMINOR_TARGET, x.reshape(-1, nk), kg,
                                forward=kind == "fftn", scale=scale)
            return y.reshape(x.shape)
        lead = x.reshape(-1, nk).T                        # the transform is k-leading
        return (_kfft(lead, kg, kind, backend) * scale).T.reshape(x.shape)
    return _kminor


def live_row_mask(live, n_rows: int, per_row: int = 1):
    """The mask of a padded pass's live rows: ``live`` (int32 [2]) is ``[lo, hi)`` in rows of
    ``per_row`` entries; an axis of ``n_rows * per_row`` entries.  The mathdx kconv calls take ``live``
    as an optional last operand, skip the rest and store them as zeros (mode 11: add nothing);
    the XLA compositions zero them with this."""
    r = jnp.arange(n_rows * per_row) // per_row
    return (r >= live[0]) & (r < live[1])


def _sharded(local, mesh, in_specs, out_spec):
    from jax import shard_map
    return shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=out_spec,
                     check_vma=False)


def make_kfft_klead(mesh: Mesh, kgrid, spec: P, *, kind: str, norm: str | None) -> Callable:
    """Sharded ``fn(X) -> Y`` over the leading flat-k axis; ``spec`` is the 3-D-form spec."""
    flat = validate_flat_spec(spec, "the input")
    return _sharded(make_local_kfft_klead(mesh, kgrid, kind=kind, norm=norm),
                    mesh, (flat,), flat)


def make_kfft_kminor(mesh: Mesh, kgrid, spec: P, *, kind: str, norm: str | None) -> Callable:
    """Sharded ``fn(X) -> Y`` over the three trailing k axes; they must be replicated."""
    axes = tuple(spec)
    if len(axes) < 3 or any(a is not None for a in axes[-3:]):
        raise ValueError(f"k-minor transform needs the three trailing k axes replicated; got {spec}")
    return _sharded(make_local_kfft_kminor(mesh, kgrid, kind=kind, norm=norm),
                    mesh, (spec,), spec)



def plane_fft_split(n: int) -> tuple[int, int] | None:
    """The Good-Thomas split ``n = n1·n2`` mode 10 transforms an axis of ``n`` points as.

    The most balanced coprime ``n1 >= n2 >= 2`` with ``n1 <= KCONV_AXIS_MAX``
    (shorter cuFFTDx thread FFTs and more lines per pass); ``(n, 1)`` for a
    prime power ``2 <= n <= KCONV_AXIS_MAX`` (one thread FFT); ``None`` when
    neither exists: a prime power above the thread-FFT limit (64, 81, 125,
    128, …) or a prime factor above it.
    """
    n = int(n)
    best = None
    for n1 in range(2, min(n, KCONV_AXIS_MAX + 1)):
        n2 = n // n1
        if n % n1 == 0 and 2 <= n2 <= n1 and math.gcd(n1, n2) == 1 and (best is None or n1 < best[0]):
            best = (n1, n2)
    if best is None and 2 <= n <= KCONV_AXIS_MAX:
        best = (n, 1)
    return best


def _plane_runs(pfc: np.ndarray, n_col: int) -> tuple:
    """The cylinder -> plane zero fill as maximal runs of the flat plane.

    ``(start, stop)`` for cells holding the consecutive columns ``[start,
    stop)``, ``(-1, length)`` for empty cells; concatenating ``F[...,
    start:stop]`` and zero blocks is ``take(F, pfc, mode='fill')`` bit for bit.
    Kept 2026-09-25 (FP, close call): the XLA route serves every plane mode 10
    cannot, among them the 80² production planes on sm_86/89/120 (99 KiB), and the
    runs cut that plane stage 5.54 -> 4.98 ms on A100 (claim 2746), ~2-5% of FFT time.
    """
    empty = pfc >= n_col
    brk = np.flatnonzero(np.r_[True, (empty[1:] != empty[:-1])
                               | (~empty[1:] & (pfc[1:] != pfc[:-1] + 1))])
    ends = np.r_[brk[1:], pfc.size]
    return tuple((-1, int(e - s)) if empty[s] else (int(pfc[s]), int(pfc[e - 1]) + 1)
                 for s, e in zip(brk, ends))


@lru_cache(maxsize=None)
def _optin_smem_bytes(ordinal: int = 0) -> int | None:
    """The device's opt-in shared memory per block (libcuda attribute 97); None without a driver."""
    try:
        import ctypes
        cu = ctypes.CDLL("libcuda.so.1")
        dev, v = ctypes.c_int(), ctypes.c_int()
        if (cu.cuInit(0) or cu.cuDeviceGet(ctypes.byref(dev), int(ordinal))
                or cu.cuDeviceGetAttribute(ctypes.byref(v), 97, dev)):
            return None
        return int(v.value)
    except Exception:                                         # noqa: BLE001
        return None


def plane_resident_bytes(nb: int, nc: int) -> int:
    """Shared memory one mode-10 block needs for an ``(nb, nc)`` plane, static included.

    The dynamic plane ``16·nb·(nc|1)`` plus the kernel's static tables
    ``live[nb]`` (1 B), ``rowb[rows]`` (4 B, rows <= nb) and ``foff[PB]`` (8 B,
    PB = 1 for any plane near the limit), with 16 B of alignment slack.  Mode 10
    serves the plane only when this fits the device's opt-in shared memory per
    block; the handler's build() applies the same bound (and adds the
    asynchronous gather's staging only when that fits as well).
    """
    nb, nc = int(nb), int(nc)
    return 16 * nb * (nc | 1) + 5 * nb + 8 + 16


def make_plane_fft_gather(mesh: Mesh, plane_from_col, n_col: int, plane_shape) -> Callable:
    """Rank-local ``fn(F) -> Y``: the route-G plane transform read straight from the cylinder.

    ``F (..., n_col)`` c128 holds the occupied cells of an ``(n_b, n_c)``
    plane; ``plane_from_col (n_b·n_c,)`` (host, static) is each cell's column,
    ``n_col`` on an empty one.  ``Y (..., n_b, n_c)`` is::

        Y[..., k_b, k_c] = Σ_{b,c} plane[..., b, c] e^{-2πi (b k_b/n_b + c k_c/n_c)},
        plane = take(F, plane_from_col, axis=-1, mode='fill').reshape(..., n_b, n_c)

    i.e. ``fftn(plane, axes=(-2, -1), norm='backward')``.  ``fn(F, start,
    size)`` transforms the slab ``F[:, start:start+size]`` of ``F (A, S, ...,
    n_col)`` in place (``start`` traced, clamped as ``lax.dynamic_slice``
    clamps; ``size`` static) and returns ``(A, size, ..., n_b, n_c)``.  CUDA: nvidia-mathdx
    mode 10, which never writes the zero plane: the row FFTs run on the
    occupied rows only and gather their cells on load, the column FFTs read
    dead rows as zero, and the plane is stored once.  A plane mode 10 cannot
    serve (an axis with no thread-FFT split, :func:`plane_fft_split`, or a
    plane above the device's opt-in shared memory) takes the XLA route
    (static-run concatenate + cuFFT 2-D), decided here once and announced.
    Elsewhere the XLA route.  The returned function's ``route`` attribute names
    the one taken (``'mathdx'`` or ``'xla'``).
    """
    nb, nc = (int(v) for v in plane_shape)
    n_col = int(n_col)
    pfc = np.asarray(plane_from_col, dtype=np.int64).reshape(-1)
    if pfc.size != nb * nc:
        raise ValueError(f"plane_from_col has {pfc.size} cells; want n_b·n_c = {nb * nc}")
    if pfc.size and not (0 <= int(pfc.min()) and int(pfc.max()) <= n_col):
        raise ValueError(f"plane_from_col entries must lie in [0, n_col={n_col}] (n_col = "
                         f"empty); got [{int(pfc.min())}, {int(pfc.max())}]")
    runs = _plane_runs(pfc, n_col)

    def _check_c128(F):
        if jnp.dtype(F.dtype) != jnp.dtype(jnp.complex128):
            raise TypeError(f"GATE plane-fft-dtype: got F {F.dtype}; want complex128 (mode 10 and "
                            "its XLA route keep one contract; the kernel is fp64); fix: pass complex128")

    def _xla(F, start=None, size=None):
        _check_c128(F)
        if start is not None:
            F = jax.lax.dynamic_slice_in_dim(F, start, int(size), axis=1)
        z = lambda n: jnp.zeros(F.shape[:-1] + (n,), F.dtype)
        st = jnp.concatenate([F[..., a:e] if a >= 0 else z(e) for a, e in runs], axis=-1)
        from common.fft_helpers import local_fftn3     # see the import-cycle note
        return local_fftn3(st.reshape(F.shape[:-1] + (nb, nc)), axes=(-2, -1), norm="backward")

    _xla.route = "xla"
    if kconv_backend(mesh) != "mathdx":
        return _xla
    from ffi.gate import announce_once
    sb, sc = plane_fft_split(nb), plane_fft_split(nc)
    need, have = plane_resident_bytes(nb, nc), _optin_smem_bytes()
    why = ("an axis has no coprime split into cuFFTDx thread FFTs (<= "
           f"{KCONV_AXIS_MAX})" if sb is None or sc is None else
           f"the resident plane needs {need} B > {have} B of opt-in shared memory"
           if have is None or need > have else "")
    if why:
        announce_once(("plane_fft", nb, nc),
                      f"[plane_fft] plane ({nb},{nc}): XLA route (run concatenate + "
                      f"cuFFT 2-D), not mathdx mode 10: {why}", scope="rank0")
        return _xla
    _require_target(PLANE_FFT_GATHER_TARGET, "CUDA")
    occ = (pfc < n_col).reshape(nb, nc)
    rows = np.flatnonzero(occ.any(axis=1)).astype(np.int32)
    gidx = np.where(occ[rows], pfc.reshape(nb, nc)[rows], -1).astype(np.int32)
    attrs = dict(nb=np.int64(nb), nc=np.int64(nc), b1=np.int64(sb[0]), c1=np.int64(sc[0]),
                 **_mathdx_common())
    announce_once(("plane_fft", nb, nc),
                  f"[plane_fft] plane ({nb},{nc}): mathdx mode 10, splits {sb} x {sc}, "
                  f"{rows.size} of {nb} rows occupied", scope="rank0")

    def _mathdx(F, start=None, size=None):
        _check_c128(F)
        if int(F.shape[-1]) != n_col:
            raise ValueError(f"plane FFT expects F (..., n_col={n_col}); got {F.shape}")
        lead = F.shape[:-1]
        if start is not None:
            if F.ndim < 3 or not 0 < int(size) <= F.shape[1]:
                raise ValueError(f"slab form needs F (A, S, ..., n_col) and 0 < size <= S; got "
                                 f"{F.shape}, size={size}")
            lead = (F.shape[0], int(size)) + F.shape[2:-1]
        out = jax.ShapeDtypeStruct(lead + (nb, nc), F.dtype)
        s0 = jnp.asarray(0 if start is None else start, jnp.int32)
        return jax.ffi.ffi_call(PLANE_FFT_GATHER_TARGET, out)(
            F, jnp.asarray(gidx), jnp.asarray(rows), s0, **attrs)
    _mathdx.route = "mathdx"
    return _mathdx

def make_kconv_klead(mesh: Mesh, kgrid, t_spec: P, w_spec: P, *,
                     norm: str | None = "ortho", mult: float = 1.0) -> KConvStored:
    """The Σ-family k-LEADING convolution, routed by platform.

    ``U = mult · fftn(ifftn(T) · ifftn(W)[:, None, :, None, :])`` for
    ``T``/``U`` ``(nk, a, mx, b, my)`` and ``W`` ``(nk, mx, my)`` c128 (flat k
    leading, specs in the 3-D form).  Returns :class:`KConvStored`.

    CUDA: ``prep`` is the mathdx k-leading transform (``ifftn(W)`` into R space,
    once per W) and ``apply`` the fused mathdx T·W pass, in place on T.  The XLA
    backend: the same split in XLA ops.
    """
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid))
    nk = kg[0] * kg[1] * kg[2]
    t_flat = validate_flat_spec(t_spec, "T")
    w_flat = validate_flat_spec(w_spec, "W")
    prep_local, apply_local = _klead_locals(mesh, kg, norm, mult)

    prep_sm = _sharded(prep_local, mesh, (w_flat,), w_flat)
    apply_sm = _sharded(apply_local, mesh, (t_flat, w_flat), t_flat)

    def prep(W):
        _check_complex(W)
        if W.ndim != 3 or int(W.shape[0]) != nk:
            raise ValueError(f"k-leading conv expects W (nk={nk}, mx, my); got {W.shape}")
        return prep_sm(W)

    def apply(T, W_prep):
        _check_complex(T, W_prep)
        if T.ndim != 5 or int(T.shape[0]) != nk:
            raise ValueError(f"k-leading conv expects T (nk={nk}, a, mx, b, my); got {T.shape}")
        return apply_sm(T, W_prep)

    return KConvStored(prep=prep, apply=apply)


def make_local_kconv_klead(mesh: Mesh, kgrid, *, norm: str | None = "ortho",
                           mult: float = 1.0) -> Callable:
    """Rank-local k-LEADING stored-kernel convolution ``fn(T, V_R) -> U`` with the
    kernel ALREADY in R space on every backend, for code inside a shard_map.

    ``U = mult · fftn(ifftn(T) · V_R[:, None, :, None, :])`` for ``T``/``U``
    ``(nk, a, mx, b, my)`` and ``V_R`` ``(nk, mx, my)``.  CUDA: mathdx mode 2, the
    Σ τ pass of :func:`make_kconv_klead`, in place on T.  Elsewhere the XLA backend (as
    :func:`make_local_kconv_kminor`), not the gw_conv host handler, which takes
    k-space W.  The BSE W term holds its T k-leading so that the encode and
    decode are batched ZGEMMs with no T-sized transpose, and calls this factory.
    """
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid))
    nk = kg[0] * kg[1] * kg[2]
    scale = ffi_fft_scale("ifftn", norm, nk) * ffi_fft_scale("fftn", norm, nk) * float(mult)
    if kconv_backend(mesh, kgrid) == "mathdx":
        _require_target(KCONV_KLEAD_TARGET, "CUDA")
        attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                     scale=np.float64(scale))

        def _mathdx(t, v_r):
            return jax.ffi.ffi_call(
                KCONV_KLEAD_TARGET, jax.ShapeDtypeStruct(t.shape, t.dtype),
                input_output_aliases={0: 0})(t, v_r, **attrs, **_mathdx_common())
        return _mathdx

    backend = kconv_backend(mesh, kgrid)

    def _xla(t, v_r):
        t_r = _kfft(t, kg, "ifftn", backend) * scale
        return _kfft(t_r * v_r[:, None, :, None, :], kg, "fftn", backend)
    return _xla


def klead_outer_refusal(mesh: Mesh, kgrid, optin: int | None = None) -> str | None:
    """``None`` when :func:`make_local_kconv_klead_outer` serves this mesh and grid, else why not.

    CUDA needs the handler in the loaded library and the load's 64-column k-box bank,
    ``64·16·((nx·ny·(nz|1))|1)`` B, within the opt-in shared memory per block (the handler's
    GATE mathdx-kconv-outer-tile; ``optin`` defaults to the device's attribute).  An XLA-backend
    mesh is always served.  Callers that get a reason keep the unfused encode +
    :func:`make_local_kconv_klead` chain and say so.
    """
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid))
    if kconv_backend(mesh, kgrid) != "mathdx":
        return None
    from ffi.common import ffi_loader
    ok, why = ffi_loader.probe_target(KCONV_KLEAD_OUTER_TARGET, "CUDA")
    if not ok:
        return f"no {KCONV_KLEAD_OUTER_TARGET} handler ({why})"
    have = _optin_smem_bytes() if optin is None else int(optin)
    if have is None:
        return "no CUDA driver to read the opt-in shared memory"
    bank = 64 * 16 * ((kg[0] * kg[1] * (kg[2] | 1)) | 1)
    if bank > have:
        return f"the 64-column bank needs {bank} B; the device has {have} B of opt-in shared memory"
    return None


def make_local_kconv_klead_outer(mesh: Mesh, kgrid, *, norm: str | None = "ortho",
                                 mult: float = 1.0) -> Callable:
    """Rank-local ``fn(L, R, V_R, conj_r=False) -> U``: :func:`make_local_kconv_klead` of ``T = Σ_K L R``.

    ``U = mult · fftn(ifftn(T) · V_R)`` with ``T[k,a,x,b,y] = Σ_K L[k,a,x,K] R[k,K,b,y]`` for
    ``L`` ``(nk, a, mx, K)``, ``R`` ``(nk, K, b, my)``, ``V_R`` ``(mx, my, nk)`` k-MINOR -- the
    W_R tile as the BSE builds it, so no transpose is made -- and ``U`` ``(nk, a, mx, b, my)``.  CUDA: the
    outer-product load (``kconv_outer_cuda_ffi.cc``) forms T in shared memory on the fp64
    tensor cores and never stores it; the transforms, the kernel multiply and the scaled store
    are mathdx mode 2's.  Its K sum reproduces XLA's batched ZGEMM of the same contraction bit
    for bit (A100), so U equals ``make_local_kconv_klead(einsum(L, R), moveaxis(V_R, -1, 0))``.  ``conj_r`` (static)
    reads ``conj(R)`` instead (the BSE right leg is a conjugated wavefunction; reading it from
    the wavefunction avoids a conjugated copy, bit for bit the explicit ``conj``).  K is zero-padded
    to a multiple of 4 (the m8n8k4 chunk; exact).  The XLA backend: that composition in XLA.
    Check :func:`klead_outer_refusal` first.
    """
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid))
    nk = kg[0] * kg[1] * kg[2]
    scale = ffi_fft_scale("ifftn", norm, nk) * ffi_fft_scale("fftn", norm, nk) * float(mult)
    if kconv_backend(mesh, kgrid) == "mathdx":
        fma = _outer_ksum_fma()
        target = KCONV_KLEAD_OUTER_KSUM_TARGET if fma else KCONV_KLEAD_OUTER_TARGET
        _require_target(target, "CUDA")
        attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                     scale=np.float64(scale), **(dict(ksum=np.int64(1)) if fma else {}))

        def _mathdx(l, r, v_r, conj_r=False):
            shape = (l.shape[0], l.shape[1], l.shape[2], r.shape[2], r.shape[3])
            pad = -l.shape[3] % 4
            if pad:
                l = jnp.pad(l, ((0, 0), (0, 0), (0, 0), (0, pad)))
                r = jnp.pad(r, ((0, 0), (0, pad), (0, 0), (0, 0)))
            return jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(shape, l.dtype))(
                l, r, v_r, conj_r=np.int64(bool(conj_r)), **attrs, **_mathdx_common())
        return _mathdx
    apply = make_local_kconv_klead(mesh, kg, norm=norm, mult=mult)

    def _xla(l, r, v_r, conj_r=False):
        return apply(jnp.einsum("kaxK,kKby->kaxby", l, jnp.conj(r) if conj_r else r),
                     jnp.moveaxis(v_r, -1, 0))
    return _xla


def klead_outer_decode_refusal(mesh: Mesh, kgrid, n_c: int, optin: int | None = None) -> str | None:
    """``None`` when :func:`make_local_kconv_klead_outer_decode` serves this mesh, grid and band count.

    CUDA needs :func:`klead_outer_refusal`'s conditions, the decode handler, the ping-pong's two
    banks, ``2·64·16·RS`` B, within the opt-in shared memory (the handler stages the V tile beside
    them when that fits too, else its Mid reads V from L2), and the per-lane accumulator
    ``ceil(nk/16)·ceil(n_c/8) <= 8`` m8n8 blocks (the handler's GATE mathdx-kconv-outer-decode-tile).
    An XLA-backend mesh is always served.  Callers that get a reason keep the outer conv + XLA decode.
    """
    why = klead_outer_refusal(mesh, kgrid, optin)
    if why is not None or kconv_backend(mesh, kgrid) != "mathdx":
        return why
    from ffi.common import ffi_loader
    ok, why = ffi_loader.probe_target(KCONV_KLEAD_OUTER_DECODE_TARGET, "CUDA")
    if not ok:
        return f"no {KCONV_KLEAD_OUTER_DECODE_TARGET} handler ({why})"
    kg = _check_kgrid(kgrid, "mathdx")
    nk = kg[0] * kg[1] * kg[2]
    have = _optin_smem_bytes() if optin is None else int(optin)
    need = 2 * 64 * 16 * ((kg[0] * kg[1] * (kg[2] | 1)) | 1)
    if have is None or need + 16 > have:
        return f"the two banks need {need} B; the device has {have} B of opt-in shared memory"
    blocks = -(-nk // 16) * -(-int(n_c) // 8)
    if blocks > 8:
        return f"the decode accumulator needs {blocks} m8n8 blocks per lane (nk {nk}, n_c {n_c}); the limit is 8"
    return None


def _outer_ksum_fma() -> int:
    """``LORRAX_BSE_OUTER_KSUM`` (A/B, docs/reference/env_vars.md): ``dmma`` (default) runs the outer load's
    and the fused decode's K sums on the fp64 tensor cores, ``fma`` on the fp64 FMA pipe (the same
    fragment contract; round-off class), for comparing the two where their rates differ (H100, B200).
    Anything else refuses."""
    import os
    v = os.environ.get("LORRAX_BSE_OUTER_KSUM", "dmma").strip().lower()
    if v not in ("dmma", "fma"):
        raise ValueError(f"LORRAX_BSE_OUTER_KSUM={v!r}: want 'dmma' (default) or 'fma'")
    if v == "fma":
        from ffi.gate import announce_once
        announce_once(("bse", "outer_ksum", v), "[kconv_outer] K sums on the fp64 FMA pipe "
                      "(LORRAX_BSE_OUTER_KSUM=fma; A/B only)")
    return int(v == "fma")


# Kernel lessons: the BSE W term's outer-product load and fused decode (numbers: sandbox claim ids).
# Over plain JAX: one-trial matvec 9.76 -> 4.8 ms against the XLA encode and decode ZGEMMs around
#   mode 2 (2.0x; 2844, 2874); CrI3 8x8 SOC Haydock step 55.0 -> 13.6 ms with the layout fixes
#   (4.0x; 2834, 2874).
# Paid: T formed on the load on DMMA and never stored, 9.76 -> 6.80 ms (2844); the decode in the
#   store, U never stored, 7.0 -> 5.4 ms (2846); ping-pong of 2 x 8 warps, kernel 1.14x (2874);
#   row-fastest FFT lines, mode 2 6.0 -> 4.4 ms (2845).
# Did not pay: the FMA K sum, 8.85 against 4.97 ms on DMMA (2846; kept buildable); prefetch, unroll
#   and L2-persisting variants (2844); a reordered decode, no speed while memory-bound (2844); the
#   Mid folded into the inverse x pass 4.73 vs 4.65 ms, psi_c loads hoisted per k pair 5.06 ms,
#   per-warp base pointers 4.69 ms, an even stream-K split 6.12 ms from L2 thrash (2846); 1 or 4 k
#   in flight in the load, 3% slower than 2 (2874); a pair tile reusing both ISDF legs does not fit
#   one SM (I).
# Decides it: T bytes (the encode is ~7 flop/B) until T and U leave HBM; then one block per SM,
#   fixed by the 128 KB register accumulator; now L2 operand latency (long_scoreboard 38%, pipe
#   58%; 2874).  DMMA and named barriers are CUDA-only; ROCm takes the XLA encode and mode 2 (I).
def make_local_kconv_klead_outer_decode(mesh: Mesh, kgrid, *, norm: str | None = "ortho") -> tuple:
    """Rank-local ``(prep, apply)``: :func:`make_local_kconv_klead_outer` with the BSE decode's first
    contraction fused into its store, so U never reaches HBM.

    ``prep(Pc)`` tiles ``Pc`` (nk, n_c, a, mx) once per matvec (``ψ^X_c``); ``apply(L, R, V_R, Pc_t,
    conj_r=False)`` returns ``A[k,c,b,y] = Σ_{a,x} conj(Pc[k,c,a,x]) U[k,a,x,b,y]`` (nk, n_c, b, my), with
    ``U`` the outer conv of ``(L, R, V_R, conj_r)`` -- ``bse_stack_matvec._decode``'s (t, μ) contraction,
    whose (s, ν) contraction stays the caller's einsum.  CUDA (``kconv_outer_cuda_ffi.cc``): one resident
    block per SM, two 8-warp groups ping-ponging on two banks; the (phase, item) tiles go round robin
    so a wave shares one x window in L2, and the tile completing an item sums its partials in phase
    order (deterministic).  ``L`` and ``R`` are
    tiled into fragment order per call (K zero-padded to a multiple of 4, μ and ν to 8).  The XLA
    backend: the outer conv and the einsum in XLA.
    Check :func:`klead_outer_decode_refusal` first.
    """
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid))
    nk = kg[0] * kg[1] * kg[2]
    if kconv_backend(mesh, kgrid) != "mathdx":
        outer = make_local_kconv_klead_outer(mesh, kg, norm=norm)

        def _prep_xla(pc):
            return pc

        def _apply_xla(l, r, v_r, pc, conj_r=False):
            return jnp.einsum("kctM,ktMsN->kcsN", jnp.conj(pc), outer(l, r, v_r, conj_r=conj_r))
        return _prep_xla, _apply_xla
    _require_target(KCONV_KLEAD_OUTER_TARGET, "CUDA")
    _require_target(KCONV_KLEAD_OUTER_DECODE_TARGET, "CUDA")
    scale = ffi_fft_scale("ifftn", norm, nk) * ffi_fft_scale("fftn", norm, nk)
    attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]), scale=np.float64(scale),
                 ksum=np.int64(_outer_ksum_fma()))

    def _prep(pc):                              # (nk, c, a, mx) -> (a, nxb, nk, MB, 2, 8, 4)
        n_k, n_c, na, mx = pc.shape
        mb, nxb = -(-n_c // 8), -(-mx // 8)
        pc = jnp.pad(pc, ((0, 0), (0, 8 * mb - n_c), (0, 0), (0, 8 * nxb - mx)))
        return pc.reshape(n_k, mb, 8, na, nxb, 2, 4).transpose(3, 4, 0, 1, 5, 2, 6), n_c

    def _apply(l, r, v_r, pc_t, conj_r=False):
        pcr, n_c = pc_t
        n_k, na, mx, kr = l.shape
        nb, my = r.shape[2], r.shape[3]
        h, nxb, nyb = -(-kr // 4), -(-mx // 8), -(-my // 8)
        # Lr (a, nxb, nk, H, 8, 4) = L[k, a, 8 xb + g, 4h + t]; Rr (b, nyb, nk, H, 8, 4) = R[k, 4h + t, b, 8 yb + g]
        lr = jnp.pad(l, ((0, 0), (0, 0), (0, 8 * nxb - mx), (0, 4 * h - kr)))
        lr = lr.reshape(n_k, na, nxb, 8, h, 4).transpose(1, 2, 0, 4, 3, 5)
        rr = jnp.pad(r, ((0, 0), (0, 4 * h - kr), (0, 0), (0, 8 * nyb - my)))
        rr = rr.reshape(n_k, h, 4, nb, nyb, 8).transpose(3, 4, 0, 1, 5, 2)
        return jax.ffi.ffi_call(KCONV_KLEAD_OUTER_DECODE_TARGET, jax.ShapeDtypeStruct((n_k, n_c, nb, my), l.dtype))(
            lr, rr, v_r, pcr, conj_r=np.int64(bool(conj_r)), **attrs, **_mathdx_common())
    return _prep, _apply


def _klead_locals(mesh, kg, norm, mult):
    """Rank-local ``(prep, apply)`` of :func:`make_kconv_klead`: ``ifftn(W)`` into R
    space, then the T·W_R pass; on the plan backend the identity and the host
    gw_conv handler, which transforms W itself."""
    if kconv_backend(mesh, kg) == "plan":
        return (lambda w: w), _host_gw_conv_local(kg, norm, mult)
    return (make_local_kfft_klead(mesh, kg, kind="ifftn", norm=norm),
            make_local_kconv_klead(mesh, kg, norm=norm, mult=mult))


def _store_row_map(store_rows, nk: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, kout)``: the stored full-k rows and the (nk,) map k -> output row (-1 = none)."""
    rows = np.asarray(store_rows, dtype=np.int64).reshape(-1)
    if rows.size == 0 or rows.min() < 0 or rows.max() >= nk or np.unique(rows).size != rows.size:
        raise ValueError(f"{label}: store_rows must be distinct full-k rows in [0, {nk}); "
                         f"got {rows.tolist()}")
    kout = np.full(nk, -1, dtype=np.int32)
    kout[rows] = np.arange(rows.size, dtype=np.int32)
    return rows, kout


def x_block_rows(rows) -> np.ndarray:
    """The local left centroid of every row of an x block ``(x0, bx, xs, xn)``: ``(r // bx)*xs + x0 + r % bx``."""
    x0, bx, xs, xn = (int(v) for v in rows)
    r = np.arange(xn * bx)
    return (r // bx) * xs + x0 + r % bx


# Kernel lessons: the k-box k-convolution, modes 2-5, 7-9 and 11 (numbers: sandbox claim ids).
# Over plain JAX: BSE W conv, jnp.fft chain -> mode 4, 33.5 -> 7.1 ms (4.7x, 3x3x1 mu2048; 2672);
#   kpair 8^3, cuFFT plan chain -> fused cuFFTDx, 115 -> 12.5 ms (9.2x; 2651); mode 7's load vs the
#   XLA unfold + spin FFI + mode 2, 26.0 -> 17.2 ms per tau node (2705).
# Paid: tile tables at n_s 4, mode 7 1.90x, mode 11 1.73x (2799); mode 7 on the stage with
#   conflict-free lines and 2 blocks/SM, Fe 8^3 1.68x (2943); mode 11 accumulating on every thread,
#   1.44x (2789); x blocks instead of d x d spin blocks, mu3088 d=1 201.8 -> 58.5 s (2841); tile
#   super-order, Fe 8^3 1.28x (2956, branch); mode 11 on the metal direct stream: the freed memory,
#   not the kernel, cut Green pairs 344 -> 100 per Fe 8^3 map (2950); mode 11's split arm (a pair
#   above the opt-in memory, 20^3 ns 2) with a warp-shuffle pencil and 16 warps per SM, P64 tile
#   316 -> 204 ms per tau node, bitwise (3077): both passes had held one block of 8 warps.
# Residency: mode 8's split vertex pencil staged nkx*ty*(ns^2+17) elements at ty 128/ns^2 and
#   refused the two-spinor Dirac quarters at >= 16^3; ty now comes from the opt-in budget (3080).
# Paid (one warp group pencil, kbox_stage.cuh pencil_group_warp_pass): mode 8's vertex pencil on one
#   warp per pair, 20^3 ns-2 Dirac-quarter kconv call 848 -> 646 ms (pencil 2.3x), ns 4 1.14x (3080);
#   modes 7/9 on the modes 2/3 split arm where a block cannot hold a group of two or more columns,
#   P64 tile mode 7 555 -> 237 ms (the 11-row sub-tile pass 22.6 -> 11.6 ms), mode 9 64.9 -> 32.0 ms
#   (3086): one column per block had re-gathered the pair's group per column (1.3 TB of L2).
# Paid (B9, 3112): mode 7's split passes as separate entry points (one kernel held every pass at
#   the x-pencil Mid's 240 registers, 8 warps per SM; the gather alone needs 104, so 16 warps),
#   plane tiles of the most whole groups the opt-in memory holds (16 -> 24 columns at 20^3 ns 2),
#   and where one column's padded box fits a block (20^3: 134 of 163 KB) the x pencil + Mid, the
#   forward plane and the pencil + store as one column-resident pass (the intermediate read once,
#   not three reads and writes), and the gather's plane tiles in tile-major order (a tile's planes
#   together, so star members find their parents in L2: DRAM 1.46 -> 1.15 GB, 1.05x): pass shape
#   11.66 -> 7.9 ms, P64 tile 237 -> 173 ms, bitwise.
# Did not pay: phase-balanced thread counts 1.014-1.029x (2827); cp.async double buffering -21%
#   (2799); a staged load reading its tables per cell, 1.63x slower on mode 11, 1.10x on mode 7
#   (2789); padded shared rows +7% on mode 11, +19% on mode 7 (2845); a per-member vertex Mid
#   0.76-0.94x (2947); tile tables at n_s 2, CrI3 8x8 Sigma tau 5.58 -> 6.24 s (2799); small-grid
#   butterflies, kernel -28%, tau sweep flat (2320); tau batching, Sigma_mn projection fusion and
#   W-prep fusion have no mechanism, FP32 adds an error source (I); ping-pong: HIP has no named
#   barriers and mode 7 already keeps 3-4 blocks per SM (I); mode 11's split plane pass as two
#   8-column tiles per SM, 1.17x slower than one 16-column tile (a tile's pairs share their tables
#   in L1), and at a carveout of 100, 1.10x slower (3077); a column-resident mode 11 cannot fit an
#   A100: the pair's two transformed columns and its accumulator are 384 KB against 163 KB (I);
#   modes 7/9 split at one 4-column group per block (12^3 ns 2) 0.84x, and 1-column plane tiles
#   (16-byte runs) left mode 9 at its single-arm wall (3086); mode 7's entries at 512 threads each
#   (the Mid spills at 128 registers) 0.92x of 512/256/512, two 12-column plane blocks per SM 0.94x
#   of one 24-column block; B bracket Greens per W load: 2-4 % at most, W_prep is already formed
#   once per pass for every bracket (not built, 3112).  Left (3112): with G fully L2-resident the
#   gather is still 2.73 ms (L2 traffic, 16 warps), so the parent regather bounds a gain at ~12 % of
#   mode 7; tile-major order rereads the per-(k, nu) unfold tables per tile (0.32 -> 0.77 GB), which
#   only op-indexed tables would avoid; a single-sweep column pass would redo each pair's 2x2 unfold
#   per spin column (~4x the gather's L2), more than the intermediate round trip it saves (~1.8 ms).
# Paid (B10, 3120): mode 8's split passes on two entry points (every pass but W_R's plane at 512
#   threads, the vertex pencil included at a 112-byte spill: 3.54 -> 2.59 ms a chunk; W_R's plane,
#   212 registers, at 256), wide plane tiles and tile-major gathers: 20^3 ns 2 Dirac quarter
#   648.6 -> 554.9 ms, ns 4 366.6 -> 315.4 ms; mode 11's pencil on its own entry (the vertex pencil
#   kept 256 threads at nkx > 12 for both passes) and a tile-major plane: 20^3 vertex 123.6 -> 113.1
#   ms, plain 207.0 -> 194.9 ms; all bitwise.
# Did not pay (B10, 3120): mode 9 at ns 1 as one column-resident sweep (one pair per block: every
#   (k, pair) reads its own tables, L2-bound) 31.3 -> 58.8 ms at 512 threads; mode 9's gather
#   tile-major 0.96x; the vertex pencil at two 256-thread blocks per SM (500-byte spill) 0.98x; mode
#   11's plane tile at 24 columns 0.97x (plain); W_R's plane at two blocks per SM (its tile holds one)
#   0.93x on that pass.
# Decides it: blocks resident per SM (<= 64 registers, >= 2 blocks) and odd, conflict-free shared
#   strides, not HBM or FP64 (Fe 8^3 mode 7 at ~50 GB/s and 0.8 TF/s; 2935); after that the
#   unfold gather's L2 latency (long_scoreboard 49%; 2956).
def make_kconv_klead_unfold(mesh: Mesh, kgrid, tables, *, store_rows, norm: str | None = "ortho",
                            mult: float = 1.0) -> Callable:
    """The Σ k-leading convolution read from the RAW-PARENT Green: ``fn(G, Gt, W_prep, load=None) -> U``.

    ``G`` ``(n_parent, mu, ns, nu, ns)`` c128 at ``P(None,'x',None,'y',None)``
    is the centroid-major parent Green (``gw.greens_function_kernel.
    build_G_parents``) and ``Gt`` its transposed partner for the antiunitary
    rows (``None`` when no row is antiunitary); ``tables`` are
    ``symmetry_maps.unfold_load_tables`` of the same plan; ``W_prep`` is
    :func:`make_kconv_klead`'s ``prep(W)`` for the same ``(kgrid, norm, mult)``.
    ``store_rows`` names the full-k rows the caller consumes (the Σ consumers
    pass the plan's ``parent_full_rows``).  Returns ``U``
    ``(len(store_rows), ns, mu, ns, nu)`` at ``P(None,None,'x',None,'y')``,
    equal to ``apply(sigma_conv_operand(unfold_spin_centroid_operator(G, Gt)),
    W_prep)[store_rows]``: the typed unfold, the spin action and the
    spin-major reorder happen on the convolution's load, and every other k
    row is transformed but never stored.  ``load``, when given, is the same
    tables on the devices (``symmetry_maps.device_load_tables``, or a pass's
    cut of them, ``gw.subtile_stream.window_load``), read as operands so the
    consumer's program holds no table constants.  CUDA: nvidia-mathdx mode 7;
    elsewhere the service's reference composition, then the XLA convolution and
    the row selection.

    ``apply(..., rows=(x0, bx, xs, xn))`` stores one x block with the whole spin group: block
    row ``r`` in ``[0, xn*bx)`` of every rank's ``mu`` tile is the local left centroid
    ``(r // bx)*xs + x0 + r % bx`` (``>= mu_local``: a zero padding row), so the output is
    ``(len(store_rows), ns, P_x*xn*bx, ns, nu)`` at the same spec (``rows=None``: every mu).
    A call reads only its own pairs' Green and W sources, so a consumer that bounds the output
    tile by x blocks reads the Green once over all of them.  (An output spin block, the
    rejected alternative, reads every source of each pair per block: the spin action mixes
    them.)

    ``apply(..., live=...)``: the pass is padded to a scan's largest pass and ``live`` (int32
    ``[2]``, replicated) is its live stored rows ``[lo, hi)`` (the ``x``-block rows, or ``mu``):
    the rest do no gather or transform and come back zero.
    """
    from symmetry_maps import (DEVICE_LOAD_SPECS, DeviceLoadTables,
                               apply_unfold_load_tables_local, local_unfold_load_tables)
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid))
    nk = kg[0] * kg[1] * kg[2]
    if int(tables.row.shape[0]) != nk:
        raise ValueError(f"k-leading unfold conv: tables cover {tables.row.shape[0]} k, grid has {nk}")
    rows, kout = _store_row_map(store_rows, nk, "k-leading unfold conv")
    n_out = int(rows.size)
    ns = int(tables.spin.shape[-1])
    spin_host = np.asarray(tables.spin)
    needs_partner = bool(np.any(np.asarray(tables.trs)))
    mesh_shape = (int(mesh.shape["x"]), int(mesh.shape["y"]))
    if tuple(tables.mesh_shape) != mesh_shape:
        raise ValueError(f"k-leading unfold conv: tables were cut for a {tuple(tables.mesh_shape)} "
                         f"mesh; this mesh is {mesh_shape}")
    si, sf = ffi_fft_scale("ifftn", norm, nk), ffi_fft_scale("fftn", norm, nk)
    if kconv_backend(mesh, kgrid) == "mathdx":
        _require_target(KCONV_KLEAD_UNFOLD_TARGET, "CUDA")
        attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                     scale=np.float64(si * sf * float(mult)), **_mathdx_common())

        def apply_tables(g, gt, v_r, t, conj_src, block, live):
            n_par, mx, _, my, _ = (int(v) for v in g.shape)
            x0, bx, xs, xn = (0, 0, 0, 0) if block is None else block
            flat = lambda a: a.reshape(n_par, mx * ns, my * ns)
            out = jax.ShapeDtypeStruct((n_out, ns, xn * bx if bx else mx, ns, my), g.dtype)
            return jax.ffi.ffi_call(KCONV_KLEAD_UNFOLD_TARGET, out)(
                flat(g), flat(gt), t.row, t.trs, t.lsrc, t.rsrc, t.mph, t.nph, t.spin,
                jnp.asarray(kout), v_r, *(() if live is None else (live,)),
                conj_src=np.int64(bool(conj_src)),
                x0=np.int64(x0), bx=np.int64(bx), xs=np.int64(xs), xn=np.int64(xn), **attrs)
    else:
        _, conv_local = _klead_locals(mesh, kg, norm, mult)

        def apply_tables(g, gt, v_r, t, conj_src, block, live):
            n_par, mx, _, my, _ = (int(v) for v in g.shape)
            flat = lambda a: a.reshape(n_par, mx * ns, my * ns)
            gt = jnp.conj(g) if conj_src else gt
            O = apply_unfold_load_tables_local(flat(g), flat(gt), t, spin_host)
            U = jnp.take(conv_local(jnp.transpose(O, (0, 2, 1, 4, 3)), v_r),
                         jnp.asarray(rows), axis=0)
            if block is not None:
                idx = x_block_rows(block)
                keep = jnp.asarray(idx < mx)[None, None, :, None, None]
                U = jnp.where(keep, jnp.take(U, jnp.asarray(np.minimum(idx, mx - 1)), axis=2), 0)
            if live is not None:
                U = jnp.where(live_row_mask(live, int(U.shape[2]))[None, None, :, None, None], U, 0)
            return U

    def local(g, gt, v_r, *rest, conj_src=False, block=None, has_live=False):
        # ``rest``: live (with has_live), then this rank's slices of the placed tables
        # (DeviceLoadTables fields); without them, the host tables are cut here (baked).
        live, load = (rest[0], rest[1:]) if has_live else (None, rest)
        t = (tables._replace(**dict(zip(DeviceLoadTables._fields, load))) if load
             else local_unfold_load_tables(tables))
        return apply_tables(g, gt, v_r, t, conj_src, block, live)

    g_spec = P(None, "x", None, "y", None)
    sm = {}

    def sharded(conj_src, block, placed, has_live):
        key = (conj_src, block, placed, has_live)
        if key not in sm:
            sm[key] = _sharded(partial(local, conj_src=conj_src, block=block, has_live=has_live), mesh,
                               (g_spec, g_spec, P(None, "x", "y")) + ((P(None),) if has_live else ())
                               + (DEVICE_LOAD_SPECS if placed else ()),
                               P(None, None, "x", None, "y"))
        return sm[key]

    def apply(G, Gt, W_prep, *, conj_partner=False, rows=None, load=None, live=None):
        """``conj_partner``: the antiunitary partner is ``conj(G)`` (a Green of real weights),
        read from ``G`` on the load, so no partner tile exists (``Gt`` must be ``None``)."""
        _check_complex(G, W_prep)
        if G.ndim != 5 or int(G.shape[2]) != ns or int(G.shape[4]) != ns:
            raise ValueError(f"k-leading unfold conv expects G (n_parent, mu, {ns}, nu, {ns}); "
                             f"got {G.shape}")
        if conj_partner and Gt is not None:
            raise ValueError("k-leading unfold conv: conj_partner reads conj(G); pass Gt=None")
        if Gt is None:
            if needs_partner and not conj_partner:
                raise ValueError("k-leading unfold conv: the plan has antiunitary rows, so the "
                                 "transposed parent Green Gt is required")
            Gt = G
        if Gt.shape != G.shape or W_prep.shape != (nk, G.shape[1], G.shape[3]):
            raise ValueError(f"k-leading unfold conv: Gt {Gt.shape} / W_prep {W_prep.shape} do not "
                             f"match G {G.shape} and nk={nk}")
        # The kernel addresses parent row row[k] and local sources below the
        # tables' widths without bounds checks: the operands must be the ones
        # the tables were built for.
        if (int(G.shape[0]) != int(tables.n_parent)
                or int(G.shape[1]) * ns != int(tables.lsrc.shape[1])
                or int(G.shape[3]) * ns != int(tables.rsrc.shape[1])):
            raise ValueError(
                f"k-leading unfold conv: G {G.shape} does not match its tables (n_parent="
                f"{tables.n_parent}, endpoints {tables.lsrc.shape[1]}/{tables.rsrc.shape[1]} "
                f"merged over ns={ns})")
        if rows is not None:
            mx = int(G.shape[1]) // mesh_shape[0]
            x0, bx, xs, xn = rows = tuple(int(v) for v in rows)
            if not (bx >= 1 and xn >= 1 and 0 <= x0 < mx and x0 + bx <= xs):
                raise ValueError(f"k-leading unfold conv: x block {rows} must be pieces [x0, x0+bx) "
                                 f"at stride xs >= x0+bx starting inside the local extent {mx}")
            if rows == (0, mx, mx, 1):
                rows = None
        return sharded(bool(conj_partner and needs_partner), rows, load is not None, live is not None)(
            G, Gt, W_prep, *(() if live is None else (live,)), *(() if load is None else tuple(load)))
    return apply


def _vertex_tables(vertices, ns: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    """``(perm, phase)`` monomial vertices → concatenated perm and phase-code attributes."""
    if not 1 <= len(vertices) <= 4:
        raise ValueError(f"k-conv {label}: 1..4 Lorentz vertices per side, got {len(vertices)}")
    perms = [_check_perm(perm, ns, f"{label} vertex {i}") for i, (perm, _) in enumerate(vertices)]
    codes = [_conv_kpair_phase_codes(phase, ns, f"{label} vertex {i}")
             for i, (_, phase) in enumerate(vertices)]
    return np.concatenate(perms).astype(np.int64), np.concatenate(codes).astype(np.int64)


def make_kconv_lorentz_unfold(mesh: Mesh, kgrid, tables, *, w_tables, left_vertices, right_vertices,
                              store_rows, norm: str | None = "ortho",
                              mult: float = 1.0) -> Callable:
    """The four-current Σ convolution read from the RAW-PARENT Green and W: ``fn(G, Gt, W, Wt) -> U``.

    ``U[k,a,x,b,y] = mult · fftn( Σ_ij (γ_i ifftn(Ĝ) γ_j†)[a,x,b,y] · ifftn(Ŵ)[k,x,i,y,j] )``

    ``Ĝ`` is the full-k Green :func:`make_kconv_klead_unfold` reads from ``G``/``Gt``
    and ``tables`` (the typed unfold, spin action and spin-major order on the
    load); ``Ŵ`` the full-q interaction read the same way from its
    irreducible-q parent tile ``W`` ``(nq_irr, mx, nA, my, nB)`` (centroid-major,
    ``gw.greens_function_kernel.build_G_parents``'s order) and partner ``Wt``
    through ``w_tables`` (``symmetry_maps.unfold_load_tables`` of the q plan,
    ``trs_rule="pair_transpose"``, the endpoints' Lorentz actions as
    ``spin``/``spin_r``).  ``left_vertices``/``right_vertices`` are the Lorentz
    vertices ``γ_i``, ``γ_j`` as ``(perm, phase)`` monomial pairs
    (``common.gamma_matrices.gamma_perm_phase``: ``γ[α,β] = phase[α] δ_{β,perm[α]}``).
    One transform of the Green serves every block.  Returns ``U``
    ``(len(store_rows), ns, mx, ns, my)`` at ``P(None,None,'x',None,'y')``, the
    rows ``store_rows`` of the full-k result.  Neither a full-q W nor a
    full-grid W_R exists: each tile's W columns are unfolded and transformed on
    the kernel's load.

    CUDA: nvidia-mathdx mode 8 with the second (W) load; it rounds as mode 9
    on ``W`` then the V_R kconv call (bit for bit).  Elsewhere the service's
    reference unfold of both operands and the XLA convolution.  ``load``/``w_load``
    (``symmetry_maps.DeviceLoadTables`` of ``tables``/``w_tables``, or of a
    row pass's cut of them): the tables enter as device operands, so the
    program holds no table constants; without them the host tables are baked.
    """
    from symmetry_maps import (DEVICE_LOAD_SPECS, DeviceLoadTables,
                               apply_unfold_load_tables_local, local_unfold_load_tables)
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid))
    nk = kg[0] * kg[1] * kg[2]
    if int(tables.row.shape[0]) != nk or int(w_tables.row.shape[0]) != nk:
        raise ValueError(f"k-leading lorentz conv: tables cover {tables.row.shape[0]} k and "
                         f"{w_tables.row.shape[0]} q, grid has {nk}")
    if int(w_tables.conj_trs):
        raise ValueError("k-leading lorentz conv: W tables must use the pair_transpose rule")
    rows, kout = _store_row_map(store_rows, nk, "k-leading lorentz conv")
    n_out = int(rows.size)
    ns = int(tables.spin.shape[-1])
    spin_host = np.asarray(tables.spin)
    w_spin_l = np.asarray(w_tables.spin)
    w_spin_r = w_spin_l if w_tables.spin_r is None else np.asarray(w_tables.spin_r)
    needs_partner = bool(np.any(np.asarray(tables.trs)))
    mesh_shape = (int(mesh.shape["x"]), int(mesh.shape["y"]))
    if tuple(tables.mesh_shape) != mesh_shape or tuple(w_tables.mesh_shape) != mesh_shape:
        raise ValueError(f"k-leading lorentz conv: tables were cut for a {tuple(tables.mesh_shape)} "
                         f"/ {tuple(w_tables.mesh_shape)} mesh; this mesh is {mesh_shape}")
    perm_l, phase_l = _vertex_tables(left_vertices, ns, "left")
    perm_r, phase_r = _vertex_tables(right_vertices, ns, "right")
    na, nb = len(left_vertices), len(right_vertices)
    if (int(w_spin_l.shape[-1]), int(w_spin_r.shape[-1])) != (na, nb):
        raise ValueError(f"k-leading lorentz conv: W tables act on ({w_spin_l.shape[-1]}, "
                         f"{w_spin_r.shape[-1]}) components; the vertices are ({na}, {nb})")
    si, sf = ffi_fft_scale("ifftn", norm, nk), ffi_fft_scale("fftn", norm, nk)
    n_fields = len(DeviceLoadTables._fields)

    def local_tables(loads):
        # ``loads``: this rank's slices of the placed G then W tables
        # (DeviceLoadTables fields); without them the host tables are cut here.
        if not loads:
            return local_unfold_load_tables(tables), local_unfold_load_tables(w_tables)
        return (tables._replace(**dict(zip(DeviceLoadTables._fields, loads[:n_fields]))),
                w_tables._replace(**dict(zip(DeviceLoadTables._fields, loads[n_fields:]))))
    if kconv_backend(mesh, kgrid) == "mathdx":
        _require_target(KCONV_KLEAD_LORENTZ_TARGET, "CUDA")
        attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                     scale_g=np.float64(si), scale_f=np.float64(sf), mult=np.float64(mult),
                     scale_w=np.float64(si), perm_l=perm_l, phase_l=phase_l, perm_r=perm_r,
                     phase_r=phase_r, **_mathdx_common())

        def local(g, gt, w, wt, *rest, conj_src=False, has_live=False):
            live, loads = (rest[0], rest[1:]) if has_live else (None, rest)
            t, tw = local_tables(loads)
            n_par, mx, _, my, _ = (int(d) for d in g.shape)
            n_w = int(w.shape[0])
            flat = lambda a: a.reshape(n_par, mx * ns, my * ns)
            wflat = lambda a: a.reshape(n_w, mx * na, my * nb)
            out = jax.ShapeDtypeStruct((n_out, ns, mx, ns, my), g.dtype)
            return jax.ffi.ffi_call(KCONV_KLEAD_LORENTZ_TARGET, out)(
                flat(g), flat(gt), t.row, t.trs, t.lsrc, t.rsrc, t.mph, t.nph, t.spin,
                jnp.asarray(kout), wflat(w), wflat(wt), tw.row, tw.trs, tw.lsrc, tw.rsrc,
                tw.mph, tw.nph, tw.spin, tw.spin_r, *(() if live is None else (live,)),
                conj_src=np.int64(bool(conj_src)), **attrs)
    else:
        prep_local = make_local_kfft_klead(mesh, kg, kind="ifftn", norm=norm)
        forward_local = make_local_kfft_klead(mesh, kg, kind="fftn", norm=norm)
        quarter = np.asarray([1, 1j, -1, -1j], dtype=np.complex128)
        left = [(perm_l[i * ns:(i + 1) * ns], quarter[phase_l[i * ns:(i + 1) * ns]]) for i in range(na)]
        right = [(perm_r[j * ns:(j + 1) * ns], quarter[phase_r[j * ns:(j + 1) * ns]]) for j in range(nb)]

        def local(g, gt, w, wt, *rest, conj_src=False, has_live=False):
            live, loads = (rest[0], rest[1:]) if has_live else (None, rest)
            t, tw = local_tables(loads)
            n_par, mx, _, my, _ = (int(d) for d in g.shape)
            n_w = int(w.shape[0])
            flat = lambda a: a.reshape(n_par, mx * ns, my * ns)
            wflat = lambda a: a.reshape(n_w, mx * na, my * nb)
            gt = jnp.conj(g) if conj_src else gt
            O = apply_unfold_load_tables_local(flat(g), flat(gt), t, spin_host)
            green = prep_local(jnp.transpose(O, (0, 2, 1, 4, 3)))
            v_r = prep_local(apply_unfold_load_tables_local(wflat(w), wflat(wt), tw, w_spin_l, w_spin_r))
            total = jnp.zeros_like(green)
            for i, (pl, hl) in enumerate(left):
                for j, (pr, hr) in enumerate(right):
                    # gamma_A on the left spin axis, gamma_B^dagger on the right:
                    # a gather and an exact quarter-turn phase each.
                    value = (jnp.take(green, jnp.asarray(pl), axis=1)
                             * jnp.asarray(hl).reshape(1, ns, 1, 1, 1))
                    value = (jnp.take(value, jnp.asarray(pr), axis=3)
                             * jnp.asarray(np.conj(hr)).reshape(1, 1, 1, ns, 1))
                    total = total + value * v_r[:, None, :, i, None, :, j]
            U = jnp.take(forward_local(total) * mult, jnp.asarray(rows), axis=0)
            if live is None:
                return U
            return jnp.where(live_row_mask(live, mx)[None, None, :, None, None], U, 0)

    g_spec = P(None, "x", None, "y", None)
    sm = {}

    def sharded(c, placed, has_live):
        if (c, placed, has_live) not in sm:
            sm[c, placed, has_live] = _sharded(
                partial(local, conj_src=c, has_live=has_live), mesh,
                (g_spec, g_spec, g_spec, g_spec) + ((P(None),) if has_live else ())
                + ((*DEVICE_LOAD_SPECS, *DEVICE_LOAD_SPECS) if placed else ()),
                P(None, None, "x", None, "y"))
        return sm[c, placed, has_live]
    w_partner = bool(np.any(np.asarray(w_tables.trs)))

    def apply(G, Gt, W, Wt, *, conj_partner=False, load=None, w_load=None, live=None):
        """``conj_partner``: as :func:`make_kconv_klead_unfold`'s.  ``Wt`` may be ``None``
        only when no q row of ``w_tables`` is antiunitary.  ``load``/``w_load``: the
        placed tables (both or neither).  ``live``: as :func:`make_kconv_klead_unfold`'s
        (the Green's and W's shared left rows)."""
        if (load is None) != (w_load is None):
            raise ValueError("k-leading lorentz conv: pass both placed loads or neither")
        _check_complex(G, W)
        if G.ndim != 5 or int(G.shape[2]) != ns or int(G.shape[4]) != ns:
            raise ValueError(f"k-leading lorentz conv expects G (n_parent, mu, {ns}, nu, {ns}); "
                             f"got {G.shape}")
        if conj_partner and Gt is not None:
            raise ValueError("k-leading lorentz conv: conj_partner reads conj(G); pass Gt=None")
        if Gt is None:
            if needs_partner and not conj_partner:
                raise ValueError("k-leading lorentz conv: the plan has antiunitary rows, so the "
                                 "transposed parent Green Gt is required")
            Gt = G
        if Wt is None:
            if w_partner:
                raise ValueError("k-leading lorentz conv: the q plan has antiunitary rows, so the "
                                 "parent W's partner Wt is required")
            Wt = W
        if (Gt.shape != G.shape or Wt.shape != W.shape
                or tuple(W.shape[1:]) != (G.shape[1], na, G.shape[3], nb)
                or int(W.shape[0]) != int(w_tables.n_parent)):
            raise ValueError(f"k-leading lorentz conv: Gt {Gt.shape} / W {W.shape} / Wt {Wt.shape} do "
                             f"not match G {G.shape}, ({na}, {nb}) vertices and "
                             f"{w_tables.n_parent} q parents")
        if (int(G.shape[0]) != int(tables.n_parent)
                or int(G.shape[1]) * ns != int(tables.lsrc.shape[1])
                or int(G.shape[3]) * ns != int(tables.rsrc.shape[1])
                or int(W.shape[1]) * na != int(w_tables.lsrc.shape[1])
                or int(W.shape[3]) * nb != int(w_tables.rsrc.shape[1])):
            raise ValueError(
                f"k-leading lorentz conv: G {G.shape} / W {W.shape} do not match their tables "
                f"(n_parent={tables.n_parent}, endpoints {tables.lsrc.shape[1]}/{tables.rsrc.shape[1]} "
                f"merged over ns={ns}; W endpoints {w_tables.lsrc.shape[1]}/{w_tables.rsrc.shape[1]})")
        placed = load is not None
        return sharded(bool(conj_partner and needs_partner), placed, live is not None)(
            G, Gt, W, Wt, *(() if live is None else (live,)), *((*load, *w_load) if placed else ()))
    return apply


def make_kfft_klead_unfold(mesh: Mesh, kgrid, tables, *, norm: str | None = "ortho") -> Callable:
    """An interaction's R-space operand read from its q WEDGE: ``fn(Wp, Wt=None, load=None) -> Y``.

    ``Wp`` ``(n_wedge, ml, nl)`` c128 at ``P(None,'x','y')`` holds the
    interaction on the wedge rows (merged endpoints ``ml = mx*n_l``,
    ``nl = my*n_r``; a scalar W has ``n_l = n_r = 1``, a Lorentz block
    ``(mx, nA, my, nB)`` flattened); ``Wt`` is its transposed partner, needed
    only when ``tables`` use the pair-transpose rule on antiunitary rows
    (``tables.conj_trs = 0``).  ``tables`` are
    ``symmetry_maps.unfold_load_tables`` of the q wedge; ``load``, when given,
    is the same tables on the devices (``symmetry_maps.device_load_tables``),
    read as operands so a consumer's jit holds no table constants.  Returns ``Y``
    ``(nk, ml, nl)`` at ``P(None,'x','y')``, equal to
    ``make_kconv_klead(...).prep`` of the full-zone interaction
    (``unfold_isdf_operator``, then the endpoint actions): the unfold is the
    transform's load, so the full-zone interaction is never stored.  CUDA:
    nvidia-mathdx mode 9; elsewhere the service's reference composition, then
    ``ifftn``.  ``live`` (int32 ``[2]``, replicated): the pass is padded
    to a scan's largest pass and ``[lo, hi)`` its live left centroid rows; the rest come back zero.
    """
    from symmetry_maps import apply_unfold_load_tables_local, local_unfold_load_tables
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid))
    nk = kg[0] * kg[1] * kg[2]
    if int(tables.row.shape[0]) != nk:
        raise ValueError(f"k-leading unfold fft: tables cover {tables.row.shape[0]} k, grid has {nk}")
    spin_l = np.asarray(tables.spin)
    spin_r = spin_l if tables.spin_r is None else np.asarray(tables.spin_r)
    n_l, n_r = int(spin_l.shape[-1]), int(spin_r.shape[-1])
    conj = int(tables.conj_trs)
    needs_partner = bool(np.any(np.asarray(tables.trs))) and not conj
    if kconv_backend(mesh, kgrid) == "mathdx":
        _require_target(KFFT_KLEAD_UNFOLD_TARGET, "CUDA")
        attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                     scale=np.float64(ffi_fft_scale("ifftn", norm, nk)), conj_trs=np.int64(conj),
                     **_mathdx_common())

        def apply_tables(w, wt, t, live):
            out = jax.ShapeDtypeStruct((nk, int(w.shape[1]), int(w.shape[2])), w.dtype)
            return jax.ffi.ffi_call(KFFT_KLEAD_UNFOLD_TARGET, out)(
                w, wt, t.row, t.trs, t.lsrc, t.rsrc, t.mph, t.nph, t.spin, t.spin_r,
                *(() if live is None else (live,)), **attrs)
    else:
        # The plan backend's gw_conv apply transforms W itself, so its prep stays in k space.
        prep_local = ((lambda o: o) if kconv_backend(mesh, kgrid) == "plan"
                      else make_local_kfft_klead(mesh, kg, kind="ifftn", norm=norm))

        def apply_tables(w, wt, t, live):
            O = apply_unfold_load_tables_local(w, wt, t, spin_l,
                                               None if tables.spin_r is None else spin_r)
            Y = prep_local(O.reshape(nk, int(w.shape[1]), int(w.shape[2])))
            if live is None:
                return Y
            return jnp.where(live_row_mask(live, int(w.shape[1]) // n_l, n_l)[None, :, None], Y, 0)

    def local(w, wt, *rest, has_live=False, placed=False):
        live, load = (rest[0], rest[1:]) if has_live else (None, rest)
        t = (tables._replace(**dict(zip(DeviceLoadTables._fields, load))) if placed
             else local_unfold_load_tables(tables))
        return apply_tables(w, wt, t, live)

    from symmetry_maps import DEVICE_LOAD_SPECS, DeviceLoadTables
    spec = P(None, "x", "y")
    sm = {}

    def sharded(placed, has_live):
        if (placed, has_live) not in sm:
            sm[placed, has_live] = _sharded(
                partial(local, has_live=has_live, placed=placed), mesh,
                (spec, spec) + ((P(None),) if has_live else ()) + (DEVICE_LOAD_SPECS if placed else ()),
                spec)
        return sm[placed, has_live]

    def fn(Wp, Wt=None, load=None, live=None):
        _check_complex(Wp)
        if Wp.ndim != 3 or int(Wp.shape[1]) % n_l or int(Wp.shape[2]) % n_r:
            raise ValueError(f"k-leading unfold fft expects Wp (n_wedge, mx*{n_l}, my*{n_r}); "
                             f"got {Wp.shape}")
        if int(Wp.shape[0]) != int(tables.n_parent):
            raise ValueError(f"k-leading unfold fft: Wp has {Wp.shape[0]} wedge rows, the "
                             f"tables {tables.n_parent}")
        if Wt is None:
            if needs_partner:
                raise ValueError("k-leading unfold fft: the tables read the transposed partner on "
                                 "antiunitary rows (pair_transpose), so Wt is required")
            Wt = Wp
        return sharded(load is not None, live is not None)(
            Wp, Wt, *(() if live is None else (live,)), *(() if load is None else tuple(load)))
    return fn


def chi_unfold_scratch_bytes(kgrid, ns: int, tile_bytes: int, optin: int | None = None) -> int:
    """Per-rank bytes mode 11 draws from XLA's scratch allocator at run time.

    The split arm (a pair's ``2 ns^2`` columns do not fit the opt-in shared memory)
    chunks its pairs through a ``(N_k, chunk·2ns²)``
    intermediate bounded by ``scratch_bytes``. The default is at most 1 GiB
    or one local parent-Green tile, whichever is smaller, and at least one
    pair's full-k transform. Pair chunks own disjoint output entries; their
    size changes neither the transforms nor the accumulation order. The single pass draws none. A runtime
    allocation: compiled ``memory_analysis()`` does not count it, so callers price it.
    """
    nx, ny, nz = (int(v) for v in kgrid)
    have = _optin_smem_bytes() if optin is None else int(optin)
    grp = 2 * int(ns) * int(ns)
    single = have is not None and grp * (((nx * ny * (nz | 1)) | 1) * 16) <= have
    per_pair = nx * ny * nz * grp * 16
    return 0 if single else max(per_pair, min(int(tile_bytes), 1 << 30))


def klead_unfold_scratch_bytes(kgrid, ns: int, pairs: int, optin: int | None = None) -> int:
    """Per-rank bytes one mode-7 call (:func:`make_kconv_klead_unfold`) draws from XLA's
    scratch allocator at run time, for ``pairs`` stored (μ, ν) pairs of the whole spin group.

    The handler's build() takes a split arm when a block cannot hold a spin group of two or
    more columns: the group's ``ns²`` columns of ``16·((nx·ny·(nz|1))|1)`` B exceed the
    opt-in shared memory, or (``ns = 1``) four columns do.  The split arm chunks the pairs
    through an ``(N_k, chunk·ns²)`` intermediate of at most 1 GiB and at least one pair; the
    single arm draws none.  A runtime allocation: compiled ``memory_analysis()`` does not
    count it, so callers price it.
    """
    nx, ny, nz = (int(v) for v in kgrid)
    have = _optin_smem_bytes() if optin is None else int(optin)
    grp = int(ns) * int(ns)
    col = grp * (((nx * ny * (nz | 1)) | 1) * 16)
    if have is not None and col <= have and (grp > 1 or 4 * col <= have):
        return 0
    per_pair = nx * ny * nz * grp * 16
    return max(1, min(int(pairs), (1 << 30) // per_pair)) * per_pair


def make_kconv_chi_unfold(mesh: Mesh, kgrid, tables, *, n_out: int, complete: bool,
                          norm: str | None = "ortho", scratch_bytes: int | None = None) -> Callable:
    """One tau node of the chi0 response read from the RAW-PARENT Green pair:
    ``fn(acc, Gv, Gc, alpha, Gvt=None, Gct=None, load=None) -> acc``.

    ``Gv``/``Gc`` ``(n_parent, mu, ns, nu, ns)`` c128 at ``P(None,'x',None,'y',None)`` are the
    centroid-major parent Greens (``gw.greens_function_kernel.build_G_parents``), NOT
    conjugated; ``tables`` are their plan's ``symmetry_maps.unfold_load_tables``
    (``trs_rule="pair_transpose"``).  ``Gvt``/``Gct`` are the partner tiles an antiunitary row
    reads; ``None`` means the partner is ``conj(G)`` (a Green of real weights), which the load
    forms from ``G`` itself.  ``acc`` ``(n_out, nk, mu, nu)`` c128 at ``P(None,None,'x','y')``
    is updated in place (donate it) and ``alpha`` ``(n_out,)`` c128 is replicated:

        acc[o] += alpha[o] * chi_tau,   chi_tau = sum_ab conj(Gc'_ab) Gv'_ab  (+ c.c. if complete)

    with ``G' = ifftn_k(U_k G[row(k)] U_k^dagger)`` (``norm``) the unfolded Green in R space:
    the response the chi0 minimax kernels accumulate (their ``fftn(conj(G))`` pair, whose
    product is the same) before the one forward transform after the tau sum.  No full-k Green
    exists.  ``load``, when given, is the same tables on the devices
    (``symmetry_maps.device_load_tables``), read as operands so a consumer's jit holds no
    table constants (baked, they are ~0.57 GB of HLO literal at Fe 20^3 with 1792 centroids).
    CUDA: nvidia-mathdx mode 11 on the k-box stage (one pass; a chunked split pass on
    large grids, its intermediate bounded by ``scratch_bytes``); elsewhere the service's reference
    composition, then the XLA transforms.
    """
    from symmetry_maps import (DEVICE_LOAD_SPECS, apply_unfold_load_tables_local,
                               local_unfold_load_tables)
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid))
    nk = kg[0] * kg[1] * kg[2]
    if int(tables.row.shape[0]) != nk:
        raise ValueError(f"k-leading chi unfold: tables cover {tables.row.shape[0]} k, grid has {nk}")
    if int(tables.conj_trs) != 0 or tables.spin_r is not None:
        raise ValueError("k-leading chi unfold: a Green pair's tables use the pair-transpose rule "
                         "and one spin action")
    ns = int(tables.spin.shape[-1])
    spin_host = np.asarray(tables.spin)
    mesh_shape = (int(mesh.shape["x"]), int(mesh.shape["y"]))
    if tuple(tables.mesh_shape) != mesh_shape:
        raise ValueError(f"k-leading chi unfold: tables were cut for a {tuple(tables.mesh_shape)} "
                         f"mesh; this mesh is {mesh_shape}")
    n_out, complete = int(n_out), bool(complete)
    si = ffi_fft_scale("ifftn", norm, nk)
    flat = lambda g: g.reshape(g.shape[0], g.shape[1] * ns, g.shape[3] * ns)
    if kconv_backend(mesh, kgrid) == "mathdx":
        _require_target(KCONV_CHI_UNFOLD_TARGET, "CUDA")

        def apply_tables(acc, gv, gc, alpha, gvt, gct, conj_src, t, live):
            budget = (int(scratch_bytes) if scratch_bytes is not None
                      else chi_unfold_scratch_bytes(kg, ns, int(gv.size) * 16))
            call = jax.ffi.ffi_call(KCONV_CHI_UNFOLD_TARGET, jax.ShapeDtypeStruct(acc.shape, acc.dtype),
                                    input_output_aliases={12: 0})
            return call(flat(gv), flat(gvt), flat(gc), flat(gct), t.row, t.trs, t.lsrc, t.rsrc,
                        t.mph, t.nph, t.spin, alpha, acc, *(() if live is None else (live,)),
                        nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                        si=np.float64(si), conj_trs=np.int64(2 if conj_src else 0),
                        complete=np.int64(complete), scratch_bytes=np.int64(budget),
                        **_mathdx_common())
    else:
        ifft_local = make_local_kfft_klead(mesh, kg, kind="ifftn", norm=norm)

        def apply_tables(acc, gv, gc, alpha, gvt, gct, conj_src, t, live):
            if conj_src:
                gvt, gct = jnp.conj(gv), jnp.conj(gc)
            n_par, mx, _, my, _ = (int(v) for v in gv.shape)

            def unfolded(g, gt):
                O = apply_unfold_load_tables_local(flat(g), flat(gt), t, spin_host)
                return ifft_local(O.reshape(nk, mx * ns, my * ns)).reshape(nk, mx, ns, my, ns)
            chi = jnp.einsum("kxayb,kxayb->kxy", jnp.conj(unfolded(gc, gct)), unfolded(gv, gvt))
            if complete:
                chi = chi + jnp.conj(chi)
            if live is not None:
                chi = jnp.where(live_row_mask(live, mx)[None, :, None], chi, 0)
            return acc + alpha[:, None, None, None] * chi[None]

    def local(acc, gv, gc, alpha, gvt, gct, *rest, conj_src=False, placed=False, has_live=False):
        live, load = (rest[0], rest[1:]) if has_live else (None, rest)
        t = (tables._replace(**dict(zip(DeviceLoadTables._fields, load))) if placed
             else local_unfold_load_tables(tables))
        return apply_tables(acc, gv, gc, alpha, gvt, gct, conj_src, t, live)

    from symmetry_maps import DeviceLoadTables
    g_spec, acc_spec = P(None, "x", None, "y", None), P(None, None, "x", "y")
    sm = {}

    def sharded(conj_src, placed, has_live):
        if (conj_src, placed, has_live) not in sm:
            sm[conj_src, placed, has_live] = _sharded(
                partial(local, conj_src=conj_src, placed=placed, has_live=has_live), mesh,
                (acc_spec, g_spec, g_spec, P(None), g_spec, g_spec) + ((P(None),) if has_live else ())
                + (DEVICE_LOAD_SPECS if placed else ()), acc_spec)
        return sm[conj_src, placed, has_live]

    def fn(acc, Gv, Gc, alpha, Gvt=None, Gct=None, load=None, live=None):
        _check_complex(acc, Gv, Gc, alpha)
        if Gv.ndim != 5 or int(Gv.shape[2]) != ns or int(Gv.shape[4]) != ns or Gc.shape != Gv.shape:
            raise ValueError(f"k-leading chi unfold expects Gv = Gc (n_parent, mu, {ns}, nu, {ns}); "
                             f"got {Gv.shape} / {Gc.shape}")
        if tuple(acc.shape) != (n_out, nk, int(Gv.shape[1]), int(Gv.shape[3])) or alpha.shape != (n_out,):
            raise ValueError(f"k-leading chi unfold: acc {acc.shape} / alpha {alpha.shape} do not match "
                             f"n_out={n_out}, nk={nk} and G {Gv.shape}")
        if (Gvt is None) != (Gct is None):
            raise ValueError("k-leading chi unfold: pass both partners or neither")
        conj_src = Gvt is None
        if conj_src:
            Gvt, Gct = Gv, Gc
        elif Gvt.shape != Gv.shape or Gct.shape != Gv.shape:
            raise ValueError("k-leading chi unfold: the partners must match the Greens' shape")
        return sharded(conj_src, load is not None, live is not None)(
            acc, Gv, Gc, alpha, Gvt, Gct, *(() if live is None else (live,)),
            *(() if load is None else tuple(load)))
    return fn


def make_kconv_chi_vertex(mesh: Mesh, kgrid, tables, *, left_vertices, right_vertices,
                          sign_c=None, norm: str | None = "ortho",
                          scratch_bytes: int | None = None) -> Callable:
    """Mode 11 with channel vertices: ``fn(acc, Gv, Gc, Gvt=None, Gct=None, load=None) -> acc``.

    For channel ``ch = i*nb + j`` of the ``na = len(left_vertices)`` x
    ``nb = len(right_vertices)`` monomial vertices ``(perm, phase)``
    (``γ[α,β] = phase[α] δ_{β,perm[α]}``, ``common.gamma_matrices``)::

        acc[ch, k] += sum_ab conj(phase_i[a]) phase_j[b] conj(Gc'[perm_i a, perm_j b]) Gv'_ab

    with ``G'`` the unfolded Green in R space exactly as :func:`make_kconv_chi_unfold`
    reads it (the same ``tables``, one spin action, a single output weight of 1).
    ``sign_c`` ``(nk,)`` real +-1, when given, multiplies Gc's unfolded k rows
    (a Dirac-half quadrant's own sign relative to ``tables``).  ``acc``
    ``(na*nb, nk, mu, nu)`` at ``P(None,None,'x','y')``, donated.  No full-k
    Green exists.  ``load``, when given, is the same tables on the devices
    (``symmetry_maps.device_load_tables``), read as operands so a consumer's
    jit holds no table constants.  CUDA: nvidia-mathdx mode 11 (LRX_VTX); elsewhere
    the service's reference composition.
    """
    from symmetry_maps import apply_unfold_load_tables_local, local_unfold_load_tables
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid))
    nk = kg[0] * kg[1] * kg[2]
    if int(tables.row.shape[0]) != nk:
        raise ValueError(f"chi vertex: tables cover {tables.row.shape[0]} k, grid has {nk}")
    if int(tables.conj_trs) != 0 or tables.spin_r is not None:
        raise ValueError("chi vertex: a Green pair's tables use the pair-transpose rule and one spin action")
    ns = int(tables.spin.shape[-1])
    spin_host = np.asarray(tables.spin)
    mesh_shape = (int(mesh.shape["x"]), int(mesh.shape["y"]))
    if tuple(tables.mesh_shape) != mesh_shape:
        raise ValueError(f"chi vertex: tables were cut for a {tuple(tables.mesh_shape)} mesh; "
                         f"this mesh is {mesh_shape}")
    perm_l, phase_l = _vertex_tables(left_vertices, ns, "chi left")
    perm_r, phase_r = _vertex_tables(right_vertices, ns, "chi right")
    na, nb = len(left_vertices), len(right_vertices)
    if na > 3 or nb > 3:
        raise ValueError("chi vertex: at most three vertices per side")
    n_ch = na * nb
    signed = sign_c is not None
    sign_host = (np.ones(nk) if sign_c is None else np.asarray(sign_c, dtype=np.float64).reshape(-1))
    if sign_host.shape != (nk,) or not np.all(np.abs(sign_host) == 1.0):
        raise ValueError("chi vertex: sign_c must be (nk,) of +-1")
    si = ffi_fft_scale("ifftn", norm, nk)
    flat = lambda g: g.reshape(g.shape[0], g.shape[1] * ns, g.shape[3] * ns)
    if kconv_backend(mesh, kgrid) == "mathdx":
        _require_target(KCONV_CHI_VERTEX_TARGET, "CUDA")

        def apply_tables(acc, gv, gc, gvt, gct, conj_src, t, live):
            budget = (int(scratch_bytes) if scratch_bytes is not None
                      else chi_unfold_scratch_bytes(kg, ns, int(gv.size) * 16))
            call = jax.ffi.ffi_call(KCONV_CHI_VERTEX_TARGET, jax.ShapeDtypeStruct(acc.shape, acc.dtype),
                                    input_output_aliases={13: 0})
            return call(flat(gv), flat(gvt), flat(gc), flat(gct), t.row, t.trs, t.lsrc, t.rsrc,
                        t.mph, t.nph, t.spin, jnp.ones((1,), jnp.complex128),
                        jnp.asarray(sign_host), acc, *(() if live is None else (live,)),
                        nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                        si=np.float64(si), conj_trs=np.int64(2 if conj_src else 0),
                        scratch_bytes=np.int64(budget), signed_c=np.int64(signed),
                        perm_l=perm_l, phase_l=phase_l, perm_r=perm_r, phase_r=phase_r,
                        na=np.int64(na), nb=np.int64(nb), **_mathdx_common())
    else:
        ifft_local = make_local_kfft_klead(mesh, kg, kind="ifftn", norm=norm)
        codes = np.asarray([1, 1j, -1, -1j])
        pl, hl = perm_l.reshape(na, ns), codes[phase_l.reshape(na, ns)]
        pr, hr = perm_r.reshape(nb, ns), codes[phase_r.reshape(nb, ns)]

        def apply_tables(acc, gv, gc, gvt, gct, conj_src, t, live):
            if conj_src:
                gvt, gct = jnp.conj(gv), jnp.conj(gc)
            n_par, mx, _, my, _ = (int(v) for v in gv.shape)

            def unfolded(g, gt, sign=None):
                O = apply_unfold_load_tables_local(flat(g), flat(gt), t, spin_host)
                if sign is not None:
                    # sign_c is per full-k row, so it acts on the unfolded k rows before the
                    # transform to R (mode 11 applies it on its Gc load, at the same point).
                    O = O * jnp.asarray(sign)[:, None, None, None, None]
                return ifft_local(O.reshape(nk, mx * ns, my * ns)).reshape(nk, mx, ns, my, ns)
            lower = unfolded(gv, gvt)
            upper = unfolded(gc, gct, sign_host if signed else None)
            planes = []
            for i in range(na):
                for j in range(nb):
                    up = upper[:, :, pl[i]][:, :, :, :, pr[j]]
                    w = np.conj(hl[i])[:, None] * hr[j][None, :]
                    planes.append(jnp.einsum("kxayb,ab,kxayb->kxy", jnp.conj(up), w, lower))
            planes = jnp.stack(planes)
            if live is not None:
                planes = jnp.where(live_row_mask(live, mx)[None, None, :, None], planes, 0)
            return acc + planes

    def local(acc, gv, gc, gvt, gct, *rest, conj_src=False, placed=False, has_live=False):
        live, load = (rest[0], rest[1:]) if has_live else (None, rest)
        t = (tables._replace(**dict(zip(DeviceLoadTables._fields, load))) if placed
             else local_unfold_load_tables(tables))
        return apply_tables(acc, gv, gc, gvt, gct, conj_src, t, live)

    from symmetry_maps import DEVICE_LOAD_SPECS, DeviceLoadTables
    g_spec, acc_spec = P(None, "x", None, "y", None), P(None, None, "x", "y")
    sm = {}

    def sharded(conj_src, placed, has_live):
        if (conj_src, placed, has_live) not in sm:
            sm[conj_src, placed, has_live] = _sharded(
                partial(local, conj_src=conj_src, placed=placed, has_live=has_live), mesh,
                (acc_spec, g_spec, g_spec, g_spec, g_spec) + ((P(None),) if has_live else ())
                + (DEVICE_LOAD_SPECS if placed else ()), acc_spec)
        return sm[conj_src, placed, has_live]

    def fn(acc, Gv, Gc, Gvt=None, Gct=None, load=None, live=None):
        _check_complex(acc, Gv, Gc)
        if Gv.ndim != 5 or int(Gv.shape[2]) != ns or int(Gv.shape[4]) != ns or Gc.shape != Gv.shape:
            raise ValueError(f"chi vertex expects Gv = Gc (n_parent, mu, {ns}, nu, {ns}); "
                             f"got {Gv.shape} / {Gc.shape}")
        if tuple(acc.shape) != (n_ch, nk, int(Gv.shape[1]), int(Gv.shape[3])):
            raise ValueError(f"chi vertex: acc {acc.shape} is not ({n_ch}, {nk}, mu, nu)")
        if (Gvt is None) != (Gct is None):
            raise ValueError("chi vertex: pass both partners or neither")
        conj_src = Gvt is None
        if conj_src:
            Gvt, Gct = Gv, Gc
        elif Gvt.shape != Gv.shape or Gct.shape != Gv.shape:
            raise ValueError("chi vertex: the partners must match the Greens' shape")
        return sharded(conj_src, load is not None, live is not None)(
            acc, Gv, Gc, Gvt, Gct, *(() if live is None else (live,)), *(() if load is None else tuple(load)))
    return fn


def kconv_kminor_out_shape(x_shape, out_layout: int) -> tuple[int, ...]:
    """Output shape of :func:`make_kconv_kminor` for ``X`` of ``x_shape``."""
    d0, d1, d2, d3, d4, nk = (int(v) for v in x_shape)
    if out_layout == 0:
        return (d0, d1, d2, d3, d4, nk)
    if out_layout == 1:
        return (d0, nk, d3, d1, d4, d2)
    raise ValueError(f"out_layout must be 0 or 1, got {out_layout!r}")


def _kminor_out_spec(x_spec: P, out_layout: int) -> P:
    ax = tuple(x_spec)
    return P(*ax) if out_layout == 0 else P(ax[0], ax[5], ax[3], ax[1], ax[4], ax[2])


def make_local_kconv_kminor(mesh: Mesh, kgrid, *, norm: str | None = "ortho",
                            mult: float = 1.0, out_layout: int = 0) -> Callable:
    """Rank-local ``fn(X, K_R) -> U`` of :func:`make_kconv_kminor`, for code already
    inside a shard_map: ``X`` ``(d0, d1, d2, d3, d4, nk)``, ``K_R`` ``(d1, d2, nk)``."""
    if out_layout not in (0, 1):
        raise ValueError(f"out_layout must be 0 or 1, got {out_layout!r}")
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid, kminor=True))
    nk = kg[0] * kg[1] * kg[2]
    scale = ffi_fft_scale("ifftn", norm, nk) * ffi_fft_scale("fftn", norm, nk) * float(mult)
    if kconv_backend(mesh, kgrid, kminor=True) == "mathdx":
        _require_target(KCONV_KMINOR_TARGET, "CUDA")
        attrs = dict(nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                     scale=np.float64(scale), out_layout=np.int64(out_layout))

        def _mathdx(x, k_r):
            _check_complex(x, k_r)
            out = jax.ShapeDtypeStruct(kconv_kminor_out_shape(x.shape, out_layout), x.dtype)
            kw = {"input_output_aliases": {0: 0}} if out_layout == 0 else {}
            return jax.ffi.ffi_call(KCONV_KMINOR_TARGET, out, **kw)(
                x, k_r, **attrs, **_mathdx_common())
        return _mathdx

    backend = kconv_backend(mesh, kgrid, kminor=True)

    def _xla(x, k_r):
        _check_complex(x, k_r)
        lead = jnp.moveaxis(x, -1, 0)                      # the transform is k-leading
        u = _kfft(_kfft(lead, kg, "ifftn", backend)
                  * jnp.moveaxis(k_r, -1, 0)[:, None, :, :, None, None], kg, "fftn", backend)
        u = jnp.moveaxis(u * scale, 0, -1)
        return u if out_layout == 0 else jnp.transpose(u, (0, 5, 3, 1, 4, 2))
    return _xla


def make_kconv_kminor(mesh: Mesh, kgrid, x_spec: P, k_spec: P, *,
                      norm: str | None = "ortho", mult: float = 1.0,
                      out_layout: int = 0) -> Callable:
    """The BSE k-MINOR stored-kernel convolution ``fn(X, K_R) -> U``, routed by platform.

    ``U = mult · fftn_k(ifftn_k(X) · K_R[None, :, :, None, None, :])`` for
    ``X`` ``(d0, d1, d2, d3, d4, nk)`` and ``K_R`` ``(d1, d2, nk)`` already in R
    space (the caller made it once with :func:`make_kfft_kminor`).  ``U`` has
    X's layout (``out_layout=0``, in place) or ``(d0, nk, d3, d1, d4, d2)``
    (``out_layout=1``, emitted by the store).  The k axis of both specs must be
    replicated and ``K``'s ``(d1, d2)`` must sit on X's mesh axes.
    """
    if out_layout not in (0, 1):
        raise ValueError(f"out_layout must be 0 or 1, got {out_layout!r}")
    kg = _check_kgrid(kgrid, kconv_backend(mesh, kgrid, kminor=True))
    nk = kg[0] * kg[1] * kg[2]
    xax, kax = tuple(x_spec), tuple(k_spec)
    if len(xax) != 6 or xax[5] is not None or len(kax) != 3 or kax[2] is not None \
            or (kax[0], kax[1]) != (xax[1], xax[2]):
        raise ValueError(f"k-minor conv wants X (d0..d4, nk) and K (d1, d2, nk) with k "
                         f"replicated and (d1, d2) on the same mesh axes; got {x_spec} / {k_spec}")
    _local = make_local_kconv_kminor(mesh, kg, norm=norm, mult=mult, out_layout=out_layout)
    sm = _sharded(_local, mesh, (x_spec, k_spec), _kminor_out_spec(x_spec, out_layout))

    def conv(X, K_R):
        _check_complex(X, K_R)
        if X.ndim != 6 or K_R.ndim != 3 or int(X.shape[5]) != nk or int(K_R.shape[2]) != nk:
            raise ValueError(f"k-minor conv expects X (d0..d4, {nk}) and K (d1, d2, {nk}); "
                             f"got {X.shape} / {K_R.shape}")
        return sm(X, K_R)

    return conv


# =============================================================================
# The local Fourier plan's CUDA leg (``common.fourier_plan.LocalFourierPlan``)
# =============================================================================
FOURIER_PLAN_TARGET = "lorrax_fourier_plan_mathdx"
# The nvidia-mathdx wheels the plan's fused pair is validated with: its NVRTC build
# defines CCCL's structured-bindings include guard to reconcile the wheel's CUTLASS
# with CUDA 13's CCCL (fourier_plan_cuda_ffi.cc, pair_kernel).  Another wheel serves
# every other kernel (the k-convolution family needs only its cuFFTDx headers,
# require_kconv); a plan built on it runs its GEMM pairs as the cuBLAS chain.
PAIR_MATHDX_WHEELS = ("25.6.0",)


def require_fourier_plan(mesh: Mesh, *, announce: bool = True) -> str:
    """Startup check of ``LocalFourierPlan``'s leg on this mesh; returns it or refuses.

    CUDA: the ``lorrax_fourier_plan_mathdx`` handler must be in the loaded library
    (the plan's CUDA leg is that one custom call).  The leg itself needs no mathdx
    wheel; the fused pair checks its wheel where a plan builds it
    (:func:`pair_build_attrs`).  cpu: the XLA ops, nothing to probe.
    """
    from ffi.gate import announce_once, mesh_ffi_platform
    if mesh_ffi_platform(mesh) != "CUDA":
        return "xla"
    _require_target(FOURIER_PLAN_TARGET, "CUDA")
    announce_once(("fourier_plan", "cuda"),
                  f"[fourier_plan] LocalFourierPlan CUDA leg: {FOURIER_PLAN_TARGET} "
                  "(cuBLAS Fourier GEMMs, the fused cuBLASDx pair on a validated nvidia-mathdx "
                  "wheel, one cuFFT group)", scope="rank0", emit=announce)
    return "ffi"


def pair_build_attrs() -> dict:
    """The fused pair's build attributes for :func:`fourier_plan_ffi`, checked once per plan.

    ``{mathdx_root, cubin_dir}`` when the installed nvidia-mathdx wheel is one of
    :data:`PAIR_MATHDX_WHEELS`; otherwise ``{}``, announced, and the plan's GEMM
    pairs run as two cuBLAS GEMMs (the path a pair that does not fit shared memory
    takes).  A missing wheel refuses as the k-convolution family does
    (``GATE mathdx-headers``, :func:`mathdx_root`).
    """
    from importlib.metadata import PackageNotFoundError, version
    from ffi.gate import announce_once
    root = mathdx_root()
    try:
        wheel = version("nvidia-mathdx")
    except PackageNotFoundError:
        wheel = None
    if wheel not in PAIR_MATHDX_WHEELS:
        announce_once(("fourier_plan", "pair-wheel", wheel),
                      f"[fourier_plan] fused pair off: nvidia-mathdx {wheel} is not among the "
                      f"validated wheels {PAIR_MATHDX_WHEELS}; GEMM pairs run as the cuBLAS chain "
                      "(validate the pair on the new wheel with tests/test_fourier_plan.py on the "
                      "ffi leg, then add it to ffi.fft.PAIR_MATHDX_WHEELS)", scope="rank0")
        return {}
    return dict(mathdx_root=root, cubin_dir=cubin_cache_dir())


def fourier_plan_ffi(x, *, n, kin, kout, in_idx, out_idx, sup_in, sup_out, gemm, scale,
                     order, sign, mathdx_root="", cubin_dir=""):
    """One ``lorrax_fourier_plan_mathdx`` custom call over the ``len(n)`` trailing axes
    of ``x`` (row-major in, row-major out; ``cpp/cufft/fourier_plan_cuda_ffi.cc``).

    Every attribute is per transform axis in physical order: full extent
    ``n``, compact extents ``kin``/``kout``, the concatenated supports
    ``in_idx``/``out_idx`` (identity ranges on an axis without one), the 0/1
    flags ``sup_in``/``sup_out``/``gemm``, the axis' jnp.fft ``scale``, and
    ``order``: GEMM axes in execution order with -1 where the FFT group runs.
    ``mathdx_root``/``cubin_dir`` build the fused supported pair (the two
    trailing axes' GEMMs back to back as one cuBLASDx kernel, when a block fits
    it on the device); ``""`` runs every GEMM through cuBLAS.
    """
    d = len(n)
    out = jax.ShapeDtypeStruct(tuple(x.shape[:-d]) + tuple(int(k) for k in kout), x.dtype)
    i64 = lambda v: np.asarray(v, dtype=np.int64)
    return jax.ffi.ffi_call(FOURIER_PLAN_TARGET, out)(
        x, n=i64(n), kin=i64(kin), kout=i64(kout), in_idx=i64(in_idx), out_idx=i64(out_idx),
        sup_in=i64(sup_in), sup_out=i64(sup_out), gemm=i64(gemm),
        scale=np.asarray(scale, dtype=np.float64), order=i64(order), sign=np.int64(sign),
        mathdx_root=mathdx_root, cubin_dir=cubin_dir)
