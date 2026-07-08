"""ω-projection + accumulator for the Σ_c(ω) GN-PPM integration.

"What happens to σ^τ after the kernel returns": the ω-kernel projection (one
numpy function — the single source of truth), the accumulator protocol, and
**one** async-D2H accumulator (`_TauAccumulator`) with **two sinks** deciding
whether a finished window becomes an in-memory Σ tensor (`_MemoryTileSink`) or
streamed h5 read-modify-write additions (`_H5Sink`).  The τ loop doesn't care
which sink is behind the accumulator; it just adds per-τ contributions and
knows when a window boundary falls.

Imports ``_SigmaWindow`` (type only) from the ``ppm_windows`` leaf; nothing here
imports the driver or the device kernel.
"""

from __future__ import annotations

import enum
from collections import deque
from typing import Callable

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from .ppm_windows import _SigmaWindow


class _AccumMode(enum.Enum):
    """How Σ_c(ω, k, m, n) is materialised inside the τ-loop.

    ``KIJ_HOST`` keeps Σ as per-rank numpy tiles matching the (m_X, n_Y)
    shard layout of σ^τ; the full (n_ω, n_k, n_b, n_b) buffer never lives
    on any GPU.  Default for typical ω grids.

    ``KIJ_STREAM`` accumulates each window's contribution on host and writes
    it (ω-batched, read-modify-write) directly to an ``sigma_c_kij_ry`` HDF5
    dataset.  Single-process only (multi-process falls back to KIJ_HOST
    because the read-modify-write storm is a real perf problem at scale).
    Useful when the kij accumulator blows the GPU budget.
    """
    KIJ_HOST = "kij_host"
    KIJ_STREAM = "kij_stream"


def _select_accum_mode(
    requested: str, *,
    sigma_kij_h5_path: str | None,
    kij_bytes: float,
    n_proc: int,
) -> _AccumMode:
    """Map ``omega_accumulation`` (``auto`` / ``kij`` / ``kij_stream``) to a mode.

    - "kij"      → KIJ_HOST
    - "kij_stream" → KIJ_STREAM if a path is set AND single-process,
                     else KIJ_HOST (safety fallback)
    - "auto"     → KIJ_HOST if no path AND grid fits in 0.5 GiB host buffer,
                   otherwise KIJ_STREAM (with the same multi-process safety)
    """
    requested = str(requested).strip().lower()
    if requested not in ("auto", "kij", "kij_stream"):
        raise ValueError("omega_accumulation must be one of: auto, kij, kij_stream.")

    if requested == "kij":
        return _AccumMode.KIJ_HOST

    if requested == "auto":
        small_grid = kij_bytes <= 0.5 * 1024**3
        if sigma_kij_h5_path is None and small_grid:
            return _AccumMode.KIJ_HOST
        # Fall through: large grid OR a stream path is set → try streaming.

    # requested == "kij_stream" or auto-selected streaming
    if not sigma_kij_h5_path or n_proc != 1:
        # Multi-process: read-modify-write storm is too expensive (~hundreds
        # of collective MPI-IO round-trips per branch).  Fall back to host
        # accumulation until we wire the collective-flush variant.
        return _AccumMode.KIJ_HOST
    return _AccumMode.KIJ_STREAM


def _project_tau_onto_omega_np(
    sigma_re: np.ndarray, sigma_im: np.ndarray, omega_vec: np.ndarray,
    t_node: complex, alpha_eff: complex, omega_sign: float, pref: float,
    project_code: int,
) -> np.ndarray:
    """Apply the ω-kernel exp(i·ω_sign·ω·t) and project onto σ channels (host).

    This is the **single** ω-projector in LORRAX (both sinks call it — there is
    no jax mirror to keep in sync).  Returns the single-τ contribution at every
    ω in ``omega_vec``:

        contrib[ω, k, i, j] = pref · α_eff · exp(i·sign·ω·t) · P(σ_re, σ_im)

    where σ^τ is carried as a real/imag pair because the crossing window's HGL
    quadrature needs only Im[coeff·σ^τ] — carrying complex σ^τ through the FFT
    pipeline would double memory for no benefit.  The window's ``project_code``:

        code=0 ("full")  Laplace window (single/stripe/slab) — keep the full
                         complex product (coeff_re + i·coeff_im)·(σ_re + i·σ_im).
        code=1 ("imag")  Crossing window — keep only Im[coeff·σ]
                         = coeff_re·σ_im + coeff_im·σ_re (real, up-cast to c128).
    """
    omega_kernel = np.exp(1j * omega_sign * omega_vec * t_node)
    coeff = alpha_eff * omega_kernel
    coeff_re = np.real(coeff).reshape(-1, 1, 1, 1)
    coeff_im = np.imag(coeff).reshape(-1, 1, 1, 1)
    if project_code == 0:           # full Laplace window
        sigma_full = sigma_re[None, ...] + 1j * sigma_im[None, ...]
        contrib = (coeff_re + 1j * coeff_im) * sigma_full
    elif project_code == 1:         # crossing window — keep Im[coeff·σ]
        contrib = coeff_re * sigma_im[None, ...] + coeff_im * sigma_re[None, ...]
    else:
        raise ValueError(f"Unknown project_code {project_code}")
    return (pref * contrib).astype(np.complex128)


# ---------------------------------------------------------------------------
#  Sigma accumulator — one async-D2H machine, two sinks.
#
#  The tau loop doesn't care whether its outputs land in a GPU-adjacent host
#  buffer or on disk; it just needs to add per-tau contributions and knows when
#  a window boundary falls.  A single _TauAccumulator owns the async-D2H deque
#  and the numpy projector; a sink owns the "what happens to a finished window"
#  decision.
# ---------------------------------------------------------------------------

class _SigmaAccumulator:
    """Minimal protocol used by _integrate_tau_windows_for_branch.

    Lifecycle per branch::

        acc = _TauAccumulator(omega_vec, sink)              # ω_vec is branch-scoped
        for each window:
            acc.begin_window(window)                        # window carries its own metadata
            for each tau:
                acc.add_tau(σ_re, σ_im, t_c, α_eff_c)       # only per-τ data
            acc.end_window()
        acc.finalize()

    ``window`` is the same :class:`_SigmaWindow` built by
    ``_build_windows_for_branch`` — accumulators read whatever they need
    off it (``window.omega_sign``, ``window.prefactor``,
    ``window.project_code``) instead of being fed a parallel stream of
    scalars.  ``window.prefactor`` already carries every sign the branch's
    physics fixes (including the −1 that the −ω half contributes), so there is
    no separate scale argument.  ``t_c`` / ``α_eff_c`` are Python/host complex
    scalars computed once in :func:`minimax_tau_integrate_sigma`.
    """
    def begin_window(self, window: '_SigmaWindow') -> None: ...
    def add_tau(self, sigma_re, sigma_im, t_c: complex, alpha_eff_c: complex) -> None: ...
    def end_window(self) -> None: ...
    def finalize(self) -> jax.Array | None: ...


class _TauAccumulator(_SigmaAccumulator):
    """Async-D2H deque + the single numpy ω-projector, feeding a sink.

    σ(τ) arrives already sharded (m_X, n_Y) from the reduce-scatter tail of
    ``_sigma_kij_kernel``.  Per τ we kick off ``copy_to_host_async`` on **every
    addressable shard** — not just shard 0 — because a single process can own
    more than one GPU (`lxalloc N` interactive, an end-user 2-GPU desktop; the
    code ships to arbitrary device counts, so no shard-0 assumption may survive
    — this was Bug C).  The host reads (and projects/accumulates) each shard's
    σ tile ``lag`` iterations later, so GPU work on τ_{k+lag} overlaps with the
    numpy accumulate of τ_k.  ``end_window`` drains the queue and hands the
    per-shard window tiles to the sink.

    The ω-kernel + accumulate runs entirely in numpy on host — no JAX sharding
    machinery, no collectives, no shard_map.  Each shard's contribution is
    independent (the projector is elementwise in (k, i, j) times the ω-kernel),
    so per-shard projection is exact.
    """

    def __init__(self, omega_vec: jax.Array, sink: '_WindowSink', *, lag: int = 2):
        self._omega_vec_np = np.asarray(jax.device_get(omega_vec),
                                        dtype=np.complex128)
        self._sink = sink
        self._lag = int(lag)
        self._pending: deque = deque()
        # Per-addressable-shard window accumulators + their layout, discovered
        # at the first add_tau (constant across τ for a given window shape).
        self._win_shards: list[np.ndarray | None] | None = None
        self._shard_index: list[tuple] | None = None
        self._shard_devices: list | None = None
        # Window-scoped scalars cached at begin_window.
        self._omega_sign_f: float = 0.0
        self._pref_f: float = 0.0
        self._project_code: int = 0

    def begin_window(self, window: _SigmaWindow) -> None:
        self._pending.clear()
        self._win_shards = None
        self._omega_sign_f = float(window.omega_sign)
        self._pref_f       = float(window.prefactor)
        self._project_code = window.project_code

    def add_tau(self, sigma_re, sigma_im, t_c: complex, alpha_eff_c: complex) -> None:
        # Grab EVERY local shard handle and start the D2H copies now — do NOT
        # materialize to numpy yet.  ``copy_to_host_async`` returns the same
        # shard object with the transfer kicked off in the background.
        shards_re = sigma_re.addressable_shards
        shards_im = sigma_im.addressable_shards
        if self._shard_index is None:
            self._shard_index = [s.index for s in shards_re]
            self._shard_devices = [s.device for s in shards_re]
        handles = []
        for sr, si in zip(shards_re, shards_im):
            sr.data.copy_to_host_async()
            si.data.copy_to_host_async()
            handles.append((sr.data, si.data))
        self._pending.append((handles, t_c, alpha_eff_c))
        if len(self._pending) > self._lag:
            self._drain_one()

    def _drain_one(self) -> None:
        handles, t_c, alpha_eff_c = self._pending.popleft()
        if self._win_shards is None:
            self._win_shards = [None] * len(handles)
        for d, (hr, hi) in enumerate(handles):
            # ``np.asarray`` of an addressable shard waits on the D2H we kicked
            # off earlier — by now the transfer has had ``lag`` iterations of
            # GPU work to overlap with.
            proj = _project_tau_onto_omega_np(
                np.asarray(hr), np.asarray(hi), self._omega_vec_np,
                t_c, alpha_eff_c, self._omega_sign_f, self._pref_f, self._project_code,
            )
            if self._win_shards[d] is None:
                self._win_shards[d] = np.zeros_like(proj)
            self._win_shards[d] += proj

    def end_window(self) -> None:
        while self._pending:
            self._drain_one()
        assert self._win_shards is not None
        self._sink.consume_window(
            self._win_shards, self._shard_index, self._shard_devices)
        self._win_shards = None

    def finalize(self) -> jax.Array | None:
        return self._sink.result()


class _WindowSink:
    """Protocol for what a finished window's per-shard tiles become."""
    def consume_window(self, win_shards, shard_index, shard_devices) -> None: ...
    def result(self) -> jax.Array | None: ...


class _MemoryTileSink(_WindowSink):
    """Σ_c(ω, k, m, n) held as per-rank numpy tiles — never GPU HBM.

    Each window's per-shard tiles are added into a running per-shard total.  At
    :meth:`result`, each shard tile is placed back on the device that owned it
    (``jax.device_put``) and the shards are assembled into a process-sharded
    ``jax.Array`` via ``make_array_from_single_device_arrays`` — which is
    correct on single-process multi-GPU and multi-process alike (each process
    supplies exactly its addressable shards), unlike the old
    ``make_array_from_process_local_data`` fed a shard-0-sized tile.  Downstream
    (``sigma_kij_host`` add, ``_to_host_np``) sees the same (m_X, n_Y) sharding.
    """

    def __init__(self, shape: tuple[int, int, int, int], sharding: NamedSharding):
        self._shape = shape
        self._sharding = sharding
        self._total_shards: list[np.ndarray] | None = None
        self._devices: list | None = None

    def consume_window(self, win_shards, shard_index, shard_devices) -> None:
        if self._total_shards is None:
            self._total_shards = [np.zeros_like(w) for w in win_shards]
            self._devices = list(shard_devices)
        for d, w in enumerate(win_shards):
            self._total_shards[d] += w

    def result(self) -> jax.Array:
        if self._total_shards is None:
            # No window contributed a tile (empty branch): return a zero Σ.
            local_shape = self._sharding.shard_shape(self._shape)
            devices = list(self._sharding.addressable_devices)
            arrays = [
                jax.device_put(np.zeros(local_shape, dtype=np.complex128), d)
                for d in devices
            ]
        else:
            arrays = [
                jax.device_put(t, d)
                for t, d in zip(self._total_shards, self._devices)
            ]
        return jax.make_array_from_single_device_arrays(
            self._shape, self._sharding, arrays)


class _H5Sink(_WindowSink):
    """Assemble each window's full (n_ω, nk, nb, nb) host buffer from its
    per-shard tiles and RMW-add it into the streamed h5 dataset (rank-0),
    ω-batched.

    Streaming is single-process only (``_select_accum_mode`` gates on
    ``n_proc == 1``), so this process owns every shard and the assembled buffer
    is the complete window.  This replaces the old per-τ streamed accumulator:
    D2H is now the ω-independent per-shard σ tile (was n_ω× the volume), and the
    h5 read-modify-write count drops from ~n_τ per branch to n_windows per
    branch.  The analytic q→0 head is RMW-added into this same dataset later by
    ``ppm_pipeline._inject_analytic_head`` (idempotent via the ``head_injected``
    attr); this sink leaves the dataset in exactly the head-less layout that
    step expects.
    """

    def __init__(self, writer: Callable[[np.ndarray, np.ndarray], None], *,
                 omega_global_idx: np.ndarray, omega_batch_size: int,
                 full_spatial_shape: tuple[int, int, int]):
        self._writer = writer
        self._omega_global_idx = np.asarray(omega_global_idx, dtype=np.int64)
        self._batch = int(max(1, omega_batch_size))
        self._spatial = tuple(int(s) for s in full_spatial_shape)

    def consume_window(self, win_shards, shard_index, shard_devices) -> None:
        n_omega = int(win_shards[0].shape[0])
        full = np.zeros((n_omega,) + self._spatial, dtype=np.complex128)
        for tile, idx in zip(win_shards, shard_index):
            full[(slice(None),) + tuple(idx)] = tile
        for ibeg in range(0, n_omega, self._batch):
            iend = min(ibeg + self._batch, n_omega)
            self._writer(self._omega_global_idx[ibeg:iend], full[ibeg:iend])

    def result(self) -> None:
        return None
