"""ω-projection + accumulators for the Σ_c(ω) GN-PPM integration.

"What happens to σ^τ after the kernel returns": the ω-kernel projection (a
jax/numpy mirror pair), the accumulator protocol, and the two concrete
accumulators (host-tile and streamed-h5).  The τ loop doesn't care whether its
outputs land in a GPU-adjacent host buffer or on disk; it just adds per-τ
contributions and knows when a window boundary falls.

Imports ``_SigmaWindow`` (type only) from the ``ppm_windows`` leaf; nothing here
imports the driver or the device kernel.
"""

from __future__ import annotations

import enum
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from .ppm_windows import _SigmaWindow


class _AccumMode(enum.Enum):
    """How Σ_c(ω, k, m, n) is materialised inside the τ-loop.

    ``KIJ_HOST`` keeps Σ as per-rank numpy tiles matching the (m_X, n_Y)
    shard layout of σ^τ; the full (n_ω, n_k, n_b, n_b) buffer never lives
    on any GPU.  Default for typical ω grids.

    ``KIJ_STREAM`` writes per-(τ × ω-batch) contributions directly to an
    ``sigma_c_kij_ry`` HDF5 dataset.  Single-process only (multi-process
    falls back to KIJ_HOST because the read-modify-write storm is a real
    perf problem at scale).  Useful when the kij accumulator blows the
    GPU budget.
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


def _combine_coeff_with_sigma_tau(
    coeff_re: jax.Array,
    coeff_im: jax.Array,
    sigma_tau_kij_re: jax.Array,
    sigma_tau_kij_im: jax.Array,
    project_code: jax.Array,
) -> jax.Array:
    """Multiply ω-kernel coefficient by σ^τ, keeping only the physical piece.

    σ^τ is carried as a real/imag pair because the crossing window's HGL
    quadrature needs only Im[ coeff·σ^τ ] — carrying complex σ^τ through
    the FFT pipeline would double memory for no benefit.  The window sets
    ``project_code`` via ``_SigmaWindow.project``:

        code=0 ("full")  Laplace window (stripe, slab, single) — keep the
                         full complex product  (coeff_re + i·coeff_im) · (σ_re + i·σ_im).
        code=1 ("imag")  Crossing window      — keep only  Im[coeff·σ]
                                              = coeff_re·σ_im + coeff_im·σ_re.

    The historical "real" code path (Re[coeff·σ]) is unused by every current
    window builder.  lax.switch is retained with a 2-way dispatch so that
    the generated HLO matches the previous "full" / "imag" lowering exactly
    and no minimax-table consumer gets a silent behavior change.
    """

    def _full(_):
        sigma_full = sigma_tau_kij_re[None, ...] + 1j * sigma_tau_kij_im[None, ...]
        return (coeff_re + 1j * coeff_im) * sigma_full

    def _imag(_):
        return coeff_re * sigma_tau_kij_im[None, ...] + coeff_im * sigma_tau_kij_re[None, ...]

    return jax.lax.switch(project_code, (_full, _imag), operand=None)


@jax.jit
def _project_tau_onto_omega(
    sigma_tau_kij_re: jax.Array,
    sigma_tau_kij_im: jax.Array,
    omega_vec: jax.Array,
    t_node: jax.Array,
    alpha_eff: jax.Array,
    omega_sign: jax.Array,
    pref: jax.Array,
    project_code: jax.Array,
) -> jax.Array:
    """Apply ω-kernel exp(i·ω_sign·ω·t_node) and project onto σ channels.

    Returns the single-tau contribution at every ω in ``omega_vec``:
        contrib[ω, k, i, j] = pref · α_eff · exp(i·sign·ω·t) · P(σ_re, σ_im)

    where P selects {full, imag} via ``project_code`` — see
    ``_combine_coeff_with_sigma_tau`` for why σ^τ is kept as a (re, im) pair.
    Callers either accumulate the result on-GPU (+=) or write it to disk —
    this kernel is agnostic to the consumer.
    """
    omega_kernel = jnp.exp(1j * omega_sign * omega_vec * t_node)
    coeff = alpha_eff * omega_kernel
    coeff_re = jnp.real(coeff)[:, None, None, None]
    coeff_im = jnp.imag(coeff)[:, None, None, None]
    contrib = _combine_coeff_with_sigma_tau(
        coeff_re, coeff_im, sigma_tau_kij_re, sigma_tau_kij_im, project_code
    )
    return pref * contrib.astype(jnp.complex128)


def _project_tau_onto_omega_np(
    sigma_re: np.ndarray, sigma_im: np.ndarray, omega_vec: np.ndarray,
    t_node: complex, alpha_eff: complex, omega_sign: float, pref: float,
    project_code: int,
) -> np.ndarray:
    """Numpy mirror of :func:`_project_tau_onto_omega` for host-side use.

    Matches the jax version exactly: code=0 ("full") returns the full
    complex product; code=1 ("imag") returns ``coeff_re·σ_im +
    coeff_im·σ_re`` (a real array up-cast to complex128 with Im=0).
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
#  Sigma accumulators — one interface, two strategies.
#
#  The tau loop doesn't care whether its outputs land in a GPU buffer or on
#  disk; it just needs to add per-tau contributions and knows when a window
#  boundary falls.  The two implementations differ only in what "add" means.
# ---------------------------------------------------------------------------

class _SigmaAccumulator:
    """Minimal protocol used by _integrate_tau_windows_for_branch.

    Lifecycle per branch::

        acc = AccumulatorCls(shape, gpu_mesh, omega_vec)    # ω_vec is branch-scoped
        for each window:
            acc.begin_window(window, scale=scale)           # window carries its own metadata
            for each tau:
                acc.add_tau(σ_re, σ_im, t_c, α_eff_c)       # only per-τ data
            acc.end_window()
        acc.finalize()

    ``window`` is the same :class:`_SigmaWindow` built by
    ``_build_windows_for_branch`` — accumulators read whatever they need
    off it (``window.omega_sign``, ``window.prefactor``,
    ``window.project_code``) instead of being fed a parallel stream of
    scalars.  ``t_c`` / ``α_eff_c`` are Python/host complex scalars
    computed once in :func:`minimax_tau_integrate_sigma`.
    """
    def begin_window(self, window: '_SigmaWindow', *, scale: float) -> None: ...
    def add_tau(self, sigma_re, sigma_im, t_c: complex, alpha_eff_c: complex) -> None: ...
    def end_window(self) -> None: ...
    def finalize(self) -> jax.Array | None: ...


class _HostOmegaAccumulator(_SigmaAccumulator):
    """Σ_c(ω, k, m, n) held as per-rank numpy tiles — never GPU HBM.

    σ(τ) arrives already sharded (m_X, n_Y) from the reduce-scatter tail
    of ``_sigma_kij_kernel``; ``addressable_data(0)`` is this rank's
    local tile (one GPU per process), exactly the shard it owns in the
    final Σ accumulator.  The ω-kernel + accumulate runs in numpy — no
    JAX sharding machinery, no collectives, no shard_map.

    Async pipeline: each τ calls ``copy_to_host_async`` on the local σ
    shards and appends to a small pending deque; the host actually reads
    (and accumulates) the σ tile ``lag`` iterations later.  That lets
    GPU work on τ_{k+lag} overlap with the numpy accumulate of τ_k.
    ``end_window`` drains the queue.

    At :meth:`finalize`, per-rank tiles are reassembled into a
    process-sharded ``jax.Array`` via ``make_array_from_process_local_data``
    so downstream (``per_half + sigma_kij``, ``_to_host_np``) sees the
    same interface as the on-GPU accumulator.
    """

    def __init__(self, shape: tuple[int, int, int, int], gpu_mesh: Mesh,
                 omega_vec: jax.Array, *, lag: int = 2):
        self._shape = shape
        self._sharding = NamedSharding(gpu_mesh, P(None, None, 'x', 'y'))
        self._local_shape = self._sharding.shard_shape(shape)
        self._omega_vec_np = np.asarray(jax.device_get(omega_vec),
                                        dtype=np.complex128)
        self._total = np.zeros(self._local_shape, dtype=np.complex128)
        self._win_acc: np.ndarray | None = None
        self._lag = int(lag)
        from collections import deque
        self._pending: deque = deque()
        # Window-scoped scalars cached at begin_window.
        self._omega_sign_f: float = 0.0
        self._pref_f: float = 0.0
        self._project_code: int = 0

    def begin_window(self, window: _SigmaWindow, *, scale: float) -> None:
        self._win_acc = np.zeros(self._local_shape, dtype=np.complex128)
        self._pending.clear()
        self._omega_sign_f = float(window.omega_sign)
        self._pref_f       = float(window.prefactor * scale)
        self._project_code = window.project_code

    def add_tau(self, sigma_re, sigma_im, t_c: complex, alpha_eff_c: complex) -> None:
        # Grab local shard handles and start the D2H copy now — do NOT
        # materialize to numpy yet.  ``copy_to_host_async`` returns the
        # same shard object with the transfer kicked off in the background.
        local_re = sigma_re.addressable_data(0)
        local_im = sigma_im.addressable_data(0)
        local_re.copy_to_host_async()
        local_im.copy_to_host_async()
        self._pending.append((local_re, local_im, t_c, alpha_eff_c))
        if len(self._pending) > self._lag:
            self._drain_one()

    def _drain_one(self) -> None:
        local_re, local_im, t_c, alpha_eff_c = self._pending.popleft()
        # ``np.asarray`` of an addressable shard waits on the D2H we
        # kicked off earlier — by now the transfer has had ``lag``
        # iterations of GPU work to overlap with.
        sig_re = np.asarray(local_re)
        sig_im = np.asarray(local_im)
        self._win_acc += _project_tau_onto_omega_np(
            sig_re, sig_im, self._omega_vec_np,
            t_c, alpha_eff_c, self._omega_sign_f, self._pref_f, self._project_code,
        )

    def end_window(self) -> None:
        assert self._win_acc is not None
        while self._pending:
            self._drain_one()
        self._total += self._win_acc
        self._win_acc = None

    def finalize(self) -> jax.Array:
        return jax.make_array_from_process_local_data(
            self._sharding, self._total, global_shape=self._shape)


class _StreamedH5Accumulator(_SigmaAccumulator):
    """Project each tau contribution in ω-batches and hand to a writer callable.

    The writer is expected to read-modify-write the backing HDF5 dataset;
    this class is agnostic to the storage (rank-0 h5py, SlabIO, …).

    Note on the FFI flush path (future work, comment-only here): a third
    accumulator — _CollectiveFlushSlabIoAccumulator — would keep the running Σ
    sharded (m_X, n_Y) on GPU (the _make_project_ri_reduce_scatter layout),
    stack many τ contributions per window without flushing, and at end_window() issue
    a single collective parallel-HDF5 write via SlabIO.write_slab against
    a pre-opened zarr-style (n_ω, n_k, m, n) dataset.  This removes the
    per-τ read-modify-write roundtrip that makes _StreamedH5Accumulator
    catastrophic at multi-process scale.  Implement when the upstream
    reduce-scatter project lands (without it, there's no point — σ^τ is
    still gathered on every rank).
    """
    def __init__(self, writer: Callable[[np.ndarray, jax.Array], None],
                 omega_vec: jax.Array, *,
                 omega_global_idx: np.ndarray, omega_batch_size: int):
        self._writer = writer
        self._omega_vec = omega_vec
        self._omega_global_idx = np.asarray(omega_global_idx, dtype=np.int64)
        self._batch = int(max(1, omega_batch_size))
        self._omega_sign_j = None
        self._pref_j = None
        self._project_code_j = None

    def begin_window(self, window: _SigmaWindow, *, scale: float) -> None:
        self._omega_sign_j   = jnp.asarray(float(window.omega_sign),       dtype=jnp.float64)
        self._pref_j         = jnp.asarray(float(window.prefactor * scale), dtype=jnp.float64)
        self._project_code_j = jnp.asarray(window.project_code,            dtype=jnp.int32)

    def add_tau(self, sigma_re, sigma_im, t_c: complex, alpha_eff_c: complex) -> None:
        t_j     = jnp.asarray(t_c,         dtype=jnp.complex128)
        alpha_j = jnp.asarray(alpha_eff_c, dtype=jnp.complex128)
        n_omega = int(self._omega_vec.shape[0])
        for ibeg in range(0, n_omega, self._batch):
            iend = min(ibeg + self._batch, n_omega)
            batch_proj = _project_tau_onto_omega(
                sigma_re, sigma_im, self._omega_vec[ibeg:iend],
                t_j, alpha_j, self._omega_sign_j,
                self._pref_j, self._project_code_j,
            )
            self._writer(self._omega_global_idx[ibeg:iend], batch_proj)

    def end_window(self) -> None:
        pass

    def finalize(self) -> None:
        return None
