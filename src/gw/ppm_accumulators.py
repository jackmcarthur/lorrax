"""ω-projection + accumulator for the Σ_c(ω) GN-PPM integration.

"What happens to σ^τ after the kernel returns": the ω-kernel projection (one
numpy function — the single source of truth), the accumulator protocol, and
two accumulators: `_TauAccumulator` (async-D2H per τ; its `_MemoryTileSink`
holds each finished window as per-rank host Σ tiles) and
`DeviceOmegaAccumulator` (folds each σ^τ into a sharded on-device ω-cube;
nothing moves to host before finalize).
(The streamed-h5 sink twin and the `kij_stream` accumulation mode were
REMOVED 2026-07-31: single-process-only, superseded by the host tiles +
sharded ω-cube layout.)  The τ loop just adds per-τ contributions and
knows when a window boundary falls.

Imports ``_SigmaWindow`` (type only) from the ``ppm_windows`` leaf; nothing here
imports the driver or the device kernel.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache


import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from common.collectives import device_put_process_local

from .ppm_windows import _SigmaWindow


def _omega_coefficient(xp, omega, t, alpha, sign, prefactor, e_ref=0.0):
    """Frequency coefficient shared by the host and device folds."""
    return ((prefactor * alpha)
            * xp.exp(-1j * (e_ref - sign * omega) * t))


def _coeff_times_x(np_, coeff, X):
    """``coeff[ω] · X``, with or without a LEADING BAND-BRACKET AXIS on X.

    ``X`` is ``(nk, i, j)`` when the caller planned no brackets and
    ``(n_bracket, nk, i, j)`` when it planned some.  Both are live contracts
    and this helper is the one place that knows the difference:

    * ``ppm_tau_kernel._NO_BRACKETS`` states it outright — "the band-bracket
      plan a caller gets when it asks for none: ONE bracket over every band,
      and NO leading bracket axis on the output".  The MPA / shared-multipole
      shape and every direct consumer of this function (notably
      ``tests/test_ppm_crossing_completion.py``, which exercises the crossing
      completion on a bare window) are un-bracketed.
    * The band-extrapolation path plans three brackets and ships the 4-D form.

    THE BRACKET AXIS IS NOT FUSED WITH ANYTHING.  It stays a leading INDEX;
    ω is inserted after it, and the (i, j) band axes are untouched.  That is
    the same separability rule the bracket loop in ``ppm_tau_kernel`` keeps —
    the partition is an index operation on the band axis and never mixes with
    an elementwise factor along it.
    """
    if X.ndim == 4:                     # (n_bracket, nk, i, j)
        return coeff.reshape(1, -1, 1, 1, 1) * X[:, None, ...]
    if X.ndim == 3:                     # (nk, i, j) -- no bracket axis
        return coeff.reshape(-1, 1, 1, 1) * X[None, ...]
    raise ValueError(
        f"_project_tau_onto_omega_np: sigma tile must be (nk, i, j) or "
        f"(n_bracket, nk, i, j); got shape {tuple(X.shape)}")


def _project_tau_onto_omega_np(
    sigma_re: np.ndarray, sigma_im: np.ndarray | None, omega_vec: np.ndarray,
    t_node: complex, alpha_eff: complex, omega_sign: float, pref: float,
    project_code: int,
) -> np.ndarray:
    """Apply the ω-kernel exp(i·ω_sign·ω·t) and project onto σ channels (host).

    This is the **single** ω-projector in LORRAX (both sinks call it — there is
    no jax mirror to keep in sync).  σ^τ arrives as ``(n_brk, nk, i, j)`` — the
    band-bracket axis LEADS (``gw.band_extrapolation``; length 1 in the
    ordinary case) — and the ω axis is inserted BEHIND it, so the result is
    ``(n_brk, n_ω, nk, i, j)``.  The ω kernel is elementwise in every other
    index, so a bracket is carried through untouched; that is exactly why the
    axis can lead here without the projector knowing what a bracket is:

        contrib[b, ω, k, i, j] = pref · α_eff · exp(i·sign·ω·t) · P(σ_re, σ_im)[b]

    The window's ``project_code`` names the consumer:

        code=0 ("full")  Laplace window (single/stripe/slab) — keep the full
                         complex product coeff·X with X = ψ†σψ = S_R + i·S_I.
                         Only that recombined complex object is ever consumed
                         here, which is what licenses the merged single-chain
                         plan (the default and only Laplace path — owner
                         order 2026-07-28).
        code=1 ("imag")  Crossing window — return the ONE-SIDED half
                         Z_τ = coeff·X with X = S_R + i·S_I, i.e. exactly
                         the same complex product as code=0.  The crossing
                         window's τ grid is one-sided (τ_j > 0), and the
                         quadrature it stands in for is a SINE sum,
                         G(u) ≈ Σ_l α_l sin(τ_l u) at the pole argument
                         u_μν = ω̃ − E_A − Ω_μν; sin(τu) = (e^{iτu} −
                         e^{−iτu})/2i, and only the e^{+iτu} half is on the
                         grid.  The missing half is the BAND ADJOINT of the
                         first — see :func:`_complete_one_sided_tau` for
                         the derivation — so the window is closed by
                         (Z − Z†)/2i with Z = Σ_τ Z_τ, applied ONCE per
                         window by the accumulator (linearity: the
                         completion commutes with the τ sum) and, crucially,
                         on the GLOBAL band matrix rather than per shard.
                         This consumer therefore returns an INCOMPLETE
                         contribution and must not be used by anything that
                         does not close the window that way.

    Channel carriers mirror the codes one-to-one.  Laplace (code=0): the
    merged kernel ships X = S_R + i·S_I as ONE complex tile —
    ``sigma_im is None`` and ``sigma_re`` carries X (bilinearity:
    ψ†σψ = ψ†σ_Rψ + i·ψ†σ_Iψ, so coeff·X equals the recombined two-channel
    product to GEMM-association roundoff; gated at 1e-12).  Crossing
    (code=1): σ^τ arrives as the real/imag pair ``(sigma_re, sigma_im)``,
    and this consumer recombines them into X on host.  Since the crossing
    completion turned out to need only X (2026-08-09), that split is now a
    channel-plan/perf choice, NOT a mathematical necessity — the old
    rationale ("the crossing math cannot be recovered from X") was the
    elementwise-Im defect talking, and collapsing crossing onto the merged
    single-chain kernel is a real, owner-scoped option that is deliberately
    NOT taken here (it would move the τ-kernel dispatch, the collective
    payloads and the D2H volume, none of which this physics fix should
    touch).  Until it is taken, the carrier contract stands and any
    cross-pairing is a dispatch bug and raises: a merged tile at code=1,
    and a two-channel pair at code=0 (whose recombine branch died when the
    merge became the default).
    """
    # ``pref`` is folded into the (n_ω,)-sized coeff instead of multiplying
    # the full (n_ω, nk, i, j) contrib afterwards — removes one full-size
    # array pass + temporary per τ.  Value-identical, NOT bitwise (the
    # per-element product associates (pref·c)·σ instead of pref·(c·σ));
    # gated by the 1e-12 output parity suite at nb=128 AND nb=256
    # (host_accum follow-up, 2026-07-28).  The final ``.astype`` copy is
    # likewise dropped: both branches already produce complex128
    # (``np.asarray`` with a matching dtype is a no-copy view) — that one
    # is bit-exact.
    coeff = _omega_coefficient(
        np, omega_vec, t_node, alpha_eff, omega_sign, pref)
    if sigma_im is None:
        # Laplace plan: sigma_re IS the complex X = S_R + i·S_I.
        if project_code != 0:
            raise ValueError(
                "merged single-chain σ (X = ψ†σψ) reached a crossing "
                "consumer (project_code=1) — the crossing window needs the "
                "(S_R, S_I) pair and must dispatch the two-channel kernel "
                "(ppm_sigma window dispatch bug).")
        contrib = _coeff_times_x(np, coeff, sigma_re)
        return np.asarray(contrib, dtype=np.complex128)
    # Two-channel (S_R, S_I) pair: crossing windows only.  The Laplace
    # recombine branch (pair at code=0) died when the merge became the
    # default (owner order 2026-07-28) — reaching it is a dispatch bug.
    if project_code == 0:
        raise ValueError(
            "two-channel (sigma_re, sigma_im) pair reached a Laplace "
            "consumer (project_code=0) — Laplace windows always dispatch "
            "the merged single-chain kernel and ship X = ψ†σψ as (X, None) "
            "(ppm_sigma window dispatch bug).")
    if project_code != 1:
        raise ValueError(f"Unknown project_code {project_code}")
    # Crossing window — the ONE-SIDED half Z_τ = coeff·X only.  What used to
    # stand here,
    #     contrib = coeff_re·S_I + coeff_im·S_R,
    # is the ELEMENTWISE Im[coeff·σ(μ,ν)] taken scalar by scalar before the
    # ψ projection.  That is the completion of the sine sum only where σ^τ is
    # complex-symmetric — under time reversal, only at k ≡ −k mod G — so it
    # broke the Σ_c k-star relation at every non-TRIM k while leaving the
    # TRIM k exact (KNOWN_FAILURES, 2026-08-09).  The correct completion is
    # the band-adjoint one in _complete_one_sided_tau, and it is a WINDOW-level
    # operator: it couples element (i, j) to element (j, i), which the
    # (m_X, n_Y)-sharded per-shard tiles here cannot see.
    X = sigma_re + 1j * sigma_im
    contrib = _coeff_times_x(np, coeff, X)
    return np.asarray(contrib, dtype=np.complex128)


# ---------------------------------------------------------------------------
#  The crossing window's one-sided-τ completion.
# ---------------------------------------------------------------------------

def _band_axis_partition_counts(sigma_sharding) -> tuple[int, int]:
    """How many mesh slices σ^τ's two band axes ``(m, n)`` are cut into.

    ``sigma_sharding`` is whatever sharding the ``(..., m_X, n_Y)`` σ^τ tiles
    arrived with — ``P(None, None, 'x', 'y')`` on the production mesh, whose
    leading axes are the band bracket and k.  The two band axes are read as
    the LAST two entries of the spec rather than by absolute position, so a
    leading axis added in front of k cannot silently shift which axis is
    tested for splitting.  A sharding we cannot read (``None``, or anything
    that is not a ``NamedSharding``) is reported as UNSPLIT, which is correct
    for the single-device and plain-numpy callers and is why the caller must
    never manufacture one.
    """
    mesh = getattr(sigma_sharding, 'mesh', None)
    spec = getattr(sigma_sharding, 'spec', None)
    if mesh is None or spec is None or len(spec) < 3:
        return 1, 1

    def _count(entry) -> int:
        if entry is None:
            return 1
        names = (entry,) if isinstance(entry, str) else tuple(entry)
        n = 1
        for a in names:
            n *= int(mesh.shape[a])
        return n

    return _count(spec[-2]), _count(spec[-1])


@lru_cache(maxsize=8)
def _antiherm_band_fn(sharding: NamedSharding):
    """``Z -> (Z − Z†)/2i`` on the trailing (i, j) axes, output re-sharded back.

    Cached per sharding so the resharding collective is compiled once per Σ
    stage rather than once per window.
    """
    return jax.jit(
        lambda Z: (Z - jnp.conj(jnp.swapaxes(Z, -1, -2))) / 2j,
        out_shardings=sharding,
    )


def _complete_one_sided_tau(win_shards, shard_devices, sigma_sharding):
    """Close a crossing window: ``Z -> (Z − Z†)/2i`` on the GLOBAL band matrix.

    **What is being completed.**  A crossing (HGL) window's minimax rule fits
    its regularization target as a SINE sum on a one-sided grid,
    ``G(u) ≈ Σ_l α_l sin(τ_l u)`` with ``τ_l > 0``
    (``minimax_screening.solve_phase_minimax_bandwidth``).  Collecting every
    phase the τ kernel and this module apply — ``e^{i·sign·ω·τ}`` from the
    ω-kernel, ``e^{-iτ(E_A - E_ref_A)}`` from ``build_G_tau``,
    ``e^{-iτ(Ω_q - E_ref_B)}`` from ``_build_W_t_q``, and the
    ``e^{-i·E_ref_sum·τ}`` folded into α_eff, whose reference shifts cancel
    exactly — one τ node contributes, per band pair (i, j),

        Z_τ(i,j) = pref·α_l · Σ_{n,q,μ,ν} [ψ*_i(μ)ψ_n(μ)][ψ*_n(ν)ψ_j(ν)]
                                          · B_q(μ,ν) · e^{+iτ·u},
        u = sign·ω − E_A(k−q, n) − Ω_q(μ,ν),

    which is the ``e^{+iτu}`` half of ``sin(τu) = (e^{+iτu} − e^{−iτu})/2i``.
    The half that is NOT on the grid is obtained by conjugating the (i, j)
    pair, not each scalar: writing out ``conj(Z_τ(j,i))``, relabelling
    μ ↔ ν and using the two premises the GN-PPM fit already carries —
    ``B_q† = B_q`` and ``Ω_q`` real symmetric (derived in
    ``docs/dev/notes/DERIVATION_channel_hermiticity.md`` §1.3, the same
    premises that make ``[σ^τ]† = σ^{−τ}``) — returns exactly the same sum
    with ``e^{−iτu}``.  Hence

        Σ_c^{crossing} = (Z − Z†)/2i,      Z = Σ_τ Z_τ,

    with † the adjoint on the QP band pair.  The completion is LINEAR, so
    applying it once to the window's τ-sum is identical to applying it per τ,
    and one application per window is what this function is.

    **Why it cannot live in the per-τ projector.**  ``(Z − Z†)`` couples
    element (i, j) to element (j, i).  Σ_c tiles are sharded
    ``P(None, None, 'x', 'y')`` — m over ``'x'``, n over ``'y'``
    (``ppm_tau_kernel._make_project_ri_reduce_scatter``) — so rank (a, b)
    holds the band BLOCK (rows a, cols b) and the partner of every one of
    its elements is on rank (b, a).  A ``swapaxes(-1, -2)`` inside
    ``_project_tau_onto_omega_np`` would transpose within the local block:
    on a square mesh the shapes match and it returns a WRONG answer in
    silence; on a non-square mesh it raises a broadcast error.  Both were
    verified on a 2×2 CPU mesh (``tests/test_ppm_crossing_completion.py``).
    So the adjoint is taken here, on the assembled array:

      * band axes unsplit (``p_m = p_n = 1`` — single device, or any mesh
        that does not cut the QP band window): pure numpy, no device hop,
        which is also the path every unit test exercises;
      * band axes split: the per-rank tiles are published as ONE sharded
        ``jax.Array`` (``make_array_from_single_device_arrays``, each process
        supplying exactly its addressable shards — correct single-process
        multi-GPU and multi-process alike, same idiom as
        ``_MemoryTileSink.result``), the adjoint is taken there so XLA owns
        the transpose collective, and the result is read back to the same
        per-rank tiles.

    Cost, stated because this module's contract is "no collectives in the τ
    loop": ONE round trip per crossing window, and crossing windows are one
    window on each of two branches (cond +ω, val −ω), i.e. **2 per Σ stage**
    — not per τ, of which there are hundreds.  The transient is one window's
    Σ tile, the same object ``_MemoryTileSink.result`` already device_puts.
    """
    p_m, p_n = _band_axis_partition_counts(sigma_sharding)
    if p_m == 1 and p_n == 1:
        # The whole band matrix is local: the numpy adjoint IS the global one.
        return [(t - np.conj(np.swapaxes(t, -1, -2))) / 2j for t in win_shards]
    mesh = sigma_sharding.mesh
    spec = tuple(sigma_sharding.spec)
    # σ^τ is (n_brk, nk, m, n); the window tiles carry the ω axis INSERTED
    # behind the leading bracket axis, i.e. (n_brk, n_ω, nk, m, n).
    sharding_w = NamedSharding(mesh, P(spec[0], None, *spec[1:]))
    t0 = win_shards[0]
    gshape = tuple(int(v) for v in t0.shape[:-2]) + (
        int(t0.shape[-2]) * p_m, int(t0.shape[-1]) * p_n)
    arrays = [jax.device_put(t, d) for t, d in zip(win_shards, shard_devices)]
    completed = _antiherm_band_fn(sharding_w)(
        jax.make_array_from_single_device_arrays(gshape, sharding_w, arrays))
    by_device = {s.device: np.asarray(s.data)
                 for s in completed.addressable_shards}
    return [by_device[d] for d in shard_devices]


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

    For Laplace (project="full") windows ``add_tau`` receives
    ``(X, None, t_c, α_eff_c)`` — the single complex X = ψ†σψ = S_R + i·S_I
    in the ``sigma_re`` slot (the default and only Laplace channel plan);
    crossing windows always deliver the (σ_re, σ_im) pair.
    """
    def begin_window(self, window: '_SigmaWindow') -> None: ...
    def add_tau(self, sigma_re, sigma_im, t_c: complex, alpha_eff_c: complex) -> None: ...
    def end_window(self) -> None: ...
    def finalize(self) -> jax.Array | None: ...


class _WindowAccum:
    """Per-window accumulation context carried by the pending deque.

    Holds the window-scoped projector scalars and the running per-shard
    tiles, so a pending τ entry can be drained AFTER its window closed —
    the cross-window pipelining that keeps the device queue full at
    ``end_window`` (the old flush-per-window drain idled the device for
    ``lag`` kernel walls at every window boundary; measured 5.8 s of the
    71.2 s sigma.exec at b300/P=16, job 7884656).
    """

    __slots__ = ('omega_sign_f', 'pref_f', 'project_code', 'omega_vec',
                 'omega_indices', 'win_shards', 'pending_count', 'closed')

    def __init__(self, window: '_SigmaWindow', omega_vec: np.ndarray):
        self.omega_sign_f = float(window.omega_sign)
        self.pref_f       = float(window.prefactor)
        self.project_code = window.project_code
        self.omega_indices = (
            None if window.omega_indices is None
            else np.asarray(window.omega_indices, dtype=np.int64))
        omega_values = getattr(window, "omega_values", None)
        if omega_values is not None and self.omega_indices is None:
            raise ValueError("omega_values requires omega_indices")
        self.omega_vec = (
            omega_vec if self.omega_indices is None
            else (omega_vec[self.omega_indices] if omega_values is None
                  else np.asarray(omega_values, dtype=np.complex128)))
        if (self.omega_indices is not None
                and self.omega_vec.shape != self.omega_indices.shape):
            raise ValueError("omega_values must match omega_indices")
        self.win_shards: list | None = None
        self.pending_count = 0
        self.closed = False


class _TauAccumulator(_SigmaAccumulator):
    """Async-D2H deque + the single numpy ω-projector, feeding a sink.

    σ(τ) arrives already sharded (m_X, n_Y) from the reduce-scatter tail of
    ``_sigma_kij_kernel``.  Per τ we kick off ``copy_to_host_async`` on **every
    addressable shard** — not just shard 0 — because a single process can own
    more than one GPU (`lxalloc N` interactive, an end-user 2-GPU desktop; the
    code ships to arbitrary device counts, so no shard-0 assumption may survive
    — this was Bug C).  The host reads (and projects/accumulates) each shard's
    σ tile ``lag`` iterations later, so GPU work on τ_{k+lag} overlaps with the
    numpy accumulate of τ_k.

    **The deque persists across window boundaries** (2026-08-01): each pending
    entry carries its window's :class:`_WindowAccum`, so ``end_window`` merely
    CLOSES the window — the tail τ's drain while the NEXT window's kernels are
    already dispatched, and the device queue never empties inside a branch.
    A window's tiles reach the sink when its last entry drains; entries are
    FIFO, so windows complete in order and the sink sees the exact
    consume_window sequence (and per-window float-add order) of the old
    flush-per-window schedule — bit-identical results, different schedule.
    Only the BRANCH tail (``finalize``/``finalize_host_tiles``) flushes.

    The ω-kernel + accumulate runs entirely in numpy on host — no JAX sharding
    machinery, no collectives, no shard_map IN THE τ LOOP.  Each shard's
    per-τ contribution is independent (the projector is elementwise in
    (k, i, j) times the ω-kernel), so per-shard projection is exact.

    The one thing that is NOT elementwise in (i, j) is a crossing window's
    completion of its one-sided τ grid, ``(Z − Z†)/2i``, which pairs band
    element (i, j) with (j, i) and therefore rank (a, b) with rank (b, a).
    That is a WINDOW-level operator (linearity lets it commute out of the τ
    sum), it runs in :meth:`_finish_window` via
    :func:`_complete_one_sided_tau`, and it is the sole place this class
    touches a collective — twice per Σ stage, at window boundaries, never
    inside the τ loop.
    """

    def __init__(self, omega_vec: jax.Array, sink: '_WindowSink', *, lag: int = 2):
        self._omega_vec_np = np.asarray(jax.device_get(omega_vec),
                                        dtype=np.complex128)
        self._sink = sink
        self._lag = int(lag)
        self._pending: deque = deque()
        self._shard_index: list[tuple] | None = None
        self._shard_devices: list | None = None
        # σ^τ's own sharding, captured off the first tile: the ONLY thing that
        # says whether the QP band axes are split across ranks, which is what
        # _complete_one_sided_tau has to know to take a global band adjoint.
        self._sigma_sharding = None
        # Current (open) window context; pending entries hold their own.
        self._cur: _WindowAccum | None = None

    def begin_window(self, window: _SigmaWindow) -> None:
        # Pending entries from earlier (closed) windows stay queued — they
        # drain under THIS window's dispatches.  A begin without a matching
        # end_window is a driver bug; catch it rather than mis-attribute τ's.
        if self._cur is not None and not self._cur.closed:
            raise RuntimeError(
                "_TauAccumulator.begin_window: previous window never closed")
        self._cur = _WindowAccum(window, self._omega_vec_np)

    def add_tau(self, sigma_re, sigma_im, t_c: complex, alpha_eff_c: complex) -> None:
        # Grab EVERY local shard handle and start the D2H copies now — do NOT
        # materialize to numpy yet.  ``copy_to_host_async`` returns the same
        # shard object with the transfer kicked off in the background.
        # ``sigma_im is None`` = Laplace window: ``sigma_re`` carries the
        # single complex X = S_R + i·S_I, and the per-τ D2H volume HALVES
        # (one c128 tile per shard instead of the re/im pair).
        shards_re = sigma_re.addressable_shards
        shards_im = (sigma_im.addressable_shards
                     if sigma_im is not None else None)
        if self._shard_index is None:
            self._shard_index = [s.index for s in shards_re]
            self._shard_devices = [s.device for s in shards_re]
            self._sigma_sharding = getattr(sigma_re, 'sharding', None)
        handles = []
        if shards_im is None:
            for sr in shards_re:
                sr.data.copy_to_host_async()
                handles.append((sr.data, None))
        else:
            for sr, si in zip(shards_re, shards_im):
                sr.data.copy_to_host_async()
                si.data.copy_to_host_async()
                handles.append((sr.data, si.data))
        assert self._cur is not None and not self._cur.closed
        self._cur.pending_count += 1
        self._pending.append((handles, t_c, alpha_eff_c, self._cur))
        if len(self._pending) > self._lag:
            self._drain_one()

    def _drain_one(self) -> None:
        from common import timing
        handles, t_c, alpha_eff_c, wctx = self._pending.popleft()
        if wctx.win_shards is None:
            wctx.win_shards = [None] * len(handles)
        for d, (hr, hi) in enumerate(handles):
            # ``np.asarray`` of an addressable shard waits on the D2H we kicked
            # off earlier — by now the transfer has had ``lag`` iterations of
            # GPU work to overlap with.
            #
            # The two rows below make the parent 'sigma.tau.host_accum' row
            # DISCRIMINATE (QUALITY_PATTERNS addendum): 'd2h_wait' blocks on
            # τ_{i-lag}'s device COMPUTE + transfer, so on a device-bound τ
            # loop it absorbs the kernel time (it read 72.7 s of an 87 s
            # sigma.exec at run_L1_b256 nb=256 — that is the DEVICE wall
            # surfacing here, not host work); 'omega_project' is the pure
            # numpy ω-kernel + accumulate (0.71 s/176 τ at nb=128, job
            # 7878038 pass b).  A future "host_accum exploded" reading must
            # cite the omega_project row, not the parent (2026-07-28).
            with timing.section("sigma.tau.d2h_wait"):
                sr = np.asarray(hr)
                si = None if hi is None else np.asarray(hi)
            with timing.section("sigma.tau.omega_project"):
                proj = _project_tau_onto_omega_np(
                    sr, si, wctx.omega_vec,
                    t_c, alpha_eff_c, wctx.omega_sign_f, wctx.pref_f,
                    wctx.project_code,
                )
                if wctx.win_shards[d] is None:
                    wctx.win_shards[d] = np.zeros_like(proj)
                wctx.win_shards[d] += proj
        wctx.pending_count -= 1
        if wctx.closed and wctx.pending_count == 0:
            # Last τ of a closed window: hand its tiles to the sink now.
            # FIFO drains ⇒ windows complete in dispatch order.
            self._finish_window(wctx)

    def _finish_window(self, wctx: _WindowAccum) -> None:
        """Every path out of a window goes through here — sink the tiles, and
        for a CROSSING window complete its one-sided τ grid first.

        The completion (``(Z − Z†)/2i``, :func:`_complete_one_sided_tau`) is
        a window-level operator on the global band matrix, not a per-τ
        elementwise one, so this is the only place it can run: after the
        window's last τ has drained and before the sink adds the window into
        the running Σ total.  Laplace windows pass straight through.
        """
        shards = wctx.win_shards
        if wctx.project_code == 1:
            shards = _complete_one_sided_tau(
                shards, self._shard_devices, self._sigma_sharding)
        if wctx.omega_indices is None:
            self._sink.consume_window(
                shards, self._shard_index, self._shard_devices)
        else:
            self._sink.consume_window(
                shards, self._shard_index, self._shard_devices,
                omega_indices=wctx.omega_indices)
        wctx.win_shards = None

    def end_window(self) -> None:
        # No flush: close the window and keep the pipeline primed.  The
        # tail τ's drain under the next window's dispatches; an all-drained
        # window (short window, or the branch's last add_tau already pushed
        # everything through) is consumed immediately.
        assert self._cur is not None
        self._cur.closed = True
        if self._cur.pending_count == 0:
            # Every τ of this window already drained (short window vs lag).
            # A window always has ≥1 τ, so its tiles exist.
            assert self._cur.win_shards is not None
            self._finish_window(self._cur)

    def _drain_all(self) -> None:
        while self._pending:
            self._drain_one()

    def finalize(self) -> jax.Array | None:
        self._drain_all()
        return self._sink.result()

    def finalize_host_tiles(self):
        """Branch tail WITHOUT the device round trip — see
        :meth:`_MemoryTileSink.host_tiles`."""
        self._drain_all()
        return self._sink.host_tiles()


@lru_cache(maxsize=8)
def _device_output_zeros(shape, sharding):
    return jax.jit(
        lambda: jnp.zeros(shape, dtype=jnp.complex128),
        out_shardings=sharding)


def _omega_fold(acc, sigma, coeff, omega_axis):
    """Add ``coeff[omega] * sigma`` with an explicit omega-axis position."""
    coeff_shape = ((1,) * omega_axis + (coeff.shape[0],)
                   + (1,) * (acc.ndim - omega_axis - 1))
    return acc + coeff.reshape(coeff_shape) * jnp.expand_dims(
        sigma, axis=omega_axis)


@lru_cache(maxsize=16)
def _device_omega_add(sharding, omega_axis):
    return jax.jit(
        lambda acc, sigma, coeff: _omega_fold(
            acc, sigma, coeff, omega_axis),
        donate_argnums=(0,), out_shardings=sharding)


@lru_cache(maxsize=8)
def _device_output_add(sharding):
    return jax.jit(
        lambda total, window: total + window,
        donate_argnums=(0,), out_shardings=sharding)


class DeviceOmegaAccumulator:
    """Fold one transient sigma(tau) tile into a sharded omega cube.

    No tau history is retained.  A full/Laplace window accumulates directly
    into the result.  A one-sided sine window uses one temporary omega cube,
    applies ``(Z-Z†)/(2i)`` once after its last tau, then adds it to the same
    result.  ``alpha`` is the quadrature weight before reference rephasing;
    combining ``E_ref_sum`` and omega in one exponential avoids separately
    overflowing two factors whose product is well conditioned.
    """

    def __init__(self, omega_vec, *, shape, sharding, omega_axis):
        self._shape = tuple(int(n) for n in shape)
        self._sharding = sharding
        self._replicated = NamedSharding(sharding.mesh, P())
        self._omega = np.asarray(jax.device_get(omega_vec), np.complex128)
        self._omega_axis = int(omega_axis)
        if self._omega_axis < 0:
            self._omega_axis += len(self._shape)
        if not 0 <= self._omega_axis < len(self._shape):
            raise ValueError(
                "DeviceOmegaAccumulator: omega_axis outside output rank")
        if self._shape[self._omega_axis] != self._omega.size:
            raise ValueError(
                "DeviceOmegaAccumulator: shape[omega_axis] must equal "
                "n_omega")
        # Per-rank ω-cube: nω·nk·(nb_pad/p_x)·(nb_pad/p_y)·16 bytes (c128),
        # ×2 while a crossing window holds its temporary cube open.
        self._total = _device_output_zeros(self._shape, sharding)()
        self._window = None
        self._coeff = None
        self._index = 0

    def begin_window(self, t, alpha, *, omega_sign, prefactor,
                     e_ref_sum=0.0, antihermitian=False,
                     omega_indices=None, omega_values=None):
        if self._coeff is not None:
            raise RuntimeError("previous frequency window is still open")
        t = np.asarray(jax.device_get(t), np.complex128)
        alpha = np.asarray(jax.device_get(alpha), np.complex128)
        if t.ndim != 1 or alpha.shape != t.shape or t.size == 0:
            raise ValueError("t and alpha must be nonempty equal vectors")
        if omega_indices is None:
            if omega_values is not None:
                raise ValueError("omega_values requires omega_indices")
            omega = self._omega
            indices = None
        else:
            indices = np.asarray(omega_indices, dtype=np.int64)
            omega = np.asarray(omega_values, dtype=np.complex128)
            if (indices.ndim != 1 or omega.shape != indices.shape
                    or np.any(indices < 0) or np.any(indices >= self._omega.size)):
                raise ValueError("invalid active frequency indices/values")
        active = np.asarray(_omega_coefficient(
            np, omega[None, :], t[:, None], alpha[:, None],
            float(omega_sign), float(prefactor), float(e_ref_sum)),
            np.complex128)
        if indices is None:
            self._coeff = active
        else:
            self._coeff = np.zeros(
                (t.size, self._omega.size), dtype=np.complex128)
            self._coeff[:, indices] = active
        self._index = 0
        self._window = (_device_output_zeros(
            self._shape, self._sharding)() if antihermitian else None)

    def precompile_tau_add(self, *, sigma_shape, sigma_sharding):
        """Compile the accumulator fold before the timed tau sweep marker."""
        sigma = jax.ShapeDtypeStruct(
            tuple(int(n) for n in sigma_shape), jnp.complex128,
            sharding=sigma_sharding)
        coeff = jax.ShapeDtypeStruct(
            (self._omega.size,), jnp.complex128,
            sharding=self._replicated)
        _device_omega_add(self._sharding, self._omega_axis).lower(
            self._total, sigma, coeff).compile()

    def add_tau(self, sigma_tau):
        if self._coeff is None:
            raise RuntimeError("no open frequency window")
        if self._index >= self._coeff.shape[0]:
            raise RuntimeError("more sigma(tau) tiles than quadrature nodes")
        coeff = device_put_process_local(
            self._coeff[self._index], self._replicated)
        self._index += 1
        if self._window is None:
            self._total = _device_omega_add(
                self._sharding, self._omega_axis)(
                self._total, sigma_tau, coeff)
        else:
            self._window = _device_omega_add(
                self._sharding, self._omega_axis)(
                self._window, sigma_tau, coeff)

    def end_window(self):
        if self._coeff is None:
            raise RuntimeError("no open frequency window")
        if self._index != self._coeff.shape[0]:
            raise RuntimeError("frequency window ended before all tau nodes")
        if self._window is not None:
            completed = _antiherm_band_fn(self._sharding)(self._window)
            self._total = _device_output_add(self._sharding)(
                self._total, completed)
        self._window = None
        self._coeff = None
        self._index = 0

    def add_direct(self, sigma_omega, omega_index, *, coefficient=1.0):
        """Add one exact direct-frequency tile outside the tau lifecycle.

        Direct reciprocal terms already carry their complete denominator,
        so there is no tau coefficient or open window.  Reusing the ordinary
        sharded omega-add primitive keeps the output layout and accumulation
        order identical to tau contributions while making the separate cost
        currency explicit at the call site.
        """
        if self._coeff is not None:
            raise RuntimeError(
                "cannot add a direct Sigma tile while a tau window is open")
        index = int(omega_index)
        if not 0 <= index < self._omega.size:
            raise ValueError(
                f"direct omega index {index} outside [0,{self._omega.size})")
        coeff = np.zeros(self._omega.size, dtype=np.complex128)
        coeff[index] = complex(coefficient)
        coeff = device_put_process_local(coeff, self._replicated)
        self._total = _device_omega_add(
            self._sharding, self._omega_axis)(
            self._total, sigma_omega, coeff)

    def finalize(self):
        if self._coeff is not None:
            raise RuntimeError("cannot finalize an open frequency window")
        return self._total


class _WindowSink:
    """Protocol for what a finished window's per-shard tiles become."""
    def consume_window(
        self, win_shards, shard_index, shard_devices, *, omega_indices=None,
    ) -> None: ...
    def result(self) -> jax.Array | None: ...


class _MemoryTileSink(_WindowSink):
    """Σ_c(ω, k, m, n) held as per-rank numpy tiles — never GPU HBM.

    The Σ driver's per-branch tail reads :meth:`host_tiles` (host numpy in,
    host numpy out — the single-gather-at-stage-end path); :meth:`result`
    is the original per-branch device-assembly variant, kept as the
    _WindowSink protocol's generic answer and for any future caller that
    genuinely wants a device Σ.

    Each window's per-shard tiles are added into a running per-shard total.  At
    :meth:`result`, each shard tile is placed back on the device that owned it
    (``jax.device_put``) and the shards are assembled into a process-sharded
    ``jax.Array`` via ``make_array_from_single_device_arrays`` — which is
    correct on single-process multi-GPU and multi-process alike (each process
    supplies exactly its addressable shards), unlike the old
    ``make_array_from_process_local_data`` fed a shard-0-sized tile.  Downstream
    (``sigma_kij_host`` add, ``_to_host_np``) sees the same (m_X, n_Y) sharding.
    """

    def __init__(self, shape: tuple[int, ...], sharding: NamedSharding,
                 *, omega_axis: int):
        self._shape = shape
        self._sharding = sharding
        self._omega_axis = int(omega_axis)
        if self._omega_axis < 0:
            self._omega_axis += len(self._shape)
        if not 0 <= self._omega_axis < len(self._shape):
            raise ValueError("_MemoryTileSink: omega_axis outside output rank")
        self._total_shards: list[np.ndarray] | None = None
        self._devices: list | None = None
        self._omega_clustered: bool | None = None
        # 4-D σ^τ shard indices (n_brk, nk, m, n) captured with the first
        # window — needed by host_tiles() to place each tile in the 5-D
        # global Σ.
        self._index: list[tuple] | None = None

    def consume_window(
        self, win_shards, shard_index, shard_devices, *, omega_indices=None,
    ) -> None:
        clustered = omega_indices is not None
        if self._omega_clustered is None:
            self._omega_clustered = clustered
        elif self._omega_clustered != clustered:
            raise RuntimeError(
                "_MemoryTileSink cannot mix clustered and full-omega "
                "windows in one branch")
        if self._total_shards is None:
            if omega_indices is None:
                # Incumbent full-omega path: retain its allocation and add
                # order exactly.
                self._total_shards = [np.zeros_like(w) for w in win_shards]
            else:
                self._total_shards = []
                for w in win_shards:
                    shape = list(w.shape)
                    shape[self._omega_axis] = int(
                        self._shape[self._omega_axis])
                    self._total_shards.append(
                        np.zeros(tuple(shape), dtype=w.dtype))
            self._devices = list(shard_devices)
            self._index = [tuple(ix) for ix in shard_index]
        for d, w in enumerate(win_shards):
            if omega_indices is None:
                self._total_shards[d] += w
            else:
                self._total_shards[d][:, omega_indices, ...] += w

    def host_tiles(self):
        """Per-shard host tiles, their 4-D global indices, and owning devices.

        The no-round-trip branch tail — scorecard AK.9's second named lever
        ("``_to_host_np(sigma_kij, tiled=False)`` is a P-INDEPENDENT gather
        — once per Σ branch, n_ω_branch·nk·nb²·16 bytes onto EVERY rank,
        ≈237 MB/branch at 606c/b160, growing as nb²").  The old tail:
        :meth:`result` re-uploads the per-rank host tiles (device_put),
        assembles a global sharded array, and the driver then
        process_allgather's the full (n_ω, nk, nb, nb) slab back to every
        rank's host — once PER BRANCH, i.e. 4 device round trips + 4
        64-process allgathers per Σ stage.  This accessor instead hands the
        already-correct host tiles straight to the driver, which sums
        branches in numpy at their global ω indices and pays ONE
        device-assembly + gather at the end of the stage
        (``ppm_sigma.compute_sigma_c_ppm_omega_grid``).  Pure data plumbing:
        the tile VALUES are byte-identical to what result() would have
        gathered; only the movement schedule changes.

        Measured domain (claim-decay rule): at AQ 4962c/P=64 nb=128 (job
        7878038, 2026-07-28) the new rows read sigma.finalize 0.000 s ×4 +
        sigma.host_gather 0.244 s — but the OLD per-branch gathers there
        cost only ~1-4 s total (the baseline's Finished→Started seams were
        already 0-1 s; the analysts' "4-5 s/branch tail" was the async-D2H
        deque drain inside the branch, which this fix deliberately does NOT
        touch).  The win at small nb is small; the lever's value is the
        nb²-growth term AK.9 names.

        Scale/design-envelope note: the object moved is the QP-band-window
        Σ tile — (n_ω, nk, nb_σ/p_x, nb_σ/p_y) per rank — which is
        independent of N_μ and SHRINKS with P, so the fix stays valid as
        n_atoms → hundreds, N_μ → tens of thousands, P → thousands, on CPU
        and GPU backends alike; going from 4 full-slab replications to 1 is
        a flat 4× cut in the stage-tail collective volume at every scale.

        Empty branch (no window contributed): returns zero tiles with the
        indices/devices the sharding prescribes — same semantics as
        result()'s zero path.
        """
        if self._total_shards is None:
            devices = list(self._sharding.addressable_devices)
            dmap = self._sharding.devices_indices_map(self._shape)
            local_shape = self._sharding.shard_shape(self._shape)
            tiles = [np.zeros(local_shape, dtype=np.complex128)
                     for _ in devices]
            index4 = [tuple(dmap[d]) for d in devices]
            return tiles, index4, devices
        # The sink inserts the omega axis at the same explicit position in
        # the global shape and in each sigma(tau) shard index.
        index_out = [tuple(ix[:self._omega_axis]) + (slice(None),)
                     + tuple(ix[self._omega_axis:]) for ix in self._index]
        return self._total_shards, index_out, self._devices

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
