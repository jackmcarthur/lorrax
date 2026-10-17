"""Accumulate time-domain Sigma matrices into the on-shell dynamic Sigma.

For each quadrature node, the spatial kernel produces one ``sigma(tau)``
matrix. :meth:`DeviceOmegaAccumulator.run_window` runs a whole window's nodes
as one executable and folds each ``sigma(tau)`` into what the QSGW ansatz
``½[Σ_mn(E_m) + Σ_mn(E_n)]`` and the eqp writers read
(``gw.ppm_sigma.OnShellRequest``),

    pair[0, k, m, n] = Σ_mn(k, ω_km)        pair[1, k, m, n] = Σ_mn(k, ω_kn)
    slot[j, k, n]    = Σ_nn(k, ω^(j)_kn)    slot[-1, k, n]   = dΣ_nn/dω (k, ω_kn)

with ``ω_kn`` the per-state evaluation energies.  The coefficient of a node
is the scalar ``prefactor·α·exp(-i(E_ref − s·|ω|)t)`` evaluated per state on
the device, so the fold reads each ``σ(τ)`` once and writes a depth-2 matrix;
the derivative slot is the exact ω-derivative of the same quadrature,
``i·s·s_ext·t`` times it.  It serves MPA, one-pole GN/HL-PPM and shared-pole
data and never stores a history of time-domain matrices.

Some one-sided control quadratures first sum their time-node contributions
into a temporary ``Z`` and then add ``(Z - Z†) / (2i)`` to the result. All
matrices retain the result's distribution over the two band axes.
"""

from __future__ import annotations

from functools import lru_cache


import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from common.collectives import device_put_process_local


class WindowTable:
    """Every planned window's coefficient inputs, placed once per Σ call.

    ``fields`` are replicated device arrays with one leading row per window:
    ``t``/``alpha`` padded to the node capacity, ``n`` the active node count,
    ``own`` ``(n_table, n_parent, nb_carrier)`` the states whose energy the
    window's frequencies hold, ``scal`` (prefactor, e_ref_sum, omega_sign,
    s_ext) and caller ``extras`` (``x_<name>``).
    :meth:`DeviceOmegaAccumulator.window_pick` reads one row on device behind
    a device-carried window counter, so a window transfers nothing from the
    host.  ``nodes`` are the host node counts (progress).
    """

    def __init__(self, fields, nodes):
        self.fields = fields
        self.nodes = tuple(int(n) for n in nodes)

    @property
    def nbytes(self) -> int:
        """Bytes of the table on each device (every field is replicated)."""
        return sum(int(v.nbytes) for v in self.fields.values())


@lru_cache(maxsize=None)
def _window_pick_program(replicated):
    """``(table, w) -> (t_pad, pack, n_active, extras, w + 1)`` for row ``w``, on device.

    ``pack`` is the row's ``alpha``, ``own`` and ``scal``: the node
    coefficients are made per state inside the window
    (:func:`_on_shell_node_coefficients`).
    """
    def pick(fields, w):
        row = {k: jax.lax.dynamic_index_in_dim(v, w, 0, keepdims=False)
               for k, v in fields.items()}
        pack = {k: row[k] for k in ("alpha", "own", "scal")}
        extras = {k[2:]: v for k, v in row.items() if k.startswith("x_")}
        return row["t"], pack, row["n"], extras, w + jnp.ones((), w.dtype)
    return jax.jit(pick, out_shardings=replicated)


@lru_cache(maxsize=None)
def _zeros_program(sharding, shapes):
    """The accumulator's zero totals, made on device (no fill-value transfer)."""
    return jax.jit(lambda: tuple(jnp.zeros(one, jnp.complex128) for one in shapes),
                   out_shardings=(sharding,) * len(shapes))


@lru_cache(maxsize=None)
def _window_counter_program(replicated):
    """The first window's counter, made on device (a compiled constant, no transfer)."""
    return jax.jit(lambda: jnp.zeros((), jnp.int32), out_shardings=replicated)

_WINDOW_COMPILED = {}

# The overlapped window (lane LHS 2026-10-01, sandbox runs/DEV/730_lhs_20261001): two τ nodes
# per loop trip, compiled with XLA's latency-hiding scheduler, so the second node's W(τ)
# synthesis and its batch-to-face all_to_all run beside the first node's k-convolutions.
# Ni 20^3 SC map 0 at P64 (band extrapolation on, 657 nodes): window executables 351.7 ->
# 322.3 s (-8.4 %); the compiled window holds a second node's live set, 9.76 -> 16.95 GB per
# rank (stage peak 21.3 -> 28.5 GB); the first window's cold compile 13 -> 34 s.  Without band
# brackets (harness, 8 nodes): 0.549 -> 0.492 s per node, +2.5 GB per rank.
# The scheduler with one node per trip gains nothing (+0.3 %): a node's work waits on its own W.
# Pairing moves Σ(ω) at round-off (5.8e-16 relative); the scheduler on top of it is bitwise.
# Memory first: the caller pairs only when the compiled paired window fits the device budget
# beside the live stages (gw.mpa.sigma.SynthesisTau.fits); otherwise one node per trip and the
# default schedule, the program before this change.
# R82 (remat cloned a counter-indexed read past the counter's in-place increment, which only
# remat does): every counter read in the window's loops sits behind an optimization barrier,
# and rematerialization is off for every program (runtime.disable_xla_rematerialization).
WINDOW_OVERLAP = {"xla_gpu_enable_latency_hiding_scheduler": True}

#: Whether the Σ window pairs two τ nodes per trip, by ``lxkit`` leg key (no row:
#: one node per trip, the plain schedule).  The pairing pays only where one node's
#: W exchange can run beside the other's compute, i.e. with asynchronous collectives.
#: CUDA: on, the LHS lane's measurement above (async collectives on; not re-measured
#: since).  ROCM: off.  XLA async collectives are disabled on that leg
#: (``lxkit.LEGS`` ROCM ``xla_flag_defaults``, the RCCL cross-communicator deadlock),
#: so nothing overlaps: Fe V2 P=4 (pool logs stageprof_ab_on / stageprof_ab_off,
#: 2026-10-16) Σ τ windows 9.60 -> 8.63 s and the compiled window 1.08 -> 0.93 GB
#: with one node per trip, 0.0 ms of RCCL time beside compute, eqp identical.
WINDOW_PAIRING = {"CUDA": True, "ROCM": False}


def window_pairing(mesh) -> bool:
    """The leg's :data:`WINDOW_PAIRING` row for ``mesh``'s devices (False without one)."""
    from lxkit import device_vendor, leg_for_vendor
    leg = leg_for_vendor(device_vendor(mesh.devices.flat[0]))
    return bool(WINDOW_PAIRING.get(None if leg is None else leg.key, False))


def _on_shell_node_coefficients(cpack, t):
    """``(n_slot, n_parent, nb)`` coefficients of one node at every state.

    ``cpack``: ``omega`` (n_table, k, nb), ``own`` (same, bool), ``scal``
    ``(prefactor·α, E_ref, s, s_ext)``.  ``|ω| = s_ext·ω`` on the window's
    frequency half.  Rows: the value on each table, then ``d/dω`` on table 0
    (``ω = s_ext·|ω|``, so ``dc/dω = i·s·s_ext·t·c``).
    """
    pa, e_ref, sign, ext = (cpack["scal"][i] for i in range(4))
    c = pa * jnp.exp(-1j * (e_ref - sign * (ext * cpack["omega"])) * t)
    c = jnp.where(cpack["own"], c, jnp.zeros((), c.dtype))
    return jnp.concatenate([c, (1j * sign * ext * t * c[0])[None]], axis=0)


def _on_shell_fold(acc, parts, c):
    """Fold one node's ``(matrices, count diagonals)`` into the on-shell total.

    ``acc``: ``(matrices, diagonals)``; each matrix ``(2, k, nb, nb)``, each
    diagonal ``(n_slot, k, nb, p_y)`` (``ppm_sigma.band_diagonal_slots``).
    ``parts``: the node's ``(k, nb, nb)`` matrices and ``(k, nb, p_y)``
    diagonals, one per kept count.  ``c``: :func:`_on_shell_node_coefficients`.
    """
    matrices, diagonals = acc
    m_parts, d_parts = parts
    rows, cols = c[0][:, :, None], c[0][:, None, :]
    matrices = tuple(a + jnp.stack([rows * s, cols * s])
                     for a, s in zip(matrices, m_parts))
    diagonals = tuple(a + c[:, :, :, None] * d[None]
                      for a, d in zip(diagonals, d_parts))
    return matrices, diagonals


def _on_shell_antihermitian(total, Z):
    """``(Z - Z†)/(2i)`` for on-shell slots, added to ``total``.

    ``Z†`` at the frequency carried by the ROW index is ``conj(Z[n, m])`` at
    that same frequency, which the COLUMN slot holds: ``Z_L† = conj(Z_Rᵀ)``
    and ``Z_R† = conj(Z_Lᵀ)``.  A diagonal's dagger is its conjugate (also
    for the derivative slot: ω is real).
    """
    def dag(z):
        return jnp.conj(jnp.swapaxes(z, -1, -2))
    (t_m, t_d), (z_m, z_d) = total, Z
    matrices = tuple(
        a + jnp.stack([(z[0] - dag(z[1])) / 2j, (z[1] - dag(z[0])) / 2j])
        for a, z in zip(t_m, z_m))
    diagonals = tuple(a + (z - jnp.conj(z)) / 2j for a, z in zip(t_d, z_d))
    return matrices, diagonals



@lru_cache(maxsize=16)
def _device_window_runner(tau_kernel, sharding, antihermitian, reduce, overlap=False):
    """ONE executable for a whole quadrature window: loop the time nodes on device.

    ``pack`` is the window's on-shell coefficient pack (``alpha`` padded to
    the plan's node capacity, the ``own`` mask, ``scal`` and the resident
    ``omega`` tables), so every window of a plan shares one signature and one
    compile.  The node loop runs ``n_active`` iterations; nothing returns to
    the host between nodes.  An anti-Hermitian window sums its one-sided
    ``Z`` first and adds ``(Z - Z†)/(2i)`` once
    (:func:`_on_shell_antihermitian`).

    ``reduce`` maps each ``sigma(t)`` to ``(matrices, diagonals)``, two tuples
    of arrays, before the fold; ``total`` has that structure
    (:func:`_on_shell_fold`).

    ``overlap`` (:data:`WINDOW_OVERLAP`): the loop runs two nodes per trip,
    folded in node order, and an odd last node after it; the program is
    compiled with the latency-hiding scheduler.
    """
    def pinned(tree):
        return jax.tree.map(
            lambda a: jax.lax.with_sharding_constraint(a, sharding), tree)

    def run(total, tau_arguments, t_nodes, coeff, n_active, active_count):
        def node(i):
            # The node's time and weight are read from the counter once,
            # behind a barrier: remat may otherwise clone a counter-indexed
            # slice after the counter's in-place increment (the R82 hazard).
            t, a = jax.lax.optimization_barrier((t_nodes[i], coeff["alpha"][i]))
            pack = dict(coeff, scal=coeff["scal"].at[0].multiply(a))
            return t, _on_shell_node_coefficients(pack, t)

        def fold(acc, sigma, c):
            return pinned(_on_shell_fold(acc, reduce(sigma), c))

        def one(i, acc):
            t, c = node(i)
            return fold(acc, tau_kernel(*tau_arguments, t, active_count), c)

        def two(j, acc):
            # Nodes 2j and 2j+1, read behind one barrier; neither node's Σ(τ)
            # waits on the other's, so the scheduler runs one's W exchange
            # beside the other's compute.  Folded in node order.
            (t0, c0), (t1, c1) = jax.lax.optimization_barrier(
                (node(2 * j), node(2 * j + 1)))
            s0 = tau_kernel(*tau_arguments, t0, active_count)
            s1 = tau_kernel(*tau_arguments, t1, active_count)
            return fold(fold(acc, s0, c0), s1, c1)

        def nodes(acc):
            if not overlap:
                return jax.lax.fori_loop(0, n_active, one, acc)
            acc = jax.lax.fori_loop(0, n_active // 2, two, acc)
            return jax.lax.cond(n_active % 2 == 1, lambda a: one(n_active - 1, a),
                                lambda a: a, acc)

        if not antihermitian:
            return pinned(nodes(total))
        Z = nodes(pinned(jax.tree.map(jnp.zeros_like, total)))
        return pinned(_on_shell_antihermitian(total, Z))

    return jax.jit(run, donate_argnums=(0,),
                   compiler_options=WINDOW_OVERLAP if overlap else None)


class DeviceOmegaAccumulator:
    """Build the on-shell Sigma without retaining time-domain matrices.

    :meth:`run_window` evaluates one window's quadrature nodes in a device
    loop and folds each ``sigma(t)`` into every state's slots with its
    coefficient ``prefactor * alpha * exp(-i(E_ref_sum - sign*|omega|) t)``.
    A one-sided window first forms ``Z = sum_t coefficient * sigma(t)`` and
    then adds ``(Z-Z†)/(2i)`` on the band indices.  ``alpha`` is the
    quadrature weight before the reference energy is included; combining
    ``E_ref_sum`` and omega in one exponential avoids separately evaluating
    two large factors whose product is well conditioned.
    """

    def __init__(self, request, *, carrier, shape, sharding, reduce):
        """``request`` (``gw.ppm_sigma.OnShellRequest``): its parent-row tables
        and frequency list (the windows' ``omega_indices`` index it), padded
        here to the band ``carrier``.  ``reduce`` returns the kept matrices
        and the diagonal of EVERY count; ``shape`` is the ``(matrices,
        diagonals)`` pair of shape tuples it folds into, with leading extents
        2 and ``request.n_slot``, every member at ``sharding``;
        :meth:`finalize` returns that pair."""
        self._sharding = sharding
        self._reduce = reduce
        self._replicated = NamedSharding(sharding.mesh, P())
        from lxkit import device_vendor
        self._gpu = device_vendor(sharding.mesh.devices.flat[0]) in ("cuda", "rocm")
        self._frequencies = np.asarray(request.frequencies_ry, np.float64)
        pad = int(carrier) - int(request.parent_ry.shape[-1])
        if pad < 0:
            raise ValueError(f"on-shell tables carry {request.parent_ry.shape[-1]} bands, "
                             f"the Sigma carrier {carrier}")
        widths = ((0, 0), (0, 0), (0, pad))
        self._index = np.pad(request.parent_index, widths, constant_values=-1)
        self._omega = device_put_process_local(
            np.pad(request.parent_ry, widths), self._replicated)
        shapes = tuple(tuple(int(n) for n in one) for one in (*shape[0], *shape[1]))
        want = ((2,),) * len(shape[0]) + ((request.n_slot,),) * len(shape[1])
        if tuple(one[:1] for one in shapes) != want:
            raise ValueError(
                "DeviceOmegaAccumulator: on-shell matrices carry 2 slots and "
                f"diagonals {request.n_slot}; got {shapes}")
        # Each rank stores every slot and parent-k point for its assigned
        # block of the two band axes.
        totals = _zeros_program(sharding, shapes)()
        self._total = (totals[:len(shape[0])], totals[len(shape[0]):])

    def integrate_window(self, tau_kernel, tau_arguments, t, alpha, *,
                         active_count, capacity, omega_sign, prefactor,
                         e_ref_sum=0.0, antihermitian=False, omega_indices,
                         omega_values, overlap=False, compile_only=False):
        """Evaluate ``tau_kernel`` at a window's nodes and fold them in.

        The window is a one-row :meth:`window_table` (its frequencies are the
        list positions ``omega_indices`` holding ``omega_values``), read by
        :meth:`window_pick`: the sweep's one coefficient path.  ``compile_only``
        compiles the window executable without running it and returns it
        (for a caller that admits its peak).  ``overlap``: the overlapped
        window (:data:`WINDOW_OVERLAP`) on GPU, the plain one elsewhere.
        """
        table = self.window_table([dict(
            t=jax.device_get(t), alpha=jax.device_get(alpha), omega_indices=omega_indices,
            omega_values=omega_values, omega_sign=omega_sign, prefactor=prefactor,
            e_ref_sum=e_ref_sum)], capacity=capacity)
        t_pad, pack, n_active, _, _ = self.window_pick(table, self.window_counter())
        return self.run_window(tau_kernel, tau_arguments, t_pad, pack, n_active, active_count,
                               antihermitian=antihermitian, overlap=overlap,
                               compile_only=compile_only)

    def window_table(self, windows, *, capacity, extras=None):
        """Place every window's coefficient inputs (and ``extras``) once: a :class:`WindowTable`.

        ``windows`` are dicts with ``t``, ``alpha``, ``omega_indices``,
        ``omega_values``, ``omega_sign``, ``prefactor``, ``e_ref_sum``;
        ``extras`` maps a name to one host array per window (equal shapes).
        A state belongs to a window when its frequency-list position is one
        of the window's ``omega_indices``; the window's frequencies all lie on
        one half, whose sign ``s_ext`` turns the state's signed ``ω`` into the
        window's ``|ω|``.  One process-local placement per field per Σ call.
        """
        n_w, capacity = len(windows), int(capacity)
        host = dict(t=np.zeros((n_w, capacity), np.complex128),
                    alpha=np.zeros((n_w, capacity), np.complex128),
                    n=np.zeros(n_w, np.int32),
                    own=np.zeros((n_w,) + self._index.shape, bool),
                    scal=np.zeros((n_w, 4), np.complex128))
        for i, window in enumerate(windows):
            t = np.asarray(window["t"], np.complex128)
            alpha = np.asarray(window["alpha"], np.complex128)
            columns = np.asarray(window["omega_indices"], dtype=np.int64)
            omega_abs = np.asarray(window["omega_values"], dtype=np.complex128)
            if t.ndim != 1 or alpha.shape != t.shape or not 0 < t.size <= capacity:
                raise ValueError("t and alpha must be nonempty equal vectors within capacity")
            if (columns.ndim != 1 or omega_abs.shape != columns.shape
                    or np.any(columns < 0) or np.any(columns >= self._frequencies.size)
                    or np.unique(columns).size != columns.size):
                raise ValueError("invalid active frequency indices/values")
            signed = self._frequencies[columns]
            ext = -1.0 if np.any(signed < 0.0) else 1.0
            if not np.allclose(ext * signed, np.real(omega_abs), rtol=0.0, atol=1.0e-12):
                raise ValueError("on-shell window: its frequencies straddle omega = 0 or do "
                                 "not match their list positions")
            host["t"][i, :t.size], host["alpha"][i, :t.size], host["n"][i] = t, alpha, t.size
            host["own"][i] = np.isin(self._index, columns)
            host["scal"][i] = (window["prefactor"], window.get("e_ref_sum", 0.0),
                               window["omega_sign"], ext)
        for name, rows in (extras or {}).items():
            host["x_" + name] = np.stack([np.asarray(r) for r in rows])
        fields = {k: device_put_process_local(np.ascontiguousarray(v), self._replicated)
                  for k, v in host.items()}
        return WindowTable(fields, host["n"])

    def window_counter(self):
        """A device window counter at 0 for :meth:`window_pick`."""
        return _window_counter_program(self._replicated)()

    def window_pick(self, table, counter):
        """Row ``counter`` of ``table`` on device: ``(t_pad, pack, n_active, extras, counter + 1)``."""
        return _window_pick_program(self._replicated)(table.fields, counter)

    def run_window(self, tau_kernel, tau_arguments, t_pad, pack, n_active, active_count, *,
                   antihermitian=False, overlap=False, compile_only=False):
        """Run (or, ``compile_only``, compile) one window from its device operands."""
        run = _device_window_runner(
            tau_kernel, self._sharding, bool(antihermitian), self._reduce,
            bool(overlap) and self._gpu)
        arguments = (self._total, tuple(tau_arguments), t_pad, dict(pack, omega=self._omega),
                     n_active, active_count)
        if compile_only:
            # A later SC map asks again for the same runner and signature;
            # the admitted executable is the same, so compile it once.
            key = (run, jax.tree.structure(arguments),
                   tuple((tuple(x.shape), str(x.dtype), x.sharding)
                         for x in jax.tree.leaves(arguments)))
            if key not in _WINDOW_COMPILED:
                _WINDOW_COMPILED[key] = run.lower(*arguments).compile()
            return _WINDOW_COMPILED[key]
        self._total = run(*arguments)
        return self._total

    def finalize(self):
        return self._total
