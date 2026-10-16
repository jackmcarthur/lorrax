"""Accumulate time-domain Sigma matrices into dynamic Sigma at real frequencies.

For each quadrature node, the spatial kernel produces one ``sigma(tau)``
matrix. ``DeviceOmegaAccumulator.integrate_window`` runs a whole window's
nodes as one executable: each ``sigma(tau)`` is multiplied by its scalar
coefficient at every output frequency and added to ``sigma(omega)`` on
device. It serves MPA, one-pole GN/HL-PPM and shared-pole data and never
stores a history of time-domain matrices or moves them to a host calculation.

Some one-sided control quadratures first sum their time-node contributions
into a temporary ``Z`` and then add ``(Z - Z†) / (2i)`` to the result. All
matrices retain the result's distribution over the two band axes.

Two evaluations share the one window runner (:func:`_device_window_runner`):

* the GRID evaluation (``sigma_omega_grid_debug``): ``Σ_mn(k, ω)`` at every
  output frequency, ``(n_omega, k, nb, nb)``;
* the ON-SHELL evaluation (the default, :class:`OnShellTables`): only what the
  QSGW ansatz ``½[Σ_mn(E_m) + Σ_mn(E_n)]`` and the eqp writers read,

      pair[0, k, m, n] = Σ_mn(k, ω_km)        pair[1, k, m, n] = Σ_mn(k, ω_kn)
      slot[j, k, n]    = Σ_nn(k, ω^(j)_kn)    slot[-1, k, n]   = dΣ_nn/dω (k, ω_kn)

  with ``ω_kn`` the per-state evaluation energies.  The coefficient of a node
  is the same scalar ``prefactor·α·exp(-i(E_ref − s·|ω|)t)`` evaluated per
  state on the device, so the fold reads each ``σ(τ)`` once and writes a
  depth-2 matrix instead of a depth-``n_omega`` one; the derivative slot is
  the exact ω-derivative of the same quadrature, ``i·s·s_ext·t`` times it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple


import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from common.collectives import device_put_process_local

def _omega_coefficient(xp, omega, t, alpha, sign, prefactor, e_ref=0.0):
    """Return the scalar multiplying ``sigma(t)`` at each output frequency."""
    return ((prefactor * alpha)
            * xp.exp(-1j * (e_ref - sign * omega) * t))


class OnShellTables(NamedTuple):
    """The per-state evaluation energies of an on-shell Sigma, on the parent k rows.

    ``omega`` ``(n_table, n_parent, nb_carrier)`` float64, Ry, measured from
    the run's omega reference (the same frame as the planner's frequency
    list).  Table 0 is the ``pair`` table: both matrix slots and the
    derivative slot read it.  ``index`` (same shape, int32) is each entry's
    position in the planner's frequency list, ``-1`` on a padded carrier
    band; a window owns an entry when it owns that position.  Both are
    replicated on the devices.
    """
    omega: jax.Array
    index: np.ndarray
    omega_host: np.ndarray

    @property
    def n_slot(self) -> int:
        """Diagonal slots: one value per table, then the derivative on table 0."""
        return int(self.omega.shape[0]) + 1


def _omega_fold(acc, sigma, coeff, omega_axis):
    """Add one time-domain Sigma matrix to every represented frequency."""
    coeff_shape = ((1,) * omega_axis + (coeff.shape[0],)
                   + (1,) * (acc.ndim - omega_axis - 1))
    return acc + coeff.reshape(coeff_shape) * jnp.expand_dims(
        sigma, axis=omega_axis)


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
def _device_window_runner(tau_kernel, sharding, omega_axis, antihermitian,
                          reduce=None, overlap=False, on_shell=False):
    """ONE executable for a whole quadrature window: loop the time nodes on device.

    ``coeff`` is ``(capacity, n_omega)`` over the COMPLETE output frequency
    axis, exactly zero outside the window's frequencies, so every window of a
    plan shares one signature (``capacity`` is the plan's largest node count)
    and one compile.  The node loop runs ``n_active`` iterations; nothing
    returns to the host between nodes.  An anti-Hermitian window sums its
    one-sided ``Z`` first and adds ``(Z - Z†)/(2i)`` once.

    ``reduce`` maps each ``sigma(t)`` to ``(matrices, diagonals)``, two tuples
    of arrays, before the fold; ``total`` has that structure, so only what is
    folded is stored at every frequency.  A diagonal's dagger is its
    conjugate.

    ``overlap`` (:data:`WINDOW_OVERLAP`): the loop runs two nodes per trip,
    folded in node order, and an odd last node after it; the program is
    compiled with the latency-hiding scheduler.

    ``on_shell``: ``coeff`` is the on-shell pack (``alpha`` padded to
    ``capacity``, ``omega``/``own`` tables, ``scal``) and the node's
    coefficients are made on the device per state
    (:func:`_on_shell_node_coefficients`); ``reduce`` is required and ``total``
    is ``(matrices, diagonals)`` (:func:`_on_shell_fold`).
    """
    if on_shell and reduce is None:
        raise ValueError("the on-shell window folds a reduced sigma(t); reduce is required")
    def pinned(tree):
        return jax.tree.map(
            lambda a: jax.lax.with_sharding_constraint(a, sharding), tree)

    def run(total, tau_arguments, t_nodes, coeff, n_active, active_count):
        def node(i):
            # The node's time and coefficients are read from the counter once,
            # behind a barrier: remat may otherwise clone a counter-indexed
            # slice after the counter's in-place increment (the R82 hazard).
            if not on_shell:
                return jax.lax.optimization_barrier((t_nodes[i], coeff[i]))
            t, a = jax.lax.optimization_barrier((t_nodes[i], coeff["alpha"][i]))
            pack = dict(coeff, scal=coeff["scal"].at[0].multiply(a))
            return t, _on_shell_node_coefficients(pack, t)

        def fold(acc, sigma, c):
            if on_shell:
                return pinned(_on_shell_fold(acc, reduce(sigma), c))
            if reduce is None:
                return _omega_fold(acc, sigma, c, omega_axis)
            return jax.tree.map(
                lambda a, part: _omega_fold(a, part, c, omega_axis),
                acc, reduce(sigma))

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
            out = nodes(total)
            return out if reduce is None else pinned(out)
        if on_shell:
            Z = nodes(pinned(jax.tree.map(jnp.zeros_like, total)))
            return pinned(_on_shell_antihermitian(total, Z))
        if reduce is None:
            Z = jax.lax.with_sharding_constraint(jnp.zeros_like(total), sharding)
            Z = nodes(Z)
            return total + (Z - jnp.conj(jnp.swapaxes(Z, -1, -2))) / 2j
        matrices, diagonals = nodes(pinned(jax.tree.map(jnp.zeros_like, total)))
        return pinned((
            tuple(a + (z - jnp.conj(jnp.swapaxes(z, -1, -2))) / 2j
                  for a, z in zip(total[0], matrices)),
            tuple(a + (z - jnp.conj(z)) / 2j
                  for a, z in zip(total[1], diagonals))))

    options = WINDOW_OVERLAP if overlap else None
    if reduce is None:
        return jax.jit(run, donate_argnums=(0,), out_shardings=sharding,
                       compiler_options=options)
    return jax.jit(run, donate_argnums=(0,), compiler_options=options)


class DeviceOmegaAccumulator:
    """Build real-frequency Sigma without retaining time-domain matrices.

    :meth:`integrate_window` evaluates one window's quadrature nodes in a
    device loop and folds each ``sigma(t)`` into the output frequencies with
    its coefficient ``prefactor * alpha * exp(-i(E_ref_sum - sign*omega) t)``.
    A one-sided window first forms ``Z(omega) = sum_t coefficient * sigma(t)``
    and then adds ``(Z-Z†)/(2i)`` on the band indices.  ``alpha`` is the
    quadrature weight before the reference energy is included; combining
    ``E_ref_sum`` and omega in one exponential avoids separately evaluating
    two large factors whose product is well conditioned.
    """

    def __init__(self, omega_vec, *, shape, sharding, omega_axis, reduce=None,
                 on_shell: OnShellTables | None = None):
        """With ``reduce`` (:func:`_device_window_runner`), ``shape`` is the
        ``(matrices, diagonals)`` pair of shape tuples it folds into, every
        member at ``sharding``, and :meth:`finalize` returns that pair.

        ``on_shell`` (:class:`OnShellTables`): the on-shell evaluation.
        ``omega_vec`` is then the planner's frequency list (the windows'
        ``omega_indices`` index it), ``reduce`` returns the kept matrices and
        the diagonal of EVERY count, and ``shape`` is the
        ``(matrices, diagonals)`` pair with leading extents 2 and
        ``on_shell.n_slot``."""
        self._sharding = sharding
        self._reduce = reduce
        self._on_shell = on_shell
        self._replicated = NamedSharding(sharding.mesh, P())
        from lxkit import device_vendor
        self._gpu = device_vendor(sharding.mesh.devices.flat[0]) in ("cuda", "rocm")
        self._omega = np.asarray(jax.device_get(omega_vec), np.complex128)
        self._omega_axis = int(omega_axis)
        shapes = ((shape,) if reduce is None else (*shape[0], *shape[1]))
        shapes = tuple(tuple(int(n) for n in one) for one in shapes)
        if on_shell is not None:
            if reduce is None:
                raise ValueError("DeviceOmegaAccumulator: on-shell needs a reduce")
            want = ((2,),) * len(shape[0]) + ((on_shell.n_slot,),) * len(shape[1])
            if tuple(one[:1] for one in shapes) != want:
                raise ValueError(
                    "DeviceOmegaAccumulator: on-shell matrices carry 2 slots and "
                    f"diagonals {on_shell.n_slot}; got {shapes}")
            shapes_checked = ()
        else:
            shapes_checked = shapes
        for one in shapes_checked:
            if not 0 <= self._omega_axis < len(one):
                raise ValueError(
                    "DeviceOmegaAccumulator: omega_axis outside output rank")
            if one[self._omega_axis] != self._omega.size:
                raise ValueError(
                    "DeviceOmegaAccumulator: shape[omega_axis] must equal "
                    "n_omega")
        # Each rank stores every output frequency and parent-k point for its
        # assigned block of the two band axes.
        totals = tuple(jnp.zeros(one, jnp.complex128, device=sharding) for one in shapes)
        n_matrix = 1 if reduce is None else len(shape[0])
        self._total = (totals[0] if reduce is None else
                       (totals[:n_matrix], totals[n_matrix:]))

    def integrate_window(self, tau_kernel, tau_arguments, t, alpha, *,
                         n_active, active_count, capacity, omega_sign,
                         prefactor, e_ref_sum=0.0, antihermitian=False,
                         omega_indices=None, omega_values=None,
                         overlap=False, compile_only=False):
        """Evaluate ``tau_kernel`` at a window's first ``n_active`` nodes and fold them in.

        The window's coefficients (``omega_indices`` places ``omega_values``;
        omitted, every output frequency), scattered into the complete
        frequency axis and padded to ``capacity`` nodes, go to the
        device once; the node loop is one executable
        (:func:`_device_window_runner`).  ``compile_only`` lowers and compiles
        that executable without running it and returns the compiled
        executable (for a caller that admits its peak).  ``overlap``: the
        overlapped window (:data:`WINDOW_OVERLAP`) on GPU, the plain one elsewhere.
        """
        t = np.asarray(jax.device_get(t), np.complex128)
        alpha = np.asarray(jax.device_get(alpha), np.complex128)
        if t.ndim != 1 or alpha.shape != t.shape or t.size == 0:
            raise ValueError("t and alpha must be nonempty equal vectors")
        if not 0 < int(n_active) <= t.size <= int(capacity):
            raise ValueError("require 0 < n_active <= len(t) <= capacity")
        if omega_indices is None:
            if omega_values is not None:
                raise ValueError("omega_values requires omega_indices")
            columns = np.arange(self._omega.size)
            omega = self._omega
        else:
            columns = np.asarray(omega_indices, dtype=np.int64)
            omega = np.asarray(omega_values, dtype=np.complex128)
            if (columns.ndim != 1 or omega.shape != columns.shape
                    or np.any(columns < 0) or np.any(columns >= self._omega.size)
                    or np.unique(columns).size != columns.size):
                raise ValueError("invalid active frequency indices/values")
        t_pad = np.zeros(int(capacity), np.complex128)
        t_pad[:t.size] = t
        if self._on_shell is None:
            coeff = np.zeros((int(capacity), self._omega.size), np.complex128)
            coeff[:t.size, columns] = _omega_coefficient(
                np, omega[None, :], t[:, None], alpha[:, None],
                float(omega_sign), float(prefactor), float(e_ref_sum))
            coeff = device_put_process_local(coeff, self._replicated)
        else:
            coeff = self._on_shell_pack(alpha, columns, omega, capacity=capacity,
                                        omega_sign=omega_sign, prefactor=prefactor,
                                        e_ref_sum=e_ref_sum)
        t_pad, n_active = (
            device_put_process_local(np.asarray(x), self._replicated)
            for x in (t_pad, np.int32(n_active)))
        run = _device_window_runner(
            tau_kernel, self._sharding, self._omega_axis, bool(antihermitian),
            self._reduce, bool(overlap) and self._gpu, self._on_shell is not None)
        arguments = (self._total, tuple(tau_arguments), t_pad, coeff,
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

    def _on_shell_pack(self, alpha, columns, omega_abs, *, capacity, omega_sign,
                       prefactor, e_ref_sum):
        """The window's on-shell coefficient pack (:func:`_device_window_runner`).

        A state belongs to the window when its frequency-list position is one
        of the window's ``columns``; the window's frequencies all lie on one
        half, whose sign ``s_ext`` turns the state's signed ``ω`` into the
        window's ``|ω|``.  Only the node weights, the ownership mask and four
        scalars travel per window: the energy tables stay on the devices.
        """
        tables = self._on_shell
        signed = np.real(self._omega[columns])
        ext = -1.0 if np.any(signed < 0.0) else 1.0
        if not np.allclose(ext * signed, np.real(omega_abs), rtol=0.0, atol=1.0e-12):
            raise ValueError("on-shell window: its frequencies straddle omega = 0 or do not "
                             "match their list positions")
        own = np.isin(tables.index, columns)
        alpha_pad = np.zeros(int(capacity), np.complex128)
        alpha_pad[:alpha.size] = alpha
        scal = np.asarray([prefactor, e_ref_sum, omega_sign, ext], np.complex128)
        alpha_pad, own, scal = (device_put_process_local(np.asarray(x), self._replicated)
                                for x in (alpha_pad, own, scal))
        return dict(alpha=alpha_pad, own=own, omega=tables.omega, scal=scal)

    def finalize(self):
        return self._total
