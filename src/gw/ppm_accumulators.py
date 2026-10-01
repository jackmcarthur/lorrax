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
"""

from __future__ import annotations

from functools import lru_cache


import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from common.collectives import device_put_process_local

def _omega_coefficient(xp, omega, t, alpha, sign, prefactor, e_ref=0.0):
    """Return the scalar multiplying ``sigma(t)`` at each output frequency."""
    return ((prefactor * alpha)
            * xp.exp(-1j * (e_ref - sign * omega) * t))


@lru_cache(maxsize=8)
def _device_output_zeros(shape, sharding):
    return jax.jit(
        lambda: jnp.zeros(shape, dtype=jnp.complex128),
        out_shardings=sharding)


def _omega_fold(acc, sigma, coeff, omega_axis):
    """Add one time-domain Sigma matrix to every represented frequency."""
    coeff_shape = ((1,) * omega_axis + (coeff.shape[0],)
                   + (1,) * (acc.ndim - omega_axis - 1))
    return acc + coeff.reshape(coeff_shape) * jnp.expand_dims(
        sigma, axis=omega_axis)


@lru_cache(maxsize=16)
def _omega_slice(n_omega, omega_axis, sharding):
    return jax.jit(
        lambda a: jax.lax.slice_in_dim(a, 0, n_omega, axis=omega_axis),
        out_shardings=sharding)


_WINDOW_COMPILED = {}


@lru_cache(maxsize=16)
def _device_window_runner(tau_kernel, sharding, omega_axis, antihermitian,
                          reduce=None):
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
    """
    def pinned(tree):
        return jax.tree.map(
            lambda a: jax.lax.with_sharding_constraint(a, sharding), tree)

    def run(total, tau_arguments, t_nodes, coeff, n_active, active_count):
        def one(i, acc):
            sigma = tau_kernel(*tau_arguments, t_nodes[i], active_count)
            if reduce is None:
                return _omega_fold(acc, sigma, coeff[i], omega_axis)
            return jax.tree.map(
                lambda a, part: _omega_fold(a, part, coeff[i], omega_axis),
                acc, reduce(sigma))

        if not antihermitian:
            out = jax.lax.fori_loop(0, n_active, one, total)
            return out if reduce is None else pinned(out)
        if reduce is None:
            Z = jax.lax.with_sharding_constraint(jnp.zeros_like(total), sharding)
            Z = jax.lax.fori_loop(0, n_active, one, Z)
            return total + (Z - jnp.conj(jnp.swapaxes(Z, -1, -2))) / 2j
        matrices, diagonals = jax.lax.fori_loop(
            0, n_active, one, pinned(jax.tree.map(jnp.zeros_like, total)))
        return pinned((
            tuple(a + (z - jnp.conj(jnp.swapaxes(z, -1, -2))) / 2j
                  for a, z in zip(total[0], matrices)),
            tuple(a + (z - jnp.conj(z)) / 2j
                  for a, z in zip(total[1], diagonals))))

    if reduce is None:
        return jax.jit(run, donate_argnums=(0,), out_shardings=sharding)
    return jax.jit(run, donate_argnums=(0,))


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
                 omega_capacity=None):
        """With ``reduce`` (:func:`_device_window_runner`), ``shape`` is the
        ``(matrices, diagonals)`` pair of shape tuples it folds into, every
        member at ``sharding``, and :meth:`finalize` returns that pair.

        ``omega_capacity`` (a held SC run's, never below ``n_omega``) is the
        stored frequency extent: the frequencies past ``n_omega`` get zero
        coefficients, so the window executables keep one signature while a
        ``cover`` grid grows, and :meth:`finalize` returns the first
        ``n_omega``."""
        self._sharding = sharding
        self._reduce = reduce
        self._replicated = NamedSharding(sharding.mesh, P())
        self._omega = np.asarray(jax.device_get(omega_vec), np.complex128)
        self._omega_axis = int(omega_axis)
        shapes = ((shape,) if reduce is None else (*shape[0], *shape[1]))
        shapes = tuple(tuple(int(n) for n in one) for one in shapes)
        for one in shapes:
            if not 0 <= self._omega_axis < len(one):
                raise ValueError(
                    "DeviceOmegaAccumulator: omega_axis outside output rank")
            if one[self._omega_axis] != self._omega.size:
                raise ValueError(
                    "DeviceOmegaAccumulator: shape[omega_axis] must equal "
                    "n_omega")
        # Each rank stores every output frequency and parent-k point for its
        # assigned block of the two band axes.
        self._capacity = max(int(omega_capacity or 0), self._omega.size)
        shapes = tuple(one[:self._omega_axis] + (self._capacity,)
                       + one[self._omega_axis + 1:] for one in shapes)
        totals = tuple(_device_output_zeros(one, sharding)() for one in shapes)
        n_matrix = 1 if reduce is None else len(shape[0])
        self._total = (totals[0] if reduce is None else
                       (totals[:n_matrix], totals[n_matrix:]))

    def integrate_window(self, tau_kernel, tau_arguments, t, alpha, *,
                         n_active, active_count, capacity, omega_sign,
                         prefactor, e_ref_sum=0.0, antihermitian=False,
                         omega_indices=None, omega_values=None,
                         compile_only=False):
        """Evaluate ``tau_kernel`` at a window's first ``n_active`` nodes and fold them in.

        The window's coefficients (``omega_indices`` places ``omega_values``;
        omitted, every output frequency), scattered into the complete
        frequency axis and padded to ``capacity`` nodes, go to the
        device once; the node loop is one executable
        (:func:`_device_window_runner`).  ``compile_only`` lowers and compiles
        that executable without running it and returns the compiled
        executable (for a caller that admits its peak).
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
        coeff = np.zeros((int(capacity), self._capacity), np.complex128)
        coeff[:t.size, columns] = _omega_coefficient(
            np, omega[None, :], t[:, None], alpha[:, None],
            float(omega_sign), float(prefactor), float(e_ref_sum))
        t_pad = np.zeros(int(capacity), np.complex128)
        t_pad[:t.size] = t
        t_pad, coeff, n_active = (
            device_put_process_local(np.asarray(x), self._replicated)
            for x in (t_pad, coeff, np.int32(n_active)))
        run = _device_window_runner(
            tau_kernel, self._sharding, self._omega_axis, bool(antihermitian),
            self._reduce)
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

    def finalize(self):
        if self._capacity == self._omega.size:
            return self._total
        cut = _omega_slice(self._omega.size, self._omega_axis, self._sharding)
        total, self._total = self._total, None
        return jax.tree.map(cut, total)
