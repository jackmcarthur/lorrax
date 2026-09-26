"""Accumulate time-domain Sigma matrices into dynamic Sigma at real frequencies.

For each quadrature node, the spatial kernel produces one ``sigma(tau)``
matrix. ``DeviceOmegaAccumulator.integrate_window`` runs a window's nodes in
blocks of ``_TAU_BLOCK`` inside one executable: each ``sigma(tau)`` is
multiplied by its scalar coefficient at every output frequency and added to
``sigma(omega)`` on device. It serves MPA, one-pole GN/HL-PPM and shared-pole
data and never stores a history of time-domain matrices or moves them to a
host calculation.

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


_WINDOW_COMPILED = {}

# Nodes per dispatch of the window executable.  A static trip count is a loop
# the GPU runtime runs without copying a predicate to the host and
# synchronizing (a dynamic count cost two such round trips per node with the
# G partner's cond: 2.05 s of a 2.9 s Fe 4^3 sweep).  A window runs
# ceil(n/B) blocks; the tail's coefficients are zero, so at most B-1 masked
# nodes per window are evaluated and add nothing.
_TAU_BLOCK = 4


@lru_cache(maxsize=16)
def _device_window_runner(tau_kernel, sharding, omega_axis, antihermitian):
    """ONE executable for ``_TAU_BLOCK`` consecutive nodes of a quadrature window.

    ``coeff`` is ``(capacity, n_omega)`` over the COMPLETE output frequency
    axis, exactly zero outside the window's frequencies and past its last
    node, so every block of every window of a plan shares one signature
    (``capacity``, the plan's largest node count rounded up to the block)
    and one compile.  ``start`` (replicated device scalar) is the block's
    first node; the loop's bound is static, so nothing returns to the host
    between nodes and the host dispatches the blocks ahead of the device.
    Nodes are folded one at a time in window order.  An anti-Hermitian
    window carries ``(total, Z)``: its blocks sum the one-sided ``Z`` and
    :func:`_antihermitian_completion` adds ``(Z - Z†)/(2i)`` once.
    """
    def run(state, tau_arguments, t_nodes, coeff, start, active_count):
        def one(j, acc):
            i = start + j
            sigma = tau_kernel(*tau_arguments, t_nodes[i], active_count)
            return _omega_fold(acc, sigma, coeff[i], omega_axis)

        if not antihermitian:
            return jax.lax.fori_loop(0, _TAU_BLOCK, one, state, unroll=1)
        total, Z = state
        return total, jax.lax.fori_loop(0, _TAU_BLOCK, one, Z, unroll=1)

    return jax.jit(run, donate_argnums=(0,),
                   out_shardings=(sharding, sharding) if antihermitian else sharding)


@lru_cache(maxsize=8)
def _antihermitian_completion(sharding):
    """``total + (Z - Z†)/(2i)`` on the two band axes, once per one-sided window."""
    return jax.jit(
        lambda total, Z: total + (Z - jnp.conj(jnp.swapaxes(Z, -1, -2))) / 2j,
        donate_argnums=(0, 1), out_shardings=sharding)


@lru_cache(maxsize=4096)
def _block_start(start, sharding):
    """The replicated device scalar naming a block's first node (reused across windows)."""
    return device_put_process_local(np.int32(start), sharding)


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
        # Each rank stores every output frequency and parent-k point for its
        # assigned block of the two band axes.
        self._total = _device_output_zeros(self._shape, sharding)()

    def integrate_window(self, tau_kernel, tau_arguments, t, alpha, *,
                         n_active, active_count, capacity, omega_sign,
                         prefactor, e_ref_sum=0.0, antihermitian=False,
                         omega_indices=None, omega_values=None,
                         compile_only=False):
        """Evaluate ``tau_kernel`` at a window's first ``n_active`` nodes and fold them in.

        The window's coefficients (``omega_indices`` places ``omega_values``;
        omitted, every output frequency), scattered into the complete
        frequency axis and padded with zeros to ``capacity`` rounded up to
        ``_TAU_BLOCK`` nodes, go to the device once; the host then dispatches
        ``ceil(n_active/_TAU_BLOCK)`` blocks of one executable
        (:func:`_device_window_runner`) without waiting on any of them.
        ``compile_only`` lowers and compiles that executable without running
        it and returns the compiled executable (for a caller that admits its
        peak).
        """
        t = np.asarray(jax.device_get(t), np.complex128)
        alpha = np.asarray(jax.device_get(alpha), np.complex128)
        if t.ndim != 1 or alpha.shape != t.shape or t.size == 0:
            raise ValueError("t and alpha must be nonempty equal vectors")
        if not 0 < int(n_active) <= t.size <= int(capacity):
            raise ValueError("require 0 < n_active <= len(t) <= capacity")
        n_active = int(n_active)
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
        rows = -(-int(capacity) // _TAU_BLOCK) * _TAU_BLOCK
        coeff = np.zeros((rows, self._omega.size), np.complex128)
        coeff[:n_active, columns] = _omega_coefficient(
            np, omega[None, :], t[:n_active, None], alpha[:n_active, None],
            float(omega_sign), float(prefactor), float(e_ref_sum))
        # A masked node repeats the window's last node: a time the kernel
        # evaluates finitely, times a zero coefficient.
        t_pad = np.full(rows, t[n_active - 1], np.complex128)
        t_pad[:t.size] = t
        t_pad, coeff = (device_put_process_local(np.asarray(x), self._replicated)
                        for x in (t_pad, coeff))
        run = _device_window_runner(
            tau_kernel, self._sharding, self._omega_axis, bool(antihermitian))
        tau_arguments = tuple(tau_arguments)
        if compile_only:
            # A later SC map asks again for the same runner and signature;
            # the admitted executable is the same, so compile it once.
            state = ((self._total, jax.ShapeDtypeStruct(
                self._shape, jnp.complex128, sharding=self._sharding))
                     if antihermitian else self._total)
            arguments = (state, tau_arguments, t_pad, coeff,
                         _block_start(0, self._replicated), active_count)
            key = (run, jax.tree.structure(arguments),
                   tuple((tuple(x.shape), str(x.dtype), x.sharding)
                         for x in jax.tree.leaves(arguments)))
            if key not in _WINDOW_COMPILED:
                _WINDOW_COMPILED[key] = run.lower(*arguments).compile()
            return _WINDOW_COMPILED[key]
        state = ((self._total, _device_output_zeros(self._shape, self._sharding)())
                 if antihermitian else self._total)
        for start in range(0, n_active, _TAU_BLOCK):
            state = run(state, tau_arguments, t_pad, coeff,
                        _block_start(start, self._replicated), active_count)
        self._total = (_antihermitian_completion(self._sharding)(*state)
                       if antihermitian else state)
        return self._total

    def finalize(self):
        return self._total
