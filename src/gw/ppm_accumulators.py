"""Accumulate time-domain Sigma matrices into dynamic Sigma at real frequencies.

For each quadrature node, the spatial kernel produces one ``sigma(tau)``
matrix. ``DeviceOmegaAccumulator`` multiplies that matrix by the scalar
coefficient for each requested real frequency and immediately adds the result
to ``sigma(omega)``. It serves both MPA and one-pole GN/HL-PPM data and never
stores a history of time-domain matrices or moves them to a host calculation.

Some one-sided control quadratures first sum their time-node contributions into
a temporary ``Z`` and then add ``(Z - Z†) / (2i)`` to the result. That
temporary contains only the frequencies selected by the control window. All
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
def _antiherm_band_fn(sharding: NamedSharding):
    """Convert a one-sided sum ``Z`` to ``(Z-Z†)/(2i)`` on its band axes.

    The two trailing axes are the outgoing and incoming band indices. The
    result keeps their requested distribution across ranks. Reusing this JAX
    function for the same distribution avoids rebuilding it for every planned
    frequency window.
    """
    return jax.jit(
        lambda Z: (Z - jnp.conj(jnp.swapaxes(Z, -1, -2))) / 2j,
        out_shardings=sharding,
    )


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


def _active_omega_fold(acc, sigma, coeff, indices, omega_axis,
                       contiguous=False):
    """Add one Sigma matrix at frequencies named by distinct output indices."""
    if contiguous:
        selected = jax.lax.dynamic_slice_in_dim(
            acc, indices[0], coeff.shape[0], axis=omega_axis)
        updated = _omega_fold(selected, sigma, coeff, omega_axis)
        return jax.lax.dynamic_update_slice_in_dim(
            acc, updated, indices[0], axis=omega_axis)
    selected = jnp.take(acc, indices, axis=omega_axis)
    updated = _omega_fold(selected, sigma, coeff, omega_axis)
    where = (slice(None),) * omega_axis + (indices,)
    return acc.at[where].set(updated, unique_indices=True)


@lru_cache(maxsize=16)
def _device_active_omega_add(sharding, omega_axis, contiguous=False):
    return jax.jit(
        lambda acc, sigma, coeff, indices: _active_omega_fold(
            acc, sigma, coeff, indices, omega_axis, contiguous),
        donate_argnums=(0,), out_shardings=sharding)


@lru_cache(maxsize=16)
def _device_active_window_add(sharding, omega_axis, contiguous=False):
    def add(total, window, indices):
        if contiguous:
            selected = jax.lax.dynamic_slice_in_dim(
                total, indices[0], window.shape[omega_axis], axis=omega_axis)
            return jax.lax.dynamic_update_slice_in_dim(
                total, selected + window, indices[0], axis=omega_axis)
        where = (slice(None),) * omega_axis + (indices,)
        return total.at[where].add(window, unique_indices=True)
    return jax.jit(add, donate_argnums=(0,), out_shardings=sharding)


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


_WINDOW_COMPILED = {}


@lru_cache(maxsize=16)
def _device_window_runner(tau_kernel, sharding, omega_axis, antihermitian):
    """ONE executable for a whole quadrature window: loop the time nodes on device.

    ``coeff`` is ``(capacity, n_omega)`` over the COMPLETE output frequency
    axis, exactly zero outside the window's frequencies, so every window of a
    plan shares one signature (``capacity`` is the plan's largest node count)
    and one compile.  The node loop runs ``n_active`` iterations; nothing
    returns to the host between nodes.  An anti-Hermitian window sums its
    one-sided ``Z`` first and adds ``(Z - Z†)/(2i)`` once, as ``end_window``
    does.
    """
    def run(total, tau_arguments, t_nodes, coeff, n_active, active_count):
        def one(i, acc):
            sigma = tau_kernel(*tau_arguments, t_nodes[i], active_count)
            return _omega_fold(acc, sigma, coeff[i], omega_axis)

        if not antihermitian:
            return jax.lax.fori_loop(0, n_active, one, total)
        Z = jax.lax.with_sharding_constraint(jnp.zeros_like(total), sharding)
        Z = jax.lax.fori_loop(0, n_active, one, Z)
        return total + (Z - jnp.conj(jnp.swapaxes(Z, -1, -2))) / 2j

    return jax.jit(run, donate_argnums=(0,), out_shardings=sharding)


class DeviceOmegaAccumulator:
    """Build real-frequency Sigma without retaining time-domain matrices.

    ``begin_window`` records a sequence of quadrature nodes and the output
    frequencies they contribute to. Each call to ``add_tau`` consumes the
    spatial Sigma matrix for the next node and updates those output frequencies
    in the declared order.

    Most windows update the final result immediately. A one-sided window first
    forms ``Z(omega) = sum_t coefficient(omega, t) * sigma(t)`` for only its
    selected frequencies. ``end_window`` then computes
    ``(Z-Z†)/(2i)`` on the band indices and adds it to the final result.
    ``alpha`` is the quadrature weight before the reference energy is included.
    Combining ``E_ref_sum`` and omega in one exponential avoids separately
    evaluating two large factors whose product is well conditioned.
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
        # assigned block of the two band axes. A one-sided window additionally
        # stores Z for only the output frequencies that window selects.
        self._total = _device_output_zeros(self._shape, sharding)()
        self._window = None
        self._coeff = None
        self._index = 0
        self._indices = None
        self._indices_device = None
        self._contiguous = False
        self._compiled_adds = set()

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
            if np.unique(indices).size != indices.size:
                raise ValueError("active frequency indices must be distinct")
            if np.array_equal(indices, np.arange(self._omega.size)):
                indices = None
        active = np.asarray(_omega_coefficient(
            np, omega[None, :], t[:, None], alpha[:, None],
            float(omega_sign), float(prefactor), float(e_ref_sum)),
            np.complex128)
        self._coeff = active
        self._indices = indices
        # omega_values enters the coefficient. omega_indices says where the
        # resulting contributions belong in the output. A descending or gapped
        # index list is read and written by explicit index in the given order.
        omega_partition = (
            self._sharding.spec[self._omega_axis]
            if self._omega_axis < len(self._sharding.spec) else None)
        self._contiguous = bool(
            indices is not None and indices.size > 0
            and omega_partition is None
            and np.all(np.diff(indices) == 1))
        self._indices_device = device_put_process_local(
            np.asarray([] if indices is None else indices, np.int32),
            self._replicated)
        self._index = 0
        window_shape = list(self._shape)
        window_shape[self._omega_axis] = omega.size
        self._window = (_device_output_zeros(
            tuple(window_shape), self._sharding)() if antihermitian else None)

    def precompile_tau_add(self, *, sigma_shape, sigma_sharding):
        """Compile the next time-node update without evaluating a Sigma matrix.

        JAX reuses compiled code when the operation, array shapes, data types,
        and distribution across ranks match. Call this after ``begin_window``;
        it compiles the update for that window's number of selected frequencies
        once per accumulator. The return value is true only on this
        accumulator's first request for that combination; JAX may satisfy the
        request from an executable already held in a process or persistent
        cache.
        """
        sigma = jax.ShapeDtypeStruct(
            tuple(int(n) for n in sigma_shape), jnp.complex128,
            sharding=sigma_sharding)
        n_omega = (self._omega.size if self._coeff is None
                   else self._coeff.shape[1])
        if n_omega == 0:
            return False
        coeff = jax.ShapeDtypeStruct(
            (n_omega,), jnp.complex128,
            sharding=self._replicated)
        if self._window is None and self._indices is not None:
            run = _device_active_omega_add(
                self._sharding, self._omega_axis, self._contiguous)
            arguments = (self._total, sigma, coeff, self._indices_device)
        else:
            carry = self._total if self._window is None else self._window
            run = _device_omega_add(self._sharding, self._omega_axis)
            arguments = (carry, sigma, coeff)
        signature = (run, tuple(
            (tuple(x.shape), str(x.dtype), x.sharding) for x in arguments))
        if signature in self._compiled_adds:
            return False
        run.lower(*arguments).compile()
        self._compiled_adds.add(signature)
        return True


    def add_tau(self, sigma_tau):
        if self._coeff is None:
            raise RuntimeError("no open frequency window")
        if self._index >= self._coeff.shape[0]:
            raise RuntimeError("more sigma(tau) tiles than quadrature nodes")
        if self._coeff.shape[1] == 0:
            self._index += 1
            return self._total
        coeff = device_put_process_local(
            self._coeff[self._index], self._replicated)
        self._index += 1
        if self._window is None:
            if self._indices is None:
                self._total = _device_omega_add(
                    self._sharding, self._omega_axis)(
                    self._total, sigma_tau, coeff)
            else:
                self._total = _device_active_omega_add(
                    self._sharding, self._omega_axis, self._contiguous)(
                    self._total, sigma_tau, coeff, self._indices_device)
        else:
            self._window = _device_omega_add(
                self._sharding, self._omega_axis)(
                self._window, sigma_tau, coeff)
        # Returning the updated array lets timing code wait for the addition as
        # well as the spatial Sigma calculation. Normal execution waits only at
        # progress milestones and otherwise leaves JAX work asynchronous.
        return self._total if self._window is None else self._window


    def end_window(self):
        if self._coeff is None:
            raise RuntimeError("no open frequency window")
        if self._index != self._coeff.shape[0]:
            raise RuntimeError("frequency window ended before all tau nodes")
        if self._window is not None and self._coeff.shape[1] != 0:
            completed = _antiherm_band_fn(self._sharding)(self._window)
            if self._indices is None:
                self._total = _device_output_add(self._sharding)(
                    self._total, completed)
            else:
                self._total = _device_active_window_add(
                    self._sharding, self._omega_axis, self._contiguous)(
                    self._total, completed, self._indices_device)
        self._window = None
        self._coeff = None
        self._index = 0
        self._indices = self._indices_device = None
        self._contiguous = False

    def integrate_window(self, tau_kernel, tau_arguments, t, alpha, *,
                         n_active, active_count, capacity, omega_sign,
                         prefactor, e_ref_sum=0.0, antihermitian=False,
                         omega_indices=None, omega_values=None,
                         compile_only=False):
        """Evaluate ``tau_kernel`` at a window's first ``n_active`` nodes and fold them in.

        The same coefficients as :meth:`begin_window`, scattered into the
        complete frequency axis and padded to ``capacity`` nodes, go to the
        device once; the node loop is one executable
        (:func:`_device_window_runner`).  ``compile_only`` lowers and compiles
        that executable without running it and returns the compiled
        executable (for a caller that admits its peak).
        """
        if self._coeff is not None:
            raise RuntimeError("a per-node frequency window is still open")
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
        coeff = np.zeros((int(capacity), self._omega.size), np.complex128)
        coeff[:t.size, columns] = _omega_coefficient(
            np, omega[None, :], t[:, None], alpha[:, None],
            float(omega_sign), float(prefactor), float(e_ref_sum))
        t_pad = np.zeros(int(capacity), np.complex128)
        t_pad[:t.size] = t
        t_pad, coeff, n_active = (
            device_put_process_local(np.asarray(x), self._replicated)
            for x in (t_pad, coeff, np.int32(n_active)))
        run = _device_window_runner(
            tau_kernel, self._sharding, self._omega_axis, bool(antihermitian))
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

    def add_direct(self, sigma_omega, omega_index, *, coefficient=1.0):
        """Add a Sigma matrix already evaluated at one output frequency.

        A direct reciprocal-space term already includes its complete
        denominator, so it needs no time-node coefficient. The update retains
        the same output distribution and numerical addition order as the
        quadrature contributions.
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
