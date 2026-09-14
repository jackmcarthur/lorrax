"""Shared on-device omega accumulator for dynamic Sigma.

``DeviceOmegaAccumulator`` is the sole consumer of transient sigma(tau)
tiles for both MPA and one-pole GN/HL-PPM stores.  It folds directly into the
sharded omega cube and retains no tau history or host-side projection path.
The anti-Hermitian window completion is used only by the shared ``panes``
control planner; production denominator-box windows are fully causal.
"""

from __future__ import annotations

from functools import lru_cache


import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from common.collectives import device_put_process_local

def _omega_coefficient(xp, omega, t, alpha, sign, prefactor, e_ref=0.0):
    """Frequency coefficient shared by the host and device folds."""
    return ((prefactor * alpha)
            * xp.exp(-1j * (e_ref - sign * omega) * t))


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


def _active_omega_fold(acc, sigma, coeff, indices, omega_axis,
                       contiguous=False):
    """Update distinct selected frequency rows in the original node order."""
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


class DeviceOmegaAccumulator:
    """Fold one transient sigma(tau) tile into a sharded omega cube.

    No tau history is retained.  A full/Laplace window accumulates directly
    into the result. A one-sided sine window uses a temporary containing only
    its active frequencies,
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
        # A crossing window additionally holds only its active omega planes.
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
        # Frequency values may be abs/rephased; only storage coordinates
        # decide contiguity. Descending or gapped selections retain order.
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
        """Warm each cheap frequency-fold signature once, without a tau tile.

        Call after ``begin_window`` to warm its exact active width. Returns
        whether a signature was newly warmed by this accumulator instance.
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
        # The caller may block on the updated accumulator when profiling.
        # Production ignores this return and keeps the incumbent async path.
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
