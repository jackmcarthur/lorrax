"""Shift pencils of the response rules on the process's device.

``minimax.ResponseGroupTree`` scans six proposal geometries per group; each
geometry's pencil (a stacked Hankel QR, the SVD of its R factor and the
projected shift) is ~1.3 s of host LAPACK at n = 1460 on 7 Frontier threads,
60-70 % of a failing group's attempt (Fe V2, pool log rcgpu_probe1). The pencil
depends on its geometry only, never on a decision, so it runs here on the GCD
the rank owns: :data:`DEVICE_PENCILS` is a source with the contract of
``minimax.host_pencils``. Every decision (node count, sampled error,
coefficient mass, stagnation, group split) stays in minimax on the host.

Arithmetic. QR is the leg's ``jnp.linalg.qr``; the SVD of ``R^H`` is the
Hermitian dilation ``[[0, R^H], [R, 0]]`` solved by
:func:`distrib_la.eigh_local` (eigenvalues +-sigma; the positive half's
eigenvectors are ``[u; w] / sqrt 2``), as ``distrib_la``'s polar factor does.
rocSOLVER's ``zgesvd`` behind ``jnp.linalg.svd`` takes 5.3 s on the n = 1460
pencil and its Jacobi driver returns wrong singular values (s rel err 0.96),
so neither is the route (rcgpu_probe1: QR + dilation 0.34 s for the pencil).
The dilation resolves a small singular value only to ~eps*sigma_max, where
host LAPACK's bidiagonalization keeps it to ~1e-4 relative; a pencil read near
its rounding floor (Fe V2 group [0:22]: sigma_80/sigma_1 ~ 1e-14 at the
accepted count) then moves its nodes at 1e-3. :attr:`DevicePencils.sigma_error`
declares that floor, and minimax decides an accepting geometry whose count it
does not resolve again on ``host_pencils``: such a rule is the host's bytes
(Fe: group [0:22] only; pool log rcgpu_ab8).

Shapes are exact. Zero-padded buckets (zero rows and trailing zero columns of
the stack, mathematically neutral) come back non-finite from the ROCm leg on
954 of the 1020 pencils the Fe V2 tree can build, with or without distinct
dummy singular values in the padded corner; exact shapes: 0 of 1020 (pool
log rcgpu_probe2). The cross-rank compile agreement (INVARIANTS 21) needs
every rank to request the same compiles in the same order while a rank's
attempts are its own, so every rank compiles each round's shapes before the
round (:meth:`DevicePencils.prepare`, through ``compile_ahead``; the caller
``gw.response_bank._gather_group_rules``); the attempts then only execute.
Fe V2: 31 shapes over the whole tree.

Syncs. The next geometry's pencil is dispatched before the current one is
read, so the device builds it while the host scans; each consumed geometry
costs ONE device-to-host read of ``(s, shift)`` (<= 192 + 192^2 complex).
An abandoned prefetch is dropped unread. Device bytes are transient (the
stack, its Q factor and the dilation; the executable's temp), nothing stays
resident.

A non-finite pencil (rocSOLVER ``zheevd`` can fail silently with NaN
vectors) refuses by GATE ``response_pencil_finite``: a scan on it would read
every node count as rejected and change the rule silently.
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _pencil(zps, columns, padding, step, *, size, rank):
    """``(s[:rank], shift[rank, rank])`` of ``minimax.host_pencils`` for one geometry."""
    from distrib_la import eigh_local
    rows = jnp.arange(size, dtype=jnp.float64)[:, None]
    base = -padding + (rows + columns[None, :].astype(jnp.float64))*step     # base[:, columns]
    block = 1/(base[None] - zps[:, None, None])                              # (P, size, C)
    scale = 1/jnp.sqrt(jnp.sum(jnp.real(block*jnp.conj(block)), axis=(1, 2)))
    shifted = jnp.swapaxes(scale[:, None, None]/(base[None] + step - zps[:, None, None]), 0, 1)
    hh = jnp.conj(jnp.swapaxes(scale[:, None, None]*block, 1, 2)).reshape(-1, size)   # H^H
    q, r = jnp.linalg.qr(hh, mode="reduced")
    a = jnp.conj(r.T)                                                         # R^H: size x k
    k = a.shape[1]
    dilation = jnp.zeros((size + k, size + k), a.dtype)
    dilation = dilation.at[:size, size:].set(a).at[size:, :size].set(jnp.conj(a.T))
    w, v = eigh_local(dilation)
    top = math.sqrt(2.)*v[:, ::-1][:, :rank]                                  # descending sigma
    return w[::-1][:rank], jnp.conj(top[:size].T)@(shifted.reshape(size, -1)@q)@top[size:]


_PROGRAMS: dict = {}


def _program(shape):
    """The jitted program of ``(size, poles, columns)`` and its abstract arguments."""
    if shape not in _PROGRAMS:
        from minimax import RESPONSE_RULE_CAPACITY
        size, poles, columns = shape
        rank = min(RESPONSE_RULE_CAPACITY, poles*columns, size)
        S = jax.ShapeDtypeStruct
        _PROGRAMS[shape] = (jax.jit(partial(_pencil, size=size, rank=rank)),
                            (S((poles,), jnp.complex128), S((columns,), jnp.int64),
                             S((), jnp.float64), S((), jnp.float64)))
    return _PROGRAMS[shape]


class DevicePencils:
    """``minimax.host_pencils`` on the default device, on executables compiled ahead."""

    #: Absolute singular-value error of the dilation, in units of sigma_max
    #: (Fe V2 [0:22], geometry 3: sigma_80 = 8.9e-15 sigma_1 off by 17 %, ~7 eps);
    #: minimax re-decides an acceptance whose count this does not resolve.
    sigma_error = 8*float(np.finfo(np.float64).eps)

    def __init__(self):
        self._ready = {}                                   # (size, poles, columns) -> Future[Compiled]

    def prepare(self, shapes):
        """Compile the ``(size, poles, columns)`` shapes not yet compiled, in the given order.

        Every rank must call this with the same shapes at the same point of its
        program: each shape takes one compile request number (INVARIANTS 21).
        Returns the number of compiles requested."""
        from common.jax_compile_cache import compile_ahead
        todo = [s for s in dict.fromkeys(tuple(map(int, s)) for s in shapes) if s not in self._ready]
        for shape in todo:
            fn, avals = _program(shape)
            self._ready[shape] = compile_ahead(fn, *avals)
        return len(todo)

    def executable_bytes(self):
        """Per-shape device bytes (arguments + outputs + temp) of the compiled executables."""
        out = {}
        for shape, future in self._ready.items():
            m = future.result().memory_analysis()
            out[shape] = None if m is None else int(m.argument_size_in_bytes + m.output_size_in_bytes
                                                     + m.temp_size_in_bytes)
        return out

    def __call__(self, zps, size, columns, steps):
        if not jax.config.jax_enable_x64:
            raise ValueError("GATE response_pencil_x64: jax_enable_x64 is off, so the device pencil "
                             "would run in complex64 and every rule decision would move; refusing.")
        shape = (int(size), len(zps), len(columns))
        if shape not in self._ready:
            if jax.process_count() > 1:
                raise ValueError(
                    f"GATE response_pencil_prepared: pencil shape {shape} (size, poles, columns) "
                    f"was not prepared on every rank; compiling it here alone would break the "
                    f"cross-rank compile agreement (INVARIANTS 21). Prepare every node's "
                    f"pencil_shapes first.")
            self.prepare([shape])
        run = self._ready[shape].result()
        # Attempts belong to individual processes after collective prepare.
        # Eager jnp.asarray would compile a conversion only on that owner,
        # while idle peers already wait in the rule-result gather.
        device = jax.local_devices()[0]
        zd = jax.device_put(np.asarray(zps, np.complex128), device)
        cd = jax.device_put(np.asarray(columns, np.int64), device)

        def launch(padding, step):
            return run(zd, cd, jax.device_put(np.float64(padding), device),
                       jax.device_put(np.float64(step), device))
        ahead = launch(*steps[0])
        for j in range(len(steps)):
            now, ahead = ahead, (launch(*steps[j + 1]) if j + 1 < len(steps) else None)
            s, shift = jax.device_get(now)                  # the one read of this geometry
            if not (np.isfinite(s).all() and np.isfinite(shift).all()):
                raise ValueError(
                    f"GATE response_pencil_finite: the device shift pencil of geometry {j} "
                    f"(size {size}, {len(zps)} poles) is not finite (eigh_local of its "
                    f"dilation); refusing rather than scanning it as rejected.")
            yield s, shift


#: The process's device pencil source (its compiled shapes live for the process).
DEVICE_PENCILS = DevicePencils()


def response_pencil_source():
    """:data:`DEVICE_PENCILS` on an accelerator leg, else ``None`` (minimax's host LAPACK
    reference). Read from ``lxkit.LEGS``."""
    import lxkit
    leg = lxkit.leg_for_jax_platform(jax.devices()[0].platform)
    return DEVICE_PENCILS if leg is not None and leg.accelerator else None
