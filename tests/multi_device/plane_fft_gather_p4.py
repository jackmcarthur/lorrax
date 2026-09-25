"""P=4 gate: the route-G plane FFT door on CUDA (nvidia-mathdx mode 10).

``ffi.fft.make_plane_fft_gather`` against ``fftn(take(F, plane_from_col,
mode='fill'))`` evaluated by XLA (cuFFT) on the same GPU, over the QE-style
plane sides 24..250 (square), rectangular planes, centred-disk supports at
20% and 45% of the cells, random 30% supports, an empty support, and batch
tails (1, 7, 37 planes; a rank-3 leading shape).  Every rank runs every case
on its own GPU (the runtime's cross-rank compile agreement wants identical
modules).

* A plane whose axes split into cuFFTDx thread FFTs and whose resident plane
  plus the kernel's static tables (``plane_resident_bytes``) fits the device's
  opt-in shared memory must take route ``mathdx``; any other (64, 125, 128,
  250; large planes; the ceiling-edge planes 75x138, 57x182, 112x92 on A100)
  route ``xla``, both asserted.  105x99 sits just inside the A100 edge.
* Parity: ``max|Y - ref| <= 1e-13 max|ref|`` (not bitwise vs cuFFT).
* Red twin: the door built on a support rolled by one cell misses by > 1e-3.
* The slab form ``fn(F, start, size)`` (traced start, including a clamped
  one) equals the reference on ``lax.dynamic_slice_in_dim(F, start, size, 1)``.

Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/plane_fft_gather_p4.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from ffi.fft import (_optin_smem_bytes, kconv_backend, make_plane_fft_gather,  # noqa: E402
                    plane_fft_split, plane_resident_bytes)

TAG = "[plane-fft-p4]"
TOL, RED = 1.0e-13, 1.0e-3
QE = (24, 27, 30, 36, 40, 45, 48, 54, 60, 64, 72, 75, 80, 90, 96, 100, 108, 120, 125,
      128, 144, 150, 160, 180, 216, 240, 250)


def support(nb, nc, frac, rng, kind):
    """(nb*nc,) plane_from_col: a centred (wrapped) disk, a random set, or nothing."""
    if kind == "disk":
        hb, hc = np.fft.fftfreq(nb) * nb, np.fft.fftfreq(nc) * nc
        r2 = hb[:, None] ** 2 / nb ** 2 + hc[None, :] ** 2 / nc ** 2
        occ = (r2 <= np.quantile(r2, frac)).ravel()
    elif kind == "random":
        occ = rng.random(nb * nc) < frac
    else:
        occ = np.zeros(nb * nc, bool)
    cols = np.flatnonzero(occ)
    pfc = np.full(nb * nc, cols.size, np.int64)
    pfc[cols] = np.arange(cols.size)
    return pfc, int(cols.size)


def reference(F, pfc, n_col, nb, nc):
    if n_col == 0:
        return jnp.zeros(F.shape[:-1] + (nb, nc), F.dtype)
    x = jnp.take(F, jnp.asarray(pfc), axis=-1, mode="fill", fill_value=0)
    return jnp.fft.fftn(x.reshape(F.shape[:-1] + (nb, nc)), axes=(-2, -1))


def main():
    rank, nproc = jax.process_index(), jax.process_count()
    mesh = Mesh(np.array(jax.devices()).reshape(2, -1), ("x", "y"))
    assert kconv_backend(mesh) == "mathdx", kconv_backend(mesh)
    dev = jax.local_devices()[0]
    optin = _optin_smem_bytes()
    cases = []
    for n in QE:
        cases.append((n, n, 0.2, "disk", (7,)))
        cases.append((n, n, 0.45, "disk", (1,)))
        cases.append((n, n, 0.3, "random", (37,)))
    cases += [(54, 45, 0.2, "disk", (2, 3, 5)), (72, 80, 0.3, "random", (9,)),
              (24, 100, 0.45, "disk", (4,)), (54, 54, 0.0, "empty", (3,)),
              # The opt-in shared-memory edge (A100: 166912 B).  (105, 99) fits with its
              # static tables (mathdx); the other three fit the plane alone but not with
              # them, so they must route xla (the FFT audit's failing planes).
              (105, 99, 0.3, "disk", (2,)), (75, 138, 0.3, "disk", (2,)),
              (57, 182, 0.3, "disk", (2,)), (112, 92, 0.3, "disk", (2,))]
    worst, fails, n_m10, n_xla = 0.0, [], 0, 0
    for ci, (nb, nc, frac, kind, lead) in enumerate(cases):
        rng = np.random.default_rng(1000 * ci + 17)
        pfc, n_col = support(nb, nc, frac, rng, kind)
        fn = make_plane_fft_gather(mesh, pfc, n_col, (nb, nc))
        want = ("mathdx" if plane_fft_split(nb) and plane_fft_split(nc)
                and plane_resident_bytes(nb, nc) <= optin else "xla")
        if fn.route != want:
            fails.append(f"({nb},{nc}) route {fn.route} != {want}")
        n_m10 += fn.route == "mathdx"
        n_xla += fn.route == "xla"
        F = jax.device_put(rng.standard_normal(lead + (n_col,))
                           + 1j * rng.standard_normal(lead + (n_col,)), dev)
        y = jax.jit(fn)(F)
        ref = jax.jit(lambda F: reference(F, pfc, n_col, nb, nc))(F)
        scale = float(jnp.max(jnp.abs(ref))) if n_col else 1.0
        err = float(jnp.max(jnp.abs(y - ref))) / scale
        worst = max(worst, err) if fn.route == "mathdx" else worst
        line = f"{TAG} rank{rank} ({nb:3d},{nc:3d}) {kind:6s} {frac:.2f} lead={lead} n_col={n_col} route={fn.route} rel={err:.2e}"
        if not err <= TOL:
            fails.append(line)
        if fn.route == "mathdx" and n_col:
            bad = make_plane_fft_gather(mesh, np.roll(pfc, 1), n_col, (nb, nc))
            red = float(jnp.max(jnp.abs(jax.jit(bad)(F) - ref))) / scale
            line += f" red={red:.2e}"
            if not red > RED:
                fails.append(line + " RED TWIN PASSED")
        print(line, flush=True)
    # The slab form fn(F, start, size): F[:, start:start+size] in place, start traced
    # (a start past S - size clamps like lax.dynamic_slice).
    for nb, nc, start in ((54, 54, 1), (80, 80, 4), (45, 54, 0)):
        rng = np.random.default_rng(nb + nc + start)
        pfc, n_col = support(nb, nc, 0.2, rng, "disk")
        fn = make_plane_fft_gather(mesh, pfc, n_col, (nb, nc))
        F = jax.device_put(rng.standard_normal((3, 5, 2, n_col)) + 1j * rng.standard_normal((3, 5, 2, n_col)), dev)
        y = jax.jit(lambda F, s: fn(F, s, 3))(F, jnp.int32(start))
        ref = reference(jax.lax.dynamic_slice_in_dim(F, start, 3, axis=1), pfc, n_col, nb, nc)
        err = float(jnp.max(jnp.abs(y - ref)) / jnp.max(jnp.abs(ref)))
        line = f"{TAG} rank{rank} slab ({nb},{nc}) start={start} route={fn.route} shape={tuple(y.shape)} rel={err:.2e}"
        if not (err <= TOL and y.shape == (3, 3, 2, nb, nc)):
            fails.append(line)
        print(line, flush=True)
    nf = multihost_utils.process_allgather(np.array([len(fails), n_m10, n_xla], np.int64))
    for f in fails:
        print(f"{TAG} FAIL {f}", flush=True)
    if rank == 0:
        tot = nf.reshape(-1, 3).sum(0)
        print(f"{TAG} {'PASS' if tot[0] == 0 else 'FAIL'}: {len(cases)} cases, {tot[1]} mathdx, "
              f"{tot[2]} xla route, failures {tot[0]}; worst mathdx rel (rank0) {worst:.2e}", flush=True)
    if nf.reshape(-1, 3)[:, 0].sum():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
    finalize_process()
