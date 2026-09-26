"""Benchmark: W_q(G, G') on the q wedge at a real crystal's shapes, synthetic χ.

    lx run -N 1 -G 4 -n 4 python3 -u tests/bench/bench_plane_wave_screening.py \
        --wfn WFN.h5 --linalg local --n-p 8 --out arm.json [--cutoff-scale s]

* shapes come from the WFN header: the ψ spheres, the FFT box, the k-grid, the q
  wedge (``sym.q_irr_kgrid_int``) and the response sphere from
  ``screened_coulomb_cutoff`` = s·ecutwfc (``response_spheres``: resolved once on the
  full grid; refused past the box's alias cap);
* values are synthetic: χ_q(z_j) = −(R R^H)·a/M on the live slots at every wedge q and
  each of the 2n_p MPA samples (``double_parallel_grid``), and small S, Y, Z;
* stages, each timed cold (first call, compiles included) and warm (second call),
  with a device sync: the Dyson solve at every sample, the Γ head (the wing fold and
  the mini-BZ cell average per sample), W^c, and the MPA fit of every element;
* per-GPU peak bytes (``memory_stats``) against the module's memory law.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from runtime import initialize_communicator_stack, run_main_and_finalize  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import numpy as np  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402


def say(*a):
    if jax.process_index() == 0:
        print("[pw-screening-bench]", *a, flush=True)


def _peak():
    st = jax.local_devices()[0].memory_stats() or {}
    return int(st.get("peak_bytes_in_use", 0))


def _sync(x):
    jax.block_until_ready(x)
    return x


def setup(args, mesh):
    from file_io import WfnLoader
    from symmetry_maps import bgw_integer_q_to_fractional
    from vcoul import CoulombGeometry
    import h5py
    from gw.mixed_basis_pair_convolution import SphereSet
    from gw.plane_wave_screening import SphereScreening, response_spheres
    with h5py.File(args.wfn, "r") as f:
        ecut = float(f["mf_header/kpoints/ecutwfc"][()])
    with WfnLoader(args.wfn, backend="eager") as w:
        sym = w.symmetry()
        kgrid = tuple(int(v) for v in w.kgrid)
        box = tuple(int(v) for v in w.fft_grid)
        geo = CoulombGeometry.from_wfn(w)
        kf = np.asarray(w.kvecs(k="full_bz"))
        gf, nf = np.asarray(w.gvecs(k="full_bz")), np.asarray(w.ngk_valid(k="full_bz"))
    q_irr = bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, kgrid)
    rs = response_spheres(fft_grid=box, psi=SphereSet(gf, nf, kf), bvec=np.asarray(geo.bvec),
                          kgrid=kgrid, q_irr_frac=q_irr, ecutwfc=ecut,
                          screened_coulomb_cutoff=args.cutoff_scale * ecut)
    scr = SphereScreening(mesh, sphere=rs.irr, geometry=geo, sys_dim=3, kgrid=kgrid,
                          linalg=args.linalg)
    return scr, dict(ecut=ecut, kgrid=kgrid, box=box, n_q=int(rs.irr.n), n_full=int(rs.full.n))


def synthetic_chi(mesh, scr, n_z, seed=0):
    """(n_z, n_q, M, M) at P(None, None, 'x', 'y'): −a·R R^H/M on the live slots, one z at a time."""
    n_q, M = scr.sphere.n, scr.M
    spec3 = NamedSharding(mesh, P(None, "x", "y"))
    spec4 = NamedSharding(mesh, P(None, None, "x", "y"))
    ngk = jnp.asarray(scr.sphere.ngk, jnp.int32)
    a = 0.3 / float(np.max(scr.v))

    @jax.jit
    def one(key):
        k1, k2 = jax.random.split(key)
        R = jax.lax.with_sharding_constraint(
            jax.random.normal(k1, (n_q, M, M)) + 1j * jax.random.normal(k2, (n_q, M, M)), spec3)
        chi = -a * jnp.einsum("qab,qcb->qac", R, jnp.conj(R)) / M
        live = jnp.arange(M)[None, :] < ngk[:, None]
        m = live[:, :, None] & live[:, None, :]
        return jax.lax.with_sharding_constraint(jnp.where(m, chi, 0.0), spec3)
    stack = jax.jit(lambda *xs: jnp.stack(xs), out_shardings=spec4)
    return stack(*[one(jax.random.PRNGKey(seed + j)) for j in range(n_z)])


def small_fields(mesh, scr, n_z, seed=1):
    rng = np.random.default_rng(seed)
    M, w = scr.M, scr.sphere.width
    S = -0.01 * np.broadcast_to(np.eye(3), (n_z, 3, 3)).copy()
    Y = np.zeros((n_z, 3, M), np.complex128)
    Z = np.zeros((n_z, M, 3), np.complex128)
    Y[:, :, 1:w] = 1e-3 * rng.standard_normal((n_z, 3, w - 1))
    Z[:, 1:w, :] = 1e-3 * rng.standard_normal((n_z, w - 1, 3))
    put = lambda x, spec: jax.make_array_from_callback(x.shape, NamedSharding(mesh, spec),
                                                       lambda i: x[i])
    return S, put(Y, P(None, None, "x")), put(Z, P(None, "y", None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn", required=True)
    ap.add_argument("--linalg", default="local", choices=("local", "distributed"))
    ap.add_argument("--n-p", type=int, default=8)
    ap.add_argument("--cutoff-scale", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from gw.mpa import sampling
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, -1), ("x", "y"))
    t0 = time.perf_counter()
    scr, info = setup(args, mesh)
    say(f"setup {time.perf_counter() - t0:.2f} s; {info}")
    n_p = int(args.n_p)
    n_z = 2 * n_p
    say(scr.describe(n_z=n_z, n_p=n_p))
    z = sampling.double_parallel_grid(n_p, 2.0, energy_unit="Ry")
    S, Y, Z = small_fields(mesh, scr, n_z)
    rec = dict(info, linalg=args.linalg, n_p=n_p, M=scr.M, width=scr.sphere.width, P=scr.P,
               law=scr.describe(n_z=n_z, n_p=n_p), walls={})
    for label in ("cold", "warm"):
        chi = _sync(synthetic_chi(mesh, scr, n_z))
        base = _peak()
        t = time.perf_counter()
        W = _sync(scr.solve_samples(chi))
        t_dy = time.perf_counter() - t
        pk_dy = _peak()
        del chi
        t = time.perf_counter()
        vc0, w0, _ = scr.gamma_head(W, S, Y, Z)
        t_head = time.perf_counter() - t
        t = time.perf_counter()
        Wc, v = scr.correlation(W, wcoul0=w0, vc0=vc0)
        Wc = _sync(Wc)
        t_wc = time.perf_counter() - t
        del W
        qb = scr.fit_q_batch(n_p)
        t = time.perf_counter()
        Om, B, _, cond = scr.fit_poles(Wc, z, n_p)
        _sync((Om, B))
        t_fit = time.perf_counter() - t
        rec[f"fit_q_batch"] = qb
        walls = dict(dyson=t_dy, head=t_head, wc=t_wc, fit=t_fit,
                     total=t_dy + t_head + t_wc + t_fit)
        rec["walls"][label] = walls
        rec[f"peak_{label}_GB"] = _peak() / 1e9
        say(f"{label}: Dyson {t_dy:.3f} s ({n_z} samples x {scr.sphere.n} q), head {t_head:.3f} s, "
            f"W^c {t_wc:.3f} s, MPA fit {t_fit:.3f} s ({qb} q rows per batch, "
            f"{t_fit / (scr.sphere.n * scr.M ** 2 / scr.P) * 1e6:.2f} us per element); total "
            f"{walls['total']:.3f} s; peak after Dyson {pk_dy / 1e9:.2f}, after fit "
            f"{_peak() / 1e9:.2f} GB/GPU (before Dyson {base / 1e9:.2f} GB); "
            f"wcoul0[0] {w0[0]:.4f}, vc0 {vc0.real:.4f}; max fit cond {float(cond):.2e}")
        del Wc, Om, B
    if args.out and jax.process_index() == 0:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    run_main_and_finalize(main)
