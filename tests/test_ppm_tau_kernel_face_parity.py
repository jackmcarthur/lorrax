"""Compare parent Sigma(tau) against independent NumPy band/q sums on real P4."""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    _TESTS = os.path.dirname(os.path.abspath(__file__))
    _REPO = os.path.dirname(_TESTS)
    for _svc in ("lxkit", "distrib_la"):
        _src = os.path.join(_REPO, "services", _svc, "src")
        if os.path.isdir(_src) and _src not in sys.path:
            sys.path.insert(0, _src)
    from lxkit.gate import platform_from_env
    from runtime import initialize_communicator_stack
    _plat = platform_from_env()
    _RUNTIME = initialize_communicator_stack(
        platform="gpu" if _plat == "CUDA" else "cpu")

import argparse

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.ppm_tau_kernel import get_shared_sigma_tau_kernel
from gw.wavefunction_bundle import BandSlices, parent_sigma_operands, sigma_face_kernel_kwargs
from multi_device.full_photon_head_sigma_gate import _bundle

PX = PY = 2


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


#: (ns, nk_tuple, n_rmu, nb_full, nb_sigma, weight_kind, brackets,
#:  seed) -- see module docstring for what each case exercises.
#: weight_kind: 'bool' -> build_G_tau's mask= seam; 'float' -> band_weight=.
_CASES = (
    ("identity_ns1", dict(
        ns=1, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8, nb_sigma=8,
        weight_kind="bool", brackets=((0, None),), seed=1)),
    ("identity_ns2", dict(
        ns=2, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8, nb_sigma=8,
        weight_kind="bool", brackets=((0, None),), seed=2)),
    ("real_weight_ns1_nondivisible", dict(
        ns=1, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8, nb_sigma=5,
        weight_kind="float", brackets=((0, None),), seed=3)),
    ("real_weight_ns2", dict(
        ns=2, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8, nb_sigma=8,
        weight_kind="float", brackets=((0, None),), seed=4)),
    ("brackets_ns1", dict(
        ns=1, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8, nb_sigma=8,
        weight_kind="bool",
        brackets=((0, 3), (3, 6), (6, None)), seed=5)),
    ("brackets_ns2", dict(
        ns=2, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=10, nb_sigma=10,
        weight_kind="bool", brackets=((0, 3), (3, 7), (7, 10)), seed=202)),
    ("signed_brackets_ns4_nondivisible", dict(
        ns=4, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=10, nb_sigma=5,
        weight_kind="float", brackets=((0, 3), (3, 7), (7, 10)), seed=203)),
    ("unbracketed_ns4", dict(
        ns=4, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8, nb_sigma=5,
        weight_kind="float", brackets=None, seed=204)),
)


def _to_host(x, mesh):
    from jax.experimental import multihost_utils as mhu
    del mesh
    return np.asarray(mhu.process_allgather(x, tiled=True))


def check_tau_kernel_parent_dense(mesh, *, ns, nk_tuple, n_rmu, nb_full,
                                 nb_sigma, weight_kind, brackets,
                                 seed):
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    nkx, nky, nkz = nk_tuple
    nk = nkx * nky * nkz
    kgrid = nk_tuple
    rng = np.random.default_rng(seed)

    psi_full = _crand(rng, nk, nb_full, ns, n_rmu)
    E_A = jnp.asarray(rng.uniform(-1.0, 1.0, size=(nk, nb_full)))
    if weight_kind == "bool":
        sel = jnp.asarray(rng.uniform(size=(nk, nb_full)) > 0.5)
    else:
        sel = jnp.asarray(rng.uniform(-0.125, 1.125, size=(nk, nb_full)))
    B_q = jnp.asarray(_crand(rng, nk, n_rmu, n_rmu))
    Omega_q = jnp.asarray(
        rng.uniform(0.2, 3.0, size=(nk, n_rmu, n_rmu))
        - 1j * rng.uniform(0.0, 0.5, size=(nk, n_rmu, n_rmu)))
    mask_B = jnp.asarray(rng.uniform(size=(nk, n_rmu, n_rmu)) > 0.3)
    B_poles = jnp.where(mask_B, B_q, 0.0)[None, ...]
    Omega_poles = Omega_q[None, ...]
    pole_indices = jnp.asarray([0], dtype=jnp.int32)
    bounds = jnp.asarray([[0.0, np.inf, -np.inf, -np.inf,
                           np.inf, np.inf]], dtype=jnp.float64)
    phase_real = jnp.asarray([False])
    E_ref_A = jnp.asarray(0.0, dtype=jnp.float64)
    E_ref_B = jnp.asarray(0.0, dtype=jnp.float64)
    t_node = jnp.asarray(0.15 + 0.07j, dtype=jnp.complex128)

    slices = BandSlices.from_band_edges(0, 0, 0, nb_sigma, nb_full)
    wfns = _bundle(mesh, psi_full, np.asarray(E_A), np.zeros((nk, nb_full)), slices)
    xn, yr, xr, yn, energy, _ = parent_sigma_operands(wfns)
    kernel = get_shared_sigma_tau_kernel(
        mesh_xy=mesh, kgrid=kgrid, brackets=brackets,
        **sigma_face_kernel_kwargs(wfns))
    got = _to_host(kernel(xn, yr, xr, yn, energy, sel,
        B_poles, Omega_poles, pole_indices, bounds, phase_real,
        E_ref_A, E_ref_B, t_node), mesh)[..., :nb_sigma, :nb_sigma]
    reference = _dense_tau(psi_full, np.asarray(E_A), np.asarray(sel),
        np.asarray(B_poles[0]), np.asarray(Omega_q), complex(t_node),
        brackets, nb_sigma)
    assert got.shape == reference.shape
    error = float(np.max(np.abs(got - reference)))
    relative = error / max(float(np.max(np.abs(reference))), 1e-300)
    p0(f"ns={ns} nb_sigma={nb_sigma} brackets={brackets}: "
       f"NumPy max_abs={error:.16e} max_rel={relative:.16e}")
    assert relative < 2e-12


def _dense_tau(psi, energy, selector, residue, pole, t, brackets, nb_sigma):
    """Sum intermediate bands and momentum transfers with the literal negative GW convolution."""
    nk, nb, ns, mu = psi.shape
    weights = selector * np.exp(-1j * energy * t)
    W = residue * np.exp(-1j * pole * t)
    outputs = []
    for lo, hi in (((0, nb),) if brackets is None else brackets):
        mask = (np.arange(nb) >= lo) & (np.arange(nb) < (nb if hi is None else hi))
        green = np.einsum("kb,kbsm,kbtn->ksmtn", weights * mask, psi, psi.conj())
        out = np.zeros((nk, nb_sigma, nb_sigma), complex)
        for k in range(nk):
            sigma = np.zeros((ns, mu, ns, mu), complex)
            for q in range(nk):
                sigma -= green[(k-q) % nk] * W[q][None, :, None, :] / nk
            out[k] = np.einsum("asm,smtn,btn->ab",
                psi[k, :nb_sigma].conj(), sigma, psi[k, :nb_sigma])
        outputs.append(out)
    return outputs[0] if brackets is None else np.stack(outputs)


@pytest.mark.parametrize("name,kwargs", _CASES, ids=[c[0] for c in _CASES])
def test_tau_kernel_parent_matches_dense(name, kwargs):
    if jax.process_count() < PX * PY:
        pytest.skip(
            f"needs {PX * PY} REAL processes for gemm_plan's cuBLASMp "
            f"GUARD 4 (got process_count={jax.process_count()}); run "
            f"`lx run -N 1 -G 4 -n 4 ... {__file__} --mesh 2x2` for the "
            f"real check (see this module's docstring)")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_tau_kernel_parent_dense(mesh, **kwargs)


def _cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="2x2", help="PxQ process mesh")
    args = ap.parse_args()
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()}")
    if jax.device_count() != px * py:
        p0(f"REFUSE: need exactly {px * py} devices for a {args.mesh} mesh; "
           f"got {jax.device_count()}")
        return 1
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))
    failures = 0
    for name, kwargs in _CASES:
        try:
            check_tau_kernel_parent_dense(mesh, **kwargs)
            p0(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {name}: {exc}")
    p0(f"done: {len(_CASES) - failures}/{len(_CASES)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_cli_main)
