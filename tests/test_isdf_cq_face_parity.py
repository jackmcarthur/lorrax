"""Shared rectangular Cq matches direct NumPy sums on four CPU or GPU devices."""
from __future__ import annotations

import os
import sys

# CLI multi-rank mode uses the same runtime boundary as production drivers
# -- mirrors test_distrib_la_multiproc.py exactly (init BEFORE importing
# jax/distrib_la, guarded by __main__, so a plain pytest collection never
# pays or triggers this).
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

from isdf.core import c_q_downfold
from common.gamma_matrices import gamma_perm_phase

PX = PY = 2

_CASES = (
    ("ns1_asym_upper", dict(ns=1, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8,
                             l_range=(0, 5), r_range=(0, 8), seed=1)),
    ("ns2_spinor",     dict(ns=2, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8,
                             l_range=(0, 5), r_range=(0, 8), seed=2)),
    ("ns1_asym_lower", dict(ns=1, nk_tuple=(1, 2, 1), n_rmu=4, nb_full=8,
                             l_range=(2, 8), r_range=(0, 8), seed=3)),
)

_GAMMA_CASES = tuple(
    (f"gamma_mu{mu_l}_nu{nu_l}",
     dict(ns=4, nk_tuple=(2, 1, 1), n_rmu=4, nb_full=8,
          l_range=(0, 5), r_range=(0, 8), seed=100 + 4 * mu_l + nu_l,
          gamma_mu_L=mu_l, gamma_nu_L=nu_l))
    for mu_l in range(4) for nu_l in range(4)
    if not (mu_l == 0 and nu_l == 0)
)


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def check_shared_cq(mesh, *, ns, nk_tuple, n_rmu, nb_full, l_range,
                    r_range, seed, gamma_mu_L=0, gamma_nu_L=0, n_col=None):
    """Compare the retained rectangular downfold Gram with explicit q and band sums."""
    from test_isdf_zq_parent_parity import _dense_pair_rhs
    from jax.experimental import multihost_utils as mhu

    nk = int(np.prod(nk_tuple))
    n_col = n_rmu if n_col is None else n_col
    psi = _crand(np.random.default_rng(seed), nk, nb_full, ns, n_col)
    bra = psi[:, :, :, :n_rmu].conj().transpose(0, 3, 1, 2)
    row = NamedSharding(mesh, P(None, "x", None, None))
    col = NamedSharding(mesh, P(None, None, None, "y"))
    l0, l1 = l_range
    r0, r1 = r_range
    gamma_l = None if gamma_mu_L == 0 else gamma_perm_phase(gamma_mu_L)
    gamma_r = None if gamma_nu_L == 0 else gamma_perm_phase(gamma_nu_L)
    got = c_q_downfold(
        jax.device_put(bra[:, :, l0:l1], row),
        jax.device_put(psi[:, l0:l1], col),
        jax.device_put(bra[:, :, r0:r1], row),
        jax.device_put(psi[:, r0:r1], col),
        gamma_L=gamma_l, gamma_R=gamma_r, kgrid=nk_tuple, mesh_xy=mesh)
    indices = np.arange(nb_full)
    reference = _dense_pair_rhs(
        psi, np.arange(n_rmu), nk_tuple,
        ((indices >= l0) & (indices < l1)).astype(float),
        ((indices >= r0) & (indices < r1)).astype(float), gamma_mu_L, gamma_nu_L)
    actual = np.asarray(mhu.process_allgather(got, tiled=True))
    assert actual.shape == (nk, n_rmu, n_col)
    relative = np.max(np.abs(actual-reference)) / np.max(np.abs(reference))
    assert relative < 1e-10, (ns, n_rmu, n_col, gamma_mu_L, gamma_nu_L, relative)



_ALL_CASES = _CASES + _GAMMA_CASES + (
    ("rectangular_ns1", dict(ns=1, nk_tuple=(2, 1, 1), n_rmu=4, n_col=8,
     nb_full=8, l_range=(0, 5), r_range=(2, 8), seed=31)),
    ("rectangular_ns2", dict(ns=2, nk_tuple=(1, 2, 1), n_rmu=4, n_col=8,
     nb_full=8, l_range=(2, 8), r_range=(0, 8), seed=32)),
)


@pytest.mark.parametrize("name,kwargs", _ALL_CASES, ids=[c[0] for c in _ALL_CASES])
def test_shared_cq_matches_direct_sums(name, kwargs):
    if jax.device_count() < PX * PY:
        pytest.skip("requires four devices, including an emulated CPU mesh")
    mesh = Mesh(np.asarray(jax.devices()).reshape(PX, PY), ("x", "y"))
    check_shared_cq(mesh, **kwargs)


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
    for name, kwargs in _ALL_CASES:
        try:
            check_shared_cq(mesh, **kwargs)
            p0(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            p0(f"FAIL {name}: {exc}")
    p0(f"done: {len(_ALL_CASES) - failures}/{len(_ALL_CASES)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_cli_main)
