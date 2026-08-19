"""``h_transform``'s kpath gates: the P>1 output fetch, and the metric route.

Two cells, two claims from HTRANSFORM_FFT.md, both reached by driving the real
``h_transform`` on a synthetic-but-legal Galerkin triple:

* ``test_post_kpath_outputs_are_replicated`` — the RED TWIN for the crash that
  killed two of the four reference decks at P=4 (PROFILE_htransform_exciton
  §1.5): ``_post_kpath`` carried no ``out_shardings``, so its outputs inherited
  the q-sharding of the batches it concatenates and the host fetch on the next
  line died whenever ``nq`` divided the device count.
* ``test_identity_metric_route_matches_the_cholesky_route`` — S is the identity
  from its one producer, so the two triangular solves per q are vestigial;
  removing them is analytic, not bitwise, and this pins the size of the
  difference.
"""
from __future__ import annotations

import numpy as np
import pytest


def _mesh(n=1):
    import jax
    from jax.sharding import Mesh
    devs = jax.devices()
    if len(devs) < n * n:
        pytest.skip(f"needs {n * n} devices, have {len(devs)}")
    return Mesh(np.asarray(devs[:n * n]).reshape(n, n), ("x", "y"))


def _synthetic(nk_grid=(2, 2, 1), nb=4, rank=32, n_mu=6, ns=2, seed=11):
    """Band-orthonormal ``ctilde`` per k — what ``streaming_galerkin_solve``
    produces and what ``build_fH_R``'s gate requires.  Same construction as
    ``test_bse_setup_qchunk._synthetic``."""
    rng = np.random.default_rng(seed)
    nk = nk_grid[0] * nk_grid[1] * nk_grid[2]
    ct = np.empty((nk, nb, rank), dtype=np.complex128)
    for k in range(nk):
        z = (rng.standard_normal((rank, nb))
             + 1j * rng.standard_normal((rank, nb)))
        q, _ = np.linalg.qr(z)
        ct[k] = np.conj(q.T)
    enk = (np.linspace(-0.6, 0.4, nb)[:, None]
           + 0.05 * np.cos(2 * np.pi * np.arange(nk) / nk)[None, :])
    B = (rng.standard_normal((rank, ns, n_mu))
         + 1j * rng.standard_normal((rank, ns, n_mu)))
    return ct, enk, B, nk_grid


def _kpath_inputs(nk_grid=(2, 2, 1), nb=4, rank=32, nq=8):
    from types import SimpleNamespace
    ct, enk, _B, _ = _synthetic(nk_grid=nk_grid, nb=nb, rank=rank)
    meta = SimpleNamespace(
        nkx=nk_grid[0], nky=nk_grid[1], nkz=nk_grid[2], nval=2)
    wfn = SimpleNamespace(efermi=0.0, nelec=2)
    rng = np.random.default_rng(5)
    kpath = rng.uniform(-0.5, 0.5, size=(nq, 3))
    x_path = np.arange(nq, dtype=float)
    return meta, ct, enk, wfn, (kpath, x_path, [0], [None], [])


def test_vbm_index_is_local_to_nonzero_band_window():
    """A CrI3-like 20-band window starts at 120, so its VBM is column 9."""
    from bandstructure.htransform import _local_vbm_index

    assert _local_vbm_index(nelec=130, nval=10, nb_keep=20) == 9
    with pytest.raises(ValueError, match="outside the loaded band window"):
        _local_vbm_index(nelec=130, nval=0, nb_keep=20)


@pytest.mark.parametrize("ndev", [1, 4])
def test_post_kpath_outputs_are_replicated(ndev):
    """RED TWIN for the P>1 crash.

    ``_post_kpath`` used to carry no ``out_shardings``, so its outputs
    inherited the q-sharding of the batches it concatenates.  Whether the SPMD
    partitioner then replicated them or left them split depended on whether
    ``nq`` divided the device count — and when it did, the ``np.asarray`` on
    the next line raised ``Fetching value for jax.Array that spans
    non-addressable devices`` in a MULTI-PROCESS run.  ``nq = 8`` here divides
    4, which is exactly the ``ht40.in`` at P=4 geometry that died.

    This cell cannot raise that error single-process (every device is
    addressable), so it gates the INVARIANT that makes the fetch legal: the
    driver reports the spec it actually got, and it must be fully replicated.
    """
    pytest.importorskip("jax")
    import jax
    from bandstructure.htransform import h_transform
    mesh = _mesh(1 if ndev == 1 else 2)
    meta, ct, enk, wfn, kpath_data = _kpath_inputs()
    import jax.numpy as jnp
    rank = ct.shape[2]
    S = jnp.eye(rank, dtype=jnp.complex128)
    lines = []
    with mesh:
        res = h_transform(meta, S, jnp.asarray(ct), jnp.asarray(enk), wfn,
                          kpath_data, lines.append, mesh, diagnostics=False)
    banner = " ".join(lines)
    assert "[gate] _post_kpath out spec:" in banner, banner
    spec = banner.split("[gate] _post_kpath out spec:")[1].split()[0]
    assert spec in ("P()", "PartitionSpec()"), (
        f"_post_kpath returned {spec}, not the replicated P().  At P>1 with "
        f"nq divisible by the device count this is the non-addressable-fetch "
        f"crash of PROFILE_htransform_exciton §1.5.")
    assert isinstance(res["energies_sorted"], np.ndarray)
    assert res["energies_sorted"].shape[0] == 8
    _ = jax.devices()          # keep the jax import meaningful to linters


def test_identity_metric_route_matches_the_cholesky_route():
    """The S = I fast path drops two triangular solves per q.  It is NOT
    bit-identical — the shipped path divides every eigenvalue by (1+1e-10)
    through the ridge on chol(S) — so this cell pins the size of that
    difference rather than asserting an equality that is not true."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from bandstructure.htransform import h_transform
    mesh = _mesh(1)
    meta, ct, enk, wfn, kpath_data = _kpath_inputs()
    rank = ct.shape[2]
    S_eye = jnp.eye(rank, dtype=jnp.complex128)
    # A metric that is the identity to the LAST BIT except for one entry
    # perturbed far above the route's ``== 0.0`` test: same physics, but it
    # forces the Cholesky route so the two can be compared at all.
    S_near = S_eye.at[0, 0].add(1e-13)
    outs = []
    for S in (S_eye, S_near):
        lines = []
        with mesh:
            outs.append(h_transform(meta, S, jnp.asarray(ct), jnp.asarray(enk),
                                    wfn, kpath_data, lines.append, mesh,
                                    diagnostics=False))
        banner = " ".join(lines)
        outs[-1] = (outs[-1], "identity" in banner.split("metric: ")[1][:10])
    (a, a_id), (b, b_id) = outs
    assert a_id and not b_id, (a_id, b_id)
    d = float(np.max(np.abs(a["energies_sorted"] - b["energies_sorted"])))
    assert d < 1e-6, f"identity route moved an energy by {d:.3e} Ry"
