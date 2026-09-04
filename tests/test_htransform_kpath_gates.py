"""``h_transform``'s kpath gates: the P>1 output fetch, and the metric route.

Two cells, two claims from HTRANSFORM_FFT.md, both reached by driving the real
``h_transform`` on a synthetic-but-legal Galerkin triple:

* ``test_post_kpath_outputs_are_replicated`` — the RED TWIN for the crash that
  killed two of the four reference decks at P=4 (PROFILE_htransform_exciton
  §1.5): ``_post_kpath`` carried no ``out_shardings``, so its outputs inherited
  the q-sharding of the batches it concatenates and the host fetch on the next
  line died whenever ``nq`` divided the device count.  The same cell pins the
  native batch to one whole matrix/device; a placement-legal fixed width of 32
  OOMed the rank-4800/P16 CrI3 arm by assigning two per device.
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
    # A one-device submesh must be process-local in a multi-process P4 gate;
    # ``jax.devices()[0]`` is rank 0's GPU and is non-addressable on ranks 1--3.
    # The 2x2 case intentionally exercises the global four-device mesh.
    devs = jax.local_devices() if n == 1 else jax.devices()
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
    meta = SimpleNamespace(nkx=nk_grid[0], nky=nk_grid[1], nkz=nk_grid[2])
    wfn = SimpleNamespace(efermi=0.0, nelec=2)
    rng = np.random.default_rng(5)
    kpath = rng.uniform(-0.5, 0.5, size=(nq, 3))
    x_path = np.arange(nq, dtype=float)
    return meta, ct, enk, wfn, (kpath, x_path, [0], [None], [])


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
    lines = []
    with mesh:
        res = h_transform(meta, jnp.asarray(ct), jnp.asarray(enk), wfn,
                          kpath_data, lines.append, mesh,
                          n_return_bands=ct.shape[1] - 1)
    banner = " ".join(lines)
    assert (f"kpath native eig ledger: q-batch={ndev}, ndev={ndev}, "
            f"whole matrices/device=1" in banner), banner
    assert "[gate] _post_kpath out spec:" in banner, banner
    spec = banner.split("[gate] _post_kpath out spec:")[1].split()[0]
    assert spec in ("P()", "PartitionSpec()"), (
        f"_post_kpath returned {spec}, not the replicated P().  At P>1 with "
        f"nq divisible by the device count this is the non-addressable-fetch "
        f"crash of PROFILE_htransform_exciton §1.5.")
    assert isinstance(res["energies_sorted"], np.ndarray)
    assert res["energies_sorted"].shape == (8, ct.shape[1] - 1)
    assert res["energies_on_path"].shape == (8, ct.shape[1] - 1)
    _ = jax.devices()          # keep the jax import meaningful to linters


def test_htransform_carries_no_dense_identity_metric():
    """The selected-state basis owns one orthonormal gauge, not a dense S."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from bandstructure.htransform import h_transform
    mesh = _mesh(1)
    meta, ct, enk, wfn, kpath_data = _kpath_inputs()
    lines = []
    with mesh:
        out = h_transform(meta, jnp.asarray(ct), jnp.asarray(enk), wfn,
                          kpath_data, lines.append, mesh,
                          n_return_bands=ct.shape[1] - 1)
    assert np.all(np.isfinite(out["energies_sorted"]))
    banner = " ".join(lines)
    assert "metric: identity" in banner
    assert "no dense" in banner


@pytest.mark.parametrize("ndev", [1, 4])
def test_fh_builder_consumes_rank_shards_and_keeps_small_certificate(ndev):
    """The coarse-grid certificate survives removal of replicated ctilde.

    This is the red twin for the 64x64x1 MoS2 memory wall.  The old builder
    required ``P()`` for both coefficient operands, so a 4096x80x1656 c128
    table occupied 8.68 GiB on every device and remained live beside fH.  A
    source rank shard must now reach the real fH/FFT/path pipeline without an
    all-gather, while the synthetic fixture's exact coarse points remain
    below the same value-level certificate.
    """
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from bandstructure.htransform import h_transform
    from symmetry_maps import SymMaps

    mesh = _mesh(1 if ndev == 1 else 2)
    meta, ct, enk, wfn, _ = _kpath_inputs(
        nk_grid=(2, 2, 1), nb=4, rank=8, nq=4)
    coarse_k = np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0],
        [-0.5, 0.0, 0.0],
        [-0.5, -0.5, 0.0],
    ])
    kpath_data = (coarse_k, np.arange(4.0), [0], ["Gamma"], [0])
    sym = object.__new__(SymMaps)
    sym.unfolded_kpts = coarse_k
    lines = []
    with mesh:
        ct_x = jax.device_put(
            jnp.asarray(ct), NamedSharding(mesh, P(None, None, 'x')))
        out = h_transform(
            meta, ct_x, jnp.asarray(enk), wfn, kpath_data, lines.append,
            mesh, n_return_bands=ct.shape[1] - 1, sym=sym)

    assert out["coincident_max_abs_ry"] < 2.0e-11
    assert np.all(np.isfinite(out["energies_sorted"]))
    assert (
        "paired rank shards P(None,None,'x') / P(None,None,'y'); "
        "no replicated ctilde in fH" in " ".join(lines))


def test_active_character_follows_state_through_guard_energy_crossing():
    """A lower DFT guard must not replace a raised active QP state."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from bandstructure.htransform import select_active_eigenpairs

    # Active state 2 lies above guard state 1.  Lowest-energy truncation would
    # return (-3, -2, -1); character selection must return (-3, -2, +1).
    values = jnp.asarray([[-3.0, -2.0, -1.0, 1.0, 2.0]])
    vectors = jnp.eye(5, dtype=jnp.complex128)[None]
    active = jnp.diag(jnp.asarray([1.0, 1.0, 0.0, 1.0, 0.0]))[None]
    (selected, scores, gap, tol, cluster_values, cluster_mask,
     min_nonreturned) = (
        select_active_eigenpairs(values, vectors, active, 3))

    np.testing.assert_array_equal(np.asarray(selected), [[-3.0, -2.0, 1.0]])
    np.testing.assert_array_equal(np.asarray(scores), [[1.0, 1.0, 1.0]])
    assert float(gap[0]) == 1.0
    assert float(gap[0]) > float(tol[0])
    assert not np.any(np.asarray(cluster_mask))
    assert np.asarray(cluster_values).shape == (1, 5)
    np.testing.assert_array_equal(np.asarray(min_nonreturned), [-1.0])


def test_degenerate_character_boundary_has_invariant_energy_multiset():
    """A basis rotation at an exact crossing cannot change published energy."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from bandstructure.htransform import select_active_eigenpairs

    values = jnp.asarray([[-1.0, 0.0, 0.0, 2.0]])
    active = np.diag([1.0, 1.0, 0.0, 0.0]).astype(np.complex128)
    identity = np.eye(4, dtype=np.complex128)
    angle = np.pi / 4.0
    rotated = identity.copy()
    rotated[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]]

    outputs = []
    gaps = []
    boundaries = []
    for vectors in (identity, rotated):
        (selected, _scores, gap, tol, cluster_values, cluster_mask,
         _min_nonreturned) = (
            select_active_eigenpairs(
                values, jnp.asarray(vectors[None]),
                jnp.asarray(active[None]), 2))
        outputs.append(np.asarray(selected))
        gaps.append(float(gap[0]))
        boundaries.append(np.asarray(cluster_values)[np.asarray(cluster_mask)])
        assert float(gap[0]) <= float(tol[0]) or vectors is identity

    np.testing.assert_array_equal(outputs[0], [[-1.0, 0.0]])
    np.testing.assert_array_equal(outputs[1], [[-1.0, 0.0]])
    # The unresolved character boundary in the rotated representation lies
    # wholly inside the zero-energy multiplet, so the energy output is safe.
    assert gaps[1] <= 1.0e-14
    np.testing.assert_array_equal(boundaries[1], [0.0, 0.0])


def test_null_carrier_character_cannot_displace_a_fitted_state():
    """Only the fitted physical spectrum participates in state selection."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from bandstructure.htransform import select_active_eigenpairs

    values = jnp.asarray([[-3.0, -2.0, -1.0, 0.0]])
    vectors = jnp.eye(4, dtype=jnp.complex128)[None]
    # The rank-minus-state null carrier is deliberately assigned the largest
    # apparent character.  It is not a fitted state and must be sliced out.
    active = jnp.diag(jnp.asarray([1.0, 1.0, 0.0, 10.0]))[None]
    selected, *_rest = select_active_eigenpairs(
        values, vectors, active, 2, n_physical=3)
    np.testing.assert_array_equal(np.asarray(selected), [[-3.0, -2.0]])


def test_character_tie_cluster_includes_every_boundary_member():
    """Three-way tie red twin: the nondegenerate third member cannot hide."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from bandstructure.htransform import select_active_eigenpairs

    values = jnp.asarray([[-1.0, 0.0, 0.0, 0.1]])
    vectors = jnp.eye(4, dtype=jnp.complex128)[None]
    active = jnp.diag(jnp.asarray([1.0, 0.5, 0.5, 0.5]))[None]
    (_selected, _scores, gap, tol,
     cluster_values, cluster_mask, _min_nonreturned) = select_active_eigenpairs(
         values, vectors, active, 2)
    assert float(gap[0]) <= float(tol[0])
    tied = np.asarray(cluster_values)[np.asarray(cluster_mask)]
    np.testing.assert_array_equal(tied, [0.0, 0.0, 0.1])
    assert float(tied.max() - tied.min()) > 1.0e-3


def test_htransform_active_window_beats_lower_guard_on_the_path():
    """Drive the real fH/FFT/Newton path through the crossing red twin."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from types import SimpleNamespace
    from bandstructure.htransform import h_transform
    from symmetry_maps import SymMaps

    mesh = _mesh(1)
    nk, states, rank = 2, 6, 8
    ctilde = np.broadcast_to(
        np.eye(states, rank, dtype=np.complex128),
        (nk, states, rank)).copy()
    # First three rows are the requested active block.  Guard row 3 is lower
    # than active row 2, while rows 4/5 keep the f-transform shoulder above it.
    energies = np.broadcast_to(
        np.asarray([-3.0, -2.0, 1.0, -1.0, 2.0, 3.0])[:, None],
        (states, nk)).copy()
    meta = SimpleNamespace(nkx=2, nky=1, nkz=1)
    wfn = SimpleNamespace(efermi=0.0, nelec=2)
    kpath = np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    kpath_data = (kpath, np.arange(2.0), [0], ["Gamma"], [0])
    sym = object.__new__(SymMaps)
    sym.unfolded_kpts = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    lines = []
    with mesh:
        result = h_transform(
            meta, jnp.asarray(ctilde), jnp.asarray(energies), wfn,
            kpath_data, lines.append, mesh,
            n_return_bands=3, sym=sym)

    # VBM is active row 1 at -2 Ry, so the raised active state appears at +3
    # Ry.  The old energy truncation published the lower guard at +1 Ry.
    np.testing.assert_allclose(
        result["energies_sorted"], [[-1.0, 0.0, 3.0]] * 2,
        rtol=0.0, atol=2.0e-11)
    np.testing.assert_array_equal(result["coincident_path_indices"], [0])
    np.testing.assert_array_equal(result["coincident_coarse_indices"], [0])
    np.testing.assert_allclose(
        result["coincident_exact"], [[-1.0, 0.0, 3.0]],
        rtol=0.0, atol=2.0e-11)
    assert result["coincident_max_abs_ry"] < 2.0e-11
    np.testing.assert_allclose(
        result["gamma_exact"], [-1.0, 0.0, 3.0],
        rtol=0.0, atol=2.0e-11)
    assert "active/guard character selection" in " ".join(lines)
    assert "path/coarse coincidences: 1 path row" in " ".join(lines)


@pytest.mark.parametrize(("first_dft_guard", "must_refuse"), (
    (2.0, False),
    (-1.5, True),
))
def test_authenticated_qp_corrected_margin_protects_returned_interior(
        first_dft_guard, must_refuse):
    """A wider authenticated QP block is required and its energy margin gates."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from types import SimpleNamespace
    from bandstructure.htransform import h_transform

    mesh = _mesh(1)
    nk, states, rank = 2, 7, 8
    ctilde = np.broadcast_to(
        np.eye(states, rank, dtype=np.complex128),
        (nk, states, rank)).copy()
    # [0,4) is the authenticated QP block; [0,3) is returned.  A DFT guard
    # below band 2 makes a future active/guard character swap observable and
    # must refuse even though the pointwise character order itself is exact.
    energies = np.broadcast_to(
        np.asarray([-3.0, -2.0, -1.0, 1.0,
                    first_dft_guard, 3.0, 4.0])[:, None],
        (states, nk)).copy()
    meta = SimpleNamespace(nkx=2, nky=1, nkz=1)
    wfn = SimpleNamespace(efermi=0.0, nelec=2)
    kpath = np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    kpath_data = (kpath, np.arange(2.0), [0], ["Gamma"], [0])
    lines = []

    def run():
        with mesh:
            return h_transform(
                meta, jnp.asarray(ctilde), jnp.asarray(energies), wfn,
                kpath_data, lines.append, mesh,
                band_start=0, n_return_bands=3,
                qp_corrected_band_range=(0, 4))

    if must_refuse:
        with pytest.raises(ValueError, match="protect the returned interior"):
            run()
    else:
        result = run()
        np.testing.assert_allclose(
            result["energies_sorted"], [[-1.0, 0.0, 1.0]] * 2,
            rtol=0.0, atol=2.0e-11)
        assert "corrected interior margin" in " ".join(lines)
