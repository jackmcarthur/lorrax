"""The ζ fit exposes only selected q rows at its host commit boundaries."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np


def test_z_q_phase_keeps_full_numerical_kernel_before_external_selection(
    monkeypatch,
):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    import isdf.core as core

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    full = jnp.arange(5 * 4 * 6, dtype=jnp.float64).reshape(5, 4, 6)
    monkeypatch.setattr(core, "z_q_from_psi_sm", lambda **_kwargs: full)
    meta = SimpleNamespace(
        nk_tot=5, n_rmu=4, n_rmu_padded=4, nspinor=1,
        kgrid=(5, 1, 1), fft_grid=(1, 1, 1),
    )
    fn = core._make_fit_one_rchunk_kernel(
        mesh, meta, ((0, 1),), 1,
        object(), q_irr_full_idx=np.asarray([0, 3], dtype=np.int32),
        k_unfold_plan=object())
    got = fn.z_q_phase(
        None,
        jnp.zeros((1, 1, 4, 1), dtype=jnp.complex128),
        jnp.ones((1,)), jnp.ones((1,)), None, None, None)

    assert got.shape == (5, 4, 6)
    assert np.array_equal(np.asarray(got), np.asarray(full))


def test_c_and_prebuilt_z_select_before_outer_block_until_ready():
    """Pin the two host seams whose old ordering retained K-Q rows."""
    from gw import isdf_fitting
    import isdf.core as core

    fit_source = inspect.getsource(isdf_fitting.fit_zeta_to_h5)
    c_slice = fit_source.index("C_q_flat = slice_q_full_to_ibz(")
    c_wait = fit_source.index("C_q_flat.block_until_ready()", c_slice)
    assert c_slice < c_wait
    assert "C_q.block_until_ready()" not in fit_source

    chunk_source = inspect.getsource(core.fit_one_rchunk)
    ordinary = chunk_source.index("Z_q = fn.z_q_phase(")
    ordinary_slice = chunk_source.index("Z_q = Z_q[jnp.asarray(", ordinary)
    ordinary_wait = chunk_source.index(
        "Z_q.block_until_ready()", ordinary_slice)
    assert ordinary < ordinary_slice < ordinary_wait

    prebuilt = chunk_source.index("Z_q = _prebuilt_Z_q")
    z_slice = chunk_source.index("Z_q = Z_q[jnp.asarray(", prebuilt)
    z_wait = chunk_source.index("Z_q.block_until_ready()", z_slice)
    assert prebuilt < z_slice < z_wait
