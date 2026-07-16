"""Gate: the non-TDA RPA-screening resolvent reproduces the GW static W.

``bse_w_exact`` computes W(0) - v = v (0 - H_RPA)^{-1} v with the non-TDA
symplectic RPA density-response Hamiltonian (ring kernel V in both blocks).
This must reproduce the restart's head-less (W0_qmunu - V_qmunu) q=0 tile to the
GW minimax-integration floor (~1e-9 on the gnppm fixture).  Exercises the whole
chain touched by the W(0) cross-check: ``inject_head=False`` loading, the
``screening=True`` B-block, ``ensure_W_R``, the probe generator/snapshot, and the
[f; -f] / X+Y symplectic convention.
"""
import numpy as np
import pytest
import jax
from jax.sharding import Mesh

from bse import bse_io
from bse.bse_w_exact import (
    _build_rpa_resolvent, _resolve_wc_columns, _select_compare_cols,
)


@pytest.mark.gpu
def test_w0_resolvent_matches_restart(gnppm_session):
    input_path = str(gnppm_session.run_dir / gnppm_session.input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))

    # FULL chi0 window (n_occ x n_cond), head-less bodies on both sides.
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=input_path, inject_head=False)
    nlog = int(data["n_rmu"])
    T = (np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
         - np.asarray(jax.device_get(data["V_q0"])))

    matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh, data)
    cols, _ = _select_compare_cols(T, nlog, 4, seed=0)
    wc, resids = _resolve_wc_columns(
        cols, 0.0 + 0.0j, data, matvec, diag_h, gen, snapshot, sh,
        max_iter=200, tol=1e-10)

    for i, nu0 in enumerate(cols):
        tcol = T[:nlog, int(nu0)]
        rel = float(np.linalg.norm(wc[i, :nlog] - tcol) / np.linalg.norm(tcol))
        assert resids[i] < 1e-6, f"col {nu0}: GMRES not converged (resid={resids[i]:.2e})"
        assert rel < 1e-6, f"col {nu0}: W(0) resolvent closure rel_err={rel:.2e} (>1e-6)"
