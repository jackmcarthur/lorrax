"""P=4 gate for the WP3 analytic suffix's collective append lifecycle."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
for _service in ("distrib_la", "lxkit"):
    sys.path.insert(0, os.path.join(ROOT, "services", _service, "src"))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack()

import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from common.collectives import barrier, process_count, process_rank, resolve_mesh  # noqa: E402
from file_io import mpa_store  # noqa: E402
from file_io.slab_io import mesh_divisible_shape  # noqa: E402


N_Q = 2
N_MU = 5
N_FIT = 2
N_INTRA = 1
CERT = {
    "condition_max_allowed": 1.0e12,
    "backward_error_max_allowed": 1.0,
    "intraband_total_sample_max_rel_error": 2.0e-4,
    "intraband_total_sample_max_rel_error_max_allowed": 4.0e-3,
    "intraband_static_max_rel_error": 2.0e-14,
    "intraband_static_max_rel_error_max_allowed": 2.0e-11,
    "intraband_gap_max_rel_error": 6.0e-4,
    "intraband_gap_max_rel_error_max_allowed": 1.0e-3,
}


def _fail(message):
    raise RuntimeError(f"[intraband_store rank={process_rank()}] {message}")


def main():
    if process_count() != 4:
        _fail(f"requires four processes, got {process_count()}")
    mesh = resolve_mesh()
    evidence = os.environ.get("INTRABAND_STORE_GATE_DIR", os.getcwd())
    step = os.environ.get("SLURM_STEP_ID", "manual")
    root = os.path.join(evidence, "intraband_store_gate_" + step)
    path = os.path.join(root, "poles.h5")
    if process_rank() == 0:
        os.makedirs(root, exist_ok=False)
    barrier("intraband_store_gate_mkdir")

    mpa_store.allocate_fit_store_collective(
        path, mesh_xy=mesh, n_q=N_Q, n_mu=N_MU,
        n_p=N_FIT + N_INTRA, n_p_fit=N_FIT,
        intraband_model="intraband_eigenmode_v1",
        diagnostic_keys=mpa_store.REQUIRED_DIAGNOSTICS,
        energy_unit="Ry")
    if process_rank() == 0:
        for iq in range(N_Q):
            omega = np.full(
                (N_FIT, N_MU, N_MU), 0.7 + 0.1 * iq - 0.02j,
                np.complex128)
            residues = np.full_like(omega, 0.2 + 0.03j)
            diagnostics = {
                "condition": np.full((N_MU, N_MU), 12.0 + iq),
                "backward_error": np.full((N_MU, N_MU), 2.0e-14),
            }
            mpa_store.write_fit_block(
                path, iq, np.arange(N_MU), omega, residues, diagnostics)
    barrier("intraband_store_gate_fit_prefix")

    spec = P(None, None, "x", "y")
    padded = mesh_divisible_shape((N_INTRA, 1, N_MU, N_MU), mesh, spec)
    sharding = NamedSharding(mesh, spec)
    for iq in range(N_Q):
        omega_host = np.ones(padded, np.complex128)
        residue_host = np.zeros(padded, np.complex128)
        omega_host[0, 0, :N_MU, :N_MU] = 0.05 - 0.004j
        residue_host[0, 0, :N_MU, :N_MU] = (iq + 1) * (0.1 + 0.02j)
        mpa_store.write_intraband_row_collective(
            path, iq, jax.device_put(omega_host, sharding),
            jax.device_put(residue_host, sharding), mesh_xy=mesh,
            poles_finite=True, poles_causal=True,
            anomaly_counts={"folded_modes": iq})
    if process_rank() == 0:
        ledger = mpa_store.finalize_fit_store(path, certification=CERT)
        if not ledger["complete"] or not ledger["intraband_complete"]:
            _fail(f"final ledger is incomplete: {ledger}")
        omega, residues, _diagnostics, got = mpa_store.read_fit_tensors(path)
        if omega.shape != (3, N_Q, N_MU, N_MU):
            _fail(f"unexpected pole shape {omega.shape}")
        if not np.all(omega[N_FIT:].real == 0.05):
            _fail("analytic suffix pole bytes changed")
        if not np.allclose(
                residues[N_FIT, 1], 2.0 * (0.1 + 0.02j)):
            _fail("analytic suffix residue bytes changed")
        mpa_store.validate_fit_store(path)
        print(
            "[intraband_store] PASS world=4 split=2+1 rows=2 "
            f"folded_modes={got['provenance'].get('folded_modes', 'attr')}",
            flush=True)
    barrier("intraband_store_gate_done")
    return 0


if __name__ == "__main__":
    finalize_process(main())
