"""Regression for the product-padded centroid carrier at the W/Dyson seam.

The production failure needed a 6x6 mesh: 2070 is divisible by either mesh
side, so a per-axis SlabIO round-up leaves it unchanged, while the canonical
product-padded carrier is 2088.  The synthetic geometry below uses the same
hostile arithmetic at small size (42 -> 72) and executes the production MPA
full-slab reader followed by the distributed Dyson A-build.

The 36-device mesh is CPU-emulated in a fresh process because JAX fixes its
device count at import.  The distributed LU provider is replaced only after
the A-build by a capture plan returning A.  Thus this cell covers the reader,
canonical extent, exact-zero pad, sharding, and ``_a_local`` contraction, but
not ScaLAPACK/cuSOLVERMp numerics; the service and real-process gates own that
separate contract.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


_CHILD = "LORRAX_TEST_W_DYSON_PAD_6X6_CHILD"


def _run_6x6_child() -> None:
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import distrib_la
    from file_io import mpa_store
    import file_io.slab_io as slab_io
    from runtime.padding import padded_mu_extent

    devices = np.asarray(jax.devices("cpu"))
    assert devices.size == 36, devices.size
    mesh = Mesh(devices.reshape(6, 6), ("x", "y"))
    n_q = 1
    n_logical = 42
    n_padded = int(padded_mu_extent(n_logical, mesh))

    # Anti-tautology: this is precisely the arithmetic the Bi failure needs.
    assert n_logical % 6 == 0
    assert n_logical % 36 != 0
    assert n_padded == 72

    chi_logical = np.zeros((n_q, n_logical, n_logical), np.complex128)
    diagonal = np.arange(n_logical)
    chi_logical[:, diagonal, diagonal] = 0.1
    header = {
        "n_omega": 1,
        "n_q_on_disk": n_q,
        "n_mu": n_logical,
        "data_ready": np.asarray([True]),
    }
    mpa_store.read_w_header = lambda *_args, **_kwargs: header

    class _MemorySlabIO:
        """Return the logical payload in the production requested carrier."""

        def __init__(self, _path, *, mode, mesh):
            assert mode == "r"
            assert mesh is not None
            self.mesh = mesh

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read_slab(self, _name, *, shape, offset, valid_shape,
                      partition_spec):
            assert shape == (1, n_q, n_padded, n_padded)
            assert offset == (0, 0, 0, 0)
            assert valid_shape == (1, n_q, n_logical, n_logical)
            assert partition_spec == P(None, None, "x", "y")
            carrier = np.zeros(shape, np.complex128)
            carrier[0, :, :n_logical, :n_logical] = chi_logical
            return jax.device_put(
                carrier, NamedSharding(self.mesh, partition_spec))

    slab_io.SlabIO = _MemorySlabIO
    chi_q, _ = mpa_store.read_w_slab_collective(
        "synthetic_samples.h5", "chi0_qmunu_z", 0, mesh_xy=mesh)
    assert tuple(chi_q.shape) == (n_q, n_padded, n_padded)
    chi_host = np.asarray(chi_q).copy()
    assert np.count_nonzero(chi_host[:, n_logical:, :]) == 0
    assert np.count_nonzero(chi_host[:, :, n_logical:]) == 0

    V_host = np.zeros((n_q, n_padded, n_padded), np.complex128)
    V_host[:, diagonal, diagonal] = 2.0
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    # Poison both pad zones after proving the reader itself returned zeros.
    # The W owner must make exact-zero padding structural before _a_local.
    V_host[:, n_logical:, :] = 7.0
    V_host[:, :, n_logical:] = 7.0
    chi_host[:, n_logical:, :] = 11.0
    chi_host[:, :, n_logical:] = 11.0
    V_q = jax.device_put(V_host, sharding)
    chi_q = jax.device_put(chi_host, sharding)

    requested = {}
    captured = {}

    class _CaptureAPlan:
        def describe(self):
            return "test capture after distributed Dyson A-build"

        def batched(self, A, B):
            captured["B"] = B
            return A

    def _capture_plan(op, plan_mesh, *, backend, n, batched_route):
        requested.update(
            op=op, mesh=plan_mesh, backend=backend, n=n,
            batched_route=batched_route)
        return _CaptureAPlan()

    distrib_la.plan = _capture_plan
    from gw.w_isdf import solve_w

    meta = SimpleNamespace(
        nk_tot=n_q, nspin=1, nspinor_wfnfile=1, n_rmu=n_logical)
    A_q = solve_w(
        V_q, chi_q, meta, mesh, dyson_solver="distributed",
        distrib_la_batched_route="batch_reshard")
    A_host = np.asarray(jax.block_until_ready(A_q))
    B_host = np.asarray(jax.block_until_ready(captured["B"]))

    assert requested == {
        "op": "solve_lu",
        "mesh": mesh,
        "backend": "distributed",
        "n": n_padded,
        "batched_route": "batch_reshard",
    }
    assert A_q.sharding.spec == P(None, "x", "y")
    expected = np.eye(n_padded, dtype=np.complex128)[None]
    expected[:, diagonal, diagonal] = 0.6  # I - V * (pref=2) * chi
    np.testing.assert_allclose(A_host, expected, rtol=0.0, atol=2e-15)
    assert np.count_nonzero(B_host[:, n_logical:, :]) == 0
    assert np.count_nonzero(B_host[:, :, n_logical:]) == 0


def test_solve_w_refuses_equal_but_noncanonical_centroid_extents():
    """Equality alone must not authenticate the weak P36 2070 carrier."""
    from types import SimpleNamespace

    import pytest

    from gw.w_isdf import solve_w

    operand = SimpleNamespace(shape=(65, 2070, 2070))
    meta = SimpleNamespace(nk_tot=65, n_rmu=2070)
    mesh = SimpleNamespace(axis_names=("x", "y"),
                           shape={"x": 6, "y": 6})
    with pytest.raises(ValueError, match=(
            r"canonical product-padded centroid carrier "
            r"\(\*,2088,2088\).*got equal .*\(65, 2070, 2070\)")):
        solve_w(operand, operand, meta, mesh, dyson_solver="distributed")


def test_solve_w_geometry_accepts_irreducible_q_extent():
    """The Dyson seam owns mu padding, not full-BZ versus wedge q mapping."""
    from types import SimpleNamespace

    from gw.w_isdf import _require_w_operand_geometry

    operand = SimpleNamespace(shape=(65, 2088, 2088))
    meta = SimpleNamespace(nk_tot=512, n_rmu=2070)
    mesh = SimpleNamespace(axis_names=("x", "y"),
                           shape={"x": 6, "y": 6})
    assert _require_w_operand_geometry(operand, operand, meta, mesh) == 2070


def test_solve_w_geometry_accepts_explicit_packed_carrier():
    """A direct-sum photon carrier is not the scalar charge prefix."""
    from types import SimpleNamespace

    from gw.w_isdf import _require_w_operand_geometry

    operand = SimpleNamespace(shape=(4, 32, 32))
    meta = SimpleNamespace(nk_tot=4, n_rmu=6)
    mesh = SimpleNamespace(axis_names=("x", "y"),
                           shape={"x": 2, "y": 2})
    assert _require_w_operand_geometry(
        operand, operand, meta, mesh, n_rmu_logical=32) == 32


def test_precompile_solve_w_refuses_the_same_noncanonical_carrier(
        monkeypatch):
    """AOT setup must not compile a geometry the runtime seam rejects."""
    from types import SimpleNamespace

    import gw.w_isdf as w_isdf
    import pytest

    monkeypatch.setattr(w_isdf, "ensure_jax_compile_cache", lambda: None)
    operand = SimpleNamespace(shape=(65, 2070, 2070))
    meta = SimpleNamespace(nk_tot=512, n_rmu=2070)
    mesh = SimpleNamespace(axis_names=("x", "y"),
                           shape={"x": 6, "y": 6})
    with pytest.raises(ValueError, match=(
            r"canonical product-padded centroid carrier "
            r"\(\*,2088,2088\).*got equal .*\(65, 2070, 2070\)")):
        w_isdf.precompile_solve_w(
            operand, operand, meta, mesh, dyson_solver="distributed")


def test_distributed_solve_cache_is_keyed_by_logical_extent(monkeypatch):
    """One padded extent must not reuse another caller's zero-mask closure."""
    from types import SimpleNamespace

    import jax
    import numpy as np
    from jax.sharding import Mesh

    import distrib_la
    import gw.w_isdf as w_isdf

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    monkeypatch.setattr(
        distrib_la, "plan",
        lambda *_args, **_kwargs: SimpleNamespace(describe=lambda: "capture"))
    w_isdf._w_solve_cache.clear()
    scalar = w_isdf._get_w_solve_fn_distributed(
        mesh, nq=1, n_rmu=8, n_rmu_logical=6)
    packed = w_isdf._get_w_solve_fn_distributed(
        mesh, nq=1, n_rmu=8, n_rmu_logical=8)
    assert scalar is not packed


def test_mpa_reader_and_distributed_dyson_a_build_on_true_6x6_mesh():
    """Logical 42 must reach ``_a_local`` as one zero-padded 72 carrier."""
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({
        _CHILD: "1",
        "JAX_ENABLE_X64": "1",
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=36",
    })
    service_paths = [
        repo / "src",
        repo / "services" / "distrib_la" / "src",
        repo / "services" / "lxkit" / "src",
        repo / "services" / "minimax" / "src",
        repo / "services" / "symmetry_maps" / "src",
        repo / "services" / "wfn_loader" / "src",
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in service_paths), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=repo, env=env, capture_output=True, text=True, timeout=180,
        check=False)
    assert proc.returncode == 0, (
        f"6x6 child failed with rc={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")


if __name__ == "__main__" and os.environ.get(_CHILD):
    _run_6x6_child()
