"""Early exchange-storage refusal and conduction-cache ownership/padding."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import runtime  # Install BLAS policy before numerical libraries in the subprocess.
import h5py
import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src/bse/exciton_bands.py"


def _owners(names, namespace):
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def _zeta(path, rows, *, grid=(3, 3, 1), complete=True):
    with h5py.File(path, "w") as f:
        f["mf_header/kpoints/kgrid"] = grid
        f["isdf_header/zeta_layout"] = np.bytes_("G_flat")
        f["isdf_header/zeta_is_done"] = complete
        # Large sparse datasets: a count preflight must never read these.
        f.create_dataset("zeta_q_G", (rows, 10000, 10000), dtype="c16", chunks=True)
        f.create_dataset("isdf_header/gvec_components", (rows, 3, 10000), dtype="i4")


def test_exchange_storage_preflight_reads_counts_and_precedes_wavefunctions(tmp_path, monkeypatch):
    owner = _owners({"require_zeta_for_interp"}, dict(os=os, np=np))["require_zeta_for_interp"]
    restart = tmp_path / "isdf_tensors.h5"
    zeta = tmp_path / "zeta_q.h5"
    _zeta(zeta, 3)
    read = h5py.Dataset.__getitem__
    seen = []

    def scalar_metadata(dataset, key):
        seen.append(dataset.name)
        assert dataset.name in ("/mf_header/kpoints/kgrid", "/isdf_header/zeta_layout",
                                "/isdf_header/zeta_is_done")
        return read(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", scalar_metadata)
    assert owner(restart, "ongrid", (3, 3, 1)) == "" and not seen
    assert owner(restart, "refit", (3, 3, 1)) == str(zeta)
    for mode in ("interp", "both"):
        with pytest.raises(SystemExit, match=r"3 of 9.*--vq-mode=refit"):
            owner(restart, mode, (3, 3, 1))
    with pytest.raises(SystemExit, match="coarse WFN grid"):
        owner(restart, "refit", (9, 1, 1))
    _zeta(zeta, 9)
    assert owner(restart, "interp", (3, 3, 1)) == str(zeta)
    _zeta(zeta, 3, complete=False)
    with pytest.raises(SystemExit, match="incomplete"):
        owner(restart, "refit", (3, 3, 1))
    _zeta(zeta, 10)
    with pytest.raises(SystemExit, match="10 rows for a 9-q grid"):
        owner(restart, "refit", (3, 3, 1))
    # The refusal must run before either fine-grid BSE loading or path Φ.
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = {name: [n.lineno for n in ast.walk(main)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == name]
             for name in ("require_zeta_for_interp", "load_bse_data_from_restart_sharded",
                          "compute_wfns_fi")}
    assert len(calls["require_zeta_for_interp"]) == 1
    assert calls["require_zeta_for_interp"][0] < min(
        calls["load_bse_data_from_restart_sharded"] + calls["compute_wfns_fi"])


def _compiled_stack_controls():
    from functools import partial
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.sharding_fit import fit_sharding

    jax.config.update("jax_enable_x64", True)
    namespace = _owners({"_conduction_stack_program", "build_conduction_stacks"},
                        dict(jax=jax, jnp=jnp, partial=partial,
                             NamedSharding=NamedSharding, P=P, PAD_EPS_GUARD_RY=1000.))
    records = []
    for p in (1, 2):
        mesh = Mesh(np.array(jax.devices("cpu")[:p*p]).reshape(p, p), ("x", "y"))
        rep = NamedSharding(mesh, P())
        for nc, ns, mu, ncp, mup in ((4, 2, 8, 4, 8), (3, 2, 7, 4, 8),
                                     (1, 1, 8, 2, 8), (4, 2, 7, 4, 8)):
            shape = (3, 4, nc, ns, mu)
            values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
            values = values + 1j*(values[::-1] + .25)
            energies = np.arange(3*4*nc, dtype=np.float64).reshape(12, nc)
            yspec = fit_sharding(mesh, P(None, None, None, "y"), (12, nc, ns, mu))
            # The consuming owner receives device-owned buffers. CPU device_put
            # may otherwise borrow aligned NumPy storage and prevent donation.
            y = jax.device_put(values.reshape(12, nc, ns, mu), yspec).copy()
            eps = jax.device_put(energies, rep).copy()
            # P1 explicitly exercises X/Y sharing one underlying allocation.
            tx = y if p == 1 else jax.device_put(values.reshape(12, nc, ns, mu).transpose(0, 3, 1, 2),
                   fit_sharding(mesh, P(None, "x", None, None), (12, mu, nc, ns)))
            bundle = SimpleNamespace(psi_rmu_Y=y, psi_rmuT_X=tx, enk_full=eps)
            program = namespace["_conduction_stack_program"](mesh, shape, (ncp, mup))
            memory = program.lower(y, eps).compile().memory_analysis()
            legacy = jax.jit(program.__wrapped__, out_shardings=(
                NamedSharding(mesh, P(None, None, None, None, "x")),
                NamedSharding(mesh, P(None, None, None, None, "y")), rep))
            old_memory = legacy.lower(y, eps).compile().memory_analysis()
            # Previously the unused X layout stayed live during this program.
            old_x_bytes = tx.addressable_shards[0].data.nbytes if tx is not y else 0
            price = lambda m: (m.argument_size_in_bytes + m.output_size_in_bytes
                               + m.temp_size_in_bytes - m.alias_size_in_bytes)
            old_price = price(old_memory) + old_x_bytes
            new_price = price(memory)
            assert new_price <= old_price
            if old_x_bytes or (nc, mu) == (ncp, mup):
                assert new_price < old_price
            pointers = {s.data.unsafe_buffer_pointer() for s in y.addressable_shards}
            actual = namespace["build_conduction_stacks"](bundle, 3, 4, nc, ncp, mu, mup, mesh)
            jax.block_until_ready(actual)
            expected = np.pad(values, ((0, 0), (0, 0), (0, ncp-nc), (0, 0), (0, mup-mu)))
            expected_eps = np.pad(energies.reshape(3, 4, nc),
                                  ((0, 0), (0, 0), (0, ncp-nc)), constant_values=1000.)
            np.testing.assert_array_equal(actual[0], expected)
            np.testing.assert_array_equal(actual[1], expected)
            np.testing.assert_array_equal(actual[2], expected_eps)
            assert bundle.psi_rmu_Y is bundle.psi_rmuT_X is bundle.enk_full is None
            if (nc, mu) == (ncp, mup):
                assert memory.alias_size_in_bytes >= memory.argument_size_in_bytes
                output_pointers = {s.data.unsafe_buffer_pointer() for a in actual[:2]
                                   for s in a.addressable_shards}
                assert pointers <= output_pointers
                assert y.is_deleted() and eps.is_deleted()
            records.append(dict(mesh=p*p, shape=shape, padded=(ncp, mup),
                                baseline_program_and_live_X_bytes=old_price,
                                candidate_program_bytes=new_price,
                                memory={key: int(getattr(memory, key)) for key in
                                        ("argument_size_in_bytes", "output_size_in_bytes",
                                         "temp_size_in_bytes", "alias_size_in_bytes")}))
        # A padded htransform result names its physical centroid count.
        # Preserve that authority while consuming its already-legal carrier.
        logical, carried = 13, (13 if p == 1 else 16)
        values = np.arange(3*3*2*carried).reshape(3, 3, 2, carried).astype(complex)
        values[..., logical:] = 0
        y = jax.device_put(values, fit_sharding(
            mesh, P(None, None, None, "y"), values.shape))
        energies = jax.device_put(np.arange(9.).reshape(3, 3), rep)
        bundle = SimpleNamespace(psi_rmu_Y=y, psi_rmuT_X=None, enk_full=energies,
                                 n_mu_logical=logical, n_mu_carrier=carried)
        actual = namespace["build_conduction_stacks"](
            bundle, 1, 3, 3, 4, logical, carried, mesh)
        wanted = np.pad(values[None], ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0)))
        np.testing.assert_array_equal(actual[0], wanted)
        np.testing.assert_array_equal(actual[1], wanted)
        # A wrong basis must refuse before taking ownership of the buffers.
        bundle = SimpleNamespace(psi_rmu_Y=np.zeros((12, 4, 2, 7)),
                                 psi_rmuT_X=object(), enk_full=np.zeros((12, 4)))
        with pytest.raises(ValueError, match="restart bundle stores mu=8"):
            namespace["build_conduction_stacks"](bundle, 3, 4, 4, 4, 8, 8, mesh)
        assert bundle.psi_rmu_Y is not None and bundle.psi_rmuT_X is not None
    print(json.dumps(records))


def test_conduction_stacks_cpu_p1_p4_padding_alias_and_compiled_memory():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
               XLA_FLAGS="--xla_force_host_platform_device_count=4", OPENBLAS_NUM_THREADS="1",
               JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stack-controls"],
                            env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"mesh": 4' in result.stdout


if __name__ == "__main__" and "--stack-controls" in sys.argv:
    _compiled_stack_controls()
