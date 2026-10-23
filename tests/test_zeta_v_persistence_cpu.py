"""The persisted charge ζ must be the same operand that formed V."""
import ast
from contextlib import nullcontext
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _fixture(monkeypatch, tmp_path, *, complete=True, fail_write=False):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from isdf import zeta_mubatch as zmb
    from runtime.padding import padded_axis
    from common import collectives, gpu_utils
    from file_io import isdf_header, slab_io
    from lxkit import memory

    jax.config.update("jax_enable_x64", True)
    device = jax.devices("cpu")[0]
    mesh = Mesh(np.array([[device]]), ("x", "y"))
    rng = np.random.default_rng(93)
    raw = rng.normal(size=(2, 8, 7)) + 1j*rng.normal(size=(2, 8, 7))
    # Ragged q spheres and a partial final G tile must both be inert.
    ngk = np.array([5, 7], np.int32)
    raw[0, :, 5:] = 0
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j*rng.normal(size=(8, 8)))
    factor = np.stack([q * np.geomspace(1., 1.e-4, 8)[None, :]]*2)
    v = rng.uniform(.2, 1.3, (2, 7))
    stored = np.zeros_like(raw)
    events, reads = [], []

    class Writer:
        def __init__(self, path, *, mode, mesh):
            assert mode == "a"
        def __enter__(self):
            events.append("open")
            return self
        def write_slab(self, name, value, *, offset):
            assert name == "zeta_q_G"
            if fail_write:
                raise OSError("synthetic failed tile write")
            start = offset[2]
            count = min(value.shape[2], stored.shape[2]-start)
            stored[:, :, start:start+count] = np.asarray(value)[:, :, :count]
            events.append("write")
        def __exit__(self, *_args):
            events.append("closed")

    def stamp(_path, provenance=None):
        assert "closed" in events
        events.append("done" if provenance is None else "provenance")

    monkeypatch.setattr(slab_io, "SlabIO", Writer)
    monkeypatch.setattr(isdf_header, "mark_zeta_done", stamp)
    monkeypatch.setattr(isdf_header, "stamp_fit_provenance", stamp)
    monkeypatch.setattr(collectives, "barrier", lambda name: events.append(name))
    monkeypatch.setattr(collectives, "process_rank", lambda: 0)
    monkeypatch.setattr(gpu_utils, "shared_device_room", lambda _where: 1 << 30)
    monkeypatch.setattr(memory, "book", lambda _price: None)
    zmb._FACE_CHAINS.clear()
    zmb._kernel_cache.clear()
    source = np.pad(raw, ((0, 0), (0, 0), (0, 1)))

    def read_tile(t):
        reads.append(t)
        return jax.device_put(source[:, :, t*4:(t+1)*4],
                             NamedSharding(mesh, P(("x", "y"), None, None)))

    store = SimpleNamespace(Q=2, Q_pad=2, mu_pad=8, g_tile=4, n_Gt=2,
        g_axis=padded_axis(7, 4, name="fixture G"), read_tile=read_tile,
        placement="host", t_read=0.)
    zeta = zmb.ZetaG(store, mesh=mesh,
        L_q=jax.device_put(factor, NamedSharding(mesh, P(None, "x", "y"))),
        lu_piv=None, solver_kind="rank_truncate", batched_route="auto",
        n_rmu_solve=8, n_rmu=8, mu_basis=None, ngk_per_q=ngk,
        gvec_components=np.zeros((2, 3, 7), np.int32), path=tmp_path/"zeta.h5")
    zeta.pending_file = {"complete": complete, "provenance": '{"fixture":93}'}
    return zeta, v, raw, factor, stored, events, reads


def test_cold_v_and_saved_zeta_reuse_match_literal_gram(monkeypatch, tmp_path):
    zeta, v, raw, factor, saved, events, reads = _fixture(monkeypatch, tmp_path)
    cold = np.asarray(zeta.contract_v(v, keep=np.zeros((2, 1), np.int32)))
    expected_zeta = factor @ (factor.conj().swapaxes(-1, -2) @ raw)
    np.testing.assert_allclose(saved, expected_zeta, atol=2e-15, rtol=2e-12)
    reference = np.stack([(row.conj()*weight[None, :]) @ row.T
                          for row, weight in zip(saved, v)])
    np.testing.assert_allclose(cold, reference, atol=2e-14, rtol=2e-12)
    # A reuse consumer has only the file, not the fit factor or live Z store.
    reuse = np.einsum("qmg,qg,qng->qmn", saved.conj(), v, saved)
    np.testing.assert_allclose(cold, reuse, atol=2e-14, rtol=2e-12)
    assert reads == [0, 1] and events.count("write") == 2
    assert events.index("closed") < events.index("provenance") < events.index("done")
    assert zeta.pending_file is None


@pytest.mark.parametrize("complete,fail_write", [(False, False), (True, True)])
def test_partial_or_failed_v_pass_cannot_authenticate_file(
        monkeypatch, tmp_path, complete, fail_write):
    zeta, v, _raw, _factor, _saved, events, _reads = _fixture(
        monkeypatch, tmp_path, complete=complete, fail_write=fail_write)
    if fail_write:
        with pytest.raises(OSError, match="failed tile write"):
            zeta.contract_v(v, keep=np.zeros((2, 1), np.int32))
        assert zeta.pending_file is not None
    else:
        zeta.contract_v(v, keep=np.zeros((2, 1), np.int32)).block_until_ready()
    assert "closed" in events and "done" not in events and "provenance" not in events


@pytest.mark.parametrize("failed_phase", ["provenance", "done"])
def test_completion_error_uses_collective_io_verdict(monkeypatch, tmp_path, failed_phase):
    import h5py
    from common import collectives
    from file_io import isdf_header
    mark_done, stamp = isdf_header.mark_zeta_done, isdf_header.stamp_fit_provenance
    zeta, v, _raw, _factor, _saved, events, _reads = _fixture(monkeypatch, tmp_path)
    with h5py.File(zeta.path, "w") as h5:
        h5.create_group("isdf_header").create_dataset("zeta_is_done", data=False)
    verdicts = []
    def fail_metadata(*_args):
        raise OSError("synthetic metadata failure")
    def agree(error, *, path, stage):
        verdicts.append((error, stage))
        if error is not None:
            raise RuntimeError("agreed metadata refusal") from error
    monkeypatch.setattr(isdf_header, "mark_zeta_done",
                        fail_metadata if failed_phase == "done" else mark_done)
    monkeypatch.setattr(isdf_header, "stamp_fit_provenance",
                        fail_metadata if failed_phase == "provenance" else stamp)
    monkeypatch.setattr(collectives, "agree_io_error", agree)
    with pytest.raises(RuntimeError, match="agreed metadata refusal"):
        zeta.contract_v(v, keep=np.zeros((2, 1), np.int32))
    assert "closed" in events
    with h5py.File(zeta.path, "r") as h5:
        assert not bool(h5["isdf_header/zeta_is_done"][()])
        assert ("fit_provenance" in h5["isdf_header"]) == (failed_phase == "done")
    assert isinstance(verdicts[0][0], OSError) and verdicts[0][1] == "zeta_v_provenance"
    assert zeta.pending_file is not None


@pytest.mark.parametrize("truncated", [False, True])
def test_charge_caller_defers_payload_and_provenance(monkeypatch, truncated):
    from gw import isdf_fitting
    from file_io import isdf_header
    zeta = SimpleNamespace(pending_file={"complete": not truncated, "provenance": None})
    calls = []
    def fit(**kwargs):
        calls.append(kwargs)
        return 0, {0: zeta}
    def premature_stamp(*_args):
        raise AssertionError("a deferred file cannot be authenticated before V")
    monkeypatch.setattr(isdf_fitting, "fit_zeta_to_h5", fit)
    monkeypatch.setattr(isdf_header, "stamp_fit_provenance", premature_stamp)
    source = Path(__file__).parents[1]/"src/gw/gw_init.py"
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "_fit_charge_zeta_channel")
    namespace = dict(jax=SimpleNamespace(process_index=lambda: 0),
        timing=SimpleNamespace(section=lambda *_a: nullcontext()),
        jax_profile=SimpleNamespace(trace_section=lambda *_a: nullcontext()),
        log=SimpleNamespace(info=lambda *_a: None),
        _gate_fresh_zeta_rank_findings=lambda *_a, **_k: None,
        active_zeta_truncating_knobs=lambda: [("LORRAX_MAX_RCHUNKS", "1")] if truncated else [],
        barrier=lambda *_a: None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    caller = namespace[node.name]
    arguments = {name: None for name in inspect.signature(caller).parameters}
    arguments.update(_provenance='{"fixture":93}', _reuse_charge=False,
        _write_ibz_only_charge=True, _zeta_cutoff=25., chunks={"mubatch": object()},
        cfg=SimpleNamespace(write_restart_tensors=True, backend=SimpleNamespace(
            distrib_la_batched_route="auto", zeta_rcond=1e-8)),
        meta=SimpleNamespace(nspinor=2), representation=SimpleNamespace(charge_lift=None))
    assert caller(**arguments)[2] is zeta
    assert calls[0]["defer_zeta_write"] is True and calls[0]["write_zeta_file"] is True
    assert zeta.pending_file["provenance"] == (None if truncated else '{"fixture":93}')
