import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from gw.mpa import model
from gw.mpa import sigma as mpa_sigma


def _mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def test_iteration_artifacts_retain_only_the_completed_map(
        tmp_path, monkeypatch):
    root = tmp_path / "mpa"
    root.mkdir()
    keep = model.iteration_artifact_paths(root, "sc_0002")
    for path in (*keep,
                 *model.iteration_artifact_paths(root, "sc_0000"),
                 *model.iteration_artifact_paths(root, "sc_0007")):
        open(path, "wb").close()
    unrelated = root / "mpa_fit_external.h5"
    unrelated.touch()
    barriers = []
    monkeypatch.setattr("common.collectives.process_rank", lambda: 0)
    monkeypatch.setattr(
        "common.collectives.barrier",
        lambda label, print_fn=print: barriers.append(label))

    model.retain_iteration_artifacts(root, "sc_0002", print_fn=lambda *_: None)

    assert all(os.path.exists(path) for path in keep)
    assert unrelated.exists()
    assert sorted(path.name for path in root.iterdir()) == sorted(
        [os.path.basename(path) for path in keep] + [unrelated.name])
    assert barriers == ["mpa.model.retain.sc_0002"]


def test_q_wedge_returns_the_resolution_verdict(monkeypatch):
    verdict = object()
    resolution = SimpleNamespace(verdict=verdict)
    monkeypatch.setattr(
        "gw.v_q_g_flat._resolve_ibz_q_list",
        lambda **kwargs: (
            None, np.zeros((1, 3)), np.zeros(1, np.int32),
            np.zeros(1, np.int32), np.zeros((1, 1), np.int32),
            np.zeros((1, 1, 3), np.int32), True, resolution))
    sym = SimpleNamespace(
        q_irr_full_idx=np.array([0], np.int32),
        sym_matrices=np.eye(3, dtype=np.int32)[None, ...])
    q_idx, _tables, got = model._q_wedge(
        sym, np.zeros((1, 3), np.int32),
        SimpleNamespace(kgrid=(1, 1, 1), fft_grid=(1, 1, 1)))
    np.testing.assert_array_equal(q_idx, [0])
    assert got is verdict


def test_explicit_fit_reuse_asserts_every_cross_run_identity(
        tmp_path, monkeypatch):
    fit = tmp_path / "mpa_fit_oneshot.h5"
    fit.touch()
    calls = {}

    class _Tables:
        def logical(self, n_mu):
            calls["logical_n_mu"] = n_mu
            return self

        def canonical(self):
            return self

        def digest(self):
            return "table-current"

    monkeypatch.setattr(
        model, "_q_wedge",
        lambda *_args: (
            np.arange(3, dtype=np.int32), _Tables(),
            SimpleNamespace(centroid_hash="centroids-current")))

    def validate(path, **kwargs):
        calls["validate"] = (path, kwargs)
        return {
            "n_p": 8, "n_q": 3, "n_mu": 5,
            "ordered_residues": False,
        }

    monkeypatch.setattr(model.mpa_store, "validate_fit_store", validate)
    monkeypatch.setattr(
        model, "_canonical_wfn_identity",
        lambda source, binding=None: {
            "wfn_fingerprint_scheme": "mean-field-content-v1",
            "wfn_fingerprint": "wfn-current",
        })
    monkeypatch.setattr(
        model.mpa_store, "assert_occupation_stamps",
        lambda path, state, **kwargs: calls.setdefault(
            "occupation", (path, state, kwargs)))
    live_plan, stored_plan = object(), object()
    monkeypatch.setattr(
        model, "make_mpa_plan_from_fit",
        lambda *_args, **_kwargs: stored_plan)
    z = np.asarray([2.0e-5j, 1.0 + 0.2j])
    monkeypatch.setattr(model.sample_plan, "plan_z", lambda _plan: z)
    occ = SimpleNamespace(f_kn=np.ones((1, 5)), occ_hash="current")
    config = SimpleNamespace(
        mpa=SimpleNamespace(n_poles=8),
        screening=SimpleNamespace(diagrams="w_rpa"))
    meta = SimpleNamespace(n_rmu=5, b_id_4_user=5)
    sym = SimpleNamespace(trs_allowed=True)

    got = model.validate_reused_mpa_fit(
        fit, config=config, live_plan=live_plan, sym=sym,
        centroid_indices=None, meta=meta, mesh_xy=None,
        wfn=object(),
        charge_zeta_identity={"scheme": "zeta-v1", "digest": "zeta-a"},
        occupation_state=occ, material_class="metal",
        print_fn=lambda *_args: None)

    assert got == str(fit.resolve())
    _, kwargs = calls["validate"]
    assert kwargs["expected_identity"] == {
        "w_table_hash": "table-current",
        "w_centroid_hash": "centroids-current",
        "wfn_fingerprint_scheme": "mean-field-content-v1",
        "wfn_fingerprint": "wfn-current",
        "charge_zeta_identity_scheme": "zeta-v1",
        "charge_zeta_identity": "zeta-a",
    }
    assert calls["logical_n_mu"] == 5
    assert kwargs["expected_screening_diagrams"] == "w_rpa"
    assert calls["occupation"][0] == str(fit.resolve())
    from gw.efermi import occupation_digest
    assert occupation_digest(occ.f_kn, band_extent=8) in (
        calls["occupation"][2]["compatible_occ_hashes"])


def test_explicit_fit_reuse_accepts_conservative_frequency_ceiling(
        tmp_path, monkeypatch):
    """Producer padding may widen a fit grid; it may never under-cover live."""
    fit = tmp_path / "mpa_fit_oneshot.h5"
    fit.touch()

    class _Tables:
        def logical(self, _n_mu):
            return self
        def canonical(self):
            return self
        def digest(self):
            return "table"

    monkeypatch.setattr(
        model, "_q_wedge", lambda *_args: (
            np.arange(1), _Tables(), SimpleNamespace(centroid_hash="centroids")))
    monkeypatch.setattr(
        model.mpa_store, "validate_fit_store", lambda *_args, **_kwargs: {
            "n_p": 2, "n_q": 1, "n_mu": 3, "ordered_residues": False})
    monkeypatch.setattr(
        model, "_canonical_wfn_identity",
        lambda source, binding=None: {
            "wfn_fingerprint_scheme": "mean-field-content-v1",
            "wfn_fingerprint": "wfn-current",
        })
    monkeypatch.setattr(model.mpa_store, "assert_occupation_stamps",
                        lambda *_args, **_kwargs: None)
    stored_plan, live_plan = object(), object()
    monkeypatch.setattr(model, "make_mpa_plan_from_fit",
                        lambda *_args, **_kwargs: stored_plan)
    monkeypatch.setattr(
        model.sample_plan, "plan_z",
        lambda plan: (np.asarray([0.0 + 0.2j, 6.0 + 0.2j])
                      if plan is stored_plan else
                      np.asarray([0.0 + 0.2j, 5.0 + 0.2j])))
    config = SimpleNamespace(
        mpa=SimpleNamespace(n_poles=2),
        screening=SimpleNamespace(diagrams="w_rpa"))
    messages = []
    occ = SimpleNamespace(f_kn=np.ones((1, 3)), occ_hash="current")
    meta = SimpleNamespace(n_rmu=3, b_id_4_user=3)

    got = model.validate_reused_mpa_fit(
        fit, config=config, live_plan=live_plan,
        sym=SimpleNamespace(trs_allowed=True), centroid_indices=None,
        meta=meta, mesh_xy=None, wfn=object(),
        charge_zeta_identity={"scheme": "zeta-v1", "digest": "zeta-a"},
        occupation_state=occ, material_class="metal",
        print_fn=messages.append)

    assert got == str(fit.resolve())
    assert any("conservatively covers" in message for message in messages)

    monkeypatch.setattr(
        model.sample_plan, "plan_z",
        lambda plan: (np.asarray([0.0 + 0.2j, 4.0 + 0.2j])
                      if plan is stored_plan else
                      np.asarray([0.0 + 0.2j, 5.0 + 0.2j])))
    import pytest
    with pytest.raises(ValueError, match="under-covers"):
        model.validate_reused_mpa_fit(
            fit, config=config, live_plan=live_plan,
            sym=SimpleNamespace(trs_allowed=True), centroid_indices=None,
            meta=meta, mesh_xy=None, wfn=object(),
            charge_zeta_identity={"scheme": "zeta-v1", "digest": "zeta-a"},
            occupation_state=occ, material_class="metal",
            print_fn=lambda *_args: None)


def test_dyson_walk_holds_one_chi_frequency_and_writes_wc(monkeypatch):
    V = jnp.asarray(np.stack([np.eye(2), 2 * np.eye(2)]),
                    dtype=jnp.complex128)
    chi = [jnp.full((2, 2, 2), k + 1, dtype=jnp.complex128)
           for k in range(3)]
    events = []

    def read(_path, name, i, *, mesh_xy):
        assert name == model._CHI
        events.append(("read", i))
        return chi[i], {}

    def write(_path, name, i, value, *, mesh_xy, global_shape):
        assert name == model._WC
        events.append(("write", i, np.asarray(value)))

    monkeypatch.setattr(model.mpa_store, "read_w_slab_collective", read)
    monkeypatch.setattr(model.mpa_store, "write_w_slab_collective", write)

    import gw.w_isdf as w_isdf
    monkeypatch.setattr(
        w_isdf, "solve_w",
        lambda Vq, cq, *_args, **_kw: Vq + 3.0 * cq)

    model._solve_wc(
        "samples.h5", V, 3, np.array([0, 1]),
        SimpleNamespace(n_rmu=2), _mesh())

    assert [e[:2] for e in events] == [
        ("read", 0), ("write", 0),
        ("read", 1), ("write", 1),
        ("read", 2), ("write", 2),
    ]
    for i, event in enumerate(events[1::2]):
        np.testing.assert_array_equal(event[2], 3.0 * np.asarray(chi[i]))


def test_dyson_resume_skips_independently_ready_wc_slabs(monkeypatch):
    V = jnp.asarray(np.stack([np.eye(2), 2 * np.eye(2)]),
                    dtype=jnp.complex128)
    chi = [jnp.full((2, 2, 2), k + 1, dtype=jnp.complex128)
           for k in range(3)]
    events = []

    def read(_path, name, i, *, mesh_xy):
        assert name == model._CHI
        events.append(("read", i))
        return chi[i], {}

    def write(_path, name, i, value, *, mesh_xy, global_shape):
        assert name == model._WC
        events.append(("write", i))

    monkeypatch.setattr(model.mpa_store, "read_w_slab_collective", read)
    monkeypatch.setattr(model.mpa_store, "write_w_slab_collective", write)
    import gw.w_isdf as w_isdf
    monkeypatch.setattr(
        w_isdf, "solve_w",
        lambda Vq, cq, *_args, **_kw: Vq + 3.0 * cq)

    model._solve_wc(
        "samples.h5", V, 3, np.array([0, 1]),
        SimpleNamespace(n_rmu=2), _mesh(),
        wc_ready=np.asarray([False, True, False]))

    assert events == [
        ("read", 0), ("write", 0),
        ("read", 2), ("write", 2),
    ]


def test_ladder_wc_source_preserves_head_but_skips_ready_wc_writes(
        monkeypatch):
    from gw import screening_bse

    z = np.asarray([0.1j, 0.8 + 0.1j], np.complex128)
    slabs = jnp.asarray(np.stack([
        np.eye(2, dtype=np.complex128),
        2.0 * np.eye(2, dtype=np.complex128),
    ]))[:, None, :, :]
    wedge = SimpleNamespace(wc=slabs)
    head = object()
    writes = []
    monkeypatch.setattr(
        screening_bse, "prepare_ladder_restart", lambda *a, **k: None)
    monkeypatch.setattr(
        screening_bse, "_ladder_wedge", lambda *a, **k: wedge)
    monkeypatch.setattr(
        screening_bse, "_assert_wedge_matches_run", lambda *a, **k: None)
    monkeypatch.setattr(
        screening_bse, "_finalize_ladder_head", lambda *a, **k: head)
    monkeypatch.setattr(
        screening_bse, "_assert_mu_width", lambda value, *_a, **_k: value)
    monkeypatch.setattr(
        "file_io.mpa_store.write_w_slab_collective",
        lambda _path, _name, index, *_a, **_k: writes.append(index))
    source = screening_bse.make_ladder_wc_source(
        None, None, quad=None, e_ref=0.0, sym=None,
        centroid_indices=None,
        config=SimpleNamespace(input_file=""),
        meta=None, mesh_xy=None, tensors_filename="restart.h5",
        head_resolver=SimpleNamespace(wfn=None))

    got = source(
        "samples.h5", jnp.zeros((1, 2, 2), jnp.complex128), z,
        np.asarray([0]), SimpleNamespace(n_rmu=2), None,
        wc_ready=np.asarray([True, False]))

    assert got is head
    assert writes == [1]


def test_dyson_refuses_mixed_centroid_carrier_extents_before_dispatch():
    """The P36 2088-V/2070-chi failure is rejected at the public seam."""
    import pytest
    from gw.w_isdf import solve_w

    V = SimpleNamespace(shape=(65, 2088, 2088))
    chi = SimpleNamespace(shape=(65, 2070, 2070))
    with pytest.raises(ValueError, match=(
            r"same padded .*V_q\.shape=\(65, 2088, 2088\).*"
            r"chi0_q\.shape=\(65, 2070, 2070\)")):
        solve_w(V, chi, meta=None, mesh_xy=None, dyson_solver="distributed")


def test_fit_walk_consumes_wc_not_chi(monkeypatch):
    seen = {}

    def run(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return "ledger", "report"

    monkeypatch.setattr(model.fit_driver, "run_fit_driver", run)
    got = model._fit_body(
        "samples.h5", "poles.h5", np.array([0.0j, 1.0j]), 1, None,
        _mesh())
    assert got == ("ledger", "report")
    assert seen["args"][:3] == (
        "samples.h5", model._WC, "poles.h5")
    assert seen["kwargs"]["mesh_xy"] == _mesh()


class _FakeReader:
    """Stands in for ``mpa_store.PoleReader``, counting its own lifetime."""

    def __init__(self, log):
        self.log = log
        self.closed = False

    def read(self, pole_slice, **kwargs):
        assert not self.closed, "read after close"
        self.log.append((pole_slice.start, pole_slice.stop, kwargs))
        n = pole_slice.stop - pole_slice.start
        pair = (jnp.zeros((n, 1, 1, 1), jnp.complex128),
                jnp.zeros((n, 1, 1, 1), jnp.complex128))
        return (*pair, None) if kwargs.get("include_odd") else pair

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


def test_sigma_store_reads_contiguous_four_pole_ranges(monkeypatch):
    """Contiguous four-pole ranges, and ONE reader for the whole walk.

    The range arithmetic is the original claim: no complete pole axis
    exists on host or device, so the executor walks ``pole_batch_size``
    at a time.  The reader COUNT is the audit-A1 claim added beside it —
    the store used to be opened and closed once per batch, through two
    different HDF5 library instances, and now one handle serves them all.
    """
    reads = []
    opened = []

    def consume(_wfns, batches, n_poles, *_args, **_kwargs):
        got = [(lo, int(O.shape[0]), int(B.shape[0]))
               for lo, O, B, _D in batches]
        assert n_poles == 10
        return got

    def open_reader(src, *, mesh_xy, **kwargs):
        opened.append(src)
        return _FakeReader(reads)

    monkeypatch.setattr(mpa_sigma, "open_pole_reader", open_reader)
    monkeypatch.setattr(mpa_sigma, "_integrate_sigma_batches", consume)
    got = mpa_sigma.integrate_sigma_store(
        None, "poles.h5", 10, (), np.array([0.0]), None, _mesh(),
        pole_batch_size=4)
    assert got == [(0, 4, 4), (4, 4, 4), (8, 2, 2)]
    assert [(lo, hi) for lo, hi, _ in reads] == [(0, 4), (4, 8), (8, 10)]
    assert all(row[2]["unfold"] and row[2]["return_sharded"]
               for row in reads)
    assert opened == ["poles.h5"], (
        f"one reader for the whole walk, not one per batch; got {opened}")


def test_sigma_store_reuses_a_reader_the_caller_already_opened(monkeypatch):
    """A live reader passed in is USED, not reopened.

    This is how ``compute_sigma_c_mpa_omega_grid`` shares one collective
    handle between its census walk and this executor walk — the whole
    Σ stage of an iteration on one open file.  If this path reopened, the
    census and the executor would be two handles again and audit A1's
    churn cut would be half done while looking whole.
    """
    reads = []

    def consume(_wfns, batches, *_args, **_kwargs):
        return [lo for lo, _O, _B, _D in batches]

    def open_reader(*_a, **_k):
        raise AssertionError("a live reader must not be reopened")

    monkeypatch.setattr(mpa_sigma, "open_pole_reader", open_reader)
    monkeypatch.setattr(mpa_sigma, "_integrate_sigma_batches", consume)
    reader = _FakeReader(reads)
    monkeypatch.setattr(mpa_sigma, "PoleReader", _FakeReader)
    got = mpa_sigma.integrate_sigma_store(
        None, reader, 6, (), np.array([0.0]), None, _mesh(),
        pole_batch_size=4)
    assert got == [0, 4]
    assert not reader.closed, (
        "the caller owns the handle it passed in; the executor must not "
        "close it out from under the census that is still using it")


def test_sigma_store_accepts_eight_and_refuses_more_resident_poles(monkeypatch):
    reads = []

    monkeypatch.setattr(
        mpa_sigma, "open_pole_reader", lambda *_a, **_k: _FakeReader(reads))
    monkeypatch.setattr(
        mpa_sigma, "_integrate_sigma_batches",
        lambda _w, batches, *_a, **_k: [
            int(O.shape[0]) for _, O, _B, _D in batches])
    got = mpa_sigma.integrate_sigma_store(
        None, "poles.h5", 8, (), np.array([0.0]), None, _mesh(),
        pole_batch_size=8)
    assert got == [8]

    for size in (0, 9):
        try:
            mpa_sigma.integrate_sigma_store(
                None, "poles.h5", 8, (), np.array([0.0]), None, _mesh(),
                pole_batch_size=size)
        except ValueError as exc:
            assert "[1, 8]" in str(exc)
        else:
            raise AssertionError(f"pole_batch_size={size} was accepted")
