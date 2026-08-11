"""The multipole lane's restart seam: which stage a bundle lets a run skip.

CPU only, ``JAX_PLATFORMS=cpu``; nothing here touches a cluster.

WHAT THIS SUITE PROVES.  ``gw.mpa.w_restart`` is the one place the MPA
driver decides whether to fit or to integrate, and the claim it makes
is that resuming changes NOTHING about the numbers — the pole field a
run gets by fitting a resumed W(omega) store is the field it would
have got by fitting the same store directly, byte for byte, and the
field a second run gets by skipping the fit is the first run's own
bytes.  That is asserted as bit-identity on the tensors the Sigma pass
consumes, never as a tolerance: a restart that changed the last digit
of ``Omega_p`` would be a restart that is not a restart.

AND THE REFUSALS ARE THE OTHER HALF, because a resume that cannot be
wrong is a resume nobody can trust.  Four red twins, each of which is
a file that LOOKS complete: a store swept from a different WFN, a
store with one frequency slab allocated and unwritten, a store holding
``W`` instead of ``W_c``, and a store that declares neither.  Every one
of them is refused by a check the format already owned — this suite
asserts they are REACHED from the seam, not that they exist.

The geometry, the planted pole field and the W(omega) writer are
``test_mpa_fit_driver``'s, imported rather than rebuilt: two copies of
an orbit-closed synthetic centroid set is two claims about what closure
means.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("symmetry_maps.qirr_store")
jax = pytest.importorskip("jax")

from file_io import mpa_store as MS                               # noqa: E402
from gw.mpa import fit_driver, w_restart                          # noqa: E402

from test_mpa_fit_driver import (_N_P, _N_Q_IBZ, _SAMPLING,       # noqa: E402
                                 _W_NAME, _geometry,
                                 _planted_field, _protocol_grid,
                                 _synthesize_w)


# ---------------------------------------------------------------------------
# Fixtures: one WFN stand-in, one bundle holding only W(omega)
# ---------------------------------------------------------------------------

class _Wfn:
    """The one attribute :func:`w_restart.wfn_identity` reads.

    ``wfn._filename`` is the same source path ``fit_zeta_to_h5`` copies
    ``mf_header`` from and the same one ``_zeta_fit_provenance``
    records, so the stand-in is the real contract and not a mock of it.
    """

    def __init__(self, path):
        self._filename = str(path)


@pytest.fixture(scope="module")
def wfn_a(tmp_path_factory):
    p = tmp_path_factory.mktemp("wfn_a") / "WFN.h5"
    p.write_bytes(b"\x00" * 4096)
    return _Wfn(p)


@pytest.fixture(scope="module")
def wfn_b(tmp_path_factory):
    """A DIFFERENT WFN of a DIFFERENT size — the stale-store twin."""
    p = tmp_path_factory.mktemp("wfn_b") / "WFN.h5"
    p.write_bytes(b"\x01" * 8192)
    return _Wfn(p)


def _write_bundle(path, wfn, *, n_p=_N_P, screening_content="W_c",
                  ready_all=True):
    """A bundle holding ONLY a W(omega) tensor, written through the seam.

    ``ready_all=False`` leaves the last slab allocated and unwritten,
    which is the state a preempted sweep leaves behind and the state
    the readiness refusal exists for.  It is built by the writer's own
    ``ready=`` flag rather than by hand-forging attrs.
    """
    tables, verdict, n_mu = _geometry()
    z = _protocol_grid(n_p)
    Omega, B = _planted_field(n_p, _N_Q_IBZ, n_mu)
    W = _synthesize_w(Omega, B, z)
    line = np.array([0] * n_p + [1] * n_p, dtype=np.int32)
    if ready_all:
        w_restart.write_w_omega(
            str(path), W, tables=tables, omega=z,
            sampling=dict(_SAMPLING, n_p=n_p), omega_line=line,
            closure_verdict=verdict,
            identity=w_restart.wfn_identity(wfn),
            screening_content=screening_content)
    else:
        MS.allocate_w_omega(
            str(path), _W_NAME, n_omega=2 * n_p, n_q_on_disk=_N_Q_IBZ,
            n_mu=n_mu, tables=tables, omega=z,
            sampling=dict(_SAMPLING, n_p=n_p), omega_line=line,
            closure_verdict=verdict,
            screening_content=screening_content,
            provenance=w_restart.wfn_identity(wfn))
        for i in range(2 * n_p):
            MS.write_w_slab(str(path), _W_NAME, i, W[i],
                            ready=(i < 2 * n_p - 1))
    return {"z": z, "Omega": Omega, "B": B, "n_mu": n_mu, "n_p": n_p}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, wfn_a):
    """A bundle with a complete W(omega) and no poles yet."""
    root = tmp_path_factory.mktemp("mpa_restart")
    path = root / "mpa_bundle.h5"
    field = _write_bundle(path, wfn_a)
    field["root"] = root
    field["path"] = path
    return field


# ---------------------------------------------------------------------------
# (a) THE STAGE LADDER
# ---------------------------------------------------------------------------

def test_a_bundle_with_only_w_omega_asks_for_the_fit(bundle, wfn_a):
    """W(omega) present, poles absent -> STAGE_FIT, and it says so."""
    d = w_restart.resolve_mpa_restart(
        str(bundle["path"]), identity=w_restart.wfn_identity(wfn_a))
    assert d.stage == w_restart.STAGE_FIT
    assert d.w_header is not None and d.fit_ledger is None
    line = w_restart.describe(d)
    assert "RESUMING from its W(omega)" in line
    assert "SKIPPED" in line
    # THE ANNOUNCEMENT CARRIES THE PROVENANCE, which is the whole point
    # of announcing: a skipped stage is invisible in every other way.
    assert str(wfn_a._filename) in line
    assert "W_c" in line and "grid_hash=" in line


def test_a_missing_bundle_is_build_and_not_a_traceback(tmp_path):
    d = w_restart.resolve_mpa_restart(str(tmp_path / "nope.h5"))
    assert d.stage == w_restart.STAGE_BUILD
    assert "nothing to resume from" in w_restart.describe(d)


def test_a_finalized_pole_field_skips_the_fit_too(bundle, wfn_a):
    """Poles present and finalized -> STAGE_SIGMA: sweep AND fit skipped."""
    d = w_restart.resolve_mpa_restart(
        str(bundle["path"]), identity=w_restart.wfn_identity(wfn_a))
    assert d.stage == w_restart.STAGE_FIT, "fixture order"
    w_restart.fit_from_w_omega(d, print_fn=lambda *_: None)
    d2 = w_restart.resolve_mpa_restart(
        str(bundle["path"]), identity=w_restart.wfn_identity(wfn_a))
    assert d2.stage == w_restart.STAGE_SIGMA
    assert d2.fit_ledger is not None and d2.fit_ledger["complete"]
    # THE W(omega) TENSOR IS STILL THERE.  ``allocate_fit_store``
    # deletes Omega_p / B_p / fit_* and nothing else, which is what
    # makes one file a bundle rather than two files a convention.
    assert d2.w_header is not None
    line = w_restart.describe(d2)
    assert "RESUMING from its pole field" in line and "n_p=" in line


# ---------------------------------------------------------------------------
# (b) THE ROUND TRIP: resuming changes no byte the Sigma pass reads
# ---------------------------------------------------------------------------

def test_the_resumed_fit_is_bit_identical_to_the_direct_fit(
        tmp_path, wfn_a):
    """write -> resume -> fit  ==  write -> fit, byte for byte.

    THE ARMS DIFFER ONLY IN WHO CALLS ``run_fit_driver``.  The control
    arm calls it the way every test and every probe script has; the
    restart arm reaches it through ``resolve_mpa_restart`` +
    ``fit_from_w_omega``, which take ``n_p`` and the abscissae off the
    store's own stamps instead of off the caller.  Bit-identity is the
    assertion because that is exactly the claim: the seam re-derives
    the fit's inputs from the file and must re-derive them EXACTLY.
    A tolerance here would pass a seam that read the sampling record's
    ``n_p`` and rounded it.
    """
    ctl_path = tmp_path / "control.h5"
    rst_path = tmp_path / "restart.h5"
    field = _write_bundle(ctl_path, wfn_a)
    _write_bundle(rst_path, wfn_a)

    fit_driver.run_fit_driver(
        str(ctl_path), _W_NAME, str(ctl_path), field["z"], field["n_p"])

    d = w_restart.resolve_mpa_restart(
        str(rst_path), identity=w_restart.wfn_identity(wfn_a))
    assert d.stage == w_restart.STAGE_FIT
    w_restart.fit_from_w_omega(d, print_fn=lambda *_: None)

    # THE SIGMA PASS'S OWN READ, not a peek at the raw datasets: the
    # claim is about what the tau build consumes.
    for p in range(field["n_p"]):
        a_O, a_B = MS.read_pole_slice(str(ctl_path), p)
        b_O, b_B = MS.read_pole_slice(str(rst_path), p)
        assert np.array_equal(
            np.asarray(a_O).view(np.uint8), np.asarray(b_O).view(np.uint8)
        ), f"Omega_p differs at pole {p}"
        assert np.array_equal(
            np.asarray(a_B).view(np.uint8), np.asarray(b_B).view(np.uint8)
        ), f"B_p differs at pole {p}"

    # AND THE DECLARATIONS TRAVELLED.  A bit-identical pole field under
    # a different unit or a different screening_content would be a
    # different physical object with the same bytes.
    lc = MS.fit_completion_ledger(str(ctl_path))
    lr = MS.fit_completion_ledger(str(rst_path))
    for key in ("energy_unit", "screening_content", "n_p", "n_q",
                "q_storage", "table_hash", "n_mu"):
        assert lc[key] == lr[key], key
    assert bool(lr["complete"]) and np.all(lr["q_done"])


def test_the_resume_reaches_the_fit_with_the_stores_own_n_p(tmp_path,
                                                            wfn_a):
    """``n_p`` comes off the sampling stamp, so no deck key can disagree.

    Written with a DIFFERENT ``n_p`` from the module default so a seam
    that hard-coded one would be red rather than accidentally right.
    """
    path = tmp_path / "np3.h5"
    _write_bundle(path, wfn_a, n_p=3)
    d = w_restart.resolve_mpa_restart(
        str(path), identity=w_restart.wfn_identity(wfn_a))
    assert d.w_header["sampling"]["n_p"] == 3
    w_restart.fit_from_w_omega(d, print_fn=lambda *_: None)
    assert MS.fit_completion_ledger(str(path))["n_p"] == 3


# ---------------------------------------------------------------------------
# (c) THE RED TWINS — four files that look complete
# ---------------------------------------------------------------------------

def test_a_store_swept_from_another_wfn_is_refused_by_name(bundle,
                                                           wfn_b):
    """The one check the store layer cannot make for itself."""
    with pytest.raises(ValueError) as exc:
        w_restart.resolve_mpa_restart(
            str(bundle["path"]), identity=w_restart.wfn_identity(wfn_b))
    msg = str(exc.value)
    assert "DIFFERENT" in msg and "wavefunction" in msg
    # BOTH paths are named: the only useful thing to say about this
    # refusal is which two files the run is choosing between.
    assert str(wfn_b._filename) in msg


def test_a_half_swept_store_is_refused_and_names_the_slab(tmp_path,
                                                          wfn_a):
    path = tmp_path / "half.h5"
    _write_bundle(path, wfn_a, ready_all=False)
    with pytest.raises(ValueError) as exc:
        w_restart.resolve_mpa_restart(
            str(path), identity=w_restart.wfn_identity(wfn_a))
    msg = str(exc.value)
    assert "frequency slabs ready" in msg
    assert str(2 * _N_P - 1) in msg, "the missing slab is not named"


def test_a_store_holding_W_not_W_c_is_refused_at_the_seam(tmp_path,
                                                          wfn_a):
    """``require_correlation_part`` — reached from the resolver."""
    path = tmp_path / "fullW.h5"
    _write_bundle(path, wfn_a, screening_content="W")
    with pytest.raises(ValueError) as exc:
        w_restart.resolve_mpa_restart(
            str(path), identity=w_restart.wfn_identity(wfn_a))
    assert "correlation part" in str(exc.value)
    assert "130.651" in str(exc.value), "the measured cost is the message"


def test_an_undeclared_store_is_refused_rather_than_guessed(tmp_path,
                                                            wfn_a):
    tables, verdict, n_mu = _geometry()
    z = _protocol_grid(_N_P)
    Omega, B = _planted_field(_N_P, _N_Q_IBZ, n_mu)
    W = _synthesize_w(Omega, B, z)
    path = tmp_path / "undeclared.h5"
    MS.allocate_w_omega(
        str(path), _W_NAME, n_omega=2 * _N_P, n_q_on_disk=_N_Q_IBZ,
        n_mu=n_mu, tables=tables, omega=z, sampling=_SAMPLING,
        omega_line=np.array([0] * _N_P + [1] * _N_P, dtype=np.int32),
        closure_verdict=verdict,
        provenance=w_restart.wfn_identity(wfn_a))
    for i in range(2 * _N_P):
        MS.write_w_slab(str(path), _W_NAME, i, W[i], ready=True)
    with pytest.raises(ValueError) as exc:
        w_restart.resolve_mpa_restart(
            str(path), identity=w_restart.wfn_identity(wfn_a))
    assert "does not declare WHICH screening object" in str(exc.value)


def test_the_writer_refuses_a_store_with_no_wfn_stamp(tmp_path, wfn_a):
    """A store nobody can attribute is refused AT WRITE, not at read.

    The read-side check can only compare a stamp that exists; the
    moment to insist on one is the only moment the answer is free.
    """
    tables, verdict, n_mu = _geometry()
    z = _protocol_grid(_N_P)
    Omega, B = _planted_field(_N_P, _N_Q_IBZ, n_mu)
    W = _synthesize_w(Omega, B, z)
    with pytest.raises(ValueError) as exc:
        w_restart.write_w_omega(
            str(tmp_path / "nostamp.h5"), W, tables=tables, omega=z,
            sampling=_SAMPLING, closure_verdict=verdict,
            omega_line=np.array([0] * _N_P + [1] * _N_P, dtype=np.int32),
            identity={})
    assert "wfn_identity" in str(exc.value)


def test_wfn_identity_follows_the_zeta_rule(tmp_path):
    """realpath + size, and NOT mtime — the zeta precedent, not a copy.

    A symlink to the same file must read as the same WFN (the fleet
    stages every leg behind its own directory), and touching the file
    must not force a re-sweep.
    """
    real = tmp_path / "WFN.h5"
    real.write_bytes(b"\x00" * 128)
    link = tmp_path / "staged.h5"
    link.symlink_to(real)
    a = w_restart.wfn_identity(_Wfn(real))
    b = w_restart.wfn_identity(_Wfn(link))
    assert a == b, "a staging symlink read as a different WFN"
    assert set(a) == {"wfn_file", "wfn_bytes"}, (
        f"identity grew a field: {sorted(a)} — mtime is deliberately "
        f"absent (gw_init._zeta_fit_provenance)")


def test_a_read_only_bundle_refuses_the_fit_instead_of_mutating_it(
        tmp_path, wfn_a):
    """The poles go INTO the bundle, so a read-only one is refused.

    MEASURED ON THE REAL ARTIFACT, which is why this cell exists: the
    campaign's production W(z) store
    (``mpa_wcprod_0809/stores/W_omega_full_wc.h5``, 20.8 GB, 16 samples,
    64 q, W_c, Ha) resolves to STAGE_FIT through this seam — correctly —
    and the fleet treats that file as a read-only input.  Without this
    refusal a deck pointing ``mpa_fit_file`` at it would rewrite 20 GB
    of someone else's evidence, and would learn about it from an h5py
    permission traceback partway through the walk.
    """
    path = tmp_path / "ro.h5"
    _write_bundle(path, wfn_a)
    d = w_restart.resolve_mpa_restart(
        str(path), identity=w_restart.wfn_identity(wfn_a))
    path.chmod(0o444)
    try:
        with pytest.raises(PermissionError) as exc:
            w_restart.fit_from_w_omega(d, print_fn=lambda *_: None)
    finally:
        path.chmod(0o644)
    assert "not writable" in str(exc.value)
    assert str(path) in str(exc.value)


def test_a_fit_only_bundle_does_not_announce_a_W_it_does_not_hold(
        tmp_path, wfn_a):
    """A production fit store's samples are usually gone; say so.

    ``mpa_fit_np8_wc.h5`` on the cluster is exactly this shape — a
    finalized pole field with no W(omega) beside it — and a line of
    ``?`` in every slot reads as a store that failed to describe
    itself rather than one with nothing left to describe.
    """
    path = tmp_path / "fitonly.h5"
    field = _write_bundle(path, wfn_a)
    fit_driver.run_fit_driver(str(path), _W_NAME, str(path),
                              field["z"], field["n_p"])
    with h5py.File(str(path), "a") as f:
        for key in (_W_NAME, _W_NAME + "__qirr", _W_NAME + "__mpa"):
            if key in f:
                del f[key]
    d = w_restart.resolve_mpa_restart(
        str(path), identity=w_restart.wfn_identity(wfn_a))
    assert d.stage == w_restart.STAGE_SIGMA and d.w_header is None
    line = w_restart.describe(d)
    assert "not in this bundle" in line
    assert "?" not in line, line


# ---------------------------------------------------------------------------
# (d) THE DRIVER SEAM IS ONE CALL, AND THE REFUSALS ARE NOT DUPLICATED
# ---------------------------------------------------------------------------

def test_the_mpa_driver_takes_the_decision_through_the_seam():
    """A source ratchet: ``mpa_pipeline`` asks ``w_restart``, once.

    The ruling this branch implements sizes the feature by how much
    machinery a CORE DRIVER carries, so the count is asserted rather
    than described.  Executable lines only — the tree's comments are
    load-bearing documentation and nobody wants them counted against
    the budget.
    """
    import inspect

    from gw import mpa_pipeline

    src = inspect.getsource(mpa_pipeline.compute_mpa_sigma_pipeline)
    assert "w_restart.resolve_mpa_restart" in src
    assert "w_restart.fit_from_w_omega" in src
    body = [ln.strip() for ln in src.splitlines()]
    spent = [ln for ln in body
             if ("w_restart" in ln or "restart.announce" in ln
                 or ln.startswith("restart = ") or ln == "restart = (")]
    assert len(spent) <= 12, (
        f"the restart feature costs {len(spent)} driver lines; the "
        f"ruling's budget is a dozen.\n" + "\n".join(spent))


def test_the_seam_does_not_re_implement_a_store_refusal():
    """Every refusal is the format's own call, not a second copy.

    The hazard this ratchet closes is the one the restart consolidation
    was written about: a second implementation of a check is a second
    claim about the file, and the copy is the one that goes stale.
    """
    import inspect

    src = inspect.getsource(w_restart)
    for owned in ("read_w_header", "require_correlation_part",
                  "canonical_energy_unit", "fit_completion_ledger"):
        assert f"mpa_store.{owned}" in src, owned
    # The two digests and the table hash are read_w_header's business
    # and must not be recomputed here.
    for forbidden in ("omega_grid_digest", "validate_qirr_tables",
                      "digest()"):
        assert forbidden not in src, (
            f"w_restart recomputes {forbidden}; read_w_header owns it")
