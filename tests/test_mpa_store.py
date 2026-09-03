"""The frequency-resolved W schema and the staged B/Ω store, at the seams.

WHAT THIS SUITE IS AND WHAT IT DELIBERATELY IS NOT.  Every gate below
runs HOST-SIDE with plain h5py at LOGICAL extents.  The phdf5 FFI is not
built on WSL, so the format is tested where its claims actually live —
the attrs, the ranks, the ledgers and the refusals — exactly the way the
symmetry lane tested the q_irr format it extends.  The ``SlabIO`` write
path, where every rank contributes its own (μ, ν) hyperslab and no rank
holds the whole array, is what ``mpa_store.stamp_w_omega`` exists for;
its Perlmutter leg is ``tests/multi_device/mpa_fit_stream_gate.py``.
Nothing here claims
multi-rank coverage; ``test_the_column_block_is_sharded_on_rows_only``
tests the REFUSAL that keeps the sharding 1-D, not the sharding itself.

THE SYNTHETIC GEOMETRY IS ORBIT-CLOSED BY CONSTRUCTION, not by luck.
The centroid set is the union of a handful of seeds' orbits under a
non-symmorphic glide, so ``verify_centroid_orbit_closure`` passes as a
property of the builder and not of a file somebody might regenerate.
The glide is chosen (rather than a symmorphic group) so ``L_table`` is
non-zero and VARIES across μ, which keeps the umklapp phase live in the
per-frequency unfold arm — the phase is ``exp(2πi q·(L_μ − L_ν))``, a
DIFFERENCE, so a set whose centroids share one L makes it identically 1
however large L is.

NO DECK FIXTURE IS READ.  Everything is synthesised in memory, which is
what makes this suite runnable in a tree that has the symmetry
checkpoint's format layer and nothing else.
"""

from __future__ import annotations

import inspect
import os
import sys
import tempfile

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("symmetry_maps.qirr_store")

from symmetry_maps import verify_centroid_orbit_closure           # noqa: E402
from symmetry_maps import qirr_store as QS                        # noqa: E402

from file_io import mpa_store as MS                               # noqa: E402
from gw.mpa import tiling                                         # noqa: E402
from tests._mpa_test_geometry import (                            # noqa: E402
    FFT as _FFT, SYMS as _SYMS, TNP as _TNP, N_Q_FULL as _N_Q_FULL,
    N_Q_IBZ as _N_Q_IBZ, HostSlabIO, closed_centroid_set, geometry)

#: Eight seeds whose glide orbits union to this suite's centroid set;
#: the builder itself lives in tests/_mpa_test_geometry.py.
_SEEDS = ((0, 0, 0), (4, 5, 0), (2, 2, 6), (9, 7, 6),
          (1, 2, 3), (7, 3, 4), (1, 1, 1), (5, 8, 9))

#: The certification every ``finalize_fit_store`` below must carry, since
#: 2026-08-15: a store the validator would refuse cannot be finalized at
#: all (audit §E.3 item 1), so a cell whose subject is something else
#: passes a pair of thresholds loose enough never to be the reason it
#: fails.  The cells that are ABOUT certification state their own.
_CERT = {"condition_max_allowed": 1.0e12,
         "backward_error_max_allowed": 1.0}

#: The sampling record every W(ω) file must carry.  Si-shaped: the
#: double-parallel lines ϖ₁ = 0.1 Ha and ϖ₂ = 1 Ha, α = 1 for an
#: insulator, and an n_p that makes 2·n_p = the frequency count below.
_SAMPLING = {"varpi": [0.1, 1.0], "n_p": 3, "alpha": 1,
             "omega_max": 2.5, "protocol": "double_parallel"}


# ---------------------------------------------------------------------------
# Geometry — one shared builder, this suite's seeds
# ---------------------------------------------------------------------------

def _closed_centroid_set():
    return closed_centroid_set(_SEEDS)


def _geometry():
    return geometry(_SEEDS)


def _w_omega(n_omega, n_q, n_mu, seed=17):
    """A frequency-resolved wedge with a DIFFERENT tensor at every ω.

    Different per slab on purpose: a builder that broadcast one tensor
    across ω would let a reader that returns the wrong slab pass every
    round-trip assertion below.
    """
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((n_omega, n_q, n_mu, n_mu))
         + 1j * rng.standard_normal((n_omega, n_q, n_mu, n_mu)))
    return 0.5 * (a + np.swapaxes(a.conj(), -1, -2))


def _hand_unfold(W_ibz, tables):
    """The per-element reference, in plain numpy::

        W_full[q, μ, ν] = exp(2πi q_irr·(L_{s,μ} − L_{s,ν}))
                          · W_ibz[i(q), α_s(μ), α_s(ν)]

    conjugated on TRS rows.  Written out rather than reused from the
    kernel so the unfold arm compares two independent computations.
    """
    irr = np.asarray(tables.irr_idx_q)
    sym = np.asarray(tables.sym_idx_q)
    n_mu = int(np.asarray(tables.sym_perm).shape[-1])
    out = np.zeros((len(irr), n_mu, n_mu), dtype=W_ibz.dtype)
    for iq in range(len(irr)):
        parent, s = int(irr[iq]), int(sym[iq])
        p = np.asarray(tables.sym_perm[s])
        blk = W_ibz[parent][np.ix_(p, p)]
        qL = (np.asarray(tables.L_table[s], dtype=float)
              @ np.asarray(tables.q_irr_frac)[parent])
        ph = np.exp(2j * np.pi * qL)
        blk = ph[:, None] * blk * np.conj(ph)[None, :]
        out[iq] = np.conj(blk) if s >= int(tables.n_sym_spatial) else blk
    return out


def _assert_offdiag_elementwise(got, want, label):
    """μ ≠ ν, every element, never a norm.

    Two conjugation bugs shipped in this area and BOTH were
    diagonal-preserving and off-diagonal-destroying — 183.61 eV of error
    with the diagonal exactly zero, and hermiticity, the electron count,
    the spectrum and eqp.dat all green through them.  A Frobenius norm
    or a trace cannot see that class of failure.
    """
    g = np.asarray(got)
    w = np.asarray(want)
    assert g.shape == w.shape, f"{label}: {g.shape} vs {w.shape}"
    n = g.shape[-1]
    mask = ~np.eye(n, dtype=bool)
    gm, wm = g[..., mask], w[..., mask]
    bad = ~np.isclose(gm, wm, rtol=1e-13, atol=1e-13)
    assert not bad.any(), (
        f"{label}: {int(bad.sum())} of {gm.size} OFF-DIAGONAL elements "
        f"differ, worst {np.max(np.abs(gm - wm)):.3e}")


def _is_empty(path):
    with h5py.File(path, "r") as f:
        return len(list(f.keys())) == 0


def _mesh():
    import jax
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1),
                ("x", "y"))


def _write_w(path, name="W_qmunu_omega", n_omega=6, ready=True, seed=17):
    """A complete W(ω) file, plus the tensor it was written from."""
    tables, verdict, n_mu = _geometry()
    W = _w_omega(n_omega, _N_Q_IBZ, n_mu, seed=seed)
    omega = np.array([0.0 + 0.1j, 0.4 + 0.1j, 0.9 + 0.1j,
                      0.0 + 1.0j, 0.5 + 1.0j, 1.2 + 1.0j][:n_omega])
    line = np.array([0, 0, 0, 1, 1, 1][:n_omega], dtype=np.int32)
    MS.allocate_w_omega(
        path, name, n_omega=n_omega, n_q_on_disk=_N_Q_IBZ, n_mu=n_mu,
        tables=tables, omega=omega, sampling=_SAMPLING, omega_line=line,
        closure_verdict=verdict, provenance={"deck": "synthetic-glide"})
    for i in range(n_omega):
        MS.write_w_slab(path, name, i, W[i], ready=ready)
    return W, tables, verdict, n_mu, omega, line


@pytest.fixture()
def tmpdir_path():
    d = tempfile.mkdtemp(prefix="mpa_store_")
    try:
        yield os.path.join(d, "w.h5")
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Arm 1 — the round trip, per frequency slab
# ---------------------------------------------------------------------------

def test_every_frequency_slab_round_trips_bit_identically(tmpdir_path):
    """Bit equality, per slab, off-diagonals asserted element-wise.

    ``array_equal`` and not a tolerance: this is the same bytes through
    a file, so nothing may differ.  The off-diagonal check runs beside
    it because bit equality on the whole array would already have caught
    a conjugation bug — but the ELEMENT-WISE form is what stays correct
    when a later arm relaxes to a tolerance, and it is the shape the
    lane's gates are written in.
    """
    W, _, _, _, _, _ = _write_w(tmpdir_path)
    for i in range(W.shape[0]):
        got, hdr = MS.read_w_slab(tmpdir_path, "W_qmunu_omega", i)
        assert np.array_equal(got, W[i]), f"slab {i} is not bit-identical"
        _assert_offdiag_elementwise(got, W[i], f"slab {i}")
        assert hdr["n_omega"] == W.shape[0]
        assert bool(hdr["data_ready"][i])


def test_collective_writer_keeps_large_bytes_behind_slabio(
        tmpdir_path, monkeypatch):
    """The production helper uses SlabIO; h5py only stamps metadata."""
    import common.collectives as collectives
    import file_io.slab_io as slab_io

    calls = []

    class RecordingSlabIO(HostSlabIO):
        """The shared host shim, plus the call census this test needs."""

        def __init__(self, path, *, mode, mesh):
            calls.append(("open", mode, mesh))
            super().__init__(path, mode=mode, mesh=mesh)

        def __exit__(self, *exc):
            super().__exit__(*exc)
            calls.append(("close",))

        def create_dataset(self, name, **kw):
            calls.append(("create", name))
            super().create_dataset(name, **kw)

        def write_slab(self, name, value, **kw):
            super().write_slab(name, value, **kw)
            calls.append(("write", name))

    monkeypatch.setattr(slab_io, "SlabIO", RecordingSlabIO)
    monkeypatch.setattr(collectives, "process_rank", lambda: 0)
    monkeypatch.setattr(collectives, "barrier", lambda _name: False)

    tables, verdict, n_mu = _geometry()
    omega = np.asarray([0.2 + 0.1j])
    sampling = dict(_SAMPLING, n_p=1)
    mesh = _mesh()
    shape = (1, _N_Q_IBZ, n_mu, n_mu)
    assert MS.allocate_w_omega_collective(
        tmpdir_path, "W", mesh_xy=mesh, n_omega=1,
        n_q_on_disk=_N_Q_IBZ, n_mu=n_mu, tables=tables, omega=omega,
        sampling=sampling, closure_verdict=verdict) is None
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    W_host = _w_omega(1, _N_Q_IBZ, n_mu)[0]
    W = jax.device_put(W_host, NamedSharding(mesh, P(None, "x", "y")))
    # BOTH COLLECTIVES RETURN None ON EVERY RANK (audit §E.3 item 3).
    # They used to hand back the real thing on rank 0 and None
    # elsewhere, which is a mesh divergence waiting for the first caller
    # that branches on the return; the readiness count they used to carry
    # is read from the ledger below, where it lives.
    assert MS.write_w_slab_collective(
        tmpdir_path, "W", 0, W, mesh_xy=mesh,
        global_shape=shape) is None

    got, header = MS.read_w_slab(tmpdir_path, "W", 0)
    np.testing.assert_array_equal(got, W_host)
    assert header["data_ready"].tolist() == [True]
    assert header["n_ready"] == 1
    assert [call[0] for call in calls] == [
        "open", "create", "close", "open", "write", "close"]


def test_complete_pole_writer_preserves_2d_sharding_and_finalizes(
        tmpdir_path, monkeypatch):
    """An algebraic pole model enters the ordinary MPA store unchanged."""
    import common.collectives as collectives
    import file_io.slab_io as slab_io
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    calls = []

    class RecordingSlabIO(HostSlabIO):
        def __init__(self, path, *, mode, mesh):
            calls.append(("open", mode))
            super().__init__(path, mode=mode, mesh=mesh)

        def write_slab(self, name, value, **kw):
            calls.append(("write", name, value.sharding.spec))
            super().write_slab(name, value, **kw)

    monkeypatch.setattr(slab_io, "SlabIO", RecordingSlabIO)
    monkeypatch.setattr(collectives, "process_rank", lambda: 0)
    monkeypatch.setattr(collectives, "barrier", lambda _name: False)

    mesh = _mesh()
    sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    # The final row/column are device-count-dependent pad and must not land.
    shape = (1, 2, 4, 4)
    Omega = np.full(shape, 0.8, dtype=np.float64)
    B = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 100.0
    Omega[..., -1, :] = 0.0
    Omega[..., :, -1] = 0.0
    B[..., -1, :] = 0.0
    B[..., :, -1] = 0.0
    ledger = MS.write_complete_pole_store_collective(
        tmpdir_path,
        jax.device_put(Omega, sharding),
        jax.device_put(B.astype(np.complex128), sharding),
        mesh_xy=mesh,
        n_mu_logical=3,
        energy_unit="Ry",
        provenance={"fit_protocol": "synthetic_algebraic",
                    "screening_diagrams": "rpa"},
        certification={"condition_max_allowed": 1.0,
                       "backward_error_max_allowed": 1.0},
    )

    assert ledger["complete"] and ledger["n_p"] == 1
    assert ledger["n_done"] == ledger["n_total"] == 6
    assert ledger["condition_max"] == ledger["backward_error_max"] == 0.0
    assert ledger["provenance"]["fit_protocol"] == "synthetic_algebraic"
    got_Omega, got_B = MS.read_poles(tmpdir_path)
    np.testing.assert_array_equal(got_Omega, Omega[..., :3, :3])
    np.testing.assert_array_equal(got_B, B[..., :3, :3])
    writes = [row for row in calls if row[0] == "write"]
    assert [row[1] for row in writes] == ["Omega_p", "B_p"]
    assert all(row[2] == P(None, None, "x", "y") for row in writes)


def test_complete_pole_writer_round_trips_ordered_residues(
        tmpdir_path, monkeypatch):
    """The algebraic GN store carries D beside B with an integrity stamp."""
    import common.collectives as collectives
    import file_io.slab_io as slab_io
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    monkeypatch.setattr(slab_io, "SlabIO", HostSlabIO)
    monkeypatch.setattr(collectives, "process_rank", lambda: 0)
    monkeypatch.setattr(collectives, "barrier", lambda _name: False)

    mesh = _mesh()
    sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    Omega = np.asarray([[[[0.8, 0.9], [1.0, 1.1]]]], np.complex128)
    B = np.asarray([[[[0.4, 0.3j], [-0.2j, 0.5]]]], np.complex128)
    D = np.asarray([[[[0.1j, 0.02], [0.02, -0.1j]]]], np.complex128)
    ledger = MS.write_complete_pole_store_collective(
        tmpdir_path, jax.device_put(Omega, sharding),
        jax.device_put(B, sharding), B_odd_p=jax.device_put(D, sharding),
        mesh_xy=mesh, n_mu_logical=2, energy_unit="Ry",
        provenance={"fit_protocol": "ordered_two_point_ppm"},
        certification={"condition_max_allowed": 1.0,
                       "backward_error_max_allowed": 1.0})

    assert ledger["ordered_residues"] is True
    got_Omega, got_B, got_D = MS.read_poles(tmpdir_path, include_odd=True)
    np.testing.assert_array_equal(got_Omega, Omega)
    np.testing.assert_array_equal(got_B, B)
    np.testing.assert_array_equal(got_D, D)
    with h5py.File(tmpdir_path, "a") as grp:
        del grp["B_odd_p"]
    with pytest.raises(ValueError, match="partial ordered-residue schema"):
        MS.fit_completion_ledger(tmpdir_path)


def test_complete_pole_writer_refuses_noncausal_live_poles(
        tmpdir_path, monkeypatch):
    """A dormant zero pole is legal; the same pole with residue is not."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = _mesh()
    sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    Omega = jax.device_put(
        np.asarray([[[[0.0 + 0.0j, 0.7 + 0.1j],
                      [0.9 + 0.0j, 1.1 + 0.0j]]]]), sharding)
    B = jax.device_put(
        np.asarray([[[[0.0 + 0.0j, 0.2 + 0.0j],
                      [0.3 + 0.0j, 0.4 + 0.0j]]]]), sharding)
    with pytest.raises(ValueError, match="1 live poles.*Im Omega > 0"):
        MS.write_complete_pole_store_collective(
            tmpdir_path, Omega, B, mesh_xy=mesh, n_mu_logical=2,
            energy_unit="Ry", provenance={"fit_protocol": "bad"},
            certification={"condition_max_allowed": 1.0,
                           "backward_error_max_allowed": 1.0})


def test_the_collective_w_writers_answer_the_same_on_every_rank(
        tmpdir_path, monkeypatch):
    """RED: a collective whose RETURN TYPE depends on the rank is a trap.

    ``allocate_w_omega_collective`` used to hand back rank 0's header and
    ``None`` elsewhere; ``write_w_slab_collective`` the new ready count
    and ``None`` elsewhere.  Every rank calls these, in order, so a caller
    that writes ``if io_result:`` around the next collective takes the
    branch on rank 0 only and hangs the mesh — latent solely because both
    production callers discard the return (audit §E.3 item 3).  This runs
    the same two calls as a NON-ZERO rank, which is the leg the cell above
    (``process_rank`` pinned to 0) structurally cannot reach.
    """
    import common.collectives as collectives
    import file_io.slab_io as slab_io
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    monkeypatch.setattr(slab_io, "SlabIO", HostSlabIO)
    monkeypatch.setattr(collectives, "barrier", lambda _name: False)
    tables, verdict, n_mu = _geometry()
    mesh = _mesh()
    shape = (1, _N_Q_IBZ, n_mu, n_mu)
    W = jax.device_put(_w_omega(1, _N_Q_IBZ, n_mu)[0],
                       NamedSharding(mesh, P(None, "x", "y")))
    for rank in (0, 1):
        monkeypatch.setattr(collectives, "process_rank", lambda r=rank: r)
        assert MS.allocate_w_omega_collective(
            tmpdir_path, "W", mesh_xy=mesh, n_omega=1, mode="w",
            n_q_on_disk=_N_Q_IBZ, n_mu=n_mu, tables=tables,
            omega=np.asarray([0.2 + 0.1j]),
            sampling=dict(_SAMPLING, n_p=1),
            closure_verdict=verdict) is None, f"rank {rank}"
        assert MS.write_w_slab_collective(
            tmpdir_path, "W", 0, W, mesh_xy=mesh,
            global_shape=shape) is None, f"rank {rank}"


def test_collective_reader_keeps_one_frequency_slab_sharded(
        tmpdir_path, monkeypatch):
    W, _, _, _, _, _ = _write_w(tmpdir_path)
    calls = []

    class RecordingSlabIO(HostSlabIO):
        """The shared host shim, plus the open/close census."""

        def __enter__(self):
            calls.append("open")
            return super().__enter__()

        def __exit__(self, *exc):
            calls.append("close")
            super().__exit__(*exc)

    import file_io.slab_io as slab_io
    monkeypatch.setattr(slab_io, "SlabIO", RecordingSlabIO)
    got, header = MS.read_w_slab_collective(
        tmpdir_path, "W_qmunu_omega", 1, mesh_xy=_mesh())
    np.testing.assert_array_equal(np.asarray(got), W[1])
    assert header["data_ready"][1]
    assert calls == ["open", "close"]


def test_a_single_q_of_a_slab_is_the_same_bytes(tmpdir_path):
    W, _, _, _, _, _ = _write_w(tmpdir_path)
    for iq in range(_N_Q_IBZ):
        got, _ = MS.read_w_slab(tmpdir_path, "W_qmunu_omega", 2, q=iq)
        assert np.array_equal(got, W[2, iq])


def test_the_header_reports_the_grid_and_the_protocol(tmpdir_path):
    _, _, _, _, omega, line = _write_w(tmpdir_path)
    hdr = MS.read_w_header(tmpdir_path, "W_qmunu_omega")
    assert np.array_equal(hdr["omega"], omega)
    assert np.array_equal(hdr["omega_line"], line)
    assert hdr["omega_units"] == "Ha"
    assert hdr["sampling"]["n_p"] == _SAMPLING["n_p"]
    assert hdr["sampling"]["alpha"] == _SAMPLING["alpha"]
    assert np.allclose(hdr["sampling"]["varpi"], _SAMPLING["varpi"])
    assert hdr["sampling"]["protocol"] == "double_parallel"
    assert hdr["format_version"] == MS.QIRR_FORMAT_VERSION_FREQ == 2
    assert hdr["freq_axis"] == "leading"
    assert hdr["q_storage"] == "ibz"
    assert "qirr_generator_commit" in hdr["provenance"]
    assert hdr["provenance"]["deck"] == "synthetic-glide"


# ---------------------------------------------------------------------------
# Arm 2 — the leading dimension is REMOVABLE
# ---------------------------------------------------------------------------

def test_the_leading_axis_is_removable(tmpdir_path):
    """A single-ω slice equals a frequency-free file, attr-for-attr.

    THE CLAIM UNDER TEST is the one the whole layout rests on: the
    leading axis is a CONTAINER, not a change of meaning, so it can be
    dropped later without a format migration.  Two things have to hold
    and both are asserted:

    * the BYTES of slab i equal the bytes a version-1 q_irr file written
      from that slab holds — ``array_equal``, not a tolerance;
    * the ATTRS agree name for name and value for value, once the
      version number and the frequency-axis attrs this format adds are
      set aside.  Nothing else may differ: the tables, the digest, the
      closure record, the q_storage verdict, the logical μ extent and
      the provenance are all the SAME claims about the SAME wedge.

    The exempt set is written out explicitly rather than filtered by
    prefix, so a future attr that quietly appears on one side and not
    the other fails this test instead of being swallowed by a
    ``startswith``.
    """
    W, tables, verdict, _, _, _ = _write_w(tmpdir_path)
    v1 = tmpdir_path.replace("w.h5", "v1.h5")
    slab = 2
    QS.write_qirr_tensor(v1, "W0_qmunu", W[slab], tables=tables,
                         closure_verdict=verdict,
                         provenance={"deck": "synthetic-glide"})

    got, _ = MS.read_w_slab(tmpdir_path, "W_qmunu_omega", slab)
    ref, _ = QS.read_tensor(v1, "W0_qmunu", unfold=False)
    assert np.array_equal(got, ref), (
        "the frequency slab is not the bytes a frequency-free file holds")

    exempt = {QS.QIRR_VERSION_ATTR, MS._FREQ_ATTR} | set(MS._MPA_OWNED_ATTRS)
    # Volatile by construction — a timestamp and a writer name, which say
    # WHEN and BY WHAT and not WHAT.
    exempt |= {"qirr_written_utc", "qirr_writer", "mpa_writer"}
    with h5py.File(tmpdir_path, "r") as f4, h5py.File(v1, "r") as f3:
        a4 = {k: v for k, v in f4["W_qmunu_omega"].attrs.items()
              if k not in exempt}
        a3 = {k: v for k, v in f3["W0_qmunu"].attrs.items()
              if k not in exempt}
        assert set(a4) == set(a3), (
            f"attr NAMES differ once the version and the frequency "
            f"attrs are set aside: only in the v2 file "
            f"{sorted(set(a4) - set(a3))}, only in the v1 file "
            f"{sorted(set(a3) - set(a4))}")
        for key in sorted(a4):
            assert np.all(a4[key] == a3[key]), (
                f"attr {key!r} differs: v2 {a4[key]!r} vs v1 {a3[key]!r}")
        # And the tables are the same tables, dataset for dataset.
        for key in QS._TABLE_ORDER:
            assert np.array_equal(
                f4["W_qmunu_omega" + QS.QIRR_TABLE_SUFFIX][key][()],
                f3["W0_qmunu" + QS.QIRR_TABLE_SUFFIX][key][()]), key


# ---------------------------------------------------------------------------
# Arm 3 — readiness, per frequency.  RED TWIN.
# ---------------------------------------------------------------------------

def test_an_unstamped_frequency_slab_refuses(tmpdir_path):
    """RED TWIN of arm 1: presence is never readiness, per slab.

    The producer fills ω one line-batched sweep at a time, so a file
    with some slabs written and some not is a state the pipeline REACHES
    — after a preemption, or mid-run — rather than one it crashes into.
    ``write_w_slab(ready=False)`` is that state, built by the writer
    rather than hand-forged, and the twin beside it is the same read on
    the same file with the bit set.
    """
    W, _, _, _, _, _ = _write_w(tmpdir_path, n_omega=4, ready=True)
    MS.write_w_slab(tmpdir_path, "W_qmunu_omega", 1, W[1], ready=False)

    MS.read_w_slab(tmpdir_path, "W_qmunu_omega", 0)                # TWIN
    with pytest.raises(ValueError, match="data_ready bit is False"):
        MS.read_w_slab(tmpdir_path, "W_qmunu_omega", 1)
    # Announced, the placeholder is inspectable.
    got, hdr = MS.read_w_slab(tmpdir_path, "W_qmunu_omega", 1,
                              require_ready=False)
    assert np.array_equal(got, W[1])
    assert hdr["n_ready"] == 3


def test_the_readiness_scalar_must_agree_with_the_ledger(tmpdir_path):
    """RED: a v1-honouring reader must not be told a comforting lie.

    ``qirr_data_ready`` is the scalar any version-1 reader will look at,
    and this format keeps it as ``all(ledger)`` — the CONSERVATIVE
    summary.  A file whose scalar says True while slabs are missing is a
    file that would read as data through the old door, so the
    disagreement is refused rather than resolved in either direction.
    """
    W, _, _, _, _, _ = _write_w(tmpdir_path, n_omega=4)
    MS.write_w_slab(tmpdir_path, "W_qmunu_omega", 2, W[2], ready=False)
    MS.read_w_header(tmpdir_path, "W_qmunu_omega")                 # TWIN
    with h5py.File(tmpdir_path, "a") as f:
        f["W_qmunu_omega"].attrs["qirr_data_ready"] = True
    with pytest.raises(ValueError, match="qirr_data_ready"):
        MS.read_w_header(tmpdir_path, "W_qmunu_omega")


# ---------------------------------------------------------------------------
# Arm 4 — the wedge applies PER FREQUENCY
# ---------------------------------------------------------------------------

def test_the_wedge_unfolds_per_frequency(tmpdir_path):
    """``unfold(stored slab) == full slab``, at every ω separately.

    The symmetry operation acts on (q, μ, ν) and does not touch ω, so
    the stored tables are shared across the whole frequency axis and the
    unfold is the landed kernel applied slab by slab.  The reference is
    the independent per-element numpy formula, with the umklapp phase
    and the TRS conjugation both live on this geometry, and the
    comparison is element-wise on the off-diagonals.
    """
    pytest.importorskip("jax")
    W, tables, _, _, _, _ = _write_w(tmpdir_path)
    mesh = _mesh()
    for i in range(W.shape[0]):
        got, hdr = MS.read_w_slab(tmpdir_path, "W_qmunu_omega", i,
                                  unfold=True, mesh_xy=mesh)
        assert got.shape == (_N_Q_FULL, W.shape[-1], W.shape[-1])
        want = _hand_unfold(W[i], tables)
        _assert_offdiag_elementwise(got, want, f"unfold at ω index {i}")
        assert np.allclose(got, want, rtol=1e-13, atol=1e-13)
        assert hdr["q_storage"] == "ibz"


def test_the_unfold_of_a_single_stored_q_refuses(tmpdir_path):
    """RED: a per-q unfold is not a thing, and saying so beats guessing.

    The unfold gathers every full-BZ row from its IBZ parent, so it
    needs the whole wedge at that ω.  Silently unfolding "the wedge
    containing q" would return ``n_q_full`` rows for a caller who asked
    for one.
    """
    _write_w(tmpdir_path)
    with pytest.raises(ValueError, match="cannot unfold a single"):
        MS.read_w_slab(tmpdir_path, "W_qmunu_omega", 0, q=1,
                       unfold=True, mesh_xy=None)


# ---------------------------------------------------------------------------
# Arm 5 — the column budget, its arithmetic, and its refusal
# ---------------------------------------------------------------------------

def test_the_column_budget_is_the_tile_arithmetic():
    """The Si-scale worked example, exactly, plus the closed form.

    N_μ = 480 centroids and n_omega = 16 sampling points (an n_p = 8
    fit's 2·n_p):

        one tile   = 480 * 480 * 16 B = 3 686 400 B  (3.52 MiB)
        per column = 16 * 480 * 16 B  =   122 880 B
        budget     = 3 686 400 / 122 880 = 30 columns EXACTLY,
        and 16 * 480 * 30 * 16 B = 3 686 400 B — the tile to the byte.

    The closed form is ``n_cols = N_μ // n_omega``: the frequency axis
    is paid for out of the column count, one for one.
    """
    assert MS.one_tile_bytes(480) == 3_686_400
    assert MS.choose_column_budget(480, 16) == 30
    assert 16 * 480 * 30 * MS.COMPLEX128_BYTES == MS.one_tile_bytes(480)
    # The closed form, across the schedules the theory plan names:
    # n_p = 6..12 -> n_omega = 12..24.
    for n_omega in (12, 16, 20, 24):
        assert MS.choose_column_budget(480, n_omega) == 480 // n_omega
    # And at the production 960-centroid shape.
    assert MS.choose_column_budget(960, 16) == 60
    # A doubled budget doubles the columns; the relation is linear.
    assert MS.choose_column_budget(
        480, 16, tile_bytes=2 * MS.one_tile_bytes(480)) == 60
    # Never zero, and never more columns than there are.
    assert MS.choose_column_budget(8, 4096) == 1
    assert MS.choose_column_budget(16, 1) == 16
    # BOTH CLAMPS ARE PUBLISHED NOW (audit §E.3 item 5): the upper one is
    # what a generous budget meets, and the docstring's own worked case.
    assert MS.choose_column_budget(480, 16, 100 * 2 ** 20) == 480

    # And the sentence prints the arithmetic that RAN (item 4): the tile
    # product only when the budget IS one tile, and a call a reader can
    # type to get the number beside it.
    one_tile = MS.describe_column_cost(480, 16, 30)
    assert "one (N_mu, N_mu) tile, 480*480*16 B = 3686400 B" in one_tile
    assert ("choose_column_budget(480, 16, tile_bytes=3686400, "
            "itemsize=16) allows 30") in one_tile
    big = MS.describe_column_cost(480, 16, 100, tile_bytes=100 * 2 ** 20)
    assert "480*480*16" not in big, big      # the false equation is gone
    assert f"budget of {100 * 2 ** 20} B" in big
    assert (f"choose_column_budget(480, 16, tile_bytes={100 * 2 ** 20}, "
            "itemsize=16) allows 480") in big


def test_a_column_request_that_busts_the_budget_refuses(tmpdir_path):
    """RED TWIN of the budget: the refusal carries the arithmetic.

    A budget that is enforced by silent truncation is not a budget; one
    enforced by a bare ``ValueError`` is a budget nobody can act on.  So
    the message names the per-column cost, the total, the tile it is
    measured against, the ratio, and the count that would have fit — and
    this test asserts those numbers are IN it, not merely that something
    raised.
    """
    _, _, _, n_mu, _, _ = _write_w(tmpdir_path, n_omega=6)
    budget = MS.choose_column_budget(n_mu, 6)
    assert budget >= 1
    MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 0,
                      range(budget))                              # TWIN
    over = list(range(budget + 1))
    with pytest.raises(ValueError) as exc:
        MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 0, over)
    msg = str(exc.value)
    assert "read_w_columns" in msg and "W_qmunu_omega" in msg
    cost = 6 * n_mu * len(over) * 16
    assert f"6*{n_mu}*{len(over)}*16 B = {cost} B" in msg, msg
    assert f"{n_mu}*{n_mu}*16 B = {n_mu * n_mu * 16} B" in msg, msg
    assert f"allows {budget}" in msg, msg
    # And the budget can be raised DELIBERATELY, which is the point of
    # refusing rather than clamping.
    MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 0, over,
                      tile_bytes=4 * MS.one_tile_bytes(n_mu))


def test_read_w_columns_is_the_slab_read_transposed(tmpdir_path):
    """The block is (n_omega, N_μ_rows, n_cols) and it is the same bytes.

    Contiguous and scattered column selections both, because they take
    different paths through HDF5 (one hyperslab versus a point
    selection) and a format that returned different bytes down the two
    would be wrong in exactly the way nobody looks for.
    """
    W, _, _, n_mu, _, _ = _write_w(tmpdir_path, n_omega=6)
    budget = MS.choose_column_budget(n_mu, 6)
    for q in range(_N_Q_IBZ):
        cols = list(range(2, 2 + min(budget, 3)))
        blk = MS.read_w_columns(tmpdir_path, "W_qmunu_omega", q, cols)
        assert blk.shape == (6, n_mu, len(cols))
        assert np.array_equal(blk, W[:, q][:, :, cols])
    scattered = [0, 3, n_mu - 1][:budget]
    blk = MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 1, scattered)
    assert np.array_equal(blk, W[:, 1][:, :, scattered])


def test_read_w_columns_refuses_a_half_filled_grid(tmpdir_path):
    """RED: a column block spans ALL of ω, so all of ω must be data.

    Stricter than the slab read on purpose.  An unwritten slab reads as
    zeros and the Padé solve fits them happily, so a fit on the ready
    half of the grid returns poles that are WRONG rather than absent —
    the failure the whole readiness ledger exists to prevent, in its
    most expensive form.
    """
    W, _, _, n_mu, _, _ = _write_w(tmpdir_path, n_omega=6)
    MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 0, [0, 1])     # TWIN
    MS.write_w_slab(tmpdir_path, "W_qmunu_omega", 4, W[4], ready=False)
    with pytest.raises(ValueError, match="frequency slabs are not ready"):
        MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 0, [0, 1])
    blk = MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 0, [0, 1],
                            require_ready=False)
    assert blk.shape == (6, n_mu, 2)


def test_the_column_block_is_sharded_on_rows_only(tmpdir_path):
    """RED TWIN: a 2-D sharding is refused, by name, at the read.

    The fit is elementwise in (μ, ν): splitting the column axis as well
    buys no parallelism the column loop does not already have, while
    making each rank's column count a function of the mesh shape — and
    the column count is precisely the quantity the tile budget was
    computed against.  So a 2-D spec is refused HERE, where the budget
    still means what the caller computed, rather than downstream where
    it no longer does.
    """
    _write_w(tmpdir_path, n_omega=6)
    ok = tiling.row_shard_spec()
    assert ok == (None, "x", None)
    MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 0, [0, 1],
                      out_spec=ok)                                 # TWIN
    for bad in ((None, "x", "y"), ("x", "y", None), (None, None, "y"),
                ("x", None, None)):
        with pytest.raises(ValueError, match="ROW AXIS"):
            MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 0, [0, 1],
                              out_spec=bad)
    with pytest.raises(ValueError, match="three entries"):
        MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 0, [0, 1],
                          out_spec=(None, "x"))


def test_the_walk_covers_every_column_exactly_once():
    """The schedule is a partition, not a cover.

    Overlapping blocks would fit a column twice and hand the staged
    store a double write, which it refuses; a gap would leave a column
    unfitted and the finalize would refuse.  Both are worth an assertion
    because the short last block is where an off-by-one lives.
    """
    for n_mu, n_omega in ((480, 16), (500, 16), (13, 4), (7, 7), (31, 30)):
        plan = tiling.plan_column_walk(n_mu, n_omega)
        seen = [c for lo, hi in plan["blocks"] for c in range(lo, hi)]
        assert seen == list(range(n_mu)), (n_mu, n_omega, plan["blocks"])
        assert plan["n_cols"] == MS.choose_column_budget(n_mu, n_omega)
    steps = tiling.fit_schedule(3, 480, 16)
    assert len(steps) == 3 * 16
    assert steps[0] == (0, 0, 30) and steps[16] == (1, 0, 30)


# ---------------------------------------------------------------------------
# Arm 6 — the version-2 discriminant.  THE RED TWIN THAT MOTIVATES IT.
# ---------------------------------------------------------------------------

def test_a_rank_4_tensor_stamped_version_1_is_refused_by_the_v1_reader(
        tmpdir_path):
    """THE HAZARD, CONSTRUCTED — and refused where every consumer reads.

    The malicious case is built to be maximally quiet: ``n_omega`` is
    chosen EQUAL to the wedge extent, so a version-1 reader's
    ``ds.shape[0]`` is the number it expects, ``ds.shape[-1]`` is
    genuinely N_μ, the tables validate against both, the q_storage
    cross-check agrees, the table digest matches and the readiness flag
    is set.  Nothing in the version-1 path has anything to object to —
    except the rank, which is a property of the bytes.

    THIS TEST CHANGED SIDES AT THE LANDING, AS IT SAID IT WOULD.  It was
    written while the rank refusal lived only in ``mpa_store``, and half
    of it asserted the DEFECT: that ``qirr_store.read_tensor`` returned
    these bytes without raising, handing back an array whose leading axis
    is FREQUENCY while its header called that axis q.  The docstring
    registered the trigger in as many words — "if a later change makes
    the v1 reader refuse on its own, this line fails and the follow-up
    below can be closed".  That change is the q_irr rank-refusal branch,
    which put the check inside ``read_tensor`` above ``read_tables`` and
    before any extent is believed.  So the assertion is inverted here
    rather than deleted: the line that used to pin the corruption now
    pins its absence.

    ``qirr_store.read_tensor`` is where the refusal matters: it protects
    the consumer who has never heard of a frequency axis and is simply
    reading a restart file.  (``mpa_store``'s dispatcher wrapper, which
    reached the same refusal by delegation, is deleted — a wrapper
    protects only the callers who use it.)
    """
    tables, verdict, n_mu = _geometry()
    n_omega = _N_Q_IBZ            # THE COINCIDENCE, on purpose
    W = _w_omega(n_omega, _N_Q_IBZ, n_mu)
    name = "W_qmunu_omega"
    MS.allocate_w_omega(
        tmpdir_path, name, n_omega=n_omega, n_q_on_disk=_N_Q_IBZ,
        n_mu=n_mu, tables=tables, omega=np.arange(n_omega) * 0.3 + 0.1j,
        sampling=_SAMPLING, closure_verdict=verdict)
    for i in range(n_omega):
        MS.write_w_slab(tmpdir_path, name, i, W[i])
    # Forge the version attr back to 1 — a writer that gained the axis
    # without bumping, or a hand-edited file.
    with h5py.File(tmpdir_path, "a") as f:
        f[name].attrs[QS.QIRR_VERSION_ATTR] = np.int64(1)
        del f[name].attrs[MS._FREQ_ATTR]

    # THE FIX, WHERE IT NOW LIVES.  The version-1 reader refuses these
    # bytes on its own, naming both ranks — no wrapper required.
    with pytest.raises(ValueError) as exc:
        QS.read_tensor(tmpdir_path, name, unfold=False)
    msg = str(exc.value)
    assert "qirr_format_version=1" in msg and "rank 3" in msg, msg
    assert "rank 4" in msg or str(tuple(W.shape)) in msg, msg
    assert "RANK IS THE DISCRIMINANT" in msg, msg


def test_the_frequency_attr_and_the_version_must_agree(tmpdir_path):
    """RED: a half stamp is refused, not read as the nearer version."""
    _write_w(tmpdir_path, n_omega=4)
    with h5py.File(tmpdir_path, "a") as f:
        del f["W_qmunu_omega"].attrs[MS._FREQ_ATTR]
    with pytest.raises(ValueError, match="mpa_freq_axis"):
        MS.read_w_header(tmpdir_path, "W_qmunu_omega")


def test_the_frequency_stamp_refuses_a_rank_3_dataset(tmpdir_path):
    """RED, at the WRITE seam: the stamp will not claim a missing axis."""
    W, tables, verdict, _, _, _ = _write_w(tmpdir_path, n_omega=4)
    v1 = tmpdir_path.replace("w.h5", "v1.h5")
    QS.write_qirr_tensor(v1, "W0_qmunu", W[0], tables=tables,
                         closure_verdict=verdict)
    with pytest.raises(ValueError, match="rank 4"):
        MS.stamp_w_omega(v1, "W0_qmunu", tables=tables,
                         omega=[0.1j], sampling=_SAMPLING,
                         closure_verdict=verdict)


# ---------------------------------------------------------------------------
# Arm 7 — shape-versus-attribute, and the grid's identity
# ---------------------------------------------------------------------------

def test_the_shape_wins_the_argument_about_n_omega(tmpdir_path):
    """RED: the SHAPE is the discriminant, the attr is its cross-check."""
    _write_w(tmpdir_path, n_omega=4)
    MS.read_w_header(tmpdir_path, "W_qmunu_omega")                 # TWIN
    with h5py.File(tmpdir_path, "a") as f:
        f["W_qmunu_omega"].attrs["mpa_n_omega"] = np.int64(3)
    with pytest.raises(ValueError, match="mpa_n_omega=3"):
        MS.read_w_header(tmpdir_path, "W_qmunu_omega")


def test_a_half_stamped_version_2_file_refuses(tmpdir_path):
    """RED: the sampling record is half the stamp, and half is not read.

    The rank check settles WHICH format a file is; this settles whether
    that format's own record is whole.  The attr dropped here is the
    partition α — the one parameter that differs between insulators and
    metals while changing nothing about the shapes, so its absence is
    invisible to every shape check there is.
    """
    _write_w(tmpdir_path, n_omega=4)
    MS.read_w_header(tmpdir_path, "W_qmunu_omega")                 # TWIN
    with h5py.File(tmpdir_path, "a") as f:
        del f["W_qmunu_omega"].attrs["mpa_alpha"]
    with pytest.raises(ValueError, match=r"missing \['mpa_alpha'\]"):
        MS.read_w_header(tmpdir_path, "W_qmunu_omega")


def test_a_tampered_omega_grid_refuses(tmpdir_path):
    """RED: the abscissae are hashed WITH the protocol that made them.

    A fit is a solve against particular sampling points.  Moving one and
    leaving the stamp is a file whose poles were fitted against
    abscissae the file no longer reports, and every pole out of it is
    wrong in a way no shape check can see.
    """
    _write_w(tmpdir_path, n_omega=4)
    MS.read_w_header(tmpdir_path, "W_qmunu_omega")                 # TWIN
    with h5py.File(tmpdir_path, "a") as f:
        g = f["W_qmunu_omega" + MS.MPA_GROUP_SUFFIX]
        w = g["omega"][()]
        w[2] += 1e-9
        g["omega"][...] = w
    with pytest.raises(ValueError, match="ω-grid hash mismatch"):
        MS.read_w_header(tmpdir_path, "W_qmunu_omega")


def test_the_protocol_moves_the_grid_hash_even_at_a_fixed_grid():
    """Two runs can sample the same points from different partitions.

    The digest covers the protocol as well as the abscissae, so a
    nested-partition chain cannot be confused with an unrelated fit that
    happened to land on the same 2·n_p points.
    """
    omega = np.array([0.1j, 0.3 + 0.1j, 1.0j])
    line = np.array([0, 0, 1], dtype=np.int32)
    base = MS.omega_grid_digest(omega, line, _SAMPLING)
    for key, val in (("alpha", 2), ("n_p", 4), ("omega_max", 3.0),
                     ("protocol", "single_line")):
        moved = dict(_SAMPLING)
        moved[key] = val
        assert MS.omega_grid_digest(omega, line, moved) != base, key
    assert MS.omega_grid_digest(omega, np.array([0, 1, 1], np.int32),
                                _SAMPLING) != base


def test_the_sampling_record_is_required():
    """RED: α and n_p are not optional telemetry.

    A fit whose partition nobody recorded cannot be extended — nested
    partitions are the whole reason growing n_p ADDS samples instead of
    moving them — and α is the one parameter that differs between
    insulators and metals while changing nothing about the shapes.
    """
    for drop in MS._SAMPLING_REQUIRED:
        bad = {k: v for k, v in _SAMPLING.items() if k != drop}
        with pytest.raises(ValueError, match=drop):
            MS.omega_grid_digest([0.1j], [0], bad)
    with pytest.raises(ValueError, match="negative line offset"):
        MS.omega_grid_digest([0.1j], [0], dict(_SAMPLING,
                                               varpi=[-0.1, 1.0]))


def test_a_grid_of_the_wrong_length_refuses(tmpdir_path):
    """RED: one ω per slab, and the leading axis IS the frequency axis."""
    tables, verdict, n_mu = _geometry()
    with pytest.raises(ValueError, match="leading axis"):
        MS.allocate_w_omega(
            tmpdir_path, "W_qmunu_omega", n_omega=4,
            n_q_on_disk=_N_Q_IBZ, n_mu=n_mu, tables=tables,
            omega=[0.1j, 0.2j], sampling=_SAMPLING,
            closure_verdict=verdict)


# ---------------------------------------------------------------------------
# Arm 8 — the μ pad does not reach disk, per frequency
# ---------------------------------------------------------------------------

def test_the_pad_is_reconstructed_by_the_reader_not_stored(tmpdir_path):
    """A file written at the LOGICAL extent re-pads to the READER's own.

    Inherited from the q_irr checkpoint and re-asserted through the
    frequency axis, because a layout that gains an axis is exactly where
    a per-axis rule gets applied to the wrong one.  The pad grows the μ
    axes and NOT the frequency axis; the padded rows and columns are
    zero; and the logical block is bit-identical to the unpadded read.
    """
    W, _, _, n_mu, _, _ = _write_w(tmpdir_path, n_omega=4)
    hdr = MS.read_w_header(tmpdir_path, "W_qmunu_omega")
    assert hdr["n_rmu_logical"] == n_mu == hdr["n_mu"]

    padded, _ = MS.read_w_slab(tmpdir_path, "W_qmunu_omega", 1,
                               n_mu_padded=n_mu + 5)
    assert padded.shape == (_N_Q_IBZ, n_mu + 5, n_mu + 5)
    assert np.array_equal(padded[:, :n_mu, :n_mu], W[1])
    assert not padded[:, n_mu:, :].any()
    assert not padded[:, :, n_mu:].any()

    blk = MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 0, [0, 1],
                            n_mu_padded=n_mu + 5)
    assert blk.shape == (4, n_mu + 5, 2), (
        "the ROW axis takes the pad; the columns are a selection the "
        "caller chose and padding them would shift every index in it")
    assert np.array_equal(blk[:, :n_mu, :], W[:, 0][:, :, [0, 1]])
    assert not blk[:, n_mu:, :].any()


def test_padding_down_refuses(tmpdir_path):
    """RED: the pad only ever grows the extent."""
    _, _, _, n_mu, _, _ = _write_w(tmpdir_path, n_omega=4)
    with pytest.raises(ValueError, match="pad DOWN"):
        MS.read_w_slab(tmpdir_path, "W_qmunu_omega", 0,
                       n_mu_padded=n_mu - 1)
    with pytest.raises(ValueError, match="pad the row axis DOWN"):
        MS.read_w_columns(tmpdir_path, "W_qmunu_omega", 0, [0],
                          n_mu_padded=n_mu - 1)


# ---------------------------------------------------------------------------
# Arm 9 — the staged B/Ω store: ledger, finalize, and the partial door
# ---------------------------------------------------------------------------

def _fit_block(n_p, n_rows, n_cols, seed):
    rng = np.random.default_rng(seed)
    Om = (rng.standard_normal((n_p, n_rows, n_cols))
          - 1j * np.abs(rng.standard_normal((n_p, n_rows, n_cols))))
    Bp = (rng.standard_normal((n_p, n_rows, n_cols))
          + 1j * rng.standard_normal((n_p, n_rows, n_cols)))
    diag = {
        "condition": np.abs(rng.standard_normal((n_rows, n_cols))) * 1e3,
        "backward_error": np.abs(
            rng.standard_normal((n_rows, n_cols))) * 1e-12,
    }
    return Om, Bp, diag


def _staged_fit(path, n_q=2, n_mu=6, n_p=3, n_cols=2, stop_after=None,
                energy_unit=None):
    """Walk the schedule, writing blocks; optionally stop early."""
    MS.allocate_fit_store(
        path, n_q=n_q, n_mu=n_mu, n_p=n_p,
        grid_hash="sha256:deadbeef", energy_unit=energy_unit)
    truth = {}
    steps = [(q, lo, hi) for q in range(n_q)
             for lo, hi in tiling.column_blocks(n_mu, n_cols)]
    for k, (q, lo, hi) in enumerate(steps):
        if stop_after is not None and k >= stop_after:
            break
        cols = list(range(lo, hi))
        Om, Bp, diag = _fit_block(n_p, n_mu, len(cols), seed=100 + k)
        MS.write_fit_block(path, q, cols, Om, Bp, diag)
        truth[(q, lo, hi)] = (Om, Bp, diag)
    return truth, steps


def test_b_and_omega_round_trip_with_condition_payload_intact(tmpdir_path):
    """Every staged block comes back exactly with its condition map.

    The diagnostics are the point, not decoration: the Σ stage refuses
    poles that fail certification, and a pole whose condition number and
    backward error nobody recorded can only be trusted.  Condition therefore
    round-trips under the same bit-equality as the poles; backward error is
    reduced into the ledger rather than retained as a full tensor.
    """
    truth, _ = _staged_fit(tmpdir_path)
    MS.finalize_fit_store(
        tmpdir_path, certification=dict(_CERT, condition_max_allowed=1e8))
    for (q, lo, hi), (Om, Bp, diag) in truth.items():
        gOm, gBp, gdiag, ledger = MS.read_fit_block(
            tmpdir_path, q, list(range(lo, hi)))
        assert np.array_equal(gOm, Om), (q, lo, hi)
        assert np.array_equal(gBp, Bp), (q, lo, hi)
        assert np.array_equal(gdiag["condition"], diag["condition"])
        assert set(gdiag) == {"condition"}
        assert ledger["complete"]

    Om, Bp, diag, ledger = MS.read_fit_tensors(tmpdir_path)
    assert Om.shape == (3, 2, 6, 6) and Bp.shape == Om.shape
    assert set(diag) == {"condition"}
    assert ledger["n_done"] == ledger["n_total"] == 2 * 6
    want_cond = max(float(d["condition"].max())
                    for _, _, d in truth.values())
    want_backward = max(float(d["backward_error"].max())
                        for _, _, d in truth.values())
    assert ledger["condition_max"] == pytest.approx(want_cond)
    assert ledger["backward_error_max"] == pytest.approx(want_backward)
    with h5py.File(tmpdir_path, "r") as f:
        assert "fit_condition" in f
        assert "fit_backward_error" not in f
        assert "fit_residual" not in f
        assert "fit_n_valid" not in f
        assert f.attrs["mpa_fit_condition_max"] == pytest.approx(want_cond)
        assert f.attrs["mpa_cert_condition_max_allowed"] == 1e8
        assert f.attrs["mpa_fit_w_grid_hash"] == "sha256:deadbeef"
        assert int(f.attrs["mpa_fit_n_blocks"]) == 2 * 3


def test_pole_slice_owns_units_and_the_existing_q_unfold_tables(tmpdir_path):
    tables, _verdict, n_mu = _geometry()
    MS.allocate_fit_store(
        tmpdir_path, n_q=_N_Q_IBZ, n_mu=n_mu, n_p=1,
        energy_unit="Ha", unfold_tables=tables)
    for q in range(_N_Q_IBZ):
        Om = np.full((1, n_mu, n_mu), 0.7 - 0.1j)
        Bp = np.full((1, n_mu, n_mu), 0.2 + 0.3j)
        diag = {"condition": np.ones((n_mu, n_mu)),
                "backward_error": np.zeros((n_mu, n_mu))}
        MS.write_fit_block(tmpdir_path, q, np.arange(n_mu), Om, Bp, diag)
    MS.finalize_fit_store(tmpdir_path, certification=_CERT)

    ledger = MS.fit_completion_ledger(tmpdir_path)
    assert ledger["energy_unit"] == "Ha"
    assert ledger["q_storage"] == "ibz"
    assert ledger["n_q_full"] == tables.n_q_full
    assert ledger["table_hash"] == tables.canonical().digest()
    Om, Bp = MS.read_poles(tmpdir_path, pole_slice=0, to_unit="Ry")
    Om, Bp = Om[0], Bp[0]
    np.testing.assert_array_equal(Om, 2.0 * (0.7 - 0.1j))
    np.testing.assert_array_equal(Bp, 2.0 * (0.2 + 0.3j))


def test_scalar_head_fit_round_trip_units_and_readiness(tmpdir_path):
    """The tiny head axis converts only its energy-like quantities."""
    MS.allocate_fit_store(
        tmpdir_path, n_q=1, n_mu=2, n_p=1, energy_unit="Ha")
    z = np.asarray([0.0 + 0.1j, 0.7 + 0.1j])
    wc = np.asarray([-14.0 + 0.0j, -3.0 + 0.2j])
    poles = np.asarray([0.8 - 0.06j, 1.4 - 0.2j])
    residues = np.asarray([0.3 + 0.1j, -0.2 + 0.05j])
    MS.write_head_fit(
        tmpdir_path, z, wc, poles, residues, energy_unit="Ha",
        fit_condition=17.0, fit_backward_error=2.0e-12,
        fit_max_abs_residual=4.0e-8, model="fixed_dft_gn")

    got = MS.read_head_fit(tmpdir_path, to_unit="Ry")
    np.testing.assert_array_equal(got["sample_z"], 2.0 * z)
    np.testing.assert_array_equal(got["sample_Wc"], wc)
    np.testing.assert_array_equal(got["Omega_p"], 2.0 * poles)
    np.testing.assert_array_equal(got["B_p"], 2.0 * residues)
    assert got["units"] == {
        "frequency": "Ry", "Wc": "a.u.", "residue": "Ry*a.u."}
    assert got["diagnostics"] == {
        "fit_condition": 17.0,
        "fit_backward_error": 2.0e-12,
        "fit_max_abs_residual": 4.0e-8,
    }

    with h5py.File(tmpdir_path, "a") as f:
        f[MS.MPA_HEAD_SUFFIX].attrs["ready"] = False
    with pytest.raises(ValueError, match="NOT READY"):
        MS.read_head_fit(tmpdir_path)


@pytest.mark.parametrize("solve", ("loewner", "companion", "thiele"))
@pytest.mark.parametrize(
    "head", ("dft_direct", "bgw_q0shift", "bse_resolvent_micro",
             "qsgw_direct", "qsgw_schur"))
def test_pole_solver_head_models_are_certified_store_protocols(head, solve):
    """Every producer model tag must survive the reader allowlist."""
    assert f"{head}_{solve}" in MS._HEAD_FIT_MODELS


def test_collective_head_writer_refuses_protocol_unknown_to_reader():
    """A producer cannot publish a tag the matching reader will reject."""

    one = np.ones(1, dtype=np.complex128)
    with pytest.raises(ValueError, match="matching reader cannot consume"):
        MS.write_head_fit_collective(
            "/not/opened/when/model/is/refused.h5",
            one, one, one, one,
            mesh_xy=None,
            energy_unit="Ry",
            fit_condition=1.0,
            fit_backward_error=0.0,
            fit_max_abs_residual=0.0,
            grid_hash="unused",
            fit_provenance={},
            model="not_a_protocol",
        )


def test_collective_head_writer_closes_all_ledger_readers_before_rank0_write():
    """The all-rank h5py read must finish before rank 0 opens for write."""
    source = inspect.getsource(MS.write_head_fit_collective)
    ledger = source.index("ledger = fit_completion_ledger(dest)")
    handoff = source.index('barrier("mpa_head_ledger_readers_closed")')
    writer = source.index("if process_rank() == 0:")
    assert ledger < handoff < writer


def test_sigma_fit_contract_exposes_identity_and_enforces_certificate(
        tmpdir_path):
    MS.allocate_fit_store(
        tmpdir_path, n_q=1, n_mu=2, n_p=1, energy_unit="Ry",
        grid_hash="grid", table_hash="table", centroid_hash="centroids",
        provenance={"solver": "unit-test"})
    Om = np.full((1, 2, 2), 0.7 - 0.1j)
    Bp = np.ones_like(Om)
    diag = {"condition": np.full((2, 2), 3.0),
            "backward_error": np.full((2, 2), 2.0e-8)}
    MS.write_fit_block(tmpdir_path, 0, [0, 1], Om, Bp, diag)
    ledger = MS.finalize_fit_store(
        tmpdir_path, certification={
            "condition_max_allowed": 4.0,
            "backward_error_max_allowed": 1.0e-7,
        })
    assert ledger["w_grid_hash"] == "grid"
    assert ledger["w_table_hash"] == "table"
    assert ledger["w_centroid_hash"] == "centroids"
    assert ledger["provenance"]["solver"] == "unit-test"
    MS.validate_fit_store(tmpdir_path, expected_identity={
        "w_grid_hash": "grid", "w_table_hash": "table",
        "w_centroid_hash": "centroids"})
    with pytest.raises(ValueError, match="identity mismatch"):
        MS.validate_fit_store(
            tmpdir_path, expected_identity={"w_grid_hash": "wrong"})


def test_finalize_refuses_a_store_sigma_could_never_read(tmpdir_path):
    """RED: the brick is refused at the finalize, not found at Σ.

    Omitting the certification used to SUCCEED and leave a store
    permanently unusable (audit §E.3 item 1): ``validate_fit_store``
    refuses an uncertified store, a second finalize is refused, and every
    writer refuses a finalized one — so nothing could repair it through
    this API.  The refusal covers every shape of the same fault, because
    a half-certified store bricks exactly as hard as an uncertified one:
    absent, empty, one of the pair, present-but-infinite,
    present-but-zero, present-but-not-a-number.
    """
    _staged_fit(tmpdir_path, n_q=1, n_mu=2, n_p=1, n_cols=2,
                energy_unit="Ry")
    with pytest.raises(ValueError) as exc:
        MS.finalize_fit_store(tmpdir_path)
    msg = str(exc.value)
    assert "got   :" in msg and "want  :" in msg and "fix   :" in msg, msg
    assert "condition_max_allowed" in msg, msg
    assert "backward_error_max_allowed" in msg, msg
    for cert in ({},
                 {"condition_max_allowed": 1e8},
                 dict(_CERT, backward_error_max_allowed=np.inf),
                 dict(_CERT, condition_max_allowed=0.0),
                 dict(_CERT, condition_max_allowed="loose")):
        with pytest.raises(ValueError, match="could never read"):
            MS.finalize_fit_store(tmpdir_path, certification=cert)

    # GREEN TWIN, and the point of refusing BEFORE the store is opened:
    # none of that touched the file, so the certified finalize still
    # round trips — through the validator that used to be unreachable.
    ledger = MS.finalize_fit_store(tmpdir_path, certification=_CERT)
    assert ledger["complete"]
    assert ledger["certification"]["condition_max_allowed"] == \
        _CERT["condition_max_allowed"]
    assert MS.validate_fit_store(tmpdir_path)["complete"]


def test_sigma_still_refuses_an_uncertified_store_it_finds_on_disk(
        tmpdir_path):
    """A COMPLETE ledger is not an accuracy certificate.

    ``finalize_fit_store`` can no longer WRITE this file, but a store
    stamped by an older writer can still be on disk — the validator is
    the Σ stage's own gate and does not trust the producer, so it keeps
    its refusal rather than inheriting the new precondition.
    """
    _staged_fit(tmpdir_path, n_q=1, n_mu=2, n_p=1, n_cols=2,
                energy_unit="Ry")
    with h5py.File(tmpdir_path, "a") as f:      # the old writer's stamp
        f.attrs["mpa_fit_complete"] = True
    with pytest.raises(ValueError, match="requires certified pole fits"):
        MS.validate_fit_store(tmpdir_path)


def test_an_unfinalized_store_is_readable_only_through_the_announced_door(
        tmpdir_path):
    """RED TWIN: partial reads must be ASKED for, never inferred.

    An unfitted column reads back as zeros, and a zero pole is not an
    absent pole: B_p = 0 at Ω_p = 0 contributes nothing to
    W(τ) = Σ_p B_p e^{−iΩ_p τ} and therefore looks exactly like a
    converged fit of a genuinely dark screening channel.  So the refusal
    is on the LEDGER and never on the data, and the partial door is a
    keyword the caller had to type.
    """
    truth, steps = _staged_fit(tmpdir_path, stop_after=4)
    with pytest.raises(ValueError, match="NOT FINALIZED"):
        MS.read_fit_tensors(tmpdir_path)
    with pytest.raises(ValueError, match="NOT FINALIZED"):
        MS.read_fit_block(tmpdir_path, 0, [0, 1])

    Om, Bp, diag, ledger = MS.read_fit_tensors(tmpdir_path,
                                               allow_partial=True)
    assert not ledger["complete"]
    assert ledger["n_done"] == 4 * 2
    (q, lo, hi) = steps[0]
    gOm, _, _, _ = MS.read_fit_block(tmpdir_path, q, list(range(lo, hi)),
                                     allow_partial=True)
    assert np.array_equal(gOm, truth[(q, lo, hi)][0])

    # Announced or not, a column that was never fitted still refuses:
    # "the file is incomplete" and "the columns you asked for are
    # incomplete" are different facts.
    with pytest.raises(ValueError, match="are not fitted"):
        MS.read_fit_block(tmpdir_path, 1, [4, 5], allow_partial=True)


def test_finalize_flips_the_gate_and_names_the_gaps(tmpdir_path):
    """RED then GREEN: the refusal says WHICH columns, as ranges.

    "Which columns are missing" is the only question a driver that
    crashed halfway actually has, so the refusal answers it in the form
    a human can act on rather than as a count.
    """
    _staged_fit(tmpdir_path, stop_after=4)
    with pytest.raises(ValueError) as exc:
        MS.finalize_fit_store(tmpdir_path, certification=_CERT)
    msg = str(exc.value)
    assert "4 of 12" in msg, msg
    assert "q=1: columns 2-5" in msg, msg

    # Fill the rest, and the same call now succeeds.
    for lo, hi in tiling.column_blocks(6, 2)[1:]:
        Om, Bp, diag = _fit_block(3, 6, hi - lo, seed=lo)
        MS.write_fit_block(tmpdir_path, 1, list(range(lo, hi)), Om, Bp,
                           diag)
    ledger = MS.finalize_fit_store(tmpdir_path, certification=_CERT)
    assert ledger["complete"] and ledger["finalized_utc"]
    MS.read_fit_tensors(tmpdir_path)


def test_a_second_finalize_refuses(tmpdir_path):
    """RED TWIN of the finalize: one stamp, one measured state.

    A second finalize stamps completeness against a state the first one
    did not measure.  If blocks were written since, they went into a
    file that already claimed to be done — which is the bug, not the fix
    — and the writer refuses that too.
    """
    _staged_fit(tmpdir_path)
    MS.finalize_fit_store(tmpdir_path, certification=_CERT)        # TWIN
    with pytest.raises(ValueError, match="already finalized"):
        MS.finalize_fit_store(tmpdir_path, certification=_CERT)
    Om, Bp, diag = _fit_block(3, 6, 2, seed=1)
    with pytest.raises(ValueError, match="FINALIZED"):
        MS.write_fit_block(tmpdir_path, 0, [0, 1], Om, Bp, diag)


def test_a_scattered_column_block_writes_and_reads_the_same_bytes(
        tmpdir_path):
    """The walk emits contiguous blocks; the store does not require it.

    ``fit_schedule`` only ever hands out contiguous ranges — one HDF5
    hyperslab — but a caller retrying the columns a previous pass could
    not fit will hand a scattered set, and that path goes through a
    point selection instead.  Same bytes down both, which is the thing
    nobody checks and which the ledger (not the journal's span) is the
    authority on.
    """
    MS.allocate_fit_store(tmpdir_path, n_q=1, n_mu=6, n_p=2)
    Om, Bp, diag = _fit_block(2, 6, 3, seed=21)
    MS.write_fit_block(tmpdir_path, 0, [0, 3, 5], Om, Bp, diag)
    ledger = MS.fit_completion_ledger(tmpdir_path)
    assert ledger["blocks_done"][0].tolist() == [1, 0, 0, 1, 0, 1]
    assert ledger["journal"].tolist() == [[0, 0, 6]], (
        "the journal records the SPAN; blocks_done records the truth")
    gOm, gBp, gdiag, _ = MS.read_fit_block(tmpdir_path, 0, [0, 3, 5],
                                           allow_partial=True)
    assert np.array_equal(gOm, Om) and np.array_equal(gBp, Bp)
    assert np.array_equal(gdiag["condition"], diag["condition"])
    with pytest.raises(ValueError, match="are not fitted"):
        MS.read_fit_block(tmpdir_path, 0, [1, 2], allow_partial=True)


def test_a_block_written_twice_refuses(tmpdir_path):
    """RED: the journal must not carry two entries for one column.

    Rewriting a fitted block would replace poles the ledger already
    certified, and the journal would hold two condition numbers for one
    column with no rule saying which one the diagnostics belong to.
    """
    _staged_fit(tmpdir_path, stop_after=1)
    Om, Bp, diag = _fit_block(3, 6, 2, seed=9)
    with pytest.raises(ValueError, match="already fitted"):
        MS.write_fit_block(tmpdir_path, 0, [0, 1], Om, Bp, diag)


def test_the_diagnostics_are_required_and_must_be_finite(tmpdir_path):
    """RED: no condition number, no write; and NaN is not a number.

    ``NaN > tol`` is False, so a NaN condition number written as data
    would pass the Σ stage's threshold comparison silently — the pole it
    describes would be certified by the absence of evidence.
    """
    MS.allocate_fit_store(tmpdir_path, n_q=1, n_mu=4, n_p=2)
    Om, Bp, diag = _fit_block(2, 4, 2, seed=5)
    with pytest.raises(ValueError, match="backward_error"):
        MS.write_fit_block(tmpdir_path, 0, [0, 1], Om, Bp,
                           {"condition": diag["condition"]})
    nan = dict(diag)
    nan["condition"] = diag["condition"].copy()
    nan["condition"][1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        MS.write_fit_block(tmpdir_path, 0, [0, 1], Om, Bp, nan)
    with pytest.raises(ValueError, match="expected \\(4, 2\\)"):
        MS.write_fit_block(tmpdir_path, 0, [0, 1], Om, Bp,
                           {"condition": np.zeros((4, 3)),
                            "backward_error": np.zeros((4, 3))})
    MS.write_fit_block(tmpdir_path, 0, [0, 1], Om, Bp, diag)       # TWIN


def test_extra_diagnostics_are_validated_but_not_persisted(tmpdir_path):
    """Ephemeral fit-health arrays cannot expand the production schema."""
    MS.allocate_fit_store(tmpdir_path, n_q=1, n_mu=4, n_p=2)
    Om, Bp, diag = _fit_block(2, 4, 4, seed=6)
    diag["n_poles_pruned"] = np.full((4, 4), 2.0)
    MS.write_fit_block(tmpdir_path, 0, list(range(4)), Om, Bp, diag)
    MS.finalize_fit_store(tmpdir_path, certification=_CERT)
    _, _, got, _ = MS.read_fit_block(tmpdir_path, 0, list(range(4)))
    assert set(got) == {"condition"}
    with h5py.File(tmpdir_path, "r") as f:
        assert "fit_n_poles_pruned" not in f


def test_ephemeral_diagnostics_may_change_without_changing_schema(tmpdir_path):
    """Only the fixed condition payload crosses blocks; extras are ephemeral."""
    MS.allocate_fit_store(tmpdir_path, n_q=1, n_mu=4, n_p=2)
    Om, Bp, diag = _fit_block(2, 4, 2, seed=7)
    MS.write_fit_block(tmpdir_path, 0, [0, 1], Om, Bp,
                       dict(diag, held_out=np.zeros((4, 2))))
    MS.write_fit_block(tmpdir_path, 0, [2, 3], Om, Bp, diag)
    MS.finalize_fit_store(tmpdir_path, certification=_CERT)
    _, _, got, _ = MS.read_fit_tensors(tmpdir_path)
    assert set(got) == {"condition"}


def test_the_ledger_records_ranges_and_their_worst_diagnostics(
        tmpdir_path):
    """The journal answers "what did the block that did it look like".

    ``blocks_done`` says whether a column is done; the journal says what
    the block that did it measured.  The Σ stage's certification needs
    the second, and a per-file scalar would hide the block that was
    worst.
    """
    truth, steps = _staged_fit(tmpdir_path, n_q=2, n_mu=6, n_p=3,
                               n_cols=3)
    ledger = MS.fit_completion_ledger(tmpdir_path)
    assert ledger["journal"].shape == (len(steps), 3)
    assert [tuple(r[:2]) for r in ledger["journal"].tolist()] == \
        [(q, lo) for q, lo, _ in steps]
    for k, (q, lo, hi) in enumerate(steps):
        _, _, diag = truth[(q, lo, hi)]
        assert ledger["block_condition_max"][k] == pytest.approx(
            float(diag["condition"].max()))
        assert ledger["block_backward_error_max"][k] == pytest.approx(
            float(diag["backward_error"].max()))


# ---------------------------------------------------------------------------
# Arm 10 — the closure prerequisite still holds, through the ω axis
# ---------------------------------------------------------------------------

def test_the_writer_still_refuses_a_non_closed_centroid_set(tmpdir_path):
    """RED: a wedge with no permutation α is unrecoverable AT EVERY ω.

    The frequency axis multiplies the damage rather than diluting it —
    n_omega slabs of a tensor that cannot be unfolded — so the landed
    refusal is re-asserted through this door to prove the delegation is
    real and not a path that skipped it.
    """
    tables, _, n_mu = _geometry()
    # BREAK IT ON THE GRID, not off it.  A perturbed centroid is
    # incommensurate with the FFT grid and ``verify_centroid_orbit_
    # closure`` diagnoses that separately — correctly, since a rounded
    # image that cannot land on a grid point says nothing about the
    # GROUP.  So the break here is the honest one: delete one member of
    # a two-point orbit, leaving its partner with nowhere to map.
    cent = _closed_centroid_set()
    rinv = np.rint(np.linalg.inv(np.asarray(_SYMS[1], float))).astype(int)
    tint = np.rint(_TNP[1] / (2.0 * np.pi) * _FFT).astype(np.int64)
    img = (rinv @ cent.T).T + tint
    moved = np.flatnonzero(
        np.any((img % _FFT) != (cent % _FFT), axis=1))
    assert moved.size, "the glide fixes every centroid; pick other seeds"
    broken = np.delete(cent, moved[0], axis=0).astype(np.float64) / _FFT
    verdict = verify_centroid_orbit_closure(broken, _SYMS, tnp=_TNP,
                                            fft_grid=_FFT)
    assert not verdict.closed, verdict.describe()
    with pytest.raises(RuntimeError, match="NOT CLOSED"):
        MS.allocate_w_omega(
            tmpdir_path, "W_qmunu_omega", n_omega=2,
            n_q_on_disk=_N_Q_IBZ, n_mu=n_mu, tables=tables,
            omega=[0.1j, 1.0j], sampling=_SAMPLING,
            closure_verdict=verdict)
    # AND IT LEFT NOTHING BEHIND.  A refused allocation that dropped a
    # correctly-shaped dataset on disk would be the all-zero-screening
    # hazard wearing this format's clothes.
    assert not os.path.exists(tmpdir_path) or _is_empty(tmpdir_path)


def test_a_tensor_without_its_omega_group_refuses(tmpdir_path):
    """RED: the grid lives BESIDE the tensor or the file is not readable.

    Same rule the unfold tables have, for the same reason: a tensor
    whose sampling grid lives in another file is a tensor that silently
    decays when the protocol is regenerated.
    """
    _write_w(tmpdir_path, n_omega=4)
    with h5py.File(tmpdir_path, "a") as f:
        del f["W_qmunu_omega" + MS.MPA_GROUP_SUFFIX]
    with pytest.raises(ValueError, match="carries no"):
        MS.read_w_header(tmpdir_path, "W_qmunu_omega")


def test_the_column_helpers_refuse_a_malformed_request(tmpdir_path):
    """RED cluster on the column argument itself."""
    _, _, _, n_mu, _, _ = _write_w(tmpdir_path, n_omega=4)
    with pytest.raises(ValueError, match="repeats a column") as exc:
        MS.normalise_columns([1, 1, 2, 5, 5], n_mu)
    # THE COLUMNS THAT REPEAT, under the label "repeats a column".  It
    # used to name the first four DISTINCT columns, which sends a reader
    # debugging a duplicate to three innocent indices (audit §E.3 item 11).
    assert "[1, 5]" in str(exc.value), str(exc.value)
    with pytest.raises(IndexError, match="logical columns"):
        MS.normalise_columns([0, n_mu], n_mu)
    with pytest.raises(TypeError, match="boolean mask"):
        MS.normalise_columns(np.ones(n_mu, dtype=bool), n_mu)
    with pytest.raises(ValueError, match="empty"):
        MS.normalise_columns([], n_mu)
    with pytest.raises(IndexError, match="outside"):
        MS.read_w_columns(tmpdir_path, "W_qmunu_omega", _N_Q_IBZ, [0])
    with pytest.raises(IndexError, match="outside"):
        MS.read_w_slab(tmpdir_path, "W_qmunu_omega", 99)


def test_a_slab_of_the_wrong_shape_refuses(tmpdir_path):
    """RED: the wedge and the μ extent are the same at every ω."""
    W, _, _, n_mu, _, _ = _write_w(tmpdir_path, n_omega=4)
    with pytest.raises(ValueError, match="per frequency"):
        MS.write_w_slab(tmpdir_path, "W_qmunu_omega", 0,
                        W[0][:, :n_mu - 1, :n_mu - 1])


# ---------------------------------------------------------------------------
# Arm 11 — EVERY h5py open this module makes is declared
# ---------------------------------------------------------------------------

def test_the_table_read_is_declared_to_the_registry(
        tmpdir_path, tmp_path, monkeypatch):
    """RED: ``read_w_tables`` was the ONE h5py open nobody could see.

    It forwarded ``src`` straight to ``qirr_store.read_tables``, which
    opens h5py itself, so the open was never declared to
    :mod:`file_io.hdf5_owner` — and it is not a rare path: the fit driver
    calls it on EVERY RANK to fetch the unfold tables (audit §E.3 item 2).
    A blind spot there is a blind spot exactly where the store is
    hottest, and neither the refusal nor the count could fire.

    THE SYNTHETIC TWO-STACK SHAPE, as in ``test_hdf5_one_owner``: the FFI
    side is a real ``note_open`` claim rather than a real phdf5 handle,
    because what is under test is the DECLARATION, and faking the
    transport must not fake the ownership.  Both instruments are asserted
    — the registry (it refuses, and the refusal names this function) and
    the journal (the allowed read leaves an h5py open line on this path,
    in the one stream that carries both stacks).
    """
    from file_io import h5_journal as J
    from file_io import hdf5_owner as HO

    _write_w(tmpdir_path, n_omega=2)
    monkeypatch.setenv(J.MODE_ENV, "1")
    monkeypatch.setenv(J.DIR_ENV, str(tmp_path))
    monkeypatch.delenv(HO.POLICY_ENV, raising=False)
    HO.reset_for_test()
    J.reset_for_test()
    try:
        # 1. THE REGISTRY.  With the FFI holding a write handle on this
        #    path, a serial-h5py read is the undefined cross-stack case
        #    A1 names — and the refusal carries the function, so a later
        #    reader of the message knows which open to move.
        token = HO.note_open(tmpdir_path, HO.STACK_FFI, "a",
                             where="test: SlabIO(mode='a')")
        try:
            with pytest.raises(RuntimeError) as exc:
                MS.read_w_tables(tmpdir_path, "W_qmunu_omega")
            msg = str(exc.value)
            assert "one-owner-per-file" in msg, msg
            assert "mpa_store.read_w_tables" in msg, msg
        finally:
            HO.note_close(tmpdir_path, token)

        # 2. THE JOURNAL.  The allowed read is COUNTED: the tables still
        #    come back from the format layer, and the open is on the line.
        tables = MS.read_w_tables(tmpdir_path, "W_qmunu_omega")
        assert tables.canonical().digest()
        with open(J.journal_path()) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        want = f"stack=h5py op=open path={os.path.abspath(tmpdir_path)} "
        assert [ln for ln in lines if want in ln], lines
    finally:
        HO.reset_for_test()
        J.reset_for_test()


if __name__ == "__main__":                              # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
