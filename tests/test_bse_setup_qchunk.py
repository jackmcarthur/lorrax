"""``compute_wfns_fi``'s q-CHUNK and its two eigenvalue paths.

Covers the three things the fine-grid chunking contract promises:

1. the chunk width is an INPUT KEY (``wfn_fi_q_chunk``) whose default is
   ``N_q_co = prod(kgrid_co)`` — not a bare constant;
2. ``use_low_mem_eigh`` selects the distributed (face-sharded) eigh and
   REFUSES when it cannot be honoured, never demoting to the whole-matrix
   path it exists to avoid;
3. eigenvalues are accumulated to the FULL band count across chunks
   (``.lam_all_fi``), while ψ/coeffs stay windowed.

INSTRUMENT DISCIPLINE (README §5.1).  The value comparator used by the
chunk-invariance cells is exercised on a PLANTED deviation in
``test_chunk_invariance_comparator_goes_red`` — a chunk-invariance PASS is
only informative because that cell shows the same comparator failing.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
#  fixtures / helpers
# ---------------------------------------------------------------------------

def _mesh_1x1():
    import jax
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _mesh_cpu_1x1():
    import jax
    from jax.sharding import Mesh
    devs = jax.devices("cpu")
    return Mesh(np.asarray(devs[:1]).reshape(1, 1), ("x", "y"))


def _synthetic(nk_grid=(2, 2, 1), nb=4, rank=32, n_mu=6, ns=2, seed=11):
    """A structurally real (ctilde, enk, B) triple — band-orthonormal
    ``ctilde`` per k, exactly as ``streaming_galerkin_solve`` produces (see
    ``test_ffi_linalg_contract._synthetic_htransform``, same construction)."""
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


def _run(mesh, *, kgrid_fi=(4, 4, 1), band_window=(1, 3), log=None, **kw):
    import jax.numpy as jnp
    from bandstructure.bse_setup import compute_wfns_fi
    ct, enk, B, kgrid_co = _synthetic()
    with mesh:
        return compute_wfns_fi(
            ctilde=jnp.asarray(ct), B_at_mu=jnp.asarray(B),
            enk_sigma=jnp.asarray(enk), kgrid_co=kgrid_co,
            kgrid_fi=kgrid_fi, band_window_fi=band_window, mesh_xy=mesh,
            log_fn=log, **kw)


def _maxdiff(a, b):
    """THE comparator these cells gate on.  Shown going red in
    ``test_chunk_invariance_comparator_goes_red``."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _phase_aligned_maxdiff(a, b, *, band_axis=1):
    """Compare per-q band vectors modulo each eigenvector's scalar phase."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    if not a.size:
        return 0.0
    # ``psi_rmu_Y`` carries the band at axis 1; the canonical transposed
    # bundle and raw coefficient matrix carry it at axis 2.  Put (q, band)
    # first before flattening the physical vector components; flattening the
    # first two axes blindly would align one phase per centroid/basis row.
    af = np.moveaxis(a, band_axis, 1).reshape(a.shape[0], a.shape[band_axis], -1)
    bf = np.moveaxis(b, band_axis, 1).reshape(b.shape[0], b.shape[band_axis], -1)
    overlap = np.sum(np.conj(af) * bf, axis=-1)
    phase = np.ones_like(overlap)
    nonzero = np.abs(overlap) > 0.0
    phase[nonzero] = np.conj(overlap[nonzero]) / np.abs(overlap[nonzero])
    aligned = bf * phase[..., None]
    return float(np.max(np.abs(af - aligned)))


# ---------------------------------------------------------------------------
#  1. the chunk width and its default
# ---------------------------------------------------------------------------

def test_q_chunk_defaults_to_n_q_co():
    """No ``batch_size`` → the chunk is N_q_co, announced as such.

    The old default was a literal 32, which on this fixture (N_q_co = 4)
    would build a chunk 8x wider than the coarse Hamiltonian the deck has
    already had to afford.
    """
    pytest.importorskip("jax")
    lines = []
    out = _run(_mesh_1x1(), log=lines.append)
    banner = " ".join(lines)
    assert "q-chunk=4" in banner, banner
    assert "default N_q_co=4" in banner, banner
    assert out.psi_rmu_Y.shape[0] == 16          # 4x4x1 fine grid


def test_q_chunk_explicit_key_is_honoured():
    pytest.importorskip("jax")
    lines = []
    _run(_mesh_1x1(), batch_size=8, log=lines.append)
    banner = " ".join(lines)
    assert "q-chunk=8" in banner and "wfn_fi_q_chunk=8" in banner, banner


def test_q_chunk_zero_and_none_both_mean_the_default():
    """``wfn_fi_q_chunk = 0`` is the input file's spelling of "unset"; it
    must not become a zero-width chunk (an infinite loop) or a 1."""
    pytest.importorskip("jax")
    for spelling in (0, None, -1):
        lines = []
        _run(_mesh_1x1(), batch_size=spelling, log=lines.append)
        assert "q-chunk=4" in " ".join(lines), (spelling, lines)


def test_kgrid_co_inconsistent_with_ctilde_is_refused():
    """The default reads N_q_co off ``kgrid_co``; a kgrid_co that does not
    match ctilde's k-axis would silently pick the wrong default (and
    build_fH_R would ifft-reshape into the wrong grid)."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from bandstructure.bse_setup import compute_wfns_fi
    ct, enk, B, _ = _synthetic()
    mesh = _mesh_1x1()
    with pytest.raises(ValueError, match="N_q_co"):
        with mesh:
            compute_wfns_fi(
                ctilde=jnp.asarray(ct), B_at_mu=jnp.asarray(B),
                enk_sigma=jnp.asarray(enk), kgrid_co=(3, 3, 1),
                kgrid_fi=(4, 4, 1), band_window_fi=(1, 3), mesh_xy=mesh)


# ---------------------------------------------------------------------------
#  2. values do not depend on the chunk width  (+ the comparator's red twin)
# ---------------------------------------------------------------------------

_CHUNKS = [1, 2, 4, 5, 8, 16, 32]


#: The chunk width is a 1-ULP axis, NOT a bit-exact one.  MEASURED, job
#: 7883840: on a 1x1 mesh at rank 32, ``lam_fi`` from a 1-wide chunk and a
#: 2-wide chunk differ by 2.220446e-16 — one ULP at |lam| ~ 1.  Nothing in
#: the loop is order-dependent (every q is independent; the Fourier sum, the
#: eigh and the projection are per-q; the trailing pad rows are sliced off),
#: so the difference is inside ``jnp.linalg.eigh`` itself, which selects a
#: different LAPACK/XLA kernel by BATCH SIZE.  That is the same class as the
#: cross-P reduction-blocking difference this deck already pins at 0.740 meV,
#: four orders below it.  This tolerance is stated in Ry on the recovered
#: energies — the quantity the physics gate reads — and is 1e-11 eV, i.e. 8
#: orders below the 1 meV .dat tolerance.
CHUNK_TOL_RY = 1.0e-12


def test_values_are_invariant_to_the_chunk_width():
    """A chunk boundary is a LOOP boundary, not an algorithmic one.

    Gated at :data:`CHUNK_TOL_RY` rather than at 0.0 because the eigh kernel
    is batch-size dependent — see the note on that constant.  The ACTUAL
    spread is printed by ``test_chunk_width_ulp_spread_is_reported`` so the
    tolerance can be checked against the measurement instead of trusted.
    """
    pytest.importorskip("jax")
    mesh = _mesh_1x1()
    ref = _run(mesh, batch_size=_CHUNKS[0], return_coeffs=True)
    for bs in _CHUNKS[1:]:
        got = _run(mesh, batch_size=bs, return_coeffs=True)
        assert _maxdiff(ref.enk_full, got.enk_full) < CHUNK_TOL_RY, bs
        assert _maxdiff(ref.lam_fi, got.lam_fi) < CHUNK_TOL_RY, bs
        assert _maxdiff(ref.lam_all_fi, got.lam_all_fi) < CHUNK_TOL_RY, bs
        assert _phase_aligned_maxdiff(
            ref.psi_rmu_Y, got.psi_rmu_Y) < 1e-10, bs
        assert _phase_aligned_maxdiff(
            ref.psi_rmuT_X, got.psi_rmuT_X, band_axis=2) < 1e-10, bs
        assert _phase_aligned_maxdiff(
            ref.coeffs_fi, got.coeffs_fi, band_axis=2) < 1e-10, bs


def test_chunk_width_ulp_spread_is_reported(capsys):
    """Print the FULL width-vs-width difference table.

    The batch-size-selected eigensolver may move the recovered eigenvalues by
    a few ULPs.  The production contract is the stated energy tolerance, not
    byte identity between implementation-dependent batched kernels.
    """
    pytest.importorskip("jax")
    mesh = _mesh_1x1()
    outs = {bs: _run(mesh, batch_size=bs) for bs in _CHUNKS}
    ref = outs[_CHUNKS[-1]]
    rows = []
    for bs in _CHUNKS:
        d_lam = _maxdiff(ref.lam_fi, outs[bs].lam_fi)
        d_e = _maxdiff(ref.enk_full, outs[bs].enk_full)
        rows.append((bs, d_lam, d_e))
    with capsys.disabled():
        print(f"\n  chunk-width spread vs bs={_CHUNKS[-1]} (rank 32, 1x1 mesh):")
        for bs, d_lam, d_e in rows:
            print(f"    bs={bs:<4} max|d lam| = {d_lam:.6e}   "
                  f"max|d enk| = {d_e:.6e} Ry"
                  f"{'   (bit-identical)' if d_lam == 0.0 else ''}")
    assert max(r[1] for r in rows) < CHUNK_TOL_RY
    # …and the comparator is not merely reading the same object twice.  A
    # plant must exceed the ULP to exist at all: +1e-18 on values of
    # magnitude ~0.4 (ULP ~5.5e-17) is a NO-OP and made this very line
    # assert 0.0 > 0.0 — a red twin that was itself void (job 7883854).
    planted = np.array(ref.lam_fi)
    planted.flat[0] = np.nextafter(planted.flat[0], np.inf)
    assert _maxdiff(ref.lam_fi, planted) > 0.0


def test_chunk_invariance_comparator_goes_red():
    """THE RED TWIN.  ``_maxdiff`` must report a planted deviation.

    Without this cell, ``== 0.0`` above could be reading two references to
    the same array, an empty array, or a comparator that always returns 0 —
    all of which have happened in this project (README §5.1).
    """
    pytest.importorskip("jax")
    mesh = _mesh_1x1()
    ref = _run(mesh, batch_size=2)
    planted = np.array(ref.lam_fi)
    planted[0, 0] += 1e-13          # a hundred times below the eV tolerances
    assert _maxdiff(ref.lam_fi, planted) > 0.0
    psi = np.array(ref.psi_rmu_Y)
    psi[-1, -1, -1, -1] += 1e-13
    assert _maxdiff(ref.psi_rmu_Y, psi) > 0.0
    # and it must NOT be fooled by a shape change
    with pytest.raises(AssertionError):
        _maxdiff(ref.lam_fi, np.asarray(ref.lam_fi)[:-1])


def test_q_count_not_a_multiple_of_the_chunk():
    """The pad-and-slice path: 16 fine q with a 5-wide chunk (4 chunks, 4
    padded rows discarded).  Same values as an exact division."""
    pytest.importorskip("jax")
    mesh = _mesh_1x1()
    a = _run(mesh, batch_size=4)
    b = _run(mesh, batch_size=5)
    assert a.lam_fi.shape == b.lam_fi.shape == (16, 2)
    assert _maxdiff(a.lam_fi, b.lam_fi) < CHUNK_TOL_RY
    assert _maxdiff(a.psi_rmu_Y, b.psi_rmu_Y) < 1e-10


# ---------------------------------------------------------------------------
#  3. eigenvalues accumulated to the FULL band count across chunks
# ---------------------------------------------------------------------------

def test_lam_all_fi_is_the_full_spectrum_and_agrees_with_the_window():
    """``.lam_all_fi`` is (nq, rank) — every band — and its band-window
    slice is BIT-identical to ``.lam_fi``, i.e. the two are one solve, not
    two."""
    pytest.importorskip("jax")
    ct, _, _, _ = _synthetic()
    rank = ct.shape[2]
    out = _run(_mesh_1x1(), batch_size=3, band_window=(1, 3))
    assert out.lam_all_fi.shape == (16, rank)
    assert _maxdiff(np.asarray(out.lam_all_fi)[:, 1:3], out.lam_fi) == 0.0
    # ascending within each q (jnp.linalg.eigh's contract, and what the
    # band-window slice relies on to mean "the lowest nb_fi at/above b_min")
    lam = np.asarray(out.lam_all_fi)
    assert np.all(np.diff(lam, axis=1) >= -1e-12)


def test_lam_all_fi_survives_a_chunk_boundary():
    """Accumulation across chunks, not just within one: a 3-wide chunk over
    16 q writes six chunks and the last is padding-only in part."""
    pytest.importorskip("jax")
    mesh = _mesh_1x1()
    one = _run(mesh, batch_size=32)      # one chunk covers everything
    many = _run(mesh, batch_size=3)      # six chunks
    assert _maxdiff(one.lam_all_fi, many.lam_all_fi) < CHUNK_TOL_RY


# ---------------------------------------------------------------------------
#  4. use_low_mem_eigh — selection, and refusal without fallback
# ---------------------------------------------------------------------------

def test_use_low_mem_eigh_with_off_is_a_contradiction():
    pytest.importorskip("jax")
    with pytest.raises(ValueError, match="contradiction"):
        _run(_mesh_1x1(), use_low_mem_eigh=True, eigh_backend="off")


def test_use_low_mem_eigh_refuses_rather_than_running_native(monkeypatch):
    """NO SILENT FALLBACK.

    ``slate`` eigh on a CPU mesh is refused unconditionally by
    ``distrib_la.resolve_backend`` (bug L-2: SLATE's host heev SIGSEGVs).
    Asking
    for it under ``use_low_mem_eigh`` must therefore RAISE — the failure
    mode this guards against is the opposite: quietly computing the answer
    with the q-batched native eigh, which is the whole-matrix path the flag
    says will not fit, and reporting success.
    """
    jax = pytest.importorskip("jax")
    try:
        mesh = _mesh_cpu_1x1()
    except (RuntimeError, IndexError):        # no CPU devices addressable
        pytest.skip("no CPU device for the host-mesh refusal cell")

    # Isolate the eigensolver resolver this cell specifies.  The real
    # ``build_fH_R`` first calls the mandatory FFT FFI, making the test depend
    # on a locally built host library and fail before it reaches the refusal.
    import jax.numpy as jnp
    import bandstructure.bse_setup as setup

    def _fH_without_fft(ctilde, enk_sigma, kgrid_co, mesh_xy, **kwargs):
        del mesh_xy, kwargs
        nk = int(np.prod(kgrid_co))
        rank = int(ctilde.shape[-1])
        nb = int(enk_sigma.shape[0])
        return (jnp.zeros((nk, rank, rank), dtype=ctilde.dtype),
                (1.0, 2.0, 1.0), -np.ones((nb, nk), dtype=np.float64), None)

    monkeypatch.setattr(setup, "build_fH_R", _fH_without_fft)
    with pytest.raises(RuntimeError) as exc:
        _run(mesh, use_low_mem_eigh=True, eigh_backend="slate")
    msg = str(exc.value)
    assert "use_low_mem_eigh=True could not be honoured" in msg, msg
    assert "REJECTED on CPU" in msg, msg          # the underlying guard
    assert "No fallback" in msg, msg


def test_use_low_mem_eigh_keeps_an_explicitly_named_library():
    """The flag is a rename onto ``distributed``; it must not overwrite a
    library the deck named.  Checked at the resolve seam so the cell needs
    no FFI build."""
    from gw.gw_config import resolve_eigh_backend
    assert resolve_eigh_backend(
        {"eigh_backend": "scalapack", "use_low_mem_eigh": True}) == "scalapack"
    assert resolve_eigh_backend(
        {"eigh_backend": "cusolvermp", "use_low_mem_eigh": True}) == "cusolvermp"


# ---------------------------------------------------------------------------
#  4b. …and the two raw-params drivers actually CALL the resolver
# ---------------------------------------------------------------------------
#
# ``use_low_mem_eigh`` was a key that parsed, defaulted, validated, and was
# read by nobody on ``bandstructure.htransform`` and ``bse.exciton_bands``:
# both spelled the CLI-over-deck precedence inline
# (``args.X if args.X is not None else params.get("eigh_backend")``) and
# never called ``resolve_eigh_backend`` at all.  Grepping for the key
# proved nothing — it was there, in the defaults table and in the parser.
# These cells perturb it and watch the RESOLVED value move, which is the
# only evidence that distinguishes a live knob from a stored one.

def test_the_cli_override_still_folds_in_the_low_memory_intent():
    """CLI names the LIBRARY, the deck key names the INTENT, one resolver.

    RED ARM: an implementation that let ``override`` bypass the
    combination (an early ``return override``) passes the first two
    asserts and fails the third — ``off`` under the low-memory flag is a
    contradiction whichever spelling asked for it.
    """
    from gw.gw_config import resolve_eigh_backend
    deck = {"eigh_backend": "auto", "use_low_mem_eigh": False}
    assert resolve_eigh_backend(deck, override=None) == "auto"
    assert resolve_eigh_backend(deck, override="slate") == "slate"
    low = {"eigh_backend": "auto", "use_low_mem_eigh": True}
    assert resolve_eigh_backend(low, override=None) == "distributed"
    assert resolve_eigh_backend(low, override="scalapack") == "scalapack"
    with pytest.raises(ValueError, match="contradiction"):
        resolve_eigh_backend(low, override="off")


#: The two drivers that read a raw params dict instead of a LorraxConfig.
#: Read as TEXT, never imported: ``bandstructure.htransform`` runs the
#: whole runtime bootstrap at import (``initialize_communicator_stack``
#: at module scope) and refuses without an FFI ``.so``, so an
#: import-based cell would be a skip on every laptop and on the WSL leg —
#: i.e. exactly the "green because it never ran" shape this key already
#: fell into once.
_RAW_PARAMS_DRIVERS = ("src/bandstructure/htransform.py",
                       "src/bse/exciton_bands.py")


def _driver_source(rel: str) -> str:
    import pathlib
    return (pathlib.Path(__file__).resolve().parents[1] / rel).read_text()


@pytest.mark.parametrize("driver", _RAW_PARAMS_DRIVERS)
def test_the_raw_params_drivers_resolve_the_eigh_axis(driver):
    """Both drivers must REACH ``resolve_eigh_backend``, not re-implement it.

    RED ARM: ``forbidden`` is the pre-replumb spelling verbatim, so this
    cell fails on the tree it was written against — which is what makes
    the green mean something.
    """
    src = _driver_source(driver)
    assert "resolve_eigh_backend(" in src, (
        f"{driver} does not call gw_config.resolve_eigh_backend, so its "
        f"``use_low_mem_eigh`` deck key is parsed and never read")
    forbidden = 'else str(params.get("eigh_backend", "auto"))'
    assert forbidden not in src, (
        f"{driver} still inlines the CLI-over-deck precedence "
        f"({forbidden!r}), which is the spelling that skipped the "
        f"low-memory intent entirely")
    assert "use_low_mem_eigh" in src, (
        f"{driver} must thread the low-memory INTENT to compute_wfns_fi; "
        f"the resolved library name alone leaves bse_setup's no-fallback "
        f"refusal disarmed")


@pytest.mark.parametrize("driver", _RAW_PARAMS_DRIVERS)
def test_the_cli_eigh_vocabulary_is_the_resolvers_own(driver):
    """The two ``--eigh-backend`` flags accepted auto|off|cusolvermp|slate,
    a hand-copied tuple missing ``distributed`` and ``scalapack`` — i.e.
    the CPU distributed eigh was unreachable from either CLI, on the one
    platform where ``distributed`` is the ONLY distributed eigh there is.
    """
    src = _driver_source(driver)
    assert 'choices=("auto", "off", "cusolvermp", "slate")' not in src, (
        f"{driver} still hard-codes the stale eigh vocabulary")
    assert "choices=eigh_backend_choices()" in src, (
        f"{driver}'s --eigh-backend must read the resolver's own list")


# ---------------------------------------------------------------------------
#  5. the input keys themselves
# ---------------------------------------------------------------------------

def test_eigh_backend_vocabulary_is_the_resolvers_own():
    """gw_config's fallback list must equal the distrib_la door's.

    They HAD drifted — the parser accepted only auto|off|cusolvermp|slate
    while the resolver had grown ``distributed`` and ``scalapack``, so the
    low-memory eigh was unrequestable through an input file on a host mesh.
    """
    pytest.importorskip("jax")
    from ffi import _services
    _services.ensure_on_path()
    from distrib_la import BACKEND_CHOICES
    from gw.gw_config import eigh_backend_choices
    assert set(eigh_backend_choices()) == set(BACKEND_CHOICES["eigh"])
    # …and the hard-coded fallback (used when ffi cannot be imported) too.
    import inspect
    from gw import gw_config
    src = inspect.getsource(gw_config.eigh_backend_choices)
    for name in BACKEND_CHOICES["eigh"]:
        assert f'"{name}"' in src, f"fallback tuple is missing {name!r}"


def test_the_vocabulary_came_from_the_resolver_and_not_the_fallback():
    """The equality above cannot see which branch produced the answer.

    ARM B, SEAM 7.  The literal and the door's list are EQUAL today, so
    ``set(a) == set(b)`` passes identically whether ``gw_config`` imported
    the resolver or its ``except`` quietly caught a ModuleNotFoundError and
    returned the frozen tuple.  On a tree where the door IS importable —
    which is what the assert above just established — taking the fallback
    is a defect, and this is the observable that says so.

    RED ARM: with ``ffi._services`` shadowed by a module that raises on
    import, ``EIGH_CHOICES_SOURCE`` reads ``fallback (ImportError: …)`` and
    this assert fires while the equality above still passes — which is the
    whole point of the pair.
    """
    pytest.importorskip("jax")
    from ffi import _services
    _services.ensure_on_path()
    import distrib_la                                        # noqa: F401
    from gw import gw_config
    gw_config.eigh_backend_choices()
    assert gw_config.EIGH_CHOICES_SOURCE == "distrib_la.BACKEND_CHOICES", (
        f"the deck parser answered from its frozen literal on a tree where "
        f"distrib_la imports fine: {gw_config.EIGH_CHOICES_SOURCE}")


def test_the_vocabulary_source_check_can_fail(monkeypatch):
    """The FALSE case of the cell above, constructed.

    Break the import the way a bad path would break it and watch BOTH
    halves: the source label must say ``fallback``, and the returned tuple
    must still equal the door's list — i.e. the equality assert is exactly
    as green as before.  That is the tautology, exhibited.
    """
    pytest.importorskip("jax")
    import sys
    from ffi import _services
    _services.ensure_on_path()
    from distrib_la import BACKEND_CHOICES
    from gw import gw_config

    monkeypatch.setitem(sys.modules, "distrib_la", None)     # -> ImportError
    got = gw_config.eigh_backend_choices()
    assert gw_config.EIGH_CHOICES_SOURCE.startswith("fallback ("), (
        gw_config.EIGH_CHOICES_SOURCE)
    assert set(got) == set(BACKEND_CHOICES["eigh"])          # still equal!


def test_a_door_that_renamed_the_op_raises_instead_of_falling_back(monkeypatch):
    """A KeyError from the door is a CONTRACT break, not an environment fact.

    The subscript sits OUTSIDE the narrowed ``except ImportError``, so a
    door that stopped publishing an ``eigh`` row cannot be answered with a
    frozen literal that describes a vocabulary nothing dispatches on.

    RED ARM: widen the guard back to ``except Exception`` and this cell
    fails with the fallback tuple in hand instead of the KeyError.
    """
    pytest.importorskip("jax")
    from ffi import _services
    _services.ensure_on_path()
    import distrib_la
    from gw import gw_config

    monkeypatch.setattr(distrib_la, "BACKEND_CHOICES",
                        {"cholesky": ("native",)})
    with pytest.raises(KeyError):
        gw_config.eigh_backend_choices()


def test_resolve_eigh_backend_passthrough_and_promotion():
    from gw.gw_config import resolve_eigh_backend
    assert resolve_eigh_backend({}) == "auto"
    assert resolve_eigh_backend({"eigh_backend": "off"}) == "off"
    assert resolve_eigh_backend({"eigh_backend": "OFF  "}) == "off"
    assert resolve_eigh_backend(
        {"eigh_backend": "auto", "use_low_mem_eigh": True}) == "distributed"
    with pytest.raises(ValueError, match="invalid"):
        resolve_eigh_backend({"eigh_backend": "replicated"})


def test_input_keys_parse(tmp_path):
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck.in"
    p.write_text(
        "[cohsex]\n"
        "nval = 4\n"
        "wfn_fi_q_chunk = 12\n"
        "use_low_mem_eigh = true\n"
        "eigh_backend = distributed\n")
    params = read_lorrax_input(str(p))
    assert params["wfn_fi_q_chunk"] == 12 and isinstance(
        params["wfn_fi_q_chunk"], int)
    assert params["use_low_mem_eigh"] is True
    assert params["eigh_backend"] == "distributed"
    # defaults, when the deck says nothing
    q = tmp_path / "bare.in"
    q.write_text("[cohsex]\nnval = 4\n")
    bare = read_lorrax_input(str(q))
    assert bare["wfn_fi_q_chunk"] == 0        # 0 == "use N_q_co"
    assert bare["use_low_mem_eigh"] is False
