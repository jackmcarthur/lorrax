"""THE WIRING-CLOSURE GATE for ``screening_diagrams = w_bse``.

WHAT IS BEING PROVEN.  The ladder facade's ``include_w=False`` limit is
the RPA operator, so the whole ``w_bse`` assembly run at that setting must
reproduce the W(0) the production ``w_rpa`` path already writes — not
approximately, and not on a body-vs-body slice, but on the FULL tile, at
q = 0 AND at a finite q_irr, after every step this feature adds: the
q-wedge resolvent, the ``+ v`` add-back, the μ-pad, the wedge→full-BZ
unfold, and the q=0 head policy.

The user's requirement, in the user's words, is that "W_RPA still agrees
with the actual W by quadrature at zero freq".  ``bse_w_exact --compare-w0``
already proves that for ONE column of ONE tile through a hand-driven CLI.
This cell proves it for the PLUMBING: the same identity, reached through
``gw.screening_bse``'s production assembly instead of a script, which is
the only version of the claim that can protect the feature.

WHY THE FLOOR IS ~2.5e-9 AND NOT machine epsilon.  ``W0_qmunu`` on disk is
the RPA W(0) built from χ₀(iω) MINIMAX-QUADRATURED to ω = 0, while the
resolvent uses the exact ``1/(e_c − e_v)`` static denominator.  The
difference between them is therefore the GW minimax integration noise, and
the solver residual sits orders below it.  The number and its reading are
``src/bse/bse_w_exact.py``'s own "Interpretation" block, quoted there
because that is where it was measured.  A DISAGREEMENT MUCH SMALLER than
this floor would be as suspicious as one much larger: it would mean the
two sides are not independent.

SCOPE OF WHAT THIS DOES NOT SAY.  Nothing about the ladder itself.
``include_w=False`` deliberately removes the direct rung — the thing the
feature exists for — so that the WIRING can be certified separately from
the physics.  The ladder operator's own gates are Agent A's dense oracle
and its red twin (design section 6.2/6.3).  Reading a green cell here as
"the ladder is right" is the circular-exoneration mistake
(QUALITY_PATTERNS #1 corollary).

TWO CELLS IN THIS FILE WERE DELIBERATELY RED from 2026-08-15 (JID
57064957, CLAIMS 0215) until 2026-08-16, and their history is the reason
the sentence above matters:

* ``test_the_ladder_w_passes_the_production_w_gate_at_finite_q`` — the
  assembled ladder W through ``gw/screening._gate_w``.  Red at 7.251e-05
  against the 1e-05 reciprocity bar (per-q hermiticity clean throughout);
  GREEN at **7.514e-12** after the finite-q fixes.
* ``test_the_ladder_operator_obeys_q_conjugate_reciprocity_without_the_unfold``
  — the same statistic with NO symmetry table, solving separately at ``q``
  and ``-q``.  Red at LADDER 3.579e-04 vs RPA 4.081e-11; GREEN at
  **LADDER 4.743e-11** vs the same RPA arm.

What fixed them (both required; FINITE_Q_ROW.md in the sandbox report dir
carries the full record): the TRS/Kramers pair gauge
(``bse_w_exact.enforce_trs_pair_gauge`` — the conj-pattern anti-resonant
channel is exact only in ``psi(-k) = Theta psi(k)``) and the rung's
physical operand slots (``ladder_rung_slots`` — the direct rung must not
run on the payload's conjugated density arrays).  The ``include_w=False``
closure control read 7.153e-12 against the production W(0)'s own 7.151e-12
throughout, which is what licensed blaming the operator and not this
file's assembly.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import jax
from jax.sharding import Mesh

import harness

_REG = pathlib.Path(__file__).resolve().parents[0] / "regression"

#: The documented minimax-quadrature floor, ``bse_w_exact.py``'s
#: --compare-w0 interpretation block.  A ceiling, not a target.
_QUADRATURE_FLOOR = 2.5e-9

#: Slack over the measured floor.  The floor was measured on ONE deck; the
#: gate has to survive a different one without being loosened by hand, and
#: 4x is small enough that a real break (the ones this tree has measured
#: sit at 1e-4 and above) cannot hide under it.
_REL_TOL = 4.0 * _QUADRATURE_FLOOR


def _require_ladder_facade():
    try:
        import bse.w_ladder                                    # noqa: F401
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"bse.w_ladder is not in the tree yet: {exc}")


@pytest.fixture(scope="session")
def wbse_closure_run(tmp_path_factory):
    """A COHSEX run of the gnppm fixture, wired for the ladder handoff.

    Three mutations off the shipped deck, each of them a documented
    precondition of ``screening_diagrams = w_bse`` rather than a
    convenience:

    * ``compute_mode = cohsex`` — one W role, so the comparison is about
      the assembly and not about the probe leg.
    * ``restart_q_storage = full`` — the sharded BSE loader's hyperslab
      transport refuses a q wedge by name (``bse_loading._MunuSlabPlan``),
      which is why ``screening_bse._refuse_unusable_restart`` demands the
      full BZ before any compute.
    * ``write_restart_tensors`` left on — the persist IS the handoff.

    The run is a plain ``w_rpa`` run: it produces the REFERENCE.  The
    ``w_bse`` side is driven in-process below, off the file this run left.

    THE DRIVER SUBPROCESS IS PINNED TO ONE DEVICE, and that is a structural
    requirement rather than a speed choice.  This cell's assembly arm wants
    four devices visible to ONE process (``LORRAX_MESH_CELL=1``, the 2x2
    arm below), and the GW driver cannot run in that shape at all: its
    ζ-fit writes through ``file_io.slab_io.SlabIO``, whose collective open
    refuses ``mesh 2x2=4 != jax.process_count()=1`` by name
    (``src/ffi/io.py``, measured 2026-08-15, JID 57064957 — the same
    refusal ``test_bse_w_ladder_dense`` records for the restart-based
    cells).  The reference is therefore produced at P=1 and the assembly is
    exercised on the wider mesh; both sides are the same operator, and the
    gate's ceiling is four orders above P-reassociation noise.
    """
    run_dir = harness.copy_fixture(
        _REG / "gnppm_debug",
        tmp_path_factory.mktemp("wbse_closure") / "cohsex_full")
    harness.mutate_input(run_dir / "gnppm_test.in", {
        "compute_mode = gn_ppm": "compute_mode = cohsex",
        "use_ppm_sigma = true": "use_ppm_sigma = false",
        "sigma_freq_debug_output = true": "sigma_freq_debug_output = false",
    }, append="restart_q_storage = full\n")
    res = harness.run_gw_jax(run_dir, "gnppm_test.in",
                             extra_env=_single_device_env())
    if res.returncode != 0:                        # pragma: no cover
        pytest.fail(
            f"the w_bse closure reference run failed.\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}")
    tensors = sorted((run_dir / "tmp").glob("isdf_tensors_*.h5"))
    if not tensors:                                # pragma: no cover
        pytest.fail(f"no restart tensors under {run_dir / 'tmp'}")
    return run_dir, tensors[0]


def _rebuild_run_state(run_dir, tensors_path):
    """The GW-side scaffolding the assembly needs, off the run's own inputs.

    Rebuilt rather than smuggled out of the driver: what is being tested
    is that the assembly works from the objects a run actually has (the
    WFN's symmetry tables, the deck's centroid file, the restart's
    tensors), and a state handed over by the code under test would be the
    circular reference QUALITY_PATTERNS #1 warns about.
    """
    from ffi import _services
    _services.ensure_on_path()
    import symmetry_maps
    from wfn_loader import WfnLoader

    from common import Meta
    from file_io import load_centroids
    from gw.gw_config import LorraxConfig

    config = LorraxConfig.from_input_file(
        str(run_dir / "gnppm_test.in"), print_fn=lambda *a, **k: None)
    wfn = WfnLoader(str(run_dir / "WFN.h5"))
    sym = symmetry_maps.SymMaps(wfn)
    _frac, centroid_indices, n_rmu = load_centroids(
        str(run_dir / config.paths.centroids_file), tuple(wfn.fft_grid))
    meta = Meta.from_system(
        wfn, sym, config.nval, config.ncond, config.nband, n_rmu)
    # The driver assigns this after Meta construction.  Body-only closure
    # cells never noticed its absence; the q0 scalar reduction is dimension
    # aware and would otherwise compare the run's slab head to a bulk cell.
    meta.sys_dim = config.sys_dim
    return config, sym, centroid_indices, meta


def _single_device_env():
    """``extra_env`` that gives a child process exactly one of our devices.

    A no-op (empty dict) when this process already has one — so the
    ordinary suite, where ``conftest.pytest_configure`` has pinned every
    pytest process to a single GPU, launches the driver exactly as it
    always did.
    """
    import os

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [d for d in visible.split(",") if d]
    if len(ids) <= 1:
        return None
    return {"CUDA_VISIBLE_DEVICES": ids[0]}


def _closure_mesh():
    """The widest mesh this PROCESS can carry, 2x2 or 1x1 — never a skip.

    THE CELL MUST NOT SKIP ITSELF ON DEVICE COUNT.  ``conftest`` pins every
    pytest process to one GPU, so a ``skip(device_count() < 4)`` here would
    make the centrepiece gate of this feature skip in every census on every
    node — the exact silent gap the ``mesh(n)`` marker exists to close
    (``conftest.py``, 2026-08-10).  The four-device arm is reached by
    running this file under ``LORRAX_MESH_CELL=1`` on a 4-GPU step, which
    is how the suite spells "this process IS the widened one":

        LORRAX_MESH_CELL=1 lx run -G 4 -n 1 -- \
            python3 -m pytest -q tests/test_w_bse_wiring_closure.py

    The ``mesh(4)`` MARKER is not usable here for the reason recorded in
    ``test_bse_w_ladder_dense``: its child sets ``LORRAX_FFT_FFI=0``, which
    the GW driver this file's fixture runs refuses.

    At 1x1 the shard boundaries this assembly crosses
    (``slice_q_full_to_ibz``, the wedge's ``P(None, None, 'x', 'y')``
    constraint, ``unfold_isdf_operator``'s permutation) are no-ops, so a
    green 1x1 arm is a statement about the
    ALGEBRA of the assembly and not about its sharding.  Say which one ran:
    the mesh shape is asserted into the failure text below.
    """
    devs = jax.devices()
    if len(devs) >= 4:
        return Mesh(np.asarray(devs[:4]).reshape(2, 2), axis_names=("x", "y"))
    return Mesh(np.asarray(devs[:1]).reshape(1, 1), axis_names=("x", "y"))


def _force_serial_restart_transport(monkeypatch, mesh_xy):
    """Read the restart with the serial h5py tile readers on a wide mesh.

    ONLY when this process holds a multi-device mesh and is ONE process,
    which is exactly the arrangement ``SlabIO`` refuses: its collective open
    asserts ``mesh devices == jax.process_count()`` (``src/ffi/io.py``), and
    an in-process 2x2 is 4 != 1.  ``bse_loading`` already carries the serial
    tile readers as its documented fallback for a stack without the phdf5
    FFI, and its own transport-seam docstring states the parity bar between
    the two branches is BIT EQUALITY (identical element selection, not a
    reduction-order change) — so forcing them changes which bytes move and
    nothing about the tensors this gate then measures.

    Not a workaround for a defect: the SlabIO transport at P=4 is a
    MULTI-PROCESS statement and belongs to ``tests/multi_device/``.  What
    this line buys is that the ASSEMBLY under test still gets a real 2x2.
    """
    if int(mesh_xy.devices.size) <= 1 or int(jax.process_count()) > 1:
        return False
    from bse import bse_loading

    monkeypatch.setattr(bse_loading, "_bse_slabio_usable",
                        lambda log_fn=print: False)
    return True


def _read_full_bz(path, name, n_rmu):
    """Read a ``(nq, n_rmu, n_rmu)`` restart tensor as plain numpy.

    Refuses a wedge instead of unfolding one: the fixture pins
    ``restart_q_storage = full`` precisely so this reader has nothing to
    decide, and a silent unfold here would put the thing under test on
    both sides of the comparison.
    """
    import h5py

    with h5py.File(path, "r") as f:
        assert name in f, f"{path} has no {name}"
        arr = np.asarray(f[name][()])
        ready_attr = {"W0_qmunu": "W0_ready", "V_qmunu": "V_ready"}[name]
        assert bool(f[name].attrs.get(ready_attr, name == "V_qmunu")), (
            f"{name} is present but {ready_attr} is False — the placeholder, "
            f"not a written tensor")
        assert "qirr_format_version" not in f[name].attrs, (
            f"{name} is stored on the q wedge; the fixture pins "
            f"restart_q_storage = full")
    assert arr.ndim == 3 and arr.shape[-1] == n_rmu, arr.shape
    return arr


@pytest.mark.gpu
def test_include_w_false_reproduces_the_production_rpa_w0(wbse_closure_run,
                                                          monkeypatch):
    """THE CENTREPIECE.  Full tile, q=0 AND a finite q_irr, to the floor.

    Every step the feature adds is inside the measured object: the
    q-wedge resolvent call, the ``+ v`` add-back, the μ-pad embed, and the
    ``unfold_isdf_operator`` wedge→full-BZ expansion with its umklapp
    phase and TRS conj.  The q=0 arm and the finite-q arm are BOTH
    required because the finite-q vertex-conjugation bug this tree has
    already paid for was invisible at q=0 (KNOWN_FAILURES:1248).
    """
    _require_ladder_facade()
    run_dir, tensors_path = wbse_closure_run
    mesh_xy = _closure_mesh()
    mesh_shape = tuple(int(n) for n in mesh_xy.devices.shape)
    _force_serial_restart_transport(monkeypatch, mesh_xy)
    config, sym, centroid_indices, meta = _rebuild_run_state(
        run_dir, tensors_path)

    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from runtime.padding import padded_mu_extent
    from gw import screening_bse
    from wfn_loader import WfnLoader

    n_rmu = int(meta.n_rmu)
    # Keyed on THE MESH, not on ``jax.device_count()``: the pad extent has
    # to match the one the wedge/unfold tables were built for, and those
    # come from the mesh this cell actually assembles on.
    mu_pad = padded_mu_extent(n_rmu, int(mesh_xy.devices.size))
    W0_ref = _read_full_bz(tensors_path, "W0_qmunu", n_rmu)
    V_ref = _read_full_bz(tensors_path, "V_qmunu", n_rmu)

    # Sharded via a CONSTRAINT, not ``device_put``: the multi-process
    # transfer runs a silent assert_equal all-gather (QUALITY_PATTERNS #5,
    # and rules-gate B4), which is exactly the hidden O(P) cost a
    # mu^2-class operand must not pay inside a gate.
    V_q = jax.lax.with_sharding_constraint(
        jnp.pad(jnp.asarray(V_ref, dtype=jnp.complex128),
                ((0, 0), (0, mu_pad - n_rmu), (0, mu_pad - n_rmu))),
        NamedSharding(mesh_xy, P(None, 'x', 'y')))

    # THE TIGHT TOLERANCE, DELIBERATELY.  This cell measures the assembly
    # against the ~2.5e-9 minimax-quadrature floor of the stored W(0), so the
    # solver residual has to sit BELOW the thing being measured.  The
    # PRODUCTION constant is chosen for the QP energies instead (measured
    # tolerance ladder, 2026-08-16, at `screening_bse._GMRES_TOL`) and is four
    # decades looser; running this cell at it would measure the solver, not
    # the wiring.  `_GMRES_TOL_TIGHT` exists for exactly this, and naming it
    # here keeps the two questions apart instead of forcing one constant to
    # answer both.
    wfn = WfnLoader(str(run_dir / "WFN.h5"))
    wedge = screening_bse._ladder_wedge(
        str(tensors_path), [0.0 + 0.0j], mesh_xy,
        input_file=config.input_file, include_w=False,
        gmres_tol=screening_bse._GMRES_TOL_TIGHT,
        print_fn=lambda *a, **k: None, config=config, meta=meta, wfn=wfn)
    screening_bse._assert_wedge_matches_run(wedge, sym)

    # QSGW-hat guide Test 2, on the REAL production payload.  Kd=0 is the
    # facade's include_w=False arm, but Kx (the finite-G Hartree/ring term)
    # remains in the full 2N resolvent.  It must therefore produce the same
    # macroscopic RPA head as the direct S + Y W_body Z Schur path persisted
    # by the reference driver.  This is deliberately not the K=0/IP control,
    # which removes Kx too and reproduces only the no-local-field epsilon head.
    assert wedge.head_result is not None
    xi = np.asarray(wedge.head_result.xi[0])
    xi_long = 0.5 * (xi + xi.T)
    import h5py
    with h5py.File(tensors_path, "r") as f:
        S_schur = np.asarray(f["S_cart_head"][:], dtype=np.complex128)
        W_schur = complex(np.asarray(f["whead"][:]).reshape(-1)[0])
    tensor_rel = float(np.linalg.norm(xi_long - S_schur)) / max(
        float(np.linalg.norm(S_schur)), 1.0e-300)
    from gw.vcoul import compute_q0_averages
    _, W_resolvent = compute_q0_averages(
        wfn, jnp.asarray(0.0, dtype=jnp.float64), meta,
        S_cart=jnp.asarray(xi_long),
        analytic_sphere=bool(config.head.analytic_q0_sphere))
    scalar_rel = abs(complex(W_resolvent) - W_schur) / max(
        abs(W_schur), 1.0e-300)
    print(
        f"[head Test2 {mesh_shape[0]}x{mesh_shape[1]}] Kd=0, Kx retained: "
        f"tensor rel {tensor_rel:.3e}; W resolvent={complex(W_resolvent).real:.9f}, "
        f"W Schur={W_schur.real:.9f}, rel {scalar_rel:.3e}; "
        f"head residual={float(np.max(wedge.head_result.resids)):.2e}")
    assert scalar_rel < 1.0e-6, (
        "QSGW-hat Test 2 failed: the micro-reducible Kd=0, Kx-retained "
        f"resolvent head differs from the once-folded RPA head by "
        f"{scalar_rel:.3e} (tensor difference {tensor_rel:.3e}).  Folding "
        "the resolvent again is not a repair; it would double-count Kx.")
    W_full = np.asarray(jax.device_get(screening_bse._assemble_full_bz_w(
        wedge.wc[0], V_q, sym=sym, centroid_indices=centroid_indices,
        meta=meta, mesh_xy=mesh_xy, label="static",
        print_fn=lambda *a, **k: None)))
    W_wired = W_full[:, :n_rmu, :n_rmu]

    assert W_wired.shape == W0_ref.shape, (
        f"the wired assembly returned {W_wired.shape}, the production W(0) "
        f"is {W0_ref.shape}")

    # THE PADDED BLOCK MUST STILL BE ZERO after the unfold.  The producer
    # guarantees it on the wedge (``w_ladder._assert_pad_block_is_zero``);
    # this is the same statement AFTER the ``+ v`` add-back and the
    # symmetry unfold, i.e. that the sym tables' identity tail and zero
    # wrap carried it through.  The check is EXACT rather than a
    # finiteness test on purpose: a pad that is merely finite still rides
    # into every downstream contraction as a centroid that does not exist,
    # and a NaN there is not inert either (``0 * NaN`` is NaN).  This is
    # the assertion that caught the zero-rhs GMRES pad columns on
    # 2026-08-15.
    assert np.all(W_full[:, n_rmu:, :] == 0.0), (
        f"pad ROWS of the assembled W are not zero; worst "
        f"|.| = {np.abs(np.nan_to_num(W_full[:, n_rmu:, :], nan=np.inf)).max():.3e}")
    assert np.all(W_full[:, :, n_rmu:] == 0.0), (
        f"pad COLUMNS of the assembled W are not zero; worst "
        f"|.| = {np.abs(np.nan_to_num(W_full[:, :, n_rmu:], nan=np.inf)).max():.3e}")

    # THE RECIPROCITY CONTROL ARM.  ``W_q = conj(W_{−q})`` is the property
    # the BSE kernel's hermiticity reduces to, and the one the production
    # gate checks over the whole flat-q axis.  Measured HERE, on the
    # include_w=False assembly, it is the control that tells a ladder-side
    # break apart from a plumbing-side one: this tile went through exactly
    # the steps the ladder tile goes through (facade, + v, unfold) with the
    # direct rung removed, so whatever it reads is the ASSEMBLY's own
    # residual.  The production W(0) read straight off disk is the second
    # reference: it is what this path has to be no worse than.
    from common import sanity

    neg = sanity.neg_q_index(tuple(meta.kgrid))

    def _recip(a):
        return (float(np.max(np.abs(a - np.conj(a[neg]))))
                / max(float(np.max(np.abs(a))), 1.0e-300))

    recip_ref, recip_wired = _recip(W0_ref), _recip(W_wired)
    print(f"[closure {mesh_shape[0]}x{mesh_shape[1]}] reciprocity "
          f"max|W_q - conj(W_-q)|/|W|:  production W0 on disk "
          f"{recip_ref:.3e}   include_w=False assembly {recip_wired:.3e}")
    assert recip_wired <= max(10.0 * recip_ref, 1.0e-9), (
        f"the include_w=False assembly loses q<->-q reciprocity that the "
        f"production W(0) has: {recip_wired:.3e} vs {recip_ref:.3e}.  Both "
        f"are unfolded from an orbit-closed wedge, where reciprocity holds "
        f"BY CONSTRUCTION and the residual is pure arithmetic, so a gap "
        f"here is in this feature's assembly — not in the operator.")

    q_finite = _first_finite_irreducible_q(sym)
    for label, q in (("q=0", 0), (f"q_irr={q_finite}", q_finite)):
        ref, got = W0_ref[q], W_wired[q]
        scale = float(np.max(np.abs(ref)))
        rel = float(np.max(np.abs(got - ref))) / max(scale, 1.0e-300)
        print(f"[closure {mesh_shape[0]}x{mesh_shape[1]}] {label}: "
              f"max_rel = {rel:.6e}  (ceiling {_REL_TOL:.1e}, floor "
              f"{_QUADRATURE_FLOOR:.1e})")
        assert rel < _REL_TOL, (
            f"{label} on a {mesh_shape[0]}x{mesh_shape[1]} mesh: the "
            f"include_w=False assembly does not close on the "
            f"production RPA W(0).  max rel = {rel:.3e} (ceiling "
            f"{_REL_TOL:.1e}, documented minimax floor "
            f"{_QUADRATURE_FLOOR:.1e}).  Both sides are head-less bodies of "
            f"the same operator, so a break here is in the wiring — the "
            f"add-back, the pad, or the unfold — not in the quadrature.")
        # THE OBSERVABLE MUST DISCRIMINATE (QUALITY_PATTERNS addendum).  A
        # rel of exactly 0 would mean the two sides are not independent —
        # that the "wired" tile is the reference read back.
        assert rel > 0.0, (
            f"{label}: the two tiles are BIT-IDENTICAL.  They cannot be: "
            f"one is minimax-quadratured chi0 through a Dyson solve, the "
            f"other an exact-denominator resolvent.  Zero here means the "
            f"comparison is reading one object twice.")


@pytest.mark.gpu
def test_the_ladder_w_passes_the_production_w_gate_at_finite_q(
        wbse_closure_run, monkeypatch):
    """THE LADDER W's OWN gate, at q=0 AND at every irreducible finite q.

    OPEN RISK 3 of the design, closed or refuted here.  Until this cell the
    ladder operator's hermiticity was measured at ``q = 0`` only
    (``test_bse_w_ladder_identities::test_ladder_w_tile_is_hermitian_at_q0``,
    a sub-tile on a 4-column probe), and the reciprocity ``W_q =
    conj(W_{−q})`` — the property the BSE kernel's own hermiticity actually
    reduces to (``common.sanity.check_q_conjugate_reciprocity``) — was not
    measured for it at all.  Both are load-bearing and INDEPENDENT: this
    tree has measured a tile that passes per-q hermiticity at 1e-15 while
    failing reciprocity at 9.1e-4 (``gw/screening._gate_w``'s own note,
    armA_base480).

    WHY A FIRING HERE IS EVIDENCE AND NOT AN INCONVENIENCE.  The q=0
    hermiticity observable is exactly what exposed the anti-resonant-row
    defect on 2026-08-15 — it read 2.13e-05 under the naive symplectic row
    while every other cell in the suite, dense oracle included, stayed
    green, because the oracle carried the same row.  Finite q is the arm
    that assumption has never been tested on, and the finite-q vertex
    conjugation bug this tree has already paid for (KNOWN_FAILURES:1248)
    was invisible at q=0.  So: the tolerances here are the PRODUCTION ones,
    reached through the production gate, and they are not to be loosened.

    THREE OBSERVABLES, and the per-q hermiticity is measured on the WEDGE
    tiles rather than on the unfolded ones on purpose.  Every wedge tile is
    an independent resolvent solve; the full-BZ tiles outside the wedge are
    built from them BY SYMMETRY, so their hermiticity — and much of the
    reciprocity residual — would hold by construction and certify the
    unfold rather than the operator.
    """
    _require_ladder_facade()
    run_dir, tensors_path = wbse_closure_run
    mesh_xy = _closure_mesh()
    mesh_shape = tuple(int(n) for n in mesh_xy.devices.shape)
    _force_serial_restart_transport(monkeypatch, mesh_xy)
    config, sym, centroid_indices, meta = _rebuild_run_state(
        run_dir, tensors_path)

    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from runtime.padding import padded_mu_extent
    from common import sanity
    from gw import screening_bse
    from gw.screening import ScreeningRequest, _gate_w
    from gw.vcoul import compute_q0_averages
    from wfn_loader import WfnLoader

    n_rmu = int(meta.n_rmu)
    mu_pad = padded_mu_extent(n_rmu, int(mesh_xy.devices.size))
    V_ref = _read_full_bz(tensors_path, "V_qmunu", n_rmu)
    V_q = jax.lax.with_sharding_constraint(
        jnp.pad(jnp.asarray(V_ref, dtype=jnp.complex128),
                ((0, 0), (0, mu_pad - n_rmu), (0, mu_pad - n_rmu))),
        NamedSharding(mesh_xy, P(None, 'x', 'y')))

    # Tight tolerance, same reason as the cell above: this one asserts the
    # ladder W's per-q hermiticity and its finite-q reciprocity against the
    # PRODUCTION 1e-6 gate tolerance, so the solve must not be the thing
    # supplying the 1e-6.
    wfn = WfnLoader(str(run_dir / "WFN.h5"))
    wedge = screening_bse._ladder_wedge(
        str(tensors_path), [0.0 + 0.0j], mesh_xy,
        input_file=config.input_file, include_w=True,
        gmres_tol=screening_bse._GMRES_TOL_TIGHT, print_fn=print,
        config=config, meta=meta, wfn=wfn)
    screening_bse._assert_wedge_matches_run(wedge, sym)

    # The q=0 head must ride the SAME micro-reducible resolvent as the body.
    # This is the production w_bse physics check, not a post-hoc Schur fold:
    # Kx (finite-G Hartree/local fields) is already in ``xi``, while Kd is
    # the direct ladder rung.  Folding this result again would count Kx
    # twice.  For this semiconducting fixture the ladder must increase the
    # screening, hence lower W relative to the once-folded RPA reference.
    assert wedge.head_result is not None
    xi = np.asarray(wedge.head_result.xi[0])
    xi_long = 0.5 * (xi + xi.T)
    v_head, W_bse = compute_q0_averages(
        wfn, jnp.asarray(0.0, dtype=jnp.float64), meta,
        S_cart=jnp.asarray(xi_long),
        analytic_sphere=bool(config.head.analytic_q0_sphere))
    import h5py
    with h5py.File(tensors_path, "r") as f:
        W_rpa = complex(np.asarray(f["whead"][:]).reshape(-1)[0])
    eps_rpa = complex(v_head).real / W_rpa.real
    eps_bse = complex(v_head).real / complex(W_bse).real
    print(
        f"[head trend {mesh_shape[0]}x{mesh_shape[1]}] "
        f"W_RPA={W_rpa.real:.9f}, eps_RPA={eps_rpa:.9f}; "
        f"W_BSE={complex(W_bse).real:.9f}, eps_BSE={eps_bse:.9f}; "
        f"W_BSE/W_RPA={complex(W_bse).real / W_rpa.real:.9f}; "
        f"head residual={float(np.max(wedge.head_result.resids)):.2e}")
    assert 0.0 < complex(W_bse).real < W_rpa.real, (
        "the ladder head did not increase static screening on the closure "
        f"fixture: W_BSE={complex(W_bse).real:.9f}, "
        f"W_RPA={W_rpa.real:.9f}")

    # (1) PER-q HERMITICITY OF THE LADDER W ITSELF, on the wedge, on the
    #     logical block.  ``W = (W - v) + v`` and v is Hermitian, so the
    #     body's residual is the tile's.
    q_kgrid = np.asarray(wedge.q_irr_kgrid_int, dtype=int)
    body = np.asarray(jax.device_get(wedge.wc[0]))[:, :n_rmu, :n_rmu]
    herm = []
    for iq in range(body.shape[0]):
        a = body[iq]
        dev = float(np.max(np.abs(a - a.conj().T)))
        scale = max(float(np.max(np.abs(a))), 1.0e-300)
        herm.append(dev / scale)
        print(f"[ladder-gate {mesh_shape[0]}x{mesh_shape[1]}] "
              f"q_irr[{iq}] = {tuple(q_kgrid[iq])}: "
              f"max|W - W^dag|/|W| = {dev / scale:.3e}")
    assert len(herm) >= 2, (
        "this fixture's wedge has one q; the finite-q arm of the ladder "
        "gate needs a second irreducible q")

    # (2) THE ASSEMBLED FULL-BZ TILE THROUGH THE PRODUCTION GATE, in STRICT
    #     mode so a violation RAISES instead of printing a warning into a
    #     log nobody reads (``common.sanity``: LORRAX_SANITY=strict is the
    #     documented setting for a regression gate).
    W_q = screening_bse._assemble_full_bz_w(
        wedge.wc[0], V_q, sym=sym, centroid_indices=centroid_indices,
        meta=meta, mesh_xy=mesh_xy, label="static", print_fn=print)
    W_host = np.asarray(jax.device_get(W_q))[:, :n_rmu, :n_rmu]
    neg = sanity.neg_q_index(tuple(meta.kgrid))
    recip = (float(np.max(np.abs(W_host - np.conj(W_host[neg]))))
             / max(float(np.max(np.abs(W_host))), 1.0e-300))
    print(f"[ladder-gate {mesh_shape[0]}x{mesh_shape[1]}] full BZ "
          f"({W_host.shape[0]} q): max|W_q - conj(W_-q)|/|W| = {recip:.3e}")

    monkeypatch.setenv("LORRAX_SANITY", "strict")
    _gate_w(W_q, ScreeningRequest(0.0 + 0.0j, "static"),
            print_fn=print, kgrid=tuple(meta.kgrid))

    # (3) THE COVERAGE EXTENSION, stated as its own assertion: the
    #     production gate checks hermiticity at q=0 only, so the finite-q
    #     arm has to be asserted here or it is not gated at all.  Same
    #     tolerance the production gate uses at q=0 (1e-6, screening.py):
    #     structural mixing, not roundoff.
    worst_finite = max(herm[1:])
    assert worst_finite < 1.0e-6, (
        f"the ladder W is NOT Hermitian at finite q: worst "
        f"max|W - W^dag|/|W| = {worst_finite:.3e} over q_irr[1:] "
        f"(q=0 reads {herm[0]:.3e}).  This is EVIDENCE ABOUT THE OPERATOR, "
        f"not a tolerance to loosen — the q=0 twin of this observable is "
        f"what exposed the anti-resonant row (2.13e-05 under the naive "
        f"symplectic row, 6.9e-15 under the derived one), and finite q is "
        f"where the vertex-conjugation convention has never been tested.")


@pytest.mark.gpu
def test_the_ladder_operator_obeys_q_conjugate_reciprocity_without_the_unfold(
        wbse_closure_run, monkeypatch):
    """``W(-q) = conj(W(q))`` SOLVED at both q, with no symmetry step at all.

    THE MEASUREMENT THAT ATTRIBUTES A RECIPROCITY BREAK.  The assembled
    gate above reads the statistic on a tile that was unfolded from the
    wedge, so a violation there has two possible homes: the operator, or
    the assumption that the ladder W is symmetry-covariant enough for the
    unfold to build ``-q`` from ``q``'s orbit.  This cell removes the
    second one — it solves the resolvent SEPARATELY at ``q`` and at
    ``-q`` (both are ordinary points for ``build_finite_q_data``) and
    compares.  Nothing symmetric happens in between.

    The include_w=False arm is the control: same two solves, same
    comparison, RPA operator.  It says what this pair of solves costs in
    pure arithmetic, which is the only way to read the ladder number.
    """
    _require_ladder_facade()
    run_dir, tensors_path = wbse_closure_run
    mesh_xy = _closure_mesh()
    mesh_shape = tuple(int(n) for n in mesh_xy.devices.shape)
    _force_serial_restart_transport(monkeypatch, mesh_xy)
    config, sym, centroid_indices, meta = _rebuild_run_state(
        run_dir, tensors_path)

    from bse.bse_io import load_bse_data_from_restart_sharded
    from bse.w_ladder import sweep_q_wedge

    data = load_bse_data_from_restart_sharded(
        str(tensors_path), n_val=10**9, n_cond=10**9, mesh_xy=mesh_xy,
        input_file=config.input_file, inject_head=False, load_v_full=True)
    n_pad, nlog = int(data["V_q0"].shape[0]), int(data["n_rmu"])
    G = np.zeros((n_pad, n_pad), dtype=np.float64)
    G[:nlog, :] = np.eye(n_pad, dtype=np.float64)[:nlog, :]

    # The first FINITE irreducible q of this run's wedge, and its negative
    # folded back into the grid.  q=0 is its own negative, which is exactly
    # why a q=0-only reciprocity check is worthless (``common.sanity``).
    kgrid = tuple(int(v) for v in meta.kgrid)
    q_pos = tuple(int(v) for v in np.asarray(sym.q_irr_kgrid_int)[1])
    q_neg = tuple((-np.asarray(q_pos)) % np.asarray(kgrid))

    def _solve_pair(include_w):
        got = {}

        def _on_result(iq, q, iz, z, c0, n_real, W_tile, resids, its):
            got[q] = (np.asarray(jax.device_get(W_tile))[:nlog, :nlog],
                      float(np.max(np.asarray(jax.device_get(resids)))))

        sweep_q_wedge(
            data, mesh_xy, [q_pos, q_neg], [0.0 + 0.0j],
            include_w=include_w,
            probe_blocks_for_q=lambda _iq, _q: [(0, nlog, G)],
            gmres_tol=1.0e-10, gmres_max_iter=200, on_result=_on_result)
        a, ra = got[tuple(q_pos)]
        b, rb = got[tuple(q_neg)]
        assert max(ra, rb) < 1.0e-8, (
            f"include_w={include_w}: GMRES did not converge "
            f"(max residual {max(ra, rb):.2e}); the comparison below would "
            f"be reading solver error, not the operator")
        return (float(np.max(np.abs(b - np.conj(a))))
                / max(float(np.max(np.abs(a))), 1.0e-300))

    rpa = _solve_pair(False)
    ladder = _solve_pair(True)
    print(f"[operator-reciprocity {mesh_shape[0]}x{mesh_shape[1]}] "
          f"q = {q_pos}, -q = {tuple(int(v) for v in q_neg)}:  "
          f"max|W(-q) - conj(W(q))|/|W(q)|   RPA {rpa:.3e}   "
          f"LADDER {ladder:.3e}")
    assert ladder <= max(100.0 * rpa, 1.0e-9), (
        f"THE LADDER OPERATOR ITSELF violates W(-q) = conj(W(q)): "
        f"{ladder:.3e}, against {rpa:.3e} for the RPA operator through the "
        f"identical pair of solves.  No symmetry table and no unfold is "
        f"involved on either side, so this is the OPERATOR — the finite-q "
        f"vertex/conjugation convention of the direct rung (w_ladder.py's "
        f"closing 'Vertex convention' paragraph asserts exactly this "
        f"property and it is what is being measured).  It is the class of "
        f"defect that is invisible at q=0 (KNOWN_FAILURES:1248).")


def _first_finite_irreducible_q(sym):
    """A q_irr that is not Gamma, in FLAT full-BZ indexing.

    Row 0 of the wedge is Gamma by construction; row 1 is the first
    finite one.  Skips rather than passes vacuously on a one-q wedge — a
    gate that silently degrades to its q=0 arm certifies half of what its
    name says.
    """
    idx = np.asarray(sym.q_irr_full_idx, dtype=int)
    if idx.size < 2:                               # pragma: no cover
        pytest.skip(
            "this fixture's q wedge has one point; the finite-q arm of the "
            "closure gate needs a second irreducible q")
    return int(idx[1])


@pytest.mark.gpu
def test_the_wired_path_refuses_a_restart_it_cannot_hand_over(
        wbse_closure_run, tmp_path):
    """Persist-before-load, stated as a refusal rather than a hope.

    Three preconditions, all knowable from the deck, all refused BEFORE
    the chi0 build: writes off, file absent, wedge storage.  The one that
    matters most is the first — with the writes off the persist is a
    silent no-op and the loader falls back to BARE V with a banner, which
    is the April all-zero-screening incident's exact shape.
    """
    import dataclasses

    run_dir, tensors_path = wbse_closure_run
    config, sym, centroid_indices, meta = _rebuild_run_state(
        run_dir, tensors_path)

    with pytest.raises(ValueError, match="w_bse_needs_restart_writes"):
        screening_bse_refuse(
            dataclasses.replace(config, write_restart_tensors=False),
            meta, sym, centroid_indices, str(tensors_path))
    with pytest.raises(ValueError, match="w_bse_needs_restart_writes"):
        screening_bse_refuse(
            config, meta, sym, centroid_indices,
            str(tmp_path / "absent.h5"))
    with pytest.raises(ValueError, match="w_bse_needs_the_restart_path"):
        screening_bse_refuse(config, meta, sym, centroid_indices, None)


def screening_bse_refuse(config, meta, sym, centroid_indices, path):
    from gw import screening_bse
    return screening_bse._refuse_unusable_restart(
        config, meta, sym, centroid_indices, path,
        print_fn=lambda *a, **k: None)
