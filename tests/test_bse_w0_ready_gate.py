"""The vacuous-presence gate: presence is not persistence.

``gw_init`` allocates a full-size ZERO ``W0_qmunu`` unconditionally — an
h5py allocation, not a write, which is why the file jumps by the whole
tensor size and reads exactly like progress.  A run whose ``persist_w0``
never fired therefore leaves a restart file containing a ``W0_qmunu`` of
precisely the right shape, full of zeros.  Any reader that asks "is the
dataset there?" gets True.

That is the mechanism behind the April BSE incident: a plausible excitonic
spectrum came out of an all-zero screening tensor, with hermiticity, the
shapes and every downstream health check green, because the *direct* term
was identically zero and nothing asked whether W had ever been written.

``tagged_arrays`` already stamps the answer — ``W0_ready``, False on the
placeholder and True when real data lands — and ``bse_io`` already gates
on it in both of its readers.  ``bse.vq_interp.load_zeta_coarse`` did not:
it asked ``if "W0_qmunu" in fr``, which is the presence question.  This
file pins the fix and the hazard it closes.

TWO KINDS OF CELL, DELIBERATELY.  The BEHAVIOURAL cells build the
placeholder with the production writers and measure what it actually is —
right shape, all zeros, flag False, and present, so the presence test
really does pass on it.  The SOURCE cell asserts by ``ast`` that the
interpolation path's W0 binding is guarded by the flag, because the
end-to-end call needs an out-of-repo MoS2 fixture (``test_bse_vq_interp``
skips without it) and a gate that can only run on one machine is a gate
that rots.  The same choice, for the same reason, as
``test_kin_ion_star_broadcast.py``'s AST check on the ``"ibz_slab"``
literal — and like that one, it comes with a red twin showing the matcher
detects the pre-fix shape.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
_VQ_INTERP = _SRC / "bse" / "vq_interp.py"
_BSE_IO = _SRC / "bse" / "bse_io.py"

N_MU = 4
NQ = 2
NK, NB = 2, 3


def _mesh_1x1():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                axis_names=("x", "y"))


def _placeholder_restart_or_skip(path):
    """A restart file in the state ``gw_init`` leaves it: W0 ALLOCATED.

    The same call ``gw_init`` makes on the non-restart path —
    ``write_restart_state_to_h5(..., init_W0=True, ...)`` — and nothing
    else, so what is measured is what production actually leaves behind
    rather than a reconstruction of it.

    SKIPS where the writer cannot run.  ``write_restart_state_to_h5`` goes
    through SlabIO, which REFUSES without the built FFI ("this stack cannot
    write one tile per rank"), and that is a property of the box rather
    than of the gate — ``tests/test_restart_pad_roundtrip.py`` is red on
    this checkout for exactly that reason.  The hand-built twin below runs
    everywhere and carries the same claim.
    """
    from file_io import write_restart_state_to_h5

    rng = np.random.default_rng(20260808)
    V = (rng.standard_normal((NQ, N_MU, N_MU))
         + 1j * rng.standard_normal((NQ, N_MU, N_MU)))
    enk = rng.standard_normal((NK, NB))
    try:
        write_restart_state_to_h5(
            path, n_rmu_logical=N_MU, V_qmunu=jnp.asarray(V),
            enk_full=jnp.asarray(enk), init_W0=True, mesh=_mesh_1x1(),
            mode="w", kgrid=(NQ, 1, 1))
    except RuntimeError as exc:
        if "SlabIO REFUSED" not in str(exc):
            raise
        pytest.skip("restart writer needs the built FFI (SlabIO refused)")
    return V


def _handmade_placeholder(path):
    """The same file state, written with bare h5py.

    An ALLOCATION with no data plus ``W0_ready = False`` — the two lines
    ``tagged_arrays.write_restart_state_to_h5`` runs on its ``init_W0``
    branch (``io.create_dataset("W0_qmunu", shape=..., dtype=...)`` at
    :225 and ``f["W0_qmunu"].attrs["W0_ready"] = w0_ready`` at :244, with
    ``w0_ready`` initialised False at :215).  This is a reconstruction and
    is only worth anything because
    :func:`test_the_reconstruction_matches_what_the_production_writer_does`
    reads those lines out of the module and checks it still is one.
    """
    import h5py

    with h5py.File(path, "w") as f:
        ds = f.create_dataset("W0_qmunu", shape=(NQ, N_MU, N_MU),
                              dtype=np.complex128)
        ds.attrs["W0_ready"] = False


# ---------------------------------------------------------------------------
# The hazard, constructed
# ---------------------------------------------------------------------------

def test_the_placeholder_passes_every_presence_test_and_is_all_zeros(tmp_path):
    """The April mechanism, built with the production writer and measured.

    Four facts, and it is the conjunction that is dangerous: the dataset is
    PRESENT, its SHAPE is exactly right, its CONTENT is entirely zero, and
    ``W0_ready`` is False.  A reader gating on presence or on shape gets a
    green light and a screening tensor that does not exist.
    """
    import h5py

    path = str(tmp_path / "isdf_tensors_test.h5")
    _placeholder_restart_or_skip(path)
    with h5py.File(path, "r") as f:
        assert "W0_qmunu" in f, (
            "the placeholder is not even allocated; this test's premise "
            "has changed and the gate below may be unnecessary")
        ds = f["W0_qmunu"]
        assert ds.shape == (NQ, N_MU, N_MU)
        assert not np.asarray(ds[()]).any(), "the placeholder is not zeros"
        assert bool(ds.attrs.get("W0_ready", False)) is False


def test_the_ready_flag_flips_only_when_real_data_lands(tmp_path):
    """TWIN: the same file after ``persist_w0`` fires.

    Same path, same dataset, same shape — the only thing that changes is
    the flag and the content.  Which is precisely why the flag, and not the
    dataset, is the thing to ask about.
    """
    import h5py
    from file_io import write_w0_qmunu_to_h5

    path = str(tmp_path / "isdf_tensors_test.h5")
    _placeholder_restart_or_skip(path)
    rng = np.random.default_rng(7)
    W0 = (rng.standard_normal((NQ, N_MU, N_MU))
          + 1j * rng.standard_normal((NQ, N_MU, N_MU)))
    write_w0_qmunu_to_h5(path, jnp.asarray(W0), n_rmu_logical=N_MU,
                         mesh=_mesh_1x1())
    with h5py.File(path, "r") as f:
        ds = f["W0_qmunu"]
        assert bool(ds.attrs.get("W0_ready", False)) is True
        assert np.abs(np.asarray(ds[()])).max() > 0.0
        np.testing.assert_array_equal(np.asarray(ds[()]), W0)


def test_the_handmade_placeholder_shows_the_hazard_on_any_box(tmp_path):
    """The same four facts, without needing the FFI.

    PRESENT, right SHAPE, all ZEROS, flag False.  It is the conjunction
    that is dangerous: a reader gating on presence or on shape gets a green
    light and a screening tensor that does not exist.
    """
    import h5py

    path = str(tmp_path / "handmade.h5")
    _handmade_placeholder(path)
    with h5py.File(path, "r") as f:
        ds = f["W0_qmunu"]
        assert "W0_qmunu" in f
        assert ds.shape == (NQ, N_MU, N_MU)
        assert not np.asarray(ds[()]).any()
        assert bool(ds.attrs.get("W0_ready", False)) is False
        # And the pre-fix predicate says YES to it.  That is the bug,
        # written as an assertion rather than as a comment.
        assert ("W0_qmunu" in f) is True


def test_the_reconstruction_matches_what_the_production_writer_does():
    """What anchors the hand-built file to production.

    A forged fixture is worth exactly as much as the claim that production
    produces the same thing.  Three source facts, read out of
    ``tagged_arrays``: the ``init_W0`` branch creates ``W0_qmunu`` by SHAPE
    with no ``data=`` (an allocation, so the content is zeros); the flag it
    stamps on that branch is the one initialised False; and the dedicated
    W0 writer stamps True.
    """
    src = (_SRC / "file_io" / "tagged_arrays.py").read_text()
    tree = ast.parse(src)

    def _calls(attr):
        return [c for c in ast.walk(tree)
                if isinstance(c, ast.Call)
                and getattr(c.func, "attr", None) == attr
                and c.args and isinstance(c.args[0], ast.Constant)
                and c.args[0].value == "W0_qmunu"]

    creates = _calls("create_dataset")
    writes = _calls("write_slab")
    assert len(creates) == 2 and len(writes) == 1, (
        f"tagged_arrays makes {len(creates)} W0_qmunu allocations and "
        f"{len(writes)} writes; the gate's premise is that exactly one "
        f"allocation is never filled")
    for c in creates:
        kw = {k.arg for k in c.keywords}
        assert "shape" in kw and "data" not in kw, (
            "a W0_qmunu create_dataset now carries data=; the placeholder "
            "would no longer be zeros and this whole gate needs re-reading")
    # ONE allocation with no write beside it — the placeholder — and the
    # flag that distinguishes them.  ``w0_ready`` starts False and is
    # stamped on the placeholder branch; the dedicated writer stamps True.
    assert 'attrs["W0_ready"] = w0_ready' in src
    assert 'attrs["W0_ready"] = True' in src
    assert "w0_ready = False" in src


# ---------------------------------------------------------------------------
# The fix, asserted where it lives
# ---------------------------------------------------------------------------

def _guards_on_w0_ready(source, binder):
    """Is every assignment matching ``binder`` guarded by a W0_ready test?

    Walks ``if`` statements, keeps the ones whose body performs the
    assignment ``binder`` names, and asks whether the CONDITION mentions
    ``W0_ready``.  Structural rather than a substring search, because a
    substring passes on the attr name appearing in a docstring three
    hundred lines away — which is the shape this most needs to reject.
    """
    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        body_src = ast.dump(ast.Module(body=node.body, type_ignores=[]))
        if binder not in body_src:
            continue
        cond = ast.dump(node.test)
        found.append("W0_ready" in cond)
    return found


def test_the_interpolation_path_gates_on_w0_ready_like_bse_io_does():
    """``vq_interp`` binds W0 only behind the persisted flag.

    The pre-fix line was ``if "W0_qmunu" in fr:`` — the presence question —
    while ``bse_io`` two modules over had been asking the right one all
    along.  Both are asserted here, so the two readers cannot drift apart
    again silently: if somebody relaxes either, this fails.
    """
    vq_guards = _guards_on_w0_ready(_VQ_INTERP.read_text(), "'W0'")
    assert vq_guards, (
        "no `if` in bse/vq_interp.py binds zx['W0'] any more; the gate "
        "has moved and this cell is no longer measuring it")
    assert all(vq_guards), (
        "bse/vq_interp.py binds zx['W0'] behind a condition that does not "
        "mention W0_ready.  Presence is not persistence: gw_init allocates "
        "a full-size ZERO W0 unconditionally, so `\"W0_qmunu\" in fr` is "
        "true on a run whose persist_w0 never fired.  That is the April "
        "all-zero-screening mechanism.")

    io_guards = _guards_on_w0_ready(_BSE_IO.read_text(), "wq_key")
    assert any(io_guards), (
        "bse/bse_io.py no longer gates its W0 selection on W0_ready")


def test_the_ast_matcher_detects_the_pre_fix_shape():
    """RED TWIN.  The matcher above must FAIL on the presence-only code.

    A structural check that returns "all guarded" for an empty result set,
    or that cannot tell the two conditions apart, would pass forever.  Both
    spellings are put through it: the line as it was, and the line as it
    is.
    """
    pre_fix = (
        'if "W0_qmunu" in fr:\n'
        '    zx["W0"] = fr["W0_qmunu"]\n')
    post_fix = (
        'if "W0_qmunu" in fr and bool('
        'fr["W0_qmunu"].attrs.get("W0_ready", False)):\n'
        '    zx["W0"] = fr["W0_qmunu"]\n')
    assert _guards_on_w0_ready(pre_fix, "'W0'") == [False]
    assert _guards_on_w0_ready(post_fix, "'W0'") == [True]
    # And a file that binds nothing yields an EMPTY result, which the cell
    # above rejects separately rather than reading as success.
    assert _guards_on_w0_ready("x = 1\n", "'W0'") == []


def test_build_hdir_names_the_diagnosis_instead_of_key_erroring_blankly():
    """With W0 unbound, the consumer must say WHY, not just KeyError.

    The loader has already made the diagnosis — the flag said the tensor
    was never persisted — and dropping that on the floor so the caller sees
    a bare ``KeyError: 'W0'`` is how a clear refusal becomes a puzzle.
    """
    from bse import vq_interp as vqi

    with pytest.raises(KeyError, match="persisted W0_qmunu"):
        vqi.build_hdir({"kgrid": np.array([1, 1, 1])}, 0)


def test_the_qirr_reader_makes_the_same_refusal():
    """The new format inherits the gate rather than the hazard.

    A q_irr restart file is the same tensor eight times smaller, so a
    placeholder in that format is the same incident with a smaller file.
    ``symmetry_maps.qirr_store`` stamps ``qirr_data_ready`` and its reader
    refuses on False; this cell is here so the two mechanisms are visible
    in one place, and it SKIPS rather than fails where the service is not
    on the path.
    """
    sm = pytest.importorskip("symmetry_maps")
    # THROUGH THE DOOR.  ``symmetry_maps.qirr_store`` is a submodule the
    # door does not re-export, so naming it here would be a past-the-door
    # reach — and one the layering ratchet could not even see, because an
    # importorskip string is invisible to its AST scan.  The door-exported
    # function knows where it lives.
    src = pathlib.Path(inspect.getsourcefile(sm.read_tensor)).read_text()
    assert "qirr_data_ready" in src
    assert "require_persisted" in src
    assert "PLACEHOLDER, not" in src, (
        "the qirr reader's placeholder refusal has moved or been reworded; "
        "check it still refuses rather than warning")


# ---------------------------------------------------------------------------
# The two holes the phase-3 map found in this same gate family
# ---------------------------------------------------------------------------
# Both are the SAME defect shape as the April incident — a reader that
# accepts something it cannot distinguish from good data — and neither was
# reachable from the W0 flag, which is why they survived the fix above.

def test_v_qmunu_carries_a_persisted_flag_too(tmp_path):
    """W0 has said "I was written" since April.  V now says it as well.

    The asymmetry was real and it was on ONE LINE: the full-file reader
    gated W0 on ``W0_ready`` and read ``V_qmunu`` unconditionally,
    immediately above it.  It was safe by accident — V has no placeholder
    path, so present implied written — and "the invariant happens to
    hold" is a different state from "the file says so".  Only the second
    survives a writer that grows one, which the q_irr work is about to.
    """
    path = tmp_path / "restart.h5"
    _placeholder_restart_or_skip(path)
    import h5py
    with h5py.File(path, "r") as f:
        assert bool(f["V_qmunu"].attrs["V_ready"]) is True
        # ...and the W0 placeholder beside it still says the opposite,
        # which is what makes this a flag and not a constant.
        assert bool(f["W0_qmunu"].attrs["W0_ready"]) is False


def test_the_writer_stamps_v_ready_where_it_stamps_w0_ready():
    """SOURCE cell — runs on a box with no FFI, like its W0 twin.

    ``_placeholder_restart_or_skip`` skips without the built SlabIO, so
    the behavioural cell above cannot be the only evidence.  Both stamps
    must live in the same rank-0 block, after SlabIO releases the file:
    stamping V from a different place would reintroduce the interleave
    the W0 comment warns about.
    """
    import file_io.tagged_arrays as ta

    src = pathlib.Path(inspect.getsourcefile(
        ta.write_restart_state_to_h5)).read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "write_restart_state_to_h5")
    stamped = {
        n.slice.value for n in ast.walk(fn)
        if isinstance(n, ast.Subscript)
        and isinstance(n.slice, ast.Constant)
        and isinstance(n.value, ast.Attribute)
        and n.value.attr == "attrs"
    }
    assert {"W0_ready", "V_ready"} <= stamped, (
        f"write_restart_state_to_h5 stamps {sorted(stamped)}; both "
        f"persisted flags must be set here, in one rank-0 block")


def test_a_file_that_says_v_was_never_persisted_is_refused(tmp_path):
    """The refusal, and the legacy path beside it.

    Three states, and the middle one is the whole backward-compatibility
    story: ABSENT means ready, because every restart file written before
    the attr existed carries nothing and must keep loading byte-for-byte.
    """
    import h5py
    from bse.bse_io import _refuse_unpersisted

    path = tmp_path / "v.h5"
    with h5py.File(path, "w") as f:
        legacy = f.create_dataset("legacy", shape=(NQ, N_MU, N_MU),
                                  dtype=np.complex128)
        ready = f.create_dataset("ready", shape=(NQ, N_MU, N_MU),
                                 dtype=np.complex128)
        ready.attrs["V_ready"] = True
        never = f.create_dataset("never", shape=(NQ, N_MU, N_MU),
                                 dtype=np.complex128)
        never.attrs["V_ready"] = False

        _refuse_unpersisted(legacy, "legacy", str(path))     # absent -> ok
        _refuse_unpersisted(ready, "ready", str(path))       # True   -> ok
        with pytest.raises(ValueError, match="never persisted"):
            _refuse_unpersisted(never, "never", str(path))


def test_both_readers_ask_before_reading_v():
    """SOURCE cell: the gate is at BOTH V reads, not just the one named.

    ``_load_ring_subset`` is the reader the map found; the sharded loader
    reads V too, and a gate on one of two readers is a gate a caller can
    route around by choosing a path.
    """
    src = _BSE_IO.read_text()
    tree = ast.parse(src)
    guarded = {
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and any(isinstance(c, ast.Call)
                and getattr(c.func, "id", None) == "_refuse_unpersisted"
                for c in ast.walk(n))
    }
    assert {"_load_ring_subset",
            "load_bse_data_from_restart_sharded"} <= guarded, (
        f"only {sorted(guarded)} refuse an unpersisted V")


def test_build_hdir_refuses_a_w0_whose_q_axis_is_not_the_full_bz():
    """The SILENT site, made loud.

    ``build_hdir`` indexes W0 by FULL-BZ q (``qkk`` comes from
    ``k_lookup``).  Hand it a q_irr wedge and every index is in range —
    it returns rows belonging to other q's and reports a plausible,
    wrong exciton spectrum.  Nothing downstream can tell.  This is the
    April failure SHAPE with a different cause, and the guard is asked
    of the lazy handle before any work happens.
    """
    from bse import vq_interp as vqi

    kg = np.array([2, 2, 2])            # 8 full-BZ q
    wedge = np.zeros((2, N_MU, N_MU), dtype=np.complex128)   # an IBZ wedge
    with pytest.raises(ValueError, match=r"2 q rows.*needs 8"):
        vqi.build_hdir({"W0": wedge, "kgrid": kg}, 0)


def test_the_full_bz_w0_gets_past_that_guard():
    """RED TWIN for the cell above: the guard is not simply always-raise.

    It must reject the wedge and pass the full-BZ tensor.  A correctly
    sized W0 gets through this check and fails LATER, on the deck data
    this synthetic ``zx`` does not carry — which is the proof that the q
    extent is what the previous cell measured and not an accident of the
    stub being incomplete.
    """
    from bse import vq_interp as vqi

    kg = np.array([2, 2, 2])
    full = np.zeros((8, N_MU, N_MU), dtype=np.complex128)
    with pytest.raises((KeyError, TypeError, IndexError, AttributeError)):
        vqi.build_hdir({"W0": full, "kgrid": kg}, 0)
