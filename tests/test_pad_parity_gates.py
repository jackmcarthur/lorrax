"""Parity gates for the four padded axes that had neither claim nor gate.

The rule these gates encode, established by measurement (ledger 0086/0087):

    A zero pad is INERT for operators linear or bilinear in the padded
    axis, and a WRONG NUMBER for a diagonalisation.

so each axis below is gated for the case it actually is, not uniformly:

* **BSE val/cond band pad** (``bse.bse_io``) — DIAGONALISATION.  ΔE =
  ε_c − ε_v is the diagonal of the operator every BSE driver
  diagonalises, so the ε pad is a signed sentinel (±``PAD_EPS_GUARD_RY``)
  and the padded modes are dropped BY COUNT.  ψ stays zero-padded, which
  is what decouples the pad block exactly.
* **W-solve q pad** (``gw.w_isdf``) — ELEMENT SELECTION.  Each q is an
  independent Dyson solve inside one ``fori_loop``; the pad adds trivial
  systems and is sliced off.  The parity class is therefore BIT EQUALITY,
  not a tolerance — checked as such.
* **htransform Galerkin rank pad** (``bandstructure.htransform``) —
  DIAGONALISATION, currently safe only because fH ⪯ 0 puts the injected
  exact zeros at the TOP of an ascending spectrum while every selection
  takes from the bottom.  That is an argued invariant with no assertion
  behind it, and it is what these gates assert.  ``LORRAX_EXTRA_RANK_PAD``
  is exercised here for the first time.
* **ψ spinor 2→4** (``common.wfn_transforms``) — NOT A PAD.
  ``meta.nspinor`` is categorical (``4 if bispinor``), never rounded to a
  device count, so there is no divisor and no inertness claim to make.
  Zero-filling it deletes the small components; the branch refuses now,
  and the gate is that the refusal is reachable and says so.

Scope, stated because a gate that hides its scope is worse than none:
these are SEAM gates.  They drive the real pad/strip/selection code with
synthetic operands on a single process.  They do not run a pipeline, do
not exercise Σ/W/ISDF physics, and do not cover real multi-process
collectives.  ``tests/test_invariance_gates.py`` is where the
end-to-end pad-flip gates live.

Deliberately NO ULP counts anywhere in this file.  0086 measured the
padding drift scaling with reduction length (0.2 eps at nk·nb²=243,
39.9 eps at 29768) and with trajectory (≤8.3 eps contracting, 2.9e5 eps
stalled), with the only bound that held being ``|ΔH| ≤ 6.1e-8`` of the
per-element RESIDUAL.  A test pinning a few ULPs passes on a fixture and
fails at production shapes; where the class is bit equality we check bit
equality, and where it is not we check the structural property.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════
#  1. BSE val/cond band pad — the diagonalisation case
# ═══════════════════════════════════════════════════════════════════════

def _bse_eps(nk, n_val, n_cond, nv_pad, nc_pad, *, guard):
    """(eps_v, eps_c) at the padded extents with a signed ``guard`` fill.

    Energies are referenced so that valence < 0 < conduction, the layout
    ``bse_jax``'s own synthetic fixture uses (``eps_v`` in [-0.5, -0.1],
    ``eps_c`` in [0.1, 0.5] Ry).
    """
    rng = np.random.default_rng(11)
    eps_v = np.full((nk, nv_pad), -guard, dtype=np.float64)
    eps_c = np.full((nk, nc_pad), guard, dtype=np.float64)
    eps_v[:, :n_val] = rng.uniform(-0.5, -0.1, size=(nk, n_val))
    eps_c[:, :n_cond] = rng.uniform(0.1, 0.5, size=(nk, n_cond))
    return jnp.asarray(eps_v), jnp.asarray(eps_c)


def _delta_e(eps_c, eps_v):
    """The BSE diagonal, spelled exactly as ``bse_ring_comm._apply_D_term``."""
    return eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]


def test_zero_eps_pad_puts_spurious_transitions_below_the_onset():
    """WHY the BSE ε pad is not zero — the wrong number, exhibited.

    This is the falsifiability twin of the gate below: it drives the OLD
    fill through the SAME diagonal construction and shows the failure is
    reachable, so the passing gate is not passing vacuously.
    """
    nk, n_val, n_cond, nv_pad, nc_pad = 3, 3, 3, 4, 4
    eps_v, eps_c = _bse_eps(nk, n_val, n_cond, nv_pad, nc_pad, guard=0.0)
    dE = np.asarray(_delta_e(eps_c, eps_v)).real

    onset = float(np.min(dE[:, :n_cond, :n_val, :]))
    pad = np.ones(dE.shape, dtype=bool)
    pad[:, :n_cond, :n_val, :] = False

    # The GLOBAL minimum of the padded diagonal is a pad transition — the
    # (c_pad, v_pad) family sits at exactly 0 — so "the lowest transition"
    # is a pad state, and a seed built from the lowest transitions starts
    # inside the decoupled pad block.
    assert float(np.min(dE[pad])) < onset, (
        "the zero-pad wrong number did not reproduce; this gate's twin is "
        "now vacuous and the sentinel gate below proves nothing")
    assert pad[np.unravel_index(int(np.argmin(dE)), dE.shape)], (
        "the global minimum transition is no longer a pad state")

    # The mixed families (c_pad, v_real) -> |eps_v| and (c_real, v_pad) ->
    # eps_c are the dangerous ones: they are POSITIVE, so the historical
    # `flat > 1e-12` value filter passed them, and each is smaller than
    # eps_c + |eps_v|, so some land under the onset.  Not ALL of them do —
    # that depends on the window — which is exactly why a value filter
    # cannot be the thing keeping them out.  Assert only what is
    # structural: a nonzero number of them get through.
    n_below = int(np.sum(dE[pad] < onset))
    assert n_below > 0
    survives_value_filter = dE[pad] > 1e-12
    assert int(np.sum(survives_value_filter & (dE[pad] < onset))) > 0, (
        "no pad transition both survives the old value filter AND falls "
        "below the onset — the failure this gate documents is unreachable "
        "on this fixture, so report UNFALSIFIABLE rather than PASS")


def test_sentinel_eps_pad_keeps_every_pad_transition_out_of_the_window():
    """The gate: with the shipped fill no pad transition can be selected."""
    from bse.bse_io import PAD_EPS_GUARD_RY

    nk, n_val, n_cond, nv_pad, nc_pad = 3, 3, 3, 4, 4
    eps_v, eps_c = _bse_eps(nk, n_val, n_cond, nv_pad, nc_pad,
                            guard=PAD_EPS_GUARD_RY)
    dE = np.asarray(_delta_e(eps_c, eps_v)).real

    phys = dE[:, :n_cond, :n_val, :]
    pad = np.ones(dE.shape, dtype=bool)
    pad[:, :n_cond, :n_val, :] = False

    assert float(np.min(dE[pad])) >= 0.5 * PAD_EPS_GUARD_RY, (
        "a pad transition is inside the physical window — check the SIGNS: "
        "eps_c pads +guard, eps_v pads -guard, so every pad DeltaE >= guard")
    assert float(np.max(phys)) < 0.5 * PAD_EPS_GUARD_RY

    # The physical block is untouched: the sentinel is a fill, and the
    # diagonal is elementwise, so this is BIT equality, not a tolerance.
    eps_v0, eps_c0 = _bse_eps(nk, n_val, n_cond, nv_pad, nc_pad, guard=0.0)
    dE0 = np.asarray(_delta_e(eps_c0, eps_v0)).real
    assert np.array_equal(phys, dE0[:, :n_cond, :n_val, :]), (
        "the sentinel moved a PHYSICAL transition; the ep pad must not be "
        "reachable from the logical block")


def test_pad_axis_fill_is_keyword_only_and_signed():
    """The seam itself: psi pads 0, eps pads +/-guard, BOTH extents are named.

    ``runtime.padding.pad_axis`` is the single implementation since
    2026-08-22.  It replaced two helpers whose second tuple element was the
    OPPOSITE extent from the same slot -- ``pad_axis_to`` returned the
    LOGICAL one, ``bse_window._pad_axis_to_multiple`` the PADDED one -- and
    the BSE helper's own comment recorded a wrong answer that came from
    exactly that confusion.  The repair is not "pick a convention": it is
    that a caller must NAME which extent it wants.
    """
    from runtime.padding import PadAxisResult, pad_axis
    from bse.bse_io import PAD_EPS_GUARD_RY

    x = jnp.ones((2, 3, 5), dtype=jnp.complex128)
    r = pad_axis(x, 4, axis=1)
    assert isinstance(r, PadAxisResult)
    # BOTH extents, and they are DIFFERENT here -- which is the whole case
    # the old single-value return could not express.
    assert (r.logical, r.padded) == (3, 4)
    assert r.array.shape[1] == 4
    assert np.all(np.asarray(r.array)[:, 3:, :] == 0.0), "psi pad must be zero"

    e = jnp.zeros((2, 3), dtype=jnp.float64)
    ep = pad_axis(e, 4, axis=1, fill=PAD_EPS_GUARD_RY)
    assert np.all(np.asarray(ep.array)[:, 3:] == PAD_EPS_GUARD_RY)
    em = pad_axis(e, 4, axis=1, fill=-PAD_EPS_GUARD_RY)
    assert np.all(np.asarray(em.array)[:, 3:] == -PAD_EPS_GUARD_RY)

    # Keyword-only: a positional fill would let a call site sign the guard
    # by accident, which is the one way to put pad modes back under the
    # onset.  Make that unspellable.  ``axis`` is keyword-only too, so a
    # positional third argument cannot become the fill.
    with pytest.raises(TypeError):
        pad_axis(e, 4, 1, PAD_EPS_GUARD_RY)   # noqa: B026

    # Divisible extent: unchanged OBJECT (byte-identical production path),
    # and the two extents coincide.
    same = pad_axis(x, 3, axis=1)
    assert same.logical == 3 and same.padded == 3 and same.array is x


def test_pad_axis_is_bit_identical_to_both_helpers_it_replaced():
    """The A/B that licensed the deletion.  BIT equality, not a tolerance.

    Both deleted bodies are re-stated here VERBATIM (they are four lines
    each) and driven over the cases that separate them: divisible and
    indivisible extents, zero and signed fills, complex and real dtypes.
    The parity class is bit equality because ``jnp.pad`` with a constant is
    a copy, not an arithmetic operation -- there is no reduction whose
    blocking could move (contrast the D10 ragged-vs-padded gate, which is
    1e-12 relative for exactly that reason).
    """
    from runtime.padding import pad_axis
    from bse.bse_io import PAD_EPS_GUARD_RY

    def _deleted_bse_helper(x, axis, multiple, *, fill=0.0):
        """``bse_window._pad_axis_to_multiple`` -> (padded, PADDED extent)."""
        size = x.shape[axis]
        pad = (-size) % multiple
        if pad == 0:
            return x, size
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (0, pad)
        return jnp.pad(x, pad_width, mode="constant",
                       constant_values=fill), size + pad

    def _deleted_runtime_helper(A, divisor, *, axis=-1):
        """``runtime.padding.pad_axis_to`` -> (padded, LOGICAL extent)."""
        from runtime.padding import round_up
        ax = int(axis) % int(A.ndim)
        n = int(A.shape[ax])
        n_pad = round_up(n, divisor)
        if n_pad == n:
            return A, n
        widths = [(0, 0)] * A.ndim
        widths[ax] = (0, n_pad - n)
        return jnp.pad(A, widths), n

    rng = np.random.default_rng(0)
    cases = [
        (jnp.asarray(rng.normal(size=(2, 5, 3))
                     + 1j * rng.normal(size=(2, 5, 3))), 4, 1, 0.0),
        (jnp.asarray(rng.normal(size=(2, 8, 3))), 4, 1, 0.0),      # divisible
        (jnp.asarray(rng.normal(size=(3, 7))), 4, 1, PAD_EPS_GUARD_RY),
        (jnp.asarray(rng.normal(size=(3, 7))), 4, 1, -PAD_EPS_GUARD_RY),
        (jnp.asarray(rng.normal(size=(2, 3, 9))), 5, -1, 0.0),     # NRHS pad
    ]
    for A, div, axis, fill in cases:
        new = pad_axis(A, div, axis=axis, fill=fill)
        old_bse, old_bse_ext = _deleted_bse_helper(
            A, axis % A.ndim, div, fill=fill)
        assert np.array_equal(np.asarray(new.array), np.asarray(old_bse)), (
            f"pad_axis diverged from the BSE helper at {A.shape} / {div}")
        assert new.padded == old_bse_ext, "PADDED extent moved"
        if fill == 0.0:
            old_rt, old_rt_ext = _deleted_runtime_helper(A, div, axis=axis)
            assert np.array_equal(np.asarray(new.array), np.asarray(old_rt))
            assert new.logical == old_rt_ext, "LOGICAL extent moved"


def test_there_is_exactly_one_mesh_divisibility_pad_helper():
    """Source gate: the deleted twin must not come back under any name.

    A second helper is not a style problem here -- the two that existed
    returned opposite extents from the same slot, so a call site copied
    from the wrong neighbour was wrong ONLY when the extent was not already
    a mesh multiple.  That is invisible on every mesh-divisible validated
    run, which is why this is a gate and not a review note.
    """
    src = Path(__file__).resolve().parents[1] / "src"
    hits = []
    for path in sorted(src.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        # Code tokens only (TASTE 17): the register narrative and the
        # PadAxisResult docstring both name the dead helper on purpose.
        code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        if "def _pad_axis_to_multiple" in code or "def pad_axis_to(" in code:
            hits.append(str(path.relative_to(src)))
    assert hits == [], (
        f"a second mesh-divisibility pad helper is back in {hits}; "
        "runtime.padding.pad_axis is the one implementation")


def test_fftgrid_clients_delegate_divisor_and_extent_arithmetic():
    """The kmeans/htransform seam has one spec and one arithmetic backend.

    This is deliberately a source gate: importing htransform initializes the
    communicator/FFI stack, while the property under test is which authority
    its static layout planning names.  Value/HLO parity remains in the driver
    and staged-reshard suites.
    """
    repo = Path(__file__).resolve().parents[1]
    src = repo / "src"
    layout_path = src / "common" / "wfn_layout.py"
    app_paths = (
        layout_path,
        src / "common" / "mtxel_sweep.py",
        src / "common" / "parallel_transport.py",
        src / "common" / "wfn_transforms.py",
        src / "common" / "psi_G_store.py",
        src / "bandstructure" / "htransform.py",
        src / "file_io" / "parallel_transport.py",
        src / "gw" / "kin_ion_io.py",
        src / "gw" / "qsgw_density.py",
        src / "gw" / "sc_iteration.py",
        src / "psp" / "get_dipole_mtxels.py",
    )
    mtxel = (src / "common" / "mtxel_sweep.py").read_text()
    bundle = (src / "gw" / "wavefunction_bundle.py").read_text()
    parallel = (src / "common" / "parallel_transport.py").read_text()
    wfn = (src / "common" / "wfn_transforms.py").read_text()
    store = (src / "common" / "psi_G_store.py").read_text()
    ht = (src / "bandstructure" / "htransform.py").read_text()
    staged = (src / "common" / "staged_reshard.py").read_text()
    fit = (src / "common" / "sharding_fit.py").read_text()
    gflat_body = wfn.split("def gflat_to_rmu(", 1)[1].split("\ndef ", 1)[0]
    accumulate_body = wfn.split(
        "def accumulate_rchunk_to_gflat(", 1)[1].split("\ndef ", 1)[0]
    centroid_body = wfn.split(
        "def load_centroids_band_chunked(", 1)[1].split("\ndef ", 1)[0]
    galerkin_body = ht.split("def streaming_galerkin_solve(", 1)[1].split(
        "\ndef ", 1)[0]
    move_body = staged.split("def band_to_product_r_reshard(", 1)[1].split(
        "\ndef ", 1)[0]
    subset_body = fit.split("def _largest_divisible_subset(", 1)[1].split(
        "\ndef ", 1)[0]

    # Code-token census, not a grep through comments/docstrings (TASTE 17):
    # the application-side ψ(G-flat) clients may contain exactly one literal,
    # at the dependency-light authority.  A retyped literal in any live client
    # makes this fail even if that client still imports the authority too.
    def is_band_sphere_literal(node):
        if not isinstance(node, ast.Call) or len(node.args) != 4:
            return False
        if not isinstance(node.func, ast.Name) or node.func.id != "P":
            return False
        first, axes, third, fourth = node.args
        if not all(isinstance(arg, ast.Constant) and arg.value is None
                   for arg in (first, third, fourth)):
            return False
        if not isinstance(axes, (ast.Tuple, ast.List)) or len(axes.elts) != 2:
            return False
        return [getattr(item, "value", None) for item in axes.elts] == ["x", "y"]

    literal_sites = {
        path.relative_to(src).as_posix(): [
            node.lineno for node in ast.walk(ast.parse(path.read_text()))
            if is_band_sphere_literal(node)
        ]
        for path in app_paths
    }
    literal_sites = {path: lines for path, lines in literal_sites.items() if lines}
    assert list(literal_sites) == ["common/wfn_layout.py"], literal_sites
    assert len(literal_sites["common/wfn_layout.py"]) == 1, literal_sites

    # The two low-memory face layouts likewise have one common-layer
    # spelling.  The GW bundle re-exports them for compatibility, while the
    # finite-q common-layer endpoint consumes the common owner directly.
    layout_tree = ast.parse(layout_path.read_text())
    layout_names = {
        target.id
        for node in ast.walk(layout_tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert {"PSI_NMU_SPEC", "PSI_MUN_SPEC"} <= layout_names
    bundle_tree = ast.parse(bundle)
    bundle_assignments = {
        target.id
        for node in ast.walk(bundle_tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert not ({"PSI_NMU_SPEC", "PSI_MUN_SPEC"} & bundle_assignments)
    assert "from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC" in bundle
    assert "from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC, band_sphere_spec" in mtxel
    assert "NamedSharding(geom.mesh, PSI_NMU_SPEC)" in mtxel
    assert "NamedSharding(geom.mesh, PSI_MUN_SPEC)" in mtxel
    endpoint = next(
        node for node in ast.parse(mtxel).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "finite_transfer_current_to_centroids")
    old_face_literals = {
        (None, "x", None, None, "y"),
        (None, None, None, "x", "y"),
    }
    endpoint_literals = {
        tuple(arg.value for arg in node.args)
        for node in ast.walk(endpoint)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "P"
        and all(isinstance(arg, ast.Constant) for arg in node.args)
    }
    assert endpoint_literals.isdisjoint(old_face_literals), endpoint_literals

    # The old mtxel_sweep location is not a compatibility facade.  Scan import
    # nodes over the bounded source root so a new indirect consumer cannot make
    # two apparent owners without reintroducing the literal itself.
    legacy_imports = []
    for root in (src, repo / "tests"):
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.ImportFrom)
                        and node.module == "common.mtxel_sweep"
                        and any(alias.name == "band_sphere_spec"
                                for alias in node.names)):
                    legacy_imports.append(
                        f"{path.relative_to(repo).as_posix()}:{node.lineno}")
    assert legacy_imports == [], legacy_imports
    mtxel_all = next(
        node.value for node in ast.walk(ast.parse(mtxel))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets)
    )
    assert "band_sphere_spec" not in {
        item.value for item in mtxel_all.elts if isinstance(item, ast.Constant)
    }

    assert "GFLAT_LOAD_SPEC" not in wfn
    assert "band_sphere_spec()" in store
    assert "p = spec_divisor(mesh_xy, band_sphere_spec(), axis=1)" in store
    assert "p = int(mesh_xy.shape['x']) * int(mesh_xy.shape['y'])" not in store
    assert "p_prod    = spec_divisor(mesh, band_sphere_spec(), axis=1)" in gflat_body
    assert "np.prod([mesh.shape" not in gflat_body
    assert "mu_gflat_spec = P(None, ('x', 'y'), None)" in accumulate_body
    assert "p_prod    = spec_divisor(mesh, mu_gflat_spec, axis=1)" in accumulate_body
    assert "np.prod([mesh.shape" not in accumulate_body
    assert "p_band = spec_divisor(mesh_xy, sharding_load, axis=1)" in centroid_body
    assert "nb_padded_global = round_up(nb_total, p_band)" in centroid_body
    assert "(nb_total + n_devices - 1) // n_devices" not in centroid_body
    assert "divisor = spec_divisor(mesh, band_sphere_spec(), axis=1)" in parallel
    assert "_p_band = spec_divisor(mesh_xy, band_sphere_spec(), axis=1)" in galerkin_body
    assert "_p_band = max(1, int(mesh_xy.size))" not in galerkin_body
    assert "n_pad = round_up(nq, batch_size) - nq" in ht
    assert "n_pad = (-nq) % batch_size" not in ht
    assert "band_divisor = spec_divisor(mesh, in_spec, axis=1)" in move_body
    assert "r_divisor = spec_divisor(mesh, out_spec, axis=3)" in move_body
    assert "ndev = p_x * p_y" not in move_body
    assert "m_pad = round_up(m_loc, p_y)" in staged
    assert "m_pad = -(-m_loc // p_y) * p_y" not in staged
    assert "p = shard_factor(mesh, picked)" in subset_body
    assert "p *= int(mesh.shape[a])" not in subset_body


def test_init_bse_subspace_drops_the_pad_by_count_not_by_value():
    """The Davidson seed must have exact-zero support on the pad block.

    Driven with a ZERO eps pad on purpose: the old code's value filter
    (``flat > 1e-12``) was only ever correct for the (c_pad, v_pad)
    family, and this pins that the count is what does the work now — the
    seed must be clean even when the sentinel is absent.
    """
    from bse.bse_davidson_helpers import init_bse_subspace

    nk, n_val, n_cond, nv_pad, nc_pad = 3, 3, 3, 4, 4
    eps_v, eps_c = _bse_eps(nk, n_val, n_cond, nv_pad, nc_pad, guard=0.0)

    V = np.asarray(init_bse_subspace(
        eps_c, eps_v, n_eig=6, n_cond=n_cond, n_val=n_val,
        n_random=3, seed=3))
    assert V.shape == (6, nc_pad, nv_pad, nk)

    pad = np.ones(V.shape[1:], dtype=bool)
    pad[:n_cond, :n_val, :] = False
    leaked = float(np.max(np.abs(V[:, pad])))
    assert leaked == 0.0, (
        f"trial subspace has {leaked:.3e} amplitude on the padded, decoupled "
        f"block — a solver started there converges to pad modes")
    # and it is not vacuous: the physical block must be populated.
    assert float(np.max(np.abs(V[:, :n_cond, :n_val, :]))) > 0.0


def test_init_bse_subspace_without_counts_still_leaks_the_pad():
    """Falsifiability twin: omitting the counts reproduces the old failure.

    ``n_cond``/``n_val`` default to None ("the arrays are unpadded"), which
    is the pre-change behaviour.  If this ever stops leaking, the gate
    above has stopped testing the count.
    """
    from bse.bse_davidson_helpers import init_bse_subspace

    nk, n_val, n_cond, nv_pad, nc_pad = 3, 3, 3, 4, 4
    eps_v, eps_c = _bse_eps(nk, n_val, n_cond, nv_pad, nc_pad, guard=0.0)
    V = np.asarray(init_bse_subspace(eps_c, eps_v, n_eig=6, n_random=3, seed=3))
    pad = np.ones(V.shape[1:], dtype=bool)
    pad[:n_cond, :n_val, :] = False
    assert float(np.max(np.abs(V[:, pad]))) > 0.0


def test_init_bse_subspace_is_byte_identical_when_the_extent_divides():
    """No pad ⇒ no change, byte for byte — the 0086 contract."""
    from bse.bse_davidson_helpers import init_bse_subspace

    nk, nb = 3, 4
    eps_v, eps_c = _bse_eps(nk, nb, nb, nb, nb, guard=0.0)
    a = np.asarray(init_bse_subspace(eps_c, eps_v, n_eig=5, n_random=2, seed=1))
    b = np.asarray(init_bse_subspace(eps_c, eps_v, n_eig=5, n_cond=nb,
                                     n_val=nb, n_random=2, seed=1))
    assert np.array_equal(a, b), (
        "passing the logical counts changed a run whose extent already "
        "divides; the pad path must be a no-op there")


def test_pad_zone_mask_counts_match_the_bundle():
    from bse.bse_io import pad_zone_mask_np, n_pad_transitions

    m = pad_zone_mask_np(3, 2, 4, 4, 5)
    assert m.shape == (1, 4, 4, 5)
    assert int(m.sum()) == 3 * 2 * 5
    data = {"nkx": 5, "nky": 1, "nkz": 1, "n_cond": 3, "n_val": 2,
            "n_cond_pad": 4, "n_val_pad": 4}
    assert n_pad_transitions(data) == 5 * (16 - 6)


# ═══════════════════════════════════════════════════════════════════════
#  2. W-solve q pad — the element-selection case (bit equality)
# ═══════════════════════════════════════════════════════════════════════

def _dyson_reference(V, chi, pref, n_log):
    """The logical-extent Dyson solve, per q, outside any pad machinery."""
    out = np.zeros_like(np.asarray(V))
    for iq in range(V.shape[0]):
        A = np.eye(n_log) - np.asarray(V[iq])[:n_log, :n_log] @ (
            pref * np.asarray(chi[iq])[:n_log, :n_log])
        out[iq, :n_log, :n_log] = np.linalg.solve(
            A, np.asarray(V[iq])[:n_log, :n_log])
    return out


def test_w_solve_q_pad_is_stripped_and_bit_exact_on_the_logical_q():
    """The q pad adds INDEPENDENT trivial systems, so parity is bit equality.

    ledger 0087 recorded this pad as one that "leaks the padded q into
    ``W_flat``".  It does not: ``_solve_w`` slices ``W_flat[:nq_local]``
    before the output sharding constraint, and has since the pad was
    introduced (``1971df9``).  What is true is that it had no gate.

    The q axis is not a contraction axis — each q is its own
    ``lu_factor``/``lu_solve`` inside one ``fori_loop`` — so a pad q
    contributes A = I, RHS = 0 and no reduction sees it.  That makes the
    class element SELECTION and the right check bit equality, exactly as
    ledger 85/128 argue for the loader hyperslabs.
    """
    from runtime.padding import round_up

    nq_local, n = 5, 4
    ndev = 4
    rng = np.random.default_rng(5)
    V = jnp.asarray(rng.standard_normal((nq_local, n, n)) * 0.1)
    chi = jnp.asarray(rng.standard_normal((nq_local, n, n)) * 0.1)
    pref = 0.7

    nq_padded = round_up(nq_local, ndev)
    assert nq_padded > nq_local, "fixture must actually pad, or this is vacuous"
    pad = nq_padded - nq_local

    Vp = jnp.pad(V, ((0, pad), (0, 0), (0, 0)))
    chip = jnp.pad(chi, ((0, pad), (0, 0), (0, 0)))

    ref = _dyson_reference(V, chi, pref, n)
    got = _dyson_reference(Vp, chip, pref, n)

    # (a) the pad q rows solve to exactly zero — A = I, RHS = 0.
    assert np.array_equal(got[nq_local:], np.zeros_like(got[nq_local:])), (
        "pad q block is not exactly zero; the strip would then hide a "
        "nonzero the sharding constraint could still read")
    # (b) the logical q rows are BIT identical, not close.
    assert np.array_equal(got[:nq_local], ref), (
        "padding the q axis moved a logical q — the q axis is being "
        "reduced over somewhere, which would change this pad's class")


def test_w_isdf_strips_the_q_pad_at_the_seam():
    """Source-level: the strip exists and is not conditional on the plan."""
    src = (Path(__file__).resolve().parents[1] / "src" / "gw" / "w_isdf.py"
           ).read_text()
    assert "W_flat = W_flat[:nq_local]" in src, (
        "the q-pad strip in _solve_w is gone — the padded q would reach "
        "W_flat and every consumer of the (nq, mu, mu) stack")


# ═══════════════════════════════════════════════════════════════════════
#  3. htransform Galerkin rank pad — LORRAX_EXTRA_RANK_PAD, first exercise
# ═══════════════════════════════════════════════════════════════════════

def test_resolve_extra_rank_pad_reads_the_env_and_refuses_garbage():
    """``LORRAX_EXTRA_RANK_PAD`` had ONE read site and zero tests that set it.

    ``tests/test_layering.py`` names it in ``_L1_LIBRARY_ENV_READS``, but
    both consumers of that dict are ``ast.parse`` static analysis — they
    never import the module, never touch ``os.environ``, and cannot tell a
    working resolver from a dead one.  This is the first test that runs it.
    """
    from bandstructure.htransform import resolve_extra_rank_pad

    keep = os.environ.pop("LORRAX_EXTRA_RANK_PAD", None)
    try:
        assert resolve_extra_rank_pad() == 0
        os.environ["LORRAX_EXTRA_RANK_PAD"] = "8"
        assert resolve_extra_rank_pad() == 8
        os.environ["LORRAX_EXTRA_RANK_PAD"] = "  "
        assert resolve_extra_rank_pad() == 0
        # both refusal branches — neither had a red twin before now.
        os.environ["LORRAX_EXTRA_RANK_PAD"] = "-1"
        with pytest.raises(ValueError):
            resolve_extra_rank_pad()
        os.environ["LORRAX_EXTRA_RANK_PAD"] = "not-an-int"
        with pytest.raises(ValueError):
            resolve_extra_rank_pad()
    finally:
        os.environ.pop("LORRAX_EXTRA_RANK_PAD", None)
        if keep is not None:
            os.environ["LORRAX_EXTRA_RANK_PAD"] = keep


def test_extra_rank_pad_only_adds_mesh_aligned_null_directions(monkeypatch):
    """The knob's arithmetic: rank stays lcm-aligned and only grows."""
    import math
    from bandstructure.htransform import resolve_extra_rank_pad
    from runtime.padding import round_up

    align = math.lcm(2, 4)          # a 2x4 mesh
    rank_phys = 37

    monkeypatch.delenv("LORRAX_EXTRA_RANK_PAD", raising=False)
    base = round_up(rank_phys, align)

    for extra in (1, 8, 16):
        monkeypatch.setenv("LORRAX_EXTRA_RANK_PAD", str(extra))
        got = round_up(base + resolve_extra_rank_pad(), align)
        assert got % align == 0, "the knob broke mesh alignment"
        assert got > base, "the knob did not add pad directions"
        assert got - rank_phys >= extra


def test_fh_pad_zeros_sort_above_every_selected_band():
    """The invariant the rank pad's safety rests on, asserted for the first time.

    ``build_fH_R`` forms ``fH_k = -Σ_n w_n c_n c_nᴴ`` with ``w_n = -f(ε_n)
    ≥ 0``, so **fH is negative semidefinite**.  ``eigvalsh`` returns
    ascending, band selection is ``[b_min:b_max]`` from the bottom, and
    the ``n_pad`` extra exact-zero eigenvalues therefore sort ABOVE every
    selected band.  That argument is written out at ``htransform.py:381``
    and nothing checked it.

    It is fragile in one specific way and the gate says so: the sign is
    load bearing.  A LARGE POSITIVE sentinel on this pad block would also
    be safe; a large NEGATIVE one would sort below every physical band and
    silently steal ``n_pad`` band slots.  So the rank pad must NOT be
    "made consistent" with the BSE sentinel without flipping its sign.
    """
    nb, rank_phys, n_pad = 3, 5, 2
    rank = rank_phys + n_pad
    rng = np.random.default_rng(19)
    ctilde = (rng.standard_normal((nb, rank_phys))
              + 1j * rng.standard_normal((nb, rank_phys)))
    w = rng.uniform(0.2, 1.0, size=nb)                # = -f(eps) >= 0
    weighted = ctilde * np.sqrt(w)[:, None]
    fH_phys = -(weighted.conj().T @ weighted)         # (rank_phys, rank_phys)

    ev_phys = np.linalg.eigvalsh(fH_phys)
    assert np.all(ev_phys <= 1e-12), (
        "fH is not negative semidefinite — the whole reason the rank pad's "
        "exact-zero eigenvalues are harmless has stopped holding")

    fH_pad = np.zeros((rank, rank), dtype=complex)
    fH_pad[:rank_phys, :rank_phys] = fH_phys
    ev_pad = np.linalg.eigvalsh(fH_pad)

    # The pad injects exactly n_pad zeros, and they land at the TOP.
    assert ev_pad.shape[0] == rank
    np.testing.assert_allclose(ev_pad[:nb], ev_phys[:nb], atol=1e-12,
                               err_msg="the pad moved a SELECTED band")
    assert np.all(ev_pad[-n_pad:] >= ev_pad[nb - 1]), (
        "the injected zeros did not sort above the selected window")


def test_a_negative_rank_sentinel_would_steal_band_slots():
    """Falsifiability twin for the sign: the failure mode is reachable."""
    nb, rank_phys, n_pad = 3, 5, 2
    rank = rank_phys + n_pad
    rng = np.random.default_rng(19)
    ctilde = (rng.standard_normal((nb, rank_phys))
              + 1j * rng.standard_normal((nb, rank_phys)))
    w = rng.uniform(0.2, 1.0, size=nb)
    weighted = ctilde * np.sqrt(w)[:, None]
    fH_phys = -(weighted.conj().T @ weighted)
    ev_phys = np.linalg.eigvalsh(fH_phys)

    fH_bad = np.zeros((rank, rank), dtype=complex)
    fH_bad[:rank_phys, :rank_phys] = fH_phys
    for i in range(rank_phys, rank):
        fH_bad[i, i] = -1.0e10                        # the WRONG sign
    ev_bad = np.linalg.eigvalsh(fH_bad)
    assert not np.allclose(ev_bad[:nb], ev_phys[:nb]), (
        "a negative sentinel no longer displaces the selected window; the "
        "sign warning in the gate above is now untested")


# ═══════════════════════════════════════════════════════════════════════
#  4. ψ spinor 2→4 — not a pad; the refusal
# ═══════════════════════════════════════════════════════════════════════

def test_spinor_zero_fill_refuses_and_names_the_small_components():
    from common.wfn_transforms import _refuse_spinor_zero_fill

    with pytest.raises(ValueError) as exc:
        _refuse_spinor_zero_fill(4, 2, origin="unit-test")
    msg = str(exc.value)
    assert "bispinor=True" in msg, "the refusal must name the fix"
    assert "small components" in msg or "psi_S" in msg


def test_no_spinor_axis_zero_fill_survives_in_the_tree():
    """Static gate: the 2→4 zero fill must not come back anywhere.

    Scans the three sites the survey found and asserts each now routes to
    the refusal.  Falsifiability is exercised by
    ``test_spinor_zero_fill_gate_can_fail`` below, which injects the old
    spelling into a real file from this same scanned tree.
    """
    root = Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        if "meta.nspinor) > ns_" not in text:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "meta.nspinor) > ns_" in line:
                nxt = text.splitlines()[i:i + 3]
                if not any("_refuse_spinor_zero_fill" in ln for ln in nxt):
                    offenders.append(f"{path}:{i}")
    assert not offenders, (
        "spinor-extent mismatch is being zero-filled again at: "
        + ", ".join(offenders))


def test_spinor_zero_fill_gate_can_fail(tmp_path):
    """The gate above, run against a REAL file from the tree with a
    canonical violation injected — so "green" cannot mean "unreachable".

    A rule whose allowlist covers 100% of violations is UNFALSIFIABLE, not
    PASS.  This re-runs the same scan over a copy of an actual source file
    with the pre-change spelling restored, and requires it to go red.
    """
    real = (Path(__file__).resolve().parents[1] / "src" / "gw" / "kin_ion_io.py")
    text = real.read_text()
    assert "_refuse_spinor_zero_fill" in text, "fixture file no longer relevant"
    injected = text.replace(
        "        from common.wfn_transforms import _refuse_spinor_zero_fill\n"
        "        _refuse_spinor_zero_fill(int(meta.nspinor), ns_have,\n"
        "                                 origin=\"kin_ion_io._load_rotated_occ_fftbox\")",
        "        psi_g = jnp.pad(psi_g, ((0, 0), (0, 0),\n"
        "                                (0, int(meta.nspinor) - ns_have), (0, 0)))")
    assert injected != text, (
        "could not inject the canonical violation into a real file — the "
        "gate's failure case is unreachable and it must report UNFALSIFIABLE")

    victim = tmp_path / "kin_ion_io.py"
    victim.write_text(injected)

    offenders = []
    for i, line in enumerate(injected.splitlines(), 1):
        if "meta.nspinor) > ns_" in line:
            nxt = injected.splitlines()[i:i + 3]
            if not any("_refuse_spinor_zero_fill" in ln for ln in nxt):
                offenders.append(f"{victim}:{i}")
    assert offenders, (
        "the scan did not flag an injected zero fill in a real tree file; "
        "the green result above is vacuous")
