"""The rCROP band axes are PADDED, not degraded to an unsharded history.

``_run_rcrop`` used to carry a ``DEGRADE, DO NOT REFUSE`` branch: a band
count that did not divide the mesh fell back to an unsharded history, i.e.
to the 92.2 GB-on-one-device wall the function's own residency budget
exists to describe.  It pads both band axes instead.

These cells pin the four things that make the pad safe, and they are AST
plus source text -- no jax, no devices, no ``XLA_FLAGS`` -- so they run
anywhere and cannot leak a device-count environment into a sibling test
(see the combined-invocation failure ledgered 2026-08-04).

What they deliberately do NOT pin is a ULP count on the iterate.  The pad
is reduction-order gauge and the drift is trajectory-dependent: measured
0.2 eps after one iteration at nk*nb^2 = 243, 39.9 eps at nk*nb^2 = 29768,
and on a stalled trajectory it reaches 2.9e5 eps by iteration 12.  A cell
asserting "within a few ULPs" would pass on the fixture and fail at
production shapes, which is worse than no cell at all.  The bound that
does hold is against the RESIDUAL, and it belongs in a numerical gate with
a real trajectory, not here.
"""
import ast
import pathlib

import pytest


_PATH = (pathlib.Path(__file__).resolve().parents[1]
         / "src" / "gw" / "sc_iteration.py")
_SRC = _PATH.read_text()
_TREE = ast.parse(_SRC)


def _block(name):
    """Source text of a top-level ``def``, up to the next top-level one."""
    start = _SRC.index(f"\ndef {name}(") + 1
    nxt = _SRC.find("\ndef ", start + 1)
    return _SRC[start: nxt if nxt != -1 else len(_SRC)]


_RCROP = _block("_run_rcrop")


def test_the_unsharded_degrade_branch_is_gone():
    """The failure signature, not a success marker.

    ``entry_sh = None`` is the whole degrade: it is what made the history
    stacks uncommitted and landed 2*m copies of the carry on one device.
    If it comes back, every residency number in the docstring is fiction.
    """
    assert "entry_sh = None" not in _RCROP
    assert "history NOT sharded" not in _RCROP


def test_the_band_divisor_comes_from_the_spec_not_from_the_mesh_axes():
    """``spec_divisor``, the same expression every other band pad uses.

    Deriving it from ``px`` and ``py`` directly over-pads whenever the spec
    replicates a band axis, and it is exactly the drift that made
    ``_resolve_sc_eigh`` take its divisor from the spec (24e341d).
    """
    assert "spec_divisor(mesh, spec, 1)" in _RCROP
    assert "spec_divisor(mesh, spec, 2)" in _RCROP
    # one extent for both axes, or the carry stops being square
    assert "_math.lcm(" in _RCROP


def test_the_pad_goes_through_the_shared_helper():
    """``runtime.padding.pad_axis`` -- not a hand-rolled ``jnp.pad``.

    The helper is what guarantees the no-op case returns the SAME array,
    which is what makes a divisible nb byte-identical to the pre-pad code.
    A local ``jnp.pad`` would allocate even at zero pad and lose that.
    """
    assert "from runtime.padding import" in _RCROP
    assert "pad_axis(A, band_div, axis=1).array" in _RCROP
    assert "pad_axis(A, band_div, axis=2).array" in _RCROP
    assert "jnp.pad(" not in _RCROP


def test_the_carry_reaches_the_map_at_the_logical_extent():
    """THE correctness condition for the pad, and it is not about memory.

    ``gw_iteration_map`` and ``eigvalsh_kshard`` must never see the padded
    H.  A zero-padded Hermitian matrix has (nb_pad - nb) extra EXACT-ZERO
    eigenvalues, and ``eigvalsh`` would fold them into the RMS-DeltaE
    history and into any occupation/E_F logic downstream -- a wrong number,
    not a gauge shift.  ``_to_carry`` slicing is what prevents it.
    """
    assert "A[:, :nb, :nb]" in _RCROP


def test_the_tolerance_is_built_from_the_logical_element_count():
    """Counting pad modes would loosen the tolerance for free.

    ``n_elem`` is taken from ``H0.shape`` before any pad, and the pad modes
    contribute exactly zero to the residual 2-norm, so the converted
    tolerance stays the per-element RMS it claims to be.
    """
    fn = next(n for n in ast.walk(_TREE)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_rcrop")
    # n_elem is assigned from nk/nb, which are unpacked from H0.shape
    assert "nk, nb, _ = H0.shape" in _RCROP
    assert "n_elem = nk * nb * nb" in _RCROP
    # ...and that assignment precedes the pad, so it cannot see nb_pad
    assert _RCROP.index("n_elem = nk * nb * nb") < _RCROP.index("nb_pad =")
    del fn


def test_inertness_is_checked_at_runtime_and_separately_from_parity():
    """Two different claims; this session measured them coming apart.

    The pad modes were bit-for-bit 0.0 in every configuration while the
    iterate still moved by reduction order.  So a comment asserting
    inertness is not evidence -- the check has to run.
    """
    assert "pad inertness" in _RCROP
    assert "result.x[:, nb:, :]" in _RCROP
    assert "result.x[:, :, nb:]" in _RCROP


@pytest.mark.parametrize("phrase", [
    "reduction-order",
    "NOT BIT-EXACT",
    "WHAT BOUNDS IT IS THE RESIDUAL",
])
def test_the_parity_contract_is_stated_and_is_not_a_bit_exactness_claim(phrase):
    """The contract is subtle enough that losing it in an edit is likely.

    Specifically it must not degrade into "padding is bit-exact": that is
    false, it was measured false, and a future reader who believes it will
    write the ULP test this module's docstring explains cannot hold.  The
    residual bound is the load-bearing half -- drop it and the remaining
    text reads as "this change moves numbers", with nothing saying by how
    little relative to what.
    """
    contract = _RCROP[_RCROP.index("PARITY CONTRACT"):]
    assert phrase in contract
