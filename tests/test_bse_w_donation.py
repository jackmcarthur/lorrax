"""Gates for the `W_R = ifft_q(W_q)` donation at the satellite BSE drivers.

`bse_lanczos` hoisted its W_R build to a real top-level boundary and DONATED
`W_q` there, because a jit parameter's buffer is owned by the caller for the
whole call: run the transform inside the solve and both `W_q` and `W_R` stay
resident for its entire duration.  Three other drivers build the same W_R with
the same helper at the same kind of boundary and never got the same treatment
(FFT_DONATION_AUDIT.md 2.3):

    davidson_absorption.py   bse_nontda.py   exciton_bands.py (x2 paths)

The audit measured that the *in-jit* peak is identical donated or not
(112.50 MiB/rank either way at this deck's W shape) — the whole win is
caller-side, one live W tile instead of two: **56.25 MiB/rank on the Si 4x4x4
deck, 404 MB/rank at mu=10015 / P=64, 4.1 GB/rank at mu=32k.**

Each cell below is paired with the failure it exists to catch:

* `test_donation_grants_the_alias` — RED TWIN for the MECHANISM.  Reads
  `input_output_alias` out of the compiled HLO, never from the presence of
  `donate_argnums` (which is a request, not a grant).  The undonated control
  must show none, or the cell is measuring nothing.
* `test_donation_frees_the_input_buffer` — RED TWIN for the CALLER-SIDE win,
  which is the only win there is: after a donated call the operand's buffer
  must actually be gone.
* `test_the_four_sites_donate_and_drop_the_reference` — RED TWIN for the
  WIRING, at all four call sites, including that the reference is dropped.
  Donation without dropping the Python reference leaves a deleted-buffer
  array in `data`, which is a worse failure than the leak it fixes.
* `test_coarse_w_path_is_not_donated` — the deliberate exception: the
  `--w-coarse-grid` path changes shape, so donation is declined there anyway
  AND it still needs `W_q` to build the sub-grid.  Pinned so nobody
  "completes" the change by donating it too.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from common.fft_helpers import make_sharded_ifftn_3d  # noqa: E402


def _mesh():
    from jax.sharding import Mesh
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))


def _w_and_ifftn(mu=8, nu=6, nk=(4, 4, 4)):
    """The production W_R build, at a small W shape with the deck's structure."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    mesh = _mesh()
    spec = P("x", "y", None, None, None)
    rng = np.random.default_rng(5)
    W_q = jnp.asarray(rng.normal(size=(mu, nu, *nk))
                      + 1j * rng.normal(size=(mu, nu, *nk)))
    with mesh:
        W_q = jax.device_put(W_q, NamedSharding(mesh, spec))
        ifftn = make_sharded_ifftn_3d(mesh, spec, spec, axes=(2, 3, 4),
                                      norm="ortho")
    return mesh, W_q, ifftn


def test_donation_grants_the_alias():
    """RED TWIN: read the ALIAS out of the HLO, not the request out of python.

    `donate_argnums` is a request.  XLA grants it only when some output can
    take the operand's buffer — same shape, same layout — and the audit's
    whole method is to check the grant.
    """
    mesh, W_q, ifftn = _w_and_ifftn()
    with mesh:
        donated = jax.jit(ifftn, donate_argnums=(0,)).lower(W_q).compile()
        plain = jax.jit(ifftn).lower(W_q).compile()
    d_txt, p_txt = donated.as_text(), plain.as_text()
    assert "input_output_alias" in d_txt, (
        "donate_argnums=(0,) was requested but XLA granted no alias — W_R is "
        "a second buffer and the donation buys nothing")
    assert "input_output_alias" not in p_txt, (
        "the undonated control ALSO aliases, so this cell cannot tell a "
        "granted donation from a compiler default and proves nothing")


def test_donation_frees_the_input_buffer():
    """RED TWIN for the only win there is: W_q's buffer must actually go."""
    mesh, W_q, ifftn = _w_and_ifftn()
    with mesh:
        W_R = jax.jit(ifftn, donate_argnums=(0,))(W_q)
        jax.block_until_ready(W_R)
    assert W_q.is_deleted(), (
        "W_q survived a donated call, so both tiles are still live and the "
        "caller-side win did not happen")
    # ...and the transform is still the transform.
    mesh2, W_q2, ifftn2 = _w_and_ifftn()
    with mesh2:
        ref = jax.block_until_ready(jax.jit(ifftn2)(W_q2))
    assert np.array_equal(np.asarray(W_R), np.asarray(ref)), (
        "donation changed the VALUE — it is supposed to change only which "
        "buffer the answer is written into")


@pytest.mark.parametrize("module,func,label", [
    ("bse.davidson_absorption", None, "davidson_absorption W_R build"),
    ("bse.bse_nontda", "_full_matvec_and_args", "bse_nontda W_R build"),
    ("bse.exciton_bands", None, "exciton_bands W_R build"),
])
def test_the_four_sites_donate_and_drop_the_reference(module, func, label):
    """RED TWIN for the WIRING at each site, including the dropped reference.

    Source-level, deliberately: each of these lives inside a driver `main()`
    that needs a loaded restart and a mesh to reach, which is a
    regression-scale fixture for a one-line question.  The mechanism itself is
    gated behaviourally above, and the end-to-end proof is a deck leg per
    driver (FIX_smallwins.md).
    """
    import importlib
    mod = importlib.import_module(module)
    src = inspect.getsource(getattr(mod, func) if func else mod)
    # Look only at the W_R build, so an unrelated donate_argnums elsewhere in
    # a 1100-line driver cannot make this cell pass by accident.
    assert "donate_argnums=(0,)" in src, \
        f"{label}: W_q is not donated — two live W tiles for the whole run"
    assert 'data["W_q"] = None' in src, (
        f"{label}: W_q was donated but the caller-side reference was not "
        f"dropped, which leaves a DELETED-buffer array in `data` — a worse "
        f"failure than the leak, and the reason bse_lanczos:212 exists")


def test_coarse_w_path_is_not_donated():
    """The deliberate exception, pinned so nobody 'completes' the change.

    `--w-coarse-grid` sub-samples W_q onto a coarse sub-grid before the
    densifier, so the operand shape changes and XLA declines the alias anyway
    — and that path still READS `data["W_q"]` to build the sub-grid, so
    donating would delete a buffer it is about to use.
    """
    from bse import exciton_bands as EB
    src = inspect.getsource(EB)
    i = src.index("decimate_W_q_to_subgrid(data[\"W_q\"]")
    window = src[max(0, i - 600):i]
    assert "_ifftn_donated" not in window.split("if cg == (nkx, nky, nkz):")[-1], (
        "the coarse-W branch appears to donate W_q, but it still needs it to "
        "build the coarse sub-grid")
