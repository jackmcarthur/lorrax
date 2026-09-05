"""Tier-0 internal full-frequency route: config and refusal contracts."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from gw.gw_config import (
    LorraxConfig, SigmaFrequencyRoute, coerce_sigma_frequency_route)
from gw.internal_ff_cd import (
    REAL_MAX_EV, RESPONSE_WIDTHS_EV, _load_checkpoint,
    _real_coefficients, _save_checkpoint, imag_grid, real_grid)


BASE = """\
[cohsex]
nval = 2
ncond = 2
number_bands = 10
memory_per_device_gb = 4.0
"""

METAL = """\
compute_mode = mpa
occ_smearing_family = mp1
occ_smearing_width_ry = 0.02
fermi_reference = mp1_fixed_n
"""


def config(tmp_path, extra=""):
    path = tmp_path / "tier0.in"
    path.write_text(BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *args, **kwargs: None)


def test_route_axis_default_and_exact_spellings(tmp_path):
    assert config(tmp_path).sigma.freq_route is SigmaFrequencyRoute.MPA
    cfg = config(
        tmp_path, METAL
        + "sigma_freq_route = internal_ff_cd\n"
        + "qp_solver = one_shot_dft\n")
    assert cfg.sigma.freq_route is SigmaFrequencyRoute.INTERNAL_FF_CD
    assert coerce_sigma_frequency_route(" INTERNAL_FF_CD ") \
        is SigmaFrequencyRoute.INTERNAL_FF_CD
    with pytest.raises(ValueError, match="sigma_freq_route"):
        coerce_sigma_frequency_route("internal_ff_pade")


@pytest.mark.parametrize("extra,match", [
    ("compute_mode = gn_ppm\nsigma_freq_route = internal_ff_cd\n",
     "compute_mode = mpa"),
    (METAL + "sigma_freq_route = internal_ff_cd\n"
     + "qp_solver = fixed_point\n", "one_shot_dft"),
    ("compute_mode = mpa\nsigma_freq_route = internal_ff_cd\n",
     "occ_smearing_family = mp1"),
])
def test_route_scope_is_refused_at_config_resolution(tmp_path, extra, match):
    with pytest.raises(ValueError, match=match):
        config(tmp_path, extra)


def test_referee_grids_are_fixed_covering_and_nested():
    for width in RESPONSE_WIDTHS_EV:
        grid = real_grid(width)
        assert grid[0] == 0.0
        assert grid[-1] == REAL_MAX_EV
        assert np.all(np.diff(grid) > 0.0)
    grid = imag_grid()
    assert grid[0] == 0.0 and grid[-1] == 100.0
    assert np.all(np.diff(grid) > 0.0)


def test_streamed_real_coefficients_are_exact_linear_interpolation():
    grid = real_grid(RESPONSE_WIDTHS_EV[-1])
    x = np.asarray([[[0.003]], [[4.137]], [[65.411]]])
    sign = np.asarray([[[-1.0]], [[0.37]], [[1.0]]])
    reconstructed = sum(
        _real_coefficients(grid, iw, x, sign) * grid[iw]
        for iw in range(grid.size))
    np.testing.assert_allclose(reconstructed, sign * x, rtol=0.0, atol=1e-12)


def test_no_host_materialization_of_chi_or_w_in_tier0_module():
    source = (Path(__file__).parents[1] / "src/gw/internal_ff_cd.py").read_text()
    tree = ast.parse(source)
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = ast.unparse(node.func)
        arg_node = node.args[0]
        arg = ast.unparse(arg_node)
        if (name in ("np.asarray", "np.array")
                and isinstance(arg_node, ast.Name)
                and arg_node.id in ("chi", "chi_bq", "w_wedge",
                                    "wc_wedge", "wc_full")):
            forbidden.append((name, arg))
    assert forbidden == [], (
        "Tier 0 gathered an N_mu^2 chi/W object onto a process: "
        f"{forbidden}")


def test_input_reference_registers_route_key():
    text = (Path(__file__).parents[1] / "docs/input_reference.md").read_text()
    row = next(line for line in text.splitlines()
               if line.startswith("| `sigma_freq_route`"))
    assert "internal_ff_cd" in row and "mpa" in row


def test_scalar_checkpoint_is_atomic_resumable_and_identity_strict(tmp_path):
    path = tmp_path / "real_eta.npz"
    identity = {"schema": 1, "grid_n": 9, "grid_sha256": "abc"}
    rows = (np.arange(3) + 1j * np.arange(3), np.zeros(3, np.complex128))
    _save_checkpoint(path, identity, 8, rows, 4.5, 1.25)
    completed, got, chi_s, contract_s = _load_checkpoint(
        path, identity, n_targets=3, n_accumulators=2)
    assert completed == 8
    np.testing.assert_array_equal(got, rows)
    assert chi_s == 4.5 and contract_s == 1.25
    assert not path.with_suffix(".npz.tmp").exists()
    with pytest.raises(ValueError, match="stale identity"):
        _load_checkpoint(
            path, {**identity, "grid_sha256": "changed"},
            n_targets=3, n_accumulators=2)


def test_weighted_contract_matches_explicit_oracle_on_square_mesh():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw.internal_ff_cd import make_weighted_contract_kernel

    side = 2 if len(jax.devices()) >= 4 else 1
    mesh = Mesh(np.asarray(jax.devices()[:side * side]).reshape(side, side),
                ("x", "y"))
    rng = np.random.default_rng(19)
    nk, ns, nmu, nb = 3, 1, 4, 2
    px = rng.normal(size=(nk, ns, nmu, nb)) \
        + 1j * rng.normal(size=(nk, ns, nmu, nb))
    py = rng.normal(size=(nk, ns, nmu, nb)) \
        + 1j * rng.normal(size=(nk, ns, nmu, nb))
    wc = rng.normal(size=(nk, nmu, nmu)) \
        + 1j * rng.normal(size=(nk, nmu, nmu))
    target_k = np.asarray([0, 2], np.int32)
    target_b = np.asarray([1, 0], np.int32)
    kmq = np.asarray([[0, 2, 1], [2, 1, 0]], np.int32)
    coeff = rng.normal(size=(2, nk, nb))

    def put(a, spec):
        return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))

    kernel = make_weighted_contract_kernel(
        mesh, n_targets=2, inner_stop=nb, tile=2)
    got = np.asarray(kernel(
        put(px, P(None, None, "x", None)),
        put(py, P(None, None, "y", None)),
        put(wc, P(None, "x", "y")),
        put(target_k, P(None)), put(target_b, P(None)),
        put(kmq, P(None, None)), put(coeff, P(None, None, None))))

    want = np.zeros(2, np.complex128)
    for t in range(2):
        for q in range(nk):
            for n in range(nb):
                dx = np.einsum(
                    "su,su->u", np.conj(px[target_k[t], :, :, target_b[t]]),
                    px[kmq[t, q], :, :, n])
                dy = np.einsum(
                    "su,su->u", np.conj(py[target_k[t], :, :, target_b[t]]),
                    py[kmq[t, q], :, :, n])
                want[t] += coeff[t, q, n] * dx @ wc[q] @ np.conj(dy)
    np.testing.assert_allclose(got, want, rtol=2e-12, atol=2e-12)
