"""Production contract for ``bgw_metal_q0_treatment``.

The fixtures are deliberately small.  They pin the deck semantics and the
two scalar reductions against the R5 sodium consumables without requiring a
48-band WFN or an N_mu-sized plane-wave dielectric matrix.
"""
from __future__ import annotations

import dataclasses
import pickle
import sys
import types
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ffi import _services

_services.ensure_on_path()
from vcoul import minibz_average  # noqa: E402

from gw.gw_config import LorraxConfig  # noqa: E402
from gw.head_correction import (  # noqa: E402
    bgw_q0shift_head_sample,
    finite_q0_epsinv_head,
    resolve_bgw_q0_channel,
)


BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 4
sys_dim = 3
memory_per_device_gb = 4.0
"""


def _config(tmp_path, extra="", *, messages=None, sys_dim=3):
    path = tmp_path / "q0.in"
    path.write_text(
        BASE.replace("sys_dim = 3", f"sys_dim = {int(sys_dim)}") + extra)
    sink = messages if messages is not None else []
    # Config construction needs only these two file_io symbols, but importing
    # the package door also imports h5py.  Keep this pure config unit runnable
    # in the minimal ``lx test`` environment, where h5py is intentionally not
    # installed; production tests exercise the real path.
    file_io = types.ModuleType("file_io")
    file_io.__path__ = []
    file_io.resolve_input_paths = lambda params, input_dir: None
    kin_ion = types.ModuleType("file_io.kin_ion")
    kin_ion.HARTREE_SOURCES = ("auto", "stored", "isdf", "gspace")
    with mock.patch.dict(
            sys.modules, {"file_io": file_io, "file_io.kin_ion": kin_ion}):
        return LorraxConfig.from_input_file(
            str(path),
            print_fn=lambda *a, **k: sink.append(" ".join(map(str, a))))


def _one_device_mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _plain_head_number(config):
    # A fixed, nonsingular mini-BZ fixture.  Both the absent key and explicit
    # exact mode must take the historical plain-MC arm and return the same
    # bytes, not merely agree within a tolerance.
    dq = [np.asarray([[0.31, 0.12, -0.08],
                      [-0.14, 0.27, 0.19],
                      [0.22, -0.25, 0.11]], dtype=np.float64)]
    return np.float64(minibz_average(
        np.zeros(3), dq, kind="bulk_3d", celvol=1.0, n_kpts=8,
        q0sph2=0.01, analytic_sphere=config.head.analytic_q0_sphere,
        adaptive=False))


def test_absent_and_explicit_exact_are_byte_identical(tmp_path):
    absent = _config(tmp_path)
    absent_bytes = pickle.dumps(absent, protocol=5)
    absent_head = _plain_head_number(absent).tobytes()

    exact = _config(tmp_path, "bgw_metal_q0_treatment = exact\n")
    exact_bytes = pickle.dumps(exact, protocol=5)
    exact_head = _plain_head_number(exact).tobytes()

    assert dataclasses.asdict(absent) == dataclasses.asdict(exact)
    assert absent_bytes == exact_bytes
    assert absent_head == exact_head


def test_bgw_q0shift_overrides_inherited_body_average_and_announces(tmp_path):
    messages = []
    cfg = _config(
        tmp_path,
        "bgw_metal_q0_treatment = bgw_q0shift\ncompute_mode = mpa\n",
        messages=messages)
    assert cfg.head.mc_average_vcoul_body is False
    assert cfg.head.analytic_q0_sphere is True
    report = "\n".join(messages)
    assert "overriding mc_average_vcoul_body -> false" in report
    assert "eta/broadening and MPA quadrature are unchanged" in report


def test_explicit_contradictory_body_average_refuses_by_both_names(tmp_path):
    with pytest.raises(ValueError) as error:
        _config(
            tmp_path,
            "bgw_metal_q0_treatment = bgw_q0shift\n"
            "mc_average_vcoul_body = true\n")
    message = str(error.value)
    assert "bgw_metal_q0_treatment" in message
    assert "mc_average_vcoul_body" in message


def test_bgw_q0shift_refuses_outside_three_dimensions(tmp_path):
    with pytest.raises(ValueError, match=(
            "bgw_metal_q0_treatment.*bgw_q0shift.*sys_dim = 3")):
        _config(
            tmp_path,
            "bgw_metal_q0_treatment = bgw_q0shift\n",
            sys_dim=2)


def test_bgw_analytic_sphere_reduced_fixture_matches_r5_vhead():
    # R5 consumable values (Na 8x8x8): normalized v_head and cell volume.
    target = 9.746958155858e-2
    nk = 512
    cell_volume = 254.476249
    raw_target = target * nk * cell_volume

    # The Na reciprocal primitive has nearest-vector length 1.11293325.
    # Its largest inscribed mini-BZ sphere therefore has R=|b_min|/(2*8).
    radius = 1.11293325 / 16.0
    r2 = radius * radius
    analytic = 4.0 * radius * cell_volume * nk / np.pi
    outside_mean = raw_target - analytic
    assert outside_mean > 0.0

    # A one-value reduced outer-MC consumable: 8*pi/r^2 equals BGW's stored
    # outside-sphere mean.  The production estimator uses many current-seed
    # draws; this fixture isolates the exact split and full-count denominator.
    outer_radius = np.sqrt(8.0 * np.pi / outside_mean)
    assert outer_radius > radius
    dq = [np.asarray([[outer_radius, 0.0, 0.0]], dtype=np.float64)]
    got_raw = minibz_average(
        np.zeros(3), dq, kind="bulk_3d", celvol=cell_volume, n_kpts=nk,
        q0sph2=r2, analytic_sphere=True, adaptive=False)
    got = got_raw / (nk * cell_volume)
    assert got == pytest.approx(target, rel=0.0, abs=2.0e-14)


def test_shifted_q0_binds_to_the_stored_wedge_representative():
    kvecs = np.asarray([
        (x, y, z) for x in range(2) for y in range(2) for z in range(2)
    ], dtype=np.int32)
    sym = SimpleNamespace(
        kvecs_asints=kvecs,
        irr_idx_q=np.arange(8, dtype=np.int32),
    )
    head = SimpleNamespace(
        uses_bgw_metal_q0shift=True,
        bgw_metal_q0_vector=(0.0, 0.0, 0.5),
    )
    config = SimpleNamespace(head=head)
    channel = SimpleNamespace(
        g_head=np.arange(16, dtype=np.float64).reshape(8, 1, 2),
        v_bare=np.arange(8, dtype=np.float64) + 10.0,
        mult=np.ones(8, dtype=np.int32),
    )
    got = resolve_bgw_q0_channel(
        config, sym, np.arange(8, dtype=np.int32), channel,
        kgrid=(2, 2, 2))
    assert got.requested_full_index == 1
    assert got.representative_full_index == 1
    assert got.wedge_row == 1
    np.testing.assert_array_equal(got.g_head, channel.g_head[1, 0:1])


def test_finite_q0_head_wings_reduction_matches_r5_epsinv():
    target = 3.009041950876e-2
    v_q0 = 1298.618549
    chi_eff = (1.0 - 1.0 / target) / v_q0
    chi = np.asarray([[[chi_eff]]], dtype=np.complex128)
    V = np.asarray([[[v_q0]]], dtype=np.complex128)
    W = np.linalg.solve(np.eye(1)[None] - V @ chi, V)
    g = np.ones((1, 1), dtype=np.complex128)

    mesh = _one_device_mesh()
    matrix_sharding = NamedSharding(mesh, P(None, "x", "y"))
    vector_sharding = NamedSharding(mesh, P(None, "x"))
    got = finite_q0_epsinv_head(
        jax.device_put(jnp.asarray(chi), matrix_sharding),
        jax.device_put(jnp.asarray(W), matrix_sharding),
        jax.device_put(jnp.asarray(g), vector_sharding),
        v_q0,
        1.0,
        mesh_xy=mesh,
    )
    assert complex(np.asarray(got)[0]).real == pytest.approx(
        target, rel=0.0, abs=2.0e-14)

    sample = bgw_q0shift_head_sample(
        9.746958155858e-2, complex(np.asarray(got)[0]), 0.0)
    assert (sample.wcoul0 / sample.vc0).real == pytest.approx(
        target, rel=0.0, abs=2.0e-14)
