"""The dense-H preflight (``psp.operator_checks.validate_dense_h_inputs``).

One cell per refusal class, each built from an input that passes and then
broken in exactly the field the class reads, so a cell that stays green while
its field is ignored cannot pass the positive control.  No files beyond one
empty UPF stand-in in ``tmp_path``; no device.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from psp.operator_checks import (DenseHRefusal, dense_h_bytes,
                                 validate_dense_h_inputs)


def _crystal(**over):
    base = SimpleNamespace(
        functional="PBE", xc_extensions=(), domag=False, nspin=1,
        assume_isolated="none", pseudo_files=(("Si", "Si.upf"),),
        atom_types=np.array([14, 14]), _save_dir="/nowhere/silicon.save")
    for key, value in over.items():
        setattr(base, key, value)
    return base


def _pseudos(**header):
    fields = dict(element="Si", pseudo_type="NC", is_ultrasoft=False,
                  is_paw=False, functional="PBE")
    fields.update(header)
    return {"Si": SimpleNamespace(pp_header=SimpleNamespace(**fields))}


def _check(tmp_path, crystal=None, pseudos=None, *, sys_dim=3,
           fields=("MillerIndices", "rhotot_g"), n_basis=1000,
           budget=1e12, upf=True):
    if upf:
        (tmp_path / "Si.upf").write_text("")
    return validate_dense_h_inputs(
        crystal or _crystal(), pseudos or _pseudos(), sys_dim=sys_dim,
        pseudo_dir=str(tmp_path), charge_density_fields=fields,
        n_basis_max=n_basis, budget_bytes=budget)


def _refuses(rule, **kw):
    with pytest.raises(DenseHRefusal) as err:
        _check(**kw)
    assert err.value.rule == rule, str(err.value)
    return str(err.value)


def test_positive_control_passes(tmp_path):
    ctx = _check(tmp_path)
    assert ctx.truncation_2d is False and ctx.sys_dim == 3


def test_missing_upf_refuses(tmp_path):
    msg = _refuses("upf_missing", tmp_path=tmp_path, upf=False)
    assert "Si.upf" in msg


@pytest.mark.parametrize("functional", ["PBESOL", "SCAN", "HSE", "B3LYP", "PZ"])
def test_functional_other_than_pbe_refuses(tmp_path, functional):
    msg = _refuses("functional", tmp_path=tmp_path,
                   crystal=_crystal(functional=functional))
    assert functional in msg


@pytest.mark.parametrize("ext", ["hybrid", "vdW", "dftU"])
def test_hybrid_vdw_plus_u_refuse(tmp_path, ext):
    msg = _refuses("xc_extension", tmp_path=tmp_path,
                   crystal=_crystal(xc_extensions=(ext,)))
    assert ext in msg


def test_upf_functional_mismatch_refuses(tmp_path):
    _refuses("functional", tmp_path=tmp_path,
             pseudos=_pseudos(functional="PZ"))


@pytest.mark.parametrize("header", [dict(pseudo_type="US"),
                                    dict(pseudo_type="PAW"),
                                    dict(is_ultrasoft=True),
                                    dict(is_paw="T")])
def test_ultrasoft_and_paw_refuse(tmp_path, header):
    _refuses("pseudo_type", tmp_path=tmp_path, pseudos=_pseudos(**header))


@pytest.mark.parametrize("over", [dict(nspin=2), dict(domag=True)])
def test_magnetic_run_refuses(tmp_path, over):
    _refuses("magnetism", tmp_path=tmp_path, crystal=_crystal(**over))


@pytest.mark.parametrize("fields", [(), ("MillerIndices",)])
def test_missing_charge_density_refuses(tmp_path, fields):
    _refuses("charge_density", tmp_path=tmp_path, fields=fields)


@pytest.mark.parametrize("sys_dim,isolated", [(2, "none"), (3, "2D")])
def test_truncation_mismatch_refuses(tmp_path, sys_dim, isolated):
    _refuses("truncation_2d", tmp_path=tmp_path, sys_dim=sys_dim,
             crystal=_crystal(assume_isolated=isolated))


def test_slab_with_matching_truncation_passes(tmp_path):
    ctx = _check(tmp_path, _crystal(assume_isolated="2D"), sys_dim=2)
    assert ctx.truncation_2d is True


def test_memory_budget_refuses(tmp_path):
    n = 4000
    _check(tmp_path, n_basis=n, budget=dense_h_bytes(n))          # fits exactly
    msg = _refuses("memory", tmp_path=tmp_path, n_basis=n,
                   budget=dense_h_bytes(n) - 1)
    assert "memory_per_device_gb" in msg
