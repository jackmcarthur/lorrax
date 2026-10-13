"""Retained local banks and coupled contraction are priced before the Z store."""
import dataclasses
import math
from types import SimpleNamespace

import pytest


def case(monkeypatch, budget=66.36301091856679):
    from common import gpu_utils
    from gw import gflat_memory_model as model
    monkeypatch.setattr(gpu_utils, 'host_bytes_per_process', lambda _: 1000e9)
    meta = SimpleNamespace(nk_tot=9, nspinor=4, n_rmu=4262,
                           n_rmu_padded=4272, fft_grid=(80, 80, 250))
    return model, dict(meta=meta, mesh_xy=SimpleNamespace(shape={'x': 4, 'y': 4}),
                      n_q_selected=9, ngkmax=612333, psi_ngkmax=76533,
                      fit_nb=200, n_col=4000, n_s=120, budget_gb=budget,
                      n_parent=3, orbit_width=12, n_vertex=3)


def account(callback, extra=0., geometry=(8, 64, 256)):
    return dict(current_local_geometry=geometry,
                current_local_resident_bytes=extra,
                current_local_workspace=callback)


def test_ordinary_defaults_preserve_the_plan_and_tile(monkeypatch):
    model, args = case(monkeypatch)
    first = model.plan_zeta_route_g(**args)
    explicit = model.plan_zeta_route_g(**args, current_local_geometry=None,
                                     current_local_resident_bytes=0.,
                                     current_local_workspace=None)
    for field in dataclasses.fields(first):
        if field.name not in ('working_set', 'p_chunk'):
            assert getattr(first, field.name) == getattr(explicit, field.name)
    assert first.format() == explicit.format()
    assert first.working_set(first.b, first.r_sub, first.c_out, first.n_blk) == \
        explicit.working_set(explicit.b, explicit.r_sub, explicit.c_out, explicit.n_blk)
    per_g = 6 * 16 * 4272
    expected = min(math.ceil(612333 / 16) * 16,
                   max(16, int((.25 * first.target_bytes // per_g) // 16 * 16)))
    assert first.g_tile == expected


def test_coupled_nonlinear_owner_selects_largest_safe_tile_before_store(monkeypatch):
    model, args = case(monkeypatch)
    calls = []
    def quote(tile):
        calls.append(tile)
        fixed = 35e9
        scratch = 1e6 * tile + 100. * tile * tile
        return dict(solved_banks_and_local=fixed, coupled_scratch=scratch,
                    total=fixed + scratch)
    plan = model.plan_zeta_route_g(**args, **account(quote, extra=1e9))
    assert plan.g_tile % 16 == 0 and plan.g_tile < 40448
    assert len(calls) < 20 and all(tile % 16 == 0 for tile in calls)
    ordinary = model.plan_zeta_route_g(**args)
    shared = sum(ordinary.breakdown[key] for key in
                 ('C factor', 'centroid faces', 'sphere tables',
                  'conj ψ(G) slice (resident)')) + 1e9
    assert shared + quote(plan.g_tile)['total'] <= plan.target_bytes
    assert shared + quote(plan.g_tile + 16)['total'] > plan.target_bytes
    assert plan.hwm_bytes <= plan.target_bytes
    expected_store = 3 * 16 * 9 / 16 * (plan.n_batch * plan.b) * \
        (math.ceil(612333 / plan.g_tile) * plan.g_tile)
    assert plan.store_bytes == expected_store
    # Deferred radial packets do not allocate full banks before OWN3. Their
    # solved outputs belong only to the callback's later contraction phase.
    assert plan.min_config_bytes == ordinary.min_config_bytes + 1e9
    assert 'raw local current RHS banks' not in plan.breakdown


def test_actual_retained_extra_can_refuse_before_any_quote_or_store(monkeypatch):
    model, args = case(monkeypatch, budget=10.)
    called = []
    with pytest.raises(ValueError, match='minimum PW-fit residency'):
        model.plan_zeta_route_g(**args, **account(lambda tile: called.append(tile), extra=11e9))
    assert called == []


def test_minimum_contraction_can_refuse_before_store(monkeypatch):
    model, args = case(monkeypatch)
    with pytest.raises(ValueError, match='minimum contraction residency'):
        model.plan_zeta_route_g(**args, **account(lambda tile: dict(fixed=70e9, total=70e9)))


@pytest.mark.parametrize('kind', ['negative', 'nan', 'inconsistent', 'decreasing'])
def test_invalid_pricing_owner_refuses(monkeypatch, kind):
    model, args = case(monkeypatch)
    def quote(tile):
        val = {'negative': -1., 'nan': math.nan,
               'inconsistent': 1., 'decreasing': 1e9 - 100. * tile}[kind]
        return dict(bank=val, total=val + (1. if kind == 'inconsistent' else 0.))
    with pytest.raises(ValueError, match='invalid live-byte|not monotone'):
        model.plan_zeta_route_g(**args, **account(quote))


@pytest.mark.parametrize('mutation', ['no_geometry', 'no_owner', 'wrong_channels',
                                    'negative_extra', 'nonfinite_extra', 'bad_geometry'])
def test_internal_current_contract_requires_actual_complete_inputs(monkeypatch, mutation):
    model, args = case(monkeypatch)
    kwargs = account(lambda tile: dict(fixed=1., total=1.))
    if mutation == 'no_geometry': kwargs['current_local_geometry'] = None
    elif mutation == 'no_owner': kwargs['current_local_workspace'] = None
    elif mutation == 'wrong_channels': args['n_vertex'] = 1
    elif mutation == 'negative_extra': kwargs['current_local_resident_bytes'] = -1.
    elif mutation == 'nonfinite_extra': kwargs['current_local_resident_bytes'] = math.inf
    else: kwargs['current_local_geometry'] = (8, 64, 0)
    with pytest.raises(ValueError):
        model.plan_zeta_route_g(**args, **kwargs)
