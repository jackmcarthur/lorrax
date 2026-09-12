"""Resource selection at the real MPA/shared-pole dispatch seam."""
import pytest

from gw.sigma_dispatch import _mpa_sigma_model_resources


def test_incumbent_uses_same_body_and_head():
    assert _mpa_sigma_model_resources({'mpa_fit': 'incumbent.h5'}, 'mpa') == (
        'incumbent.h5', 'incumbent.h5', None, None)


def test_shared_dispatch_keeps_authenticated_body_with_explicit_head_off():
    identity = {'iteration_id': 'current'}
    roles = {'mpa_fit': 'stale.h5', 'shared_pole': dict(
        path='current.h5',
        identity=identity, digest='payload', K=[2, 0])}
    assert _mpa_sigma_model_resources(roles, 'shared_pole', 'off') == (
        'current.h5', None, identity, 'payload')


@pytest.mark.parametrize('missing', ['path', 'identity', 'digest'])
def test_incomplete_handle_never_falls_back_to_incumbent(missing):
    handle = dict(path='current.h5', head_fit_path='head.h5', identity={}, digest='d')
    del handle[missing]
    with pytest.raises(KeyError, match=missing):
        _mpa_sigma_model_resources({'shared_pole': handle, 'mpa_fit': 'old.h5'}, 'shared_pole', 'off')


@pytest.mark.parametrize('policy', [None, 'full', 'no_local_fields'])
def test_shared_head_enabled_requires_current_fit(policy):
    handle = dict(path='current.h5', identity={}, digest='d')
    with pytest.raises(ValueError, match='current body-bound MPA scalar head'):
        _mpa_sigma_model_resources({'shared_pole': handle}, 'shared_pole', policy)


def test_shared_head_is_bound_to_current_body():
    handle = dict(path='current.h5', identity={'iteration_id': 'sc_0002'}, digest='d')
    head = dict(identity=handle['identity'], body_digest='d', Omega_p=[1j], B_p=[2])
    roles = dict(shared_pole=handle, mpa_head=head)
    assert _mpa_sigma_model_resources(roles, 'shared_pole', 'full')[1] is head
    roles['shared_pole'] = dict(handle, digest='new_W')
    with pytest.raises(ValueError, match='missing or stale'):
        _mpa_sigma_model_resources(roles, 'shared_pole', 'full')
