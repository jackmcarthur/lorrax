"""Residual-probe rank admission uses the shared owners, not inverse policy."""
import jax
import numpy as np
import pytest

from common import rank_criterion
from gw.transition_galerkin import _canonical_qr_rotation

jax.config.update('jax_enable_x64', True)


def test_original_seed_scale_and_shared_probe_owner(monkeypatch):
    calls=[]
    original=rank_criterion.probe_is_independent

    def observe(after, before, scale=0., *, rtol):
        calls.append((float(after), float(before), float(scale), rtol))
        return original(after, before, scale, rtol=rtol)

    monkeypatch.setattr(rank_criterion, 'probe_is_independent', observe)
    rotation=_canonical_qr_rotation(np.diag([.1, .001]), .01, 2.)
    assert rotation.shape == (2, 1)
    assert len(calls) == 2
    assert all(before == 2. and scale == .1 for _, before, scale, _ in calls)
    np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(1))


def test_near_degenerate_residual_block_is_not_split():
    with pytest.raises(ValueError, match='transition_qr'):
        _canonical_qr_rotation(np.diag([1., 1.-5e-7]), 1.-2.5e-7, 1.)


def test_dependent_residual_returns_no_arbitrary_qr_completion():
    rotation=_canonical_qr_rotation(np.zeros((2, 2), complex), 1e-8, 3.)
    assert rotation.shape == (2, 0)
