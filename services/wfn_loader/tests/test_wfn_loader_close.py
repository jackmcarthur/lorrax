"""Deterministic close reports failures; only destructor cleanup suppresses."""
from unittest.mock import Mock

import pytest

from wfn_loader import WfnLoader


@pytest.mark.parametrize('method', ['close', '__exit__'])
def test_explicit_close_propagates_and_releases_both_handles(method):
    loader = object.__new__(WfnLoader)
    error = OSError('injected h5py close failure')
    first, second = Mock(), Mock()
    first.close.side_effect = error
    loader._file, loader._slab_io = first, second
    with pytest.raises(OSError) as raised:
        getattr(loader, method)()
    assert raised.value is error
    first.close.assert_called_once()
    second.close.assert_called_once()
    loader.close()
    second.close.assert_called_once()


def test_destructor_suppresses_with_diagnostic(capsys):
    loader = object.__new__(WfnLoader)
    loader._slab_io = Mock()
    loader._slab_io.close.side_effect = RuntimeError('queued write failed')
    loader.__del__()
    assert 'RuntimeError: queued write failed' in capsys.readouterr().err
    assert loader._slab_io is None
