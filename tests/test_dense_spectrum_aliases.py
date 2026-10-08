"""Original reconstruction identities survive exact-byte relocation only."""
import hashlib
from pathlib import Path

import pytest

from file_io.dense_spectrum import verify_source_bindings


def fixture(tmp_path):
    old = str(tmp_path/'original.dat')
    actual = tmp_path/'copy.dat'
    actual.write_bytes(b'original immutable reconstruction bytes')
    bindings = {old: hashlib.sha256(actual.read_bytes()).hexdigest()}
    return old, actual, bindings


def test_exact_relocation_preserves_original_identity(tmp_path):
    old, actual, bindings = fixture(tmp_path)
    original = dict(bindings)
    assert verify_source_bindings(bindings, source_aliases={old:str(actual)}) == {old:str(actual)}
    assert bindings == original


def test_default_original_path_is_still_authenticated(tmp_path):
    old, actual, bindings = fixture(tmp_path)
    Path(old).write_bytes(actual.read_bytes())
    assert verify_source_bindings(bindings) == {old:old}


@pytest.mark.parametrize('kind', ['missing', 'unknown', 'relative', 'changed'])
def test_alias_failure_guards(tmp_path, kind):
    old, actual, bindings = fixture(tmp_path)
    aliases={old:str(actual)}
    if kind=='missing': aliases={}
    if kind=='unknown': aliases['unknown']=str(actual)
    if kind=='relative': aliases[old]='relative.dat'
    if kind=='changed': actual.write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='dense_spectrum_reference'):
        verify_source_bindings(bindings, source_aliases=aliases)
