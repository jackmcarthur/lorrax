"""write_poles exports an ordered model with the representation it declares.

The export writer omitted ``ordered=``, so ``_metadata`` re-derived it from the measured TRS
state and refused every ordered deck with "TRS-broken representation is unsupported" -- after the
whole construction had been paid for. The call site now passes the source model's own flag.
"""
import inspect
from types import SimpleNamespace

import pytest

from file_io import shared_pole_store as store


def test_the_model_export_passes_the_source_representation_to_the_writer():
    source = inspect.getsource(store.export_shared_pole_outputs)
    assert 'ordered=bool(model.get("ordered", False))' in source, \
        "the model export must pass the source store's representation to the writer"


@pytest.mark.parametrize("ordered,trs_allowed,refused", [
    (True, False, False),   # ordered model on a measured-broken-TRS deck: the case that was refused
    (False, True, False),   # the TRS path, unchanged
    (True, True, True),     # an ordered store against allowed TRS stays refused
    (False, False, True),   # and so does a TRS store on a broken-TRS deck
])
def test_metadata_refuses_only_when_the_flag_contradicts_the_measured_state(ordered, trs_allowed, refused):
    """_metadata owns the agreement; this pins which combinations it admits.

    The refusal fires before any table work, so a bare sym stub reaches it; the admitted
    combinations fail later on the stub instead, never on the representation.
    """
    meta = SimpleNamespace(nspinor=1, mu_basis=SimpleNamespace(n_logical=4, n_packed=4, is_identity=True))
    tables = {"sym": SimpleNamespace(trs_allowed=trs_allowed)}
    identity = {key: "planted-" + key for key in store._IDENTITY_KEYS}
    if refused:
        with pytest.raises(ValueError, match="representation"):
            store._metadata(meta, tables, {"version": "v"}, identity, ordered=ordered)
    else:
        with pytest.raises(Exception) as excinfo:
            store._metadata(meta, tables, {"version": "v"}, identity, ordered=ordered)
        assert "representation is unsupported" not in str(excinfo.value)
