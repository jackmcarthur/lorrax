"""The one-shot static head carries the deck's Hubbard input (CPU, seconds).

``HeadResolver`` hands its params to ``_check_dipole_provenance``, which
resolves the DFT+U stamp the dipole file must carry. Without the deck's
``hubbard_input`` / ``hubbard_occupations`` in those params, a WFN whose QE
schema declares DFT+U refuses with ``GATE dftu_velocity_input`` before
map 0. A dipole file without i[r, V_U] (stamp ``'none'`` or absent) refuses
by name as a velocity-operator mismatch, under ``LORRAX_SANITY=0`` too. The
schema, the card parse, the dipole file and the dipole producer module (whose
import starts the runtime) are stubbed; the refusal rule and the params
hand-off are the production ones.
"""
import sys
from types import SimpleNamespace

import pytest

from ffi import _services

_services.ensure_on_path()

import file_io.restart_bundle as restart_bundle  # noqa: E402
import psp.hubbard_ops as hubbard_ops  # noqa: E402
from gw import head_correction  # noqa: E402

_CARD = "/deck/qe/nscf.in"
_OCC = "/deck/qe/NiPS3.save/occup.txt"
_STAMP = '{"scheme": "lorrax.dftu_velocity/v1", "test": true}'
_REAL_MISMATCHES = restart_bundle.dipole_operator_mismatches


def _config(hubbard_input, hubbard_occupations):
    head = SimpleNamespace(
        wcoul0_source="s_tensor", wcoul0_eta=0.0, vhead=None,
        whead_0freq=None, whead_imfreq=None, head_minibz_average=False,
        bgw_metal_q0_treatment=None, correction="no_local_fields")
    return SimpleNamespace(
        head=head, bispinor=False, bispinor_gw="bare_transverse",
        nval=4, ncond=4, nband=8, vnl_velocity_sign="+1", do_screened=True,
        hubbard_input=hubbard_input, hubbard_occupations=hubbard_occupations)


@pytest.fixture
def dftu_schema(monkeypatch):
    """A WFN whose QE schema declares DFT+U; the dipole file is stubbed."""
    seen = {}
    monkeypatch.setattr(hubbard_ops, "qe_dftu_declaration", lambda wfn: {
        "schema_path": "data-file-schema.xml", "projector": "ortho-atomic",
        "kind": 0, "U_eV": {("Ni1", "3d"): 4.0}})
    real_resolve = hubbard_ops.resolve_hubbard_input

    def resolve(hubbard_input, hubbard_occupations, **kw):
        if not (hubbard_input or hubbard_occupations):
            return real_resolve(hubbard_input, hubbard_occupations, **kw)
        seen["keys"] = (hubbard_input, hubbard_occupations)
        return SimpleNamespace(provenance=_STAMP)

    monkeypatch.setattr(hubbard_ops, "resolve_hubbard_input", resolve)
    monkeypatch.setitem(sys.modules, "psp.get_dipole_mtxels", SimpleNamespace(
        resolve_vnl_velocity_sign=lambda cli, deck: int(deck),
        _prov_ne=lambda got, want: str(got) != want if isinstance(want, str)
        else int(got) != int(want)))

    def mismatches(path, *, hubbard=None, **kw):
        seen["hubbard"] = hubbard
        return []

    monkeypatch.setattr(restart_bundle, "dipole_operator_mismatches", mismatches)
    monkeypatch.setattr(restart_bundle, "check_dipole_provenance", lambda path, **kw: True)
    monkeypatch.delenv("LORRAX_SANITY", raising=False)
    return seen


def _check(config):
    resolver = head_correction.HeadResolver(
        config, "/deck", wfn=SimpleNamespace(), sym=None, meta=None,
        print_fn=lambda *a, **k: None)
    head_correction._check_dipole_provenance(
        "/deck/dipole.h5", params=resolver._params, wfn=resolver.wfn,
        print_fn=lambda *a, **k: None)


def test_head_resolver_passes_deck_hubbard_input(dftu_schema):
    _check(_config(_CARD, _OCC))
    assert dftu_schema["keys"] == (_CARD, _OCC)
    assert dftu_schema["hubbard"] == _STAMP


def test_dftu_schema_without_hubbard_keys_still_refuses(dftu_schema):
    with pytest.raises(ValueError, match="GATE dftu_velocity_input"):
        _check(_config("", ""))


@pytest.mark.parametrize("stamp", ["none", None])
def test_dipole_without_hubbard_stamp_refuses_with_sanity_off(dftu_schema, monkeypatch, stamp):
    attrs = {"prov_skip_vnl": 0, "prov_vnl_mode": "analytic", "prov_vnl_velocity_sign": 1}
    if stamp is not None:
        attrs["prov_hubbard"] = stamp

    class _Dipole:
        def __init__(self, path, mode):
            self.attrs = attrs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(restart_bundle, "h5py", SimpleNamespace(File=_Dipole))
    monkeypatch.setattr(restart_bundle, "dipole_operator_mismatches", _REAL_MISMATCHES)
    monkeypatch.setenv("LORRAX_SANITY", "0")
    with pytest.raises(ValueError, match="(?s)GATE static_head_dipole_operator.*prov_hubbard"):
        _check(_config(_CARD, _OCC))
