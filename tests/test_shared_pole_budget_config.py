"""Fixed body rank changes must leave the independently bound problem fixed."""
import configparser
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from gw.gw_config import (DynamicSigmaConfig, _DEFAULTS, _DECK_NAMED_KEYS,
                          _parse_input_keys, _resolve_shared_pole_inputs)
from gw.shared_pole_recipe import (RECIPE_HASH, RECIPE_VERSION,
    bind_shared_pole_census, parse_pole_budget, resolve_shared_pole_recipe)


def sigma(**overrides):
    values = dict(omega_min_ev=None, omega_max_ev=None, omega_step_ev=.5,
        regularization_ev=.25, window_edge_factor=2., fermi_reference="midgap",
        sigma_at_dft_energies=True, w_model="shared_pole")
    values.update(overrides)
    return DynamicSigmaConfig(**values)


def problem():
    # The independent physical table is shared across caps. No response or
    # coefficient arrays are manufactured here, and no device allocation is needed.
    wfns = SimpleNamespace(enk=np.asarray([[-1.,-.2,.5,1.5],[-.9,-.1,.6,1.6]]),
        occ=np.asarray([[1.,1.,0.,0.],[1.,1.,0.,0.]]), valid_kn=None,
        slices=SimpleNamespace(b0=0,b4_logical=4,val=slice(0,2),
                               cond=slice(2,4),cond_all_logical=slice(2,4)))
    meta = SimpleNamespace(nk_tot=2,nspin=1,nspinor=1,nspinor_wfnfile=1,
                           n_rmu=8,cell_volume=100.,b_id_4_chi_user=4)
    bind_shared_pole_census(wfns,meta,occupation_state=None,trs_allowed=True,
                           state_capacity=2.,kweights=[.5,.5])
    return wfns,meta


def resolve(monkeypatch, config=None, *, session=None):
    import common.gpu_utils as gpu
    monkeypatch.setattr(gpu,"device_budget_bytes",lambda:8_000_000_000)
    w,m = problem()
    return resolve_shared_pole_recipe(
        SimpleNamespace(sigma=sigma() if config is None else config,bispinor=False),
        w,m,mesh_xy=SimpleNamespace(shape={"x":2,"y":2}),print_fn=lambda _:None,
        support_reads_ev=[-12.,-4.,2.,12.,25.],support_session=session)


def assert_same(a,b):
    assert set(a)==set(b)
    for key in a:
        if isinstance(a[key],np.ndarray):
            np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        else:
            assert a[key]==b[key],key


@pytest.mark.parametrize("value",[True,False,np.bool_(True),1.5,4.,"4",None,np.nan,np.inf,-1])
def test_invalid_caps_never_truncate_or_select_an_accidental_rank(value):
    with pytest.raises(ValueError,match="shared_pole_budget"):
        parse_pole_budget(value)
    with pytest.raises(ValueError,match="shared_pole_budget"):
        sigma(w_pole_budget=value)


def test_zero_default_is_the_legacy_recipe_and_identity(monkeypatch):
    # An old caller with no field is supported and has exactly the same result,
    # not merely an equal budget or an equal sampling interval.
    old=vars(sigma()).copy();old.pop("w_pole_budget")
    legacy=resolve(monkeypatch,SimpleNamespace(**old))
    explicit_zero=resolve(monkeypatch,sigma(w_pole_budget=0))
    assert_same(legacy,explicit_zero)
    assert explicit_zero["recipe_version"]==RECIPE_VERSION
    assert explicit_zero["recipe_hash"]==RECIPE_HASH
    assert explicit_zero["pole_budget"]==15  # ceil(1.8*8), independent expectation.
    assert not any(k in explicit_zero for k in ("pole_budget_override","pole_budget_policy"))


@pytest.mark.parametrize("cap",[5,8,12,15,16])
def test_only_cap_and_its_authenticated_identity_change(monkeypatch,cap):
    base=resolve(monkeypatch)
    changed=resolve(monkeypatch,sigma(w_pole_budget=np.int64(cap)))
    assert changed["pole_budget"]==cap
    assert changed["automatic_pole_budget"]==15
    assert changed["pole_budget_override"]==cap
    assert changed["pole_budget_policy"]=="fixed-body-pole-budget-v1"
    assert changed["recipe_version"]==RECIPE_VERSION+"+pole_budget"
    expected=hashlib.sha256((RECIPE_HASH+"|fixed-body-pole-budget-v1|"+str(cap)).encode()).hexdigest()
    assert changed["recipe_hash"]==expected!=base["recipe_hash"]
    exempt={"pole_budget","recipe_hash","recipe_version","automatic_pole_budget",
            "pole_budget_override","pole_budget_policy"}
    assert_same({k:v for k,v in base.items() if k not in exempt},
                {k:v for k,v in changed.items() if k not in exempt})
    # In particular all physical sites/roles, f/energy census, infinity panel
    # width, direction widths and gate hash are already covered by that equality.


def test_explicit_support_and_cap_identities_compose_without_moving_a_site(monkeypatch):
    text="0,3,12,30 | 3,6,20"
    support=resolve(monkeypatch,sigma(w_support_sites_ev=text))
    fixed=resolve(monkeypatch,sigma(w_support_sites_ev=text,w_pole_budget=8))
    assert fixed["recipe_version"]==support["recipe_version"]+"+pole_budget"
    assert fixed["recipe_hash"]!=support["recipe_hash"]!=RECIPE_HASH
    for key in ("line_ev","imaginary_ev","held_line_ev","held_imaginary_ev","z_ry",
                "role","distinct_id","held","support_pair","fit_ids","held_ids"):
        np.testing.assert_array_equal(fixed[key],support[key])
    assert fixed["infinity_width"]==support["infinity_width"]
    assert fixed["gate_hash"]==support["gate_hash"]


def test_support_session_cannot_claim_a_hit_after_the_policy_changes(monkeypatch):
    session={}
    resolve(monkeypatch,session=session)  # reference map, does not retain.
    resolve(monkeypatch,session=session)
    key=session["key"]
    same=resolve(monkeypatch,session=session)
    assert same["support_envelope"]["status"]=="hit"
    fixed=resolve(monkeypatch,sigma(w_pole_budget=8),session=session)
    assert fixed["support_envelope"]["status"]=="policy_changed"
    assert session["key"]!=key
    np.testing.assert_array_equal(fixed["z_ry"],same["z_ry"])


def test_different_positive_caps_cannot_share_a_constructor_identity(monkeypatch):
    a=resolve(monkeypatch,sigma(w_pole_budget=5))
    b=resolve(monkeypatch,sigma(w_pole_budget=8))
    assert a["recipe_hash"]!=b["recipe_hash"]
    assert a["gate_hash"]==b["gate_hash"]


@pytest.mark.parametrize("model,accuracy",[("mpa","production"),("shared_pole","relaxed")])
def test_positive_cap_has_no_silent_unrelated_consumer(model,accuracy):
    with pytest.raises(ValueError,match="production shared_pole"):
        sigma(w_model=model,w_accuracy=accuracy,w_pole_budget=5)
    assert parse_pole_budget(0,model=model,accuracy=accuracy)==0


def parse(text):
    deck=configparser.ConfigParser();deck.read_string("[cohsex]\n"+text)
    params=_parse_input_keys(deck["cohsex"])
    _resolve_shared_pole_inputs(params)
    return params


def test_real_deck_parser_carries_the_declared_integer():
    params=parse("compute_mode=mpa\nsigma_w_model=shared_pole\nsigma_w_pole_budget=8\n")
    assert params["sigma_w_pole_budget"]==8
    assert "sigma_w_pole_budget" in params[_DECK_NAMED_KEYS]
    assert _DEFAULTS["sigma_w_pole_budget"]==0


@pytest.mark.parametrize("text",["true","2.5","4.0","nan","-1"])
def test_real_deck_parser_refuses_invalid_cap_spelling(text):
    with pytest.raises(ValueError):
        parse("compute_mode=mpa\nsigma_w_model=shared_pole\nsigma_w_pole_budget="+text+"\n")


@pytest.mark.parametrize("text",[
    "compute_mode=mpa\nsigma_w_model=mpa\nsigma_w_pole_budget=0\n",
    "compute_mode=x_only\nsigma_w_pole_budget=5\n",
    "compute_mode=mpa\nsigma_w_model=shared_pole\nsigma_w_accuracy=relaxed\nsigma_w_pole_budget=5\n",
    "compute_mode=mpa\nsigma_w_model=shared_pole\nbispinor=true\nsigma_w_pole_budget=5\n",
])
def test_named_unused_or_unsupported_policy_refuses_before_any_physics(text):
    with pytest.raises(ValueError,match="shared_pole"):
        parse(text)


def test_relaxed_zero_remains_unbudgeted(monkeypatch):
    result=resolve(monkeypatch,sigma(w_accuracy="relaxed",w_pole_budget=0))
    assert result["pole_budget"] is None
    assert result["recipe_hash"]==RECIPE_HASH
