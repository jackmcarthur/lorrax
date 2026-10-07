"""QE producer metadata, independent endpoint moments and refusal controls."""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from file_io import CrystalData
from psp.radial_tables import (_qe_simpsn_weights, qe_beta_radial_scheme,
    projector_reduced_origin, resolve_beta_simpson_rule)
from symmetry_maps import read_qe_symmetry_receipt


@pytest.mark.parametrize('version,rule', [('7.4','qe74'),('7.4.1','qe74'),('7.5','qe75'),('7.5.0','qe75')])
def test_authenticated_version_selects_even_endpoint(version,rule):
    mf=SimpleNamespace(qe_creator_name='PWSCF',qe_creator_version=version)
    sp=SimpleNamespace(kkbeta=160,n_proj=1,r=np.arange(161))
    assert resolve_beta_simpson_rule(mf,[sp])==rule


@pytest.mark.parametrize('name,version', [(None,None),('PWSCF',None),('PWSCF','7.6'),('CP','7.5'),('PWSCF','7.5-devel')])
def test_ambiguous_even_native_endpoint_refuses(name,version):
    mf=SimpleNamespace(qe_creator_name=name,qe_creator_version=version)
    with pytest.raises(ValueError,match='authenticated PWSCF'):
        resolve_beta_simpson_rule(mf,[SimpleNamespace(kkbeta=160,n_proj=1,r=np.arange(161))])


def test_odd_beta_mesh_has_one_rule_without_version_inference():
    assert resolve_beta_simpson_rule(SimpleNamespace(),[SimpleNamespace(kkbeta=161,n_proj=1,r=np.arange(161))])=='common_odd'
    np.testing.assert_array_equal(_qe_simpsn_weights(161,'qe74'),_qe_simpsn_weights(161,'qe75'))
    np.testing.assert_array_equal(_qe_simpsn_weights(161,'qe74'),_qe_simpsn_weights(161,'common_odd'))
    with pytest.raises(ValueError,match='even mesh'):_qe_simpsn_weights(160,'common_odd')


@pytest.mark.parametrize('rule,expected', [('qe74',[1/3,4/3,1/3,0]),('qe75',[1/3,15/12,1,5/12])])
def test_independent_four_point_endpoint_and_constant_integral(rule,expected):
    w=_qe_simpsn_weights(4,rule)
    np.testing.assert_allclose(w,expected,atol=1e-15,rtol=0)
    assert abs(w.sum()-(2 if rule=='qe74' else 3))<1e-15
    if rule=='qe75':
        np.testing.assert_allclose(w@np.arange(4.)**2,9.,atol=2e-15,rtol=0)


@pytest.mark.parametrize('rule', ['qe74','qe75'])
def test_cutoff_weights_and_reduced_origin_share_same_moment(rule):
    r=np.arange(1.,7.);rab=np.ones(6)
    sp=SimpleNamespace(r=r,rab=rab,kkbeta=4,proj_l=np.asarray([1]),beta_r=np.asarray([[1.,2.,3.,4.,100.,100.]]))
    n,w=qe_beta_radial_scheme(r,rab,4,rule=rule)
    assert n==4
    np.testing.assert_allclose(projector_reduced_origin(sp,0,rule=rule),np.dot(sp.beta_r[0,:4]*r[:4]**3,w)/3,atol=1e-13,rtol=0)


@pytest.mark.parametrize('version',['7.4.1','7.5'])
def test_one_xml_creator_owner_feeds_receipt_and_crystal(tmp_path,version):
    fixture=Path(__file__).parent/'hsuite/fixture/data-file-schema.xml'
    tree=ET.parse(fixture)
    creator=next(e for e in tree.getroot().iter() if e.tag.split('}')[-1]=='creator')
    creator.set('NAME','PWSCF');creator.set('VERSION',version)
    save=tmp_path/'test.save';save.mkdir();p=save/'data-file-schema.xml';tree.write(p)
    receipt=read_qe_symmetry_receipt(p);crystal=CrystalData.from_qe_save(str(save))
    assert receipt.creator_name==crystal.qe_creator_name=='PWSCF'
    assert receipt.creator_version==crystal.qe_creator_version==version
