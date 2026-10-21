"""Current tangential action/side contracts, no native/material admission."""
import numpy as np


def matmul(a,b,transa=None,transb=None):
    if transa=='C':a=a.conj().swapaxes(-1,-2)
    if transb=='C':b=b.conj().swapaxes(-1,-2)
    return a@b


def test_original_parity_and_implicit_mirrors_preserve_actions_and_cache_identity():
    from gw.shared_pole_screening import _general_initial_supports,_general_support_panels
    rng=np.random.default_rng(552)
    sites=np.array([.2+.4j,.6+.3j])
    values=[rng.normal(size=(3,2,2))+1j*rng.normal(size=(3,2,2)) for _ in sites]
    directions={i:rng.normal(size=(3,2,1))+1j*rng.normal(size=(3,2,1)) for i in range(2)}
    supports=_general_initial_supports(sites,[0,1]);minus=np.array([0,2,1])
    current=dict(mm=matmul,recipe={},minus=minus)
    samples=dict(sites=sites,values=values,supports=supports)
    left,right=_general_support_panels(current,samples,directions,eigh_plan=None,column_extent=None)
    assert [row['side'] for row in supports]==['L','R','R','L']
    np.testing.assert_allclose(left[0][2],matmul(directions[0],values[0],transa='C'))
    np.testing.assert_allclose(right[0][2],matmul(values[0][minus].conj(),directions[0][minus].conj()))
    np.testing.assert_allclose(right[1][2],matmul(values[1],directions[1]))
    np.testing.assert_allclose(left[1][2],matmul(directions[1][minus].conj(),values[1][minus].conj(),transa='C'))
    again=_general_support_panels(current,samples,directions,eigh_plan=None,column_extent=None)
    assert again[0][0][2] is left[0][2] and again[1][1][2] is right[1][2]


def test_new_residue_supports_use_explicit_sides_not_sample_parity():
    from gw.shared_pole_screening import _general_support_panels
    # Left/right orthogonal: swapping the R tangent into a left observation
    # would be identically dark, although both proper actions are nonzero.
    l=np.array([1.,0.],complex)[None,:,None]
    r=np.array([0.,1.],complex)[None,:,None]
    w=l@r.conj().swapaxes(-1,-2)
    supports=[dict(sample_index=0,z_ry=.3+.4j,side='R',panel_id='right',candidate_index=0),
              dict(sample_index=1,z_ry=.4+.5j,side='L',panel_id='left',candidate_index=0)]
    current=dict(mm=matmul,recipe={},minus=np.array([0]),refinement_panels=dict(right=r,left=l))
    samples=dict(sites=np.array([.3+.4j,.4+.5j]),values=[w,w],supports=supports)
    left,right=_general_support_panels(current,samples,{},eigh_plan=None,column_extent=None)
    assert np.linalg.norm(left[0][2])==1 and np.linalg.norm(right[0][2])==1
    assert np.linalg.norm(matmul(r,w,transa='C'))==0
