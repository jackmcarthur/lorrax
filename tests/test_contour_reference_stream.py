"""Streamed node algebra versus the literal pre-factoring implementation.

The frozen baseline is d1b6150bf real_part; it never enters production.
A cubic response independently tests the exact Hermite interpolation law.
"""
import numpy as np
import pytest
from gw import contour_reference as cr
from gw.contour_reference import _inputs,_kernels,_residue_weights,_apply,_dagger

def legacy_real_part(values, derivatives_s, values_t, derivatives_t_s, x, occ, nodes,
              *, eta, n_active=None, band_valid=None, spacing=None,
              analytic_convention="time_ordered_fractional", xp=np):
    """Cubic Hermite residues on an authenticated uniform positive real grid.

    Values are sampled at ``nodes+i eta``; the partner samples are already
    transposed and contracted with the current pair density. Both derivative
    arrays use the derivative of that same positive real coordinate. Every
    nonzero physical residue must have two bracketing nodes. ``n_active``
    excludes explicitly zero carrier bands, never a physical band tail.
    ``band_valid`` optionally masks per-k ragged spectrum padding explicitly;
    padded wavefunction coefficients must already be exact zero at the caller.
    ``spacing`` can select an integer coarsening of the same node grid.
    """
    xh, fh = _inputs(x, occ, eta)
    _kernels(xh, fh, eta, analytic_convention, np)
    nodes = np.asarray(nodes, np.float64)
    if nodes.ndim != 1 or nodes.size < 2 or nodes[0] != 0 or not np.isfinite(nodes).all():
        raise ValueError("CD residues need a finite uniform grid starting at zero")
    base = nodes[1] - nodes[0]
    if base <= 0 or not np.allclose(np.diff(nodes), base, rtol=2e-12, atol=2e-14 * base):
        raise ValueError("CD residue grid must be strictly increasing and uniform")
    spacing = base if spacing is None else float(spacing)
    stride = round(spacing / base)
    if stride < 1 or not np.isclose(stride * base, spacing, rtol=2e-12, atol=0):
        raise ValueError("CD residue spacing must be an integer multiple of its base grid")
    active, residue = _residue_weights(xh, fh, n_active, band_valid)
    query = np.abs(xh)
    last = nodes[((nodes.size - 1) // stride) * stride]
    if np.any(active & (residue != 0) & (query > last)):
        raise ValueError("CD residue coverage: an active crossing exceeds the last coarse-grid node")
    for samples in (values, derivatives_s, values_t, derivatives_t_s):
        if len(samples) != len(nodes):
            raise ValueError("CD values/derivatives must cover every declared real node")
    index = np.floor(query / spacing).astype(np.int64)
    index = np.minimum(index, max(0, (nodes.size - 1) // stride - 1))
    t = query / spacing - index
    coeff = (2*t**3 - 3*t**2 + 1, -2*t**3 + 3*t**2,
             spacing*(t**3 - 2*t**2 + t), spacing*(t**3 - t**2))
    sign = xp.asarray(xh < 0)
    active = xp.asarray(active)
    index = xp.asarray(index)
    residue = xp.asarray(residue)
    coeff = tuple(xp.asarray(c) for c in coeff)
    total = 0
    for inode in range(0, len(nodes), stride):
        node = inode // stride
        z = nodes[inode] + 1j * eta
        partner_value = values_t[inode]
        partner_slope = 2*z*derivatives_t_s[inode]
        if analytic_convention == "retarded":
            partner_value = _dagger(partner_value, xp)
            partner_slope = _dagger(partner_slope, xp)
        for deriv, value, partner in ((False, values[inode], partner_value),
                                      (True, 2*z*derivatives_s[inode], partner_slope)):
            left, right = coeff[2:] if deriv else coeff[:2]
            selected = xp.where(active, residue * (xp.where(index == node, left, 0.)
                                                   + xp.where(index + 1 == node, right, 0.)), 0.)
            total = total + _apply(value, xp.where(sign, 0., selected), xp)
            total = total + _apply(partner, xp.where(sign, selected, 0.), xp)
    return total


def fixture(cubic=False):
    eta=.08;nodes=.25*np.arange(17,dtype=np.float64)
    energies=np.array([-.1,.2,.8,1.1,1e4]);external=np.array([.2,.4])
    x=np.broadcast_to(external[:,None,None,None]-energies[None,None,None,:],(2,3,1,5)).copy()
    f=np.broadcast_to(np.array([.9,.7,.5,.1,0.])[None,None,None,:],x.shape).copy()
    valid=np.broadcast_to(np.array([True,True,True,True,False])[None,None,None,:],x.shape).copy()
    u=np.array([1,.3+1j,-.4+.2j]);r=np.outer(u,u.conj())
    v=np.array([.2-.1j,.7,1+.3j]);s=np.outer(v,v.conj())
    weights=np.array([[1,.7,.3,.2,0.],[.8,.6,.4,.1,0.]])[:,None,None,:]
    def sample(z):
        if cubic:
            m=r+(s+1j*r)*z+(.3*r-1j*s)*z*z+.2*s*z**3
            dm=(s+1j*r)+2*(.3*r-1j*s)*z+.6*s*z*z
        else:
            m=r/(z-.65)-r.T/(z+.65)+.3*s/(z-1.35)-.3*s.T/(z+1.35)
            dm=-r/(z-.65)**2+r.T/(z+.65)**2-.3*s/(z-1.35)**2+.3*s.T/(z+1.35)**2
        value=-m[None,:,:,None]*weights;derivative=-dm[None,:,:,None]*weights/(2*z)
        return value,derivative,value.swapaxes(1,2),derivative.swapaxes(1,2)
    data=[sample(float(t)+1j*eta) for t in nodes]
    return eta,nodes,x,f,valid,sample,tuple([d[i] for d in data] for i in range(4))

@pytest.mark.parametrize('convention',['time_ordered_fractional','retarded'])
@pytest.mark.parametrize('spacing',[.25,.5])
def test_stream_preserves_original_sum(convention,spacing):
    eta,nodes,x,f,valid,_,data=fixture()
    expected=legacy_real_part(*data,x,f,nodes,eta=eta,band_valid=valid,spacing=spacing,analytic_convention=convention)
    prepared=cr.prepare_real_residue_grid(x,f,nodes,eta=eta,band_valid=valid,spacing=spacing,analytic_convention=convention)
    actual=0
    for inode in range(0,len(nodes),prepared.stride):
        for term in prepared(inode,*(d[inode] for d in data)):actual=actual+term
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(cr.real_part(*data,x,f,nodes,eta=eta,band_valid=valid,spacing=spacing,analytic_convention=convention),expected)

@pytest.mark.parametrize('convention',['time_ordered_fractional','retarded'])
@pytest.mark.parametrize('spacing',[.25,.5])
def test_cubic_stream_matches_exact_crossings(convention,spacing):
    eta,nodes,x,f,valid,sample,data=fixture(cubic=True)
    prepared=cr.prepare_real_residue_grid(x,f,nodes,eta=eta,band_valid=valid,spacing=spacing,analytic_convention=convention)
    actual=0
    for inode in range(0,len(nodes),prepared.stride):
        for term in prepared(inode,*(d[inode] for d in data)):actual=actual+term
    physical,residue=cr._residue_weights(x,f,None,valid)
    crossing=np.unique(np.abs(x[np.broadcast_to(physical,x.shape)&(residue!=0)]));expected=0
    for node in crossing:
        a,_,p,_=sample(float(node)+1j*eta)
        expected=expected+cr.real_residue_node(a,p,x,f,float(node),eta=eta,band_valid=valid,analytic_convention=convention)
    np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=3e-14)

def test_stream_rejects_bad_node_and_insufficient_coarse_coverage():
    eta,nodes,x,f,valid,_,data=fixture();prepared=cr.prepare_real_residue_grid(x,f,nodes,eta=eta,band_valid=valid,spacing=.5)
    for index in (True,.5,-1,len(nodes),1):
        with pytest.raises(ValueError):prepared(index,*(d[0] for d in data))
    with pytest.raises(ValueError,match='coverage'):
        cr.prepare_real_residue_grid(x,f,nodes[:3],eta=eta,band_valid=valid,spacing=.5)
