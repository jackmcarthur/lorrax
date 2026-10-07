"""Asymmetric causal +/-q operands, distinct from antiunitary transport.

Positive and negative spectral residues at q are unequal Hermitian PSD
matrices and have different frequencies. The minus-q fields exchange them
under a nontrivial inter-sphere permutation and transpose. No response fit,
head or pole-model consumer supplies the independent convolution oracle.
"""
import numpy as np
import pytest
from gw import contour_reference as cd


def plant():
    a=np.asarray([[1.,.3+.2j],[-.4+.5j,.7],[.2-.6j,.4+.1j]])
    b=np.asarray([[.6+.2j,.5],[.8-.1j,-.2+.3j],[.3,.9-.4j]])
    A=.01*a@a.conj().T;B=.02*b@b.conj().T
    P=np.eye(3)[[2,0,1]]
    rho=np.asarray([[[1.+.4j,.2-.5j,-.3+.8j],[.3+.1j,.5+.7j,1.-.2j],[.8-.4j,-.3+.2j,.6+.9j]],
                    [[-.4+.3j,.7-.2j,.8+.5j],[.9+.2j,-.6+.1j,.3-.4j],[.2+.8j,.6-.7j,-.4+.1j]]])
    rho_minus=np.einsum('gh,amh->amg',P,rho.conj())
    omega_plus,omega_minus=.4,.9
    def project(matrix,pair):
        return np.einsum('amg,gh,bmh->abm',pair.conj(),matrix,pair)[None]
    RA,RB=project(A,rho),project(B,rho)
    minus_positive=P@B.T@P.T;minus_negative=P@A.T@P.T
    def sample(z,negative=False):
        if negative:
            op,om=omega_minus,omega_plus;pos,neg=minus_positive,minus_negative;pair=rho_minus
        else:op,om=omega_plus,omega_minus;pos,neg=A,B;pair=rho
        value=project(-pos/(z-op)+neg/(z+om),pair)
        derivative=project(pos/(z-op)**2-neg/(z+om)**2,pair)/(2*z)
        # Real residue owner receives already external-endpoint-transposed
        # minus-q projection at the SAME z, rather than a causal conjugate.
        if negative:value,derivative=value.swapaxes(-3,-2),derivative.swapaxes(-3,-2)
        return value,derivative
    x0=np.asarray([-1.2,-.9,-.4,-1e-9,0.,1e-9,.4,.9,1.2])
    x=np.broadcast_to(x0[None,None,:,None],(1,2,len(x0),3))
    valid=np.broadcast_to(np.asarray([True,True,False]),x.shape)
    return sample,RA,RB,x,valid,omega_plus,omega_minus


def integrate(sample,x,f,valid,eta,convention,n,*,wrong_imaginary_partner=False):
    kw=dict(eta=eta,analytic_convention=convention,band_valid=valid)
    def imaginary(z):return sample(z,negative=wrong_imaginary_partner)
    value,slope=imaginary(1j*eta)
    result,cp,cm,beta=cd.anchor_part(value,slope,x,f,**kw)
    nodes,weights=cd.imaginary_rule(n,eta,scale=.6)
    for u,w in zip(nodes,weights):
        result+=cd.imag_remainder_node(imaginary(1j*u)[0],u,w,x,f,cp,cm,beta,**kw)
    residue_weights=np.where(x<0,f,1-f)
    real=np.unique(abs(x)[valid&(residue_weights!=0)])
    for u in real:
        result+=cd.real_residue_node(sample(u+1j*eta)[0],sample(u+1j*eta,negative=True)[0],x,f,u,**kw)
    return result


@pytest.mark.parametrize('occupation',[0.,.25,.5,1.])
@pytest.mark.parametrize('convention',['time_ordered_fractional','retarded'])
def test_asymmetric_directed_q_integral_matches_independent_lehmann_denominators(occupation,convention):
    sample,RA,RB,x,valid,op,om=plant();eta=.035;f=np.full(x.shape,occupation)
    sheet=-1 if convention=='time_ordered_fractional' else 1
    expected=(np.einsum('kabl,kael->kabe',RA,np.where(valid,(1-f)/(x-op+1j*eta),0.))
             +np.einsum('kabl,kael->kabe',RB,np.where(valid,f/(x+om+sheet*1j*eta),0.)))
    coarse=integrate(sample,x,f,valid,eta,convention,128)
    fine=integrate(sample,x,f,valid,eta,convention,256)
    scale=np.max(abs(expected))
    assert np.max(abs(fine-expected))/scale<2e-7
    assert np.max(abs(coarse-fine))/scale<2e-7


def test_same_q_negative_imaginary_half_is_adjoint_not_minus_q_or_plain_conjugation():
    sample,*_=plant();z=.3+.27j
    positive=sample(z)[0];lower=sample(z.conjugate())[0]
    adjoint=positive.conj().swapaxes(-3,-2)
    np.testing.assert_allclose(lower,adjoint,rtol=2e-14,atol=2e-16)
    assert np.max(abs(lower-positive.conj()))/np.max(abs(lower))>.01
    # The SAME-z minus-q external transpose exchanges the two residues;
    # it is independently needed by occupied real crossings, not by loweriu.
    assert np.max(abs(lower-sample(z,negative=True)[0]))/np.max(abs(lower))>.01


def test_wrong_minus_q_imaginary_operand_is_decisively_rejected_by_the_oracle():
    sample,RA,RB,x,valid,op,om=plant();eta=.035;f=np.full(x.shape,.25)
    expected=(np.einsum('kabl,kael->kabe',RA,np.where(valid,(1-f)/(x-op+1j*eta),0.))
             +np.einsum('kabl,kael->kabe',RB,np.where(valid,f/(x+om+1j*eta),0.)))
    wrong=integrate(sample,x,f,valid,eta,'retarded',256,wrong_imaginary_partner=True)
    assert np.max(abs(wrong-expected))/np.max(abs(expected))>.01
