"""Unimplemented transition-Galerkin prescription: tiny exact math oracles.

No production constructor, quadrature, spatial projection or GPU is used.
"""
import numpy as np

from tests._full_microscopic_oracle import ordered_pr,density_vertices,retarded_reference_pairs


def _transition_maps(psi,energy,rho,partner):
    nk,nb,ns,nmu=psi.shape;nt=nk*nb*nb
    t=np.zeros((nmu,nt),complex);kmat=np.zeros((nt,nt),complex);delta=[]
    index=lambda k,b,d:(k*nb+b)*nb+d
    for k in range(nk):
        kd=partner[k]
        for b in range(nb):
            for d in range(nb):
                j=index(k,b,d);delta.append(energy[k,b]-energy[kd,d])
                for mu in range(nmu):
                    for spin in range(ns):
                        t[mu,j]+=psi[k,b,spin,mu]*psi[kd,d,spin,mu].conjugate()/np.sqrt(nk)
                for a in range(nb):
                    for c in range(nb):
                        # Explicit output density action. Moving it through
                        # the resolvent would silently change endpoint order.
                        kmat[index(k,a,c),j]=((a==b)*rho[kd,c,d].conjugate()
                                              -rho[k,a,b]*(c==d))
    return t@kmat,t,np.asarray(delta),kmat


def _project(a,b,delta,q,gamma):
    return a@q,b@q,q.conj().T@np.diag(delta)@q-1j*gamma*np.eye(q.shape[1])


def _column_sigma_from_feedback(vertices,energy,rho,external,feedback,left,right,poles,eigenvectors,eta):
    """Independent resolvent/projector form, not a pole-residue sum."""
    nk,na,ni,nmu=vertices.shape
    out=[np.zeros((2,nk,na,na),complex) for _ in range(2)]
    tol=64*np.finfo(float).eps*max(np.max(abs(poles)),1.)
    pos=(poles.real>tol).astype(float)+.5*(abs(poles.real)<=tol)
    neg=(poles.real< -tol).astype(float)+.5*(abs(poles.real)<=tol)
    inv=np.linalg.inv(eigenvectors)
    projectors=[(eigenvectors*pos)@inv,-(eigenvectors*neg)@inv]
    for shell in range(2):
        for k in range(nk):
            for a in range(na):
                for b in range(na):
                    for n in range(ni):
                        z=external[k,a if shell==0 else b]-energy[k,n]+1j*eta
                        resolvent=np.linalg.solve(z*np.eye(len(poles))-feedback,np.eye(len(poles)))
                        for branch,projector in enumerate(projectors):
                            w=left@resolvent@projector@right
                            for m in range(ni):
                                weight=(float(m==n)-rho[k,m,n]) if branch==0 else rho[k,m,n]
                                for mu in range(nmu):
                                    for nu in range(nmu):
                                        out[branch][shell,k,a,b]+=(vertices[k,a,m,mu]*w[mu,nu]
                                            *vertices[k,b,n,nu].conjugate()*weight)
    return tuple(out)


def test_identity_transition_space_reproduces_full_coherent_PR_Dyson_and_column_complex_Sigma():
    rng=np.random.default_rng(714);nk,nb,ns,nmu=2,2,2,3
    psi=.25*(rng.normal(size=(nk,nb,ns,nmu))+1j*rng.normal(size=(nk,nb,ns,nmu)))
    energy=np.array([[-.4,.27],[-.31,.39]])
    rho=[]
    for _ in range(nk):
        u=np.linalg.qr(rng.normal(size=(nb,nb))+1j*rng.normal(size=(nb,nb)))[0]
        rho.append(u@np.diag([.13,.86])@u.conj().T)
    rho=np.asarray(rho)
    assert np.linalg.norm(rho*energy[:,None,:]-energy[:,:,None]*rho)>.01
    c=rng.normal(size=(nmu,nmu))+1j*rng.normal(size=(nmu,nmu));v=.01*(c@c.conj().T)
    sites=np.array([-.23+.3j,.18+.4j]);gamma=.035
    for qindex in range(nk):
        partner=(np.arange(nk)-qindex)%nk
        a,b,delta,kmat=_transition_maps(psi,energy,rho,partner)
        np.testing.assert_allclose(kmat,kmat.conj().T,atol=2e-15)
        ar,br,dred=_project(a,b,delta,np.eye(len(delta)),gamma)
        feedback=dred+br.conj().T@v@ar
        exact=ordered_pr(psi,energy,rho,partner,sites,gamma)
        for z,pref in zip(sites,exact):
            pred=ar@np.linalg.solve(z*np.eye(len(delta))-dred,br.conj().T)
            np.testing.assert_allclose(pred,pref,rtol=4e-13,atol=3e-15)
            direct=np.linalg.solve(np.eye(nmu)-v@pref,v)-v
            auxiliary=v@ar@np.linalg.solve(z*np.eye(len(delta))-feedback,br.conj().T@v)
            np.testing.assert_allclose(auxiliary,direct,rtol=5e-11,atol=2e-16)
        # q1 has distinct bare gaps, avoiding irrelevant dark degenerate
        # coordinates when checking the explicit modal representation.
        if qindex==1:
            poles,ev=np.linalg.eig(feedback);evinv=np.linalg.inv(ev)
            left=v@ar@ev;right=evinv@br.conj().T@v
            residues=np.stack([np.outer(left[:,j],right[j]) for j in range(len(poles))])
            vertices=density_vertices(psi,partner,[0,1]);eta=.2
            actual=retarded_reference_pairs(vertices,energy[partner],rho[partner],energy,poles,residues,eta)
            expected=_column_sigma_from_feedback(vertices,energy[partner],rho[partner],energy,
                feedback,v@ar,br.conj().T@v,poles,ev,eta)
            for x,y in zip(actual,expected):np.testing.assert_allclose(x,y,rtol=3e-11,atol=3e-15)
            assert np.linalg.norm(actual[0][1]-actual[0][1].conj().swapaxes(-1,-2))>1e-7


def test_orthogonal_transition_projection_keeps_Hermitian_gap_generator_and_exact_lag_decay():
    rng=np.random.default_rng(715);nt,ports,keep=9,3,5
    delta=np.linspace(-1.2,1.7,nt);gamma=.043
    a=rng.normal(size=(ports,nt))+1j*rng.normal(size=(ports,nt))
    b=rng.normal(size=(ports,nt))+1j*rng.normal(size=(ports,nt))
    q=np.linalg.qr(rng.normal(size=(nt,keep))+1j*rng.normal(size=(nt,keep)))[0]
    ar,br,dred=_project(a,b,delta,q,gamma)
    np.testing.assert_allclose(dred+1j*gamma*np.eye(keep),(dred+1j*gamma*np.eye(keep)).conj().T,atol=3e-16)
    np.testing.assert_allclose(np.linalg.eigvals(dred).imag,-gamma,atol=4e-16)
    assert ar.shape==br.shape==(ports,keep) # ALL spatial ports remain


def test_live_pair_Galerkin_preserves_genuine_inverted_collective_gain_without_PSD_projection():
    gamma=.05;v=np.array([[.8]])
    delta=np.array([0.,-1.,1.,0.]);t=np.ones((1,4),complex)
    a=t@np.diag([0.,1.,-1.,0.]);b=t
    u=np.array([[1.,1j],[1j,1.]])/np.sqrt(2)
    q=np.eye(4)[:,[1,2]]@u # nontrivial orthonormal live-pair basis
    ar,br,dred=_project(a,b,delta,q,gamma)
    np.testing.assert_allclose(np.linalg.eigvals(dred).imag,-gamma,atol=3e-16)
    feedback=dred+br.conj().T@v@ar
    poles=np.linalg.eigvals(feedback)
    np.testing.assert_allclose(np.sort(poles.imag),np.sort([-gamma-np.sqrt(.6),-gamma+np.sqrt(.6)]),atol=4e-15)
    assert poles.imag.max()>0.7 # physical inversion is not numerically stabilized
    z=1.3j
    pr=ar@np.linalg.solve(z*np.eye(2)-dred,br.conj().T)
    np.testing.assert_allclose(pr,[[-2/((z+1j*gamma)**2-1)]],atol=4e-16)
    w=v@ar@np.linalg.solve(z*np.eye(2)-feedback,br.conj().T@v)
    np.testing.assert_allclose(w,np.linalg.solve(np.eye(1)-v@pr,v)-v,atol=4e-16)
