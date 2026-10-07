"""Static transverse geometry through positive curl-Poisson field maps.

This low-level primitive supplies P_ij(K)/K^2 with inverse Fourier
(2*pi)^-3 integral exp(iK.r)dK. A=1/(4*pi*r), Phi=A*J, and
<J,D K>=<curl Phi_J,curl Phi_K>. It contains no physical Breit sign,
4*pi, Rydberg, FFT/cell normalization, head policy or fitting-channel solve.

Piecewise current values may jump at their compact support boundary.
Differentiating the continuous Poisson potential includes the resulting
normal-current divergence contact without differentiating the current.
Ordinary three-dimensional static kernels only; no Helmholtz/truncation or
production current-routing admission is supplied by this module.
"""
from __future__ import annotations

from math import comb
import numpy as np


def _full_shell_labels(maximum):
    return np.asarray([(l,m) for l in range(maximum+1) for m in range(-l,l+1)],dtype=np.int64)


def curl_angular_maps(lm):
    """Cartesian Gaunt/curl maps on complete complex Condon--Shortley shells.

    Source LM order must be complete canonical shells. Output includes the
    full Lmax+1 shell. Normal map is <Y_out|rhat_a Y_in>; the two sparse
    gradient branches connect only Lout=Lin-1 and Lin+1. This fixed angular
    geometry is independent of radial samples, atom and momentum transfer.
    """
    from scipy.special import sph_harm_y

    labels=np.asarray(lm)
    if (labels.ndim!=2 or labels.shape[1]!=2 or not len(labels)
            or not np.all(np.isfinite(labels)) or np.any(labels!=np.round(labels))):
        raise ValueError("curl maps require finite integer LM labels")
    maximum=int(np.max(labels[:,0]))
    if maximum<0 or not np.array_equal(labels,_full_shell_labels(maximum)):
        raise ValueError("curl maps require complete canonical LM shells")
    target=_full_shell_labels(maximum+1)
    ct,wt=np.polynomial.legendre.leggauss(maximum+5)
    phi=np.arange(4*maximum+16)*2*np.pi/(4*maximum+16)
    theta=np.arccos(ct)
    direction=np.stack(np.broadcast_arrays(np.sqrt(1-ct[:,None]**2)*np.cos(phi),
                         np.sqrt(1-ct[:,None]**2)*np.sin(phi),ct[:,None]),axis=-1).reshape(-1,3)
    weight=np.repeat(wt,len(phi))*2*np.pi/len(phi)
    source_Y=np.asarray([sph_harm_y(l,m,theta[:,None],phi).ravel() for l,m in labels])
    target_Y=np.asarray([sph_harm_y(l,m,theta[:,None],phi).ravel() for l,m in target])
    normal=np.einsum('os,s,sa,is->aoi',target_Y.conj(),weight,direction,source_Y)
    lower=target[:,0,None]==labels[None,:,0]-1
    upper=target[:,0,None]==labels[None,:,0]+1
    forbidden=np.logical_not(lower|upper)
    if np.max(np.abs(normal*forbidden[None]))>1024*np.finfo(float).eps*(maximum+1):
        raise ValueError("angular curl quadrature failed its exact degree selection")
    normal*=np.logical_not(forbidden)[None]
    epsilon=np.zeros((3,3,3))
    for p,a,b in ((0,1,2),(1,2,0),(2,0,1)):
        epsilon[p,a,b]=1;epsilon[p,b,a]=-1
    curl_plus=np.einsum('pab,aoi,oi->pobi',epsilon,normal,upper)
    curl_minus=np.einsum('pab,aoi,oi->pobi',epsilon,normal,lower)
    return dict(lm=labels.astype(np.int64),output_lm=target,normal=normal,
                lower=lower,upper=upper,epsilon=epsilon,
                curl_plus=curl_plus,curl_minus=curl_minus)


def radial_poisson_gradient_maps(radius,lm_ell,*,support_radius,
                                  interpolation_degree=3,quadrature_order=None):
    r"""Exact panel integrals and positive quadrature for Poisson gradients.

    Same physical-density cardinal rule as radial_coulomb_metric_interpolated:
    fixed-width local polynomials between positive samples, regular r^L on
    the first origin panel. No divide-by-r^L interpolation of sample noise.
    Weak-cusp origin refinement remains a caller convergence obligation.

    Returns g_plus=-r^(-L-2) integral_0^r t^(L+2)J_L(t)dt and
    g_minus=r^(L-1) integral_r^R t^(1-L)J_L(t)dt, each (degree,quad,sample).
    The outer weighted panel antiderivative uses an analytic hypergeometric
    integral in the local coordinate to avoid subtracting large powers.
    Moments0/2 use the identical interpolant. Boundary current may be nonzero.
    """
    from scipy.special import hyp2f1

    r=np.asarray(radius,dtype=np.float64);ell=np.asarray(lm_ell)
    R=float(support_radius);degree=int(interpolation_degree)
    if (r.ndim!=1 or len(r)<2 or not np.all(np.isfinite(r)) or np.any(r<=0)
            or np.any(np.diff(r)<=0) or ell.ndim!=1 or not len(ell)
            or not np.all(np.isfinite(ell)) or np.any(ell<0) or np.any(ell!=np.round(ell))
            or not np.isfinite(R) or R<r[-1] or degree!=interpolation_degree
            or degree<1 or degree>=len(r)):
        raise ValueError("invalid compact current radial interpolant")
    degrees=np.unique(ell.astype(np.int64));nr=len(r)
    order=max(16,int(degrees[-1])+degree+4) if quadrature_order is None else int(quadrature_order)
    if order<2 or (quadrature_order is not None and order!=quadrature_order):
        raise ValueError("radial field quadrature must have integer order >=2")
    x,w=np.polynomial.legendre.leggauss(order);u,wu=(x+1)/2,w/2
    edges=np.concatenate(([0.],r,([R] if R>r[-1] else [])))
    nq=(len(edges)-1)*order
    query=np.empty(nq);weights=np.empty(nq);interpolation=np.zeros((nq,nr));panels=[]
    for panel,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        h=b-a;first,last=panel*order,(panel+1)*order
        query[first:last]=a+h*u;weights[first:last]=h*wu
        if panel==0:
            interpolation[first:last,0]=1.;continue
        left=int(np.searchsorted(r,(a+b)/2))-(degree+1)//2
        left=max(0,min(left,nr-degree-1));columns=np.arange(left,left+degree+1)
        z=(r[columns]-a)/h;basis=np.empty((degree+1,degree+1))
        for i in range(degree+1):
            other=np.delete(z,i)
            basis[:,i]=np.polynomial.polynomial.polyfromroots(other)/np.prod(z[i]-other)
        interpolation[first:last,columns]=np.polynomial.polynomial.polyval(u,basis).T
        panels.append((first,last,a,h,columns,basis))
    plus=[];minus=[];moment0=[];moment2=[];origin=[]
    for value in degrees:
        l=int(value);M=np.zeros((nq,nr));N=np.zeros_like(M)
        M[:order,0]=query[:order]**(2*l+3)/(r[0]**l*(2*l+3))
        prefix=np.zeros(nr);prefix[0]=r[0]**(l+3)/(2*l+3)
        q2=np.zeros(nr);q2[0]=r[0]**(l+5)/(2*l+5)
        for first,last,a,h,columns,basis in panels:
            partial=[]
            for exponent in (l+2,l+4):
                power=np.asarray([comb(exponent,k)*a**(exponent-k)*h**k
                                  for k in range(exponent+1)])
                antiderivative=np.zeros((exponent+degree+2,degree+1))
                for i in range(degree+1):
                    polynomial=np.polynomial.polynomial.polymul(power,basis[:,i])
                    antiderivative[1:len(polynomial)+1,i]=h*polynomial/np.arange(1,len(polynomial)+1)
                partial.append(antiderivative)
            M[first:last]=prefix
            M[first:last,columns]+=np.polynomial.polynomial.polyval(u,partial[0]).T
            prefix[columns]+=np.polynomial.polynomial.polyval(1.,partial[0])
            q2[columns]+=np.polynomial.polynomial.polyval(1.,partial[1])
        suffix=np.zeros(nr)
        if l>0:
            for first,last,a,h,columns,basis in reversed(panels):
                anti=np.asarray([h*a**(1-l)*u**(k+1)/(k+1)*
                     hyp2f1(l-1,k+1,k+2,-h*u/a) for k in range(degree+1)])
                full=np.asarray([h*a**(1-l)/(k+1)*hyp2f1(l-1,k+1,k+2,-h/a)
                                 for k in range(degree+1)])
                values=anti.T@basis;total=full@basis
                N[first:last]=suffix
                N[first:last,columns]+=total[None]-values
                suffix[columns]+=total
            N[:order]=suffix
            N[:order,0]+=(r[0]**2-query[:order]**2)/(2*r[0]**l)
        plus.append(-M/query[:,None]**(l+2))
        minus.append(N*query[:,None]**(l-1) if l>0 else N)
        moment0.append(prefix);moment2.append(q2);origin.append((query[:order]/r[0])**l)
    arrays=(plus,minus,moment0,moment2,origin,query,weights,interpolation)
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Poisson gradient maps are unresolved in float64")
    return dict(degrees=degrees,plus=np.asarray(plus),minus=np.asarray(minus),
                moments0=np.asarray(moment0),moments2=np.asarray(moment2),
                quadrature_radius=query,quadrature_weights_dr=weights,
                interpolation_map=interpolation,origin_factors=np.asarray(origin),
                origin_row_count=order,support_radius=R,
                interpolation_degree=degree,quadrature_order=order)


def static_transverse_geometry(radius,lm,**controls):
    """Small separable angular/radial arrays; no dense Cartesian/radial tensor."""
    angular=curl_angular_maps(lm)
    radial=radial_poisson_gradient_maps(radius,angular['lm'][:,0],**controls)
    return dict(angular=angular,radial=radial,kernel='static_transverse_P_over_K2',
                physical_scale='caller-owned Breit sign, 4pi, Ry and FFT/cell factors')


def build_two_moment_compensation(lm_ell,*,support_radius,power=6):
    r"""Analytic compact duals to physical Q0 and Q2 current moments.

    Profiles h_j=r^L (r/R)^(2j) (1-r²/R²)^power / R^(2L+3), j=0,1.
    A dimensionless beta matrix maps these to [Q0,Q2/R²]; only its second
    inverse column carries the final R^-2 conversion. The resulting duals
    have moments (1,0) and (0,1). No density sample interpolation defines g.
    """
    from scipy.special import beta

    values=np.asarray(lm_ell);R=float(support_radius);p=int(power)
    if (values.ndim!=1 or not len(values) or not np.all(np.isfinite(values))
            or np.any(values<0) or np.any(values!=np.round(values))
            or not np.isfinite(R) or R<=0 or p!=power or p<2):
        raise ValueError("invalid two-moment compact compensation")
    degrees=np.unique(values.astype(np.int64));matrices=[];duals=[];conditions=[]
    for l in degrees:
        matrix=np.asarray([[beta(l+1.5+j,p+1)/2 for j in range(2)],
                           [beta(l+2.5+j,p+1)/2 for j in range(2)]])
        inverse=np.linalg.solve(matrix,np.eye(2));condition=np.linalg.cond(matrix)
        if not np.all(np.isfinite(inverse)) or condition*np.finfo(float).eps>1e-9:
            raise ValueError("two-moment beta dual is unresolved in float64")
        matrices.append(matrix);conditions.append(condition)
        duals.append(inverse*np.asarray([1.,1/R**2])[None])
    return dict(degrees=degrees,profile_coefficients=np.asarray(duals),
                dimensionless_moment_matrix=np.asarray(matrices),
                condition_number=np.asarray(conditions),support_radius=R,power=p,
                moments='physical Q0=int r^(L+2)J dr; Q2=int r^(L+4)J dr')


def evaluate_two_moment_compensation(compensation,radius):
    """Physical radial dual functions (degree, Q0/Q2 dual, radius)."""
    r=np.asarray(radius,dtype=np.float64);R=compensation['support_radius']
    if r.ndim!=1 or not np.all(np.isfinite(r)) or np.any(r<0):
        raise ValueError("compensation radii must be a finite nonnegative vector")
    x=np.minimum(r/R,1.);window=np.maximum(1-x*x,0.)**compensation['power']
    profiles=np.asarray([np.stack((r**l*window/R**(2*l+3),
                                  r**l*window*x*x/R**(2*l+3)))
                         for l in compensation['degrees']])
    return np.einsum('ljs,ljd->lds',profiles,compensation['profile_coefficients'])


def _sonine_profile_transform(l,p,R,magnitude):
    """Integral r² h_p(r) j_L(Kr)dr for h_p=r^L window^p/R^(2L+3)."""
    from scipy.special import gammaln,spherical_jn

    x=magnitude*R;n=l+p+1;small=x<1e-3
    # j_n(x)/x^(p+1) has an analytic x^L limit; never evaluate 0/0.
    coefficient=np.exp((p-n-1)*np.log(2.)+gammaln(p+1)-gammaln(n+1.5)+.5*np.log(np.pi))
    t=x*x
    series=coefficient*x**l*(1-t/(2*(2*n+3))+t*t/(8*(2*n+3)*(2*n+5)))
    safe=np.where(small,1.,x)
    direct=np.exp(p*np.log(2.)+gammaln(p+1))*spherical_jn(n,safe)/safe**(p+1)
    return np.where(small,series,direct)/R**l


def two_moment_compensation_radial_fourier(compensation,magnitude):
    """Exact Sonine radial transforms of the Q0/Q2 duals, no fitted K cache."""
    from scipy.special import beta,gammaln

    K=np.asarray(magnitude,dtype=np.float64)
    if K.ndim!=1 or not np.all(np.isfinite(K)) or np.any(K<0):
        raise ValueError("compensation momenta must be a finite nonnegative vector")
    R,p=compensation['support_radius'],compensation['power'];profiles=[]
    for l in compensation['degrees']:
        first=_sonine_profile_transform(int(l),p,R,K)
        second=first-_sonine_profile_transform(int(l),p+1,R,K)
        profiles.append(np.stack((first,second)))
    result=np.einsum('ljg,ljd->ldg',np.asarray(profiles),compensation['profile_coefficients'])
    small=K*R<1e-3
    if np.any(small):
        # Combine the dual's moment series, rather than cancelling two
        # finite monopoles. Q0=(1,0), Q2=(0,1) are exact defining moments.
        q=K[small]
        for row,l in enumerate(compensation['degrees']):
            series=np.broadcast_to(np.asarray([1.,0.])[:,None],(2,len(q))).copy()
            term=np.ones_like(q)
            for order in range(1,4):
                term*=-q*q/(2*order*(2*l+2*order+1))
                moment=(np.asarray([0.,1.]) if order==1 else
                    np.asarray([R**(2*order)*beta(l+1.5+j+order,p+1)/2 for j in range(2)])
                    @compensation['profile_coefficients'][row])
                series+=moment[:,None]*term[None]
            prefactor=np.exp(.5*np.log(np.pi)-(l+1)*np.log(2.)-gammaln(l+1.5))*q**l
            result[row,:,small]=(series*prefactor).T
    return result


def current_radial_moments(current,geometry):
    """Full-Bloch current moments of the same physical source interpolant."""
    value=_currents(current,geometry);angular,radial=geometry['angular'],geometry['radial']
    rows=np.searchsorted(radial['degrees'],angular['lm'][:,0])
    table=np.stack((radial['moments0'][rows],radial['moments2'][rows]),axis=1)
    return np.einsum('idn,...bin->...bid',table,value)


def two_moment_current_fourier(moments,lm,compensation,wavevectors,*,center_cart=None):
    """Physical compensated Cartesian FT, shape (...,3,G); no P/sign/units.

    Input moments end (...,3,LM,2) in physical Q0/Q2 units. Caller must
    extract them from full unwrapped Bloch current before this operation.
    """
    from scipy.special import sph_harm_y

    angular=curl_angular_maps(lm);labels=angular['lm']
    value=np.asarray(moments,dtype=np.complex128);K=np.asarray(wavevectors,dtype=np.float64)
    if (value.ndim<3 or value.shape[-3:]!=(3,len(labels),2)
            or not np.all(np.isfinite(value)) or K.ndim!=2 or K.shape[1]!=3
            or not np.all(np.isfinite(K))):
        raise ValueError("invalid physical Cartesian compensation Fourier input")
    rows=np.searchsorted(compensation['degrees'],labels[:,0])
    if np.any(rows>=len(compensation['degrees'])) or not np.array_equal(compensation['degrees'][rows],labels[:,0]):
        raise ValueError("compensation does not span every current degree")
    magnitude=np.linalg.norm(K,axis=1)
    theta=np.arccos(np.clip(np.divide(K[:,2],magnitude,out=np.ones(len(K)),where=magnitude>0),-1,1))
    phi=np.arctan2(K[:,1],K[:,0])
    Y=np.asarray([4*np.pi*(-1j)**int(l)*sph_harm_y(l,m,theta,phi) for l,m in labels])
    radial=two_moment_compensation_radial_fourier(compensation,magnitude)[rows]
    result=np.einsum('...bid,idg,ig->...bg',value,radial,Y)
    if center_cart is not None:
        center=np.asarray(center_cart,dtype=np.float64)
        if center.shape!=(3,) or not np.all(np.isfinite(center)):
            raise ValueError("compensation center must be a finite Cartesian triple")
        result*=np.exp(-1j*(K@center))
    return result


def compensation_poisson_gradient_maps(compensation,radius):
    """Small analytic g Poisson-gradient maps, (degree,radius,moment_dual)."""
    from scipy.special import beta,betaincc,betainc

    r=np.asarray(radius,dtype=np.float64);R,p=compensation['support_radius'],compensation['power']
    if (r.ndim!=1 or not len(r) or not np.all(np.isfinite(r))
            or np.any(r<=0) or np.any(r>R)):
        raise ValueError("analytic compensation field nodes must lie in (0,R]")
    plus=[];minus=[];x2=(r/R)**2
    for row,l in enumerate(compensation['degrees']):
        M=np.asarray([beta(l+1.5+j,p+1)*betainc(l+1.5+j,p+1,x2)/2 for j in range(2)])
        N=np.asarray([R**(-2*l-1)*beta(j+1,p+1)*betaincc(j+1,p+1,x2)/2 for j in range(2)])
        dual=compensation['profile_coefficients'][row]
        plus.append(-np.einsum('jq,jd->qd',M,dual)/r[:,None]**(l+2))
        minus.append(np.einsum('jq,jd->qd',N,dual)*r[:,None]**(l-1) if l else np.zeros((len(r),2)))
    if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
        raise ValueError("analytic compensation gradients are unresolved in float64")
    return dict(plus=np.asarray(plus),minus=np.asarray(minus),degrees=compensation['degrees'],
                radius=r,support_radius=R,power=p)


def compensation_curl_field_tile(moments,compensation,geometry,*,quadrature_slice,field_maps=None):
    """Analytic g curl fields on source quadrature; never interpolate g samples."""
    angular,radial=geometry['angular'],geometry['radial'];labels=angular['lm']
    value=np.asarray(moments,dtype=np.complex128)
    if value.ndim<3 or value.shape[-3:]!=(3,len(labels),2) or not np.all(np.isfinite(value)):
        raise ValueError("compensated current moments must end in (3,N_lm,2)")
    R=compensation['support_radius']
    if R!=radial['support_radius']:
        raise ValueError("compensation and source support radii must agree")
    r=radial['quadrature_radius'][quadrature_slice]
    if r.ndim!=1 or not len(r):
        raise ValueError("compensation field tile must select nonempty nodes")
    rows=np.searchsorted(compensation['degrees'],labels[:,0])
    if np.any(rows>=len(compensation['degrees'])) or not np.array_equal(compensation['degrees'][rows],labels[:,0]):
        raise ValueError("compensation does not span every current degree")
    if field_maps is None:
        maps=compensation_poisson_gradient_maps(compensation,r)
        plus,minus=maps['plus'][rows],maps['minus'][rows]
    else:
        if (field_maps['support_radius']!=R or field_maps['power']!=compensation['power']
                or not np.array_equal(field_maps['degrees'],compensation['degrees'])
                or not np.array_equal(field_maps['radius'],radial['quadrature_radius'])):
            raise ValueError("cached compensation gradients do not bind this geometry")
        plus=field_maps['plus'][:,quadrature_slice][rows]
        minus=field_maps['minus'][:,quadrature_slice][rows]
    upper=np.einsum('iqd,...bid->...qbi',plus,value)
    lower=np.einsum('iqd,...bid->...qbi',minus,value)
    return (np.einsum('pobi,...qbi->...qpo',angular['curl_plus'],upper)
            +np.einsum('pobi,...qbi->...qpo',angular['curl_minus'],lower))


def static_compensation_bilinear(left_moments,right_moments,compensation,geometry,*,
                                quadrature_tile=32,include_exterior=True,field_maps=None):
    """Analytic g metric with the identical positive quadrature and exact exterior.

    The compensation stays analytic in both this metric and its Sonine FT.
    Treating samples of g as the source cardinal interpolant is a different
    density model and would spoil exact Q0/Q2 matching.
    """
    lhs=np.asarray(left_moments,dtype=complex);rhs=np.asarray(right_moments,dtype=complex)
    width=int(quadrature_tile)
    if width!=quadrature_tile or width<=0:
        raise ValueError("compensation field tile must be a positive integer")
    r=geometry['radial']['quadrature_radius']
    maps=(compensation_poisson_gradient_maps(compensation,r) if field_maps is None else field_maps)
    weights=geometry['radial']['quadrature_weights_dr']*r*r
    result=np.zeros(np.broadcast_shapes(lhs.shape[:-3],rhs.shape[:-3]),dtype=complex)
    for first in range(0,len(r),width):
        tile=slice(first,min(first+width,len(r)))
        a=compensation_curl_field_tile(lhs,compensation,geometry,quadrature_slice=tile,field_maps=maps)
        b=compensation_curl_field_tile(rhs,compensation,geometry,quadrature_slice=tile,field_maps=maps)
        result+=np.einsum('...qpo,q,...qpo->...',a.conj(),weights[tile],b)
    if include_exterior:
        angular=geometry['angular']
        a=-np.einsum('pobi,...bi->...po',angular['curl_plus'],lhs[...,0])
        b=-np.einsum('pobi,...bi->...po',angular['curl_plus'],rhs[...,0])
        l=angular['output_lm'][:,0];w=np.zeros(len(l));use=l>0
        w[use]=compensation['support_radius']**(1-2*l[use])/(2*l[use]-1)
        result+=np.einsum('...po,o,...po->...',a.conj(),w,b)
    return result


def _currents(current,geometry):
    value=np.asarray(current,dtype=np.complex128)
    angular,radial=geometry['angular'],geometry['radial']
    expected=(3,len(angular['lm']),radial['plus'].shape[-1])
    if value.ndim<3 or value.shape[-3:]!=expected or not np.all(np.isfinite(value)):
        raise ValueError("Cartesian current must end in (3,N_lm,N_radial_sample)")
    return value


def curl_poisson_field_tile(current,geometry,*,quadrature_slice):
    """Apply the positive field factor on an explicitly bounded radial tile.

    Input (...,3,lm,sample), output (...,quad_tile,3,output_lm). The caller
    owns current/band/centroid distribution and never needs a full field cloud.
    Low-level NumPy geometry only; a distributed fitting owner can translate
    these same finite array contractions to its existing JAX/GEMM programs.
    """
    value=_currents(current,geometry);angular,radial=geometry['angular'],geometry['radial']
    indices=np.arange(len(radial['quadrature_radius']))[quadrature_slice]
    if indices.ndim!=1 or not len(indices) or np.any(np.diff(indices)<=0):
        raise ValueError("field quadrature slice must select increasing nonempty nodes")
    lookup={int(l):i for i,l in enumerate(radial['degrees'])}
    rows=np.asarray([lookup[int(l)] for l in angular['lm'][:,0]])
    plus=radial['plus'][:,indices][rows]
    minus=radial['minus'][:,indices][rows]
    # All angular maps are constant size; radial field tiles remain bounded.
    upper=np.einsum('iqn,...bin->...qbi',plus,value)
    lower=np.einsum('iqn,...bin->...qbi',minus,value)
    return (np.einsum('pobi,...qbi->...qpo',angular['curl_plus'],upper)
            +np.einsum('pobi,...qbi->...qpo',angular['curl_minus'],lower))


def exterior_curl_coefficients(current,geometry):
    """Coefficient of r^(-Lout-1) in the exterior curl-Poisson field.

    Only source L→outputL+1 survives outside. Multipole equality therefore
    cancels this complete exterior block in delta-minus-compensation metrics.
    """
    value=_currents(current,geometry);angular,radial=geometry['angular'],geometry['radial']
    lookup={int(l):i for i,l in enumerate(radial['degrees'])}
    moments=radial['moments0'][[lookup[int(l)] for l in angular['lm'][:,0]]]
    Q0=np.einsum('in,...bin->...bi',moments,value)
    return -np.einsum('pobi,...bi->...po',angular['curl_plus'],Q0)


def static_transverse_bilinear(left,right,geometry,*,quadrature_tile=32,include_exterior=True):
    """Sesquilinear P/K² form, streaming positive field tiles over quadrature.

    Batch axes of left/right broadcast; it does not materialize a pair matrix.
    Omitting the exterior is appropriate for a signed delta/g difference
    only after per-component ordinary multipoles have been authenticated equal.
    """
    lhs,rhs=_currents(left,geometry),_currents(right,geometry)
    width=int(quadrature_tile)
    if width!=quadrature_tile or width<=0:
        raise ValueError("field quadrature tile must be a positive integer")
    radial=geometry['radial'];r=radial['quadrature_radius']
    result=np.zeros(np.broadcast_shapes(lhs.shape[:-3],rhs.shape[:-3]),dtype=np.complex128)
    weights=radial['quadrature_weights_dr']*r*r
    for first in range(0,len(r),width):
        stop=min(first+width,len(r));tile=slice(first,stop)
        a=curl_poisson_field_tile(lhs,geometry,quadrature_slice=tile)
        b=curl_poisson_field_tile(rhs,geometry,quadrature_slice=tile)
        result+=np.einsum('...qpo,q,...qpo->...',a.conj(),weights[tile],b)
    if include_exterior:
        a,b=exterior_curl_coefficients(lhs,geometry),exterior_curl_coefficients(rhs,geometry)
        l=geometry['angular']['output_lm'][:,0];w=np.zeros(len(l));use=l>0
        w[use]=radial['support_radius']**(1-2*l[use])/(2*l[use]-1)
        result+=np.einsum('...po,o,...po->...',a.conj(),w,b)
    return result


def vector_harmonic_basis(lm):
    """Unitary Cartesian→(J,L,M) vector harmonics from J² and a J− ladder.

    Three-state (m,s) blocks determine highest-M coefficients; lowering
    fixes one common phase across each M multiplet. A fixed overall phase
    per (J,L) is arbitrary and used consistently by the transform and metric.
    No tabulated CG dependency or fitting/current factor enters this geometry.
    """
    labels=np.asarray(lm)
    if labels.ndim!=2 or labels.shape[1]!=2 or not len(labels):
        raise ValueError("vector harmonics require complete canonical LM shells")
    maximum=int(np.max(labels[:,0]))
    if not np.array_equal(labels,_full_shell_labels(maximum)):
        raise ValueError("vector harmonics require complete canonical LM shells")
    nh=len(labels);rows={tuple(v):i for i,v in enumerate(labels)}
    spin={-1:np.asarray([1.,-1j,0])/np.sqrt(2),0:np.asarray([0.,0.,1.]),
           1:np.asarray([-1.,-1j,0])/np.sqrt(2)}
    columns=[];modes=[]
    states=lambda L,M:[(M-s,s) for s in range(-1,2) if abs(M-s)<=L]
    for L in range(maximum+1):
        for J in (range(abs(L-1),L+2) if L else (1,)):
            basis=states(L,J);matrix=np.zeros((len(basis),len(basis)))
            for i,(m,s) in enumerate(basis):
                matrix[i,i]=L*(L+1)+2+2*m*s;target=(m+1,s-1)
                if target in basis:
                    j=basis.index(target)
                    matrix[i,j]=matrix[j,i]=np.sqrt((L*(L+1)-m*(m+1))*(2-s*(s-1)))
            eigen,U=np.linalg.eigh(matrix);index=int(np.argmin(abs(eigen-J*(J+1))))
            if abs(eigen[index]-J*(J+1))>1e-11:
                raise ValueError("unresolved vector total angular momentum")
            vector=U[:,index];vector*=1 if vector[np.argmax(abs(vector))]>=0 else -1
            for M in range(J,-J-1,-1):
                column=np.zeros((3,nh),complex)
                for amplitude,(m,s) in zip(vector,basis):column[:,rows[L,m]]=amplitude*spin[s]
                columns.append(column.ravel());modes.append((J,L,M))
                if M==-J:break
                following=states(L,M-1);lowered=np.zeros(len(following))
                for amplitude,(m,s) in zip(vector,basis):
                    if m>-L:lowered[following.index((m-1,s))]+=amplitude*np.sqrt(L*(L+1)-m*(m-1))
                    if s>-1:lowered[following.index((m,s-1))]+=amplitude*np.sqrt(2-s*(s-1))
                vector=lowered/np.sqrt(J*(J+1)-M*(M-1));basis=following
    order=sorted(range(len(modes)),key=lambda i:(modes[i][0],modes[i][2],modes[i][1]))
    transform=np.asarray(columns)[order].T;modes=np.asarray(modes)[order]
    error=float(np.max(abs(transform.conj().T@transform-np.eye(3*nh))))
    if error>1024*np.finfo(float).eps*(maximum+1):
        raise ValueError("vector harmonics failed unitary closure")
    return dict(transform=transform,modes=modes,unitary_error=error)


def static_transverse_block_factors(geometry,compensation):
    """Positive interior radial blocks, reused for every complete J,M sector.

    Magnetic L=J and electric L=J±1 sectors have at most nr and 2nr
    source coordinates; analytic compensation has 2 and 4 moment coordinates.
    One thin QR of the concatenated source and analytic-compensation field
    maps retains ALL columns, including the structural J=0 longitudinal
    null and finite-L edge sectors. Their common coordinates support smooth
    source versus neutral-residual cross forms without a second radial fit.
    No spectral cutoff or new fit points is introduced.
    """
    angular,radial=geometry['angular'],geometry['radial'];nh=len(angular['lm'])
    if (compensation['support_radius']!=radial['support_radius']
            or not np.array_equal(compensation['degrees'],radial['degrees'])):
        raise ValueError("vector blocks require matching source/compensation geometry")
    basis=vector_harmonic_basis(angular['lm']);U=basis['transform'];modes=basis['modes']
    T=U.reshape(3,nh,-1)
    plus=np.einsum('pobi,bik->pok',angular['curl_plus'],T).reshape(-1,len(modes))
    minus=np.einsum('pobi,bik->pok',angular['curl_minus'],T).reshape(-1,len(modes))
    branches=np.stack((plus,minus),axis=-1)
    A=branches.reshape(branches.shape[0],-1);gram=A.conj().T@A
    repeated=np.repeat(modes,2,axis=0)
    allowed=((repeated[:,0,None]==repeated[None,:,0])
             &(repeated[:,2,None]==repeated[None,:,2])
             &((repeated[:,1,None]%2)==(repeated[None,:,1]%2)))
    forbidden_error=float(np.max(abs(gram*~allowed)))
    gp=compensation_poisson_gradient_maps(compensation,radial['quadrature_radius'])
    weight=np.sqrt(radial['quadrature_weights_dr'])*radial['quadrature_radius']
    sectors=[];covariance_error=0.
    for J in np.unique(modes[:,0]):
        for kind in ('magnetic','electric'):
            selected=(modes[:,0]==J)&(modes[:,2]==J)
            selected &= (modes[:,1]==J) if kind=='magnetic' else (modes[:,1]!=J)
            index=np.flatnonzero(selected)
            if not len(index):continue
            Ls=modes[index,1];C=branches[:,index].reshape(branches.shape[0],-1)
            _,RA=np.linalg.qr(C,mode='reduced');H=C.conj().T@C
            by_M=[]
            for M in range(-int(J),int(J)+1):
                cols=np.asarray([np.flatnonzero((modes==[J,L,M]).all(axis=1))[0] for L in Ls])
                by_M.append(cols);value=branches[:,cols].reshape(branches.shape[0],-1)
                covariance_error=max(covariance_error,float(np.max(abs(value.conj().T@value-H))))
            def field_map(upper,lower,width):
                value=np.zeros((RA.shape[0],len(weight),len(Ls),width),complex)
                for i,L in enumerate(Ls):
                    row=int(np.flatnonzero(radial['degrees']==L)[0])
                    value[:,:,i]=(RA[:,2*i,None,None]*upper[row][None]
                                   +RA[:,2*i+1,None,None]*lower[row][None])
                value*=weight[None,:,None,None]
                return value.reshape(-1,len(Ls)*width)
            density=field_map(radial['plus'],radial['minus'],radial['plus'].shape[-1])
            analytic=field_map(gp['plus'],gp['minus'],2)
            joint=np.linalg.qr(np.concatenate((density,analytic),axis=1),mode='reduced')[1]
            fd,fg=joint[:,:density.shape[1]],joint[:,density.shape[1]:]
            sectors.append(dict(J=int(J),kind=kind,Ls=Ls,indices_by_M=tuple(by_M),
                                delta_factor=fd,compensation_factor=fg))
    tolerance=2048*np.finfo(float).eps*(int(angular['lm'][-1,0])+1)
    if forbidden_error>tolerance or covariance_error>tolerance:
        raise ValueError("Cartesian vector block covariance or parity closure failed")
    return dict(basis=basis,sectors=tuple(sectors),angular_forbidden_error=forbidden_error,
        angular_M_covariance_error=covariance_error,no_cutoff=True,
        retained_columns=sum((2*s['J']+1)*s['delta_factor'].shape[1] for s in sectors),
        retained_compensation_columns=sum((2*s['J']+1)*s['compensation_factor'].shape[1] for s in sectors),
        common_field_coordinates=sum((2*s['J']+1)*s['delta_factor'].shape[0] for s in sectors),
        cross_space='common_full_column_density_and_analytic_compensation',
        factor_bytes=sum(s['delta_factor'].nbytes+s['compensation_factor'].nbytes for s in sectors))
