"""Stopping diagnostic for actual analytic two-moment currents.

No atomic reference, interpolation, normal-equation factor or GW kernel is
introduced here. Inputs must be the actual provider compensation and solved
moments, followed by actual typed current-Sigma covectors/occupations.
"""
from math import comb
import numpy as np


def laplacian3_positive_map(compensation):
    """Exact positive polynomial GL map for ||Delta^3 g_face||^2.

    The compensation owner has zero-exterior profiles with power>=6.
    Their first five normal derivatives vanish at R, so the weak sixth
    derivatives contain no surface distributions. The angular Y_LM norm is1.
    """
    p=int(compensation['power']);R=float(compensation['support_radius'])
    degrees=np.asarray(compensation['degrees'],int)
    dual=np.asarray(compensation['profile_coefficients'],float)
    if p<6 or R<=0 or not np.isfinite(R) or dual.shape!=(len(degrees),2,2):
        raise ValueError('Tail stopping requires actual C5 compact Q0/Q2 duals')
    order=int(degrees.max())+2*(p+1)+2
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2
    maps=[];absolute_maps=[]
    for row,l in enumerate(degrees):
        raw=np.zeros((2,order));raw_abs=np.zeros_like(raw)
        for j in range(2):
            for t in range(p+1):
                n=t+j
                if n<3:continue
                multiplier=comb(p,t)*(-1)**t
                for h in range(3):multiplier*=2*(n-h)*(2*l+2*(n-h)+1)
                term=multiplier*x**(l+2*n-6)/R**(l+9)
                raw[j]+=term;raw_abs[j]+=abs(term)
        # sqrt(r^2 dr) applies the positive physical radial norm.
        weight=np.sqrt(R**3*w*x*x)[None]
        maps.append((dual[row].T@raw)*weight)
        absolute_maps.append((abs(dual[row]).T@raw_abs)*weight)
    result=np.asarray(maps)
    if not np.isfinite(result).all():raise ValueError('Unresolved analytic compensation norm map')
    return dict(degrees=degrees,map=result,absolute_map=np.asarray(absolute_maps),
                rounding_relative_envelope=4096*np.finfo(float).eps,
                support_radius=R,power=p,
                GL_order=order,polynomial_degree_max=2*int(degrees.max())+4*p-6,
                units='Delta^3 of actual compensation face; radial integral r^2 dr')


def per_centroid_tail_norm_upper(moments,lm,positive_map,*,fft_points,volume,cutoff_Ry,
                                 array_api=np):
    """Tail Fourier l2 upper bound, without low-G subtraction or anotherGscan.

    moments axes (...,current,mu,atom,LM,2). This consumes the actual solved
    current moments, not a reinterpolated pseudo-current or charge factor.
    The provider should tile mu to bound temporary memory.
    """
    xp=array_api;labels=np.asarray(lm,int);rows=np.searchsorted(positive_map['degrees'],labels[:,0])
    if not np.array_equal(positive_map['degrees'][rows],labels[:,0]):raise ValueError('Moment LM map differs')
    value=xp.asarray(moments);L=xp.asarray(positive_map['map'][rows])
    if value.shape[-1]!=2 or value.shape[-2]!=len(labels):raise ValueError('Actual Q0/Q2/LM axes required')
    C=float(cutoff_Ry);N=int(fft_points);V=float(volume)
    if C<=0 or N<1 or V<=0:raise ValueError('Actual source/body normalization required')
    field=xp.einsum('...ald,lds->...als',value,L)
    error=xp.einsum('...ald,lds->...als',xp.abs(value),xp.asarray(positive_map['absolute_map'][rows]))*positive_map['rounding_relative_envelope']
    norm=xp.sum((xp.abs(field)+error)**2,axis=(-3,-2,-1))
    return norm*(N*N/V)/(C**6)


def occupied_exchange_entry_bound(covectors,tail_norm_upper,weights,*,prefactor_Ry,cutoff_Ry,
                                  array_api=np):
    """Cauchy bound for every receiving matrix entry in one actual q block.

    Covectors axes (receiving,occupied_transition,current,mu), tail norm
    (current,mu), actual nonnegative occupation/BZ weights(transition).
    Caller owns phases, current basis, q join and endpoint normalization.
    This bounds bare current Sigma_X only, not nonlinear screened W/Sigma_c.
    """
    xp=array_api;F=xp.asarray(covectors);R=xp.asarray(tail_norm_upper);w=xp.asarray(weights)
    if F.ndim!=4 or R.shape!=F.shape[-2:] or w.shape!=(F.shape[1],):
        raise ValueError('Actual matched transition/current/centroid axes required')
    if xp is np and (not np.isfinite(F).all() or not np.isfinite(R).all()
                     or not np.isfinite(w).all() or np.any(R<0) or np.any(w<0)):
        raise ValueError('Actual tail bounds and occupied weights must be finite nonnegative')
    # Absolute values make this triangle bound independent of a bra-versus-
    # ket storage convention; no cancellation is used for admission.
    amplitudes=xp.sum(xp.abs(F)*xp.sqrt(xp.maximum(R,0))[None,None],axis=-1)
    diagonal=xp.sum(w[None,:]*xp.sum(amplitudes**2,axis=-1),axis=1)
    return (2*abs(float(prefactor_Ry))/float(cutoff_Ry))*xp.sqrt(diagonal[:,None]*diagonal[None,:])


def make_sobolev_norm_kernel(mesh, *, mu, atoms, lm, positive_map, fft_points, volume,
                              centroid_tile=128):
    """Bounded existing-moment observer; no coefficient or operator mutation."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax import shard_map
    labels=np.asarray(lm,int);nh=len(labels)
    rows=np.searchsorted(positive_map['degrees'],labels[:,0])
    maps=positive_map['map'][rows];absolute=positive_map['absolute_map'][rows]
    mt=int(centroid_tile);mup=((int(mu)+mt-1)//mt)*mt
    eps=float(positive_map['rounding_relative_envelope']);scale=int(fft_points)**2/float(volume)
    def observe(moments,L,A):
        outputs=[]
        for moment in moments:
            zero=jnp.zeros((moment.shape[0],mup),jnp.float64)
            def panel(out,step):
                first=step*mt;ids=first+jnp.arange(mt);real=ids<int(mu)
                values=moment[:,jnp.minimum(ids,int(mu)-1)].reshape(moment.shape[0],mt,atoms,nh,2)
                field=jnp.einsum('bmaid,ids->bmais',values,L)
                error=eps*jnp.einsum('bmaid,ids->bmais',jnp.abs(values),A)
                norm=jnp.sum((jnp.abs(field)+error)**2,axis=(-3,-2,-1))*scale
                norm=jnp.where(real[None],norm,0.)
                return jax.lax.dynamic_update_slice(out,norm,(0,first)),None
            result=jax.lax.scan(panel,zero,jnp.arange(mup//mt),unroll=1)[0]
            outputs.append(result[:,:int(mu)])
        return tuple(outputs)
    qs=P(('x','y'),None,None)
    kernel=jax.jit(shard_map(observe,mesh=mesh,in_specs=((qs,)*3,P(),P()),
                              out_specs=(P(('x','y'),None),)*3,check_vma=False))
    return kernel,maps,absolute
