"""Shared damped poles with exact M0, Mm1, M1 residue equalities.

The extended real Gram is Y Y.T for rows
[H_train; A_train; M0; Mm1; M1], all in the same V-whitened orthonormal
real Hermitian matrix coordinates. H/A use W=H+i*A, as in varpro.py.
Moments are the physical positive-frequency dA moments, in Ry units.
No matrix of spatial dimension n is needed here.
"""

import time

import numpy as np

try:
    from . import varpro
except ImportError:
    import varpro

basis = varpro.basis


def moment_coefficients(poles_ry):
    """Return rows (M0,Mm1,M1) for each Hermitian damped-pair residue.

    Parameters
    ----------
    poles_ry : complex ndarray, shape (p,)
        Positive real frequencies a and widths gamma=-Im(pole)>0, in Ry.

    Returns
    -------
    ndarray, shape (3,p)
        Coefficients with units (1, 1/Ry, Ry). Multiplying physical residue
        R_p gives moments of dA=-Im(W(omega+i0))/pi on omega>0.

    Notes
    -----
    The scalar spectral density is gamma/pi times the difference of the
    Lorentzians centered at a and -a. Its integrals against 1, 1/omega,
    omega are (2/pi)atan(a/gamma), a/(a*a+gamma*gamma), a respectively.
    The inverse moment is also -phi(0)/2, by the even causal dispersion
    relation. In the first moment the two logarithmic tails cancel before
    integration; integrate the difference, never the two divergent terms.
    """
    poles = np.asarray(poles_ry, complex)
    a, gamma = poles.real, -poles.imag
    if poles.ndim != 1 or np.any(a <= 0) or np.any(gamma <= 0):
        raise ValueError('Moment coefficients require causal positive-frequency damped poles')
    return np.vstack((2/np.pi*np.arctan2(a,gamma), a/(a*a+gamma*gamma), a))


def _constraint_design(theta):
    """Dimensionless moment equalities and analytic log-pole derivatives."""
    p = theta.size//2
    a, gamma = np.exp(theta[:p]), np.exp(theta[p:])
    radius2 = a*a+gamma*gamma
    c = moment_coefficients(a-1j*gamma)
    d0 = 2/np.pi*a*gamma/radius2
    derivative_u = np.vstack((d0, a*(gamma*gamma-a*a)/radius2**2, a))
    derivative_v = np.vstack((-d0, -2*a*gamma*gamma/radius2**2, np.zeros(p)))
    return c, derivative_u, derivative_v


def _constrained_eliminate(a, c, data, moments):
    """Enforce C Q=M by SVD nullspace, then use the existing LS owner.

    C+ M is a feasible particular solution and Z spans ker(C). With B=A Z,
    Q=C+ M+Z B+ (D-A C+ M). Both inverses are SVD inverses; no normal
    equations are formed. Rank truncation is refused because it changes
    feasibility or invalidates the exact reduced-Jacobian formula.
    """
    u, s, vh = np.linalg.svd(c, full_matrices=True)
    rank = int(np.sum(s > s[0]*1e-13))
    if rank != 3:
        raise ValueError(f'Moment constraint rank {rank}<3; equalities underidentified')
    inverse_c = (vh[:3].T/s)@u.T
    nullspace = vh[3:].T
    particular = inverse_c@moments
    reduced = a@nullspace
    if nullspace.shape[1]:
        residual, free, inverse_b, sb, rank_b = varpro._eliminate(reduced, data-a@particular)
        if rank_b != nullspace.shape[1]:
            raise ValueError(f'Constrained design rank {rank_b}<{nullspace.shape[1]}; truncated derivative refused')
        coefficients = particular+nullspace@free
    else:
        inverse_b = np.zeros((0,a.shape[0]))
        sb = np.empty(0)
        residual = data-a@particular
        coefficients = particular
    return residual, coefficients, dict(inverse_c=inverse_c, nullspace=nullspace,
               reduced=reduced, inverse_b=inverse_b, constraint_singular=s,
               reduced_singular=sb)


def _residual_jac(theta, z, sqrt_weights, data, moments):
    """Exact generalized variable-projection Jacobian with moving constraints.

    At full constraint/reduced rank, let lambda=C+^T A^T r, B=A Z,
    dQ0=-C+ dC Q, t=dA Q+A dQ0. Differentiating feasibility and the
    constrained least-squares stationarity gives
    dr=-(I-B B+)t-B+^T Z^T(dA^T r-dC^T lambda).
    This avoids differentiating arbitrary SVD nullspace vectors and requires
    neither normal equations nor finite differences. It retains the exact
    transpose-projection term omitted by the Kaufman approximation.
    """
    a, au, av = varpro._design(theta,z)
    a = sqrt_weights[:,None]*a
    c, cu, cv = _constraint_design(theta)
    residual, coefficients, state = _constrained_eliminate(a,c,data,moments)
    ci, nullspace = state['inverse_c'], state['nullspace']
    reduced, bi = state['reduced'], state['inverse_b']
    multiplier = ci.T@(a.T@residual)
    columns = []
    for j in range(theta.size):
        pole = j % coefficients.shape[0]
        da = sqrt_weights*(au[:,pole] if j < coefficients.shape[0] else av[:,pole])
        dc = cu[:,pole] if j < coefficients.shape[0] else cv[:,pole]
        dparticular = -(ci@dc)[:,None]*coefficients[pole]
        t = da[:,None]*coefficients[pole]+a@dparticular
        response = da@residual-dc@multiplier
        correction = nullspace[pole,:,None]*response[None,:]
        derivative = -(t-reduced@(bi@t))-bi.T@correction
        columns.append(derivative.ravel())
    return residual.ravel(), np.stack(columns,axis=1), (coefficients,state)


def _compress_gram(z,weights,gram):
    """Fixed-scale joint compression, normalized by sample loss only."""
    nt = len(z)
    sw = np.sqrt(np.r_[weights,weights])
    moment_scale = np.array([1/varpro.BANDWIDTH_RY,1.,1/varpro.BANDWIDTH_RY**2])
    transform = np.r_[sw,moment_scale]
    transformed = transform[:,None]*((gram+gram.T)/2)*transform[None,:]
    eigenvalues,vectors = np.linalg.eigh(transformed)
    if eigenvalues[-1]<=0 or eigenvalues[0]<-1e-10*eigenvalues[-1]:
        raise ValueError('Extended Gram must be nonzero positive semidefinite')
    sample_norm = np.sqrt(np.trace(transformed[:2*nt,:2*nt]))
    if not sample_norm>0:
        raise ValueError('Zero weighted training norm')
    compressed = vectors*np.sqrt(np.maximum(eigenvalues,0))[None,:]/sample_norm
    return sw,moment_scale,compressed[:2*nt],compressed[2*nt:],eigenvalues


def real_q_gradient_check(z_ry,weights_loss,channel_gram,sketches,p):
    """AAA-seed exact objective-gradient gate including moving moment constraints."""
    z,weights,gram = np.asarray(z_ry,complex),np.asarray(weights_loss,float),np.asarray(channel_gram).real
    sw,_,data,moments,_ = _compress_gram(z,weights,gram)
    seed,initializer = varpro._aaa_initial(z/varpro.BANDWIDTH_RY,sketches,p)
    theta = np.log(np.r_[seed.real,-seed.imag])
    def rank_check(state):
        # The owner refuses either SVD rank loss before returning this state.
        return {'constraint_rank':3,'reduced_design_rank':state[1]['nullspace'].shape[1]}
    receipt = varpro._gradient_receipt(
        lambda t:_residual_jac(t,z/varpro.BANDWIDTH_RY,sw,data,moments),theta,rank_check)
    receipt.update(initializer=initializer,bandwidth_ry=varpro.BANDWIDTH_RY)
    return receipt


def fit(z_ry, weights_loss, channel_gram, sketches, p, initial=None):
    """Fit damped poles with three exact physical dA moment constraints.

    Parameters
    ----------
    z_ry : complex ndarray, shape (Nt,)
        Training points in the upper half-plane, Ry; held rows excluded.
    weights_loss : float ndarray, shape (Nt,)
        Nonnegative loss weights applied here exactly once.
    channel_gram : float ndarray, shape (2*Nt+3,2*Nt+3)
        Unweighted extended Gram of [H;A;M0;Mm1;M1], same whitened coordinates.
        Cross blocks are required. The moments must be parent moments, not
        signed candidate-parent defects from the SHIFT artifact.
    sketches : complex ndarray, shape (Nt,5), or None
        Same sample trace and four quadratic forms, only for VF initialization.
    p : int
        Number of positive poles, at least three for independent equalities.
    initial : complex ndarray, shape (p,), optional
        Warm poles in Ry; every frequency and width remains free.

    Returns
    -------
    dict
        row_map (p,2*Nt+3) acts on original unweighted rows, returning physical
        Hermitian residues. Other diagnostics are small; no Sigma/passivity
        verdict is inferred. Equality enforcement is through the linear solve,
        never an after-fit residue correction.
    """
    start = time.monotonic()
    z = np.asarray(z_ry,complex)
    weights = np.asarray(weights_loss,float)
    raw = np.asarray(channel_gram)
    if np.iscomplexobj(raw) and np.linalg.norm(raw.imag)>1e-12*max(np.linalg.norm(raw),1e-300):
        raise ValueError('Extended Hermitian-channel Gram must be real')
    gram = np.asarray(raw.real,float)
    nt = z.size
    if z.ndim!=1 or not nt or weights.shape!=(nt,) or gram.shape!=(2*nt+3,2*nt+3):
        raise ValueError('Extended Gram/z/weight shape mismatch')
    if not all(np.all(np.isfinite(x)) for x in (z,weights,gram)):
        raise ValueError('Inputs must be finite')
    if np.any(z.imag<=0) or np.any(weights<0) or not np.any(weights>0):
        raise ValueError('Upper-half-plane samples and nonzero nonnegative weights required')
    if p<3 or p-3>2*np.count_nonzero(weights):
        raise ValueError('Need p>=3 and at least p-3 active real data rows')
    if np.linalg.norm(gram-gram.T)>1e-10*max(np.linalg.norm(gram),1e-300):
        raise ValueError('Extended Gram is not symmetric')
    bandwidth = varpro.BANDWIDTH_RY
    scaled_z = z/bandwidth
    # Q=R/b: moments become (M0/b, Mm1, M1/b^2) and C uses dimensionless poles.
    sw,moment_scale,data,moments,eigenvalues = _compress_gram(z,weights,gram)
    if initial is None:
        seed,initializer = varpro._aaa_initial(scaled_z,sketches,p)
    else:
        seed = np.asarray(initial,complex)/bandwidth
        initializer = {'method':'warm start'}
    if seed.shape!=(p,) or np.any(seed.real<=0) or np.any(seed.imag>=0) or not np.all(np.isfinite(seed)):
        raise ValueError('Initial poles must be finite, Re>0, Im<0, shape(p,)')
    theta = np.log(np.r_[seed.real,-seed.imag])
    lower,upper = -25.,20.
    if np.any(theta<=lower) or np.any(theta>=upper):
        raise ValueError('Initializer exceeds reported numerical log bounds')
    cache = {}

    def evaluate(t):
        if 'theta' not in cache or not np.array_equal(t,cache['theta']):
            cache['theta'] = t.copy()
            cache['value'] = _residual_jac(t,scaled_z,sw,data,moments)
        return cache['value']

    result,iteration_receipt = varpro._minimize(evaluate,theta)
    residual,jacobian,(_,state) = evaluate(result.x)
    a = sw[:,None]*varpro._design(result.x,scaled_z)[0]
    zi = state['nullspace']@state['inverse_b']
    row_map = bandwidth*np.hstack((zi*sw[None,:],
                (state['inverse_c']-zi@a@state['inverse_c'])*moment_scale[None,:]))
    poles = bandwidth*(np.exp(result.x[:p])-1j*np.exp(result.x[p:]))
    order = np.argsort(poles.real)
    row_map,poles = row_map[order],poles[order]
    target = np.hstack((np.zeros((3,2*nt)),np.eye(3)))
    equality_map_error = moment_coefficients(poles)@row_map-target
    moment_error2 = np.einsum('ij,jk,ik->i',equality_map_error,gram,equality_map_error)
    moment_norm2 = np.diag(gram)[-3:]
    if np.any(moment_norm2<0) or np.any(moment_error2 < -1e-10*np.maximum(moment_norm2,1e-300)):
        raise ValueError('Materially negative moment residual/reference Gram norm')
    moment_relative = np.sqrt(np.maximum(moment_error2,0)/np.maximum(moment_norm2,1e-300))
    jsv = np.linalg.svd(jacobian,compute_uv=False)
    asv = np.linalg.svd(a,compute_uv=False)
    csv = np.linalg.svd(np.sqrt(weights)[:,None]*basis(z,poles),compute_uv=False)
    bs = state['reduced_singular']
    degeneracy = varpro._degeneracy(row_map,gram,poles)
    eligible = iteration_receipt['success'] and not degeneracy['unresolved_tiny_residue_degeneracy']
    return {'poles_ry':poles,'row_map':row_map,
            'relative_training_error':float(np.linalg.norm(residual)),
            'cond_phi':float(csv[0]/csv[-1]) if p<=nt and csv[-1] else float('inf'),
            'cond_real_design':float(asv[0]/asv[-1]) if p<=2*nt and asv[-1] else float('inf'),
            'cond_constraint':float(state['constraint_singular'][0]/state['constraint_singular'][-1]),
            'cond_reduced_design':float(bs[0]/bs[-1]) if bs.size else 1.,
            'constraint_rank':3,'reduced_design_rank':p-3,
            'jac_sigma_min':float(jsv[-1]) if jacobian.shape[0]>=jacobian.shape[1] else 0.,
            'jac_singular_values':jsv,'jac_sigma_min_certified':True,
            'jac_sigma_min_scope':'Full constraint and reduced ranks; exact moving-constraint residual derivative',
            'jac_normalization':'relative sample loss; log dimensionless poles',
            'jacobian_method':'exact generalized variable projection, including changing constraints; not Kaufman approximation',
            'moment_equality_map_error_max':float(np.max(np.abs(equality_map_error))),
            'moment_equality_map_error_by_row':np.max(np.abs(equality_map_error),axis=1),
            'moment_residual_relative':moment_relative,
            'moment_residual_relative_max':float(np.max(moment_relative)),
            'moment_residual_scope':'Each physical moment Frobenius norm, evaluated from extended Gram; order M0,Mm1,M1',
            'moment_constraints':['M0','Mm1','M1'],
            'moment_row_scales':moment_scale,'bandwidth_ry':bandwidth,
            'gram_negative_roundoff_mass':float(-np.minimum(eigenvalues,0).sum()),
            'gram_positive_rank':int(np.count_nonzero(eigenvalues>0)),
            'initializer':initializer,'nfev':result.nfev,
            'message':result.message,'numerical_log_bounds':[lower,upper],
            'active_bounds':result.active_mask,'wall_seconds':time.monotonic()-start,
            'residue_constraint':'Hermitian real channels with exact parent dA moment equalities',
            'numerical_candidate_eligible':bool(eligible),
            'candidate_scope':'Numerical prerequisites only; unresolved degeneracy or unconverged fit is not a candidate',
            **iteration_receipt,**degeneracy}


def synthetic_check():
    """CPU algebra check of moving equalities, exact derivative, and Ry export."""
    rng = np.random.default_rng(305306)
    z = np.linspace(.05,1.5,24)+.025j
    poles = np.array([.24-.012j,.53-.025j,.87-.05j,1.17-.08j])
    residue = rng.normal(size=(4,7))
    samples = basis(z,poles)@residue
    moments = moment_coefficients(poles)@residue
    y = np.vstack((samples.real,samples.imag,moments))
    weights = 1/(1+z.real**2)
    sw = np.sqrt(np.r_[weights,weights])
    norm = np.linalg.norm(sw[:,None]*y[:48])
    data = sw[:,None]*y[:48]/norm
    m = moments/norm
    theta = np.log(np.r_[poles.real*1.03,-poles.imag*1.07])
    residual,jacobian,_ = _residual_jac(theta,z,sw,data,m)
    eps = 1e-6
    eye = np.eye(theta.size)
    finite = np.stack([(_residual_jac(theta+eps*step,z,sw,data,m)[0]-
                        _residual_jac(theta-eps*step,z,sw,data,m)[0])/(2*eps)
                       for step in eye],axis=1)
    derivative_error = np.linalg.norm(jacobian-finite)/np.linalg.norm(finite)
    fitted = fit(z,weights,y@y.T,samples[:,:5],4,initial=poles*(1.01+.002j))
    actual_residue = fitted['row_map']@y
    equality_error = np.linalg.norm(moment_coefficients(fitted['poles_ry'])@actual_residue-moments)/np.linalg.norm(moments)
    prediction_error = np.linalg.norm(basis(z,fitted['poles_ry'])@actual_residue-samples)/np.linalg.norm(samples)
    assert derivative_error<1e-6,derivative_error
    assert equality_error<1e-10,equality_error
    assert prediction_error<1e-6,prediction_error
    assert fitted['success'] and fitted['accepted_objective_monotonic'],fitted['termination_reason']
    return {'exact_jacobian_relative_error':float(derivative_error),
            'moment_equality_relative_error':float(equality_error),
            'prediction_relative_error':float(prediction_error),
            'moment_equality_map_error_max':fitted['moment_equality_map_error_max'],
            'max_pole_error_ry':float(np.max(np.abs(fitted['poles_ry']-poles))),
            'accepted_iterations':fitted['accepted_iterations'],
            'relative_step_converged':fitted['relative_step_converged']}
