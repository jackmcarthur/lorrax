"""Analytic local pole sensitivity for nondegenerate unconstrained VARPRO.

For fixed sample points, weights, whitening coordinates and pole count,
F(theta,D)=||Q(theta) D||_F^2/2, Q=I-A A+, C=A+ D.
The stationarity equation g(theta,D)=0 gives
dtheta=-H_exact^{-1} (partial_D g)[dD]. No poles are frozen.

The campaign interface consumes only the real channel Gram G=Y Y.T and
dG=dY Y.T+Y dY.T. Y contains unweighted [H_train; A_train] coordinates.
Thus neither the full spatial matrices nor their vectorization is required.
This is a local derivative at a stationary strict minimum of a fixed-rank
unconstrained functional. It is not a moment-constrained sensitivity, a
global SC convergence statement, or a cure for coalesced/vanishing poles.
It differentiates the full data-Gram objective, not a spectral-truncation
algorithm for that Gram. All positive data modes are retained; only negative
roundoff in its PSD factorization is clipped and reported.
"""

import numpy as np

try:
    from . import varpro
except ImportError:
    import varpro


def _second_design(theta,z):
    """Second log-pole derivatives of phi, preserving the real H/A layout."""
    p = len(theta)//2
    a,gamma = np.exp(theta[:p]),np.exp(theta[p:])
    minus = z[:,None]-a+1j*gamma
    plus = z[:,None]+a+1j*gamma
    first_u = a*(1/minus**2+1/plus**2)
    first_v = 1j*gamma*(-1/minus**2+1/plus**2)
    second_uu = first_u+2*a*a*(1/minus**3-1/plus**3)
    second_uv = -2j*a*gamma*(1/minus**3+1/plus**3)
    second_vv = first_v-2*gamma*gamma*(1/minus**3-1/plus**3)
    return tuple(np.vstack((x.real,x.imag)) for x in (second_uu,second_uv,second_vv))


def _stationarity(theta,z,weighted_gram,sqrt_weights,with_hessian=True):
    """Exact gradient/Hessian of half squared residual, via SVD elimination.

    g_i=-<R,A_i C>. For a parameter direction j,
    C_j=-A+ A_j C+(A+ A+^T) A_j^T R and R_j=-A_j C-A C_j.
    Hence H_ij=-<R_j,A_i C>-<R,A_ij C+A_i C_j>.
    This includes the second derivative and residual terms absent from a
    Gauss-Newton Hessian. Products of SVD inverses are used for derivatives;
    no normal-equation matrix is formed or factorized.
    """
    phi,au,av = varpro._design(theta,z)
    a = sqrt_weights[:,None]*phi
    first = sqrt_weights[:,None]*np.hstack((au,av))
    eigenvalues,vectors = np.linalg.eigh(weighted_gram)
    if eigenvalues[-1]<=0 or eigenvalues[0]<-1e-10*eigenvalues[-1]:
        raise ValueError('Sensitivity requires a nonzero positive-semidefinite channel Gram')
    data = vectors*np.sqrt(np.maximum(eigenvalues,0))[None,:]
    residual,c,inverse,singular,rank = varpro._eliminate(a,data)
    p = len(theta)//2
    if rank!=p:
        raise ValueError(f'Sensitivity refuses truncated design rank {rank}<p={p}')
    gradient = np.array([-np.dot(first[:,i]@residual,c[i%p]) for i in range(2*p)])
    state = {'gradient':gradient,'objective':float(np.sum(residual*residual)/2),
             'data_norm_squared':float(np.trace(weighted_gram)),
             'design_rank':rank,'design_condition':float(singular[0]/singular[-1]),
             'inverse':inverse,'design':a,'first_derivatives':first,
             'residual':residual,'coefficients':c}
    state['gram_negative_roundoff_mass'] = float(-np.minimum(eigenvalues,0).sum())
    if not with_hessian:
        return state
    second = tuple(sqrt_weights[:,None]*value for value in _second_design(theta,z))
    hessian = np.empty((2*p,2*p))
    for j in range(2*p):
        pole = j%p
        direction = first[:,j]
        dc = -(inverse@direction)[:,None]*c[pole]
        dc += (inverse@inverse[pole])[:,None]*(direction@residual)[None,:]
        dr = -direction[:,None]*c[pole]-a@dc
        for i in range(2*p):
            ipole = i%p
            value = -np.dot(first[:,i]@dr,c[ipole])-np.dot(first[:,i]@residual,dc[ipole])
            if ipole==pole:
                block = 0 if i<p and j<p else (2 if i>=p and j>=p else 1)
                value -= np.dot(second[block][:,pole]@residual,c[ipole])
            hessian[i,j] = value
    asymmetry = np.linalg.norm(hessian-hessian.T)/max(np.linalg.norm(hessian),1e-300)
    if asymmetry>1e-8:
        raise ValueError(f'Analytic Hessian lost numerical symmetry: {asymmetry}')
    state.update(exact_hessian=(hessian+hessian.T)/2,hessian_relative_asymmetry=float(asymmetry))
    return state


def implicit_direction(exact_hessian,mixed_gradient,relative_eigenvalue_floor=1e-10):
    """Solve the implicit stationarity equation, refusing nonidentifiability.

    Parameters
    ----------
    exact_hessian : float ndarray, shape (2p,2p)
        Exact Hessian of F=||R||^2/2 at the minimum, not a GN approximation.
        A supplied matrix is the caller's mathematical responsibility.
    mixed_gradient : float ndarray, shape (2p,)
        Analytic partial_D gradient derivative in the requested data direction,
        using the same objective normalization as exact_hessian.
    relative_eigenvalue_floor : float
        Refusal threshold on lambda_min/lambda_max. No pseudoinverse,
        regularization, dropping of poles, or frozen parameter is substituted.

    Returns
    -------
    tuple
        dtheta, plus positive-definiteness and linear-solve diagnostics.
    """
    hessian = np.asarray(exact_hessian,float)
    mixed = np.asarray(mixed_gradient,float)
    if hessian.shape!=(len(mixed),len(mixed)) or not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(mixed)):
        raise ValueError('Implicit sensitivity inputs must be finite and square-compatible')
    if np.linalg.norm(hessian-hessian.T)>1e-10*max(np.linalg.norm(hessian),1e-300):
        raise ValueError('Supplied exact Hessian is not symmetric')
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[-1]<=0 or eigenvalues[0]<=relative_eigenvalue_floor*eigenvalues[-1]:
        raise ValueError(f'No identifiable strict minimum: Hessian eigenvalue range {eigenvalues[0]}, {eigenvalues[-1]}')
    direction = np.linalg.solve(hessian,-mixed)
    error = np.linalg.norm(hessian@direction+mixed)/max(np.linalg.norm(mixed),1e-300)
    return direction,{'hessian_min_eigenvalue':float(eigenvalues[0]),
                      'hessian_max_eigenvalue':float(eigenvalues[-1]),
                      'hessian_condition':float(eigenvalues[-1]/eigenvalues[0]),
                      'relative_eigenvalue_floor':relative_eigenvalue_floor,
                      'implicit_solve_relative_residual':float(error)}


def pole_data_direction(theta,z_ry,weights_loss,channel_gram,dchannel_gram,*,
                        exact_hessian=None,stationarity_tolerance=1e-8):
    """Return analytic dtheta and dOmega for a directional sample-data change.

    Parameters
    ----------
    theta : float ndarray, shape (2p,)
        [log(Re(Omega)/b), log(-Im(Omega)/b)] at a converged minimum,
        b=150 eV in Ry, with the current pole ordering retained.
    z_ry : complex ndarray, shape (Nt,)
        Fixed training points, Ry. No held points or moments are included.
    weights_loss : float ndarray, shape (Nt,)
        Fixed nonnegative loss weights, applied here to both Gram arguments.
    channel_gram : float ndarray, shape (2Nt,2Nt)
        Original unweighted Hermitian-channel Y Y.T, ordered H then A.
    dchannel_gram : float ndarray, shape (2Nt,2Nt)
        Directional derivative dY Y.T+Y dY.T per unit perturbation, not a
        finite perturbed Gram itself. All entries share fixed whitening axes.
    exact_hessian : float ndarray, shape (2p,2p), optional
        Exact Hessian in the raw F=||sqrt(w)R||^2/2 convention. If omitted,
        it is built analytically including residual and second-design terms.
    stationarity_tolerance : float
        Refusal threshold for ||gradient||/||weighted data||^2.

    Returns
    -------
    dict
        Directional dtheta and complex dOmega_ry, exact Hessian, mixed
        gradient, and local rank/identifiability diagnostics. This operation
        does not optimize or modify poles. It differentiates their optimum.
    """
    theta = np.asarray(theta,float)
    z = np.asarray(z_ry,complex)
    weights = np.asarray(weights_loss,float)
    gram = np.asarray(channel_gram)
    delta = np.asarray(dchannel_gram)
    if theta.ndim!=1 or not len(theta) or len(theta)%2 or z.ndim!=1 or weights.shape!=z.shape:
        raise ValueError('Sensitivity parameter/sample shapes invalid')
    if gram.shape!=(2*len(z),2*len(z)) or delta.shape!=gram.shape:
        raise ValueError('Sensitivity requires two matching real channel Grams')
    for value in (theta,z,weights,gram,delta):
        if not np.all(np.isfinite(value)):
            raise ValueError('Sensitivity inputs must be finite')
    for value in (gram,delta):
        if np.iscomplexobj(value) and np.linalg.norm(value.imag)>1e-12*max(np.linalg.norm(value),1e-300):
            raise ValueError('Sensitivity channel Grams must be real')
        if np.linalg.norm(value.real-value.real.T)>1e-10*max(np.linalg.norm(value.real),1e-300):
            raise ValueError('Sensitivity Gram or its directional derivative is not symmetric')
    if np.any(z.imag<=0) or np.any(weights<0) or not np.any(weights>0):
        raise ValueError('Upper-half-plane points and nonzero nonnegative weights required')
    sw = np.sqrt(np.r_[weights,weights])
    weighted = sw[:,None]*gram.real*sw[None,:]
    dweighted = sw[:,None]*delta.real*sw[None,:]
    state = _stationarity(theta,z/varpro.BANDWIDTH_RY,weighted,sw,with_hessian=exact_hessian is None)
    relative_gradient = np.linalg.norm(state['gradient'])/max(state['data_norm_squared'],1e-300)
    if relative_gradient>stationarity_tolerance:
        raise ValueError(f'Pole point is not stationary: normalized gradient {relative_gradient}')
    inverse,a,first = state['inverse'],state['design'],state['first_derivatives']
    p = len(theta)//2
    # g_j=-tr(G Q A_j A+), hence dg_j=-A+[pole_j] dG Q a'_j.
    mixed = np.array([-inverse[j%p]@dweighted@(first[:,j]-a@(inverse@first[:,j]))
                      for j in range(2*p)])
    hessian = state['exact_hessian'] if exact_hessian is None else np.asarray(exact_hessian,float)
    direction,receipt = implicit_direction(hessian,mixed)
    poles = varpro.BANDWIDTH_RY*(np.exp(theta[:p])-1j*np.exp(theta[p:]))
    pole_direction = poles.real*direction[:p]+1j*poles.imag*direction[p:]
    return {'dtheta':direction,'dpoles_ry':pole_direction,'exact_hessian':hessian,
            'mixed_gradient':mixed,'gradient':state['gradient'],
            'normalized_gradient':float(relative_gradient),'stationarity_tolerance':stationarity_tolerance,
            'design_rank':state['design_rank'],'design_condition':state['design_condition'],
            'gram_negative_roundoff_mass':state['gram_negative_roundoff_mass'],
            'hessian_source':'analytic residual/design second derivatives' if exact_hessian is None else 'caller-supplied exact Hessian',
            'objective_convention':'one half raw weighted squared Frobenius residual; no data-norm division',
            'scope':'local fixed-grid/weights/whitening/p full-rank unconstrained strict minimum',**receipt}


def synthetic_check():
    """Independent small CPU finite-perturbation and Hessian checks with noise."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(30609)
    z = np.linspace(.05,1.3,30)+.035j
    poles = np.array([.31-.04j,.87-.065j])
    residue = rng.normal(size=(2,5))
    values = varpro.basis(z,poles)@residue
    y = np.vstack((values.real,values.imag))
    # Nonzero residual makes the exact-Hessian test distinct from a GN test.
    y += 1e-3*rng.normal(size=y.shape)
    dy = rng.normal(size=y.shape)
    weights = 1/(1+z.real*z.real)
    sw = np.sqrt(np.r_[weights,weights])
    theta0 = np.log(np.r_[poles.real/varpro.BANDWIDTH_RY,-poles.imag/varpro.BANDWIDTH_RY])

    def optimize(data,initial):
        def evaluate(t):
            return varpro._residual_jac(t,z/varpro.BANDWIDTH_RY,sw,sw[:,None]*data,exact=True)
        result = least_squares(lambda t:evaluate(t)[0],initial,jac=lambda t:evaluate(t)[1],
                               max_nfev=100,ftol=1e-14,xtol=1e-14,gtol=1e-12)
        return result.x

    theta = optimize(y,theta0)
    gram = y@y.T
    derivative_gram = dy@y.T+y@dy.T
    response = pole_data_direction(theta,z,weights,gram,derivative_gram)
    weighted = sw[:,None]*gram*sw[None,:]
    h = 1e-5
    finite_hessian = np.stack([(_stationarity(theta+h*step,z/varpro.BANDWIDTH_RY,weighted,sw,False)['gradient']-
                                _stationarity(theta-h*step,z/varpro.BANDWIDTH_RY,weighted,sw,False)['gradient'])/(2*h)
                               for step in np.eye(len(theta))],axis=1)
    hessian_error = np.linalg.norm(response['exact_hessian']-finite_hessian)/np.linalg.norm(finite_hessian)
    checks = []
    for epsilon in (1e-4,3e-5):
        plus = optimize(y+epsilon*dy,theta)
        minus = optimize(y-epsilon*dy,theta)
        finite = (plus-minus)/(2*epsilon)
        error = np.linalg.norm(finite-response['dtheta'])/np.linalg.norm(response['dtheta'])
        checks.append({'epsilon':epsilon,'relative_direction_error':float(error)})
    assert hessian_error<1e-6,hessian_error
    assert all(item['relative_direction_error']<1e-4 for item in checks),checks
    refused = False
    try:
        implicit_direction(np.diag([1.,0.]),np.ones(2))
    except ValueError:
        refused = True
    assert refused,'Singular Hessian was not refused'
    return {'analytic_hessian_relative_error':float(hessian_error),'finite_perturbations':checks,
            'normalized_gradient':response['normalized_gradient'],
            'hessian_condition':response['hessian_condition'],'singular_hessian_refused':refused}
