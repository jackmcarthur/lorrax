"""Small-Gram variable projection with Hermitian shared damped residues.

All public frequencies use Ry. Write samples W_i=H_i+i*A_i and expand each
Hermitian H_i,A_i in an orthonormal real Hermitian matrix basis. Stack real
coordinates Y=[H; A], in that order; channel_gram=Y@Y.T, including cross
blocks Tr(H_i A_j). Re(W W†) alone loses the Hermitian-residue constraint.
Loss weights (not amplitude weights) are applied exactly once, inside fit.
"""

import time
import hashlib
import inspect
from pathlib import Path
import warnings

import numpy as np
import scipy
from scipy.optimize import least_squares

EV = 13.605693122994
BANDWIDTH_RY = 150.0/EV


def basis(z_ry, poles_ry):
    """Return conjugate-paired causal phi(z), shape (Nz,p), in 1/Ry."""
    z = np.asarray(z_ry, complex)[:, None]
    pole = np.asarray(poles_ry, complex)[None, :]
    return 1 / (z - pole) - 1 / (z + pole.conj())


def _design(theta, z):
    """Dimensionless real channel design and analytic log-pole derivatives."""
    p = theta.size // 2
    a, gamma = np.exp(theta[:p]), np.exp(theta[p:])
    minus = z[:, None] - a + 1j * gamma
    plus = z[:, None] + a + 1j * gamma
    phi = 1 / minus - 1 / plus
    derivative_u = a * (1 / minus**2 + 1 / plus**2)
    derivative_v = 1j * gamma * (-1 / minus**2 + 1 / plus**2)
    return tuple(np.vstack((x.real, x.imag)) for x in (phi, derivative_u, derivative_v))


def _eliminate(design, data):
    """Eliminate linear residues by SVD, never normal equations."""
    u, s, vh = np.linalg.svd(design, full_matrices=False)
    keep = s > s[0] * 1e-13
    inverse = (vh[keep].T / s[keep]) @ u[:, keep].T
    coefficients = inverse @ data
    # Apply the orthogonal projector itself: A(A+ D) amplifies cancellation
    # by cond(A), even though the subspace residual is well conditioned.
    residual = data - u[:, keep] @ (u[:, keep].T @ data)
    return residual, coefficients, inverse, s, int(keep.sum())


def _residual_jac(theta, z, sqrt_weights, compressed, exact=False):
    """Kaufman Jacobian; exact adds the transpose-projection derivative.

    For R=(I-AA+)B, dR=-(I-AA+)dA C-A+^T dA^T R. Kaufman
    omits the second term, which vanishes in the objective gradient.
    """
    a, derivative_u, derivative_v = _design(theta, z)
    a = sqrt_weights[:, None] * a
    r, c, inverse, singular, rank = _eliminate(a, compressed)
    if rank == a.shape[1]:
        orthogonal = np.linalg.qr(a, mode="reduced")[0]
    else:
        orthogonal = np.linalg.svd(a, full_matrices=False)[0][:, :rank]
    columns = []
    for parameter, derivative in enumerate(np.hstack((derivative_u, derivative_v)).T):
        pole = parameter % c.shape[0]
        dcol = sqrt_weights * derivative
        perpendicular = dcol - orthogonal @ (orthogonal.T @ dcol)
        dr = -perpendicular[:, None] * c[pole]
        if exact:
            dr -= inverse[pole, :, None] * (dcol @ r)[None, :]
        columns.append(dr.ravel())
    return r.ravel(), np.stack(columns, axis=1), (inverse, singular, rank)


def _vf_initial(z, sketches, p):
    """Multi-response VF only initializes; final fit is variable projection.

    Relocation solves y=sum c_j/(z-a_j)-y sum d_j/(z-a_j), with
    shared denominator poles eig(diag(a)-1 d^T). Limit initializer order
    to an overdetermined solve; complete missing seeds from the sample grid.
    """
    positive = np.abs(z.real[np.abs(z.real) > 1e-8])
    lo = max(float(positive.min()) if positive.size else 0.01, 1e-5)
    hi = max(float(np.max(np.abs(z.real))), lo * 10)
    width = max(float(np.min(z.imag)) / 2, 1e-4)
    seed = np.geomspace(lo, hi, p) - 1j * width
    if sketches is None:
        return seed, {"method": "sample-grid fallback", "reason": "no sketches supplied"}
    y = np.asarray(sketches, complex)
    if y.ndim != 2 or y.shape[0] != z.size:
        raise ValueError("sketches must have shape (Nz, nsketch)")
    norms = np.linalg.norm(y, axis=0)
    y = y[:, norms > 0] / norms[norms > 0]
    if not y.shape[1]:
        return seed, {"method": "sample-grid fallback", "reason": "zero sketches"}
    order = min(p, max(1, (z.size - 2) // 3))
    positive_seed = np.geomspace(lo, hi, order) - 1j * width
    poles = np.r_[positive_seed, -positive_seed.conj()]
    count = poles.size
    for _ in range(6):
        rational = 1 / (z[:, None] - poles)
        matrix = np.zeros((z.size * y.shape[1], count * (y.shape[1] + 1)), complex)
        for channel in range(y.shape[1]):
            rows = slice(channel * z.size, (channel + 1) * z.size)
            matrix[rows, channel * count:(channel + 1) * count] = rational
            matrix[rows, -count:] = -y[:, channel, None] * rational
        solution = np.linalg.lstsq(matrix, y.T.ravel(), rcond=1e-12)[0]
        poles = np.linalg.eigvals(np.diag(poles) - np.ones((count, 1)) * solution[-count:][None, :])
        poles = poles.real - 1j * np.maximum(np.abs(poles.imag), 1e-6)
    accepted = poles[(poles.real > 1e-6) & (poles.real < 10 * hi)]
    accepted = accepted[np.argsort(accepted.real)]
    # Causal projection is only of seeds, never of fitted residues or poles.
    if accepted.size:
        slots = np.linspace(0, p - 1, min(p, accepted.size)).round().astype(int)
        seed[slots] = accepted[:slots.size]
    return seed, {"method": "multi-response VF plus sample-grid completion", "vf_order": count,
                  "vf_positive_seeds": int(accepted.size), "iterations": 6}


def _aaa_initial(z, sketches, p):
    """AAA trace seeds with explicit completion to a fixed positive-pole count.

    Causal symmetry supplies (-conj(z),conj(trace)); these are transformed
    copies, not additional sampled matrices. AAA cleanup is disabled. Only
    initializer locations are reflected into the causal quadrant; fitted
    poles and residues are never projected or dropped.
    """
    try:
        from scipy.interpolate import AAA
    except ImportError as exc:
        raise RuntimeError(f'scipy.interpolate.AAA unavailable in SciPy {scipy.__version__}; no VF fallback') from exc
    y = np.asarray(sketches,complex) if sketches is not None else np.empty((0,0))
    if y.ndim!=2 or y.shape[0]!=len(z) or y.shape[1]<1 or not np.all(np.isfinite(y)):
        raise ValueError('AAA requires finite trace sketches for every training row')
    source = Path(inspect.getsourcefile(AAA))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    nodes = np.r_[z,-np.conj(z)]
    values = np.r_[y[:,0],np.conj(y[:,0])]
    nodes,unique = np.unique(nodes,return_index=True)
    values = values[unique]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        approx = AAA(nodes,values,max_terms=min(2*p+1,len(nodes)),rtol=1e-12,clean_up=False)
    raw = np.asarray(approx.poles(),complex)
    keep = np.isfinite(raw)&(raw.real>1e-8)
    positive = raw[keep]
    positive = positive[np.argsort(positive.real)]
    if len(positive)>p:
        ids = np.linspace(0,len(positive)-1,p).round().astype(int)
        positive = positive[ids]
    seeds = list(positive.real-1j*np.maximum(np.abs(positive.imag),1e-6))
    real_nodes = np.abs(np.asarray(z).real)
    nonzero = real_nodes[real_nodes>1e-8]
    lo = max(float(nonzero.min())/4 if len(nonzero) else 1e-4,1e-6)
    hi = max(float(real_nodes.max()),lo*10)
    candidates = np.geomspace(lo,hi,max(8*p,32))
    completion = []
    while len(seeds)<p:
        if seeds:
            distance = np.min(np.abs(np.log(candidates[:,None])-np.log(np.real(seeds))[None,:]),axis=1)
            index = int(np.argmax(distance))
        else:
            index = len(candidates)//2
        seed = candidates[index]-1j*max(float(np.min(np.imag(z)))/2,1e-6)
        seeds.append(seed); completion.append(seed)
        candidates = np.delete(candidates,index)
    seed = np.asarray(seeds)[np.argsort(np.real(seeds))]
    design = np.vstack((basis(z,seed).real,basis(z,seed).imag))
    sketch_rows = np.vstack((y.real,y.imag))
    sketch_residual,_,_,_,sketch_rank = _eliminate(design,sketch_rows)
    return seed, {'method':'AAA trace with conjugate symmetry and fixed-count completion',
        'scipy_version':scipy.__version__,'aaa_source':str(source),'aaa_source_sha256':source_sha,
        'aaa_raw_poles':int(len(raw)),'aaa_positive_poles':int(np.count_nonzero(keep)),
        'aaa_selected_positive_poles':int(len(positive)),'completion_count':len(completion),
        'completion_poles_dimensionless':[[v.real,v.imag] for v in completion],
        'causal_seed_reflections':int(np.sum(positive.imag>=0)),
        'aaa_cleanup':False,'aaa_warnings':[str(item.message) for item in captured],
        'sketch_seed_relative_errors':(np.linalg.norm(sketch_residual,axis=0)/np.maximum(np.linalg.norm(sketch_rows,axis=0),1e-300)).tolist(),
        'sketch_seed_design_rank':sketch_rank,'vf_used':False}


def _minimize(evaluate, theta):
    """One accepted-iteration trace and strict relative-step convergence owner."""
    if 'callback' not in inspect.signature(least_squares).parameters:
        raise RuntimeError(f'SciPy {scipy.__version__} lacks least_squares callback; accepted-step audit unavailable')
    previous = theta.copy()
    initial_residual = evaluate(theta)[0]
    trace = [{'iteration':0,'objective':float(initial_residual@initial_residual),
              'relative_residual':float(np.linalg.norm(initial_residual)),'relative_step':None}]
    converged = False

    def callback(intermediate_result):
        nonlocal previous,converged
        current = intermediate_result.x
        delta = np.linalg.norm(current-previous)
        if delta==0:
            return
        step = float(delta/max(1.,np.linalg.norm(previous)))
        residual = evaluate(current)[0]
        trace.append({'iteration':len(trace),'objective':float(residual@residual),
                      'relative_residual':float(np.linalg.norm(residual)),'relative_step':step})
        previous = current.copy()
        if step<1e-8:
            converged = True
            raise StopIteration

    result = least_squares(lambda t:evaluate(t)[0],theta,jac=lambda t:evaluate(t)[1],
        bounds=(-25.,20.),method='trf',max_nfev=10000,ftol=None,gtol=None,xtol=1e-14,
        callback=callback)
    objectives = np.array([item['objective'] for item in trace])
    monotonic = bool(np.all(np.diff(objectives)<=1e-12*np.maximum(objectives[:-1],1e-300)+1e-15))
    receipt = {'accepted_iteration_trace':trace,'accepted_iterations':len(trace)-1,
        'accepted_objective_monotonic':monotonic,'relative_step_converged':bool(converged),
        'success':bool(converged and monotonic),'optimizer_native_success':bool(result.success),
        'optimizer_status':int(result.status),'termination_reason':('relative accepted log-parameter step < 1e-8'
            if converged else f'Not relatively step-converged: {result.message}'),
        'iteration_budget_nfev':10000,'callback_available':True,'scipy_version':scipy.__version__}
    return result,receipt


def _degeneracy(row_map, gram, poles):
    """Diagnose tiny residues without changing the fixed-p functional."""
    norm2 = np.einsum('ij,jk,ik->i',row_map,gram,row_map)
    norms = np.sqrt(np.maximum(norm2,0))
    relative = norms/max(float(norms.max()),1e-300)
    tiny = np.flatnonzero(relative<1e-8)
    return {'residue_coordinate_norms':norms,'residue_norm_relative':relative,
            'tiny_residue_relative_threshold':1e-8,'tiny_residue_poles':tiny,
            'unresolved_tiny_residue_degeneracy':bool(tiny.size),
            'postfit_poles_dropped':0,'pole_count_fixed':len(poles)}


def _gradient_receipt(evaluate, theta, rank_check):
    """Finite-difference objective gradient gate, independent of residual Jacobian."""
    residual,jacobian,state = evaluate(theta)
    ranks = rank_check(state)
    analytic = 2*jacobian.T@residual
    checks = []
    for step in (1e-5,1e-6):
        finite = []
        for direction in np.eye(len(theta)):
            plus = evaluate(theta+step*direction);minus = evaluate(theta-step*direction)
            rank_check(plus[2]);rank_check(minus[2])
            finite.append((plus[0]@plus[0]-minus[0]@minus[0])/(2*step))
        finite = np.asarray(finite)
        error = np.linalg.norm(finite-analytic)/max(np.linalg.norm(finite),np.linalg.norm(analytic),1e-300)
        checks.append({'log_parameter_step':step,'relative_gradient_error':float(error)})
    passed = min(item['relative_gradient_error'] for item in checks)<1e-6
    receipt = {'passed':bool(passed),'threshold_relative':1e-6,'checks':checks,'ranks':ranks,
               'objective':float(residual@residual),'analytic_gradient_norm':float(np.linalg.norm(analytic))}
    if not passed:
        raise ValueError(f'Real-q objective-gradient gate failed: {receipt}')
    return receipt


def _compress_gram(z, weights, gram):
    """Fixed 150-eV scale and weighted real-channel Gram compression."""
    sw = np.sqrt(np.r_[weights,weights])
    weighted = sw[:,None]*((gram+gram.T)/2)*sw[None,:]
    eigenvalues,vectors = np.linalg.eigh(weighted)
    if eigenvalues[-1]<=0 or eigenvalues[0]<-1e-10*eigenvalues[-1]:
        raise ValueError('channel_gram must be nonzero positive semidefinite')
    keep = eigenvalues>eigenvalues[-1]*1e-14
    norm = np.sqrt(np.maximum(eigenvalues,0).sum())
    compressed = vectors[:,keep]*(np.sqrt(eigenvalues[keep])/norm)
    return sw,compressed,int(keep.sum()),float(np.maximum(eigenvalues[~keep],0).sum()/norm**2)


def real_q_gradient_check(z_ry, weights_loss, channel_gram, sketches, p):
    """AAA-seed real-q finite-difference gate; refuse truncated design ranks."""
    z,weights,gram = np.asarray(z_ry,complex),np.asarray(weights_loss,float),np.asarray(channel_gram).real
    sw,compressed,_,_ = _compress_gram(z,weights,gram)
    seed,initializer = _aaa_initial(z/BANDWIDTH_RY,sketches,p)
    theta = np.log(np.r_[seed.real,-seed.imag])
    def rank_check(state):
        if state[2]!=p:
            raise ValueError(f'Real-q gradient gate uncertified: design rank {state[2]} < p={p}')
        return {'design_rank':state[2]}
    receipt = _gradient_receipt(lambda t:_residual_jac(t,z/BANDWIDTH_RY,sw,compressed,exact=True),theta,rank_check)
    receipt.update(initializer=initializer,bandwidth_ry=BANDWIDTH_RY)
    return receipt


def fit(z_ry, weights_loss, channel_gram, sketches, p, initial=None):
    """Fit moving damped poles with real Hermitian-channel residue elimination.

    Parameters
    ----------
    z_ry : complex ndarray, shape (Nz,)
        Upper-half-plane training points in Ry. Held rows must be excluded.
    weights_loss : float ndarray, shape (Nz,)
        Nonnegative quadrature times frequency times line loss weights.
    channel_gram : float ndarray, shape (2*Nz, 2*Nz)
        Unweighted V-whitened real channel Gram; see module docstring.
    sketches : complex ndarray, shape (Nz, 5), or None
        Trace and four quadratic forms of the same samples, for VF seeds only.
    p : int
        Distinct positive frequencies, not residue rank K.
    initial : complex ndarray, shape (p,), optional
        Warm-start poles in Ry, with positive real and negative imaginary parts.

    Returns
    -------
    dict
        row_map (p,2*Nz) maps original unweighted Y to Hermitian residue
        coordinates in Ry units: W(z)=basis(z,poles_ry)@(row_map@Y).
        Diagnostics concern only this fit, not passivity or Sigma accuracy.
    """
    start = time.monotonic()
    z = np.asarray(z_ry, complex)
    weights = np.asarray(weights_loss, float)
    raw = np.asarray(channel_gram)
    if np.iscomplexobj(raw) and np.max(np.abs(raw.imag)) > 1e-12 * max(np.linalg.norm(raw), 1e-300):
        raise ValueError("channel_gram must be real Hermitian-channel Gram")
    gram = np.asarray(raw.real, float)
    if z.ndim != 1 or weights.shape != z.shape or gram.shape != (2*z.size, 2*z.size):
        raise ValueError("inconsistent z, weights, or channel_gram shapes")
    if np.any(z.imag <= 0) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("upper-half-plane samples and nonnegative nonzero weights required")
    if p < 1 or p > 2 * np.count_nonzero(weights):
        raise ValueError("p exceeds the number of active real channel rows")
    bandwidth = BANDWIDTH_RY
    scaled_z = z / bandwidth
    if np.linalg.norm(gram - gram.T) > 1e-10 * max(np.linalg.norm(gram), 1e-300):
        raise ValueError("channel_gram is not symmetric")
    sw,compressed,gram_rank,discarded_fraction = _compress_gram(z,weights,gram)
    if initial is None:
        seed, initializer = _aaa_initial(scaled_z, sketches, p)
    else:
        seed = np.asarray(initial, complex) / bandwidth
        initializer = {"method": "warm start"}
    if seed.shape != (p,) or np.any(seed.real <= 0) or np.any(seed.imag >= 0):
        raise ValueError("initial poles require shape (p,), Re>0, Im<0")
    theta = np.log(np.r_[seed.real, -seed.imag])
    # Numerical guard bounds are exposed in every result, never hidden physics.
    lower, upper = -25.0, 20.0
    if np.any(theta <= lower) or np.any(theta >= upper):
        raise ValueError("initializer exceeds reported numerical log bounds")
    cache = {}

    def evaluate(t):
        if "theta" not in cache or not np.array_equal(t, cache["theta"]):
            cache["theta"] = t.copy()
            cache["value"] = _residual_jac(t, scaled_z, sw, compressed)
        return cache["value"]

    result,iteration_receipt = _minimize(evaluate,theta)
    residual, jacobian, (inverse, singular, rank) = _residual_jac(
        result.x, scaled_z, sw, compressed, exact=True)
    jac_singular = np.linalg.svd(jacobian, compute_uv=False)
    poles = bandwidth * (np.exp(result.x[:p]) - 1j * np.exp(result.x[p:]))
    ordering = np.argsort(poles.real)
    complex_singular = np.linalg.svd(np.sqrt(weights)[:, None] * basis(z, poles), compute_uv=False)
    complex_condition = (float(complex_singular[0] / complex_singular[-1])
                         if p <= z.size and complex_singular[-1] else float("inf"))
    row_map = bandwidth * inverse[ordering] * sw[None,:]
    degeneracy = _degeneracy(row_map,gram,poles[ordering])
    eligible = iteration_receipt['success'] and rank==p and not degeneracy['unresolved_tiny_residue_degeneracy']
    return {"poles_ry": poles[ordering], "row_map": row_map,
            "relative_training_error": float(np.linalg.norm(residual)),
            "cond_phi": complex_condition,
            "cond_real_design": float(singular[0] / singular[-1]) if singular[-1] else float("inf"),
            "design_rank": rank, "jac_sigma_min": (float(jac_singular[-1])
                                                     if jacobian.shape[0] >= jacobian.shape[1] else 0.0),
            "jac_sigma_min_certified": bool(rank == p),
            "jac_sigma_min_scope": ("Full-column-rank design: constant-rank reduced-residual derivative"
                                    if rank == p else
                                    "Uncertified: truncated-SVD derivative is not the full/constant-rank pseudoinverse derivative"),
            "jac_singular_values": jac_singular, "jac_normalization": "relative loss; log dimensionless poles",
            "bandwidth_ry": bandwidth, "wall_seconds": time.monotonic() - start,
            "nfev": result.nfev, "message": result.message,
            "initializer": initializer, "gram_rank": gram_rank,
            "gram_discarded_fraction": discarded_fraction,
            "numerical_log_bounds": [lower, upper], "active_bounds": result.active_mask,
            "moment_constraints": "none", "residue_constraint": "Hermitian real channels",
            "numerical_candidate_eligible":bool(eligible),
            "candidate_scope":"Numerical prerequisites only; unresolved degeneracy or unconverged fit is not a candidate",
            **iteration_receipt,**degeneracy}


def synthetic_check():
    """Small CPU check of exact derivative, Kaufman gradient, and moving poles."""
    rng = np.random.default_rng(306)
    z = np.linspace(0.04, 1.5, 24) + 0.025j
    poles = np.array([0.24 - 0.012j, 0.71 - 0.037j, 1.17 - 0.08j])
    residues = rng.normal(size=(3, 7))
    samples = basis(z, poles) @ residues
    y = np.vstack((samples.real, samples.imag))
    weights = 1 / (1 + z.real**2)
    sw = np.sqrt(np.r_[weights, weights])
    data = sw[:, None] * y
    data /= np.linalg.norm(data)
    theta = np.log(np.r_[poles.real * 1.03, -poles.imag * 1.07])
    residual, exact, _ = _residual_jac(theta, z, sw, data, exact=True)
    _, kaufman, _ = _residual_jac(theta, z, sw, data)
    eps = 1e-6
    finite = np.stack([(_residual_jac(theta + eps*np.eye(6)[k], z, sw, data)[0]
                       - _residual_jac(theta - eps*np.eye(6)[k], z, sw, data)[0]) / (2*eps)
                      for k in range(6)], axis=1)
    fitted = fit(z, weights, y @ y.T, samples[:, :5], 3, initial=poles * (1.01 + 0.002j))
    derivative_error = np.linalg.norm(exact - finite) / np.linalg.norm(finite)
    gradient_error = np.linalg.norm((exact - kaufman).T @ residual)
    pole_error = np.max(np.abs(fitted["poles_ry"] - poles))
    assert derivative_error < 1e-6, derivative_error
    assert gradient_error < 1e-10, gradient_error
    assert fitted["relative_training_error"] < 1e-7, fitted
    assert pole_error < 1e-6, pole_error
    assert fitted['success'] and fitted['accepted_objective_monotonic'], fitted['termination_reason']
    return {"exact_derivative_relative_error": float(derivative_error),
            "kaufman_gradient_absolute_error": float(gradient_error),
            "training_relative_error": fitted["relative_training_error"],
            "max_pole_error_ry": float(pole_error),
            "accepted_iterations":fitted['accepted_iterations'],
            "relative_step_converged":fitted['relative_step_converged']}
