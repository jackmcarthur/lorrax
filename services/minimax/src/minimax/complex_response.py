"""Response-bank exponential sums shared by groups of samples.

Scalar construction only: a stacked Hankel shift pencil chooses complex
Laplace times shared by every forward (z) and reverse (-conj z) pole of a
sample group; linear projection fits 1/(d-p) and 1/(d-p)^2 on those nodes.
No continuum certificate is claimed. Energies are Ry, times Ry^-1, slopes are
d/d(z^2).
"""
import numpy as np
from scipy import linalg as la

from .uniform_rule import _map_rows, _pinned_blas_threads

RESPONSE_RULE_CAPACITY = 192
# Node slots of one rule: a shared set uses at most one pencil's capacity; a
# single sample whose forward and reverse poles need separate sets uses two.
RESPONSE_NODE_CAPACITY = 2*RESPONSE_RULE_CAPACITY
_RESPONSE_MAX_KAPPA = 5000.
# A group's pencil geometry is abandoned after this many more exponentials
# without a decade of accuracy; the next geometry, then a smaller group, is
# tried. A single sample keeps the full scan, so it never refuses earlier.
_STAGNATION_NODES = 24


def _project(lo, hi, pole, times, decay_rate=0.):
    eta = abs(pole.imag)
    span, zp = (hi-lo)/eta, (pole-lo)/eta
    x = np.unique(np.r_[0., span,
        span/2*(1-np.cos(np.pi*(np.arange(1400)+.5)/1400)),
        zp.real+np.linspace(-5, 5, 500)])
    x = x[(x >= 0) & (x <= span)]
    f = 1/(x-zp)
    origin = 0. if decay_rate else lo
    def basis(xx):
        # Row blocks on the service's pool: each row is the same elementwise
        # arithmetic as the whole-matrix expression, so the same bytes.
        d = lo + eta*xx
        damp = np.minimum(decay_rate*d, 0.)
        a = np.empty((d.size, times.size), np.complex128)
        def fill(i, j):
            a[i:j] = np.exp(damp[i:j, None] - (d[i:j, None]-origin)*times)
        _map_rows(fill, d.size, times.size)
        return a, np.exp(damp)
    a, weight = basis(x)
    a = a/abs(f[:, None])
    scale = la.norm(a, axis=0)
    if np.any(scale == 0) or not np.isfinite(scale).all():
        return None
    coeff = la.lstsq(a/scale, weight[:, None]*np.column_stack((f, f*f))/abs(f[:, None]),
                    cond=1e-14, lapack_driver='gelsd', check_finite=False)[0]/scale[:, None]
    probe = np.unique(np.r_[x, np.linspace(0, span, 2501)])
    error = np.zeros(2)
    # One row-blocked basis for the whole probe; the products stay on the
    # same 16 row chunks as before (a chunk of rows is a contiguous view).
    a, weight = basis(probe)
    start = 0
    for xx in np.array_split(probe, 16):
        stop = start + xx.size
        exact = 1/(xx-zp)
        error = np.maximum(error, np.max(abs(a[start:stop]@coeff
                       - weight[start:stop, None]*np.column_stack((exact, exact*exact))), axis=0))
        start = stop
    mass = np.sum(abs(coeff), axis=0)
    return coeff / np.array([eta, eta*eta]), error, mass


def _poles(z):
    """Forward poles z and reverse poles -conj(z), all in the upper half plane.

    One Green-pair evaluation A(t) serves both orientations: the reverse
    product at time conj(t) is conj(A(t)). An imaginary-axis sample is its own
    reverse pole, so it contributes one pole, not two.
    """
    poles = []
    for p in np.asarray(z, dtype=np.complex128):
        for q in (p, -p.conjugate()):
            if not any(abs(q - r) <= 1e-12*abs(q) for r in poles):
                poles.append(q)
    return np.asarray(poles)


def _fits(lo, hi, poles, times, tol, decay_rate, order=None):
    """Project every pole on shared times; stop at the first rejected pole.

    ``order`` is a caller-owned list of pole indices; a rejected pole moves to
    its front, so the next candidate is tested on the hardest pole first.
    """
    order = list(range(len(poles))) if order is None else order
    fits = [None]*len(poles)
    worst = 0.
    for position, index in enumerate(list(order)):
        pole = poles[index]
        fit = _project(lo, hi, pole, times, decay_rate)
        error = np.inf if fit is None else float(np.max(fit[1]*[1., pole.imag/(2*abs(pole))]))
        mass_rejected = fit is not None and error <= tol and fit[2][0] > _RESPONSE_MAX_KAPPA
        if fit is None or not error <= tol or not np.isfinite(fit[2]).all() or mass_rejected:
            order.insert(0, order.pop(position))
            return None, error, bool(mass_rejected)
        worst = max(worst, error)
        fits[index] = fit
    return fits, worst, False


def _pencil(lo, hi, poles, tol, decay_rate):
    """(eta, scaled poles, window, sample count) of a group's shift pencil.

    The window is in units of the narrowest pole height and reaches 8x the
    farthest pole, so one narrow and one far pole set the size together.
    """
    eta = float(poles.imag.min())
    origin = 0. if decay_rate else lo
    span = max(abs(lo), abs(hi))/eta if decay_rate else (hi-lo)/eta
    zps = (poles-origin)/eta
    geometry = max(span, 8*max(float(zps.real.max()), 0.))
    # Resolve the reciprocal and its square before taking the shift pencil:
    # principal-log modes stop at pi/step; their tails decay as t*exp(-t).
    horizon = -np.log(tol) + np.log(-np.log(tol)) + 6.
    size = max(800, int(np.ceil((geometry + 16.)*horizon/(2*np.pi))))
    return eta, zps, geometry, size


def _shared_times(lo, hi, poles, tol, previous=None, decay_rate=0., patience=None, pencil=None):
    """Complex times shared by every pole, or None.

    A stacked (multi-channel) Hankel shift pencil: each pole contributes the
    Hankel matrix of 1/(x - z_p) sampled at the group's finest scale; their
    common row space holds the shared exponentials. Coefficients are then the
    per-pole linear projection of the single-pole construction, with the same
    sampled-error and coefficient-mass acceptance.
    """
    order = list(range(len(poles)))
    if previous is not None and len(previous):
        fits, _, _ = _fits(lo, hi, poles, previous, tol, decay_rate, order)
        if fits is not None:
            return previous, fits
    eta, zps, geometry, size = _pencil(lo, hi, poles, tol, decay_rate)
    columns = _pencil_columns(size, len(zps))
    steps = [(padding, (geometry+padding)/(2*size)) for padding, _ in _GEOMETRIES]
    # Each geometry's pencil depends on its padding only, never on a decision,
    # so a source may build the next one while this one is scanned.
    source = (host_pencils if pencil is None else pencil)(zps, size, columns, steps)
    try:
        for g, ((_, flatten), (_, step), (s, shift)) in enumerate(zip(_GEOMETRIES, steps, source)):
            start = list(order)
            got = _scan(lo, hi, poles, tol, decay_rate, patience, order, eta, step, flatten, s, shift)
            if got is not None and pencil is not None and not _resolved(pencil, s, got[2]):
                # The source cannot resolve the accepted count's smallest
                # singular value: the geometry is decided on the reference
                # pencil, rebuilt and scanned again from the same pole order.
                # The accepted node set is then the reference's own (bitwise,
                # when the source's earlier decisions match); a rejection
                # there is the reference's, and the scan goes on.
                order[:] = start
                s, shift = next(host_pencils(zps, size, columns, steps[g:g+1]))
                got = _scan(lo, hi, poles, tol, decay_rate, patience, order, eta, step, flatten, s, shift)
            if got is not None:
                return got[:2]
    finally:
        source.close()
    return None


#: A pencil source's acceptance stands when its declared singular-value error
#: (``pencil.sigma_error``, in units of sigma_max) resolves the accepted count's
#: smallest singular value to this relative accuracy; otherwise the reference
#: decides that geometry. Fe V2 (device dilation, sigma_error 8 eps): the
#: accepted counts' bounds are 2.5e-2 for group [0:22] (sigma_80/sigma_1 =
#: 8.9e-15, nodes off by 4e-3) and <= 4.7e-5 for every other group (nodes
#: within 3e-7 of the reference), pool logs rcgpu_ab4, rcgpu_ab6.
_SOURCE_SIGMA_RTOL = 1e-3


def _resolved(pencil, s, n):
    """True when the source resolves sigma_n of its accepted count n (see _SOURCE_SIGMA_RTOL)."""
    error = getattr(pencil, "sigma_error", None)
    return error is not None and error*s[0] <= _SOURCE_SIGMA_RTOL*s[n - 1]


#: (padding, flattening) of the proposal geometries, scanned in order.
# A short interval ending near the resonance gives growing modes. Extend
# only the proposal geometry; fit/check the physical interval.
_GEOMETRIES = ((0., 1.), (2., .9), (4., .9), (8., 1.), (10., 1.), (16., .9))


def _pencil_columns(size, npoles):
    """Column subset of each pole's Hankel block.

    The shared exponentials live in the ROW space (index i). A column subset
    dense near j=0, where 1/(x-z_p) varies fastest, and geometric beyond spans
    it; the stack stays ~2*size wide for any group.
    """
    ids = np.arange(size)
    width = max(32, 2*size//npoles)
    return np.unique(np.r_[ids[:16], np.round(np.geomspace(16, size-1, width)).astype(int)])


def host_pencils(zps, size, columns, steps):
    """The reference shift pencils, one ``(s, shift)`` per ``(padding, step)``, lazily.

    Pole p's block is ``B_p[i, c] = 1/(-padding + (i + columns[c])*step - zp)``
    over ``i < size``, scaled to unit Frobenius norm, and its shifted block is
    the same scale over ``base + step``. The stack ``H = [B_1 .. B_P]`` is
    factored through its tall adjoint, ``H = R^H Q^H``, and the SVD of the
    small ``R^H = U S W^H`` gives H's singular triplets at QR cost. ``s`` holds
    the singular values in descending order (at least ``rank``) and ``shift``
    is the ``rank x rank`` projection ``U_r^H [shifted] Q W_r``, ``rank =
    min(RESPONSE_RULE_CAPACITY, len(s))``.

    This is the contract of the ``pencil`` argument of
    :class:`ResponseGroupTree`: a source with the same signature (a generator
    of the same pairs) may compute them anywhere; :func:`_shared_times` reads
    nothing else of a geometry. Host scipy at the pinned BLAS thread count.
    """
    ids = np.arange(size)
    for padding, step in steps:
        base = -padding+(ids[:, None]+ids[None, :])*step
        blocks, shifted = [], []
        for zp in zps:
            block = 1/(base[:, columns]-zp)
            scale = 1/la.norm(block)
            blocks.append(scale*block)
            shifted.append(scale/(base[:, columns]+step-zp))
        # The stack is size x (poles*size). Factor its tall adjoint once,
        # H = R^H Q^H, and take the SVD of the small R^H: the same singular
        # triplets as a direct SVD of the wide stack, at QR cost.
        q, r = la.qr(np.hstack(blocks).conj().T, mode="economic", check_finite=False)
        u, s, wh = la.svd(r.conj().T, check_finite=False)
        rank = min(RESPONSE_RULE_CAPACITY, len(s))
        yield s, u[:, :rank].conj().T@(np.hstack(shifted)@q)@wh[:rank].conj().T


def _scan(lo, hi, poles, tol, decay_rate, patience, order, eta, step, flatten, s, shift):
    """Node counts n = 8, 12, ... of one geometry's pencil: ``(times, fits, n)`` or None."""
    best, best_n = np.inf, 0
    for n in range(8, len(shift) + 1, 4):
        roots = la.eigvals(shift[:n, :n]/np.sqrt(s[:n, None]*s[None, :n]),
                          check_finite=False)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = -np.log(roots)/step
        if not np.isfinite(t).all() or np.any(t.real < 0):
            continue
        times = (flatten*t.real+1j*t.imag)/eta
        if decay_rate:
            times = times[times.real <= decay_rate]
        if not len(times):
            continue
        fits, error, mass_rejected = _fits(lo, hi, poles, times, tol, decay_rate, order)
        if fits is not None:
            return times, fits, n
        if mass_rejected:
            return None  # Try the next padding/flattening geometry, not more cancelling terms.
        if error < best/10:
            best, best_n = error, n
        elif patience is not None and n - best_n >= patience:
            return None  # No decade of accuracy in this many more exponentials.
    return None


def _imaginary_times(lo, hi, poles, tol, decay_rate, capacity):
    """Closed-form times on the imaginary axis shared by every pole, or None.

    The pencil's decaying nodes (Re t > 0) must carry a pole at Re z with
    coefficients ~exp(Re t * (Re z - reference)): far up the interval the
    coefficient-mass gate refuses every geometry (Si 4^3, 44.8 + 2.6i eV in a
    0.69-52.5 eV interval: 5.2e3 at best). For Im z > 0,
    1/(x - z) = i int_0^inf exp(-i (x - z) s) ds, so nodes t = i s (Re t = 0:
    no decay, no growth, inside every decay_rate bound) carry coefficients
    bounded by the quadrature weights, mass O(1) wherever Re z lies. The
    integral is cut at S = horizon (in 1/eta, tail exp(-S) < tol), and n
    Gauss-Legendre nodes resolve the phase across the interval width W = (hi -
    lo)/eta: n = W*S/4. The coefficients come from the same projection and
    acceptance as the pencil's. None when n exceeds ``capacity``.
    """
    eta = float(poles.imag.min())
    horizon = -np.log(tol) + np.log(-np.log(tol)) + 6.
    n = int(np.ceil((hi-lo)/eta*horizon/4))
    if n > capacity:
        return None
    u, _ = np.polynomial.legendre.leggauss(n)
    times = 1j*horizon*(u+1)/(2*eta)
    fits, _, _ = _fits(lo, hi, poles, times, tol, decay_rate)
    return None if fits is None else (times, fits)


def _rule(z, sets):
    """Scatter pole fits on node sets into forward and reverse sample rows.

    ``sets`` is a list of (times, poles, fits). A node t is one Green pair
    A(t): forward rows fit 1/(d-z) on t; reverse rows, evaluated at conj(t)
    from conj(A(t)), fit 1/(d+z) as the conjugate of the -conj(z) fit on t.
    A set without a sample's pole contributes zero weight to that row.
    """
    times = np.concatenate([t for t, _, _ in sets])
    count = len(times)
    t = np.zeros(RESPONSE_NODE_CAPACITY, complex)
    t[:count] = times
    shape = (len(z), 2, RESPONSE_NODE_CAPACITY)
    value, derivative = np.zeros(shape, complex), np.zeros(shape, complex)
    errors, mass = np.zeros((len(z), 2, 2)), np.zeros((len(z), 2, 2))
    for j, point in enumerate(z):
        start = 0
        for set_times, poles, fits in sets:
            stop = start + len(set_times)
            for side, pole in enumerate((point, -point.conjugate())):
                match = np.flatnonzero(abs(poles - pole) <= 1e-12*abs(pole))
                if not match.size:
                    continue
                coefficient, fit_error = fits[int(match[0])][:2]
                if side:
                    coefficient = np.conj(coefficient)
                value[j, side, start:stop] = coefficient[:, 0]
                derivative[j, side, start:stop] = (1 if side == 0 else -1)*coefficient[:, 1]/(2*point)
                errors[j, side] = [fit_error[0], fit_error[1]*abs(point.imag/(2*point))]
            start = stop
        for side in (0, 1):
            mass[j, side] = [point.imag*np.sum(abs(value[j, side])),
                             point.imag**3*np.sum(abs(derivative[j, side]))]
    return dict(t=t, value=value, derivative=derivative, count=count,
                sampled_error=errors, coefficient_mass=mass)


def response_group_rules(lo_ry, hi_ry, z_ry, *, rel_tol=1e-8, previous=None,
                         decay_rate=0., pencil=None):
    """Shared complex-time rules for a group of response samples.

    Built at the Sigma rule builder's pinned BLAS thread count: the Hankel
    QR/SVD, eigvals and least squares sum in a thread-count dependent order,
    and a different order picks different node times (Fe 4^3 map-1 group:
    one node count, three time digests at 8, 16 and 32 threads, P2-S
    2026-09-25). The shared-pole W poles inherit these rules.

    Every node t is ONE Green-pair evaluation A(t): forward rows use the
    exponential exp[-(d-reference_ry)*t] and fit 1/(d-z); reverse rows use
    conj(A(t)), i.e. the exponential at conj(t), and fit 1/(d+z). A group
    whose pencil fit fails takes the closed-form imaginary-axis set
    (``_imaginary_times``) when that fits the pencil's capacity, else is split
    in halves, down to single samples; a group whose pencil is wider than both
    halves' is split without a try.
    A successful shared rule can use more nodes than one member would need
    alone; its Green-pair evaluations serve all members of the group.

    Returns a list of rules. Each has ``members`` (indices into ``z_ry``),
    ``t[RESPONSE_NODE_CAPACITY]``, ``value``/``derivative`` of shape
    ``[members, 2 (forward, reverse), RESPONSE_NODE_CAPACITY]`` (value and
    d/d(z^2)), ``count``, ``sampled_error`` and ``coefficient_mass``
    ``[members, 2, 2]``, and ``reference_ry``. ``previous`` is a list of
    earlier rules; one with the same member set (any order) is tried first. A positive
    decay_rate bounds occupation products by min(1,exp(decay_rate*d)); errors
    then use that envelope and 0<=Re(t)<=decay_rate. Bounds are sampled, not
    proven.

    The serial walk of :class:`ResponseGroupTree`; a caller with idle
    processes evaluates the tree's attempts there and passes them to
    :meth:`ResponseGroupTree.rules` for the same bytes.

    ``pencil`` is the source of each geometry's shift pencil (the contract of
    :func:`host_pencils`, the default). Only the pencil's arithmetic moves
    with it: every acceptance decision, tolerance and group stays here, on
    the host. A geometry that accepts on another source is rebuilt on
    :func:`host_pencils` and scanned again, so an accepted node set is the
    reference's own; only rejections are decided on the source's rounding.
    """
    return ResponseGroupTree(lo_ry, hi_ry, z_ry, rel_tol=rel_tol, previous=previous,
                             decay_rate=decay_rate, pencil=pencil).rules()


class ResponseGroupTree:
    """The halving tree of :func:`response_group_rules`, one node per member set.

    A node is a tuple of indices into ``z_ry``; its children are its two
    halves. ``attempt(node)`` is the node's whole-group construction (shared
    pencil, else the imaginary-axis set; for one sample also the separate
    forward/reverse sets) and is a pure function of the node on one machine
    class at the pinned BLAS thread count, so it may run on any process. The
    serial walk builds a node whole when its attempt succeeds and its halves
    otherwise; ``rules(attempts)`` takes attempts evaluated elsewhere, keyed by
    node, and computes any it is not given. A one-sample attempt never returns
    None: it returns its rule or raises the refusal. ``pencil`` is the shift
    pencils' source (:func:`response_group_rules`); with a device source an
    attempt is a pure function of its node on one device class instead.
    """

    def __init__(self, lo_ry, hi_ry, z_ry, *, rel_tol=1e-8, previous=None, decay_rate=0.,
                 pencil=None):
        lo, hi = float(lo_ry), float(hi_ry)
        z = np.asarray(z_ry, dtype=np.complex128).reshape(-1)
        if (not z.size or not np.isfinite([lo, hi, rel_tol, decay_rate]).all()
                or not np.isfinite(z).all() or decay_rate < 0 or hi <= lo
                or np.any(z.imag <= 0) or not 1e-13 <= rel_tol < .1):
            raise ValueError('invalid response frequency/domain/tolerance')
        self.lo, self.hi, self.z, self.rel_tol, self.decay_rate = lo, hi, z, rel_tol, decay_rate
        self.pencil = pencil
        self.reference = 0. if decay_rate else lo
        self.root = tuple(range(len(z)))
        self._old = {frozenset(rule["members"]): rule["t"][:rule["count"]]
                     for rule in (previous or ())}

    def size(self, members):
        """Shift-pencil sample count of a node (its attempt costs ~size^3 per geometry)."""
        return _pencil(self.lo, self.hi, _poles(self.z[list(members)]), self.rel_tol/2,
                       self.decay_rate)[3]

    def pencil_shapes(self, members):
        """``(size, poles, columns)`` of every shift pencil an attempt on the node may build.

        The group's stacked pencil and, for one sample, each pole's own (its
        separate-sets fallback). A device pencil source compiles these ahead.
        """
        poles = _poles(self.z[list(members)])
        out = []
        for ps in [poles] + ([poles[i:i+1] for i in range(len(poles))] if len(members) == 1 else []):
            size = _pencil(self.lo, self.hi, ps, self.rel_tol/2, self.decay_rate)[3]
            out.append((size, len(ps), len(_pencil_columns(size, len(ps)))))
        return out

    def children(self, members):
        """The two halves of a node; () for one sample."""
        half = len(members)//2
        return () if len(members) == 1 else (tuple(members[:half]), tuple(members[half:]))

    def tried(self, members):
        """False when the node's pencil is wider than both halves': built as halves untried."""
        # The narrowest height (one half) and the farthest reach (the
        # other) widen the whole pencil past both halves': it costs size^3
        # per geometry, and every such group measured failed its shared
        # fit after all six geometries (Si 40 and Fe 27 samples; Si 22
        # samples reaching 22-42 eV; CHIRULE 2026-10-06), so the halves
        # are built directly, exactly as they were after that failure.
        halves = self.children(members)
        return not halves or self.size(members) <= max(self.size(h) for h in halves)

    def attempt(self, members):
        """``[rule]`` built whole on the node, or None (build its halves)."""
        with _pinned_blas_threads():
            return self._attempt(list(members))

    def _attempt(self, members):
        lo, hi, z, tol, decay_rate = self.lo, self.hi, self.z, self.rel_tol/2, self.decay_rate
        poles = _poles(z[members])
        got = _shared_times(lo, hi, poles, tol, self._old.get(frozenset(members)), decay_rate,
                            patience=None if len(members) == 1 else _STAGNATION_NODES,
                            pencil=self.pencil)
        if got is None:
            # Where the pencil fails, the closed-form imaginary-axis set serves
            # every pole when it fits the pencil's capacity (two pencils' for
            # one sample, whose poles could otherwise take two sets).
            got = _imaginary_times(lo, hi, poles, tol, decay_rate, RESPONSE_NODE_CAPACITY
                                   if len(members) == 1 else RESPONSE_RULE_CAPACITY)
        if got is not None:
            rule = _rule(z[members], [(*got[:1], poles, got[1])])
            return [dict(rule, members=list(members), reference_ry=self.reference)]
        if len(members) > 1:
            return None
        # Forward and reverse poles on separate node sets: each node then
        # serves one orientation, exactly as separate per-pole rules do.
        sets = []
        for pole in poles:
            one = np.asarray([pole])
            got = _shared_times(lo, hi, one, tol, None, decay_rate, pencil=self.pencil)
            if got is None:
                raise ValueError(f'response exponential fit failed: interval={lo, hi}, '
                                 f'sample={z[members[0]]}, pole={pole}, tolerance={self.rel_tol}')
            sets.append((got[0], one, got[1]))
        rule = _rule(z[members], sets)
        return [dict(rule, members=list(members), reference_ry=self.reference)]

    def pending(self, attempts):
        """Tried nodes of the serial walk whose attempts are not in ``attempts``, in walk order."""
        out = []

        def walk(node):
            if self.tried(node):
                if node not in attempts:
                    out.append(node)
                    return
                if attempts[node] is not None:
                    return
            for half in self.children(node):
                walk(half)
        walk(self.root)
        return out

    def rules(self, attempts=None):
        """The serial walk's rules; attempts missing from ``attempts`` are computed here."""
        attempts = {} if attempts is None else attempts

        def build(node):
            if self.tried(node):
                got = attempts[node] if node in attempts else self._attempt(list(node))
                if got is not None:
                    return got
            return [rule for half in self.children(node) for rule in build(half)]

        with _pinned_blas_threads():
            return build(self.root)
