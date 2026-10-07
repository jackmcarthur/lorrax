"""Stream compensated atom-radial zeta contributions through the charge V pass.

This file owns the representation seam, not a new Coulomb kernel.  The smooth
kernel is supplied by the existing V consumer.  Local tables use the ordinary
three-dimensional 1/r identity in isdf.augmentation.  Their physical integrals
are converted once to LORRAX's grid-sum/Rydberg convention.
"""
from __future__ import annotations

import numpy as np


def make_periodic_compensation_action(mesh, cache, *, centroid_basis, fft_points):
    r"""Plan the global compensation action ``conj(M) G transpose(M)``.

    Parameters
    ----------
    cache : dict
        Authenticated periodic cache bundle. ``gram`` has physical Ry units
        for unit multipoles; both canonical atom/lm axes remain P(None,x,y).
    centroid_basis : common.centroid_basis.PackedCentroidBasis
        Existing centroid solve-axis receipt and authoritative active mask.
        Interleaved packed ghosts are not inferred from a physical prefix.
    fft_points : int
        Real-space FFT point count. The metric receives (Nfft/Omega)^2 once.

    Returns
    -------
    action : callable
        ``action(moment_rows, gram)`` returns the face-sharded Ry matrix.
        Moment rows are (nq, mu_carrier, moment_carrier), P(None,x,y), in
        grid-sum units and canonical atom-major complete complex-lm order.
        The Gram is an explicit JIT operand, never a replicated closure.
        Inactive centroid rows and harmonic tails are zeroed before either GEMM.
    receipt : dict
        Shapes, array ownership and the two public N,N GEMM plans. This
        function introduces no local metric, periodic mean or head term.
    """
    import jax
    import jax.numpy as jnp
    import hashlib
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import transpose_xy
    from distrib_la import gemm_plan, workspace_bytes_per_rank
    from runtime.padding import (PaddedAxis, authenticate_padded_axis,
                                 pad_to_axis, pad_square)
    from isdf.coulomb_fourier_cache import PERIODIC_SCHEMA, _periodic_face

    metadata = cache['metadata']
    centroid_axis = centroid_basis.solve_axis
    active = np.array(centroid_basis.active_mask, copy=True)
    if (metadata.get('schema') != PERIODIC_SCHEMA or not isinstance(centroid_axis, PaddedAxis)
            or active.dtype != np.dtype(bool) or active.shape != (centroid_axis.carrier,)
            or np.count_nonzero(active) != int(centroid_basis.n_logical)
            or int(centroid_basis.n_packed) != centroid_axis.carrier):
        raise ValueError('global compensation requires authenticated cache and centroid axis')
    face, moment_axis = _periodic_face(cache['gram'], mesh, metadata['logical_shape'])
    if cache['moment_axis'] != moment_axis:
        raise ValueError('global compensation moment axis receipt mismatch')
    authenticate_padded_axis(centroid_axis.logical, centroid_axis.carrier,
        centroid_axis, name='global compensation centroid')
    if (centroid_axis.logical < 1 or centroid_axis.divisor % moment_axis.divisor
            or int(fft_points) != fft_points or int(fft_points) < 1):
        raise ValueError('global compensation centroid carrier or FFT count mismatch')
    active.flags.writeable = False
    nq = int(metadata['logical_shape'][0])
    mu, na = centroid_axis.carrier, moment_axis.carrier
    volume = float(metadata['geometry']['cell_volume_bohr3'])
    scale = (int(fft_points)/volume)**2
    left = gemm_plan(mesh, m=mu, n=na, k=na, nq=nq,
        dtype=jnp.complex128, layout='face', warmup=False)
    right = gemm_plan(mesh, m=mu, n=mu, k=na, nq=nq,
        dtype=jnp.complex128, layout='face', warmup=False)

    def contract(moment_rows, gram):
        if (moment_rows.shape != (nq, mu, na) or gram.shape != (nq, na, na)
                or moment_rows.dtype != jnp.complex128 or gram.dtype != jnp.complex128):
            raise ValueError('global compensation operand shape or dtype mismatch')
        moments = pad_to_axis(pad_to_axis(moment_rows, centroid_axis, axis=1),
                              moment_axis, axis=2)
        moments = jnp.where(jnp.asarray(active)[None, :, None], moments, 0.)
        metric = pad_square(gram, moment_axis)*scale
        value = right(left(moments.conj(), metric), transpose_xy(moments, mesh))
        value = pad_square(value, centroid_axis)
        return jnp.where((jnp.asarray(active)[:, None]&jnp.asarray(active)[None, :])[None], value, 0.)

    action = jax.jit(contract, in_shardings=(face, face), out_shardings=face)
    receipt = dict(cache_path=cache['path'], cache_file_sha256=cache['file_sha256'],
        logical_centroids=int(centroid_basis.n_logical), carrier_centroids=mu,
        active_centroid_mask_sha256=hashlib.sha256(active.tobytes()).hexdigest(),
        centroid_mask_owner='existing PackedCentroidBasis.active_mask; includes interleaved ghosts',
        logical_moments=moment_axis.logical, carrier_moments=na, q_count=nq,
        fft_points=int(fft_points), cell_volume_bohr3=volume, metric_grid_scale=scale,
        metric_scale_applied_once=True, input_and_output_sharding='P(None,x,y)',
        complementary_face='common.collectives.transpose_xy; one partner tile exchange',
        gram_is_explicit_operand=True, runtime_axis_permutation=False,
        gemm_shapes=[dict(m=mu, n=na, k=na, nq=nq), dict(m=mu, n=mu, k=na, nq=nq)],
        public_gemm_backends=[left.backend, right.backend],
        vendor_gemm_workspace_bytes_per_rank=[
            workspace_bytes_per_rank(left, 'gemm', ((nq, mu, na), (nq, na, na)), np.complex128),
            workspace_bytes_per_rank(right, 'gemm', ((nq, mu, na), (nq, na, mu)), np.complex128)],
        arrays_per_rank=dict(moment_rows_bytes=nq*mu*na*16//mesh.size,
            gram_bytes=nq*na*na*16//mesh.size,
            intermediate_bytes=nq*mu*na*16//mesh.size,
            result_bytes=nq*mu*mu*16//mesh.size),
        scope='Global periodic compensation Gram only; free local, both means and head are separate existing owners.')
    return action, receipt


def _angular_channels(lm):
    """Authenticate a complete integer harmonic space before using its metric."""
    channels = np.asarray(lm)
    if (channels.ndim != 2 or channels.shape[1] != 2 or channels.shape[0] == 0
            or not np.all(np.isfinite(channels))
            or not np.all(channels == np.round(channels))):
        raise ValueError("local harmonics must be a nonempty integer (l,m) table")
    channels = channels.astype(np.int64)
    rows = [tuple(row) for row in channels]
    wanted = {(l, m) for l in range(int(channels[:, 0].max())+1)
              for m in range(-l, l+1)}
    if len(set(rows)) != len(rows) or set(rows) != wanted:
        raise ValueError("local harmonics must contain each (l,m) through lmax exactly once")
    return channels


def atomic_radial_metrics(radius, weights_dr, lm_ell, *, support_radius,
                          fft_points, cell_volume, interpolation_degree=None,
                          quadrature_order=None):
    """Prepare per-degree radial metrics and compact compensation moment rows.

    ``interpolation_degree=None`` retains the incumbent radial collocation.
    A positive integer enables the physical-density interpolant and positive
    field-energy Gram from ``radial_coulomb_metric_interpolated``. In that
    mode compensation is the analytic compact polynomial, normalized by its
    exact beta-function multipole, with its enclosed field and Fourier map
    integrated on the same positive quadrature returned for the delta density.

    Returned arrays are small species tables: ``delta_metric`` and
    ``compensation_metric`` (N_degree,N_r,N_r), ``moments`` and
    ``compensation_shapes`` (N_degree,N_r).  A local zeta is represented by
    its rho_lm values on this radial grid, not by thin-shell delta functions.
    The discrete Poisson quadrature is independently converged.  Metrics
    include the exact 2*(N_fft/Omega)^2 Hartree-to-Rydberg/grid conversion.
    """
    from isdf.augmentation import radial_coulomb_potential

    r = np.asarray(radius, dtype=np.float64)
    w = np.asarray(weights_dr, dtype=np.float64)
    ell = np.asarray(lm_ell)
    if (r.ndim != 1 or w.shape != r.shape or r.size < 2
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(w))
            or np.any(r <= 0) or np.any(np.diff(r) <= 0) or np.any(w <= 0)
            or ell.size == 0 or not np.all(np.isfinite(ell))
            or not np.all(ell == np.round(ell))):
        raise ValueError("invalid radial Coulomb grid or angular degrees")
    degrees = np.unique(ell.astype(np.int64))
    if (not np.isfinite(cell_volume) or float(cell_volume) <= 0
            or not np.isfinite(fft_points) or int(fft_points) != fft_points
            or int(fft_points) < 1 or not np.isfinite(support_radius)
            or float(support_radius) < r[-1] or np.any(degrees < 0)):
        raise ValueError("invalid radial Coulomb geometry")
    scale = 2*(float(fft_points)/float(cell_volume))**2
    if interpolation_degree is not None:
        from isdf.augmentation import radial_coulomb_metric_interpolated
        from scipy.special import beta, betainc

        tables = radial_coulomb_metric_interpolated(
            r, degrees, support_radius=support_radius,
            interpolation_degree=interpolation_degree,
            quadrature_order=quadrature_order)
        qr, qw = tables['quadrature_radius'], tables['quadrature_weights_dr']
        R = float(support_radius)
        shapes, qshapes, gmetric, gself = [], [], [], []
        for row, l in enumerate(degrees):
            normalization = R**(2*l+3)*beta(l+1.5, 7.)/2
            shape = r**l*np.maximum(1-(r/R)**2, 0)**6/normalization
            qshape = qr**l*np.maximum(1-(qr/R)**2, 0)**6/normalization
            enclosed = betainc(l+1.5, 7., (qr/R)**2)
            self_metric = (4*np.pi*np.dot(qw, enclosed**2/qr**(2*l+2))
                           + 4*np.pi/(2*l+1)/R**(2*l+1))
            moment = tables['moments'][row]
            gmetric.append(scale*np.outer(moment, moment)*self_metric)
            shapes.append(shape)
            qshapes.append(qshape)
            gself.append(self_metric)
        return dict(tables, delta_metric=scale*tables['metric'],
                    compensation_metric=np.asarray(gmetric),
                    compensation_shapes=np.asarray(shapes),
                    compensation_quadrature_shapes=np.asarray(qshapes),
                    compensation_self=np.asarray(gself))
    if quadrature_order is not None:
        raise ValueError("field quadrature order requires a radial interpolation degree")
    eye = np.eye(len(r))
    rows, metric, gmetric, shapes = [], [], [], []
    envelope = np.maximum(1-(r/float(support_radius))**2, 0)**6
    for l in degrees:
        potential = radial_coulomb_potential(eye, r, w, l)
        K = (w*r*r)[:, None] * potential.T
        moment = w*r**(l+2)
        shape = r**l * envelope
        shape = shape / (shape @ moment)
        Kg = np.outer(moment, moment) * (shape @ K @ shape)
        rows.append(moment)
        metric.append(scale*K)
        gmetric.append(scale*Kg)
        shapes.append(shape)
    return dict(degrees=degrees, moments=np.asarray(rows),
                delta_metric=np.asarray(metric), compensation_metric=np.asarray(gmetric),
                compensation_shapes=np.asarray(shapes))


def _radial_fourier_cache(tables, maximum_wavevector, fourier_points):
    """Bounded one-dimensional Fourier table with direct-quadrature pins.

    Stores (degree,N_k,N_r) density rows and (degree,N_k) compensation
    spectra. Construction streams N_k in chunks of 64; no Q*G*N_quad array
    is formed. Cubic interpolation is accepted only after deterministic
    off-grid direct-quadrature comparisons and the G=0 multipole pin.
    """
    from scipy.interpolate import CubicSpline
    from scipy.special import spherical_jn

    count = int(fourier_points)
    maximum = float(maximum_wavevector)
    if (count != fourier_points or count < 4 or not np.isfinite(maximum)
            or maximum < 0):
        raise ValueError("radial Fourier table requires >=4 points and a finite nonnegative extent")
    # A nonzero interval supports the entirely-zero-momentum synthetic case.
    extent = maximum if maximum > 0 else 1.
    kgrid = np.linspace(0., extent, count)
    qr, qw = tables['quadrature_radius'], tables['quadrature_weights_dr']
    base_map = tables['interpolation_map']
    origin_count = tables['origin_row_count']
    degrees = tables['degrees']
    shape = tables['compensation_quadrature_shapes']

    def direct(wavevectors):
        wavevectors = np.asarray(wavevectors, dtype=np.float64)
        rows, compensation = [], []
        for row, l in enumerate(degrees):
            weighted = spherical_jn(l, wavevectors[:, None]*qr)*(qw*qr*qr)
            density = weighted @ base_map
            density[:, 0] += weighted[:, :origin_count] @ (tables['origin_factors'][row]-1.)
            rows.append(density)
            compensation.append(weighted @ shape[row])
        return np.asarray(rows), np.asarray(compensation)

    values = np.empty((len(degrees), count, base_map.shape[1]))
    comp = np.empty((len(degrees), count))
    for first in range(0, count, 64):
        last = min(first+64, count)
        values[:, first:last], comp[:, first:last] = direct(kgrid[first:last])
    density_spline = CubicSpline(kgrid, values, axis=1)
    compensation_spline = CubicSpline(kgrid, comp, axis=1)
    # Probe cells throughout the entire extent, including the boundaries.
    cells = np.unique(np.linspace(0, count-2, min(129, count-1)).astype(int))
    fraction = .5+.2*np.sin(np.arange(len(cells))*1.618033988749895)
    probes = kgrid[cells]+fraction*(kgrid[1]-kgrid[0])
    exact, exact_comp = direct(probes)
    predicted, predicted_comp = density_spline(probes), compensation_spline(probes)
    error = np.max(np.abs(predicted-exact), axis=1)
    comp_error = np.max(np.abs(predicted_comp-exact_comp), axis=1)
    scale = np.max(np.abs(values), axis=1)
    comp_scale = np.max(np.abs(comp), axis=1)
    if (np.any(error > 1e-12+1e-10*scale)
            or np.any(comp_error > 1e-12+1e-10*comp_scale)):
        raise ValueError("radial Fourier interpolation failed direct-quadrature tolerance; refine fourier_points")
    for row, l in enumerate(degrees):
        target = tables['moments'][row] if l == 0 else np.zeros(base_map.shape[1])
        comp_target = 1. if l == 0 else 0.
        if (not np.allclose(values[row, 0], target, rtol=1e-10, atol=1e-12)
                or not np.isclose(comp[row, 0], comp_target, rtol=1e-10, atol=1e-12)):
            raise ValueError("radial Fourier zero-momentum multipole pin failed")
    return dict(density=density_spline, compensation=compensation_spline,
                points=count, maximum_wavevector=maximum,
                table_bytes=values.nbytes+comp.nbytes,
                retained_spline_bytes=density_spline.c.nbytes+compensation_spline.c.nbytes+kgrid.nbytes,
                max_density_validation_error=float(np.max(error)),
                max_compensation_validation_error=float(np.max(comp_error)),
                validation_points=len(probes))


def _smooth_neutral_tables(tables, *, support_radius, fft_points, cell_volume):
    """Same-interpolant ``<f_i|v|f_j-M_j g_l>`` with physical grid/Ry units.

    A residual with zero multipoles has zero exterior potential, so the
    smooth/residual cross is exactly restricted to the same atomic sphere.
    The nonsymmetric cross matrix must retain its complex adjoint in V.
    """
    from scipy.special import beta, betainc

    q, w = tables['quadrature_radius'], tables['quadrature_weights_dr']
    R = float(support_radius)
    scale = 2*(float(fft_points)/float(cell_volume))**2
    cross, neutral = [], []
    for row, l in enumerate(tables['degrees']):
        sample = tables['interpolation_map'].copy()
        sample[:tables['origin_row_count'], 0] = tables['origin_factors'][row]
        normalization = R**(2*l+3)*beta(l+1.5, 7.)/2
        potential = 4*np.pi/(2*l+1)*(
            betainc(l+1.5, 7., (q/R)**2)/q**(l+1)
            + q**l*R**2/(14*normalization)*(1-(q/R)**2)**7)
        u = scale*(w*q*q*potential) @ sample
        cross.append(u)
        neutral.append(tables['delta_metric'][row] - np.outer(u, tables['moments'][row]))
    if not np.all(np.isfinite(neutral)):
        raise ValueError("smooth-neutral Coulomb cross is unresolved on the physical radial grid")
    return dict(smooth_compensation_cross=np.asarray(cross),
                smooth_neutral_metric=np.asarray(neutral))


def radial_coulomb_provider(zeta_g, rhs, *, smooth_rhs=None, monopole_rhs=None, radius, weights_dr, lm,
                            centers_cart, q_plus_G_cart, cell_volume,
                            fft_points, support_radius, interpolation_degree=None,
                            quadrature_order=None, fourier_points=4097, prepared_cache=None):
    r"""Build the procedural local provider consumed by ZetaG.contract_v.

    Parameters
    ----------
    zeta_g : ZetaG
        The existing smooth-fit object; its factor is reused for local rhs.
    rhs : (Q_pad,mu_packed,N_atom*N_lm*N_r) complex128, q-owner sharded
        Raw local normal-equation coefficients in (atom,lm,radius) order.
        The provider never factors a second Gram matrix.
    smooth_rhs : same shape and q ownership as rhs, optional
        Smooth PS atomic-density RHS. Supplied only for the explicit onsite
        smooth-neutral cross policy. Both RHSs are concatenated once and
        solved with the same charge factor. The physical Fourier image
        remains smooth+delta; V body uses smooth+compensation in this mode.
    monopole_rhs : (Q_pad,mu_packed,N_atom) complex128, optional
        Exact served-field radial Y00 moment RHS, in the same grid units.
        With smooth_rhs it enriches the incumbent paired delta/PS action.
        Without smooth_rhs the explicit delta/exactM0 action has nf+N_atom
        columns and applies enriched free delta-delta minus compensation.
        Its coefficients share the same charge solve;
        delta and compensation receive the identical analytic g0 enrichment.
        This is a physical-field integral, never a native-overlap charge pin.
    radius, weights_dr : (N_r,) float64
        Positive ordered radial points in bohr and positive dr weights.
    lm : (N_lm,2) int
        Complete retained spherical harmonics, in the RHS coefficient order.
    centers_cart : (N_atom,3) float64
        Centers in bohr. Spheres and periodic images must be nonoverlapping.
    q_plus_G_cart : (Q,N_G,3) float64
        The consuming V pass's physical reciprocal momenta, bohr^-1.
    cell_volume, fft_points, support_radius : float, int, float
        Cell geometry and local compact compensation radius. The caller
        independently authenticates tail, angular, radial and PW convergence.
    interpolation_degree, quadrature_order, fourier_points : int or None
        None interpolation preserves the incumbent shell quadrature. Otherwise
        delta uses a piecewise physical-density field metric, compensation is
        analytic r^l(1-r^2/R^2)^6 with exact unit multipole, and a cubic radial
        |q+G| table (default4097 points) replaces per-tile large quadrature maps.
        Direct off-grid and G=0 pins refuse an unconverged Fourier table.
    prepared_cache : authenticated dict, optional
        Explicitly loaded exact CubicSpline artifact. Its source, density
        model, resolution and consuming extent are checked, then the same
        direct pins are repeated. A mismatch refuses without rebuilding.

    Returns
    -------
    dict
        ``rhs``, ``fourier_tile(t,coefficients)`` and ``onsite(coefficients)``.
        These kernels use the existing whole-q-owner layout, stream degrees
        and atoms through scans, and do not replicate a Coulomb matrix.
        The representation has no transverse/Breit locality certificate.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.shard_map import shard_map
    from common.collectives import device_put_process_local
    from scipy.special import spherical_jn, sph_harm_y

    st, mesh = zeta_g.store, zeta_g.mesh
    harmonics = _angular_channels(lm)
    centers = np.asarray(centers_cart, dtype=np.float64)
    kg = np.asarray(q_plus_G_cart, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    w = np.asarray(weights_dr, dtype=np.float64)
    nr, nh, na = len(r), len(harmonics), len(centers)
    Qp, mu, gt = int(st.Q_pad), int(st.mu_pad), int(st.g_tile)
    XY = ('x', 'y')
    qspec = P(XY, None, None)
    qsh = NamedSharding(mesh, qspec)
    if (centers.shape != (na, 3) or na == 0
            or not np.all(np.isfinite(centers)) or not np.all(np.isfinite(kg))
            or kg.shape != (st.Q, st.g_axis.logical, 3)
            or rhs.shape != (Qp, mu, na*nh*nr)
            or rhs.sharding != qsh):
        raise ValueError("radial Coulomb provider geometry or q-owner RHS disagrees with smooth fit")
    compensated_body = smooth_rhs is not None
    enriched_monopole = monopole_rhs is not None
    delta_only_enrichment = enriched_monopole and not compensated_body
    if compensated_body or delta_only_enrichment:
        from isdf.zeta_mubatch import _require_q_owned
        if interpolation_degree is None:
            raise ValueError("onsite smooth-neutral cross requires physical density interpolation")
        if zeta_g.solver_kind == 'lu':
            raise ValueError("onsite smooth-neutral provider requires the scalar charge factor")
        _require_q_owned(rhs, mesh, (Qp, mu, na*nh*nr), name='delta local RHS')
        if compensated_body:
            _require_q_owned(smooth_rhs, mesh, rhs.shape, name='smooth PS local RHS')
        if enriched_monopole:
            _require_q_owned(monopole_rhs, mesh, (Qp, mu, na), name='served monopole local RHS')
    tables = atomic_radial_metrics(r, w, harmonics[:, 0], support_radius=support_radius,
                                  fft_points=fft_points, cell_volume=cell_volume,
                                  interpolation_degree=interpolation_degree,
                                  quadrature_order=quadrature_order)
    if compensated_body or delta_only_enrichment:
        cross_tables = _smooth_neutral_tables(tables, support_radius=support_radius,
                                             fft_points=fft_points, cell_volume=cell_volume)
        inputs = ((rhs,monopole_rhs) if delta_only_enrichment else
                  ((rhs, smooth_rhs, monopole_rhs) if enriched_monopole else (rhs, smooth_rhs)))
        provider_rhs = jnp.concatenate(inputs, axis=-1)
        nc = (na*nh*nr if delta_only_enrichment else 2*na*nh*nr)+(na if enriched_monopole else 0)
        _require_q_owned(provider_rhs, mesh, (Qp, mu, nc), name='delta/exactM0 or paired local RHS')
    else:
        cross_tables, provider_rhs = None, rhs
    cache = None
    if prepared_cache is not None and interpolation_degree is None:
        raise ValueError("prepared Fourier cache requires physical density interpolation")
    if interpolation_degree is not None:
        valid = np.arange(kg.shape[1])[None, :] < zeta_g.ngk_per_q[:, None]
        maximum = np.max(np.where(valid, np.linalg.norm(kg, axis=-1), 0.))
        if prepared_cache is None:
            cache = _radial_fourier_cache(tables, maximum, fourier_points)
        else:
            from isdf.coulomb_fourier_cache import validate_coulomb_fourier_cache
            cache = validate_coulomb_fourier_cache(prepared_cache, tables, maximum, fourier_points)
    degrees = tables['degrees']
    degree_row = np.searchsorted(degrees, harmonics[:, 0])
    steps = np.asarray([(a, h) for a in range(na) for h in range(nh)], dtype=np.int32)
    rep = NamedSharding(mesh, P())
    put = lambda a, spec=P(): device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    rows = put(tables['moments'])
    difference = put(tables['delta_metric']-tables['compensation_metric'])
    dr = put(degree_row)
    scan_steps = put(steps)
    neutral = None if cross_tables is None else put(cross_tables['smooth_neutral_metric'])
    monopole_harmonic = int(np.flatnonzero(np.all(harmonics == (0, 0), axis=1))[0])
    moment_cross = None
    if enriched_monopole:
        zero_row = int(np.flatnonzero(degrees == 0)[0])
        scale = 2*(float(fft_points)/float(cell_volume))**2
        # <g0|v|delta-M delta*g0>; the quadratic enrichment cancels.
        moment_cross = put(cross_tables['smooth_compensation_cross'][zero_row]
            - scale*tables['compensation_self'][zero_row]*tables['moments'][zero_row])

    def ft_local(coefficients, bessel, angular, g_radial, steps_, degree_rows, moment_rows):
        # Every operand is on this q owner. No all-gather in this body.
        delta_coefficients = coefficients[..., :na*nh*nr] if compensated_body or delta_only_enrichment else coefficients
        coeff = delta_coefficients.reshape(coefficients.shape[0], mu, na, nh, nr)
        zero = jnp.zeros((coeff.shape[0], mu, gt), jnp.complex128)
        def add(acc, ah):
            delta, comp = acc
            a, h = ah
            lrow = degree_rows[h]
            values = coeff[:, :, a, h]
            radial = jnp.einsum('qmr,qgr->qmg', values, bessel[:, lrow])
            moment = jnp.einsum('qmr,r->qm', values, moment_rows[lrow])
            if enriched_monopole:
                offset = na*nh*nr if delta_only_enrichment else 2*na*nh*nr
                exact = coefficients[..., offset+a]
                epsilon = jnp.where(h == monopole_harmonic, exact-moment, 0.)
                enriched = radial+epsilon[:, :, None]*g_radial[:, lrow, None, :]
                # A zero enrichment retains the incumbent arithmetic exactly.
                radial = jnp.where(epsilon[:, :, None] == 0., radial, enriched)
                moment = jnp.where(epsilon == 0., moment, exact)
            angle = angular[:, a, h]
            return (delta + radial*angle[:, None, :],
                    comp + moment[:, :, None]*g_radial[:, lrow, None, :]*angle[:, None, :]), None
        return jax.lax.scan(add, (zero, zero), steps_, unroll=1)[0]

    fourier_kernel = jax.jit(shard_map(ft_local, mesh=mesh,
        in_specs=(qspec, P(XY, None, None, None), P(XY, None, None, None),
                  P(XY, None, None), P(), P(), P()),
        out_specs=(qspec, qspec), check_vma=False))

    def onsite_local(coefficients, metric, steps_, degree_rows):
        coeff = coefficients.reshape(coefficients.shape[0], mu, na, nh, nr)
        zero = jnp.zeros((coeff.shape[0], mu, mu), jnp.complex128)
        def add(acc, ah):
            a, h = ah
            values = coeff[:, :, a, h]
            term = jnp.einsum('qmr,rt,qnt->qmn', jnp.conj(values), metric[degree_rows[h]], values)
            return acc + term, None
        return jax.lax.scan(add, zero, steps_, unroll=1)[0]

    onsite_kernel = jax.jit(shard_map(onsite_local, mesh=mesh,
        in_specs=(qspec, P(), P(), P()), out_specs=qspec, check_vma=False))

    if compensated_body or delta_only_enrichment:
        def onsite_cross_local(coefficients, metric, cross_metric, steps_, degree_rows, moment_row=None):
            panels = 1 if delta_only_enrichment else 2
            density = coefficients[..., :panels*na*nh*nr] if enriched_monopole else coefficients
            coeff = density.reshape(coefficients.shape[0], mu, panels, na, nh, nr)
            if enriched_monopole:
                interpolated = jnp.einsum('qmar,r->qma', coeff[:, :, 0, :, monopole_harmonic], tables['moments'][0])
                epsilon = coefficients[..., panels*na*nh*nr:]-interpolated
            zero = jnp.zeros((coeff.shape[0], mu, mu), jnp.complex128)
            def add(acc, ah):
                a, h = ah
                row = degree_rows[h]
                delta = coeff[:, :, 0, a, h]
                term = jnp.einsum('qmr,rt,qnt->qmn', jnp.conj(delta), metric[row], delta)
                if delta_only_enrichment:
                    result = acc + term
                else:
                    smooth = coeff[:, :, 1, a, h]
                    cross = jnp.einsum('qmr,rt,qnt->qmn', jnp.conj(smooth), cross_metric[row], delta)
                    result = acc + term + cross + jnp.conj(jnp.swapaxes(cross, -2, -1))
                if enriched_monopole:
                    def enrich(value):
                        b_cross = jnp.einsum('qmr,r->qm', delta, moment_row)
                        extra = jnp.conj(epsilon[:, :, a, None])*b_cross[:, None, :]
                        return value+extra+jnp.conj(jnp.swapaxes(extra, -2, -1))
                    result = jax.lax.cond((h == monopole_harmonic)&jnp.any(epsilon[:, :, a] != 0.),
                                          enrich, lambda value: value, result)
                return result, None
            return jax.lax.scan(add, zero, steps_, unroll=1)[0]
        specs = (qspec, P(), P(), P(), P(), P()) if enriched_monopole else (qspec, P(), P(), P(), P())
        onsite_cross_kernel = jax.jit(shard_map(onsite_cross_local, mesh=mesh,
            in_specs=specs, out_specs=qspec, check_vma=False))

    def fourier_tile(t, coefficients):
        start = int(t)*gt
        vectors = np.zeros((Qp, gt, 3), dtype=np.float64)
        n = max(0, min(gt, kg.shape[1]-start))
        vectors[:st.Q, :n] = kg[:, start:start+n]
        lengths = np.linalg.norm(vectors, axis=-1)
        cos_theta = np.divide(vectors[..., 2], lengths, out=np.ones_like(lengths), where=lengths > 0)
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        phi = np.arctan2(vectors[..., 1], vectors[..., 0])
        if cache is None:
            bessel = np.stack([spherical_jn(l, lengths[..., None]*r)*(w*r*r)
                               for l in degrees], axis=1)
            g_radial = np.einsum('qlgr,lr->qlg', bessel, tables['compensation_shapes'])
        else:
            bessel = np.moveaxis(cache['density'](lengths), 0, 1)
            g_radial = np.moveaxis(cache['compensation'](lengths), 0, 1)
        angle = np.stack([4*np.pi*(-1j)**l*sph_harm_y(l, m, theta, phi)
                          for l, m in harmonics], axis=1)
        phase = np.exp(-1j*np.einsum('qgi,ai->qag', vectors, centers))
        angular = angle[:, None]*phase[:, :, None]*(float(fft_points)/float(cell_volume))
        # Mask every nonphysical G slot, including real q rows' sphere pads.
        index = start+np.arange(gt)
        valid = np.zeros((Qp, gt), bool)
        valid[:st.Q] = index[None, :] < zeta_g.ngk_per_q[:, None]
        angular *= valid[:, None, None, :]
        return fourier_kernel(coefficients, put(bessel, P(XY, None, None, None)),
                              put(angular, P(XY, None, None, None)),
                              put(g_radial, P(XY, None, None)), scan_steps, dr, rows)

    def onsite(coefficients):
        if compensated_body or delta_only_enrichment:
            if enriched_monopole:
                return onsite_cross_kernel(coefficients, difference, neutral, scan_steps, dr, moment_cross)
            return onsite_cross_kernel(coefficients, difference, neutral, scan_steps, dr)
        return onsite_kernel(coefficients, difference, scan_steps, dr)

    diagnostics = {} if cache is None else {
        key: cache[key] for key in ('points', 'maximum_wavevector', 'table_bytes', 'retained_spline_bytes',
                                   'max_density_validation_error',
                                   'max_compensation_validation_error', 'validation_points')}
    return dict(rhs=provider_rhs, fourier_tile=fourier_tile, onsite=onsite,
                radial_tables=tables if delta_only_enrichment else None,
                exact_monopole_offset=na*nh*nr if delta_only_enrichment else 2*na*nh*nr,
                radial_fourier_diagnostics=diagnostics,
                compensated_body=compensated_body,
                moment_enrichment='served_monopole' if enriched_monopole else 'none',
                body_metric='compensated' if compensated_body else 'mixed_reciprocal',
                local_cross_policy='onsite_smooth_neutral' if compensated_body else 'plane_wave_smooth_delta')
