"""The q -> 0 head of W at complex frequency, from the dipole matrix
elements.

STAGING LOCATION, same as the rest of ``gw/mpa/`` -- see the package
docstring.  This module is the companion of
:mod:`gw.mpa.chi0_resolvent`: that one builds the BODY of chi0 at a
strip sample by naming the transitions in the ISDF centroid basis, and
this one builds the HEAD and the WINGS of the same object at the same
samples by naming the same transitions with plane-wave weights instead.

WHY THE HEAD NEEDS ITS OWN ROUTE AT ALL
---------------------------------------
An ISDF centroid basis has no G = 0 row.  The centroids are real-space
points; ``mu`` indexes a position, not a reciprocal-lattice vector, so
there is no element of ``W_q(mu, nu)`` that IS the macroscopic
screening.  Every finite-q column of the multipole store is therefore
complete and the q -> 0 head slot is empty, and the emptiness is not
visible in any norm: the body is a perfectly good matrix that simply
does not contain the long-wavelength channel.

The substitutes are worse than the gap.  ``SI_MPA_FIRST_LIGHT``
measured two of them: the basis-independent trace invariant
``Tr[chi0 W]`` puts its dominant pole at 27-29 eV with a 6-8 eV width
and moves it by 14 eV over an n_p scan (it is reporting the whole
particle-hole continuum, faithfully), and the |B|-weighted median over
matrix elements sits at 64-73 eV (a typical centroid element is a
short-wavelength local-field channel).  Projecting on the softest
Coulomb eigenvector works at finite q -- 17.85 eV at (0,0,1/4), stable
to 162 meV across n_p = 3..8 -- and fails at Gamma, where it returns
35.38 eV, because at Gamma there is no 8*pi/|q+G|**2 divergence for the
top eigenvector to single out.  That 35.38 eV row is the hole this
module fills.

THE PHYSICS, IN ONE LINE
------------------------
The pair density's long-wavelength limit is a dipole:

    rho_vck(q, G=0) = <c, k+q| e^{i q.r} |v, k>  ->  i q . d_vck

because the zeroth-order term vanishes by orthogonality.  Substituting
that into the same Lehmann sum the body uses gives the head as a
TENSOR-valued transition sum over the SAME unified kernel

    K_z(Delta) = 1/(z - Delta) - 1/(z + Delta) = -2 Delta/(Delta**2 - z**2)

with dipole weights in place of pair densities -- see
:func:`head_tensor`.  The wings are the cross term, one power of q, and
:func:`wing_tensors` builds them.  :func:`macroscopic_head_tensor` then
does the q -> 0 block Dyson solve, whose entire content is that the
wings' local-field contribution is ALSO a quadratic form in q-hat, so
the whole q -> 0 problem collapses onto a single 3x3 tensor that the
shipped mini-BZ averager already knows how to consume.

THE CONVENTION, AND WHERE IT IS VALIDATED
-----------------------------------------
The dipole convention is NOT invented here.  It is
``common.chi_from_dipole.compute_S_omega``'s, the shipped q -> 0 head
of the plasmon-pole path, which reads ``dipole.h5`` and returns the
same 3x3 object this module calls ``A(z)``.  Concretely: the file's
``dipole_cart[a, k, m, n]`` is a velocity matrix element and its
``deltaE[k, m, n] = E_m - E_n``, so the position dipole is
``d = dipole_cart / deltaE`` up to a phase that cancels in every
bilinear this module forms.  ``compute_S_omega`` never forms ``d`` --
it divides by ``Delta * (z**2 - Delta**2)`` in one step -- and
:func:`head_tensor` forms it and multiplies by ``K_z``.  The two are
the same rational function of ``Delta``, so the agreement is a
floating-point statement and not a physics one; that is exactly what
makes it a usable bridge gate, and
``tests/test_mpa_head_dipole.py::test_static_bridge_against_s_tensor``
is the measurement.

WHAT THE ENERGIES COME FROM.  ``deltaE`` is a redundant copy: the
symmetry task-4 audit measured it bit-identical to the outer difference
of the WFN eigenvalues on all four decks.  :func:`head_tensor`
therefore takes ``deltaE`` when the caller has the file open and
``delta_from_energies`` when it has the WFN, and the two are the same
numbers.  A caller that has BOTH should pass the WFN energies, because
those are the ones the body route
(:func:`gw.mpa.chi0_resolvent.chi0_resolvent`) uses and a head fitted
against different energies than its body is a bug with no shape error.
"""

import numpy as np

import jax
import jax.numpy as jnp

from gw.mpa import sample_plan

__all__ = [
    "head_tensor",
    "wing_tensors",
    "macroscopic_head_tensor",
    "delta_from_energies",
    "transition_dipoles",
    "head_prefactor",
    "head_channel_samples",
    "plasma_frequency_ry",
    "fsum_residual",
    "head_fsum_from_transitions",
    "isotropy_report",
]

#: The three placements of the complex conjugate in the head's
#: bilinear, ``A_ab = sum_t (d_a) (d_b) K_z`` with conjugation applied
#: to the left factor, the right factor, or neither.
#:
#: ``"left"`` is ``compute_S_omega``'s spelling and ``"right"`` is the
#: one that falls out of ``chi0_{mu nu} = sum rho_mu conj(rho_nu) K_z``
#: with ``rho_0 = i q.d``.  THEY ARE THE SAME PHYSICS and the module
#: says so on purpose: the two differ by a transpose, the head enters
#: everything downstream only as the quadratic form
#: ``qhat_a A_ab qhat_b``, and ``qhat_a qhat_b`` is symmetric, so the
#: antisymmetric part is annihilated identically.  A head-only test
#: therefore CANNOT discriminate them, and pretending otherwise would
#: be a gate that passes for the wrong reason.  ``"none"`` is a real
#: bug and the head does see it.  The conjugation that only the WINGS
#: can see is :func:`wing_tensors`' ``crossed``.
CONJ_MODES = ("left", "right", "none")


def delta_from_energies(enk, nelec):
    """``Delta[k, c, v] = E_ck - E_vk`` from the WFN band energies.

    Vertical transitions only: the head is the q -> 0 limit, so the
    conduction and valence k index are the SAME k.  That is what makes
    this route cheap enough to run on a laptop and it is also why it
    cannot be folded into :mod:`gw.mpa.chi0_resolvent`, whose whole
    apparatus is the ``k - q`` roll.
    """

    e = np.asarray(enk, dtype=np.float64)
    n_occ = int(nelec)
    if not 0 < n_occ < e.shape[1]:
        raise ValueError(
            f"GATE occupied_window: nelec={nelec} does not split the "
            f"{e.shape[1]}-band window into a nonempty valence and a "
            f"nonempty conduction set. FALSE case: 0 < nelec < n_bands.")
    return e[:, n_occ:, None] - e[:, None, :n_occ]


def transition_dipoles(dipole_cart, delta, nelec):
    """``d[a, k, c, v] = dipole_cart[a, k, c, v] / Delta[k, c, v]``.

    The position dipole, up to the factor ``i`` that the velocity form
    carries.  Every use in this module is bilinear in ``d`` with one
    conjugated factor, so that phase cancels identically and is not
    carried; :func:`head_tensor`'s ``d* d`` and :func:`wing_tensors`'
    ``d* M`` both see ``|i|**2 = 1``.  Where the conjugate GOES is not
    free, and :data:`CONJ_MODES` is where that is decided.
    """

    v = np.asarray(dipole_cart)
    n_occ = int(nelec)
    n_b = v.shape[2]
    blk = v[:, :, n_occ:n_b, :n_occ]
    d = np.asarray(delta, dtype=np.float64)
    if blk.shape[1:] != d.shape:
        raise ValueError(
            f"GATE dipole_delta_shape: the (c, v) dipole block is "
            f"{blk.shape[1:]} but Delta is {d.shape}.  The two must be "
            f"cut from the same band window; a head fitted against a "
            f"different window than its energies has no shape error to "
            f"announce it.")
    return blk / d[None, :, :, :]


def _kernel_weights(delta_flat, z_values):
    """``K_z(Delta)`` at every sample, ``(n_z, n_trans)`` complex128.

    ``sample_plan.KERNEL_FACTOR`` is read rather than written so the
    head, the body and the quadrature substrate cannot drift apart on
    the sign of the kernel -- the same discipline
    ``chi0_resolvent._accumulate_block`` keeps.
    """

    d = jnp.asarray(delta_flat, dtype=jnp.float64)
    z = jnp.asarray(z_values, dtype=jnp.complex128)
    return sample_plan.KERNEL_FACTOR * d[None, :] / (
        d[None, :] ** 2 - z[:, None] ** 2)


def head_prefactor(cell_volume, nk_tot, nspin, nspinor):
    """``2 / (Omega * N_k * n_spin * n_spinor)`` -- stated, not implied.

    This is ``compute_S_omega``'s ``4 / (Omega N_k n_spin n_spinor)``
    halved, and the half is the ``2`` that ``K_z`` carries and the S
    route does not: ``K_z(Delta) = 2 Delta / (z**2 - Delta**2)`` while
    the S route's weight is ``1 / (Delta (z**2 - Delta**2))``, and
    ``d = v / Delta`` supplies the remaining ``Delta**2``.  Writing the
    two prefactors as one function is the only way the identity stays
    true when somebody edits one of them.

    The ``n_spinor`` division is a spin-degeneracy statement, not a
    normalisation: on a bispinor deck the occupied spinor bands already
    carry both spins, so the factor 2 that a spin-restricted sum needs
    must come back out.  Si's committed fixture is nspinor = 2 with
    nelec = 8, and the f-sum rule of :func:`plasma_frequency_ry` is what
    would catch this being wrong by a factor of two.
    """

    return 2.0 / (float(cell_volume) * float(nk_tot)
                  * float(max(int(nspin), 1)) * float(max(int(nspinor), 1)))


def _conj_pair(d_flat, conj_mode):
    """The (left, right) dipole factors for one of :data:`CONJ_MODES`."""

    if conj_mode not in CONJ_MODES:
        raise ValueError(
            f"GATE conj_mode: {conj_mode!r} is not one of {CONJ_MODES}. "
            f"FALSE case: 'left' is the convention; the other two exist "
            f"only so a red twin can name which wrong one it ran.")
    if conj_mode == "left":
        return jnp.conj(d_flat), d_flat
    if conj_mode == "right":
        return d_flat, jnp.conj(d_flat)
    return d_flat, d_flat


def head_tensor(dipole_cart, delta, z_values, *, nelec, cell_volume,
                nk_tot, nspin, nspinor, conj_mode="left"):
    """``A(z)``, the q -> 0 head as a 3x3 TENSOR at every sample.

    Returns ``(n_z, 3, 3)`` complex128 with

        A[z]_{ab} = pref * sum_{cvk} conj(d_a) d_b K_z(Delta),

    ``pref`` from :func:`head_prefactor`, so that the head of chi0 at
    small q is ``chi0_00(q, z) = |q|**2 * qhat . A(z) . qhat`` and the
    head of the RPA dielectric matrix is ``1 - 8*pi * qhat . A . qhat``
    -- the ``|q|**2`` cancelling the Coulomb ``8*pi/|q|**2`` exactly,
    which is why the head is finite and direction-dependent rather than
    divergent.

    THE OUTPUT IS A TENSOR AND STAYS ONE.  For a cubic crystal it is
    ``a(z) * I`` and a scalar would do; the committed Si fixture is
    isotropic to 5.3e-7 in the diagonal spread and 5.0e-10 in the
    off-diagonal, which is a fact about silicon and not about this
    code.  The committed hBN fixture, run through the same call, gives
    diag(-0.2189, -0.2189, -0.1289) at z = 0 -- eps_in-plane 6.50
    against eps_c-axis 4.24, a 48 % anisotropy.  A scalar head is
    therefore not a simplification on hBN, it is a 48 % error, and the
    two materials next in the queue after silicon are hBN and TiO2.
    The scalar is available downstream, once, where the mini-BZ average
    contracts the tensor with the sampled directions -- see
    :func:`head_channel_samples`.

    THE SAMPLE AXIS IS BROADCAST HERE AND SCANNED IN THE BODY, and the
    asymmetry is deliberate: the body's weighted copy is
    ``(n_z, n_trans, n_mu)`` and reaches gigabytes, while this one is
    ``(n_z, n_trans, 3)`` and does not.  The transition axis for the
    head is ``n_k * n_v * n_c`` vertical transitions -- 26 624 on the Si
    fixture -- so the whole build is a few megabytes and one einsum.
    """

    delta = np.asarray(delta, dtype=np.float64)
    d = transition_dipoles(dipole_cart, delta, nelec)
    d_flat = jnp.asarray(d.reshape(3, -1), dtype=jnp.complex128)
    left, right = _conj_pair(d_flat, conj_mode)
    w = _kernel_weights(delta.reshape(-1), z_values)
    pref = head_prefactor(cell_volume, nk_tot, nspin, nspinor)
    out = jnp.einsum("at,zt,bt->zab", left, w, right, optimize=True)
    return pref * out


def wing_tensors(dipole_cart, delta, pair_amplitudes, z_values, *, nelec,
                 cell_volume, nk_tot, nspin, nspinor, crossed=False):
    """``(Y(z), Z(z))``, the two wings, at one power of ``q``.

    ``pair_amplitudes`` is ``M[k, c, v, mu]`` on the SAME vertical
    (k, c, v) transition set the head uses -- the q = 0 case of
    ``chi0_resolvent._pair_amplitude``, i.e. ``sum_s conj(psi_c) psi_v``
    at each centroid, with no k roll because q = 0 makes the roll the
    identity.

    Returns ``Y`` of shape ``(n_z, 3, n_mu)`` and ``Z`` of shape
    ``(n_z, n_mu, 3)``:

        Y[z]_{a nu} = pref * sum_t d_a  conj(M_nu) K_z(Delta)
        Z[z]_{mu b} = pref * sum_t M_mu conj(d_b)  K_z(Delta)

    so that ``chi0_{0 nu}(q, z) = i |q| qhat_a Y_{a nu}`` and
    ``chi0_{mu 0}(q, z) = -i |q| Z_{mu b} qhat_b``.  One power of q on
    each side, against two on the head and none on the body: that
    counting is the whole reason the local fields survive the q -> 0
    limit, since the Coulomb head supplies ``|q|**-2`` and the two wings
    supply ``|q|**2`` between them.

    ``Z`` is ``conj(Y)`` transposed in the (a, nu) indices ONLY when
    ``K_z`` is real, i.e. on the static and imaginary cells; on the
    strip it is an independent object and is built as one.  Assuming
    the relation off the imaginary axis is a plausible optimisation and
    a wrong one, so it is not taken.

    ``crossed=True`` swaps which factor of each wing carries the
    conjugate.  This is the conjugation red twin that the HEAD cannot
    run: the head's two spellings differ by a transpose that its
    quadratic form annihilates (see :data:`CONJ_MODES`), while the
    wings enter :func:`macroscopic_head_tensor` as ``Y (...) Z`` with
    no symmetrisation, so a crossed wing changes the local-field
    correction and hence ``eps_M``.  The twin is the only place in this
    module where the dipole's conjugation is observable, so it is the
    only place the convention is actually pinned.
    """

    delta = np.asarray(delta, dtype=np.float64)
    d = transition_dipoles(dipole_cart, delta, nelec)
    d_flat = jnp.asarray(d.reshape(3, -1), dtype=jnp.complex128)
    d_y = jnp.conj(d_flat) if crossed else d_flat
    d_z = d_flat if crossed else jnp.conj(d_flat)
    M = jnp.asarray(np.asarray(pair_amplitudes), dtype=jnp.complex128)
    n_mu = M.shape[-1]
    M_flat = M.reshape((-1, n_mu))
    if M_flat.shape[0] != d_flat.shape[1]:
        raise ValueError(
            f"GATE wing_transition_axis: the pair amplitudes carry "
            f"{M_flat.shape[0]} transitions and the dipoles carry "
            f"{d_flat.shape[1]}.  Both must be the SAME vertical (k, c, "
            f"v) set in the same order, or the wing pairs a dipole with "
            f"another transition's pair density.")
    w = _kernel_weights(delta.reshape(-1), z_values)
    pref = head_prefactor(cell_volume, nk_tot, nspin, nspinor)
    Y = pref * jnp.einsum("at,zt,tn->zan", d_y, w, jnp.conj(M_flat),
                          optimize=True)
    Z = pref * jnp.einsum("tm,zt,bt->zmb", M_flat, w, d_z, optimize=True)
    return Y, Z


def macroscopic_head_tensor(A, Y, Z, chi0_body, v_body):
    """``A_M(z)``: the head tensor with the local fields folded in.

    THE Q -> 0 BLOCK DYSON SOLVE, and the reason it has an answer.  In
    block form with the head/wing/body split, the dielectric matrix is

        eps = [[1 - v_h chi0_hh,  -v_h chi0_hw],
               [   -v_b chi0_wh,  1 - v_b chi0_bb]]

    and ``v_h = 8*pi/|q|**2`` diverges while ``v_b`` does not.  Counting
    powers of ``|q|``: the head block is finite (``|q|**-2`` against
    ``|q|**2``), the upper-right block diverges as ``|q|**-1``, and the
    lower-left block vanishes as ``|q|**1``.  The macroscopic dielectric
    function is the Schur complement

        eps_M(qhat, z) = eps_hh - eps_hw (eps_bb)^{-1} eps_wh

    in which those two powers of q cancel EXACTLY, leaving

        eps_M = 1 - 8*pi * qhat . [A + Y (eps_bb)^{-1} v_b Z] . qhat.

    So the local-field correction is itself a quadratic form in qhat,
    the whole q -> 0 problem collapses onto one 3x3 tensor, and the
    mini-BZ averager that already consumes ``A`` consumes ``A_M``
    unchanged.  That is the design decision this function exists to
    carry: the head Dyson solve is not a new solver, it is a rank-3
    correction to a tensor.

    THE FORM IS BILINEAR IN qhat, NOT SESQUILINEAR.  ``chi0(z)`` is not
    Hermitian off the real axis, so ``qhat`` is contracted without a
    conjugate on either side and ``A_M`` is symmetric-in-the-limit
    rather than Hermitian.  Conjugating one side would be invisible at
    z = 0 and wrong on the strip, which is why the static bridge alone
    cannot certify this function and the GN probe is a separate gate.

    Parameters
    ----------
    A : (n_z, 3, 3)
        From :func:`head_tensor`.
    Y, Z : (n_z, 3, n_mu), (n_z, n_mu, 3)
        From :func:`wing_tensors`.
    chi0_body : (n_z, n_mu, n_mu)
        The body, in ``compute_chi0``'s normalisation -- exactly what
        :func:`gw.mpa.chi0_resolvent.chi0_resolvent` returns at
        ``q_int = (0, 0, 0)``.
    v_body : (n_mu, n_mu)
        The Coulomb operator in the same basis, as ``solve_w`` uses it.

    Returns
    -------
    (n_z, 3, 3) complex128
    """

    A = jnp.asarray(A, dtype=jnp.complex128)
    Y = jnp.asarray(Y, dtype=jnp.complex128)
    Z = jnp.asarray(Z, dtype=jnp.complex128)
    X = jnp.asarray(chi0_body, dtype=jnp.complex128)
    V = jnp.asarray(v_body, dtype=jnp.complex128)
    n_mu = V.shape[0]
    eye = jnp.eye(n_mu, dtype=jnp.complex128)
    eps_bb = eye - jnp.einsum("mn,znk->zmk", V, X, optimize=True)
    # (eps_bb)^{-1} v_b Z, solved rather than inverted: the body block
    # is the same matrix ``solve_w`` factors, and forming its inverse to
    # multiply a 3-column right-hand side is the one place this path
    # could become the expensive one.
    rhs = jnp.einsum("mn,znb->zmb", V, Z, optimize=True)
    corr = jnp.linalg.solve(eps_bb, rhs)
    return A + jnp.einsum("zan,znb->zab", Y, corr, optimize=True)


def head_channel_samples(head_tensor_z, geometry, kgrid, *,
                         analytic_sphere=True, nsamples=2 ** 18,
                         method="auto", qmc_reps=10):
    """``(v_head, W_head(z))``: the bare and screened q -> 0 head slots.

    Runs the SHIPPED mini-BZ averager once per sample:
    ``vcoul``'s 3D bulk ``q0_average`` with ``S_cart`` set to this
    sample's head tensor, which is the same call
    ``gw.head_correction.resolve_head_sample`` makes at omega = 0 and at
    the plasmon-pole probe.  The head channel is therefore not a new
    q -> 0 convention beside the production one; it is the production
    one evaluated at n_omega frequencies instead of two.

    WHY THIS DRAW AND NOT THE BODY-SLOT DRAW.  The landed body-head
    machinery (``vcoul.minibz.build_v_head_miniBZ_fn_3d``) averages
    ``8*pi/|K + dq|**2`` over a centrosymmetric Monte-Carlo cloud and
    injects the result at every slot attaining ``argmin |q+G|**2``.
    That machinery EXCLUDES Gamma by construction -- ``v_qG_table``
    skips q with ``min |q+G|**2 < 1e-12``, because that head is the
    separate rank-1 term this module is computing -- and the exclusion
    is not bookkeeping.  MEASURED on the Si fixture: the centrosymmetric
    draw evaluated at ``K = 0`` returns 3206.16 Ry against the deck's
    BerkeleyGW anchor ``vhead = 3303.748102``, low by 3.0 %, because at
    K = 0 the integrable ``1/|dq|**2`` singularity sits INSIDE the cell
    where plain Monte Carlo estimates it badly; at finite K it does not,
    which is why the body slots never needed the correction.  The
    Baldereschi-Tosatti analytic-sphere estimator that
    ``analytic_sphere=True`` selects returns 3304.0806 Ry, agreeing with
    the anchor to 1.0e-4 relative.  So the q -> 0 head uses the
    analytic-sphere draw, the body slots use the centrosymmetric one,
    and the two are different estimators of different integrals rather
    than a convention that drifted.

    THE ANISOTROPIC AVERAGE DOES NOT FACTOR.  ``q0_average`` contracts
    ``qhat . A . qhat`` inside the sample loop, so
    ``W_head = < 8*pi / (|dq|**2 (1 - 8*pi dqhat.A.dqhat)) >`` is the
    average of a direction-dependent quotient.  On a cubic crystal that
    equals ``v_head / eps_M`` and the tensor could have been contracted
    first; on hBN it does not, and contracting first is the scalar
    shortcut this module refuses.

    THE ONE PLACE THIS ROUTE CANNOT USE ``q0_average`` VERBATIM, and
    why.  With ``analytic_sphere=True`` that function adds the
    Baldereschi-Tosatti sphere term to ``vc0_mean`` and NOT to
    ``wcoul0``: the bare head is corrected for the integrable
    ``1/|dq|**2`` singularity and the screened head is a plain
    Monte-Carlo mean over the same draw.  For the plasmon-pole path
    that is harmless, because it consumes the two numbers separately.
    For the multipole path it is not, because the object fitted is
    ``W - v``, and two estimators of the same bare integral differ by a
    CONSTANT that does not decay with ``z``.  MEASURED on the Si
    fixture at the production draw (2**18 points, 10 repetitions):
    ``vc0_mean = 3304.0806`` against ``wcoul0(S = 0) = 3291.8854``, a
    12.195 Ry offset -- 0.37 % of the bare head, but 11.8 % of the
    screened head at omega = 0, and 100 % of what the fit sees at large
    ``|z|`` where ``W_c`` should be going to zero.

    The correction taken here is the multiplicative one:
    ``W_head = W_plain * (v_sphere / v_plain)``, with ``v_head`` the
    sphere-corrected bare head.  For an isotropic ``eps_M`` this is
    EXACT -- the direction-independent factor comes out of both
    integrals, so the ratio is the ratio of the two estimators of the
    same bare integral and nothing else -- and it makes
    ``W_c(z) -> 0`` as ``|z| -> infinity`` identically.  For an
    anisotropic ``eps_M`` it is an approximation: it assumes the
    angular average of ``1/eps_M`` over the sphere term equals its
    average over the whole cell.  That is stated rather than hidden,
    the error is bounded by the head's anisotropy times the angular
    difference between a sphere and a nearly spherical Voronoi cell,
    and the exact replacement is an angularly resolved sphere term,
    which is specified here and not built.

    Returns
    -------
    v_head : complex
        ``< 8*pi/|dq|**2 >`` over the mini-BZ, in Ry, sphere-corrected
        when ``analytic_sphere``.  Sample independent: this is the
        multipole model's ``v`` term, and on the Si fixture it
        reproduces the deck's BerkeleyGW anchor ``vhead = 3303.748102``
        to 1.0e-4 relative.
    W_head : (n_z,) complex128
        ``< v(dq) / eps_M(dqhat, z) >`` at each sample, in Ry, on the
        same estimator as ``v_head``.  The correlation part the fit
        consumes is ``W_head - v_head``.
    """

    from vcoul.bulk_3d import Bulk3D

    kernel = Bulk3D()
    A = np.asarray(jax.device_get(head_tensor_z))
    if A.ndim != 3 or A.shape[1:] != (3, 3):
        raise ValueError(
            f"GATE head_tensor_shape: expected (n_z, 3, 3) and got "
            f"{A.shape}.  A scalar head would be accepted silently by "
            f"the averager and would be wrong on every non-cubic cell.")
    zero = jnp.zeros((3, 3), dtype=jnp.complex128)
    common = dict(nsamples=nsamples, method=method, qmc_reps=qmc_reps,
                  analytic_sphere=analytic_sphere)
    # One S = 0 call recovers BOTH estimators of the bare head: the
    # sphere-corrected ``vc0_mean`` and the plain mean that the screened
    # branch actually uses, since ``eps_M = 1`` at S = 0.
    v_sphere, v_plain = kernel.q0_average(geometry, kgrid, S_cart=zero,
                                          **common)
    ratio = complex(v_sphere) / complex(v_plain)
    w_out = np.empty(A.shape[0], dtype=np.complex128)
    for i in range(A.shape[0]):
        _vc0, w0 = kernel.q0_average(geometry, kgrid,
                                     S_cart=jnp.asarray(A[i]), **common)
        w_out[i] = complex(w0) * ratio
    return complex(v_sphere), w_out


def plasma_frequency_ry(n_elec, cell_volume):
    """``omega_p = sqrt(16 pi n)`` in Ry -- the classical plasma
    frequency of a homogeneous gas at the crystal's valence density.

    The 16 rather than 4 is the Rydberg unit: in Hartree atomic units
    ``omega_p**2 = 4 pi n`` and a Rydberg frequency is twice a Hartree
    one, so the square picks up a factor 4.  Si's committed fixture has
    8 valence electrons in 270.107 bohr**3, giving 1.22015 Ry =
    16.601 eV, which is Silicon's measured bulk plasmon (16.7 eV) to
    0.1 eV -- the f-sum rule is that good, and that is why it is worth
    using as a gate.
    """

    n = float(n_elec) / float(cell_volume)
    return float(np.sqrt(16.0 * np.pi * n))


def fsum_residual(Omega_p, B_p, v_head, n_elec, cell_volume,
                  omega_p_ry=None):
    """The f-sum rule on the FITTED head poles, as a relative residual.

    A free, closed-form gate that costs nothing and tests the fit rather
    than the physics.  As ``z -> infinity`` the multipole model (3) goes
    as ``sum_p 2 Omega_p B_p / z**2`` while the exact head goes as
    ``v_head * omega_p**2 / z**2`` -- because
    ``eps_M -> 1 - omega_p**2/z**2`` by the f-sum rule and
    ``W_c = v/eps_M - v``.  Hence

        sum_p 2 Omega_p B_p = v_head * omega_p**2,
        omega_p**2 = 16 pi n_elec / Omega_cell   (Ry**2)

    exactly, with no fitted quantity on the right.  The residual
    reported here is the relative disagreement.

    WHAT A LARGE RESIDUAL MEANS is ambiguous by design and the caller
    must say which it is, which is what ``omega_p_ry`` is for.  Left
    unset, the right side uses the CLASSICAL plasma frequency and the
    residual conflates two things: whether the fit captured the leading
    asymptote (a method statement) and whether the deck's transition
    manifold saturates the sum rule (a deck statement).  Passing
    ``head_fsum_from_transitions(...)["omega_p_ry"]`` removes the second
    and leaves a pure method number.  The Si fixture needs this: it
    over-saturates by 1.63, so the classical right side is wrong by
    that factor before the fit is even consulted.
    """

    Om = np.asarray(jax.device_get(Omega_p), dtype=np.complex128)
    B = np.asarray(jax.device_get(B_p), dtype=np.complex128)
    lhs = complex(np.sum(2.0 * Om * B))
    wp = (plasma_frequency_ry(n_elec, cell_volume)
          if omega_p_ry is None else float(omega_p_ry))
    rhs = complex(v_head) * wp ** 2
    return {
        "fit_sum_2_Omega_B": lhs,
        "v_head_times_omega_p_squared": rhs,
        "omega_p_ry": wp,
        "rel_residual": float(abs(lhs - rhs) / abs(rhs)),
    }


def head_fsum_from_transitions(dipole_cart, delta, *, nelec, cell_volume,
                               nk_tot, nspin, nspinor, qhat=(1.0, 0.0, 0.0)):
    """The deck's OWN plasma frequency, from the transition manifold.

    ``omega_p**2(qhat) = 16 pi * pref * sum_t |qhat.d|**2 Delta`` is the
    large-``|z|`` asymptote of ``1 - eps_M``, and comparing it with
    :func:`plasma_frequency_ry` says how well the deck's band window and
    pseudopotential saturate the Thomas-Reiche-Kuhn sum rule.  It is a
    property of ``dipole.h5``, not of this module, and it caps what any
    head-channel number computed from that file can claim.

    MEASURED, so that a later reader is not surprised: the committed Si
    fixture gives 21.183 eV against the classical 16.601 eV, i.e. an
    effective 13.03 electrons per cell against 8 -- the manifold
    over-saturates by 1.63.  The committed hBN fixture over-saturates by
    1.22 in plane and 1.07 along c.  The two ratios are different, so
    this is not a constant missing from the convention; it is a property
    of those two decks, and the standing project rule is that the
    nonlocal-pseudopotential contraction is not the place to look for
    it.
    """

    delta = np.asarray(delta, dtype=np.float64)
    d = transition_dipoles(dipole_cart, delta, nelec)
    q = np.asarray(qhat, dtype=np.float64)
    q = q / np.linalg.norm(q)
    proj = np.einsum("a,akcv->kcv", q, d)
    pref = head_prefactor(cell_volume, nk_tot, nspin, nspinor)
    wp2 = 16.0 * np.pi * pref * float(np.sum(np.abs(proj) ** 2 * delta))
    wp_classical = plasma_frequency_ry(nelec, cell_volume)
    return {
        "omega_p_ry": float(np.sqrt(wp2)),
        "omega_p_classical_ry": wp_classical,
        "saturation_ratio": float(wp2 / wp_classical ** 2),
    }


def isotropy_report(A):
    """How far a head tensor is from a multiple of the identity.

    The cubic-symmetry twin's measurement.  Returns the diagonal
    spread and the largest off-diagonal, both relative to the mean
    diagonal, per sample.  On a cubic crystal both are numerical zero
    and on a uniaxial one the diagonal spread is the anisotropy.
    """

    A = np.asarray(jax.device_get(A))
    if A.ndim == 2:
        A = A[None, :, :]
    out = []
    for M in A:
        dg = np.diag(M)
        scale = abs(np.mean(dg)) or 1.0
        off = M - np.diag(dg)
        out.append({
            "diag": dg,
            "diag_spread_rel": float(
                (np.max(np.abs(dg)) - np.min(np.abs(dg))) / scale),
            "max_offdiag_rel": float(np.max(np.abs(off)) / scale),
        })
    return out
