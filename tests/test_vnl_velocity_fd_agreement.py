"""The numeric arm validates the analytic one -- and now something runs it.

THE POLICY FAILURE THIS CLOSES.  ``psp.get_dipole_mtxels`` has carried a
``--vnl-mode numeric`` finite-difference path for as long as it has
carried the analytic one, and the numeric path exists for exactly one
reason: to check that the analytic path -- the only arm that ever
produces a ``dipole.h5`` -- computes the derivative it claims to.  That
check existed in principle and had never once been run.  It was found in
2026-08 that the two arms had been sitting on OPPOSITE SIGNS: the
numeric branch formed ``-(V(k+h) - V(k-h)) / 2h`` while
``compute_vnl_velocity_cart`` returns ``+dV_NL/dK``, so the validation
arm disagreed with the production arm by a factor of -1 and nothing
noticed, because nothing compared them.  A validation path that is never
executed is not a validation path; it is a second implementation with
its own bugs and no consequences.

So this file is deliberately CHEAP and deliberately NOT opt-in.  One
k-point, two bands, one central difference per Cartesian direction --
seven projector builds in total -- so that it can sit in the default
suite forever without anybody wanting to mark it slow.

WHAT IS COMPARED, AND AT WHICH LEVEL.  The production spellings live in
``psp.get_dipole_mtxels`` (``compute_vnl_matrix_from_setup`` for the
finite difference, ``compute_vnl_velocity_cart`` for the analytic), and
that module calls ``runtime.initialize_communicator_stack()`` at import,
which refuses wherever the FFI host library is not built.  A gate that
imported it would be skipped on every developer box -- which is the
failure this file exists to end, reintroduced in a new costume.  Both
production functions are thin wrappers over ``vnl_ops.vnl_matrix`` and
``vnl_ops.vnl_velocity_matrix``, so the comparison is made against those
directly and runs everywhere.  The driver-level twin, on a real
pseudopotential, is a cluster leg.

WHY THE TOLERANCE IS A DIRECTION AND NOT A DECIMAL.  ``_interp_with_deriv``
does not differentiate the interpolant; it carries a SEPARATE physical
``dG/dq`` table and returns that as the q-tangent, deliberately, "to
avoid the O(dq^2) bias that afflicts the forward-slope derivative of the
linear interpolant" (its own docstring).  That is the right choice for
physics and it means the analytic derivative and a finite difference of
the interpolated form factor are NOT THE SAME QUANTITY: they differ by
the interpolation bias, which on this synthetic fixture's coarse
``dq = 0.02`` table is a few percent and does not shrink with ``h``.
Measured here across two k and two band counts: cosine similarity
+0.927 to +0.980, magnitude ratio 0.883 to 0.913, both flat in ``h``
from 1e-3 down to 1e-6.  A tight elementwise tolerance would therefore
be a flake generator, and a loose elementwise tolerance would be blind.

The quantity that IS sharp is the direction.  A sign inversion -- the
defect that actually happened -- drives the cosine to about -0.95 while
leaving every magnitude untouched, so a threshold anywhere in the wide
gap between -0.95 and +0.93 separates them with enormous margin.  The
gate asserts direction tightly and magnitude only loosely, which is an
honest statement of what a finite difference against a
separately-tabulated derivative can actually prove.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from psp import vnl_ops

#: Cosine similarity between the finite difference and the analytic
#: velocity.  Measured +0.927 .. +0.980; an inverted sign gives the
#: negative of that.  0.5 sits in the middle of a gap nothing physical
#: occupies, so this catches a sign with ~0.4 of margin on either side
#: while never tripping on the interpolation bias.
MIN_COSINE = 0.5

#: |FD| / |analytic|.  Measured 0.883 .. 0.913 -- the interpolation bias
#: above, not an error.  The band is wide because its job is to catch a
#: derivative that is wrong by a FACTOR (a missing chain rule, a dropped
#: 2, a units slip), not to pin the bias.
MIN_SCALE, MAX_SCALE = 0.5, 2.0

#: Central-difference step in Cartesian |K| units.  Flat in h over three
#: decades here, so this is not a convergence knob; it is just small.
H_CART = 1.0e-5


def _both_derivatives(ik=0, nb=2, h=H_CART, invert_analytic=False):
    """(analytic, finite-difference) velocity matrices at one k.

    Returns two ``(3, nb, nb)`` complex arrays: ``vnl_velocity_matrix``,
    which is what the production analytic path returns, and the central
    difference of ``vnl_matrix``, which is what the production numeric
    path forms.  ``invert_analytic`` is the red twin's hook and is the
    exact defect that shipped: one arm negated relative to the other.
    """
    from tests.test_dipole_vnl_velocity_sign import _fixture, _vnl_setup

    setup = _vnl_setup()
    psi, gv, gmask, _bidx, kvecs, bvec, blat = _fixture()
    B = np.asarray(bvec, dtype=np.float64) * float(blat)
    Binv = np.linalg.inv(B)
    ket = jnp.asarray(psi[ik] * gmask[ik][None, None, :])[:nb]

    def V(kvec):
        kd = vnl_ops.build_vnl_kdata_traced(
            jnp.asarray(kvec), jnp.asarray(gv[ik]), setup, compute_dZ=False)
        ns_e = int(kd.E_super.shape[0])
        return np.asarray(jax.device_get(
            vnl_ops.vnl_matrix(ket[:, :ns_e], kd.Z, kd.E_super)))

    kd = vnl_ops.build_vnl_kdata_traced(
        jnp.asarray(kvecs[ik]), jnp.asarray(gv[ik]), setup, compute_dZ=True)
    ns_e = int(kd.E_super.shape[0])
    analytic = np.asarray(jax.device_get(vnl_ops.vnl_velocity_matrix(
        ket[:, :ns_e], kd.Z, kd.dZ, kd.E_super)))
    if invert_analytic:
        analytic = -analytic

    fd = np.zeros_like(analytic)
    for ic in range(3):
        step = np.zeros(3, dtype=np.float64)
        step[ic] = h
        dk = step @ Binv                      # Cartesian step -> crystal
        fd[ic] = (V(kvecs[ik] + dk) - V(kvecs[ik] - dk)) / (2.0 * h)
    return analytic, fd


def _cosine_and_scale(analytic, fd):
    a, f = analytic.ravel(), fd.ravel()
    na, nf = np.linalg.norm(a), np.linalg.norm(f)
    assert na > 0.0 and nf > 0.0, (
        "one side of the comparison is identically zero, which makes the "
        "cosine undefined and this file a tautology")
    return float(np.real(np.vdot(a, f)) / (na * nf)), float(nf / na)


# ---------------------------------------------------------------------------
# 1. The gate the failure needed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ik", [0, 1])
def test_finite_difference_agrees_with_the_analytic_velocity(ik):
    """``--vnl-mode numeric`` must validate, not contradict, the analytic arm.

    This is the comparison that had never run.  It is the whole reason
    the numeric path is in the tree, and for as long as nothing executed
    it the two arms were free to drift onto opposite signs -- which they
    did, and stayed there.
    """
    analytic, fd = _both_derivatives(ik=ik)
    cosine, scale = _cosine_and_scale(analytic, fd)

    assert cosine > MIN_COSINE, (
        f"k={ik}: the finite difference of V_NL points at cosine "
        f"{cosine:+.4f} to the analytic velocity.  Below {MIN_COSINE} the "
        f"two arms are not computing the same derivative; at a negative "
        f"cosine one of them is negated, which is the defect this gate "
        f"exists for and which shipped undetected because nothing ran "
        f"this comparison.")
    assert MIN_SCALE < scale < MAX_SCALE, (
        f"k={ik}: |FD| / |analytic| = {scale:.4f}, outside "
        f"[{MIN_SCALE}, {MAX_SCALE}].  The direction is right, so this is "
        f"a magnitude fault -- a dropped chain-rule factor or a units "
        f"slip -- rather than a sign.")


# ---------------------------------------------------------------------------
# 2. The false cases
# ---------------------------------------------------------------------------

def test_an_inverted_analytic_sign_trips_the_gate():
    """RED TWIN, and it reproduces the ACTUAL defect rather than a proxy.

    Negating one arm is precisely what the tree did: the numeric branch
    carried a leading minus that ``compute_vnl_velocity_cart`` does not.
    If this cell ever passes while the gate above also passes, the gate
    has stopped reading the sign.
    """
    analytic, fd = _both_derivatives(invert_analytic=True)
    cosine, scale = _cosine_and_scale(analytic, fd)

    assert cosine < -MIN_COSINE, (
        f"an inverted analytic velocity still agreed in direction "
        f"(cosine {cosine:+.4f}), so the gate above cannot be reading "
        f"the sign at all")
    assert MIN_SCALE < scale < MAX_SCALE, (
        "the inversion must move the DIRECTION and leave the magnitude "
        "alone -- if it moved the magnitude too, this twin would be "
        "passing for the wrong reason and would not prove the cosine is "
        "what caught it")


def test_the_comparison_has_signal_to_compare():
    """The nonlocal velocity must be non-trivial on this fixture.

    A null ``E_super`` or a null ``dZ`` makes both sides vanish; the
    cosine is then undefined and every assertion above is vacuous.
    ``_cosine_and_scale`` refuses that case, and this cell is where the
    refusal is exercised on purpose rather than discovered in a year.
    """
    analytic, fd = _both_derivatives()
    assert np.linalg.norm(analytic) > 1e-8
    assert np.linalg.norm(fd) > 1e-8

    dead = np.zeros_like(analytic)
    with pytest.raises(AssertionError):
        _cosine_and_scale(dead, fd)


# ---------------------------------------------------------------------------
# 3. The driver-level twin, which closes the loop the cells above cannot
# ---------------------------------------------------------------------------
#
# HONEST LIMIT OF EVERYTHING ABOVE.  Those cells compare
# ``vnl_ops.vnl_velocity_matrix`` against a finite difference of
# ``vnl_ops.vnl_matrix``, and so they PIN THE CONVENTION that
# ``psp.get_dipole_mtxels``'s two modes must both meet -- but they do not
# execute either mode.  The defect that shipped lived in the producer's
# numeric branch, one level above these kernels, and re-introducing a
# leading minus there would leave every assertion above green.  This cell
# is the one that would go red, and it needs the FFI.

def _ffi_available():
    try:
        import runtime                                    # noqa: F401
        from ffi.fft import GATE
        from jax.sharding import Mesh
        GATE.enforce(Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                          ('x', 'y')))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ffi_available(),
                    reason="the production numeric/analytic branches live in "
                           "psp.get_dipole_mtxels, which calls "
                           "initialize_communicator_stack() at module scope "
                           "and refuses without the FFI host library")
def test_the_producers_two_modes_agree_at_the_kernel_boundary():
    """``compute_vnl_velocity_cart`` vs a difference of
    ``compute_vnl_matrix_from_setup`` -- the producer's own two spellings.

    Same comparison as the cells above, made through the functions
    ``--vnl-mode analytic`` and ``--vnl-mode numeric`` actually call, so
    that a sign re-introduced in the producer is caught here even though
    the kernels underneath it are unchanged.
    """
    from psp.get_dipole_mtxels import (compute_vnl_matrix_from_setup,
                                       compute_vnl_velocity_cart)
    from tests.test_dipole_vnl_velocity_sign import (GRID, NGK, NS, _fixture,
                                                     _vnl_setup)

    setup = _vnl_setup()
    psi, gv, gmask, _bidx, kvecs, bvec, blat = _fixture()
    B = np.asarray(bvec, dtype=np.float64) * float(blat)
    Binv = np.linalg.inv(B)
    ik, nb = 0, 2
    gm = gmask[ik]

    # These two take an FFT-BOX state and gather it onto the G-sphere
    # themselves, so the fixture's sphere-layout psi has to be scattered
    # back into a box first.  Only the NGK physical columns are placed:
    # the pad columns carry no amplitude and the mask zeroes them on the
    # way out, which is the same thing the cells above rely on.
    nx, ny, nz = GRID
    box = np.zeros((nb, NS, nx, ny, nz), dtype=np.complex128)
    for i in range(NGK):
        gx, gy, gz = (int(c) for c in gv[ik][i])
        box[:, :, gx, gy, gz] = psi[ik][:nb, :, i]
    wfn_k = jnp.asarray(box)

    analytic = np.asarray(jax.device_get(compute_vnl_velocity_cart(
        wfn_k, gv[ik], kvecs[ik], setup, g_mask=gm)))
    fd = np.zeros_like(analytic)
    for ic in range(3):
        step = np.zeros(3, dtype=np.float64)
        step[ic] = H_CART
        dk = step @ Binv
        Vp = np.asarray(jax.device_get(compute_vnl_matrix_from_setup(
            wfn_k, gv[ik], kvecs[ik] + dk, setup, g_mask=gm)))
        Vm = np.asarray(jax.device_get(compute_vnl_matrix_from_setup(
            wfn_k, gv[ik], kvecs[ik] - dk, setup, g_mask=gm)))
        fd[ic] = (Vp - Vm) / (2.0 * H_CART)

    cosine, scale = _cosine_and_scale(analytic, fd)
    assert cosine > MIN_COSINE, (
        f"the producer's numeric and analytic spellings point at cosine "
        f"{cosine:+.4f}.  A negative value is the 2026-08 defect back "
        f"again: the numeric branch negating what the analytic branch "
        f"returns.")
    assert MIN_SCALE < scale < MAX_SCALE
