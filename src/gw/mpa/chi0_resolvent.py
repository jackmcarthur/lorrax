"""chi0 at COMPLEX frequency by the exact per-transition resolvent.

STAGING LOCATION, same as the rest of ``gw/mpa/`` -- see the package
docstring.  This module is the ONE cell of ``sample_plan``'s 2x2 table
that neither the static kernel nor the imaginary-axis kernel can reach:
a sample ``z = omega + i*varpi`` with BOTH parts nonzero.

WHY A TRANSITION SUM AND NOT A WEIGHTS CHANGE
---------------------------------------------
The production chi0 (``w_isdf.compute_chi0``) never materialises a
transition.  It builds two imaginary-time Green's functions, multiplies
them in real space, and FFTs to q; the (k, v, c) index is summed away
inside the FFT convolution.  The only per-transition freedom it has is
the function

    g(Delta) = sum_l alpha_l exp(-tau_l Delta),      tau_l REAL, >= 0

because ``_get_chi_minimax_kernel`` casts the node to ``float64``
(``tau_real = jnp.real(t_scalar)``) before it reaches ``build_G_tau``.
That family is a Laplace transform, and the strip kernel is NOT in it:

    K_z(Delta) = -2 Delta / (Delta**2 - z**2)
               = 1/(z - Delta) - 1/(z + Delta)

has a pole at ``Delta = Re z`` on the positive-Delta axis, and
``1/(Delta - z) = int_0^inf exp(-(Delta - z) tau) dtau`` diverges for
``Delta < Re z``.  So no real-tau weight vector represents it: the
manual's own "two sums rule"
(``manual/04_gw_real_space/4.2_two_sums_rule.md``) is exactly the
property that has to be given up to sample the strip.  The theory plan's
damped-tau sweep gives it up differently -- it moves tau onto the
IMAGINARY axis, where the per-transition function becomes
``sin(Delta t)`` -- and that is a real-time G build, not a reweighting
either.  Both routes leave the Laplace family; this one leaves it by
naming the transitions.

WHAT IT COSTS.  The transition axis is ``n_k * n_v * n_c`` and the
contraction is a rank-``n_trans`` update of a ``(mu, mu)`` matrix per
sample, i.e. ``8 * n_mu**2 * n_trans`` flops per (q, z).  The pair
amplitudes do not depend on ``z``, so a whole line of samples rides one
build of ``M``; that is the only economy available and it is taken.

THE CONVENTION, AND WHERE IT IS VALIDATED
-----------------------------------------
The pair amplitude and the k-shift are NOT invented here.  They are the
ones ``bse/bse_w_exact.build_finite_q_data`` already uses, whose
docstring records that the resulting k-sum "is exactly the GW producer's
chi0(q) convolution sum_k Gc_k Gv*_{k+q}" and which
``tests/test_bse_w0_resolvent.py::test_wq_resolvent_matches_restart_finite_q``
holds to ``rel < 1e-6`` against the on-disk ``W_q`` tile.  Concretely,
with ``q`` in integer grid steps and the conduction quantities rolled by
``+q`` on the C-order ``(nkx, nky, nkz)`` k-axis so that slot ``k``
carries the value at ``k - q``::

    M^q_{cvk}(mu) = sum_s conj(psi_c[k-q](s, r_mu)) psi_v[k](s, r_mu)
    D^q_{cvk}     = eps_c[k-q] - eps_v[k]

and NO umklapp Bloch phase -- the production chi0 is a plain periodic
FFT convolution over k and reads the raw stored psi at the wrapped
index.  Applying the design-doc umklapp phase breaks the match (that
mistake is recorded in ``build_finite_q_data``'s docstring at rel_err
0.6-3.2, and it would break this module identically).

THE OUTPUT CONVENTION IS ``compute_chi0``'S, EXACTLY
----------------------------------------------------
This module returns the array ``w_isdf.solve_w`` consumes, in the same
normalisation ``w_isdf.compute_chi0`` returns it in, so that the Dyson
solve is byte-for-byte the same call.  That normalisation carries a
``1/sqrt(n_k)`` that is NOT physics: it is the residue of doing three
``norm='ortho'`` FFTs (k->R on each G, R->q on chi) where the
mathematics wants one plain convolution.  ``_get_chi_minimax_kernel``
computes

    chi_q = (1/sqrt(N)) sum_k conj(Gc_k) Gv_{k+q}

and ``solve_w`` then applies ``2/(sqrt(n_q) n_spin n_spinor)``, so the
two square roots multiply to the physical ``1/n_k``.  ``KERNEL_FACTOR``
(-2) lives inside ``K_z`` on both paths.  :data:`CHI0_ORTHO_NORM` names
the residue rather than leaving it as a bare ``** -0.5``.

THE BRIDGE GATE.  ``K_0(Delta) = -2/Delta`` is the exact value the
static minimax quadrature approximates, so :func:`chi0_resolvent` at
``z = 0`` and ``compute_chi0`` compute the SAME object by two different
routes.  They cannot agree bitwise -- one is a quadrature and the other
is closed form -- and a caller who demands bit-identity is asking for
the quadrature error to be zero.  What they must agree to is the
quadrature's own certified error; :func:`bridge_gate_report` measures
that and is the honest form of the gate.
"""

import numpy as np

import jax
import jax.numpy as jnp

from gw.mpa import evaluator, sample_plan


def chi0_ortho_norm(n_k):
    """The production kernel's FFT-normalisation residue, ``n_k**-0.5``.

    Left over by ``_get_chi_minimax_kernel``'s three ``norm='ortho'``
    FFTs (k->R on each G, R->q on chi) where the mathematics wants one
    plain convolution.  It has a name because it is a convention of the
    production kernel rather than a physical factor: reproduce it and
    ``solve_w`` is the same call on both routes; drop it and every W on
    a 64-k deck is wrong by a factor of 8, silently, because chi0 is
    never printed.
    """

    return float(n_k) ** -0.5


def roll_k_axis(arr, q_int, kgrid):
    """``out[k] = arr[k - q]`` on the wrapped C-order k-axis.  Host numpy.

    Byte-for-byte the shift ``bse_w_exact._roll_k_axis_host`` applies;
    duplicated rather than imported because ``gw.mpa`` must not grow a
    dependency on ``bse`` for four lines of numpy, and because the BSE
    copy is a private helper.  Any change to one is a change to both --
    the convention is shared, and :func:`chi0_resolvent`'s docstring
    says where it is validated.
    """

    nkx, nky, nkz = (int(v) for v in kgrid)
    a = np.asarray(arr)
    tail = a.shape[1:]
    a = a.reshape((nkx, nky, nkz) + tail)
    a = np.roll(a, shift=(int(q_int[0]), int(q_int[1]), int(q_int[2])),
                axis=(0, 1, 2))
    return a.reshape((nkx * nky * nkz,) + tail)


def _pair_amplitude(psi_c, psi_v):
    """``M[k, c, v, mu] = sum_s conj(psi_c[k,c,s,mu]) psi_v[k,v,s,mu]``.

    The same contraction as ``bse.bse_serial.compute_pair_amplitude``;
    see the module docstring for why the convention is not free.
    """

    return jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c), psi_v,
                      optimize=True)


@jax.jit
def _accumulate_block(M, delta, z_values):
    """One k-block's contribution to chi0 at every sample, unnormalised.

    ``M`` is ``(n_k_blk, n_c, n_v, n_mu)`` and ``delta`` is
    ``(n_k_blk, n_c, n_v)``; ``z_values`` is ``(n_z,)`` complex.
    Returns ``(n_z, n_mu, n_mu)``.

    The weight is the ONLY z-dependent factor, which is the whole reason
    the transition axis is worth materialising: ``M`` is built once and
    every sample on both lines rides it.

    THE SAMPLE AXIS IS SCANNED, NOT BROADCAST, and that is a memory
    decision rather than a style one.  The natural
    ``einsum('zt,tm,tn->zmn')`` invites XLA to form the ``(n_z, n_trans,
    n_mu)`` weighted copy -- 3.4 GB at n_z=16, a 16-k block and 1128
    centroids, which is most of a device for an intermediate.  Scanning
    z holds one ``(n_trans, n_mu)`` weighted copy at a time and turns
    the whole thing into n_z plain GEMMs, which is also the shape that
    reaches the fp64 tensor cores.
    """

    n_mu = M.shape[-1]
    Mf = M.reshape((-1, n_mu))                       # (n_trans, n_mu)
    d = delta.reshape((-1,)).astype(jnp.float64)     # (n_trans,)
    Mc = jnp.conj(Mf)

    def _one(_, zj):
        # K_z(Delta) = -2 Delta / (Delta**2 - z**2); ``damped_kernel``'s
        # closed form, inlined because that one is host numpy and this is
        # the device path.  The factor is read from ``sample_plan`` so the
        # two forms cannot drift apart.
        w = sample_plan.KERNEL_FACTOR * d / (d ** 2 - zj ** 2)
        return None, jnp.matmul((Mf * w[:, None]).T, Mc)

    _, out = jax.lax.scan(_one, None, jnp.asarray(z_values,
                                                  dtype=jnp.complex128))
    return out


def chi0_resolvent(psi, enk, slices, z_values, q_int, kgrid, *,
                   k_block=8, energy_reference=0.0):
    """chi0(q; z) by the exact per-transition resolvent, at every ``z``.

    Parameters
    ----------
    psi : (n_k, n_b, n_spinor, n_mu) array
        UN-CONJUGATED psi at the centroids over the full band window --
        i.e. ``wfns.psi_yr`` (or ``psi_xr``; they are the same numbers in
        different layouts, see ``wavefunction_bundle.build_wavefunctions``).
    enk : (n_k, n_b) array
        Band energies in Ry, same window and order as ``psi``.
    slices : BandSlices
        Supplies ``.val`` and ``.cond``; the same object
        ``compute_chi0`` slices with, so the two routes cannot disagree
        about which bands are occupied.
    z_values : (n_z,) complex
        Sample points in Ry.  ``z = 0`` is legal and gives the static
        kernel's own target ``-2/Delta``.
    q_int : (3,) int
        The q point as integer grid steps.
    kgrid : (3,) int
        ``(nkx, nky, nkz)``.
    k_block : int
        k points per device step.  The transition axis is materialised
        one block at a time; the peak device array is
        ``k_block * n_c * n_v * n_mu`` complex128, so this is the
        module's memory dial and nothing else depends on it.
    energy_reference
        Subtracted from every band energy before the difference is
        taken.  Algebraically invariant (only differences enter) and
        present only so a caller can pass ``compute_chi0``'s own
        ``e_ref`` and keep the two calls textually parallel.

    Returns
    -------
    (n_z, n_mu, n_mu) complex128
        In ``compute_chi0``'s normalisation -- feed it straight to
        ``solve_w``.
    """

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "GATE x64_enabled: jax x64 is off, so chi0 would be built in "
            "single precision, where a strip sample near its own pole "
            "loses the imaginary part that makes it well posed. FALSE "
            "case: jax.config.update('jax_enable_x64', True) before "
            "import, or JAX_ENABLE_X64=1 in the environment.")

    psi_h = np.asarray(jax.device_get(psi))
    enk_h = np.asarray(jax.device_get(enk), dtype=np.float64)
    eref = 0.0 if energy_reference is None else float(energy_reference)

    psi_v = psi_h[:, slices.val]
    psi_c = psi_h[:, slices.cond]
    eps_v = enk_h[:, slices.val] - eref
    eps_c = enk_h[:, slices.cond] - eref

    # Slot k carries the conduction quantity at k - q.  See the module
    # docstring: this shift, and the ABSENCE of an umklapp phase, is what
    # matches the production FFT convolution.
    psi_c_q = roll_k_axis(psi_c, q_int, kgrid)
    eps_c_q = roll_k_axis(eps_c, q_int, kgrid)

    n_k = psi_h.shape[0]
    n_mu = psi_h.shape[-1]
    z = jnp.asarray(np.asarray(z_values, dtype=np.complex128))
    acc = jnp.zeros((z.shape[0], n_mu, n_mu), dtype=jnp.complex128)

    for k0 in range(0, n_k, int(k_block)):
        k1 = min(k0 + int(k_block), n_k)
        M = _pair_amplitude(
            jnp.asarray(psi_c_q[k0:k1], dtype=jnp.complex128),
            jnp.asarray(psi_v[k0:k1], dtype=jnp.complex128))
        delta = jnp.asarray(
            eps_c_q[k0:k1][:, :, None] - eps_v[k0:k1][:, None, :],
            dtype=jnp.float64)
        acc = acc + _accumulate_block(M, delta, z)

    return acc * chi0_ortho_norm(n_k)


def bridge_gate_report(chi0_static_production, chi0_static_resolvent, *,
                       quad_rel_error=None):
    """Compare the two routes at ``z = 0``.  Returns a dict of measures.

    THE GATE IS NOT BIT-IDENTITY and cannot be: the production route
    evaluates ``-2/Delta`` through a minimax Laplace quadrature and this
    one evaluates it in closed form.  What the comparison certifies is
    that the two routes agree to the QUADRATURE's error -- i.e. that the
    transition convention, the k-shift, the conjugation and the
    normalisation are the production kernel's, which is the part a
    convention bug would break by orders of magnitude rather than by
    parts in ``1e-6``.

    ``quad_rel_error``, when given, is the minimax solve's own reported
    max relative error on ``1/x``; the report then carries the ratio of
    the measured disagreement to it, which is the number that says
    "quadrature error" rather than "bug" out loud.
    """

    a = np.asarray(jax.device_get(chi0_static_production))
    b = np.asarray(jax.device_get(chi0_static_resolvent))
    if a.shape != b.shape:
        raise ValueError(
            f"bridge_gate_report: shape mismatch {a.shape} vs {b.shape} -- "
            f"the two routes must be compared on the same (q, mu, nu) block.")
    diff = np.abs(a - b)
    scale = np.abs(a)
    denom = float(np.max(scale)) or 1.0
    out = {
        "max_abs_production": float(np.max(scale)),
        "max_abs_diff": float(np.max(diff)),
        "max_rel_diff": float(np.max(diff) / denom),
        "mean_rel_diff": float(np.mean(diff) / denom),
        "ratio_max": float(np.max(np.abs(b)) / denom),
    }
    if quad_rel_error is not None and float(quad_rel_error) > 0.0:
        out["quad_rel_error"] = float(quad_rel_error)
        out["diff_over_quad_error"] = out["max_rel_diff"] / float(quad_rel_error)
    return out


def cost_model(n_k, n_v, n_c, n_mu, n_z, n_q):
    """Flop and byte counts for one full strip build.  Pure arithmetic.

    Reported honestly per theory-plan section B's cost-report rule: the
    logical output count, the actual work, and the per-output Dyson
    solves.  A caller prints this NEXT TO the measured wall so the
    two can be compared -- the resolvent route's price is a finding, not
    an implementation detail.
    """

    n_trans = int(n_k) * int(n_v) * int(n_c)
    # complex128 GEMM: 8 real flops per complex multiply-add.
    flops_per_qz = 8.0 * float(n_mu) ** 2 * float(n_trans)
    return {
        "n_transitions_per_q": n_trans,
        "flops_per_sample_per_q": flops_per_qz,
        "flops_total": flops_per_qz * float(n_z) * float(n_q),
        "store_bytes": 16.0 * float(n_z) * float(n_q) * float(n_mu) ** 2,
        "n_logical_outputs": int(n_z) * int(n_q),
        "n_dyson_solves": int(n_z) * int(n_q),
    }
