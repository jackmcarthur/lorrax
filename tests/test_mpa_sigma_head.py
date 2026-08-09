"""The MPA q -> 0 head, and the driver seam that reads it.

WHAT THIS FILE IS ABOUT.  ``gw.mpa.sigma_head`` claims something strong:
that the multipole head Sigma is the SHIPPED two-point head kernel, called
once per complex pole, with no new quadrature and no approximation.  A
claim like that is either exact or it is a plausible number, so every cell
below is a bit-identity or a named refusal, and each has a FALSE case
constructed beside it.

THE ONE CLAIM THAT IS DELIBERATELY NOT PINNED HERE, because it cannot be:
the energy UNIT of the store's pole axis.  ``test_the_z0_gate_is_blind_to
_the_pole_axis_unit`` measures that blindness rather than leaving a reader
to assume the gate covers it -- a gate that cannot fail for the thing it
is quoted as checking is the failure mode this project has paid for most.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from file_io import mpa_store
from gw.head_correction import HeadGNParams, compute_ppm_head_sigma_diag
from gw.mpa import sigma_head


# ---------------------------------------------------------------------------
#  Fixtures: a band window, and a head axis on a real store
# ---------------------------------------------------------------------------

def _window(seed=20260809, nk=3, nb=5, nw=7):
    """A small (omega, k, band) window with the Sigma kernel's arguments."""
    rng = np.random.default_rng(seed)
    enk = np.sort(rng.normal(0.0, 1.0, size=(nk, nb)), axis=1)
    return dict(
        omega_grid_ry=np.linspace(-1.5, 1.5, nw),
        enk_ry=enk,
        efermi_ry=0.13,
        n_occ=2,
        cell_volume=270.107,
        nk_tot=64,
    )


def _head_dict(Omega_p, B_p, z=None, w=None, vhead=None):
    """The shape ``mpa_store.read_head_poles`` returns, without a file."""
    Om = np.asarray(Omega_p, dtype=np.complex128)
    Bp = np.asarray(B_p, dtype=np.complex128)
    if z is None:
        z = np.concatenate([[0.0], np.linspace(0.3, 3.0, 2 * Om.size - 1)])
        z = np.asarray(z, dtype=np.complex128) + 0.2j
        z[0] = 0.0
    z = np.asarray(z, dtype=np.complex128)
    if w is None:
        w = np.sum(Bp / (z[:, None] - Om) - Bp / (z[:, None] + Om), axis=1)
    return {"n_p": int(Om.size), "z": z, "w": np.asarray(w),
            "Omega_p": Om, "B_p": Bp, "vhead": vhead, "written_utc": "n/a"}


# ---------------------------------------------------------------------------
#  1. The identification, as a bit-identity
# ---------------------------------------------------------------------------

def test_a_single_real_pole_is_the_godby_needs_head_bit_for_bit():
    """THE SHARPEST AVAILABLE CHECK: the n_p = 1, Gamma -> 0 limit IS the
    two-point head.

    ``compute_ppm_head_sigma_diag`` regularises its real pole with a
    retarded ``eta``; an MPA pole at ``Omega = omega_h - i*eta`` puts the
    SAME number in the SAME slot.  So this is not "agrees to a tolerance",
    it is ``np.array_equal`` -- and if the mapping ever picks up a factor,
    a sign or a second regulator, exact equality is what notices.
    """
    win = _window()
    R_h, omega_h, eta = 0.8123, 0.9111, 1.0e-6
    gn = HeadGNParams(omega_h_sq=omega_h ** 2, omega_h=omega_h,
                      B_h=2.0 * omega_h * R_h, R_h=R_h, wc_head_0=0.0,
                      wc_head_iwp=0.0, vc0=0.0, omega_p=omega_h)
    reference = compute_ppm_head_sigma_diag(gn, eta=eta, **win)

    head = _head_dict([omega_h - 1j * eta], [R_h])
    got = sigma_head.mpa_head_sigma_diag(head, **win)

    assert got.shape == reference.shape == (7, 3, 5)
    assert np.array_equal(got, reference)


def test_a_complex_pole_reproduces_the_written_out_self_energy_bit_for_bit():
    """The other half of the identification: the WIDTH lands in eta.

    The expression on the right is the complex-pole head Sigma written out
    from the module docstring -- ``f/(delta + Omega) + (1-f)/(delta -
    Omega)`` -- with no reference to the shipped kernel at all.  Bit
    equality is what says the kernel's ``-i eta`` / ``+i eta`` pair is the
    fourth-quadrant pole and not merely close to it.
    """
    win = _window()
    a_p, gamma_p, B_p = 0.7314159, 0.0312, 0.4231 - 0.1177j
    Omega_p = a_p - 1j * gamma_p
    got = sigma_head.mpa_head_sigma_diag(_head_dict([Omega_p], [B_p]), **win)

    f = np.zeros(win["enk_ry"].shape[1])
    f[: win["n_occ"]] = 1.0
    delta = (win["omega_grid_ry"][:, None, None]
             - (win["enk_ry"] - win["efermi_ry"])[None, :, :])
    want = (B_p / (win["cell_volume"] * win["nk_tot"])) * (
        f[None, None, :] / (delta + Omega_p)
        + (1.0 - f[None, None, :]) / (delta - Omega_p))
    assert np.array_equal(got, want)


def test_the_false_case_a_pole_reflected_into_the_upper_half_plane():
    """RED TWIN of the two cells above, and it is refused rather than
    merely different.

    ``Im Omega > 0`` reaches the kernel as a NEGATIVE eta and evaluates
    the advanced self-energy under the retarded one's name -- a finite,
    smooth, wrong Sigma.  The store refuses to hold such a pole at write
    time; this checks the reader does not have to trust that.
    """
    win = _window()
    with pytest.raises(ValueError) as exc:
        sigma_head.mpa_head_sigma_diag(
            _head_dict([0.73 + 0.03j], [0.4]), **win)
    assert "Im Omega_p > 0" in str(exc.value)
    assert "advanced self-energy" in str(exc.value)


def test_a_pole_with_a_non_positive_real_part_is_refused_by_name():
    """The guard nothing else in the tree makes.

    ``write_head_axis`` checks the imaginary part and not the real one.
    ``Re Omega <= 0`` swaps which side of the Fermi level each term of the
    kernel describes; a pruned pole (``B_p = 0``) is exempt because it
    contributes nothing, and that exemption is exercised here too.
    """
    win = _window()
    with pytest.raises(ValueError, match="Re Omega_p <= 0"):
        sigma_head.mpa_head_sigma_diag(
            _head_dict([-0.5 - 0.01j], [0.4]), **win)
    # ...and a PRUNED pole at the same place costs nothing and is allowed.
    got = sigma_head.mpa_head_sigma_diag(
        _head_dict([-0.5 - 0.01j, 0.8 - 0.01j], [0.0, 0.4]), **win)
    only = sigma_head.mpa_head_sigma_diag(
        _head_dict([0.8 - 0.01j], [0.4]), **win)
    assert np.array_equal(got, only)


def test_the_head_is_a_sum_over_poles_and_the_sum_is_the_whole_content():
    """Linearity, which is the lemma the per-pole loop rests on.

    An n_p-pole head must equal the sum of n_p one-pole heads.  It is not
    bit-exact -- floating-point addition is not associative and the loop
    accumulates in pole order -- so this is the one tolerance cell in the
    file, and the tolerance is at the rounding scale rather than at a
    physics scale.
    """
    win = _window()
    Om = np.array([0.41 - 0.02j, 1.30 - 0.11j, 2.75 - 0.60j])
    Bp = np.array([0.5 - 0.1j, -0.25 + 0.3j, 0.125 + 0.05j])
    total = sigma_head.mpa_head_sigma_diag(_head_dict(Om, Bp), **win)
    piecewise = sum(
        sigma_head.mpa_head_sigma_diag(_head_dict(Om[i:i + 1], Bp[i:i + 1]),
                                       **win)
        for i in range(Om.size))
    assert np.allclose(total, piecewise, rtol=0, atol=1e-18)


# ---------------------------------------------------------------------------
#  2. The z = 0 gate, and exactly how far it reaches
# ---------------------------------------------------------------------------

def test_the_z0_gate_reproduces_the_stored_sample_on_a_real_fit():
    """THE UNITS GATE, run against poles a REAL fit produced.

    A store whose ``head_w`` was synthesised from its own poles would make
    this a tautology, so the samples here come from an independent
    three-pole ``W_c`` and the poles come from ``gw.mpa.pade_fit`` --
    the producer's own solver, guards and residue refit.  The gate then
    asks the question the driver seam asks: does the model, as
    ``sigma_head`` spells it, reproduce the sample the store kept beside
    the poles?  ``z = 0`` is the first point of the insulator protocol's
    near line, exactly.
    """
    import jax.numpy as jnp

    from gw.mpa import pade_fit, sampling

    n_p = 3
    z = sampling.double_parallel_grid(n_p, 3.0, energy_unit="Ry")
    assert z[0] == 0.0, "the insulator protocol's first near-line sample"
    Om_true = np.array([0.6 - 0.05j, 1.4 - 0.20j, 2.6 - 0.55j])
    B_true = np.array([-0.30 + 0.02j, -0.15 - 0.01j, -0.05 + 0.00j])
    w_c = np.sum(B_true / (z[:, None] - Om_true)
                 - B_true / (z[:, None] + Om_true), axis=1)

    Om, B, _ = pade_fit.fit_mpa_poles(jnp.asarray(w_c), jnp.asarray(z), n_p)
    head = _head_dict(np.asarray(Om), np.asarray(B), z=z, w=w_c)

    resid = sigma_head.head_sample_residual(head)
    assert resid["z0_index"] == 0
    assert resid["z0_rel_residual"] < 1e-10, resid
    assert resid["max_rel_residual"] < 1e-10, resid


def test_the_gate_catches_a_head_that_stored_W_instead_of_W_minus_v():
    """THE FALSE CASE the gate exists for, and it is a real one.

    ``build_head_axis`` documents that ``head_w`` is the CORRELATION part
    ``W_head - v_head``; a producer that stored ``W_head`` would write a
    file that passes every shape, dtype and readiness check in the format.
    On silicon ``v_head`` is ~3300 Ry, so the ABSOLUTE miss is three
    orders of magnitude; the relative one saturates near 1 because the
    offset dominates the scale it is measured against, which is still six
    orders above the pipeline's tolerance.
    """
    Om = np.array([0.6 - 0.05j, 1.4 - 0.20j])
    Bp = np.array([-0.30 + 0.02j, -0.15 - 0.01j])
    good = _head_dict(Om, Bp)
    bad = dict(good, w=good["w"] + 3303.748102)

    from gw.mpa_pipeline import HEAD_SAMPLE_REL_TOL

    assert sigma_head.head_sample_residual(good)["max_rel_residual"] < 1e-12
    got = sigma_head.head_sample_residual(bad)
    assert got["max_residual"] > 1e3
    assert got["max_rel_residual"] > 0.5
    assert got["max_rel_residual"] > 1e5 * HEAD_SAMPLE_REL_TOL


def test_the_z0_gate_is_blind_to_the_pole_axis_unit():
    """THE LIMIT OF THE GATE, MEASURED rather than reasoned about.

    At ``z = 0`` the model is ``-2 sum_p B_p/Omega_p``, a ratio in which a
    common rescaling of the pole axis cancels identically -- and every
    other sample rescales its own ``z`` along with the poles.  So a store
    fitted in Hartree and read as Rydberg passes this gate EXACTLY, which
    is why ``mpa_pole_energy_unit`` is a deck key and why the pipeline
    prints the f-sum residual (which scales as the square) beside it.
    """
    Om = np.array([0.6 - 0.05j, 1.4 - 0.20j])
    Bp = np.array([-0.30 + 0.02j, -0.15 - 0.01j])
    head_ry = _head_dict(Om, Bp)
    head_ha = _head_dict(Om / 2.0, Bp / 2.0, z=head_ry["z"] / 2.0,
                         w=head_ry["w"])

    assert sigma_head.head_sample_residual(head_ry)["max_rel_residual"] < 1e-14
    assert sigma_head.head_sample_residual(head_ha)["max_rel_residual"] < 1e-14
    # Identical at z = 0 to the last bit, which is the blindness itself.
    assert (sigma_head.head_model_at(head_ry, 0.0)
            == sigma_head.head_model_at(head_ha, 0.0))
    # And the Sigma they produce is NOT identical -- which is the damage.
    win = _window()
    assert not np.allclose(
        sigma_head.mpa_head_sigma_diag(head_ha, **win),
        sigma_head.mpa_head_sigma_diag(head_ry, **win))


def test_the_hartree_conversion_scales_both_poles_and_residues():
    """One factor, applied to both arrays, and it is the SAME factor.

    ``[B_p] = [W_c] * [z]``, so a Hartree store read as Hartree must give
    exactly the Sigma of the doubled poles read as Rydberg.  Reading the
    same store as Rydberg gives a different answer, which is the FALSE
    case this pins.
    """
    win = _window()
    Om_ha = np.array([0.3 - 0.025j, 0.7 - 0.10j])
    B_ha = np.array([-0.15 + 0.01j, -0.075 - 0.005j])
    ha = _head_dict(Om_ha, B_ha)
    ry = _head_dict(Om_ha * 2.0, B_ha * 2.0)

    assert np.array_equal(
        sigma_head.mpa_head_sigma_diag(ha, pole_energy_unit="Ha", **win),
        sigma_head.mpa_head_sigma_diag(ry, pole_energy_unit="Ry", **win))
    assert not np.allclose(
        sigma_head.mpa_head_sigma_diag(ha, pole_energy_unit="Ha", **win),
        sigma_head.mpa_head_sigma_diag(ha, pole_energy_unit="Ry", **win))


def test_an_unknown_pole_unit_refuses_rather_than_defaulting():
    win = _window()
    with pytest.raises(ValueError) as exc:
        sigma_head.mpa_head_sigma_diag(
            _head_dict([0.8 - 0.01j], [0.4]), pole_energy_unit="eV", **win)
    assert "'eV'" in str(exc.value) and "Ha" in str(exc.value)


def test_the_parsers_legal_set_and_the_conversion_table_are_one_set():
    """Two lists of unit spellings is one list too many.

    ``gw_config`` owns the parse-time set (so a deck typo is caught
    without the config layer importing the Sigma stage) and
    ``sigma_head`` owns the factors.  They must name the same units or a
    deck can name a unit nothing knows how to convert.
    """
    from gw.gw_config import POLE_ENERGY_UNIT_SPELLINGS, coerce_pole_energy_unit

    assert set(POLE_ENERGY_UNIT_SPELLINGS) == set(sigma_head.POLE_ENERGY_UNITS)
    assert coerce_pole_energy_unit("ry") == "Ry"
    assert coerce_pole_energy_unit("HA") == "Ha"
    with pytest.raises(ValueError, match="mpa_pole_energy_unit"):
        coerce_pole_energy_unit("hartree")


def test_the_two_spellings_of_the_model_agree():
    """``B/(z-W) - B/(z+W)`` against ``2 W B/(z^2 - W^2)``.

    The factor of two between the two forms is exactly the ``R_h = B_h /
    (2 omega_h)`` the two-point head fit carries, and getting it backwards
    halves or doubles the whole q -> 0 head.
    """
    Om = np.array([0.6 - 0.05j, 1.4 - 0.20j])
    Bp = np.array([-0.30 + 0.02j, -0.15 - 0.01j])
    head = _head_dict(Om, Bp)
    for z in (0.0, 0.31, 0.5 + 0.1j, 2.0 - 0.7j):
        even = complex(np.sum(2.0 * Om * Bp / (z ** 2 - Om ** 2)))
        assert abs(sigma_head.head_model_at(head, z) - even) < 1e-14


# ---------------------------------------------------------------------------
#  3. The driver seam's refusals, exercised through the pipeline
# ---------------------------------------------------------------------------

def _fit_store(tmp_path, *, n_p=2, n_q=2, n_mu=3, head=True, name="fit.h5",
               labels=(None,)):
    path = str(tmp_path / name)
    mpa_store.allocate_fit_store(path, n_q=n_q, n_mu=n_mu, n_p=n_p,
                                 energy_unit="Ry", screening_content="W_c")
    rng = np.random.default_rng(7)
    for q in range(n_q):
        cols = list(range(n_mu))
        a = np.sort(rng.uniform(0.2, 3.0, size=(n_p, n_mu, n_mu)), axis=0)
        g = rng.uniform(0.01, 0.5, size=a.shape)
        mpa_store.write_fit_block(
            path, q, cols, a - 1j * g,
            rng.normal(size=a.shape) + 1j * rng.normal(size=a.shape),
            {"condition": np.ones((n_mu, n_mu)),
             "backward_error": np.full((n_mu, n_mu), 1e-12)})
    mpa_store.finalize_fit_store(path)
    if head:
        # Two head sets, differing in their pole, is the shape the store
        # takes while the velocity-commutator sign is an open decision.
        for i, lab in enumerate(labels):
            mpa_store.allocate_head_axis(path, n_p=n_p, label=lab)
            Om = np.array([0.6 - 0.05j, 1.4 - 0.20j])[:n_p] * (1.0 + 0.2 * i)
            Bp = np.array([-0.30 + 0.02j, -0.15 - 0.01j])[:n_p]
            h = _head_dict(Om, Bp)
            mpa_store.write_head_axis(path, h["z"], h["w"], Om, Bp,
                                      vhead=3303.748102 + 0j, label=lab)
    return path


def _config(path, *, nk_tot, head_label="as_shipped"):
    from gw.gw_config import QPSolver

    return types.SimpleNamespace(
        do_screened=True,
        qp_solver=QPSolver.ONE_SHOT_DFT,
        compute_mode=types.SimpleNamespace(value="mpa", ppm_model=None),
        paths=types.SimpleNamespace(mpa_fit_file=path,
                                    sigma_omega_h5_file="sigma_mnk.h5"),
        ppm=types.SimpleNamespace(regularization_ev=0.25,
                                  window_edge_factor=4.0,
                                  fermi_reference="midgap"),
        sigma_quadrature_config=types.SimpleNamespace(
            target_error=1e-6, max_nodes=64, crossing_eps_q=1e-3,
            crossing_max_nodes=500, use_shipped_tables=True),
        omega_grid_ry=np.linspace(-1.0, 1.0, 5),
        mpa_pole_energy_unit="Ry",
        mpa_head_label=head_label,
    ), types.SimpleNamespace(nk_tot=nk_tot)


def test_a_headless_store_refuses_by_name_before_any_pole_is_integrated(
        tmp_path):
    """THE CHEAP REFUSAL, AND IT MUST COME FIRST.

    A store with no ``__mpahead`` group is a complete, finalized,
    perfectly readable file whose Sigma would simply be missing the q -> 0
    term at every frequency.  The pipeline reads the head BEFORE the pass
    loop precisely so this costs milliseconds, and passing ``wfns=None``
    is how that ordering is measured: if the head read ever moved below
    the pass loop this cell would raise ``AttributeError`` on ``None``
    instead of the refusal.
    """
    from gw.mpa_pipeline import compute_mpa_sigma_pipeline

    path = _fit_store(tmp_path, head=False)
    config, meta = _config(path, nk_tot=2)
    with pytest.raises(ValueError) as exc:
        compute_mpa_sigma_pipeline(
            wfns=None, sig_x=None, sig_h=None, config=config, meta=meta,
            mesh_xy=None, band_slices=None, wfn=None, sym=None,
            input_dir=str(tmp_path), print_fn=lambda *_a, **_k: None)
    msg = str(exc.value)
    assert "read_head_poles" in msg
    assert "__mpahead" in msg
    assert "finite, smooth and wrong" in msg


def test_a_wedge_shaped_store_refuses_by_name_in_the_pass_loop(tmp_path):
    """The other store refusal, reached through the same entry point.

    ``n_q = 2`` against a full zone of ``nk_tot = 8`` is a wedge, and
    unfolding a POLE FIELD is not the operation that unfolds W.  Same
    ``wfns=None`` argument as above: the refusal has to come before the
    psi padding or this cell reports an AttributeError.
    """
    from gw.mpa_pipeline import compute_mpa_sigma_pipeline

    path = _fit_store(tmp_path, n_q=2)
    config, meta = _config(path, nk_tot=8)
    with pytest.raises(NotImplementedError) as exc:
        compute_mpa_sigma_pipeline(
            wfns=None, sig_x=None, sig_h=None, config=config, meta=meta,
            mesh_xy=None, band_slices=None, wfn=None, sym=None,
            input_dir=str(tmp_path), print_fn=lambda *_a, **_k: None)
    msg = str(exc.value)
    assert "refuse" not in msg.lower()[:10]      # it names the situation
    assert "n_q=2" in msg and "n_k_tot=8" in msg
    assert "unfold_isdf_operator" in msg
    assert "exp(+Gamma*tau)" in msg


def test_a_two_point_mode_cannot_borrow_the_multipole_pipeline(tmp_path):
    """The entry condition, read the opposite way from ``ppm_pipeline``'s.

    A GN run reaching here would build its Sigma_c from whatever store the
    deck named -- a complete, finite answer for a screening it never
    computed.
    """
    from gw.mpa_pipeline import compute_mpa_sigma_pipeline

    path = _fit_store(tmp_path)
    config, meta = _config(path, nk_tot=2)
    config.compute_mode = types.SimpleNamespace(value="gn_ppm", ppm_model="gn")
    with pytest.raises(NotImplementedError, match="two-point plasmon-pole"):
        compute_mpa_sigma_pipeline(
            wfns=None, sig_x=None, sig_h=None, config=config, meta=meta,
            mesh_xy=None, band_slices=None, wfn=None, sym=None,
            input_dir=str(tmp_path), print_fn=lambda *_a, **_k: None)


def test_self_consistency_with_a_frozen_pole_store_refuses(tmp_path):
    """A QSGW loop over a screening that cannot be refitted is not QSGW."""
    from gw.gw_config import QPSolver
    from gw.mpa_pipeline import compute_mpa_sigma_pipeline

    path = _fit_store(tmp_path)
    config, meta = _config(path, nk_tot=2)
    config.qp_solver = QPSolver.SELF_CONSISTENT
    with pytest.raises(NotImplementedError) as exc:
        compute_mpa_sigma_pipeline(
            wfns=None, sig_x=None, sig_h=None, config=config, meta=meta,
            mesh_xy=None, band_slices=None, wfn=None, sym=None,
            input_dir=str(tmp_path), print_fn=lambda *_a, **_k: None)
    assert "fitted once" in str(exc.value)
    assert "one_shot_dft" in str(exc.value)


def test_a_deck_naming_no_fit_store_refuses_by_name(tmp_path):
    """The under-specified deck, refused at the config layer.

    ``screening_requests_for(mpa)`` asks the screening stage for nothing,
    so there is no W anywhere in the run to fall back to -- which is
    exactly why "no store named" cannot be treated as "use the default".
    """
    from gw.gw_config import ComputeMode, refuse_missing_mpa_fit_store
    from gw.mpa_pipeline import compute_mpa_sigma_pipeline

    config, meta = _config(None, nk_tot=2)
    with pytest.raises(ValueError) as exc:
        refuse_missing_mpa_fit_store(config, context="a test")
    assert "mpa_fit_file" in str(exc.value)
    assert "a test" in str(exc.value)
    # RED TWIN: every other mode is allowed to name none, and gets None.
    other = types.SimpleNamespace(
        compute_mode=ComputeMode.GN_PPM,
        paths=types.SimpleNamespace(mpa_fit_file=None))
    assert refuse_missing_mpa_fit_store(other) is None
    # And the pipeline restates it rather than trusting the entry check.
    with pytest.raises(ValueError, match="mpa_fit_file"):
        compute_mpa_sigma_pipeline(
            wfns=None, sig_x=None, sig_h=None, config=config, meta=meta,
            mesh_xy=None, band_slices=None, wfn=None, sym=None,
            input_dir=str(tmp_path), print_fn=lambda *_a, **_k: None)


def test_the_deck_chooses_which_head_set_and_a_missing_one_refuses(tmp_path):
    """``mpa_head_label`` reaches the store, and names the set it read.

    A store may carry several q -> 0 head sets while the sign of the
    nonlocal velocity commutator is an open owner decision, and the two
    conventions move the head pole by ~3 eV.  So the deck must be able to
    say which, the run must SAY which it used, and asking for one that is
    not there must be a refusal rather than a silent fall back to the
    default -- which would be the whole hazard, since the default set is
    the one the decision is about.
    """
    from gw.mpa_pipeline import compute_mpa_sigma_pipeline

    path = _fit_store(tmp_path, labels=(None, "commutator_flipped"))

    # The label reaches the store: two sets, two different heads.
    shipped = mpa_store.read_head_poles(path)
    flipped = mpa_store.read_head_poles(path, label="commutator_flipped")
    assert shipped["label"] == "as_shipped"
    assert flipped["label"] == "commutator_flipped"
    assert not np.allclose(shipped["Omega_p"], flipped["Omega_p"])
    win = _window()
    assert not np.allclose(sigma_head.mpa_head_sigma_diag(shipped, **win),
                           sigma_head.mpa_head_sigma_diag(flipped, **win))

    # ...and the run announces the set it actually read.
    lines = []
    # nk_tot = 8 against the store's n_q = 2 makes the WEDGE refusal the
    # thing that stops the run, which is after the head read and its
    # announcement -- so the log line is produced and then the pipeline
    # halts without ever needing a wavefunction bundle.
    config, meta = _config(path, nk_tot=8, head_label="commutator_flipped")
    with pytest.raises(NotImplementedError):     # the wedge refusal, later
        compute_mpa_sigma_pipeline(
            wfns=None, sig_x=None, sig_h=None, config=config, meta=meta,
            mesh_xy=None, band_slices=None, wfn=None, sym=None,
            input_dir=str(tmp_path), print_fn=lambda s: lines.append(str(s)))
    assert any("commutator_flipped" in ln for ln in lines), lines

    # RED TWIN: a label the store does not carry is refused, and the
    # message says which sets it DOES carry.
    config, meta = _config(path, nk_tot=8, head_label="no_such_set")
    with pytest.raises(ValueError) as exc:
        compute_mpa_sigma_pipeline(
            wfns=None, sig_x=None, sig_h=None, config=config, meta=meta,
            mesh_xy=None, band_slices=None, wfn=None, sym=None,
            input_dir=str(tmp_path), print_fn=lambda *_a, **_k: None)
    msg = str(exc.value)
    assert "no_such_set" in msg
    assert "__mpahead__commutator_flipped" in msg


def test_an_empty_head_label_reads_the_default_set(tmp_path):
    """An unset deck key is "the store's one head set", not "no set".

    ``head_group_name(None)`` and ``head_group_name('as_shipped')`` are
    the same bare group, which is what makes every store written before
    labels existed still readable.  The config layer normalises an unset
    key to the empty string, so the pipeline maps that to ``None``.
    """
    path = _fit_store(tmp_path)
    config, _ = _config(path, nk_tot=2, head_label="")
    assert (config.mpa_head_label or None) is None
    assert mpa_store.read_head_poles(path, label=None)["label"] == "as_shipped"


def test_the_three_deck_keys_survive_a_real_parse(tmp_path):
    """The keys through the ACTUAL parser, not through a stub config.

    Every cell above hands the pipeline a ``SimpleNamespace``, which
    proves the pipeline reads the right attribute and proves nothing at
    all about whether a deck can set it.  This is the other half: a real
    file through ``LorraxConfig.from_input_file``, including the path
    resolution (relative to the DECK's directory, like every other path
    key) and the emptiness convention that keeps "not set" distinct from
    "the deck's own directory".
    """
    from gw.gw_config import LorraxConfig, refuse_missing_mpa_fit_store

    deck = tmp_path / "cohsex.in"
    deck.write_text(
        "[cohsex]\n"
        "compute_mode = mpa\n"
        "mpa_fit_file = fit.h5\n"
        "mpa_pole_energy_unit = Ha\n"
        "mpa_head_label = commutator_flipped\n"
        "nband = 10\n")
    cfg = LorraxConfig.from_input_file(str(deck), print_fn=lambda *a, **k: None)
    assert cfg.compute_mode.value == "mpa"
    assert cfg.paths.mpa_fit_file == str(tmp_path / "fit.h5")
    assert cfg.mpa_pole_energy_unit == "Ha"
    assert cfg.mpa_head_label == "commutator_flipped"
    assert refuse_missing_mpa_fit_store(cfg) == str(tmp_path / "fit.h5")

    # RED TWIN 1: the mode without the store, through the same parser.
    bare = tmp_path / "bare.in"
    bare.write_text("[cohsex]\ncompute_mode = mpa\nnband = 10\n")
    cfg2 = LorraxConfig.from_input_file(str(bare), print_fn=lambda *a, **k: None)
    assert cfg2.paths.mpa_fit_file is None, (
        "an unset optional path must stay unset — joining '' onto the deck "
        "directory yields a real path that is not a file")
    with pytest.raises(ValueError, match="mpa_fit_file"):
        refuse_missing_mpa_fit_store(cfg2)

    # RED TWIN 2: a mistyped unit is caught at PARSE time, not at the head.
    typo = tmp_path / "typo.in"
    typo.write_text("[cohsex]\ncompute_mode = mpa\nmpa_fit_file = fit.h5\n"
                    "mpa_pole_energy_unit = hartree\n")
    with pytest.raises(ValueError, match="mpa_pole_energy_unit"):
        LorraxConfig.from_input_file(str(typo), print_fn=lambda *a, **k: None)


def test_a_named_store_that_does_not_exist_refuses_before_the_stage_runs(
        tmp_path):
    from gw.mpa_pipeline import compute_mpa_sigma_pipeline

    config, meta = _config(str(tmp_path / "absent.h5"), nk_tot=2)
    with pytest.raises(FileNotFoundError) as exc:
        compute_mpa_sigma_pipeline(
            wfns=None, sig_x=None, sig_h=None, config=config, meta=meta,
            mesh_xy=None, band_slices=None, wfn=None, sym=None,
            input_dir=str(tmp_path), print_fn=lambda *_a, **_k: None)
    assert "unscreened" in str(exc.value)
