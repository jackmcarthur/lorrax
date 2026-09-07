"""``sc_head_update = dft_velocity``: vocabulary, dispatch, and head chain.

Three questions, one per section:

1. **Config** — is ``dft_velocity`` a legal value, does the mandatory rule
   for a fractionally occupied deck accept EITHER metal head mode, and is
   the refusal that catches a metal deck with no head mode intact?
2. **Dispatch** — does the ``dft_velocity`` route reach the
   parallel-transport loader?  It must not: this mode deliberately reads only
   the velocity stage and has no finite-link DeltaH derivative.  Pinned by
   monkeypatching the loader to raise, with the ``parallel_transport`` arm as
   the control that proves the trap is armed.
3. **Head chain** — does the mode run the same S(z)/Drude/wing chain on
   the DFT p-matrix velocity, and does it still rotate that velocity into
   the current QP basis every iteration?  The velocity rotation is the
   whole reason a QSGW iteration differs from the one-shot tool route, so
   it is pinned directly, both as an equality against the rotated velocity
   and as a difference from the unrotated one.

Everything here is at the fixture scale of
``test_qsgw_parallel_transport_head.py`` (a handful of k points and bands),
with no WFN and no artifact on disk.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from common.parallel_transport import build_forward_neighbor_table
from gw import qsgw_head
from gw.gw_config import METAL_HEAD_UPDATES, LorraxConfig
from gw.qsgw_head import (
    build_iteration_head_response,
    head_s_tensor_sharded,
    rotate_velocity_active_to_qp,
)
from gw.sc_iteration import load_head_velocity_source

jax.config.update("jax_enable_x64", True)


def _mesh():
    devices = np.asarray(jax.devices())
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    else:
        devices = devices[:1].reshape(1, 1)
    return Mesh(devices, ("x", "y"))


# ---------------------------------------------------------------------------
# 1. Config: the vocabulary and the widened mandatory rule
# ---------------------------------------------------------------------------

_BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""

# The three keys the fractional-occupation rule already required before
# this mode existed; only the head-mode value is under test below.
_FRACTIONAL = (
    "qp_solver = self_consistent\n"
    "sc_accelerator = linear\n"
    "occ_broadening = 0.13605693122994\n"
)


def _config(tmp_path, extra: str = "", name: str = "head_mode.in"):
    path = tmp_path / name
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: None)


def test_the_metal_head_vocabulary_is_exactly_the_two_modes():
    # One tuple owns the pair; a consumer that spells it out again is the
    # drift this constant exists to prevent.
    assert METAL_HEAD_UPDATES == ("parallel_transport", "dft_velocity")


@pytest.mark.parametrize("mode", METAL_HEAD_UPDATES)
def test_a_fractional_deck_accepts_either_metal_head_mode(tmp_path, mode):
    cfg = _config(tmp_path, _FRACTIONAL + f"sc_head_update = {mode}\n")
    assert cfg.sc.head_update == mode
    assert cfg.screening.occ_broadening_ev > 0.0


def test_dft_velocity_parses_on_an_insulating_deck_too(tmp_path):
    # The head mode is not itself the metal switch; occ_broadening is.
    cfg = _config(tmp_path, "sc_head_update = dft_velocity\n")
    assert cfg.sc.head_update == "dft_velocity"
    assert cfg.screening.occ_broadening_ev == 0.0


def test_a_head_update_cannot_override_head_correction_off(tmp_path):
    with pytest.raises(ValueError, match="does not override"):
        _config(
            tmp_path,
            "qp_solver = self_consistent\n"
            "head_correction = off\n"
            "sc_head_update = dft_velocity\n")


def test_a_fractional_deck_with_the_head_off_is_still_refused(tmp_path):
    # UNCHANGED behaviour: widening the rule to two values must not turn it
    # into no rule at all.
    with pytest.raises(ValueError, match="sc_head_update"):
        _config(tmp_path, _FRACTIONAL + "sc_head_update = off\n")


def test_a_fractional_deck_defaulting_the_head_key_is_still_refused(tmp_path):
    # ...including when the deck simply omits the key (default "off").
    with pytest.raises(ValueError, match="occ_broadening"):
        _config(tmp_path, _FRACTIONAL)


@pytest.mark.parametrize("mode", METAL_HEAD_UPDATES)
def test_the_other_two_fractional_preconditions_are_unchanged(tmp_path, mode):
    # A legal head mode does not excuse the solver.
    with pytest.raises(ValueError, match="self_consistent"):
        _config(
            tmp_path,
            "occ_broadening = 0.13605693122994\n"
            f"sc_head_update = {mode}\n")


@pytest.mark.parametrize("mode", METAL_HEAD_UPDATES)
def test_rcrop_is_legal_on_a_metallic_deck(tmp_path, mode):
    """The entry-solve rule (2026-08-15) makes F(H) a self-map of H alone,
    so the accelerator refusal is gone: a metallic rCROP deck parses."""
    cfg = _config(
        tmp_path,
        "qp_solver = self_consistent\n"
        "sc_accelerator = rcrop\n"
        "occ_broadening = 0.13605693122994\n"
        f"sc_head_update = {mode}\n")
    assert cfg.sc.accelerator == "rcrop"


def test_an_unknown_head_update_value_refuses_and_names_both_modes(tmp_path):
    with pytest.raises(ValueError, match="sc_head_update") as exc:
        _config(tmp_path, "sc_head_update = dft_velocities\n")
    message = str(exc.value)
    for mode in METAL_HEAD_UPDATES:
        assert mode in message


# ---------------------------------------------------------------------------
# 2. Dispatch: dft_velocity never touches the parallel-transport loader
# ---------------------------------------------------------------------------

_SENTINEL = SimpleNamespace(
    nb_logical=7, velocity_dft_cart=None, forward_links=None,
    forward_neighbors=None,
    reciprocal_lattice_cart=None, validation=None)


def _stub_config(mode: str, *, do_G0: bool = True):
    return SimpleNamespace(
        sc=SimpleNamespace(head_update=mode),
        do_G0=do_G0,
        paths=SimpleNamespace(parallel_transport_file="parallel_transport.h5"),
    )


def _stub_wfn_meta(nb: int = 4):
    """Minimal ``wfn``/``meta`` for a PURE DISPATCH test.

    Added 2026-08-23 (audit finding: D3's preflight refusals read
    ``wfn``/``meta`` unconditionally at the TOP of
    ``load_head_velocity_source``, ahead of the mode dispatch this
    section actually tests -- the original ``wfn=None, meta=None`` calls
    below crashed with ``AttributeError`` on ``meta.b_id_4_user`` before
    ever reaching either loader).  ``energies`` has exactly ``nb``
    columns, which trips ``_refuse_degenerate_window_edge``'s own
    documented no-op guard (``e.shape[1] <= nb``: "no bands beyond the
    window are on hand" -- an honest scope limit, not a pass) rather than
    exercising real degeneracy physics this section is not testing.
    ``kgrid`` is a clean 8x8x8 so the per-axis stencil preflight
    (``parallel_transport`` mode only) is silent too.
    """
    wfn = SimpleNamespace(energies=np.zeros((1, nb)), kgrid=(8, 8, 8))
    meta = SimpleNamespace(b_id_4_user=nb)
    return wfn, meta


@pytest.fixture
def armed_loaders(monkeypatch):
    """Both loaders replaced: PT raises, DFT-velocity records its path."""
    calls: dict[str, object] = {}

    def _pt_boom(path, **kwargs):
        calls["parallel_transport"] = path
        raise AssertionError(
            "load_parallel_transport_head must not be reached on the "
            "dft_velocity path")

    def _dft(path, **kwargs):
        calls["dft_velocity"] = path
        return _SENTINEL

    monkeypatch.setattr(
        qsgw_head, "load_parallel_transport_head", _pt_boom)
    monkeypatch.setattr(qsgw_head, "load_dft_velocity_head", _dft)
    return calls


def test_dft_velocity_never_calls_the_parallel_transport_loader(
        armed_loaders, tmp_path):
    wfn, meta = _stub_wfn_meta()
    got = load_head_velocity_source(
        _stub_config("dft_velocity"), str(tmp_path),
        mesh=None, sym=SimpleNamespace(trs_allowed=True), wfn=wfn,
        meta=meta, print_fn=lambda *a, **k: None)
    assert got is _SENTINEL
    assert "parallel_transport" not in armed_loaders
    assert armed_loaders["dft_velocity"] == str(
        tmp_path / "parallel_transport.h5")


def test_the_transport_arm_does_call_it(armed_loaders, tmp_path):
    # Control: without this cell the one above would also pass if the
    # dispatch loaded nothing at all, or if the monkeypatch missed.
    wfn, meta = _stub_wfn_meta()
    with pytest.raises(AssertionError, match="must not be reached"):
        load_head_velocity_source(
            _stub_config("parallel_transport"), str(tmp_path),
            mesh=None, sym=SimpleNamespace(trs_allowed=True), wfn=wfn,
            meta=meta, print_fn=lambda *a, **k: None)
    assert "parallel_transport" in armed_loaders


def test_head_update_off_loads_neither(armed_loaders, tmp_path):
    got = load_head_velocity_source(
        _stub_config("off"), str(tmp_path),
        mesh=None, sym=None, wfn=None, meta=None,
        print_fn=lambda *a, **k: None)
    assert got is None
    assert armed_loaders == {}


@pytest.mark.parametrize("mode", METAL_HEAD_UPDATES)
def test_both_modes_refuse_without_do_G0(armed_loaders, tmp_path, mode):
    # The rebuilt head has no consumer without do_G0, on either route.
    with pytest.raises(ValueError, match=mode):
        load_head_velocity_source(
            _stub_config(mode, do_G0=False), str(tmp_path),
            mesh=None, sym=None, wfn=None, meta=None,
            print_fn=lambda *a, **k: None)
    assert armed_loaders == {}


def test_the_dispatch_announces_the_dropped_correction(
        armed_loaders, tmp_path):
    wfn, meta = _stub_wfn_meta()
    lines: list[str] = []
    load_head_velocity_source(
        _stub_config("dft_velocity"), str(tmp_path),
        mesh=None, sym=SimpleNamespace(trs_allowed=True), wfn=wfn, meta=meta,
        print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))))
    said = "\n".join(lines)
    assert "DFT p-matrix velocity" in said
    assert "0183" in said


# ---------------------------------------------------------------------------
# 3. The head chain on DFT velocities
# ---------------------------------------------------------------------------

_KGRID = (5, 5, 5)
_KCOORDS = np.stack(
    np.meshgrid(*(np.arange(n) for n in _KGRID), indexing="ij"), axis=-1
).reshape(-1, 3)
_FORWARD_NEIGHBORS = build_forward_neighbor_table(_KCOORDS, _KGRID)


def _head_fixture(seed: int):
    """A whole small head problem: velocity, bands, occupations, wings."""
    rng = np.random.default_rng(seed)
    nk, nb, na, nmu, ns = int(np.prod(_KGRID)), 6, 3, 4, 2
    raw = rng.normal(size=(3, nk, nb, nb)) + 1j * rng.normal(
        size=(3, nk, nb, nb))
    velocity = raw + np.swapaxes(raw.conj(), -1, -2)
    energies = np.sort(rng.uniform(-0.7, 1.1, (nk, nb)), axis=1)
    occupations = np.clip(
        rng.uniform(-0.02, 1.02, (nk, nb))[:, ::-1].copy(), -0.02, 1.0)
    surface = rng.uniform(0.0, 0.4, (nk, nb))
    psi = rng.normal(size=(nk, ns, nmu, nb)) + 1j * rng.normal(
        size=(nk, ns, nmu, nb))
    U = np.stack([
        np.linalg.qr(
            rng.normal(size=(na, na)) + 1j * rng.normal(size=(na, na)))[0]
        for _ in range(nk)])
    bvec = np.asarray([[1.7, 0.0, 0.0], [0.0, 1.7, 0.0], [0.0, 0.0, 1.7]])
    return SimpleNamespace(
        nk=nk, nb=nb, na=na, velocity=velocity, energies=energies,
        occupations=occupations, surface=surface,
        wfns_qp=SimpleNamespace(
            layout="face", slices=None, enk=energies, occ=occupations,
            psi_mun=jnp.asarray(psi),
            psi_nmu=jnp.asarray(psi.transpose(0, 3, 1, 2))),
        U=U, bvec=bvec,
        omegas=np.asarray([0.31 + 0.05j, 0.77 + 0.05j, 1.4 + 0.05j]),
        wfn=SimpleNamespace(nspin=1),
        meta=SimpleNamespace(
            cell_volume=97.3, nk_tot=nk, nspinor=2,
            nspinor_wfnfile=2),
        config=SimpleNamespace(head=SimpleNamespace(wcoul0_eta=0.0)),
    )


def _response(fx, *, links, delta, U, wings=True):
    return build_iteration_head_response(
        delta,
        links,
        (_FORWARD_NEIGHBORS if links is not None else None),
        jnp.asarray(fx.velocity),
        jnp.asarray(U),
        jnp.asarray(fx.energies),
        jnp.asarray(fx.occupations),
        fx.omegas,
        surface_weight_qp_kn=jnp.asarray(fx.surface),
        mesh=_mesh(),
        kgrid=_KGRID,
        bvec_cart=fx.bvec,
        nb_logical=fx.nb,
        sigma_energies_ry=fx.energies,
        efermi_ry=0.21,
        wfn=fx.wfn,
        meta=fx.meta,
        config=fx.config,
        wfns_qp=(fx.wfns_qp if wings else None),
        eta_ry=0.05,
    )


def test_the_dft_velocity_chain_reproduces_the_transport_chain_at_zero_dh():
    """The whole chain, both routes, where they must agree exactly.

    With DeltaH = 0 the finite-link correction vanishes, so the
    transport route reduces to the DFT-velocity route.  Agreement across
    S(z), the Drude term (surface weights are supplied), both ISDF wings
    and the static kappa^2 is the smoke test that the new branch skips only
    the correction and nothing else.
    """
    fx = _head_fixture(2026)
    identity_links = jnp.broadcast_to(
        jnp.eye(fx.nb, dtype=jnp.complex128),
        (3, fx.nk, fx.nb, fx.nb))
    zeros_dh = jnp.zeros((fx.nk, fx.nb, fx.nb), dtype=jnp.complex128)

    transport = _response(fx, links=identity_links, delta=zeros_dh, U=fx.U)
    dft = _response(fx, links=None, delta=None, U=fx.U)

    np.testing.assert_allclose(
        np.asarray(dft.S_direct), np.asarray(transport.S_direct),
        rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(dft.Y_x), np.asarray(transport.Y_x),
        rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(dft.Z_y), np.asarray(transport.Z_y),
        rtol=1e-13, atol=1e-13)
    assert dft.static_kappa2_bohr2 == pytest.approx(
        transport.static_kappa2_bohr2, rel=1e-14)
    assert dft.omegas == transport.omegas
    assert dft.efermi_ry == transport.efermi_ry


def test_a_real_delta_h_is_what_the_dft_velocity_mode_drops():
    """Negative control for the cell above: with genuine links and
    DeltaH the two routes MUST differ, or the equality proved nothing."""
    fx = _head_fixture(3031)
    rng = np.random.default_rng(77)
    links = jnp.broadcast_to(
        jnp.eye(fx.nb, dtype=jnp.complex128),
        (3, fx.nk, fx.nb, fx.nb))
    dh = np.zeros((fx.nk, fx.nb, fx.nb), dtype=np.complex128)
    block = rng.normal(size=(fx.nk, fx.na, fx.na)) + 1j * rng.normal(
        size=(fx.nk, fx.na, fx.na))
    dh[:, :fx.na, :fx.na] = block + np.swapaxes(block.conj(), -1, -2)
    dh[:, np.arange(fx.na, fx.nb), np.arange(fx.na, fx.nb)] = rng.normal(
        size=(fx.nk, fx.nb - fx.na))

    transport = _response(
        fx, links=links, delta=jnp.asarray(dh), U=fx.U, wings=False)
    dft = _response(fx, links=None, delta=None, U=fx.U, wings=False)
    assert np.max(np.abs(
        np.asarray(dft.S_direct) - np.asarray(transport.S_direct))) > 1e-3


def test_dft_velocity_rotates_into_the_qp_basis_every_iteration():
    """The per-iteration U rotation is NOT dropped with the finite links.

    The R2/R3 tool route was one-shot, so its velocity was never rotated.
    A QSGW iteration's is: the mode feeds ``U^dag v U`` on the active block
    to the same S tensor, using the U the head carry already threads.
    """
    fx = _head_fixture(4102)
    got = np.asarray(_response(
        fx, links=None, delta=None, U=fx.U, wings=False).S_direct)

    rotated = rotate_velocity_active_to_qp(
        jnp.asarray(fx.velocity), jnp.asarray(fx.U), mesh=_mesh())
    common = dict(
        mesh=_mesh(), nb_logical=fx.nb,
        cell_volume=float(fx.meta.cell_volume), nk_tot=fx.nk,
        nspin=int(fx.wfn.nspin), nspinor=int(fx.meta.nspinor),
        eta_ry=0.05, surface_weight_kn=jnp.asarray(fx.surface))
    ref = np.asarray(head_s_tensor_sharded(
        rotated, jnp.asarray(fx.energies), jnp.asarray(fx.occupations),
        fx.omegas, **common))
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)

    # ...and the rotation is load-bearing: the unrotated velocity gives a
    # different head on this fixture.
    unrotated = np.asarray(head_s_tensor_sharded(
        jnp.asarray(fx.velocity), jnp.asarray(fx.energies),
        jnp.asarray(fx.occupations), fx.omegas, **common))
    assert np.max(np.abs(got - unrotated)) > 1e-3


def test_the_loader_the_driver_takes_is_the_one_the_tool_takes():
    """No second implementation: the tool's --dft-velocity-only stage
    loader moved into gw.qsgw_head and the tool imports it from there."""
    source = (
        __import__("pathlib").Path(qsgw_head.__file__).resolve().parents[2]
        / "tools" / "qsgw_head_spectrum.py"
    ).read_text()
    assert "load_dft_velocity_head" in source
    assert "_load_dft_velocity_stage" not in source


def test_the_map_entry_solves_its_own_occupations():
    """The rCROP-enabling invariant, pinned at the source level.

    F(H) must be a self-map of H alone: gw_iteration_map solves its MP1
    occupation state at ENTRY from the spectrum of the H it was handed,
    and the carried state is diagnostic only.  The discriminating failure
    this guards: a consumer rewired back to state.occupation_state would
    make trial/accepted rCROP iterates consume occupations from a different
    trajectory point.

    The energy ladder handed to the solve used to be spelled
    ``wfns_qp.enk`` and is now built here as ``enk_entry`` — the immutable
    DFT ladder with the active block replaced by THIS call's ``E_full``,
    which is what ``rotate_wavefunctions`` puts in ``wfns_qp.enk`` anyway
    (``wavefunction_bundle.py:616-619``).  The move is what lets the
    sum-band tail scissor classify its bands from the same occupation
    state; (a2) below pins the ordering that makes that true, so the two
    fits can never be fed different occupations.
    """
    import ast
    import inspect
    import re
    from gw import sc_iteration

    src = inspect.getsource(sc_iteration.gw_iteration_map)
    # (a) the entry solve exists and feeds from THIS call's spectrum -- the
    # active block of the ladder is E_full, not anything off the carry.
    assert re.search(
        r"entry_occ_state, entry_surface_weight_kn = "
        r"_solve_head_occupations\(\s*\n?\s*inputs, enk_entry\)", src), \
        "gw_iteration_map lost its entry occupation solve"
    assert re.search(
        r"enk_entry = jax\.lax\.with_sharding_constraint\(\s*\n"
        r"\s*jnp\.asarray\(inputs\.wfns_dft\.enk\)\.at\[\s*\n?"
        r"\s*:, inputs\.band_slices\.sigma\]\.set\(\s*\n"
        r"\s*jnp\.asarray\(E_full,", src), \
        ("the entry ladder is no longer 'the DFT ladder with the active "
         "block set to this call's E_full'")
    # (a2) ...and it is solved BEFORE the sum-band tail scissor, so ONE
    # occupation state serves both scissor fits.  Reversing these two would
    # make the tail fit's band classification a generation stale, or
    # circular (the tail scissor writes the ladder the solve would read).
    i_solve = src.index("_solve_head_occupations(")
    i_tail = src.index("tail_fit = fit_scissor(")
    assert i_solve < i_tail, (
        "the entry occupation solve must precede the sum-band tail scissor "
        "fit; the tail fit's val/cond/crossing classes come from it")
    # (b) the metal chi/head/Sigma threading consumes the ENTRY state
    assert "metal_occ_state = (entry_occ_state" in src
    assert "head_occ_kn = entry_occ_state.f_kn" in src
    # (c) the carried state is read ONLY by the mu-drift diagnostic:
    # count attribute reads of state.occupation_state in the function.
    tree = ast.parse(src)
    reads = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "occupation_state"
        and isinstance(n.value, ast.Name) and n.value.id == "state"
    ]
    assert len(reads) == 2, (
        f"state.occupation_state is read {len(reads)} times in "
        "gw_iteration_map; the entry-solve rule allows exactly the two "
        "reads of the mu-drift diagnostic (guard + subtraction)")
