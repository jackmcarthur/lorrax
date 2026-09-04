"""Scalar-relativistic (nspinor=1) pseudopotential semantics, and the
FLAGLESS automatic V_NL spin-orbit resolution (owner ruling 2026-08-28).

* ``resolve_soc_mode``: no soc/spinor flag exists anywhere.  SR pseudos
  resolve False quietly for ANY nspinor; QE's ``<spinorbit>`` is
  authoritative when present; an FR pseudo with nspinor=1 is FORCED to
  the j-averaged operator (QE ``average_pp``) and a spinorbit=true claim
  with nspinor=1 REFUSES — a j-resolved V_NL has no representation on
  one-component wavefunctions.  FR + nspinor=2 + no record returns None:
  the answer is MEASURED (``measure_soc_mode``), pinned below on real
  fixture WFNs for both arms plus the refusal twins.
* ``build_E_blocks_full`` on a j-free UPF: E^{σσ'} = D ⊗ δ_{σσ'}, so the
  nspinor=1 slice E[:1,:1] IS the scalar D-block, bit for bit, and the
  ``soc`` argument cannot matter.
* ``dipole.h5`` carries ``prov_nspinor``/``prov_soc``; a stamped nspinor
  mismatch refuses (INVARIANTS row 3: representation is part of the reuse
  contract) while an unstamped legacy file is still accepted.
* ``extract_species``: the write-only ``nspinor``/``proj_j`` fields are
  gone, and NLCC reads the ``core_correction`` UpfLogical by MEMBER —
  ``bool()`` of any Enum member, FALSE included, is True.

Every refusal here has both twins: the case that fires and the case that
must not.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal fake pseudos.  ``pseudo_has_j_channels`` reads beta.lll/.jjj;
# ``pseudo_soc_strength_ry`` and ``build_E_blocks_full`` additionally read
# pp_dij.value and pp_header.number_of_proj.  No file, no parser.
# ---------------------------------------------------------------------------

def _beta(l, j=None):
    b = SimpleNamespace(lll=l, angular_momentum=l)
    if j is not None:
        b.jjj = j          # only FR betas carry jjj — SR fakes must NOT
    return b


def _pseudo(betas, dij):
    dij = np.asarray(dij, dtype=np.float64)
    return SimpleNamespace(
        pp_header=SimpleNamespace(number_of_proj=len(betas)),
        pp_nonlocal=SimpleNamespace(
            pp_beta=list(betas),
            pp_dij=SimpleNamespace(value=dij.ravel()),
        ),
    )


def _sr_pseudo():
    """ONCVPSP SR shape: 2 projectors per ℓ, no jjj anywhere.

    The ℓ=0 block carries an off-diagonal coupling so the g_r ≠ g_c path
    of the spin-identity branch is exercised, not just the diagonal.
    """
    D = np.zeros((4, 4))
    D[0, 0], D[1, 1] = 0.7, -0.2
    D[0, 1] = D[1, 0] = 0.05
    D[2, 2], D[3, 3] = 0.4, 0.3
    return _pseudo([_beta(0), _beta(0), _beta(1), _beta(1)], D), D


def _fr_pseudo(d_minus=0.3, d_plus=0.1):
    """One ℓ=1 shell resolved into j = 1/2, 3/2 — ΔD = |d_minus - d_plus|."""
    D = np.diag([d_minus, d_plus])
    return _pseudo([_beta(1, j=0.5), _beta(1, j=1.5)], D)


# ---------------------------------------------------------------------------
# resolve_soc_mode — the AUTOMATIC contract (owner ruling 2026-08-28):
# no soc/spinor flag exists anywhere.  Metadata arms here; the measured
# arm (FR + nspinor=2 + no <spinorbit>) is pinned on real fixture WFNs
# further down.
# ---------------------------------------------------------------------------

def test_sr_pseudo_resolves_false_quietly_for_any_nspinor():
    from psp.vnl_ops import resolve_soc_mode

    sr, _ = _sr_pseudo()
    for nspinor in (1, 2):
        lines = []
        assert resolve_soc_mode({"Si": sr}, nspinor=nspinor,
                                print_fn=lines.append) is False
        # the quiet contract: no choice exists, so no announcement
        assert lines == []


def test_the_soc_flag_is_gone():
    # The owner's ruling is a surface contract, not a default: there must be
    # NO caller-declared arm left to reach.  A caller still passing soc=
    # gets a TypeError, not a silently ignored knob (the parsed-but-unread
    # defect class this repo has paid for).
    from psp.vnl_ops import build_vnl_setup, resolve_soc_mode

    with pytest.raises(TypeError):
        resolve_soc_mode({"Mo": _fr_pseudo()}, soc=True, nspinor=2)
    with pytest.raises(TypeError):
        build_vnl_setup(None, pseudos={"Mo": _fr_pseudo()}, nspinor=2,
                        soc=True)


def test_fr_nspinor2_undeclared_returns_none_meaning_measure():
    # The old assume-True banner is gone: with an FR pseudo, 2-component
    # wavefunctions and no authoritative record, the resolver hands the
    # question to the MEASUREMENT (build_vnl_setup / measure_soc_mode)
    # instead of deciding.
    from psp.vnl_ops import resolve_soc_mode

    lines = []
    assert resolve_soc_mode({"Mo": _fr_pseudo()}, nspinor=2,
                            print_fn=lines.append) is None


def test_fr_nspinor1_undeclared_is_forced_averaged_and_says_so():
    from psp.vnl_ops import resolve_soc_mode

    lines = []
    assert resolve_soc_mode({"Mo": _fr_pseudo()}, nspinor=1,
                            print_fn=lines.append) is False
    text = "\n".join(lines)
    # Red twin of a silent resolution: the forced choice must be announced,
    # with the discarded spin-orbit strength on the record.
    assert "j-AVERAGED" in text
    assert "nspinor=1" in text
    assert "0.200000" in text          # |0.3 - 0.1| Ry, the ΔD at stake


def test_fr_nspinor1_qe_spinorbit_true_refuses_naming_the_source():
    # The contradictory-file case: a wfn claiming lspinorb with 1-component
    # wavefunctions cannot be honored, and the refusal must say WHERE the
    # spin-orbit claim came from — and must not point at flags that no
    # longer exist.
    from psp.vnl_ops import resolve_soc_mode

    wfn = SimpleNamespace(spinorbit=True)
    with pytest.raises(ValueError, match="QE <spinorbit>") as err:
        resolve_soc_mode({"Mo": _fr_pseudo()}, wfn, nspinor=1)
    msg = str(err.value)
    for needle in ("got:", "want:", "fix:", "nspinor=1", "average_pp"):
        assert needle in msg, f"refusal must carry {needle!r}: {msg}"
    for gone in ("--soc", "soc=False", "soc=True", "pass soc"):
        assert gone not in msg, f"refusal points at a removed flag: {msg}"


def test_qe_spinorbit_is_authoritative_both_ways():
    from psp.vnl_ops import resolve_soc_mode

    lines = []
    assert resolve_soc_mode({"Mo": _fr_pseudo()},
                            SimpleNamespace(spinorbit=True), nspinor=2,
                            print_fn=lines.append) is True
    assert any("j-RESOLVED" in ln and "QE <spinorbit>" in ln for ln in lines)

    lines = []
    for nspinor in (1, 2):
        assert resolve_soc_mode({"Mo": _fr_pseudo()},
                                SimpleNamespace(spinorbit=False),
                                nspinor=nspinor,
                                print_fn=lines.append) is False
    assert any("j-AVERAGED" in ln and "QE <spinorbit>" in ln for ln in lines)


def test_resolve_soc_mode_requires_nspinor():
    # nspinor=1 was a silent default inconsistent with the rest of the
    # module (build_vnl_setup resolves it from the WFN); it is now required.
    from psp.vnl_ops import resolve_soc_mode

    with pytest.raises(TypeError):
        resolve_soc_mode({"Mo": _fr_pseudo()})


# ---------------------------------------------------------------------------
# build_E_blocks_full on a j-free pseudo
# ---------------------------------------------------------------------------

def test_jfree_blocks_are_D_times_spin_identity_and_soc_cannot_matter():
    from psp.radial.build_projectors_qe import (
        build_E_blocks_full, pseudo_has_j_channels)

    sr, D = _sr_pseudo()
    assert pseudo_has_j_channels(sr) is False

    blocks_t = build_E_blocks_full(sr, soc=True)
    blocks_f = build_E_blocks_full(sr, soc=False)
    assert set(blocks_t) == set(blocks_f) == {0, 1}

    sub = {0: D[:2, :2], 1: D[2:, 2:]}
    for l in (0, 1):
        msize = 2 * l + 1
        want = np.kron(sub[l], np.eye(msize)).astype(np.complex128)
        for E in (blocks_t[l], blocks_f[l]):
            # E[s,s'] = D·δ_ss', bit for bit — == , not allclose
            assert np.array_equal(E[0, 0], want)
            assert np.array_equal(E[1, 1], want)
            assert not E[0, 1].any() and not E[1, 0].any()
            # the nspinor=1 slice consumed by build_vnl_setup/E_super
            # (vnl_ops: ch.E[:nspinor, :nspinor]) IS the scalar block
            assert np.array_equal(E[:1, :1], want[None, None])
        # the soc argument is inert on a j-free pseudo
        assert np.array_equal(blocks_t[l], blocks_f[l])


# ---------------------------------------------------------------------------
# dipole.h5 provenance: prov_nspinor / prov_soc
# ---------------------------------------------------------------------------

class _FakeWfn:
    # same shape as test_psp_padded_gvectors._FakeWfn: everything
    # common.parallel_transport.wfn_fingerprint samples on a loaded WFN
    def __init__(self, seed=0, nbands=8, nelec=4, nspinor=1, nk=3):
        rng = np.random.default_rng(seed)
        self.energies = rng.standard_normal((1, nk, nbands))
        self.kpoints = rng.standard_normal((nk, 3))
        self.nelec, self.nspinor, self.nbands = nelec, nspinor, nbands


def _write_stamped(path, wfn, **kw):
    import h5py
    from psp.get_dipole_mtxels import stamp_dipole_provenance

    with h5py.File(str(path), "w") as h5:
        h5.create_dataset("dipole_cart", data=np.zeros((3, 1, 1, 1)))
        stamp_dipole_provenance(h5, wfn=wfn, wfn_path="WFN.h5",
                                nval=2, ncond=3, nband=8,
                                nb_written=4, bispinor=False,
                                skip_vnl=False, vnl_mode="analytic", **kw)


def test_prov_nspinor_and_soc_are_stamped_and_accepted(tmp_path):
    import h5py
    from psp.get_dipole_mtxels import check_dipole_provenance

    wfn = _FakeWfn(nspinor=1)
    p = tmp_path / "dipole.h5"
    _write_stamped(p, wfn, nspinor=1, soc=False)
    with h5py.File(str(p), "r") as h5:
        assert int(h5.attrs["prov_nspinor"]) == 1
        assert bool(h5.attrs["prov_soc"]) is False
    assert check_dipole_provenance(p, wfn=wfn, nval=2, ncond=3, nband=8,
                                   print_fn=lambda *a: None) is True


def test_prov_nspinor_mismatch_refuses(tmp_path, monkeypatch):
    # Red twin: an nspinor=1 artifact has the right SHAPE for an nspinor=2
    # run of the same crystal, so only the stamp can catch the reuse.
    from common import sanity
    from psp.get_dipole_mtxels import check_dipole_provenance

    monkeypatch.setattr(sanity, "sanity_strict", lambda: False)
    p = tmp_path / "dipole.h5"
    _write_stamped(p, _FakeWfn(nspinor=1), nspinor=1, soc=False)

    lines = []
    ok = check_dipole_provenance(p, wfn=_FakeWfn(nspinor=2), nval=2,
                                 ncond=3, nband=8, print_fn=lines.append)
    assert ok is False
    text = "\n".join(lines)
    assert "prov_nspinor" in text
    assert "file=1" in text and "run=2" in text


def test_prov_nspinor_missing_stamp_is_legacy_accepted(tmp_path):
    # A file written by a pre-stamp producer carries no prov_nspinor and
    # must keep working — same reading as prov_vnl_velocity_sign.
    import h5py
    from psp.get_dipole_mtxels import check_dipole_provenance

    wfn = _FakeWfn(nspinor=2)
    p = tmp_path / "dipole.h5"
    _write_stamped(p, wfn)                       # no nspinor=, no soc=
    with h5py.File(str(p), "r") as h5:
        assert "prov_nspinor" not in h5.attrs
        assert "prov_soc" not in h5.attrs
    assert check_dipole_provenance(p, wfn=wfn, nval=2, ncond=3, nband=8,
                                   print_fn=lambda *a: None) is True


# ---------------------------------------------------------------------------
# extract_species: dead spin fields removed, NLCC flag read by Enum member
# ---------------------------------------------------------------------------

def _full_pseudo(core_correction, nlcc_payload):
    """A complete-enough UPF object for extract_species (no parser)."""
    n_r = 8
    r = np.linspace(0.0, 1.4, n_r)
    beta = [SimpleNamespace(value=np.arange(n_r, dtype=float) * (ip + 1),
                            angular_momentum=ip % 2)
            for ip in range(2)]
    return SimpleNamespace(
        pp_header=SimpleNamespace(z_valence=4.0, number_of_proj=2,
                                  core_correction=core_correction),
        pp_mesh=SimpleNamespace(pp_r=SimpleNamespace(value=r),
                                pp_rab=SimpleNamespace(value=r * 0.01)),
        pp_local=SimpleNamespace(value=-np.ones(n_r)),
        pp_nonlocal=SimpleNamespace(
            pp_beta=beta,
            pp_dij=SimpleNamespace(value=np.eye(2).ravel())),
        pp_nlcc=(SimpleNamespace(value=nlcc_payload)
                 if nlcc_payload is not None else None),
    )


def test_nlcc_enum_false_with_payload_present_is_not_nlcc():
    # Red twin of the Enum fix: bool(UpfLogical.FALSE) is True, so the old
    # expression called this pseudo NLCC and integrated its core charge.
    from psp.species import extract_species
    from psp.upf.upf_model_2_0_1 import UpfLogical

    payload = np.full(8, 0.3)
    sp, = extract_species(
        {"Si": _full_pseudo(UpfLogical.FALSE, payload)})
    assert sp.has_nlcc is False
    assert not sp.rho_core_r.any()


def test_nlcc_enum_true_reads_the_grid_and_none_or_missing_do_not():
    from psp.species import extract_species
    from psp.upf.upf_model_2_0_1 import UpfLogical

    payload = np.full(8, 0.3)
    sp, = extract_species({"Si": _full_pseudo(UpfLogical.TRUE, payload)})
    assert sp.has_nlcc is True
    assert np.array_equal(sp.rho_core_r, payload)

    # the flag alone cannot supply a grid
    sp, = extract_species({"Si": _full_pseudo(UpfLogical.TRUE, None)})
    assert sp.has_nlcc is False

    # header attribute absent from the file (Optional, default None)
    sp, = extract_species({"Si": _full_pseudo(None, payload)})
    assert sp.has_nlcc is False


def test_extract_species_carries_no_spin_fields():
    # nspinor/proj_j were stored and never read anywhere — the exact
    # parsed-but-ignored-key defect class.  Their absence is the contract.
    from psp.species import extract_species
    from psp.upf.upf_model_2_0_1 import UpfLogical

    with pytest.raises(TypeError):
        extract_species({"Si": _full_pseudo(None, None)}, nspinor=1)
    sp, = extract_species({"Si": _full_pseudo(UpfLogical.FALSE, None)})
    assert not hasattr(sp, "nspinor") and not hasattr(sp, "proj_j")


# ---------------------------------------------------------------------------
# A real scalar-relativistic ONCVPSP UPF, when the staged copy exists
# ---------------------------------------------------------------------------

_SR_SI = Path.home() / ("projects/nonspinor_2026-08-28/pseudos/"
                        "nc-sr-04_pbe_standard/Si.upf")


@pytest.mark.skipif(not _SR_SI.exists(), reason=f"{_SR_SI} not staged")
def test_real_sr_si_upf_is_scalar_end_to_end():
    from psp.radial.build_projectors_qe import (
        build_E_blocks_full, pseudo_has_j_channels)
    from psp.species import extract_species
    from psp.upf.load_upf import load_upf
    from psp.upf.normalize import normalize_dataclass
    from psp.vnl_ops import resolve_soc_mode

    p = normalize_dataclass(load_upf(_SR_SI))
    assert pseudo_has_j_channels(p) is False

    lines = []
    assert resolve_soc_mode({"Si": p}, nspinor=1,
                            print_fn=lines.append) is False
    assert lines == []                # SR: no choice exists, quiet contract

    blocks_t = build_E_blocks_full(p, soc=True)
    blocks_f = build_E_blocks_full(p, soc=False)
    for l in blocks_t:
        assert np.array_equal(blocks_t[l], blocks_f[l])
        assert not blocks_t[l][0, 1].any() and not blocks_t[l][1, 0].any()
        assert np.array_equal(blocks_t[l][0, 0], blocks_t[l][1, 1])

    sp, = extract_species({"Si": p})
    # ONCVPSP SR standard: 2 projectors per ℓ, strictly diagonal PP_DIJ
    assert sp.n_proj == len(sp.proj_l)
    off = sp.dij - np.diag(np.diag(sp.dij))
    assert not off.any()


# ---------------------------------------------------------------------------
# The MEASURED resolution arm, on real fixture WFNs (2026-08-28).
#
# Ground truth, established from the spectra themselves (an average_pp
# Hamiltonian commutes with every spin rotation, so ALL its eigenvalues
# have even multiplicity at every k):
#   * si_cohsex_debug/WFN.h5 — exact spin pairing everywhere (worst broken
#     (2n,2n+1) pair 0.000000 eV): lspinorb=.false. (matches its QE run
#     06_si_4x4x4_nosoc, <spinorbit>false</spinorbit>).
#   * cohsex_debug/WFNsmall.h5 (MoS2, FR Mo+S) — spin pairing broken by up
#     to 0.136 eV at K: genuinely lspinorb=.true.  (gnppm_debug measures
#     0.138 eV — also true; cohsex_debug is used here because it carries
#     its own UPFs.)
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[1]
_SI_SPINOR_WFN = _REPO / "tests/regression/si_cohsex_debug/WFN.h5"
_MOS2_SOC_DECK = _REPO / "tests/regression/cohsex_debug"
_FR_SI_UPF = Path("/home/jackm/SOURCES/agent_tasks/qe-gw-pipeline/"
                  "assets/pseudos_standard/Si.upf")


def _wfn_loader(path):
    from wfn_loader import WfnLoader

    return WfnLoader(str(path))


@pytest.fixture(scope="module")
def si_spinor_measured():
    """(wfn, setup, printed lines) for the Si spinor deck + FR Si.upf."""
    if not (_SI_SPINOR_WFN.exists() and _FR_SI_UPF.exists()):
        pytest.skip("si_cohsex_debug WFN or FR Si.upf not staged")
    from psp.upf.load_upf import load_upf
    from psp.upf.normalize import normalize_dataclass
    from psp.vnl_ops import build_vnl_setup

    wfn = _wfn_loader(_SI_SPINOR_WFN)
    pseudo = normalize_dataclass(load_upf(_FR_SI_UPF))
    lines: list[str] = []
    setup = build_vnl_setup(wfn, pseudos={"Si": pseudo}, nspinor=2,
                            print_fn=lines.append)
    return wfn, setup, lines


def test_measurement_selects_javeraged_on_the_lspinorb_false_wfn(
        si_spinor_measured):
    """The auto-fix of the defect-#21 class: the spinor Si fixture
    (lspinorb=false WFN + FR pseudo) now measures its way to j-AVERAGED —
    the historically-correct answer the old assume-True default got wrong."""
    import re as _re

    _wfn, setup, lines = si_spinor_measured
    assert setup.soc is False
    assert "j-AVERAGED selected by measurement" in setup.soc_provenance
    # Both discriminants are on the record, at the calibrated scales:
    # the wrong (j-resolved) candidate splits probed multiplets at the
    # SOC scale (Si Δ0 ~ 4.8e-2 eV on V_NL alone), the right one sits at
    # the numerical floor — seven decades apart.
    nums = [float(x) for x in
            _re.findall(r"(\d\.\d+e[+-]\d+) eV", setup.soc_provenance)]
    assert len(nums) >= 2
    s_avg, s_res = nums[0], nums[1]
    assert s_avg < 1e-6, setup.soc_provenance
    assert s_res > 1e-3, setup.soc_provenance
    # the announcement also went through print_fn (debug stream)
    assert any("selected by measurement" in ln for ln in lines)


def test_measurement_selects_jresolved_on_the_lspinorb_true_wfn():
    """The other arm: a genuine SOC WFN (MoS2, spin pairing broken by
    0.136 eV) must measure its way to j-RESOLVED."""
    wfn_path = _MOS2_SOC_DECK / "WFNsmall.h5"
    if not wfn_path.exists():
        pytest.skip("cohsex_debug fixture not staged")
    from psp.pseudos import load_pseudopotentials
    from psp.vnl_ops import build_vnl_setup

    wfn = _wfn_loader(wfn_path)
    pseudos = load_pseudopotentials(str(_MOS2_SOC_DECK))
    assert sorted(pseudos) == ["Mo", "S"]
    lines: list[str] = []
    setup = build_vnl_setup(wfn, pseudos=pseudos, nspinor=2,
                            print_fn=lines.append)
    assert setup.soc is True
    assert "j-RESOLVED selected by measurement" in setup.soc_provenance
    # On this deck the probed multiplets are (P)T-protected doublets that
    # BOTH candidates leave invariant; the positive lspinorb evidence is
    # el itself breaking exact spin degeneracy (impossible for average_pp).
    assert "spin degeneracy" in setup.soc_provenance


def test_measurement_refusals_are_loud_and_the_equivalence_arm_is_green(
        si_spinor_measured):
    """Red twins for the measured arm — the refusals must fire, and the
    same calls with the benign condition must NOT (no tautology)."""
    import jax.numpy as jnp
    from psp.radial.build_projectors_qe import build_E_blocks_full
    from psp.upf.load_upf import load_upf
    from psp.upf.normalize import normalize_dataclass
    from psp.vnl_ops import measure_soc_mode

    wfn, setup, _ = si_spinor_measured

    # 1. No WFN reader behind the object: refuse at the door.
    with pytest.raises(RuntimeError, match="cannot be measured"):
        measure_soc_mode(object(), None, None, None, delta_d_ry=0.1)

    # 2. Candidates indistinguishable on everything probed while ΔD says
    #    they differ materially: UNMEASURABLE, loudly.
    with pytest.raises(RuntimeError, match="UNMEASURABLE"):
        measure_soc_mode(wfn, setup, setup.E_super, setup.E_super,
                         delta_d_ry=0.148, print_fn=lambda *a: None)

    # 2g. Green twin of 2: the SAME call with ΔD below the floor is the
    #     equivalence verdict, not a refusal.
    soc, line = measure_soc_mode(wfn, setup, setup.E_super, setup.E_super,
                                 delta_d_ry=0.0, print_fn=lambda *a: None)
    assert soc is True and "equivalent" in line

    # 3. BOTH candidates split (here: the j-resolved E handed in as both
    #    arms, against the average_pp WFN): wrong-pairing refusal.
    blocks_res = build_E_blocks_full(
        normalize_dataclass(load_upf(_FR_SI_UPF)), soc=True)
    ns, r_tot = setup.nspinor, setup.total_R
    e_sup = np.zeros((ns, ns, r_tot, r_tot), dtype=np.complex128)
    off = 0
    for ch in setup.channels:                     # single species: Si
        e_np = np.asarray(blocks_res[ch.l])[:ns, :ns]
        for _a in range(ch.natoms):
            e_sup[:, :, off:off + ch.R, off:off + ch.R] = e_np
            off += ch.R
    e_res = jnp.asarray(e_sup)
    with pytest.raises(RuntimeError, match="BOTH candidate"):
        measure_soc_mode(wfn, setup, e_res, e_res,
                         delta_d_ry=0.148, print_fn=lambda *a: None)


# ---------------------------------------------------------------------------
# QE radial quadrature conventions (2026-08-28 KIH-offset fix).
#
# Two schemes were brought into line with QE 7.4 after a per-term diagonal
# comparison against pw2bgw kih.dat isolated two constant offsets:
#
# * V_loc (build_projectors_qe._qe_vloc_radial_scheme): the alpha-Z / SR
#   tables integrated with BARE rab weights (rectangle rule) over the FULL
#   mesh.  Euler-Maclaurin: rectangle-vs-Simpson error ~ -(h^2/12) f'(0)
#   with f'(0) = Z e^2 exactly, a pseudo-independent -0.08 meV constant on
#   every KIH diagonal (Si, h=0.01); the full-mesh tail added +0.068 meV
#   for the FR Si pseudo.  QE: Modules/read_pseudo.f90 truncates at
#   r <= 10 bohr, forces the count odd, upflib/simpsn.f90 weights.
#
# * V_NL (radial_tables.build_all_tables): beta integrals ran over the
#   full mesh with generic Simpson weights; QE (upflib/beta_mod.f90) uses
#   simpsn over EXACTLY upf%kkbeta points.  ONCVPSP l=2 betas end on
#   ~1e-4 nonzero values at that index -> +0.004 meV V_NL constant.
#
# Each test pins the fixed scheme against an INDEPENDENT implementation
# (scipy.integrate.simpson on the same truncated mesh) and carries a red
# twin: the OLD scheme's value must FAIL the new tolerance, so the test
# cannot be satisfied by a tautology.
# ---------------------------------------------------------------------------

def _staged_upfs():
    fr = Path.home() / ("projects/nonspinor_2026-08-28/runs/Si_sr_4x4x4/"
                        "qe_fravg/nscf/Si.upf")
    out = []
    if _SR_SI.exists():
        out.append(("sr", _SR_SI))
    if fr.exists():
        out.append(("fr", fr))
    return out


@pytest.mark.skipif(not _staged_upfs(), reason="no staged Si UPFs")
@pytest.mark.parametrize("tag,path", _staged_upfs())
def test_qe_vloc_radial_scheme_matches_qe_and_old_rectangle_fails(tag, path):
    from scipy.integrate import simpson as scipy_simpson

    from psp.radial.build_projectors_qe import _qe_vloc_radial_scheme
    from psp.species import extract_species
    from psp.upf.load_upf import load_upf
    from psp.upf.normalize import normalize_dataclass

    sp, = extract_species({"Si": normalize_dataclass(load_upf(path))})
    r, rab, v, z = sp.r, sp.rab, sp.vloc_r, sp.z_valence
    e2 = 2.0

    msh, w = _qe_vloc_radial_scheme(r, rab)
    # QE's msh rule: odd, and the last included point is the first r > 10
    # (or the trimmed full mesh when r never exceeds 10 bohr).
    assert msh % 2 == 1
    if r[-1] > 10.0:
        assert r[msh - 1] <= 10.0 + r[1] - r[0] and r[min(msh, len(r) - 1)] >= 10.0 - 1e-12

    f = r * (r * v + z * e2)                      # alpha-Z integrand
    alpha_fixed = float(np.sum(f[:msh] * w))
    # Independent reference: scipy's composite Simpson on the SAME odd
    # truncated mesh (dx from rab: uniform ONCVPSP grid).
    alpha_ref = float(scipy_simpson(f[:msh], x=r[:msh]))
    assert abs(alpha_fixed - alpha_ref) < 1e-10

    # RED TWIN: the pre-fix scheme (bare rab over the full mesh) misses the
    # reference by ~6.7e-5 (SR) / ~1.3e-5 (FR) Ry*bohr^3 -- an 0.08 / 0.016
    # meV constant on every KIH diagonal.  It must FAIL the tolerance above.
    alpha_old = float(np.sum(f * rab))
    assert abs(alpha_old - alpha_ref) > 1e-6

    # Same for the q>0 short-range table at a representative q.
    from scipy.special import erf as _erf
    q = 2.0
    a = r * v + z * e2 * _erf(r)
    g = a * np.sin(q * r) / q
    t_fixed = float(np.sum(g[:msh] * w))
    t_ref = float(scipy_simpson(g[:msh], x=r[:msh]))
    assert abs(t_fixed - t_ref) < 1e-10


@pytest.mark.skipif(not _SR_SI.exists(), reason=f"{_SR_SI} not staged")
def test_beta_tables_use_kkbeta_and_full_mesh_generic_weights_fail():
    from scipy.special import spherical_jn

    from psp.radial_tables import (_qe_simpsn_weights, _simpson_weights,
                                   build_all_tables)
    from psp.species import extract_species
    from psp.upf.load_upf import load_upf
    from psp.upf.normalize import normalize_dataclass

    sp, = extract_species({"Si": normalize_dataclass(load_upf(_SR_SI))})
    assert 0 < sp.kkbeta < len(sp.r)              # 196 for this UPF

    tables = build_all_tables([sp], q_max=5.0, n_q=101)
    q = tables["q"]
    F = tables["proj_tables"][0]                  # (n_proj, n_q)

    # Reference: direct QE-simpsn integral over kkbeta points, per q.
    kk = sp.kkbeta
    w_qe = _qe_simpsn_weights(kk) * sp.rab[:kk]
    w_old = _simpson_weights(len(sp.r)) * sp.rab  # pre-fix: full mesh
    ip = int(np.argmax(sp.proj_l))                # an l=2 projector
    l = int(sp.proj_l[ip])
    integ = sp.beta_r[ip] * sp.r ** 2             # r^2 (beta/r) = r * beta_upf
    worst_fix = worst_old = 0.0
    for iq in (30, 60, 90):
        jl = spherical_jn(l, q[iq] * sp.r)
        ref = float(np.sum(integ[:kk] * jl[:kk] * w_qe))
        old = float(np.sum(integ * jl * w_old))
        worst_fix = max(worst_fix, abs(F[ip, iq] - ref))
        worst_old = max(worst_old, abs(old - ref))
    assert worst_fix < 1e-12
    # RED TWIN: the old full-mesh generic weights disagree at ~1e-6 for
    # this l=2 projector (beta ends on ~1e-4 nonzero values at kkbeta),
    # the +0.004 meV V_NL diagonal constant.
    assert worst_old > 1e-8


def test_qe_simpsn_weights_match_upflib_conventions():
    """Odd n: standard composite Simpson.  Even n: QE's simpsn drops the
    last point and closes with net 1/3 on the one before -- i.e. the odd
    rule on the first n-1 points, NOT scipy's even-interval handling."""
    from psp.radial_tables import _qe_simpsn_weights

    w7 = _qe_simpsn_weights(7)
    assert np.allclose(w7 * 3.0, [1, 4, 2, 4, 2, 4, 1])
    w8 = _qe_simpsn_weights(8)
    assert w8[7] == 0.0
    assert np.allclose(w8[:7], w7)


# ---------------------------------------------------------------------------
# 2026-08-28 follow-up: the SAME QE schemes on the scf-potential lane
# (build_ionic_and_core / scf_potential — LORRAX's own NSCF solver path).
# The kih-producer port left radial_tables.alpha_z and the
# build_all_tables vloc/nlcc rows on full-mesh generic-Simpson weights;
# these tests pin the ported scheme with independent references and carry
# the old scheme as a red twin, exactly like the producer-lane tests above.
# QE references: upflib/vloc_mod.f90 init_tab_vloc (q>0 AND q=0 branches),
# upflib/rhoc_mod.f90 init_tab_rhc — both integrate DO ir = 1, msh.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _staged_upfs(), reason="no staged Si UPFs")
@pytest.mark.parametrize("tag,path", _staged_upfs())
def test_alpha_z_uses_qe_msh_scheme_and_full_mesh_simpson_fails(tag, path):
    from scipy.integrate import simpson as scipy_simpson

    from psp.radial_tables import (_simpson_weights, alpha_z,
                                   qe_vloc_radial_scheme)
    from psp.species import extract_species
    from psp.upf.load_upf import load_upf
    from psp.upf.normalize import normalize_dataclass

    sp, = extract_species({"Si": normalize_dataclass(load_upf(path))})
    r, rab, v, z = sp.r, sp.rab, sp.vloc_r, sp.z_valence
    e2 = 2.0
    vol = 270.1071612068362                  # Si 4x4x4 deck cell volume

    msh, _w = qe_vloc_radial_scheme(r, rab)
    f = r * (r * v + z * e2)
    ref = float(scipy_simpson(f[:msh], x=r[:msh]))
    got = alpha_z(sp, vol) * vol / (4.0 * np.pi)
    assert abs(got - ref) < 1e-10

    # RED TWIN: the pre-port scheme — generic Simpson over the FULL mesh —
    # keeps the r > 10 tail where tabulated vloc deviates from -Z e2/r:
    # measured |old-ref| = 6.4e-7 (SR) / 5.4e-5 (FR) Ry*bohr^3, i.e.
    # -0.0008 / +0.0680 meV on every KIH diagonal of this deck.
    old = float(np.sum(f * _simpson_weights(len(r)) * rab))
    assert abs(old - ref) > 1e-7


@pytest.mark.skipif(not _staged_upfs(), reason="no staged Si UPFs")
@pytest.mark.parametrize("tag,path", _staged_upfs())
def test_vloc_nlcc_tables_use_qe_msh_scheme_and_full_mesh_fails(tag, path):
    import dataclasses

    from scipy.special import erf as scipy_erf, spherical_jn

    from psp.radial_tables import (_simpson_weights, build_all_tables,
                                   core_charge_table, qe_vloc_radial_scheme,
                                   vloc_sr_table)
    from psp.species import extract_species
    from psp.upf.load_upf import load_upf
    from psp.upf.normalize import normalize_dataclass

    sp, = extract_species({"Si": normalize_dataclass(load_upf(path))})
    r, rab = sp.r, sp.rab
    e2 = 2.0
    # Synthetic core charge with FULL-mesh support: the real Si rho_core
    # is identically zero beyond ~r=6, so truncation cannot show on it —
    # this row exists to prove the nlcc integral runs the msh scheme.
    sp_syn = dataclasses.replace(sp, rho_core_r=1.0 / (1.0 + r * r),
                                 has_nlcc=True)
    tables = build_all_tables([sp, sp_syn], q_max=5.0, n_q=101)
    q = tables["q"]

    msh, w_qe = qe_vloc_radial_scheme(r, rab)
    w_old = _simpson_weights(len(r)) * rab

    # vloc row at a low-q grid point (largest tail sensitivity)
    safe_r = np.where(r > 0, r, 1.0)
    a = sp.vloc_r + sp.z_valence * e2 * np.where(
        r > 0, scipy_erf(r) / safe_r, 2.0 / np.sqrt(np.pi))
    iq = 10
    g = spherical_jn(0, q[iq] * r) * r * r * a
    ref = float(np.sum(g[:msh] * w_qe))
    old = float(np.sum(g * w_old))
    assert abs(tables["vloc"][0, iq] - ref) < 1e-12
    # RED TWIN: full-mesh generic Simpson misses by 4.8e-7 (SR) /
    # 2.1e-6 (FR) — the scf-lane sibling of the producer's vloc defect.
    assert abs(old - ref) > 1e-7

    # nlcc row (synthetic full-support core charge)
    for iq in (10, 60):
        g = spherical_jn(0, q[iq] * r) * r * r * sp_syn.rho_core_r
        ref = float(np.sum(g[:msh] * w_qe))
        old = float(np.sum(g * w_old))
        assert abs(tables["nlcc"][1, iq] - ref) < 1e-12
        # RED TWIN: 6.6e-2 / 9.3e-4 — the msh truncation is load-bearing.
        assert abs(old - ref) > 1e-4

    # The numpy twins (vloc_sr_table / core_charge_table) must match the
    # batched JAX rows: two independent Bessel implementations (scipy vs
    # the Miller-recurrence kernel), one quadrature convention.
    assert np.max(np.abs(vloc_sr_table(sp, q) - tables["vloc"][0])) < 1e-10
    assert np.max(np.abs(core_charge_table(sp, q) - tables["nlcc"][0])) < 1e-10


@pytest.mark.skipif(not _SR_SI.exists(), reason=f"{_SR_SI} not staged")
def test_projector_numpy_twins_match_batched_tables():
    """projector_table / projector_deriv_table (kkbeta scheme) against the
    batched build_all_tables rows — scipy Bessel vs the JAX Miller kernel,
    one quadrature convention.  Guards the twins against future drift."""
    from psp.radial_tables import (build_all_tables, projector_deriv_table,
                                   projector_table)
    from psp.species import extract_species
    from psp.upf.load_upf import load_upf
    from psp.upf.normalize import normalize_dataclass

    sp, = extract_species({"Si": normalize_dataclass(load_upf(_SR_SI))})
    tables = build_all_tables([sp], q_max=5.0, n_q=101)
    q = tables["q"]
    F = tables["proj_tables"][0]
    H = tables["deriv_tables"][0]

    for ip in range(sp.n_proj):
        l = int(sp.proj_l[ip])
        assert np.max(np.abs(projector_table(sp, ip, q) - F[ip])) < 1e-10
        D_ref = -H[ip].copy() if l == 0 else np.concatenate(
            [[0.0], -H[ip][1:] / q[1:] ** l])
        assert np.max(np.abs(projector_deriv_table(sp, ip, q) - D_ref)) < 1e-10
