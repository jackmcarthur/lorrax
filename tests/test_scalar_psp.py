"""Scalar-relativistic (nspinor=1) pseudopotential semantics.

Pins the nspinor=1 generalization of the V_NL spin-orbit resolution
(2026-08-28) and its refusals:

* ``resolve_soc_mode``: SR pseudos resolve False for ANY nspinor; an FR
  pseudo with nspinor=1 is FORCED to the j-averaged operator when soc is
  undeclared (QE ``average_pp`` for any lspinorb=.false. start) and
  REFUSES soc=True — a j-resolved V_NL has no representation on
  one-component wavefunctions.
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
# resolve_soc_mode
# ---------------------------------------------------------------------------

def test_sr_pseudo_resolves_false_for_any_nspinor_and_any_request():
    from psp.vnl_ops import resolve_soc_mode

    sr, _ = _sr_pseudo()
    for nspinor in (1, 2):
        for soc in (None, False, True):
            lines = []
            assert resolve_soc_mode({"Si": sr}, soc=soc, nspinor=nspinor,
                                    print_fn=lines.append) is False
            if soc is None:
                # the quiet contract: no choice exists, so no announcement
                assert lines == []


def test_fr_nspinor2_undeclared_keeps_jresolved_and_announces():
    # No-op pin of the historical nspinor=2 behaviour: the scalar work must
    # not have moved this banner by so much as a branch.
    from psp.vnl_ops import resolve_soc_mode

    lines = []
    assert resolve_soc_mode({"Mo": _fr_pseudo()}, soc=None, nspinor=2,
                            print_fn=lines.append) is True
    text = "\n".join(lines)
    assert "UNDETERMINED" in text
    assert "j-RESOLVED" in text


def test_fr_nspinor1_undeclared_is_forced_averaged_and_says_so():
    from psp.vnl_ops import resolve_soc_mode

    lines = []
    assert resolve_soc_mode({"Mo": _fr_pseudo()}, soc=None, nspinor=1,
                            print_fn=lines.append) is False
    text = "\n".join(lines)
    # Red twin of a silent resolution: the forced choice must be announced,
    # with the discarded spin-orbit strength on the record.
    assert "j-AVERAGED" in text
    assert "nspinor=1" in text
    assert "0.200000" in text          # |0.3 - 0.1| Ry, the ΔD at stake


def test_fr_nspinor1_soc_true_refuses():
    from psp.vnl_ops import resolve_soc_mode

    with pytest.raises(ValueError, match="no representation") as err:
        resolve_soc_mode({"Mo": _fr_pseudo()}, soc=True, nspinor=1)
    msg = str(err.value)
    for needle in ("got:", "want:", "fix:", "nspinor=1", "average_pp"):
        assert needle in msg, f"refusal must carry {needle!r}: {msg}"


def test_fr_nspinor1_qe_spinorbit_true_also_refuses_naming_the_source():
    # The contradictory-file case: a wfn claiming lspinorb with 1-component
    # wavefunctions is not honorable either, and the refusal must say WHERE
    # the soc=True came from.
    from psp.vnl_ops import resolve_soc_mode

    wfn = SimpleNamespace(spinorbit=True)
    with pytest.raises(ValueError, match="QE <spinorbit>"):
        resolve_soc_mode({"Mo": _fr_pseudo()}, wfn, soc=None, nspinor=1)


def test_fr_nspinor2_soc_true_does_not_refuse():
    # Green twin of the refusal: the same request on 2-component
    # wavefunctions is the historical, correct path.
    from psp.vnl_ops import resolve_soc_mode

    assert resolve_soc_mode({"Mo": _fr_pseudo()}, soc=True, nspinor=2,
                            print_fn=lambda *_: None) is True


def test_fr_nspinor1_soc_false_is_the_ordinary_averaged_path():
    from psp.vnl_ops import resolve_soc_mode

    lines = []
    assert resolve_soc_mode({"Mo": _fr_pseudo()}, soc=False, nspinor=1,
                            print_fn=lines.append) is False
    assert any("j-AVERAGED" in ln for ln in lines)


def test_resolve_soc_mode_requires_nspinor():
    # nspinor=1 was a silent default inconsistent with the rest of the
    # module (build_vnl_setup resolves it from the WFN); it is now required.
    from psp.vnl_ops import resolve_soc_mode

    with pytest.raises(TypeError):
        resolve_soc_mode({"Mo": _fr_pseudo()}, soc=False)


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
    assert resolve_soc_mode({"Si": p}, soc=None, nspinor=1,
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
