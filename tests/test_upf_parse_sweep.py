"""UPF parse-and-build coverage across pseudopotential shapes (2026-08-28).

Distilled from the 144-file sweep over the staged PseudoDojo NC v0.4 PBE
standard SR set and the FR assets set (
``~/projects/nonspinor_2026-08-28/runs/atoms_sweep/parse_sweep.tsv``): five
representative files pin the axes the full sweep measured — smallest lmax
with an ODD per-ℓ projector count and no NLCC (H), the NLCC + even-kkbeta
workhorse (SR Si), the highest lmax that exists in either set (ℓ=3, Cs; NO
file reaches ℓ>3), the PP_SPIN_ORB attach + j-pair machinery (FR Si), and
the FR 5d j-average NaN repro (FR Au).

Every file is skip-if-absent, like ``test_scalar_psp``'s staged-Si tests.
Header facts (nbeta, per-ℓ counts, mesh, kkbeta) are pinned to the staged
v0.4 files on purpose: a re-download that changes them should trip the test,
not drift silently.

Sweep facts these five stand in for: all 144 files parse; all meshes AND all
kkbeta are even (the odd-parity file axis does not exist in these sets — the
odd-n quadrature branch is still exercised on every file because
``qe_vloc_radial_scheme`` forces msh odd); all PP_DIJ are exactly diagonal;
the ONLY build failures are the six FR 5d metals (Au, Hf, Hg, Ir, Os, Re)
whose first ℓ=1 shell has opposite-sign D_jj, where the j-AVERAGED arm
silently manufactures NaN (the xfail below).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_SR_DIR = Path.home() / "projects/nonspinor_2026-08-28/pseudos/nc-sr-04_pbe_standard"
_FR_DIR = Path("/home/jackm/SOURCES/agent_tasks/qe-gw-pipeline/assets/pseudos_standard")


def _load(path):
    from psp.upf.load_upf import load_upf
    from psp.upf.normalize import normalize_dataclass

    return normalize_dataclass(load_upf(path))


# tag -> (path, element, relativistic, per_l counts, nlcc, mesh, kkbeta)
_CASES = {
    "sr_H_min_lmax_no_nlcc": (
        _SR_DIR / "H.upf", "H", "scalar", {0: 2, 1: 1}, False, 1166, 104),
    "sr_Si_nlcc_even_kkbeta": (
        _SR_DIR / "Si.upf", "Si", "scalar", {0: 2, 1: 2, 2: 2}, True, 1510, 196),
    "sr_Cs_highest_lmax": (
        _SR_DIR / "Cs.upf", "Cs", "scalar", {0: 2, 1: 2, 2: 2, 3: 2}, True, 2414, 232),
    "fr_Si_spin_orb": (
        _FR_DIR / "Si.upf", "Si", "full", {0: 2, 1: 4, 2: 4}, True, 1528, 196),
    "fr_Au_javg_nan_shape": (
        _FR_DIR / "Au.upf", "Au", "full", {0: 2, 1: 4, 2: 4, 3: 2}, True, 1356, 164),
}


def _param_cases():
    return [pytest.param(tag, marks=pytest.mark.skipif(
        not _CASES[tag][0].exists(), reason=f"{_CASES[tag][0]} not staged"))
        for tag in _CASES]


@pytest.mark.parametrize("tag", _param_cases())
def test_parse_extract_eblocks_tables(tag):
    from psp import radial_tables
    from psp.radial.build_projectors_qe import (
        build_E_blocks_full, pseudo_has_j_channels)
    from psp.species import extract_species
    from psp.upf.upf_model_2_0_1 import UpfLogical

    path, element, rel, per_l, nlcc, mesh, kkbeta = _CASES[tag]
    p = _load(path)

    hdr = p.pp_header
    assert hdr.element.strip() == element
    assert getattr(hdr.relativistic, "value", hdr.relativistic) == rel
    is_fr = rel == "full"
    assert pseudo_has_j_channels(p) is is_fr
    nlcc_flag = hdr.core_correction in (UpfLogical.TRUE, UpfLogical.T)
    assert nlcc_flag is nlcc

    sp, = extract_species({element: p})
    ls, counts = np.unique(sp.proj_l, return_counts=True)
    assert dict(zip(ls.tolist(), counts.tolist())) == per_l
    assert sp.n_proj == sum(per_l.values())
    assert sp.has_nlcc is nlcc
    # the staged v0.4 facts the sweep measured, pinned on purpose
    assert len(sp.r) == mesh and mesh % 2 == 0
    assert sp.kkbeta == kkbeta and kkbeta % 2 == 0   # even-n simpsn branch
    # ONCVPSP writes strictly diagonal PP_DIJ in both sets
    off = sp.dij - np.diag(np.diag(sp.dij))
    assert not off.any()

    # FR files must have jjj attached on every ell>0 beta (PP_SPIN_ORB
    # attach through the PP_RELBETA preprocessing in load_upf)
    if is_fr:
        for b in p.pp_nonlocal.pp_beta:
            l = int(getattr(b, "lll", b.angular_momentum))
            if l > 0:
                assert getattr(b, "jjj", None) is not None

    # E blocks.  SR: the soc argument is inert and blocks are spin-diagonal.
    # FR: the j-RESOLVED arm must build finite for every file (the averaged
    # arm's 5d NaN is pinned separately below).
    arms = [True] if (is_fr and element == "Au") else [True, False]
    blocks = {soc: build_E_blocks_full(p, soc=soc) for soc in arms}
    for soc, bl in blocks.items():
        assert set(bl) == set(per_l)
        for l, E in bl.items():
            msize = 2 * l + 1
            assert E.shape == (2, 2, per_l[l] * msize, per_l[l] * msize)
            assert np.all(np.isfinite(E)), f"soc={soc} l={l} not finite"
    if not is_fr:
        for l in blocks[True]:
            assert np.array_equal(blocks[True][l], blocks[False][l])
            assert not blocks[True][l][0, 1].any()
            assert np.array_equal(blocks[True][l][0, 0], blocks[True][l][1, 1])

    # Radial tables on a small q-grid: all finite, NLCC row live iff NLCC.
    t = radial_tables.build_all_tables([sp], q_max=5.0, n_q=64)
    assert np.all(np.isfinite(t["vloc"])) and np.all(np.isfinite(t["nlcc"]))
    assert np.all(np.isfinite(t["proj_tables"][0]))
    assert np.all(np.isfinite(t["deriv_tables"][0]))
    assert bool(t["has_nlcc"][0]) is nlcc
    if nlcc:
        assert t["nlcc"][0].any()
    else:
        assert not t["nlcc"][0].any()


# ---------------------------------------------------------------------------
# lmax bounds.  No staged file reaches ell>3 (sweep fact), and the per-l
# eager path refuses ell=4 loudly.  Red twin + green twin.
# ---------------------------------------------------------------------------

def test_solid_harmonics_l3_works_and_l4_refuses():
    import jax.numpy as jnp

    from psp.radial.solid_harmonics import solid_harmonics_jax

    K = jnp.asarray(np.random.default_rng(0).standard_normal((5, 3)))
    S3 = solid_harmonics_jax(3, K)                 # green twin: ell=3 is real
    assert S3.shape == (7, 5) and bool(jnp.all(jnp.isfinite(S3)))
    with pytest.raises(NotImplementedError, match="l=4"):
        solid_harmonics_jax(4, K)                  # red twin: ell=4 refuses


def test_all_solid_harmonics_lmax4_refuses():
    """FIXED 2026-08-28: all_solid_harmonics used to pad rows above ell=3
    with silent zeros; it now refuses like solid_harmonics_jax, so a
    g-projector pseudo can never receive a silently zero V_NL channel
    (vnl_ops.py:364->573 consumes S_all rows by the pseudo's l_max)."""
    import jax.numpy as jnp
    import pytest as _pt

    from psp.radial.solid_harmonics import all_solid_harmonics

    K = jnp.asarray(np.random.default_rng(0).standard_normal((5, 3)))
    with _pt.raises(NotImplementedError, match="l_max=4"):
        all_solid_harmonics(K, l_max=4)
    S = all_solid_harmonics(K, l_max=3)            # green twin: l<=3 fine
    assert bool(jnp.all(jnp.isfinite(S)))


# ---------------------------------------------------------------------------
# FR 5d j-average refusal (was DEFECT-1, FIXED 2026-08-28).  The first
# ell=1 shell of the FR 5d PseudoDojo files (Au, Hf, Hg, Ir, Os, Re) has
# opposite-sign D_jj (Au: d(1/2)=-0.4798, d(3/2)=+2.5893 Ry), so
# average_pp's sqrt(D/D_avg) is imaginary and QE would NaN identically:
# build_E_blocks_full(soc=False) must refuse loudly, never return NaN.
# ---------------------------------------------------------------------------

_FR_AU = _FR_DIR / "Au.upf"
_FR_MO = _FR_DIR / "Mo.upf"


@pytest.mark.skipif(not _FR_AU.exists(), reason=f"{_FR_AU} not staged")
def test_fr_au_j_average_refuses_not_nan():
    from psp.radial.build_projectors_qe import build_E_blocks_full

    p = _load(_FR_AU)
    with pytest.raises(ValueError) as ei:
        build_E_blocks_full(p, soc=False)
    msg = str(ei.value)
    # The refusal must name the shell, both D values, and the fix — and
    # the fix must not point at soc flags, which no longer exist.
    assert "opposite-sign" in msg
    assert "-0.47" in msg and "2.58" in msg
    assert "nspinor=2 WFN" in msg and "nc-sr" in msg
    assert "soc=True" not in msg and "--soc" not in msg
    # Green twin 1: the j-RESOLVED arm of the same pseudo stays finite.
    E_res = build_E_blocks_full(p, soc=True)
    for _l, E in E_res.items():
        assert np.all(np.isfinite(E))


@pytest.mark.skipif(not _FR_MO.exists(), reason=f"{_FR_MO} not staged")
def test_fr_mo_j_average_still_builds():
    """Green twin 2 (over-fire control): a normal FR pseudo whose j pairs
    share the sign of their weighted mean must still j-average finitely --
    the guard discriminates, it does not blanket-refuse soc=False."""
    from psp.radial.build_projectors_qe import build_E_blocks_full

    E_avg = build_E_blocks_full(_load(_FR_MO), soc=False)
    for _l, E in E_avg.items():
        assert np.all(np.isfinite(E))


def test_pseudo_summary_lines_one_line_per_species_with_generator_tag():
    """The Pseudopotentials report block: ONE line per species carrying
    file, SR/FR verdict (beta channels win over a lying header), z_val,
    NLCC, and a short generator tag when the file names one."""
    from psp.pseudos import pseudo_summary_lines

    sr_path = _SR_DIR / "Si.upf"
    fr_path = _FR_DIR / "Au.upf"
    if not (sr_path.exists() and fr_path.exists()):
        pytest.skip("staged pseudo sets absent")
    sr, fr = _load(sr_path), _load(fr_path)
    # production load_pseudopotentials stamps the source path; _load is bare
    setattr(sr, "_source_path", str(sr_path))
    setattr(fr, "_source_path", str(fr_path))
    lines = pseudo_summary_lines({"Si": sr, "Au": fr})
    assert len(lines) == 2                        # ONE line per species
    text = "\n".join(lines)
    assert "Si.upf — scalar-relativistic" in text
    assert "Au.upf — fully-relativistic" in text
    assert "z_val 4" in text and "NLCC yes" in text
    # generator tag from PP_INFO ("ONCVPSP ... version 3.3.0"), ≤15 chars
    assert "(ONCVPSP-3.3.0)" in text
    # the old two-line detail went to the chopping block, not the report
    for gone in ("kkbeta", "mesh", "proj "):
        assert gone not in text
    # Red twin: lie in the SR header — the betas must win, loudly.
    class _Hdr:
        pass
    hdr = _Hdr()
    for a in ("element", "z_valence", "relativistic"):
        setattr(hdr, a, getattr(sr.pp_header, a, None))
    hdr.relativistic = "full"
    forged = type(sr).__new__(type(sr))
    forged.__dict__.update(sr.__dict__)
    forged.pp_header = hdr
    hdr.z_valence = sr.pp_header.z_valence
    lied_lines = pseudo_summary_lines({"Si": forged})
    assert len(lied_lines) == 1                   # still one line
    lied = lied_lines[0]
    assert "scalar-relativistic" in lied          # betas' verdict
    assert "beta channels win" in lied            # the disagreement is named
    # forged header has no `generated`; PP_INFO still resolves the tag
    assert "(ONCVPSP-3.3.0)" in lied
