"""
psp/operator_checks.py — Pre-flight checks for DFT operator construction.

Call `validate_operator_inputs(...)` before computing kin+ion, dipole
matrix elements, or any other quantity that depends on pseudopotentials
and the Coulomb truncation scheme.  The function raises immediately on
fatal problems (missing pseudopotentials, atom/PP mismatch) and returns
a validated `truncation_2d` flag derived from `sys_dim`.

Usage
-----
    from psp.operator_checks import validate_operator_inputs

    ctx = validate_operator_inputs(
        pseudos=pseudos,
        wfn=wfn,
        sys_dim=params.get("sys_dim", 3),
    )
    # ctx.truncation_2d  — bool, safe to pass to build_local_ionic_potential_on_G_total
    # ctx.pseudos        — the validated dict (same object, just confirms non-empty)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from common.log import get_logger, message, record  # noqa: E402

log = get_logger(__name__)

from wfn_loader import WfnLoader                                    # noqa: E402


@dataclass(frozen=True)
class OperatorContext:
    """Validated operator construction parameters."""

    truncation_2d: bool
    sys_dim: int
    pseudos: dict


def validate_operator_inputs(
    pseudos: dict,
    wfn: WfnLoader,
    sys_dim: int = 3,
    *,
    caller: str = "",
) -> OperatorContext:
    """Validate inputs before any operator matrix-element computation.

    Parameters
    ----------
    pseudos : dict
        Element-symbol → parsed UPF object, from ``load_pseudopotentials``.
    wfn : WFNReader
        Loaded wavefunction file (needed for atom types).
    sys_dim : int
        System dimensionality: 0 (molecule), 2 (slab), 3 (bulk).
    caller : str, optional
        Label for error messages (e.g. ``"get_kin_ion_chunked"``).

    Returns
    -------
    OperatorContext with validated ``truncation_2d`` flag and references.

    Raises
    ------
    RuntimeError
        If pseudopotentials are missing or do not cover all atom species,
        or if ``sys_dim`` is not one of {0, 2, 3}.
    """
    tag = f"[{caller}] " if caller else ""

    # ---- pseudopotentials must be present ----
    if not pseudos:
        raise RuntimeError(
            f"{tag}No pseudopotentials loaded.  "
            "Ensure *.upf files are in the working directory or pass an "
            "explicit pseudo_dir.  Without pseudopotentials V_loc and V_NL "
            "are zero and the resulting matrix elements are meaningless."
        )

    # ---- every atom species must have a matching PP ----
    import numpy as np

    atom_types = np.asarray(wfn.atom_types, dtype=int)
    unique_z = set(int(z) for z in atom_types)

    from psp.pseudos import symbol_to_Z as _symbol_to_Z

    covered_z = set()
    for elem in pseudos:
        z = _symbol_to_Z(elem)
        if z is not None:
            covered_z.add(z)

    missing = unique_z - covered_z
    if missing:
        raise RuntimeError(
            f"{tag}Pseudopotentials missing for atomic numbers {sorted(missing)}.  "
            f"Loaded PPs cover Z = {sorted(covered_z)}, "
            f"but the structure contains Z = {sorted(unique_z)}."
        )

    # ---- sys_dim → truncation_2d ----
    if sys_dim not in (0, 2, 3):
        raise RuntimeError(
            f"{tag}sys_dim={sys_dim} is not supported (must be 0, 2, or 3)."
        )

    truncation_2d = sys_dim == 2

    return OperatorContext(
        truncation_2d=truncation_2d,
        sys_dim=sys_dim,
        pseudos=pseudos,
    )


# ---------------------------------------------------------------------------
# Degeneracy consistency: does the operator have the symmetry of the states?
# ---------------------------------------------------------------------------

def check_degeneracy_consistency(
    H, energies, *, el_tol_ry: float = 1e-9, split_tol_ry: float = 1e-5,
    max_report: int = 8, label: str = "kin_ion", print_fn=None,
) -> dict:
    """Flag operator blocks that split a manifold the eigenvalues do not.

    THE ARGUMENT.  A set of bands degenerate in ``el`` spans an invariant
    subspace of the Hamiltonian that produced them.  ``H`` here is a PIECE of
    that Hamiltonian (T + V_loc + V_NL).  If ``H`` restricted to such a
    manifold is NOT a multiple of the identity, then ``H`` distinguishes
    states the full Hamiltonian did not — it has strictly lower symmetry than
    the operator the wavefunctions diagonalise.  For a fully-relativistic
    pseudopotential the usual cause is that V_NL was built j-RESOLVED while
    the wavefunctions came from a j-AVERAGED (``lspinorb=.false.``) run: the
    j-resolved V_NL carries spin-orbit, so it splits exactly the manifolds
    spin-orbit is allowed to split, in exactly the pattern it would produce.

    This needs NO metadata — not ``<spinorbit>``, not a deck key, not the UPF.
    It compares the operator against the wavefunctions it is being applied to,
    which is the only comparison that is always available.

    Eigenvalues of the restricted block are used, not diagonal entries: within
    a degenerate manifold the band basis is an arbitrary unitary, so only the
    block SPECTRUM is gauge-invariant.  A diagonal-only test reports scatter
    where there is a clean split and misses the structure entirely.

    WHEN THIS RETURNS CLEAN (constructed, not assumed):
      * scalar (nspinor=1) runs — V_NL is spin-scalar, cannot split;
      * genuine ``lspinorb=.true.`` runs — ``el`` is ALREADY split into the
        double-group multiplets, so each manifold is a single irrep and the
        j-resolved V_NL is a multiple of the identity on it;
      * FR pseudo + j-AVERAGED V_NL — spin-scalar by construction.
    Verified against this repo's fixtures: the Si ``lspinorb=.true.`` archive
    run is clean at Γ, the ``lspinorb=.false.`` one is not.

    KNOWN FALSE-POSITIVE CHANNEL, stated so it is not mistaken for proof: a
    manifold that is an ACCIDENTAL degeneracy of two distinct irreps may be
    split by a perfectly correct operator.  A split here means "this operator
    has lower symmetry than ``el``", which is a fact worth surfacing; it is
    evidence of a mode mismatch, not a proof of one.

    Parameters
    ----------
    H : (nk, nb, nb) — operator matrix in the band basis, Ry.  Anything that
        slices, including an open h5py dataset: only the manifold blocks
        are read.
    energies : (nk, nb) — the eigenvalues those bands carry, Ry.
    el_tol_ry : bands closer than this are treated as one manifold.
    split_tol_ry : a block spectrum spread above this is reported.

    Returns
    -------
    dict with ``n_manifolds``, ``n_split``, ``max_split_ry``, ``worst`` and
    ``clean``.
    """
    import numpy as np

    en = np.asarray(energies, dtype=np.float64)
    nk = min(H.shape[0], en.shape[0])
    nb = min(H.shape[1], en.shape[1])

    n_manifolds = 0
    findings = []
    max_split = 0.0

    for ik in range(nk):
        e = en[ik, :nb]
        i = 0
        while i < nb:
            j = i
            while j + 1 < nb and abs(e[j + 1] - e[i]) <= el_tol_ry:
                j += 1
            deg = j - i + 1
            if deg > 1:
                n_manifolds += 1
                blk = np.asarray(H[ik, i:j + 1, i:j + 1])
                w = np.linalg.eigvalsh((blk + blk.conj().T) / 2.0).real
                spread = float(w.max() - w.min())
                max_split = max(max_split, spread)
                if spread > split_tol_ry:
                    dev = w - w.mean()
                    # degeneracy pattern of the SPLIT spectrum: a 2+4 here is
                    # the Γ8/Γ7 signature, a scatter is something else.
                    pat, run = [], 1
                    srt = np.sort(dev)
                    for q in range(1, deg):
                        if abs(srt[q] - srt[q - 1]) <= max(split_tol_ry * 0.1,
                                                           1e-12):
                            run += 1
                        else:
                            pat.append(run)
                            run = 1
                    pat.append(run)
                    findings.append(dict(ik=ik, band0=i, deg=deg, el=float(e[i]),
                                         spread_ry=spread, pattern=pat,
                                         dev_ry=dev.tolist()))
            i = j + 1

    findings.sort(key=lambda d: -d["spread_ry"])
    clean = not findings
    RY2MEV = 13605.693122994

    if not clean:
        bar = "=" * 78
        summary = (f"{len(findings)} of {n_manifolds} degenerate manifolds "
                   f"(|Δel| ≤ {el_tol_ry:.1e} Ry) are split by more than "
                   f"{split_tol_ry * RY2MEV:.3f} meV.")
        worst = f"worst split = {max_split * RY2MEV:.4f} meV"
        table = [f"  {'k':>5} {'band0':>6} {'deg':>4} {'el (Ry)':>14} "
                 f"{'split (meV)':>13}  pattern"]
        for d in findings[:max_report]:
            table.append(f"  {d['ik']:5d} {d['band0']:6d} {d['deg']:4d} "
                         f"{d['el']:14.9f} {d['spread_ry'] * RY2MEV:13.4f}  "
                         f"{d['pattern']}")
        if len(findings) > max_report:
            table.append(f"  ... {len(findings) - max_report} more")
        story = ("This operator distinguishes states the wavefunctions'",
                 "Hamiltonian did not.  With a fully-relativistic",
                 "pseudopotential the usual cause is a j-RESOLVED V_NL built",
                 "against wavefunctions from noncolin=.true.,",
                 "lspinorb=.false. (QE ran average_pp; its eigenvalues carry",
                 "no spin-orbit).  A 2+4 pattern on a 6-fold at Γ is the",
                 "Γ8/Γ7 spin-orbit signature.  The V_NL builder reads",
                 "j-resolved vs j-averaged from QE <spinorbit>, so a split",
                 "HERE means the operator disagrees with the wavefunctions",
                 "anyway — check the WFN / pseudopotential pairing.")
        if print_fn is not None:
            print_fn(f"\n{bar}")
            print_fn(f"DEGENERACY MISMATCH: {label} splits manifolds that "
                     f"'el' does not.")
            print_fn(bar)
            print_fn(f"  {summary}")
            print_fn(f"  {worst}")
            print_fn("")
            for line in table:
                print_fn(line)
            print_fn("")
            for line in story:
                print_fn(f"  {line}")
            print_fn(bar + "\n")
        else:
            log.warning(
                f"{label} splits degenerate manifolds that the wavefunctions "
                f"do not",
                extra=message(
                    id="degeneracy_mismatch",
                    found=("split manifolds", f"{summary[:-1]}; {worst}"),
                    lines=(" ".join(story),),
                    fix="check the WFN / pseudopotential pairing"))
            for line in table:
                log.info(line)
            log.debug("degeneracy findings", extra=record(
                "operator_checks.degeneracy", label=label,
                n_manifolds=n_manifolds, n_split=len(findings),
                max_split_ry=max_split, worst=findings[:max_report]))
    else:
        say = print_fn if print_fn is not None else log.info
        say(f"  {label}: degeneracy-consistent — "
            f"{n_manifolds} degenerate manifolds, "
            f"max split {max_split * RY2MEV:.4f} meV "
            f"(≤ {split_tol_ry * RY2MEV:.3f} meV).")

    return dict(n_manifolds=n_manifolds, n_split=len(findings),
                max_split_ry=max_split, worst=findings[:max_report],
                clean=clean)


# ---------------------------------------------------------------------------
# QE-run preflight: can this QE run's Hamiltonian be rebuilt exactly?
# ---------------------------------------------------------------------------

#: QE <dft><functional> names whose V_xc ``psp.xc`` reproduces
#: (``dft_operators.compute_V_H_and_V_xc`` evaluates PBE only).  The NSCF and
#: dense-H drivers build the same V_scf (``psp.scf_potential``).
DENSE_H_FUNCTIONALS = ("PBE",)

#: Peak dense bytes per k, in units of ``N²·16`` (complex128), N = nspinor·ngk:
#: the unit-vector block, H, the eigenvectors and the solver workspace (two).
DENSE_H_MATRICES_PER_K = 5

#: An occupation farther than this from 0 or 1 in the QE schema makes the run
#: a metal (``gw.gw_config.infer_material_class``'s threshold).
QE_FRACTIONAL_OCCUPATION_TOL = 1.0e-6

#: The way out the refusals name when the psp operators cannot rebuild a run.
_QE_ROUTE = "QE nscf + pw2bgw.x, which writes this run's own bands"


class QERunRefusal(RuntimeError):
    """A QE run the psp operators cannot rebuild exactly.

    ``failures`` is every failing check as ``(rule, text)``; the message has
    one ``GATE <driver>_<rule>: got ...; want ...; why: ...; fix: ...`` line
    per failure, and ``rule`` is the first one's.
    """

    def __init__(self, driver: str, failures):
        self.driver = str(driver)
        self.failures = tuple((str(r), str(t)) for r, t in failures)
        self.rule = self.failures[0][0]
        self.rules = tuple(r for r, _ in self.failures)
        super().__init__("\n".join(f"GATE {self.driver}_{rule}: {text}"
                                   for rule, text in self.failures))


#: The dense-H driver's name for the same refusal (its ``rule`` and ``rules``).
DenseHRefusal = QERunRefusal


def _upf_flag(value) -> bool:
    value = getattr(value, "value", value)
    if isinstance(value, str):
        return value.strip().strip(".").lower() in ("t", "true")
    return bool(value)


def dense_h_bytes(n_basis: int) -> int:
    """Peak per-rank device bytes of one k's dense solve (N = nspinor·ngk)."""
    return DENSE_H_MATRICES_PER_K * int(n_basis) ** 2 * 16


def qe_occupations(save_dir: str) -> tuple[str, float]:
    """``(kind, max fractional part)`` of the QE run's occupations.

    ``kind`` is the schema's ``occupations_kind`` (else the input
    ``occupations``) with its smearing, e.g. ``"smearing (fd, degauss 0.005
    Ry)"``; the second value is the largest distance of an
    ``output/band_structure/ks_energies/occupations`` entry from 0 or 1.
    """
    import os
    import xml.etree.ElementTree as ET
    import numpy as np
    root = ET.parse(os.path.join(str(save_dir), "data-file-schema.xml")).getroot()
    bands = root.find("output/band_structure")
    kind = ((bands.findtext("occupations_kind") if bands is not None else None)
            or root.findtext("input/bands/occupations") or "unknown").strip()
    smearing = (bands.find("smearing") if bands is not None else None)
    if smearing is None:
        smearing = root.find("input/bands/smearing")
    if kind == "smearing" and smearing is not None:
        degauss = smearing.attrib.get("degauss")
        kind += (f" ({(smearing.text or '').strip()}"
                 + (f", degauss {2.0 * float(degauss):g} Ry" if degauss else "")
                 + ")")
    frac = 0.0
    for occ in (bands.iter("occupations") if bands is not None else ()):
        if occ.text and occ.text.split():
            v = np.asarray(occ.text.split(), dtype=np.float64)
            frac = max(frac, float(np.max(np.abs(v - np.rint(v)))))
    return kind, frac


def _qe_run_failures(crystal, pseudos: dict, *, pseudo_dir: str,
                     charge_density_fields: tuple, driver: str,
                     nc_gga_branch: str | None = None) -> list:
    """Every check a QE run fails, as ``(rule, text)``.

    ``driver`` is ``"nscf"`` or ``"dense_h"``.  Both rebuild V_scf =
    V_loc + V_H[rho] + V_xc[rho + rho_core] from the SCF density with
    norm-conserving V_NL (overlap S = 1).  Only dense-H carries B_xc (a
    noncollinear magnetic run, ``nc_gga_branch = "general"``) and the source
    WFN's occupations (a metal); NSCF refuses both, and collinear nspin = 2
    refuses on both.
    """
    import os
    failures = []
    pdir_fix = ("--pseudo-dir" if driver == "dense_h" else "--pseudo_dir")

    # ---- every species' UPF is on disk ----
    for species, fname in crystal.pseudo_files:
        if not os.path.isfile(os.path.join(pseudo_dir, fname)):
            failures.append((
                "upf_missing",
                f"got species {species!r} with no {fname!r} in {pseudo_dir}; "
                f"want the UPF the QE run used for every species; why: V_loc "
                f"and V_NL come from it; fix: copy {fname} beside the .save or "
                f"pass {pdir_fix} <dir>"))

    # ---- the functional ----
    if crystal.xc_extensions:
        failures.append((
            "xc_extension",
            f"got a QE run with {', '.join(crystal.xc_extensions)}; want a "
            f"semilocal functional alone; why: the psp operators rebuild "
            f"semilocal V_xc only (no hybrid, vdW or +U term); fix: {_QE_ROUTE}"))
    if crystal.functional not in DENSE_H_FUNCTIONALS:
        failures.append((
            "functional",
            f"got QE functional {crystal.functional or '<absent>'!r}; want "
            f"{', '.join(DENSE_H_FUNCTIONALS)}; why: the psp V_xc evaluates "
            f"{', '.join(DENSE_H_FUNCTIONALS)} only "
            f"(psp.dft_operators.compute_V_H_and_V_xc); fix: a PBE QE run, or "
            f"{_QE_ROUTE}"))

    # ---- norm-conserving pseudopotentials of the same functional ----
    for elem, pp in pseudos.items():
        h = pp.pp_header
        kind = str(getattr(h.pseudo_type, "value", h.pseudo_type) or "").upper()
        if (kind not in ("NC", "SL") or _upf_flag(h.is_ultrasoft)
                or _upf_flag(h.is_paw)):
            failures.append((
                "pseudo_type",
                f"got {elem} UPF pseudo_type={kind!r}; want norm-conserving; "
                f"why: H(G,G') is built with overlap S = 1, which holds for "
                f"norm-conserving pseudopotentials only (no USPP or PAW); fix: "
                f"a norm-conserving UPF in the QE run, or {_QE_ROUTE}"))
        pp_functional = str(h.functional or "").strip().upper()
        if pp_functional and pp_functional != crystal.functional:
            failures.append((
                "functional",
                f"got {elem} UPF generated with {pp_functional!r} in a "
                f"{crystal.functional!r} QE run; want one functional; why: the "
                f"rebuilt V_xc is the run's, and the UPF's core and projectors "
                f"belong to another; fix: a UPF of the run's functional"))

    # ---- spin and magnetism ----
    if int(crystal.nspin) == 2:
        failures.append((
            "nspin2",
            "got a collinear spin-polarized QE run (nspin = 2); want nspin 1; "
            "why: H_k and the WFN writer carry one spin channel; fix: run QE "
            "noncollinear (noncolin = .true.), or " + _QE_ROUTE))
    if bool(crystal.domag) and driver == "nscf":
        failures.append((
            "magnetism",
            "got a magnetic QE run (do_magnetization true); want a nonmagnetic "
            "one; why: psp.run_nscf builds no B_xc term and folds k by time "
            "reversal; fix: psp.run_dense_h --nc-gga-branch general for a small "
            "cell, or " + _QE_ROUTE))
    if bool(crystal.domag) and driver == "dense_h" and nc_gga_branch != "general":
        failures.append((
            "nc_gga_branch",
            f"got a noncollinear magnetic GGA run with nc_gga_branch="
            f"{nc_gga_branch!r}; want --nc-gga-branch general; why: the .save "
            f"does not record which branch QE took, and the fixed-axis branch "
            f"(lsign = .true.) needs QE's ux, which the .save does not keep; "
            f"fix: add --nc-gga-branch general if the QE SCF output has no "
            f"'Fixed quantization axis for GGA' line (the patched SOC pw.x or "
            f"nonparallel seeds)"))

    # ---- metals ----
    if driver == "nscf":
        kind, frac = qe_occupations(crystal._save_dir)
        if frac > QE_FRACTIONAL_OCCUPATION_TOL:
            failures.append((
                "metal",
                f"got a metal: occupations {kind}, fractional by up to "
                f"{frac:.3g}; want integer occupations (an insulator or a "
                f"semiconductor); why: psp.run_nscf writes insulator "
                f"occupations; fix: {_QE_ROUTE}, with the smeared occupations"))

    # ---- the density the potential is built from ----
    if "rhotot_g" not in charge_density_fields:
        failures.append((
            "charge_density",
            f"got no charge-density.hdf5 with rhotot_g in {crystal._save_dir}; "
            f"want the SCF density; why: V_H and V_xc are built from it; fix: "
            f"point --save at the SCF run's .save"))
    missing_m = [f for f in ("m_x", "m_y", "m_z")
                 if f not in charge_density_fields]
    if bool(crystal.domag) and driver == "dense_h" and missing_m:
        failures.append((
            "charge_density",
            f"got a magnetic run whose charge-density.hdf5 lacks {missing_m}; "
            f"want m_x, m_y and m_z; why: B_xc is built from the SCF "
            f"magnetization; fix: point --save at the SCF run's .save"))
    return failures


def validate_nscf_inputs(crystal, pseudos: dict, *, pseudo_dir: str,
                         charge_density_fields: tuple) -> None:
    """Refuse, before any work, a QE run ``psp.run_nscf`` cannot rebuild.

    Raises one :class:`QERunRefusal` naming every failing check
    (``GATE nscf_<rule>``): ``upf_missing``, ``xc_extension`` (hybrid, vdW,
    +U), ``functional`` (non-PBE), ``pseudo_type`` (USPP, PAW), ``nspin2``,
    ``magnetism``, ``metal``, ``charge_density``.  The checks read the parsed
    QE schema and the UPF headers only.  The B_xc path and magnetic k folding
    that would let magnetic runs through are a to-do.
    """
    failures = _qe_run_failures(
        crystal, pseudos, pseudo_dir=pseudo_dir,
        charge_density_fields=charge_density_fields, driver="nscf")
    if failures:
        raise QERunRefusal("nscf", failures)


def validate_dense_h_inputs(
    crystal,
    pseudos: dict,
    *,
    sys_dim: int,
    pseudo_dir: str,
    charge_density_fields: tuple,
    n_basis_max: int,
    budget_bytes: float,
    nc_gga_branch: str | None = None,
) -> OperatorContext:
    """Refuse, before any heavy work, a QE run whose H the psp operators cannot rebuild.

    ``crystal`` is the ``CrystalData`` of the QE ``.save``; the checks read
    only its parsed XML facts, so each refusal class is testable without
    files.  Every failing check goes into one :class:`QERunRefusal`
    (``DenseHRefusal``), ``GATE dense_h_<rule>`` with a ``fix:`` each:
    ``upf_missing``, ``functional``, ``xc_extension``, ``pseudo_type``,
    ``nspin2``, ``nc_gga_branch``, ``charge_density``, ``truncation_2d``.
    Dense matrices over ``budget_bytes`` warn, not refuse.  On success the
    validated operator context comes back.

    Magnetism: a noncollinear magnetic run (``do_magnetization``) is
    rebuilt with V_xc = v δ + B·σ from ρ and m (``psp.xc``), in QE's general
    noncollinear GGA branch (``lsign = .false.``).  The ``.save`` does not
    record which branch the QE run took (stock QE fixes a quantization axis
    when every starting moment is parallel; the sandbox's patched QE and
    tilted seeds do not), so the caller states it: ``nc_gga_branch``
    ``"general"`` is rebuilt, anything else refuses.  Collinear ``nspin = 2``
    refuses: H_k and the WFN writer carry one spin channel.  A metal passes:
    the written WFN keeps the source WFN's occupations.
    """
    failures = _qe_run_failures(
        crystal, pseudos, pseudo_dir=pseudo_dir,
        charge_density_fields=charge_density_fields, driver="dense_h",
        nc_gga_branch=nc_gga_branch)

    # ---- Coulomb truncation agrees with the QE run ----
    qe_2d = crystal.assume_isolated == "2D"
    if (int(sys_dim) == 2) != qe_2d:
        failures.append((
            "truncation_2d",
            f"got --sys-dim {int(sys_dim)} with QE assume_isolated="
            f"{crystal.assume_isolated!r}; want sys_dim 2 exactly when "
            f"assume_isolated is '2D'; why: V_H must use the truncation the "
            f"SCF used; fix: --sys-dim {2 if qe_2d else 3}"))
    if failures:
        raise QERunRefusal("dense_h", failures)

    # ---- the dense matrices against the one device budget (warn only) ----
    need = dense_h_bytes(n_basis_max)
    if need > float(budget_bytes):
        from lxkit import memory
        memory.admit(f"dense H_k (N = {int(n_basis_max)})",
                     memory.priced(need, "closed form"), budget_bytes, device=False)

    return validate_operator_inputs(pseudos, crystal, sys_dim=int(sys_dim),
                                    caller="dense_h")
