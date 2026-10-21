"""``dipole.h5``: the q→0 velocity matrix, its band energies, finite-q SOS.

Layout (``psp.get_dipole_mtxels`` writes it, the GW head and BSE absorption
read it):

* ``dipole_cart`` — ``(3, nk, nb, nb)`` complex128, ⟨mk|v̂_α|nk⟩ in Ry
  units on the full BZ: the band velocity ``v = i[H, r] = p + i[V_NL, r]``
  (``p`` alone when the root attribute ``skip_vnl`` is set).
* ``band_energies`` — ``(nk, nb)`` float64, the WFN mean-field eigenvalue of
  each band at each full-BZ k (Ry): the star parent's row, which is the
  energy ψ(Sk) carries.
* ``finite_q/`` — optional finite-q SOS block (``rho_cvkq``, ``v_cvkq``,
  ``kminq_idx``, ``iq_list``, and for bispinors ``alpha_cvkq`` and
  ``ward_residual_cvkq``).
* root attributes — ``nbands``, ``nk``, ``skip_vnl``, ``note``, ``basis``
  (``"dft"`` or ``"qp"``: whose states the band indices label; absent reads
  ``"dft"``), ``format_version`` (:data:`DIPOLE_FORMAT_VERSION`),
  ``band_min``/``band_max`` (the absolute band range ``[0, nb)`` the band axis
  holds; ``nbands`` is the WFN's band count, not the stored one; readers gate
  their window with :func:`require_dipole_bands`) and the ``prov_*``
  provenance stamp (``psp.get_dipole_mtxels.dipole_provenance``).

FORMAT VERSION.  Version 2 (2026-10-20) is the first format whose velocity
can only be ``i[H, r]``.  Every reader calls :func:`require_dipole_format`
first, so a file without the stamp (version 1, which may hold the retired
``p - i[V_NL, r]`` assembly) refuses by name instead of being read.

ΔE IS DERIVED ON READ.  ``ΔE[k, m, n] = E_m(k) − E_n(k)`` carries no
information beyond ``band_energies`` and was 14 % of every file.
:func:`delta_e` and :func:`delta_e_cv` rebuild it with the expression the
writer used to store, so the result is bit-identical.  A file written before
2026-09-25 stores ``deltaE`` and no ``band_energies``; the readers take the
stored table then.

The velocity is written through :mod:`file_io.slab_io` from its shards: no
rank gathers ``(nk, 3, nb, nb)``.  The finite-q arrays arrive replicated on
the host and land in SlabIO's rank-0 metadata reopen, exactly as the serial
writer wrote them.
"""
from __future__ import annotations

import numpy as np

DIPOLE_DATASET = "dipole_cart"
BAND_ENERGIES_DATASET = "band_energies"
#: Stored ΔE in files written before 2026-09-25; read, never written.
LEGACY_DELTA_E_DATASET = "deltaE"
FINITE_Q_GROUP = "finite_q"
#: Root attribute: the one-particle basis the band indices label.
DIPOLE_BASIS_ATTR = "basis"
#: Root attribute and value of the current layout; absent reads 1.
DIPOLE_FORMAT_ATTR = "format_version"
DIPOLE_FORMAT_VERSION = 2
#: Root attributes: the absolute 0-based band range ``[band_min, band_max)``
#: the stored band axis holds.  Every writer stores from band 0; a file
#: written before the stamp holds ``[0, nb_stored)``.
DIPOLE_BAND_MIN_ATTR = "band_min"
DIPOLE_BAND_MAX_ATTR = "band_max"

__all__ = [
    "BAND_ENERGIES_DATASET",
    "DIPOLE_BAND_MAX_ATTR",
    "DIPOLE_BAND_MIN_ATTR",
    "DIPOLE_BASIS_ATTR",
    "DIPOLE_DATASET",
    "DIPOLE_FORMAT_ATTR",
    "DIPOLE_FORMAT_VERSION",
    "FINITE_Q_GROUP",
    "LEGACY_DELTA_E_DATASET",
    "band_energies_on_full_bz",
    "require_dipole_bands",
    "require_dipole_basis",
    "require_dipole_format",
    "require_dipole_k_grid",
    "velocity_stamp",
    "wfn_psi_basis",
    "delta_e",
    "delta_e_cv",
    "read_band_energies",
    "energy_extent",
    "finite_q_payload",
    "write_dipole",
]


def band_energies_on_full_bz(wfn, sym, nb: int) -> np.ndarray:
    """``(nk_full, nb)`` WFN eigenvalues, each full-BZ k at its star parent.

    ``sym.irr_idx_k`` must address every full-BZ k.  A table that does not
    refuses by name (``GATE dipole_energy_star_map``); it used to fall back
    to the k's own index, which silently pairs a k with another k's
    energies.
    """
    energies = np.asarray(wfn.energies, dtype=np.float64)
    energies = energies[0] if energies.ndim == 3 else energies
    nk = int(sym.nk_tot)
    parent = np.asarray(sym.irr_idx_k)
    if (energies.ndim != 2 or parent.shape != (nk,)
            or parent.size == 0 or int(parent.min()) < 0
            or int(parent.max()) >= energies.shape[0]
            or energies.shape[1] < int(nb)):
        raise ValueError(
            "GATE dipole_energy_star_map: got irr_idx_k shape "
            f"{parent.shape} over WFN energies {energies.shape}; want one "
            f"parent row per full-BZ k (nk={nk}) inside the WFN's "
            f"{energies.shape[0] if energies.ndim == 2 else '?'} rows and "
            f">= {int(nb)} bands; why: a missing parent pairs a k with "
            "another k's energies; fix: rebuild the symmetry tables from "
            "this WFN (WfnLoader.symmetry()).")
    return np.ascontiguousarray(energies[parent, :int(nb)])


def finite_q_payload(*, rho_cvkq, v_cvkq, kminq_idx, iq_list, n_occ, v_lo,
                     c_hi, alpha_cvkq=None, ward_residual_cvkq=None) -> dict:
    """The ``finite_q/`` group for :func:`write_dipole`: datasets and attrs."""
    datasets = {
        "rho_cvkq": (rho_cvkq, None),
        "v_cvkq": (v_cvkq, None),
        "kminq_idx": (kminq_idx, None),
        "iq_list": (np.asarray(iq_list, dtype=np.int32), None),
    }
    attrs = {
        "n_occ": int(n_occ), "v_lo": int(v_lo), "c_hi": int(c_hi),
        "note": (
            "rho_cvkq[c, v, k, q] = <u_{c, k-q}|u_{v, k}>_cell; "
            "v_cvkq[a, c, v, k, q] = symmetric (v_R + v_L)/2 of "
            "<u_{c, k-q}|v^a|u_{v, k}>_cell  (kinetic + VNL); "
            "kminq_idx[k, q] = canonical k-q index in unfolded_kpts."),
    }
    if alpha_cvkq is not None:
        from common.bispinor_init import (
            ALPHA_FS, DIRAC_ALPHA_VERTEX_PROVENANCE,
            KINETIC_BALANCE_LIFT_PROVENANCE, NO_PAIR_DIRAC_CURRENT_MODEL)
        datasets["alpha_cvkq"] = (alpha_cvkq, {
            "operator": "<u_{c,k-q}|alpha_i=gamma^0 gamma^i|u_{v,k}>_cell",
            "units": "dimensionless",
            "normalization": (
                "same unrenormalized kinetic-balance four-spinors as rho_cvkq"),
        })
        datasets["ward_residual_cvkq"] = (ward_residual_cvkq, {
            "units": "rydberg",
            "formula": ("(E_c(k-q)-E_v(k))_Ry*rho_cvkq + "
                        "q_cart_bohr^-1 dot (2*alpha_cvkq/alpha_fs)"),
            "energy_source": "WFN mean-field eigenvalues",
        })
        attrs.update({
            "selected_current_model": NO_PAIR_DIRAC_CURRENT_MODEL,
            "selected_current_lift": KINETIC_BALANCE_LIFT_PROVENANCE,
            "selected_current_operator": DIRAC_ALPHA_VERTEX_PROVENANCE,
            "selected_current_gauge_completion": "none_diagnostic_only",
            "alpha_fs": float(ALPHA_FS),
        })
    return {"datasets": datasets, "attrs": attrs}


def wfn_psi_basis(wfn_path) -> str:
    """``"qp"`` when ``wfn_path`` carries the QP-WFN stamp, else ``"dft"``."""
    from .restart_bundle import read_qp_wfn_stamp
    return "qp" if read_qp_wfn_stamp(wfn_path) is not None else "dft"


def require_dipole_basis(attrs, *, dipole_path, wfn_path) -> None:
    """Refuse a dipole whose band basis is not the basis of ``wfn_path``'s ψ.

    ``gw.qsgw_head.write_qsgw_dipole`` stamps ``basis = "qp"`` (U^H v U
    between the SC map's QP states); ``psp.get_dipole_mtxels`` stamps the
    basis of the WFN it read.  A file with no stamp reads ``"dft"``.
    Contracting one basis's velocities with the other's exciton amplitudes
    keeps every shape and is silently wrong absorption.
    """
    got = attrs.get(DIPOLE_BASIS_ATTR, "dft")
    got = got.decode() if isinstance(got, bytes) else str(got)
    want = wfn_psi_basis(wfn_path)
    if got != want:
        raise ValueError(
            f"GATE dipole_basis: {dipole_path!s} holds velocities between "
            f"{got!r}-basis states, but the wavefunctions ({wfn_path!s}) are "
            f"{want!r}-basis states.  Build the dipole from the WFN the BSE "
            f"runs on (python -m psp.get_dipole_mtxels -i <deck>); "
            f"dipole_qsgw.h5 pairs only with a QP WFN.")


def require_dipole_format(attrs, *, dipole_path) -> None:
    """Refuse a ``dipole.h5`` written before :data:`DIPOLE_FORMAT_VERSION`.

    ``attrs`` are the file's root attributes (an h5py ``AttributeManager`` or
    a dict).  A version-1 file (no stamp) predates the one-operator velocity
    and cannot say which assembly built it.
    """
    raw = attrs.get(DIPOLE_FORMAT_ATTR, 1)
    got = int(np.asarray(raw).reshape(-1)[0])
    if got != DIPOLE_FORMAT_VERSION:
        raise ValueError(
            f"GATE dipole_format_version: {dipole_path!s} has dipole format "
            f"version {got}; want {DIPOLE_FORMAT_VERSION}; why: files before "
            f"version {DIPOLE_FORMAT_VERSION} may hold a velocity other than "
            f"i[H, r] and carry no stamp that says which; fix: regenerate "
            f"dipole.h5 (python -m psp.get_dipole_mtxels -i <deck>).")


def require_dipole_bands(attrs, nb_stored: int, band_lo: int, band_hi: int, *,
                         dipole_path, where: str) -> None:
    """Refuse a dipole whose stored bands do not cover ``[band_lo, band_hi)``.

    ``attrs`` are the file's root attributes (``{}`` when the caller holds
    only the array), ``nb_stored`` the stored band axis, and the range is in
    absolute 0-based band indices.  The stored range is ``[band_min,
    band_max)`` (:data:`DIPOLE_BAND_MIN_ATTR`), ``[0, nb_stored)`` when
    unstamped.  Readers index the stored axis as absolute bands, so a file
    that does not start at band 0, or whose stamp disagrees with its axis,
    refuses too.  A short file used to slice to a shorter window silently.
    """
    lo = int(np.asarray(attrs.get(DIPOLE_BAND_MIN_ATTR, 0)).reshape(-1)[0])
    hi = int(np.asarray(attrs.get(DIPOLE_BAND_MAX_ATTR, lo + int(nb_stored))).reshape(-1)[0])
    if lo != 0 or hi - lo != int(nb_stored) or not 0 <= band_lo < band_hi <= hi:
        raise ValueError(
            f"GATE dipole_band_window: {dipole_path!s} stores bands "
            f"[{lo},{hi}) ({int(nb_stored)} on its axis); want [0, n) covering "
            f"the {where} [{int(band_lo)},{int(band_hi)}); why: the band axis "
            f"is read as absolute bands from 0, and a file short of the window "
            f"slices to fewer bands than the window holds; fix: regenerate it "
            f"with python -m psp.get_dipole_mtxels and nval/ncond/number_bands "
            f"reaching band {int(band_hi)}.")


def require_dipole_k_grid(attrs, nk: int, *, dipole_path) -> None:
    """Refuse a dipole whose full-BZ k list is not the BSE bundle's.

    ``dipole_cart`` is contracted against the transition index ``(c, v, k)``,
    so the file's k list must be the bundle's, in the bundle's order.  A
    coarse file sliced onto a ``bse_k_grid`` bundle used to crash after the
    eigensolve; a ``bse_k_grid`` bundle carries its own dipoles
    (``bse.bse_densify.densified_dipoles``) and never slices a file.
    """
    got = int(attrs.get("nk", -1))
    if got != int(nk):
        raise ValueError(
            f"GATE dipole_k_grid: {dipole_path!s} holds nk={got} k-points; "
            f"want the BSE bundle's nk={int(nk)}; why: the dipole is "
            f"contracted against the transition index (c, v, k), so the two "
            f"k lists must be one list in one order; fix: the dipole.h5 of "
            f"the deck's WFN (python -m psp.get_dipole_mtxels -i <deck>).")


def require_dipole_wfn_fingerprint(attrs, *, dipole_path, wfn_path, want) -> None:
    """Refuse a velocity stamped for another WFN than the one it is contracted with.

    ``want`` is ``common.parallel_transport.wfn_fingerprint`` of ``wfn_path``
    (the caller holds the loader).  Shapes cannot tell: another NSCF, cutoff
    or structure on the same k grid has the same ``(nk, nb)``.  A file with
    no ``prov_wfn_sha256`` stamp refuses too.
    """
    got = attrs.get("prov_wfn_sha256")
    got = got.decode() if isinstance(got, bytes) else got
    if got != want:
        raise ValueError(
            f"GATE velocity_wfn_fingerprint: {dipole_path!s} is stamped for WFN "
            f"{got!r}, not {wfn_path!s} ({want!r}); why: the velocity is "
            f"contracted against this WFN's states, and another WFN's file with "
            f"the same k list passes every shape check; fix: the dipole.h5 built "
            f"from this WFN, or the SC run's dipole_qsgw.h5 beside its WFN_qp.h5")


def velocity_stamp(path, *, wfn, wfn_path) -> tuple[str, str, np.ndarray]:
    """``(velocity label, basis, band_energies)`` of a velocity of ``wfn``'s states.

    Refuses another basis (:func:`require_dipole_basis`) or another WFN's
    ``prov_wfn_sha256``.  A ``psp.get_dipole_mtxels`` file has no
    ``velocity`` attribute: it is the DFT ``p + i[V_NL, r]``.
    """
    import h5py
    from common.parallel_transport import wfn_fingerprint

    def text(value):
        return value.decode() if isinstance(value, bytes) else value

    with h5py.File(str(path), "r") as h5:
        attrs = dict(h5.attrs)
        energies = np.asarray(h5[BAND_ENERGIES_DATASET][()], dtype=np.float64)
    require_dipole_format(attrs, dipole_path=path)
    require_dipole_basis(attrs, dipole_path=path, wfn_path=wfn_path)
    require_dipole_wfn_fingerprint(attrs, dipole_path=path, wfn_path=wfn_path,
                                   want=wfn_fingerprint(wfn))
    label = text(attrs.get("velocity", "p + i[V_NL, r] (DFT velocity)"))
    return str(label), str(text(attrs.get(DIPOLE_BASIS_ATTR, "dft"))), energies


def write_dipole(path, velocity_kmajor, band_energies, *, mesh, attrs,
                 finite_q=None) -> None:
    """Write ``dipole.h5`` through SlabIO.  COLLECTIVE over ``mesh``.

    ``velocity_kmajor`` is ``(nk, 3, nb_pad, nb_pad)``: the sweep's sharded
    output (band axes on ``('x','y')``) or a replicated host array.  The
    logical ``nb`` comes from ``band_energies``, ``(nk, nb)``; pad rows past
    it are dropped by SlabIO.  ``attrs`` are the root attributes.
    ``finite_q`` is ``None`` or ``{"datasets": {name: (array, attrs)},
    "attrs": {...}}``, all replicated host values.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from .slab_io import SlabIO

    energies = np.asarray(band_energies, dtype=np.float64)
    nk, nb = (int(n) for n in energies.shape)
    if (len(velocity_kmajor.shape) != 4
            or tuple(velocity_kmajor.shape[:2]) != (nk, 3)
            or min(velocity_kmajor.shape[-2:]) < nb):
        raise ValueError(
            f"velocity must be (nk={nk}, 3, >= {nb}, >= {nb}); got "
            f"{tuple(velocity_kmajor.shape)}")
    if isinstance(velocity_kmajor, jax.Array):
        velocity = jax.jit(
            lambda v: jnp.moveaxis(v, 1, 0),
            out_shardings=NamedSharding(mesh, P(None, None, "x", "y")),
        )(velocity_kmajor)
    else:
        velocity = np.ascontiguousarray(np.moveaxis(
            np.asarray(velocity_kmajor, dtype=np.complex128), 1, 0))
    with SlabIO(str(path), mode="w", mesh=mesh) as io:
        io.create_dataset(DIPOLE_DATASET, shape=(3, nk, nb, nb),
                          dtype=np.complex128)
        io.write_slab(DIPOLE_DATASET, velocity)
        io.write_attr(BAND_ENERGIES_DATASET, energies)
        if finite_q is not None:
            for name, (value, ds_attrs) in finite_q["datasets"].items():
                io.write_attr(f"{FINITE_Q_GROUP}/{name}", value)
                if ds_attrs:
                    io.stamp_dataset_attrs(f"{FINITE_Q_GROUP}/{name}",
                                           ds_attrs)
            io.stamp_dataset_attrs(FINITE_Q_GROUP, dict(finite_q["attrs"]))
        io.stamp_dataset_attrs(
            "/", {**dict(attrs), DIPOLE_FORMAT_ATTR: DIPOLE_FORMAT_VERSION,
                  DIPOLE_BAND_MIN_ATTR: 0, DIPOLE_BAND_MAX_ATTR: nb})


def energy_extent(h5) -> tuple[int, ...] | None:
    """``(nk, nb)`` of the ΔE a file describes, from whichever table it has."""
    if BAND_ENERGIES_DATASET in h5:
        return tuple(int(n) for n in h5[BAND_ENERGIES_DATASET].shape)
    if LEGACY_DELTA_E_DATASET in h5:
        shape = tuple(int(n) for n in h5[LEGACY_DELTA_E_DATASET].shape)
        return shape[:1] + shape[-1:] if len(shape) == 3 else shape
    return None


def _whole_float64(h5, name) -> np.ndarray | None:
    """Host read of one small table on this process; ``None`` if absent.

    ``band_energies`` is ``(nk, nb)``.  A legacy ``deltaE`` is
    ``(nk, nb, nb)``, the same serial read ``load_dipole_h5`` always did.
    """
    if name not in h5:
        return None
    return np.asarray(h5[name][:], dtype=np.float64)


def _band_energies(h5) -> np.ndarray | None:
    return _whole_float64(h5, BAND_ENERGIES_DATASET)


def read_band_energies(path) -> np.ndarray:
    """``(nk, nb)`` float64 ``band_energies`` of a current-format ``dipole.h5`` (Ry).

    The full-BZ eigenvalues the producer wrote beside ``dipole_cart``; a
    file without the table (written before 2026-09-25) or of an older format
    refuses.
    """
    import h5py
    with h5py.File(str(path), "r") as h5:
        require_dipole_format(h5.attrs, dipole_path=path)
        energies = _band_energies(h5)
    if energies is None:
        raise KeyError(f"{path!s} has no {BAND_ENERGIES_DATASET!r} table; "
                       "regenerate it with python -m psp.get_dipole_mtxels")
    return energies


def delta_e(h5) -> np.ndarray:
    """``(nk, nb, nb)`` float64 ``E_m(k) − E_n(k)`` of an open ``dipole.h5``."""
    energies = _band_energies(h5)
    if energies is not None:
        return energies[:, :, None] - energies[:, None, :]
    if LEGACY_DELTA_E_DATASET in h5:
        return _whole_float64(h5, LEGACY_DELTA_E_DATASET)
    raise KeyError(
        f"{getattr(h5, 'filename', 'dipole.h5')} has neither "
        f"{BAND_ENERGIES_DATASET!r} nor {LEGACY_DELTA_E_DATASET!r}; it is "
        "not a psp.get_dipole_mtxels artifact")


def delta_e_cv(h5, *, nv: int, nc: int) -> np.ndarray | None:
    """``(nk, nc, nv)`` ``E_c − E_v`` for ``c = nv..nv+nc``, ``v < nv``.

    ``None`` for a legacy file, whose stored table the caller reads
    through SlabIO instead.
    """
    energies = _band_energies(h5)
    if energies is None:
        return None
    return (energies[:, nv:nv + nc, None] - energies[:, None, :nv])
