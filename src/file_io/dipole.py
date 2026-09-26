"""``dipole.h5``: the q→0 velocity matrix, its band energies, finite-q SOS.

Layout (``psp.get_dipole_mtxels`` writes it, the GW head and BSE absorption
read it):

* ``dipole_cart`` — ``(3, nk, nb, nb)`` complex128, ⟨mk|v̂_α|nk⟩ in Ry
  units on the full BZ, at the arm the file's ``prov_vnl_velocity_sign``
  names.
* ``band_energies`` — ``(nk, nb)`` float64, the WFN mean-field eigenvalue of
  each band at each full-BZ k (Ry): the star parent's row, which is the
  energy ψ(Sk) carries.
* ``finite_q/`` — optional finite-q SOS block (``rho_cvkq``, ``v_cvkq``,
  ``kminq_idx``, ``iq_list``, and for bispinors ``alpha_cvkq`` and
  ``ward_residual_cvkq``).
* root attributes — ``nbands``, ``nk``, ``skip_vnl``, ``note`` and the
  ``prov_*`` provenance stamp (``psp.get_dipole_mtxels.dipole_provenance``).

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

__all__ = [
    "BAND_ENERGIES_DATASET",
    "DIPOLE_DATASET",
    "FINITE_Q_GROUP",
    "LEGACY_DELTA_E_DATASET",
    "band_energies_on_full_bz",
    "delta_e",
    "delta_e_cv",
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
        io.stamp_dataset_attrs("/", dict(attrs))


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
