"""Inert schema and provenance owner for ``dipole.h5``.

Importing this module performs no runtime, communicator, accelerator, or FFI
initialization.  Producers and read-only consumers share the same contract.
"""

from __future__ import annotations

import h5py
import numpy as np

from common.parallel_transport import (
    WFN_FINGERPRINT_SCHEME,
    wfn_fingerprint,
)


DIPOLE_Q0_OPERATOR_SCHEME = "lorrax.dipole_q0.exact_reduced_origin/v1"

_PROV_ATTRS = (
    "prov_wfn_sha256", "prov_wfn_fingerprint_scheme", "prov_nval",
    "prov_ncond", "prov_nband", "prov_nb_written", "prov_bispinor",
    "prov_skip_vnl", "prov_vnl_mode", "prov_wfn_file",
    "prov_vnl_velocity_sign", "prov_nspinor", "prov_soc",
    "prov_q0_operator_scheme",
)


def stamp_dipole_provenance(
        h5, *, wfn, wfn_path, nval, ncond, nband, nb_written, bispinor,
        skip_vnl, vnl_mode, vnl_velocity_sign=None, nspinor=None,
        soc=None) -> None:
    """Record the WFN, band window, and velocity convention of a store."""
    h5.attrs["prov_wfn_sha256"] = wfn_fingerprint(wfn)
    h5.attrs["prov_wfn_fingerprint_scheme"] = WFN_FINGERPRINT_SCHEME
    h5.attrs["prov_wfn_file"] = str(wfn_path)
    h5.attrs["prov_nval"] = int(nval)
    h5.attrs["prov_ncond"] = int(ncond)
    h5.attrs["prov_nband"] = int(nband)
    h5.attrs["prov_nb_written"] = int(nb_written)
    h5.attrs["prov_bispinor"] = bool(bispinor)
    h5.attrs["prov_skip_vnl"] = bool(skip_vnl)
    h5.attrs["prov_vnl_mode"] = str(vnl_mode)
    h5.attrs["prov_q0_operator_scheme"] = DIPOLE_Q0_OPERATOR_SCHEME
    if vnl_velocity_sign is not None:
        h5.attrs["prov_vnl_velocity_sign"] = float(vnl_velocity_sign)
    if nspinor is not None:
        h5.attrs["prov_nspinor"] = int(nspinor)
    if soc is not None:
        h5.attrs["prov_soc"] = bool(soc)


def resolve_dipole_nb_written(wfn, *, ncond, nband) -> int:
    """Resolve the physical square band extent of the q=0 operator."""
    return min(
        int(wfn.nbands),
        max(int(wfn.nelec) + int(ncond), int(nband)),
    )


def _q0_ncond_coverage(h5, *, wfn, ncond, nband) -> tuple[bool, str]:
    expected = resolve_dipole_nb_written(
        wfn, ncond=int(ncond), nband=int(nband))
    if "finite_q" in h5:
        return False, (
            "finite_q/ is present and its stored conduction axis is sized by "
            "the producer's ncond")

    problems = []
    if "prov_nb_written" not in h5.attrs:
        problems.append("prov_nb_written is absent")
    else:
        got = int(np.asarray(h5.attrs["prov_nb_written"]))
        producer_expected = resolve_dipole_nb_written(
            wfn,
            ncond=int(np.asarray(h5.attrs["prov_ncond"])),
            nband=int(np.asarray(h5.attrs.get("prov_nband", nband))),
        )
        if got != producer_expected:
            problems.append(
                f"prov_nb_written: file={got} producer-resolved="
                f"{producer_expected}")
        if got != expected:
            problems.append(
                f"prov_nb_written: file={got} run-resolved={expected}")

    shapes = {}
    for name, rank in (("dipole_cart", 4), ("deltaE", 3)):
        if name not in h5:
            problems.append(f"{name} is absent")
            continue
        shape = tuple(int(v) for v in h5[name].shape)
        shapes[name] = shape
        if len(shape) != rank or shape[-2:] != (expected, expected):
            problems.append(
                f"{name} shape={shape}, expected square band axes "
                f"({expected},{expected})")
    if (len(shapes.get("dipole_cart", ())) >= 2
            and len(shapes.get("deltaE", ())) >= 1
            and shapes["dipole_cart"][1] != shapes["deltaE"][0]):
        problems.append(
            "dipole_cart and deltaE carry different k extents "
            f"({shapes['dipole_cart'][1]} versus {shapes['deltaE'][0]})")
    return not problems, ("; ".join(problems) if problems
                          else f"identical q→0 extent {expected}")


def check_dipole_provenance(
        path, *, wfn, nval, ncond, nband, bispinor=None, skip_vnl=None,
        vnl_mode=None, vnl_velocity_sign=None, wfn_fingerprint_binding=None,
        print_fn=print) -> bool:
    """Authenticate one dipole store against its WFN and operator request."""
    from common import sanity
    from common.parallel_transport import fingerprint_from_binding

    try:
        with h5py.File(str(path), "r") as h5:
            attrs = {k: h5.attrs[k] for k in _PROV_ATTRS if k in h5.attrs}
            ncond_mismatch = (
                "prov_ncond" not in attrs
                or _prov_ne(attrs["prov_ncond"], int(ncond)))
            q0_ncond_ok, q0_ncond_detail = (False, "prov_ncond is absent")
            if ncond_mismatch and "prov_ncond" in attrs:
                q0_ncond_ok, q0_ncond_detail = _q0_ncond_coverage(
                    h5, wfn=wfn, ncond=ncond, nband=nband)
    except OSError as exc:
        print_fn(f"  [dipole provenance] cannot open {path} "
                 f"({type(exc).__name__}: {exc})")
        return False

    if "prov_wfn_sha256" not in attrs:
        print_fn(f"  [dipole provenance] {path} carries no provenance stamp "
                 "(written before the guard existed). Regenerate with "
                 "`python -m psp.get_dipole_mtxels` to make it checkable.")
        return False

    got_scheme = attrs.get("prov_wfn_fingerprint_scheme")
    if isinstance(got_scheme, bytes):
        got_scheme = got_scheme.decode()
    fingerprint_checkable = got_scheme == WFN_FINGERPRINT_SCHEME
    if got_scheme is None:
        print_fn(
            "  [dipole provenance] the WFN fingerprint predates the "
            f"location-independent {WFN_FINGERPRINT_SCHEME!r} scheme and "
            "cannot be compared across checkouts; regenerate dipole.h5.")
    elif not fingerprint_checkable:
        print_fn(
            "  [dipole provenance] the WFN fingerprint uses unsupported "
            f"scheme {got_scheme!r}, not {WFN_FINGERPRINT_SCHEME!r}; "
            "regenerate dipole.h5.")
    if not fingerprint_checkable:
        return False

    want = {
        "prov_nval": int(nval), "prov_ncond": int(ncond),
        "prov_nband": int(nband),
        "prov_q0_operator_scheme": DIPOLE_Q0_OPERATOR_SCHEME,
        "prov_wfn_sha256": (
            wfn_fingerprint(wfn) if wfn_fingerprint_binding is None
            else fingerprint_from_binding(wfn_fingerprint_binding, wfn)),
    }
    optional = {
        "prov_bispinor": bispinor, "prov_skip_vnl": skip_vnl,
        "prov_vnl_mode": vnl_mode,
        "prov_vnl_velocity_sign": vnl_velocity_sign,
    }
    want.update({key: value for key, value in optional.items()
                 if value is not None})
    bad = [
        (key, attrs.get(key, "<absent>"), expected)
        for key, expected in want.items()
        if (key != "prov_ncond" or not q0_ncond_ok)
        and (key not in attrs or _prov_ne(attrs[key], expected))
    ]
    if "prov_nspinor" in attrs and _prov_ne(
            attrs["prov_nspinor"], int(wfn.nspinor)):
        bad.append(("prov_nspinor", attrs["prov_nspinor"], int(wfn.nspinor)))
    if bad:
        detail = "; ".join(
            f"{key}: file={_prov_show(got)} run={_prov_show(expected)}"
            for key, got, expected in bad)
        if ncond_mismatch and not q0_ncond_ok:
            detail += f"; q→0 coverage refusal: {q0_ncond_detail}"
        sanity.warn(
            f"{path} was generated from a DIFFERENT DFT solution, spin "
            "representation, band window, or velocity/representation "
            f"convention ({detail}). dipole.h5 has the right shape either "
            "way, so a shape-only reader would not notice: the q→0 head "
            "S(ω), and every Σ_SX/Σ_COH correction built from it, would "
            "be assembled from incompatible velocity matrix elements. "
            "Regenerate it with `python -m psp.get_dipole_mtxels -i <deck>`.",
            print_fn=print_fn)
        return False

    if ncond_mismatch:
        print_fn(
            "  [dipole provenance] producer "
            f"ncond={int(np.asarray(attrs['prov_ncond']))} differs from run "
            f"ncond={int(ncond)}, accepted because {q0_ncond_detail}; the "
            "ordinary payload is the same full-square operator.")
    print_fn(
        f"  dipole.h5 provenance OK (WFN {want['prov_wfn_sha256'][:12]}…, "
        f"window nval={int(nval)} ncond={int(ncond)} nband={int(nband)}"
        + (f", bispinor={bool(bispinor)}" if bispinor is not None else "")
        + (f", vnl_velocity_sign={float(vnl_velocity_sign):+.1f}"
           if vnl_velocity_sign is not None else "")
        + ")")
    return True


def _prov_ne(got, expected) -> bool:
    if isinstance(expected, str):
        got = got.decode() if isinstance(got, bytes) else str(got)
        return got != expected
    return int(np.asarray(got)) != int(expected)


def _prov_show(value) -> str:
    if isinstance(value, bytes):
        value = value.decode()
    return (value[:12] + "…"
            if isinstance(value, str) and len(value) > 13 else str(value))


__all__ = [
    "DIPOLE_Q0_OPERATOR_SCHEME",
    "check_dipole_provenance",
    "resolve_dipole_nb_written",
    "stamp_dipole_provenance",
]
