"""``Wavefunctions`` layout tag: legacy carrier untouched, face carrier
correct, accessors refuse by name.  Emulated CPU mesh (unit-scope; the
FOUR-GPU RULE exempts unit/CPU cells — QUALITY_PATTERNS.md).

Guide: reports/gwjax_low_mem_bands_audit_2026-08-22/report.md, census rows
1/10/11 and §5/§7.

Five things this file holds:

1. LEGACY BIT-IDENTICAL.  ``build_wavefunctions`` on this tree produces the
   exact same four arrays, byte for byte, as the pristine
   ``origin/main`` version of the same function (loaded straight out of
   ``git show``, not paraphrased) — the guide's "hash the four arrays
   against main" verification, without needing a second checkout.
2. TAG DISPATCH.  ``layout`` defaults to "legacy"; an explicit "face"
   bundle carries ``None`` on all four legacy fields and populated
   psi_nmu/psi_mun; an unrecognised tag refuses at construction.
3. ACCESSOR REFUSAL.  ``.xn()``/``.xr()``/``.yr()``/``.yn()`` work under
   layout="legacy" (as always) and raise ``ValueError`` naming the
   accessor and the layout under layout="face" — never silently
   rebuilding a legacy replica.
4. FACE DERIVATION IS THE SAME ψ.  ``psi_nmu`` (face) is bit-identical to
   ``psi_yr`` (legacy); ``psi_mun`` (face) is bit-identical to
   ``psi_xr.transpose(0, 2, 3, 1)`` (legacy).  Both bundles are built from
   the SAME host input, so this checks the face builder reproduces the
   already-trusted legacy values under a different sharding, not a
   second hand-rolled reference.
5. MEMORY MODEL AGREES WITH THE CARRIER.  ``gflat_memory_model``'s
   layout-resolved ``psi_copies``/``E_base`` terms match closed-form
   2*S/Px+2*S/Py (legacy) and 2*S/(Px*Py) (face), and the face carrier's
   OWN addressable-shard byte count matches the same face closed form.
"""
from __future__ import annotations

import subprocess
import sys
import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw.wavefunction_bundle import (  # noqa: E402
    BandSlices,
    PSI_MUN_SPEC,
    PSI_NMU_SPEC,
    Wavefunctions,
    build_wavefunctions,
    build_wavefunctions_face,
)

_C128 = 16


def _mesh_xy():
    devices = np.asarray(jax.devices("cpu"), dtype=object)
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    elif devices.size >= 2:
        devices = devices[:2].reshape(1, 2)
    else:
        devices = devices[:1].reshape(1, 1)
    return Mesh(devices, ("x", "y"))


def _put(a, mesh, spec):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))


def _host_inputs(rng, nk, nb, ns, nmu):
    """(psi_rmu_Y, psi_rmuT_X) host numpy, matching
    ``build_wavefunctions``'s documented conventions: psi_rmu_Y is
    un-conjugated ψ (nk,nb,ns,nmu); psi_rmuT_X is CONJUGATED ψ*
    (nk,nmu,nb,ns) — the same un-conjugated ψ, conjugated and permuted.
    """
    psi = (rng.standard_normal((nk, nb, ns, nmu))
           + 1j * rng.standard_normal((nk, nb, ns, nmu)))
    psi_rmuT_X = np.conj(psi).transpose(0, 3, 1, 2)   # (nk, nmu, nb, ns)
    return psi, psi_rmuT_X


def _gather(arr):
    return np.asarray(jax.device_get(arr))


def _load_main_wavefunction_bundle():
    """The pristine ``origin/main`` module, imported under a private name
    so it does not collide with this tree's ``gw.wavefunction_bundle`` in
    ``sys.modules``.  Skips (not fails) if the ref is unavailable, e.g. a
    shallow/detached checkout with no ``origin/main``.
    """
    try:
        src = subprocess.run(
            ["git", "show", "origin/main:src/gw/wavefunction_bundle.py"],
            cwd=__file__.rsplit("/tests/", 1)[0], capture_output=True,
            text=True, timeout=30, check=True).stdout
    except Exception as exc:
        pytest.skip(f"could not read origin/main's wavefunction_bundle.py "
                    f"({type(exc).__name__}: {exc})")
    mod = types.ModuleType("_main_wavefunction_bundle_reference")
    mod.__dict__["__name__"] = mod.__name__
    # dataclass field resolution (BandSlices' string-annotated fields)
    # looks the defining module up via ``sys.modules[cls.__module__]`` --
    # register it BEFORE exec so that lookup finds a real module dict
    # rather than None.
    sys.modules[mod.__name__] = mod
    exec(compile(src, "<origin/main:wavefunction_bundle.py>", "exec"),
         mod.__dict__)
    return mod


# ---------------------------------------------------------------------------
# 1. legacy bit-identical to origin/main
# ---------------------------------------------------------------------------

def test_legacy_build_wavefunctions_bit_identical_to_main():
    main_mod = _load_main_wavefunction_bundle()
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260822)
    nk, nb, ns, nmu = 2, 6, 2, 8
    psi_rmu_Y, psi_rmuT_X = _host_inputs(rng, nk, nb, ns, nmu)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)

    y_in = _put(psi_rmu_Y, mesh, P(None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))

    mine = build_wavefunctions(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)
    theirs = main_mod.build_wavefunctions(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)

    for field in ("psi_xn", "psi_xr", "psi_yr", "psi_yn", "enk", "occ"):
        a, b = _gather(getattr(mine, field)), _gather(getattr(theirs, field))
        assert np.array_equal(a, b), (
            f"{field}: this tree's legacy build_wavefunctions diverged "
            f"from origin/main's — low_mem_bands=false must be the exact "
            f"existing path.")
    assert mine.layout == "legacy"
    assert mine.psi_nmu is None and mine.psi_mun is None


# ---------------------------------------------------------------------------
# 2. tag dispatch + construction refusal
# ---------------------------------------------------------------------------

def test_layout_defaults_and_bad_tag_refuses():
    mesh = _mesh_xy()
    rep = _put(np.zeros((1, 1)), mesh, P(None, None))
    slices = BandSlices.from_band_edges(0, 0, 1, 1, 1)
    legacy = Wavefunctions(enk=rep, occ=rep, slices=slices)
    assert legacy.layout == "legacy"
    assert legacy.psi_nmu is None and legacy.psi_mun is None
    assert legacy.psi_xn is None  # default-constructed, none of the four set

    face = Wavefunctions(enk=rep, occ=rep, slices=slices,
                         psi_nmu=rep, psi_mun=rep, layout="face")
    assert face.layout == "face"
    assert face.psi_xn is None and face.psi_xr is None

    with pytest.raises(ValueError):
        Wavefunctions(enk=rep, occ=rep, slices=slices, layout="bogus")


# ---------------------------------------------------------------------------
# 3. accessor refusal
# ---------------------------------------------------------------------------

def test_face_accessors_refuse_legacy_accessors_work():
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260822)
    nk, nb, ns, nmu = 2, 4, 2, 8
    psi_rmu_Y, psi_rmuT_X = _host_inputs(rng, nk, nb, ns, nmu)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)
    y_in = _put(psi_rmu_Y, mesh, P(None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))

    legacy = build_wavefunctions(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)
    face = build_wavefunctions_face(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)

    # legacy accessors still slice exactly as before
    got = _gather(legacy.xn(slices.sigma))
    want = _gather(legacy.psi_xn)[:, :, :, slices.sigma]
    assert np.array_equal(got, want)

    for name in ("xn", "xr", "yr", "yn"):
        with pytest.raises(ValueError, match=f"{name}.*face|face.*{name}"):
            getattr(face, name)(slices.sigma)


# ---------------------------------------------------------------------------
# 4. face derivation reproduces the legacy values
# ---------------------------------------------------------------------------

def test_face_matches_legacy_same_psi():
    mesh = _mesh_xy()
    rng = np.random.default_rng(20260822)
    nk, nb, ns, nmu = 3, 10, 2, 12
    psi_rmu_Y, psi_rmuT_X = _host_inputs(rng, nk, nb, ns, nmu)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    slices = BandSlices.from_band_edges(0, 0, 3, nb, nb)
    y_in = _put(psi_rmu_Y, mesh, P(None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))

    legacy = build_wavefunctions(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)
    face = build_wavefunctions_face(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)

    psi_yr = _gather(legacy.psi_yr)
    psi_xr = _gather(legacy.psi_xr)
    psi_nmu = _gather(face.psi_nmu)
    psi_mun = _gather(face.psi_mun)

    assert np.array_equal(psi_nmu, psi_yr), (
        "face psi_nmu(k,n,s,mu) must equal legacy psi_yr(k,n,s,mu) "
        "exactly -- same un-conjugated psi, different sharding only.")
    assert np.array_equal(psi_mun, psi_xr.transpose(0, 2, 3, 1)), (
        "face psi_mun(k,s,mu,n) must equal legacy "
        "psi_xr.transpose(0,2,3,1) exactly.")

    # shardings are the declared face specs, not whatever propagation
    # happened to produce
    got_nmu_spec = tuple(face.psi_nmu.sharding.spec)
    got_mun_spec = tuple(face.psi_mun.sharding.spec)
    want_nmu = tuple(PSI_NMU_SPEC)
    want_mun = tuple(PSI_MUN_SPEC)
    # PartitionSpec may report trailing Nones trimmed; pad for comparison.
    def _padded(t, n):
        return tuple(t) + (None,) * (n - len(t))
    assert _padded(got_nmu_spec, 4) == _padded(want_nmu, 4)
    assert _padded(got_mun_spec, 4) == _padded(want_mun, 4)
    assert face.layout == "face"


# ---------------------------------------------------------------------------
# 5. memory model agrees with the carrier
# ---------------------------------------------------------------------------

def test_memory_model_prices_resolved_layout():
    from gw.gflat_memory_model import _persistent_bytes

    nk, ns, mu, nb = 4, 2, 512, 64
    p_x, p_y = 2, 2
    s = _C128 * nk * ns * mu * nb  # one global complex128 psi image

    legacy = _persistent_bytes(
        nk=nk, ns=ns, nq=1, nq_disk=1, mu=mu, nb=nb, ngkmax=1, n_rtot=1,
        p_x=p_x, p_y=p_y, low_mem_bands=False)
    face = _persistent_bytes(
        nk=nk, ns=ns, nq=1, nq_disk=1, mu=mu, nb=nb, ngkmax=1, n_rtot=1,
        p_x=p_x, p_y=p_y, low_mem_bands=True)

    assert legacy["psi_copies"] == pytest.approx(2 * s / p_x + 2 * s / p_y)
    assert face["psi_copies"] == pytest.approx(2 * s / (p_x * p_y))
    # the face reduction is the 2*sqrt(P) the audit claims, on a square mesh
    assert legacy["psi_copies"] / face["psi_copies"] == pytest.approx(
        2 * (p_x * p_y) ** 0.5)


def test_face_carrier_addressable_bytes_match_2s_over_p():
    mesh = _mesh_xy()
    px, py = mesh.devices.shape
    p = px * py
    rng = np.random.default_rng(20260822)
    nk, nb, ns, nmu = 2, 8, 2, 8  # nmu divisible by 2 for a clean p_y=2 shard
    psi_rmu_Y, psi_rmuT_X = _host_inputs(rng, nk, nb, ns, nmu)
    enk = np.sort(rng.standard_normal((nk, nb)), axis=1)
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb)
    y_in = _put(psi_rmu_Y, mesh, P(None, None, None, "y"))
    x_in = _put(psi_rmuT_X, mesh, P(None, "x", None, None))
    enk_in = _put(enk, mesh, P(None, None))

    face = build_wavefunctions_face(
        y_in, x_in, enk_full=enk_in, slices=slices, mesh_xy=mesh)

    s = _C128 * nk * ns * nmu * nb
    want_per_rank = 2 * s / p  # both faces, full mesh sharded

    for arr in (face.psi_nmu, face.psi_mun):
        for sh in arr.addressable_shards:
            got = int(np.asarray(sh.data).nbytes)
            # one shard's worth of ONE face: s/p -- the per-DEVICE figure
            # the guide's "~2S/P per rank" claim is about (one rank == one
            # device on the real multi-rank launch this emulates).
            assert got == pytest.approx(s / p), (
                f"face shard bytes {got} != expected s/p={s / p}")
