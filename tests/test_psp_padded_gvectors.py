"""Owner decision D10, psp side: the seven remaining ragged-G consumers.

``tests/test_kin_ion_padded_gvectors.py`` gates the three kernels
``gw.kin_ion_io`` drives.  This file gates what the **psp drivers** added
on top of them:

* the umklapp G lookup in ``psp.get_dipole_mtxels`` — the one place where
  a pad row does not merely alias Γ but can alias a *different, real* bra
  G-vector after the umklapp shift, so "the mask makes it inert" needs
  proving rather than asserting;
* the ψ-side mask on the momentum operator;
* ``vnl_ops.build_vnl_kdata``'s contract that it returns Z already zeroed
  on the pad columns, so a caller that masks nothing still gets the right
  answer;
* the ``dipole.h5`` provenance guard, which must be able to REFUSE.

As in the kin_ion file, every agreement assertion is paired with a
negative control that must be WRONG by a wide margin: a pad row is a
valid FFT-box index, so a masking bug is silent rather than fatal.

On the pad VALUE, after the 2026-08-08 change
--------------------------------------------
``WfnLoader.gvecs`` now pads with the FFT-box pad sentinel, not with
``(0,0,0)``; ``tests/test_kin_ion_padded_gvectors.py`` was re-derived
against that.  The fixtures HERE deliberately keep the legacy zero pad,
because the hazard this file exists to demonstrate is not "the pad
aliases Γ" — it is that **after an umklapp shift a pad row resolves to a
different, real bra G**.  That mechanism is a property of the shift, not
of the pad value: ``sentinel + G0`` lands on some other box cell just as
``0 + G0`` does, and a bra sphere containing it makes the lookup return a
spurious cross term either way.  Zero is simply the sharpest witness,
because the extra assertion below ("the last ``(0,0,0)`` row of a padded
bra list must win the dict") pins the collision to a row the reader can
name.  Nothing in this file reads a loader-produced table.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

RTOL_D10 = 1e-12


def _dev(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _scale(a) -> float:
    return float(np.max(np.abs(np.asarray(a))))


# ---------------------------------------------------------------------------
# Fixture: one k-pair with a real umklapp shift and a genuine pad
# ---------------------------------------------------------------------------

def _sphere(seed, n, box=(6, 6, 8)):
    """EXACTLY ``n`` distinct box indices, always including ``(0,0,0)``.

    The Γ row is forced in at a non-zero position so that a dropped mask
    collides with a physical component rather than an empty slot — the
    exact failure mode the pad contract has to survive.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = box
    flat = rng.choice(nx * ny * nz, size=min(n + 4, nx * ny * nz),
                      replace=False)
    G = np.stack(np.unravel_index(flat, (nx, ny, nz)), axis=-1).astype(np.int32)
    G[2] = 0                                        # force a Γ row
    seen, keep = set(), []
    for row in G:
        t = tuple(int(v) for v in row)
        keep.append(t not in seen)
        seen.add(t)
    G = G[np.asarray(keep, dtype=bool)]
    assert G.shape[0] >= n, "fixture could not build a large enough sphere"
    return G[:n]


def _pad(G, ngkmax):
    n = int(G.shape[0])
    assert n < ngkmax, "fixture must actually pad"
    Gp = np.concatenate([G, np.zeros((ngkmax - n, 3), dtype=np.int32)], axis=0)
    m = np.concatenate([np.ones(n), np.zeros(ngkmax - n)]).astype(np.float64)
    return Gp, m


# ---------------------------------------------------------------------------
# 1. The umklapp lookup
# ---------------------------------------------------------------------------

def test_g_lookup_padded_matches_ragged_on_the_physical_prefix():
    from psp.get_dipole_mtxels import _build_g_lookup

    G_k = _sphere(11, 29)
    G_can = _sphere(12, 33)
    ngk_k, ngk_can = int(G_k.shape[0]), int(G_can.shape[0])
    ngkmax = 40
    Gp_k, _ = _pad(G_k, ngkmax)
    Gp_can, _ = _pad(G_can, ngkmax)
    G_wrap = np.asarray([1, 0, -1], dtype=np.int32)

    ref_map, ref_mask = _build_g_lookup(G_can, G_k, G_wrap,
                                         ngk_kmq=ngk_can, ngk_k=ngk_k)
    pad_map, pad_mask = _build_g_lookup(Gp_can, Gp_k, G_wrap,
                                         ngk_kmq=ngk_can, ngk_k=ngk_k)

    assert pad_map.shape == (ngkmax,) and pad_mask.shape == (ngkmax,)
    assert np.array_equal(pad_map[:ngk_k], ref_map[:ngk_k])
    assert np.array_equal(pad_mask[:ngk_k], ref_mask[:ngk_k])
    # The pad rows must take NO part.
    assert not pad_mask[ngk_k:].any()
    # ...and the lookup must have found something, or the test is vacuous.
    assert ref_mask.sum() > 0


def test_g_lookup_pad_rows_would_alias_a_real_bra_G_without_the_extent():
    """The failure this guard exists to stop — shown happening.

    A pad row is ``(0,0,0)``; after the umklapp shift it looks up
    ``G_wrap``, which is generally a REAL member of the bra sphere.  So
    unlike everywhere else in the D10 work, the damage here is not a
    doubled Γ but a spurious cross term between two different G.
    """
    from psp.get_dipole_mtxels import _build_g_lookup

    G_k = _sphere(21, 25)
    G_can = _sphere(22, 30)
    ngk_k, ngk_can = int(G_k.shape[0]), int(G_can.shape[0])
    ngkmax = 36
    Gp_k, _ = _pad(G_k, ngkmax)
    Gp_can, _ = _pad(G_can, ngkmax)
    # Pick a shift that IS in the bra sphere, so a pad row resolves.
    G_wrap = np.asarray(G_can[5], dtype=np.int32)

    good_map, good_mask = _build_g_lookup(Gp_can, Gp_k, G_wrap,
                                           ngk_kmq=ngk_can, ngk_k=ngk_k)
    # NEGATIVE CONTROL: pretend the whole padded width is physical.
    bad_map, bad_mask = _build_g_lookup(Gp_can, Gp_k, G_wrap,
                                         ngk_kmq=ngkmax, ngk_k=ngkmax)

    assert not good_mask[ngk_k:].any(), "pad rows must be gated off"
    assert bad_mask[ngk_k:].any(), (
        "control did not fire: this fixture no longer demonstrates the "
        "aliasing it exists to demonstrate")
    # And the bad run additionally rebinds Γ to the LAST pad index.
    gamma_rows = np.flatnonzero(~Gp_can[:ngk_can].any(axis=1))
    assert gamma_rows.size == 1, "fixture must have exactly one physical Γ"
    # The Γ key in the over-wide dictionary points past the physical sphere.
    over = {tuple(int(x) for x in g): i for i, g in enumerate(Gp_can)}
    assert over[(0, 0, 0)] >= ngk_can, (
        "the last (0,0,0) row of a padded bra list must win the dict "
        "binding — that is why the physical extent is passed explicitly")


# ---------------------------------------------------------------------------
# 2. The ψ-side mask on the momentum operator
# ---------------------------------------------------------------------------

def test_momentum_padded_matches_ragged_and_mask_is_load_bearing():
    from psp.get_dipole_mtxels import compute_p_operator_k

    G = _sphere(31, 37)
    Gp, mask = _pad(G, 48)
    rng = np.random.default_rng(5)
    box = (6, 6, 8)
    psi = jnp.asarray(rng.standard_normal((4, 2) + box)
                      + 1j * rng.standard_normal((4, 2) + box))
    # A k AWAY from Γ: the momentum operator multiplies by (k+G), which is
    # exactly 0 on a pad row at Γ, so the control could not fire there.
    kvec = np.asarray([0.2, -0.35, 0.1])
    bvec = np.eye(3) * 1.3
    blat = 2.0
    bdot = (bvec * blat) @ (bvec * blat).T

    ragged = compute_p_operator_k(psi, G, kvec, bdot, bvec, blat)
    padded = compute_p_operator_k(psi, Gp, kvec, bdot, bvec, blat, g_mask=mask)
    unmasked = compute_p_operator_k(psi, Gp, kvec, bdot, bvec, blat)

    s = _scale(ragged)
    assert s > 0
    assert _dev(padded, ragged) <= RTOL_D10 * s
    assert _dev(unmasked, ragged) > 1e-6 * s


def test_momentum_control_is_inert_at_gamma_by_construction():
    """Document the one place the control legitimately cannot fire.

    At k=Γ a pad row contributes ``2·(0+0)·|ψ(Γ)|² = 0``.  A gate that
    asserted "the control always fires" would therefore be asserting
    something false, and a gate that quietly passed here would be hiding
    it.  Pin the behaviour instead.
    """
    from psp.get_dipole_mtxels import compute_p_operator_k

    G = _sphere(41, 30)
    Gp, mask = _pad(G, 40)
    rng = np.random.default_rng(6)
    box = (6, 6, 8)
    psi = jnp.asarray(rng.standard_normal((3, 1) + box)
                      + 1j * rng.standard_normal((3, 1) + box))
    bvec, blat = np.eye(3) * 1.1, 1.0
    bdot = (bvec * blat) @ (bvec * blat).T
    kG = np.zeros(3)

    ragged = compute_p_operator_k(psi, G, kG, bdot, bvec, blat)
    unmasked = compute_p_operator_k(psi, Gp, kG, bdot, bvec, blat)
    # Not "small": the pad terms are 2·0·ψ = exact zeros, so only XLA's
    # reduction blocking can separate the two, at the 1e-16 level.
    assert _dev(unmasked, ragged) <= 1e-13 * _scale(ragged)


# ---------------------------------------------------------------------------
# 3. build_vnl_kdata returns Z already inert on the pad
# ---------------------------------------------------------------------------

def test_build_vnl_kdata_zeroes_Z_on_the_pad(monkeypatch):
    import psp.dft_operators as dop
    import psp.vnl_ops as vnl_ops

    ngkmax, ngk = 12, 8
    gv = np.zeros((1, ngkmax, 3), dtype=np.int32)
    gv[0, :ngk] = _sphere(51, ngk, box=(4, 4, 4))[:ngk]

    class _Loader:
        def gvecs(self, *, k="full_bz"):
            return gv

        def ngk_valid(self, *, k="full_bz"):
            return np.asarray([ngk], dtype=np.int32)

    monkeypatch.setattr(dop, "_as_loader", lambda w: _Loader())

    total_R = 3
    setup = vnl_ops.VNLSetup(
        channels=[], dq=0.01, n_q=64, q_max=1.0,
        G_table=jnp.ones((1, 64), dtype=jnp.float64),
        Gp_table=jnp.zeros((1, 64), dtype=jnp.float64),
        prefactor=1.0, B=np.eye(3), cell_volume=1.0,
        total_R=total_R, nspinor=1,
        E_super=jnp.zeros((1, 1, total_R, total_R), dtype=jnp.complex128),
        l_max=0,
        row_beta_idx=jnp.zeros(total_R, dtype=jnp.int32),
        row_l=jnp.zeros(total_R, dtype=jnp.int32),
        row_m=jnp.zeros(total_R, dtype=jnp.int32),
        row_tau=jnp.zeros((total_R, 3), dtype=jnp.float64),
    )

    class _Sym:
        unfolded_kpts = np.asarray([[0.25, -0.125, 0.0]])

    kd = vnl_ops.build_vnl_kdata(0, setup, object(), _Sym(), None)
    Z = np.asarray(kd.Z)
    assert Z.shape == (total_R, ngkmax)
    assert np.all(Z[:, ngk:] == 0.0), "pad columns of Z must be exactly zero"
    assert np.max(np.abs(Z[:, :ngk])) > 0.0, "fixture produced a null Z"
    assert kd.g_mask is not None and kd.g_mask.sum() == ngk

    # NEGATIVE CONTROL: the core, which the SymMaps wrapper masks, leaves
    # the pad columns FINITE — that is the thing being neutralised.
    raw = vnl_ops.build_vnl_kdata_from_kvec(
        _Sym.unfolded_kpts[0], gv[0], setup)
    assert np.max(np.abs(np.asarray(raw.Z)[:, ngk:])) > 0.0


# ---------------------------------------------------------------------------
# 4. The dipole.h5 provenance guard must be able to REFUSE
# ---------------------------------------------------------------------------

class _FakeWfn:
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
                                 nb_written=4, bispinor=False,
                                 skip_vnl=False, vnl_mode="analytic", **kw)


def test_provenance_guard_accepts_its_own_stamp(tmp_path):
    from psp.get_dipole_mtxels import check_dipole_provenance

    wfn = _FakeWfn()
    p = tmp_path / "dipole.h5"
    _write_stamped(p, wfn, nval=2, ncond=3, nband=8)
    assert check_dipole_provenance(p, wfn=wfn, nval=2, ncond=3, nband=8,
                                    print_fn=lambda *a: None) is True


def test_provenance_guard_refuses_a_different_wfn(tmp_path, monkeypatch):
    """The case nothing on disk could detect before: same shapes, different
    DFT solution."""
    from common import sanity
    from psp.get_dipole_mtxels import check_dipole_provenance

    monkeypatch.setattr(sanity, "sanity_strict", lambda: False)
    old = _FakeWfn(seed=0)
    new = _FakeWfn(seed=1)                      # same shapes, new eigenvalues
    assert old.energies.shape == new.energies.shape
    p = tmp_path / "dipole.h5"
    _write_stamped(p, old, nval=2, ncond=3, nband=8)

    lines = []
    ok = check_dipole_provenance(p, wfn=new, nval=2, ncond=3, nband=8,
                                  print_fn=lines.append)
    assert ok is False
    assert any("DIFFERENT DFT solution" in ln for ln in lines)


def test_provenance_guard_refuses_a_changed_band_window(tmp_path, monkeypatch):
    from common import sanity
    from psp.get_dipole_mtxels import check_dipole_provenance

    monkeypatch.setattr(sanity, "sanity_strict", lambda: False)
    wfn = _FakeWfn()
    p = tmp_path / "dipole.h5"
    _write_stamped(p, wfn, nval=2, ncond=3, nband=8)
    lines = []
    assert check_dipole_provenance(p, wfn=wfn, nval=2, ncond=5, nband=8,
                                    print_fn=lines.append) is False
    assert any("prov_ncond" in ln for ln in lines)


def test_provenance_guard_reports_an_unstamped_file(tmp_path):
    import h5py
    from psp.get_dipole_mtxels import check_dipole_provenance

    p = tmp_path / "old_dipole.h5"
    with h5py.File(str(p), "w") as h5:
        h5.create_dataset("dipole_cart", data=np.zeros((3, 1, 1, 1)))
        h5.attrs["nbands"] = 8            # the pre-guard attrs, on their own
    lines = []
    assert check_dipole_provenance(p, wfn=_FakeWfn(), nval=2, ncond=3,
                                    nband=8, print_fn=lines.append) is False
    assert any("no provenance stamp" in ln for ln in lines)


def test_provenance_guard_reports_legacy_fingerprint_as_unverifiable(tmp_path):
    """A pre-content-scheme hash must not masquerade as a DFT mismatch."""
    import h5py
    from psp.get_dipole_mtxels import check_dipole_provenance

    p = tmp_path / "legacy_dipole.h5"
    with h5py.File(str(p), "w") as h5:
        h5.attrs["prov_wfn_sha256"] = "0" * 64
        h5.attrs["prov_nval"] = 2
        h5.attrs["prov_ncond"] = 3
        h5.attrs["prov_nband"] = 8
    lines = []
    assert check_dipole_provenance(
        p, wfn=_FakeWfn(), nval=2, ncond=3, nband=8,
        print_fn=lines.append) is False
    assert any("predates" in line and "cannot be compared" in line
               for line in lines)
    assert not any("DIFFERENT DFT solution" in line for line in lines)


def test_wfn_fingerprint_moves_with_the_eigenvalues():
    from psp.get_dipole_mtxels import wfn_fingerprint

    a = _FakeWfn(seed=0)
    b = _FakeWfn(seed=0)
    assert wfn_fingerprint(a) == wfn_fingerprint(b)
    b.energies = b.energies.copy()
    b.energies[0, 0, 0] += 1e-9
    assert wfn_fingerprint(a) != wfn_fingerprint(b), (
        "a 1e-9 Ry shift in one eigenvalue must change the fingerprint, "
        "or the guard cannot see a regenerated WFN")


# ---------------------------------------------------------------------------
# 5. The psp drivers must reach the distribution service, not JAX directly
# ---------------------------------------------------------------------------

def _driver_sources():
    """The three psp CLI sources, read from wherever ``psp`` actually is.

    Resolved through a MODULE, not the package: ``psp`` ships no
    ``__init__.py``, so it is a namespace package and ``psp.__file__`` is
    ``None`` — which is how the first version of this check died with a
    TypeError instead of scanning anything.
    """
    import pathlib
    import psp.dft_operators as _anchor
    root = pathlib.Path(_anchor.__file__).resolve().parent
    return {name: (root / name).read_text()
            for name in ("get_dipole_mtxels.py", "get_DFT_mtxels.py",
                         "run_nscf.py")}


def test_psp_drivers_do_not_hand_roll_the_distribution_layer():
    """No ``multihost_utils`` / ``Mesh`` / ``NamedSharding`` in a driver.

    The point of the exercise: a core driver should read as physics.  Three
    psp CLIs built their own mesh, spelled their own ``PartitionSpec`` and
    called ``process_allgather`` by hand at four sites; all of that now goes
    through ``common.collectives``.

    NEGATIVE CONTROL below — the same scan run against a synthetic source
    that DOES contain the banned spellings must flag it, otherwise this is a
    structural check that scans nothing.
    """
    banned = ("multihost_utils", "NamedSharding", "PartitionSpec",
              "jax.sharding")
    offenders = {
        name: [b for b in banned if b in src]
        for name, src in _driver_sources().items()
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"psp drivers still hand-roll plumbing: {offenders}"


def test_the_driver_scan_can_fail():
    """The negative control for the check above."""
    banned = ("multihost_utils", "NamedSharding", "PartitionSpec",
              "jax.sharding")
    fake = "from jax.experimental import multihost_utils as _mhu\n"
    assert [b for b in banned if b in fake] == ["multihost_utils"], (
        "the scan does not detect the very spelling it bans")


def test_psp_drivers_import_the_service():
    """...and they actually reach for it, rather than simply going without."""
    srcs = _driver_sources()
    assert "from common.collectives import" in srcs["get_dipole_mtxels.py"]
    assert "from common.collectives import" in srcs["get_DFT_mtxels.py"]
    assert "from common.collectives import" in srcs["run_nscf.py"]


# ---------------------------------------------------------------------------
# The dipole driver's RESIDENCY contract (measured jobs 7883152 / 7883153)
# ---------------------------------------------------------------------------
#
# `psp.get_dipole_mtxels` used to build ψ for ALL k in the FFT box
# (`read_Gvecs_to_devices`) and then make a SECOND k-major copy of the same
# thing (`shard_over_k`).  Measured at P=1 on MoS2 4x4 / 128 bands, that was
# 2 x 2.812 GiB of the 7.34 GiB peak; streaming one k at a time through
# `load_kpoint_fftbox_local` — the contract `gw.kin_ion_io` already runs on —
# took the driver to 1.57 GiB with dipole_cart BIT-IDENTICAL.
#
# The two spellings below are what separates those states, so the check is
# structural: nothing else in the file changes if someone reintroduces the
# resident load, and the failure would be a silent 4.5x memory regression
# that the value gates cannot see.
#
# WHAT IS RESIDENT NOW, and why it is not the thing this pins.  The default
# path is ``common.mtxel_sweep``, which holds ψ on the G-SPHERE for all k,
# band-sharded (nk·nb·ns·ngkmax·16 B — 1.2 GB globally at b600, ≈19 MB/rank
# at P=64).  That is the OTHER array: the box is nx·ny·nz per band where the
# sphere is ngkmax, and the sweep forms no box at all.  The streaming
# spellings below stay because the per-k route survives for
# ``--vnl-mode=numeric`` and the ``--debug`` table.

_RESIDENT_SPELLINGS = ("read_Gvecs_to_devices", "shard_over_k")
_STREAMING_SPELLINGS = ("load_kpoint_fftbox_local", "gather_k_blocks")


def test_dipole_driver_streams_k_rather_than_materialising_all_of_them():
    src = _driver_sources()["get_dipole_mtxels.py"]
    resident = [s for s in _RESIDENT_SPELLINGS if s in src]
    missing = [s for s in _STREAMING_SPELLINGS if s not in src]
    assert not resident, (
        f"get_dipole_mtxels is holding every k's ψ again ({resident}); that "
        f"was 2 x 2.812 GiB of a 7.34 GiB peak on MoS2 4x4/128 bands")
    assert not missing, (
        f"get_dipole_mtxels no longer streams k ({missing} absent)")


def test_the_residency_scan_can_fail():
    """Negative control: the scan must flag a source that DOES hold all k."""
    fake = ("global_psi_G, nb = read_Gvecs_to_devices(...)\n"
            "wfn_k_sharded = shard_over_k(global_psi_G, mesh_xy)\n")
    assert [s for s in _RESIDENT_SPELLINGS if s in fake] == list(
        _RESIDENT_SPELLINGS), "the scan does not detect the pattern it bans"
    assert [s for s in _STREAMING_SPELLINGS if s not in fake] == list(
        _STREAMING_SPELLINGS), "the scan would pass a source that streams nothing"


# ===========================================================================
# THE NON-DEFAULT-WINDOW RED TWIN
# ===========================================================================
# Two failures wore one message.  ``gw.head_correction._dipole_window_from_
# params`` defaulted an absent window to 5/5/max(nbands, nelec+5), and
# ``HeadResolver`` used to hand it ``config.head`` alone — six keys with no
# band window in them — so EVERY deck whose window is not 5/5 was accused of
# being stale.  Measured on the MoS2 production deck (JID 57269074): a
# dipole.h5 built from the very same WFN and deck reported
# ``prov_nval: file=26 run=5 ... prov_nband: file=600 run=610``.  Under
# ``LORRAX_SANITY=strict`` that false warning is an unconditional refusal.
#
# The three cells below are the discriminator the register asked for: on ONE
# non-default window, a correct file must PASS and a genuinely stale WFN must
# still REFUSE.  Without both arms the guard's silence and its noise are
# indistinguishable.

#: Deliberately NOT 5/5 — that pair is the value the pre-fix helper invented,
#: so a fixture written at 5/5 would agree with the defect and certify it.
_NONDEFAULT_WINDOW = dict(nval=2, ncond=3, nband=8)


def _head_params(**window):
    """``HeadResolver._params`` as the resolver builds it, window included."""
    return {"wcoul0_source": "s_tensor", "wcoul0_eta": 0.0,
            "vhead": None, "whead_0freq": None, "whead_imfreq": None,
            "head_minibz_average": False, **window}


def test_a_non_default_window_dipole_passes_its_own_provenance_guard(tmp_path):
    """GREEN arm: the file the deck itself produced is not accused."""
    from gw.head_correction import _check_dipole_provenance

    wfn = _FakeWfn()
    p = tmp_path / "dipole.h5"
    _write_stamped(p, wfn, **_NONDEFAULT_WINDOW)
    lines = []
    _check_dipole_provenance(p, params=_head_params(**_NONDEFAULT_WINDOW),
                             wfn=wfn, print_fn=lines.append)
    assert any("provenance OK" in ln for ln in lines), lines
    assert not any("DIFFERENT DFT solution" in ln for ln in lines), lines


def test_a_stale_wfn_still_refuses_on_that_same_window(tmp_path, monkeypatch):
    """RED arm: the guard has not been muted, only aimed.

    Same non-default window on both sides, a different DFT solution behind
    the file.  The complaint must name the fingerprint and NOT the window —
    the pre-fix message named the window on every deck, which is how a real
    stale-WFN report became unreadable.
    """
    from common import sanity
    from gw.head_correction import _check_dipole_provenance

    monkeypatch.setattr(sanity, "sanity_strict", lambda: False)
    old, new = _FakeWfn(seed=0), _FakeWfn(seed=1)
    p = tmp_path / "dipole.h5"
    _write_stamped(p, old, **_NONDEFAULT_WINDOW)
    lines = []
    _check_dipole_provenance(p, params=_head_params(**_NONDEFAULT_WINDOW),
                             wfn=new, print_fn=lines.append)
    assert any("DIFFERENT DFT solution" in ln for ln in lines), lines
    assert any("prov_wfn_sha256" in ln for ln in lines), lines
    assert not any("prov_nval" in ln or "prov_ncond" in ln or "prov_nband" in ln
                   for ln in lines), (
        "a stale-WFN report must not also accuse the band window", lines)


def test_the_head_checker_refuses_a_params_dict_with_no_window(tmp_path):
    """The defect itself, pinned: no window reaching the checker is a REFUSAL.

    This is the cell that fails on the pre-fix source, where the same call
    silently resolved 5/5/8 and reported a mismatch against a correct file.
    """
    import pytest as _pytest
    from gw.head_correction import _check_dipole_provenance

    wfn = _FakeWfn()
    p = tmp_path / "dipole.h5"
    _write_stamped(p, wfn, **_NONDEFAULT_WINDOW)
    with _pytest.raises(ValueError, match="band window is missing"):
        _check_dipole_provenance(p, params=_head_params(), wfn=wfn,
                                 print_fn=lambda *_a: None)
