"""Sanity-gate tests that need jax and/or h5py.

**READY TO RUN — NOT YET EXECUTED.**  The login-node ``python3`` has
neither jax nor a working h5py (``libhdf5.so.103`` missing), so this file
was written and left runnable rather than run.  Execute it inside the
container, e.g.::

    /scratch2/08271/jackmc/lorrax_setup/alloc_run.sh 1 1 \
        /work2/08271/jackmc/frontera/wt-F/src \
        /work2/08271/jackmc/frontera/wt-F \
        python -u tests/test_sanity_gates_jax.py

or under pytest with the venv on PATH.  It needs **one** process and no
GPU; total runtime is seconds.  The pure-python half of the coverage
(``tests/test_sanity_gates.py``) already passes on the login node — this
file adds only the on-device reduction path and the HDF5 provenance
guards.

Covered here
------------
1. ``common.sanity`` reductions against real ``jax.Array`` inputs,
   including a *sharded* array (the production case — the reduction must
   produce one replicated scalar, not a per-shard value).
2. ``file_io.zeta_loader.ZetaLoader``'s ``zeta_is_done`` refusal — the
   guard that turns a ζ from a job that died mid-write from "silently
   reusable" into an error.
3. ``gw.gw_init._check_zeta_h5_matches_basis`` on a G-flat file — the
   case the old guard missed entirely because it probed ``zeta_q``
   while the production writer creates ``zeta_q_G``.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Four emulated host devices so the sharded-reduction test exercises a real
# multi-device global array (the production case) instead of skipping.
os.environ.setdefault("XLA_FLAGS",
                      "--xla_force_host_platform_device_count=4")

import h5py                                          # noqa: E402
import pytest                                         # noqa: E402
import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from common import sanity                            # noqa: E402


class _Log:
    def __init__(self):
        self.lines = []

    def __call__(self, *a, **k):
        self.lines.append(" ".join(str(x) for x in a))

    @property
    def text(self):
        return "\n".join(self.lines)

    @property
    def failures(self):
        return [ln for ln in self.lines if "LORRAX SANITY FAILURE" in ln]


def _mesh(nx=1, ny=1):
    devs = np.array(jax.devices()[: nx * ny]).reshape(nx, ny)
    return Mesh(devs, ("x", "y"))


# ---------------------------------------------------------------------------
# 1. Device-side reductions
# ---------------------------------------------------------------------------

def test_device_check_finite_clean():
    a = jnp.arange(64, dtype=jnp.complex128).reshape(8, 8)
    log = _Log()
    assert sanity.check_finite("V_q", a, print_fn=log) is True
    assert log.failures == []


def test_device_check_finite_catches_nan():
    a = jnp.ones((8, 8), dtype=jnp.complex128)
    a = a.at[3, 4].set(jnp.nan)
    a = a.at[5, 1].set(jnp.inf)
    log = _Log()
    assert sanity.check_finite("W", a, print_fn=log) is False
    assert "2 non-finite entries of 64" in log.failures[0], log.text


def test_device_check_finite_complex_imag_nan():
    """``inf`` / ``nan`` hiding in the imaginary part must be caught."""
    a = jnp.ones((4, 4), dtype=jnp.complex128) + 1j * jnp.zeros((4, 4))
    a = a.at[1, 1].set(1.0 + 1j * jnp.nan)
    log = _Log()
    assert sanity.check_finite("Sigma", a, print_fn=log) is False


def test_device_check_finite_on_sharded_array():
    """The reduction must be global, not per-shard.

    A sharded (nq, mu, mu) V_q is the real input here; a reduction that
    silently operated on the local shard would miss a NaN living on
    another device — which is precisely the multi-rank failure mode these
    gates exist for.
    """
    n_dev = len(jax.devices())
    if n_dev < 2:
        print("  SKIP test_device_check_finite_on_sharded_array "
              "(needs >=2 devices; set XLA_FLAGS="
              "--xla_force_host_platform_device_count=4)")
        return
    mesh = _mesh(1, min(4, n_dev))
    sh = NamedSharding(mesh, P(None, "y"))
    host = np.ones((4, mesh.shape["y"] * 3), dtype=np.complex128)
    host[0, -1] = np.nan                    # lands on the LAST shard only
    a = jax.device_put(jnp.asarray(host), sh)
    log = _Log()
    assert sanity.check_finite("V_q(sharded)", a, print_fn=log) is False
    assert "1 non-finite" in log.failures[0], log.text


def test_device_check_hermitian():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    herm = jnp.asarray(0.5 * (m + m.conj().T))
    assert sanity.check_hermitian("V_q[0]", herm, print_fn=_Log()) is True
    broken = herm.at[2, 5].add(0.7)
    log = _Log()
    assert sanity.check_hermitian("V_q[0]", broken, print_fn=log) is False
    assert "is NOT Hermitian" in log.failures[0]


def test_device_gate_cost_is_one_fetch():
    """Sanity: the reduction returns scalars, not the array (cheapness gate)."""
    big = jnp.ones((256, 256), dtype=jnp.complex128)
    n_bad, n_nan, max_abs, size = sanity._finite_stats(big)
    assert (n_bad, n_nan, size) == (0, 0, 256 * 256)
    assert abs(max_abs - 1.0) < 1e-12


def test_off_switch_skips_device_work():
    prev = os.environ.get("LORRAX_SANITY")
    os.environ["LORRAX_SANITY"] = "0"
    try:
        log = _Log()
        bad = jnp.full((4, 4), jnp.nan, dtype=jnp.complex128)
        assert sanity.check_finite("V", bad, print_fn=log) is True
        assert log.lines == []
    finally:
        if prev is None:
            os.environ.pop("LORRAX_SANITY", None)
        else:
            os.environ["LORRAX_SANITY"] = prev


# ---------------------------------------------------------------------------
# 2/3. HDF5 provenance guards on zeta_q.h5
# ---------------------------------------------------------------------------

_FIXTURE_WFN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "regression", "cohsex_debug", "WFNsmall.h5")


def _fixture_fft_grid():
    """FFT grid of the regression WFN, so both headers agree."""
    with h5py.File(_FIXTURE_WFN, "r") as f:
        return tuple(int(v) for v in f["mf_header/gspace/FFTgrid"][:])


def _make_zeta_h5(path, *, n_rmu, n_q=2, ngkmax=7, done=True,
                  fft_grid=None, layout="G_flat", with_mf_header=True):
    """Minimal zeta_q.h5 carrying just what the guards read.

    ``ZetaLoader`` reads BOTH headers at open (``mf_header`` for the
    crystal/k-grid surface, ``isdf_header`` for the centroid surface), so
    a fixture with only the latter fails in the reader before reaching
    any gate.  The mf_header is copied verbatim from the read-only
    regression WFN using the same ``copy_mf_header`` helper the
    production ζ writer uses — the source is opened read-only and never
    modified.
    """
    from file_io.isdf_header import IsdfHeader, write_isdf_header
    from file_io.mf_header import copy_mf_header

    if fft_grid is None:
        fft_grid = _fixture_fft_grid()

    idx = np.stack(np.meshgrid(
        np.arange(n_rmu) % fft_grid[0],
        np.zeros(1, dtype=int), np.zeros(1, dtype=int),
        indexing="ij"), axis=-1).reshape(n_rmu, 3).astype(np.int32)
    gvec = np.zeros((n_q, 3, ngkmax), dtype=np.int32)
    ngk = np.full((n_q,), ngkmax, dtype=np.int32)
    hdr = IsdfHeader.build(
        r_mu_fft_idx=idx, fft_grid=fft_grid, density="scalar",
        vertex_mu_L=0, zeta_is_done=done, zeta_layout=layout,
        gvec_components=gvec, ngk_per_q=ngk, zeta_cutoff_ry=10.0)
    with h5py.File(path, "w") as f:
        if layout == "G_flat":
            f.create_dataset("zeta_q_G",
                             data=np.zeros((n_q, n_rmu, ngkmax),
                                           dtype=np.complex128))
        else:
            f.create_dataset("zeta_q",
                             data=np.zeros((n_q, 64, n_rmu),
                                           dtype=np.complex128))
    write_isdf_header(path, hdr, mode="a")
    if with_mf_header:
        copy_mf_header(_FIXTURE_WFN, path, dst_mode="a")
    return path


def test_zeta_basis_guard_sees_gflat_dataset():
    """The regression this fixes: the old guard probed the wrong dataset.

    A stale 276-centroid ζ in the run directory of a 606-centroid run has
    always been supposed to raise here.  Before the fix it did not, for
    every G-flat (i.e. every production) file.
    """
    from gw.gw_init import _check_zeta_h5_matches_basis
    with tempfile.TemporaryDirectory() as d:
        grid = _fixture_fft_grid()
        p = _make_zeta_h5(os.path.join(d, "zeta_q.h5"), n_rmu=276)
        # Matching basis ⇒ silent.
        _check_zeta_h5_matches_basis(p, 276, print_fn=_Log(), fft_grid=grid)
        # Mismatched basis ⇒ must raise.
        try:
            _check_zeta_h5_matches_basis(p, 606, print_fn=_Log(),
                                         fft_grid=grid)
        except ValueError as exc:
            assert "n_mu=276" in str(exc)
        else:
            raise AssertionError(
                "stale-basis zeta_q.h5 was accepted (the pre-fix bug)")


def test_zeta_basis_guard_catches_wrong_fft_grid():
    from gw.gw_init import _check_zeta_h5_matches_basis
    with tempfile.TemporaryDirectory() as d:
        # File built on the fixture's (large) grid; the run claims a
        # much smaller one, so the stored centroid indices fall outside it.
        p = _make_zeta_h5(os.path.join(d, "zeta_q.h5"), n_rmu=16)
        try:
            _check_zeta_h5_matches_basis(p, 16, print_fn=_Log(),
                                         fft_grid=(4, 4, 4))
        except ValueError as exc:
            assert "different FFT grid" in str(exc)
        else:
            raise AssertionError("wrong-grid zeta_q.h5 was accepted")


def test_zeta_basis_guard_warns_on_incomplete():
    from gw.gw_init import _check_zeta_h5_matches_basis
    log = _Log()
    with tempfile.TemporaryDirectory() as d:
        p = _make_zeta_h5(os.path.join(d, "zeta_q.h5"), n_rmu=16, done=False)
        _check_zeta_h5_matches_basis(p, 16, print_fn=log,
                                     fft_grid=_fixture_fft_grid())
    assert "zeta_is_done=False" in log.text, log.text


def test_zeta_loader_refuses_incomplete_file():
    """A ζ from a crashed fit must not open for reading."""
    from zeta_loader import ZetaLoader
    with tempfile.TemporaryDirectory() as d:
        p = _make_zeta_h5(os.path.join(d, "zeta_q.h5"), n_rmu=16, done=False)
        try:
            ZetaLoader(p, mesh=_mesh()).close()
        except ValueError as exc:
            assert "zeta_is_done=False" in str(exc)
        else:
            raise AssertionError(
                "ZetaLoader opened a ζ whose fit never finished")


def test_zeta_loader_override_env():
    from zeta_loader import ZetaLoader
    prev = os.environ.get("LORRAX_ALLOW_PARTIAL_ZETA")
    os.environ["LORRAX_ALLOW_PARTIAL_ZETA"] = "1"
    try:
        with tempfile.TemporaryDirectory() as d:
            p = _make_zeta_h5(os.path.join(d, "zeta_q.h5"), n_rmu=16,
                              done=False)
            ZetaLoader(p, mesh=_mesh()).close()      # must NOT raise
    finally:
        if prev is None:
            os.environ.pop("LORRAX_ALLOW_PARTIAL_ZETA", None)
        else:
            os.environ["LORRAX_ALLOW_PARTIAL_ZETA"] = prev


def test_zeta_loader_accepts_complete_file():
    from zeta_loader import ZetaLoader
    with tempfile.TemporaryDirectory() as d:
        p = _make_zeta_h5(os.path.join(d, "zeta_q.h5"), n_rmu=16, done=True)
        with ZetaLoader(p, mesh=_mesh()) as z:
            assert int(z.n_rmu) == 16
            assert z.zeta_layout == "G_flat"


# ---------------------------------------------------------------------------
# 3b. The eqp mean-field window is single-sourced from gw_output
# ---------------------------------------------------------------------------

def test_implied_vxc_window_is_sourced_from_gw_output():
    """With the driver importable, the writer must adopt ITS window.

    The pure-python suite can only check the fallback literals (it runs
    without jax, so ``gw.gw_output`` is unimportable there).  Here the real
    module is available, so this pins the actual single-sourcing: if N's
    work retunes the physical V_xc window, the writer-side gate follows
    automatically instead of silently drifting.
    """
    from gw import eqp_bgw, gw_output
    lo, hi = eqp_bgw._implied_vxc_window_ev()
    assert lo == float(gw_output._VXC_IMPLIED_MIN_EV), (lo, hi)
    assert hi == float(gw_output._VXC_IMPLIED_MAX_EV), (lo, hi)


# ---------------------------------------------------------------------------
# 3c. make_eqp_bgw honours the V_H-source contract (folded / stored)
# ---------------------------------------------------------------------------

def _make_eqp_cli_inputs(d, *, has_hartree=False, stored=False,
                         nk=2, nb=4, nb_file=6,
                         n_omega=5, kin_ion_ev=-300.0, v_h_ev=280.0,
                         e_dft_ev=-40.0, sigma_x_ev=-20.0,
                         stored_v_h_ev=None):
    """Build a minimal run dir for ``make_eqp_bgw``.

    Three shapes of the same physical system, which must therefore all
    produce the same eqp0:
      * legacy      — pristine kin_ion, no attrs; V_H from sigma_mnk
      * has_hartree — V_H folded INTO kin_ion's values (pre-v_hartree)
      * stored      — pristine kin_ion + a separate ``v_hartree`` array

    ``stored_v_h_ev`` overrides the value written into the stored array
    (default: the same ``v_h_ev`` sigma_mnk carries), which is how the
    substitution can be told apart from the suppression.

    The constants are chosen so the implied V_xc lands INSIDE the
    physical window (E_DFT − (kin_ion + V_H) ≈ −18…−12 eV).  That is not
    cosmetic: a test fixture that trips the very guard under test trains
    the reader to ignore the guard's output, which is the exact failure
    mode this whole workstream exists to prevent.  The one place a firing
    is expected is
    ``test_make_eqp_bgw_double_count_would_be_caught``, where it is the
    assertion's whole point.
    """
    from common.units import RYD_TO_EV
    band_start, band_stop = 1, 1 + nb

    kin_diag_ev = kin_ion_ev + (v_h_ev if has_hartree else 0.0)
    kin = np.zeros((nk, nb_file, nb_file))
    for b in range(nb_file):
        kin[:, b, b] = kin_diag_ev
    with h5py.File(os.path.join(d, "kin_ion.h5"), "w") as f:
        ds = f.create_dataset("kin_ion", data=kin / RYD_TO_EV)
        if has_hartree:
            ds.attrs["has_hartree"] = True
        if stored:
            vh_stored = np.zeros((nk, nb_file, nb_file), dtype=np.complex128)
            for b in range(nb_file):
                vh_stored[:, b, b] = (
                    v_h_ev if stored_v_h_ev is None else stored_v_h_ev)
            f.create_dataset("v_hartree", data=vh_stored / RYD_TO_EV)

    with h5py.File(os.path.join(d, "qp_wfn_rotations.h5"), "w") as f:
        f.create_dataset("band_range", data=np.array([band_start, band_stop]))
        f.create_dataset("kirr_to_kfull", data=np.arange(nk))

    # WFN: nb_file bands, VBM inside the window so the E_F check passes.
    en = np.zeros((1, nk, nb_file))
    for b in range(nb_file):
        en[0, :, b] = (e_dft_ev + 2.0 * b) / RYD_TO_EV
    with h5py.File(os.path.join(d, "WFN.h5"), "w") as f:
        g = f.create_group("mf_header/kpoints")
        g.create_dataset("rk", data=np.zeros((nk, 3)))
        g.create_dataset("nspin", data=np.int32(1))
        g.create_dataset("el", data=en)
        g.create_dataset("ifmax", data=np.full((1, nk), band_start + 2))

    # sigma_mnk: the ISDF Hartree column is ALWAYS non-zero on disk --
    # that is the mixed case this contract exists for.
    sx = np.zeros((nk, nb_file, nb_file), dtype=np.complex128)
    vh = np.zeros((nk, nb_file, nb_file), dtype=np.complex128)
    for b in range(nb_file):
        sx[:, b, b] = sigma_x_ev
        vh[:, b, b] = v_h_ev
    sc = np.zeros((n_omega, nk, nb_file, nb_file), dtype=np.complex128)
    with h5py.File(os.path.join(d, "sigma_mnk.h5"), "w") as f:
        f.create_dataset("omega_ev", data=np.linspace(-10.0, 10.0, n_omega))
        f.create_dataset("sigma_sx_kij_ev", data=sx)
        f.create_dataset("hartree_kij_ev", data=vh)
        f.create_dataset("sigma_c_kij_ev", data=sc)
    return d


def _read_eqp_qp_column(path):
    vals = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        t = line.split()
        if len(t) == 4 and "." not in t[0]:
            vals.append(float(t[3]))
    return np.array(vals)


def test_make_eqp_bgw_suppresses_hartree_when_folded():
    """A folded kin_ion must NOT get sigma_mnk's ISDF V_H added on top.

    Production job 7874840 hit exactly this: Q's regenerated
    ``kin_ion.h5`` (has_hartree=True) symlinked into a c606 run dir whose
    ``sigma_mnk.h5`` predated the contract.  ~500 eV of Hartree was
    counted twice, the rebuilt QP gap came out at -453 eV, and the CLI
    exited 0.  Both modes below describe the same physical system, so
    their eqp0 columns must agree.
    """
    from gw.eqp_bgw import make_eqp_bgw
    with tempfile.TemporaryDirectory() as d_leg, \
         tempfile.TemporaryDirectory() as d_new:
        _make_eqp_cli_inputs(d_leg, has_hartree=False)
        _make_eqp_cli_inputs(d_new, has_hartree=True)
        make_eqp_bgw(d_leg)
        make_eqp_bgw(d_new)
        legacy = _read_eqp_qp_column(os.path.join(d_leg, "eqp0.dat"))
        folded = _read_eqp_qp_column(os.path.join(d_new, "eqp0.dat"))
    assert legacy.size and legacy.size == folded.size
    assert np.allclose(legacy, folded, atol=1e-9), (
        f"folded-kin_ion path disagrees with legacy on identical physics:\n"
        f"  legacy {legacy[:4]}\n  folded {folded[:4]}\n"
        f"  max|diff| = {np.max(np.abs(legacy - folded)):.6f} eV")


def test_make_eqp_bgw_stored_array_matches_the_folded_route():
    """**THE migration gate.**  Stored ``v_hartree`` ≡ the fold-in it replaces.

    Same physics, three encodings — legacy (V_H only in sigma_mnk),
    folded (V_H inside kin_ion's values, the format N gated), and stored
    (pristine kin_ion + its own ``v_hartree`` array).  eqp0 must be
    identical, or the new file format silently changes answers.
    """
    from gw.eqp_bgw import make_eqp_bgw
    with tempfile.TemporaryDirectory() as d_fold, \
         tempfile.TemporaryDirectory() as d_store:
        _make_eqp_cli_inputs(d_fold, has_hartree=True)
        _make_eqp_cli_inputs(d_store, stored=True)
        make_eqp_bgw(d_fold)
        make_eqp_bgw(d_store)
        folded = _read_eqp_qp_column(os.path.join(d_fold, "eqp0.dat"))
        stored = _read_eqp_qp_column(os.path.join(d_store, "eqp0.dat"))
    assert stored.size and stored.size == folded.size
    assert np.allclose(folded, stored, atol=1e-9), (
        f"stored-v_hartree route disagrees with the fold-in it replaces:\n"
        f"  folded {folded[:4]}\n  stored {stored[:4]}\n"
        f"  max|diff| = {np.max(np.abs(folded - stored)):.6f} eV")


def test_make_eqp_bgw_stored_array_is_substituted_not_ignored():
    """The stored array must REPLACE sigma_mnk's ISDF column, not be a no-op.

    Pin the size of the substitution: a stored V_H that differs from the
    ISDF one by 30 eV must move eqp0 by exactly that.  Without this, a
    silently-ignored ``v_hartree`` would still pass the equality gate
    above (both files carry the same number there).
    """
    from gw.eqp_bgw import make_eqp_bgw
    with tempfile.TemporaryDirectory() as d_a, \
         tempfile.TemporaryDirectory() as d_b:
        _make_eqp_cli_inputs(d_a, stored=True)
        _make_eqp_cli_inputs(d_b, stored=True, stored_v_h_ev=310.0)
        make_eqp_bgw(d_a)
        make_eqp_bgw(d_b)
        a = _read_eqp_qp_column(os.path.join(d_a, "eqp0.dat"))
        b = _read_eqp_qp_column(os.path.join(d_b, "eqp0.dat"))
    shift = float(np.mean(b - a))
    assert abs(shift - 30.0) < 1e-6, (
        f"expected the stored V_H (+30 eV) to move eqp0 by 30 eV; "
        f"got {shift:.6f} eV — the array is being ignored")


def test_kin_ion_hartree_source_resolution():
    """The precedence ladder and the refusals, on real files."""
    from file_io.kin_ion import (
        kin_ion_hartree_source, resolve_hartree_source)
    with tempfile.TemporaryDirectory() as d_leg, \
         tempfile.TemporaryDirectory() as d_fold, \
         tempfile.TemporaryDirectory() as d_store:
        _make_eqp_cli_inputs(d_leg)
        _make_eqp_cli_inputs(d_fold, has_hartree=True)
        _make_eqp_cli_inputs(d_store, stored=True)
        p_leg = os.path.join(d_leg, "kin_ion.h5")
        p_fold = os.path.join(d_fold, "kin_ion.h5")
        p_store = os.path.join(d_store, "kin_ion.h5")

        assert kin_ion_hartree_source(p_leg) == "none"
        assert kin_ion_hartree_source(p_fold) == "folded"
        assert kin_ion_hartree_source(p_store) == "stored"

        # auto: stored wins, then folded, then isdf.
        assert resolve_hartree_source(p_store, "auto", print_fn=_Log()) == "stored"
        assert resolve_hartree_source(p_fold, "auto", print_fn=_Log()) == "folded"
        assert resolve_hartree_source(p_leg, "auto", print_fn=_Log()) == "isdf"

        # Explicit requests are honoured on a pristine file...
        assert resolve_hartree_source(p_store, "isdf", print_fn=_Log()) == "isdf"
        assert resolve_hartree_source(p_leg, "gspace", print_fn=_Log()) == "gspace"

        # ...but a folded file may not be combined with another source:
        # its values already contain V_H, so anything else double counts.
        for req in ("isdf", "gspace"):
            try:
                resolve_hartree_source(p_fold, req, print_fn=_Log())
            except ValueError as exc:
                assert "double count" in str(exc)
            else:
                raise AssertionError(
                    f"hartree_source={req} on a folded file must raise")

        # stored requested but absent must name the fix, not fall back.
        try:
            resolve_hartree_source(p_leg, "stored", print_fn=_Log())
        except ValueError as exc:
            assert "v_hartree" in str(exc)
        else:
            raise AssertionError("hartree_source=stored on a legacy file must raise")

        # An unknown value never silently becomes 'auto'.
        try:
            resolve_hartree_source(p_leg, "exact", print_fn=_Log())
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown hartree_source must raise")


def test_kin_ion_has_hartree_is_false_for_the_stored_format():
    """Back-compat in the safe direction.

    ``kin_ion_has_hartree`` means "V_H is inside the kin_ion VALUES".  A
    new-format file has pristine values, so it must answer False — that
    is exactly what lets an OLD reader treat it as an ionic-only file and
    correctly add its own ISDF V_H instead of double counting.
    """
    from file_io.kin_ion import kin_ion_has_hartree
    with tempfile.TemporaryDirectory() as d:
        _make_eqp_cli_inputs(d, stored=True)
        assert kin_ion_has_hartree(os.path.join(d, "kin_ion.h5")) is False


def _eqp_body_bytes(path):
    """The file minus its ``#`` provenance line (which carries a timestamp)."""
    with open(path) as fh:
        return "".join(l for l in fh if not l.startswith("#")).encode()


def _assemble_from_run_dir_via_in_memory(d, out0, out1):
    """Load the CLI's artifacts by hand and go through the LIVE entry point.

    Deliberately does NOT reuse ``make_eqp_bgw``'s loader: the point is to
    reach ``write_eqp_bgw_in_memory`` — what ``gw_output.write_results``
    calls — with the same numbers the CLI reads, and prove the two entry
    points cannot produce different files.
    """
    from common.units import RYD_TO_EV
    from gw.eqp_bgw import write_eqp_bgw_in_memory
    from file_io.kin_ion import (kin_ion_hartree_source, HARTREE_DATASET,
                                 read_star_map)
    from file_io.sigma_output import k_irr_rows_for

    with h5py.File(os.path.join(d, "WFN.h5"), "r") as w:
        kpts = np.asarray(w["mf_header/kpoints/rk"])
        en = np.asarray(w["mf_header/kpoints/el"])
        ifmax = np.asarray(w["mf_header/kpoints/ifmax"])
    with h5py.File(os.path.join(d, "qp_wfn_rotations.h5"), "r") as q:
        b0, b1 = (int(x) for x in np.asarray(q["band_range"]))
        kmap = np.asarray(q["kirr_to_kfull"], dtype=np.int64)
    e_dft = en[0, :, b0:b1] * RYD_TO_EV
    nocc = int(np.max(ifmax[0])) - 1 - b0
    efermi = 0.5 * (float(np.max(e_dft[:, : nocc + 1]))
                    + float(np.min(e_dft[:, nocc + 1:])))
    kip = os.path.join(d, "kin_ion.h5")
    with h5py.File(kip, "r") as k:
        kin = np.asarray(k["kin_ion"])[kmap, b0:b1, b0:b1]
    kin_diag = np.real(np.diagonal(kin, axis1=1, axis2=2)) * RYD_TO_EV
    # The same k-row resolution ``gw.eqp_bgw`` does, for the same reason:
    # since the k_irr extraction ``sigma_mnk.h5`` may store one row per
    # star, and the row this gate wants is then at ``irr_idx_k[kmap]``
    # rather than at ``kmap``.  An unstamped file is full-BZ and takes the
    # first branch, which is what every fixture and older run does.
    smp = os.path.join(d, "sigma_mnk.h5")
    sig_star = read_star_map(smp, "sigma_c_kij_ev", k_axis=1)
    krows = kmap if sig_star is None else k_irr_rows_for(
        kmap, sig_star[0], what="sigma_mnk.h5 (sanity gate)")
    with h5py.File(smp, "r") as s:
        om = np.asarray(s["omega_ev"], dtype=np.float64)
        sx = np.asarray(s["sigma_sx_kij_ev"])[krows][:, b0:b1, b0:b1]
        vh = np.asarray(s["hartree_kij_ev"])[krows][:, b0:b1, b0:b1]
        sc = np.asarray(s["sigma_c_kij_ev"])[:, krows][:, :, b0:b1, b0:b1]
    src = kin_ion_hartree_source(kip)
    vh_exact = None
    if src == "stored":
        with h5py.File(kip, "r") as k:
            vhf = np.asarray(k[HARTREE_DATASET])[kmap, b0:b1, b0:b1]
        vh_exact = np.real(np.diagonal(vhf, axis1=1, axis2=2)) * RYD_TO_EV
    return write_eqp_bgw_in_memory(
        eqp0_path=os.path.join(d, out0), eqp1_path=os.path.join(d, out1),
        kpoints_irr_frac=kpts, band_offset=b0, e_dft_ev=e_dft,
        kin_ion_diag_ev=kin_diag,
        hartree_diag_ev=np.real(np.diagonal(vh, axis1=1, axis2=2)),
        sigma_x_diag_ev=np.real(np.diagonal(sx, axis1=1, axis2=2)),
        sigma_c_at_dft_diag_ev=None,
        sigma_c_omega_diag_ev=np.diagonal(sc, axis1=2, axis2=3),
        omega_rel_ev=om, e_dft_rel_ev=e_dft - efermi,
        nspin=1, hartree_source=src, exact_hartree_diag_ev=vh_exact,
    )


def test_both_eqp_entry_points_are_byte_identical():
    """**THE unification gate (AD).**  One assembly, one formatter.

    ``gw_output.write_results`` (live driver) and ``make_eqp_bgw``
    (post-hoc CLI) now differ only in how the arrays are obtained: both
    end in ``assemble_eqp`` → ``EqpAssembly.write``.  Fed the same
    artifacts they must therefore emit eqp0.dat / eqp1.dat that are
    byte-identical past the provenance line — including the V_H seam,
    which is the piece that used to be written out twice and drifted
    (job 7874840, QP gap −453 eV at rc=0).

    Run on all three file shapes, because the seam takes a different
    branch in each: legacy (as-given), folded (suppressed), stored
    (substituted).
    """
    from gw.eqp_bgw import make_eqp_bgw
    for label, kw in (("legacy", {}),
                      ("folded", dict(has_hartree=True)),
                      ("stored", dict(stored=True, stored_v_h_ev=310.0))):
        with tempfile.TemporaryDirectory() as d:
            _make_eqp_cli_inputs(d, **kw)
            make_eqp_bgw(d)                       # -> eqp0.dat / eqp1.dat
            _assemble_from_run_dir_via_in_memory(
                d, "eqp0_mem.dat", "eqp1_mem.dat")
            for a, b in (("eqp0.dat", "eqp0_mem.dat"),
                         ("eqp1.dat", "eqp1_mem.dat")):
                ba = _eqp_body_bytes(os.path.join(d, a))
                bb = _eqp_body_bytes(os.path.join(d, b))
                assert ba == bb, (
                    f"[{label}] {a} from the CLI differs from {b} from the "
                    f"live in-memory entry point — the two paths have "
                    f"independent semantics again.\n"
                    f"  CLI  : {ba[:200]!r}\n  live : {bb[:200]!r}")


def test_make_eqp_bgw_double_count_would_be_caught():
    """Pin the size of the bug: not suppressing V_H shifts eqp0 by ~V_H."""
    from gw import eqp_bgw
    with tempfile.TemporaryDirectory() as d:
        _make_eqp_cli_inputs(d, has_hartree=True)
        eqp_bgw.make_eqp_bgw(d)
        good = _read_eqp_qp_column(os.path.join(d, "eqp0.dat"))
        # Simulate the pre-fix behaviour by stripping the contract flag.
        with h5py.File(os.path.join(d, "kin_ion.h5"), "a") as f:
            del f["kin_ion"].attrs["has_hartree"]
        eqp_bgw.make_eqp_bgw(d, eqp0_out="eqp0_bad.dat",
                             eqp1_out="eqp1_bad.dat")
        bad = _read_eqp_qp_column(os.path.join(d, "eqp0_bad.dat"))
    shift = float(np.mean(bad - good))
    assert abs(shift - 280.0) < 1e-6, (
        f"expected the un-suppressed run to be high by the V_H column "
        f"(280 eV); got {shift:.3f} eV")


# ---------------------------------------------------------------------------
# 3d. kin_ion_io's V_H defaults — computed, stored separately, never folded
# ---------------------------------------------------------------------------

def test_kin_ion_io_writes_the_stored_hartree_array_by_default():
    """V_H is computed by default and STORED SEPARATELY, never folded.

    Three defaults are pinned here because each silently changes what
    every downstream consumer sees:
      * ``hartree`` on  — the file carries ``v_hartree``, so a run gets
        the exact V_H without a second generator pass;
      * ``fold_hartree`` off — ``kin_ion`` stays pristine, which is what
        keeps one file usable by both the exact and the ISDF route and
        keeps ``kin_ion_has_hartree`` False for old readers;
      * ``--no-hartree`` still available for an ionic-only file.
    """
    from gw.kin_ion_io import build_argparser
    p = build_argparser()
    a = p.parse_args(["-i", "x.in"])
    assert a.hartree is True, "V_H must be computed by default"
    assert a.fold_hartree is False, (
        "kin_ion must stay pristine by default — V_H goes in its own array")
    assert p.parse_args(["-i", "x.in", "--no-hartree"]).hartree is False
    assert p.parse_args(["-i", "x.in", "--fold-hartree"]).fold_hartree is True
    # Symmetric pair: last flag wins, in both orders.
    assert p.parse_args(["-i", "x.in", "--hartree", "--no-hartree"]).hartree is False
    assert p.parse_args(["-i", "x.in", "--no-hartree", "--hartree"]).hartree is True


def test_kin_ion_io_catches_a_BROKEN_multiprocess_launch():
    """The generator now distributes — but a *failed* distributed init
    still has to fail loudly.

    The old guard refused ``srun -n P`` outright.  Since the CLI is
    k-partitioned with a coordinated rank-0 write, ``-n P`` is the
    supported way to run it.  What remains fatal is the launcher
    advertising P tasks while ``jax.distributed`` joined a world of 1:
    then there is no world to partition over, every task computes the
    whole thing and every task believes it is rank 0 — the original
    clobber with none of the safety.  This test drives exactly that
    state (SLURM_NTASKS=4 in a single-process test process) and fires
    before any file is opened (hence the nonexistent input path).
    """
    from gw import kin_ion_io
    prev = os.environ.get("SLURM_NTASKS")
    prev_pc = os.environ.get("JAX_PROCESS_COUNT")
    os.environ.pop("JAX_PROCESS_COUNT", None)
    os.environ["SLURM_NTASKS"] = "4"
    try:
        msg = None
        try:
            kin_ion_io.main(["-i", "/nonexistent/deck.in"])
        except SystemExit as exc:
            msg = str(exc)
        assert msg and "joined a world of" in msg, (
            f"expected a broken-launch refusal, got: {msg!r}")
    finally:
        for k, v in (("SLURM_NTASKS", prev), ("JAX_PROCESS_COUNT", prev_pc)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# 3b. The exact-V_H distribution layer (workstream X)
# ---------------------------------------------------------------------------

def test_rho_work_items_cover_every_band_exactly_once():
    """The ρ partition must be a partition — no band counted twice, none
    dropped.  A duplicated (k, band) silently inflates a ~500 eV term."""
    from gw.kin_ion_io import rho_work_items
    for nk, nocc, world in ((9, 26, 1), (9, 26, 4), (9, 26, 16), (9, 26, 64),
                            (144, 26, 1), (144, 26, 16), (144, 26, 80),
                            (144, 26, 144), (144, 26, 512), (4, 3, 64)):
        items = rho_work_items(nk, nocc, world)
        seen = {}
        for ik, lo, hi in items:
            for b in range(lo, hi):
                key = (ik, b)
                assert key not in seen, f"duplicate {key} at P={world}"
                seen[key] = True
        assert len(seen) == nk * nocc, (
            f"P={world}: covered {len(seen)} of {nk * nocc} (k, band) pairs")


def test_rho_work_items_are_the_serial_sweep_when_P_le_nk():
    """THE BIT-PARITY PRECONDITION.

    At ``world <= nk`` the sweep must be exactly the serial one — one
    item per k, in k order, with the whole occupied manifold — because
    that is what makes the P=1 result bit-for-bit the pre-distribution
    result rather than merely equal to 1e-16.
    """
    from gw.kin_ion_io import rho_work_items
    for world in (1, 2, 9, 144):
        nk, nocc = 144, 26
        if world > nk:
            continue
        assert rho_work_items(nk, nocc, world) == [
            (ik, 0, nocc) for ik in range(nk)], f"P={world} reordered the sweep"


def test_rho_work_items_balance_within_one_item():
    """Round-robin, not contiguous blocks: nk=9 at P=4 must be 3/2/2/2,
    not 3/3/3/0 (which idles a quarter of the machine)."""
    from gw.kin_ion_io import rho_work_items
    items = rho_work_items(9, 26, 4)
    counts = [len(items[r::4]) for r in range(4)]
    assert max(counts) - min(counts) <= 1, counts
    assert sum(counts) == len(items)


def test_valence_density_nocc_none_matches_the_sliced_form():
    """``nocc=None`` is a slicing convention, not a second quadrature."""
    from psp.get_DFT_mtxels import valence_density_from_kpoint
    rng = np.random.default_rng(0)
    box = (rng.standard_normal((5, 2, 4, 4, 6))
           + 1j * rng.standard_normal((5, 2, 4, 4, 6)))
    box = jnp.asarray(box, dtype=jnp.complex128)
    a = valence_density_from_kpoint(box, nocc=5, weight=0.25,
                                    cell_volume=13.0, spin_degeneracy=2.0)
    b = valence_density_from_kpoint(box, nocc=None, weight=0.25,
                                    cell_volume=13.0, spin_degeneracy=2.0)
    assert float(jnp.abs(a - b).max()) == 0.0
    # and the sub-window spelling equals summing the whole one
    c = valence_density_from_kpoint(box[:2], nocc=None, weight=0.25,
                                    cell_volume=13.0, spin_degeneracy=2.0)
    d = valence_density_from_kpoint(box[2:], nocc=None, weight=0.25,
                                    cell_volume=13.0, spin_degeneracy=2.0)
    assert float(jnp.abs(a - (c + d)).max()) < 1e-13 * float(jnp.abs(a).max())


def test_collective_helpers_are_the_identity_at_P1():
    """P=1 must not go anywhere near a collective — that is what makes
    the single-node CLI bit-identical to its pre-distribution self."""
    import jax as _jax
    # From the SERVICE, not from the kin_ion driver: these are generic
    # k-partition plumbing and the driver only ever re-exported them.
    from common.collectives import (psum_replicate as _psum_replicate,
                                    gather_indexed_blocks as _gather_indexed_blocks,
                                    replicate_to_mesh, resolve_mesh)
    if int(_jax.process_count()) != 1:
        return
    rng = np.random.default_rng(1)
    rho = rng.standard_normal((3, 4, 5))
    out = _psum_replicate(rho, resolve_mesh())
    assert np.array_equal(out, rho)

    vals = rng.standard_normal((3, 2, 2)) + 0j
    idx = np.array([2, 0, -1], dtype=np.int32)      # one padding slot
    g = _gather_indexed_blocks(vals, idx, 3)
    assert np.array_equal(g[2], vals[0]) and np.array_equal(g[0], vals[1])
    assert np.array_equal(g[1], np.zeros((2, 2), dtype=g.dtype))

    r = replicate_to_mesh(rho, resolve_mesh())
    assert np.allclose(np.asarray(r), rho)


def test_hartree_mesh_resolution_is_square_and_covers_every_device():
    from common.collectives import resolve_mesh
    m = resolve_mesh()
    assert tuple(m.axis_names) == ('x', 'y')
    assert int(m.devices.size) == int(jax.device_count())


_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "regression", "cohsex_debug")


def test_rotated_density_load_reduces_to_the_plain_load_at_U_identity():
    """The QSGW density seam must be a no-op at U = 1.

    ``build_valence_density_distributed(psi_rotation=U)`` is what lets a
    density-updating SC loop rebuild ρ from the CURRENT orbitals.  With
    U the identity it must reproduce the DFT-orbital load exactly, or
    the seam is quietly changing the one-shot answer too.
    """
    wfn_path = os.path.join(_FIXTURE, "WFNsmall.h5")
    if not os.path.exists(wfn_path):
        # A bare ``return`` here until 2026-08-07: pytest reports PASS for a
        # cell that ran nothing, which is worse than a skip because nothing
        # in the summary line says the coverage went away.  The fixture is
        # CHECKED IN (tests/regression/cohsex_debug/WFNsmall.h5, chmod a-w
        # by tests/conftest.py), so this never fires in the monorepo -- and
        # that is exactly why the silent form could sit here unnoticed.
        pytest.skip(f"checked-in deck absent: {wfn_path} is not in this "
                    f"tree, so the QSGW density seam has no operands -- "
                    f"covered by the monorepo run, where the fixture is "
                    f"committed")
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader
    from common import Meta
    import symmetry_maps
    from common.wfn_transforms import load_kpoint_fftbox_local
    from gw.kin_ion_io import _load_rotated_occ_fftbox
    wfn = WfnLoader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)
    meta = Meta.from_system(wfn, sym, 4, 4, 8, 0, False)
    nmix = 8
    plain = load_kpoint_fftbox_local(wfn, meta, 0, nmix)
    U = np.eye(nmix, dtype=np.complex128)
    rot = _load_rotated_occ_fftbox(wfn, meta, 0, U)
    assert plain.shape == rot.shape, (plain.shape, rot.shape)
    assert float(jnp.abs(plain - rot).max()) == 0.0
    # a pure band SWAP must permute, not change, the density
    from psp.get_DFT_mtxels import valence_density_from_kpoint
    Us = np.eye(nmix, dtype=np.complex128)[:, [1, 0] + list(range(2, nmix))]
    swapped = _load_rotated_occ_fftbox(wfn, meta, 0, Us)
    kw = dict(nocc=None, weight=1.0, cell_volume=float(wfn.cell_volume),
              spin_degeneracy=1.0)
    r0 = valence_density_from_kpoint(plain, **kw)
    r1 = valence_density_from_kpoint(swapped, **kw)
    assert float(jnp.abs(r0 - r1).max()) < 1e-10 * float(jnp.abs(r0).max())
    wfn.close()


def test_process_local_load_matches_the_legacy_wrapper():
    """``load_kpoint_fftbox`` must keep its values while gaining a
    process-local backend — every existing single-process caller
    (orbital_magnetization, scf_potential, run_sternheimer, …) depends
    on it."""
    wfn_path = os.path.join(_FIXTURE, "WFNsmall.h5")
    if not os.path.exists(wfn_path):
        # A bare ``return`` here until 2026-08-07: pytest reports PASS for a
        # cell that ran nothing, which is worse than a skip because nothing
        # in the summary line says the coverage went away.  The fixture is
        # CHECKED IN (tests/regression/cohsex_debug/WFNsmall.h5, chmod a-w
        # by tests/conftest.py), so this never fires in the monorepo -- and
        # that is exactly why the silent form could sit here unnoticed.
        pytest.skip(f"checked-in deck absent: {wfn_path} is not in this "
                    f"tree, so the QSGW density seam has no operands -- "
                    f"covered by the monorepo run, where the fixture is "
                    f"committed")
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader
    from common import Meta
    import symmetry_maps
    from common.wfn_transforms import (load_kpoint_fftbox,
                                       load_kpoint_fftbox_local)
    wfn = WfnLoader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)
    meta = Meta.from_system(wfn, sym, 4, 4, 8, 0, False)
    a = load_kpoint_fftbox(wfn, sym, meta, 1, 6)
    b = load_kpoint_fftbox_local(wfn, meta, 1, 6)
    assert float(jnp.abs(a - b).max()) == 0.0
    # the band sub-window is a plain slice of the full window
    c = load_kpoint_fftbox_local(wfn, meta, 1, 6, b_lo=2)
    assert float(jnp.abs(a[2:6] - c).max()) == 0.0
    wfn.close()


def test_hartree_cache_can_be_invalidated_for_a_density_updating_loop():
    from gw import sigma_dispatch
    sigma_dispatch._hartree_cache[("sentinel",)] = ("isdf", None)
    sigma_dispatch.invalidate_hartree_cache()
    assert sigma_dispatch._hartree_cache == {}


# ---------------------------------------------------------------------------
# 4. Collective barrier + fail-fast hook
# ---------------------------------------------------------------------------

def test_barrier_single_process_is_noop():
    from common.collectives import barrier, process_count
    assert process_count() == jax.process_count()
    if jax.process_count() <= 1:
        assert barrier("unit_test", print_fn=_Log()) is False
    else:
        assert barrier("unit_test", print_fn=_Log()) is True


def test_failfast_hook_not_installed_single_process():
    """The hook must stay OFF in single-process runs (normal tracebacks).

    ``SLURM_NTASKS`` is inherited from the batch script even when this
    suite runs on one rank, so the single-process condition is forced
    here rather than read from the ambient environment — otherwise the
    assertion silently no-ops under sbatch and tests nothing.  The
    install sentinel is cleared too, since an earlier test in the same
    interpreter may have set it.
    """
    import runtime
    prev_n = os.environ.get("SLURM_NTASKS")
    prev_hook = sys.excepthook
    had_sentinel = getattr(sys, "_lorrax_failfast_installed", False)
    os.environ["SLURM_NTASKS"] = "1"
    if had_sentinel:
        del sys._lorrax_failfast_installed
    try:
        runtime.install_failfast_excepthook()
        assert sys.excepthook is prev_hook, (
            "fail-fast hook installed in a single-process run; it would "
            "replace normal tracebacks with os._exit(1) for no reason")
    finally:
        sys.excepthook = prev_hook
        if had_sentinel:
            sys._lorrax_failfast_installed = True
        if prev_n is None:
            os.environ.pop("SLURM_NTASKS", None)
        else:
            os.environ["SLURM_NTASKS"] = prev_n


def test_failfast_hook_respects_optout():
    import runtime
    prev_n = os.environ.get("SLURM_NTASKS")
    prev_f = os.environ.get("LORRAX_FAILFAST")
    os.environ["SLURM_NTASKS"] = "4"
    os.environ["LORRAX_FAILFAST"] = "0"
    before = sys.excepthook
    had_sentinel = getattr(sys, "_lorrax_failfast_installed", False)
    if had_sentinel:
        del sys._lorrax_failfast_installed
    try:
        runtime.install_failfast_excepthook()
        assert sys.excepthook is before, (
            "LORRAX_FAILFAST=0 did not suppress hook installation")
    finally:
        sys.excepthook = before
        if had_sentinel:
            sys._lorrax_failfast_installed = True
        for k, v in (("SLURM_NTASKS", prev_n), ("LORRAX_FAILFAST", prev_f)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------

def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}", flush=True)
        except Exception as exc:               # noqa: BLE001 - test runner
            import traceback
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())


def test_check_hermitian_sharded_no_full_gather():
    """The hermiticity gate must never materialise a replicated (n, n) tile.

    Regression for the AQ 4962c/P=64 collective-table finding (2026-07-28):
    the eager transpose+subtract — and its first, naively-jitted fix — let
    the SPMD partitioner resolve the P(x,y)/P(y,x) operand conflict by
    ALL-GATHERING BOTH full (μ, μ) operands onto every rank (2 × 398.72 MB
    at μ=4962, modules jit_subtract 0730/0973 then jit_fn 0536/0698).  The
    fix pins the transposed operand to the input sharding
    (``with_sharding_constraint``), forcing a tile-local reshard.  This
    test compiles the fused stats kernel on a 2×2-sharded tile and asserts
    the optimized HLO carries no all-gather with the full (n, n) extent —
    the same predicate wk_AN/colltable.py applies to production dumps.
    """
    # NOT a ``mesh(4)`` cell, and the reason is at the top of this file:
    # line 40 pins ``JAX_PLATFORMS=cpu`` for the whole module, at import,
    # because that is the only place it can be set.  So this gate is an
    # EMULATED-device gate by construction — it asks the CPU backend for four
    # host devices — and the mesh marker would hand it to a child started for
    # four A100s, which then refuses to be quietly emulated (correctly) and
    # reports a red that is about the marker rather than about the code.
    # MEASURED, Perlmutter 2026-08-10: exactly that, `came up with 1 on
    # platform 'cpu'`.
    #
    # It therefore keeps the inline guard AND its old behaviour: in a census
    # some earlier module has already built the CUDA backend, this module's
    # `JAX_PLATFORMS` write is inert, `len(jax.devices())` is 1, and the cell
    # skips.  That is a real loss and it is NOT fixed here — it is the
    # separate problem of a module that must own the whole process's platform
    # to run at all, which no per-cell marker can solve.  Recorded as an owner
    # row rather than papered over.
    if len(jax.devices()) < 4:
        import pytest
        pytest.skip("needs 4 (emulated) devices; this module pins "
                    "JAX_PLATFORMS=cpu at import, so under the suite the "
                    "backend is already CUDA and the pin is inert")
    mesh = _mesh(2, 2)
    n = 64
    sh = NamedSharding(mesh, P("x", "y"))
    rng = np.random.default_rng(7)
    m = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    herm = jnp.asarray(m + m.conj().T)          # Hermitian by construction
    a = jax.device_put(herm, sh)

    # Numeric contract: Hermitian input → ~0 relative residual; a broken
    # tile (one element flipped) must be caught.
    assert sanity.check_hermitian("gate[test]", a, rtol=1e-10)
    bad = herm.at[3, 5].set(herm[3, 5] + 17.0)
    log = _Log()
    assert not sanity.check_hermitian(
        "gate[test-bad]", jax.device_put(bad, sh), rtol=1e-10, print_fn=log)

    # Collective contract: no full-(n, n) all-gather in the compiled HLO.
    fn = sanity._HERM_STATS_CACHE[("fn", a.shape, str(a.dtype), a.sharding)]
    txt = fn.lower(a).compile().as_text()
    offenders = [
        ln for ln in txt.splitlines()
        if "all-gather" in ln and f"[{n},{n}]" in ln.replace(" ", "")
    ]
    assert not offenders, (
        "hermiticity gate re-grew a full-tile all-gather:\n"
        + "\n".join(offenders))
