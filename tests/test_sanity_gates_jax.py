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
    from file_io.zeta_loader import ZetaLoader
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
    from file_io.zeta_loader import ZetaLoader
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
    from file_io.zeta_loader import ZetaLoader
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
# 3c. make_eqp_bgw honours the has_hartree no-double-count contract
# ---------------------------------------------------------------------------

def _make_eqp_cli_inputs(d, *, has_hartree, nk=2, nb=4, nb_file=6,
                         n_omega=5, kin_ion_ev=-300.0, v_h_ev=280.0,
                         e_dft_ev=-40.0, sigma_x_ev=-20.0):
    """Build a minimal run dir for ``make_eqp_bgw``.

    ``kin_ion`` is written as if V_H were folded in when
    ``has_hartree`` is set, so the two modes describe the SAME physical
    system and must therefore produce the same eqp0.

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
