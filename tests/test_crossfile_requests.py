"""Gates for the seven-agent cross-file request set landed 2026-07-30.

RUNNABLE ON THE LOGIN NODE with plain ``python3`` — no jax, no h5py::

    cd /work2/08271/jackmc/frontera/lorrax
    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 tests/test_crossfile_requests.py

(also collects under pytest.)

Why source-level
----------------
Every file these requests touch — ``gw/gw_jax.py``, ``bse/bse_loading.py``,
``file_io/_slab_io_ffi.py``, ``gw/isdf_fitting.py`` — imports jax at module
scope, so importing them needs a compute node.  Parsing them does not.  The
two requests that are *behavioural* rather than structural
(``runtime._env_falsy``, ``runtime._check_allocator_env``) live in
``src/runtime/__init__.py``, which imports only ``os`` and ``subprocess``,
so those are tested for real.

Every auditor here has a NEGATIVE CONTROL beside it
(``test_auditor_*_can_fail``) that feeds it a synthetic source string
carrying the defect and asserts the auditor reports it.  Standing lesson 1
of ``wk_REL/README.md``: an instrument that has never been shown failing is
not evidence.  Four of the six void checks in the 2026-07-30 session were
cheerfully green.
"""
import ast
import os
import subprocess
import sys
import tempfile
import textwrap

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import runtime as _runtime            # noqa: E402  (os + subprocess only)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(relpath):
    with open(os.path.join(_SRC, relpath), encoding="utf-8") as fh:
        return fh.read()


def _tree(src, name="<synthetic>"):
    return ast.parse(src, name)


def _const_str(node):
    """String value of a literal node, else None.

    py3.7-compatible: that parser emits ``ast.Str`` for string literals,
    not ``ast.Constant`` (isinstance(a, ast.Constant) matches NOTHING
    there, silently voiding any auditor built on it).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if node.__class__.__name__ == "Str":            # py<3.8
        return node.s
    return None


def _call_names(tree):
    """``{callee-name: [lineno, ...]}`` over every ``ast.Call`` in ``tree``."""
    out = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.id if isinstance(f, ast.Name) else (
            f.attr if isinstance(f, ast.Attribute) else None)
        if name:
            out.setdefault(name, []).append(n.lineno)
    return out


def _imported_from(tree, module_suffix):
    """Names imported via ``from <...module_suffix> import a, b``."""
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module \
                and n.module.endswith(module_suffix):
            names.extend(a.name for a in n.names)
    return names


class _Env:
    """Set/clear env vars, always restore."""

    def __init__(self, **kw):
        self._want = kw
        self._old = {}

    def __enter__(self):
        for k, v in self._want.items():
            self._old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


# ===========================================================================
# R1 — gw_jax must get its mesh AND its warm-up from the runtime entry point
# ===========================================================================
#
# 2026-07-31 RATCHET (not a weakening): mesh + warm-up ownership moved one
# level DOWN, from ``common.collectives.prepare_mesh`` called by the driver
# to ``runtime.initialize_communicator_stack`` (gw_jax.py:33,45), which
# itself calls ``prepare_mesh`` — that lower link is asserted by
# ``test_entry_point_calls_prepare_mesh_in_the_service`` so the full chain
#     gw_jax -> initialize_communicator_stack -> prepare_mesh
# stays gated end-to-end.  The driver now makes exactly ONE
# ``initialize_communicator_stack()`` call and ZERO direct ``prepare_mesh``
# calls: a second Mesh object is a second set of communicators
# (gw_jax.py "Do NOT call prepare_mesh() again here").

def _audit_driver_warmup(src, name="<driver>"):
    """``[]`` when the driver delegates mesh+warm-up; complaints otherwise.

    Four things make a driver's warm-up incidental rather than owned:
    a hand-rolled mesh factorisation, a direct ``nccl_warmup`` import (the
    GPU half only — under ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` that
    leaves the MPI cliques to be created by whichever physics kernel fires
    first), the absence of exactly one ``initialize_communicator_stack``
    call, and any direct ``prepare_mesh`` call beside it.
    """
    tree = _tree(src, name)
    bad = []
    defs = [n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if "_build_mesh" in defs:
        bad.append("%s defines its own _build_mesh" % name)
    warms = [a for a in _imported_from(tree, "runtime")
             if a in ("nccl_warmup", "warm_mesh_cliques")]
    if warms:
        bad.append("%s imports a warm-up itself: %s" % (name, warms))
    calls = _call_names(tree)
    n_init = len(calls.get("initialize_communicator_stack", []))
    if n_init != 1:
        bad.append("%s makes %d initialize_communicator_stack calls, "
                   "expected 1" % (name, n_init))
    n_prep = len(calls.get("prepare_mesh", []))
    if n_prep != 0:
        bad.append("%s calls prepare_mesh directly %d time(s); the mesh "
                   "belongs to initialize_communicator_stack" % (name, n_prep))
    return bad


def test_gw_jax_gets_mesh_and_warmup_from_prepare_mesh():
    """R1.  ``gw/gw_jax.py`` called only ``runtime.nccl_warmup``, so under
    ``impl=mpi`` its MPI cliques were warmed INCIDENTALLY — by whichever
    physics kernel fired the first collective (``common/zeta_projection.py``
    x5, ``common/contract_bands.py``).  That holds only while those early
    programs stay small enough for XLA's SEQUENTIAL thunk executor; the
    parallel executor lands on the ``MPI_Is_thread_main`` refusal that
    killed the BSE TDA Lanczos (32 refusals at P=16, gate 7881216).

    Since the 2026-07-31 ratchet the owned startup call is
    ``runtime.initialize_communicator_stack`` (see block comment above).
    """
    assert _audit_driver_warmup(_read("gw/gw_jax.py"), "gw_jax") == []


def test_auditor_driver_warmup_can_fail():
    """NEGATIVE CONTROL: the pre-fix gw_jax shape must be reported."""
    prefix = textwrap.dedent("""
        import jax
        import numpy as np
        from jax.sharding import Mesh

        def _build_mesh():
            total = jax.process_count() * jax.local_device_count()
            return Mesh(np.array(jax.devices()).reshape(1, total), ['x', 'y'])

        def _setup_runtime(config, mesh_xy):
            from runtime import nccl_warmup
            nccl_warmup(mesh_xy)

        def main():
            mesh_xy = _build_mesh()
            _setup_runtime(None, mesh_xy)
    """)
    bad = _audit_driver_warmup(prefix, "prefix_gw_jax")
    assert len(bad) == 3, bad
    joined = " | ".join(bad)
    assert "_build_mesh" in joined
    assert "nccl_warmup" in joined
    assert "0 initialize_communicator_stack" in joined

    # A driver that calls the entry point but ALSO builds its own second
    # mesh via a direct prepare_mesh call must be reported too.
    second_mesh = textwrap.dedent("""
        from runtime import initialize_communicator_stack
        RUNTIME = initialize_communicator_stack()

        def main():
            from common.collectives import prepare_mesh
            mesh2 = prepare_mesh(axis_names=('x', 'y'))
    """)
    bad = _audit_driver_warmup(second_mesh, "second_mesh_gw_jax")
    assert len(bad) == 1 and "prepare_mesh directly" in bad[0], bad


def test_gw_jax_imports_prepare_mesh_from_the_service():
    """gw_jax takes its whole communicator stack from the runtime service
    (gw_jax.py:33) and no longer touches ``prepare_mesh`` itself — the
    prepare_mesh link now lives inside ``initialize_communicator_stack``
    and is asserted by the next test."""
    tree = _tree(_read("gw/gw_jax.py"))
    assert "initialize_communicator_stack" in _imported_from(tree, "runtime"), (
        "gw_jax must take initialize_communicator_stack from runtime, not "
        "re-implement or re-export it")
    assert "prepare_mesh" not in _imported_from(tree, "common.collectives"), (
        "gw_jax must NOT import prepare_mesh any more: a second "
        "prepare_mesh call is a second Mesh, i.e. a second set of "
        "communicators")


def test_entry_point_calls_prepare_mesh_in_the_service():
    """The lower link of the ratcheted chain: the runtime entry point
    itself imports ``prepare_mesh`` from ``common.collectives`` and calls
    it inside ``initialize_communicator_stack``."""
    tree = _tree(_read("runtime/__init__.py"), "runtime/__init__.py")
    assert "prepare_mesh" in _imported_from(tree, "common.collectives"), (
        "runtime must take prepare_mesh from common.collectives")
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
           and n.name == "initialize_communicator_stack"]
    assert len(fns) == 1, "expected exactly one initialize_communicator_stack"
    calls = _call_names(fns[0])
    assert calls.get("prepare_mesh"), (
        "initialize_communicator_stack must call prepare_mesh itself")


# ---------------------------------------------------------------------------
# R1 extension (2026-08-01): ALL SEVEN chain drivers own their startup.
#
# gw_jax adopted ``initialize_communicator_stack`` on 2026-07-31; the other
# six drivers still ran ``bootstrap()`` (or, in psp.get_dipole_mtxels, two
# hand-picked pieces of it) and got their mesh + clique warm-up ad hoc —
# htransform and the centroid path via interim direct ``prepare_mesh``
# routing (repo 24e4dc3), kin_ion/dipole via their own calls, the BSE pair
# via the bse_ring_comm factory alone.  Same audit, same budget for every
# driver: exactly ONE startup call, ZERO direct prepare_mesh calls, no
# hand-rolled mesh, no direct warm-up import.  This list is a ratchet —
# a new chain driver is added here, not exempted.
# ---------------------------------------------------------------------------

_STARTUP_DRIVERS = (
    "gw/gw_jax.py",
    "centroid/kmeans_cli.py",
    "psp/get_dipole_mtxels.py",
    "gw/kin_ion_io.py",
    "bandstructure/htransform.py",
    "bse/bse_jax.py",
    "bse/exciton_bands.py",
)


def test_every_chain_driver_owns_its_startup():
    """Each of the seven chain drivers makes exactly one
    ``initialize_communicator_stack()`` call and no mesh/warm-up call of
    its own.  The auditor is ``_audit_driver_warmup`` — already shown able
    to fail by ``test_auditor_driver_warmup_can_fail`` above, so this test
    inherits that negative control."""
    complaints = []
    for relpath in _STARTUP_DRIVERS:
        complaints.extend(_audit_driver_warmup(_read(relpath), relpath))
    assert complaints == [], complaints


def test_the_runtime_docstring_names_all_adopters():
    """The module docstring is where a new reader learns who makes the
    startup call; a list that stops at gw_jax sends the other six back to
    ``bootstrap()``.  Module-path spelling (``gw.gw_jax``)."""
    doc = ast.get_docstring(_tree(_read("runtime/__init__.py"), "runtime"))
    assert doc is not None
    for relpath in _STARTUP_DRIVERS:
        modpath = relpath[:-3].replace("/", ".")
        assert modpath in doc, (
            "runtime docstring no longer names adopter %s" % modpath)


# ===========================================================================
# R2 — bse_io must refuse a mesh this process owns no device on
# ===========================================================================

#: The three shard-aware readers whose ``make_array_from_process_local_data``
#: raised a bare ``StopIteration('')`` on a zero-addressable mesh
#: (``jax/_src/array.py:1017``; reproduced at P=2 in job 7882420, where rank 0
#: SUCCEEDS and rank 1 dies with an empty exception).
_BSE_SHARDED_READERS = ("_read_psi_mu_sharded", "_read_vq0_sharded",
                        "_read_wq_sharded")


def _audit_addressability_guard(src, name="<mod>"):
    """``[]`` when every process-local array placement is guarded.

    The guard belongs in ``_get_local_mesh_coords``, not at the three
    ``make_array_from_process_local_data`` call sites: that helper's very
    next statement is ``np.argwhere(devices_2d == d)[0]`` over a device the
    mesh does not contain, which raises an equally anonymous ``IndexError``
    BEFORE a guard placed lower down could speak.
    """
    tree = _tree(src, name)
    bad = []
    helper = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "_get_local_mesh_coords":
            helper = n
    if helper is None:
        return ["%s has no _get_local_mesh_coords" % name]
    if "_require_addressable" not in _call_names(helper):
        bad.append("_get_local_mesh_coords does not call _require_addressable")
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for reader in _BSE_SHARDED_READERS:
        if reader not in fns:
            bad.append("%s: reader %s is gone" % (name, reader))
            continue
        calls = _call_names(fns[reader])
        if "make_array_from_process_local_data" not in calls:
            continue                      # nothing to guard in this one
        if "_get_local_mesh_coords" not in calls:
            bad.append("%s places a process-local array without going "
                       "through the guarded helper" % reader)
    return bad


def test_bse_io_refuses_a_mesh_with_no_addressable_device():
    """R2.  Hardening, not a live bug: these readers are called with the run's
    full 2-D mesh, so every rank is addressable today."""
    assert _audit_addressability_guard(_read("bse/bse_loading.py"), "bse_loading") == []


def test_auditor_addressability_guard_can_fail():
    """NEGATIVE CONTROL: unguarded helper AND an unguarded reader."""
    prefix = textwrap.dedent("""
        def _get_local_mesh_coords(mesh_xy):
            devices_2d = np.asarray(mesh_xy.devices)
            return [], 1, 1

        def _read_psi_mu_sharded(dset, mesh_xy):
            local_coords, gx, gy = _get_local_mesh_coords(mesh_xy)
            return jax.make_array_from_process_local_data(s, a, shape)

        def _read_vq0_sharded(dset, mesh_xy):
            return jax.make_array_from_process_local_data(s, a, shape)

        def _read_wq_sharded(dset, mesh_xy):
            local_coords, gx, gy = _get_local_mesh_coords(mesh_xy)
            return jax.make_array_from_process_local_data(s, a, shape)
    """)
    bad = _audit_addressability_guard(prefix, "prefix_bse_io")
    assert len(bad) == 2, bad
    assert any("_require_addressable" in b for b in bad)
    assert any("_read_vq0_sharded" in b for b in bad)


def test_bse_io_guard_comes_from_the_service():
    src = _read("bse/bse_loading.py")
    assert "_require_addressable" in _imported_from(src and _tree(src),
                                                    "common.collectives"), (
        "bse_loading must use collectives._require_addressable, not a private copy")


# ===========================================================================
# R3 — no swallowed barrier in the FFI slab writer
# ===========================================================================

def _audit_swallowed_barriers(src, name="<mod>"):
    """Every ``try: <barrier>; except: pass`` in ``src``, by line.

    A barrier that silently does nothing is worse than no barrier: each use
    in the slab writer orders a WRITE (inode replacement before the Lustre
    stripe hints, prestripe before the collective open, the rank-0 attr
    rewrite before close), so swallowing it lets a rank read a file the
    writer has not finished changing — and reports rc=0.
    """
    tree = _tree(src, name)
    bad = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Try):
            continue
        swallows = any(len(h.body) == 1 and isinstance(h.body[0], ast.Pass)
                       for h in n.handlers)
        if not swallows:
            continue
        called = _call_names(ast.Module(body=n.body, type_ignores=[]))
        for fn in ("sync_global_devices", "barrier", "_barrier"):
            if fn in called:
                bad.append("%s:%d swallows %s" % (name, n.lineno, fn))
    return bad


def test_slab_io_ffi_has_no_swallowed_barrier():
    """R3.  The local ``_barrier`` was a verbatim ``except: pass`` seven lines
    below the file's own ``from common.collectives import ...``."""
    assert _audit_swallowed_barriers(_read("file_io/_slab_io_ffi.py"),
                                     "_slab_io_ffi") == []


def test_slab_io_ffi_barrier_is_the_service_one():
    names = _imported_from(_tree(_read("file_io/_slab_io_ffi.py")),
                           "common.collectives")
    assert "barrier" in names, names


def test_auditor_swallowed_barriers_can_fail():
    """NEGATIVE CONTROL: both the pre-fix helper and the pre-fix close path."""
    prefix = textwrap.dedent("""
        def _barrier(tag):
            try:
                multihost_utils.sync_global_devices(tag)
            except Exception:
                pass

        def close(self):
            try:
                multihost_utils.sync_global_devices("close_attrs")
            except Exception:
                pass
    """)
    bad = _audit_swallowed_barriers(prefix, "prefix_slab")
    assert len(bad) == 2, bad


def test_auditor_swallowed_barriers_does_not_fire_on_a_real_handler():
    """The auditor must not flag a try/except that actually HANDLES."""
    ok = textwrap.dedent("""
        def f(tag):
            try:
                barrier(tag)
            except Exception as exc:
                raise RuntimeError(tag) from exc
    """)
    assert _audit_swallowed_barriers(ok, "ok") == []


# ===========================================================================
# R4 — a truncated zeta must not be stamped complete
# ===========================================================================

def _audit_truncation_guard(src, name, stamp_fn):
    """``[]`` when ``stamp_fn`` is reached only past a truncation check."""
    tree = _tree(src, name)
    calls = _call_names(tree)
    stamps = calls.get(stamp_fn, [])
    guards = calls.get("active_zeta_truncating_knobs", [])
    if not stamps:
        return ["%s never calls %s" % (name, stamp_fn)]
    if not guards:
        return ["%s calls %s with no truncation guard" % (name, stamp_fn)]
    if min(guards) >= min(stamps):
        return ["%s evaluates the truncation guard AFTER %s" % (name, stamp_fn)]
    return []


def test_isdf_fitting_does_not_mark_a_truncated_zeta_done():
    """R4.  ``LORRAX_MAX_RCHUNKS=N`` breaks the r-chunk loop early and the
    writer downstream still stamped ``zeta_is_done=True`` — the file lying
    about itself.  ``gw_init``'s existing guard blocks REUSE; this one stops
    the claim being written."""
    assert _audit_truncation_guard(
        _read("gw/isdf_fitting.py"), "isdf_fitting", "mark_zeta_done") == []


def test_gw_init_still_guards_the_provenance_stamp():
    """The sibling guard must not have regressed while R4 landed."""
    assert _audit_truncation_guard(
        _read("gw/gw_init.py"), "gw_init", "stamp_fit_provenance") == []


def test_auditor_truncation_guard_can_fail():
    """NEGATIVE CONTROL: unguarded, and guarded-too-late."""
    unguarded = "def f(p):\n    mark_zeta_done(p)\n"
    assert _audit_truncation_guard(unguarded, "u", "mark_zeta_done")
    too_late = ("def f(p):\n    mark_zeta_done(p)\n"
                "    t = active_zeta_truncating_knobs()\n")
    bad = _audit_truncation_guard(too_late, "l", "mark_zeta_done")
    assert bad and "AFTER" in bad[0], bad
    missing = "def f(p):\n    pass\n"
    assert _audit_truncation_guard(missing, "m", "mark_zeta_done")


# ===========================================================================
# R5 — LORRAX_FORCE_FULL_BZ has ONE grammar across all five sites
# ===========================================================================

_FFBZ = "LORRAX_FORCE_FULL_BZ"

#: The modules that read it.  Enumerated rather than discovered so a NEW
#: reader appearing with a hand-rolled parse fails this test.  gw_init
#: went 3 → 2 on 2026-08-04: the transverse ζ IBZ gate is now DERIVED
#: from the charge one (``_write_ibz_only_transverse = cfg.bispinor and
#: _write_ibz_only_charge``) instead of re-reading the environment, so
#: the ζ_T provenance stamp and the ζ_T fit call cannot disagree about
#: what was requested.  Total is 4.
_FFBZ_SITES = {"gw/gw_init.py": 2, "gw/screening.py": 1, "gw/v_q_g_flat.py": 1}


def _audit_env_grammar(src, name, knob):
    """``(n_env_bool, [raw-read complaints])`` for ``knob`` in ``src``."""
    tree = _tree(src, name)
    good = 0
    bad = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        fname = f.id if isinstance(f, ast.Name) else (
            f.attr if isinstance(f, ast.Attribute) else None)
        args = [a for a in n.args if _const_str(a) == knob]
        if not args:
            continue
        if fname == "env_bool":
            good += 1
        elif fname in ("get", "getenv"):
            bad.append("%s:%d reads %s with a raw os.environ %s()"
                       % (name, n.lineno, knob, fname))
        else:
            bad.append("%s:%d reads %s via %s()" % (name, n.lineno, knob, fname))
    return good, bad


def test_force_full_bz_has_one_grammar_at_every_site():
    """R5.  All five were ``bool(int(os.environ.get(...)))``, which accepts
    decimal digits ONLY: ``=true``/``=on``/``=yes`` raised a bare
    ``invalid literal for int()`` from inside the ISDF / V_q / W paths, and
    ``=2`` silently meant "on".  A previous agent deliberately fixed NONE of
    them, on the grounds that a split grammar is worse than one wrong one —
    so they had to move together."""
    total, complaints = 0, []
    for relpath, expect in _FFBZ_SITES.items():
        good, bad = _audit_env_grammar(_read(relpath), relpath, _FFBZ)
        complaints.extend(bad)
        assert good == expect, (
            "%s: %d env_bool reads of %s, expected %d"
            % (relpath, good, _FFBZ, expect))
        total += good
    assert complaints == [], complaints
    assert total == 4, total


def test_no_module_anywhere_reads_force_full_bz_by_hand():
    """A sixth reader with its own parse re-splits the grammar."""
    complaints = []
    for root, _dirs, files in os.walk(_SRC):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, _SRC)
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            if _FFBZ not in src:
                continue
            try:
                _, bad = _audit_env_grammar(src, rel, _FFBZ)
            except SyntaxError:
                continue
            complaints.extend(bad)
    assert complaints == [], complaints


def test_auditor_env_grammar_can_fail():
    """NEGATIVE CONTROL: the exact pre-fix expression, both spellings."""
    prefix = ('import os\n'
              '_a = bool(int(os.environ.get("LORRAX_FORCE_FULL_BZ", "0")))\n'
              '_b = bool(int(os.getenv("LORRAX_FORCE_FULL_BZ", "0")))\n')
    good, bad = _audit_env_grammar(prefix, "prefix", _FFBZ)
    assert good == 0 and len(bad) == 2, (good, bad)


# ===========================================================================
# R6 — blank is UNSET in runtime, as everywhere else  (behavioural)
# ===========================================================================

def test_runtime_env_falsy_reads_blank_as_unset():
    """R6.  ``""`` used to be in ``_FALSY_TOKENS``, so
    ``export LORRAX_MALLOC_TUNE=`` DISABLED a default-on knob."""
    assert "" not in _runtime._FALSY_TOKENS
    for blank in ("", " ", "\t", "\n"):
        with _Env(LORRAX_MALLOC_TUNE=blank):
            assert _runtime._env_falsy("LORRAX_MALLOC_TUNE") is False
            assert _runtime._env_falsy("LORRAX_MALLOC_TUNE", "0") is True


def test_runtime_env_falsy_still_honours_real_falsy_tokens():
    """The half of the parser the fix must NOT have broken."""
    for tok in ("0", "off", "OFF", "false", " no ", "No"):
        with _Env(LORRAX_MALLOC_TUNE=tok):
            assert _runtime._env_falsy("LORRAX_MALLOC_TUNE") is True, tok
    for tok in ("1", "on", "yes", "true"):
        with _Env(LORRAX_MALLOC_TUNE=tok):
            assert _runtime._env_falsy("LORRAX_MALLOC_TUNE") is False, tok
    with _Env(LORRAX_MALLOC_TUNE=None):
        assert _runtime._env_falsy("LORRAX_MALLOC_TUNE") is False


# ===========================================================================
# R7 — an allocator typo must name itself  (behavioural)
# ===========================================================================

_ALLOC = "XLA_PYTHON_CLIENT_ALLOCATOR"


def test_allocator_env_accepts_exactly_what_jaxlib_accepts():
    """R7.  jaxlib compares ``os.getenv(...).lower()`` against
    ``('default', 'platform', 'bfc', 'cuda_async')`` — after ``.lower()`` and
    WITHOUT stripping."""
    assert _runtime.ALLOCATOR_SPELLINGS == (
        "default", "platform", "bfc", "cuda_async")
    for tok in ("default", "platform", "bfc", "cuda_async",
                "BFC", "Cuda_Async"):
        with _Env(**{_ALLOC: tok}):
            _runtime._check_allocator_env()          # must not raise
            assert os.environ[_ALLOC] == tok         # and must not rewrite


def test_allocator_env_refuses_a_typo_and_names_it():
    """The measured failure this replaces is
    ``Backend 'cuda' is not in the list of known backends`` — which names
    neither the variable nor the value, and reads as missing hardware."""
    for tok in ("platfrom", "cudaasync", "cuda-async", "bfc2", " bfc", "bfc "):
        with _Env(**{_ALLOC: tok}):
            try:
                _runtime._check_allocator_env()
            except ValueError as exc:
                msg = str(exc)
                assert _ALLOC in msg, msg
                assert repr(tok) in msg, msg
                assert "cuda_async" in msg, msg
            else:
                raise AssertionError("%r was accepted" % tok)


def test_allocator_env_removes_a_blank_value():
    """jaxlib compares ``''`` against its tuple and REJECTS it, so blank
    cannot merely be waved through: LORRAX reads blank as unset, and unset
    is what jaxlib needs in order to apply its own default."""
    for blank in ("", " "):
        with _Env(**{_ALLOC: blank}):
            _runtime._check_allocator_env(print_fn=lambda *a, **k: None)
            assert _ALLOC not in os.environ, blank
    with _Env(**{_ALLOC: None}):
        _runtime._check_allocator_env()
        assert _ALLOC not in os.environ


def test_allocator_check_runs_inside_set_default_env():
    """A validator nobody calls is not a validator.  ``set_default_env`` is
    the slot: it is the last thing before the first ``jax.devices()``."""
    tree = _tree(_read("runtime/__init__.py"), "runtime")
    fn = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "set_default_env":
            fn = n
    assert fn is not None
    assert "_check_allocator_env" in _call_names(fn)
    with _Env(**{_ALLOC: "platfrom", "JAX_PLATFORMS": None}):
        try:
            _runtime.set_default_env(platform="cpu")
        except ValueError as exc:
            assert _ALLOC in str(exc)
        else:
            raise AssertionError("set_default_env accepted a bad allocator")


def test_allocator_spellings_match_the_installed_jaxlib_if_present():
    """Read jaxlib's own tuple when a copy is on disk, rather than trusting
    this module's transcription of it.  Skipped, loudly, when absent —
    a check that cannot run must not report as one that passed."""
    # Bounded lookup only: importing jaxlib needs a compute node, and a
    # recursive walk of $WORK on Lustre takes minutes.  ``LORRAX_JAXLIB_DIR``
    # lets a compute-node harness point straight at it.
    cands = []
    hint = os.environ.get("LORRAX_JAXLIB_DIR")
    if hint:
        cands.append(os.path.join(hint, "xla_client.py"))
    for base in sys.path:
        if base:
            cands.append(os.path.join(base, "jaxlib", "xla_client.py"))
    hits = [c for c in cands if os.path.isfile(c)]
    if not hits:
        print("      [skip] no jaxlib/xla_client.py importable to cross-check "
              "(set LORRAX_JAXLIB_DIR to force it)")
        return
    with open(hits[0], encoding="utf-8") as fh:
        text = fh.read()
    marker = "if allocator not in ("
    assert marker in text, hits[0]
    frag = text.split(marker, 1)[1].split(")", 1)[0]
    found = tuple(p.strip().strip("'\"") for p in frag.split(",") if p.strip())
    assert found == _runtime.ALLOCATOR_SPELLINGS, (found, hits[0])


# ===========================================================================
# R8 / R9 — the library->driver edge is gone
# ===========================================================================

#: Generic k-partition plumbing that lives in ``common.collectives`` and was
#: only ever re-exported by the kin_ion CLI.
_SERVICE_NAMES = ("sweep_local_k", "replicate_to_mesh", "resolve_mesh",
                  "psum_replicate", "gather_indexed_blocks",
                  "_psum_replicate", "_gather_indexed_blocks")


def _audit_driver_imports(src, name):
    """Complaints for every collectives name imported from the kin_ion CLI."""
    bad = []
    for n in ast.walk(_tree(src, name)):
        if not isinstance(n, ast.ImportFrom) or not n.module:
            continue
        if not n.module.endswith("kin_ion_io"):
            continue
        for a in n.names:
            if a.name in _SERVICE_NAMES:
                bad.append("%s:%d imports %s from the kin_ion DRIVER"
                           % (name, n.lineno, a.name))
    return bad


def test_no_module_imports_collectives_plumbing_from_the_kin_ion_driver():
    """R8 + R9.  ``gw/sigma_dispatch.py`` (a library) reached into a CLI for
    ``replicate_to_mesh``; two test modules did the same.  With all of them
    moved, the forwarding-shim block at the bottom of ``gw/kin_ion_io.py``
    has no consumers left and is deletable — filed as a request, since that
    file belongs to another workstream."""
    complaints = []
    roots = [(_SRC, ""), (os.path.join(_REPO, "tests"), "tests")]
    for root, _tag in roots:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                if os.path.abspath(path) == os.path.abspath(__file__):
                    continue
                if os.path.basename(path) == "kin_ion_io.py":
                    continue          # the shim block itself
                with open(path, encoding="utf-8") as fh:
                    src = fh.read()
                if "kin_ion_io" not in src:
                    continue
                try:
                    complaints.extend(_audit_driver_imports(
                        src, os.path.relpath(path, _REPO)))
                except SyntaxError:
                    continue
    assert complaints == [], complaints


def test_sigma_dispatch_still_takes_the_physics_from_the_driver():
    """Only real physics crosses the lazy driver boundary.

    ``compute_hartree_matrix`` stays a lazy driver import — that laziness is
    why the ISDF route does not pull in the psp stack.  The live G-space path
    now returns its native sharded array, so it needs neither the former
    ``replicate_to_mesh`` plumbing import nor a replacement gather helper.
    """
    tree = _tree(_read("gw/sigma_dispatch.py"), "sigma_dispatch")
    assert "compute_hartree_matrix" in _imported_from(tree, "kin_ion_io")
    assert "replicate_to_mesh" not in _imported_from(
        tree, "common.collectives")


def test_auditor_driver_imports_can_fail():
    """NEGATIVE CONTROL: the pre-fix sigma_dispatch import line."""
    prefix = ("def f():\n"
              "    from gw.kin_ion_io import compute_hartree_matrix, "
              "replicate_to_mesh\n")
    bad = _audit_driver_imports(prefix, "prefix_sigma")
    assert len(bad) == 1 and "replicate_to_mesh" in bad[0], bad


# ===========================================================================
# R10 — the dipole provenance checker is actually called
# ===========================================================================

def test_head_correction_checks_dipole_provenance_and_coverage():
    """R10.  ``psp.get_dipole_mtxels`` stamps ``prov_*`` attrs and ships
    ``check_dipole_provenance`` to read them back; nothing called it.  The
    two checks answer different questions — coverage asks "is the file big
    enough", provenance asks "is it the right file at all" — and a dipole.h5
    from a different WFN has exactly the right shape."""
    src = _read("gw/head_correction.py")
    calls = _call_names(_tree(src, "head_correction"))
    assert "_check_dipole_coverage" in calls
    assert "check_dipole_provenance" in calls, (
        "head_correction still never calls the provenance checker")
    assert "_dipole_window_from_params" in calls


def test_dipole_provenance_checker_still_exists_with_that_signature():
    """The caller passes ``wfn``/``nval``/``ncond``/``nband`` by keyword."""
    tree = _tree(_read("psp/dipole_store.py"), "dipole_store")
    fn = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "check_dipole_provenance":
            fn = n
    assert fn is not None, "the checker R10 wires up has vanished"
    kwonly = [a.arg for a in fn.args.kwonlyargs]
    for want in ("wfn", "nval", "ncond", "nband", "print_fn"):
        assert want in kwonly, (want, kwonly)


def test_the_head_reader_consumes_the_window_rather_than_mirroring_defaults():
    """INVERTED 2026-08-22, and the inversion IS the fix.

    This cell used to require that ``gw/head_correction.py`` contain the
    writer's own default expressions — ``params.get("nval", 5)`` and friends
    — on the reasoning that the reader "has to derive them the same way or
    the check is noise".  The writer resolves those defaults against the
    DECK.  The reader was handed ``config.head``, a six-key dict with no
    band window in it, so the shared spelling produced a shared NUMBER only
    on a 5/5 deck: the mirror was of the fallback, not of the derivation.
    Measured on the MoS2 production deck (JID 57269074) — a dipole.h5 built
    from the same WFN and the same deck reported ``file=26/26/600`` against
    an invented ``run=5/5/610``, and under ``LORRAX_SANITY=strict`` that
    false warning refuses a correct file.

    So the reader must NOT carry a default at all.  It consumes the run's
    RESOLVED window and refuses when one is absent; the writer keeps its
    deck defaults, because resolving a deck is its job.
    """
    src_r = _read("gw/head_correction.py")
    src_w = _read("psp/get_dipole_mtxels.py")
    for frag in ('params.get("nval", 5)', 'params.get("ncond", 5)'):
        assert frag in src_w, "writer changed: %s" % frag
        assert frag not in src_r, (
            "the head reader re-grew the writer's DEFAULT %s.  A provenance "
            "reference the checker invents cannot fail for the reason it "
            "claims: it accuses every deck whose window is not 5/5 and "
            "vouches for nothing." % frag)
    # and it refuses instead — the behaviour is pinned in
    # tests/test_head_correction.py and tests/test_psp_padded_gvectors.py;
    # here we only require that the refusal exists in the file at all.
    assert "band window is missing" in src_r, (
        "the reader dropped the refusal it replaced the defaults with")


# ===========================================================================
# R11 — REFUTED.  A package __init__ runs BEFORE its submodule's body.
# ===========================================================================

def test_package_init_runs_before_submodule_body_under_dash_m():
    """R11 as filed claimed ``gw/__init__.py:6``'s ``setdefault`` is INERT
    because ``gw_jax.py:41`` imports jax first.  It is not: ``python -m
    gw.gw_jax`` imports the PACKAGE before the module, so line 6 runs first.
    Measured here rather than argued, because the request was filed on the
    opposite belief.

    (The line is still redundant for the drivers — ``gw_jax`` and
    ``kin_ion_io`` both call ``runtime.bootstrap()`` before their own
    ``import jax`` — which is what the corrected comment now says.)
    """
    with tempfile.TemporaryDirectory() as tmp:
        pkg = os.path.join(tmp, "pkgprobe")
        os.makedirs(pkg)
        with open(os.path.join(pkg, "__init__.py"), "w") as fh:
            fh.write('import os\n'
                     'print("INIT:%r" % os.environ.get("PROBE_X64"))\n'
                     'os.environ.setdefault("PROBE_X64", "1")\n')
        with open(os.path.join(pkg, "mod.py"), "w") as fh:
            fh.write('import os\n'
                     'print("MOD:%r" % os.environ.get("PROBE_X64"))\n')
        env = dict(os.environ)
        env.pop("PROBE_X64", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        out = subprocess.run([sys.executable, "-m", "pkgprobe.mod"],
                             cwd=tmp, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT).stdout.decode()
    assert "INIT:None" in out, out
    assert "MOD:'1'" in out, out
    assert out.index("INIT:") < out.index("MOD:"), out


def test_gw_package_init_no_longer_claims_to_be_the_guarantee():
    """The package stays JAX-free until a lazy compatibility export is used."""
    src = _read("gw/__init__.py")
    assert 'os.environ.setdefault("JAX_ENABLE_X64", "1")' in src
    assert "runtime.set_default_env" in src, (
        "gw/__init__ must point at the canonical setter")
    assert "def __getattr__" in src
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(_REPO, "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys, gw; "
         "print(int('jax' in sys.modules), "
         "int('gw.gw_config' in sys.modules))"],
        cwd=_REPO, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "0 0", probe.stdout


# ===========================================================================

def _main():
    fns = [(n, o) for n, o in sorted(globals().items())
           if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in fns:
        try:
            fn()
        except Exception as exc:            # noqa: BLE001 — this IS the tally
            failed.append((name, exc))
            print("FAIL %s\n     %s: %s" % (name, type(exc).__name__, exc))
        else:
            print("ok   %s" % name)
    print("\n%d/%d passed, %d failed"
          % (len(fns) - len(failed), len(fns), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
