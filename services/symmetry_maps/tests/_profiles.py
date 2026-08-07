"""Machine profiles for the symmetry_maps skip-honesty gate.

WHY THIS FILE EXISTS AND NOT A ROW IN ``lxkit.testing.PROFILES``.  lxkit is
read-only to the wave-1 services, and its table declares two rows —
``perlmutter`` (four FFI backends distrib_la owns) and ``unknown`` (asserts
nothing, deliberately).  Neither describes what THIS service promises on
the dev box: symmetry_maps has no ``.so``, no MPI and no GPU claim, and its
MUST rows are about FIXTURES and DEVICES, which are properties of the
checkout and the invocation.  So the profile is built here, on top of
lxkit's vocabulary (:class:`MachineProfile`, :class:`AllowedSkip`), and
lxkit's ``PROFILES`` is consulted first so a Perlmutter leg still gets the
Perlmutter row.

WHAT THE MUST ROWS PROMISE.  Four things, all checkable, and none of them
about an ``.so`` — this service has no FFI layer, so its ``MustRow``s name
CAPABILITIES instead of backends and :func:`probe_capability` below is the
probe lxkit's positive half is handed.  lxkit owns the policy, the service
owns its table and its probe, exactly as ``distrib_la.loader.probe_target``
does for the linalg backends:

* the four in-tree WFN decks are present, so every deck-table cell RUNS
  (nothing here is gated on ``/pscratch``);
* ``cohsex_debug/WFNsmall.h5`` specifically — it is also what
  ``tests/test_density_symmetry_check.py:33`` gates its four fixture cells
  on, the ONLY in-tree TRS-active measurement of the density verdict, and
  the deck the I5 tripwire needs;
* ``h5py`` imports, so the L-a+ tier is not silently a no-op;
* four host devices are FORCEABLE — ``--xla_force_host_platform_device_count
  =4`` really does give four CPU devices on this machine, measured in a
  child process rather than assumed.

THE DEVICE ROW IS ABOUT THE CAPABILITY, NOT ABOUT THIS PROCESS.  The
emulated tier RUNS whenever the process actually has four devices, which is
the service-only invocation, where this suite's conftest sets ``XLA_FLAGS``
before the first jax import.  In the FULL monorepo run ``tests/`` imports
jax during collection first, the flag is inert by design, and the universal
``needs >= 4 devices`` row covers the resulting skips.  Promising "four
devices always" would be promising something about how the leg was
LAUNCHED, which no profile can know; promising that the forcing WORKS is a
property of the machine, which is exactly what a machine profile is for.

Detection: ``LX_MACHINE``/``NERSC_HOST`` first (lxkit's own rule, so a leg
can say what it is and the red twins can name a machine that does not
exist), then the kernel string.  WSL2 puts ``microsoft`` in
``/proc/version``; ``WSL_DISTRO_NAME`` is the belt to that suspenders.
Anything else is ``unknown`` and asserts nothing — the honest answer for a
machine nobody has measured.
"""

from __future__ import annotations

import os
import subprocess
import sys

from lxkit.testing import (PROFILES, _UNIVERSAL_SKIPS, AllowedSkip,
                           MachineProfile, MustRow, detect_machine,
                           machine_profile)

#: Skips THIS service may emit on a machine that declares a profile, on top
#: of the universal rows lxkit ships.  Every row names the leg that runs the
#: cell instead; a skip nobody can point at a covering run for is evaporated
#: coverage.
_ALLOWED = (
    # The parity arm re-derives the deck tables through the PRODUCTION
    # loader.  A standalone install of this service has no lorrax to import
    # ``file_io`` from, and that is the whole point of the quarantine.
    AllowedSkip("deck_tables", "no lorrax file_io",
                "the monorepo run (`pytest services/symmetry_maps/tests` "
                "from the checkout root), where file_io imports and the "
                "WfnLoader parity arm re-derives every table"),
    # h5py is not a declared dependency of symmetry_maps — the package
    # answers every table question from arrays.  It is a dependency of the
    # TEST that reads a deck header.
    AllowedSkip("", "h5py is not importable",
                "any leg with h5py; the pure-array L-a tier asserts the "
                "same star algebra with no file at all"),
    # A STRICT XFAIL REPORTS AS A SKIP, so the gate rules on it and must be
    # told.  Same shape as distrib_la's SLATE SIGSEGV row: a CARRIED
    # upstream defect, named, with no covering leg claimed — because there
    # is none.  MEASURED 2026-08-07 on an EMULATED four-device mesh: jax
    # 0.9.1 / XLA CPU loses NaN in a PARTITIONED max reduction (-inf, the
    # reduce identity, comes back instead), so `KStarMap.spread_rel` cannot
    # refuse a poisoned SHARDED operand there and
    # `sc_iteration._check_kstar_spread`'s `not (spread <= tol)` passes it.
    # A REAL `srun -n 4` mesh PROPAGATES the NaN (measured the same day,
    # jax 0.7.0, at 2x2 and 4x1), so topology-vs-jax-version is unresolved.
    # The cell's docstring carries both legs and the full table.
    # The wsl-profile detection cell asserts is_wsl() -> the wsl row; on
    # every other machine that predicate is FALSE BY CONSTRUCTION and the
    # cell skips.  First tripped on the real Perlmutter leg 2026-08-07 —
    # the gate did exactly its job: a skip is legal only once this row
    # names the leg that runs the cell.
    AllowedSkip("the_wsl_profile_is_the_one_this_dev_box_selects",
                "not a WSL kernel",
                "the WSL dev-box leg (every local run of this suite), where "
                "is_wsl() is true and the row's four MUST cells are "
                "asserted; the machine actually running this session has "
                "its OWN row asserted by "
                "test_the_machine_profile_asserts_something"),
    AllowedSkip("test_a_nan_survives_the_sharded_reduction", "",
                "nothing — this is a CARRIED upstream defect (jax 0.9.1 / "
                "XLA CPU partitioned max discards NaN), not covered "
                "coverage.  The strict xfail IS the artifact: it turns RED "
                "the day the reduction propagates, and the owner report "
                "carries the P>1 consequence for the k-star spread gate"),
)

#: What a machine running this suite PROMISES.  ``target`` is a capability
#: string :func:`probe_capability` understands, not an FFI symbol: the
#: vocabulary belongs to the service, and this service's is fixtures and
#: devices.  Shared by every declared profile — these are properties of a
#: CHECKOUT plus a CPU, so a machine that has the tree has them, and one
#: that does not should say so loudly rather than skip four tiers.
_MUST = (
    MustRow("decks", "symmetry_maps", "decks:all-four", "in-tree"),
    # Named separately from ``decks`` because this one file also gates
    # tests/test_density_symmetry_check.py's four fixture cells (:33) —
    # the crown-jewel MEASUREMENT of the TRS verdict — and the §8.1 I5
    # tripwire.  Losing it costs two suites, not one.
    MustRow("density_fixture", "symmetry_maps",
            "file:cohsex_debug/WFNsmall.h5", "in-tree"),
    MustRow("h5py", "symmetry_maps", "module:h5py", "any"),
    MustRow("emulated_devices", "symmetry_maps", "devices:4-forceable", "cpu"),
)

#: The dev box.  ``min_collected`` is a FLOOR, not a count: it fails the
#: 38-byte-junitxml shape (a session that collected nothing skips nothing
#: and fails nothing) without turning every added cell into an edit here.
#: Set just under the L-a tier's size so a collection error that silently
#: dropped a module trips it.
#: ``_UNIVERSAL_SKIPS`` IS PART OF THIS ROW, not an optional extra.  lxkit
#: builds ``_PERLMUTTER`` as ``_UNIVERSAL_SKIPS + (site rows)``; a profile
#: written here as ``_ALLOWED`` alone inherits none of them, and the FIRST
#: consequence is that the L-b tier's own skips fail the gate.  MEASURED
#: 2026-08-07 while writing the skip-honesty cells: in the full monorepo
#: run jax is imported during collection, this suite's XLA flag is inert by
#: design, the eight emulated-mesh cells skip with ``needs >= 4 devices on
#: platform 'cpu', have 1`` — the exact case lxkit's universal row exists to
#: cover, naming the service-only leg — and the session died at
#: ``sessionfinish`` with "does not allow 1 of the N skip(s)".  A gate that
#: fires on the invocation it was designed for is a gate that gets
#: disarmed, so the universal rows are here rather than rediscovered.
_WSL = MachineProfile(
    machine="wsl",
    must=_MUST,                   # no FFI backends: this service has none
    allowed_skips=_UNIVERSAL_SKIPS + _ALLOWED,
    min_collected=40,
    asserts_nothing=False,
)


def _four_devices_are_forceable() -> tuple[bool, str]:
    """Does ``--xla_force_host_platform_device_count=4`` actually work here?

    MEASURED, in a child, because the answer in THIS process is a property
    of the invocation: the full monorepo run has already imported jax by
    the time our conftest runs and the flag is inert there by design.  When
    this process does have four devices — the service-only leg — that IS
    the measurement and the child is skipped; otherwise one subprocess
    (~1.3 s, once per session) answers it.
    """
    try:
        import jax
        if jax.device_count() >= 4:
            return True, ""
    except Exception as exc:                                   # noqa: BLE001
        return False, f"jax is not importable here: {type(exc).__name__}: {exc}"
    env = dict(os.environ)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + " --xla_force_host_platform_device_count=4").strip()
    env["JAX_PLATFORMS"] = "cpu"
    proc = subprocess.run(
        [sys.executable, "-c",
         "import jax; print('DEVICES', jax.device_count())"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = proc.stdout.decode()
    got = next((int(ln.split()[1]) for ln in out.splitlines()
                if ln.startswith("DEVICES")), -1)
    if got >= 4:
        return True, ""
    return False, (
        f"a child with XLA_FLAGS=--xla_force_host_platform_device_count=4 "
        f"reported {got} cpu device(s), not 4 (rc={proc.returncode}).  The "
        f"L-b tier cannot run anywhere on this machine, so its skips are "
        f"lost coverage rather than a property of the invocation.\n"
        f"--- stdout ---\n{out}\n--- stderr ---\n{proc.stderr.decode()}")


def probe_capability(target: str, platform: str) -> tuple[bool, str]:
    """THE PROBE.  ``(ok, reason)`` for one :data:`_MUST` target.

    Same contract as ``distrib_la.loader.probe_target`` so lxkit's
    :func:`~lxkit.testing.assert_must_probe` can be handed it unchanged.
    The reason is quoted VERBATIM into the failure, so each rung says what
    to DO: a missing blob is a checkout problem, a missing h5py is an
    environment problem, and a machine that cannot force four host devices
    is neither — it is a machine this tier has never run on.
    """
    import importlib.util

    from _deck_stub import DECK_WFN, deck_available, deck_path

    if target == "decks:all-four":
        missing = [d for d in sorted(DECK_WFN) if not deck_available(d)]
        if missing:
            return False, (
                f"{len(missing)} of 4 in-tree WFN decks are absent: "
                f"{missing}.  They are committed fixtures, not generated "
                f"data — a checkout without them was made without the "
                f"fixture blobs, and the deck-table tier (including the "
                f"bit-equality acceptance gate) measures nothing.")
        return True, ""
    if target.startswith("file:"):
        deck, name = target[len("file:"):].split("/", 1)
        path = deck_path(deck, name)
        if not os.path.isfile(path):
            return False, (
                f"{path} is absent.  It gates this suite's I5 tripwire AND "
                f"tests/test_density_symmetry_check.py's four fixture cells "
                f"(:33) — the only in-tree MEASUREMENT of the TRS verdict.")
        return True, ""
    if target.startswith("module:"):
        name = target[len("module:"):]
        if importlib.util.find_spec(name) is None:
            return False, (
                f"{name} is not importable.  It is not a dependency of the "
                f"symmetry_maps PACKAGE (which answers every table question "
                f"from arrays) but it is one of this SUITE: without it the "
                f"whole L-a+ deck tier skips.")
        return True, ""
    if target == "devices:4-forceable":
        return _four_devices_are_forceable()
    return False, (
        f"{target!r} is not a capability symmetry_maps knows how to probe; "
        f"the vocabulary is decks:all-four, file:<deck>/<name>, "
        f"module:<name>, devices:4-forceable")


def is_wsl() -> bool:
    """True when this kernel is WSL.  Cheap, and read once per call."""
    if (os.environ.get("WSL_DISTRO_NAME") or "").strip():
        return True
    try:
        with open("/proc/version", "r") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def profile_for_this_machine(machine: str | None = None) -> MachineProfile:
    """The profile in force.  lxkit's table first, then ``wsl``.

    ``machine`` is for the red twins, which have to be able to name a
    machine this process is not running on.

    A machine lxkit already describes (today: ``perlmutter``) keeps its own
    row and GAINS this service's MUST rows — decks and devices are promises
    of any machine with the checkout, and lxkit is read-only this wave so
    the rows cannot be added there.  ``must_for('distrib_la')`` is
    unaffected: that service filters by name, and this function is not the
    one it calls.
    """
    name = (machine or detect_machine()).strip().lower()
    if name in PROFILES and name != "unknown":
        base = PROFILES[name]
        return base._replace(must=tuple(base.must) + _MUST)
    if name == "wsl" or (machine is None and name == "unknown" and is_wsl()):
        return _WSL
    return machine_profile(name)


def allowed_skips_for(profile: MachineProfile):
    """Extra allowlist rows for ``profile``.

    Perlmutter gets them too: the WfnLoader parity arm and the h5py row are
    properties of the INVOCATION (standalone install / no h5py), not of the
    machine, and a row that only existed on the dev box would make the
    Perlmutter leg fail for a reason that is not about Perlmutter.
    """
    if profile.asserts_nothing:
        return ()
    return _ALLOWED
