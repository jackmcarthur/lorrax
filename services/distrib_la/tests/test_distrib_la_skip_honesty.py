"""The skip-honesty gate, distrib_la's half.

The charter: each machine declares an expected-backend profile, and one
gate per service asserts that reality matches it.  An unexpected skip is a
FAILURE, because a skip reads as "not applicable on this machine" and this
tree has the receipt for what that costs — 2026-08-06, nineteen
ScaLAPACK/SLATE contract cells reported as "19 skipped" beside "0 failed",
suite green, coverage gone.

The mechanism is :mod:`lxkit.testing` (profiles, both halves, the red
twins for each).  What lives HERE is the part only this service can
supply: the probe.  ``lxkit`` owns the policy, ``distrib_la.loader`` owns
the table, and the row that belongs to NEITHER — phdf5, which is slab_io's
library — is reported as an explicit GAP rather than passed over, because
a known hole and a forgotten one look identical from the outside.

MEASURED, Perlmutter 2026-08-07 (`lx run -N 1 -G 1`, BUILD_NOTES pins,
inside the lorrax_J070 Shifter image)::

    NERSC_HOST='perlmutter'   detect_machine() -> perlmutter
    profile perlmutter        asserts_nothing=False
    scalapack   lorrax_scalapack_eigh    cpu   -> available
    slate       lorrax_slate_potrf       cpu   -> available
    cusolvermp  lorrax_cusolvermp_eigh   CUDA  -> available
    phdf5       lorrax_phdf5_read        cpu   -> available   (lorrax probe)

That is what the profile ASSERTS, not what it assumes: the same call runs
below, and on the machine that promises those four it FAILS if one of them
stops answering.  The NEGATIVE half — every skip diffed against the
allowlist, plus the minimum-collected floor — is armed for this whole suite
in ``conftest.py`` and runs at ``sessionfinish``, so it needs no cell here.
"""

from __future__ import annotations

import pytest
from lxkit.testing import (MachineProfile, MustRow, assert_must_probe,
                           machine_profile)

from distrib_la.loader import probe_target

#: The probes this service can answer for.  A row whose service is not in
#: here comes back as UNASSERTED, which is the point of the mapping.
_PROBES = {"distrib_la": probe_target}


def test_this_machines_declared_backends_are_actually_available():
    """THE GATE.  On a machine that promises backends, they are there.

    On Perlmutter that is scalapack + slate + cusolvermp (the fourth MUST
    row, phdf5, is slab_io's — see below).  Anywhere else the profile is
    ``unknown``, which asserts nothing, on purpose: a laptop has no
    ScaLAPACK and a gate that demanded one would train everybody to
    ignore it.

    A failure here quotes the probe's own three-way reason (unknown target
    / library would not load / loaded but does not export the symbol),
    because those have three different fixes and "unavailable" names none
    of them.
    """
    profile = machine_profile()
    if profile.asserts_nothing:
        pytest.skip(
            f"machine {profile.machine!r} declares no expected-backend "
            f"profile, so there is nothing to assert — covered by the "
            f"Perlmutter leg, where NERSC_HOST selects the profile that "
            f"names scalapack, slate, cusolvermp and phdf5 as MUST rows")
    gaps = assert_must_probe(profile, _PROBES, service="distrib_la")
    assert gaps == [], (
        f"service='distrib_la' filtered the rows, so nothing should have "
        f"been unassertable; got {gaps}")


def test_the_rows_this_service_cannot_probe_are_named_not_dropped():
    """phdf5 is a MACHINE promise that distrib_la has no library for.

    It is declared in the profile anyway — the promise belongs to the
    machine, and a gate that listed only the rows it could already check
    would silently shrink to whatever was convenient.  It comes back as a
    GAP, and this cell is where the gap is written down: the slab_io
    retrofit (charter wave 1b) supplies the probe and this assertion
    inverts.
    """
    profile = machine_profile("perlmutter")
    # A STUB probe for the rows this service owns: the claim here is about
    # which rows come back UNASSERTED, and running the real probe would
    # make the cell also depend on whether this machine has the .so —
    # measuring two things and reporting one.
    gaps = assert_must_probe(profile, {"distrib_la": lambda t, p: (True, "")})
    assert [r.name for r in gaps] == ["phdf5"]
    assert gaps[0].service == "slab_io"
    # Measured on Perlmutter 2026-08-07 through lorrax's own loader:
    # lorrax_phdf5_read on 'cpu' -> available.  The row is not in doubt;
    # only WHO asserts it is.
    assert gaps[0].target == "lorrax_phdf5_read"


def test_the_gate_can_fail():
    """RED TWIN, with THIS service's real probe.

    lxkit's own twin uses a stub probe, which proves the reporting.  This
    one proves the WIRING: a MUST row naming a target distrib_la's loader
    has never heard of must make the real ``probe_target`` say no, and
    that no must reach the failure with its reason intact.  Without this
    cell, a mis-plumbed ``_PROBES`` mapping would make the gate above pass
    on any machine by asserting nothing.
    """
    fake = MachineProfile(
        machine="perlmutter",
        must=(MustRow("ghost", "distrib_la", "lorrax_no_such_target", "cpu"),),
        allowed_skips=(), min_collected=1)
    # ``pytest.fail`` raises ``Failed``, which derives from BaseException
    # and NOT from Exception — ``pytest.raises(Exception)`` does not catch
    # it, and a twin written that way ERRORS instead of passing.  Measured
    # writing this file.
    with pytest.raises(pytest.fail.Exception) as ei:
        assert_must_probe(fake, _PROBES, service="distrib_la")
    msg = str(ei.value)
    assert "lorrax_no_such_target" in msg
    assert "DEFECT" in msg
    # The probe's OWN taxonomy reached the message: this is the
    # unknown-target rung, and it must not read like the other two.
    assert "not a target" in msg or "unknown" in msg.lower(), msg


def test_a_platform_this_package_builds_nothing_for_refuses_by_name():
    """The rocm tier, which the backend table represents and nobody has
    run.  A declared-untested platform must produce a NAMED refusal about
    the library rather than a nonsense host-only/CUDA-only message about
    the backend, which is what it did before the extraction."""
    ok, reason = probe_target("lorrax_slate_potrf", "rocm")
    assert not ok
    assert "rocm" in reason and "distrib_la" in reason


def test_the_profile_names_the_machine_it_is_talking_about():
    """A failure that says "some backend is missing" without saying where
    sends the reader to guess.  Cheap to assert, and it is the part of the
    message a person acts on first."""
    assert machine_profile("perlmutter").machine == "perlmutter"
    assert machine_profile("a-laptop").machine == "a-laptop"
