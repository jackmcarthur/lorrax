"""``Gate`` — ONE env-gated, rank-local capability dial.

Ported from LORRAX ``src/ffi/gate.py`` at 96a6399 (the resolver behind
``LORRAX_BANDS_GEMM_FFI``, ``LORRAX_FFT_FFI`` and ``LORRAX_FFT_FFI_FUSED``).
It owns the four things every service dial has to get right, in one place
instead of N drifting copies:

1. **Grammar** — a strict, per-gate NAMED vocabulary (:data:`MODE_SPELLINGS`).
   A value outside the gate's own vocabulary is announced once and takes
   the gate's DEFAULT — never silently ignored.  This is the defect that
   made ``LORRAX_FFT_FFI_FUSED=Y`` a no-op with no output.  (The fallback
   used to be ``off``; since the FFI layer became required — see below —
   the direction that cannot break a run is the certified default.)
2. **Platform** — resolved in TWO separate tiers (below), because one of
   them must not initialize the JAX backend.
3. **Probe** — the service's own ``probe_target``, whose three-way reason
   (*unknown target* / *library could not be loaded* / *loaded but does not
   export*) is quoted verbatim into every refusal.  A bare bool would tell
   somebody to rebuild a library that is fine.  lxkit ships the reason
   VOCABULARY (:mod:`lxkit.probe`); the table belongs to the service.
4. **Announce or refuse** — a request that cannot be honored RAISES at
   resolve time with the reason and the fix; the only silences are the
   ones a gate DECLARES (:attr:`Gate.silent_platform_demote`).

REQUIRED, NOT OPTIONAL (owner ruling, ``docs/architecture/decisions.md``
2026-08-01).  The FFI layer is an essential part of the build: every gate
now defaults ON, a missing or unloadable library is a STARTUP refusal that
names the ``.so`` (:meth:`Gate.enforce`, called by
``runtime.initialize_communicator_stack`` right after the mesh is built),
and ``=0`` is an explicit debug opt-OUT whose meaning is per-gate
(:attr:`Gate.off_policy`): where a native-JAX twin still exists for a
structural reason it runs, announced; where the certified FFI path made the
twin deletable, the twin is GONE and ``=0`` refuses, naming the deletion.
The old ``auto`` capability-detection tier was deleted with the same
ruling — auto-demotion to a duplicate compute path is exactly what the
ruling forbids ("a refusal at startup ... not a silent demotion to a
slower path").  ``auto`` is gone from :data:`MODE_SPELLINGS` too: no
gate could declare it (:meth:`enabled` / :meth:`resolve` / :meth:`enforce`
have no auto branch, so a gate listing it would resolve ``=auto``
silently as ``on``), and a stale ``=auto`` export already announces the
grammar note and takes the default without a vocabulary entry.

Two tiers — keeping them apart is the whole reason this module exists
---------------------------------------------------------------------
:meth:`Gate.enabled` — **tier 1, LEXICAL.**  Reads the env var and nothing
    else (the modes are two-valued now that ``auto`` is gone, so no probe
    and no platform read).  It never touches the JAX backend, so it is safe
    before ``jax.distributed.initialize`` — which it must be, because
    consumers use it as a KERNEL-CACHE KEY at factory time
    (``gw.ppm_tau_kernel``'s ``pipeline_key``/``cache_key``; aggregate them
    with :func:`dial_key`).

:meth:`Gate.resolve` — **tier 2, MESH-AWARE.**  Takes the platform from the
    live devices, so it is exact; it needs a ``Mesh`` and therefore an
    initialized backend.  Returns the FFI platform key to lower on, or
    ``None`` when the gate is off.  This is where requests refuse.

:meth:`Gate.enforce` — **tier 2, STARTUP.**  The required-layer contract:
    called once per gate right after the mesh exists, so a missing library
    refuses AT STARTUP, naming the ``.so``, instead of at the first kernel
    factory mid-run.

A single-tier resolver cannot serve both: tier 2 is exact but far too late
for a cache key, tier 1 is early enough but only lexically right.  That —
not taste — is why these dials do not fold into a ``plan(op, mesh,
backend=, n=)``-shaped resolver, which receives everything as arguments and
resolves once.

Two-PHASE refusal, also deliberate.  Only the platform and handler guards
can fire at resolve time.  Operand dtype/rank/extent are trace-time facts,
so those refusals necessarily live in the wrapper body; a ``plan()``-shaped
single-phase API would have to lie about when it checked.

Rank discipline
---------------
:func:`rank_id` reads the LAUNCHER's rank (``SLURM_PROCID`` / ``PMI_RANK`` /
``OMPI_COMM_WORLD_RANK``) — the SAME rule, in the same order, as the C++
side's ``announce_here()`` (``cpp/mklblas/gemm_batch_ffi.cc:229-235``), so
Python and C++ agree on who speaks.  It deliberately does NOT call
``jax.process_index()`` first: that goes through ``get_backend()``
(``jax/_src/xla_bridge.py:1119``) and INITIALIZES the backend, which would
destroy tier 1's whole promise.  ``jax.process_index()`` remains the
fallback when no launcher variable is set (single process, or a launcher we
do not know), where it is both correct and harmless — and it is the ONLY
jax reference in this module, inside a ``try`` inside a function body, so
``import lxkit.gate`` stays stdlib-only.

Scope of an announcement follows one rule: *can this decision differ per
rank?*

* platform / mesh geometry — no.  Rank 0 speaks for all (``scope="rank0"``).
* env-var grammar, and a FAILED probe — **yes**: the env is per-process and
  a probe failure is usually one rank's ``LD_LIBRARY_PATH``.  Announced
  from the rank it happened on (``scope="local"``), rank-tagged on ranks
  >= 1 so it is attributable and so rank 0's line stays byte-identical to
  what the gate harnesses grep for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Optional

__all__ = [
    "Gate", "MODE_SPELLINGS", "MODE_HELP", "FFI_PLATFORM_MAP",
    "rank_id", "rank0", "announce_once", "mesh_ffi_platform",
    "platform_from_env", "reset_gate_state", "dial_key",
]

# ---------------------------------------------------------------------------
# The named vocabulary.  A gate declares WHICH of these modes it accepts;
# anything else is a grammar error.  ("" — unset or whitespace — always maps
# to the gate's declared default, never to a mode spelling.)
#
# TWO-VALUED, deliberately.  The ``auto`` capability-detection tier was
# deleted with decisions.md 2026-08-01 and there is no ``auto`` branch left
# in :meth:`Gate.enabled` / :meth:`Gate.resolve` / :meth:`Gate.enforce` — so
# a vocabulary entry for it could only mislead: a gate that declared
# ``modes=("auto", ...)`` would accept ``=auto`` and then behave as ``on``
# with no announcement, which is the silent-downgrade shape the ruling
# forbids.  A stale ``=auto`` export is a grammar error and is announced as
# one.  If an auto tier ever returns it needs a resolver, not a token.
# ---------------------------------------------------------------------------
MODE_SPELLINGS: dict[str, tuple[str, ...]] = {
    "off":  ("0", "off", "false", "no"),
    "on":   ("1", "on", "true", "yes"),
}

#: How each mode is spelled in a grammar-error message.
MODE_HELP: dict[str, str] = {
    "off":  "0/off/false/no",
    "on":   "1/on/true/yes",
}

#: A grammar error resolves to the gate's DEFAULT (announced).  An
#: unparseable value is not evidence of intent either way, so take the
#: direction that cannot break a run — which, now that the FFI layer is
#: required (decisions.md 2026-08-01), is the certified default path, not
#: ``off``: for a gate whose deleted native twin makes ``off`` a REFUSAL,
#: falling back to ``off`` would turn a typo into a dead run.
#: (Kept as prose rather than a constant: the fallback IS ``self.default``.)

_ANNOUNCED: set = set()


# ---------------------------------------------------------------------------
# Rank / announcement plumbing (replaces 3 copies of _rank0 + 2 dedupe sets)
# ---------------------------------------------------------------------------

_RANK_ENV = ("SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK")


def rank_id() -> Optional[int]:
    """This process's rank from the LAUNCHER env, or ``None`` if unknown.

    Backend-init-free by construction (module docstring, "Rank
    discipline").  ``None`` means "single process, or an unrecognized
    launcher" and every caller treats it as rank 0.
    """
    for var in _RANK_ENV:
        raw = os.environ.get(var)
        if raw is not None and raw.strip() != "":
            try:
                return int(raw)
            except ValueError:
                return None
    try:
        import jax
        return int(jax.process_index())
    except Exception:                                     # noqa: BLE001
        return None


def rank0() -> bool:
    """True on the rank designated to speak for facts that cannot differ."""
    r = rank_id()
    return r is None or r == 0


def announce_once(key, msg: str, *, scope: str = "rank0") -> bool:
    """Print ``msg`` at most once per process for ``key``.  Returns whether
    it printed.

    ``scope="rank0"``  the decision is rank-invariant; only rank 0 prints.
    ``scope="local"``  the decision is rank-LOCAL (env grammar, a failed
                       probe); the rank it happened on prints, tagged on
                       ranks >= 1.  Suppressing these on rank 0's behalf is
                       exactly how a one-rank misconfiguration hides.
    """
    # Validate BEFORE touching the dedupe set: a rejected call must not burn
    # the key, or the retry after the fix would be silently swallowed.
    if scope not in ("rank0", "local"):
        raise ValueError(f"announce scope must be 'rank0' or 'local', "
                         f"got {scope!r}")
    if not msg or not msg.strip():
        # A blank announcement is worse than none: it tells nobody anything
        # while looking like the doctrine was honored.  Callers that can
        # reach here supply a generated fallback (see Gate._demote_msg).
        raise ValueError(f"announce_once({key!r}): empty message")
    if key in _ANNOUNCED:
        return False
    _ANNOUNCED.add(key)
    r = rank_id()
    if scope == "rank0":
        if not (r is None or r == 0):
            return False
        print(msg, flush=True)
        return True
    print(msg if (r is None or r == 0) else f"[rank {r}] {msg}", flush=True)
    return True


def reset_gate_state() -> None:
    """Forget every memoized announcement (tests only).

    ``_ANNOUNCED`` is process-global, so without this every
    announcement-asserting test is order-dependent: whichever cell runs
    first burns the key and the rest observe silence.  In lorrax at
    96a6399 this function had ZERO callers.  :func:`lxkit.testing.gate_state`
    is the autouse fixture that gives it one.
    """
    _ANNOUNCED.clear()


#: The default FFI platform vocabulary: the two platforms LORRAX builds
#: libraries for.  It is a PARAMETER of :func:`mesh_ffi_platform` and a
#: FIELD of :class:`Gate` because the strings are a service's vocabulary,
#: not a fact about meshes — a service with a third platform key passes its
#: own map instead of forking this module.
FFI_PLATFORM_MAP: Mapping[str, str] = MappingProxyType({
    "cpu":  "cpu",
    "gpu":  "CUDA",
    "cuda": "CUDA",
})


def mesh_ffi_platform(mesh, platform_map: Mapping[str, str] = FFI_PLATFORM_MAP
                      ) -> str:
    """The FFI platform key for ``mesh``'s devices, per ``platform_map``.

    ``"cpu"`` / ``"CUDA"`` for the two platforms that have libraries;
    anything else passes through UNMAPPED so a refusal can name what it
    actually saw (``'tpu'`` reads better than ``'unknown'``).
    """
    plat = mesh.devices.flat[0].platform
    return platform_map.get(plat, plat)


def platform_from_env(default: str = "CUDA") -> str:
    """Resolve the FFI platform from ``JAX_PLATFORMS`` WITHOUT touching the
    JAX backend — safe before ``jax.distributed.initialize``.  The first
    entry wins, mirroring how JAX picks its default backend from the list.
    """
    first = os.environ.get("JAX_PLATFORMS", "").split(",")[0].strip().lower()
    if not first:
        return default
    return "CUDA" if first in ("cuda", "gpu") else "cpu"


def dial_key(*gates: "Gate") -> tuple:
    """The ONE cache-key component capturing every factory-time dial.

    Kernel factories read their backend dials at FACTORY time, so a kernel
    cache that omits the dials serves a stale backend after a mid-process
    flag flip (tests flip them; the service contract says the dial MUST be
    in every consumer cache key).  This helper is the single owner of
    "which dials were live when this factory ran"; consumers fold the
    returned tuple into their cache keys instead of each re-listing the
    dials (and drifting when a dial is added).

    Every read is tier-1 lexical (no JAX backend init) and O(1) — safe in
    any cache-lookup path at any P.
    """
    return tuple((g.env, g.enabled()) for g in gates)


def _no_probe_configured(target: str, platform: str) -> tuple[bool, str]:
    """The DEFAULT :attr:`Gate.probe`: refuse, and say why.

    A permissive default (``lambda *_: (True, "assumed")``) is exactly the
    silent downgrade this module exists to forbid — it would turn "nobody
    wired a probe" into "the handler is fine", and the first symptom would
    be an XLA error from inside a lowered kernel instead of a refusal that
    names the library.  So the unwired gate refuses at :meth:`Gate.require`.
    """
    return (False, (
        f"no probe is wired into this Gate, so nothing can confirm that "
        f"target {target!r} is usable on platform {platform!r}.  lxkit "
        f"refuses rather than assuming it is: a permissive default would be "
        f"the silent downgrade this gate exists to prevent.  Construct the "
        f"Gate with probe=<callable (target, platform) -> (ok, reason)> — "
        f"the service's own probe_target."))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """One env-gated, REQUIRED capability (decisions.md 2026-08-01).

    Every message is a field rather than generated prose: these strings are
    the product (harnesses grep the announce lines; refusals must name the
    caller's fix), so each service owns its own wording while the *policy* —
    when to warn, when to raise, what ``=0`` means, and who speaks — lives
    here and is identical for all of them.
    """

    env: str                          #: e.g. "LORRAX_FFT_FFI"
    target: str                       #: default FFI target to probe
    platforms: tuple[str, ...]        #: platform keys this dial exists on
    modes: tuple[str, ...]            #: THIS gate's vocabulary, in help order
    default: str                      #: what unset/empty means: "on"|"off"
    off_label: str                    #: what =0 selects, for announcements
    label: Mapping[str, str] = field(default_factory=dict)   #: platform → name
    #: ``(target, platform) -> (ok, reason)``.  The service's own probe;
    #: :data:`lxkit.probe.ProbeResult` is a ``tuple[bool, str]`` so a probe
    #: may return either.  ``None`` means unwired, which REFUSES — see
    #: :func:`_no_probe_configured`.
    probe: Callable[[str, str], tuple[bool, str]] | None = None
    #: jax-platform → FFI-platform vocabulary for this gate's meshes.
    #: ``default_factory`` because a ``mappingproxy`` is unhashable and
    #: ``dataclasses`` rejects unhashable defaults; the factory hands out the
    #: one shared read-only table, not a copy.
    platform_map: Mapping[str, str] = field(
        default_factory=lambda: FFI_PLATFORM_MAP)
    # -- required-layer policy (decisions.md 2026-08-01) ----------------
    #: What an explicit ``=0`` means for THIS gate:
    #: ``"fallback"`` — a native-JAX path still exists (kept for a
    #:     structural reason, e.g. the GEMM minor-order case); ``=0`` runs
    #:     it, announced once as an uncertified debug opt-out.
    #: ``"refuse"``   — the native duplicate was DELETED under the ruling;
    #:     ``=0`` names the deletion and raises at :meth:`enforce` (and the
    #:     factory raises too, for callers outside the startup path).
    off_policy: str = "fallback"
    off_announce_msg: str = ""             #: fallback announce (no fields)
    off_refuse_msg: str = ""               #: refuse prose (no fields)
    # -- prose: announce -----------------------------------------------
    resolved_msg: Mapping[str, str] = field(default_factory=dict)  #: {target}
    #: Non-empty ⇒ an out-of-scope platform is skipped SILENTLY by
    #: :meth:`enforce`/:meth:`resolve`, and this string is the recorded
    #: reason.  Silence is allowed only when declared: the alternative
    #: (announcing "no GPU GEMM dial here" on every GPU run) is noise about
    #: a decision the user cannot act on.
    silent_platform_demote: str = ""
    platform_demote_msg: str = ""          #: {platform}  (announced demote)
    # -- prose: refuse -------------------------------------------------
    #: Refusal prose.  A gate MAY leave these empty — :meth:`require` then
    #: generates a serviceable message from ``env``/``target``/``platforms``.
    #: Write them when the refusal has a domain reason worth stating (why
    #: the dial is scoped to this platform, what to do instead); never leave
    #: a refusal without SOME message, which is why the fallback exists.
    refuse_platform_msg: str = ""          #: {platform}
    refuse_probe_msg: str = ""             #: {target} {platform} {label} {reason}

    def __post_init__(self) -> None:
        """Refuse a gate whose declared vocabulary this resolver cannot serve.

        At construction, not at the first env read: ``mode()`` would
        otherwise ``KeyError`` on ``MODE_SPELLINGS[m]`` on whichever rank
        happened to read the variable first.  The check is what makes the
        two-valued table above safe to rely on — a mode with no resolver
        branch cannot be declared into existence.
        """
        unknown = [m for m in self.modes if m not in MODE_SPELLINGS]
        if unknown:
            raise ValueError(
                f"Gate({self.env}): mode(s) {unknown} are not in the gate "
                f"vocabulary {tuple(MODE_SPELLINGS)}.  Every declared mode "
                f"needs a resolver branch in enabled()/resolve()/enforce(); "
                f"adding a token alone makes it behave as 'on', silently.")
        if self.default not in MODE_SPELLINGS:
            raise ValueError(
                f"Gate({self.env}): default={self.default!r} is not one of "
                f"{tuple(MODE_SPELLINGS)}.")

    # -- tier 0: grammar ------------------------------------------------

    def mode(self) -> str:
        """``"on"`` | ``"off"`` — this gate's strict grammar.

        Unset/empty → :attr:`default`.  A value outside :attr:`modes`
        announces once (rank-locally: the env is per-process) and returns
        the DEFAULT — the certified required path, the direction that
        cannot break a run now that ``off`` may itself refuse.
        """
        v = os.environ.get(self.env, "").strip().lower()
        if v == "":
            return self.default
        for m in self.modes:
            if v in MODE_SPELLINGS[m]:
                return m
        announce_once(
            (self.env, "grammar"),
            f"*** {self.env}={v!r} is not a recognized value "
            f"(accepted: {', '.join(MODE_HELP[m] for m in self.modes)}).  "
            f"Treating as the default ({self.default.upper()} — the FFI "
            f"layer is required, decisions.md 2026-08-01). ***",
            scope="local")
        return self.default

    # -- tier 1: lexical (cache-key safe) -------------------------------

    def enabled(self) -> bool:
        """Is this capability ON?  **Never initializes the JAX backend.**

        THE function consumers key their kernel caches on.  Answers from
        the env alone (the vocabulary is two-valued since the ``auto``
        capability-detection tier was deleted, decisions.md 2026-08-01);
        no probe, no platform read, O(1) at any call site.
        """
        return self.mode() != "off"

    def _demote_msg(self, plat: str) -> str:
        """The announced platform demote.  Generated when the gate did not
        write one, so an unset field can never turn a demote silent —
        silence requires :attr:`silent_platform_demote`, which is checked by
        the caller and must carry its reason."""
        if self.platform_demote_msg:
            return self.platform_demote_msg.format(platform=plat)
        return (f"[{self.env}] OFF — this dial exists on "
                f"{'/'.join(self.platforms)} and the platform resolved to "
                f"{plat!r}; keeping the default lowering.")

    # -- tier 2: mesh-aware --------------------------------------------

    def platform_ok(self, mesh) -> bool:
        """True when ``mesh``'s devices are a platform this dial exists on."""
        return mesh_ffi_platform(mesh, self.platform_map) in self.platforms

    def require(self, mesh, *, target: str | None = None) -> str:
        """Announce-or-REFUSE: the FFI platform key, or ``RuntimeError``.

        Deliberately MODE-INDEPENDENT.  It answers exactly one question —
        *can this mesh serve this handler?* — which is what an explicit
        request asks and also what a wrapper built directly by a caller
        asks (a wrapper constructed by a consumer that has already decided;
        its refusals must not depend on re-reading a flag the consumer
        already read).  Refuses on an out-of-scope platform, and on a
        handler this platform's library cannot serve — quoting
        :attr:`probe`'s reason, because "the .so would not load" and "the
        .so has no such handler" have different fixes.

        Prints the service's first-use receipt once on rank 0.
        """
        tgt = target or self.target
        raw = mesh.devices.flat[0].platform
        plat = mesh_ffi_platform(mesh, self.platform_map)
        if plat not in self.platforms:
            raise RuntimeError(
                self.refuse_platform_msg.format(platform=raw)
                if self.refuse_platform_msg else
                f"{self.env} requested the {tgt!r} FFI handler, but the mesh "
                f"devices are {raw!r} — this dial exists on "
                f"{'/'.join(self.platforms)} only.  Unset {self.env} on this "
                f"mesh (explicit requests are never silently downgraded).")
        ok, reason = (self.probe or _no_probe_configured)(tgt, plat)
        if not ok:
            raise RuntimeError(
                self.refuse_probe_msg.format(
                    target=tgt, platform=plat,
                    label=self.label.get(plat, plat), reason=reason)
                if self.refuse_probe_msg else
                f"{self.env} requested the "
                f"{self.label.get(plat, plat)} backend, but FFI target "
                f"{tgt!r} is unusable on platform {plat!r}: {reason}")
        msg = self.resolved_msg.get(plat)
        if msg:
            announce_once((self.env, "resolved", tgt, plat),
                          msg.format(target=tgt), scope="rank0")
        return plat

    def resolve(self, mesh, *, target: str | None = None) -> str | None:
        """Mode-aware resolution: the FFI platform key, or ``None`` if OFF.

        * ``off`` — ``None``.  No probe, no library load, nothing printed
          (the opt-out announcement is :meth:`enforce`'s, once at startup).
        * ``on`` (the default) — out-of-scope platform ⇒ ``None``,
          announced on rank 0 unless :attr:`silent_platform_demote`
          declares why silence is right (the platform's own native lowering
          IS the required path there); in scope ⇒ :meth:`require`, which
          RAISES when the handler cannot be served — never downgrades.
        """
        if self.mode() == "off":
            return None
        if not self.platform_ok(mesh):
            if not self.silent_platform_demote:
                announce_once(
                    (self.env, "mesh", "platform"),
                    self._demote_msg(mesh.devices.flat[0].platform),
                    scope="rank0")
            return None
        return self.require(mesh, target=target)

    def enforce(self, mesh) -> str | None:
        """The startup contract of the REQUIRED FFI layer (decisions.md
        2026-08-01).  Called once per gate right after the mesh is built,
        so the outcome is decided — and any refusal fires — at STARTUP, not
        at the first kernel factory mid-run.

        * default / ``on`` — :meth:`require`: a missing or unloadable
          library REFUSES here, naming the ``.so`` (the probe's three-way
          reason, verbatim).  Out-of-scope platform ⇒ ``None``, silently
          when :attr:`silent_platform_demote` declares why (the platform's
          native lowering is the required path there).
        * ``off`` with ``off_policy="fallback"`` — the retained native
          path runs; announced once, rank-locally, as an uncertified debug
          opt-out.
        * ``off`` with ``off_policy="refuse"`` — the native duplicate was
          deleted; RAISES, naming the deletion and the fix.
        """
        if self.mode() == "off":
            if self.off_policy == "refuse":
                raise RuntimeError(
                    self.off_refuse_msg or
                    f"{self.env}=0: there is nothing to opt out to.  The "
                    f"native-JAX duplicate of the {self.target!r} path was "
                    f"deleted under the FFI-required ruling "
                    f"(docs/architecture/decisions.md, 2026-08-01); recover "
                    f"it from git history for a debugging build, or unset "
                    f"{self.env}.")
            announce_once(
                (self.env, "opt-out"),
                self.off_announce_msg or
                f"[{self.env}] =0: explicit debug opt-out — running the "
                f"retained native-JAX path ({self.off_label}).  This path "
                f"is UNCERTIFIED for production (the FFI layer is required, "
                f"decisions.md 2026-08-01).",
                scope="local")
            return None
        if not self.platform_ok(mesh):
            if not self.silent_platform_demote:
                announce_once(
                    (self.env, "mesh", "platform"),
                    self._demote_msg(mesh.devices.flat[0].platform),
                    scope="rank0")
            return None
        return self.require(mesh)
