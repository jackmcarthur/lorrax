"""``Gate`` — one env-gated, rank-local capability dial, and the platform key.

The one resolver behind every service dial (``LORRAX_BANDS_GEMM_FFI``);
LORRAX's ``ffi.gate`` binds it to its own probe.
It owns four things every dial has to get right:

1. **Grammar.** One spelling table (:data:`MODE_SPELLINGS`), strict per
   gate.  A value outside the gate's vocabulary is announced once and
   resolves to the gate's DEFAULT, never to ``off``: where ``off`` refuses,
   a typo must not kill a run.
2. **Rank discipline.** A decision that cannot differ per rank (platform,
   handler resolved) speaks from rank 0; one that can (env grammar, a failed
   probe) speaks from the rank it happened on, tagged ``[rank N]``.
3. **Probe.** The service's own ``probe_target``, whose reason distinguishes
   *unknown target*, *library could not be loaded* and *loaded but does not
   export*: three different fixes.
4. **Two tiers.** :meth:`Gate.enabled` reads the env only and never touches
   the JAX backend, so it is safe before ``jax.distributed.initialize`` and
   serves as a kernel-cache key.  :meth:`Gate.require`, :meth:`Gate.resolve`
   and :meth:`Gate.enforce` read the live mesh.

THE PLATFORM KEY.  :func:`device_vendor` reads the vendor from the device
client (``client.platform``, ``client.platform_version``, ``device_kind``),
never from the string ``gpu``, which JAX reports for every GPU vendor.
:func:`mesh_ffi_platform` maps the vendor to the FFI library key: ``"CUDA"``
and ``"cpu"`` have LORRAX libraries; every other vendor (``"rocm"``, an
unrecognised GPU) passes through, has no native target, and runs the XLA
path (``docs/architecture/decisions.md#xla-reference``).

``import lxkit.gate`` is stdlib-only: every jax reference sits inside a
function body.  Contract: ``docs/dev/ffi_gate_contract.md``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Optional

__all__ = [
    "Gate", "MODE_SPELLINGS", "MODE_HELP", "FFI_PLATFORM_MAP",
    "rank_id", "rank0", "announce_once", "reset_gate_state",
    "device_vendor", "mesh_ffi_platform", "platform_from_env", "dial_key",
]

#: The mode vocabulary.  Two-valued: a gate's ``off`` either runs an
#: announced fallback or refuses (:attr:`Gate.off_policy`).
MODE_SPELLINGS: dict[str, tuple[str, ...]] = {
    "off":  ("0", "off", "false", "no"),
    "on":   ("1", "on", "true", "yes"),
}

#: How each mode is spelled in a grammar-error message.
MODE_HELP: dict[str, str] = {
    "off":  "0/off/false/no",
    "on":   "1/on/true/yes",
}

_ANNOUNCED: set = set()
_RANK_ENV = ("SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK")


def rank_id() -> Optional[int]:
    """This process's rank from the launcher env, else ``jax.process_index()``.

    The launcher variables come first, in the C++ ``announce_here()`` order,
    because ``jax.process_index()`` initializes the backend.  ``None`` means
    single process or an unknown launcher; callers treat it as rank 0.
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


def announce_once(key, msg: str, *, scope: str = "rank0",
                  emit: bool = True) -> bool:
    """Print ``msg`` at most once per process for ``key``; return whether it printed.

    ``scope="rank0"``: the decision is rank-invariant and only rank 0 prints.
    ``scope="local"``: the decision is rank-local and the rank it happened on
    prints, tagged on ranks >= 1.  ``emit=False`` burns the key without
    printing, so a receipt that startup hides is not printed later.
    """
    if scope not in ("rank0", "local"):
        raise ValueError(f"announce scope must be 'rank0' or 'local', "
                         f"got {scope!r}")
    if not msg or not msg.strip():
        raise ValueError(f"announce_once({key!r}): empty message")
    if key in _ANNOUNCED:
        return False
    _ANNOUNCED.add(key)
    if not emit:
        return False
    r = rank_id()
    if scope == "rank0":
        if not (r is None or r == 0):
            return False
        print(msg, flush=True)
        return True
    print(msg if (r is None or r == 0) else f"[rank {r}] {msg}", flush=True)
    return True


def reset_gate_state() -> None:
    """Forget every memoized announcement (tests only)."""
    _ANNOUNCED.clear()


# ---------------------------------------------------------------------------
# The platform key
# ---------------------------------------------------------------------------

#: Words in a device client's identity strings that name the vendor
#: (AMD architecture names, ``gfx90a``, also mark ROCm).
_ROCM_WORDS = frozenset(("rocm", "hip", "amd", "instinct", "radeon"))
_CUDA_WORDS = frozenset(("cuda", "nvidia"))

#: Vendor -> FFI library key.  Only CUDA and cpu have LORRAX libraries; a
#: vendor missing here passes through unmapped, so a refusal names what it saw.
FFI_PLATFORM_MAP: Mapping[str, str] = MappingProxyType({
    "cpu":  "cpu",
    "cuda": "CUDA",
})


def device_vendor(device) -> str:
    """``"cpu"``, ``"cuda"``, ``"rocm"``, or the device's own platform string.

    Read from the device and its client: ``client.platform``,
    ``client.platform_version`` and ``device_kind`` (for example
    ``"cuda 13020"``, ``"NVIDIA A100-SXM4-80GB"``, ``"rocm 60300"``,
    ``"AMD Instinct MI250X"``).  ``device.platform`` alone is ``"gpu"`` for
    every GPU vendor.  A GPU whose strings name no known vendor returns its
    platform string (``"gpu"``) and has no vendor route.
    """
    plat = str(getattr(device, "platform", "") or "").lower()
    if plat == "cpu":
        return "cpu"
    client = getattr(device, "client", None)
    text = " ".join(str(s or "") for s in (
        getattr(client, "platform", ""), getattr(client, "platform_version", ""),
        getattr(device, "device_kind", ""), plat)).lower()
    words = set(re.findall(r"[a-z]+", text))
    if words & _ROCM_WORDS or any(w.startswith("gfx") for w in words):
        return "rocm"
    if words & _CUDA_WORDS:
        return "cuda"
    return plat


def mesh_ffi_platform(mesh, platform_map: Mapping[str, str] = FFI_PLATFORM_MAP
                      ) -> str:
    """The FFI platform key of ``mesh``'s devices: ``"CUDA"``, ``"cpu"``, or the
    unmapped vendor (``"rocm"``, ``"gpu"``, ``"tpu"``), which has no library."""
    vendor = device_vendor(mesh.devices.flat[0])
    return platform_map.get(vendor, vendor)


def platform_from_env(default: str = "CUDA") -> str:
    """The FFI platform key from ``JAX_PLATFORMS``, without touching the backend.

    For code that must decide before ``jax.distributed.initialize``.  The
    first entry wins, as in JAX: ``cuda``/``gpu`` -> ``"CUDA"``, ``rocm`` ->
    ``"rocm"``, anything else -> ``"cpu"``; unset -> ``default``.
    """
    first = os.environ.get("JAX_PLATFORMS", "").split(",")[0].strip().lower()
    if not first:
        return default
    if first in ("cuda", "gpu"):
        return "CUDA"
    return "rocm" if first == "rocm" else "cpu"


def dial_key(*gates: "Gate") -> tuple:
    """The cache-key component of every factory-time dial: ``(env, enabled)``
    per gate, read at tier 1 (no backend init)."""
    return tuple((g.env, g.enabled()) for g in gates)


def _no_probe_configured(target: str, platform: str) -> tuple[bool, str]:
    """The default :attr:`Gate.probe`: refuse, because nothing can confirm the target."""
    return (False, (
        f"no probe is wired into this Gate, so nothing can confirm that "
        f"target {target!r} is usable on platform {platform!r}; construct the "
        f"Gate with probe=<callable (target, platform) -> (ok, reason)>, the "
        f"service's own probe_target."))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """One env-gated capability.  Messages are fields (harnesses grep them;
    refusals must name the fix); the policy is here and identical for all."""

    env: str                          #: e.g. "LORRAX_BANDS_GEMM_FFI"
    target: str                       #: default FFI target to probe
    platforms: tuple[str, ...]        #: platform keys this dial exists on
    modes: tuple[str, ...]            #: this gate's vocabulary, in help order
    default: str                      #: what unset/empty means: "on"|"off"
    off_label: str                    #: what =0 selects, for announcements
    label: Mapping[str, str] = field(default_factory=dict)   #: platform -> name
    #: ``(target, platform) -> (ok, reason)``; ``None`` refuses.
    probe: Callable[[str, str], tuple[bool, str]] | None = None
    #: vendor -> FFI-platform vocabulary for this gate's meshes.
    platform_map: Mapping[str, str] = field(
        default_factory=lambda: FFI_PLATFORM_MAP)
    #: What an explicit ``=0`` means: ``"fallback"`` runs the retained XLA
    #: path, announced; ``"refuse"`` raises, naming why there is no path.
    off_policy: str = "fallback"
    off_announce_msg: str = ""             #: fallback announce (no fields)
    off_refuse_msg: str = ""               #: refuse prose (no fields)
    resolved_msg: Mapping[str, str] = field(default_factory=dict)  #: {target}
    #: Non-empty: an out-of-scope platform is skipped silently, for this reason.
    silent_platform_demote: str = ""
    platform_demote_msg: str = ""          #: {platform}  (announced demote)
    refuse_platform_msg: str = ""          #: {platform}
    refuse_probe_msg: str = ""             #: {target} {platform} {label} {reason}

    def __post_init__(self) -> None:
        """Refuse, at construction, a vocabulary this resolver cannot serve."""
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
        """``"on"`` | ``"off"``.  Unset -> the default; an unrecognised value is
        announced on the rank that read it and resolves to the default."""
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
            f"Treating as the default ({self.default.upper()}). ***",
            scope="local")
        return self.default

    # -- tier 1: lexical (cache-key safe) -------------------------------

    def enabled(self) -> bool:
        """Is this capability on?  The env alone; never initializes the backend."""
        return self.mode() != "off"

    def _demote_msg(self, plat: str) -> str:
        if self.platform_demote_msg:
            return self.platform_demote_msg.format(platform=plat)
        return (f"[{self.env}] OFF — this dial exists on "
                f"{'/'.join(self.platforms)} and the platform resolved to "
                f"{plat!r}; keeping the default lowering.")

    def _skip_platform(self, mesh, *, announce: bool = True) -> None:
        if not self.silent_platform_demote:
            announce_once((self.env, "mesh", "platform"),
                          self._demote_msg(mesh_ffi_platform(mesh, self.platform_map)),
                          scope="rank0", emit=announce)

    # -- tier 2: mesh-aware --------------------------------------------

    def platform_ok(self, mesh) -> bool:
        """True when ``mesh``'s devices are a platform this dial exists on."""
        return mesh_ffi_platform(mesh, self.platform_map) in self.platforms

    def require(self, mesh, *, target: str | None = None,
                announce: bool = True) -> str:
        """The FFI platform key, or ``RuntimeError``.  Mode-independent: it
        answers whether this mesh can serve this handler, quoting the probe's
        reason on failure, and prints the first-use receipt once on rank 0."""
        tgt = target or self.target
        plat = mesh_ffi_platform(mesh, self.platform_map)
        if plat not in self.platforms:
            raise RuntimeError(
                self.refuse_platform_msg.format(platform=plat)
                if self.refuse_platform_msg else
                f"{self.env} requested the {tgt!r} FFI handler, but the mesh "
                f"devices are {plat!r} — this dial exists on "
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
                          msg.format(target=tgt), scope="rank0", emit=announce)
        return plat

    def resolve(self, mesh, *, target: str | None = None) -> str | None:
        """The FFI platform key, or ``None`` when off or out of scope (announced
        unless :attr:`silent_platform_demote` says why not); an in-scope
        handler that cannot be served raises."""
        if self.mode() == "off":
            return None
        if not self.platform_ok(mesh):
            self._skip_platform(mesh)
            return None
        return self.require(mesh, target=target)

    def enforce(self, mesh, *, announce: bool = True) -> str | None:
        """The startup contract, once per gate after the mesh exists: ``on``
        requires the handler (a missing library refuses here, naming the
        ``.so``); ``off`` announces the fallback or refuses per
        :attr:`off_policy`; an out-of-scope platform returns ``None``."""
        if self.mode() == "off":
            if self.off_policy == "refuse":
                raise RuntimeError(
                    self.off_refuse_msg or
                    f"{self.env}=0: there is nothing to opt out to; the "
                    f"{self.target!r} path has no other implementation.  "
                    f"Unset {self.env}.")
            announce_once(
                (self.env, "opt-out"),
                self.off_announce_msg or
                f"[{self.env}] =0: explicit debug opt-out — running the "
                f"retained XLA path ({self.off_label}).",
                scope="local", emit=announce)
            return None
        if not self.platform_ok(mesh):
            self._skip_platform(mesh, announce=announce)
            return None
        return self.require(mesh, announce=announce)
