"""THE CLASS-B LINT — rank-dependent values reaching a compile-time surface.

WHAT THIS FILE IS FOR
---------------------
Every cache PITA of the 2026-08-08 BSE performance campaign was one of two
classes.  Class A — *the program cannot persist* — is caught at RUN time by
``tests/test_jax_cache_contract.py``'s veto arm, because a host callback is
a property of the emitted module and not of the source.  Class B is not:

    a value that DIFFERS BY RANK reaches a jit signature, so each rank
    compiles a different program, holds a different persistent-cache key,
    and — because JAX writes cache entries from process 0 only — the peers
    miss and recompile while process 0 hits.  Asymmetric hit/miss across
    ranks is the collective-compile deadlock PRECONDITION.

Class B IS visible in the source, and it is visible BEFORE the four-GPU leg
that would otherwise be the only thing that could find it.  That is what
this lint is: the cheap half of the contract.

    ``jit__multi_slice``   FIX_multislice_cachekey.md, canonicalized d6303e61
    the five siblings      that report's §6.1, all five fixed or red-listed
                           on this branch

THE FIVE RULES, AND WHY EACH ONE EXISTS
---------------------------------------
Every rule was written against a site that actually shipped.  A rule with no
site behind it is a rule that will only ever produce false positives.

``rank-static-arg``   a rank-dependent value passed where a ``jax.jit``
                      declared ``static_argnums`` / ``static_argnames``.
                      The canonical case: jax's own ``_multi_slice``.
``rank-shape``        a rank-dependent value reaching an ARRAY SHAPE.  A
                      shape is static to XLA whether or not anyone spelled
                      it ``static``, so ``jnp.arange(lo, hi)`` with a rank
                      slab is the same defect wearing different clothes.
                      Sites: ``vq_interp``, ``kin_ion_io``, ``charge_density``.
``rank-cache-key``    a rank-dependent value inside a memo/kernel-cache KEY.
                      Two ranks then cannot share an entry by construction,
                      and the key is the written record of the divergence.
                      Site: ``vq_interp._MBZ_DQ_CACHE``.
``rank-branch``       a rank-dependent CONDITION guarding a region that
                      compiles.  This is the one no shape check can see:
                      the surplus ranks of ``world > n_items`` compile
                      NOTHING while their peers compile everything, which is
                      maximal asymmetry.  Sites: ``vq_interp``'s
                      ``if hi > lo:``, ``charge_density``'s ``i % world``,
                      ``local_share``'s sanctioned empty share.
``env-dial``          a PER-PROCESS environment dial reaching a kernel
                      factory's cache key without being declared in the
                      cross-rank fingerprint.  Non-uniform env across ranks
                      then means a different HLO BODY per rank — the same
                      divergence arriving through the environment instead of
                      through a shard offset.  Site: ``ffi.ffi_dial_key``.

WHY AN ALLOWLIST EXISTS AND WHAT IT COSTS
------------------------------------------
A site that is genuinely non-mechanical to canonicalize is RED-LISTED rather
than forced: it gets a row in :data:`ALLOW` naming the file, the rule, the
date and the reason, and the contract gate reads the same rows.  An
allowlist entry is a DEBT with a name on it, not a silence — that is why
each row carries a date and why :func:`stale_allow_rows` exists to fail when
a row no longer matches anything.

WHY IT IS A PURE MODULE AND NOT A PYTEST FILE
----------------------------------------------
Same convention as ``fast_gate`` and ``harness.pin_one_gpu``: the decisions
are functions over text, so they are falsifiable without a pytest session
and without jax.  ``tests/test_cache_key_lint.py`` is the twenty lines that
run them.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Roots this lint is responsible for.  ``services/*/src`` and not
#: ``services/*``, because a service's own tests may construct the pattern
#: on purpose (that is what a red twin IS).
SCAN_ROOTS = ("src", "services")


# ---------------------------------------------------------------------------
# TAINT SOURCES — spelled out, never pattern-matched on the word "rank"
# ---------------------------------------------------------------------------
# A lint whose source list is a regex over identifiers containing "rank"
# flags every loop counter in the tree and is switched off within a week.
# These are the exact spellings by which a rank-dependent value enters
# LORRAX, each one verified against a call site in this repository.

#: Calls whose RESULT is rank-dependent.
RANK_CALLS = frozenset({
    "process_index",          # jax.process_index()
    "process_rank",           # common.collectives.process_rank()
    "local_share",            # ...[rank::world] — a rank-dependent LIST
    "local_devices",          # jax.local_devices() — this process's devices
    "addressable_data",
    "addressable_devices_indices_map",
    "addressable_shards",
})

#: Calls whose result IS a work partition, with no arithmetic needed at the
#: call site: ``local_share(items)`` is already ``items[rank::world]``.
#: ``addressable_devices_indices_map`` and ``local_devices`` are DELIBERATELY
#: absent.  Both are rank sources, but their common use is an OWNERSHIP TEST
#: (``if not idx_map:`` — does this process hold any device at all), which
#: divides no work and which FIX_multislice_cachekey.md §4.1 already cleared
#: by name.  They still count as rank SOURCES, so arithmetic on them is
#: still a partition.
PARTITION_CALLS = frozenset({"local_share", "addressable_data"})

#: Calls returning ``(rank, world)``.  Element 0 is rank-dependent and
#: element 1 is NOT — a lint that taints ``world`` flags every correct
#: partition in the tree, so the tuple is destructured explicitly.
RANK_WORLD_CALLS = frozenset({"process_rank_world"})

#: Attributes whose value is this process's share of something.
RANK_ATTRS = frozenset({"addressable_shards", "addressable_data"})

#: Environment variables that name THIS RANK.  Reading one is a taint
#: source in the same way ``process_index()`` is.
RANK_ENV = frozenset({
    "SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "JAX_PROCESS_INDEX",
})

#: Functions whose return value is a digest of PER-PROCESS environment.
#: Distinct from :data:`RANK_CALLS` because the defect and the fix differ:
#: the value is not "which rank am I", it is "what was my environment", and
#: the fix is to declare it in the cross-rank fingerprint rather than to
#: canonicalize a shape.
ENV_KEY_CALLS = frozenset({"ffi_dial_key"})


# ---------------------------------------------------------------------------
# SINKS
# ---------------------------------------------------------------------------
#: ``jnp``/``np`` constructors whose ARGUMENTS become an array SHAPE.  A
#: shape is static to XLA whether or not it was spelled ``static``.
SHAPE_CALLS = frozenset({
    "arange", "zeros", "ones", "empty", "full", "eye", "linspace",
    "reshape", "broadcast_to", "zeros_like", "tile", "repeat",
})

#: Names that mark a value as a CACHE KEY.  Deliberately a name test: the
#: pattern being caught is a tuple built to key a memo dict, and there is no
#: structural signal for that other than what the author called it.
KEY_NAMES = ("cache_key", "key", "_key", "memo_key", "pipeline_key")

#: Calls that take a caller-built key and hand back a compiled kernel.
KEYED_FACTORY_CALLS = frozenset({"_cached_jit", "cached_jit"})

#: Anything under these prefixes COMPILES when it runs.  Used by the
#: ``rank-branch`` rule to decide whether a rank-guarded region is a region
#: that would have compiled something.
#:
#: ``jax.`` IS NOT ON THIS LIST, and that is the whole accuracy of the rule.
#: MEASURED on this tree: with a bare ``jax.`` prefix the lint reported 102
#: ``rank-branch`` findings, and the "compiling call" it had found inside a
#: ``if jax.process_index() == 0:`` region was ``jax.process_index`` — the
#: guard itself.  A rule that treats its own taint source as the compile it
#: is looking for is a rule that fires everywhere and gets switched off.
COMPILING_PREFIXES = ("jnp.", "lax.", "jax.numpy.", "jax.lax.")

#: The ``jax.`` names that DO reach XLA.  Spelled out for the reason above.
COMPILING_JAX_CALLS = frozenset({
    "jax.jit", "jax.vmap", "jax.pmap", "jax.grad", "jax.value_and_grad",
    "jax.device_put", "jax.make_array_from_process_local_data",
    "jax.make_array_from_single_device_arrays", "jax.eval_shape",
})


# ---------------------------------------------------------------------------
# THE RECIPE — carried in every message, because a lint that says only NO
# gets switched off and a lint that says what to do instead gets used.
# ---------------------------------------------------------------------------
RECIPE = {
    "rank-static-arg": (
        "CANONICALIZE like jit__multi_slice (FIX_multislice_cachekey.md §2): "
        "keep the SHARD-INVARIANT half static and pass the rank-dependent "
        "half as a DYNAMIC OPERAND.  For slice geometry that is: sizes "
        "static, offsets through lax.dynamic_slice.  In-bounds unit-stride "
        "dynamic_slice IS slice, so values stay bit-identical."),
    "rank-shape": (
        "CANONICALIZE the SHAPE, not the value: give every rank the same "
        "extent (ceil(n/world)) and carry the rank's offset as a dynamic "
        "operand; mask the tail rather than shortening it.  A ragged last "
        "chunk is a second compiled program that exactly one rank owns."),
    "rank-cache-key": (
        "REMOVE the rank slab from the key.  Key on the SHARD-INVARIANT "
        "geometry (the uniform chunk size, the global extent) so two ranks "
        "can share an entry; a key that contains lo/hi cannot, by "
        "construction, and is the written record of the divergence."),
    "rank-branch": (
        "MAKE THE COMPILE UNCONDITIONAL.  A rank that takes the empty branch "
        "compiles NOTHING while its peers compile everything — maximal "
        "asymmetry, and the one shape no key check can see.  Let the surplus "
        "rank run one fully-masked item instead of skipping the region."),
    "env-dial": (
        "DECLARE the dial in the cross-rank fingerprint: add its env name to "
        "common/jax_compile_cache.py::RANK_FINGERPRINT_ENV.  The agreement "
        "then compares it across ranks and turns the cache off LOUDLY on a "
        "mismatch, instead of each rank silently compiling a different HLO "
        "body."),
}


@dataclass(frozen=True)
class Finding:
    path: str          # repo-relative
    line: int
    rule: str
    symbol: str        # what was tainted / what the sink was
    detail: str

    def __str__(self) -> str:
        return (f"{self.path}:{self.line}: [{self.rule}] {self.detail}\n"
                f"    {RECIPE[self.rule]}")

    @property
    def site(self) -> str:
        return f"{self.path}:{self.rule}"


# ---------------------------------------------------------------------------
# THE RED LIST.  A row here is a DEBT WITH A NAME ON IT.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AllowRow:
    path: str          # repo-relative path this row covers
    rule: str
    dated: str         # YYYY-MM-DD the row was written
    reason: str        # why forcing the fix would be worse than the defect
    ledger: str        # the dated ledger file that carries the full account


#: ONE CONDITION, THREE SPELLINGS.  Every row below is the same thing: a
#: rank that draws NO WORK compiles none of the loop body its peers compile.
#: That is sibling 4 of FIX_multislice_cachekey.md §6.1 — ``local_share``'s
#: sanctioned empty share — appearing once at its carrier and twice at call
#: sites that spell the partition inline.  The other four siblings are fixed
#: on this branch and have no row here.
_EMPTY_SHARE = (
    "THE SANCTIONED EMPTY SHARE.  At world > n_items a rank draws nothing "
    "and therefore compiles nothing, while its peers compile the whole loop "
    "body — maximal hit/miss asymmetry.  The design sanctions the empty "
    "share explicitly (common/collectives.py's own note: 'P=64 on a 16-k "
    "deck leaves world-nk ranks with an empty list') and TWO cells pin it as "
    "CORRECT: test_kin_ion_padded_gvectors::test_sweep_local_k_empty_rank_"
    "keeps_the_collective_shape and test_collectives_distribution's "
    "round-robin pin.  The COLLECTIVE is already protected — item_shape is "
    "required precisely so an empty rank contributes the right shape.  The "
    "COMPILE is not, and closing it needs either a padded work list with a "
    "sentinel item every consumer must be taught to execute, or a per-"
    "consumer warm-up trace: a second, parallel notion of 'the work list'.  "
    "Either is a contract change to common.collectives plus edits in every "
    "consumer, and either invalidates the two pinning cells.  RED-LISTED "
    "rather than forced.  ")

ALLOW: tuple[AllowRow, ...] = (
    AllowRow(
        path="src/common/collectives.py", rule="rank-branch",
        dated="2026-08-10",
        reason=_EMPTY_SHARE + (
            "THE CARRIER: sweep_local_k runs per_k(ik) once per item of "
            "local_share's round-robin share, and per_k is a caller-supplied "
            "callback this module cannot see inside."),
        ledger="tests/known_failures/2026-08-10-jax-cache-contract.md"),
    AllowRow(
        path="src/gw/kin_ion_io.py", rule="rank-branch",
        dated="2026-08-10",
        reason=_EMPTY_SHARE + (
            "A CONSUMER.  The other half of this site — the RAGGED band "
            "chunk, which gave the rank holding the short chunk its own "
            "compiled FFT and its own cache key — IS fixed on this branch "
            "(rho_work_items now snaps n_bchunk to a divisor of nocc, so "
            "every chunk is the same width).  What is left is only the "
            "empty-share half, which belongs to the carrier above."),
        ledger="tests/known_failures/2026-08-10-jax-cache-contract.md"),
    AllowRow(
        path="src/psp/run_nscf.py", rule="rank-branch",
        dated="2026-08-10",
        reason=_EMPTY_SHARE + (
            "FOUND BY THIS LINT, not by the campaign: run_nscf spells the "
            "same partition inline as `if ik % n_proc != rank: continue` "
            "around the per-k Davidson solve.  Every k has the same shape, "
            "so there is no ragged-extent half here at all — the only "
            "divergence is nk < n_proc, i.e. exactly the empty share.  "
            "RAISED AS AN OWNER ROW in the ledger; it is a sixth site and "
            "was not in this lane's scope to redesign."),
        ledger="tests/known_failures/2026-08-10-jax-cache-contract.md"),
)


def allow_index() -> dict:
    return {(r.path, r.rule): r for r in ALLOW}


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------
def _dotted(node) -> str:
    """``a.b.c`` for an Attribute/Name chain, else ``""``."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ".".join(reversed(parts)) if parts else ""


def _tail(node) -> str:
    """The last component of a call target: ``jax.process_index`` -> ``...``."""
    d = _dotted(node)
    return d.rsplit(".", 1)[-1] if d else ""


class _FunctionScan(ast.NodeVisitor):
    """Intraprocedural taint over NAMES, plus the four structural rules.

    Flow-INSENSITIVE on purpose: a name that is ever bound to a
    rank-dependent value is treated as rank-dependent throughout the
    function.  The alternative (real dataflow) buys precision this lint does
    not need and costs a class of silent misses it cannot afford — the whole
    point is to be conservative in the direction of asking a human.
    """

    def __init__(self, path: str, jit_statics: dict, findings: list,
                 compiling=(), fn_name: str = "", param_taint=None,
                 call_taint=None):
        self.path = path
        self.jit_statics = jit_statics    # local jit name -> set of static params
        self.findings = findings
        #: Function names whose body reaches XLA.  Supplied by the caller so
        #: it can be a WHOLE-TREE index: `valence_density_from_kpoint` lives
        #: in ``psp`` and is called from ``gw``, and a rule that only knew
        #: about the current file would call that call site inert.
        self.compiling = set(compiling)
        self.fn_name = fn_name
        #: ``{function: {param}}`` learned by :func:`_propagate_param_taint`.
        self.param_taint = param_taint if param_taint is not None else {}
        #: Collector for the fixpoint: ``{callee: {param}}`` seen tainted here.
        self.call_taint = call_taint if call_taint is not None else {}
        self.tainted: set[str] = set(self.param_taint.get(fn_name, ()))
        #: Names holding a BARE rank id.  Not a finding on their own — see
        #: the two-level note in :meth:`_is_tainted_expr`.
        self.rank_names: set[str] = set()
        self.env_dials: set[str] = set()
        #: The enclosing function's parameters — see :func:`compiling_calls_in`.
        self.callbacks: set[str] = set()

    # -- taint recognition -------------------------------------------------
    # TWO LEVELS, and the distinction is the whole precision of this lint.
    #
    #   RANK       a bare "which process am I" — ``rank = process_index()``.
    #   PARTITION  a WORK DIVISION derived from it: a slab bound, a
    #              round-robin share, an addressable slice.
    #
    # Only PARTITION fires a rule.  MEASURED on this tree: firing on RANK
    # gave 84 findings, of which the large majority were the
    # ``if jax.process_index() == 0:`` writer/logger idiom — deliberate,
    # universal, and not what the five siblings are.  A rank-0-only region
    # that compiles IS a hazard, but it is a different conversation with
    # ~20 sites and no owner ruling behind it, and a lint that opens with
    # 20 findings nobody asked for is a lint that gets switched off.  That
    # class is recorded in this branch's ledger row as a found-state
    # observation instead of being silently dropped.
    #
    # The discriminator is arithmetic: ``rank == 0`` is a COMPARISON and
    # divides no work; ``rank * n // world``, ``i % world != rank`` and
    # ``items[rank::world]`` are how a rank's SHARE gets computed, and every
    # one of the five siblings is one of those three spellings.
    _SLAB_OPS = (ast.FloorDiv, ast.Mod, ast.Mult, ast.Add, ast.Sub, ast.Div)

    def _has_rank_source(self, node) -> bool:
        """Does ``node`` mention this process's identity at all?

        Attribute BASES do not count, for the reason given in
        :meth:`_is_tainted_expr`: ``vals.shape[1:]`` is the item shape.
        Without this exclusion the ``[1:]`` alone made every item-shape
        expression in ``collectives`` read as a rank slab.
        """
        attr_bases = {id(a.value) for a in ast.walk(node)
                      if isinstance(a, ast.Attribute)}
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Name)
                    and (sub.id in self.rank_names or sub.id in self.tainted)
                    and id(sub) not in attr_bases):
                return True
            if isinstance(sub, ast.Call):
                t = _tail(sub.func)
                if t in RANK_CALLS or t in RANK_WORLD_CALLS:
                    return True
                if t == "get" and self._env_name(sub) in RANK_ENV:
                    return True
            if isinstance(sub, ast.Attribute) and sub.attr in RANK_ATTRS:
                return True
            if isinstance(sub, ast.Subscript) and \
                    self._env_subscript(sub) in RANK_ENV:
                return True
        return False

    def _is_tainted_expr(self, node) -> bool:
        """Is ``node`` a WORK PARTITION derived from this process's rank?"""
        if node is None:
            return False
        # A TYPE or PRESENCE test is not a work division.  ``isinstance(x,
        # jax.Array)`` and ``x is None`` ask what this object IS, and the
        # answer is uniform across ranks in every LORRAX pattern; ``hi > lo``
        # asks how much of the work this rank drew.
        if isinstance(node, ast.Call) and _tail(node.func) == "isinstance":
            return False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._is_tainted_expr(node.operand)
        if isinstance(node, ast.Compare) and all(
                isinstance(o, (ast.Is, ast.IsNot)) for o in node.ops):
            return False
        # Already-partitioned names, and the carriers that ARE a partition
        # by construction.  A partitioned name reached only THROUGH an
        # attribute does not propagate: ``vals.shape[1:]`` is the item shape,
        # uniform by the collective contract that ``gather_indexed_blocks_to_
        # owner`` states in its own docstring ("every rank must call it with
        # the same blk and item shape").  The block-COUNT case that is
        # genuinely divergent is caught by ``rank-branch`` at the partition
        # itself, which is where it is actionable.
        attr_bases = {id(a.value) for a in ast.walk(node)
                      if isinstance(a, ast.Attribute)}
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Name) and sub.id in self.tainted
                    and id(sub) not in attr_bases):
                return True
            if isinstance(sub, ast.Call) and _tail(sub.func) in PARTITION_CALLS:
                return True
            if isinstance(sub, ast.Attribute) and sub.attr in RANK_ATTRS:
                return True
        if not self._has_rank_source(node):
            return False
        # A rank id is mentioned.  It is a partition only if it is being
        # used to CUT something: arithmetic, or a stride/slice.
        for sub in ast.walk(node):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, self._SLAB_OPS):
                return True
            # A SLICE cuts (``items[rank::world]``); a plain index does not
            # (``jax.local_devices()[0]`` picks a device, it partitions
            # nothing).
            if isinstance(sub, ast.Slice):
                return True
        return False

    def _is_bare_rank(self, node) -> bool:
        """``jax.process_index()`` / ``process_rank()`` / ``int(...)`` of one."""
        n = node
        while isinstance(n, ast.Call) and _tail(n.func) in ("int", "float") \
                and n.args:
            n = n.args[0]
        if isinstance(n, ast.Call) and _tail(n.func) in (
                "process_index", "process_rank"):
            return True
        return isinstance(n, ast.Name) and n.id in self.rank_names

    @staticmethod
    def _env_name(call) -> str:
        """``os.environ.get("X")`` -> ``X``."""
        if _dotted(call.func).endswith("environ.get") and call.args:
            a = call.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                return a.value
        return ""

    @staticmethod
    def _env_subscript(sub) -> str:
        if _dotted(sub.value).endswith("environ") and isinstance(
                sub.slice, ast.Constant) and isinstance(sub.slice.value, str):
            return sub.slice.value
        return ""

    def _bind(self, target, value_tainted: bool) -> None:
        """Bind ``target`` as partitioned.  NAME targets only.

        ``f["pool"] = <rank-dependent>`` does NOT make ``f`` a partition —
        it makes one entry of a dict rank-dependent.  Walking the whole
        target tainted the container, and that is how ``runtime``'s startup
        fact dict came to be reported as a rank-dependent cache key.
        """
        if not value_tainted:
            return
        stack = [target]
        while stack:
            n = stack.pop()
            if isinstance(n, ast.Name):
                self.tainted.add(n.id)
            elif isinstance(n, (ast.Tuple, ast.List)):
                stack.extend(n.elts)
            elif isinstance(n, ast.Starred):
                stack.append(n.value)
            # Attribute / Subscript targets: deliberately ignored.

    # -- statements --------------------------------------------------------
    def visit_Assign(self, node):
        # process_rank_world() -> (rank, world): taint ONLY element 0.
        # ``world`` is the SAME on every rank, and a lint that taints it
        # flags every correct partition in the tree — measured: it was 81
        # findings, almost all of them `world > 1`.
        if (isinstance(node.value, ast.Call)
                and _tail(node.value.func) in RANK_WORLD_CALLS
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)
                and len(node.targets[0].elts) == 2):
            tgt0 = node.targets[0].elts[0]
            if isinstance(tgt0, ast.Name):
                self.rank_names.add(tgt0.id)
            self.generic_visit(node)
            return
        # A BARE rank id binds into `rank_names`, not into `tainted`.
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and self._is_bare_rank(node.value)):
            self.rank_names.add(node.targets[0].id)
            self.generic_visit(node)
            return
        # ELEMENTWISE for `a, b = x, y`.  Binding the whole target from the
        # whole value is what made `world, rank = process_count(),
        # process_rank()` taint `world` as well.
        for tgt in node.targets:
            if (isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple)
                    and len(tgt.elts) == len(node.value.elts)):
                for sub_t, sub_v in zip(tgt.elts, node.value.elts):
                    if (isinstance(sub_t, ast.Name)
                            and self._is_bare_rank(sub_v)):
                        self.rank_names.add(sub_t.id)
                    else:
                        self._bind(sub_t, self._is_tainted_expr(sub_v))
            else:
                self._bind(tgt, self._is_tainted_expr(node.value))
        if self._is_tainted_expr(node.value):
            self._check_key_assignment(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        t = self._is_tainted_expr(node.value)
        self._bind(node.target, t)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._bind(node.target, self._is_tainted_expr(node.value))
        self.generic_visit(node)

    def visit_For(self, node):
        # `for item in local_share(...)` binds a rank-dependent item AND
        # gates the body on a rank-dependent trip count.
        if self._is_tainted_expr(node.iter):
            self._bind(node.target, True)
            self._flag_branch(node, node.iter, "loop trip count", region=node)
        # THE SKIP IDIOM.  `for i, x in enumerate(chunks): if (i % world) !=
        # rank: continue` gates the whole REST of the loop body, and the
        # guard's own body is one `continue` statement with nothing in it.
        # Read literally, the If is inert; read as written, it is the work
        # partition.  charge_density.py is exactly this shape.
        for stmt in node.body:
            if (isinstance(stmt, ast.If)
                    and all(isinstance(s, (ast.Continue, ast.Pass))
                            for s in stmt.body)
                    and not stmt.orelse
                    and self._is_tainted_expr(stmt.test)):
                self._flag_branch(stmt, stmt.test, "skip guard", region=node)
        self.generic_visit(node)

    def visit_If(self, node):
        if self._is_tainted_expr(node.test):
            self._flag_branch(node, node.test, "condition", region=node)
        self.generic_visit(node)

    # -- the rules ---------------------------------------------------------
    def _flag_branch(self, node, expr, what: str, region=None) -> None:
        """rank-branch: a rank-gated region that COMPILES."""
        compiles = compiling_calls_in(region if region is not None else node,
                                      self.jit_statics, self.compiling,
                                      self.callbacks)
        if not compiles:
            return
        self.findings.append(Finding(
            self.path, node.lineno, "rank-branch",
            ast.unparse(expr)[:70],
            f"a rank-dependent {what} `{ast.unparse(expr)[:70]}` guards a "
            f"region that compiles ({len(compiles)} jit-bearing call(s), "
            f"e.g. {compiles[0]}): the ranks that take the empty side "
            f"compile none of it"))

    def _check_key_assignment(self, node) -> None:
        """rank-cache-key: a tainted value inside a thing called a key."""
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not any(n == k or n.endswith(k) for n in names for k in KEY_NAMES):
            return
        bad = sorted({n.id for n in ast.walk(node.value)
                      if isinstance(n, ast.Name) and n.id in self.tainted})
        self.findings.append(Finding(
            self.path, node.lineno, "rank-cache-key", ", ".join(bad) or "?",
            f"the cache key `{names[0]}` is built from rank-dependent "
            f"value(s) {bad}: no two ranks can ever share an entry"))

    def visit_Call(self, node):
        d = _dotted(node.func)
        tail = _tail(node.func)

        # -- rank-shape ---------------------------------------------------
        # ``jnp.``/``jax.numpy.`` ONLY.  A host ``np.arange(s0, s1)`` shapes
        # nothing XLA will ever see; it becomes a compile-time surface only
        # where it crosses into jnp, and that crossing is itself a jnp call
        # this rule already watches.  MEASURED: allowing ``np.`` added four
        # findings on this tree, all of them host-side masks.
        if tail in SHAPE_CALLS and (d.startswith("jnp.")
                                    or d.startswith("jax.numpy.")):
            bad = self._tainted_names_in(node.args) | self._tainted_names_in(
                [kw.value for kw in node.keywords])
            if bad:
                self.findings.append(Finding(
                    self.path, node.lineno, "rank-shape", ", ".join(sorted(bad)),
                    f"`{d}(...)` takes rank-dependent value(s) "
                    f"{sorted(bad)} into an ARRAY SHAPE: rank r then traces "
                    f"every downstream jit at an extent no other rank traces"))

        # -- rank-static-arg ----------------------------------------------
        statics = self.jit_statics.get(tail)
        if statics:
            for i, a in enumerate(node.args):
                if i in statics["nums"] and self._is_tainted_expr(a):
                    self._flag_static(node, tail, f"positional {i}", a)
            for kw in node.keywords:
                if kw.arg in statics["names"] and self._is_tainted_expr(kw.value):
                    self._flag_static(node, tail, f"keyword {kw.arg}", kw.value)

        # -- rank-cache-key, via a keyed factory --------------------------
        if tail in KEYED_FACTORY_CALLS:
            bad = self._tainted_names_in(node.args)
            if bad:
                self.findings.append(Finding(
                    self.path, node.lineno, "rank-cache-key",
                    ", ".join(sorted(bad)),
                    f"`{d}(...)` is handed a key containing rank-dependent "
                    f"value(s) {sorted(bad)}"))

        # -- env-dial ------------------------------------------------------
        if tail in ENV_KEY_CALLS:
            self.env_dials.add(tail)

        # -- record tainted arguments for the interprocedural fixpoint ----
        # A rank slab that crosses a call boundary is still a rank slab.
        # ``_mbz_dq(..., lo=lo, hi=hi)`` is where site 1's divergence
        # actually leaves the function that computed it, and without this
        # step the shape and key rules inside ``_mbz_dq`` see two ordinary
        # integer parameters.
        if tail:
            slot = self.call_taint.setdefault(tail, set())
            for i, a in enumerate(node.args):
                if self._is_tainted_expr(a):
                    slot.add(i)
            for kw in node.keywords:
                if kw.arg and self._is_tainted_expr(kw.value):
                    slot.add(kw.arg)

        self.generic_visit(node)

    def _flag_static(self, node, name, where, arg) -> None:
        self.findings.append(Finding(
            self.path, node.lineno, "rank-static-arg", ast.unparse(arg)[:60],
            f"`{name}` declares {where} STATIC and it is given the "
            f"rank-dependent expression `{ast.unparse(arg)[:60]}`: one "
            f"compiled program per rank, one cache key per rank"))

    def _tainted_names_in(self, nodes) -> set:
        """Partitioned names used DIRECTLY, not through an attribute.

        ``vals.shape[1:]`` is the ITEM shape and is uniform across ranks
        even when ``vals`` is this rank's block; counting ``vals`` there
        reported every correct gather helper in ``collectives`` as a defect.
        """
        out = set()
        for n in nodes:
            bases = {id(a.value) for a in ast.walk(n)
                     if isinstance(a, ast.Attribute)}
            for sub in ast.walk(n):
                if (isinstance(sub, ast.Name) and sub.id in self.tainted
                        and id(sub) not in bases):
                    out.add(sub.id)
        return out


def _jit_statics_in_module(tree) -> dict:
    """Local name -> the static params of the ``jax.jit`` it is bound to.

    Covers the two spellings this tree uses: the ``@partial(jax.jit,
    static_argnums=..., static_argnames=...)`` decorator, and a module-level
    ``F = jax.jit(f, static_argnums=...)``.
    """
    out: dict = {}

    def _read(call) -> dict | None:
        d = _dotted(call.func)
        if not (d.endswith("jax.jit") or d == "jit"
                or d.endswith("functools.partial") or d == "partial"):
            return None
        if d.endswith("partial") or d == "partial":
            if not (call.args and _dotted(call.args[0]).endswith("jit")):
                return None
        nums, names = set(), set()
        for kw in call.keywords:
            if kw.arg == "static_argnums":
                for e in ast.walk(kw.value):
                    if isinstance(e, ast.Constant) and isinstance(e.value, int):
                        nums.add(e.value)
            elif kw.arg == "static_argnames":
                for e in ast.walk(kw.value):
                    if isinstance(e, ast.Constant) and isinstance(e.value, str):
                        names.add(e.value)
        return {"nums": nums, "names": names} if (nums or names) else None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    s = _read(dec)
                    if s:
                        out[node.name] = s
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            s = _read(node.value)
            if s:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        out[t.id] = s
    return out


def _bare_call_ids(node) -> set:
    """``id()`` of every Call that is a STATEMENT — its result is discarded."""
    return {id(s.value) for s in ast.walk(node)
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)}


def compiling_calls_in(node, jit_statics, compiling, callbacks=()) -> list:
    """The calls under ``node`` that reach XLA when they run.

    ``callbacks`` are the enclosing function's own PARAMETERS.  Calling one
    is the generic-carrier shape — ``sweep_local_k(ks_local, per_k)`` runs
    ``per_k(ik)`` and cannot know what that compiles — and it is exactly the
    site whose empty share is sanctioned.  Counted, because "I cannot see
    what this compiles" is a reason to ask, not a reason to stay quiet.
    """
    out = []
    # A callback whose result is THROWN AWAY is a logger, not a kernel.
    # ``print_fn(f"...")`` and ``log(...)`` are bare statements; ``per_k(ik)``
    # is consumed by ``blocks.append(...)``.  MEASURED: without this,
    # gw_output/gw_init contributed four findings whose entire evidence was
    # a progress line.
    discarded = _bare_call_ids(node)
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        d = _dotted(sub.func)
        t = _tail(sub.func)
        if any(d.startswith(p) for p in COMPILING_PREFIXES):
            out.append(d)
        elif d in COMPILING_JAX_CALLS:
            out.append(d)
        elif t in jit_statics or t in compiling:
            out.append(t)
        elif (d and d == t and t in callbacks
              and id(sub) not in discarded):
            out.append(f"{t}() [callback parameter]")
    return out


def _param_names(fn) -> set:
    """Every parameter name of ``fn`` — see :func:`compiling_calls_in`."""
    a = fn.args
    return {p.arg for p in (list(a.args) + list(a.posonlyargs)
                            + list(a.kwonlyargs))}


def _functions(tree):
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def compiling_functions(tree, seed=()) -> set:
    """Names of functions in ``tree`` that reach XLA, to a fixpoint.

    ``seed`` lets a whole-tree scan carry names learned from OTHER files, so
    ``gw.kin_ion_io``'s loop body is recognised as compiling even though the
    kernel it calls (``psp.get_DFT_mtxels.valence_density_from_kpoint``) is
    two packages away.  Name-keyed and therefore collision-prone in the
    conservative direction: a shared name makes the lint MORE likely to ask
    a human, never less.
    """
    known = set(seed)
    fns = _functions(tree)
    statics = _jit_statics_in_module(tree)
    known |= set(statics)
    for _ in range(4):                       # converges in 2 on this tree
        grew = False
        for fn in fns:
            if fn.name in known:
                continue
            if compiling_calls_in(fn, statics, known):
                known.add(fn.name)
                grew = True
        if not grew:
            break
    return known


def _propagate_param_taint(tree, path, statics, compiling, rounds: int = 3):
    """``{function: {tainted param names}}`` to a fixpoint within the module."""
    fns = _functions(tree)
    by_name = {}
    for fn in fns:
        by_name.setdefault(fn.name, fn)
    param_taint: dict = {}
    for _ in range(rounds):
        call_taint: dict = {}
        sink = []                            # findings discarded on these passes
        for fn in fns:
            scan = _FunctionScan(path, statics, sink, compiling, fn.name,
                                 param_taint, call_taint)
            scan.callbacks = _param_names(fn)
            for stmt in fn.body:
                scan.visit(stmt)
        grew = False
        for callee, slots in call_taint.items():
            fn = by_name.get(callee)
            if fn is None:
                continue
            names = [a.arg for a in fn.args.args]
            kwonly = [a.arg for a in fn.args.kwonlyargs]
            got = set()
            for s in slots:
                if isinstance(s, int) and s < len(names):
                    got.add(names[s])
                elif isinstance(s, str) and (s in names or s in kwonly):
                    got.add(s)
            cur = param_taint.setdefault(callee, set())
            if not got <= cur:
                cur |= got
                grew = True
        if not grew:
            break
    return param_taint


def scan_source(source: str, path: str, compiling_seed=()) -> list:
    """Every :class:`Finding` in one file's text.  Pure."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Finding(path, getattr(exc, "lineno", 1) or 1, "rank-shape",
                        "<unparseable>", f"could not parse: {exc}")]
    statics = _jit_statics_in_module(tree)
    compiling = compiling_functions(tree, compiling_seed)
    param_taint = _propagate_param_taint(tree, path, statics, compiling)
    findings: list = []
    for fn in _functions(tree):
        scan = _FunctionScan(path, statics, findings, compiling, fn.name,
                             param_taint, {})
        scan.callbacks = _param_names(fn)
        for stmt in fn.body:
            scan.visit(stmt)
    # One function can be visited twice (a nested def is walked by its
    # parent too); de-duplicate on the exact finding.
    return sorted(set(findings), key=lambda f: (f.line, f.rule, f.symbol))


# ---------------------------------------------------------------------------
# THE ``env-dial`` RULE — a whole-tree rule, not a per-file one
# ---------------------------------------------------------------------------
# The other four rules are properties of one function.  This one is a
# property of a RELATION between two files, so it cannot be expressed as a
# visitor: the defect is that a dial ``ffi_dial_key`` folds into a kernel
# factory's cache key is NOT folded into the cross-rank fingerprint that
# ``jax_compile_cache`` compares during the agreement.
#
# Both halves are read out of the AST rather than imported: this lint must
# run with no jax, no FFI .so and no backend, which is exactly the situation
# in which someone is most likely to be adding a dial.

#: Where the fingerprint declares the per-process env it covers.
FINGERPRINT_FILE = "src/common/jax_compile_cache.py"
FINGERPRINT_NAME = "RANK_FINGERPRINT_ENV"

#: Where the FFI dials are declared.
DIAL_FILE = "src/ffi/__init__.py"
DIAL_NAME = "FFI_DIAL_ENV"


def _string_tuple(path: Path, name: str) -> set:
    """The set of string constants in a module-level ``NAME = (...)``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return {e.value for e in ast.walk(node.value)
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    return set()


def env_dial_findings(root: Path = None) -> list:
    """Dials that reach a kernel-factory cache key but are not fingerprinted."""
    root = Path(root or REPO_ROOT)
    dial_path = root / DIAL_FILE
    if not dial_path.is_file():
        return []
    dials = _string_tuple(dial_path, DIAL_NAME)
    if not dials:
        # The declaration itself is missing.  Report AT the file, because a
        # dial set nobody declares is exactly the found state this rule was
        # written for — silence here would be the bug.
        return [Finding(
            DIAL_FILE, 1, "env-dial", DIAL_NAME,
            f"`{DIAL_NAME}` is not declared in {DIAL_FILE}, so there is no "
            f"statement anywhere of WHICH per-process dials change the "
            f"emitted HLO body.  ffi_dial_key() folds them into four GW "
            f"kernel-factory cache keys; nothing compares them across ranks")]
    covered = _string_tuple(root / FINGERPRINT_FILE, FINGERPRINT_NAME)
    missing = sorted(dials - covered)
    if not missing:
        return []
    return [Finding(
        DIAL_FILE, 1, "env-dial", ", ".join(missing),
        f"per-process dial(s) {missing} reach a kernel factory's cache key "
        f"(ffi_dial_key -> ppm_tau_kernel / cohsex_sigma / w_isdf) but are "
        f"absent from {FINGERPRINT_FILE}::{FINGERPRINT_NAME}: a rank whose "
        f"dial differs emits a different HLO BODY and nothing notices")]


def iter_sources(root: Path = None, roots=SCAN_ROOTS):
    """Every ``.py`` under the scan roots, repo-relative, sorted."""
    root = Path(root or REPO_ROOT)
    out = []
    for r in roots:
        base = root / r
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            # A service's OWN tests are where red twins live on purpose.
            if "/tests/" in f"/{rel}" or rel.startswith("tests/"):
                continue
            if "__pycache__" in rel:
                continue
            out.append((rel, p))
    return out


def tree_compiling_index(root: Path = None, roots=SCAN_ROOTS) -> set:
    """Every function name in the tree that reaches XLA.  Two passes.

    The second pass is not decoration: pass 1 learns ``local_ifftn3``
    compiles, and only then can pass 2 conclude that
    ``valence_density_from_kpoint`` does, and pass 3 that
    ``build_valence_density_distributed``'s loop body does.  Three packages,
    one chain, and the middle link is what a per-file lint cannot see.
    """
    trees = []
    for rel, p in iter_sources(root, roots):
        try:
            trees.append(ast.parse(p.read_text(encoding="utf-8")))
        except (OSError, SyntaxError):
            continue
    known: set = set()
    for _ in range(3):
        before = len(known)
        for tree in trees:
            known |= compiling_functions(tree, known)
        if len(known) == before:
            break
    return known


def scan_tree(root: Path = None, roots=SCAN_ROOTS) -> list:
    root = Path(root or REPO_ROOT)
    seed = tree_compiling_index(root, roots)
    findings = []
    for rel, p in iter_sources(root, roots):
        try:
            findings.append(scan_source(p.read_text(encoding="utf-8"), rel, seed))
        except OSError:
            continue
    flat = [f for group in findings for f in group]
    flat.extend(env_dial_findings(root))
    return sorted(flat, key=lambda f: (f.path, f.line, f.rule))


def partition(findings) -> tuple:
    """``(unallowed, allowed)`` against :data:`ALLOW`."""
    idx = allow_index()
    bad, ok = [], []
    for f in findings:
        (ok if (f.path, f.rule) in idx else bad).append(f)
    return bad, ok


def stale_allow_rows(findings) -> list:
    """Red-list rows that no longer match anything — a debt already paid.

    A red list that keeps rows for sites that were since fixed is a red list
    nobody trusts, and it is how a real regression hides behind a stale row.
    """
    seen = {(f.path, f.rule) for f in findings}
    return [r for r in ALLOW if (r.path, r.rule) not in seen]


def format_report(findings) -> str:
    if not findings:
        return "clean"
    return "\n".join(str(f) for f in findings)
