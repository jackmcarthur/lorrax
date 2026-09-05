"""A ``PartitionSpec`` that is LEGAL for the extents actually in hand.

WHY THIS EXISTS
---------------
LORRAX has two conventions for the ISDF centroid extent ``n_mu`` and they
disagree:

* ``bse.bse_io`` **pads** it — ``runtime.padding.padded_mu_extent(n_mu,
  px*py)`` — before it shards anything, so its tiles always divide the mesh.
* ``bandstructure.bse_setup`` and ``bse.vq_interp`` shard the **raw** count.

``n_mu`` is a k-means output, so the raw count divides the mesh axis only by
luck.  The MoS2 4x4 deck's 800-centroid set has 785 unique centroids;
785 = 5 x 157, which divides neither 4 nor 8.  Every ``exciton_bands`` run on
that deck therefore died at **every** P > 1 with

    IndivisibleError: Sharding NamedSharding(..., spec=P(None, None, None,'y'))
    implies that array axis 3 is partitioned 4 times, but the dimension size
    is 785   (full shape: (208, 4, 2, 785))

— ``bandstructure/bse_setup.py::_make_bundle``, job 7882476 cell exb16s at
P=16 and, with a second offender (a 4-band window over an 8-wide mesh), cell
exb64s at P=64.  It went unnoticed because the published 12x12 reference used
n_mu = 640 on a 4x4 mesh (640/4 = 160) and the 6x6 fixture n_mu = 1496 on 4x4
(374): both divisible by accident.  The same arithmetic refuses
``vq_interp``'s (mu, nu) face shardings, and refuses its q-batch sharding
outright on any deck with fewer BZ q-points than devices (a 4x4x1 grid has
nq = 16; at P = 64 there is nothing to split).

WHAT THIS MODULE DOES
---------------------
:func:`legal_spec` returns the requested spec unchanged when every sharded
axis divides evenly — the SAME object, so nothing about a currently-working
run moves, not one byte of HLO — and otherwise returns it with the offending
mesh axes dropped, i.e. that array dimension replicated (or, for a tuple
entry, only PARTLY replicated; see "partial retention" below).

This is a **placement** decision, not an algorithmic one.  Replicating a
dimension changes where operands live; it does not change any contraction,
any reduction order, or any collective the jitted body issues.  Values are
unaffected.

WHAT IT COSTS, AND WHAT IT IS NOT
---------------------------------
Per-device memory for that array rises by the dropped factor, so this is a
correctness floor, not a scaling strategy.  It is ANNOUNCED once per distinct
(site, spec, shape) — with the per-device byte cost spelled out — so the
degradation can never be silent.  The failure it replaces was loud; the
success it produces must not be quietly expensive.

PARTIAL RETENTION (tuple entries)
---------------------------------
``P(('x','y'), ...)`` flattens BOTH mesh axes onto one array dimension, so the
divisor is the product.  When the product does not divide but a sub-product
does, dropping the whole entry replicates ``px*py`` ways where ``px`` ways
were available for free.  The exb wall is exactly that case: a 32-wide q-batch
on an 8x8 mesh is indivisible by 64, divisible by 8.  :func:`legal_spec`
therefore keeps the largest order-preserving subset of the tuple whose product
divides — the same argument the single-axis case has always made ("drop THAT
axis and only that one"), applied inside a tuple.  Never more replication than
before, often 8x less.

THE DURABLE FIX IS TO NOT NEED THIS AT ALL
------------------------------------------
:func:`runtime.padding.padded_axis` is the producer half and the one that should
be reached for first: pad the extent so the requested spec is legal, instead
of degrading the spec so the extent fits.

* ``bandstructure.bse_setup`` now pads its **q-batch** with it —
  ``padded_axis(batch_size, mesh, spec=P(('x','y')), axis=0)``.  Measured on the MoS2 4x4
  deck, 91-Q exciton path: the 32-wide batch could not split 64 devices, so
  every device ran all 32 ``eigh`` instead of its own share, and
  ``htransform_psi_cQ`` cost 866.5 s at P=64 against 105.7 s at P=16 (jobs
  7882533 / 7882569) — 64 devices SLOWER than 16.
* ``mu`` is the remaining offender: ``bse.bse_io`` pads it
  (``runtime.padding.padded_mu_extent(n_mu, px*py)``) and ``bse.vq_interp``
  does not.  At the converged 12x12 / n_mu = 2412 reference on an 8x8 mesh
  (2412 % 8 = 4) ``build_cq``'s ``P(None,'x','y')`` degrades on BOTH face
  axes and this resolver replicates a 13.4 GB (12.484 GiB) C_q per device — a
  stop sign, not a shrug.  Padding n_mu to 2416 = round_up(2412, 8) makes the
  face legal and the same tile costs 0.196 GiB (210 MB) per device, a factor
  64 down.  Single-axis entries have no partial
  retention to fall back on (a NamedSharding cannot split an array axis by a
  sub-factor of ONE mesh axis), so for C_q padding is the only lever.

Gated by ``tests/test_sharding_fit.py``, whose control establishes that jax
really does refuse the raw spec (a fitter for a non-problem would pass every
other assertion in that file and mean nothing).
"""
from __future__ import annotations

from itertools import combinations
from runtime.padding import spec_divisor

__all__ = ["legal_spec", "fit_sharding", "shard_factor"]

_ANNOUNCED: set = set()


def shard_factor(mesh, entry) -> int:
    """How many shards a single ``PartitionSpec`` entry asks for.

    ``None`` -> 1; ``'x'`` -> ``mesh.shape['x']``; ``('x','y')`` -> the
    product, because a tuple entry flattens those mesh axes onto one array
    axis.
    """
    return spec_divisor(mesh, (entry,), axis=0)


def _largest_divisible_subset(mesh, names, dim: int):
    """Longest order-preserving subset of ``names`` whose product divides.

    Ties (equal length) are broken by the earliest-starting subset, so the
    choice is deterministic and does not depend on dict/set iteration order.
    Under the square-mesh ruling (repo ``docs/architecture/decisions.md``
    2026-08-01) every mesh axis has the SAME extent, so equal-length subsets
    have equal products and no larger-product tie-break exists to make — the
    rectangular-mesh tie-break this function used to carry was deleted with
    that ruling.  Returns ``()`` when nothing divides.
    """
    for k in range(len(names) - 1, 0, -1):
        for sub in combinations(range(len(names)), k):
            picked = tuple(names[i] for i in sub)
            p = shard_factor(mesh, picked)
            from runtime.padding import padded_axis
            if padded_axis(
                    dim, p, name="candidate sharding subset").carrier == dim:
                return picked
    return ()


def legal_spec(mesh, spec, shape, what: str = "", print_fn=None,
               itemsize: int = 16):
    """``spec``, with every mesh axis dropped that cannot divide its dimension.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
    spec : jax.sharding.PartitionSpec
    shape : tuple[int, ...]
        The GLOBAL shape the spec will be applied to.
    what : str
        Call-site label used in the one-time announcement.
    itemsize : int
        Bytes per element, used ONLY to quantify the announcement.  Defaults
        to 16 (complex128), which is what every array routed through this
        module is today; the announcement labels the assumption so a float64
        site reads 2x high rather than misleading silently.

    Returns the input ``spec`` object itself when nothing had to change.
    """
    from jax.sharding import PartitionSpec as P

    entries = list(spec)
    out, dropped = [], []
    for i, e in enumerate(entries):
        n = shard_factor(mesh, e)
        if n == 1:
            out.append(e)
            continue
        dim = int(shape[i]) if i < len(shape) else 0
        from runtime.padding import padded_axis
        if padded_axis(dim, n, name="candidate sharding spec").carrier == dim:
            out.append(e)
            continue
        # Tuple entries flatten several mesh axes onto ONE array axis, so a
        # sub-product may still divide (32 on an 8x8 mesh: 64 does not, 8
        # does).  Keep the most of it that is legal; a bare string entry has
        # no sub-product and falls through to None as before.
        names = () if isinstance(e, str) else tuple(e)
        keep = _largest_divisible_subset(mesh, names, dim) if names else ()
        if keep:
            kept = keep[0] if len(keep) == 1 else keep
            out.append(kept)
            dropped.append((i, e, dim, n, shard_factor(mesh, kept)))
        else:
            out.append(None)
            dropped.append((i, e, dim, n, 1))
    if not dropped:
        return spec

    key = (what, tuple(str(e) for e in entries), tuple(int(s) for s in shape))
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        if print_fn is None:
            print_fn = _rank0_print
        det = "; ".join(
            f"axis {i} ({e}): {dim} % {n} != 0"
            + (f", kept {k}-way" if k > 1 else "")
            for i, e, dim, n, k in dropped)
        # Quantify it.  A generic "memory rises by the dropped factor" reads
        # the same at 150 MB and at 13.4 GB; the second is a stop sign.
        elems = 1
        for s in shape:
            elems *= int(s)
        want = 1
        for e in entries:
            want *= shard_factor(mesh, e)
        got = 1
        for e in out:
            got *= shard_factor(mesh, e)
        gib = elems * int(itemsize) / float(got) / 1024 ** 3
        print_fn(
            f"  [sharding_fit] {what} {tuple(int(s) for s in shape)}: "
            f"{P(*entries)} -> {P(*out)}  [{det}].  Placement only — the "
            f"arithmetic and every collective are unchanged.  Per-device "
            f"residency for this array is now {gib:.3f} GiB "
            f"(x{want // got if got else want} the intended "
            f"{elems * int(itemsize) / float(want) / 1024 ** 3:.3f} GiB; "
            f"assumes {itemsize} B/element).  Durable fix: pad the extent — "
            "runtime.padding.padded_axis with the intended PartitionSpec.")
    return P(*out)


def fit_sharding(mesh, spec, shape, what: str = "", print_fn=None,
                 itemsize: int = 16):
    """:class:`NamedSharding` over ``mesh`` for ``shape``, made legal."""
    from jax.sharding import NamedSharding

    return NamedSharding(
        mesh, legal_spec(mesh, spec, shape, what, print_fn, itemsize))


def _rank0_print(msg):
    try:
        import jax

        if jax.process_index() != 0:
            return
    except Exception:
        pass
    print(msg, flush=True)
