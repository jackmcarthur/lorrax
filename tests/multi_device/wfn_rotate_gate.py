"""Certify the sharded-U rotation of the four ISDF-centroid psi copies.

``gw.wavefunction_bundle.rotate_wavefunctions`` used to contract a
REPLICATED ``U`` (nk, nb, nb) -- 6.5 MB/rank per k at nb=640, and the
(nk, nb, nb) object that reaches 9.2 GB at nb=2000/nk=144.  It now shards
U with the contracted index m on the mesh axis each psi copy's mu does NOT
own, so no rank holds a full (nb, nb), psi never moves, and the band sum
is a psum along one mesh axis.

Checks, strongest first:

  1. PER-SHARD AGREEMENT with an explicit host rotation, 1e-12 relative,
     on all four copies.  The host reference uses the COLUMN convention
     psi~_n = sum_m U[m,n] psi_m.
  2. CONVENTION PIN / negative control.  For a unitary Q, Q^T is also
     unitary and mixes only within the occupied block, so norms,
     orthonormality and occupied-block invariance all survive a transposed
     U.  Check 1 is therefore also run against the transposed host
     rotation, and THAT one must disagree.
  3. SHARDINGS PRESERVED.  Each output carries its input's PSI_*_SPEC.
     The docstring promises it and every downstream consumer assumes it.
  4. INACTIVE BANDS UNTOUCHED, bit-identical -- no Sigma exists for them.
  5. U IS NEVER REPLICATED.  Per-rank U bytes measured off the addressable
     shards, against the replicated bytes the old path held.
  6. THE REDUCTION IS ONE MESH AXIS, not a global collective.  Every
     collective in the compiled module is listed with its result bytes and
     replica-group size; the group size must be <= max(px, py) and no
     collective may carry more than one psi output shard.
  7. INDIVISIBLE ACTIVE WINDOW.  nb_active is a band window (b3 - b0), not
     the loader's mesh-divisible extent, so it need not divide px or py.
     Run for both the device-U (pad inside the jit) and host-U (pad on the
     host) paths.
  8. U LAYOUT INVARIANCE.  Replicated, ``band_rotation_spec`` and a HOST
     numpy U -- what sc_iteration's k-star broadcast hands over on a
     reduced k-set -- must all give the same rotated bundle.

Env: WR_NK, WR_NB, WR_NS, WR_NMU, WR_NACT, WR_NOCC.
"""
import functools
import math
import os
import re
import sys

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402

from common.collectives import (process_rank, process_count,   # noqa: E402
                                resolve_mesh)
from runtime.padding import round_up                           # noqa: E402
from gw.qsgw_density import band_rotation_spec                 # noqa: E402
from gw.wavefunction_bundle import (                           # noqa: E402
    BandSlices, Wavefunctions, band_mix_spec, rotate_wavefunctions,
    _rotate_kernel, PSI_XN_SPEC, PSI_XR_SPEC, PSI_YR_SPEC, PSI_YN_SPEC)

NK = int(os.environ.get("WR_NK", "4"))
NB = int(os.environ.get("WR_NB", "16"))
NS = int(os.environ.get("WR_NS", "2"))
NMU = int(os.environ.get("WR_NMU", "24"))
NACT = int(os.environ.get("WR_NACT", "8"))
NOCC = int(os.environ.get("WR_NOCC", "4"))
RTOL = 1e-12

#: (field, spec, band array axis, transpose from the host (k, n, s, mu))
_COPIES = (
    ('psi_xn', PSI_XN_SPEC, 3, (0, 2, 3, 1)),
    ('psi_xr', PSI_XR_SPEC, 1, (0, 1, 2, 3)),
    ('psi_yr', PSI_YR_SPEC, 1, (0, 1, 2, 3)),
    ('psi_yn', PSI_YN_SPEC, 3, (0, 2, 3, 1)),
)


def _haar(rng, n):
    """A Haar-ish unitary: QR of a complex Gaussian."""
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    return Q * (np.diagonal(R) / np.abs(np.diagonal(R)))[None, :]


def _spec_tuple(spec, ndim):
    t = list(tuple(spec))
    return tuple(t + [None] * (ndim - len(t)))


def _put(host, mesh, spec):
    return jax.make_array_from_callback(
        host.shape, NamedSharding(mesh, spec), lambda idx: host[idx])


def _addressable_bytes(arr):
    return sum(int(np.asarray(s.data).nbytes) for s in arr.addressable_shards)


def _build_bundle(psi_host, enk_host, mesh, slices, efermi):
    """The four device copies of one host psi (nk, nb, ns, mu)."""
    arrays = {field: _put(np.ascontiguousarray(psi_host.transpose(tr)),
                          mesh, spec)
              for field, spec, _ax, tr in _COPIES}
    occ = (enk_host <= efermi).astype(np.float64)
    rep2 = P(None, None)
    return Wavefunctions(
        enk=_put(enk_host, mesh, rep2), occ=_put(occ, mesh, rep2),
        slices=slices, **arrays)


def _worst_shard_rel(arr, ref_host):
    """max over addressable shards of |got - ref[shard]| / max|ref[shard]|."""
    worst = 0.0
    for sh in arr.addressable_shards:
        got = np.asarray(sh.data)
        want = ref_host[sh.index]
        worst = max(worst, float(np.abs(got - want).max())
                    / max(float(np.abs(want).max()), 1e-300))
    return worst


def _host_rotate(psi_host, U, nact, transposed=False):
    """psi~ with the active block rotated, on the host.

    COLUMN convention: psi~_n = sum_m U[k,m,n] psi_m.  ``transposed`` is
    the row form -- the negative control, which every invariance check in
    this file survives and which is why check 1 alone would prove nothing.
    """
    out = psi_host.copy()
    eq = 'knm,kmsu->knsu' if transposed else 'kmn,kmsu->knsu'
    out[:, :nact] = np.einsum(eq, U, psi_host[:, :nact], optimize=True)
    return out


def _all_copies_rel(bundle, ref_host):
    return max(_worst_shard_rel(getattr(bundle, field),
                                np.ascontiguousarray(ref_host.transpose(tr)))
               for field, _spec, _ax, tr in _COPIES)


def _worst_pair_rel(a, b):
    """Per-shard relative difference between two arrays of the same layout.

    Neither can be gathered to the host at P>1, so the comparison is shard
    by shard, matched on the global index each shard covers.
    """
    ref = {str(sh.index): np.asarray(sh.data) for sh in b.addressable_shards}
    worst = 0.0
    for sh in a.addressable_shards:
        want = ref[str(sh.index)]
        got = np.asarray(sh.data)
        worst = max(worst, float(np.abs(got - want).max())
                    / max(float(np.abs(want).max()), 1e-300))
    return worst


def _make_replicated_kernel(nact, mesh):
    """THE PATH THIS CHANGE REPLACES, verbatim in structure.

    Four einsums against a REPLICATED U, then a dynamic-update-slice back
    into the full psi.  Kept in the gate rather than in the module so the
    1e-12 claim is against the code that actually shipped, not against a
    paraphrase of it, and so the two can be compiled side by side for the
    residency A/B.

    ``out_shardings`` pins the four outputs at the canonical specs, which
    the old path got from sharding propagation through the
    dynamic-update-slice.  Pinning them changes no arithmetic and makes the
    per-shard comparison well defined even if propagation were to pick
    something else.
    """
    outs = tuple(NamedSharding(mesh, spec) for _f, spec, _a, _t in _COPIES)

    @functools.partial(jax.jit, out_shardings=outs)
    def fn(psi_xn, psi_xr, psi_yr, psi_yn, U):
        xn = jnp.einsum('ksum,kmn->ksun',
                        jax.lax.slice_in_dim(psi_xn, 0, nact, axis=3), U,
                        optimize=True)
        xr = jnp.einsum('kmn,kmsu->knsu', U,
                        jax.lax.slice_in_dim(psi_xr, 0, nact, axis=1),
                        optimize=True)
        yr = jnp.einsum('kmn,kmsu->knsu', U,
                        jax.lax.slice_in_dim(psi_yr, 0, nact, axis=1),
                        optimize=True)
        yn = jnp.einsum('ksum,kmn->ksun',
                        jax.lax.slice_in_dim(psi_yn, 0, nact, axis=3), U,
                        optimize=True)
        return (jax.lax.dynamic_update_slice_in_dim(psi_xn, xn, 0, axis=3),
                jax.lax.dynamic_update_slice_in_dim(psi_xr, xr, 0, axis=1),
                jax.lax.dynamic_update_slice_in_dim(psi_yr, yr, 0, axis=1),
                jax.lax.dynamic_update_slice_in_dim(psi_yn, yn, 0, axis=3))
    return fn


def _mem(compiled):
    """``(argument, temp, output)`` bytes per rank, or None if unavailable."""
    try:
        ma = compiled.memory_analysis()
    except Exception:
        return None
    if ma is None:
        return None
    return (int(getattr(ma, "argument_size_in_bytes", 0)),
            int(getattr(ma, "temp_size_in_bytes", 0)),
            int(getattr(ma, "output_size_in_bytes", 0)))


#: Collectives that carry ``replica_groups``, i.e. the ones whose SPAN is a
#: statement about how much of the mesh a reduction touches.  A
#: ``collective-permute`` carries ``source_target_pairs`` instead and is
#: counted separately: it is a device-order shuffle, not a reduction, so the
#: question to ask of it is its payload, not its group.
_GROUPED = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all")
_KINDS = _GROUPED + ("collective-permute",)


def _group_size(line):
    """Replica-group size from an HLO line; -1 when unparseable.

    Two spellings in the wild: the explicit ``replica_groups={{0,1},{2,3}}``
    and the iota form ``replica_groups=[2,2]<=[4]`` (n_groups, group_size).
    """
    m = re.search(r'replica_groups=\{\{(.*?)\}', line)
    if m:
        return len([v for v in m.group(1).split(',') if v.strip()])
    m = re.search(r'replica_groups=\[(\d+),(\d+)\]', line)
    if m:
        return int(m.group(2))
    return -1


def _collective_census(txt, p0, tag, show_permute=False):
    """Print every collective; return a summary dict.

    ``worst_b`` is the worst RESULT bytes over all collectives, ``worst_g``
    the worst replica-group size over the grouped kinds, ``unparsed`` the
    number of grouped collectives whose group could not be read (a census
    that cannot read its own instrument proves nothing, so the caller fails
    on it rather than reporting a vacuous pass).
    """
    out = dict(worst_b=0, worst_g=0, n=0, unparsed=0, perm_b=0,
               counts={k: 0 for k in _KINDS})
    for line in txt.splitlines():
        kind = next((k for k in _KINDS
                     if f" {k}(" in line or f"= {k}" in line), None)
        if kind is None:
            continue
        out['n'] += 1
        out['counts'][kind] += 1
        m = re.search(r'c(?:64|128)\[[\d,]+\]', line)
        nbytes = 0
        if m:
            width = 16 if '128' in m.group(0) else 8
            nbytes = width
            for d in re.findall(r'\d+', m.group(0))[1:]:
                nbytes *= int(d)
        gsz = _group_size(line)
        if kind in _GROUPED:
            if gsz > 0:
                out['worst_g'] = max(out['worst_g'], gsz)
            else:
                out['unparsed'] += 1
        else:
            out['perm_b'] = max(out['perm_b'], nbytes)
        out['worst_b'] = max(out['worst_b'], nbytes)
        p0(f"[wr]   {tag} {kind:18s} {m.group(0) if m else '?':26s} "
           f"{nbytes / 2**20:9.4f} MiB  group={gsz}")
        if show_permute and kind == "collective-permute":
            p0(f"[wr]     {line.strip()[:240]}")
    return out


def main():
    rank, world = process_rank(), process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    px, py = (int(s) for s in mesh.devices.shape)
    p0(f"[wr] world={world} mesh=({px},{py}) nk={NK} nb={NB} ns={NS} "
       f"nmu={NMU} nact={NACT} nocc={NOCC}")

    rng = np.random.default_rng(20260805)
    psi_host = (rng.standard_normal((NK, NB, NS, NMU))
                + 1j * rng.standard_normal((NK, NB, NS, NMU)))
    enk_host = np.sort(rng.standard_normal((NK, NB)), axis=1)
    efermi = float(enk_host[:, NOCC - 1].max())
    slices = BandSlices.from_band_edges(0, 0, NOCC, NACT, NB)
    wfns = _build_bundle(psi_host, enk_host, mesh, slices, efermi)

    U = np.stack([_haar(rng, NACT) for _ in range(NK)])
    enk_new = enk_host[:, :NACT] + 0.1
    U_rep = _put(U, mesh, P(None, None, None))
    ref = _host_rotate(psi_host, U, NACT)
    ref_T = _host_rotate(psi_host, U, NACT, transposed=True)

    out = rotate_wavefunctions(
        wfns, U_rep, enk_active_new=enk_new, efermi=efermi, mesh_xy=mesh,
        active_slice=slices.sigma)

    # ---- 1. per-shard agreement, all four copies ------------------------
    worst = _all_copies_rel(out, ref)
    ok1 = worst <= RTOL
    p0(f"[wr] 1. four copies vs host rotation   per-shard rel {worst:.3e}  "
       f"{'PASS' if ok1 else 'FAIL'}")

    # ---- 1b. vs the REPLICATED-U path this change replaces --------------
    rep_kernel = _make_replicated_kernel(NACT, mesh)
    with mesh:
        rep_out = rep_kernel(wfns.psi_xn, wfns.psi_xr, wfns.psi_yr,
                             wfns.psi_yn, U_rep)
    worst_rep = max(_worst_pair_rel(getattr(out, field), rep_out[i])
                    for i, (field, _s, _a, _t) in enumerate(_COPIES))
    ok1b = worst_rep <= RTOL
    p0(f"[wr] 1b. vs the replicated-U path      per-shard rel "
       f"{worst_rep:.3e}  {'PASS' if ok1b else 'FAIL'}")

    # ---- 2. convention pin: the transpose must DISAGREE -----------------
    worst_T = _all_copies_rel(out, ref_T)
    ok2 = worst_T > 1e-6
    p0(f"[wr] 2. transposed-U reference must DIFFER  rel {worst_T:.3e}  "
       f"{'PASS' if ok2 else 'FAIL'}")

    # ---- 3. the four shardings are preserved ----------------------------
    ok3 = True
    for field, spec, _ax, _tr in _COPIES:
        arr = getattr(out, field)
        got = _spec_tuple(arr.sharding.spec, arr.ndim)
        want = _spec_tuple(spec, arr.ndim)
        if got != want:
            ok3 = False
            p0(f"[wr]    {field}: spec {got} != {want}")
    p0(f"[wr] 3. four psi shardings preserved  {'PASS' if ok3 else 'FAIL'}")

    # ---- 4. bands outside the active window are BIT-IDENTICAL -----------
    ok4 = True
    for field, _spec, band_axis, tr in _COPIES:
        sl = [slice(None)] * 4
        sl[band_axis] = slice(NACT, NB)
        src_h = np.ascontiguousarray(psi_host.transpose(tr))
        for sh in getattr(out, field).addressable_shards:
            if not np.array_equal(np.asarray(sh.data)[tuple(sl)],
                                  src_h[sh.index][tuple(sl)]):
                ok4 = False
    enk_out = np.asarray(jax.device_get(out.enk))
    ok4 = (ok4 and np.array_equal(enk_out[:, NACT:], enk_host[:, NACT:])
           and np.array_equal(enk_out[:, :NACT], enk_new))
    p0(f"[wr] 4. inactive psi bands + enk bit-identical  "
       f"{'PASS' if ok4 else 'FAIL'}")

    # ---- 5. U residency, before vs after --------------------------------
    # BEFORE: every rank held the whole (nk, nb_act, nb_act) and fed it to
    # four einsums.  AFTER: two band_mix_spec layouts, one per contraction
    # axis, measured off the addressable shards.
    if NACT % px or NACT % py:
        raise SystemExit(
            f"[wr] WR_NACT={NACT} must divide the mesh {px}x{py} for the "
            f"residency measurement; the PADDED path is check 7's job.")
    with mesh:
        Ux = jax.lax.with_sharding_constraint(
            U_rep, NamedSharding(mesh, band_mix_spec('x')))
        Uy = jax.lax.with_sharding_constraint(
            U_rep, NamedSharding(mesh, band_mix_spec('y')))
    b_before = _addressable_bytes(U_rep)
    b_x, b_y = _addressable_bytes(Ux), _addressable_bytes(Uy)
    x_shape = tuple(np.asarray(Ux.addressable_shards[0].data).shape)
    y_shape = tuple(np.asarray(Uy.addressable_shards[0].data).shape)
    # THE CRITERION IS PER-OPERAND, NOT THE SUM.  Each einsum sees ONE of
    # the two layouts, so what the scaling target forbids -- a full
    # (nb, nb) required on one rank -- is bounded by the LARGER of the two.
    # Their sum only beats the replicated bytes when px*py > px+py, i.e.
    # not on a 2x2 mesh; the operand bound improves on every mesh with an
    # axis > 1.
    ok5 = (x_shape[1] == NACT // px and y_shape[1] == NACT // py
           and max(b_x, b_y) <= b_before
           and (max(px, py) == 1 or max(b_x, b_y) < b_before))
    p0(f"[wr] 5. per-rank U bytes  replicated {b_before} -> m-on-x {b_x} "
       f"{x_shape} + m-on-y {b_y} {y_shape}; worst operand "
       f"{max(b_x, b_y)} ({b_before / max(max(b_x, b_y), 1):.3f}x), "
       f"sum {b_x + b_y} ({b_before / max(b_x + b_y, 1):.3f}x)  "
       f"{'PASS' if ok5 else 'FAIL'}")
    _mib = 640 * 640 * 16 / 2**20
    p0(f"[wr] 5b. model per k at nb=640: replicated {_mib:.3f} MiB/rank; "
       f"worst operand 8x8 {_mib / 8:.3f} 16x16 {_mib / 16:.3f} MiB; "
       f"sum 8x8 {_mib / 4:.3f} 16x16 {_mib / 8:.3f} MiB")

    # ---- 5c. the same question asked of the compiled executables --------
    nb_pad = round_up(NACT, math.lcm(px, py))
    kern = _rotate_kernel(mesh, 0, NACT, nb_pad)
    psi_args = (wfns.psi_xn, wfns.psi_xr, wfns.psi_yr, wfns.psi_yn)
    with mesh:
        c_new = kern.lower(*psi_args, U_rep).compile()
        c_old = rep_kernel.lower(*psi_args, U_rep).compile()
    m_new, m_old = _mem(c_new), _mem(c_old)
    if m_new and m_old:
        p0(f"[wr] 5c. compiled per-rank (arg, temp, out) bytes: "
           f"replicated-U {m_old} -> sharded-U {m_new}")
    else:
        p0("[wr] 5c. memory_analysis unavailable on this backend; the "
           "shard measurement above is the residency evidence")

    # ---- 6. the reduction is ONE mesh axis ------------------------------
    cen = _collective_census(c_new.as_text(), p0, "rot", show_permute=True)
    cen_old = _collective_census(c_old.as_text(), p0, "rep")
    # The largest legitimate collective is the psum that finishes the band
    # sum, whose result is ONE output shard: (nk, nb_act, ns, nmu/p) c128
    # with p the SMALLER mesh axis (the larger of the two tiles).  A
    # collective-permute is bounded by the FULLY sharded active tile, which
    # is smaller again by the other axis.
    out_shard = NK * NACT * NS * (NMU // min(px, py)) * 16
    act_tile = NK * (NACT // max(px, py)) * NS * (NMU // min(px, py)) * 16
    ok6 = (cen['unparsed'] == 0
           and cen['worst_g'] <= max(px, py)
           and cen['worst_b'] <= out_shard
           and cen['perm_b'] <= act_tile
           and (max(px, py) == 1 or cen['counts']['all-reduce'] > 0))
    p0(f"[wr] 6. collectives {cen['counts']} unparsed={cen['unparsed']}; "
       f"worst result {cen['worst_b'] / 2**20:.4f} MiB (one psi out-shard "
       f"= {out_shard / 2**20:.4f} MiB); worst permute "
       f"{cen['perm_b'] / 2**20:.4f} MiB (one sharded active tile = "
       f"{act_tile / 2**20:.4f} MiB); worst replica group {cen['worst_g']} "
       f"of mesh {px}x{py}  {'PASS' if ok6 else 'FAIL'}")
    p0(f"[wr] 6b. the replicated-U path for contrast: {cen_old['counts']}, "
       f"worst result {cen_old['worst_b'] / 2**20:.4f} MiB")

    # ---- 7. an active window that does not divide the mesh --------------
    ok7 = True
    nact_odd = NACT - 1
    if nact_odd > 0 and (nact_odd % px or nact_odd % py):
        sl_odd = BandSlices.from_band_edges(0, 0, min(NOCC, nact_odd),
                                            nact_odd, NB)
        w_odd = _build_bundle(psi_host, enk_host, mesh, sl_odd, efermi)
        U_odd = np.stack([_haar(rng, nact_odd) for _ in range(NK)])
        r_odd = _host_rotate(psi_host, U_odd, nact_odd)
        for name, Uv in (("device-U", _put(U_odd, mesh, P(None, None, None))),
                         ("host-U", U_odd)):
            o_odd = rotate_wavefunctions(
                w_odd, Uv, enk_active_new=enk_host[:, :nact_odd],
                efermi=efermi, mesh_xy=mesh, active_slice=sl_odd.sigma)
            w7 = _all_copies_rel(o_odd, r_odd)
            good = w7 <= RTOL
            ok7 = ok7 and good
            p0(f"[wr] 7. nb_active={nact_odd} on {px}x{py}, {name:9s} "
               f"rel {w7:.3e}  {'PASS' if good else 'FAIL'}")
    else:
        p0(f"[wr] 7. nb_active={nact_odd} already divides {px}x{py}; "
           f"pad path not exercised at this shape")

    # ---- 8. U layout invariance -----------------------------------------
    variants = [("host-numpy", U)]
    if NACT % px == 0 and NACT % py == 0:
        variants.append(("band_rotation_spec",
                         _put(U, mesh, band_rotation_spec())))
    ok8 = True
    for name, Uv in variants:
        ov = rotate_wavefunctions(
            wfns, Uv, enk_active_new=enk_new, efermi=efermi, mesh_xy=mesh,
            active_slice=slices.sigma)
        wv = _all_copies_rel(ov, ref)
        good = wv <= RTOL
        ok8 = ok8 and good
        p0(f"[wr] 8. U as {name:20s} rel {wv:.3e}  "
           f"{'PASS' if good else 'FAIL'}")

    ok = (ok1 and ok1b and ok2 and ok3 and ok4 and ok5 and ok6 and ok7
          and ok8)
    p0(f"[wr] VERDICT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import traceback
    rc = 1
    try:
        rc = main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        rc = 1
    finalize_process(rc)
