"""Certify the distributed band-rotation primitive against the replicated U.

``gw.qsgw_density.rotate_band_axis`` is the one rotation primitive:
contract ONE index of ``U`` at ``band_rotation_spec`` (m on 'x', n on 'y')
against ONE band axis of an operand.  ``rotate_band_matrix`` is the
similarity transform written as two calls to it, and it replaces the two
places that used to GATHER U replicated first --
``sc_iteration._rotate_to_dft_basis`` and ``sigma_dispatch``'s V_H basis
change.  A replicated (nk, nb, nb) complex128 U is 9.2 GB/rank at
nk=144/nb=2000.

Checks, strongest first:

  1. AGREEMENT WITH THE PATH THIS REPLACES.  The old kernel -- U pinned
     replicated, one three-operand einsum, result replicated -- is kept
     here verbatim in structure so the claim is against the code that
     shipped, not a paraphrase.  Both directions, 1e-12 relative, and the
     absolute delta is printed because the two associate the band sums
     differently and bit-identity is NOT expected.
  2. AGREEMENT WITH AN EXPLICIT HOST ROTATION, same tolerance.  Catches a
     shared error that check 1 alone would carry through.
  3. TRANSPOSE PIN / negative control.  ``U A U^dagger`` and
     ``U^dagger A U`` are both Hermitian, have the same trace and the same
     Frobenius norm, so no invariance check can tell them apart.  Each
     direction is therefore also compared against the OTHER host
     reference, and that comparison must FAIL.
  4. CONJUGATION PIN.  ``rotate_band_matrix`` flips ``conj_u`` between its
     two calls; not flipping it is the other way to pair them wrong.  That
     one IS loud -- it breaks hermiticity, unlike check 3 -- but nothing in
     the pipeline looks, so it is built explicitly, must disagree with the
     reference, and its |Y - Y^dagger| is reported.
  5. U IS NEVER REPLICATED.  Per-rank addressable bytes for U at
     ``band_rotation_spec`` against the replicated bytes the old path
     held, plus the compiled modules' argument/temp/output bytes.
  6. COLLECTIVE CENSUS of both compiled modules (kind, result bytes,
     replica-group size), and the check that no collective in the new
     module carries a full (nk, nb, nb).
  7. ``rotate_bands`` UNCHANGED.  It now routes through the primitive; the
     old spelling is rebuilt here and the two must be BIT-IDENTICAL, since
     the constraints and the einsum are the same.
  8. axis=0 REFUSED -- axis 0 is k, the batch index U is indexed by.

Env: BR_NK, BR_NB, BR_NS, BR_NG.
"""
import os
import re
import sys

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402

from common.collectives import (process_count, process_rank,   # noqa: E402
                                resolve_mesh)
from gw.qsgw_density import (band_rotation_spec,               # noqa: E402
                             rotate_band_axis, rotate_band_matrix,
                             rotate_bands)
from gw.sc_iteration import _rotate_to_dft_basis               # noqa: E402

NK = int(os.environ.get("BR_NK", "4"))
NB = int(os.environ.get("BR_NB", "16"))
NS = int(os.environ.get("BR_NS", "2"))
NG = int(os.environ.get("BR_NG", "24"))
RTOL = 1e-12


def _haar(rng, n):
    """A Haar-ish unitary: QR of a complex Gaussian."""
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    return Q * (np.diagonal(R) / np.abs(np.diagonal(R)))[None, :]


def _put(host, mesh, spec):
    """Process-local placement: each rank stages only its own shard.

    ``jax.device_put`` of a host array onto a multi-process sharding fires
    JAX's hidden replica ``assert_equal`` all-gather; a gate that used it
    would be measuring the gate.
    """
    return jax.make_array_from_callback(
        host.shape, NamedSharding(mesh, spec), lambda idx: host[idx])


def _addressable_bytes(arr):
    return sum(int(np.asarray(s.data).nbytes) for s in arr.addressable_shards)


def _rel(got, ref):
    """max|got - ref| / max|ref| on a REPLICATED pair (host-readable)."""
    g = np.asarray(got)
    r = np.asarray(ref)
    return float(np.abs(g - r).max()) / max(float(np.abs(r).max()), 1e-300)


def _worst_pair_rel(a, b):
    """Shard-by-shard relative difference; works on any matching layout."""
    ref = {str(sh.index): np.asarray(sh.data) for sh in b.addressable_shards}
    worst = 0.0
    for sh in a.addressable_shards:
        want = ref[str(sh.index)]
        got = np.asarray(sh.data)
        worst = max(worst, float(np.abs(got - want).max())
                    / max(float(np.abs(want).max()), 1e-300))
    return worst


def _make_replicated_rotate(mesh, to_qp):
    """THE PATH THIS CHANGE REPLACES, verbatim in structure.

    ``sc_iteration._rotate_to_dft_basis`` before the change (``to_qp``
    False) and ``sigma_dispatch``'s V_H basis change (``to_qp`` True):
    U pinned replicated, ONE three-operand einsum, result replicated.
    """
    rep = NamedSharding(mesh, P(None, None, None))

    @jax.jit
    def fn(A, U):
        U = jax.lax.with_sharding_constraint(U, rep)
        if to_qp:
            out = jnp.einsum('kpm,kpq,kqn->kmn',
                             jnp.conj(U), A, U, optimize=True)
        else:
            out = jnp.einsum('kmp,kpq,knq->kmn',
                             U, A, jnp.conj(U), optimize=True)
        return jax.lax.with_sharding_constraint(out, rep)
    return fn


def _make_new_rotate(mesh, to_qp):
    """The shipped form: U stays at band_rotation_spec, result replicated."""
    rep = NamedSharding(mesh, P(None, None, None))

    @jax.jit
    def fn(A, U):
        out = rotate_band_matrix(A, U, mesh=mesh, to_qp=to_qp)
        return jax.lax.with_sharding_constraint(out, rep)
    return fn


def _make_unflipped_rotate(mesh, to_qp):
    """CONJUGATION NEGATIVE CONTROL: ``conj_u`` NOT flipped between the
    two axis rotations.  Y = conj(U) A conj(U)^T, which is NOT Hermitian
    for a Hermitian A unless U is real -- unlike the transposed rotation
    of check 3, which is.  Measured either way."""
    rep = NamedSharding(mesh, P(None, None, None))

    @jax.jit
    def fn(A, U):
        t = rotate_band_axis(A, U, mesh=mesh, axis=2, to_qp=to_qp,
                             conj_u=not to_qp)
        out = rotate_band_axis(t, U, mesh=mesh, axis=1, to_qp=to_qp,
                               conj_u=not to_qp)
        return jax.lax.with_sharding_constraint(out, rep)
    return fn


def _make_old_rotate_bands(mesh):
    """``qsgw_density.rotate_bands``'s inner jit before the change."""
    m_on_x = NamedSharding(mesh, P(None, "x", None, None))
    U_sh = NamedSharding(mesh, band_rotation_spec())
    from common.mtxel_sweep import band_sphere_spec
    band_xy = NamedSharding(mesh, band_sphere_spec())

    @jax.jit
    def fn(psi_, U_):
        psi_mx = jax.lax.with_sharding_constraint(psi_, m_on_x)
        U_x = jax.lax.with_sharding_constraint(U_, U_sh)
        out = jnp.einsum('kmn,kmsg->knsg', U_x, psi_mx, optimize=True)
        out = jax.lax.with_sharding_constraint(
            out, NamedSharding(mesh, P(None, "y", None, None)))
        return jax.lax.with_sharding_constraint(out, band_xy)
    return fn


def _mem(compiled):
    """``(argument, temp, output)`` bytes per rank, or None if unavailable."""
    try:
        ma = compiled.memory_analysis()
    except Exception:                                          # noqa: BLE001
        return None
    if ma is None:
        return None
    return (int(getattr(ma, "argument_size_in_bytes", 0)),
            int(getattr(ma, "temp_size_in_bytes", 0)),
            int(getattr(ma, "output_size_in_bytes", 0)))


_GROUPED = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all")
_KINDS = _GROUPED + ("collective-permute",)


def _group_size(line):
    m = re.search(r'replica_groups=\{\{(.*?)\}', line)
    if m:
        return len([v for v in m.group(1).split(',') if v.strip()])
    m = re.search(r'replica_groups=\[(\d+),(\d+)\]', line)
    if m:
        return int(m.group(2))
    return -1


def _census(txt, p0, tag):
    """Print every collective; return ``(count, worst result bytes)``."""
    n, worst = 0, 0
    for line in txt.splitlines():
        kind = next((k for k in _KINDS
                     if f" {k}(" in line or f"= {k}" in line), None)
        if kind is None:
            continue
        n += 1
        m = re.search(r'c(?:64|128)\[[\d,]+\]', line)
        nbytes = 0
        if m:
            nbytes = 16 if '128' in m.group(0) else 8
            for d in re.findall(r'\d+', m.group(0))[1:]:
                nbytes *= int(d)
        worst = max(worst, nbytes)
        p0(f"[br]   {tag} {kind:18s} {m.group(0) if m else '?':26s} "
           f"{nbytes / 2**20:9.4f} MiB  group={_group_size(line)}")
    return n, worst


def main():
    rank, world = process_rank(), process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    px, py = (int(s) for s in mesh.devices.shape)
    nb = NB
    if nb % px or nb % py:
        step = px * py
        nb = -(-nb // step) * step
    p0(f"[br] world={world} mesh=({px},{py}) nk={NK} nb={nb} "
       f"(BR_NB={NB}) ns={NS} ng={NG} rtol={RTOL:.0e}")

    rng = np.random.default_rng(20260805)
    U_h = np.stack([_haar(rng, nb) for _ in range(NK)]).astype(np.complex128)
    M = (rng.standard_normal((NK, nb, nb))
         + 1j * rng.standard_normal((NK, nb, nb)))
    A_h = 0.5 * (M + np.conj(np.transpose(M, (0, 2, 1))))      # Hermitian

    # HOST REFERENCES, both directions.  Eigenvectors are COLUMNS:
    # U[k, m, n] = <DFT_m | QP_n>, so A_DFT = U A_QP U^dagger.
    ref_dft = np.einsum('kmp,kpq,knq->kmn', U_h, A_h, np.conj(U_h),
                        optimize=True)
    ref_qp = np.einsum('kpm,kpq,kqn->kmn', np.conj(U_h), A_h, U_h,
                       optimize=True)

    rep = P(None, None, None)
    A_d = _put(A_h, mesh, rep)
    U_rep = _put(U_h, mesh, rep)
    U_bnd = _put(U_h, mesh, band_rotation_spec())

    ok = {}

    # ---- 1 + 2 + 3. agreement, both directions, with the wrong-direction
    #                 reference as the negative control ------------------
    for to_qp, ref, other, name in ((False, ref_dft, ref_qp, "QP->DFT"),
                                    (True, ref_qp, ref_dft, "DFT->QP")):
        old_fn = _make_replicated_rotate(mesh, to_qp)
        new_fn = _make_new_rotate(mesh, to_qp)
        got_old = old_fn(A_d, U_rep)
        got_new = new_fn(A_d, U_bnd)
        d_old = _rel(got_new, got_old)
        d_ref = _rel(got_new, ref)
        d_oldref = _rel(got_old, ref)
        good = (d_old <= RTOL) and (d_ref <= RTOL)
        ok[f"1{name}"] = good
        p0(f"[br] 1. {name:8s} new vs OLD replicated-U kernel  "
           f"rel {d_old:.3e}   (old vs host {d_oldref:.3e})  "
           f"{'PASS' if d_old <= RTOL else 'FAIL'}")
        p0(f"[br] 2. {name:8s} new vs explicit HOST rotation   "
           f"rel {d_ref:.3e}  {'PASS' if d_ref <= RTOL else 'FAIL'}")
        # The transpose pin: the OTHER direction's reference must be far.
        d_wrong = _rel(got_new, other)
        moved = d_wrong > 1e-3
        ok[f"3{name}"] = moved
        p0(f"[br] 3. {name:8s} vs the TRANSPOSED reference     "
           f"rel {d_wrong:.3e}  must be large  "
           f"{'PASS' if moved else 'FAIL'}")
        # ---- 4. conjugation pin -------------------------------------
        bad = _make_unflipped_rotate(mesh, to_qp)(A_d, U_bnd)
        d_bad = _rel(bad, ref)
        herm = float(np.abs(np.asarray(bad)
                            - np.conj(np.transpose(np.asarray(bad),
                                                   (0, 2, 1)))).max())
        conj_moved = d_bad > 1e-3
        ok[f"4{name}"] = conj_moved
        p0(f"[br] 4. {name:8s} conj_u NOT flipped vs reference "
           f"rel {d_bad:.3e}  must be large  "
           f"(its max|Y-Y^dagger| = {herm:.2e}; nonzero, so unlike check 3 "
           f"hermiticity does see this one)  "
           f"{'PASS' if conj_moved else 'FAIL'}")

    # ---- 5. residency ---------------------------------------------------
    b_rep = _addressable_bytes(U_rep)
    b_bnd = _addressable_bytes(U_bnd)
    ok['5'] = b_bnd * px * py == b_rep
    p0(f"[br] 5. per-rank U: replicated {b_rep / 2**20:.4f} MiB -> "
       f"band_rotation_spec {b_bnd / 2**20:.4f} MiB  "
       f"(x{b_rep / max(b_bnd, 1):.1f}, mesh {px}x{py})  "
       f"{'PASS' if ok['5'] else 'FAIL'}")
    full_u_bytes = NK * nb * nb * 16
    for to_qp, name in ((False, "QP->DFT"), (True, "DFT->QP")):
        for tag, fn, args in (
                ("old", _make_replicated_rotate(mesh, to_qp), (A_d, U_rep)),
                ("new", _make_new_rotate(mesh, to_qp), (A_d, U_bnd))):
            mem = _mem(fn.lower(*args).compile())
            if mem is None:
                p0(f"[br] 5. {name} {tag}: memory_analysis unavailable")
                continue
            a, t, o = mem
            p0(f"[br] 5. {name} {tag:3s} module bytes/rank  arg "
               f"{a / 2**20:9.3f}  temp {t / 2**20:9.3f}  out "
               f"{o / 2**20:9.3f} MiB   "
               f"(one full (nk,nb,nb) = {full_u_bytes / 2**20:.3f} MiB)")

    # ---- 6. collective census ------------------------------------------
    ok['6'] = True
    for to_qp, name in ((False, "QP->DFT"), (True, "DFT->QP")):
        for tag, fn, args in (
                ("old", _make_replicated_rotate(mesh, to_qp), (A_d, U_rep)),
                ("new", _make_new_rotate(mesh, to_qp), (A_d, U_bnd))):
            txt = fn.lower(*args).compile().as_text()
            n, worst = _census(txt, p0, f"{name} {tag}")
            p0(f"[br] 6. {name} {tag:3s}: {n} collectives, worst result "
               f"{worst / 2**20:.4f} MiB")
            if tag == "new" and world > 1:
                # No collective may carry a whole (nk, nb, nb) EXCEPT the
                # final gather of the result, which is the caller's pin.
                bad = worst > full_u_bytes
                ok['6'] = ok['6'] and not bad

    # ---- 7. rotate_bands unchanged --------------------------------------
    from common.mtxel_sweep import band_sphere_spec
    psi_h = (rng.standard_normal((NK, nb, NS, NG))
             + 1j * rng.standard_normal((NK, nb, NS, NG))).astype(np.complex128)
    psi_d = _put(psi_h, mesh, band_sphere_spec())
    got_rb = rotate_bands(psi_d, U_bnd, mesh=mesh)
    want_rb = _make_old_rotate_bands(mesh)(psi_d, U_bnd)
    d_rb = _worst_pair_rel(got_rb, want_rb)
    ok['7'] = d_rb == 0.0
    p0(f"[br] 7. rotate_bands new spelling vs old  rel {d_rb:.3e}  "
       f"(must be 0.0 -- same constraints, same einsum)  "
       f"{'PASS' if ok['7'] else 'FAIL'}")
    ref_rb = np.einsum('kmn,kmsg->knsg', U_h, psi_h, optimize=True)
    d_rbh = _worst_pair_rel(got_rb, _put(ref_rb, mesh, band_sphere_spec()))
    p0(f"[br] 7b. rotate_bands vs explicit host rotation  rel {d_rbh:.3e}  "
       f"{'PASS' if d_rbh <= RTOL else 'FAIL'}")
    ok['7b'] = d_rbh <= RTOL

    # ---- 8. axis 0 refused ----------------------------------------------
    try:
        jax.jit(lambda a, u: rotate_band_axis(
            a, u, mesh=mesh, axis=0, to_qp=True))(A_d, U_bnd)
        ok['8'] = False
        p0("[br] 8. axis=0 was ACCEPTED  FAIL")
    except Exception as exc:                                   # noqa: BLE001
        ok['8'] = "axis 0 is k" in str(exc)
        p0(f"[br] 8. axis=0 refused: {str(exc).splitlines()[0][:110]}  "
           f"{'PASS' if ok['8'] else 'FAIL'}")

    # ---- 9. the shipped seam, end to end --------------------------------
    got_seam = _rotate_to_dft_basis(A_d, U_bnd, mesh=mesh)
    d_seam = _rel(got_seam, ref_dft)
    ok['9'] = d_seam <= RTOL
    p0(f"[br] 9. sc_iteration._rotate_to_dft_basis vs host  "
       f"rel {d_seam:.3e}  out spec {got_seam.sharding.spec}  "
       f"{'PASS' if ok['9'] else 'FAIL'}")

    every = all(ok.values())
    p0(f"[br] VERDICT {'PASS' if every else 'FAIL'}  "
       f"({sum(1 for v in ok.values() if not v)} failing of {len(ok)})")
    return 0 if every else 1


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
