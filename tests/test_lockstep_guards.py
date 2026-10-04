"""Run-condition guards on toy inputs (CPU only, seconds).

1. ``runtime.aot_memory.check_chunk`` compares the largest compiled figure any
   rank read: a rank whose own figure fits still recompiles when another
   rank's does not, so no rank waits in a compile the others never start.
   ``runtime.aot_memory.step_up`` clamps a step to ``top`` before the caller's
   snap, which route G defines only up to ``top``.
2. ``gw.shared_pole_sectors.reduce_sector_pencil`` hands its final eigensolver
   an exactly Hermitian operand, so the whole-mesh eigh (checked against its
   operand) and the local one (which symmetrizes) solve the same matrix.
3. ``gw.response_bank._unitary_inversion`` takes inversion for the -q mirror
   only when the symmetry layer authorizes it as a unitary row.
"""
from types import SimpleNamespace

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


def test_check_chunk_compares_the_largest_figure(monkeypatch):
    from runtime import aot_memory

    built = []

    def build(chunk):
        built.append(int(chunk))
        return SimpleNamespace(chunk=int(chunk))

    # This rank's figure fits (1000 <= 1100); another rank reads 20 % more.
    monkeypatch.setattr(aot_memory, "compiled_new_bytes",
                        lambda ex, extra=0, platform=None: 100 * ex.chunk + int(extra))
    monkeypatch.setattr(aot_memory, "agreed_chunk", lambda v: min(int(v), int(round(1.2 * v))))
    check = aot_memory.check_chunk(10, build=build, fixed=0.0, per_unit=100.0, room=1100.0,
                                   stage="lockstep toy")
    assert built == [10, 9]
    assert check.chunk == 9 and check.recompiled


def test_step_up_never_snaps_past_top():
    from runtime import aot_memory

    n_grp = 12      # route G's plane groups; its snap divides by zero past n_grp
    snap = lambda v: v if v < 2 else -(-n_grp // (-(-n_grp // (v - 1)) - 1))
    build = lambda v: SimpleNamespace(v=int(v))
    # 3.5x over the room at 2 blocks: the implied step (14) is past top.
    value, compiled = aot_memory.step_up(
        2, n_grp, build=build, compiled=build(2), figure=lambda ex: 600 + 200 // ex.v,
        room=100.0, stage="snap toy", snap=snap)
    assert value == n_grp and compiled.v == n_grp


def test_sector_reduction_eigh_operand_is_hermitian():
    import jax.numpy as jnp
    from gw.shared_pole_local import _mm
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    from gw.shared_pole_sectors import reduce_sector_pencil

    rng = np.random.default_rng(7)
    n, m = 48, 5
    q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    # A joint metric down to its keep cut, where Y^H V Y loses Hermiticity.
    metric = (q * np.logspace(-7.5, 0, n)) @ q.conj().T
    v = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    panels = [rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n)) for _ in range(2)]
    pencil = tuple(jnp.asarray(a)[None] for a in
                   ((metric + metric.conj().T) / 2, (v + v.conj().T) / 2, *panels))
    seen = []

    def eigh(a):
        seen.append(np.asarray(a))
        return jnp.linalg.eigh(a)

    reduce_sector_pencil(pencil, eigh=eigh, matmul=_mm, gates=gates)
    final = seen[-1][0]
    assert np.array_equal(final, final.conj().T)


def test_inversion_mirror_needs_an_authorized_unitary_row():
    from gw.response_bank import _unitary_inversion

    ops = np.stack([np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)])

    def plan(rows):
        return SimpleNamespace(spatial_ops=ops, sym_perm=np.zeros((4, 6), np.int32),
                               n_sym_spatial=2, sym=SimpleNamespace(active_symmetry_rows=np.asarray(rows)))

    assert _unitary_inversion(plan([0, 1, 2, 3])) == 1        # time reversal allowed
    assert _unitary_inversion(plan([0, 1])) == 1              # unitary inversion (FM with I)
    assert _unitary_inversion(plan([0, 3])) is None           # I only with time reversal (PT AFM)
