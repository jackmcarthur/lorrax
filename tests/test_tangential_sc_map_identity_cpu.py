"""An isospectral SC rotation must not authenticate an earlier map's bank."""
import os
from pathlib import Path
import subprocess
import sys


def _controls(*, expect_collision=False):
    import runtime
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure()
    import gc
    import hashlib
    import json
    from types import SimpleNamespace as NS
    from unittest.mock import patch
    import weakref
    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw.tangential_recipe import bind_tangential_sc_identity
    from gw.tangential_screening import tangential_identity
    from gw import response_bank
    from file_io.tangential_store import _check_identity

    jax.config.update("jax_enable_x64", True)
    energy, occupations = np.array([[-1., 1.]]), np.array([[1., 0.]])
    # Both H have EXACTLY the same eigenvalues and occupations in the
    # same unchanged DFT basis. Their occupied/empty subspaces differ.
    hamiltonians = (np.diag(energy[0]), np.array([[0., 1.], [1., 0.]]))
    psi = (np.eye(2), np.array([[-1., 1.], [1., 1.]])/np.sqrt(2))
    transition = [p[1].conj()*p[0] for p in psi]
    response = [4*np.outer(f, f.conj())/(1j**2-4) for f in transition]
    response_difference = float(np.linalg.norm(response[0]-response[1]))
    assert response_difference > .3
    records, labels_by_n = [], {}
    for side, n in ((1, 2), (2, 2), (1, 3), (2, 3)):
        if n == 3:
            energy, occupations = np.array([[-1., 1., 3.]]), np.array([[1., 0., 0.]])
            hamiltonians = (np.diag(energy[0]),
                np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 3.]]))
        mesh = Mesh(np.array(jax.devices("cpu")[:side*side]).reshape(side, side), ("x", "y"))
        rep, face = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "x", "y"))
        wfns = NS(enk=jax.device_put(energy, rep), occ=jax.device_put(occupations, rep),
                  slices=NS(b0=0, b4_logical=n))
        meta = NS(b_id_4_chi_user=n,
                  tangential_recipe=dict(recipe_hash="same-recipe", gate_hash="same-gates"))
        occ = NS(occ_hash=hashlib.sha256(occupations.tobytes()).hexdigest())
        cache_before = dict(response_bank._OPERATOR_DIGESTS)

        def bind(h):
            # The actual map's logical odd-width carry is replicated;
            # the digest itself must still compute through all-P tiles.
            matrix = jax.device_put(h[None].astype(complex), rep if n % side else face)
            ref = weakref.ref(matrix)
            state = NS(iteration=3, H_qp_dft=matrix)
            bind_tangential_sc_identity(meta, state, occupation_state=occ, print_fn=lambda *_a: None)
            # Only the unchanged original WFN fingerprint is a fixture.
            # All current-map labels, E/f hashes and store checks are real.
            with patch("common.parallel_transport.wfn_fingerprint", return_value="same-DFT-WFN"):
                identity = tangential_identity(wfns, meta, label="sc_0003", wfn=None,
                    binding=None, centroid_indices=np.array([[0, 0, 0], [1, 0, 0]]))
            return identity, ref

        a, wa = bind(hamiltonians[0])
        same, wsame = bind(hamiltonians[0].copy())
        b, wb = bind(hamiltonians[1])
        assert a == same
        _check_identity(a, same)  # Re-evaluating the same physical input may reuse.
        assert a["energies"] == b["energies"] and a["occupations"] == b["occupations"]
        collision = a == b
        assert collision == expect_collision
        if collision:
            _check_identity(a, b)  # Reproduce the original unsafe acceptance.
        else:
            try:
                _check_identity(a, b)
            except ValueError as error:
                assert "stale input/SC identity" in str(error) and "hamiltonian" in str(error)
            else:
                raise AssertionError("a different SC Hamiltonian reused an older bank identity")
        gc.collect()
        assert all(ref() is None for ref in (wa, wsame, wb))
        assert response_bank._OPERATOR_DIGESTS == cache_before
        labels = [a["hamiltonian"], b["hamiltonian"]]
        if not expect_collision:
            for label, h in zip(labels, hamiltonians):
                source = h[None].astype(np.complex128)
                words = np.ascontiguousarray(source).view(np.uint64).reshape(-1)
                total = sum((2*i+1)*int(word) for i, word in enumerate(words)) % (1 << 64)
                reference = hashlib.sha256(f"{source.shape}:{total:016x}".encode()).hexdigest()
                assert label.endswith(":"+reference)
        assert labels_by_n.setdefault(n, labels) == labels
        records.append(dict(ranks=side*side, logical_bands=n, collision=collision,
                            same_input_identity_reused=True,
                            distinct_hamiltonian_refused=not collision,
                            input_matrices_not_retained=True,
                            energy_hash=a["energies"], occupation_hash=a["occupations"],
                            hamiltonian_labels=labels))
    # Other ragged layouts may not silently gather a large operator just
    # to obtain its identity. The actual SC logical carry is replicated.
    ragged = jax.device_put(np.zeros((4, 3, 3), complex),
                           NamedSharding(mesh, P(("x", "y"), None, None)))
    if not expect_collision:
        try:
            response_bank.operator_digest(ragged, None, cache=False)
        except ValueError as error:
            assert "no gather is allowed" in str(error)
        else:
            raise AssertionError("an unpriced ragged gather was admitted")
    print(json.dumps(dict(records=records, independent_response_difference=response_difference,
                          scope="same iteration/rerun identity, not a cause of fresh-map Gram failure")))


def test_isospectral_rotation_cannot_reuse_sc_bank_on_p1_p4_cpu():
    root = Path(__file__).parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/"src"), JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--controls"],
                            cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    assert '"distinct_hamiltonian_refused": true' in result.stdout


if __name__ == "__main__" and "--controls" in sys.argv:
    _controls(expect_collision="--expect-collision" in sys.argv)
