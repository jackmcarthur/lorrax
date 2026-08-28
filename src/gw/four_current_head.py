"""Small frequency-axis carrier for the four-current head response."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from common.collectives import gather_to_host


@dataclass(frozen=True)
class FrequencyResolvedFourCurrentHead:
    """Direct four-current head coefficients on one explicit frequency axis.

    The leading dimension of ``Q0_direct``, ``H_linear`` and ``S_direct`` is
    the ordered ``omega_ry`` axis.  Frequency methods choose that axis and fit
    downstream; this carrier contains neither method roles nor fitted poles.
    Every array is small and replicated.
    """

    omega_ry: jax.Array                 # (nw,)
    Q0_direct: jax.Array                # (nw,4,4)
    H_linear: jax.Array                 # (nw,2,4,4)
    S_direct: jax.Array                 # (nw,2,2,4,4)

    @staticmethod
    def canonical_frequencies(omegas_ry) -> tuple[complex, ...]:
        """Return one nonempty, finite, unique frequency axis."""
        values = tuple(complex(value) for value in tuple(omegas_ry))
        if not values:
            raise ValueError("four-current head requires a frequency")
        if any(not (np.isfinite(value.real) and np.isfinite(value.imag))
               for value in values):
            raise ValueError("four-current head has a non-finite frequency")
        if len(set(values)) != len(values):
            raise ValueError(
                f"four-current head has duplicate frequencies: {values}")
        return values

    def __post_init__(self) -> None:
        omega = self.canonical_frequencies(
            gather_to_host(self.omega_ry).reshape(-1))
        nw = len(omega)
        expected = {
            "omega_ry": (nw,),
            "Q0_direct": (nw, 4, 4),
            "H_linear": (nw, 2, 4, 4),
            "S_direct": (nw, 2, 2, 4, 4),
        }
        for name, shape in expected.items():
            array = getattr(self, name)
            if np.dtype(array.dtype) != np.dtype(np.complex128):
                raise TypeError(
                    f"four-current head {name} dtype {array.dtype} != "
                    "complex128")
            value = gather_to_host(array)
            if value.shape != shape:
                raise ValueError(
                    f"four-current head {name} shape {value.shape} != {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"four-current head {name} is not finite")

    def index(self, omega_ry) -> int:
        """Return the unique exact index of one stored frequency."""
        omega = complex(omega_ry)
        if not (np.isfinite(omega.real) and np.isfinite(omega.imag)):
            raise ValueError("four-current head lookup needs a finite frequency")
        available = tuple(complex(value) for value in gather_to_host(
            self.omega_ry))
        try:
            return available.index(omega)
        except ValueError:
            raise ValueError(
                "four-current head has no exact response at "
                f"omega_ry={omega!r}; available={available}") from None


__all__ = [
    "FrequencyResolvedFourCurrentHead",
]
