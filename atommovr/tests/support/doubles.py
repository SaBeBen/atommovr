"""
Shared test doubles:
fake/spy/stub classes or
lightweight replacement implementations of real collaborators.
"""

import numpy as np
from numpy.typing import NDArray


class TimingSpyErrorModel:
    """
    Deterministic seam-test double for `move_atoms` timing/error orchestration.

    This is intentionally minimal: it records which phase-specific error hooks were
    called and with which eligibility masks, and records the total evolution time
    passed into `get_atom_loss`.
    """

    def __init__(self) -> None:
        self.pickup_time = 1.4
        self.accel_time = 2.8
        self.decel_time = 3.3
        self.putdown_time = 4.6
        self.calls: list[tuple[str, NDArray[np.bool_]]] = []
        self.loss_times: list[float] = []

    def apply_pickup_errors_mask(
        self,
        event_mask: NDArray[np.int_],
        eligible: NDArray[np.bool_],
    ) -> None:
        self.calls.append(("pickup", eligible.copy()))

    def apply_accel_errors_mask(
        self,
        event_mask: NDArray[np.int_],
        eligible: NDArray[np.bool_],
    ) -> None:
        self.calls.append(("accel", eligible.copy()))

    def apply_decel_errors_mask(
        self,
        event_mask: NDArray[np.int_],
        eligible: NDArray[np.bool_],
    ) -> None:
        self.calls.append(("decel", eligible.copy()))

    def apply_putdown_errors_mask(
        self,
        event_mask: NDArray[np.int_],
        eligible: NDArray[np.bool_],
    ) -> None:
        self.calls.append(("putdown", eligible.copy()))

    def apply_inevitable_collision_mask(
        self,
        event_mask: NDArray[np.int_],
        eligible: NDArray[np.bool_],
    ) -> None:
        self.calls.append(("collision_inevitable", eligible.copy()))

    def apply_avoidable_collision_mask(
        self,
        event_mask: NDArray[np.int_],
        eligible: NDArray[np.bool_],
    ) -> None:
        self.calls.append(("collision_avoidable", eligible.copy()))

    def get_atom_loss(
        self,
        matrix: NDArray,
        evolution_time: float,
        n_species: int,
    ) -> tuple[NDArray, int]:
        self.loss_times.append(float(evolution_time))
        return matrix, 0


class BoomErrorModel:
    pickup_time = 1.0
    accel_time = 1.0
    decel_time = 1.0
    putdown_time = 1.0

    def __getattr__(self, _name):
        raise AssertionError(
            "Error model should not be touched on invalid input state."
        )
