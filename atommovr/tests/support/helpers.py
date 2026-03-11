"""
Plain reusable test utilities: 
small builders, mask constructors, 
serializers, invariant checkers, 
replay helpers
"""

import numpy as np
from numpy.typing import NDArray

import atommovr.utils as movr
from atommovr.utils.failure_policy import FailureBit, bit_value

def mask_of(*bits: FailureBit) -> np.uint64:
    m = np.uint64(0)
    for b in bits:
        m |= bit_value(b)
    return m

def boom(*args, **kwargs):
    raise AssertionError("Expensive crossed-tone bookkeeping should not run on fast path.")

def _assert_binary_occupancy_3d(matrix: NDArray) -> None:
    """
    Helper assertion used across tests.

    Why this exists
    ---------------
    For noiseless algorithm planning, the occupancy representation should remain a
    {0,1} tensor at all times. This is a strong regression tripwire for:
    - dtype under/overflow (e.g. unsigned wraparound),
    - accidental multi-occupancy creation,
    - incorrect move application ordering.
    """
    assert matrix.ndim == 3 and matrix.shape[2] == 1
    assert np.issubdtype(matrix.dtype, np.integer)
    mn = int(matrix.min(initial=0))
    mx = int(matrix.max(initial=0))
    assert mn >= 0
    assert mx <= 1


def _n_atoms(matrix: NDArray) -> int:
    """
    Count atoms robustly (signed accumulation).

    Why this exists
    ---------------
    Several regressions you hit came from unsigned accumulation / subtraction.
    For tests, we make the counting itself immune to uint overflow.
    """
    return int(np.sum(matrix, dtype=np.int64))

def _moves_to_tuples(round_moves):
    return [[(m.from_row, m.from_col, m.to_row, m.to_col) for m in mvlist] for mvlist in round_moves]


def _replay_and_check_noiseless_conservation(
    aa0: movr.AtomArray,
    move_rounds: list[list[movr.Move]],
) -> movr.AtomArray:
    """
    Replay algorithm-produced moves and check invariants after each round.

    Why this exists
    ---------------
    Many BCv2 bugs manifest as:
    - a move round that doesn't change state (infinite loop risk),
    - a hidden collision causing loss (sum decreases),
    - dtype regressions creating wraparound or multi-occupancy.

    This helper makes those issues fail fast with a clear message.
    """
    aa = aa0  # mutate in place
    n0 = _n_atoms(aa.matrix)
    _assert_binary_occupancy_3d(aa.matrix)

    for k, round_moves in enumerate(move_rounds):
        before = aa.matrix.copy()
        # evaluate_moves expects a "list of move_lists"
        aa.evaluate_moves([round_moves])

        # Must make progress if round_moves is non-empty.
        if len(round_moves) > 0:
            assert not np.array_equal(
                aa.matrix, before
            ), f"Round {k} had moves but matrix did not change (possible infinite loop trigger)."

        # Noiseless planning => conservation + binary occupancy.
        assert _n_atoms(aa.matrix) == n0, f"Atom count changed after round {k}."
        _assert_binary_occupancy_3d(aa.matrix)

    return aa