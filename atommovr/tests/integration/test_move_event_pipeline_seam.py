import numpy as np
import pytest

from atommovr.utils.Move import Move
from atommovr.utils.ErrorModel import ErrorModel
from atommovr.utils.error_utils import finalize_events_to_moves
from atommovr.utils.failure_policy import FailureBit, FailureEvent, FailureFlag, bit_value


class _StubMove:
    """
    Minimal duck-typed stand-in for Move used by finalize_events_to_moves().
    We only need set_failure_event(...) for the seam tests.
    """
    def __init__(self) -> None:
        self.failure_event = int(FailureEvent.SUCCESS)
        self.set_failure_event_calls: list[int] = []

    def set_failure_event(self, ev: int) -> None:
        self.failure_event = int(ev)
        self.set_failure_event_calls.append(int(ev))


def _mask_of(*bits: FailureBit) -> np.uint64:
    m = np.uint64(0)
    for b in bits:
        m |= bit_value(b)
    return m

def test_event_pipeline_seam_with_real_move_end_to_end() -> None:
    """
    Integration seam test using the real Move class:
      event_mask updates -> suppression/resolution -> write-back to Move.fail_event/fail_flag
    """
    moves = [
        Move(0, 0, 0, 1),  # move 0
        Move(1, 1, 1, 2),  # move 1
        Move(2, 2, 3, 2),  # move 2
        Move(3, 3, 4, 4),  # move 3
        Move(4, 4, 4, 4),  # move 4
        Move(5, 5, 5, 6),  # move 5
    ]
    n = len(moves)
    event_mask = np.zeros(n, dtype=np.uint64)

    em = ErrorModel(
        pickup_fail_rate=1.0,
        putdown_fail_rate=1.0,
        accel_fail_rate=1.0,
        decel_fail_rate=1.0,
        seed=0,
    )

    has_atom = np.array([True, True, True, True, True, False], dtype=bool)

    eligible_collision_inevitable = np.array([False, False, False, True, False, False], dtype=bool)
    eligible_collision_avoidable = np.array([False, True, False, False, False, False], dtype=bool)

    eligible_pickup = np.array([True, False, False, False, False, False], dtype=bool)
    eligible_accel = np.array([True, True, False, False, False, False], dtype=bool)
    eligible_decel = np.array([False, False, True, False, False, False], dtype=bool)
    eligible_putdown = np.array([False, False, True, False, False, False], dtype=bool)

    # Gating by atom presence (same pattern as move_atoms)
    eligible_collision_inevitable &= has_atom
    eligible_collision_avoidable &= has_atom
    eligible_pickup &= has_atom
    eligible_accel &= has_atom
    eligible_decel &= has_atom
    eligible_putdown &= has_atom

    # Deterministic NO_ATOM tagging
    event_mask[~has_atom] |= bit_value(FailureBit.NO_ATOM)

    # Apply tags
    em.apply_inevitable_collision_mask(event_mask, eligible_collision_inevitable)
    em.apply_avoidable_collision_mask(event_mask, eligible_collision_avoidable)

    em.apply_pickup_errors_mask(event_mask, eligible_pickup)
    em.apply_accel_errors_mask(event_mask, eligible_accel)
    em.apply_decel_errors_mask(event_mask, eligible_decel)
    em.apply_putdown_errors_mask(event_mask, eligible_putdown)

    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)

    expected_primary = [
        FailureEvent.PICKUP_FAIL,     # pickup suppresses accel
        FailureEvent.COLLISION_AVOIDABLE,  # crossed_moving suppresses accel
        FailureEvent.DECEL_FAIL,      # decel suppresses putdown
        FailureEvent.COLLISION_INEVITABLE,  # dominant
        FailureEvent.SUCCESS,
        FailureEvent.NO_ATOM,
    ]

    # Primary array
    assert primary.tolist() == [int(x) for x in expected_primary]

    # Real Move state updates
    assert [m.fail_event for m in moves] == [int(x) for x in expected_primary]

    # fail_flag mapping also exercised
    assert moves[0].fail_flag == FailureFlag.NO_PICKUP
    assert moves[1].fail_flag == FailureFlag.LOSS
    assert moves[2].fail_flag == FailureFlag.LOSS
    assert moves[3].fail_flag == FailureFlag.LOSS
    assert moves[4].fail_flag == FailureFlag.SUCCESS
    assert moves[5].fail_flag == FailureFlag.NO_ATOM

    # Optional stored masks (post-suppression)
    assert moves[0].fail_mask == int(_mask_of(FailureBit.PICKUP_FAIL))
    assert moves[1].fail_mask == int(_mask_of(FailureBit.COLLISION_AVOIDABLE))
    assert moves[2].fail_mask == int(_mask_of(FailureBit.DECEL_FAIL))
    assert moves[3].fail_mask == int(_mask_of(FailureBit.COLLISION_INEVITABLE))
    assert moves[4].fail_mask == 0
    assert moves[5].fail_mask == int(_mask_of(FailureBit.NO_ATOM))


def test_event_pipeline_seam_with_real_move_empty_inputs() -> None:
    moves: list[Move] = []
    event_mask = np.zeros(0, dtype=np.uint64)

    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)

    assert primary.shape == (0,)
    assert primary.dtype == np.int32

def test_event_pipeline_seam_smoke_end_to_end_mixed_processes() -> None:
    """
    End-to-end seam test for the event pipeline:
      apply_*_mask -> suppress/resolve -> write back to moves

    This test deliberately creates overlapping bits so suppression/precedence
    behavior is exercised (e.g. pickup suppresses downstream motion failures).
    """
    n = 6
    moves = [_StubMove() for _ in range(n)]
    event_mask = np.zeros(n, dtype=np.uint64)

    # Deterministic rates: all eligible stochastic events will be tagged.
    em = ErrorModel(
        pickup_fail_rate=1.0,
        putdown_fail_rate=1.0,
        accel_fail_rate=1.0,
        decel_fail_rate=1.0,
        seed=0,
    )

    # Build eligibilities (one boolean mask per process)
    has_atom = np.array([True, True, True, True, True, False], dtype=bool)

    eligible_collision_inevitable = np.array([False, False, False, True, False, False], dtype=bool)
    eligible_collision_avoidable = np.array([False, True, False, False, False, False], dtype=bool)

    eligible_pickup = np.array([True, False, False, False, False, False], dtype=bool)
    eligible_accel = np.array([True, True, False, False, False, False], dtype=bool)
    eligible_decel = np.array([False, False, True, False, False, False], dtype=bool)
    eligible_putdown = np.array([False, False, True, False, False, False], dtype=bool)

    # Mimic move_atoms() behavior: only apply move-related processes to moves with atoms
    eligible_collision_inevitable &= has_atom
    eligible_collision_avoidable &= has_atom
    eligible_pickup &= has_atom
    eligible_accel &= has_atom
    eligible_decel &= has_atom
    eligible_putdown &= has_atom

    # Deterministic NO_ATOM tagging for move 5
    no_atom_eligible = ~has_atom
    event_mask[no_atom_eligible] |= bit_value(FailureBit.NO_ATOM)

    # Apply deterministic crossed tags
    em.apply_inevitable_collision_mask(event_mask, eligible_collision_inevitable)
    em.apply_avoidable_collision_mask(event_mask, eligible_collision_avoidable)

    # Apply stochastic process tags (all deterministic here because p=1)
    em.apply_pickup_errors_mask(event_mask, eligible_pickup)
    em.apply_accel_errors_mask(event_mask, eligible_accel)
    em.apply_decel_errors_mask(event_mask, eligible_decel)
    em.apply_putdown_errors_mask(event_mask, eligible_putdown)

    # Store raw mask for inspection after suppression
    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)

    # Expected outcomes by move index:
    # 0: pickup + accel -> pickup suppresses accel -> PICKUP_FAIL
    # 1: crossed_moving + accel -> crossed_moving suppresses accel -> CROSSED_MOVING
    # 2: decel + putdown -> decel suppresses putdown -> DECEL_FAIL
    # 3: crossed_static -> CROSSED_STATIC (dominant over other non-NO_ATOM bits)
    # 4: no bits -> SUCCESS
    # 5: NO_ATOM -> NO_ATOM
    expected_primary = np.array(
        [
            int(FailureEvent.PICKUP_FAIL),
            int(FailureEvent.COLLISION_AVOIDABLE),
            int(FailureEvent.DECEL_FAIL),
            int(FailureEvent.COLLISION_INEVITABLE),
            int(FailureEvent.SUCCESS),
            int(FailureEvent.NO_ATOM),
        ],
        dtype=np.int32,
    )

    assert primary.dtype == np.int32
    assert np.array_equal(primary, expected_primary)

    # Verify write-back to moves happened
    assert [m.failure_event for m in moves] == expected_primary.tolist()
    for m, ev in zip(moves, expected_primary):
        assert m.set_failure_event_calls[-1] == int(ev)

    # Verify stored masks reflect suppression results (not raw pre-suppression combinations)
    assert moves[0].fail_mask == int(_mask_of(FailureBit.PICKUP_FAIL))
    assert moves[1].fail_mask == int(_mask_of(FailureBit.COLLISION_AVOIDABLE))
    assert moves[2].fail_mask == int(_mask_of(FailureBit.DECEL_FAIL))
    assert moves[3].fail_mask == int(_mask_of(FailureBit.COLLISION_INEVITABLE))
    assert moves[4].fail_mask == 0
    assert moves[5].fail_mask == int(_mask_of(FailureBit.NO_ATOM))


def test_event_pipeline_seam_pickup_suppresses_crossed_moving_and_downstream_bits() -> None:
    """
    Regression-style seam test:
    If PICKUP_FAIL and CROSSED_MOVING are both tagged, pickup should suppress
    crossed-moving (and other downstream motion failures), leaving PICKUP_FAIL.
    """
    n = 1
    moves = [_StubMove() for _ in range(n)]
    event_mask = np.zeros(n, dtype=np.uint64)

    em = ErrorModel(
        pickup_fail_rate=1.0,
        accel_fail_rate=1.0,
        decel_fail_rate=1.0,
        putdown_fail_rate=1.0,
        seed=0,
    )

    eligible = np.array([True], dtype=bool)

    em.apply_avoidable_collision_mask(event_mask, eligible)
    em.apply_pickup_errors_mask(event_mask, eligible)
    em.apply_accel_errors_mask(event_mask, eligible)
    em.apply_decel_errors_mask(event_mask, eligible)
    em.apply_putdown_errors_mask(event_mask, eligible)

    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)

    assert primary[0] == int(FailureEvent.PICKUP_FAIL)
    assert moves[0].failure_event == int(FailureEvent.PICKUP_FAIL)
    assert moves[0].fail_mask == int(_mask_of(FailureBit.PICKUP_FAIL))


def test_event_pipeline_seam_no_atom_blocks_other_processes_when_eligibility_is_gated() -> None:
    """
    Mimics move_atoms() gating: if a move has no atom, move-related eligibilities are
    masked out, and the final result should remain NO_ATOM only.
    """
    n = 3
    moves = [_StubMove() for _ in range(n)]
    event_mask = np.zeros(n, dtype=np.uint64)

    em = ErrorModel(
        pickup_fail_rate=1.0,
        putdown_fail_rate=1.0,
        accel_fail_rate=1.0,
        decel_fail_rate=1.0,
        seed=0,
    )

    has_atom = np.array([True, False, True], dtype=bool)
    no_atom_eligible = ~has_atom
    event_mask[no_atom_eligible] |= bit_value(FailureBit.NO_ATOM)

    # Start with all moves eligible, then apply the same gating pattern as move_atoms()
    eligible_pickup = np.array([True, True, False], dtype=bool) & has_atom
    eligible_accel = np.array([False, True, True], dtype=bool) & has_atom
    eligible_decel = np.array([False, True, True], dtype=bool) & has_atom
    eligible_putdown = np.array([False, True, True], dtype=bool) & has_atom
    eligible_collision_inevitable = np.array([False, True, False], dtype=bool) & has_atom
    eligible_collision_avoidable = np.array([False, True, False], dtype=bool) & has_atom

    em.apply_pickup_errors_mask(event_mask, eligible_pickup)
    em.apply_accel_errors_mask(event_mask, eligible_accel)
    em.apply_decel_errors_mask(event_mask, eligible_decel)
    em.apply_putdown_errors_mask(event_mask, eligible_putdown)
    em.apply_inevitable_collision_mask(event_mask, eligible_collision_inevitable)
    em.apply_avoidable_collision_mask(event_mask, eligible_collision_avoidable)

    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)

    assert primary.tolist() == [
        int(FailureEvent.PICKUP_FAIL),  # move 0
        int(FailureEvent.NO_ATOM),      # move 1 (all other eligibilities gated off)
        int(FailureEvent.ACCEL_FAIL),   # move 2: accel + decel + putdown -> accel dominates/suppresses
    ]

    assert moves[1].fail_mask == int(_mask_of(FailureBit.NO_ATOM))


def test_event_pipeline_seam_empty_inputs() -> None:
    """
    Empty pipeline should be a no-op and return an empty primary array.
    """
    moves: list[_StubMove] = []
    event_mask = np.zeros(0, dtype=np.uint64)

    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)

    assert isinstance(primary, np.ndarray)
    assert primary.shape == (0,)
    assert primary.dtype == np.int32