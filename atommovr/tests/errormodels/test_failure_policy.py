import numpy as np
import pytest

from atommovr.utils import failure_policy as fp
from atommovr.utils.failure_policy import (
    FailureBit,
    FailureEvent,
    bit_value,
    suppress_inplace,
    resolve_primary_events,
)
from atommovr.tests.support.helpers import mask_of


def test_bit_value_is_one_shift_position() -> None:
    assert bit_value(FailureBit.PICKUP_FAIL) == (np.uint64(1) << np.uint64(0))
    assert bit_value(FailureBit.PUTDOWN_FAIL) == (np.uint64(1) << np.uint64(1))
    assert bit_value(FailureBit.NO_ATOM) == (np.uint64(1) << np.uint64(2))
    assert bit_value(FailureBit.COLLISION_INEVITABLE) == (np.uint64(1) << np.uint64(6))
    assert bit_value(FailureBit.COLLISION_AVOIDABLE) == (np.uint64(1) << np.uint64(7))


def test_suppress_noop_for_empty() -> None:
    event_mask = np.zeros(0, dtype=np.uint64)
    suppress_inplace(event_mask)
    assert event_mask.size == 0


def test_suppress_all_bits_collapses_to_no_atom() -> None:
    # A useful integration sanity check: NO_ATOM dominates everything.
    all_bits_mask = np.uint64(0)
    for b in FailureBit:
        all_bits_mask |= bit_value(b)
    event_mask = np.array([all_bits_mask], dtype=np.uint64)

    suppress_inplace(event_mask)
    assert event_mask[0] == mask_of(FailureBit.NO_ATOM)


def test_suppress_no_atom_dominates_all() -> None:
    event_mask = np.array(
        [mask_of(
            FailureBit.NO_ATOM,
            FailureBit.PICKUP_FAIL,
            FailureBit.COLLISION_AVOIDABLE,
            FailureBit.COLLISION_INEVITABLE,
            FailureBit.ACCEL_FAIL,
            FailureBit.DECEL_FAIL,
            FailureBit.PUTDOWN_FAIL,
            FailureBit.TRANSPORT_FAIL,
        )],
        dtype=np.uint64,
    )
    suppress_inplace(event_mask)
    assert event_mask[0] == mask_of(FailureBit.NO_ATOM)


def test_suppress_collision_inevitable_dominates_except_no_atom() -> None:
    event_mask = np.array(
        [mask_of(
            FailureBit.COLLISION_INEVITABLE,
            FailureBit.PICKUP_FAIL,
            FailureBit.PUTDOWN_FAIL,
            FailureBit.ACCEL_FAIL,
        )],
        dtype=np.uint64,
    )
    suppress_inplace(event_mask)
    assert event_mask[0] == mask_of(FailureBit.COLLISION_INEVITABLE)


def test_suppress_pickup_fail_clears_avoidable_collision_and_downstream() -> None:
    event_mask = np.array(
        [mask_of(
            FailureBit.PICKUP_FAIL,
            FailureBit.COLLISION_AVOIDABLE,
            FailureBit.ACCEL_FAIL,
            FailureBit.TRANSPORT_FAIL,
            FailureBit.DECEL_FAIL,
            FailureBit.PUTDOWN_FAIL,
        )],
        dtype=np.uint64,
    )
    suppress_inplace(event_mask)
    assert event_mask[0] == mask_of(FailureBit.PICKUP_FAIL)


def test_REGRESSION_suppress_collision_avoidable_clears_downstream_when_not_pickup() -> None:
    # COLLISION_AVOIDABALE is dominant unless PICKUP_FAIL (per exception list),
    # and also suppresses downstream bits (per suppression rules).
    event_mask = np.array(
        [mask_of(
            FailureBit.COLLISION_AVOIDABLE,
            FailureBit.ACCEL_FAIL,
            FailureBit.TRANSPORT_FAIL,
            FailureBit.DECEL_FAIL,
            FailureBit.PUTDOWN_FAIL,
        )],
        dtype=np.uint64,
    )
    suppress_inplace(event_mask)
    assert event_mask[0] == mask_of(FailureBit.COLLISION_AVOIDABLE)


def test_suppress_accel_fail_clears_transport_decel_putdown() -> None:
    event_mask = np.array(
        [mask_of(
            FailureBit.ACCEL_FAIL,
            FailureBit.TRANSPORT_FAIL,
            FailureBit.DECEL_FAIL,
            FailureBit.PUTDOWN_FAIL,
        )],
        dtype=np.uint64,
    )
    suppress_inplace(event_mask)
    assert event_mask[0] == mask_of(FailureBit.ACCEL_FAIL)


def test_suppress_transport_fail_clears_decel_putdown() -> None:
    event_mask = np.array(
        [mask_of(
            FailureBit.TRANSPORT_FAIL,
            FailureBit.DECEL_FAIL,
            FailureBit.PUTDOWN_FAIL,
        )],
        dtype=np.uint64,
    )
    suppress_inplace(event_mask)
    assert event_mask[0] == mask_of(FailureBit.TRANSPORT_FAIL)


def test_suppress_decel_fail_clears_putdown() -> None:
    event_mask = np.array(
        [mask_of(
            FailureBit.DECEL_FAIL,
            FailureBit.PUTDOWN_FAIL,
        )],
        dtype=np.uint64,
    )
    suppress_inplace(event_mask)
    assert event_mask[0] == mask_of(FailureBit.DECEL_FAIL)


def test_suppress_handles_signed_int_dtype() -> None:
    event_mask = np.array(
        [int(mask_of(FailureBit.PICKUP_FAIL, FailureBit.ACCEL_FAIL, FailureBit.PUTDOWN_FAIL))],
        dtype=np.int64,
    )
    suppress_inplace(event_mask)
    assert np.uint64(event_mask[0]) == mask_of(FailureBit.PICKUP_FAIL)


# ----------------------------
# Table-driven rule verification
# ----------------------------

def test_dominance_rules_match_table() -> None:
    """
    Verify each dominance rule behaves exactly as declared:
    - If dominant bit is set and none of its exceptions are set, output is exactly that bit.
    - If any exception bit is also set, dominance must NOT collapse to just the dominant bit.
    """
    for dominant, excepts in fp._DOMINANCE_RULES:
        dom_bit = dominant
        exc_bits = tuple(excepts)

        # Case A: dominant bit set, no exceptions -> collapses to dominant bit only.
        base = mask_of(dom_bit, FailureBit.ACCEL_FAIL, FailureBit.PUTDOWN_FAIL)
        # Ensure we didn't accidentally include an exception in base.
        for e in exc_bits:
            assert (base & bit_value(e)) == 0

        event_mask = np.array([base], dtype=np.uint64)
        suppress_inplace(event_mask)
        assert event_mask[0] == mask_of(dom_bit)

        # Case B: dominant bit set + one exception -> must NOT collapse to only dominant bit.
        for e in exc_bits:
            m = mask_of(dom_bit, e, FailureBit.ACCEL_FAIL)
            event_mask = np.array([m], dtype=np.uint64)
            suppress_inplace(event_mask)
            assert event_mask[0] != mask_of(dom_bit)


def test_suppression_rules_clear_exactly_table_bits() -> None:
    """
    Verify each suppression rule clears exactly the listed bits.

    We avoid adding arbitrary "sentinel" bits because dominance rules
    (e.g. NO_ATOM, CROSSED_STATIC) can legally collapse the mask and
    remove other bits.
    """
    for trigger, clears in fp._SUPPRESSION_RULES:
        trigger_bit = trigger
        clear_bits = tuple(clears)

        # Construct mask: trigger + all bits that should be cleared by that trigger
        m = mask_of(trigger_bit, *clear_bits)
        event_mask = np.array([m], dtype=np.uint64)

        suppress_inplace(event_mask)
        out = event_mask[0]

        # Trigger must still be set
        assert (out & bit_value(trigger_bit)) != 0

        # All "clears" bits must be cleared
        if clear_bits:
            clears_mask = mask_of(*clear_bits)
            assert (out & clears_mask) == 0

        # And since we didn't include any other bits, the output should be exactly the trigger bit.
        assert out == mask_of(trigger_bit)

# ----------------------------
# Primary event resolution
# ----------------------------

def test_resolve_primary_events_success_when_no_bits() -> None:
    event_mask = np.array([0, 0, 0], dtype=np.uint64)
    primary = resolve_primary_events(event_mask)
    assert primary.dtype == np.int32
    assert np.all(primary == int(FailureEvent.SUCCESS))


def test_resolve_primary_events_uses_precedence_order() -> None:
    # PRIMARY_EVENT_ORDER:
    # NO_ATOM > COLLISION_INEVITABLE > PICKUP_FAIL > CROSSED_STATIC > ACCEL_FAIL > TRANSPORT_FAIL > DECEL_FAIL > PUTDOWN_FAIL
    event_mask = np.array(
        [
            mask_of(FailureBit.PUTDOWN_FAIL, FailureBit.DECEL_FAIL),         # -> DECEL_FAIL
            mask_of(FailureBit.ACCEL_FAIL, FailureBit.PICKUP_FAIL),          # -> PICKUP_FAIL
            mask_of(FailureBit.TRANSPORT_FAIL, FailureBit.PUTDOWN_FAIL),     # -> TRANSPORT_FAIL
            mask_of(FailureBit.COLLISION_AVOIDABLE, FailureBit.PICKUP_FAIL), # -> PICKUP_FAIL (precedence only)
            mask_of(FailureBit.COLLISION_INEVITABLE, FailureBit.ACCEL_FAIL), # -> COLLISION_INEVITABLE
        ],
        dtype=np.uint64,
    )
    primary = resolve_primary_events(event_mask)

    assert primary[0] == int(FailureEvent.DECEL_FAIL)
    assert primary[1] == int(FailureEvent.PICKUP_FAIL)
    assert primary[2] == int(FailureEvent.TRANSPORT_FAIL)
    assert primary[3] == int(FailureEvent.PICKUP_FAIL)
    assert primary[4] == int(FailureEvent.COLLISION_INEVITABLE)


def test_resolve_primary_events_does_not_require_suppress() -> None:
    event_mask = np.array(
        [
            mask_of(FailureBit.COLLISION_INEVITABLE, FailureBit.NO_ATOM),          # -> NO_ATOM
            mask_of(FailureBit.COLLISION_INEVITABLE, FailureBit.PICKUP_FAIL),      # -> COLLISION_INEVITABLE (without suppression)
        ],
        dtype=np.uint64,
    )
    primary = resolve_primary_events(event_mask)

    assert primary[0] == int(FailureEvent.NO_ATOM)
    assert primary[1] == int(FailureEvent.COLLISION_INEVITABLE)
