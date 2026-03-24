import numpy as np
import pytest
from unittest.mock import Mock

from atommovr.utils.failure_policy import FailureBit, FailureEvent, bit_value
from atommovr.utils.error_utils import (
    set_event_bit_inplace,
    eligible_from_indices,
    eligible_from_moves,
    write_primary_events_to_moves,
    finalize_events_to_moves,
    apply_bernoulli_event_inplace,
)
from atommovr.tests.support.helpers import mask_of


## test functions ##
def test_apply_bernoulli_p0_sets_nothing() -> None:
    n = 50
    event_mask = np.zeros(n, dtype=np.uint64)
    eligible = np.ones(n, dtype=bool)
    rng = np.random.default_rng(123)

    apply_bernoulli_event_inplace(
        event_mask=event_mask,
        eligible=eligible,
        p_fail=0.0,
        bit=FailureBit.PUTDOWN_FAIL,
        rng=rng,
    )

    assert np.all(event_mask == 0)


def test_apply_bernoulli_p1_sets_all_eligible() -> None:
    n = 50
    event_mask = np.zeros(n, dtype=np.uint64)
    eligible = np.zeros(n, dtype=bool)
    eligible[::2] = True  # half eligible
    rng = np.random.default_rng(123)

    apply_bernoulli_event_inplace(
        event_mask=event_mask,
        eligible=eligible,
        p_fail=1.0,
        bit=FailureBit.ACCEL_FAIL,
        rng=rng,
    )

    bv = bit_value(FailureBit.ACCEL_FAIL)
    assert np.all(event_mask[eligible] == bv)
    assert np.all(event_mask[~eligible] == 0)


def test_apply_bernoulli_no_eligible_is_noop() -> None:
    n = 50
    event_mask = np.zeros(n, dtype=np.uint64)
    eligible = np.zeros(n, dtype=bool)
    rng = np.random.default_rng(123)

    apply_bernoulli_event_inplace(
        event_mask=event_mask,
        eligible=eligible,
        p_fail=1.0,
        bit=FailureBit.DECEL_FAIL,
        rng=rng,
    )

    assert np.all(event_mask == 0)


def test_apply_bernoulli_does_not_clear_existing_bits() -> None:
    n = 10
    event_mask = np.zeros(n, dtype=np.uint64)

    # pre-set a different bit on all entries
    preset = bit_value(FailureBit.PICKUP_FAIL)
    event_mask[:] = preset

    eligible = np.ones(n, dtype=bool)
    rng = np.random.default_rng(123)

    apply_bernoulli_event_inplace(
        event_mask=event_mask,
        eligible=eligible,
        p_fail=1.0,
        bit=FailureBit.PUTDOWN_FAIL,
        rng=rng,
    )

    # now both bits should be present everywhere
    expected = preset | bit_value(FailureBit.PUTDOWN_FAIL)
    assert np.all(event_mask == expected)


def test_apply_bernoulli_is_deterministic_for_same_seed_and_calls() -> None:
    n = 100
    eligible = np.zeros(n, dtype=bool)
    eligible[10:90] = True

    # Run twice with same seed on fresh masks/RNGs
    m1 = np.zeros(n, dtype=np.uint64)
    m2 = np.zeros(n, dtype=np.uint64)

    rng1 = np.random.default_rng(2026)
    rng2 = np.random.default_rng(2026)

    apply_bernoulli_event_inplace(
        event_mask=m1,
        eligible=eligible,
        p_fail=0.25,
        bit=FailureBit.TRANSPORT_FAIL,
        rng=rng1,
    )
    apply_bernoulli_event_inplace(
        event_mask=m2,
        eligible=eligible,
        p_fail=0.25,
        bit=FailureBit.TRANSPORT_FAIL,
        rng=rng2,
    )

    assert np.array_equal(m1, m2)


@pytest.mark.slow
def test_apply_bernoulli_rate_is_reasonable_statistically() -> None:
    # Optional: distribution sanity check (avoid in normal fast CI runs).
    n = 50_000
    p = 0.2
    eligible = np.ones(n, dtype=bool)
    event_mask = np.zeros(n, dtype=np.uint64)
    rng = np.random.default_rng(0)

    apply_bernoulli_event_inplace(
        event_mask=event_mask,
        eligible=eligible,
        p_fail=p,
        bit=FailureBit.PUTDOWN_FAIL,
        rng=rng,
    )

    bv = bit_value(FailureBit.PUTDOWN_FAIL)
    hits = (event_mask & bv) != 0
    phat = hits.mean()

    # 5-sigma bound for binomial proportion
    sigma = np.sqrt(p * (1 - p) / n)
    assert abs(phat - p) <= 5 * sigma


def test_apply_bernoulli_empty_event_mask_is_noop() -> None:
    event_mask = np.zeros(0, dtype=np.uint64)
    eligible = np.zeros(0, dtype=bool)
    rng = np.random.default_rng(0)
    apply_bernoulli_event_inplace(
        event_mask, eligible, 1.0, FailureBit.PUTDOWN_FAIL, rng
    )
    assert event_mask.size == 0


def test_apply_bernoulli_negative_p_raises_valueerror() -> None:
    n = 10
    event_mask = np.zeros(n, dtype=np.uint64)
    eligible = np.ones(n, dtype=bool)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        apply_bernoulli_event_inplace(
            event_mask, eligible, -0.1, FailureBit.PUTDOWN_FAIL, rng
        )
    # assert np.all(event_mask == 0)


def test_apply_bernoulli_p_gt_1_raises_valueerror() -> None:
    n = 10
    event_mask = np.zeros(n, dtype=np.uint64)
    eligible = np.zeros(n, dtype=bool)
    eligible[[1, 3, 9]] = True
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        apply_bernoulli_event_inplace(
            event_mask, eligible, 1.5, FailureBit.PUTDOWN_FAIL, rng
        )


def test_set_event_bit_inplace_sets_only_eligible() -> None:
    n = 10
    event_mask = np.zeros(n, dtype=np.uint64)
    eligible = np.zeros(n, dtype=bool)
    eligible[[0, 5]] = True

    set_event_bit_inplace(event_mask, eligible, FailureBit.COLLISION_INEVITABLE)

    bv = bit_value(FailureBit.COLLISION_INEVITABLE)
    assert np.all(event_mask[eligible] == bv)
    assert np.all(event_mask[~eligible] == 0)


def test_eligible_from_indices_basic_and_duplicates() -> None:
    m = eligible_from_indices(8, [1, 1, 6])
    assert m.dtype == bool
    assert m.shape == (8,)
    assert np.array_equal(np.where(m)[0], np.array([1, 6], dtype=np.int64))


def test_eligible_from_moves_identity_and_ignores_unknown() -> None:
    class Dummy:
        pass

    a, b, c = Dummy(), Dummy(), Dummy()
    all_moves = [a, b]
    subset = [b, c]  # c not in all_moves -> ignored

    m = eligible_from_moves(all_moves, subset)
    assert np.array_equal(m, np.array([False, True]))


def test_write_primary_events_to_moves_calls_set_failure_event() -> None:
    moves = [Mock(), Mock(), Mock()]
    primary = np.array([1, 2, 3], dtype=np.int32)

    write_primary_events_to_moves(moves, primary)

    for mv, ev in zip(moves, primary, strict=True):
        mv.set_failure_event.assert_called_once_with(int(ev))


def test_finalize_events_to_moves_suppresses_and_sets_fail_mask_if_requested() -> None:
    # Create moves as mocks; finalize should call set_failure_event and optionally set fail_mask.
    moves = [Mock(), Mock()]

    # Mask with multiple bits; per your precedence NO_ATOM > ...
    event_mask = np.array(
        [
            mask_of(FailureBit.NO_ATOM, FailureBit.PICKUP_FAIL),
            mask_of(FailureBit.PICKUP_FAIL, FailureBit.PUTDOWN_FAIL),
        ],
        dtype=np.uint64,
    )

    primary = finalize_events_to_moves(moves, event_mask, store_mask_on_move=True)

    # After suppression: first collapses to NO_ATOM; second suppresses PUTDOWN and keeps PICKUP
    assert primary[0] == int(FailureEvent.NO_ATOM)
    assert primary[1] == int(FailureEvent.PICKUP_FAIL)

    moves[0].set_failure_event.assert_called_once_with(int(FailureEvent.NO_ATOM))
    moves[1].set_failure_event.assert_called_once_with(int(FailureEvent.PICKUP_FAIL))

    # store_mask_on_move=True should attach fail_mask attributes on mocks
    assert hasattr(moves[0], "fail_mask")
    assert hasattr(moves[1], "fail_mask")
    assert moves[0].fail_mask == int(mask_of(FailureBit.NO_ATOM))
    assert moves[1].fail_mask == int(mask_of(FailureBit.PICKUP_FAIL))
