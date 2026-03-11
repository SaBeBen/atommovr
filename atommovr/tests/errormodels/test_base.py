import numpy as np
import pytest

from atommovr.utils.ErrorModel import ErrorModel
from atommovr.utils.failure_policy import FailureBit, bit_value
from atommovr.tests.support.helpers import mask_of


def test_get_atom_loss_base_model_is_noop() -> None:
    em = ErrorModel(seed=0)
    state = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    new_state, loss_flag = em.get_atom_loss(state, evolution_time=1.23, n_species=1)

    assert np.array_equal(new_state, state)
    assert loss_flag is False

def test_repr_returns_name() -> None:
    em = ErrorModel(seed=0)
    assert repr(em) == "Generic ErrorModel object"

@pytest.mark.parametrize(
    "rate_attr, method_name, bit",
    [
        ("pickup_fail_rate", "apply_pickup_errors_mask", FailureBit.PICKUP_FAIL),
        ("putdown_fail_rate", "apply_putdown_errors_mask", FailureBit.PUTDOWN_FAIL),
        ("accel_fail_rate", "apply_accel_errors_mask", FailureBit.ACCEL_FAIL),
        ("decel_fail_rate", "apply_decel_errors_mask", FailureBit.DECEL_FAIL),
    ],
)
def test_mask_methods_p0_noop(rate_attr: str, method_name: str, bit: FailureBit) -> None:
    n = 20
    event_mask = np.zeros(n, dtype=np.uint64)
    eligible = np.ones(n, dtype=bool)

    em = ErrorModel(seed=0)
    setattr(em, rate_attr, 0.0)

    getattr(em, method_name)(event_mask, eligible)

    assert np.all(event_mask == 0)

@pytest.mark.parametrize(
    "rate_attr, method_name",
    [
        ("pickup_fail_rate", "apply_pickup_errors_mask"),
        ("putdown_fail_rate", "apply_putdown_errors_mask"),
        ("accel_fail_rate", "apply_accel_errors_mask"),
        ("decel_fail_rate", "apply_decel_errors_mask"),
    ],
)
def test_mask_methods_empty_event_mask_noop(rate_attr: str, method_name: str) -> None:
    em = ErrorModel(seed=0)
    setattr(em, rate_attr, 1.0)
    m = np.zeros(0, dtype=np.uint64)
    eligible = np.zeros(0, dtype=bool)
    getattr(em, method_name)(m, eligible)
    assert m.size == 0

@pytest.mark.parametrize(
    "rate_attr, method_name, bit",
    [
        ("pickup_fail_rate", "apply_pickup_errors_mask", FailureBit.PICKUP_FAIL),
        ("putdown_fail_rate", "apply_putdown_errors_mask", FailureBit.PUTDOWN_FAIL),
        ("accel_fail_rate", "apply_accel_errors_mask", FailureBit.ACCEL_FAIL),
        ("decel_fail_rate", "apply_decel_errors_mask", FailureBit.DECEL_FAIL),
    ],
)
def test_mask_methods_p1_sets_all_eligible(rate_attr: str, method_name: str, bit: FailureBit) -> None:
    n = 21
    event_mask = np.zeros(n, dtype=np.uint64)
    eligible = np.zeros(n, dtype=bool)
    eligible[::3] = True  # some eligible

    em = ErrorModel(seed=0)
    setattr(em, rate_attr, 1.0)

    getattr(em, method_name)(event_mask, eligible)

    bv = bit_value(bit)
    assert np.all(event_mask[eligible] == bv)
    assert np.all(event_mask[~eligible] == 0)


@pytest.mark.parametrize(
    "rate_attr, method_name, bit",
    [
        ("pickup_fail_rate", "apply_pickup_errors_mask", FailureBit.PICKUP_FAIL),
        ("putdown_fail_rate", "apply_putdown_errors_mask", FailureBit.PUTDOWN_FAIL),
        ("accel_fail_rate", "apply_accel_errors_mask", FailureBit.ACCEL_FAIL),
        ("decel_fail_rate", "apply_decel_errors_mask", FailureBit.DECEL_FAIL),
    ],
)
def test_mask_methods_do_not_clear_existing_bits(rate_attr: str, method_name: str, bit: FailureBit) -> None:
    n = 10
    preset = bit_value(FailureBit.NO_ATOM)

    event_mask = np.zeros(n, dtype=np.uint64)
    event_mask[:] = preset

    eligible = np.ones(n, dtype=bool)

    em = ErrorModel(seed=0)
    setattr(em, rate_attr, 1.0)

    getattr(em, method_name)(event_mask, eligible)

    expected = preset | bit_value(bit)
    assert np.all(event_mask == expected)


@pytest.mark.parametrize(
    "rate_attr, method_name",
    [
        ("pickup_fail_rate", "apply_pickup_errors_mask"),
        ("putdown_fail_rate", "apply_putdown_errors_mask"),
        ("accel_fail_rate", "apply_accel_errors_mask"),
        ("decel_fail_rate", "apply_decel_errors_mask"),
    ],
)
def test_mask_methods_deterministic_for_same_seed_and_call_sequence(
    rate_attr: str, method_name: str
) -> None:
    n = 100
    eligible = np.zeros(n, dtype=bool)
    eligible[10:90] = True

    # Two models with same seed should produce identical masks
    em1 = ErrorModel(seed=2026)
    em2 = ErrorModel(seed=2026)
    setattr(em1, rate_attr, 0.25)
    setattr(em2, rate_attr, 0.25)

    m1 = np.zeros(n, dtype=np.uint64)
    m2 = np.zeros(n, dtype=np.uint64)

    getattr(em1, method_name)(m1, eligible)
    getattr(em2, method_name)(m2, eligible)

    assert np.array_equal(m1, m2)


def test_inevitable_collision_mask_sets_bit_deterministically() -> None:
    n = 12
    event_mask = np.zeros(n, dtype=np.uint64)
    eligible = np.zeros(n, dtype=bool)
    eligible[[0, 3, 7]] = True

    em = ErrorModel(seed=0)
    em.apply_inevitable_collision_mask(event_mask, eligible)

    bv = bit_value(FailureBit.COLLISION_INEVITABLE)
    assert np.all(event_mask[eligible] == bv)
    assert np.all(event_mask[~eligible] == 0)


def test_avoidable_collision_mask_sets_bit_deterministically() -> None:
    n = 12
    event_mask = np.zeros(n, dtype=np.uint64)
    eligible = np.zeros(n, dtype=bool)
    eligible[[1, 2, 11]] = True

    em = ErrorModel(seed=0)
    em.apply_avoidable_collision_mask(event_mask, eligible)

    bv = bit_value(FailureBit.COLLISION_AVOIDABLE)
    assert np.all(event_mask[eligible] == bv)
    assert np.all(event_mask[~eligible] == 0)
