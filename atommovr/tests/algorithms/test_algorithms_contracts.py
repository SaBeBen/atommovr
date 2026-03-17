import numpy as np
import pytest

from atommovr.algorithms.Algorithm_class import Algorithm
from atommovr.algorithms.dual_species import InsideOut, NaiveParHung
from atommovr.algorithms.single_species import (
    BCv2,
    BalanceAndCompact,
    GeneralizedBalance,
    Hungarian,
    ParallelHungarian,
    ParallelLBAP,
)
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.core import Configurations

# Add new built-in algorithm classes here as they are created
BUILTIN_ALGORITHMS = [
    Algorithm,
    ParallelHungarian,
    ParallelLBAP,
    GeneralizedBalance,
    Hungarian,
    BCv2,
    BalanceAndCompact,
    InsideOut,
    NaiveParHung,
]

# Specify which algorithms are dual-species algorithms, to be used in tests that need to know this distinction.
DUAL_SPECIES_ALGORITHMS = {
    InsideOut,
    NaiveParHung,
}

REQUIRED_METHODS = [
    "__repr__",
    "get_moves",
    "get_success_flag",
]


def _make_single_species_array() -> AtomArray:
    # For simplicity, we just use an empty array to test types of algorithms' outputs.
    arr = AtomArray(shape=[4, 4], n_species=1)
    arr.generate_target(Configurations.MIDDLE_FILL)
    return arr


def _make_dual_species_array() -> AtomArray:
    # For simplicity, we just use an empty array to test types of algorithms' outputs.
    arr = AtomArray(shape=[4, 4], n_species=2)
    arr.generate_target(Configurations.CHECKERBOARD)
    return arr


@pytest.mark.parametrize("algo_cls", BUILTIN_ALGORITHMS)
def test_builtin_algorithms_are_algorithm_subclasses(algo_cls) -> None:
    algo = algo_cls()
    assert isinstance(algo, Algorithm)


@pytest.mark.parametrize("algo_cls", BUILTIN_ALGORITHMS)
@pytest.mark.parametrize("method_name", REQUIRED_METHODS)
def test_builtin_algorithms_have_required_methods(algo_cls, method_name: str) -> None:
    algo = algo_cls()
    method = getattr(algo, method_name, None)
    assert callable(
        method
    ), f"{algo_cls.__name__}.{method_name} is missing or not callable"


@pytest.mark.parametrize("algo_cls", BUILTIN_ALGORITHMS)
def test_builtin_algorithms_repr_contract(algo_cls) -> None:
    algo = algo_cls()
    rep = repr(algo)
    assert isinstance(rep, str)
    assert len(rep.strip()) > 0


@pytest.mark.parametrize("algo_cls", BUILTIN_ALGORITHMS)
def test_builtin_algorithms_get_moves_return_types(algo_cls) -> None:
    """Test that get_moves returns (AtomArray, list, bool)."""
    algo = algo_cls()

    if algo_cls in DUAL_SPECIES_ALGORITHMS:
        arr = _make_dual_species_array()
    else:
        arr = _make_single_species_array()

    config, move_set, success_flag = algo.get_moves(arr)

    assert isinstance(
        config, (np.ndarray, AtomArray)
    ), f"{algo_cls.__name__}.get_moves() config should be np.ndarray or AtomArray, got {type(config)}"
    assert isinstance(
        move_set, list
    ), f"{algo_cls.__name__}.get_moves() move_set should be list, got {type(move_set)}"
    assert isinstance(
        success_flag, bool
    ), f"{algo_cls.__name__}.get_moves() success_flag should be bool, got {type(success_flag)}"
