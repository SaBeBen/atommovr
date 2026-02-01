import os
import numpy as np
import pytest

from atommover.utils.core import Configurations, array_shape_for_geometry
from atommover.utils import make_single_species_gif
from atommover.utils.AtomArray import AtomArray
from atommover.algorithms.single_species import (
	Hungarian,
	ParallelHungarian,
	ParallelLBAP,
	BCv2,
	PCFA,
	Tetris,
	BalanceAndCompact,
	GeneralizedBalance,
)
from atommover.algorithms.source.Hungarian_works import (
	parallel_LBAP_algorithm_works,
	parallel_Hungarian_algorithm_works,
)
from atommover.utils.imaging.visualization import visualize_move_batches, visualize_batch_moves_on_image
from atommover.utils.errormodels import (
	ZeroNoise,
	UniformVacuumTweezerError,
	YbRydbergAODErrorModel,
)
import random


def _centered_target_mask(array_shape: tuple[int, int], target_size: int) -> tuple[np.ndarray, tuple[int, int]]:
	mask = np.zeros(array_shape, dtype=int)
	r0 = max(0, (array_shape[0] - target_size) // 2)
	c0 = max(0, (array_shape[1] - target_size) // 2)
	mask[r0 : r0 + target_size, c0 : c0 + target_size] = 1
	return mask, (r0, c0)


def _default_source_state(array_shape: tuple[int, int], target_size: int) -> np.ndarray:
	rows, cols = array_shape
	t = target_size
	state = np.zeros(array_shape, dtype=int)
	band_rows = min(rows, t + 2)
	band_cols = min(cols, t + 2)
	state[:band_rows, :] = 1
	state[:, :band_cols] = 1
	state[:, -band_cols:] = 1
	return state


def _pcfa_source_state(array_shape: tuple[int, int], target_size: int) -> np.ndarray:
	rows, cols = array_shape
	state = np.zeros(array_shape, dtype=int)
	right_band = max(2, target_size // 2)
	top_band = min(rows, target_size + 2)
	state[:top_band, :] = 1
	state[:, cols - right_band :] = 1
	return state


def _build_array(array_shape: tuple[int, int], target_size: int, initializer) -> tuple[AtomArray, np.ndarray, tuple[int, int]]:
	mask, origin = _centered_target_mask(array_shape, target_size)
	state = initializer(array_shape, target_size)
	state[mask == 1] = 0

	arr = AtomArray(list(array_shape), n_species=1)
	arr.matrix[:, :, 0] = state
	arr.target = mask.reshape(array_shape[0], array_shape[1], 1)
	return arr, mask, origin


def _contains_target_block(state: np.ndarray, target_size: int) -> bool:
	rows, cols = state.shape
	t = target_size
	if t > rows or t > cols:
		return False
	block = np.ones((t, t), dtype=int)
	for r0 in range(rows - t + 1):
		for c0 in range(cols - t + 1):
			if np.array_equal(state[r0 : r0 + t, c0 : c0 + t], block):
				return True
	return False


def _line_shift_state(num_cols: int, fill: int) -> tuple[np.ndarray, np.ndarray]:
	state = np.zeros((1, num_cols), dtype=int)
	state[0, :fill] = 1
	target = np.zeros_like(state)
	target[0, num_cols - fill :] = 1
	return state, target


ALGORITHM_CASES = [
	{
		"name": "Hungarian",
		"cls": Hungarian,
		"target_size": 4,
		"initializer": _default_source_state,
		"kwargs": {"do_ejection": False},
	},
	{
		"name": "ParallelHungarian",
		"cls": ParallelHungarian,
		"target_size": 4,
		"initializer": _default_source_state,
		"kwargs": {"do_ejection": False},
	},
	{
		"name": "ParallelLBAP",
		"cls": ParallelLBAP,
		"target_size": 4,
		"initializer": _default_source_state,
		"kwargs": {"do_ejection": False},
	},
	{
		"name": "BalanceAndCompact",
		"cls": BalanceAndCompact,
		"target_size": 4,
		"initializer": _default_source_state,
		"kwargs": {"do_ejection": False},
	},
	{
		"name": "GeneralizedBalance",
		"cls": GeneralizedBalance,
		"target_size": 4,
		"initializer": _default_source_state,
		"kwargs": {"do_ejection": False},
	},
	{
		"name": "BCv2",
		"cls": BCv2,
		"target_size": 4,
		"initializer": _default_source_state,
		"kwargs": {"do_ejection": False},
	},
	{
		"name": "PCFA",
		"cls": PCFA,
		"target_size": 4,
		"initializer": _pcfa_source_state,
		"kwargs": {"do_ejection": False},
	},
	{
		"name": "Tetris",
		"cls": Tetris,
		"target_size": 4,
		"initializer": _default_source_state,
		"kwargs": {"do_ejection": False},
	},
]


@pytest.mark.parametrize("case", ALGORITHM_CASES, ids=lambda case: case["name"])
def test_single_species_algorithms_cover_target_shapes(case):
	target_size = case["target_size"]
	initializer = case["initializer"]
	algo = case["cls"]()

	# compute array shape using central geometry helper
	array_shape = tuple(array_shape_for_geometry(getattr(algo, "preferred_geometry_spec", None), target_size))

	arr, mask, (r0, c0) = _build_array(array_shape, target_size, initializer)
	_, move_batches, success = algo.get_moves(arr, **case.get("kwargs", {}))
	assert success, f"{case['name']} reported failure"

	visualize_move_batches(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan")
	# visualize_batch_moves_on_image(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan Overlay")

	arr.evaluate_moves(move_batches)
	submatrix = arr.matrix[r0 : r0 + target_size, c0 : c0 + target_size, 0]
	assert np.array_equal(submatrix, np.ones((target_size, target_size), dtype=int)), f"{case['name']} did not fill the target region"


@pytest.mark.parametrize("case", ALGORITHM_CASES, ids=lambda case: case["name"])
def test_single_species_algorithms_natively(case):
	target_size = case["target_size"]
	algo = case["cls"]()

	# choose array shape from geometry helper
	array_shape = tuple(array_shape_for_geometry(getattr(algo, "preferred_geometry_spec", None), target_size))

	arr = AtomArray(list(array_shape), n_species=1)
	arr.load_tweezers()
	arr.generate_target(Configurations.MIDDLE_FILL, middle_size=(target_size, target_size), occupation_prob=0.6)

	_, move_batches, success = algo.get_moves(arr, **case.get("kwargs", {}))

	visualize_move_batches(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan")
	# visualize_batch_moves_on_image(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan Overlay")

	assert success, f"{case['name']} reported failure"

	arr.evaluate_moves(move_batches)
	filled = _contains_target_block(arr.matrix[:, :, 0], target_size)
	assert filled, f"{case['name']} did not realize the required {target_size} block anywhere in the array"


@pytest.mark.parametrize("case", ALGORITHM_CASES, ids=lambda case: case["name"])
def test_single_species_multiple_shots_natively(case):
	target_size = case["target_size"]
	algo = case["cls"]()

	array_shape = tuple(array_shape_for_geometry(getattr(algo, "preferred_geometry_spec", None), target_size))
	arr = AtomArray(list(array_shape), n_species=1)

	n_shots = case.get("n_shots", 5)
	for shot in range(n_shots):
		arr.load_tweezers()
		arr.generate_target(Configurations.MIDDLE_FILL, middle_size=(target_size, target_size), occupation_prob=0.6)
		_, move_batches, success = algo.get_moves(arr, **case.get("kwargs", {}))

		visualize_move_batches(arr, move_batches, save_path=None, title_suffix=f"{case['name']}_{shot}_Move_Plan")
		# visualize_batch_moves_on_image(arr, move_batches, save_path=None, title_suffix=f"{case['name']}_{shot}_Move_Plan_Overlay")

		assert success, f"{case['name']} reported failure on shot {shot}"

		arr.evaluate_moves(move_batches)
		filled = _contains_target_block(arr.matrix[:, :, 0], target_size)
		assert filled, f"{case['name']} did not realize the required {target_size} block anywhere in the array on shot {shot}"

		make_single_species_gif(arr, move_batches, savename=f"test_{case['name']}_shot{shot}_rearrangement")


def test_parallel_assignment_algorithms_complete_long_paths():
	state, target = _line_shift_state(12, 6)
	final_lbap, _, lbap_success = parallel_LBAP_algorithm_works(state.copy(), target.copy(), round_lim=50)
	assert lbap_success, "Parallel LBAP failed to complete deterministic long-path assignment"
	assert np.array_equal(final_lbap, target)

	final_hung, _, hung_success = parallel_Hungarian_algorithm_works(state.copy(), target.copy(), round_lim=50)
	assert hung_success, "Parallel Hungarian failed to complete deterministic long-path assignment"
	assert np.array_equal(final_hung, target)


def test_hungarian_smoke():
	shape = [6, 6]
	L = 4
	arr = AtomArray(shape, n_species=1)
	init = np.zeros(shape, dtype=int)
	# Populate two donor bands to guarantee surplus atoms.
	init[0, :] = 1
	init[1, 0:2] = 1
	init[4:, :] = 1
	arr.matrix[:, :, 0] = init

	r0 = (shape[0] - L) // 2
	c0 = (shape[1] - L) // 2
	target = np.zeros(shape, dtype=int)
	target[r0:r0 + L, c0:c0 + L] = 1
	arr.target[:, :, 0] = target

	algo = Hungarian()
	_, moves, success = algo.get_moves(arr, do_ejection=True)
	assert success
	arr.evaluate_moves(moves)
	assert np.array_equal(arr.matrix[r0:r0 + L, c0:c0 + L, 0], np.ones((L, L), dtype=int))


ERROR_MODELS = [ZeroNoise, UniformVacuumTweezerError, YbRydbergAODErrorModel]


@pytest.mark.parametrize("error_model_cls", ERROR_MODELS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("case", ALGORITHM_CASES, ids=lambda case: case["name"])
def test_algorithms_with_error_models(case, error_model_cls):
	"""Run each algorithm with each error model to ensure behavior and stability.

	- For `ZeroNoise` we expect success and filling of the target region.
	- For other models we only assert that evaluation runs without exceptions and
	  that the array shape/dtype are preserved.
	"""
	# make randomness deterministic for test reproducibility
	random.seed(0)

	target_size = case["target_size"]
	algo = case["cls"]()

	array_shape = tuple(array_shape_for_geometry(getattr(algo, "preferred_geometry_spec", None), target_size))

	arr = AtomArray(list(array_shape), n_species=1)
	arr.load_tweezers()
	arr.generate_target(Configurations.MIDDLE_FILL, middle_size=(target_size, target_size), occupation_prob=0.6)

	# attach the error model instance
	err = error_model_cls()
	arr.error_model = err

	_, move_batches, success = algo.get_moves(arr, **case.get("kwargs", {}))

	# visualize the planned moves
	visualize_move_batches(arr, move_batches, save_path=None, title_suffix=f"{case['name']}_{error_model_cls.__name__}_Move_Plan")
	# visualize_batch_moves_on_image(arr, move_batches, save_path=None, title_suffix=f"{case['name']}_{error_model_cls.__name__}_Move_Plan_Overlay")

	# evaluation should not raise
	try:
		arr.evaluate_moves(move_batches)
	except Exception as e:
		pytest.fail(f"Evaluation raised with {error_model_cls.__name__} on {case['name']}: {e}")

	# basic sanity checks
	assert isinstance(arr.matrix, np.ndarray)
	assert arr.matrix.shape[0] == array_shape[0] and arr.matrix.shape[1] == array_shape[1]

	target = arr.get_target()[:, :, 0]

	# For zero-noise we expect the algorithm to succeed and fill the target
	# if error_model_cls is ZeroNoise:
	assert success, f"{case['name']} reported failure with ZeroNoise"
	submatrix = arr.matrix[
		(target == 1)
	]
	assert np.all(submatrix == 1), f"{case['name']} did not fill the target region with ZeroNoise"
	# else:
	# 	# For other error models we do not enforce success, but reasonable filling
	# 	submatrix = arr.matrix[
	# 		(target == 1)
	# 	]
	# 	fill_fraction = np.sum(submatrix) / np.size(submatrix)
	# 	assert 0.2 <= fill_fraction <= 1.0, f"{case['name']} had unreasonable fill fraction {fill_fraction:.2f} with {error_model_cls.__name__}"
