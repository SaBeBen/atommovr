import os
import numpy as np
from atommover.utils.core import Configurations
from atommover.utils import make_single_species_gif
import pytest

from atommover.utils.AtomArray import AtomArray
from atommover.algorithms.single_species import (
	Hungarian,
	ParallelHungarian,
	ParallelLBAP,
	BCv2,
	PCFA,
	Tetris,
	BalanceAndCompact,
	GeneralizedBalance
)
from atommover.algorithms.source.Hungarian_works import (
	parallel_LBAP_algorithm_works,
	parallel_Hungarian_algorithm_works,
)
from atommover.utils.imaging.visualization import visualize_move_batches, visualize_batch_moves_on_image


def _centered_target_mask(array_shape: tuple[int, int], target_shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
	mask = np.zeros(array_shape, dtype=int)
	r0 = max(0, (array_shape[0] - target_shape[0]) // 2)
	c0 = max(0, (array_shape[1] - target_shape[1]) // 2)
	mask[r0 : r0 + target_shape[0], c0 : c0 + target_shape[1]] = 1
	return mask, (r0, c0)


def _default_source_state(array_shape: tuple[int, int], target_shape: tuple[int, int]) -> np.ndarray:
	rows, cols = array_shape
	t_rows, t_cols = target_shape
	state = np.zeros(array_shape, dtype=int)
	band_rows = min(rows, t_rows + 2)
	band_cols = min(cols, t_cols + 2)
	state[:band_rows, :] = 1
	state[:, :band_cols] = 1
	state[:, -band_cols:] = 1
	return state


def _pcfa_source_state(array_shape: tuple[int, int], target_shape: tuple[int, int]) -> np.ndarray:
	rows, cols = array_shape
	state = np.zeros(array_shape, dtype=int)
	right_band = max(2, target_shape[1] // 2)
	top_band = min(rows, target_shape[0] + 2)
	state[:top_band, :] = 1
	state[:, cols - right_band :] = 1
	return state


def _build_array(array_shape: tuple[int, int], target_shape: tuple[int, int], initializer) -> tuple[AtomArray, np.ndarray, tuple[int, int]]:
	mask, origin = _centered_target_mask(array_shape, target_shape)
	state = initializer(array_shape, target_shape)
	state[mask == 1] = 0

	arr = AtomArray(list(array_shape), n_species=1)
	arr.matrix[:, :, 0] = state
	arr.target = mask.reshape(array_shape[0], array_shape[1], 1)
	return arr, mask, origin


def _contains_target_block(state: np.ndarray, target_shape: tuple[int, int]) -> bool:
	"""Return True if any submatrix matches an all-ones block of target_shape."""
	rows, cols = state.shape
	t_rows, t_cols = target_shape
	if t_rows > rows or t_cols > cols:
		return False
	block = np.ones(target_shape, dtype=int)
	for r0 in range(rows - t_rows + 1):
		for c0 in range(cols - t_cols + 1):
			sub = state[r0 : r0 + t_rows, c0 : c0 + t_cols]
			if np.array_equal(sub, block):
				return True
	return False


def _line_shift_state(num_cols: int, fill: int) -> tuple[np.ndarray, np.ndarray]:
	state = np.zeros((1, num_cols), dtype=int)
	state[0, :fill] = 1
	target = np.zeros_like(state)
	target[0, num_cols - fill :] = 1
	return state, target


ALGORITHM_CASES = [
	# {	
	# 	"name": "BalanceAndCompact",
	# 	"cls": BalanceAndCompact,
	# 	"array_shape": (6,6),
	# 	"target_shape": (4,4),
	# 	"initializer": _default_source_state,
	# 	"kwargs": {"do_ejection": False},
	# },
	# {
	# 	"name": "GeneralizedBalance",
	# 	"cls": GeneralizedBalance,
	# 	"array_shape": (6,6),
	# 	"target_shape": (4,4),
	# 	"initializer": _default_source_state,
	# 	"kwargs": {"do_ejection": False},
	# },
	# {
	# 	"name": "Hungarian",
	# 	"cls": Hungarian,
	# 	"array_shape": (6,6),
	# 	"target_shape": (4,4),
	# 	"initializer": _default_source_state,
	# 	"kwargs": {"do_ejection": False},
	# },
	# {
	# 	"name": "ParallelHungarian",
	# 	"cls": ParallelHungarian,
	# 	"array_shape": (6,6),
	# 	"target_shape": (4,4),
	# 	"initializer": _default_source_state,
	# 	"kwargs": {"do_ejection": False},
	# },
	# {
	# 	"name": "ParallelLBAP",
	# 	"cls": ParallelLBAP,
	# 	"array_shape": (6,6),
	# 	"target_shape": (4,4),
	# 	"initializer": _default_source_state,
	# 	"kwargs": {"do_ejection": False},
	# },
	{
		"name": "BCv2",
		"cls": BCv2,
		"array_shape": (6,6),
		"target_shape": (4,4),
		"initializer": _default_source_state,
		"kwargs": {"do_ejection": False},
	},
	{
		"name": "PCFA",
		"cls": PCFA,
		"array_shape": (4,9),
		"target_shape": (4,4),
		"initializer": _pcfa_source_state,
		"kwargs": {"do_ejection": False},
	},
	{
		"name": "Tetris",
		"cls": Tetris,
		"array_shape": (6,6),
		"target_shape": (4,4),
		"initializer": _default_source_state,
		"kwargs": {"do_ejection": False},
	},
	
]


@pytest.mark.parametrize("case", ALGORITHM_CASES, ids=lambda case: case["name"])
def test_single_species_algorithms_cover_target_shapes(case):
	array_shape = case["array_shape"]
	target_shape = case["target_shape"]
	initializer = case["initializer"]
	arr, mask, (r0, c0) = _build_array(array_shape, target_shape, initializer)
	algo = case["cls"]()
	kwargs = case.get("kwargs", {})

	_, move_batches, success = algo.get_moves(arr, **kwargs)
	assert success, f"{case['name']} reported failure"

	visualize_move_batches(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan")
	visualize_batch_moves_on_image(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan Overlay")

	arr.evaluate_moves(move_batches)
	submatrix = arr.matrix[r0 : r0 + target_shape[0], c0 : c0 + target_shape[1], 0]
	assert np.array_equal(submatrix, np.ones(target_shape, dtype=int)), f"{case['name']} did not fill the target region"


@pytest.mark.parametrize("case", ALGORITHM_CASES, ids=lambda case: case["name"])
def test_single_species_algorithms_natively(case):
	array_shape = case["array_shape"]
	target_shape = case["target_shape"]

	# use AtomArray initializer that fills the target and leaves enough atoms outside
	arr = AtomArray(list(array_shape), n_species=1)
	arr.load_tweezers()
	arr.image(savename=f"test_{case['name']}_init.png")
	arr.generate_target(Configurations.MIDDLE_FILL, middle_size=target_shape, occupation_prob=0.6)

	algo = case["cls"]()
	kwargs = case.get("kwargs", {})

	_, move_batches, success = algo.get_moves(arr, **kwargs)

	visualize_move_batches(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan")
	visualize_batch_moves_on_image(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Move Plan Overlay")

	assert success, f"{case['name']} reported failure"

	arr.evaluate_moves(move_batches)
	arr.image(savename=f"test_{case['name']}_final.png")

	filled = _contains_target_block(arr.matrix[:, :, 0], target_shape)
	assert filled, f"{case['name']} did not realize the required {target_shape} block anywhere in the array"

	make_single_species_gif(arr, move_batches, savename=f"test_{case['name']}_rearrangement.gif")


@pytest.mark.parametrize("case", ALGORITHM_CASES, ids=lambda case: case["name"])
def test_single_species_multiple_shots_natively(case):
	array_shape = case["array_shape"]
	target_shape = case["target_shape"]

	algo = case["cls"]()
	kwargs = case.get("kwargs", {})

	preferred_initial_shape = algo.preferred_initial_shape(target_size=target_shape[0])
	if preferred_initial_shape is not None:
		array_shape = (preferred_initial_shape[1], preferred_initial_shape[1])
		print(f"{case['name']} prefers initial shape {preferred_initial_shape} for target shape {array_shape}")
		
	# use AtomArray initializer that fills the target and leaves enough atoms outside
	arr = AtomArray(list(array_shape), n_species=1)

	n_shots = case.get("n_shots", 10)
	for shot in range(n_shots):
		arr.load_tweezers()
		arr.generate_target(Configurations.MIDDLE_FILL, middle_size=target_shape, occupation_prob=0.6)
		_, move_batches, success = algo.get_moves(arr, **kwargs)

		visualize_move_batches(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Shot {shot} Move Plan")
		visualize_batch_moves_on_image(arr, move_batches, save_path=None, title_suffix=f"{case['name']} Shot {shot} Move Plan Overlay")

		assert success, f"{case['name']} reported failure on shot {shot}"

		arr.evaluate_moves(move_batches)
		arr.image(savename=f"test_{case['name']}_shot{shot}_final.png")

		filled = _contains_target_block(arr.matrix[:, :, 0], target_shape)
		assert filled, f"{case['name']} did not realize the required {target_shape} block anywhere in the array on shot {shot}"

		make_single_species_gif(arr, move_batches, savename=f"test_{case['name']}_shot{shot}_rearrangement.gif")


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

