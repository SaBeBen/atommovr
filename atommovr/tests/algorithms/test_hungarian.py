import numpy as np

from atommovr.algorithms.source import Hungarian_works as hw


def _make_random_case(
    rng: np.random.Generator,
    shape: tuple[int, int],
    fill_prob: float = 0.35,
    target_prob: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random binary matrix/target pair."""
    matrix: np.ndarray = (rng.random(shape) < fill_prob).astype(np.uint8, copy=False)
    target: np.ndarray = (rng.random(shape) < target_prob).astype(np.uint8, copy=False)
    return matrix, target


def _serialize_move_set(move_set: list) -> list[list[tuple[int, int, int, int]]]:
    """Convert nested Move objects into plain tuples for equality checks."""
    serialized: list[list[tuple[int, int, int, int]]] = []
    for move_group in move_set:
        serialized.append(
            [
                (move.from_row, move.from_col, move.to_row, move.to_col)
                for move in move_group
            ]
        )
    return serialized


class TestDEFINE_CURRENT_AND_TARGET_FAST:
    def test_matches_original_when_target_has_trailing_singleton_axis(self) -> None:
        """
        Regression test: fast coordinate extraction must preserve 2D coordinates
        when the target has shape (rows, cols, 1).
        """
        matrix = np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=np.uint8,
        )
        target = np.array(
            [
                [[0], [1], [0]],
                [[1], [0], [0]],
                [[1], [0], [1]],
            ],
            dtype=np.uint8,
        )

        ref_current, ref_target = hw.define_current_and_target(matrix, target)
        new_current, new_target = hw.define_current_and_target_fast(matrix, target)

        assert ref_current == new_current
        assert ref_target == new_target
        assert all(len(pos) == 2 for pos in new_current)
        assert all(len(pos) == 2 for pos in new_target)


class TestGENERATE_ASSIGNMENTS_FAST:
    def test_returns_only_2d_coordinates_with_singleton_target_axis(self) -> None:
        """
        Regression test: fast assignments must remain (row, col) pairs when the
        target includes a singleton species axis.
        """
        matrix = np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )
        target = np.array(
            [
                [[0], [1], [0]],
                [[1], [0], [0]],
            ],
            dtype=np.uint8,
        )

        assignments = hw.generate_assignments_fast(matrix, target, [])

        for start, end in assignments:
            assert len(start) == 2
            assert len(end) == 2


## test functions to verify that this


class TestDefineCurrentAndTarget:
    def test_matches_original_on_random_square_inputs(self) -> None:
        """
        Behavioral regression test: vectorized coordinate extraction must match
        the original function exactly on representative square grids.
        """
        rng = np.random.default_rng(0)

        for side in [1, 2, 5, 10, 20]:
            for _ in range(100):
                matrix, target = _make_random_case(rng, (side, side))

                ref_current, ref_target = hw.define_current_and_target(matrix, target)
                new_current, new_target = hw.define_current_and_target_fast(
                    matrix, target
                )

                assert new_current == ref_current
                assert new_target == ref_target

    def test_handles_empty_and_full_masks(self) -> None:
        """
        Edge-case regression test: coordinate extraction should behave correctly
        when one or both outputs are empty.
        """
        matrix = np.zeros((4, 4), dtype=np.uint8)
        target = np.zeros((4, 4), dtype=np.uint8)

        ref_current, ref_target = hw.define_current_and_target(matrix, target)
        new_current, new_target = hw.define_current_and_target_fast(matrix, target)

        assert new_current == ref_current == []
        assert new_target == ref_target == []


class TestGenerateCostMatrix:
    def test_matches_original_on_random_inputs(self) -> None:
        """
        Behavioral regression test: broadcasted pairwise distances must match
        the original nested-loop implementation.
        """
        rng = np.random.default_rng(1)

        for n_current in [0, 1, 2, 5, 10]:
            for n_target in [0, 1, 3, 7]:
                for _ in range(50):
                    current = [
                        tuple(x) for x in rng.integers(0, 20, size=(n_current, 2))
                    ]
                    target = [tuple(x) for x in rng.integers(0, 20, size=(n_target, 2))]

                    ref_cost = hw.generate_cost_matrix(current, target)
                    new_cost = hw.generate_cost_matrix_fast(current, target)

                    assert new_cost.shape == ref_cost.shape
                    assert np.array_equal(new_cost, ref_cost)

    def test_matches_known_small_example(self) -> None:
        """
        Sanity test: the refactor should reproduce a simple hand-checkable case.
        """
        current = [(0, 0), (3, 4)]
        target = [(0, 4), (3, 0)]

        ref_cost = hw.generate_cost_matrix(current, target)
        new_cost = hw.generate_cost_matrix_fast(current, target)

        expected = np.array(
            [
                [4.0, 3.0],
                [3.0, 4.0],
            ],
            dtype=np.float64,
        )

        assert np.array_equal(ref_cost, expected)
        assert np.array_equal(new_cost, expected)


class TestGenerateAssignments:
    def test_matches_original_on_random_square_inputs(self) -> None:
        """
        Behavioral regression test: assignment generation must preserve the
        original source-target pairing order.
        """
        rng = np.random.default_rng(2)

        for side in [2, 4, 6, 10]:
            for _ in range(100):
                matrix, target = _make_random_case(rng, (side, side))

                ref_assignments = hw.generate_assignments(matrix, target, [])
                new_assignments = hw.generate_assignments_fast(matrix, target, [])

                assert new_assignments == ref_assignments

    def test_returns_empty_when_no_assignable_pairs_exist(self) -> None:
        """
        Edge-case regression test: empty assignment problems should return an
        empty list rather than erroring.
        """
        matrix = np.zeros((5, 5), dtype=np.uint8)
        target = np.zeros((5, 5), dtype=np.uint8)

        ref_assignments = hw.generate_assignments(matrix, target, [])
        new_assignments = hw.generate_assignments_fast(matrix, target, [])

        assert ref_assignments == []
        assert new_assignments == []


class TestHungarianAlgorithmWorks:
    def test_fast_version_matches_original_on_small_random_cases(self) -> None:
        """
        End-to-end regression test: the light refactor should preserve final
        configuration, move sequence, and success flag on small problems.
        """
        rng = np.random.default_rng(3)

        for side in [6, 8, 10]:
            for _ in range(20):
                matrix, target = _make_random_case(rng, (side, side))

                ref_config, ref_moves, ref_success = hw.Hungarian_algorithm_works(
                    matrix.copy(),
                    target.copy(),
                    do_ejection=False,
                    final_size=[],
                )
                new_config, new_moves, new_success = hw.Hungarian_algorithm_works_fast(
                    matrix.copy(),
                    target.copy(),
                    do_ejection=False,
                    final_size=[],
                )

                assert np.array_equal(new_config, ref_config)
                assert _serialize_move_set(new_moves) == _serialize_move_set(ref_moves)
                assert new_success == ref_success
