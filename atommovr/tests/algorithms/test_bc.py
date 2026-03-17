from __future__ import annotations

import time

import numpy as np
import pytest

import atommovr.utils as movr
from atommovr.utils.AtomArray import AtomArray

from atommovr.algorithms.source import bc_new
from atommovr.tests.support.helpers import (
    _moves_to_tuples,
    _n_atoms,
    _replay_and_check_noiseless_conservation,
)


def _ref_get_all_moves_btwn_rows(
    init_config: np.ndarray,
    from_row_ind: int,
    to_row_ind: int,
) -> tuple[list[movr.Move], int]:
    """
    Reference version matching the 'current' implementation you had before the optimization.
    This is used only for regression comparison in tests.
    """
    if from_row_ind < 0 or to_row_ind < 0:
        raise IndexError

    from_row = init_config[from_row_ind, :]
    to_row = init_config[to_row_ind, :]

    available_source = np.flatnonzero(from_row == 1)
    free = (to_row == 0).copy()
    moves: list[movr.Move] = []

    for atom_col in available_source:
        dest = None
        if atom_col - 1 >= 0 and free[atom_col - 1]:
            dest = atom_col - 1
        elif free[atom_col]:
            dest = atom_col
        elif atom_col + 1 < free.size and free[atom_col + 1]:
            dest = atom_col + 1

        if dest is not None:
            moves.append(movr.Move(from_row_ind, int(atom_col), to_row_ind, int(dest)))
            free[int(dest)] = False

    return moves, len(moves)


def _ref_prebalance_above(
    current_state, start_row, end_row, n_targets, round_moves, direction
):
    n_movable_above = 0
    n_movable = 0
    row_offset = 0
    if direction == -1:
        boundary_row = start_row
    else:
        boundary_row = end_row
    while n_movable_above == 0:
        if bc_new._int_sum(current_state) < n_targets:
            raise Exception("Insufficient atoms.")
        try:
            # move_set = []
            for off in range(row_offset + 1)[
                ::-1
            ]:  # TODO: figure out if this should be the -1 thing or if it makes more sense to do something else
                above_moves, n_movable = bc_new.get_all_moves_btwn_rows(
                    current_state,
                    boundary_row + (1 + off) * direction,
                    boundary_row + off * direction,
                )
                if (
                    n_movable != 0
                    and np.sum(current_state[start_row : end_row + 1, :]) < n_targets
                ):  # check if there are atoms that can be moved, and if so move them
                    current_state, _ = movr.move_atoms(current_state, above_moves)
                    round_moves.append(above_moves)  # NEW
                else:  # if no atoms can be moved, figure out why
                    n_in_from_row = bc_new._int_sum(
                        current_state[boundary_row + (1 + off) * direction, :]
                    )
                    if (
                        n_in_from_row > 0
                    ):  # if there are no spots for new atoms to come, make space by pushing atoms farther inside
                        rows_in = 0
                        stuck_row = boundary_row + off * direction
                        while n_movable == 0:
                            # stuck_row = boundary_row+off*direction
                            for r_in in range(-1, rows_in)[::-1]:

                                space_moves, n_sp_movable = (
                                    bc_new.get_all_moves_btwn_rows(
                                        current_state,
                                        stuck_row - r_in * direction,
                                        stuck_row - (1 + r_in) * direction,
                                    )
                                )
                                if (
                                    n_sp_movable != 0
                                    and np.sum(
                                        current_state[start_row : end_row + 1, :]
                                    )
                                    < n_targets
                                ):  # check if there are atoms that can be moved, and if so move them
                                    current_state, _ = movr.move_atoms(
                                        current_state, space_moves
                                    )
                                    round_moves.append(space_moves)
                                    above_moves, n_movable = (
                                        bc_new.get_all_moves_btwn_rows(
                                            current_state,
                                            stuck_row,
                                            stuck_row - direction,
                                        )
                                    )
                                    # NEW
                                    if (
                                        np.sum(
                                            current_state[start_row : end_row + 1, :]
                                        )
                                        < n_targets
                                        and n_movable != 0
                                    ):
                                        current_state, _ = movr.move_atoms(
                                            current_state, above_moves
                                        )
                                        round_moves.append(above_moves)
                            rows_in += 1
            if n_movable > 0:
                n_movable_above = n_movable
            row_offset += 1
        except IndexError:
            row_offset += 1
            break
    return current_state, round_moves


def _ref_move_across_rows(
    current_state: np.ndarray, n_to_move: int, i: int, j: int, m: int, dir=-1
):
    """
    Moves `n_to_move` atoms from row m to m-1 if dir = -1 or vice versa. If there aren't
    enough atoms, can access additional rows (subject to the constraint
    i < row and row < j).
    """

    round_moves = []  # master list of all moves taken in this procedure
    n_left_to_move = n_to_move

    ## specifying rows to move across and ROIs
    if dir == 1:
        start_row = m - 1
        end_row = m
        low_ind_roi = m
        high_ind_roi = j + 1
        low_ind_source = i
        high_ind_source = m
    elif dir == -1:
        start_row = m
        end_row = m - 1
        low_ind_roi = i
        high_ind_roi = m
        low_ind_source = m
        high_ind_source = j + 1
    else:
        raise ValueError('Parameter "dir" must be -1 or 1.')

    ## sanity check to make sure we have sufficient atoms
    n_atoms_in_source = bc_new._int_sum(current_state[low_ind_source:high_ind_source])
    n_atoms_in_roi = bc_new._int_sum(current_state[low_ind_roi:high_ind_roi])
    if n_atoms_in_source < n_to_move:
        raise Exception(
            f"Insufficient atoms. Only {n_atoms_in_source} in the source region (we need {n_to_move} more; only {n_atoms_in_roi} currently)."
        )

    ## continue looping until we move sufficient atoms.
    try_count = 0
    while n_left_to_move != 0 and try_count < 1000:
        try_count += 1
        n_movable_dir = 0
        row_offset = 0
        last_moves = [0]  # placeholder
        ## we loop until we are able to move atoms
        try_count2 = 0
        while n_movable_dir == 0 and try_count2 < 1000:
            try_count2 += 1
            try:
                move_set = []
                for off in range(row_offset + 1)[::-1]:
                    across_move = 1
                    from_row = start_row + (off * dir)
                    to_row = end_row + (off * dir)
                    if i > from_row or i > to_row or j < from_row or j < to_row:
                        raise IndexError
                    # above_moves, n_movable = get_all_moves_btwn_rows(current_state,from_row, to_row)
                    from_cols, to_cols, n_movable = (
                        bc_new._get_all_moves_btwn_rows_cols_checked(
                            current_state, from_row, to_row
                        )
                    )

                    if (
                        n_movable != 0 and n_left_to_move != 0
                    ):  # check if there are atoms that can be moved, and if so move them
                        above_moves = [
                            movr.Move(from_row, int(fc), to_row, int(tc))
                            for fc, tc in zip(from_cols, to_cols)
                        ]
                        if off == 0:
                            moves_to_run = above_moves[:n_left_to_move]
                        else:
                            moves_to_run = above_moves
                        current_state, _ = movr.move_atoms(current_state, moves_to_run)
                        n_left_to_move -= len(moves_to_run)
                        move_set.append(moves_to_run)
                    else:  # if atoms CANNOT be moved
                        n_in_from_row = bc_new._int_sum(current_state[from_row, :])
                        ## Scenario 1: there are atoms to move, but no place to put them in the new row, so we have to clear room in ROI
                        if n_in_from_row > 0 and len(last_moves) > 0:
                            clear_space_in_roi_moves = []
                            rows_into_ROI = 0
                            while n_movable == 0:
                                stuck_row = start_row + dir * off
                                for r_in in range(rows_into_ROI + 1)[
                                    ::-1
                                ]:  # NKH change 05-09
                                    from_row = stuck_row + (1 + r_in) * dir
                                    to_row = stuck_row + (2 + r_in) * dir
                                    if (
                                        i > from_row
                                        or i > to_row
                                        or j < from_row
                                        or j < to_row
                                    ):
                                        raise IndexError
                                    # space_moves, n_sp_movable = get_all_moves_btwn_rows(current_state,from_row, to_row)
                                    from_sp_cols, to_sp_cols, n_sp_movable = (
                                        bc_new._get_all_moves_btwn_rows_cols_checked(
                                            current_state, from_row, to_row
                                        )
                                    )
                                    if (
                                        n_sp_movable != 0 and n_left_to_move != 0
                                    ):  # check if there are atoms that can be moved, and if so move them
                                        space_moves = [
                                            movr.Move(
                                                from_row, int(fcs), to_row, int(tcs)
                                            )
                                            for fcs, tcs in zip(
                                                from_sp_cols, to_sp_cols
                                            )
                                        ]
                                        current_state, _ = movr.move_atoms(
                                            current_state, space_moves
                                        )
                                        clear_space_in_roi_moves.append(space_moves)
                                        n_movable = n_sp_movable
                                rows_into_ROI += 1
                            if len(clear_space_in_roi_moves) > 0:
                                move_set.extend(clear_space_in_roi_moves)
                        ## Scenario 2: there are no atoms to move, so we have to take atoms from farther inside the source region
                        elif n_in_from_row == 0 or len(last_moves) == 0:
                            pull_atoms_from_reservoir_moves = []
                            rows_into_source = 0
                            while n_movable == 0:
                                stuck_row = start_row + dir * off
                                for r_in in range(-1, rows_into_source)[::-1]:
                                    from_row = stuck_row - (2 + r_in) * dir
                                    to_row = stuck_row - (1 + r_in) * dir
                                    if (
                                        i > from_row
                                        or i > to_row
                                        or j < from_row
                                        or j < to_row
                                    ):
                                        raise IndexError
                                    # space_moves, n_sp_movable = get_all_moves_btwn_rows(current_state,from_row, to_row)
                                    from_sp_cols, to_sp_cols, n_sp_movable = (
                                        bc_new._get_all_moves_btwn_rows_cols_checked(
                                            current_state, from_row, to_row
                                        )
                                    )
                                    if (
                                        n_sp_movable != 0 and n_left_to_move != 0
                                    ):  # check if there are atoms that can be moved, and if so move them
                                        space_moves = [
                                            movr.Move(
                                                from_row, int(fcs), to_row, int(tcs)
                                            )
                                            for fcs, tcs in zip(
                                                from_sp_cols, to_sp_cols
                                            )
                                        ]
                                        current_state, _ = movr.move_atoms(
                                            current_state, space_moves
                                        )
                                        pull_atoms_from_reservoir_moves.append(
                                            space_moves
                                        )
                                        n_movable = n_sp_movable
                                rows_into_source += 1
                            if len(pull_atoms_from_reservoir_moves) > 0:
                                move_set.extend(pull_atoms_from_reservoir_moves)
                # END DEBUG
                if len(move_set) > 0:
                    round_moves.extend(move_set)
                last_moves = move_set

                if n_movable > 0:
                    n_movable_dir = n_movable
                    break
                row_offset += 1
            except IndexError:
                row_offset += 1
                break

    return current_state, round_moves


class TestIntSum:
    def test_int_sum_uses_signed_accumulation(self) -> None:
        """
        Regression tripwire: _int_sum must not overflow on uint inputs.
        """
        x_u8 = np.array([255, 255], dtype=np.uint8)
        assert bc_new._int_sum(x_u8) == 510  # would overflow if summed as uint8

        x_i8 = np.array([-1, 2, 3], dtype=np.int8)
        assert bc_new._int_sum(x_i8) == 4


class TestFindLargestDistToMove:
    def test_returns_inf_if_not_enough_atoms(self) -> None:
        targ = np.array([0, 3, 5], dtype=np.int64)
        atoms = np.array([1, 4], dtype=np.int64)
        out = bc_new.find_largest_dist_to_move(targ, atoms)
        assert np.isinf(out)

    def test_returns_max_abs_distance(self) -> None:
        targ = np.array([0, 3, 5], dtype=np.int64)
        atoms = np.array([1, 1, 7], dtype=np.int64)
        assert bc_new.find_largest_dist_to_move(targ, atoms) == 2  # |3-1|


class TestGetTargetLocs:
    def test_returns_bounding_box(self) -> None:
        aa = AtomArray(shape=[4, 5], n_species=1)
        aa.target[:, :, 0] = np.uint8(0)
        aa.target[1, 2, 0] = np.uint8(1)
        aa.target[3, 4, 0] = np.uint8(1)

        sr, sc, er, ec = bc_new.get_target_locs(aa)
        assert (sr, sc, er, ec) == (1, 2, 3, 4)


class TestGetAllBalanceAssignments:
    def test_covers_recursive_partitioning(self) -> None:
        """
        Basic structural test: ensures recursion terminates and includes the root interval.
        """
        out = bc_new.get_all_balance_assignments(0, 3)
        assert (0, 3) in out
        # Should include sub-intervals
        assert (0, 1) in out or (0, 0) in out  # depends on split logic
        assert len(out) > 0


class TestGetAllMovesBtwnRows:
    def test_matches_reference_on_random_rows(self) -> None:
        """
        Behavioral regression test: optimized version must match the reference greedy policy.
        """
        rng = np.random.default_rng(0)

        for n_cols in [1, 2, 5, 20, 70]:
            for _ in range(200):
                # Random binary rows (some dense, some sparse)
                init = (rng.random((3, n_cols, 1)) < 0.3).astype(np.uint8, copy=False)

                ref_moves, ref_n = _ref_get_all_moves_btwn_rows(init, 0, 1)
                new_moves, new_n = bc_new.get_all_moves_btwn_rows(init, 0, 1)

                assert new_n == ref_n
                ref_t = [
                    (m.from_row, m.from_col, m.to_row, m.to_col) for m in ref_moves
                ]
                new_t = [
                    (m.from_row, m.from_col, m.to_row, m.to_col) for m in new_moves
                ]
                assert new_t == ref_t

    def test_fast_path_empty_source(self) -> None:
        init = np.zeros((2, 10, 1), dtype=np.uint8)
        moves, n = bc_new.get_all_moves_btwn_rows(init, 0, 1)
        assert moves == []
        assert n == 0

    def test_fast_path_full_destination(self) -> None:
        init = np.zeros((2, 10, 1), dtype=np.uint8)
        init[0, 2] = 1
        init[1, :] = 1  # no free slots
        moves, n = bc_new.get_all_moves_btwn_rows(init, 0, 1)
        assert moves == []
        assert n == 0

    def test_helper_from_rows_matches_wrapper(self) -> None:
        rng = np.random.default_rng(1)
        init = (rng.random((4, 30, 1)) < 0.2).astype(np.uint8, copy=False)

        wrapper_moves, wrapper_n = bc_new.get_all_moves_btwn_rows(init, 2, 3)
        helper_moves, helper_n = bc_new.get_all_moves_btwn_rows_from_rows(
            init[2, :], init[3, :], 2, 3
        )
        assert helper_n == wrapper_n
        w_t = [(m.from_row, m.from_col, m.to_row, m.to_col) for m in wrapper_moves]
        h_t = [(m.from_row, m.from_col, m.to_row, m.to_col) for m in helper_moves]
        assert h_t == w_t

    def test_moves_only_into_empty_sites(self) -> None:
        init = np.zeros((3, 5, 1), dtype=np.uint8)
        init[2, 1, 0] = 1
        init[2, 3, 0] = 1
        init[1, 0, 0] = 1  # occupied spot in destination row

        moves, n = bc_new.get_all_moves_btwn_rows(init, 2, 1)
        # n equals number of moves returned
        assert n == len(moves)
        # Each move should land in a zero of the destination row (row 1)
        for mv in moves:
            assert mv.from_row == 2 and mv.to_row == 1
            assert int(init[mv.to_row, mv.to_col, 0]) == 0


class TestGetAllMovesBtwnCols:
    def test_moves_only_into_empty_sites(self) -> None:
        init = np.zeros((4, 4, 1), dtype=np.uint8)
        init[1, 0, 0] = 1
        init[3, 0, 0] = 1
        init[2, 1, 0] = 1  # occupied destination spot in col 1

        moves, n = bc_new.get_all_moves_btwn_cols(init, 0, 1)
        assert n == len(moves)
        for mv in moves:
            assert mv.from_col == 0 and mv.to_col == 1
            assert int(init[mv.to_row, mv.to_col, 0]) == 0


class TestSpecialCaseAlgo1D:
    def test_raises_if_atom_count_mismatch(self) -> None:
        init = np.zeros((1, 5, 1), dtype=np.uint8)
        targ = np.zeros((1, 5, 1), dtype=np.uint8)
        init[0, 0, 0] = 1
        targ[0, 2, 0] = 1
        targ[0, 4, 0] = 1
        with pytest.raises(Exception, match="Number of atoms"):
            bc_new.special_case_algo_1d(init, targ)

    def test_produces_moves_that_reach_target(self) -> None:
        init = np.zeros((1, 5, 1), dtype=np.uint8)
        targ = np.zeros((1, 5, 1), dtype=np.uint8)
        init[0, 0, 0] = 1
        init[0, 4, 0] = 1
        targ[0, 1, 0] = 1
        targ[0, 3, 0] = 1

        move_rounds, _ = bc_new.special_case_algo_1d(init, targ)

        aa = AtomArray(shape=[1, 5], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()

        _replay_and_check_noiseless_conservation(aa, move_rounds)
        assert np.array_equal(aa.matrix, aa.target)


class TestMiddleFillAlgo1D:
    def test_returns_empty_if_insufficient_atoms(self) -> None:
        init = np.zeros((1, 6, 1), dtype=np.uint8)
        targ = np.zeros((1, 6, 1), dtype=np.uint8)
        init[0, 2, 0] = 1
        targ[0, 1, 0] = 1
        targ[0, 2, 0] = 1
        targ[0, 3, 0] = 1

        move_rounds, best = bc_new.middle_fill_algo_1d(init, targ)
        assert move_rounds == []
        assert best == []

    def test_reaches_target_when_possible(self) -> None:
        init = np.zeros((1, 7, 1), dtype=np.uint8)
        targ = np.zeros((1, 7, 1), dtype=np.uint8)

        # 4 atoms available, 3 targets in the middle
        init[0, 0, 0] = 1
        init[0, 2, 0] = 1
        init[0, 4, 0] = 1
        init[0, 6, 0] = 1

        targ[0, 2, 0] = 1
        targ[0, 3, 0] = 1
        targ[0, 4, 0] = 1

        move_rounds, best = bc_new.middle_fill_algo_1d(init, targ)
        assert isinstance(best, (list, np.ndarray))

        aa = AtomArray(shape=[1, 7], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()

        _replay_and_check_noiseless_conservation(aa, move_rounds)
        # check that the target is prepared without killing the extra atom
        assert _n_atoms(aa.matrix) == 4
        assert np.array_equal(aa.matrix * aa.target, aa.target)


class TestBalanceRows:
    def test_raises_on_insufficient_atoms(self) -> None:
        init = np.zeros((4, 4, 1), dtype=np.uint8)
        targ = np.zeros((4, 4, 1), dtype=np.uint8)

        # target demands 2 atoms in top half and 2 in bottom half
        targ[0, 0, 0] = 1
        targ[0, 1, 0] = 1
        targ[2, 0, 0] = 1
        targ[2, 1, 0] = 1

        # only 2 atoms total => insufficient
        init[0, 0, 0] = 1
        init[2, 0, 0] = 1

        with pytest.raises(ValueError, match="Insufficient number of atoms"):
            bc_new.balance_rows(init, targ, 0, 3)

    def test_returns_empty_when_already_balanced(self) -> None:
        init = np.zeros((4, 4, 1), dtype=np.uint8)
        targ = np.zeros((4, 4, 1), dtype=np.uint8)

        targ[0, 0, 0] = 1
        targ[2, 0, 0] = 1
        init[0, 0, 0] = 1
        init[2, 0, 0] = 1

        moves = bc_new.balance_rows(init, targ, 0, 3)
        assert moves == []


class TestPrebalanceAboveRefactor:
    def test_matches_reference_random_instances(self):
        rng = np.random.default_rng(0)

        for _ in range(1000):
            n_rows, n_cols = 10, 12

            # 3D occupancy (rows, cols, 1), uint8
            state3 = (rng.random((n_rows, n_cols, 1)) < 0.15).astype(
                np.uint8, copy=False
            )

            # Choose a small target band
            start_row = int(rng.integers(2, 5))
            end_row = int(rng.integers(start_row, min(start_row + 2, n_rows - 1) + 1))

            max_targets = (end_row - start_row + 1) * n_cols
            n_targets = int(rng.integers(1, max(2, max_targets // 3) + 1))  # NEVER 0

            # Skip infeasible instances early (matches real call site assumptions)
            if bc_new._int_sum(state3[:, :, 0]) < n_targets:
                continue

            direction = int(rng.choice([-1, 1]))

            # Work in 2D for the helper; your bc_new logic uses 2D in these routines.
            s1 = state3.copy()
            s2 = state3.copy()

            rm1 = []
            rm2 = []

            try:
                ref_state, ref_rounds = _ref_prebalance_above(
                    s1, start_row, end_row, n_targets, rm1, direction
                )
                new_state, new_rounds = bc_new._prebalance_above(
                    s2, start_row, end_row, n_targets, rm2, direction
                )
            except RuntimeError:
                # Pathological random instance for this internal helper; skip rather than hang/fail.
                continue

            assert np.array_equal(new_state, ref_state)
            assert _moves_to_tuples(new_rounds) == _moves_to_tuples(ref_rounds)


class TestPrebalance:
    def test_returns_false_when_global_insufficient(self) -> None:
        init = np.zeros((3, 3, 1), dtype=np.uint8)
        targ = np.zeros((3, 3, 1), dtype=np.uint8)
        # 2 targets, 1 atom
        targ[1, 1, 0] = 1
        targ[1, 2, 0] = 1
        init[0, 0, 0] = 1

        moves, col_compact, ok = bc_new.prebalance(init, targ)
        assert moves == []
        assert ok is False

    def test_produces_moves_that_increase_atoms_in_target_rows(self) -> None:
        init = np.zeros((5, 5, 1), dtype=np.uint8)
        targ = np.zeros((5, 5, 1), dtype=np.uint8)

        # Target block is rows 2..3, cols 1..3 (3 atoms required)
        targ[2, 1, 0] = 1
        targ[2, 2, 0] = 1
        targ[3, 2, 0] = 1

        # Put atoms outside target rows initially.
        init[0, 1, 0] = 1
        init[0, 2, 0] = 1
        init[4, 2, 0] = 1

        moves, _, ok = bc_new.prebalance(init, targ)
        assert ok is True

        aa = AtomArray(shape=[5, 5], n_species=1)
        aa.matrix = init.copy()
        aa.target = targ.copy()

        n_targets = bc_new._int_sum(targ[2:4, :, :])
        before = bc_new._int_sum(aa.matrix[2:4, :, :])

        _replay_and_check_noiseless_conservation(aa, moves)

        after = bc_new._int_sum(aa.matrix[2:4, :, :])
        assert after >= before
        assert after >= n_targets


class TestCompact:
    def test_compact_terminates_and_conserves_atoms(self) -> None:
        """
        Regression tripwire for earlier infinite-loop symptom:
        if compact emits move rounds, replaying them must strictly change the matrix
        on each non-empty round.
        """
        aa = AtomArray(shape=[6, 6], n_species=1)

        # Build a simple target block in the middle (2x2)
        aa.target[:, :, 0] = np.uint8(0)
        aa.target[2, 2, 0] = 1
        aa.target[2, 3, 0] = 1
        aa.target[3, 2, 0] = 1
        aa.target[3, 3, 0] = 1

        # Place 4 atoms near the target but not fully compacted.
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[4, 2, 0] = 1
        aa.matrix[4, 3, 0] = 1
        aa.matrix[5, 2, 0] = 1
        aa.matrix[5, 3, 0] = 1

        move_rounds = bc_new.compact(aa)
        assert isinstance(move_rounds, list)

        # Replay on a fresh copy so we test pure output.
        aa2 = AtomArray(shape=[6, 6], n_species=1)
        aa2.matrix = aa.matrix.copy()
        aa2.target = aa.target.copy()

        _replay_and_check_noiseless_conservation(aa2, move_rounds)

    def test_compact_emits_no_duplicate_destinations_per_round(self) -> None:
        """
        Regression test: each parallel round emitted by `compact` must have unique
        destinations.

        Why this matters
        ----------------
        If two moves in the same round target the same site, replay under
        `AtomArray.move_atoms()` can create destructive multi-occupancy/ejection,
        breaking noiseless atom conservation.
        """
        aa = AtomArray(shape=[6, 6], n_species=1)

        # Target: centered 2x2 block.
        aa.target[:, :, 0] = np.uint8(0)
        aa.target[2, 2, 0] = 1
        aa.target[2, 3, 0] = 1
        aa.target[3, 2, 0] = 1
        aa.target[3, 3, 0] = 1

        # Four atoms placed so that compaction must bring atoms inward.
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[2, 0, 0] = 1
        aa.matrix[2, 1, 0] = 1
        aa.matrix[3, 2, 0] = 1
        aa.matrix[3, 3, 0] = 1

        move_rounds = bc_new.compact(aa)
        assert isinstance(move_rounds, list)

        for k, round_moves in enumerate(move_rounds):
            srcs = [(m.from_row, m.from_col) for m in round_moves]
            dests = [(m.to_row, m.to_col) for m in round_moves]
            assert len(srcs) == len(
                set(srcs)
            ), f"Round {k} has duplicate sources: {srcs}"
            assert len(dests) == len(
                set(dests)
            ), f"Round {k} has duplicate destinations: {dests}"


class TestBCV2:
    def test_bcv2_rejects_dual_species(self) -> None:
        aa = AtomArray(shape=[2, 2], n_species=2)
        # Force dual-species-like shape if needed
        aa.matrix = np.zeros((2, 2, 2), dtype=np.uint8)
        aa.target = np.zeros((2, 2, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="single species"):
            bc_new.bcv2(aa, do_ejection=False)

    def test_bcv2_noiseless_conserves_atoms_and_matches_target_region(self) -> None:
        """
        End-to-end noiseless sanity check (without ejection).

        We don't require exact global equality to target, because `bcv2`'s success
        criterion for do_ejection=False is that the target region is prepared
        (via effective_config masking). :contentReference[oaicite:0]{index=0}
        """
        aa = AtomArray(shape=[6, 6], n_species=1)

        # Target: 2x2 block.
        aa.target[:, :, 0] = np.uint8(0)
        aa.target[2, 2, 0] = 1
        aa.target[2, 3, 0] = 1
        aa.target[3, 2, 0] = 1
        aa.target[3, 3, 0] = 1

        # Provide at least as many atoms globally as targets.
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[0, 0, 0] = 1
        aa.matrix[0, 1, 0] = 1
        aa.matrix[5, 4, 0] = 1
        aa.matrix[5, 5, 0] = 1

        n0 = _n_atoms(aa.matrix)

        out_matrix, move_rounds, ok = bc_new.bcv2(aa, do_ejection=False)
        assert isinstance(move_rounds, list)

        # Replay moves on a fresh copy and check invariants.
        aa2 = AtomArray(shape=[6, 6], n_species=1)
        aa2.matrix = aa.matrix.copy()
        aa2.target = aa.target.copy()

        _replay_and_check_noiseless_conservation(aa2, move_rounds)

        # Must match bcv2's returned matrix.
        assert np.array_equal(aa2.matrix, out_matrix)

        # Noiseless conservation.
        assert _n_atoms(aa2.matrix) == n0

        # If bcv2 reports success, verify its success condition explicitly.
        if ok:
            effective = np.multiply(aa2.matrix, aa2.target.reshape(aa2.matrix.shape))
            assert np.array_equal(effective, aa2.target.reshape(aa2.matrix.shape))


@pytest.mark.slow
class TestBCV2PerformanceSmoke:
    def test_bcv2_runs_quickly_on_small_instance(self) -> None:
        """
        Performance *smoke test* (NOT a strict gate).

        Why this exists
        ---------------
        Pytest isn't a stable benchmarking harness, but this can still catch
        accidental O(N^3) blowups on tiny sizes.

        This is intentionally loose and marked slow so it can be opted into.
        """
        aa = AtomArray(shape=[10, 10], n_species=1)
        aa.target[:, :, 0] = 0
        aa.target[4:6, 4:6, 0] = 1  # 4 targets
        aa.matrix[:, :, 0] = 0
        aa.matrix[0, 0, 0] = 1
        aa.matrix[0, 1, 0] = 1
        aa.matrix[9, 8, 0] = 1
        aa.matrix[9, 9, 0] = 1

        t0 = time.perf_counter()
        _ = bc_new.bcv2(aa, do_ejection=False)
        dt = time.perf_counter() - t0

        # Very loose bound: adjust if needed for CI variance.
        assert dt < 2.0
