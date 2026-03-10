# Tests for utility functions (in progress)

import numpy as np

from atommover.utils.AtomArray import AtomArray
from atommover.utils.Move import Move, FailureEvent, FailureFlag
from atommover.utils.move_utils import get_move_list_from_AOD_cmds, get_AOD_cmds_from_move_list

def test_move():
    move = Move(1,0,2,3)
    assert move._move_str() == move.__repr__()
    assert move.__repr__() == '(1, 0) -> (2, 3)'
    assert move.from_row == 1
    assert move.from_col == 0
    assert move.to_row == 2
    assert move.to_col == 3

class TestAODToMoveListAndViceVersa:
    def test_move_list_AOD_conversion(self):
        move = [Move(3,5,4,6)]
        horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = get_AOD_cmds_from_move_list(np.zeros([10,10,1]),move)
        move_list = get_move_list_from_AOD_cmds(horiz_AOD_cmds, vert_AOD_cmds)
        assert move[0].to_col == move_list[0].to_col
        assert move[0].to_row == move_list[0].to_row
        assert move[0].from_col == move_list[0].from_col
        assert move[0].from_row == move_list[0].from_row

    def test_regression_move_list_aod_roundtrip_preserves_row_col_order(self) -> None:
        move_seq = [Move(1, 4, 2, 3)]  # (from_row, from_col, to_row, to_col)

        horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = get_AOD_cmds_from_move_list(
            np.zeros((10, 10, 1), dtype=np.uint8), move_seq
        )
        assert parallel_success_flag is True
        move_list = get_move_list_from_AOD_cmds(horiz_AOD_cmds, vert_AOD_cmds)
        assert len(move_list) == 1
        assert move_list[0] == move_seq[0]  # uses Move.__eq__
    

    def test_regression_move_list_aod_roundtrip_non_square_matrix(self) -> None:
        move_seq = [Move(2, 7, 3, 6)]
        matrix = np.zeros((6, 12, 1), dtype=np.uint8)  # non-square

        horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = get_AOD_cmds_from_move_list(matrix, move_seq)
        assert parallel_success_flag is True

        roundtrip = get_move_list_from_AOD_cmds(horiz_AOD_cmds, vert_AOD_cmds)
        assert len(roundtrip) == 1
        assert roundtrip[0] == move_seq[0]