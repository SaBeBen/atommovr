# Core functions and classes for moving atoms

import random
import copy
import numpy as np
from enum import IntEnum
from collections import Counter

from atommover.utils.Move import Move
from atommover.utils.ErrorModel import ErrorModel
from atommover.utils.errormodels import ZeroNoise
from atommover.utils.core import PhysicalParams

class MoveType(IntEnum):
    """
    Class to be used in conjunction with `move_atoms()`
    """
    ILLEGAL_MOVE = 0
    LEGAL_MOVE = 1
    EJECT_MOVE = 2
    NO_ATOM_TO_MOVE = 3
    
def move_atoms(init_matrix, moves: list[Move], error_model = ZeroNoise(), params: PhysicalParams = PhysicalParams(), look_for_flag: bool = False):
    """
    Moving atoms to adjacent sites (including diagonals) in parallel.
    """
    matrix_out = copy.deepcopy(init_matrix)
    if np.max(init_matrix) > 1:
        raise Exception("Variable `init_matrix` cannot have values outside of {0,1}. ")

    # make sure `moves` is a list and not just a singular `Move` object
    try:
        moves[0]
    except TypeError:
        moves = [moves]

    # evaluating moves from error model
    moves = error_model.get_move_errors(init_matrix, moves)

    # prescreening moves to remove any intersecting tweezers
    matrix_out, duplicate_move_inds = _find_and_resolve_crossed_moves(moves, matrix_out)

    # applying moves on
    matrix_out, failed_moves, flags = _apply_moves(init_matrix, 
                                                  matrix_out, 
                                                  moves, 
                                                  duplicate_move_inds,
                                                  look_for_flag = look_for_flag)

    # if there are multiple atoms in a trap, they repel each other
    if np.max(matrix_out) > 1:
        for i in range(len(matrix_out)):
            for j in range(len(matrix_out[0])):
                if matrix_out[i,j] > 1:
                    matrix_out[i,j] = 0

    # calculating the time it took the atoms to be moved
    max_distance = 0
    for move in moves:
        dist = move.distance*params.spacing + (error_model.putdown_time + error_model.pickup_time)*params.AOD_speed
        if dist > max_distance:
            max_distance = dist

        move_time = max_distance/params.AOD_speed

    # evaluating atom loss process from error model
    matrix_out, _ = error_model.get_atom_loss(matrix_out, move_time, n_species = 1)

    return matrix_out, [failed_moves, flags]
    
def _get_duplicate_vals_from_list(l):
    return [k for k,v in Counter(l).items() if v>1]

def _find_and_resolve_crossed_moves(move_list: list, matrix_copy: np.ndarray) -> 'tuple[np.ndarray, list]':
    """
    Identifies sets of moves where the AOD tweezers cross over each other (and destroy the atoms).
    NB: this ONLY works for moves where you only move by one column or one row.
    """
    # 1. getting midpoints of moves
    midpoints = []
    for move in move_list:
        midpoints.append((move.midx, move.midy))

    # 2. Finding duplicate midpoints
    duplicate_vals = _get_duplicate_vals_from_list(midpoints)

    # 3. Sorting duplicate entries into distinct sets
    crossed_move_sets = []
    duplicate_move_inds = []
    for i in range(len(duplicate_vals)):
        crossed_move_sets.append([])
    if len(crossed_move_sets) > 0:
        for m_ind, move in enumerate(move_list):
            try:
                d_ind = duplicate_vals.index((move.midx, move.midy))
                crossed_move_sets[d_ind].append(m_ind)
                duplicate_move_inds.append(m_ind)
            except ValueError:
                pass
        # 4. iterature through the sets of overlapping moves
        for crossed_move_set in crossed_move_sets:
            # 4.1. check to see if there are atoms that would be moved
            for move_ind in crossed_move_set:
                move = move_list[move_ind]
                if matrix_copy[move.from_row][move.from_col] == 1:
                    # 4.2. if so, check whether the tweezer fails to pick up the atom
                    # if it picks up the atom, then the atom is ejected due to the collision with the other tweezer
                    if move.failure_flag != 1:
                        matrix_copy[move.from_row][move.from_col] = 0
                else:
                    move.failure_flag = 3 # meaning that there is no atom to move
    return matrix_copy, duplicate_move_inds


def _apply_moves(init_matrix: np.ndarray, 
                matrix_out: np.ndarray, 
                moves: list, 
                duplicate_move_inds: list = [],
                look_for_flag: bool = False) -> tuple[np.ndarray, list, list]:
    """ 
    Applies moves to an array of atoms (represented by `matrix_out`). 
    The function assumes that any moves which involve crossing tweezers
    have already been filtered out by `find_and_resolve_crossed_moves()`.
    
    NB: `init_matrix` is the initial array before crossed moves were resolved,
    and `matrix_out` is the array following resolution of crossed moves. 
    """
    failed_moves = []
    flags = []
    # evaluate and run each move
    for move_ind, move in enumerate(moves):
        if move_ind in duplicate_move_inds:
            failed_moves.append(move_ind)
            flags.append(move.failure_flag)
        else:
            # fail flag code for the move: SUCCESS[0], PICKUPFAIL[1], PUTDOWNFAIL[2], NOATOM[3]
            # move.failure_flag = random.choices([0, 1, 2], weights=[1-pickup_fail_rate-putdown_fail_rate, pickup_fail_rate, putdown_fail_rate])[0]
            
            # Classify the move as: 
            #   a) legal (there is an atom in the pickup position and NO atom in the putdown position),
            #   b) illegal (there is an atom in the pickup pos and an atom in the putdown pos)
            #   c) eject (there is an atom in the pickup pos and the putdown pos is outside of the array)
            #   d) no atom to move (there is NO atom in the pickup pos)

            # if there is an atom in the pickup pos
            val_pickup = init_matrix[move.from_row][move.from_col]
            if isinstance(val_pickup, np.ndarray) and val_pickup.ndim > 0:
                val_pickup = val_pickup.item()
            
            if int(val_pickup) == 1:
                try:
                    # check if there is NO atom in the putdown pos
                    val_putdown = init_matrix[move.to_row][move.to_col]
                    if isinstance(val_putdown, np.ndarray) and val_putdown.ndim > 0:
                        val_putdown = val_putdown.item()

                    if int(val_putdown) == 0 and move.to_col >= 0 and move.to_row >= 0:
                        movetype = MoveType.LEGAL_MOVE
                    # check if there is an atom in the putdown pos
                    elif int(val_putdown) == 1 and move.to_col >= 0 and move.to_row >= 0:
                        movetype = MoveType.ILLEGAL_MOVE
                    elif move.to_col >= 0 and move.to_row >= 0:
                        raise Exception(f"{int(val_putdown)} is not a valid matrix entry.")
                    else:
                        raise IndexError
                except IndexError:
                    movetype = MoveType.EJECT_MOVE
            else: # if there is no atom in the pickup pos
                movetype = MoveType.NO_ATOM_TO_MOVE
                move.failure_flag = 3
            
            # if the move fails due to the atom not being picked up or put down correctly, make a note of this.
            if (move.failure_flag != 0 and look_for_flag):
                failed_moves.append(move_ind) 
                flags.append(move.failure_flag)
                if move.failure_flag == 2: #PUTDOWNFAIL, see above
                    if matrix_out[move.from_row][move.from_col] == 0:
                        raise Exception(f"Error occured in MoveType. There is NO atom at ({move.from_row}, {move.from_col}).")
                    matrix_out[move.from_row][move.from_col] -= 1
            # elif the move is valid, implement it
            elif movetype == MoveType.LEGAL_MOVE or movetype == MoveType.ILLEGAL_MOVE:
                if matrix_out[move.from_row][move.from_col] > 0:
                    matrix_out[move.from_row][move.from_col] -= 1
                    matrix_out[move.to_row][move.to_col] += 1
            # elif the move is an ejection move or moves an atom into an occupied site, remove the atom(s)
            elif movetype == MoveType.EJECT_MOVE:
                if matrix_out[move.from_row][move.from_col] == 0:
                    raise Exception(f"Error occured in MoveType assignment. There is NO atom at ({move.from_row}, {move.from_col}).")
                matrix_out[move.from_row][move.from_col] -= 1
    return matrix_out, failed_moves, flags


## AOD cmd functions ##

AOD_cmd_to_pos_shift = {'1': 0, '2': 1, '3': -1}

def get_move_list_from_AOD_cmds(horiz_AOD_cmds,vert_AOD_cmds):
    move_list = []
    for row_ind in range(len(vert_AOD_cmds)):
        row_cmd = int(vert_AOD_cmds[row_ind])
        if row_cmd == 0:
            pass
        else:
            row_shift = AOD_cmd_to_pos_shift['{}'.format(row_cmd)]
            for col_ind in range(len(horiz_AOD_cmds)):
                col_cmd = int(horiz_AOD_cmds[col_ind])
                if col_cmd != 0 and col_cmd*row_cmd != 1:
                    # make a move
                    col_shift = AOD_cmd_to_pos_shift['{}'.format(col_cmd)]
                    move_list.append(Move(col_ind,row_ind, col_ind+col_shift, row_ind+row_shift))
    return move_list

def get_AOD_cmds_from_move_list(matrix, move_seq):
    row_num = len(matrix)
    col_num = len(matrix[0])
    horiz_AOD_cmds = np.zeros([row_num])
    vert_AOD_cmds = np.zeros([col_num])
    parallel_success_flag = True
    op_matrix = copy.deepcopy(matrix)

    # Generate AOD commands for a given row and column number
    for move in move_seq:
        # Chnage the status of vertical AOD commands
        if move.from_row > move.to_row:
            if vert_AOD_cmds[move.from_row] == 0:
                vert_AOD_cmds[move.from_row] = 3
            elif vert_AOD_cmds[move.from_row] != 3:
                parallel_success_flag = False
                break
        elif move.from_row < move.to_row:
            if vert_AOD_cmds[move.from_row] == 0:
                vert_AOD_cmds[move.from_row] = 2
            elif vert_AOD_cmds[move.from_row] != 2:
                parallel_success_flag = False
                break
        else:
            if vert_AOD_cmds[move.from_row] == 0:
                vert_AOD_cmds[move.from_row] = 1
            elif vert_AOD_cmds[move.from_row] != 1:
                parallel_success_flag = False
                break

        # Change the status of horizontal AOD commands
        if move.from_col > move.to_col:
            if horiz_AOD_cmds[move.from_col] == 0:
                horiz_AOD_cmds[move.from_col] = 3
            elif horiz_AOD_cmds[move.from_col] != 3:
                parallel_success_flag = False
                break
        elif move.from_col < move.to_col:
            if horiz_AOD_cmds[move.from_col] == 0:
                horiz_AOD_cmds[move.from_col] = 2
            elif horiz_AOD_cmds[move.from_col] != 2:
                parallel_success_flag = False
                break
        else:
            if horiz_AOD_cmds[move.from_col] == 0:
                horiz_AOD_cmds[move.from_col] = 1
            elif horiz_AOD_cmds[move.from_col] != 1:
                parallel_success_flag = False
                break
        
        # Check if there is an atom from source position
        if op_matrix[move.from_row][move.from_col] == 0:
            parallel_success_flag = False
            break

    if parallel_success_flag:
        move_list = get_move_list_from_AOD_cmds(vert_AOD_cmds, horiz_AOD_cmds)
        matrix_from_AOD, _ = move_atoms(copy.deepcopy(matrix), move_list)
        matrix_from_seq, _ = move_atoms(copy.deepcopy(matrix), move_seq)

        if not np.array_equal(matrix_from_AOD, matrix_from_seq):
            parallel_success_flag = False

    return horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag