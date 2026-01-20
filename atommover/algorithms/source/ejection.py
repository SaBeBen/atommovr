# Ejection subroutine

import copy
import numpy as np

from atommover.utils.core import *
from atommover.utils.move_utils import *

def ejection(
    init_config: np.ndarray,
    target_config: np.ndarray,
    final_size = None,
    method: str = "sublattice",
) -> tuple[list, np.ndarray]:
    """Remove leftover atoms once the target is filled.

    Parameters
    ----------
    init_config, target_config : np.ndarray
        Two-dimensional single-species occupancy grids.
    final_size : tuple | None
        Optional bounds used by the legacy sublattice routine.
    method : {"sublattice", "column"}
        ``"sublattice"`` runs the historic row/column slicing procedure that
        incrementally shuttles atoms out of each quadrant. ``"column"`` executes a
        simplified, maximally parallel routine that strips entire columns from the
        left and right edges, ejecting them one at a time. The column strategy is
        fastest when the desired target occupies a compact block in the center of
        the array and any atoms beyond that block can be discarded. Because it has
        no notion of buffer zones or future donor usage, it should not be used
        before the target is fully populated; otherwise, required donor atoms near
        the perimeter may be lost permanently.
    """
    matrix = copy.deepcopy(init_config)
    move_list = []

    if final_size == None:
        final_size = np.shape(init_config)

    if method not in {"sublattice", "column"}:
        raise ValueError(f"Unknown ejection method '{method}'.")

    if method == "column":
        matrix, column_moves = _parallel_square_ejection(matrix, target_config)
        move_list.extend(column_moves)
        return move_list, matrix

    #Identify unwanted atoms
    left_eject, bot_eject, top_eject, right_eject = generate_eject_coordinates(matrix, target_config)
    #Left sublattice ejection; Left = -1, Right = 1
    if len(left_eject) > 0:
        # print(f"Left Ejection: {left_eject}")
        row_min, row_max, col_min, col_max = generate_sublattice(matrix, left_eject, final_size, direction = 'left')
        matrix, move_list = hor_ejection(matrix, target_config, move_list, row_min, row_max, col_min, col_max, direction = -1)    
    #Right sublattice ejection
        #Bottom sublattice ejection; Top = -1, Bottom = 1
    if len(bot_eject) > 0:
        # print(f"Bottom Ejection: {bot_eject}")
        row_min, row_max, col_min, col_max = generate_sublattice(matrix, bot_eject, final_size, direction = 'bottom')
        matrix, move_list = ver_ejection(matrix, target_config, move_list, row_min, row_max, col_min, col_max, direction = 1)
    if len(right_eject) > 0:
        # print(f"Right Ejection: {right_eject}")
        row_min, row_max, col_min, col_max = generate_sublattice(matrix, right_eject, final_size, direction = 'right')
        matrix, move_list = hor_ejection(matrix, target_config, move_list, row_min, row_max, col_min, col_max, direction = 1)
    #Top sublattice ejection
    if len(top_eject) > 0:
        # print(f"Top Ejection: {top_eject}")
        row_min, row_max, col_min, col_max = generate_sublattice(matrix, top_eject, final_size, direction = 'top') 
        matrix, move_list = ver_ejection(matrix, target_config, move_list, row_min, row_max, col_min, col_max, direction = -1)

    return move_list, matrix


def _parallel_square_ejection(matrix: np.ndarray, target_config: np.ndarray) -> tuple[np.ndarray, list[list[Move]]]:
    """Strip columns then rows outside the target bounds with maximal parallelism.

    Each column strictly outside the target's bounding box is removed in one
    parallel batch, followed by analogous sweeps for rows above and below the
    target. The runtime is linear in the number of exterior rows/columns. This
    assumes ``target_config`` is a binary mask matching ``matrix``.

    Potential dangers: if the target occupies disjoint regions or if downstream
    logic still needs atoms outside the bounding box (e.g., future donor usage),
    those atoms vanish immediately. The routine also ignores stray atoms *inside*
    the bounding box; run the traditional ejection if interior cleanup is needed.
    """
    temp_matrix = copy.deepcopy(matrix)
    move_batches: list[list[Move]] = []

    if temp_matrix.ndim != 2:
        raise ValueError("Column-strip ejection only supports 2-D single-species arrays.")

    target_coords = np.argwhere(target_config == 1)
    if target_coords.size == 0:
        raise ValueError("Target consists of all zeros; column ejection would remove every atom.")

    rows, cols = temp_matrix.shape
    min_row = int(np.min(target_coords[:, 0]))
    max_row = int(np.max(target_coords[:, 0]))
    min_col = int(np.min(target_coords[:, 1]))
    max_col = int(np.max(target_coords[:, 1]))

    # Process columns strictly to the left of the target.
    for col in range(0, min_col):
        batch = [Move(row, col, row, col - 1) for row in range(rows) if temp_matrix[row, col] == 1]
        if batch:
            move_batches.append(batch)
            temp_matrix, _ = move_atoms(temp_matrix, batch)

    # Process columns strictly to the right of the target.
    for col in range(cols - 1, max_col, -1):
        batch = [Move(row, col, row, col + 1) for row in range(rows) if temp_matrix[row, col] == 1]
        if batch:
            move_batches.append(batch)
            temp_matrix, _ = move_atoms(temp_matrix, batch)

    # Process rows strictly above the target.
    for row in range(0, min_row):
        batch = [Move(row, col, row - 1, col) for col in range(cols) if temp_matrix[row, col] == 1]
        if batch:
            move_batches.append(batch)
            temp_matrix, _ = move_atoms(temp_matrix, batch)

    # Process rows strictly below the target.
    for row in range(rows - 1, max_row, -1):
        batch = [Move(row, col, row + 1, col) for col in range(cols) if temp_matrix[row, col] == 1]
        if batch:
            move_batches.append(batch)
            temp_matrix, _ = move_atoms(temp_matrix, batch)

    return temp_matrix, move_batches

# Ejection from Left and Right Side; Left = -1, Right = 1
def hor_ejection(matrix, target_config, move_list, row_min, row_max, col_min, col_max, direction):
    # If there are excess atoms, do ejection in the while loop
    while np.sum(matrix[row_min:row_max+1, col_min: col_max+1]) >= np.sum(target_config[row_min:row_max+1, col_min:col_max+1]) \
        and np.array_equal(matrix[row_min:row_max+1, col_min:col_max+1].reshape(row_max+1-row_min, col_max+1-col_min),
                           target_config[row_min:row_max+1, col_min:col_max+1].reshape(row_max+1-row_min, col_max+1-col_min)) == False:
        atom_dict = {}
        target_replacement_dict = {}

        # Iterate each row in the sublattice
        for row in range(row_min, row_max + 1):
            # check if there are too many atoms
            # print(f"Row: {row}, atoms in region: {matrix[row,col_min: col_max+1]}, targets: {target_config[row,col_min: col_max+1]}, N filled targets: {np.sum(np.dot(target_config[row, col_min:col_max+1].reshape(col_max+1-col_min),
            #                                                                  matrix[row,col_min: col_max+1].reshape(col_max+1-col_min)))}")
            if np.sum(matrix[row,col_min: col_max+1]) > np.sum(target_config[row,col_min: col_max + 1]):
                col = left_right_atom_in_row(matrix[row,col_min: col_max+1], direction)
                atom_dict[row] = col + col_min
                
            #check that the target sites are filled
            if np.sum(target_config[row, col_min:col_max+1]) > np.sum(np.dot(target_config[row, col_min:col_max+1].reshape(col_max+1-col_min),
                                                                             matrix[row,col_min: col_max+1].reshape(col_max+1-col_min))):
                # find the rightmost position where there SHOULD be a target atom but there isn't
                atom_absent_cols = []
                for col_ind in range(col_min, col_max + 1):
                    if target_config[row, col_ind] == 1 and matrix[row, col_ind] == 0:
                        atom_absent_cols.append(col_ind)
            
                # then find the rightmost position where there is an atom TO THE Right of this previous position
                for atom_absent_col in atom_absent_cols:
                    if direction == -1:
                        try:
                            atom_to_move_col = min([col_ind for col_ind in range(atom_absent_col, col_max+1) if matrix[row, col_ind] == 1])
                        except ValueError:
                            atom_to_move_col = None
                    else:
                        try:
                            atom_to_move_col = max([col_ind for col_ind in range(col_min, atom_absent_col + 1) if matrix[row, col_ind] == 1])
                        except ValueError:
                            atom_to_move_col = None
                    # try:
                    target_replacement_dict[row] = atom_to_move_col
                    # except UnboundLocalError:
                    #     pass
            
        
        rows_to_move = []
        rows_to_fill = []

        if direction == -1:
            try:
                farthest_col_in_atom_dict = max(value for value in atom_dict.values() if value is not None)
            except ValueError:
                farthest_col_in_atom_dict = 0
            try:
                farthest_col_in_target_r_dict = max(value for value in target_replacement_dict.values() if value is not None)
            except ValueError:
                farthest_col_in_target_r_dict = 0
            col = max(farthest_col_in_atom_dict, farthest_col_in_target_r_dict)
        else:
            try:
                farthest_col_in_atom_dict = min(value for value in atom_dict.values() if value is not None)
            except ValueError:
                farthest_col_in_atom_dict = col_max
            try:
                farthest_col_in_target_r_dict = min(value for value in target_replacement_dict.values() if value is not None)
            except ValueError:
                farthest_col_in_target_r_dict = col_max
            col = min(farthest_col_in_atom_dict, farthest_col_in_target_r_dict)

        while col_max >= col >= col_min: #From first col to the right bound of the submatrix
            moves_in_scan = []

            for row in range(row_min, row_max + 1):
                if matrix[row][col] == 1 and target_config[row][col] == 1 and row in rows_to_fill:
                    rows_to_fill.remove(row)

            rows_to_move.extend([key for key, value in atom_dict.items() if value == col])
            rows_to_fill.extend([key for key, value in target_replacement_dict.items() if value == col])
            #rows_to_move.extend([key for key, value in target_replacement_dict.items() if value == col])

            for row in range(row_min, row_max+1):
                if row in rows_to_move:
                    move = Move(row, col, row, col + direction)
                    moves_in_scan.append(move)
                if row in rows_to_fill:
                    move = Move(row, col, row, col + direction)
                    moves_in_scan.append(move)
            
            if len(moves_in_scan) > 0:
                move_list.append(moves_in_scan)
                matrix, _ = move_atoms(matrix, moves_in_scan)

            col += direction
    return matrix, move_list

# Ejection from Top and Bottom Side
def ver_ejection(matrix, target_config, move_list, row_min, row_max, col_min, col_max, direction):
    while np.sum(matrix[row_min:row_max+1, col_min: col_max+1]) >= np.sum(target_config[row_min:row_max+1, col_min:col_max+1]) \
        and np.array_equal(matrix[row_min:row_max+1, col_min:col_max+1].reshape(row_max+1-row_min, col_max+1-col_min),
                           target_config[row_min:row_max+1, col_min:col_max+1].reshape(row_max+1-row_min, col_max+1-col_min)) == False:
        atom_dict = {}
        target_replacement_dict = {}
        for col in range(col_min, col_max + 1):
            # check if there are too many atoms
            if np.sum(matrix[row_min:row_max+1,col]) > np.sum(target_config[row_min:row_max+1,col]):
                row = top_bot_atom_in_col(matrix[row_min:row_max+1,col], direction)
                atom_dict[col] = row + row_min
                
            #check that the target sites are filled
            if np.sum(target_config[row_min:row_max+1,col]) > np.sum(np.dot(target_config[row_min:row_max+1,col].reshape(row_max+1-row_min),
                                                                            matrix[row_min:row_max+1,col].reshape(row_max+1-row_min))):
                # find the rightmost position where there SHOULD be a target atom but there isn't
                atom_absent_rows = []
                for row_ind in range(row_min, row_max + 1):
                    if target_config[row_ind, col] == 1 and matrix[row_ind, col] == 0:
                        atom_absent_rows.append(row_ind)
            
                # then find the rightmost position where there is an atom TO THE Right of this previous position
                for atom_absent_row in atom_absent_rows:
                    if direction == -1:
                        try:
                            atom_to_move_row = min([row_ind for row_ind in range(atom_absent_row, row_max+1) if matrix[row_ind, col] == 1])
                        except ValueError:
                            pass
                    else:
                        try:
                            atom_to_move_row = max([row_ind for row_ind in range(row_min, atom_absent_row + 1) if matrix[row_ind, col] == 1])
                        except ValueError:
                            pass
                    target_replacement_dict[col] = atom_to_move_row

        cols_to_move = []
        cols_to_fill = []

        if direction == -1:
            try:
                farthest_row_in_atom_dict = max(value for value in atom_dict.values() if value is not None)
            except ValueError:
                farthest_row_in_atom_dict = 0
            try:
                farthest_row_in_target_r_dict = max(value for value in target_replacement_dict.values() if value is not None)
            except ValueError:
                farthest_row_in_target_r_dict = 0
            row = max(farthest_row_in_atom_dict, farthest_row_in_target_r_dict)
        else:
            try:
                farthest_row_in_atom_dict = min(value for value in atom_dict.values() if value is not None)
            except ValueError:
                farthest_row_in_atom_dict = 0
            try:
                farthest_row_in_target_r_dict = min(value for value in target_replacement_dict.values() if value is not None)
            except ValueError:
                farthest_row_in_target_r_dict = 0
            row = max(farthest_row_in_atom_dict, farthest_row_in_target_r_dict)

        while row_max >= row >= row_min: #From first col to the right bound of the submatrix
            moves_in_scan = []

            for col in range(col_min, col_max + 1):
                if matrix[row][col] == 1 and target_config[row][col] == 1 and col in cols_to_fill:
                    cols_to_fill.remove(col)

            cols_to_move.extend([key for key, value in atom_dict.items() if value == row])
            cols_to_fill.extend([key for key, value in target_replacement_dict.items() if value == row])

            for col in range(col_min, col_max+1):
                if col in cols_to_move:
                    move = Move(row, col, row + direction, col)
                    moves_in_scan.append(move)
                if col in cols_to_fill:
                    move = Move(row, col, row + direction, col)
                    moves_in_scan.append(move)
            
            if len(moves_in_scan) > 0:
                move_list.append(moves_in_scan)
                matrix, _ = move_atoms(matrix, moves_in_scan)

            row += direction
    return matrix, move_list

def generate_sublattice(matrix, sub_eject, final_size, direction):
    row_max = 0
    row_min = len(matrix)
    col_max = 0
    col_min = len(matrix[0])
    # Define the boundary of sublattice
    for row, col in sub_eject:
        if row > row_max:
            row_max = row
        if row < row_min: 
            row_min = row
        if col > col_max:
            col_max = col 
        if col < col_min:
            col_min = col   
    if direction == 'left':
        return max(row_min, final_size[0]), min(row_max,final_size[1]), max(final_size[2],int(0)), min(final_size[3],col_max)
    if direction == 'right':
        return max(row_min, final_size[0]), min(row_max, final_size[1]), max(col_min, final_size[2]), min(int(len(matrix[0]) - 1), final_size[3])
    if direction == 'top':
        return max(int(0), final_size[0]), min(row_max, final_size[1]), max(col_min, final_size[2]), min(col_max, final_size[3])
    if direction == 'bottom':
        return max(row_min, final_size[0]), min(int(len(matrix) - 1), final_size[1]), max(col_min, final_size[2]), min(col_max, final_size[3])
    
def generate_eject_coordinates(matrix, target_config):
    left_eject, bot_eject, top_eject, right_eject = [], [], [], []
    len_x = len(matrix[0]) # columns
    len_y = len(matrix) # rows
    for x in range(len_x):
        for y in range(len_y):
            if matrix[y][x] == 1 and target_config[y][x] == 0:
                if x >= y and x < len_y - y:
                    # left_eject.append((y, x))
                    top_eject.append((y, x))
                elif x >= y and x >= len_y - y:
                    # bot_eject.append((y, x))
                    right_eject.append((y, x))
                elif x < y and x <= len_y - y:
                    # top_eject.append((y, x))
                    left_eject.append((y, x))
                elif x <= y and x >= len_y - y:
                    # right_eject.append((y, x))
                    bot_eject.append((y, x))
    return left_eject, bot_eject, top_eject, right_eject