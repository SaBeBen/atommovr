import numpy as np
import copy
from collections import deque
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

from atommover.algorithms.Algorithm_class import Algorithm
from atommover.utils.core import random_loading, generate_middle_fifty, Configurations
from atommover.utils.move_utils import Move, move_atoms, get_move_list_from_AOD_cmds
from atommover.algorithms.source.scaling_lower_bound import make_cost_matrix_square
from atommover.algorithms.source.PPSU_weight_matching import bttl_threshold

def parallel_LBAP_algorithm_works(atom_arrays: np.ndarray, target_config: np.ndarray, round_lim: int = 15):
    # Initialize the variables
    LBAP_success_flag = False
    complete_flag = False
    move_set = []
    matrix = copy.deepcopy(atom_arrays)
    round_count = 0

    while (complete_flag == False) and (round_count < round_lim):
        # print(f"Got here_{round_count}")
        N_independent_moves_path = []
        # 1. Generate the assignments
        prepared_assignments = generate_LBAP_assignments(matrix, target_config)

        # 2. Find out N independent paths
        for start, target in prepared_assignments:
            single_move_path = generate_path(matrix, start, target)
            if single_move_path == []:
                pass
            # Decompose the single_move_path into independent moves of several obstacle atoms
            else:
                N_independent_moves_path.append(single_move_path)

        # 3. Transform the N_independent_moves_path into a list of moves
        matrix, Hung_parallel_move_set = transform_paths_into_moves(matrix, N_independent_moves_path)
        move_set.extend(Hung_parallel_move_set)

        # effective_config = np.multiply(matrix, target_config)
        if Algorithm.get_success_flag(matrix, target_config, do_ejection=False, n_species = 1):
            complete_flag = True
            LBAP_success_flag = True
        round_count += 1

    return matrix, move_set, LBAP_success_flag

def generate_LBAP_assignments(matrix, target_config):
    
    #Define target positions for the center square in a matrix.
    current_positions, target_positions = define_current_and_target(matrix, target_config)

    # Generate the cost matrix using the current atom positions and the target positions
    cost_matrix = generate_cost_matrix(current_positions, target_positions)

    sq_cost = make_cost_matrix_square(cost_matrix)


    max_val = np.max(sq_cost)
    reverse_cost_mat = np.zeros_like(sq_cost)
    for i in range(len(reverse_cost_mat)):
        for j in range(len(reverse_cost_mat[0])):
            reverse_cost_mat[i,j] = max_val + 1 - sq_cost[i,j]

    sparsemat = csr_matrix(reverse_cost_mat)
    result_dict = bttl_threshold(sparsemat.indptr, 
                                    sparsemat.indices, 
                                    sparsemat.data,
                                    sparsemat.shape[0],
                                    sparsemat.shape[1])
    col_inds = result_dict['match']
    col_ind = []
    row_ind = []
    for c_ind in range(len(col_inds)):
        col = col_inds[c_ind]
        row = c_ind
        try:
            cost_matrix[row, col]
            col_ind.append(col)
            row_ind.append(row)
        except IndexError:
            pass
    # costs = []
    # for row_ind in range(len(sq_cost)):
    #     col_ind = col_inds[row_ind]
    #     costs.append(sq_cost[row_ind, col_ind])

    # Pair up row_ind and col_ind and sort by col_ind
    paired_indices = sorted(zip(row_ind, col_ind), key=lambda x: x[1])

    if paired_indices:
        # Unzip the sorted pairs if paired_indices is not empty
        sorted_row_ind, sorted_col_ind = zip(*paired_indices)
    else:
        # Assign default values if paired_indices is empty
        sorted_row_ind, sorted_col_ind = [], []

    prepared_assignments = [(current_positions[i], target_positions[j]) for i, j in zip(sorted_row_ind, sorted_col_ind)]

    return prepared_assignments

def Hungarian_algorithm_works(atom_arrays: np.ndarray, target_config: np.ndarray):
    move_set = []
    matrix = copy.deepcopy(atom_arrays)

    #Define target positions for the center square in a matrix.
    current_positions, target_positions = define_current_and_target(matrix, target_config)

    # Generate the cost matrix using the current atom positions and the target positions
    cost_matrix = generate_cost_matrix(current_positions, target_positions)

    # row_ind and col_ind are arrays of indices indicating the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Pair up row_ind and col_ind and sort by col_ind
    paired_indices = sorted(zip(row_ind, col_ind), key=lambda x: x[1])

    if paired_indices:
        # Unzip the sorted pairs if paired_indices is not empty
        sorted_row_ind, sorted_col_ind = zip(*paired_indices)
    else:
        # Assign default values if paired_indices is empty
        sorted_row_ind, sorted_col_ind = [], []
    prepared_assignments = [(current_positions[i], target_positions[j]) for i, j in zip(sorted_row_ind, sorted_col_ind)]

    for start, target in prepared_assignments:
        Hungarian_move = []
        Hungarian_move = move_atom_and_show_grid(matrix, start, target)
        move_set.extend(Hungarian_move)

    success_flag = Algorithm.get_success_flag(matrix.reshape(np.shape(target_config)), target_config, do_ejection=False, n_species = 1)

    return matrix, move_set, success_flag

def parallel_Hungarian_algorithm_works(atom_arrays: np.ndarray, target_config: np.ndarray, round_lim: int = 15):
    # Initialize the variables
    Hungarian_success_flag = False
    complete_flag = False
    move_set = []
    matrix = copy.deepcopy(atom_arrays)
    round_count = 0

    while (complete_flag == False) and (round_count < round_lim):
        N_independent_moves_path = []
        # 1. Generate the assignments
        prepared_assignments = generate_assignments(matrix, target_config)

        # 2. Find out N independent paths
        for start, target in prepared_assignments:
            single_move_path = generate_path(matrix, start, target)
            if single_move_path == []:
                pass
            # Decompose the single_move_path into independent moves of several obstacle atoms
            else:
                N_independent_moves_path.append(single_move_path)

        # 3. Transform the N_independent_moves_path into a list of moves
        matrix, Hung_parallel_move_set = transform_paths_into_moves(matrix, N_independent_moves_path)
        move_set.extend(Hung_parallel_move_set)

        # effective_config = np.multiply(matrix, target_config)
        if Algorithm.get_success_flag(matrix, target_config, do_ejection=False, n_species = 1):
            complete_flag = True
            Hungarian_success_flag = True
        round_count += 1

    return matrix, move_set, Hungarian_success_flag

def generate_assignments(matrix, target_config):

    #Define target positions for the center square in a matrix.
    current_positions, target_positions = define_current_and_target(matrix, target_config)

    # Generate the cost matrix using the current atom positions and the target positions
    cost_matrix = generate_cost_matrix(current_positions, target_positions)

    # row_ind and col_ind are arrays of indices indicating the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Pair up row_ind and col_ind and sort by col_ind
    paired_indices = sorted(zip(row_ind, col_ind), key=lambda x: x[1])

    if paired_indices:
        # Unzip the sorted pairs if paired_indices is not empty
        sorted_row_ind, sorted_col_ind = zip(*paired_indices)
    else:
        # Assign default values if paired_indices is empty
        sorted_row_ind, sorted_col_ind = [], []

    prepared_assignments = [(current_positions[i], target_positions[j]) for i, j in zip(sorted_row_ind, sorted_col_ind)]

    return prepared_assignments

def generate_path(arrays, start, end):
    grid = copy.deepcopy(arrays)
    #Initialize current position
    current_pos = start
    path = []
    
    while current_pos != end:
        path, current_pos = bfs_move_atom(grid, current_pos, end, path)
        
    path = flatten_tuple(path)[::-1]
    grid, path = generate_decomposed_move_set(grid, path)

    return path

def define_current_and_target(matrix, target_config):
    """Return (atoms, targets) as (row, col) coordinate tuples.

    The implementation scans rows first to align with the ndarray layout and
    includes every occupied site as a movable atom candidate. Likewise, every
    target cell with a value of 1 is included irrespective of its current
    occupancy; the assignment solver will naturally keep atoms in place when a
    target is already satisfied (zero-cost match).
    """
    curr_rows = len(matrix)
    curr_cols = len(matrix[0]) if curr_rows else 0
    current_positions = []
    target_positions = []
    for r in range(curr_rows):
        for c in range(curr_cols):
            if matrix[r][c] == 1:
                current_positions.append((r, c))
            if target_config[r][c] == 1:
                target_positions.append((r, c))
    return current_positions, target_positions

#Generate a cost matrix for the Hungarian Algorithm.
def generate_cost_matrix(current_positions, target_positions):
    num_atoms = len(current_positions)
    num_targets = len(target_positions)
    cost_matrix = np.zeros((num_atoms, num_targets))

    for i, current in enumerate(current_positions):
        for j, target in enumerate(target_positions):
            cost_matrix[i, j] = np.sqrt((current[0] - target[0])**2 + (current[1] - target[1])**2)
    return cost_matrix

##Move the atom from start to end according to Hungarian assignment
def move_atom_and_show_grid(grid, start, end):
    #Initialize current position
    current_pos = start
    path = []
    
    while current_pos != end:
        path, current_pos = bfs_move_atom(grid, current_pos, end, path)
        
    path = flatten_tuple(path)[::-1]
    grid, path = generate_decomposed_move_set(grid, path)

    return path

def generate_AOD_cmds(matrix, move_seq):
    row_num = len(matrix)
    col_num = len(matrix[0])
    horiz_AOD_cmds = np.zeros([col_num])
    vert_AOD_cmds = np.zeros([row_num])
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

def generate_decomposed_move_set(grid, path):
    decomposed_move_set = []

    #Iterate all path segments (((a1,b1), (a2,b2), (a3, b3), (a4,b4)), ((c1,d1), (c2,d2)))
    try:
        for segmant in path:
            from_row, from_col = segmant[0]
            for coordinate in segmant:
                to_row, to_col = coordinate
                # To exclude the frist move
                if from_row != to_row or from_col != to_col:
                    decomposed_move_set.append([Move(from_row, from_col, to_row, to_col)])
                    grid[from_row][from_col] = 0
                    grid[to_row][to_col] = 1
                    from_row, from_col = to_row, to_col
            # decomposed_move_set.append(segmant_moves)
    except IndexError:
        return grid, []
        
    return grid, decomposed_move_set

def regroup_parallel_moves(matrix, move_seqq):
    matrix_copy = copy.deepcopy(matrix)
    parallel_seq = []
    parallel_ind_set = set()

    # Iterate through all size of subset
    for move_ind, move in enumerate(move_seqq):
        if move_ind in parallel_ind_set or matrix_copy[move.from_row][move.from_col] == 0:
            continue
        parallel_moves = [move]
        parallel_ind_set.add(move_ind)

        for p_move_ind, p_move in enumerate(move_seqq):

            if p_move_ind in parallel_ind_set:
                continue

            horiz_AOD_cmds, vert_AOD_cmds, can_parallelize = generate_AOD_cmds(matrix_copy, parallel_moves + [p_move])

            if not can_parallelize:
                continue
            else:
                parallel_moves_test = parallel_moves + [p_move]
                if matrix_copy[p_move.from_row][p_move.from_col] == 0:
                    can_parallelize = False
                    continue
                sanit_check_matrix = copy.deepcopy(matrix_copy)
                total_atom_num_init = np.sum(sanit_check_matrix)
                matrix_copy, _ = move_atoms(matrix_copy, parallel_moves_test)
                total_atom_num_final = np.sum(matrix_copy)
                
                if total_atom_num_init == total_atom_num_final:
                    parallel_moves += [p_move]
                    parallel_ind_set.add(p_move_ind)
                    matrix_copy = copy.deepcopy(sanit_check_matrix)
                else:
                    matrix_copy = copy.deepcopy(sanit_check_matrix)
                    continue
            
        sanit_check_matrix = copy.deepcopy(matrix_copy)
        total_atom_num_init = np.sum(sanit_check_matrix)
        matrix_copy, _ = move_atoms(matrix_copy, parallel_moves)
        total_atom_num_final = np.sum(matrix_copy)
              
        if total_atom_num_init == total_atom_num_final:
            parallel_seq.append(parallel_moves)
        else:
            matrix_copy = copy.deepcopy(sanit_check_matrix)

    return parallel_seq

def transform_paths_into_moves(matrix, N_independent_moves_path):
    parallel_move_set = []

    # 1. Build up intersection information for these N independent paths
    intersection_matrix = np.zeros((len(N_independent_moves_path), len(N_independent_moves_path), 1))
    intersection_coordinates = [[[] for i in range(len(N_independent_moves_path))] for j in range(len(N_independent_moves_path))]
    intersection_set = {}

    for i in range(len(N_independent_moves_path)):
        for j in range(i, len(N_independent_moves_path)):
            if i != j:
                intersection_matrix[i][j], intersection_coordinates[i][j] = check_intersection(N_independent_moves_path[i], N_independent_moves_path[j])
                if len(intersection_coordinates[i][j]) > 0:
                    for intersection in intersection_coordinates[i][j]:
                        # Add a list of intersection coordinates
                        if intersection not in intersection_set:
                            intersection_set[intersection] = 0
                        # If the intersection is already in the set, increase the counter
                        else:
                            intersection_set[intersection] = intersection_set[intersection] + 1

    # 2. Implement the moves via N_independent_moves_path
    # 2.1 Reconstruct new move list regarding the parallel moves
    safety_cap = matrix.size * 4 if matrix.size else 0
    rounds_executed = 0
    while True:
        moves_in_scan = []
        destination_set = set()

        # 2.1.1 If there is no crossing path, implement one move for each path
        for path_in_moves in N_independent_moves_path:
            while path_in_moves:
                move = path_in_moves[0]
                crossing_path_flag = check_crossing_path(matrix, move[0], intersection_set, destination_set, path_in_moves)
                if crossing_path_flag:
                    break
                moves_in_scan.append(move[0])
                path_in_moves.pop(0)
                destination_set.add((move[0].to_row, move[0].to_col))

        if not moves_in_scan:
            break

        moves_in_scan = regroup_parallel_moves(matrix, moves_in_scan)
        if not moves_in_scan:
            break

        # 2.1.3 Implement the moves
        parallel_move_set.extend(moves_in_scan)
        for moves in moves_in_scan:
            matrix, _ = move_atoms(matrix, moves)
            for move in moves:
                if (move.from_row, move.from_col) in intersection_set:
                    if intersection_set[(move.from_row, move.from_col)] > 0:
                        intersection_set[(move.from_row, move.from_col)] -= 1
                    else:
                        del intersection_set[(move.from_row, move.from_col)]

                if (move.to_row, move.to_col) in intersection_set:
                    if intersection_set[(move.to_row, move.to_col)] > 0:
                        intersection_set[(move.to_row, move.to_col)] -= 1
                    else:
                        del intersection_set[(move.to_row, move.to_col)]

        rounds_executed += 1
        if safety_cap and rounds_executed > safety_cap:
            raise RuntimeError("Exceeded safety cap while transforming paths into moves")
            
    return matrix, parallel_move_set

##Find possible path between start and end position
def bfs_move_atom(grid, start, end, prev_path):
    queue = deque([(start[0], start[1], [(start[0], start[1])])]) #Use the queue to record current position and path
    visited = set() #Record the visited positions
    visited.add((start[0], start[1]))
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows else 0

    #Start finding the path
    while queue:
        current_row, current_col, path = queue.popleft() #Update current position

        #If we arrive end point, return the path
        if (current_row, current_col) == end:
            if len(prev_path) > 0:
                prev_path = prev_path, path
            else:
                prev_path = path

            return prev_path, end

        #Explore the next step (based on current position and end point)
        len_path = len(path) - 1
        dr = 1 if end[0] > path[len_path][0] else (-1 if end[0] < path[len_path][0] else 0) 
        dc = 1 if end[1] > path[len_path][1] else (-1 if end[1] < path[len_path][1] else 0)
        new_row, new_col = current_row + dr, current_col + dc

        #Check if there is an obstacle there (If no, start from this new point to find next step)
        if (
            0 <= new_row < n_rows
            and 0 <= new_col < n_cols
            and (new_row, new_col) not in visited
            and grid[new_row][new_col] == 0
        ):
            visited.add((new_row, new_col))
            queue.append((new_row, new_col, path + [(new_row, new_col)])) 

    #If there is an obstacle on the path, we decompose the path: start->obstacle->target
    #Define the obstacle position
    obstacle_row = path[len_path][0] + dr
    obstacle_col = path[len_path][1] + dc
    if not (0 <= obstacle_row < n_rows and 0 <= obstacle_col < n_cols):
        obstacle_row, obstacle_col = path[len_path][0], path[len_path][1]
    obstacle = (obstacle_row, obstacle_col)

    # Update the move in path until obstacle
    if len(prev_path) > 0:
        prev_path = prev_path, path + [obstacle]
    else:
        prev_path = path + [obstacle]

    #[Path between start and obstacle] + [Path between obstacle and end]
    return prev_path, obstacle

def check_intersection(path1, path2):
    # Extract destination coordinates from both lists
    destinations1 = {(move[0].to_row, move[0].to_col) for move in path1}
    destinations1.add((path1[0][0].from_row, path1[0][0].from_col))
    destinations2 = {(move[0].to_row, move[0].to_col) for move in path2}
    destinations2.add((path2[0][0].from_row, path2[0][0].from_col))

    # Find intersections
    intersections = destinations1 & destinations2

    # Return result
    if intersections:
        return True, list(intersections)
    else:
        return False, []

def check_crossing_path(matrix, move, intersection_set, delay_destination, path_in_moves):
    # Check if the destination is not in the intersection. If no intersection, implement the move
    if (move.to_row, move.to_col) not in intersection_set:
        return False
    # Check if the destination is end point of the path. If True, delay the move
    elif intersection_set[(move.to_row, move.to_col)] > 0 and (move.to_row, move.to_col) == (path_in_moves[-1][0].to_row, path_in_moves[-1][0].to_col):
        return True
    # If the destination is in the intersection set, but not passed yet this round, implement the move
    elif (move.to_row, move.to_col) not in delay_destination and matrix[move.to_row][move.to_col] == 0:
        delay_destination.add((move.to_row, move.to_col))
        return False
    else:
        return True

def flatten_tuple(nested_tuple):
    # This function will flatten a nested tuple of lists into a single tuple of lists
    result = []
    
    def recursive_flatten(element):
        if isinstance(element, tuple):
            # If the element is a tuple, apply recursion to each item
            for item in element:
                recursive_flatten(item)
        elif isinstance(element, list):
            # If the element is a list, append it to the result
            result.append(tuple(element))

    # Start the recursion with the entire nested tuple
    recursive_flatten(nested_tuple)
    
    # Convert the list of tuples into a single tuple
    return tuple(result)

    left_eject, bot_eject, top_eject, right_eject = [], [], [], []
    len_x = len(matrix[0])
    len_y = len(matrix)
    for x in range(len_x):
        for y in range(len_y):
            if matrix[x][y] == 1 and target_config[x][y] == 0:
                if x >= y and x < len_y - y:
                    left_eject.append((x, y))
                elif x >= y and x >= len_y - y:
                    bot_eject.append((x, y))
                elif x < y and x <= len_y - y:
                    top_eject.append((x, y))
                elif x <= y and x >= len_y - y:
                    right_eject.append((x, y))
    return left_eject, bot_eject, top_eject, right_eject

def generate_target_config(size: list, pattern: Configurations = 0, middle_size: list = [], probability: float = 0.5) -> np.ndarray:
    """ A function for generating common target configurations,
        such as checkerboard, zebra stripes, and middle fill.
    """
    array = np.zeros(size)

    if len(middle_size) == 0:
        middle_size = generate_middle_fifty(size[0])
        
    if pattern == Configurations.ZEBRA_HORIZONTAL: # every other row
        for i in range(0, size[0], 2):
            array[i,:] = 1
    elif pattern == 1: # every other col
        for i in range(0, size[1], 2):
            array[:,i] = 1
    elif pattern == 2: # checkerboard
        array = np.indices(size).sum(axis=0) % 2
    elif pattern == 3: # middle fill
        mrow = np.zeros([1,size[1]])
        mrow[0,int(size[1]/2-middle_size[1]/2):int(size[1]/2-middle_size[1]/2)+middle_size[1]] = 1
        for i in range(int(size[0]/2-middle_size[0]/2), int(size[0]/2-middle_size[0]/2)+middle_size[0]):
            array[i,:] = mrow
    elif pattern == 4:
        for i in range(middle_size[0]):
            array[:,i] = 1
    elif pattern == 5:
        array = random_loading(size,probability=probability)
    return array