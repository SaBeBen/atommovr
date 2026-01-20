import atommover.utils as movr
import copy
import numpy as np

def bcv2(array):
    if len(np.shape(array.matrix)) > 2 and np.shape(array.matrix)[2] == 2:
        raise ValueError(f'Atom array has shape {np.shape(array.matrix)}, which is not correct for single species. Did you meant to use a dual species algorithm?')
    success_flag = False
    arr1 = copy.deepcopy(array)
    start_row, start_col, end_row, end_col = get_target_locs(arr1)
    # 1. prebalance (making sure target rows/cols have enough atoms)
    master_move_list, col_compact, success_flag = prebalance(arr1.matrix, arr1.target)
    _,_ = arr1.evaluate_moves(master_move_list)
    if success_flag: # and col_compact == False:
        # _,_ = arr1.evaluate_moves(master_move_list)
        # 2. balance (distributing atoms between target rows according to needs)
        assignments = get_all_balance_assignments(start_row, end_row)
        # print(assignments)
        for assignment in assignments:
            try:
                bal_moves = balance_rows(arr1.matrix, arr1.target, assignment[0], assignment[1])
                if assignment[0] != assignment[1] and len(bal_moves) > 0:
                    _, _ = arr1.evaluate_moves(bal_moves)
                    master_move_list.extend(bal_moves)
            # print(f'finished assignment {assignment}')
            except ValueError:
                return arr1.matrix, master_move_list, False
        
    
        # print('finished balance')
        # 3. compact
        com_moves = compact(arr1)
        # print('finished compact')
        if len(com_moves) > 0:
            _, _ = arr1.evaluate_moves(com_moves)
            master_move_list.extend(com_moves)

    effective_config = np.multiply(arr1.matrix, arr1.target.reshape(np.shape(arr1.matrix)))
    if np.array_equal(effective_config, arr1.target.reshape(np.shape(arr1.matrix))):
        success_flag = True
    return arr1.matrix, master_move_list, success_flag

def special_case_algo_1d(init_config: np.ndarray, target_config: np.ndarray) -> 'list':
    arr_copy = movr.AtomArray(np.shape(init_config)[:2])
    arr_copy.target = copy.deepcopy(target_config)
    arr_copy.matrix = copy.deepcopy(init_config)
    
    # first, find the column indices of the target sites
    # and those of the sites with atoms
    target_indices = np.where(arr_copy.target == 1)[1]
    atom_indices = np.where(arr_copy.matrix == 1)[1]
    
    if len(target_indices) != len(atom_indices):
        raise Exception(f"Number of atoms ({len(atom_indices)}) does not equal number of target sites ({len(target_indices)}).")

    # second, we can pair the atoms and make a list
    pairs = []
    for ind, target_index in enumerate(target_indices):
        atom_index = atom_indices[ind]
        pair = (target_index, atom_index)
        pairs.append(pair)
    # lastly, we can move atoms towards their target positions
    target_prepared = np.array_equal(arr_copy.target, arr_copy.matrix)
    move_set = []
    while not target_prepared:
        move_list = []
        for i, pair in enumerate(pairs):
            target_index, atom_index = pair
            if target_index != atom_index:
                new_atom_index = int(atom_index+np.sign(target_index-atom_index))
                move = movr.Move(0, atom_index, 0, new_atom_index)
                move_list.append(move)
                pairs[i] = (target_index, new_atom_index)
        if move_list != []:
            _, _ = arr_copy.evaluate_moves([move_list])
            move_set.append(move_list)
        else:
            break
    
    return move_set, atom_indices



# utility function that calculates the longest move distance between target sites and atom sites
def find_largest_dist_to_move(target_inds, atom_inds):
    if len(target_inds) > len(atom_inds):
        return np.inf
    max_dist = 0
    for ind, target_loc in enumerate(target_inds):
        atom_loc = atom_inds[ind]
        distance = np.abs(target_loc-atom_loc)
        if distance > max_dist:
            max_dist = distance
    return max_dist

def middle_fill_algo_1d(init_config: np.ndarray, target_config: np.ndarray) -> 'list':
    arr_copy = movr.AtomArray(np.shape(init_config)[:2])
    arr_copy.target = copy.deepcopy(target_config)
    arr_copy.matrix = copy.deepcopy(init_config)
    # first, find the column indices of the target sites
    # and those of the sites with atoms
    target_indices = np.where(arr_copy.target == 1)[1]
    atom_indices = np.where(arr_copy.matrix == 1)[1]
    n_targets = len(target_indices)

    # second, find the optimal pairing of atoms if 
    if n_targets == len(atom_indices):
        return special_case_algo_1d(init_config, target_config)
    elif n_targets > len(atom_indices):
        return [], []
    
    # third, find the centermost set of atoms
    avg_targ_pos = int(np.ceil(np.mean(target_indices)))
    count = 0
    sufficient_atoms = False
    while not sufficient_atoms:
        center_region = arr_copy.matrix[0,avg_targ_pos-count:avg_targ_pos+count+1]
        n_atoms_in_center_region = np.sum(center_region)
        sufficient_atoms = n_targets <= n_atoms_in_center_region
        if not sufficient_atoms:
            count+=1
        else:
            break
    first_atom_loc = np.where(center_region == 1)[0][0]+avg_targ_pos-count

    # fourth, look to the adjacent sets and see if these are better
    look_right = True
    right_count = 0
    while look_right:
        list_ind = np.where(atom_indices==first_atom_loc)[0][0]
        current_r_atom_set = atom_indices[list_ind+right_count:list_ind+right_count+n_targets]
        right_atom_set = atom_indices[list_ind+right_count+1:list_ind+right_count+n_targets+1]
        dist_r_current = find_largest_dist_to_move(target_indices, current_r_atom_set)
        dist_right = find_largest_dist_to_move(target_indices, right_atom_set)
        if dist_right > dist_r_current:
            look_right = False
        else:
            right_count += 1
    
    look_left = True
    left_count = 0
    while look_left:
        list_ind = np.where(atom_indices==first_atom_loc)[0][0]
        current_l_atom_set = atom_indices[list_ind-left_count:list_ind-left_count+n_targets]
        left_atom_set = atom_indices[list_ind-left_count-1:list_ind-left_count+n_targets-1]
        dist_l_current = find_largest_dist_to_move(target_indices, current_l_atom_set)
        dist_left = find_largest_dist_to_move(target_indices, left_atom_set)
        if dist_left > dist_l_current:
            look_left = False
        else:
            left_count += 1
    
    if dist_l_current < dist_r_current:
        best_atom_set = current_l_atom_set
    else:
        best_atom_set = current_r_atom_set
    
    # fifth, find the best set and assign pairs
    pairs = []
    for ind, target_index in enumerate(target_indices):
        atom_index = best_atom_set[ind]
        pair = (target_index, atom_index)
        pairs.append(pair)

    # lastly, we can move atoms towards their target positions
    target_prepared = np.array_equal(arr_copy.target, arr_copy.matrix)
    move_set = []
    while not target_prepared:
        move_list = []
        for i, pair in enumerate(pairs):
            target_index, atom_index = pair
            if target_index != atom_index:
                new_atom_index = int(atom_index+np.sign(target_index-atom_index))
                move = movr.Move(0, atom_index, 0, new_atom_index)
                move_list.append(move)
                pairs[i] = (target_index, new_atom_index)
        if move_list != []:
            _, _ = arr_copy.evaluate_moves([move_list])
            move_set.append(move_list)
        else:
            target_prepared = True
    
    return move_set, best_atom_set


# Balance and Compact

def balance_rows(init_config: np.ndarray, target_config: np.ndarray, i: int, j: int):
    if i == j:
        return []
    l = j-i+1
    m = i+(l//2)
    n_req_top = np.sum(target_config[i:m,:])
    n_atoms_top = np.sum(init_config[i:m,:])
    n_req_bot = np.sum(target_config[m:j+1,:])
    n_atoms_bot = np.sum(init_config[m:j+1,:])
    diff_top = n_atoms_top-n_req_top
    diff_bot = n_atoms_bot-n_req_bot
    if (diff_top+diff_bot) < 0:
        raise ValueError(f"Insufficient number of atoms: deficit in rows {i}-{m-1} is {diff_top} and deficit in rows {m}-{j} is {diff_bot}.")

    current_state = copy.deepcopy(init_config)
    moves = []
    n_to_move = int(np.floor(np.abs(diff_bot-diff_top)/2))
    # print(f'Top: {diff_top}; Bot: {diff_bot}; n to move: {n_to_move}; top region: {i}-{m-1}')
    if diff_bot == diff_top or (diff_bot > 0 and diff_top > 0):
        pass
    elif diff_top < diff_bot:
        current_state, round_moves = move_across_rows(current_state, n_to_move, i,j,m,-1)
        if len(round_moves) > 0:
            moves.extend(round_moves)
    elif diff_bot < diff_top:
        current_state, round_moves = move_across_rows(current_state, n_to_move, i,j,m, 1)
        if len(round_moves) > 0:
            moves.extend(round_moves)
    return moves

def _prebalance_above(current_state,start_row, end_row, n_targets, round_moves, direction):
    n_movable_above = 0
    row_offset = 0
    if direction == -1:
        boundary_row = start_row
    else:
        boundary_row = end_row
    while n_movable_above == 0:
        if np.sum(current_state) < n_targets:
            raise Exception('Insufficient atoms.')
        try:
            # move_set = []
            for off in range(row_offset+1)[::-1]: # TODO: figure out if this should be the -1 thing or if it makes more sense to do something else
                above_moves, n_movable = get_all_moves_btwn_rows(current_state,boundary_row+(1+off)*direction, boundary_row+off*direction)
                if n_movable != 0 and np.sum(current_state[start_row: end_row + 1, :]) < n_targets: # check if there are atoms that can be moved, and if so move them
                    current_state, _ = movr.move_atoms(current_state,above_moves)
                    round_moves.append(above_moves) # NEW
                    # print(0,direction, above_moves)
                else: # if no atoms can be moved, figure out why
                    n_in_from_row = np.sum(current_state[boundary_row+(1+off)*direction,:])
                    if n_in_from_row > 0: # if there are no spots for new atoms to come, make space by pushing atoms farther inside
                        rows_in = 0
                        stuck_row = boundary_row+off*direction
                        while n_movable == 0:
                            # stuck_row = boundary_row+off*direction
                            for r_in in range(-1,rows_in)[::-1]:

                                space_moves, n_sp_movable = get_all_moves_btwn_rows(current_state,stuck_row-r_in*direction, stuck_row-(1+r_in)*direction)
                                # print(f'Can move {n_sp_movable} from {stuck_row-(r_in)*direction} to {stuck_row-(1+r_in)*direction}')
                                if n_sp_movable != 0 and np.sum(current_state[start_row: end_row + 1, :]) < n_targets: # check if there are atoms that can be moved, and if so move them
                                    # print(f'Trying to move atoms from {stuck_row-(1+r_in)*direction} to {stuck_row-(2+r_in)*direction}')
                                    current_state, _ = movr.move_atoms(current_state,space_moves)
                                    round_moves.append(space_moves)
                                    # print(1,direction, space_moves)
                                    above_moves, n_movable = get_all_moves_btwn_rows(current_state,stuck_row, stuck_row-direction)
                                    # NEW
                                    if np.sum(current_state[start_row: end_row + 1, :]) < n_targets and n_movable != 0:
                                        current_state, _ = movr.move_atoms(current_state,above_moves)
                                        round_moves.append(above_moves)
                                        # print(2,direction, above_moves)
                            rows_in += 1
                        # NEW
                #         if np.sum(current_state[start_row: end_row + 1, :]) < n_targets:
                #             current_state, _ = movr.move_atoms(current_state,above_moves)
                #             move_set.append(above_moves)
                # if len(above_moves) > 0:
                #     move_set.append(above_moves)
            # array = movr.AtomArray([12,12])
            # array.matrix = current_state.reshape([12,12,1])
            # array.image()
            # print(f'Number of atoms is {np.sum(current_state)}')
            if n_movable > 0:
                n_movable_above = n_movable
            row_offset += 1
            # if len(move_set) > 0:
            #     round_moves.extend(move_set)
            # if direction == 1:
            #     print(round_moves)
        except IndexError:
            row_offset += 1
            break
        # print(move_set)
    return current_state, round_moves


def prebalance(init_config, target_config):
    success_flag = False

    # Find the relevant rows and columns of the target configuration
    row_max = 0
    row_min = len(target_config)-1
    col_max = 0
    col_min = len(target_config[0])-1
    for row in range(len(target_config)):
        for col in range(len(target_config[0])):
            if target_config[row,col] == 1:
                if row > row_max:
                    row_max = row
                if row < row_min: 
                    row_min = row
                if col > col_max:
                    col_max = col 
                if col < col_min:
                    col_min = col 
    start_row, start_col, end_row, end_col = row_min, col_min, row_max, col_max
    
    n_atoms_row_region = np.sum(init_config[start_row: end_row + 1, :])
    n_atoms_col_region = np.sum(init_config[:, start_col: end_col + 1])
    n_atoms_global = np.sum(init_config)
    n_targets = np.sum(target_config[start_row:end_row+1, :])

    if n_atoms_global < n_targets:
        return [], None, success_flag

    # finding how many atoms we need to fill and generating moves
    n_to_fill_row = n_targets-n_atoms_row_region
    n_to_fill_col = n_targets-n_atoms_col_region
    
    moves = []
    if n_to_fill_row <= 0:
        col_compact = False
        success_flag = True
        return moves, col_compact, success_flag
    # elif n_to_fill_col <= 0:
    #     col_compact = True
    #     success_flag = True
    #     return moves, col_compact, success_flag
    # elif n_to_fill_col >= n_to_fill_row:
    else:
        col_compact = False

        current_state = copy.deepcopy(init_config)
        while np.sum(current_state[start_row: end_row + 1, :]) < n_targets and np.sum(current_state)>=n_targets:
            round_moves = []
            current_state, round_moves = _prebalance_above(current_state, start_row, end_row, n_targets, round_moves, -1)
            # MOVING FROM ABOVE
            # n_movable_above = 0
            # row_offset = 0
            # while n_movable_above == 0:
            #     try:
            #         move_set = []
            #         for off in range(row_offset+1)[::-1]:
            #             above_moves, n_movable = get_all_moves_btwn_rows(current_state,start_row-1-off, start_row-off)
            #             if n_movable != 0 and np.sum(current_state[start_row: end_row + 1, :]) < n_targets: # check if there are atoms that can be moved, and if so move them
            #                 current_state, _ = movr.move_atoms(current_state,above_moves)
            #             else: # if no atoms can be moved, figure out why
            #                 n_in_from_col = np.sum(current_state[start_row-1-off,:])
            #                 if n_in_from_col > 0: # if there are no spots for new atoms to come, make space by pushing atoms farther inside
            #                     rows_in = 0
            #                     while n_movable == 0:
            #                         stuck_row = start_row-off
            #                         for r_in in range(rows_in+1)[::-1]:
            #                             space_moves, n_sp_movable = get_all_moves_btwn_rows(current_state,stuck_row+1+r_in, stuck_row+2+r_in)
            #                             if n_sp_movable != 0 and np.sum(current_state[start_row: end_row + 1, :]) < n_targets: # check if there are atoms that can be moved, and if so move them
            #                                 current_state, _ = movr.move_atoms(current_state,space_moves)
            #                                 move_set.append(space_moves)
            #                                 above_moves, n_movable = get_all_moves_btwn_rows(current_state,stuck_row, stuck_row+1)
            #                         rows_in += 1
            #                     if np.sum(current_state[start_row: end_row + 1, :]) < n_targets:
            #                         current_state, _ = movr.move_atoms(current_state,above_moves)
            #             if len(above_moves) > 0:
            #                 move_set.append(above_moves)
            #         if n_movable > 0:
            #             n_movable_above = n_movable
            #         row_offset += 1
            #         if len(move_set) > 0:
            #             round_moves.extend(move_set)
            #     except IndexError:
            #         row_offset += 1
            #         break
        
            # MOVING FROM BELOW
            if np.sum(current_state[start_row: end_row + 1, :]) < n_targets:
                current_state, round_moves = _prebalance_above(current_state, start_row, end_row, n_targets, round_moves, 1)
                # get atoms from below
                # n_movable_below = 0
                # row_offset = 0
                # while n_movable_below == 0:
                #     try:
                #         move_set = []
                #         for off in range(row_offset+1)[::-1]:
                #             below_moves, n_movable = get_all_moves_btwn_rows(current_state,end_row+1+off, end_row+off)
                #             if n_movable != 0 and np.sum(current_state[start_row: end_row + 1, :]) < n_targets: # check if there are atoms that can be moved, and if so move them
                #                 current_state, _ = movr.move_atoms(current_state,below_moves)
                #             else: # if no atoms can be moved, figure out why
                #                 n_in_from_col = np.sum(current_state[end_row+1+off,:])
                #                 if n_in_from_col > 0: # if there are no spots for new atoms to come, make space by pushing atoms farther inside
                #                     rows_in = 0
                #                     while n_movable == 0:
                #                         stuck_row = end_row+off
                #                         for r_in in range(rows_in+1)[::-1]:
                #                             space_moves, n_sp_movable = get_all_moves_btwn_rows(current_state,stuck_row-1-r_in, stuck_row-2-r_in)
                #                             if n_sp_movable != 0 and np.sum(current_state[start_row: end_row + 1, :]) < n_targets: # check if there are atoms that can be moved, and if so move them
                #                                 current_state, _ = movr.move_atoms(current_state,space_moves)
                #                                 move_set.append(space_moves)
                #                                 below_moves, n_movable = get_all_moves_btwn_rows(current_state,stuck_row, stuck_row-1)
                #                         rows_in += 1
                #                     if np.sum(current_state[start_row: end_row + 1, :]) < n_targets:
                #                         current_state, _ = movr.move_atoms(current_state,below_moves)
                #             if len(below_moves) > 0:
                #                 move_set.append(below_moves)
                #         if n_movable > 0:
                #             n_movable_below = n_movable
                #         row_offset += 1
                #         if len(move_set) > 0:
                #             round_moves.extend(move_set)
                #     except IndexError:
                #         row_offset += 1
                #         break
            moves.extend(round_moves)
            # arr = movr.AtomArray([5,5])
            # arr.matrix = current_state
            # _m,_ = arr.evaluate_moves(move_set)
            # arr.image(move_list=move_set[0])

        if np.sum(current_state[start_row: end_row + 1, :]) >= n_targets:
            success_flag = True
        return moves, col_compact, success_flag
                
    # else:
    #     col_compact = True

    #     current_state = copy.deepcopy(init_config)
    #     while np.sum(current_state[:, start_col:end_col+1]) < n_targets:
    #         round_moves = []
           
    #         # MOVING FROM ABOVE
    #         n_movable_above = 0
    #         col_offset = 0
    #         while n_movable_above == 0:
    #             try:
    #                 move_set = []
    #                 for off in range(col_offset+1)[::-1]:
    #                     above_moves, n_movable = get_all_moves_btwn_cols(current_state,start_col-1-off, start_col-off)
    #                     if n_movable != 0 and np.sum(current_state[:,start_col: end_col + 1]) < n_targets: # check if there are atoms that can be moved, and if so move them
    #                         current_state, _ = movr.move_atoms(current_state,above_moves)
    #                     else: # if no atoms can be moved, figure out why
    #                         n_in_from_col = np.sum(current_state[:,start_col-1-off])
    #                         if n_in_from_col > 0: # if there are no spots for new atoms to come, make space by pushing atoms farther inside
    #                             cols_in = 0
    #                             while n_movable == 0:
    #                                 stuck_col = start_col-off
    #                                 for c_in in range(cols_in+1)[::-1]:
    #                                     space_moves, n_sp_movable = get_all_moves_btwn_cols(current_state,stuck_col+1+c_in, stuck_col+2+c_in)
    #                                     if n_sp_movable != 0 and np.sum(current_state[:,start_col: end_col + 1]) < n_targets: # check if there are atoms that can be moved, and if so move them
    #                                         current_state, _ = movr.move_atoms(current_state,space_moves)
    #                                         move_set.append(space_moves)
    #                                         above_moves, n_movable = get_all_moves_btwn_cols(current_state,stuck_col, stuck_col+1)
    #                                 cols_in += 1
    #                             if np.sum(current_state[:, start_col: end_col + 1]) < n_targets:
    #                                 current_state, _ = movr.move_atoms(current_state,above_moves)
    #                     if len(above_moves) > 0:
    #                         move_set.append(above_moves)
    #                 if n_movable > 0:
    #                     n_movable_above = n_movable
    #                 col_offset += 1
    #                 if len(move_set) > 0:
    #                     round_moves.extend(move_set)
    #             except IndexError:
    #                 col_offset += 1
    #                 break
        
    #         # MOVING FROM BELOW
    #         if np.sum(current_state[:, start_row: end_row + 1]) < n_targets:

    #             # get atoms from below
    #             n_movable_below = 0
    #             col_offset = 0
    #             while n_movable_below == 0:
    #                 try:
    #                     move_set = []
    #                     for off in range(col_offset+1)[::-1]:
    #                         below_moves, n_movable = get_all_moves_btwn_cols(current_state,end_col+1+off, end_col+off)
    #                         if n_movable != 0 and np.sum(current_state[:, start_col: end_col + 1]) < n_targets: # check if there are atoms that can be moved, and if so move them
    #                             current_state, _ = movr.move_atoms(current_state,below_moves)
    #                         else: # if no atoms can be moved, figure out why
    #                             n_in_from_col = np.sum(current_state[end_col+1+off,:])
    #                             if n_in_from_col > 0: # if there are no spots for new atoms to come, make space by pushing atoms farther inside
    #                                 cols_in = 0
    #                                 while n_movable == 0:
    #                                     stuck_col = end_col+off
    #                                     for c_in in range(cols_in+1)[::-1]:
    #                                         space_moves, n_sp_movable = get_all_moves_btwn_cols(current_state,stuck_col-1-c_in, stuck_col-2-c_in)
    #                                         if n_sp_movable != 0 and np.sum(current_state[:,start_col: end_col + 1]) < n_targets: # check if there are atoms that can be moved, and if so move them
    #                                             current_state, _ = movr.move_atoms(current_state,space_moves)
    #                                             move_set.append(space_moves)
    #                                             below_moves, n_movable = get_all_moves_btwn_cols(current_state,stuck_col, stuck_col-1)
    #                                     cols_in += 1
    #                                 if np.sum(current_state[:, start_col: end_col + 1]) < n_targets:
    #                                     current_state, _ = movr.move_atoms(current_state,below_moves)
    #                         if len(below_moves) > 0:
    #                             move_set.append(below_moves)
    #                     if n_movable > 0:
    #                         n_movable_below = n_movable
    #                     col_offset += 1
    #                     if len(move_set) > 0:
    #                         round_moves.extend(move_set)
    #                 except IndexError:
    #                     col_offset += 1
    #                     break
    #         moves.extend(round_moves)
    #     if np.sum(current_state[:, start_col: end_col + 1]) >= n_targets:
    #         success_flag = True
    #     return moves, col_compact, success_flag


def get_all_moves_btwn_rows(init_config, from_row_ind, to_row_ind):
    if from_row_ind < 0 or to_row_ind < 0:
        raise IndexError
    from_row = init_config[from_row_ind,:]
    to_row = init_config[to_row_ind,:]

    available_source = np.where(from_row == 1)[0]
    available_spots = np.where(to_row == 0)[0]

    moves = []
    for atom_col in available_source:
        move = None
        if atom_col-1 in available_spots:
            move = movr.Move(from_row_ind, atom_col, to_row_ind, atom_col-1)
            available_spots = available_spots[~np.isin(available_spots, atom_col-1)]
        elif atom_col in available_spots:
            move = movr.Move(from_row_ind, atom_col, to_row_ind, atom_col)
            available_spots = available_spots[~np.isin(available_spots, atom_col)]
        elif atom_col+1 in available_spots:
            move = movr.Move(from_row_ind, atom_col, to_row_ind, atom_col+1)
            available_spots = available_spots[~np.isin(available_spots, atom_col+1)]
        if move is not None:    
            moves.append(move)
    n_atoms_movable = len(moves)
    return moves, n_atoms_movable

def get_all_moves_btwn_cols(init_config, from_col_ind, to_col_ind):
    from_col = init_config[:,from_col_ind]
    to_col = init_config[:,to_col_ind,:]

    available_source = np.where(from_col == 1)[0]
    available_spots = np.where(to_col == 0)[0]

    moves = []
    for atom_row in available_source:
        move = None
        if atom_row-1 in available_spots:
            move = movr.Move(atom_row, from_col_ind, atom_row-1, to_col_ind)
            available_spots = available_spots[~np.isin(available_spots, atom_row-1)]
        elif atom_row in available_spots:
            move = movr.Move(atom_row, from_col_ind, atom_row, to_col_ind)
            available_spots = available_spots[~np.isin(available_spots, atom_row)]
        elif atom_row+1 in available_spots:
            move = movr.Move(atom_row, from_col_ind, atom_row+1, to_col_ind)
            available_spots = available_spots[~np.isin(available_spots, atom_row+1)]
        if move is not None:    
            moves.append(move)
    n_atoms_movable = len(moves)
    return moves, n_atoms_movable

def move_across_rows(current_state: np.ndarray, n_to_move: int, i: int, j: int, m: int, dir = -1):
    """
    Moves `n_to_move` atoms from row m to m-1 if dir = -1 or vice versa. If there aren't
    enough atoms, can access additional rows (subject to the constraint
    i < row and row < j).
    """

    round_moves = [] # master list of all moves taken in this procedure
    n_left_to_move = n_to_move
    
    ## specifying rows to move across and ROIs
    if dir == 1:
        start_row = m-1
        end_row = m
        low_ind_roi = m
        high_ind_roi = j+1
        low_ind_source = i
        high_ind_source = m
    elif dir == -1:
        start_row = m
        end_row = m-1
        low_ind_roi = i
        high_ind_roi = m
        low_ind_source = m
        high_ind_source = j+1
    else:
        raise ValueError('Parameter "dir" must be -1 or 1.')
    
    ## sanity check to make sure we have sufficient atoms
    n_atoms_in_source = np.sum(current_state[low_ind_source:high_ind_source])
    n_atoms_in_roi = np.sum(current_state[low_ind_roi:high_ind_roi])
    if n_atoms_in_source < n_to_move:
        raise Exception(f'Insufficient atoms. Only {n_atoms_in_source} in the source region.')
    
    ## continue looping until we move sufficient atoms.
    try_count = 0
    while n_left_to_move != 0 and try_count < 1000:
        try_count += 1
        # print(f'{n_left_to_move} atoms left to move from {low_ind_source}-{high_ind_source-1} to {low_ind_roi}-{high_ind_roi-1}')
        # if n_left_to_move == 4:
        #     arr = movr.AtomArray([30,30])
        #     arr.matrix = current_state
        #     arr.image()
        n_movable_dir = 0
        row_offset = 0
        last_moves = [0] # placeholder
        ## we loop until we are able to move atoms
        try_count2 = 0
        while n_movable_dir == 0 and try_count2 < 1000:
            try_count2 += 1
            try:
                move_set = []
                for off in range(row_offset+1)[::-1]:
                    across_move = 1
                    from_row = start_row+(off*dir)
                    to_row = end_row+(off*dir)
                    if i>from_row or i>to_row or j<from_row or j<to_row:
                        # print("outside of bounds", off, from_row, to_row)
                        raise IndexError
                    above_moves, n_movable = get_all_moves_btwn_rows(current_state,from_row, to_row)
                    # print(f'{n_movable} atoms can be moved from {from_row} to {to_row}.')

                    if n_movable != 0 and n_left_to_move != 0: # check if there are atoms that can be moved, and if so move them
                        if off == 0:
                            moves_to_run = above_moves[:n_left_to_move]
                        else:
                            moves_to_run = above_moves
                        current_state, _ = movr.move_atoms(current_state,moves_to_run)
                        n_left_to_move -= len(moves_to_run)
                        move_set.append(moves_to_run) 
                    else: # if atoms CANNOT be moved
                        n_in_from_row = np.sum(current_state[from_row,:])
                        # print(f'Stuck: n_atoms in from row are {n_in_from_row}')
                        ## Scenario 1: there are atoms to move, but no place to put them in the new row, so we have to clear room in ROI
                        if n_in_from_row > 0 and len(last_moves) > 0:
                            clear_space_in_roi_moves = []
                            rows_into_ROI = 0
                            while n_movable == 0:
                                stuck_row = start_row+dir*off
                                for r_in in range(rows_into_ROI+1)[::-1]: # NKH change 05-09
                                    from_row = stuck_row+(1+r_in)*dir
                                    to_row = stuck_row+(2+r_in)*dir
                                    if i>from_row or i>to_row or j<from_row or j<to_row:
                                        # print("S1: outside of bounds", off, from_row, to_row)
                                        raise IndexError
                                    space_moves, n_sp_movable = get_all_moves_btwn_rows(current_state,from_row, to_row)
                                    if n_sp_movable != 0 and n_left_to_move != 0: # check if there are atoms that can be moved, and if so move them
                                        current_state, _ = movr.move_atoms(current_state,space_moves)
                                        clear_space_in_roi_moves.append(space_moves)
                                        n_movable = n_sp_movable
                                        # print('clear in roi', space_moves)
                                rows_into_ROI += 1
                            if len(clear_space_in_roi_moves) > 0:
                                move_set.extend(clear_space_in_roi_moves)
                            # print('clearing room in roi', clear_space_in_roi_moves)
                        ## Scenario 2: there are no atoms to move, so we have to take atoms from farther inside the source region
                        elif n_in_from_row == 0 or len(last_moves) == 0:
                            pull_atoms_from_reservoir_moves = []
                            rows_into_source = 0
                            while n_movable == 0:
                                stuck_row = start_row+dir*off
                                for r_in in range(-1,rows_into_source)[::-1]:
                                    from_row = stuck_row-(2+r_in)*dir
                                    to_row = stuck_row-(1+r_in)*dir
                                    if i>from_row or i>to_row or j<from_row or j<to_row:
                                        # print(f"S2: outside of bounds. Attempted from row is {from_row}, but bound is {to_row}]", off)
                                        raise IndexError
                                    space_moves, n_sp_movable = get_all_moves_btwn_rows(current_state,from_row, to_row)
                                    if n_sp_movable != 0 and n_left_to_move != 0: # check if there are atoms that can be moved, and if so move them
                                        current_state, _ = movr.move_atoms(current_state,space_moves)
                                        pull_atoms_from_reservoir_moves.append(space_moves)
                                        # print('take from source', space_moves)
                                        n_movable = n_sp_movable
                                rows_into_source += 1
                            if len(pull_atoms_from_reservoir_moves) > 0:
                                move_set.extend(pull_atoms_from_reservoir_moves)
                            # print('getting more source atoms', pull_atoms_from_reservoir_moves)
                    # if len(above_moves) > 0:
                    #     if off == 0 and n_left_to_move != 0 and across_move:
                    #         moves_to_run = above_moves[:n_left_to_move]
                    #         move_set.append(moves_to_run)
                    #         n_left_to_move -= len(moves_to_run)
                    #     else:
                    #         move_set.append(above_moves)
                
                
                # if len(round_moves) > 0:
                #     # round_moves.extend(move_set)
                #     print(round_moves)
                # DEBUG
                # arr = movr.AtomArray([13,13])
                # arr.matrix = current_state
                # # _m,_ = arr.evaluate_moves(round_moves)
                # arr.image(move_list=round_moves[0])
                # END DEBUG
                if len(move_set) > 0:
                    round_moves.extend(move_set)
                last_moves = move_set
        
                if n_movable > 0:
                    n_movable_dir = n_movable
                    break
                row_offset += 1
            except IndexError:
                # print('index errored') # DEBUG
                row_offset += 1
                break
        
    return current_state, round_moves


def get_all_balance_assignments(start, end):
    assignments = []
    i = start
    j = end
    new_assignments = [(i,j)]
    n_a = len(new_assignments)
    while n_a > 0:
        assignment_list = []
        for assignment in new_assignments:
            i = assignment[0]
            j = assignment[1]
            next_layer = get_next_balance_assignment(i,j)
            assignment_list.extend(next_layer)
        assignments.extend(new_assignments)
        if len(assignment_list) > 0:
            new_assignments = assignment_list
        else:
            break
    return assignments

    
def get_next_balance_assignment(i,j):
    l = j-i+1
    m = i+(l//2)
    next_list = []
    if i != j and i<j:
        next_list.append((i,m-1))
        next_list.append((m,j))
    return next_list


def get_target_locs(array):# Find the relevant rows and columns of the target configuration
    """
    Finds the boundaries of array.target (i.e. the biggest square which contains all atoms in the target config).DS_Store

    ## Parameters:
        array : AtomArray()
    
    ## Returns:
        start_row : int 
            the index of the first row where there are atoms in the target config
        start_col : int
            the index of the first column where there are atoms in the target config
        end_row : int 
            the index of the last row where there are atoms in the target config
        end_col : int
            the index of the last column where there are atoms in the target config
    """
    n_rows = len(array.target)
    n_cols = len(array.target[0])
    row_max = int(0)
    row_min = int(n_rows-1)
    col_max = int(0)
    col_min = int(n_cols-1)
    for row in range(n_rows):
        for col in range(n_cols):
            if array.target[row,col,0] == 1:
                if row > row_max:
                    row_max = row
                if row < row_min: 
                    row_min = row
                if col > col_max:
                    col_max = col 
                if col < col_min:
                    col_min = col 
    start_row, start_col, end_row, end_col = row_min, col_min, row_max, col_max
    return start_row, start_col, end_row, end_col

def compact(array):
    arr1 = copy.deepcopy(array)

    start_row, start_col, end_row, end_col = get_target_locs(arr1)
    n_rows = len(arr1.target)
    n_cols = len(arr1.target[0])

    global_move_set = []
    while True:  
        """
        1. Loop through columns in target config and count how many atoms there are.
        2. Select the column with the least number of atoms.
        3. For all unoccupied rows in this column:
            i. Count the number of atoms that want to move from their current position towards the selected column.
            ii. Count the number of atoms that do NOT want to move.
            iii. If the number of atoms that want to move is greater than the number of atoms that do NOT want to move, add the row to row_list
        4. Condense all atoms in rows in rows_list inwards."""

        # counting how many vacancies are in columns
        col_ns = []
        for col in range(start_col, end_col+1):
            n_in_col = np.sum(arr1.matrix[start_row:end_row+1,col, 0])
            col_ns.append(n_in_col)
        min_n_col = min(col_ns)
        min_col_ind = np.where(col_ns == min_n_col)[0][0]+start_col

        r_vote_tally = np.zeros(len(range(start_row, end_row+1)))
        l_vote_tally = np.zeros(len(range(start_row, end_row+1)))
        move_arr = np.zeros([end_row-start_row+1, 2], dtype='object')
        for i, row in enumerate(range(start_row, end_row+1)):
            atom_in_row = arr1.matrix[row, min_col_ind,0]
            if atom_in_row != 0:
                r_vote_tally[i] = -np.e # code for automatic no vote
                l_vote_tally[i] = -np.e
            move_set, best_atom_set = middle_fill_algo_1d(arr1.matrix[row,:,:].reshape(1, len(arr1.target[0]),1), arr1.target[row,:,:].reshape(1, len(arr1.target[0]),1))
            move_arr[i, 1] = move_set
            move_arr[i, 0] = best_atom_set
        for col in range(n_cols):
            move_dir = np.sign(min_col_ind-col)
            if move_dir == -1:
                for i, row in enumerate(range(start_row, end_row+1)):
                    cond1 = r_vote_tally[i] != -np.e
                    cond2 = int(arr1.matrix[row, col,0]) == 1
                    cond3 = col in move_arr[i,0]
                    if cond1 and cond2 and cond3:
                        vote = int(movr.Move(0, col, 0, col-1) in move_arr[i,1][0])
                        r_vote_tally[i] += -1 + 2*vote
            elif move_dir == 1:
                for i, row in enumerate(range(start_row, end_row+1)):
                    cond1 = l_vote_tally[i] != -np.e
                    cond2 = int(arr1.matrix[row, col,0]) == 1
                    cond3 = col in move_arr[i,0]
                    if cond1 and cond2 and cond3:
                        vote = int(movr.Move(0, col, 0, col+1) in move_arr[i,1][0])
                        l_vote_tally[i] += -1 + 2*vote

        vert_AOD_cmds = np.zeros(n_cols)
        horiz_AOD_cmds = np.zeros(n_rows)
        total_vote_sum = 0
        collision_inds = [] # FIX
        # checking for collisions in the center column around which we condense
        if min_col_ind not in [0,n_cols-1]:
            collisions = arr1.matrix[:,min_col_ind-1,0]*arr1.matrix[:,min_col_ind+1,0]
            if np.sum(collisions) > 0:
                collision_inds = np.where(collisions == 1)[0]
        for row_ind in range(len(r_vote_tally)):

            if r_vote_tally[row_ind]+l_vote_tally[row_ind] > 0 and row_ind+start_row not in collision_inds:
                horiz_AOD_cmds[row_ind+start_row] = 1
                total_vote_sum += r_vote_tally[row_ind]+l_vote_tally[row_ind]
        for col_ind in range(n_cols):
            if np.sign(col_ind-min_col_ind) == 1:
                vert_AOD_cmds[col_ind] = 3
            elif np.sign(col_ind-min_col_ind) == -1:
                vert_AOD_cmds[col_ind] = 2
        
        r_vert_AOD_cmds = np.zeros(n_cols)
        l_vert_AOD_cmds = np.zeros(n_cols)
        r_horiz_AOD_cmds = np.zeros(n_rows)
        l_horiz_AOD_cmds = np.zeros(n_rows)
        r_vote_sum = 0
        l_vote_sum = 0
        for row_ind in range(len(r_vote_tally)):
            n_r_votes = r_vote_tally[row_ind]
            n_l_votes = l_vote_tally[row_ind]
            if n_r_votes > 0:
                r_horiz_AOD_cmds[row_ind+start_row] = 1
                r_vote_sum += n_r_votes
            elif n_l_votes > 0:
                l_horiz_AOD_cmds[row_ind+start_row] = 1
                l_vote_sum += n_l_votes
        for col_ind in range(min_col_ind+1, n_cols):
            r_vert_AOD_cmds[col_ind] = 3
        for col_ind in range(0,min_col_ind):
            l_vert_AOD_cmds[col_ind] = 2
        
        crunch_moves = movr.get_move_list_from_AOD_cmds(horiz_AOD_cmds, vert_AOD_cmds)
        r_moves = movr.get_move_list_from_AOD_cmds(r_horiz_AOD_cmds, r_vert_AOD_cmds)
        l_moves = movr.get_move_list_from_AOD_cmds(l_horiz_AOD_cmds, l_vert_AOD_cmds)
        moves_options = [crunch_moves, r_moves, l_moves]
        vote_sums = [total_vote_sum, r_vote_sum, l_vote_sum]
        most_votes = np.max(vote_sums)
        move_list = moves_options[np.where(vote_sums == most_votes)[0][0]]
        if move_list != []:
            arr1.move_atoms(move_list)
            global_move_set.append(move_list)
        else:
            break

    return global_move_set


# def balance_and_compact(array):
#     arr1 = copy.deepcopy(array)
#     start_row, start_col, end_row, end_col = get_target_locs(arr1)
#     # 1. prebalance (making sure target rows/cols have enough atoms)
#     master_move_list, col_compact, success_flag = prebalance(arr1.matrix, arr1.target)
#     if col_compact == False and success_flag:
#         _,_ = arr1.evaluate_moves(master_move_list)
#         # 2. balance (distributing atoms between target rows according to needs)
#         assignments = get_all_balance_assignments(start_row, end_row)
#         for assignment in assignments:
#             bal_moves = balance_rows(arr1.matrix, arr1.target, assignment[0], assignment[1])
#             if assignment[0] != assignment[1] and len(bal_moves) > 0:
#                 _, _ = arr1.evaluate_moves(bal_moves)
#                 master_move_list.extend(bal_moves)
#         # 3. compact
#         com_moves = compact(arr1)
#         if len(com_moves) > 0:
#             _, _ = arr1.evaluate_moves(com_moves)
#             master_move_list.extend(com_moves)
#     return master_move_list