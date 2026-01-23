# Code to visualize the atom array and generate gifs of the rearrangement process.

import imageio.v2 as imageio
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from atommover.utils.core import *
from atommover.utils.move_utils import MoveType
from atommover.utils.customize import *

######################################
# Single frame visualization/imaging #
######################################

def single_species_image(matrix, move_list: list = [], plt_spacer: float = 0.25, padding: list = [], title: str = '', atom_color: str = SPECIES1COL, savename = ''):
    
    fig, ax = plt.subplots()

    if padding != []:
        ax.add_patch(Rectangle((padding[1][0]-0.5, padding[0][0]-0.5), len(matrix[0]) - np.sum(padding[1]), len(matrix) - np.sum(padding[0]),
                edgecolor = EDGECOL,
                facecolor = 'b',
                fill=False,
                lw=4))

    dotsize = np.min([800/np.sqrt(len(matrix)**2 + len(matrix[0])**2), 80])
    filled_inds_x, filled_inds_y, empty_inds_x, empty_inds_y = _get_inds_for_circ_matr_plot(matrix)
    ax.scatter(filled_inds_x, filled_inds_y, s=dotsize, c=atom_color)#, edgecolor=EDGECOL)
    ax.scatter(empty_inds_x, empty_inds_y, s=dotsize, c=NOATOMCOL, edgecolor=EDGECOL)
    
    # plotting arrows to indicate moves (NB: this does NOT apply the moves, but rather only adds the visualization)
    if len(move_list) > 0:
        for move in move_list:
            ax.arrow(move.from_col+np.sign(move.dx)*plt_spacer, 
                     move.from_row+np.sign(move.dy)*plt_spacer, # len(matrix[0])-1-(move.from_row+np.sign(move.dy)*plt_spacer), 
                     move.dx-np.sign(move.dx)*2*plt_spacer,
                     move.dy-np.sign(move.dy)*2*plt_spacer, # -move.dy+np.sign(move.dy)*2*plt_spacer,
                     color = ARROWCOL, 
                     width = 0.03, 
                     length_includes_head = True)

    ## matplotlib formatting
    if title != '':
        ax.set_title(title)

    _check_and_fix_lims(ax, len(matrix[0]), len(matrix))

    ax.set_aspect('equal') # Make the circles appear closer
    ax.axis('off') # Turn off the axis for a prettier plot
    plt.gca().invert_yaxis() # invert y axis so that it visually represents the matrix state
    if savename != '':
        plt.savefig(f'figs/{savename}')
    plt.show()


#Generate atom arrays figure
def dual_species_image(matrix, move_list: list = [], plt_spacer: float = 0.25, atoms: str = 'all', savename = ''):
    
    fig, ax = plt.subplots()

    blue_inds_x, blue_inds_y, yellow_inds_x, yellow_inds_y, white_inds_x, white_inds_y = _dual_species_get_inds_for_circ_matr_plot(matrix)

    dotsize = np.min([800/np.sqrt(len(matrix)**2 + len(matrix[0])**2), 80])
    if atoms == SPECIES2NAME:
        ax.scatter(blue_inds_x, blue_inds_y, s=dotsize, c=NOATOMCOL, edgecolor=EDGECOL) # Plot correct color for Cs atoms
        ax.scatter(yellow_inds_x, yellow_inds_y, s=dotsize, c=SPECIES2COL, edgecolor=EDGECOL)
    elif atoms == SPECIES1NAME:
        ax.scatter(blue_inds_x, blue_inds_y, s=dotsize, c=SPECIES1COL, edgecolor=EDGECOL)
        ax.scatter(yellow_inds_x, yellow_inds_y, s=dotsize, c=NOATOMCOL, edgecolor=EDGECOL)
    elif atoms == 'all':
        ax.scatter(blue_inds_x, blue_inds_y, s=dotsize, c=SPECIES1COL)#, edgecolor=EDGECOL)
        ax.scatter(yellow_inds_x, yellow_inds_y, s=dotsize, c=SPECIES2COL)#, edgecolor=EDGECOL)
    
    ax.scatter(white_inds_x, white_inds_y, s=dotsize, c=NOATOMCOL, edgecolor=EDGECOL)
    
    eject_x = []
    eject_y = []
    pickup_fail_x = []
    pickup_fail_y = []
    putdown_fail_x = []
    putdown_fail_y = []
    collision_fail_x = []
    collision_fail_y = []
    crossed_x = []
    crossed_y = []
    # plotting arrows to indicate moves
    if len(move_list) > 0:
        for move_ind, move in enumerate(move_list):
            ax.arrow(move.from_col+np.sign(move.dx)*plt_spacer, 
                            len(matrix[0])-(move.from_row+np.sign(move.dy)*plt_spacer), 
                            move.dx-np.sign(move.dx)*2*plt_spacer, 
                            -move.dy+np.sign(move.dy)*2*plt_spacer, 
                            color = ARROWCOL, 
                            width = 0.03, 
                            length_includes_head = True)
            
    _check_and_fix_lims(ax, len(matrix[0]), len(matrix))

    # Make the circles appear closer
    ax.set_aspect('equal')
    ax.axis('off')
    plt.gca().invert_yaxis() # invert y axis so that it visually represents the matrix state
    if savename != '':
        plt.savefig(f'figs/{savename}')
    
    plt.show()

########################
# Movie/gif generation #
########################

def make_single_species_gif(single_species_array, move_list, params: PhysicalParams = PhysicalParams(), savename: str = 'matrix_animation', plt_spacer: float = 0.25, duration: float = 200):
    # making reference time
    t_total = 0

    dotsize = np.min([800/np.sqrt(len(single_species_array.matrix)**2 + len(single_species_array.matrix[0])**2), 80])
    
    # plotting the initial configuration
    fig, ax = plt.subplots()
    blue_inds_x, blue_inds_y, white_inds_x, white_inds_y = _get_inds_for_circ_matr_plot(single_species_array.matrix)
    ax.scatter(blue_inds_x, blue_inds_y, s=dotsize, c=SPECIES1COL, edgecolor=EDGECOL)
    ax.scatter(white_inds_x, white_inds_y, s=dotsize, c=NOATOMCOL, edgecolor=EDGECOL)
    if t_total> 1e-3:
        ax.set_title(f"t = {round(t_total*1e3,3)} ms")
    else:
        ax.set_title(f"t = {int(t_total*1e6)} \u03bc s")
    
    _check_and_fix_lims(ax, len(single_species_array.matrix[0]), len(single_species_array.matrix))
    
    ax.set_aspect('equal') # Make the circles appear closer
    ax.axis('off') # Turn off the axis for a prettier plot
    plt.gca().invert_yaxis()# invert y axis so that it visually represents the matrix state
    plt.savefig('./figs/frames/frame0')
    plt.close()

    # iterating through moves and creating new frames
    for move_ind, move_set in enumerate(move_list):
        # performing the move
        [failed_moves, flags], move_time = single_species_array.move_atoms(move_set)

        # plotting the frame
        fig, ax = plt.subplots()
        blue_inds_x, blue_inds_y, white_inds_x, white_inds_y = _get_inds_for_circ_matr_plot(single_species_array.matrix)
        ax.scatter(blue_inds_x, blue_inds_y, s=dotsize, c=SPECIES1COL, edgecolor=EDGECOL)
        ax.scatter(white_inds_x, white_inds_y, s=dotsize, c=NOATOMCOL, edgecolor=EDGECOL)

        distances = []
        eject_x = []
        eject_y = []
        pickup_fail_x = []
        pickup_fail_y = []
        putdown_fail_x = []
        putdown_fail_y = []
        collision_fail_x = []
        collision_fail_y = []
        crossed_x = []
        crossed_y = []
        for move_set_ind, move in enumerate(move_set):
            # checking whether the move failed or not
            if move_set_ind in failed_moves:
                fail_flag = flags[np.where(np.isclose(failed_moves,move_set_ind))[0][0]]
            else:
                fail_flag = 0

            # calculating the distance of the move
            distances.append(move.distance)


            if move.movetype == MoveType.EJECT_MOVE:
                # plot a green dot if ejection succeeded
                if fail_flag == 0:
                    eject_x.append(move.from_col)
                    eject_y.append(move.from_row) #len(matrix[0])-move.from_row)
                    
            else:
                #plotting an arrow for each individual move
                ax.arrow(move.from_col+np.sign(move.dx)*plt_spacer, 
                          move.from_row+np.sign(move.dy)*plt_spacer, #len(matrix[0])-(move.from_row+np.sign(move.dy)*plt_spacer), 
                          move.dx-np.sign(move.dx)*2*plt_spacer, 
                          move.dy-np.sign(move.dy)*2*plt_spacer, #-move.dy+np.sign(move.dy)*2*plt_spacer, 
                          color = ARROWCOL, 
                          width = 0.03, 
                          length_includes_head = True)
            
            if fail_flag == 1:
                # plot a yellow dot if pickup failed
                pickup_fail_x.append(move.from_col)
                pickup_fail_y.append(move.from_row)
            elif fail_flag == 2:
                # plot a magenta dot if putdown failed
                putdown_fail_x.append(move.from_col)
                putdown_fail_y.append(move.from_row)
            elif fail_flag == 4:
                crossed_x.append(move.from_col)
                crossed_y.append(move.from_row)
            elif move.movetype == MoveType.ILLEGAL_MOVE and fail_flag == 0:
                # plot red dots if atoms collided
                collision_fail_x.append(move.from_col)
                collision_fail_y.append(move.from_row)
                collision_fail_x.append(move.to_col)
                collision_fail_y.append(move.to_row)

            if len(eject_x) > 0:
                ax.scatter(eject_x, eject_y, s=dotsize, c=EJECTCOL, edgecolor=EDGECOL)
            if len(pickup_fail_x) > 0:
                ax.scatter(pickup_fail_x, pickup_fail_y, s=dotsize, c=PICKUPFAILCOL, edgecolor=EDGECOL)
            if len(putdown_fail_x) > 0:
                ax.scatter(putdown_fail_x, putdown_fail_y, s=dotsize, c=PUTDOWNFAILCOL, edgecolor=EDGECOL)
            if len(collision_fail_x) > 0:
                ax.scatter(collision_fail_x, collision_fail_y, s=dotsize, c=COLLISIONFAILCOL, edgecolor=EDGECOL)
            if len(crossed_x) > 0:
                ax.scatter(crossed_x, crossed_y, s=dotsize, c=CROSSEDFAILCOL, edgecolor=EDGECOL)

        # calculating the time
        t_total += move_time

        if t_total> 1e-3:
            ax.set_title(f"t = {round(t_total*1e3,3)} ms")
        else:
            ax.set_title(f"t = {int(t_total*1e6)} \u03bcs")
        
        _check_and_fix_lims(ax, len(single_species_array.matrix[0]), len(single_species_array.matrix))
        
        ax.set_aspect('equal') # Make the circles appear closer
        ax.axis('off') # Turn off the axis for a prettier plot
        plt.gca().invert_yaxis() # invert y axis so that it visually represents the matrix state
        plt.savefig(f'./figs/frames/frame{move_ind+1}')
        plt.close()

        # # simulating atom loss NB: this is now done in `ErrorModel` (see atommover.utils.errormodels)
        # matrix, loss_flag = atom_loss(matrix, t_move, params.lifetime)

    with imageio.get_writer(f'./figs/resorting/{savename}.gif', mode='I', duration = duration) as writer:
        for i in range(len(move_list)+1):
            filename = f'./figs/frames/frame{i}.png'
            image = imageio.imread(filename)
            writer.append_data(image)
    writer.close()

    return t_total

def make_dual_species_gif(dual_species_array, move_list: list, savename = 'matrix_animation',plt_spacer = 0.25, duration = 0.2):
    # making reference time
    t_total = 0
    # arrays = copy.deepcopy(dual_species_matrix)
    
    dotsize = np.min([800/np.sqrt(len(dual_species_array.matrix)**2 + len(dual_species_array.matrix[0])**2), 80])
    # plotting the initial configuration
    fig, ax = plt.subplots()
    blue_inds_x, blue_inds_y, \
    yellow_inds_x, yellow_inds_y, \
    white_inds_x, white_inds_y = _dual_species_get_inds_for_circ_matr_plot(dual_species_array.matrix)

    ax.scatter(blue_inds_x, blue_inds_y, s=dotsize, c=SPECIES1COL, edgecolor=EDGECOL)
    ax.scatter(yellow_inds_x, yellow_inds_y, s=dotsize, c=SPECIES2COL, edgecolor=EDGECOL)
    ax.scatter(white_inds_x, white_inds_y, s=dotsize, c=NOATOMCOL, edgecolor=EDGECOL)

    ax.set_aspect('equal')
    if t_total> 1e-3:
        ax.set_title(f"t = {round(t_total*1e3,3)} ms")
        ax.axis('off')
    else:
        ax.set_title(f"t = {int(t_total*1e6)} \u03bc s")
        ax.axis('off')
    plt.gca().invert_yaxis() # invert y axis so that it visually represents the matrix state
    plt.savefig('./figs/frames/frame0')
    plt.clf()

    # iterating through moves and creating new frames
    for move_ind, move_set in enumerate(move_list):
        # performing the move
        [failed_moves, flags], move_time = dual_species_array.move_atoms(move_set)
        # plotting the frame
        fig, ax = plt.subplots()
        blue_inds_x, blue_inds_y, \
        yellow_inds_x, yellow_inds_y, \
        white_inds_x, white_inds_y = _dual_species_get_inds_for_circ_matr_plot(dual_species_array.matrix)
    
        ax.scatter(blue_inds_x, blue_inds_y, s=dotsize, c=SPECIES1COL, edgecolor=EDGECOL)
        ax.scatter(yellow_inds_x, yellow_inds_y, s=dotsize, c=SPECIES2COL, edgecolor=EDGECOL)
        ax.scatter(white_inds_x, white_inds_y, s=dotsize, c=NOATOMCOL, edgecolor=EDGECOL)


        distances = []
        eject_x = []
        eject_y = []
        pickup_fail_x = []
        pickup_fail_y = []
        putdown_fail_x = []
        putdown_fail_y = []
        collision_fail_x = []
        collision_fail_y = []
        for move_set_ind, move in enumerate(move_set):
            # checking whether the move failed or not
            if move_set_ind in failed_moves:
                fail_flag = flags[np.where(np.isclose(failed_moves,move_set_ind))[0][0]]
            else:
                fail_flag = 0

            if move.to_row > len(dual_species_array.matrix)-1 or move.to_row < 0 or move.to_col < 0 or move.to_col > len(dual_species_array.matrix[0])-1 and fail_flag == 0:
                # plot a green dot if ejection succeeded
                if fail_flag == 0:
                    eject_x.append(move.from_col)
                    eject_y.append(len(dual_species_array.matrix[0])-move.from_row)
                    
            # plotting an arrow for each individual move
            ax.arrow(move.from_col+np.sign(move.dx)*plt_spacer, 
                        len(dual_species_array.matrix[0])-(move.from_row+np.sign(move.dy)*plt_spacer), 
                        move.dx-np.sign(move.dx)*2*plt_spacer, 
                        -move.dy+np.sign(move.dy)*2*plt_spacer, 
                        color = ARROWCOL, 
                        width = 0.03, 
                        length_includes_head = True)
            if fail_flag == 1:
                # plot a yellow dot if pickup failed
                pickup_fail_x.append(move.from_col)
                pickup_fail_y.append(len(dual_species_array.matrix[0])-move.from_row)
            elif fail_flag == 2:
                # plot a magenta dot if putdown failed
                putdown_fail_x.append(move.from_col)
                putdown_fail_y.append(len(dual_species_array.matrix[0])-move.from_row)
            elif fail_flag == 3:
                # plot red dots if atoms collided
                collision_fail_x.append(move.from_col)
                collision_fail_y.append(len(dual_species_array.matrix[0])-move.from_row)
                collision_fail_x.append(move.to_col)
                collision_fail_y.append(len(dual_species_array.matrix[0])-move.to_row)

            if len(eject_x) > 0:
                ax.scatter(eject_x, eject_y, s=dotsize, c=EJECTCOL, edgecolor=EDGECOL)
            if len(pickup_fail_x) > 0:
                ax.scatter(pickup_fail_x, pickup_fail_y, s=dotsize, c=PICKUPFAILCOL, edgecolor=EDGECOL)
            if len(putdown_fail_x) > 0:
                ax.scatter(putdown_fail_x, putdown_fail_y, s=dotsize, c=PUTDOWNFAILCOL, edgecolor=EDGECOL)
            if len(collision_fail_x) > 0:
                ax.scatter(collision_fail_x, collision_fail_y, s=dotsize, c=COLLISIONFAILCOL, edgecolor=EDGECOL)

        # keeping track of the time
        t_total += move_time
        ax.set_aspect('equal')
        if t_total> 1e-3:
            ax.set_title(f"t = {round(t_total*1e3,3)} ms")
            ax.axis('off')
        else:
            ax.set_title(f"t = {int(t_total*1e6)} \u03bcs")
            ax.axis('off')
        plt.gca().invert_yaxis() # invert y axis so that it visually represents the matrix state
        plt.savefig(f'./figs/frames/frame{move_ind+1}')
        plt.clf()

        # # simulating atom loss
        # arrays, loss_flag = atom_loss_dual(arrays, t_move, params.lifetime)

    with imageio.get_writer(f'./figs/{savename}.gif', mode='I', duration = duration) as writer:
        for i in range(len(move_list)+1):
            filename = f'./figs/frames/frame{i}.png'
            image = imageio.imread(filename)
            writer.append_data(image)
    writer.close()

    return t_total

#########
# Utils #
#########

def _get_inds_for_circ_matr_plot(matrix: np.ndarray):
    """
        Given a matrix whose entries $ in {0,1} $,
        returns lists of coordinates which give the locations
        of the 1s and 0s, respectively, in the matrix.
    
        Used to minimize calls to `plt.scatter()` during
        `single_species_image()` and `make_single_species_gif()` 
    """
    filled_inds_x = []
    filled_inds_y = []
    empty_inds_x = []
    empty_inds_y = []
    for row_ind in range(len(matrix)):
        for col_ind in range(len(matrix[0])):
            try:
                matval = matrix[row_ind][col_ind]
                if matval == 1:
                    pass
            except ValueError:
                matval = matrix[row_ind][col_ind][0]
            if matval == 1:
                filled_inds_x.append(col_ind)
                filled_inds_y.append(row_ind) #len(matrix)-1-row_ind)
            else:
                empty_inds_x.append(col_ind)
                empty_inds_y.append(row_ind) #len(matrix)-1-row_ind)
    return filled_inds_x, filled_inds_y, empty_inds_x, empty_inds_y

def _dual_species_get_inds_for_circ_matr_plot(matrix: np.ndarray):
    r""" Given a matrix whose entries $$ \in {0,1, 2} $$,
         returns lists of coordinates which give the locations
         of the 1s and 0s, respectively, in the matrix.
        
         Used to minimize calls to `plt.scatter()` during
         `dual_species_image()` and `make_dual_species_gif()` """
    blue_inds_x = []
    blue_inds_y = []
    yellow_inds_x = []
    yellow_inds_y = []
    white_inds_x = []
    white_inds_y = []

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j][0] == 1:
                blue_inds_x.append(j)
                blue_inds_y.append(len(matrix[0])-i)
            elif matrix[i][j][1] == 1:
                yellow_inds_x.append(j)
                yellow_inds_y.append(len(matrix[0])-i)
            else:
                white_inds_x.append(j)
                white_inds_y.append(len(matrix[0])-i)
                
    return blue_inds_x, blue_inds_y, yellow_inds_x, yellow_inds_y, white_inds_x, white_inds_y


def _check_and_fix_lims(ax, xlen, ylen):
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    pct_extension = 0.05
    if xright < xlen*(1+pct_extension):
        xmean = (xleft+xright)/2
        new_xright = xlen*(1+pct_extension)
        ax.set_xlim([xmean-(new_xright-xmean), new_xright])
    if ytop < ylen*(1+pct_extension):
        ymean = (ybottom+ytop)/2
        new_ytop = ylen*(1+pct_extension)
        ax.set_ylim([ymean-(new_ytop-ymean), new_ytop])