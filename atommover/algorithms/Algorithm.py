# Parent class for all algorithms.
# For developers: feel free to use this as a template for new algorithms.
#                 Each function below describes the requirements/what you need to put in it.

# Author: Nikhil Harle

import numpy as np

from atommover.utils.AtomArray import AtomArray

class Algorithm:
    """ Parent class for all algorithms. 
    
    
        NB: The following functions are placeholders for illustrative purposes only and 
        should be overwritten for your particular algorithm.
        
        If your algorithm can only prepare select target configurations, please list them here.
        
        e.g:
        
        Supported configurations: Middle Fill (see `atommover.utils.core.Configurations`
        for a list of configurations). 
        
         """
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return 'Insert the name of your algorithm here. This is what will show up on your benchmarking plots.'

    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False) -> tuple[np.ndarray, list, bool]:
        """ 
        This is the main function for the algorithm. 
        
        It should take as input:
            1. an AtomArray object (which, btw, contains the initial 
               configuration `atom_array.matrix` and the target configuration 
               `atom_array.target`).
            2. an optional argument to run an ejection subroutine
               (see `atommover.algorithms.source.ejection.py` for the protocol).
            3. any other (optional!) kwargs you see fit to include :)
        
        It should provide as output:
            1. `config` (np.ndarray) - the final configuration after all moves have been applied
               (ideally, this should just be the target configuration)

            2. `move_list` (list of lists of `Move` objects) - contains all 
               the moves to transform the initial configuration into the final configuration. 
               Each list inside `move_list` is a set of moves that will be done in parallel.
    
               If you're confused by lists inside of lists, consider the following example:
               `move_list = [small_move_list, small_move_list1, small_move_list2]`
               `small_move_list = [Move1], small_move_list1 = [Move2, Move3], small_move_list2 = 
               [Move4, Move5]`
               
               When this is read by the framework, it will first execute Move1, then will execute
               Move2 and Move3 in parallel, then after that Move4 and Move5 will be executed in 
               parallel.

            3. `success_flag` (bool) - simple sanity check. This should be set to `True` if the 
               algorithm prepares the final configuration and `False` if it does not. This is
               helpful during benchmarking.
        """
        config = np.zeros(atom_array.shape)
        move_list = []
        success_flag = False

        # your code here #

        return config, move_list, success_flag
    
