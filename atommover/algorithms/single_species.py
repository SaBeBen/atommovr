# Single-species algorithms. 

# FOR CONTRIBUTORS:
# - Please write your algorithm in a separate .py file
# - Once you have done that, please make an algorithm class with the following three functions (see the `Algorithm` class for more details)
#   1. __repr__(self) - this should return the name of your algorithm, to be used in plots.
#   2. get_moves(self) - given an AtomArray object, returns a list of lists of Move() objects.
#   3. (optional) __init__() - if your algorithm needs to use arguments that cannot be specified in AtomArray
import math
import numpy as np

from atommover.utils.AtomArray import AtomArray
from atommover.algorithms.Algorithm_class import Algorithm
from atommover.algorithms.source.balance_compact import balance_and_compact
from atommover.algorithms.source.bc_new import bcv2
from atommover.algorithms.source.ejection import ejection
from atommover.algorithms.source.generalized_balance import generalized_balance
try:
    from atommover.algorithms.source.Hungarian_works import parallel_Hungarian_algorithm_works, parallel_LBAP_algorithm_works, Hungarian_algorithm_works
except Exception:
    parallel_Hungarian_algorithm_works = None
    parallel_LBAP_algorithm_works = None
    Hungarian_algorithm_works = None
from atommover.algorithms.source.pcfa import pcfa_algorithm
from atommover.utils.core import ArrayGeometry, ArrayGeometrySpec
from atommover.algorithms.source.tetris import tetris_algorithm


def _to_single_species_plane(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 3:
        return arr[:, :, 0]
    return arr.copy()


def _finalize_with_standard_ejection(
    state: np.ndarray,
    target: np.ndarray,
    move_batches: list,
    do_ejection: bool | str = False
) -> tuple[np.ndarray, list, bool]:
    target_2d = _to_single_species_plane(target)
    state_2d = _to_single_species_plane(state)
    combined_moves = list(move_batches)
    ejection_method = "sublattice"
    ejection_flag = bool(do_ejection)
    if isinstance(do_ejection, str):
        ejection_method = do_ejection
        ejection_flag = True
    if ejection_flag:
        bounds = [0, state_2d.shape[0] - 1, 0, state_2d.shape[1] - 1]
        eject_moves, state_2d = ejection(state_2d, target_2d, bounds, method=ejection_method)
        combined_moves.extend(eject_moves)
        success_flag = Algorithm.get_success_flag(state_2d, target_2d, do_ejection=True, n_species=1)
    else:
        success_flag = Algorithm.get_success_flag(state_2d, target_2d, do_ejection=False, n_species=1)
    return state_2d, combined_moves, success_flag

##########################
# Bernien Lab algorithms #
##########################

# Parallel Hungarian
class ParallelHungarian(Algorithm):
    """ A variant on the Hungarian matching algorithm that parallelizes the moves
        instead of executing them sequentially (one by one).
        
        Supported configurations: all. """
    def __repr__(self):
        return 'Parallel Hungarian'
    
    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False, final_size: list = [], round_lim: int = 0):
        if atom_array.n_species != 1:
            raise ValueError(f"Single-species algorithm cannot process atom array with {atom_array.n_species} species.")
        if round_lim == 0:
            round_lim = int(np.sum(atom_array.target))
        state = _to_single_species_plane(atom_array.matrix)
        target = _to_single_species_plane(atom_array.target)
        config, moves, _ = parallel_Hungarian_algorithm_works(state, target, round_lim=round_lim)
        return _finalize_with_standard_ejection(config, atom_array.target, moves, do_ejection)


class ParallelLBAP(Algorithm):
    """ Solves the linear bottleneck assignment problem and parallelizes the moves.
        Code taken from ParallelHungarian.
        
        Supported configurations: all. """
    def __repr__(self):
        return 'Parallel LBAP'
    
    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False, final_size: list = [], round_lim: int = 0):
        if atom_array.n_species != 1:
            raise ValueError(f"Single-species algorithm cannot process atom array with {atom_array.n_species} species.")
        if round_lim == 0:
            round_lim = int(np.sum(atom_array.target))
        state = _to_single_species_plane(atom_array.matrix)
        target = _to_single_species_plane(atom_array.target)
        config, moves, _ = parallel_LBAP_algorithm_works(state, target, round_lim=round_lim)
        return _finalize_with_standard_ejection(config, atom_array.target, moves, do_ejection)
    


# Generalized Balance
class GeneralizedBalance(Algorithm):
    """Implements the generalized balance algorithm, which alternatively operates 
       row balance and column balance algorithms, as originally described by Bo-Yu
       and Nikhil in the Bernien lab meeting GM 268.
       
       Supported configurations: all. """

    def __repr__(self):
        return 'Generalized Balance'
    
    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False):
        if atom_array.n_species != 1:
            raise ValueError(f"Single-species algorithm cannot process atom array with {atom_array.n_species} species.")
        state = _to_single_species_plane(atom_array.matrix)
        target = _to_single_species_plane(atom_array.target)
        config, moves, _ = generalized_balance(state, target, do_ejection=False)
        return _finalize_with_standard_ejection(config, atom_array.target, moves, do_ejection)



###########################################
# Existing algorithms from the literature #
###########################################

# Hungarian
class Hungarian(Algorithm):
    """ Implements the Hungarian matching algorithm, which generates a cost 
        matrix mapping available atoms to the target spots, and solves the
        linear assignment problem to find an efficient set of moves.
        
        Supported configurations: all. """

    def __repr__(self):
        return 'Hungarian'

    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False):
        if atom_array.n_species != 1:
            raise ValueError(f"Single-species algorithm cannot process atom array with {atom_array.n_species} species.")
        state = _to_single_species_plane(atom_array.matrix)
        target = _to_single_species_plane(atom_array.target)
        config, moves, _ = Hungarian_algorithm_works(state, target)
        return _finalize_with_standard_ejection(config, atom_array.target, moves, do_ejection)



# Balance and Compact
class BCv2(Algorithm):
    """Implements the Balance and Compact algorithm, as originally described
       in [PRA 70, 040302(R) (2004)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.040302)
       
       Supported configurations: `Configurations.MIDDLE_FILL`"""

    def __repr__(self):
        return 'Balance & Compact'

    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False):
        if atom_array.n_species != 1:
            raise ValueError(f"Single-species algorithm cannot process atom array with {atom_array.n_species} species.")
        config, moves, _ = bcv2(atom_array)
        return _finalize_with_standard_ejection(config, atom_array.target, moves, do_ejection)
    


# Balance and Compact
class BalanceAndCompact(Algorithm):
    """ NOTE: we recommend that you use the (faster) BCv2 algorithm. 
        This is an older version that we used to make Fig. 2 in the paper.

        A slow implementation of the Balance and Compact algorithm, as originally described
        in [PRA 70, 040302(R) (2004)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.040302)
       
        Supported configurations: `Configurations.MIDDLE_FILL`"""

    def __repr__(self):
        return 'Balance & Compact (slow)'

    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False):
        if atom_array.n_species != 1:
            raise ValueError(f"Single-species algorithm cannot process atom array with {atom_array.n_species} species.")
        state = _to_single_species_plane(atom_array.matrix)
        target = _to_single_species_plane(atom_array.target)
        config, moves, _ = balance_and_compact(state, target, do_ejection=False)
        return _finalize_with_standard_ejection(config, atom_array.target, moves, do_ejection)



# Parallel Compression Filling Algorithm (PCFA)
class PCFA(Algorithm):
    """Implements the Parallel Compression Filling Algorithm as described in Sections 2 & 3
       of the provided PCFA paper excerpt. Steps: row compression, fill defective rows,
       optional ejection of excess atoms. Axis-aligned moves only (same row/column) with
       an optional degree-of-parallelism (dop) limit.

       Supported configurations: square targets (L x L) positioned at top-left of array."""

    preferred_width_factor = 2.1
    min_extra_columns = 4

    def __repr__(self):
        return 'PCFA'

    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False, dop: int | None = None):
        if atom_array.n_species != 1:
            raise ValueError(f"Single-species algorithm cannot process atom array with {atom_array.n_species} species.")
        state = _to_single_species_plane(atom_array.matrix)
        target = _to_single_species_plane(atom_array.target)
        config, moves, _ = pcfa_algorithm(state, target, dop=dop)
        return _finalize_with_standard_ejection(config, atom_array.target, moves, do_ejection)

    preferred_geometry_spec = ArrayGeometrySpec(
        ArrayGeometry.RECTANGLE_TALL,
        {"preferred_width_factor": 2.1, "min_extra_columns": 4},
    )


class Tetris(Algorithm):
    """Implements the Tetris rearrangement protocol from PRAppl 19, 054032.

    Steps: horizontal row construction followed by column compression, with an
    optional ejection pass for leftover atoms.

    Supported configurations: all rectangular targets."""

    def __repr__(self):
        return 'Tetris'

    def get_moves(self, atom_array: AtomArray, do_ejection: bool = False):
        if atom_array.n_species != 1:
            raise ValueError(
                'Single-species algorithm cannot process atom array with '
                f"{atom_array.n_species} species."
            )
        state = _to_single_species_plane(atom_array.matrix)
        target = _to_single_species_plane(atom_array.target)
        config, moves, _ = tetris_algorithm(state, target)
        return _finalize_with_standard_ejection(config, atom_array.target, moves, do_ejection)
        
