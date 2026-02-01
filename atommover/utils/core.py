# Core utilities for initializing and analyzing atom arrays 

import copy
import math
import random
import numpy as np
try:
    from numba import jit
except ImportError:
    def jit(func=None, **kwargs):
        def wrapper(f):
            return f
        if func is not None:
            return wrapper(func)
        return wrapper
import math
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional

###########
# Classes #
###########

class Configurations(IntEnum):
    """ Class to be used in conjunction with `AtomArray.generate_target()`
        and `generate_target_config()` to prepare common patterns of atoms. """
    ZEBRA_HORIZONTAL = 0
    ZEBRA_VERTICAL = 1
    CHECKERBOARD = 2
    MIDDLE_FILL = 3
    Left_Sweep = 4
    SEPARATE = 5 # for dual-species only
    RANDOM = 6

CONFIGURATION_PLOT_LABELS = {Configurations.ZEBRA_HORIZONTAL: 'Horizontal zebra stripes',
                             Configurations.ZEBRA_VERTICAL: 'Vertical zebra stripes',
                             Configurations.CHECKERBOARD: 'Checkerboard',
                             Configurations.MIDDLE_FILL: 'Middle fill rectangle', 
                             Configurations.Left_Sweep: 'Left Sweep',
                             Configurations.RANDOM: 'Random'}

class PhysicalParams:
    """ Class used to store various physical parameters corresponding to atom, array and tweezer properties. 
    
    ## Parameters
    AOD_speed : float
        the speed of the moving tweezers, in um/us. Default: 0.1
    spacing : float
        spacing between adjacent atoms in the square array, in m. Default: 5e-6
    loading_prob : float
        the probability that a single site will be filled during loading. Default: 0.6
    target_occup_prob : float
        if the target configuration is random, the probability that a site in the
        configuration will be occupied by an atom. Default: 0.5
    """
    def __init__(self, 
                 AOD_speed: float = 0.1, 
                 spacing: float = 5e-6, 
                 loading_prob: float = 0.6,
                 target_occup_prob: float = 0.5) -> None:
        # array parameters
        self.spacing = spacing
        if loading_prob > 1 or loading_prob < 0:
            raise ValueError("Variable `loading_prob` must be in range [0,1].")
        if target_occup_prob > 1 or target_occup_prob < 0:
            raise ValueError("Variable `target_occup_prob` must be in range [0,1].")
        self.loading_prob = loading_prob
        self.target_occup_prob = target_occup_prob
        
        # tweezer parameters
        self.AOD_speed = AOD_speed

class ArrayGeometry(IntEnum):
    """ Class that specifies the geometry of the atom array. See references
        [LattPy](https://lattpy.readthedocs.io/en/latest/)
    """
    SQUARE = 0
    RECTANGULAR = 1 # NOT SUPPORTED YET; see CONTRIBUTING.md
    TRIANGULAR = 2 # NSY
    BRAVAIS = 3 # NSY
    DECORATED_BRAVAIS = 4 # NSY
    RECTANGLE_TALL = 5


#############
# Functions #
#############

def random_loading(nrows, ncols, probability):
    # Use Python's `random` module for determinism when tests call random.seed()
    matrix = np.zeros((nrows, ncols), dtype=np.int64)
    for i in range(nrows):
        for j in range(ncols):
            if random.random() < probability:
                matrix[i, j] = 1
    return matrix

def generate_random_init_target_configs(n_shots,load_prob, max_sys_size, target_config = None):
    init_config_storage = []
    target_config_storage = []
    for _ in range(n_shots):
        initial_config = random_loading(max_sys_size, max_sys_size, load_prob)
        init_config_storage.append(initial_config)
        if target_config == [Configurations.RANDOM]:
            target = random_loading(max_sys_size, max_sys_size, load_prob - 0.1)
            target_config_storage.append(target)
    return init_config_storage, target_config_storage

def generate_random_init_configs(n_shots, load_prob, shape, n_species=1):
    init_config_storage = []
    rows, cols = int(shape[0]), int(shape[1])
    base_shape = (rows, cols)
    for _ in range(n_shots):
        if n_species == 1:
            initial_config = random_loading(rows, cols, load_prob)
        elif n_species == 2:
            initial_config = np.zeros((rows, cols, 2))
            dual_species_prob = 2 - 2*math.sqrt(1-load_prob)
            initial_config[:,:,0] = random_loading(rows, cols, dual_species_prob/2)
            initial_config[:,:,1] = random_loading(rows, cols, dual_species_prob/2)

            # Randomly leave one atom if two species load into the same tweezer
            for i in range(len(initial_config)):
                for j in range(len(initial_config[0])):
                    if initial_config[i][j][0] == 1 and initial_config[i][j][1] == 1:
                        random_index = random.randint(0, 1)
                        initial_config[i][j][random_index] = 0
        else:
            raise ValueError(f'Argument `n_species` must be either 1 or 2; the provided value is {n_species}.')

        init_config_storage.append(initial_config)

    return init_config_storage

def generate_random_target_configs(n_shots: int, targ_occup_prob: float, shape: list):
    """
    Generates random target configurations, with site 
    occupation probability equal to targ_occup_prob.
    """
    target_config_storage = []
    rows, cols = int(shape[0]), int(shape[1])
    for shot in range(n_shots):
        target = random_loading(rows, cols, targ_occup_prob)
        target_config_storage.append(target)
    return target_config_storage


def count_atoms_in_columns(matrix):
    num_columns = len(matrix[0])

    # Initialize a list to store the count of atoms in each column
    column_counts = [0] * num_columns

    # Iterate through each column and count the number of atoms
    for col in range(num_columns):
        for row in matrix:
            column_counts[col] += row[col]

    return column_counts

def left_right_atom_in_row(row, direction):
    for i in range(len(row))[::-direction]:
        if row[i] == 1:
            return i

def top_bot_atom_in_col(col, direction):
    for i in range(len(col))[::-direction]:
        if col[i] == 1:
            return i
        
def find_lowest_atom_in_col(col):
    for i in range(len(col))[::-1]:
        if col[i] == 1:
            return i

def get_move_distance(from_row, from_col, to_row, to_col, spacing = 5e-6):
    move_distance = ((abs(from_row - to_row)) + (abs(from_col - to_col)))*spacing
    return move_distance

def atom_loss(matrix: np.ndarray, move_time: float, lifetime: float = 30) -> tuple[np.ndarray, bool]:
    """ 
        Given an array of atoms, simulates the process of atom loss
        over a length of time `move_time`.
    
        Specifically, for each atom it calculates the probability (equal
        to exp(-move_time/lifetime)) of a background gas particle colliding
        with the atom and knocking it out of its trap.
    """
    loss_flag = 0
    # matrix can be 2D (single-species) or 3D (dual-species)
    shape = np.shape(matrix)
    if len(shape) == 2:
        rows, cols = int(shape[0]), int(shape[1])
        loss_mask = random_loading(rows, cols, np.exp(-move_time/lifetime))
        matrix_copy = np.multiply(matrix, loss_mask)
    elif len(shape) == 3:
        rows, cols, species = int(shape[0]), int(shape[1]), int(shape[2])
        loss_mask = np.zeros((rows, cols, species), dtype=matrix.dtype)
        for k in range(species):
            loss_mask[:,:,k] = random_loading(rows, cols, np.exp(-move_time/lifetime))
        matrix_copy = np.multiply(matrix, loss_mask)
    else:
        raise ValueError("Unsupported matrix shape for atom_loss")

    if not np.array_equal(matrix, matrix_copy):
        loss_flag = 1
    return matrix_copy, loss_flag

def atom_loss_dual(matrix: np.ndarray, move_time: float, lifetime: float = 30) -> tuple[np.ndarray, bool]:
    """ 
        Given a Numpy array representing a dual-species atom array, 
        simulates the process of atom loss over a length of time `move_time`.
    """
    loss_flag = 0
    shape = np.shape(matrix)
    if len(shape) != 3:
        raise ValueError('atom_loss_dual expects a 3D matrix with shape (rows,cols,2)')
    rows, cols, species = int(shape[0]), int(shape[1]), int(shape[2])
    loss_mask = np.zeros((rows, cols, species), dtype=matrix.dtype)
    for k in range(species):
        loss_mask[:,:,k] = random_loading(rows, cols, np.exp(-move_time/lifetime))
    matrix_copy = np.multiply(matrix, loss_mask)
    for i in range(rows):
        for j in range(cols):
            if matrix_copy[i][j][0] != matrix[i][j][0] or matrix_copy[i][j][1] != matrix[i][j][1]:
                loss_flag = 1
    return matrix_copy, loss_flag

def count_atoms_in_row(row):
    return np.sum(row)

def calculate_filling_fraction(atom_count, row_length):
    return (atom_count / row_length) * 100

def save_frames(temp_frames, combined_frames):
    combined_frames.extend(temp_frames)
    temp_frames.clear()
    return temp_frames, combined_frames

def generate_middle_fifty(length, filling_threshold = 0.5):
    # TODO this only works for square arrays, generalize to rectangular
    max_L = length
    while (max_L**2)/(length**2) >= filling_threshold:
        max_L -= 1
    return [max_L , max_L]


@dataclass
class ArrayGeometrySpec:
    """Specification for desired loading geometry.

    - kind: an ArrayGeometry member.
    - params: optional dict of parameters (algorithm-specific).
    """
    kind: ArrayGeometry
    params: Optional[dict] = None


def array_shape_for_geometry(geometry_spec, target_size: int, loading_prob: float = 0.6) -> tuple[int, int]:
    """Return (rows, cols) for a loading array given `geometry_spec`.

    Accepted `geometry_spec` types:
    - None: square fallback (see rules).
    - `ArrayGeometrySpec` instance: follows `kind`.
    - tuple/list of two ints: interpreted as (rows, cols).
    """
    try:
        t = int(target_size)
        if t <= 0:
            raise ValueError
    except Exception:
        raise ValueError("target_size must be a positive integer.")

    # If a plain two-int tuple/list provided -> coerce
    if isinstance(geometry_spec, (list, tuple)) and len(geometry_spec) == 2:
        try:
            rows = int(geometry_spec[0])
            cols = int(geometry_spec[1])
        except Exception:
            raise ValueError("Geometry tuple/list must contain two integers.")
        rows = max(rows, t)
        cols = max(cols, t)
        return rows, cols

    # If None or SQUARE: square fallback scaled by loading_prob
    if geometry_spec is None or (isinstance(geometry_spec, ArrayGeometrySpec) and geometry_spec.kind == ArrayGeometry.SQUARE):
        side = int(math.ceil(math.sqrt(t)))
        scale = int(math.ceil(1.0 / math.sqrt(float(loading_prob)))) if loading_prob and loading_prob > 0 else 1
        side = side * scale
        # Ensure some donor margin around the target so rearrangement algorithms
        # can source atoms from outside the target region. Make array at least
        # target_size + 2 on a side.
        side = max(side, t + 2)
        return side, side

    # ArrayGeometrySpec handling
    if isinstance(geometry_spec, ArrayGeometrySpec):
        if geometry_spec.kind == ArrayGeometry.RECTANGLE_TALL:
            params = geometry_spec.params or {}
            preferred_width_factor = float(params.get("preferred_width_factor", 2.0))
            min_extra_columns = int(params.get("min_extra_columns", 2))
            rows = int(t)
            cols = int(math.ceil(t * preferred_width_factor))
            cols = max(cols, t + min_extra_columns)
            rows = max(rows, t)
            cols = max(cols, t)
            return rows, cols
        # Fallback: unknown kinds -> square fallback
        side = int(math.ceil(math.sqrt(t)))
        scale = int(math.ceil(1.0 / math.sqrt(float(loading_prob)))) if loading_prob and loading_prob > 0 else 1
        side = side * scale + 2
        side = max(side, t)
        return side, side

    raise ValueError("Unsupported `geometry_spec` type; expected ArrayGeometrySpec, (rows,cols), or None.")