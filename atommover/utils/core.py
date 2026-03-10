# Core utilities for initializing and analyzing atom arrays 

import math
import numpy as np
from enum import IntEnum
from numpy.typing import NDArray
from typing import Tuple

###########
# Classes #
###########

class Configurations(IntEnum):
    """
    Enumerate common target configurations for atom arrays.
    To be used in conjunction with `AtomArray.generate_target()`
    and `generate_target_config()` to prepare common patterns of atoms.
    """
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
    """
    Stores physical parameters for array loading, targeting, and transport.

    Parameters
    ----------
    AOD_speed : float, optional
        Tweezer transport speed in um/us.
    spacing : float, optional
        Lattice spacing in meters.
    loading_prob : float, optional
        Per-site loading probability.
    target_occup_prob : float, optional
        Per-site occupation probability for random target generation.
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
    """ 
    Enumerate supported and planned atom-array geometries.
    See references [LattPy](https://lattpy.readthedocs.io/en/latest/)
    """
    SQUARE = 0
    RECTANGULAR = 1 # NOT SUPPORTED YET (NSY); see CONTRIBUTING.md
    TRIANGULAR = 2 # NSY
    BRAVAIS = 3 # NSY
    DECORATED_BRAVAIS = 4 # NSY


#############
# Functions #
#############


def _coerce_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """Return a numpy Generator, creating a fresh one if None."""
    return np.random.default_rng() if rng is None else rng


def random_loading(size, 
                   probability: float, 
                   rng: np.random.Generator | None = None
                   ) -> NDArray:
    """
    Sample a random binary occupancy array.

    Parameters
    ----------
    size : sequence
        Array shape, with at least two entries.
    probability : float
        Probability that a site is occupied.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        ``uint8`` occupancy array with entries in ``{0, 1}``.

    Raises
    ------
    ValueError
        If ``size`` is too short or ``probability`` is outside ``[0, 1]``.
    """
    if len(size) < 2:
        raise ValueError(f"`size` must have at least 2 entries; got {size}.")

    n0 = int(size[0])
    n1 = int(size[1])

    if probability < 0 or probability > 1:
        raise ValueError(f"`probability` must be in [0,1]; got {probability}.")

    rng = _coerce_rng(rng)

    if probability == 0:
        return np.zeros((n0, n1), dtype=np.uint8)
    if probability == 1:
        return np.ones((n0, n1), dtype=np.uint8)
    x = rng.random((n0,n1))
    return (x > (1.0 - float(probability))).astype(np.uint8)


def generate_random_init_target_configs(n_shots: int,
                                        load_prob: float,
                                        max_sys_size: int,
                                        target_config=None,
                                        rng: np.random.Generator | None = None,
                                        ) -> Tuple[list, list]:
    """
    Generate paired random initial and target configurations.

    Parameters
    ----------
    n_shots : int
        Number of configurations to generate.
    load_prob : float
        Per-site loading probability for initial configurations.
    max_sys_size : int
        Maximum length along a single axis.
    target_config : object, optional
        Target configuration selector.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    list
        Random initial configurations.
    list
        Random target configurations.
    """
    rng = _coerce_rng(rng)
    init_config_storage = []
    target_config_storage = []

    for _ in range(n_shots):
        initial_config = random_loading([max_sys_size, max_sys_size], load_prob, rng=rng)
        init_config_storage.append(initial_config)

        if target_config == [Configurations.RANDOM]:
            target = random_loading([max_sys_size, max_sys_size], load_prob - 0.1, rng=rng)
            target_config_storage.append(target)

    return init_config_storage, target_config_storage


def generate_random_init_configs(n_shots: int,
                                 load_prob: float,
                                 max_sys_size: int,
                                 n_species: int = 1,
                                 rng: np.random.Generator | None = None,
                                 ) -> list:
    """
    Generate random initial configurations for one- or two-species arrays.

    Parameters
    ----------
    n_shots : int
        Number of configurations to generate.
    load_prob : float
        Marginal per-site loading probability.
    max_sys_size : int
        Maximum length along one axis.
    n_species : int, optional
        Number of species. Must be ``1`` or ``2``.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    list
        Randomly generated initial configurations.

    Raises
    ------
    ValueError
        If ``n_species`` is not supported.
    """
    rng = _coerce_rng(rng)

    init_config_storage = []

    for _ in range(n_shots):
        if n_species == 1:
            initial_config = random_loading([max_sys_size, max_sys_size], load_prob, rng=rng)

        elif n_species == 2:
            initial_config = np.zeros((max_sys_size, max_sys_size, 2), dtype=np.uint8)

            dual_species_prob = 2 - 2 * math.sqrt(1 - load_prob)
            p_each = dual_species_prob / 2

            initial_config[:, :, 0] = random_loading([max_sys_size, max_sys_size], p_each, rng=rng)
            initial_config[:, :, 1] = random_loading([max_sys_size, max_sys_size], p_each, rng=rng)

            # Resolve double-occupancy sites by dropping one species uniformly at random.
            both = (initial_config[:, :, 0] == 1) & (initial_config[:, :, 1] == 1)
            if np.any(both):
                idx = np.argwhere(both)  # shape (k, 2)
                drop = rng.integers(0, 2, size=idx.shape[0])  # which species channel to drop
                initial_config[idx[:, 0], idx[:, 1], drop] = 0

        else:
            raise ValueError(
                f"Argument `n_species` must be either 1 or 2; the provided value is {n_species}."
            )

        init_config_storage.append(initial_config)

    return init_config_storage


def generate_random_target_configs(n_shots: int,
                                   targ_occup_prob: float,
                                   shape: list,
                                   rng: np.random.Generator | None = None,
                                   ):
    """
    Generate random target configurations for one- or two-species arrays.

    Parameters
    ----------
    n_shots : int
        Number of configurations to generate.
    load_prob : float
        Marginal per-site loading probability.
    max_sys_size : int
        Maximum length along one axis.
    n_species : int, optional
        Number of species. Must be ``1`` or ``2``.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    list
        Randomly generated target configurations.

    Raises
    ------
    ValueError
        If ``n_species`` is not supported.
    """
    rng = _coerce_rng(rng)
    target_config_storage = []
    for _ in range(n_shots):
        target = random_loading(shape, targ_occup_prob, rng=rng)
        target_config_storage.append(target)
    return target_config_storage


def count_atoms_in_columns(matrix: NDArray) -> list:
    """
    Count atoms in each column.

    Returns a Python list for backward compatibility.
    """
    return np.sum(np.asarray(matrix), axis=0).tolist()


def left_right_atom_in_row(row: int, direction: int) -> int | None:
    """Returns the leftmost or rightmost occupied site in a row."""
    row_arr = np.asarray(row)
    occupied = np.flatnonzero(row_arr == 1)
    if occupied.size == 0:
        return None
    # direction convention preserved from old implementation:
    # direction=1 -> rightmost (via [::-1]); direction=-1 -> leftmost
    return int(occupied[-1] if direction == 1 else occupied[0])


def top_bot_atom_in_col(col, direction):
    """Returns the topmost or bottommost occupied site in a column."""
    col_arr = np.asarray(col)
    occupied = np.flatnonzero(col_arr == 1)
    if occupied.size == 0:
        return None
    return int(occupied[-1] if direction == 1 else occupied[0])


def find_lowest_atom_in_col(col: int) -> int | None:
    """Returns the bottommost occupied site in a column."""
    col_arr = np.asarray(col)
    occupied = np.flatnonzero(col_arr == 1)
    if occupied.size == 0:
        return None
    return int(occupied[-1])


def get_move_distance(from_row: int, 
                      from_col: int, 
                      to_row: int, 
                      to_col: int, 
                      spacing: float = 5e-6
                      ) -> float:
    """Returns the Manhattan distance of a move."""
    move_distance = (abs(from_row - to_row) + abs(from_col - to_col)) * spacing
    return move_distance


def atom_loss(matrix: np.ndarray,
              move_time: float,
              lifetime: float = 30,
              rng: np.random.Generator | None = None,
    ) -> Tuple[NDArray, bool]:
    """
    Sample atom loss over a finite evolution time.

    Parameters
    ----------
    matrix : np.ndarray
        Occupancy array.
    move_time : float
        Evolution time.
    lifetime : float, optional
        Vacuum-limited lifetime of a single atom in a tweezer.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Post-loss occupancy array.
    bool
        ``True`` if at least one atom was lost.

    Raises
    ------
    ValueError
        If ``lifetime`` is nonpositive or ``matrix`` has unsupported rank.
    """
    if lifetime <= 0:
        raise ValueError(f"`lifetime` must be > 0; got {lifetime}.")

    p_survive = float(np.exp(-move_time / lifetime))
    rng = _coerce_rng(rng)

    # Build a 2D survival mask
    mask2d = random_loading(list(np.shape(matrix)), p_survive, rng=rng)

    if matrix.ndim == 2:
        mask = mask2d
    elif matrix.ndim == 3:
        mask = mask2d[:, :, None]  # broadcast over species axis
    else:
        raise ValueError(f"`matrix` must be 2D or 3D; got shape {matrix.shape}.")

    matrix_copy = np.asarray(matrix).copy()
    matrix_copy = matrix_copy * mask

    loss_flag = bool(np.any(matrix_copy != matrix))
    return matrix_copy, loss_flag


def atom_loss_dual(matrix: NDArray,
                   move_time: float,
                   lifetime: float = 30,
                   rng: np.random.Generator | None = None,
    ) -> Tuple[NDArray, bool]:
    """
    Sample atom loss for a dual-species array.

    Parameters
    ----------
    matrix : np.ndarray
        Dual-species occupancy array of shape ``(rows, cols, 2)``.
    move_time : float
        Evolution time.
    lifetime : float, optional
        Vacuum-limited lifetime of a single atom in a tweezer.
    rng : np.random.Generator | None, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Post-loss occupancy array.
    bool
        ``True`` if at least one atom was lost.

    Raises
    ------
    ValueError
        If ``matrix`` is not a dual-species array or ``lifetime`` is nonpositive.
    """
    if lifetime <= 0:
        raise ValueError(f"`lifetime` must be > 0; got {lifetime}.")

    if np.asarray(matrix).ndim != 3 or np.asarray(matrix).shape[-1] != 2:
        raise ValueError(f"`matrix` must have shape (rows, cols, 2); got {np.shape(matrix)}.")

    return atom_loss(matrix, move_time, lifetime=lifetime, rng=rng)

def count_atoms_in_row(row: int) -> int:
    return np.sum(row)

def calculate_filling_fraction(atom_count: int, row_length: int) -> float:
    return (atom_count / row_length) * 100

def save_frames(temp_frames: list, combined_frames: list) -> Tuple[list, list]:
    combined_frames.extend(temp_frames)
    temp_frames.clear()
    return temp_frames, combined_frames

def generate_middle_fifty(length: int, filling_threshold: float = 0.5) -> list[int]:
    # TODO this only works for square arrays, generalize to rectangular
    max_L = length
    while (max_L**2)/(length**2) >= filling_threshold:
        max_L -= 1
    return [max_L , max_L]