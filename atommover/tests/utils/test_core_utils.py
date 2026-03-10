import math
import numpy as np
import pytest

from atommover.utils.core import (
    Configurations,
    random_loading,
    generate_random_init_target_configs,
    generate_random_init_configs,
    generate_random_target_configs,
    count_atoms_in_columns,
    left_right_atom_in_row,
    top_bot_atom_in_col,
    find_lowest_atom_in_col,
    atom_loss,
    atom_loss_dual,
)


# -----------------------------
# Old reference implementations (seeded, embedded in tests)
# -----------------------------

def old_random_loading_ref(size, probability, rng: np.random.Generator) -> np.ndarray:
    """
    Old semantics (loop-based), but uses passed-in RNG for deterministic comparison.
    Returns float array like old implementation.
    """
    x = rng.random((size[0], size[1]))
    matrix = np.zeros_like(x)  # float
    for i in range(size[0]):
        for j in range(size[1]):
            if x[i, j] > 1 - probability:
                matrix[i, j] = 1
    return matrix


def old_generate_random_init_target_configs_ref(
    n_shots, load_prob, max_sys_size, target_config=None, rng=None
):
    rng = np.random.default_rng() if rng is None else rng
    init_config_storage = []
    target_config_storage = []
    for _ in range(n_shots):
        initial_config = old_random_loading_ref([max_sys_size, max_sys_size], load_prob, rng)
        init_config_storage.append(initial_config)
        if target_config == [Configurations.RANDOM]:
            target = old_random_loading_ref([max_sys_size, max_sys_size], load_prob - 0.1, rng)
            target_config_storage.append(target)
    return init_config_storage, target_config_storage


def old_generate_random_init_configs_ref(n_shots, load_prob, max_sys_size, n_species=1, rng=None):
    """
    Old semantics but all randomness comes from numpy Generator `rng` so output is reproducible.
    For dual species overlap resolution, use rng.integers(0,2) instead of python random.randint.
    """
    rng = np.random.default_rng() if rng is None else rng
    init_config_storage = []

    for _ in range(n_shots):
        if n_species == 1:
            initial_config = old_random_loading_ref([max_sys_size, max_sys_size], load_prob, rng)

        elif n_species == 2:
            initial_config = np.zeros((max_sys_size, max_sys_size, 2), dtype=float)
            dual_species_prob = 2 - 2 * math.sqrt(1 - load_prob)
            initial_config[:, :, 0] = old_random_loading_ref([max_sys_size, max_sys_size], dual_species_prob / 2, rng)
            initial_config[:, :, 1] = old_random_loading_ref([max_sys_size, max_sys_size], dual_species_prob / 2, rng)

            for i in range(len(initial_config)):
                for j in range(len(initial_config[0])):
                    if initial_config[i][j][0] == 1 and initial_config[i][j][1] == 1:
                        random_index = int(rng.integers(0, 2))
                        initial_config[i][j][random_index] = 0
        else:
            raise ValueError(f'Argument `n_species` must be either 1 or 2; the provided value is {n_species}.')

        init_config_storage.append(initial_config)

    return init_config_storage


def old_generate_random_target_configs_ref(n_shots, targ_occup_prob, shape, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    out = []
    for _ in range(n_shots):
        out.append(old_random_loading_ref(shape, targ_occup_prob, rng))
    return out


def old_atom_loss_ref(matrix: np.ndarray, move_time: float, lifetime: float, rng: np.random.Generator):
    if lifetime <= 0:
        raise ValueError
    p_survive = np.exp(-move_time / lifetime)
    loss_mask_vals = old_random_loading_ref(list(np.shape(matrix)), p_survive, rng)
    loss_mask = loss_mask_vals.reshape(np.shape(matrix))
    matrix_copy = np.array(matrix, copy=True)
    matrix_copy = np.multiply(matrix_copy, loss_mask).reshape(np.shape(matrix))
    loss_flag = not np.array_equal(matrix, matrix_copy)
    return matrix_copy, bool(loss_flag)


# -----------------------------
# random_loading
# -----------------------------

def test_REGRESSION_random_loading_matches_old_reference_bitwise_with_seed() -> None:
    rng_old = np.random.default_rng(123)
    rng_new = np.random.default_rng(123)

    old = old_random_loading_ref([8, 9], 0.37, rng_old)
    new = random_loading([8, 9], 0.37, rng=rng_new)

    # old is float {0,1}; new is uint8 {0,1}. Compare values.
    assert np.array_equal(old.astype(np.uint8), new)
    assert new.dtype == np.uint8


@pytest.mark.parametrize("p, expected", [(0.0, 0), (1.0, 1)])
def test_random_loading_boundary_probs(p: float, expected: int) -> None:
    m = random_loading([5, 6], p, rng=np.random.default_rng(0))
    assert m.dtype == np.uint8
    assert np.all(m == expected)


def test_random_loading_invalid_probability_raises() -> None:
    with pytest.raises(ValueError):
        random_loading([3, 4], -0.1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        random_loading([3, 4], 1.1, rng=np.random.default_rng(0))

@pytest.mark.slow
def test_random_loading_rate_is_reasonable_statistically() -> None:
    p = 0.37
    m = random_loading([500, 500], p, rng=np.random.default_rng(123))
    phat = m.mean()

    n = m.size
    sigma = np.sqrt(p * (1 - p) / n)
    assert abs(phat - p) <= 5 * sigma

# -----------------------------
# generate_random_* configs (old-vs-new where meaningful)
# -----------------------------

def test_generate_random_init_target_configs_matches_old_reference_seeded() -> None:
    rng_old = np.random.default_rng(2026)
    rng_new = np.random.default_rng(2026)

    old_init, old_targ = old_generate_random_init_target_configs_ref(
        n_shots=3, load_prob=0.6, max_sys_size=5, target_config=[Configurations.RANDOM], rng=rng_old
    )
    new_init, new_targ = generate_random_init_target_configs(
        n_shots=3, load_prob=0.6, max_sys_size=5, target_config=[Configurations.RANDOM], rng=rng_new
    )

    assert len(old_init) == len(new_init) == 3
    assert len(old_targ) == len(new_targ) == 3

    for o, n in zip(old_init, new_init):
        assert np.array_equal(o.astype(np.uint8), n)

    for o, n in zip(old_targ, new_targ):
        assert np.array_equal(o.astype(np.uint8), n)


def test_generate_random_init_configs_single_species_matches_old_reference_seeded() -> None:
    rng_old = np.random.default_rng(7)
    rng_new = np.random.default_rng(7)

    old = old_generate_random_init_configs_ref(4, 0.55, 6, n_species=1, rng=rng_old)
    new = generate_random_init_configs(4, 0.55, 6, n_species=1, rng=rng_new)

    assert len(old) == len(new) == 4
    for o, n in zip(old, new):
        assert np.array_equal(o.astype(np.uint8), n)
        assert n.dtype == np.uint8


def test_generate_random_init_configs_dual_species_matches_seeded_reference() -> None:
    """
    This compares against a seeded 'old semantics' reference implementation
    (with numpy RNG replacing python random.randint for determinism).
    """
    rng_old = np.random.default_rng(99)
    rng_new = np.random.default_rng(99)

    old = old_generate_random_init_configs_ref(3, 0.6, 5, n_species=2, rng=rng_old)
    new = generate_random_init_configs(3, 0.6, 5, n_species=2, rng=rng_new)

    assert len(old) == len(new) == 3
    for o, n in zip(old, new):
        assert np.array_equal(o.astype(np.uint8), n)
        assert n.dtype == np.uint8
        # No double occupancy after resolution
        assert np.all(np.sum(n, axis=-1) <= 1)


def test_generate_random_init_configs_invalid_n_species_raises() -> None:
    with pytest.raises(ValueError):
        generate_random_init_configs(1, 0.5, 4, n_species=3, rng=np.random.default_rng(0))


def test_generate_random_target_configs_matches_old_reference_seeded() -> None:
    rng_old = np.random.default_rng(314)
    rng_new = np.random.default_rng(314)

    old = old_generate_random_target_configs_ref(4, 0.4, [5, 7], rng=rng_old)
    new = generate_random_target_configs(4, 0.4, [5, 7], rng=rng_new)

    assert len(old) == len(new) == 4
    for o, n in zip(old, new):
        assert np.array_equal(o.astype(np.uint8), n)


# -----------------------------
# atom_loss / atom_loss_dual
# -----------------------------

def test_atom_loss_matches_old_reference_seeded_2d() -> None:
    matrix = np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    rng_old = np.random.default_rng(1234)
    rng_new = np.random.default_rng(1234)

    old_matrix, old_flag = old_atom_loss_ref(matrix, move_time=0.7, lifetime=30.0, rng=rng_old)
    new_matrix, new_flag = atom_loss(matrix, move_time=0.7, lifetime=30.0, rng=rng_new)

    # Value equality, dtype may differ because old reference uses float mask math.
    assert np.array_equal(old_matrix.astype(new_matrix.dtype), new_matrix)
    assert old_flag == new_flag


def test_atom_loss_broadcasts_same_site_mask_across_species() -> None:
    matrix = np.ones((8, 9, 2), dtype=np.uint8)
    new_matrix, loss_flag = atom_loss(matrix, move_time=1.0, lifetime=5.0, rng=np.random.default_rng(0))

    assert new_matrix.shape == matrix.shape
    assert np.array_equal(new_matrix[:, :, 0], new_matrix[:, :, 1])
    assert isinstance(loss_flag, bool)


def test_REGRESSION_atom_loss_dual_applies_loss_and_returns_modified_matrix() -> None:
    matrix = np.ones((6, 6, 2), dtype=np.uint8)
    new_matrix, loss_flag = atom_loss_dual(matrix, move_time=100.0, lifetime=1.0, rng=np.random.default_rng(0))

    # With huge move_time / short lifetime, almost surely all lost
    assert np.all(new_matrix == 0)
    assert loss_flag is True


@pytest.mark.parametrize("fn", [atom_loss, atom_loss_dual])
def test_atom_loss_negative_or_zero_lifetime_raises(fn) -> None:
    if fn is atom_loss:
        matrix = np.ones((3, 3), dtype=np.uint8)
    else:
        matrix = np.ones((3, 3, 2), dtype=np.uint8)

    with pytest.raises(ValueError):
        fn(matrix, move_time=1.0, lifetime=0.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        fn(matrix, move_time=1.0, lifetime=-1.0, rng=np.random.default_rng(0))

@pytest.mark.slow
def test_atom_loss_single_species_survival_rate_is_reasonable_statistically() -> None:
    """
    Statistical sanity check for single-species atom_loss.

    We start with all sites occupied so survival rate can be estimated directly.
    """
    rows, cols = 300, 300
    move_time = 0.7
    lifetime = 3.5
    p_survive = float(np.exp(-move_time / lifetime))

    matrix = np.ones((rows, cols), dtype=np.uint8)
    new_matrix, loss_flag = atom_loss(
        matrix, move_time=move_time, lifetime=lifetime, rng=np.random.default_rng(0)
    )

    # Since input is all ones, output mean is empirical survival probability
    phat = new_matrix.mean()
    n = rows * cols
    sigma = np.sqrt(p_survive * (1 - p_survive) / n)

    assert abs(phat - p_survive) <= 5 * sigma
    assert isinstance(loss_flag, bool)


@pytest.mark.slow
def test_atom_loss_dual_species_survival_rate_is_reasonable_statistically_and_channels_match() -> None:
    """
    Statistical sanity check for dual-species atom_loss_dual.

    The implementation applies the same 2D survival mask to both species channels,
    so channels should be identical after loss when starting from all ones.
    """
    rows, cols = 300, 300
    move_time = 1.2
    lifetime = 4.0
    p_survive = float(np.exp(-move_time / lifetime))

    matrix = np.ones((rows, cols, 2), dtype=np.uint8)
    new_matrix, loss_flag = atom_loss_dual(
        matrix, move_time=move_time, lifetime=lifetime, rng=np.random.default_rng(1)
    )

    assert new_matrix.shape == (rows, cols, 2)
    assert np.array_equal(new_matrix[:, :, 0], new_matrix[:, :, 1])

    # Either channel estimates the same site-level survival probability
    phat = new_matrix[:, :, 0].mean()
    n = rows * cols
    sigma = np.sqrt(p_survive * (1 - p_survive) / n)

    assert abs(phat - p_survive) <= 5 * sigma
    assert isinstance(loss_flag, bool)



# -----------------------------
# deterministic helper functions
# -----------------------------

def test_count_atoms_in_columns() -> None:
    matrix = np.array(
        [
            [1, 1, 1],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert count_atoms_in_columns(matrix) == [1, 3, 2]


def test_left_right_atom_in_row() -> None:
    row = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
    # Old semantics: direction=1 -> rightmost, direction=-1 -> leftmost
    assert left_right_atom_in_row(row, direction=1) == 3
    assert left_right_atom_in_row(row, direction=-1) == 1
    assert left_right_atom_in_row(np.zeros(5, dtype=np.uint8), direction=1) is None


def test_top_bot_atom_in_col_and_find_lowest() -> None:
    col = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
    assert top_bot_atom_in_col(col, direction=1) == 3
    assert top_bot_atom_in_col(col, direction=-1) == 1
    assert find_lowest_atom_in_col(col) == 3
    assert find_lowest_atom_in_col(np.zeros(5, dtype=np.uint8)) is None