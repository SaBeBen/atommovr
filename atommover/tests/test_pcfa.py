import numpy as np
import pytest

from atommover.utils.AtomArray import AtomArray
from atommover.utils.core import Configurations
from atommover.algorithms.Algorithm_class import Algorithm
from atommover.algorithms.single_species import PCFA, Hungarian, Tetris
import os
from atommover.utils.imaging.animation import visualize_move_batches


def apply_moves(arr: AtomArray, move_batches):
    t_total, _ = arr.evaluate_moves(move_batches)
    return t_total


def test_pcfa_axis_aligned_moves_and_row_compression():
    # 1 row, 6 cols; atoms at sparse columns should compress into a contiguous block
    arr = AtomArray([1, 6], n_species=1)
    arr.matrix[:, :, 0] = np.array([[1, 0, 1, 0, 0, 1]])
    # Target size L=3 centered in the row
    L = 3
    target = np.zeros((1, 6), dtype=int)
    # Only the size (sum) matters for PCFA; shape's ones count -> L
    c0 = (6 - L) // 2
    target[0, c0:c0+L] = 1
    arr.target = target.reshape(1, 6, 1)

    algo = PCFA()
    _, move_batches, _ = algo.get_moves(arr, do_ejection=False)
    os.makedirs('output', exist_ok=True)

    visualize_move_batches(arr, move_batches, save_path='figs/pcfa_test/pcfa_row_compress_moves.png')

    # Check axis-aligned property of all moves
    for batch in move_batches:
        for mv in batch:
            assert (mv.from_row == mv.to_row) or (mv.from_col == mv.to_col), "Non axis-aligned move created"

    # Apply moves and verify compression achieved in-row (center 3 columns filled)
    apply_moves(arr, move_batches)
    c0 = (arr.shape[1] - L) // 2
    assert np.array_equal(arr.matrix[0, c0:c0+L, 0].flatten(), np.ones(L)), "Row compression failed"


def test_pcfa_fill_and_optional_ejection():
    # 4x8 array, target LxL = 4x4 in top-left
    shape = [4, 8]
    arr = AtomArray(shape, n_species=1)

    # Construct initial: ensure enough atoms exist to fill LxL.
    # Start with some atoms inside LxL and fill all columns >= L as donors.
    init = np.zeros(shape, dtype=int)
    init[0, 0] = 1
    init[0, 2] = 1
    init[1, 1] = 1
    init[2, 3] = 1
    # Make every site beyond L a donor atom to guarantee sufficiency
    L = 4
    init[:, L:] = 1

    arr.matrix[:, :, 0] = init

    # Target is 4x4 fully filled centered
    target = np.zeros(shape, dtype=int)
    # Only the sum matters; mark any L*L ones
    r0 = (shape[0] - L) // 2
    c0 = (shape[1] - L) // 2
    target[r0:r0+L, c0:c0+L] = 1
    arr.target = target.reshape(shape[0], shape[1], 1)

    # 1) No ejection: should fill LxL but may leave extras beyond L
    algo = PCFA()
    _, move_batches, _ = algo.get_moves(arr, do_ejection=False)

    visualize_move_batches(arr, move_batches, save_path='figs/pcfa_test/pcfa_fill_batches.png')

    apply_moves(arr, move_batches)

    # Check centered target region filled
    r0 = (shape[0] - L) // 2
    c0 = (shape[1] - L) // 2
    assert np.array_equal(arr.matrix[r0:r0+L, c0:c0+L, 0], np.ones((L, L))), "PCFA failed to fill target region without ejection"
    # Allow remaining atoms beyond L (no ejection)
    assert np.sum(arr.matrix[:, L:, 0]) >= 0

    # 2) With ejection: extras should be removed outside centered block
    arr2 = AtomArray(shape, n_species=1)
    arr2.matrix[:, :, 0] = init.copy()
    arr2.target = target.reshape(shape[0], shape[1], 1)

    _, move_batches2, _ = algo.get_moves(arr2, do_ejection=True)
    visualize_move_batches(arr2, move_batches2, save_path='figs/pcfa_test/pcfa_eject_batches.png')

    apply_moves(arr2, move_batches2)

    assert np.array_equal(arr2.matrix[r0:r0+L, c0:c0+L, 0], np.ones((L, L))), "PCFA failed to fill target region with ejection"
    # Outside target columns to the right should be empty
    assert np.sum(arr2.matrix[:, c0+L:, 0]) == 0, "Extras not ejected when do_ejection=True"


def test_pcfa_vs_hungarian_smoke():
    shape = [5, 10]
    L = 5
    arr = AtomArray(shape, n_species=1)

    # random-ish initial with sufficient atoms
    init = np.zeros(shape, dtype=int)
    rng = np.random.default_rng(0)
    filled_positions = rng.choice(shape[0]*shape[1], size=L*L+3, replace=False)
    for idx in filled_positions:
        r = idx // shape[1]
        c = idx % shape[1]
        init[r, c] = 1
    arr.matrix[:, :, 0] = init

    target = np.zeros(shape, dtype=int)
    r0 = (shape[0] - L) // 2
    c0 = (shape[1] - L) // 2
    target[r0:r0+L, c0:c0+L] = 1
    arr.target = target.reshape(shape[0], shape[1], 1)

    # PCFA run
    pcfa = PCFA()
    _, pcfa_batches, _ = pcfa.get_moves(arr, do_ejection=True)
    try:
        visualize_move_batches(arr, pcfa_batches, save_path='figs/pcfa_test/pcfa_smoke_batches.png')
    except Exception:
        pass
    apply_moves(arr, pcfa_batches)

    # Assert success inside centered target
    assert np.array_equal(arr.matrix[r0:r0+L, c0:c0+L, 0], np.ones((L, L)))

    # Compare with Hungarian baseline (smoke)
    arr_h = AtomArray(shape, n_species=1)
    arr_h.matrix[:, :, 0] = init.copy()
    arr_h.target = arr.target.copy()

    hung = Hungarian()
    _, h_moves, success = hung.get_moves(arr_h, do_ejection=True)
    assert success
    apply_moves(arr_h, h_moves)
    assert np.array_equal(arr_h.matrix[r0:r0+L, c0:c0+L, 0], np.ones((L, L)))

def test_tetris_fills_rectangular_target():
    """Tetris should fill an arbitrary rectangular target using axis-aligned moves."""
    shape = [6, 8]
    arr = AtomArray(shape, n_species=1)

    init = np.zeros(shape, dtype=int)
    # Populate rows with staggered donors; ensure multiple donors beyond target columns
    init[0, :6] = 1
    init[1, 2:] = 1
    init[2, 1:7] = 1
    init[3, 4:] = 1
    arr.matrix[:, :, 0] = init

    target = np.zeros(shape, dtype=int)
    target[1:5, 2:6] = 1  # 4x4 block away from the edges
    arr.target = target.reshape(shape[0], shape[1], 1)

    algo = Tetris()
    _, move_batches, success = algo.get_moves(arr, do_ejection=False)

    assert success, "Tetris failed to find plan despite ample donors"
    for batch in move_batches:
        for mv in batch:
            assert (mv.from_row == mv.to_row) or (mv.from_col == mv.to_col), "Non axis-aligned Tetris move"

    apply_moves(arr, move_batches)
    filled = arr.matrix[:, :, 0]
    assert np.array_equal(filled[target == 1], np.ones(int(np.sum(target)))), "Target mask not fully populated"


def test_tetris_ejection_clears_excess_atoms():
    shape = [5, 7]
    arr = AtomArray(shape, n_species=1)

    init = np.zeros(shape, dtype=int)
    init[0:3, :] = 1  # donors everywhere in top rows
    init[3:, 4:] = 1  # extra donors outside target columns
    arr.matrix[:, :, 0] = init

    target = np.zeros(shape, dtype=int)
    target[:3, :4] = 1
    arr.target = target.reshape(shape[0], shape[1], 1)

    algo = Tetris()
    _, move_batches, success = algo.get_moves(arr, do_ejection=True)

    assert success, "Tetris should succeed before ejection"
    apply_moves(arr, move_batches)

    filled = arr.matrix[:, :, 0]
    assert np.array_equal(filled[:3, :4], np.ones((3, 4))), "Target subarray not filled"
    assert np.sum(filled * (1 - target)) == 0, "Excess atoms remained after ejection"


def test_tetris_fails_when_atoms_insufficient():
    shape = [4, 4]
    arr = AtomArray(shape, n_species=1)

    init = np.zeros(shape, dtype=int)
    init[0, :2] = 1  # only two atoms available
    arr.matrix[:, :, 0] = init

    target = np.zeros(shape, dtype=int)
    target[0:2, 0:2] = 1  # need four atoms
    arr.target = target.reshape(shape[0], shape[1], 1)

    algo = Tetris()
    _, move_batches, success = algo.get_moves(arr, do_ejection=False)

    assert not success, "Algorithm should report failure when donors are insufficient"
    assert move_batches == [], "No moves should be scheduled when impossible"


def test_pcfa_large_dense_donors():
    """
    Larger system with many donors outside the LxL region to stress PCFA.
    Ensures final LxL is filled and extras ejected when do_ejection=True.
    """
    np.random.seed(42)
    shape = [8, 12]
    L = 8

    arr = AtomArray(shape, n_species=1)
    # Heavier loading to ensure plenty of donors, then guarantee donors on right side
    arr.params.loading_prob = 0.75
    arr.load_tweezers()
    # Force donors on columns >= L to guarantee sufficiency
    arr.matrix[:, L:, 0] = 1

    target = np.zeros(shape, dtype=int)
    target.flat[:L*L] = 1
    arr.target = target.reshape(shape[0], shape[1], 1)

    algo = PCFA()
    _, move_batches, _ = algo.get_moves(arr, do_ejection=True)
    visualize_move_batches(arr, move_batches, save_path='figs/pcfa_test/pcfa_large_dense.png')

    apply_moves(arr, move_batches)

    # Check centered target region fully filled and no atoms outside when ejection=True
    r0 = (shape[0] - L) // 2
    c0 = (shape[1] - L) // 2
    assert np.array_equal(arr.matrix[r0:r0+L, c0:c0+L, 0], np.ones((L, L)))
    assert np.sum(arr.matrix[:, c0+L:, 0]) == 0


def test_pcfa_sparse_complex():
    """
    Complex sparse initial loading with holes inside target and donors outside.
    Validate axis-aligned moves, unique destinations per batch, and final fill.
    """
    np.random.seed(7)
    shape = [6, 14]
    L = 6

    arr = AtomArray(shape, n_species=1)
    arr.params.loading_prob = 0.5
    arr.load_tweezers()

    # Carve intentional holes inside target and ensure donors outside
    holes = [(0, 0), (1, 4), (3, 2), (4, 5), (5, 1)]
    for r, c in holes:
        if r < L and c < L:
            arr.matrix[r, c, 0] = 0
    # Populate a donor band on the right and bottom
    arr.matrix[:, L:, 0] = 1
    arr.matrix[L:, :, 0] = 1

    target = np.zeros(shape, dtype=int)
    target.flat[:L*L] = 1
    arr.target = target.reshape(shape[0], shape[1], 1)

    algo = PCFA()
    # Without ejection first to check fill happens even with extras
    _, move_batches_no_eject, _ = algo.get_moves(arr, do_ejection=False)
    visualize_move_batches(arr, move_batches_no_eject, save_path='figs/pcfa_test/pcfa_sparse_complex_no_eject.png')
    apply_moves(arr, move_batches_no_eject)

    r0 = (shape[0] - L) // 2
    c0 = (shape[1] - L) // 2
    assert np.array_equal(arr.matrix[r0:r0+L, c0:c0+L, 0], np.ones((L, L)))

    # Now run with ejection on the same initial state to validate cleanup
    arr2 = AtomArray(shape, n_species=1)
    arr2.matrix[:, :, 0] = 0
    # Reconstruct the same initial by reseeding and repeating the steps deterministically
    np.random.seed(7)
    arr2.params.loading_prob = 0.5
    arr2.load_tweezers()
    for r, c in holes:
        if r < L and c < L:
            arr2.matrix[r, c, 0] = 0
    arr2.matrix[:, L:, 0] = 1
    arr2.matrix[L:, :, 0] = 1
    arr2.target = target.reshape(shape[0], shape[1], 1)

    _, move_batches, _ = algo.get_moves(arr2, do_ejection=True)
    visualize_move_batches(arr2, move_batches, save_path='figs/pcfa_test/pcfa_sparse_complex.png')

    # Axis-aligned and unique-destination checks per batch
    for batch in move_batches:
        dests = set()
        for mv in batch:
            assert (mv.from_row == mv.to_row) or (mv.from_col == mv.to_col), "Non axis-aligned move created"
            if 0 <= mv.to_row < shape[0] and 0 <= mv.to_col < shape[1]:
                dst = (mv.to_row, mv.to_col)
                assert dst not in dests, "Two moves target the same destination in one batch"
                dests.add(dst)

    apply_moves(arr2, move_batches)
    assert np.array_equal(arr2.matrix[r0:r0+L, c0:c0+L, 0], np.ones((L, L)))
    assert np.sum(arr2.matrix[:, c0+L:, 0]) == 0


def test_pcfa_ejection_clears_rectangular_padding():
    """Ensure do_ejection=True removes atoms above/below the centered block on wide arrays."""
    shape = [8, 16]
    L = 5
    arr = AtomArray(shape, n_species=1)
    # Load every site so donors exist both vertically and horizontally
    arr.matrix[:, :, 0] = 1

    target = np.zeros(shape, dtype=int)
    r0 = (shape[0] - L) // 2
    c0 = (shape[1] - L) // 2
    target[r0:r0+L, c0:c0+L] = 1
    arr.target = target.reshape(shape[0], shape[1], 1)

    algo = PCFA()
    _, move_batches, _ = algo.get_moves(arr, do_ejection=True)
    apply_moves(arr, move_batches)

    # No atoms should remain outside the centered LxL block
    assert np.sum(arr.matrix[:, :, 0]) == L * L
    assert np.array_equal(arr.matrix[:, :, 0], arr.target[:, :, 0])
    assert Algorithm.get_success_flag(arr.matrix, arr.target, do_ejection=True)
