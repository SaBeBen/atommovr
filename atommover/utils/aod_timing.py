"""
Authors: Claude AI, ChatGPT, Nikhil Harle

Description: AOD timing analysis for physically accurate rearrangement simulations.

This module analyzes transitions between AOD command states to determine
which atoms need pickup, putdown, acceleration, and deceleration operations.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray


# -----------------------------------------------------------------------------
# Transition tables (kept as-is, but typed + dtyped explicitly)
# -----------------------------------------------------------------------------

transition_table_pickup_accel: NDArray[np.bool_] = np.array(
    [
        # curr: 0      1      2      3
        [[0, 0], [1, 0], [1, 1], [1, 1]],  # prev=0 (SLM)
        [[0, 0], [0, 0], [0, 1], [0, 1]],  # prev=1 (hold)
        [[0, 0], [0, 0], [0, 0], [0, 1]],  # prev=2 (move +)
        [[0, 0], [0, 0], [0, 1], [0, 0]],  # prev=3 (move -)
    ],
    dtype=np.bool_,
)

transition_table_decel_putdown: NDArray[np.bool_] = np.array(
    [
        # next: 0      1      2      3
        [[0, 0], [0, 0], [0, 0], [0, 0]],  # curr=0 (off)
        [[0, 1], [0, 0], [0, 0], [0, 0]],  # curr=1 (hold -> off implies putdown)
        [[1, 1], [1, 0], [0, 0], [1, 0]],  # curr=2 (moving -> off implies decel+putdown)
        [[1, 1], [1, 0], [1, 0], [0, 0]],  # curr=3 (moving -> off implies decel+putdown)
    ],
    dtype=np.bool_,
)

# Single-tone remaining currents table (prev=0)
# flags: decel, putdown, pickup, accel
single_tone_transition_table: NDArray[np.bool_] = np.array(
    [[[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]],
    dtype=np.bool_,
)


# -----------------------------------------------------------------------------
# Tiny internal helpers
# -----------------------------------------------------------------------------
# NOTE: abbreviated docstrings (per your rule) for trivial helpers.

def _as_cmd_array(cmds: Sequence[int] | NDArray[np.integer]) -> NDArray[np.int8]:
    """
    Convert an AOD command sequence into a compact int8 NumPy array.
    """
    arr: NDArray[np.int8] = np.asarray(cmds, dtype=np.int8)
    if arr.ndim != 1:
        raise ValueError(f"AOD command arrays must be 1D, got {arr.ndim}D.")
    return arr


def _is_moving_cmd(cmds: NDArray[np.int8]) -> NDArray[np.bool_]:
    """
    Return a boolean mask for commands in {2, 3}.
    """
    return (cmds == np.int8(2)) | (cmds == np.int8(3))


# -----------------------------------------------------------------------------
# Core timing inference (vectorized, no Python index lists)
# -----------------------------------------------------------------------------

def _get_pickup_accel_flags(
    prev_cmds: NDArray[np.int8],
    curr_cmds: NDArray[np.int8],
) -> Tuple[bool, bool, NDArray[np.intp], NDArray[np.intp]]:
    """
    Infer pickup/accel requirements for one axis using only (prev, curr) tone commands.

    Why this exists
    ---------------
    The simulation cannot reliably condition timing decisions on real-time tweezer
    occupancy. Timing is therefore inferred *purely* from AOD command evolution.

    Key rule: OR over multiple preimages (tone convergence)
    -------------------------------------------------------
    Each active predecessor tone has a unique destination index in the next command
    list, but multiple predecessor tones may map to the same destination index
    (non-injective forward map).

    When multiple predecessors map to the same current index j, we apply a conservative
    "don't heat the atom" closure:

        accel_required_at_j = OR_{i: dest(i)=j, prev[i]!=0} accel_required_if_predecessor_is_i

    Returns
    -------
    needs_pickup, needs_accel
        Whether *any* pickup/accel is needed on this axis this round.
    pickup_inds, accel_inds
        Indices (in curr_cmds index-space) whose source sites should be treated as
        pickup/accel eligible on this axis.
    """
    try:
        n: int = int(curr_cmds.size)
    except AttributeError:
        if isinstance(curr_cmds, list):
            curr_cmds = _as_cmd_array(curr_cmds)
        if isinstance(prev_cmds, list):
            prev_cmds = _as_cmd_array(prev_cmds)
        n: int = int(curr_cmds.size)

    # Forward map: active prev tones -> destination indices in curr index-space.
    prev_inds: NDArray[np.intp] = np.flatnonzero(prev_cmds) #np.nonzero(prev_cmds != np.int8(0))[0]
    if prev_inds.size == 0:
        # No predecessor tones: treat currents as "prev=0 -> curr" transitions.
        remaining_mask: NDArray[np.bool_] = curr_cmds != np.int8(0)
        remaining_inds: NDArray[np.intp] = remaining_mask.nonzero()[0]
        if remaining_inds.size == 0:
            return False, False, remaining_inds, remaining_inds

        remaining_vals: NDArray[np.int8] = curr_cmds[remaining_inds]
        st_flags: NDArray[np.bool_] = single_tone_transition_table[0, remaining_vals]
        pickup_flags: NDArray[np.bool_] = st_flags[:, 2]
        accel_flags: NDArray[np.bool_] = st_flags[:, 3]
        pickup_inds: NDArray[np.intp] = remaining_inds[pickup_flags]
        accel_inds: NDArray[np.intp] = remaining_inds[accel_flags]
        return bool(pickup_inds.size), bool(accel_inds.size), pickup_inds, accel_inds

    prev_vals: NDArray[np.int8] = prev_cmds[prev_inds]

    dest_inds: NDArray[np.intp] = prev_inds.copy()
    dest_inds[prev_vals == np.int8(2)] += 1
    dest_inds[prev_vals == np.int8(3)] -= 1

    in_bounds: NDArray[np.bool_] = (dest_inds >= 0) & (dest_inds < n)

    # Allocate tone-level flags in curr index-space.
    pickup_tone: NDArray[np.bool_] = np.zeros(n, dtype=np.bool_)
    accel_tone: NDArray[np.bool_] = np.zeros(n, dtype=np.bool_)
    checked: NDArray[np.bool_] = np.zeros(n, dtype=np.bool_)

    if in_bounds.any():
        valid_prev_vals: NDArray[np.int8] = prev_vals[in_bounds]
        valid_dest_inds: NDArray[np.intp] = dest_inds[in_bounds]
        valid_curr_vals: NDArray[np.int8] = curr_cmds[valid_dest_inds]

        flags: NDArray[np.bool_] = transition_table_pickup_accel[valid_prev_vals, valid_curr_vals]
        pickup_flags: NDArray[np.bool_] = flags[:, 0]
        accel_flags: NDArray[np.bool_] = flags[:, 1]

        # OR-reduce onto destination indices (handles >1 preimage).
        np.logical_or.at(pickup_tone, valid_dest_inds, pickup_flags)
        np.logical_or.at(accel_tone, valid_dest_inds, accel_flags)

        checked[valid_dest_inds] = True

    # Step 2: currents with no predecessor mapping are treated as (prev=0 -> curr).
    remaining_mask = (curr_cmds != np.int8(0)) & (~checked)
    remaining_inds = np.flatnonzero(remaining_mask)#[0]
    if remaining_inds.size != 0:
        remaining_vals = curr_cmds[remaining_inds]
        st_flags = single_tone_transition_table[0, remaining_vals]
        pickup_new = st_flags[:, 2]
        accel_new = st_flags[:, 3]
        pickup_tone[remaining_inds] |= pickup_new
        accel_tone[remaining_inds] |= accel_new

    pickup_inds = pickup_tone.nonzero()[0]
    accel_inds = accel_tone.nonzero()[0]
    return bool(pickup_inds.size), bool(accel_inds.size), pickup_inds, accel_inds


def _get_decel_putdown_flags(
    curr_cmds: NDArray[np.int8],
    next_cmds: NDArray[np.int8],
) -> tuple[bool, bool, NDArray[np.int8], NDArray[np.int8]]:
    """
    Get (decel, putdown) flags and the tone indices that trigger them for one axis.

    Design intent
    -------------
    This helper is *occupancy agnostic*: it infers end-of-round “slow down” operations
    using only how AOD tone commands evolve between (curr_cmds, next_cmds).

    Mapping convention
    ------------------
    A nonzero command at index i in curr_cmds maps to a destination index in the next-step
    index space:
        - cmd==1 maps to i
        - cmd==2 maps to i+1
        - cmd==3 maps to i-1

    The (decel, putdown) decision for the current tone is then read from
    transition_table_decel_putdown[curr_val, next_val_at_destination].

    Out-of-bounds convention (matches current tests)
    -----------------------------------------------
    - cmd==2 stepping out of bounds at the right edge is treated as requiring PUTDOWN only.
    - cmd==3 stepping out of bounds at the left edge is ignored (no decel/putdown bookkeeping).

    Returns
    -------
    needs_decel, needs_putdown : bool
        Whether any tone requires decel/putdown on this axis.
    decel_inds, putdown_inds : list[int]
        Indices (in curr_cmds index space) triggering decel/putdown.
    """

    try:
        n: int = int(curr_cmds.size)
    except AttributeError:
        if isinstance(curr_cmds, list) or curr_cmds.dtype != np.int8:
            curr_cmds = _as_cmd_array(curr_cmds)
        if isinstance(next_cmds, list) or next_cmds.dtype != np.int8:
            next_cmds = _as_cmd_array(next_cmds)
        n: int = int(curr_cmds.size)

    if int(next_cmds.size) != n:
        raise ValueError(
            f"curr_cmds and next_cmds must have same length, got {n} and {int(next_cmds.size)}."
        )

    curr_inds = np.flatnonzero(curr_cmds) #np.nonzero(curr_cmds != 0)[0]
    if curr_inds.size == 0:
        return False, False, np.zeros_like(curr_cmds), np.zeros_like(curr_cmds)

    curr_vals = curr_cmds[curr_inds]

    dest_inds = curr_inds.copy()
    dest_inds[curr_vals == 2] += 1
    dest_inds[curr_vals == 3] -= 1

    in_bounds = (dest_inds >= 0) & (dest_inds < n)

    # OOB right-ejection rule: cmd==2 is ignored

    # OOB left-ejection rule: cmd==3 is ignored

    valid_curr_inds = curr_inds[in_bounds]
    valid_curr_vals = curr_vals[in_bounds]
    valid_next_vals = next_cmds[dest_inds[in_bounds]]

    flags = transition_table_decel_putdown[valid_curr_vals, valid_next_vals]
    decel_flags = flags[:, 0]
    putdown_flags = flags[:, 1]

    decel_inds_arr = valid_curr_inds[decel_flags]
    putdown_inds_arr = valid_curr_inds[putdown_flags]

    needs_decel = bool(decel_inds_arr.size)
    needs_putdown = bool(putdown_inds_arr.size)

    return needs_decel, needs_putdown, decel_inds_arr, putdown_inds_arr


def _classify_new_and_continuing_tones(
    prev_cmds: NDArray[np.int8],
    curr_cmds: NDArray[np.int8],
) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    Classify current-tone indices as either "new/moving" or "continuing".

    Why this exists
    ---------------
    Cross-axis acceleration detection needs a robust notion of “new tone sites”
    that respects AOD ramp semantics (cmd 2/3 imply shifts in index-space).
    This classifier is intentionally occupancy-agnostic: it only interprets tone
    evolution.

    Returns
    -------
    new_or_moving_inds
        Indices in curr_cmds that are nonzero and are not explainable as a pure
        continuation from prev.
    continuing_inds
        Indices in curr_cmds that are nonzero and *are* explainable as continuation.
    """

    try:
        n: int = int(curr_cmds.size)
    except AttributeError:
        if isinstance(curr_cmds, list):
            curr_cmds = _as_cmd_array(curr_cmds)
        if isinstance(prev_cmds, list):
            prev_cmds = _as_cmd_array(prev_cmds)
        n: int = int(curr_cmds.size)
    curr_nonzero: NDArray[np.bool_] = curr_cmds != np.int8(0)

    cont: NDArray[np.bool_] = np.zeros(n, dtype=np.bool_)

    # Same index continuation: prev tone active and curr tone active.
    cont |= (prev_cmds != np.int8(0)) & curr_nonzero

    # Arrived from left: prev[i]==2 arrives at i+1.
    cont[1:] |= (prev_cmds[:-1] == np.int8(2)) & curr_nonzero[1:]

    # Arrived from right: prev[i]==3 arrives at i-1.
    cont[:-1] |= (prev_cmds[1:] == np.int8(3)) & curr_nonzero[:-1]

    new_mask: NDArray[np.bool_] = curr_nonzero & (~cont)
    cont_mask: NDArray[np.bool_] = curr_nonzero & cont

    return new_mask.nonzero()[0], cont_mask.nonzero()[0]


def _find_cross_axis_accel_tones(
    prev_x: NDArray[np.int8],
    curr_x: NDArray[np.int8],
    prev_y: NDArray[np.int8],
    curr_y: NDArray[np.int8],
) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    Detect cross-axis accel requirements that aren't visible to per-axis transition logic.

    Why this exists
    ---------------
    In crossed-AOD tweezer systems, motion on one axis can require a smooth ramp
    (“accel”) on the orthogonal axis even if that axis’ tones appear static by
    a naive per-axis comparison.

    Operational rule (as in the prior implementation)
    -------------------------------------------------
    - If Y has any motion (cmd in {2,3}), then *new static* tones on X should be
      treated as accel-eligible (to avoid heating when the orthogonal axis is moving).
    - Symmetrically, if X has any motion, then new static tones on Y are accel-eligible.

    Returns
    -------
    accel_x_inds, accel_y_inds
        Indices (in curr_x / curr_y index-space) that should be marked accel-eligible
        due to cross-axis coupling logic.
    """
    try:
        n = int(curr_x.size)
    except AttributeError:
        if isinstance(curr_x, list):
            curr_x = _as_cmd_array(curr_x)
        if isinstance(prev_x, list):
            prev_x = _as_cmd_array(prev_x)
        if isinstance(curr_y, list):
            curr_y = _as_cmd_array(curr_y)
        if isinstance(prev_y, list):
            prev_y = _as_cmd_array(prev_y)

    new_x, _cont_x = _classify_new_and_continuing_tones(prev_x, curr_x)
    new_y, _cont_y = _classify_new_and_continuing_tones(prev_y, curr_y)

    # “New static” tones: new indices whose cmd==1.
    new_x_static: NDArray[np.intp] = new_x[curr_x[new_x] == np.int8(1)]
    new_y_static: NDArray[np.intp] = new_y[curr_y[new_y] == np.int8(1)]

    y_has_motion: bool = bool(_is_moving_cmd(curr_y).any())
    x_has_motion: bool = bool(_is_moving_cmd(curr_x).any())

    accel_x: NDArray[np.intp] = new_x_static if y_has_motion else np.zeros(0, dtype=np.intp)
    accel_y: NDArray[np.intp] = new_y_static if x_has_motion else np.zeros(0, dtype=np.intp)
    return accel_x, accel_y


# -----------------------------------------------------------------------------
# Public-ish mask builders used by AtomArray.move_atoms
# -----------------------------------------------------------------------------

def _detect_pickup_and_accel_masks(
    prev_h: Optional[NDArray[np.int8]],
    prev_v: Optional[NDArray[np.int8]],
    curr_h: NDArray[np.int8],
    curr_v: NDArray[np.int8],
    curr_move_set: Sequence[object],
    source_cols: NDArray[np.int_] | None = None,
    source_rows: NDArray[np.int_] | None = None,
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Build per-move pickup/accel eligibility masks from tone evolution.
    TODO: add check that raises an error if moves in current_move_set do not 
    correspond to current AOD commands (curr_h and curr_v).

    Why this exists
    ---------------
    The simulation’s time accounting and error processes depend on *which move
    sources* undergo pickup or acceleration at the start of a round. This
    function converts axis-level timing inference into per-move boolean masks
    aligned with `curr_move_set`.
    """
    n_moves: int = len(curr_move_set)
    if n_moves == 0:
        return np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.bool_)
    if source_cols is None:
        source_cols = np.asarray([m.from_col for m in curr_move_set], dtype=np.intp).reshape(-1)
    if source_rows is None:
        source_rows = np.asarray([m.from_row for m in curr_move_set], dtype=np.intp).reshape(-1)
    if int(source_cols.size) != n_moves or int(source_rows.size) != n_moves:
        raise ValueError("source_cols/source_rows must be 1D and aligned with curr_move_set.")

    # First-round special case: everything is "picked up".
    if prev_h is None or prev_v is None:
        pickup_mask: NDArray[np.bool_] = np.ones(n_moves, dtype=np.bool_)
        accel_mask: NDArray[np.bool_] = _is_moving_cmd(curr_h[source_cols]) | _is_moving_cmd(
            curr_v[source_rows]
        )
        return pickup_mask, accel_mask

    _h_needs_pickup, _h_needs_accel, h_pickup_inds, h_accel_inds = _get_pickup_accel_flags(
        prev_h, curr_h
    )
    _v_needs_pickup, _v_needs_accel, v_pickup_inds, v_accel_inds = _get_pickup_accel_flags(
        prev_v, curr_v
    )

    # Cross-axis accel supplement.
    accel_x_extra, accel_y_extra = _find_cross_axis_accel_tones(
        prev_h, curr_h, prev_v, curr_v
    )

    # Axis-level tone masks (index-space masks).
    h_pickup_tone: NDArray[np.bool_] = np.zeros(curr_h.size, dtype=np.bool_)
    h_accel_tone: NDArray[np.bool_] =  np.zeros(curr_h.size, dtype=np.bool_)
    v_pickup_tone: NDArray[np.bool_] = np.zeros(curr_v.size, dtype=np.bool_)
    v_accel_tone: NDArray[np.bool_] =  np.zeros(curr_v.size, dtype=np.bool_)

    if h_pickup_inds.size:
        h_pickup_tone[h_pickup_inds] = True
    if h_accel_inds.size:
        h_accel_tone[h_accel_inds] = True
    if v_pickup_inds.size:
        v_pickup_tone[v_pickup_inds] = True
    if v_accel_inds.size:
        v_accel_tone[v_accel_inds] = True

    if accel_x_extra.size:
        h_accel_tone[accel_x_extra] = True
    if accel_y_extra.size:
        v_accel_tone[accel_y_extra] = True

    pickup_mask = h_pickup_tone[source_cols] | v_pickup_tone[source_rows]
    accel_mask = h_accel_tone[source_cols] | v_accel_tone[source_rows]
    # pickup_mask = np.asarray(pickup_mask, dtype=np.bool_).reshape(n_moves)
    # accel_mask = np.asarray(accel_mask, dtype=np.bool_).reshape(n_moves)
    return pickup_mask, accel_mask

def _detect_decel_and_putdown_masks(
    curr_h: NDArray[np.int8],
    curr_v: NDArray[np.int8],
    next_h: NDArray[np.int8],
    next_v: NDArray[np.int8],
    curr_move_set: Sequence[object],
    source_cols: NDArray[np.int_] | None = None,
    source_rows: NDArray[np.int_] | None = None,
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Build per-move decel/putdown eligibility masks from tone evolution.
    TODO: add check that raises an error if moves in current_move_set do not 
    correspond to current AOD commands (curr_h and curr_v).

    Why this exists
    ---------------
    End-of-round processes (deceleration and putdown) are also driven by tone
    evolution. This converts axis-level inference into per-move boolean masks.
    """
    n_moves: int = len(curr_move_set)
    if n_moves == 0:
        return np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.bool_)
    if source_cols is None:
        source_cols = np.asarray([m.from_col for m in curr_move_set], dtype=np.intp).reshape(-1)
    if source_rows is None:
        source_rows = np.asarray([m.from_row for m in curr_move_set], dtype=np.intp).reshape(-1)
    if int(source_cols.size) != n_moves or int(source_rows.size) != n_moves:
        raise ValueError("source_cols/source_rows must be 1D and aligned with curr_move_set.")

    _h_needs_decel, _h_needs_putdown, h_decel_inds, h_putdown_inds = _get_decel_putdown_flags(
        curr_h, next_h
    )
    _v_needs_decel, _v_needs_putdown, v_decel_inds, v_putdown_inds = _get_decel_putdown_flags(
        curr_v, next_v
    )

    h_decel_tone: NDArray[np.bool_] =   np.zeros(curr_h.size, dtype=np.bool_)
    h_putdown_tone: NDArray[np.bool_] = np.zeros(curr_h.size, dtype=np.bool_)
    v_decel_tone: NDArray[np.bool_] =   np.zeros(curr_v.size, dtype=np.bool_)
    v_putdown_tone: NDArray[np.bool_] = np.zeros(curr_v.size, dtype=np.bool_)

    if h_decel_inds.size:
        h_decel_tone[h_decel_inds] = True
    if h_putdown_inds.size:
        h_putdown_tone[h_putdown_inds] = True
    if v_decel_inds.size:
        v_decel_tone[v_decel_inds] = True
    if v_putdown_inds.size:
        v_putdown_tone[v_putdown_inds] = True

    decel_mask = h_decel_tone[source_cols] | v_decel_tone[source_rows]
    putdown_mask = h_putdown_tone[source_cols] | v_putdown_tone[source_rows]
    # decel_mask = np.asarray(decel_mask, dtype=np.bool_).reshape(n_moves)
    # putdown_mask = np.asarray(putdown_mask, dtype=np.bool_).reshape(n_moves)
    return decel_mask, putdown_mask

    


# -----------------------------------------------------------------------------
# Crossed-tone detection helpers (kept close to your existing behavior)
# -----------------------------------------------------------------------------
def _has_colliding_tones_axis(a: NDArray[np.int_]) -> bool:
    """
    Quick pre-check for whether an axis command list contains any “crossing” pattern.

    Why this exists
    ---------------
    In most realistic runs, crossed tones should be extremely rare. The full crossed-tone
    analysis builds masks and index lists; this helper does a minimal set of vector checks
    so the common case (“no crossing”) is fast.

    Notes
    -----
    This matches exactly the same local patterns handled by
    `_remove_collisions_axis_np_with_cross_info`: [2,1], [1,3], [2,3], and [2,x,3] with x at distance 1.
    """
    n = int(a.size)
    if n < 2:
        return False

    if ((a[:-1] == 2) & (a[1:] == 1)).any():
        return True
    if ((a[:-1] == 1) & (a[1:] == 3)).any():
        return True
    if ((a[:-1] == 2) & (a[1:] == 3)).any():
        return True
    if n >= 3 and ((a[:-2] == 2) & (a[2:] == 3)).any():
        return True
    return False


def _has_colliding_tones(
    vert: NDArray[np.int_],
    horiz: NDArray[np.int_],
) -> bool:
    """
    Axis-pair crossed-tone pre-check used to skip expensive crossed-tone bookkeeping.
    """
    return _has_colliding_tones_axis(np.concatenate([vert, np.zeros(2, dtype=np.int8), horiz], dtype = np.int8))

def _remove_collisions_axis_np(
    arr: NDArray[np.int_],
) -> NDArray[np.int_]:
    """
    Remove colliding tones on a single axis using numpy arrays.

    This exists to enforce the "one tone per site per axis" physical constraint in
    the simplified command model.
    """
    a: NDArray[np.int_] = np.asarray(arr)
    n: int = int(a.size)
    mask: NDArray[np.bool_] = np.zeros(n, dtype=np.bool_)

    # Patterns: [2,1], [1,3], [2,3]
    mask[:-1] |= (a[:-1] == 2) & (a[1:] == 1)
    mask[1:] |= (a[:-1] == 2) & (a[1:] == 1)

    mask[:-1] |= (a[:-1] == 1) & (a[1:] == 3)
    mask[1:] |= (a[:-1] == 1) & (a[1:] == 3)

    mask[:-1] |= (a[:-1] == 2) & (a[1:] == 3)
    mask[1:] |= (a[:-1] == 2) & (a[1:] == 3)

    # Pattern [2,x,3] (x can be any value)
    mask[:-2] |= (arr[:-2] == 2) & (arr[2:] == 3)
    mask[1:-1] |= (arr[:-2] == 2) & (arr[2:] == 3)
    mask[2:] |= (arr[:-2] == 2) & (arr[2:] == 3)

    return np.where(mask, 0, a)

def _classify_fatal_and_nonfatal_colliding_tones(
    arr: NDArray[np.int_],
    return_clean: bool = False,
) -> tuple[NDArray[np.int_] | None, NDArray[np.int_], NDArray[np.int_]]:
    """
    Classify a single-axis set of AOD commands into fatal (inevitable) and nonfatal
    (avoidable) with pickup error colliding-tone masks, and optionally return a 
    cleaned copy of the input.

    Collision rules
    ---------------
    Priority 1:
    - 2, x, 3:
        - middle entry is always fatal (inevitable)
        - if x is 0 or 1: both outer entries are nonfatal (avoidable)
        - if x is 2: left outer entry is nonfatal, right outer entry is fatal
        - if x is 3: left outer entry is fatal, right outer entry is nonfatal 

    Priority 2 (applied only to adjacent pairs not already covered by a
    priority-1 triad):
    - 2, 1 -> first nonfatal, second fatal
    - 1, 3 -> first fatal, second nonfatal
    - 2, 3 -> both fatal

    Parameters
    ----------
    arr : NDArray[np.int_]
        One-dimensional array of single-axis AOD commands.
    return_clean : bool, default=False
        Whether to return a cleaned copy of ``arr`` with all fatal and
        nonfatal crossed tones set to zero. If False, the first returned
        value is None.

    Returns
    -------
    arr_clean : NDArray[np.int_] | None
        Copy of ``arr`` with all classified crossed tones set to zero if
        ``return_clean`` is True; otherwise None.
    fatal : NDArray[np.bool_]
        Boolean mask marking entries classified as fatal (inevitable)
    nonfatal : NDArray[np.bool_]
        Boolean mask marking entries classified as nonfatal (avoidable with pickup error)

    Notes
    -----
    Priority-1 triads suppress priority-2 pair classification on any adjacent
    pair touching a triad-covered entry.
    """
    fatal: NDArray[np.bool_] = np.zeros_like(arr, dtype=bool)
    nonfatal: NDArray[np.bool_] = np.zeros_like(arr, dtype=bool)
    triad_covered: NDArray[np.bool_] = np.zeros_like(arr, dtype=bool)

    if arr.size >= 3:
        triad_2x3: NDArray[np.bool_] = (arr[:-2] == 2) & (arr[2:] == 3)
        mid_vals: NDArray[np.int_] = arr[1:-1]

        # Mark triad coverage so pair rules do not reclassify touched entries.
        triad_covered[:-2] |= triad_2x3
        triad_covered[1:-1] |= triad_2x3
        triad_covered[2:] |= triad_2x3

        # Middle entry of every triad is always fatal.
        fatal[1:-1] |= triad_2x3

        # Left outer entry:
        # nonfatal unless x == 3, in which case fatal.
        fatal[:-2] |= triad_2x3 & (mid_vals == 3)
        nonfatal[:-2] |= triad_2x3 & (mid_vals != 3)

        # Right outer entry:
        # nonfatal unless x == 2, in which case fatal.
        fatal[2:] |= triad_2x3 & (mid_vals == 2)
        nonfatal[2:] |= triad_2x3 & (mid_vals != 2)

    if arr.size >= 2:
        pair_valid: NDArray[np.bool_] = (~triad_covered[:-1]) & (~triad_covered[1:])
        left: NDArray[np.int_] = arr[:-1]
        right: NDArray[np.int_] = arr[1:]

        pair_21: NDArray[np.bool_] = (left == 2) & (right == 1) & pair_valid
        nonfatal[:-1] |= pair_21
        fatal[1:] |= pair_21

        pair_13: NDArray[np.bool_] = (left == 1) & (right == 3) & pair_valid
        fatal[:-1] |= pair_13
        nonfatal[1:] |= pair_13

        pair_23: NDArray[np.bool_] = (left == 2) & (right == 3) & pair_valid
        fatal[:-1] |= pair_23
        fatal[1:] |= pair_23
    
    fatal_idx = np.flatnonzero(fatal)
    nonfatal_idx = np.flatnonzero(nonfatal)

    if return_clean:
        arr_clean: NDArray[np.int_] = arr.copy()
        arr_clean[fatal | nonfatal] = 0
        return arr_clean, fatal_idx, nonfatal_idx

    return None, fatal_idx, nonfatal_idx

def _find_colliding_tones(
    vert_AOD_cmds: NDArray[np.int_],
    horiz_AOD_cmds:NDArray[np.int_],
    return_clean: bool = False,
) -> Union[Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]], Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]]:
    """
    Report collision indices (fatal vs nonfatal) and optionally return a copy with noncolliding tones replaced with zeros.

    Returns cleaned command lists for backward compatibility, and numpy index arrays
    for collision victim reporting.
    """

    if return_clean:
        v_clean, v_collision_inevitable, v_collision_avoidable = _classify_fatal_and_nonfatal_colliding_tones(vert_AOD_cmds, return_clean=return_clean)
        h_clean, h_collision_inevitable, h_collision_avoidable = _classify_fatal_and_nonfatal_colliding_tones(horiz_AOD_cmds, return_clean=return_clean)

        return (
            v_clean,
            h_clean,
            v_collision_inevitable,
            v_collision_avoidable,
            h_collision_inevitable,
            h_collision_avoidable,
        )

    else:
        _, v_collision_inevitable, v_collision_avoidable = _classify_fatal_and_nonfatal_colliding_tones(vert_AOD_cmds, return_clean=return_clean)
        _, h_collision_inevitable, h_collision_avoidable = _classify_fatal_and_nonfatal_colliding_tones(horiz_AOD_cmds, return_clean=return_clean)

        return (
            v_collision_inevitable,
            v_collision_avoidable,
            h_collision_inevitable,
            h_collision_avoidable,
        )
        
def collision_eligibility_from_tones(
    move_set: Sequence[object],
    v_collision_inevitable: NDArray[np.intp],
    v_collision_avoidable: NDArray[np.intp],
    h_collision_inevitable: NDArray[np.intp],
    h_collision_avoidable: NDArray[np.intp],
    source_rows: NDArray | None = None,
    source_cols: NDArray | None = None,
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Build per-move eligibility masks for victims of both avoidable and inevitable collisions.

    Collision-tone detection yields axis indices (row/col indices). We map those
    back to per-move eligibility by checking whether each move's *source* row/col
    lies on a colliding index.

    Implementation detail:
    We avoid `np.isin` by building boolean lookup tables (masks) and indexing them
    with `source_rows/source_cols`. The masks must be sized to cover BOTH:
      - the colliding indices, and
      - the maximum source index present in move_set
    otherwise indexing can go out of bounds when (e.g.) the only crossed index is 0
    but some move sources are at row 1, 2, ...
    """
    n_moves: int = len(move_set)
    if n_moves == 0:
        return np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.bool_)

    if source_rows is None:
        source_rows = np.asarray([m.from_row for m in move_set], dtype=np.intp)
    if source_cols is None:
        source_cols = np.asarray([m.from_col for m in move_set], dtype=np.intp)

    eligible_static: NDArray[np.bool_] = np.zeros(n_moves, dtype=np.bool_)
    eligible_moving: NDArray[np.bool_] = np.zeros(n_moves, dtype=np.bool_)

    # ---- Vertical axis lookup tables ----
    if v_collision_inevitable.size:
        max_needed_v = int(max(source_rows.max(initial=0), v_collision_inevitable.max(initial=0)))
        v_mask = np.zeros(max_needed_v + 1, dtype=np.bool_)
        v_mask[v_collision_inevitable] = True
        eligible_static |= v_mask[source_rows]

    if v_collision_avoidable.size:
        max_needed_v = int(max(source_rows.max(initial=0), v_collision_avoidable.max(initial=0)))
        v_mask_m = np.zeros(max_needed_v + 1, dtype=np.bool_)
        v_mask_m[v_collision_avoidable] = True
        eligible_moving |= v_mask_m[source_rows]

    # ---- Horizontal axis lookup tables ----
    if h_collision_inevitable.size:
        max_needed_h = int(max(source_cols.max(initial=0), h_collision_inevitable.max(initial=0)))
        h_mask = np.zeros(max_needed_h + 1, dtype=np.bool_)
        h_mask[h_collision_inevitable] = True
        eligible_static |= h_mask[source_cols]

    if h_collision_avoidable.size:
        max_needed_h = int(max(source_cols.max(initial=0), h_collision_avoidable.max(initial=0)))
        h_mask_m = np.zeros(max_needed_h + 1, dtype=np.bool_)
        h_mask_m[h_collision_avoidable] = True
        eligible_moving |= h_mask_m[source_cols]

    return eligible_static, eligible_moving