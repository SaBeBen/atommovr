"""
demo_awg_control.py
===================
Demonstrates the AWG/AOD RF-command model implemented in ``awg_control.py``.

Key concept — the AOD tone model
---------------------------------
A 2D AOD trap grid is driven by *two* independent 1D AOD channels:

  * **Channel 0** (V-AOD / row axis): one DDS tone per *row*.
    Every atom in a given row is addressed by the same frequency.
  * **Channel 1** (H-AOD / col axis): one DDS tone per *column*.
    Every atom in a given column is addressed by the same frequency.

Consequently, within a single trigger event (one ``AWGBatch``):
  * A source row can be mapped to **exactly one** target row.
  * A source column can be mapped to **exactly one** target column.

This is a hard hardware constraint: a single tone cannot simultaneously
drive atoms in the same column to two different positions.  The
``RFConverter`` enforces this and raises ``ValueError`` for conflicting
targets.

The demo below illustrates:
  1. A FAILED batch (the original scenario — intentional conflict).
  2. A VALID single batch (non-conflicting source columns and rows).
  3. A SEQUENTIAL two-batch solution that achieves the original intent.
"""

import os
import sys

# Ensure repo root is on sys.path so demos can be run from the demos/ folder
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from atommover.utils.awg_control import AODSettings, RFConverter
from atommover.utils.core import PhysicalParams
from atommover.utils.Move import Move


SEP = "-" * 70


def _print_batch(batch, label: str) -> None:
    """Pretty-print one AWGBatch."""
    print(f"\n{label}")
    print(f"  Duration : {batch.total_duration_s * 1e6:.2f} µs")
    print(f"  Ramps    : {len(batch.ramps)}")
    header = f"  {'Ch':<4} {'Core':<6} {'f_start (MHz)':<16} {'f_end (MHz)':<16} {'moving?'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in batch.ramps:
        moving = "YES" if abs(r.f_end - r.f_start) > 1 else "-"
        print(
            f"  {r.channel:<4} {r.core:<6} "
            f"{r.f_start / 1e6:<16.3f} {r.f_end / 1e6:<16.3f} {moving}"
        )


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Setup
    # ------------------------------------------------------------------ #
    print(SEP)
    print("1. Configuring AOD and Physical Parameters")
    print(SEP)

    aod = AODSettings(
        f_min_v=80e6, f_max_v=89e6,
        f_min_h=80e6, f_max_h=89e6,
        grid_rows=10, grid_cols=10,
    )
    phys = PhysicalParams(
        AOD_speed=0.1,   # m/s  (= µm/µs)
        spacing=5e-6,    # 5 µm inter-site spacing
    )
    converter = RFConverter(aod, phys)

    print(f"  Grid          : {aod.grid_rows} rows × {aod.grid_cols} cols")
    print(f"  Freq range V  : {aod.f_min_v/1e6:.0f}–{aod.f_max_v/1e6:.0f} MHz  "
          f"(step {aod.f_spacing_v/1e6:.3f} MHz)")
    print(f"  Freq range H  : {aod.f_min_h/1e6:.0f}–{aod.f_max_h/1e6:.0f} MHz  "
          f"(step {aod.f_spacing_h/1e6:.3f} MHz)")
    print(f"  Core map ch0  : {converter.core_map[0]}")
    print(f"  Core map ch1  : {converter.core_map[1]}")

    # ------------------------------------------------------------------ #
    # 2. FAILED scenario: conflicting column targets
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("2. Intentionally INVALID batch (demonstrates the hardware constraint)")
    print(SEP)
    print("""
  Original scenario:
    Atom A: (row=0, col=0) → (row=0, col=1)   col 0 → 1
    Atom B: (row=1, col=0) → (row=1, col=2)   col 0 → 2   ← CONFLICT
    Atom C: (row=0, col=5) → (row=0, col=6)   col 5 → 6

  WHY THIS FAILS:
    Atoms A and B both originate from column 0.  The H-AOD uses a single
    DDS tone per source column — col-0 tone cannot simultaneously sweep to
    80+1×step MHz (for atom A) AND to 80+2×step MHz (for atom B).
    The converter raises ValueError to prevent an impossible command.

  This behaviour is CORRECT and WANTED.  It mirrors the physical
  impossibility of driving one AOD tone to two frequencies at once.
""")

    conflicting_moves = [
        Move(0, 0, 0, 1),   # col 0 → 1
        Move(1, 0, 1, 2),   # col 0 → 2  ← conflicts with above
        Move(0, 5, 0, 6),   # col 5 → 6
    ]
    try:
        converter.convert_moves(conflicting_moves)
        print("  *** ERROR: expected ValueError was not raised! ***")
    except ValueError as exc:
        print(f"  ✓ ValueError raised as expected:")
        print(f"    {exc}")

    # ------------------------------------------------------------------ #
    # 3. VALID single batch: distinct source columns and rows
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("3. VALID single batch (non-conflicting moves)")
    print(SEP)
    print("""
  Corrected scenario — every source column appears at most once:
    Atom A: (row=0, col=0) → (row=0, col=1)   col 0 → 1
    Atom B: (row=1, col=3) → (row=1, col=5)   col 3 → 5   (different source col)
    Atom C: (row=2, col=5) → (row=2, col=6)   col 5 → 6

  NOTE: rows 0, 1, 2 all remain on the same row (no row conflict either).
""")

    valid_moves = [
        Move(0, 0, 0, 1),   # col 0 → 1
        Move(1, 3, 1, 5),   # col 3 → 5
        Move(2, 5, 2, 6),   # col 5 → 6
    ]
    batch = converter.convert_moves(valid_moves)
    _print_batch(batch, "Batch result:")

    ch0_amp = sum(r.amplitude_pct for r in batch.ramps if r.channel == 0)
    ch1_amp = sum(r.amplitude_pct for r in batch.ramps if r.channel == 1)
    print(f"\n  Amplitude budget  ch0: {ch0_amp:.1f}%   ch1: {ch1_amp:.1f}%  "
          f"(limit: 40% each)")

    # ------------------------------------------------------------------ #
    # 4. Achieving the original intent via sequential batches
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("4. SEQUENTIAL batches — achieving the original intent")
    print(SEP)
    print("""
  To move atoms A and B (both from col 0) to different columns, split
  the movement into two sequential trigger events:

    Batch 1: Move atom A  (row=0, col=0) → (row=0, col=1)
             Move atom C  (row=0, col=5) → (row=0, col=6)
             [Atom B stays at (row=1, col=0) — static tone]

    Batch 2: Move atom B  (row=1, col=0) → (row=1, col=2)
             [Atom A now at col=1, atom C at col=6 — static tones]
""")

    sequential_batches = converter.convert_sequence([
        [Move(0, 0, 0, 1), Move(0, 5, 0, 6)],  # batch 1
        [Move(1, 0, 1, 2)],                      # batch 2
    ])

    for i, b in enumerate(sequential_batches, 1):
        _print_batch(b, f"Batch {i}:")

    total_us = sum(b.total_duration_s for b in sequential_batches) * 1e6
    print(f"\n  Total sequential duration: {total_us:.2f} µs  "
          f"(vs {sequential_batches[0].total_duration_s*1e6:.2f} µs for a single parallel batch)")

    # ------------------------------------------------------------------ #
    # 5. Holding configuration (between rounds)
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("5. Holding configuration (static — sent between rearrangement rounds)")
    print(SEP)

    holding = converter.holding_config()
    _print_batch(holding, "Holding batch (all f_start == f_end):")
    moving_in_hold = [r for r in holding.ramps if abs(r.f_end - r.f_start) > 1]
    assert len(moving_in_hold) == 0, "BUG: holding config has non-static ramps!"
    print("\n  ✓ All ramps are static — atoms held in place.")

    print(f"\n{SEP}")
    print("Demo complete.")
    print(SEP)


if __name__ == "__main__":
    main()
