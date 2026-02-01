
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

from atommover.utils.Move import Move
from atommover.utils.core import PhysicalParams

@dataclass
class AODSettings:
    """
    Settings for the AOD RF conversion.
    """
    # Frequency ranges (Hz)
    f_min_v: float = 60e6
    f_max_v: float = 100e6
    f_min_h: float = 60e6
    f_max_h: float = 100e6
    
    # Grid dimensions (Total available sites)
    grid_rows: int = 10
    grid_cols: int = 10
    
    # Target array dimensions (L x L or R x C)
    # Used to determine the active region if needed, or for validation.
    target_rows: int = 10
    target_cols: int = 10
    
    # Alignment of the grid within the frequency range
    # 'center': The grid is centered in the frequency range
    # 'start': The grid starts at f_min
    alignment: str = 'center'

    @property
    def f_spacing_v(self) -> float:
        if self.grid_rows <= 1:
            return 0.0
        # Calculate spacing based on full frequency range and grid size
        # Assuming f_min corresponds to index 0 and f_max to index N-1 is one way.
        # Or assuming the range covers the whole FOV and we fit N sites in it.
        # Let's assume the range is the accessible range, and we distribute N sites evenly.
        return (self.f_max_v - self.f_min_v) / (self.grid_rows - 1)

    @property
    def f_spacing_h(self) -> float:
        if self.grid_cols <= 1:
            return 0.0
        return (self.f_max_h - self.f_min_h) / (self.grid_cols - 1)

@dataclass
class RFRamp:
    """
    Represents an RF ramp command for a single tone.
    """
    channel: str  # 'V' or 'H'
    f_start: float # Hz
    f_end: float   # Hz
    t_start: float # seconds (relative to batch start)
    duration: float # seconds
    amplitude: float # 0.0 to 1.0

@dataclass
class AWGBatch:
    """
    Represents a set of RF commands to be executed in parallel (one batch).
    """
    ramps: List[RFRamp]
    total_duration: float

class RFConverter:
    def __init__(self, settings: AODSettings, physical_params: PhysicalParams):
        self.settings = settings
        self.params = physical_params

    def _grid_to_freq(self, index: int, axis: str) -> float:
        """
        Convert grid index to frequency.
        axis: 'V' (row) or 'H' (col)
        """
        if axis == 'V':
            f_min = self.settings.f_min_v
            spacing = self.settings.f_spacing_v
            # If alignment is center, we might want to shift indices?
            # But usually indices 0..N-1 map to f_min..f_max
            # Let's stick to direct mapping for now as it's most robust.
            return f_min + index * spacing
        else:
            f_min = self.settings.f_min_h
            spacing = self.settings.f_spacing_h
            return f_min + index * spacing

    def convert_moves(self, moves: List[Move]) -> AWGBatch:
        """
        Convert a list of parallel moves (a batch) into RF commands.
        Deduplicates identical ramps (e.g. multiple atoms in the same row).
        """
        if not moves:
            return AWGBatch(ramps=[], total_duration=0.0)

        # Calculate max distance to determine batch duration
        max_dist = 0.0
        for move in moves:
            # Using Move's distance (which is in grid units) * spacing
            dist = move.distance * self.params.spacing
            if dist > max_dist:
                max_dist = dist
        
        # Duration = distance / speed
        duration = max_dist / self.params.AOD_speed
        if duration == 0:
            duration = 1e-6 # Min duration for zero-distance moves

        v_ramps: Dict[Tuple[float, float], RFRamp] = {}
        h_ramps: Dict[Tuple[float, float], RFRamp] = {}

        for move in moves:
            # Vertical AOD (Rows)
            f_v_start = self._grid_to_freq(move.from_row, 'V')
            f_v_end = self._grid_to_freq(move.to_row, 'V')
            
            key_v = (f_v_start, f_v_end)
            if key_v not in v_ramps:
                v_ramps[key_v] = RFRamp(
                    channel='V',
                    f_start=f_v_start,
                    f_end=f_v_end,
                    t_start=0.0,
                    duration=duration,
                    amplitude=1.0
                )
            
            # Horizontal AOD (Cols)
            f_h_start = self._grid_to_freq(move.from_col, 'H')
            f_h_end = self._grid_to_freq(move.to_col, 'H')
            
            key_h = (f_h_start, f_h_end)
            if key_h not in h_ramps:
                h_ramps[key_h] = RFRamp(
                    channel='H',
                    f_start=f_h_start,
                    f_end=f_h_end,
                    t_start=0.0,
                    duration=duration,
                    amplitude=1.0
                )

        all_ramps = list(v_ramps.values()) + list(h_ramps.values())
        return AWGBatch(ramps=all_ramps, total_duration=duration)

    def convert_sequence(self, move_batches: List[List[Move]]) -> List[AWGBatch]:
        """
        Convert a sequence of batches (list of lists of moves) into a sequence of AWG batches.
        """
        return [self.convert_moves(batch) for batch in move_batches]

