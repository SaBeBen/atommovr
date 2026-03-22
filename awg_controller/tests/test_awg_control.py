import unittest
import numpy as np
from awg_controller.src.awg_control import RFConverter, AODSettings, RFRamp, AWGBatch
from atommovr.utils.Move import Move
from atommovr.utils.core import PhysicalParams

class TestAWGControl(unittest.TestCase):
    def setUp(self):
        self.settings = AODSettings(
            f_min_v=80e6,
            f_max_v=89e6,
            f_min_h=80e6,
            f_max_h=89e6,
            grid_rows=10,
            grid_cols=10
        )
        # AOD_speed 0.1 um/us = 0.1 m/s
        # spacing 5e-6 m = 5 um
        self.params = PhysicalParams(
            AOD_speed=0.1, 
            spacing=5e-6 
        )
        self.converter = RFConverter(self.settings, self.params)

    def test_grid_to_freq(self):
        # Center (0,0) should be 80 MHz
        self.assertEqual(self.converter._grid_to_freq(0, 'V'), 80e6)
        self.assertEqual(self.converter._grid_to_freq(0, 'H'), 80e6)
        
        # Index 1 should be 81 MHz
        self.assertEqual(self.converter._grid_to_freq(1, 'V'), 81e6)
        
        # Index -1 should be 79 MHz
        self.assertEqual(self.converter._grid_to_freq(-1, 'H'), 79e6)

    def test_convert_single_move(self):
        move = Move(0, 0, 1, 1) # (0,0) -> (1,1)
        # Distance = sqrt(1^2 + 1^2) = sqrt(2) sites
        # Physical distance = sqrt(2) * 5e-6 m
        # Duration = sqrt(2) * 5e-6 / 0.1 = sqrt(2) * 5e-5 s ~= 70.7 us
        
        batch = self.converter.convert_moves([move])
        
        expected_duration = np.sqrt(2) * 5e-6 / 0.1
        self.assertAlmostEqual(batch.total_duration, expected_duration)
        
        self.assertEqual(len(batch.ramps), 2) # 1 V ramp, 1 H ramp
        
        # Check V ramp
        v_ramps = [r for r in batch.ramps if r.channel == 'V']
        self.assertEqual(len(v_ramps), 1)
        self.assertEqual(v_ramps[0].f_start, 80e6)
        self.assertEqual(v_ramps[0].f_end, 81e6)
        
        # Check H ramp
        h_ramps = [r for r in batch.ramps if r.channel == 'H']
        self.assertEqual(len(h_ramps), 1)
        self.assertEqual(h_ramps[0].f_start, 80e6)
        self.assertEqual(h_ramps[0].f_end, 81e6)

    def test_deduplication(self):
        # Two atoms moving in the same row
        # Move 1: (0,0) -> (0,1)
        # Move 2: (0,2) -> (0,3)
        # Vertical AOD: Both start at row 0 and end at row 0. f_start=80e6, f_end=80e6.
        # Should be deduplicated to 1 V ramp.
        # Horizontal AOD: 
        # Move 1: col 0 -> 1 (80->81)
        # Move 2: col 2 -> 3 (82->83)
        # Should be 2 H ramps.
        
        moves = [Move(0, 0, 0, 1), Move(0, 2, 0, 3)]
        batch = self.converter.convert_moves(moves)
        
        v_ramps = [r for r in batch.ramps if r.channel == 'V']
        h_ramps = [r for r in batch.ramps if r.channel == 'H']
        
        self.assertEqual(len(v_ramps), 1)
        self.assertEqual(len(h_ramps), 2)

    def test_batch_duration(self):
        # Move 1: distance 1 site
        # Move 2: distance 2 sites
        # Duration should be determined by Move 2
        moves = [Move(0, 0, 0, 1), Move(1, 0, 1, 2)]
        batch = self.converter.convert_moves(moves)
        
        expected_duration = 2 * 5e-6 / 0.1 # 2 sites
        self.assertAlmostEqual(batch.total_duration, expected_duration)
        
        # All ramps should have this duration
        for ramp in batch.ramps:
            self.assertEqual(ramp.duration, expected_duration)

    def test_empty_batch(self):
        batch = self.converter.convert_moves([])
        self.assertEqual(len(batch.ramps), 0)
        self.assertEqual(batch.total_duration, 0.0)

if __name__ == '__main__':
    unittest.main()
