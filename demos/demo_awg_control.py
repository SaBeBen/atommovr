import os
import sys

# Ensure repo root is on sys.path so demos can be run from the demos/ folder
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from atommover.utils.awg_control import RFConverter, AODSettings, PhysicalParams 
from atommover.utils.Move import Move

def main():
    # 1. Setup Configuration
    print("1. Configuring AOD and Physical Parameters...")
    aod_settings = AODSettings(
        f_min_v=80e6,
        f_max_v=89e6,
        f_min_h=80e6,
        f_max_h=89e6,
        grid_rows=10,
        grid_cols=10
    )
    
    phys_params = PhysicalParams(
        AOD_speed=0.1,    # 0.1 m/s (or um/us)
        spacing=5e-6      # 5 um spacing
    )
    
    converter = RFConverter(aod_settings, phys_params)
    print("   Configuration complete.")

    # 2. Define some moves (a single parallel batch)
    print("\n2. Defining a batch of moves...")
    # Scenario: 
    # Atom 1: (0,0) -> (0,1) (Moving right in row 0)
    # Atom 2: (1,0) -> (1,2) (Moving right in row 1, further)
    # Atom 3: (0,5) -> (0,6) (Moving right in row 0, same row as Atom 1)
    moves = [
        Move(0, 0, 0, 1),
        Move(1, 0, 1, 2),
        Move(0, 5, 0, 6)
    ]
    for m in moves:
        print(f"   Move: {m}")

    # 3. Convert to RF commands
    print("\n3. Converting to RF commands...")
    batch = converter.convert_moves(moves)
    
    print(f"   Batch Total Duration: {batch.total_duration*1e6:.2f} us")
    print(f"   Number of Ramps: {len(batch.ramps)}")
    
    # 4. Inspect Ramps
    print("\n4. Generated RF Ramps:")
    print(f"   {'Channel':<8} | {'Start Freq (MHz)':<16} | {'End Freq (MHz)':<16} | {'Duration (us)':<14}")
    print("-" * 65)
    
    for ramp in batch.ramps:
        print(f"   {ramp.channel:<8} | {ramp.f_start/1e6:<16.2f} | {ramp.f_end/1e6:<16.2f} | {ramp.duration*1e6:<14.2f}")

    print("\nObservations:")
    print(" - Note that for Row 0 (Vertical Channel), we have two atoms moving.")
    print("   Since they both start at Row 0 and end at Row 0, they share the same Vertical tone.")
    print("   The converter should deduplicate this to a single V ramp.")
    
    v_ramps = [r for r in batch.ramps if r.channel == 'V']
    print(f"   Vertical Ramps: {len(v_ramps)} (Expected 2: one for Row 0, one for Row 1)")
    
    print(" - The duration is determined by the longest move (Atom 2 moving 2 sites).")

if __name__ == "__main__":
    main()
