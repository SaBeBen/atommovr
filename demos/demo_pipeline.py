
import os
import sys
import numpy as np

# Ensure repo root is on sys.path so demos can be run from the demos/ folder
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from atommover.utils.AtomArray import AtomArray
from atommover.utils.core import PhysicalParams, array_shape_for_geometry
from atommover.utils.awg_control import AODSettings
from atommover.algorithms.single_species import PCFA
from atommover.utils.pipeline_integration import get_awg_sequence_for_algorithm  

def main():
    print("=== AtomMover Pipeline Demo ===")
    
    # 1. Select Algorithm and Determine Shape
    print("\n1. Configuring Algorithm and Array Shape...")
    algo = PCFA()
    target_L = 6

    # 2. Setup Physical Parameters and AOD Settings (needed to compute loading prob)
    print("\n2. Configuring System...")
    phys_params = PhysicalParams(
        AOD_speed=0.1,    # 0.1 m/s
        spacing=5e-6      # 5 um
    )

    # Determine rows/cols from the algorithm's preferred geometry spec
    rows, cols = array_shape_for_geometry(getattr(algo, "preferred_geometry_spec", None), target_L, phys_params.loading_prob)
    print(f"   PCFA preferred shape for {target_L}x{target_L} target: {rows}x{cols}")
    
    # Calculate max frequencies to maintain ~1MHz spacing
    f_spacing = 1e6
    f_min_v = 80e6
    f_max_v = f_min_v + (rows - 1) * f_spacing
    f_min_h = 80e6
    f_max_h = f_min_h + (cols - 1) * f_spacing

    aod_settings = AODSettings(
        f_min_v=f_min_v,
        f_max_v=f_max_v,
        f_min_h=f_min_h,
        f_max_h=f_max_h,
        grid_rows=rows,
        grid_cols=cols
    )
    
    # 3. Setup Atom Array
    print("\n3. Initializing Atom Array...")
    atom_array = AtomArray(shape=[rows, cols], n_species=1, params=phys_params)
    
    # Create a random loading (60% fill)
    phys_params.loading_prob = 0.6
    atom_array.load_tweezers()
    
    # Set a target configuration (L x L at top left)
    target = np.zeros((rows, cols, 1))
    target[:target_L, :target_L, 0] = 1
    atom_array.target = target
    
    print(f"   Initial Atoms: {np.sum(atom_array.matrix)}")
    print(f"   Target Atoms: {np.sum(atom_array.target)}")
    
    # 4. Run Pipeline
    print("\n4. Running PCFA Algorithm...")
    
    try:
        awg_sequence, (final_config, move_list, success) = get_awg_sequence_for_algorithm(
            algo, atom_array, aod_settings, phys_params
        )
        
        print(f"   Algorithm Success: {success}")
        print(f"   Number of Move Steps: {len(move_list)}")
        print(f"   Number of AWG Batches: {len(awg_sequence)}")
        
        total_duration = sum(batch.total_duration for batch in awg_sequence)
        print(f"   Total Sequence Duration: {total_duration*1e6:.2f} us")
        
        # Inspect first batch if exists
        if awg_sequence:
            print("\n   [First Batch Details]")
            first_batch = awg_sequence[0]
            print(f"   Duration: {first_batch.total_duration*1e6:.2f} us")
            print(f"   Ramps: {len(first_batch.ramps)}")
            for i, ramp in enumerate(first_batch.ramps[:5]):
                print(f"     Ramp {i}: {ramp.channel} {ramp.f_start/1e6:.1f}->{ramp.f_end/1e6:.1f} MHz")
            if len(first_batch.ramps) > 5:
                print("     ...")

    except Exception as e:
        print(f"   Error running pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
