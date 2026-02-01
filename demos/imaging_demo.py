"""
Simple demo of the new imaging integration in atommovr.

This shows how to:
1. Generate realistic atom array images using AtomArray.render_realistic_image() 
2. Extract grids from images using the imaging.extraction pipeline
3. Use multiple angle estimation techniques
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path so demos can be run from the demos/ folder
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from atommover.utils.core import ArrayGeometry
try:
    from atommover.utils.AtomArray import AtomArray
    from atommover.utils.imaging.extraction import BlobDetection
except Exception as e:
    # Catch any import-time issue (including SyntaxError in imaging modules)
    print(f"Import error or other issue while loading imaging modules: {e}")
    print("Skipping imaging demo (module import failed).")
    exit(0)

from atommover.utils.imaging.animation import make_single_species_gif
from atommover.utils.Move import Move

def demo_image_generation():
    """Demo realistic image synthesis from atom arrays."""
    print("=== Image Generation Demo ===")
    
    # Create and load a small atom array
    arr = AtomArray(shape=[4, 6], n_species=1)
    arr.load_tweezers()
    print(f"Loaded array with {np.sum(arr.matrix)} atoms")
    
    # Generate realistic image with different parameters
    img1 = arr.render_realistic_image(sigma=1.0, image_shape=(1280, 1280), noise_level=0.01)
    img2 = arr.render_realistic_image(sigma=2.5, image_shape=(1280, 1280), noise_level=0.05, angle=10)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(img1, cmap='Blues')
    ax1.set_title('Sharp atoms (σ=1.0, angle=0°)')
    ax1.axis('off')
    
    ax2.imshow(img2, cmap='Blues') 
    ax2.set_title('Blurry atoms (σ=2.5, angle=10°)')
    ax2.axis('off')
    
    plt.tight_layout()
    os.makedirs('figs/imaging', exist_ok=True)
    plt.savefig('figs/imaging/demo_generation.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

def demo_extraction_pipeline():
    """Demo the full extraction pipeline with angle estimation."""
    print("=== Extraction Pipeline Demo ===")
    
    # Create a test grid with some rotation
    shape = (6, 8)
    arr = AtomArray(shape=shape, n_species=1)
    arr.load_tweezers()
    print(f"Loaded array with {int(np.sum(arr.matrix))} atoms")

    angle = 15  # degrees
    
    # Generate realistic image with different parameters
    img = arr.render_realistic_image(sigma=1.0, image_shape=(1280, 1280), noise_level=0.01, angle=angle)
    print(f"Generated test image with angle {angle}")
    plt.imshow(img, cmap='Blues')
    plt.title(f'Simulated Atom Image (angle={angle}°)')
    plt.axis('off')
    plt.savefig('figs/imaging/demo_extraction_input.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Show different extraction methods and angle estimators
    methods = [('blob', 'pca'), ('blob', 'fit_rect'), ('blob', None)]

    blob_detector = BlobDetection(
        shape=(shape[0], shape[1]),
        scale=(1, 1),
        logger=None,
    )
    
    for i, (method, angle_method) in enumerate(methods):
        _, angle_deg, _ = blob_detector.extract_estimate_rotate_and_assign(
            img,
            grid_shape=shape,
            angle_method=angle_method,
        )

        print(f"Method {i+1}: method={method}, angle_method={angle_method}, extracted angle={angle_deg}")


# create a source and a target AtomArray to show start and goal states and visualize them
def create_demo_arrays():
    shape = (6, 6)
    arr_source = AtomArray(shape=shape, n_species=1, geom=ArrayGeometry.RECTANGULAR)
    arr_source.load_tweezers()
    
    arr_target = AtomArray(shape=shape, n_species=1, geom=ArrayGeometry.RECTANGULAR)
    arr_target.load_tweezers()
    # target is a filled rectangle in the center
    r0 = (shape[0] - 4) // 2
    c0 = (shape[1] - 4) // 2
    target_mask = np.zeros(shape, dtype=int)
    target_mask[r0:r0+4, c0:c0+4] = 1
    arr_target.matrix[:,:,0] = target_mask
    
    # visualize start and goal states with the realistic rendering
    img_source = arr_source.render_realistic_image(sigma=1.5, brightness=1, image_shape=(1280, 1280), noise_level=0.01, angle=10)
    img_target = arr_target.render_realistic_image(sigma=1.5, brightness=1, image_shape=(1280, 1280), noise_level=0.01)
    img_adjusted = arr_source.render_realistic_image(sigma=1.5, brightness=1, image_shape=(1280, 1280), noise_level=0.01)

    # run blob detection on the source image and overlay detected centroids
    spots = int(sum(arr_source.matrix[:,:,0].flatten()))
    print(f"Expecting to detect {spots} blobs in the source image")

    blob_detector = BlobDetection(
        shape=(shape[0], shape[1]),
        spots=spots,
        scale=(1, 1),
        logger=None,
    )
    binary_grid, _, _ = blob_detector.extract_estimate_rotate_and_assign(img_source, grid_shape=shape, visualize=True)
    n_detected = 0 if binary_grid is None else np.sum(binary_grid)
    print(f"Detected {n_detected} blobs on the source image")

    # save images
    os.makedirs('figs/imaging', exist_ok=True)

    plt.imsave('figs/imaging/extraction_demo_start_state.png', img_source, cmap='Blues')
    plt.imsave('figs/imaging/extraction_demo_target_state.png', img_target, cmap='Blues')
    plt.imsave('figs/imaging/extraction_demo_adjusted_state.png', img_adjusted, cmap='Blues')
    plt.close()



def demo_gif_generation():
    """Generate a small demo GIF using a handcrafted move list."""
    print("=== GIF Generation Demo ===")
    arr = AtomArray(shape=[4, 4], n_species=1)
    arr.load_tweezers()
    # place a few atoms manually
    arr.matrix[:, :, 0] = 0
    arr.matrix[0, 0, 0] = 1
    arr.matrix[0, 3, 0] = 1
    arr.matrix[3, 0, 0] = 1

    # construct a simple batch: move (0,0)->(1,1) and (0,3)->(2,2)
    moves = [
        Move(0, 0, 1, 1),
        Move(0, 3, 2, 2),
    ]
    # single batch (list-of-lists)
    move_batches = [moves]

    print("Saving frames to ./figs/frames/ and gif to ./figs/resorting/")
    make_single_species_gif(arr, move_batches, savename="demo_single_species_gif", duration=0.3)
    print("GIF generation complete: ./figs/resorting/demo_single_species_gif.gif")



if __name__ == "__main__":
    print("AtomMovr Imaging Integration Demo")
    print("=" * 40)
    
    try:
        demo_image_generation()
        demo_extraction_pipeline() 
        create_demo_arrays()
        demo_gif_generation()

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
