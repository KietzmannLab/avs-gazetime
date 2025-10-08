"""
Extract saliency values for fixations from Deepgaze IIE saliency maps.

This script mimics the fix2cap processing pipeline to extract saliency values
around fixation points within a 30 pixel radius. The output CSV contains:
- subject, session, scene_id, trial, start_time, mean_gx, mean_gy, saliency_value

Based on the structure of the avs-gazetime project, following the design principles
of existing fix2cap scripts.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

# AVS-specific imports
try:
    from avs_gazetime.config import PLOTS_DIR_NO_SUB, PLOTS_DIR, PLOTS_DIR_BEHAV
    import avs_machine_room.dataloader.tools.avs_directory_tools as avs_directory
    from avs_gazetime.scenes.saliency.deepgaze import load_saliency_map
except ImportError:
    print("Warning: AVS-specific imports not available. Using fallback paths.")
    def get_input_dirs(server="uos"):
        return "/share/klab/datasets/avs/input"


def load_saliency_map_from_npy(coco_id, saliency_dir):
    """Load a generated saliency map by cocoID from NPY file."""
    filename = f"{str(int(coco_id)).zfill(12)}_MEG_size.npy"
    filepath = os.path.join(saliency_dir, filename)

    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        print(f"Warning: Saliency map not found for scene {coco_id} at {filepath}")
        return None


def screen_to_image_coords(gx_screen, gy_screen, screen_size=(1024, 768), image_size=(947, 710)):
    """
    Convert screen-based coordinates to image-based coordinates.

    During the experiment, images were centered on the screen.
    Eye tracking coordinates (mean_gx, mean_gy) are in screen coordinates.
    We need to convert them to image coordinates for saliency extraction.

    Parameters:
    -----------
    gx_screen, gy_screen : float
        Fixation coordinates in screen pixels
    screen_size : tuple
        Screen size in pixels (width, height)
    image_size : tuple
        Image size in pixels (width, height)

    Returns:
    --------
    tuple : (gx_image, gy_image) coordinates in image space
    """
    screen_w, screen_h = screen_size
    image_w, image_h = image_size

    # Calculate offset (images were centered on screen)
    offset_x = (screen_w - image_w) / 2
    offset_y = (screen_h - image_h) / 2

    # Convert screen coordinates to image coordinates
    gx_image = gx_screen - offset_x
    gy_image = gy_screen - offset_y

    return gx_image, gy_image


def extract_saliency_around_fixation(saliency_map, gx_screen, gy_screen, radius=30,
                                     screen_size=(1024, 768), image_size=None):
    """
    Extract mean saliency value in a circular region around fixation point.

    Parameters:
    -----------
    saliency_map : np.ndarray
        2D saliency map
    gx_screen, gy_screen : float
        Fixation coordinates in screen pixels (not image pixels!)
    radius : int
        Radius in pixels around fixation point
    screen_size : tuple
        Screen size (width, height) in pixels
    image_size : tuple or None
        Image size (width, height). If None, inferred from saliency_map shape

    Returns:
    --------
    float : Mean saliency value in the region
    """
    if saliency_map is None:
        return np.nan

    h, w = saliency_map.shape

    # If image_size not provided, use saliency map dimensions
    if image_size is None:
        image_size = (w, h)

    # Convert screen coordinates to image coordinates
    gx_image, gy_image = screen_to_image_coords(gx_screen, gy_screen, screen_size, image_size)

    # Convert to integer coordinates
    center_x = int(round(gx_image))
    center_y = int(round(gy_image))

    # Check bounds
    if center_x < 0 or center_x >= w or center_y < 0 or center_y >= h:
        return np.nan

    # Create coordinate arrays
    y_coords, x_coords = np.ogrid[:h, :w]

    # Create circular mask
    mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2

    # Extract values within the circle
    values_in_circle = saliency_map[mask]

    if len(values_in_circle) == 0:
        return np.nan

    return np.mean(values_in_circle)


def load_fixation_data(crops_image_dir, subjects=[1, 2, 3, 4, 5]):
    """
    Load fixation data from crops metadata CSV files.
    Uses the same data source as memorability and ease-of-recognition analyses.

    Parameters:
    -----------
    crops_image_dir : str
        Directory containing crop images and metadata
    subjects : list
        List of subject IDs to load

    Returns:
    --------
    pd.DataFrame
        Combined fixation data from all subjects
    """
    all_subjects_data = []

    for subject in subjects:
        metadata_file = os.path.join(crops_image_dir, "metadata", f"as{subject:02d}_crops_metadata.csv")

        if not os.path.exists(metadata_file):
            print(f"Warning: Metadata file not found for subject {subject}: {metadata_file}")
            continue

        metadata_df = pd.read_csv(metadata_file, low_memory=False)
        print(f"Loaded {len(metadata_df)} events for subject {subject}")

        # Apply basic filtering (matching memorability analysis)
        metadata_df = metadata_df[metadata_df['type'] == 'fixation']
        metadata_df = metadata_df[metadata_df['recording'] == 'scene']

        all_subjects_data.append(metadata_df)

    if not all_subjects_data:
        print("Error: No metadata files found for any subjects")
        return None

    # Combine all subjects
    et_events = pd.concat(all_subjects_data, ignore_index=True)
    print(f"\nCombined data: {len(et_events)} total fixation events from {len(all_subjects_data)} subjects")

    return et_events


def process_fixations_with_saliency(fixation_data, saliency_dir, output_path, radius=30,
                                   screen_size=(1024, 768), image_size=(947, 710)):
    """
    Process fixation data and extract saliency values.

    Parameters:
    -----------
    fixation_data : pd.DataFrame
        DataFrame with fixation data
    saliency_dir : str
        Directory containing saliency map NPY files
    output_path : str
        Path to save output CSV
    radius : int
        Radius in pixels for saliency extraction
    screen_size : tuple
        Screen size (width, height) in pixels during experiment
    image_size : tuple
        Image size (width, height) in pixels
    """
    print(f"Processing {len(fixation_data)} fixations...")
    print(f"Screen size: {screen_size[0]}x{screen_size[1]} pixels")
    print(f"Image size: {image_size[0]}x{image_size[1]} pixels")
    print(f"Screen-to-image offset: ({(screen_size[0]-image_size[0])/2:.1f}, {(screen_size[1]-image_size[1])/2:.1f})")

    # Prepare output columns
    required_columns = ['subject', 'session', 'scene_id', 'trial', 'start_time', 'mean_gx', 'mean_gy']

    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in fixation_data.columns]
    if missing_columns:
        print(f"Error: Missing columns in fixation data: {missing_columns}")
        return None

    # Copy relevant columns
    output_data = fixation_data[required_columns].copy()
    #output_data.rename(columns={'scene_id': 'scene_id'}, inplace=True)

    # Initialize saliency value column
    output_data['saliency_value'] = np.nan

    # Process unique scenes
    unique_scenes = output_data['scene_id'].unique()
    print(f"Processing {len(unique_scenes)} unique scenes...")

    # Track failed scenes
    failed_scenes = []
    successful_count = 0

    for i, scene_id in enumerate(unique_scenes):
        if (i + 1) % 100 == 0:
            print(f"Processing scene {i+1}/{len(unique_scenes)}: {scene_id}")

        # Load saliency map for this scene
        saliency_map = load_saliency_map_from_npy(scene_id, saliency_dir)

        if saliency_map is None:
            failed_scenes.append(scene_id)
            continue

        # Get all fixations for this scene
        scene_fixations = output_data[output_data['scene_id'] == scene_id]

        # Extract saliency values for each fixation
        for idx, fixation in scene_fixations.iterrows():
            saliency_value = extract_saliency_around_fixation(
                saliency_map,
                fixation['mean_gx'],
                fixation['mean_gy'],
                radius=radius,
                screen_size=screen_size,
                image_size=image_size
            )
            output_data.loc[idx, 'saliency_value'] = saliency_value

        successful_count += 1

    # Summary statistics
    total_fixations = len(output_data)
    valid_saliency = output_data['saliency_value'].notna().sum()

    print(f"\n=== Processing Summary ===")
    print(f"Total fixations: {total_fixations}")
    print(f"Fixations with valid saliency: {valid_saliency}")
    print(f"Success rate: {valid_saliency/total_fixations:.1%}")
    print(f"Scenes processed successfully: {successful_count}/{len(unique_scenes)}")
    print(f"Failed scenes: {len(failed_scenes)}")

    if failed_scenes:
        print(f"First 10 failed scene IDs: {failed_scenes[:10]}")

    # Save results
    output_data.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    return output_data


def visualize_saliency_extraction_examples(fixation_data, saliency_dir, scenes_dir,
                                          output_dir, radius=30, n_examples=2, seed=42,
                                          screen_size=(1024, 768), image_size=(947, 710)):
    """
    Create visualization showing how saliency values are extracted from scenes.

    Plots example scenes with saliency maps overlaid and 30-pixel circles around fixations.
    This illustrates the extraction method and verifies coordinate alignment.

    Parameters:
    -----------
    fixation_data : pd.DataFrame
        DataFrame with fixation data
    saliency_dir : str
        Directory containing saliency map NPY files
    scenes_dir : str
        Directory containing scene images
    output_dir : str
        Directory to save visualization
    radius : int
        Radius in pixels for extraction circles
    n_examples : int
        Number of example scenes to visualize
    seed : int
        Random seed for scene selection
    screen_size : tuple
        Screen size (width, height) in pixels
    image_size : tuple
        Image size (width, height) in pixels
    """
    print(f"\n=== Creating Saliency Extraction Visualizations ===")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Select random scenes with sufficient fixations
    scene_fixation_counts = fixation_data.groupby('scene_id').size()
    scenes_with_fixations = scene_fixation_counts[scene_fixation_counts >= 5].index.tolist()

    if len(scenes_with_fixations) < n_examples:
        print(f"Warning: Only {len(scenes_with_fixations)} scenes with enough fixations")
        n_examples = len(scenes_with_fixations)

    selected_scenes = np.random.choice(scenes_with_fixations, size=n_examples, replace=False)

    for scene_idx, scene_id in enumerate(selected_scenes):
        print(f"Processing example {scene_idx + 1}/{n_examples}: Scene {scene_id}")

        # Load scene image
        scene_filename = f"{str(int(scene_id)).zfill(12)}_MEG_size.jpg"
        scene_path = os.path.join(scenes_dir, scene_filename)

        if not os.path.exists(scene_path):
            print(f"  Scene image not found: {scene_path}")
            continue

        scene_img = Image.open(scene_path)
        scene_array = np.array(scene_img)

        # Load saliency map
        saliency_map = load_saliency_map_from_npy(scene_id, saliency_dir)

        if saliency_map is None:
            print(f"  Saliency map not found for scene {scene_id}")
            continue

        # Get fixations for this scene
        scene_fixations = fixation_data[fixation_data['scene_id'] == scene_id].copy()

        # Randomly sample up to 10 fixations to avoid overcrowding
        if len(scene_fixations) > 10:
            scene_fixations = scene_fixations.sample(n=10, random_state=seed + scene_idx)

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Left plot: Scene image with fixation circles
        axes[0].imshow(scene_array)
        axes[0].set_title(f"Scene {scene_id} - Fixation Locations", fontsize=14)
        axes[0].axis('off')

        # Draw circles around fixations (converting screen to image coordinates)
        for _, fix in scene_fixations.iterrows():
            gx_screen, gy_screen = fix['mean_gx'], fix['mean_gy']
            # Convert screen coordinates to image coordinates
            gx_img, gy_img = screen_to_image_coords(gx_screen, gy_screen, screen_size, image_size)

            circle = patches.Circle((gx_img, gy_img), radius, linewidth=2,
                                   edgecolor='red', facecolor='none', alpha=0.7)
            axes[0].add_patch(circle)
            # Add crosshair at fixation center
            axes[0].plot(gx_img, gy_img, 'r+', markersize=10, markeredgewidth=2)

        # Right plot: Saliency map with fixation circles
        # Normalize saliency map for visualization
        saliency_normalized = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

        im = axes[1].imshow(saliency_normalized, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f"Saliency Map - {radius}px Extraction Radius", fontsize=14)
        axes[1].axis('off')

        # Draw circles around fixations on saliency map (same image coordinates)
        for _, fix in scene_fixations.iterrows():
            gx_screen, gy_screen = fix['mean_gx'], fix['mean_gy']
            # Convert screen coordinates to image coordinates
            gx_img, gy_img = screen_to_image_coords(gx_screen, gy_screen, screen_size, image_size)

            circle = patches.Circle((gx_img, gy_img), radius, linewidth=2,
                                   edgecolor='cyan', facecolor='none', alpha=0.8)
            axes[1].add_patch(circle)
            # Add crosshair at fixation center
            axes[1].plot(gx_img, gy_img, 'c+', markersize=10, markeredgewidth=2)

        # Add colorbar for saliency map
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Saliency', rotation=270, labelpad=20)

        # Add text annotation explaining the visualization
        offset_x = (screen_size[0] - image_size[0]) / 2
        offset_y = (screen_size[1] - image_size[1]) / 2
        fig.text(0.5, 0.02,
                f'Red/Cyan circles show {radius}-pixel radius regions around each fixation.\n'
                f'Saliency values are averaged within these circular regions.\n'
                f'Eye-tracking coordinates (mean_gx, mean_gy) are in screen space ({screen_size[0]}×{screen_size[1]}px),\n'
                f'converted to image space ({image_size[0]}×{image_size[1]}px) by subtracting offset ({offset_x:.1f}, {offset_y:.1f}).',
                ha='center', fontsize=9, style='italic', wrap=True)

        plt.tight_layout(rect=[0, 0.06, 1, 1])

        # Save figure
        output_filename = os.path.join(output_dir,
                                      f"saliency_extraction_example_scene_{scene_id}.png")
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization: {output_filename}")
        plt.close()

        # Print some statistics for this scene
        valid_saliency = scene_fixations['saliency_value'].notna() if 'saliency_value' in scene_fixations.columns else []
        if len(valid_saliency) > 0:
            print(f"  Scene {scene_id}: {len(scene_fixations)} fixations visualized")
            print(f"  Image size: {scene_array.shape[1]}x{scene_array.shape[0]} pixels")
            print(f"  Saliency map size: {saliency_map.shape[1]}x{saliency_map.shape[0]} pixels")

    print(f"Created {n_examples} example visualizations in {output_dir}")


def main():
    """Main function to extract saliency values for fixations."""
    print("=== Extracting Saliency Values for Fixations ===")

    # Configuration
    radius = 30  # pixels
    model_name = "deepgaze_iie"
    centerbias_name = "mit1003"
    crop_size_pix = 100  # Match memorability analysis
    subjects = [1, 2, 3, 4, 5]

    # Coordinate system parameters (from experiment setup)
    screen_size = (1024, 768)  # Screen size in pixels (width, height)
    image_size = (947, 710)    # Image size in pixels (width, height)

    # Setup paths
    try:
        input_dir = avs_directory.get_input_dirs(server="uos")
    except:
        input_dir = "/share/klab/datasets/avs/input"

    # Crops directory (same as memorability and ease-of-recognition)
    crops_dir = os.path.join(input_dir, "fixation_crops")
    crop_image_subdir_name = f"avs_meg_fixation_crops_scene_{crop_size_pix}"
    crops_image_dir = os.path.join(crops_dir, crop_image_subdir_name)

    # Scenes directory (for visualization)
    scenes_dir = os.path.join(input_dir, "NSD_scenes_MEG_size_adjusted_925")

    # Saliency maps directory
    saliency_dir = os.path.join(input_dir, "saliency_maps",
                               f"{model_name}_{centerbias_name}")

    # Output directory
    try:
        output_dir = PLOTS_DIR_BEHAV
    except:
        output_dir = os.path.join(input_dir, "results")

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"fixation_saliency_values_{model_name}_{centerbias_name}_radius_{radius}px.csv")

    print(f"Crops metadata directory: {crops_image_dir}")
    print(f"Saliency directory: {saliency_dir}")
    print(f"Output path: {output_path}")

    # Check if saliency directory exists
    if not os.path.exists(saliency_dir):
        print(f"Error: Saliency directory not found: {saliency_dir}")
        print("Please ensure saliency maps have been generated using deepgaze.py")
        return

    # Check for saliency files
    npy_files = [f for f in os.listdir(saliency_dir) if f.endswith('.npy')]
    if not npy_files:
        print(f"Error: No .npy saliency files found in {saliency_dir}")
        return

    print(f"Found {len(npy_files)} saliency map files")

    # Load fixation data from crops metadata
    print("\nLoading fixation data from crops metadata...")
    fixation_data = load_fixation_data(crops_image_dir, subjects=subjects)

    if fixation_data is None:
        print("Error: Could not load fixation data")
        return

    print(f"Loaded {len(fixation_data)} fixation events")
    print(f"Columns: {list(fixation_data.columns)}")

    # Process fixations with saliency
    result_data = process_fixations_with_saliency(
        fixation_data,
        saliency_dir,
        output_path,
        radius=radius,
        screen_size=screen_size,
        image_size=image_size
    )

    if result_data is not None:
        # Display sample results
        print(f"\nSample of results:")
        print(result_data[['subject', 'session', 'scene_id', 'trial', 'saliency_value']].head(10))

        # Basic statistics
        print(f"\nSaliency value statistics:")
        print(result_data['saliency_value'].describe())

        # Create visualization examples
        visualize_saliency_extraction_examples(
            result_data,
            saliency_dir,
            scenes_dir,
            output_dir,
            radius=radius,
            n_examples=2,
            seed=42,
            screen_size=screen_size,
            image_size=image_size
        )


if __name__ == "__main__":
    main()