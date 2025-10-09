"""
Extract saliency values for fixations from Deepgaze IIE saliency maps.

This script extracts saliency values around fixation points within a 30 pixel radius.
The output CSV contains:
- subject, session, scene_id, trial, start_time, mean_gx, mean_gy, gx_image, gy_image, saliency_value

Corrected coordinate transformation:
- Eye tracking: (0,0) at top-left of screen (1024×768 pixels)
- Image: centred on screen, size (947×710 pixels)
- Transformation: subtract centering offset
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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


def normalize_log_density_to_probability(log_density_map):
    """
    Convert log density saliency map to probability density (0-1 normalised).

    Following pysaliency conventions:
    - Apply exponential to convert from log space
    - Normalise to sum to 1 (proper probability distribution)

    Parameters:
    -----------
    log_density_map : np.ndarray
        Saliency map in log density space

    Returns:
    --------
    np.ndarray : Probability density map (sums to 1)
    """
    # Convert from log density to density using exponential
    # Subtract max for numerical stability before exp
    log_density_shifted = log_density_map - np.max(log_density_map)
    density_map = np.exp(log_density_shifted)

    # Normalise to sum to 1 (probability distribution)
    prob_density = density_map / np.sum(density_map)

    return prob_density


def load_saliency_map_from_npy(coco_id, saliency_dir, normalize_to_prob=True):
    """
    Load a generated saliency map by cocoID from NPY file.

    Parameters:
    -----------
    coco_id : int
        Scene ID (COCO ID)
    saliency_dir : str
        Directory containing saliency NPY files
    normalize_to_prob : bool
        If True, convert log density to probability density (0-1)

    Returns:
    --------
    np.ndarray or None : Saliency map (probability density if normalize_to_prob=True)
    """
    filename = f"{str(int(coco_id)).zfill(12)}_MEG_size.npy"
    filepath = os.path.join(saliency_dir, filename)

    if os.path.exists(filepath):
        saliency_map = np.load(filepath)

        if normalize_to_prob:
            # DeepGaze outputs are in log density space
            saliency_map = normalize_log_density_to_probability(saliency_map)

        return saliency_map
    else:
        print(f"Warning: Saliency map not found for scene {coco_id} at {filepath}")
        return None


def eyetracking_to_image_coords(gx_et, gy_et, screen_size=(1024, 768), image_size=(947, 710)):
    """
    Convert eye tracking coordinates to image coordinates.
    
    Eye tracking system: (0,0) at top-left, but y increases UPWARD (mathematical convention)
    Image/NumPy system: (0,0) at top-left, y increases DOWNWARD (screen convention)
    Screen size: 1024×768 pixels
    Image: centred on screen, size 947×710 pixels
    
    Steps:
    1. Flip y-axis: y_flipped = screen_height - y_eyetracking
    2. Subtract centering offset to get image coordinates
    
    Parameters:
    -----------
    gx_et, gy_et : float
        Eye tracking coordinates (origin at screen top-left, y increases upward)
    screen_size : tuple
        Screen size in pixels (width, height)
    image_size : tuple
        Image size in pixels (width, height)
        
    Returns:
    --------
    tuple : (gx_image, gy_image) coordinates in image space (origin at image top-left, y down)
    """
    screen_w, screen_h = screen_size
    image_w, image_h = image_size
    
    # Calculate centering offset (image is centred on screen)
    offset_x = (screen_w - image_w) / 2
    offset_y = (screen_h - image_h) / 2
    
    # X: just subtract offset (same direction in both systems)
    gx_image = gx_et - offset_x
    
    # Y: flip axis AND subtract offset
    # Eye tracking: y increases upward from bottom
    # Image: y increases downward from top
    gy_image = (screen_h - gy_et) - offset_y
    
    return gx_image, gy_image


def add_image_coordinates_to_fixations(fixation_data, screen_size=(1024, 768), 
                                       image_size=(947, 710), output_dir=None):
    """
    Add image coordinate columns to fixation dataframe.
    
    This preprocessing step computes image coordinates once and stores them
    as new columns, which can then be used by both extraction and visualisation.
    
    Parameters:
    -----------
    fixation_data : pd.DataFrame
        DataFrame with fixation data (must have 'mean_gx' and 'mean_gy' columns)
    screen_size : tuple
        Screen size in pixels (width, height)
    image_size : tuple
        Image size in pixels (width, height)
    output_dir : str or None
        Directory to save diagnostic plot (if None, saves to current directory)
        
    Returns:
    --------
    pd.DataFrame : Input dataframe with added 'gx_image' and 'gy_image' columns
    """
    print("\n=== Converting Eye Tracking Coordinates to Image Coordinates ===")
    print(f"Screen size: {screen_size[0]}×{screen_size[1]} pixels")
    print(f"Image size: {image_size[0]}×{image_size[1]} pixels")
    print(f"Note: Eye tracking y-axis increases UPWARD, image y-axis increases DOWNWARD")
    
    offset_x = (screen_size[0] - image_size[0]) / 2
    offset_y = (screen_size[1] - image_size[1]) / 2
    print(f"Image offset on screen: ({offset_x}, {offset_y}) pixels")
    
    # Vectorised coordinate transformation
    gx_image, gy_image = eyetracking_to_image_coords(
        fixation_data['mean_gx'].values,
        fixation_data['mean_gy'].values,
        screen_size=screen_size,
        image_size=image_size
    )
    
    fixation_data['gx_image'] = gx_image
    fixation_data['gy_image'] = gy_image
    
    # Validation: check bounds
    n_total = len(fixation_data)
    n_valid_x = ((gx_image >= 0) & (gx_image < image_size[0])).sum()
    n_valid_y = ((gy_image >= 0) & (gy_image < image_size[1])).sum()
    n_valid_both = ((gx_image >= 0) & (gx_image < image_size[0]) & 
                    (gy_image >= 0) & (gy_image < image_size[1])).sum()
    
    print(f"\nCoordinate bounds check:")
    print(f"  Total fixations: {n_total}")
    print(f"  Valid X coordinates (0-{image_size[0]}): {n_valid_x} ({n_valid_x/n_total:.1%})")
    print(f"  Valid Y coordinates (0-{image_size[1]}): {n_valid_y} ({n_valid_y/n_total:.1%})")
    print(f"  Valid both X and Y: {n_valid_both} ({n_valid_both/n_total:.1%})")
    
    # Show example transformations
    print(f"\nExample coordinate transformations:")
    print(f"  Eye tracking (gx, gy) -> Image (gx_img, gy_img)")
    for i in range(min(5, len(fixation_data))):
        print(f"  ({fixation_data['mean_gx'].iloc[i]:.1f}, {fixation_data['mean_gy'].iloc[i]:.1f}) -> "
              f"({fixation_data['gx_image'].iloc[i]:.1f}, {fixation_data['gy_image'].iloc[i]:.1f})")
    
    # Create diagnostic visualisation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Eye tracking coordinates on screen
    axes[0].scatter(fixation_data['mean_gx'], fixation_data['mean_gy'], 
                   s=1, alpha=0.3, c='blue')
    axes[0].set_xlim(0, screen_size[0])
    axes[0].set_ylim(0, screen_size[1])
    axes[0].invert_yaxis()  # Eye tracking y increases upward
    axes[0].axvline(offset_x, color='red', linestyle='--', alpha=0.5, label='Image bounds')
    axes[0].axvline(offset_x + image_size[0], color='red', linestyle='--', alpha=0.5)
    axes[0].axhline(offset_y, color='red', linestyle='--', alpha=0.5)
    axes[0].axhline(offset_y + image_size[1], color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels, increases upward)')
    axes[0].set_title('Eye Tracking Coordinates\n(on screen)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right: Image coordinates
    axes[1].scatter(fixation_data['gx_image'], fixation_data['gy_image'], 
                   s=1, alpha=0.3, c='green')
    axes[1].set_xlim(0, image_size[0])
    axes[1].set_ylim(0, image_size[1])
    axes[1].invert_yaxis()  # Image y increases downward (standard)
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels, increases downward)')
    axes[1].set_title('Image Coordinates\n(after transformation)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_dir is None:
        output_dir = '.'
    diag_path = os.path.join(output_dir, 'coordinate_transformation_diagnostic.png')
    plt.savefig(diag_path, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to: {diag_path}")
    plt.close()
    
    return fixation_data


def extract_saliency_around_fixation(saliency_map, gx_image, gy_image, radius=30):
    """
    Extract mean saliency value in a circular region around fixation point.
    
    Parameters:
    -----------
    saliency_map : np.ndarray
        2D saliency map
    gx_image, gy_image : float
        Fixation coordinates in image space (origin at top-left)
    radius : int
        Radius in pixels around fixation point
        
    Returns:
    --------
    float : Median saliency value in the region
    """
    if saliency_map is None:
        return np.nan
    
    h, w = saliency_map.shape
    
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
    
    return np.median(values_in_circle)


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
        metadata_file = os.path.join(crops_image_dir, "metadata", 
                                     f"as{subject:02d}_crops_metadata.csv")

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
    print(f"\nCombined data: {len(et_events)} total fixation events from "
          f"{len(all_subjects_data)} subjects")

    return et_events


def process_fixations_with_saliency(fixation_data, saliency_dir, output_path, 
                                   radius=30):
    """
    Process fixation data and extract saliency values.
    
    Note: fixation_data must already have 'gx_image' and 'gy_image' columns
    from add_image_coordinates_to_fixations().

    Parameters:
    -----------
    fixation_data : pd.DataFrame
        DataFrame with fixation data (must have 'gx_image' and 'gy_image')
    saliency_dir : str
        Directory containing saliency map NPY files
    output_path : str
        Path to save output CSV
    radius : int
        Radius in pixels for saliency extraction
    """
    print(f"\n=== Processing {len(fixation_data)} Fixations ===")

    # Check required columns
    required_columns = ['subject', 'session', 'sceneID', 'trial', 'start_time', 
                       'mean_gx', 'mean_gy', 'gx_image', 'gy_image']
    missing_columns = [col for col in required_columns if col not in fixation_data.columns]
    if missing_columns:
        print(f"Error: Missing columns in fixation data: {missing_columns}")
        return None

    # Copy relevant columns
    output_data = fixation_data[required_columns].copy()
    output_data.rename(columns={'sceneID': 'scene_id'}, inplace=True)

    # Initialise saliency value column
    output_data['saliency_value'] = np.nan

    # Process unique scenes
    unique_scenes = output_data['scene_id'].unique()
    print(f"Processing {len(unique_scenes)} unique scenes...")

    # Track statistics
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
                fixation['gx_image'],
                fixation['gy_image'],
                radius=radius
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
    print(f"\nResults saved to: {output_path}")

    return output_data


def visualize_saliency_extraction_examples(fixation_data, saliency_dir, scenes_dir,
                                          output_dir, radius=30, n_examples=2, seed=42):
    """
    Create visualisation showing how saliency values are extracted from scenes.

    Plots example scenes with saliency maps overlaid and circles around fixations.
    This illustrates the extraction method and verifies coordinate alignment.
    
    Note: fixation_data must have 'gx_image' and 'gy_image' columns.

    Parameters:
    -----------
    fixation_data : pd.DataFrame
        DataFrame with fixation data (must include 'gx_image', 'gy_image', 'saliency_value')
    saliency_dir : str
        Directory containing saliency map NPY files
    scenes_dir : str
        Directory containing scene images
    output_dir : str
        Directory to save visualisation
    radius : int
        Radius in pixels for extraction circles
    n_examples : int
        Number of example scenes to visualise
    seed : int
        Random seed for scene selection
    """
    print(f"\n=== Creating Saliency Extraction Visualisations ===")

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
        
        # Subsample to 1 subject for clarity
        available_subjects = scene_fixations['subject'].unique()
        if len(available_subjects) > 0:
            scene_fixations = scene_fixations[scene_fixations['subject'] == available_subjects[0]]
        
        # Randomly sample up to 10 fixations to avoid overcrowding
        if len(scene_fixations) > 10:
            scene_fixations = scene_fixations.sample(n=10, random_state=seed + scene_idx)

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Left plot: Scene image with fixation circles
        axes[0].imshow(scene_array)
        axes[0].set_title(f"Scene {scene_id} - Fixation Locations", fontsize=14)
        axes[0].axis('off')
        
        marker_size = 100
        for _, fix in scene_fixations.iterrows():
            # Use precomputed image coordinates
            gx_img, gy_img = fix['gx_image'], fix['gy_image']
            axes[0].scatter(gx_img, gy_img, s=marker_size, c='red', 
                          edgecolors='black', alpha=0.7, zorder=5)
            
        # Right plot: Saliency map with fixation circles
        saliency_for_viz = saliency_map.copy()

        im = axes[1].imshow(saliency_for_viz, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f"Saliency Probability Density - {radius}px Extraction Radius", 
                         fontsize=14)
        axes[1].axis('off')

        # Draw circles around fixations on saliency map
        for _, fix in scene_fixations.iterrows():
            # Use precomputed image coordinates
            gx_img, gy_img = fix['gx_image'], fix['gy_image']
            axes[1].scatter(gx_img, gy_img, s=marker_size, c='cyan', 
                          edgecolors='black', alpha=0.7, zorder=5)
          
        # Add colourbar for saliency map
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Fixation Probability Density', rotation=270, labelpad=20)

        plt.tight_layout()
        
        # Save figure
        output_filename = os.path.join(output_dir,
                                      f"saliency_extraction_example_scene_{scene_id}.png")
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"  Saved visualisation: {output_filename}")
        plt.close()

        # Print statistics for this scene
        if 'saliency_value' in scene_fixations.columns:
            valid_count = scene_fixations['saliency_value'].notna().sum()
            print(f"  Scene {scene_id}: {len(scene_fixations)} fixations visualised "
                  f"({valid_count} with saliency)")
            print(f"  Image size: {scene_array.shape[1]}×{scene_array.shape[0]} pixels")
            print(f"  Saliency map size: {saliency_map.shape[1]}×{saliency_map.shape[0]} pixels")

    print(f"\nCreated {len(selected_scenes)} example visualisations in {output_dir}")


def main():
    """Main function to extract saliency values for fixations."""
    print("=== Extracting Saliency Values for Fixations ===")

    # Configuration
    radius = 30  # pixels
    model_name = "deepgaze_iie"
    centerbias_name = "mit1003"
    crop_size_pix = 100  # Match memorability analysis
    subjects = [1,2,3,4,5]
    test_scenes = []#[65400, 106113]  # For quick testing, set to [] for all scenes
    
    # Display parameters
    screen_size = (1024, 768)  # Screen size in pixels
    image_size = (947, 710)    # Image size in pixels

    # Setup paths
    try:
        input_dir = avs_directory.get_input_dirs(server="uos")
    except:
        input_dir = "/share/klab/datasets/avs/input"

    # Crops directory (same as memorability and ease-of-recognition)
    crops_dir = os.path.join(input_dir, "fixation_crops")
    crop_image_subdir_name = f"avs_meg_fixation_crops_scene_{crop_size_pix}"
    crops_image_dir = os.path.join(crops_dir, crop_image_subdir_name)

    # Scenes directory (for visualisation)
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

    output_path = os.path.join(output_dir, 
                              f"fixation_saliency_values_{model_name}_{centerbias_name}_"
                              f"radius_{radius}px.csv")

    print(f"\nConfiguration:")
    print(f"  Screen size: {screen_size[0]}×{screen_size[1]} pixels")
    print(f"  Image size: {image_size[0]}×{image_size[1]} pixels")
    print(f"  Extraction radius: {radius} pixels")
    print(f"  Subjects: {subjects}")
    print(f"\nPaths:")
    print(f"  Crops metadata: {crops_image_dir}")
    print(f"  Saliency maps: {saliency_dir}")
    print(f"  Output: {output_path}")

    # Check if saliency directory exists
    if not os.path.exists(saliency_dir):
        print(f"\nError: Saliency directory not found: {saliency_dir}")
        print("Please ensure saliency maps have been generated using deepgaze.py")
        return

    # Check for saliency files
    npy_files = [f for f in os.listdir(saliency_dir) if f.endswith('.npy')]
    if not npy_files:
        print(f"\nError: No .npy saliency files found in {saliency_dir}")
        return

    print(f"  Found {len(npy_files)} saliency map files")

    # Load fixation data from crops metadata
    print("\n=== Loading Fixation Data ===")
    fixation_data = load_fixation_data(crops_image_dir, subjects=subjects)

    if fixation_data is None:
        print("Error: Could not load fixation data")
        return

    print(f"Loaded {len(fixation_data)} fixation events")
    
    # Subsample scenes for quick testing
    if len(test_scenes) > 0:
        fixation_data = fixation_data[fixation_data['sceneID'].isin(test_scenes)]
        print(f"Subsampled to {len(fixation_data)} fixations from {len(test_scenes)} test scenes")
    
    # CRITICAL STEP: Convert eye tracking coordinates to image coordinates
    fixation_data = add_image_coordinates_to_fixations(
        fixation_data,
        screen_size=screen_size,
        image_size=image_size,
        output_dir=output_dir
    )
    
    # Process fixations with saliency
    result_data = process_fixations_with_saliency(
        fixation_data,
        saliency_dir,
        output_path,
        radius=radius
    )

    if result_data is not None:
        # Display sample results
        print(f"\n=== Sample Results ===")
        display_cols = ['subject', 'session', 'scene_id', 'trial', 
                       'gx_image', 'gy_image', 'saliency_value']
        print(result_data[display_cols].head(10))

        # Basic statistics
        print(f"\n=== Saliency Value Statistics (Probability Density) ===")
        print(result_data['saliency_value'].describe())

        # Create visualisation examples
        visualize_saliency_extraction_examples(
            result_data,
            saliency_dir,
            scenes_dir,
            output_dir,
            radius=radius,
            n_examples=40,
            seed=42
        )

    print("\n=== Processing Complete ===")


if __name__ == "__main__":
    main()