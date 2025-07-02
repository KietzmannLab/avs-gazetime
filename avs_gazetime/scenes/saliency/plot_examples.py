"""
Plot 3 example scenes with Deepgaze IIE saliency overlays for Nature Neuroscience paper.
Simple, clear visualization following the established AVS style.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# AVS-specific imports (adjust paths as needed)
try:
    import avs_machine_room.dataloader.tools.avs_directory_tools as avs_directory
    from avs_gazetime.config import PLOTS_DIR_NO_SUB
except ImportError:
    print("Warning: AVS-specific imports not available. Using fallback paths.")
    def get_input_dirs(server="uos"):
        return "/share/klab/datasets/avs/input"
    PLOTS_DIR_NO_SUB = "/share/klab/psulewski/psulewski/active-visual-semantics-MEG/results/fullrun/plots"


def load_scene_image(coco_id, scenes_dir):
    """Load scene image by COCO ID."""
    scene_filename = f"{str(int(coco_id)).zfill(12)}_MEG_size.jpg"
    scene_path = os.path.join(scenes_dir, scene_filename)
    
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene not found: {scene_path}")
    
    image = Image.open(scene_path).convert('RGB')
    return np.array(image), scene_path


def load_saliency_map(coco_id, h5_path):
    """Load saliency map by COCO ID from H5 file."""
    scene_id_str = str(int(coco_id)).zfill(12)
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Saliency file not found: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        if scene_id_str in f:
            saliency = f[scene_id_str][:]
            # Remove extra dimensions if present
            if saliency.ndim == 3 and saliency.shape[0] == 1:
                saliency = saliency.squeeze(0)
            return saliency
        else:
            raise KeyError(f"Saliency map not found for scene {coco_id}")


def plot_scene_with_saliency(scene_image, saliency_map, ax, title=None, alpha=0.6):
    """
    Plot scene with saliency overlay.
    
    Parameters:
    -----------
    scene_image : np.ndarray
        RGB scene image
    saliency_map : np.ndarray
        Saliency map values
    ax : matplotlib.axes.Axes
        Axis to plot on
    title : str, optional
        Title for the subplot
    alpha : float
        Transparency of saliency overlay
    """
    # Display scene image
    ax.imshow(scene_image)
    
    # Normalize saliency map for better visualization
    saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    
    # Create saliency overlay using 'hot' colormap
    im = ax.imshow(saliency_norm, cmap='hot', alpha=alpha, interpolation='bilinear')
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    return im


def get_example_scenes(avs_scenes, n_examples=3, seed=42):
    """
    Select example scenes with good coverage of semantic diversity.
    
    Parameters:
    -----------
    avs_scenes : pd.DataFrame
        DataFrame with scene information
    n_examples : int
        Number of example scenes to select
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    selected_coco_ids : list
        List of selected COCO IDs
    """
    np.random.seed(seed)
    
    # For reproducible examples, select specific scenes that work well
    # These should represent diverse scene types
    all_coco_ids = avs_scenes['cocoID'].tolist()
    
    # Select examples with good spacing
    indices = np.linspace(0, len(all_coco_ids)-1, n_examples, dtype=int)
    selected_coco_ids = [all_coco_ids[i] for i in indices]
    
    return selected_coco_ids


def main():
    """Generate figure with 3 example scenes with DG saliency overlays."""
    
    print("=== Plotting 3 Example Scenes with Deepgaze IIE Saliency Overlays ===")
    
    # Configuration
    model_name = "deepgaze_iie"
    centerbias_name = "mit1003"
    n_examples = 3
    
    # Setup paths
    try:
        input_dir = avs_directory.get_input_dirs(server="uos")
    except:
        input_dir = "/share/klab/datasets/avs/input"
    
    scenes_dir = os.path.join(input_dir, "NSD_scenes_MEG_size_adjusted_925")
    avs_scene_selection_path = os.path.join(input_dir, "scene_sampling_MEG", 
                                          "experiment_cocoIDs.csv")
    
    saliency_dir = os.path.join(input_dir, "saliency_maps", 
                               f"{model_name}_{centerbias_name}")
    h5_filename = f"saliency_maps_{model_name}_{centerbias_name}.h5"
    h5_path = os.path.join(saliency_dir, h5_filename)
    
    # Output directory
    output_dir = os.path.join(PLOTS_DIR_NO_SUB, "saliency_examples")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Scenes directory: {scenes_dir}")
    print(f"Saliency file: {h5_path}")
    print(f"Output directory: {output_dir}")
    
    # Load AVS scene information
    if not os.path.exists(avs_scene_selection_path):
        print(f"Scene selection file not found, using NSD info as fallback")
        nsd_info_path = os.path.join(input_dir, "NSD_info", 
                                   "NSD_ids_with_shared1000_and_special100.csv")
        avs_scenes = pd.read_csv(nsd_info_path)
        if 'cocoId' in avs_scenes.columns:
            avs_scenes.rename(columns={'cocoId': 'cocoID'}, inplace=True)
    else:
        avs_scenes = pd.read_csv(avs_scene_selection_path)
    
    print(f"Found {len(avs_scenes)} available scenes")
    
    # Check if saliency file exists
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Saliency file not found: {h5_path}")
    
    # Select example scenes
    selected_coco_ids = get_example_scenes(avs_scenes, n_examples)
    print(f"Selected scenes: {selected_coco_ids}")
    
    # Set up the plot with Nature Neuroscience style
    plt.style.use('default')  # Start with default style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white'
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_examples, figsize=(15, 5), dpi=300)
    if n_examples == 1:
        axes = [axes]
    
    # Process each example scene
    for i, coco_id in enumerate(selected_coco_ids):
        try:
            # Load scene image
            scene_image, scene_path = load_scene_image(coco_id, scenes_dir)
            print(f"Loaded scene {coco_id}: {os.path.basename(scene_path)}")
            
            # Load saliency map
            saliency_map = load_saliency_map(coco_id, h5_path)
            print(f"Loaded saliency map for scene {coco_id}")
            
            # Plot scene with saliency overlay
            title = f"Scene {coco_id}"
            im = plot_scene_with_saliency(scene_image, saliency_map, axes[i], 
                                        title=title, alpha=0.6)
            
        except Exception as e:
            print(f"Error processing scene {coco_id}: {e}")
            # Plot placeholder
            axes[i].text(0.5, 0.5, f"Error loading\nscene {coco_id}", 
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=12, bbox=dict(boxstyle="round", facecolor='lightgray'))
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    
    # Add colorbar for saliency intensity
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', 
                       fraction=0.05, pad=0.08, shrink=0.8)
    cbar.set_label('Saliency intensity (normalized)', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_filename = f"dg_saliency_examples_{model_name}_{centerbias_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    # Also save as PDF for publications
    output_path_pdf = output_path.replace('.png', '.pdf')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print(f"\n=== Saved figures ===")
    print(f"PNG: {output_path}")
    print(f"PDF: {output_path_pdf}")
    
    plt.show()
    
    return output_path, selected_coco_ids





if __name__ == "__main__":
    # Plot 3 example scenes
    output_path, selected_ids = main()