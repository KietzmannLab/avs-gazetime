"""
Generate Deepgaze IIE saliency maps for all scenes in the Active Visual Semantics (AVS) dataset.

This script processes pre-sized NSD scenes used in the AVS experiment and generates 
corresponding saliency maps using the Deepgaze IIE model with center bias applied.

Dependencies:
- torch
- torchvision  
- deepgaze_pytorch (install: pip install git+https://github.com/matthias-k/DeepGaze.git)
- pandas
- numpy
- PIL
- scipy
- avs_gazetime.config (local import)
- avs_machine_room.dataloader.tools.avs_directory_tools (local import)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import zoom
from scipy.special import logsumexp
import warnings
warnings.filterwarnings('ignore')

# AVS-specific imports (adjust paths as needed)
try:
    from avs_gazetime.config import PLOTS_DIR_NO_SUB, PLOTS_DIR, PLOTS_DIR_BEHAV
    import avs_machine_room.dataloader.tools.avs_directory_tools as avs_directory
except ImportError:
    print("Warning: AVS-specific imports not available. Using fallback paths.")
    def get_input_dirs(server="uos"):
        return "/share/klab/datasets/avs/input"

# Import Deepgaze IIE
try:
    from deepgaze_pytorch import DeepGazeIIE
except ImportError:
    print("Error: deepgaze_pytorch not installed. Install with: pip install deepgaze-pytorch")
    raise


def load_deepgaze_model(device='cuda'):
    """Load and initialize the Deepgaze IIE model."""
    print(f"Loading Deepgaze IIE model on {device}...")
    model = DeepGazeIIE(pretrained=True)
    model = model.to(device)
    model.eval()
    print("Deepgaze IIE model loaded successfully")
    return model


def preprocess_image_for_deepgaze(image_path):
    """Preprocess pre-sized AVS image for Deepgaze IIE model."""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Convert to tensor and transpose to CHW format (DeepGaze expects this)
    image_array = np.array(image).astype(np.float32)
    # Transpose from HWC to CHW format
    image_tensor = torch.tensor(image_array.transpose(2, 0, 1)).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_size


def generate_saliency_map(model, image_tensor, centerbias_tensor, device='cuda'):
    """Generate saliency map using Deepgaze IIE with center bias."""
    image_tensor = image_tensor.to(device)
    centerbias_tensor = centerbias_tensor.to(device)
    
    with torch.no_grad():
        log_density = model(image_tensor, centerbias_tensor)
        saliency_map = log_density.squeeze().cpu().numpy()
    
    return saliency_map


def process_single_scene(coco_id, scenes_dir, model, centerbias_tensor, device, output_dir):
    """Process a single scene to generate saliency map and save as NPY."""
    try:
        # Find scene file using the MEG naming pattern
        scene_filename = f"{str(int(coco_id)).zfill(12)}_MEG_size.jpg"
        scene_path = os.path.join(scenes_dir, scene_filename)
        
        if not os.path.exists(scene_path):
            print(f"Scene {coco_id} not found at {scene_path}")
            return False
        
        # Process image
        image_tensor, original_size = preprocess_image_for_deepgaze(scene_path)
        
        saliency_map = generate_saliency_map(model, image_tensor, centerbias_tensor, device)
        
        # Save as NPY file with same naming pattern as short script
        output_filename = f"{str(int(coco_id)).zfill(12)}_MEG_size.npy"
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, saliency_map)
        
        return True
        
    except Exception as e:
        print(f"Error processing scene {coco_id}: {str(e)}")
        return False


def main():
    """Main function to generate Deepgaze IIE saliency maps for all AVS scenes."""
    print("=== Deepgaze IIE Saliency Map Generation for AVS Scenes ===")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "deepgaze_iie"
    centerbias_name = "mit1003"
    
    print(f"Device: {device}")
    
    # Setup paths
    try:
        input_dir = avs_directory.get_input_dirs(server="uos")
    except:
        input_dir = "/share/klab/datasets/avs/input"
    
    # Path to pre-sized scenes with MEG naming pattern
    scenes_dir = os.path.join(input_dir, "NSD_scenes_MEG_size_adjusted_925")
    
    # Path to AVS scene selection
    avs_scene_selection_path = os.path.join(input_dir, "scene_sampling_MEG", 
                                          "experiment_cocoIDs.csv")
    
    # Create output directory named by model and centerbias
    output_dir = os.path.join(input_dir, "saliency_maps", 
                             f"{model_name}_{centerbias_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Pre-sized scenes directory: {scenes_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load center bias template - EXACTLY like short script
    print("Loading center bias template...")
    centerbias_path = "/share/klab/datasets/avs/input/models/deepgaze/centerbias_mit1003.npy"
    try:
        centerbias_template = np.load(centerbias_path)
        print("Loaded MIT1003 center bias")
    except FileNotFoundError:
        print(f"MIT1003 center bias not found at {centerbias_path}, using uniform center bias")
        centerbias_template = np.zeros((1024, 1024))
        centerbias_name = "uniform"  # Update centerbias name for output directory
        # Update output directory path
        output_dir = os.path.join(input_dir, "saliency_maps", 
                                 f"{model_name}_{centerbias_name}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Load AVS scene IDs
    if not os.path.exists(avs_scene_selection_path):
        print(f"Error: AVS scene selection file not found at {avs_scene_selection_path}")
        # Try alternative path
        nsd_info_path = os.path.join(input_dir, "NSD_info", 
                                   "NSD_ids_with_shared1000_and_special100.csv")
        if os.path.exists(nsd_info_path):
            avs_scenes = pd.read_csv(nsd_info_path)
            if 'cocoId' in avs_scenes.columns:
                avs_scenes.rename(columns={'cocoId': 'cocoID'}, inplace=True)
        else:
            raise FileNotFoundError("Could not find scene information files")
    else:
        avs_scenes = pd.read_csv(avs_scene_selection_path)
    
    print(f"Found {len(avs_scenes)} scenes to process")
    
    # Get image size from first scene
    coco_ids = avs_scenes['cocoID'].tolist()
    first_scene_filename = f"{str(int(coco_ids[0])).zfill(12)}_MEG_size.jpg"
    first_scene_path = os.path.join(scenes_dir, first_scene_filename)
    
    if not os.path.exists(first_scene_path):
        raise FileNotFoundError(f"First scene not found at {first_scene_path}")
    
    # Get image dimensions
    with Image.open(first_scene_path) as img:
        avs_width, avs_height = img.size
    
    print(f"Detected image size: {avs_width}x{avs_height}")
    
    # Pre-resize center bias for detected image size - EXACTLY like short script
    centerbias_resized = zoom(
        centerbias_template,
        (
            avs_height / centerbias_template.shape[0],
            avs_width / centerbias_template.shape[1],
        ),
        order=0,
        mode="nearest",
    )
    # Renormalize log density - EXACTLY like short script
    centerbias_resized -= logsumexp(centerbias_resized)
    centerbias_tensor = torch.tensor([centerbias_resized])
    
    print(f"Center bias resized to {avs_height}x{avs_width}")
    
    # Load Deepgaze IIE model
    model = load_deepgaze_model(device)
    
    # Check if output files already exist
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
    if existing_files:
        print(f"Found {len(existing_files)} existing saliency files in {output_dir}")
        response = input("Continue and overwrite existing files? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return
    
    # Process scenes and save individual NPY files
    print(f"\nStarting saliency map generation...")
    print(f"Output directory: {output_dir}")
    
    successful_count = 0
    failed_scenes = []
    
    for i, coco_id in enumerate(coco_ids):
        if (i + 1) % 100 == 0:
            print(f"Processing scene {i+1}/{len(coco_ids)}: {coco_id}")
        
        success = process_single_scene(coco_id, scenes_dir, model, 
                                     centerbias_tensor, device, output_dir)
        
        if success:
            successful_count += 1
        else:
            failed_scenes.append(coco_id)
    
    # Summary
    print(f"\n=== Processing Complete ===")
    print(f"Total scenes: {len(coco_ids)}")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {len(failed_scenes)}")
    
    if failed_scenes:
        print(f"Failed scene IDs: {failed_scenes[:10]}")
        if len(failed_scenes) > 10:
            print(f"... and {len(failed_scenes) - 10} more")
        
        # Save failed scenes list
        failed_df = pd.DataFrame({'failed_cocoID': failed_scenes})
        failed_path = os.path.join(output_dir, 'failed_scenes.csv')
        failed_df.to_csv(failed_path, index=False)
        print(f"Failed scenes saved to: {failed_path}")
    
    print(f"Saliency maps saved to: {output_dir}")


def load_saliency_map(coco_id, saliency_dir):
    """Load a generated saliency map by cocoID from NPY file."""
    filename = f"{str(int(coco_id)).zfill(12)}_MEG_size.npy"
    filepath = os.path.join(saliency_dir, filename)
    
    if os.path.exists(filepath):
        return np.load(filepath)
    else:
        print(f"Saliency map not found for scene {coco_id} at {filepath}")
        return None


if __name__ == "__main__":
    main()