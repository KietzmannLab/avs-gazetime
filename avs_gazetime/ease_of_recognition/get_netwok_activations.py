#!/usr/bin/env python3
"""
Script to extract neural network model activations from fixation crops
using a ResNet-50 model trained on EcoSet.

Usage:
    python extract_activations.py --subject_id <subject_id> --crop_size <crop_size>
"""
import os
import torch
import argparse
from tqdm import tqdm
import gc

# thingsvision imports
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader

# Custom project imports
from avs_machine_room.dataloader.tools.avs_directory_tools import get_input_dirs
from avs_gazetime.config import PLOTS_DIR, PLOTS_DIR_BEHAV

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Extract neural network activations from fixation crops')
    parser.add_argument('--subject_id', type=int, required=True, help='Subject ID number')
    parser.add_argument('--crop_size', type=int, default=112, help='Crop size in pixels')
    parser.add_argument('--style', type=str, default='crops', choices=['crops', 'retinawarp'], 
                        help='Image style to process')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for feature extraction')
    parser.add_argument('--model_name', type=str, default='resnet50_ecoset_crop', help='Model name')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing features')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    return parser.parse_args()

def main():
    """Extract neural activations from fixation crops."""
    # Parse arguments
    args = parse_args()
    
    # Configuration
    SUBJECT_ID = args.subject_id
    CROP_SIZE = args.crop_size
    STYLE = args.style
    BATCH_SIZE = args.batch_size
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = args.model_name
    SOURCE = "torchvision"
    PRETRAINED = True
    WEIGHTS_PATH = "/share/klab/datasets/texture2shape_projects/share/ecoset_patches_trained/adult/checkpoint_epoch_300.pth"
    LAYERS = [
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "avgpool",
        "fc"
    ]
    VERBOSE = args.verbose
    OVERWRITE = args.overwrite
    
    print(f"Starting activation extraction for subject {SUBJECT_ID}")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Crop size: {CROP_SIZE}px")
    print(f"Weights path: {WEIGHTS_PATH}")
    
    # Get input directories
    input_dir = get_input_dirs(server="uos")
    
    # Create base output directory
    activations_base_path = os.path.join(PLOTS_DIR_BEHAV, "crop_activations", f"{CROP_SIZE}px")
    os.makedirs(activations_base_path, exist_ok=True)
    
    # Process the subject
    process_subject(SUBJECT_ID, input_dir, activations_base_path, CROP_SIZE, STYLE, BATCH_SIZE,
                   MODEL_NAME, SOURCE, PRETRAINED, WEIGHTS_PATH, LAYERS, VERBOSE, OVERWRITE, DEVICE)
    
    print("\nProcessing complete!")

def process_subject(subject, input_dir, activations_base_path, crop_size, style, batch_size,
                  model_name, source, pretrained, weights_path, layers, verbose, overwrite, device):
    """Process a single subject's data."""
    # Format subject ID with leading zeros
    subject_str = f"as{str(subject).zfill(2)}"
    print(f"\n{'='*80}\nProcessing subject {subject_str}\n{'='*80}")
    
    # Define paths
    crops_dir = os.path.join(input_dir, "fixation_crops")
    crop_image_subdir_name = f"avs_meg_fixation_crops_scene_{crop_size}"
    crops_image_dir = os.path.join(crops_dir, crop_image_subdir_name, style)
    crops_image_dir_sub = os.path.join(crops_image_dir, subject_str)
    
    # Create output directory
    activations_dir = os.path.join(activations_base_path, subject_str, model_name)
    os.makedirs(activations_dir, exist_ok=True)
    
    # Check if input directory exists and has files
    if not os.path.exists(crops_image_dir_sub):
        print(f"Error: Input directory {crops_image_dir_sub} does not exist.")
        return False
    
    if len(os.listdir(crops_image_dir_sub)) == 0:
        print(f"Error: Input directory {crops_image_dir_sub} is empty.")
        return False
    
    try:
        # Get the extractor
        if weights_path and os.path.exists(weights_path):
            # Load custom ResNet-50 model with provided weights
            print(f"Loading model from {weights_path}")
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 565)  # Adjust for EcoSet classes
            
            # Load weights
            checkpoint = torch.load(weights_path, map_location=device)
            # Handle different state dict formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove module. prefix if present
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            
            # Get extractor from loaded model
            from thingsvision import get_extractor_from_model
            extractor = get_extractor_from_model(
                model=model,
                backend="pt",
                device=device
            )
        else:
            # Use standard pretrained model
            print(f"Using standard pretrained {model_name} from {source}")
            extractor = get_extractor(
                model_name=model_name,
                source=source,
                device=device,
                pretrained=pretrained
            )
        
        # Create dataset
        print(f"Creating dataset from: {crops_image_dir_sub}")
        dataset = ImageDataset(
            root=crops_image_dir_sub,
            out_path=activations_dir,
            backend=extractor.get_backend(),
            transforms=extractor.get_transformations()
        )
        
        # Count images
        n_images = len(dataset)
        print(f"Found {n_images} images to process")
        
        if n_images == 0:
            print("Error: No valid images found in the dataset.")
            return False
        
        batches = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            backend=extractor.get_backend()
        )
        
        # Extract features from each layer
        for module_name in layers:
            # Check if already processed
            module_dir = os.path.join(activations_dir, module_name)
            if not overwrite and os.path.exists(module_dir) and os.listdir(module_dir):
                print(f"Skipping {module_name} - already processed")
                continue
            
            print(f"Extracting features from {module_name}")
            os.makedirs(module_dir, exist_ok=True)
            
            try:
                # Extract features
                features = extractor.extract_features(
                    batches=batches,
                    module_name=module_name,
                    flatten_acts=True
                )
                
                # Print feature stats if verbose
                if verbose:
                    print(f"Features shape: {features.shape}")
                    print(f"Features stats - min: {features.min():.4f}, max: {features.max():.4f}, " 
                          f"mean: {features.mean():.4f}, std: {features.std():.4f}")
                
                # Save features
                print(f"Saving features to {module_dir}")
                save_features(features, out_path=module_dir, file_format='hdf5')
                
            except Exception as e:
                print(f"Error extracting features from {module_name}: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up memory
                if 'features' in locals():
                    del features
                torch.cuda.empty_cache()
                gc.collect()
        
        print(f"Processing complete for subject {subject_str}")
        return True
        
    except Exception as e:
        print(f"Error processing subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()