#!/usr/bin/env python3
"""
Script to package AVS scene images into a new subdirectory.
This script copies all scene images used in the AVS dataset into a consolidated directory.
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import avs_machine_room.dataloader.tools.avs_directory_tools as avs_directory

def copy_scene_image(cocoID, coco_dir, output_dir):
    """
    Copy a scene image to the output directory.
    
    Args:
        cocoID: COCO image ID
        coco_dir: Directory containing COCO images (with 12-digit naming)
        output_dir: Target directory for copied images
    
    Returns:
        True if successful, False otherwise
    """
    # Create filename with 12-digit zero-padded naming convention
    filename = str(int(cocoID)).zfill(12) + '_MEG_size.jpg'
    source_path = os.path.join(coco_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(source_path):
        print(f"Warning: Scene {cocoID} not found at {source_path}")
        return False
    
    try:
        shutil.copy2(source_path, output_path)
        return True
    except Exception as e:
        print(f"Error copying {source_path} to {output_path}: {e}")
        return False

def main():
    # Configuration
    avs_scene_selection_path = "/share/klab/datasets/avs/input/scene_sampling_MEG/experiment_cocoIDs.csv"
    input_dir = avs_directory.get_input_dirs(server="uos")
    coco_dir = os.path.join(input_dir, "NSD_scenes_MEG_size_adjusted_925")
    
    # Create output directory
    output_dir = os.path.join(input_dir, "avs_scenes")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"COCO directory: {coco_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load AVS scene selection
    if not os.path.exists(avs_scene_selection_path):
        print(f"Error: AVS scene selection file not found at {avs_scene_selection_path}")
        return
    
    avs_scenes = pd.read_csv(avs_scene_selection_path)
    print(f"Found {len(avs_scenes)} AVS scenes to package")
    
    # Get unique COCO IDs
    coco_ids = avs_scenes['cocoID'].unique()
    print(f"Processing {len(coco_ids)} unique COCO IDs")
    
    # Copy images
    successful_copies = 0
    failed_copies = 0
    
    for i, cocoID in enumerate(coco_ids):
        if i % 100 == 0:
            print(f"Processing image {i+1}/{len(coco_ids)}: {cocoID}")
        
        if copy_scene_image(cocoID, coco_dir, output_dir):
            successful_copies += 1
        else:
            failed_copies += 1
    
    print(f"\nPackaging complete!")
    print(f"Successfully copied: {successful_copies} images")
    print(f"Failed to copy: {failed_copies} images")
    print(f"Output directory: {output_dir}")
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "packaging_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"AVS Scene Images Packaging Summary\n")
        f.write(f"==================================\n\n")
        f.write(f"Source AVS scenes file: {avs_scene_selection_path}\n")
        f.write(f"Source COCO directory: {coco_dir}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        f.write(f"Total scenes to process: {len(coco_ids)}\n")
        f.write(f"Successfully copied: {successful_copies}\n")
        f.write(f"Failed to copy: {failed_copies}\n")
        f.write(f"Success rate: {successful_copies/len(coco_ids)*100:.1f}%\n")
    
    print(f"Summary written to: {summary_path}")

if __name__ == '__main__':
    main()