#!/usr/bin/env python3
"""
Compute memorability scores for fixation crops using the ResMem model.
This script processes fixation crops and assigns memorability scores that
will later be used to analyze the relationship between memorability and fixation duration.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Import configuration
from avs_gazetime.config import (
    SUBJECT_ID, PLOTS_DIR_BEHAV
)

# Import tools
from avs_machine_room.dataloader.tools.avs_directory_tools import get_input_dirs

# Import ResMem (you'll need to ensure this is properly installed)
try:
    from resmem import ResMem, transformer
except ImportError:
    print("Warning: ResMem not found. Please install resmem package.")
    transformer = None
    ResMem = None


class FixationCropsDataset(Dataset):
    """
    Dataset class for loading fixation crops for memorability scoring.
    """
    def __init__(self, crops_dir, metadata_df, transform=transformer):
        self.crops_dir = crops_dir
        self.metadata_df = metadata_df.copy()
        self.transform = transform if transform is not None else lambda x: x
        
        # Filter for fixation events only
        self.metadata_df = self.metadata_df[self.metadata_df['type'] == 'fixation']
        
        # Drop rows with NaN crop filenames
        self.metadata_df = self.metadata_df.dropna(subset=['crop_filename'])
        
        # Reset index
        self.metadata_df = self.metadata_df.reset_index(drop=True)
        
        print(f"Dataset contains {len(self.metadata_df)} fixation crops")
    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.metadata_df.iloc[idx]['crop_filename']
        
        # Try different possible paths for the image
        possible_paths = [
            os.path.join(self.crops_dir, "crops", f"as{SUBJECT_ID:02d}", img_name)
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Could not find image file: {img_name}")
        
        # Load and transform image
        img = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(img)
        
        return image_tensor, img_name


def compute_memorability_scores(subject_id, crop_size_pix=164, batch_size=32, 
                               num_workers=8, device=None):
    """
    Compute memorability scores for fixation crops using ResMem model.
    
    Parameters:
    -----------
    subject_id : int
        Subject identifier
    crop_size_pix : int
        Size of the crops in pixels
    batch_size : int
        Batch size for DataLoader
    num_workers : int
        Number of workers for DataLoader
    device : str or None
        Device to use ('cuda' or 'cpu'). If None, automatically determined.
        
    Returns:
    --------
    metadata_df : pd.DataFrame
        DataFrame with added memorability scores
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if ResMem is available
    if ResMem is None:
        raise ImportError("ResMem package not found. Please install resmem package.")
    
    # Set up directories
    input_dir = get_input_dirs(server="uos")
    crops_dir = os.path.join(input_dir, "fixation_crops")
    crop_image_subdir_name = f"avs_meg_fixation_crops_scene_{crop_size_pix}"
    crops_image_dir = os.path.join(crops_dir, crop_image_subdir_name)
    
    # Output directory
    memscore_dir = os.path.join(PLOTS_DIR_BEHAV, "memorability_scores")
    os.makedirs(memscore_dir, exist_ok=True)
    
    # Load metadata
    metadata_file = os.path.join(crops_image_dir, "metadata", f"as{subject_id:02d}_crops_metadata.csv")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    metadata_df = pd.read_csv(metadata_file, low_memory=False)
    print(f"Loaded metadata for {len(metadata_df)} events")
    
    # Create dataset and dataloader
    dataset = FixationCropsDataset(crops_image_dir, metadata_df, transform=transformer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=device == 'cuda',
        shuffle=False  # Important: keep order for matching predictions to metadata
    )
    
    # Load ResMem model
    print("Loading ResMem model...")
    model = ResMem(pretrained=True)
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    # Compute memorability scores
    print("Computing memorability scores...")
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for batch_idx, (images, names) in enumerate(tqdm(dataloader, desc="Processing batches")):
            if device == 'cuda' and torch.cuda.is_available():
                images = images.cuda()
            
            # Get predictions
            batch_size = images.shape[0]
            outputs = model(images)
            
            # Average predictions if needed (for multi-scale models)
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                batch_predictions = outputs.mean(dim=1)
            else:
                batch_predictions = outputs.squeeze()
            
            # Convert to CPU numpy
            batch_predictions = batch_predictions.cpu().numpy()
            
            # Handle single item batches
            if batch_predictions.ndim == 0:
                batch_predictions = batch_predictions.reshape(1)
            
            predictions.extend(batch_predictions.tolist())
            filenames.extend(names)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    print(f"Computed memorability scores for {len(predictions)} images")
    
    # Add predictions to metadata
    dataset.metadata_df['mem_score'] = np.nan
    
    # Match predictions to metadata rows
    for pred, filename in zip(predictions, filenames):
        mask = dataset.metadata_df['crop_filename'] == filename
        dataset.metadata_df.loc[mask, 'mem_score'] = pred
    
    # Save results
    output_file = os.path.join(memscore_dir, f"as{subject_id:02d}_crops_metadata_with_memscore_{crop_size_pix}.csv")
    dataset.metadata_df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    
    # Print basic statistics
    print(f"\nMemorability score statistics:")
    print(f"  Count: {len(dataset.metadata_df)}")
    print(f"  Mean: {dataset.metadata_df['mem_score'].mean():.4f}")
    print(f"  Std: {dataset.metadata_df['mem_score'].std():.4f}")
    print(f"  Min: {dataset.metadata_df['mem_score'].min():.4f}")
    print(f"  Max: {dataset.metadata_df['mem_score'].max():.4f}")
    
    return dataset.metadata_df


def compute_memorability_scores_batch(subjects=None, crop_size_pix=164, **kwargs):
    """
    Compute memorability scores for multiple subjects.
    
    Parameters:
    -----------
    subjects : list or None
        List of subject IDs. If None, processes all subjects [1, 2, 3, 4, 5]
    crop_size_pix : int
        Size of crops in pixels
    **kwargs : dict
        Additional arguments passed to compute_memorability_scores
    """
    if subjects is None:
        subjects = [1, 2, 3, 4, 5]
    
    print(f"Computing memorability scores for {len(subjects)} subjects")
    
    all_results = {}
    for subject_id in subjects:
        print(f"\n{'='*50}")
        print(f"Processing Subject {subject_id}")
        print(f"{'='*50}")
        
        try:
            results = compute_memorability_scores(
                subject_id=subject_id,
                crop_size_pix=crop_size_pix,
                **kwargs
            )
            all_results[subject_id] = results
            print(f"Successfully processed Subject {subject_id}")
        except Exception as e:
            print(f"Error processing Subject {subject_id}: {str(e)}")
            continue
    
    print(f"\nCompleted processing {len(all_results)} subjects")
    return all_results


if __name__ == "__main__":
    # Configuration
    SUBJECTS = [1, 2, 3, 4, 5]
    CROP_SIZE_PIX = 100
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    
    # Process all subjects or just the current subject
    if SUBJECT_ID in SUBJECTS:
        # Process just the current subject (useful when running as array job)
        print(f"Processing Subject {SUBJECT_ID} (from environment)")
        results = compute_memorability_scores(
            subject_id=SUBJECT_ID,
            crop_size_pix=CROP_SIZE_PIX,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
    else:
        # Process all subjects
        print("Processing all subjects")
        results = compute_memorability_scores_batch(
            subjects=SUBJECTS,
            crop_size_pix=CROP_SIZE_PIX,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
    
    print("Memorability score computation completed successfully!")