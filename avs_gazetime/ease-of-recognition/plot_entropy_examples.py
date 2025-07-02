#!/usr/bin/env python3
"""
Plot n_examples images in a row sorted by classification entropy.
Shows examples of fixation targets from high to low entropy (challenging to easy recognition).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from scipy.special import softmax

# Import configuration and parameters  
from avs_gazetime.config import PLOTS_DIR_BEHAV
from avs_machine_room.dataloader.tools.avs_directory_tools import get_input_dirs

# Configuration - matching the ease-of-recognition analysis
SUBJECTS = [1, 2, 3, 4, 5]
CROP_SIZE_PIX = 100
MODEL_NAME = 'resnet50_ecoset_crop'
MODULE_NAMES = ["fc"]


def classification_entropy(features, apply_softmax=True):
    """
    Compute classification entropy from network activations.
    
    Parameters:
    -----------
    features : np.ndarray
        Network activations (n_samples, n_features)
    apply_softmax : bool
        Whether to apply softmax normalisation
        
    Returns:
    --------
    entropy : np.ndarray
        Classification entropy values
    """
    if apply_softmax:
        features = softmax(features, axis=1)
    
    # Avoid log(0) by adding small epsilon
    features = np.maximum(features, np.finfo(float).eps)
    return -np.sum(features * np.log(features), axis=1)


def load_network_activations(subject, crop_size_pix, model_name, module_name):
    """
    Load network activations for a given subject and module.
    
    Parameters:
    -----------
    subject : int
        Subject ID
    crop_size_pix : int
        Crop size in pixels
    model_name : str
        Name of the neural network model
    module_name : str
        Name of the module/layer
        
    Returns:
    --------
    features : np.ndarray
        Network features
    filenames_df : pd.DataFrame
        DataFrame with filenames and feature mappings
    """
    subject_str = f"as{subject:02d}"
    activations_dir = f"{PLOTS_DIR_BEHAV}/crop_activations/{crop_size_pix}px"
    activations_path = os.path.join(activations_dir, subject_str, model_name, module_name)
    
    # Load features from HDF5 file
    features_file = None
    for file in os.listdir(activations_path):
        if file.endswith('.hdf5'):
            features_file = os.path.join(activations_path, file)
            break
    
    if not features_file:
        raise FileNotFoundError(f"No HDF5 file found in {activations_path}")
    
    with h5py.File(features_file, "r") as f:
        features = f["features"][:]
    
    # Load filenames
    txt_fname = os.path.join(activations_dir, subject_str, model_name, "file_names.txt")
    if not os.path.exists(txt_fname):
        # Look for any text file in the activations directory
        txt_files = [f for f in os.listdir(activations_path) if f.endswith('.txt')]
        if txt_files:
            txt_fname = os.path.join(activations_path, txt_files[0])
        else:
            raise FileNotFoundError(f"No filename mapping found for {subject_str}")
    
    with open(txt_fname, "r") as f:
        filenames_full = [line.strip() for line in f.readlines()]
    
    # Parse filenames to extract metadata
    filenames = [os.path.splitext(os.path.basename(filename))[0].split("_") 
                for filename in filenames_full]
    filenames = [[int(x) for x in filename] for filename in filenames]
    
    # Determine column names based on filename structure
    if len(filenames[0]) == 4:
        colnames = ["subject", "trial", "fix_sequence", "sceneID"]
    elif len(filenames[0]) == 5:
        colnames = ["subject", "trial", "fix_sequence", "start_time", "sceneID"]
    else:
        raise ValueError(f"Unexpected filename structure: {filenames[0]}")
    
    filenames_df = pd.DataFrame(filenames, columns=colnames)
    filenames_df["full_filename"] = filenames_full
    
    return features, filenames_df


def plot_entropy_sorted_images(data, crops_image_dir, n_examples=12, save_path=None):
    """
    Plot n_examples images in a row sorted by classification entropy (high to low).
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing entropy scores and crop filenames
    crops_image_dir : str
        Directory containing crop images
    n_examples : int, default=12
        Number of images to display
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Sort by entropy (highest last - most challenging) and select examples
    sorted_data = data.sort_values(by='ease_fc', ascending=True)
    
    # Subsample a random selection of n_examples based on percentile
    percentiles = np.linspace(0, 100, n_examples)
    indices = np.percentile(np.arange(len(sorted_data)), percentiles).astype(int)
    selected_data = sorted_data.iloc[indices]
    
    sns.set_context("poster")
    
    # Create figure
    fig, axes = plt.subplots(1, n_examples, figsize=(n_examples * 1.2, 3))
    
    # Handle single image case
    if n_examples == 1:
        axes = [axes]
    
    for i, (idx, row) in enumerate(selected_data.iterrows()):
        # Construct image path
        image_path = os.path.join(
            crops_image_dir, 
            "crops", 
            f"as{str(row.subject).zfill(2)}", 
            row.crop_filename
        )
        
        try:
            # Load and display image
            image = plt.imread(image_path)
            axes[i].imshow(image)
            axes[i].axis('off')
            
            # Add entropy score as title
            axes[i].set_title(f"{row.ease_fc:.2f}")
            
        except Exception as e:
            print(f"Could not load image {i}: {image_path}")
            axes[i].text(0.5, 0.5, 'Image\nNot Found', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    
    # Make super title
    fig.suptitle("Example fixation targets sorted by classification entropy", 
                )
    
    # Tight layout to avoid overlap
    fig.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def load_entropy_data():
    """Load entropy data from all subjects using the ease-of-recognition pipeline."""
    # Set up directories
    input_dir = get_input_dirs(server="uos")
    crops_dir = os.path.join(input_dir, "fixation_crops")
    crop_image_subdir_name = f"avs_meg_fixation_crops_scene_{CROP_SIZE_PIX}"
    crops_image_dir = os.path.join(crops_dir, crop_image_subdir_name)
    
    all_subjects_data = []
    
    for subject in SUBJECTS:
        print(f"Processing Subject {subject}")
        
        # Load metadata
        metadata_file = os.path.join(crops_image_dir, "metadata", f"as{subject:02d}_crops_metadata.csv")
        metadata_df = pd.read_csv(metadata_file, low_memory=False)
        print(f"Loaded {len(metadata_df)} events")
        
        # Process each module (typically just 'fc' for final classification layer)
        for module_name in MODULE_NAMES:
            print(f"Processing module: {module_name}")
            
            # Load network activations
            features, filenames_df = load_network_activations(
                subject, CROP_SIZE_PIX, MODEL_NAME, module_name
            )
            
            # Compute classification entropy
            entropy_values = classification_entropy(features)
            filenames_df[f"ease_{module_name}"] = entropy_values
            
            # Merge with metadata
            merge_cols = ["full_filename", f"ease_{module_name}"]
            metadata_df = pd.merge(
                metadata_df,
                filenames_df[merge_cols],
                how="inner",
                left_on="crop_filename",
                right_on="full_filename"
            )
            
            print(f"Missing network features: {metadata_df[f'ease_{module_name}'].isna().sum()}")
        
        # Basic filtering - match the ease-of-recognition analysis
        metadata_df = metadata_df[metadata_df.type == 'fixation']
        metadata_df = metadata_df[metadata_df.recording == 'scene']
        metadata_df = metadata_df.dropna(subset=['crop_filename', 'ease_fc'])
        
        # Exclude last fixations  
        metadata_df = metadata_df[metadata_df["fix_sequence_from_last"] < -1]
        
        all_subjects_data.append(metadata_df)
    
    # Combine all subjects
    combined_df = pd.concat(all_subjects_data, ignore_index=True)
    
    # Remove entropy outliers (2nd-98th percentile)
    entropy_col = 'ease_fc'
    combined_df = combined_df.groupby("subject").apply(
        lambda x: x[(x[entropy_col] > x[entropy_col].quantile(0.5)) & 
                   (x[entropy_col] < x[entropy_col].quantile(0.95))]
    ).reset_index(drop=True)
    
    print(f"Final combined dataset: {len(combined_df)} fixations")
    print(f"Entropy range: {combined_df[entropy_col].min():.3f} - {combined_df[entropy_col].max():.3f}")
    print(f"Mean entropy: {combined_df[entropy_col].mean():.3f}")
    
    return combined_df, crops_image_dir


def main():
    """Main function to demonstrate the entropy-sorted plotting."""
    # Load data
    data, crops_image_dir = load_entropy_data()
    
    # Set output directory and filename
    output_dir = os.path.join(PLOTS_DIR_BEHAV, "ease_of_recognition", MODEL_NAME)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"entropy_sorted_15_examples_{CROP_SIZE_PIX}px.pdf")
    
    # Create plot
    fig = plot_entropy_sorted_images(
        data=data,
        crops_image_dir=crops_image_dir,
        n_examples=12,
        save_path=save_path
    )
    
    print(f"Figure saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()