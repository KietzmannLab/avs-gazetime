#!/usr/bin/env python3
"""
Plot n_examples images in a row sorted by memorability.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import configuration and parameters
from avs_gazetime.config import PLOTS_DIR_BEHAV
from avs_gazetime.memorability.memorability_analysis_params import *
from avs_machine_room.dataloader.tools.avs_directory_tools import get_input_dirs


def plot_memorability_sorted_images(data, crops_image_dir, n_examples=12, save_path=None):
    """
    Plot n_examples images in a row sorted by memorability.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing memorability scores and crop filenames
    crops_image_dir : str
        Directory containing crop images
    n_examples : int, default=15
        Number of images to display
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Sort by memorability (highest first) and select examples
    sorted_data = data.sort_values(by='mem_score', ascending=True)
    # subsample a random selection of n_examples based on percentile
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
            
            # Add memorability score as title
            axes[i].set_title(f"{row.mem_score:.2f}")
            
        except Exception as e:
            print(f"Could not load image {i}: {image_path}")
            axes[i].text(0.5, 0.5, 'Image\nNot Found', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    # make super tittle
    fig.suptitle(f"Example fixation targets sorted by memorability")
    # tight layout to avoid overlap
    fig.tight_layout()
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def load_memorability_data():
    """Load memorability data from all subjects."""
    # Set up directories
    input_dir = get_input_dirs(server="uos")
    memscore_dir = os.path.join(PLOTS_DIR_BEHAV, "memorability_scores")
    crops_dir = os.path.join(input_dir, "fixation_crops")
    crop_image_subdir_name = f"avs_meg_fixation_crops_scene_{CROP_SIZE_PIX}"
    crops_image_dir = os.path.join(crops_dir, crop_image_subdir_name)
    
    all_subjects_data = []
    
    for subject in SUBJECTS:
        # Load metadata with memorability scores
        metadata_df = pd.read_csv(
            os.path.join(memscore_dir, f"as{str(subject).zfill(2)}_crops_metadata_with_memscore_{CROP_SIZE_PIX}.csv"),
            low_memory=False
        )
        
        # Basic filtering
        metadata_df = metadata_df[metadata_df.type == 'fixation']
        metadata_df = metadata_df[metadata_df.recording == 'scene']
        metadata_df = metadata_df.dropna(subset=['crop_filename', 'mem_score'])
        
        all_subjects_data.append(metadata_df)
    
    # Combine all subjects
    combined_df = pd.concat(all_subjects_data, ignore_index=True)
    
    return combined_df, crops_image_dir


def main():
    """Main function to demonstrate the plotting."""
    # Load data
    data, crops_image_dir = load_memorability_data()
    # Set output directory and filename
    output_dir = os.path.join(PLOTS_DIR_BEHAV, "memorability_analysis")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"memorability_sorted_15_examples_{CROP_SIZE_PIX}px.pdf")
    
    # Create plot
    fig = plot_memorability_sorted_images(
        data=data,
        crops_image_dir=crops_image_dir,
        n_examples=15,
        save_path=save_path
    )
    
    print(f"Figure saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()