#!/usr/bin/env python3
"""
Main script for computing neural dynamics over time.
This analyzes how multivariate patterns change between timepoints
for different brain regions during visual exploration.
"""

import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Import custom modules
from dynamics_dataloader import load_roi_data_single, optimize_n_jobs
from dynamics_functions import compute_change_over_time

# Import configuration
from avs_gazetime.config import (
    SESSIONS, SUBJECT_ID, PLOTS_DIR, CH_TYPE, SUBJECTS_DIR
)

from dynamics_params import (
    EVENT_TYPE, TIME_WINDOW, CHANGE_METRIC, DELTA_T,
    STRIDE, ROI_GROUPS, HEMI, N_JOBS, remove_erfs
)

def main():
    """
    Main function to run neural dynamics analysis.
    """
    # Convert delta_t from ms to samples
    hz = 500  # Sampling frequency in Hz
    delta_t_samples = int(DELTA_T * hz / 1000)
    
    # Optimize parallel processing
    parallel_jobs = optimize_n_jobs()
    n_jobs = N_JOBS if N_JOBS else parallel_jobs["dynamics_computation"]
    
    print(f"Running dynamics analysis with {n_jobs} parallel jobs")
    print(f"Delta T: {DELTA_T}ms ({delta_t_samples} samples)")
    print(f"Stride: {STRIDE} samples")
    print(f"Change metric: {CHANGE_METRIC}")
    print(f"Hemisphere: {HEMI}")
    
    # Set up paths
    subject = SUBJECT_ID
    sub_name = f"as{str(subject).zfill(2)}"
    
    # Output directory
    output_dir = os.path.join(PLOTS_DIR, "dynamics", sub_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Loop through ROI groups - processing one at a time to save memory
    for roi_name, roi_labels in tqdm(ROI_GROUPS.items(), desc="Processing ROIs"):
        print(f"\nProcessing ROI: {roi_name} with {len(roi_labels)} labels")
        
        # Load data for this ROI only
        meg_data, metadata, times = load_roi_data_single(
            subject, roi_name, roi_labels, EVENT_TYPE, CH_TYPE, 
            SESSIONS, TIME_WINDOW, HEMI, remove_erfs
        )
        
        if meg_data is None:
            print(f"Skipping ROI {roi_name} - no data found")
            continue
            
        print(f"Loaded data shape: {meg_data.shape}")
        
        # Save metadata on first ROI only (it's the same for all ROIs)
        if not os.path.exists(os.path.join(output_dir, f"metadata_{EVENT_TYPE}.csv")):
            metadata.to_csv(os.path.join(output_dir, f"metadata_{EVENT_TYPE}.csv"))
            
            # Save timepoints for dynamics
            times_dynamics = np.array([times[t] for t in range(0, len(times) - delta_t_samples, STRIDE)])
            np.save(os.path.join(output_dir, 
                f"times_{EVENT_TYPE}_{CHANGE_METRIC}_{DELTA_T}ms.npy"), 
                times_dynamics)
        
        # Compute dynamics for this ROI
        print(f"Computing dynamics for {roi_name}")
        dynamics = compute_change_over_time(
            meg_data, delta_t_samples, STRIDE, 
            CHANGE_METRIC, n_jobs, zscore=True
        )
        
        # Save results for this ROI
        print(f"Saving dynamics for {roi_name}, shape: {dynamics.shape}")
        np.save(os.path.join(output_dir, 
            f"dynamics_{EVENT_TYPE}_{roi_name}_{HEMI}_{CHANGE_METRIC}_{DELTA_T}ms.npy"), 
            dynamics)
        
        # Clear memory
        del meg_data, dynamics
    
    print("Analysis complete")

if __name__ == "__main__":
    main()
