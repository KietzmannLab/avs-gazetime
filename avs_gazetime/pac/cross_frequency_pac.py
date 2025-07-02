#!/usr/bin/env python3
"""
Memory-optimized script for ROI-based cross-frequency PAC computation with surrogate testing.
Processes one ROI at a time, loading only the relevant channel indices to conserve RAM.
Supports multiple surrogate methods for robust statistical testing.
Features incremental saving of vertex results and the ability to resume interrupted jobs.
"""

import os
import sys
import numpy as np
import mne
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse

import avs_gazetime.utils.load_data as load_data
from avs_gazetime.config import (
    SESSIONS, SUBJECT_ID, PLOTS_DIR, CH_TYPE, SUBJECTS_DIR, MEG_DIR 
)

from params_pac import (
    EVENT_TYPE, remove_erfs, TIME_WINDOW, FF_BAND_PHASE, FF_BAND_AMPLITUDE, 
    PAC_METHOD, THETA_STEPS, GAMMA_STEPS, SURROGATE_STYLE
)

# Import data loader and PAC functions
from pac_dataloader import load_meg_data, optimize_n_jobs
from pac_functions import compute_full_cross_frequency_pac_matrix

# Get optimized parallel job settings
PARALLEL_JOBS = optimize_n_jobs()

def get_roi_vertices(subject_id, roi_group, roi_labels, subjects_dir=None):
    """
    Get valid vertex indices for specified ROIs that exist in the source estimate.
    
    Parameters:
    -----------
    subject_id : int
        Subject ID number
    roi_group : str
        Name of ROI group (used for output organization)
    roi_labels : list
        List of ROI labels to include
    subjects_dir : str, optional
        Path to subjects directory
    
    Returns:
    --------
    vertices_dict : dict
        Dictionary with ROI group as key and hemisphere vertices as values
    """
    if subjects_dir is None:
        subjects_dir = SUBJECTS_DIR
    
    vertices_dict = {roi_group: {'lh': [], 'rh': []}}
    sub_name = f"as{subject_id:02d}"
    
    # Load subject's source estimate to find valid vertices
    method = "beamformer"  # Using beamformer as shown in example
    stc_fname = f"{MEG_DIR}/{sub_name}a_stcs_saccade"
   
    stc = mne.read_source_estimate(stc_fname)
    print(f"Loaded source estimate from {stc_fname}")
    
    for hemi in ["lh", "rh"]:
        l_short = "R" if hemi == "rh" else "L"
        
        for roi in roi_labels:
            label_fname = os.path.join(subjects_dir, sub_name, "label", f"{hemi}.{l_short}_{roi}_ROI.label")
            if os.path.exists(label_fname):
                label = mne.read_label(label_fname)
                
                # If stc is available, use in_label to get valid vertices
                if stc is not None:
                    # Get stc in this label
                    stc_in_label = stc.in_label(label)
                    # Extract the valid vertices for this hemisphere
                    hemi_idx = 0 if hemi == "lh" else 1
                    valid_vertices = stc_in_label.vertices[hemi_idx]
                    # get the index of the vertices in the stc
                    vertices_idx = np.where(np.isin(stc.vertices[hemi_idx], valid_vertices))[0]
                    if hemi == "rh":
                        vertices_idx += stc.vertices[0].size
                    print(f"Found {len(vertices_idx)} vertices for {roi} in {label_fname}")
                    vertices_dict[roi_group][hemi].extend(vertices_idx)
             
            else:
                print(f"No label found for {roi} in {label_fname}")
    # make sure the vertices are unique and sorted
    for hemi in ["lh", "rh"]:
        
        vertices_dict[roi_group][hemi].sort()
        #assert no duplicates
        assert len(vertices_dict[roi_group][hemi]) == len(set(vertices_dict[roi_group][hemi])), \
            f"Duplicate vertices found in {roi_group} {hemi} for {roi_labels}"
    return vertices_dict

def process_roi_chunk(area, hemi, vertices, event_type, ch_type, sessions, remove_erfs, time_window, 
                     phase_band, amp_band, method, steps_phase, steps_amp, surrogate_style, 
                     output_dir=None, overwrite=False):
    """
    Process a single ROI chunk with specified surrogate method, with incremental saving.
    
    Parameters:
    -----------
    area : str
        ROI area name
    hemi : str
        Hemisphere ('lh' or 'rh')
    vertices : array
        Array of vertex indices to process
    event_type : str
        Type of events to analyze ('fixation' or 'saccade')
    ch_type : str
        Channel type ('mag', 'grad', or 'stc')
    sessions : list
        List of session numbers to include
    remove_erfs : list
        List of ERF types to remove
    time_window : tuple
        Analysis time window (start, end) in seconds
    phase_band : tuple
        Frequency range for phase (min, max)
    amp_band : tuple
        Frequency range for amplitude (min, max)
    method : str
        PAC computation method
    steps_phase : int
        Step size for phase frequency
    steps_amp : int
        Step size for amplitude frequency
    surrogate_style : str
        Method for surrogate generation ('phase_shuffle', 'session_aware', or 'single_cut')
    output_dir : str, optional
        Directory to save results. If None, results are not saved incrementally.
    overwrite : bool, default=False
        Whether to overwrite existing results files.
    
    Returns:
    --------
    vertex_ff_maps : dict
        Dictionary mapping vertex indices to PAC matrices
    """
    print(f"Processing {len(vertices)} vertices in {area} {hemi} using {surrogate_style} surrogate method")
    
    # Create output directory for this ROI and hemisphere
    area_dir = None
    if output_dir is not None:
        area_dir = os.path.join(output_dir, f"{area}_{hemi}")
        os.makedirs(area_dir, exist_ok=True)
    
    # Check which vertices need processing (skip existing unless overwrite=True)
    vertices_to_process = []
    vertex_indices_to_process = []
    
    for idx, vertex in enumerate(vertices):
        # Determine if this vertex needs processing
        needs_processing = True
        if area_dir is not None:
            vertex_filename = os.path.join(area_dir, f"vertex_{vertex}_ffmap_{surrogate_style}.npy")
            if os.path.exists(vertex_filename) and not overwrite:
                print(f"Skipping vertex {vertex} - result file already exists")
                needs_processing = False
        
        if needs_processing:
            vertices_to_process.append(vertex)
            vertex_indices_to_process.append(idx)
    
    # If all vertices are already processed, return early
    if not vertices_to_process:
        print(f"All vertices in {area} {hemi} are already processed. Use overwrite=True to recompute.")
        return {}
    
    print(f"Processing {len(vertices_to_process)} out of {len(vertices)} vertices")
    
    # Memory-efficient approach: Load only the data for vertices that need processing
    meg_data, merged_df, times = load_meg_data(
        event_type=event_type,
        ch_type=ch_type,
        sessions=sessions,
        channel_idx=[vertices[i] for i in vertex_indices_to_process],  # Only load needed vertices
        remove_erfs=remove_erfs
    )
    
    # Get durations and sessions
    dur_col = "duration" if event_type == "fixation" else "associated_fixation_duration"
    durations = merged_df[dur_col].values
    sessions_array = merged_df["session"].values
    
    print(f"Loaded data shape: {meg_data.shape} for {len(vertices_to_process)} vertices")
    
    # Track processed vertices and results
    vertex_ff_maps = {}
    
    # Process each vertex individually to allow incremental saving
    for local_idx, (global_idx, vertex) in enumerate(tqdm(
            zip(vertex_indices_to_process, vertices_to_process), 
            desc=f"Computing FF maps for {area} {hemi}", 
            total=len(vertices_to_process))):
        
        try:
            # Process this vertex
            result = compute_full_cross_frequency_pac_matrix(
                meg_data, 500, [local_idx], phase_band, amp_band, times, time_window,
                n_bootstraps=200, method=method, random_seed=42,
                steps_theta=steps_phase, steps_gamma=steps_amp,
                durations=durations, sessions=sessions_array,
                surrogate_style=surrogate_style
            )
            
            # The result is a dict with key local_idx, extract the actual data
            ff_map = result[local_idx]
            vertex_ff_maps[vertex] = ff_map
            
            # Save result for this vertex immediately
            if area_dir is not None:
                vertex_filename = os.path.join(area_dir, f"vertex_{vertex}_ffmap_{surrogate_style}.npy")
                np.save(vertex_filename, ff_map)
                print(f"Saved result for vertex {vertex}")
                
        except Exception as e:
            print(f"Error processing vertex {vertex}: {str(e)}")
            continue
        
        # Force garbage collection after each vertex to minimize memory usage
        import gc
        gc.collect()
    
    return vertex_ff_maps

def main():
    """Main function to run memory-optimized ROI-based FF-PAC analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ROI-based cross-frequency PAC computation with surrogate testing")
    parser.add_argument("roi_name", help="Name of the ROI group")
    parser.add_argument("roi_definition", help="Comma-separated list of ROI labels")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing vertex result files")
    
    args = parser.parse_args()
    
    roi_name = args.roi_name
    roi_labels = args.roi_definition.split(',')
    overwrite = args.overwrite
    
    # Get surrogate settings from params_pac.py
    surrogate_style = SURROGATE_STYLE

    
    print(f"Running with optimized parallel job settings: {PARALLEL_JOBS}")
    print(f"Processing ROI {roi_name} with labels: {roi_labels}")
    print(f"Using surrogate method: {surrogate_style} for FF-PAC computation")
    print(f"Overwrite existing results: {overwrite}")
    
    # Process single subject from SUBJECT_ID
    subject = SUBJECT_ID
    print(f"\nProcessing subject {subject}")
    
    # Define analysis parameters
    phase_band = FF_BAND_PHASE
    amp_band = FF_BAND_AMPLITUDE
    time_window = TIME_WINDOW
    steps_phase = THETA_STEPS
    steps_amp = GAMMA_STEPS
    
    # Get ROI vertices for just this subject and ROI
    vertices_dict = get_roi_vertices(subject, roi_name, roi_labels)
    print(vertices_dict)
   
    # Setup output directory - include surrogate style in path
    phase_band_str = f"{phase_band[0]}-{phase_band[1]}"
    amp_band_str = f"{amp_band[0]}-{amp_band[1]}"
    time_win_str = f"{time_window[0]}-{time_window[1]}"
    
    output_dir = os.path.join(
        PLOTS_DIR, 
        f"roi_ffpac_{SUBJECT_ID}_{CH_TYPE}_{EVENT_TYPE}_{phase_band_str}_{amp_band_str}_{time_win_str}_{surrogate_style}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each hemisphere for this ROI
    for area in vertices_dict:
        for hemi in vertices_dict[area]:
            if not vertices_dict[area][hemi]:
                continue
                
            print(f"\n==== Processing {area} {hemi} with {surrogate_style} surrogate ====")
            
            # Process this ROI chunk with incremental saving
            process_roi_chunk(
                area=area,
                hemi=hemi,
                vertices=np.array(vertices_dict[area][hemi]),
                event_type=EVENT_TYPE,
                ch_type=CH_TYPE,
                sessions=SESSIONS,
                remove_erfs=remove_erfs,
                time_window=time_window,
                phase_band=phase_band,
                amp_band=amp_band,
                method=PAC_METHOD,
                steps_phase=steps_phase,
                steps_amp=steps_amp,
                surrogate_style=surrogate_style,
                output_dir=output_dir,   # Enable incremental saving
                overwrite=overwrite      # Control whether to recompute existing results
            )
            
            print(f"Completed processing {area} {hemi} using {surrogate_style} surrogate method")
            
            # Force garbage collection
            import gc
            gc.collect()

if __name__ == "__main__":
    main()