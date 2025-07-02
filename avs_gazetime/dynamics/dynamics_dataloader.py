"""
Module for loading MEG data for neural dynamics analysis.
"""
import os
import numpy as np
import mne
from joblib import Parallel, delayed

import avs_gazetime.utils.load_data as load_data
from avs_gazetime.config import (
    SESSIONS, SUBJECT_ID, PLOTS_DIR, CH_TYPE, SUBJECTS_DIR
)

def get_roi_vertices(subject_id, roi_name, roi_labels, hemi="both", subjects_dir=None):
    """
    Get valid vertex indices for specified ROIs that exist in the source estimate.
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (without "as" prefix).
    roi_name : str
        Name of the ROI group.
    roi_labels : list
        List of ROI labels to include.
    hemi : str
        Hemisphere to use ('lh', 'rh', or 'both').
    subjects_dir : str
        Directory containing subject data.
        
    Returns:
    --------
    vertices_dict : dict
        Dictionary with hemispheres as keys and vertex indices as values.
    """
    if subjects_dir is None:
        subjects_dir = SUBJECTS_DIR
    
    from avs_gazetime.config import MEG_DIR

    vertices_dict = {}
    sub_name = f"as{subject_id:02d}"
    
    # Load subject's source estimate to find valid vertices
    stc_fname = os.path.join(MEG_DIR, f"{sub_name}a_stcs_saccade")
    try:
        stc = mne.read_source_estimate(stc_fname)
        print(f"Loaded source estimate from {stc_fname}")
    except Exception as e:
        print(f"Error loading source estimate from {stc_fname}: {e}")
        print("Will try to proceed with label vertices only")
        stc = None
    
    hemis = ["lh", "rh"] if hemi == "both" else [hemi]
    
    for h in hemis:
        vertices_dict[h] = []
        l_short = "R" if h == "rh" else "L"
        
        for roi in roi_labels:
            label_fname = os.path.join(subjects_dir, sub_name, "label", f"{h}.{l_short}_{roi}_ROI.label")
            if os.path.exists(label_fname):
                label = mne.read_label(label_fname)
                
              
                # Get stc in this label
                stc_in_label = stc.in_label(label)
                # Extract the valid vertices for this hemisphere
                hemi_idx = 0 if h == "lh" else 1
                valid_vertices = stc_in_label.vertices[hemi_idx]
                # Get the indices of these vertices in the original stc
                vertices_idx = np.where(np.isin(stc.vertices[hemi_idx], valid_vertices))[0]
                print(f"Found {len(vertices_idx)} valid vertices for {roi} in {h}")
                # please note that stc.data is has shape (n_vertices, n_times), whereas the stc.vertices is separated by hemispheres with lh first and rh second
                # this means that we need to add the number of vertices in the left hemisphere to get the correct index for the right hemisphere

                if hemi == "rh":
                    vertices_idx += len(stc.vertices[0])
                vertices_dict[h].extend(vertices_idx.tolist())
               
            else:
                print(f"No label found for {roi} in {label_fname}")
    # assert that the vertices are unique and sorted
    for h in hemis:
        # assert for duplicates
        n_duplicates = len(vertices_dict[h]) - len(set(vertices_dict[h]))
        if n_duplicates > 0:
            roi_name_str = f"{roi_name} {h}" if roi_name else f"{h}"
            ValueError(f"Warning: Found {n_duplicates} duplicate vertices for {roi_name_str}. Removing duplicates.")
        
        # sort the vertices
        vertices_dict[h] = sorted(set(vertices_dict[h]))
    # If no vertices were found, return None
    if all(len(vertices) == 0 for vertices in vertices_dict.values()):
        return None
        
    return vertices_dict

def load_meg_data(event_type, ch_type, sessions, channel_idx=None, remove_erfs=None):
    """
    Load MEG data and apply necessary preprocessing steps including optional ERF removal.
    
    Parameters:
    -----------
    event_type : str
        Type of events to load ('fixation' or 'saccade').
    ch_type : str
        Channel type ('mag', 'grad', or 'stc').
    sessions : list
        List of sessions to include.
    channel_idx : list or None
        Indices of channels to include. If None, all channels are included.
    remove_erfs : list or None
        List of ERF types to remove ('saccade', 'fixation'). If None, no ERFs are removed.
        
    Returns:
    --------
    meg_data : np.ndarray
        Preprocessed MEG data of shape (n_epochs, n_channels, n_times).
    merged_df : pd.DataFrame
        DataFrame containing metadata for the epochs.
    times : np.ndarray
        Timepoints corresponding to the MEG data.
    """
    # Load the appropriate metadata based on event type
    if event_type == "saccade":
        # Load both fixation and saccade data for alignment
        merged_df_saccade = load_data.merge_meta_df("saccade")
        merged_df_fixation = load_data.merge_meta_df("fixation")
        merged_df = load_data.match_saccades_to_fixations(merged_df_fixation, merged_df_saccade, saccade_type="pre-saccade")
        
        # Load MEG data for saccades
        meg_data = load_data.process_meg_data_for_roi(ch_type, "saccade", sessions, apply_median_scale=True, channel_idx=channel_idx)
        
        # Apply index mask to meg data
        meg_data = meg_data[merged_df.index, :, :]
        
        # Reset index after masking
        merged_df.reset_index(drop=True, inplace=True)
    else:
        # Load metadata for the specified event type
        merged_df = load_data.merge_meta_df(event_type)
        
        # Load MEG data
        meg_data = load_data.process_meg_data_for_roi(ch_type, event_type, sessions, apply_median_scale=True, channel_idx=channel_idx)
            
    if remove_erfs and 'saccade' in remove_erfs:
       
        # Remove saccade ERF if specified
        if "saccade" in remove_erfs:
            print("Removing saccade ERF")
            sacc_erf = np.median(meg_data, axis=0)
            meg_data = meg_data - sacc_erf
        
        # Calculate sampling frequency for time shifting
        S_FREQ = 500  # Sampling frequency in Hz
        
        # Get saccade durations and calculate shifts
        saccade_duration = merged_df_saccade["duration"].values
        n_shifts_per_event = (saccade_duration * S_FREQ).astype(int)
        
        # Shift data to align with fixation onset
        meg_data = np.array([np.roll(meg_data[i], -n_shifts_per_event[i]) for i in range(meg_data.shape[0])])
        
        # Remove fixation ERF if specified
        if "fixation" in remove_erfs:
            print("Removing fixation ERF")
            fix_erf = np.median(meg_data, axis=0)
            meg_data = meg_data - fix_erf
        
        # Remove events without associated fixation duration
        valid_fixation_mask = ~np.isnan(merged_df["associated_fixation_duration"])
        merged_df = merged_df[valid_fixation_mask]
        meg_data = meg_data[valid_fixation_mask]
        
        
  
        
    dur_col = "duration" if event_type == "fixation" else "associated_fixation_duration"
        
    print("using duration column: ", dur_col)
    # Filter by duration to remove outliers
    longest_dur = np.percentile(merged_df[dur_col], 98)
    print("longest duration: ", longest_dur)
    shortest_dur = np.percentile(merged_df[dur_col], 2)
    print("shortest duration: ", shortest_dur)
    dur_mask = (merged_df[dur_col] < longest_dur) & (merged_df[dur_col] > shortest_dur)
    merged_df = merged_df[dur_mask]
    meg_data = meg_data[dur_mask, :, :]
    
    # Reset index after duration filtering
    merged_df.reset_index(drop=True, inplace=True)
    
    # Load time points
    times = load_data.read_hd5_timepoints(event_type=event_type)
    
    # Special adjustments for saccade events
    if event_type == "saccade":
        # Change type column entries to fixation and fill start_time with associated_start_time
        merged_df["type"] = "fixation"
        merged_df["start_time"] = merged_df["associated_fix_start_time"]
    
    return meg_data, merged_df, times

def optimize_n_jobs(debug=False):
    """
    Optimizes parallel job allocation based on available CPU cores.
    """
    import os
    import psutil

    # Get number of available CPU cores
    cpu_count = os.cpu_count()
    
    try:
        # Try to get affinity if supported
        available_cores = len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), 'cpu_affinity') else cpu_count
    except (AttributeError, ImportError):
        # Fall back to CPU count if psutil isn't available or doesn't support affinity
        available_cores = cpu_count
    
    # Scale reserved percentage based on core count (less overhead with more cores)
    reserved_percentage = max(0.02, min(0.1, 0.1 * (32 / max(32, available_cores))))
    reserved_cores = max(1, int(available_cores * reserved_percentage))
    usable_cores = max(1, available_cores - reserved_cores)
    
    # Create a dictionary with settings that scale with available cores
    jobs_dict = {
        "dynamics_computation": max(2, min(usable_cores - 2, int(usable_cores * 0.8))),
        "batch_size": max(5, min(30, int(usable_cores / 12) + 5))
    }
    
    if debug:
        # Debug mode: set all jobs to 1 core
        jobs_dict = {key: 1 for key in jobs_dict.keys()}
    
    return jobs_dict

def load_roi_data_single(subject_id, roi_name, roi_labels, event_type, ch_type, sessions, time_window, hemi="both", remove_erfs=None):
    """
    Load MEG data for a single ROI to minimize memory usage.
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (without "as" prefix).
    roi_name : str
        Name of the ROI group.
    roi_labels : list
        List of ROI labels for this ROI group.
    event_type : str
        Type of events to load ('fixation' or 'saccade').
    ch_type : str
        Channel type ('mag', 'grad', or 'stc').
    sessions : list
        List of sessions to include.
    time_window : tuple
        Time window to analyze (start, end) in seconds.
    hemi : str
        Hemisphere to use.
    remove_erfs : list
        List of ERF types to remove.
        
    Returns:
    --------
    roi_data : np.ndarray
        MEG data for this ROI.
    merged_df : pd.DataFrame
        DataFrame containing metadata for the epochs.
    times : np.ndarray
        Timepoints corresponding to the MEG data.
    """
    # For stc data, we always load saccade-locked data regardless of event_type
    actual_event_type = "saccade" if ch_type == "stc" else event_type
    
    if ch_type == "stc":
        #try:
        # Source space analysis - load ROI vertices
        vertices_dict = get_roi_vertices(subject_id, roi_name, roi_labels, hemi)
        
        if vertices_dict is None:
            print(f"No vertices found for ROI {roi_name}")
            return None, None, None
        
        # Process each hemisphere separately and then combine
        meg_data_list = []
        for h, verts in vertices_dict.items():
            if not verts:
                print(f"No vertices found for {roi_name} in {h}")
                continue
            
            # Remove duplicates and sort
            vertices = np.array(sorted(set(verts)))
            
            if len(vertices) == 0:
                print(f"No vertices found for ROI {roi_name} in {h}")
                continue
                
            print(f"Loading data for {len(vertices)} vertices in {roi_name} {h}")
            
            #try:
                # Load saccade-locked data for specific vertices
            hemi_data, merged_df, times = load_meg_data(
                    actual_event_type, ch_type, sessions, 
                    channel_idx=vertices, remove_erfs=remove_erfs
                )
            meg_data_list.append(hemi_data)
            print(f"Successfully loaded data for {roi_name} {h} with shape {hemi_data.shape}")
            #except Exception as e:
                #print(f"Error loading data for {roi_name} {h}: {e}")
                # print(f"Vertices indices: min={vertices.min()}, max={vertices.max()}")
            #    continue
        
        if not meg_data_list:
            print(f"Failed to load any data for ROI {roi_name}")
            return None, None, None
        
        # Combine data from all hemispheres
        meg_data = np.concatenate(meg_data_list, axis=1)
        print(f"Combined data shape for {roi_name}: {meg_data.shape}")
        
        # If the requested event_type is fixation but we loaded saccade data, 
        # we need to roll the data to align with fixation onset
        if event_type == "fixation" and actual_event_type == "saccade" and not remove_erfs:
            print("Aligning saccade-locked source data to fixation onset")
            # Calculate sampling frequency for time shifting
            S_FREQ = 500  # Sampling frequency in Hz
            
            sacca_dur_col = "duration" if actual_event_type == "saccade" else "associated_fixation_duration"
            saccade_duration = merged_df[sacca_dur_col].values
            # Calculate shifts
            n_shifts_per_event = (saccade_duration * S_FREQ).astype(int)
            
            # Shift data to align with fixation onset
            meg_data = np.array([np.roll(meg_data[i], -n_shifts_per_event[i], axis=1) 
                                for i in range(meg_data.shape[0])])
    #except Exception as e:
        #print(f"Error in source space loading for {roi_name}: {e}")
    #    return None, None, None
    else:
        # Sensor space analysis - load all sensors of specified type
        meg_data, merged_df, times = load_meg_data(event_type, ch_type, sessions, remove_erfs=remove_erfs)
    
    # Filter time window
    times_mask = (times >= time_window[0]) & (times <= time_window[1])
    times = times[times_mask]
    meg_data = meg_data[:, :, times_mask]
    
    return meg_data, merged_df, times