"""
Module for loading and preprocessing MEG data for PAC analysis.
"""
import numpy as np
import pandas as pd
import avs_gazetime.utils.load_data as load_data

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
        List of ERF types to remove. Options:
        - 'saccade': Remove pre-saccade ERF
        - 'fixation': Remove fixation ERF (after aligning to fixation onset)
        - 'saccade_post': Remove post-saccade ERF (subsequent saccade)
        If None, no ERFs are removed.

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
    if remove_erfs and 'saccade' in remove_erfs:
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

        # Remove saccade ERF if specified
        if "saccade" in remove_erfs:
            print("Removing saccade ERF")
            sacc_erf = np.median(meg_data, axis=0)
            meg_data = meg_data - sacc_erf

        # Calculate sampling frequency for time shifting
        S_FREQ = 500  # Sampling frequency in Hz

        # Get saccade durations and calculate shifts
        saccade_duration = merged_df["duration"].values
        n_shifts_per_event = (saccade_duration * S_FREQ).astype(int)

        # Shift data to align with fixation onset
        meg_data = np.array([np.roll(meg_data[i], -n_shifts_per_event[i], axis=-1) for i in range(meg_data.shape[0])])

        # Remove fixation ERF if specified
        if "fixation" in remove_erfs:
            print("Removing fixation ERF")
            fix_erf = np.median(meg_data, axis=0)
            meg_data = meg_data - fix_erf

        # Remove post-saccade ERF if specified (subsequent saccade)
        if "saccade_post" in remove_erfs:
            print("Removing post-saccade ERF (subsequent saccade)")
            # Get fixation durations to roll backwards to subsequent saccade onset
            fixation_durations = merged_df["associated_fixation_duration"].values
            n_shifts_fixation = (fixation_durations * S_FREQ).astype(int)

            # Roll backwards by fixation duration to align to subsequent saccade onset
            meg_data_rolled = np.array([np.roll(meg_data[i], -n_shifts_fixation[i], axis=-1) for i in range(meg_data.shape[0])])

            # Remove the post-saccade ERF
            post_sacc_erf = np.median(meg_data_rolled, axis=0)
            meg_data_rolled = meg_data_rolled - post_sacc_erf

            # Roll forward to get back to fixation onset
            meg_data = np.array([np.roll(meg_data_rolled[i], n_shifts_fixation[i], axis=-1) for i in range(meg_data_rolled.shape[0])])

        # Remove events without associated fixation duration
        valid_fixation_mask = ~np.isnan(merged_df["associated_fixation_duration"])
        merged_df = merged_df[valid_fixation_mask]
        meg_data = meg_data[valid_fixation_mask]

        # Set appropriate duration column
        dur_col = "associated_fixation_duration"
    else:
        # Load metadata for the specified event type
        merged_df = load_data.merge_meta_df(event_type)

        # Load MEG data
        meg_data = load_data.process_meg_data_for_roi(ch_type, event_type, sessions, apply_median_scale=True, channel_idx=channel_idx)

        # Set appropriate duration column
        dur_col = "duration" if event_type == "fixation" else "associated_fixation_duration"
    
    # Filter by duration to remove outliers
    longest_dur = np.percentile(merged_df[dur_col], 98)
    shortest_dur = np.percentile(merged_df[dur_col], 2)
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

def get_channel_chunks(ch_type, event_type, channel_chunk=None):
    """
    Get channel chunks for processing.
    
    Parameters:
    -----------
    ch_type : str
        Channel type ('mag', 'grad', or 'stc').
    event_type : str
        Event type ('fixation' or 'saccade').
    channel_chunk : list or None
        Channel chunk specification from command line.
        
    Returns:
    --------
    channels : list
        List of channel indices to process.
    """
    if ch_type == "stc" and channel_chunk is not None:
        # Check available channels
        channels = load_data.check_available_channels("a", ch_type, event_type)
        # Subselect the channels based on the chunk
        num_chunks = int(channel_chunk[1])
        chunk_idx = int(channel_chunk[0])
        chunk_size = len(channels) // num_chunks
        start_idx = chunk_idx * chunk_size
        end_idx = (chunk_idx + 1) * chunk_size
        if chunk_idx == num_chunks - 1:
            end_idx = len(channels)
        return [channels[start_idx:end_idx]]
    else:
        return [None]
    
def optimize_n_jobs(debug=True):
    """
    Optimized job allocation for high-performance computing environments (100+ CPUs).
    Returns a dictionary with optimized settings for different computation contexts.
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
    # From 10% at 8 cores down to 2% at 128+ cores
    reserved_percentage = max(0.02, min(0.1, 0.1 * (32 / max(32, available_cores))))
    reserved_cores = max(1, int(available_cores * reserved_percentage))
    usable_cores = max(1, available_cores - reserved_cores)
    
    jobs_dict = {
    # Filter jobs: 10-20% of cores, minimum 2, maximum 24
    "filter": max(2, min(24, int(usable_cores * 0.15))),  # 15% of cores = 6
    
    # PAC computation: 40-60% of cores, scales with core count
    "pac_computation": max(4, min(int(usable_cores * 0.5), usable_cores - 10)),  # 50% = 20 cores
    
    # Bootstrap: 20-30% of cores, scales with system size
    "bootstrap": max(4, min(int(usable_cores * 0.25), 20)),  # 25% = 10 cores
    
    # Cross-frequency: 70-90% of cores, with minimum overhead reserved
    "cross_freq": max(8, min(usable_cores - 4, int(usable_cores * 0.8))),  # 80% = 32 cores
    
    # Batch size: scales with core count to optimize throughput
    "batch_size": max(5, min(30, int(usable_cores / 8) + 4))  # More optimal scaling = 9
    }
    # Print detected environment and settings
    print(f"Detected {available_cores} cores, using {usable_cores} for computation")
    print(f"Job allocation: {jobs_dict}")
    
    if debug:
        # Debug mode: set all jobs to 1 core
        jobs_dict = {key: 1 for key in jobs_dict.keys()}
        print("Running in debug mode, setting all jobs to 1 core")
    
    return jobs_dict