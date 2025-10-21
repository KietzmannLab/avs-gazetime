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
    if remove_erfs and len(remove_erfs) > 0:
        # Load both fixation and saccade data for alignment
        merged_df_saccade = load_data.merge_meta_df("saccade")
        merged_df_fixation = load_data.merge_meta_df("fixation")

        # Use extended matching if fixation_post is requested
        if "fixation_post" in remove_erfs:
            merged_df = load_data.match_saccades_to_fixations_extended(
                merged_df_saccade, merged_df_fixation, saccade_type="pre-saccade"
            )
        else:
            merged_df = load_data.match_saccades_to_fixations(
                merged_df_saccade, merged_df_fixation, saccade_type="pre-saccade"
            )

        # Load MEG data for saccades
        meg_data = load_data.process_meg_data_for_roi(ch_type, "saccade", sessions, apply_median_scale=True, channel_idx=channel_idx, scale_with_std=False)

        # Apply index mask to meg data
        meg_data = meg_data[merged_df.index, :, :]
        
        
        # Reset index after masking
        merged_df.reset_index(drop=True, inplace=True)
        
        # max per epoch 
        max_per_epoch = np.max(np.abs(meg_data), axis=(1, 2))
        threshold = np.percentile(max_per_epoch, 99)
        good_epochs = max_per_epoch < threshold

        print(f"Rejecting {(~good_epochs).sum()} / {len(good_epochs)} epochs (max amplitude > {threshold:.3f})")
        
      
        # Apply rejection to both MEG data and behavioral dataframe
        meg_data = meg_data[good_epochs]
        merged_df = merged_df[good_epochs].reset_index(drop=True)
        
        # remove epochs that dont correlate well with the median
        # Calculate median ERF for the epochs
        median_erf = np.median(meg_data, axis=0)
        # Calculate correlation of each epoch with the median ERF
        correlations = np.array([np.corrcoef(epoch.flatten(), median_erf.flatten())[0, 1] for epoch in meg_data])
        # Threshold for correlation (e.g., 0.5)
        correlation_threshold = np.percentile(correlations, 1)
        # Keep epochs with correlation above the threshold
        good_epochs = correlations > correlation_threshold
        print(f"Rejecting {(~good_epochs).sum()} / {len(good_epochs)} epochs (correlation < {correlation_threshold:.3f})")
        # Apply rejection to both MEG data and behavioral dataframe
        meg_data = meg_data[good_epochs]
        merged_df = merged_df[good_epochs].reset_index(drop=True)


        # ERF removal sequence: Start at saccade onset, end at fixation onset
        # We progressively roll forward through events, removing ERFs, then roll back

        # Calculate sampling frequency for time shifting
        S_FREQ = 500  # Sampling frequency in Hz

        # Get all durations upfront
        saccade_duration = merged_df["duration"].values
        fixation_duration = merged_df["associated_fixation_duration"].values

        # Track cumulative shift from saccade onset
        cumulative_shift = np.zeros(len(merged_df), dtype=int)

        # 1. Remove saccade ERF at saccade onset (t=0)
        if "saccade" in remove_erfs:
            print("Removing saccade ERF at saccade onset")
            sacc_erf = np.median(meg_data, axis=0)
            meg_data = meg_data - sacc_erf

        # 2. Roll forward to fixation onset
        n_shifts_saccade = (saccade_duration * S_FREQ).astype(int)
        meg_data = np.array([np.roll(meg_data[i], -n_shifts_saccade[i], axis=-1) for i in range(meg_data.shape[0])])
        cumulative_shift += n_shifts_saccade

        # 3. Remove fixation ERF at fixation onset
        if "fixation" in remove_erfs:
            print("Removing fixation ERF at fixation onset")
            fix_erf = np.median(meg_data, axis=0)
            meg_data = meg_data - fix_erf

        # 4. Roll forward to subsequent saccade onset (if needed)
        if "saccade_post" in remove_erfs:
            n_shifts_fixation = (fixation_duration * S_FREQ).astype(int)
            meg_data = np.array([np.roll(meg_data[i], -n_shifts_fixation[i], axis=-1) for i in range(meg_data.shape[0])])
            cumulative_shift += n_shifts_fixation

            print("Removing post-saccade ERF at subsequent saccade onset")
            post_sacc_erf = np.median(meg_data, axis=0)
            meg_data = meg_data - post_sacc_erf

        # 5. Roll forward to subsequent fixation onset (if needed)
        if "fixation_post" in remove_erfs:
            subsequent_saccade_duration = merged_df["subsequent_saccade_duration"].values
            n_shifts_saccade_post = (subsequent_saccade_duration * S_FREQ).astype(int)
            meg_data = np.array([np.roll(meg_data[i], -n_shifts_saccade_post[i], axis=-1) for i in range(meg_data.shape[0])])
            cumulative_shift += n_shifts_saccade_post

            print("Removing post-fixation ERF at subsequent fixation onset")
            post_fix_erf = np.median(meg_data, axis=0)
            meg_data = meg_data - post_fix_erf

        # 6. Roll back to fixation onset for analysis
        if "saccade_post" in remove_erfs or "fixation_post" in remove_erfs:
            print(f"Rolling back to fixation onset")
            # We want to be at fixation onset, so roll back everything except the initial saccade shift
            rollback_shift = cumulative_shift - n_shifts_saccade
            meg_data = np.array([np.roll(meg_data[i], rollback_shift[i], axis=-1) for i in range(meg_data.shape[0])])

        # Set appropriate duration column
        dur_col = "associated_fixation_duration"

        # Remove events without associated fixation duration
        valid_fixation_mask = ~np.isnan(merged_df[dur_col])
        merged_df = merged_df[valid_fixation_mask]
        meg_data = meg_data[valid_fixation_mask]

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

def split_epochs_by_memorability(merged_df, meg_data, mem_split=None, mem_crop_size=100,
                                 balance_epochs=True, match_duration_distribution=True, dur_col="duration"):
    """
    Split epochs into high/low memorability groups based on memorability scores.

    This function performs balanced epoch selection to control for confounds:
    1. Ensures equal number of epochs in high/low groups
    2. Optionally matches duration distributions between groups

    This controls for the confound that longer fixations may have different memorability scores.

    Parameters:
    -----------
    merged_df : pd.DataFrame
        Metadata for the epochs (should already be duration-filtered).
    meg_data : np.ndarray
        MEG data array of shape (n_epochs, n_channels, n_times).
    mem_split : str or None
        Split specification like "50/50" or "25/25".
        Format: "bottom_percent/top_percent"
        If None, returns all data as a single group.
    mem_crop_size : int
        Crop size in pixels for memorability analysis (must match precomputed scores).
    balance_epochs : bool
        If True, downsample to ensure equal number of epochs in each group.
    match_duration_distribution : bool
        If True, use stratified sampling to match duration distributions between groups.
        Requires balance_epochs=True.
    dur_col : str
        Name of duration column to use for distribution matching.

    Returns:
    --------
    list of tuples: [(split_name, meg_data_subset, merged_df_subset), ...]
        If mem_split is None: [("all", full_data, full_df)]
        If mem_split="50/50": [("low_mem", bottom_50%, ...), ("high_mem", top_50%, ...)]
    """
    if mem_split is None:
        print("No memorability split requested - using all epochs")
        return [("all", meg_data, merged_df)]

    print(f"Splitting epochs by memorability: {mem_split}")

    # Parse the split specification
    try:
        bottom_pct, top_pct = map(int, mem_split.split("/"))
    except ValueError:
        raise ValueError(f"Invalid mem_split format: '{mem_split}'. Expected format: 'bottom/top' (e.g., '50/50', '25/25')")

    # Load memorability scores
    from avs_gazetime.memorability.mem_tools import get_memorability_scores

    # Get subject ID from dataframe
    subject_id = merged_df["subject"].iloc[0]

    print(f"Loading memorability scores for subject {subject_id} with crop size {mem_crop_size}...")

    # Add memorability scores to metadata
    merged_df_with_mem = get_memorability_scores(
        merged_df.copy(),
        subject_id,
        targets=["memorability"],
        model_task="regression",
        crop_size_pix=mem_crop_size
    )

    # Check for missing memorability scores
    n_missing = merged_df_with_mem["mem_score"].isna().sum()
    if n_missing > 0:
        print(f"WARNING: {n_missing} epochs missing memorability scores - excluding from split")
        valid_mask = ~merged_df_with_mem["mem_score"].isna()
        merged_df_with_mem = merged_df_with_mem[valid_mask].reset_index(drop=True)
        meg_data = meg_data[valid_mask]

    # Calculate percentile thresholds
    mem_scores = merged_df_with_mem["mem_score"].values
    low_threshold = np.percentile(mem_scores, bottom_pct)
    high_threshold = np.percentile(mem_scores, 100 - top_pct)

    print(f"Memorability score range: [{np.min(mem_scores):.3f}, {np.max(mem_scores):.3f}]")
    print(f"Low threshold (bottom {bottom_pct}%): {low_threshold:.3f}")
    print(f"High threshold (top {top_pct}%): {high_threshold:.3f}")

    # Create masks
    low_mask = mem_scores <= low_threshold
    high_mask = mem_scores >= high_threshold

    # Split the data
    results = []

    # Get initial groups
    low_df = merged_df_with_mem[low_mask].copy()
    high_df = merged_df_with_mem[high_mask].copy()

    print(f"  Initial low memorability group: {len(low_df)} epochs (mem_score ≤ {low_threshold:.3f})")
    print(f"  Initial high memorability group: {len(high_df)} epochs (mem_score ≥ {high_threshold:.3f})")

    # Report excluded middle range
    n_excluded = len(merged_df_with_mem) - len(low_df) - len(high_df)
    if n_excluded > 0:
        print(f"  Excluded middle range: {n_excluded} epochs ({low_threshold:.3f} < mem_score < {high_threshold:.3f})")

    # Balance epochs between groups if requested
    if balance_epochs and len(low_df) > 0 and len(high_df) > 0:
        print(f"\n  Balancing epoch counts between groups...")

        # Determine target count (minimum of the two groups)
        n_target = min(len(low_df), len(high_df))
        print(f"    Target count per group: {n_target} epochs")

        if match_duration_distribution:
            print(f"    Using stratified sampling to match duration distributions...")

            # Create duration bins for stratification (quintiles)
            n_bins = 5
            all_durations = pd.concat([low_df[dur_col], high_df[dur_col]])
            duration_bins = pd.qcut(all_durations, q=n_bins, labels=False, duplicates='drop')
            n_bins_actual = len(np.unique(duration_bins))

            # Assign bins to each group
            low_df['_duration_bin'] = pd.qcut(low_df[dur_col], q=n_bins, labels=False, duplicates='drop')
            high_df['_duration_bin'] = pd.qcut(high_df[dur_col], q=n_bins, labels=False, duplicates='drop')

            # Sample from each group to match target count and distribution
            def stratified_sample(df, n_target, random_state=42):
                """Sample to target count while preserving duration distribution."""
                sampled_indices = []
                bin_counts = df['_duration_bin'].value_counts()

                for bin_id in sorted(df['_duration_bin'].unique()):
                    bin_df = df[df['_duration_bin'] == bin_id]
                    # Proportional sample from this bin
                    n_from_bin = int(np.round(n_target * len(bin_df) / len(df)))
                    n_from_bin = min(n_from_bin, len(bin_df))  # Don't exceed available

                    if n_from_bin > 0:
                        sampled = bin_df.sample(n=n_from_bin, random_state=random_state)
                        sampled_indices.extend(sampled.index.tolist())

                # If we're short of target, randomly sample remaining
                if len(sampled_indices) < n_target:
                    remaining = df.loc[~df.index.isin(sampled_indices)]
                    n_additional = min(n_target - len(sampled_indices), len(remaining))
                    additional = remaining.sample(n=n_additional, random_state=random_state)
                    sampled_indices.extend(additional.index.tolist())

                # If we're over target, randomly remove excess
                if len(sampled_indices) > n_target:
                    np.random.seed(random_state)
                    sampled_indices = np.random.choice(sampled_indices, size=n_target, replace=False).tolist()

                return df.loc[sampled_indices]

            low_df_balanced = stratified_sample(low_df, n_target, random_state=42)
            high_df_balanced = stratified_sample(high_df, n_target, random_state=43)

            # Remove temporary column
            low_df_balanced = low_df_balanced.drop(columns=['_duration_bin'])
            high_df_balanced = high_df_balanced.drop(columns=['_duration_bin'])

            # Report duration statistics
            print(f"    Duration statistics after balancing:")
            print(f"      Low group:  mean={low_df_balanced[dur_col].mean():.3f}s, std={low_df_balanced[dur_col].std():.3f}s")
            print(f"      High group: mean={high_df_balanced[dur_col].mean():.3f}s, std={high_df_balanced[dur_col].std():.3f}s")

        else:
            # Simple random sampling without duration matching
            print(f"    Using simple random sampling...")
            low_df_balanced = low_df.sample(n=n_target, random_state=42)
            high_df_balanced = high_df.sample(n=n_target, random_state=43)

        # Update to use balanced versions
        low_df = low_df_balanced.reset_index(drop=True)
        high_df = high_df_balanced.reset_index(drop=True)

        print(f"  Final balanced counts: Low={len(low_df)}, High={len(high_df)}")

    # Get MEG data for selected epochs
    low_meg = meg_data[low_df.index.values]
    high_meg = meg_data[high_df.index.values]

    # Reset indices
    low_df = low_df.reset_index(drop=True)
    high_df = high_df.reset_index(drop=True)

    results.append(("low_mem", low_meg, low_df))
    results.append(("high_mem", high_meg, high_df))

    return results

def split_epochs_by_duration(merged_df, meg_data, duration_split=None,
                              balance_epochs=True, dur_col="duration",
                              min_duration=None, max_duration=None):
    """
    Split epochs into short/long duration groups.

    Parameters:
    -----------
    merged_df : pd.DataFrame
        Metadata for the epochs (should already be duration-filtered).
    meg_data : np.ndarray
        MEG data array of shape (n_epochs, n_channels, n_times).
    duration_split : float or None
        Duration threshold in milliseconds (e.g., 350).
        If None, returns all data as a single group.
    balance_epochs : bool
        If True, downsample to ensure equal number of epochs in each group.
    dur_col : str
        Name of duration column to use for splitting.
    min_duration : float or None
        Minimum duration in seconds to include (e.g., for offset-locked PAC window).
        If provided, epochs shorter than this will be excluded before splitting.
    max_duration : float or None
        Maximum duration in seconds to include (e.g., epoch recording length).
        If provided, epochs longer than this will be excluded before splitting.

    Returns:
    --------
    list of tuples: [(split_name, meg_data_subset, merged_df_subset), ...]
        If duration_split is None: [("all", full_data, full_df)]
        If duration_split=350: [("short_dur", <350ms, ...), ("long_dur", ≥350ms, ...)]
    """
    if duration_split is None:
        print("No duration split requested - using all epochs")
        return [("all", meg_data, merged_df)]

    print(f"Splitting epochs by duration: {duration_split} ms threshold")

    # Convert threshold to seconds
    threshold_s = duration_split / 1000.0

    # Get durations
    durations = merged_df[dur_col].values

    print(f"Initial duration range: [{np.min(durations)*1000:.1f}, {np.max(durations)*1000:.1f}] ms")
    print(f"Duration threshold: {threshold_s*1000:.1f} ms")

    # Filter by minimum and maximum duration if provided (for offset-locked analysis)
    if min_duration is not None or max_duration is not None:
        valid_duration_mask = np.ones(len(durations), dtype=bool)

        if min_duration is not None:
            print(f"Filtering to epochs >= {min_duration*1000:.0f}ms (PAC window duration)")
            valid_duration_mask &= (durations >= min_duration)
            n_excluded_min = np.sum(durations < min_duration)
            if n_excluded_min > 0:
                print(f"  Excluding {n_excluded_min} epochs shorter than PAC window")

        if max_duration is not None:
            print(f"Filtering to epochs <= {max_duration*1000:.0f}ms (epoch recording length)")
            valid_duration_mask &= (durations <= max_duration)
            n_excluded_max = np.sum(durations > max_duration)
            if n_excluded_max > 0:
                print(f"  Excluding {n_excluded_max} epochs longer than epoch recording")

        merged_df = merged_df[valid_duration_mask].reset_index(drop=True)
        meg_data = meg_data[valid_duration_mask]
        durations = merged_df[dur_col].values
        print(f"  Remaining epochs: {len(durations)}")
        print(f"  Filtered duration range: [{np.min(durations)*1000:.1f}, {np.max(durations)*1000:.1f}] ms")

    # Create masks
    short_mask = durations < threshold_s
    long_mask = durations >= threshold_s

    # Split the data
    results = []

    # Get initial groups
    short_df = merged_df[short_mask].copy()
    long_df = merged_df[long_mask].copy()

    print(f"  Initial short duration group: {len(short_df)} epochs (< {threshold_s*1000:.1f} ms)")
    print(f"  Initial long duration group: {len(long_df)} epochs (≥ {threshold_s*1000:.1f} ms)")

    # Balance epochs between groups if requested
    if balance_epochs and len(short_df) > 0 and len(long_df) > 0:
        print(f"\n  Balancing epoch counts between groups...")

        # Determine target count (minimum of the two groups)
        n_target = min(len(short_df), len(long_df))
        print(f"    Target count per group: {n_target} epochs")

        # Simple random sampling
        print(f"    Using simple random sampling...")
        short_df_balanced = short_df.sample(n=n_target, random_state=42)
        long_df_balanced = long_df.sample(n=n_target, random_state=43)

        # Update to use balanced versions
        short_df = short_df_balanced.reset_index(drop=True)
        long_df = long_df_balanced.reset_index(drop=True)

        print(f"  Final balanced counts: Short={len(short_df)}, Long={len(long_df)}")

        # Report duration statistics
        print(f"  Duration statistics after balancing:")
        print(f"    Short group: mean={short_df[dur_col].mean()*1000:.1f}ms, std={short_df[dur_col].std()*1000:.1f}ms")
        print(f"    Long group:  mean={long_df[dur_col].mean()*1000:.1f}ms, std={long_df[dur_col].std()*1000:.1f}ms")

    # Get MEG data for selected epochs
    short_meg = meg_data[short_df.index.values]
    long_meg = meg_data[long_df.index.values]

    # Reset indices
    short_df = short_df.reset_index(drop=True)
    long_df = long_df.reset_index(drop=True)

    results.append(("short_dur", short_meg, short_df))
    results.append(("long_dur", long_meg, long_df))

    return results

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

def aggregate_pac_results(chunks_dir, output_fname):
    """
    Aggregate all chunk CSV files into a single consolidated CSV.

    This function reads all chunk files from the chunks directory,
    removes duplicates (keeping the first occurrence), and saves
    the aggregated results to the output file.

    Parameters:
    -----------
    chunks_dir : str
        Directory containing chunk_*.csv files
    output_fname : str
        Path to save the aggregated CSV file
    """
    import glob
    import os

    # Find all chunk files
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.csv")))

    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}")
        return

    print(f"Found {len(chunk_files)} chunk files to aggregate")

    # Read all chunk files
    all_chunks = []
    for chunk_file in chunk_files:
        try:
            chunk_df = pd.read_csv(chunk_file, index_col=0)
            all_chunks.append(chunk_df)
            print(f"  Loaded {os.path.basename(chunk_file)}: {len(chunk_df)} rows")
        except Exception as e:
            print(f"  Warning: Could not read {os.path.basename(chunk_file)}: {e}")

    if not all_chunks:
        print("No valid chunk files could be read")
        return

    # Concatenate all chunks
    aggregated_df = pd.concat(all_chunks, ignore_index=True)
    print(f"\nTotal rows before deduplication: {len(aggregated_df)}")

    # Remove duplicates based on channel and split_group (keep first occurrence)
    if 'split_group' in aggregated_df.columns:
        aggregated_df = aggregated_df.drop_duplicates(subset=['channel', 'split_group'], keep='first')
    else:
        aggregated_df = aggregated_df.drop_duplicates(subset=['channel'], keep='first')

    print(f"Total rows after deduplication: {len(aggregated_df)}")

    # Sort by channel for better readability
    aggregated_df = aggregated_df.sort_values('channel').reset_index(drop=True)

    # Save aggregated results
    aggregated_df.to_csv(output_fname)
    print(f"\nAggregated results saved to: {output_fname}")

    # Display summary statistics
    print(f"\n{'='*60}")
    print("Aggregated results summary:")
    print(f"{'='*60}")
    if 'split_group' in aggregated_df.columns:
        summary = aggregated_df.groupby('split_group').agg({
            'pac': ['count', 'mean', 'std'],
            'n_epochs': 'first'
        })
        print(summary)
    else:
        print(aggregated_df.describe())

    # Count significant channels
    n_sig = np.sum(aggregated_df["pac"] > 1.96)
    print(f"\nSignificant channels (z > 1.96): {n_sig}/{len(aggregated_df)} ({100*n_sig/len(aggregated_df):.1f}%)")