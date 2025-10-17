#!/usr/bin/env python3
"""
This script computes phase-amplitude coupling (PAC) for MEG data.
It uses modular functions for data loading and PAC computation.
"""

import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Import custom modules
from pac_dataloader import load_meg_data, get_channel_chunks, split_epochs_by_memorability, split_epochs_by_duration, aggregate_pac_results
from pac_functions import compute_pac_hilbert, compute_full_cross_frequency_pac_matrix

# Import configuration
from avs_gazetime.config import (
    S_FREQ,
    SESSIONS,
    SUBJECT_ID,
    PLOTS_DIR,
    CH_TYPE
)

from params_pac import (
    QUANTILES, EVENT_TYPE, DECIM,
    PHASE_OR_POWER, remove_erfs,
    TIME_WINDOW, THETA_BAND, GAMMA_BAND, PAC_METHOD,
    THETA_STEPS, GAMMA_STEPS, SURROGATE_STYLE,
    MEM_SPLIT, MEM_CROP_SIZE,
    DURATION_SPLIT, DURATION_BALANCE, OFFSET_LOCKED
)


def main():
    """Main function to run PAC analysis."""
    # Get optimized parallel job settings
    from avs_gazetime.pac.pac_dataloader import optimize_n_jobs
    parallel_jobs = optimize_n_jobs()
    print(f"Optimized parallel job settings: {parallel_jobs}")
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        channel_chunk = sys.argv[1].split("_") #This is a tuple of chunk and number of chunks total
    else:
        channel_chunk = None
    print("Processing channel chunk", channel_chunk)
    
    # Get channel chunks for processing
    channels = get_channel_chunks(CH_TYPE, EVENT_TYPE, channel_chunk)
    print(channels)
    
    # Iterate over the channels
    for channel_idx in channels:
        # Load and preprocess the MEG data
        meg_data, merged_df, times = load_meg_data(
            event_type=EVENT_TYPE,
            ch_type=CH_TYPE,
            sessions=SESSIONS,
            channel_idx=channel_idx,
            remove_erfs=remove_erfs
        )
        
        print(len(merged_df), meg_data.shape)

        # Get the appropriate duration column
        dur_col = "duration" if EVENT_TYPE == "fixation" else "associated_fixation_duration"

        # Set up parameters for PAC computation
        time_window = TIME_WINDOW
        theta_band = THETA_BAND
        gamma_band = GAMMA_BAND
        surrogate_style = SURROGATE_STYLE  # Using the parameter from params_pac

        print(f"Using surrogate style: {surrogate_style}")

        # Split epochs by memorability or duration if requested
        print(f"\n{'='*60}")
        print(f"Memorability split configuration: {MEM_SPLIT}")
        print(f"Duration split configuration: {DURATION_SPLIT}")
        print(f"Offset-locked: {OFFSET_LOCKED}")
        print(f"{'='*60}")

        # Validate: only one split type allowed at a time
        if DURATION_SPLIT is not None and MEM_SPLIT is not None:
            raise ValueError(
                "ERROR: Cannot use both DURATION_SPLIT and MEM_SPLIT simultaneously.\n"
                f"  DURATION_SPLIT = {DURATION_SPLIT}\n"
                f"  MEM_SPLIT = {MEM_SPLIT}\n"
                "Please set one to None in params_pac.py"
            )

        # Apply the appropriate split
        if DURATION_SPLIT is not None:
            # Calculate minimum and maximum duration for offset-locked analysis
            min_duration = None
            max_duration = None
            if OFFSET_LOCKED:
                window_duration = time_window[1] - time_window[0]
                # Allow ±50ms extension beyond fixation boundaries (100ms total)
                min_duration = window_duration - 0.1
                max_duration = times[-1]  # Maximum recordable fixation duration
                print(f"Offset-locked mode: filtering to epochs >= {min_duration*1000:.0f}ms and <= {max_duration*1000:.0f}ms")
                print(f"  (PAC window: {window_duration*1000:.0f}ms with ±50ms extension allowance)")

            epoch_splits = split_epochs_by_duration(
                merged_df, meg_data,
                duration_split=DURATION_SPLIT,
                balance_epochs=DURATION_BALANCE,
                dur_col=dur_col,
                min_duration=min_duration,
                max_duration=max_duration
            )
        elif MEM_SPLIT is not None:
            epoch_splits = split_epochs_by_memorability(
                merged_df, meg_data,
                mem_split=MEM_SPLIT,
                mem_crop_size=MEM_CROP_SIZE,
                balance_epochs=True,
                match_duration_distribution=True,
                dur_col=dur_col
            )
        else:
            # No split - use all data
            epoch_splits = [("all", meg_data, merged_df)]

        # Check which channels need processing (skip already computed channels)
        print(f"\n{'='*60}")
        print("Checking for existing results...")
        print(f"{'='*60}")

        # Build expected filenames
        split_str = ""
        if DURATION_SPLIT is not None:
            split_str = f"_dursplit_{DURATION_SPLIT}ms"
        elif MEM_SPLIT is not None:
            split_str = f"_memsplit_{MEM_SPLIT.replace('/', '-')}"

        offset_str = "_offset" if OFFSET_LOCKED else ""
        base_fname = f"pac_results_{SUBJECT_ID}_{CH_TYPE}_{EVENT_TYPE}_{theta_band[0]}-{theta_band[1]}_{gamma_band[0]}-{gamma_band[1]}_{time_window[0]}-{time_window[1]}_{remove_erfs}_{surrogate_style}{split_str}{offset_str}"

        # Final aggregated CSV filename
        pac_fname = f"{PLOTS_DIR}/{base_fname}.csv"

        # Job-specific subdirectory for partial results
        chunks_dir = f"{PLOTS_DIR}/{base_fname}_chunks"
        os.makedirs(chunks_dir, exist_ok=True)

        print(f"Chunks directory: {chunks_dir}")
        print(f"Final aggregated results: {pac_fname}")

        # Load existing results from ALL chunk files to check what's already computed
        existing_results = None
        chunk_files = [f for f in os.listdir(chunks_dir) if f.startswith('chunk_') and f.endswith('.csv')] if os.path.exists(chunks_dir) else []

        if chunk_files:
            print(f"Found {len(chunk_files)} existing chunk files")
            chunk_dfs = []
            for chunk_file in chunk_files:
                chunk_path = os.path.join(chunks_dir, chunk_file)
                try:
                    chunk_df = pd.read_csv(chunk_path, index_col=0)
                    chunk_dfs.append(chunk_df)
                except Exception as e:
                    print(f"Warning: Could not read {chunk_file}: {e}")

            if chunk_dfs:
                existing_results = pd.concat(chunk_dfs, ignore_index=True)
                print(f"Loaded {len(existing_results)} rows from existing chunks")
                print(f"Existing columns: {existing_results.columns.tolist()}")
        else:
            print(f"No existing chunk files found")

        # Store all PAC results across splits
        all_pac_results = []

        # Process each split group
        for split_name, meg_data_split, merged_df_split in epoch_splits:
            print(f"\n{'='*60}")
            print(f"Processing {split_name} group: {len(merged_df_split)} epochs")
            print(f"{'='*60}")

            # Determine which channels need processing for this split
            channel_indices_all = range(meg_data_split.shape[1])
            channels_to_process = []
            channels_to_skip = []

            if existing_results is not None:
                # Check which channels are already computed for this split group
                for ch_idx in channel_indices_all:
                    actual_channel = channel_idx[ch_idx] if channel_idx is not None else ch_idx

                    # Check if this channel + split_group combination exists
                    existing_entry = existing_results[
                        (existing_results["channel"] == actual_channel) &
                        (existing_results["split_group"] == split_name)
                    ]

                    if len(existing_entry) > 0:
                        channels_to_skip.append(ch_idx)
                        print(f"  Skipping channel {actual_channel} ({split_name}) - already computed")
                    else:
                        channels_to_process.append(ch_idx)
            else:
                # No existing results, process all channels
                channels_to_process = list(channel_indices_all)

            if not channels_to_process:
                print(f"  All channels for {split_name} already processed. Skipping this group.")
                continue

            print(f"  Processing {len(channels_to_process)}/{len(channel_indices_all)} channels for {split_name}")

            # Get session info for this split
            sessions_split = merged_df_split["session"].values

            # Pre-filter all channels in this chunk once (major speedup vs per-channel filtering)
            # Memory is managed by running multiple smaller chunks via SLURM array jobs
            print("Pre-filtering all channels for theta and gamma bands...")
            from mne.filter import filter_data

            print(f"  Filtering theta band for {meg_data_split.shape[1]} channels...")
            theta_data_all = filter_data(
                meg_data_split.astype(float), 500,
                theta_band[0], theta_band[1],
                method='fir', phase='minimum',
                n_jobs=-2,
                verbose=0
            )
            print(f"  Theta filtering complete. Shape: {theta_data_all.shape}")

            print(f"  Filtering gamma band for {meg_data_split.shape[1]} channels...")
            gamma_data_all = filter_data(
                meg_data_split.astype(float), 500,
                gamma_band[0], gamma_band[1],
                method='fir', phase='minimum',
                n_jobs=-2,
                verbose=0
            )
            print(f"  Gamma filtering complete. Shape: {gamma_data_all.shape}")

            # Compute PAC values only for channels that need processing
            pac_per_channel = Parallel(n_jobs=parallel_jobs["pac_computation"])(
                delayed(compute_pac_hilbert)(
                    None, 500, channel, theta_band=theta_band, gamma_band=gamma_band,
                    times=times, time_window=time_window, n_bootstraps=200,
                    plot=False, verbose=False, durations=merged_df_split[dur_col].values,
                    method=PAC_METHOD, sessions=sessions_split, surrogate_style=surrogate_style,
                    theta_data_prefiltered=theta_data_all, gamma_data_prefiltered=gamma_data_all,
                    offset_locked=OFFSET_LOCKED
                ) for channel in tqdm(channels_to_process, desc=f"Computing PAC for {split_name}", unit="channel")
            )

            # Store results with split_group label
            for ch_local_idx, pac in enumerate(pac_per_channel):
                ch_global = channels_to_process[ch_local_idx]
                actual_channel = channel_idx[ch_global] if channel_idx is not None else ch_global

                result_entry = {
                    "subject": SUBJECT_ID,
                    "channel": actual_channel,
                    "pac": pac,
                    "split_group": split_name,
                    "n_epochs": len(merged_df_split),
                    "surrogate_style": surrogate_style
                }
                all_pac_results.append(result_entry)

            print(f"\n{split_name} group complete: {len(pac_per_channel)} channels processed")

        # Convert all results to dataframe
        pac_results_this_run = pd.DataFrame(all_pac_results)

        # Save job-specific results only if we computed something new
        if len(pac_results_this_run) > 0:
            # Determine channel range for filename
            channels_processed = pac_results_this_run['channel'].unique()
            min_channel = int(channels_processed.min())
            max_channel = int(channels_processed.max())

            # Job-specific filename based on channel range
            job_fname = f"{chunks_dir}/chunk_ch{min_channel}-{max_channel}.csv"

            print(f"\n{'='*60}")
            print(f"Saving {len(pac_results_this_run)} results to job-specific file")
            print(f"Channel range: {min_channel}-{max_channel}")
            print(f"File: {job_fname}")
            print(f"{'='*60}")

            pac_results_this_run.to_csv(job_fname)
            print(f"Successfully saved chunk file")

            # Display summary statistics for this chunk
            if len(pac_results_this_run) > 0:
                print(f"\n{'='*60}")
                print(f"Summary for this chunk:")
                print(f"{'='*60}")
                print(pac_results_this_run.describe())
                # How many channels have a significant PAC (z > 1.96)
                n_sig = np.sum(pac_results_this_run["pac"] > 1.96)
                print(f"Significant channels (z > 1.96): {n_sig}/{len(pac_results_this_run)}")
        else:
            print(f"\n{'='*60}")
            print("No new results computed - all channels already processed")
            print(f"{'='*60}")

        # Run aggregation to update the final CSV
        print(f"\n{'='*60}")
        print("Running aggregation to update final results CSV...")
        print(f"{'='*60}")
        aggregate_pac_results(chunks_dir, pac_fname)
        
       
if __name__ == "__main__":
    main()