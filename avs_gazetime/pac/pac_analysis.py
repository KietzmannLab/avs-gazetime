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
from pac_dataloader import load_meg_data, get_channel_chunks, split_epochs_by_memorability
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
    MEM_SPLIT, MEM_CROP_SIZE
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

        # Split epochs by memorability if requested
        print(f"\n{'='*60}")
        print(f"Memorability split configuration: {MEM_SPLIT}")
        print(f"{'='*60}")
        epoch_splits = split_epochs_by_memorability(
            merged_df, meg_data,
            mem_split=MEM_SPLIT,
            mem_crop_size=MEM_CROP_SIZE,
            balance_epochs=True,
            match_duration_distribution=True,
            dur_col=dur_col
        )

        # Check which channels need processing (skip already computed channels)
        print(f"\n{'='*60}")
        print("Checking for existing results...")
        print(f"{'='*60}")

        # Build expected filename
        mem_split_str = f"_memsplit_{MEM_SPLIT.replace('/', '-')}" if MEM_SPLIT else ""
        pac_fname = f"{PLOTS_DIR}/pac_results_{SUBJECT_ID}_{CH_TYPE}_{EVENT_TYPE}_{theta_band[0]}-{theta_band[1]}_{gamma_band[0]}-{gamma_band[1]}_{time_window[0]}-{time_window[1]}_{remove_erfs}_{surrogate_style}{mem_split_str}.csv"

        # Load existing results if available
        existing_results = None
        if os.path.exists(pac_fname):
            existing_results = pd.read_csv(pac_fname, index_col=0)
            print(f"Found existing results: {len(existing_results)} rows")
            print(f"Existing columns: {existing_results.columns.tolist()}")
        else:
            print(f"No existing results found at: {pac_fname}")

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
                    theta_data_prefiltered=theta_data_all, gamma_data_prefiltered=gamma_data_all
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

        # Merge with existing results
        if existing_results is not None and len(pac_results_this_run) > 0:
            print(f"\n{'='*60}")
            print(f"Merging {len(pac_results_this_run)} new results with {len(existing_results)} existing results")
            print(f"{'='*60}")
            pac_results = pd.concat([existing_results, pac_results_this_run], ignore_index=True)
        elif len(pac_results_this_run) > 0:
            pac_results = pac_results_this_run
        else:
            # No new results computed (all channels already done)
            if existing_results is not None:
                print(f"\n{'='*60}")
                print("No new results to add - all channels already computed")
                print(f"{'='*60}")
                pac_results = existing_results
            else:
                print(f"\n{'='*60}")
                print("WARNING: No results to save!")
                print(f"{'='*60}")
                return

        # Display summary
        print(f"\n{'='*60}")
        print(f"Total results summary (after merging):")
        print(f"{'='*60}")
        if 'split_group' in pac_results.columns:
            print(pac_results.groupby('split_group').agg({
                'pac': ['count', 'mean', 'std'],
                'n_epochs': 'first'
            }))
        else:
            print(pac_results.describe())
        
        # Save the results
        pac_results.to_csv(pac_fname)
        
        # Display summary statistics
        print(pac_results.describe())
        # How many channels have a significant PAC (z > 1.96)
        print(np.sum(pac_results["pac"] > 1.96))
        # What is the fraction of channels with significant PAC
        print(np.sum(pac_results["pac"] > 1.96) / len(pac_results))
        
        # Process cross-frequency analysis for significant channels
        significant_threshold = 1.56  # z > 2.56 corresponds to p < 0.01
        significant_channels = pac_results[pac_results["pac"] > significant_threshold]["channel"].values
        # Remove duplicates
        significant_channels = np.unique(significant_channels)
        print(f"Found {len(significant_channels)} significant channels: {significant_channels}")
        
       
if __name__ == "__main__":
    main()