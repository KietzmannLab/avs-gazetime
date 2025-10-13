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
from pac_dataloader import load_meg_data, get_channel_chunks
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
    THETA_STEPS, GAMMA_STEPS, SURROGATE_STYLE
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
        sessions = merged_df["session"].values
        time_window = TIME_WINDOW
        theta_band = THETA_BAND
        gamma_band = GAMMA_BAND
        surrogate_style = SURROGATE_STYLE  # Using the parameter from params_pac
        
        print(f"Using surrogate style: {surrogate_style}")

        # Pre-filter all data once before the channel loop (major speedup)
        print("Pre-filtering all channels for theta and gamma bands...")
        from mne.filter import filter_data

        theta_data_all = filter_data(
            meg_data.astype(float), 500,
            theta_band[0], theta_band[1],
            method='fir', phase='minimum',
            n_jobs=-2,
            verbose=0
        )
        print(f"Theta filtering complete. Shape: {theta_data_all.shape}")

        gamma_data_all = filter_data(
            meg_data.astype(float), 500,
            gamma_band[0], gamma_band[1],
            method='fir', phase='minimum',
            n_jobs=-2,
            verbose=0
        )
        print(f"Gamma filtering complete. Shape: {gamma_data_all.shape}")

        # Compute PAC values for all channels in parallel
        channel_indices = range(meg_data.shape[1])
        pac_per_channel = Parallel(n_jobs=parallel_jobs["pac_computation"])(
            delayed(compute_pac_hilbert)(
                None, 500, channel, theta_band=theta_band, gamma_band=gamma_band,
                times=times, time_window=time_window, n_bootstraps=200,
                plot=False, verbose=False, durations=merged_df[dur_col].values,
                method=PAC_METHOD, sessions=sessions, surrogate_style=surrogate_style,
                theta_data_prefiltered=theta_data_all, gamma_data_prefiltered=gamma_data_all
            ) for channel in tqdm(channel_indices, desc="Computing PAC per channel", unit="channel")
        )
        
        # Store results in a dataframe
        pac_results_this_run = pd.DataFrame()
        if channel_idx is not None:
            for ch, pac in enumerate(pac_per_channel):
                new_rows = pd.DataFrame({
                    "subject": [SUBJECT_ID],
                    "channel": [channel_idx[ch]],
                    "pac": [pac],
                    "surrogate_style": [surrogate_style]
                })
                pac_results_this_run = pd.concat([pac_results_this_run, new_rows], ignore_index=True)
        else:
            for ch, pac in enumerate(pac_per_channel):
                new_rows = pd.DataFrame({
                    "subject": [SUBJECT_ID],
                    "channel": [ch],
                    "pac": [pac],
                    "surrogate_style": [surrogate_style]
                })
                pac_results_this_run = pd.concat([pac_results_this_run, new_rows], ignore_index=True)
        
        print(pac_results_this_run)
        
        # Save results with surrogate style in the filename
        pac_fname = f"{PLOTS_DIR}/pac_results_{SUBJECT_ID}_{CH_TYPE}_{EVENT_TYPE}_{theta_band[0]}-{theta_band[1]}_{gamma_band[0]}-{gamma_band[1]}_{time_window[0]}-{time_window[1]}_{remove_erfs}_{surrogate_style}.csv"
        if os.path.exists(pac_fname):
            pac_results = pd.read_csv(pac_fname, index_col=0)
            # Concatenate the dataframes
            pac_results = pd.concat([pac_results, pac_results_this_run], ignore_index=True)
        else:
            pac_results = pac_results_this_run
        
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