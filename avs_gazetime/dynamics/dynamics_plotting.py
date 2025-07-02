#!/usr/bin/env python3
"""
Script for plotting and analyzing neural dynamics computed by dynamics_analysis.py.
This script visualizes how neural patterns change over time, focusing on the relationship
between pattern change and fixation duration.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, friedmanchisquare
import statsmodels.stats.multitest as smt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.signal import butter, filtfilt

# Import configuration
from avs_gazetime.config import (
    SESSIONS, SUBJECT_ID, PLOTS_DIR, PLOTS_DIR_NO_SUB
)

from dynamics_params import (
    EVENT_TYPE, CHANGE_METRIC, DELTA_T, ROI_GROUPS, HEMI
)

from dynamics_plot_params import (
    ANALYSIS_TYPE, FILTER_DYNAMICS, Z_SCORE_PER_SUBJECT, DURATION_QUANTILES,
    PLOT_TIME_WINDOW, PEAK_WINDOW, MAX_HALFWAY_WINDOW, MAX_POST_FIXATION,
    SUBJECTS, N_BOOT
)

from dynamics_plotting_tools import (
    load_dynamics_data, get_downward_flank_halfway, get_peak_latency,
    get_peak_amplitude, filter_dynamics
)

def prepare_data_for_analysis(subjects, roi_names, event_type, hemi, change_metric, delta_t):
    """
    Load and prepare data for analysis from multiple subjects and ROIs.
    
    Parameters:
    -----------
    subjects : list
        List of subject IDs.
    roi_names : list
        List of ROI names.
    event_type : str
        Type of events.
    hemi : str
        Hemisphere ('lh', 'rh', or 'both').
    change_metric : str
        Distance metric used for dynamics.
    delta_t : int
        Time difference in ms used for dynamics.
        
    Returns:
    --------
    all_dynamics : pd.DataFrame
        DataFrame with dynamics data in long format.
    all_metadata : pd.DataFrame
        DataFrame with analysis results.
    times : np.ndarray
        Timepoints corresponding to the dynamics.
    result_label : str
        Analysis type being performed.
    """
    all_dynamics = []
    all_metadata = []
    times = None
    
    # Fixed: Determine result_label based on ANALYSIS_TYPE
    result_label = ANALYSIS_TYPE
    
    for roi in roi_names:
        for subject in tqdm(subjects, desc=f"Loading data for {roi}"):
            # Load data
            dynamics, metadata, t = load_dynamics_data(
                subject, roi, event_type, hemi, change_metric, delta_t
            )
            
            if dynamics is None:
                print(f"Skipping {subject}, {roi} - no data found")
                continue
                
            print(f"Loaded data for {subject}, {roi}, shape: {dynamics.shape}")
            
            # Handle duration column naming
            if event_type == "fixation":
                metadata["duration_post"] = metadata["associated_fixation_duration"]
            else:
                # Fixed: Handle saccade case properly
                metadata["duration_post"] = metadata["duration"]
            
            # Store times from first successful load
            if times is None:
                times = t
            
            # Apply optional filtering
            if FILTER_DYNAMICS:
                dynamics = filter_dynamics(dynamics, times, cutoff_hz=FILTER_DYNAMICS)
            
            # Calculate analysis result based on selected method
            if ANALYSIS_TYPE == "halfway":
                result = get_downward_flank_halfway(
                    dynamics, times, metadata["duration_post"].values,
                    tmin_peak=PEAK_WINDOW[0], tmax_peak=PEAK_WINDOW[1],
                    max_t_to_halfway=MAX_HALFWAY_WINDOW, max_post_fixation=MAX_POST_FIXATION
                )
                result_label = "t_halfway"
            elif ANALYSIS_TYPE == "peak_latency":
                result = get_peak_latency(
                    dynamics, times, metadata["duration_post"].values,
                    tmin_peak=PEAK_WINDOW[0], tmax_peak=PEAK_WINDOW[1],
                    max_post_fixation=MAX_POST_FIXATION
                )
                result_label = "peak_latency"
            elif ANALYSIS_TYPE == "peak_amplitude":
                result = get_peak_amplitude(
                    dynamics, times, metadata["duration_post"].values,
                    tmin_peak=PEAK_WINDOW[0], tmax_peak=PEAK_WINDOW[1],
                    max_post_fixation=MAX_POST_FIXATION
                )
                result_label = "peak_amplitude"
            
            # Create metadata for this subject/ROI
            metadata_subj = pd.DataFrame({
                "subject": subject,
                "roi": roi,
                "duration_post": metadata["duration_post"].values,
                result_label: result,
                "hemisphere": metadata["hemisphere"].values if "hemisphere" in metadata else hemi
            })
            
            print(f"  Analysis result ({result_label}) calculated")
            
            # Assign duration quantiles
            metadata_subj["duration_quantile"] = pd.qcut(
                metadata_subj["duration_post"], 
                4, 
                labels=DURATION_QUANTILES
            )
            
            # Prepare dynamics for long-format conversion
            dynamics_df = pd.DataFrame(dynamics)
            dynamics_df["subject"] = subject
            dynamics_df["roi"] = roi
            dynamics_df["duration_post"] = metadata["duration_post"].values
            dynamics_df["duration_quantile"] = metadata_subj["duration_quantile"].values
            dynamics_df[result_label] = result
            dynamics_df["hemisphere"] = metadata["hemisphere"].values if "hemisphere" in metadata else hemi
            
            all_dynamics.append(dynamics_df)
            all_metadata.append(metadata_subj)
    
    # Combine data from all subjects/ROIs
    if not all_dynamics:
        raise ValueError("No data loaded. Check subjects and ROI names.")
        
    all_dynamics = pd.concat(all_dynamics, ignore_index=True)
    all_metadata = pd.concat(all_metadata, ignore_index=True)
    
    print(f"\nCombined data summary:")
    print(f"  Dynamics shape: {all_dynamics.shape}")
    print(f"  Metadata shape: {all_metadata.shape}")
    print(f"  Number of hemispheres: {all_metadata['hemisphere'].nunique()}")
    print(f"  Hemisphere values: {all_metadata['hemisphere'].unique()}")
    
    # Convert dynamics to long format with time as column
    all_dynamics_long = pd.melt(
        all_dynamics,
        id_vars=["subject", "roi", "duration_post", "duration_quantile", result_label, "hemisphere"],
        value_name="dynamics",
        var_name="time_idx"
    )
    
    # Map time index to actual time value
    all_dynamics_long["time"] = all_dynamics_long["time_idx"].astype(int).map(
        {i: t for i, t in enumerate(times)}
    )
    
    # Clean up data
    all_dynamics_long = all_dynamics_long.drop(columns=["time_idx"])
    all_dynamics_long = all_dynamics_long[
        (all_dynamics_long["time"] >= PLOT_TIME_WINDOW[0]) & 
        (all_dynamics_long["time"] <= PLOT_TIME_WINDOW[1])
    ]
    
    # Mask timepoints where time exceeds duration
    # for each quantile compute the mean duration and mask the timepoints where time exceeds the mean duration
    for quantile in DURATION_QUANTILES:
        mean_duration = all_dynamics_long[all_dynamics_long["duration_quantile"] == quantile]["duration_post"].mean()
        print(f"Mean duration for quantile {quantile}: {mean_duration:.3f}s")
        # mask the data belonging to the quantile where the time exceeds the mean duration
        mask = all_dynamics_long["duration_quantile"] == quantile
        mask &= all_dynamics_long["time"] >= mean_duration
        print(f"  Masking {np.sum(mask)} of {len(all_dynamics_long[all_dynamics_long['duration_quantile'] == quantile])} timepoints")
        all_dynamics_long = all_dynamics_long[~mask]
    
    # Apply z-scoring if requested
    if Z_SCORE_PER_SUBJECT:
        all_dynamics_long["dynamics_z"] = all_dynamics_long.groupby(
            ["subject"]
        )["dynamics"].transform(lambda x: zscore(x, nan_policy="omit"))
    else:
        all_dynamics_long["dynamics_z"] = all_dynamics_long["dynamics"]
    
    return all_dynamics_long, all_metadata, times, result_label

def generate_filename_suffix():
    """
    Generate filename suffix with filter and z-score information.
    
    Returns:
    --------
    suffix : str
        Filename suffix string
    """
    suffix = ""
    
    # Add filter information
    if FILTER_DYNAMICS:
        if isinstance(FILTER_DYNAMICS, (int, float)):
            suffix += f"_filt{FILTER_DYNAMICS}Hz"
        else:
            suffix += "_filtered"
    
    # Add z-score information
    if Z_SCORE_PER_SUBJECT:
        suffix += "_zscored"
    
    # Add bilateral information
    suffix += "_bilateral"
    
    return suffix

def plot_dynamics_by_quantile(dynamics_long, result_label, output_dir):
    """
    Plot dynamics by duration quantile for each ROI in separate figures.
    """
    # Convert time to milliseconds for plotting
    dynamics_long["time_ms"] = dynamics_long["time"] * 1000
    
    # Get unique ROIs
    roi_names = dynamics_long["roi"].unique()
    
    # Set poster style
    sns.set_context("poster")
    
    # Generate filename suffix
    suffix = generate_filename_suffix()
    
    # Create separate plots for each ROI
    for roi in roi_names:
        # Create figure with same height as barplot figure
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        # Filter data for this ROI
        roi_data = dynamics_long[dynamics_long["roi"] == roi]
        
        # For halfway analysis, precompute the halfway points for visualization
        precomputed_halfway = {}
        if result_label == "t_halfway":
            for q, quantile in enumerate(DURATION_QUANTILES):
                # Get data for this quantile
                quantile_data = roi_data[roi_data["duration_quantile"] == quantile]
                
                # Compute mean dynamics
                mean_dynamics = quantile_data.groupby("time_ms")["dynamics_z"].mean().reset_index()
                
                # Get mean duration
                mean_duration = quantile_data["duration_post"].mean()
                
                # Find halfway point
                t_halfway = get_downward_flank_halfway(
                    mean_dynamics["dynamics_z"].values,
                    mean_dynamics["time_ms"].values / 1000,  # Convert back to seconds
                    mean_duration,
                    tmin_peak=PEAK_WINDOW[0],
                    tmax_peak=PEAK_WINDOW[1]
                )
                
                if not np.isnan(t_halfway):
                    # Find the dynamics value at halfway point
                    halfway_idx = np.abs(mean_dynamics["time_ms"].values / 1000 - t_halfway).argmin()
                    halfway_val = mean_dynamics["dynamics_z"].iloc[halfway_idx]
                    precomputed_halfway[q] = (t_halfway * 1000, halfway_val)  # Store in ms
        
        # Plot dynamics by quantile (pooled across hemispheres)
        sns.lineplot(
            data=roi_data,
            x="time_ms",
            y="dynamics_z",
            hue="duration_quantile",
            palette="magma",
            ax=ax,
            estimator="mean",
            ci=95,
            n_boot=N_BOOT
        )
        
        # Add titles and labels
        ax.set_xlabel("time [ms]")
        
        ax.set_ylabel(f"ROI pattern change\n[{CHANGE_METRIC} distance]")
        
        # Set x-axis limits
        ax.set_xlim(PLOT_TIME_WINDOW[0] * 1000, PLOT_TIME_WINDOW[1] * 1000)
        
        # Add vertical line at t=0
        ax.axvline(0, color="black", linestyle="--", zorder=1)
        
        # Remove top and right spines
        sns.despine(ax=ax)
        
        # Add halfway points if available
        if result_label == "t_halfway":
            for q, quantile in enumerate(DURATION_QUANTILES):
                if q in precomputed_halfway:
                    t_half, half_val = precomputed_halfway[q]
                    color = sns.color_palette("magma", 4)[q]
                    ax.scatter(t_half, half_val, color=color, s=450, edgecolor="white", zorder=1100)
            
            # Add legend entry for halfway points
            ax.scatter([], [], color="black", s=450, edgecolor="white", label="post-peak halfway")
        
        # Add legend
        ax.legend(frameon=False, title="fixation duration")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure for this ROI with filter info
        fname = os.path.join(output_dir, f"dynamics_quantile_{roi}_{EVENT_TYPE}_{CHANGE_METRIC}_{HEMI}_{DELTA_T}ms{suffix}")
        
        print(f"Saving dynamics plot for {roi} to {fname}.pdf")
        fig.savefig(fname + ".pdf")
        fig.savefig(fname + ".png")
        plt.close(fig)  # Close figure to free memory
    
    print(f"Saved {len(roi_names)} separate dynamics plots")

def plot_dynamics_by_subject_and_quantile(dynamics_long, result_label, output_dir):
    """
    Plot dynamics by duration quantile separately for each subject and ROI.
    """
    # Convert time to milliseconds for plotting
    dynamics_long["time_ms"] = dynamics_long["time"] * 1000
    
    # Get unique ROIs and subjects
    roi_names = dynamics_long["roi"].unique()
    subjects = dynamics_long["subject"].unique()
    n_subjects = len(subjects)
    
    # Set poster style
    sns.set_context("poster")
    
    # Generate filename suffix
    suffix = generate_filename_suffix()
    
    # Create separate plots for each ROI
    for roi in roi_names:
        # Filter data for this ROI
        roi_data = dynamics_long[dynamics_long["roi"] == roi]
        
        # Create figure with one row per subject
        fig, axes = plt.subplots(n_subjects, 1, figsize=(12, 6*n_subjects), dpi=300, sharex=True)
        
        # Process each subject
        for i, subject in enumerate(subjects):
            ax = axes[i] if n_subjects > 1 else axes
            
            # Filter data for this subject
            subject_data = roi_data[roi_data["subject"] == subject]
            
            # For halfway analysis, precompute the halfway points for visualization
            precomputed_halfway = {}
            if result_label == "t_halfway":
                for q, quantile in enumerate(DURATION_QUANTILES):
                    # Get data for this quantile
                    quantile_data = subject_data[subject_data["duration_quantile"] == quantile]
                    
                    if len(quantile_data) == 0:
                        continue
                    
                    # Compute mean dynamics
                    mean_dynamics = quantile_data.groupby("time_ms")["dynamics_z"].mean().reset_index()
                    
                    # Get mean duration
                    mean_duration = quantile_data["duration_post"].mean()
                    
                    # Find halfway point
                    t_halfway = get_downward_flank_halfway(
                        mean_dynamics["dynamics_z"].values,
                        mean_dynamics["time_ms"].values / 1000,  # Convert back to seconds
                        mean_duration,
                        tmin_peak=PEAK_WINDOW[0],
                        tmax_peak=PEAK_WINDOW[1]
                    )
                    
                    if not np.isnan(t_halfway):
                        # Find the dynamics value at halfway point
                        halfway_idx = np.abs(mean_dynamics["time_ms"].values / 1000 - t_halfway).argmin()
                        halfway_val = mean_dynamics["dynamics_z"].iloc[halfway_idx]
                        precomputed_halfway[q] = (t_halfway * 1000, halfway_val)  # Store in ms
            
            # Plot dynamics by quantile for this subject
            if len(subject_data) > 0:
                sns.lineplot(
                    data=subject_data,
                    x="time_ms",
                    y="dynamics_z",
                    hue="duration_quantile",
                    palette="magma",
                    ax=ax,
                    estimator="mean",
                    ci=95,
                    n_boot=N_BOOT
                )
                
                # Add titles and labels
                ax.set_title(f"Subject {subject}")
                
                # Set x-axis limits
                ax.set_xlim(PLOT_TIME_WINDOW[0] * 1000, PLOT_TIME_WINDOW[1] * 1000)
                
                # Add vertical line at t=0
                ax.axvline(0, color="black", linestyle="--", zorder=1)
                
                # Remove top and right spines
                sns.despine(ax=ax)
                
                # Add halfway points if available
                if result_label == "t_halfway":
                    for q, quantile in enumerate(DURATION_QUANTILES):
                        if q in precomputed_halfway:
                            t_half, half_val = precomputed_halfway[q]
                            color = sns.color_palette("magma", 4)[q]
                            ax.scatter(t_half, half_val, color=color, s=250, edgecolor="white", zorder=1100)
                
                # Add legend only to the first subplot to avoid redundancy
                if i == 0:
                    ax.legend(frameon=False, title="fixation duration")
                else:
                    ax.legend().remove()
            else:
                ax.text(0.5, 0.5, f"No data for Subject {subject}", 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Add common labels
        fig.text(0.5, 0.04, "time [ms]", ha='center', va='center')
        fig.text(0.04, 0.5, f"ROI pattern change\n[{CHANGE_METRIC} distance)]", 
                ha='center', va='center', rotation='vertical')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure for this ROI with filter info
        fname = os.path.join(output_dir, f"dynamics_by_subject_{roi}_{EVENT_TYPE}_{CHANGE_METRIC}_{HEMI}_{DELTA_T}ms{suffix}")
        
        print(f"Saving subject-resolved dynamics plot for {roi} to {fname}.pdf")
        fig.savefig(fname + ".pdf")
        fig.savefig(fname + ".png")
        plt.close(fig)  # Close figure to free memory
    
    print(f"Saved {len(roi_names)} subject-resolved dynamics plots")

def plot_result_by_quantile(quantile_results, result_label, output_dir):
    """
    Plot analysis results by duration quantile.
    """
    print(quantile_results.head())
    
    # Reset index and drop NaN values
    quantile_results = quantile_results.reset_index(drop=True).dropna(subset=[result_label])
    
    # Convert time values to milliseconds
    if result_label in ["t_halfway", "peak_latency"]:
        quantile_results[result_label] = quantile_results[result_label] * 1000
    
    quantile_results["duration_post"] = quantile_results["duration_post"] * 1000
    
    # Set poster style
    sns.set_context("poster")
    
    # Generate filename suffix
    suffix = generate_filename_suffix()
    
    # Create figure
    fig, [ax2, ax1] = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 8),
        dpi=300, sharey=False, width_ratios=[1, 2]
    )
    
    # Plot barplot (pooled across hemispheres)
    sns.barplot(
        data=quantile_results,
        x="roi",
        y=result_label,
        ci=95,
        palette="magma",
        hue="duration_quantile",
        errcolor="black",
        capsize=0,
        errwidth=3.5,
        n_boot=N_BOOT,
        ax=ax1, 
        order=ROI_GROUPS.keys(
        )  # Ensure ROIs are ordered by ROI_GROUPS  
    )
    current_labels = ax1.get_xticklabels()
    new_labels = [label.get_text().replace("(mid)", "").strip() for label in current_labels]
    ax1.set_xticklabels(new_labels)
    # Add another barplot that plots the mean duration_post per duration_quantile
    sns.barplot(
        data=quantile_results,
        x="duration_quantile",
        y="duration_post",
        ci=95,
        palette="magma",
        hue="duration_quantile",
        errcolor="black",
        capsize=0,
        errwidth=3.5,
        n_boot=N_BOOT,
        ax=ax2
    )
    
    # Set y label
    ax2.set_ylabel("mean fixation duration [ms]")
    # Set xlabel
    ax2.set_xlabel("duration quartile")
    # Remove xticklabels
    ax2.set_xticklabels([])
    # Set both lims to 550
    ax2.set_ylim(0, 525)
    ax1.set_ylim(0, 525)
    # Update plot style
    sns.despine()
    ax1.legend(frameon=False, title="fixation duration")
    
    # Set labels
    ax1.set_xlabel("ROI")
    
    if result_label == "t_halfway":
        ylabel = "pattern change\npost-peak halfway latency [ms]"
    elif result_label == "peak_latency":
        ylabel = "peak latency [ms]"
    elif result_label == "peak_amplitude":
        ylabel = "peak amplitude"
    
    ax1.set_ylabel(ylabel)
    fig.tight_layout()
    
    # Save figure with filter info
    fname = os.path.join(output_dir, f"{result_label}_quantile_differences_{EVENT_TYPE}_{CHANGE_METRIC}_{HEMI}_{DELTA_T}ms{suffix}")
    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png")
    
    print(f"Saved result plot to {fname}.pdf")

def illustrate_quantiles(metadata):
    """
    Illustrate quartile distribution with a histogram of durations colored by quartile.
    """
    # Create a copy to avoid modifying original data
    metadata_copy = metadata.copy()
    sns.set_context("poster")
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    # Set the context to poster
   
    # Make this ms
    metadata_copy["duration_post"] = metadata_copy["duration_post"] * 1000
    
    # we only need one roi, hemi for this plot
    random_roi = metadata_copy["roi"].unique()[0]
    random_hemi = metadata_copy["hemisphere"].unique()[0]
    metadata_copy = metadata_copy[
        (metadata_copy["roi"] == random_roi) &
        (metadata_copy["hemisphere"] == random_hemi)
    ]
    # Ensure duration_post is numeric
    
    # Plot the histogram (one shared histogram but the bars are colored according to the quartile)
    sns.histplot(metadata_copy, x="duration_post", hue="duration_quantile", multiple="stack", 
                palette="magma", ax=ax, bins=40)
    
    # Set the labels
    ax.set_xlabel("fixation duration [ms]")
    ax.set_ylabel("count")
    
    # Despine and tight layout
    sns.despine()
    plt.tight_layout()
    
    # Generate filename suffix
    suffix = generate_filename_suffix()
    
    # Save the figure with filter info
    output_dir = os.path.join(PLOTS_DIR, "dynamics", "analysis")
    plt.savefig(os.path.join(output_dir, f"quartile_distribution{suffix}.pdf"))
    plt.savefig(os.path.join(output_dir, f"quartile_distribution{suffix}.png"))
    print("Saved quartile distribution plot")

def compute_quantile_results(dynamics_long, result_label):
    """
    Compute quantile results and save with filter info in filename.
    """
    print(f"Computing quantile results for {result_label}...")
    
    # [Your existing computation code here - unchanged]
    median_dynamics = dynamics_long.groupby(
        ["subject", "roi", "duration_quantile", "time", "hemisphere"]
    ).agg(
        dynamics_z=("dynamics_z", "median"),
        duration_post=("duration_post", "mean")
    ).reset_index()
    
    quantile_results = pd.DataFrame(columns=["subject", "roi", "duration_quantile", "duration_post", result_label, "hemisphere"])
    
    for (subject, roi, quantile, hemisphere), group in median_dynamics.groupby(["subject", "roi", "duration_quantile", "hemisphere"]):
        dynamics = group["dynamics_z"].values
        times = group["time"].values
        
        if result_label == "t_halfway":
            result = get_downward_flank_halfway(
                dynamics, times, 
                group["duration_post"].mean(),
                tmin_peak=PEAK_WINDOW[0], tmax_peak=PEAK_WINDOW[1],
                max_t_to_halfway=MAX_HALFWAY_WINDOW, max_post_fixation=MAX_POST_FIXATION
            )
        elif result_label == "peak_latency":
            result = get_peak_latency(
                dynamics, times, 
                group["duration_post"].mean(),
                tmin_peak=PEAK_WINDOW[0], tmax_peak=PEAK_WINDOW[1],
                max_post_fixation=MAX_POST_FIXATION
            )
        elif result_label == "peak_amplitude":
            result = get_peak_amplitude(
                dynamics, times, 
                group["duration_post"].mean(),
                tmin_peak=PEAK_WINDOW[0], tmax_peak=PEAK_WINDOW[1],
                max_post_fixation=MAX_POST_FIXATION
            )
        
        new_row = {
            "subject": subject,
            "roi": roi,
            "duration_quantile": quantile,
            "duration_post": group["duration_post"].mean(),
            result_label: result,
            "hemisphere": hemisphere
        }
        quantile_results = pd.concat([quantile_results, pd.DataFrame([new_row])], ignore_index=True)
    
    quantile_results = quantile_results.reset_index(drop=True)
    
    # Generate filename suffix
    suffix = generate_filename_suffix()
    
    # Save the quantile results to a CSV file with filter info
    filename = f"quantile_results_{result_label}_{EVENT_TYPE}_{CHANGE_METRIC}_{HEMI}_{DELTA_T}ms{suffix}.csv"
    
    quantile_results.to_csv(
        os.path.join(PLOTS_DIR, "dynamics", "analysis", filename),
        index=False
    )
    
    return quantile_results




def main():
    """
    Main function to run the analysis.
    """
    # Define ROIs based on config
    roi_names = list(ROI_GROUPS.keys())
    
    # Create output directory
    output_dir = os.path.join(PLOTS_DIR, "dynamics", "analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analysis type: {ANALYSIS_TYPE}")
    print(f"Loading data for subjects: {SUBJECTS}")
    print(f"ROIs: {roi_names}")
    print(f"Hemisphere setting: {HEMI}")
    
    # Force bilateral analysis if HEMI is not explicitly set to 'lh' or 'rh'
    hemi_for_analysis = HEMI if HEMI in ['lh', 'rh'] else 'both'
  
    # Load and prepare data
    dynamics_long, metadata, times, result_label = prepare_data_for_analysis(
        SUBJECTS, roi_names, EVENT_TYPE, hemi_for_analysis, CHANGE_METRIC, DELTA_T
    )
    
    # Illustrate quartile distribution
    illustrate_quantiles(metadata)
    # break here
    
    # Plot result by quantile
    quantile_results = compute_quantile_results(dynamics_long, result_label)
   
    plot_result_by_quantile(quantile_results, result_label, output_dir)
    

     # Plot dynamics by quantile
    plot_dynamics_by_quantile(dynamics_long, result_label, output_dir)
    
    plot_dynamics_by_subject_and_quantile(dynamics_long, result_label, output_dir)
    
    
    
    
    print("Analysis complete")

if __name__ == "__main__":
    main()