#!/usr/bin/env python3
"""
Plot entropy predictions from MEG decoding analysis.

Sister script to plot_memorability_predictions.py.
This script:
1. Loads predicted classification entropy values from decoder across timepoints
2. Plots correlation between predicted entropy and fixation duration over time
3. Plots duration by entropy quartile at peak prediction timepoint
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import configuration
from avs_gazetime.config import PLOTS_DIR_NO_SUB
from avs_gazetime.memorability.memorability_analysis_params import NUM_QUARTILES, Q_LABELS, SUBJECTS


# Entropy-specific targets
ENTROPY_TARGETS = ["entropy", "entropy_relative"]


def load_prediction_data(subjects, target_col, ch_type="mag"):
    """Load prediction data for all subjects."""
    all_data = []

    for subject in subjects:
        subject_str = f"as{subject:02d}"
        plots_dir = os.path.join(PLOTS_DIR_NO_SUB, subject_str)
        pred_file = os.path.join(
            plots_dir,
            "entropy_decoder",
            f"predicted_entropy_{target_col}_{subject_str}_{ch_type}.csv"
        )

        if not os.path.exists(pred_file):
            print(f"Warning: File not found: {pred_file}")
            continue

        print(f"Loading predictions for subject {subject}...")
        df = pd.read_csv(pred_file)
        all_data.append(df)

    if len(all_data) == 0:
        raise FileNotFoundError("No prediction files found")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} prediction records across {len(all_data)} subjects")

    return combined_df


def compute_correlation_over_time(predictions_df):
    """Compute correlation between predicted entropy and duration at each timepoint."""
    results = []

    subjects = predictions_df['subject'].unique()
    timepoints = sorted(predictions_df['meg_time'].unique())

    # Per-subject correlations
    for subject in subjects:
        subject_data = predictions_df[predictions_df['subject'] == subject]

        for timepoint in timepoints:
            time_data = subject_data[subject_data['meg_time'] == timepoint]

            if len(time_data) > 10:
                corr, pval = stats.pearsonr(
                    time_data['predicted_entropy'],
                    time_data['duration']
                )
                results.append({
                    'subject': subject,
                    'meg_time': timepoint,
                    'correlation': corr,
                    'p_value': pval
                })

    return pd.DataFrame(results)


def plot_correlation_over_time(corr_df, output_dir, target_col):
    """Plot correlation between predicted entropy and duration over MEG time."""
    sns.set_context("poster")
    sns.set_palette("colorblind")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot per-subject correlations (transparent)
    for subject in corr_df['subject'].unique():
        subject_data = corr_df[corr_df['subject'] == subject]
        ax.plot(
            subject_data['meg_time'],
            subject_data['correlation'],
            alpha=0.3,
            linewidth=1,
            color='gray'
        )

    # Compute grand average with CI
    grand_avg = corr_df.groupby('meg_time')['correlation'].agg(['mean', 'sem']).reset_index()
    grand_avg['ci_lower'] = grand_avg['mean'] - 1.96 * grand_avg['sem']
    grand_avg['ci_upper'] = grand_avg['mean'] + 1.96 * grand_avg['sem']

    # Plot grand average
    ax.plot(
        grand_avg['meg_time'],
        grand_avg['mean'],
        linewidth=3,
        color='black',
        label='Grand average'
    )

    # Plot confidence interval
    ax.fill_between(
        grand_avg['meg_time'],
        grand_avg['ci_lower'],
        grand_avg['ci_upper'],
        alpha=0.2,
        color='black'
    )

    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)

    ax.set_xlabel('MEG time [ms]')
    ax.set_ylabel('correlation [r]')
    ax.set_title(f'fixation duration from MEG predicted entropy')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    sns.despine()

    save_path = os.path.join(output_dir, f"prediction_duration_correlation_{target_col}.pdf")
    plt.savefig(save_path)
    print(f"Saved correlation plot to {save_path}")
    plt.close()

    return grand_avg


def find_peak_timepoint_per_subject(corr_df):
    """Find the peak correlation timepoint for each subject."""
    peak_times = {}

    for subject in corr_df['subject'].unique():
        subject_data = corr_df[corr_df['subject'] == subject]
        peak_idx = subject_data['correlation'].idxmax()
        peak_time = subject_data.loc[peak_idx, 'meg_time']
        peak_times[subject] = peak_time

    return peak_times


def plot_quartile_analysis(predictions_df, corr_df, output_dir, target_col):
    """Plot duration by entropy quartile at peak timepoint per subject."""
    sns.set_context("poster")
    sns.set_palette("colorblind")

    # Find peak timepoint per subject
    peak_times = find_peak_timepoint_per_subject(corr_df)

    # Extract data at peak timepoint for each subject
    quartile_data = []

    for subject, peak_time in peak_times.items():
        subject_data = predictions_df[
            (predictions_df['subject'] == subject) &
            (predictions_df['meg_time'] == peak_time)
        ]

        if len(subject_data) == 0:
            continue

        # Compute quartiles per subject
        subject_data = subject_data.copy()
        subject_data['quartile_entropy'] = pd.qcut(
            subject_data['predicted_entropy'],
            NUM_QUARTILES,
            labels=Q_LABELS,
            duplicates='drop'
        )

        quartile_data.append(subject_data)

    # Combine all subjects
    quartile_df = pd.concat(quartile_data, ignore_index=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Line plot with error bars
    sns.lineplot(
        x="quartile_entropy",
        y="duration",
        data=quartile_df,
        ci=95,
        estimator=np.nanmean,
        color='#440154',
        markers=False,
        dashes=False,
        ax=ax,
        legend=False
    )

    # Point plot
    sns.pointplot(
        x="quartile_entropy",
        y="duration",
        data=quartile_df,
        ci=95,
        estimator=np.nanmean,
        color='#440154',
        ax=ax,
        dodge=False,
        join=False
    )

    ax.set_xlabel("predicted entropy [quantile bin]")
    ax.set_xticklabels(Q_LABELS + 1)
    ax.set_ylabel("mean fixation duration [ms]")
    ax.set_title(f'fixation duration from MEG predicted entropy')

    plt.tight_layout()
    sns.despine()

    save_path = os.path.join(output_dir, f"duration_per_predicted_entropy_quartile_{target_col}.pdf")
    plt.savefig(save_path)
    print(f"Saved quartile plot to {save_path}")
    plt.close()

    # Compute and print regression statistics
    compute_quartile_regression(quartile_df, target_col)


def compute_quartile_regression(quartile_df, target_col):
    """Compute regression statistics for quartile analysis."""
    # Compute quartile means per subject
    quartile_means = quartile_df.groupby(["subject", "quartile_entropy"])["duration"].mean().reset_index()

    # Arrays for regression
    X_all = []
    y_all = []
    subjects_all = []

    for subject in quartile_means["subject"].unique():
        subj_data = quartile_means[quartile_means["subject"] == subject]
        if len(subj_data) == NUM_QUARTILES:
            X_all.extend(subj_data["quartile_entropy"].values)
            y_all.extend(subj_data["duration"].values)
            subjects_all.extend([subject] * len(subj_data))

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    subjects_all = np.array(subjects_all)

    if len(X_all) == 0:
        print(f"No complete quartile data for {target_col}")
        return

    # Overall regression
    slope_overall, intercept, r_value, p_value, std_err = stats.linregress(X_all, y_all)

    # Per-subject slopes
    subject_slopes = []
    for subject in np.unique(subjects_all):
        subj_mask = subjects_all == subject
        if np.sum(subj_mask) >= 3:
            subj_slope, _, _, _, _ = stats.linregress(X_all[subj_mask], y_all[subj_mask])
            subject_slopes.append(subj_slope)

    subject_slopes = np.array(subject_slopes)

    # Calculate confidence interval
    if len(subject_slopes) > 1:
        slope_mean = np.mean(subject_slopes)
        slope_sem = stats.sem(subject_slopes)
        slope_ci = stats.t.interval(0.95, len(subject_slopes)-1,
                                   loc=slope_mean, scale=slope_sem)
    else:
        slope_mean = slope_overall
        slope_ci = (np.nan, np.nan)

    print(f"\n=== Quartile Regression Statistics: {target_col} ===")
    if not np.isnan(slope_ci[0]):
        print(f"Slope: {slope_mean:.3f} ms/sextile (95% CI: [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}])")
    else:
        print(f"Slope: {slope_mean:.3f} ms/sextile")
    print(f"R² = {r_value**2:.3f}, p = {p_value:.4f}")
    print(f"n = {len(subject_slopes)} subjects")


def main():
    """Main function to create entropy prediction plots."""
    print("=== ENTROPY PREDICTION PLOTTING ===\n")

    # Set plotting style
    sns.set_context("poster")
    sns.set_palette("colorblind")

    # Process each entropy target
    for target_col in ENTROPY_TARGETS:
        print(f"\n=== Processing {target_col} ===")

        # Load prediction data
        predictions_df = load_prediction_data(SUBJECTS, target_col)

        # Create output directory
        output_dir = os.path.join(PLOTS_DIR_NO_SUB, "entropy_decoder", "predictions")
        os.makedirs(output_dir, exist_ok=True)

        # Compute correlations over time
        print("Computing correlations over time...")
        corr_df = compute_correlation_over_time(predictions_df)

        # Plot correlation over time
        print("Plotting correlation over time...")
        grand_avg = plot_correlation_over_time(corr_df, output_dir, target_col)

        # Plot quartile analysis at peak timepoint
        print("Plotting quartile analysis at peak timepoint...")
        plot_quartile_analysis(predictions_df, corr_df, output_dir, target_col)

    print(f"\n=== Analysis complete! ===")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
