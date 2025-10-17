#!/usr/bin/env python3
"""
Illustrate offset-locked PAC window selection and duration splitting.

This script creates a visualization showing:
1. Fixation epochs sorted by duration
2. The offset-locked PAC window for each epoch (last N samples before fixation end)
3. The separation between short and long duration groups

Uses actual MEG data loading similar to pac_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Import PAC modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pac_dataloader import load_meg_data, split_epochs_by_duration
from config import SESSIONS, SUBJECT_ID, PLOTS_DIR, S_FREQ
from params_pac import (EVENT_TYPE, TIME_WINDOW, DURATION_SPLIT,
                        DURATION_BALANCE, OFFSET_LOCKED, remove_erfs)

# Set up plotting style
sns.set_context("poster")
sns.set_style("white")

print("="*70)
print("Offset-Locked PAC Window Selection - Illustration")
print("="*70)

# Parameters
duration_threshold = DURATION_SPLIT/1000  # Convert ms to s
time_window = TIME_WINDOW
window_duration = time_window[1] - time_window[0]
sfreq = S_FREQ
ch_type = "mag"  # Use mag for fast loading

print(f"\nConfiguration:")
print(f"  Subject: {SUBJECT_ID}")
print(f"  Event type: {EVENT_TYPE}")
print(f"  Channel type: {ch_type}")
print(f"  PAC window duration: {window_duration*1000:.0f}ms (last N samples before fixation end)")
print(f"  Duration threshold: {duration_threshold*1000:.0f}ms (split short/long)")
print(f"  Sessions: {SESSIONS[:2]}")  # Just first 2 sessions for speed

# Load MEG data (just first 2 sessions for illustration)
print(f"\n{'='*70}")
print("Loading MEG data...")
print(f"{'='*70}")

meg_data, merged_df, times = load_meg_data(
    event_type=EVENT_TYPE,
    ch_type=ch_type,
    sessions=SESSIONS[:2],  # Just first 2 sessions
    channel_idx=None,
    remove_erfs=remove_erfs
)

print(f"\nLoaded data:")
print(f"  Epochs: {len(merged_df)}")
print(f"  Shape: {meg_data.shape}")
print(f"  Times: {len(times)} samples ({times[0]:.3f} to {times[-1]:.3f}s)")

# Get duration column
dur_col = "duration" if EVENT_TYPE == "fixation" else "associated_fixation_duration"
all_durations = merged_df[dur_col].values

max_epoch_length = times[-1] - times[0]
print(f"  Epoch length: {max_epoch_length:.3f}s")

# For visualization, show ALL epochs (including those too short or too long for PAC)
# But mark which ones would be included in the analysis
all_durations = merged_df[dur_col].values
max_duration = times[-1]  # Maximum recordable fixation duration

# Split into groups: too_short, short, long, too_long
# With ±50ms extension allowance, minimum duration is (window_duration - 0.1)
min_duration_with_extension = window_duration - 0.1
too_short_mask = all_durations < min_duration_with_extension
short_mask = (all_durations >= min_duration_with_extension) & (all_durations < duration_threshold) & (all_durations <= max_duration)
long_mask = (all_durations >= duration_threshold) & (all_durations <= max_duration)
too_long_mask = all_durations > max_duration

print(f"\n{'='*70}")
print("Epoch distribution:")
print(f"{'='*70}")
print(f"  PAC window: {window_duration*1000:.0f}ms with ±50ms extension allowance")
print(f"  Too short (< {min_duration_with_extension*1000:.0f}ms): {np.sum(too_short_mask)} epochs - EXCLUDED")
print(f"  Short ({min_duration_with_extension*1000:.0f}-{duration_threshold*1000:.0f}ms): {np.sum(short_mask)} epochs")
print(f"  Long ({duration_threshold*1000:.0f}-{max_duration*1000:.0f}ms): {np.sum(long_mask)} epochs")
print(f"  Too long (> {max_duration*1000:.0f}ms): {np.sum(too_long_mask)} epochs - EXCLUDED")

# For balanced counts (what would be used in analysis)
n_analysis = min(np.sum(short_mask), np.sum(long_mask))
print(f"  Balanced analysis groups: {n_analysis} epochs each")

# Store the FULL dataset masks and counts for histogram
full_too_short_mask = too_short_mask.copy()
full_short_mask = short_mask.copy()
full_long_mask = long_mask.copy()
full_durations = all_durations.copy()
full_n_analysis = n_analysis

# Combine all durations and create labels
all_labels = np.zeros(len(all_durations))
all_labels[short_mask] = 1  # short = 1
all_labels[long_mask] = 2   # long = 2
all_labels[too_long_mask] = 3  # too_long = 3
# too_short remains 0

# Sort by duration for visualization
sorted_indices = np.argsort(all_durations)
sorted_durations = all_durations[sorted_indices]
sorted_labels = all_labels[sorted_indices]

# Find split points
split_idx_1 = np.where(sorted_labels >= 1)[0][0] if np.any(sorted_labels >= 1) else len(sorted_durations)
split_idx_2 = np.where(sorted_labels >= 2)[0][0] if np.any(sorted_labels >= 2) else len(sorted_durations)

# Subsample for visualization if too many epochs
n_epochs_to_show = min(200, len(sorted_durations))
if len(sorted_durations) > n_epochs_to_show:
    subsample_indices = np.linspace(0, len(sorted_durations)-1, n_epochs_to_show, dtype=int)
    sorted_durations = sorted_durations[subsample_indices]
    sorted_labels = sorted_labels[subsample_indices]
    split_idx_1 = np.where(sorted_labels >= 1)[0][0] if np.any(sorted_labels >= 1) else len(sorted_durations)
    split_idx_2 = np.where(sorted_labels >= 2)[0][0] if np.any(sorted_labels >= 2) else len(sorted_durations)
    print(f"\nSubsampled to {n_epochs_to_show} epochs for visualization")

# ============================================================================
# Create visualization
# ============================================================================
print(f"\n{'='*70}")
print("Creating visualization...")
print(f"{'='*70}")
# context poster
sns.set_context("poster")
fig, axes = plt.subplots(1, 2, figsize=(10, 8))

# ============================================================================
# Panel 1: Offset-locked window selection with heatmap-style visualization
# ============================================================================
ax1 = axes[0]

# Create a matrix showing the offset-locked windows
n_epochs = len(sorted_durations)
window_samples = int(window_duration * sfreq)

# Create colormap for the windows
cmap_epochs = plt.cm.viridis

for i, duration in enumerate(sorted_durations):
    label = sorted_labels[i]

    # Full epoch extent (light gray background)
    ax1.fill_between([times[0], times[-1]], i-0.4, i+0.4,
                     color='lightgray', alpha=0.2, linewidth=0)

    # Plot fixation duration with color based on group
    # Fixation starts at t=0 and ends at t=duration (clipped to recorded data)
    if label == 0:  # Too short - light gray
        fix_color = 'lightgray'
    elif label == 1:  # Short - green
        fix_color = 'darkgreen'
    elif label == 2:  # Long - orange
        fix_color = 'darkorange'
    else:  # Too long (label == 3) - dark gray
        fix_color = 'darkgray'

    # Clip duration to maximum recordable time
    duration_clipped = min(duration, max_duration)

    ax1.fill_between([0, duration_clipped], i-0.4, i+0.4,
                     color=fix_color, alpha=0.4, linewidth=0)

    # Plot PAC window (red highlight) only for valid analysis epochs
    if label in [1, 2]:  # Only short and long groups
        # PAC window: last N samples before fixation end (offset-locked)
        # With SYMMETRIC ±50ms extension allowance (can extend beyond fixation boundaries)

        fixation_end_time = duration  # Already filtered to be <= max_duration
        end_idx = np.argmin(np.abs(times - fixation_end_time))
        fixation_start_idx = np.argmin(np.abs(times - 0))
        max_extension_samples = int(0.05 * sfreq)  # 50ms = 25 samples at 500Hz

        # Calculate ideal window: N samples ending at fixation offset
        ideal_start_idx = end_idx - window_samples
        ideal_end_idx = end_idx

        # Check how much extension is needed before and after fixation
        extension_before_samples = 0
        extension_after_samples = 0

        if ideal_start_idx < fixation_start_idx:
            extension_before_samples = fixation_start_idx - ideal_start_idx

        # For very short fixations, we might need to extend AFTER fixation end too
        # This happens when: fixation_duration < window_duration - max_extension
        fixation_duration_samples = end_idx - fixation_start_idx
        if window_samples > fixation_duration_samples + max_extension_samples:
            # Need to extend after fixation end as well
            extension_after_samples = window_samples - fixation_duration_samples - extension_before_samples

        # Determine actual window boundaries with symmetric extension
        actual_start_idx = max(0, ideal_start_idx)
        actual_end_idx = min(len(times) - 1, ideal_end_idx + extension_after_samples)

        # Visualize the window with three regions: pre-extension, main, post-extension
        window_start_time = times[actual_start_idx]
        fixation_onset_time = times[fixation_start_idx]
        fixation_end_time_plot = times[end_idx]
        window_end_time = times[actual_end_idx]

        # Plot pre-fixation extension (if any)
        if extension_before_samples > 0 and actual_start_idx < fixation_start_idx:
            ax1.fill_between([window_start_time, fixation_onset_time], i-0.4, i+0.4,
                             color='red', alpha=0.25, linewidth=0, hatch='///', edgecolor='red')

        # Plot main PAC window (within fixation)
        main_start = fixation_onset_time if extension_before_samples > 0 else window_start_time
        main_end = fixation_end_time_plot if extension_after_samples > 0 else window_end_time
        ax1.fill_between([main_start, main_end], i-0.4, i+0.4,
                         color='red', alpha=0.5, linewidth=0)

        # Plot post-fixation extension (if any)
        if extension_after_samples > 0 and actual_end_idx > end_idx:
            ax1.fill_between([fixation_end_time_plot, window_end_time], i-0.4, i+0.4,
                             color='red', alpha=0.25, linewidth=0, hatch='\\\\\\', edgecolor='red')
    # remove y tickslabels
    ax1.set_yticks([])

    # Mark fixation end with vertical line
    ax1.plot([duration_clipped, duration_clipped], [i-0.4, i+0.4],
             color='darkblue', linewidth=1, alpha=0.6)

# Add horizontal lines separating groups
ax1.axhline(y=split_idx_1-0.5, color='black' ,linestyle='--', alpha=0.5, zorder=10)
ax1.axhline(y=split_idx_2-0.5, color='black' ,linestyle='--', alpha=0.5, zorder=10)

ax1.set_xlabel('time [s]')
ax1.set_ylabel('fixation epochs [sorted by duration]')
ax1.set_title('Offset-Locked PAC window')
ax1.set_xlim(-0.2, 0.5)  # Trim to -200ms to 800ms
ax1.set_ylim(-1, n_epochs)

# Add fixation onset line
ax1.axvline(x=0, color='black', linestyle=':', alpha=0.5)


# ============================================================================
# Panel 2: Duration distributions and window extraction illustration
# ============================================================================
ax2 = axes[1]

# Use FULL dataset for histogram (not subsampled)
# Create bins for histogram
bins = np.linspace(0, full_durations.max()*1.1, 40)

# Plot histograms for all three groups using FULL data
if np.sum(full_too_short_mask) > 0:
    ax2.hist(full_durations[full_too_short_mask]*1000, bins=bins*1000,
            alpha=0.5, color='gray', label=f'Too short (n={np.sum(full_too_short_mask)})',
            edgecolor='black', linewidth=0.5)

if np.sum(full_short_mask) > 0:
    ax2.hist(full_durations[full_short_mask]*1000, bins=bins*1000,
            alpha=0.6, color='darkgreen', label=f'Short (n={np.sum(full_short_mask)})',
            edgecolor='black', linewidth=0.5)

if np.sum(full_long_mask) > 0:
    ax2.hist(full_durations[full_long_mask]*1000, bins=bins*1000,
            alpha=0.6, color='darkorange', label=f'Long (n={np.sum(full_long_mask)})',
            edgecolor='black', linewidth=0.5)

# Add threshold lines
ax2.axvline(x=duration_threshold*1000, color='black', linewidth=2,
           linestyle='--', alpha=0.7, zorder=10)

ax2.axvline(x=window_duration*1000, color='red', linewidth=2,
           linestyle='--', alpha=0.7, zorder=10)

ax2.set_xlabel('Fixation Duration [ms]', fontsize=14, fontweight='bold')
ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
ax2.set_title('Duration Distribution & Splitting', fontsize=16, fontweight='bold', pad=20)
ax2.legend(loc='upper right', frameon=True, fontsize=10, fancybox=False, shadow=False)
ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
ax2.set_facecolor('#f0f0f0')

# Add statistics text box using FULL dataset
stats_text = (
    f"TOO SHORT (excluded):\n"
    f"  n = {np.sum(full_too_short_mask)}\n\n"
    f"SHORT (analysis):\n"
    f"  n = {np.sum(full_short_mask)}\n"
    f"  mean = {full_durations[full_short_mask].mean()*1000:.1f} ms\n\n"
    f"LONG (analysis):\n"
    f"  n = {np.sum(full_long_mask)}\n"
    f"  mean = {full_durations[full_long_mask].mean()*1000:.1f} ms\n\n"
    f"Balanced: n = {full_n_analysis}"
)
ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8),
        family='monospace')

plt.tight_layout()

# Save figure
output_fname = os.path.join(PLOTS_DIR, f'offset_locked_pac_illustration_{SUBJECT_ID}.png')
plt.savefig(output_fname, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved as: {output_fname}")

plt.savefig(output_fname.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure saved as: {output_fname.replace('.png', '.pdf')}")

plt.show()

# Print summary
print("\n" + "="*70)
print("OFFSET-LOCKED PAC WINDOW EXTRACTION - SUMMARY")
print("="*70)
print(f"Epoch recording length: {max_epoch_length:.3f}s ({len(times)} samples @ {sfreq}Hz)")
print(f"PAC window: {window_duration*1000:.0f}ms before fixation end")
print(f"Duration threshold: {duration_threshold*1000:.0f}ms")
print(f"Balanced analysis groups: {n_analysis} epochs each (short & long)")
print("="*70)

print("\nDone!")
