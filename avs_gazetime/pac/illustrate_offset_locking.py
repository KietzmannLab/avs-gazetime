#!/usr/bin/env python3
"""
Illustrate offset-locked PAC window selection and duration splitting.

This script creates a visualization showing:
1. Fixation epochs sorted by duration
2. The offset-locked PAC window for each epoch (last N samples before fixation offset + 75ms)
3. The separation between short and long duration groups
4. How the 75ms post-offset extension allows shorter fixations to be included

Key implementation details:
- PAC window: (TIME_WINDOW[1] - TIME_WINDOW[0]) seconds
- Window ends at: fixation_offset + 75ms (post-fixation extension)
- Minimum duration: window_duration - 0.075s
- Maximum duration: epoch recording length (times[-1])

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
post_fixation_extension = 0.075  # 75ms post-fixation extension
sfreq = S_FREQ
ch_type = "parietal"  # Use mag for fast loading

print(f"\nConfiguration:")
print(f"  Subject: {SUBJECT_ID}")
print(f"  Event type: {EVENT_TYPE}")
print(f"  Channel type: {ch_type}")
print(f"  PAC window duration: {window_duration*1000:.0f}ms")
print(f"  Window ends at: fixation offset + {post_fixation_extension*1000:.0f}ms")
print(f"  Duration threshold: {duration_threshold*1000:.0f}ms (split short/long)")
print(f"  Sessions: {SESSIONS}")

# Load MEG data (just first 2 sessions for illustration)
print(f"\n{'='*70}")
print("Loading MEG data...")
print(f"{'='*70}")

meg_data, merged_df, times = load_meg_data(
    event_type=EVENT_TYPE,
    ch_type=ch_type,
    sessions=SESSIONS,  # Just first 2 sessions
    channel_idx=None,
    remove_erfs=remove_erfs,
)
#plot erf (all chanels as fine lines)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
mean_erp = np.mean(meg_data, axis=0)
print(f"mean_erp shape: {mean_erp.shape}")  
for chan_idx in range(mean_erp.shape[0]):
    ax.plot(times, mean_erp[chan_idx, :], color='lightgray', alpha=0.3)
ax.plot(times, np.mean(mean_erp, axis=0), color='blue', linewidth=2, label='Mean ERP')
ax.set_title(f'Subject {SUBJECT_ID} - ERF across channels')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude [fT/cm]')
ax.axvline(x=0, color='black', linestyle='--', label='Event onset')
ax.legend()
fig_filename_erp = os.path.join(f"subject{SUBJECT_ID}_erf_all_channels.png")
plt.tight_layout()
plt.savefig(fig_filename_erp, dpi=300)
print(f"\nERP figure saved to: {fig_filename_erp}")

print(f"\nLoaded data:")
print(f"  Epochs: {len(merged_df)}")
print(f"  Shape: {meg_data.shape}")
print(f"  Times: {len(times)} samples ({times[0]:.3f} to {times[-1]:.3f}s)")

#print the range of the meg data
print(f"  MEG data range: {meg_data.min():.8f} to {meg_data.max():.8f}")
print(f"  MEG data mean: {meg_data.mean():.8f}, std: {meg_data.std():.8f}")

# ============================================================================


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
# PAC window ends at fixation_offset + 75ms
# Minimum duration: window_duration - post_fixation_extension
# This is because the 75ms post-offset extension provides part of the window
min_duration_with_extension = window_duration - post_fixation_extension
too_short_mask = all_durations < min_duration_with_extension
short_mask = (all_durations >= min_duration_with_extension) & (all_durations < duration_threshold) & (all_durations <= max_duration)
long_mask = (all_durations >= duration_threshold) & (all_durations <= max_duration)
too_long_mask = all_durations > max_duration

print(f"\n{'='*70}")
print("Epoch distribution:")
print(f"{'='*70}")
print(f"  PAC window: {window_duration*1000:.0f}ms ending at fixation offset + {post_fixation_extension*1000:.0f}ms")
print(f"  Minimum valid duration: {min_duration_with_extension*1000:.0f}ms")
print(f"    (= {window_duration*1000:.0f}ms window - {post_fixation_extension*1000:.0f}ms extension)")
print(f"")
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
n_epochs_to_show = min(1000, len(sorted_durations))
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
fig, ax1 = plt.subplots(1, 1, figsize=(8,10))

# ============================================================================
# Panel 1: Offset-locked window selection with heatmap-style visualization


# Create a matrix showing the offset-locked windows
n_epochs = len(sorted_durations)
window_samples = int(window_duration * sfreq)

# Create colormap for the windows
cmap_epochs = plt.cm.viridis

for i, duration in enumerate(sorted_durations):
    label = sorted_labels[i]

    # # Full epoch extent (light gray background)
    # ax1.fill_between([times[0], times[-1]], i-0.4, i+0.4,
    #                  color='lightgray', alpha=0.2)

    # Plot fixation duration with color based on group
    # Fixation starts at t=0 and ends at t=duration (clipped to recorded data)
    if label == 0:  # Too short - light gray
        fix_color = 'lightgray'
    elif label == 1:  # Short - green
        fix_color = 'darksalmon'
    elif label == 2:  # Long - orange
        fix_color = 'cornflowerblue'
    else:  # Too long (label == 3) - dark gray
        fix_color = 'darkgray'

    # Clip duration to maximum recordable time
    duration_clipped = min(duration, max_duration)

    ax1.fill_between([0, duration_clipped], i-0.4, i+0.4,
                     color=fix_color, alpha=0.4)

    # Plot PAC window (red highlight) only for valid analysis epochs
    if label in [1, 2]:  # Only short and long groups
        # PAC window extraction logic (matches pac_functions.py lines 422-453)
        # Key: window ends at fixation_offset + post_fixation_extension
        fixation_end_time = duration  # Already filtered to be <= max_duration

        # PAC window ends at fixation_offset + 75ms
        pac_window_end_time = fixation_end_time + post_fixation_extension

        # Find indices in times array
        fixation_end_idx = np.argmin(np.abs(times - fixation_end_time))
        pac_window_end_idx = np.argmin(np.abs(times - pac_window_end_time))

        # Calculate window start: N samples (window_samples) before PAC window end
        # This matches pac_functions.py line 444: start_idx = end_idx - n_samples
        window_start_idx = pac_window_end_idx - window_samples
        window_start_idx = max(0, window_start_idx)  # Ensure non-negative

        # Get times for visualization
        window_start_time = times[window_start_idx]
        window_end_time = times[pac_window_end_idx]
        fixation_end_time_plot = times[fixation_end_idx]

        # Plot main PAC window (within fixation - solid red)
        # From window start to fixation end
        ax1.fill_between([window_start_time, fixation_end_time_plot], i-0.5, i+0.5,
                         color='purple', alpha=0.5, linewidth=0)

        # Plot post-fixation extension (0-75ms after fixation end - hatched red)
        # From fixation end to PAC window end
        if pac_window_end_idx > fixation_end_idx:
            ax1.fill_between([fixation_end_time_plot, window_end_time], i-0.4, i+0.4,
                             color='purple', alpha=0.25, linewidth=0, hatch='\\\\\\')
    # remove y tickslabels
    ax1.set_yticks([])

    # Mark fixation end with vertical line
    ax1.plot([duration_clipped, duration_clipped], [i-0.5, i+0.5],
             color='darkblue', alpha=0.6)

# Add horizontal lines separating groups
ax1.axhline(y=split_idx_1-0.5, color='black' ,linestyle='--', alpha=0.5, zorder=10)
ax1.axhline(y=split_idx_2-0.5, color='black' ,linestyle='--', alpha=0.5, zorder=10)

ax1.set_xlabel('time from fixation onset [s]')
ax1.set_ylabel('fixation epochs\n[sorted by duration]')
ax1.set_title('offset-locked PAC windows', pad=15)
ax1.set_xlim(-.05, 0.450)  # Trim to -100ms to 450ms
ax1.set_ylim(-1, n_epochs)

# Add fixation onset line
ax1.axvline(x=0, color='black', linestyle='-', alpha=0.7, label='fixation onset')

# Add legend for PAC window components
from matplotlib.patches import Patch


# make horizontal lines for (i) max duration, (ii) min duration with extension
ax1.axvline(x=DURATION_SPLIT/1000, color='black', linestyle='--', alpha=0.7, linewidth=2, label='duration split threshold')
#ax1.axvline(x=window_duration - post_fixation_extension, color='brown', linestyle='-.', alpha=0.7, linewidth=2, label='Min duration for PAC')

legend_elements = [
    Patch(facecolor='cornflowerblue', edgecolor='k', alpha=0.4, label='shorter fixation'),
    Patch(facecolor='darksalmon', edgecolor='k', alpha=0.4, label='longer fixation'),
    Patch(facecolor='purple', edgecolor='k', alpha=0.5, label='PAC analysis window'),
    #Patch(facecolor='darkgreen', edgecolor='k', alpha=0.25, hatch='\\\\\\', label='PAC window post-fixation extension'),
    #Patch(facecolor='none', edgecolor='darkblue', label='fixation offset'),
    
]
    
ax1.legend(handles=legend_elements, frameon=False, loc="lower right")


# save figure
fig_filename = os.path.join(PLOTS_DIR, f"illustrate_offset_locked_pac_windows_subject{SUBJECT_ID}.png")
plt.tight_layout()
plt.savefig(fig_filename, dpi=300)
print(f"\nFigure saved to: {fig_filename}")