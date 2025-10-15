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
duration_threshold = 0.350  # 350ms for illustration
time_window = TIME_WINDOW
window_duration = time_window[1] - time_window[0]
sfreq = S_FREQ
ch_type = "mag"  # Use mag for fast loading

print(f"\nConfiguration:")
print(f"  Subject: {SUBJECT_ID}")
print(f"  Event type: {EVENT_TYPE}")
print(f"  Channel type: {ch_type}")
print(f"  Time window: {time_window[0]:.3f} - {time_window[1]:.3f}s")
print(f"  Window duration: {window_duration*1000:.0f}ms")
print(f"  Duration threshold: {duration_threshold*1000:.0f}ms")
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

# Apply duration split
print(f"\n{'='*70}")
print("Applying duration split...")
print(f"{'='*70}")

# For offset-locked, we need to filter epochs shorter than window duration
min_duration = window_duration  # All fixations must be at least this long

epoch_splits = split_epochs_by_duration(
    merged_df, meg_data,
    duration_split=duration_threshold * 1000,  # Convert to ms
    balance_epochs=DURATION_BALANCE,
    dur_col=dur_col,
    min_duration=min_duration  # Filter out epochs shorter than PAC window
)

# Extract split groups
short_name, short_meg, short_df = epoch_splits[0]
long_name, long_meg, long_df = epoch_splits[1]

short_durations = short_df[dur_col].values
long_durations = long_df[dur_col].values

# Combine and sort by duration for visualization
all_durations_split = np.concatenate([short_durations, long_durations])
all_labels = np.concatenate([
    np.zeros(len(short_durations)),  # 0 = short
    np.ones(len(long_durations))     # 1 = long
])

sorted_indices = np.argsort(all_durations_split)
sorted_durations = all_durations_split[sorted_indices]
sorted_labels = all_labels[sorted_indices]

# Find split point
split_idx = np.where(sorted_labels == 1)[0][0]

print(f"\nSplit groups:")
print(f"  Short: {len(short_durations)} epochs, mean={short_durations.mean()*1000:.1f}ms")
print(f"  Long: {len(long_durations)} epochs, mean={long_durations.mean()*1000:.1f}ms")
print(f"  Split index: {split_idx}")

# Subsample for visualization if too many epochs
n_epochs_to_show = min(200, len(sorted_durations))
if len(sorted_durations) > n_epochs_to_show:
    subsample_indices = np.linspace(0, len(sorted_durations)-1, n_epochs_to_show, dtype=int)
    sorted_durations = sorted_durations[subsample_indices]
    sorted_labels = sorted_labels[subsample_indices]
    split_idx = np.where(sorted_labels == 1)[0][0]
    print(f"\nSubsampled to {n_epochs_to_show} epochs for visualization")

# ============================================================================
# Create visualization
# ============================================================================
print(f"\n{'='*70}")
print("Creating visualization...")
print(f"{'='*70}")

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

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
    # Full epoch extent (light gray background)
    ax1.fill_between([times[0], times[-1]], i-0.4, i+0.4,
                     color='lightgray', alpha=0.2, linewidth=0)

    # Plot fixation duration (blue gradient based on duration)
    # Fixation starts at t=0 and ends at t=duration
    color_val = duration / sorted_durations.max()
    ax1.fill_between([0, duration], i-0.4, i+0.4,
                     color=cmap_epochs(color_val), alpha=0.4, linewidth=0)

    # PAC window: last N samples before fixation end
    # Find the time index corresponding to fixation end
    fixation_end_time = duration
    end_idx = np.argmin(np.abs(times - fixation_end_time))
    start_idx = max(0, end_idx - window_samples)

    # Get actual time values for the window
    window_start_time = times[start_idx]
    window_end_time = times[end_idx]

    # Plot PAC window (red highlight)
    ax1.fill_between([window_start_time, window_end_time], i-0.4, i+0.4,
                     color='red', alpha=0.7, linewidth=0)

    # Mark fixation end with vertical line
    ax1.plot([duration, duration], [i-0.4, i+0.4],
             color='darkblue', linewidth=1.5, alpha=0.8)

# Add horizontal line separating short and long
ax1.axhline(y=split_idx-0.5, color='white', linewidth=3, linestyle='--', zorder=10)

# Add text labels for groups
mid_short = split_idx / 2
mid_long = split_idx + (n_epochs - split_idx) / 2

ax1.text(times[0] + 0.05, mid_short, 'SHORT\nFIXATIONS',
        fontsize=16, fontweight='bold', ha='left', va='center',
        color='white', bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.8))
ax1.text(times[0] + 0.05, mid_long, 'LONG\nFIXATIONS',
        fontsize=16, fontweight='bold', ha='left', va='center',
        color='white', bbox=dict(boxstyle='round', facecolor='darkorange', alpha=0.8))

ax1.set_xlabel('Time [s]', fontsize=14, fontweight='bold')
ax1.set_ylabel('Fixation Epochs [sorted by duration]', fontsize=14, fontweight='bold')
ax1.set_title('Offset-Locked PAC Window Selection', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlim(times[0] - 0.05, times[-1] + 0.05)
ax1.set_ylim(-1, n_epochs)

# Add fixation onset line
ax1.axvline(x=0, color='white', linewidth=2, linestyle=':', alpha=0.8, label='Fixation onset')

# Add legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='lightgray', alpha=0.2, label=f'Epoch extent ({max_epoch_length:.2f}s)'),
    Patch(facecolor='steelblue', alpha=0.4, label='Fixation duration'),
    Patch(facecolor='red', alpha=0.7, label=f'PAC window ({window_duration*1000:.0f}ms before end)'),
    Line2D([0], [0], color='white', linewidth=3, linestyle='--',
           label=f'Threshold ({duration_threshold*1000:.0f}ms)'),
    Line2D([0], [0], color='white', linewidth=2, linestyle=':',
           label='Fixation onset (t=0)')
]
ax1.legend(handles=legend_elements, loc='upper right', frameon=True,
          fontsize=11, fancybox=True, shadow=True)

ax1.grid(True, axis='x', alpha=0.3, linestyle='--')
ax1.set_facecolor('#f0f0f0')

# ============================================================================
# Panel 2: Duration distributions and window extraction illustration
# ============================================================================
ax2 = axes[1]

# Create bins for histogram
bins = np.linspace(0, max(sorted_durations.max(), duration_threshold*1.5), 40)

# Plot histograms
short_mask = sorted_labels == 0
long_mask = sorted_labels == 1

ax2.hist(sorted_durations[short_mask]*1000, bins=bins*1000,
        alpha=0.6, color='steelblue', label=f'Short (n={np.sum(short_mask)})',
        edgecolor='black', linewidth=1)
ax2.hist(sorted_durations[long_mask]*1000, bins=bins*1000,
        alpha=0.6, color='orange', label=f'Long (n={np.sum(long_mask)})',
        edgecolor='black', linewidth=1)

# Add threshold line
ax2.axvline(x=duration_threshold*1000, color='black', linewidth=3,
           linestyle='--', label=f'Threshold ({duration_threshold*1000:.0f}ms)', zorder=10)

# Add PAC window minimum duration line
ax2.axvline(x=window_duration*1000, color='red', linewidth=2.5,
           linestyle=':', alpha=0.8,
           label=f'Min PAC window ({window_duration*1000:.0f}ms)', zorder=10)

# Add shaded region for PAC window
ax2.axvspan(0, window_duration*1000, color='red', alpha=0.1, zorder=1)
ax2.text(window_duration*500, ax2.get_ylim()[1]*0.95,
        'Shorter than\nPAC window', ha='center', va='top',
        fontsize=10, color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax2.set_xlabel('Fixation Duration [ms]', fontsize=14, fontweight='bold')
ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
ax2.set_title('Duration Distribution & Splitting', fontsize=16, fontweight='bold', pad=20)
ax2.legend(loc='upper right', frameon=True, fontsize=11, fancybox=True, shadow=True)
ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
ax2.set_facecolor('#f0f0f0')

# Add statistics text box
stats_text = (
    f"SHORT FIXATIONS:\n"
    f"  n = {np.sum(short_mask)}\n"
    f"  mean = {sorted_durations[short_mask].mean()*1000:.1f} ms\n"
    f"  std = {sorted_durations[short_mask].std()*1000:.1f} ms\n"
    f"  range = [{sorted_durations[short_mask].min()*1000:.0f}, {sorted_durations[short_mask].max()*1000:.0f}] ms\n\n"
    f"LONG FIXATIONS:\n"
    f"  n = {np.sum(long_mask)}\n"
    f"  mean = {sorted_durations[long_mask].mean()*1000:.1f} ms\n"
    f"  std = {sorted_durations[long_mask].std()*1000:.1f} ms\n"
    f"  range = [{sorted_durations[long_mask].min()*1000:.0f}, {sorted_durations[long_mask].max()*1000:.0f}] ms"
)
ax2.text(0.98, 0.50, stats_text, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=1),
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
print("OFFSET-LOCKED PAC WINDOW EXTRACTION - KEY POINTS")
print("="*70)
print(f"1. All epochs have fixed recording length: {max_epoch_length:.3f}s ({len(times)} samples @ {sfreq}Hz)")
print(f"2. PAC window duration: {window_duration*1000:.0f}ms ({window_samples} samples)")
print(f"3. Window extracted from LAST {window_duration*1000:.0f}ms before each fixation END")
print(f"4. Epochs < {window_duration*1000:.0f}ms are EXCLUDED (cannot provide full PAC window)")
print(f"5. Duration threshold: {duration_threshold*1000:.0f}ms splits remaining epochs into short/long")
print(f"6. Both groups balanced to same N = {min(len(short_durations), len(long_durations))}")
print(f"7. Window 'slides' to different absolute times depending on fixation duration")
print(f"8. This ensures PAC is always measured at the SAME relative time (fixation end)")
print(f"9. Short group: ALL fixations are >= {window_duration*1000:.0f}ms but < {duration_threshold*1000:.0f}ms")
print(f"10. Long group: ALL fixations are >= {duration_threshold*1000:.0f}ms")
print("="*70)

print("\nDone!")
