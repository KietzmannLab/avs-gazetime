#!/usr/bin/env python3
"""
Illustrate offset-locked PAC window selection and duration splitting.

This script creates a visualization showing:
1. Fixation epochs sorted by duration
2. The offset-locked PAC window for each epoch (last N samples before fixation end)
3. The separation between short and long duration groups
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
sns.set_context("notebook")
sns.set_style("whitegrid")

# Simulation parameters
n_epochs = 100  # Number of example epochs to show
sfreq = 500  # Sampling frequency in Hz
duration_threshold = 0.350  # 350ms threshold
pac_window = (0.150, 0.400)  # PAC time window
window_duration = pac_window[1] - pac_window[0]  # 250ms
max_epoch_length = 1.302  # Maximum epoch length in seconds (651 samples)

# Generate realistic fixation durations (in seconds)
# Simulate a distribution similar to real data
np.random.seed(42)
short_durations = np.random.gamma(2, 0.08, size=n_epochs//2)  # Shorter fixations
long_durations = np.random.gamma(3, 0.15, size=n_epochs//2) + 0.35  # Longer fixations
all_durations = np.concatenate([short_durations, long_durations])

# Clip to reasonable range
all_durations = np.clip(all_durations, 0.15, 0.95)

# Sort by duration
sorted_indices = np.argsort(all_durations)
sorted_durations = all_durations[sorted_indices]

# Determine which epochs are short vs long
is_short = sorted_durations < duration_threshold

# Find the split point
split_idx = np.where(~is_short)[0][0] if np.any(~is_short) else len(sorted_durations)

print(f"Total epochs: {n_epochs}")
print(f"Short fixations (< {duration_threshold*1000:.0f}ms): {np.sum(is_short)}")
print(f"Long fixations (≥ {duration_threshold*1000:.0f}ms): {np.sum(~is_short)}")
print(f"Mean duration (short): {sorted_durations[is_short].mean()*1000:.1f}ms")
print(f"Mean duration (long): {sorted_durations[~is_short].mean()*1000:.1f}ms")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# ============================================================================
# Panel 1: Offset-locked window selection
# ============================================================================
ax1 = axes[0]

# Create a grid showing each epoch
for i, duration in enumerate(sorted_durations):
    # Full epoch extent (always fixed length)
    epoch_samples = int(max_epoch_length * sfreq)

    # Fixation duration in samples
    duration_samples = int(duration * sfreq)

    # PAC window: last N samples before fixation end
    window_samples = int(window_duration * sfreq)
    window_start = max(0, duration_samples - window_samples)
    window_end = duration_samples

    # Convert to time (seconds)
    time_array = np.arange(epoch_samples) / sfreq

    # Plot full epoch extent (gray background)
    ax1.fill_between([0, max_epoch_length], i-0.4, i+0.4,
                     color='lightgray', alpha=0.3, linewidth=0)

    # Plot fixation duration (blue)
    ax1.fill_between([0, duration], i-0.4, i+0.4,
                     color='steelblue', alpha=0.4, linewidth=0)

    # Plot PAC window (red highlight)
    window_start_time = window_start / sfreq
    window_end_time = window_end / sfreq
    ax1.fill_between([window_start_time, window_end_time], i-0.4, i+0.4,
                     color='red', alpha=0.6, linewidth=0,
                     label='PAC window' if i == 0 else '')

    # Mark fixation end with vertical line
    ax1.plot([duration, duration], [i-0.4, i+0.4],
             color='darkblue', linewidth=2, alpha=0.8)

# Add horizontal line separating short and long
ax1.axhline(y=split_idx-0.5, color='black', linewidth=2.5, linestyle='--',
           label=f'Split at {duration_threshold*1000:.0f}ms')

# Add text labels for groups
ax1.text(0.02, split_idx/2, 'SHORT\nFIXATIONS',
        transform=ax1.get_yaxis_transform(), fontsize=14, fontweight='bold',
        ha='left', va='center', color='darkgreen')
ax1.text(0.02, split_idx + (n_epochs-split_idx)/2, 'LONG\nFIXATIONS',
        transform=ax1.get_yaxis_transform(), fontsize=14, fontweight='bold',
        ha='left', va='center', color='darkgreen')

ax1.set_xlabel('Time [s]', fontsize=12)
ax1.set_ylabel('Fixation Epoch [sorted by duration]', fontsize=12)
ax1.set_title('Offset-Locked PAC Window Selection', fontsize=14, fontweight='bold')
ax1.set_xlim(-0.05, max_epoch_length + 0.05)
ax1.set_ylim(-1, n_epochs)

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightgray', alpha=0.3, label='Epoch extent (1.3s)'),
    Patch(facecolor='steelblue', alpha=0.4, label='Fixation duration'),
    Patch(facecolor='red', alpha=0.6, label=f'PAC window ({window_duration*1000:.0f}ms)'),
    plt.Line2D([0], [0], color='black', linewidth=2.5, linestyle='--',
               label=f'Duration threshold ({duration_threshold*1000:.0f}ms)')
]
ax1.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)

# Add grid
ax1.grid(True, axis='x', alpha=0.3)

# ============================================================================
# Panel 2: Duration distributions
# ============================================================================
ax2 = axes[1]

# Plot histograms
bins = np.linspace(0, 1.0, 30)
ax2.hist(sorted_durations[is_short]*1000, bins=bins*1000,
        alpha=0.6, color='steelblue', label=f'Short (n={np.sum(is_short)})',
        edgecolor='black', linewidth=0.5)
ax2.hist(sorted_durations[~is_short]*1000, bins=bins*1000,
        alpha=0.6, color='orange', label=f'Long (n={np.sum(~is_short)})',
        edgecolor='black', linewidth=0.5)

# Add threshold line
ax2.axvline(x=duration_threshold*1000, color='black', linewidth=2.5,
           linestyle='--', label=f'Threshold ({duration_threshold*1000:.0f}ms)')

# Add PAC window minimum duration line
ax2.axvline(x=window_duration*1000, color='red', linewidth=2,
           linestyle=':', alpha=0.7, label=f'Min for PAC window ({window_duration*1000:.0f}ms)')

ax2.set_xlabel('Fixation Duration [ms]', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Duration Distribution and Splitting', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(True, axis='y', alpha=0.3)

# Add statistics text
stats_text = (
    f"Short fixations:\n"
    f"  n = {np.sum(is_short)}\n"
    f"  mean = {sorted_durations[is_short].mean()*1000:.1f}ms\n"
    f"  std = {sorted_durations[is_short].std()*1000:.1f}ms\n\n"
    f"Long fixations:\n"
    f"  n = {np.sum(~is_short)}\n"
    f"  mean = {sorted_durations[~is_short].mean()*1000:.1f}ms\n"
    f"  std = {sorted_durations[~is_short].std()*1000:.1f}ms"
)
ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_fname = 'offset_locked_pac_illustration.png'
plt.savefig(output_fname, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as: {output_fname}")

plt.savefig(output_fname.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
print(f"Figure saved as: {output_fname.replace('.png', '.pdf')}")

plt.show()

print("\n" + "="*70)
print("Offset-Locked PAC Window Extraction - Key Points")
print("="*70)
print(f"1. All epochs have fixed length: {max_epoch_length}s ({int(max_epoch_length*sfreq)} samples)")
print(f"2. PAC window duration: {window_duration*1000:.0f}ms ({int(window_duration*sfreq)} samples)")
print(f"3. Window is extracted from LAST {window_duration*1000:.0f}ms before fixation end")
print(f"4. Fixations < {window_duration*1000:.0f}ms cannot provide full PAC window")
print(f"5. Duration threshold: {duration_threshold*1000:.0f}ms splits into short/long groups")
print(f"6. Both groups balanced to same N after splitting")
print("="*70)
