#!/usr/bin/env python3
"""
Integrate memorability decoding results across subjects and create plots.
Updated with improved styling and axis handling inspired by the reference script.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from avs_gazetime.dynamics.dynamics_plotting_tools import filter_dynamics
from decoding_params import MEMORABILITY_TARGETS, TG_DECIM
print(f"Using memorability targets: {MEMORABILITY_TARGETS}")

# Set plotting style
sns.set_context("poster")
sns.set_palette("colorblind")

def load_subject_results(subjects=[1,2,3,4,5], ch_type='mag'):
    """Load timepoint and temporal generalization results across subjects."""
    all_tp_results = {}
    all_tg_results = {}
    
    for target in MEMORABILITY_TARGETS:
        all_tp_results[target] = {}
        all_tg_results[target] = {}
        
       
        tp_scores = []
        dummy_scores = []  # Add dummy scores storage
        tg_matrices = []
        times = None
        # set the ch_type as environment variable
        os.environ['CH_TYPE_GAZETIME'] = ch_type
        from avs_gazetime.config import PLOTS_DIR_NO_SUB
        
        for subject in subjects:
            subject_dir = os.path.join(PLOTS_DIR_NO_SUB, f"as{subject:02d}", "memorability_decoder")
            
            # Load timepoint results
            tp_file = os.path.join(subject_dir, f"timepoint_results_{target}_{subject}_{ch_type}.npy")
            print(f"Loading timepoint results from {tp_file}")
            if os.path.exists(tp_file):
                tp_result = np.load(tp_file, allow_pickle=True).item()
                tp_scores.append(tp_result['scores'])
                # Load dummy scores if available
                if 'dummy_scores' in tp_result:
                    dummy_scores.append(tp_result['dummy_scores'])
                if times is None:
                    times = tp_result['times']
            else:
                print(f"Timepoint results not found for {target} in subject {subject}")
            
            # Load temporal generalization results
            tg_file = os.path.join(subject_dir, f"tg_results_{target}_{subject}.npy")
            if os.path.exists(tg_file):
                tg_result = np.load(tg_file, allow_pickle=True).item()
                tg_matrices.append(tg_result['tg_scores'])
            else:
                print(f"Temporal generalization results not found for {target} in subject {subject}")
        
        if tp_scores:
            all_tp_results[target][ch_type] = {
                'scores': np.array(tp_scores),
                'times': times
            }
            # Add dummy scores if available
            if dummy_scores:
                all_tp_results[target][ch_type]['dummy_scores'] = np.array(dummy_scores)
        
        if tg_matrices:
            all_tg_results[target][ch_type] = {
                'tg_scores': np.array(tg_matrices),
                'times': times
            }
    
    print(f"Loaded results for {len(subjects)} subjects and {len(ch_type)} channel types.")
    return all_tp_results, all_tg_results


def plot_timepoint_decoding(tp_results, output_dir, ch_type='mag', lowpass = 40):
    """Create timepoint decoding plot with both targets and baselines."""
    # Prepare data for seaborn
    data_list = []
    
    target_labels = {
        'memorability': 'ResMem score',
        'mem_relative': 'relative ResMem score\n[z-scored per scene]'
    }
    

    
    for target in MEMORABILITY_TARGETS:
        data = tp_results[target][ch_type]
        times = data['times']
        scores = data['scores']  # Shape: (n_subjects, n_times)
        dummy_scores = data.get('dummy_scores', None)
        
      
            
        # filter the dec
        
        # Convert decoder scores to long format
        for subj_idx in range(scores.shape[0]):
            scores_sub = scores[subj_idx, :]
        
            if lowpass:
                print(scores_sub.shape, "scores_sub shape")
                scores_sub = filter_dynamics(scores_sub, times, cutoff_hz=lowpass)
            for time_idx, time_val in enumerate(times):
                
                
                data_list.append({
                    'time': time_val,
                    'scores': scores_sub[ time_idx],
                    'target': target_labels.get(target, target),
                    'decoder_type': '',
                    'subject': subj_idx
                })
        
        # Add dummy baseline scores if available
        if dummy_scores is not None:
            for subj_idx in range(dummy_scores.shape[0]):
                for time_idx, time_val in enumerate(times):
                    data_list.append({
                        'time': time_val,
                        'scores': dummy_scores[subj_idx, time_idx],
                        'target': target_labels.get(target, target),
                        'decoder_type': 'baseline',
                        'subject': subj_idx
                    })
   
    df = pd.DataFrame(data_list)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    # scale to ms
    df["time"] = df["time"] * 1000  # Convert to milliseconds    
    # Plot with seaborn lineplot using bootstrap CI
    sns.lineplot(data=df, x='time', y='scores', hue='target', style='decoder_type',
                ci=95, n_boot=1000, ax=ax, palette='plasma', 
                linewidth=3, alpha=0.9)
    
    # Add significance testing and bars for real decoders only
    for i, target in enumerate(MEMORABILITY_TARGETS):
        if target not in tp_results or ch_type not in tp_results[target]:
            continue
            
        data = tp_results[target][ch_type]
        times = data['times'] * 1000
        scores = data['scores']
        baseline =  data.get('dummy_scores', None)
        
        # Bootstrap significance testing agains baseline
        from scipy.stats import ttest_1samp, ttest_rel
        sig_mask = np.zeros(scores.shape[1], dtype=bool)
        for time_idx in range(scores.shape[1]):
            t_stat, p_val = ttest_rel(scores[:, time_idx], baseline[:, time_idx],  alternative='greater') if baseline is not None else ttest_1samp(scores[:, time_idx], 0, alternative='greater')
            if p_val < 0.05:
                sig_mask[time_idx] = True
        
        if np.any(sig_mask):
            y_sig = -0.0035 + (i * 0.001)
            color = sns.color_palette('plasma', len(MEMORABILITY_TARGETS))[i]
            
            # Create significance periods
            sig_periods = []
            start_idx = None
            for idx, is_sig in enumerate(sig_mask):
                if is_sig and start_idx is None:
                    start_idx = idx
                elif not is_sig and start_idx is not None:
                    sig_periods.append((start_idx, idx-1))
                    start_idx = None
            if start_idx is not None:
                sig_periods.append((start_idx, len(sig_mask)-1))
            
            # Plot significance bars
            for start_idx, end_idx in sig_periods:
                ax.plot([times[start_idx], times[end_idx]], [y_sig, y_sig], 
                       color=color, linewidth=5, alpha=0.8)
        
        # Print peak statistics for real decoder
        peak_scores = np.max(scores, axis=1)
        peak_times = times[np.argmax(scores, axis=1)]
        print(f"{target} - Peak R²: {np.mean(peak_scores):.4f}, Peak time: {np.mean(peak_times):.1f} ms")
        
        # Print baseline statistics if available
        dummy_scores = data.get('dummy_scores', None)
        if dummy_scores is not None:
            baseline_mean = np.mean(dummy_scores)
            print(f"{target} - Baseline R²: {baseline_mean:.4f}")
    
    # Styling
    #ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # Add fixation onset label
    ax.text(0, ax.get_ylim()[1] * 0.85, 'fixation\nonset', 
            horizontalalignment='center', verticalalignment='center', rotation=90)
    
    # Set limits and labels
    ax.set_xlim(-200, 500)
    ax.set_xlabel('time [ms]')
    ax.set_ylabel(f'memorability decoding [R²]')
    
    # Add subject count
    #n_subjects = df['subject'].nunique()
    # ax.text(0.95, 0.95, f'n = {n_subjects}', 
    #         horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    
    # Legend
    ax.legend(frameon=False, loc='upper right')
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'memorability_timepoint_decoding_with_baseline_{ch_type}.pdf'),
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, f'memorability_timepoint_decoding_with_baseline_{ch_type}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def get_peak_latency_with_ci(tp_results, confidence=0.95, n_bootstrap=1000):
    """Calculate peak latency with bootstrapped confidence intervals."""
    results = {}
    
    for target in tp_results.keys():
        results[target] = {}
        
        for ch_type, data in tp_results[target].items():
            times = data['times'] * 1000  # Convert to ms
            scores = data['scores']
            
            # Find peak latency for each subject
            peak_indices = np.argmax(scores, axis=1)
            peak_times = times[peak_indices]
            
            # Calculate original mean
            mean_latency = np.mean(peak_times)
            n_subjects = len(peak_times)
            
            # Bootstrap confidence intervals
            bootstrap_means = []
            np.random.seed(42)  # For reproducibility
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_sample = np.random.choice(peak_times, size=n_subjects, replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            bootstrap_means = np.array(bootstrap_means)
            
            # Calculate percentile-based CI
            alpha = 1 - confidence
            ci_lower = np.percentile(bootstrap_means, 100 * alpha/2)
            ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
            
            results[target][ch_type] = {
                'mean_latency_ms': mean_latency,
                'ci_lower_ms': ci_lower,
                'ci_upper_ms': ci_upper,
                'std_ms': np.std(peak_times, ddof=1),
                'n_subjects': n_subjects,
                'bootstrap_distribution': bootstrap_means  # Optional: keep for further analysis
            }
            
            # Print results
            print(f"{target} - {ch_type.upper()}:")
            print(f"  Peak latency: {mean_latency:.1f} ms [95% CI: {ci_lower:.1f}-{ci_upper:.1f}] (bootstrapped)")
            print(f"  N = {n_subjects} subjects, {n_bootstrap} bootstrap samples")
    
    return results

def plot_temporal_generalization(tg_results, output_dir, ch_type='mag'):
    """Create temporal generalization matrices with improved styling."""
    
    # Set seaborn context for better aesthetics (from reference script)
    sns.set_context("poster")
    
    for target in MEMORABILITY_TARGETS:
        
        n_channels = len(tg_results[target])
        
        # Improved figure size - single plot gets square aspect, multiple plots get wider
        if n_channels == 1:
            fig, axes = plt.subplots(1, n_channels, figsize=(8, 6.5))
            axes = [axes]
        else:
            fig, axes = plt.subplots(1, n_channels, figsize=(6*n_channels, 6.5), sharex=True, sharey=True)
        
        # Color limits based on target (from reference script)
        if target == 'mem_relative':
            vmin, vmax = 0, 0.005
        elif target == 'memorability':
            vmin, vmax =0, None
        else:
            vmin, vmax = 0, 0.01
        
        print(f"\n=== {target.upper()} TEMPORAL GENERALIZATION ===")
        
        for j, ch_type_key in enumerate(tg_results[target].keys()):
            ax = axes[j]
            data = tg_results[target][ch_type_key]
            times = data['times'] * 1000  # Convert to ms
            
            # Subsampling based on TG_DECIM
            if TG_DECIM > 1:
                times = times[::TG_DECIM]
                
            tg_scores = data['tg_scores']
            
            # Average across subjects
            mean_tg = np.mean(tg_scores, axis=0)
            
            # Statistical testing for significance mask
            from scipy.stats import ttest_1samp
            mask = np.ones(mean_tg.shape, dtype=bool)
            p_values = np.zeros_like(mean_tg)
            
            for i in range(mean_tg.shape[0]):
                for k in range(mean_tg.shape[1]):
                    scores_pair = tg_scores[:, i, k]
                    t_stat, p_val = ttest_1samp(scores_pair, 0, alternative='greater')
                    p_values[i, k] = p_val
                    if p_val < 0.05:
                        mask[i, k] = False
            
            print(mean_tg.shape, "mean_tg shape")
            
    
            # Plot temporal generalization matrix with improved styling
            im = ax.pcolormesh(times, times, mean_tg, cmap='plasma',
                              vmin=vmin, vmax=vmax, 
                              alpha=np.where(mask, 0.7, 1.0), 
                              shading='nearest')
            
      
            
            # Add reference lines (improved styling from reference)
            ax.axhline(0, ls='--', color='w', alpha=0.5, linewidth=1)
            ax.axvline(0, ls='--', color='w', alpha=0.5, linewidth=1)
            
            # Set limits
            ax.set_xlim(-100, 350)
            ax.set_ylim(-100, 350)
            
            # Improved title styling (from reference script)
            ax.set_title(f"{ch_type_key.upper()} decoding " + r'$R^2$')
            
            # Labels (consistent with reference script)
            ax.set_xlabel('testing time [ms]')
            if j == 0:
                ax.set_ylabel('training time [ms]')
            
            
            # Count significant pixels
            n_sig = np.sum(~mask)
            total_pixels = mask.size
            sig_percent = 100 * n_sig / total_pixels
            print(f"{ch_type_key.upper()}: {sig_percent:.1f}% significant time-time pairs")
        
        # Improved colorbar styling (from reference script)
        if n_channels == 1:
            # For single plot, use reference script approach
            cbar = plt.colorbar(im, ax=ax)
        else:
            # For multiple plots
            cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20, pad=0.02)
        
        # Colorbar formatting (from reference script)
        cbar.ax.locator_params(nbins=6)
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
        
        # Remove suptitle for cleaner look (reference script doesn't use it)
        # plt.suptitle(f'Temporal Generalization: {target}', y=1.02, fontsize=18)
        
        # Set DPI and use tight layout (from reference script)
        fig.set_dpi(300)
        plt.tight_layout()
        
        # Save with improved naming convention
        fname_base = f'temporal_generalization_fixation_{ch_type}_{target}'
        plt.savefig(os.path.join(output_dir, f'{fname_base}.png'), 
                    bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_dir, f'{fname_base}.pdf'),
                    bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Saved: {fname_base}.png and {fname_base}.pdf")



def main():
    """Main integration function."""
    print("Loading memorability decoding results...")
    ch_types = ["mag"]#, "grad"]  #
    for ch_type in ch_types:
        # Load results
        tp_results, tg_results = load_subject_results(ch_type=ch_type)
        
        if not tp_results:
            print("No results found. Check file paths.")
            return
        
        # Create output directory
        os.environ['CH_TYPE_GAZETIME'] = ch_type
        from avs_gazetime.config import PLOTS_DIR_NO_SUB
        output_dir = os.path.join(PLOTS_DIR_NO_SUB, "memorability_decoding")
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating publication-quality plots...")
        print(tp_results)
        # Generate plots
        get_peak_latency_with_ci(tp_results)
        plot_timepoint_decoding(tp_results, output_dir, ch_type= ch_type)
        plot_temporal_generalization(tg_results, output_dir, ch_type= ch_type)
      
        
    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()