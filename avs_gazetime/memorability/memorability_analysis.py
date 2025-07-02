"""
This script analyzes the relationship between memorability scores and fixation durations
at the group level across all subjects.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Import configuration and parameters
from avs_gazetime.config import PLOTS_DIR_BEHAV
from avs_gazetime.memorability.memorability_analysis_params import *
from avs_machine_room.dataloader.tools.avs_directory_tools import get_input_dirs

def load_and_prepare_data():
    """Load and prepare data from all subjects."""
    # Set up directories
    input_dir = get_input_dirs(server="uos")
    memscore_dir = os.path.join(PLOTS_DIR_BEHAV, "memorability_scores")
    crops_dir = os.path.join(input_dir, "fixation_crops")
    crop_image_subdir_name = f"avs_meg_fixation_crops_scene_{CROP_SIZE_PIX}"
    crops_image_dir = os.path.join(crops_dir, crop_image_subdir_name)
    
    all_subjects_data = []
    fixation_count = 0
    for subject in SUBJECTS:
        print(f"Loading data for subject {subject}")
        
        # Load data
        metadata_df = pd.read_csv(
            os.path.join(memscore_dir, f"as{str(subject).zfill(2)}_crops_metadata_with_memscore_{CROP_SIZE_PIX}.csv"),
            low_memory=False
        )
     
        # Basic filtering
        metadata_df = metadata_df[metadata_df.type == 'fixation']
        metadata_df = metadata_df[metadata_df.recording == 'scene']
        fixation_count += len(metadata_df)
        
        
        metadata_df = metadata_df.dropna(subset=['crop_filename'])
        metadata_df = metadata_df[metadata_df.fix_sequence_from_last != -1]
   
        
        # Filter extreme durations
        metadata_df = metadata_df[
            metadata_df.duration < np.percentile(metadata_df.duration, DURATION_PERCENTILE)
        ]
        metadata_df = metadata_df[
            metadata_df.duration > np.percentile(metadata_df.duration, 100-DURATION_PERCENTILE)
        ]
        
        
        
        # Compute additional features
        metadata_df['mem_scene'] = metadata_df.groupby('sceneID')['mem_score'].transform(np.nanmean)
        metadata_df['mem_score_relative'] = metadata_df.groupby('sceneID')['mem_score'].transform(
            lambda x: (x - np.nanmean(x)) / np.nanstd(x)
        )
        if Z_SORE_MEMSCORE:
            metadata_df['mem_score'] = (metadata_df['mem_score'] - np.nanmean(metadata_df['mem_score'])) / np.nanstd(metadata_df['mem_score'])
       
        if Z_SCORE_DURATION:
            metadata_df['duration'] = (metadata_df['duration'] - np.nanmean(metadata_df['duration'])) / np.nanstd(metadata_df['duration'])
       
        all_subjects_data.append(metadata_df)
    
    # Combine all subjects
    all_subjects_df = pd.concat(all_subjects_data, ignore_index=True)
    
    # Apply transformations
    if LOG_DURATION:
        all_subjects_df['duration'] = np.log(all_subjects_df['duration'])
    
   
    
    # Create long format dataframe
    all_subjects_df_long = pd.melt(
        all_subjects_df,
        id_vars=['subject', 'sceneID', 'duration', "crop_filename"],
        value_vars=['mem_score', 'mem_score_relative', 'mem_scene'],
        var_name='memscore_type',
        value_name='memscore_value'
    )
    
    # Add quartiles
    add_quartiles(all_subjects_df_long)
    print(f"Total number of fixations across all subjects: {fixation_count}")
    return all_subjects_df, all_subjects_df_long, crops_image_dir


def add_quartiles(all_subjects_df_long):
    """Add duration and memorability quartiles to the data."""
    # Duration quartiles (per subject)
    for subject in SUBJECTS:
        mask = all_subjects_df_long.subject == subject
        all_subjects_df_long.loc[mask, 'quartile'] = pd.qcut(
            all_subjects_df_long[mask].duration, NUM_QUARTILES, labels=Q_LABELS
        )
    
    # Memorability quartiles (per subject and type)
    for subject in SUBJECTS:
        for mem_type in ['mem_score', 'mem_score_relative', 'mem_scene']:
            mask = (all_subjects_df_long.subject == subject) & \
                   (all_subjects_df_long.memscore_type == mem_type)
            all_subjects_df_long.loc[mask, 'quartile_mem'] = pd.qcut(
                all_subjects_df_long[mask].memscore_value,
                NUM_QUARTILES,
                labels=Q_LABELS
            )


def get_y_label():
    """Get appropriate y-axis label based on transformations."""
    if Z_SCORE_DURATION:
        if LOG_DURATION:
            y_label = "z-scored log(fixation duration [ms])"
        else:
            y_label = "z-scored fixation duration [ms]"
    else:
        if LOG_DURATION:
            y_label = "log(fixation duration [ms])"
        else:
            y_label = "fixation duration [ms]"
    return y_label


def plot_histogram_by_subject(all_subjects_df_long, output_dir):
    """Plot histogram of memorability scores by subject."""
    # set context
    sns.set_context("poster")
    g = sns.displot(
        all_subjects_df_long, 
        x="memscore_value", 
        #hue="subject", 
        kind="kde", 
        fill=True, 
        height=7, 
        aspect=1, 
        col="memscore_type", 
        facet_kws=dict(margin_titles=False, sharex=False, sharey=False)
    )
    
    # Customize labels
    g.axes[0,0].set_xlabel("ResMem score")
    g.axes[0,1].set_xlabel("ResMem score\n(z-scored per scene)")
    g.axes[0,2].set_xlabel("ResMem score\n(average per scene)")
    g.set_titles("")
    
    fname = f"all_subjects_{CROP_SIZE_PIX}_log_{LOG_DURATION}_zdur_{Z_SCORE_DURATION}_memscore_hist_hue_subject.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


def plot_memorability_vs_duration(all_subjects_df_long, output_dir):
    """Plot memorability vs duration with subject lines."""
    # Convert duration to milliseconds for plotting (if not z-scored)
    plot_data = all_subjects_df_long.copy()
    if not Z_SCORE_DURATION and not LOG_DURATION:
        plot_data['duration'] = plot_data['duration'] * 1000
    
    y_label = get_y_label()
    
    g = sns.lmplot(
        x="memscore_value", 
        y="duration", 
        data=plot_data, 
        col="memscore_type", 
        x_bins=10, 
        sharex=False, 
        sharey=True, 
        hue="subject",
        line_kws={'alpha':0.2}, 
        scatter_kws={'alpha':0.4}, 
        fit_reg=False, 
        height=7, 
        aspect=1, palette="plasma",
    )
    
    # Add grand average
    for i, mem_type in enumerate(['mem_score', 'mem_score_relative', 'mem_scene']):
        data_subset = plot_data[plot_data.memscore_type == mem_type]
        sns.regplot(
            x="memscore_value", 
            y="duration", 
            data=data_subset, 
            x_bins=10, 
            scatter_kws={'alpha':1, 'color':'black'}, 
            fit_reg=False, 
            ax=g.axes[0,i]
        )
    
    # Customize labels
    g.axes[0,0].set_xlabel("ResMem score")
    g.axes[0,1].set_xlabel("ResMem score\n(z-scored per scene)")
    g.axes[0,2].set_xlabel("ResMem score\n(average per scene)")
    g.axes[0,0].set_ylabel(y_label)
    g.set_titles("")
    
    plt.tight_layout()
    fname = f"all_subjects_{CROP_SIZE_PIX}_log_{LOG_DURATION}_zdur_{Z_SCORE_DURATION}_memscore_vs_fixdur.pdf"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


def plot_2d_histogram(all_subjects_df_long, output_dir):
    """Plot 2D histogram of memorability vs duration."""
    # Convert duration to milliseconds for plotting (if not z-scored)
    plot_data = all_subjects_df_long.copy()
    if not Z_SCORE_DURATION and not LOG_DURATION:
        plot_data['duration'] = plot_data['duration'] * 1000
    
    y_label = get_y_label()
    
    g = sns.FacetGrid(
        plot_data, 
        col="memscore_type", 
        sharex=False, 
        sharey=False, 
        height=8, 
        aspect=1
    )
    
    g.map(sns.histplot, "memscore_value", "duration", bins=50, cmap="magma")
    
    # Set y-axis limits (adjusted for milliseconds)
    if not LOG_DURATION and not Z_SCORE_DURATION:
        g.set(ylim=(0, 700))
    
    # Customize labels
    g.axes[0,0].set_xlabel("ResMem score")
    g.axes[0,1].set_xlabel("ResMem score\n(z-scored per scene)")
    g.axes[0,2].set_xlabel("ResMem score\n(average per scene)")
    g.axes[0,0].set_ylabel(y_label)
    
    # Set titles
    g.axes[0,0].set_title("patch memorability")
    g.axes[0,1].set_title("relative patch memorability")
    g.axes[0,2].set_title("average scene memorability")
    
    plt.tight_layout()
    fname = f"all_subjects_{CROP_SIZE_PIX}_log_{LOG_DURATION}_zdur_{Z_SCORE_DURATION}_memscore_hist.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()


def plot_quartile_analyses(all_subjects_df_long, output_dir):
    """Plot quartile-based analyses."""
    # Convert duration to milliseconds for plotting (if not z-scored)
    plot_data = all_subjects_df_long.copy()
    if not Z_SCORE_DURATION and not LOG_DURATION:
        plot_data['duration'] = plot_data['duration'] * 1000
    
    y_label = get_y_label()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Line plot with error bars
    sns.lineplot(
        x="quartile_mem", 
        y="duration", 
        data=plot_data[plot_data.memscore_type != 'mem_scene'], 
        hue="memscore_type", 
        ci=95,
        estimator=np.nanmean, 
        palette='plasma', 
        markers=False, 
        dashes=False, 
        ax=ax, 
        hue_order=['mem_score', 'mem_score_relative'], 
        legend=False
    )
    
    # Point plot
    sns.pointplot(
        x="quartile_mem", 
        y="duration", 
        data=plot_data[plot_data.memscore_type != 'mem_scene'], 
        hue="memscore_type", 
        ci=95,
        hue_order=['mem_score', 'mem_score_relative'], 
        estimator=np.nanmean, 
        palette='plasma', 
        ax=ax, 
        dodge=False, 
        join=False
    )
    
    ax.set_xlabel("memorability quantile")
    ax.set_xticklabels(Q_LABELS + 1)
    ax.set_ylabel(f"mean {y_label}")
    ax.legend(title="memorability type", frameon=False)
    
    plt.tight_layout()
    sns.despine()
    
    fname = f"all_subjects_{CROP_SIZE_PIX}_log_{LOG_DURATION}_zdur_{Z_SCORE_DURATION}_duration_per_quartile_mem.pdf"
    fig.savefig(os.path.join(output_dir, fname))
    plt.close()


def plot_scatter_with_crops(all_subjects_df_long, crops_image_dir, output_dir):
    """Plot scatter plot with actual crop images as points."""
    if not PLOT_MEMSCORE_STUFF:
        return
    
    # KDE plot first
    fig, ax_kde = plt.subplots(figsize=(6, 4), dpi=300)
     # what is the hex for plasma[0]?
    
    hex = "#9c179eff"
    kdeplot = sns.histplot(
        data=all_subjects_df_long[all_subjects_df_long.memscore_type == 'mem_score'], 
        x="memscore_value", 
        #hue="subject", 
        ax=ax_kde, 
        fill=True, 
        alpha=0.5, 
        palette="plasma", 
        color=hex,
        #common_norm=True, 
        legend=True,
        bins=25,
        kde=True,
    )
   
#    kdeplot.legend_.set_frame_on(False)
    #ax_kde.set_xlim(0.4, 1)
    ax_kde.set_xlabel("ResMem score")
    ax_kde.set_ylabel("count")
    plt.tight_layout()
    sns.despine()
    
    fname = f"all_subjects_{CROP_SIZE_PIX}_log_{LOG_DURATION}_zdur_{Z_SCORE_DURATION}_hist_with_example_crops.pdf"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()
    
    # Scatter plot with crops
    for seed in SEEDS:
        fig, ax_scatter = plt.subplots(figsize=(13, 8), dpi=300)
        
        # Sample observations
        sample = all_subjects_df_long[
            all_subjects_df_long.memscore_type == 'mem_score'
        ].sample(NUM_IMAGES, random_state=seed)
        
        # Convert duration to milliseconds if needed
        if not Z_SCORE_DURATION:
            sample = sample.copy()
            if not LOG_DURATION:
                sample['duration'] = sample['duration'] * 1000
            max_duration = 800
        else:
            max_duration = 3
        
        # Filter samples
        sample = sample[
            (sample.memscore_value > 0.5) & 
            (sample.duration < max_duration)
        ]
        
        # Plot scatter points (invisible - will be replaced by images)
        sns.scatterplot(
            data=sample, 
            x="memscore_value", 
            y="duration", 
            hue="subject", 
            ax=ax_scatter, 
            alpha=0.0,  # Make points invisible
            palette="colorblind", 
            legend=False
        )
        
        # Add crop images
        for i, obs in sample.iterrows():
            try:
                image_path = os.path.join(
                    crops_image_dir, 
                    "crops", 
                    f"as{str(obs.subject).zfill(2)}", 
                    obs.crop_filename
                )
                image = plt.imread(image_path)
                
                # Create OffsetImage
                offset_image = OffsetImage(image, zoom=0.55)
                ab = AnnotationBbox(
                    offset_image, 
                    (obs.memscore_value, obs.duration), 
                    frameon=True, 
                    pad=0.01
                )
                ax_scatter.add_artist(ab)
            except Exception as e:
                print(f"Could not load image for row {i}: {e}")
                continue
        
        # Set labels and limits
        ax_scatter.set_xlabel("ResMem score")
        
        # Use the consistent y-label from get_y_label()
        ax_scatter.set_ylabel(get_y_label())
        
        if Z_SCORE_DURATION:
            ax_scatter.set_ylim(-2.5, 2.5)
        
        ax_scatter.set_xlim(0.55, 0.95)
        
        # Reduce number of ticks
        ax_scatter.locator_params(axis='x', nbins=4)
        ax_scatter.locator_params(axis='y', nbins=4)
        
        plt.tight_layout()
        sns.despine()
        
        fname = f"all_subjects_{CROP_SIZE_PIX}_log_{LOG_DURATION}_zdur_{Z_SCORE_DURATION}_scatter_with_example_crops_seed_{seed}.pdf"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()


def fit_linear_mixed_model(all_subjects_df, output_dir):
    """Fit linear mixed model with memorability predictors."""
    # Clean data
    if not Z_SCORE_DURATION and not LOG_DURATION:
        all_subjects_df['duration'] = all_subjects_df['duration'] * 1000
    
    clean_data = all_subjects_df.dropna(
        subset=['mem_score', 'mem_score_relative', 'mem_scene', 'subject']
    )
    
        # Absolute memorability model
    formula1 = "duration ~ mem_score"
    model1 = smf.mixedlm(formula1, clean_data, groups="subject").fit()
    print(model1.summary())
    # Relative memorability model  
    formula2 = "duration ~ mem_score_relative"
    model2 = smf.mixedlm(formula2, clean_data, groups="subject").fit()
    print(model2.summary())


def main():
    """Main function to run the memorability analysis."""
    # Set output directory
    output_dir = os.path.join(PLOTS_DIR_BEHAV, "memorability_analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set plotting style
    sns.set_context("poster")
    sns.set_palette("colorblind")
    
    print("Starting memorability analysis...")
    
    # Load and prepare data
    print("Loading and preparing data...")
    all_subjects_df, all_subjects_df_long, crops_image_dir = load_and_prepare_data()
    
    # Print data summary
    print(f"Number of fixations for analysis: {len(all_subjects_df)}")
    print("Fixations per subject:")
    print(all_subjects_df[["subject", "mem_score"]].groupby("subject").count())
    
    # Run analyses based on flags
    if PLOT_RELATIONSHIPS:
        print("Creating group analyses...")
        plot_histogram_by_subject(all_subjects_df_long, output_dir)
        plot_memorability_vs_duration(all_subjects_df_long, output_dir)
        plot_2d_histogram(all_subjects_df_long, output_dir)
        plot_quartile_analyses(all_subjects_df_long, output_dir)
        
        # Fit linear mixed model
        print("Fitting linear mixed model...")
        fit_linear_mixed_model(all_subjects_df, output_dir)
        mem_slope_results = compute_memorability_quartile_slopes(all_subjects_df_long, output_dir)
    # Plot memorability examples with crops
    if PLOT_MEMSCORE_STUFF:
        print("Creating memorability visualizations...")
        plot_scatter_with_crops(all_subjects_df_long, crops_image_dir, output_dir)
    
    print(f"Analysis complete! Results saved to: {output_dir}")


def compute_memorability_quartile_slopes(all_subjects_df_long, output_dir):
    """
    Compute regression slopes for memorability quartile bin averages.
    """
    from scipy import stats
    import numpy as np
    
    # Filter for relevant memorability types
    mem_types = ['mem_score', 'mem_score_relative']
    results = {}
    
    for mem_type in mem_types:
        print(f"\n=== Analyzing {mem_type} ===")
        
        # Filter data for this memorability type
        type_data = all_subjects_df_long[
            all_subjects_df_long["memscore_type"] == mem_type
        ].copy().dropna(subset=['quartile_mem', 'duration'])
        
        # Compute quartile means per subject
        quartile_means = type_data.groupby(["subject", "quartile_mem"])["duration"].mean().reset_index()
        
        # Arrays for regression
        X_all = []  # quartile bins
        y_all = []  # duration means
        subjects_all = []
        
        for subject in quartile_means["subject"].unique():
            subj_data = quartile_means[quartile_means["subject"] == subject]
            # Only include if we have data for all quartiles
            if len(subj_data) == NUM_QUARTILES:
                X_all.extend(subj_data["quartile_mem"].values)
                y_all.extend(subj_data["duration"].values)
                subjects_all.extend([subject] * len(subj_data))
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        subjects_all = np.array(subjects_all)
        
        if len(X_all) == 0:
            print(f"No complete data for {mem_type}")
            continue
        
        # Overall regression
        slope_overall, intercept, r_value, p_value, std_err = stats.linregress(X_all, y_all)
        
        # Per-subject slopes for confidence interval
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
            slope_sem = np.nan
        
        # Convert to ms if needed
        if not Z_SCORE_DURATION and not LOG_DURATION:
            slope_mean *= 1000
            slope_ci = (slope_ci[0] * 1000, slope_ci[1] * 1000)
            units = "ms/sextile"
        elif LOG_DURATION and not Z_SCORE_DURATION:
            units = "log(ms)/sextile"
        elif Z_SCORE_DURATION:
            units = "z-score/sextile"
        else:
            units = "units/sextile"
        
        # Store results
        results[mem_type] = {
            'slope_mean': slope_mean,
            'slope_ci_lower': slope_ci[0],
            'slope_ci_upper': slope_ci[1],
            'r_squared': r_value**2,
            'p_value': p_value,
            'n_subjects': len(subject_slopes),
            'units': units
        }
        
        # Print results
        if not np.isnan(slope_ci[0]):
            print(f"Slope: {slope_mean:.3f} {units} (95% CI: [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}])")
        else:
            print(f"Slope: {slope_mean:.3f} {units}")
        print(f"R² = {r_value**2:.3f}, p = {p_value:.4f}")
        print(f"n = {len(subject_slopes)} subjects")
    
    # Create summary table
    summary_data = []
    for mem_type, res in results.items():
        mem_name = "absolute memorability" if mem_type == "mem_score" else "relative memorability"
        summary_data.append({
            'Condition': mem_name,
            f'Slope ({res["units"]})': f"{res['slope_mean']:.3f}",
            '95% CI': f"[{res['slope_ci_lower']:.3f}, {res['slope_ci_upper']:.3f}]",
            'R²': f"{res['r_squared']:.3f}",
            'p-value': f"{res['p_value']:.4f}",
            'n': res['n_subjects']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n=== Memorability Slope Summary ===")
    print(summary_df.to_string(index=False))
    
    # Save results
    fname = f"memorability_quartile_regression_log_{LOG_DURATION}_zdur_{Z_SCORE_DURATION}.csv"
    summary_df.to_csv(os.path.join(output_dir, fname), index=False)
    
    # Generate manuscript text
    print("\n=== Manuscript Text ===")
    for mem_type, res in results.items():
        condition = "absolute memorability" if mem_type == "mem_score" else "relative memorability (within-scene z-scored)"
        if not np.isnan(res['slope_ci_lower']):
            print(f"For {condition}, fixation duration increased by {res['slope_mean']:.1f} {res['units']} "
                  f"(95% CI: [{res['slope_ci_lower']:.1f}, {res['slope_ci_upper']:.1f}], "
                  f"R² = {res['r_squared']:.3f}, p = {res['p_value']:.3f}, n = {res['n_subjects']} subjects).")
    
    return results

if __name__ == "__main__":
    main()