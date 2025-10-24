#!/usr/bin/env python3
"""
Clean script for ease of recognition analysis using neural network activations.
Analyzes the relationship between classification entropy and fixation duration.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import statsmodels.formula.api as smf
from scipy.special import softmax
from avs_machine_room.dataloader.tools.avs_directory_tools import get_input_dirs
from avs_gazetime.config import PLOTS_DIR_BEHAV

# Configuration
SUBJECTS = [1, 2, 3, 4, 5]
CROP_SIZE_PIX = 100
MODEL_NAME = 'resnet50_ecoset_crop'
ACTIVATION_FEATURE = 'entropy'
MODULE_NAMES = ["fc"]
N_QUARTILES = 6

# Data preprocessing options
Z_SCORE_DURATION_PER_SUB = False
LOG_DURATION = False
LOG_ACTIVATION = True

# Set plotting style
sns.set_context("poster")


def classification_entropy(features, apply_softmax=True):
    """
    Compute classification entropy from network activations.
    
    Parameters:
    -----------
    features : np.ndarray
        Network activations (n_samples, n_features)
    apply_softmax : bool
        Whether to apply softmax normalization
        
    Returns:
    --------
    entropy : np.ndarray
        Classification entropy values
    """
    if apply_softmax:
        features = softmax(features, axis=1)
    
    # Avoid log(0) by adding small epsilon
    features = np.maximum(features, np.finfo(float).eps)
    return -np.sum(features * np.log(features), axis=1)


def load_network_activations(subject, crop_size_pix, model_name, module_name):
    """
    Load network activations for a given subject and module.
    
    Parameters:
    -----------
    subject : int
        Subject ID
    crop_size_pix : int
        Crop size in pixels
    model_name : str
        Name of the neural network model
    module_name : str
        Name of the module/layer
        
    Returns:
    --------
    features : np.ndarray
        Network features
    filenames_df : pd.DataFrame
        DataFrame with filenames and feature mappings
    """
    subject_str = f"as{subject:02d}"
    activations_dir = f"{PLOTS_DIR_BEHAV}/crop_activations/{crop_size_pix}px"
    activations_path = os.path.join(activations_dir, subject_str, model_name, module_name)
    
    # Load features from HDF5 file
    features_file = None
    for file in os.listdir(activations_path):
        if file.endswith('.hdf5'):
            features_file = os.path.join(activations_path, file)
            break
    
    if not features_file:
        raise FileNotFoundError(f"No HDF5 file found in {activations_path}")
    
    with h5py.File(features_file, "r") as f:
        features = f["features"][:]
    
    # Load filenames
    txt_fname = os.path.join(activations_dir, subject_str, model_name, "file_names.txt")
    if not os.path.exists(txt_fname):
        # Look for any text file in the activations directory
        txt_files = [f for f in os.listdir(activations_path) if f.endswith('.txt')]
        if txt_files:
            txt_fname = os.path.join(activations_path, txt_files[0])
        else:
            raise FileNotFoundError(f"No filename mapping found for {subject_str}")
    
    with open(txt_fname, "r") as f:
        filenames_full = [line.strip() for line in f.readlines()]
    
    # Parse filenames to extract metadata
    filenames = [os.path.splitext(os.path.basename(filename))[0].split("_") 
                for filename in filenames_full]
    filenames = [[int(x) for x in filename] for filename in filenames]
    
    # Determine column names based on filename structure
    if len(filenames[0]) == 4:
        colnames = ["subject", "trial", "fix_sequence", "sceneID"]
    elif len(filenames[0]) == 5:
        colnames = ["subject", "trial", "fix_sequence", "start_time", "sceneID"]
    else:
        raise ValueError(f"Unexpected filename structure: {filenames[0]}")
    
    filenames_df = pd.DataFrame(filenames, columns=colnames)
    filenames_df["full_filename"] = filenames_full
    
    return features, filenames_df


def process_features(features, activation_feature):
    """
    Process network features to compute the desired activation measure.
    
    Parameters:
    -----------
    features : np.ndarray
        Raw network features
    activation_feature : str
        Type of feature to compute ('entropy' or 'norm')
        
    Returns:
    --------
    processed_features : np.ndarray
        Processed features
    feature_label : str
        Label for the feature type
    """
    if activation_feature == "norm":
        processed_features = np.linalg.norm(features, axis=1)
        feature_label = "activation norm"
    elif activation_feature == "entropy":
        processed_features = classification_entropy(features)
        feature_label = "classification entropy"
    else:
        raise ValueError("activation_feature must be 'norm' or 'entropy'")
    
    print(f"Mean {activation_feature}: {np.nanmean(processed_features):.3f}")
    print(f"Std {activation_feature}: {np.nanstd(processed_features):.3f}")
    
    return processed_features, feature_label


def preprocess_data(metadata_df, z_score_duration=False, log_duration=False, log_activation=False, module_name="fc"):
    """
    Apply preprocessing steps to the data.
    
    Parameters:
    -----------
    metadata_df : pd.DataFrame
        Combined metadata
    z_score_duration : bool
        Whether to z-score duration per subject
    log_duration : bool
        Whether to log-transform duration
    log_activation : bool
        Whether to log-transform activations
    module_name : str
        Name of the module for activation column
        
    Returns:
    --------
    processed_df : pd.DataFrame
        Preprocessed data
    """
    df = metadata_df.copy()
    
    # Remove last fixations
    len_before = len(df)

    df = df[df["fix_sequence_from_last"] < -1] #Exclude last fixations (and make sure that we only keep fixations generally
    print(f"Excluded {len_before - len(df)} last fixations")
    df = df[df["recording"] == "scene"] #Ensure we only keep scene recordings not the caption recordings
    print(f"Excluded {len_before - len(df)} caption recordings")
    
   
    
    # Remove duration outliers per subject (2nd-98th percentile)
    df = df.groupby("subject").apply(
        lambda x: x[(x["duration"] > x["duration"].quantile(0.02)) & 
                   (x["duration"] < x["duration"].quantile(0.98))]
    ).reset_index(drop=True)
    
    # Remove activation outliers per subject
    activation_col = f"ease_{module_name}"
    df = df.groupby("subject").apply(
        lambda x: x[(x[activation_col] > x[activation_col].quantile(0.02)) & 
                   (x[activation_col] < x[activation_col].quantile(0.98))]
    ).reset_index(drop=True)
    
    if df.empty:
        raise ValueError("DataFrame is empty after outlier removal")
    
    # Apply transformations
    if log_duration:
        df["duration"] = np.log(df["duration"])
    
    if z_score_duration:
        df["duration"] = df.groupby("subject")["duration"].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    
    if log_activation:
        df[activation_col] = np.log(df[activation_col])
        
    if not log_duration and not z_score_duration:
        # Convert duration to milliseconds
        df["duration"] = df["duration"] * 1000
    
    return df


def create_relative_activations(df, module_name):
    """
    Create scene-relative activation measures.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with activations (please note that activations in this case are not raw network activations but correspond to the choosen derivative, such as entropy etc.)
    module_name : str
        Name of the module
        
    Returns:
    --------
    df : pd.DataFrame
        Data with relative activations added
    """
    activation_col = f"ease_{module_name}"
    relative_col = f"rel_ease_{module_name}"
    
    # Store original values
    df["ease_untouched"] = df[activation_col].copy()
    
    # Create scene-relative z-scored activations
    df[relative_col] = df.groupby(["subject", "sceneID"])["ease_untouched"].transform(
        lambda x: (x - np.nanmean(x)) / np.nanstd(x)
    )
    
    df[activation_col] = df.groupby("subject")["ease_untouched"].transform(
        lambda x: (x - np.nanmean(x)) / np.nanstd(x)
    )
    
    return df


def plot_quartile_analysis(df, module_name, activation_feature, output_dir, fname_suffix):
    """
    Create quartile-based analysis plots.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data for plotting
    module_name : str
        Name of the module
    activation_feature : str
        Type of activation feature
    output_dir : str
        Output directory
    fname_suffix : str
        Filename suffix
    """
    # Convert to long format for analysis
    value_vars = [f"ease_{module_name}", f"rel_ease_{module_name}"]
    id_vars = ["subject", "duration", "type", "fix_sequence", 
               "fix_sequence_from_last", "sceneID", "start_time"]
    
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="ease_type",
        value_name="ease_value"
    )
    
    # Assign quartiles per subject and norm type
    q_labels = np.arange(N_QUARTILES)
    for subject in long_df["subject"].unique():
        for ease_type in long_df["ease_type"].unique():
            mask = (long_df["subject"] == subject) & (long_df["ease_type"] == ease_type)
            data_subset = long_df[mask]
            
            if len(data_subset) > N_QUARTILES: 
                quartiles = pd.qcut(data_subset["ease_value"], N_QUARTILES, labels=q_labels)
                long_df.loc[mask, "feature_quartile"] = quartiles
    
    # Create quartile plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    sns.lineplot(x="feature_quartile", y="duration", data=long_df, hue="ease_type", ci=95,
                    estimator=np.nanmean, palette="viridis", markers=False, dashes=False, size_norm=0, 
                    ax=ax, seed=42, lw=0, legend=False)
    
    sns.pointplot(
        data=long_df,
        x="feature_quartile", 
        y="duration",
        hue="ease_type",
        ci=95,
        estimator=np.nanmean,
        palette="viridis",
        ax=ax,
        dodge=False,
        join=False
    )
    
    

    # Customize plot
    if activation_feature == "norm":
        ax.set_xlabel(f"{module_name} activation norm [quantile bins]")
    elif activation_feature == "entropy":
        ax.set_xlabel("classification entropy [quantile bins]")
    
    ax.set_ylabel("mean fixation duration [ms]")
    ax.set_xticklabels(q_labels + 1)
    ax.legend(frameon=False)
    
    sns.despine()
    plt.tight_layout()
    
    # Save plot
    fname = os.path.join(output_dir, f"{fname_suffix}_{activation_feature}_{module_name}_quartile_plot.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create 2D histogram
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ease_data = long_df[long_df["ease_type"] == f"ease_{module_name}"]
    sns.histplot(
        data=ease_data,
        x="ease_value",
        y="duration",
        bins=50,
        cbar=True,
        ax=ax,
        cmap="magma"
    )
    
    feature_label = "classification entropy" if activation_feature == "entropy" else "activation norm"
    ax.set_xlabel(feature_label)
    ax.set_ylabel("fixation duration [ms]")
    ax.set_title(f"{feature_label} vs. fixation duration")
    
    fname = os.path.join(output_dir, f"{fname_suffix}_{activation_feature}_{module_name}_2d_hist.pdf")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def fit_mixed_effects_models(df, module_name, output_dir, fname_suffix):
    """
    Fit mixed-effects models for ease of recognition analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with entropy measures and duration
    module_name : str
        Name of the module
    output_dir : str
        Directory to save results
    fname_suffix : str
        Filename suffix for saving
        
    Returns:
    --------
    results : dict
        Dictionary with model results
    """
    # Prepare clean data
    required_cols = [f'ease_{module_name}', f'rel_ease_{module_name}', 'duration', 'subject']
    clean_data = df.dropna(subset=required_cols)
    
    print(f"Fitting mixed-effects models with {len(clean_data)} observations")
    print(f"Subjects: {clean_data['subject'].nunique()}")
    
    results = {}
    
    # Model 1: Absolute entropy
    print("\n=== Absolute Entropy Model ===")
    formula1 = f"duration ~ ease_{module_name}"
    try:
        model1 = smf.mixedlm(formula1, clean_data, groups="subject")
        result1 = model1.fit()
        results['absolute'] = result1
        print(result1.summary())
        coef = result1.params[f'ease_{module_name}']
        ci_lower, ci_upper = result1.conf_int().loc[f'ease_{module_name}']
        p_value = result1.pvalues[f'ease_{module_name}']
        
        print(f"Coefficient: {coef:.2f}")
        print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
        
    except Exception as e:
        print(f"Error fitting absolute entropy model: {e}")
        results['absolute'] = None
    
    # Model 2: Relative entropy (scene-normalized)
    print("\n=== Relative Entropy Model ===")
    formula2 = f"duration ~ rel_ease_{module_name}"
    try:
        model2 = smf.mixedlm(formula2, clean_data, groups="subject")
        result2 = model2.fit()
        results['relative'] = result2
        print(result2.summary())
        coef = result2.params[f'rel_ease_{module_name}']
        ci_lower, ci_upper = result2.conf_int().loc[f'rel_ease_{module_name}']
        p_value = result2.pvalues[f'rel_ease_{module_name}']
        
        print(f"Coefficient: {coef:.2f}")
        print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
        
    except Exception as e:
        print(f"Error fitting relative entropy model: {e}")
        results['relative'] = None
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"{fname_suffix}_mixed_effects_results.txt")
    with open(results_file, 'w') as f:
        if results['absolute']:
            f.write("ABSOLUTE ENTROPY MODEL\n")
            f.write("=" * 50 + "\n")
            f.write(str(results['absolute'].summary()))
            f.write("\n\n")
        
        if results['relative']:
            f.write("RELATIVE ENTROPY MODEL\n")
            f.write("=" * 50 + "\n")
            f.write(str(results['relative'].summary()))
    
    # Create summary table
    summary_data = []
    for model_name, model_type in [('absolute', 'Absolute'), ('relative', 'Relative')]:
        if results[model_name]:
            res = results[model_name]
            param_name = f'ease_{module_name}' if model_name == 'absolute' else f'rel_ease_{module_name}'
            
            coef = res.params[param_name]
            ci_lower, ci_upper = res.conf_int().loc[param_name]
            p_value = res.pvalues[param_name]
            
            summary_data.append({
                'Model': f'{model_type} Classification Entropy',
                'Coefficient ': f"{coef:.2f}",
                '95% CI': f"[{ci_lower:.2f}, {ci_upper:.2f}]",
                'p-value': f"{p_value:.4f}",
                'N fixations': len(clean_data),
                'N subjects': clean_data['subject'].nunique()
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n=== Model Summary ===")
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_file = os.path.join(output_dir, f"{fname_suffix}_model_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    return results


def main():
    """Main analysis function."""
    print("=== Ease of Recognition Analysis ===")
    
    # Setup paths
    input_dir = get_input_dirs(server="uos")
    crops_dir = os.path.join(input_dir, "fixation_crops")
    crop_image_subdir_name = f"avs_meg_fixation_crops_scene_{CROP_SIZE_PIX}"
    crops_image_dir = os.path.join(crops_dir, crop_image_subdir_name)
    project_dir = "/share/klab/psulewski/psulewski/active-visual-semantics-MEG/results/fullrun/"
    
    # Create output directory
    plots_dir = os.path.join(PLOTS_DIR_BEHAV, "ease_of_recognition", MODEL_NAME)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate filename suffix
    fname_suffix = f"all_subjects_{CROP_SIZE_PIX}px_crops_{ACTIVATION_FEATURE}"
    if Z_SCORE_DURATION_PER_SUB:
        fname_suffix += "_z_scored"
    if LOG_DURATION:
        fname_suffix += "_log_duration"
    if LOG_ACTIVATION:
        fname_suffix += "_log_activation"
    
    all_subject_data = []
    
    # Process each subject
    for subject in SUBJECTS:
        print(f"\n--- Processing Subject {subject} ---")
        
        # Load metadata
        metadata_file = os.path.join(crops_image_dir, "metadata", f"as{subject:02d}_crops_metadata.csv")
        metadata_df = pd.read_csv(metadata_file, low_memory=False)
        print(f"Loaded {len(metadata_df)} events")
        
        # Process each module
        for module_name in MODULE_NAMES:
            print(f"Processing module: {module_name}")
            
            # Load network activations
            features, filenames_df = load_network_activations(
                subject, CROP_SIZE_PIX, MODEL_NAME, module_name
            )
            
            # Process features
            processed_features, feature_label = process_features(features, ACTIVATION_FEATURE)
            filenames_df[f"ease_{module_name}"] = processed_features
            
            # Merge with metadata
            merge_cols = ["full_filename", f"ease_{module_name}"]
            metadata_df = pd.merge(
                metadata_df,
                filenames_df[merge_cols],
                how="inner",
                left_on="crop_filename",
                right_on="full_filename"
            )
            
            print(f"Missing network features: {metadata_df[f'ease_{module_name}'].isna().sum()}")
        
        
        all_subject_data.append(metadata_df)
    
    # Combine all subjects
    print("\n--- Combining All Subjects ---")
    combined_df = pd.concat(all_subject_data, axis=0, ignore_index=True)
    print(f"Combined data: {len(combined_df)} total events")
    
    # Preprocess data
    processed_df = preprocess_data(
        combined_df,
        z_score_duration=Z_SCORE_DURATION_PER_SUB,
        log_duration=LOG_DURATION,
        log_activation=LOG_ACTIVATION,
        module_name=MODULE_NAMES[0]
    ) 
    
    # Create relative activations
    processed_df = create_relative_activations(processed_df, MODULE_NAMES[0])
    
    print(f"Final dataset: {len(processed_df)} events")
    print(f"Mean duration: {processed_df['duration'].mean():.1f} ms")
    
    # Run analysis
    print("\n--- Creating Plots ---")
    plot_quartile_analysis(
        processed_df, MODULE_NAMES[0], ACTIVATION_FEATURE, plots_dir, fname_suffix
    )
    
    print("\n--- Fitting Mixed-Effects Models ---")
    model_results = fit_mixed_effects_models(
        processed_df, MODULE_NAMES[0], plots_dir, fname_suffix
    )
    
    print(f"\nAnalysis complete! Results saved to: {plots_dir}")


if __name__ == "__main__":
    main()