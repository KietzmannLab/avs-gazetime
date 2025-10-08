"""
Minimalistic analysis script examining how saliency, memorability, and ease-of-recognition
relate to fixation duration, with unique variance analysis.

This script integrates three visual features:
- Saliency (from Deepgaze IIE)
- Memorability (from ResMem)
- Ease-of-Recognition (from ResNet50 classification entropy)

Follows the design principles of the avs-gazetime project.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# AVS-specific imports
try:
    from avs_gazetime.config import PLOTS_DIR_BEHAV, SUBJECT_ID
    from avs_gazetime.memorability.mem_tools import get_memorability_scores
    from avs_machine_room.dataloader.tools.avs_directory_tools import get_input_dirs
except ImportError:
    print("Warning: AVS-specific imports not available. Using fallback paths.")
    PLOTS_DIR_BEHAV = "/share/klab/psulewski/psulewski/active-visual-semantics-MEG/results/fullrun/analysis/gazetime/submission_checks/behav"

# Configuration
SUBJECTS = [1, 2, 3, 4, 5]
CROP_SIZE_PIX = 100
LOG_DURATION = False
Z_SCORE_FEATURES = True
SALIENCY_RADIUS = 30

# Set plotting style
plt.rcParams.update({'font.size': 12})
sns.set_context("poster")


def load_saliency_data():
    """Load saliency values for fixations."""
    try:
        input_dir = get_input_dirs(server="uos")
    except:
        input_dir = "/share/klab/datasets/avs/input"

    saliency_file = os.path.join(
        PLOTS_DIR_BEHAV,
        f"fixation_saliency_values_deepgaze_iie_mit1003_radius_{SALIENCY_RADIUS}px.csv"
    )

    if not os.path.exists(saliency_file):
        print(f"Warning: Saliency file not found: {saliency_file}")
        return None

    saliency_df = pd.read_csv(saliency_file)
    print(f"Loaded {len(saliency_df)} saliency records")
    return saliency_df


def load_memorability_data(subject):
    """Load memorability scores for a subject."""
    mem_file = os.path.join(
        PLOTS_DIR_BEHAV,
        "memorability_scores",
        f"as{str(subject).zfill(2)}_crops_metadata_with_memscore_{CROP_SIZE_PIX}.csv"
    )

    if not os.path.exists(mem_file):
        print(f"Warning: Memorability file not found for subject {subject}: {mem_file}")
        return None

    mem_df = pd.read_csv(mem_file, low_memory=False)
    mem_df = mem_df[mem_df['type'] == 'fixation']
    mem_df = mem_df[mem_df['recording'] == 'scene']
    mem_df = mem_df[mem_df['fix_sequence_from_last'] != -1]
    mem_df = mem_df.dropna(subset=['mem_score'])

    print(f"Loaded {len(mem_df)} memorability records for subject {subject}")
    return mem_df[['subject', 'session', 'sceneID', 'trial', 'start_time', 'duration', 'mem_score']]


def load_eor_data(subject):
    """Load ease-of-recognition (entropy) data for a subject."""
    subject_str = f"as{subject:02d}"
    model_name = 'resnet50_ecoset_crop'
    module_name = "fc"

    activations_dir = f"{PLOTS_DIR_BEHAV}/crop_activations/{CROP_SIZE_PIX}px"
    activations_path = os.path.join(activations_dir, subject_str, model_name, module_name)

    if not os.path.exists(activations_path):
        print(f"Warning: EoR activations not found for subject {subject}: {activations_path}")
        return None

    # Load features from HDF5 file
    import h5py
    from scipy.special import softmax

    features_file = None
    for file in os.listdir(activations_path):
        if file.endswith('.hdf5'):
            features_file = os.path.join(activations_path, file)
            break

    if not features_file:
        print(f"Warning: No HDF5 file found for subject {subject}")
        return None

    with h5py.File(features_file, "r") as f:
        features = f["features"][:]

    # Compute classification entropy
    features_softmax = softmax(features, axis=1)
    features_softmax = np.maximum(features_softmax, np.finfo(float).eps)
    entropy = -np.sum(features_softmax * np.log(features_softmax), axis=1)

    # Load filenames
    txt_fname = os.path.join(activations_dir, subject_str, model_name, "file_names.txt")
    if not os.path.exists(txt_fname):
        print(f"Warning: Filenames file not found for subject {subject}")
        return None

    filenames_df = pd.read_csv(txt_fname, header=None, names=["crop_filename"])
    filenames_df['eor_entropy'] = entropy
    filenames_df['subject'] = subject

    print(f"Loaded {len(filenames_df)} EoR records for subject {subject}")
    return filenames_df


def merge_all_data():
    """Merge saliency, memorability, and ease-of-recognition data."""
    print("=== Loading and merging all data ===")

    # Load saliency data (already includes EoR if available)
    saliency_file = os.path.join(
        PLOTS_DIR_BEHAV,
        f"fixation_saliency_values_deepgaze_iie_mit1003_radius_{SALIENCY_RADIUS}px.csv"
    )

    if not os.path.exists(saliency_file):
        print(f"Error: Saliency file not found: {saliency_file}")
        return None

    saliency_df = pd.read_csv(saliency_file)
    print(f"Loaded {len(saliency_df)} saliency records")

    # Check if ease-of-recognition is already in the file
    has_eor = 'ease_fc' in saliency_df.columns
    if has_eor:
        print("Ease-of-recognition values found in saliency file")
    else:
        print("Warning: Ease-of-recognition values not found in saliency file")

    all_data = []

    for subject in SUBJECTS:
        print(f"\nProcessing subject {subject}...")

        # Load memorability data
        mem_df = load_memorability_data(subject)
        if mem_df is None:
            continue

        # Get saliency data for this subject
        sal_df = saliency_df[saliency_df['subject'] == subject].copy()

        # Merge on common columns
        merge_cols = ['subject', 'session', 'scene_id', 'trial', 'start_time']

        # Rename sceneID to scene_id in memorability data for consistency
        mem_df = mem_df.rename(columns={'sceneID': 'scene_id'})

        # Merge saliency (with EoR) and memorability
        merged = pd.merge(sal_df, mem_df, on=merge_cols, how='inner', suffixes=('_sal', '_mem'))

        print(f"Found {len(merged)} overlapping sal/mem records for subject {subject}")

        if has_eor:
            eor_count = merged['ease_fc'].notna().sum()
            print(f"  {eor_count} records have ease-of-recognition values")

        if len(merged) > 0:
            all_data.append(merged)

    if not all_data:
        print("Error: No overlapping data found across subjects")
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} fixations across {len(SUBJECTS)} subjects")

    if has_eor:
        total_eor = combined_df['ease_fc'].notna().sum()
        print(f"Total with ease-of-recognition: {total_eor}")

    print(f"Columns: {list(combined_df.columns)}")
    return combined_df


def preprocess_data(df):
    """Apply preprocessing to the combined dataset."""
    print("\n=== Preprocessing data ===")

    # Check if EoR is available
    has_eor = 'ease_fc' in df.columns

    # Use memorability duration (more complete dataset)

    # Remove outliers (2nd-98th percentile per subject)
    df = df.groupby('subject').apply(
        lambda x: x[(x['duration'] > x['duration'].quantile(0.02)) &
                   (x['duration'] < x['duration'].quantile(0.98))]
    ).reset_index(drop=True)

    # Remove missing values for required columns
    required_cols = ['saliency_value', 'mem_score', 'duration']
    if has_eor:
        # Only keep rows with valid EoR values
        required_cols.append('ease_fc')

    df = df.dropna(subset=required_cols)

    # Log transform duration if specified
    if LOG_DURATION:
        df['duration'] = np.log(df['duration'])

    # Z-score features if specified
    if Z_SCORE_FEATURES:
        df['saliency_z'] = (df['saliency_value'] - df['saliency_value'].mean()) / df['saliency_value'].std()
        df['mem_score_z'] = (df['mem_score'] - df['mem_score'].mean()) / df['mem_score'].std()
        if has_eor:
            df['eor_z'] = (df['ease_fc'] - df['ease_fc'].mean()) / df['ease_fc'].std()
    else:
        df['saliency_z'] = df['saliency_value']
        df['mem_score_z'] = df['mem_score']
        if has_eor:
            df['eor_z'] = df['ease_fc']

    print(f"Final dataset: {len(df)} fixations")
    print(f"Subjects: {sorted(df['subject'].unique())}")
    if has_eor:
        print(f"Ease-of-recognition available: Yes ({df['eor_z'].notna().sum()} valid values)")
    else:
        print("Ease-of-recognition available: No")

    return df


def compute_correlations(df):
    """Compute pairwise correlations between all measures."""
    print("\n=== Computing correlations ===")

    measures = ['duration', 'saliency_z', 'mem_score_z']
    if 'eor_z' in df.columns:
        measures.append('eor_z')

    correlations = {}

    for i, measure1 in enumerate(measures):
        for j, measure2 in enumerate(measures[i+1:], i+1):
            r, p = pearsonr(df[measure1], df[measure2])
            correlations[f"{measure1}_vs_{measure2}"] = {'r': r, 'p': p}
            print(f"{measure1} vs {measure2}: r = {r:.3f}, p = {p:.3e}")

    return correlations


def fit_individual_models(df):
    """Fit individual regression models for each predictor."""
    print("\n=== Individual regression models ===")

    predictors = ['saliency_z', 'mem_score_z']
    if 'eor_z' in df.columns:
        predictors.append('eor_z')

    individual_results = {}

    for predictor in predictors:
        # Simple linear regression
        X = df[[predictor]].values
        y = df['duration'].values

        reg = LinearRegression().fit(X, y)
        r2 = r2_score(y, reg.predict(X))

        # Mixed-effects model (account for subject random effects)
        try:
            formula = f"duration ~ {predictor}"
            model = smf.mixedlm(formula, df, groups=df["subject"])
            result = model.fit()

            individual_results[predictor] = {
                'r2': r2,
                'coef': reg.coef_[0],
                'intercept': reg.intercept_,
                'mixed_coef': result.params[predictor],
                'mixed_pvalue': result.pvalues[predictor]
            }

            print(f"{predictor}: R² = {r2:.3f}, β = {reg.coef_[0]:.3f}, p = {result.pvalues[predictor]:.3e}")

        except Exception as e:
            print(f"Mixed-effects model failed for {predictor}: {e}")
            individual_results[predictor] = {
                'r2': r2,
                'coef': reg.coef_[0],
                'intercept': reg.intercept_
            }

    return individual_results


def fit_combined_model(df):
    """Fit combined regression model with all predictors."""
    print("\n=== Combined regression model ===")

    has_eor = 'eor_z' in df.columns

    try:
        # Mixed-effects model with all predictors
        if has_eor:
            formula = "duration ~ saliency_z + mem_score_z + eor_z"
            predictor_cols = ['saliency_z', 'mem_score_z', 'eor_z']
        else:
            formula = "duration ~ saliency_z + mem_score_z"
            predictor_cols = ['saliency_z', 'mem_score_z']

        model = smf.mixedlm(formula, df, groups=df["subject"])
        result = model.fit()

        print("Combined model results:")
        print(result.summary())

        # Simple R² for comparison
        X = df[predictor_cols].values
        y = df['duration'].values
        reg = LinearRegression().fit(X, y)
        combined_r2 = r2_score(y, reg.predict(X))

        results_dict = {
            'model': result,
            'r2': combined_r2,
            'saliency_coef': result.params['saliency_z'],
            'mem_coef': result.params['mem_score_z'],
            'saliency_p': result.pvalues['saliency_z'],
            'mem_p': result.pvalues['mem_score_z']
        }

        if has_eor:
            results_dict['eor_coef'] = result.params['eor_z']
            results_dict['eor_p'] = result.pvalues['eor_z']

        return results_dict

    except Exception as e:
        print(f"Combined mixed-effects model failed: {e}")
        return None


def unique_variance_analysis(df, individual_results, combined_results):
    """Perform unique variance analysis."""
    print("\n=== Unique variance analysis ===")

    if combined_results is None:
        print("Cannot perform unique variance analysis without combined model")
        return None

    has_eor = 'eor_z' in df.columns

    # R² values
    r2_saliency = individual_results['saliency_z']['r2']
    r2_memory = individual_results['mem_score_z']['r2']
    r2_combined = combined_results['r2']

    results = {
        'r2_saliency': r2_saliency,
        'r2_memory': r2_memory,
        'r2_combined': r2_combined,
    }

    print(f"Individual R² - Saliency: {r2_saliency:.3f}")
    print(f"Individual R² - Memory: {r2_memory:.3f}")

    if has_eor:
        r2_eor = individual_results['eor_z']['r2']
        results['r2_eor'] = r2_eor
        print(f"Individual R² - EoR: {r2_eor:.3f}")

        # For 3-way analysis, fit pairwise combined models
        try:
            # Saliency + Memory
            X_sm = df[['saliency_z', 'mem_score_z']].values
            y = df['duration'].values
            reg_sm = LinearRegression().fit(X_sm, y)
            r2_sal_mem = r2_score(y, reg_sm.predict(X_sm))

            # Saliency + EoR
            X_se = df[['saliency_z', 'eor_z']].values
            reg_se = LinearRegression().fit(X_se, y)
            r2_sal_eor = r2_score(y, reg_se.predict(X_se))

            # Memory + EoR
            X_me = df[['mem_score_z', 'eor_z']].values
            reg_me = LinearRegression().fit(X_me, y)
            r2_mem_eor = r2_score(y, reg_me.predict(X_me))

            # Unique variance (beyond the other two)
            unique_saliency = r2_combined - r2_mem_eor
            unique_memory = r2_combined - r2_sal_eor
            unique_eor = r2_combined - r2_sal_mem

            results['r2_sal_mem'] = r2_sal_mem
            results['r2_sal_eor'] = r2_sal_eor
            results['r2_mem_eor'] = r2_mem_eor
            results['unique_saliency'] = unique_saliency
            results['unique_memory'] = unique_memory
            results['unique_eor'] = unique_eor

            print(f"Combined R² (all three): {r2_combined:.3f}")
            print(f"Combined R² (Sal + Mem): {r2_sal_mem:.3f}")
            print(f"Combined R² (Sal + EoR): {r2_sal_eor:.3f}")
            print(f"Combined R² (Mem + EoR): {r2_mem_eor:.3f}")
            print(f"\nUnique variance - Saliency: {unique_saliency:.3f}")
            print(f"Unique variance - Memory: {unique_memory:.3f}")
            print(f"Unique variance - EoR: {unique_eor:.3f}")

        except Exception as e:
            print(f"Error computing 3-way unique variance: {e}")
    else:
        # 2-way analysis (original code)
        unique_saliency = r2_combined - r2_memory
        unique_memory = r2_combined - r2_saliency
        shared_variance = r2_saliency + r2_memory - r2_combined

        results['unique_saliency'] = unique_saliency
        results['unique_memory'] = unique_memory
        results['shared_variance'] = shared_variance

        print(f"Combined R²: {r2_combined:.3f}")
        print(f"Unique variance - Saliency: {unique_saliency:.3f}")
        print(f"Unique variance - Memory: {unique_memory:.3f}")
        print(f"Shared variance: {shared_variance:.3f}")

    return results


def create_plots(df, output_dir):
    """Create visualization plots."""
    print("\n=== Creating plots ===")

    has_eor = 'eor_z' in df.columns

    if has_eor:
        # Set up plotting (3x2 grid for 3 features)
        fig, axes = plt.subplots(3, 2, figsize=(14, 16))

        # Correlation matrix
        corr_data = df[['duration', 'saliency_z', 'mem_score_z', 'eor_z']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                    square=True, ax=axes[0,0])
        axes[0,0].set_title('Correlation Matrix')

        # Saliency vs Duration
        sns.scatterplot(data=df, x='saliency_z', y='duration', alpha=0.4, ax=axes[0,1])
        axes[0,1].set_xlabel('Saliency (z-scored)')
        axes[0,1].set_ylabel('Fixation Duration (ms)')
        axes[0,1].set_title('Saliency vs Duration')

        # Memory vs Duration
        sns.scatterplot(data=df, x='mem_score_z', y='duration', alpha=0.4, ax=axes[1,0])
        axes[1,0].set_xlabel('Memory Score (z-scored)')
        axes[1,0].set_ylabel('Fixation Duration (ms)')
        axes[1,0].set_title('Memory vs Duration')

        # EoR vs Duration
        sns.scatterplot(data=df, x='eor_z', y='duration', alpha=0.4, ax=axes[1,1])
        axes[1,1].set_xlabel('Ease-of-Recognition (z-scored)')
        axes[1,1].set_ylabel('Fixation Duration (ms)')
        axes[1,1].set_title('EoR vs Duration')

        # Saliency vs Memory
        sns.scatterplot(data=df, x='saliency_z', y='mem_score_z', alpha=0.4, ax=axes[2,0])
        axes[2,0].set_xlabel('Saliency (z-scored)')
        axes[2,0].set_ylabel('Memory Score (z-scored)')
        axes[2,0].set_title('Saliency vs Memory')

        # Saliency vs EoR
        sns.scatterplot(data=df, x='saliency_z', y='eor_z', alpha=0.4, ax=axes[2,1])
        axes[2,1].set_xlabel('Saliency (z-scored)')
        axes[2,1].set_ylabel('Ease-of-Recognition (z-scored)')
        axes[2,1].set_title('Saliency vs EoR')

    else:
        # Original 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Correlation matrix
        corr_data = df[['duration', 'saliency_z', 'mem_score_z']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                    square=True, ax=axes[0,0])
        axes[0,0].set_title('Correlation Matrix')

        # Saliency vs Duration
        sns.scatterplot(data=df, x='saliency_z', y='duration', alpha=0.6, ax=axes[0,1])
        axes[0,1].set_xlabel('Saliency (z-scored)')
        axes[0,1].set_ylabel('Fixation Duration (ms)')
        axes[0,1].set_title('Saliency vs Duration')

        # Memory vs Duration
        sns.scatterplot(data=df, x='mem_score_z', y='duration', alpha=0.6, ax=axes[1,0])
        axes[1,0].set_xlabel('Memory Score (z-scored)')
        axes[1,0].set_ylabel('Fixation Duration (ms)')
        axes[1,0].set_title('Memory vs Duration')

        # Saliency vs Memory
        sns.scatterplot(data=df, x='saliency_z', y='mem_score_z', alpha=0.6, ax=axes[1,1])
        axes[1,1].set_xlabel('Saliency (z-scored)')
        axes[1,1].set_ylabel('Memory Score (z-scored)')
        axes[1,1].set_title('Saliency vs Memory')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'saliency_mem_eor_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {plot_path}")


def main():
    """Main analysis function."""
    print("=== Saliency, Memorability, and Ease-of-Recognition Analysis ===")

    # Create output directory
    output_dir = os.path.join(PLOTS_DIR_BEHAV, "saliency_mem_eor_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Load and merge all data
    df = merge_all_data()
    if df is None:
        print("Error: Could not load and merge data")
        return

    # Preprocess
    df = preprocess_data(df)

    # Compute correlations
    correlations = compute_correlations(df)

    # Fit individual models
    individual_results = fit_individual_models(df)

    # Fit combined model
    combined_results = fit_combined_model(df)

    # Unique variance analysis
    unique_variance = unique_variance_analysis(df, individual_results, combined_results)

    # Create plots
    create_plots(df, output_dir)

    # Save results
    results = {
        'correlations': correlations,
        'individual_results': individual_results,
        'combined_results': combined_results,
        'unique_variance': unique_variance,
        'n_fixations': len(df),
        'n_subjects': len(df['subject'].unique())
    }

    results_path = os.path.join(output_dir, 'analysis_results.txt')
    with open(results_path, 'w') as f:
        f.write("Saliency, Memorability, and Ease-of-Recognition Analysis Results\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Dataset: {results['n_fixations']} fixations from {results['n_subjects']} subjects\n\n")

        f.write("CORRELATIONS:\n")
        for key, value in correlations.items():
            f.write(f"{key}: r = {value['r']:.3f}, p = {value['p']:.3e}\n")

        f.write("\nINDIVIDUAL MODEL RESULTS:\n")
        for predictor, values in individual_results.items():
            f.write(f"{predictor}: R² = {values['r2']:.3f}, β = {values['coef']:.3f}\n")

        if unique_variance:
            f.write("\nUNIQUE VARIANCE ANALYSIS:\n")
            f.write(f"Individual R² - Saliency: {unique_variance['r2_saliency']:.3f}\n")
            f.write(f"Individual R² - Memory: {unique_variance['r2_memory']:.3f}\n")

            if 'r2_eor' in unique_variance:
                f.write(f"Individual R² - EoR: {unique_variance['r2_eor']:.3f}\n")
                f.write(f"\nPairwise Combined R²:\n")
                f.write(f"  Saliency + Memory: {unique_variance['r2_sal_mem']:.3f}\n")
                f.write(f"  Saliency + EoR: {unique_variance['r2_sal_eor']:.3f}\n")
                f.write(f"  Memory + EoR: {unique_variance['r2_mem_eor']:.3f}\n")
                f.write(f"\nThree-way Combined R²: {unique_variance['r2_combined']:.3f}\n")
                f.write(f"\nUnique variance (beyond other two):\n")
                f.write(f"  Saliency: {unique_variance['unique_saliency']:.3f}\n")
                f.write(f"  Memory: {unique_variance['unique_memory']:.3f}\n")
                f.write(f"  EoR: {unique_variance['unique_eor']:.3f}\n")
            else:
                f.write(f"\nCombined R²: {unique_variance['r2_combined']:.3f}\n")
                f.write(f"Unique variance - Saliency: {unique_variance['unique_saliency']:.3f}\n")
                f.write(f"Unique variance - Memory: {unique_variance['unique_memory']:.3f}\n")
                if 'shared_variance' in unique_variance:
                    f.write(f"Shared variance: {unique_variance['shared_variance']:.3f}\n")

    print(f"Results saved to: {results_path}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()