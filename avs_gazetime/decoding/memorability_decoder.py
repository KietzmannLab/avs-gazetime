#!/usr/bin/env python3
"""
Optimized memorability decoder with PCA dimensionality reduction and efficiency improvements.
Based on MNE decoding tutorial best practices.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNetCV, RidgeCV, Ridge
from sklearn.preprocessing import RobustScaler
from mne.decoding import Scaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
# MNE imports
import mne
from mne.decoding import (cross_val_multiscore, GeneralizingEstimator, 
                         SlidingEstimator, Vectorizer)

# Import utility functions
import avs_gazetime.utils.load_data as load_data
from avs_gazetime.memorability.mem_tools import get_memorability_scores
from avs_gazetime.config import (
    S_FREQ, SESSIONS, SUBJECT_ID, PLOTS_DIR, MEG_DIR, CH_TYPE
)

# Import parameters from external file
from decoding_params import (
    EVENT_TYPE, DECIM, APPLY_MEDIAN_SCALE, N_FOLDS, N_JOBS,
    MEMORABILITY_TARGETS, CROP_SIZE_PIX, MODEL_TASK,
    LOG_TRANSFORM, CLIP_OUTLIERS, CLIP_STD_THRESHOLD,
    TEMPORAL_GENERALIZATION, TG_DECIM, LOWPASS_FILTER, USE_PCA,
    PCA_VARIANCE_THRESHOLD, L1_RATIOS, ALPHAS, USE_SCENE_GROUPS
)



def clip_outliers(data, std_threshold=3, verbose=True):
    """Clip outliers beyond specified standard deviations."""
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    if verbose:
        print(f"Clipping outliers: mean={mean:.3f}, std={std:.3f}, threshold={std_threshold}, median={median:.3f}")
    lower_bound = mean - std_threshold * std
    upper_bound = mean + std_threshold * std
    return np.clip(data, lower_bound, upper_bound)

def boxcar_filter(dynamics, times, cutoff_hz=50):
    """Apply boxcar filter with cutoff frequency in Hz."""
    from scipy.ndimage import uniform_filter1d
    
    sfreq = 1 / np.mean(np.diff(times))
    window_size = max(1, int(sfreq / (2 * cutoff_hz)))
    
    print(f"Boxcar filter: {cutoff_hz}Hz → {window_size} samples ({window_size/sfreq*1000:.1f}ms)")
    return uniform_filter1d(dynamics.astype(float), size=window_size, axis=1, mode='nearest')

def prepare_memorability_data(meg_data, meta_df, times, target_col="memorability",
                             log_transform=False, clip_outliers_flag=False,
                             clip_std_threshold=3, decim=1):
    """Prepare data for memorability decoding."""
    # Get target values and remove NaNs
    y = meta_df[target_col].values
    groups = meta_df["sceneID"].values
    valid_mask = ~np.isnan(y)

    print(f"Target: {target_col}")
    print(f"Valid epochs: {np.sum(valid_mask)}/{len(y)}")
    print(f"Unique scenes: {len(np.unique(groups[valid_mask]))}")
    print(f"Target stats: mean={np.nanmean(y):.3f}, std={np.nanstd(y):.3f}")

    # Apply valid mask
    X = meg_data[valid_mask]
    y = y[valid_mask]
    groups = groups[valid_mask]
    meta_df_filtered = meta_df[valid_mask].reset_index(drop=True)

    if LOWPASS_FILTER:
        print(f"Applying boxcar filter with high frequency cutoff at {LOWPASS_FILTER} Hz")
        X = boxcar_filter(X, times, cutoff_hz=LOWPASS_FILTER)

    # Apply log transformation if requested
    if log_transform:
        print(f"Applying log transformation to {target_col}")
        y = np.log(y + 1e-10)

    # Clip outliers in MEG data if requested
    if clip_outliers_flag:
        print(f"Clipping MEG outliers beyond {clip_std_threshold} std")
        X_clipped = X.copy()
        if APPLY_MEDIAN_SCALE:
            print("Since median scaling is applied, clipping will be directly on the data.")
            X_clipped = np.clip(X_clipped, -clip_std_threshold, clip_std_threshold)
        else:
            print("Clipping based on standard deviations.")
            X_clipped = clip_outliers(X_clipped, clip_std_threshold, verbose=True)


        clipped_ratio = np.sum(X != X_clipped) / X.size
        print(f"Clipped {clipped_ratio:.4%} of MEG data points")
        X = X_clipped

    # Decimate time dimension
    times_decimated = times[::decim]
    X = X[:, :, ::decim]

    print(f"Final data shape: {X.shape}")
    print(f"Decimated times: {len(times_decimated)} points")

    return X, y, groups, times_decimated, meta_df_filtered

def create_optimized_time_decoder():
    """Create time-decoding pipeline with PCA."""
    steps = [('scaler', RobustScaler(unit_variance=False)),
             ('vectorizer', Vectorizer())
             ]  # Vectorize time series data
        # Vectorize time series data
    #print("Creating time-decoding pipeline with PCA optimization...")
    
    # Add PCA if requested
    if USE_PCA:
        pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=42)
        steps.append(('pca', pca))
        print(f"Added PCA retaining {PCA_VARIANCE_THRESHOLD*100}% variance")
        # how many components?
        print(f"PCA components: {pca.n_components}")
    
    # Use ElasticNetCV
    estimator = RidgeCV(
        #alphas=ALPHAS,
        #l1_ratio=L1_RATIOS,
       # cv=3,
        #random_state=42,
       # max_iter=1000,
        #precompute=False
    )
    
    steps.append(('estimator', estimator))
    
    # Create base pipeline
    base_pipeline = Pipeline(steps)
    
    # Wrap with SlidingEstimator for time decoding
    time_decoder = SlidingEstimator(
        base_pipeline,
        scoring='r2',
        n_jobs=N_JOBS
    )
    
    return time_decoder

def create_optimized_temporal_generalization():
    """Create temporal generalization pipeline with PCA."""
    
    steps = [('scaler', RobustScaler(unit_variance=False)),
             ('vectorizer', Vectorizer())
             ]  # Vectorize time series data
    
    # Add PCA if requested
    if USE_PCA:
        pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, random_state=42)
        steps.append(('pca', pca))
        print(f"Added PCA retaining {PCA_VARIANCE_THRESHOLD*100}% variance for TG")
    
    # Use ElasticNetCV
    estimator = RidgeCV(
        #alphas=ALPHAS,
        #l1_ratio=L1_RATIOS,
        #cv=3,
        #random_state=42,
        #max_iter=1000,
        #precompute=False
    )
    
    steps.append(('estimator', estimator))
    
    # Create base pipeline
    base_pipeline = Pipeline(steps)
    
    # Wrap with GeneralizingEstimator
    time_gen = GeneralizingEstimator(
        base_pipeline,
        scoring='r2',
        n_jobs=N_JOBS
    )
    
    return time_gen

def get_cross_validator(groups=None, n_folds=N_FOLDS):
    """Get appropriate cross-validator."""
    if groups is not None and USE_SCENE_GROUPS:
        return GroupShuffleSplit(
            n_splits=n_folds, 
            random_state=42
        )
    else:
        return StratifiedKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=42
        )

def fit_and_predict_timeseries(X, y, groups, times, target_col):
    """Fit decoder at each timepoint and generate predictions."""
    print(f"Fitting decoders and generating predictions for {target_col}...")
    print(f"Data shape: {X.shape}")

    # Create time decoder
    time_decoder = create_optimized_time_decoder()

    # Fit and predict at each timepoint
    print("Training decoders on full dataset...")
    time_decoder.fit(X, y)

    # Generate predictions
    print("Generating predictions...")
    predictions = time_decoder.predict(X)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")

    return predictions

def run_timepoint_decoder(X, y, groups, times, target_col):
    """Run timepoint decoding with PCA optimization and dummy baseline."""
    from sklearn.dummy import DummyRegressor

    print(f"Running time decoding with PCA and dummy baseline...")
    print(f"Data shape: {X.shape}")
    if USE_SCENE_GROUPS:
        print(f"Using scene-based CV with {len(np.unique(groups))} unique scenes")

    # Create time decoder
    time_decoder = create_optimized_time_decoder()

    # Create dummy baseline decoder
    dummy_decoder = SlidingEstimator(
        DummyRegressor(strategy='mean'),
        scoring='r2',
        n_jobs=N_JOBS
    )

    # Get cross-validator
    cv = get_cross_validator(groups)

    if USE_SCENE_GROUPS and groups is not None:
        print("Using group-based cross-validation for time decoding")
        # Real decoder scores
        scores = cross_val_multiscore(
            time_decoder, X, y,
            cv=cv,
            groups=groups,
            n_jobs=-2
        )
        # Dummy baseline scores
        dummy_scores = cross_val_multiscore(
            dummy_decoder, X, y,
            cv=cv,
            groups=groups,
            n_jobs=-2
        )
    else:
        # Convert continuous y to discrete for StratifiedKFold
        y_discrete = pd.qcut(y, q=5, labels=False, duplicates='drop')
        cv_strat = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

        # Real decoder scores
        scores = cross_val_multiscore(
            time_decoder, X, y,
            cv=cv_strat,
            n_jobs=-2
        )
        # Dummy baseline scores
        dummy_scores = cross_val_multiscore(
            dummy_decoder, X, y,
            cv=cv_strat,
            n_jobs=-2
        )

    # Average across CV folds
    mean_scores = np.mean(scores, axis=0)
    mean_dummy_scores = np.mean(dummy_scores, axis=0)

    print(f"Timepoint decoding completed. Max R² = {np.max(mean_scores):.4f}")
    print(f"Dummy baseline completed. Mean R² = {np.mean(mean_dummy_scores):.4f}")

    return {
        'times': times,
        'scores': mean_scores,
        'dummy_scores': mean_dummy_scores,
        'target_col': target_col,
        'use_scene_groups': USE_SCENE_GROUPS,
        'n_unique_scenes': len(np.unique(groups)),
        'use_pca': USE_PCA
    }

def run_temporal_generalization(X, y, groups, times, target_col):
    """Run temporal generalization with PCA optimization."""
    print(f"Running temporal generalization with PCA...")
    print(f"Data shape: {X.shape}")
    if USE_SCENE_GROUPS:
        print(f"Using scene-based CV with {len(np.unique(groups))} unique scenes")
    
    # Create temporal generalization estimator
    time_gen = create_optimized_temporal_generalization()
    
    # Get cross-validator
    cv = get_cross_validator(groups)
    
    if USE_SCENE_GROUPS and groups is not None:
        # Use group-based cross-validation
        print("Using group-based cross-validation for temporal generalization")
        scores = cross_val_multiscore(
            time_gen, X, y,
            cv=cv,
            groups=groups,
            n_jobs=-2
        )
    else:
        # Convert continuous y to discrete for StratifiedKFold
        y_discrete = pd.qcut(y, q=5, labels=False, duplicates='drop')
        scores = cross_val_multiscore(
            time_gen, X, y,
            cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42),
            n_jobs=-2
        )
    
    # Average across CV folds
    tg_scores = np.mean(scores, axis=0)
    
    print(f"Temporal generalization completed. Max R² = {np.max(tg_scores):.4f}")
    
    return {
        'times': times,
        'tg_scores': tg_scores,
        'target_col': target_col,
        'use_scene_groups': USE_SCENE_GROUPS,
        'n_unique_scenes': len(np.unique(groups)),
        'use_pca': USE_PCA
    }

def plot_decoder_performance(results, save_path=None):
    """Plot timepoint decoder performance."""
    times = results['times']
    scores = results['scores']
    target_col = results['target_col']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot scores
    ax.plot(times, scores, 'b-', linewidth=2, label='R² score')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cross-validated R²')
    
    # Build title
    title = f'Memorability Decoding: {target_col}'
    if results.get('use_pca', False):
        title += f' (PCA: {PCA_VARIANCE_THRESHOLD*100}% variance)'
    if results.get('use_scene_groups', False):
        title += f'\n(Scene-based CV: {results.get("n_unique_scenes", 0)} scenes)'
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Find and annotate peak
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) > 0:
        peak_idx = np.nanargmax(scores)
        peak_time = times[peak_idx]
        peak_score = scores[peak_idx]
        ax.annotate(f'Peak: {peak_score:.4f} at {peak_time:.3f}s',
                    xy=(peak_time, peak_score), xytext=(10, 10),
                    textcoords='offset points', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_temporal_generalization(tg_results, save_path=None):
    """Plot temporal generalization matrix."""
    times = tg_results['times']
    tg_scores = tg_results['tg_scores']
    target_col = tg_results['target_col']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create time matrix in milliseconds
    times_ms = times * 1000
    
    # Plot temporal generalization matrix
    im = ax.imshow(tg_scores, origin='lower', aspect='auto',
                   extent=[times_ms[0], times_ms[-1], times_ms[0], times_ms[-1]],
                   cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cross-validated R²')
    
    # Add diagonal line
    ax.plot([times_ms[0], times_ms[-1]], [times_ms[0], times_ms[-1]], 
            'k--', alpha=0.5, linewidth=1)
    
    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Testing time (ms)')
    ax.set_ylabel('Training time (ms)')
    
    # Build title
    title = f'Temporal Generalization: {target_col}'
    if tg_results.get('use_pca', False):
        title += f' (PCA: {PCA_VARIANCE_THRESHOLD*100}% variance)'
    if tg_results.get('use_scene_groups', False):
        title += f'\n(Scene-based CV: {tg_results.get("n_unique_scenes", 0)} scenes)'
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Add reference lines
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Testing time (ms)')
    ax.set_ylabel('Training time (ms)')
    
    # Build title with optimization info
    title = f'Temporal Generalization: {target_col}'
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    """Main function to run memorability decoding analysis with PCA optimization."""
    print("=== MEMORABILITY DECODER WITH PCA ===")
    print(f"PCA enabled: {USE_PCA} (retaining {PCA_VARIANCE_THRESHOLD*100}% variance)")
    print()

    print("Loading MEG data and metadata...")

    # Load metadata and MEG data
    merged_df = load_data.merge_meta_df(EVENT_TYPE)
    meg_data = load_data.process_meg_data_for_roi(
        CH_TYPE, EVENT_TYPE, SESSIONS, apply_median_scale=APPLY_MEDIAN_SCALE, scale_with_std=True, sensors_to_femto=False)
    times = load_data.read_hd5_timepoints(event_type=EVENT_TYPE)

    print(f"Loaded data: {meg_data.shape} epochs, {len(times)} timepoints")
    print(f"Metadata: {len(merged_df)} events")
    print(f"Channel type: {CH_TYPE}")

    # Remove outlier durations
    longest_dur = np.percentile(merged_df["duration"], 98)
    shortest_dur = np.percentile(merged_df["duration"], 2)
    dur_mask = (merged_df["duration"] < longest_dur) & (merged_df["duration"] > shortest_dur)

    merged_df = merged_df[dur_mask]
    meg_data = meg_data[dur_mask, :, :]
    merged_df.reset_index(drop=True, inplace=True)

    print(f"After duration filtering: {meg_data.shape[0]} epochs")

    # Reject extreme epochs based on maximum amplitude (99th percentile)
    print("\nRejecting extreme epochs...")
    max_per_epoch = np.max(np.abs(meg_data), axis=(1, 2))
    threshold = np.percentile(max_per_epoch, 99)
    good_epochs = max_per_epoch < threshold

    print(f"Rejecting {(~good_epochs).sum()} / {len(good_epochs)} epochs (max amplitude > {threshold:.3f})")

    merged_df = merged_df[good_epochs].reset_index(drop=True)
    meg_data = meg_data[good_epochs]

    # Reject epochs with low correlation to median ERF (1st percentile)
    print("\nRejecting epochs with low correlation to median ERF...")
    median_erf = np.median(meg_data, axis=0)
    correlations = np.array([np.corrcoef(epoch.flatten(), median_erf.flatten())[0, 1] for epoch in meg_data])
    correlation_threshold = np.percentile(correlations, 1)
    good_epochs = correlations > correlation_threshold

    print(f"Rejecting {(~good_epochs).sum()} / {len(good_epochs)} epochs (correlation < {correlation_threshold:.3f})")

    merged_df = merged_df[good_epochs].reset_index(drop=True)
    meg_data = meg_data[good_epochs]

    print(f"After outlier rejection: {meg_data.shape[0]} epochs")
    
    # Add memorability scores
    print("Loading memorability scores...")
    merged_df = get_memorability_scores(
        merged_df, SUBJECT_ID, MEMORABILITY_TARGETS, MODEL_TASK, 
        crop_size_pix=CROP_SIZE_PIX
    )
    
    # Remove epochs without memorability scores
    valid_mem_mask = ~merged_df["memorability"].isna()
    merged_df = merged_df[valid_mem_mask]
    meg_data = meg_data[valid_mem_mask]
    merged_df.reset_index(drop=True, inplace=True)
    
    print(f"After memorability filtering: {meg_data.shape[0]} epochs")
    
    # Create output directory
    output_dir = os.path.join(PLOTS_DIR, "memorability_decoder")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run decoding for each memorability target
    for target_col in MEMORABILITY_TARGETS:
        print(f"\n=== Decoding {target_col} ===")

        # Prepare data
        X, y, groups, times_dec, meta_df_filtered = prepare_memorability_data(
            meg_data, merged_df, times, target_col=target_col,
            log_transform=LOG_TRANSFORM, clip_outliers_flag=CLIP_OUTLIERS,
            clip_std_threshold=CLIP_STD_THRESHOLD, decim=DECIM
        )


        # Run timepoint decoding
        print("Running timepoint decoding...")
        tp_results = run_timepoint_decoder(X, y, groups, times_dec, target_col)

        # Plot timepoint results
        tp_plot_path = os.path.join(output_dir, f"timepoint_decoding_{target_col}_{SUBJECT_ID}_{CH_TYPE}.png")
        plot_decoder_performance(tp_results, tp_plot_path)

        # Save timepoint results
        tp_save_path = os.path.join(output_dir, f"timepoint_results_{target_col}_{SUBJECT_ID}_{CH_TYPE}.npy")
        np.save(tp_save_path, tp_results)

        # Generate predictions for each timepoint
        print("\nGenerating predictions for regression analysis...")
        predictions = fit_and_predict_timeseries(X, y, groups, times_dec, target_col)

        # Create long format dataframe
        print("Creating long format prediction dataframe...")
        prediction_records = []
        for trial_idx in range(len(y)):
            for time_idx, time_val in enumerate(times_dec):
                prediction_records.append({
                    'subject': meta_df_filtered.iloc[trial_idx]['subject'],
                    'session': meta_df_filtered.iloc[trial_idx]['session'],
                    'trial': meta_df_filtered.iloc[trial_idx]['trial'],
                    'sceneID': meta_df_filtered.iloc[trial_idx]['sceneID'],
                    'duration': meta_df_filtered.iloc[trial_idx]['duration'],
                    'memorability': y[trial_idx],
                    'meg_time': time_val * 1000,  # Convert to ms
                    'predicted_memorability': predictions[trial_idx, time_idx]
                })

        predictions_df = pd.DataFrame(prediction_records)

        # Save predictions
        pred_save_path = os.path.join(output_dir, f"predicted_memorability_{target_col}_as{SUBJECT_ID:02d}_{CH_TYPE}.csv")
        predictions_df.to_csv(pred_save_path, index=False)
        print(f"Saved predictions to {pred_save_path}")
        print(f"Prediction dataframe shape: {predictions_df.shape}")
        
        # Run temporal generalization if requested
        if TEMPORAL_GENERALIZATION:
            print("Running temporal generalization...")
            
            # Additional decimation for TG to speed up computation
            X_tg = X[:, :, ::TG_DECIM]
            times_tg = times_dec[::TG_DECIM]
            
            print(f"TG data shape: {X_tg.shape}")
            
            tg_results = run_temporal_generalization(X_tg, y, groups, times_tg, target_col)
            
            # Plot temporal generalization
            tg_plot_path = os.path.join(output_dir, f"temporal_generalization_{target_col}_{SUBJECT_ID}_{CH_TYPE}.png")
            plot_temporal_generalization(tg_results, tg_plot_path)
            
            # Save temporal generalization results
            tg_save_path = os.path.join(output_dir, f"tg_results_{target_col}_{SUBJECT_ID}.npy")
            np.save(tg_save_path, tg_results)
        # clear memory
        del X, y, groups, times_dec, X_tg, times_tg
    
    print("\nMemorability decoding analysis completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()